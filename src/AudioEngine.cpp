#include <immintrin.h> // AVX2

//====================================================
// AVX2 クロスフェード（double）
//====================================================
static inline void crossfadeAVX2(
    double* dstL, double* dstR,
    const double* newL, const double* newR,
    const double* oldL, const double* oldR,
    int numSamples,
    double gStart,
    double gStep)
{
    int i = 0;
    for (; i + 3 < numSamples; i += 4)
    {
        __m256d g = _mm256_set_pd(
            gStart + gStep * (i + 3),
            gStart + gStep * (i + 2),
            gStart + gStep * (i + 1),
            gStart + gStep * (i + 0)
        );
        __m256d gOld = _mm256_sub_pd(_mm256_set1_pd(1.0), g);
        __m256d nL = _mm256_loadu_pd(newL + i);
        __m256d oL = _mm256_loadu_pd(oldL + i);
        __m256d nR = _mm256_loadu_pd(newR + i);
        __m256d oR = _mm256_loadu_pd(oldR + i);
        __m256d outL = _mm256_add_pd(_mm256_mul_pd(nL, g), _mm256_mul_pd(oL, gOld));
        __m256d outR = _mm256_add_pd(_mm256_mul_pd(nR, g), _mm256_mul_pd(oR, gOld));
        _mm256_storeu_pd(dstL + i, outL);
        _mm256_storeu_pd(dstR + i, outR);
    }
    for (; i < numSamples; ++i)
    {
        double g = gStart + gStep * i;
        double gOld = 1.0 - g;
        dstL[i] = newL[i] * g + oldL[i] * gOld;
        dstR[i] = newR[i] * g + oldR[i] * gOld;
    }
}
//============================================================================
// AudioEngine.cpp  ── v0.2 (JUCE 8.0.12対応)
// AudioEngineの実装
//============================================================================

#include <JuceHeader.h>
#include "AudioEngine.h"
#include "InputBitDepthTransform.h"
#include "OutputFilter.h"

extern std::atomic<bool> gShuttingDown;

// fastTanh 高精度 Padé 近似用の定数
//----------------------------------------------------------------------------
namespace TanhApprox {
    // Padé [5/4] 近似係数
    constexpr double NUM_A = 10395.0;
    constexpr double NUM_B = 1260.0;
    constexpr double NUM_C = 21.0;
    constexpr double DEN_A = 10395.0;
    constexpr double DEN_B = 4725.0;
    constexpr double DEN_C = 210.0;

    // 近似有効範囲
    constexpr double CLIP_THRESHOLD = 4.5;
}




static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

//==============================================================================
// 等電力クロスフェード用近似関数（Audio Thread安全・libm不使用）
//==============================================================================
static inline float equalPowerSinFloat(float x) noexcept
{
    x = juce::jlimit(0.0f, 1.0f, x);
    const float t = x * 1.5707963267948966f;  // π/2
    const float t2 = t * t;
    return t * (1.0f + t2 * (-1.0f/6.0f + t2 * (1.0f/120.0f
             + t2 * (-1.0f/5040.0f + t2 * (1.0f/362880.0f)))));
}

static inline double equalPowerSinDouble(double x) noexcept
{
    x = juce::jlimit(0.0, 1.0, x);
    const double t = x * 1.5707963267948966;
    const double t2 = t * t;
    return t * (1.0 + t2 * (-1.0/6.0 + t2 * (1.0/120.0
             + t2 * (-1.0/5040.0 + t2 * (1.0/362880.0)))));
}

//==============================================================================

// =============================================================
// Rebuild request coalescing (Stage 3)
// =============================================================

AudioEngine::AudioEngine() : uiEqEditor(*this)
{
    m_fadeFloatBuffer.setSize(2, 4096);
    m_fadeDoubleBuffer.setSize(2, 4096);
    m_tmpA.setSize(2, 4096);
    m_tmpB.setSize(2, 4096);
    
    noiseShaperLearner = std::make_unique<NoiseShaperLearner>(*this, captureQueue);
    
    startTimer(100);
}

AudioEngine::~AudioEngine()
{
    // 1. タイマーを停止し、メンテナンスの呼び出しを無くす
    stopTimer();

    // 2. シャットダウンフラグを立て、非同期コールバックを無効化
    uiConvolverProcessor.m_isShuttingDown.store(true, std::memory_order_release);


    // 3. その他のクリーンアップ
    shutdownInProgress.store(true);
}

void AudioEngine::prepareToPlay(int samplesPerBlock, double sampleRate)
{
    m_fadeFloatBuffer.setSize(2, samplesPerBlock);
    m_fadeDoubleBuffer.setSize(2, samplesPerBlock);
    m_tmpA.setSize(2, samplesPerBlock);
    m_tmpB.setSize(2, samplesPerBlock);
    
    currentSampleRate.store(sampleRate, std::memory_order_release);
    maxSamplesPerBlock.store(samplesPerBlock, std::memory_order_release);

    uiConvolverProcessor.prepareToPlay(sampleRate, samplesPerBlock);
    uiEqEditor.prepareToPlay(sampleRate, samplesPerBlock);
}
void AudioEngine::requestRebuild(convo::RebuildKind kind) noexcept
{
    if (shutdownInProgress.load(std::memory_order_acquire) ||
        gShuttingDown.load(std::memory_order_acquire))
        return;

    if (kind == convo::RebuildKind::None)
        return;

    if (kind == convo::RebuildKind::IRContent)
    {
        if (!uiConvolverProcessor.isIRFinalized())
            return;

        const int64_t nowTicks = juce::Time::getHighResolutionTicks();
        const int64_t lastTicks = lastIRContentRebuildTicks_.load(std::memory_order_relaxed);
        const int64_t minDelta = juce::Time::getHighResolutionTicksPerSecond() / 5; // 200ms

        if (lastTicks > 0 && (nowTicks - lastTicks) < minDelta)
            return;

        lastIRContentRebuildTicks_.store(nowTicks, std::memory_order_relaxed);
    }

    const uint32_t mask = convo::toMask(kind);
    const uint32_t prev = pendingRebuildMask_.fetch_or(mask, std::memory_order_acq_rel);

    if (prev == 0)
        triggerAsyncUpdate();
}

void AudioEngine::handleAsyncUpdate()
{
    if (shutdownInProgress.load(std::memory_order_acquire) ||
        gShuttingDown.load(std::memory_order_acquire))
        return;

    executeCommit();
    processRebuildRequestsInternal();
}

void AudioEngine::processLearningCommands() noexcept
{
    if (learnerDispatchOverflow.load(std::memory_order_acquire))
    {
        const LearnerDispatchAction last = lastFailedAction.load(std::memory_order_acquire);
        if (enqueueLearnerDispatch(last))
            learnerDispatchOverflow.store(false, std::memory_order_release);
    }

    LearningCommand cmd;
    while (dequeueLearningCommand(cmd))
    {
        switch (cmd.type)
        {
            case LearningCommand::Type::Start:
            {
                requestedLearningMode = cmd.mode;
                requestedLearningResume = cmd.resume;
                requestedLearningGeneration = cmd.irGeneration;

                const auto& activeView = getActiveView();
                const bool dspReady = activeView.current.isValid;
                const bool isAdaptive9th = activeView.current.noiseShaperType == NoiseShaperType::Adaptive9thOrder;

                if (!dspReady || !isAdaptive9th)
                {
                    learningRuntimeState = LearningRuntimeState::WaitingForDSP;
                    break;
                }

                if (learningRuntimeState == LearningRuntimeState::Running)
                {
                    const LearnerDispatchAction stopAction {
                        LearnerDispatchAction::Type::Stop,
                        false,
                        requestedLearningMode
                    };

                    if (!enqueueLearnerDispatch(stopAction))
                    {
                        DBG("[AudioEngine] processLearningCommands: learner stop queue overflow");
                    }
                }
                break;
            }

            case LearningCommand::Type::Stop:
            {
                requestedLearningResume = false;
                requestedLearningGeneration = currentIRGeneration;

                const LearnerDispatchAction stopAction {
                    LearnerDispatchAction::Type::Stop,
                    false,
                    requestedLearningMode
                };

                if (!enqueueLearnerDispatch(stopAction))
                {
                    DBG("[AudioEngine] processLearningCommands: learner stop queue overflow");
                }

                learningRuntimeState = LearningRuntimeState::Idle;
                break;
            }

            case LearningCommand::Type::IRChanged:
            {
                const bool shouldRestart = (learningRuntimeState != LearningRuntimeState::Idle);
                requestedLearningGeneration = cmd.irGeneration;

                const LearnerDispatchAction stopAction {
                    LearnerDispatchAction::Type::Stop,
                    false,
                    requestedLearningMode
                };

                if (!enqueueLearnerDispatch(stopAction))
                {
                    DBG("[AudioEngine] processLearningCommands: learner stop queue overflow");
                }

                if (shouldRestart)
                {
                    requestedLearningResume = false;
                    learningRuntimeState = LearningRuntimeState::WaitingForDSP;
                }
                else
                {
                    learningRuntimeState = LearningRuntimeState::Idle;
                }
                break;
            }

            case LearningCommand::Type::DSPReady:
            {
                currentIRGeneration = cmd.irGeneration;

                // irGeneration チェックを削除: WaitingForDSP 状態であれば遅延なく学習開始
                if (learningRuntimeState == LearningRuntimeState::WaitingForDSP)
                {
                    const LearnerDispatchAction startAction {
                        LearnerDispatchAction::Type::Start,
                        requestedLearningResume,
                        requestedLearningMode
                    };

                    if (enqueueLearnerDispatch(startAction))
                    {
                        learningRuntimeState = LearningRuntimeState::Running;
                    }
                    else
                    {
                        DBG("[AudioEngine] processLearningCommands: DSPReady learner start queue overflow");
                    }
                }
                break;
            }
        }
    }
}

void AudioEngine::processDeferredLearningActions()
{
    LearnerDispatchAction action;
    while (dequeueLearnerDispatch(action))
    {
        if (noiseShaperLearner == nullptr)
            continue;

        if (action.type == LearnerDispatchAction::Type::Stop)
        {
            noiseShaperLearner->stopLearning();
            continue;
        }

        noiseShaperLearner->setLearningMode(action.mode);
        noiseShaperLearner->startLearning(action.resume);
    }
}

void AudioEngine::resetLearningControlState() noexcept
{
    learningCommandWrite = 0;
    learningCommandRead = 0;
    learnerDispatchWrite = 0;
    learnerDispatchRead = 0;
    learnerDispatchOverflow.store(false, std::memory_order_release);
    lastFailedAction.store(LearnerDispatchAction {}, std::memory_order_release);
    learningRuntimeState = LearningRuntimeState::Idle;
    requestedLearningMode = pendingLearningMode.load(std::memory_order_acquire);
    requestedLearningResume = false;
    requestedLearningGeneration = pendingIRGeneration;
    currentIRGeneration = pendingIRGeneration;
}

void AudioEngine::timerCallback()
{
    processRebuildRequestsInternal();

    // フェイルセーフ: current snapshot が欠落した状態を放置すると
    // EQ変更が演算経路へ乗らないため、Message Thread 側で自己修復する。
    if (!shutdownInProgress.load(std::memory_order_acquire)
        && !getActiveView().current.isValid)
    {
        diagLog("[VERIFY] snapshot bootstrap: current was invalid, requesting worker snapshot refresh");
        requestRebuild(convo::RebuildKind::Structural);
    }

    {
        const auto& view = getActiveView();
        const bool dspReady = view.current.isValid;
        
        // 旧診断ログ用変数は削除、または新システムViewから取得するように修正
    }

    if (!shutdownInProgress.load(std::memory_order_acquire)
        && deferredStructuralRebuildPending_.load(std::memory_order_acquire))
    {
        const int64_t dueTicks = deferredStructuralRebuildDueTicks_.load(std::memory_order_acquire);
        const int64_t nowTicks = juce::Time::getHighResolutionTicks();

        if (dueTicks > 0 && nowTicks >= dueTicks)
        {
            deferredStructuralRebuildPending_.store(false, std::memory_order_release);
            deferredStructuralRebuildDueTicks_.store(0, std::memory_order_release);

            if (uiConvolverProcessor.isIRLoaded())
            {
                diagLog("[DIAG] timerCallback: issuing deferred Structural rebuild after prepared IR apply");
                requestRebuild(convo::RebuildKind::Structural);

                ++pendingIRGeneration;
                setIRChangeFlag();

                const LearningCommand cmd {
                    LearningCommand::Type::IRChanged,
                    false,
                    pendingLearningMode.load(std::memory_order_acquire),
                    pendingIRGeneration
                };

                if (!enqueueLearningCommand(cmd))
                {
                    DBG("[AudioEngine] timerCallback: deferred command queue overflow");
                }
            }
        }
    }

    if (!shutdownInProgress.load(std::memory_order_acquire)
        && deferredFinalizeAwareRebuildPending_.load(std::memory_order_acquire))
    {
        const int queuedGeneration = rebuildGeneration.load(std::memory_order_acquire);
        const int committedGeneration = lastCommittedRebuildGeneration.load(std::memory_order_acquire);
        const bool outstandingRebuild = queuedGeneration > committedGeneration;
        const bool irLoaded = uiConvolverProcessor.isIRLoaded();
        const bool irFinalized = uiConvolverProcessor.isIRFinalized();
        const bool irLoading = uiConvolverProcessor.isLoadingIR();
        const bool structuralDeferred = deferredStructuralRebuildPending_.load(std::memory_order_acquire);
        const bool pendingIrChange = m_pendingIRChange.load(std::memory_order_acquire);

        // IR 遷移が完全に落ち着いてから 1 回だけ再構築を発火する。
        if ((!irLoaded || irFinalized)
            && !irLoading
            && !structuralDeferred
            && !pendingIrChange
            && !outstandingRebuild)
        {
            deferredFinalizeAwareRebuildPending_.store(false, std::memory_order_release);

            const double sr = getActiveView().current.sampleRate;
            if (!m_isRestoringState && sr > 0.0)
            {
                diagLog("[DIAG] timerCallback: issuing deferred finalize-aware rebuild");
                requestRebuild(sr, maxSamplesPerBlock.load(std::memory_order_acquire));
            }
        }
    }

    processLearningCommands();
    processDeferredLearningActions();

    if (!shutdownInProgress.load(std::memory_order_acquire) &&

        !dspCrossfadePending.load(std::memory_order_acquire) &&
        fadeQueued.exchange(false, std::memory_order_acq_rel))
    {

        {
            const double fadeSec = queuedNextFadeTimeSec.load(std::memory_order_acquire);
            queuedFadeTimeSec.store(fadeSec, std::memory_order_release);

            dspCrossfadePending.store(true, std::memory_order_release);
            setIRChangeFlag();
        }
    }

    // Grace period に基づく安全なリリース遅延は、
    // RCU v17.15 では ReclaimerThread が自動的に担当するため不要。
    // processDeferredReleases(); // 削除

    const auto& view = getActiveView();
    if (view.previousValid && view.alpha >= 1.0f)
    {
        sendChangeMessage();
    }

    // 内部プロセッサのクリーンアップを実行する。
    uiEqEditor.cleanup();
    uiConvolverProcessor.cleanup();

    // 退役キューのメンテナンスを実行
    uiConvolverProcessor.tickMaintenance();
    // EQState の退役キューも定期的に解放（同一メッセージスレッドで安全）
    uiEqEditor.reclaimRetiredEQStates();
}

void AudioEngine::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    if (source == &uiEqEditor)
    {
        // UI (SpectrumAnalyzerComponent など) が EQ 編集を即時反映できるよう通知する。
        // 実 DSP 反映は従来どおり requestRebuild() 経由で行う。
        sendChangeMessage();
        requestRebuild(convo::RebuildKind::Structural);
    }
}

void AudioEngine::convolverParamsChanged(ConvolverProcessor* processor)
{
    if (processor == &uiConvolverProcessor)
    {
        if (uiConvolverProcessor.isIRLoaded())
        {
            const uint64_t uiStructuralHash = uiConvolverProcessor.getStructuralHash();
            const uint64_t prevHash = lastIssuedConvolverStructuralHash_.load(std::memory_order_acquire);
            
            if (prevHash != uiStructuralHash)
            {
                lastIssuedConvolverStructuralHash_.store(uiStructuralHash, std::memory_order_release);
                diagLog("[DIAG] convolverParamsChanged: issuing Structural rebuild hash=" 
                    + juce::String::toHexString((int64_t)uiStructuralHash));
                requestRebuild(convo::RebuildKind::Structural);
                
                ++pendingIRGeneration;
                setIRChangeFlag();
                
                const LearningCommand cmd {
                    LearningCommand::Type::IRChanged,
                    false,
                    pendingLearningMode.load(std::memory_order_acquire),
                    pendingIRGeneration
                };
                enqueueLearningCommand(cmd);
            }
        }
    }
}

//--------------------------------------------------------------
// releaseResources
// デバイス停止時に呼ばれる（Audio Thread停止後）
// JUCE v8.0.12 完全対応版（MMCSSはJUCEが自動管理）
//--------------------------------------------------------------
void AudioEngine::releaseResources()
{
    diagLog("[DIAG] releaseResources: enter");
    shutdownInProgress.store(true, std::memory_order_release);
    
    if (noiseShaperLearner)
        noiseShaperLearner->stopLearning();

    resetLearningControlState();

    uiConvolverProcessor.releaseResources();
    uiEqEditor.releaseResources();

    diagLog("[DIAG] releaseResources: exit");
}

//--------------------------------------------------------------
// getNextAudioBlock - オーディオ処理コールバック (Audio Thread)
// リアルタイム制約 (Real-time Constraints)
//    1. メモリ割り当て禁止 (No memory allocation): new, malloc, vector::resize, AudioBuffer::setSize 等はNG。
//    2. ロック禁止 (No locks): Mutex, CriticalSection 等によるブロックはNG。
//    3. システムコール禁止 (No system calls): ファイルI/O, コンソール出力(printf) 等はNG。
//    4. 待機禁止 (No waiting): sleep や 重い計算によるストールを避ける。IRの再ロードもNG。
//    5. 禁止API: AudioBlock::allocate, AudioBlock::copyFrom (確保伴うもの), FFT::performFrequencyOnlyForwardTransform (事前確保なしはNG)
//    6. std::vector使用時は、必ず AudioBuffer / 生ポインタを wrap する形で使用すること。
//    7. MMCSS設定禁止: AvSetMmThreadCharacteristics 等の呼び出しは禁止。
//--------------------------------------------------------------
void AudioEngine::processBlockDouble(juce::AudioBuffer<double>& buffer)
{
    juce::AudioSourceChannelInfo info(&buffer, 0, buffer.getNumSamples());
    
    const double sr = currentSampleRate.load(std::memory_order_acquire);
    if (sr <= 0.0)
    {
        buffer.clear();
        return;
    }

    if (shutdownInProgress.load(std::memory_order_acquire))
    {
        buffer.clear();
        return;
    }

    // Double-precision processing logic if needed, 
    // for now we can downcast to float for processors and upcast back, 
    // or implement double-precision path in processors.
    // Given the current architecture, we use float processing internally.
    
    m_fadeFloatBuffer.setSize(buffer.getNumChannels(), buffer.getNumSamples(), false, false, true);
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        const double* src = buffer.getReadPointer(ch);
        float* dst = m_fadeFloatBuffer.getWritePointer(ch);
        for (int s = 0; s < buffer.getNumSamples(); ++s)
            dst[s] = (float)src[s];
    }

    juce::AudioSourceChannelInfo floatInfo(&m_fadeFloatBuffer, 0, buffer.getNumSamples());
    getNextAudioBlock(floatInfo);

    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        const float* src = m_fadeFloatBuffer.getReadPointer(ch);
        double* dst = buffer.getWritePointer(ch);
        for (int s = 0; s < buffer.getNumSamples(); ++s)
            dst[s] = (double)src[s];
    }
}
void AudioEngine::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{

    const juce::ScopedNoDenormals noDenormals;
    m_audioBlockCounter.fetch_add(1, std::memory_order_release);

    // 入力検証
    if (bufferToFill.buffer == nullptr)
        return;

    const int numSamples = bufferToFill.numSamples;
    const int startSample = bufferToFill.startSample;
    auto* buffer = bufferToFill.buffer;

    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20;
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    if (startSample < 0 || startSample + numSamples > buffer->getNumSamples())
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // ========================================================================
    // ダブルバッファモデル：スロット取得（lock-free）
    // ========================================================================
    int idx = m_activeIndex.load(std::memory_order_acquire);
    const convo::EngineView& view = m_views[idx];

    const convo::EngineState& cur  = view.current;
    const convo::EngineState& prev = view.previous;
    const float alpha = view.alpha;

    // ========================================================================
    // フェード処理または通常処理
    // ========================================================================
    if (view.previousValid && alpha < 1.0f)
    {
        // クロスフェード中：prev と cur をブレンド
        m_tmpA.clear();
        m_tmpB.clear();

        processWithState(m_tmpA, prev, 0, numSamples, 1.0f - alpha);
        processWithState(m_tmpB, cur,  0, numSamples, alpha);

        const int numChannels = buffer->getNumChannels();
        for (int ch = 0; ch < numChannels; ++ch)
        {
            buffer->addFrom(ch, startSample, m_tmpA, ch, 0, numSamples);
            buffer->addFrom(ch, startSample, m_tmpB, ch, 0, numSamples);
        }
    }
    else
    {
        // 通常処理：current のみ
        processWithState(*buffer, cur, startSample, numSamples, 1.0f);
    }

    // ========================================================================
    // NoiseShaperLearner 用のキャプチャ処理
    // ========================================================================
    if (learningRuntimeState.load(std::memory_order_acquire) == LearningRuntimeState::Running)
    {
        // 簡易実装: 256サンプルごとにブロックを作成してキューイング
        // 本来はバッファリングが必要だが、ここでは一旦直接的なコピーを試行
        if (numSamples >= 256)
        {
            AudioBlock block;
            block.numSamples = 256;
            block.sampleRateHz = static_cast<int>(cur.sampleRate);
            block.adaptiveCoeffBankIndex = currentAdaptiveCoeffBankIndex.load(std::memory_order_relaxed);
            block.sessionId = cur.captureSessionId;
            
            const float* srcL = buffer->getReadPointer(0, startSample);
            const float* srcR = buffer->getReadPointer(1, startSample);
            
            for (int i = 0; i < 256; ++i)
            {
                block.L[i] = static_cast<double>(srcL[i]);
                block.R[i] = static_cast<double>(srcR[i]);
            }
            
            captureQueue.push(block);
        }
    }
}

// ============================================================================
// CONTROL THREAD: 状態構築ヘルパー
// ============================================================================

convo::EngineState AudioEngine::buildCurrentState() noexcept 
{
    convo::EngineState state;
    
    // 1. ConvolverProcessor の状態をシリアライズ
    uiConvolverProcessor.serializeTo(state.dspBlob, sizeof(state.dspBlob));
    
    // 2. EQEditor の状態をシリアライズ
    if (const auto* eqState = uiEqEditor.getEQStateSnapshot())
    {
        eqState->serializeTo(state.eqBlob, sizeof(state.eqBlob), currentSampleRate.load(std::memory_order_relaxed));
    }
    
    // 3. スナップショット情報（ノイズシェイパー係数など）をシリアライズ
    std::memset(state.snapBlob, 0, sizeof(state.snapBlob));
    
    state.generation++;
    state.sampleRate = currentSampleRate.load(std::memory_order_relaxed);
    state.bitDepth = getDitherBitDepth();
    state.noiseShaperType = NoiseShaperType::Adaptive9thOrder; // TODO: Get from UI or config
    state.captureSessionId = 0; // TODO: generation logic if needed
    state.isValid = true;
    
    return state;
}

// ============================================================================
// CONTROL THREAD: 唯一の書き込みパス
// ============================================================================

void AudioEngine::publishEngineState(convo::EngineState&& newState, float fadeTimeSec) 
{
    // CRITICAL: Single-Writer Guarantee (この関数は単一スレッドからのみ呼ばれる前提)
    
    // 1. 非アクティブスロットの取得
    int active = m_activeIndex.load(std::memory_order_acquire);
    int write  = 1 - active;
    
    convo::EngineView& dst = m_views[write];
    const convo::EngineView& src = m_views[active];
    
    // 2. previous 状態のセットアップ (フェードありの場合)
    if (fadeTimeSec > 0.0f) 
    {
        if (src.previousValid) 
        {
            dst.previous.copyFrom(src.previous);
            dst.previousValid = true;
        } 
        else 
        {
            dst.previous.copyFrom(src.current);
            dst.previousValid = true;
        }
        dst.alpha = 0.0f;
    } 
    else 
    {
        // フェードなし：即時切り替え
        dst.previousValid = false;
        dst.alpha = 1.0f;
    }
    
    // 3. current 状態の更新 (完全コピー)
    dst.current.copyFrom(newState);
    
    // 4. 公開
    m_activeIndex.store(write, std::memory_order_release);
}

void AudioEngine::advanceFade(float step)
{
    int active = m_activeIndex.load(std::memory_order_acquire);
    int write  = 1 - active;

    const convo::EngineView& src = m_views[active];
    convo::EngineView& dst = m_views[write];

    dst.current.copyFrom(src.current);

    if (src.previousValid) {
        dst.previous.copyFrom(src.previous);
        dst.previousValid = true;
    } else {
        dst.previousValid = false;
    }

    dst.alpha = (src.alpha + step > 1.0f) ? 1.0f : (src.alpha + step);

    m_activeIndex.store(write, std::memory_order_release);
}

// ============================================================================
// AUDIO THREAD: DSP 処理実体
// ============================================================================

void AudioEngine::processWithState(juce::AudioBuffer<float>& output, 
                                   const convo::EngineState& state, 
                                   int startSample, 
                                   int numSamples, 
                                   float gain) 
{
    // 1. ゲイン適用（フェード用）
    if (gain != 1.0f && gain > 0.0f) 
    {
        for (int ch = 0; ch < output.getNumChannels(); ++ch) 
        {
            output.applyGain(ch, startSample, numSamples, gain);
        }
    }
    
    // 2. Convolver 処理
    uiConvolverProcessor.processWithBlob(state.dspBlob, output, m_fadeDoubleBuffer, startSample, numSamples);
    
    // 3. EQ 処理
    uiEqEditor.processWithBlob(state.eqBlob, output, m_fadeDoubleBuffer, startSample, numSamples);
}


bool AudioEngine::enqueueLearningCommand(const LearningCommand& cmd) noexcept
{
    const uint32_t currentWrite = learningCommandWrite.load(std::memory_order_relaxed);
    const uint32_t currentRead = learningCommandRead.load(std::memory_order_acquire);
    const uint32_t next = (currentWrite + 1u) & learningCommandBufferMask;
    if (next == currentRead) return false;
    learningCommandBuffer[currentWrite] = cmd;
    learningCommandWrite.store(next, std::memory_order_release);
    return true;
}

bool AudioEngine::dequeueLearningCommand(LearningCommand& cmd) noexcept
{
    const uint32_t currentRead = learningCommandRead.load(std::memory_order_relaxed);
    const uint32_t currentWrite = learningCommandWrite.load(std::memory_order_acquire);
    if (currentRead == currentWrite) return false;
    cmd = learningCommandBuffer[currentRead];
    learningCommandRead.store((currentRead + 1u) & learningCommandBufferMask, std::memory_order_release);
    return true;
}

bool AudioEngine::enqueueLearnerDispatch(const LearnerDispatchAction& action) noexcept
{
    const uint32_t currentWrite = learnerDispatchWrite.load(std::memory_order_relaxed);
    const uint32_t currentRead = learnerDispatchRead.load(std::memory_order_acquire);
    const uint32_t next = (currentWrite + 1u) & learnerDispatchBufferMask;
    if (next == currentRead) {
        lastFailedAction.store(action, std::memory_order_release);
        learnerDispatchOverflow.store(true, std::memory_order_release);
        return false;
    }
    learnerDispatchBuffer[currentWrite] = action;
    learnerDispatchWrite.store(next, std::memory_order_release);
    return true;
}

bool AudioEngine::dequeueLearnerDispatch(LearnerDispatchAction& action) noexcept
{
    const uint32_t currentRead = learnerDispatchRead.load(std::memory_order_relaxed);
    const uint32_t currentWrite = learnerDispatchWrite.load(std::memory_order_acquire);
    if (currentRead == currentWrite) return false;
    action = learnerDispatchBuffer[currentRead];
    learnerDispatchRead.store((currentRead + 1u) & learnerDispatchBufferMask, std::memory_order_release);
    return true;
}

void AudioEngine::setAdaptiveNoiseShaperState(int bankIndex, const NoiseShaperLearner::State& state)
{
    if (bankIndex >= 0 && bankIndex < kAdaptiveNoiseShaperSampleRateBankCount)
        savedStates[bankIndex] = state;
}

bool AudioEngine::getAdaptiveNoiseShaperState(int bankIndex, NoiseShaperLearner::State& state) const
{
    if (bankIndex >= 0 && bankIndex < kAdaptiveNoiseShaperSampleRateBankCount)
    {
        state = savedStates[bankIndex];
        return true;
    }
    return false;
}

void AudioEngine::getAdaptiveCoefficientsForSampleRateAndBitDepth(double sr, int bd, double* coeffs, int order) const
{
    // Placeholder implementation
    std::memcpy(coeffs, NoiseShaperLearner::kDefaultCoeffs.data(), sizeof(double) * std::min(order, (int)kAdaptiveNoiseShaperOrder));
}

void AudioEngine::executeCommit()
{
    publishEngineState(buildCurrentState(), 0.03f); // 30ms fade default
}

void AudioEngine::processRebuildRequestsInternal()
{
    const uint32_t mask = pendingRebuildMask_.exchange(0, std::memory_order_acq_rel);
    if (mask == 0) return;
    
    executeCommit();
}

bool AudioEngine::enqueueSnapshotCommand()
{
    requestRebuild(convo::RebuildKind::Structural);
    return true;
}

void AudioEngine::requestRebuild(double sampleRate, int maxBlockSize)
{
    juce::ignoreUnused(sampleRate, maxBlockSize);
    requestRebuild(convo::RebuildKind::Structural);
}

AudioEngine::LatencyBreakdown AudioEngine::getCurrentLatencyBreakdown() const
{
    LatencyBreakdown b;
    b.totalLatencyBaseRateSamples = uiConvolverProcessor.getLatencyBreakdown().totalLatencyBaseRateSamples;
    return b;
}
