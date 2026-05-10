#include <JuceHeader.h>
#include "AudioEngine.h"
#include "core/RCUReader.h"

static thread_local convo::RCUReader tls_rcuReader;

static void retireDSP(AudioEngine::DSPCore* dsp)
{
    if (dsp) convo::retireObject(dsp, [](void* p) { delete static_cast<AudioEngine::DSPCore*>(p); });
}

namespace
{
    inline double absDiffNoLibm(double a, double b) noexcept
    {
        return absNoLibm(a - b);
    }
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_AUDIO_BLOCK)

void AudioEngine::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    const juce::ScopedNoDenormals noDenormals;
    m_audioBlockCounter.fetch_add(1, std::memory_order_release);

    // 入力検証 (Input Validation)
    if (bufferToFill.buffer == nullptr)
        return;

    const int numSamples = bufferToFill.numSamples;
    const int startSample = bufferToFill.startSample;
    auto* buffer = bufferToFill.buffer;

    // 事前サニティチェック: 絶対的な上限 (1<<20 ≒ 100万サンプル) で明らかな破損データを弾く。
    // DSPCore の maxSamplesPerBlock は prepareToPlay() でホスト指定値を反映して設定されるため、
    // ここで SAFE_MAX_BLOCK_SIZE (65536) を使うと、131072 等の正当なブロックを誤って拒否する。
    // 【Bug Fix】SAFE_MAX_BLOCK_SIZE による早期リジェクトを廃止し、dsp->maxSamplesPerBlock で
    //            正確なチェックを行う (下記 DSPCore 取得後)。
    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20; // 破損データ検出用上限
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // startSampleの妥当性チェック
    if (startSample < 0 || startSample + numSamples > buffer->getNumSamples())
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // Epoch tracking for lock-free Audio Thread safety
    convo::RCUReaderGuard rcuGuard(tls_rcuReader);

    DSPCore* dsp = currentDSP.load(std::memory_order_acquire);
    if (dsp == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    if (dsp != nullptr)
    {
        // DSPCore 固有の上限チェック
        // DSPCore::prepare() でホスト指定の samplesPerBlock を反映した maxSamplesPerBlock が設定される。
        // dsp は RCU で公開済みのため maxSamplesPerBlock は Audio Thread から安全に読み出せる。
        if (numSamples > dsp->maxSamplesPerBlock)
        {
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        // 安全対策: サンプルレート不整合チェック
        // DSPのサンプルレートとエンジンの現在のサンプルレートが一致しない場合、
        // レート変更処理中とみなし、グリッチを防ぐために無音を出力する。
        const double engineSampleRate = currentSampleRate.load(std::memory_order_relaxed);
        if (absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
        {
            // 不整合時はレベルメーターもリセットして誤表示を防ぐ
            inputLevelLinear.store(0.0f);
            outputLevelLinear.store(0.0f);
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        // パラメータのロード
        // 【Parameter安全設計】
        // Audio ThreadではAtomic変数の読み取りのみを行い、ロックやメモリ確保を伴う処理は行わない。
        // 構造変更が必要な場合は、別途フラグやUIスレッド経由で再構築を行う。
        // ── Audio Thread 最適化: GlobalSnapshot を優先し、fallback で atomics を読む ──
        const convo::GlobalSnapshot* snap = m_coordinator.getCurrent();
        const bool eqBypassed               = (snap != nullptr) ? snap->eqBypass : eqBypassRequested.load(std::memory_order_acquire);
        const bool convBypassed             = (snap != nullptr) ? snap->convBypass : convBypassRequested.load(std::memory_order_acquire);
        const ProcessingOrder order         = (snap != nullptr) ? snap->processingOrder : currentProcessingOrder.load(std::memory_order_relaxed);
        const AnalyzerSource analyzerSource = currentAnalyzerSource.load(std::memory_order_relaxed);
        const bool analyzerEnabledNow       = analyzerEnabled.load(std::memory_order_relaxed);
        const bool softClip                 = (snap != nullptr) ? snap->softClipEnabled : softClipEnabled.load(std::memory_order_relaxed);
        const float satAmt                  = (snap != nullptr) ? snap->saturationAmount : saturationAmount.load(std::memory_order_relaxed);
        const double headroomGain           = (snap != nullptr) ? snap->inputHeadroomGain : inputHeadroomGain.load(std::memory_order_relaxed);
        const double makeupGain             = (snap != nullptr) ? snap->outputMakeupGain : outputMakeupGain.load(std::memory_order_relaxed);
        const double convInputTrimGain      = (snap != nullptr) ? snap->convInputTrimGain : convolverInputTrimGain.load(std::memory_order_relaxed);
        const convo::HCMode hcMode      = convHCFilterMode.load(std::memory_order_relaxed);
        const convo::LCMode lcMode      = convLCFilterMode.load(std::memory_order_relaxed);
        const convo::HCMode lpfMode     = eqLPFFilterMode.load(std::memory_order_relaxed);
        const int adaptiveCoeffBankIndex    = currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
        const auto& adaptiveCoeffBank       = getAdaptiveCoeffBankForIndex(adaptiveCoeffBankIndex);
        const bool adaptiveCaptureEnabled   = noiseShaperLearner && noiseShaperLearner->isRunning();

        // RCU スナップショット取得：generation と active ポインタはダブルバッファリングにより一貫性が保証される
        const uint32_t genSnapshot = adaptiveCoeffBank.generation.load(std::memory_order_acquire);
        const CoeffSet* safeAdaptiveSet = AudioEngine::getActiveCoeffSet(adaptiveCoeffBank);
        // safeAdaptiveSet は、genSnapshot 時点で有効な係数セットを指す。
        // Writer が後で切り替えても、このポインタの指す内容は不変である。

        // 念のため nullptr チェック
        if (!safeAdaptiveSet) {
            // フォールバック：デフォルト係数を使用するなどの処理（必要に応じて）
            // ここでは単に nullptr のまま処理を続行（process() 側で対処）
        }
        const uint32_t adaptiveGenAfter = genSnapshot; // 互換性のため変数名を維持

        // UI表示用: 比較なしで直接ストア（ロード→比較→ストアより高速）
        eqBypassActive.store(eqBypassed, std::memory_order_relaxed);
        convBypassActive.store(convBypassed, std::memory_order_relaxed);

        DSPCore::ProcessingState procState {
            .eqBypassed               = eqBypassed,
            .convBypassed             = convBypassed,
            .order                    = order,
            .analyzerSource           = analyzerSource,
            .analyzerEnabled          = analyzerEnabledNow,
            .softClipEnabled          = softClip,
            .saturationAmount         = satAmt,
            .inputHeadroomGain        = headroomGain,
            .outputMakeupGain         = makeupGain,
            .convolverInputTrimGain   = convInputTrimGain,
            .convHCMode               = hcMode,
            .convLCMode               = lcMode,
            .eqLPFMode                = lpfMode,
            .adaptiveCoeffBankIndex   = adaptiveCoeffBankIndex,
            .adaptiveCoeffSet         = safeAdaptiveSet,
            .adaptiveCoeffGeneration  = adaptiveGenAfter,
            .adaptiveCaptureSampleRateHz = static_cast<int>(dsp->sampleRate + 0.5),
            .adaptiveCaptureBitDepth  = dsp->ditherBitDepth,
            .captureSessionId         = dsp->currentCaptureSessionId,
            .adaptiveCaptureQueue     = adaptiveCaptureEnabled ? &audioCaptureQueue : nullptr
        };

        if (m_coordinator.isFading())
            m_coordinator.advanceFade(numSamples);
        debugLastCoordinatorIsFading.store(m_coordinator.isFading() ? 1 : 0, std::memory_order_relaxed);

        float snapshotAlpha = 1.0f;
        const convo::GlobalSnapshot* snapshotFrom = nullptr;
        const convo::GlobalSnapshot* snapshotTo = nullptr;
        const bool updateFadeReturned = m_coordinator.updateFade(snapshotAlpha, snapshotFrom, snapshotTo);
        debugLastUpdateFadeReturned.store(updateFadeReturned ? 1 : 0, std::memory_order_relaxed);
        debugLastSnapshotFromNull.store(snapshotFrom == nullptr ? 1 : 0, std::memory_order_relaxed);
        debugLastSnapshotToNull.store(snapshotTo == nullptr ? 1 : 0, std::memory_order_relaxed);

        const bool snapshotFading = updateFadeReturned
            && snapshotTo != nullptr;

        if (snapshotFading)
        {
            const int fadeChannels = std::min(dspCrossfadeFloatBuffer.getNumChannels(), buffer->getNumChannels());
            for (int ch = 0; ch < fadeChannels; ++ch)
                dspCrossfadeFloatBuffer.clear(ch, 0, numSamples);

            juce::AudioSourceChannelInfo oldInfo(&dspCrossfadeFloatBuffer, 0, numSamples);
            processWithSnapshot(oldInfo, snapshotFrom, true);
            processWithSnapshot(bufferToFill, snapshotTo, false);

            const float gNew = snapshotAlpha;
            const float gOld = 1.0f - snapshotAlpha;
            const int outChannels = std::min(buffer->getNumChannels(), dspCrossfadeFloatBuffer.getNumChannels());
            float* dstL = (outChannels > 0) ? buffer->getWritePointer(0, startSample) : nullptr;
            float* dstR = (outChannels > 1) ? buffer->getWritePointer(1, startSample) : nullptr;
            const float* oldL = (outChannels > 0) ? dspCrossfadeFloatBuffer.getReadPointer(0, 0) : nullptr;
            const float* oldR = (outChannels > 1) ? dspCrossfadeFloatBuffer.getReadPointer(1, 0) : nullptr;

            for (int i = 0; i < numSamples; ++i)
            {
                if (dstL != nullptr)
                    dstL[i] = dstL[i] * gNew + oldL[i] * gOld;
                if (dstR != nullptr)
                    dstR[i] = dstR[i] * gNew + oldR[i] * gOld;
            }

            return;
        }

        DSPCore* fading = sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire));
        bool useDryAsOld = dspCrossfadeUseDryAsOld.load(std::memory_order_acquire);
        int pendingFadeDelayBlocks = dspCrossfadeStartDelayBlocks.load(std::memory_order_acquire);
        if (fading != nullptr
            && !useDryAsOld
            && dspCrossfadePending.load(std::memory_order_acquire)
            && pendingFadeDelayBlocks > 0)
        {
            dspCrossfadeStartDelayBlocks.store(pendingFadeDelayBlocks - 1, std::memory_order_release);

            auto fadingState = procState;
            fadingState.analyzerEnabled = false;
            fadingState.adaptiveCaptureQueue = nullptr;

            std::atomic<float> fadingInputMeter { 0.0f };
            std::atomic<float> fadingOutputMeter { 0.0f };
            fading->process(bufferToFill, analyzerFifo, inputLevelLinear, outputLevelLinear, fadingState);
            return;
        }

        if ((fading != nullptr || firstIrDryCrossfadePending.load(std::memory_order_acquire))
            && dspCrossfadePending.exchange(false, std::memory_order_acq_rel))
        {
            const double fadeSec = std::max(0.001, queuedFadeTimeSec.load(std::memory_order_acquire));
            dspCrossfadeGain.reset(std::max(1.0, dsp->sampleRate), fadeSec);
            dspCrossfadeGain.setCurrentAndTargetValue(0.0);
            dspCrossfadeGain.setTargetValue(1.0);

            // レイテンシ整合値を Audio Thread スナップショットへ反映する。
            latencyDelayOld_RT = latencyDelayOld.load(std::memory_order_acquire);
            latencyDelayNew_RT = latencyDelayNew.load(std::memory_order_acquire);

            if (firstIrDryCrossfadePending.exchange(false, std::memory_order_acq_rel))
            {
                dspCrossfadeUseDryAsOld.store(true, std::memory_order_release);
                useDryAsOld = true;
            }
        }

        const bool canCrossfade = (fading != nullptr || useDryAsOld)
            && dspCrossfadeGain.isSmoothing()
            && dspCrossfadeFloatBuffer.getNumChannels() >= 2
            && dspCrossfadeFloatBuffer.getNumSamples() >= numSamples;

        if (canCrossfade)
        {
            juce::AudioSourceChannelInfo fadeInfo(&dspCrossfadeFloatBuffer, 0, numSamples);
            dspCrossfadeFloatBuffer.clear(0, 0, numSamples);
            dspCrossfadeFloatBuffer.clear(1, 0, numSamples);

            auto fadingState = procState;
            fadingState.analyzerEnabled = false;
            fadingState.adaptiveCaptureQueue = nullptr;

            std::atomic<float> fadingInputMeter { 0.0f };
            std::atomic<float> fadingOutputMeter { 0.0f };
            if (useDryAsOld)
            {
                const int outChannels = std::min(2, buffer->getNumChannels());
                if (outChannels > 0)
                    juce::FloatVectorOperations::copy(dspCrossfadeFloatBuffer.getWritePointer(0, 0), buffer->getReadPointer(0, startSample), numSamples);
                if (outChannels > 1)
                    juce::FloatVectorOperations::copy(dspCrossfadeFloatBuffer.getWritePointer(1, 0), buffer->getReadPointer(1, startSample), numSamples);
            }
            else
            {
                // EBR: lifetime managed by RCUReader
                fading->processToBuffer(bufferToFill, dspCrossfadeFloatBuffer, analyzerFifo,
                                       fadingInputMeter, fadingOutputMeter, fadingState);
            }
            dsp->process(bufferToFill, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

            const int outChannels = std::min(2, buffer->getNumChannels());
            float* dstL = (outChannels > 0) ? buffer->getWritePointer(0, startSample) : nullptr;
            float* dstR = (outChannels > 1) ? buffer->getWritePointer(1, startSample) : nullptr;
            const float* oldL = (outChannels > 0) ? dspCrossfadeFloatBuffer.getReadPointer(0, 0) : nullptr;
            const float* oldR = (outChannels > 1) ? dspCrossfadeFloatBuffer.getReadPointer(1, 0) : nullptr;

            const int bufferSize = latencyBufSize;
            int writePos = latencyWritePos;
            const int delayOld = latencyDelayOld_RT;
            const int delayNew = latencyDelayNew_RT;
            if (latencyResetPending.exchange(false, std::memory_order_acq_rel))
            {
                if (latencyBufOldL) std::memset(latencyBufOldL, 0, sizeof(double) * bufferSize);
                if (latencyBufOldR) std::memset(latencyBufOldR, 0, sizeof(double) * bufferSize);
                if (latencyBufNewL) std::memset(latencyBufNewL, 0, sizeof(double) * bufferSize);
                if (latencyBufNewR) std::memset(latencyBufNewR, 0, sizeof(double) * bufferSize);
                writePos = 0;
            }
            for (int i = 0; i < numSamples; ++i)
            {
                latencyBufOldL[writePos] = (oldL != nullptr) ? static_cast<double>(oldL[i]) : 0.0;
                latencyBufOldR[writePos] = (oldR != nullptr) ? static_cast<double>(oldR[i]) : 0.0;
                latencyBufNewL[writePos] = (dstL != nullptr) ? static_cast<double>(dstL[i]) : 0.0;
                latencyBufNewR[writePos] = (dstR != nullptr) ? static_cast<double>(dstR[i]) : 0.0;

                int readOld = writePos - delayOld;
                int readNew = writePos - delayNew;
                while (readOld < 0) readOld += bufferSize;
                while (readOld >= bufferSize) readOld -= bufferSize;
                while (readNew < 0) readNew += bufferSize;
                while (readNew >= bufferSize) readNew -= bufferSize;

                const double alignedOldL = latencyBufOldL[readOld];
                const double alignedOldR = latencyBufOldR[readOld];
                const double alignedNewL = latencyBufNewL[readNew];
                const double alignedNewR = latencyBufNewR[readNew];

                const double gNew = dspCrossfadeGain.getNextValue();
                const double dryScale = useDryAsOld ? dspCrossfadeDryScaleGain.getNextValue() : 1.0;
                const double gOld = 1.0 - gNew;
                const double dryScaledL = alignedOldL * dryScale;
                const double dryScaledR = alignedOldR * dryScale;
                if (dstL != nullptr)
                    dstL[i] = static_cast<float>(alignedNewL * gNew + dryScaledL * gOld);
                if (dstR != nullptr)
                    dstR[i] = static_cast<float>(alignedNewR * gNew + dryScaledR * gOld);

                writePos++;
                if (writePos >= bufferSize)
                    writePos = 0;
            }
            latencyWritePos = writePos;

            if (!useDryAsOld)
            {
                // EBR: fading lifetime managed by RCUReaderGuard
            }

            if (!dspCrossfadeGain.isSmoothing())
            {
                if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                    retireDSP(done);
                dspCrossfadeGain.setCurrentAndTargetValue(1.0);
                dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
                dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
            }
        }
        else
        {
            // 通常パス（クロスフェードなし）：RCU で dsp の生存が保証されるため addRef/release 不要
            dsp->process(bufferToFill, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

            if (fading != nullptr && !dspCrossfadeGain.isSmoothing())
            {
                if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                    retireDSP(done);
            }
            if (!dspCrossfadeGain.isSmoothing())
                dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
        }
    }

}

#endif
