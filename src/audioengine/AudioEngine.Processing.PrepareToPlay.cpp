#include <JuceHeader.h>
#include "AudioEngine.h"

extern std::atomic<bool> gShuttingDown;

static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_PREPARE_TO_PLAY)

void AudioEngine::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
{
    ASSERT_NON_RT_THREAD();
    diagLog("[DIAG] prepareToPlay: enter spb=" + juce::String(samplesPerBlockExpected) + " sr=" + juce::String(sampleRate, 2));

    const auto rollbackPrepareFailure = [this]() noexcept
    {
        if (latencyBufOldL) { _aligned_free(latencyBufOldL); latencyBufOldL = nullptr; }
        if (latencyBufOldR) { _aligned_free(latencyBufOldR); latencyBufOldR = nullptr; }
        if (latencyBufNewL) { _aligned_free(latencyBufNewL); latencyBufNewL = nullptr; }
        if (latencyBufNewR) { _aligned_free(latencyBufNewR); latencyBufNewR = nullptr; }
        latencyBufSize = 0;
        latencyWritePos = 0;
        latencyDelayOld.store(0, std::memory_order_release);
        latencyDelayNew.store(0, std::memory_order_release);
        latencyResetPending.store(false, std::memory_order_release);
        lifecycleState.store(EngineLifecycleState::Unprepared, std::memory_order_release);
    };

    auto previousState = lifecycleState.load(std::memory_order_acquire);
    for (;;)
    {
        if (previousState == EngineLifecycleState::Releasing
            || previousState == EngineLifecycleState::Destroyed
            || previousState == EngineLifecycleState::Preparing)
        {
            diagLog("[DIAG] prepareToPlay: blocked by lifecycle state=" + juce::String(static_cast<int>(previousState)));
            return;
        }

        if (lifecycleState.compare_exchange_weak(previousState,
                                                 EngineLifecycleState::Preparing,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire))
            break;
    }

    if (previousState == EngineLifecycleState::Prepared)
        diagLog("[DIAG] prepareToPlay: re-prepare requested without release; proceeding with safe reinitialization");

    gShuttingDown.store(false, std::memory_order_release);
    shutdownInProgress.store(false, std::memory_order_release);
    setShutdownPhase(ShutdownPhase::Running, "prepareToPlay");

    // releaseResources() で停止済みの場合に備えて、必要なら rebuild thread を再起動する。
    if (!rebuildThread.joinable())
    {
        {
            std::lock_guard<std::mutex> lock(rebuildMutex);
            rebuildThreadShouldExit.store(false, std::memory_order_release);
            hasPendingTask = false;
            pendingTask = RebuildTask{};
        }
        rebuildThread = std::thread(&AudioEngine::rebuildThreadLoop, this);
    }

    // --- AudioEngine::prepareToPlay ---
    // ※本関数は「AudioThread停止中のみ呼ぶ」ことがJUCE AudioSource仕様上の前提です。
    //   これを破るとバッファfree競合・data raceの危険があります。

    // パラメータ検証 (Parameter Validation)
    double safeSampleRate = sampleRate;
    if (safeSampleRate <= 0.0 || safeSampleRate > SAFE_MAX_SAMPLE_RATE || !std::isfinite(safeSampleRate)) {
        jassertfalse;
        safeSampleRate = 48000.0;
    }
    if (samplesPerBlockExpected <= 0) {
        jassertfalse;
        samplesPerBlockExpected = 512;
    }
    const int bufferSize = samplesPerBlockExpected;

    // サンプルレート・ブロックサイズ変更検知
    const bool rateChanged = (std::abs(currentSampleRate.load() - safeSampleRate) > 1e-6);
    const bool blockSizeChanged = (maxSamplesPerBlock.load() != bufferSize);

    maxSamplesPerBlock.store(bufferSize);
    currentSampleRate.store(safeSampleRate);
    {
        const int irFadeSamples = juce::jmax(0, m_irFadeSamples.load(std::memory_order_relaxed));
        const double irFadeSec = (irFadeSamples > 0)
            ? (static_cast<double>(irFadeSamples) / safeSampleRate)
            : 0.001;
        m_irFadeTimeSec.store(irFadeSec, std::memory_order_relaxed);
    }
    dspCrossfadeGain.reset(safeSampleRate, 0.03);
    dspCrossfadeGain.setCurrentAndTargetValue(1.0);
    dspCrossfadePending.store(false, std::memory_order_release);
    {
        const auto* runtimeGraph = getRuntimeGraphState();
        auto* currentForPublish = resolveCurrentDSPFromRuntimePublish(runtimeGraph);
        auto* fadingForPublish = resolveFadingDSPFromRuntimePublish(runtimeGraph);
        const bool hasAnyRuntime = (currentForPublish != nullptr) || (fadingForPublish != nullptr);
        if (hasAnyRuntime)
        {
            const auto ts = getRuntimeTransitionStateForDebug();
            publishRuntimeSnapshots(currentForPublish,
                                    fadingForPublish,
                                    ts.policy,
                                    ts.fadeTimeSec,
                                    ts.active);
        }
    }
    selectAdaptiveCoeffBankForCurrentSettings();

    dspCrossfadeFloatBuffer.setSize(2, std::max(SAFE_MAX_BLOCK_SIZE, bufferSize), false, false, true);
    dspCrossfadeDoubleBuffer.setSize(2, std::max(SAFE_MAX_BLOCK_SIZE, bufferSize), false, false, true);

    analyzerFifo.prepare(2, FIFO_SIZE);
    inputLevelLinear.store(0.0f);
    outputLevelLinear.store(0.0f);

    eqBypassActive.store(eqBypassRequested.load(std::memory_order_relaxed), std::memory_order_relaxed);
    convBypassActive.store(convBypassRequested.load(std::memory_order_relaxed), std::memory_order_relaxed);

    // --- レイテンシ整合バッファの再確保 ---
    // ※本関数はAudioThread停止中のみ呼ぶこと！
    if (latencyBufOldL) { _aligned_free(latencyBufOldL); latencyBufOldL = nullptr; }
    if (latencyBufOldR) { _aligned_free(latencyBufOldR); latencyBufOldR = nullptr; }
    if (latencyBufNewL) { _aligned_free(latencyBufNewL); latencyBufNewL = nullptr; }
    if (latencyBufNewR) { _aligned_free(latencyBufNewR); latencyBufNewR = nullptr; }

    // 最大遅延（2秒上限・kMaxLatencySamples制限）
    // +blockSizeはwrap安全余裕（リングバッファwrap時の読み出し安全域）
    const int maxDelay = std::min(kMaxLatencySamples, static_cast<int>(safeSampleRate * 2.0));
    latencyBufSize = maxDelay + bufferSize + 2;

    latencyBufOldL = (double*)_aligned_malloc(sizeof(double) * latencyBufSize, 64);
    latencyBufOldR = (double*)_aligned_malloc(sizeof(double) * latencyBufSize, 64);
    latencyBufNewL = (double*)_aligned_malloc(sizeof(double) * latencyBufSize, 64);
    latencyBufNewR = (double*)_aligned_malloc(sizeof(double) * latencyBufSize, 64);

    // malloc失敗時は安全フェイル
    if (!latencyBufOldL || !latencyBufOldR || !latencyBufNewL || !latencyBufNewR) {
        rollbackPrepareFailure();
        return;
    }

    std::memset(latencyBufOldL, 0, sizeof(double) * latencyBufSize);
    std::memset(latencyBufOldR, 0, sizeof(double) * latencyBufSize);
    std::memset(latencyBufNewL, 0, sizeof(double) * latencyBufSize);
    std::memset(latencyBufNewR, 0, sizeof(double) * latencyBufSize);

    latencyWritePos = 0;
    latencyDelayOld.store(0, std::memory_order_release);
    latencyDelayNew.store(0, std::memory_order_release);
    latencyResetPending.store(false, std::memory_order_release);

    latencyDelayOld_RT = 0;
    latencyDelayNew_RT = 0;

    // 初回IRロード前でも currentDSP を常に有効にし、DSP->DSP クロスフェードへ統一する。
    const auto* engineRuntime = getEngineRuntimeState();
    const auto* runtimeGraph = getRuntimeGraphState();
    const bool hasPublishedCurrent = (runtimePublishedCurrentDSP(engineRuntime, runtimeGraph) != nullptr);
    if (!hasPublishedCurrent && activeDSP == nullptr)
    {
        std::unique_ptr<DSPCore> placeholderDSP;
        try
        {
            placeholderDSP = std::make_unique<DSPCore>();
            placeholderDSP->convolverRt().setVisualizationEnabled(false);
            placeholderDSP->prepare(safeSampleRate,
                                    bufferSize,
                                    ditherBitDepth.load(std::memory_order_relaxed),
                                    manualOversamplingFactor.load(std::memory_order_relaxed),
                                    oversamplingType.load(std::memory_order_relaxed),
                                    noiseShaperType.load(std::memory_order_relaxed),
                                    this);
            placeholderDSP->convolverRt().setBypass(true);

            int predictedLatency = juce::nextPowerOfTwo(std::max(bufferSize, 64));
            predictedLatency = juce::jlimit(0, latencyBufSize - 1, predictedLatency);
            placeholderDSP->setFixedLatencySamples(predictedLatency);
        }
        catch (...)
        {
            rollbackPrepareFailure();
            return;
        }

        activeDSP = placeholderDSP.release();
        currentDSP.store(activeDSP, std::memory_order_release);
        publishRuntimeSnapshots(activeDSP,
                    nullptr,
                    convo::TransitionPolicy::HardReset,
                    0.0,
                    false);
        publishRuntimeTransitionState(activeDSP, nullptr, convo::TransitionPolicy::HardReset, 0.0, false);
    }

    // --- DSP再ビルド判定・同期 ---
    uiConvolverProcessor.prepareToPlay(safeSampleRate, bufferSize);
    if (rateChanged)
        uiConvolverProcessor.invalidatePendingLoads();
    const auto* runtimeGraphForRebuildCheck = getRuntimeGraphState();
    const bool hasCurrentRuntime = (resolveCurrentDSPFromRuntimePublish(
                                                                       runtimeGraphForRebuildCheck) != nullptr);
    if (rateChanged || blockSizeChanged || !hasCurrentRuntime) {
        if (juce::MessageManager::getInstance()->isThisTheMessageThread()) {
            requestRebuild(safeSampleRate, bufferSize);
        } else {
            requestRebuild(convo::RebuildKind::Structural);
        }
    }
    lifecycleState.store(EngineLifecycleState::Prepared, std::memory_order_release);
    diagLog("[DIAG] prepareToPlay: exit currentSR=" + juce::String(currentSampleRate.load(), 2) + " maxSPB=" + juce::String(maxSamplesPerBlock.load()));
}

#endif
