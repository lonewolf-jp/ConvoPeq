#include <JuceHeader.h>
#include "AudioEngine.h"

static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_PREPARE_TO_PLAY)

void AudioEngine::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
{
    ASSERT_NON_RT_THREAD();
    // P0-A0: LifecycleIsolationRuntime integration
    auto lifecycleToken = lifecycleRuntime_.enterPrepare(samplesPerBlockExpected, static_cast<int>(sampleRate));

    diagLog("[DIAG] prepareToPlay: enter spb=" + juce::String(samplesPerBlockExpected) + " sr=" + juce::String(sampleRate, 2));

    const auto rollbackPrepareFailure = [this]() noexcept
    {
        if (latencyBufOldL) { convo::aligned_free(latencyBufOldL); latencyBufOldL = nullptr; }
        if (latencyBufOldR) { convo::aligned_free(latencyBufOldR); latencyBufOldR = nullptr; }
        if (latencyBufNewL) { convo::aligned_free(latencyBufNewL); latencyBufNewL = nullptr; }
        if (latencyBufNewR) { convo::aligned_free(latencyBufNewR); latencyBufNewR = nullptr; }
        latencyBufSize = 0;
        latencyWritePos = 0;
        convo::publishAtomic(latencyDelayOld, 0, std::memory_order_release);
        convo::publishAtomic(latencyDelayNew, 0, std::memory_order_release);
        convo::publishAtomic(latencyResetPending, false, std::memory_order_release);
        convo::publishAtomic(dspCrossfadePending, false, std::memory_order_release);
        convo::publishAtomic(firstIrDryCrossfadePending, false, std::memory_order_release);
        convo::publishAtomic(dspCrossfadeUseDryAsOld, false, std::memory_order_release);
        convo::publishAtomic(dspCrossfadeStartDelayBlocks, 0, std::memory_order_release);
        convo::publishAtomic(dspCrossfadeDryHoldSamples, 0, std::memory_order_release);
        refreshCrossfadePreparedSnapshotFromAtomics();
        convo::publishAtomic(lifecycleState, EngineLifecycleState::Unprepared, std::memory_order_release);
    };

    auto previousState = convo::consumeAtomic(lifecycleState, std::memory_order_acquire);
    for (;;)
    {
        if (previousState == EngineLifecycleState::Releasing
            || previousState == EngineLifecycleState::Destroyed
            || previousState == EngineLifecycleState::Preparing)
        {
            diagLog("[DIAG] prepareToPlay: blocked by lifecycle state=" + juce::String(static_cast<int>(previousState)));
            return;
        }

        if (convo::compareExchangeAtomic(lifecycleState,
                         previousState,
                         EngineLifecycleState::Preparing,
                         std::memory_order_acq_rel,
                         std::memory_order_acquire))
            break;
    }

    if (previousState == EngineLifecycleState::Prepared)
        diagLog("[DIAG] prepareToPlay: re-prepare requested without release; proceeding with safe reinitialization");

    setShutdownPhase(ShutdownPhase::Running, "prepareToPlay");

    // releaseResources() で停止済みの場合に備えて、必要なら rebuild thread を再起動する。
    if (!rebuildThread.joinable())
    {
        {
            std::lock_guard<std::mutex> lock(rebuildMutex);
            convo::publishAtomic(rebuildThreadShouldExit, false, std::memory_order_release);
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
    const bool rateChanged = (std::abs(convo::consumeAtomic(currentSampleRate, std::memory_order_acquire) - safeSampleRate) > 1e-6);
    const bool blockSizeChanged = (convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire) != bufferSize);

    convo::publishAtomic(maxSamplesPerBlock, bufferSize, std::memory_order_release);
    convo::publishAtomic(currentSampleRate, safeSampleRate, std::memory_order_release);
    {
        const int irFadeSamples = juce::jmax(0, convo::consumeAtomic(m_irFadeSamples, std::memory_order_acquire));
        const double irFadeSec = (irFadeSamples > 0)
            ? (static_cast<double>(irFadeSamples) / safeSampleRate)
            : 0.001;
        convo::publishAtomic(m_irFadeTimeSec, irFadeSec, std::memory_order_release);
    }
    dspCrossfadeGain.reset(safeSampleRate, 0.03);
    dspCrossfadeGain.setCurrentAndTargetValue(1.0);
    convo::publishAtomic(dspCrossfadePending, false, std::memory_order_release);
    convo::publishAtomic(firstIrDryCrossfadePending, false, std::memory_order_release);
    convo::publishAtomic(dspCrossfadeUseDryAsOld, false, std::memory_order_release);
    convo::publishAtomic(dspCrossfadeStartDelayBlocks, 0, std::memory_order_release);
    convo::publishAtomic(dspCrossfadeDryHoldSamples, 0, std::memory_order_release);
    {
        const auto* runtimeWorld = runtimeStore.observe();
        const auto* runtimeGraph = (runtimeWorld != nullptr) ? &runtimeWorld->graph : nullptr;
        auto* currentForPublish = (runtimeGraph != nullptr)
            ? static_cast<DSPCore*>(runtimeGraph->activeNode)
            : nullptr;
        auto* fadingForPublish = resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph);
        const bool hasAnyRuntime = (currentForPublish != nullptr) || (fadingForPublish != nullptr);
        if (hasAnyRuntime)
        {
            const auto ts = runtimeWorld != nullptr ? runtimeWorld->engine.transition : convo::TransitionState{};
            RuntimePublicationCoordinator::create(RuntimePublicationBridge { *this }, runtimeStore)
                .publishState(currentForPublish,
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
    convo::publishAtomic(inputLevelLinear, 0.0f, std::memory_order_release);
    convo::publishAtomic(outputLevelLinear, 0.0f, std::memory_order_release);

    convo::publishAtomic(eqBypassActive, convo::consumeAtomic(eqBypassRequested, std::memory_order_acquire), std::memory_order_release);
    convo::publishAtomic(convBypassActive, convo::consumeAtomic(convBypassRequested, std::memory_order_acquire), std::memory_order_release);

    // --- レイテンシ整合バッファの再確保 ---
    // ※本関数はAudioThread停止中のみ呼ぶこと！
    // 最大遅延（2秒上限・kMaxLatencySamples制限）
    // +blockSizeはwrap安全余裕（リングバッファwrap時の読み出し安全域）
    const int maxDelay = std::min(kMaxLatencySamples, static_cast<int>(safeSampleRate * 2.0));
    const int requiredLatencyBufSize = maxDelay + bufferSize + 2;
    const bool needsLatencyReallocation = (latencyBufSize != requiredLatencyBufSize)
        || (latencyBufOldL == nullptr)
        || (latencyBufOldR == nullptr)
        || (latencyBufNewL == nullptr)
        || (latencyBufNewR == nullptr);

    if (needsLatencyReallocation)
    {
        if (latencyBufOldL) { convo::aligned_free(latencyBufOldL); latencyBufOldL = nullptr; }
        if (latencyBufOldR) { convo::aligned_free(latencyBufOldR); latencyBufOldR = nullptr; }
        if (latencyBufNewL) { convo::aligned_free(latencyBufNewL); latencyBufNewL = nullptr; }
        if (latencyBufNewR) { convo::aligned_free(latencyBufNewR); latencyBufNewR = nullptr; }

        latencyBufSize = requiredLatencyBufSize;
        latencyBufOldL = convo::makeAlignedArray<double>(static_cast<size_t>(latencyBufSize)).release();
        latencyBufOldR = convo::makeAlignedArray<double>(static_cast<size_t>(latencyBufSize)).release();
        latencyBufNewL = convo::makeAlignedArray<double>(static_cast<size_t>(latencyBufSize)).release();
        latencyBufNewR = convo::makeAlignedArray<double>(static_cast<size_t>(latencyBufSize)).release();

        // malloc失敗時は安全フェイル
        if (!latencyBufOldL || !latencyBufOldR || !latencyBufNewL || !latencyBufNewR) {
            rollbackPrepareFailure();
            return;
        }
    }

    std::memset(latencyBufOldL, 0, sizeof(double) * latencyBufSize);
    std::memset(latencyBufOldR, 0, sizeof(double) * latencyBufSize);
    std::memset(latencyBufNewL, 0, sizeof(double) * latencyBufSize);
    std::memset(latencyBufNewR, 0, sizeof(double) * latencyBufSize);

    latencyWritePos = 0;
    convo::publishAtomic(latencyDelayOld, 0, std::memory_order_release);
    convo::publishAtomic(latencyDelayNew, 0, std::memory_order_release);
    convo::publishAtomic(latencyResetPending, false, std::memory_order_release);
    refreshCrossfadePreparedSnapshotFromAtomics();

    latencyDelayOld_RT = 0;
    latencyDelayNew_RT = 0;

    // 初回IRロード前でも currentDSP を常に有効にし、DSP->DSP クロスフェードへ統一する。
    const auto runtimePublishView = getRuntimePublishView();
    const bool hasPublishedCurrent = (runtimePublishView.graph != nullptr)
        && (static_cast<DSPCore*>(runtimePublishView.graph->activeNode) != nullptr);
    if (!hasPublishedCurrent && activeDSP == nullptr)
    {
        convo::aligned_unique_ptr<DSPCore> placeholderDSP;
        try
        {
            placeholderDSP = convo::aligned_make_unique<DSPCore>();
            placeholderDSP->convolverRt().setVisualizationEnabled(false);
            placeholderDSP->prepare(safeSampleRate,
                                    bufferSize,
                                    convo::consumeAtomic(ditherBitDepth, std::memory_order_acquire),
                                    convo::consumeAtomic(manualOversamplingFactor, std::memory_order_acquire),
                                    convo::consumeAtomic(oversamplingType, std::memory_order_acquire),
                                    convo::consumeAtomic(noiseShaperType, std::memory_order_acquire),
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
        convo::publishAtomic(lastCommittedConvolverHasIr_, false, std::memory_order_release);
        convo::publishAtomic(lastCommittedConvolverStructuralHash_, 0, std::memory_order_release);
        RuntimePublicationCoordinator::create(RuntimePublicationBridge { *this }, runtimeStore)
            .publishState(activeDSP,
                          nullptr,
                          convo::TransitionPolicy::HardReset,
                          0.0,
                          false);
    }

    // --- DSP再ビルド判定・同期 ---
    uiConvolverProcessor.prepareToPlay(safeSampleRate, bufferSize);
    if (rateChanged)
        uiConvolverProcessor.invalidatePendingLoads();
    const bool hasCurrentRuntime = (runtimePublishView.graph != nullptr)
        && (static_cast<DSPCore*>(runtimePublishView.graph->activeNode) != nullptr);
    if (rateChanged || blockSizeChanged || !hasCurrentRuntime) {
        if (juce::MessageManager::getInstance()->isThisTheMessageThread()) {
            requestRebuild(safeSampleRate, bufferSize);
        } else {
            requestRebuild(convo::RebuildKind::Structural);
        }
    }
    convo::publishAtomic(lifecycleState, EngineLifecycleState::Prepared, std::memory_order_release);
    diagLog("[DIAG] prepareToPlay: exit currentSR=" + juce::String(convo::consumeAtomic(currentSampleRate, std::memory_order_acquire), 2) + " maxSPB=" + juce::String(convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire)));

    // P0-A0: LifecycleIsolationRuntime integration - leave prepare phase
    lifecycleRuntime_.leavePrepare(lifecycleToken);
}

#endif
