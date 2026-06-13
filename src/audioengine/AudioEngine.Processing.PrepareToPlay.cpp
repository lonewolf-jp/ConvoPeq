#include <JuceHeader.h>
#include "AudioEngine.h"
#include "core/RuntimeReaderContext.h"
#include "RuntimeBuilder.h"
#include "RuntimePublicationOrchestrator.h"

namespace {
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

void AudioEngine::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
{
    ASSERT_NON_RT_THREAD();
    // P0-A0: LifecycleIsolationRuntime integration
    auto lifecycleToken = lifecycleRuntime_.enterPrepare(samplesPerBlockExpected, static_cast<int>(sampleRate));

    diagLog("[DIAG] prepareToPlay: enter spb=" + juce::String(samplesPerBlockExpected) + " sr=" + juce::String(sampleRate, 2));
    diagLog("[DIAG] prepareToPlay: lifecycleToken acquired");

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
        crossfadeRuntime_.reset();
        refreshCrossfadePreparedSnapshotFromAtomics();
        convo::publishAtomic(lifecycleState, EngineLifecycleState::Unprepared, std::memory_order_release);
    };

    // ★ C-4: prepareToPlay 開始時に HealthState をリセット
    //   （releaseResources 直後には呼ばない — Shutdown 診断情報を観測する前に消えるのを防ぐため）
    m_healthMonitor.reset();

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

    diagLog("[DIAG] prepareToPlay: lifecycle state set to Preparing");
    setShutdownPhase(ShutdownPhase::Running, "prepareToPlay");
    diagLog("[DIAG] prepareToPlay: shutdownPhase set to Running");

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
        diagLog("[DIAG] prepareToPlay: rebuild thread started");
    }
    diagLog("[DIAG] prepareToPlay: rebuild thread check done");

    // ★ P1-6: 出版停滞監視のタイムスタンプを再初期化
    if (runtimeOrchestrator_) {
        runtimeOrchestrator_->resetProgressObservation();
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
    crossfadeRuntime_.reset();
    crossfadeRuntime_.getGain().reset(safeSampleRate, 0.03);
    crossfadeRuntime_.getGain().setCurrentAndTargetValue(1.0);
    const convo::RuntimeReaderContext messageCtx{ messageThreadRcuReader, convo::ObserveChannel::Message };
    const auto runtimeReadHandle = makeRuntimeReadHandle(messageCtx);
    {
        auto* currentForPublish = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        auto* fadingForPublish = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        const bool hasAnyRuntime = (currentForPublish != nullptr) || (fadingForPublish != nullptr);
        if (hasAnyRuntime)
        {
            const auto policy = getTransitionPolicyFromRuntimeWorld(runtimeReadHandle, convo::TransitionPolicy::SmoothOnly);
            const auto fadeTimeSec = getOverlapFadeTimeFromRuntimeWorld(runtimeReadHandle, 0.0);
            const bool transitionActive = hasFadingRuntimeInWorld(runtimeReadHandle);

            // Migrated to publishWorld() with pre-built RuntimePublishWorld (Sprint-2 P1-A)
            auto coordinator = makeRuntimePublicationCoordinator();
            auto worldBuilder = convo::RuntimeBuilder(*this);
            worldBuilder.setHealthStateRef(getHealthStateRef());
            auto worldOwner = worldBuilder.buildRuntimePublishWorld(currentForPublish,
                                                                     fadingForPublish,
                                                                     policy,
                                                                     fadeTimeSec,
                                                                     transitionActive);
            coordinator.publishWorld(std::move(worldOwner));
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
    publishLatencyDelayAtomics(0, 0);
    convo::publishAtomic(latencyResetPending, false, std::memory_order_release);
    refreshCrossfadePreparedSnapshotFromAtomics();

    resetLatencyDelayRtState();

    // Runtime publication admission は lifecycle=Prepared を前提にしているため、
    // 初期 placeholder publish / startup rebuild intent の前に Prepared を公開する。
    // JUCE 契約上 prepareToPlay 実行中は Audio Thread callback が走らないため、
    // ここでの状態公開は安全。
    diagLog("[DIAG] prepareToPlay: latency buffers ready");
    convo::publishAtomic(lifecycleState, EngineLifecycleState::Prepared, std::memory_order_release);
    diagLog("[DIAG] prepareToPlay: lifecycleState set to Prepared");

    // 初回IRロード前でも currentDSP を常に有効にし、DSP->DSP クロスフェードへ統一する。
    const bool hasPublishedCurrent = (resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle) != nullptr);
    diagLog("[DIAG] prepareToPlay: hasPublishedCurrent=" + juce::String(static_cast<int>(hasPublishedCurrent)));
    if (!hasPublishedCurrent && !hasActiveRuntimeDSP())
    {
        diagLog("[DIAG] prepareToPlay: creating placeholderDSP");
        convo::aligned_unique_ptr<DSPCore> placeholderDSP;
        try
        {
            placeholderDSP = convo::aligned_make_unique<DSPCore>();
            diagLog("[DIAG] prepareToPlay: placeholderDSP created");
            placeholderDSP->convolverRt().setVisualizationEnabled(false);
            diagLog("[DIAG] prepareToPlay: calling placeholderDSP->prepare");
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

        setActiveRuntimeDSP(placeholderDSP.release());
        convo::publishAtomic(lastCommittedConvolverHasIr_, false, std::memory_order_release);
        convo::publishAtomic(lastCommittedConvolverStructuralHash_, 0, std::memory_order_release);

        // Migrated to publishWorld() with pre-built RuntimePublishWorld (Sprint-2 P1-A)
        {
            auto coordinator = makeRuntimePublicationCoordinator();
            auto worldBuilder = convo::RuntimeBuilder(*this);
            worldBuilder.setHealthStateRef(getHealthStateRef());
            auto worldOwner = worldBuilder.buildRuntimePublishWorld(getActiveRuntimeDSP(),
                                                                     nullptr,
                                                                     convo::TransitionPolicy::HardReset,
                                                                     0.0,
                                                                     false);
            coordinator.publishWorld(std::move(worldOwner));
        }
    }

    // --- DSP再ビルド判定・同期 ---
    uiConvolverProcessor.prepareToPlay(safeSampleRate, bufferSize);
    if (rateChanged)
        uiConvolverProcessor.invalidatePendingLoads();
    const bool hasCurrentRuntime = (resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle) != nullptr);
    if (rateChanged || blockSizeChanged || !hasCurrentRuntime) {
        if (juce::MessageManager::getInstance()->isThisTheMessageThread()) {
            submitRebuildIntent(convo::RebuildKind::Structural,
                                RebuildTelemetryReason::RequestRebuildKindEntry,
                                RebuildTelemetryClass::Structural,
                                RebuildTelemetryPolicy::Replaceable);
        } else {
            submitRebuildIntent(convo::RebuildKind::Structural,
                                RebuildTelemetryReason::PrepareToPlayNonMt,
                                RebuildTelemetryClass::Structural,
                                RebuildTelemetryPolicy::Replaceable);
        }
    }
    diagLog("[DIAG] prepareToPlay: exit currentSR=" + juce::String(convo::consumeAtomic(currentSampleRate, std::memory_order_acquire), 2) + " maxSPB=" + juce::String(convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire)));

    // P0-A0: LifecycleIsolationRuntime integration - leave prepare phase
    lifecycleRuntime_.leavePrepare(lifecycleToken);
}
