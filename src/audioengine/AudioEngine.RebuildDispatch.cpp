#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"
#include "RuntimeBuilder.h"

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

struct BuildParameterSnapshot
{
    int ditherDepth = 0;
    int oversamplingFactor = 0;
    AudioEngine::OversamplingType oversamplingType = AudioEngine::OversamplingType::IIR;
    AudioEngine::NoiseShaperType noiseShaperType = AudioEngine::NoiseShaperType::Psychoacoustic;
};

static BuildParameterSnapshot captureBuildParameterSnapshot(const AudioEngine& engine) noexcept
{
    BuildParameterSnapshot snapshot {};
    snapshot.ditherDepth = convo::consumeAtomic(engine.ditherBitDepth, std::memory_order_acquire);
    snapshot.oversamplingFactor = convo::consumeAtomic(engine.manualOversamplingFactor, std::memory_order_acquire);
    snapshot.oversamplingType = convo::consumeAtomic(engine.oversamplingType, std::memory_order_acquire);
    snapshot.noiseShaperType = convo::consumeAtomic(engine.noiseShaperType, std::memory_order_acquire);
    return snapshot;
}

static bool equalsBuildParameterSnapshot(const BuildParameterSnapshot& lhs,
                                         const BuildParameterSnapshot& rhs) noexcept
{
    return lhs.ditherDepth == rhs.ditherDepth
        && lhs.oversamplingFactor == rhs.oversamplingFactor
        && lhs.oversamplingType == rhs.oversamplingType
        && lhs.noiseShaperType == rhs.noiseShaperType;
}

static bool shouldRetryWarmupFailure(const AudioEngine::DSPCore& dsp) noexcept
{
    return dsp.convolverRt().isLoadingIR();
}
}


#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_REBUILD_DISPATCH)

void AudioEngine::submitRebuildIntent(convo::RebuildKind kind,
                                      RebuildTelemetryReason reason,
                                      RebuildTelemetryClass rebuildClass,
                                      RebuildTelemetryPolicy collapsePolicy) noexcept
{
    constexpr const char* kPhase5TagReduce = "phase5_reduce_target";
    constexpr const char* kPhase5TagKeep = "phase5_keep_target";

    const int64_t nowTicks = juce::Time::getHighResolutionTicks();
    uint64_t structuralHash = 0;
    uint64_t fingerprint = 0;
    bool isMessageThread = false;
    if (auto* mm = juce::MessageManager::getInstanceWithoutCreating(); mm != nullptr)
        isMessageThread = mm->isThisTheMessageThread();

    if (kind == convo::RebuildKind::Structural && isMessageThread)
    {
        if (uiConvolverProcessor.isIRLoaded())
            structuralHash = uiConvolverProcessor.getStructuralHash();
        fingerprint = uiConvolverProcessor.captureBuildSnapshot().fingerprint;
    }

    const double srSnapshot = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
    const int bsSnapshot = convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire);
    const bool deferCategory = (kind == convo::RebuildKind::Structural) && !(srSnapshot > 0.0 && bsSnapshot > 0);

    bool sameAsPendingWouldMerge = false;
    bool shouldApplyLatestWinsMerge = false;
    int64_t latestWinsWindowTicks = 0;
    int latestWinsWindowMs = 0;

    if (collapsePolicy == RebuildTelemetryPolicy::Replaceable)
    {
        // UI burst 吸収専用。MustExecute には適用しない。
        latestWinsWindowMs = (kind == convo::RebuildKind::Structural)
            ? std::max(1, uiConvolverProcessor.getRebuildDebounceMs())
            : 50;
        latestWinsWindowTicks = (juce::Time::getHighResolutionTicksPerSecond() * latestWinsWindowMs) / 1000;
    }

    {
        std::lock_guard<std::mutex> lock(rebuildAdmissionIntentMutex_);
        const auto& pending = rebuildAdmissionPendingIntent_;
        sameAsPendingWouldMerge = pending.valid
            && pending.kind == kind
            && pending.rebuildClass == rebuildClass
            && pending.collapsePolicy == collapsePolicy
            && pending.structuralHash == structuralHash
            && pending.fingerprint == fingerprint
            && pending.deferCategory == deferCategory;

        if (sameAsPendingWouldMerge
            && collapsePolicy == RebuildTelemetryPolicy::Replaceable
            && pending.lastIntentTicks > 0
            && latestWinsWindowTicks > 0)
        {
            const int64_t elapsed = nowTicks - pending.lastIntentTicks;
            shouldApplyLatestWinsMerge = (elapsed >= 0 && elapsed <= latestWinsWindowTicks);
        }

        rebuildAdmissionPendingIntent_.valid = true;
        rebuildAdmissionPendingIntent_.kind = kind;
        rebuildAdmissionPendingIntent_.rebuildClass = rebuildClass;
        rebuildAdmissionPendingIntent_.collapsePolicy = collapsePolicy;
        rebuildAdmissionPendingIntent_.structuralHash = structuralHash;
        rebuildAdmissionPendingIntent_.fingerprint = fingerprint;
        rebuildAdmissionPendingIntent_.deferCategory = deferCategory;
        rebuildAdmissionPendingIntent_.lastIntentTicks = nowTicks;
    }

    const uint64_t intentId = nextRebuildTelemetryIntentId();
    emitRebuildTelemetry(RebuildTelemetryEvent::Requested,
                         intentId,
                         reason,
                         RebuildTelemetryDecision::Accepted,
                         structuralHash,
                         fingerprint,
                         rebuildClass,
                         collapsePolicy);

    if (sameAsPendingWouldMerge)
    {
        emitRebuildTelemetry(RebuildTelemetryEvent::Merged,
                             intentId,
                             RebuildTelemetryReason::SameAsPendingWouldMerge,
                             RebuildTelemetryDecision::Merged,
                             structuralHash,
                             fingerprint,
                             rebuildClass,
                             collapsePolicy,
                             deferCategory ? "defer_candidate" : "dispatch_candidate");

        if (shouldApplyLatestWinsMerge)
        {
            emitRebuildTelemetry(RebuildTelemetryEvent::Merged,
                                 intentId,
                                 RebuildTelemetryReason::SameAsPendingWouldMerge,
                                 RebuildTelemetryDecision::Merged,
                                 structuralHash,
                                 fingerprint,
                                 rebuildClass,
                                 collapsePolicy,
                                 kPhase5TagReduce,
                                 static_cast<double>(latestWinsWindowMs));
            return;
        }
    }

    if (isShutdownInProgress())
    {
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                     intentId,
                     RebuildTelemetryReason::ShutdownInProgress,
                     RebuildTelemetryDecision::Suppressed,
                     0,
                     0,
                     RebuildTelemetryClass::Structural,
                     collapsePolicy,
                     kPhase5TagKeep);
        return;
    }

    if (kind == convo::RebuildKind::None || kind == convo::RebuildKind::Runtime)
    {
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                     intentId,
                     RebuildTelemetryReason::KindFiltered,
                     RebuildTelemetryDecision::Suppressed,
                     0,
                     0,
                     rebuildClass,
                     collapsePolicy,
                     kPhase5TagKeep);
        return;
    }

    // Message Thread かつ実行コンテキスト有効時は直接 rebuild 実行へ進む
    if (kind == convo::RebuildKind::Structural)
    {
        if (isMessageThread)
        {
            clearRebuildReason(RebuildReason::StructuralFromNonMT);

            const double sr = srSnapshot;
            const int bs = bsSnapshot;
            if (sr > 0.0 && bs > 0)
            {
                emitRebuildTelemetry(RebuildTelemetryEvent::Dispatched,
                                     intentId,
                                     RebuildTelemetryReason::DelegateRequestRebuildSrBs,
                                     RebuildTelemetryDecision::Dispatched,
                                     structuralHash,
                                     fingerprint,
                                     RebuildTelemetryClass::Structural,
                                     collapsePolicy);
                requestRebuild(sr, bs, collapsePolicy == RebuildTelemetryPolicy::MustExecute);
                return;
            }

            setRebuildReason(RebuildReason::DeferredFinalizeAware);
            emitRebuildTelemetry(RebuildTelemetryEvent::Deferred,
                                 intentId,
                                 RebuildTelemetryReason::MissingSrBs,
                                 RebuildTelemetryDecision::Deferred,
                                 0,
                                 0,
                                 RebuildTelemetryClass::FinalizeAware,
                                 collapsePolicy);
            return;
        }
    }

    // 非MTパス: bool フラグをセットして MT へ通知
    const bool wasNewlyPending = setRebuildReason(RebuildReason::StructuralFromNonMT);
    if (wasNewlyPending)
    {
        emitRebuildTelemetry(RebuildTelemetryEvent::Dispatched,
                             intentId,
                             RebuildTelemetryReason::NonMtTriggerAsync,
                             RebuildTelemetryDecision::Dispatched,
                             0,
                             0,
                             RebuildTelemetryClass::Structural,
                             collapsePolicy);
        triggerAsyncUpdate();
    }
    else
    {
        emitRebuildTelemetry(RebuildTelemetryEvent::Merged,
                             intentId,
                             RebuildTelemetryReason::NonMtAlreadyPending,
                             RebuildTelemetryDecision::Merged,
                             0,
                             0,
                             RebuildTelemetryClass::Structural,
                             collapsePolicy);
    }
}

void AudioEngine::handleAsyncUpdate()
{
    if (isShutdownInProgress())
        return;

    executeCommit();

    // 非MT起点の Structural rebuild 要求を消費して実行する
    if (clearRebuildReason(RebuildReason::StructuralFromNonMT))
    {
        RebuildTelemetryPolicy collapsePolicy = RebuildTelemetryPolicy::NA;
        {
            std::lock_guard<std::mutex> lock(rebuildAdmissionIntentMutex_);
            const auto& pending = rebuildAdmissionPendingIntent_;
            if (pending.valid && pending.kind == convo::RebuildKind::Structural)
                collapsePolicy = pending.collapsePolicy;
        }

        const uint64_t intentId = nextRebuildTelemetryIntentId();
        emitRebuildTelemetry(RebuildTelemetryEvent::Requested,
                     intentId,
                     RebuildTelemetryReason::AsyncBridgeConsume,
                     RebuildTelemetryDecision::Accepted,
                     0,
                     0,
                     RebuildTelemetryClass::Structural,
                     collapsePolicy);

        const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
        const int bs = convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire);
        if (sr > 0.0 && bs > 0)
        {
            emitRebuildTelemetry(RebuildTelemetryEvent::Dispatched,
                                 intentId,
                                 RebuildTelemetryReason::AsyncBridgeDelegateSrBs,
                                 RebuildTelemetryDecision::Dispatched,
                                 0,
                                 0,
                                 RebuildTelemetryClass::Structural,
                                 collapsePolicy);
            requestRebuild(sr, bs, collapsePolicy == RebuildTelemetryPolicy::MustExecute);
        }
        else
        {
            setRebuildReason(RebuildReason::DeferredFinalizeAware);
            emitRebuildTelemetry(RebuildTelemetryEvent::Deferred,
                                 intentId,
                                 RebuildTelemetryReason::AsyncBridgeMissingSrBs,
                                 RebuildTelemetryDecision::Deferred,
                                 0,
                                 0,
                                 RebuildTelemetryClass::FinalizeAware,
                                 collapsePolicy);
        }
    }
}

void AudioEngine::requestRebuild(convo::RebuildKind kind) noexcept
{
    submitRebuildIntent(kind,
                        RebuildTelemetryReason::RequestRebuildKindEntry,
                        RebuildTelemetryClass::Structural,
                        RebuildTelemetryPolicy::Replaceable);
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_REBUILD_EXECUTE)

//--------------------------------------------------------------
// requestRebuild - DSPグラフの再構築 (Message Thread)
//--------------------------------------------------------------
void AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock, bool bypassLegacySuppression)
{
    constexpr const char* kPhase5TagReduce = "phase5_reduce_target";
    constexpr const char* kPhase5TagKeep = "phase5_keep_target";

    const auto collapsePolicy = bypassLegacySuppression
        ? RebuildTelemetryPolicy::MustExecute
        : RebuildTelemetryPolicy::Replaceable;

    const uint64_t intentId = nextRebuildTelemetryIntentId();
    const double requestStartMs = juce::Time::getMillisecondCounterHiRes();
    emitRebuildTelemetry(RebuildTelemetryEvent::Requested,
                         intentId,
                         RebuildTelemetryReason::RequestRebuildSrBs,
                         RebuildTelemetryDecision::Accepted,
                         0,
                         0,
                         RebuildTelemetryClass::Structural,
                         collapsePolicy);

    convo::fetchAddAtomic(debugRebuildDispatchRequestCount, 1, std::memory_order_acq_rel);

    // UIコンポーネントへのアクセスを行うため、必ずMessage Threadで実行すること
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    // シャットダウン中のリクエストは早期に切る
    if (isShutdownInProgress())
    {
        // Phase5: 維持対象（グローバル安全ガード）
        diagLog("[DIAG][PHASE5-KEEP] requestRebuild(sr,bs): ignored during shutdown");
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                     intentId,
                     RebuildTelemetryReason::ShutdownInProgress,
                     RebuildTelemetryDecision::Suppressed,
                     0,
                     0,
                     RebuildTelemetryClass::Structural,
                     collapsePolicy,
                     kPhase5TagKeep);
        return;
    }

    // Phase5: 縮退対象（legacy deferred window suppress）は削除済み。
    // DeferredStructural が立っていても、ここでは suppress せず通常の queue 判定へ進める。

    const int publishedFftSize = uiConvolverProcessor.getActiveCacheFFTSize();
    const int targetFftSize = uiConvolverProcessor.getTargetUpgradeFFTSize();
    const bool suppressIntermediateMixedPhasePublish =
        uiConvolverProcessor.isProgressiveUpgradeEnabled()
        && uiConvolverProcessor.getPhaseMode() == ConvolverProcessor::PhaseMode::Mixed
        && publishedFftSize > 0
        && publishedFftSize < targetFftSize;

    if (suppressIntermediateMixedPhasePublish)
    {
        // Phase5: 維持対象（中間状態 publish 防止の安全ガード）
        diagLog("[DIAG][PHASE5-KEEP] requestRebuild(sr,bs): SUPPRESSED intermediate progressive mixed-phase publish fft="
                + juce::String(publishedFftSize)
                + " targetFFT=" + juce::String(targetFftSize)
                + " SR=" + juce::String(sampleRate, 2));
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                     intentId,
                     RebuildTelemetryReason::MixedPhaseIntermediate,
                     RebuildTelemetryDecision::Suppressed,
                     0,
                     0,
                     RebuildTelemetryClass::Structural,
                 collapsePolicy,
                 kPhase5TagKeep);
        return;
    }

    if (noiseShaperLearner && noiseShaperLearner->isRunning())
        noiseShaperLearner->stopLearning();

    // rebuild 開始時のパラメータを凍結し、
    // task 作成・重複判定・runtime command で同一 snapshot を使う。
    const BuildParameterSnapshot paramSnapshot = captureBuildParameterSnapshot(*this);
    DSPCore* current = activeDSP; // 現在のアクティブDSPをキャプチャ
    int generation = 0;

    RebuildTask task;
    task.currentDSP = current;
    task.buildInput.sampleRate = sampleRate;
    task.buildInput.blockSize = samplesPerBlock;
    task.buildInput.ditherBitDepth = paramSnapshot.ditherDepth;
    task.buildInput.oversamplingFactor = paramSnapshot.oversamplingFactor;
    task.buildInput.oversamplingType = static_cast<int>(paramSnapshot.oversamplingType);
    task.buildInput.noiseShaperType = static_cast<int>(paramSnapshot.noiseShaperType);
    task.convolverBuildSnapshot = uiConvolverProcessor.captureBuildSnapshot();
    const uint64_t structuralHash = uiConvolverProcessor.isIRLoaded() ? uiConvolverProcessor.getStructuralHash() : 0;

    DSPCore* currentToRelease = nullptr;
    bool queued = false;
    bool blockedAsDuplicate = false;
    const bool allowDuplicateSuppression = !bypassLegacySuppression;
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        if (hasPendingTask)
        {
            if (allowDuplicateSuppression)
            {
                BuildParameterSnapshot pendingSnapshot {};
                pendingSnapshot.ditherDepth = pendingTask.buildInput.ditherBitDepth;
                pendingSnapshot.oversamplingFactor = pendingTask.buildInput.oversamplingFactor;
                pendingSnapshot.oversamplingType = static_cast<OversamplingType>(pendingTask.buildInput.oversamplingType);
                pendingSnapshot.noiseShaperType = static_cast<NoiseShaperType>(pendingTask.buildInput.noiseShaperType);

                const bool sameAsPending =
                    std::abs(pendingTask.buildInput.sampleRate - sampleRate) <= 1.0e-6
                    && pendingTask.buildInput.blockSize == samplesPerBlock
                    && equalsBuildParameterSnapshot(pendingSnapshot, paramSnapshot)
                    && pendingTask.convolverBuildSnapshot.fingerprint == task.convolverBuildSnapshot.fingerprint;

                if (sameAsPending)
                {
                    blockedAsDuplicate = true;
                }
                else
                {
                    currentToRelease = pendingTask.currentDSP;
                }
            }
            else
            {
                currentToRelease = pendingTask.currentDSP;
            }
        }

        if (!blockedAsDuplicate)
        {
            generation = ++rebuildGeneration;
            task.generation = generation;
            pendingTask = task;
            hasPendingTask = true;
            lastQueuedTaskSignature = task;
            lastQueuedTaskTicks = juce::Time::getHighResolutionTicks();
            queued = true;
        }
    }

    if (queued)
    {
        convo::fetchAddAtomic(debugRebuildDispatchQueuedCount, 1, std::memory_order_acq_rel);
        rebuildCV.notify_all();
        diagLog("[DIAG] requestRebuild(sr,bs): task queued generation=" + juce::String(generation)
            + " SR=" + juce::String(sampleRate, 2));
        const double latencyMs = juce::Time::getMillisecondCounterHiRes() - requestStartMs;
        emitRebuildTelemetry(RebuildTelemetryEvent::Dispatched,
                     intentId,
                     RebuildTelemetryReason::TaskQueued,
                     RebuildTelemetryDecision::Dispatched,
                             structuralHash,
                             task.convolverBuildSnapshot.fingerprint,
                     RebuildTelemetryClass::Structural,
                 collapsePolicy,
                     "N/A",
                     latencyMs);
    }
    else
    {
        // Phase5: PendingDuplicate は維持対象（pending queue の過負荷抑止ガード）。
        // 直近縮退（RecentDuplicate / DeferredStructuralWindow）後も、
        // 同一 pending の無限置換を防ぐため merge 扱いを維持する。
        convo::fetchAddAtomic(debugRebuildDispatchBlockedPendingDuplicateCount, 1, std::memory_order_acq_rel);
        diagLog("[DIAG][PHASE5-KEEP] requestRebuild(sr,bs): BLOCKED duplicate pending task SR="
            + juce::String(sampleRate, 2));
        emitRebuildTelemetry(RebuildTelemetryEvent::Merged,
                             intentId,
                             RebuildTelemetryReason::PendingDuplicate,
                             RebuildTelemetryDecision::Merged,
                             structuralHash,
                             task.convolverBuildSnapshot.fingerprint,
                             RebuildTelemetryClass::Structural,
                             collapsePolicy,
                             kPhase5TagKeep);
    }

    // Destroy orphaned DSP objects outside the lock.
    if (currentToRelease)
        retireDSP(currentToRelease);

    if (!queued)
    {
        if (current)
        {
            // EBR: lifetime managed by RCUReader
        }
    }
}
#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_REBUILD_EXECUTE)


void AudioEngine::stopRebuildThread()
{
    setShutdownPhase(ShutdownPhase::StopWorkers, "stopRebuildThread");

    // exit フラグを立てる（predicate が次に評価された時に break する）
    convo::publishAtomic(rebuildThreadShouldExit, true, std::memory_order_release);

    // 待機中のスレッドを確実に起こす
    rebuildCV.notify_all();

    if (rebuildThread.joinable())
        rebuildThread.join();

}
#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_REBUILD_EXECUTE)


void AudioEngine::rebuildThreadLoop()
{
    affinityManager.applyCurrentThreadPolicy(ThreadType::HeavyBackground);

    // Set denormal handling modes for this thread. This is crucial for performance
    // in MKL VML and AVX/SSE operations, which can be significantly slowed down
    // by subnormal numbers. This setting is thread-local.
    vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    convo::publishAtomic(rebuildThreadIsRunning, true, std::memory_order_release);

    while (true)
    {
        try
        {
            RebuildTask task;
            {
                std::unique_lock<std::mutex> lock(rebuildMutex);
                rebuildCV.wait(lock, [this] { return hasPendingTask || convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire); });

                if (convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire)) break;
                if (isShutdownInProgress())
                {
                    hasPendingTask = false;
                    pendingTask.currentDSP = nullptr;
                    break;
                }

                // Copy task and clear pendingTask pointers to transfer ownership
                task = pendingTask;
                pendingTask.currentDSP = nullptr;

                hasPendingTask = false;
            }

            struct DSPGuard
            {
                AudioEngine* owner;
                DSPCore* ptr;
                ~DSPGuard()
                {
                    if (owner != nullptr && ptr != nullptr)
                        owner->retireDSP(ptr);
                }
            } dspGuard { this, nullptr };

            convo::RuntimeBuilder runtimeBuilder(*this);

            // Helper to check obsolescence
            const auto isObsolete = [&] {
                return isRebuildObsolete(task.generation) || convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire);
            };

            if (isObsolete())
                continue;

            // 1. Prepare (メモリ確保)
            convo::BuildResult buildResult = runtimeBuilder.build(task.buildInput, task.convolverBuildSnapshot);

            if (buildResult.runtime == nullptr)
            {
                diagLog("[DIAG] rebuildThreadLoop: RuntimeBuilder build failed generation="
                        + juce::String(task.generation)
                        + " error=" + juce::String(convo::toString(buildResult.error))
                        + " source=task-snapshot");
                continue;
            }

            dspGuard.ptr = buildResult.runtime;
            auto* newDSP = buildResult.runtime;

            if (isObsolete())
                continue;

            // 2. Reuse Logic
            //
            // 【task.currentDSP (= 旧 activeDSP) の安全性証明】
            //
            // task.currentDSP は RCU 設計により、退役後も一定期間（Epoch）生存が
            // 保証される。rebuild タスクはこの期間内に完了することを前提としている。
            // (通常、数ミリ秒〜数十ミリ秒。EBR Queue の遅延削除により安全性確保)
            //
            // 【shared_ptr 不採用の理由】
            //     Audio Thread は std::shared_ptr の参照カウント操作を禁止している
            //     (コーディング規約)。currentDSP の RCU 設計はこの制約に基づく。
            bool irReused = false;
            if (task.currentDSP)
            {

                if (std::abs(task.currentDSP->sampleRate - task.buildInput.sampleRate) < 1e-6 &&
                    task.currentDSP->oversamplingFactor == newDSP->oversamplingFactor &&
                    task.currentDSP->convolverRt().getCurrentBufferSize() == newDSP->convolverRt().getCurrentBufferSize())
                {
                    // IRの生成条件が一致しているか確認
                    if (newDSP->convolverRt().getIRName() == task.currentDSP->convolverRt().getIRName() &&
                        newDSP->convolverRt().getPhaseMode() == task.currentDSP->convolverRt().getPhaseMode() &&
                        std::abs(newDSP->convolverRt().getMixedTransitionStartHz() - task.currentDSP->convolverRt().getMixedTransitionStartHz()) < 0.001f &&
                        std::abs(newDSP->convolverRt().getMixedTransitionEndHz() - task.currentDSP->convolverRt().getMixedTransitionEndHz()) < 0.001f &&
                        std::abs(newDSP->convolverRt().getMixedPreRingTau() - task.currentDSP->convolverRt().getMixedPreRingTau()) < 0.001f &&
                        newDSP->convolverRt().getExperimentalDirectHeadEnabled() == task.currentDSP->convolverRt().getExperimentalDirectHeadEnabled() &&
                        std::abs(newDSP->convolverRt().getTargetIRLength() - task.currentDSP->convolverRt().getTargetIRLength()) < 0.001f)
                    {
                        // 既存のConvolutionエンジンを共有（クローン回避・グリッチ防止）
                        newDSP->convolverRt().shareConvolutionEngineFrom(task.currentDSP->convolverRt());
                        irReused = true;
                    }
                }
            }

            // 3. Rebuild IR if needed (Heavy operation)
            if (!irReused && newDSP->convolverRt().getIRLength() > 0)
            {
                if (isObsolete())
                    continue;
                newDSP->convolverRt().rebuildAllIRsSynchronous(isObsolete);
            }

            const auto warmupError = runtimeBuilder.validateWarmup(*newDSP);
            if (warmupError != convo::BuildError::None)
            {
                const bool retryable = shouldRetryWarmupFailure(*newDSP);
                diagLog("[DIAG] rebuildThreadLoop: RuntimeBuilder warmup failed generation="
                    + juce::String(task.generation)
                    + " error=" + juce::String(convo::toString(warmupError))
                    + " retryable=" + juce::String(static_cast<int>(retryable))
                    + " irLoaded=" + juce::String(static_cast<int>(newDSP->convolverRt().isIRLoaded()))
                    + " irFinalized=" + juce::String(static_cast<int>(newDSP->convolverRt().isIRFinalized()))
                    + " irLoading=" + juce::String(static_cast<int>(newDSP->convolverRt().isLoadingIR())));

                if (retryable)
                    submitRebuildIntent(convo::RebuildKind::Structural,
                                        RebuildTelemetryReason::RebuildThreadWarmupRetry,
                                        RebuildTelemetryClass::Structural,
                                        RebuildTelemetryPolicy::Replaceable);

                continue;
            }

            if (isObsolete())
                continue;

            // 4. Refresh Latency (Prevent pitch slide during fade-in)
            newDSP->convolverRt().refreshLatency();

            // 5. Fade In
            newDSP->ramps().fadeInSamplesLeft = DSPCore::FADE_IN_SAMPLES;

            // 6. Commit on Message Thread
            // Release ownership from guard, pass to commitNewDSP
            DSPCore* dspToCommit = dspGuard.ptr;
            dspGuard.ptr = nullptr;
            prepareCommit(dspToCommit, task.generation);
        }
        catch (const std::exception& e)
        {
            DBG("AudioEngine::rebuildThreadLoop exception: " << e.what());
            juce::ignoreUnused(e);
        }
        catch (...)
        {
            DBG("AudioEngine::rebuildThreadLoop unknown exception");
        }
    }

    convo::publishAtomic(rebuildThreadIsRunning, false, std::memory_order_release);
}
#endif
