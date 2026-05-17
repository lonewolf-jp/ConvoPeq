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

void AudioEngine::requestRebuild(convo::RebuildKind kind) noexcept
{
    if (isShutdownInProgress())
        return;

    if (kind == convo::RebuildKind::None || kind == convo::RebuildKind::Runtime)
        return;

    // Message Thread かつ実行コンテキスト有効時は直接 rebuild 実行へ進む
    if (kind == convo::RebuildKind::Structural)
    {
        if (auto* mm = juce::MessageManager::getInstanceWithoutCreating();
            mm != nullptr && mm->isThisTheMessageThread())
        {
            clearRebuildReason(RebuildReason::StructuralFromNonMT);

            const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
            const int bs = convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire);
            if (sr > 0.0 && bs > 0)
            {
                requestRebuild(sr, bs);
                return;
            }

            setRebuildReason(RebuildReason::DeferredFinalizeAware);
            return;
        }
    }

    // 非MTパス: bool フラグをセットして MT へ通知
    const bool wasNewlyPending = setRebuildReason(RebuildReason::StructuralFromNonMT);
    if (wasNewlyPending)
        triggerAsyncUpdate();
}

void AudioEngine::handleAsyncUpdate()
{
    if (isShutdownInProgress())
        return;

    executeCommit();

    // 非MT起点の Structural rebuild 要求を消費して実行する
    if (clearRebuildReason(RebuildReason::StructuralFromNonMT))
    {
        const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
        const int bs = convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire);
        if (sr > 0.0 && bs > 0)
            requestRebuild(sr, bs);
        else
            setRebuildReason(RebuildReason::DeferredFinalizeAware);
    }
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_REBUILD_EXECUTE)

//--------------------------------------------------------------
// requestRebuild - DSPグラフの再構築 (Message Thread)
//--------------------------------------------------------------
void AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock)
{
    debugRebuildDispatchRequestCount.fetch_add(1, std::memory_order_acq_rel);

    // UIコンポーネントへのアクセスを行うため、必ずMessage Threadで実行すること
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    // シャットダウン中のリクエストは早期に切る
    if (isShutdownInProgress())
    {
        diagLog("[DIAG] requestRebuild(sr,bs): ignored during shutdown");
        return;
    }

    if (hasRebuildReason(RebuildReason::DeferredStructural))
    {
        const int64_t dueTicks = convo::consumeAtomic(deferredStructuralRebuildDueTicks_, std::memory_order_acquire);
        const int64_t nowTicks = juce::Time::getHighResolutionTicks();
        const bool uiHasIr = uiConvolverProcessor.isIRLoaded();
        const bool committedHasIr = convo::consumeAtomic(lastCommittedConvolverHasIr_, std::memory_order_acquire);

        if (dueTicks > 0 && nowTicks < dueTicks && uiHasIr && !committedHasIr)
        {
            diagLog("[DIAG] requestRebuild(sr,bs): SUPPRESSED direct rebuild during deferred Structural window SR="
                + juce::String(sampleRate, 2));
            return;
        }
    }

    const int publishedFftSize = uiConvolverProcessor.getActiveCacheFFTSize();
    const int targetFftSize = uiConvolverProcessor.getTargetUpgradeFFTSize();
    const bool suppressIntermediateMixedPhasePublish =
        uiConvolverProcessor.isProgressiveUpgradeEnabled()
        && uiConvolverProcessor.getPhaseMode() == ConvolverProcessor::PhaseMode::Mixed
        && publishedFftSize > 0
        && publishedFftSize < targetFftSize;

    if (suppressIntermediateMixedPhasePublish)
    {
        diagLog("[DIAG] requestRebuild(sr,bs): SUPPRESSED intermediate progressive mixed-phase publish fft="
                + juce::String(publishedFftSize)
                + " targetFFT=" + juce::String(targetFftSize)
                + " SR=" + juce::String(sampleRate, 2));
        return;
    }

    if (noiseShaperLearner && noiseShaperLearner->isRunning())
        noiseShaperLearner->stopLearning();

    // A-14: rebuild 開始時のパラメータを凍結し、
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

    DSPCore* currentToRelease = nullptr;
    bool queued = false;
    bool blockedAsDuplicate = false;
    bool blockedAsRecentDuplicate = false;
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        if (hasPendingTask)
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

        if (!blockedAsDuplicate)
        {
            BuildParameterSnapshot lastQueuedSnapshot {};
            lastQueuedSnapshot.ditherDepth = lastQueuedTaskSignature.buildInput.ditherBitDepth;
            lastQueuedSnapshot.oversamplingFactor = lastQueuedTaskSignature.buildInput.oversamplingFactor;
            lastQueuedSnapshot.oversamplingType = static_cast<OversamplingType>(lastQueuedTaskSignature.buildInput.oversamplingType);
            lastQueuedSnapshot.noiseShaperType = static_cast<NoiseShaperType>(lastQueuedTaskSignature.buildInput.noiseShaperType);

            const bool sameAsLastQueued =
                std::abs(lastQueuedTaskSignature.buildInput.sampleRate - sampleRate) <= 1.0e-6
                && lastQueuedTaskSignature.buildInput.blockSize == samplesPerBlock
                && equalsBuildParameterSnapshot(lastQueuedSnapshot, paramSnapshot)
                && lastQueuedTaskSignature.convolverBuildSnapshot.fingerprint == task.convolverBuildSnapshot.fingerprint;

            if (sameAsLastQueued)
            {
                const int64_t nowTicks = juce::Time::getHighResolutionTicks();
                const int64_t minDeltaTicks = juce::Time::getHighResolutionTicksPerSecond() / 5; // 200ms
                if (lastQueuedTaskTicks > 0 && (nowTicks - lastQueuedTaskTicks) < minDeltaTicks)
                    blockedAsRecentDuplicate = true;
            }
        }

        if (!blockedAsDuplicate && !blockedAsRecentDuplicate)
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
        debugRebuildDispatchQueuedCount.fetch_add(1, std::memory_order_acq_rel);
        rebuildCV.notify_all();
        diagLog("[DIAG] requestRebuild(sr,bs): task queued generation=" + juce::String(generation)
            + " SR=" + juce::String(sampleRate, 2));
    }
    else
    {
        if (blockedAsRecentDuplicate)
        {
            debugRebuildDispatchBlockedRecentDuplicateCount.fetch_add(1, std::memory_order_acq_rel);
            diagLog("[DIAG] requestRebuild(sr,bs): BLOCKED duplicate recent task SR="
                + juce::String(sampleRate, 2));
        }
        else
        {
            debugRebuildDispatchBlockedPendingDuplicateCount.fetch_add(1, std::memory_order_acq_rel);
            diagLog("[DIAG] requestRebuild(sr,bs): BLOCKED duplicate pending task SR="
                + juce::String(sampleRate, 2));
        }
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
                rebuildCV.wait(lock, [this] { return hasPendingTask || convo::consumeAtomic(rebuildThreadShouldExit); });

                if (convo::consumeAtomic(rebuildThreadShouldExit)) break;
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
                return isRebuildObsolete(task.generation) || convo::consumeAtomic(rebuildThreadShouldExit);
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
                    requestRebuild(convo::RebuildKind::Structural);

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
