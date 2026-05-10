#include <JuceHeader.h>
#include "AudioEngine.h"

extern std::atomic<bool> gShuttingDown;

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

static void retireDSP(AudioEngine::DSPCore* dsp)
{
    if (dsp) convo::retireObject(dsp, [](void* p) { delete static_cast<AudioEngine::DSPCore*>(p); });
}
}


#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_REBUILD_DISPATCH)

// =============================================================
// Rebuild request coalescing (Stage 3)
// =============================================================
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

void AudioEngine::processRebuildRequestsInternal()
{
    // 1. mask 取得（完全 drain）
    const uint32_t mask = pendingRebuildMask_.exchange(0, std::memory_order_acq_rel);
    if (mask == 0)
        return;

    // 2. 現在の DSP パラメータ取得
    const double sr = currentSampleRate.load(std::memory_order_acquire);
    const int bs = maxSamplesPerBlock.load(std::memory_order_acquire);

    // 3. 無効状態 → 再投入（重要）
    if (sr <= 0.0 || bs <= 0)
    {
        pendingRebuildMask_.fetch_or(mask, std::memory_order_release);
        return;
    }

    // 4. 優先度制御
    // =============================

    // --- HIGH: Structural ---
    if (mask & static_cast<uint32_t>(convo::RebuildKind::Structural))
    {
        requestRebuild(sr, bs);
        return; // 他は defer
    }

    // --- MID: IRContent ---
    if (mask & static_cast<uint32_t>(convo::RebuildKind::IRContent))
    {
        requestRebuild(sr, bs);
        return;
    }

    // --- LOW: Runtime / UIOnly ---
    // 現状は何もしない（将来拡張ポイント）
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_REBUILD_EXECUTE)

//--------------------------------------------------------------
// requestRebuild - DSPグラフの再構築 (Message Thread)
//--------------------------------------------------------------
void AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock)
{
    // UIコンポーネント(uiEqEditor等)へのアクセスやMKLメモリ確保を行うため、必ずMessage Threadで実行すること
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    if (deferredStructuralRebuildPending_.load(std::memory_order_acquire))
    {
        const int64_t dueTicks = deferredStructuralRebuildDueTicks_.load(std::memory_order_acquire);
        const int64_t nowTicks = juce::Time::getHighResolutionTicks();
        const bool uiHasIr = uiConvolverProcessor.isIRLoaded();
        const bool activeHasIr = (activeDSP != nullptr) && activeDSP->convolver.isIRLoaded();

        if (dueTicks > 0 && nowTicks < dueTicks && uiHasIr && !activeHasIr)
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

    // 新しいDSPコアを作成
    DSPCore* newDSP = new DSPCore();
    newDSP->convolver.setVisualizationEnabled(false); // DSP用は可視化データ不要

    // UIプロセッサから状態をコピー
    // EQ状態は Snapshot 経由で反映するため、ここでの直接同期は行わない。
    newDSP->convolver.syncStateFrom(uiConvolverProcessor);

    // キャプチャ用変数
    int ditherDepth = ditherBitDepth.load();
    int osFactor = manualOversamplingFactor.load();
    OversamplingType osType = oversamplingType.load();
    NoiseShaperType nsType = noiseShaperType.load();
    DSPCore* current = activeDSP; // 現在のアクティブDSPをキャプチャ
    int generation = 0;

    RebuildTask task;
    task.newDSP = newDSP;
    task.currentDSP = current;
    task.sampleRate = sampleRate;
    task.samplesPerBlock = samplesPerBlock;
    task.ditherDepth = ditherDepth;
    task.manualOversamplingFactor = osFactor;
    task.oversamplingType = osType;
    task.noiseShaperType = nsType;

        DSPCore* dspToDestroy = nullptr; // To be destroyed outside the lock
    DSPCore* currentToRelease = nullptr;
    bool queued = false;
    bool blockedAsDuplicate = false;
    bool blockedAsRecentDuplicate = false;
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        if (hasPendingTask)
        {
            const bool sameAsPending =
                std::abs(pendingTask.sampleRate - sampleRate) <= 1.0e-6
                && pendingTask.samplesPerBlock == samplesPerBlock
                && pendingTask.ditherDepth == ditherDepth
                && pendingTask.manualOversamplingFactor == osFactor
                && pendingTask.oversamplingType == osType
                && pendingTask.noiseShaperType == nsType;

            if (sameAsPending)
            {
                blockedAsDuplicate = true;
            }
            else
            {
                dspToDestroy = pendingTask.newDSP;
                currentToRelease = pendingTask.currentDSP;
            }
        }

        if (!blockedAsDuplicate)
        {
            const bool sameAsLastQueued =
                std::abs(lastQueuedTaskSignature.sampleRate - sampleRate) <= 1.0e-6
                && lastQueuedTaskSignature.samplesPerBlock == samplesPerBlock
                && lastQueuedTaskSignature.ditherDepth == ditherDepth
                && lastQueuedTaskSignature.manualOversamplingFactor == osFactor
                && lastQueuedTaskSignature.oversamplingType == osType
                && lastQueuedTaskSignature.noiseShaperType == nsType;

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
        rebuildCV.notify_all();
        diagLog("[DIAG] requestRebuild(sr,bs): task queued generation=" + juce::String(generation)
            + " SR=" + juce::String(sampleRate, 2));
    }
    else
    {
        if (blockedAsRecentDuplicate)
        {
            diagLog("[DIAG] requestRebuild(sr,bs): BLOCKED duplicate recent task SR="
                + juce::String(sampleRate, 2));
        }
        else
        {
            diagLog("[DIAG] requestRebuild(sr,bs): BLOCKED duplicate pending task SR="
                + juce::String(sampleRate, 2));
        }
    }

    // Destroy orphaned DSP objects outside the lock.
    if (dspToDestroy)
        retireDSP(dspToDestroy);
    if (currentToRelease)
        retireDSP(currentToRelease);

    if (!queued)
    {
        retireDSP(newDSP);
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
    // exit フラグを立てる（predicate が次に評価された時に break する）
    rebuildThreadShouldExit.store(true, std::memory_order_release);

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

    rebuildThreadIsRunning.store(true, std::memory_order_release);

    while (true)
    {
        try
        {
            RebuildTask task;
            {
                std::unique_lock<std::mutex> lock(rebuildMutex);
                rebuildCV.wait(lock, [this] { return hasPendingTask || rebuildThreadShouldExit.load(); });

                if (rebuildThreadShouldExit.load()) break;

                // Copy task and clear pendingTask pointers to transfer ownership
                task = pendingTask;
                pendingTask.newDSP = nullptr;
                pendingTask.currentDSP = nullptr;

                hasPendingTask = false;
            }

            if (task.newDSP == nullptr)
            {
                jassertfalse;
                continue;
            }

            struct DSPGuard
            {
                DSPCore* ptr;
                ~DSPGuard()
                {
                    if (ptr != nullptr)
                        retireDSP(ptr);
                }
            } dspGuard { task.newDSP };

            // Helper to check obsolescence
            const auto isObsolete = [&] {
                return isRebuildObsolete(task.generation) || rebuildThreadShouldExit.load();
            };

            if (isObsolete())
                continue;

            // 1. Prepare (メモリ確保)
            task.newDSP->prepare(task.sampleRate, task.samplesPerBlock, task.ditherDepth, task.manualOversamplingFactor, task.oversamplingType, task.noiseShaperType, this);

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

                if (std::abs(task.currentDSP->sampleRate - task.sampleRate) < 1e-6 &&
                    task.currentDSP->oversamplingFactor == task.newDSP->oversamplingFactor &&
                    task.currentDSP->convolver.getCurrentBufferSize() == task.newDSP->convolver.getCurrentBufferSize())
                {
                    // IRの生成条件が一致しているか確認
                    if (task.newDSP->convolver.getIRName() == task.currentDSP->convolver.getIRName() &&
                        task.newDSP->convolver.getPhaseMode() == task.currentDSP->convolver.getPhaseMode() &&
                        std::abs(task.newDSP->convolver.getMixedTransitionStartHz() - task.currentDSP->convolver.getMixedTransitionStartHz()) < 0.001f &&
                        std::abs(task.newDSP->convolver.getMixedTransitionEndHz() - task.currentDSP->convolver.getMixedTransitionEndHz()) < 0.001f &&
                        std::abs(task.newDSP->convolver.getMixedPreRingTau() - task.currentDSP->convolver.getMixedPreRingTau()) < 0.001f &&
                        task.newDSP->convolver.getExperimentalDirectHeadEnabled() == task.currentDSP->convolver.getExperimentalDirectHeadEnabled() &&
                        std::abs(task.newDSP->convolver.getTargetIRLength() - task.currentDSP->convolver.getTargetIRLength()) < 0.001f)
                    {
                        // 既存のConvolutionエンジンを共有（クローン回避・グリッチ防止）
                        task.newDSP->convolver.shareConvolutionEngineFrom(task.currentDSP->convolver);
                        irReused = true;
                    }
                }
            }

            // 3. Rebuild IR if needed (Heavy operation)
            if (!irReused && task.newDSP->convolver.getIRLength() > 0)
            {
                if (isObsolete())
                    continue;
                task.newDSP->convolver.rebuildAllIRsSynchronous(isObsolete);
            }

            if (isObsolete())
                continue;

            // 4. Refresh Latency (Prevent pitch slide during fade-in)
            task.newDSP->convolver.refreshLatency();

            // 5. Fade In
            task.newDSP->fadeInSamplesLeft.store(DSPCore::FADE_IN_SAMPLES, std::memory_order_relaxed);

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

    rebuildThreadIsRunning.store(false, std::memory_order_release);
}
#endif
