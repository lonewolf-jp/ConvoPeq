#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"
#include "RuntimeBuilder.h"

extern std::atomic<bool> gShuttingDown;

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

static std::uint64_t makeRuntimePayloadHash(int ditherDepth,
                                            int oversamplingFactor,
                                            AudioEngine::OversamplingType oversamplingType,
                                            AudioEngine::NoiseShaperType noiseShaperType) noexcept
{
    std::uint64_t value = static_cast<std::uint64_t>(static_cast<std::uint32_t>(ditherDepth));
    value = (value << 8) ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(oversamplingFactor));
    value = (value << 8) ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(oversamplingType));
    value = (value << 8) ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(noiseShaperType));
    value ^= (value << 13);
    value ^= (value >> 7);
    value ^= (value << 17);
    return value;
}

static convo::EngineCommand makeRuntimeCommand(double sampleRate,
                                               int samplesPerBlock,
                                               int ditherDepth,
                                               int oversamplingFactor,
                                               AudioEngine::OversamplingType oversamplingType,
                                               AudioEngine::NoiseShaperType noiseShaperType,
                                               std::uint64_t revision) noexcept
{
    convo::EngineCommand cmd {};
    cmd.type = convo::CommandType::UpdateParameters;
    cmd.meta.revision = revision;
    cmd.meta.issuedTick = static_cast<std::uint64_t>(juce::Time::getHighResolutionTicks());
    cmd.meta.highPriority = true;
    cmd.sampleRate = sampleRate;
    cmd.blockSize = samplesPerBlock;
    cmd.intValue = ditherDepth;
    cmd.oversamplingFactor = oversamplingFactor;
    cmd.oversamplingType = static_cast<int>(oversamplingType);
    cmd.noiseShaperType = static_cast<int>(noiseShaperType);
    cmd.payloadHash = makeRuntimePayloadHash(ditherDepth,
                                             oversamplingFactor,
                                             oversamplingType,
                                             noiseShaperType);
    return cmd;
}

static bool shouldRetryWarmupFailure(const AudioEngine::DSPCore& dsp) noexcept
{
    return dsp.convolver.isLoadingIR();
}
}


#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_REBUILD_DISPATCH)

void AudioEngine::requestRebuild(convo::RebuildKind kind) noexcept
{
    if (shutdownInProgress.load(std::memory_order_acquire) ||
        gShuttingDown.load(std::memory_order_acquire))
        return;

    if (kind == convo::RebuildKind::None || kind == convo::RebuildKind::Runtime)
        return;

    // Message Thread かつ実行コンテキスト有効時は直接 rebuild 実行へ進む
    if (kind == convo::RebuildKind::Structural)
    {
        if (auto* mm = juce::MessageManager::getInstanceWithoutCreating();
            mm != nullptr && mm->isThisTheMessageThread())
        {
            pendingStructuralRebuildFromNonMT_.store(false, std::memory_order_release);

            const double sr = currentSampleRate.load(std::memory_order_acquire);
            const int bs = maxSamplesPerBlock.load(std::memory_order_acquire);
            if (sr > 0.0 && bs > 0)
            {
                requestRebuild(sr, bs);
                return;
            }

            deferredFinalizeAwareRebuildPending_.store(true, std::memory_order_release);
            return;
        }
    }

    // 非MTパス: bool フラグをセットして MT へ通知
    const bool wasAlreadyPending = pendingStructuralRebuildFromNonMT_.exchange(true, std::memory_order_acq_rel);
    if (!wasAlreadyPending)
        triggerAsyncUpdate();
}

void AudioEngine::handleAsyncUpdate()
{
    if (shutdownInProgress.load(std::memory_order_acquire) ||
        gShuttingDown.load(std::memory_order_acquire))
        return;

    executeCommit();

    // 非MT起点の Structural rebuild 要求を消費して実行する
    if (pendingStructuralRebuildFromNonMT_.exchange(false, std::memory_order_acq_rel))
    {
        const double sr = currentSampleRate.load(std::memory_order_acquire);
        const int bs = maxSamplesPerBlock.load(std::memory_order_acquire);
        if (sr > 0.0 && bs > 0)
            requestRebuild(sr, bs);
        else
            deferredFinalizeAwareRebuildPending_.store(true, std::memory_order_release);
    }
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_REBUILD_EXECUTE)

//--------------------------------------------------------------
// requestRebuild - DSPグラフの再構築 (Message Thread)
//--------------------------------------------------------------
void AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock)
{
    // UIコンポーネントへのアクセスを行うため、必ずMessage Threadで実行すること
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

    // キャプチャ用変数
    int ditherDepth = ditherBitDepth.load();
    int osFactor = manualOversamplingFactor.load();
    OversamplingType osType = oversamplingType.load();
    NoiseShaperType nsType = noiseShaperType.load();
    DSPCore* current = activeDSP; // 現在のアクティブDSPをキャプチャ
    int generation = 0;

    RebuildTask task;
    task.currentDSP = current;
    task.buildInput.sampleRate = sampleRate;
    task.buildInput.blockSize = samplesPerBlock;
    task.buildInput.ditherBitDepth = ditherDepth;
    task.buildInput.oversamplingFactor = osFactor;
    task.buildInput.oversamplingType = static_cast<int>(osType);
    task.buildInput.noiseShaperType = static_cast<int>(nsType);

    DSPCore* currentToRelease = nullptr;
    bool queued = false;
    bool blockedAsDuplicate = false;
    bool blockedAsRecentDuplicate = false;
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        if (hasPendingTask)
        {
            const bool sameAsPending =
                std::abs(pendingTask.buildInput.sampleRate - sampleRate) <= 1.0e-6
                && pendingTask.buildInput.blockSize == samplesPerBlock
                && pendingTask.buildInput.ditherBitDepth == ditherDepth
                && pendingTask.buildInput.oversamplingFactor == osFactor
                && pendingTask.buildInput.oversamplingType == static_cast<int>(osType)
                && pendingTask.buildInput.noiseShaperType == static_cast<int>(nsType);

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
            const bool sameAsLastQueued =
                std::abs(lastQueuedTaskSignature.buildInput.sampleRate - sampleRate) <= 1.0e-6
                && lastQueuedTaskSignature.buildInput.blockSize == samplesPerBlock
                && lastQueuedTaskSignature.buildInput.ditherBitDepth == ditherDepth
                && lastQueuedTaskSignature.buildInput.oversamplingFactor == osFactor
                && lastQueuedTaskSignature.buildInput.oversamplingType == static_cast<int>(osType)
                && lastQueuedTaskSignature.buildInput.noiseShaperType == static_cast<int>(nsType);

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
        const auto runtimeCommand = makeRuntimeCommand(sampleRate,
                                                       samplesPerBlock,
                                                       ditherDepth,
                                                       osFactor,
                                                       osType,
                                                       nsType,
                                                       static_cast<std::uint64_t>(generation));
        if (!m_runtimeCommandQueue.enqueue(runtimeCommand))
            diagLog("[DIAG] requestRebuild(sr,bs): runtime command queue full generation=" + juce::String(generation));

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
    // exit フラグを立てる（predicate が次に評価された時に break する）
    rebuildThreadShouldExit.store(true, std::memory_order_release);

    // 待機中のスレッドを確実に起こす
    rebuildCV.notify_all();

    if (rebuildThread.joinable())
        rebuildThread.join();

    // Stop/start ライフサイクルを跨いで古い command が残らないようにする。
    m_runtimeCommandQueue.clear();
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
            convo::EngineCommand drainedCommands[8] {};
            int drainedCommandCount = 0;
            {
                std::unique_lock<std::mutex> lock(rebuildMutex);
                rebuildCV.wait(lock, [this] { return hasPendingTask || rebuildThreadShouldExit.load(); });

                if (rebuildThreadShouldExit.load()) break;
                if (shutdownInProgress.load(std::memory_order_acquire)
                    || gShuttingDown.load(std::memory_order_acquire))
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

            drainedCommandCount = m_runtimeCommandQueue.drainCoalesced(drainedCommands,
                                                                       static_cast<int>(std::size(drainedCommands)));

            struct DSPGuard
            {
                DSPCore* ptr;
                ~DSPGuard()
                {
                    if (ptr != nullptr)
                        retireDSP(ptr);
                }
            } dspGuard { nullptr };

            convo::RuntimeBuilder runtimeBuilder(*this);

            // Helper to check obsolescence
            const auto isObsolete = [&] {
                return isRebuildObsolete(task.generation) || rebuildThreadShouldExit.load();
            };

            if (isObsolete())
                continue;

            // 1. Prepare (メモリ確保)
            convo::BuildResult buildResult {};
            bool builtFromCommand = false;
            for (int index = 0; index < drainedCommandCount; ++index)
            {
                const auto& drained = drainedCommands[index];
                if (drained.meta.revision != static_cast<std::uint64_t>(task.generation))
                    continue;

                buildResult = runtimeBuilder.build(drained);
                builtFromCommand = true;

                if (buildResult.runtime == nullptr && buildResult.error == convo::BuildError::InvalidInput)
                {
                    diagLog("[DIAG] rebuildThreadLoop: invalid runtime command payload, fallback to task.buildInput generation="
                            + juce::String(task.generation));
                    buildResult = runtimeBuilder.build(task.buildInput);
                }

                break;
            }

            if (!builtFromCommand)
                buildResult = runtimeBuilder.build(task.buildInput);

            if (buildResult.runtime == nullptr)
            {
                diagLog("[DIAG] rebuildThreadLoop: RuntimeBuilder build failed generation="
                        + juce::String(task.generation)
                        + " error=" + juce::String(convo::toString(buildResult.error))
                        + " source=" + juce::String(builtFromCommand ? "runtime-command" : "task-build-input"));
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
                    task.currentDSP->convolver.getCurrentBufferSize() == newDSP->convolver.getCurrentBufferSize())
                {
                    // IRの生成条件が一致しているか確認
                    if (newDSP->convolver.getIRName() == task.currentDSP->convolver.getIRName() &&
                        newDSP->convolver.getPhaseMode() == task.currentDSP->convolver.getPhaseMode() &&
                        std::abs(newDSP->convolver.getMixedTransitionStartHz() - task.currentDSP->convolver.getMixedTransitionStartHz()) < 0.001f &&
                        std::abs(newDSP->convolver.getMixedTransitionEndHz() - task.currentDSP->convolver.getMixedTransitionEndHz()) < 0.001f &&
                        std::abs(newDSP->convolver.getMixedPreRingTau() - task.currentDSP->convolver.getMixedPreRingTau()) < 0.001f &&
                        newDSP->convolver.getExperimentalDirectHeadEnabled() == task.currentDSP->convolver.getExperimentalDirectHeadEnabled() &&
                        std::abs(newDSP->convolver.getTargetIRLength() - task.currentDSP->convolver.getTargetIRLength()) < 0.001f)
                    {
                        // 既存のConvolutionエンジンを共有（クローン回避・グリッチ防止）
                        newDSP->convolver.shareConvolutionEngineFrom(task.currentDSP->convolver);
                        irReused = true;
                    }
                }
            }

            // 3. Rebuild IR if needed (Heavy operation)
            if (!irReused && newDSP->convolver.getIRLength() > 0)
            {
                if (isObsolete())
                    continue;
                newDSP->convolver.rebuildAllIRsSynchronous(isObsolete);
            }

            const auto warmupError = runtimeBuilder.validateWarmup(*newDSP);
            if (warmupError != convo::BuildError::None)
            {
                const bool retryable = shouldRetryWarmupFailure(*newDSP);
                diagLog("[DIAG] rebuildThreadLoop: RuntimeBuilder warmup failed generation="
                    + juce::String(task.generation)
                    + " error=" + juce::String(convo::toString(warmupError))
                    + " retryable=" + juce::String(static_cast<int>(retryable))
                    + " irLoaded=" + juce::String(static_cast<int>(newDSP->convolver.isIRLoaded()))
                    + " irFinalized=" + juce::String(static_cast<int>(newDSP->convolver.isIRFinalized()))
                    + " irLoading=" + juce::String(static_cast<int>(newDSP->convolver.isLoadingIR())));

                if (retryable)
                    requestRebuild(convo::RebuildKind::Structural);

                continue;
            }

            if (isObsolete())
                continue;

            // 4. Refresh Latency (Prevent pitch slide during fade-in)
            newDSP->convolver.refreshLatency();

            // 5. Fade In
            newDSP->fadeInSamplesLeft.store(DSPCore::FADE_IN_SAMPLES, std::memory_order_relaxed);

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
