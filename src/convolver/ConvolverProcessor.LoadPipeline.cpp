#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "audioengine/AudioEngine.h"
#include "CacheManager.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "core/EpochManager.h"
#include "AlignedAllocation.h"
#include "ProgressiveUpgradeThread.h"
#include "PreparedIRState.h"

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOAD_PIPELINE)

// ────────────────────────────────────────────────────────────────
// Load Pipeline (loadImpulseResponse, applyPreparedIRState, finalize)
// ────────────────────────────────────────────────────────────────

bool ConvolverProcessor::loadImpulseResponse(const juce::File& irFile, bool optimizeForRealTime)
{
    // ファイル指定あり: 新規ロード
    // ファイル指定なし: 現在のデータでリビルド (SR変更時など)
    bool isRebuild = (irFile == juce::File());

    if (isRebuild)
    {
        if (isRebuilding.exchange(true, std::memory_order_acquire))
        {
            juce::Logger::writeToLog("ConvolverProcessor::rebuild (via loadImpulseResponse) already in progress, skipping");
            return true;
        }
        const IRState* state = acquireIRState();
        const bool missingState = (state == nullptr || state->ir == nullptr
                                || state->ir->getNumSamples() == 0 || state->sampleRate <= 0.0);
        releaseIRState(state);
        if (missingState)
        {
            isRebuilding.store(false, std::memory_order_release);
            return false;
        }
    }

    if (!isRebuild && !irFile.existsAsFile())
    {
        return false;
    }

    isLoading.store(true);
    irFinalized.store(false, std::memory_order_release);
    lastError.clear(); // 新しいロード開始時にエラーをクリア

    // 既存のローダーを停止してゴミ箱へ退避 (即時resetによるブロックを回避)
    if (activeLoader)
    {
        activeLoader->signalThreadShouldExit();
        loaderTrashBin.push_back(std::move(activeLoader));
    }
    const double rawProcessingSampleRate = currentSampleRate.load(std::memory_order_acquire);
    const double processingSampleRate = (std::isfinite(rawProcessingSampleRate) && rawProcessingSampleRate > 0.0)
                                          ? rawProcessingSampleRate
                                          : 48000.0;
    const int processingBlockSize = juce::jlimit(1, MAX_BLOCK_SIZE,
                                                 [&]{ const int bs = currentBufferSize.load(std::memory_order_acquire); return bs > 0 ? bs : 512; }());
    if (isRebuild)
    {
        const IRState* state = acquireIRState();
        if (state == nullptr || state->ir == nullptr || state->sampleRate <= 0.0)
        {
            releaseIRState(state);
            isLoading.store(false, std::memory_order_release);
            isRebuilding.store(false, std::memory_order_release);
            return false;
        }

        activeLoader = std::make_unique<LoaderThread>(*this, *(state->ir), state->sampleRate, processingSampleRate, processingBlockSize, getPhaseMode(),
                                                      mixedTransitionStartHz.load(std::memory_order_acquire), mixedTransitionEndHz.load(std::memory_order_acquire),
                                                      mixedPreRingTau.load(std::memory_order_acquire), currentIRScale.load(std::memory_order_acquire));
        releaseIRState(state);
    }
    else
    {
        activeLoader = std::make_unique<LoaderThread>(*this, irFile, processingSampleRate, processingBlockSize, getPhaseMode(),
                                                      mixedTransitionStartHz.load(std::memory_order_acquire), mixedTransitionEndHz.load(std::memory_order_acquire),
                                                      mixedPreRingTau.load(std::memory_order_acquire));
        currentIrOptimized.store(optimizeForRealTime);
    }

    activeLoader->startThread();

    return true;
}

void ConvolverProcessor::stopUpgradeThread()
{
    if (upgradeThread)
    {
        upgradeThread->cancel();
        upgradeThread->stopThread(2000);
        upgradeThread.reset();
    }
}

void ConvolverProcessor::startProgressiveUpgrade(const juce::File& file,
                                                 double sampleRate,
                                                 int currentFFTSize,
                                                 uint64_t generation,
                                                 uint64_t baseKey)
{
    if (!enableProgressiveUpgrade.load(std::memory_order_acquire))
        return;

    const int targetFFT = getTargetUpgradeFFTSize();
    if (currentFFTSize >= targetFFT)
        return;

    stopUpgradeThread();

    upgradeThread = std::make_unique<ProgressiveUpgradeThread>(*this,
                                                                file,
                                                                sampleRate,
                                                                currentFFTSize,
                                                                targetFFT,
                                                                static_cast<int>(getPhaseMode()),
                                                                generation,
                                                                baseKey,
                                                                *irConverter,
                                                                *cacheManager,
                                                                rcuProvider != nullptr
                                                                    ? const_cast<ThreadAffinityManager*>(&rcuProvider->getAffinityManager())
                                                                    : nullptr);
    upgradeThread->startThread();
}

void ConvolverProcessor::setTargetUpgradeFFTSize(int fftSize)
{
    static constexpr int allowed[] = { 512, 1024, 2048, 4096 };
    int resolved = 4096;
    for (int a : allowed)
    {
        if (fftSize <= a)
        {
            resolved = a;
            break;
        }
    }
    targetUpgradeFFTSize.store(resolved, std::memory_order_release);
}

int ConvolverProcessor::getTargetUpgradeFFTSize() const
{
    return targetUpgradeFFTSize.load(std::memory_order_acquire);
}

void ConvolverProcessor::setEnableProgressiveUpgrade(bool enable)
{
    enableProgressiveUpgrade.store(enable, std::memory_order_release);
    if (!enable)
        stopUpgradeThread();
}

bool ConvolverProcessor::isProgressiveUpgradeEnabled() const
{
    return enableProgressiveUpgrade.load(std::memory_order_acquire);
}

void ConvolverProcessor::setMaxCacheEntries(size_t maxEntries)
{
    const size_t clamped = juce::jlimit<size_t>(1, 64, maxEntries);
    maxCacheEntries.store(clamped, std::memory_order_release);
    if (cacheManager)
        cacheManager->evictLRU(clamped);
}

size_t ConvolverProcessor::getMaxCacheEntries() const
{
    return maxCacheEntries.load(std::memory_order_acquire);
}

void ConvolverProcessor::clearCache()
{
    stopUpgradeThread();
    if (cacheManager)
        cacheManager->clear();
}

bool ConvolverProcessor::isCacheEntrySafeToDelete(uint64_t cacheKey, int fftSize) const
{
    const uint64_t activeKey = activeCacheKey.load(std::memory_order_acquire);
    const int activeFFT = activeCacheFFTSize.load(std::memory_order_acquire);

    if (cacheKey == activeKey && fftSize == activeFFT)
        return false;

    // Index 2 for background/cache reader
    struct LocalGuard {
        const ConvolverProcessor& cp;
        LocalGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterStateReader(2); }
        ~LocalGuard() { cp.exitStateReader(2); }
    } guard(*this);

    if (auto* state = rcuSwapper.getState())
    {
        if (convolverStateGeneration.isCurrentGeneration(state->generationId)
            && state->fftSize == fftSize
            && cacheKey == activeKey)
        {
            return false;
        }
    }

    return true;
}

void ConvolverProcessor::loadIR(const juce::File& irFile)
{
    JUCE_ASSERT_MESSAGE_THREAD;

    if (!irFile.existsAsFile())
        return;

    {
        const juce::ScopedLock sl(irFileLock);
        currentIrFile = irFile;
    }

    stopUpgradeThread();

    const uint64_t generation = convolverStateGeneration.bumpGeneration();
    const double sr = currentSampleRate.load(std::memory_order_acquire);
    const int targetFFT = getTargetUpgradeFFTSize();
    const int lowResFFT = 512;
    const int phase = static_cast<int>(getPhaseMode());
    const size_t cacheLimit = maxCacheEntries.load(std::memory_order_acquire);

    int appliedFft = 0;
    const uint64_t targetKey = CacheManager::computeKey(irFile, targetFFT, sr, phase, targetFFT);

    if (cacheManager)
    {
        auto directTarget = cacheManager->load(targetKey, targetFFT, generation);
        if (directTarget)
        {
            directTarget->originalFileName = irFile.getFileNameWithoutExtension();
            appliedFft = targetFFT;
            applyPreparedIRState(std::move(directTarget));
        }
    }

    if (appliedFft == 0 && cacheManager && irConverter)
    {
        const uint64_t lowResKey = CacheManager::computeKey(irFile, lowResFFT, sr, phase, lowResFFT);
        auto cachedLow = cacheManager->load(lowResKey, lowResFFT, generation);

        if (cachedLow)
        {
            cachedLow->originalFileName = irFile.getFileNameWithoutExtension();
            appliedFft = lowResFFT;
            applyPreparedIRState(std::move(cachedLow));
        }
        else
        {
            IRConverter::ConvertConfig cfg;
            cfg.fftSize = lowResFFT;
            cfg.targetSampleRate = sr;
            cfg.phaseMode = phase;
            cfg.partitionSize = lowResFFT;
            cfg.generationId = generation;
            cfg.cacheKey = lowResKey;

            auto prepared = irConverter->convertFile(irFile, cfg, [this, generation]()
            {
                return !convolverStateGeneration.isCurrentGeneration(generation);
            });

            if (prepared)
            {
                prepared->originalFileName = irFile.getFileNameWithoutExtension();
                cacheManager->save(lowResKey, lowResFFT, *prepared);
                cacheManager->evictLRU(cacheLimit);
                appliedFft = lowResFFT;
                applyPreparedIRState(std::move(prepared));
            }
        }
    }

    if (appliedFft > 0)
    {
        startProgressiveUpgrade(irFile, sr, appliedFft, generation, targetKey);
    }
}

void ConvolverProcessor::applyPreparedIRState(std::unique_ptr<PreparedIRState> prepared)
{
    if (!prepared)
        return;

    if (!convolverStateGeneration.isCurrentGeneration(prepared->generationId))
        return;

    const double sr = currentSampleRate.load(std::memory_order_acquire);
    if (sr > 0.0 && std::abs(prepared->sampleRate - sr) > 1e-6)
        return;

    JUCE_ASSERT_MESSAGE_THREAD;

    // scaleFactor 適用（timeDomainIR はコピーしてから適用し、共有元を保護）
    if (prepared->hasScaleFactor && prepared->scaleFactor != 1.0)
    {
        const double sf = prepared->scaleFactor;

        if (prepared->timeDomainIR)
        {
            auto scaledTimeIR = std::make_unique<juce::AudioBuffer<double>>(*prepared->timeDomainIR);
            for (int ch = 0; ch < scaledTimeIR->getNumChannels(); ++ch)
            {
                double* data = scaledTimeIR->getWritePointer(ch);
                const int numSamples = scaledTimeIR->getNumSamples();
                cblas_dscal(numSamples, sf, data, 1);

                for (int i = 0; i < numSamples; ++i)
                {
                    if (!std::isfinite(data[i]))
                        data[i] = 0.0;
                }
            }
            prepared->timeDomainIR = std::move(scaledTimeIR);
        }

        if (prepared->partitionData && prepared->partitionSizeBytes > 0)
        {
            const size_t numDoubles = prepared->partitionSizeBytes / sizeof(double);
            cblas_dscal(static_cast<MKL_INT>(numDoubles), sf, prepared->partitionData, 1);
        }

        juce::Logger::writeToLog("applyPreparedIRState: applied scaleFactor=" + juce::String(sf)
            + " to timeDomainIR and partitionData");
    }

    if (prepared->timeDomainIR)
    {
        bool valid = true;
        const int channels = prepared->timeDomainIR->getNumChannels();
        const int samples = prepared->timeDomainIR->getNumSamples();
        double newPeak = 0.0;
        double newEnergy = 0.0;

        for (int ch = 0; ch < channels && valid; ++ch)
        {
            const double* data = prepared->timeDomainIR->getReadPointer(ch);
            for (int i = 0; i < samples; ++i)
            {
                const double value = data[i];
                if (!std::isfinite(value) || std::abs(value) > 10.0)
                {
                    valid = false;
                    break;
                }

                newPeak = std::max(newPeak, std::abs(value));
                newEnergy += value * value;
            }
        }

        if (valid && samples > 0)
        {
            const double newRms = std::sqrt(newEnergy / static_cast<double>(channels * samples));
            const IRState* irState = acquireIRState();
            auto currentIr = (irState != nullptr) ? irState->ir : nullptr;

            if (currentIr && currentIr->getNumChannels() > 0 && currentIr->getNumSamples() > 0)
            {
                double currentPeak = 0.0;
                double currentEnergy = 0.0;
                const int currentChannels = currentIr->getNumChannels();
                const int currentSamples = currentIr->getNumSamples();

                for (int ch = 0; ch < currentChannels; ++ch)
                {
                    const double* data = currentIr->getReadPointer(ch);
                    for (int i = 0; i < currentSamples; ++i)
                    {
                        const double value = data[i];
                        currentPeak = std::max(currentPeak, std::abs(value));
                        currentEnergy += value * value;
                    }
                }

                const double currentRms = std::sqrt(currentEnergy / static_cast<double>(currentChannels * currentSamples));
                const bool excessivePeakJump = currentPeak > 1.0e-9 && newPeak > currentPeak * 4.0 && newPeak > 0.5;
                const bool excessiveRmsJump = currentRms > 1.0e-9 && newRms > currentRms * 4.0 && newRms > 0.25;
                if (excessivePeakJump || excessiveRmsJump)
                    valid = false;
            }

            releaseIRState(irState);
        }

        if (!valid)
        {
            lastError = "Invalid IR (amplitude out of range or sudden level jump)";
            isLoading.store(false, std::memory_order_release);
            return;
        }
    }

    // 1. UI 用レガシー状態の更新
    {
        const juce::ScopedLock sl(irFileLock);
        irName = prepared->originalFileName.isNotEmpty()
               ? prepared->originalFileName
               : currentIrFile.getFileNameWithoutExtension();
    }

    currentSampleRate.store(prepared->sampleRate, std::memory_order_release);
    irLength.store(prepared->timeDomainIR ? prepared->timeDomainIR->getNumSamples() : 0,
                   std::memory_order_release);

    // RCU経路では legacy convolution を経由しないため、UI表示用のレイテンシー推定値を更新する。
    {
        const bool directHeadActive = experimentalDirectHeadEnabled.load(std::memory_order_acquire);
        const int algorithmLatency = directHeadActive ? 0 : juce::jmax(0, prepared->fftSize);

        int irPeakLatency = 0;
        if (prepared->timeDomainIR && prepared->timeDomainIR->getNumChannels() > 0)
        {
            const int channels = prepared->timeDomainIR->getNumChannels();
            const int samples = prepared->timeDomainIR->getNumSamples();
            double bestAbs = 0.0;
            int bestIndex = 0;

            for (int ch = 0; ch < channels; ++ch)
            {
                const double* src = prepared->timeDomainIR->getReadPointer(ch);
                for (int i = 0; i < samples; ++i)
                {
                    const double a = std::abs(src[i]);
                    if (a > bestAbs)
                    {
                        bestAbs = a;
                        bestIndex = i;
                    }
                }
            }

            irPeakLatency = juce::jmax(0, bestIndex);
        }

        const int totalLatency = juce::jmin(juce::jmax(0, algorithmLatency + irPeakLatency), MAX_TOTAL_DELAY);
        uiAlgorithmLatencySamples.store(algorithmLatency, std::memory_order_release);
        uiIrPeakLatencySamples.store(irPeakLatency, std::memory_order_release);
        uiTotalLatencySamples.store(totalLatency, std::memory_order_release);
        uiDirectHeadActive.store(directHeadActive, std::memory_order_release);
        updateLatencyCache();
        requestHostDisplayUpdate();
    }

    // 2. 波形／スペクトルスナップショットの生成
    if (visualizationEnabled && prepared->timeDomainIR && prepared->timeDomainIR->getNumSamples() > 0)
    {
        createWaveformSnapshot(*(prepared->timeDomainIR));
        createFrequencyResponseSnapshot(*(prepared->timeDomainIR), prepared->sampleRate);
    }

    // loadIR() (RCU経路) では applyNewState() が呼ばれないため、
    // DSP側 rebuildAllIRsSynchronous() が参照する originalIR をここで保持する。
    if (prepared->timeDomainIR && prepared->timeDomainIR->getNumSamples() > 0)
    {
        auto irShared = std::shared_ptr<juce::AudioBuffer<double>>(std::move(prepared->timeDomainIR));
        updateIRState(irShared, prepared->sampleRate);
    }

    // 3. RCU 状態の更新

    auto newState = std::make_unique<convo::ConvolverState>(prepared->partitionData,
                                                          prepared->partitionSizeBytes,
                                                          prepared->numPartitions,
                                                          prepared->fftSize,
                                                          prepared->generationId,
                                                          prepared->sampleRate);

    prepared->partitionData = nullptr;

    activeCacheKey.store(prepared->cacheKey, std::memory_order_release);
    activeCacheFFTSize.store(newState->fftSize, std::memory_order_release);

    runtime.reallocate(newState->fftSize, newState->numPartitions);
    updateConvolverState(std::move(newState));

    // 4. FINAL COMMIT: 確定フラグを立ててからレイテンシを1回だけ反映する。
    irFinalized.store(true, std::memory_order_release);
    refreshLatency();

    // 4. UI 通知
    postCoalescedChangeNotification();
    lastPreparedIRApplyTicks.store(juce::Time::getHighResolutionTicks(), std::memory_order_release);

    // DSP リビルドトリガー:
    // Progressive upgrade が有効な場合、途中ステップ（512/1024/2048）では
    // convolverParamsChanged を呼ばない。targetFFT 到達時のみ DSP rebuild を起動する。
    // これにより CMA-ES Mixed Phase 計算が1回だけ実行され、CPU スパイクによる
    // 音切れ（タタタタ）が複数回発生しなくなる。
    // Progressive upgrade が無効な場合は常に通知する（upgrade OFF の動作を保持）。
    {
        const bool progressiveEnabled = enableProgressiveUpgrade.load(std::memory_order_acquire);
        const int publishedFftSize    = activeCacheFFTSize.load(std::memory_order_acquire);
        const bool isFinalPublish     = !progressiveEnabled
                                     || (publishedFftSize >= getTargetUpgradeFFTSize());
        if (isFinalPublish)
            listeners.call(&Listener::convolverParamsChanged, this);
    }

    isLoading.store(false, std::memory_order_release);
    setLoadingProgress(1.0f);
}

void ConvolverProcessor::handleLoadError(const juce::String& error)
{
    lastError = error;
    irFinalized.store(isIRLoaded(), std::memory_order_release);
    isLoading.store(false);
    isRebuilding.store(false, std::memory_order_release);
    // UIに通知してエラーメッセージを表示させる
    postCoalescedChangeNotification();
}

void ConvolverProcessor::cleanup()
{
    // LoaderThread のクリーンアップ (Message Thread Only)
    // 終了したスレッドのみを削除する (waitForThreadToExit(0) はブロックしない)
    for (auto it = loaderTrashBin.begin(); it != loaderTrashBin.end(); )
    {
        if ((*it)->waitForThreadToExit(0))
        {
            // [Fix] スレッドは終了済みのため、reset() を直接呼んでブロックしない。
            // JUCE の stopThread() は isThreadRunning() == false の場合に即リターンするため、
            // わざわざ detached スレッドで実行する必要はない。
            it->reset();
            it = loaderTrashBin.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // 【Leak Fix】LoaderThreadの異常蓄積防止
    // スレッドが終了しない場合でも、一定数を超えたら強制削除してメモリを解放する。
    // [FIX] detached thread はプロセス終了時に未定義動作を引き起こすため、
    //       同期的なチェックと削除に切り替える。
    while (loaderTrashBin.size() > 2)
    {
        // 最も古いスレッドが終了しているか非ブロックで確認
        if (loaderTrashBin.front() && loaderTrashBin.front()->waitForThreadToExit(0))
        {
            // 終了済みなら安全に削除 (unique_ptrのデストラクタが呼ばれる)
            loaderTrashBin.pop_front();
        }
        else
        {
            // 終了していないスレッドが見つかったら、今回はここまで。次回タイマーで再試行。
            break;
        }
    }
}

void ConvolverProcessor::finalizeNUCEngineOnMessageThread(convo::ScopedAlignedPtr<double> irL,
                                                          convo::ScopedAlignedPtr<double> irR,
                                                          int length,
                                                          double sr,
                                                          int peakDelay,
                                                          int maxFFTSize,
                                                          int knownBlockSize,
                                                          int firstPartition,
                                                          int preferredCallSize,
                                                          bool isRebuild,
                                                          const juce::File& irFile,
                                                          double scaleFactor,
                                                          std::shared_ptr<juce::AudioBuffer<double>> loadedIR,
                                                          std::shared_ptr<juce::AudioBuffer<double>> displayIR)
{
    // ここはMessage Thread上で実行されるためMKL規約を完全に遵守する
    // メモリ確保失敗に備えて try-catch を使用する
    try
    {
        void* mem = convo::aligned_malloc(sizeof(StereoConvolver), 64);
        new (mem) StereoConvolver();
        auto* newConv = static_cast<StereoConvolver*>(mem);

        convo::FilterSpec spec;
        spec.sampleRate = sr;
        spec.hcMode = static_cast<convo::HCMode>(nucHCMode.load(std::memory_order_acquire));
        spec.lcMode = static_cast<convo::LCMode>(nucLCMode.load(std::memory_order_acquire));
        spec.tailMode = tailProcessingMode.load(std::memory_order_acquire);
        spec.tailRolloffStartHz = tailRolloffStartHz.load(std::memory_order_acquire);
        spec.tailRolloffStrength = tailRolloffStrength.load(std::memory_order_acquire);
        spec.partitionTailStrength = partitionTailStrength.load(std::memory_order_acquire);

        if (newConv->init(irL.release(), irR.release(), length, sr, peakDelay,
                  maxFFTSize, knownBlockSize, firstPartition, preferredCallSize, scaleFactor,
                  experimentalDirectHeadEnabled.load(std::memory_order_acquire),
                  &spec, this))
        {
            jassert(newConv->areNUCDescriptorsCommitted());
            applyNewState(newConv, loadedIR, sr, length, isRebuild, irFile, scaleFactor, displayIR);
        }
        else
        {
            StereoConvolver::retireStereoConvolver(newConv);
            handleLoadError("Failed to initialize NUC engine (Memory allocation or MKL setup failed).");
        }
    }
    catch (const std::bad_alloc&)
    {
        handleLoadError("Failed to initialize NUC engine (Memory allocation or MKL setup failed).");
    }
}

void ConvolverProcessor::applyNewState(StereoConvolver* newConv,
                                       std::shared_ptr<juce::AudioBuffer<double>> loadedIR,
                                       double loadedSR,
                                       int targetLength,
                                       bool isRebuild,
                                       const juce::File& file,
                                       double scaleFactor,
                                       std::shared_ptr<juce::AudioBuffer<double>> displayIR)
{
    // 元データの更新 (新規ロード時のみ)
    if (!isRebuild)
    {
        updateIRState(loadedIR, loadedSR);
        {
            const juce::ScopedLock sl(irFileLock);
            currentIrFile = file;
        }
        irName = file.getFileNameWithoutExtension();
        currentIRScale.store(scaleFactor, std::memory_order_release);  // [Bug 4 fix] atomic store
    }

    // スナップショット更新 (表示用)
    if (visualizationEnabled && displayIR)
    {
        createWaveformSnapshot(*displayIR);
        createFrequencyResponseSnapshot(*displayIR, loadedSR);
    }

    switchEngineOnMessageThread(newConv);

    irLength.store(targetLength, std::memory_order_release);
    currentSampleRate.store(loadedSR, std::memory_order_release);

    // FINAL COMMIT: フラグ確定後にレイテンシを1回だけ反映する。
    irFinalized.store(true, std::memory_order_release);
    refreshLatency();

    isLoading.store(false);
    isRebuilding.store(false, std::memory_order_release);
    if (rebuildPendingAfterLoad.exchange(false, std::memory_order_acq_rel) && isIRLoaded())
        requestDebouncedRebuild();
    updateLatencyCache();
    requestHostDisplayUpdate();
    postCoalescedChangeNotification();
}

void ConvolverProcessor::switchEngineOnMessageThread(StereoConvolver* newEngine) noexcept
{
    if (newEngine == nullptr)
        return;

    auto* oldEngine = m_activeEngine.exchange(newEngine, std::memory_order_acq_rel);
    if (oldEngine)
        retireStereoConvolver(oldEngine, 0);

    convo::EpochManager::instance().advanceEpoch();
}

// applyNewState へのシンプルな転送。新しいコードからは commitNewConvolver を使用する。
void ConvolverProcessor::commitNewConvolver(StereoConvolver* newConv,
                                            std::shared_ptr<juce::AudioBuffer<double>> loadedIR,
                                            double loadedSR, int targetLength, bool isRebuild,
                                            const juce::File& file, double scaleFactor,
                                            std::shared_ptr<juce::AudioBuffer<double>> displayIR)
{
    applyNewState(newConv, std::move(loadedIR), loadedSR, targetLength, isRebuild, file, scaleFactor, std::move(displayIR));
}

void ConvolverProcessor::evictOldestCacheEntry()
{
    const juce::ScopedLock sl(cacheMutex);
    if (irCache.size() <= MAX_CACHE_ENTRIES) return;

    auto oldest = irCache.begin();
    uint32_t minTime = std::numeric_limits<uint32_t>::max();

    for (auto it = irCache.begin(); it != irCache.end(); ++it)
    {
        if (it->second.lastUsedTime < minTime)
        {
            minTime = it->second.lastUsedTime;
            oldest = it;
        }
    }

    if (oldest != irCache.end())
        irCache.erase(oldest);
}

void ConvolverProcessor::setLoadingProgress(float p)
{
    loadProgress.store(p, std::memory_order_release);
    // sendChangeMessage() はメッセージスレッド専用。LoaderThread など任意の
    // スレッドから呼ばれるため、既存の postCoalescedChangeNotification() を使う。
    // 進捗通知は合体（coalesce）して問題ない（最新値が loadProgress に保持される）。
    postCoalescedChangeNotification();
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOAD_PIPELINE
