#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "audioengine/AudioEngine.h"
#include "CacheManager.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "AlignedAllocation.h"
#include "ProgressiveUpgradeThread.h"

#include "audioengine/AtomicAccess.h"

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOAD_PIPELINE)

// ────────────────────────────────────────────────────────────────
// Load Pipeline (loadImpulseResponse, applyComputedIR, finalize)
// ────────────────────────────────────────────────────────────────

bool ConvolverProcessor::loadImpulseResponse(const juce::File& irFile, bool optimizeForRealTime)
{
    // ファイル指定あり: 新規ロード
    // ファイル指定なし: 現在のデータでリビルド (SR変更時など)
    bool isRebuild = (irFile == juce::File());

    if (isRebuild)
    {
        if (convo::exchangeAtomic(isRebuilding, true, std::memory_order_acquire)) // acquire: 先行 publishAtomic(isRebuilding=false, release) と HB
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
            convo::publishAtomic(isRebuilding, false, std::memory_order_release); // release: timer/load 経路の acquire と HB
            return false;
        }
    }

    if (!isRebuild && !irFile.existsAsFile())
    {
        return false;
    }

    convo::publishAtomic(isLoading, true, std::memory_order_release);   // release: timer/UI 側 isLoading acquire と HB
    convo::publishAtomic(irFinalized, false, std::memory_order_release); // release: Runtime 側 irFinalized acquire と HB
    lastError.clear(); // 新しいロード開始時にエラーをクリア

    // 既存のローダーを停止してゴミ箱へ退避 (即時resetによるブロックを回避)
    if (activeLoader)
    {
        activeLoader->signalThreadShouldExit();
        loaderTrashBin.push_back(std::move(activeLoader));
    }
    const double rawProcessingSampleRate = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire); // acquire: prepareToPlay/applyNewState の publishAtomic release と HB
    const double processingSampleRate = (std::isfinite(rawProcessingSampleRate) && rawProcessingSampleRate > 0.0)
                                          ? rawProcessingSampleRate
                                          : 48000.0;
    const int processingBlockSize = juce::jlimit(1, MAX_BLOCK_SIZE,
                                                 [&]{ const int bs = convo::consumeAtomic(currentBufferSize, std::memory_order_acquire); return bs > 0 ? bs : 512; }()); // acquire: prepareToPlay の publishAtomic release と HB
    const BuildSnapshot buildSnapshot = captureBuildSnapshot();
    const int clampedPhaseMode = juce::jlimit(static_cast<int>(PhaseMode::AsIs),
                                              static_cast<int>(PhaseMode::Minimum),
                                              buildSnapshot.phaseMode);
    const PhaseMode snapshotPhaseMode = static_cast<PhaseMode>(clampedPhaseMode);
    if (isRebuild)
    {
        const IRState* state = acquireIRState();
        if (state == nullptr || state->ir == nullptr || state->sampleRate <= 0.0)
        {
            releaseIRState(state);
            convo::publishAtomic(isLoading, false, std::memory_order_release);   // release: timer/UI 側 acquire と HB
            convo::publishAtomic(isRebuilding, false, std::memory_order_release); // release: timer/load 経路 acquire と HB
            return false;
        }

        activeLoader = std::make_unique<LoaderThread>(*this, *(state->ir), state->sampleRate, processingSampleRate, processingBlockSize, snapshotPhaseMode,
                                                      buildSnapshot.mixedTransitionStartHz, buildSnapshot.mixedTransitionEndHz,
                                                      buildSnapshot.mixedPreRingTau,
                                                      convo::consumeAtomic(currentIRScale, std::memory_order_acquire), // acquire: applyNewState/snapshot restore 側 publishAtomic release と HB
                                                      buildSnapshot);
        releaseIRState(state);
    }
    else
    {
        activeLoader = std::make_unique<LoaderThread>(*this, irFile, processingSampleRate, processingBlockSize, snapshotPhaseMode,
                                                      buildSnapshot.mixedTransitionStartHz, buildSnapshot.mixedTransitionEndHz,
                                                      buildSnapshot.mixedPreRingTau,
                                                      buildSnapshot);
        convo::publishAtomic(currentIrOptimized, optimizeForRealTime, std::memory_order_release); // release: UI/loader 側 acquire と HB
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
    if (!isProgressiveUpgradeEnabled())
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
                                                                getRcuProvider() != nullptr
                                                                    ? &getRcuProvider()->getAffinityManager()
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
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        pendingOverride.targetUpgradeFFTSize = resolved;
    }
}

[[nodiscard]] int ConvolverProcessor::getTargetUpgradeFFTSize() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return pendingOverride.targetUpgradeFFTSize;
}

void ConvolverProcessor::setEnableProgressiveUpgrade(bool enable)
{
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        pendingOverride.enableProgressiveUpgrade = enable;
    }
    if (!enable)
        stopUpgradeThread();
}

[[nodiscard]] bool ConvolverProcessor::isProgressiveUpgradeEnabled() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return pendingOverride.enableProgressiveUpgrade;
}

void ConvolverProcessor::setMaxCacheEntries(size_t maxEntries)
{
    const size_t clamped = juce::jlimit<size_t>(1, 64, maxEntries);
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        pendingOverride.maxCacheEntries = static_cast<int>(clamped);
    }
    if (cacheManager)
        cacheManager->evictLRU(clamped);
}

[[nodiscard]] size_t ConvolverProcessor::getMaxCacheEntries() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return static_cast<size_t>(pendingOverride.maxCacheEntries);
}

void ConvolverProcessor::clearCache()
{
    stopUpgradeThread();
    if (cacheManager)
        cacheManager->clear();
}

[[nodiscard]] bool ConvolverProcessor::isCacheEntrySafeToDelete(uint64_t cacheKey, int fftSize) const
{
    const uint64_t activeKey = convo::consumeAtomic(activeCacheKey, std::memory_order_acquire); // acquire: applyComputedIR の publishAtomic release と HB
    const int activeFFT = convo::consumeAtomic(activeCacheFFTSize, std::memory_order_acquire);   // acquire: applyComputedIR の publishAtomic release と HB

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
    const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire); // acquire: prepareToPlay/applyNewState の publishAtomic release と HB
    const int targetFFT = getTargetUpgradeFFTSize();
    const int lowResFFT = 512;
    const int phase = static_cast<int>(getPhaseMode());
    const size_t cacheLimit = getMaxCacheEntries();

    int appliedFft = 0;
    const uint64_t targetKey = CacheManager::computeKey(irFile, targetFFT, sr, phase, targetFFT);

    if (cacheManager)
    {
        auto directTarget = cacheManager->loadPreparedState(targetKey, targetFFT, generation);
        if (directTarget)
        {
            directTarget->originalFileName = irFile.getFileNameWithoutExtension();
            appliedFft = targetFFT;
            applyComputedIR(std::move(directTarget));
        }
    }

    if (appliedFft == 0 && cacheManager && irConverter)
    {
        const uint64_t lowResKey = CacheManager::computeKey(irFile, lowResFFT, sr, phase, lowResFFT);
        auto cachedLow = cacheManager->loadPreparedState(lowResKey, lowResFFT, generation);

        if (cachedLow)
        {
            cachedLow->originalFileName = irFile.getFileNameWithoutExtension();
            appliedFft = lowResFFT;
            applyComputedIR(std::move(cachedLow));
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
                applyComputedIR(std::move(prepared));
            }
        }
    }

    if (appliedFft > 0)
    {
        startProgressiveUpgrade(irFile, sr, appliedFft, generation, targetKey);
    }
}

void ConvolverProcessor::applyComputedIR(std::unique_ptr<ConvolverIRPayload> prepared)
{
    if (!prepared)
        return;

    if (!convolverStateGeneration.isCurrentGeneration(prepared->generationId))
        return;

    const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire); // acquire: prepareToPlay/applyNewState の publishAtomic release と HB
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

        juce::Logger::writeToLog("applyComputedIR: applied scaleFactor=" + juce::String(sf)
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
            convo::publishAtomic(isLoading, false, std::memory_order_release); // release: timer/UI 側 acquire と HB
            return;
        }
    }

    // 1. UI 用状態の更新
    {
        const juce::ScopedLock sl(irFileLock);
        irName = prepared->originalFileName.isNotEmpty()
               ? prepared->originalFileName
               : currentIrFile.getFileNameWithoutExtension();
    }

    convo::publishAtomic(currentSampleRate, prepared->sampleRate, std::memory_order_release); // release: Runtime/UI 側 acquire と HB
    convo::publishAtomic(irLength, prepared->timeDomainIR ? prepared->timeDomainIR->getNumSamples() : 0, std::memory_order_release); // release: UI 側 acquire と HB

    // RCU経路では convolution を経由しないため、UI表示用のレイテンシー推定値を更新する。
    {
        const bool directHeadActive = getExperimentalDirectHeadEnabled();
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
        convo::publishAtomic(uiAlgorithmLatencySamples, algorithmLatency, std::memory_order_release); // release: UI 側 acquire と HB
        convo::publishAtomic(uiIrPeakLatencySamples, irPeakLatency, std::memory_order_release);       // release: UI 側 acquire と HB
        convo::publishAtomic(uiTotalLatencySamples, totalLatency, std::memory_order_release);          // release: UI 側 acquire と HB
        convo::publishAtomic(uiDirectHeadActive, directHeadActive, std::memory_order_release);         // release: UI 側 acquire と HB
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
        updateIRState(*(prepared->timeDomainIR), prepared->sampleRate);

    // 3. RCU 状態の更新（★ 軽量化: partitionData/numPartitions/partitionSizeBytes はデッドコードのため除去）

    auto newState = std::make_unique<convo::ConvolverState>(prepared->fftSize,
                                                          prepared->generationId,
                                                          prepared->sampleRate);

    convo::publishAtomic(activeCacheKey, prepared->cacheKey, std::memory_order_release); // release: cache 判定側 acquire と HB
    convo::publishAtomic(activeCacheFFTSize, newState->fftSize, std::memory_order_release); // release: cache 判定側 acquire と HB

    updateConvolverState(std::move(newState));

    // 4. FINAL COMMIT: 確定フラグを立ててからレイテンシを1回だけ反映する。
    convo::publishAtomic(irFinalized, true, std::memory_order_release); // release: Runtime 側 irFinalized acquire と HB
    refreshLatency();

    // 4. UI 通知
    postCoalescedChangeNotification();
    convo::publishAtomic(lastPreparedIRApplyTicks, juce::Time::getHighResolutionTicks(), std::memory_order_release); // release: UI/診断側 acquire と HB

    convo::publishAtomic(isLoading, false, std::memory_order_release); // release: timer/UI 側 acquire と HB
    setLoadingProgress(1.0f);
}

void ConvolverProcessor::handleLoadError(const juce::String& error)
{
    lastError = error;
    convo::publishAtomic(irFinalized, isIRLoaded(), std::memory_order_release); // release: Runtime 側 irFinalized acquire と HB
    convo::publishAtomic(isLoading, false, std::memory_order_release);           // release: timer/UI 側 acquire と HB
    convo::publishAtomic(isRebuilding, false, std::memory_order_release);        // release: timer/load 経路 acquire と HB
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
    if (loaderTrashBin.size() > 2)
    {
        for (auto it = loaderTrashBin.begin(); it != loaderTrashBin.end() && loaderTrashBin.size() > 2; )
        {
            if (*it != nullptr && (*it)->waitForThreadToExit(0))
                it = loaderTrashBin.erase(it);
            else
                ++it;
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
                                                          const BuildSnapshot& buildSnapshot,
                                                          double scaleFactor,
                                                          std::unique_ptr<juce::AudioBuffer<double>> loadedIR,
                                                          std::unique_ptr<juce::AudioBuffer<double>> displayIR)
{
    // ここはMessage Thread上で実行されるためMKL規約を完全に遵守する
    // メモリ確保失敗に備えて try-catch を使用する
    try
    {
        auto newConv = convo::aligned_make_unique<StereoConvolver>();

        convo::FilterSpec spec;
        spec.sampleRate = sr;
        {
            spec.hcMode = static_cast<convo::HCMode>(buildSnapshot.nucHCMode);
            spec.lcMode = static_cast<convo::LCMode>(buildSnapshot.nucLCMode);
            spec.tailMode = juce::jlimit(static_cast<int>(TailMode::AirAbsorption),
                                         static_cast<int>(TailMode::Bypass),
                                         buildSnapshot.tailMode);
            spec.tailEnabled = (spec.tailMode != static_cast<int>(TailMode::Bypass));
            spec.tailStartSeconds = static_cast<double>(buildSnapshot.tailStartSec);
            spec.tailStrength = static_cast<double>(buildSnapshot.tailStrength);
            spec.tailL1L2Multiplier = buildSnapshot.tailL1L2Multiplier;
        }

        if (newConv->init(irL.release(), irR.release(), length, sr, peakDelay,
                  maxFFTSize, knownBlockSize, firstPartition, preferredCallSize, scaleFactor,
                  getExperimentalDirectHeadEnabled(),
                  &spec, this))
        {
            jassert(newConv->areNUCDescriptorsCommitted());
            applyNewState(newConv.release(), std::move(loadedIR), sr, length, isRebuild, irFile, scaleFactor, std::move(displayIR));
        }
        else
        {
            handleLoadError("Failed to initialize NUC engine (Memory allocation or MKL setup failed).");
        }
    }
    catch (const std::bad_alloc&)
    {
        handleLoadError("Failed to initialize NUC engine (Out of memory).");
    }
    catch (const std::exception& e)
    {
        handleLoadError(juce::String("NUC engine initialization failed: ") + e.what());
    }
    catch (...)
    {
        handleLoadError("NUC engine initialization failed: Unknown error");
    }
}

void ConvolverProcessor::applyNewState(StereoConvolver* newConv,
                                       std::unique_ptr<juce::AudioBuffer<double>> loadedIR,
                                       double loadedSR,
                                       int targetLength,
                                       bool isRebuild,
                                       const juce::File& file,
                                       double scaleFactor,
                                       std::unique_ptr<juce::AudioBuffer<double>> displayIR)
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
        convo::publishAtomic(currentIRScale, scaleFactor, std::memory_order_release);  // release: Loader/Rebuild 側 currentIRScale acquire と HB
    }

    // スナップショット更新 (表示用)
    if (visualizationEnabled && displayIR)
    {
        createWaveformSnapshot(*displayIR);
        createFrequencyResponseSnapshot(*displayIR, loadedSR);
    }

    switchEngineOnMessageThread(newConv);

    convo::publishAtomic(irLength, targetLength, std::memory_order_release);       // release: UI 側 acquire と HB
    convo::publishAtomic(currentSampleRate, loadedSR, std::memory_order_release);  // release: Runtime/Loader 側 acquire と HB

    // FINAL COMMIT: フラグ確定後にレイテンシを1回だけ反映する。
    convo::publishAtomic(irFinalized, true, std::memory_order_release); // release: Runtime 側 irFinalized acquire と HB
    refreshLatency();

    convo::publishAtomic(isLoading, false, std::memory_order_release);    // release: timer/UI 側 acquire と HB
    convo::publishAtomic(isRebuilding, false, std::memory_order_release); // release: timer/load 経路 acquire と HB
    if (convo::exchangeAtomic(rebuildPendingAfterLoad, false, std::memory_order_acq_rel) && isIRLoaded()) // acq_rel: acquire で既存要求観測; release で false 公開
    {
        const bool queued = juce::MessageManager::callAsync([weakThis = juce::WeakReference<ConvolverProcessor>(this)]()
        {
            if (auto* self = weakThis.get())
                self->rebuildAllIRs();
        });

        if (!queued)
            convo::publishAtomic(rebuildPendingAfterLoad, true, std::memory_order_release); // release: timer 側 acquire と HB
    }
    updateLatencyCache();
    requestHostDisplayUpdate();
    postCoalescedChangeNotification();
}

void ConvolverProcessor::switchEngineOnMessageThread(StereoConvolver* newEngine) noexcept
{
    if (newEngine == nullptr)
        return;

    auto* oldEngine = exchangeActiveEngine(newEngine, std::memory_order_acq_rel);
    // [work21 P1-15/Phase-D] Router経由でepoch進捗. provider必須 (fallback削除).
    if (auto* provider = getRcuProvider())
        provider->advanceRetireEpoch();
    else
        jassertfalse; // provider must be set before any engine switch
    if (oldEngine)
        retireStereoConvolver(oldEngine, 0);
}

// applyNewState へのシンプルな転送。新しいコードからは commitNewConvolver を使用する。
void ConvolverProcessor::commitNewConvolver(StereoConvolver* newConv,
                                            std::unique_ptr<juce::AudioBuffer<double>> loadedIR,
                                            double loadedSR, int targetLength, bool isRebuild,
                                            const juce::File& file, double scaleFactor,
                                            std::unique_ptr<juce::AudioBuffer<double>> displayIR)
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
    convo::publishAtomic(loadProgress, p, std::memory_order_release); // release: UI 側 progress acquire と HB
    // sendChangeMessage() はメッセージスレッド専用。LoaderThread など任意の
    // スレッドから呼ばれるため、既存の postCoalescedChangeNotification() を使う。
    // 進捗通知は合体（coalesce）して問題ない（最新値が loadProgress に保持される）。
    postCoalescedChangeNotification();
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOAD_PIPELINE
