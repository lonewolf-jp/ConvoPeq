#include "ProgressiveUpgradeThread.h"

#include "ConvolverProcessor.h"
#include "IRConverter.h"
#include "CacheManager.h"
#include "PreparedIRState.h"
#include "core/ThreadAffinityManager.h"

#include <cmath>

ProgressiveUpgradeThread::ProgressiveUpgradeThread(ConvolverProcessor& p,
                                                   const juce::File& file,
                                                   double sr,
                                                   int currentFft,
                                                   int targetFft,
                                                   int phase,
                                                   uint64_t baseGeneration,
                                                   uint64_t key,
                                                   IRConverter& conv,
                                                   CacheManager& cache,
                                                   ThreadAffinityManager* affinityMgr)
    : juce::Thread("ConvolverProgressiveUpgrade"),
      processor(p),
      irFile(file),
            sampleRate(sr),
            currentFFTSize(currentFft),
            targetFFTSize(targetFft),
            phaseMode(phase),
      taskGeneration(baseGeneration),
            baseCacheKey(key),
      converter(conv),
    cacheManager(cache),
    affinityManager(affinityMgr)
{
        static constexpr int kStepTable[] = { 1024, 2048, 4096 };
        for (int step : kStepTable)
        {
                if (step > currentFFTSize && step <= targetFFTSize)
                        upgradeSteps.push_back(step);
        }
}

ProgressiveUpgradeThread::~ProgressiveUpgradeThread()
{
    cancel();
    stopThread(2000);
}

void ProgressiveUpgradeThread::cancel()
{
    cancelled.store(true, std::memory_order_relaxed);
    signalThreadShouldExit();
}

bool ProgressiveUpgradeThread::isGenerationValid() const
{
    return !cancelled.load(std::memory_order_relaxed)
        && processor.isConvolverGenerationCurrent(taskGeneration);
}

bool ProgressiveUpgradeThread::checkAndCancel()
{
    if (!isGenerationValid())
    {
        cancelled.store(true, std::memory_order_relaxed);
        return true;
    }
    return false;
}

void ProgressiveUpgradeThread::run()
{
    if (affinityManager != nullptr)
        affinityManager->applyCurrentThreadPolicy(ThreadType::HeavyBackground);

    // JUCE のクロスプラットフォーム優先度設定
    setPriority(Priority::low);

    if (checkAndCancel())
        return;

    for (int step : upgradeSteps)
    {
        if (!isGenerationValid())
            return;

        if (!upgradeStep(step))
            return;
    }
}

bool ProgressiveUpgradeThread::upgradeStep(int nextFFTSize)
{
    if (!isGenerationValid())
        return false;

    const uint64_t stepKey = CacheManager::computeKey(irFile,
                                                      nextFFTSize,
                                                      sampleRate,
                                                      phaseMode,
                                                      nextFFTSize);

    auto prepared = cacheManager.load(stepKey, nextFFTSize, taskGeneration);
    if (!prepared)
    {
        prepared = converter.convertToHighRes(irFile,
                                              sampleRate,
                                              nextFFTSize,
                                              taskGeneration,
                                              stepKey,
                                              [this]()
                                              {
                                                  return this->threadShouldExit() || this->checkAndCancel();
                                              });
        if (!prepared)
            return false;

        if (!isGenerationValid())
            return false;

        cacheManager.save(stepKey, nextFFTSize, *prepared);
        cacheManager.evictLRU(processor.getMaxCacheEntries());
    }

    if (prepared)
        prepared->originalFileName = irFile.getFileNameWithoutExtension();

    if (!isGenerationValid())
        return false;

    if (prepared && prepared->timeDomainIR)
    {
        double peak = 0.0;
        const int channels = prepared->timeDomainIR->getNumChannels();
        const int samples = prepared->timeDomainIR->getNumSamples();

        for (int ch = 0; ch < channels; ++ch)
        {
            const double* data = prepared->timeDomainIR->getReadPointer(ch);
            for (int i = 0; i < samples; ++i)
            {
                const double value = data[i];
                if (!std::isfinite(value))
                    return false;
                peak = std::max(peak, std::abs(value));
            }
        }

        if (peak > 2.0)
        {
            DBG("ProgressiveUpgradeThread: generated IR has excessive peak, skipping");
            return false;
        }
    }

    // 中間ステップの engine swap は音切れ・CPU スパイクの原因となるため、
    // 最終 FFT サイズ到達時のみ publish する。
    // 中間ステップの結果はキャッシュに保存済みなので処理は無駄にならない。
    const bool isFinalStep = (nextFFTSize >= targetFFTSize);
    if (isFinalStep)
    {
        juce::MessageManager::callAsync([this, prepared = std::move(prepared)]() mutable
        {
            if (this->checkAndCancel())
                return;

            processor.applyPreparedIRState(std::move(prepared));
        });
    }

    return true;
}
