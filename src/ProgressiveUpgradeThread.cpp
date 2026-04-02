#include "ProgressiveUpgradeThread.h"

#include <Windows.h>

#include "ConvolverProcessor.h"
#include "IRConverter.h"
#include "CacheManager.h"
#include "PreparedIRState.h"

ProgressiveUpgradeThread::ProgressiveUpgradeThread(ConvolverProcessor& p,
                                                   const juce::File& file,
                                                   double sr,
                                                   int currentFft,
                                                   int targetFft,
                                                   int phase,
                                                   uint64_t baseGeneration,
                                                   uint64_t key,
                                                   IRConverter& conv,
                                                   CacheManager& cache)
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
      cacheManager(cache)
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
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);

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

    if (!isGenerationValid())
        return false;

    juce::MessageManager::callAsync([this, prepared = std::move(prepared)]() mutable
    {
        if (this->checkAndCancel())
            return;

        processor.applyPreparedIRState(std::move(prepared));
    });

    return true;
}
