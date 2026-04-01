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
                                                   int targetLength,
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
            targetLengthSamples(targetLength),
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

    for (size_t i = 0; i < upgradeSteps.size(); ++i)
    {
        if (!isGenerationValid())
            return;

        const bool isFinalStep = (i == upgradeSteps.size() - 1);
        if (!upgradeStep(upgradeSteps[i], isFinalStep))
            return;
    }
}

bool ProgressiveUpgradeThread::upgradeStep(int nextFFTSize, bool isFinalStep)
{
    if (!isGenerationValid())
        return false;

    const uint64_t stepKey = CacheManager::computeKey(irFile,
                                                      nextFFTSize,
                                                      sampleRate,
                                                      phaseMode,
                                                      targetLengthSamples);

    auto prepared = cacheManager.load(stepKey, nextFFTSize, taskGeneration);
    if (!prepared)
    {
        prepared = converter.convertToHighRes(irFile,
                                              sampleRate,
                                              nextFFTSize,
                                              targetLengthSamples,
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

    juce::MessageManager::callAsync([this, prepared = std::move(prepared), isFinalStep]() mutable
    {
        if (this->checkAndCancel())
            return;

        // Only update UI visualization on final step to avoid multiple updates
        if (!isFinalStep)
        {
            processor.setVisualizationEnabled(false);
            processor.applyPreparedIRState(std::move(prepared));
            processor.setVisualizationEnabled(true);
        }
        else
        {
            processor.applyPreparedIRState(std::move(prepared));
        }
    });

    return true;
}
