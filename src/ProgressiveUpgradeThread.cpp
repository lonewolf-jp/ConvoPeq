#include "ProgressiveUpgradeThread.h"

#include <Windows.h>

#include "ConvolverProcessor.h"
#include "IRConverter.h"
#include "CacheManager.h"
#include "PreparedIRState.h"

ProgressiveUpgradeThread::ProgressiveUpgradeThread(ConvolverProcessor& p,
                                                   const juce::File& file,
                                                   double targetSR,
                                                   int fft,
                                                   uint64_t baseGeneration,
                                                   uint64_t key,
                                                   IRConverter& conv,
                                                   CacheManager& cache)
    : juce::Thread("ConvolverProgressiveUpgrade"),
      processor(p),
      irFile(file),
      sampleRate(targetSR),
      targetFFTSize(fft),
      taskGeneration(baseGeneration),
      cacheKey(key),
      converter(conv),
      cacheManager(cache)
{
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

    IRConverter::ConvertConfig cfg;
    cfg.fftSize = targetFFTSize;
    cfg.targetSampleRate = sampleRate;
    cfg.generationId = taskGeneration;
    cfg.cacheKey = cacheKey;

    auto prepared = converter.convertFile(irFile, cfg, [this]()
    {
        return this->threadShouldExit() || this->checkAndCancel();
    });

    if (!prepared || checkAndCancel())
        return;

    cacheManager.save(cacheKey, *prepared);

    juce::MessageManager::callAsync([this, prepared = std::move(prepared)]() mutable
    {
        if (this->checkAndCancel())
            return;

        processor.applyPreparedIRState(std::move(prepared));
    });
}
