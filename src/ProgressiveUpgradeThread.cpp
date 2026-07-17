#include "ProgressiveUpgradeThread.h"

#include "ConvolverProcessor.h"
#include "IRConverter.h"
#include "CacheManager.h"
#include "PreparedIRState.h"
#include "core/ThreadAffinityManager.h"

#include <cmath>
#include <xmmintrin.h>   // _MM_SET_FLUSH_ZERO_MODE
#include <pmmintrin.h>   // _MM_SET_DENORMALS_ZERO_MODE

#include "audioengine/AtomicAccess.h"

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
    convo::publishAtomic(cancelled, true, std::memory_order_release);
    signalThreadShouldExit();
}

bool ProgressiveUpgradeThread::isGenerationValid() const
{
    return !convo::consumeAtomic(cancelled, std::memory_order_acquire)
        && processor.isConvolverGenerationCurrent(taskGeneration);
}

bool ProgressiveUpgradeThread::checkAndCancel()
{
    if (!isGenerationValid())
    {
        convo::publishAtomic(cancelled, true, std::memory_order_release);
        return true;
    }
    return false;
}

void ProgressiveUpgradeThread::run()
{
    // ★ Bug#4: FTZ/DAZ 有効化（専用スレッド: 設定のみ、RAII 保存＋復元は不要）
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

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

    auto prepared = cacheManager.loadPreparedState(stepKey, nextFFTSize, taskGeneration);
    if (!prepared)
    {
        juce::WeakReference<ConvolverProcessor> weakOwner(&processor);
        // cancelled へのローカル参照。convertToHighRes は同期的に実行されるため、
        // upgradeStep のスタックフレーム生存期間内で完結する。
        // cancel() が他スレッドから cancelled を true に設定すると、
        // ラムダ内の cancelledRef がそれを観測し、早期復帰する。
        std::atomic<bool>& cancelledRef = cancelled;
        const uint64_t expectedGeneration = taskGeneration;

        prepared = converter.convertToHighRes(irFile,
                                              sampleRate,
                                              nextFFTSize,
                                              taskGeneration,
                                              stepKey,
                                              [weakOwner, &cancelledRef, expectedGeneration]()
                                              {
                                                  auto* owner = weakOwner.get();
                                                  if (owner == nullptr)
                                                      return true;

                                                  return juce::Thread::currentThreadShouldExit()
                                                      || convo::consumeAtomic(cancelledRef, std::memory_order_acquire)
                                                      || !owner->isConvolverGenerationCurrent(expectedGeneration);
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
        juce::WeakReference<ConvolverProcessor> weakProcessor(&processor);
        const uint64_t expectedGeneration = taskGeneration;
        auto* preparedRaw = prepared.release();
        juce::MessageManager::callAsync([weakProcessor, expectedGeneration, preparedRaw]()
        {
            std::unique_ptr<PreparedIRState> prepared(preparedRaw);
            auto* owner = weakProcessor.get();
            if (owner == nullptr)
                return;

            if (!owner->isConvolverGenerationCurrent(expectedGeneration))
                return;

            owner->applyComputedIR(std::move(prepared));
        });
    }

    return true;
}
