#pragma once

#include <atomic>
#include <memory>

#include <JuceHeader.h>

class ConvolverProcessor;
class IRConverter;
class CacheManager;

class ProgressiveUpgradeThread : public juce::Thread
{
public:
    ProgressiveUpgradeThread(ConvolverProcessor& processor,
                             const juce::File& irFile,
                             double targetSampleRate,
                             int targetFFTSize,
                             uint64_t baseGeneration,
                             uint64_t cacheKey,
                             IRConverter& converter,
                             CacheManager& cacheManager);

    ~ProgressiveUpgradeThread() override;

    void run() override;
    void cancel();

private:
    bool isGenerationValid() const;
    bool checkAndCancel();

    ConvolverProcessor& processor;
    juce::File irFile;
    double sampleRate = 0.0;
    int targetFFTSize = 0;
    uint64_t taskGeneration = 0;
    uint64_t cacheKey = 0;
    std::atomic<bool> cancelled{false};

    IRConverter& converter;
    CacheManager& cacheManager;
};
