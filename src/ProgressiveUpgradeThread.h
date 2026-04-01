#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include <JuceHeader.h>

class ConvolverProcessor;
class IRConverter;
class CacheManager;

class ProgressiveUpgradeThread : public juce::Thread
{
public:
    ProgressiveUpgradeThread(ConvolverProcessor& processor,
                             const juce::File& irFile,
                             double sampleRate,
                             int currentFFTSize,
                             int targetFFTSize,
                             int targetLengthSamples,
                             int phaseMode,
                             uint64_t baseGeneration,
                             uint64_t baseCacheKey,
                             IRConverter& converter,
                             CacheManager& cacheManager);

    ~ProgressiveUpgradeThread() override;

    void run() override;
    void cancel();

private:
    bool isGenerationValid() const;
    bool checkAndCancel();
    bool upgradeStep(int nextFFTSize, bool isFinalStep);

    ConvolverProcessor& processor;
    juce::File irFile;
    double sampleRate = 0.0;
    int currentFFTSize = 0;
    int targetFFTSize = 0;
    int targetLengthSamples = 0;
    int phaseMode = 0;
    uint64_t taskGeneration = 0;
    uint64_t baseCacheKey = 0;
    std::vector<int> upgradeSteps;
    std::atomic<bool> cancelled{false};

    IRConverter& converter;
    CacheManager& cacheManager;
};
