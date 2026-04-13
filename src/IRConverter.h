#pragma once

#include <functional>
#include <memory>

#include <JuceHeader.h>

#include "PreparedIRState.h"

class IRConverter
{
public:
    struct ScaleFactorResult
    {
        double scaleFactor = 1.0;
        bool hasScaleFactor = false;
    };

    struct ConvertConfig
    {
        int fftSize = 512;
        int phaseMode = 0;
        int partitionSize = 512;
        double targetSampleRate = 0.0;
        uint64_t generationId = 0;
        uint64_t cacheKey = 0;
    };

    std::unique_ptr<PreparedIRState> convertFile(const juce::File& irFile,
                                                 const ConvertConfig& config,
                                                 const std::function<bool()>& shouldCancel) const;

    std::unique_ptr<PreparedIRState> convertToHighRes(const juce::File& irFile,
                                                      double sampleRate,
                                                      int nextFFTSize,
                                                      uint64_t generationId,
                                                      uint64_t cacheKey,
                                                      const std::function<bool()>& shouldCancel) const;

    static ScaleFactorResult computeScaleFactor(const juce::AudioBuffer<double>& ir,
                                                const juce::AudioBuffer<double>* currentIr = nullptr,
                                                double currentScale = 1.0) noexcept;

private:
    static bool loadAudioFile(const juce::File& file,
                              juce::AudioBuffer<double>& out,
                              double& sampleRateOut);
};
