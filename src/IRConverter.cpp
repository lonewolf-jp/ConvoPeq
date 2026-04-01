#include "IRConverter.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <mkl.h>

bool IRConverter::loadAudioFile(const juce::File& file,
                                juce::AudioBuffer<double>& out,
                                double& sampleRateOut)
{
    if (!file.existsAsFile())
        return false;

    juce::AudioFormatManager fm;
    fm.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader(fm.createReaderFor(file));
    if (!reader)
        return false;

    const int64 n = reader->lengthInSamples;
    if (n <= 0 || n > static_cast<int64>(std::numeric_limits<int>::max()))
        return false;

    const int channels = static_cast<int>(reader->numChannels);
    if (channels <= 0)
        return false;

    juce::AudioBuffer<float> temp(channels, static_cast<int>(n));
    if (!reader->read(&temp, 0, static_cast<int>(n), 0, true, true))
        return false;

    out.setSize(channels, static_cast<int>(n));
    for (int ch = 0; ch < channels; ++ch)
    {
        const float* src = temp.getReadPointer(ch);
        double* dst = out.getWritePointer(ch);
        for (int i = 0; i < static_cast<int>(n); ++i)
            dst[i] = static_cast<double>(src[i]);
    }

    sampleRateOut = reader->sampleRate;
    return true;
}

juce::AudioBuffer<double> IRConverter::resampleLinear(const juce::AudioBuffer<double>& input,
                                                      double srcRate,
                                                      double dstRate,
                                                      const std::function<bool()>& shouldCancel)
{
    if (srcRate <= 0.0 || dstRate <= 0.0 || std::abs(srcRate - dstRate) < 1.0e-6)
        return input;

    const double ratio = dstRate / srcRate;
    const int inN = input.getNumSamples();
    const int outN = juce::jmax(1, static_cast<int>(std::ceil(static_cast<double>(inN) * ratio)));

    juce::AudioBuffer<double> out(input.getNumChannels(), outN);
    out.clear();

    for (int ch = 0; ch < input.getNumChannels(); ++ch)
    {
        const double* src = input.getReadPointer(ch);
        double* dst = out.getWritePointer(ch);

        for (int i = 0; i < outN; ++i)
        {
            if ((i & 0xF) == 0 && shouldCancel && shouldCancel())
                return {};

            const double pos = static_cast<double>(i) / ratio;
            const int i0 = juce::jlimit(0, inN - 1, static_cast<int>(std::floor(pos)));
            const int i1 = juce::jlimit(0, inN - 1, i0 + 1);
            const double frac = pos - static_cast<double>(i0);
            dst[i] = src[i0] * (1.0 - frac) + src[i1] * frac;
        }
    }

    return out;
}

std::unique_ptr<PreparedIRState> IRConverter::convertFile(const juce::File& irFile,
                                                          const ConvertConfig& config,
                                                          const std::function<bool()>& shouldCancel) const
{
    juce::AudioBuffer<double> ir;
    double sourceRate = 0.0;
    if (!loadAudioFile(irFile, ir, sourceRate))
        return nullptr;

    if (shouldCancel && shouldCancel())
        return nullptr;

    juce::AudioBuffer<double> converted = ir;
    if (config.targetSampleRate > 0.0 && sourceRate > 0.0 && std::abs(sourceRate - config.targetSampleRate) > 1.0e-6)
    {
        converted = resampleLinear(ir, sourceRate, config.targetSampleRate, shouldCancel);
        if (converted.getNumSamples() <= 0)
            return nullptr;
    }

    if (shouldCancel && shouldCancel())
        return nullptr;

    const int fftSize = juce::jmax(32, config.fftSize);
    const int usableChannels = juce::jmax(1, converted.getNumChannels());
    const int maxTargetSamples = (config.targetLengthSamples > 0)
                                   ? config.targetLengthSamples
                                   : converted.getNumSamples();
    const int samples = juce::jmin(converted.getNumSamples(), maxTargetSamples);

    const int numPartitions = juce::jmax(1, (samples + fftSize - 1) / fftSize);
    const size_t totalSamples = static_cast<size_t>(numPartitions) * static_cast<size_t>(fftSize) * static_cast<size_t>(usableChannels);
    const size_t bytes = totalSamples * sizeof(double);

    double* data = static_cast<double*>(mkl_malloc(bytes, 64));
    if (!data)
        return nullptr;

    std::memset(data, 0, bytes);

    for (int ch = 0; ch < usableChannels; ++ch)
    {
        const double* src = converted.getReadPointer(ch);
        for (int i = 0; i < samples; ++i)
        {
            if ((i & 0xF) == 0 && shouldCancel && shouldCancel())
            {
                mkl_free(data);
                return nullptr;
            }

            const size_t idx = static_cast<size_t>(ch) * static_cast<size_t>(numPartitions) * static_cast<size_t>(fftSize)
                             + static_cast<size_t>(i);
            data[idx] = src[i];
        }
    }

    auto prepared = std::make_unique<PreparedIRState>();
    prepared->partitionData = data;
    prepared->partitionSizeBytes = bytes;
    prepared->numPartitions = numPartitions * usableChannels;
    prepared->fftSize = fftSize;
    prepared->numSamples = samples;    // actual IR length before partition padding
    prepared->numChannels = usableChannels;
    prepared->sampleRate = (config.targetSampleRate > 0.0) ? config.targetSampleRate : sourceRate;
    prepared->generationId = config.generationId;
    prepared->cacheKey = config.cacheKey;
    prepared->originalFileName = irFile.getFileNameWithoutExtension();
    prepared->originalFilePath = irFile.getFullPathName();

    return prepared;
}

std::unique_ptr<PreparedIRState> IRConverter::convertToHighRes(const juce::File& irFile,
                                                               double sampleRate,
                                                               int nextFFTSize,
                                                               int targetLengthSamples,
                                                               uint64_t generationId,
                                                               uint64_t cacheKey,
                                                               const std::function<bool()>& shouldCancel) const
{
    ConvertConfig cfg;
    cfg.fftSize = nextFFTSize;
    cfg.partitionSize = nextFFTSize;
    cfg.targetLengthSamples = targetLengthSamples;
    cfg.targetSampleRate = sampleRate;
    cfg.generationId = generationId;
    cfg.cacheKey = cacheKey;
    return convertFile(irFile, cfg, shouldCancel);
}
