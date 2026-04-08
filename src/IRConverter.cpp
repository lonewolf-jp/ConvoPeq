#include "IRConverter.h"
#include "IRDSP.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <mkl.h>
#include <mkl_cblas.h>

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
        converted = IRDSP::resampleIR(ir, sourceRate, config.targetSampleRate, shouldCancel);
        if (converted.getNumSamples() <= 0)
            return nullptr;
    }

    if (shouldCancel && shouldCancel())
        return nullptr;

    const int fftSize = juce::jmax(32, config.fftSize);
    const int usableChannels = juce::jmax(1, converted.getNumChannels());
    const int samples = converted.getNumSamples();

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
    prepared->numChannels = usableChannels;
    prepared->sampleRate = (config.targetSampleRate > 0.0) ? config.targetSampleRate : sourceRate;
    prepared->generationId = config.generationId;
    prepared->cacheKey = config.cacheKey;

    // 時間領域 IR を保持（UI 表示用）
    // converted はリサンプリング済みの加工済み IR
    prepared->timeDomainIR = std::make_unique<juce::AudioBuffer<double>>(std::move(converted));

    // LoaderThread::performLoad と同等の scaleFactor 計算
    if (prepared->timeDomainIR && prepared->timeDomainIR->getNumSamples() > 0)
    {
        double maxChannelEnergy = 0.0;
        const int numSamples = prepared->timeDomainIR->getNumSamples();
        const int numChannels = prepared->timeDomainIR->getNumChannels();

        for (int ch = 0; ch < numChannels; ++ch)
        {
            const double* chData = prepared->timeDomainIR->getReadPointer(ch);
            const double energy = cblas_ddot(numSamples, chData, 1, chData, 1);
            if (!std::isfinite(energy) || energy <= 1.0e-12)
                continue;
            if (energy > maxChannelEnergy)
                maxChannelEnergy = energy;
        }

        if (maxChannelEnergy > 1.0e-12 && std::isfinite(maxChannelEnergy))
        {
            const double makeup = 1.0 / std::sqrt(maxChannelEnergy);
            constexpr double safetyMargin = 0.5011872336272722;
            prepared->scaleFactor = makeup * safetyMargin;
            prepared->hasScaleFactor = true;
        }
    }

    return prepared;
}

std::unique_ptr<PreparedIRState> IRConverter::convertToHighRes(const juce::File& irFile,
                                                               double sampleRate,
                                                               int nextFFTSize,
                                                               uint64_t generationId,
                                                               uint64_t cacheKey,
                                                               const std::function<bool()>& shouldCancel) const
{
    ConvertConfig cfg;
    cfg.fftSize = nextFFTSize;
    cfg.partitionSize = nextFFTSize;
    cfg.targetSampleRate = sampleRate;
    cfg.generationId = generationId;
    cfg.cacheKey = cacheKey;
    return convertFile(irFile, cfg, shouldCancel);
}
