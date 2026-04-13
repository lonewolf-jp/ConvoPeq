#include "IRConverter.h"
#include "IRDSP.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <mkl.h>
#include <mkl_cblas.h>

IRConverter::ScaleFactorResult IRConverter::computeScaleFactor(const juce::AudioBuffer<double>& ir,
                                                               const juce::AudioBuffer<double>* currentIr,
                                                               double currentScale) noexcept
{
    ScaleFactorResult result;

    const int numSamples = ir.getNumSamples();
    const int numChannels = ir.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0)
        return result;

    double maxChannelEnergy = 0.0;
    double irPeak = 0.0;
    double irEnergySum = 0.0;

    for (int ch = 0; ch < numChannels; ++ch)
    {
        const double* data = ir.getReadPointer(ch);
        const double energy = cblas_ddot(numSamples, data, 1, data, 1);
        if (std::isfinite(energy) && energy > 1.0e-18)
            maxChannelEnergy = std::max(maxChannelEnergy, energy);

        for (int i = 0; i < numSamples; ++i)
        {
            const double value = data[i];
            irPeak = std::max(irPeak, std::abs(value));
            irEnergySum += value * value;
        }
    }

    if (!(maxChannelEnergy > 1.0e-18) || !std::isfinite(maxChannelEnergy))
        return result;

    const double makeup = 1.0 / std::sqrt(maxChannelEnergy);
    constexpr double safetyMargin = 0.5011872336272722;
    result.scaleFactor = makeup * safetyMargin;
    result.hasScaleFactor = true;

    const int totalSamples = numChannels * numSamples;
    if (totalSamples > 0)
    {
        const double irRms = std::sqrt(irEnergySum / static_cast<double>(totalSamples));
        constexpr double kMaxEffectivePeak = 0.98;
        constexpr double kMaxEffectiveRms = 0.25;

        double absoluteClamp = 1.0;
        if (irPeak > 1.0e-12)
            absoluteClamp = std::min(absoluteClamp, kMaxEffectivePeak / (irPeak * result.scaleFactor));
        if (irRms > 1.0e-12)
            absoluteClamp = std::min(absoluteClamp, kMaxEffectiveRms / (irRms * result.scaleFactor));

        if (std::isfinite(absoluteClamp) && absoluteClamp > 0.0 && absoluteClamp < 1.0)
            result.scaleFactor *= absoluteClamp;
    }

    if (currentIr != nullptr && currentIr->getNumChannels() > 0 && currentIr->getNumSamples() > 0)
    {
        auto computePeakAndRmsWithScale = [](const juce::AudioBuffer<double>& buffer, double scale) -> std::pair<double, double>
        {
            const int channels = buffer.getNumChannels();
            const int samples = buffer.getNumSamples();
            if (channels <= 0 || samples <= 0)
                return { 0.0, 0.0 };

            double peak = 0.0;
            double energy = 0.0;
            for (int ch = 0; ch < channels; ++ch)
            {
                const double* data = buffer.getReadPointer(ch);
                for (int i = 0; i < samples; ++i)
                {
                    const double value = data[i] * scale;
                    peak = std::max(peak, std::abs(value));
                    energy += value * value;
                }
            }

            return { peak, std::sqrt(energy / static_cast<double>(channels * samples)) };
        };

        const auto [currentPeak, currentRms] = computePeakAndRmsWithScale(*currentIr, currentScale);
        const auto [newPeak, newRms] = computePeakAndRmsWithScale(ir, result.scaleFactor);

        const bool excessivePeakJump = currentPeak > 1.0e-9 && newPeak > currentPeak * 4.0 && newPeak > 0.5;
        const bool excessiveRmsJump = currentRms > 1.0e-9 && newRms > currentRms * 4.0 && newRms > 0.25;

        if (excessivePeakJump || excessiveRmsJump)
        {
            double clampByPeak = std::numeric_limits<double>::infinity();
            double clampByRms = std::numeric_limits<double>::infinity();

            if (newPeak > 1.0e-12 && currentPeak > 1.0e-12)
                clampByPeak = (currentPeak * 4.0) / newPeak;
            if (newRms > 1.0e-12 && currentRms > 1.0e-12)
                clampByRms = (currentRms * 4.0) / newRms;

            const double clampRatio = std::min(clampByPeak, clampByRms);
            if (std::isfinite(clampRatio) && clampRatio > 0.0 && clampRatio < 1.0)
                result.scaleFactor *= clampRatio;
        }
    }

    return result;
}

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

    if (prepared->timeDomainIR && prepared->timeDomainIR->getNumSamples() > 0)
    {
        const auto scaleInfo = computeScaleFactor(*prepared->timeDomainIR);
        prepared->scaleFactor = scaleInfo.scaleFactor;
        prepared->hasScaleFactor = scaleInfo.hasScaleFactor;
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
