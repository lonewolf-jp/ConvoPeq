#pragma once

#include <JuceHeader.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include <mkl.h>

class MklFftEvaluator
{
public:
    static constexpr int kFftLength = 4096;
    static constexpr int kSpectrumBins = (kFftLength / 2) + 1;
    static constexpr double kDefaultSampleRateHz = 48000.0;

    struct Result
    {
        double noisePower = 0.0;
        double spectralFlatnessPenalty = 0.0;
        double hfPenalty = 0.0;
        double compositeScore = 0.0;
    };

    MklFftEvaluator()
    {
        inputLeft = static_cast<double*>(mkl_malloc(sizeof(double) * kFftLength, 64));
        inputRight = static_cast<double*>(mkl_malloc(sizeof(double) * kFftLength, 64));
        spectrumLeft = static_cast<MKL_Complex16*>(mkl_malloc(sizeof(MKL_Complex16) * kSpectrumBins, 64));
        spectrumRight = static_cast<MKL_Complex16*>(mkl_malloc(sizeof(MKL_Complex16) * kSpectrumBins, 64));

        mkl_set_num_threads_local(1);

        DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_REAL, 1, kFftLength);
        DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(descriptor, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(descriptor, DFTI_FORWARD_SCALE, 1.0);
        DftiSetValue(descriptor, DFTI_THREAD_LIMIT, 1);
        DftiCommitDescriptor(descriptor);

        configureForSampleRate(kDefaultSampleRateHz);
    }

    ~MklFftEvaluator()
    {
        if (descriptor != nullptr)
            DftiFreeDescriptor(&descriptor);

        if (inputLeft != nullptr)
            mkl_free(inputLeft);

        if (inputRight != nullptr)
            mkl_free(inputRight);

        if (spectrumLeft != nullptr)
            mkl_free(spectrumLeft);

        if (spectrumRight != nullptr)
            mkl_free(spectrumRight);
    }

    void configureForSampleRate(double sampleRateHz) noexcept
    {
        const double safeSampleRateHz = std::max(1.0, sampleRateHz);
        const double nyquistHz = 0.5 * safeSampleRateHz;
        const double binWidthHz = nyquistHz / static_cast<double>(kSpectrumBins - 1);

        auto hzToBin = [binWidthHz](double hz) noexcept
        {
            if (binWidthHz <= 0.0)
                return 0;

            return std::clamp(static_cast<int>(std::lround(hz / binWidthHz)), 0, kSpectrumBins - 1);
        };

        configuredSampleRateHz = safeSampleRateHz;
        configuredWeightSum = 0.0;
        flatnessPenaltyWeight = 0.35;
        hfPenaltyWeight = std::clamp(0.20 * std::sqrt(48000.0 / safeSampleRateHz), 0.05, 0.20);

        double flatnessStartHz = std::min(12000.0, nyquistHz * 0.60);
        double flatnessEndHz = std::min(18000.0, nyquistHz * 0.82);
        if (flatnessEndHz <= flatnessStartHz + (binWidthHz * 8.0))
        {
            flatnessStartHz = nyquistHz * 0.50;
            flatnessEndHz = nyquistHz * 0.80;
        }

        flatnessStartBin = hzToBin(flatnessStartHz);
        flatnessEndBin = std::max(flatnessStartBin + 1, hzToBin(flatnessEndHz));

        double highBandStartHz = std::max(14000.0, nyquistHz * 0.60);
        if (highBandStartHz >= nyquistHz)
            highBandStartHz = nyquistHz * 0.60;

        double ultraHighStartHz = nyquistHz * 0.85;
        if (ultraHighStartHz <= highBandStartHz + (binWidthHz * 8.0))
            ultraHighStartHz = highBandStartHz + (binWidthHz * 8.0);

        highBandStartBin = hzToBin(highBandStartHz);
        ultraHighStartBin = std::max(highBandStartBin + 1, hzToBin(ultraHighStartHz));

        const int highBandBins = std::max(1, kSpectrumBins - highBandStartBin);
        const int ultraHighBins = std::max(1, kSpectrumBins - ultraHighStartBin);
        expectedUltraHighShare = static_cast<double>(ultraHighBins) / static_cast<double>(highBandBins);

        auto lerp = [](double a, double b, double t) noexcept
        {
            return a + ((b - a) * std::clamp(t, 0.0, 1.0));
        };

        auto bandWeightForHz = [&lerp, nyquistHz](double frequencyHz) noexcept
        {
            if (frequencyHz <= 250.0)
                return 5.5;

            if (frequencyHz <= 1000.0)
                return lerp(5.5, 4.4, (frequencyHz - 250.0) / 750.0);

            if (frequencyHz <= 4000.0)
                return lerp(4.4, 2.8, (frequencyHz - 1000.0) / 3000.0);

            if (frequencyHz <= 8000.0)
                return lerp(2.8, 1.8, (frequencyHz - 4000.0) / 4000.0);

            if (frequencyHz <= 12000.0)
                return lerp(1.8, 1.25, (frequencyHz - 8000.0) / 4000.0);

            if (frequencyHz <= 18000.0)
                return lerp(1.25, 0.85, (frequencyHz - 12000.0) / 6000.0);

            if (nyquistHz <= 18000.0)
                return 0.85;

            return lerp(0.85, 0.60, (frequencyHz - 18000.0) / std::max(nyquistHz - 18000.0, 1.0));
        };

        for (int bin = 0; bin < kSpectrumBins; ++bin)
        {
            const double frequencyHz = static_cast<double>(bin) * binWidthHz;
            weights[static_cast<size_t>(bin)] = bandWeightForHz(frequencyHz);
            configuredWeightSum += weights[static_cast<size_t>(bin)];
        }
    }

    Result evaluate(const double* errorLeft, const double* errorRight) noexcept
    {
        static thread_local bool currentThreadConfigured = false;
        if (!currentThreadConfigured)
        {
            mkl_set_num_threads_local(1);
            currentThreadConfigured = true;
        }

        juce::FloatVectorOperations::copy(inputLeft, errorLeft, kFftLength);
        juce::FloatVectorOperations::copy(inputRight, errorRight, kFftLength);

        DftiComputeForward(descriptor, inputLeft, spectrumLeft);
        DftiComputeForward(descriptor, inputRight, spectrumRight);

        double weightedNoise = 0.0;
        double flatnessLogSum = 0.0;
        double flatnessPowerSum = 0.0;
        double highBandEnergy = 0.0;
        double ultraHighEnergy = 0.0;
        int flatnessBins = 0;
        constexpr double kEpsilon = 1.0e-24;

        for (int bin = 0; bin < kSpectrumBins; ++bin)
        {
            const double magSqLeft = (spectrumLeft[bin].real * spectrumLeft[bin].real)
                                   + (spectrumLeft[bin].imag * spectrumLeft[bin].imag);
            const double magSqRight = (spectrumRight[bin].real * spectrumRight[bin].real)
                                    + (spectrumRight[bin].imag * spectrumRight[bin].imag);
            const double averageMagSq = 0.5 * (magSqLeft + magSqRight);
            const double safeMagSq = averageMagSq + kEpsilon;

            weightedNoise += weights[static_cast<size_t>(bin)] * averageMagSq;

            if (bin >= flatnessStartBin && bin <= flatnessEndBin)
            {
                flatnessLogSum += std::log(safeMagSq);
                flatnessPowerSum += safeMagSq;
                ++flatnessBins;
            }

            if (bin >= highBandStartBin)
                highBandEnergy += averageMagSq;

            if (bin >= ultraHighStartBin)
                ultraHighEnergy += averageMagSq;
        }

        Result result;
        result.noisePower = weightedNoise / std::max(configuredWeightSum, kEpsilon);

        if (flatnessBins > 0)
        {
            const double arithmeticMean = flatnessPowerSum / static_cast<double>(flatnessBins);
            const double geometricMean = std::exp(flatnessLogSum / static_cast<double>(flatnessBins));
            const double flatness = std::clamp(geometricMean / std::max(arithmeticMean, kEpsilon), 0.0, 1.0);
            result.spectralFlatnessPenalty = 1.0 - flatness;
        }

        const double observedUltraHighShare = ultraHighEnergy / std::max(highBandEnergy + kEpsilon, kEpsilon);
        const double excessUltraHighShare = std::max(0.0, observedUltraHighShare - expectedUltraHighShare);
        result.hfPenalty = excessUltraHighShare / std::max(1.0 - expectedUltraHighShare, kEpsilon);
        result.compositeScore = result.noisePower
                              * (1.0
                                 + (flatnessPenaltyWeight * result.spectralFlatnessPenalty)
                                 + (hfPenaltyWeight * result.hfPenalty));
        return result;
    }

private:
    double* inputLeft = nullptr;
    double* inputRight = nullptr;
    MKL_Complex16* spectrumLeft = nullptr;
    MKL_Complex16* spectrumRight = nullptr;
    std::array<double, kSpectrumBins> weights {};
    DFTI_DESCRIPTOR_HANDLE descriptor = nullptr;
    double configuredSampleRateHz = kDefaultSampleRateHz;
    int flatnessStartBin = 0;
    int flatnessEndBin = kSpectrumBins - 1;
    int highBandStartBin = 0;
    int ultraHighStartBin = kSpectrumBins - 1;
    double expectedUltraHighShare = 0.0;
    double configuredWeightSum = 1.0;
    double flatnessPenaltyWeight = 0.35;
    double hfPenaltyWeight = 0.20;
};
