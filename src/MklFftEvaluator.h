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
        double timeDomainRms = 0.0;
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

        auto bandWeightForHz = [nyquistHz](double f) noexcept
        {
            // ITU-R BS.468-4 weighting approximation
            // Reference: https://en.wikipedia.org/wiki/ITU-R_BS.468_weighting
            if (f < 1.0) f = 1.0;
            
            const double f2 = f * f;
            const double h1 = -4.737338981378384e-24 * f2 * f2 * f2 + 2.043828333606125e-15 * f2 * f2 - 1.363894795463638e-7 * f2 + 1.0;
            const double h2 = 1.306612257402824e-19 * f2 * f2 * f - 2.118150887541247e-11 * f2 * f + 5.559488023498642e-4 * f;
            const double r_f = (1.246332637532143e-4 * f) / std::sqrt(h1 * h1 + h2 * h2);
            
            // Convert to power weight (squared)
            double w = r_f * r_f;
            
            // Add extra high-frequency roll-off if above 18kHz to avoid over-optimizing inaudible range
            if (f > 18000.0)
            {
                const double rollOff = std::pow(10.0, -12.0 * (f - 18000.0) / std::max(1000.0, nyquistHz - 18000.0) / 20.0);
                w *= rollOff * rollOff;
            }
            
            return std::max(1.0e-6, w);
        };

        for (int bin = 0; bin < kSpectrumBins; ++bin)
        {
            const double frequencyHz = static_cast<double>(bin) * binWidthHz;
            weights[static_cast<size_t>(bin)] = bandWeightForHz(frequencyHz);
            configuredWeightSum += weights[static_cast<size_t>(bin)];
        }
    }

    Result evaluate(const double* errorLeft, const double* errorRight, const std::vector<double>* maskingThresholds = nullptr) noexcept
    {
        static thread_local bool currentThreadConfigured = false;
        if (!currentThreadConfigured)
        {
            mkl_set_num_threads_local(1);
            currentThreadConfigured = true;
        }

        double sumSq = 0.0;
        for (int i = 0; i < kFftLength; ++i)
        {
            sumSq += 0.5 * (errorLeft[i] * errorLeft[i] + errorRight[i] * errorRight[i]);
        }
        const double timeRms = std::sqrt(sumSq / kFftLength);

        juce::FloatVectorOperations::copy(inputLeft, errorLeft, kFftLength);
        juce::FloatVectorOperations::copy(inputRight, errorRight, kFftLength);

        DftiComputeForward(descriptor, inputLeft, spectrumLeft);
        DftiComputeForward(descriptor, inputRight, spectrumRight);

        double weightedNoise = 0.0;
        double flatnessLogSum = 0.0;
        double flatnessPowerSum = 0.0;
        double highBandEnergy = 0.0;
        double ultraHighEnergy = 0.0;
        double peakEnergy = 0.0;
        double totalEnergy = 0.0;
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

            double effectiveNoise = averageMagSq;
            if (maskingThresholds != nullptr && bin < (int)maskingThresholds->size())
            {
                effectiveNoise = std::max(0.0, averageMagSq - (*maskingThresholds)[bin]);
            }

            weightedNoise += weights[static_cast<size_t>(bin)] * effectiveNoise;
            totalEnergy += averageMagSq;

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

            // Tonal detection: find peaks relative to neighbors
            if (bin > 0 && bin < kSpectrumBins - 1)
            {
                const double prevMagSq = 0.5 * ((spectrumLeft[bin-1].real * spectrumLeft[bin-1].real + spectrumLeft[bin-1].imag * spectrumLeft[bin-1].imag) +
                                                (spectrumRight[bin-1].real * spectrumRight[bin-1].real + spectrumRight[bin-1].imag * spectrumRight[bin-1].imag));
                const double nextMagSq = 0.5 * ((spectrumLeft[bin+1].real * spectrumLeft[bin+1].real + spectrumLeft[bin+1].imag * spectrumLeft[bin+1].imag) +
                                                (spectrumRight[bin+1].real * spectrumRight[bin+1].real + spectrumRight[bin+1].imag * spectrumRight[bin+1].imag));
                
                const double localAvg = (prevMagSq + averageMagSq + nextMagSq) / 3.0;
                if (averageMagSq > 6.0 * localAvg) // 6dB threshold
                    peakEnergy = std::max(peakEnergy, averageMagSq);
            }
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
        result.timeDomainRms = timeRms;

        // Tonal penalty (idle tone detection)
        const double tonalRatio = peakEnergy / (totalEnergy + kEpsilon);
        const double tonalPenalty = std::max(0.0, tonalRatio - 0.05) * 10.0;

        result.compositeScore = result.noisePower
                              * (1.0
                                 + (flatnessPenaltyWeight * result.spectralFlatnessPenalty)
                                 + (hfPenaltyWeight * result.hfPenalty)
                                 + tonalPenalty);
        return result;
    }

    double computeMaskingThreshold(double energy, double freq) const noexcept
    {
        // Very simplified masking threshold based on Bark scale
        // Reference: https://en.wikipedia.org/wiki/Bark_scale
        const double bark = 13.0 * std::atan(0.00076 * freq) + 3.5 * std::atan(std::pow(freq / 7500.0, 2.0));
        
        // Spreading function approximation (simplified)
        // Masking is stronger at higher frequencies
        const double offset = -15.0 - (bark * 0.5); // dB
        return energy * std::pow(10.0, offset / 10.0);
    }

    void computeFft(const double* dataL, const double* dataR, MKL_Complex16* outL, MKL_Complex16* outR) noexcept
    {
        juce::FloatVectorOperations::copy(inputLeft, dataL, kFftLength);
        juce::FloatVectorOperations::copy(inputRight, dataR, kFftLength);
        DftiComputeForward(descriptor, inputLeft, outL);
        DftiComputeForward(descriptor, inputRight, outR);
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
