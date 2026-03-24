#pragma once

#include <JuceHeader.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

class LatticeNoiseShaper
{
public:
    static constexpr int kOrder = 9;
    static constexpr int kNumChannels = 2;

    void prepare(int bitDepth) noexcept
    {
        currentBitDepth = bitDepth;

        if (bitDepth <= 0)
        {
            invScale = 1.0;
            scale = 1.0;
            reset();
            return;
        }

        const int safeBits = std::clamp(bitDepth, 1, 32);
        currentBitDepth = safeBits;
        invScale = std::ldexp(1.0, safeBits - 1);
        scale = 1.0 / invScale;
        reset();
    }

    void reset() noexcept
    {
        for (auto& channelState : states)
            juce::FloatVectorOperations::clear(channelState.data(), kOrder);
    }

    void setCoefficients(const double* newCoeffs, int numCoeffs) noexcept
    {
        const int limit = std::min(kOrder, std::max(0, numCoeffs));

        for (int i = 0; i < limit; ++i)
            coeffs[static_cast<size_t>(i)] = clampCoeff(newCoeffs[i]);

        for (int i = limit; i < kOrder; ++i)
            coeffs[static_cast<size_t>(i)] = 0.0;
    }

    void applyMatchedCoefficients(const double* newCoeffs, int numCoeffs) noexcept
    {
        std::array<double, kOrder> oldCoeffs = coeffs;
        std::array<double, kOrder> clampedCoeffs {};

        const int limit = std::min(kOrder, std::max(0, numCoeffs));
        for (int i = 0; i < limit; ++i)
            clampedCoeffs[static_cast<size_t>(i)] = clampCoeff(newCoeffs[i]);

        for (int channel = 0; channel < kNumChannels; ++channel)
            // matchState(oldCoeffs.data(), clampedCoeffs.data(), states[static_cast<size_t>(channel)].data());
            ;

        coeffs = clampedCoeffs;
    }

    const double* getCoefficients() const noexcept
    {
        return coeffs.data();
    }

    void processStereoBlock(double* dataL, double* dataR, int numSamples, double headroom) noexcept
    {
        if (dataL == nullptr || numSamples <= 0)
            return;

        if (currentBitDepth <= 0)
        {
            for (int i = 0; i < numSamples; ++i)
                dataL[i] *= headroom;

            if (dataR != nullptr)
                for (int i = 0; i < numSamples; ++i)
                    dataR[i] *= headroom;

            return;
        }

        auto& leftState = states[0];
        for (int i = 0; i < numSamples; ++i)
            dataL[i] = processSample(dataL[i], leftState, headroom);

        if (dataR != nullptr)
        {
            auto& rightState = states[1];
            for (int i = 0; i < numSamples; ++i)
                dataR[i] = processSample(dataR[i], rightState, headroom);
        }
    }

private:
    static inline double clampCoeff(double value) noexcept
    {
        constexpr double kLimit = 0.995;
        if (std::isnan(value))
            return 0.0;
        if (value > kLimit)
            return kLimit;
        if (value < -kLimit)
            return -kLimit;
        return value;
    }

    static inline double absNoLibm(double x) noexcept
    {
        union { double d; std::uint64_t u; } value { x };
        value.u &= 0x7fffffffffffffffULL;
        return value.d;
    }

    static inline double killDenormal(double x) noexcept
    {
        return absNoLibm(x) < 1.0e-24 ? 0.0 : x;
    }

    static void matchState(const double* oldCoeffs, const double* newCoeffs, double* state) noexcept
    {
        double temp[kOrder] = {};

        // Step 1: Extract "forward" errors from "backward" errors (state) using old coefficients
        for (int i = kOrder - 1; i >= 0; --i)
        {
            double value = state[i];
            for (int j = i + 1; j < kOrder; ++j)
                value -= oldCoeffs[j] * temp[j];

            temp[i] = value;
        }

        // Step 2: Reconstruct new "backward" errors (state) from "forward" errors using new coefficients
        // BUG FIX: Use temp[j] instead of state[j] to avoid using partially updated values.
        for (int i = 0; i < kOrder; ++i)
        {
            double value = temp[i];
            for (int j = 0; j < i; ++j)
                value += newCoeffs[j] * temp[j];

            state[i] = killDenormal(value);
        }
    }

    inline double quantize(double value) const noexcept
    {
        const double minValue = -1.0;
        const double maxValue = 1.0 - (1.0 / invScale);

        if (value < minValue)
            value = minValue;
        else if (value > maxValue)
            value = maxValue;

        __m128d rounded = _mm_set_sd(value * invScale);
        rounded = _mm_round_sd(rounded, rounded, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        return _mm_cvtsd_f64(rounded) * scale;
    }

    inline double computeFeedback(const std::array<double, kOrder>& channelState) const noexcept
    {
        double feedback = 0.0;

        for (int i = 0; i < kOrder; ++i)
            feedback = killDenormal(feedback + coeffs[static_cast<size_t>(i)] * channelState[static_cast<size_t>(i)]);

        return feedback;
    }

    inline void advanceState(std::array<double, kOrder>& channelState, double error) const noexcept
    {
        double forward = error;
        double prev_backward = error;

        for (int i = 0; i < kOrder; ++i)
        {
            const double backward = channelState[static_cast<size_t>(i)];
            const double nextForward = killDenormal(forward + coeffs[static_cast<size_t>(i)] * backward);
            const double nextBackward = killDenormal(coeffs[static_cast<size_t>(i)] * forward + backward);

            channelState[static_cast<size_t>(i)] = prev_backward;

            forward = nextForward;
            prev_backward = nextBackward;
        }
    }

    inline double processSample(double inputSample,
                                std::array<double, kOrder>& channelState,
                                double headroom) noexcept
    {
        const double feedback = computeFeedback(channelState);
        const double shapedInput = (inputSample * headroom) + feedback;
        const double quantized = quantize(shapedInput);
        const double error = killDenormal(quantized - shapedInput);
        advanceState(channelState, error);
        return quantized;
    }

    std::array<double, kOrder> coeffs {};
    std::array<std::array<double, kOrder>, kNumChannels> states {};
    int currentBitDepth = 0;
    double scale = 1.0;
    double invScale = 1.0;
};
