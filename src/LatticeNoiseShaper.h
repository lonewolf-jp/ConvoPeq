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

    struct CoefficientRamp
    {
        std::array<double, kOrder> current {};
        std::array<double, kOrder> target {};
        std::array<double, kOrder> delta {};
        int samplesRemaining = 0;
        static constexpr int kRampLength = 512;
    };

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
        
        ramp.samplesRemaining = 0;
    }

    void setCoefficients(const double* newCoeffs, int numCoeffs) noexcept
    {
        const int limit = std::min(kOrder, std::max(0, numCoeffs));

        for (int i = 0; i < limit; ++i)
            coeffs[static_cast<size_t>(i)] = clampCoeff(newCoeffs[i]);

        for (int i = limit; i < kOrder; ++i)
            coeffs[static_cast<size_t>(i)] = 0.0;
        
        ramp.current = coeffs;
        ramp.target = coeffs;
        ramp.samplesRemaining = 0;
    }

    void startCoefficientRamp(const double* newCoeffs) noexcept
    {
        for (int i = 0; i < kOrder; ++i)
        {
            ramp.target[static_cast<size_t>(i)] = clampCoeff(newCoeffs[i]);
            ramp.delta[static_cast<size_t>(i)] = (ramp.target[static_cast<size_t>(i)] - ramp.current[static_cast<size_t>(i)]) / CoefficientRamp::kRampLength;
        }
        ramp.samplesRemaining = CoefficientRamp::kRampLength;
    }

    void applyMatchedCoefficients(const double* newCoeffs, int numCoeffs) noexcept
    {
        // 新係数をクリップ
        std::array<double, kOrder> clampedCoeffs{};
        const int limit = std::min(kOrder, std::max(0, numCoeffs));
        for (int i = 0; i < limit; ++i)
            clampedCoeffs[static_cast<size_t>(i)] = clampCoeff(newCoeffs[i]);

        // ランプで徐々に切り替え（クリック回避）
        startCoefficientRamp(clampedCoeffs.data());
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

        for (int i = 0; i < numSamples; ++i)
        {
            if (ramp.samplesRemaining > 0)
            {
                for (int c = 0; c < kOrder; ++c)
                    ramp.current[static_cast<size_t>(c)] += ramp.delta[static_cast<size_t>(c)];
                
                if (--ramp.samplesRemaining == 0)
                    ramp.current = ramp.target;
                
                coeffs = ramp.current;
            }

            dataL[i] = processSample(0, dataL[i], states[0], headroom);
            if (dataR != nullptr)
                dataR[i] = processSample(1, dataR[i], states[1], headroom);
        }

        // 内部状態の飽和保護 (SIMD)
        for (int ch = 0; ch < kNumChannels; ++ch)
            clampStateSIMD(states[static_cast<size_t>(ch)].data());
    }

private:
    static constexpr double kStateLimit = 1.0e12;

    inline void clampStateSIMD(double* state) noexcept
    {
        const __m256d limit = _mm256_set1_pd(kStateLimit);
        const __m256d negLimit = _mm256_set1_pd(-kStateLimit);
        
        // kOrder is 9. 9 = 4 + 4 + 1.
        __m256d v0 = _mm256_loadu_pd(state);
        v0 = _mm256_min_pd(v0, limit);
        v0 = _mm256_max_pd(v0, negLimit);
        _mm256_storeu_pd(state, v0);

        __m256d v1 = _mm256_loadu_pd(state + 4);
        v1 = _mm256_min_pd(v1, limit);
        v1 = _mm256_max_pd(v1, negLimit);
        _mm256_storeu_pd(state + 4, v1);

        state[8] = std::clamp(state[8], -kStateLimit, kStateLimit);
    }
    static inline double clampCoeff(double value) noexcept
    {
        // 【修正】9th-order の安定性を優先し 0.85 に設定
        constexpr double kLimit = 0.85;
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

    // Xorshift64* 一様乱数 (周期 2^64-1)
    static inline uint64_t xorshift64star(uint64_t & state) noexcept
    {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    }

    inline double uniform(uint64_t & state) const noexcept
    {
        return static_cast<double>(xorshift64star(state)) / 18446744073709551616.0;
    }

    inline double quantize(double value, uint64_t & rng) const noexcept
    {
        const double minValue = -1.0;
        const double maxValue = 1.0 - (1.0 / invScale);

        // 【修正】TPDF dither を復活
        const double u1 = uniform(rng);
        const double u2 = uniform(rng);
        value += (u1 + u2 - 1.0) * scale;

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

        // State clamping limit to prevent explosion
        constexpr double kLatticeStateLimit = 2.0;

        for (int i = 0; i < kOrder; ++i)
        {
            const double backward = channelState[static_cast<size_t>(i)];
            const double nextForward = killDenormal(forward + coeffs[static_cast<size_t>(i)] * backward);
            const double nextBackward = killDenormal(coeffs[static_cast<size_t>(i)] * forward + backward);

            // Clamp state to prevent numerical instability
            channelState[static_cast<size_t>(i)] = std::clamp(prev_backward, -kLatticeStateLimit, kLatticeStateLimit);

            forward = nextForward;
            prev_backward = nextBackward;
        }
    }

    inline double processSample(int channel,
                                double inputSample,
                                std::array<double, kOrder>& channelState,
                                double headroom) noexcept
    {
        const double feedback = computeFeedback(channelState);
        const double shapedInput = (inputSample * headroom) + feedback;
        const double quantized = quantize(shapedInput, rngState[channel]);
        const double error = killDenormal(quantized - shapedInput);
        const double clampedError = std::clamp(error, -2.0 * scale, 2.0 * scale);
        advanceState(channelState, clampedError);
        return quantized;
    }

    std::array<double, kOrder> coeffs {};
    std::array<std::array<double, kOrder>, kNumChannels> states {};
    uint64_t rngState[kNumChannels] = {0x9e3779b97f4a7c15ULL, 0xbf58476d1ce4e5b9ULL};
    CoefficientRamp ramp;
    int currentBitDepth = 0;
    double scale = 1.0;
    double invScale = 1.0;
};
