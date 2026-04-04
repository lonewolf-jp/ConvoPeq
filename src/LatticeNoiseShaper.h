#pragma once

#include <JuceHeader.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include "DspNumericPolicy.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4324)
#endif

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

        int i = 0;
        const int rampSamples = std::min(numSamples, ramp.samplesRemaining);

        if (rampSamples > 0)
        {
            if (dataR != nullptr)
            {
                for (; i < rampSamples; ++i)
                {
                    stepRampCurrent();
                    const double* activeCoeffs = ramp.current.data();
                    dataL[i] = processSample(0, dataL[i], states[0], activeCoeffs, headroom);
                    dataR[i] = processSample(1, dataR[i], states[1], activeCoeffs, headroom);
                }
            }
            else
            {
                for (; i < rampSamples; ++i)
                {
                    stepRampCurrent();
                    const double* activeCoeffs = ramp.current.data();
                    dataL[i] = processSample(0, dataL[i], states[0], activeCoeffs, headroom);
                }
            }

            ramp.samplesRemaining -= rampSamples;
            if (ramp.samplesRemaining == 0)
            {
                ramp.current = ramp.target;
                coeffs = ramp.target;
            }
        }

        const double* activeCoeffs = coeffs.data();
        if (dataR != nullptr)
        {
            for (; i < numSamples; ++i)
            {
                dataL[i] = processSample(0, dataL[i], states[0], activeCoeffs, headroom);
                dataR[i] = processSample(1, dataR[i], states[1], activeCoeffs, headroom);
            }
        }
        else
        {
            for (; i < numSamples; ++i)
            {
                dataL[i] = processSample(0, dataL[i], states[0], activeCoeffs, headroom);
            }
        }

        clampStateSIMD(states[0].data());
        if (dataR != nullptr)
            clampStateSIMD(states[1].data());
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

    // 安全マージン付き clampCoeff（外部から上限指定可能）
    static inline double clampCoeff(double value, double margin) noexcept
    {
        const double kLimit = margin;
        if (std::isnan(value))
            return 0.0;
        if (value > kLimit) return kLimit;
        if (value < -kLimit) return -kLimit;
        return value;
    }

    // 安定性簡易チェック（格子フィルタ理論に基づく）
    static bool isStable(const double* parcor, int order) noexcept
    {
        // 反射係数の絶対値が全て 1 未満であることを確認
        for (int i = 0; i < order; ++i)
        {
            if (std::abs(parcor[i]) >= 1.0 - 1e-12)
                return false;
        }
        return true;  // 格子フィルタは反射係数が全て |k|<1 であれば安定
    }

private:
    static constexpr double kStateLimit = 1.0e12;

    inline void stepRampCurrent() noexcept
    {
        double* current = ramp.current.data();
        const double* delta = ramp.delta.data();

        __m256d cur0 = _mm256_loadu_pd(current);
        __m256d del0 = _mm256_loadu_pd(delta);
        _mm256_storeu_pd(current, _mm256_add_pd(cur0, del0));

        __m256d cur1 = _mm256_loadu_pd(current + 4);
        __m256d del1 = _mm256_loadu_pd(delta + 4);
        _mm256_storeu_pd(current + 4, _mm256_add_pd(cur1, del1));

        current[8] += delta[8];
    }

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

    struct Xoshiro256State
    {
        uint64_t s[4];
    };

    static inline uint64_t rotl(const uint64_t x, int k) noexcept
    {
        return (x << k) | (x >> (64 - k));
    }

    // Xoshiro256++ 1.0 (周期 2^256-1)
    static inline uint64_t xoshiro256plusplus(Xoshiro256State& state) noexcept
    {
        const uint64_t result = rotl(state.s[0] + state.s[3], 23) + state.s[0];
        const uint64_t t = state.s[1] << 17;
        state.s[2] ^= state.s[0];
        state.s[3] ^= state.s[1];
        state.s[1] ^= state.s[2];
        state.s[0] ^= state.s[3];
        state.s[2] ^= t;
        state.s[3] = rotl(state.s[3], 45);
        return result;
    }

    inline double uniform(Xoshiro256State& state) const noexcept
    {
        // 64bit 整数を [0, 1) の double に変換 (53bit 精度)
        return (xoshiro256plusplus(state) >> 11) * (1.0 / 9007199254740992.0);
    }

    inline double quantize(double value, Xoshiro256State& rng) const noexcept
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

    inline double computeFeedback(const std::array<double, kOrder>& channelState,
                                  const double* activeCoeffs) const noexcept
    {
        const double* state = channelState.data();
        __m256d v0 = _mm256_loadu_pd(state);
        __m256d c0 = _mm256_loadu_pd(activeCoeffs);
        __m256d vSum = _mm256_mul_pd(v0, c0);

        __m256d v1 = _mm256_loadu_pd(state + 4);
        __m256d c1 = _mm256_loadu_pd(activeCoeffs + 4);
        vSum = _mm256_fmadd_pd(v1, c1, vSum);

        // Horizontal add
        __m128d vLow = _mm256_castpd256_pd128(vSum);
        __m128d vHigh = _mm256_extractf128_pd(vSum, 1);
        __m128d vSum128 = _mm_add_pd(vLow, vHigh);
        vSum128 = _mm_hadd_pd(vSum128, vSum128);
        double feedback = _mm_cvtsd_f64(vSum128);

        feedback += state[8] * activeCoeffs[8];
        return feedback;
    }

    inline void advanceState(std::array<double, kOrder>& channelState,
                             double error,
                             const double* activeCoeffs) const noexcept
    {
        double forward = error;
        double prev_backward = error;
        double* state = channelState.data();

        // State clamping limit to prevent explosion
        constexpr double kLatticeStateLimit = 2.0;

        for (int i = 0; i < kOrder; ++i)
        {
            const double backward = state[i];
            const double nextForward = forward + activeCoeffs[i] * backward;
            const double nextBackward = activeCoeffs[i] * forward + backward;

            // Clamp state to prevent numerical instability
            state[i] = std::clamp(prev_backward, -kLatticeStateLimit, kLatticeStateLimit);

            forward = nextForward;
            prev_backward = nextBackward;
        }
    }

    inline double processSample(int channel,
                                double inputSample,
                                std::array<double, kOrder>& channelState,
                                const double* activeCoeffs,
                                double headroom) noexcept
    {
        const double feedback = computeFeedback(channelState, activeCoeffs);
        const double shapedInputClean = (inputSample * headroom) + feedback;
        const double quantized = quantize(shapedInputClean, rngState[channel]);
        const double error = quantized - shapedInputClean;
        const double clampedError = std::clamp(error, -2.0 * scale, 2.0 * scale);
        advanceState(channelState, clampedError, activeCoeffs);
        return quantized;
    }

    alignas(64) std::array<double, kOrder> coeffs {};
    alignas(64) std::array<std::array<double, kOrder>, kNumChannels> states {};
    Xoshiro256State rngState[kNumChannels] = {
        {{ 0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL, 0x0123456789ABCDEFULL, 0xEFCDAB8967452301ULL }},
        {{ 0x89ABCDEF01234567ULL, 0x76543210FEDCBA98ULL, 0xABCDEF0123456789ULL, 0x67452301EFCDAB89ULL }}
    };
    CoefficientRamp ramp;
    int currentBitDepth = 0;
    double scale = 1.0;
    double invScale = 1.0;
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
