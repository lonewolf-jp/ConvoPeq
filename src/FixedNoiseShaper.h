//============================================================================
#pragma once
// FixedNoiseShaper.h
// 4-tap error-feedback noise shaper (RT-safe, allocation-free in process)
//============================================================================

#include <JuceHeader.h>
#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace convo
{

class FixedNoiseShaper
{
public:
    static constexpr int MAX_CHANNELS = 8;
    static constexpr int ORDER = 4;

    struct Diagnostics
    {
        float rmsErrorL = 0.0f;
        float rmsErrorR = 0.0f;
        float peakAbsError = 0.0f;
        uint32_t windowSamples = 0;
        int bitDepth = 0;
    };

    bool setCoefficients(const std::array<double, ORDER>& newCoeffs) noexcept
    {
        const double sum = newCoeffs[0] + newCoeffs[1] + newCoeffs[2] + newCoeffs[3];
        if (std::abs(sum - 1.0) > 1.0e-12)
            return false;

        coeffs = newCoeffs;
        return true;
    }

    void setDiagnosticsWindowSamples(uint32_t samples) noexcept
    {
        const uint32_t clamped = std::clamp<uint32_t>(samples, 256u, 262144u);
        publishWindowSamples.store(clamped, std::memory_order_relaxed);
    }

    uint32_t getDiagnosticsWindowSamples() const noexcept
    {
        return publishWindowSamples.load(std::memory_order_relaxed);
    }

    Diagnostics getDiagnostics() const noexcept
    {
        Diagnostics d;
        const float meanSqL = meanSqErrorL.load(std::memory_order_relaxed);
        const float meanSqR = meanSqErrorR.load(std::memory_order_relaxed);
        d.rmsErrorL = std::sqrt(std::max(0.0f, meanSqL));
        d.rmsErrorR = std::sqrt(std::max(0.0f, meanSqR));
        d.peakAbsError = peakAbsError.load(std::memory_order_relaxed);
        d.windowSamples = windowSamples.load(std::memory_order_relaxed);
        d.bitDepth = currentBitDepth;
        return d;
    }

    void prepare(double sampleRate, int bitDepth) noexcept
    {
        // プリセット選択と補間
        int idxLow = 0, idxHigh = 0;
        double t = 0.0;
        selectPresetWithInterpolation(sampleRate, idxLow, idxHigh, t);

        std::array<double, ORDER> interpCoeffs;
        if (t < 1e-12) {
            interpCoeffs = COEFF_PRESETS[idxLow];
        } else if (t > 1.0 - 1e-12) {
            interpCoeffs = COEFF_PRESETS[idxHigh];
        } else {
            const auto& cLow = COEFF_PRESETS[idxLow];
            const auto& cHigh = COEFF_PRESETS[idxHigh];
            for (int i = 0; i < ORDER; ++i) {
                interpCoeffs[i] = (1.0 - t) * cLow[i] + t * cHigh[i];
            }
        }
        setCoefficients(interpCoeffs);

        currentBitDepth = bitDepth;
        if (bitDepth <= 0)
        {
            invScale = 1.0;
            scale = 1.0;
            reset();
            return;
        }

        // デバイス設定の異常値に備えて安全域にクランプ (通常運用は16/24/32bit)。
        const int safeBits = std::clamp(bitDepth, 1, 32);
        currentBitDepth = safeBits;

        // 2^(bits-1) を安全に計算する (シフト由来の未定義動作を回避)。
        invScale = std::ldexp(1.0, safeBits - 1);
        scale = 1.0 / invScale;
        reset();
    }

    void reset() noexcept
    {
        for (auto& channelState : errors)
            juce::FloatVectorOperations::clear(channelState.data(), ORDER);
        writePos.fill(0);
        resetDiagnostics();
    }

    inline void processStereoBlock(double* dataL, double* dataR, int numSamples, double headroom) noexcept
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

        double sumSqL = 0.0;
        double sumSqR = 0.0;
        double peakAbs = 0.0;

        for (int i = 0; i < numSamples; ++i)
        {
            double error = 0.0;
            dataL[i] = processSample(dataL[i] * headroom, 0, error);
            sumSqL += error * error;
            const double absErr = absNoLibm(error);
            if (absErr > peakAbs)
                peakAbs = absErr;
        }

        if (dataR != nullptr)
            for (int i = 0; i < numSamples; ++i)
            {
                double error = 0.0;
                dataR[i] = processSample(dataR[i] * headroom, 1, error);
                sumSqR += error * error;
                const double absErr = absNoLibm(error);
                if (absErr > peakAbs)
                    peakAbs = absErr;
            }

        publishDiagnostics(sumSqL, sumSqR, peakAbs, static_cast<uint32_t>(numSamples), dataR != nullptr);
    }

private:
    inline double processSample(double x, int channel, double& outError) noexcept
    {
        auto& channelErrors = errors[static_cast<size_t>(channel)];
        int& idx = writePos[static_cast<size_t>(channel)];

        const double fb = coeffs[0] * get(channelErrors, idx, 0)
                        + coeffs[1] * get(channelErrors, idx, 1)
                        + coeffs[2] * get(channelErrors, idx, 2)
                        + coeffs[3] * get(channelErrors, idx, 3);

        const double y = x - fb;
        const double yq = quantize(y, rngState[static_cast<size_t>(channel)]);
        const double error = yq - y;
        outError = error;

        const double clampedError = std::clamp(error, -2.0 * scale, 2.0 * scale);
        idx = (idx - 1 + ORDER) % ORDER;
        channelErrors[static_cast<size_t>(idx)] = clampedError;

        return yq;
    }

    inline void publishDiagnostics(double sumSqLBlock, double sumSqRBlock, double peakAbsBlock,
                                   uint32_t sampleCountBlock, bool hasRightChannel) noexcept
    {
        diagSumSqL += sumSqLBlock;
        if (hasRightChannel)
            diagSumSqR += sumSqRBlock;
        if (peakAbsBlock > diagPeakAbs)
            diagPeakAbs = peakAbsBlock;
        diagSampleCount += sampleCountBlock;

        const uint32_t publishWindow = publishWindowSamples.load(std::memory_order_relaxed);
        if (diagSampleCount >= publishWindow)
        {
            const double invCount = 1.0 / static_cast<double>(diagSampleCount);
            const float meanSqL = static_cast<float>(diagSumSqL * invCount);
            const float meanSqR = hasRightChannel ? static_cast<float>(diagSumSqR * invCount) : 0.0f;

            meanSqErrorL.store(meanSqL, std::memory_order_relaxed);
            meanSqErrorR.store(meanSqR, std::memory_order_relaxed);
            peakAbsError.store(static_cast<float>(diagPeakAbs), std::memory_order_relaxed);
            windowSamples.store(diagSampleCount, std::memory_order_relaxed);

            diagSumSqL = 0.0;
            diagSumSqR = 0.0;
            diagPeakAbs = 0.0;
            diagSampleCount = 0;
        }
    }

    inline void resetDiagnostics() noexcept
    {
        diagSumSqL = 0.0;
        diagSumSqR = 0.0;
        diagPeakAbs = 0.0;
        diagSampleCount = 0;
        meanSqErrorL.store(0.0f, std::memory_order_relaxed);
        meanSqErrorR.store(0.0f, std::memory_order_relaxed);
        peakAbsError.store(0.0f, std::memory_order_relaxed);
        windowSamples.store(0u, std::memory_order_relaxed);
    }

    inline double absNoLibm(double x) const noexcept
    {
        union { double d; uint64_t u; } v { x };
        v.u &= 0x7fffffffffffffffULL;
        return v.d;
    }

    inline double get(const std::array<double, ORDER>& buffer, int idx, int k) const noexcept
    {
        int i = idx + k;
        if (i >= ORDER)
            i -= ORDER;
        return buffer[static_cast<size_t>(i)];
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

    inline double quantize(double v, Xoshiro256State& rng) const noexcept
    {
        const double minV = -1.0;
        const double maxV = 1.0 - (1.0 / invScale);

        // TPDF dither
        const double u1 = uniform(rng);
        const double u2 = uniform(rng);
        v += (u1 + u2 - 1.0) * scale;

        if (v < minV)
            v = minV;
        else if (v > maxV)
            v = maxV;

        __m128d d = _mm_set_sd(v * invScale);
        d = _mm_round_sd(d, d, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const double q = _mm_cvtsd_f64(d);
        return q * scale;
    }

    static constexpr std::array<double, 10> PRESET_SAMPLE_RATES = {
        44100.0, 48000.0, 88200.0, 96000.0, 176400.0, 192000.0,
        352800.0, 384000.0, 705600.0, 768000.0
    };

    alignas(64) static constexpr std::array<std::array<double, ORDER>, 10> COEFF_PRESETS = {{
        { 0.394958, 0.319775, 0.145569, 0.139697 },  // 44.1 kHz
        { 0.460000, 0.280000, 0.170000, 0.090000 },  // 48 kHz   (基準)
        { 0.727810, 0.189547, 0.125028, -0.042385 }, // 88.2 kHz
        { 0.742333, 0.185474, 0.106133, -0.033940 }, // 96 kHz
        { 0.775904, 0.126967, 0.043467, 0.053661 },  // 176.4 kHz
        { 0.774132, 0.117440, 0.047291, 0.061137 },  // 192 kHz
        { 0.724647, 0.094403, 0.113208, 0.067743 },  // 352.8 kHz
        { 0.714605, 0.097798, 0.124553, 0.063045 },  // 384 kHz
        { 0.635851, 0.161114, 0.194506, 0.008529 },  // 705.6 kHz
        { 0.624827, 0.174509, 0.201424, -0.000760 }  // 768 kHz
    }};

    static void selectPresetWithInterpolation(double sampleRate, int& idxLow, int& idxHigh, double& t) noexcept
    {
        if (sampleRate <= PRESET_SAMPLE_RATES.front())
        {
            idxLow = idxHigh = 0;
            t = 0.0;
            return;
        }
        if (sampleRate >= PRESET_SAMPLE_RATES.back())
        {
            idxLow = idxHigh = static_cast<int>(PRESET_SAMPLE_RATES.size()) - 1;
            t = 0.0;
            return;
        }

        for (size_t i = 0; i < PRESET_SAMPLE_RATES.size() - 1; ++i)
        {
            if (sampleRate >= PRESET_SAMPLE_RATES[i] && sampleRate < PRESET_SAMPLE_RATES[i + 1])
            {
                idxLow = static_cast<int>(i);
                idxHigh = static_cast<int>(i + 1);
                t = (sampleRate - PRESET_SAMPLE_RATES[i]) / (PRESET_SAMPLE_RATES[i + 1] - PRESET_SAMPLE_RATES[i]);
                return;
            }
        }
    }

    std::array<double, ORDER> coeffs { 0.46, 0.28, 0.17, 0.09 };

    std::array<std::array<double, ORDER>, MAX_CHANNELS> errors {};
    std::array<int, MAX_CHANNELS> writePos {};
    Xoshiro256State rngState[MAX_CHANNELS] = {
        {{ 0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL, 0x0123456789ABCDEFULL, 0xEFCDAB8967452301ULL }},
        {{ 0x89ABCDEF01234567ULL, 0x76543210FEDCBA98ULL, 0xABCDEF0123456789ULL, 0x67452301EFCDAB89ULL }},
        {{ 0x456789ABCDEF0123ULL, 0x3210FEDCBA987654ULL, 0xCDEF0123456789ABULL, 0x2301EFCDAB896745ULL }},
        {{ 0xCDEF0123456789ABULL, 0x2301EFCDAB896745ULL, 0x456789ABCDEF0123ULL, 0x3210FEDCBA987654ULL }},
        {{ 0x0123456789ABCDEFULL, 0xEFCDAB8967452301ULL, 0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL }},
        {{ 0xABCDEF0123456789ULL, 0x67452301EFCDAB89ULL, 0x89ABCDEF01234567ULL, 0x76543210FEDCBA98ULL }},
        {{ 0x2301EFCDAB896745ULL, 0x456789ABCDEF0123ULL, 0x3210FEDCBA987654ULL, 0xCDEF0123456789ABULL }},
        {{ 0x67452301EFCDAB89ULL, 0xABCDEF0123456789ULL, 0x76543210FEDCBA98ULL, 0x89ABCDEF01234567ULL }}
    };
    int currentBitDepth = 0;
    double scale = 1.0;
    double invScale = 1.0;

    double diagSumSqL = 0.0;
    double diagSumSqR = 0.0;
    double diagPeakAbs = 0.0;
    uint32_t diagSampleCount = 0;
    std::atomic<float> meanSqErrorL { 0.0f };
    std::atomic<float> meanSqErrorR { 0.0f };
    std::atomic<float> peakAbsError { 0.0f };
    std::atomic<uint32_t> windowSamples { 0u };
    std::atomic<uint32_t> publishWindowSamples { 8192u };
};

} // namespace convo
