//============================================================================
#pragma once
// Fixed15TapNoiseShaper.h
// 15-tap error-feedback noise shaper (RT-safe, allocation-free in process)
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

class Fixed15TapNoiseShaper
{
public:
    static constexpr int MAX_CHANNELS = 8;
    static constexpr int ORDER = 15;

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

        double fb = 0.0;
        for (int i = 0; i < ORDER; ++i)
        {
            fb += coeffs[i] * get(channelErrors, idx, i);
        }

        const double y = x - fb;
        const double yq = quantize(y, rngState[static_cast<size_t>(channel)]);
        const double error = yq - y;
        outError = error;

        const double clampedError = std::clamp(error, -2.0 * scale, 2.0 * scale);
        idx = (idx - 1 + ORDER) % ORDER;
        channelErrors[static_cast<size_t>(idx)] = killDenormal(clampedError);

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
        // 44.1 kHz
        { 2.157553, -2.356649, 2.179194, -1.802605, 1.429476, -1.073975, 0.775233, -0.535496, 0.360294, -0.229526, 0.143225, -0.081483, 0.045992, -0.021109, 0.009877 },
        // 48 kHz (基準)
        { 2.172009, -2.313034, 2.092949, -1.698718, 1.304487, -0.946581, 0.645299, -0.415598, 0.251068, -0.141026, 0.072650, -0.033120, 0.012821, -0.004274, 0.001068 },
        // 88.2 kHz
        { 1.458665, -1.271063, 1.372588, -1.257752, 1.186326, -1.042666, 0.931875, -0.787020, 0.671068, -0.541164, 0.438950, -0.333234, 0.250772, -0.174640, 0.097295 },
        // 96 kHz
        { 1.366976, -1.123204, 1.234291, -1.119397, 1.063887, -0.931030, 0.838107, -0.707665, 0.608977, -0.492384, 0.404256, -0.308827, 0.236248, -0.167088, 0.096853 },
        // 176.4 kHz
        { 0.892356, -0.425055, 0.645737, -0.531778, 0.565511, -0.483687, 0.474500, -0.404025, 0.379228, -0.317474, 0.286683, -0.233505, 0.199702, -0.166141, 0.117948 },
        // 192 kHz
        { 0.842437, -0.356337, 0.593464, -0.477529, 0.519248, -0.440863, 0.438827, -0.372969, 0.354221, -0.297057, 0.271334, -0.222591, 0.192842, -0.164283, 0.119255 },
        // 352.8 kHz
        { 0.576947, -0.000943, 0.355358, -0.225398, 0.306449, -0.241465, 0.271718, -0.228634, 0.237327, -0.205281, 0.201703, -0.179310, 0.166143, -0.176849, 0.142236 },
        // 384 kHz
        { 0.550200, 0.035746, 0.334748, -0.202925, 0.287573, -0.223403, 0.255932, -0.214959, 0.225551, -0.196308, 0.194281, -0.175339, 0.163224, -0.180050, 0.145728 },
        // 705.6 kHz
        { 0.403358, 0.274330, 0.229984, -0.085257, 0.190310, -0.131467, 0.169688, -0.142598, 0.154703, -0.144947, 0.142117, -0.148598, 0.132904, -0.195545, 0.151017 },
        // 768 kHz
        { 0.390229, 0.306061, 0.221612, -0.075413, 0.182734, -0.125438, 0.162912, -0.138648, 0.149015, -0.142960, 0.137870, -0.149116, 0.130580, -0.202133, 0.152692 }
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

    std::array<double, ORDER> coeffs { 2.172009, -2.313034, 2.092949, -1.698718, 1.304487, -0.946581, 0.645299, -0.415598, 0.251068, -0.141026, 0.072650, -0.033120, 0.012821, -0.004274, 0.001068 };

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
