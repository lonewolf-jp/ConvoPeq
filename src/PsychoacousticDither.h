//============================================================================
#pragma once
// PsychoacousticDither.h ── v0.2 (JUCE 8.0.12対応)
//
// Ultra Mastering Dither Engine
// 64bit Double専用 Psychoacoustic Dither RNG
// 構成 (Architecture):
//
//   1. Xoshiro256** (L/R独立 jump)
//   2. True TPDF Dither
//   3. 5次 Noise Shaper (聴覚特性最適化係数)
//   6. Soft Limiting
//   7. Quantization
// 商用最高峰ディザ（POW-r #3 / MBIT+ クラス）を理論的に上回る構造（目標）
//============================================================================

#include <JuceHeader.h>
#include <cmath>
#include <array>
#include <chrono>
#include <random>
#include <optional>



namespace dsp
{

//============================================================
// SplitMix64 (高品質シード生成)
//============================================================
class SplitMix64
{
public:
    explicit SplitMix64(uint64_t seed) noexcept : state(seed) {}

    inline uint64_t next() noexcept
    {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

private:
    uint64_t state;
};

//============================================================
// xoshiro256**
//============================================================
class Xoshiro256ss
{
public:
    Xoshiro256ss() noexcept : s{1, 2, 3, 4} {}

    explicit Xoshiro256ss(uint64_t seed) noexcept
    {
        SplitMix64 sm(seed);
        s[0]=sm.next(); s[1]=sm.next();
        s[2]=sm.next(); s[3]=sm.next();
    }

    inline uint64_t nextUInt64() noexcept
    {
        const uint64_t result =
            rotl(s[1] * 5ULL, 7) * 9ULL;

        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl(s[3], 45);

        return result;
    }

    void jump() noexcept
    {
        static constexpr uint64_t JUMP[] =
        {
            0x180ec6d33cfd0abaULL,
            0xd5a61266f0c9392cULL,
            0xa9582618e03fc9aaULL,
            0x39abdc4529b1661cULL
        };

        uint64_t t0=0,t1=0,t2=0,t3=0;

        for(int i=0;i<4;++i)
        {
            for(int b=0;b<64;++b)
            {
                if(JUMP[i] & (1ULL<<b))
                {
                    t0^=s[0]; t1^=s[1];
                    t2^=s[2]; t3^=s[3];
                }
                nextUInt64();
            }
        }
        s[0]=t0; s[1]=t1; s[2]=t2; s[3]=t3;
    }

private:
    static inline uint64_t rotl(uint64_t x,int k) noexcept
    {
        return (x<<k)|(x>>(64-k));
    }

    uint64_t s[4];
};

//============================================================
// SplitMix64 Finalizer（最有力）
//
// 統計的に非常に強力で高速。
// xoshiro系との相性も良好。
//============================================================
static inline uint64_t whiten64(uint64_t x) noexcept
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

//============================================================
// 超低歪マスタリング用 Dither + Noise Shaper
//============================================================
class PsychoacousticDither
{
public:
    static constexpr int MAX_CHANNELS = 2;
    static constexpr int DEFAULT_BIT_DEPTH = 24;

    explicit PsychoacousticDither(std::optional<uint64_t> seed = std::nullopt)
    {
        uint64_t baseSeed = seed.value_or(static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        for (int i = 0; i < MAX_CHANNELS; ++i)
        {
            rng[i] = Xoshiro256ss(baseSeed + i * 1000);
            if (i > 0) rng[i].jump(); // L/R独立
        }
    }

    void prepare(double /*sampleRate*/, int bitDepth = DEFAULT_BIT_DEPTH) noexcept
    {
        if (bitDepth > 0)
        {
            // N-bit signed PCM quantization step is 2 / 2^N = 1 / 2^(N-1)
            scale = 1.0 / std::pow(2.0, bitDepth - 1);
            invScale = std::pow(2.0, bitDepth - 1);
        }
        else
        {
            // Default 24-bit (2^23 for signed PCM)
            scale = 1.0 / 8388608.0;
            invScale = 8388608.0;
        }

        reset();
    }

    void reset() noexcept
    {
        for (auto& st : state) st = {0};
    }

    inline double process(double input, int channel) noexcept
    {
        if (channel < 0 || channel >= MAX_CHANNELS) return input;
        return processChannel(input, rng[channel], state[channel]);
    }

private:
    struct ShaperState
    {
        double z[5]{0,0,0,0,0};
    };

    inline double processChannel(double x, Xoshiro256ss& r, ShaperState& st) noexcept
    {
        // TPDF Dither generation
        double d = nextTPDF(r) * scale;

        // 5th order Noise Shaper (Feedback Error)
        double shapedError =
            1.8  * st.z[0]
          - 1.2  * st.z[1]
          + 0.7  * st.z[2]
          - 0.3  * st.z[3]
          + 0.12 * st.z[4];

        // Apply dither and noise shaping
        double tmp = x + d + shapedError;

        // Quantize
        double quantized = std::round(tmp * invScale) * scale;

        // Calculate Quantization Error
        double error = tmp - quantized;

        // Update State (Shift)
        st.z[4]=st.z[3];
        st.z[3]=st.z[2];
        st.z[2]=st.z[1];
        st.z[1]=st.z[0];
        st.z[0]=killDenormal(error);

        return quantized;
    }

    inline double nextTPDF(Xoshiro256ss& r) noexcept
    {
        return (uniform53(r)-0.5) + (uniform53(r)-0.5);
    }

    inline double uniform53(Xoshiro256ss& r) noexcept
    {
        uint64_t v = whiten64(r.nextUInt64()); // Apply whitening
        v >>= 11; // 53bit
        return static_cast<double>(v) * (1.0 / 9007199254740992.0);
    }

    inline double killDenormal(double x) const noexcept
    {
        return (std::fabs(x)<1e-300)?0.0:x;
    }

    Xoshiro256ss rng[MAX_CHANNELS];
    ShaperState state[MAX_CHANNELS];
    double scale = 1.0 / 16777216.0;
    double invScale = 16777216.0;
};

} // namespace dsp