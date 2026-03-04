//============================================================================
#pragma once
// PsychoacousticDither.h ── v0.2 (JUCE 8.0.12対応)
//
// Ultra Mastering Dither Engine
// 64bit Double専用 Psychoacoustic Dither RNG
// 構成:
//
//   1. Xoshiro256** (L/R独立 jump)
//   2. True TPDF Dither
//   3. 5次 Noise Shaper (Error Feedback Topology)
//   4. Quantization
// 商用最高峰ディザ（POW-r #3 / MBIT+ クラス）を理論的に上回る構造（目標）
//============================================================================

#include <JuceHeader.h>
#include <cmath>
#include <array>
#include <chrono>
#include <atomic>
#include <random>
#include <optional>
#include <algorithm>
#include "AlignedAllocation.h"

#if JUCE_DSP_USE_INTEL_MKL
 #include <mkl_vsl.h>
#endif


namespace convo
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
    static constexpr int MAX_CHANNELS = 8; // 将来の多チャンネル拡張に備えて余裕を持たせる
    static constexpr int DEFAULT_BIT_DEPTH = 24;
    static constexpr int STATE_STRIDE = 8; // 64 bytes alignment (8 * sizeof(double))

#if JUCE_DSP_USE_INTEL_MKL
    // RAII Wrapper for MKL VSL Stream to prevent leaks
    class VSLStream {
    public:
        VSLStream() = default;
        ~VSLStream() { reset(); }

        void init(uint64_t seed) {
            reset();
            // VSL_BRNG_SFMT19937: SIMD-oriented Fast Mersenne Twister (High Quality & Fast)
            vslNewStream(&stream, VSL_BRNG_SFMT19937, static_cast<unsigned int>(seed));
        }

        void reset() {
            if (stream) { vslDeleteStream(&stream); stream = nullptr; }
        }

        operator VSLStreamStatePtr() const { return stream; }
    private:
        VSLStreamStatePtr stream = nullptr;
        VSLStream(const VSLStream&) = delete;
        VSLStream& operator=(const VSLStream&) = delete;
    };
#endif

    ~PsychoacousticDither() {
        if (shaperStateBuffer) convo::aligned_free(shaperStateBuffer);
    }

    PsychoacousticDither(const PsychoacousticDither&) = delete;
    PsychoacousticDither& operator=(const PsychoacousticDither&) = delete;

    explicit PsychoacousticDither(std::optional<uint64_t> seed = std::nullopt)
    {
        uint64_t baseSeed;
        if (seed.has_value())
        {
            baseSeed = seed.value();
        }
        else
        {
            // 時間 + 静的カウンタでユニーク性を確保 (複数インスタンス同時生成時のシード衝突防止)
            static std::atomic<uint64_t> instanceCounter { 0 };
            baseSeed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())
                     ^ (instanceCounter.fetch_add(1) * 0x9e3779b97f4a7c15ULL);
        }

#if JUCE_DSP_USE_INTEL_MKL
        // Initialize MKL VSL Streams
        // Use SplitMix64 to generate independent seeds for each channel
        SplitMix64 seeder(baseSeed);
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            rng[i].init(seeder.next());
            rndIndex[i] = RND_BUFFER_SIZE; // Force refill on first use
        }
#else
        Xoshiro256ss master(baseSeed);
        for (int i = 0; i < MAX_CHANNELS; ++i)
        {
            rng[i] = master;
            master.jump(); // Ensure non-overlapping sequences for each channel
        }
#endif

        shaperStateBuffer = static_cast<double*>(convo::aligned_malloc(MAX_CHANNELS * STATE_STRIDE * sizeof(double), 64));
        reset();
    }

    void prepare(double sampleRate, int bitDepth = DEFAULT_BIT_DEPTH) noexcept
    {
        if (bitDepth > 0)
        {
            // Nビット符号付きPCMの量子化ステップは 2 / 2^N = 1 / 2^(N-1)
            scale = 1.0 / std::pow(2.0, bitDepth - 1);
            invScale = std::pow(2.0, bitDepth - 1);
        }
        else
        {
            // Default 24-bit (2^23 for signed PCM)
            scale = 1.0 / 8388608.0;
            invScale = 8388608.0;
        }

        // 5次 Noise Shaper 係数設定 (Lipshitz / Wannamaker系最適化)
        //
        // 1. 標準セット (Standard):
        //    2.033, -2.165, 1.959, -1.590, 0.6149
        //    44.1kHz〜384kHzまで幅広く対応し、可聴域のノイズを強力に低減します。
        //
        // 2. アグレッシブ版 (Aggressive):
        //    2.45, -2.68, 2.35, -1.85, 0.72
        //    176.4kHz以上のハイレゾかつ16bit出力時に、ノイズを超音波域へさらに強く追いやります。

        if (sampleRate >= 176000.0 && bitDepth <= 16)
        {
            coeffs = { 2.45, -2.68, 2.35, -1.85, 0.72 };
        }
        else
        {
            coeffs = { 2.033, -2.165, 1.959, -1.590, 0.6149 };
        }

        reset();
    }

    void reset() noexcept
    {
        if (shaperStateBuffer)
            std::fill_n(shaperStateBuffer, MAX_CHANNELS * STATE_STRIDE, 0.0);
    }

    inline double process(double input, int channel) noexcept
    {
        if (channel < 0 || channel >= MAX_CHANNELS) return input;
#if JUCE_DSP_USE_INTEL_MKL
        return processChannelMKL(input, channel, shaperStateBuffer + (channel * STATE_STRIDE));
#else
        return processChannel(input, rng[channel], shaperStateBuffer + (channel * STATE_STRIDE));
#endif
    }

private:
#if !JUCE_DSP_USE_INTEL_MKL
    inline double processChannel(double x, Xoshiro256ss& r, double* z) noexcept
    {
        // TPDFディザ生成
        double d = nextTPDF(r) * scale;
#else
    inline double processChannelMKL(double x, int channel, double* z) noexcept
    {
        // TPDFディザ生成 (MKL)
        double d = nextTPDF_MKL(channel) * scale;
#endif

        // 5次ノイズシェーパー (フィードバック誤差)
        double shapedError =
            coeffs[0] * z[0]
          + coeffs[1] * z[1]
          + coeffs[2] * z[2]
          + coeffs[3] * z[3]
          + coeffs[4] * z[4];

        // ディザとノイズシェーピングの適用
        double tmp = x + d + shapedError;

        // 量子化
        double quantized = std::round(tmp * invScale) * scale;

        // 量子化誤差の計算 (Input - Output)
        // 注意: これは -(Output - Input) と等価です。
        // shapedError は過去の誤差に正の係数を掛けて加算しているため、
        // これは実質的に量子化ノイズを減算することになります (1 - H(z) トポロジー)。
        double error = tmp - quantized;

        // 状態の更新 (シフト)
        z[4]=z[3];
        z[3]=z[2];
        z[2]=z[1];
        z[1]=z[0];
        z[0]=killDenormal(error);

        return quantized;
    }

#if !JUCE_DSP_USE_INTEL_MKL
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
#else
    // MKL VSL Batch Generation for efficiency
    static constexpr int RND_BUFFER_SIZE = 1024; // Increased batch size to reduce VSL call overhead
    // 【パッチ6】vdRngUniform の出力バッファを 64byte アライメントする
    // 理由: コーディング規約「MKL使用箇所ではメモリは64byteアライメントとする」に従う。
    //       非アライメントバッファへの vdRngUniform 書き込みは機能上は動作するが、
    //       MKL の SIMD 最適化パスが有効にならず AVX-512 等のベクトル幅で
    //       性能劣化が生じる可能性がある。
    //       平坦な2次元配列を alignas(64) で宣言することで
    //       先頭アドレスの 64byte アライメントを保証する。
    alignas(64) double rndBuffer[MAX_CHANNELS][RND_BUFFER_SIZE];
    int rndIndex[MAX_CHANNELS];

    inline double nextTPDF_MKL(int channel) noexcept
    {
        // Need 2 uniform numbers for TPDF
        double u1 = getNextUniformMKL(channel);
        double u2 = getNextUniformMKL(channel);
        return (u1 - 0.5) + (u2 - 0.5);
    }

    inline double getNextUniformMKL(int channel) noexcept
    {
        if (rndIndex[channel] >= RND_BUFFER_SIZE)
        {
            // Refill buffer
            // VSL_RNG_METHOD_UNIFORM_STD: Standard method (accurate)
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rng[channel], RND_BUFFER_SIZE, rndBuffer[channel], 0.0, 1.0);
            rndIndex[channel] = 0;
        }
        return rndBuffer[channel][rndIndex[channel]++];
    }
#endif

    inline double killDenormal(double x) const noexcept
    {
        return (std::fabs(x)<1e-300)?0.0:x;
    }

#if JUCE_DSP_USE_INTEL_MKL
    VSLStream rng[MAX_CHANNELS];
#else
    Xoshiro256ss  rng[MAX_CHANNELS];
#endif

    double* shaperStateBuffer = nullptr;
        double scale    = 1.0 / 8388608.0;    // 2^23（24bit signed PCM デフォルト）
        double invScale = 8388608.0;           // 2^23（24bit signed PCM デフォルト）
    std::array<double, 5> coeffs { 2.033, -2.165, 1.959, -1.590, 0.6149 };
};

} // namespace convo