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

#include <immintrin.h>
#include <mkl_vsl.h>


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
// 超低歪マスタリング用 Dither + Noise Shaper
//============================================================
class PsychoacousticDither
{
public:
    static constexpr int MAX_CHANNELS = 8; // 将来の多チャンネル拡張に備えて余裕を持たせる
    static constexpr int DEFAULT_BIT_DEPTH = 24;
    static constexpr int STATE_STRIDE = 8; // 64 bytes alignment (8 * sizeof(double))

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

        // Initialize MKL VSL Streams
        // Use SplitMix64 to generate independent seeds for each channel
        SplitMix64 seeder(baseSeed);
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            rng[i].init(seeder.next());

            // 【追加】初回バッファ充填 (Audio Threadでの初回ジッター防止)
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rng[i], RND_BUFFER_SIZE, rndBuffer[i], 0.0, 1.0);
            rndIndex[i] = 0;
        }

        shaperStateBuffer = static_cast<double*>(convo::aligned_malloc(MAX_CHANNELS * STATE_STRIDE * sizeof(double), 64));
        reset();
    }

    void prepare(double sampleRate, int bitDepth = DEFAULT_BIT_DEPTH) noexcept
    {
        // bitDepth <= 0 の場合はディザリングが無効化されるため、スケール計算は不要。
        // AudioEngine::DSPCore::processOutput() の applyDither フラグで処理がスキップされる。
        if (bitDepth <= 0)
            return;

        // Nビット符号付きPCMの量子化ステップは 2 / 2^N = 1 / 2^(N-1)
        scale = 1.0 / std::pow(2.0, bitDepth - 1);
        invScale = std::pow(2.0, bitDepth - 1);

        // 5次 Noise Shaper 係数設定 (Lipshitz / Wannamaker系最適化)
        //
        // 1. 標準セット (Standard):
        //    2.033, -2.165, 1.959, -1.590, 0.6149
        //    44.1kHz〜384kHzまで幅広く対応し、可聴域のノイズを強力に低減します。
        //
        // 2. アグレッシブ版 (Aggressive):
        //    2.45, -2.68, 2.35, -1.85, 0.72
        //    176.4kHz以上のハイレゾかつ16bit出力時に、ノイズを超音波域へさらに強く追いやります。
        //
        // 【最適化】サンプルレートに応じて係数を動的に補間し、帯域幅を最大限活用する。
        static constexpr std::array<double, 5> coeffs44k  = { 2.033, -2.165, 1.959, -1.590, 0.6149 };
        static constexpr std::array<double, 5> coeffs176k = { 2.45,  -2.68,  2.35,  -1.85,  0.72   };

        // 44.1kHz 〜 176.4kHz 間で線形補間 (t=0.0 -> Standard, t=1.0 -> Aggressive)
        double t = (sampleRate - 44100.0) / (176400.0 - 44100.0);
        t = std::max(0.0, std::min(1.0, t));

        for (int i = 0; i < 5; ++i)
            coeffs[i] = coeffs44k[i] * (1.0 - t) + coeffs176k[i] * t;

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
        return processChannelMKL(input, channel, shaperStateBuffer + (channel * STATE_STRIDE));
    }

    // 【最適化】ステレオ/モノラル ブロック処理
    // ループをインライン化し、関数呼び出しオーバーヘッドを削減。
    // ステレオ時はL/Rを並列処理し、CPUパイプライン効率(ILP)を向上させる。
    // 量子化ステップではSSE4.1を使用してL/Rを同時に丸める。
    inline void processStereoBlock(double* dataL, double* dataR, int numSamples, double headroom) noexcept
    {
        double* zL = shaperStateBuffer + (0 * STATE_STRIDE);

        if (dataR != nullptr)
        {
            // --- Stereo Path ---
            double* zR = shaperStateBuffer + (1 * STATE_STRIDE);
            const __m128d v_scale = _mm_set1_pd(scale);
            const __m128d v_invScale = _mm_set1_pd(invScale);

            for (int i = 0; i < numSamples; ++i)
            {
                // Shaped Error (Scalar, compiler will optimize with FMA)
                const double shapedErrorL = coeffs[0] * zL[0] + coeffs[1] * zL[1] + coeffs[2] * zL[2] + coeffs[3] * zL[3] + coeffs[4] * zL[4];
                const double shapedErrorR = coeffs[0] * zR[0] + coeffs[1] * zR[1] + coeffs[2] * zR[2] + coeffs[3] * zR[3] + coeffs[4] * zR[4];

                // Dither (MKL)
                const double dL = nextTPDF_MKL(0) * scale;
                const double dR = nextTPDF_MKL(1) * scale;

                // Combine
                const double tmpL = (dataL[i] * headroom) + dL + shapedErrorL;
                const double tmpR = (dataR[i] * headroom) + dR + shapedErrorR;

                // Quantize (SSE4.1/AVX)
                const __m128d v_tmp = _mm_set_pd(tmpR, tmpL);
                const __m128d v_scaled = _mm_mul_pd(v_tmp, v_invScale);
                const __m128d v_rounded = _mm_round_pd(v_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                const __m128d v_quantized = _mm_mul_pd(v_rounded, v_scale);

                alignas(16) double quantized[2];
                _mm_store_pd(quantized, v_quantized); // [quantizedL, quantizedR]

                // Error and State Update (Scalar)
                const double errorL = tmpL - quantized[0];
                zL[4]=zL[3]; zL[3]=zL[2]; zL[2]=zL[1]; zL[1]=zL[0]; zL[0]=killDenormal(errorL);
                dataL[i] = quantized[0];

                const double errorR = tmpR - quantized[1];
                zR[4]=zR[3]; zR[3]=zR[2]; zR[2]=zR[1]; zR[1]=zR[0]; zR[0]=killDenormal(errorR);
                dataR[i] = quantized[1];
            }
        }
        else
        {
            // --- Mono Path ---
            for (int i = 0; i < numSamples; ++i)
            {
                const double d = nextTPDF_MKL(0) * scale;
                const double shapedError = coeffs[0] * zL[0] + coeffs[1] * zL[1] + coeffs[2] * zL[2] + coeffs[3] * zL[3] + coeffs[4] * zL[4];
                const double tmp = (dataL[i] * headroom) + d + shapedError;
                // Quantize (Round to nearest even using SSE4.1)
                __m128d v = _mm_set_sd(tmp * invScale);
                v = _mm_round_sd(v, v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                const double quantized = _mm_cvtsd_f64(v) * scale;
                const double error = tmp - quantized;
                zL[4]=zL[3]; zL[3]=zL[2]; zL[2]=zL[1]; zL[1]=zL[0]; zL[0]=killDenormal(error);
                dataL[i] = quantized;
            }
        }
    }

private:
    inline double processChannelMKL(double x, int channel, double* z) noexcept
    {
        // TPDFディザ生成 (MKL)
        double d = nextTPDF_MKL(channel) * scale;

        // 5次ノイズシェーパー (フィードバック誤差)
        double shapedError =
            coeffs[0] * z[0]
          + coeffs[1] * z[1]
          + coeffs[2] * z[2]
          + coeffs[3] * z[3]
          + coeffs[4] * z[4];

        // ディザとノイズシェーピングの適用
        double tmp = x + d + shapedError;

        // 量子化 (Round to nearest even using SSE4.1)
        __m128d v = _mm_set_sd(tmp * invScale);
        v = _mm_round_sd(v, v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        double quantized = _mm_cvtsd_f64(v) * scale;

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

    // MKL VSL Batch Generation for efficiency
    static constexpr int RND_BUFFER_SIZE = 512; // Reduced batch size to spread load and minimize jitter
    // 【パッチ6】rndBufferをインスタンスメンバー化 (データレース防止)
    // 理由: static変数は全インスタンスで共有されるため、複数のPsychoacousticDither
    //       インスタンス（例: RCUによる新旧DSPCore）が同時に存在すると、
    //       MKLの乱数生成バッファへのアクセスが競合し、RNG状態が破壊される。
    //       インスタンスメンバーにすることで、各インスタンスが独立したバッファを持つ。
    //       alignas(64) はMKLのSIMD最適化のために維持する。
    alignas(64) double rndBuffer[MAX_CHANNELS][RND_BUFFER_SIZE] {};
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
            if (vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rng[channel], RND_BUFFER_SIZE, rndBuffer[channel], 0.0, 1.0) != VSL_STATUS_OK)
            {
                // MKLエラー時はディザを無効化（ゼロで埋める）して安全に継続
                std::fill_n(rndBuffer[channel], RND_BUFFER_SIZE, 0.5); // 0.5 for uniform [0,1] to produce 0 TPDF
            }
            rndIndex[channel] = 0;
        }
        return rndBuffer[channel][rndIndex[channel]++];
    }

    inline double killDenormal(double x) const noexcept
    {
        return (std::fabs(x)<1e-300)?0.0:x;
    }

    VSLStream rng[MAX_CHANNELS];

    double* shaperStateBuffer = nullptr;
        double scale    = 1.0 / 8388608.0;    // 2^23（24bit signed PCM デフォルト）
        double invScale = 8388608.0;           // 2^23（24bit signed PCM デフォルト）
    std::array<double, 5> coeffs { 2.033, -2.165, 1.959, -1.590, 0.6149 };
};

} // namespace convo