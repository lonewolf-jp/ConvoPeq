//============================================================================
#pragma once
// PsychoacousticDither.h ── v0.3 (JUCE 8.0.12対応)
//
// Ultra Mastering Dither Engine
// 64bit Double専用 Psychoacoustic Dither RNG
// 構成:
//
//   1. Xoshiro256** (L/R独立 jump)
//   2. True TPDF Dither
//   3. 12次 Noise Shaper (Error Feedback Topology)
//      ├ 出力サンプルレート帯域: 5バンド
//      └ ビット深度プリセット: 16bit(強め) / 24bit(標準) / 32bit(控えめ)
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
#include <thread>
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
    static constexpr int MAX_CHANNELS    = 8;  // 将来の多チャンネル拡張に備えて余裕を持たせる
    static constexpr int DEFAULT_BIT_DEPTH = 24;
    static constexpr int NS_ORDER        = 12;  // ノイズシェーパー次数

    // STATE_STRIDE: チャンネルごとの状態配列オフセット (double 単位)。
    // 64 バイト境界アライメントを各チャンネルで維持するには、
    //   STATE_STRIDE × sizeof(double) ≡ 0 (mod 64)
    //   → STATE_STRIDE は 8 の倍数でなければならない。
    // NS_ORDER = 12 を収める最小の 8 倍数は 16 (128 バイト/チャンネル)。
    static constexpr int STATE_STRIDE    = 16; // 16 × 8 = 128 bytes/ch (64-byte aligned)

    // RAII Wrapper for MKL VSL Stream to prevent leaks
    class VSLStream {
    public:
        VSLStream() = default;
        ~VSLStream() { reset(); }

        void init(uint64_t seed) {
            reset();
            // VSL_BRNG_SFMT19937: SIMD-oriented Fast Mersenne Twister (High Quality & Fast)
            // [Fix] vslNewStream() の戻り値を必ず検査する。
            // 失敗した場合 stream = nullptr のままにして、呼び出し側で isValid() により保護する。
            const MKL_INT status = vslNewStream(&stream, VSL_BRNG_SFMT19937,
                                                static_cast<unsigned int>(seed));
            if (status != VSL_STATUS_OK)
                stream = nullptr;
        }

        bool isValid() const noexcept { return stream != nullptr; }

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
        stopRngProducer();
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
            const uint64_t seedValue = seeder.next();
            rng[i].init(seedValue);
            fallbackState[i] = seedValue ^ 0xd1b54a32d192ed03ULL;
            rngReadPos[i].store(0u, std::memory_order_relaxed);
            rngWritePos[i].store(0u, std::memory_order_relaxed);
        }

        shaperStateBuffer = static_cast<double*>(convo::aligned_malloc(MAX_CHANNELS * STATE_STRIDE * sizeof(double), 64));
        reset();
    }

    // -------------------------------------------------------------------------
    // 12次 Noise Shaper 係数テーブル
    //
    // インデックス: [SR帯域(0-5)][ビット深度プリセット(0=16bit/1=24bit/2=32bit)][タップ(0-11)]
    //
    // SR帯域:
    //   0: 44.1kHz
    //   1: 48kHz
    //   2: 96kHz
    //   3: 176.4kHz / 192kHz
    //   4: 352.8kHz / 384kHz
    //   5: 705.6kHz 以上
    //
    // ビット深度プリセット:
    //   0: 16bit 強め  (ノイズを超音波域へ強く押し出す)
    //   1: 24bit 標準  (バランス型、POW-r #3 クラス)
    //   2: 32bit 控えめ (フロアノイズが低いため穏やかな形状で十分)
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // 係数テーブル
    //
    // 【エラーフィードバック型NSFの安定性について】
    //   H(z) = c0*z⁻¹ + ... + c11*z⁻¹² は FIR フィルタ (極は全てz=0)。
    //   量子化器により e[n] = tmp[n] - Q(tmp[n]) は常に有界 (|e[n]| ≤ scale/2)。
    //   shapedError[n] = Σ c_k * e[n-k-1] も有界 → 常にBIBO安定。
    //   「NTF零点が単位円外 → 不安定」は直接形 (IIR予測型) NSFの理論であり、
    //   エラーフィードバック型には適用されない。
    //
    //   最大整形誤差 = (scale/2) × Σ|c_k|  [最悪ケース 705.6k/16bit: ≈ -66 dBFS]
    //
    // インデックス: [SR帯域(0-5)][ビット深度プリセット(0=16bit/1=24bit/2=32bit)][タップ(0-11)]
    //
    // SR帯域:
    //   0: 44.1kHz
    //   1: 48kHz
    //   2: 96kHz
    //   3: 176.4kHz / 192kHz
    //   4: 352.8kHz / 384kHz
    //   5: 705.6kHz 以上
    //
    // ビット深度プリセット:
    //   0: 16bit 強め  (ノイズを超音波域へ強く押し出す)
    //   1: 24bit 標準  (バランス型、POW-r #3 クラス)
    //   2: 32bit 控えめ (フロアノイズが低いため穏やかな形状で十分)
    // -------------------------------------------------------------------------
    static constexpr int SR_BANDS = 6;
    static constexpr double kCoeffTable[SR_BANDS][3][NS_ORDER] = {
        // ── SR帯域 0: 44.1kHz ──────────────────────────────────────────────────
        {
            { 2.93, -5.06,  6.97, -7.66,  7.11, -5.63,  3.96, -2.18,  0.80, -0.24,  0.10, -0.04 }, // 16bit 強め
            { 2.49, -4.30,  5.92, -6.51,  6.05, -4.79,  3.37, -1.86,  0.68, -0.20,  0.08, -0.03 }, // 24bit 標準
            { 2.04, -3.52,  4.85, -5.34,  4.95, -3.92,  2.76, -1.52,  0.56, -0.17,  0.07, -0.03 }, // 32bit 控えめ
        },
        // ── SR帯域 1: 48kHz ────────────────────────────────────────────────────
        {
            { 2.85, -4.92,  6.78, -7.45,  6.92, -5.48,  3.85, -2.12,  0.78, -0.23,  0.09, -0.04 }, // 16bit 強め
            { 2.42, -4.18,  5.75, -6.32,  5.87, -4.65,  3.27, -1.80,  0.66, -0.20,  0.08, -0.03 }, // 24bit 標準
            { 1.98, -3.42,  4.71, -5.18,  4.81, -3.81,  2.68, -1.47,  0.54, -0.16,  0.06, -0.03 }, // 32bit 控えめ
        },
        // ── SR帯域 2: 96kHz ────────────────────────────────────────────────────
        {
            { 3.28, -5.66,  7.80, -8.57,  7.96, -6.30,  4.43, -2.44,  0.90, -0.27,  0.11, -0.05 }, // 16bit 強め
            { 2.78, -4.80,  6.61, -7.26,  6.75, -5.34,  3.75, -2.07,  0.76, -0.23,  0.09, -0.04 }, // 24bit 標準
            { 2.28, -3.94,  5.42, -5.95,  5.53, -4.38,  3.08, -1.69,  0.62, -0.19,  0.07, -0.03 }, // 32bit 控えめ
        },
        // ── SR帯域 3: 176.4kHz / 192kHz ────────────────────────────────────────
        {
            { 3.71, -6.40,  8.82, -9.69,  9.00, -7.12,  5.01, -2.76,  1.02, -0.31,  0.12, -0.05 }, // 16bit 強め
            { 3.15, -5.44,  7.50, -8.24,  7.65, -6.05,  4.25, -2.34,  0.86, -0.26,  0.10, -0.04 }, // 24bit 標準
            { 2.58, -4.46,  6.15, -6.75,  6.27, -4.96,  3.48, -1.92,  0.70, -0.21,  0.08, -0.03 }, // 32bit 控えめ
        },
        // ── SR帯域 4: 352.8kHz / 384kHz ────────────────────────────────────────
        {
            { 4.12, -7.10,  9.78,-10.75,  9.98, -7.89,  5.55, -3.06,  1.13, -0.34,  0.14, -0.06 }, // 16bit 強め
            { 3.49, -6.03,  8.31, -9.13,  8.47, -6.70,  4.71, -2.59,  0.95, -0.29,  0.11, -0.05 }, // 24bit 標準
            { 2.86, -4.94,  6.81, -7.48,  6.94, -5.49,  3.86, -2.12,  0.78, -0.23,  0.09, -0.04 }, // 32bit 控えめ
        },
        // ── SR帯域 5: 705.6kHz 以上 ────────────────────────────────────────────
        {
            { 4.48, -7.73, 10.64,-11.70, 10.86, -8.59,  6.04, -3.33,  1.23, -0.37,  0.15, -0.06 }, // 16bit 強め
            { 3.80, -6.56,  9.04, -9.93,  9.22, -7.29,  5.13, -2.82,  1.04, -0.31,  0.12, -0.05 }, // 24bit 標準
            { 3.11, -5.37,  7.41, -8.13,  7.55, -5.97,  4.20, -2.31,  0.85, -0.26,  0.10, -0.04 }, // 32bit 控えめ
        },
    };

    void prepare(double sampleRate, int bitDepth = DEFAULT_BIT_DEPTH) noexcept
    {
        stopRngProducer();

        // bitDepth <= 0 の場合はディザリングが無効化されるため、スケール計算は不要。
        // AudioEngine::DSPCore::processOutput() の applyDither フラグで処理がスキップされる。
        if (bitDepth <= 0)
        {
            clearRandomRing();
            return;
        }

        // Nビット符号付きPCMの量子化ステップは 2 / 2^N = 1 / 2^(N-1)
        scale    = 1.0 / std::pow(2.0, bitDepth - 1);
        invScale = std::pow(2.0, bitDepth - 1);

        // ── SR帯域の選択 ────────────────────────────────────────────────────────
        // 各帯域の中間点をしきい値として使用 (整数比 ×1.5 倍)
        //   44.1k ↔ 48k  分岐点: 46.05kHz
        //   48k ↔ 96k    中間 ≒ 72kHz
        //   96k ↔ 176.4k 中間 ≒ 144kHz
        //   176.4k ↔ 352.8k 中間 ≒ 264.6kHz
        //   352.8k ↔ 705.6k 中間 ≒ 529.2kHz
        int srBand;
        if      (sampleRate <  46050.0) srBand = 0;  // 44.1kHz
        else if (sampleRate <  72000.0) srBand = 1;  // 48kHz
        else if (sampleRate < 144000.0) srBand = 2;  // 96kHz
        else if (sampleRate < 264600.0) srBand = 3;  // 176.4kHz / 192kHz
        else if (sampleRate < 529200.0) srBand = 4;  // 352.8kHz / 384kHz
        else                            srBand = 5;  // 705.6kHz 以上
        jassert(srBand >= 0 && srBand < SR_BANDS);

        // ── ビット深度プリセットの選択 ──────────────────────────────────────────
        int bpIdx;
        if      (bitDepth <= 16) bpIdx = 0;  // 16bit 強め
        else if (bitDepth <= 24) bpIdx = 1;  // 24bit 標準
        else                     bpIdx = 2;  // 32bit 控えめ

        // 係数を実行時バッファにコピー (Audio Threadからはこちらを参照する)
        for (int i = 0; i < NS_ORDER; ++i)
            coeffs[i] = kCoeffTable[srBand][bpIdx][i];

        clearRandomRing();
        prefillRandomRing();
        startRngProducer();

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
            const __m128d v_scale    = _mm_set1_pd(scale);
            const __m128d v_invScale = _mm_set1_pd(invScale);

            // 係数をローカルにキャッシュ (ループ内での繰り返しロードを防ぐ)
            const double c0 = coeffs[0], c1 = coeffs[1], c2 = coeffs[2],
                         c3 = coeffs[3], c4 = coeffs[4], c5 = coeffs[5],
                         c6 = coeffs[6], c7 = coeffs[7], c8 = coeffs[8],
                         c9 = coeffs[9], c10 = coeffs[10], c11 = coeffs[11];

            for (int i = 0; i < numSamples; ++i)
            {
                // 12次 Shaped Error (コンパイラが FMA に最適化)
                const double shapedErrorL = c0*zL[0] + c1*zL[1] + c2*zL[2]
                                          + c3*zL[3] + c4*zL[4] + c5*zL[5]
                                          + c6*zL[6] + c7*zL[7] + c8*zL[8]
                                          + c9*zL[9] + c10*zL[10] + c11*zL[11];
                const double shapedErrorR = c0*zR[0] + c1*zR[1] + c2*zR[2]
                                          + c3*zR[3] + c4*zR[4] + c5*zR[5]
                                          + c6*zR[6] + c7*zR[7] + c8*zR[8]
                                          + c9*zR[9] + c10*zR[10] + c11*zR[11];

                // Dither (MKL)
                const double dL = nextTPDF_MKL(0) * scale;
                const double dR = nextTPDF_MKL(1) * scale;

                // Combine
                const double tmpL = (dataL[i] * headroom) + dL + shapedErrorL;
                const double tmpR = (dataR[i] * headroom) + dR + shapedErrorR;

                // Quantize (SSE4.1 でステレオ同時演算)
                const __m128d v_tmp       = _mm_set_pd(tmpR, tmpL);
                const __m128d v_scaled    = _mm_mul_pd(v_tmp, v_invScale);
                const __m128d v_rounded   = _mm_round_pd(v_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                const __m128d v_quantized = _mm_mul_pd(v_rounded, v_scale);

                alignas(16) double quantized[2];
                _mm_store_pd(quantized, v_quantized); // [quantizedL, quantizedR]

                // Error and State Update (12タップ シフトレジスタ)
                const double errorL = tmpL - quantized[0];
                {
                    for (int t = NS_ORDER - 1; t > 0; --t)
                        zL[t] = zL[t - 1];
                    zL[0] = killDenormal(errorL);
                }
                dataL[i] = quantized[0];

                const double errorR = tmpR - quantized[1];
                {
                    for (int t = NS_ORDER - 1; t > 0; --t)
                        zR[t] = zR[t - 1];
                    zR[0] = killDenormal(errorR);
                }
                dataR[i] = quantized[1];
            }
        }
        else
        {
            // --- Mono Path ---
            const double c0 = coeffs[0], c1 = coeffs[1], c2 = coeffs[2],
                         c3 = coeffs[3], c4 = coeffs[4], c5 = coeffs[5],
                         c6 = coeffs[6], c7 = coeffs[7], c8 = coeffs[8],
                         c9 = coeffs[9], c10 = coeffs[10], c11 = coeffs[11];

            for (int i = 0; i < numSamples; ++i)
            {
                const double shapedError = c0*zL[0] + c1*zL[1] + c2*zL[2]
                                         + c3*zL[3] + c4*zL[4] + c5*zL[5]
                                         + c6*zL[6] + c7*zL[7] + c8*zL[8]
                                         + c9*zL[9] + c10*zL[10] + c11*zL[11];
                const double d   = nextTPDF_MKL(0) * scale;
                const double tmp = (dataL[i] * headroom) + d + shapedError;

                // Quantize (SSE4.1)
                __m128d v = _mm_set_sd(tmp * invScale);
                v = _mm_round_sd(v, v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                const double quantized = _mm_cvtsd_f64(v) * scale;

                const double error = tmp - quantized;
                // シフトレジスタ (モノラル)
                {
                    for (int t = NS_ORDER - 1; t > 0; --t)
                        zL[t] = zL[t - 1];
                    zL[0] = killDenormal(error);
                }
                dataL[i] = quantized;
            }
        }
    }

private:
    static constexpr uint32_t RNG_RING_SIZE = 1u << 16;
    static constexpr uint32_t RNG_RING_MASK = RNG_RING_SIZE - 1u;
    static constexpr uint32_t RNG_REFILL_CHUNK = 2048u;

    static_assert((RNG_RING_SIZE & (RNG_RING_SIZE - 1u)) == 0u, "RNG_RING_SIZE must be power of two");

    inline void clearRandomRing() noexcept
    {
        for (int ch = 0; ch < MAX_CHANNELS; ++ch)
        {
            rngReadPos[ch].store(0u, std::memory_order_relaxed);
            rngWritePos[ch].store(0u, std::memory_order_relaxed);
        }
    }

    inline void fillChunkForChannel(int channel, uint32_t writePos, uint32_t count) noexcept
    {
        const uint32_t start = writePos & RNG_RING_MASK;
        const uint32_t firstCount = std::min(count, RNG_RING_SIZE - start);
        const uint32_t secondCount = count - firstCount;

        auto fillWithFallback = [this, channel](uint32_t offset, uint32_t n) noexcept {
            for (uint32_t i = 0; i < n; ++i)
            rngRing[channel][offset + i] = 0.5;
        };

        if (rng[channel].isValid())
        {
            const MKL_INT status1 = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rng[channel],
                                                 static_cast<MKL_INT>(firstCount),
                                                 &rngRing[channel][start], 0.0, 1.0);
            if (status1 != VSL_STATUS_OK)
            {
                fillWithFallback(start, firstCount);
            }

            if (secondCount > 0u)
            {
                const MKL_INT status2 = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rng[channel],
                                                     static_cast<MKL_INT>(secondCount),
                                                     &rngRing[channel][0], 0.0, 1.0);
                if (status2 != VSL_STATUS_OK)
                    fillWithFallback(0u, secondCount);
            }
        }
        else
        {
            fillWithFallback(start, firstCount);
            if (secondCount > 0u)
                fillWithFallback(0u, secondCount);
        }
    }

    void prefillRandomRing() noexcept
    {
        for (int ch = 0; ch < MAX_CHANNELS; ++ch)
        {
            fillChunkForChannel(ch, 0u, RNG_RING_SIZE);
            rngWritePos[ch].store(RNG_RING_SIZE, std::memory_order_release);
        }
    }

    void startRngProducer() noexcept
    {
        rngProducerStopRequested.store(false, std::memory_order_release);
        try
        {
            rngProducerThread = std::thread([this]() { rngProducerLoop(); });
            rngProducerRunning.store(true, std::memory_order_release);
        }
        catch (...)
        {
            rngProducerRunning.store(false, std::memory_order_release);
            rngProducerStopRequested.store(true, std::memory_order_release);
        }
    }

    void stopRngProducer() noexcept
    {
        const bool wasRunning = rngProducerRunning.exchange(false, std::memory_order_acq_rel);
        if (!wasRunning)
            return;

        rngProducerStopRequested.store(true, std::memory_order_release);
        if (rngProducerThread.joinable())
            rngProducerThread.join();
    }

    void rngProducerLoop() noexcept
    {
        while (!rngProducerStopRequested.load(std::memory_order_acquire))
        {
            bool didWork = false;

            for (int ch = 0; ch < MAX_CHANNELS; ++ch)
            {
                const uint32_t readPos = rngReadPos[ch].load(std::memory_order_acquire);
                const uint32_t writePos = rngWritePos[ch].load(std::memory_order_relaxed);
                const uint32_t available = writePos - readPos;
                const uint32_t freeSpace = RNG_RING_SIZE - available;

                if (freeSpace >= RNG_REFILL_CHUNK)
                {
                    fillChunkForChannel(ch, writePos, RNG_REFILL_CHUNK);
                    rngWritePos[ch].store(writePos + RNG_REFILL_CHUNK, std::memory_order_release);
                    didWork = true;
                }
            }

            if (!didWork)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    inline double fallbackUniform(int channel) noexcept
    {
        uint64_t x = fallbackState[channel];
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        fallbackState[channel] = x;

        constexpr uint64_t mul = 2685821657736338717ULL;
        const uint64_t z = x * mul;
        constexpr double inv53 = 1.0 / 9007199254740992.0;
        return static_cast<double>(z >> 11) * inv53;
    }

    inline double popUniformFromRing(int channel) noexcept
    {
        // SPSC (Single Producer / Single Consumer) リングバッファ前提:
        // - consumer 側 readPos はこのスレッドのみが更新するため relaxed で十分。
        // - producer が rngWritePos を release store した後のリング書き込み可視性は、
        //   consumer 側の rngWritePos acquire load で同期される。
        // - したがってここで readPos を acquire に強めても正しさは変わらず、
        //   主同期点は writePos (release/acquire) のペアとなる。
        const uint32_t readPos = rngReadPos[channel].load(std::memory_order_relaxed);
        const uint32_t writePos = rngWritePos[channel].load(std::memory_order_acquire);

        if (readPos == writePos)
            return fallbackUniform(channel);

        const double value = rngRing[channel][readPos & RNG_RING_MASK];
        rngReadPos[channel].store(readPos + 1u, std::memory_order_release);
        return value;
    }

    inline double processChannelMKL(double x, int channel, double* z) noexcept
    {
        // TPDFディザ生成 (MKL)
        const double d = nextTPDF_MKL(channel) * scale;

        // 12次ノイズシェーパー (フィードバック誤差)
        const double shapedError =
              coeffs[0] * z[0]
            + coeffs[1] * z[1]
            + coeffs[2] * z[2]
            + coeffs[3] * z[3]
            + coeffs[4] * z[4]
            + coeffs[5] * z[5]
            + coeffs[6] * z[6]
            + coeffs[7] * z[7]
            + coeffs[8] * z[8]
            + coeffs[9] * z[9]
            + coeffs[10] * z[10]
            + coeffs[11] * z[11];

        // ディザとノイズシェーピングの適用
        const double tmp = x + d + shapedError;

        // 量子化 (Round to nearest even using SSE4.1)
        __m128d v = _mm_set_sd(tmp * invScale);
        v = _mm_round_sd(v, v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const double quantized = _mm_cvtsd_f64(v) * scale;

        // 量子化誤差の計算 (Input - Output)
        // 注意: これは -(Output - Input) と等価です。
        // shapedError は過去の誤差に正の係数を掛けて加算しているため、
        // これは実質的に量子化ノイズを減算することになります (1 - H(z) トポロジー)。
        const double error = tmp - quantized;

        // 状態の更新 (12タップ シフトレジスタ)
        {
            for (int t = NS_ORDER - 1; t > 0; --t)
                z[t] = z[t - 1];
            z[0] = killDenormal(error);
        }

        return quantized;
    }

    alignas(64) double rngRing[MAX_CHANNELS][RNG_RING_SIZE] {};
    std::atomic<uint32_t> rngReadPos[MAX_CHANNELS] {};
    std::atomic<uint32_t> rngWritePos[MAX_CHANNELS] {};
    uint64_t fallbackState[MAX_CHANNELS] {};

    std::thread rngProducerThread;
    std::atomic<bool> rngProducerRunning { false };
    std::atomic<bool> rngProducerStopRequested { false };

    inline double nextTPDF_MKL(int channel) noexcept
    {
        const double u1 = popUniformFromRing(channel);
        const double u2 = popUniformFromRing(channel);
        return (u1 - 0.5) + (u2 - 0.5);
    }

    inline double killDenormal(double x) const noexcept
    {
        return (std::fabs(x) < convo::numeric_policy::kDenormThresholdAudioState) ? 0.0 : x;
    }

    VSLStream rng[MAX_CHANNELS];

    double* shaperStateBuffer = nullptr;
    double scale    = 1.0 / 8388608.0;  // 2^23（24bit signed PCM デフォルト）
    double invScale = 8388608.0;         // 2^23（24bit signed PCM デフォルト）

    // 実行時係数バッファ (prepare() で kCoeffTable から選択・コピー)
    // Audio Thread からはこちらを参照する (constexpr テーブルへの直接アクセスを避ける)
    std::array<double, NS_ORDER> coeffs {};
};

} // namespace convo
