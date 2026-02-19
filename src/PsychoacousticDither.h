//============================================================================
#pragma once
// PsychoacousticDither.h
//
// Psychoacoustic Noise Shaping Dither (POW-R Type 1 approximation)
//
// 人間の聴覚特性を考慮したディザリング (Error Feedback Topology)
// 2-4kHz帯域のノイズを削減し、15kHz以上に移動させます。
//
// リファレンス:
//   - POW-r Dithering: https://en.wikipedia.org/wiki/Dither#Noise_shaping
//   - Lipshitz, Vanderkooy, "A Theory of Nonsubtractive Dither" (1992)
//============================================================================

#include <JuceHeader.h>
#include <cmath>
#include <optional>

//--------------------------------------------------------------
// Xoshiro256** PRNG Implementation
// 高速かつ高品質な乱数生成器 (周期 2^256 - 1)
// オーディオディザリングに最適
//--------------------------------------------------------------
struct Xoshiro256
{
    uint64_t s[4];

    static inline uint64_t rotl(const uint64_t x, int k)
    {
        return (x << k) | (x >> (64 - k));
    }

    void seed(uint64_t seed)
    {
        // SplitMix64で初期状態を生成
        for (int i = 0; i < 4; ++i)
        {
            seed += 0x9E3779B97F4A7C15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            s[i] = z ^ (z >> 31);
        }
    }

    // [0, 1) のdouble乱数を生成
    double nextDouble()
    {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl(s[3], 45);

        // 53-bit precision double
        return (result >> 11) * (1.0 / (1ULL << 53));
    }
};

class PsychoacousticDither
{
public:
    static constexpr int DEFAULT_BIT_DEPTH = 24;

    explicit PsychoacousticDither(std::optional<uint64_t> seed = std::nullopt)
    {
        // シードの初期化 (SplitMix64による非相関化)
        initialiseSeeds(seed);
    }

    //----------------------------------------------------------
    // 準備: サンプルレートとビット深度を設定
    //----------------------------------------------------------
    void prepare(double /*sampleRate*/, int bitDepth = DEFAULT_BIT_DEPTH) noexcept
    {
        // 量子化ステップサイズ (1 LSB) の計算
        // 24bit signed の場合: 2^23 = 8,388,608
        scale = std::pow(2.0, static_cast<double>(bitDepth - 1));
        invScale = 1.0 / scale;

        // TPDFディザ振幅 (通常 2 LSB peak-to-peak)
        // ここでは -1 LSB ~ +1 LSB の範囲とする
        ditherAmplitude = 1.0 / scale;

        reset();
    }

    //----------------------------------------------------------
    // 処理: ノイズシェーピングディザを適用して量子化
    //----------------------------------------------------------
    double process(double input, int channel) noexcept
    {
        // チャンネルに応じた乱数生成器と状態変数を選択
        Xoshiro256& random = (channel == 0) ? randomL : randomR;
        double& s1 = (channel == 0) ? state1L : state1R;
        double& s2 = (channel == 0) ? state2L : state2R;

        // 1. TPDF (三角確率密度関数) ディザ生成
        const double r1 = random.nextDouble() * 2.0 - 1.0;
        const double r2 = random.nextDouble() * 2.0 - 1.0;
        const double tpdf = (r1 - r2) * 0.5 * ditherAmplitude;

        // 2. 過去の量子化誤差をフィードバック (Error Feedback)
        const double shapedError = shaping_a1 * s1 + shaping_a2 * s2;
        const double signalToQuantize = input - shapedError;

        // 3. ディザを加えて量子化
        const double ditheredSignal = signalToQuantize + tpdf;
        double quantized = std::round(ditheredSignal * scale) * invScale;

        // 4. 量子化誤差を計算し、次のサンプルのために状態を更新
        //    誤差 = (量子化後の値) - (量子化前の値)
        const double error = quantized - signalToQuantize;
        s2 = s1;
        s1 = error;

        // 5. Denormal対策
        if (std::abs(s1) < 1.0e-15) s1 = 0.0;
        if (std::abs(s2) < 1.0e-15) s2 = 0.0;

        return quantized;
    }

    //----------------------------------------------------------
    // リセット: 内部状態をクリア
    //----------------------------------------------------------
    void reset() noexcept
    {
        state1L = 0.0;
        state2L = 0.0;
        state1R = 0.0;
        state2R = 0.0;
    }

private:
    Xoshiro256 randomL;
    Xoshiro256 randomR;
    double scale = 8388608.0;
    double invScale = 1.0 / 8388608.0;
    double ditherAmplitude = 1.0 / 8388608.0;

    // POW-R Type 1 近似係数
    // 2-4kHz帯域で約-6dB、15kHz以上で+3dBの特性を持つハイパス型シェーパー
    static constexpr double shaping_a1 = 2.033;
    static constexpr double shaping_a2 = -1.165;

    // 状態変数 (State Variables)
    double state1L = 0.0;
    double state2L = 0.0;
    double state1R = 0.0;
    double state2R = 0.0;

    // SplitMix64 PRNG (シード生成用)
    static uint64_t splitmix64(uint64_t& x)
    {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // シード初期化
    void initialiseSeeds(std::optional<uint64_t> seed)
    {
        uint64_t baseSeed;

        if (seed.has_value())
            baseSeed = seed.value();
        else
            baseSeed = (uint64_t)juce::Random::getSystemRandom().nextInt64();

        uint64_t s = baseSeed;

        uint64_t seedL = splitmix64(s);
        uint64_t seedR = splitmix64(s);

        randomL.seed(seedL);
        randomR.seed(seedR);
    }
};