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

class PsychoacousticDither
{
public:
    static constexpr int DEFAULT_BIT_DEPTH = 24;

    PsychoacousticDither()
    {
        // ステレオチャンネル間で相関を持たないようにランダムシードを設定
        random.setSeedRandomly();
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
    double process(double input) noexcept
    {
        // 1. TPDF (Triangular Probability Density Function) ディザ生成
        // r1, r2 は [-1.0, 1.0]
        // nextFloat() returns [0.0, 1.0)
        const double r1 = random.nextDouble() * 2.0 - 1.0;
        const double r2 = random.nextDouble() * 2.0 - 1.0;

        // TPDFノイズ: (r1 - r2) は [-2.0, 2.0] なので 0.5倍して [-1.0, 1.0] に正規化し、振幅を掛ける
        const double tpdf = (r1 - r2) * 0.5 * ditherAmplitude;

        // 2. Error Feedback Noise Shaping
        // ノイズ伝達関数 (NTF): H(z) = 1 - 2.033 z^-1 + 1.165 z^-2
        // 入力に加算する補正項: - (H(z) - 1) * E(z)
        // Correction = - (-2.033 * e1 + 1.165 * e2) = 2.033 * e1 - 1.165 * e2

        const double shapedError = shaping_a1 * e1 + shaping_a2 * e2;

        // ディザ加算とシェーピング済み誤差の減算
        double current = input + tpdf - shapedError;

        // 3. 量子化 (Quantize)
        // ターゲットビット深度のグリッドに合わせて丸める
        double quantized = std::round(current * scale) * invScale;

        // 4. 誤差計算 (Error Calculation)
        // Error = Quantized - Ideal
        double error = quantized - current;

        // 5. 状態更新
        e2 = e1;
        e1 = error;

        // Denormal対策 (極小値の循環防止)
        if (std::abs(e1) < 1.0e-15) e1 = 0.0;
        if (std::abs(e2) < 1.0e-15) e2 = 0.0;

        return quantized;
    }

    //----------------------------------------------------------
    // リセット: 内部状態をクリア
    //----------------------------------------------------------
    void reset() noexcept
    {
        e1 = 0.0;
        e2 = 0.0;
    }

private:
    juce::Random random;
    double scale = 8388608.0;
    double invScale = 1.0 / 8388608.0;
    double ditherAmplitude = 1.0 / 8388608.0;

    // POW-R Type 1 近似係数
    // 2-4kHz帯域で約-6dB、15kHz以上で+3dBの特性を持つハイパス型シェーパー
    static constexpr double shaping_a1 = 2.033;
    static constexpr double shaping_a2 = -1.165;

    // 誤差履歴 (Error History)
    double e1 = 0.0;
    double e2 = 0.0;
};