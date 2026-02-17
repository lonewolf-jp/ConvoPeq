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
        // 1. TPDF (三角確率密度関数) ディザ生成
        const double r1 = random.nextDouble() * 2.0 - 1.0;
        const double r2 = random.nextDouble() * 2.0 - 1.0;
        const double tpdf = (r1 - r2) * 0.5 * ditherAmplitude;

        // 2. 過去の量子化誤差をフィードバック (Error Feedback)
        //    入力信号から、シェーピングされた過去の誤差を減算する。
        //    これにより、量子化ノイズの周波数特性が変化する（ノイズシェーピング）。
        const double shapedError = shaping_a1 * state1 + shaping_a2 * state2;
        const double signalToQuantize = input - shapedError;

        // 3. ディザを加えて量子化
        const double ditheredSignal = signalToQuantize + tpdf;
        double quantized = std::round(ditheredSignal * scale) * invScale;

        // 4. 量子化誤差を計算し、次のサンプルのために状態を更新
        //    誤差 = (量子化後の値) - (量子化前の値)
        const double error = quantized - ditheredSignal;
        state2 = state1;
        state1 = error;

        // 5. Denormal対策
        if (std::abs(state1) < 1.0e-15) state1 = 0.0;
        if (std::abs(state2) < 1.0e-15) state2 = 0.0;

        return quantized;
    }

    //----------------------------------------------------------
    // リセット: 内部状態をクリア
    //----------------------------------------------------------
    void reset() noexcept
    {
        state1 = 0.0;
        state2 = 0.0;
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

    // 状態変数 (State Variables)
    double state1 = 0.0;
    double state2 = 0.0;
};