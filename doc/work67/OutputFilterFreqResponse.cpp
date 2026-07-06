//==============================================================================
// OutputFilterFreqResponse.cpp
// ★ [P2-4] OutputFilter 周波数応答測定ツール
//
// 48kHz 動作時、Sharp モード (fc≈19kHz) の可聴帯域上端における
// マグニチュード平坦性を確認する。
//
// ビルド: テストプロジェクトとして CMake に追加するか、別途コンパイル
//   cl /std:c++20 /EHsc /arch:AVX2 /I.. /I../../JUCE/modules OutputFilterFreqResponse.cpp
//==============================================================================

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// OutputFilter は JUCE 非依存では使用できないため、
// ここでは同等の Biquad フィルタを実装して測定する。
// OutputFilter.cpp の makeLPF と同じ RBJ Cookbook 係数を使用。

struct BiquadCoeff {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
};

struct BiquadState {
    double w1 = 0.0, w2 = 0.0;
    double process(double x, const BiquadCoeff& c) noexcept {
        const double y = c.b0 * x + w1;
        w1 = c.b1 * x - c.a1 * y + w2;
        w2 = c.b2 * x - c.a2 * y;
        return y;
    }
};

// RBJ Cookbook LPF
BiquadCoeff makeLPF(double fc, double Q, double fs) noexcept {
    const double nyq = fs * 0.4999;
    if (fc >= nyq || Q <= 0.0 || fs <= 0.0) return {1,0,0,0,0};
    const double w0    = 2.0 * 3.141592653589793 * fc / fs;
    const double sn    = std::sin(w0);
    const double cs    = std::cos(w0);
    const double alpha = sn / (2.0 * Q);
    const double a0inv = 1.0 / (1.0 + alpha);
    BiquadCoeff c;
    c.b0 = (1.0 - cs) * 0.5 * a0inv;
    c.b1 = (1.0 - cs) * a0inv;
    c.b2 = (1.0 - cs) * 0.5 * a0inv;
    c.a1 = (-2.0 * cs) * a0inv;
    c.a2 = (1.0 - alpha) * a0inv;
    return c;
}

// 複数周波数でのマグニチュードを測定
// 入力: fs=48000, SharpモードHC: fc=19000, Q1=0.5412, Q2=1.3066 (4次カスケード)
// NaturalモードHC: fc=22000, Q1=Q2=0.7071
// SoftモードHC: fc=19000, Q1=0.5, Q2=identity
int main() {
    constexpr double fs = 48000.0;
    constexpr int kNumSamples = 48000 * 2; // 2秒
    constexpr double kAmp = 0.5;

    struct ModeConfig {
        const char* name;
        double fc;
        double q1, q2;
    };

    ModeConfig modes[] = {
        {"Sharp_HC",   19000.0, 0.5412, 1.3066},
        {"Natural_HC", 22000.0, 0.7071, 0.7071},
        {"Soft_HC",    19000.0, 0.5,    0.0},    // Q2=0 → identity
        {"EQ_LPF_Sharp",   19000.0, 0.5412, 1.3066},
        {"EQ_LPF_Natural", 24000.0, 0.7071, 0.7071},
        {"EQ_LPF_Soft",    19000.0, 0.5,    0.0},
    };

    // 測定周波数 (Hz): 可聴帯域上端 + ナイキスト近傍
    double testFreqs[] = {
        100, 1000, 5000, 10000, 15000, 17000, 18000, 19000, 19500,
        20000, 21000, 22000, 23000, 23900
    };

    std::printf("=== OutputFilter 周波数応答測定 ===\n");
    std::printf("サンプルレート: %.0f Hz\n", fs);
    std::printf("測定信号: %.1f秒 正弦波スイープ\n\n", (double)kNumSamples / fs);

    for (const auto& mode : modes) {
        // 係数準備
        BiquadCoeff c1 = makeLPF(mode.fc, mode.q1, fs);
        BiquadCoeff c2 = (mode.q2 > 0.01) ? makeLPF(mode.fc, mode.q2, fs) : BiquadCoeff{1,0,0,0,0};

        std::printf("--- %s (fc=%.0f Hz, Q1=%.4f, Q2=%.4f) ---\n",
                    mode.name, mode.fc, mode.q1, mode.q2);

        for (double freq : testFreqs) {
            BiquadState s1, s2;
            const double omega = 2.0 * 3.141592653589793 * freq / fs;

            // 定常状態になるまで十分なサンプルを処理
            constexpr int kSettle = 48000; // 1秒分
            double maxOutput = 0.0;

            for (int i = 0; i < kSettle; ++i) {
                const double input = kAmp * std::sin(omega * i);
                double out = s1.process(input, c1);
                if (mode.q2 > 0.01) out = s2.process(out, c2);
                if (std::abs(out) > maxOutput) maxOutput = std::abs(out);
            }

            const double magnitudeDb = 20.0 * std::log10(maxOutput / kAmp);
            // ±0.1dB 以内なら PASS
            const bool pass = (std::abs(magnitudeDb) < 0.11);

            std::printf("  %6.0f Hz: mag = %+.3f dB  [%s]\n",
                        freq, magnitudeDb, pass ? "PASS" : "FAIL");
        }
        std::printf("\n");
    }

    // 総合評価
    std::printf("=== 総合評価 ===\n");
    std::printf("Sharp モード fc=19kHz @48kHz: 可聴帯域上端(20kHz)でのロールオフ確認\n");
    std::printf("基準: ±0.1dB 以内 = PASS\n");

    return 0;
}
