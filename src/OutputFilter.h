//============================================================================
#pragma once
// OutputFilter.h  ── v1.0
//
// 出力周波数フィルター
//
// ■ 機能:
//   ① コンボルバーが最終段（convolver単体 / EQ→Convolver）の場合:
//       - ハイカットフィルター: Sharp(Butterworth 4次) / Natural(LR 4次) / Soft(2次)
//         fc = 19kHz (fs≤48kHz) / 22kHz (fs>48kHz)
//       - ローカットフィルター: Natural(2次Butterworth HPF 18Hz) / Soft(2次 HPF Q=0.5 15Hz)
//
//   ② EQが最終段（EQ単体 / Convolver→EQ）の場合:
//       - ハイパスフィルター (固定): Butterworth 2次, 20Hz
//       - ローパスフィルター: Sharp(Q=1.0) / Natural(Q=0.7071) / Soft(Q=0.5) × 2段
//         fc = 19kHz (fs≤48kHz) / 24kHz (fs>48kHz)
//
// ■ スレッド安全設計:
//   - prepare()   : Message Thread から呼ぶ (std::sin/cos使用)
//   - process()   : Audio Thread から呼ぶ (libm呼び出しなし・メモリ確保なし)
//   - reset()     : Audio Thread から呼ぶ (フィルター状態クリア)
//   - 全モード分の係数を prepare() で事前計算し、process() はテーブル参照のみ
//============================================================================

#include <JuceHeader.h>
#include <cmath>
#include <array>

namespace convo {

//──────────────────────────────────────────────────────────────────────────
// Biquad フィルター係数 (Direct Form II Transposed)
//
//   y[n] = b0·x[n] + w1[n-1]
//   w1[n] = b1·x[n] − a1·y[n] + w2[n-1]
//   w2[n] = b2·x[n] − a2·y[n]
//
// a0 は除算済み（正規化済み係数として保持）
//──────────────────────────────────────────────────────────────────────────
struct BiquadCoeff
{
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;
};

//──────────────────────────────────────────────────────────────────────────
// Biquad 状態変数 (チャンネル/ステージごとに保持)
//──────────────────────────────────────────────────────────────────────────
struct BiquadState
{
    double w1 = 0.0, w2 = 0.0;

    void reset() noexcept { w1 = w2 = 0.0; }

    // Audio Thread 安全: libm 呼び出しなし
    // FTZ/DAZ フラグがスレッドに設定済みであることを前提とするが、
    // 念のため閾値判定によるデノーマル対策も行う
    inline double process(double x, const BiquadCoeff& c) noexcept
    {
        const double y = c.b0 * x + w1;
        w1 = c.b1 * x - c.a1 * y + w2;
        w2 = c.b2 * x - c.a2 * y;
        // デノーマル対策
        constexpr double kDenorm = 1.0e-20;
        if (w1 > -kDenorm && w1 < kDenorm) w1 = 0.0;
        if (w2 > -kDenorm && w2 < kDenorm) w2 = 0.0;
        return y;
    }
};

//──────────────────────────────────────────────────────────────────────────
// ハイカット / EQ ローパスフィルターモード
// ① ハイカット と ② EQ ローパスの両方で使用
//──────────────────────────────────────────────────────────────────────────
enum class HCMode
{
    Sharp   = 0,  // Butterworth 4次カスケード (急峻、境界まで音圧維持)
    Natural = 1,  // Linkwitz-Riley 4次 (デフォルト、位相特性良好)
    Soft    = 2   // 2次 Q=0.5 (緩やか、時間軸の滲みなし)
};

//──────────────────────────────────────────────────────────────────────────
// ローカットフィルターモード (①コンボルバー最終段でのみ使用)
//──────────────────────────────────────────────────────────────────────────
enum class LCMode
{
    Natural = 0,  // Butterworth 2次 HPF, 18Hz (余韻の乱れ最小)
    Soft    = 1   // 2次 HPF Q=0.5, 15Hz (より穏やか、サブソニック除去)
};

//──────────────────────────────────────────────────────────────────────────
// OutputFilter
//──────────────────────────────────────────────────────────────────────────
class OutputFilter
{
public:
    OutputFilter() = default;
    ~OutputFilter() = default;

    //------------------------------------------------------------------
    // prepare() ── Message Thread のみ (std::sin / std::cos 使用)
    // サンプルレートに応じた全モード分の係数を事前計算する
    // DSPCore::prepare() から呼ぶこと
    //------------------------------------------------------------------
    void prepare(double sampleRate) noexcept;

    //------------------------------------------------------------------
    // reset() ── Audio Thread 安全
    // 全フィルター状態変数をゼロにリセットする
    // DSPCore::reset() および モード変更時のフェード後に呼ぶ
    //------------------------------------------------------------------
    void reset() noexcept;

    //------------------------------------------------------------------
    // process() ── Audio Thread のみ (libm 呼び出しなし・メモリ確保なし)
    //
    // @param block       処理対象の AudioBlock (インプレース)
    // @param convIsLast  true  = ① コンボルバー最終段
    //                    false = ② EQ最終段
    // @param hcMode      ハイカットモード (① で使用)
    // @param lcMode      ローカットモード (① で使用)
    // @param lpMode      ローパスモード   (② で使用)
    //------------------------------------------------------------------
    void process(juce::dsp::AudioBlock<double>& block,
                 bool convIsLast,
                 HCMode hcMode,
                 LCMode lcMode,
                 HCMode lpMode) noexcept;

private:
    //──── 事前計算済み係数 (prepare()で設定、process()で参照) ────────

    // ① ハイカット: 3モード × 2カスケード段
    // Sharp  : Butterworth 4次 (Q1=0.5412, Q2=1.3066 を各段に設定)
    // Natural: Linkwitz-Riley 4次 (Q=0.7071 を両段に設定)
    // Soft   : 2次 Q=0.5 (stage[1]=identity)
    BiquadCoeff hcCoeff[3][2]; // [HCMode][stage 0/1]

    // ① ローカット: 2モード × 1段 (単一2次HPF)
    BiquadCoeff lcCoeff[2];    // [LCMode]

    // ② ハイパス: 固定 (Butterworth 2次, 20Hz)
    BiquadCoeff hpfCoeff;

    // ② ローパス: 3モード × 2カスケード段
    BiquadCoeff lpCoeff[3][2]; // [HCMode][stage 0/1]

    //──── フィルター状態変数 (チャンネル/ステージごと) ────────────────
    // MAX_CHANNELS=2 (ステレオ固定)

    BiquadState hcState[2][2]; // [ch][stage]  ① ハイカット
    BiquadState lcState[2];    // [ch]          ① ローカット
    BiquadState hpfState[2];   // [ch]          ② ハイパス
    BiquadState lpState[2][2]; // [ch][stage]  ② ローパス

    //──── 係数計算ヘルパー (Message Thread 専用、std::sin/cos 使用) ──

    // 2次ローパスフィルター係数 (RBJ Audio EQ Cookbook)
    // fc がナイキスト (fs*0.4999) 以上の場合は identity を返す
    static BiquadCoeff makeLPF(double fc, double Q, double fs) noexcept;

    // 2次ハイパスフィルター係数 (RBJ Audio EQ Cookbook)
    static BiquadCoeff makeHPF(double fc, double Q, double fs) noexcept;

    // 恒等変換係数 (b0=1, 他=0)
    static BiquadCoeff makeIdentity() noexcept;

    JUCE_DECLARE_NON_COPYABLE(OutputFilter)
};

} // namespace convo
