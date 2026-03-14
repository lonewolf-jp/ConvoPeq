//==============================================================================
#pragma once

namespace convo::numeric_policy
{
    // Audio DSP の状態変数 (IIR/SVF/DC blocker 等) 向け。
    // 音質影響が実質ゼロの極小領域のみを明示フラッシュする。
    inline constexpr double kDenormThresholdAudioState = 1.0e-20;

    // 入力サニタイズ向け (NaN/Inf除去 + 極小値クリーンアップ)。
    // AudioState よりわずかに低い閾値を使い、前処理で過剰に削らない。
    inline constexpr double kDenormThresholdInputSanitize = 1.0e-25;

    // スペアナ/UI表示向け float 処理のノイズ床目安 (音声本流には使わない)。
    inline constexpr float kAnalyzerFloatFloor = 1.0e-12f;
}
