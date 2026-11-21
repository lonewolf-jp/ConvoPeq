#pragma once

//============================================================================
// TailLayerMultiplier.h  ── tailL1L2Multiplier の 2 冪制約（BUG-013）
//
// MKLNonUniformConvolver は L2（テール第 2 層）のパーティション長を
//     l2Part = l1Part * tailL1L2Multiplier
// で決め、その FFT 長を 2 * l2Part にする。Intel IPP の実数 FFT
// （ippsFFTGetSize_R_64f / ippsFFTInit_R_64f）は 2 の冪長しか扱えないため、
// 倍率が 2 の冪でないと FFT プランが実長より小さく作られ、L2 テールの
// 周波数領域データが壊れる（既定値 8 は 2 の冪なので既定構成では発火しない）。
//
// 倍率の契約を 1 箇所に閉じ込めるため、丸め規則をこのヘッダに置く。
// 呼出側（MKLNonUniformConvolver::setImpulse）はこの関数だけを使うこと。
//
// 丸め規則: [kTailL1L2MultMin, kTailL1L2MultMax] の 2 の冪のうち、
//   要求値に最も近いもの。距離が等しいときは大きい方（テール分解能を削らない）。
//============================================================================

#include <cstdint>

namespace convo
{

inline constexpr int kTailL1L2MultMin     = 2;
inline constexpr int kTailL1L2MultMax     = 16;
inline constexpr int kTailL1L2MultDefault = 8;

/// v が 2 の冪（1, 2, 4, ...）なら true。0 と負値は false。
[[nodiscard]] constexpr bool isPowerOfTwoValue(int v) noexcept
{
    return v > 0 && (v & (v - 1)) == 0;
}

/// 要求された倍率を [kTailL1L2MultMin, kTailL1L2MultMax] の 2 の冪へ丸める。
/// 戻り値は必ず 2 の冪であり、kTailL1L2MultMin <= 戻り値 <= kTailL1L2MultMax。
[[nodiscard]] constexpr int sanitizeTailL1L2Multiplier(int requested) noexcept
{
    if (requested <= kTailL1L2MultMin)
        return kTailL1L2MultMin;

    if (requested >= kTailL1L2MultMax)
        return kTailL1L2MultMax;

    // requested 以下の最大の 2 冪
    int lower = kTailL1L2MultMin;
    while (lower * 2 <= requested)
        lower *= 2;

    const int upper = lower * 2;

    if (upper > kTailL1L2MultMax)
        return kTailL1L2MultMax;

    // 近い方へ。同距離は大きい方（テール分解能を削らない）。
    return (requested - lower >= upper - requested) ? upper : lower;
}

/// Air Absorption (0) と Layer Tail Contouring (1) は L2 の分解能を保つため、
/// 倍率を 8 以上へ引き上げる（MKLNonUniformConvolver::SetImpulse の規則）。
/// 保存値・表示値・実効値を一致させるため、setter 側もこの関数を使う。
[[nodiscard]] constexpr int effectiveTailL1L2Multiplier(int requested, int tailMode) noexcept
{
    const int sanitized = sanitizeTailL1L2Multiplier(requested);
    if (tailMode == 0 || tailMode == 1)
        return (sanitized < 8) ? 8 : sanitized;
    return sanitized;
}

} // namespace convo
