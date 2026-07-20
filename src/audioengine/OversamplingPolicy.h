#pragma once

#include "RuntimeBuildTypes.h"

namespace convo {

//==============================================================================
// ★ v14.27: OversamplingPolicy — オーバーサンプリング倍率決定ポリシー。
//
//   ISR Authority Singularization: resolve() が倍率決定における唯一の決定権限。
//   Builder 専有のポリシー。Planner は決定ロジックを一切知らず、
//   Snapshot.oversampling.resolvedOsFactor を読み取り専用で参照する。
//
//   入力 SR と許可倍率（ルックアップ方式）:
//     入力SR        許可倍率
//     44.1kHz       x1, x2, x4, x8
//     48.0kHz       x1, x2, x4, x8
//     88.2kHz       x1, x2, x4, x8
//     96.0kHz       x1, x2, x4, x8
//     176.4kHz      x1, x2, x4
//     192.0kHz      x1, x2, x4
//     352.8kHz      x1, x2
//     384.0kHz      x1, x2
//     705.6kHz      x1
//     768.0kHz      x1
//     > 768kHz      入力不可（supported=false）
//
//   最大許可倍率の決定（SR→maxFactor ルックアップ）:
//     sr ≤ 96000   → maxFactor = 8   （96k x8 = 768k ≤ 768k）
//     sr ≤ 192000  → maxFactor = 4   （192k x4 = 768k）
//     sr ≤ 384000  → maxFactor = 2   （384k x2 = 768k）
//     sr ≤ 768000  → maxFactor = 1   （768k x1 = 768k）
//     sr > 768000  → maxFactor = 0   （入力不可）
//==============================================================================
struct OversamplingPolicy {
    static constexpr double kMaxInternalRate = 768000.0;
    static constexpr int kMaxFactor = 8;

    // ★ v14.45: maxAllowedFactor() — 指定 SR における最大許可倍率を返す。
    //   GUI などが参照可能で、決定権限は持たない（Authority は resolve()）。
    //   名前の通り「許可される最大倍率」を返し、集合（{1,2,4,8}等）は返さない。
    [[nodiscard]] static int maxAllowedFactor(double sampleRate) noexcept
    {
        if (sampleRate <= 96000.0)   return 8;
        if (sampleRate <= 192000.0)  return 4;
        if (sampleRate <= 384000.0)  return 2;
        if (sampleRate <= 768000.0)  return 1;
        return 0;  // 768kHz 超: 許可倍率なし（supported==false）
    }

    // ★ v14.30: resolve() — 唯一の決定権限。
    //   BuildInput から OversamplingResult を生成する。
    //   - requestedOsFactor が {0,1,2,4,8} 以外の異常値の場合、Auto 扱い
    //   - requestedOsFactor > 0 なら、requestedOsFactor ≤ 最大許可倍率 であることを検証
    //   - 不整合時は最大許可倍率を使用（安全側フォールバック）
    //   - resolvedOsFactor は常に power-of-2（1, 2, 4, 8 のみ）
    [[nodiscard]] static OversamplingResult resolve(const BuildInput& input) noexcept
    {
        OversamplingResult result{};
        result.requestedOsFactor = input.oversamplingFactor;

        const int maxF = maxAllowedFactor(input.sampleRate);
        result.supported = (maxF > 0);

        if (maxF == 0) {
            result.resolvedOsFactor = 1;  // 最小倍率（supported==false）
            result.isAutoResolved = true;
            return result;
        }

        result.isAutoResolved = (input.oversamplingFactor == 0);

        // 異常値フォールバック
        int effectiveRequested = input.oversamplingFactor;
        if (effectiveRequested != 0 && effectiveRequested != 1 && effectiveRequested != 2
            && effectiveRequested != 4 && effectiveRequested != 8)
            effectiveRequested = 0;  // 異常値 → Auto 扱い

        if (effectiveRequested > 0)
            result.resolvedOsFactor = (effectiveRequested <= maxF) ? effectiveRequested : maxF;
        else
            result.resolvedOsFactor = maxF;  // Auto: 最大許可倍率

        return result;
    }
};

} // namespace convo
