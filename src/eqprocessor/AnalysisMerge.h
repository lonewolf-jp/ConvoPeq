#pragma once

#include "EQAnalysisTypes.h"
#include <vector>
#include <algorithm>
#include <cmath>

//==============================================================================
// 統合ユーティリティ — mergeAndSort / deduplicate / renumber
//
// 責務:
// - mergeAndSort: coarse + adaptive を周波数昇順に stable_sort
// - deduplicate: 同一周波数の重複を除去（adaptive優先）
// - renumber: origin.sampleIndex を 0..N-1 に再採番
//==============================================================================

/// Phase A: ソート（周波数昇順、stable）
inline std::vector<MergedSample> mergeAndSort(
    const CoarseScanResult& coarse,
    const AdaptiveScanResult& adaptive)
{
    std::vector<MergedSample> result;
    result.reserve(coarse.samples.size() + adaptive.samples.size());

    // 粗探索を先に追加
    for (const auto& s : coarse.samples)
        result.push_back(s);
    // 適応サンプリングを後に追加
    for (const auto& s : adaptive.samples)
        result.push_back(s);

    // stable_sort: 周波数昇順。同値は coarse→adaptive 順を維持
    std::stable_sort(result.begin(), result.end(),
        [](const MergedSample& a, const MergedSample& b) {
            return a.freqHz < b.freqHz;
        });

    return result;
}

/// Phase B: 同一周波数重複除去（adaptive優先、coarse破棄）
/// 「同一周波数」の判定は完全一致（freqHz のビット単位一致）で行う。
inline std::vector<MergedSample> deduplicate(const std::vector<MergedSample>& sorted)
{
    if (sorted.empty())
        return {};

    std::vector<MergedSample> result;
    result.reserve(sorted.size());
    result.push_back(sorted[0]);

    for (size_t i = 1; i < sorted.size(); ++i)
    {
        // 完全一致チェック
        if (sorted[i].freqHz == result.back().freqHz)
        {
            // 同一周波数: adaptive 優先（origin.type が Coarse なら上書き）
            // stable_sort の性質上、adaptive は常に後方にある
            if (result.back().origin.type == EQProcessor::SampleOrigin::Coarse
                && sorted[i].origin.type == EQProcessor::SampleOrigin::Adaptive)
            {
                result.back() = sorted[i]; // adaptive で上書き
            }
            // 両方 adaptive または両方 coarse の場合は最初のものを維持
        }
        else
        {
            result.push_back(sorted[i]);
        }
    }

    return result;
}

/// Phase C: origin.sampleIndex を 0..N-1 に再採番
inline void renumber(std::vector<MergedSample>& samples)
{
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i].origin.sampleIndex = static_cast<int>(i);
}
