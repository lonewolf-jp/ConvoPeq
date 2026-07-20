#pragma once

#include "EQAnalysisTypes.h"
#include <vector>
#include <cmath>

//==============================================================================
// EQResponseSampler — 周波数応答のサンプリング（stateless）
//
// 責務:
// - 粗探索600点（10Hz〜min(20kHz, Nyquist) 対数分布）
// - 候補Band判定（measured 用: isBoosting, upperBound 用: max|Hi-1|>0.1）
// - Shelf/LPF/HPF 追加評価
// - union区間統合 + 比例配分
// - 適応サンプリング
//==============================================================================

class EQResponseSampler {
public:
    EQResponseSampler(double processingRate, bool isParallel) noexcept
        : processingRate_(processingRate)
        , isParallel_(isParallel)
        , nyquist_(processingRate * 0.5)
        , maxFreq_(std::min(20000.0, nyquist_))
    {}

    /// 粗探索600点を実行
    CoarseScanResult runCoarse(const BandCollection& bands) const;

    /// measured 用候補Band判定（isBoosting()==true）
    std::vector<const BandInfo*> findMeasuredCandidates(const BandCollection& bands) const;

    /// upperBound 用候補Band判定（max|Hi-1| > 0.1）
    std::vector<const BandInfo*> findUpperBoundCandidates(
        const BandCollection& bands,
        const std::array<double, 20>& bandMaxDelta) const;

    /// 適応サンプリング（union統合+比例配分）
    AdaptiveScanResult runAdaptive(
        const BandCollection& bands,
        const std::vector<const BandInfo*>& measuredCands,
        const std::vector<const BandInfo*>& upperBoundCands,
        const CoarseScanResult& coarseResult) const;

    /// 1点評価
    MergedSample evaluate(double freqHz, const BandCollection& bands) const;

    // 定数
    static constexpr int kCoarsePoints = 600;
    static constexpr int kAdaptivePoints = 128;
    static constexpr double kDeltaThreshold = 0.1;

private:
    double processingRate_;
    bool isParallel_;
    double nyquist_;
    double maxFreq_;
};
