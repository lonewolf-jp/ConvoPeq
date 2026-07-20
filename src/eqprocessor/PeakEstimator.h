#pragma once

#include "EQAnalysisTypes.h"
#include <vector>

//==============================================================================
// PeakEstimator — MergedSample から measured 最大値を推定
//
// 責務:
// - 全サンプルから measured 最大値を探索
// - 最大値周辺の3点で Lagange 放物線補間（対数周波数軸+dB空間）
// - EQAnalysisResult::PeakInfo に依存しない PeakEstimate を返す
//==============================================================================

class PeakEstimator {
public:
    /// 全サンプルから measured 最大値を推定
    /// @param samples 周波数昇順ソート済み、重複除去済み
    /// @return PeakEstimate（補間後/補間前 両方を含む）
    static PeakEstimate estimate(const std::vector<MergedSample>& samples);

    /// 放物線補間（Lagrange一般3点、対数周波数軸+dB空間）
    /// @param y0,y1,y2 dB空間のゲイン値
    /// @return 補間後のピーク値。
    ///   3点不足時や分母 < 1e-12 の場合は y[1] を返す。
    ///   注意: この 1e-12 はゼロ除算防止の閾値であり、数値比較の許容誤差（1e-9）とは
    ///   目的が異なる。
    static double interpolateParabolic(double x0, double y0,
                                        double x1, double y1,
                                        double x2, double y2);

private:
    /// 大域的最大値のインデックスを探索
    static int findGlobalPeak(const std::vector<MergedSample>& samples);
};
