#pragma once

#include "EQAnalysisTypes.h"
#include <vector>

//==============================================================================
// UpperBoundEstimator — MergedSample から upperBound 最大値を選択
//
// 責務:
// - MergedSample.upperBoundDb から最大値を選択（補間なし）
// - 安全側保証のため、評価点最大値をそのまま採用
//==============================================================================

class UpperBoundEstimator {
public:
    /// MergedSample.upperBoundDb から最大値を選択（補間なし）
    static UpperBoundEstimate estimateMax(const std::vector<MergedSample>& samples);
};
