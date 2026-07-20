#pragma once

#include "EQAnalysisTypes.h"
#include "EQProcessor.h"

//==============================================================================
// BandHelper — EQState から解析用 BandCollection を生成
//
// 責務:
// - EQState から有効バンドを収集
// - SVF→Biquad 変換
// - maxActiveQ / maxTotalQ の算出
//==============================================================================

class BandHelper {
public:
    /// EQState から有効バンドを収集し BandCollection を生成
    static BandCollection collectActiveBands(const EQProcessor& processor,
                                              const EQProcessor::EQState& state,
                                              double processingRate);
};
