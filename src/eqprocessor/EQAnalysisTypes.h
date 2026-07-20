#pragma once

#include "EQProcessor.h"
#include <vector>
#include <array>
#include <cstdint>

//==============================================================================
// EQAnalysis 3層分割 共通型定義
//
// v14.47: computeEstimatedMaxGainComplex() のリファクタリングに伴う型抽出
// 設計書: refactor-3layer-plan-v5.md
//==============================================================================

//==============================================================================
// 1バンドの解析用情報（BandHelper が生成）
//==============================================================================
struct BandInfo {
    int index;                          // バンド番号 (0..19)
    double freq;                        // 中心周波数 [Hz]
    double q;                           // Q値
    EQBandType type;                    // フィルタ種別
    float gain;                         // ゲイン [dB]
    EQCoeffsBiquad biquad;             // 表示用 Biquad 係数
    bool isBoosting;                    // isBoostingBand(type, gain)

    /// BandType ごとに最適化された探索範囲
    std::pair<double, double> searchRange(double maxFreq) const noexcept {
        switch (type) {
            case EQBandType::Peaking:
                return { std::max(10.0, freq / 4.0), std::min(maxFreq, freq * 4.0) };
            case EQBandType::LowShelf:
                return { 10.0, std::min(maxFreq, freq * 2.0) };
            case EQBandType::HighShelf:
                return { std::max(10.0, freq / 2.0), maxFreq };
            default: // LowPass, HighPass
                return { std::max(10.0, freq / 4.0), std::min(maxFreq, freq * 4.0) };
        }
    }
};

//==============================================================================
// バンド収集結果
//==============================================================================
struct BandCollection {
    std::vector<BandInfo> bands;
    float maxActiveQ = 0.0f;    // ブーストバンド中の最大Q（Planner 使用）
    float maxTotalQ = 0.0f;     // 全有効バンド最大Q（diagnostics 専用）
};

//==============================================================================
// SampleOrigin — 評価点の origin（既存の EQProcessor::SampleOrigin を再利用）
//==============================================================================

//==============================================================================
// 統合サンプル（measured と upperBound を一元管理）
//★ linearMagnitude を保持し measuredDb を保持しない理由:
//   1. dB変換は PeakEstimator で1回のみ実行されるため、事前変換のメリットがない
//   2. linear 値を保持することで、将来の FFT 置換時に変換が不要
//   3. SampleOrigin と合わせて48byteに収まり、キャッシュライン境界に整合
//==============================================================================
struct MergedSample {
    double freqHz;
    double linearMagnitude;     // |H(freq)|（linear、dB変換は利用時）
    double upperBoundDb;        // (20/ln10) * Σln(1+|Hi-1|)
    EQProcessor::SampleOrigin origin;
};

//==============================================================================
// 粗探索結果
//==============================================================================
struct CoarseScanResult {
    std::vector<MergedSample> samples;
    std::array<double, 20> bandMaxDelta{};
    std::array<double, 20> bandMaxMagnitude{};
};

//==============================================================================
// 適応サンプリング結果
//==============================================================================
struct AdaptiveScanResult {
    std::vector<MergedSample> samples;
};

//==============================================================================
// PeakEstimator の戻り値
//==============================================================================
struct PeakEstimate {
    float interpolatedDb = 0.0f;     // 放物線補間後のゲイン [dB]
    float interpolatedFreqHz = 0.0f;
    float rawDb = 0.0f;              // 補間前の最大ゲイン [dB]
    float rawFreqHz = 0.0f;
    int rawSampleIndex = -1;         // 統合後 vector 内インデックス
};

//==============================================================================
// UpperBoundEstimator の戻り値
//==============================================================================
struct UpperBoundEstimate {
    float maxDb = 0.0f;
    float freqHz = 0.0f;
    int sampleIndex = -1;
};
