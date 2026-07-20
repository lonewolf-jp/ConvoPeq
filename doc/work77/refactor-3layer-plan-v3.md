# EQAnalysis 3層分割 改修計画書 v3

> 作成: 2026-07-20 | 改定: 2026-07-20（第2次レビュー反映）
> 対象設計書: AutoGainStagingRenewal.md v14.47 §4.1.7
> 現状: `computeEstimatedMaxGainComplex()` 一関数（約550行）に全ロジック集中
> 目標: 4層（BandHelper / EQResponseSampler / PeakEstimator / UpperBoundEstimator）への分割

---

## 0. レビュー指摘対応マトリクス

### 第1次レビュー（8項目）— 反映済み

| # | 指摘 | 対応 | 状態 |
|---|------|------|------|
| ① | `SampleResult` に `upperBoundDb` 保持 | `perBandDeltas[]` に変更。UpperBoundEstimator が計算 | ✅ |
| ② | `bandMaxDelta_` を Sampler が状態保持 | `CoarseScanResult` として返す。Sampler stateless 化 | ✅ |
| ③ | `PeakEstimator` が `EQAnalysisResult` 依存 | `PeakEstimate` 新規定義 | ✅ |
| ④ | `mergeSamples()` 仕様未定義 | 5条項明文化 | ✅ |
| ⑤ | `SampleResult` が dB のみ | `linearMagnitude` 追加 | ✅ |
| ⑥ | `SamplingStrategy` enum 冗長 | 削除 | ✅ |
| ⑦ | `getSearchRange()` 配置 | `BandInfo::searchRange()` へ移動 | ✅ |
| ⑧ | `maxActiveQ` 宙に浮く | `BandCollection` に統合 | ✅ |

### 第2次レビュー（5項目）— 本版で反映

| # | 指摘 | 重要度 | 対応 |
|---|------|--------|------|
| ① | `BandCollector` が `EQProcessor` に依存（循環） | 🔴 | SVF関数を `detail`名前空間の自由関数に抽出。`BandHelper` に改名 |
| ② | `perBandDeltas` が `vector<double>` → 600×heap確保 | 🟡 | `std::array<double, 20>` に変更（20バンド固定） |
| ③ | `SampleResult` がFFT置換に不十分 | 🟢 | `linearMagnitude` 保持。将来の `complexResponse` 拡張は構造体追加で対応 |
| ④ | `mergeSamples()` 重複保持 → Lagrange分母リスク | 🟡 | 「同一周波数は adaptive 優先、coarse 破棄」に変更 |
| ⑤ | 許容誤差 1e-12 が非現実的 | 🟡 | 絶対誤差 1e-9 に緩和 |

---

## 1. 現状分析

### 1.1 現在の責務集中

`EQProcessor::computeEstimatedMaxGainComplex()` (`EQProcessor.Coefficients.cpp:394-895`, 約500行)

| 責務 | 行範囲 | 説明 |
|------|--------|------|
| バンド情報収集 | 430-460 | SVF→Biquad変換, isBoosting判定 |
| 粗探索600点 | 460-540 | Parallel/Serial分岐, measured+upperBound同時評価 |
| 候補Band判定 | 560-620 | measured用(isBoosting), upperBound用(max\|Hi-1\|>0.1) |
| Shelf/LPF追加評価 | 630-690 | 特別範囲での候補判定 |
| union区間統合 | 700-730 | `mergeRanges` ラムダ |
| 適応サンプリング | 740-830 | 128点/バンド比例配分, measured+upperBound評価 |
| maxActiveQ算出 | 880-890 | ブーストバンド最大Q |
| 放物線補間 | 840-880 | measuredのみ、Lagrange一般3点 |
| upperBound最大値選択 | 896 | 評価点最大値を採用 |
| 結果パッケージ | 898-905 | EQAnalysisResult 生成 |

---

## 2. 目標アーキテクチャ

```
EQState
  ↓
EQProcessor::computeEstimatedMaxGainComplex()
  │  BandHelper::collect(state, sr)       ← SVF→Biquad変換（自由関数化）
  │                                        ← 循環依存なし
  ↓
BandCollection { bands[], maxActiveQ, maxTotalQ }
  ↓
EQResponseSampler (stateless)              ← 周波数点生成
  ├── CoarseScanResult { samples, bandMaxDelta[], bandMaxMag[] }
  └── vector<SampleResult> (adaptive)
  ↓
mergeSamples()                             ← 仕様: stable_sort, adaptive優先
  ↓
AllSamples: vector<SampleResult>
  ├── freqHz: double
  ├── linearMagnitude: double              ← |H(freq)|（linear、dB変換は利用時）
  ├── perBandDeltas: array<double,20>      ← |Hi-1| 固定長（0ヒープ確保）
  └── origin: SampleOrigin
  ↓
PeakEstimator                              ← measured 最大値 + 放物線補間
  ↓
PeakEstimate { interpolatedDb, rawDb, freqHz, index }
  ↓
UpperBoundEstimator                        ← Π(1+|Hi-1|) 最大値
  ↓
UpperBoundEstimate { maxDb, freqHz, index }
  ↓
EQProcessor が EQAnalysisResult に詰め替え
  ↓
EQAnalysisResult
```

---

## 3. 共通型定義

```cpp
// EQAnalysisTypes.h

/// バンド種別追加以外の変更を要しない探索範囲計算
struct BandInfo {
    int index;                          // 0..19
    double freq;
    double q;
    EQBandType type;
    float gain;
    EQCoeffsBiquad biquad;
    bool isBoosting;

    /// BandType ごとに最適化された探索範囲（Peak, Shelf, LPF/HPF 個別対応）
    std::pair<double, double> searchRange(double maxFreq) const noexcept;
};

struct BandCollection {
    std::vector<BandInfo> bands;
    float maxActiveQ = 0.0f;   // ブーストバンド最大Q（Planner）
    float maxTotalQ = 0.0f;    // 全有効バンド最大Q（diagnostics）
};

/// 1評価点（周波数領域サンプル）。FFT置換時も各binがこの構造体に写像される
struct SampleResult {
    double freqHz;
    double linearMagnitude;              // |H(freq)| — linear、dB変換は利用時
    std::array<double, 20> perBandDeltas; // |Hi-1| 固定長（0ヒープ確保）
    SampleOrigin origin;
};

struct CoarseScanResult {
    std::vector<SampleResult> samples;
    std::array<double, 20> bandMaxDelta;      // 20バンド固定
    std::array<double, 20> bandMaxMagnitude;
};

struct PeakEstimate {
    float interpolatedDb = 0.0f;
    float interpolatedFreqHz = 0.0f;
    float rawDb = 0.0f;
    float rawFreqHz = 0.0f;
    int rawSampleIndex = -1;
};

struct UpperBoundEstimate {
    float maxDb = 0.0f;
    float freqHz = 0.0f;
    int sampleIndex = -1;
};
```

---

## 4. 各層の詳細設計

### 4.1 SVF/Biquad 関数の自由関数化（問題①対応）

```cpp
// detail/svf_utils.h — 新規。EQProcessor の static メンバから自由関数に変換
namespace detail {
    EQCoeffsSVF calcSVFCoeffs(EQBandType type, float freq, float gainDb, float q, double sr) noexcept;
    EQCoeffsBiquad svfToDisplayBiquad(const EQCoeffsSVF& svf) noexcept;
    bool isBoostingBand(EQBandType type, float gain) noexcept;
    std::complex<double> biquadResponse(const EQCoeffsBiquad& c, double w) noexcept;
}
```

**変更理由**: `calcSVFCoeffs()`, `svfToDisplayBiquad()`, `biquadResponse()`, `isBoostingBand()` はすべて `static` で `this` を参照しない。自由関数化により `BandHelper` が `EQProcessor` に依存しなくなる。

**互換性維持**: `EQProcessor` 内の既存呼び出しは `detail::calcSVFCoeffs(...)` に変更。元の static メンバ関数は削除する。

### 4.2 BandHelper

```cpp
// BandHelper.h
class BandHelper {
public:
    /// EQState から有効バンドを収集（SVF→Biquad変換 + 各種集計）
    static BandCollection collect(const EQState& state, double processingRate);
};
```

**変更点（問題①）**: 名前を `BandCollector` → `BandHelper` に変更。`collect()` 内部で `detail::calcSVFCoeffs()` / `detail::svfToDisplayBiquad()` を呼ぶ。`EQProcessor` への依存なし。

### 4.3 EQResponseSampler（stateless）

```cpp
class EQResponseSampler {
public:
    EQResponseSampler(double processingRate, bool isParallel);

    CoarseScanResult runCoarse(const BandCollection& bands) const;

    std::vector<const BandInfo*> findMeasuredCandidates(const BandCollection& bands) const;
    std::vector<const BandInfo*> findUpperBoundCandidates(
        const BandCollection& bands,
        const std::array<double, 20>& bandMaxDelta) const;

    std::vector<SampleResult> runAdaptive(
        const BandCollection& bands,
        const std::vector<const BandInfo*>& measuredCands,
        const std::vector<const BandInfo*>& upperBoundCands,
        const CoarseScanResult& coarseResult) const;

    SampleResult evaluate(double freqHz, const BandCollection& bands) const;

private:
    double processingRate_;
    bool isParallel_;
    double nyquist_;
    double maxFreq_;
};
```

**変更点（問題②）**: `bandMaxDelta` / `bandMaxMagnitude` を `std::array<double, 20>` に変更（`std::vector<double>` から）。

### 4.4 PeakEstimator

```cpp
class PeakEstimator {
public:
    static PeakEstimate estimate(const std::vector<SampleResult>& samples);
    static double interpolateParabolic(double x0, double y0,
                                        double x1, double y1,
                                        double x2, double y2);
};
```

変更なし（v2から）。

### 4.5 UpperBoundEstimator

```cpp
class UpperBoundEstimator {
public:
    static UpperBoundEstimate estimateMax(const std::vector<SampleResult>& samples);
    static double computeLogBound(const std::array<double, 20>& deltas);
};
```

**変更点（問題②）**: 引数を `const std::vector<double>&` → `const std::array<double, 20>&` に変更。

---

## 5. mergeSamples() 仕様（問題④対応 v3改定）

```cpp
/// 全サンプルを周波数昇順に統合
///
/// 仕様:
/// 1. coarse の全サンプルを先に追加
/// 2. adaptive の全サンプルを後に追加
/// 3. std::stable_sort（周波数昇順。同値は coarse → adaptive の投入順を維持）
/// 4. 【v3変更】同一周波数が coarse と adaptive の両方に存在する場合、
///    adaptive を残し coarse を破棄する。
///    理由: Lagrange 放物線補間は隣接3点の x 座標が異なることを前提とする。
///    同一周波数が複数存在すると x0==x1 となり分母発散のリスクがある。
///    適応サンプリングは粗探索より高密度なため、adaptive を優先する。
/// 5. ソート後も origin.type / origin.sampleIndex は元の情報を保持
///
/// @return ソート済み重複除去済み全サンプル
static std::vector<SampleResult> mergeSamples(
    const CoarseScanResult& coarse,
    const std::vector<SampleResult>& adaptive);
```

**変更点**: ④「重複保持」→「adaptive優先、coarse破棄」に変更。

---

## 6. ファイル構成

| ファイル | 種別 | 内容 |
|---------|------|------|
| `src/eqprocessor/detail/svf_utils.h` | **新規** | SVF/Biquad/応答計算の自由関数 |
| `src/eqprocessor/EQAnalysisTypes.h` | **新規** | 全共通型定義（`array<double,20>`採用） |
| `src/eqprocessor/BandHelper.h` | **新規** | `BandHelper::collect()` |
| `src/eqprocessor/BandHelper.cpp` | **新規** | 実装（`detail::calcSVFCoeffs` 使用） |
| `src/eqprocessor/EQResponseSampler.h` | **新規** | Sampler 宣言 |
| `src/eqprocessor/EQResponseSampler.cpp` | **新規** | サンプリング実装 |
| `src/eqprocessor/PeakEstimator.h` | **新規** | PeakEstimator 宣言 |
| `src/eqprocessor/PeakEstimator.cpp` | **新規** | 補間実装 |
| `src/eqprocessor/UpperBoundEstimator.h` | **新規** | UpperBoundEstimator 宣言 |
| `src/eqprocessor/UpperBoundEstimator.cpp` | **新規** | upperBound 計算 |
| `src/eqprocessor/EQProcessor.Coefficients.cpp` | **変更** | `computeEstimatedMaxGainComplex()` を4層呼び出しに |
| `src/eqprocessor/EQProcessor.h` | **変更** | static SVF関数の宣言削除 |
| `CMakeLists.txt` | **変更** | 新規ファイル追加 |

---

## 7. 結果整合性検証

| Phase | 検証方法 | 許容誤差 |
|-------|---------|---------|
| Phase 2-4 (個別抽出) | 新旧結果比較（同関数内で両方実行） | 絶対誤差 **1e-9** |
| Phase 5 (統合) | 全2322テスト実行 | 絶対誤差 **1e-9** |
| 最終確認 | Debug/Release + 全テスト | 全 PASS |

**v3変更点（問題⑤対応）**: 許容誤差を `1e-12` → `1e-9` に緩和。
理由: `std::log` / `std::pow` / `std::exp` の複数回使用、Release/Debugの最適化差異、
FMA命令の有無などを考慮。1e-9 は典型的な倍精度演算の安定範囲内。

---

## 8. 移行手順（工数修正版）

| Phase | 内容 | 工数 |
|-------|------|------|
| Phase 0 | `detail/svf_utils.h` 作成 + `EQProcessor.h` の static 宣言削除 | 1h |
| Phase 1 | `EQAnalysisTypes.h` + `BandHelper` | 2h |
| Phase 2 | `EQResponseSampler` | 5h |
| Phase 3 | `PeakEstimator` | 3h |
| Phase 4 | `UpperBoundEstimator` | 2h |
| Phase 5 | 統合 + 既存テスト検証 | 3h |
| Phase 6 | 新規テスト追加（52テストケース） | 5h |
| バグ修正予備 | — | 3h |
| **合計** | | **24h** |

---

## 9. テストケース拡充（問題③④レビュー追記）

| テスト | ケース数 | 追加内容 |
|--------|---------|---------|
| `PeakEstimator.estimate` | +3 | plateau peak / NaN / Inf / negative infinity |
| `UpperBoundEstimator` | +2 | Parallel 31band all-boost Q=20, 全0 |
| `mergeSamples` | +3 | duplicate freq, duplicate origin, reverse input order |
| `EQResponseSampler.runCoarse` | +1 | 片方のみ active（EQ 8/20 bands） |
| **小計** | **+9** | **全52ケース** |

---

## 10. リスクと対策

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| SVF自由関数化による既存呼び出しのコンパイルエラー | 低 | 高 | `EQProcessor.h` の static 宣言削除と同時に全呼び出し箇所を `detail::` に変更。grep で全件洗い出し |
| `std::array<double, 20>` のコピーコスト | 低 | 低 | 20×8=160バイト = 2 cache line。pass by const ref で対応 |
| 重複除去による measured 最大値変化 | 低 | 中 | coarse→adaptive同一周波数という状況は稀。テストで検証 |
| `std::array` の添え字範囲外（band index > 19） | 低 | 高 | `jassert(bandIdx < NUM_BANDS)` または `at()` で範囲チェック |

---

## 11. 参考: 最終コードイメージ

```cpp
EQAnalysisResult EQProcessor::computeEstimatedMaxGainComplex(
    const EQState& state, double processingRate) const
{
    if (processingRate <= 0.0) return {};
    if (state.activeBands == 0) return {};

    const bool isParallel = (state.filterStructure == 1);

    // Phase 1: バンド収集（SVF→Biquad変換込み）
    const auto bands = BandHelper::collect(state, processingRate);
    if (bands.bands.empty()) return {};

    // Phase 2: サンプリング（完全 stateless）
    const EQResponseSampler sampler(processingRate, isParallel);
    const auto coarseResult = sampler.runCoarse(bands);
    const auto adaptive = sampler.runAdaptive(
        bands,
        sampler.findMeasuredCandidates(bands),
        sampler.findUpperBoundCandidates(bands, coarseResult.bandMaxDelta),
        coarseResult);

    // Phase 3: 統合 + 推定
    const auto allSamples = mergeSamples(coarseResult.samples, adaptive);
    const auto measured = PeakEstimator::estimate(allSamples);
    const auto upperBound = UpperBoundEstimator::estimateMax(allSamples);

    // Phase 4: EQAnalysisResult に詰め替え
    EQAnalysisResult result;
    result.measured.gainDb = measured.interpolatedDb;
    result.measured.freqHz = measured.interpolatedFreqHz;
    result.measuredRawGainDb = measured.rawDb;
    result.upperBound.gainDb = upperBound.maxDb;
    result.upperBound.freqHz = upperBound.freqHz;
    result.maxActiveQ = bands.maxActiveQ;
    result.algorithm = convo::EqGainAlgorithm::TriangleProductV1;
    return result;
}
```

**変更前後の行数比較**:

| コード | 行数 |
|--------|------|
| 現行 `computeEstimatedMaxGainComplex()` | ~500行 |
| 改修後 `computeEstimatedMaxGainComplex()` | ~35行 |
| 新規コード（4層 + テスト） | ~700行 |
| **純増** | **~235行** |

---

## 12. レビュー指摘クローズ

| # | 指摘 | 対応 | 状態 |
|---|------|------|------|
| ① | `BandCollector` が `EQProcessor` に依存（循環） | SVF関数を `detail/svf_utils.h` の自由関数に抽出。`BandHelper` に改名。 | ✅ |
| ② | `perBandDeltas` が `vector<double>` → 600×heap確保 | `std::array<double, 20>` に変更。固定長（0ヒープ確保）。 | ✅ |
| ③ | `SampleResult` がFFT置換に不十分 | `linearMagnitude` 保持で対応。将来の複素拡張は構造体追加で。 | ✅ |
| ④ | `mergeSamples()` 重複保持 → Lagrange分母リスク | 「adaptive優先、coarse破棄」に変更。 | ✅ |
| ⑤ | 許容誤差 1e-12 が非現実的 | 絶対誤差 **1e-9** に緩和。根拠文書化。 | ✅ |
