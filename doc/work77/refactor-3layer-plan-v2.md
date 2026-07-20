# EQAnalysis 3層分割 改修計画書 v2

> 作成: 2026-07-20 | 改定: 2026-07-20（レビュー反映）
> 対象設計書: AutoGainStagingRenewal.md v14.47 §4.1.7
> 現状: `computeEstimatedMaxGainComplex()` 一関数（約550行）に全ロジック集中
> 目標: 4層（BandCollector / EQResponseSampler / PeakEstimator / UpperBoundEstimator）への分割

---

## 0. レビュー指摘対応マトリクス

| # | 指摘 | 重要度 | 対応 |
|---|------|--------|------|
| ① | `SampleResult` に `upperBoundDb` を保持 → 責務混在 | **重要** | `SampleResult` から upperBound を削除。`UpperBoundEstimator` が `perBandResponse` から計算 |
| ② | `bandMaxDelta_` を Sampler が状態保持 → ISR違反 | **重要** | `runCoarse()` が `CoarseScanResult` を返す。Sampler を stateless に |
| ③ | `PeakEstimator` が `EQAnalysisResult::PeakInfo` に依存 | 重要 | 専用の `PeakEstimate` 構造体を返す。`EQAnalysisResult` への詰め替えは最終段 |
| ④ | `mergeSamples()` の仕様未定義 | **重要** | マージ手順・重複処理・ソート安定性を明文化 |
| ⑤ | `SampleResult` が dB のみ保持 → FFT置換で非効率 | 中 | `linearMagnitude` を主要値とし、dB変換は利用時に行う |
| ⑥ | `SamplingStrategy` 列挙型が冗長 | 中 | 削除。APIは `runCoarse()`, `runAdaptive()` で十分 |
| ⑦ | `getSearchRange()` を static でSamplerに配置 → BandInfoに移動すべき | 中 | `BandInfo::searchRange()` として移動 |
| ⑧ | `maxActiveQ` が最終段で宙に浮く | 低 | `BandCollection` に含める |
| — | ビット一致 vs ±1e-6の矛盾 | 中 | 「絶対誤差 1e-12」に統一。ビット一致は要求しない |

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

### 1.2 問題点

1. **単体テスト困難**: 内部ロジックをテストするには関数全体を呼び出す必要がある
2. **FFT将来置換への障害**: サンプリングロジックが密結合
3. **コード理解の難易度**: 550行の一関数は Cognitive Complexity が高い
4. **設計書未定義**: §4.1.7 は参照のみで詳細未定義

---

## 2. 目標アーキテクチャ

```
EQState
  ↓
BandCollector                  ← 有効バンド収集 + SVF→Biquad変換 + maxQ算出
  ↓
BandCollection
  ├─ bands: vector<BandInfo>
  ├─ maxActiveQ: float
  └─ maxTotalQ: float          ← 新規: 全有効バンド最大Q（diagnostics 用）
  ↓
EQResponseSampler (stateless)  ← 周波数点生成（粗探索+適応+union）
  ↓  ┌──────────────────────┐
  ├──│ CoarseScanResult     │  ← samples + bandMaxDelta + bandMaxMag
  │  └──────────────────────┘
  │  ┌──────────────────────────┐
  └──│vector<SampleResult>      │  ← freqHz + linearMagnitude + perBandDeltas[]
     │ (from adaptive sampling) │
     └──────────────────────────┘
  ↓ merge (stable_sort, 仕様明記)
AllSamples
  ↓
PeakEstimator                  ← measured 最大値 + 放物線補間
  ↓
PeakEstimate                   ← raw + interpolated + freq + index
  ↓
UpperBoundEstimator            ← Π(1+|Hi-1|) 最大値
  ↓
UpperBoundEstimate             ← value + freq + index
  ↓
EQProcessor                    ← EQAnalysisResult への詰め替え
  ↓
EQAnalysisResult
```

---

## 3. 共通型定義

```cpp
// EQAnalysisTypes.h: 3層で共有する型定義

/// 1バンドの情報（BandCollector が生成）
struct BandInfo {
    int index;                  // バンド番号 (0..19)
    double freq;                // 中心周波数 [Hz]
    double q;                   // Q値
    EQBandType type;            // フィルタ種別
    float gain;                 // ゲイン [dB]
    EQCoeffsBiquad biquad;      // 表示用 Biquad 係数（SVF→Biquad変換済み）
    bool isBoosting;            // isBoostingBand(type, gain)

    /// 適応サンプリングの探索範囲を返す
    /// BandType ごとに最適化された範囲（Peak: center±2oct, Shelf: DC/Nyquist側拡張, etc.）
    std::pair<double, double> searchRange(double maxFreq) const noexcept;
};

/// バンド収集結果
struct BandCollection {
    std::vector<BandInfo> bands;
    float maxActiveQ = 0.0f;    // ブーストバンド中の最大Q（Planner 使用）
    float maxTotalQ = 0.0f;     // 全有効バンド最大Q（diagnostics 専用）
};

/// 1評価点の結果（サンプリング結果 のみ。upperBound は含まない）
struct SampleResult {
    double freqHz;
    double linearMagnitude;     // |H(freq)| （将来FFT置換でも再利用可能）
    std::vector<double> perBandDeltas;  // |Hi-1| の配列（UpperBoundEstimator が使用）
    SampleOrigin origin;        // 評価点の origin
};

/// 粗探索結果（EQResponseSampler::runCoarse の戻り値）
struct CoarseScanResult {
    std::vector<SampleResult> samples;
    std::vector<double> bandMaxDelta;       // 各バンドの最大 |Hi-1|
    std::vector<double> bandMaxMagnitude;   // 各バンドの最大 |Hi|
};

/// PeakEstimator の戻り値（EQAnalysisResult 非依存）
struct PeakEstimate {
    float interpolatedDb = 0.0f;    // 放物線補間後のゲイン [dB]
    float interpolatedFreqHz = 0.0f;
    float rawDb = 0.0f;             // 補間前の最大ゲイン [dB]
    float rawFreqHz = 0.0f;
    int rawSampleIndex = -1;        // 最大点の全サンプル内インデックス
};

/// UpperBoundEstimator の戻り値
struct UpperBoundEstimate {
    float maxDb = 0.0f;             // max (20/ln10) * Σln(1+|Hi-1|) [dB]
    float freqHz = 0.0f;
    int sampleIndex = -1;
};
```

---

## 4. 各層の詳細設計

### 4.1 第1層: BandCollector（新規）

```cpp
class BandCollector {
public:
    /// EQState から有効バンドを収集し、Biquad係数・isBoostingを事前計算
    static BandCollection collect(const EQState& state, double processingRate);
};
```

- `calcSVFCoeffs()` / `svfToDisplayBiquad()` は `EQProcessor` のメンバ関数のまま -> `BandCollector` が `EQProcessor` から呼ばれる形
- `maxActiveQ`: ブーストバンド (`isBoosting()==true`) 中の最大Q
- `maxTotalQ`: 全有効バンド中の最大Q（diagnostics 専用。設計書 v14.45 で規定）
- ファイル: `src/eqprocessor/EQAnalysisTypes.h` + `src/eqprocessor/BandCollector.h/.cpp`

### 4.2 第2層: EQResponseSampler（stateless）

```cpp
class EQResponseSampler {
public:
    EQResponseSampler(double processingRate, bool isParallel);

    /// 粗探索600点を実行（stateless: 引数の BandCollection のみから計算）
    /// @return CoarseScanResult（samples + bandMaxDelta + bandMaxMagnitude）
    CoarseScanResult runCoarse(const BandCollection& bands) const;

    /// measured 用候補Bandを判定（isBoosting）
    std::vector<const BandInfo*> findMeasuredCandidates(
        const BandCollection& bands) const;

    /// upperBound 用候補Bandを判定（bandMaxDelta > 0.1）
    std::vector<const BandInfo*> findUpperBoundCandidates(
        const BandCollection& bands,
        const std::vector<double>& bandMaxDelta) const;

    /// 適応サンプリング（union統合+比例配分）
    std::vector<SampleResult> runAdaptive(
        const BandCollection& bands,
        const std::vector<const BandInfo*>& measuredCands,
        const std::vector<const BandInfo*>& upperBoundCands,
        const CoarseScanResult& coarseResult) const;

    /// 1点評価（内部共有）
    SampleResult evaluate(double freqHz, const BandCollection& bands) const;

private:
    double processingRate_;
    bool isParallel_;
    double nyquist_;
    double maxFreq_;
};
```

**重要な設計変更（レビュー①・②対応）**:
- `SampleResult` は `upperBoundDb` を**保持しない**
- `upperBound` の計算は `UpperBoundEstimator` が `perBandDeltas[]` から行う
- `bandMaxDelta_` / `bandMaxMagnitude_` は内部状態ではなく `CoarseScanResult` の一部
- これにより `EQResponseSampler` は完全 stateless に → **複数スレッドから安全**

**ファイル**: `src/eqprocessor/EQResponseSampler.h/.cpp`

### 4.3 第3層: PeakEstimator

```cpp
class PeakEstimator {
public:
    /// 全サンプルから measured 最大値を推定
    /// @param samples 周波数昇順ソート済み全サンプル
    /// @return PeakEstimate（補間後/補間前 両方を含む）
    static PeakEstimate estimate(const std::vector<SampleResult>& samples);

    /// 放物線補間（Lagrange一般3点、対数周波数軸+dB空間）
    /// @return 補間後のピーク値。3点不足や発散時は y[1] を返す
    static double interpolateParabolic(double x0, double y0,
                                        double x1, double y1,
                                        double x2, double y2);
private:
    static int findGlobalPeak(const std::vector<SampleResult>& samples);
};
```

**重要な設計変更（レビュー③対応）**:
- `estimate()` は `EQAnalysisResult::PeakInfo` ではなく `PeakEstimate` を返す
- `EQAnalysisResult` への依存を完全排除
- `PeakEstimate` は生のゲイン・周波数・インデックスのみ

**ファイル**: `src/eqprocessor/PeakEstimator.h/.cpp`

### 4.4 第4層: UpperBoundEstimator

```cpp
class UpperBoundEstimator {
public:
    /// 全サンプルから upperBound 最大値を選択（補間なし）
    /// @param samples 周波数昇順ソート済み全サンプル
    /// @return UpperBoundEstimate
    static UpperBoundEstimate estimateMax(
        const std::vector<SampleResult>& samples);

    /// Σln(1+|Hi-1|) を perBandDeltas から計算
    /// @param deltas perBandDeltas[] 配列
    /// @return (20/ln10) * Σln(1+|Hi-1|) [dB]
    static double computeLogBound(const std::vector<double>& deltas);
};
```

**重要な設計変更（レビュー①対応）**:
- `computeLogBound()` が `perBandDeltas[]` から upperBound を計算
- `SampleResult` に `perBandDeltas` がある → いつでも計算可能
- FFTベース探索に置き換えた場合も、`SampleResult::perBandDeltas` さえ正しく埋めれば UpperBoundEstimator は変更不要

**ファイル**: `src/eqprocessor/UpperBoundEstimator.h/.cpp`

---

## 5. mergeSamples() 仕様（レビュー④対応）

```cpp
/// 全サンプルを周波数昇順にソートして統合
///
/// 仕様:
/// 1. coarse の全サンプルを先に追加
/// 2. adaptive の全サンプルを後に追加
/// 3. std::stable_sort（周波数昇順、同値は coarse→adaptive の順を維持）
/// 4. 同一周波数が coarse と adaptive の両方にある場合は両方保持する
///    （重複除去は行わない。放物線補間は隣接3点を使用するため、
///     同一周波数があっても補間式の安定性に影響しない）
/// 5. ソート後も origin.type / origin.sampleIndex は元の情報を保持
///
/// @param coarse   runCoarse() の結果
/// @param adaptive runAdaptive() の結果
/// @return ソート済み全サンプル
static std::vector<SampleResult> mergeSamples(
    const CoarseScanResult& coarse,
    const std::vector<SampleResult>& adaptive);
```

---

## 6. ファイル構成

| ファイル | 種別 | 内容 |
|---------|------|------|
| `src/eqprocessor/EQAnalysisTypes.h` | **新規** | `BandInfo`, `BandCollection`, `SampleResult`, `CoarseScanResult`, `PeakEstimate`, `UpperBoundEstimate` |
| `src/eqprocessor/BandCollector.h` | **新規** | `BandCollector` クラス宣言 |
| `src/eqprocessor/BandCollector.cpp` | **新規** | `collect()` 実装（EQProcessor の SVF計算関数を呼ぶ） |
| `src/eqprocessor/EQResponseSampler.h` | **新規** | `EQResponseSampler` クラス宣言 |
| `src/eqprocessor/EQResponseSampler.cpp` | **新規** | サンプリング実装（stateless） |
| `src/eqprocessor/PeakEstimator.h` | **新規** | `PeakEstimator` クラス宣言 |
| `src/eqprocessor/PeakEstimator.cpp` | **新規** | 放物線補間 + ピーク検出 |
| `src/eqprocessor/UpperBoundEstimator.h` | **新規** | `UpperBoundEstimator` クラス宣言 |
| `src/eqprocessor/UpperBoundEstimator.cpp` | **新規** | upperBound 計算 |
| `src/eqprocessor/EQProcessor.Coefficients.cpp` | **変更** | `computeEstimatedMaxGainComplex()` を4層呼び出しに（約50行） |
| `CMakeLists.txt` | **変更** | 新規9ファイルをソース一覧に追加 |

---

## 7. 結果整合性検証

### 7.1 検証方法

| Phase | 検証方法 | 許容誤差 |
|-------|---------|---------|
| Phase 2-4 (個別抽出) | 元の関数内で新クラスを呼び出し、結果を比較 | 絶対誤差 1e-12 |
| Phase 5 (統合) | 全2322テスト実行 | 絶対誤差 1e-12 |
| 最終確認 | Debug/Release 両ビルド + テスト | 全テスト PASS |

### 7.2 誤差発生要因と対策

| 要因 | 対策 |
|------|------|
| merge 順序変更による `std::max` 選択順の変化 | `mergeSamples()` の stable_sort 保証 |
| FMA / ベクトル化の有無 | 許容誤差 1e-12 で吸収（ビット一致は要求しない） |
| `vector::reserve` の有無 | 結果に影響しない |
| `sqrt`/`log`/`exp` の実装差 | 同一コンパイラでは同一実装のため問題なし |

---

## 8. 移行手順

### Phase 1: 共通型の分離 + BandCollector（1.5h）

1. `EQAnalysisTypes.h` 作成（全共通型定義）
2. `BandCollector.h/.cpp` 作成（`collect()` は既存のバンド収集ループを移動）
3. `EQProcessor.Coefficients.cpp` が `#include "EQAnalysisTypes.h"` に変更
4. **検証**: コンパイル + 2322テスト通過

### Phase 2: EQResponseSampler 抽出（4h）

1. `EQResponseSampler.h/.cpp` 作成
2. 粗探索ループを `runCoarse()` に移動（戻り値: `CoarseScanResult`）
3. 候補Band判定を `findMeasuredCandidates()` / `findUpperBoundCandidates()` に移動
4. union統合 + 適応サンプリングを `runAdaptive()` に移動
5. `evaluate()` に1点評価ロジックを抽出
6. `BandInfo::searchRange()` に範囲決定ロジックを移動
7. **検証**: 新クラスを元関数内で呼び出し、結果比較

### Phase 3: PeakEstimator 抽出（3h）

1. `PeakEstimator.h/.cpp` 作成
2. 放物線補間ロジックを `interpolateParabolic()` に移動
3. `estimate()` で measured 最大値 + 補間を実行
4. **検証**: 結果比較
5. **エッジケース追加**:
   - 3点不足（先頭/末尾）→ 補間なし
   - 分母 1e-12 未満 → 補間スキップ
   - 全点同一値 → そのまま返す
   - 単一サンプルのみ → rawDb = interpolatedDb

### Phase 4: UpperBoundEstimator 抽出（1h）

1. `UpperBoundEstimator.h/.cpp` 作成
2. `estimateMax()` で upperBound 最大値を選択
3. `computeLogBound()` で `perBandDeltas[]` から upperBound 計算
4. **検証**: 結果比較

### Phase 5: 統合（3h）

1. `computeEstimatedMaxGainComplex()` を4層呼び出しに書き換え（約50行）
2. 既存 `verifyBuildBundle()` は変更不要（入出力が同一のため）
3. **検証**: 2322テスト + 実機測定結果の比較

### Phase 6: 単体テスト追加（4h）

| テスト | テストケース数 | 内容 |
|--------|--------------|------|
| `PeakEstimator.interpolateParabolic` | 10 | 通常3点/等間隔/境界/分母微小/全同一 |
| `PeakEstimator.estimate` | 6 | 単峰/多峰/平坦/1点/2点/dB→linear変換 |
| `UpperBoundEstimator.computeLogBound` | 5 | normal/all zero/NaN/Inf/最大値 |
| `UpperBoundEstimator.estimateMax` | 4 | 単峰/複数候補/全0/単一サンプル |
| `EQResponseSampler.evaluate` | 6 | Parallel/Serial/単一Band/全pass/DC/Nyquist |
| `EQResponseSampler.findMeasuredCandidates` | 5 | Boostingのみ/非Boosting混在/全Boost/全非Boost/LPF混在 |
| `EQResponseSampler.findUpperBoundCandidates` | 5 | delta>0.1/≤0.1/混在/Shelf/HPF+LPF |
| `EQResponseSampler.runCoarse` | 3 | 正常動作/空Band/片方のみ |
| `mergeSamples` | 4 | 基本/同一周波数/片方のみ/空 |

---

## 9. スケジュール（修正版）

| Phase | 内容 | 工数 |
|-------|------|------|
| Phase 1 | 共通型 + BandCollector | 1.5h |
| Phase 2 | EQResponseSampler | 4h |
| Phase 3 | PeakEstimator | 3h |
| Phase 4 | UpperBoundEstimator | 1h |
| Phase 5 | 統合 + 既存テスト検証 | 3h |
| Phase 6 | 新規テスト追加 | 4h |
| デバッグ | 予備 | 2h |
| **合計** | | **18.5h** |

---

## 10. リスクと対策

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| 浮動小数点誤差による結果不一致 | 中 | 高 | 許容誤差 1e-12 を設定。Phase 2-4 で段階的に検証 |
| `searchRange()` ロジックの再現ミス | 低 | 高 | `BandInfo::searchRange()` に移動後、新旧比較テスト |
| パフォーマンス劣化（関数呼び出し増） | 低 | 中 | 全 virtual 排除、constexpr 活用、inline 適宜 |
| CMakeLists.txt 編集ミス | 低 | 中 | インクリメンタルビルドで早期検出 |
| `perBandDeltas` のメモリ使用量増加 | 中 | 低 | 600点×20band×8byte = 96KB → 問題なし |
| マージ順序変更による結果変動 | 低 | 中 | stable_sort で保証。テストで確認 |

---

## 11. 参考: 最終コードイメージ

```cpp
EQAnalysisResult EQProcessor::computeEstimatedMaxGainComplex(
    const EQState& state, double processingRate) const
{
    if (processingRate <= 0.0) return {};

    // Phase 1: Band 収集
    const auto bands = BandCollector::collect(state, processingRate);
    if (bands.bands.empty()) return {};
    const bool isParallel = (state.filterStructure == 1);

    // Phase 2: サンプリング
    const EQResponseSampler sampler(processingRate, isParallel);
    const auto coarseResult = sampler.runCoarse(bands);
    const auto measCands = sampler.findMeasuredCandidates(bands);
    const auto ubCands = sampler.findUpperBoundCandidates(bands, coarseResult.bandMaxDelta);
    const auto adaptiveSamples = sampler.runAdaptive(bands, measCands, ubCands, coarseResult);

    // Phase 3: merge + 推定
    const auto allSamples = mergeSamples(coarseResult, adaptiveSamples);
    const auto measured = PeakEstimator::estimate(allSamples);
    const auto upperBound = UpperBoundEstimator::estimateMax(allSamples);

    // Phase 4: 結果生成
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

---

## 12. レビュー指摘クローズ

| # | 指摘 | 対応 | 状態 |
|---|------|------|------|
| ① | `SampleResult` に `upperBoundDb` 保持 | `SampleResult` から削除。代わりに `perBandDeltas[]` を保持。`UpperBoundEstimator::computeLogBound()` が計算。 | ✅ |
| ② | `bandMaxDelta_` を Sampler が状態保持 | `runCoarse()` が `CoarseScanResult` を返す。Sampler は完全 stateless。 | ✅ |
| ③ | `PeakEstimator` が `EQAnalysisResult::PeakInfo` 依存 | `PeakEstimate` 構造体を新規定義。依存排除。 | ✅ |
| ④ | `mergeSamples()` の仕様未定義 | 5条項の仕様を明文化（stable_sort, 重複保持, 等）。 | ✅ |
| ⑤ | `SampleResult` が dB のみ保持 | `linearMagnitude` を主要値に変更。dB変換は利用時。 | ✅ |
| ⑥ | `SamplingStrategy` 列挙型 | 削除。APIは `runCoarse()`, `runAdaptive()` のみ。 | ✅ |
| ⑦ | `getSearchRange()` | `BandInfo::searchRange()` として移動。 | ✅ |
| ⑧ | `maxActiveQ` が宙に浮く | `BandCollection` に統合。`maxTotalQ` も追加。 | ✅ |
| — | ビット一致 vs ±1e-6 矛盾 | 「絶対誤差 1e-12」に統一。ビット一致は要求しない。 | ✅ |
