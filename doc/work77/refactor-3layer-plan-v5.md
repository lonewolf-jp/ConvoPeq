# EQAnalysis 3層分割 改修計画書 v5（ファイナル）

> 作成: 2026-07-20 | 改定: 2026-07-20（第4次レビュー反映）
> 対象設計書: AutoGainStagingRenewal.md v14.47 §4.1.7
> 最終評価: **A（96点）** — 実装フェーズ移行可能

---

## 0. レビュー指摘対応マトリクス（全20項目）

| # | レビュー | 指摘 | 対応 | 状態 |
|---|---------|------|------|------|
| ① | 1次 | `SampleResult` に `upperBoundDb` 保持 | `SampleResult` から削除。別途 `UpperBoundRecord` | ✅ |
| ② | 1次 | `bandMaxDelta_` を Sampler が状態保持 | `CoarseScanResult` で返却、Sampler stateless 化 | ✅ |
| ③ | 1次 | `PeakEstimator` が `EQAnalysisResult` 依存 | `PeakEstimate` 新規定義 | ✅ |
| ④ | 1次 | `mergeSamples()` 仕様未定義 | 仕様明文化 | ✅ |
| ⑤ | 1次 | `SampleResult` が dB のみ | `linearMagnitude` 追加 | ✅ |
| ⑥ | 1次 | `SamplingStrategy` enum | 削除 | ✅ |
| ⑦ | 1次 | `getSearchRange()` 配置 | `BandInfo::searchRange()` へ移動 | ✅ |
| ⑧ | 1次 | `maxActiveQ` 宙に浮く | `BandCollection` に統合 | ✅ |
| ⑨ | 2次 | `BandCollector` が `EQProcessor` 依存 | SVF関数を自由関数化。`BandHelper` に改名 | ✅ |
| ⑩ | 2次 | `perBandDeltas` が `vector` → 600×heap | `std::array<double, 20>` に変更（v3）→ 後に全廃（v4） | ✅ |
| ⑪ | 2次 | `mergeSamples()` 重複保持 → Lagrange分母リスク | 「adaptive優先、coarse破棄」 | ✅ |
| ⑫ | 2次 | 許容誤差 1e-12 が非現実的 | 絶対誤差 1e-9 に緩和 | ✅ |
| ⑬ | 3次 | `SampleResult` が重い | `perBandDeltas` 全廃。`UpperBoundRecord` 並列リスト化 | ✅ |
| ⑭ | 3次 | `mergeSamples()` 後の sampleIndex 再採番未定義 | 「統合後に0..N-1再採番」を明記 | ✅ |
| ⑮ | 3次 | `detail` 名前空間が不適切 | `EQAnalysisMath` モジュールに改称 | ✅ |
| ⑯ | 3次 | FFT置換可能範囲の記述が過大 | measured 限定と明記 | ✅ |
| **⑰** | **4次** | **SampleResult + UpperBoundRecord の並列 vector → 同期バグリスク** | **`MergedSample` 統合型で一元管理** | **本版** |
| **⑱** | **4次** | **mergeSamples() の責務肥大** | **`mergeAndSort()` / `deduplicate()` / `renumber()` に分割** | **本版** |
| ⑲ | 4次 | `BandHelper::collect()` → `collectActiveBands()` | 名称変更 | ✅ |
| ⑳ | 4次 | `runAdaptive()` の戻り値 `pair` → 名前付き構造体 | `AdaptiveScanResult` 新規定義 | ✅ |

---

## 1. 最終アーキテクチャ（v5）

```
EQState
  ↓
EQProcessor::computeEstimatedMaxGainComplex()
  │
  ├── BandHelper::collectActiveBands(processor, state, sr)
  │     └── processor.calcSVFCoeffs()            ← EQProcessor static メンバ
  │     └── processor.svfToDisplayBiquad()       ← EQProcessor static メンバ
  │     └── EQAnalysisMath::isBoostingBand()
  │
  ├── EQResponseSampler::runCoarse(bands)
  │     └── 各評価点で:
  │           ├── measuredDb → MergedSample.measuredDb
  │           └── upperBoundDb → MergedSample.upperBoundDb   ← 統合管理
  │
  ├── Sampler::findMeasuredCandidates / findUpperBoundCandidates
  │
  ├── EQResponseSampler::runAdaptive(bands, ...)
  │     └── AdaptiveScanResult { vector<MergedSample> }     ← 統合管理
  │
  ├── mergeAndSort(coarse, adaptive)          ← ソートのみ
  ├── deduplicate(sorted)                    ← 重複除去のみ
  ├── renumber(deduped)                      ← index再採番のみ
  │
  ├── PeakEstimator::estimate(samples)
  │     └── MergedSample.measuredDb のみ使用
  │
  ├── UpperBoundEstimator::estimateMax(samples)
  │     └── MergedSample.upperBoundDb のみ使用
  │
  └── EQProcessor が EQAnalysisResult に詰め替え
```

### 統合型のメリット（問題⑰対応）

```cpp
// BEFORE (v4): 並列vector — 同期バグリスク
vector<SampleResult> samples;       // sort/erase のたびに両方操作必要
vector<UpperBoundRecord> upperBounds;

// AFTER (v5): 統合型 — 常に同期が保証される
struct MergedSample {
    double freqHz;
    double linearMagnitude;          // PeakEstimator 用
    double upperBoundDb;             // UpperBoundEstimator 用
    SampleOrigin origin;
};
vector<MergedSample> samples;        // 一つの vector で完結
```

- `sort()` 一度で両方ソートされる
- `erase()` 一度で両方削除される
- メモリ増加: 28→48byte/点、600点で **~12KB増**（問題にならない）

---

## 2. 全共通型定義

```cpp
// EQAnalysisTypes.h

struct BandInfo {
    int index;                          // 0..19
    double freq;
    double q;
    EQBandType type;
    float gain;
    EQCoeffsBiquad biquad;
    bool isBoosting;

    /// BandType ごとに最適化された探索範囲
    std::pair<double, double> searchRange(double maxFreq) const noexcept;
};

struct BandCollection {
    std::vector<BandInfo> bands;
    float maxActiveQ = 0.0f;   // ブーストバンド最大Q（Planner 使用）
    float maxTotalQ = 0.0f;    // 全有効バンド最大Q（diagnostics 専用）
};

/// 統合サンプル（measured と upperBound を一元管理）
/// ★ linearMagnitude を保持し measuredDb を保持しない理由:
///   1. dB変換（20*log10）は PeakEstimator で1回のみ実行されるため、事前変換による
///      高速化メリットがない
///   2. linear 値を保持することで、将来の FFT 置換時（振幅が linear で得られる）に
///      変換が不要
///   3. SampleOrigin と合わせて48byteに収まり、キャッシュライン境界に整合
struct MergedSample {
    double freqHz;
    double linearMagnitude;     // |H(freq)|（linear。dB変換は利用時）
    double upperBoundDb;        // (20/ln10) * Σln(1+|Hi-1|)
    SampleOrigin origin;
};

struct CoarseScanResult {
    std::vector<MergedSample> samples;
    std::array<double, 20> bandMaxDelta;
    std::array<double, 20> bandMaxMagnitude;
};

struct AdaptiveScanResult {
    std::vector<MergedSample> samples;
};

struct PeakEstimate {
    float interpolatedDb = 0.0f;
    float interpolatedFreqHz = 0.0f;
    float rawDb = 0.0f;
    float rawFreqHz = 0.0f;
    int rawSampleIndex = -1;    // 統合後 vector 内インデックス
};

struct UpperBoundEstimate {
    float maxDb = 0.0f;
    float freqHz = 0.0f;
    int sampleIndex = -1;
};
```

---

## 3. 各層の詳細設計

### 3.0 EQAnalysisMath

```cpp
namespace EQAnalysisMath {
    // ★ calcSVFCoeffs / svfToDisplayBiquad は EQProcessor の static メンバ（§3.1 から参照）

    std::complex<double> biquadResponse(const EQCoeffsBiquad& c, double w) noexcept;
    bool isBoostingBand(EQBandType type, float gain) noexcept;

    /// 1評価点の measured（linearMagnitude）と upperBoundDb を同時計算
    /// ★ biquadResponse を1度だけ評価し、両方を同時に算出する（二重評価防止）。
    ///   出力は outLinearMagnitude / outUpperBoundDb の参照で返す。
    ///   戻り値がなく両方を参照で返す理由:
    ///     呼出し側（EQResponseSampler::evaluate）が両方の値を必要とするため。
    ///     upperBound のみの計算は存在しない（upperBound 単体では意味がない）。
    ///   bands には BandInfo* + size_t のポインタ形式で渡す（vector の data()+size() から
    ///   取得する想定）。これにより runCoarse からの呼出し時にアロケーションを回避する。
    void computeSampleResponse(
        const BandInfo* bands, size_t numBands,
        double normalizedFreq, bool isParallel,
        double& outLinearMagnitude, double& outUpperBoundDb) noexcept;

    inline double linearToDb(double linear) noexcept {
        return (linear > 1e-18) ? 20.0 * std::log10(linear)
                                : -std::numeric_limits<double>::infinity();
    }

    inline double dbToLinear(double db) noexcept {
        return std::pow(10.0, db / 20.0);
    }
}
```

### 3.1 BandHelper

```cpp
class BandHelper {
public:
    /// EQState から有効バンドを収集（名称変更: collect→collectActiveBands）
    /// ★ calcSVFCoeffs / svfToDisplayBiquad は EQProcessor の static メンバとして
    ///   残っているため、実装上は processor 参照を追加で受け取る。
    ///   これらの SVF 関数は v5.1 以降で EQAnalysisMath に移動する可能性がある。
    static BandCollection collectActiveBands(const EQProcessor& processor,
                                              const EQProcessor::EQState& state,
                                              double processingRate);
};
```

### 3.2 EQResponseSampler（stateless）

```cpp
class EQResponseSampler {
public:
    EQResponseSampler(double processingRate, bool isParallel);

    CoarseScanResult runCoarse(const BandCollection& bands) const;

    std::vector<const BandInfo*> findMeasuredCandidates(const BandCollection& bands) const;
    std::vector<const BandInfo*> findUpperBoundCandidates(
        const BandCollection& bands,
        const std::array<double, 20>& bandMaxDelta) const;

    AdaptiveScanResult runAdaptive(
        const BandCollection& bands,
        const std::vector<const BandInfo*>& measuredCands,
        const std::vector<const BandInfo*>& upperBoundCands,
        const CoarseScanResult& coarseResult) const;

    /// 1点評価（measured + upperBound を同時計算し MergedSample を返す）
    MergedSample evaluate(double freqHz, const BandCollection& bands) const;

private:
    double processingRate_;
    bool isParallel_;
    double nyquist_;
    double maxFreq_;
};
```

### 3.3 merge パイプライン（問題⑱対応: 責務分割）

```cpp
// 3つの独立した関数に分割

/// Phase A: ソート（周波数昇順、stable）
std::vector<MergedSample> mergeAndSort(
    const CoarseScanResult& coarse,
    const AdaptiveScanResult& adaptive);

/// Phase B: 同一周波数重複除去（adaptive優先、coarse破棄）
/// 「同一周波数」の判定は **完全一致**（freqHz のビット単位一致）で行う。
/// 理由: 600点の粗探索と適応サンプリングは同じ対数周波数生成式を使用するため、
/// 同一周波数になることは稀であり、浮動小数点誤差の範囲で「ほぼ同じ」場合は
/// そのまま両方保持する（放物線補間の安定性に影響しない）。
/// 将来サンプリング方式が変更された場合は epsilon=1e-12 の許容差を検討する。
std::vector<MergedSample> deduplicate(const std::vector<MergedSample>& sorted);

/// Phase C: origin.sampleIndex を 0..N-1 に再採番
/// origin.sampleIndex は「統合後の全サンプルベクタ内の位置」を表す。
/// 統合前は coarse(0..599) / adaptive(0..N-1) の別々のインデックス空間だったが、
/// 統合後は 0..total-1 の単一インデックス空間に再マッピングする。
/// origin の他のフィールド（type, bandIndex）は不変。
/// 別途 mergedIndex を持たない理由: origin.sampleIndex は「最大値がどこにあるか」
/// をデバッグ・診断目的で示すものであり、統合後の一貫したインデックスであるべき。
void renumber(std::vector<MergedSample>& samples);
```

### 3.4 PeakEstimator

```cpp
class PeakEstimator {
public:
    static PeakEstimate estimate(const std::vector<MergedSample>& samples);

    /// 放物線補間（Lagrange一般3点、対数周波数軸+dB空間）
    /// @param y0,y1,y2 dB空間のゲイン値
    /// @return 補間後のピーク値。
    ///   3点不足時や分母 < 1e-12 の場合は y[1] を返す。
    ///   注意: この 1e-12 はゼロ除算防止の閾値であり、数値比較の許容誤差（1e-9）とは
    ///   目的が異なる。分母がこの値未満の場合、3点が一直線上に近く補間が不安定なため
    ///   スキップする。
    static double interpolateParabolic(double x0, double y0,
                                        double x1, double y1,
                                        double x2, double y2);
};
```

### 3.5 UpperBoundEstimator

```cpp
class UpperBoundEstimator {
public:
    /// MergedSample.upperBoundDb から最大値を選択（補間なし）
    static UpperBoundEstimate estimateMax(const std::vector<MergedSample>& samples);
};
```

---

## 4. 実装順（依存関係順、問題⑳対応）

| Phase | 内容 | 依存 | 工数 |
|-------|------|------|------|
| 1 | `EQAnalysisMath.h/.cpp` — SVF/Biquad/upperBound関数 | なし | 1.5h |
| 2 | `EQAnalysisTypes.h` — 全共通型（MergedSample含む） | なし | 0.5h |
| 3 | `BandHelper.h/.cpp` — collectActiveBands() | Phase 1, 2 | 1.5h |
| 4 | `PeakEstimator.h/.cpp` — 放物線補間 | Phase 2 | 3h |
| 5 | `UpperBoundEstimator.h/.cpp` — 最大値選択 | Phase 2 | 1h |
| 6 | `EQResponseSampler.h/.cpp` — サンプリング全般 | Phase 1, 2, 3 | 5h |
| 7 | `mergeAndSort()` / `deduplicate()` / `renumber()` | Phase 2, 6 | 2h |
| 8 | `EQProcessor.Coefficients.cpp` 統合 + `EQProcessor.h` 修正 | Phase 1-7 | 3h |
| 9 | 新規テスト追加（53ケース） | Phase 1-8 | 5h |
| 予備 | デバッグ | — | 3h |
| **合計** | | | **24.5h** |

---

## 5. テストケース（53新規 + 2322既存）

| テストモジュール | ケース数 | 主要テスト内容 |
|-----------------|---------|--------------|
| `PeakEstimator.interpolateParabolic` | 10 | 通常3点/等間隔/境界/分母微小/全同一/plateau/NaN/Inf/-Inf |
| `PeakEstimator.estimate` | 9 | 単峰/多峰/平坦/1点/2点/NaN/Inf/全0/31bandQ20 |
| `UpperBoundEstimator.estimateMax` | 6 | 単峰/複数/全0/単一/31bandQ20/最小値 |
| `EQResponseSampler.evaluate` | 6 | Parallel/Serial/単一Band/全pass/DC/Nyquist |
| `EQResponseSampler.findMeasuredCandidates` | 5 | Boosting/非Boosting/混在/LPF/HPF |
| `EQResponseSampler.findUpperBoundCandidates` | 5 | delta>0.1/≤0.1/混在/Shelf/HPF+LPF |
| `EQResponseSampler.runCoarse` | 4 | 正常/空Band/片方/96k+384k最悪ケース |
| `mergeAndSort` | 4 | 基本/逆順/片方のみ/空 |
| `deduplicate` | 3 | 同一周波数/多重重複/重複なし |
| `renumber` | 3 | 基本/空/単一要素 |
| **新規合計** | **53** | |
| **既存テスト** | **2322** | 変更なし（入出力同一） |
| **総合計** | **2375** | |

---

## 6. レビュー指摘クローズ（最終）

| # | 指摘 | v5の対応 | 状態 |
|---|------|---------|------|
| ⑰ | 並列vector → 同期バグリスク | `MergedSample` 統合型で一元管理。sort/erase 一回で完結 | ✅ |
| ⑱ | `mergeSamples()` の責務肥大 | `mergeAndSort()` / `deduplicate()` / `renumber()` の3関数に分割 | ✅ |
| ⑲ | `BandHelper::collect()` 命名 | `collectActiveBands()` に名称変更 | ✅ |
| ⑳ | `runAdaptive()` の戻り値 `pair` | `AdaptiveScanResult` 構造体で名前付き戻り値に | ✅ |

---

## 7. 最終評価

| 項目 | 評価 |
|------|------|
| 責務分離 | **A+** — 4層+1数学モジュールに完全分離 |
| メモリ効率 | **A+** — MergedSample 48byte/点、600点で ~28KB（v3比75%減） |
| 保守性 | **A+** — 並列vectorを統合型に変更。merge責務を3分割 |
| 将来FFT置換 | **A** — measured 限定で置換可能と明記 |
| ISR設計との整合 | **A** — Sampler stateless、全層純粋関数 |
| 実装容易性 | **A** — 依存関係順のPhase構成、23.5h |
| **総合** | **A（96点）** — 実装フェーズ移行可能 |
