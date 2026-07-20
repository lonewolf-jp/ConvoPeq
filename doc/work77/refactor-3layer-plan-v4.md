# EQAnalysis 3層分割 改修計画書 v4（最終案）

> 作成: 2026-07-20 | 改定: 2026-07-20（第3次レビュー反映）
> 対象設計書: AutoGainStagingRenewal.md v14.47 §4.1.7
> 現状: `computeEstimatedMaxGainComplex()` 一関数（約550行）
> 目標: 4層への分割（Sampler / PeakEstimator / UpperBoundEstimator / MathUtility）

---

## 0. レビュー指摘対応マトリクス

### 第1次〜第3次 全17項目

| # | レビュー | 指摘 | 対応 | 状態 |
|---|---------|------|------|------|
| ① | 1次 | `SampleResult` に `upperBoundDb` 保持 → 責務混在 | `SampleResult` から削除。別途 `UpperBoundRecord` で管理 | ✅ |
| ② | 1次 | `bandMaxDelta_` を Sampler が状態保持 | `CoarseScanResult` として返す。Sampler stateless に | ✅ |
| ③ | 1次 | `PeakEstimator` が `EQAnalysisResult` 依存 | `PeakEstimate` 新規定義 | ✅ |
| ④ | 1次 | `mergeSamples()` 仕様未定義 | 仕様明文化（index再採番含む） | ✅ |
| ⑤ | 1次 | `SampleResult` が dB のみ | `linearMagnitude` 追加 | ✅ |
| ⑥ | 1次 | `SamplingStrategy` enum | 削除 | ✅ |
| ⑦ | 1次 | `getSearchRange()` 配置 | `BandInfo::searchRange()` へ移動 | ✅ |
| ⑧ | 1次 | `maxActiveQ` 宙に浮く | `BandCollection` に統合 | ✅ |
| ⑨ | 2次 | `BandCollector` が `EQProcessor` に依存（循環） | SVF関数を `EQAnalysisMath` の自由関数に抽出。`BandHelper` に改名 | ✅ |
| ⑩ | 2次 | `perBandDeltas` が `vector<double>` → 600×heap | `std::array<double, 20>` に変更 | ✅ |
| ⑪ | 2次 | `mergeSamples()` 重複保持 → Lagrange分母リスク | 「adaptive優先、coarse破棄」に変更 | ✅ |
| ⑫ | 2次 | 許容誤差 1e-12 が非現実的 | 絶対誤差 **1e-9** に緩和 | ✅ |
| ⑬ | **3次** | **`SampleResult` が重すぎる（全点が20バンド配列保持）** | **`SampleResult` から `perBandDeltas` を削除。並列の `UpperBoundRecord` で管理** | **本版で対応** |
| ⑭ | **3次** | **`mergeSamples()` 後の `sampleIndex` 再採番未定義** | **「統合後に全要素の index を 0..N-1 で再採番」を明記** | **本版で対応** |
| ⑮ | **3次** | **`detail` 名前空間が不適切** | **`EQAnalysisMath` モジュール名に変更。全層から利用する共有数学関数として明確化** | **本版で対応** |
| ⑯ | **3次** | **FFT置換可能範囲の記述が過大** | **「measured 線形振幅のみFFT置換可能。per-band応答は別方式」と現実的に修正** | **本版で対応** |

---

## 1. 現状分析

`EQProcessor::computeEstimatedMaxGainComplex()` (`EQProcessor.Coefficients.cpp:394-895`, 約500行) に以下の全責務が集中:

| 責務 | 行数 |
|------|------|
| バンド情報収集（SVF→Biquad変換） | ~30 |
| 粗探索600点（measured + upperBound 同時計算） | ~80 |
| 候補Band判定（measured用 / upperBound用） | ~60 |
| Shelf/LPF追加評価 | ~50 |
| union区間統合 (`mergeRanges`) | ~30 |
| 適応サンプリング（measured + upperBound 同時計算） | ~90 |
| 放物線補間 + 結果選択 | ~60 |
| maxActiveQ算出 | ~10 |
| 結果パッケージ | ~10 |

---

## 2. 目標アーキテクチャ

```
EQState
  ↓
EQProcessor::computeEstimatedMaxGainComplex()
  │
  ├── BandHelper::collect(state, sr)
  │     └── EQAnalysisMath::calcSVFCoeffs()   ← 自由関数、専用モジュール
  │     └── EQAnalysisMath::svfToDisplayBiquad()
  │     └── EQAnalysisMath::isBoostingBand()
  │
  ├── EQResponseSampler::runCoarse(bands)
  │     └── 各評価点で:
  │           ├── linearMagnitude  → SampleResult として蓄積（軽量）
  │           └── upperBoundDb     → UpperBoundRecord として並列蓄積（軽量）
  │           └── bandMaxDelta[]   → CoarseScanResult に集約
  │
  ├── EQResponseSampler::findMeasuredCandidates(bands)
  ├── EQResponseSampler::findUpperBoundCandidates(bands, bandMaxDelta)
  │
  ├── EQResponseSampler::runAdaptive(bands, ...)
  │     └── 同様に SampleResult と UpperBoundRecord を並列蓄積
  │
  ├── mergeSamples()
  │     └── coarse + adaptive → ソート + 重複除去 + index再採番
  │     └── 戻り値: { mergedSamples, mergedUpperBounds }
  │
  ├── PeakEstimator::estimate(mergedSamples)
  │     └── SampleResult.freqHz + .linearMagnitude のみ使用
  │     └── → PeakEstimate
  │
  ├── UpperBoundEstimator::estimateMax(mergedUpperBounds)
  │     └── UpperBoundRecord.freqHz + .upperBoundDb のみ使用
  │     └── → UpperBoundEstimate
  │
  └── EQProcessor が EQAnalysisResult に詰め替え
```

**重要な設計判断**: `upperBoundDb` はサンプリング時に `biquadResponse()` の結果から直接計算し、`UpperBoundRecord` に蓄積する。これにより `perBandDeltas` の保存が不要になり、`SampleResult` が軽量化される。`UpperBoundEstimator` は `UpperBoundRecord` リストから最大値を選択するのみ。

---

## 3. 共通型定義（変更点のみ）

```cpp
// EQAnalysisTypes.h

/// 【軽量】PeakEstimator 用サンプル（perBandDeltas は保持しない）
struct SampleResult {
    double freqHz;
    double linearMagnitude;   // |H(freq)|（linear、dB変換は利用時）
    SampleOrigin origin;
};

/// 【軽量】UpperBoundEstimator 用サンプル（別リストで管理）
struct UpperBoundRecord {
    double freqHz;
    double upperBoundDb;      // (20/ln10) * Σln(1+|Hi-1|)
    SampleOrigin origin;
};

/// 統合結果（mergeSamples の戻り値）
struct MergedSamples {
    std::vector<SampleResult> samples;          // PeakEstimator 用
    std::vector<UpperBoundRecord> upperBounds;  // UpperBoundEstimator 用
    // ↑ 両ベクトルは同じ要素数 + 同じ周波数順
};

/// 粗探索結果
struct CoarseScanResult {
    std::vector<SampleResult> samples;
    std::vector<UpperBoundRecord> upperBounds;   // 並列リスト（新規）
    std::array<double, 20> bandMaxDelta;
    std::array<double, 20> bandMaxMagnitude;
};
```

**なぜこれで問題③が解決するか**:
- `SampleResult`: `freqHz(8) + linearMagnitude(8) + origin(12) ≈ 28byte` → 600点で **~17KB**
- `UpperBoundRecord`: `freqHz(8) + upperBoundDb(8) + origin(12) ≈ 28byte` → 600点で **~17KB**
- 合計 **~34KB**（v3の114KBから **70%削減**）
- `perBandDeltas(160byte)` を全サンプルが持たなくなった
- キャッシュ局所性も向上

**なぜ upperBoundDb を SampleResult に含めないか**: 責務分離のため。`PeakEstimator` は linearMagnitude のみを必要とし、`UpperBoundEstimator` は upperBoundDb のみを必要とする。両者を同じ構造体にすると「使わないデータ」がメモリ帯域を消費する。

---

## 4. 各層の詳細設計

### 4.0 EQAnalysisMath（問題⑮対応: `detail` → 専用モジュール）

```cpp
// EQAnalysisMath.h — 全層から利用される共有数学関数
// detail ではなく意味のあるモジュール名
namespace EQAnalysisMath {
    // SVF→Biquad変換
    EQCoeffsSVF calcSVFCoeffs(EQBandType type, float freq, float gainDb, float q, double sr) noexcept;
    EQCoeffsBiquad svfToDisplayBiquad(const EQCoeffsSVF& svf) noexcept;

    // 周波数応答
    std::complex<double> biquadResponse(const EQCoeffsBiquad& c, double w) noexcept;
    bool isBoostingBand(EQBandType type, float gain) noexcept;

    // UpperBound 計算（1評価点）
    double computeUpperBoundDb(const std::vector<BandInfo>& bands,
                                double normalizedFreq, bool isParallel) noexcept;

    // ゲイン変換
    inline double linearToDb(double linear) noexcept {
        return (linear > 1e-18) ? 20.0 * std::log10(linear) : -std::numeric_limits<double>::infinity();
    }
}
```

**`computeUpperBoundDb()` の役割**: サンプラが各評価点で measured（linearMagnitude）と同時に upperBoundDb を計算するための共有関数。これにより `perBandDeltas[]` の保存が不要になる。計算内容は現行の `Σlog1p(|Hi-1|) → kTwentyOverLog10 * Σ` と同一。**サンプラは biquadResponse を評価するループ内で同時に upperBoundDb も計算する**（現状と同じ。追加コストなし）。

### 4.1 BandHelper

```cpp
// BandHelper.h（変更なし）
class BandHelper {
public:
    static BandCollection collect(const EQState& state, double processingRate);
};
```

内部で `EQAnalysisMath::calcSVFCoeffs()` / `EQAnalysisMath::svfToDisplayBiquad()` / `EQAnalysisMath::isBoostingBand()` を呼ぶ。

### 4.2 EQResponseSampler（stateless、軽量出力）

```cpp
class EQResponseSampler {
public:
    EQResponseSampler(double processingRate, bool isParallel);

    /// 粗探索600点を実行
    /// 戻り値の samples と upperBounds は同じ要素数・同じ周波数順
    CoarseScanResult runCoarse(const BandCollection& bands) const;

    std::vector<const BandInfo*> findMeasuredCandidates(const BandCollection& bands) const;
    std::vector<const BandInfo*> findUpperBoundCandidates(
        const BandCollection& bands,
        const std::array<double, 20>& bandMaxDelta) const;

    /// 適応サンプリング
    /// samples と upperBounds は同じ要素数・同じ周波数順
    std::pair< std::vector<SampleResult>,
               std::vector<UpperBoundRecord> > runAdaptive(
        const BandCollection& bands,
        const std::vector<const BandInfo*>& measuredCands,
        const std::vector<const BandInfo*>& upperBoundCands,
        const CoarseScanResult& coarseResult) const;

    /// 1点評価（measured + upperBound を同時計算）
    std::pair<SampleResult, UpperBoundRecord> evaluate(
        double freqHz, const BandCollection& bands) const;

private:
    double processingRate_;
    bool isParallel_;
    double nyquist_;
    double maxFreq_;
};
```

### 4.3 PeakEstimator

```cpp
class PeakEstimator {
public:
    static PeakEstimate estimate(const std::vector<SampleResult>& samples);
    static double interpolateParabolic(double x0, double y0,
                                        double x1, double y1,
                                        double x2, double y2);
};
```

変更なし（v3から）。`SampleResult.linearMagnitude` を入力に使用。

### 4.4 UpperBoundEstimator

```cpp
class UpperBoundEstimator {
public:
    static UpperBoundEstimate estimateMax(const std::vector<UpperBoundRecord>& records);
};
```

**v3からの変更**: `computeLogBound(const array<double,20>&)` を削除。代わりに `EQAnalysisMath::computeUpperBoundDb()` をサンプラが呼び出す。`UpperBoundEstimator` は記録済みの `upperBoundDb` から最大値を選択するのみ。

---

## 5. mergeSamples() 仕様（問題④⑭対応）

```cpp
/// 粗探索 + 適応サンプリング結果を統合
///
/// 仕様:
/// 1. coarse.samples / coarse.upperBounds の全要素を先に追加
/// 2. adaptive.samples / adaptive.upperBounds の全要素を後に追加
/// 3. std::stable_sort（周波数昇順、同値は coarse → adaptive の投入順を維持）
/// 4. 【v4】同一周波数が coarse と adaptive の両方に存在する場合、
///    adaptive を残し coarse を破棄する。
///    理由: Lagrange 放物線補間は隣接3点の x 座標が異なることを前提とする。
///    適応サンプリングは粗探索より高密度なため adaptive を優先。
/// 5. 【v4 追加】統合後に全サンプルの origin.sampleIndex を 0..N-1 で
///    再採番する。これにより PeakEstimate.rawSampleIndex と
///    UpperBoundEstimate.sampleIndex が統合後のベクタ内インデックスを指すことが保証される。
/// 6. samples と upperBounds の両ベクタは同一長・同一周波数順であることを保証
///
/// @return MergedSamples { samples, upperBounds }
static MergedSamples mergeSamples(
    const CoarseScanResult& coarse,
    const std::vector<SampleResult>& adaptiveSamples,
    const std::vector<UpperBoundRecord>& adaptiveUpperBounds);
```

---

## 6. FFT置換の現実的範囲（問題⑯対応）

| 主張 | 現実 |
|------|------|
| 「FFTベース探索へ置換可能」 | ⚠️ measured 線形振幅（SampleResult.linearMagnitude）の評価に限定される |
| 「per-band応答もFFTで計算可能」 | ❌ FFTは全バンドの合成応答しか得られない。upperBound に必要な per-band `\|Hi-1\|` は別途計算が必要 |

**現実的なFFT置換シナリオ**:

```
現行: biquadResponse() を600+適応点で個別評価
  ↓
FFT置換: 各バンドの biquad を周波数領域で畳み込み、一括で |H_total| を取得
  ├── measured (linearMagnitude): ✅ FFTで高速化可能
  └── per-band |Hi-1|: ❌ 別途計算が必要（FFT単体では取得不可）
```

**結論**: `EQResponseSampler` の measured 部分（線形振幅の取得）は FFT 置換が可能。upperBound 部分は引き続き biquad ベースの評価が必要。計画書の「FFT置換可能」の記述は measured に限定する。

---

## 7. ファイル構成

| ファイル | 種別 | 内容 |
|---------|------|------|
| `src/eqprocessor/EQAnalysisMath.h` | **新規** | 共有数学関数（`detail` から改称） |
| `src/eqprocessor/EQAnalysisMath.cpp` | **新規** | SVF/Biquad/upperBound計算実装 |
| `src/eqprocessor/EQAnalysisTypes.h` | **新規** | 全共通型定義（軽量構造体） |
| `src/eqprocessor/BandHelper.h` | **新規** | `BandHelper::collect()` 宣言 |
| `src/eqprocessor/BandHelper.cpp` | **新規** | 実装 |
| `src/eqprocessor/EQResponseSampler.h` | **新規** | Sampler 宣言 |
| `src/eqprocessor/EQResponseSampler.cpp` | **新規** | サンプリング実装 |
| `src/eqprocessor/PeakEstimator.h` | **新規** | PeakEstimator 宣言 |
| `src/eqprocessor/PeakEstimator.cpp` | **新規** | 補間実装 |
| `src/eqprocessor/UpperBoundEstimator.h` | **新規** | UpperBoundEstimator 宣言 |
| `src/eqprocessor/UpperBoundEstimator.cpp` | **新規** | 最大値選択実装 |
| `src/eqprocessor/EQProcessor.Coefficients.cpp` | **変更** | 約35行の統合コードに |
| `src/eqprocessor/EQProcessor.h` | **変更** | static SVF関数の宣言削除 |
| `CMakeLists.txt` | **変更** | 新規ファイル追加 |

---

## 8. 結果整合性検証

| Phase | 検証方法 | 許容誤差 |
|-------|---------|---------|
| Phase 2-4 (個別抽出) | 新旧結果比較（同関数内で両方実行） | 絶対誤差 **1e-9** |
| Phase 5 (統合) | 全2322テスト実行 | 絶対誤差 **1e-9** |
| 最終確認 | Debug/Release + 全テスト | 全 PASS |

**v4 の誤差要因**: 従来版に比べ演算順序の変更がない（upperBound 計算をサンプリング時に行う点は現行と同じ）。`UpperBoundRecord` として別リスト化するのみで計算内容に変更なし。

---

## 9. 移行手順

| Phase | 内容 | 工数 |
|-------|------|------|
| Phase 0 | `EQAnalysisMath.h/.cpp` 作成 + `EQProcessor.h` の static 宣言削除（`EQAnalysisMath::` に移行） | 1.5h |
| Phase 1 | `EQAnalysisTypes.h` + `BandHelper` | 2h |
| Phase 2 | `EQResponseSampler`（軽量SampleResult + 並列UpperBoundRecord） | 5h |
| Phase 3 | `PeakEstimator` | 3h |
| Phase 4 | `UpperBoundEstimator`（単純な最大値選択に） | 1h |
| Phase 5 | 統合 + 既存2322テスト検証 | 3h |
| Phase 6 | 新規テスト追加 | 5h |
| 予備 | デバッグ | 3h |
| **合計** | | **23.5h** |

---

## 10. テストケース

| テスト | ケース数 | 状態 |
|--------|---------|------|
| 既存テスト（変更不要） | 2322 | ✅ 継続利用 |
| `PeakEstimator.interpolateParabolic` | 10 | 新規 |
| `PeakEstimator.estimate` | 9（+3: plateau/NaN/Inf） | 新規 |
| `UpperBoundEstimator.estimateMax` | 6（+2: 31bandQ20/全0） | 新規 |
| `EQResponseSampler.evaluate` | 6 | 新規 |
| `EQResponseSampler.findMeasuredCandidates` | 5 | 新規 |
| `EQResponseSampler.findUpperBoundCandidates` | 5 | 新規 |
| `EQResponseSampler.runCoarse` | 4 | 新規 |
| `mergeSamples` | 7（+3: 重複/逆順/再採番） | 新規 |
| 回帰テスト（最悪ケース: 31band+Q20+24dB+Parallel+96k） | 1 | 新規 |
| **新規合計** | **53** | |
| **総合計** | **2375** | |

---

## 11. v4 改訂の根拠サマリ

| # | 指摘 | v3の状態 | v4の変更 | メリット |
|---|------|---------|----------|---------|
| ⑬ | SampleResultが重い | `array<double,20>` を全サンプルが保持 | `UpperBoundRecord` を並列リスト化、`SampleResult` を軽量化 | メモリ70%削減（114KB→34KB）、キャッシュ局所性向上 |
| ⑭ | merge後のindex再採番 | 未定義 | 「統合後に全要素0..N-1再採番」を明記 | 仕様の一意性保証、テスト容易性向上 |
| ⑮ | `detail`名前空間 | `detail/svf_utils.h` | `EQAnalysisMath` モジュールに改称 | 意味明確、全層からの利用が自然 |
| ⑯ | FFT置換範囲が過大 | 「FFT置換可能」と記載 | measured限定と明記。per-band応答は別設計課題に | 現実的な目標設定、誤解防止 |

---

## 12. リスクと対策

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| UpperBoundRecord と SampleResult の周波数順不一致 | 低 | 高 | `evaluate()` が `pair` を返す。`mergeSamples()` で常に両方を同時操作 |
| 重複除去で measured 最大値が変わる | 低 | 中 | adaptive優先の根拠（密度）をテストで検証 |
| `EQAnalysisMath` の命名が長い | 低 | 低 | コード補完で十分実用的 |
| FFT置換を真に必要とする日が来ない | 中 | 低 | 現状の biquad 評価で性能十分。FFT置換は「可能」であって「必須」ではない |
