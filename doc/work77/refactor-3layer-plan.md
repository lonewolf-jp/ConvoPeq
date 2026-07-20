# EQAnalysis 3層分割 改修計画書

> 作成: 2026-07-20 | 対象設計書: AutoGainStagingRenewal.md v14.47 §4.1.7
> 現状: `computeEstimatedMaxGainComplex()` 一関数（約550行）に全ロジック集中
> 目標: 3層（EQResponseSampler / PeakEstimator / UpperBoundEstimator）への分割

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

1. **単体テスト困難**: 内部ロジック（例: 放物線補間のエッジケース）をテストするには関数全体を呼び出す必要がある
2. **FFT将来置換への障害**: サンプリングロジックが密結合のため、FFTベース探索に差し替える際に全ロジックの書き換えが必要
3. **コード理解の難易度**: 550行の一関数は Cognitve Complexity が高い
4. **設計書未定義**: §4.1.7 は参照のみで詳細未定義。実装者がインターフェースを設計する必要がある

---

## 2. 目標アーキテクチャ

```
BandParams[]
     ↓
EQResponseSampler           ← 周波数点生成(粗探索+適応+union)
     ↓
SamplePoint[] (freq, measuredGain, perBandDeltas)
     ↓
PeakEstimator               ← measured 最大値 + 放物線補間
UpperBoundEstimator         ← Π(1+|Hi-1|) 最大値
     ↓
EQAnalysisResult { measured, measuredRaw, upperBound, maxActiveQ, algorithm }
```

### 2.1 インターフェース

```cpp
// === 第1層: サンプリング ===

/// 1評価点における計算結果
struct SampleResult {
    double freqHz;
    double measuredDb;          // 20*log10(|H|)
    double upperBoundDb;        // kTwentyOverLog10 * Σln(1+|Hi-1|)
    double rawUpperBoundDb;     // デバッグ用。未補正logBound
};

/// サンプリング戦略
enum class SamplingStrategy {
    Coarse,    // 粗探索600点
    Adaptive,  // 適応128点/バンド
    ShelfExtra // Shelf追加評価点
};

/// サンプリング点を生成し評価する
class EQResponseSampler {
public:
    /// バンド情報で初期化（コピー軽量のため参照推奨）
    EQResponseSampler(const std::vector<BandInfo>& bands, double processingRate, bool isParallel);

    /// 粗探索600点を実行
    std::vector<SampleResult> runCoarse();

    /// measured 用候補Bandを判定（isBoosting）
    std::vector<const BandInfo*> findMeasuredCandidates() const;

    /// upperBound 用候補Bandを判定（max|Hi-1| > 0.1）
    std::vector<const BandInfo*> findUpperBoundCandidates() const;

    /// 適応サンプリング（union統合+比例配分）
    /// @param measuredCands  measured用候補バンド
    /// @param upperBoundCands upperBound用候補バンド
    /// @param coarseResults   粗探索結果（領域統合用）
    /// @return 統合後の評価点
    std::vector<SampleResult> runAdaptive(
        const std::vector<const BandInfo*>& measuredCands,
        const std::vector<const BandInfo*>& upperBoundCands,
        const std::vector<SampleResult>& coarseResults);

    /// 1点評価（内部共有）
    SampleResult evaluate(double freqHz) const;

    /// 候補バンドの探索範囲を取得
    static std::pair<double, double> getSearchRange(const BandInfo& band);

private:
    const std::vector<BandInfo>& bands_;   // 有効バンド情報
    double processingRate_;
    bool isParallel_;
    double nyquist_;
    double maxFreq_;
    // 粗探索時の max|Hi-1| を保持（upperBound候補判定用）
    std::vector<double> bandMaxDelta_;
    // 粗探索時の measured 振幅を保持
    std::vector<double> bandMaxMeasuredMag_;
};


// === 第2層: measured ピーク推定 ===

class PeakEstimator {
public:
    /// サンプリング結果から measured 最大値を推定（放物線補間含む）
    /// @param samples 全サンプル（周波数昇順ソート済み）
    /// @param coarseCount 粗探索点数（適応サンプリングと区別するため）
    /// @return measured.gainDb: 補間後max / measuredRawGainDb: 補間前max
    static EQAnalysisResult::PeakInfo estimate(
        const std::vector<SampleResult>& samples, int coarseCount);

    /// 放物線補間（Lagrange一般3点、対数周波数軸+dB空間）
    /// @return 補間後のピーク値 (gainDb)。3点不足や発散時はy[1]を返す
    static double interpolateParabolic(
        double x0, double y0,
        double x1, double y1,
        double x2, double y2);

    /// 大域的最大値を探索
    static int findGlobalPeak(const std::vector<SampleResult>& samples);
};


// === 第3層: upperBound 推定 ===

class UpperBoundEstimator {
public:
    /// サンプリング結果から upperBound 最大値を選択（補間なし）
    static EQAnalysisResult::PeakInfo estimateMax(
        const std::vector<SampleResult>& samples);

    /// upperBound の保守性指標 boundExcessDb を計算
    /// boundExcessDb = max(0, upperBoundDb - measuredDb)
    static float computeExcess(float upperBoundDb, float measuredDb);
};
```

---

## 3. ファイル分割案

### 3.1 新規ファイル

| ファイル | 内容 |
|---------|------|
| `src/eqprocessor/EQResponseSampler.h` | `EQResponseSampler` クラス宣言 |
| `src/eqprocessor/EQResponseSampler.cpp` | サンプリング実装（粗探索600点、適応128点、union統合） |
| `src/eqprocessor/PeakEstimator.h` | `PeakEstimator` クラス宣言 |
| `src/eqprocessor/PeakEstimator.cpp` | 放物線補間実装（Lagrange一般3点、エッジケース含む） |
| `src/eqprocessor/UpperBoundEstimator.h` | `UpperBoundEstimator` クラス宣言 |
| `src/eqprocessor/UpperBoundEstimator.cpp` | upperBound 最大値選択 + boundExcessDb計算 |

### 3.2 共通ヘッダ

| ファイル | 内容 |
|---------|------|
| `src/eqprocessor/EQAnalysisTypes.h` | `SampleResult`, `BandInfo` 等の共通型定義（新規作成、または既存 `EQProcessor.h` に追加） |

### 3.3 既存ファイルの変更

| ファイル | 変更内容 |
|---------|---------|
| `src/eqprocessor/EQProcessor.Coefficients.cpp` | `computeEstimatedMaxGainComplex()` を3層呼び出しに置換（元の550行→約30行） |
| `CMakeLists.txt` | 新規6ファイルをソース一覧に追加 |

---

## 4. 移行手順

### Phase 1: 共通型の分離（1日）

1. `EQProcessor.Coefficients.cpp` から `SamplePoint`, `BandInfo`, `CandidateBand` 構造体を抽出
2. `EQAnalysisTypes.h` に移動
3. 元ファイルが `#include "EQAnalysisTypes.h"` するように変更
4. **検証**: 既存コードがコンパイル・テスト通過すること

### Phase 2: EQResponseSampler 抽出（2日）

1. `EQResponseSampler.h/.cpp` 作成
2. 粗探索ループを `runCoarse()` に移動
3. 候補Band判定を `findMeasuredCandidates()` / `findUpperBoundCandidates()` に移動
4. union統合＋適応サンプリングを `runAdaptive()` に移動
5. `evaluate()` に1点評価ロジックを抽出
6. `getSearchRange()` に範囲決定ロジックを抽出
7. **検証**: 既存の `computeEstimatedMaxGainComplex()` が `EQResponseSampler` を使用して同じ結果を返すこと

### Phase 3: PeakEstimator 抽出（1日）

1. `PeakEstimator.h/.cpp` 作成
2. 放物線補間ロジックを `interpolateParabolic()` に移動
3. 大域ピーク探索を `findGlobalPeak()` に移動
4. `estimate()` で measured 最大値＋補間を実行
5. **検証**: 補間結果が現行と同一であること（±1e-6）
6. **エッジケース追加**:
   - 3点不足（先頭/末尾） → 補間なし
   - 分母1e-12未満 → 補間スキップ
   - 全点同一値 → そのまま返す

### Phase 4: UpperBoundEstimator 抽出（0.5日）

1. `UpperBoundEstimator.h/.cpp` 作成
2. `estimateMax()` で upperBound 最大値を評価点から選択
3. `computeExcess()` で boundExcessDb 計算
4. **検証**: 結果が現行と同一であること

### Phase 5: 統合（1日）

1. `computeEstimatedMaxGainComplex()` を以下に置換:
   ```cpp
   EQAnalysisResult EQProcessor::computeEstimatedMaxGainComplex(
       const EQState& state, double processingRate) const
   {
       // 1. バンド情報収集（現状維持）
       auto activeBands = collectActiveBands(state, processingRate);
       if (activeBands.empty()) return {};

       // 2. サンプリング
       EQResponseSampler sampler(activeBands, processingRate, isParallel);
       auto coarse = sampler.runCoarse();
       auto measCands = sampler.findMeasuredCandidates();
       auto ubCands = sampler.findUpperBoundCandidates();
       auto adaptive = sampler.runAdaptive(measCands, ubCands, coarse);

       // 3. 全サンプル統合
       auto allSamples = mergeSamples(coarse, adaptive);

       // 4. measured 推定
       auto measuredInfo = PeakEstimator::estimate(allSamples, coarse.size());

       // 5. upperBound 推定
       auto upperBoundInfo = UpperBoundEstimator::estimateMax(allSamples);

       // 6. 結果生成
       EQAnalysisResult result;
       result.measured = measuredInfo;
       result.measuredRawGainDb = /* 補間前の値を別途保存 */;
       result.upperBound = upperBoundInfo;
       result.maxActiveQ = computeMaxActiveQ(activeBands);
       result.algorithm = convo::EqGainAlgorithm::TriangleProductV1;
       return result;
   }
   ```
2. **既存コード**: `biquadResponse()`, `isBoostingBand()` は `internalLinkage` として維持
3. **検証**:
   - Debugビルド成功
   - `GainStagingContractTests` 2143テスト通過
   - `EQProcessorMaxGainTests` 179テスト通過
   - 結果が現行とビット単位で一致すること

---

## 5. リスクと対策

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| 浮動小数点誤差による結果不一致 | 中 | 高 | `interpolateParabolic()` は現行と同じ `std::log`/`std::pow`/`std::exp` を使用。Phase2-4で段階的に検証 |
| union統合ロジックの再現ミス | 低 | 高 | `mergeRanges` をそのまま `EQResponseSampler` の private static 関数として移動 |
| パフォーマンス劣化 | 低 | 中 | 関数呼び出しオーバーヘッドは微少。プロファイリングで確認 |
| CMakeLists.txt 編集ミス | 低 | 中 | インクリメンタルビルドでリンクエラーを早期検出 |

---

## 6. 単体テスト追加計画

| テスト | ファイル | テストケース数 | 内容 |
|--------|---------|--------------|------|
| `PeakEstimator.interpolateParabolic` | `EQProcessorMaxGainTests.cpp` | 10 | 通常3点/等間隔/境界/分母微小/全同一 |
| `PeakEstimator.estimate` | 同上 | 5 | 単峰/多峰/平坦/1点のみ/2点のみ |
| `UpperBoundEstimator.estimateMax` | 同上 | 3 | 単峰/複数候補/全0 |
| `UpperBoundEstimator.computeExcess` | 同上 | 4 | normal/負値/0/大値 |
| `EQResponseSampler.evaluate` | 同上 | 5 | Parallel/Serial/単一Band/全バイパス/DC |
| `EQResponseSampler.findMeasuredCandidates` | 同上 | 5 | Boostingのみ/非Boosting混在/全Boost/全非Boost/LPF混在 |
| `EQResponseSampler.findUpperBoundCandidates` | 同上 | 5 | delta>0.1/≤0.1/混在/Shelf/HPF+LPF |

---

## 7. スケジュール

| Phase | 内容 | 工数 | テスト影響 |
|-------|------|------|-----------|
| Phase 1 | 共通型分離 | 1h | ❌ なし |
| Phase 2 | EQResponseSampler抽出 | 3h | ❌ 内部リファクタリング |
| Phase 3 | PeakEstimator抽出 | 2h | ❌ 同上 |
| Phase 4 | UpperBoundEstimator抽出 | 0.5h | ❌ 同上 |
| Phase 5 | 統合 & 既存テスト検証 | 2h | ⚠️ 2322テスト要再実行 |
| 追加テスト | 新規テストケース作成 | 2h | ✅ 37テスト追加 |
| **合計** | | **10.5h** | |

---

## 8. 見積もり結果

| 指標 | 値 |
|------|-----|
| 新規作成ファイル数 | 6 (.h x3 + .cpp x3) |
| 変更ファイル数 | 2 (EQProcessor.Coefficients.cpp + CMakeLists.txt) |
| 削除されるコード行数 | ~500行（`computeEstimatedMaxGainComplex` 現行） |
| 追加されるコード行数 | ~600行（3層 + テスト） |
| 総作業時間 | **約10.5時間** |
| 既存テストへの影響 | **なし**（結果が同一であることを検証） |
| 優先度 | **低**（動作に影響しないリファクタリング） |
