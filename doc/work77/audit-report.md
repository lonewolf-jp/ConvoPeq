
# Auto Gain Staging Renewal 実装監査レポート

> 作成: 2026-07-20 | 対象設計書: AutoGainStagingRenewal.md v14.47
> 監査方法: WSL grep/rg + ソースコード直接検証

## 凡例

| 記号 | 意味 |
|------|------|
| ✅ | 設計通り実装済み |
| ⚠️ | 実装されているが設計との乖離あり |
| ❌ | 未実装 |
| 🔍 | 確認が必要 |

---

## 1. データ構造定義 (RuntimeBuildTypes.h)

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 1.1 | `OversamplingResult` 構造体 | ✅ | `RuntimeBuildTypes.h:77` |
| 1.2 | `BuildDiagnostics` 構造体 | ✅ | `RuntimeBuildTypes.h:145`、`boundExcessDb` あり |
| 1.3 | `BuildAnalysis` 拡張 (eqMaxQ, irFreqPeakGainDb) | ✅ | `RuntimeBuildTypes.h:165` |
| 1.4 | `BoundMethod` enum | ✅ | `RuntimeBuildTypes.h:94` (TriangleProduct, Legacy) |
| 1.5 | `EqGainAlgorithm` enum | ✅ | `RuntimeBuildTypes.h:105` (TriangleProductV1, Legacy) |
| 1.6 | `SelectedEstimate` enum | ✅ | `RuntimeBuildTypes.h:113` (Measured, UpperBound) |
| 1.7 | `sealBuildAnalysis()` 更新 | ✅ | `RuntimeBuildTypes.h:185`、全フィールド封印 |
| 1.8 | `verifyBuildBundle()` 実装 | ✅ | `RuntimeBuildTypes.h:234`、Facade パターン |
| 1.9 | `verifyDiagnostics()` 追加 | ✅ | `RuntimeBuildTypes.h:273` |

## 2. OversamplingPolicy

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 2.1 | `OversamplingPolicy` 構造体 | ✅ | `OversamplingPolicy.h:35`、`resolve()`, `maxAllowedFactor()`, `isStructureChangeOversampling()` あり |
| 2.2 | ルックアップ方式 | ✅ | SR→maxFactor テーブル (`maxAllowedFactor()`) |

## 3. EQ 解析 (EQProcessor.Coefficients.cpp)

### 3.1 基本関数

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 3.1a | `biquadResponse()` | ✅ | 標準 `std::complex<double>` を使用、設計通り |
| 3.1b | `isBoostingBand()` | ✅ | Peaking/HighShelf/LowShelf の gain>0.01 のみ |
| 3.1c | `evaluateBandDelta()` | ⚠️ | **定義のみで未使用（デッドコード）**。設計では粗探索・Shelf評価・候補Band判定で再利用するとあるが、実際のコードではインラインで `biquadResponse()` を直接呼んでいる。C4505 警告の原因。 |
| 3.1d | `CandidateBand` 構造体 | ✅ | measured/upperBound 両候補あり |

### 3.2 探索アルゴリズム

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 3.2a | 粗探索600点 | ✅ | `kCoarsePoints=600` |
| 3.2b | 適応サンプリング128点/バンド | ✅ | `kAdaptivePoints=128` |
| 3.2c | union 区間統合 | ✅ | `mergeRanges()` ラムダ関数 |
| 3.2d | 比例配分 | ✅ | 各区間に `128 × length_i / totalLength` |
| 3.2e | measured 用候補 (isBoosting) | ✅ | |
| 3.2f | upperBound 用候補 (max\|Hi-1\|>0.1) | ✅ | |
| 3.2g | Shelf 追加評価点 | ✅ | LowShelf: 20Hz, center, center×2 / HighShelf: center/2, center, Nyquist×0.95 |
| 3.2h | LPF/HPF (Q>0.707) ±1oct | ✅ | |

### 3.3 upperBound 計算

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 3.3a | `log1p()` 使用 (exp回避) | ✅ | `logBound += std::log1p(delta)` |
| 3.3b | `kTwentyOverLog10` 定数 | ✅ | `20.0 / std::log(10.0)` |
| 3.3c | NaN/Inf ガード | ✅ | `!std::isfinite(delta) continue` |
| 3.3d | 微小項切り捨て (delta>1e-6) | ✅ | |
| 3.3e | upperBound は補間しない | ✅ | 評価点最大値を採用 |

### 3.4 放物線補間

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 3.4a | measured のみ補間 | ✅ | |
| 3.4b | 対数周波数軸 + dB空間 | ✅ | |
| 3.4c | Lagrange 二次補間（一般3点） | ✅ | 不等間隔対応 |
| 3.4d | 分母ガード (1e-12) | ✅ | |

### 3.5 totalGain

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 3.5a | クランプ撤廃、Planner側でmax(0,..) | ✅ | |

### 3.6 戻り値 (EQAnalysisResult)

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 3.6a | `EQAnalysisResult` 二層構造 | ✅ | measured/upperBound/measuredRawGainDb/maxActiveQ/algorithm |
| 3.6b | `PeakInfo` (gainDb, freqHz, origin) | ✅ | |
| 3.6c | `SampleOrigin` (Type, bandIndex, sampleIndex) | ✅ | |
| 3.6d | `measuredRawGainDb` (補間前) | ✅ | |

### 3.7 3層分解 (EQResponseSampler/PeakEstimator/UpperBoundEstimator)

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 3.7a | 3層への分割実装 | ❌ | **未実装**。設計 4.1.7 節で「責務過多を避けるため3層に分割」とあるが、実際は `computeEstimatedMaxGainComplex()` 一関数に全ロジックが詰まっている（約550行）。将来の FFT ベース探索への差し替えが困難。 |

## 4. IR ゲイン指標 V2

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 4.1 | `computeScaleFactor` → scaledIR | ✅ | `IRConverter.cpp:342` |
| 4.2 | `estimateMaxFrequencyResponseGain(scaledIR)` | ✅ | 自己完結FFT 実装済み |
| 4.3 | `convertFile()` 内で `irFreqPeakGainDb` 設定 | ✅ | `IRConverter.cpp:361` |
| 4.4 | `getIrFreqPeakGainDb()` | ✅ | `ConvolverProcessor.h` |

## 5. AutoGainPlanner V2

### 5.1 定数

| # | 項目 | 設計値 | 実装値 | 状態 |
|---|------|--------|--------|------|
| 5.1a | kMarginEqFirst | 1.5 | 1.5 | ✅ |
| 5.1b | kMarginConvFirst | 1.0 | 1.0 | ✅ |
| 5.1c | kMarginInterStage | 1.0 | 1.0 | ✅ |
| 5.1d | kSafetyMarginBase | 0.8 | 0.8 | ✅ |
| 5.1e | kSafetyMarginCoeffQ | 0.12 | 0.12 | ✅ |
| 5.1f | kSafetyMarginCoeffGain | 0.04 | 0.04 | ✅ |
| 5.1g | kSafetyMarginMax | 2.5 | 2.5 | ✅ |
| 5.1h | kClampInputMin | -18.0 | -18.0 | ✅ |
| 5.1i | kClampInputMax | 0.0 | 0.0 | ✅ |
| 5.1j | kClampTrimMin | -12.0 | -12.0 | ✅ |
| 5.1k | kClampTrimMax | 0.0 | 0.0 | ✅ |
| 5.1l | kClampMakeupMin | 0.0 | 0.0 | ✅ |
| 5.1m | kClampMakeupMax | 12.0 | 12.0 | ✅ |

### 5.2 構造体

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 5.2a | `PlannerInput` DTO | ✅ | 3フィールド (eqMaxGainDb, eqMaxQ, irFreqPeakGainDb) |
| 5.2b | `EmpiricalSafetyMarginPolicy` | ✅ | `evaluate(eqGainDb, maxQ)` 実装済み、Q<0.707ガードあり |
| 5.2c | `PlanDiagnostics` | ✅ | 全フィールドあり (qMargin, eqBoost, convBoost, clamped, etc.) |

### 5.3 4パターン分岐

| 条件 | Input | Trim | 状態 |
|------|-------|------|------|
| convBypassed (PEQのみ) | -(eqBoost - 1.5) - margin | 0 | ✅ |
| eqBypassed (Convのみ) | -(convBoost - 1.0) | 0 | ✅ |
| ConvThenEQ | -(convBoost - 1.0) - (eqBoost - 1.0) - margin | 0 | ✅ |
| EQThenConv | -(eqBoost - 1.5) - margin | -(convBoost - 1.0) | ✅ |

## 6. Builder 統合 (AudioEngine.RebuildDispatch.cpp)

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 6.1 | `computeEstimatedMaxGainComplex()` 呼び出し | ✅ | `processingRate` のみ渡す |
| 6.2 | Builder collapse: `max(measured, upperBound)` | ✅ | |
| 6.3 | `BuildDiagnostics` への診断値設定 | ✅ | 全フィールド設定 |
| 6.4 | `BuildAnalysis` 更新 (eqMaxQ, irFreqPeakGainDb) | ✅ | |
| 6.5 | `OversamplingResult` で resolvedOsFactor 決定 | ✅ | |

## 7. AutoGainPlanner 統合 (RuntimeBuilder.cpp)

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 7.1 | `PlannerInput` 生成 → `AutoGainPlanner::plan()` | ✅ | |
| 7.2 | ゲイン上書き (dB→線形) | ✅ | |
| 7.3 | `PlanDiagnostics` ログ出力 | ✅ | `AUTO_GAIN_PLAN` / `AUTO_GAIN_CLAMP` |

## 8. DiagEvent

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 8.1 | `DiagEvent::AutoGainClamped` データ構造 | ✅ | `AudioEngine.h:341` |
| 8.2 | `AutoGainClampedData` 構造体 | ✅ | `AudioEngine.h:421` |

## 9. CLI 機能（今回追加）

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 9.1 | `--cli-ir` IR読み込み | ✅ | 200ms遅延読み込み |
| 9.2 | `--cli-rebuild` 強制リビルド | ✅ | 500ms遅延 |
| 9.3 | `--cli-eq-*` フラグ | ✅ | `--cli-eq-gain-db`, `--cli-eq-freq-hz`, `--cli-eq-q` |
| 9.4 | `--cli-log-file` 診断ログ | ✅ | |
| 9.5 | `--cli-exit-ms` 自動終了 | ✅ | |

## 10. 診断ログ出力

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 10.1 | `[AUTO_GAIN_PLAN]` | ✅ | Releaseでも出力 |
| 10.2 | `[AUTO_GAIN_CLAMP]` | ✅ | Releaseでも出力 |
| 10.3 | `[AUTO_GAIN_ANALYSIS]` | ✅ | boundExcessDb含む |
| 10.4 | `[DIAG_IR_FREQ]` | ✅ | irFreqPeakGainDb含む |

---

## ⚠️ 発見された問題点

### P1: `evaluateBandDelta()` デッドコード — **✅ 修正済み**

- **ファイル**: `EQProcessor.Coefficients.cpp:383`（削除）
- **修正**: 未使用の static 関数を削除。C4505 警告も解消。
- **検証**: Debug/Release ビルド成功、C4505 警告消失を確認。

### P2: 3層分割 (EQResponseSampler/PeakEstimator/UpperBoundEstimator) が未実装

- **設計書 4.1.7 節**: 「責務過多を避けるため、`EQProcessor` 内で直接実装せず、3 層に分割して実装する」
- **現状**: `computeEstimatedMaxGainComplex()` 一関数（約550行）に全ロジックが集中。
- **影響**: ユニットテストが困難、FFTベース探索への将来の差し替えが困難、コード理解が難しい。
- **推奨対処**: リファクタリングして3層に分割する。ただし動作に影響はないため優先度は低い。

### P3: `--cli-eq-*` フラグと `--cli-ir` のタイミング競合リスク — **✅ 修正済み**

- `--cli-exit-ms` に最小値を強制: `--cli-ir` または `--cli-rebuild` 使用時は最低 3000ms に調整。
- **修正**: `MainWindow.cpp` — `minExitMs` チェック追加。500ms 指定が 3000ms に調整されることを確認済。

### P4: リサンプリングフォールバック時の周波数分析精度 — **⚠️ 問題なし（確認済み）**

- `estimateMaxFrequencyResponseGain()` はサンプルレートを引数に取らず、生サンプルの FFT 振幅のみを使用。
- 周波数軸のスケーリングにのみサンプルレートが影響するが、振幅値（`irFreqPeakGainDb`）には影響しない。
- **結論**: 実害なし。修正不要。

### P5: `--cli-log-file` のバッファフラッシュ問題 — **✅ 修正済み**

- **修正**: `MainWindow.cpp` — 終了時コールバックで `juce::Logger::setCurrentLogger(nullptr)` を明示的に呼び出し、FileLogger のデストラクタでファイルをフラッシュ。
- ログ出力 `[CLI] Auto-exit flush: shutting down` の出現を確認済。
- **検証**: ログファイルが完全に書き込まれるようになった。

---

## 総合評価

| カテゴリ | 評価 |
|---------|------|
| 設計 vs 実装一致性 | ⭐⭐⭐⭐☆ (90%) — 一部未実装の分離設計あり |
| データ構造完全性 | ✅ 全構造体・enum が設計通り実装 |
| 定数値一致性 | ✅ 全定数が設計値と一致 |
| 配線漏れ | ✅ CLI→Engine→Builder→Planner の全経路が接続済み |
| 新規バグ | なし（既知の制限のみ） |
| 警告 | ⚠️ C4505（未使用関数 evaluateBandDelta） |

### 残作業（設計書 13.1/13.2）

| 項目 | 状態 | 備考 |
|------|------|------|
| 13.1 EmpiricalSafetyMarginPolicy 係数較正 | ⏳ 測定インフラ✅、実測データ待ち | |
| 13.2 upperBound 過大評価の分布測定 | ⏳ 測定インフラ✅、実測データ待ち | |
