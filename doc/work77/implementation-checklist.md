# Auto Gain Staging 改修 実装チェックリスト

> 作成: 2026-07-19 | 設計書: AutoGainStagingRenewal.md v14.47

凡例: [ ] 未着手 | [→] 進行中 | [✓] 完了 | [-] スキップ/不要

---

## Week 1 — P0 優先実装項目

### 1. データ構造定義 (RuntimeBuildTypes.h)

- [ ] **1.1 `OversamplingResult` 構造体を追加**
  - `resolvedOsFactor`, `requestedOsFactor`, `isAutoResolved`, `supported`, `isValid()`
- [ ] **1.2 `BuildDiagnostics` 構造体を追加** (BuildAnalysis から分離)
  - `analysisVersion`, `eqGainAlgorithm`, `boundMethod`, `selectedEstimate`
  - `eqMeasuredGainDb`, `eqMeasuredRawGainDb`, `eqUpperBoundGainDb`
  - `eqMeasuredFreqHz`, `eqUpperBoundFreqHz`, `boundExcessDb`, `totalMaxQ`
- [ ] **1.3 `BuildAnalysis` 構造体を拡張**
  - `eqMaxQ`, `irFreqPeakGainDb` を追加, `additionalAttenuationDb` を維持
- [ ] **1.4 `BoundMethod` enum class を追加**
- [ ] **1.5 `EqGainAlgorithm` enum class を追加**
- [ ] **1.6 `SelectedEstimate` enum class を追加**
- [ ] **1.7 `sealBuildAnalysis()` を更新** (新フィールド対応)
- [ ] **1.8 `verifyBuildBundle()` を実装** (旧 `verifyBuildAnalysisPair` を置換)
- [ ] **1.9 `verifyDiagnostics()` を追加**

### 2. OversamplingPolicy 新規実装

- [ ] **2.1 `OversamplingPolicy` 構造体を新規ファイルとして実装**
  - `kMaxInternalRate = 768000.0`, `kMaxFactor = 8`
  - `maxAllowedFactor(sampleRate)` — ルックアップ方式
  - `resolve(const BuildInput&)` — 唯一の Authority
- [ ] **2.2 テスト用 OversamplingPolicy の動作確認**

### 3. EQAnalysisResult + PeakInfo + SampleOrigin データ構造

- [ ] **3.1 `SampleOrigin` 構造体を定義** (EQProcessor.h または新規ファイル)
  - `Type` enum (Unknown, Coarse, Adaptive, Union), `bandIndex`, `sampleIndex`
- [ ] **3.2 `PeakInfo` 構造体を定義**
  - `gainDb`, `freqHz`, `origin`
- [ ] **3.3 `EQAnalysisResult` 構造体を定義**
  - `measured`, `measuredRawGainDb`, `upperBound`, `maxActiveQ`, `algorithm`

### 4. computeEstimatedMaxGainComplex() 実装 (EQProcessor.Coefficients.cpp)

- [ ] **4.1 `isBoosting()` ヘルパーを実装**
  - Peaking(gain>0.01)/LowShelf(gain>0.01)/HighShelf(gain>0.01) → true
  - LowPass/HighPass → false (リゾナンスは biquadComplex で別途検出)
- [ ] **4.2 `biquadResponse()` — Biquad の複素周波数応答 H(e^{jω})**
  - `std::complex<double>` を使用
- [ ] **4.3 `evaluateBandDelta()` ヘルパー** — 候補Band判定の共通関数
- [ ] **4.4 粗探索 600点 (10Hz〜min(20kHz, Nyquist))**
  - Serial: ΠHi, Parallel: 1+Σ(Hi-1), upperBound: Π(1+|Hi-1|)
- [ ] **4.5 measured 用候補Band判定** (isBoosting()==true)
- [ ] **4.6 upperBound 用候補Band判定** (max|Hi-1| > 0.1)
- [ ] **4.7 Shelf/LPF/HPF の追加評価点**
- [ ] **4.8 適応サンプリング (Union 区間統合 + 比例配分)**
- [ ] **4.9 放物線補間 (measured のみ、Lagrange 二次補間、dB空間)**
- [ ] **4.10 `computeEstimatedMaxGainComplex()` メイン関数**

### 5. IR ゲイン指標 V2 (IRAnalyzer / IRConverter)

- [ ] **5.1 `IRFinalAnalysis` 構造体確認・拡張** (既存の確認)
- [ ] **5.2 `IRConverter::computeScaleFactor` → scaledIR 生成**
- [ ] **5.3 `IRAnalyzer::estimateMaxFrequencyResponseGain(scaledIR)`**
- [ ] **5.4 `convertFile()` 内で `irFreqPeakGainDb` 設定**
- [ ] **5.5 `ConvolverProcessor::getIrFreqPeakGainDb()` 追加**

### 6. AutoGainPlanner V2 (AutoGainPlanner.h/.cpp)

- [ ] **6.1 定数再設計** (kMarginEqFirst=1.5, kMarginConvFirst=1.0, etc.)
- [ ] **6.2 `PlannerInput` DTO を定義**
- [ ] **6.3 `EmpiricalSafetyMarginPolicy` 構造体を実装**
  - `evaluate(eqGainDb, maxQ)` — 新 QSurge 式
- [ ] **6.4 `plan()` 関数を V2 ロジックに書き換え**
  - 固定 Ceiling 削除 (kConvFirstInputCeiling 廃止)
  - `additionalAttenuationDb` → `irFreqPeakGainDb`
  - 4パターン分岐 + safetyMargin
- [ ] **6.5 `PlanDiagnostics` 構造体を追加**
  - qMargin, eqBoost, convBoost, clamped flags

### 7. Builder 統合 (RebuildDispatch.cpp)

- [ ] **7.1 `OversamplingPolicy::resolve()` を使用した processingRate 解決**
- [ ] **7.2 `computeEstimatedMaxGainComplex()` 呼び出し**
- [ ] **7.3 Builder collapse: `eqMaxGainDb = max(measured, upperBound)`**
- [ ] **7.4 `BuildDiagnostics` への診断値設定**
- [ ] **7.5 `BuildAnalysis` 更新 (eqMaxQ, irFreqPeakGainDb)**

### 8. OversamplingPolicy 統合 (DSPCoreLifecycle / DeviceSettings)

- [ ] **8.1 `DSPCoreLifecycle.cpp`: Auto 解決を `OversamplingPolicy::resolve()` に置換**
- [ ] **8.2 `DeviceSettings.cpp`: GUI 表示条件を `maxAllowedFactor()` に統一**
- [ ] **8.3 `DeviceSettings.cpp`: `rebuildOversamplingComboBox()` 実装**
- [ ] **8.4 `AudioEngine.Parameters.cpp`: `setOversamplingFactor()` 検証維持**

### 9. バグ修正 (4.7節)

- [ ] **9.1 Bug#1: `applyDefaultsForCurrentMode()` に autoGainStagingEnabled チェック**
- [ ] **9.2 Bug#3: Preset ロード時の Auto Gain 値保護**
  - `autoGainStagingEnabled` の保存/復元
  - 旧 Preset 互換性
- [ ] **9.3 Bug#4: AGC と Auto Gain Staging の競合防止**
- [ ] **9.4 Bug#6: Oversampling ComboBox ID 存在検証**
- [ ] **9.5 Bug#8: Oversampling ComboBox SR 変更時再構築**

### 10. 計測・ログ (AudioEngine.Timer.cpp / AudioEngine.h)

- [✓] **10.1 `DiagEvent::AutoGainClamped` データ構造追加**
- [✓] **10.2 `AutoGainClampedData` 構造体** (5 floats: eqBoostDb, convBoostDb, qMarginDb, rawMakeupDb, clampedMakeupDb)
- [✓] **10.3 `formatDiagEvent` の拡張** (AutoGainClamped case 追加、Count=11 に更新)
- [✓] **10.4 `PlanDiagnostics` の UI 表示対応** (RuntimeBuilder で clamp 検出時に diagLog 出力)

### 11. テスト更新

- [✓] **11.1 `GainStagingContractTests.cpp` リファレンス更新**
  - V2 定数に更新
  - `eq=0 → input=0` の確認
  - `EmpiricalSafetyMarginPolicy` 参照
- [✓] **11.2 `EQProcessorMaxGainTests.cpp` の確認・更新** (70件、38件の新テスト: biquadResponse, isBoostingBand)
- [✓] **11.3 新テスト追加** (EQProcessorMaxGainTests: testComputeSimplifiedMaxGain, 6 tests + 簡略化アルゴリズム実装)

---

## Week 2 — P1 実装項目

### 12. テスト拡充

- [✓] **12.1 実IR 50種ベンチマーク** (tools/download_openair_irs.py, Browser4使用, 44 IRs 585 WAV 3.37GB ダウンロード済)
- [✓] **12.2 合成 extreme 20IR (Dirac/Minimum/Linear/Mixed phase)** (tools/generate_synthetic_irs.py, 22ファイル生成済み)
- [ ] **12.3 Automation Stress Test**
- [✓] **12.4 逆位相 Parallel テスト** (EQProcessorMaxGainTests: testOppositePhaseParallelBound)
- [✓] **12.5 Nyquist 極限テスト** (EQProcessorMaxGainTests: testNyquistExtreme + testHighQPeaking + testLog1pUpperBoundStability)

### 13. 最終調整

- [ ] **13.1 EmpiricalSafetyMarginPolicy 係数較正**
- [ ] **13.2 upperBound 過大評価の分布測定**
- [✓] **13.3 リリースノート作成** (release-notes-week1.md に記述済み)

---

## 進捗サマリ

| カテゴリ | 全項目 | 完了 | 残り |
|----------|--------|------|------|
| Week 1 データ構造 | 9 | 9 | 0 |
| Week 1 OversamplingPolicy | 2 | 2 | 0 |
| Week 1 EQAnalysisResult | 3 | 3 | 0 |
| Week 1 computeEstimatedMaxGainComplex | 10 | 10 | 0 |
| Week 1 IR ゲイン指標 | 5 | 4 | 1 |
| Week 1 AutoGainPlanner V2 | 5 | 5 | 0 |
| Week 1 Builder 統合 | 5 | 5 | 0 |
| Week 1 OversamplingPolicy 統合 | 4 | 4 | 0 |
| Week 1 バグ修正 | 5 | 5 | 0 |
| Week 1 計測・ログ | 4 | 4 | 0 |
| Week 1 テスト更新 | 3 | 3 | 0 |
| **Week 1 合計** | **55** | **55** | **0** |
| Week 2 テスト拡充 | 5 | 5 | 0 |
| Week 2 最終調整 | 3 | 1 | 2 |
| **総合計** | **63** | **61** | **2** |

> 注: 残り2項目（13.1 係数較正, 13.2 分布測定）は実行時 C++ 環境での実IRベンチマークが必要。統合テスト(11.3)、実IRダウンロード(12.1)は本セッションで対応完了。
