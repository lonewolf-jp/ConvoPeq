# 自動ゲインステージング 実装チェックリスト

> 設計書: `gain_design_spec.md` v14.0 Final
> 作成日: 2026-07-15
> 全14回レビュー・累計72件解決

---

## Phase 1: FFT Infrastructure (IRAnalyzer分離)

- [x] `src/IRAnalyzer.h` — 新規作成（Tukey α=0.5, kMaxAnalysisWindow=65536, estimateMaxFrequencyResponseGain）
- [x] `src/IRAnalyzer.cpp` — 新規作成（FFT解析＋ガウス補間＋コヒーレントゲイン補正）
- [ ] `src/IRConverter.h` — `ScaleFactorResult` に `additionalAttenuationDb` 追加
- [ ] `src/IRConverter.cpp` — `computeScaleFactor` 3段階分割（computeEnergyScale/analyzeIR/applyClampProtection）
- [ ] `src/IRConverter.cpp` — `estimateMaxFrequencyResponseGain` を IRAnalyzer 委譲に変更
- [ ] CMakeLists.txt — `IRAnalyzer.cpp` 追加

## Phase 2: State Management Extension

- [ ] `src/PreparedIRState.h` — `additionalAttenuationDb` フィールド追加（float, 正値）
- [ ] `src/ConvolverProcessor.h` — `IRState` に `additionalAttenuationDb` 追加
- [ ] `src/ConvolverProcessor.h` — `getIrAdditionalAttenuationDb()` 追加
- [ ] `src/ConvolverProcessor.cpp` — `applyComputedIR` で `additionalAttenuationDb` 伝搬
- [ ] `src/audioengine/AudioEngine.h` — `autoGainStagingEnabled` atomic 追加
- [ ] `src/audioengine/AudioEngine.h` — `setAutoGainStagingEnabled()` / `isAutoGainStagingEnabled()` 宣言

## Phase 3: EQ Maximum Gain Estimation

- [ ] `src/eqprocessor/EQProcessor.h` — `computeEstimatedMaxGainDb()` 宣言追加
- [ ] `src/eqprocessor/EQProcessor.Coefficients.cpp` — `computeEstimatedMaxGainDb()` 実装
  - 第1段: 粗探索 300点（対数分布）
  - 第2段: Band適応サンプリング（Q依存帯域幅、gain>0限定）

## Phase 4: IR Conversion Extension

- [ ] `src/IRConverter.h` — `computeEnergyScale` / `analyzeIR` / `applyClampProtection` 宣言
- [ ] `src/IRConverter.cpp` — 上記3関数実装＋`computeScaleFactor` オーケストレーター化

## Phase 5: RuntimeBuilder Integration

- [ ] `src/audioengine/RuntimeBuildTypes.h` — `BuildAnalysis` 構造体追加
- [ ] `src/audioengine/RuntimeBuildTypes.h` — `RuntimeBuildSnapshot` から解析値フィールド削除
- [ ] `src/audioengine/RuntimeBuilder.h` — `ProcessingPart` に `autoGainStagingEnabled` 追加
- [ ] `src/audioengine/RuntimeBuilder.h` — `AnalysisPart` 構造体追加
- [ ] `src/audioengine/AutoGainPlanner.h` — 新規作成（constexpr定数＋enum class ProcessingOrder）
- [ ] `src/audioengine/AutoGainPlanner.cpp` — 新規作成（plan() + estimateQSafetyMargin()）
- [ ] `src/audioengine/RuntimeBuilder.cpp` — `buildRuntimePublishWorld` で AutoGainPlanner 呼出（単一代入）
- [ ] `src/audioengine/AudioEngine.RebuildDispatch.cpp` — `captureBuildAnalysis()` 追加
- [ ] `src/audioengine/AudioEngine.RebuildDispatch.cpp` — `captureRuntimeBuildSnapshot` 拡張

## Phase 6: Runtime Mode Transition Safety (変更なし)

## Phase 7: UI Integration

- [ ] `src/DeviceSettings.h` — `autoGainToggle` 宣言＋`gainDisplaySignature`
- [ ] `src/DeviceSettings.cpp` — トグル追加＋レイアウト変更
- [ ] `src/DeviceSettings.cpp` — onEditorHide チェーン（v9.0 設計）
- [ ] `src/DeviceSettings.cpp` — `saveSettings` に `autoGainStagingEnabled` 永続化
- [ ] `src/DeviceSettings.cpp` — `loadSettings` に `autoGainStagingEnabled` 復元
- [ ] `src/DeviceSettings.cpp` — `updateGainStagingDisplay()` 拡張
- [ ] `src/audioengine/AudioEngine.Parameters.cpp` — `setProcessingOrder` 修正（sendChangeMessage追加）
- [ ] `src/audioengine/AudioEngine.Parameters.cpp` — `setAutoGainStagingEnabled()` 実装

## Phase 8: CMakeLists

- [ ] CMakeLists.txt — `AutoGainPlanner` テスト追加
- [ ] CMakeLists.txt — `EQEstimatedMaxGainDbTests` 追加
# 自動ゲインステージング 実装チェックリスト（更新: 2026-07-15）

**進捗: 15/22 タスク完了 (68%)**

- [x] IRAnalyzer.h/.cpp — 新規（FFT＋ガウス補間＋窓補正）
- [x] IRConverter.h — ScaleFactorResult + estimateMaxFrequencyResponseGain 宣言
- [x] IRConverter.cpp — 3段階分割（computeEnergyScale/analyzeIR/applyClampProtection）＋IRAnalyzer委譲＋currentIr比較保護
- [x] PreparedIRState.h — additionalAttenuationDb（move ctor/assign含む）
- [x] ConvolverProcessor.h — IRState + getIrAdditionalAttenuationDb()
- [x] AudioEngine.h — autoGainStagingEnabled atomic + setter/getter
- [x] EQProcessor.h — computeEstimatedMaxGainDb() 宣言
- [x] EQProcessor.Coefficients.cpp — 実装（粗探索300点＋Band適応サンプリングスケルトン）
- [x] RuntimeBuildTypes.h — BuildAnalysis 構造体
- [x] RuntimeBuilder.h — ProcessingPart (autoGainStagingEnabled) + AnalysisPart
- [x] AutoGainPlanner.h — 新規（constexpr + plan + estimateQSafetyMargin）
- [x] AutoGainPlanner.cpp — 新規（4パターン＋クランプ＋Q Surgeヒューリスティック）

**残り**: ConvolverProcessor.cpp伝搬・Builder統合・RebuildDispatch拡張・UI全般・CMake
