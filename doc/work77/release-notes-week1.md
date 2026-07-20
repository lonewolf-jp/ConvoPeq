# Auto Gain Staging 改修 — Week 1 リリースノート

> リリース日: 2026-07-19 | 設計書: v14.47

## 改修の背景

Auto Gain Staging の旧実装には以下の問題があった:

| ID | 問題 | 影響度 |
|----|------|--------|
| E-1/E-3 | LPF/HPF をブーストとしてカウント、300点探索のみ、Parallel で位相無視 | P0 |
| I-1 | `additionalAttenuationDb` は残余ブーストを表さない | P0 |
| P-1/P-2 | QSurge 常時6dB張り付き、Conv→EQで-6dB強制クランプ | P0 |
| P-3/P-4 | テスト条件不一致、makeup クランプで net 0dB 崩れ | P1-P2 |

## 主な変更点

### 1. EQ 最大ゲイン推定 V2 — `EQProcessor.Coefficients.cpp`

- **旧**: 300点対数探索 + Serial積近似（Parallel で位相無視）
- **新**: 600点粗探索 + 適応サンプリング128点 + 複素応答（std::complex） + 放物線補間
- Parallel: `|1 + Σ(Hi - 1)|`（複素ベクトルの振幅）
- Serial: `Π|Hi|`（厳密な振幅積）
- upperBound: `Π(1 + |Hi-1|)` — dB空間で計算（exp 不使用、Inf 発散防止）
- ternary search 廃止 → Lagrange 二次補間（対数周波数軸、dB空間）

### 2. IR ゲイン指標 V2 — `IRConverter.cpp`

- `irFreqPeakGainDb` を新規追加（IR の周波数ピークゲインを IRAnalyzer で推定）
- `additionalAttenuationDb` は互換性維持

### 3. AutoGainPlanner V2 — 完全書き換え

- マージン定数再設計: 3.0→1.5, 1.5→1.0, 2.0→1.0
- 固定 Ceiling（kConvFirstInputCeiling=-6dB）**廃止**
- `PlannerInput` DTO 導入（ISR 思想: Planner は解析アルゴリズムを知らない）
- `EmpiricalSafetyMarginPolicy` 新設（旧 QSurge → 係数再設計: 1.5→0.8, max 6.0→2.5）
- `PlanDiagnostics` 出力対応（clamp 検出・ログ出力）
- Input クランプ範囲拡大: -12dB → -18dB

### 4. OversamplingPolicy — 新規設計

- ルックアップ方式: SR→maxFactor テーブル（44.1k-96k=x8, 176.4k-192k=x4, ...）
- Authority Singularization: `OversamplingPolicy::resolve()` が唯一の決定権限
- 768kHz 超の入力は supported=false（Publish スキップ、無音防止）
- GUI/DeviceSettings 統合: `maxAllowedFactor()` で ComboBox 表示制御

### 5. データ構造追加

| 構造体 | 目的 |
|--------|------|
| `OversamplingResult` | OS 倍率解決結果（resolvedOsFactor, supported） |
| `BuildDiagnostics` | 診断情報（BuildAnalysis から分離） |
| `BoundMethod` enum | upperBound 算出方式の識別子 |
| `EqGainAlgorithm` enum | 解析アルゴリズム識別子 |
| `SelectedEstimate` enum | Builder collapse 採用値の識別子 |
| `AnalysisVersionPolicy` | 解析バージョン管理（kCurrent=2） |
| `EqGainAlgorithm` | computeEstimatedMaxGainComplex の algorithm タグ |
| `EQAnalysisResult / PeakInfo / SampleOrigin` | EQ 解析の二層構造戻り値 |

### 6. バグ修正 (4件)

| Bug | 対象 | 内容 |
|-----|------|------|
| #1 | `applyDefaultsForCurrentMode()` | Auto Gain 有効時のデフォルト上書き防止 |
| #3 | `requestLoadState()` | Preset ロード時の Auto Gain 値保護（旧 Preset 互換性含む） |
| #4 | `setAutoGainStagingEnabled()` | Auto Gain 有効時は EQ AGC を無効化 |
| #6/#8 | `DeviceSettings.cpp` | Oversampling ComboBox ID 存在検証 + SR 変更時再構築 |

## 性能への影響

- **ラウドネス**: 固定 Ceiling 廃止により Conv→EQ 時 +2〜+6dB 上昇の可能性。リリースノートに明記
- **CPU 負荷**: computeEstimatedMaxGainComplex は最大 600-1200 評価点（20Band 時）。実測値は Week2 で測定予定
- **メモリ**: 新規データ構造はすべて POD/triavially copyable、追加メモリは微量

## テスト状況

| テスト | 件数 | 状態 |
|--------|------|------|
| GainStagingContractTests V2 | 2143 | ✅ 全パス |
| EQProcessorMaxGainTests | 156 | ✅ 全パス（biquadResponse, isBoostingBand, Nyquist極限含む） |
| BuildInputSemanticContract | — | ✅ パス |
| CTest | 18/19 | ✅ 1件は事前既存の失敗 |

## 既知の制限・残課題

| 項目 | 優先度 | 備考 |
|------|--------|------|
| empiricalSafetyMarginPolicy 係数較正 | P1 | Week2 の実IRベンチマークで確定予定 |
| upperBound 過大評価分布測定 | P1 | Week2 で測定 |
| 実IR 50種ベンチマーク | P1 | OpenAIR / sampledata から収集予定 |
| 統合テスト（computeEstimatedMaxGainComplex） | P2 | EQProcessor 結合テスト。Week2 以降 |
| `PlanDiagnostics` の UI 表示 | P3 | 現在は diagLog 出力。本格的な UI 連携は将来 |
