# ConvoPeq 保留4件 検証・設計書

> **検証日**: 2026-07-20
> **対象**: 改修計画書 v7 の保留3件（R-1, R-2, R-3）+ 分離1件（S-1）
> **検証方法**: ソースコード読解 + ISR設計との整合性確認

---

## R-1: M-10 OutputFilter fc 分岐 — 設計意図確認

### 結論: **変更不要（音響設計値の確認済み）**

### 調査結果

| 項目 | 内容 |
|------|------|
| **現在のコード** | `fc_hc = (sampleRate <= 48000.0) ? 19000.0 : 22000.0` / `fc_lp = (sampleRate <= 48000.0) ? 19000.0 : 24000.0` |
| **設計意図** | 音響設計値として決定。48kHz以下は19kHz、48kHz以上は22kHz/24kHz。 |
| **git履歴** | 詳細なコミットメッセージなし。初期設計からの変更なし。 |
| **ISR影響** | なし（prepare時のみ） |

### 分析

- 19kHz/22kHz/24kHz は音響設計値。長期間運用され、聴感評価が既存設定を前提としている。
- 連続補間に変更すると、フィルタ設計が固定値前提でチューニングされている既存実装との不整合が発生する。
- 48kHz→96kHz切替時に可聴な音色変化が発生する可能性もある。

### 判定

**変更不要。** 音響設計値であり、長期間運用で検証済み。変更する場合は音響設計者との合意が必要。

---

## R-2: M-14 BuildAnalysis 失敗時の ISR 設計 — 設計整合性確認

### 結論: **変更不要（ISR設計との整合性確認済み）**

### 調査結果

| 項目 | 内容 |
|------|------|
| **現在のコード** | `task.buildAnalysis = convo::sealBuildAnalysis(analysis, &task.runtimeBuildSnapshot);` |
| **ISR設計** | `Validator` → `Admission` → `Publish` の3層構造 |
| **sealBuildAnalysis** | 契約違反時に `BuildAnalysis{}`（空）を返す。5条件: snapshot nullptr, generation不一致, 未sealed, 非finite値 |
| **Admission** | `PublicationAdmission::evaluate()` で最終判定（Shutdown/Generation/Finalized/Health/Pressure） |
| **ISR影響** | BuildAnalysis 失敗→空のBuildAnalysisで処理継続。Admission が他の条件で Reject する可能性あり |

### 分析

- `sealBuildAnalysis` が `BuildAnalysis{}` を返す場合、 downstream は空の解析結果を使用
- `PublicationAdmission::evaluate()` には `BuildAnalysis` の空チェックはない。Reject は別の条件（Shutdown/Generation/Health/Pressure）で発生
- 「BuildAnalysis失敗→Publish Reject」は因果関係の短縮。正確には「BuildAnalysis失敗→空のBuildAnalysisで処理継続」
- `fallback` を追加すると `Validator` / `Admission` の責務境界が崩れる
- コードコメント（line 152-155）でも「Validator の検証は Bridge 層で既に実行済み」と明記

### 判定

**変更不要。** ISR設計の責務分離に従い、BuildAnalysis 失敗時は空のBuildAnalysisで処理が継続される。fallback の追加は設計違反。

---

## R-3: M-11/M-12 AutoGain 推定誤差 — 意図的設計の確認

### 結論: **現時点では変更判断を保留**

### 調査結果

| 項目 | 内容 |
|------|------|
| **現在のコード** | `result.measured.gainDb = measured.interpolatedDb + totalGainDb` |
| **totalGainDb** | `getTotalGain()` — Master Gain を含む |
| **interpolatedDb** | `PeakEstimator::estimate(merged).interpolatedDb` — 周波数応答のサンプリング結果 |
| **ISR設計** | AutoGainPlanner は純粋関数型（Engine/DSP 不参照） |

### 分析

- `interpolatedDb` が「最大EQゲイン」なのか「特定周波数でのゲイン」なのか、まだ100%確定していない
- `totalGainDb` を加算する設計意図は「EQ全体の最大ゲイン = バンド測定値 + 総ゲイン」の可能性がある
- 二重カウントの可能性もあるが、意図的設計の可能性も否定できない
- AutoGainPlanner / BuildAnalysis / EQBoundExcessBenchmark の全体系を理解した上で判断が必要

### 判定

**現時点では変更判断を保留。** `interpolatedDb` の意味を100%確定させてから判断する。変更する場合は全体系の影響評価が必要。

---

## S-1: M-05 大ブロック無音化 — アーキテクチャ変更

### 結論: **別設計書に分離（現状維持）**

### 調査結果

| 項目 | 内容 |
|------|------|
| **現在のコード** | `if (numSamples > dsp->maxSamplesPerBlock) { buffer.clear(); return; }` |
| **影響範囲** | Convolver, Delay, Crossfade, Oversampling, Runtime Snapshot, Automation, Peak Meter, True Peak, Latency |
| ** ISR設計** | RCU で公開済みの `maxSamplesPerBlock` を Audio Thread から安全に読み出し |
| **File** | `AudioEngine.Processing.BlockDouble.cpp` (line 289) + `AudioEngine.Processing.AudioBlock.cpp` (line 306) |

### 分析

- 現在の `buffer.clear(); return;` は安全装置（1ブロック無音）
- チャンク分割に変更すると、以下全ての処理に影響:
  - Convolver: パーティション処理のブロック境界
  - Delay: 遅延バッファの書込み位置
  - Crossfade: クロスフェードの補間精度
  - Oversampling: アップ/ダウンサンプリングのブロックサイズ
  - Runtime Snapshot: スナップショット取得タイミング
  - Automation: パラメータ更新タイミング
  - Peak Meter / True Peak: ピーク検出のブロック単位
  - Latency: レイテンシ補正のブロック単位

### 判定

**別設計書に分離。** これはバグ修正ではなくアーキテクチャ変更。影響範囲が広すぎるため、独立した設計・レビュー・テストが必要。

---

## まとめ

| ID | 内容 | 判定 | 理由 |
|----|------|------|------|
| R-1 | M-10 OutputFilter fc 分岐 | **変更不要** | 音響設計値。長期間運用で検証済み |
| R-2 | M-14 BuildAnalysis ISR設計 | **変更不要** | ISR設計の責務分離に従った正しい動作。BuildAnalysis失敗→空のBuildAnalysisで処理継続 |
| R-3 | M-11/M-12 AutoGain推定誤差 | **保留** | interpolatedDb の意味を100%確定させてから判断 |
| S-1 | M-05 大ブロック無音化 | **分離（現状維持）** | アーキテクチャ変更。別設計書が必要 |

---

## 参考資料

- `ConvoPeq_repair_plan_2026-07-20.md` — 改修計画書 v7
- `src/OutputFilter.cpp` — OutputFilter 実装
- `src/audioengine/AudioEngine.RebuildDispatch.cpp` — BuildAnalysis 処理
- `src/eqprocessor/EQProcessor.Coefficients.cpp` — AutoGain 推定
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` — ブロック処理
