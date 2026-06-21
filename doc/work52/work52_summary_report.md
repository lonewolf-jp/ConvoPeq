# work52 総合調査報告 — 低音入力時の「ジジジジ」ノイズ

- **作成日**: 2026-06-21
- **最終更新**: 2026-06-21
- **関連文書**:
  - `doc/work52/repair_plan.md` — P1〜P6 改修計画（SVF/SoftClip/AGC）
  - `doc/work52/repair_plan_bug2.md` — P7 改修計画（LatticeNoiseShaper advanceState）
  - `doc/work52/bug2_review.md` — P7 検証レビュー
  - `doc/work52/bug_review3.md` — Conv→Peq 限定ノイズ検証
  - `doc/work52/bug_review3_validation_report_v*.md` — 検証レポート v1〜v8
  - `doc/work52/unresolved_items_investigation.md` — 未確定事項調査
  - `doc/work52/repair_plan_bug2_investigation.md` — P7未確定事項調査
  - `doc/gain_staging_analysis_2026-06-21.md` — ゲインステージング解析

---

## 1. 改修実装済み項目

### P1: SVF状態変数サチュレーション除去 ✅
- **ファイル**: `src/eqprocessor/EQProcessor.Processing.cpp`
- **内容**: `processBand()` / `processBandStereo()` の saturation 適用を `ic1eq/ic2eq` → `output` 計算後に移動
- **新規関数**: `fastTanhScalarOutput()` / `fastTanhV128Output()`（クリップ閾値 4.5）
- **旧関数削除**: `fastTanhScalar` / `fastTanhV128`（未使用のため削除）
- **選択肢**: 出力段サチュレーション（Choice A）

### P2: SoftClip prevScalar不整合修正 ✅
- **ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
- **内容**: スカラーフォールバックで `prevScalar` に処理前の生入力値を保存
- **ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`
- **内容**: `prevSample` に平均化前の生入力値を保存

### P3: SoftClip midVec事前平均化削除 ✅
- **ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
- **内容**: `softClipBlockAVX2()` の midVec ブロックを完全削除

### P6: AGC時定数変更 ✅
- **ファイル**: `src/eqprocessor/EQProcessor.h`
- **内容**: `AGC_ATTACK_TIME_SEC`: 0.1s → 0.2s

### P7: LatticeNoiseShaper advanceState 状態更新バグ修正 ✅
- **ファイル**: `src/LatticeNoiseShaper.h`
- **内容**: `state[i] = clamp(prev_backward, ...)` → `state[i] = clamp(nextBackward, ...)`
- **根拠**: ARM CMSIS-DSP, MATLAB latcfilt, Proakis & Manolakis の格子フィルタ理論

---

## 2. 未実装・調査中項目

### Conv→Peq 限定「ジジジジ」ノイズ

#### バグ内容
- **条件**: Conv→Peq モード + 任意の NoiseShaper（または NS 完全OFF）+ 低音
- **症状**: 「ボーン」で「ジー」、「ボンボンボン」で「ジジジジ」
- **非発生**: PEQ-only モードでは一切発生しない

#### テスト結果一覧

| # | テスト | 結果 | 判定 |
|:-:|:------|:----:|:----|
| 1 | PEQ-only | 正常 | — |
| 2 | Conv→Peq | 発生 | 必須条件 |
| 3 | Adaptive9th → Fixed4Tap | 発生 | NS固有ではない |
| 4 | NS完全OFF | 発生 | **NS系統は全滅** |
| 5 | SoftClip OFF | 発生 | SoftClip除外 |
| 6 | Saturation=0 | 発生 | SoftClip内部非線形も除外 |
| 7 | OS=1x | 発生 | Downsampler除外 |
| 8 | PEQ全0dB | 発生 | EQ相互作用除外 |
| 9 | Output Makeup 0dB/+6dB | 発生 | **ゲインステージング除外** |
| 10 | IRに20Hz HPF | **やや改善** | 超低域が部分的に関与 |
| 11 | IR長86ms | 発生 | 長テール不要 |
| 12 | ボーン（単発） | 「ジー」 | 状態励振型 |
| 13 | ボンボンボン（連打） | 「ジジジジ」 | 状態蓄積型 |

#### 除外された仮説
- Adaptive9th 固有バグ（P7 advanceState）
- NoiseShaper 系統（全方式 + 完全OFF）
- SoftClip
- Oversampling Downsampler
- PEQ設定（EQブースト）
- ゲインステージング（Output Makeup 0dB/+6dB）
- post-NS ハードクランプ
- プリリンギング
- IRの長い残響テール

#### 消去法で残った原因領域
1. Convolver（IRとの畳み込み）そのもの
2. パーティション FFT 処理（Overlap-Save境界）
3. DC Blocker（カットオフ1Hz）では除去できない5〜20Hz超低域成分
4. IR由来の特定周波数共振（20Hz以上、HPFで除去不可）

#### 未解決
ConvoPeq内部のDC Blockerは全段1Hzカットオフのため5〜20Hzの超低域が通過する。20Hz HPFで「やや改善」したことから超低域の関与は確認されたが、完全除去には至らず、複合原因の可能性が高い。Convolver出力の直接確認（ファイル書き出し等）が次の診断ステップとして有効。

---

## 3. 重要発見: ゲインステージング解析

`doc/gain_staging_analysis_2026-06-21.md` より:

- Conv→Peq のデフォルト: inputHeadroom=-6dB, outputMakeup=+12dB
- IR Scale Factor: `makeup × 0.5012`（IRエネルギー正規化 + 安全マージン-6dB）
- 理論上の正味ゲイン: inputHeadroom(-6dB) × IR scaleFactor(-6dB) × outputMakeup(+12dB) = 0dB（基準）
- Output Makeup 0dBテストでもノイズ発生したため、ゲイン差は主原因ではない

---

## 4. 主要ファイル一覧

| ファイル | 役割 |
|:---------|:------|
| `src/LatticeNoiseShaper.h` | P7改修: advanceState() バグ修正 |
| `src/eqprocessor/EQProcessor.Processing.cpp` | P1改修: processBand/processBandStereo saturation移動 |
| `src/eqprocessor/EQProcessor.h` | P6改修: AGC_ATTACK_TIME_SEC 0.1→0.2 |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | P2/P3改修: softClipBlockAVX2 midVec削除・prevScalar修正 |
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | P2改修: prevSample修正 |
| `src/audioengine/AudioEngine.Parameters.cpp` | Conv→Peq ゲイン設定（applyDefaultsForCurrentMode） |
| `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` | processInput/processOutput, DC Blocker, kOutputHeadroom |
| `src/convolver/ConvolverProcessor.Runtime.cpp` | Convolver 実行時処理 |
| `src/MKLNonUniformConvolver.cpp` | NUC 本体（ringWrite/ringRead/FFT処理） |
| `src/UltraHighRateDCBlocker.h` | 全段DC Blocker（カットオフ1Hz） |
| `src/IRConverter.cpp` | IR scaleFactor計算 |
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | IR読み込みパイプライン |
| `src/DeviceSettings.cpp` | UIゲイン表示 |
| `src/CustomInputOversampler.cpp` | OS Up/Downsampler |
