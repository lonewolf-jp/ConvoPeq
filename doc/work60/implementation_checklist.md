# Numeric-Only DiagEvent 実装チェックリスト（完了）

**作成日**: 2026-07-02 | **最終更新**: 2026-07-02
**対象設計書**: `診断ログRT違反_numeric-only設計レポート_2026-06-30.md`
**凡例**: ✅ = 完了 | 🔄 = 作業中 | ❌ = 未着手

---

## Phase 1: AudioEngine.h — 型定義追加 ✅

| # | 項目 | 状態 | 変更内容 |
|---|------|------|---------|
| 1a | `DiagCategory` enum class | ✅ | `enum class DiagCategory : uint8_t` 追加（7カテゴリ） |
| 1b | サブ構造体（CpuMigData他7種） | ✅ | CpuMigData, CallbackSequenceData, DspTimingData, CallbackStageData, EqTimeData, ConvTimeData, StereoConvTimeData + PublicationDirection enum |
| 1c | `DiagEvent` 主構造体 | ✅ | category + eventIndex + union{7種}、6個のstatic_assert（trivially_copyable, standard_layout, trivial, trivially_destructible, alignof, offsetof）|
| 1d | `DiagRuntimeLimits` | ✅ | BufferCapacity=512, MaxDrainPerTick=64 |
| 1e | `DiagPerTickCounter` | ✅ | `struct alignas(64)` per-tick counter |
| 1f | RTAuxMutable内カウンタ | ✅ | diagTickPushed/Popped/Dropped + diagTotalPushed/Popped + setEqDiagBuffer/setConvDiagBuffer宣言 |
| 1g | diagBuffer メンバ | ✅ | `LockFreeRingBuffer<DiagEvent, 512>` を AudioEngine クラスに追加（xRunBuffer隣）|

## Phase 2: AudioBlock.cpp — CPU_MIG + CB_SEQ ✅

| # | 項目 | 状態 | 変更内容 |
|---|------|------|---------|
| 2a | CPU_MIG → DiagEvent push | ✅ | `juce::Logger::writeToLog` → `DiagEvent{}` + `diagBuffer.push()` + counter更新 |
| 2b | CB_SEQ → DiagEvent push | ✅ | 同上 |

## Phase 3: AudioBlock.cpp — DSP_TIMING + CALLBACK_STAGE ✅

| # | 項目 | 状態 | 変更内容 |
|---|------|------|---------|
| 3a | DSP_TIMING → DiagEvent push | ✅ | observeReason文字列から数値マッピング（Forward→1, Rollback→2, Replay→3）→ PublicationDirection |
| 3b | CALLBACK_STAGE → DiagEvent push | ✅ | budgetPermille保持（formatはTimer側）|

## Phase 4: BlockDouble.cpp — CPU_MIG ✅

| # | 項目 | 状態 | 変更内容 |
|---|------|------|---------|
| 4a | CPU_MIG → DiagEvent push | ✅ | AudioBlock.cppと同一パターン |

## Phase 5: DSPCoreFloat.cpp — EQ_TIME ✅

| # | 項目 | 状態 | 変更内容 |
|---|------|------|---------|
| 5a | logEqTime → DiagEvent push | ✅ | モジュール別ポインタ経由（eqDiagBuffer）+ budgetPermille計算 |

## Phase 6: DSPCoreDouble.cpp — EQ_TIME ✅

| # | 項目 | 状態 | 変更内容 |
|---|------|------|---------|
| 6a | logEqTime → DiagEvent push | ✅ | Float版と同一（setEqDiagBuffer定義はFloat側のみ、ODR回避） |

## Phase 7: ConvolverProcessor.Runtime.cpp — CONV_TIME + STCONV_TIME ✅

| # | 項目 | 状態 | 変更内容 |
|---|------|------|---------|
| 7a | CONV_TIME → DiagEvent push | ✅ | convDiagBuffer経由、callQ/lat保持 |
| 7b | STCONV_TIME → DiagEvent push | ✅ | 同上 |

## Phase 8: AudioEngine.Timer.cpp — 消費 + 統計 ✅

| # | 項目 | 状態 | 変更内容 |
|---|------|------|---------|
| 8a | diagBuffer pop + フォーマット | ✅ | while(pop()) + switch 7種 + diagLog（PublicationDirection文字列変換含む） |
| 8b | DiagStatistics | ✅ | [DIAG_STAT] pushed/popped/dropped/approxOcc/backlog/dropRate |

## Phase 9: ビルド確認 ⚠️

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 9a | Debug ビルド | ⚠️ | **ビルド環境の問題**（VS2026 previewの標準ライブラリヘッダ不足:`'algorithm':No such file or directory`）により確認不可 |
| 9b | Release ビルド | ⚠️ | 同上（ビルド環境復旧後に確認） |
| 9c | sizeof確認→static_assert== | ⏳ | ビルド通貨後に `sizeof(DiagEvent)` を出力して `== 実測値` に変更 |

---

## 変更ファイル一覧

| ファイル | 変更種別 | 概要 |
|---------|---------|------|
| `src/audioengine/AudioEngine.h` | 追加 | DiagCategory, 7種Data構造体, DiagEvent, DiagRuntimeLimits, DiagPerTickCounter, RTAuxMutableカウンタ, diagBufferメンバ, setter宣言 |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 変更 | CPU_MIG, CB_SEQ, DSP_TIMING, CALLBACK_STAGE→DiagEvent push |
| `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 変更 | CPU_MIG→DiagEvent push |
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | 変更 | logEqTime→DiagEvent push + setEqDiagBuffer定義 |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | 変更 | logEqTime→DiagEvent push（setter定義削除でODR回避） |
| `src/convolver/ConvolverProcessor.Runtime.cpp` | 変更 | CONV_TIME, STCONV_TIME→DiagEvent push + setConvDiagBuffer定義 |
| `src/audioengine/AudioEngine.Timer.cpp` | 追加 | diagBuffer pop+format+diagLog + [DIAG_STAT] |
