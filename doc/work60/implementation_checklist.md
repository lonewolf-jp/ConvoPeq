# 実装チェックリスト — 計測追加改修 v7

**対象**: ConvoPeq 3rd-observation | **開始**: 2026-06-30 | **Status**: In Progress

---

## Phase 0: 基盤整備

- [ ] F: diagLog Commit.cpp 修正（`JUCE_DEBUG` → `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`）

## Phase 1: RTLocalState メンバ変数追加（AudioEngine.h）

- [ ] RTLocalState に以下を追加:
  - `lastCallbackEntryUs` (atomic<uint64_t>)
  - `lastCallbackDriftUs` (atomic<int64_t>)
  - `lastCallbackProcessor` (atomic<uint32_t>, UINT32_MAX初期値)
  - `cpuMigrationCount` (atomic<uint64_t>)
  - `lastCallbackPublicationSeq` (atomic<uint64_t>)
  - `CallbackTimingEntry` 構造体
  - `callbackTimingHistory[32]`
  - `callbackTimingWriteCount` (atomic<uint64_t>)

## Phase 2: AudioBlock.cpp / BlockDouble.cpp 改修

- [ ] A: callback entry timing（RuntimeScope直後, 3ブロック）
- [ ] G: CPU migration（Aと同時、1ブロック）
- [ ] H: publicationSequence（Aと同時、1ブロック）
- [ ] A: DriftUs を [XRUN] ログに追記
- [ ] B: CallbackTelemetryScope 常時化 + リングバッファ書込
- [ ] B: CB_HIST ダンプ（XRUN検出ブロック内）
- [ ] B: 新 CallbackTelemetryScope 構築引数対応

## Phase 3: ConvolverProcessor.Runtime.cpp 改修

- [ ] C: ConvolverProcessor::process() 全体タイマー + [CONV_TIME] 出力
- [ ] C2: StereoConvolver::process() 単体タイマー + [STCONV_TIME] 出力

## Phase 4: DSPCoreDouble.cpp / DSPCoreFloat.cpp 改修

- [ ] D: EQ::process() タイマー + bandカウント + [EQ_TIME] 出力（各4箇所×2ファイル）

## Phase 5: DSPCoreDouble.cpp / DSPCoreIO.cpp 改修

- [ ] E: ANS applyMatchedCoefficients タイマー + [ANS_SWITCH] 出力

## Phase 6: 検証

- [ ] Release ビルド成功確認
- [ ] CLI smoke test（起動→終了）
- [ ] テストシナリオ実行（無音→IR+PEQ→ANS→音楽）
- [ ] ログに新規診断タグ出現確認
