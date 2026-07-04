# 実装チェックリスト — CallbackArrival 計測追加

**設計書**: `work61_計測項目追加_改修計画書_2026-07-03.md`
**対象ファイル**: AudioEngine.h / BlockDouble.cpp / AudioBlock.cpp / Timer.cpp
**進捗**: 0/17 items

## Step 1: AudioEngine.h — データ構造追加

- [ ] 1.1 DiagCategory enum: `CallbackArrival = 8` を追加
- [ ] 1.2 `struct CallbackArrivalData` 定義（20byte）
- [ ] 1.3 DiagEvent union に `callbackArrival` メンバー追加
- [ ] 1.4 `/ kDiagEventSizeMax = 96` のコメント更新（必要なら）
- [ ] 1.5 `RTLocalState`: `cachedThreadId` (uint32_t) 追加
- [ ] 1.6 `DiagPerTickCounter` 領域: `cbArrivalWritten`, `cbArrivalDropped` 追加

## Step 2: AudioEngine.h — recordCallbackArrival() 関数

- [ ] 2.1 `recordCallbackArrival()` インライン関数を追加（共通ヘッダ部）

## Step 3: AudioEngine.Processing.BlockDouble.cpp

- [ ] 3.1 `fetchAddAtomic(audioCallbackEpochCounter)` を RuntimeScope より前に移動
- [ ] 3.2 早期returnガード（lifecycle, shutdown）より **後** に配置
- [ ] 3.3 `recordCallbackArrival()` 呼び出しを追加（increment直後）
- [ ] 3.4 `cachedThreadId` の初期化を prepareToPlay / 初回callback に追加

## Step 4: AudioEngine.Processing.AudioBlock.cpp

- [ ] 4.1 `fetchAddAtomic(audioCallbackEpochCounter)` を RuntimeScope より前に移動
- [ ] 4.2 `recordCallbackArrival()` 呼び出しを追加

## Step 5: AudioEngine.Timer.cpp — 読出側

- [ ] 5.1 `CallbackArrivalContext` struct（static）追加
- [ ] 5.2 `case DiagCategory::CallbackArrival` 追加（drift計算＋ログ出力）
- [ ] 5.3 DIAG_STAT に `cbArrivalWritten/Dropped` 出力追加
- [ ] 5.4 リセット処理（prepareToPlay連動）

## Step 6: verify

- [ ] 6.1 `ccc` / `semble` / `rg` で全変更箇所確認
- [ ] 6.2 ビルドテスト
