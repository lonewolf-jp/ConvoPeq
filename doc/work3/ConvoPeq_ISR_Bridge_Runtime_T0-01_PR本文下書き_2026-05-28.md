# PR本文（下書き）: PR-T0-01 Admission Gate 原子契約と TOCTOU 封じ

## タイトル案

`[Tier0][PR-T0-01] Admission Gate atomic契約明示 + enqueue直前double-checkでTOCTOU封止`

## 背景

`ConvoPeq_ISR_Bridge_Runtime_T0-01_具体的変更計画_PR用_2026-05-28.md` に基づき、Tier0 の最初の必須PRとして Admission Gate の契約を明示し、shutdown 遷移競合での TOCTOU 窓を最小差分で封止する。

本PRは次Tier（T0-02以降）に進むための土台であり、挙動変更は「shutdown 競合時の reject 収束」に限定する。

## Purpose

1. `acceptsRuntimePublication()` の atomic 契約（acquire/release 意味論）を明示する
2. 判定後〜enqueue/publish 直前に double-check を追加して TOCTOU を封止する
3. shutdown 遷移後の流入を一貫して reject 側に倒す

## Scope (Files)

- `src/audioengine/AudioEngine.Commit.cpp`
  - `acceptsRuntimePublication()`
  - `appendPublicationIntentForCommitSlot(...)`
  - `prepareCommit(...)`
  - `executeCommit()`
- `src/audioengine/AudioEngine.RebuildDispatch.cpp`
  - `submitRebuildIntent(...)`
  - `requestRebuild(convo::RebuildKind)`
  - `requestRebuild(double, int, bool)`
  - `handleAsyncUpdate()`
- `src/audioengine/AudioEngine.h`
  - 宣言周辺コメント（必要に応じ helper 宣言）

## In-Scope

- Admission 判定コメント/契約の明文化
- queue反映直前の再判定（double-check）
- reject telemetry/counter の整合

## Out-of-Scope

- `waitForDrain(timeout)` 導入・調整（T0-03）
- drain SSOT 集約拡張（T0-02）
- saturation hysteresis（T0-04）
- crossfade/latency 系純化（Tier2）

## 実装方針（最小差分）

- 既存 gate を置換せず、**再判定を追加**する
- 成功経路のデータフローは維持し、失敗経路のみ shutdown-safe に強化する
- 生成済みオブジェクトは reject 時に `retireDSP()` / deferred delete で回収し、リークを発生させない

## Acceptance Criteria

1. shutdown 遷移後、publication intent の新規連結が発生しない
2. shutdown 遷移後、rebuild pending task が新規投入されない
3. `queue.push -> rejectLater` 型の後段 reject パターンを増やさない
4. reject 時に `newDSP` / `intent` がリークしない
5. 既存 commit/rebuild 成功パスの動作を崩さない

## Verification

### 静的確認

- `acceptsRuntimePublication()` 呼び出し位置を再検索し、queue反映直前未ガードを確認
- 追加 reject path の解放処理を確認

### ビルド

- `Debug Build (cmd env retry)`
- `Release Build (cmd env retry)`

### ルール

- `Strict Atomic Dot-Call Scan`

### 手動シナリオ

- Scenario A: rebuild連打中にshutdown開始
  - 期待: 新規 queue 化が reject 側へ倒れる
- Scenario B: `handleAsyncUpdate()` 実行中にshutdown開始
  - 期待: `ShutdownInProgress` 理由で suppress/return
- Scenario C: 通常運転時 rebuild
  - 期待: 既存成功パス維持

## Rollback Plan

差分を3コミットに分離し、段階rollback可能にする。

1. コメント/契約明文化（無害）
2. commit 経路 double-check
3. rebuild 経路 double-check

問題発生時は 2/3 を個別 revert。1 は原則維持。

## Risk & Mitigation

- リスク: 再判定の入れ過ぎで通常系が抑止される
  - 緩和: shutdown 条件に限定した reject 理由を telemetry で可視化
- リスク: reject 経路の回収漏れ
  - 緩和: `retireDSP` / deferred delete 経路を静的確認 + 手動再現

## Go / No-Go

- **Go**: 受入条件 1〜5 全充足 + build/scan green
- **Conditional Go**: telemetry 文言差のみ（機能影響なし）
- **No-Go**: shutdown 後 queue 化の再現、またはリーク疑い
