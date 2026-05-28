# T0-01 実装チェックリスト（コピー用）

> 使い方: PR作業中にこのまま貼り付けて、完了した項目に `[x]` を付ける。

## A. 事前確認

- [ ] 対象は T0-01 のみ（T0-02/T0-03/T0-04 の内容を混在させない）
- [ ] 1PR 1目的（Admission atomic契約 + TOCTOU封止）を確認
- [ ] Out-of-Scope を確認（waitForDrain, SSOT拡張, hysteresis, Tier2系）

## B. 対象関数チェック

### B1 Commit系

- [ ] `acceptsRuntimePublication()` を確認
- [ ] `appendPublicationIntentForCommitSlot(...)` を確認
- [ ] `prepareCommit(...)` を確認
- [ ] `executeCommit()` を確認

### B2 Rebuild dispatch系

- [ ] `submitRebuildIntent(...)` を確認
- [ ] `requestRebuild(convo::RebuildKind)` を確認
- [ ] `requestRebuild(double, int, bool)` を確認
- [ ] `handleAsyncUpdate()` を確認

### B3 宣言/コメント系

- [ ] `AudioEngine.h` の契約コメントを更新
- [ ] helper 宣言追加が必要なら最小範囲で追加

## C. 差分実装チェック

### C1 契約明文化

- [ ] `acceptsRuntimePublication()` の判定有効期間を call-scope 内と明記
- [ ] acquire/release の意味論コメントを明記

### C2 TOCTOU封止

- [ ] enqueue/publish 直前に double-check を追加
- [ ] shutdown 遷移後到達分を reject 側へ倒す

### C3 reject整合

- [ ] reject telemetry reason を `ShutdownInProgress` 系で統一
- [ ] reject counter 増加が既存設計と矛盾しない
- [ ] reject時に `newDSP` / `intent` を回収（リークなし）

## D. 静的確認

- [ ] `acceptsRuntimePublication` 呼出位置を再検索
- [ ] queue反映直前の未ガード経路が残っていない
- [ ] `queue.push -> rejectLater` 型の後段 reject を増やしていない

## E. ビルド・ルール検証

- [ ] `Strict Atomic Dot-Call Scan` 実行・問題なし
- [ ] `Debug Build (cmd env retry)` 成功
- [ ] `Release Build (cmd env retry)` 成功

## F. 手動シナリオ検証

### F1 shutdown競合

- [ ] Scenario A: rebuild連打中 shutdown
- [ ] 期待結果: 新規 queue 化が reject 側へ収束

### F2 async bridge競合

- [ ] Scenario B: `handleAsyncUpdate()` 中 shutdown
- [ ] 期待結果: suppress/return に収束

### F3 通常系回帰

- [ ] Scenario C: 通常 rebuild
- [ ] 期待結果: 既存成功パス維持

## G. 受入判定（最終）

- [ ] shutdown 遷移後 publication intent 新規連結なし
- [ ] shutdown 遷移後 rebuild pending task 新規投入なし
- [ ] 後段 reject パターン増加なし
- [ ] リークなし
- [ ] commit/rebuild 成功パス回帰なし

## H. PR提出前

- [ ] PR本文に Purpose / Scope / Acceptance / Verification / Rollback を記載
- [ ] 3コミット分割（契約明文化 / commit経路 / rebuild経路）を維持
- [ ] リスクと緩和策をPRに追記
