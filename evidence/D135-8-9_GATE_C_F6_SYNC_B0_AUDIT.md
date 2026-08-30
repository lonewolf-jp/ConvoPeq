# D135-8/9 Gate C-F6-SYNC-B0 — Read-only Source/Working-Tree Synchronization Audit

Date: 2026-08-30
制約: read-only（source/test/build/CTest 変更 0）。唯一の書き込みは **`ConvoPeq.md` の再生成**（派生スナップショットの更新であり、ユーザーの BLOCK 修復手順 `working tree → ConvoPeq.md 再生成 → F6-SYNC-B0 再監査` に準拠。production/test source・ビルド成果物・CTest には一切触れていない）。

## Verdict: **初期判定 BLOCK → 修復実施 → 最終 PASS**（Gate D へ進んでよい）

- **BLOCK 理由**: 保管 `ConvoPeq.md`（内部 `Generated: 2026-08-30 07:56:22`）が F6 実装を含まない旧世代だった。
  F6 マーカー（`deferredRetryObligationId_` / `deferredObligationCreatedAtUs` / `kDeferredWakeWatchdogTicks` /
  `coordinatorDeferredWatchdogTicks_`）の出現数 = **0**。`metadata.enqueueTimestampUs = now` の旧構造のまま。
  → 「F6 実装済み working tree ≠ 保管 ConvoPeq.md」をユーザー指摘どおり確認。
- **working tree 自体は正しい**: 実ソースには F6 が全て存在（下記 1-7 で検証）。Debug/Release build・CTest が
  F6 ソースで PASS した事実とも整合。
- **修復**: `python output_sourcecode_markdown.py` で `ConvoPeq.md` を再生成（`Generated: 2026-08-30 19:19:29`）。
  再生成後、F6 マーカー **17件**、`enqueueTimestampUs = deferredObligationCreatedAtUs` 1件、
  `F6-4: fade-complete wake` 1件、identity tuple 1件、watchdog 3件、State.h F6 コメント 2件 — 同期復元。

## 必須確認（working tree 実ソース照合）

### 1. RuntimePublicationOrchestrator.cpp/.h
- ✅ identity = `(generation, recoveryObligationId)`（Orch.cpp:475-476 `req.generation == deferredRetryGeneration_ && req.recoveryObligationId == deferredRetryObligationId_`）
- ✅ `deferredRetryObligationId_`（Orch.h:297 宣言、Orch.cpp:479 代入、:669/:176 リセット）
- ✅ `deferredObligationCreatedAtUs`（Orch.h 宣言、Orch.cpp:481 `!sameObligation` 時のみ now、:528 metadata へ）
- ✅ retention 時の `count++` 不在（`++deferredRetryCount_` の出現 **0件**）
- ✅ `enqueueDeferred()` で createdAt 維持（re-drive は `sameObligation` → 代入スキップ）
- ✅ `finishView()` terminal reset（reason≠None → invalidate、Orch.cpp:653-654）
- ✅ shutdown reset（clearDeferredForShutdown → invalidate、:560）
- ✅ terminal reset（processDeferredAdmission submit 後 !hasDeferred → invalidate、:746）
- ✅ `resetDeferredRetryBudget()` が identity + count + **createdAt まで消去**（Orch.h:175-178 の4行）

### 2. AudioEngine.Timer.cpp
- ✅ fade-complete wake（`F6-4`、fadeCompleted ブロック末尾 :1002-1020）
- ✅ idle publish 完了後・`sendChangeMessage()` 後
- ✅ `publishRetryReady` のみ設定
- ✅ `recoveryRetryReady` 非汚染（実書込みは recovery 経路 :1768 のみ）

### 3. AudioEngine.Threading.cpp
- ✅ `kDeferredWakeWatchdogTicks` / `coordinatorDeferredWatchdogTicks_` 使用（:288）
- ✅ 毎 tick の `notify_one()` 除去（`notify_one` は watchdog ゲート内 :295 の1箇所のみ）

### 4. AudioEngine.RebuildDispatch.cpp
- ✅ CV predicate 不変（:855-858 `hasPendingTask || publishRetryReady || recoveryPending || shouldExit`）
- ✅ `recoveryRetryReady` provenance 維持（exchange 消費 :888、predicate 非参加）

### 5. RuntimePublicationState.h
- ✅ F6 契約コメント（`RetryExhaustedDiscard` = Type-A 専用・retention 非算入・dormant・`>` 正・identity=(gen,oblId)）

### 6. D135-1_IMPLEMENTATION.md
- ✅ `>=` → `>` 訂正 + F6 注記（retention 非算入 / dwell TTL / dormant kMax）

### 7. test source 無変更
- ✅ `DeferredFlowIntegrationTests.cpp` に F6 マーカー **0件**（無編集）

## 同一ソース世代への帰属（判定ルールの PASS 条件）

```text
F6-B0 12/12
+ Gate C Debug 40/40
+ Gate C Release 40/40
+ Release harness ×5
```
→ いずれも **F6 実装済み working tree**（本監査で確認した実ソース）に帰属。`ConvoPeq.md`（19:19:29 版）も
同一世代と一致。よって **F6-SYNC-B0 = PASS**。

## Gate D 解禁

F6-SYNC-B0 PASS により、次は **D135-8/9 Gate D — Retry / Retention State-Machine Audit**。
対象は変更なし（F5 §F4-6/F5-9 の最終状態表）: ordinary retention/re-drive で count 不変 / 同一 (G,O) で
createdAt 不変 / 別 (G,O) で新 createdAt / recovery wake で reset / fade-complete・watchdog wake は
ordinary（reset なし）/ TTL 超過で terminal / terminal で identity+timestamp 消去 / Rejected* recovery は
F5-10 表どおり / `kMaxDeferredRetries` は retention で到達不能（dormant）。
**Gate D は「何回まで」の再テストではなく「retry accounting と retention の完全分離」の検証**である点を維持。
