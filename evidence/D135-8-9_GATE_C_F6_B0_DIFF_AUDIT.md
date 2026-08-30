# D135-8/9 Gate C-F6-B0 — Read-only Implementation Diff Audit

Date: 2026-08-30
対象: F6 実装（F5 §D 必須5点 + F6-7 lifecycle reset + F6-8 契約文書）。**この監査自体は read-only（ビルド・CTest 実施前）**。

## Verdict: **F6-B0 = PASS（12/12）** → Debug/Release build・CTest へ進んでよい

## 1. changed files（F6 で編集したもの）

| ファイル | F6 項目 |
| --- | --- |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | F6-1/2/3/7（accounting・metadata・invalidate・processDeferredAdmission・finishView・clearDeferredForShutdown） |
| `src/audioengine/RuntimePublicationOrchestrator.h` | F6-1/2/6/7/8（identity/createdAt メンバ・kMax 注記・resetDeferredRetryBudget・invalidate 宣言） |
| `src/audioengine/AudioEngine.h` | F6-5（watchdog カウンタ + named constant） |
| `src/audioengine/AudioEngine.Timer.cpp` | F6-4（fade-complete wake） |
| `src/audioengine/AudioEngine.Threading.cpp` | F6-5（coordinator poll → watchdog 化） |
| `src/audioengine/RuntimePublicationState.h` | F6-8（RetryExhaustedDiscard 契約コメント） |
| `doc/work88/D135-1_IMPLEMENTATION.md` | F6-8（`>=` → `>` 訂正 + F6 注記） |

**test source 変更: 0**（`DeferredFlowIntegrationTests.cpp` は F6 では未編集）。

## 2-12. 確認項目

| # | 項目 | 結果 | 証拠 |
| --- | --- | --- | --- |
| 2 | diff review | ✅ | 全編集領域を読み戻し（下記） |
| 3 | identity = (generation, recoveryObligationId) | ✅ | Orch.cpp:475-476 `req.generation == deferredRetryGeneration_ && req.recoveryObligationId == deferredRetryObligationId_`。単一 gen 比較の残存 **ゼロ**（grep） |
| 4 | retention path で count++ しない | ✅ | `++deferredRetryCount_` はソース全体に **存在しない**（grep CLEAN）。新 obligation のみ count=0 |
| 5 | createdAt が re-drive で更新されない | ✅ | `deferredObligationCreatedAtUs = now` は `!sameObligation` 分岐内のみ（Orch.cpp:481）。retention は不変 |
| 6 | metadata と slot timestamp の意味混線なし | ✅ | metadata.enqueueTimestampUs = `deferredObligationCreatedAtUs`（:528）、slot.enqueueTimestampUs = `now`（:532）。コメントで意味を分離明記 |
| 7 | fade-complete wake が正しい位置 | ✅ | Timer.cpp:1000 `sendChangeMessage()` 直後・fadeCompleted ブロック末尾。idle publish 完了後 |
| 8 | recoveryRetryReady に触れていない | ✅ | fade-complete wake は `publishRetryReady` のみ設定。`recoveryRetryReady` の実書込みは recovery 経路（Timer.cpp:1768）のみ |
| 9 | watchdog が毎 tick wake を完全抑制 | ✅ | Threading.cpp:288 `++coordinatorDeferredWatchdogTicks_ >= kDeferredWakeWatchdogTicks` でゲート、else でカウンタリセット |
| 10 | CV predicate 不変 | ✅ | RebuildDispatch.cpp:855-858 predicate は `hasPendingTask \|\| publishRetryReady \|\| recoveryPending \|\| shouldExit` のまま（hasDeferred_ 非追加） |
| 11 | terminal reset 全経路 | ✅ | finishView（reason≠None→invalidate :653-654）/ processDeferredAdmission（submit 後 !hasDeferred→invalidate :745-746）/ clearDeferredForShutdown（invalidate）/ resetDeferredRetryBudget（recovery wake、identity+count+createdAt 全消去 h:171-178） |
| 12 | ownership/retire 経路に余計な変更なし | ✅ | consume/discard/finishView の ownership release 構造は不変（invalidate 呼び出し追加のみ）。retire pipeline（retireDSPHandleForRuntime）無変更 |

## 追加整合（F6-B0 中に検出し修正した不整合）

- `processDeferredAdmission` の stale コメント（「Ordinary retry wakes keep the increment path intact」）→ F6 の retention 非算入に更新済み（Orch.cpp:707-713）。
- `resetDeferredRetryBudget` が新 identity（oblId/createdAt）を消去していなかった → 全 obligation metadata を消去するよう修正（Orch.h:171-178）。recovery 後の再駆動が新 dwell として扱われる（F5-9 整合）。

## 契約遵守チェック（F6 禁止事項）

- `Expired` enum: 変更なし ✅ / Rejected* terminal 再設計: なし ✅ / telemetry schema: なし（`oldestDeferredAgeMs` 意味変更なし）✅ / CV predicate: なし ✅ / recovery provenance: なし ✅ / recovery obligation table: なし ✅ / DSP retire pipeline: なし ✅ / test source: なし ✅ / `kMaxDeferredRetries` 削除: なし（dormant 保持）✅

## 次のゲート（F6-B0 PASS により許可）

```text
Debug build → Debug CTest → Release build → Release CTest → Gate C 再判定
→ DeferredFlow integration test 確認 → Release harness ×5 → TTL/retention/wake evidence → Gate C final
```
