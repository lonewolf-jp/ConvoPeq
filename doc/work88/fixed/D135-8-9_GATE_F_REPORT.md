# D135-8/9 Gate F — Work Report

**Gate:** F (End-to-End Invariant / Boundary / Regression Final Audit, read-only)
**Result:** PASS (F-1 .. F-7; final matrix 13/13 GO)
**Base:** HEAD `5f6f48c` (F6); primary reference = regenerated `ConvoPeg.md` (`Generated: 2026-08-30 19:19:29`); git clean.
**Scope:** production/test source 0 · build 0 · CTest 0 · doc 0 (evidence only).

## Purpose
Gate D (retry/retention) と Gate E (lifecycle/ownership/retire) が個別に PASS した。Gate F はそれらを **state-transition × boundary** に再構成し、`processDeferredAdmission()`(cpp:700) の一本化が全 invariant を閉じているかを最終確認。read-only（> 実装へ進む前の最終構造証明）。

## Per-item
- **F-1**: Deferred ownership end-to-end。`deferredSlot_` std::optional (h:279) 0/1 cardinality (never 0-and-live / never 2)。mono-release via `finishView` (cpp:637) / `clearDeferredForShutdown` (555)。consume/discard exactly-once (view state_ jassert cpp:678/687)。re-defer で identity/createdAt 保持 (cpp:475-483,528)。terminal invalidate 4-site (560/654/666-672/746)。
- **F-2**: 3-ledger boundary (DeferredPublish / Recovery logical / DSPHandle)。resets (finishView/invalidate/resetBudget/clearDeferredForShutdown) は L1 のみ; `resolveRecoveryObligation` L2 のみ; `retireDSPHandleForRuntime`/`destroyRolledBackDSP` L3 のみ。cross-ledger release = 0。`lastRecoveryPublishSeq_` は correlation stamp (h:170; cpp:561 reset benign)。
- **F-3**: wake/race boundary flag-matrix。`publishRetryReady` plain-bool under rebuildMutex; `recoveryRetryReady`/`deferredClearRequested_`/`hasDeferred_`/`lastRecoveryPublishSeq_` std::atomic (release-set / acq_rel exchange / acquire-load)。6 race cases all closed。lost-wake proof: predicate-ineligible persistent latches (cpp:589-592, h:303-306) — drained every wake.
- **F-4**: capacity ≠ ownership cardinality。logical obligation 32 (h:325,409) / transport 256 (h:897) / deferred slot 1 (h:279) / handle table 512 (h:5018) / quarantine fallback 1024 (h:961)。独立 container で互いに混同しない。I4 の pendingReclaimHandles_ container bound も別軸。
- **F-5**: dormant path reachability。`RetryExhaustedDiscard` (count-gate cpp:497, 0>2 false — UNREACHABLE) / `SupersededDiscard` (producer 0) / `Expired` (TTL→StaleDiscard Pub:80, producer 0) / `terminalizeFadingDSP` (decl-only impl 0/caller 0)。enum/decl 存在 ≠ production path。
- **F-6**: F6 additions (deferred accounting / metadata / wake provenance / clear latch / watchdog) vs depends-not-modify (DSPTransition/CrossfadeAuthority/DSPLifeManager/ISRRetireRouter/ISRRuntimePublicationCoordinator)。git diff 5f6f48c lifecycle additions = `retireDSPHandleForRuntime` 2 lines (cpp:466 balanced, cpp:500 dormant) only; others = 0 diff lines → no authority contamination。
- **F-7**: 13/13 GO matrix。

## Resolved / observations
- Gate A-F の read-only 監査チェーンは完了。F6 系の静止監査閉鎖。
- `5f6f48c` 内 `tests/DeferredFlowIntegrationTests.cpp` +1 line (sidenote; read-only so未編集)。
- dormant symbols (`SupersededDiscard`/`Expired`/`terminalizeFadingDSP`/`RetryExhaustedDiscard`) は Phase-II の実装 window で有効化する場合、別途監査要。

## Full evidence
`evidence/D135-8-9_GATE_F_INVARIANT_BOUNDARY_REGRESSION_FINAL.md`

## Next
Gate F 閉鎖 → **実装前に読み出し・構造証明が finish**。ソース編集 / build / CTest は行わず、次のユーザー指示 (次の実装フェーズ or Gate G+ 定義) 待ち。
