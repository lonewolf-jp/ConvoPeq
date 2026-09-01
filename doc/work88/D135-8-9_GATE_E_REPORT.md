# D135-8/9 Gate E — Work Report

**Gate:** E (Lifecycle / Ownership / DSP Retirement Regression Audit, read-only)
**Result:** PASS (7/7: E-1 .. E-7)
**Base:** HEAD `5f6f48c` (F6 commit) — git clean; `ConvoPeq.md` regenerated `2026-08-30 19:19:29`
**Scope:** source 0 edits · build 0 · CTest 0 · doc 0 (evidence only)

## Summary
Gate D が retry/retention accounting を検証したのに対し、Gate E は **F6 が既存 publish→activate→retire / DSPHandle ownership / crossfade / recovery-obligation lifecycle へ副作用を起こしていないか**を diff レベルから call-chain レベルまで検証。

## Key findings
- **E-1**: `submitPublishRequest→trySubmitImpl→executor_.publish(success)→RuntimePublishExecutor.h:104 onPublishCompleted→lifetime.activate/retire` の Execution tail を DSPTransition.h:49-155 が実装。activate は publish 成功後 (onPublishCompleted) のみ。trySubmitImpl を F6 は変更しない (lifecycle term 0 additions in 5f6f48c)。
- **E-2**: `registerDSPHandleForRuntime` (Commit.cpp:804 / h:4612) → `retireDSPHandleForRuntime` (h:4336 findAndErase-once) → `ISRRetireRouter::enqueueWithRetry` (quarantine fallback, no direct delete)。false-return は "already mapped elsewhere" → ownership not lost。
- **E-3**: crossfade normal path `claimFadingRuntimeDSP` CAS exactly-once (h:126); emergency path `exchangeFadingRuntimeDSP` CAS (h:77) + `prevRaw!=old` skip (h:80); `oldDSP!=newDSP` guards at 3 sites。`terminalizeFadingDSP` は decl-only (impl 0, caller 0) — dormant stub。Live terminalize は CAS clear fadingRuntimeDSPSlot。
- **E-4**: `finishView`/`invalidateDeferredObligation` は DSP を触らない。F6-added cpp:466 overwrite-retire は register(+1,Commit:804) / retire(+1,466) balanced — never-published handle release。recovery logical obligation は `resolveRecoveryObligation` 呼ばない。
- **E-5**: `RejectedPublishFailure` → `destroyRolledBackDSP` (direct delete, NOT EBR) (cpp:291/D4LifeManager.cpp:149-157) + ScopeExit `rollbackDSPHandleRegistration` (h:4605-4609)。old active DSP は触らない。
- **E-6 (precise git diff a65ace1..5f6f48c):** F6 lifecycle-term additions = **cpp:466 + cpp:500 (`retireDSPHandleForRuntime`) only**。lifetime.retire/activate, destroyRolledBackDSP, registerDSPHandleForRuntime, enqueueRetire, ISRRetireRouter, DSPTransition, CrossfadeAuthority(defs), claimFading/exchangeFading/beginCrossfade/endCrossfade, terminalizeFadingDSP — **すべて 0 diff lines**。ISRRC は a65ace1 が変更 (F6 非変更) → ownership ledger と completely separated。
- **E-7**: recovery obligation table `liveCount_` single +1 (tryInsert, cap 32) / single −1 (resolve CAS) (ISRRC.h:334-410)；single Completion Authority `resolveRecoveryObligation` (ISRRC.cpp:944-980, Retry early-return keeps Live)；publish live ownership findAndErase-once。silent disappearance / double-ownership / double-retire / overwrite = 0。

## Resolved / preserved
- F4 open item (double-retire on crossfade): CAS claim (h:126) + prevRaw!=old skip (h:80) → exactly-once confirmed.
- old==new retire: 3-guard confirmed (DSPTransition.h:74/109/150).
- `terminalizeFadingDSP` decl-only → "fading terminalize primitive" は CAS clear fadingRuntimeDSPSlot が実体 (Timer.cpp:959-967 等)。F6 は未変更。
- 5f6f48c に `tests/DeferredFlowIntegrationTests.cpp` +1 line (test tweak) 含む — read-only audit (0 edit) に影響なし。記録のみ。

## Full evidence
`evidence/D135-8-9_GATE_E_LIFECYCLE_OWNERSHIP_RETIRE_AUDIT.md`

## Next
Gate E PASS → **Gate F** 進行可 (定義次第)。E 同様 read-only, build 0, CTest 0 維持で実施。
