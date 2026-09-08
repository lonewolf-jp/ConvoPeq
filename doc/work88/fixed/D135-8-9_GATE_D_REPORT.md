# D135-8/9 Gate D — Work Report

**Gate:** D (Retry / Retention State-Machine Audit, read-only)
**Result:** PASS (10/10)
**Base:** HEAD `5f6f48c` (2026-08-30 19:30:11) — post-F6 + regenerated ConvoPeq.md (`Generated: 2026-08-30 19:19:29`)
**Scope:** production source 0 changes · test source 0 changes · build 0 · CTest 0 · doc 0 (evidence only)

## What was done
Single-engineer read-only state-machine audit (D-1 … D-12) tracing the **live call chain** across
`AudioEngine.Timer.cpp`, `AudioEngine.Threading.cpp`, `AudioEngine.RebuildDispatch.cpp`,
`AudioEngine.Commit.cpp`, `RuntimePublicationOrchestrator.{cpp,h}`, `PublicationAdmission.{cpp,h}`,
`RuntimePublicationState.h`.

All grep sweeps were confirmed against the regenerated ConvoPeq.md 2026-08-30 edition (old-snapshot
stale `enqueueTimestampUs = now` per re-drive は 0 件 — aggregate init 1 件のみ).

## Verdict summary
1. retention で retry count 増加 0 (+`++deferredRetryCount_` 0 sites)
2. identity = (generation, recoveryObligationId) tuple compare & paired writes
3. 同一 identity で createdAt 不変 / identity change のみ `= now`
4. metadata.enqueueTimestampUs は obligation createdAt から唯 1 派生 (source-of-truth 一点)
5. TTL `ageUs > ttlUs` strict `>` (30s obligation dwell), `>=`/`=` 0 件
6. terminal 4-site zero-reset (clearDeferredForShutdown / finishView / resetDeferredRetryBudget / post-check)
7. `resetDeferredRetryBudget` 唯一呼び出し = `processDeferredAdmission(wasRecoveryWake)` (cpp:716)
8. fade-complete: publishRetryReady only, recoveryRetryReady 非触
9. watchdog: notify は threshold 到達のみ (旧毎 tick polling churn は削除済み)
10. `kMaxDeferredRetries` (=2) exists, reset-only, retention unreachable, Type-A dormant guard

## Open items resolved in this pass
- **watchdog 周期** (was F4 "未確定"): `kDeferredWakeWatchdogTicks = 100` (~100ms @1ms tick),
  named constant / tunable fallback — **not** a spec contract (F5-6).
- **`>=` / `>` drift** (F2 D135-1): canonical `>` confirmed at cpp:497 & Pub:79; 0 `>=`.
- **SupersededDiscard**: producer 0 (dormant enum; supersession は StaleDiscard via gen/seq)。新規 finding, no action.
- **DiscardReason::Expired**: enum declared (h:15) but TTL discard reports StaleDiscard (Pub:80)。cosmetic; work37 future-split reserved.
- **enqueueDeferred thread-ownership**: 3/3 `enqueuePublicationIntentForRuntimeCommit` callers + post-check all within `rebuildThreadLoop` → RebuildThread single-owner 契約 (h:291) 再確認.

## Carry-over (out of Gate D scope, preserved)
- F4-10: `enqueueTimestampUs` → `obligationCreatedAtUs` rename (reserved, non-blocking).
- F5-10 ⚠ 列: Rejected* (non-recovery) handle 回収 follow-up (evidence/D135-8-9_GATE_C_F5...:214)。not touched.

## Full evidence
`evidence/D135-8-9_GATE_D_STATE_MACHINE_AUDIT.md` (D-1..D-12 with line numbers + decision table)

## Next
Gate E = lifecycle / ownership / DSP retirement regression audit (D135-2_GATE_E_REPORT.md の D135-2 流を踏襲) —
pending explicit go / source edit window (Gate D では build 0 / edit 0)。
