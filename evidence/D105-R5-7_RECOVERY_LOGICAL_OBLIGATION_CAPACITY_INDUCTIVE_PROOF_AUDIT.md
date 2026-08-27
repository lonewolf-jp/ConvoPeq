# D105-R5-7 — Recovery Logical Obligation Capacity Inductive Proof Audit

**Type**: Read-only formal/structural proof audit / 0 source changes
**Purpose**: Inductively prove `liveLogicalRecoveryObligationCount ≤ 32` for all reachable states of the R5-rev1/R6/R7 model; verify +1/−1 completeness, coalesce/retry/publication-failure/shutdown/race accounting holes are absent.
**Date**: 2026-08-27
**Status**: ✅ Complete — **R5-7-GATE = PASS** (no source changes; proof only)
**Prereq**: D105-R5 (R5-rev1), D105-R6, D105-R7

---

## 0. Proof Premises (re-verified against source this audit)

| # | Premise | Source evidence |
|---|---|---|
| P1 | `liveLogicalRecoveryObligationCount` (L) is a **new** atomic; absent from source today | `rg` for `liveLogicalRecoveryObligationCount|admittedLogicalObligation|logicalObligationCount|nextLogicalRecoveryObligationId` in `src/` ⇒ **0 hits** |
| P2 | `pendingIntentCount_` is pure **delivery residency** (push +1 / pop / rollback −1), not logical | mutations at cpp:626/645/850/861/949/998/1023 — all fetchAdd/fetchSub pairs |
| P3 | `PendingRecoveryAdmission` is a **single slot** (not array) ⇒ ≤1 durable reservation | ISRRuntimePublicationCoordinator.h:667 (`struct`), :682 (`pendingRecoveryAdmission_` single instance) |
| P4 | `recoveryIntentQueue_` = 256-ring (delivery buffer, not logical bound) | h:642/643 |
| P5 | `reclaimSlot` = generation-gated flag release; touches no logical count | Commit.cpp:661 (only caller); grep for `liveLogical|obligat` in Commit.cpp ⇒ only unrelated retirement at :404 |
| P6 | **+1 authority** = Coordinator admission (post-coalesce); **−1 authority** = Coordinator completion authority (single), driven by `RecoveryResolution{obligationId,outcome}` + Shutdown table-iter | R7 §5/§6 |
| P7 | Completion point = `onPublishCommitted` (carrying `recoveryObligationId`), **not** enqueue | R7 §6 |
| P8 | Idempotency guard: terminal transition only from a *live* state; 2nd resolution = no-op | R7 §10 |
| P9 | 32 is a **deliberate, directly-enforced** resource bound (INV-CAP-7, I4 D25) — not derived from upstream | cocoindex → I4 D25 |

---

## 1. Proof Target

```
∀ reachable states S :   0 ≤ L(S) ≤ 32
```
where `L(S) = liveLogicalRecoveryObligationCount` = #obligations in {Created, Transport, Durable, Building}.

Transition closure to prove (R5-7 §1):
```
L ∈ 0..31 ─unique admission─▶ L+1      L ∈ 0..31 ─coalesce─▶ L
          ├─delivery─────────▶ L        ├─retry──────▶ L
          ├─pub-success──────▶ L−1      ├─term-fail──▶ L−1
          ├─supersede────────▶ L−1      └─shutdown───▶ L−1
          └─reject──────────▶ L
L == 32  ─coalesce─▶ 32  ─all non-admit─▶ ≤32  ─unique admission─▶ REJECT (stays 32)
```

---

## 2. +1 Completeness (R5-7 §2)

**Claim**: the *only* transition with ΔL = +1 is **unique admission** (post-coalesce, no matching `CoalesceIdentity`).

- Delivery transitions (Transport↔Durable↔Building, Retry) mutate only `recoveryIntentQueue_` / `pendingRecoveryAdmission_` / `pendingIntentCount_` — none is L (P2,P3,P4).
- `reclaimSlot` never touches L (P5).
- Coalesce and Reject are defined ΔL = 0.
- L does not exist in current source (P1) ⇒ no other code path can increment it; in the designed system the single `+1` site is admission.

∴ No hidden +1 exists. □

---

## 3. −1 Completeness (R5-7 §3)

**Claim**: every live obligation reaches a terminal with **exactly one −1**.

Terminal set T = {Completed, Failed, StaleSuperseded, Superseded, ShutdownDiscard}.
- Each terminal is reached by exactly one event class (R7 §11), mapped to one `RecoveryResolution` outcome.
- The completion authority applies −1 exactly once per obligation (P6, idempotency P8).
- Shutdown iterates the table once per live O (§8).

∴ Each live obligation contributes exactly one −1 over its lifetime. □

---

## 4. R7 State-Machine Closure (R5-7 §4)

| Transition | ΔL | Upper-bound risk |
|---|---:|---|
| none → Created | +1 | **YES (guarded)** |
| Created → Transport | 0 | NO |
| Transport → Building | 0 | NO |
| Building → Durable | 0 | NO |
| Durable → Building | 0 | NO |
| Building → Completed | −1 | NO |
| Building → Failed | −1 | NO |
| Building → StaleSuperseded | −1 | NO |
| live → Superseded | −1 | NO |
| live → ShutdownDiscard | −1 | NO |
| equal admission → same O | 0 | NO |
| table-full admission → Reject | 0 | NO |

**Closure argument**: no source transition outside R7 §11 can cause a logical +1 (P1–P4) or a double −1 (single −1 authority + P8; reclaim is not a resolution, P5). □

---

## 5. Coalesce Hidden +1 Absence (R5-7 §5)

```
O1(C) + A2(C) + A3(C) + A4(C)
   lookup(C) ⇒ O1 found ⇒ A2/A3/A4 discarded
   ⇒ L = 1  (not 4)
```
If `O1.C != A2.C`: +1 occurs only after the `L < 32` guard (Admission lemma, §12-B). ∴ coalesce never injects a +1. □

---

## 6. Table-Full Boundary (R5-7 §6)

- **Case A** `L=31, new C` ⇒ `31 → 32` (guard `L<32` satisfied).
- **Case B** `L=32, new C` ⇒ **REJECT**, no `Created` generated, `L=32`.
- **Case C** `L=32, C matches O` ⇒ coalesce, `L=32`.

**Required order** (coalesce precedes full-check):
```
lookup(C)
  ├ existing ─▶ coalesce (ΔL=0)
  └ absent
        └ L==32 ? REJECT : admit (L<32 before +1)
```
∴ At L=32, no new obligation is ever created. □

---

## 7. Idempotency / Double −1 (R5-7 §7)

Guard (P8): `if O==null || O.state terminal: return`.

| Race | Sequence | Result |
|---|---|---|
| A | Published, dup Published | 1st→Completed(−1), 2nd no-op ⇒ **max 1** |
| B | Failed, ShutdownDiscard | whichever first→terminal(−1), other no-op ⇒ **max 1** |
| C | ShutdownDiscard, Failed | symmetric ⇒ **max 1** |
| D | StaleSuperseded, dup same id | 1st→terminal(−1), 2nd no-op ⇒ **max 1** |

∴ Every obligation loses at most one −1. □

---

## 8. Shutdown Proof (R5-7 §8)

```
shutdown:
  for each live O in RecoveryAdmissionTable:
      emit RecoveryResolution{O.obligationId, ShutdownDiscard}   // −1 (exactly once)
  // delivery cleanup (side effects, NO count change):
  discardRecoveryRequestsOnShutdown()   // drains transport queue
  discardPendingRecoveryAdmission()     // clears durable slot
  builder.stop()
```
Channel drains mutate `recoveryIntentQueue_` / `pendingRecoveryAdmission_` / `pendingIntentCount_` (P2,P3) — **never L**. Total −1 = #live obligations, not summed over channels ⇒ no double-count, no Building-leak (popped-but-building O is in the table, caught by iteration). □

---

## 9. Delivery vs Logical Independence (R5-7 §9)

- `Q_max=256`, `L_transport=256`, `L_durable=1`, `A_max≤257` are **delivery / trigger** capacities.
- `L ≤ 32` is a **direct resource bound** (INV-CAP-7, P9), not derived from them.
- By the inductive result (§12-G), at most 32 live obligations exist; each has ≤1 delivery entry (coalesce keeps one). So the 256 ring holds ≤32 corresponding entries + spare.

∴ `Q_max=256 ⇏ L=256`; `ring capacity 256 ⇏ L=256`; `A_max≤257 ⇏ L≤257`. 32 is enforced directly. □

---

## 10. `pendingIntentCount_` Exclusion (R5-7 §10)

All mutations are push/pop/rollback fetchAdd/fetchSub (P2). It mixes Observe/Quarantine/Recovery (h:550), excludes durable, and feeds `isFullyDrained`. It has **no unavoidable dependency** on L (L is a separate new counter, P1). ∴ It can remain as-is and is excluded from the proof. **No BLOCK.** □

---

## 11. Durable Transition Ambiguity (R5-7 §11)

`Building → Durable` is used for both **retry** (`settle(true)`) and **QueuePressure** (publish reject). Because `pendingRecoveryAdmission_` is a **single slot** (P3):

- **retry**: reuses the existing durable slot ⇒ no new entry, ΔL=0.
- **QueuePressure** (transport recovery, no durable slot yet): assigns the single slot ⇒ a 2nd slot is *impossible*, ΔL=0.
- A transport recovery in Building holds no durable slot; a durable recovery in Building holds the slot; both converge on the one slot.

∴ No double-occupancy, no double-delivery, no double-retry, no state-rollback +1. **Safe to merge ⇒ no ambiguity ⇒ R5-7-GATE stays PASS.** □

---

## 12. Proof Artifacts (R5-7 §12)

**A. State invariant**
```
I0:   0 ≤ L ≤ 32
```
**B. Admission lemma**
```
NewObligation  ⇒  L < 32  holds BEFORE the +1
```
**C. Coalesce lemma**
```
Coalesce  ⇒  ΔL = 0
```
**D. Delivery lemma**
```
Transport / Durable / Building / Retry transitions  ⇒  ΔL = 0
```
**E. Resolution lemma**
```
Terminal resolution (Completed/Failed/StaleSuperseded/Superseded/ShutdownDiscard)
           ⇒  ΔL = −1  exactly once
```
**F. Shutdown lemma**
```
Shutdown  ⇒  every live O terminates exactly once (table iteration)
```
**G. Inductive step**
```
Assume L ≤ 32 at state S. For every reachable transition t:
  • unique admission      : only fires if L<32 ⇒ L' = L+1 ≤ 32
                            (if L==32, admission is REJECTED ⇒ no transition)
  • coalesce / delivery / retry / reject : ΔL = 0 ⇒ L' = L ≤ 32
  • completion / supersede / shutdown    : ΔL = −1 ⇒ L' = L−1 ≤ 32 (≥0 by E/F)
∴ L' ≤ 32 in all cases.
```
**H. Conclusion**
```
Base: system starts empty ⇒ L = 0 ≤ 32.
Step: by G, every transition preserves L ≤ 32.
∴ ∀ reachable S : L(S) ≤ 32.   ∎
```

---

## 13. Gate (R5-7 §13)

| Condition | Result |
|---|---|
| +1 authority unique | ✅ admission only (P1–P2) |
| −1 authority unique | ✅ single completion authority (P6) |
| Δcount of all transitions fixed | ✅ §4 |
| no +1 at table-full | ✅ §6 Case B |
| no +1 at coalesce | ✅ §5 |
| retry count-neutral | ✅ §11 |
| publication failure no leak | ✅ §3 (Failed/StaleSuperseded −1; QueuePressure retry) |
| shutdown exactly-once | ✅ §8 |
| duplicate resolution idempotent | ✅ §7 |
| delivery ↔ logical independent | ✅ §9 |
| `pendingIntentCount_` not proof-dependent | ✅ §10 |
| Durable transition unambiguous | ✅ §11 |

⇒ **R5-7-GATE = PASS**. Proceed to **D105-R5-8** (Implementation Plan).

---

## 14. Traceability / Tools

- **WSL rg/sed/rtk** (ctx_batch_execute): `pendingIntentCount_` mutations; `liveLogicalRecoveryObligationCount` 0-hit in src; single durable slot; `reclaimSlot` non-count (this audit).
- **AiDex** (R5/R6/R7): `pendingIntentCount_` = 42 hits; `ObligationId` = 0 hits in source.
- **semble** (R5/R6/R7): `RecoveryResolution` / `liveLogicalRecoveryObligationCount` absent from source; only the durable-overwrite site returns.
- **cocoindex** (R5/R6): surfaced I4 **D25 INV-CAP-7** (32 = deliberate bound) — the proof premise P9.
- **graphify** (R5/R6/R7): no `LogicalRecoveryObligationId` / `RecoveryResolution` nodes exist.
- **serena MCP**: language server timed out (corroborated by the above).

**No source was modified.**
