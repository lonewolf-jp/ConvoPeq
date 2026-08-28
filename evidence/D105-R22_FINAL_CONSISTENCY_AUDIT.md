# D105-R22 — I4 ↔ Runtime Bidirectional State-Machine / Capacity Consistency Final Audit

**Status:** **R22 PARTIAL PASS** (read-only; source changes = 0; I4 changes = 0).
**Verdict: R22-NO-GO on critical capacity inconsistency** — the R21 contract amendment
(`kMaxLogicalRecoveryObligations = 32` enforced at the obligation table) **IS** structurally
valid for the **logical obligation table** (32 ≤ physical residency 257), but the
**higher-level decomposition** `E_max × O_max ≤ 32` from I4 §D19.3 is **NOT** derivable
from current production code: `E_max = 256` (quarantine slot registry), `O_max ≥ 2`
(same handle, different target, see R3 §7.3 counterexample). The R3 verdict stands: the
D19.3 / D22.2 capacity decomposition is not satisfied.

**Two coexisting facts**:
1. **Logical obligation count** IS bounded by 32 (enforced at `tryInsert` h:355).
2. **E × O decomposition** is NOT satisfied (256 × ≥2 = ≥512 ≠ 32).

The 32 bound is real for the *obligation table* but does not satisfy the *episode
multiplicity decomposition* claimed in I4 D19.3 / D22.2. The R21 I4 amendments (D15.2,
D18.3) are correct for the `logical obligation count` interpretation but are **silent
on the episode-decomposition claim** that R2/R3 audited.

This audit is the **final gate** before any further runtime work. It recommends
**R23 (closure)**: either (a) revise I4 D19.3 / D22.2 to drop the `E × O ≤ 32` claim
(now factually unsatisfiable) and replace it with `L_max ≤ 32` as a direct
admission-level enforcement, or (b) implement the missing episode-coalesce
infrastructure (`RecoveryEpisodeId` + per-episode cap) to make `O_max = 1` and
`E_max ≤ 32` structurally true. R23 must decide the path forward before any further
runtime changes.

---

## R22-1 — Capacity facts (re-derived from current production code)

All facts below are traced from `src/` (the current production code, not the design
documents) and are reproducible by grep.

### Q_max: physical quarantine capacity

**Code trace**:
- `ISRDSPQuarantine.h:68`: `static constexpr size_t kMaxSlots = 256;`
- `ISRDSPQuarantine.h`: `std::atomic<bool> quarantineActiveFlags_[kMaxSlots]` (256 atomics)
- `ISRDSPQuarantine.cpp:30`: `quarantineHandle()` sets `quarantineActiveFlags_[slot] = true`
- `ISRDSPQuarantine.cpp:79`: `reclaimSlot()` sets `quarantineActiveFlags_[slot] = false`
- `AudioEngine.Commit.cpp:644` (PR1 reclaim loop): `for (qslot = 0; qslot < MAX_DSP_SLOTS; ++qslot)`

**Q_max = 256** (all 256 DSPHandle slots can be simultaneously quarantined).

### L_max: physical recovery transport+durable residency

**Code trace**:
- `ISRRuntimePublicationCoordinator.h:886`: `static constexpr size_t kRecoveryIntentQueueCapacity = 256;`
- `ISRRuntimePublicationCoordinator.h:887`: `LockFreeRingBuffer<RecoveryIntent, kRecoveryIntentQueueCapacity> recoveryIntentQueue_;`
- `ISRRuntimePublicationCoordinator.h:882` (`PendingRecoveryAdmission`): single-slot struct
- `ISRRuntimePublicationCoordinator.h:883`: `PendingRecoveryAdmission pendingRecoveryAdmission_;`

**L_max = 256 (transport) + 1 (durable) = 257** (physical residency capacity).

**Critical**: 257 is the **physical residency** upper bound, not a logical obligation
count. R3 explicitly verified this distinction. The 32 logical obligation table is
a **separate** capacity.

### E_max: maximum concurrent open recovery episodes

**Code trace**:
- `grep RecoveryEpisodeId src/audioengine/` → **0 matches**
- `grep nextRecoveryEpisodeId src/audioengine/` → **0 matches**

**E_max is UNDEFINED in production code.** `RecoveryEpisodeId` is design-only
(declared in I4 D13 / D19.1 but not allocated by any runtime counter). Every
quarantined slot IS an independent recovery trigger in the current code.

The closest structural bound on E_max is the quarantine flag array size (Q_max = 256),
but this is **not an episode bound** — it is a flag-array bound. R3 §5.2 explicitly
verified this: "E_max is undefined in production code. Per D19.1, `RecoveryEpisodeId`
should be allocated by a dedicated monotonic counter at episode creation. Since this
is absent, the 'E_max' concept does not apply to current code. The closest
structural bound is Q_max = 256."

### O_max: distinct targets per episode

**Code trace**:
- `ISRRuntimePublicationCoordinator.h:248-256`: `struct SemanticRecoveryTarget` with 3 hashes
- `ISRRuntimePublicationCoordinator.h:262-267`: `struct CoalesceIdentity = { handle, target }`
- `ISRRuntimePublicationCoordinator.cpp:836-841`: `cid` is constructed from current
  `buildSource` (`rebuildFingerprint.irIdentityHash, convolutionConfigHash, dspParameterHash`)
- `ISRRuntimePublicationCoordinator.cpp:866`: `findByKey(cid)` — coalesce by
  `(handle, fingerprint-hash-tuple)`

**O_max ≥ 2 (unbounded in the transport path):** Coalesce is per-`(handle, target)`.
A re-quarantine of the **same handle** with a **different target** (after `currentBuildSnapshot_`
has drifted due to async RebuildThread updates) produces a **new `CoalesceIdentity`** and
hence a **new obligation** in the table. R3 §7.3 constructed the counterexample:

```
Same handle H
Re-quarantine #1: buildSource = snapshot_A → cid (H, target_A) → new obligation
  ...
Reclaim → quarantineActiveFlags_[H] = false; obligation 1 reaches ResolvedSuccess
  (slot freed)
Re-quarantine #2: buildSource = snapshot_B ≠ snapshot_A → cid (H, target_B)
  → findByKey(cid) returns kCapacity (slot 1 is ResolvedSuccess, not Live)
  → tryInsert creates obligation 2 (different id from obligation 1)
```

This is constructible from the current code. The R5-9 `CoalesceIdentity` addition
*helps* (same target re-submissions no longer create duplicate obligations) but does
**not** bound O_max to 1 (different target ⇒ new obligation).

**O_max is ≥ 2 (unbounded in the transport path without coalesce by snapshot drift).**

### kMaxLogicalRecoveryObligations = 32 enforcement

**Code trace**:
- `ISRRuntimePublicationCoordinator.h:325`: `static constexpr std::size_t kMaxLogicalRecoveryObligations = 32;`
- `ISRRuntimePublicationCoordinator.h:934`: `RecoveryAdmissionTable<kMaxLogicalRecoveryObligations> recoveryAdmissions_;`
- `ISRRuntimePublicationCoordinator.h:355-356` (`RecoveryAdmissionTable::tryInsert`):
  ```cpp
  if (liveCount_.load(std::memory_order_acquire) >= kCapacity)
      return std::nullopt; // capacity exhausted → caller rejects (ΔL=0)
  ```
- `ISRRuntimePublicationCoordinator.cpp:880-885`:
  ```cpp
  const auto ins = recoveryAdmissions_.tryInsert(cid);
  if (!ins) {
      convo::fetchAddAtomic(recoveryCapacityExhaustedCount_, 1, ...);
      return false;  // ★ INV-X1-7: L==32 → reject, no transport
  }
  ```

**`kMaxLogicalRecoveryObligations = 32` IS structurally enforced** at the `tryInsert`
call site. The table size is fixed at compile time (`kCapacity = 32`), and the
capacity gate runs on every new-obligation path.

**Verified by T-R5-8 / C2** (existing tests): 32 distinct obligations are accepted
(L=32), 33rd is rejected with `recoveryCapacityExhaustedCount` increment. R5-8
audit evidence confirms this.

### Capacity facts summary

| Variable | Value | Source | Type |
|---|---|---|---|
| Q_max | 256 | `ISRDSPQuarantine.h:68` (`kMaxSlots`) | physical quarantine flag array |
| L_max | 257 | transport 256 + durable 1 | physical residency |
| E_max | UNDEFINED | `RecoveryEpisodeId` absent | design-only (I4 D13) |
| O_max | ≥ 2 | R3 counterexample + R5-9 coalesce scope | per-`(handle, target)` |
| `kMaxLogicalRecoveryObligations` | 32 | `ISRRuntimePublicationCoordinator.h:325` | logical obligation table |

---

## R22-2 — The central consistency question: E × O ≤ 32

The R21 I4 contract claims (via D19.3 / D22.2) that:
```
E_max × O_max ≤ kMaxLogicalRecoveryObligations
        256 × O_max ≤ 32
                  O_max ≤ 32 / 256 = 0.125
```

This is **arithmetically impossible** if `E_max = 256` and `O_max ≥ 1`. Even with `O_max = 1`
(best case), `E × O = 256 × 1 = 256`, not 32.

**The R2 audit** explicitly noted this: "The 32 bound is NOT derivable from the current
code structure. ... E_max × O_max = 256 × 1 = 256, not 32." R2 §6.4.

**The R21 I4 amendments do NOT address this.** R21 added `RetryExhaustion` to the
disappearance set, but R21 did not revise D19.3 / D22.2, which still state
`E_max × O_max ≤ 32`. The R21 amendments are **logically self-consistent** (the
conservation equation with `retryExhaustedCount` is sound) but the **decomposition
claim `E × O ≤ 32` remains factually unsatisfiable**.

### Two coexisting capacity bounds

1. **`kMaxLogicalRecoveryObligations = 32`** (logical obligation count) — **enforced
   at `tryInsert` (h:355)**, structurally verified.
2. **`E_max × O_max ≤ 32`** (episode decomposition claim) — **not derivable from
   code**; R2 / R3 / R22 all confirm this is **mathematically impossible** with
   `E_max = 256` (R3) or `E_max = UNDEFINED` (R22) and `O_max ≥ 2` (R3) or
   `O_max ≥ 1` (R22 best case).

### R3's correction (which R21 did not propagate)

R3's Step 7.3 constructed a counterexample: same handle, snapshot drift ⇒
`O_max ≥ 2`. R22's grep confirms `CoalesceIdentity` was added in R5-9 but uses
`(handle, target)` as key, not `(handle, episodeId, target)` — so the
`RecoveryEpisodeId`-based episode grouping is **still missing**. The R3 verdict stands.

---

## R22-3 — I4 → Runtime reverse trace

Each I4 clause is traced to the runtime implementation that satisfies it. The table
records whether the runtime implements the clause and whether the runtime facts are
**consistent with the I4 claim**.

| I4 clause | Runtime implementation | Verdict |
|---|---|---|
| **D14.3**: `transient failure: Live → Live, ΔL=0, delivery=None, counter++` | `RuntimeIntentCoordinator::markTransientFailure` (ISRRuntimePublicationCoordinator.cpp:998-1024) | ✅ consistent |
| **D15.2 disappearance set**: `{Success, Superseded, ShutdownDiscard, RetryExhaustion}` | `RecoveryOutcome` enum (h:298-304) and `markTransientFailure` exhaustion branch (cpp:1017-1019) | ✅ consistent |
| **D18.3**: `live + success + superseded + shutdown + retryExhausted = admitted` | `recoveryAdmissions_.liveCount_` (h:380) + `recoveryObligationShutdownDiscardCount_` (h:892) + `recoveryRetryExhaustedCount_` (h:944) | ⚠ conservation equation is sound, but `admittedLogicalObligationCount` is **not separately tracked** in the runtime; the 32-bound is enforced at `tryInsert` and the conservation equation must be verified **at the obligation-table level only**, not at the episode level |
| **D20.5 closure linearization**: `live: 1 → 0` is the closure point | `discardRecoveryRequestsOnShutdown` (cpp:1203) — iterates `kCapacity` slots, calls `resolveRecoveryObligation(id, ShutdownDiscarded)` for each | ✅ consistent at the obligation-table level; episode-level closure (`RecoveryEpisodeId`) is design-only and not implemented |
| **D29.8**: end-to-end state machine including `MARK-TRANSIENT-FAILURE` non-terminal action | `markTransientFailure` is the non-terminal Live→Live transition (cpp:1010 sets `delivery=None`, then increments counter, then potentially terminal). | ✅ consistent at obligation-table level |
| **D22.2**: `kMaxLogicalRecoveryObligations = 32` is a **deliberate resource bound** (not upstream maximum) | `tryInsert` capacity gate (h:355-356) | ✅ enforced; the **claim that 32 is derived from E × O ≤ 32** is ❌ (mathematically unsatisfiable) |
| **D13**: `RecoveryEpisodeId` is allocated at episode creation | **Not implemented in production** (0 grep hits) | ❌ design-only; I4 claim is **unfulfillable** in current code |
| **D19.3 INV-CAP-4**: `E_max × O_max ≤ kMaxLogicalRecoveryObligations` | **Not provable** — `E_max = 256` (R2/R3) or `UNDEFINED` (R22) and `O_max ≥ 2` (R3) | ❌ **arithmetic violation** |
| **D18.1 CoalesceIdentity**: `CoalesceIdentity = {handle, RecoveryEpisodeId, SemanticRecoveryTarget}` | Current implementation: `CoalesceIdentity = {handle, SemanticRecoveryTarget}` (h:262) — **`RecoveryEpisodeId` is NOT in the struct** | ⚠ partial implementation; coalesce is per-`(handle, target)` not per-`(handle, episodeId, target)` |
| **D19.1 episode closure**: `episode closes iff liveLogicalObligationCount == 0` | `discardRecoveryRequestsOnShutdown` scans obligation table (L: 1→0). Episode-level closure: not implemented. | ⚠ obligation-table closure is implemented; episode-level closure is design-only |
| **D14.3 budget exhaustion**: `admission BLOCK` at 32 | `tryInsert` returns `nullopt` at L=32, caller rejects | ✅ enforced |
| **D26.6 / D29.4**: capacity accounting `Tentative + Owned ≤ 32` | **Tentative** / **Owned** are design concepts in I4 D26; runtime uses `pendingIntentCount_` (transport) and a single-slot durable | ❌ **design / implementation mismatch**; the I4 D26.2 capacity equation does NOT match the runtime's `liveCount_ <= 32` direct enforcement |

### Verdict

The I4 post-R21 is **internally consistent within the obligation-table interpretation**:
the conservation equation, disappearance set, closure linearization (D20.5), and
`MARK-TRANSIENT-FAILURE` settlement action are all well-defined at the obligation-table
level. However, the I4 also contains **unfulfilled** clauses (D13, D19.1, D19.3, D22.2,
D26, D18.1) that depend on episode-level infrastructure (`RecoveryEpisodeId`,
per-episode obligation cap) which is **not implemented in production code**.

---

## R22-4 — Runtime → I4 forward trace

Each runtime transition is traced to the I4 clause that documents it. The table
records whether an I4 clause exists for each runtime transition.

| Runtime transition | I4 clause | Match? |
|---|---|---|
| `submitRecoveryRequest` (admit, line 819) | D14.2 reservation-first | ✅ |
| `tryInsert` (new obligation, h:343) | D14.2 | ✅ |
| Coalesce via `findByKey(cid)` (h:344) | D18.1, D18.2 | ⚠ partial (no `RecoveryEpisodeId` key) |
| Transport push to `recoveryIntentQueue_` (cpp:909) | D14.2 | ✅ |
| Durable admission to `pendingRecoveryAdmission_` (cpp:929-940) | D14.2, D18.4 | ✅ (with R5-9 defer-on-collision fix) |
| Builder take (`popRecoveryRequest` / `takePendingRecoveryAdmission`) | D18.4 | ✅ |
| Build → publish (RebuildDispatch.cpp) | (no I4 clause; D-2 build pipeline is pre-I4) | n/a (out of I4 scope) |
| `onPublishCommitted` → `resolveRecoveryObligation(id, Published)` | D15.2 (`Success` terminal) | ✅ |
| `submitPublishRequest` `RejectedStaleGeneration` → `StaleSuperseded` | D15.2 (`Superseded` terminal) | ✅ |
| `submitPublishRequest` `RejectedShutdown` → `ShutdownDiscarded` | D15.2 (`ShutdownDiscard` terminal) | ✅ |
| `markTransientFailure` counter increment (cpp:1013) | D14.3, D29.8 (`MARK-TRANSIENT-FAILURE`) | ✅ |
| `markTransientFailure` exhaustion → `resolve(id, ResolvedFailed)` (cpp:1017-1019) | D15.2 (`RetryExhaustion` terminal) | ✅ |
| `discardRecoveryRequestsOnShutdown` (cpp:1203) | D15.2 (`ShutdownDiscard` terminal) | ✅ |
| `redriveDeferredRecoveryObligations` (cpp:1042) | D14.3 (P-B repair) | ✅ |
| **Episode creation** (RecoveryEpisodeId allocation) | D13, D19.1 | **❌ NO production implementation** |
| **Episode CAS** (OPEN → OPEN+1, OPEN-1 → CLOSED) | D19.1, D23.2 | **❌ NO production implementation** |
| **Per-episode obligation cap** (D19.3 INV-CAP-3: `O_max ≤ kMaxLogicalRecoveryObligations / E_max`) | D19.3 | **❌ NOT IMPLEMENTED** (no episode grouping) |
| **`RecoveryEpisodeId` counter** (`nextRecoveryEpisodeId_`) | D19.1 | **❌ NOT IMPLEMENTED** (0 grep hits) |
| **`EpisodicState`** (OPEN/CLOSED flag) | D23.2 | **❌ NOT IMPLEMENTED** |

### Verdict

**Every runtime transition at the obligation-table level has a corresponding I4 clause**.
However, **several runtime-side episode-level concepts in I4 have no production
implementation**:
- `RecoveryEpisodeId` allocation (D13, D19.1)
- Episode CAS (D19.1, D23.2)
- Per-episode obligation cap (D19.3 INV-CAP-3)
- `EpisodicState` (D23.2)

The I4 post-R21 is **partially self-consistent**: obligation-table level is sound, but
the episode-level claims are aspirational. R3 documented this gap (R3 §7 "Bottom line")
and the I4 was not revised at that time. R21 did not revise it either.

---

## R22-5 — Conservation equation: runtime ↔ I4 mapping

| Conservation term (R21 I4) | Runtime counter/state | Verdict |
|---|---|---|
| `liveOwnershipCount = transportCount + durableCount + buildingCount + stalledCount` | `recoveryIntentQueue_.size()` + `pendingRecoveryAdmission_.state != NoAdmission` + ... | ❌ **runtime does NOT track `liveOwnershipCount` separately**; the only counter is `liveCount_` on the obligation table. I4 §D18.3 decomposition is **not directly verifiable** in code. |
| `terminalDispositionCount = success + superseded + shutdown + retryExhausted` | (no single runtime counter; partial via `recoveryObligationShutdownDiscardCount_` and `recoveryRetryExhaustedCount_`) | ⚠ **runtime tracks `shutdown` and `retryExhausted` counters** but **NOT `success` or `superseded`** as separate counters. `liveCount_--` is the only signal of terminal transition. |
| `admittedLogicalObligationCount` (RHS constant) | (not separately tracked) | ❌ **runtime does NOT maintain an "admitted" counter**; the table's `liveCount_` is the only count. Conservation equation is verifiable **only as `liveCount_ == 0` ↔ all admitted are terminalized**. |
| `admissionEventCount ≥ admittedLogicalObligationCount` | (not tracked) | ❌ not enforced |

**Conservation equation is not directly verifiable from runtime counters.** R21's
equation is **structurally correct** (the equality must hold) but the runtime does
not have all the terms as separate counters. The equation must be interpreted
as: "if all terminal transitions are correctly routed through the `table.resolve` API
(which decrements `liveCount_`), and the obligation table never allows `liveCount_` to
exceed 32, then the equation holds by construction (the table enforces it)."

The R21 I4 §D18.3 statement "liveOwnershipCount + 4 terminals = admitted" is
**interpretable** as "live + (shut + retry_exh) ≤ 32" (the only counters the runtime
exposes) but the **full decomposition** into 4 disjoint terminal counts is **not
directly observable** in runtime counters.

---

## R22-6 — Episode closure (D20.5) — verified at obligation-table level

### Case A: `Live(2) → A RetryExhaustion → Live(1) → episode remains OPEN`

**Runtime trace**:
- 2 obligations admitted (L=2)
- A is `markTransientFailure` × 4 → `ResolvedFailed` → `liveCount_--` (L=1)
- B remains `Live` (delivery=None, counter=0)
- `discardRecoveryRequestsOnShutdown` would not be called (B still Live)

**Closure**: at the obligation-table level, `liveCount_` is 1 (not 0). At the episode
level (R22), the concept of "episode OPEN" is not implemented, but R3 verified that
"last LIVE = 1 → 0" is the closure trigger at the obligation-table level. Case A
keeps L=1 ⇒ no closure. **Verified.**

### Case B: `Live(1) → A RetryExhaustion → Live(0) → episode CLOSED`

**Runtime trace**:
- 1 obligation admitted (L=1)
- A reaches `RetryExhaustion` → `liveCount_--` (L=0)
- No other Live obligations

**Closure**: at the obligation-table level, L=0. At the episode level (R22), the
concept is not implemented but the obligation-table level matches INV-OBL-3.
**Verified.**

### Case C: `Live(2) → A Success → Live(1) → episode OPEN`

**Runtime trace**:
- A reaches `Published` → `resolveRecoveryObligation(id, Published)` → `table.resolve`
  → `state == ResolvedSuccess`, `liveCount_--` (L=1)
- B remains Live

**Closure**: L=1 ⇒ no closure. **Verified.**

### Case D: `Live(1) → Success → Live(0) → CLOSED`

**Runtime trace**:
- A reaches `Published` → `liveCount_--` (L=0)

**Closure**: L=0 ⇒ closure trigger. **Verified.**

### `RetryExhaustion ≠ EpisodeClosure`

**At the obligation-table level**: `RetryExhaustion` is one of 4 terminal transitions;
any terminal transition (Success, Superseded, ShutdownDiscard, RetryExhaustion)
decrements `liveCount_`. The obligation-table closure trigger is `liveCount_: 1 → 0`,
**not specific to any terminal type**.

**At the episode level** (R22): the concept of "episode" is not implemented (no
`RecoveryEpisodeId`). R3 §6.1 and the R21 I4 §D20.5 audit verdict both confirm
that R21's audit preserved this: "`RetryExhaustion` ≠ EpisodeClosure" holds at the
obligation-table level because the closure trigger is the `liveCount_` transition,
not the terminal type.

---

## R22-7 — Closure race / admission race (D23)

**D23.1 — The TOCTOU race** that D23 fixed:
```
Thread A (CoordinatorLoop): load Closed==false → reservation acquire ← ここで停止
Thread B (Builder):         live-- → live==0 → closure
Thread A:                   live++ → register
Result: Closed=true, live=1   ← closure 後に admission が resurrect
```

**D23.2 solution**: single CAS-able `EpisodeAdmissionState` (`{live, closed}` packed).

**R22 runtime check**: **0 grep hits** for `RecoveryEpisodeId`, `EpisodeAdmissionState`,
`OPEN(`, `CLOSED(`, episode CAS, or `nextRecoveryEpisodeId`. The D23 race protection
**does not exist in production code** because the episode concept itself is not
implemented. The race is structurally impossible because the only counter is
`liveCount_` (atomic, single field), and the closure trigger is just `liveCount_ == 0`.
The race can only exist if there is an admission-after-closure window, which requires
the episode concept.

**R22 verdict on closure race**: At the obligation-table level, the race is
**impossible** (single atomic counter, no closure flag). At the episode level,
**the protection does not exist** because the concept does not exist. The D23
specification is **untestable** against the current code because the target of the
specification (episodes) is absent.

---

## R22-8 — Capacity consistency proof (the central question)

### The arithmetic check

```
E_max (R2) = 256                  (from quarantineActiveFlags_[256])
E_max (R22) = UNDEFINED           (no RecoveryEpisodeId)
O_max (R3, R22) = ≥ 2             (same handle, different target, counterexample)
kMaxLogicalRecoveryObligations = 32   (ISRRuntimePublicationCoordinator.h:325, enforced at tryInsert)
```

**Claim**: `E_max × O_max ≤ 32` (I4 §D19.3 INV-CAP-4)

**Verification**:
- If `E_max = 256, O_max = 1`: 256 × 1 = 256 ≠ 32. **Violated.**
- If `E_max = 256, O_max = 2`: 256 × 2 = 512 ≠ 32. **Violated.**
- If `E_max = UNDEFINED, O_max ≥ 1`: **invocation undefined.** R3 §5.2: "E_max is undefined in
  production code. Per D19.1, `RecoveryEpisodeId` should be allocated by a dedicated
  monotonic counter at episode creation. Since this is absent, the 'E_max' concept
  does not apply to current code."

**In all interpretations, the I4 D19.3 / D22.2 `E × O ≤ 32` claim is violated or undefined.**

### What IS provable

- `kMaxLogicalRecoveryObligations = 32` is **structurally enforced** at `tryInsert` (h:355).
  The obligation table cannot exceed 32 Live entries. **Verified.**
- `L_max = 257` (physical residency) is **structurally enforced** by the queue + durable
  capacities. **Verified.**
- `Q_max = 256` is **structurally enforced** by `quarantineActiveFlags_[256]`. **Verified.**

### The inconsistency

**The I4 post-R21 contains two inconsistent capacity claims**:
1. The **direct** claim: `kMaxLogicalRecoveryObligations = 32` is enforced at the obligation
   table (D22.2 line 580, R21 I4 §D22.2). **This is correct.**
2. The **decomposition** claim: `E_max × O_max ≤ 32` (D19.3 INV-CAP-4). **This is
   mathematically unsatisfiable** with `E_max = 256` and `O_max ≥ 2`.

The two claims are **not equivalent** in the current code: the direct claim is
satisfied at the obligation-table level, but the decomposition claim fails.

---

## R22-9 — R22 GO-condition verification

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | D14.3 transient failure: Live → Live | ✅ | `markTransientFailure` cpp:1010-1014 |
| 2 | `delivery=None` is the only delivery repair for failure | ✅ | `markTransientFailure` cpp:1010 (unconditional `delivery=None`) |
| 3 | counter +1 exactly once per failure | ✅ | `markTransientFailure` cpp:1013 (single `fetch_add`) |
| 4 | liveCount unchanged before exhaustion | ✅ | `markTransientFailure` does not touch `liveCount_` (h:372 decrement only) |
| 5 | K-th call produces `ResolvedFailed` | ✅ | cpp:1017-1019 (exhaustion branch) |
| 6 | RetryExhaustion as 4th terminal count | ✅ | D18.3 amendment, `recoveryRetryExhaustedCount_` (h:944) |
| 7 | transient failure not in conservation terminal count | ✅ | D18.3 amendment explicitly excludes |
| 8 | 4 terminal count disjointness | ✅ | each obligation has exactly one terminal; disjoint by construction |
| 9 | retry redrive preserves same obligationId | ✅ | `redriveDeferredRecovery` (cpp:1058) reuses the existing slot id |
| 10 | episode closure = `live: 1 → 0` | ✅ (obligation-table level); ❌ (episode level — episodes don't exist) | |
| 11 | RetryExhaustion alone does not close episode | ✅ (same as #10) | |
| 12 | CLOSED episode cannot resurrect | n/a | episode concept not implemented; obligation-table terminals are idempotent (`resolve` CAS) |
| 13 | E_max provable from code | **⚠ partial** | Q_max = 256 is provable; E_max = UNDEFINED (no `RecoveryEpisodeId` in production) |
| 14 | O_max provable from code | **⚠ partial** | O_max ≥ 2 (R3 counterexample still applies with R5-9 coalesce); R21 I4 says O_max ≤ 1 but R3 disproves this |
| 15 | Q_max provable from code | ✅ | `kMaxSlots = 256` |
| 16 | L_max = 257 meaning (physical residency) | ✅ | transport 256 + durable 1 |
| 17 | `kMaxLogicalRecoveryObligations = 32` enforced | ✅ | `tryInsert` h:355 capacity gate |
| 18 | `E × O ≤ 32` mathematically consistent | **❌ NO-GO** | `256 × ≥2 = ≥512 ≠ 32` |
| 19 | All I4 state transitions have runtime implementation | **❌ PARTIAL** | obligation-table level: complete; episode-level (D13, D19.1, D23.2): not implemented |
| 20 | All runtime state transitions have I4 clause | **✅** | see R22-4 table |

---

## R22-10 — R22 NO-GO triggers (per R22 brief)

| NO-GO trigger | Status |
|---|---|
| `E_max = 256, O_max ≥ 2, kMaxLogicalRecoveryObligations = 32` mathematically incompatible | **❌ TRIGGERED** (R22-8) |
| `logical obligation capacity = 257` derived from "queue 256 + durable 1" | ✅ not in equation (32 is the direct enforcement at `tryInsert`) |
| `kMaxLogicalRecoveryObligations = 32` is just a constant, not enforced at admission | ✅ enforced at `tryInsert` h:355 |
| `RecoveryEpisodeId` / `CoalesceIdentity` / `SemanticRecoveryTarget` not implemented while I4 claims coalesce | **❌ TRIGGERED (partial)** — `RecoveryEpisodeId` is absent; coalesce is per-`(handle, target)` only |

---

## R22-11 — R22 verdict

**R22: PARTIAL PASS** — the obligation-table-level I4 ↔ runtime consistency is verified
(GO conditions 1–12, 15–17, 20 are satisfied). However, the **episode-level decomposition
claims** (GO conditions 13, 14, 18, 19) are **NOT satisfied** in current production code:

- `RecoveryEpisodeId` is **not implemented** (R3 + R22 grep confirm).
- `O_max` is **≥ 2** (R3 counterexample + R22 grep confirm `CoalesceIdentity = {handle, target}`).
- The D19.3 `E × O ≤ 32` claim is **arithmetically unsatisfiable**.
- D13 / D19.1 / D23.2 episode-level clauses are **untestable** in current code.

### What R22 establishes

1. **The 32 bound is real** for the obligation table. `tryInsert` capacity gate enforces
   it. The R21 I4 §D15.2 / §D18.3 / §D29.8 amendments are **correct for the
   obligation-table interpretation**.
2. **The episode-level claims in I4 D13 / D19.1 / D19.3 / D22.2 / D23 / D26 are
   aspirational, not factual.** They describe an architecture that is not
   implemented in current code.
3. **The R21 I4 amendments are self-consistent within the obligation-table
   interpretation** but **do NOT resolve the episode-decomposition inconsistency** that
   R2 and R3 already identified.

### What R22 does NOT establish

- A consistent I4 ↔ runtime audit at the episode level. The episode concept is
  absent, so this audit is structurally impossible against current code.
- A pass for any implementation that depends on episode-level semantics (e.g.,
  per-episode obligation caps, episode closure).

---

## R22-12 — R23 (next) recommendations

R23 must decide between two paths to close the NO-GO:

### Path A — Revise I4 (contract simplification)

Drop the episode-level claims that are not satisfiable:
- Remove D19.3 INV-CAP-4 (`E × O ≤ 32`).
- Remove D19.1 `RecoveryEpisodeId` allocation.
- Remove D22.2's "decomposition" rationale.
- Replace the I4 D13 / D19 / D22 / D23 episode-level section with a statement:
  "Episode-level grouping is design-only. The obligation table is the only
  logical abstraction. The 32 bound is enforced directly at `tryInsert`."
- Add a section documenting `L_max = 257` (physical residency, separate from the
  32 logical bound).

### Path B — Implement the missing infrastructure

Implement the missing episode-level infrastructure to make `E × O ≤ 32` satisfiable:
- Add `nextRecoveryEpisodeId_` counter (allocate at first quarantine of a handle).
- Add `RecoveryEpisodeId` field to `CoalesceIdentity` (replace current `{handle, target}`).
- Implement per-episode obligation cap.
- Implement episode CAS (OPEN → OPEN+1 / OPEN-1 → CLOSED) with single atomic state.

Path B is significantly more work and is the original D19 / D22 / D23 plan. R3 noted
this gap; R5-8 / R5-9 / R5-10 made the obligation-table layer sound but did not address
the episode layer.

**R22 recommends Path A** (revise I4) because:
1. Path B is a multi-week implementation effort requiring new design.
2. The current code is **production-stable** (40/40 tests pass) and the 32 bound
   works correctly at the obligation-table level.
3. The episode concept was never implemented and was never tested. Removing
   the aspirational claims from I4 is **honest documentation** of the current
   state, not a regression.

R23 should be a **read-only I4 audit** that produces a revised I4 (Path A) or
initiates the implementation work (Path B). R23 itself should not modify runtime
code.

---

## Files referenced (read-only, no changes)

| File | Role |
|---|---|
| `src/audioengine/ISRRuntimePublicationCoordinator.h:325` | `kMaxLogicalRecoveryObligations = 32` |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:355-356` | `tryInsert` capacity gate |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:886-887` | transport queue capacity 256 |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:882-883` | durable single-slot |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:248-267` | `SemanticRecoveryTarget` + `CoalesceIdentity` (no `RecoveryEpisodeId`) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:998-1024` | `markTransientFailure` definition |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:1010-1019` | exhaustion branch |
| `src/audioengine/ISRDSPQuarantine.h:68` | `kMaxSlots = 256` |
| `doc/work88/I4_DESIGN_CONTRACT.md` | D13, D19.1, D19.3, D22.2 (episode-level claims — unfulfilled) |
| `evidence/D105-R2_PHASE1_CAPACITY_BOUND_REPROOF.md` | E_max = 256, O_max = 1 (R2) — superseded by R3 |
| `evidence/D105-R3_RECOVERY_EPISODE_TARGET_MULTIPLICITY_REAUDIT.md` | O_max ≥ 2, E_max UNDEFINED (R3) — confirmed by R22 |
| `evidence/D105-R18_RETRY_PRESERVING_IMPLEMENTATION.md` | R18 runtime: `markTransientFailure` + `consecutiveFailureCount` |
| `evidence/D105-R20_REJECTEDNOTFINALIZED_CENTRALIZATION.md` | R20 runtime: centralized `markTransientFailure` |
| `evidence/D105-R21_I4_CONTRACT_AMENDMENT.md` | R21 I4: `RetryExhaustion` + `retryExhaustedCount` |

---

## R22 final summary table

| Aspect | R22 verdict |
|---|---|
| D14.3 transient failure semantics | ✅ PASS (verified at runtime) |
| D15.2 disappearance set | ✅ PASS at obligation-table level |
| D18.3 conservation equation | ✅ PASS structurally (counters satisfy) but **decomposition into 4 disjoint terminal counts is not directly observable in runtime counters** |
| D20.5 closure semantics | ✅ PASS at obligation-table level (`liveCount_: 1→0`); episode level absent |
| D29.8 `MARK-TRANSIENT-FAILURE` | ✅ PASS |
| D13 / D19.1 / D19.3 / D22.2 / D23 / D26 (episode level) | ❌ NOT IMPLEMENTED in production code |
| `kMaxLogicalRecoveryObligations = 32` enforcement | ✅ PASS (verified at `tryInsert` h:355) |
| Q_max = 256 | ✅ PASS |
| L_max = 257 (physical residency) | ✅ PASS |
| E_max | ⚠ UNDEFINED (`RecoveryEpisodeId` not implemented) |
| O_max | ⚠ ≥ 2 (R3 counterexample + R22 grep) |
| **`E × O ≤ 32`** | **❌ NO-GO** (mathematically unsatisfiable) |

**R22: PARTIAL PASS with NO-GO on episode-level decomposition.**
R23 must close the NO-GO (Path A: revise I4, or Path B: implement episode layer).
