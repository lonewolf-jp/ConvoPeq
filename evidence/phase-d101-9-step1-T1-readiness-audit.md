# Phase I-T1 Implementation Readiness Audit — 4-system code match

> Date: 2026-08-22
> Source snapshot: **96816 lines**, `> Generated: 2026-08-21 20:45:49`, `Modify: 2026-08-21 20:45:54.056940200 +0900`, `sha256 df1e596c…` (first 32 hex — same snapshot as D101-8 Step 7/8 and D101-9 Step 1/2)
> Scope: **CODE CHANGE PROHIBITED** in this audit document — read-only scan. Finding is whether Phase I-T1 (telemetry-only `WorldRetirementReservation`) can be safely added as a minimal change.
> Verification: all required tool families exercised (WSL `rg/ast-grep ag/fdfind/fd/ag/fzf/sed/awk` + serena + cocoindex `ccc` + graphify + semble + AiDex + headroom/context-mode/RTK-WSL — manifest §6; internet literature 9/9 + Vyukov fallback, supplement only)

## 0. D101-9 Step 2 status

Before this audit ran, the project carried:

```
D101-9 Step 1  CONDITIONAL PASS   — A_max / K_terminal / G_contract NOT PROVEN independently
D101-9 Step 2  A_max reservation lifecycle — PASS (design-structure PASS / implementation-conditional)
```

Step 2 proved the `A_max` lifecycle (`acquire point / irreversible swap / 9 terminal release paths / exactly-once / identity binding`) is **counterexample-free as a design placement**, with the only remaining obstruction being the **stateful deleter/context-binding implementation gate** plus the missing counting telemetry. D69 already separated `T1` (counting + telemetry + sampler + export, **no** held-set / free-stack / token / R gate) from `T2` (R gate + bounded admission authority + K_terminal bounded container + G_contract). This audit checks whether that T1 can actually land safely on current `src/`.

## 1. `RuntimeWorldAuthority::publish()` — insertion order is structurally ready

### Actual current order (verified)

`src/audioengine/RuntimeWorldAuthority.h` (and its single implementation-bearing header — no separate `.cpp`) lays out the sole physical publish gateway under `INV-X4-2/3` authority singularization (`RuntimeWorldAuthority::publish` is the `sole physical publish gateway`, `RuntimeStore::publishAndSwap()` is `RuntimeWorldAuthority-owned WriteAccess only`). Tracing via `rg/ast-grep ag/serena` over all production call sites (`PublicationExecutor::executePublish`, `AudioEngine::commitRuntimePublication`, `RuntimePublicationOrchestrator`, `Coordinator/Recovery/Timer` publishers) shows a uniform shape:

```
validate(newWorld)
  ↓
admission gate (PublicationAdmission) — reversible
  ↓
owner transfer (OwnerChannel + PendingPublishRegistry — single-ownership move, D42)
  ↓
acquire(prevWorld) insertion point  ←  (future T1 observation = successful retirement-producing publish)
  ↓
publishAndSwap(next) → returns oldWorld(=prevWorld)    ← irreversible, exchangeAtomic never fails (I4 D49 D49.2)
  ↓
ISRRetireRouter::enqueueWithRetry(oldWorld, epoch)      ← exactly one oldWorld per successful publish
```

* `prevWorld` (the old current publication) is obtained immediately before the swap — `RuntimeWorldAuthority::publish()` documents `prevWorld` retrieval and `writeAccess_.publishAndSwap(next)` as the atomic store-swap.
* All **reversible failure checks** (validation, admission `PublicationAdmission → Deferred → Reject`, commit checks) sit **strictly before** `publishAndSwap` — no `acquire success → publish failure` path exists (D49 INV-R0 by construction, re-verified in §4).
* `publishAndSwap` is the unique `exchangeAtomic(prev,next)` with no failure code — once it runs, the published world identity rotates and `oldWorld` is the retired world.

### T1 implication

`D49` `acquire(prevWorld) → publishAndSwap → retire` lifecycle is **structurally placable** here. For T1 (telemetry-only), the required event is not an admission decision but an observation:

```
successful retirement-producing publish  →  onAcquire()  (outstanding +1, acquired +1, max = max(max, outstanding))
```

No `if (!tryAcquire()) return ReservationExhausted` is added — counting ≠ admission (D69/D70).

**Finding 1: READY.** The acquisition point is unambiguously before the irreversible swap and after all reversible checks; inserting a telemetry `onAcquire()` there does not invert liveness or reorder the authority.

---

## 2. `AudioEngine` lifetime / destructor — telemetry outlives every `drain`/`deleter`

### Current wiring

`src/audioengine/AudioEngine.h` + `src/audioengine/AudioEngine.CtorDtor.cpp` show:

```cpp
// AudioEngine owns by value (construction order) and destroys in reverse order:
RuntimeWorldAuthority  worldAuthority_;                 // owns RuntimeStore + WriteAccess
ISRRetireRouter        m_retireRouter;                  // owns D+Q+Emergency+Terminal stores
EpochDomain            m_epochDomain;                   // owns reader slots / global epoch
WorldRetirementReferenceObserver  worldRetirementReference_;  // non-owning observer
ISRWorldRetirementTelemetry       worldRetirementTelemetry_; // T1 counting device (acquired/released/…)
RuntimeHealthMonitor   m_healthMonitor;                 // holds ISRWorldRetirementTelemetry* refs
```

Constructor wires `worldRetirementReference_.setTelemetry(&worldRetirementTelemetry_)` and `m_retireRouter(..., &worldRetirementReference_)`, plus `m_healthMonitor` refs to the telemetry. Destruction order is the reverse C++ member order, so **telemetry's lifetime covers every `drain`/`drainAll`/`deleter`**:

```
~AudioEngine order (list.md 12.1.2 — verified in CtorDtor.cpp):
  1) stop callbacks/workers (StopAcceptingWork)
  2) detach published runtime pointers
  3) retire captured runtimes (ISRRetireRouter::enqueueWithRetry)
  4) force epoch advance
  5) deterministic drain/reclaim: m_retireRouter->drainAllQuarantineStore(),
     TerminalReclaimAuthority::drainAll(), DeferredDeletionQueue::drainAllUnsafe()
  6) member destruction — worldRetirementTelemetry_ dies AFTER all of the above
```

So a `worldRetirementTelemetry_.onReleaseObserved()` called from any `drain`/`drainAll`/`synchronous Terminal destruction` is lifetime-safe: the telemetry object is still alive when the very last `World` is physically destroyed.

**Finding 2: READY.** Telemetry member outlives every drain/destruction path; no use-after-free risk from the observer chain.

---

## 3. Retire `World` deleter — do NOT touch `void(*)(void*)`, observe at the actual destruction LP

### Current deleter type

Every `DeletionEntry` / `QuarantinedEntry` / `TerminalReclaimAuthority::Entry` carries:

```cpp
void (*deleter)(void*);   // stateless function pointer — no capture
```

This is the type exposed by `enqueueDeferredDeleteNonRt` / `retireRuntimePublishWorldNonRt` style APIs. T1 must **not** widen it to `std::function` or an engine-captured deleter — D69/D70 scope the T1 to minimum change (count + telemetry + sampler + export), and deleter widening would violate that scope.

### Where T1's `onRelease()` must actually be observed

Step 2 §5 correctly required: **`onRelease` is bound to physical `World` destruction, not to `reclaim()` entry**. The current code confirms why:

```
D (DeferredDeletionQueue)                          — reclaim() scans head, invokes deleter when epoch-safe
Q (RetireQuarantineStore m_retireQuarantine)        — drain(minReaderEpoch) invokes deleter for epoch-safe subset
E (Emergency m_emergencyQuarantine)                 — same
Terminal (TerminalReclaimAuthority m_terminalReclaim)
  ├── store(ptr,deleter,epoch,type,reason) → push_back, always true (ISRRetireRouter.h:60-140 / .cpp:27-160)
  ├── drain(minReaderEpoch)   — epoch-gated: isOlder(entry.epoch, minReaderEpoch) true → deleter
  │                              epoch-unsafe → retained, not deleted
  └── drainAll()              — shutdown-only unconditional drainAll (AudioEngine must be stopped)
  epoch-safe && !isAudioThread → synchronous destruction path
                                (terminalReclaim() immediate deleter when epoch-safe && Non-RT,
                                 ISRRetireRouter.cpp:27-160, verified by rg/serena)
```

There are **9 distinct physical `World` destruction terminals** (§8 of Step 2 evidence), of which **two are shutdown-only** (`drainAllUnsafe` / `drainAll`). Placing `onRelease()` only at `DeferredDeletionQueue::reclaim()` would miss Q/E/Terminal and the `epoch-safe && !isAudioThread` synchronous path.

The production code also shows the **safe observation point today** for T1 without widening the deleter:

```cpp
// Already present — reference observer chain (D100.4, not a new deleter):
ISRWorldRetirementReferenceObserver (src/audioengine/ISRWorldRetirementReference.h)
  onAcquire()  → worldRetirementTelemetry_.onAcquireObserved()
  onRelease()  → worldRetirementTelemetry_.onReleaseObserved()
DeferredDeletionQueue/RetireQuarantineStore/TerminalReclaimAuthority
  each calls   referenceObserver_->onRelease()  when type==World deleter runs
```

That is: every `World` destruction already fans through `referenceObserver_->onRelease()` (a **non-owning observer**, not an ownership-changing reservation authority). So T1's `WorldRetirementReservation` counting can be fed by **observing the existing `World`-type destruction**, not by changing the deleter signature.

### Implementation blocker — explicitly called out

Step 2 evidence already flagged the **stateful deleter/context-binding** blocker: the current `deleter` is stateless (`void(*)(void*)` via `enqueueDeferredDeleteNonRt`), so a future `release deleter` that needs to capture `reservation authority` or `token` cannot be expressed there without becoming `std::function` / engine-ref-capturing. **T1 must not solve this by widening the deleter** — that would be `T2` territory (held-set / free-stack / token / R gate). For T1, the correct action is:

```
existing deleter stays stateless
World destruction (type==World) already notifies referenceObserver_->onRelease()
T1 observes that notification → onRelease() telemetry (outstanding −1, released +1)
no deleter-structure change to pass T1
```

If a future audit shows `World destruction` cannot be safely joined to `telemetry owner` (e.g. observer not wired in some drain path, or `type==World` check missing), then T1 is **blocked** and must return

```
T1 implementation blocker:
  stateful deleter/context binding unresolved
```

rather than silently changing the deleter. In the current `src/` the join is **safe**: every `World` drain path with `type==World` does notify the observer, and the telemetry object outlives them all (Finding 2).

**Finding 3: READY with noted constraint.** The required `onRelease` LPs are all 9 physical destruction sites; the safe T1 insertion is via the existing `World`-type destruction observer, not via a new deleter signature. The blocker is understood and not triggered — no `std::function` change is needed for T1.

---

## 4. `ISRRetireRouter::terminalReclaim()` — sync + stored paths both covered

The Terminal stage is the liveness total absorber (`Step 8 §7: retry exhausted → Terminal absorption is total`). Latest `src/audioengine/ISRRetireRouter.cpp` (§4 `terminalReclaim` + `TerminalReclaimAuthority::store/drain/drainAll`) shows:

```
ISRRetireRouter::terminalReclaim(ptr, deleter, epoch, type, reason)
  if (epochSafe && !isAudioThread()) {
      deleter(ptr);                          // synchronous, Non-RT, immediate World destruction
      terminalReclaim.recordWorldReclaim();  // counts toward world reclamation
      return true;                           // destroyed, not stored
  }
  return m_terminalReclaim.store(ptr, deleter, epoch, type, reason);
         // growable vector push_back, ALWAYS true, stored for later drain()
```

* **Synchronous path** — `epochSafe && !isAudioThread()` performs immediate `deleter(ptr)` (no `vector`). For T1, the corresponding `onRelease` must be observable here (the `recordWorldReclaim` / `onRelease` path already does it for `type==World`).
* **Stored path** — `store()` → `vector` + `residentAtomic_`, `ALWAYS true` (P-4 invariant: `enqueueWithRetry` never returns with pointer unowned). Later `m_terminalReclaim.drain(minReaderEpoch)` reclaims when `isOlder(entry.epoch, minReaderEpoch)` (same EBR safety as Q/retire drains), and `drainAll()` unconditionally at shutdown.

Both paths are **total absorbers** — no branch drops, leaks, or silently abandons a `World`. Step 8 already closed the liveness consequence (`retry exhausted → Terminal`), and this audit confirms the code still has both Terminal paths total:

```
D full → retry(kMaxRetry=2) → Q full → E full → Terminal (sync or stored) — never abandoned
```

**Finding 4: READY.** Both Terminal destruction LPs (synchronous immediate and stored deferred) are total and observable for T1's `onRelease` without touching the deleter signature. Shutdown `drainAll` remains separate and lifetime-safe per Finding 2.

---

## 5. T1 scope lock — what is implemented vs what is explicitly forbidden

T1 implements **only** the measurement device:

```
World retirement lifecycle observation
  onAcquire()  ≡ successful retirement-producing publish  →  outstanding +1, acquired +1, max = max(max, outstanding)
  onRelease()  ≡ physical World destruction (any of the 9 terminals)  →  outstanding −1, released +1

Telemetry:
  acquired / released / outstanding (= acquired−released) / maxOutstanding / exhaustion / B_max sampler
  observation-window tag / export

Invariant (minimum testable):
  acquired ≥ released,  outstanding = acquired−released ≥ 0,  maxOutstanding ≥ outstanding
  shutdown-complete ⇒ outstanding == 0
  (does NOT imply A_max < ∞ — T1 yields evidence candidate = observed max, not contract)
```

Forbidden in T1 (per D69/D70, Step 2 §1-10, Step 1 Task 2/3/7 prohibitions):

```
held-set / free-stack / reservation token / ABA protection
R / R_cap / ReservationExhausted admission / actual admission rejection
bounded lifetime budget enforcement
K_terminal bounded container
G_contract (throughput) admission↔discharge coupling
λ < μ assumption
queue-capacity-sum as A_max
silent promotion of D101-8 CONDITIONAL PASS → PASS
```

In particular `ReservationExhausted` may have its enum value reserved in T1, but no existing publish failure may be reclassified to it (D70 boundary). And:

```
T1 counting ≠ admission — the count result never gates publish; it only measures.
```

---

## 6. Verdict — T1 implementation readiness

| Gate | Required | Finding |
|---|---|---|
| `RuntimeWorldAuthority::publish()` order verified | snapshot-safe | **PASS** — prevWorld → checks → `publishAndSwap` (irreversible) → retire is the actual order; `acquire(prevWorld)` point is before swap |
| AudioEngine lifetime vs all `drainAll`/`deleter` | telemetry outlives last destruction | **PASS** — reverse-destroy order guarantees it |
| World deleter `void(*)(void*)` | T1 must **not** embed reservation authority | **PASS** — keep deleter stateless; observe existing `World` destruction via `referenceObserver_->onRelease()` / `type==World` path |
| Release LP safety | joinable to telemetry owner | **PASS** — all 9 `World` destruction terminals join to observer/telemetry; no `std::function` change needed for T1 |
| `terminalReclaim()` both paths | sync immediate + stored deferred are total | **PASS** — both `store`/`drainAll`/`synchronous` paths absorb and are observable |
| T1 non-interference | RT = atomic count update only; sampler/export = Non-RT | **PASS** — scope matches D69 T1 (count+telemetry+sampler+export) |
| Blocker `stateful deleter/context binding` | must not be worked around | **ACKNOWLEDGED, NOT TRIGGERED** — T1 avoids it by observing destruction, not by changing deleter |

**Overall: READY for Phase I-T1 implementation (telemetry-only).**

D101-9 Step 2 is therefore:

```
D101-9 Step 2 = PASS (design-structure PASS / implementation-conditional) → Phase I-T1 Implementation Readiness Audit = READY
```

with the only future gate beyond T1 being the stateful `release` mechanism inside the deleter for T2's `R` gate (D50 honest future constraint) plus D101-9 Step 3 `K_terminal` bounded design — both correctly **not** entered in T1, per the specified order `A_max → K_terminal → G_contract`.

## 7. Next ordered verification (D70.3 as adopted)

```
D101-9 Step 2 PASS  →  T1 implementation  →  Build  →  Unit/CTest
→  RT safety audit  →  telemetry LP verification  →  B_max sampler verification
→  missing-sample / window-boundary verification  →  measurement (A_max evidence)
→  R_required / R / R_cap  →  T2 (R gate)  →  D101-9 Step 3 (K_terminal bounded design)
```

Until T1 measurement is complete, `K_terminal` bounded implementation (Step 3) and `R` gate (T2) must not be entered in parallel — the order `A_max → K_terminal → G_contract` is preserved. This audit enforces that order as a safety invariant.
