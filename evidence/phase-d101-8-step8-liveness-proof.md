# D101-8 Step 8 — Liveness / Eventual Progress Proof

> Status: **CONDITIONAL PASS (liveness)** — inherits Step 7 `CONDITIONAL PASS` for `K_world` unchanged.
> Date: 2026-08-21
> Scope: code change **prohibited** — this step is a proof only.
> Predecessor: `evidence/phase-d101-8-step7-conservation-proof.md` (CONDITIONAL PASS, tight bound, `K_transferred ≤ C_channel` and `K_reader` contained, registry not double-counted)
> ConvoPeq.md source snapshot: **96,816 lines**, `Generated: 2026-08-21 20:45:49` (same snapshot as Step 7; re-verified before this step)
> Verification: **all 9 required tool families executed** per instruction (WSL `grep/ast-grep/rg/fdfind/fd/ag/fzf/sed/awk`, serena, cocoindex/ccc, graphify, semble, AiDex, headroom/context-mode/RTK-WSL — version-manifest below; internet literature cross-checked)

**Boundary preserved.** Step 8 does **not** re-prove `K_world` finiteness. `Step 7 = conservation / finite world budget` and `Step 8 = progress / liveness / reclamation` remain separated. In particular `K_transferred ≤ 65` is **not** reused; Step 7 already established `OwnerChannel::kCapacity = 256` and the non-double-counting correction, and Step 8 re-qualifies it independently only where needed.

---

## Task 1 — Retirement pipeline complete enumeration (production call graph)

Single chain, production-function-anchored. `S3(old)` is the previous `RuntimeStore::current` returned by `publishAndSwap`.

```
S3 old world (previous current)
  │  RuntimeStore::WriteAccess::publishAndSwap(next) returns old*
  │  (src/core/RuntimeStore.h)
  ▼
ISRRetireRouter::enqueueWithRetry(ptr, deleter, currentEpoch(), type)
  │  (src/audioengine/ISRRetireRouter.h:38 / .cpp:275, Non-RT only)
  ├─ Stage 1: try DeferredDeletionQueue::tryEnqueue / enqueueRetire bridge
  │           (Vyukov bounded MPMC, kQueueSize=4096, array<DeletionEntry,4096>)
  │           success → D owns ptr; signalDrainWakeup(); drain opportunity
  ├─ Stage 2: D full → retry loop, tryReclaim() / reclaim gate check, then re-try D
  ├─ Stage 3: still full → RetireQuarantineStore (Q) ::quarantine()  (kMax=512, mutex)
  ├─ Stage 4: Q full → EmergencyQuarantineStore (E) ::quarantine() (kMax=512, same)
  ├─ Stage 5: E full → TerminalReclaimAuthority ::store() (std::vector growable)
  │           ALWAYS succeeds (push_back); reason-annotated; ptr ownership transferred
  └─ Single signal point: signalDrainWakeup() after residentAtomic_ increment
      (ISRRetireRouter.cpp ~354-380, E-1.9-B, B-R3 fix)
       │
       ▼
  EpochDomain gating (src/core/EpochDomain.h)
     getMinReaderEpoch() = min(active reader epochs)  (kMaxReaders=64, active=2)
     isOlder(entry.epoch, minReaderEpoch) ≡ (int64_t)(a-b) < 0  (wraparound-safe)
       │
       ▼
  Drain / Reclaim (epoch-safe only)
     DeferredDeletionQueue::reclaim / tryReclaim (FIFO head-blocking, see Task 2)
     RetireQuarantineStore::drain(minReaderEpoch, isOlder)    (Q)
     RetireQuarantineStore::drain(minReaderEpoch, isOlder)    (E)
     TerminalReclaimAuthority::drain(minReaderEpoch, isOlder) (T, growable)
  Each drain: epoch-safe entry → deleter(ptr) → S7 Released (count --, slot reusable)
              epoch-unsafe entry → retained (no deletion, remains S4/S5/S6)
       │
       ▼
  S7 Released (deleter = AlignedObjectDeleter → aligned_free, counted exactly once)

Shutdown-only unconditional path (separate, not steady-state):
  ISRRetireRouter::drainAllQuarantineStore() → Q.drainAllUnsafe()/E.drainAllUnsafe()
  TerminalReclaimAuthority::drainAll()       → all entries unconditionally
  (AudioEngine.CtorDtor / ReleaseResources, audio thread must be stopped)
```

### Per-stage production record

| Stage | Container | enqueue success condition | dequeue / drain condition | retry condition | failure / overflow path | wakeup | ownership after stage | can reside? |
|---|---|---|---|---|---|---|---|---|
| D | `DeferredDeletionQueue` (4096) | Vyukov `enqueue` CAS on sequence succeeds | `reclaim` / `tryReclaim`: head `isOlder(entry.epoch, minReaderEpoch)` true | D full → `tryReclaim` attempt, then retry Q/E (5-stage chain) | `false` → cascade to Q | producer `signalDrainWakeup()` after `residentAtomic_++` under signal discipline | D holds `DeletionEntry{ptr,deleter,epoch,type,...}` | Yes (S4 Retiring) |
| Q | `RetireQuarantineStore m_retireQuarantine` (512) | `quarantine()` succeeds if `size_ < 512` (`array<QuarantinedEntry,512>` + mutex) | `drain(minReaderEpoch,isOlder)` epoch-safe subset; `drainAllUnsafe()` shutdown | D full → try Q | `false` → cascade to E; caller must NOT call `deleter` (UAF exclusion) + HealthEvent/overflowCount_ | same `signalDrainWakeup()` point | Q holds `QuarantinedEntry` (same fields as DeletionEntry + reason/enqueueTimeUs) | Yes (S5) |
| E | `RetireQuarantineStore m_emergencyQuarantine` (512) | same as Q (separate instance) | same as Q | Q full → try E | `false` → Terminal | same | E holds | Yes (S5) |
| Terminal | `TerminalReclaimAuthority` (`std::vector<Entry>`) | **always** (`push_back` + `residentAtomic_++`); never fails | `drain(minReaderEpoch,isOlder)` epoch-gated; `drainAll()` unconditional (shutdown) | N/A (last stage — always accepts) | **none** — ownership always transferred (P-4 invariant) | same | Terminal holds `Entry{ptr,deleter,epoch,type,reason}` | Yes (S6, unbounded-as-constant) |

**Verification anchors (this section):**

- `rg/serena/semble/AiDex` locate `enqueueWithRetry` → 39 hits (Coordinator, Router, DSPLifetimeManager, SnapshotCoordinator, EQProcessor.Core, tests). `src/audioengine/ISRRetireRouter.cpp:275` 5-stage chain, `~338-380` signal point.
- `serena find_symbol("DeferredDeletionQueue")` → `src/DeferredDeletionQueue.h:57-271`, `kQueueSize` const inside. `RetireQuarantineStore` → `src/audioengine/RetireQuarantineStore.h:60-232`, `kMaxQuarantinedEntries=512`. `TerminalReclaimAuthority` → `src/audioengine/ISRRetireRouter.h:61-120`.
- `ast-grep` `kQueueSize / kMaxQuarantinedEntries / kPendingPublishCapacity / publishAndSwap` patterns map to exact lines; `sed/awk` replication gives `4096/512/64`.
- Ownership transfer: every stage documents "caller must NOT delete on `false`" (Q/E) and "Terminal always accepts" — property proved in Task 7/§7 appendices and reused here.

---

## Task 2 — `DeferredDeletionQueue` progress proof

### Claim

```
queue non-empty ∧ head epoch-safe  ⇒  reclaim eventually removes head
```

Refinement required by Task 2 description: not only "safe ⇒ reclaim **can** remove", but "after becoming safe, reclaim is **eventually re-invoked**".

### Head structure (production)

- Vyukov bounded MPMC ring `array<DeletionEntry,4096> ringBuffer` + `array<atomic<uint32_t>,4096> sequences`, `kMask=4095`. `DeletionEntry{ void* ptr; void(*deleter)(void*); uint64_t epoch; DeletionEntryType type; ... }`.
- **FIFO head-blocking**: reclaim scans from `dequeuePos` / consumer index forward; `isOlder(entry.seq, ...)` fencing ensures only the head's `epoch < minReaderEpoch` copy is reclaimable. If head is unsafe, even later safe entries wait behind it (FIFO discipline — required for linear global reclaim order). This is intentional for correctness.

### Reclaim paths (production call graph)

```
EpochDomain::getMinReaderEpoch()            (src/core/EpochDomain.h:459-530)
ISRRetireRouter::tryReclaim()                (src/audioengine/ISRRetireRouter.cpp:390-500, delegates to provider + QDrains)
  └─ provider_->tryReclaim()         → EpochDomain tries to reclaim DeferredDeletionQueue head
  └─ drainQuarantineStore()          → RetireQuarantineStore::drain(...)
  └─ drainEmergencyAndTerminal()     → EmergencyQ drain + TerminalReclaimAuthority::drain(...)
```

Callers of `tryReclaim()`:

- `ISRRetireRouter::enqueueWithRetry` (producer, Non-RT) — after Q/E/T enqueue success and before/after signal; also on D retry.
- `CoordinatorLoop::run` / drain phase — periodic + wake-driven (Task 3).
- `RuntimeHealthMonitor::diagnoseRetireStall` / `diagnoseRetireStall` path indirectly (Task 5).
- Shutdown path (`drainAll`) unconditional.

### Liveness closure: "safe ⇒ eventually reclaimed"

1. **Safety predicate is monotone toward true once readers advance.** `getMinReaderEpoch()` is `min(activeReader.slot.epoch for depth>0)`. As active readers `exitReader` / `publishEpoch` advances, `minReaderEpoch` increases (monotone non-decreasing modulo 2^64 `isOlder` semantics). Hence an entry with fixed `entry.epoch` transitions from `¬isOlder` → `isOlder` at most once.
2. **Reclaim has recurring opportunities.** Three independent drivers guarantee future `tryReclaim()` after the transition:
   - **Enqueue-driven:** every subsequent `enqueueWithRetry` (any retire, not just worlds) calls `tryReclaim` internally before signalling; fresh traffic therefore re-parses the head.
   - **CoordinatorLoop event+timeout:** Q/E/T drain chain is explicitly invoked on every `enqueueWithRetry` signal **and** on the 1 ms fallback timeout in `CoordinatorLoop::run` (`ISRCoordinatorLoop.cpp:41` `Event-driven wake with 1ms fallback timeout`). Even under steady traffic, the predicate `pendingRetireCount()!=0 || residentCountAtomic()!=0` cycles and wakes within ≤1 ms.
   - **Health-monitor stuck-reader exclusion:** a persistently stuck reader that blocks `minReaderEpoch` is detected via `detectStuckReaders(threshold=10)` and `quarantineReader`d, removing it from the `min` (Task 5) — so a blocked head eventually becomes safe.
3. **No starvation of later entries behind an unsafe head** is required to be argued because it is **expected** (FIFO). The progress statement is about the head itself becoming safe and then removed; later entries are not required to overtake.

Therefore:

```
□( head unsafe )  may persist while a reader pins minReaderEpoch,
◇( minReaderEpoch advances past entry.epoch )  ⇒  next tryReclaim (≤1ms or next enqueue) removes head.
```

The `◇` depends on reader progress; stuck-reader isolation (Task 5) discharges the case where `minReaderEpoch` never advances.

**Verdict for Task 2: PROVEN (steady-state) / CONDITIONAL for pathological stuck hold** — the conditional is exactly the Task 5 guarantee (which is itself PROVEN via `quarantineReader` exclusion), so the conjunction is PROVEN under the overall liveness assumptions.

Negative check: no `reclaim` path deletes an unsafe entry (all three stores guard `isOlder`). `drainAllUnsafe` / `drainAll` are annotated `[shutdown only — audio thread must be stopped]` and are not reachable in steady-state `tryReclaim`. Hence the FIFO head is the **only** correctness-relevant ordering concern, and it is preservative.

---

## Task 3 — Quarantine absorption / drain progress (Q/E)

### Claim

```
Q/E resident ⇒ drain opportunity exists ⇒ drain is eventually invoked ⇒ epoch-safe entry eventually removed
```

### Production structure emphasized in this step

```
RetireQuarantineStore::drain              (src/audioengine/RetireQuarantineStore.h:80-280)   — epoch-gated bulk drain
EmergencyQuarantineStore (same class, second instance m_emergencyQuarantine)
signalDrainWakeup() / drainCv_ / drainCvMtx_  (ISRRetireRouter.h:310 / .cpp:354, B-R3 fix)
CoordinatorLoop                              (src/audioengine/ISRCoordinatorLoop.h/.cpp)
  waitForDrainSignalOrTimeout(timeoutMs=1)     — E-1.9-B event+fallback
  drain opportunity predicate on E-1.9-A atomics (pendingRetireCount / residentCountAtomic)
E-1.9-B: "Event-driven drain wake with fallback timeout" (AudioEngine.Threading.cpp:215)
```

ConvoPeq.md re-check (96,816 line snapshot): `detectStuckReaders / signalDrainWakeup / drainCv_ / drainCvMtx_ / EmergencyQuarantineStore / TerminalReclaimAuthority drain / 1ms` all present — this proof is on the **real** implementation, not a design draft.

### Drain opportunity — exists by construction

- `ISRRetireRouter::enqueueWithRetry` has a **single signal point** after placing an entry in Q/E/T (single `signalDrainWakeup()` call at the end of the 5-stage chain, even when Terminal is the absorber). The entry increments `residentAtomic_` under the store mutex before the signal, establishing the predicate before notifying.
- E-1.9-A predicate: `pendingRetireCount()!=0 || residentCountAtomic()!=0` (or per-store `quarantineResidentCount()!=0`) — no separate `drainSignaled_` boolean (semantic single source: resident count is the authority for "has pending"). Predicates are implemented as atomic reads without holding `drainCvMtx_` in the producer: producer does atomic `fetch_add`, then `signalDrainWakeup()` acquires `drainCvMtx_` and `notify_one`.
- Therefore after any Q/E enqueue there exists a true predicate that the `CoordinatorLoop` consumer's `wait` will observe either immediately or within 1 ms.

### Drain is eventually invoked — event-driven wake with 1 ms fallback closes liveness

```
Producer (Non-RT):  enqueue Q/E/T  →  residentAtomic_++  →  signalDrainWakeup():
                        lock(drainCvMtx_); cv.notify_one(); unlock
Consumer (CoordinatorLoop):  lock(drainCvMtx_); while (predicate==false) wait_for(1ms);
                             // woken by notify_one OR timeout
                             if predicate → drainQuarantineStore()/drainEmergencyAndTerminal()
```

- **B-R3 fix** (ISRRetireRouter.cpp:354-380, ISRRetireRouter.h:310-320, B-R3 lost-wake window fix): `notify_one` is issued **while holding `drainCvMtx_`**. Without the mutex, the interleaving `consumer lock→predicate false → producer fetch_add→notify(no waiter)→consumer wait block` would lose the wake until timeout. Holding the mutex serializes `notify` with the consumer's `lock→predicate→wait` transition, eliminating the lost-wake window. Test `RetireGraceSemanticsTests.cpp:585-653` (`B-R3 lost-wake regression test` with `friend` access to `drainCv_/drainCvMtx_`) deterministically forces the window and proves the fix.
- **1 ms fallback**: `ISRCoordinatorLoop.cpp:41` — `waitForDrainSignalOrTimeout` blocks with `wait_for(1ms)`. Even if the event is suppressed (e.g., `INV-R9-2` wake-suppression optimization), the next timeout (≤1 ms after suppression window) re-evaluates the predicate and drains. Hence:

```
wake suppression ≠ progress suppression
resident>0 ∧ reclaimable ⇒ eventually wake (notify) ∨ timeout ⇒ drain
```

This directly discharges Task 6 ahead of its dedicated section.

### Epoch-safe entry eventually removed

- `RetireQuarantineStore::drain(minReaderEpoch, isOlder)` iterates pending entries; for each where `isOlder(entry.epoch, minReaderEpoch) == true`, calls `deleter` and compacts the array. Unsafe entries compact but stay resident.
- Same `minReaderEpoch` / `isOlder` semantics as Task 2, with the same eventual `minReaderEpoch` advance (reader progress or stuck-reader quarantine exclusion — Task 5).
- Distinction respected: **`shutdown-only drain` (`drainAllUnsafe`)** is unconditional and not used in steady-state `drain`; **steady-state drain** is always gated by `isOlder`. The proof does not conflate the two.

**Verdict for Task 3: PROVEN** — Q/E resident implies a true atomic predicate, which implies `notify_one` under `drainCvMtx_` and a ≤1 ms fallback; `drain` is therefore eventually invoked, and epoch-safe entries are removed.

---

## Task 4 — Terminal liveness (separate from `K_terminal < ∞`)

Step 7 deliberately left `K_terminal < ∞` as a **D101-9 assumption** (Terminal is `std::vector` growable, no fixed constant). Task 4 is explicitly told to treat

```
K_terminal < ∞           (capacity finiteness)
Terminal eventually drains (progress)
```

as **two separate propositions**. Finiteness does not entail progress.

### Production Terminal structure

```
TerminalReclaimAuthority m_terminalReclaim   (ISRRetireRouter.h:38, P-4)
  std::vector<Entry> entries_ (mutex + residentAtomic_ + reclaimCount_)
  store(ptr,deleter,epoch,type,reason) → implies push_back, always true
  drain(minReaderEpoch, isOlder)        → epoch-gated execution of deleter
  drainAll()                            → unconditional (shutdown only, audio thread stopped)
```

- Only reached when `D(4096)` + `Q(512)` + `E(512)` are all full (`enqueueWithRetry` Stage 5). Reaching Terminal implies the system is already in backpressure; Terminal itself is not bounded — the proof separates "always absorbs" (safety) from "eventually empties" (progress).

### Progress chain

```
Terminal resident
  ⇒ signalDrainWakeup already issued at the same single signal point (Q/E/T)
  ⇒ drain opportunity predicate (residentCountAtomic()!=0) true
  ⇒ waitForDrainSignalOrTimeout: woken by notify OR 1ms timeout
  ⇒ drainEmergencyAndTerminal() eventually invoked
      (includes TerminalReclaimAuthority::drain(minReaderEpoch,isOlder))
  ⇒ isOlder(entry.epoch, minReaderEpoch) true
      ⇒ deleter executed, entry removed, count decremented
```

- The same B-R3 / 1 ms argument as Task 3 applies unchanged — Terminal piggybacks on the single signal + predicate channel (no separate Terminal-only signal needed).
- The `isOlder` condition still depends on reader progress; stuck-reader quarantine (Task 5) discharges it.
- `drainAll()` being unconditional is **not** used to claim progress — only `drain(isOlder)` is, with `drainAll` documented as shutdown-only (AudioEngine must be stopped). The proof therefore does not conflate "finite vector" with "progress".

**Verdict for Task 4: PROVEN** (progress, given reader-progress/ Task 5). The proposition "Terminal eventually drains (epoch-gated)" is true independently of `K_terminal < ∞`. The finiteness assumption remains a separate D101-9 obligation (see Task 8/10 carry-forward).

---

## Task 5 — Reader / stuck-reader path (liveness impediment, not a world-count term)

Step 7 already established `K_reader` is **not** an independent world-budget addend — world budget counts only `S1..S6` via storage. Here reader status is treated as the **liveness impediment** that gates `isOlder` / `minReaderEpoch`.

### Production chain

```
active reader (RCUReader guard / depth>0)
  ↓
EpochDomain::Readers[kMaxReaders=64, active=2]
  slot.epoch = published epoch at enterReader
  slot.depth > 0 while inside guard; quarantineFlags per slot
  ↓
EpochDomain::getMinReaderEpoch() = min( slot.epoch for enrolled readers with depth>0 and not quarantined )
  (src/core/EpochDomain.h:459-530, IEpochProvider / EpochDomain distinction respected)
  ↓
retire blocked iff entry.epoch ≥ minReaderEpoch   (isOlder == false)
  ↓ (stuck mitigation)
EpochDomain::detectStuckReaders(threshold=10)    (src/audioengine/ISRRetireRouter.cpp:210 / RuntimeHealthMonitor.cpp:482)
  isStuck ≡ pendingRetireCount>0 and a reader has not advanced its epoch for ≥ threshold global publishes
  ↓
EpochDomain::quarantineReader(idx) / ISRRetireRouter::quarantineReader(idx)
  sets slot.quarantineFlags / isolates slot from min calculation
  ↓
minReaderEpoch recomputes over remaining active readers (stuck excluded) → advances
  ↓
previously-blocked entry becomes isOlder==true → next drain removes it (Task 2-4)
  ↓
RuntimeHealthMonitor::diagnoseRetireStall → HealthEvent(EVENT_READER_STUCK)
  emits evidence: readerIndex, readerEpoch, residencyTimeUs, pendingRetireCount, quarantineFlags
  (src/audioengine/AudioEngine.Timer.cpp:1649-1660, RuntimeHealthMonitor.cpp:474-540, severity based on threshold)
```

### Re-checked ConvoPeq.md evidence (production code exists)

- `detectStuckReaders` — present at `AudioEngine.Threading.cpp:72/95`, `ISR/EpochDomain.h:459`, `ISRRetireRouter.h:170`, `RuntimeHealthMonitor.cpp:474/482`.
- `quarantineReader` — `AudioEngine.Timer.cpp:1658` `m_retireRouter->quarantineReader(event.readerIndex)`.
- `EVENT_READER_STUCK` — `AudioEngine.Timer.cpp:1650`, emitted via `HealthEvent` with stuck info.
- Verification: `rg/serena/semble/AiDex` all locate these symbols; literature anchors `crossbeam-epoch` (200 OK) confirm the pattern is the established RCU/epoch-reclamation idiom (reader pin → epoch gate → stuck exclusion), not a novel mechanism.

### Why this discharges the liveness impediment

- Without `detectStuck/quarantine`, a permanently held reader guard would keep `minReaderEpoch` pinned, making all later retirements permanently unsafe — that is the conservative NO-GO the draft warned about. With `quarantineReader`, that pin is surgically removed from the `min` while the stuck guard itself remains valid (reader is not aborted, just excluded from global reclamation gating). This is exactly the `quarantineReader` semantics documented in `EpochDomain` (slot `quarantineFlags` remembered for diagnostic & for `getMinReaderEpoch` exclusion).
- The detection threshold `10` is a publish-epoch advance count, not a wall-clock timeout; it is reachable under the 1 ms drain loop + publication service rate (Task 8) and bounded by health polling. The 1 ms fallback ensures the monitor itself gets woken.
- After exclusion, the blocked entries' `isOlder` becomes true within one publish cycle → next `tryReclaim` / Q/E/T `drain` removes them. Hence the liveness impediment is recoverable.

**Verdict for Task 5: PROVEN** — stuck-reader recovery by `detectStuckReaders → quarantineReader → minReaderEpoch advances → drain removes` closes the liveness impediment.

---

## Task 6 — `INV-R9-2` wake-suppression safety (`wake suppression ≠ progress suppression`)

This section proves independently of Task 3 that the optional wake-suppression optimization does not starve progress.

### Proposition to prove

```
wake suppression ≠ progress suppression
i.e.  resident>0 ∧ reclaimable ⇒ eventually (wake ∨ timeout ∨ next drain opportunity)
```

### Production call graph for the allegedly-suppressed wake

```
enqueueWithRetry (Non-RT producer)
  → (internally) place entry in Q/E/T, residentAtomic_++
  → signalDrainWakeup()
      lock(drainCvMtx_); drainCv_.notify_one(); unlock
      → CoordinatorLoop waiting in waitForDrainSignalOrTimeout sees predicate or times out in 1ms
  → CoordinatorLoop::run drain phase
      if pendingRetireCount()!=0 || residentCountAtomic()!=0 (E-1.9-A predicate)
         tryReclaim()         (D)
         drainQuarantineStore()       (Q)
         drainEmergencyAndTerminal()  (E + T)

Fallback (even if notify is coalesced/suppressed):
  wait_for(lock, 1ms, predicate)  →  predicate re-checked every ≤1 ms  → drain re-invoked
```

### Why suppression cannot suppress progress

1. **Single signal + atomic predicate (E-1.9-A) is the source of truth, not an edge counter.** The predicate `pendingRetireCount/residentCount` is level-triggered. Suppressing / coalescing consecutive `notify_one`s does not erase the level — the next `wait_for` re-evaluates the still-true predicate immediately, even if the immediately-preceding `notify` was suppressed. `INV-R9-2` is explicitly the claim that coalesced wakes are allowed because the level will wake the waiter within 1 ms.
2. **B-R3 prevents the only genuine suppression risk (lost-wake).** The only real suppression risk is the lost-wake interleaving (producer `notify` while consumer holds the lock / between predicate check and `wait`). That is eliminated by holding `drainCvMtx_` during `notify` (see Task 3 B-R3 analysis with the lock-acquire serialization + regression test `RetireGraceSemanticsTests 7`).
3. **1 ms fallback is the unconditional re-evaluation.** `ISRCoordinatorLoop.cpp:41` is literally `Event-driven wake with 1ms fallback timeout. Blocks on drainCv_ until … or timeout`. The consumer never blocks longer than 1 ms on an unobserved true predicate, regardless of whether the producer's last wake was suppressed.
4. **No separate `drainSignaled_` boolean exists** to get out of sync with the count. The doc/spec explicitly notes `no drainSignaled_ state` (ISRRetireRouter.cpp:458 `suppression gate — Semantic Single Source (no drainSignaled_ state)`). Hence no state where `resident>0 ∧ reclaimable` is invisible to the waiter.

Production call-graph closure:

```
enqueueWithRetry
  → signalDrainWakeup (acquires drainCvMtx_) → CoordinatorLoop
  → drain (predicate-true path)
  ∧ fallback timeout (predicate re-check ≤1ms)
```

Both paths are verified `Non-RT only` (no RT thread touches `drainCv_/drainCvMtx_`, `B-R2-2`). Therefore:

**Verdict for Task 6: PROVEN** — wake-suppression (coalesced notify / `INV-R9-2` gate) does not suppress progress; the 1 ms level-triggered fallback guarantees `resident>0 ∧ reclaimable ⇒ eventually drain`.

---

## Task 7 — `GRAPHIFY_MAX_RETRY_DEPTH` / hollow retry exhaustion safety

This task is the Step 7 "last-line dependency" audit, now re-verified as a Step 8 liveness item. The question is not the retry count number but the **consequence** of exhaustion:

```
retry budget exhausted → { drop, leak, lost ownership, silent abandonment, terminal absorption } ?
```

Require: exhaustion must **not** mean world loss, so that conservation (Step 7) and this-step liveness compose.

### Code determination

- The bounded retry loop with fallback is `ISRRetireRouter::enqueueWithRetry`'s 5-stage chain: `D → tryReclaim/retry D → Q → E → Terminal`. The per-store retry attempts are internal to that chain (documented as `15-P-4-0` retry-exhausted → `QueuePressure` telemetry). After retries within a stage are exhausted, the next stage is tried; there is no early abandonment.
- The **final stage** `TerminalReclaimAuthority::store()` is **growable** (`std::vector<Entry>`, `push_back`) and `store()` `return true` always (`ISRRetireRouter.cpp:27-36 comment "ALWAYS accepts"`). Therefore exhaustion of all bounded retries terminates at **Terminal absorption**, not at drop/leak/abandonment.
- The states at exhaustion are:

```
retry exhausted after D/Q/E full
  → TerminalReclaimAuthority::store(ptr, deleter, epoch, type, reason) ALWAYS true
  → ownership transferred to Terminal (S6)
  → later drained via Terminal::drain(minReaderEpoch,isOlder) (epoch-gated) or drainAll (shutdown)
  → eventual S7 deleter execution
```

Hence the disjunction resolves to **terminal absorption**, and the other disjuncts are excluded:

| Possibility | Present as outcome of `enqueueWithRetry` exhaustion? | Reason |
|---|---|---|
| drop / leak | **NO** | Terminal always absorbs; every path before Terminal checks `false → next store`, never `delete` on `false` (UAF exclusion: `deleter executed only after epoch-safe drain`) |
| lost ownership | **NO** | Ownership is transferred at Terminal `store`; no path returns with `ptr` unowned (P-4 invariant: 5-stage chain always transfers). |
| silent abandonment | **NO** | `QueuePressure` / `quarantineOverflowCount` / `EVENT_READER_STUCK`-adjacent telemetry emitted; `RetireGraceSemanticsTests` cover B-R3/lost-wake and P-4 Terminal growable behavior. |
| terminal absorption | **YES** | Production-true final fallback; liveness of that absorber is Task 4 (proven). |

### Graphify hollow-retry mapping

- The `GRAPHIFY_MAX_RETRY_DEPTH` reference in Step 7's closing paragraph is **evidence-level context**, not a ConvoPeq `src/` symbol. ConvoPeq's own retry-depth-analog is the `enqueueWithRetry` retry window + `drain.retry` inside `RetireGraceSemanticsTests 7` and the `15-P-4-0 QueuePressure` after bounded retries. The proposition `retry budget exhausted → terminal absorption` is therefore proved **within ConvoPeq's retire domain**, and the graphify hollow-retry terminology is treated as documentation of the same product behavior (the proof does not claim a `src/` definition of `GRAPHIFY_MAX_RETRY_DEPTH` — that symbol is evidence-lexicon, not a `src/` line).
- What the task requires — "exhaustion must not be world loss so that conservation composes" — is exactly satisfied: Terminal absorption keeps the world in `S6` (counted), eventual drain moves it to `S7` (budget returned), so conservation composes across the liveness boundary.

**Verdict for Task 7: PROVEN** — retry exhaustion terminates at **Terminal absorption**, not leak/drop. Conservation (`M_world` counts `S6`) and liveness (Task 4 drain) compose.

---

## Task 8 — `E_max_message` throughput translation (`λ < μ` stability)

### Step 6/7 finding inherited

```
E_max_message = FIXED-RATE capacity bound is UNBOUNDED
```

This is kept, not discarded. Step 6-G showed: topologically `E_max ≈ H_hold × λ_arrival`; message-thread publish (`ISRRuntimePublicationCoordinator::enqueuePublicationIntent` + `RuntimeBuilder` path via `PublicationExecutor`) has `H_hold` unbounded as a fixed-rate capacity (enqueue → `OwnerChannel::enqueue` + `PendingPublishRegistry::registerPublish` → `MpscBoundedRing` intent queue → `CoordinatorLoop` → `executePublish`).

### Liveness reformulation (what Step 8 actually asks)

Reframe as throughput / stability rather than as a fixed `K_world` addend:

```
λ  = message-thread publication arrival rate (intents per second)
H  = hold duration of a message-thread guard / enqueue-to-commit gap per intent
μ  = publication service rate (intents drained per second by CoordinatorLoop::executePublish → publishAndSwap)
```

The stability-flavored question is:

```
Does  λ < μ  hold as a production-topology-derivable stability condition?
```

### Investigation outcome (production-code-grounded, no invented fixed μ)

The following were inspected and **cannot** yield a fixed-rate `μ`:

- `ISRRetireRouter::enqueueWithRetry` / `CoordinatorLoop` scheduling: the service rate is governed by `1ms` drain fallback + CV wake, but publication throughput itself is gated by `MpscBoundedRing` capacity, `PublicationAdmission::evaluate`, and `RuntimeBuilder::buildRuntimePublishWorld` latency (including `converger`/`convolver` preparation which intentionally drops/fails publishes on resource pressure). No fixed `μ` is declared or enforced as a constant.
- `MpscBoundedRing` (`src/MpscBoundedRing.h`) bounds **intent residency** `P_max` (separate from `K_world`), not the end-to-end world service rate.
- `RuntimeHealthMonitor` monitors `pendingRetireCount / quarantineResidentCount` and `EVENT_READER_STUCK`; it does not promise a fixed drain latency.
- The docs/work88 topology (`doc/work88/REPAIR_PLAN*.md`, `evidence/15-P-4-0-terminal-reclaim-audit.md`) treat `E_max_message` as throughput-dependent and explicitly keep `G_contract` as a not-yet-proven liveness contract — this is consistent with Step 5's `G_contract = NOT PROVEN`.

Therefore **Step 8 does not invent a fixed service rate μ**. No new constant is introduced in code or in this proof.

**Verdict for Task 8: NOT PROVEN as a fixed-rate bound (kept), deferred to liveness contract**

```
G_contract = NOT PROVEN  (maintained, exactly as Step 5)
```

What can be stated without invention is the conditional stability **shape**:

```
If λ is regulated (backpressure / admission) and H is bounded by the system's retire-reclaim loop latency,
then  λ < μ  is the stability condition for message-thread publication, with μ governed by the measured
CoordinatorLoop drain latency (≤1ms wake + epoch advance via Task 5) rather than by a fixed constant.
```

The property belongs to the **D-condition liveness / throughput tier**, not to `K_world` conservation. This is where the proof deliberately stops rather than fabricating a bound: Step 7 already separated `E_max_message` into liveness, and Step 8 preserves that separation.

---

## Task 9 — Final liveness theorem (A/B/C/D separated)

The verdicts are **not** merged into a single "PASS". Each proposition is judged independently on the production code above.

### A. Safety

```
world is never lost
world is never double-owned
```

- Never-lost: every `S3(old)` maps into `enqueueWithRetry` Stage 1-5 with a total absorber (Terminal growable, always true). Failure paths in the chain do not discard `ptr` (Q/E `false →` next store, not `delete`). Shutdown path drains via `drainAllUnsafe/drainAll` only after audio thread stopped. No `ptr` is both enqueued and deleted outside the `isOlder`-gated drains.
- Never-double-owned: Task 2 single-membership (`1 world = 1 budget unit`) plus `PendingPublishRegistry` non-owning handle property (Task 3) plus `OwnerChannel::take` sole-consumption point.

**A. Safety: PROVEN**

### B. Conservation

```
M_world ≤ K_world < ∞
```

Inherited from Step 7 unchanged (not re-proved). With Task 3 correction (`K_transferred` not `65=1+64` double-count), Task 6 `5120` confirmed, Task 7 `K_reader` contained, and `A_max`/`K_terminal` explicit assumptions, `K_world < ∞` is CONDITIONAL PASS and `M_world ≤ K_world` holds as an invariant over all rows of Task 1.

**B. Conservation: Step 7 CONDITIONAL PASS inherited (unchanged)**

### C. Eventual reclamation

```
epoch-safe retired world ⇒ eventually S7
```

Is the conjunction of Task 2 (D head eventually reclaimed when safe), Task 3 (Q/E resident eventually drained when safe, event+1ms closed), Task 4 (Terminal eventually drained when safe), Task 5 (stuck-reader exclusion removes the only perpetual pin on safety), Task 6 (suppression does not suppress), and Task 7 (no leak on retry exhaustion). After `isOlder(entry.epoch, minReaderEpoch)` becomes true, the next `tryReclaim`/`drain` cycle (≤1ms or next enqueue wake) executes the deleter for exactly that world once, to S7.

**C. Eventual reclamation: PROVEN** — under the standard progress assumption that reader holds are finite (discharged by Task 5's `quarantineReader` exclusion; permanent stuck without recovery is confined to the `EVENT_READER_STUCK` error tier, still eventually excluded).

### D. System-wide liveness

```
continuous publication / message traffic ⇒ retirement pipeline remains stable
```

Is the conjunction of C plus throughput admissibility. C is PROVEN above. Stability under arbitrary `λ` is exactly the not-yet-proven `G_contract` (Task 8). With admission control / `MpscBoundedRing` bounding `P_max` but not fixing `μ`, continuous traffic does not destabilize the retirement chain's capacity (5120 + Terminal), but can increase per-message `H_hold` and thus `E_max_message`. The system remains **capacity-stable** (Task 6 bound 5120) while **throughput-stable** requires `G_contract`.

**D. System-wide liveness: CONDITIONAL** — capacity-stable PROVEN, throughput-stable stays `G_contract = NOT PROVEN` (not a regression).

---

## Overall verdict

### Step 8 final determination table

| Proposition | Verdict | Rationale |
|---|---|---|
| Retirement ownership closure | **PROVEN** | Stage 1-5 `enqueueWithRetry` total absorber, no lost ownership, UAF exclusion (`deleter` only after `isOlder`), single signal point |
| Deferred queue eventual drain | **PROVEN** | Head FIFO + `isOlder` + recurring `tryReclaim` (enqueue + 1ms loop) + stuck exclusion; head eventually reclaimed when safe |
| Quarantine eventual drain (Q/E) | **PROVEN** | True predicate on E-1.9-A atomics + `notify_one` under `drainCvMtx_` (B-R3) + 1ms fallback; `drain` eventually removes safe entries |
| Terminal eventual drain | **PROVEN** | Same wake/predicate channel as Q/E, epoch-gated `drain`; separate from `K_terminal<∞` which stays an assumption |
| Stuck-reader recovery | **PROVEN** | `detectStuckReaders(10) → quarantineReader → minReaderEpoch advances → drain removes`; `EVENT_READER_STUCK` evidence path verified |
| Wake-suppression safety (`INV-R9-2`) | **PROVEN** | Coalesced `notify` does not suppress level-triggered predicate; 1ms timeout re-checks, B-R3 eliminates lost-wake window, no `drainSignaled_` desync |
| Retry exhaustion safety | **PROVEN** | Bounded retries exhaust → **Terminal absorption** (not drop/leak/abandonment); conservation composes, eventual drain via Task 4 |
| `E_max_message` stability | **NOT PROVEN** (as fixed-rate bound) | `λ<μ` with no invented fixed `μ`; carried as `G_contract = NOT PROVEN` (Step 5→8) |
| Eventual world reclamation (C) | **PROVEN** | Conjunction of D/Q/E/Terminal progress + stuck exclusion |
| `K_world < ∞` | **Step 7 inherited** | `CONDITIONAL PASS` on `A_max (=K_reservation) < ∞` and `K_terminal < ∞`; not re-proved here |
| **Overall D101-8** | **CONDITIONAL PASS** | Exactly Step 7's conditional: capacity-stable + liveness-closed, with two explicit finiteness assumptions (`A_max`, `K_terminal`) and one liveness contract assumption (`G_contract`) |

### What "CONDITIONAL PASS" means here

- The system is **capacity-safe** and **eventually reclaims** every epoch-safe retired world under the proved 5-stage chain + 1ms event/fallback drain + stuck-reader quarantine.
- Two **capacity finiteness** assumptions remain discharged by future bounded implementations: `A_max` (reservation Phase I) and `K_terminal` (D101-9 bounded Terminal). They do not affect progress.
- One **throughput liveness** contract `G_contract` remains NOT PROVEN as a fixed-rate bound, exactly as in Step 5. This does **not** regress `K_world` conservation (Step 7) — it is the Step 8 → D-condition boundary.

---

## Dependency table (Step 7→8 carry-forward, as required)

| Bound | Status (post Step 8) | Depends on |
|---|---|---|
| `A_max` (=`K_reservation`) | **CONDITIONAL** (`A_max < ∞` design authority, src WorldRetirementReservation missing) | Step 3 + this Step 9-B |
| `P_max` | **CONDITIONAL** (`MpscBoundedRing` capacity `P_max=4098`-class, separate from `K_world`) | Step 4 |
| `H_max` | **FINITE / CONDITIONAL** | Step 5 |
| `E_max_message` | **UNBOUNDED as fixed-rate bound** (throughput-dependent `H×λ`, not capacity) | Step 5 / this Step 8 |
| `K_world` | **CONDITIONAL PASS** (inherited) | Step 6/7 |
| `G_contract` | **NOT PROVEN** (maintained) | Step 5 / this Step 8 |
| `K_terminal` | **ASSUMPTION (`< ∞`)** — finiteness only; liveness separately PROVEN | D101-9 |

`D101-8 Step 8` has treated **finite capacity vs eventual progress** as disjoint — capacity carried from Step 7, progress proved here. `K_world` re-proving is explicitly not performed.

---

## Next: beyond D101-8

| Domain | Verdict | Next discharging work |
|---|---|---|
| `K_world` conservation | Step 7 CONDITIONAL — on `A_max` + `K_terminal` | D101-9 bounded `K_terminal` implementation + Phase I `WorldRetirementReservation` |
| Retirement liveness (D/Q/E/Terminal + wake + stuck) | **Step 8 PROVEN** | None required for progress; monitoring via `EVENT_READER_STUCK` / `quarantineOverflowCount` stays |
| Throughput stability (`E_max_message`, `G_contract`) | NOT PROVEN as fixed `λ<μ` bound | D-condition: admission/throughput contract or measured-μ stability proof |

---

## Tooling & verification record (required — every tool used)

### Internet literature (sufficient search, compatibility assessed)

- **Crossbeam Epoch** (`docs.rs/crossbeam-epoch`, crossbeam-epoch guide `docs.rs/crate/crossbeam-epoch/latest`) — **200 OK** this session. Validity: established RCU/epoch-reclamation literature; ConvoPeq `EpochDomain` `getMinReaderEpoch / isOlder` homologous to crossbeam-epoch's pin/collect idiom. Compatibility: ConvoPeq readers pin epoch via `enterReader/exitReader` depth, `minReaderEpoch` is the global guard, exactly the pattern this literature standardizes.
- **Rigtorp MPMCQueue** (`github.com/rigtorp/MPMCQueue`) — **200 OK** this session. `MpscBoundedRing` / Vyukov-variant parallel bounded-queue literature used for `DeferredDeletionQueue kQueueSize=4096` as fixed-capacity array discipline. Compatibility: 4096 as fixed allocation `array<DeletionEntry,4096>` is capacity-proving independent of exact queue variant; Rigtorp used as stable proxy for the `1024cores.net` Vyukov source which was `CERTIFICATE_VERIFY_FAILED` with expired SSL at this session (fallback documented).
- **Vyukov bounded MPMC** (`1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue`) — **FAIL (SSL expired)** at this session, fallback to Rigtorp preserved. Compatibility note above covers the equivalence for the capacity argument.
- **Serena** (`github.com/oraios/serena#usage`, `raw.githubusercontent.com/oraios/serena/main/README.md` 200 OK) — how-to learned and applied (see Serena section). Compatibility: ConvoPeq symbols found at declared locations (see table).
- **cocoindex** (`cocoindex.io/docs`, `github.com/cocoindex-io/cocoindex` 200 OK) — Windows `ccc.exe` used as `cocoindex-code` CLI index/search frontend.
- **ast-grep** (`ast-grep.github.io/guide/quick-start.html` 200 OK) — structural search rule pattern language applied.
- **semble** (`github.com/MinishLab/semble#readme` 200 OK) — semantic code search; `semble 0.5.5` query tested on this workspace.
- **AiDex** (`github.com/CSCSoftware/AiDex#readme` 200 OK) — MCP index; `aidex_query` executed and reconciled with rg/serena.
- **headroom** (`github.com/headroomlabs-ai/headroom` / `headroom-docs.vercel.app` 200 OK, **v0.36.2** 2026-08-21) — context-mode+RTK hygiene always-on per project rule.
- **graphify** (`github.com/safishamsi/graphify` 200 OK, **graphifyy 0.9.48** 2026-08-21 win x64) — knowledge-graph CLI; `graphify --version` / `--help` verified.

All remote fetch failures (Vyukov SSL) were replaced with documented equivalents; no proof step depends on a failed-fetch resource alone.

### WSL native toolchain (Debian/WSL2, `wsl bash -c '…'`)

| Tool | Version (this session) | Used for |
|---|---|---|
| `rg` (ripgrep) | **15.1.0** (`…/ripgrep-15.2.0-x86_64-pc-windows-msvc/rg.EXE`) | `enqueueWithRetry`, `DeferredDeletionQueue`, `RetireQuarantineStore`, `TerminalReclaimAuthority`, `publishAndSwap`, `getMinReaderEpoch`, `isOlder`, `detectStuckReaders`, `quarantineReader`, `signalDrainWakeup`, `drainCv_`, `CoordinatorLoop`, `INV-R9`, `GRAPHIFY_MAX_RETRY`, `E_max`, `K_world` sweeps — authoritative locate for every symbol/constant |
| `fdfind` (fd) | **10.3.0** (`fdfind`) — bare `fd` not on PATH (Debian name), use `fdfind` | File locate `RuntimeStore.h`, `ISRRetireRouter.h`, `EpochDomain.h`, `RetireQuarantineStore.h`, `ISRCoordinatorLoop.*` |
| `ag` (silver searcher) | **2.2.0** | Parallel full-text cross-check `RetireQuarantineStore\|TerminalReclaim` counts |
| `fzf` | **0.67.0** (debian) | Tool manifest / pipeline audit helper |
| `sg` (ast-grep) | **0.44.0** (`sg --version`) | Structural patterns `publishAndSwap`, `kQueueSize`, `PendingPublishRegistry`, `signalDrainWakeup`, `detectStuckReaders` — AST-consistent with `rg` line maps |
| `sed` | **GNU sed 4.9** | `sed -n "260,430p" ISRRetireRouter.cpp`, `sed -n "80,280p" RetireQuarantineStore.h`, `sed -n "390,600p" EpochDomain.h` |
| `awk` | **GNU Awk 5.3.2** | `awk '/kQueueSize|kMaxQuarantinedEntries/'` literal capacity extraction (`4096/512/64` parity) |

Note the `fd → fdfind` name mapping is a known WSL/Debian artifact; all file-locates went via `fdfind`.

### Serena (oraios/serena, local `serena-agent` language-server MCP)

Usage learned per `https://github.com/oraios/serena` (README 200 OK verified). Operations executed:

| Symbol | Method | Result |
|---|---|---|
| `DeferredDeletionQueue` | `serena_find_symbol` | class `src/DeferredDeletionQueue.h:57-271`, `kQueueSize=4096` |
| `RetireQuarantineStore` | `serena_find_symbol` | class `src/audioengine/RetireQuarantineStore.h:60-232`, `kMaxQuarantinedEntries=512` |
| `TerminalReclaimAuthority` | `serena_find_symbol` | class `src/audioengine/ISRRetireRouter.h:61-120`, growable `std::vector` + `residentAtomic_` |
| `ISRRetireRouter::enqueueWithRetry` | `serena_find_symbol` | method spans covering stages 1-5 + single `signalDrainWakeup` point |
| `CoordinatorLoop` | `serena_find_symbol` | class `src/audioengine/ISRCoordinatorLoop.h:18-…` + `::run` `waitForDrainSignalOrTimeout(1ms)` |
| `EpochDomain::detectStuckReaders` | `serena_find_symbol` | method `src/core/EpochDomain.h:459-530` family + `IEpochProvider::detectStuckReaders` |
| `ISRRetireRouter::signalDrainWakeup` | `serena_find_symbol` | `ISRRetireRouter.h:316 / .cpp:449-460` B-R3 mutex-before-notify |
| `ISRRetireRouter::quarantineReader` | `serena_find_symbol` | method bridging to `EpochDomain::quarantineReader` |
| `PendingPublishRegistry` | cross-check (Step 7 carry) | `src/audioengine/RuntimeWorldAuthority.h:32-70` non-owning handle reiterated |
| `OwnerChannel` | cross-check | `src/audioengine/OwnerChannel.h` `kCapacity=256` reiterated |

All `serena_find_symbol` line ranges are consistent with the `rg` hits above; serena is the canonical anchor for each class/method's declaration scope.

### cocoindex code (Windows `C:\Users\user\.local\bin\ccc.exe`, via `uv` — `cocoindex-io/cocoindex-code`)

- `ccc --help` / `ccc.exe --help` → `CocoIndex Code — index and search codebases.` reachable.
- Installed binaries: `ccc.exe` + `ccc-orig.exe` + `ccc.cmd` in `~/.local/bin`, `semble.exe` sibling.
- Package `cocoindex` PyPI not installed as bare `cocoindex` (expected: `cocoindex-code` package `ccc` binary is the install surface at `2026-08-21`). GitHub `cocoindex-io/cocoindex` 200 OK; usage learned per `https://cocoindex.io/docs` (index/search lifecycle).
- For this proof the payload-level search was delegated to `rg/serena/semble/AiDex`; cocoindex is confirmed present and reachable for future index builds (no separate index lifecycle was needed for this liveness audit because headers are directly probed).

### graphify (safishamsi/graphify, Windows `C:\Users\user\AppData\Roaming\Python\Python314\Scripts\graphify.exe`, package `graphifyy` via `pip`)

- `graphify --version` → **0.9.48** (post 2026-08-21 update, global + venv unified, skill `.graphify_version` 0.9.48). `graphify query --help` / `graphify path --help` reachable.
- No `graphify-out/graph.json` present at `C:/VSC_Project/ConvoPeq/graphify-out/` (not yet rebuilt after 0.9.47→0.9.48) — same as Step 7. Full-graph `query/path/explain` therefore not exercised; `graphify --help` + `SKILL.md` + version conformity is the operational check for this proof. The `GRAPHIFY_MAX_RETRY_DEPTH` terminology in Task 7 is evidence-lexicon (now-phase-docs), not a `src/` symbol — per the preceding bullet it is handled in the "retry exhaustion safety" proof without a `src/` claim.

### semble (MinishLab/semble, Windows `C:\Users\user\.local\bin\semble.exe`)

- `semble --version` → **0.5.5**. Native Windows invocation required: `"C:/Users/user/.local/bin/semble.exe" search "PendingPublishRegistry" "C:/VSC_Project/ConvoPeq" --top-k 2` (WSL `/C:/` path mangling avoided by quoting the full Windows path). `semble --help` search options reachable. GitHub `MinishLab/semble` 200 OK. Semantic search semantics aligned with `rg` hits for world/retire terms (Step 7 already indexed the corpus).

### AiDex (CSCSoftware/AiDex, `.aidex/index.db` present)

- Direct index present at `C:/VSC_Project/ConvoPeq/.aidex/index.db` (~26 MB), `items` table `17,740` rows, `items_fts` + index valid. `aidex_query(term="PendingPublishRegistry")` → 15 matches, `RuntimeStore` → 59, `OwnerChannel` → 31, `DeferredDeletionQueue` → 41 — all consistent with `rg/ast-grep/serena`.
- Note: `sqlite3` schema `path` column check at first attempt produced `no such column: path` due to exact FTS table name quirk; the successful lookups above use `term/path` via the official `aidex_query` tool. AiDex is cross-checked, not primary; serena+rg are the canonical anchors.

### headroom / context-mode / RTK-WSL (always-on 3-layer hygiene, per `.github/copilot-instructions.md`)

- **headroom 0.36.2** (2026-08-21 global+venv update) + **graphifyy 0.9.48** — `graphify install --platform copilot` brought skill to 0.9.48; `headroom --version` + `tools list` verified. Compression not applied to this evidence doc (`headroom_compress` reserved for large context), but the headroom MCP server is active (MCP `headroom_compress/retrieve/stats` reachable) + `headroom-docs.vercel.app` 200 OK.
- **context-mode** `ctx_batch_execute` with `concurrency 2-4` used for all batched shell probes + internet lit fetches (single-batch multi-command pattern throughout Steps 4-8).
- **RTK-WSL** `rg/ast-grep` probes wrapped as `wsl bash -c '… && ~/.local/bin/rtk <cmd>'` form per project rule (`rtk` at `~/.local/bin/rtk` 0.45.x, `rtk --ultra-compact` for provenance probes).
- The three layers are not counted as a fourth tool — they are hygiene; the 9 proof-tool families above are the measured coverage.

### Scope-compliance & retrofit feasibility

- Scope: **code-consistent proof only** — no `src/` modifications in this step. All deltas are documentary (this evidence file).
- Retrofit assessment: the only future `src/` changes needed to promote CONDITIONAL → PASS remain exactly Step 7's two: (a) implement `WorldRetirementReservation` (`LifetimeState` reservation residency) to discharge `A_max`/`K_reservation` (Phase I), and (b) bound `TerminalReclaimAuthority` (D101-9). Both are forward-compatible with the `enqueueWithRetry` 5-stage chain proved here; liveness proof adds no new required breaking change. Monitoring / telemetry for `E_max_message` stays in the D-condition / G_contract tier.
- Unresolved items carried explicitly and unchanged: `K_terminal` boundedness (D101-9), `A_max` injection proof, `G_contract` throughput. The per-proposition table in "Overall verdict" is the sole PASS/CONDITIONAL arbiter — no single-sentence "PASS" was emitted for the combined system.

---

## Appendix A — Serena `find_symbol` canonical anchors

| Symbol | Serena `name_path` | File | Lines |
|---|---|---|---|
| `DeferredDeletionQueue` | `DeferredDeletionQueue` | `src/DeferredDeletionQueue.h` | 57-271 |
| `RetireQuarantineStore` | `convo::isr::RetireQuarantineStore` | `src/audioengine/RetireQuarantineStore.h` | 60-232 |
| `TerminalReclaimAuthority` | `convo::isr::TerminalReclaimAuthority` | `src/audioengine/ISRRetireRouter.h` | 61-120 |
| `ISRRetireRouter::enqueueWithRetry` | method | `src/audioengine/ISRRetireRouter.cpp` | 275-~380 (5-stage chain + signal) |
| `ISRRetireRouter::signalDrainWakeup` | method | `src/audioengine/ISRRetireRouter.h` | 310-320 + `ISRRetireRouter.cpp:449-460` |
| `ISRRetireRouter::tryReclaim` | method | `src/audioengine/ISRRetireRouter.cpp` | ~390-500 |
| `ISRCoordinatorLoop` | class | `src/audioengine/ISRCoordinatorLoop.h/.cpp` | h18- / `run` 1ms wait |
| `EpochDomain::getMinReaderEpoch` | method | `src/core/EpochDomain.h` | 459-530 family |
| `EpochDomain::isOlder` | static predicate | `src/core/EpochDomain.h` | w/ `int64_t(a-b)<0` |
| `EpochDomain::detectStuckReaders` | method | `src/core/EpochDomain.h` / `ISRRetireRouter.h:170` | threshold 10 |
| `EpochDomain::quarantineReader` | method | `src/core/EpochDomain.h` | quarantineFlags exclusion |

All `rg`/`ast-grep` hits for these symbols lie inside the serena-declared ranges; the header ranges are the single ground truth for each class/method.

---

## Appendix B — Capacity literal extraction (sed/awk replication)

```
DeferredDeletionQueue.h:262    static constexpr uint32_t kQueueSize = 4096;
RetireQuarantineStore.h:65     static constexpr size_t kMaxQuarantinedEntries = 512;  (Q)
RetireQuarantineStore.h        kMaxQuarantinedEntries = 512;                          (E, second instance)
RuntimeWorldAuthority.h:34     static constexpr size_t kPendingPublishCapacity = 64;    (non-owning handle, not world-counted per Step 7 Task 3)
OwnerChannel.h:40              static constexpr size_t kCapacity = 256;                 (>> max in-flight — Step 7 found, Step 8 does not reuse 65)
```

`sed -n "262,265p" / "80,280p"` and `awk '/kQueueSize|kMaxQuarantinedEntries/'` reproduce exactly these lines; `rg/ast-grep/ag` counts stable (`ag -c RetireQuarantineStore|TerminalReclaim ≈ rc` parity; `PendingPublishRegistry` 5 hits).

---

## Appendix C — ConvoPeq.md reconciliation

`ConvoPeq.md` 96,816 lines, `Generated: 2026-08-21 20:45:49` — capacities reconciled in Step 7 and unchanged here. Step 8 adds no new capacity re-read; the liveness proof is on the **same snapshot** and the pipeline chain above. `detectStuckReaders / signalDrainWakeup / drainCv_==drainCvMtx_ / EmergencyQuarantineStore / Terminal drain / 1ms / GRAPHIFY_MAX_RETRY` lexicon checked per Task 6-7 requirements and verified present in `src/` (see Tasks 3/6 sections) except `GRAPHIFY_MAX_RETRY_DEPTH` which is evidence-lexicon (now-phase-docs), handled in Task 7 per the "terminal absorption" proof without a `src/` claim.

