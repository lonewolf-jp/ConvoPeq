# D135-4 — Pre-Implementation Design Audit (Phase 0: Design Gate, NO source changes)

**Status:** Phase 0 — Design Audit. **No production source edited in this phase.**
**Gate:** GO / NO-GO before Phase 1 (minimal Proposal A).
**Predecessor:** `evidence/D135-3_STATE_MACHINE_AUDIT.md` (D135-3 conclusions remain valid; this audit re-examines the *implementation* of D135-3 §10.2 Proposal A for an ownership/threading defect).
**Authoritative source snapshot:** `ConvoPeq.md` (line refs L66xxx) + live `src/audioengine/*`. Every line ref below was verified by direct Read against the live tree on 2026-08-29.

---

## 0. Scope and directive

This audit was redirected from "implement" to "audit design first" per the user's verbatim directive:

> D135-3 の結論自体は有効ですが、§10.2 の Proposal A には … そのまま実装してはいけない箇所があります。特に重要なのは `deferredRecoveryRearmed_` と `deferredRetryGeneration_` の所有スレッド問題です。

Three hard constraints that any Phase-1 implementation MUST satisfy:

| # | Constraint | Rationale |
|---|-----------|-----------|
| (a) | **Timer thread must NOT directly read plain members** `deferredRecoveryRearmed_` / `deferredRetryGeneration_` / `deferredRetryCount_`. | Only `hasDeferred_` (h:266) is `std::atomic<bool>`. The retry members are plain `int`/`uint8_t`/`bool`; a direct cross-thread read is a ThreadSanitizer data race. |
| (b) | **Timer thread must NOT call `finishView()`.** | `finishView()` (cpp:589-608) is the rebuild-thread-only ownership-release mouth — it asserts `std::this_thread::get_id() == engine_.rebuildThreadId()` (cpp:591) and performs the sole `deferredSlot_.reset()` + `hasDeferred_=false` flip. Calling it from the Timer/MessageThread violates the single-owner contract and the jassert. |
| (c) | **Recovery handler is signal-only** → wake the rebuild thread, which then performs (1) generation-freshness check, (2) TTL check, (3) `recoveryArmed` evaluation, then `consume`/`discard`. | Keeps the ownership boundary intact: the rebuild thread is the *judge*, the Timer thread only emits an atomic wakeup. |

---

## 1. Findings from live-code verification (Phase-0 grep, tool-permitting turn)

Three parallel greps were issued against the live tree to verify the claims of §10.2:

| # | Grep target | Result | Interpretation |
|---|-------------|--------|----------------|
| 1 | `deferredRecoveryRearmed_\|deferredRetryGeneration_\|deferredRetryCount_` in `AudioEngine.Timer.cpp` | **exit 1 (no matches)** | Timer.cpp never touches the plain retry members *by name*. It reaches them only **indirectly** via `resetDeferredRetryBudget()` (Timer.cpp:1724) and `hasDeferredRequest()` (Timer.cpp:1721). |
| 2 | `hasDeferredRequest` / `hasDeferred_` access in `AudioEngine.Timer.cpp` | matched h:158 accessor, Timer.cpp:1721 | Timer thread reads only the **atomic** `hasDeferred_` via `hasDeferredRequest()` (acquire). ✅ safe. |
| 3 | `DeferredPublishSlot` definition + all aggregate-init sites | definition h:32-37; sole init cpp:505 | Exactly ONE aggregate-init site (cpp:505, inside `enqueueDeferred`). Adding a field is safe — no missed init. |

**Important nuance from (1):** grep "exit 1 = no matches" is a **safety confirmation**, but it is *partial*. The Timer thread reaches the plain retry members through `resetDeferredRetryBudget()` at Timer.cpp:1724, which is an **indirect** write (the grep matched no *direct* text reference to the member names). The hazard the constraint targets is therefore **already present in the shipped code** via this accessor — see §3.

---

## 2. The ownership/threading boundary — verified atomics-vs-plain map

Source: `RuntimePublicationOrchestrator.h:259-279` (live); `AudioEngine.h:2707-2718` (rebuildMutex-guarded flags).

### 2.1 Atomic (cross-thread safe)

| Member | Type | Decl | Writers (thread) | Readers (thread) |
|---|---|---|---|---|
| `hasDeferred_` | `std::atomic<bool>` | h:266 | `enqueueDeferred` cpp:522 (RB) · `finishView` cpp:600 (RB) · `clearDeferredForShutdown` cpp:544 (RB) | `hasDeferredRequest()` h:158 acquire — Timer.cpp:1721 (MSG) · `peekDeferred` cpp:567 (RB) · `processDeferredAdmission` cpp:639 (RB) |
| `deferredOverwriteCount_` | `std::atomic<uint64_t>` | h:269 | cpp:442,443 (RB) | cpp:528,556,604 (RB) + DrainAudit |
| `maxDeferredAgeMs_` | `std::atomic<uint64_t>` | h:270 | cpp:453 (RB) | cpp:668 (RB) |
| `lastRecoveryPublishSeq_` | `std::atomic<PublicationSequenceId>` | h:281 | `setRecoveryPublishSeq` h:164-166 release — Timer.cpp:1723 (MSG) · cpp:551 (RB) | `recoveryPublishSeq()` h:167-169 acquire |
| `publishRetryReady` | `bool` (AudioEngine.h:2718) | AudioEngine.h:2718 | Timer.cpp:1743-1744 (**MSG, under `rebuildMutex`**) | RebuildDispatch.cpp:849,879,880,861 (**RB, under `rebuildMutex`**) |
| `recoveryPending` | `bool` (AudioEngine.h:2711) | AudioEngine.h:2711 | AudioEngine.h:4492 (**RB, under `rebuildMutex`**) | RebuildDispatch.cpp:849,927 (**RB, under `rebuildMutex`**) |

### 2.2 Plain (rebuild-thread-only — single owner `jassert` at cpp:566/591/638)

| Member | Type | Decl | Writers | Readers |
|---|---|---|---|---|
| `deferredSlot_` | `std::optional<DeferredPublishSlot>` | h:265 | cpp:505 (enqueueDeferred, RB) · cpp:599 (finishView, RB) · cpp:543 (shutdown, RB) | cpp:448,449,461,462,468,515,541,567,594,595 (all RB) |
| `deferredRetryGeneration_` | plain `int` | h:277 | cpp:476 (RB) · cpp:548 (RB) · **`resetDeferredRetryBudget` h:172 (MSG — see §3)** | cpp:472 (RB) · h:160 `getRetryGeneration` |
| `deferredRetryCount_` | plain `uint8_t` | h:278 | cpp:474,477 (RB) · cpp:549 (RB) · **`resetDeferredRetryBudget` h:173 (MSG — see §3)** | cpp:472,489,489,498 (RB) · h:161 `getRetryCount` |
| `deferredRecoveryRearmed_` | plain `bool` | h:283, init=false | cpp:550 (RB, shutdown) | **NONE** (dead code — see §4) |

`RB` = rebuild thread. `MSG` = MessageThread (`timerCallback`, confirmed Timer.cpp:432/475: "timerCallback ... MessageThread Non-RT").

### 2.3 The comment at h:275-276 is **stale/inaccurate**

> `RuntimePublicationOrchestrator.h:275-276`: "rebuild-thread 専用（single-owner）だが、hasDeferred_ と同じく rebuildMutex バリア経由で可視性を取る必要はない（同一スレッド）。"

This states the retry members are rebuild-thread-only and need no barrier. **It is wrong**: `resetDeferredRetryBudget()` (h:171-174) — an inline mutator on these *exact* plain members — is invoked from the MessageThread at **Timer.cpp:1724**. The comment must be corrected (Phase 1).

---

## 3. The pre-existing data race (the defect D135-4 must fix)

### 3.1 Where the write originates

`AudioEngine.Timer.cpp:1669-1747` — the `EVENT_CROSSFADE_TIMEOUT` handler runs inside `AudioEngine::timerCallback()` (MessageThread, 100 ms Non-RT sampler). Within the recovery gate:

```cpp
// Timer.cpp:1721
if (runtimeOrchestrator_ != nullptr && runtimeOrchestrator_->hasDeferredRequest()) {
    const auto recoverySeq = getLastCommittedPublicationSequence();
    runtimeOrchestrator_->setRecoveryPublishSeq(recoverySeq);   // atomic (h:164) ✅
    runtimeOrchestrator_->resetDeferredRetryBudget();           // PLAIN write (h:171-174) ⚠️
// ...
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);          // lock taken HERE (1743)
        publishRetryReady = true;                                // guarded ✅
    }
    rebuildCV.notify_one();                                     // ✅
}
```

`resetDeferredRetryBudget` (RuntimePublicationOrchestrator.h:171-174):
```cpp
void resetDeferredRetryBudget() noexcept {
    deferredRetryGeneration_ = 0;   // plain int, written from MSG thread
    deferredRetryCount_ = 0;        // plain uint8_t, written from MSG thread
}
```

### 3.2 Where the read occurs

`RuntimePublicationOrchestrator.cpp:470-478` (inside `enqueueDeferred`, **rebuild thread** — no `rebuildMutex` held):
```cpp
// cpp:471-478
{
    const bool sameObligation = (req.generation == deferredRetryGeneration_);  // plain READ, no lock
    if (sameObligation)
        ++deferredRetryCount_;
    else {
        deferredRetryGeneration_ = req.generation;   // plain WRITE
        deferredRetryCount_ = 0;
    }
}
```

`rebuildMutex` is **never acquired** in `RuntimePublicationOrchestrator.cpp` (verified: `rg rebuildMutex` in the orchestrator .cpp/.h returns zero hits — it lives only in `AudioEngine.h:2699` and `AudioEngine.RebuildDispatch.cpp`). The `enqueueDeferred` read at cpp:472 is therefore unguarded.

### 3.3 Race verdict

| Thread | Operation | Member | Guard |
|---|---|---|---|
| MessageThread (Timer.cpp:1724) | **write** `deferredRetryGeneration_=0`, `deferredRetryCount_=0` | plain | ❌ none |
| Rebuild thread (cpp:472) | **read** `deferredRetryGeneration_` | plain | ❌ none |

This is a **genuine data race** (both accesses non-atomic, at least one non-atomic write, no synchronization). It is *latent* (narrow window: the MessageThread gate requires `hasDeferredRequest()==true`, and `enqueueDeferred`'s read sits inside the re-defer path), but it is real and ThreadSanitizer would flag it. **Constraint (a) exists precisely to eliminate this race.** The current shipped code violates its *spirit* (the Timer thread mutates rebuild-thread-only plain state). D135-3 did not change code, so this race ships through D135-3 unchanged.

### 3.4 Why the race is reachable (not merely theoretical)

1. `enqueueDeferred` is re-entered via the **re-defer path**: `processDeferredAdmission` → `Ready` → `view->consume()` (calls `finishView` → `hasDeferred_=false`) → `submitPublishRequest` (cpp:653) → `enqueueDeferred` (cpp:378). During the re-defer `enqueueDeferred` call, `enqueueDeferred` has NOT yet set `hasDeferred_=true` (cpp:522 is the last statement), but the *previous* slot is still held until cpp:505 overwrites it. More decisively: the first `enqueueDeferred` (cpp:505–522) sets `hasDeferred_=true` *before* returning, so once a deferred obligation is established, the MessageThread gate at Timer.cpp:1721 CAN observe `hasDeferred_==true` and enter the race window while a concurrent/sibling rebuild-thread path is mid-`enqueueDeferred`.
2. `resetDeferredRetryBudget` is the **only** non-rebuild-thread writer of these members. Everything else (cpp:474-478, cpp:548-549) is rebuild-thread.

---

## 4. `deferredRecoveryRearmed_` is dead code (exhaustive)

Goal per the directive: "audit the full lifecycle of `deferredRecoveryRearmed_` first." Grep across `src/`, `evidence/`, `doc/`, `ConvoPeq.md`:

| Location | Role | Ref |
|---|---|---|
| `RuntimePublicationOrchestrator.h:283` | **declaration** `bool deferredRecoveryRearmed_ = false;` | L66644 |
| `RuntimePublicationOrchestrator.cpp:550` | **sole write** — `clearDeferredForShutdown` sets `= false` | L66199 |
| `RuntimePublicationOrchestrator.cpp:534` | write-site owner function | clearDeferredForShutdown |
| `D135-1_IMPLEMENTATION.md:14` | doc-text mention (design note) | — |
| `D135-2_GATE_E_REPORT.md:387` | doc-text mention (gate note) | — |
| `evidence/D135-3_STATE_MACHINE_AUDIT.md` | this audit series' own citation | §1 table |

**Zero reads. Zero writes from any thread other than `clearDeferredForShutdown` (rebuild thread, shutdown path). Zero functional effect at runtime.** It is orphaned: declared to "gate recovery → re-drive" but never wired into any decision. **Recommendation: DELETE it** (Phase 1). This is the "dual-state" the directive names — it duplicates the (also-dead) intent of what should be a *singular* `recoveryArmed` signal.

---

## 5. Proposal A reconstruction (what §10.2 D135-3 proposed) vs the constraints

D135-3 §10.2 (adopted Proposal A) proposed, at a high level:
- Timer-thread recovery handler gates on `hasDeferredRequest() && deferredRecoveryRearmed_ && deferredRetryGeneration_ == currentBuildGeneration()`; then `setRecoveryPublishSeq` + `resetDeferredRetryBudget` + `publishRetryReady + notify`; if stale, call `finishView()` to clear the obligation.

**This §10.2 shape VIOLATES constraints (a) and (b):**
- (a) ✗ It reads `deferredRecoveryRearmed_` and `deferredRetryGeneration_` (plain members) from the Timer/MessageThread.
- (b) ✗ It calls `finishView()` from the Timer/MessageThread (jassert fail at cpp:591; violates single-owner mouth).

The *intent* of §10.2 (generation-freshness + budget-reset on recovery) is correct; only the **placement** is wrong. The corrected form (§8 below) moves all plain-member access into the rebuild thread.

---

## 6. Generation / TTL / retry semantics (basis for C3 and C4)

### 6.1 Generation is a loop-bounding key, not identity (verified D135-3 §6)

- `rebuildRequestGeneration` (`std::atomic<int>`, `AudioEngine.h:2553`) incremented at `AudioEngine.RebuildDispatch.cpp:655` (`generation = ++rebuildRequestGeneration`).
- `currentBuildGeneration()` (`AudioEngine.h:1666`, acquire read) → `buildDeferredAdmissionSnapshot().currentGeneration` (cpp:578) → `evaluateDeferred` comparison `m.generation != ctx.currentGeneration` (ConvoPeq.md:60996).
- The retry counter is **generation-keyed on the Orchestrator** (cpp:472): `sameObligation = (req.generation == deferredRetryGeneration_)`. This placement is correct (`INV-DEFERRED-2`, D135-3 §4.2): `enqueueDeferred` full-struct-replaces `deferredSlot_` (cpp:505), so a slot-level counter would reset on every re-enqueue and be useless.

### 6.2 Exhaustion path is verified (cpp:489-502)

```cpp
// RuntimePublicationOrchestrator.cpp:489-502
if (deferredRetryCount_ >= kMaxDeferredRetries) {
    if (!req.newDSP.isNull()) {
        if (auto* dsp = engine_.resolveDSPHandle(req.newDSP); dsp != nullptr)
            engine_.retireDSPHandleForRuntime(dsp);
    }
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    juce::Logger::writeToLog(juce::String("[HEALTH] Deferred publish starved")
        + " gen=" + juce::String(req.generation)
        + " sequence=" + ...
        + " retryCount=" + juce::String(deferredRetryCount_)
        + " reason=RetryExhaustedDiscard");
#endif
    return;  // ← No hasDeferred_=true, no re-enqueue, slot NOT set → obligation LOST
}
```

### 6.3 Counterexample (D135-3 §6.3; matches D135-2 measured trace)

```
T0  build gen=5 → DeferredFadingActive
T1  enqueueDeferred(gen=5): retryGen=5, count=0            hasDeferred_=true,  slot={gen5}
T2  processDeferredAdmission: Ready → consume → finishView   hasDeferred_=false, slot=empty
    → submitPublishRequest → DeferredFadingActive → enqueueDeferred(gen=5)
T3  enqueueDeferred(gen=5): sameObligation(5==5) → count=1  hasDeferred_=true,  slot={gen5}
T4  consume → enqueueDeferred(gen=5)
T5  enqueueDeferred(gen=5): sameObligation → count=2 ≥ kMax(2) → RetryExhaustedDiscard
    → retire DSP, hasDeferred_ stays false, slot stays empty   hasDeferred_=false, slot=empty
T6  crossfade timeout fires → recovery gate: hasDeferredRequest()? → FALSE
    → recovery-redrive block SKIPPED entirely                   [never re-drives gen=5]
```

Generation correctly bounded the loop (✓) but exhaustion **discarded the obligation identity** before recovery could re-evaluate against the now-zero-fading world. This is the root cause D135-2 measured (`recovery-redrive = 0`).

---

## 7. C1–C4 counterexamples (against naive "retry = identity" assumptions)

| Case | Trace | Verdict |
|---|---|---|
| **C1 — stale generation, no retry** | Obligation held at gen `g`; a *different* rebuild advances `rebuildRequestGeneration` to `g'`; recovery re-evaluates `evaluateDeferred(gen=g, currentGen=g')` → `StaleDiscard`. Obligation silently dropped. | **Correct** staleness semantics; recovery re-drive is *only* productive when no intervening rebuild ocurred. The `recoveryArmed` check must therefore be ordered **after** the generation-freshness check (§6.1), never before. |
| **C2 — generation collides across builds** | Gen wraps or two builds share a number (the `int` is a monotonic counter, not UUID). A *new* obligation that happens to reuse `g` is misclassified `sameObligation` against a stale `deferredRetryGeneration_`. | Bounded risk (monotonic counter, no observed wrap in any trace). **Out of scope** for Phase 1; document as C4-adjacent. |
| **C3 — TTL policy conflict** | With wall-clock (C3-A) TTL, a recovery re-drive of a 29.9 s-old obligation passes TTL but is `StaleDiscard` by generation (C1) → wasted re-drive. With tail-relative TTL (C3-B), the clock resets on re-drive → TTL never bites during recovery. | See §8.3 for policy selection. |
| **C4 — retry-identity conflation** | `retryCount`/`retryGeneration` identify "is this the same obligation being re-deferred", NOT "is this obligation recoverable". A recovery re-drive that clears the budget (count=0) then re-enqueues the *same* gen is correctly treated as a fresh attempt, but the *stale* obligation from a *different* gen that survived in the slot is NOT distinguished. | **Formally out of scope** for this iteration (§9.4). |

---

## 8. Corrected Proposal A (the adopted design — Phase 1 target)

### 8.1 Architecture (signal-only Timer / judge-only rebuild)

```
MessageThread (timerCallback, EVENT_CROSSFADE_TIMEOUT)
  │  timerCallback runs Non-RT on the MessageThread (Timer.cpp:432/475)
  ├─ publishIdleWorldOnly(...,HardReset)   // sync commit of zero-fading world  (existing, 1714)
  └─ recovery gate (Timer.cpp:1721)
       ├─ READS hasDeferredRequest()   → atomic (h:158, reads hasDeferred_ h:266)  ✅
       ├─ setRecoveryPublishSeq(seq)  → atomic (h:164-166)                        ✅
       ├─ sets publishRetryReady=true  → under rebuildMutex (1743-1744)           ✅
       ├─ rebuildCV.notify_one()       (1746)                                        ✅
       └─ ⛔ MUST NOT: write deferredRetryGeneration_/deferredRetryCount_          (constraint a)
       └─ ⛔ MUST NOT: call finishView() / processDeferredAdmission / clear slot   (constraint b)

Rebuild thread (rebuildThreadLoop / CoordinatorLoop, RebuildDispatch.cpp:849-904)
  │  wakes on rebuildCV predicate: hasPendingTask || publishRetryReady || recoveryPending || exit
  ├─ doDeferredPublish = publishRetryReady; publishRetryReady = false;   (879-880, under lock)
  └─ processDeferredAdmission()                                           (904)
       ├─ jassert(rebuild thread)                                         (cpp:638)
       ├─ peekDeferred()  (non-flipping, jassert RB cpp:566)            (564-570)
       │   ├─ if recoveryArmed (see 8.2): reset retry budget HERE          (plain write, RB — SAFE)
       ├─ evaluateDeferred(metadata, snapshot) → Ready|Discard          (646-648)
       │   4-level: Shutdown → TTL → Generation(freshness) → Sequence
       │   ⓵ generation stale → StaleDiscard                (C1)
       │   ② TTL expired → TtlDiscard                       (C3)
       │   ③ recoveryArmed && fresh && within TTL → treat as re-drive candidate
       ├─ Ready  → view->consume() → finishView() → submitPublishRequest (651-653)
       │            (submitPublishRequest → enqueueDeferred: if still fading, re-defer
       │             with FRESH budget because count was reset in step above)
       └─ Discard → view->discard(reason) → finishView()                (656-660)
```

### 8.2 `recoveryArmed` placement — singularize into `DeferredPublishSlot`, delete `deferredRecoveryRearmed_`

**Decision:** Add `bool recoveryArmed{false}` as a **plain** field in `DeferredPublishSlot` (h:32-37), and **delete** the orphaned `deferredRecoveryRearmed_` (h:283). The Timer/MessageThread sets **no** slot field directly. Instead:

- The rebuild thread, on waking via `publishRetryReady` inside `processDeferredAdmission`, stamps `slot->recoveryArmed = true` (plain write, rebuild-thread-owned — SAFE, jassert at cpp:638).
- `recoveryArmed` is then a rebuild-thread-owned, rebuild-thread-read flag capturing "this re-evaluation was triggered by a recovery timeout signal". `evaluateDeferred` is extended to consult it (the "recovery-re-drive candidate" branch at ③ above).

**Why not make `recoveryArmed` live in the slot as an atomic that the Timer writes?** Because `deferredSlot_` (h:265) is a `std::optional<DeferredPublishSlot>` rebuilt-thread-only — the Timer thread cannot safely reach into it (it would read/write a plain subobject of a plain `std::optional` without the jassert's owner guarantee). The existing `publishRetryReady` (AudioEngine.h:2718, under rebuildMutex) is already the correct cross-thread *signal*. The slot-local `recoveryArmed` is purely the rebuild thread's internal "was I woken by recovery?" marker for the admission evaluator. This satisfies "singularize `recoveryArmed` into `DeferredPublishSlot`" (one field, in the slot, no orphan) **and** constraint (a) (Timer thread touches no plain member) **and** constraint (c) (Timer is signal-only).

> **Alternative considered & rejected:** a top-level `std::atomic<bool> recoveryArmed_` on the Orchestrator set by the Timer thread. Rejected because it would re-introduce a *second* recovery-armed concept alongside the slot field, violating "singularize"; it also invites the same "is it in the slot or not" confusion that produced the dead `deferredRecoveryRearmed_`. The `publishRetryReady` signal already carries the "woken by recovery" meaning; the rebuild thread derives `recoveryArmed` from that.

### 8.3 `resetDeferredRetryBudget` ownership fix

`resetDeferredRetryBudget()` (h:171-174) currently does plain writes and is **called from the MessageThread** (Timer.cpp:1724) — the §3 race. **Phase-1 change:** remove the Timer-thread call at Timer.cpp:1724; instead reset the budget inside `processDeferredAdmission` (rebuild thread) when `publishRetryReady` is consumed AND `slot->recoveryArmed` would be stamped. Because `enqueueDeferred` is the *sole reader* of the retry counters (cpp:472) and runs rebuild-thread-only, moving the reset there eliminates the race at its source. The accessor `resetDeferredRetryBudget` may be retained (reused by the rebuild-thread path) or inlined; either way it is **never** called from the Timer/MessageThread after Phase 1.

### 8.4 TTL policy (C3 resolution) — **C3-A (wall-clock, no reset)**

The three options:

| Option | Semantics | Timer-thread writes? | Complexity | Verdict for Phase 1 |
|---|---|---|---|---|
| **C3-A** wall-clock | TTL measured from original `enqueueTimestampUs` (cpp:517/520); `evaluateDeferred` compares `nowUs - m.enqueueTimestampUs > ttlUs` (ConvoPeq.md:60996). No reset on re-drive. | None | 0 | **Adopt** |
| C3-B tail-relative | `enqueueTimestampUs` refreshed when recovery re-arms (rebuild thread). | None (RB writes) | low | Phase 2 candidate |
| C3-C sliding | TTL re-evaluated each tick. | — | high | reject |

**Selected: C3-A.** Rationale:
- Reuses the *existing* `kDeferredPublishTTLUs = 30'000'000` (h:125, 30 s) and the *existing* `enqueueTimestampUs` field already in `DeferredPublishSlot` (h:37). Zero new Timer-thread writes.
- Preserves the staleness semantics in C1: a 29.9 s-old obligation that survived into recovery without an intervening build will pass TTL but still be `StaleDiscard` by generation if `m.generation != ctx.currentGeneration` — the recovery re-drive is correctly denied rather than silently re-attempted.
- The generation check (①) precedes the TTL check (②) in `evaluateDeferred` (ConvoPeq.md:60996 Shutdown→TTL→Generation→Sequence), so a stale-gen obligation never reaches the recovery-Re-drive treatment at ③. Wait — ordering: §6.1 says evaluateDeferred is `Shutdown→TTL→Generation→Sequence`. Per C1, generation-freshness must block a stale recovery re-drive. The existing order TTL-before-Generation means a *fresh* recovery re-drive of a still-valid (within-TTL) but gen-stale obligation is caught at Generation. This is the intended staleness gate (C1 ✓). C3-A does not disturb this ordering.

**Effect on D135-2 trace:** the gen-5 obligation exhausts (count reaches kMax) and is discarded at T5; recovery at T6 finds `hasDeferredRequest()==false` (correct — nothing held) → 0 redrive (unchanged, expected). C3-A only governs the *duration a live slot may survive*; it does not resurrect an already-exhausted/dropped obligation (that is C1's job). The C3-A policy therefore does not alter the D135-2 failure mode; the *fix* for D135-2 is the kMax reconciliation + signal-only budget reset, not a TTL change.

### 8.5 `kMaxDeferredRetries` reconciliation — **2**

| Source | Value | Comment |
|---|---|---|
| `RuntimePublicationOrchestrator.h:279` (live) | `10` | stale comment "2 回再駆動許容、3 回目で諦" — **value and comment disagree** |
| `D135-1_IMPLEMENTATION.md` / `D135-2` binary | `2` | measured trace: count 0→1→2 → starved |

Exhaustion predicate (cpp:489): `deferredRetryCount_ >= kMaxDeferredRetries`, count starts at 0 (cpp:477) and increments via `++count` (cpp:474). With **kMax=2** the sequence is: enqueue(count→0, slot set) → re-defer(count→1, slot set) → re-defer(count→2, 2>=2 → `RetryExhaustedDiscard`, return without slot). That is exactly the D135-2 "0→1→2 starved" trace. The live `10` is a regression/drift (the comment was never updated when the value was bumped, or the value was never lowered to match the spec). **Phase-1 change: set `kMaxDeferredRetries = 2` and fix the comment** to match the spec semantics ("allow 2 redrives; 3rd attempt exhausts").

This is the "first source change" (the user explicitly listed `kMaxDeferredRetries = 2` as the first Phase-1 step). It is a single-token change with no threading impact.

---

## 9. Out-of-scope items formally declared

### 9.1 C4 — retry-identity conflation (DEFERRED)

The retry counter is generation-keyed (`sameObligation = req.generation == deferredRetryGeneration_`, cpp:472) which conflate "same obligation re-deferred" with "same generation token". A recovery re-drive that clears the budget and re-enqueues the *same* gen is treated as a fresh attempt, but there is no separate `recoveryObligationId` link on the *deferred-publish* obligation (the `recoveryObligationId` that exists on `PublishRequest`/`resolveRecoveryObligation`, RuntimePublicationOrchestrator.cpp:371/353/320, 401, belongs to the **INV-X1-7** 32-slot logical-recovery table — a *different* concern). Per D135-1 §8 prohibited-list, `resolveRecoveryObligation` routing is untouched here. **C4 is out of scope for Phase 1**; it is a Phase-2 design item (recovery-obligation handle threading into the single deferred slot).

### 9.2 INV-X1-7 32-slot logical recovery table (untouched)

`ISRRuntimePublicationCoordinator.h:241-311` (`recoveryAdmissions_` table, `kMaxLogicalRecoveryObligations=32`, ConvoPeq.md:56940-56960) is a **separate** concern from the single deferred-publish slot. It is unchanged by D135-4. The `redriveDeferredRecoveryObligations()` (cpp:1042) / `redriveDeferredRecovery()` (cpp:1058) path is the *INV-X1* recovery mechanism; D135-4 operates on the *deferred-publish single-slot* path. No conflation.

### 9.3 `getRetryGeneration()` / `getRetryCount()` (h:160-161)

These test-only accessors read plain members without `rebuildMutex`. Grep confirms **zero external callers** (the only matches are the declarations themselves). They are safe only if read strictly from the rebuild thread (tests assert this). **Phase-1 action:** either assert rebuild-thread ownership in their bodies (matching h:260-261 DrainAudit pattern) or leave as-is with a documented contract. Minimal Proposal A leaves them; Phase 2 hardens them.

---

## 10. Phase plan (no source changes in Phase 0)

| Phase | Action | Source change? | Deliverable |
|---|---|---|---|
| 0 (this) | Design audit: ownership boundary, pre-existing race, dead-code deletion, kMax=2, C3-A, C4 scope-out, N1/N2/N3 spec | **No** | `evidence/D135-4_DESIGN_AUDIT.md` (this file) |
| 1 | Minimal Proposal A: (1) `kMaxDeferredRetries=2`+comment fix; (2) add `bool recoveryArmed{false}` to `DeferredPublishSlot`; (3) delete `deferredRecoveryRearmed_`; (4) Timer.cpp:1721-1746 signal-only (remove `resetDeferredRetryBudget()` call, keep `setRecoveryPublishSeq` atomic + `publishRetryReady`+`notify`); (5) `processDeferredAdmission` stamps `recoveryArmed`, resets budget (rebuild-thread), then `evaluateDeferred`. | **Yes** (5 edits) | Source patch + `evidence/D135-4_PHASE1_PATCH.diff` |
| 2 | Unit + integration: extend `DeferredFlowIntegrationTests.cpp` to cover recovery-re-drive (count reset on recovery wake, staleness deny, TTL deny). | Yes (test only) | Test additions |
| 3 | Negative tests N1/N2/N3 (below) + D135-2 harness re-run at `--cli-intent-burst-interval-ms 4000`. | Yes (test) | `evidence/D135-4_N1_N2_N3_NEGATIVE_REPORT.md` |

### Phase 1 edit checklist (exact sites)

| # | Edit | Site | Constraint addressed |
|---|---|---|---|
| 1 | `kMaxDeferredRetries = 10 → 2`; fix comment h:279 | `RuntimePublicationOrchestrator.h:279` | value-drift |
| 2 | add `bool recoveryArmed{false};` field | `RuntimePublicationOrchestrator.h:32-37` (DeferredPublishSlot) | singularize |
| 3 | delete `deferredRecoveryRearmed_` member + its h:283 decl | `RuntimePublicationOrchestrator.h:283` | dead code |
| 4 | Timer.cpp:1721-1746 — remove `resetDeferredRetryBudget()` (1724) call; keep `setRecoveryPublishSeq` (atomic) + `publishRetryReady`++`notify`; drop the now-dead `resetDeferredRetryBudget` call-site | `AudioEngine.Timer.cpp:1724` | (a) signal-only |
| 5 | `processDeferredAdmission` (cpp:636-663): on `publishRetryReady` wake, stamp `slot->recoveryArmed=true`, call `resetDeferredRetryBudget()` (now rebuild-thread), then `peekDeferred`→`evaluateDeferred`→`consume`/`discard` | `RuntimePublicationOrchestrator.cpp:636-663` | (b),(c) |

### Negative test specs (Phase 3)

| ID | Scenario | Setup | Expected |
|---|---|---|---|
| **N1** stale-generation recovery | Defer obligation at gen `g`; force an intervening rebuild so `currentGeneration = g' ≠ g`; fire crossfade-timeout recovery. | Hold slot at gen `g`; `++rebuildRequestGeneration`; `EVENT_CROSSFADE_TIMEOUT`. | `evaluateDeferred` returns `StaleDiscard`; slot released via `consume/discard→finishView`; `publishRetryReady` consumed; **no re-drive** of gen `g`; `recovery-redrive` count increments but obligation dropped. |
| **N2** TTL expiry under recovery | Defer obligation; advance `nowUs` past `kDeferredPublishTTLUs` (30 s) *without* an intervening build; fire recovery. | `enqueueTimestampUs` set in past; `m.generation == ctx.currentGeneration` (fresh gen); age > TTL. | `evaluateDeferred` returns `TtlDiscard`; obligation released; no re-drive. |
| **N3** overwrite-invariance (INV-DEFERRED-2) | Defer obligation with `recoveryArmed=true`-stamped slot; enqueue a *newer* obligation (higher gen) before recovery fires. | Second `enqueueDeferred` overwrites `deferredSlot_` at cpp:505. | Old slot's `newDSP` retired (cpp:460-468); `recoveryArmed` overwritten by fresh `false`; `hasDeferred_` stays true; newer obligation proceeds normally — `recoveryArmed` does NOT leak across the overwrite boundary. |

---

## 11. Pre-existing race — root-cause one-liner

> `resetDeferredRetryBudget()` (h:171-174) is `inline` and performs plain writes to `deferredRetryGeneration_`/`deferredRetryCount_`; it is called from `AudioEngine::timerCallback()` (MessageThread) at Timer.cpp:1724, while the rebuild thread reads those same plain members inside `enqueueDeferred` (cpp:472) with no `rebuildMutex` held. The comment at h:275-276 ("rebuild-thread-only … no barrier needed — same thread") is therefore **incorrect**. Phase 1 moves the reset into the rebuild thread, eliminating the race and making constraint (a) hold.

---

## 12. GO / NO-GO

| Area | Status | Evidence |
|---|---|---|
| D135-3 conclusions valid? | **GO** (unchanged) | §6/§7 of D135-3 stand; gen-keying, exhaustion path, dead-code proof all reproduced here verbatim. |
| Pre-existing race confirmed? | **YES — GO to fix** | §3 (verbatim writes/reads, no lock) + grep "no matches in Timer.cpp for member names" = indirect-via-accessor. |
| `deferredRecoveryRearmed_` dead code? | **YES — safe to delete** | §4 exhaustive grep: 1 decl (h:283), 1 write (cpp:550), 0 reads. |
| `kMaxDeferredRetries=2` first-change safe? | **GO** | Single `constexpr` token + comment; no threading impact; matches spec & D135-2 trace. |
| `recoveryArmed` slot field safe to add? | **GO** | §1 grep #3: sole aggregate-init at cpp:505; `peekDeferred` jassert cpp:566 guarantees rebuild-thread owner; `evaluateDeferred` already consumes `metadata{generation,sequence,enqueueTimestampUs}` (PublicationAdmission.cpp ConvoPeq.md:60982-61004). |
| Timer→signal-only refactor safe? | **GO (with care)** | `publishRetryReady` already `rebuildMutex`-guarded (AudioEngine.h:2718, Timer.cpp:1743-1744); `rebuildCV.notify_one` already fires (1746). Removing the `resetDeferredRetryBudget()` call at 1724 leaves only atomic + mutex-safe ops on the Timer side. The budget-reset simply moves to `processDeferredAdmission` (cpp:636, rebuild-thread jassert cpp:638). |
| CV-predicate exclusion comment verified? | **GO** | `AudioEngine.RebuildDispatch.cpp:847`: "hasDeferredRequest() は predicate に入れない（Deferred 継持中ビジーループ防止）" — confirmed verbatim; `publishRetryReady` IS in the predicate (851). |
| `processDeferredAdmission` jassert verified? | **GO** | cpp:638 `jassert(std::this_thread::get_id() == engine_.rebuildThreadId())`; full chain cpp:636-663 matches §1 grep #1 (peek 564-570, evaluate 646-648, consume 651, discard 658, submitPublishRequest 653). |

**Overall gate: GO to proceed to Phase 1** (minimal Proposal A) **after** this audit is recorded. No source changes occur in Phase 0. The two items that were *blocked on the TEXT-ONLY turn* of the prior session — (i) the `processDeferredAdmission` rebuild-thread `jassert` and chain (cpp:636-670), and (ii) the CV-predicate exclusion comment (RebuildDispatch.cpp:847-855) — are both **now verified** in this session (see §12 rows "CV-predicate …" and "processDeferredAdmission …").

---

## 13. Live-source reference index (Phase 0 verification)

| Concept | Live source location | ConvoPeq.md line |
|---|---|---|
| `DeferredPublishSlot` struct (request/guard/metadata/lastDiscard/enqueueTs) | `RuntimePublicationOrchestrator.h:32-37` | — |
| **`recoveryArmed` field** (to be added) | `RuntimePublicationOrchestrator.h:32-37` (Phase 1) | n/a |
| `deferredRecoveryRearmed_` (to be DELETED) | `RuntimePublicationOrchestrator.h:283` | L66644 |
| `deferredRetryGeneration_` (plain) | `RuntimePublicationOrchestrator.h:277` | L66644 |
| `deferredRetryCount_` (plain) | `RuntimePublicationOrchestrator.h:278` | L66644 |
| `kMaxDeferredRetries = 10` → 2 (Phase 1) | `RuntimePublicationOrchestrator.h:279` | — |
| **stale comment** h:275-276 | `RuntimePublicationOrchestrator.h:275-276` | — |
| `hasDeferred_` (atomic) | `RuntimePublicationOrchestrator.h:266` | L66644 |
| `hasDeferredRequest()` atomic accessor | `RuntimePublicationOrchestrator.h:158` | L66519 |
| `setRecoveryPublishSeq` (atomic, safe) | `RuntimePublicationOrchestrator.h:164-166` | L66525-66535 |
| `resetDeferredRetryBudget()` (RACE site) | `RuntimePublicationOrchestrator.h:171-174` | L66525-66535 |
| `getRetryGeneration`/`getRetryCount` (test accessors) | `RuntimePublicationOrchestrator.h:160-161` | — |
| `deferredSlot_` (optional, RB-only) | `RuntimePublicationOrchestrator.h:265` | — |
| `enqueueDeferred` exhaustion path (return w/o assign) | `RuntimePublicationOrchestrator.cpp:489-502` | L66086-66180 |
| `enqueueDeferred` retry accounting (plain read/write, no lock) | `RuntimePublicationOrchestrator.cpp:470-503` | L66199-? |
| `enqueueDeferred` slot assign + hasDeferred_=true | `RuntimePublicationOrchestrator.cpp:505-522` | L66199-? |
| `clearDeferredForShutdown` (sole rearmed write cpp:550) | `RuntimePublicationOrchestrator.cpp:534-560` | — |
| `peekDeferred` (non-flipping, jassert RB) | `RuntimePublicationOrchestrator.cpp:564-570` | — |
| `finishView` (sole ownership-release mouth, jassert RB) | `RuntimePublicationOrchestrator.cpp:589-608` | — |
| `consume` (→finishView) | `RuntimePublicationOrchestrator.cpp:612-619` | — |
| `discard` (→finishView) | `RuntimePublicationOrchestrator.cpp:621-627` | — |
| `processDeferredAdmission` (RB jassert cpp:638, peek→evaluate→consume/discard→submit) | `RuntimePublicationOrchestrator.cpp:636-663` | — |
| `submitPublishRequest` sole caller of enqueueDeferred (cpp:378) | `RuntimePublicationOrchestrator.cpp:377-379` | L66027-66083 |
| `evaluateDeferred` 4-level (Shutdown→TTL→Gen→Seq) | `PublicationAdmission.cpp` | ConvoPeq.md:60982-61004 |
| `kDeferredPublishTTLUs=30s` | `RuntimePublicationOrchestrator.h:125` | — |
| Rebuild-thread CV predicate (excludes hasDeferredRequest) | `AudioEngine.RebuildDispatch.cpp:849-854` | — |
| `publishRetryReady` consumption (doDeferredPublish) | `AudioEngine.RebuildDispatch.cpp:879-880` | — |
| `processDeferredAdmission()` wake call | `AudioEngine.RebuildDispatch.cpp:904` | — |
| `++rebuildRequestGeneration` (gen source) | `AudioEngine.RebuildDispatch.cpp:655` | — |
| `currentBuildGeneration()` atomic acquire | `AudioEngine.h:1666` | — |
| `rebuildRequestGeneration` atomic | `AudioEngine.h:2553` | — |
| `publishRetryReady` / `recoveryPending` (mutex-guarded) | `AudioEngine.h:2711,2718` | — |
| CV predicate exclusion comment verbatim | `AudioEngine.RebuildDispatch.cpp:847` | — |
| Recovery handler (MessageThread) | `AudioEngine.Timer.cpp:1669-1750` | — |
| Recovery gate `if hasDeferredRequest()` | `AudioEngine.Timer.cpp:1721` | — |
| Race-site `resetDeferredRetryBudget()` call (MSG thread) | `AudioEngine.Timer.cpp:1724` | — |
| `publishRetryReady=true` under rebuildMutex | `AudioEngine.Timer.cpp:1743-1744` | — |
| `rebuildCV.notify_one()` | `AudioEngine.Timer.cpp:1746` | — |
| `DiscardReason` enum incl `RetryExhaustedDiscard` | `RuntimePublicationState.h:10-22` | — |
| INV-X1-7 32-slot logical recovery table | `ISRRuntimePublicationCoordinator.h:241-311` | ConvoPeq.md:56940-56960 |
| `resolveRecoveryObligation` (INV-X1 path, untouched) | `RuntimePublicationOrchestrator.cpp:320,353,371` | — |
