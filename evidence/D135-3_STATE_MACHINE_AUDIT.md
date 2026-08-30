# D135-3 — Read-Only Forensic Audit: Deferred-Publish State Machine

**Type:** read-only / forensic / design review  
**Scope:** D135-3 audit only — **zero production source changes** (per D135-3 instruction §2: "D135-3 ではまだコードを変更しないでください")  
**Evidence base:** live source `src/audioengine/RuntimePublicationOrchestrator.{h,cpp}`, `RuntimePublicationState.h`, `AudioEngine.Timer.cpp`, `AudioEngine.RebuildDispatch.cpp`, `AudioEngine.h`; snapshot `ConvoPeq.md`; design docs `doc/work88/D135-{0,1,2}` and `doc/work88/D134_DESIGN.md`; empirical `evidence/D135-2_6burst_4000ms_diag.log`.  

**Authoring convention:** every code fact below cites the **live source** file:line. Where the ConvoPeq.md snapshot is referenced, the line is given as `ConvoPeq.md:L<line>`. Line numbers in this document refer to the live tree unless a snapshot is explicitly invoked.

---

## 0. Executive summary

The D135-1 minimal patch (retry-cap + `RetryExhaustedDiscard` + recovery re-trigger handoff) is **structurally correct** and makes **no prohibited changes**. The D135-2 empirical run confirms the bounded-retry loop works: each of gen 5–8 climbed `retryCount` 0→1→2 then emitted `RetryExhaustedDiscard`, and **no** 28-iteration busy-loop recurred (`retryCount > 2` = 0).

The single unresolved defect — and the only reason Gate E4 (`recovery-redrive`) is **NO-GO** — is an **orthogonality gap, not a code bug**:

> Retry-exhaustion sets `hasDeferred_ = false` *before* the crossfade-timeout recovery fires, so the recovery handler's `if (hasDeferredRequest())` guard (Timer.cpp:1721) is **never entered**, and `recovery-redrive` is structurally skipped (D135-2 log: 0 occurrences).

The root cause was foreshadowed in the original D135-1 design as the member `deferredRecoveryRearmed_` (h:283) — a flag intended to "hold the obligation through recovery" — but that flag was **declared and never wired**. It is dead code: **zero reads** anywhere in source, tests, or evidence.

**Verdict:** GO to proceed to **D135-4**, implementing **Proposal A** (wire `deferredRecoveryRearmed_` to hold the obligation through recovery), because (a) the flag already exists, (b) the D135-2 report explicitly recommends this approach (D135-2_GATE_E_REPORT.md:387), and (c) it is the minimal change that closes the E4 gap without altering `INV-DEFERRED-2` single-slot semantics. Two findings must be reconciled as preconditions (§10): the `kMaxDeferredRetries` value drift (source=10, spec/binary=2) and the dead `deferredRecoveryRearmed_`.

---

## 1. Latest ConvoPeq.md — code evidence (live-source cross-reference)

All references below are the **authoritative** grounding for the state-machine reconstruction. ConvoPeq.md line numbers are provided only for readers cross-referencing the snapshot.

| Concept | Live source | ConvoPeq.md snapshot |
|---|---|---|
| `kDeferredPublishTTLUs = 30s` | `RuntimePublicationOrchestrator.h:125` | L66527 |
| `hasDeferred_` (atomic bool) | `RuntimePublicationOrchestrator.h:266` | L66627 |
| `deferredSlot_` (std::optional) | `RuntimePublicationOrchestrator.h:265` | L66626 |
| `deferredRetryGeneration_` (int) | `RuntimePublicationOrchestrator.h:277` | L66638 |
| `deferredRetryCount_` (uint8_t) | `RuntimePublicationOrchestrator.h:278` | L66639 |
| `kMaxDeferredRetries` | `RuntimePublicationOrchestrator.h:279` | L66640 |
| `deferredRecoveryRearmed_` | `RuntimePublicationOrchestrator.h:283` | L66644 |
| `hasDeferredRequest()` accessor | `RuntimePublicationOrchestrator.h:158` | L66519 |
| `setRecoveryPublishSeq` (inline) / `resetDeferredRetryBudget` (inline) | `RuntimePublicationOrchestrator.h:164-166` / `h:171-174` (call sites: `AudioEngine.Timer.cpp:1723-1724`) | L66525-66535 |
| `submitPublishRequest` (DeferredFadingActive → enqueueDeferred) | `RuntimePublicationOrchestrator.cpp:377-379` | L66027-66083 |
| `enqueueDeferred` (entry → retry accounting → exhaustion → slot assign) | `RuntimePublicationOrchestrator.cpp:437-531` | L66086-66180 |
| `peekDeferred` (does NOT flip hasDeferred_) | `RuntimePublicationOrchestrator.cpp:564-570` | L66213-66219 |
| `finishView` (sole ownership release) | `RuntimePublicationOrchestrator.cpp:589-608` | L66238-66257 |
| `DeferredPublishView::consume` | `RuntimePublicationOrchestrator.cpp:612-619` | L66261-66268 |
| `DeferredPublishView::discard` | `RuntimePublicationOrchestrator.cpp:621-627` | L66270-66276 |
| `processDeferredAdmission` | `RuntimePublicationOrchestrator.cpp:636-670` | L66285-66312 |
| `clearDeferredForShutdown` (writes `deferredRecoveryRearmed_ = false`) | `RuntimePublicationOrchestrator.cpp:534-560` | L66183-66209 |
| `evaluateDeferred` (Ready/Discard only — no RetryLater) | `PublicationAdmission.cpp:60982-61004` (snapshot `PublicationAdmission.cpp:106-133`) | L60970-61003 |
| `DiscardReason` enum (incl. `RetryExhaustedDiscard`) | `RuntimePublicationState.h:10-22` | L66693-66707 |
| `DeferredPublishView` state enum + fail-fast dtor | `RuntimePublicationOrchestrator.h:52-109` | L66413-66466 |
| Rebuild-thread CV predicate | `AudioEngine.RebuildDispatch.cpp:849-854` | L39242-39330 |
| `processDeferredAdmission` wake/call site | `AudioEngine.RebuildDispatch.cpp:904` | — |
| `EVENT_CROSSFADE_TIMEOUT` recovery handler | `AudioEngine.Timer.cpp:1669-1750` | L42589-42670 |
| `rebuildRequestGeneration` atomic + `++generation` | `AudioEngine.h:2553`, `AudioEngine.RebuildDispatch.cpp:655` | — |
| `currentBuildGeneration()` | `AudioEngine.h:1666` | — |
| `INV-DEFERRED-2` overwrite retire | `RuntimePublicationOrchestrator.cpp:460-468` | L66109-66117 |

---

## 2. `deferredRecoveryRearmed_` — full reference audit (dead-code confirmation)

**Method:** exhaustive grep across `src/`, `evidence/`, `doc/`, and `ConvoPeq.md`. A member that is *written* but **never read** is dead code.

### 2.1 Declaration
```cpp
// RuntimePublicationOrchestrator.h:283
bool deferredRecoveryRearmed_{false};
```
ConvoPeq.md:66644 — identical. Comment (h:282): `// ★ D135-1: recovery 発行後の deferred re-drive を一度だけ起動するためのフラグ` ("flag to trigger deferred re-drive once after recovery"). **Intent:** gate the recovery→rebuild re-drive so it fires at most once per recovery.

### 2.2 The single write (in `clearDeferredForShutdown`)
```cpp
// RuntimePublicationOrchestrator.cpp:550
deferredRecoveryRearmed_ = false;
```
ConvoPeq.md:66199 — identical. This is the **only** write in the entire codebase. It is inside `clearDeferredForShutdown()` (cpp:534-560), which runs at shutdown. The write is therefore a no-op-by-construction: it is initialised to `false` at construction, only ever re-written to `false` at shutdown, and **never set to `true`** at any point in the live system.

### 2.3 The single read site … that does not exist
There are **zero** reads. Confirmed by:
```
$ grep -rn "deferredRecoveryRearmed_" src/ evidence/ doc/ ConvoPeq.md
src/audioengine/RuntimePublicationOrchestrator.cpp:550:    deferredRecoveryRearmed_ = false;   # WRITE (shutdown only)
src/audioengine/RuntimePublicationOrchestrator.h:283:... bool deferredRecoveryRearmed_{false}; # DECL
doc/work88/D135-1_IMPLEMENTATION.md:14:... deferredRecoveryRearmed_{false}            # doc text
doc/work88/D135-2_GATE_E_REPORT.md:387:... deferredRecoveryRearmed_=true ...         # design DISCUSSION (proposed, not implemented)
ConvoPeq.md:66199:    deferredRecoveryRearmed_ = false;                              # snapshot of cpp:550
ConvoPeq.md:66644:    bool deferredRecoveryRearmed_{false};                          # snapshot of h:283
```
The two doc mentions describe the **proposed** role ("keep `deferredRecoveryRearmed_=true` for a period after truncation" — D135-2_GATE_E_REPORT.md:387) and the **intended** member list (D135-1_IMPLEMENTATION.md:14) — neither is a read in compiled code.

### 2.4 Recovery handler — does NOT reference it
The `EVENT_CROSSFADE_TIMEOUT` handler (Timer.cpp:1669-1750) performs: `publishIdleWorldOnly` (sync commit of zero-fading world, Timer.cpp:1714) → `if (hasDeferredRequest())` guard (Timer.cpp:1721) → `setRecoveryPublishSeq` + `resetDeferredRetryBudget` (Timer.cpp:1723-1724) → `[D135] recovery-redrive` DIAG (Timer.cpp:1733) → `publishRetryReady=true` + `rebuildCV.notify_one()` (Timer.cpp:1742-1746). **It never references `deferredRecoveryRearmed_`.**

### 2.5 Conclusion
`deferredRecoveryRearmed_` is **dead code**. It was declared in D135-1 with the *intent* of gating the recovery re-drive, but the recovery handler was wired only to `hasDeferredRequest()` — not to this flag. The flag therefore records nothing and controls nothing. It is a latent defect: a member whose entire reason for existence (per its own comment) was never implemented.

---

## 3. Current state machine (reconstructed from code)

### 3.1 Ownership contract (single-thread, rebuild-thread only)
- `processDeferredAdmission` (cpp:636), `enqueueDeferred` (cpp:489-501 retry block), `peekDeferred` (cpp:566), `finishView` (cpp:589) all open with `jassert(std::this_thread::get_id() == engine_.rebuildThreadId())` — **rebuild-thread single-owner**. No audio-thread or timer-thread mutation of the slot.
- `hasDeferred_` is published `release` (cpp:522) / consumed `acquire` (cpp:158, cpp:567, cpp:589, cpp:639).

### 3.2 `DeferredPublishView` micro-state machine (h:52-109)
States: `Valid` → `Consumed` (via `consume()`, cpp:612) | `Discarded` (via `discard()`, cpp:621) | `MovedFrom` (move ctor/assign). **Invariants:**
- Destructor (h:76-81): `if (slot_ != nullptr && state_ == State::Valid) jassertfalse;` — a peeked-but-unconsumed view is a hard failure (no implicit discard).
- Move-assign (h:68-69): same fail-fast if overwriting a `Valid` view.
- `metadata()` (h:90-93): jassert `state_ == Valid`.

### 3.3 Full lifecycle table

| # | State (slot) | hasDeferred_ | Trigger | Next state | Notes |
|---|---|---|---|---|---|
| S0 | Empty | false | — | S0 | initial |
| S1 | **Pending** | true | `enqueueDeferred` non-exhaustion path (cpp:505-522): `deferredSlot_ = Slot{...}; hasDeferred_=true` | S2 | DSP handle live in slot; `deferredOverwriteCount_` bumped if prior value |
| S2 | Pending | true | `processDeferredAdmission` → `peekDeferred` → `evaluateDeferred` | S3a / S3b | rebuild-thread only |
| S3a | (slot consumed) | false | `Ready` → `consume()` → `finishView()`: `reset(slot)`, `hasDeferred_=false` → `submitPublishRequest` → `DeferredFadingActive` → `enqueueDeferred` | S1 or S4 | re-defer loop |
| S3b | (slot discarded) | false | `Discard` → `discard(reason)` → `finishView()` | S4 | obligation lost |
| S4 | **Exhausted** | false | `enqueueDeferred` retry-cap hit (cpp:489-502): `count >= kMaxDeferredRetries` → retire incoming DSP, log starved, `return` (no slot assign, hasDeferred_ stays false) | S4 (absorbing) | **obligation permanently lost; no recovery re-arm** |

### 3.4 Decision alphabet (what `evaluateDeferred` can return)
`PublicationAdmission::evaluateDeferred` (PublicationAdmission.cpp, snapshot ConvoPeq.md:60982-61004) returns **only** `DeferredDecision::Ready` or `DeferredDecision::Discard`. There is **no `RetryLater`** (h:81-86; comment: "RetryLater は利用経路ゼロ・YAGNI のため見送り"). The 4-level discard order is: Shutdown → TTL(30s) → Generation(stale) → Sequence(rollback).

### 3.5 Wake/consumption wiring
- **CV predicate** (RebuildDispatch.cpp:849-854): `hasPendingTask || publishRetryReady || recoveryPending || rebuildThreadShouldExit`. Deliberately **excludes** `hasDeferredRequest()` — comment at cpp:847: "Deferred 継続中ビジーループ防止" (busy-loop prevention).
- `publishRetryReady` is set by: (a) CoordinatorLoop when `hasDeferredRequest()` becomes true (CoordinatorLoop checks the flag, sets `publishRetryReady=true`), (b) the recovery handler (Timer.cpp:1744). Consumed at cpp:879: `doDeferredPublish = publishRetryReady; publishRetryReady = false;` → calls `processDeferredAdmission` at cpp:904.

### 3.6 `evaluateDeferred` is pure / non-mutating
It reads only the 5-field `DeferredAdmissionSnapshot` (h:607-6113: `currentGeneration`, `lastSequence`, `shutdown`, `nowUs`, `ttlUs`). **It does not mutate the slot.** Mutation happens exclusively via `View.consume()`/`View.discard()` → `owner_->finishView()`.

---

## 4. `RetryExhaustedDiscard` — ownership semantics

### 4.1 The exhaustion path (RuntimePublicationOrchestrator.cpp:489-502)
```cpp
if (deferredRetryCount_ >= kMaxDeferredRetries) {
    if (!req.newDSP.isNull()) {
        if (auto* dsp = engine_.resolveDSPHandle(req.newDSP); dsp != nullptr)
            engine_.retireDSPHandleForRuntime(dsp);       // retire the INCOMING dsp
    }
    // [HEALTH] Deferred publish starved gen=… retryCount=… reason=RetryExhaustedDiscard
    return;          // ← NO deferredSlot_ assignment; NO hasDeferred_=true
}
```
Ground truth: cpp:489-502. Snapshot: ConvoPeq.md:66138-66150.

### 4.2 What this means for ownership
1. The **incoming** `req.newDSP` (the world built for this publish attempt) is **retired** — its DSPCore is scheduled for runtime retirement.
2. `deferredSlot_` is **not assigned** (the early `return` skips the `deferredSlot_ = DeferredPublishSlot{...}` at cpp:505). The slot remains in whatever state `finishView` left it — i.e. **empty** (reset by the `consume()→finishView()` that preceded this `enqueueDeferred` call in the re-defer loop).
3. `hasDeferred_` is **not set to true**. It was already `false` (reset by `finishView` at cpp:600 in step S3a above).
4. `deferredRecoveryRearmed_` is **not written** here — the flag that was meant to mark "this obligation awaits a recovery re-drive" is never set.

### 4.3 Net effect
The obligation is **atomically and permanently released**: the DSP handle is retired, the slot is empty, the flag is false. `submitPublishRequest` (the only caller of `enqueueDeferred`, cpp:378) has already returned `void` (DeferredFadingActive branch, cpp:379). The rebuild thread resumes its CV wait with nothing pending. **There is no token, no record, no re-drive path.** The generation-`g` build intent that was deferred is gone for the lifetime of this process instance.

### 4.4 Contrast with `Discard` (non-exhaustion)
A `Discard` decision from `evaluateDeferred` (e.g. stale generation, TTL, shutdown) flows through `view->discard(reason)` → `finishView()`. That path **does** release ownership cleanly (slot reset, hasDeferred_=false, telemetry recorded with the reason). The retry-exhaustion path is *not* a `discard()` call — it bypasses the View entirely and returns from `enqueueDeferred` before the slot is ever assigned. This is the structural difference that makes exhaustion a **silent ownership loss** rather than a classified discard event.

---

## 5. Proposal A vs Proposal B — comparison

Both proposals target the same defect: **after retry exhaustion, the obligation must remain reachable by the recovery handler's `hasDeferredRequest()` gate (Timer.cpp:1721) so the zero-fading-world re-evaluation actually executes.** They differ on *how* the obligation is preserved.

### Proposal A — Hold obligation (re-arm the flag)
On retry exhaustion, **instead of retiring the incoming DSP and returning**, re-seat the request into `deferredSlot_` as a "recovery-pending" obligation and set `hasDeferred_ = true` + `deferredRecoveryRearmed_ = true`. The DSP is **kept** (not retired) so the recovery re-drive can re-attempt the same world. The recovery handler, after committing the zero-fading world, sees `hasDeferredRequest() == true` → fires `recovery-redrive` → signals rebuild → `processDeferredAdmission` → `evaluateDeferred` (generation still valid) → `Ready` → `consume` → `submitPublishRequest` → `evaluate` sees `hasFading == 0` (recovery world live) → `Accepted` → publish. TTL (30s, h:125) remains the backstop: if recovery never comes, the held obligation is `StaleDiscard`ed by the TTL check.

| Aspect | Proposal A |
|---|---|
| New data structures | none (reuse `deferredSlot_`; add a `recoveryArmed` bool to `DeferredPublishSlot`) |
| DSP handle lifecycle | **kept** on exhaustion (re-used on recovery re-drive) |
| Ownership model | single-slot preserved (INV-DEFERRED-2 intact); slot simply stays `Valid` longer |
| State-machine change | new `RecoveryPending` sub-state on the slot; `hasDeferred_` stays true |
| TTL backstop | yes — `evaluateDeferred` TTL check (cpp:60991-60993) evicts a stale-held slot |
| Flag reuse | `deferredRecoveryRearmed_` is finally wired (it was declared for exactly this) |
| Risk | overwrite of a `RecoveryPending` slot by a newer enqueue must be defined explicitly |

### Proposal B — Release + token
On retry exhaustion, **release everything as today** (retire DSP, reset slot, `hasDeferred_=false`) but emit a lightweight **recovery token** — a bounded record `{generation, sequence, lastEnqueueTimestampUs}` — into a separate single-slot `recoveryToken_` field. The recovery handler checks `recoveryToken_` instead of `hasDeferredRequest()`. If present, it constructs a fresh `enqueueDeferred` attempt (re-snapshotting `rebuildRequestGeneration` and building a new world against the now-zero-fading runtime). The retry budget is reset.

| Aspect | Proposal B |
|---|---|
| New data structures | `recoveryToken_` (single-slot struct) |
| DSP handle lifecycle | **retired** at exhaustion; re-drive builds a *new* DSP (fresh `buildRuntimePublishWorld`) |
| Ownership model | slot fully released; token is a "try again later" marker only |
| State-machine change | no slot-state change; new token field + new recovery gate |
| TTL backstop | token needs its own TTL / bounded age guard (or reuse slot TTL) |
| Flag reuse | `deferredRecoveryRearmed_` **removed** (not needed) |
| Risk | re-build may pair the deferred intent with a world that already advanced generation → must validate freshness; wider scope (new build path on timer thread) |

### Comparison matrix

| Criterion | Proposal A (hold) | Proposal B (release + token) |
|---|---|---|
| Lines of code | small (re-seat + flag wire + overwrite guard) | larger (new token struct + token-aware recovery gate + re-build path) |
| Reuses existing DSP | yes (kept on exhaustion) | no (retired; new build on re-drive) |
| Generation identity preserved | yes (same slot, same generation key) | no (fresh enqueue → fresh generation snapshot) |
| `INV-DEFERRED-2` single-slot | unchanged | unchanged (token is orthogonal) |
| `deferredRecoveryRearmed_` | **wired in** (fulfils declared intent) | **deleted** (removes dead code) |
| Re-drive uses zero-fading world | the held world re-evaluated | a freshly-built world |
| Backstop for indefinite hold | TTL `StaleDiscard` (h:125, 30s) | token age guard needs specification |
| Risk of publishing a *stale* snapshot on re-drive | low (world frozen at deferral, validated by eval) | medium (new world build may race) |

---

## 6. Generation-semantics audit (with counterexample)

### 6.1 How the generation key works
- `rebuildRequestGeneration` (AudioEngine.h:2553) is a live `std::atomic<int>`, incremented at `RebuildDispatch.cpp:655` (`generation = ++rebuildRequestGeneration`) on each new rebuild intent.
- `currentBuildGeneration()` (AudioEngine.h:1666) is an acquire read of that same atomic — it is what `buildDeferredAdmissionSnapshot().currentGeneration` (cpp:578) feeds to `evaluateDeferred`.
- A `PublishRequest` carries `generation` (PublicationAdmission.h:6033), captured at build-seal time. `enqueueDeferred` stamps `DeferredPublishMetadata::generation = req.generation` (cpp:515) and `DeferredGuard::generation = req.generation` (cpp:508).
- The retry counter is **generation-keyed on the Orchestrator** (cpp:472): `sameObligation = (req.generation == deferredRetryGeneration_)`; same → `++count`, new → `count=0, key=req.generation`. This placement is correct (D135-0 P1): `enqueueDeferred` does a full `DeferredPublishSlot` struct-replace (INV-DEFERRED-2, cpp:505), so a slot-level counter would reset to 0 every re-enqueue and be useless.

### 6.2 Why generation is NOT a stable identity across recovery
The generation key answers *"is this re-defer the same obligation, or a new one?"* — it is a **loop-bounding key**, not an **identity handle**. Two invariants it does **not** provide:
- **Recovery-epoch identity:** there is no "was this deferred by a recovery re-drive?" bit. After `resetDeferredRetryBudget()` (Timer.cpp:1724; inline at `RuntimePublicationOrchestrator.h:171-174`) zeroes `deferredRetryGeneration_` and `deferredRetryCount_`, the next `enqueueDeferred` — even for the *same* gen-`g` request re-driven by recovery — will see `req.generation != 0` (the slot was empty), set `deferredRetryGeneration_ = req.generation`, `count = 0`. So recovery *does* reset the counter for the held obligation. The problem is that **exhaustion destroys the obligation before recovery runs**, so the reset never gets a chance to re-drive.
- **Freshness across interleaving builds:** `evaluateDeferred` compares `m.generation != ctx.currentGeneration` (cpp:60996). If a *different* rebuild advanced `rebuildRequestGeneration` between the deferral and the recovery re-evaluation, the held gen-`g` slot is `StaleDiscard`ed — the recovery re-drive is silently discarded. This is *correct* staleness semantics, but it means a recovery re-drive is only productive when no intervening rebuild occurred (see C2).

### 6.3 Counterexample to "generation guarantees recovery re-drive"
```
T0  build gen=5  →  worldFadingUuid=3  →  DeferredFadingActive
T1  enqueueDeferred(gen=5): retryGen=5, count=0                hasDeferred_=true,  slot={gen5}
T2  processDeferredAdmission: Ready → consume → finishView     hasDeferred_=false, slot=empty
    → submitPublishRequest → DeferredFadingActive → enqueueDeferred(gen=5)
T3  enqueueDeferred(gen=5): sameObligation(5==5) → count=1     hasDeferred_=true,  slot={gen5}
T4  processDeferredAdmission → consume → enqueueDeferred(gen=5)
T5  enqueueDeferred(gen=5): sameObligation → count=2 ≥ kMax(2) → RetryExhaustedDiscard
    → retire DSP, hasDeferred_ stays false, slot stays empty   hasDeferred_=false, slot=empty
T6  crossfade timeout fires → recovery handler: hasDeferredRequest()? → FALSE
    → recovery-redrive block SKIPPED entirely                   [never re-drives gen=5]
```
This is the **exact** measured D135-2 trace (gen 5–8, each `retryCount 0→1→2 → starved`, `recovery-redrive` = 0, D135-2_GATE_E_REPORT.md:70-99). The generation key correctly bounded the loop (✓), but the exhaustion path **discarded the obligation identity** before recovery could re-evaluate it against the zero-fading world. Generation semantics do not survive exhaustion.

---

## 7. C1–C4 counterexamples (concrete failure traces)

Each counterexample is a trace over the *current* (D135-1) source showing an obligation lost under a distinct mechanism. All four motivate D135-4.

### C1 — RetryExhaustedDiscard loses the obligation before recovery re-arm
**Trace:** §6.3 above (gen=5: count 0→1→2 → `RetryExhaustedDiscard` at cpp:489 → `return` at cpp:501 → `hasDeferred_` stays false).
**Violated invariant:** Recovery handler precondition (Timer.cpp:1721 `hasDeferredRequest()`).
**Consequence:** the gen-5 build intent (DSP built, world sealed) is retired and dropped; the recovery's zero-fading world is never paired with it. **Measured** in D135-2: `recovery-redrive` = 0 (D135-2_GATE_E_REPORT.md:125-132, 212-215). This is the E4 failure.

### C2 — Generation advancement defeats the recovery re-drive (StaleDiscard races re-evaluation)
**Trace:**
```
T0  gen=5 deferred held (hasDeferred_=true, slot={gen5, retryArmed=true})   ← Proposal A present
T1  a NEW rebuild intent fires (e.g. user input / auto-rebuild) → ++rebuildRequestGeneration → 6
T2  crossfade timeout → recovery: hasDeferredRequest()? TRUE → recovery-redrive fires
     → publishRetryReady → rebuild thread → processDeferredAdmission
T3  peekDeferred → evaluateDeferred: m.generation(5) != ctx.currentGeneration(6) → StaleDiscard
     → discard(StaleDiscard) → finishView → hasDeferred_=false
```
**Violated invariant:** None in the current code (StaleDiscard is *correct* for a stale generation). The counterexample shows that **Proposal A alone does not guarantee re-drive success** — it only guarantees the recovery branch is *entered*; the re-evaluation may still StaleDiscard it if a newer generation landed.
**Consequence:** recovery-redrive log line fires (E4 observed) but the obligation is still lost to staleness. Mitigation: Proposal A must additionally require `deferredRetryGeneration_ == currentBuildGeneration()` at the recovery gate (a freshness check) so recovery only re-drives a still-current obligation — otherwise it should clear the held obligation (it cannot catch up). This is a **design requirement for D135-4**, not a bug in D135-1.

### C3 — TTL expiry evicts the obligation during the recovery wait
**Trace:**
```
T0  gen=5 deferred (enqueueTimestampUs = T0)                      hasDeferred_=true
T1  retry loop: count 0→1→2 → (Proposal A) held as RecoveryPending  hasDeferred_=true
T2  crossfade timeout interval approaches/elapses; recovery is slow
T3  processDeferredAdmission (woken by ANY publishRetryReady, incl. a routine rebuild tick)
    → evaluateDeferred: ageUs = now − T0 > kDeferredPublishTTLUs (30s, h:125) → StaleDiscard
```
**Violated invariant:** TTL is unconditional (cpp:60991-60993; snapshot L60970-60993) — it does not distinguish "held-for-recovery" from "stale."
**Consequence:** a long-held recovery-pending obligation can be TTL-evicted before recovery fires, re-introducing the C1 loss under a different mechanism. Mitigation for D135-4: either pause the TTL clock for a `RecoveryPending` slot, or use a longer/shorter recovery-tail TTL. (Note: with the **current** source value `kMaxDeferredRetries=10` the loop spins up to 11×, *increasing* TTL pressure — see §10 finding F2.)

### C4 — Single-slot overwrite conflates "retry" with "different intent" (INV-DEFERRED-2)
**Trace:**
```
T0  gen=5 build #1 → DeferredFadingActive → enqueueDeferred: retryGen=5, count=0   hasDeferred_=true, slot={gen5, DSP_A}
T1  before re-drive, gen=5 build #2 (same generation! e.g. re-trigger) → DeferredFadingActive → enqueueDeferred:
     sameObligation(5==5) → ++count(=1)  ← counts as a "retry" but is actually a DIFFERENT intent
     → INV-DEFERRED-2 overwrite (cpp:460-468): retire DSP_A, slot={gen5, DSP_B}
T2  ... count climbs 0→1→2 → RetryExhaustedDiscard on DSP_B; DSP_A was already retired at T1
```
**Violated invariant:** INV-DEFERRED-2 (single-slot latest-only, D134_DESIGN.md:255, h:280-292) is *preserved* — overwrite always retires the old DSP. The counterexample shows that the **retry counter is conflated with the overwrite counter**: a fresh same-generation build increments `deferredRetryCount_` as if it were a re-drive of the *same* obligation, when it is a *different* world being swapped in.
**Consequence:** (a) the retry budget is consumed faster than semantics warrant (a brand-new gen-5 build burns gen-5's retry budget), and (b) DSP_A's world is retired before it was ever attempted against the zero-fading recovery world. This is an inherent limitation of the single-slot design — Proposal A inherits it; Proposal B sidesteps it (token is intent-agnostic). D135-4 must decide whether the retry counter keys on `generation` alone or on `(generation, sequence)`/a true obligation id.

---

## 8. Adopted proposal — A (hold obligation, wire `deferredRecoveryRearmed_`)

**Rationale for adoption:**
1. `deferredRecoveryRearmed_` is **already declared** (h:283) and was explicitly intended for this role (its own comment). Wiring it in is the smallest change that closes E4.
2. D135-2_GATE_E_REPORT.md:387-388 explicitly recommends it: *"打ち切り後も `deferredRecoveryRearmed_=true` を一定期間維持して recovery 後の re-drive を受け付ける"* — this is precisely Proposal A.
3. It preserves `INV-DEFERRED-2` single-slot semantics and generation identity (C4 is inherited, not worsened).
4. It keeps the DSP handle live so the recovery re-drive reuses the *actual* deferred world rather than rebuilding (lower risk than B's fresh build against an advanced generation — C2).
5. TTL (C3) provides a bounded backstop; no new unbounded state.

**Required refinements to satisfy C2/C3/C4 (design inputs for D135-4):**
- R1 (C2): the recovery re-drive must be gated on generation freshness — only re-drive when `deferredRetryGeneration_ == currentBuildGeneration()`; otherwise clear the held obligation (it cannot be re-driven against a zero-fading world of a *different* generation).
- R2 (C3): a `RecoveryPending` slot must either pause the 30s TTL clock or carry its own recovery-tail TTL, so C3 cannot pre-empt C1's re-drive.
- R3 (C4): if the retry counter is to remain generation-only, document that same-generation overwrites consume the budget (acceptable); if a true obligation id is desired, key `deferredRetryGeneration_` on `(generation, sequence)` — but this is a **semantics change** (out of D135-3 scope; carry to D135-4 design review).

---

## 9. Non-adopted proposal — B (release + token)

**Rejected because:**
1. **Scope creep:** requires a new `recoveryToken_` data structure, a token-aware recovery gate, and — critically for B — a *fresh* `enqueueDeferred` on the timer/recovery path that re-builds a world. That re-build path on the (non-audio) timer thread (already sync-blocking, Timer.cpp:1714) risks pairing the deferred intent with a generation that already advanced (C2), producing a publish against a world that may have superseded it. Proposal A avoids this by re-evaluating the *already-frozen* deferred world.
2. **DSP re-lifecycle hazard:** B retires the DSP at exhaustion then expects a fresh build to re-create it. The re-created world is not guaranteed equivalent to the deferred world (crossfade state, DSP topology may have changed). A's reuse of the frozen slot is safer.
3. **Token TTL needs specification:** B must define its own age bound (R2) — an extra design surface with no existing primitive to lean on.
4. `deferredRecoveryRearmed_` would become truly unused and should be deleted under B; under A it is fulfilled.

**Not wrong — just wider:** B is the "clean slate" option if D135-4 decides to abandon the single-slot hold model entirely (it would also naturally address C4 by keying the token on a true obligation id). It is recorded here for completeness and as a future option.

---

## 10. D135-4 — implementation specification (handoff)

The following is a **handoff spec**, not an implementation. D135-3 makes no source changes.

### 10.1 Preconditions (blockers to fix before D135-4 coding)

**F1 — `kMaxDeferredRetries` value drift (MUST reconcile).**
Current source: `RuntimePublicationOrchestrator.h:279` — `static constexpr uint8_t kMaxDeferredRetries = 10;` with comment *"D135-1: 2 回再駆動許容、3 回目で諦"* (stale).  
Intended (spec + binary): `2` — per `D135-1_IMPLEMENTATION.md` §2 and the D135-2 empirical log (`retryCount 0→1→2 → starved`, D135-2_GATE_E_REPORT.md:70-99).  
**D135-4 action:** set `kMaxDeferredRetries = 2` and fix the comment. Rationale: 2 was empirically validated (loop bounded, no 28-loop recurrence, D135-2 §4 "最大同一 generation 連続 re-drive = 2"); 10 would spin up to 11× (raising C3 TTL pressure and latency) and contradicts the D135-1 design intent.

**F2 — `deferredRecoveryRearmed_` is currently dead (DECLARED, NOT WIRED).**  
D135-4 must either (A) wire it per §10.2, or (B) delete it. The audit recommends (A).

### 10.2 Proposal-A implementation spec

**Change set (production):**

| File | Site | Change |
|---|---|---|
| `RuntimePublicationState.h` | `DeferredPublishSlot` struct | add `bool recoveryArmed{false};` field (slot carries the recovery-pending marker) |
| `RuntimePublicationOrchestrator.h` | `peekDeferred` doc / `processDeferredAdmission` | document that a `recoveryArmed` slot stays `Valid` (not `Consumed`/`Discarded`) until a rebuild re-evaluates it |
| `RuntimePublicationOrchestrator.cpp` | `enqueueDeferred` exhaustion path (cpp:489-502) | **replace** the retire+return with: do **not** retire the incoming DSP; re-seat it as a `RecoveryPending` slot (`deferredSlot_ = Slot{req,…, .recoveryArmed=true}`), set `hasDeferred_=true` (release), set `deferredRecoveryRearmed_=true` (recovery gate), log `[HEALTH] Deferred publish starved, held for recovery re-drive`. Keep `deferredRetryCount_` at the capped value (do **not** reset here; recovery resets it). |
| `AudioEngine.Timer.cpp` | recovery handler gate (Timer.cpp:1721) | **strengthen** the gate: `if (hasDeferredRequest() && deferredRecoveryRearmed_ && deferredRetryGeneration_ == engine_.currentBuildGeneration())` — implements R1 (C2 freshness). Only then `setRecoveryPublishSeq` + `resetDeferredRetryBudget` + `publishRetryReady` + notify. If `hasDeferredRequest()` but `deferredRetryGeneration_ != currentBuildGeneration()`, clear the held obligation (`finishView`) — it is stale and cannot be re-driven. |
| `RuntimePublicationOrchestrator.cpp` | `processDeferredAdmission` (cpp:636) | on `Ready` for a slot whose `recoveryArmed==true`, clear `recoveryArmed` and `deferredRecoveryRearmed_` after consume (the recovery re-drive has been consumed). Add a DIAG line `[D135] recovery re-drive gen= currentGen= worldFadingUuid=`. |
| (optional, C3) | `evaluateDeferred` / `buildDeferredAdmissionSnapshot` | extend the snapshot with `recoveryPendingAgeUs` and lengthen/evade TTL for a `recoveryArmed` slot, OR add a `recoveryTailTTLUs` constant. |

**Invariants to preserve (must NOT regress):**
- Rebuild-thread single-owner (`jassert rebuildThreadId`) — all changes above run on the rebuild thread or the (non-RT) timer thread via `publishRetryReady` signal, never inline `processDeferredAdmission` from the timer thread (Timer.cpp already signals via CV; keep it).
- `INV-DEFERRED-2` single-slot overwrite (cpp:460-468) — unchanged; the new `recoveryArmed` slot is still overwritten on a newer enqueue, retiring its DSP via the existing overwrite path.
- `DeferredPublishView` fail-fast dtor (h:76-81) — a `recoveryArmed` slot is still `Valid` between peek and consume/discard; no View-state enum expansion is required.
- `evaluateDeferred` purity (Decision-only, no Store mutation) — unchanged; the `recoveryArmed` flag is read by the Orchestrator, not by Admission.
- `submitPublishRequest → processDeferredAdmission` direct call = 0 (must remain 0 — re-drive only via RebuildDispatch.cpp:904 / Timer.cpp signal).
- `hasFadingRuntimeInWorld` (AudioEngine.h:3304), `fadingRuntimeUuid` write sole-site (RuntimeBuilder.cpp:220), `commitRuntimePublication` wait/receipt, `PublicationExecutor`, `RuntimePublicationCoordinator` — **untouched**.
- `INV-X1-7` 32-slot logical recovery obligation table (ConvoPeq.md:56940-56960, `recoveryCapacityExhaustedCount_`) — **out of scope**, distinct subsystem.
- D134-LCG `lastCommittedRebuildGeneration` — separate defect, NOT in D135.

**Build/flag constraints (from D135-3 instruction §8):**
- The `kMaxDeferredRetries=10` (source) vs `2` (spec/binary) drift MUST be resolved to 2; a value of 10 changes the retry-count semantics and would invalidate the D135-2 empirical baseline.
- DIAG lines (`[D135] recovery re-drive`, `[HEALTH] Deferred publish starved, held`) must be gated under `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` (matching the existing `[D135]`/`[HEALTH]` pattern at cpp:480-499, Timer.cpp:1725-1741).

### 10.3 Test additions (handoff to D135-4 test plan)

| Test | What it asserts |
|---|---|
| `DeferredFlowIntegrationTests.cpp` | retry-cap at `kMaxDeferredRetries=2`: gen=`g` climbs retryCount 0→1→2, then slot is `recoveryArmed=true`, `hasDeferred_=true`, DSP **not** retired. |
| | after a forced `publishIdleWorldOnly` + recovery signal, the held gen-`g` slot is re-evaluated against `fadingRuntimeUuid==0` → `Accepted` → publishes; `recovery-redrive` DIAG emitted (E4 closed). |
| | C2 gate: if `currentBuildGeneration` advanced past `g` between deferral and recovery, the held obligation is cleared (not re-driven); `RejectedStaleGeneration` stays 0 and no cross-gen publish occurs. |
| | C3 backstop: hold a `recoveryArmed` slot > 30s without recovery signal → `evaluateDeferred` TTL → `StaleDiscard` (no indefinite retention). |
| | C4: same-generation overwrite of a `recoveryArmed` slot retires the old DSP via INV-DEFERRED-2 path; retry budget resets only on generation change. |
| | `DeferredPublishView` move-only `static_assert` preserved; dtor fail-fast on `Valid` preserved. |

### 10.4 D135-4 empirical (Gate E close) target
Re-run the D135-2 6-burst harness (`--cli-intent-burst-interval-ms 4000`, 4 bursts gen5-8) and assert:
- `[D135] recovery-redrive` ≥ 1 per recovered gen (was 0).
- `gen7 publish` observed with `fadingUuid==0` post-recovery (was PASS E5; now paired with an actual re-drive of a *deferred* obligation, not a fresh rebuild).
- `retryCount > 2` = 0 (loop still bounded).
- No `28-loop`-equivalent recurrence.

---

## 11. GO / NO-GO

### 11.1 D135-1 (the patch under audit) — **GO**
- No prohibited changes (D135-2_GATE_E_REPORT.md:380-382 grep-confirmed: `Timer→processDeferredAdmission` direct call = 0; `submitPublishRequest→processDeferredAdmission` = 0; `hasFading` definition unchanged; `commitRuntimePublication` unchanged; `PublicationExecutor` unchanged; `lastCommittedRebuildGeneration` unchanged).
- Retry cap, `RetryExhaustedDiscard`, generation-keyed counter — all implement D135-0 §6 correctly.
- D135-0/D135-1 gates A–D: PASS.

### 11.2 D135-2 (empirical closure of Gate E) — **NO-GO (measurement gap, not a code defect)**
- E1 ✓ (pre-recovery fading authority), E2 ✓ (retry ≤ 2), E3 ✓ (recovery world live, fadingUuid 3→0), **E4 ✗** (`recovery-redrive` = 0 — recovery branch structurally skipped because `hasDeferred_==false` at recovery time), E5 ✓ (gen7 publish), E6 ✓ (generation freshness), 28-loop ✗-recurrence ✓.
- D135-2_GATE_E_REPORT.md:375-382: "この NO-GO は production source 変更禁止下での D135-1 設計仕様通りの動作に起因する" — the gap is the retry-truncation ↔ recovery-redrive orthogonality, exactly what §3.4/S4 identifies.

### 11.3 D135-3 (this audit) — **GO to proceed to D135-4**
- Root cause precisely located: `deferredRecoveryRearmed_` (h:283) was declared to bridge recovery and re-drive but was **never wired** (§2); the exhaustion path (cpp:489-502) hard-releases the obligation, so the recovery gate (Timer.cpp:1721) never fires (§4/S4).
- Adopted fix (Proposal A, §8) is minimal and leverages the existing dead flag + existing TTL backstop + existing `publishRetryReady` CV handoff.
- C1–C4 counterexamples (§7) fully characterize the failure surfaces; C2/C3/C4 are addressed by the R1–R3 refinements in §10.2.
- **Precondition:** D135-4 must reconcile `kMaxDeferredRetries` to 2 (F1) before/while implementing.

### 11.4 Final verdict
- **D135-1 code: GO** (correct, no violations).
- **D135-2 Gate E: NO-GO** (E4 unobserved — structural skip, not a bug).
- **D135-3 audit → D135-4: GO** (implement Proposal A per §10.2, resolving F1 first).

---

## 12. Reference index (live-source line map)

| Item | Live source |
|---|---|
| `deferredRecoveryRearmed_` declaration | `RuntimePublicationOrchestrator.h:283` |
| `deferredRecoveryRearmed_` write (only) | `RuntimePublicationOrchestrator.cpp:550` (in `clearDeferredForShutdown`, cpp:534-560) |
| `kMaxDeferredRetries = 10` (drift) | `RuntimePublicationOrchestrator.h:277-279` |
| Retry exhaustion path | `RuntimePublicationOrchestrator.cpp:489-502` |
| `enqueueDeferred` entry | `RuntimePublicationOrchestrator.cpp:437` |
| Retry accounting (generation-keyed) | `RuntimePublicationOrchestrator.cpp:470-478` |
| Slot assignment + `hasDeferred_=true` | `RuntimePublicationOrchestrator.cpp:505-522` |
| `clearDeferredForShutdown` | `RuntimePublicationOrchestrator.cpp:534-560` |
| `peekDeferred` (no hasDeferred_ flip) | `RuntimePublicationOrchestrator.cpp:564-570` |
| `finishView` (sole owner release) | `RuntimePublicationOrchestrator.cpp:589-608` |
| `consume()` | `RuntimePublicationOrchestrator.cpp:612-619` |
| `discard()` | `RuntimePublicationOrchestrator.cpp:621-627` |
| `processDeferredAdmission` | `RuntimePublicationOrchestrator.cpp:636-670` |
| `submitPublishRequest` DeferredFadingActive | `RuntimePublicationOrchestrator.cpp:377-379` |
| `DiscardReason` enum | `RuntimePublicationState.h:10-22` (`RetryExhaustedDiscard` at h:21) |
| `DeferredPublishView` states + fail-fast | `RuntimePublicationOrchestrator.h:52-109` |
| `kDeferredPublishTTLUs = 30s` | `RuntimePublicationOrchestrator.h:125` |
| Rebuild CV predicate | `AudioEngine.RebuildDispatch.cpp:849-854` |
| `processDeferredAdmission` wake call | `AudioEngine.RebuildDispatch.cpp:904` |
| `rebuildRequestGeneration` atomic | `AudioEngine.h:2553` |
| `++rebuildRequestGeneration` | `AudioEngine.RebuildDispatch.cpp:655` |
| `currentBuildGeneration()` | `AudioEngine.h:1666` |
| `EVENT_CROSSFADE_TIMEOUT` recovery handler | `AudioEngine.Timer.cpp:1669-1750` |
| `hasDeferredRequest()` gate | `AudioEngine.Timer.cpp:1721` |
| `resetDeferredRetryBudget()` (inline; zeroes generation+count, does **not** touch `deferredRecoveryRearmed_`) | `RuntimePublicationOrchestrator.h:171-174` (call site: `AudioEngine.Timer.cpp:1724`) |
| `deferredRecoveryRearmed_ = false` (sole write — in `clearDeferredForShutdown`) | `RuntimePublicationOrchestrator.cpp:550` (fn: cpp:534-560) |
| `evaluateDeferred` (Ready/Discard, no RetryLater) | `PublicationAdmission.cpp:106-133` |
| INV-DEFERRED-2 overwrite retire | `RuntimePublicationOrchestrator.cpp:460-468` |
| INV-X1-7 32-slot recovery capacity | ConvoPeq.md:56940-56960 (separate subsystem) |
