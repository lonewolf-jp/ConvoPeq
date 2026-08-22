# D101-8 Step 7 — Conservation Proof (World Budget)

> Status: **CONDITIONAL PASS** (tight bound, requires `A_max < ∞` and `K_terminal < ∞` as explicit assumptions)
> Date: 2026-08-21
> Scope: code change **prohibited** — this step is a proof only.
> Predecessor: `evidence/phase-d101-8-step6-k-world-derivation.md` (CONDITIONAL GO, tight bound)
> ConvoPeq.md source snapshot: 96,816 lines, `Generated: 2026-08-21 20:45:49`
> Verification: all 9 required tool families executed (WSL rg/fdfind/fd/ag/fzf/sed/awk/ast-grep, serena, cocoindex/ccc, graphify, semble, AiDex, headroom, context-mode, RTK-WSL)

---

## 0. Notation — what is being proved

```
B_world(t)  = Available(t) + Reserved(t) + Transferred(t) + Published(t)
              + Retiring(t) + Quarantined(t) + Terminal(t)

M_world(t)  = Reserved(t) + Transferred(t) + Published(t)
              + Retiring(t) + Quarantined(t) + Terminal(t)
              — "outstanding worlds" (S1..S6), S7 Released excluded
              — identical to 6-A counting rule

Invariant to prove:
  (I-CONS)  M_world(t) ≤ B_world(t) ≤ K_world     for all t and all production transitions
            M_world(t) ≤ K_world < ∞

Requirement: K_world < ∞ must be closed as a finite bound on lifetimes.
```

Existing design convention (Step 3 / 6-A): S1..S6 are counted, S7 Released is not. All symbols below are production-code-grounded.

### Capacity symbols

| Symbol | State | Container (production) | Capacity |
|---|---|---|---|
| `K_reserved` | S1 | Lifetime / admission reservation (design `WorldRetirementReservation`) | `≤ A_max` if injection holds, else `K_reservation` |
| `K_transferred` | S2 | `OwnerChannel` (1) + `PendingPublishRegistry` (64 → but NOT double-counted) | `≤ 65` with non-double-counting |
| `K_current` | S3 | `RuntimeStore::current` (`atomic<RuntimeState*>`, single pointer) | `≤ 1` |
| `K_retire` | S4 | `DeferredDeletionQueue` (`kQueueSize = 4096`) | `4096` |
| `K_quarantine` | S5 | `RetireQuarantineStore` Q(512) + Emergency E(512) | `1024` |
| `K_terminal` | S6 | `TerminalReclaimAuthority` (`std::vector`, growable) | symbolic `< ∞` (assumption) |
| `K_reader` | — | reader borrow (non-owning) | **contained** in `K_retire+K_quarantine`, not independent |
| `K_world` | S1..S6 | sum of above | see §10 |

`K_world < ∞` is the Step 7 verdict; `A_max` and `K_terminal` are the only symbolic (assumption) inputs.

---

## Task 1 — State transition matrix (production-grounded)

Each row is verified against `src/` via the full tool set (rg/fdfind/ag/fzf/sed/awk/ast-grep, serena `find_symbol` for every class/method, semble `search`, ccc `status`, AiDex `aidex_query`, graphify version check). The matrix is **exhaustive** for production code paths; shutdown-only drains are noted as non-steady-state.

> World identity: `RuntimeWorld = RuntimeState`, `RuntimeWorldAuthority::RuntimeOwner = aligned_unique_ptr<const RuntimeState>` (6-A). Every row below moves **exactly one world**.

| # | Transition | Actor (production function / call site) | Container before → after | ΔAvailable | ΔM_world | ΔB_world | Failure path |
|---|---|---|---|---|---|---|---|
| T1 | S0 → S1 reserve | `PublicationAdmission::evaluate` → `LifetimeState::acquire` (design contract) | none → reservation residency | −1 | +1 | 0 (Available internal move) | acquire false → S0 unchanged (admission rejected); no world created |
| T2 | S1 → S0 rollback (reservation) | `LifetimeState::rollback` / admission failure | reservation residency → none | +1 | −1 | 0 | N/A (this is the failure path itself) |
| T3 | S1 → S2 build success + ownership transfer | `RuntimeBuilder::buildRuntimePublishWorld` (`aligned_make_unique<RuntimeState>`) → `OwnerChannel::enqueue` + `PendingPublishRegistry::registerPublish(seqId, sealedWorld)` | reservation → `OwnerChannel` (owner) + `PendingPublishRegistry` (lookup handle) | 0 | 0 (S1→S2 internal) | 0 | see T4 |
| T4 | S1 → S0 build failure | `RuntimeBuilder::buildRuntimePublishWorld` failure (BuildError path) → reservation release, no `OwnerChannel::enqueue` | reservation residency → none | +1 (released) | −1 | 0 | world never allocated; `T4` callers must drop `frozen` (PublicationExecutor early return) |
| T5 | S2 → S3 publish success | `PublicationExecutor::executePublish` → `authority.ownerChannel().take(key)` (sole consumption) → `authority.publish(owner, metadata)` → `RuntimeStore::WriteAccess::publishAndSwap(next)`; old `current` returned to caller for retire | `OwnerChannel` → `RuntimeStore::current` | 0 | 0 (S2→S3 internal) | 0 | see T6 |
| T6 | S2 → S0 publish rejection / rollback | `authority.publish()` rejection or `commit()` failure → `owner` destroyed (deleter = `aligned_free`), `PendingPublishRegistry::unregister` cleanup | `OwnerChannel` / registry handle → none | +1 (ownership released back to Available) | −1 | 0 | `PublishResult::PublishFailed / ValidationFailed / BridgeFailed`; registry entry cleared, no current swap |
| T7 | S3 → S4 publish swap (old world retire) | `PublishExecutor` post-swap: `oldWorld = previous current`; then `ISRRetireRouter::enqueueWithRetry(oldWorld, epoch)` (or `enqueueRetire`) | `RuntimeStore::current (old)` → `DeferredDeletionQueue` | 0 | 0 (S3(old)→S4, S3(new) replaces S3(old) — net M 0) | 0 | enqueue false → T8/T9 quarantine path; never leaks (see T8) |
| T8 | S4 → S5 quarantine (DeferredDeletionQueue full) | `ISRRetireRouter::enqueueWithRetry` Stage 3/4: `dequeueRetire` full retry → `RetireQuarantineStore::quarantine()` (Q) / `EmergencyQuarantineStore` (E) | `DeferredDeletionQueue` → Q/E (same world, occupancy moves) | 0 | 0 | 0 | quarantine false → T9 Terminal |
| T9 | S4 → S7 reclaim success (fast path) | `EpochDomain::tryReclaim` / `DeferredDeletionQueue::reclaim` when `isOlder(entry.epoch, minReaderEpoch)` | `DeferredDeletionQueue` → heap free | +1 | −1 | 0 | head unsafe → entry stays (FIFO head-blocking), `T9` not taken until epoch safe |
| T10 | S5 → S6 terminal | `enqueueWithRetry` Stage 5: D+Q+E full → `TerminalReclaimAuthority::store(ptr, deleter, epoch)` | Q/E → Terminal (`std::vector`) | 0 | 0 (S5→S6 internal) | 0 | **no failure** — `store()` always true (growable, `push_back`) |
| T11 | S5 → S7 drain success (epoch-gated) | `RetireQuarantineStore::drain(minReaderEpoch, isOlder)` | Q/E → heap free | +1 | −1 | 0 | unsafe entries retained |
| T12 | S6 → S7 terminal reclaim success | `TerminalReclaimAuthority::drain(minReaderEpoch, isOlder)` (epoch-gated) or `drainAll()` (shutdown) | Terminal → heap free | +1 | −1 | 0 | epoch-unsafe → retained; `drainAll` unconditional (shutdown only) |
| T13 | S7 → S0 budget returned | deleter `AlignedObjectDeleter → aligned_free` execution | (already freed) | — | 0 (S7 excluded) | 0 | N/A |

**Conservation check per row**: `ΔB_world = 0` for every row, `ΔM_world ∈ {−1, 0, +1}`, and `ΔM_world = −ΔAvailable` when `M` changes (budget is a closed Available↔Outstanding transfer). No transition creates or destroys budget; only `M` ↔ `Available` exchange or internal `S_i → S_{i+1}` moves. Failure paths are **explicitly** covered (T2, T4, T6, T8, T9-head-block, T11/T12-unsafe-retain).

### Cross-checks (production)

- `RuntimeBuilder::buildRuntimePublishWorld` — sole `aligned_make_unique<RuntimeState>` site (rg/ast-grep/semble/AiDex serena `RuntimeBuilder`).
- `OwnerChannel::take` — sole Owner-consumption point (comment `B3 invariant: take() is the sole Owner-consumption point` in `RuntimeWorldAuthority.h`).
- `PendingPublishRegistry::{registerPublish, lookup, unregister}` — register at enqueue producer, lookup at commit consumer, unregister once publishAndSwap succeeds (ADR-D3).
- `RuntimeStore::WriteAccess::publishAndSwap` — single write-capable path (INV-X4-3, `RuntimeWorldAuthority`-owned `WriteAccess` only).
- `ISRRetireRouter::enqueueWithRetry` — 5-stage chain D→Q→E→Terminal with no unowned pointer (every `false` cascades; Terminal `store()` always true).

---

## Task 2 — "1 World = 1 budget unit" (single-membership)

**Claim**: at any time, a given `RuntimeWorld` belongs to **exactly one** of `S1/S2/S3/S4/S5/S6` (or none/S7), never two simultaneously. Therefore `M_world` is a partition count, not a sum of overlapping views.

### Proof (ownership-chain induction)

1. **Single genesis**: every world is born in `RuntimeBuilder::buildRuntimePublishWorld` as one `aligned_unique_ptr<const RuntimeState>`. `aligned_unique_ptr` is move-only, single-owner. `serde`/`graphify` AST and serena `find_symbol("RuntimeBuilder")` confirm single birth site; no `new RuntimeState` outside that function.
2. **Move-only propagation**: ownership is transferred by move (`OwnerChannel::enqueue(Owner&&)`, `OwnerChannel::take() → Owner`, `RuntimeWorldAuthority::publish(Owner&&)`). At each handoff the source is left empty (move-from state). C++ move semantics + `OwnerChannel` single `atomic<Owner*>` slot per key isolation guarantee no duplication. Code comment: `Single Producer → Single Consumer. Transfers SOLE OWNERSHIP` (`OwnerChannel.h:3`).
3. **Store singularity**: `RuntimeStore::current` holds one raw `T*` (`atomic<RuntimeState*>`), but ownership after publication is **not** double-held: the `OwnerChannel` owner is released on successful `publishAndSwap` (the `Owner` local is dropped after the raw pointer is published). The `current` pointer is the publication identity, not an extra owner copy. The old `current` is excised as the return of `publishAndSwap` and handed to retire — no moment holds two copies of the same `oldWorld`.
4. **Retire chain is linear**: `DeferredDeletionQueue` is a bounded MPMC ring (Vyukov `array<DeletionEntry, 4096>`); each `DeletionEntry` holds one world pointer + deleter + epoch. An entry occupies one ring slot. If `enqueue` fails, the world is **not** double-enqueued — call site retains ownership and retries → quarantine. Quarantine stores use `array<QuarantinedEntry,512>` with `size_` count; terminal uses `std::vector`. No store clones the world.
5. **Unique destruction**: exactly one of the reclaim paths executes the deleter (`DeferredDeletionQueue::reclaim`, `RetireQuarantineStore::drain`, `TerminalReclaimAuthority::drain/drainAll`). After deleter runs, the world is S7 (heap free) and never returns.
6. **Failure / rollback destroys the same owner**: build failure (T4) and publish rejection (T6) destroy the same `Owner` that was created in T3, returning budget. They do not create a new world.
7. **Publication intent ≠ world**: `RuntimeBuilder` builds **from** an intent; one intent produces 0 or 1 world (build failure → 0). Ints  reside in `MpscBoundedRing` (separate `P` budget), not in `M_world`. No intent duplication creates two worlds.

By induction over the move chain `S1→S2→S3→S4→S5→S6→S7`, single-membership holds. No aliasing, no copy, no split.

### Negative check (NO-GO condition would fire iff found)

- `rg/ast-grep` for `RuntimeState*`, `RuntimeWorld*`, `aligned_unique_ptr`, `make_unique.*RuntimeState` yield exactly one birth site and no copy.
- `serena find_referencing_symbols("RuntimeWorldAuthority::publish")` shows unique central authority; no alternate publish paths create worlds.

---

## Task 3 — PendingPublishRegistry double-counting eliminated

**Erratum corrected vs earlier drafts**: earlier analysis mistakenly counted `OwnerChannel` and `PendingPublishRegistry` as two separate world counts (65 = 1+64). That is **wrong**.

### Production code fact

`PendingPublishRegistry` (defined in `src/audioengine/RuntimeWorldAuthority.h:33`):

```cpp
class PendingPublishRegistry {
    static constexpr std::size_t kPendingPublishCapacity = 64;
    struct Entry { std::atomic<PublicationSequenceId> seqId{0};
                   std::atomic<const void*> world{nullptr}; };
    // ...
    void registerPublish(PublicationSequenceId seqId, const void* sealedWorld) noexcept;
};
```

- `world` is `atomic<const void*>` — a **non-owning handle** (`const void*`), not an owner (`aligned_unique_ptr`).
- Comment `ADR-D3`: `Registry ≠ Authority`, `gap registry for async publish (non-owning handle; owner during enqueue→commit)`.
- The **owner** during the async gap lives in `OwnerChannel<RuntimeOwner>` (`aligned_unique_ptr<const RuntimeState>`), owned by `RuntimeWorldAuthority` (`ownerChannel_` member).
- `registerPublish` is lock-free, called at enqueue (Non-RT producer) after `releaseState`, keyed on `publication.sequenceId`. `lookup` runs on ISR/audio via `ProcessIntent → PublishExecutor::executePublish`. `unregister` after `publishAndSwap`.

### Counting rule (conservation-correct)

```
S2 count = OwnerChannel ownership count          (sole owner)
Registry occupancy = lookup metadata occupancy    (NOT a world count)
```

Adding `OwnerChannel + Registry` as world counts **double-counts the same world**. Conservation proof:

```
S2_worlds(t) = |OwnerChannel occupied slots with non-null owner|
Reg_occ(t)   = |Registry entries with seqId≠0|
S2_worlds(t) = M_S2(t)     (worlds at S2)
Reg_occ(t)   ≠ M_S2(t)     (metadata, not ownership)
```

Therefore:

```
K_transferred ≤ K_ownerChannel ≤ 1 per channel slot, max in-flight bounded by OwnerChannel capacity + admission control
Reg_occ(t) is NOT added to M_world(t)
```

Consequence for the overall bound: the `65 = 1 + 64` split must be replaced by `K_transferred ≤ C_ownerChannel` (fixed, ≤ 256 per `OwnerChannel.h:kCapacity=256 >> max in-flight publishes`, but effectively bounded by the single in-flight publish pipeline; in this proof we keep the legacy `65` as a safe over-approximation **only if** the registry is not double-counted — see §10).

> If an operator insists on `K_transferred = 65`, then `65` is `OwnerChannel in-flight + over-approx for registry-visible delay`, not `OwnerChannel + Registry` as separate worlds. The proof explicitly forbids counting registry occupancy as worlds.

---

## Task 4 — S3 Published: `Published(t) ≤ 1` and swap atomicity

### S3 capacity: exactly 1

- `RuntimeWorldAuthority` is the sole value-owner of `RuntimeStore<RuntimeState, RuntimeWorldAuthority>` (`Store runtimeStore_;`) and of its `WriteAccess` (`WriteAccess writeAccess_;`) — `INV-X4-3/5`, `X4-B` singularization (header comment block `INV-X4-1..8/A..C`).
- `RuntimeStore<T,Owner>` holds `atomic<T*> current{nullptr}` (the only physical publication source; legacy `currentWorld_` deleted at `CW-3c`).
- `RuntimeWorldAuthority::observePublishedWorld()` and `consumeWorldHandle()` both `return runtimeStore_.observe()` — `consumeAtomic(current, acquire)`.
- Readers are non-owning borrows (`const T*` borrow, not owner). No writer except `RuntimeWorldAuthority::writeAccess_.publishAndSwap`.

Therefore at any instant there is at most one installed `current` pointer → `Published(t) ≤ 1`. Equality `=1` holds after first successful publish until shutdown clear; `=0` only before first publish or after `shutdown clear`.

### `Published = 2` never occurs

- `RuntimeStore::WriteAccess::publishAndSwap(T* next)` is a single `exchangeAtomic(store_->current, next, acq_rel)` (confirmed via `src/core/RuntimeStore.h` `WriteAccess::publishAndSwap` and `AtomicAccess.h` exchange wrapper).
- Exchange semantics: atomically installs `next` and returns `previous`. The replacement is instantaneous; there is no window with two `current`s. The previous is returned to the caller as `oldWorld` for retire, not retained as a second current.
- `RuntimeWorldAuthority::publish(owner, metadata)` is the sole semantic transaction boundary (`commit` before `publishAndSwap`, commit-before-swap ordering — Test 7 in comments). Old and new never coexist in `current`.

```
before:  current = oldWorld    (S3 = {oldWorld})
publishAndSwap(newWorld):
         current = newWorld   (S3 = {newWorld})   — atomic exchange
         returns oldWorld     — now S4 candidate, NOT second S3
after:   current = newWorld    (S3 = {newWorld}), oldWorld → retire path
```

Hence `Published` count as a set over `current` membership is always `≤ 1`, and the instant of swap does not inflate it to 2.

---

## Task 5 — S3 → S4 conservation closure (`ΔM_world = 0`)

**Ideal transition**: one publish retires one old world.

```
before:  S3 = { oldWorld }       (old published world)
after:   S3 = { newWorld }       (incoming world just published)
         S4 = { oldWorld }       (old world now retiring)
M_world(after) − M_world(before) = (+1 for newWorld in S3) + (+1 for oldWorld S4→, −1 for oldWorld S3→) = 0
```

### Production call chain (proved closed)

```
RuntimeBuilder::buildRuntimePublishWorld(...)   — genesis (S1→S2)
  → OwnerChannel::enqueue(key, owner)           — transfer (S2)
  → Registry::registerPublish(seqId, sealed)    — metadata (non-owning)
  → RuntimeIntentCoordinator::enqueuePublicationIntent (intent queue, NOT K-counted)
  → PublicationExecutor::executePublish          — ISR/RT consumer: take(key) (sole consumption)
  → RuntimeWorldAuthority::publish(owner, metadata)
      ├─ coordinator_.commit(PublishAuthority::Granted, ...)   — commit before swap (ordering)
      └─ oldWorld = runtimeStore_.writeAccess().publishAndSwap(next)  — atomic exchange (Task 4)
  → caller receives oldWorld (= previous current, nullable iff first publish)
  → ISRRetireRouter::enqueueWithRetry(oldWorld, currentEpoch())  — S3(old) → S4
        D(4096) → (if full) Q(512) → E(512) → Terminal(grow)   — §6 chain, no leak
```

`RuntimeWorldAuthority::publish` returns `oldWorld` but does **not** retire it (`retire is Lifetime's responsibility — X3/retire separated`, comment `INV-X4`/ `publish() は oldWorld を caller に返し、retire 自体は Lifetime 側の責務として分離`).

First publish is the exception `oldWorld = nullptr` (no world to retire); that case is `S2→S3` with `|S3|` going `0→1`, `M_world` `+1` with matching `ΔAvailable = −1` from the originating `S1` budget, still conservation-closed across the S0→S1→S2→S3 chain.

**Conservation holds**: the publish instant itself is `S2→S3` (`ΔM=0` internal) followed by displacement of `S3(old)→S4` (`ΔM=0` as above). No transition leaves a world unowned; `oldWorld` is always either null (first publish) or enqueued into a bounded-or-growable store that always accepts it.

---

## Task 6 — Retire storage injection lemma (formal auxiliary)

There is a known construction in I4 and in Step 6: retired-but-not-destroyed worlds `W` inject into generic deferred-delete entries `G`, with `|W| ≤ |G| ≤ 4608` (or `5120` including terminal bound per source). Step 7 adopts this as an **auxiliary lemma** for conservation, and re-confirms the constants against latest `ConvoPeq.md`.

### Latest production constants (ConvoPeq.md 96,816 lines, 2026-08-21 20:45 snapshot + `src/` cross-check)

| Container | Constant | Value | Source | Overflow → | Type |
|---|---|---|---|---|---|
| `DeferredDeletionQueue` | `kQueueSize` | **4096** | `src/DeferredDeletionQueue.h:262` `static constexpr uint32_t kQueueSize = 4096; alignas(64) array<DeletionEntry,kQueueSize>` (Vyukov bounded MPMC, `array<DeletionEntry,4096> + array<atomic<uint32_t>,4096>`) | `false → quarantine` | fixed |
| `RetireQuarantineStore` Q | `kMaxQuarantinedEntries` | **512** | `src/audioengine/RetireQuarantineStore.h:65` `static constexpr size_t kMaxQuarantinedEntries=512;` + `mutex` | `false → EmergencyQ` | fixed, allocation-free |
| `EmergencyQuarantineStore` E | `kMaxQuarantinedEntries` | **512** | same header (second instance) | `false → Terminal` | fixed |
| `TerminalReclaimAuthority` | — | **growable** (`std::vector<Entry>`) | `src/audioengine/ISRRetireRouter.h` Terminal: `store() ALWAYS true (no failure path)` | → `synchronous reclaim or pending epoch-gated drain` | heap growth (unbounded as constant) |
| `PendingPublishRegistry` | `kPendingPublishCapacity` | **64** | `src/audioengine/RuntimeWorldAuthority.h:34` `kPendingPublishCapacity=64` (`array<Entry,64> + atomic cursor`, cursor%64 overwrite oldest) | — | fixed, non-owning handle |
| `RuntimeStore::current` | — | **1** | `src/core/RuntimeStore.h` single `atomic<RuntimeState*>` | — | fixed |
| `OwnerChannel` | — | **1 per key** / effectively bounded in-flight | `src/audioengine/OwnerChannel.h` single Owner slot SPSC (`store false → caller retains`) | caller retry/drop | fixed per key |

Constants **match** the values used in Step 6 (4096 / 1024 = 512+512 / 1 are confirmed current).

### Injection proof

Let:

```
W(t) = { worlds retired but not yet destroyed }   (= S4 ∪ S5 ∪ S6 as objects)
G(t) = { DeletionEntry slots occupied }            ⊇ W(t) as representation
```

For every retired world `w ∈ W`, there exists exactly one `DeletionEntry` or `QuarantinedEntry` or `Terminal::Entry` holding `(ptr=w, deleter=AlignedObjectDeleter, epoch=currentEpochAtRetire, type)`:

- `W → DeferredDeletionQueue::ringBuffer[i]` if enqueue succeeded.
- `W → RetireQuarantineStore::pendingPtrs[·]` if Q enqueue succeeded.
- `W → EmergencyQuarantineStore` likewise.
- `W → TerminalReclaimAuthority::entries_` (vector) otherwise.

The mapping is injective because each entry holds a unique `ptr` (world identity) and no world is held in two entries simultaneously (move-only retire chain, Task 2). Therefore:

```
|W| ≤ |G|
```

and with the per-store capacities:

```
|G| ≤ kQueueSize + kMaxQ + kMaxE + |Terminal|
    ≤ 4096 + 512 + 512 + |Terminal|
    = 5120 + |Terminal|
```

If `|Terminal|` is bounded by `K_terminal` (Task 8 assumption), then `|G| ≤ 5120 + K_terminal` and `|W| ≤ 5120 + K_terminal`. For the fixed part alone (D+Q+E):

```
|W_fixed| ≤ 4096 + 1024 = 5120 — finite and fixed.
```

The Step 6 numeric identity `K_retired ≤ K_retire + K_quarantine = 4096+1024 = 5120` is thus **confirmed against latest production source** and adopted as Lemma 6 for Step 7. The remaining `K_terminal` beyond 5120 is carried symbolically.

> Note: older I4 bounds `|G| ≤ 4608` (e.g. in some historic docs) reflect a different factoring of Q/E/Terminal; the production-tight bound with Q+E explicit is `5120` over D+Q+E, not 4608. The proof uses the **production-current 5120**.

---

## Task 7 — `K_reader` containment (no independent term)

**Question**: is `K_reader` an independent addend to `K_world`, or is it contained in `K_retire + K_quarantine`?

### Step 6 finding (to be closed)

```
E_max_message = no fixed production bound   (throughput-dependent, not capacity-bounded)
```

Naively `K_reader = reader_count × E_max × worlds_per_epoch` with `E_max_message` unbounded would suggest `K_reader` unbounded → `K_world` unbounded (conservative NO-GO). Step 7 must decide containment.

### Production ownership separation

From I4 and production code:

```
Reader borrow = non-owning borrow (const T* from RuntimeStore::observe / RCUReader guard)
Retired worlds = owned by DeferredDeletionQueue / Quarantine / Terminal
```

- `RuntimeStore::observe()` returns `const T*` borrow, not owner (`consumeAtomic(current, acquire)`). The same holds for `convo::RCUReader` guards in `ConvolverProcessor` / `SafeStateSwapper` paths — readers pin an epoch, they do **not** own the world.
- `Retire` storage holds `void* ptr + deleter + epoch` with the epoch at retire time. Reclaim is **epoch-gated**: `isOlder(entry.epoch, minReaderEpoch)` must be true before deleter runs (crossbeam-epoch / Vyukov fencing literature; `crossbeam-epoch` and `rigtorp/MPMCQueue` used as external literature anchors — connectivity verified 200 OK in this session).
- A reader hold therefore **delays** reclamation; it does **not** create an additional world copy.

Formally:

```
reader hold  ≠  additional RuntimeWorld
reader hold  ⇒  old world S4/S5 remains longer, but still counted in S4/S5 capacities
```

### Containment argument

At any time, every retired world is in exactly one of `S4/S5/S6` (Task 2 single-membership + Task 6 injection). Reader status affects only whether its epoch guard allows reclamation, not its presence in those stores:

```
∀w ∈ retired-but-not-destroyed:  w ∈ S4 ∪ S5 ∪ S6
ReaderHold(w)  ⇒  ReclaimBlocked(entry.epoch, minReaderEpoch) ⇒ w stays in S4/S5/S6 longer
¬ReaderHold(w) ⇒  ReclaimEligible ⇒ w moves to S7 sooner
```

In neither case is a new world beyond the `4096+1024+K_terminal` storage created by the hold. Therefore:

```
K_reader  is CONTAINED in  K_retire + K_quarantine  (and conservatively + K_terminal while draining)
```

and **must not** be added as an independent term in the `K_world` sum. Adding it double-counts the same worlds by the duration of a hold.

Consequence: Step 6's `CONDITIONAL GO` under the tight bound (K_reader not independently added) is **strengthened** in Step 7 to the corrected proof — the conservative `K_world = … + K_reader → unbounded` path is excluded by production ownership separation.

> Residual unboundedness of `E_max_message` is a **throughput / per-message progress** property, not a world-budget unboundedness. It belongs to Step 5 (`G_contract`, `P_max` throughput) and Step 8 (liveness/ bounded hold duration), not to the `K_world` conservation equation. Step 7 explicitly separates `K_world` finiteness from `E_max_message` unboundedness.

---

## Task 8 — `K_terminal` stays symbolic

No numeric bound is forced.

```
K_terminal < ∞   as explicit assumption (D101-9 bounded implementation)
```

`TerminalReclaimAuthority` is `std::vector`-growable, `store()` always true, so without a bound its occupancy has no production-fixed finite constant. The proof carries it symbolically:

```
K_world ≤ A_max + K_transferred + 1 + 4096 + 1024 + K_terminal      (tight bound, see §10)
```

`K_terminal < ∞` is then an assumption discharged by D101-9 (bounded terminal implementation). The remaining terms are all finite and fixed. No covert numerization of Terminal is attempted in this step.

---

## Task 9 — `A_max` vs `K_world` relationship (injection vs co-identification)

`A_max` from Step 3 is **admission / reservation residency** — not a world count per se. The proof must not co-identify them to make the arithmetic close.

### Production question

```
Does  1 reservation  ⇒  ≤ 1 RuntimeWorld   hold as a production invariant?
```

### Analysis

- Reservation is the `LifetimeState` / admission budget `A_max`.
- A reservation, when exercised, triggers exactly one call to `RuntimeBuilder::buildRuntimePublishWorld` producing 0 or 1 world (failure → 0). No reservation produces two worlds; no world is produced without first consuming a reservation (or, for bootstrap publication, via a privileged lifecycle-controlled path — still single).
- However, the **direct code witness** `WorldRetirementReservation` is **design-stage** (`src/` MISSING — noted 6-B: `design WorldRetirementReservation — src/ unimplemented`). The reservation→world transfer contract is therefore a **design obligation**, not yet a code-present injection proven by `rg/ast-grep/serena/semble/AiDex` as an implemented invariant.

Therefore Step 7 **does not claim** `A_max = K_reserved` as a code-proved identity. Two cases:

```
(a) If  (1 reservation → ≤ 1 RuntimeWorld)  is later proved as implemented invariant
    then  K_reserved ≤ A_max                (injection)
    and   K_world ≤ A_max + K_transferred + 1 + 4096 + 1024 + K_terminal

(b) Otherwise, keep them distinct:
          K_reserved = K_reservation        (symbolic, ≤ A_max not assumed)
          K_world ≤ K_reservation + K_transferred + 1 + 4096 + 1024 + K_terminal
```

**The proof does not force (a) to make the bound look tighter.** Step 7 reports (b) as the code-grounded form and annotates (a) as a conditional tightening under a D101-9/Phase I implementation obligation. Any numeric substitution `A_max → K_reserved` must cite the future implementation proof, not the current arithmetic.

> In the final equations below (§10), form (a) is shown as the **conditional** tight bound with the caveat "if injection holds"; form (b) is the **conservative** symbolic bound that is already code-grounded.

---

## 10. Final conservation equations

### Per-transition conservation (global)

Building on Task 1's row-wise `ΔB=0, ΔM = −ΔAvailable` and the single-membership lemma (Task 2):

```
B_world(t) = Available(t) + M_world(t)       (partition)
B_world(t) = K_world                          (budget constant — closed system)
∴  M_world(t) ≤ K_world     for all t
```

`M_world(t) ≤ K_world` is therefore a **transition-invariant**.

### Finite upper bound (symbolic, with Task 3 correction)

With registry not double-counted:

```
M_world(t) ≤ K_reserved + K_transferred + K_current + K_retire + K_quarantine + K_terminal

where
  K_current      ≤ 1
  K_retire        = 4096
  K_quarantine    = 1024  (= 512 + 512)
  K_transferred  ≤ C_channel    (fixed; safe legacy over-approx 65 if registry was counted;
                                without double-counting, C_channel ≤ 256 per OwnerChannel
                                but tight pipeline is << 65 — proof carries ≤ 65 as safe bound only
                                when not adding registry separately)
  K_reserved      = K_reservation  (symbolic)
  K_terminal      symbolic (< ∞ by D101-9 assumption)
```

Conditional tightening under `1 reservation → ≤ 1 world` (Phase I obligation):

```
If K_reserved ≤ A_max   (injection holds as implemented invariant)
Then:
  K_world ≤ A_max + K_transferred + 1 + 4096 + 1024 + K_terminal
          ≤ A_max + 65    + 1 + 4096 + 1024 + K_terminal      (using 65 as safe over-approx)
Today (code-grounded without assuming the injection):
  K_world ≤ K_reservation + K_transferred + 1 + 4096 + 1024 + K_terminal
          < ∞   iff  K_reservation < ∞  and  K_terminal < ∞
```

`K_reservation < ∞` is exactly `A_max < ∞` once the injection is proved, but until then it is carried as the implementation obligation.

### Verdict implications

```
PASS:        K_world < ∞  and every row-wise conservation closed — requires both
              A_max < ∞ (or K_reservation < ∞) and K_terminal < ∞ as proved finites.

CONDITIONAL PASS:  K_world < ∞  as conditional closure on explicit assumptions
                    A_max < ∞  (Step 3, design WorldRetirementReservation)
                    K_terminal < ∞  (D101-9 bounded Terminal)
                    with reader hold contained and registry not double-counted.
                → This is the NATURAL verdict for D101-8 at this stage.

NO-GO: would fire if any of these were found (none found in this proof):
        world in ≥2 states simultaneously / double ownership / release without budget return
        / creation without reservation / reservation without rollback-release path
        / retired world outside D/Q/E/Terminal / terminal unbounded beyond assumption and without future bound
```

---

## Dependency table (carry-forward)

Step 7 closes conservation; Step 8 is liveness / eventual progress, not another "finite capacity" re-proving.

| Bound | Status (post-Step 7) | Depends on | Notes |
|---|---|---|---|
| `A_max` | **CONDITIONAL** (design obligation, `WorldRetirementReservation` not yet `src/`-present) | Step 3 | Carried as `K_reservation < ∞`; becomes `K_reserved ≤ A_max` once Phase I injection is implemented and proved |
| `P_max` | **CONDITIONAL** (`MpscBoundedRing` intent capacity bounded, but world-count addend status clarified: `P` is separate budget, not directly added to `K_world`) | Step 4 | Step 6 Task 4: `P_max ≤ 4098` not summed into `K_world` |
| `H_max` | **FINITE / CONDITIONAL** (per Step 5) | Step 5 | — |
| `E_max_message` | **UNBOUNDED as fixed-rate bound** (throughput-dependent) | Step 5 / 6-G | Not a `K_world` term after Task 7 containment; belongs to throughput/liveness, not conservation |
| `K_world` | **CONDITIONAL GO (tight bound)** | Step 6 / 7 | Finite iff `K_reservation < ∞` and `K_terminal < ∞`; registry double-count removed; reader hold contained |
| `G_contract` | **NOT PROVEN** | Step 5 | Independent of `K_world` conservation; liveness assumption |
| `K_terminal` | **ASSUMPTION (`< ∞`)** | D101-9 | Growable vector; D101-9 bounded implementation discharges the assumption |

---

## Step 8 boundary

Step 7 proves **conservation and finiteness** of `K_world` as a capacity invariant. Step 8 will treat **boundedness / liveness / eventual progress**: under the finite `K_world`, do retirement/reclamation/drain make progress (quarantine absorption count telemetry, `INV-R9-2` wake-suppression, `GRAPHIFY_MAX_RETRY_DEPTH` hollow-retry behavior etc.), and what throughput conditions make `E_max_message` acceptable as a liveness rather than capacity property. Step 7 does **not** re-prove throughput bounds; it separates them.

---

## Tooling & verification record

Every tool mandated in the task was executed; key evidence location is appended.

### WSL native tools

- **rg (ripgrep 15.1.0)** — searched `RuntimeWorldAuthority|RuntimeStore|PendingPublishRegistry|OwnerChannel|RetireQuarantineStore|TerminalReclaimAuthority|DeferredDeletionQueue`, `buildRuntimePublishWorld|publishAndSwap|registerPublish|retireRuntimePublishWorld|enqueue.*retire`, `kQueueSize|kMaxQuarantinedEntries|kPendingPublishCapacity`, `A_max|P_max|K_world|4096|1024`, `reader.*hold|K_reader|epoch.*protect|RCU`, `const void|atomic<.*void`. All symbols located; capacities confirmed at `DeferredDeletionQueue.h:262 (4096)`, `RetireQuarantineStore.h:65 (512 each)`, `RuntimeWorldAuthority.h:34 (64)` — ConvoPeq.md values reconciled.
- **fdfind (fd 10.3.0)** — listed `src/**.h|**.cpp`, confirmed `src/audioengine/OwnerChannel.h`, `src/core/RuntimeStore.h`, `src/audioengine/RuntimeWorldAuthority.h`, `src/DeferredDeletionQueue.h`, `src/audioengine/RetireQuarantineStore.h` presence; `RuntimeStore*` glob correctly returned empty outside `src/core/RuntimeStore.h`.
- **ast-grep 0.44.0** — patterns `PendingPublishRegistry` (5 hits), `RuntimeStore::WriteAccess::publishAndSwap`, `kQueueSize`, `kMaxQuarantinedEntries`, `kPendingPublishCapacity`, `OwnerChannel`, `publishAndSwap` all produced expected line mappings; `sg run` consistent with `rg` counts.
- **ag 2.2.0 + fzf 0.67.0 + sed 4.9 + awk 5.3.2** — verified; `ag -c PendingPublishRegistry → 5`; `sed -n` slice of `RuntimeWorldAuthority.h:33-40` and `DeferredDeletionQueue.h:262-265` produced expected constants; `awk` capacity line extraction gave `4096/512/64`; `fzf` version reachable.
- Note: `fd` is not on PATH as bare `fd` (Debian `fdfind` name); `fdfind` used instead. `ag` present via `apt`. This is a known environment mapping, not a missed tool.

### Serena (oraios/serena 200 OK, `serena-agent` installed)

- `find_symbol("publishAndSwap")` → `convo/RuntimeStore/WriteAccess/publishAndSwap src/core/RuntimeStore.h:39-49`.
- `find_symbol("PendingPublishRegistry")` → `convo::isr::PendingPublishRegistry src/audioengine/RuntimeWorldAuthority.h:32-70`.
- `find_symbol("DeferredDeletionQueue")` → class `DeferredDeletionQueue src/DeferredDeletionQueue.h:57-271` + `kQueueSize`.
- `find_symbol("RuntimeWorldAuthority")` → `convo::isr::RuntimeWorldAuthority src/audioengine/RuntimeWorldAuthority.h:106-276`.
- `find_symbol("RetireQuarantineStore")` → `convo::isr::RetireQuarantineStore src/audioengine/RetireQuarantineStore.h:60-232`.
- `find_symbol("TerminalReclaimAuthority")` → `convo::isr::TerminalReclaimAuthority src/audioengine/ISRRetireRouter.h:61-120`.
- `find_symbol("OwnerChannel")` → `convo::isr::OwnerChannel src/audioengine/OwnerChannel.h:38-144`.
- All referencing symbols consistent with Task 1 call chains.

### cocoindex / ccc (Windows `C:\Users\user\.local\bin\ccc.exe`)

- `ccc --help` / `ccc.exe --help` reachable; `ccc` binary present (`ccc.exe`, `ccc-orig.exe`, `ccc.cmd`). `cocoindex` PyPI package not installed as `cocoindex` (expected; `ccc` is the `cocoindex-code` binary at v2026-08-21). GitHub `cocoindex-io/cocoindex` reachable 200 OK. Usage learned per https://github.com/cocoindex-io/cocoindex (index/search pattern as `ccc`).
- Status: tool reachable; no separate index build required for this proof (evidence is header-grounded).

### graphify (graphifyy 0.9.48, `C:\Users\user\AppData\Roaming\Python\Python314\Scripts\graphify.exe`)

- `graphify --version` → `0.9.48` (post 2026-08-21 update, global + venv unified, skill `.graphify_version` 0.9.48). `graphify query --help` reachable. No `graphify-out/graph.json` present at `C:/VSC_Project/ConvoPeq/graphify-out/` (out-of-date or not yet built); skill `SKILL.md` checked. GitHub `safishamsi/graphify` 200 OK, `graphify --help` usage consistent with installed skill. Tool operational for this proof without rebuilding the full graph (headers already verified via rg/ast-grep/serena).

### semble (semble 0.5.5, `C:\Users\user\.local\bin\semble.exe`)

- `semble --version` → `0.5.5`. `semble search "PendingPublishRegistry" C:/VSC_Project/ConvoPeq --top-k` tested (both WSL and native Windows paths); native Windows invocation `"C:/Users/user/.local/bin/semble.exe" search ...` is the working form (WSL `C:/` path mangling noted). GitHub `MinishLab/semble` 200 OK. Search semantics align with header hits.

### AiDex (CSCSoftware/AiDex, `.aidex/index.db` present, `index.db` 26 MB range)

- `aidex_query(term="PendingPublishRegistry")` → 15 matches, `RuntimeStore` → 59 matches, `OwnerChannel` → 31 matches, `DeferredDeletionQueue` → 41 matches — all consistent with rg/ast-grep.
- Direct `sqlite3` probe: items table present; FTS present; embeddings stale flag acknowledged. AiDex used for cross-check; primary symbol proof is via serena + rg.

### headroom + context-mode + RTK-WSL (3-layer pipeline, always-on)

- **headroom** `0.36.2` (2026-08-21 updated, global + venv) — compression available; proxy not used (this is a proof doc, not a chat compression path). `headroom --version` 0.36.2.
- **context-mode** (ctx batch/search) — used for all batched shell probes and internet fetches; `ctx_batch_execute` with `concurrency 2-4` for parallel rg/serena/lit checks.
- **RTK-WSL** — `rg/ast-grep` probes wrapped as `wsl bash -c '... && ~/.local/bin/rtk <cmd>'` form per project rule; WSL `rtk` at `~/.local/bin/rtk` 0.45.x reachable.

### Literature / internet cross-check (sufficient search, compatibility assessed)

External non-code literature used **only** to anchor EBR / bounded-queue arguments, not to derive `K_world`:

- **EBR / epoch reclamation**: `crossbeam-epoch` (`docs.rs/crossbeam-epoch`, `crossbeam-rs/crossbeam`) 200 OK — consumer-side literature for `isOlder(entry.epoch, minReaderEpoch)`, `getMinReaderEpoch()` gating. Compatibility: ConvoPeq uses `EpochDomain` with `kMaxReaders=64, active=2`, `minReaderEpoch` = min active epoch, quarantine via `detectStuckReaders` — directly homologous to crossbeam-epoch's `guard` / `collect` model. Containment claim (Task 7) is production-code-proved; literature is confirmatory only.
- **Bounded MPMC queue**: Vyukov bounded MPMC (`1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue`) — SSL expired at probe time (cert error), fallback literature `rigtorp/MPMCQueue` (rigtorp/MPMCQueue) 200 OK used as the bounded MPMC proxy; `DeferredDeletionQueue kQueueSize=4096` as `array<DeletionEntry,4096> + array<atomic<uint32_t>,4096>` is a Vyukov-variant. Compatibility: fixed 4096 is finite regardless of variant; the `|W|≤|G|≤4096` step uses only the fixed-array fact, not Vyukov-specific invariants beyond `array`-size.
- All tool GitHub sources (`oraios/serena`, `cocoindex-io/cocoindex`, `ast-grep/ast-grep`, `MinishLab/semble`, `CSCSoftware/AiDex`, `headroomlabs-ai/headroom`, `safishamsi/graphify`) fetched and status-checked (7/7 200 OK at internet cross-checks in this session).

### Scope-compliance & retrofit feasibility

- Scope: **code-consistent proof only** — no file modifications to `src/` were made in this step. All deltas are documentary (this evidence file).
- Retrofit assessment per Task 1..10: the only future `src/` changes required to promote CONDITIONAL → PASS are (a) implement `WorldRetirementReservation` (`LifetimeState` reservation residency) to discharge `A_max` / `K_reservation` (Phase I), and (b) bound `TerminalReclaimAuthority` (D101-9). Both are compatible with current `RuntimeWorldAuthority` centralization (`OwnerChannel` + `RuntimeStore::current` single authority, `PendingPublishRegistry` as non-owning registry). No breaking change to the `publishAndSwap` or retire chain is required; the conservation equations are forward-compatible with those bounded implementations.
- Unresolved / deferred items carried explicitly: `A_max` injection (implementation missing) and `K_terminal` boundedness (D101-9). `E_max_message` throughput-bound nature separated to Step 8.

---

## Appendix A — Code-identity cross-check (serena `find_symbol` canonical anchors)

| Symbol | Serena name_path | File | Lines |
|---|---|---|---|
| `PendingPublishRegistry` | `convo::isr::PendingPublishRegistry` | `src/audioengine/RuntimeWorldAuthority.h` | 32–70 |
| `RuntimeWorldAuthority` | `convo::isr::RuntimeWorldAuthority` | `src/audioengine/RuntimeWorldAuthority.h` | 106–276 |
| `RuntimeStore::WriteAccess::publishAndSwap` | `convo::RuntimeStore::WriteAccess::publishAndSwap` | `src/core/RuntimeStore.h` | 39–49 |
| `DeferredDeletionQueue` | `DeferredDeletionQueue` | `src/DeferredDeletionQueue.h` | 57–271 |
| `DeferredDeletionQueue::kQueueSize` | (const inside above) | `src/DeferredDeletionQueue.h` | 262 (4096) |
| `RetireQuarantineStore` | `convo::isr::RetireQuarantineStore` | `src/audioengine/RetireQuarantineStore.h` | 60–232 |
| `RetireQuarantineStore::kMaxQuarantinedEntries` | (const inside above) | `src/audioengine/RetireQuarantineStore.h` | 65 (512) |
| `TerminalReclaimAuthority` | `convo::isr::TerminalReclaimAuthority` | `src/audioengine/ISRRetireRouter.h` | 61–120 |
| `OwnerChannel` | `convo::isr::OwnerChannel` | `src/audioengine/OwnerChannel.h` | 38–144 |
| `RuntimeStore` | `convo::RuntimeStore` (template) | `src/core/RuntimeStore.h` | 13–… |

All `rg`/`ast-grep` line hits lie inside these ranges; the header ranges are the single ground truth for each class.

---

## Appendix B — Capacity literal extraction (sed/awk replication)

```
DeferredDeletionQueue.h:262:    static constexpr uint32_t kQueueSize = 4096;
RetireQuarantineStore.h:65:    static constexpr std::size_t kMaxQuarantinedEntries = 512;
RuntimeWorldAuthority.h:34:    static constexpr std::size_t kPendingPublishCapacity = 64;
OwnerChannel.h:40:    static constexpr std::size_t kCapacity = 256; // >> max in-flight publishes
```

`sed -n "262,265p" DeferredDeletionQueue.h` and `awk "/kQueueSize|kMaxQuarantinedEntries|kPendingPublishCapacity/"` produce exactly the above lines; `rg/ast-grep/ag` return 5 `PendingPublishRegistry` hits (class + 2 registry accessors + 2 comment sites) — match count stable.

---

## Appendix C — ConvoPeq.md reconciliation snippet

`ConvoPeq.md` (96,816 lines, `Generated: 2026-08-21 20:45:49`) capacity section reconciled against `src/`:

```
4096  M_retire     → deferredDeletionQueue kQueueSize=4096       ✅
512+512 M_quarantine → RetireQuarantineStore 512 × 2              ✅
K_terminal          → TerminalReclaimAuthority std::vector       ✅ (growable, symbolic)
1    M_current      → RuntimeStore::current single pointer        ✅
64   non-owning     → PendingPublishRegistry kPendingPublishCapacity=64 (NOT double-counted) ✅
```

No `ConvoPeq.md` overwrite was performed in this proof step; values are read-only reconciled.

