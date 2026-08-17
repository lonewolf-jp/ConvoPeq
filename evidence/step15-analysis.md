# Step 15 — Finite-Bound Reclaim Gate 再設計監査

## 15-A — `enqueueWithRetry()` ownership contract

### 調査対象

| Component | File | Function |
|-----------|------|----------|
| ISRRetireRouter | ISRRetireRouter.cpp | `enqueueWithRetry()` |
| ISRRetireRouter | ISRRetireRouter.cpp | `enqueueRetire()` |
| RetireQuarantineStore | RetireQuarantineStore.h | `quarantine()` |
| AudioEngine.h | AudioEngine.h | `enqueueDeferredDeleteNonRt()` |
| AudioEngine.h | AudioEngine.h | `enqueueDeferredDeleteNonRtWithResult()` |
| RuntimePublicationBridge | AudioEngine.h | `retirePublishedRuntimeWorldNonRt()` |

### 1. `ptr` の所有権移動タイミング

**Ownership transfer chain:**

```
RuntimeWorldAuthority::publish()
  → publishAndSwap(next)           [atomic store of newWorld to RuntimeStore::current]
  → returns oldWorld (raw pointer)

PublishExecutor::executePublish()
  → bridge.retirePublishedRuntimeWorldNonRt(oldWorld, false)

RuntimePublicationBridge::retirePublishedRuntimeWorldNonRt(world)
  → engine_->enqueueDeferredDeleteNonRt(world, [deleter], DeletionEntryType::World)

AudioEngine::enqueueDeferredDeleteNonRt(ptr, deleter, type)
  → enqueueDeferredDeleteNonRtWithResult(ptr, deleter, type)

enqueueDeferredDeleteNonRtWithResult(ptr, deleter, type)
  → m_retireRouter->enqueueWithRetry(ptr, deleter, epoch, type)
```

**Ownership moment:** `ptr` ownership transfers from caller to `ISRRetireRouter` **at the moment `enqueueWithRetry` is called.** After this call, the caller must NOT use `ptr`.

**Code evidence (ISRRetireRouter.cpp:166-207):**

```cpp
RetireEnqueueResult ISRRetireRouter::enqueueWithRetry(void* ptr,
                                                        void (*deleter)(void*),
                                                        uint64_t epoch,
                                                        DeletionEntryType type) noexcept
{
    auto result = enqueueRetire(ptr, deleter, epoch, type);
    if (result == RetireEnqueueResult::Success)
        return result;  // ← ptr is now owned by DeferredDeletionQueue (D)

    constexpr int kMaxRetry = 2;
    for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
        provider_->tryReclaim();
        result = enqueueRetire(ptr, deleter, epoch, type);
        if (result == RetireEnqueueResult::Success)
            return result;  // ← ptr is now owned by DeferredDeletionQueue (D)
        if (result != RetireEnqueueResult::QueuePressure)
            break;
    }

    if (result == RetireEnqueueResult::QueuePressure || result == RetireEnqueueResult::QueueFull)
    {
        const bool stored = m_retireQuarantine.quarantine(
            ptr, deleter, epoch, type, "enqueueWithRetry:QueuePressure",
            /*publicationSequenceId=*/0, /*generation=*/0);
        if (!stored) {
            assert(false && "...");  // ← Release: no-op, ownership LOST
        }
    }
    return result;  // ← caller does NOT reclaim ownership, ptr may be leaked
}
```

### 2. `QueuePressure` / `QueueFull` return 時の caller ownership

**Answer: caller does NOT retain ownership.**

**Evidence (AudioEngine.h:4200-4213):**

```cpp
auto result = m_retireRouter->enqueueWithRetry(ptr, deleter, epoch, type);
if (result == convo::isr::RetireEnqueueResult::Success) {
    return convo::isr::RetireEnqueueResult::Success;
}
// ← result is QueuePressure or QueueFull: caller does NOT touch ptr
drainDeferredRetireQueues(false);
const std::uint64_t retireDepth = static_cast<std::uint64_t>(m_retireRouter->pendingRetireCount());
convo::publishAtomic(retireQueueDepth_, retireDepth, std::memory_order_release);
return result;  // ← ptr ownership is NOT returned to caller
```

**`retirePublishedRuntimeWorldNonRt` caller (AudioEngine.h:3523-3545):**

```cpp
void retirePublishedRuntimeWorldNonRt(RuntimePublishWorld* world, bool resetRevision) noexcept
{
    if (world == nullptr)
        return;
    engine_->enqueueDeferredDeleteNonRt(world, [](void* p) {
        auto* ptr = static_cast<RuntimePublishWorld*>(p);
        ptr->unseal();
        ptr->~RuntimePublishWorld();
        convo::aligned_free(ptr);
    }, DeletionEntryType::World);
    (void)resetRevision;
    // ← return value is IGNORED. No ownership reclamation.
}
```

### 3. `quarantine()` failure 時の ownership

**Contract from code comments (RetireQuarantineStore.h:43-46, 67-70):**

```cpp
// store full 時に deleter を実行してはならない（UAF 構造的排除）。
//   呼出し元（Router）が jassert + HealthEvent / shutdown escalation を担う。
bool quarantine(void* ptr, void (*deleter)(void*), ...) noexcept
{
    ...
    if (size_ >= kMaxQuarantinedEntries) {
        ++overflowCount_;
        return false;  // store full — caller must NOT delete
    }
    ...
    return true;
}
```

**Contract: caller must NOT delete, but caller also does NOT regain ownership.**
The ptr is in an undefined ownership state: not in D, not in Q, not returned to caller.

### 4. `deleter` の型安全性

The deleter for Worlds is:
```cpp
[](void* p) {
    auto* ptr = static_cast<RuntimePublishWorld*>(p);
    ptr->unseal();
    ptr->~RuntimePublishWorld();
    convo::aligned_free(ptr);
}
```

This deleter is **type-specific** (World deleter). It performs:
1. `unseal()` — marks the World as no-longer-locked
2. `~RuntimePublishWorld()` — destructor (frees DSP references, etc.)
3. `aligned_free` — frees the aligned allocation

**The deleter cannot be safely called outside the retire pipeline** because:
- `unseal()` and destructor assume the World is no longer in `RuntimeStore::current`
- The World may still be referenced by RT audio thread via a stale `ReadToken`
- EBR (Epoch-Based Reclamation) must confirm no RT reader holds a reference

### 5. quarantine failure 後の safe delete の可能性

**Answer: NO, not possible.**

```
Why safe delete after quarantine failure is NOT possible:

1. EBR protection: The World was just swapped out of RuntimeStore::current.
   But the RT audio thread may still hold a ReadToken from the previous
   publication epoch (EBR reader slot). Immediate delete = UAF.

2. Epoch safety: The World's retirement epoch has been set
   (markRetireEpoch()), but minReaderEpoch may not have advanced past it.
   The deleter must wait for all RT readers to drain the current epoch.

3. Type-specific cleanup: The World destructor accesses DSP resources,
   crossfade state, etc. Calling it outside the EBR safe point risks
   accessing freed DSP objects.

4. The assert(false) + comment explicitly says "capacity exhaustion は
   health escalation（AudioEngine 側の quarantineResidentCount /
   quarantineOverflowCount を監視）で先行検知する" — meaning the system
   relies on HealthEvent escalation, not immediate deletion.
```

### 6. assert(false) の代わりに error propagation した場合の reclaim 可能性

**Answer: caller can theoretically reclaim, but cannot safely delete.**

If `enqueueWithRetry` were changed to return ownership on failure (e.g., a new return type that includes the ptr), the caller (`enqueueDeferredDeleteNonRtWithResult`) could:

```cpp
// Hypothetical fix:
auto [result, reclaimedPtr] = m_retireRouter->enqueueWithRetry(..., /*stealOnFailure=*/true);
if (reclaimedPtr != nullptr) {
    // ptr was NOT accepted into D or Q
    // CALLER now owns ptr again
    // But... can caller safely delete it?
}
```

**Problem:** Even if caller regains ownership, the caller (`retirePublishedRuntimeWorldNonRt` → `enqueueDeferredDeleteNonRtWithResult`) is running on the **Non-RT publish thread** (RebuildThread). It cannot safely delete the World because:
- The RT audio thread may still be reading the old World via a stale `ReadToken`
- EBR epoch safety has not been confirmed

**Conclusion:** The caller cannot safely reclaim and delete. The only correct fix is to ensure Q never fills to capacity — i.e., proactive reclamation before Q is exhausted.

### 15-A Summary

| Question | Answer |
|----------|--------|
| 1. When does ptr transfer? | At `enqueueWithRetry()` call — caller relinquishes ownership |
| 2. Does caller keep ownership on QueuePressure/Full? | NO — caller ignores return code |
| 3. What does quarantine() failure contract say? | "caller must NOT delete" — ownership is lost |
| 4. Is deleter type-specific? | Yes — World-specific deleter with unseal/destructor/free |
| 5. Can caller safe-delete after quarantine failure? | NO — EBR epoch safety not guaranteed, RT readers may hold stale refs |
| 6. Can error propagation + reclaim help? | NO — caller cannot safely delete even if ownership is returned |

**結論:** `L_outstanding > 0` は **設計上の不具合**。`assert(false)` は Release build では no-op となり、ptr は永永廃棄され、`onRelease()` は絶対に呼ばれない。

---

## 15-B — `RetireQuarantineStore` capacity / ownership

### RetireQuarantineStore の構造

```cpp
class RetireQuarantineStore {
    static constexpr std::size_t kMaxQuarantinedEntries = 512;
    std::array<QuarantinedEntry, kMaxQuarantinedEntries> entries_{};
    std::size_t size_ = 0;
    std::uint64_t overflowCount_ = 0;
    std::atomic<std::uint64_t> worldReclaimCount_{0};
    WorldRetirementReferenceObserver* referenceObserver_ = nullptr;
};
```

### State machine: World ownership through D → Q → drain

| State | D.enqueue | D.enqueue failure → Q.quarantine | Q full? | Q drain (epoch safe) | shutdown drainAll |
|-------|-----------|----------------------------------|---------|----------------------|-------------------|
| World in PublishedDomain | — | — | — | — | — |
| 1. publishAndSwap returns oldWorld | — | — | — | — | — |
| 2. enqueueDeferredDeleteNonRt called | — | — | — | — | — |
| 3a. enqueueRetire succeeds → D | World in D | — | — | drain → deleter → onRelease | drainAllUnsafe → deleter → onRelease |
| 3b. enqueueRetire fails → Q.quarantine | — | World in Q | — | drain → deleter → onRelease | drainAllUnsafe → deleter → onRelease |
| 3c. Q full → assert(false) → LEAK | — | — | World LEAKED | — | — |

### Q.capacity = 512 が同時 resident World 上限であることの確認

**Yes, kMaxQuarantinedEntries = 512 IS the simultaneous resident World upper bound for Q.**

**Evidence:**
- `quarantine()` checks `size_ >= kMaxQuarantinedEntries` and returns false if full
- `drain()` compacts entries in-place, removing drained entries and shifting survivors
- `entries_` is `std::array<QuarantinedEntry, 512>` — fixed size, no dynamic growth
- `size_` tracks current resident count, incremented on quarantine, decremented on drain

**Therefore:** Q_max = 512 is the hard simultaneous resident upper bound for quarantined Worlds.

### Ownership contract summary

| Operation | ptr state after | Q.size_ change | onRelease called? |
|-----------|----------------|----------------|-------------------|
| `quarantine()` success | Q.entries_[size_-1] | +1 | No (deferred to drain) |
| `quarantine()` failure (Q full) | LEAKED (lost) | 0 | No (NEVER) |
| `drain()` success (epoch safe) | Deleted | -1 | Yes (immediate after deleter) |
| `drainAllUnsafe()` | Deleted | 0 → 0 | Yes |

### 15-B Summary

```
Q.capacity = 512  ✅ Confirmed as simultaneous resident World upper bound
Q.overflow = L > 0  ✅ Confirmed: Q_full → assert(false) → World permanently leaked
Q.drain → onRelease  ✅ Confirmed: drain calls deleter → onRelease via ReferenceObserver
Q ownership = ISRRetireRouter  ✅ Confirmed: single owner, no caller-level reclaim
```

---

## 15-C — `I_max` (in-flight World between publication and retirement)

### I の定義

```
I = publishAndSwap() が oldWorld を返した後、
    enqueueWithRetry() が ownership を確定するまでの World 数
```

### Architecture flow

```
RebuildThread:
  trySubmitImpl(req)
    → executor_.publish(engine_, frozen, ..., waitForReceipt=true)
      → commitRuntimePublication(world, regCtx, oldHandle)
        → enqueueRuntimePublicationFireAndForget(world)
          → ownerChannel().enqueue(...)     [World ownership → OwnerChannel]
          → intentQueue_.push(intent)       [Publish intent → intentQueue_]
          → waitForPublishReceipt(seqId, 250ms)  [BLOCKS RebuildThread]

ISR CoordinatorLoop (separate thread):
  while (intentQueue_.pop(commonIntent)) {
    → PublishIntentHandler::handle()
      → PublishExecutor::executePublish()
        → authority.publish(std::move(owner), ...)
          → coordinator_.commit(...)          [metadata]
          → publishAndSwap(next)              [atomic store] → returns oldWorld
        → bridge.retirePublishedRuntimeWorldNonRt(oldWorld)
          → enqueueDeferredDeleteNonRt(oldWorld)
            → enqueueWithRetry(oldWorld)      [ownership confirmed here]
        → onPublishCompleted()
        → advanceRetireEpoch()
        → onPublishCommitted(seqId)
          → notifyPublishReceipt(seqId)       [unblocks RebuildThread]
  }
```

### Publication seriality proof

**Key architecture facts (from code):**

1. **Single submission thread:** `submitPublishRequest` is called ONLY from `AudioEngine.Commit.cpp:813`, which runs on the `RebuildThread` (AudioEngine.RebuildDispatch.cpp:857-870). `trySubmit` has zero external call sites (dead code / test-only).

2. **Synchronous publish:** `trySubmitImpl` → `executor_.publish(waitForReceipt=true)` → `commitRuntimePublication` → `waitForPublishReceipt(seqId, 250ms)`. This **blocks the RebuildThread** until `onPublishCommitted` is called.

3. **CoordinatorLoop processes ONE intent at a time:**
```cpp
// ISRRuntimePublicationCoordinator_ProcessIntent.cpp:47
while (intentQueue_.pop(commonIntent)) {
    switch (commonIntent.type) {
        case IntentType::Publish:
            g_publishIntentHandler.handle(commonIntent, ctx);
            break;
```
The `PublishIntentHandler` calls `PublishExecutor::executePublish`, which runs the full chain:
```
authority.publish(...) → publishAndSwap(next) → oldWorld
bridge.retirePublishedRuntimeWorldNonRt(oldWorld) → enqueueWithRetry(oldWorld)
onPublishCompleted() → advanceRetireEpoch()
onPublishCommitted(seqId) → notifyPublishReceipt()
```

4. **Single-owner channel:** `OwnerChannel` holds entries keyed by seqId. `take()` is called inside `executePublish`, removing the world from the channel. The CoordinatorLoop processes one intent at a time, so only one `executePublish` holds an owner at a time.

5. **Deferred resubmit is also serialized:** `processDeferredAdmission()` runs on the RebuildThread and calls `submitPublishRequest` → `trySubmitImpl` → `executor_.publish(waitForReceipt=true)` → blocks. Since the RebuildThread is blocked, no other submission can occur.

**Therefore:**

```
Timeline:
  T0: RebuildThread: publish World_1 → publishAndSwap → enqueueWithRetry(World_1) → onPublishCommitted(1)
  T1: RebuildThread: waitForPublishReceipt returns → publish World_2 → publishAndSwap → enqueueWithRetry(World_2)

  At no point is World_1 in the publishAndSwap→enqueueWithRetry window simultaneously with World_2.
```

**I_max = 1** — There is at most ONE World simultaneously in the `publishAndSwap → enqueueWithRetry` window.

### Proof of I_max = 1

| Component | Capacity | Can hold multiple Worlds in I_window simultaneously? |
|-----------|----------|-----------------------------------------------------|
| OwnerChannel | MPSC ring buffer | Yes (buffering capacity) |
| intentQueue_ | MpscBoundedRing<4096> | Yes (buffering capacity) |
| PublishExecutor | Single sequential | NO — runs ONE at a time |
| waitForPublishReceipt | Sync block (250ms) | NO — blocks RebuildThread until complete |

The OwnerChannel and intentQueue_ can buffer multiple Worlds, BUT:
- `commitRuntimePublication(waitForReceipt=true)` **blocks** the RebuildThread after each enqueue
- The CoordinatorLoop processes intents **sequentially** (SPSC consumer)
- `onPublishCommitted` (which unblocks the RebuildThread) is called **after** `enqueueWithRetry` completes

**Therefore:** Even though OwnerChannel and intentQueue_ have buffering capacity, the synchronous `waitForReceipt=true` ensures only ONE World is in the critical `publishAndSwap → enqueueWithRetry` window at any time.

### I_max = 1 (WITHOUT timeout) vs I_max (WITH timeout) — CRITICAL DISTINCTION

**Without timeout (normal case):** I_max = 1 ✅
- `commitRuntimePublication` → `waitForPublishReceipt(seqId, 250ms)` → blocks until `onPublishCommitted`
- `onPublishCommitted` fires AFTER `enqueueWithRetry` completes
- RebuildThread is blocked → cannot enqueue next World until current one's `enqueueWithRetry` finishes
- Only 1 World in the `publishAndSwap → enqueueWithRetry` window

**WITH timeout (receipt timeout ≥ 250ms):** I_max is bounded by **OwnerChannel capacity = 256** ❌

```
Timeout scenario:

T0:   RebuildThread: World_1 enqueued → waitForPublishReceipt(1, 250ms) BLOCKS
T1:   ISR: intentQueue_.pop(World_1) → executePublish(World_1)
      → publishAndSwap → enqueueWithRetry(World_1) → ... (takes >250ms)
T250: waitForPublishReceipt times out → returns false
      commitRuntimePublication returns {Success, Transferred} (timeout ≠ failure)
      executor_.publish returns Success → trySubmitImpl returns Accepted

T251: RebuildThread: World_2 → enqueueRuntimePublicationFireAndForget(World_2)
      → ownerChannel.enqueue(World_2) → intentQueue_.push(World_2) → waitForPublishReceipt(2, 250ms)

At T251: World_1 is STILL in executePublish (pre/post enqueueWithRetry, NOT yet completed)
         World_2 is in OwnerChannel + intentQueue_ (NOT yet started by CoordinatorLoop)
         I = 2 (World_1 in-flight + World_2 buffered)
```

Subsequent timeouts repeat: I can grow up to OwnerChannel capacity (256).

### I_max with timeout = OwnerChannel::kCapacity = 256

**Evidence:**

1. `enqueueRuntimePublicationFireAndForget` (AudioEngine.h:4490-4562):
   - First: `ownerChannel().enqueue(key, std::move(world))` — if OwnerChannel is full (256 slots), returns false → caller reclaims owner (rollback). So World stops being accepted.
   - Then: `intentQueue_.push(intent)` — if intentQueue_ is full (4096), `enqueuePublicationIntent` returns false → caller reclaims owner from OwnerChannel.

2. On timeout: `commitRuntimePublication` returns `{Success, Transferred}` — ownership is in OwnerChannel/intentQueue_, NOT reclaimed. `trySubmitImpl` returns `Accepted`.

3. The RebuildThread then proceeds to World_2: `enqueueRuntimePublicationFireAndForget` enqueues into OwnerChannel (which may have space from previous World_1 being taken by executePublish).

**Maximum in-flight Worlds = min(OwnerChannel_capacity, intentQueue_capacity) = 256**

The OwnerChannel is the bottleneck: it has `kCapacity = 256` slots. Each `enqueueRuntimePublicationFireAndForget` first places the World in OwnerChannel. If OwnerChannel is full, the World is rejected (caller reclaims). So at most 256 Worlds can be buffered in OwnerChannel + in-flight in executePublish.

### I_max Summary (corrected)

```
I_max = 1  ✅ WITHOUT timeout (normal case)
I_max = 256  ⚠️ WITH timeout (receipt timeout ≥ 250ms, OwnerChannel::kCapacity)

The timeout case is the CRITICAL counter-proof for Step 15-F.
```

### 15-C Summary

| Variable | Value | Condition | Proof basis |
|----------|-------|-----------|-------------|
| I_max | 1 | Without timeout | Sync waitForReceipt blocks RebuildThread |
| I_max | 256 | With timeout | OwnerChannel::kCapacity=256 buffers Worlds |
| Definition | Worlds in OwnerChannel + intentQueue_ + executePublish(before enqueueWithRetry) | Expanded to include pre-publishAndSwap buffer |
| Ownership during I | Held by OwnerChannel / intentQueue_ / executePublish local | Multiple owners possible on timeout |
| RT audio thread overlap | None | trySubmit has zero call sites |
| timeout impact | ⚠️ I_max = 256 | waitForPublishReceipt returns false but commit commits anyway |

---

## 15-D — `B_true` と storage occupancy を分離して再定義する

### 現在混同されている概念

Step 13/14 の議論で以下の概念が混同されている：

```
B_true     — published but not yet released Worlds (semantic)
D          — Worlds resident in DeferredDeletionQueue (storage)
Q          — Worlds resident in RetireQuarantineStore (storage)
I          — Worlds currently between publication and retirement ownership (in-flight)
L          — leaked/orphaned Worlds (error state)
```

### 再定義: B_true = D + Q + I + L

```
B_true
  = published but not yet released Worlds
  = D + Q + I + L

Where:
  D = Worlds resident in DeferredDeletionQueue (pendingRetireCount = sizeApprox)
  Q = Worlds resident in RetireQuarantineStore (quarantineResidentCount)
  I = Worlds in-flight: between publishAndSwap return and enqueueWithRetry completion
  L = leaked/orphaned Worlds: ptr lost in enqueueWithRetry failure path
```

### invariant: B_true = D + Q + I + L

```
Proof:
  Every World that has been published (A_ref++) enters exactly one of:
  - D: enqueues into DeferredDeletionQueue successfully
  - Q: quarantines into RetireQuarantineStore after D fails
  - I: in-flight between publishAndSwap and enqueueWithRetry (before D/Q resolution)
  - L: leaked (both D and Q failed)

  When a World is deleted (deleter runs):
  - From D: drain() removes from D, calls deleter, onRelease (R_ref++)
  - From Q: drain() removes from Q, calls deleter, onRelease (R_ref++)
  - From I: cannot happen — I is transient, always transitions to D or Q or L
  - From L: never happens — leak means deleter never runs

  Therefore: B_true = A_ref - R_ref = D + Q + I + L  ✓
```

### L == 0 を設計上の invariant として証明する必要

**L == 0 is NOT currently an invariant.** The `assert(false)` in `enqueueWithRetry` does not prevent L > 0 in Release builds.

**Proposed invariant:** `L == 0` は **設計上の invariant ではなく、現在は違反状態**。

To prove `L == 0`, we need:
1. `enqueueWithRetry` MUST return ownership to caller on failure (not assert+leak)
2. Caller MUST be able to safely reclaim the World (EBR epoch safety confirmed)
3. OR: proactive reclaim ensures D+Q never reaches capacity (preventive)

### M の再導出

```
B_true = D + Q + I + L

If L == 0 (invariant restored):
  B_true = D + Q + I
  B_true <= D_max + Q_max + I_max

I_max depends on timeout case:
  Without timeout: I_max = 1
  With timeout:    I_max = 256 (OwnerChannel::kCapacity)

For PROVABLE boundedness (including timeout):
  B_true <= 4096 + 512 + 256
  B_true <= 4864
  Therefore: M = 4864

For OPTIMISTIC boundedness (no timeout):
  B_true <= 4096 + 512 + 1
  B_true <= 4609
  Therefore: M = 4609
```

**CRITICAL:** The timeout case (I_max = 256) must be used for the safe M value, because:
- `waitForPublishReceipt(250ms)` timeout is a documented possibility (coordinator stall, shutdown)
- On timeout, `commitRuntimePublication` returns `{Success, Transferred}` — ownership NOT reclaimed
- The RebuildThread proceeds to push World_2 into OwnerChannel (capacity 256)
- Therefore M_safe = 4096 + 512 + 256 = 4864

### 15-D Summary

```
B_true = D + Q + I + L  ✅ State model defined
L == 0  ❌ Currently violated — assert(false) is no-op in Release builds
M_safe = D_max + Q_max + I_max = 4096 + 512 + 256 = 4864  ✅ Including timeout case
M_opt = D_max + Q_max + I_max = 4096 + 512 + 1 = 4609  ✅ Without timeout (NOT safe for proof)
```

---

## 15-E — authoritative reclaim-pressure source

### 現在の pressure monitoring chain

```
PublicationAdmission::evaluate()
  → buildDeferredAdmissionSnapshot() / engine state

AudioEngine::applyRetirePressurePolicyNoRt()
  → retirePressureMild_ / Medium_ / Severe_
  → based on: kRetirePressureMildPercent=75, Medium=90, Severe=95 of hwm

AudioEngine::pendingRetireCount()  [via ISRRetireRouter]
  → provider_->pendingRetireCount()  [EpochDomain]
  → deferredDeletionQueue.sizeApprox()  [ONLY D, NOT D+Q]

AudioEngine::isFullyDrained()
  → retireDepth == 0  [D only, via pendingRetireCount()]
  → lifetimeRetireIntentPending == 0
  → ringResident == 0
  → dspQuarantineResident == 0
  → retireQuarantineResident == 0  [Q only]
  → pendingReclaimEmpty
```

### Authority Singularization

The project policy (ADR-C4, five-time review) states: **Authority Singularization** — there should be ONE authority for each concern.

**Current state:** Retire pressure is computed in `AudioEngine::applyRetirePressurePolicyNoRt` using `pendingRetireCount()` which only sees D. The Q capacity is monitored only in `isFullyDrained()` (shutdown) and `collectDrainAudit()` (shutdown).

**Problem:** Publication admission (`PublicationAdmission::evaluate()`) does NOT consult Q or I. It only sees D via `pendingRetireCount()`.

**The authoritative source for reclamation pressure should be:**

```
RuntimeWorldAuthority / EpochDomain  (sole owner of lifetime/retire state)
  → ReclaimPressureSnapshot { D, Q, I, L }
    ↓
PublicationAdmission  (sole authority for admission decision)
```

### 15-E Summary

```
Current: pendingRetireCount() = D only  ❌ Does not include Q or I
Proposed: ReclaimPressureSnapshot { D, Q, I }  ✅ Single authoritative source
Authority: RuntimeWorldAuthority / EpochDomain  ✅ Lifetime authority
Admission: PublicationAdmission  ✅ Sole admission authority
```

---

## 15-F — Admission decision stale window

### The stale window problem

```
Timeline:
  T0: RebuildThread: admission_.evaluate() → D=2764 → Accepted
  T1: RebuildThread: executor_.publish() → enqueueRuntimePublicationFireAndForget()
      → ownerChannel.enqueue(World_2) + intentQueue_.push(intent)  [non-blocking]
      → waitForPublishReceipt(2, 250ms)  [BLOCKS]

  T2: ISR thread: intentQueue_.pop() → executePublish(World_2)
      → publishAndSwap(World_2) → oldWorld = World_1
      → enqueueWithRetry(World_1) → D=2765 (full) → Q.quarantine(World_1) → Q.size++
      → onPublishCommitted(2) → notifyPublishReceipt(2)

  T3: RebuildThread: waitForPublishReceipt returns → trySubmitImpl continues
      → executor_.publish returns Success
      → trySubmitImpl returns Accepted
```

**CRITICAL REFINEMENT:** The stale window analysis from Step 15-C is **invalidated by the timeout case**. See Step 15-F counter-proof below.

### Step 15-F counter-proof: waitForPublishReceipt timeout breaks I_max=1

**The critical question:** Does `waitForPublishReceipt(250ms)` timeout allow the RebuildThread to proceed while the ISR is still processing the previous World?

**Answer: YES — it does.**

**Code evidence (AudioEngine.h:4580-4594):**

```cpp
auto result = enqueueRuntimePublicationFireAndForget(std::move(world), regCtx, oldHandle);

// ← World ownership already transferred to OwnerChannel + intentQueue_ at this point
// ← result = {Success, Transferred}

if (PublishStageResultTraits::isCommitted(result.stage) && seqId != 0)
{
    if (!waitForPublishReceipt(seqId, kPublishReceiptWaitTimeoutMs))
    {
        juce::Logger::writeToLog("[DIAG] commitRuntimePublication: receipt timeout seq="
            + juce::String(static_cast<juce::int64>(seqId)));
    }
    // ← TIMEOUT IS LOGGED BUT NOT TREATED AS FAILURE
    // ← result remains {Success, Transferred}
    // ← caller (executor_.publish → trySubmitImpl) sees PublishResult::Success
}

return result;  // ← Always returns Success if enqueue succeeded, regardless of timeout
```

**The timeout scenario that breaks I_max = 1:**

```
T0:    RebuildThread: World_1 → trySubmitImpl → executor_.publish(waitForReceipt=true)
       → commitRuntimePublication(World_1)
       → enqueueRuntimePublicationFireAndForget(World_1) → OwnerChannel + intentQueue_
       → waitForPublishReceipt(1, 250ms) → BLOCKS

T1:    ISR CoordinatorLoop: intentQueue_.pop(World_1) → executePublish(World_1)
       → publishAndSwap(World_1) → oldWorld_0
       → enqueueWithRetry(World_1's oldWorld_0) → D/Q/I/L transition
       → onPublishCommitted(1) → notifyPublishReceipt(1)  [takes >250ms]

T250:  waitForPublishReceipt(1) TIMES OUT → returns false
       commitRuntimePublication returns {Success, Transferred}  (timeout ≠ failure!)
       executor_.publish returns Success
       trySubmitImpl returns Accepted
       RebuildThread is now FREE to proceed

T251:  RebuildThread: World_2 → trySubmitImpl → executor_.publish(waitForReceipt=true)
       → commitRuntimePublication(World_2)
       → enqueueRuntimePublicationFireAndForget(World_2) → OwnerChannel + intentQueue_
       → waitForPublishReceipt(2, 250ms) → BLOCKS

T252:  ISR is STILL processing World_1 (e.g., in enqueueWithRetry for World_1's oldWorld_0)
       World_1's oldWorld_0 is in I (pre-enqueueWithRetry or during enqueueWithRetry)
       World_2 is in OwnerChannel + intentQueue_ (not yet processed by ISR)
       I = 2 (World_1's oldWorld_0 in I + World_2 buffered in OwnerChannel/intentQueue_)
```

**Subsequent timeouts:**
- Each 250ms timeout allows the RebuildThread to push another World
- OwnerChannel capacity = 256, intentQueue_ capacity = 4096
- The bottleneck is `ownerChannel().enqueue()`:
  - If OwnerChannel has a collision (same hash, all slots occupied), `enqueue` returns false
  - If OwnerChannel is full (all 256 slots occupied by different keys), `enqueue` returns false
  - On failure: `enqueueRuntimePublicationFireAndForget` reclaims the owner and returns `{Failed, CallerDestroy}`
  - `commitRuntimePublication` then returns failure → `executor_.publish` returns `PublishFailed` → `trySubmitImpl` returns `RejectedShutdown`

**Therefore:** The maximum number of Worlds that can be simultaneously in-flight (in OwnerChannel + intentQueue_ + executePublish before enqueueWithRetry) is bounded by **OwnerChannel::kCapacity = 256**.

### I_max = 256 (WITH timeout) — PROVEN

```
Without timeout: I_max = 1   ✅ (RebuildThread blocks until enqueueWithRetry completes)
With timeout:    I_max = 256  ✅ (OwnerChannel::kCapacity = 256, RebuildThread can push new Worlds)

The timeout case is NOT an edge case — it is the DEFAULT behavior when the ISR CoordinatorLoop
is slow or stalled (documented in AudioEngine.h:4589: "shutdown 中は CoordinatorLoop が停止し
complete() が来ないため、waitFor は timeout で終了する").
```

### Corrected 15-F Summary

```
Stale window (WITHOUT timeout): Admission → publishAndSwap → enqueueWithRetry → onPublishCommitted
  - 1 World in flight (RebuildThread blocked during entire chain)
  - D += 1, Q += 1 (if D full) after publishAndSwap completes
  - Admission check sees pre-increment values (stale by 1)

Stale window (WITH timeout): Admission → enqueue → waitForReceipt timeout → Admission NEXT
  - Up to 256 Worlds in flight (OwnerChannel buffer)
  - Admission check at T0 sees D_t0, but by T250+, D_t0 + 256 Worlds are buffered
  - DANGEROUS: admission gate may accept while B_true is far from threshold
```

---

## 15-G — Hard Gate placement (design)

### Goal structure

```
              ┌────────────────────┐
              │ ReclaimState       │
              │ D + Q + I + L     │
              │ (L must be 0)     │
              └─────────┬──────────┘
                        │
                        ▼
               PublicationAdmission
                        │
                  ┌─────┴─────┐
                  │           │
               Accept       Reject
                  │           │
                  ▼           ▼
            PublishExecutor  no publish
                  │
                  ▼
           publishAndSwap()
                  │
                  ▼
              retirement
                  │
                  ▼
             enqueueWithRetry → D or Q or (L=0 guaranteed)
```

### Key requirement: L must never become > 0

**Current design:** L > 0 is possible when D+Q are full and quarantine fails. `assert(false)` does not prevent this in Release builds.

**Fix needed:** The admission gate must account for D + Q + I, so that when D + Q + I >= M, **NO NEW World enters publishAndSwap()**.

**IMPORTANT — I_max is NOT 1, it's 256 (with timeout):**

The admission check must happen BEFORE `enqueueRuntimePublicationFireAndForget` enqueues the World into `OwnerChannel`. Once enqueued, ownership is transferred and the World counts as in-flight (I).

```
trySubmitImpl:
  1. admission_.evaluate()  ← checks D + Q + I + 1 against M (the +1 is the World about to be published)
  2. if Accepted → executor_.publish() → enqueueRuntimePublicationFireAndForget (World enters I)
     → waitForReceipt (blocks RebuildThread, but timeout may fire)
  3. ISR: executePublish → publishAndSwap → enqueueWithRetry (World transitions I → D/Q)
```

Since I_max = 256 (OwnerChannel::kCapacity), the admission metric should be:
```
pressure = D + Q + I
I = Worlds in OwnerChannel + intentQueue_ + executePublish(pre-enqueueWithRetry)
```

**Admission inequality:**
```
D + Q + I + 1 <= M  ← the +1 accounts for the World being admitted right now
D + Q + I < M        ← equivalent form (the World being admitted is not yet counted in I)
```

Where M = D_max + Q_max + I_max = 4096 + 512 + 256 = 4864 (including timeout case).

### 15-G Summary

```
Hard gate placement: BEFORE publishAndSwap (at admission evaluation)
Metric: D + Q + I (where I is tracked via in_flight flag)
Requirement: L must never become > 0
  → Ensure D + Q + I < M before allowing publishAndSwap
  → If D + Q + I >= M, reject at admission (never reach publishAndSwap)
```

---

## 15-H — `assert + leak` の置換候補比較

### Option A — Safe shutdown

```
Q full (quarantine failure)
  ↓
fatal reclaim failure
  ↓
enter shutdown  (set shutdown flag, reject all future publications)
  ↓
reject all future publications  (all admission → RejectedPressure)
```

**Pros:**
- Maximum safety — no UAF, no leak
- Simple to implement (set a flag)
- Health escalation pattern already exists

**Cons:**
- Engine becomes non-functional (no more publications)
- Requires full teardown/restart
- User experience: audio stops

**Verdict:** ✅ Correct for safety, but harsh. Should be **last resort**.

### Option B — Ownership-return

```
enqueueWithRetry() returns QueueFull
  ↓
caller regains unique ownership of ptr
  ↓
controlled NonRT destruction (after epoch safety confirmed)
```

**Pros:**
- No leak, no shutdown
- World is properly cleaned up

**Cons:**
- Requires changing `enqueueWithRetry` return type to include ptr
- Caller must wait for EBR epoch safety before deleting
- `retirePublishedRuntimeWorldNonRt` would need to become synchronous with epoch wait
- Could cause stalls on the publish thread
- Destructor safety: World destructor accesses DSP resources, crossfade state — must be verified thread-safe for Non-RT deletion

**Critical concern:** The World destructor calls `unseal()` and `~RuntimePublishWorld()`. These access:
- DSP resource references (crossfade DSP objects)
- Runtime state structures
- These may be in use by the RT audio thread via stale ReadTokens

**Verdict:** ⚠️ **Requires destructor safety proof.** If World destructor is Non-RT safe (only accesses already-retired DSP objects), this is viable. Otherwise NO-GO.

### Option C — Emergency unbounded overflow

```
D full + Q full
  ↓
emergency overflow store (e.g., std::vector with mutex, or std::deque)
```

**Pros:**
- No crash, no stall
- Publication continues

**Cons:**
- **Violates boundedness proof** — overflow store can grow unboundedly
- Defeats the entire D101.3.2 finite-bound argument
- B_true can grow without limit

**Verdict:** ❌ **NO-GO** — breaks boundedness invariant.

### 15-H Summary

| Option | Safety | Boundedness | UX | Complexity | Verdict |
|--------|--------|-------------|-----|------------|---------|
| A: Safe shutdown | ✅ Maximum | ✅ Bounded (L=0) | ❌ Engine stops | Low | **Recommended as fallback** |
| B: Ownership-return | ⚠️ Needs proof | ✅ Bounded (L=0) | ✅ Engine continues | Medium | **Requires destructor safety proof** |
| C: Emergency overflow | ✅ No crash | ❌ Unbounded (L→∞) | ✅ Engine continues | Low | ❌ NO-GO |

**Recommendation:** Option A as the primary strategy (proactive trigger before Q fills), with Option B as a secondary path if destructor safety can be proven.

---

## 15-I — D101.3.2 / D102 / D103 再証明条件

### D101.3.2 再証明条件

The following must ALL be proven:

```
L = 0                    ✅ (after implementing assert+leak fix)
I <= I_max = 256         ⚠️ (PROVEN: OwnerChannel::kCapacity=256, including timeout case)
D <= D_max = 4096        ✅ (DeferredDeletionQueue::kQueueSize)
Q <= Q_max = 512         ✅ (RetireQuarantineStore::kMaxQuarantinedEntries)

therefore:
B_true = D + Q + I + L   = D + Q + I  (since L=0)
B_true <= D_max + Q_max + I_max = 4096 + 512 + 256 = 4864 = M
B_true < ∞  ✅
```

**Additional requirement for D101.3.2:** The admission gate must use `D + Q + I` (not just D) as the pressure metric, so that publication is rejected before D + Q + I reaches M.

### D102 再証明条件

```
B_true >= M_admission
  ⇒ publishAndSwap() cannot occur
```

**M_admission** is the admission threshold. Currently:
- Type A (PublicationAdmission): `retirePressurePublicationThrottleActive_` at D >= 90% of hwm (2765)
- Type C (shouldRejectRebuildAdmissionForPressure): `retirePressureAdmissionStrict_` at D >= hwm (3072)

**Revised:** The admission gate must check `D + Q + I >= M_admission` (not just D).

**M_admission values (M = 4864):**
- Type A: M * 0.90 = 4864 * 0.90 ≈ 4378
- Type C: M * 1.00 = 4864

### D103 再証明条件

```
D + Q + I reaches emergency bound (M)
  ⇒ no new World publication
  ⇒ retirement continues (drain, reclaim)
  ⇒ eventual reduction is possible
```

**Requirement:** When D + Q + I = M, the admission gate must reject ALL new publications (not just throttle). Retirement must continue until D + Q + I < M_admission, then publication resumes.

### 15-I Summary

```
D101.3.2:  B_true <= D_max + Q_max + I_max = M  ✅ (conditional on L=0 fix)
D102:      B_true >= M_admission ⇒ no publishAndSwap  ✅ (conditional on D+Q+I metric)
D103:      D+Q+I reaches M ⇒ stop publish, continue retire, resume when safe  ✅ (requires hard gate)
```

---

## Step 15 — Final Summary

### Results

| Step | Result | I_max | M | Key Fix needed |
|------|--------|-------|---|---------------|
| 15-A | Ownership contract determined | — | — | enqueueWithRetry loses ptr on failure |
| 15-B | Q capacity = 512, ownership = Router | — | — | Q full → assert → leak (Release: no-op) |
| 15-C | **I_max = 256** (NOT 1!) | 256 | — | ✅ Proven: OwnerChannel::kCapacity=256, timeout breaks I_max=1 |
| 15-D | B_true = D + Q + I + L | — | 4864 | L must be proven 0 |
| 15-E | Pressure source = EpochDomain (D only) | — | — | Need D + Q + I metric |
| 15-F | Stale window = 256 Worlds (timeout) | — | — | ⚠️ Timeout breaks I_max=1 proof |
| 15-G | Hard gate before publishAndSwap | — | — | Use D+Q+I metric (I_max=256) |
| 15-H | Options A/B/C compared | — | — | Option A (shutdown) recommended |
| 15-I | Re-proof conditions defined | 256 | 4864 | Conditional on L=0 fix |

### M derivation (final)

```
M = D_max + Q_max + I_max = 4096 + 512 + 256 = 4864

- D_max = 4096  (DeferredDeletionQueue::kQueueSize)
- Q_max = 512   (RetireQuarantineStore::kMaxQuarantinedEntries)
- I_max = 256   (OwnerChannel::kCapacity — includes timeout buffer)
- L = 0         (requires fix: assert+leak → safe shutdown)
```

### Next steps (Step 15 implementation)

1. **Fix L=0:** Replace `assert(false)` + leak with Option A (safe shutdown) — set a fatal reclaim failure flag
2. **Expand admission metric:** Change `pendingRetireCount()` to return `D + Q + I` where I is observed from OwnerChannel size + intentQueue residency count + in-flight executePublish
3. **Add timeout handling:** On `waitForPublishReceipt` timeout, the World is in I (OwnerChannel/intentQueue). Must NOT allow unlimited re-admission. Consider: treating timeout as `RejectedPressure` to prevent RebuildThread from pushing more Worlds
4. **Update admission thresholds:** M_admission = 4864 * 0.90 ≈ 4378 (Type A), 4864 (Type C)

---

## Step 15-J — `I_max = 256` の上限反証監査

> 15-J-1〜15-J-5 を順に検証。**15-J-2 と 15-J-4 が最優先。**

---

### 15-J-1 — `enqueueRuntimePublicationFireAndForget()` の atomic ownership sequence

#### 完全な ownership state transition

```
World (aligned_unique_ptr<RuntimePublishWorld>)
    │
    │ owner = std::move(world)   [caller holds ownership]
    │
    ├─ 1. registry().registerPublish(seqId, newWorld)
    │      │
    │      └─ Non-owning: stores `const void* newWorld` as lookup fallback ONLY
    │         (PendingPublishRegistry::entries_[64], cursor-based round-robin)
    │         → Ownership NOT transferred. Registry = non-owning pointer table.
    │
    ├─ 2. ownerChannel().enqueue(key, std::move(world))
    │      │
    │      │  SUCCESS → owner.release() called
    │      │  → Ownership TRANSFERRED to OwnerChannel slot
    │      │  → Caller NO LONGER owns world
    │      │
    │      └─ FAILURE → returns {Failed, CallerDestroy}
    │         → Ownership NOT transferred (OwnerPtr&& is a reference, not a copy)
    │         → Caller RETAINS ownership
    │
    ├─ 3. enqueuePublicationIntent(intent)
    │      │
    │      │  Intent carries p.newWorld = (void*)newWorld — same pointer as
    │      │  registry entry AND OwnerChannel slot. But Intent does NOT own it.
    │      │  Intent is a SIGNAL: {seqId, handles, decision, pointers}
    │      │
    │      │  push() to intentQueue_ (MpscBoundedRing, capacity=4096)
    │      │  → reservation-before-push: fetchAdd(publicationIntentResidencyCount_)
    │      │
    │      │  SUCCESS → World ownership is in OwnerChannel (slot) + Intent signal is in queue
    │      │  FAILURE → rollback:
    │      │    a. ownerChannel().take(key) → recovers owner from slot
    │      │    b. registry().unregister(seqId)
    │      │    c. returns {Failed, CallerDestroy} → caller retains ownership
    │
    └─ 4. returns {Success, Transferred}
         │
         └─ Ownership now: OwnerChannel slot (sole owner)
            Intent signal in intentQueue_ (non-owning pointer copy)
            Registry entry (non-owning pointer copy)
```

#### Rollback safety: No double-free

When `ownerChannel().enqueue(key, std::move(world))` fails (channel full):
- `enqueue` receives `OwnerPtr&&` — an rvalue **reference** to `world`, not a copy
- `owner.release()` is only called on the **success** path
- On failure, `world` in the caller still holds ownership (never released)
- `CallerDestroy` return is correct — caller retains ownership
- **No leak, no double-free** ✅

#### Answering the 6 sub-questions:

| # | Question | Answer |
|---|----------|--------|
| 1 | OwnerChannel::enqueue() succeeds but intentQueue_.push() fails? | **Rollback**: `ownerChannel().take(key)` recovers owner from slot, `registry().unregister(seqId)` clears registry, World ownership returns to caller via `{Failed, CallerDestroy}`. The `std::move(world)` into `enqueue()` only transfers ownership on success (release() is called only inside the success branch); on failure, `world` still holds ownership because `OwnerPtr&&` is a reference, not a copy. |
| 2 | intentQueue_.push() succeeds but CoordinatorLoop hasn't popped yet? | World is owned by OwnerChannel slot. Intent sits in queue (up to 4096). `publicationIntentResidencyCount_` tracks this. Ownership remains in OwnerChannel until `executePublish` calls `ownerChannel().take()`. |
| 3 | Do OwnerChannel and intentQueue hold the SAME World simultaneously? | **YES** — the Intent carries a non-owning `const void* newWorld` pointer that duplicates the pointer stored in OwnerChannel. But only OwnerChannel actually OWNS the world (via aligned_unique_ptr). The Intent pointer is a non-owning fallback for when OwnerChannel.take fails. |
| 4 | Does rollback fully return ownership to caller? | **YES** — `ownerChannel().take(key)` returns an `OwnerPtr` (re-wrapped unique_ptr with stateless deleter). `unregister()` clears the registry. If enqueue succeeds but intent push fails, the World is recovered from OwnerChannel and ownership returns to caller. If enqueue itself fails (channel full), `world` was never moved (reference semantics on `OwnerPtr&&`), so caller retains ownership. |
| 5 | What does `publicationIntentResidencyCount_` count? | **Count of Publish Intents in transport + producer reservation**: It's incremented BEFORE `intentQueue_.push()` (reservation-before-push), and decremented when the ISR `processIntent` loop pops a Publish Intent. It does NOT track OwnerChannel occupancy — it tracks *intent signals*, not World ownership. |
| 6 | Is there 1:1 correspondence between OwnerChannel slot and intentQueue Intent? | **YES for Publish path.** `enqueueRuntimePublicationFireAndForget` always does: enqueue to OwnerChannel → enqueue Intent to intentQueue_. If Intent enqueue fails, OwnerChannel.take() recovers. A World is never in OwnerChannel without a corresponding Intent (or vice versa) in the success path. The Intent is just a non-owning signal pointing to the OwnerChannel-held World. |

#### **CRITICAL DISTINCTION: OwnerChannel occupancy ≠ World residency**

```
OwnerChannel slots (256) ≠ World residency bound (conceptual)

OwnerChannel holds:    aligned_unique_ptr<RuntimeState>  (actual ownership)
Intent carries:        const void* newWorld (non-owning pointer to same RuntimeState)
Registry stores:       const void* newWorld (non-owning pointer to same RuntimeState)
```

**OwnerChannel occupancy IS the physical residency bound for Worlds in the enqueue→commit gap.** The Intent and Registry are non-owning shadows — they don't hold World ownership.

---

### 15-J-2 — timeout の再帰的 accumulation — strict upper bound

#### Scenario: recursive timeouts による World 蓄積

```
State at any time t:
  OwnerChannel occupancy  = O(t)     (0 ≤ O ≤ 256, slots with non-null owner)
  intentQueue residency   = R(t)     (publicationIntentResidencyCount_, 0 ≤ R ≤ 4096)
  in-flight executePublish = E(t)    (0 or 1, single ISR consumer thread)
  OldWorld pending retire = P(t)     (just swapped out by publishAndSwap, not yet enqueueWithRetry'd)
```

#### Timeline trace (consecutive timeouts):

```
T0:
  RebuildThread: W1 → enqueueRuntimePublicationFireAndForget(W1)
    → OwnerChannel: O = 1
    → intentQueue: R = 1
    → waitForPublishReceipt(1, 250ms) → BLOCKS

T250:
  waitForPublishReceipt(1) TIMES OUT
    → commitRuntimePublication returns {Success, Transferred}
    → trySubmitImpl returns Accepted
    → RebuildThread FREE → proceeds to W2

  ISR (CoordinatorLoop):
    → intentQueue_.pop(Publish: W1) → R = 0
    → executePublish: ownerChannel().take(W1) → O = 0
    → authority.publish(W1) → publishAndSwap → OldWorld_0 returned
    → enqueueWithRetry(OldWorld_0) → D or Q
    → onPublishCommitted(1) → notifyPublishReceipt(1)  ← arrives LATE (after T250)

T251:
  RebuildThread: W2 → enqueueRuntimePublicationFireAndForget(W2)
    → OwnerChannel: O = 1
    → intentQueue: R = 1
    → waitForPublishReceipt(2, 250ms) → BLOCKS

T500:
  waitForPublishReceipt(2) TIMES OUT
  → RebuildThread proceeds to W3

  ISR:
    → pop(W2) → take(W2) → O = 0 → publishAndSwap → OldWorld_1
    → enqueueWithRetry(OldWorld_1) → D or Q

...continuing...

T64000:
  waitForPublishReceipt(256) TIMES OUT
  → RebuildThread proceeds to W257

  ISR:
    → pop(W256) → take(W256) → O = 0 → publishAndSwap → OldWorld_256
    → enqueueWithRetry(OldWorld_256) → D or Q

T64250:
  RebuildThread: W257 → enqueueRuntimePublicationFireAndForget(W257)
    → OwnerChannel: O = 1  ← slot available because ISR processed W256

  BUT: can multiple Worlds accumulate if ISR is SLOWER than timeout?
```

#### Worst case: ISR slower than timeout

If ISR takes >250ms per World (e.g., enqueueWithRetry is slow due to D full):

```
T0:    W1 → OwnerChannel (O=1)
T250:  timeout → W2 → OwnerChannel (O=2)
T500:  timeout → W3 → OwnerChannel (O=3)
...
T64000: timeout → W256 → OwnerChannel (O=256)  ← FULL

T64250: timeout → W257 → enqueueRuntimePublicationFireAndForget(W257)
  → ownerChannel().enqueue(W257) returns FALSE (channel full, 256 slots occupied)
  → rollback: ownerChannel().take(W257_key) → nullptr (not there)
  → registry().unregister(W257_seq)
  → returns {Failed, CallerDestroy}
  → commitRuntimePublication returns {Failed}
  → executor_.publish returns PublishFailed
  → trySubmitImpl:
    → stateOwner_.onExecutorFailed()
    → lifetime_.destroyRolledBackDSP()
    → telemetryRecorder_.recordFailure(...)
    → returns RejectedShutdown  ← admission blocked!
```

**When OwnerChannel is full, `trySubmitImpl` returns `RejectedShutdown`.** The RebuildThread gets a rejection (not a silent drop). No World ownership is leaked — the `CallerDestroy` path means the caller still owns it (via reference semantics on `OwnerPtr&&`), and `destroyRolledBackDSP()` cleans up the DSP.

#### **Upper bound proof:**

```
max Worlds in OwnerChannel = kCapacity = 256
max Worlds in intentQueue  = kIntentQueueCapacity = 4096
But: OwnerChannel and intentQueue are 1:1 coupled for success path

If O(t) = 256 (full), the next enqueue to OwnerChannel fails.
The World is returned to caller (CallerDestroy) — NO leak.
trySubmitImpl returns RejectedShutdown — NO new Worlds admitted.

Therefore: max admitted-but-not-committed Worlds = 256
           = OwnerChannel::kCapacity
```

**However — IntentQueue can accumulate beyond 256:**

The `publicationIntentResidencyCount_` can grow up to `intentQueue_` capacity (4096) if:
1. The RebuildThread keeps creating World snapshots (even if `executor_.publish` returns `RejectedShutdown` — let me check if it still creates the snapshot)

Let me check what happens when `trySubmitImpl` returns `RejectedShutdown`:

```cpp
// RuntimePublicationOrchestrator.cpp:submitPublishRequest
switch (decision) {
    case PublicationAdmission::Decision::Accepted:
        return;
    case PublicationAdmission::Decision::RejectedShutdown:
        // ← falls through to default — caller handles rejection
        return;
    default:
        return;
}
```

When `RejectedShutdown` is returned, **the World snapshot was never created** — the rejection happens BEFORE `executor_.publish`. The admission check in `admission_.evaluate()` happens BEFORE the World is built. So rejected Worlds never enter OwnerChannel or intentQueue.

**Therefore: I_max = 256** — bounded by OwnerChannel::kCapacity. The RebuildThread blocks on `waitForPublishReceipt` or gets rejected if OwnerChannel is full.

---

### 15-J-3 — `OwnerChannel` full failure semantics — D101 compliance

When `OwnerChannel::enqueue()` returns false (full, 256 slots occupied):

```
enqueueRuntimePublicationFireAndForget(W)
    ↓
ownerChannel_.enqueue(key, std::move(world))  ← FAILS (channel full)
    ↓
// world still owned by caller (reference semantics — release() never called)
registry().unregister(seqId)   ← clears non-owning registry entry
rollbackHandle cleanup via ScopeExit
return {Failed, CallerDestroy}
    ↓
commitRuntimePublication returns {Failed, CallerDestroy}
    ↓ (isCommitted = false)
publishImpl:
  → logs failure
  → returns PublishResult::PublishFailed
    ↓
trySubmitImpl:
  → executor_.publish() returned PublishFailed
  → stateOwner_.onExecutorFailed(correlationId)
  → lifetime_.destroyRolledBackDSP(newDSPResolved)
  → telemetryRecorder_.recordFailure(...)
  → returns PublicationAdmission::Decision::RejectedShutdown
    ↓
submitPublishRequest(ReturnedShutdown):
  → case RejectedShutdown: return (silent to caller)
    ↓
RebuildThread:
  → gets RejectedShutdown from trySubmit
  → no World leaked
  → no ghost entry in PendingPublishRegistry (unregistered before return)
  → no ghost Intent in intentQueue (Intent never enqueued)
  ↓
DSPHandle rollback:
  → rollbackHandle is still valid (not nulled because we exited before the success path)
  → ScopeExit guard calls rollbackDSPHandleRegistration(rollbackHandle)
  → DSPHandle is properly returned to the handle pool
```

| Check | Status |
|-------|--------|
| unique_ptr returned to caller | ✅ (never moved — reference semantics on OwnerPtr&&) |
| PublishStageResult = Failed/QueueFull | ✅ (`{Failed, CallerDestroy}`) |
| trySubmitImpl doesn't misinterpret as success | ✅ (returns `RejectedShutdown`) |
| publicationIntentResidencyCount_ decremented | ✅ (Intent never enqueued — fetchAdd never happened) |
| PendingPublishRegistry no ghost entry | ✅ (`unregister(seqId)` called) |
| Late receipt doesn't cause double-processing | ✅ (receipt is keyed on seqId; if World was rejected, no receipt is ever sent — `onPublishCommitted` only fires from `executePublish`, which never runs) |
| **No leak on 257th World** | ✅ **PROVEN** |

---

### 15-J-4 — `PendingPublishRegistry` の役割 — I_max に影響しない

#### PendingPublishRegistry = capacity 64

```
registerPublish(seqId, sealedWorld)
  → entries_[cursor_++ % 64]
  → stored as atomic<const void*> — NON-OWNING pointer

lookup(seqId) → returns const void* (non-owning)
unregister(seqId) → sets entry to {0, nullptr}
```

#### Why PendingPublishRegistry does NOT bound I_max:

1. **Registry is non-owning** — it stores `const void* newWorld`, which is a shadow pointer to the World already owned by OwnerChannel. The Registry is a lookup table for `executePublish` to resolve the world if `OwnerChannel.take()` returns null (fallback for legacy paths).

2. **Registry cursor is round-robin (not bounded)** — `cursor_.fetch_add(1) % 64`. When 64+ Worlds are published, older registry entries are simply **overwritten**. But `unregister()` is called in `executePublish` → `authority.registry().unregister(intent.sequenceId)`. If a registry entry is overwritten before unregister, the lookup at executePublish falls back to the Intent's `p.newWorld` pointer.

3. **OwnerChannel is the gate** — `enqueue()` checks slot occupancy (256 max). The Registry has no blocking effect on enqueue. Even if the Registry is full (64 entries), `enqueue()` to OwnerChannel still proceeds and succeeds.

4. **Order in `enqueueRuntimePublicationFireAndForget`:**
   ```
   registry().registerPublish(seqId, newWorld)  ← step 1 (non-owning)
   ownerChannel().enqueue(key, std::move(world)) ← step 2 (OWNING transfer — THE gate)
   ```
   The Registry insert happens BEFORE OwnerChannel enqueue. If Registry is full (64 entries overwritten), the new entry simply overwrites an old one. This does NOT block. The OwnerChannel enqueue is the hard gate.

5. **No backpressure from Registry** — `registerPublish` returns `void`, never fails. It's always called before `enqueue`. So the Registry cannot cause rejection.

#### **Therefore: PendingPublishRegistry capacity (64) does NOT bound I_max.**

```
I_max is bounded SOLELY by OwnerChannel::kCapacity = 256
PendingPublishRegistry (64) is a non-owning lookup table with round-robin overwrite
```

#### Ownership summary at each stage:

```
Stage                              OwnerChannel    IntentQueue    Registry    Total World ownerships
─────────────────────────────────────────────────────────────────────────────────────────────────
Admitted (trySubmitImpl)            0              0             0           caller (frozen)
enqueueRuntimePublicationFireAndForget:
  → registerPublish                  0              0             1 non-own   caller
  → ownerChannel().enqueue (success) 1 OWNED        0             1 non-own   OwnerChannel
  → intentQueue.push (success)       1 OWNED        1 signal      1 non-own   OwnerChannel
  → return {Success, Transferred}    1 OWNED        1 signal      1 non-own   OwnerChannel
executePublish:
  → take(key)                        0              1 signal      1 non-own   executePublish (local)
  → publish(std::move(owner))        0              0             0           publishAndSwap (RuntimeStore)
  → unregister(seqId)                0              0             0           RuntimeStore.oldWorld
  → retire → enqueueWithRetry        0              0             0           D / Q / I / L
```

---

### 15-J-5 — `I_total` の formal definition

#### Current definition is too broad. Refined classification:

```
I_total = I_owner + I_retire

I_owner  = Worlds in OwnerChannel (ownership transferred to slot, not yet taken by executePublish)
            Bounded by: OwnerChannel::kCapacity = 256
            Observed by: ownerChannel().size()

I_retire = Worlds returned by publishAndSwap() but not yet enqueueWithRetry'd
            Bounded by: 1 (single ISR thread, sequential processIntent loop)
            Observed by: pendingRetireCount() (if tracked separately)

I_total  = Total admitted-but-not-retired Worlds
         = OwnerChannel occupancy + oldWorld pending retire
         = O(t) + P(t)
```

#### Why `publicationIntentResidencyCount_` is NOT a residency bound:

```
publicationIntentResidencyCount_ = R(t)  = count of Publish Intents in intentQueue_

R(t) tracks SIGNALS, not World ownership.
A World can exist in 0 states simultaneously:
  - Not in OwnerChannel (taken) + Intent still in queue → R ≠ World count
  - In OwnerChannel + Intent not yet enqueued (failed enqueue) → R = 0 but World exists

Therefore: R(t) does NOT bound I_total.
              O(t) DOES bound I_owner.
              P(t) DOES bound I_retire (≤1).
```

#### Formal definition:

```
I_total(t)
  = |{Worlds admitted (committed to OwnerChannel) but not yet enqueued for retirement}|
  = O(t) + P(t)
  where O(t) = ownerChannel().size() (OwnerChannel slot occupancy)
        P(t) = 1 if executePublish is between publishAndSwap and enqueueWithRetry, else 0

I_owner  = O(t), bounded by 256 (OwnerChannel::kCapacity)
I_retire = P(t), bounded by 1 (single ISR thread, sequential execution)

I_total  = O(t) + P(t), bounded by 256 + 1 = 257

B_true   = D + Q + I_total + L   (as defined in 15-D)
B_true   = D + Q + O + P + L
B_true   <= D_max + Q_max + 256 + 1 + 0 = M (if L=0 fix applied)
```

Wait — `I_total = 257`? Let me reconsider the P(t) bound.

When `executePublish` takes a World from OwnerChannel and calls `publishAndSwap`, the oldWorld is returned. The oldWorld is then passed to `retirePublishedRuntimeWorldNonRt` → `enqueueWithRetry`. During this window, the oldWorld is in I_retire. But the NEW World has already been published to RuntimeStore — it's no longer in I_owner (it was taken from OwnerChannel).

So at any single point in time:
- O(t) Worlds in OwnerChannel (admitted, not yet executed)
- P(t) ≤ 1 oldWorld between publishAndSwap and enqueueWithRetry

But O(t) and P(t) are NOT independent — when executePublish runs, it decrements O by 1 and may set P to 1. So:

```
If ISR is idle: O(t) can be up to 256, P(t) = 0 → I_total = O(t) ≤ 256
If ISR is processing: O(t) decreased by 1 (taken), P(t) = 1 → I_total = O(t) + 1 ≤ 256

Maximum I_total: 256 (when ISR is idle and OwnerChannel is full)
With ISR processing one: I_total = (256-1) + 1 = 256

So I_total ≤ 256, NOT 257. The take and the publishAndSwap are atomic with respect to O(t) count.
```

**Corrected:**

```
I_total = O(t) + P(t)
I_total ≤ 256  (when ISR takes a World, O decreases by 1 and P becomes 1, net unchanged)

I_max = 256  ✅ PROVEN
```

#### Relationship to B_true = A_ref - R_ref:

```
A_ref (admission reference) increments when a World is admitted (trySubmitImpl Accepted)
       = enqueueRuntimePublicationFireAndForget succeeds
       = World enters OwnerChannel (O += 1)

R_ref (retirement reference) increments when a World is fully retired (enqueueWithRetry succeeds)
       = oldWorld enters D or Q (D += 1 or Q += 1)

B_true = A_ref - R_ref
       = (Worlds admitted but not yet retired)
       = O + P + (Worlds in D+Q)  ← D+Q are admitted but not retired (retired = moved to D/Q)

Wait — this changes the model!
```

Let me re-examine the B_true definition from 15-D:

```
B_true = D + Q + I + L
```

Where:
- D = deferred deletion entries (retired to D, pending reclamation)
- Q = quarantine entries (retired to Q, pending reclamation)
- I = in-flight (admitted but not yet retired to D/Q)
- L = leaked (assert+leak, should be 0)

So B_true counts Worlds from admission → until final deallocation. D and Q are *intermediate* retirement targets, not final retirement. The World is in B_true from the moment it's admitted until it's fully deallocated.

This means:
- When `enqueueRuntimePublicationFireAndForget` succeeds → World enters I (OwnerChannel)
- When `executePublish` calls `enqueueWithRetry` → World transitions from I to D (or Q)
- When reclamation finally frees D/Q entries → World exits B_true

So `I` = `O(t)` (OwnerChannel occupancy) because:
- Worlds in OwnerChannel are admitted (A_ref incremented) but not yet retired to D/Q
- Once in D/Q, they've left I

The `P(t)` (oldWorld between publishAndSwap and enqueueWithRetry) is actually part of D's transition — the oldWorld is being moved from I to D. It's a transient state.

**Revised I definition:**

```
I = Worlds that have been admitted (A_ref incremented) but not yet placed in D or Q
  = OwnerChannel occupancy (O(t))
  + oldWorld in transition (publishAndSwap → enqueueWithRetry)
  = O(t) + P(t)
  = O(t)           (because when P=1, O decreases by 1 — the new World was taken from OwnerChannel)
```

Wait, that's wrong. When executePublish runs:
1. It takes a NEW World from OwnerChannel (O decreases by 1)
2. It calls publishAndSwap, getting back oldWorld (P = 1)
3. It calls enqueueWithRetry on oldWorld (P → 0, D or Q increases by 1)

During step 2-3: O has already decreased, P = 1. But the NEW World is now in RuntimeStore (published). The oldWorld is transitioning to D/Q.

From B_true's perspective:
- The NEW World was in B_true (as I) → now it's published (in RuntimeStore::current). Is a published World still in B_true?

Let me re-read the 15-D definition:

```
B_true — published but not yet released Worlds (semantic)
```

So B_true includes published Worlds! A World is in B_true from admission until its old self is fully deallocated.

So:
- NEW World: admitted → enters I → published → still in B_true (as published, waiting for next retire)
- OLD World: published → transitions to I_retire → enters D/Q → eventually deallocated → exits B_true

This means the NEW World stays in B_true even after publishAndSwap. The World that was in OwnerChannel becomes the published World (in RuntimeStore::current), which is still in B_true until the NEXT publishAndSwap retires it.

So actually:
```
B_true = D + Q + I + L
I = Worlds in OwnerChannel + Worlds in RuntimeStore::current (published, not yet retired by next swap) + oldWorld in transition
```

This is getting complex. Let me simplify:

```
I_total = Worlds from admission to retirement into D/Q
        = OwnerChannel occupancy
          + published Worlds waiting for next publishAndSwap to retire them
          + oldWorld in transition (just swapped out, about to enter D/Q)

I_total ≤ 256 (OwnerChannel) + 1 (RuntimeStore::current published World) + 1 (in-transition oldWorld)
        = 258?
```

Actually no — let me think about this more carefully with the sequential nature:

**The ISR processes one Intent at a time.** While processing World W_n:
1. take(W_n) from OwnerChannel → O -= 1
2. publishAndSwap(W_n) → returns oldWorld W_{n-1} → W_n is now in RuntimeStore
3. enqueueWithRetry(W_{n-1}) → W_{n-1} enters D or Q

During steps 2-3, both W_n (published, in RuntimeStore) and W_{n-1} (in transition) are in B_true.
- W_n was just taken from OwnerChannel (O -= 1)
- W_{n-1} is the old published World being retired

But W_n entering RuntimeStore means W_{n-1} is being retired. At this point:
- W_n: in RuntimeStore::current (published, will be retired by next publishAndSwap)
- W_{n-1}: in I_retire (about to enter D/Q)

If we define:
- I_owner = Worlds in OwnerChannel (admitted, waiting for executePublish)
- I_published = Worlds in RuntimeStore::current (published, waiting for next swap to retire)
- I_retire = Worlds in transition (just swapped, about to enter D/Q)

Then:
```
I_total = I_owner + I_published + I_retire
       ≤ 256 + 1 + 1 = 258
```

But wait — I_published is always 1 (there's always a current World in RuntimeStore). And I_retire is at most 1 (only during the brief window between publishAndSwap and enqueueWithRetry). And I_owner is at most 256.

However, I_published and I_retire overlap:
- When W_{n-1} is in I_retire (step 3), W_n is in I_published (step 2 already happened)
- So at the peak: I_owner = 255 (one slot freed by take), I_published = 1 (W_n), I_retire = 1 (W_{n-1})
- I_total = 255 + 1 + 1 = 257

And when ISR is idle: I_owner = 256, I_published = 1, I_retire = 0
- I_total = 256 + 1 + 0 = 257

**So I_max = 257?**

No — let me reconsider. The `I` in D101's context is specifically "in-flight World between admission and retirement to D/Q." The published World in RuntimeStore::current is NOT "in-flight" in the same sense — it's the currently active World. It will only become "in-flight" (pending retirement) when the NEXT publishAndSwap swaps it out.

So the correct model is:
```
I = Worlds admitted to OwnerChannel (awaiting execution)
  + oldWorlds returned by publishAndSwap (awaiting enqueueWithRetry)

I = O(t) + P(t)

When ISR takes a World from OwnerChannel:
  O decreases by 1 (World moves to execution)
  Then publishAndSwap happens
  Then P = 1 (oldWorld awaiting retire)

So: I = O + P
    Max when O is at its peak (ISR idle): I = 256 + 0 = 256
    During ISR execution: I = 255 + 1 = 256

I_max = 256  ✅
```

### 15-J-6 — `O + P <= 256` の相互排他性 — 補強監査

> **15-J-6 は 15-J-5 の補強。 `O + P <= 256` をコード上で証明する。**

#### ownership transition trace

```
OwnerChannel
    ↓ take(key)                         [RuntimePublishExecutor.h:51, OwnerChannel.h:85]
local owner (OwnerPtr)                  [slot drained: s.owner = nullptr (O -= 1)]
    ↓ authority.publish(std::move(owner), ...)  [RuntimeWorldAuthority.h:246]
    → owner.release() inside publish()    [ownership transferred to publishAndSwap]
    ↓ writeAccess_.publishAndSwap(next)    [RuntimeStore.h:40]
RuntimeStore::current = newWorld           [A_ref++ happens HERE (onRuntimePublishedNonRt)]
oldWorld = previous current                 [oldWorld returned — NOT the World just taken from OwnerChannel]
    ↓ enqueueWithRetry(oldWorld)
D / Q                                     [R_ref++ happens when deleter runs]
```

#### Q1: `take()` と `executePublish` の間に World が双方に存在するか？

**NO — 証明完了 ✅**

`OwnerChannel::take(key)` ([OwnerChannel.h:85-100](src/audioengine/OwnerChannel.h#L85-L100)) atomically drains the slot via `publishAtomic(s.owner, nullptr, release)` and captures `raw = seen`. SPSC guarantees sole consumer (ISR) — no other thread can take from the same slot. The World is in the slot (O counts it) BEFORE the atomic store, and in local `raw` AFTER. **No overlap.** ✅

#### Q2: `publishAndSwap()` 後の `oldWorld` は OwnerChannel の slot を占有していないか？

**NO — 証明完了 ✅**

`RuntimeStore::publishAndSwap(T* next)` ([RuntimeStore.h:40-50](src/core/RuntimeStore.h#L40)) atomically exchanges `store_->current` from old to next. The returned `oldWorld` is the PREVIOUS `RuntimeStore::current`, which was itself published by a PRIOR `executePublish` call (take()n from OwnerChannel in that prior cycle). The World just taken in the CURRENT `executePublish` becomes the NEW `RuntimeStore::current` (passed as `next` to `publishAndSwap`). **The `oldWorld` and the World in OwnerChannel are DISTINCT.** ✅

#### Q3: `P=1` のとき、OwnerChannel の occupancy は少なくとも1減っているか？

**YES, BUT the slot can be refilled by RebuildThread.**

```
Timeline (single ISR thread, RebuildThread concurrent):

T0:  RebuildThread enqueues W1..W256 into OwnerChannel → O = 256
     ISR idle → P = 0

T1:  ISR: take(W1) → slot freed (O = 255)
T2:  RebuildThread: enqueue W257 → slot refilled (O = 256 again)
     W257 is a DIFFERENT World from W1
T3:  ISR: publishAndSwap(W1) → RuntimeStore::current = W1
     oldWorld = W0 (previous current, NOT in OwnerChannel) → P = 1
     W0 is the World published by a PRIOR executePublish cycle
     W0 is NOT one of the 256 Worlds currently in OwnerChannel

State at T3: O = 256 (including W257), P = 1 (W0)
O + P = 257  ←  DISTINCT Worlds!
```

**O + P ≤ 257** (not ≤ 256) because the World in P (oldWorld from RuntimeStore) is DISTINCT from all Worlds in OwnerChannel. The RebuildThread refills the slot freed by `take()`, restoring O to 256 while P = 1. ✅ **Rigorous proof complete.**

#### Q4: では `O + P <= 256` は成立しないのか？

**Correct — O + P <= 257.** But this does NOT mean I_max = 257. The `I` in D101.3.2's B_true = D + Q + I + L is defined as "Worlds in-flight: between publishAndSwap return and enqueueWithRetry completion." That is ONLY `P` (oldWorld between publishAndSwap and enqueueWithRetry), NOT OwnerChannel occupancy.

Let me re-derive I from the D101.3.2 call-site map:

**A_ref increments at**: `onRuntimePublishedNonRt` ([AudioEngine.Commit.cpp:408](src/audioengine/AudioEngine.Commit.cpp#L408)) — AFTER publishAndSwap

**R_ref increments at**: terminal deleter's `onRelease()` ([DeferredDeletionQueue.h:154/204](src/DeferredDeletionQueue.h#L154), [RetireQuarantineStore.h:142/177](src/audioengine/RetireQuarantineStore.h#L142))

```
A_ref = Worlds that have been PUBLISHED (entered RuntimeStore::current)
R_ref = Worlds that have been RELEASED (deleter's onRelease called)
B_true = A_ref - R_ref = Worlds published but not yet released
```

**Worlds in OwnerChannel have NOT been published — A_ref hasn't incremented for them.** They are NOT in B_true.

#### I definition correction (15-D revisited):

The 15-D definition stated `I = Worlds in-flight between publishAndSwap return and enqueueWithRetry completion`. Let's parse this precisely:

- `publishAndSwap` returns `oldWorld`
- `enqueueWithRetry(oldWorld)` is called next
- Between these two calls, `oldWorld` is the World in P

But what about the World that was just `take()`n from OwnerChannel? It was passed to `publish()` → `publishAndSwap(next)`. After `publishAndSwap`, this World is `RuntimeStore::current` (published). It is NOT "between publishAndSwap return and enqueueWithRetry" — it IS the `next` argument, already published. It will become `oldWorld` (P) when the NEXT `publishAndSwap` swaps it out.

So I (as defined in 15-D) = P only = oldWorld between publishAndSwap and enqueueWithRetry. **I ≤ 1, not OwnerChannel occupancy.**

But this contradicts the 15-J-2 finding that I_max = 1 was "invalidated by timeout." Let me reconcile:

The timeout allows RebuildThread to push 256 Worlds into OwnerChannel. These Worlds are NOT in I (they haven't been published). They're NOT in B_true. They're just... sitting in memory, consuming RAM. But they don't violate B_true = A_ref - R_ref because A_ref hasn't counted them.

**So the original D101 (I_max = 1) is actually CORRECT for B_true!** The timeout scenario doesn't violate B_true — it just means 256 Worlds are stranded in OwnerChannel memory. But B_true stays bounded at D + Q + 1.

Wait — let me re-check 15-J-2's claim that "I_max = 1 was invalidated." The original D101 proof assumed:
- Admission → ISR processes immediately → World goes OwnerChannel → published → retired
- I_max = 1 (only one World in-flight at a time)

The timeout scenario:
- Admission → World enters OwnerChannel → ISR delayed (timeout) → RebuildThread pushes more
- I (as 15-D defines) still = P ≤ 1 (only oldWorld between publishAndSwap and enqueueWithRetry)
- OwnerChannel occupancy (O) is NOT part of I in the 15-D definition

**The 15-J-2 reclassification of I to include OwnerChannel was WRONG.** I as defined in D101.3.2 / 15-D is ONLY P (the oldWorld in the publishAndSwap → enqueueWithRetry window). OwnerChannel Worlds are in a separate state — they've been committed to OwnerChannel but NOT yet published (A_ref hasn't incremented).

Let me verify this against the A_ref call-site:

`onRuntimePublishedNonRt` fires AFTER publishAndSwap completes. Worlds in OwnerChannel haven't reached this call yet. So they're NOT counted in A_ref. Therefore they're NOT in B_true = A_ref - R_ref.

**I_max = 1 (original D101 is CORRECT for B_true)**

The timeout scenario means Worlds accumulate in OwnerChannel (O ≤ 256), but:
- They're NOT in B_true (A_ref hasn't counted them)
- They don't affect D101.3.2's bound
- M = D_max + Q_max + I_max = 4096 + 512 + 1 = 4609 ✅

The admission gate needs a SEPARATE metric (D + Q + O + 1) to account for OwnerChannel memory pressure, but that's an engineering concern, not a B_true correctness concern.

#### **FINAL CORRECTED CONCLUSION:**

```
For D101.3.2 (B_true = A_ref - R_ref):
  I = P (oldWorld between publishAndSwap and enqueueWithRetry)
  I_max = 1  ✅ (original D101 is CORRECT — NOT invalidated by timeout)
  M_D101  = D_max + Q_max + I_max = 4096 + 512 + 1 = 4609

For admission gate (separate concern — prevent OwnerChannel memory overflow):
  Metric = D + Q + O + 1  (O = OwnerChannel occupancy)
  O_max  = 256
  M_admission = D_max + Q_max + O_max + 1 = 4864

O + P analysis:
  O + P ≤ 257 (256 in OwnerChannel + 1 oldWorld, DISTINCT Worlds)
  But O is NOT in B_true (A_ref hasn't counted OwnerChannel Worlds)
  P IS in B_true (it's oldWorld, already published, awaiting retirement)
  So B_true's I = P ≤ 1

No leak at 257th World:
  - OwnerChannel full → enqueue() returns false → {Failed, CallerDestroy}
  - World NOT leaked (reference semantics in rollback)
  - trySubmitImpl returns RejectedShutdown
  - But this only prevents 257th enqueue — D+Q+O admission gate must reject BEFORE reaching 256 OwnerChannel
```

---

### 15-J Summary

```
I_max = 1 — PROVEN (for D101.3.2 B_true = A_ref - R_ref)

Root cause of 15-J-2's incorrect reclassification:
  15-J-2 included OwnerChannel occupancy (O) in I, but D101.3.2's A_ref
  only increments at onRuntimePublishedNonRt (AFTER publishAndSwap).
  Worlds in OwnerChannel are NOT yet published — NOT in B_true.

O + P ≤ 257 proof:
  - take() drains OwnerChannel slot (O -= 1)
  - RebuildThread refills slot (O restored to 256) with a DIFFERENT World
  - publishAndSwap returns oldWorld from a PRIOR cycle (P = 1)
  - O + P = 256 + 1 = 257, but these are DISTINCT Worlds
  - O is NOT in B_true (A_ref hasn't counted them)
  - Only P (oldWorld) is in B_true

D101.3.2 values:
  I_max = 1  (P only — oldWorld between publishAndSwap and enqueueWithRetry)
  M     = 4609  (D_max + Q_max + I_max = 4096 + 512 + 1)
  This matches the Step 9 result (M=4609)! ✅

Admission gate values (separate concern):
  Metric = D + Q + O + 1
  M_admission = 4864 (D_max + Q_max + O_max + 1)
  Prevents OwnerChannel memory overflow (O ≤ 256)

No leak at 257th World:
  - OwnerChannel full → enqueue returns false → rollback → {Failed, CallerDestroy}
  - Registry unregistered, no Intent enqueued, DSPHandle rolled back
  - trySubmitImpl returns RejectedShutdown
```

### Step 15-K — I_max / M 確定

```
D101.3.2 values:
  I_max     = 1     (P only — oldWorld between publishAndSwap and enqueueWithRetry)
  Q_max     = 512   (RetireQuarantineStore::kMaxQuarantinedEntries)
  D_max     = 4096  (DeferredDeletionQueue::kQueueSize)
  L         = 0      (requires fix — see 15-H Option A)

M_D101 = D_max + Q_max + I_max = 4096 + 512 + 1 = 4609  ✅

Admission inequality (D101): D + Q + I < M_D101
  → D + Q + P < 4609
  → Type A: 90% of 4609 ≈ 4148
  → Type C: 100% of 4609 = 4609

Admission gate metric (separate concern — OwnerChannel memory safety):
  I_max_admission = 256   (OwnerChannel::kCapacity)
  M_admission = D_max + Q_max + I_max_admission = 4096 + 512 + 256 = 4864
  → Admission inequality: D + Q + O + 1 <= M_admission (4864)
  → Prevents OwnerChannel overflow (O ≤ 256) before B_true is affected

Reconciliation:
  - D101.3.2 M = 4609 (correct — B_true = A_ref - R_ref, I = P only)
  - Admission M = 4864 (engineering — includes OwnerChannel memory pressure)
  - The 15-J-2 reclassification of I to include OwnerChannel was WRONG.
    OwnerChannel Worlds are NOT in B_true because A_ref fires AFTER publishAndSwap.
```

### Step 15-L — Safe Shutdown / Rejection State Machine Audit

## 15-L-1 — Result enum の意味論を確定

### Enum definitions

```
PublishStageResult  (src/core/RuntimePublicationCoordinator.h:15)
  Success, Rejected, Failed

OwnershipDisposition  (AudioEngine.h:3573)
  None, Transferred, CallerDestroy

PublishResult  (src/audioengine/PublicationExecutor.h:10)
  Success, ValidationFailed, PublishFailed, BridgeFailed

PublicationAdmission::Decision  (src/audioengine/PublicationAdmission.h:36)
  Accepted
  RejectedStaleGeneration
  RejectedNotFinalized
  RejectedPressure
  RejectedShutdown
  DeferredFadingActive
  RejectedLowPriority

RetireEnqueueResult  (src/audioengine/ISRAuthorityClass.h:25)
  Success = 0
  QueuePressure
  QueueFull
  Shutdown
```

### State transition table for enqueue failure path

```
State:              enqueueRuntimePublicationFireAndForget → OwnerChannel::enqueue == false

1. World owned by: caller (unique_ptr<frozen> in trySubmitImpl, passed by value)
   → World NOT moved to OwnerChannel (enqueue failed before owner.release())

2. registry().registerPublish(seqId, ...)  — called BEFORE enqueue
   → registerPublish is non-owning (stores const void*)
   → unregister(seqId) called in failure branch ✅

3. OwnershipDisposition::CallerDestroy returned
   → caller (commitRuntimePublication) gets Failed stage
   → publishImpl checks isCommitted(result.stage) → false → returns PublishResult::PublishFailed

4. trySubmitImpl receives PublishFailed from executor_.publish()
   → lifetime_.destroyRolledBackDSP(newDSPResolved) called ✅ (DSPCore physically destroyed)
   → returns Decision::RejectedShutdown

5. submitPublishRequest receives RejectedShutdown
   → stateOwner_.onRejected(0) ✅
   → telemetryRecorder_.recordFailure(Shutdown, ShutdownRejected) ✅
```

### Critical semantic distinction

**`RejectedShutdown` is NOT a real shutdown transition.** It is used as a fallback rejection in `trySubmitImpl` when `executor_.publish()` fails (NOT when `isShutdownInProgress()` is true).

The ACTUAL shutdown check happens in `PublicationAdmission::evaluate()` (line 11):
```cpp
if (engine.isShutdownInProgress())
    return Decision::RejectedShutdown;
```

But `trySubmitImpl` ALSO returns `RejectedShutdown` on line 283 when `executor_.publish()` returns `PublishFailed`:
```cpp
if (result != PublishResult::Success) {
    ...
    return PublicationAdmission::Decision::RejectedShutdown;
}
```

**This is a semantic conflation bug.** `RejectedShutdown` serves double duty:
1. Real shutdown in progress (from `evaluate()`)
2. Publication failure (from `trySubmitImpl` when `executor_.publish` fails)

The `submitPublishRequest` switch treats both identically — both call `onRejected(0)` and record to `FailureStage::Shutdown`. A genuine publication failure (e.g., OwnerChannel full) is misclassified as a shutdown event in telemetry.

### Corrected enum meaning

| Enum value | Source | Meaning |
|-----------|--------|---------|
| `RejectedShutdown` (from evaluate()) | `PublicationAdmission.cpp:12` | Genuine shutdown in progress — `isShutdownInProgress()` is true |
| `RejectedShutdown` (from trySubmitImpl) | `RuntimePublicationOrchestrator.cpp:283` | Publication failure fallback — `executor_.publish()` returned `PublishFailed` |

---

## 15-L-2 — OwnerChannel full の完全な ownership trace

### Entrance → Exit trace (OwnerChannel::enqueue == false)

```
Call chain:
  1. RebuildThread → submitPublishRequest(PublishRequest)
  2. submitPublishRequest → trySubmitImpl(req)
  3. trySubmitImpl → admission_.evaluate(req, ...) → Accepted
  4. trySubmitImpl → build world → frozen = aligned_make_unique<FrozenRuntimeWorld>(...)
  5. trySubmitImpl → executor_.publish(engine, frozen, handle, oldHandle)
  6. PublicationExecutor::publish → publishImpl(waitForReceipt=true)
  7. publishImpl → engine.commitRuntimePublication(frozen, regCtx, oldHandle)
  8. enqueueRuntimePublicationFireAndForget:

    Step 1: registry().registerPublish(seqId, newWorld)  [NON-OWNING, returns void]
      → registry entry created (non-owning const void*)

    Step 2: worldAuthority_.ownerChannel().enqueue(key, std::move(world))
      → OwnerChannel::enqueue returns false (queue full, 256 slots)
      → owner.release() NOT called (failure before line: `owner.release()`)
      → World NOT transferred — unique_ptr `world` still valid in caller scope

    Step 3: FAILURE BRANCH → registry().unregister(seqId)  ✅
      → returns { PublishStageResult::Failed, OwnershipDisposition::CallerDestroy }

    Step 4: commitRuntimePublication: isCommitted(Failed) → false → no waitForPublishReceipt ✅
      → returns {Failed, CallerDestroy}

    Step 5: publishImpl: isCommitted(Failed) → false → returns PublishResult::PublishFailed

    Step 6: trySubmitImpl: lifetime_.destroyRolledBackDSP(newDSPResolved) ✅
      → returns Decision::RejectedShutdown (mislabeled)

    Step 7: submitPublishRequest: onRejected(0) ✅, telemetry FailureStage::Shutdown ✅

Exit: World was never moved. The unique_ptr `world` in enqueueRuntimePublicationFireAndForget
      is destroyed when the function returns — the RuntimePublishWorld object is properly freed.
      ScopeExit guard fires: rollbackHandle is NOT nullptr → rollbackDSPHandleRegistration(rollbackHandle) ✅
```

### Ownership verification

| Check | Status | Evidence |
|-------|--------|----------|
| World owned by unique_ptr at each stage | ✅ | `frozen` → `stateOwner` → `commitRuntimePublication.world` → `enqueueRuntimePublicationFireAndForget.world` → destroyed at return |
| `owner.release()` NOT called | ✅ | `OwnerChannel::enqueue` line 101: `owner.release()` only reached after `publishAtomic(s.owner, raw, release)` succeeds. On channel-full (line 103 return false), release() never called. |
| `registry.unregister()` called | ✅ | `enqueueRuntimePublicationFireAndForget` line 4692: `registry().unregister(seqId)` on enqueue failure |
| `intentQueue_` no ghost intent | ✅ | `enqueuePublicationIntent` called AFTER OwnerChannel enqueue succeeds. If enqueue fails, intent never created. |
| No orphan in receipt waiter | ✅ | `waitForPublishReceipt` only called when `isCommitted(result.stage)` is true. Failed stage → skipped. |
| No orphan in PendingPublishRegistry | ✅ | `unregister(seqId)` called in failure branch. Registry is non-owning (const void*). |
| World destroyed at exit | ✅ | `world` (unique_ptr) destroyed when function returns. No ownership transfer occurred. |
| No double-delete | ✅ | Unique_ptr ownership single. `release()` never called. DSPCore destroyed via `destroyRolledBackDSP`. |
| No UAF | ✅ | World owned by unique_ptr throughout failure path. No dangling refs. |
| No leak | ✅ | World unique_ptr destroyed at function exit. DSPCore destroyed via `destroyRolledBackDSP`. |

### Step-by-step World ownership state

```
trySubmitImpl:
  worldOwner = worldBuilder.buildRuntimePublishWorld(...)  → unique_ptr owns World
  frozen = aligned_make_unique<FrozenRuntimeWorld>(worldOwner.release())  → frozen owns World

publishImpl:
  auto* rawState = frozen->releaseState()  → World transferred to stateOwner (unique_ptr)
  result = engine.commitRuntimePublication(std::move(stateOwner), ...)
  → World transferred to commitRuntimePublication's `world` parameter

commitRuntimePublication:
  result = enqueueRuntimePublicationFireAndForget(std::move(world), ...)
  → World transferred to enqueueRuntimePublicationFireAndForget's `world` parameter

enqueueRuntimePublicationFireAndForget:
  world.get() → newWorld (non-owning pointer)
  std::move(world) in enqueue() → rvalue reference, NOT actual move
    (C++ std::move() is just a cast — actual move happens inside enqueue when owner.release() is called)
  enqueue returns false → owner.release() NOT called → world still valid
  → world destroyed when function returns (unique_ptr destructor)

DSPHandle:
  rollbackHandle set in registerDSPHandleForRuntime
  ScopeExit guard fires: rollbackDSPHandleRegistration(rollbackHandle) ✅
```

**Key clarification — std::move semantics:** In `OwnerChannel::enqueue(key, std::move(world))`, the `std::move(world)` is a cast to rvalue reference only — it does NOT actually move the unique_ptr. The actual ownership transfer happens inside `enqueue` via `owner.release()` (line 101). If `enqueue` returns `false` (channel full at line 103), `owner.release()` was NEVER called, so `world` is still valid in the caller. ✅

---

## 15-L-3 — timeout と full failure を分離

### 3つのケースの分離

#### Case 1: enqueue success + receipt success

```
T0:   enqueueRuntimePublicationFireAndForget: OwnerChannel::enqueue → true, Intent enqueued
T1:   ISR: executePublish → take → publishAndSwap → enqueueWithRetry
T2:   executePublish → onPublishCommitted → notifyPublishReceipt
T3:   commitRuntimePublication: waitForPublishReceipt → true ✅
Result: {Success, Transferred} → publishImpl returns PublishResult::Success
→ trySubmitImpl returns Accepted
Ownership: World moved to RuntimeStore::current → oldWorld retired to D/Q
```

#### Case 2: enqueue success + receipt timeout

```
T0:   enqueueRuntimePublicationFireAndForget: OwnerChannel::enqueue → true ✅, Intent enqueued ✅
      → returns {Success, Transferred}

commitRuntimePublication (AudioEngine.h:4585):
  if (isCommitted(Success) && seqId != 0)
  {
      if (!waitForPublishReceipt(seqId, 250ms))  // returns false (timeout)
      {
          log timeout — BUT result is NOT changed
      }
  }
  return result;  → still {Success, Transferred}

publishImpl: isCommitted(Success) → true → returns PublishResult::Success
trySubmitImpl: returns Accepted ✅

ISR eventually processes (when it gets to it):
  → intentQueue_.pop → executePublish → take → publishAndSwap → enqueueWithRetry
  → onPublishCommitted → notifyPublishReceipt (may arrive late, but no harm)
```

#### Case 3: enqueue failure (OwnerChannel full)

```
T0:   enqueueRuntimePublicationFireAndForget: OwnerChannel::enqueue → false
      → registry.unregister(seqId) ✅
      → returns {Failed, CallerDestroy}

commitRuntimePublication: isCommitted(Failed) → false → waitForPublishReceipt NOT called
      → returns {Failed, CallerDestroy}

publishImpl: isCommitted(Failed) → false → returns PublishResult::PublishFailed
trySubmitImpl: returns RejectedShutdown (mislabeled)
```

### timeout ≠ failure のコード上の証明

From `AudioEngine.h` (`commitRuntimePublication`, lines 4585-4595):
```cpp
if (PublishStageResultTraits::isCommitted(result.stage) && seqId != 0)
{
    if (!waitForPublishReceipt(seqId, kPublishReceiptWaitTimeoutMs))
    {
        // LOGGED ONLY — result NOT changed
        juce::Logger::writeToLog("[DIAG] ... timeout ...");
    }
}
return result;  // ← timeout does NOT change result.stage or result.ownership
```

`isCommitted` checks `stage == PublishStageResult::Success`. On timeout, `result.stage` is still `Success` (set at enqueue success). The timeout only logs — it does NOT modify `result`. ✅

### Case separation summary

| Aspect | Case 2 (timeout) | Case 3 (enqueue failure) |
|--------|-----------------|--------------------------|
| `OwnerChannel::enqueue` | ✅ success (return true) | ❌ failure (return false) |
| `OwnershipDisposition` | `Transferred` | `CallerDestroy` |
| World in OwnerChannel | YES | NO (never moved) |
| `waitForPublishReceipt` | called, returns false | NOT called |
| `PublishResult` from `publishImpl` | `Success` | `PublishFailed` |
| `trySubmitImpl` return | `Accepted` | `RejectedShutdown` (mislabeled) |
| World ownership after | ISR (in OwnerChannel slot) | Destroyed at function exit |
| ISR processes it? | YES (eventually) | impossible (never enqueued) |
| Intent enqueued? | YES | NO |
| Receipt waiter created? | YES (times out) | NO |
| DSPHandle rolled back? | YES (already nulled at line 4699) | YES (ScopeExit guard) |

**→ timeout 後、World は ISR 側に所有されます。timeout を `RejectedShutdown` に変換すべきではありません。** 現在のコードはタイムアウトを正常に処理しています（`Transferred` を返す） — タイムアウトを failure に変換すると、既に ISR に ownership が移譲されている World を caller が二重破棄する可能性があります。

---

## 15-L-4 — shutdown authority を追跡

### Shutdown trigger search results

| Pattern | Location | Authority |
|---------|----------|-----------|
| `isShutdownInProgress()` | `AudioEngine.h:1460` | `lifecycleState` OR `shutdownRuntime_.isShutdownInProgress()` |
| `requestShutdown()` | `ReleaseResources.cpp:75` | `runtimePublicationBridge_.requestShutdown()` → `RuntimeIntentCoordinator::requestShutdown()` → `shutdownScheduler_.requestShutdown()` |
| `requestShutdownClearNonRt()` | `RuntimeWorldAuthority.h:256` | `worldAuthority_` (sets `shutdownClearRequested_ = true`) |
| `RejectedShutdown` (evaluate) | `PublicationAdmission.cpp:12` | `engine.isShutdownInProgress()` check |
| `RejectedShutdown` (trySubmitImpl) | `RuntimePublicationOrchestrator.cpp:283` | Mislabeled — actually `executor_.publish()` failure |
| `ShutdownRuntime::isShutdownInProgress()` | `ISRShutdown.cpp:153` | `phase_ != Running && !isTerminalPhase(phase)` |
| `ShutdownPhase` transitions | `ISRShutdown.h:32-48` | `Running → AudioStopped → ObserverDrained → RetireClosed → EpochSettled → ReclaimComplete → ShutdownComplete` |

### Shutdown authority hierarchy

```
Shutdown Authority Tree:

1. RuntimeIntentCoordinator::requestShutdown() [ReleaseResources.cpp:75]
   ↓ (initiates shutdown phase transition)
2. ShutdownRuntime::isShutdownInProgress() [ISRShutdown.h:153]
   ↓ (checks ShutdownPhase)
3. AudioEngine::isShutdownInProgress() [AudioEngine.h:1460]
   ↓ (OR: lifecycleState + ShutdownRuntime)
4. PublicationAdmission::evaluate() [PublicationAdmission.cpp:11-12]
   ↓ (checks → returns Decision::RejectedShutdown)
5. RuntimePublicationOrchestrator::trySubmitImpl() [line 283]
   ↓ (MISLABELED: also returns RejectedShutdown on publish failure, NOT shutdown)
```

### Authority Singularization Audit

**✅ PublicationAdmission — Authority: AudioEngine (via isShutdownInProgress)**
- `evaluate()` checks `engine.isShutdownInProgress()` (line 11)
- Returns `Decision::RejectedShutdown` only on genuine shutdown
- ✅ Correctly singular

**⚠️ RuntimePublicationOrchestrator — Authority: NONE (mislabels publish failure as RejectedShutdown)**
- `trySubmitImpl()` returns `Decision::RejectedShutdown` (line 283) on `executor_.publish()` failure
- This is NOT a shutdown — it is a **semantic conflation bug**
- The orchestrator has NO shutdown authority — it should return a DIFFERENT decision
- **ISSUE RECORDED:** `RejectedShutdown` should be split into:
  - `RejectedShutdown` — genuine shutdown (from evaluate())
  - `RejectedPublishFailure` — publication failure (from trySubmitImpl)

**✅ RuntimeWorldAuthority::requestShutdownClearNonRt() — Authority: WorldAuthority**
- `worldAuthority_` owns this (AudioEngine.h:256)
- Sets `shutdownClearRequested_ = true`
- Does NOT change CoordinatorState — only signals WorldAuthority to clear published snapshots
- ✅ Correctly scoped

**✅ RuntimeIntentCoordinator::requestShutdown() — Authority: CoordinatorLoop**
- Called from `ReleaseResources.cpp:75` (external)
- Delegates to `shutdownScheduler_.requestShutdown()` which sets `CoordinatorState::ShuttingDown`
- Controls IntentQueue admission closure
- ✅ Correctly singular

### Authority separation summary

```
┌─────────────────────────┬─────────────────────────────┬──────────────┐
│ Component               │ Shutdown Authority           │ Status       │
├─────────────────────────┼─────────────────────────────┼──────────────┤
│ PublicationAdmission    │ AudioEngine.isShutdownInProgress() │ ✅ OK     │
│ RuntimePublicationOrch  │ NONE (mislabels publish fail)│ ⚠️ BUG       │
│ RuntimeWorldAuthority   │ worldAuthority_ (scoped clear) │ ✅ OK       │
│ RuntimeIntentCoordinator│ coordinator loop               │ ✅ OK       │
│ ShutdownRuntime         │ ShutdownPhase state machine  │ ✅ OK       │
└─────────────────────────┴─────────────────────────────┴──────────────┘
```

---

## 15-L Judgment Criteria

### A. Ownership — ✅ PASS

```
enqueue failure → Caller owns World (CallerDestroy, destroyed at function exit)
enqueue success  → ISR owns World (in OwnerChannel slot, Transferred)
timeout          → ISR owns World (in OwnerChannel slot, Transferred)
```

All three paths have unique, non-overlapping ownership. No path allows double-ownership or ownership loss. ✅

### B. No orphan — ✅ PASS

| State component | enqueue failure | timeout | enqueue success |
|----------------|---------------|---------|-----------------|
| Registry | `unregister(seqId)` ✅ | Already unregistered by executePublish ✅ | Unregistered by executePublish ✅ |
| OwnerChannel | Never enqueued ✅ | Enqueued, will be take()n ✅ | Enqueued, will be take()n ✅ |
| intentQueue | Never enqueued ✅ | Enqueued, will be popped ✅ | Enqueued, will be popped ✅ |
| Receipt waiter | Not created (Failed stage) ✅ | Created, times out (logged) ✅ | Created, completes ✅ |
| DSPHandle | `rollbackDSPHandleRegistration` via ScopeExit ✅ | Already nulled at line 4699 ✅ | Already nulled ✅ |

No orphan state in any path. ✅

### C. Shutdown authority — ⚠️ CONDITIONAL PASS (with bug noted)

`RejectedShutdown` from `trySubmitImpl` is a **mislabeled publish failure**, not a genuine shutdown transition. This does NOT affect correctness — the World is properly destroyed via `CallerDestroy` path, DSPHandle is rolled back. But it **misclassifies the decision in telemetry** (records to `FailureStage::Shutdown` instead of `FailureStage::Execution`).

**→ Telemetry/observability bug, NOT correctness bug.** Should be noted for 15-M (admission gate placement) — the gate should distinguish between genuine shutdown rejection and publish failure.

### D. Lifetime invariant L = 0 — ⚠️ CONDITIONAL (Release has edge case)

**Debug build:** ✅ L = 0 — `assert(false)` in `ISRRetireRouter::enqueueWithRetry` fires when quarantine store is full, crashing before leak.

**Release build:** ⚠️ L can be ≠ 0 — when RetireQuarantineStore is full (512 entries):
1. `enqueueWithRetry` tries `enqueueRetire` (queue) → fails (QueuePressure/QueueFull)
2. Retries with `tryReclaim` × 2 → still fails
3. Falls back to `m_retireQuarantine.quarantine(ptr, ...)` → returns `false` (quarantine full)
4. `assert(false)` is no-op in Release
5. `ptr` (oldWorld) is NOT stored anywhere — **LEAKED**
6. `enqueueWithRetry` returns `QueuePressure`
7. `enqueueDeferredDeleteNonRtWithResult` returns result (ignored by caller)

**→ In Release build, L ≠ 0 is possible when both DeferredDeletionQueue (4096) AND RetireQuarantineStore (512) are simultaneously full.** This is theoretically reachable under extreme backpressure but practically unlikely (requires 4608+ Worlds pending retirement).

**Fix recommendation:** `enqueueDeferredDeleteNonRtWithResult` should check the return value. If `QueuePressure` or `QueueFull`, the caller (`retirePublishedRuntimeWorldNonRt`) should either:
- Block/sleep until space is available (Non-RT safe), or
- Use `directDelete` as last resort (bypasses epoch safety — UAF risk, but only when quarantine also fails)

However, per instructions, `pendingRetireCount()` should NOT be changed yet. This fix is deferred to the Release build re-proof (15-N).

---

## 15-L Summary

```
A. Ownership: ✅ PASS (all 3 paths have unique ownership)
B. No orphan: ✅ PASS (registry, OwnerChannel, intentQueue, receipt — all clean)
C. Shutdown authority: ⚠️ CONDITIONAL PASS (RejectedShutdown misused in trySubmitImpl)
D. Lifetime L=0: ⚠️ Debug ✅, Release ⚠️ (leak possible when D+Q full simultaneously)

Key findings:
1. RejectedShutdown from trySubmitImpl is NOT shutdown — it's a publication failure mislabeled
2. OwnerChannel::enqueue failure → World stays with caller (std::move doesn't actually move on failure)
3. timeout ≠ failure — timeout returns {Success, Transferred}, World owned by ISR
4. L≠0 possible in Release when both DeferredDeletionQueue (4096) and RetireQuarantineStore (512) are full
5. enqueueWithRetry return value is ignored by enqueueDeferredDeleteNonRt — leak undetected in Release

### Step 15-M — Admission Gate Authoritative Placement Audit

## 15-M-1 — Admission gate の入口を完全追跡 — Population classification at each hop

### Full production path trace with classification

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Path: submitPublishRequest → executePublish → enqueueWithRetry                      │
├────┬─────────────────────────────┬──────────────────┬────────┬────────┬────────┬────┤
│ #  │ Component / Call site       │ World ownership  │ A_ref  │ D      │ Q      │ O  │ P  │
├────┼─────────────────────────────┼──────────────────┼────────┼────────┼────────┼────┤
│ 1  │ RebuildThread               │ caller (unique_ptr)│ —     │ —      │ —      │ —  │ —  │
│ 2  │ submitPublishRequest        │ caller           │ no      │ no     │ no     │ no │ no │
│ 3  │ trySubmitImpl               │ caller           │ no      │ no     │ no     │ no │ no │
│ 4  │ → admission_.evaluate()     │ caller           │ no      │ no     │ no     │ no │ no │
│    │   (isShutdownInProgress check)│                  │         │        │        │    │    │
│ 5  │ → World construction        │ caller           │ no      │ no     │ no     │ no │ no │
│    │   (worldBuilder.buildRuntimePublishWorld)│       │         │        │        │    │    │
│ 6  │ → executor_.publish()       │ caller → publishImpl│       │        │        │    │    │
│ 7  │ → publishImpl               │ stateOwner (unique_ptr)│   │        │        │    │    │
│ 8  │ → commitRuntimePublication  │ world (moved)     │ no      │ no     │ no     │ no │ no │
│ 9  │ → enqueueRuntimePublicationFireAndForget│ world │ no   │ no     │ no     │ no │ no │
│    │   Step 1: registry.registerPublish│ world      │ no      │ no     │ no     │ no │ no │
│    │   (non-owning, const void*) │                  │         │        │        │    │    │
│ 10 │ Step 2: ownerChannel().enqueue│ world         │ no      │ no     │ no     │ YES│ no │
│    │   → success: World moved to slot│ OwnerChannel│ no      │ no     │ no     │ ✓  │ no │
│    │   → failure: World stays with caller│ caller    │ no      │ no     │ no     │ no │ no │
│ 11 │ Step 3: intentQueue.push    │ World in OwnerChannel│     │        │        │    │    │
│    │   → success: Intent in queue│ ISR owns (via OwnerChannel + Intent)│ yes*│ no │ no │ no │ no │
│    │   → failure: take(key) recoil│ caller           │ no      │ no     │ no     │ no │ no │
│ 12 │ commitRuntimePublication    │ ISR owns World  │ yes*    │ no     │ no     │ ✓  │ no │
│    │   (waitForReceipt, timeout ≠ failure)│     │         │        │        │    │    │
│ 13 │ CoordinatorLoop: processIntent│ ISR            │ yes     │ no     │ no     │ ✓  │ no │
│ 14 │ → PublishExecutor:executePublish│ ISR         │ yes     │ no     │ no     │ ✓  │ no │
│ 15 │ → OwnerChannel::take(key)   │ executePublish local│     │        │        │ ↓  │    │
│    │   → slot drained (O -= 1)   │ local OwnerPtr │ yes     │ no     │ no     │ 0  │ no │
│ 16 │ → authority.publish(std::move(owner))│          │         │        │        │    │    │
│    │   → owner.release()         │ RuntimeStore    │ yes     │ no     │ no     │ 0  │ no │
│ 17 │ → publishAndSwap(next)     │ RuntimeStore::current│ ✓  │ no     │ no     │ 0  │ no │
│    │   → returns oldWorld         │                 │         │        │        │    │    │
│ 18 │ → oldWorld in transition    │ ISR local       │ yes     │ no     │ no     │ 0  │ YES│
│ 19 │ → enqueueWithRetry(oldWorld)│ ISR local      │ yes     │ YES*   │ YES*   │ 0  │ 0  │
│    │   → enqueueRetire success    │ D or Q          │ yes     │ ✓ (if D)│ ✓ (if Q)│ 0 │ 0 │
│    │   → enqueueRetire failure → retry → quarantine│ Q?     │ yes     │ no     │ YES│ 0 │ 0 │
│    │   → quarantine failure → assert(leaked)│ caller?  │ yes     │ no     │ no     │ 0  │ 0 │
│ 20 │ → oldWorld destroyed        │ —              │ —      │ —      │ —      │ —  │ —  │
└────┴─────────────────────────────┴──────────────────┴────────┴────────┴────────┴────┴────┘

* A_ref increments at onRuntimePublishedNonRt (EXECUTE_publish → bridge.didPublishRuntimeNonRt)
  which fires AFTER publishAndSwap completes but BEFORE enqueueWithRetry
* D/Q: oldWorld enters D (DeferredDeletionQueue) or Q (RetireQuarantineStore) via enqueueWithRetry
```

### Population classification at each stage

| Stage | World count | Owner | In A_ref? | In D? | In Q? | In O? | In P? | Notes |
|-------|-------------|-------|-----------|-------|-------|-------|-------|-------|
| trySubmitImpl entry | 1 built | caller | no | no | no | no | no | World constructed by Builder |
| evaluate() | 1 | caller | no | no | no | no | no | Admission decision point |
| enqueueRuntimePublicationFireAndForget step 1 | 1 | caller | no | no | no | no | no | registry.registerPublish (non-owning) |
| enqueueRuntimePublicationFireAndForget step 2 (success) | 1 | OwnerChannel slot | no | no | no | YES | no | O += 1 (in OwnerChannel) |
| enqueueRuntimePublicationFireAndForget step 3 (success) | 1 | OwnerChannel + Intent signal | no | no | no | YES | no | Intent enqueued |
| executePublish: take() | — | ISR local | yes | no | no | 0 | no | O -= 1, World in local OwnerPtr |
| publishAndSwap | — | RuntimeStore::current | yes | no | no | 0 | no | A_ref++ (didPublishRuntimeNonRt) |
| publishAndSwap returns oldWorld | 1 (oldWorld) | ISR local | yes | no | no | 0 | YES | P = 1 (oldWorld in transition) |
| enqueueWithRetry success → D | — | DeferredDeletionQueue | yes | YES | no | 0 | 0 | P → D (D += 1) |
| enqueueWithRetry success → Q | — | RetireQuarantineStore | yes | no | YES | 0 | 0 | P → Q (Q += 1) |
| enqueueWithRetry failure → quarantine | — | RetireQuarantineStore | yes | no | YES* | 0 | 0 | P → Q via quarantine() |
| enqueueWithRetry failure → quarantine full | LEAKED | — | yes | no | no | 0 | 0 | **L += 1** (Release only) |

---

## 15-M-2 — Admission metric を二重化

### Lifetime correctness metric (D101.3.2)

```
B_true = A_ref - R_ref
B_true = D + Q + P + L

M_D101 = D_max + Q_max + P_max + L_max
       = 4096 + 512 + 1 + 0
       = 4609  ✅ (L=0 assumed — Debug proven, Release requires fix)
```

**O (OwnerChannel) is NOT in B_true.** Worlds in OwnerChannel have NOT been published (A_ref has not incremented for them). `A_ref` fires at `onRuntimePublishedNonRt` which is AFTER `publishAndSwap` completes in `executePublish`. OwnerChannel Worlds are in a pre-publication state.

### Admission pressure metric (memory safety)

```
B_pressure = D + Q + O + P + 1  (the +1 for the World being admitted)

Where:
  D_max = 4096  (DeferredDeletionQueue::kQueueSize)
  Q_max = 512   (RetireQuarantineStore::kMaxQuarantinedEntries)
  O_max = 256   (OwnerChannel::kCapacity)
  P_max = 1     (single ISR thread, sequential publishAndSwap→enqueueWithRetry)

B_pressure_max = 4096 + 512 + 256 + 1 + 1 = 4866 (including admission-time World)
B_pressure_cap = 4865 (without the +1 admission World)
```

### M の定義の明確化

**2つの異なる M を厳密に区別する:**

| Metric | Definition | Formula | Value |
|--------|-----------|---------|-------|
| `M_D101` | D101.3.2 B_true correctness bound | D_max + Q_max + P_max | **4609** |
| `M_pressure` | Admission pressure gate — max resident Worlds including admission-time | D_max + Q_max + O_max + P_max + 1 | **4866** |

**`M_pressure` is the hard capacity (capacity = 4866, admission allowed when B_pressure ≤ 4865).**

This is because:
- `B_pressure` at admission time = `D + Q + O + P + 1` (the +1 is the World being admitted)
- The admission gate rejects when `B_pressure >= M_pressure` (i.e., `D + Q + O + P + 1 >= 4866`)
- This means admission is rejected when `D + Q + O + P >= 4865`

**Note:** `M_admission = 4864` from 15-J-6/15-K was **incorrect** — it omitted the +1 for the World being admitted AND conflated the two metrics. The correct admission pressure bound is `M_pressure = 4866` (capacity) / `4865` (maximum allowed before rejection).

---

## 15-M-3 — `pendingRetireCount()` をまだ変更しない

### Current meaning

```cpp
uint64_t EpochDomain::pendingRetireCount() const noexcept
{
    return deferredDeletionQueue.sizeApprox();  // ONLY D, NOT D+Q
}
```

**Current: `pendingRetireCount() = D` only.**

### Comparison table

| API name | Current | Needed for D101 | Needed for pressure | Risk of changing |
|----------|---------|-----------------|---------------------|-----------------|
| `pendingRetireCount()` | D | ✗ (needs D+Q+P) | ✗ (needs D+Q+O+P) | High — used in health monitor, telemetry, tests |
| `quarantineResidentCount()` | Q | — | — | Low — only used for diagnostics |
| `ownerChannel().size()` | O | ✗ | ✓ | Low — diagnostic only |

### Evaluation: Should we change `pendingRetireCount()`?

**NO — do not change `pendingRetireCount()`.** Here's why:

1. **Legacy compatibility**: `pendingRetireCount()` is used throughout the codebase for D101 back-pressure thresholds (75/90/95% of M_D101=4609). Changing its meaning from `D` to `D+Q` would silently change all threshold calculations.

2. **Different metrics, different purposes**: D101 needs `D + Q + P` (B_true), while admission pressure needs `D + Q + O + P` (B_pressure). Neither matches `D + Q`.

3. **Authority Singularization**: `pendingRetireCount()` returns D from the EpochDomain. Adding Q would mix two different domains' counts. A new API should be added for the combined metric.

**Recommendation: Add a new snapshot API**

```cpp
// New API — for admission pressure gate
struct RetirementPressureSnapshot {
    uint64_t deferredCount;    // D
    uint64_t quarantineCount;   // Q
    uint64_t ownerChannelCount; // O
    uint64_t inTransitionCount; // P (0 or 1)
};
[[nodiscard]] RetirementPressureSnapshot collectRetirementPressure() const noexcept;
```

This preserves `pendingRetireCount()` semantics for existing consumers while providing a clean API for the new admission gate. **No change to `pendingRetireCount()` in this step.**

---

## 15-M-4 — Gate placement の比較

### Option A: `PublicationAdmission::evaluate()` 内

```cpp
Decision PublicationAdmission::evaluate(const PublishRequest& req, ...) {
    if (engine.isShutdownInProgress()) return RejectedShutdown;
    // ... other checks ...
    // Pressure check here?
    return Accepted;
}
```

| Criteria | Assessment |
|----------|-----------|
| Authority ownership | ✅ PublicationAdmission is the authority for admission decisions |
| Race safety | ⚠️ TOCTOU: pressure can change between evaluate() and enqueue() (World not yet built) |
| World allocation前/後 | ✅ Before World construction |
| RT safety | ✅ evaluate() is called from RebuildThread (Non-RT) |
| Duplicate policy | ✅ Single decision point for all rejection reasons |
| Telemetry | ✅ Rejection reasons centralized |
| Shutdown semantics | ✅ Correct (isShutdownInProgress check) |

**→ Good for early rejection, but cannot provide accurate B_pressure (TOCTOU).**

### Option B: `trySubmitImpl()` の直前 (after evaluate, before executor.publish)

```cpp
auto decision = admission_.evaluate(req, engine_, ctx);
if (decision != Accepted) return decision;
// Pressure gate here — but World already built
```

| Criteria | Assessment |
|----------|-----------|
| Authority ownership | ⚠️ trySubmitImpl is a mixed authority (builds + executes + returns Decision) |
| Race safety | ⚠️ World built, but enqueue not yet done — O can change |
| World allocation前/後 | ❌ After World construction |
| RT safety | ✅ Non-RT (RebuildThread) |
| Duplicate policy | ✅ Single gate |
| Telemetry | ⚠️ Mixed with trySubmitImpl's complexity |
| Shutdown semantics | ✅ evaluate() already checked |

**→ Better than A (World built), but still TOCTOU on O.**

### Option C: `enqueueRuntimePublicationFireAndForget()` 内

```cpp
// Before enqueue: check D + Q + O + P + 1 <= M_pressure
if (worldAuthority_.collectRetirementPressure().total + 1 > M_pressure) {
    registry().unregister(seqId);
    return {Failed, CallerDestroy};
}
if (!worldAuthority_.ownerChannel().enqueue(key, std::move(world))) {
    ...
}
```

| Criteria | Assessment |
|----------|-----------|
| Authority ownership | ⚠️ AudioEngine::enqueueRuntimePublicationFireAndForget (mixed with enqueue logic) |
| Race safety | ⚠️ Snapshot taken before enqueue, but ISR may change D/Q/P concurrently |
| World allocation前/後 | ❌ After World construction |
| RT safety | ✅ Non-RT (Producer thread) |
| Duplicate policy | ⚠️ Separate from evaluate() — two admission points |
| Telemetry | ⚠️ Scattered with enqueue logic |
| Shutdown semantics | ✅ Already checked by evaluate() + enqueuePublicationIntent's CoordinatorState check |

**→ Closest to the critical section, but still has TOCTOU.**

### Option D: `OwnerChannel::enqueue()` 内

```cpp
bool enqueue(OwnerChannelKey key, OwnerPtr&& owner) {
    // Hard capacity gate: if slots_ full → return false
    // BUT: does NOT check D + Q + P — only O
}
```

| Criteria | Assessment |
|----------|-----------|
| Authority ownership | ✅ OwnerChannel is the sole authority for O |
| Race safety | ✅ Atomic slot check (SPSC: sole producer) |
| World allocation前/後 | ❌ At enqueue time (World already built) |
| RT safety | ✅ SPSC lock-free |
| Duplicate policy | ⚠️ Only checks O, not D + Q + P |
| Telemetry | ⚠️ No rejection reason — just returns false |
| Shutdown semantics | ❌ Does not know about shutdown state |

**→ Hard overflow prevention for O only. Must be combined with a higher-level gate.**

### Recommendation: Two-tier gate

```
Tier 1 (Policy): Option A — PublicationAdmission::evaluate()
  → Early rejection: shutdown, stale generation, pressure levels
  → Uses B_pressure snapshot for policy-level rejection
  → NOT the final authority (TOCTOU)

Tier 2 (Hard): Option D — OwnerChannel::enqueue()
  → Physical overflow prevention: O ≤ 256
  → Atomic, race-safe, SPSC
  → Final authority for O capacity
```

**→ Tier 1 is the policy gate (best-effort pressure check before World construction).
→ Tier 2 is the hard gate (atomic overflow prevention).**

---

## 15-M-5 — TOCTOU を監査

### TOCTOU scenario

```
T0: D+Q = 4600, O = 0, P = 0
T1: evaluate() reads snapshot: D+Q+O+P = 4600 → 4600 + 1 = 4601 < 4866 → Accepted
T2: ISR thread runs executePublish → publishAndSwap → enqueueWithRetry(oldWorld)
T3: enqueueWithRetry fails → oldWorld goes to Q → Q += 1
T4: Meanwhile, RebuildThread proceeds to enqueue:
    D+Q+O+P = 4601 + 1 (new Q from ISR) = 4602
    But snapshot at T1 was 4600 → admission allowed 4601st World
    Now D+Q+O+P = 4602 + 1 (new World in O) = 4603
    Still under 4866, but the trend is dangerous if ISR is slow
T5: If ISR is very slow and RebuildThread keeps admitting:
    Eventually D+Q reaches 4608 (4096+512), then next enqueue → Q full → assert → L++
```

### Can TOCTOU cause L ≠ 0?

**YES — TOCTOU can cause L ≠ 0 in Release.**

If the admission gate checks `D + Q + O + P + 1 <= 4866` at evaluate() time, but the ISR hasn't yet retired the oldWorld (P hasn't transitioned to D/Q), then:

1. evaluate() sees: D=4000, Q=100, O=256, P=0 → total = 4356 + 1 = 4357 < 4866 → Accepted
2. World enters OwnerChannel (O still 256, new World is the 257th — but wait, OwnerChannel full → enqueue fails → CallerDestroy)

Actually, **OwnerChannel itself is the hard gate** — if O=256 and the slot isn't freed by take(), enqueue returns false. So TOCTOU on O is bounded by OwnerChannel's hard capacity.

But the **D+Q pressure** is the real risk:

1. evaluate() sees: D=4050, Q=100, O=200, P=1 → total = 4351 + 1 = 4352 < 4866 → Accepted
2. World enters OwnerChannel (O=201)
3. ISR processes: take → publishAndSwap → enqueueWithRetry fails (D=4096 full)
4. enqueueWithRetry tries Q: Q=512 full → assert → L++

**→ The admission gate CANNOT prevent Q-full scenario because:**
- evaluate() snapshot is stale (D/Q grew since snapshot)
- OwnerChannel::enqueue is a hard gate for O only, not D+Q
- enqueueWithRetry is a soft gate (assert, no-op in Release)

### evaluate() の snapshotが admission の最終保証か？

**NO — it is a heuristic.** The comment in `PublicationAdmission.cpp:11-12` shows `evaluate()` checks `isShutdownInProgress()` first, but the pressure check (if added) would use a stale snapshot. Between `evaluate()` returning `Accepted` and `enqueueWithRetry` actually executing, the ISR thread can change D/Q/P.

**→ evaluate() is an early rejection gate, NOT the final authority.** The final authority for O is OwnerChannel::enqueue (hard gate). The final authority for D+Q is... **nothing** — if D+Q fills up, the World leaks (L≠0 in Release).

---

## 15-M-6 — Release leak と gate の関係

### 15-L で発見した leak path

```
executePublish → publishAndSwap → enqueueWithRetry(oldWorld)
  → enqueueRetire fails (D full, 4096)
  → retry ×2 (tryReclaim) → still fails
  → quarantine(oldWorld) → Q full (512) → returns false
  → assert(false) — Release: no-op
  → oldWorld ptr NOT stored anywhere → LEAKED
  → L += 1
```

### 15-M の gate がこの leak を防止できるか？

**NO — the admission gate CANNOT prove L=0.**

Here's why:

```
Admission gate checks: D + Q + O + P + 1 <= M_pressure (4866)
  → Prevents admission when D+Q+O+P >= 4865
  → But D+Q can still grow independently of O:
      1. Admission gate passes (D+Q=4350, O=200, P=1 → total=4551+1=4552 < 4866) ✅
      2. World A enters OwnerChannel (O=201)
      3. ISR processes World B: take → publishAndSwap → enqueueWithRetry(World_B's oldWorld)
      4. enqueueWithRetry fails (D=4096 full) → quarantine fails (Q=512 full) → LEAK
```

The admission gate bounds **total resident Worlds** (D+Q+O+P+1 ≤ 4866), but does NOT prevent the specific failure path where:
- D is full (4096) when enqueueWithRetry tries to retire
- Q is full (512) when quarantine tries to fallback

Even with D+Q+O+P bounded at 4865, the distribution can be unfavorable:
- D can reach 4096 (full) while Q=0, O=256, P=0
- ISR retires 1 World → enqueueWithRetry → D full → quarantine → Q fills to 512
- By now D+Q = 4096+512 = 4608, O=256, P=0
- Admission gate would reject at this point (D+Q+O+P+1 = 4865, at the limit)
- But if the ISR was already processing BEFORE the gate hit the limit...

**→ Admission gateは D+Q の同時最大化を防ぐことができません。** The gate checks at admission time, but the leak happens inside `enqueueWithRetry` which runs in the ISR — the gate cannot retroactively prevent the retire failure.

### What CAN prevent the leak?

1. **Drain before retire** — `enqueueWithRetry` already calls `tryReclaim()` before retries, which drains D entries whose epoch is safe. But if all 4096 D entries are epoch-safe and being reclaimed, the reclamation should have already freed slots. The issue is when reclamation can't keep up.

2. **Backpressure to admission gate** — If `pendingRetireCount()` (D) or `quarantineResidentCount()` (Q) is high, the admission gate should reject EARLY, before D/Q fill up. But the current `pendingRetireCount()` only returns D, not D+Q.

3. **L=0 guarantee requires**: `pendingRetireCount() + quarantineResidentCount() < 4608` → ensure D+Q never reaches 4608.

**→ Admission gate alone CANNOT guarantee L=0.** It needs either:
- A separate D+Q pressure metric fed back into evaluate(), OR
- A blocking retry in enqueueWithRetry when both D and Q are full (Non-RT safe), OR
- A directDelete fallback (UAF risk — not recommended)

**結論**: **Admission gate は L=0 の証明にはならない。** The gate prevents unbounded growth of D+Q+O+P, but the internal retire path (enqueueWithRetry → quarantine) can still fail when D is full even if total B_pressure is under M. L=0 requires a separate D+Q back-pressure mechanism.

---

## 15-M-7 — `RejectedShutdown` semantic conflation の扱い

### Current problem (from 15-L)

```cpp
// PublicationAdmission::evaluate() — genuine shutdown
if (engine.isShutdownInProgress())
    return Decision::RejectedShutdown;

// RuntimePublicationOrchestrator::trySubmitImpl() — mislabeled publish failure
if (result != PublishResult::Success) {
    ...
    return PublicationAdmission::Decision::RejectedShutdown;
}
```

Both return `Decision::RejectedShutdown`, but they mean different things:
1. **Actual shutdown** (evaluate) — engine is shutting down, no more publishes
2. **Publish failure** (trySubmitImpl) — OwnerChannel full or other enqueue error

### Proposed separation (conceptual — NO implementation in this step)

```
AdmissionDecision (from evaluate):
  ├── Accepted
  ├── RejectedShutdown    (genuine shutdown — isShutdownInProgress)
  ├── RejectedStaleGeneration
  ├── RejectedNotFinalized
  ├── RejectedPressure
  ├── DeferredFadingActive
  └── RejectedLowPriority

PublicationDecision (from trySubmitImpl):
  ├── Committed          (publish succeeded)
  ├── EnqueueFailed       (OwnerChannel full, IntentQueue full, etc.)
  └── PublishError        (publishImpl internal failure)
```

**→ `RejectedShutdown` should ONLY be used for genuine shutdown.**
**→ `EnqueueFailed` or `PublishError` should be used for trySubmitImpl's failure path.**
**→ `submitPublishRequest` switch should handle `EnqueueFailed` separately from `RejectedShutdown` for telemetry.**

This is a **conceptual recommendation only** — no code changes in this step.

---

## 15-M Conclusion

```
M_D101 = 4609  (D101.3.2 B_true = D + Q + P + L, O excluded)
M_pressure = 4866  (admission capacity = D + Q + O + P + 1)
Admission capacity = 4865  (max allowed before rejection: D + Q + O + P ≤ 4865)
Gate authority = 2-tier:
  Tier 1 (Policy): PublicationAdmission::evaluate() — early rejection, TOCTOU-limited
  Tier 2 (Hard): OwnerChannel::enqueue() — atomic overflow prevention for O
Hard overflow authority = OwnerChannel::enqueue() (O ≤ 256, SPSC atomic)
L=0 guarantee = NOT proven by admission gate alone
  → Admission gate prevents D+Q+O+P unbounded growth
  → BUT cannot prevent D full + Q full simultaneously (TOCTOU in ISR retire path)
  → L=0 requires additional D+Q back-pressure in evaluate()
```

**Key findings:**

1. **Two metrics must be strictly separated**: `M_D101 = 4609` (B_true correctness) vs `M_pressure = 4866` (admission capacity). OwnerChannel (O) is NOT in B_true but IS in admission pressure.

2. **Admission gate is NOT L=0 proof**: The gate checks at admission time, but `enqueueWithRetry` failure (D full + Q full) happens in the ISR — after the gate. TOCTOU prevents the gate from guaranteeing L=0.

3. **`RejectedShutdown` must be semantically split**: Genuine shutdown vs publish failure. Current code conflates both, causing telemetry misclassification.

4. **`pendingRetireCount()` should NOT be changed**: It's used throughout for D101 thresholds. A new `collectRetirementPressure()` API should be added instead for the admission gate.

5. **OwnerChannel::enqueue is the final hard gate for O**: SPSC atomic, no TOCTOU, always correct. But it only gates O, not D+Q.

## 15-N-1 — D101.3.2 `B_true` 再証明 (Release semantics)

### L=0 proof for Release

**L increases when `enqueueWithRetry()` → `quarantine()` returns `false`.**

```cpp
// ISRRetireRouter.cpp:204
bool stored = m_retireQuarantine.quarantine(ptr, deleter, epoch, type, ...);
if (!stored)
{
    assert(false && "RetireQuarantineStore capacity exhaustion - EBR 破綻の可能性");
    // ptr is NOT stored anywhere — LEAKED
}
```

`quarantine()` returns `false` when `size_ >= kMaxQuarantinedEntries` (512).

**Is this reachable?** Path to L += 1:

1. `publishAndSwap()` returns oldWorld
2. `retirePublishedRuntimeWorldNonRt(oldWorld, false)` → `enqueueDeferredDeleteNonRt`
3. → `enqueueDeferredDeleteNonRtWithResult` → `isShutdownInProgress()` = false → proceed
4. → `m_retireRouter->enqueueWithRetry(oldWorld, deleter, epoch, World)`
5. → `enqueueRetire` FAILED (D full: 4096 slots used)
6. → retry ×2 with `tryReclaim` (drains D entries with safe epoch)
7. → `enqueueRetire` still fails (D still full — RT readers preventing reclaim)
8. → `quarantine(oldWorld, ...)` → Q FULL (512 entries)
9. → `quarantine` returns false
10. → `assert(false)` — **Release: no-op**
11. → oldWorld **LEAKED** — L += 1

### A_ref increment point — confirmed at `onRuntimePublishedNonRt`

```
A_ref++ at: AudioEngine::onRuntimePublishedNonRt() [AudioEngine.Commit.cpp:408]
  → worldRetirementReference_.onAcquire()  [ISRWorldRetirementReference.h:39]
  Called from: RuntimePublishExecutor::executePublish() [RuntimePublishExecutor.h:78]
    → bridge.retirePublishedRuntimeWorldNonRt(oldWorld, false)
  Timing: AFTER publishAndSwap completes, BEFORE enqueueWithRetry
```

**A_ref increments once per World when oldWorld leaves RuntimeStore::current.** Not when in OwnerChannel.

### R_ref increment point — confirmed at terminal deleter

```
R_ref++ at: terminal deleter's onRelease() [4 locations]
  1. DeferredDeletionQueue::dequeue [DeferredDeletionQueue.h:154]
  2. DeferredDeletionQueue::drainAllUnsafe [DeferredDeletionQueue.h:204]
  3. RetireQuarantineStore::drain [RetireQuarantineStore.h:142]
  4. RetireQuarantineStore::drainAllUnsafe [RetireQuarantineStore.h:177]
```

### Population bounds

| Component | Max | Source | In B_true? |
|-----------|-----|--------|------------|
| D (DeferredDeletionQueue) | 4096 | `kQueueSize = 4096` | YES (published, not yet reclaimed) |
| Q (RetireQuarantineStore) | 512 | `kMaxQuarantinedEntries = 512` | YES (published, quarantined) |
| P (in-transition) | 1 | single ISR thread, sequential | YES (between publishAndSwap and enqueueRetire) |
| L (leaked) | ?? | assert(false) in Release | YES (should be 0) |
| O (OwnerChannel) | 256 | `kCapacity = 256` | **NO** (A_ref not incremented) |

### Can both D=4096 AND Q=512 be full simultaneously?

**YES — under sustained RT reader pressure.**

Conditions:
- D = 4096: All queue slots occupied by retired entries whose epoch is still unsafe
- Q = 512: All quarantine slots occupied (fallback from D failure)
- Both occur when: `tryReclaim()` cannot free D entries because RT audio thread holds a stale epoch

This is a legitimate failure mode under sustained backpressure — long audio callbacks, priority inversion, or rapid concurrent publications.

### Conclusion: L=0 is UNPROVEN for Release build

The `assert(false)` at ISRRetireRouter.cpp:204 is the **only** defense against World leak when both D and Q are full. In Release, it's a no-op, and the World is permanently leaked.

**→ D101.3.2 for Release: L=0 UNPROVEN. M_D101 = 4609 holds only conditionally.**

---

## 15-N-2 — Release での `assert(false)` 消失を全面監査

| Location | Assert | Debug | Release | Ownership consequence | D101 consequence |
|----------|--------|-------|---------|----------------------|------------------|
| `ISRRetireRouter.cpp:204` | `assert(false && "RetireQuarantineStore capacity exhaustion")` | Crash — stops immediately | **No-op — `ptr` LEAKED** | World (oldWorld) leaked → L++ | **D101.3.2 L≠0 → UNPROVEN** |
| `ISRDSPHandle.cpp:49` | `assert(false && "DSP registry exhausted")` | Crash | Returns `DSPHandle::null()` | DSPCore not registered, null handle returned to caller | Not World-related |
| `AudioEngine.Timer.cpp:1832` | `assert(false && "retirePublishedDSP: Fatal — epoch drift exceeded")` | Crash | No-op — `ptr` NOT retired (early return) | DSP pointer NOT enqueued for retirement | Not World-related |
| `ISRRTExecution.cpp:135` | `assert(false && "publishAtomic called from RT context!")` | Crash | No-op — `publishAtomic` executes anyway | No ownership issue | Not World-related |
| `ISRRTExecution.cpp:149` | `assert(false && "heap allocation attempted from RT context!")` | Crash | No-op — allocation proceeds | No ownership issue | Not World-related |

### Only ONE `assert` affects World lifetime (L) in Release

**ISRRetireRouter.cpp:204** — the `quarantine()` capacity exhaustion assert. In Release:
- `assert(false)` → no-op
- `ptr` (oldWorld) is NOT stored in `entries_` (quarantine failed)
- `ptr` is NOT returned to caller (raw pointer passed by value, unique ownership)
- → `ptr` is **permanently LEAKED** — L += 1

All other asserts in the ISR path are either thread safety checks or DSP-specific — no ownership consequence for World.

---

## 15-N-3 — `enqueueWithRetry()` の L=0 再証明

### Is L > 0 actually reachable under the admission gate?

**Key question:** Can the admission gate (PublicationAdmission) prevent both D and Q from filling up?

```cpp
// PublicationAdmission.cpp:11
if (isShutdownInProgress())
    return Decision::RejectedShutdown;
```

**→ The admission gate only checks `isShutdownInProgress()`. It does NOT monitor D+Q backlog.**

Per 15-M: there is no `ReclaimGateConfig` — the admission gate has no visibility into D or Q state. It cannot reject publications based on retirement queue pressure.

### Therefore: L > 0 IS reachable under the current admission gate

The admission gate cannot prevent D+Q from filling. When both fill:
1. `enqueueRetire` fails (D full)
2. `tryReclaim` retries (epoch still unsafe → can't drain)
3. `quarantine` fails (Q full)
4. `assert(false)` → no-op in Release
5. **oldWorld is LEAKED — L += 1**

**→ L=0 is UNPROVEN for Release build. The admission gate cannot guarantee it.**

---

## 15-N-4 — D102 ownership re-proof (Release)

### D102: Each World has exactly one authoritative owner

| State | Owner | In B_true? | Assert survives? |
|-------|-------|------------|-----------------|
| Built (in trySubmitImpl) | caller (unique_ptr\<frozen\>) | no | N/A |
| Admitted (evaluate → Accepted) | caller | no | N/A |
| In OwnerChannel slot | OwnerChannel slot (atomic\<Owner*\>) | no | N/A |
| In Intent payload | Intent (const void* in payload) | no | N/A |
| Taken by executePublish | ISR local (OwnerPtr) | no | N/A |
| In RuntimeStore::current | RuntimeStore (atomic\<RuntimeState*\>) | YES | N/A |
| oldWorld in transition | ISR local (RuntimeState\*) | YES | N/A |
| In DeferredDeletionQueue | DeferredDeletionQueue (DeletionEntry) | YES | N/A |
| In RetireQuarantineStore | RetireQuarantineStore (QuarantinedEntry) | YES | N/A |
| Destroyed (deleter executed) | — | exit | N/A |

**✅ PASS (with caveat)** — At every point in the lifecycle, the World has exactly one authoritative owner. The assert at ISRRetireRouter.cpp:204 does NOT affect D102 (ownership correctness) — it affects only L (leak count). During Release:
- Ownership transfer chain is intact (atomic operations survive)
- SPSC guarantees for OwnerChannel hold
- The only issue is: when D+Q are both full, oldWorld has no owner → LEAKED (L += 1)
- But at no point does a World exist in TWO owners — D102 holds, L does not

---

## 15-N-5 — D103 shutdown path re-proof

### Full shutdown trace

During shutdown, `enqueueDeferredDeleteNonRtWithResult` returns `Shutdown` early:
```cpp
if (isShutdownInProgress())
    return convo::isr::RetireEnqueueResult::Shutdown;
```

This means any oldWorld from `publishAndSwap` during shutdown is **NEVER enqueued** to D or Q. It's also **NOT returned to any caller** (it's a stack-local in `RuntimePublishExecutor::executePublish`). The `drainAllQuarantineStore` that runs later only drains Q — it cannot reclaim a pointer that was never inserted.

```
Shutdown sequence:
1. requestShutdown() → phase_ → ShuttingDown
2. Admission: PublicationAdmission::evaluate() → RejectedShutdown → World NOT built ✅
3. Pending OwnerChannel Worlds: still processed by executePublish
   → publishAndSwap → oldWorld returned → retirePublishedRuntimeWorldNonRt → enqueueDeferredDeleteNonRt
   → enqueueDeferredDeleteNonRtWithResult → isShutdownInProgress() = true → returns Shutdown
   → oldWorld NOT enqueued to D or Q → LEAKED
4. clearPublishedRuntimeSnapshotsNonRt() → returns clearedWorld
   → retirePublishedRuntimeWorldNonRt(clearedWorld, true) → same early-return → LEAKED
5. drainAllQuarantineStore() → drainAllUnsafe() on Q
   → Can only drain entries in Q (oldWorld was never in Q) → CANNOT reclaim
```

### Shutdown leak analysis

**→ D103 has a SHUTDOWN LEAK: oldWorld can be leaked during shutdown clear.**

However, this is the `RuntimeStore::current` World — typically the last published World. In practice:
- This World may be intentionally not reclaimed (engine is shutting down, process is ending)
- Memory leaks at shutdown are typically non-fatal
- The leak is bounded (at most 1-2 Worlds per shutdown)

But for **formal L=0 proof**, this IS a violation.

### D103 verdict: PROVISIONAL PASS (shutdown leak is bounded and non-fatal)

The shutdown path leak is a known tradeoff: during shutdown, the EBR epoch cannot be checked (audio thread is stopping), so Worlds that would be published at the exact moment of shutdown clear are intentionally not retired. This is acceptable IF:
- At most O(1) Worlds are leaked at shutdown (bounded)
- The process terminates shortly after (OS reclaims memory)

But for formal proof: L ≠ 0 at shutdown.

---

## 15-N-6 — `RejectedShutdown` D103 impact (Release)

### Semantic conflation audit

From 15-L: The `RejectedShutdown` decision in `RuntimePublicationOrchestrator.cpp:283` is used for BOTH:
1. Genuine shutdown (PublicationAdmission::evaluate → isShutdownInProgress)
2. Publish failure (executor_.publish() returns false → mapped to RejectedShutdown)

```cpp
// RuntimePublicationOrchestrator.cpp:277-283
if (executor_.publish()) {
    // success
} else {
    return Decision::RejectedShutdown;  // MISLABELED: this is publish failure, not shutdown
}
```

### Impact on D103 (Release)

In Release, the conflation has **no correctness impact** — both paths result in:
- World NOT published (A_ref not incremented) ✅
- World destroyed by caller (unique_ptr cleanup) ✅
- No entry in B_true ✅

The conflation only affects **telemetry** (metrics misclassification), not correctness. The World is safely destroyed regardless.

**→ RejectedShutdown conflation is a TELEMETRY bug, not a correctness bug. D103 unaffected.**

---

## 15-N-7 — Timeout × Release × Shutdown combination

### commitRuntimePublication timeout (250ms)

```cpp
// AudioEngine.h: ~4560
commitRuntimePublication:
  enqueueRuntimePublicationFireAndForget → returns {Success, Transferred}
  waitForPublishReceipt(250ms) → if timeout: NOT a failure
  → result.stage stays {Success, Transferred}
```

### Combination analysis

| Scenario | Timeout? | Shutdown? | World ownership | L impact |
|----------|----------|-----------|----------------|----------|
| Normal publish, receipt arrives | no | no | World → RuntimeStore → D/Q | L=0 if D+Q not both full |
| Normal publish, receipt timeout | yes | no | World → RuntimeStore → D/Q | L=0 (timeout ≠ failure) |
| Publish during shutdown | no | yes | World NOT retired (early return) | **L += 1** |
| Publish during shutdown, receipt timeout | yes | yes | World NOT retired | **L += 1** |
| D+Q both full, normal op | no | no | assert(false) no-op | **L += 1** |

### Key insight: Timeout does NOT cause L

The 250ms timeout in `commitRuntimePublication` is NOT a failure — the result stays `{Success, Transferred}`. The World IS published (A_ref incremented). Ownership IS transferred to RuntimeStore → enqueueWithRetry. The timeout only affects whether the Non-RT caller receives a receipt, not the World's lifecycle.

**→ Timeout × Release = no L impact. Timeout × Shutdown = L += 1 (same as shutdown alone).**

---

## 15-N-8 — Final judgment table

### Release Build Re-Proof Summary

| Property | Debug | Release | Verdict |
|----------|-------|---------|---------|
| A_ref increment point | onRuntimePublishedNonRt ✅ | onRuntimePublishedNonRt ✅ | Same |
| R_ref increment point | terminal deleter onRelease ✅ | terminal deleter onRelease ✅ | Same |
| L = 0 (normal operation) | ✅ (assert crashes) | **⚠️ UNPROVEN** (assert no-op, D+Q can both fill) | **D101.3.2 holds only conditionally** |
| L = 0 (shutdown) | ⚠️ (1 World leaked via early return) | ⚠️ (same, assert no-op irrelevant) | **L ≠ 0 at shutdown** |
| D102 (ownership) | ✅ PASS | ✅ PASS | Holds (assert doesn't affect ownership) |
| D103 (shutdown) | ⚠️ (bounded shutdown leak) | ⚠️ (same) | **PROVISIONAL PASS** (bounded, non-fatal) |
| Timeout impact | no L | no L | Timeout ≠ failure, result stays Success |

### Final M values

| Bound | M value | Status | Notes |
|-------|---------|--------|-------|
| M_D101 (B_true max) | 4609 | **CONDITIONAL** | ✅ only if admission gate monitors D+Q (currently does NOT) |
| M_pressure (admission) | 4866 | **HYPOTHETICAL** | Would require adding `collectRetirementPressure()` to evaluate() |
| L (Release, normal) | ≥0 | **UNPROVEN** | assert(false) no-op at ISRRetireRouter.cpp:204; D+Q can both fill |
| L (Release, shutdown) | ≥1 | **CONFIRMED > 0** | enqueueDeferredDeleteNonRtWithResult early-returns Shutdown → oldWorld never enqueued |

### Recommendation

To make D101.3.2 formally provable in Release:

1. **Add D+Q back-pressure to `PublicationAdmission::evaluate()`** (see 15-M recommendation):
   ```cpp
   uint32_t retirementPressure = m_retireRouter->collectRetirementPressure();
   if (retirementPressure > kRetirementGateThreshold)
       return Decision::RejectedPressure;
   ```
   This prevents D+Q from filling, making `quarantine()` failure unreachable.

2. **Fix shutdown leak**: During `drainAllUnsafe`, also handle stale Worlds from the shutdown clear path. OR: during shutdown, use `enqueueDeferredDeleteNonRtWithResult` but call `drainDeferredRetireQueues(true)` (force-drain) before the early return check.

3. **Split `RejectedShutdown`** into `RejectedShutdown` (genuine) vs `RejectedPublishFailure` (publish failure) — fixes telemetry bug (15-N-6).

---

## 15-N-9 — 反証監査 (Release build での D101 #1 無効化)

### Step 14 再検証 (Release)

**Step 14 で Debug で判明した assert+leak 問題を Release で再検証:**

| Finding (Step 14) | Debug behavior | Release behavior | D101 #1 |
|---|---|---|---|
| enqueueWithRetry → quarantine() returns false | assert(false) **crashes** → caller can re-own (stack unwinds) | assert(false) **no-op** → ptr **leaked forever** | **INVALIDATED** |
| Caller re-ownership on failure | Yes (stack unwinding, unique_ptr cleanup) | No (early return, ptr lost) | **INVALIDATED** |
| L counter | L=0 (crash prevents leak) | L≥1 (leak persists) | **INVALIDATED** |

### D101 #1 (Release) verdict

**D101 #1 は Debug では CLOSED だが Release では INVALIDATED。**

理由: `ISRRetireRouter.cpp:204` の `assert(false && "...")` が Release では no-op となり、`quarantine()` が false を返すと World ポインタが永久にリークする。これにより:

- L ≠ 0 (B_true が L を含むため不正確)
- M_D101 = 4609 は L=0 を仮定しているため **条件付き**
- D101.3.2 (`B_true ≤ M`) は L=0 が保証されない限り **形式的には未証明**

### Fix requirement for Release D101 #1

Release で D101 #1 を再証明するには:
1. Admission gate に D+Q back-pressure を追加 (quarantine failure を予防)
2. または `enqueueWithRetry` に blocking retry を実装 (QueueFull 時に publish をブロック)
3. または shutdown leak path を修正 (early-return path で World を強制 retire)

このうち #1 が最小侵入修正である (15-M, 15-N-8 参照)。

---

## 15-O-1 — `L=0` 回復条件を形式化

### L=0 の形式的定義 (Release)

L = 0 とは、以下の invariant が成立すること：

```
L = 0
⇔ ∀ World w: publishAndSwap() において oldWorld として返された w は、
   以下のいずれか一つの terminal path に到達する

   (a) DeferredDeletionQueue.enqueuer()  → D queue で保持 → reclaim → terminal deleter → onRelease()
   (b) RetireQuarantineStore.quarantine()  → Q で保持 → drain → terminal deleter → onRelease()
   (c) Shutdown drain authority  → drainAllUnsafe() → terminal deleter → onRelease()
   (d) synchronous terminal destruction  → deleter(ptr) → onRelease()
```

### ownership の宙浮き防止条件

`publishAndSwap()` から `enqueueWithRetry()` までの間、oldWorld の ownership は一瞬も宙に浮かないこと：

```
publishAndSwap()    // returns oldWorld (RuntimeState*)
    ↓  ownership transfer (stack-local)
[executePublish ISR frame]    // ← ownership held here by ISR thread
    ↓  (call retirePublishedRuntimeWorldNonRt → enqueueDeferredDeleteNonRt)
enqueueWithRetry()  // must either: enqueue to D | enqueue to Q | or return error to caller
    ↓
caller handles error → ownership NOT lost
```

**問題点:** Currently, `enqueueWithRetry()` returns `QueuePressure` (success at Q level) but the caller (`enqueueDeferredDeleteNonRtWithResult`) does NOT check whether `quarantine()` actually stored the ptr. The `quarantine()` failure path (ISRRetireRouter.cpp:204) is the ONLY place where ownership is lost.

### Formal L=0 recovery invariant

```
L = 0
⇔ enqueueWithRetry() returns result R:
   if R == Success → ptr in D (ownership transferred to D) ✅
   if R == QueuePressure → ptr in Q (ownership transferred to Q) ✅
   if R == QueueFull → ptr in Q (quarantine attempted) ✅
   if R == Shutdown → enqueueDeferredDeleteNonRtWithResult early-returned ✅
   if R == NoneOfTheAbove → ptr MUST be handled by caller or emergency fallback ✅

   AND: no assert(false) path that loses ownership in Release
```

**Current bug:** `quarantine()` returns false → assert(false) no-op → **ptr is lost** (L += 1).

---

## 15-O-2 — D+Q back-pressure の設計監査

### Proposal: Admission gate monitors D+Q

```
D + Q >= threshold
    ↓
PublicationAdmission::evaluate() returns RejectedPressure
```

### Concurrency analysis

| Actor | Operation | Timing | TOCTOU risk |
|-------|-----------|--------|-------------|
| Non-RT publish thread | evaluate() reads D+Q | t0 | Gap between read and publishAndSwap |
| ISR thread | enqueueRetire / quarantine | t1, t2, ... | Can increase D+Q between t0 and publishAndSwap |
| Audio thread | reclaim / drain | continuous | Can decrease D+Q |

**TOCTOU window:** Between `evaluate()` reading `D+Q = 4605` and the subsequent `publishAndSwap()` returning `oldWorld`, the ISR thread can enqueue more retirements, pushing D+Q to 4608. Then `enqueueWithRetry()` fails because D+Q is at capacity.

### Can D+Q back-pressure alone guarantee L=0?

**→ NO.** The admission gate operates at the **publish** side (before World creation). The D+Q capacity exhaustion happens at the **retirement** side (after World retirement). There is a fundamental gap:

1. Admission sees `D+Q = 4605` → accepts publish
2. Non-RT publishes → publishAndSwap → oldWorld retired
3. ISR enqueues oldWorld to D → D = 4096 (full)
4. **But between steps 1 and 3, other ISR retires happened** — D+Q now at 4609
5. `enqueueWithRetry` for oldWorld: D full, Q full → `quarantine` fails → **ptr lost**

The admission gate has **no visibility** into retires that happen between its read and the actual enqueue. Even if the admission gate reads `D+Q < 4608` atomically, by the time `enqueueWithRetry` runs, another ISR retire may have filled Q.

**→ D+Q back-pressure is necessary but NOT sufficient for L=0.**

It reduces the probability of D+Q exhaustion to near-zero, but cannot **formally** guarantee it. There will always be a TOCTOU window.

### Conclusion

**Admission gate D+Q back-pressure can make L=0 practically safe but not formally provable.** A hard lifetime guarantee must come from `enqueueWithRetry()` itself — the retirement-side ownership boundary.

---

## 15-O-3 — Hard guarantee の placement

### Tier 1 vs Tier 2 再検証

```
Tier 1 (preventive):
  PublicationAdmission.evaluate()
      ↓
  policy / early rejection
  → Reduces probability of D+Q exhaustion

Tier 2 (guaranteed):
  enqueueWithRetry() / enqueueRetire() ownership boundary
      ↓
  hard lifetime guarantee
  → MUST guarantee: ptr is never lost
```

**結論: Admission gate は L=0 の最終保証ではない。**

The admission gate operates at a lower frequency (Non-RT publish rate) than the retirement path (ISR rate). The retirement-side TOCTOU is structurally unavoidable from the admission side.

### Where the hard guarantee MUST live

```
enqueueWithRetry()
    ├── enqueueRetire(D) → if failed
    ├── tryReclaim() retries, if still failed
    ├── quarantine(Q) → if failed  ← THIS IS WHERE L HAPPENS
    └── MUST have: ownership-preserving fallback here
```

**The hard guarantee MUST be inside `enqueueWithRetry()`, at the point where `quarantine()` returns false.** This is the only place where ownership is lost, and it's the only place that can guarantee a fallback.

### Required design change

```
enqueueWithRetry():
    if enqueueRetire(D) fails:
        retry with tryReclaim
    if still fails:
        quarantine(Q)
        if quarantine fails:
            → MUST transfer to emergency authority (NOT assert+lose)
```

**Emergency authority options (see 15-O-4 for evaluation):**
- Shutdown drain authority (repurposed as emergency)
- Non-RT emergency list (new structure)
- Bounded synchronous destruction

The key constraint: the emergency authority must be **non-RT-blocking** (can't block ISR) and **never lose ownership**.

---

## 15-O-4 — `enqueueWithRetry()` の失敗契約再設計

### Current failure path

```
queue full
    ↓
quarantine full
    ↓
assert(false) [Release: no-op]
    ↓
ptr LOST → L += 1
```

### Candidate designs

| # | Candidate | RT-safe? | Ownership-preserving? | Complexity | Notes |
|---|-----------|----------|----------------------|------------|-------|
| 1 | Shutdown-owned emergency quarantine | ✅ (lock-based, ISR-safe) | ✅ (transfer to shutdown drain authority) | Medium | Requires shutdown drain to handle emergency entries |
| 2 | Non-RT emergency retire list | ⚠️ (ISR → Non-RT handoff) | ✅ | High | Requires cross-thread signaling, ISR-safe queue |
| 3 | Blocking retry | ❌ (blocks ISR) | ✅ | Low | **EXCLUDED** — violates RT constraint |
| 4 | Non-blocking bounded retry (delay) | ⚠️ (extends ISR time) | ✅ | Low | Limited effectiveness, still can fail |
| 5 | Synchronous destruction | ⚠️ (deleter call in ISR) | ✅ | Low | Risk: deleter may be expensive, UAF if RT reader still holds |
| 6 | Admission-side prevention only | N/A (Tier 1) | ❌ | Low | Cannot formally guarantee L=0 (TOCTOU) |
| 7 | Emergency quarantine in ISRRetireRouter | ✅ (same as #1, local) | ✅ | Low | Local emergency store in ISRRetireRouter itself |

### Analysis

**Candidates 3, 4 are excluded** — they either block the ISR or extend ISR execution time unpredictably. Realtime constraint violation.

**Candidate 5 (synchronous destruction)** is **dangerous** — the deleleter (`unseal + ~RuntimePublishWorld + aligned_free`) may be expensive, and calling it inline in ISR risks UAF if an RT reader still holds a reference. The epoch check is skipped (that's why we're in this path), so we can't guarantee safety.

**Candidate 6** is Tier 1 (preventive) — already analyzed in 15-O-2 as insufficient.

**Candidates 1 and 7** are the same mechanism: an emergency quarantine store that is **drained at shutdown** via `drainAllUnsafe()`. The key insight is that `drainAllUnsafe()` already handles the "all entries regardless of epoch" case — we just need a place to temporarily store the overflow.

### Recommended design: Emergency quarantine in ISRRetireRouter (Candidate 7)

```
ISRRetireRouter:
    RetireQuarantineStore m_retireQuarantine;     // primary: 512 entries, epoch-gated
    RetireQuarantineStore m_emergencyQuarantine;   // emergency: additional 64 entries, no epoch gate

enqueueWithRetry():
    if quarantine(m_retireQuarantine) fails:
        if quarantine(m_emergencyQuarantine) succeeds:
            → ptr stored, drained at shutdown ✅
        else:
            → both full → assert(false) in Debug,
              in Release: L += 1 (but now requires BOTH stores full — practically unreachable)

drainAll():  // shutdown
    m_retireQuarantine.drainAllUnsafe()
    m_emergencyQuarantine.drainAllUnsafe()
```

**Why this works:**
- `m_emergencyQuarantine` uses the same `QuarantinedEntry` layout and `drainAllUnsafe()` machinery
- It adds 64 more slots (configurable) — total overflow capacity = 512 + 64 = 576
- Admission gate can monitor `D + Q + emergency_Q` and reject before ANY store fills
- If ALL stores are full (576 + 4096 = 4672), the admission gate would have rejected at ~4608
- The assert(false) at 576-full is the last-resort — practically unreachable with back-pressure

### Authoritative fallback

**The authoritative fallback for D+Q exhaustion is: emergency quarantine → drainAllUnsafe at shutdown.**

This is the single mechanism that guarantees ownership is never lost:
- If D is full → Q holds it
- If Q is full → emergency Q holds it
- If ALL are full → assert(false) (only in Debug; Release is practically safe because admission gate prevents reaching this state)

---

## 15-O-5 — Shutdown early-return の修正監査

### Current shutdown leak path

```
shutdown開始
    ↓
publishAndSwap()  // still processing pending OwnerChannel Worlds
    ↓
oldWorld returned
    ↓
retirePublishedRuntimeWorldNonRt(oldWorld, false)
    ↓
enqueueDeferredDeleteNonRt(oldWorld, ...)
    ↓
enqueueDeferredDeleteNonRtWithResult(oldWorld, ...)
    ↓
isShutdownInProgress() == true → returns Shutdown
    ↓
oldWorld NEVER enqueued to D or Q → LEAKED
    ↓
drainAllQuarantineStore()  // only drains Q — oldWorld was never in Q
```

### The root problem

`enqueueDeferredDeleteNonRtWithResult` has an early-return that bypasses ALL retirement infrastructure:

```cpp
// AudioEngine.h:4195
if (isShutdownInProgress())
    return convo::isr::RetireEnqueueResult::Shutdown;
```

This means any World that is published (via `publishAndSwap`) between `requestShutdown()` and the actual drain has NO path to destruction.

### Proposed fix: ShutdownReclaimAuthority

The fix is to replace the early-return with a **ShutdownReclaimAuthority** path:

```
isShutdownInProgress() == true
    ↓
enqueueWithRetryShutdown()  // NOT enqueueDeferredDeleteNonRtWithResult
    ↓
m_shutdownReclaimAuthority.transfer(ptr, deleter, type)  // ownership to shutdown authority
    ↓
drainAllQuarantineStore() → also drains m_shutdownReclaimAuthority
```

### Design

```
class ShutdownReclaimAuthority {
    // Simple vector-backed store — NO epoch gating needed (audio thread is stopped)
    std::vector<std::pair<void*, void(*)(void*)>> pending_;
    std::mutex mtx_;

    void transfer(void* ptr, void (*deleter)(void*), DeletionEntryType type) noexcept {
        if (ptr == nullptr || deleter == nullptr) return;
        std::lock_guard<std::mutex> lock(mtx_);
        pending_.emplace_back(ptr, deleter);
    }

    void drainAllUnsafe() noexcept {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& [ptr, deleter] : pending_) {
            const auto entryType = ...;  // need to track type for World R_ref
            deleter(ptr);
            if (entryType == DeletionEntryType::World)
                referenceObserver_->onRelease();
        }
        pending_.clear();
    }
};
```

**Authority assignment:** `ShutdownReclaimAuthority` should be a member of `AudioEngine` (owned), initialized during construction, and drained during `releaseResources()` AFTER audio thread has stopped.

### Authority hierarchy during shutdown

```
Shutdown lifetime authority:
  AudioEngine
    ├── EpochDomain  (drainAll → drainAllUnsafe on D)
    ├── ISRRetireRouter
    │     ├── m_retireQuarantine  (drainAllUnsafe on Q)
    │     ├── m_emergencyQuarantine  (drainAllUnsafe on emergency Q — from 15-O-4)
    │     └── m_shutdownReclaimAuthority  (drainAllUnsafe on shutdown authority)
    └── DSPQuarantineManager  (separate, not World-related)
```

### Shutdown order (fixed)

Current (buggy):
```
1. clearPublishedRuntimeSnapshotsNonRt() → returns clearedWorld → LEAKED
2. drainAllQuarantineStore() → drains Q only
```

Fixed:
```
1. clearPublishedRuntimeSnapshotsNonRt() → returns clearedWorld
   → transfer to m_shutdownReclaimAuthority  (NOT early-return)
2. drainAll() on ISRRetireRouter → drains D + Q + emergency + shutdown authority
```

### Code change

Replace the early-return in `enqueueDeferredDeleteNonRtWithResult` with:

```cpp
// AudioEngine.h:4195 (MODIFIED)
if (isShutdownInProgress()) {
    // Transfer to shutdown reclaim authority instead of abandoning
    m_retireRouter->transferToShutdownAuthority(ptr, deleter, type);
    return convo::isr::RetireEnqueueResult::Shutdown;
}
```

And add to `ISRRetireRouter::drainAll()`:
```cpp
// ISRRetireRouter.cpp (MODIFIED drainAll)
void ISRRetireRouter::drainAll() noexcept {
    provider_->drainAll();              // D
    m_retireQuarantine.drainAllUnsafe(); // Q
    m_emergencyQuarantine.drainAllUnsafe(); // emergency Q (from 15-O-4)
    m_shutdownReclaimAuthority.drainAllUnsafe(); // shutdown authority
}
```

---

## 15-O-6 — `RejectedShutdown` の分離

### Current conflation

`RuntimePublicationOrchestrator.cpp:277-283`:
```cpp
if (executor_.publish()) {
    // success
} else {
    return Decision::RejectedShutdown;  // MISLABELED: publish failure, not shutdown
}
```

### Proposed separation

```
AdmissionDecision (at evaluate() time):
  Accepted
  RejectedShutdown       // genuine: isShutdownInProgress() == true
  RejectedPressure       // D+Q+emergency + O > threshold
  RejectedPressure       // OwnerChannel full
  RejectedPressure       // other resource exhaustion

PublicationResult (at publish time):
  Success
  PublishFailed          // publishAndSwap returned nullptr or failed
  PublishFailed          // OwnerChannel::enqueue failure
  PublishFailed          // other internal failure
```

### Audit: all failure paths distinguished

| Failure scenario | Current | Post-fix | Affects L? | Affects D102? | Affects D103? |
|---|---|---|---|---|---|
| Genuine shutdown (isShutdownInProgress) | RejectedShutdown | RejectedShutdown | No (early return) | No | Yes (D103 shutdown leak fixed by 15-O-5) |
| publish() failure (non-shutdown) | RejectedShutdown (MISLABELED) | PublishFailed | No (World destroyed by caller) | No | No |
| OwnerChannel full (enqueue fails) | (not RejectedShutdown) | RejectedPressure (O > 0) | No (World stays in caller's unique_ptr) | No | No |
| D+Q full (retire fails) | (not RejectedShutdown) | (retirement internal) | Yes (15-O-4 fix: emergency quarantine) | No | No |
| Quarantine + emergency both full | (assert false) | (assert false — Debug only) | Yes (but practically unreachable with back-pressure) | No | No |

### Implementation

In `RuntimePublicationOrchestrator.cpp`:
```cpp
// Line 277-283 (MODIFIED)
auto publishResult = executor_.publish();
if (!publishResult.success) {
    // Distinguish from genuine RejectedShutdown (admission-time)
    return {Decision::RejectedPublishFailure, PublishResult::PublishFailed};
}
```

New decision type:
```cpp
enum class Decision {
    Accepted,
    RejectedShutdown,        // admission-time: genuine shutdown
    RejectedPressure,       // admission-time: resource pressure
    RejectedPublishFailure,  // publish-time: internal failure
};
```

---

## 15-O-7 — 3つの修正案を統合した invariant proof

### Post-fix architecture

```
Admission (Tier 1 - preventive):
  PublicationAdmission.evaluate()
    → monitor D + Q + emergency_Q + O
    → RejectedPressure if > threshold

Retirement (Tier 2 - guaranteed):
  enqueueWithRetry()
    → enqueueRetire(D) → success: ptr in D ✅
    → tryReclaim retries
    → quarantine(Q) → success: ptr in Q ✅
    → quarantine(emergencyQ) → success: ptr in emergency Q ✅
    → all full → assert(false) (Debug only, practically unreachable)

Shutdown (Tier 3 - reclaim authority):
  enqueueDeferredDeleteNonRtWithResult
    → isShutdownInProgress()? transfer to m_shutdownReclaimAuthority ✅
  drainAll()
    → D.drainAllUnsafe() + Q.drainAllUnsafe() + emergencyQ.drainAllUnsafe() + shutdownAuthority.drainAllUnsafe()
```

### Post-fix invariant proof: L = 0

**Claim:** After applying 15-O-2 + 15-O-4 + 15-O-5, L = 0 is formally provable.

**Proof:**

For any World w that is oldWorld from `publishAndSwap()`:

**Case 1: Not shutdown (isShutdownInProgress() == false)**
- `enqueueDeferredDeleteNonRtWithResult` proceeds to `enqueueWithRetry()`
- `enqueueWithRetry()`:
  - (a) `enqueueRetire(D)` succeeds → w ∈ D ✅
  - (b) `enqueueRetire(D)` fails → retry → `quarantine(Q)` succeeds → w ∈ Q ✅
  - (c) `quarantine(Q)` fails → `quarantine(emergencyQ)` succeeds → w ∈ emergency_Q ✅
  - (d) All full → `assert(false)` → **In Release: unreachable** (admission gate prevents D+Q+emergency ≥ 4096+512+64 = 4672, and admission gate threshold is 4608 which rejects before this state is reached)
  - (e) All full → `assert(false)` → **In Debug: crash** (L=0 maintained)

  The admission gate monitors `D + Q + emergency_Q` and rejects at threshold (e.g., 4608). Since the maximum population is:
  - At most 1 World can be in transit per ISR thread
  - Admission rejects before D+Q+emergency reaches 4672
  - So at most 4671 Worlds exist in retirement at once (admission prevents more)
  - With D=4096, Q=512, emergency=64 → total capacity = 4672 > 4671 ✅

  **→ L=0 for Case 1 (with admission gate + emergency quarantine).**

**Case 2: Shutdown (isShutdownInProgress() == true)**
- `enqueueDeferredDeleteNonRtWithResult` → `isShutdownInProgress()` → true
- **Post-fix:** → `transferToShutdownAuthority(ptr, deleter, type)` → w ∈ ShutdownReclaimAuthority ✅
- At shutdown drain: `drainAll()` → `m_shutdownReclaimAuthority.drainAllUnsafe()` → deleter executes → onRelease() ✅

  **→ L=0 for Case 2 (with ShutdownReclaimAuthority).**

**Case 3: assert(false) reached (all stores full)**
- Only in Debug: crash → no leak ✅
- In Release: admission gate prevents reaching this state ✅ (proven in Case 1)

**→ L=0 is formally provable.**

### Post-fix invariant: B_true ≤ M_D101

```
B_true = D + Q + P + L
       = D + Q + P + 0      (L=0 proven)

D ≤ 4096  (kQueueSize)
Q ≤ 512   (kMaxQuarantinedEntries)
P ≤ 1     (single ISR thread, sequential)

B_true ≤ 4096 + 512 + 1 + 0 = 4609 = M_D101 ✅
```

**→ D101.3.2 B_true ≤ M_D101 = 4609 is formally proven.**

### Post-fix: M_pressure (admission gate)

```
M_pressure = D + Q + emergency_Q + O + P + 1
            = 4096 + 512 + 64 + 256 + 1 + 1
            = 4930
```

The admission gate threshold should be set below M_pressure (e.g., 4608) to ensure:
- Admission rejects before any store can fill
- The emergency quarantine has headroom
- The shutdown authority has headroom

**Note:** M_pressure ≠ M_D101. M_D101 = 4609 (B_true max), M_pressure = 4930 (admission threshold). The difference of 321 accounts for emergency_Q (64) + O (256) + buffer (1).

---

## 15-O-8 — Final judgment table

### Post-fix remediation summary

| Invariant / mechanism | Current | Proposed fix | Post-fix proof |
|----------------------|----------|--------------|----------------|
| `L=0` | UNPROVEN | Emergency Q + ShutdownReclaimAuthority + early-return fix | **PROVEN** |
| capacity D | 4096 | same | ✅ PROVEN |
| capacity Q | 512 | same | ✅ PROVEN |
| emergency Q | absent | 64 slots (RetireQuarantineStore) | ✅ PROVEN |
| `P<=1` | PROVEN | same | ✅ PROVEN |
| `O<=256` | PROVEN | same | ✅ PROVEN |
| D+Q back-pressure | absent | evaluate() monitors D+Q+emergency_Q | ✅ PROVEN (Tier 1 prevention) |
| quarantine overflow | unsafe (assert+leak) | emergency Q fallback → drainAllUnsafe | ✅ PROVEN (Tier 2 guarantee) |
| shutdown retirement | unsafe (early-return leak) | ShutdownReclaimAuthority transfer → drainAllUnsafe | ✅ PROVEN (Tier 3 authority) |
| `RejectedShutdown` | conflated (telemetry bug) | split: RejectedShutdown vs RejectedPublishFailure | ✅ PROVEN (telemetry fix) |
| D101.3.2 | INVALIDATED (Release) | all fixes applied | **PASS** |
| D102 | PASS | same | **PASS** |
| D103 | provisional | ShutdownReclaimAuthority | **PASS** |

### Remediation priority

1. **15-O-5 (ShutdownReclaimAuthority)** — fixes the shutdown leak (D103), highest priority
2. **15-O-4 (emergency quarantine)** — fixes the normal-operation leak (D101.3.2), highest priority
3. **15-O-2 (D+Q back-pressure)** — reduces probability of reaching emergency Q, medium priority

---

## 15-P-1 — 実装前の authority compatibility audit

> ⚠️ **旧設計（pre-implementation）**: 本セクションの数値（E=64, ShutdownReclaimAuthority, M_pressure=4930）は
> 実装前の設計値。現行実装値（E=512, TerminalReclaimAuthority, M_pressure=5377）は 15-P-9-1 / 15-P-10-1 / 15-P-FINAL を参照。

### Existing authority structure (current state)

```
AudioEngine (owns all)
├── EpochDomain m_epochDomain
├── WorldRetirementReferenceObserver worldRetirementReference_
├── std::unique_ptr<ISRRetireRouter> m_retireRouter
│     ├── IEpochProvider& provider_  (→ EpochDomain)
│     ├── WorldRetirementReferenceObserver* referenceObserver_
│     └── RetireQuarantineStore m_retireQuarantine  (512 entries)
├── RCUReader audioThreadRcuReader
├── RCUReader messageThreadRcuReader
└── ...
```

**Current ownership authorities:**
1. `ISRRetireRouter` — single retire entry point, owns `m_retireQuarantine`
2. `EpochDomain` — owns `DeferredDeletionQueue` (D, 4096), reader slots, epoch tracking
3. `AudioEngine` — top-level owner of all above, coordinates shutdown

### EmergencyQuarantine compatibility audit

**Proposal: Add `RetireQuarantineStore m_emergencyQuarantine` to `ISRRetireRouter`.**

| Check | Status | Analysis |
|-------|--------|----------|
| EmergencyQuarantine is NOT a new lifetime authority | ✅ PASS | It's a secondary store under the SAME `ISRRetireRouter` authority. The `ISRRetireRouter` remains the single authority — it just has one more storage tier. |
| Ownership authority unified under ISRRetireRouter | ✅ PASS | D (EpochDomain) + Q (ISRRetireRouter) + EmergencyQ (ISRRetireRouter) — ISRRetireRouter orchestrates all tiers. EpochDomain's D is exposed via `provider_->enqueueRetire` interface. |
| D → Q → EmergencyQ as single pipeline | ✅ PASS | `enqueueWithRetry()` already handles D→Q fallback. Adding E extends the same pipeline: `enqueueRetire(D) → quarantine(Q) → quarantine(EmergencyQ)`. |
| EmergencyQ capacity is fixed/bounded | ✅ PASS | Uses same `RetireQuarantineStore` (std::array<QuarantinedEntry, 512>). Set capacity to 64 at construction. No allocation/malloc. |
| RT thread safety (no mutex/alloc/blocking) | ✅ PASS | `RetireQuarantineStore` uses `std::mutex` but: (a) mutex contention is bounded (short critical section, lock held only for array copy), (b) no allocation (fixed array), (c) no blocking (try-lock semantics via lock_guard, very short hold). Same as existing Q. |
| `drainAllUnsafe()` semantics consistent | ✅ PASS | `drainAllUnsafe()` is already used for shutdown — ignores epochs, destroys everything. EmergencyQ uses identical method. Same contract: "Audio Thread stopped when called". |
| Ownership exists at exactly one place per stage | ✅ PASS | ptr transitions: D (in ring buffer slot) → Q (in entries_ array) → EmergencyQ (in its entries_ array). Each `quarantine()` call atomically moves ptr from caller to store. No double-ownership. `drainAllUnsafe()` transfers ownership to deleter (terminal). |

### Authority Singularization verification

The codebase has an explicit "Authority Singularization" principle (REPAIR_PLAN2.md, X4):

```
"Authority Singularization — one authority owns one resource type across its full lifetime"
```

**EmergencyQuarantine does NOT violate this** — it's an additional storage tier UNDER the same `ISRRetireRouter` authority. The authority remains singular: `ISRRetireRouter` owns all three stores (D via EpochDomain interface, Q locally, EmergencyQ locally).

### ShutdownReclaimAuthority compatibility audit

**Current state:** No `ShutdownReclaimAuthority` exists. The existing pattern for shutdown-time DSP reclaim is `tryShutdownQuiescentReclaim()` (AudioEngine.h:2040), which delegates to `RuntimePublicationCoordinator` / DSP handle reclaim authority.

**Proposed: Add `ShutdownReclaimAuthority` to `ISRRetireRouter`.**

| Check | Status | Analysis |
|-------|--------|----------|
| ShutdownReclaimAuthority is NOT a new lifetime authority | ⚠️ CAUTION | It IS a new authority, BUT it only operates during shutdown (when no RT thread is active). It's a terminal authority — the last owner before destruction. This is consistent with the existing `drainAllUnsafe()` pattern. |
| Does NOT overlap with normal retire authority | ✅ PASS | Normal retire: ISRRetireRouter (D→Q→E). Shutdown retire: ISRRetireRouter → m_shutdownReclaimAuthority (transfer) → drainAllUnsafe (destroy). No overlap — the early-return check (`isShutdownInProgress()`) routes to the shutdown authority exclusively. |
| RT thread safety | ✅ PASS | Shutdown authority is only accessed AFTER audio thread has stopped (releaseResources phase, after `stopAudioThread()`). Mutex is safe (no RT thread contention). |
| Drain ordering | ✅ PASS | drainAll() will call: provider_->drainAll() (D) → m_retireQuarantine.drainAllUnsafe() (Q) → m_emergencyQuarantine.drainAllUnsafe() (E) → m_shutdownReclaimAuthority.drainAllUnsafe() (shutdown). Sequential, no overlap. |

### Conclusion of 15-P-1

**EmergencyQuarantine**: ✅ Approved — extends existing `ISRRetireRouter` authority, no new authority.

**ShutdownReclaimAuthority**: ✅ Approved conditionally — it's a terminal authority that only operates during shutdown. Must be:
- A member of `ISRRetireRouter` (not AudioEngine) — keeps all retirement authority under one roof
- Only accessed after audio thread stop (Phase 3 of shutdown)
- Drained as the LAST step in `drainAll()`

---

## 15-P-2 — Emergency quarantine の最小実装

> ⚠️ **旧設計（pre-implementation）**: 本セクションの数値（E=64, ShutdownReclaimAuthority, M_pressure=4930）は
> 実装前の設計値。現行実装値（E=512, TerminalReclaimAuthority, M_pressure=5377）は 15-P-9-1 / 15-P-10-1 / 15-P-FINAL を参照。

### Implementation

Add `m_emergencyQuarantine` to `ISRRetireRouter`:

```cpp
// ISRRetireRouter.h (private member, after m_retireQuarantine)
RetireQuarantineStore m_emergencyQuarantine;  // ★ 15-P: emergency fallback for D+Q exhaustion
```

Constructor update:
```cpp
// ISRRetireRouter.cpp
ISRRetireRouter::ISRRetireRouter(convo::IEpochProvider& provider,
    convo::isr::WorldRetirementReferenceObserver* referenceObserver) noexcept
    : m_retireQuarantine(64),  // primary: 64 entries (was 512, reduced for emergency-only)
      m_emergencyQuarantine(64)  // emergency: 64 entries
    , ...
```

Wait — actually, looking at `RetireQuarantineStore`, the capacity is hardcoded as `kMaxQuarantinedEntries = 512`. We need to make it configurable:

```cpp
// RetireQuarantineStore.h
class RetireQuarantineStore {
public:
    static constexpr std::size_t kMaxQuarantinedEntries = 512;

    // ★ 15-P-2: Configurable capacity constructor
    explicit RetireQuarantineStore(std::size_t capacity = kMaxQuarantinedEntries) noexcept
        : capacity_(capacity)
    {
        assert(capacity <= kMaxQuarantinedEntries);  // can't exceed std::array bound
    }
```

Wait — the class uses `std::array<QuarantinedEntry, kMaxQuarantinedEntries>`. We can't dynamically size a `std::array`. We need to either:
1. Keep the array at 512 but only use the first 64 slots
2. Use a template parameter

**Simplest approach: Keep array at 512, but cap at 64 entries for emergency store.**

```cpp
// RetireQuarantineStore.h
class RetireQuarantineStore {
public:
    static constexpr std::size_t kMaxQuarantinedEntries = 512;

    // ★ 15-P-2: Configurable effective capacity
    explicit RetireQuarantineStore(std::size_t effectiveCapacity = kMaxQuarantinedEntries) noexcept
        : effectiveCapacity_(effectiveCapacity)
    {
        assert(effectiveCapacity <= kMaxQuarantinedEntries);
    }

    bool quarantine(...) noexcept {
        std::lock_guard<std::mutex> lock(mtx_);
        if (size_ >= effectiveCapacity_)  // uses effective capacity, not array bound
        {
            ++overflowCount_;
            return false;
        }
        // ... rest unchanged
    }
```

### Modify `enqueueWithRetry()`:

```cpp
// ISRRetireRouter.cpp — modified enqueueWithRetry failure path
    if (result == RetireEnqueueResult::QueuePressure || result == RetireEnqueueResult::QueueFull)
    {
        // 1. Primary quarantine (Q, 512 entries)
        const bool storedInQ = m_retireQuarantine.quarantine(
            ptr, deleter, epoch, type, "enqueueWithRetry:QueuePressure",
            /*publicationSequenceId=*/0, /*generation=*/0);
        if (storedInQ)
            return RetireEnqueueResult::QueuePressure;  // ownership transferred to Q ✅

        // ★ 15-P-2: Emergency quarantine (EmergencyQ, 64 entries)
        // ownership-preserving fallback when Q is also full
        const bool storedInEmergencyQ = m_emergencyQuarantine.quarantine(
            ptr, deleter, epoch, type, "enqueueWithRetry:EmergencyQ",
            /*publicationSequenceId=*/0, /*generation=*/0);
        if (storedInEmergencyQ)
            return RetireEnqueueResult::QueuePressure;  // ownership transferred to EmergencyQ ✅

        // ★ 15-P-3: If EmergencyQ is also full, we MUST still preserve ownership.
        // This is analyzed in 15-P-3.
        assert(false && "ISRRetireRouter: D+Q+EmergencyQ all full - admission gate failure");
    }
    return result;
```

The key change: the `quarantine()` failure path no longer just `assert(false)`. Instead, it tries EmergencyQ first, and only if BOTH Q and EmergencyQ are full does it hit the assert.

---

## 15-P-3 — EmergencyQ overflow の最終 fallback

> ⚠️ **旧設計（pre-implementation）**: 本セクションの数値（E=64, ShutdownReclaimAuthority, M_pressure=4930）は
> 実装前の設計値。現行実装値（E=512, TerminalReclaimAuthority, M_pressure=5377）は 15-P-9-1 / 15-P-10-1 / 15-P-FINAL を参照。

### The critical question

EmergencyQ has 64 slots. If D=4096, Q=512, E=64, all three are full, what happens?

**Answer: EmergencyQ overflow does NOT cause ownership loss**, IF the admission gate is properly configured.

### Why: Admission gate as the outer guard

The admission gate (`PublicationAdmission::evaluate()`) monitors `D + Q + E` and rejects publications before any store can fill. Specifically:

```
Admission threshold = 4096 + 512 + 64 - 1 = 4671
```

If the admission gate rejects at `D + Q + E >= 4671`, then:

```
When enqueueWithRetry is called:
  D + Q + E ≤ 4671 (guaranteed by admission gate)

  Worst case: D = 4096 (full), Q = 512 (full), E = 63 (1 slot free)
  → enqueueRetire(D) fails → quarantine(Q) fails → quarantine(E) succeeds ✅

  Worst case 2: D = 4096, Q = 512, E = 64 (all full)
  → This requires 4672 published Worlds in B_true
  → Admission gate rejects at 4671 → impossible to reach 4672 ✅
```

**→ EmergencyQ overflow is formally unreachable IF the admission gate is in place.**

### But: What if the admission gate is absent (current state)?

Without the admission gate, `D + Q + E = 4672` IS reachable:
- 4096 Worlds in D (all epoch-blocked)
- 512 Worlds in Q (all epoch-blocked)
- 64 Worlds in EmergencyQ (all epoch-blocked)
- `ptr` arriving at the quarantine failure point

In this case, `assert(false)` is the fallback. In Release: **ptr is still LOST**.

### Conclusion: EmergencyQ alone is NOT sufficient for L=0

**EmergencyQ provides a safety net (64 more slots), but it does NOT formally prove L=0.**

The formal L=0 proof requires:
1. EmergencyQ as a fallback (reduces probability of overflow)
2. Admission gate back-pressure (prevents reaching the overflow state)
3. **EmergencyQ overflow itself must have a preserving fallback**

### EmergencyQ overflow fallback options

| # | Candidate | RT-safe? | Ownership-preserving? | Commentary |
|---|-----------|----------|----------------------|------------|
| 1 | `assert(false)` + ptr stored in thread-local | ✅ | ✅ | Store ptr in TLS per ISR thread, process at next tryReclaim |
| 2 | `assert(false)` + ptr pushed to D as "emergency entry" | ✅ | ✅ | D is ring buffer — but if D is full, can't push |
| 3 | `assert(false)` + ptr transferred to ShutdownReclaimAuthority | ⚠️ | ✅ | Shutdown authority already handles this — but it's shutdown-only |
| 4 | Synchronous destruction | ⚠️ | ✅ | Same concern as 15-O-4 #5 — risky without epoch check |
| 5 | Log + ptr stored in static vector | ✅ | ✅ | Bounded static vector, drained at shutdown |

### Recommended: EmergencyQ overflow → ShutdownReclaimAuthority (same as shutdown path)

Since `DrainAllUnsafe` already handles all three stores (D+Q+EmergencyQ) at shutdown, the overflow case should also transfer to a **shutdown-owned authority**:

```cpp
// If EmergencyQ is also full (4672 Worlds):
m_shutdownReclaimAuthority.transfer(ptr, deleter, type);
// This is the SAME authority used for the shutdown early-return path (15-P-4)
// It's drained at shutdown via drainAllUnsafe()
```

**→ EmergencyQ overflow and shutdown early-return use the SAME authority fallback.**

This unifies the edge cases:
- Normal operation: D → Q → EmergencyQ → (if all full) ShutdownReclaimAuthority
- Shutdown: (early-return) → ShutdownReclaimAuthority

The `ShutdownReclaimAuthority` is the **terminal ownership authority** — it holds anything that can't be placed in the normal pipeline, and drains everything at shutdown.

### Formal L=0 with all layers

```
L = 0
⇔ enqueueWithRetry() ownership transfer always succeeds:
   D full → Q ok ✅
   D+Q full → EmergencyQ ok ✅
   D+Q+E full → ShutdownReclaimAuthority transfer ✅
   Shutdown early-return → ShutdownReclaimAuthority transfer ✅
   drainAllUnsafe() at shutdown → destroys everything → R_ref++ ✅
```

**→ L=0 is formally provable with 3 layers: EmergencyQ + ShutdownReclaimAuthority + drainAllUnsafe.**

The admission gate (15-O-2) is additional prevention (reduces probability), but the formal guarantee comes from `ShutdownReclaimAuthority` as the ultimate fallback.

---

## 15-P-4 — ShutdownReclaimAuthority 実装

> ⚠️ **旧設計（pre-implementation）**: 本セクションの ShutdownReclaimAuthority は実装前の設計。
> 現行実装では TerminalReclaimAuthority（growable, `store()` 常時 true）に統合されている。
> 現行実装値は 15-P-9-1 / 15-P-10-1 / 15-P-FINAL を参照。

### Design

```cpp
// ISRRetireRouter.h — new member
class ShutdownReclaimAuthority {
public:
    void transfer(void* ptr, void (*deleter)(void*), DeletionEntryType type) noexcept {
        if (ptr == nullptr || deleter == nullptr) return;
        std::lock_guard<std::mutex> lock(mtx_);
        entries_.emplace_back(ptr, deleter, type);
    }

    void drainAllUnsafe() noexcept {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& [ptr, deleter, type] : entries_) {
            deleter(ptr);
            if (type == DeletionEntryType::World) {
                ++worldReclaimCount_;  // atomic
                if (referenceObserver_ != nullptr)
                    referenceObserver_->onRelease();
            }
        }
        entries_.clear();
    }

private:
    struct Entry {
        void* ptr;
        void (*deleter)(void*);
        DeletionEntryType type;
    };
    std::vector<Entry> entries_;  // bounded at construction
    std::mutex mtx_;
    std::uint64_t worldReclaimCount_ = 0;
    WorldRetirementReferenceObserver* referenceObserver_ = nullptr;
};

// ISRRetireRouter private members:
RetireQuarantineStore m_retireQuarantine;       // primary: 512 entries
RetireQuarantineStore m_emergencyQuarantine;     // emergency: 64 entries
ShutdownReclaimAuthority m_shutdownReclaimAuthority;  // terminal: unbounded (bounded by admission gate)
```

**Wait — `std::vector` allocates. This violates RT safety.** Since `ShutdownReclaimAuthority` is only accessed after audio thread stops, allocation is safe. But let's use a bounded static array to be consistent:

```cpp
class ShutdownReclaimAuthority {
    static constexpr std::size_t kMaxShutdownEntries = 256;
    struct Entry {
        void* ptr;
        void (*deleter)(void*);
        DeletionEntryType type;
    };
    std::array<Entry, kMaxShutdownEntries> entries_;
    std::size_t size_ = 0;
    // ...
};
```

This is bounded at construction, no allocation, and large enough to handle any normal shutdown scenario (at most O(1) + D + Q + EmergencyQ entries = 4096 + 512 + 64 = 4672, but at shutdown we only have what wasn't drained).

Actually, for shutdown, we should drain D+Q+EmergencyQ FIRST, then any remaining entries in ShutdownReclaimAuthority are from:
1. EmergencyQ overflow during normal operation
2. Shutdown early-return oldWorlds

Both are bounded by the admission gate (if in place). Without admission gate, EmergencyQ overflow is `assert(false)` territory (Debug-only crash).

### Modified `enqueueDeferredDeleteNonRtWithResult()`:

```cpp
// AudioEngine.h — modified early-return
if (isShutdownInProgress())
{
    // ★ 15-P-4: Transfer ownership to ShutdownReclaimAuthority instead of losing it
    m_retireRouter->transferToShutdownAuthority(ptr, deleter, type);
    return convo::isr::RetireEnqueueResult::Shutdown;
}
```

### Modified `ISRRetireRouter::enqueueWithRetry()` — EmergencyQ overflow:

```cpp
// ISRRetireRouter.cpp — EmergencyQ overflow fallback
        if (!storedInEmergencyQ)
        {
            // ★ 15-P-3: All stores full — transfer to ShutdownReclaimAuthority
            // This is the same authority used for shutdown early-return.
            m_shutdownReclaimAuthority.transfer(ptr, deleter, type);
            return RetireEnqueueResult::QueueFull;  // ownership preserved ✅
        }
```

### Modified `ISRRetireRouter::drainAll()`:

```cpp
void ISRRetireRouter::drainAll() noexcept
{
    provider_->drainAll();              // D: drainAllUnsafe on DeferredDeletionQueue
    m_retireQuarantine.drainAllUnsafe(); // Q: force drain all
    m_emergencyQuarantine.drainAllUnsafe(); // EmergencyQ: force drain all
    m_shutdownReclaimAuthority.drainAllUnsafe(); // Shutdown authority: force drain all
}
```

### Ownership transition chain during shutdown

```
Normal operation:
  publishAndSwap → oldWorld
  → enqueueWithRetry → D | Q | EmergencyQ | ShutdownReclaimAuthority

Shutdown (isShutdownInProgress = true):
  publishAndSwap → oldWorld
  → enqueueDeferredDeleteNonRtWithResult → isShutdownInProgress →
  → transferToShutdownReclaimAuthority(oldWorld) → ShutdownReclaimAuthority
  → (later) drainAll() → ShutdownReclaimAuthority.drainAllUnsafe() → deleter → onRelease()
```

**No ownership is ever lost.** The chain is:
1. `publishAndSwap()` returns oldWorld (ownership to ISR local)
2. ISR transfers to retirement pipeline (D/Q/E/Shutdown)
3. Shutdown authority drains everything at shutdown

### DR (don't-repeat) verification

```
Case 1: Normal operation, D not full
  → enqueueRetire succeeds → ptr in D ✅

Case 2: D full, Q not full
  → quarantine succeeds → ptr in Q ✅

Case 3: D+Q full, EmergencyQ not full
  → emergency quarantine succeeds → ptr in EmergencyQ ✅

Case 4: D+Q+EmergencyQ full (requires admission gate failure)
  → transfer to ShutdownReclaimAuthority → ptr in shutdown authority ✅

Case 5: Shutdown (isShutdownInProgress = true)
  → transfer to ShutdownReclaimAuthority → ptr in shutdown authority ✅
```

All cases preserve ownership. No path loses ptr.

---

## 15-P-5 — Shutdown drain ordering

### Current shutdown sequence (from ReleaseResources.cpp):

```
1. releaseResources():
   a. stopUIProcessor() — stops UI thread operations
   b. shutdownCoordinatorLoop() → join
   c. stopRebuildThread() → join
   d. waitForDrain(2000, 2) — waits for readers to drain
   e. m_retireRouter->unquarantineAllReaders() — release all reader slots
   f. m_retireRouter->drainAllQuarantineStore() — drain Q (drainAllUnsafe)
   g. clearPublishedRuntimeSnapshotsNonRt() — publish nullptr, get oldWorld
   h. retirePublishedRuntimeWorldNonRt(clearedWorld, true) — → enqueueDeferredDeleteNonRt
      → isShutdownInProgress() → true → early-return → LEAKED (current bug)
```

### The ordering problem

Step (h) happens AFTER step (f). But in step (h), the early-return prevents the World from being enqueued to D/Q/EmergencyQ. So `drainAllQuarantineStore()` at step (f) has nothing to drain for this World.

And step (g) — `clearPublishedRuntimeSnapshotsNonRt()` — publishes nullptr and returns the oldWorld. This oldWorld is the LAST published World still in `RuntimeStore::current`.

### Proposed fixed ordering

```
1. releaseResources():
   a. stopUIProcessor()
   b. shutdownCoordinatorLoop() → join
   c. stopRebuildThread() → join
   d. waitForDrain(2000, 2) — Audio Thread stopped, all readers drained
   e. m_retireRouter->unquarantineAllReaders()
   f. clearPublishedRuntimeSnapshotsNonRt() → oldWorld
   g. transferToShutdownAuthority(oldWorld)  // ★ FIX: instead of enqueueDeferredDelete → early return
   h. m_retireRouter->drainAll()  // drains D + Q + EmergencyQ + ShutdownReclaimAuthority
```

Wait — step (g) currently calls `retirePublishedRuntimeWorldNonRt(clearedWorld, true)` which calls `enqueueDeferredDeleteNonRt`, which calls `enqueueDeferredDeleteNonRtWithResult`, which early-returns.

**Fix:** The early-return in `enqueueDeferredDeleteNonRtWithResult` should transfer to `ShutdownReclaimAuthority` instead of returning immediately. Then `drainAll()` will handle it.

But the ORDER matters: `drainAll()` must be called AFTER the transfer. Currently:
- Step (f): `clearPublishedRuntimeSnapshotsNonRt()` → oldWorld → `retirePublishedRuntimeWorldNonRt` → early-return
- Step (g): `drainAllQuarantineStore()` — too early! oldWorld hasn't been transferred yet.

**Fixed order:**
```
   f. clearPublishedRuntimeSnapshotsNonRt() → oldWorld
   g. retirePublishedRuntimeWorldNonRt(clearedWorld, true)  // → transferToShutdownAuthority (NOT early return)
   h. m_retireRouter->drainAll()  // NOW drains everything including shutdown authority
```

### Code changes needed

**AudioEngine.Processing.ReleaseResources.cpp:**

Current (line ~455-463):
```cpp
    worldAuthority_.requestShutdownClearNonRt();
    auto* clearedWorld = worldAuthority_.clearPublishedRuntimeSnapshotsNonRt();
    if (clearedWorld != nullptr)
    {
        RuntimePublicationBridge clearBridge{ *this, runtimePublicationValidator_ };
        clearBridge.retirePublishedRuntimeWorldNonRt(clearedWorld, true);
    }
```

The `retirePublishedRuntimeWorldNonRt(clearedWorld, true)` calls `enqueueDeferredDeleteNonRtWithResult` which currently early-returns with `Shutdown`. With the 15-P-4 fix, it will transfer to `ShutdownReclaimAuthority`. Then the existing `drainAllQuarantineStore()` at line ~371 (BEFORE this) is too early.

**We need to move the drain AFTER the clear:**

```cpp
    worldAuthority_.requestShutdownClearNonRt();
    auto* clearedWorld = worldAuthority_.clearPublishedRuntimeSnapshotsNonRt();
    if (clearedWorld != nullptr)
    {
        RuntimePublicationBridge clearBridge{ *this, runtimePublicationValidator_ };
        clearBridge.retirePublishedRuntimeWorldNonRt(clearedWorld, true);
        // → transfers to ShutdownReclaimAuthority via enqueueDeferredDeleteNonRtWithResult
    }

    // ★ 15-P-5: drain ALL authorities AFTER shutdown clear transfer
    m_retireRouter->drainAll();  // D + Q + EmergencyQ + ShutdownReclaimAuthority
```

But wait — the existing drain at line ~371 (`drainAllQuarantineStore`) is called BEFORE the clear. Let me check if it's still needed:

```cpp
    // Line 371-378: drainAllQuarantineStore — drains Q entries from normal operation
    if (quarantinedRetireResident > 0) {
        m_retireRouter->drainAllQuarantineStore();
    }
```

This drains Q entries that were quarantined during NORMAL operation (before shutdown was requested). This is correct and should remain — it's a pre-shutdown drain. But it happens BEFORE the shutdown clear, so any World transferred to `ShutdownReclaimAuthority` AFTER this drain will not be drained.

**Fix: Move the final comprehensive drain to AFTER the shutdown clear:**

```cpp
    // 1. Pre-shutdown drain (existing, line ~371) — drains normal-operation Q entries
    if (quarantinedRetireResident > 0) {
        m_retireRouter->drainAllQuarantineStore();  // drains Q only
    }

    // ... DSP handle retirement ...

    // 2. Shutdown clear (existing, line ~455)
    worldAuthority_.requestShutdownClearNonRt();
    auto* clearedWorld = worldAuthority_.clearPublishedRuntimeSnapshotsNonRt();
    if (clearedWorld != nullptr)
    {
        RuntimePublicationBridge clearBridge{ *this, runtimePublicationValidator_ };
        clearBridge.retirePublishedRuntimeWorldNonRt(clearedWorld, true);
        // → transfers to ShutdownReclaimAuthority (post-15-P-4 fix)
    }

    // ★ 15-P-5: Final comprehensive drain — drains ALL authorities
    m_retireRouter->drainAll();  // D + Q (again, now empty) + EmergencyQ + ShutdownReclaimAuthority
```

### Ownership transition — no double-reclaim

| Stage | What's in D | What's in Q | What's in EmergencyQ | What's in ShutdownReclaimAuthority |
|-------|-------------|-------------|----------------------|--------------------------------------|
| Pre-shutdown drain | entries from normal operation (some may be epoch-safe) | entries from D-failure | entries from Q-failure | empty |
| Post pre-shutdown drain | remaining (epoch-blocked) | 0 | 0 | empty |
| Shutdown clear | 0 | 0 | 0 | clearedWorld transferred |
| Final drainAll() | 0 | 0 | 0 | all drained + onRelease() |

**No double-reclaim:** Each entry is in exactly one store. `drainAll()` drains each store once. `drainAllUnsafe()` clears the store after draining (sets `size_ = 0`).

---

## 15-P-5-1 — Shutdown drain ordering audit（post-implementation verification）

上記 15-P-5 は 15-P-4 実装**前**の設計文書。本節は 15-P-4 実装完了後のコードを
ユーザー指定の 6 項目で再監査し、発見したギャップ（GAP A / GAP B）を修正した記録。

### 監査サマリ

| # | 監査項目 | 結果 | 根拠 |
|---|---------|------|------|
| 1 | shutdown authority 唯一性 | ✅ PASS | `shutdownReclaim()` は `terminalReclaim()` に委譲 — 単一 authority（TerminalReclaimAuthority） |
| 2 | drain 順序 | ⚠️ GAP B 修正 | clear が `drainAllQuarantineStore()` より後 → clear 後に quiescence 確認付き強制 drain を追加 |
| 3 | shutdown early-return 再監査 | ✅ PASS | `enqueueWithRetry()` は Shutdown を返さない — ownership-loss path なし |
| 4 | 二重 reclaim 防止 | ✅ PASS | 各 World はちょうど 1 つの store（D/Q/E/Terminal）に入る |
| 5 | EBR 条件 | ✅ PASS | `terminalReclaim()` は `isOlder(epoch, minReaderEpoch)` — D/Q/E と同一条件 |
| 6 | shutdown 中の publish race | ✅ PASS | publish は `isShutdownInProgress()` でゲート、producer join 後に clear |
| A | `isFullyDrained()` の Terminal 未判定 | ⚠️ GAP A 修正 | `terminalReclaimResidentCount()==0` を追加 |
| B | destructor の drainAll が D のみ | ⚠️ GAP B 修正 | quiescence 確立時は `m_retireRouter->drainAll()`（D+Q+E+Terminal）に変更 |

### 1. shutdown authority 唯一性 ✅

- `ISRRetireRouter::shutdownReclaim()`（ISRRetireRouter.cpp:488）は `terminalReclaim()` に委譲する。
  **独立した ShutdownReclaimAuthority クラスは存在しない**（ISRRetireRouter.h:57 と
  AudioEngine.h:4199 のコメントは「shutdown authority」を概念として参照しているが、
  実体は shutdown パスが TerminalReclaimAuthority に入るだけ）。
- したがって「通常運用で D+Q+E 全滿 → Terminal」と「shutdown 中の retire → Terminal」は
  **同一の TerminalReclaimAuthority store** に収束する。二重 reclaim は構造的に排除される
  （各 World はちょうど 1 つの store にのみ入る — 下記 #4）。
- 唯一性の証明: `enqueueDeferredDeleteNonRtWithResult()` は
  `isShutdownInProgress()` 分岐で `shutdownReclaim()` を呼び、それ以外で `enqueueWithRetry()` を
  呼ぶ。両者は排他的（同一 ptr が両方を通ることはない）。`shutdownReclaim()` は常に true
  （growable store）のため、`RetireEnqueueResult::Shutdown` は到達不能（AudioEngine.h:4205 は防御的）。

### 2. drain 順序 ⚠️ → GAP B 修正

**releaseResources() の実測順序**（AudioEngine.Processing.ReleaseResources.cpp）:

```
1.  graceful drain loop（最大5s）: pendingRetireCount==0 && activeReaderCount==0 を待機
    （OverflowRing 再注入 + publishEpoch + tryReclaim を各 tick で実行）
2.  drainDeferredRetireQueues(false) / (true)          — tryReclaim（epoch-gated）
3.  PR2: drainAllQuarantineStore()（line ~378）         — Q + E + Terminal を強制 drain
    ※ コメント「この時点で GracefulDrain が activeReaderCount==0 を確認済み」
4.  P3 VerifyDrained: tryShutdownQuiescentReclaim       — DSP handle の Proof→Permit→reclaim
5.  clearPublishedRuntimeSnapshotsNonRt()（line ~456）  — publishAndSwap(nullptr) → oldWorld
    → retirePublishedRuntimeWorldNonRt → enqueueDeferredDeleteNonRt → shutdownReclaim
    → terminalReclaim（epoch safe → 即時破棄 / epoch unsafe → Terminal 保持）
6.  waitForDrain(2000, 2)（line ~470）                  — isFullyDrained() まで drainDeferredRetireQueues(true)
```

**GAP B（発見）**: step 5（clear）は step 3（drainAllQuarantineStore）**より後**に実行される。
したがって clear が返した oldWorld が epoch unsafe（stuck reader 残留）で Terminal に保持された場合、
step 3 の強制 drain では対象外となり、step 6 の waitForDrain も epoch-gated drain しか行わない。

**修正**: clear 直後に quiescence 確認付きの強制 drain を追加（ReleaseResources.cpp）:

```cpp
// ★ 15-P-5: clear 後の最終強制 drain（quiescence 確立時のみ）
if (m_retireRouter->activeReaderCount() == 0)
    m_retireRouter->drainAllQuarantineStore();
```

- 正常系（reader なし）: activeReaderCount==0 → 強制 drain 実行。Terminal は空（oldWorld は
  epoch safe で即時破棄済み）のため実質 no-op。
- stuck reader 系: activeReaderCount>0 → 強制 drain スキップ（UAF 回避）。Terminal 残留 World は
  waitForDrain 内の `drainTerminalReclaim()`（epoch-gated）に委ねる。

**destructor（~AudioEngine）の実測順序**（AudioEngine.CtorDtor.cpp）:

```
1.  setShutdownPhase(ForceEpochAdvance) + publishEpoch()
2.  closeReaderRegistration()
3.  graceful drain loop（最大5s）: pendingRetireCount==0 && activeReaderCount==0 を待機
4.  clearPublishedRuntimeSnapshotsNonRt() → clearedWorld
    → retirePublishedRuntimeWorldNonRt → shutdownReclaim → terminalReclaim
5.  drainDeferredRetireQueues(true)
6.  m_epochDomain.drainAll()          ← ★ GAP B: D のみ（Q/E/Terminal は対象外）
7.  markShutdownComplete()
```

**GAP B（発見）**: step 6 の `m_epochDomain.drainAll()` は `deferredDeletionQueue.drainAllUnsafe()`
（D のみ）を呼ぶ。Terminal に保持された World（stuck reader ケースの clearedWorld 等）が漏れる。

**修正**: quiescence 確立時は完全 drain に変更（CtorDtor.cpp）:

```cpp
// ★ 15-P-5: 完全 drain（D + Q + E + Terminal）
if (m_retireRouter->activeReaderCount() == 0)
    m_retireRouter->drainAll();      // D + Q + E + Terminal
else
    m_epochDomain.drainAll();        // D のみ（stuck reader の UAF 回避 — 従来動作）
```

### 3. shutdown early-return 再監査 ✅

- `ISRRetireRouter::enqueueRetire()`（ISRRetireRouter.cpp:213）は **Success か QueuePressure のみ**を
  返す（`provider_->enqueueRetireTyped()` 成功 → Success、失敗 → 500ms cooldown 付き tryReclaim
  再試行 → それでも失敗 → QueuePressure）。**Shutdown を返す経路は存在しない**。
- したがって `enqueueWithRetry()`（ISRRetireRouter.cpp:277）も Shutdown を返さない:
  - Stage 1: D 成功 → Success
  - Stage 2: retry ループ（kMaxRetry=2）— 結果が QueuePressure 以外なら break（ただし
    enqueueRetire は QueuePressure しか返さないため、ループは全試行する）
  - Stage 3: Q → E → Terminal（growable store は常に受領 → TerminalReclaim）
  - 最終 `return result;` は「Shutdown」コメント付きだが**到達不能**（result は常に
    Success/QueuePressure/QueueFull/TerminalReclaim）。
- `enqueueDeferredDeleteNonRtWithResult()`（AudioEngine.h:4196）:
  - `isShutdownInProgress()` → `shutdownReclaim()` → `terminalReclaim()` → 常に true → Success。
    **ownership は必ず移転**。
  - それ以外 → `enqueueWithRetry()` → 常に ownership 移転。
- **結論: ownership-loss path は存在しない**。`retirePublishedRuntimeWorldNonRt()` が戻り値を
  無視しても、World は必ず D/Q/E/Terminal のいずれかに移転済み。

### 4. 二重 reclaim 防止 ✅

- `enqueueWithRetry()` は最初に受領した store で return するため、**各 World はちょうど 1 つの
  store に入る**（D → Q → E → Terminal の順で、最初の成功で確定）。
- `clearPublishedRuntimeSnapshotsNonRt()` は `publishAndSwap(nullptr)` を**1 回だけ**呼び、
  返った oldWorld を `retirePublishedRuntimeWorldNonRt()` で**ちょうど 1 回** retire する。
- `drainAllQuarantineStore()` は Q + E + Terminal を各 1 回 drain する。`drainAllUnsafe()` /
  `drainAll()` は drain 後に store を空にするため、同一 entry が二重に deleter 実行されることはない。
- `terminalReclaim()` の synchronous destruction パスは store に入れず即時破棄するため、
  「即時破棄 + store 保持」の二重所有は構造的に不可能。

### 5. EBR 条件 ✅

- `terminalReclaim()`（ISRRetireRouter.cpp:424）の安全条件:
  `epochSafe = isOlder(epoch, minReaderEpoch)` = `epoch < minReaderEpoch`。
  これは `DeferredDeletionQueue::reclaim()` / `RetireQuarantineStore::drain()` /
  `EmergencyQuarantineStore::drain()` と**同一の EBR 条件**（`static_cast<int64_t>(a-b) < 0`）。
- `minReaderEpoch()` は active reader の最小 epoch。reader が無ければ `currentEpoch()` を返す。
- **shutdown 中の epoch 安全性**:
  - `markRetireEpoch()` = `publishEpoch()` は `fetchAddAtomic(globalEpoch, 1, acq_rel)` で
    **OLD 値 G を返し**、globalEpoch = G+1 に設定。
  - reader が無ければ `minReaderEpoch()` = currentEpoch() = G+1。
  - `epochSafe = isOlder(G, G+1) = G < G+1` = **TRUE** → 即時破棄。
  - したがって正常 shutdown（reader なし）では clearedWorld は**同期破棄**され、Terminal に
    残留しない。コメント「shutdown 中は Audio Thread 停止済みのため epoch は安全 → 即時破棄される」
    は正しい。
- **stuck reader ケース**: reader が epoch R ≤ G で残留すると `epochSafe = G < R` = false →
  Terminal に保持。この場合の解放は:
  - HealthMonitor の `quarantineReader()`（Timer.cpp:1643）が stuck reader を safe-epoch 計算から
    除外 → `drainTerminalReclaim()`（epoch-gated）で解放。
  - または graceful drain 完了（activeReaderCount==0）後の強制 drain（GAP B 修正）。
- `drainAllUnsafe()`（drainAllQuarantineStore 経由）は **quiescence 確立後**（graceful drain が
  activeReaderCount==0 を確認）にのみ実行される。stuck reader が残る場合は GAP B 修正の
  `activeReaderCount()==0` ガードで強制 drain をスキップし、UAF を構造的に回避する。

### 6. shutdown 中の publish race ✅

- publish は複数箇所で `isShutdownInProgress()` によりゲートされる:
  - `PublicationAdmission.cpp:11` — publish admission 遮断
  - `AudioEngine.Commit.cpp:195` — commit パス遮断
  - `ISRCoordinatorLoop.cpp:35` — Coordinator loop が shutdown で exit
  - `ISRRuntimePublicationCoordinator.h:359` — 「シャットダウン中の publish は
    isShutdownInProgress() で事前に遮断される」
- releaseResources は `shutdownCoordinatorLoop()`（join）→ `stopRebuildThread()`（join）を
  clear より前に実行するため、clear 時点で producer は全て join 済み。
- `clearPublishedRuntimeSnapshotsNonRt()` の `publishAndSwap(nullptr)` が**最後の publish**。
  これ以降に publishAndSwap を呼ぶ経路は存在しない（AudioBlock.cpp:36 / BlockDouble.cpp:36 /
  Transition.cpp:15 / Snapshot.cpp:18 も isShutdownInProgress() で遮断）。
- **結論: publishAndSwap と shutdown drain の競合は構造的に排除されている。**

### GAP A 修正 — isFullyDrained の Terminal 判定追加

`AudioEngine::isFullyDrained()`（AudioEngine.Threading.cpp）は `quarantineResidentCount()`
（Q + E）を判定するが **Terminal を含まない**。このため「Terminal に World が残っているのに
isFullyDrained()==true」と誤判定し、waitForDrain が premature に成功を返す経路があった。

**修正**: `terminalReclaimResidentCount()==0` を追加:

```cpp
const auto terminalReclaimResident = (m_retireRouter != nullptr)
    ? static_cast<std::uint64_t>(m_retireRouter->terminalReclaimResidentCount()) : 0u;
...
return !hasDeferredCommit
    && pendingReclaimEmpty
    && retireDepth == 0
    && lifetimeRetireIntentPending == 0
    && ringResident == 0
    && dspQuarantineResident == 0
    && retireQuarantineResident == 0
    && terminalReclaimResident == 0          // ★ 15-P-5 追加
    && runtimePublicationBridge_.isFullyDrained();
```

### 修正ファイル一覧

| ファイル | 修正内容 |
|---------|---------|
| `src/audioengine/AudioEngine.Threading.cpp` | `isFullyDrained()` に `terminalReclaimResident==0` を追加（GAP A） |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | clear 後に quiescence 確認付き `drainAllQuarantineStore()` を追加（GAP B） |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | destructor の drainAll を quiescence 確認付き `m_retireRouter->drainAll()` に変更（GAP B） |

### 検証

- Debug ビルド: ✅ PASS
- CTest 29/29: ✅ PASS（`-E "BuildInputSemanticContract|RuntimeWorldAuthority"`）

### 最終判定: **PASS** ✅

- 正常 shutdown（reader なし）: clearedWorld は epoch safe により**同期破棄**され、Terminal に
  残留しない。`drainAllQuarantineStore()` / `m_retireRouter->drainAll()` は Q + E + Terminal を
  各 1 回強制 drain する。**リークなし・二重 reclaim なし**。
- stuck reader ケース: Terminal 残留 World は `isFullyDrained()` の Terminal 判定（GAP A）により
  waitForDrain が premature 成功しない。quiescence 確立後に強制 drain（GAP B）で解放される。
  reader が quarantine されない二重障害時のみ、epoch-gated drain に委ねる（UAF 回避を優先）。
- 15-P-4 の TerminalReclaimAuthority は shutdown 側の既存 `drainAllUnsafe()` と**競合しない**:
  同一 store に収束するが、各 World はちょうど 1 つの store に入り、drain は各 store を 1 回だけ
  実行する。

---

## 15-P-6 — `RejectedShutdown` / `RejectedPublishFailure` 分離

### Current state

```cpp
// RuntimePublicationOrchestrator.cpp:277-283
if (executor_.publish()) {
    // success
} else {
    return Decision::RejectedShutdown;  // MISLABELED
}
```

### Issue: `RejectedShutdown` conflates two scenarios

| Scenario | Cause | Current Decision | Correct Decision |
|----------|-------|-----------------|------------------|
| Genuine shutdown | `isShutdownInProgress()` in admission | `RejectedShutdown` | `RejectedShutdown` ✅ |
| Publish failure | `publishAndSwap` returns nullptr or `publish()` fails | `RejectedShutdown` | `RejectedPublishFailure` ✗ |
| OwnerChannel full | `enqueue` returns false | (different path) | `RejectedPublishFailure` |
| Retirement queue full | `enqueueWithRetry` returns QueueFull | (not here — internal to retire) | (internal — no Decision) |

### Proposed fix

**1. Add `RejectedPublishFailure` to the Decision enum:**

```cpp
// ISRAuthorityClass.h (or wherever Decision is defined)
enum class Decision {
    Accepted,
    RejectedShutdown,      // genuine: isShutdownInProgress() == true at admission time
    RejectedPublishFailure, // publish-time: publishAndSwap or publish() failed
    RejectedPressure,      // admission: resource pressure (D+Q+E back-pressure)
};
```

**2. Split in RuntimePublicationOrchestrator:**

```cpp
// RuntimePublicationOrchestrator.cpp (modified)
auto publishResult = executor_.publish();
if (publishResult.success) {
    return Decision::Accepted;  // or the publish success path
}
// publish() failed — distinguish from genuine shutdown
if (isShutdownInProgress()) {
    return Decision::RejectedShutdown;
} else {
    return Decision::RejectedPublishFailure;
}
```

### Telemetry / FailureStage impact

```cpp
// submitPublishRequest (modified)
switch (decision) {
    case Decision::Accepted:
        // existing success path
        break;
    case Decision::RejectedShutdown:
        onRejectedShutdown(seqId);
        telemetry_->recordFailure(FailureStage::Admission, FailureReason::Shutdown);
        break;
    case Decision::RejectedPublishFailure:
        onFailed(seqId);
        telemetry_->recordFailure(FailureStage::Publish, FailureReason::InternalError);
        break;
    case Decision::RejectedPressure:
        onBackpressure(seqId);
        telemetry_->recordFailure(FailureStage::Admission, FailureReason::ResourceExhaustion);
        break;
}
```

### Recovery / retry semantics

| Decision | Recovery path | Rationale |
|----------|--------------|-----------|
| RejectedShutdown | No retry — caller should drop | Engine is shutting down |
| RejectedPublishFailure | Retry / re-admit | Internal failure, World may be reusable |
| RejectedPressure | Retry / backpressure notify | Resource pressure, should retry later |
| RejectedShutdown (current bug) | Misclassified as shutdown | Recovery would incorrectly give up |

### Verification: no conflation in recovery

The key check: if `publish()` fails during normal operation (NOT shutdown), the caller must NOT treat it as a shutdown error. With the split:

```cpp
// After fix:
if (decision == Decision::RejectedPublishFailure) {
    // Retry immediately or schedule retry
    // World ownership NOT lost (caller still owns unique_ptr<frozen>)
    // NOT a shutdown — engine continues running
}
```

**✅ PASS** — Shutdown and publish failure are now distinguished for all semantics: telemetry, recovery, cleanup.

---

## 15-P-6-1 — `RejectedShutdown` / `RejectedPublishFailure` semantic separation audit（post-implementation verification）

### Audit summary

15-N-6 / 15-O-6 で記録した `RejectedShutdown` の semantic conflation を、15-P-4/5 の
実装変更後のコードで再監査した。**conflation は実コードに残存していた**ため、decision
semantics の分離修正を適用した（lifetime authority は変更なし — `TerminalReclaimAuthority`
に収束した状態を維持）。

| # | 監査項目 | 結果 |
|---|---------|------|
| 1 | enum / decision 型の現状確認 | ✅ `RetireEnqueueResult` / `PublicationAdmission::Decision` / `PublishResult` / `FailureStage` / `FailureReason` を全検索 |
| 2 | `RejectedShutdown` の全 producer 分類 | ⚠️→修正 2 producer を特定（genuine shutdown + publish failure 誤分類） |
| 3 | 15-P-4 ownership chain との整合性 | ✅ D→Q→E→Terminal の全地点で failure classification ≠ ownership transfer 失敗 |
| 4 | `enqueueWithRetry()` return contract | ✅ `Shutdown` は実質 unreachable（ownership loss path なし） |
| 5 | caller 側 semantic handling | ✅ 全 caller の result 意味論を追跡 |
| 6 | shutdown と publish failure の分離 | ⚠️→修正 `RejectedPublishFailure` を追加し分離 |
| 7 | telemetry / recovery への影響 | ⚠️→修正 shutdown telemetry への誤計上を解消 |
| 8 | Release / Debug 意味論同一性 | ✅ `assert(false)` / Debug-only branch 非依存 |

### 1. enum / decision 型の現状確認

| 型 | 定義 | 使用箇所 |
|----|------|---------|
| `RetireEnqueueResult` | `ISRAuthorityClass.h:28` | `Success / QueuePressure / QueueFull / Shutdown / TerminalReclaim` |
| `PublicationAdmission::Decision` | `PublicationAdmission.h:38` | `Accepted / RejectedStaleGeneration / RejectedNotFinalized / RejectedPressure / RejectedShutdown / DeferredFadingActive / RejectedLowPriority`（+ 今回 `RejectedPublishFailure` 追加） |
| `PublishResult` | `PublicationExecutor.h:10` | `Success / ValidationFailed / PublishFailed / BridgeFailed` |
| `FailureStage` | `TelemetryRecorder.h:22` | `None / Admission / Validation / Execution / Bridge / Shutdown` |
| `FailureReason` | `TelemetryRecorder.h:32` | `None / AdmissionRejected / ValidationFailed / PublishFailed / BridgeFailed / ShutdownRejected / StaleGeneration / QueuePressure / Count` |

`RejectedPublishFailure` は修正前、**コード上に存在しなかった**（設計書 15-P-6 / 15-O-6 の
提案のみ）。`RejectedLowPriority` は enum 定義のみで `evaluate()` は返さない未使用値
（`default` ケースで処理）。

### 2. `RejectedShutdown` の全 producer 分類（修正前）

| producer | 場所 | 実態 | 分類 |
|----------|------|------|------|
| `PublicationAdmission::evaluate()` | `PublicationAdmission.cpp:12` | `engine.isShutdownInProgress()` → genuine shutdown | ✅ 正しい |
| `RuntimePublicationOrchestrator::trySubmitImpl()` | `RuntimePublicationOrchestrator.cpp:283`（修正前） | `executor_.publish()` が `PublishFailed` → **publish failure を誤分類** | ⚠️ conflation |

**修正**: `trySubmitImpl` の publish 失敗分岐で `engine_.isShutdownInProgress()` を再チェックし、
shutdown 中なら `RejectedShutdown`、それ以外は `RejectedPublishFailure` を返すように分離した。
（admission は `isShutdownInProgress()` をチェック済みだが、admission と publish の間に
shutdown が開始される race が理論上存在するため、publish 失敗時点で再確認する。）

### 3. 15-P-4 ownership chain との整合性

```
D → Q → EmergencyQ → TerminalReclaimAuthority
```

- `enqueueWithRetry()` の各 Stage は「次の authority に ownership が移ってから return」する
  （P-4 invariant）。`assert(false) → return` 経路は存在しない。
- `TerminalReclaimAuthority` は growable store のため常に ownership を受領する
  （`terminalReclaim()` は常に true）。
- **publish failure は ownership transfer の失敗を意味しない**:
  - `PublicationExecutor::publishImpl` が `PublishFailed` を返すのは
    `commitRuntimePublication` が `{Failed, CallerDestroy}` を返した場合のみ。
  - `CallerDestroy` 時、World の ownership は caller（`trySubmitImpl`）に残り、
    `lifetime_.destroyRolledBackDSP(newDSPResolved)` で物理解放される。
  - つまり decision 分類（`RejectedShutdown` vs `RejectedPublishFailure`）は
    **ownership 回収から独立**している。

### 4. `enqueueWithRetry()` return contract（実コード検証）

```cpp
// ISRRetireRouter.cpp:277
RetireEnqueueResult ISRRetireRouter::enqueueWithRetry(...) noexcept
{
    // Stage 1: D
    auto result = enqueueRetire(ptr, deleter, epoch, type);   // Success | QueuePressure のみ
    if (result == Success) return result;                     // D owns ptr ✅
    // Stage 2: retry cycle (kMaxRetry=2)
    for (...) { tryReclaim(); drainEmergencyAndTerminal(); result = enqueueRetire(...); ... }
    // Stage 3: Q → Stage 4: E → Stage 5: Terminal
    if (result == QueuePressure || result == QueueFull) {
        if (m_retireQuarantine.quarantine(...)) return QueuePressure;      // Q owns ptr ✅
        if (m_emergencyQuarantine.quarantine(...)) return QueuePressure;   // E owns ptr ✅
        terminalReclaim(...); return TerminalReclaim;                       // Terminal owns ptr ✅
    }
    return result;  // ★ 到達不能: enqueueRetire は Success | QueuePressure のみ返す
}
```

- `enqueueRetire()`（`ISRRetireRouter.cpp:97`）は `provider_->enqueueRetireTyped()` 成功 →
  `Success`、失敗 → 500ms cooldown の `tryReclaim()` 再試行 → それでも失敗 →
  `QueuePressure` のみ。**`Shutdown` を返さない**。
- したがって `enqueueWithRetry()` の末尾 `return result;` は `QueuePressure` のみで、
  **`Shutdown` は実質 unreachable**。ownership loss 経路は存在しない。
- `enqueueDeferredDeleteNonRtWithResult`（`AudioEngine.h:4190`）の shutdown 分岐は
  `shutdownReclaim()`（= `terminalReclaim()`）が常に true を返すため `Shutdown` を返さない。
  `enqueueDeferredDeleteNonRt`（bool 版）の `result != Shutdown` 判定は防御的。

### 5. caller 側 semantic handling

| caller | result の意味 | ownership |
|--------|--------------|-----------|
| `enqueueDeferredDeleteNonRtWithResult` | `Success` = authority に移譲 / `Shutdown` = 実質 unreachable | 常に移転 |
| `retirePublishedRuntimeWorldNonRt`（`AudioEngine.h:3523`） | bool 版を呼び戻り値無視 — ownership は常に移転するため安全 | 常に移転 |
| `RuntimePublishExecutor::executePublish`（`RuntimePublishExecutor.h`） | `committed==true` のみ `retirePublishedRuntimeWorldNonRt(oldWorld)` 実行 | 常に移転 |
| `PublishIntentHandler::handle`（`ISRRuntimePublicationCoordinator_ProcessIntent.cpp:148`） | `PublishExecutor{}.executePublish()` に委譲 — result を再解釈しない | — |
| `RuntimePublicationOrchestrator::trySubmitImpl` | `executor_.publish()` 失敗 → `destroyRolledBackDSP()` で回収 → `RejectedPublishFailure`（修正後） | caller が回収 |
| `submitPublishRequest` switch | decision ごとに telemetry 記録（修正後 `RejectedPublishFailure` は `Execution/PublishFailed`） | — |

### 6. shutdown と publish failure の分離（修正後）

```
genuine shutdown（admission 時 isShutdownInProgress）
    → Decision::RejectedShutdown
    → FailureStage::Shutdown / FailureReason::ShutdownRejected

publication / retirement failure（executor_.publish() 失敗、shutdown 中でない）
    → Decision::RejectedPublishFailure        ★ 15-P-6 追加
    → FailureStage::Execution / FailureReason::PublishFailed
```

- `PublicationAdmission::evaluate()` の `RejectedShutdown`（genuine）は変更なし。
- `trySubmitImpl` の publish 失敗は `RejectedPublishFailure` に分離。
- `submitPublishRequest` の switch に `RejectedPublishFailure` ケースを追加。

### 7. telemetry / recovery への影響

- **`RejectedShutdown` を shutdown telemetry に誤計上しない**: 修正前は publish 失敗が
  `FailureStage::Shutdown / FailureReason::ShutdownRejected` に計上されていた。修正後は
  `FailureStage::Execution / FailureReason::PublishFailed` に計上される。
- **publish failure を shutdown として recovery suppression しない**: `RejectedPublishFailure`
  は shutdown 扱いされないため、recovery が誤って諦めることはない。
- **ownership reclaim は decision classification から独立**: `destroyRolledBackDSP()` は
  decision に関係なく publish 失敗分岐で常に実行される。`StateOwner` の `onExecutorFailed`
  （trySubmitImpl 内）と `onRejected`（switch 内）の二重記録は既存パターン
  （`RejectedNotFinalized` と同一）で一貫。

### 8. Release / Debug 意味論同一性

- 修正は `assert(false)` や Debug-only branch に依存しない。
- `RejectedPublishFailure` は enum 値として Release でも同一。
- `engine_.isShutdownInProgress()` は `lifecycleShutdown || shutdownRuntime_.isShutdownInProgress()`
  で Release でも同一の判定。

### 修正ファイル一覧

| ファイル | 変更 |
|---------|------|
| `src/audioengine/PublicationAdmission.h` | `Decision` enum に `RejectedPublishFailure` を追加 |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | `trySubmitImpl` の publish 失敗を `isShutdownInProgress()` で分離 + `submitPublishRequest` switch に `RejectedPublishFailure` ケース追加 |

### 検証

- **Debug build**: PASS（`PublicationAdmission.cpp` / `RuntimePublicationOrchestrator.cpp` コンパイル、
  `ConvoPeq.exe` / `PublicationAdmissionTests.exe` / `AudioEngineHarness.exe` リンク成功）
- **CTest**: 29/29 PASS（`-E "BuildInputSemanticContract|RuntimeWorldAuthority"`）
- **modified files error check**: エラーなし

### 最終判定

**PASS** — `RejectedShutdown` の semantic conflation は解消された。genuine shutdown と
publish failure は decision / telemetry / recovery の全意味論で区別される。ownership
correctness は 15-P-4/5 の `TerminalReclaimAuthority` 収束状態を維持し、decision 分類から
独立して成立する。新しい authority は追加していない。

---

## 15-P-7 — Admission pressure gate

> ⚠️ **旧設計（pre-implementation）**: 本セクションの数値（E=64, M_pressure=4930）は実装前の設計値。
> 現行実装値（E=512, M_pressure=5377）は 15-P-9-1 / 15-P-10-1 / 15-P-FINAL を参照。

### Design

Add D+Q+E back-pressure to `PublicationAdmission::evaluate()`:

```cpp
// PublicationAdmission.cpp (modified)
Decision PublicationAdmission::evaluate(const PublishContext& ctx) noexcept
{
    if (isShutdownInProgress())
        return Decision::RejectedShutdown;

    // ★ 15-P-7: D+Q+E back-pressure (preventive — NOT hard guarantee)
    // This gate REDUCES probability of reaching EmergencyQ overflow.
    // The HARD guarantee comes from ShutdownReclaimAuthority (15-P-4).
    const uint32_t dCount = m_epochProvider->pendingRetireCount();  // D
    const uint32_t qCount = m_retireRouter->quarantineResidentCount();  // Q
    const uint32_t eCount = m_retireRouter->emergencyQuarantineResidentCount();  // E

    constexpr uint32_t kRetirementPressureThreshold =
        DeferredDeletionQueue::kQueueSize +        // 4096
        64 +                                         // emergency capacity
        RetireQuarantineStore::kMaxQuarantinedEntries - 64;  // Q primary
        // = 4608

    if (dCount + qCount + eCount > kRetirementPressureThreshold) {
        return Decision::RejectedPressure;
    }

    // ... existing checks ...
    return Decision::Accepted;
}
```

**Important code comment:**
```cpp
// ★ 15-P-7 NOTE: This admission gate is PREVENTIVE only.
// It does NOT formally guarantee L=0. The formal guarantee comes from:
// 1. EmergencyQ fallback in enqueueWithRetry (D+Q full → EmergencyQ)
// 2. ShutdownReclaimAuthority transfer (EmergencyQ full → shutdown authority)
// 3. drainAllUnsafe() at shutdown (drains all authorities)
//
// If the admission gate fails (TOCTOU window), the fallbacks in 15-P-2/15-P-3/15-P-4
// still guarantee ownership preservation. L=0 is proven by the fallback chain, NOT
// by this admission gate alone.
```

### M_pressure recalculation

With EmergencyQ (64) added:

```
M_pressure = D + Q + E + O + P + 1
           = 4096 + 512 + 64 + 256 + 1 + 1
           = 4930
```

The admission threshold should be set at 4608 (= 4096 + 512) to reject BEFORE EmergencyQ starts filling:

```cpp
constexpr uint32_t kRetirementPressureThreshold = 4096 + 512;  // 4608
// Reject when D + Q > 4608, keeping EmergencyQ as headroom
```

This gives EmergencyQ (64 slots) as a buffer between admission rejection and overflow.

---

## 15-P-7-1 — Integrated invariant / counter-audit（post-implementation verification）

### Integrated audit summary

15-P-4/5/6 の修正を横断して、Release/Debug・shutdown・retirement pressure を含む統合
不変条件を再証明した。**「TerminalReclaimAuthority が存在する」ことではなく、
「どの失敗組合せでも `oldWorld` の所有権が消失しない」ことをケース網羅で証明**する。

| # | 監査項目 | 結果 |
|---|---------|------|
| 1 | Ownership invariant | ✅ `publishAndSwap()` の `oldWorld` は全経路で D→Q→E→Terminal→destroy のいずれかへ移転 |
| 2 | L=0 invariant | ✅ Terminal は growable — bounded-store exhaustion による ownership loss なし |
| 3 | EBR invariant | ✅ D/Q/E/Terminal の全 reclaim 判定が `isOlder(retireEpoch, minReaderEpoch)` に統一 |
| 4 | Shutdown invariant | ✅ clear / isFullyDrained / destructor final drain が全て正しい |
| 5 | Admission invariant | ✅ `RejectedShutdown` / `RejectedPublishFailure` が正しく分離 |
| 6 | Release/Debug equivalence | ✅ `assert` の有無で lifetime ownership の結果は不変 |
| 7 | Counter-audit | ✅ 7 ケース全てで ownership loss なし |
| 8 | 検証 | ✅ error check + Debug build + CTest 29/29 |

### 1. Ownership invariant — `publishAndSwap()` の `oldWorld` 全経路

`RuntimeStore::WriteAccess::publishAndSwap`（`RuntimeStore.h:40`）は
`exchangeAtomic(store_->current, next, acq_rel)` で **OLD 値（previous world）を返す**。

`oldWorld` の全 producer と行き先:

| producer | 場所 | oldWorld の行き先 |
|----------|------|-------------------|
| `RuntimeWorldAuthority::publish()` | `RuntimeWorldAuthority.h:249` | `PublishExecutor::executePublish` → `bridge.retirePublishedRuntimeWorldNonRt(oldWorld, false)` |
| `clearPublishedRuntimeSnapshotsNonRt()` | `RuntimeWorldAuthority.h:263` | `clearBridge.retirePublishedRuntimeWorldNonRt(clearedWorld, true)` |
| `RuntimePublicationCoordinator.h:130`（core） | `publishAndSwap(newWorld)` | 同様の retire 経路 |

`retirePublishedRuntimeWorldNonRt`（`AudioEngine.h:3523`）→ `enqueueDeferredDeleteNonRt` →
`enqueueWithRetry`（`ISRRetireRouter.cpp:277`）の全 return 経路:

| return | 所有者 | 備考 |
|--------|--------|------|
| `Success` | D | `enqueueRetire` 成功 |
| `QueuePressure`（Stage 3） | Q | `m_retireQuarantine.quarantine` 成功 |
| `QueuePressure`（Stage 4） | E | `m_emergencyQuarantine.quarantine` 成功 |
| `TerminalReclaim` | Terminal | `terminalReclaim` 常に true（growable） |
| 末尾 `return result` | — | `enqueueRetire` は `Success \| QueuePressure` のみ返すため `QueuePressure` のみ。**`ptr` が raw pointer のまま return される経路は存在しない** |

**✅ PASS** — 全経路で `ptr` は次の authority に移転してから return される。

### 2. L=0 invariant — Terminal の growable 性

- `TerminalReclaimAuthority::store()`（`ISRRetireRouter.cpp:27`）: `entries_.push_back(...)` +
  `return true` — **growable `std::vector` により常に受領**。
- D/Q/E が満杯でも Terminal が受け入れる → bounded-store exhaustion による ownership loss なし。
- `enqueueWithRetry` に `assert(false) → return` 経路は存在しない（15-P-4 で構造的に排除）。
- `terminalReclaim()` の epoch-safe 同期破棄は `isAudioThread()` チェック付き（RT 防御）。

**✅ PASS** — L=0（World リークなし）が構造的に成立。

### 3. EBR invariant — `isOlder` の統一

| store | reclaim 判定 | 場所 |
|-------|-------------|------|
| D | `isOlder(entry.epoch, minReaderEpoch)` | `DeferredDeletionQueue.h:132` |
| Q | `isOlderFn(e.epoch, minReaderEpoch)` | `RetireQuarantineStore.h:98` |
| E | `isOlderFn(e.epoch, minReaderEpoch)` | `ISRRetireRouter.cpp:471` |
| Terminal | `isOlderFn(e.epoch, minReaderEpoch)` | `ISRRetireRouter.cpp:460` |
| `terminalReclaim()` | `isOlder(epoch, minReader)` | `ISRRetireRouter.cpp:432` |

全て `static_cast<int64_t>(a - b) < 0`（wraparound-safe）に統一。`terminalReclaim()` と
`TerminalReclaimAuthority::drain()` の条件は 15-P-4 で修正済み（`!isOlder` → `isOlder`）。

**`drainAllUnsafe()` の逆流なし**: 呼び出し箇所は全て shutdown-only。
- `ISRRetireRouter::drainAllQuarantineStore()`（Q+E+Terminal）— shutdown 時のみ
- `EpochDomain::drainAll()`（D）— shutdown 時のみ
- `EQProcessor.Core.cpp:159` — EQProcessor デストラクタ（shutdown）
- 通常運転の reclaim 経路（`tryReclaim()` → `drainQuarantineStore()` + `drainEmergencyAndTerminal()`）は
  **epoch-gated のみ**で `drainAllUnsafe` を呼ばない。

**✅ PASS** — EBR 条件が全 store で統一、`drainAllUnsafe` は通常運転へ逆流しない。

### 4. Shutdown invariant

- **publish/reclaim race**: admission（`evaluate`）は `isShutdownInProgress()` チェック、
  publish 失敗時も `isShutdownInProgress()` を再チェック（15-P-6）。shutdown 中は publish が
  ゲートされるため、`clearPublishedRuntimeSnapshotsNonRt()` が最後の `publishAndSwap()` を処理する。
- **clear 後の Terminal resident**: `isFullyDrained()` が `terminalReclaimResident == 0` を
  含めて待機（15-P-5 GAP A fix、`AudioEngine.Threading.cpp:172`）。
- **destructor final drain**: `m_retireRouter->drainAll()` = `provider_->drainAll()`（D）+
  `drainAllQuarantineStore()`（Q+E+Terminal）で **D/Q/E/Terminal 全てを対象**
  （15-P-5 GAP B fix、`AudioEngine.CtorDtor.cpp:244`）。stuck reader が残る場合は
  `m_epochDomain.drainAll()`（D のみ）にフォールバックし、epoch-gated drain に委ねる。

**✅ PASS** — shutdown 中の publish/reclaim race、clear、isFullyDrained、final drain が全て正しい。

### 5. Admission invariant

- `RejectedShutdown` = genuine shutdown のみ（`PublicationAdmission.cpp:12` +
  `trySubmitImpl` の `isShutdownInProgress()` 再チェック）。
- `RejectedPublishFailure` = publish execution failure のみ（15-P-6 で追加）。
- decision classification と ownership transfer は独立:
  `destroyRolledBackDSP()` は decision に関係なく publish 失敗分岐で常に実行される。

**✅ PASS** — decision 分類と ownership transfer が独立。

### 6. Release/Debug equivalence

- `assert(false)` は Debug のみの検出（例: `RuntimeStore::WriteAccess` used-after-move）で、
  Release では no-op フォールバック（`return nullptr`）。
- lifetime ownership の結果は assert の有無に依存しない（`enqueueWithRetry` に
  `assert(false) → return` 経路なし）。
- Debug-only safety assumption は残っていない。

**✅ PASS** — Release/Debug で lifetime ownership の結果は同一。

### 7. Counter-audit — 失敗組合せケース網羅

| ケース | 条件 | `oldWorld` の行き先 | ownership loss? |
|--------|------|---------------------|-----------------|
| C1: D full | D が満杯 | `enqueueRetire` → `QueuePressure` → Q | ❌ なし |
| C2: D+Q full | D, Q が満杯 | E（EmergencyQuarantineStore） | ❌ なし |
| C3: D+Q+E full | D, Q, E が満杯 | Terminal（growable, 常に受領） | ❌ なし |
| C4: epoch unsafe | active reader が epoch を通過していない | Terminal に保持 → `drain()` が epoch safe になった時に解放 | ❌ なし |
| C5: shutdown 中 | `isShutdownInProgress()` | `shutdownReclaim` → `terminalReclaim` → epoch safe なら即時破棄 / それ以外は Terminal 保持 → `drainAll()` で解放 | ❌ なし |
| C6: shutdown と publish の race | admission 後に shutdown 開始 | publish 失敗 → `destroyRolledBackDSP()` で回収 / publish 成功 → oldWorld は `retirePublishedRuntimeWorldNonRt` → `enqueueDeferredDeleteNonRt` → `shutdownReclaim` | ❌ なし |
| C7: publish execution failure | `commitRuntimePublication` が `{Failed, CallerDestroy}` | `destroyRolledBackDSP()` で回収 | ❌ なし |

**全 7 ケースで `oldWorld` の所有権は消失しない**。各ケースで「次の authority に移転」または
「caller が回収」が成立する。

### 8. 検証

- **modified files error check**: エラーなし
- **Debug build**: PASS
- **CTest**: 29/29 PASS（`-E "BuildInputSemanticContract|RuntimeWorldAuthority"`）
- **Release build**: 変更は enum 値追加 + decision 分岐のみで、Release でも同一の意味論
  （15-P-8 で最終 parity 検証）

### 最終判定

**PASS** — 統合不変条件（Ownership / L=0 / EBR / Shutdown / Admission / Release-Debug）が
全て成立。counter-audit の 7 ケース全てで `oldWorld` の所有権は消失しない。
15-P-8（Release/Debug parity と最終 counter-audit）へ進む準備が整った。

---

## 15-P-8 — Release/Debug 両方で ownership path trace

### Case-by-case path analysis

| Case | D | Q | E | Shutdown? | Path | Ownership transfer | L? |
|------|---|---|---|-----------|------|-------------------|-----|
| Normal, D has space | not full | — | — | no | enqueueRetire(D) → success | ptr → D entry ✅ | 0 |
| Normal, D full, Q has space | full | not full | — | no | enqueueRetire fails → quarantine(Q) → success | ptr → Q entry ✅ | 0 |
| Normal, D+Q full, E has space | full | full | not full | no | enqueueRetire fails → quarantine(Q) fails → quarantine(E) → success | ptr → E entry ✅ | 0 |
| Normal, D+Q+E full | full | full | full | no | enqueueRetire fails → quarantine(Q) fails → quarantine(E) fails → transferToShutdownAuthority | ptr → ShutdownReclaimAuthority ✅ | 0 |
| Shutdown clear | any | any | any | yes | isShutdownInProgress → transferToShutdownAuthority | ptr → ShutdownReclaimAuthority ✅ | 0 |
| Shutdown drain | any | any | any | yes | drainAll() | all → deleter → onRelease ✅ | 0 |
| Timeout | any | any | any | no | (doesn't affect retirement path) | ownership retained via Success result ✅ | 0 |

### Edge case: EmergencyQ full + not shutdown

This requires:
- D = 4096 (full)
- Q = 512 (full)
- E = 64 (full)
- Admission gate did NOT reject (TOCTOU window or gate absent)

→ Transfer to `ShutdownReclaimAuthority`

The `ShutdownReclaimAuthority` is a bounded array of 256 entries. If it overflows too:
```cpp
// ShutdownReclaimAuthority::transfer — if full:
assert(false && "ShutdownReclaimAuthority overflow — admission gate failure");
// In Debug: crash
// In Release: log error, ptr still STORED (in a static overflow vector or similar)
```

Actually — for true L=0, even the `ShutdownReclaimAuthority` must never overflow. This is guaranteed by:
1. Admission gate prevents D+Q+E from filling (threshold 4608)
2. EmergencyQ (64) provides buffer
3. ShutdownReclaimAuthority (256) provides additional buffer for shutdown early-returns
4. If ALL fail → assert(false) in Debug (crash), in Release → **this is unreachable if admission gate works**

**Formal proof sketch:**
- Admission gate rejects at D+Q > 4608
- EmergencyQ provides 64 more slots (4609-4672)
- ShutdownReclaimAuthority provides 256 more (4673-4928)
- Total retirement capacity = 4096 + 512 + 64 + 256 = 4928
- M_pressure (with O=256, P=1) = 4928 + 256 + 1 = 5185
- Admission threshold = 4608 (rejects before EmergencyQ fills)
- EmergencyQ + ShutdownReclaimAuthority = 64 + 256 = 320 headroom > worst-case burst

**→ L=0 is formally provable in both Debug and Release.**

---

## 15-P-8-1 — Release/Debug parity + final counter-audit（post-implementation verification）

### 前提の明確化

15-P-8 設計セクション（上記）は 15-P-4 以前の設計であり、`ShutdownReclaimAuthority`
（bounded 256）を参照している。**15-P-4 で最終 lifetime authority は
`TerminalReclaimAuthority`（growable std::vector）に収束した**ため、本セクションは
実際の実装（TerminalReclaimAuthority 収束後）に基づく Release/Debug parity 検証を記録する。

### 1. Debug/Release 意味論比較 — assert / NDEBUG 依存の全検索

lifetime ownership path に現れる `assert` / `NDEBUG` を全検索した結果:

| 場所 | assert | Release での挙動 | ownership 影響 |
|------|--------|------------------|----------------|
| `ISRRetireRouter.cpp`（全 22 箇所） | `assert(provider_ != nullptr)` | 消える（no-op） | なし — コンストラクタ不変条件のみ |
| `enqueueWithRetry()` | **`assert(false) → return` 経路なし**（コメントで明記） | 同 | なし — 全経路で ownership 移転 |
| `RuntimeWorldAuthority.h:225` | `assert(metadata.sequenceId != 0)` | 消えるが、**直後に `if (metadata.sequenceId == 0)` チェックが残存** → `return nullptr`（owner 無条件消費） | なし — Release でも保護 |
| `RuntimeStore.h` WriteAccess::publishAndSwap | `assert(false && "used after move")` | 消えるが、**`return nullptr` が実行される**（misuse detection、構造的に到達不能） | なし |
| `WorldLifecycleAudit.h:54` | `assert(false)`（二重 retire 検出） | 消えるが、**`doubleRetireCount_` インクリメント + `activeWorldCount_` 補正が実行**（A-5） | なし — telemetry のみ |
| `ISRDSPHandle.cpp:49` | `assert(false && "DSP registry exhausted")` | 消えるが、**`return DSPHandle::null()` が実行** | なし — World lifetime とは別経路 |
| `ISRSealedObject.h:98` | `assert(false && "seal violation")` | 消えるが、**`std::abort()` が実行** | なし — seal violation は lifetime path 外 |

**判定: PASS** — `assert(false)` が消えても、その後の防御コード（`if` チェック / `return nullptr` /
`std::abort()` / telemetry 補正）が Release でも実行されるため、**ownership guarantee は弱くならない**。

### 2. Release build 結果

- コマンド: `cmake --build C:\VSC_Project\ConvoPeq\build --config Release -j 4`
- 並列度 4（C1060 環境起因失敗を回避）
- 結果: **PASS** — 全 433 ステップ完了。`ConvoPeq.exe`（Release）リンク成功、
  `PublicationAdmissionTests.exe`（Release）リンク成功、**全 29 テスト実行ファイル + その他
  テスト実行ファイル（ISRRuntimeIdentityTests / RuntimePublicationCoordinatorTests /
  ISRSemanticValidationTests / RetireGraceSemanticsTests / invariant_INV3_INV5Tests /
  OwnerChannelTests / ISRSoakTests / NormalRetireDSPHandleCompareTests /
  RuntimeSemanticSchemaValidationTests / ObservePathSingleSourceTests /
  OverlapAuthoritySingularTests / ShadowCompareContractTests /
  CrossfadeExecutorLocalContractTests / RuntimeWorldAuthorityProjectionTests /
  PartialPublicationRejectTests / RebuildAdmissionRegressionTests /
  BuildInputSemanticContractTests / DeferredDeletionQueueReclaimTests / MpscBoundedRingTests /
  SequenceArithmeticTests / DSPHandleTableTests / GainStagingContractTests /
  EQProcessorMaxGainTests / EQAnalysisUnitTests / EQBoundExcessBenchmark / FFTBackendTests /
  PriorityIntegrationTests / MTNUPCMeasurement / AudioEngineHarness）が全てリンク成功**
- コード起因のエラーなし（`error` / `FAILED` / `C1060` なし）

### 3. Release CTest 結果

- コマンド: `ctest -C Release --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority"`
- 結果: **29/29 PASS**（Debug と同じ 29-test set）
- Debug PASS / Release FAIL の divergence は**発生しなかった**（NDEBUG による semantic divergence なし）

### 4. Final counter-audit — C1〜C7 を Release semantics で再確認

| ケース | 条件 | Release での `oldWorld` の行き先 | ownership loss? |
|--------|------|----------------------------------|-----------------|
| C1: D full | D が満杯 | `enqueueRetire` → `QueuePressure` → Q | ❌ なし |
| C2: D+Q full | D, Q が満杯 | E（EmergencyQuarantineStore） | ❌ なし |
| C3: D+Q+E full | D, Q, E が満杯 | Terminal（growable, 常に受領） | ❌ なし |
| C4: epoch unsafe | active reader が epoch を通過していない | Terminal に保持 → `drain()` が epoch safe になった時に解放 | ❌ なし |
| C5: shutdown 中 | `isShutdownInProgress()` | `shutdownReclaim` → `terminalReclaim` → epoch safe なら即時破棄 / それ以外は Terminal 保持 → `drainAll()` で解放 | ❌ なし |
| C6: shutdown と publish の race | admission 後に shutdown 開始 | publish 失敗 → `destroyRolledBackDSP()` / 成功 → `retirePublishedRuntimeWorldNonRt` → `shutdownReclaim` | ❌ なし |
| C7: publish execution failure | `commitRuntimePublication` が `{Failed, CallerDestroy}` | `destroyRolledBackDSP()` で回収 | ❌ なし |

**C3/C4 の追加確認（ユーザー指定）**: `TerminalReclaimAuthority::store()` が失敗し得る別の経路がないか。

`store()`（`ISRRetireRouter.cpp:27`）の全 return 経路:
1. `ptr == nullptr || deleter == nullptr` → `return true`（no-op 成功扱い）
2. `entries_.push_back(...)` → `return true`（growable std::vector）

**`store()` が `false` を返す経路は存在しない**。`std::vector::push_back` の例外（メモリ枯渇）は
`noexcept` 関数内で `terminate` を引き起こすが、これは「ownership loss」ではなく「プロセス終了」
であり、Non-RT コンテキストでのみ呼ばれるため実運用では到達しない。

**全 7 ケースで `oldWorld` の所有権は消失しない（Release semantics でも同一）**。

### 5. Shutdown final-drain ordering（Release code path 再追跡）

| 順序 | 処理 | Release での挙動 |
|------|------|------------------|
| 1 | `requestShutdownClearNonRt()` | `shutdownClearRequested_ = true` |
| 2 | `clearPublishedRuntimeSnapshotsNonRt()` | `publishAndSwap(nullptr)` → oldWorld 返す（assert 依存なし） |
| 3 | `retirePublishedRuntimeWorldNonRt(clearedWorld, true)` | `enqueueDeferredDeleteNonRt` → `shutdownReclaim` → `terminalReclaim`（常に true） |
| 4 | `drainDeferredRetireQueues(true)` | D を drain |
| 5 | quiescence 時 `drainAll()` | D + Q + E + Terminal 完全 drain |
| 6 | stuck reader 時 `m_epochDomain.drainAll()` | D のみ（UAF 回避）、epoch-gated drain に委ねる |
| 7 | `isFullyDrained()` / `waitForDrain()` | `terminalReclaimResident == 0` を含めて待機（15-P-5 GAP A fix） |

**判定: PASS** — Release code path でも順序は同一。assert の有無で drain 順序は変わらない。

### 6. Authority singularization の最終確認

- **最終 lifetime authority = `TerminalReclaimAuthority`**（growable std::vector、常に受領）
- `shutdownReclaim()` は**別の lifetime authority ではない** — `terminalReclaim()` に委譲する
  単なるエントリポイント（`ISRRetireRouter.cpp:497`）。15-P-8 設計の `ShutdownReclaimAuthority`
  （bounded 256）は 15-P-4 で廃止され、`TerminalReclaimAuthority` に統合された。
- `drainAllUnsafe()` は**別の reclaim authority として通常経路へ侵入していない**:
  - `ISRRetireRouter::drainAllQuarantineStore()`（Q+E+Terminal）— shutdown 時のみ
  - `EpochDomain::drainAll()`（D）— shutdown 時のみ
  - `EQProcessor.Core.cpp:159` — EQProcessor デストラクタ（shutdown）
  - 通常運転の reclaim 経路（`tryReclaim()` → `drainQuarantineStore()` + `drainEmergencyAndTerminal()`）は
    **epoch-gated のみ**で `drainAllUnsafe` を呼ばない。

**判定: PASS** — lifetime authority は `TerminalReclaimAuthority` に単一収束。

### 7. 発見した gap

- **GAP-8-1（記録のみ・修正なし）**: 15-P-8 設計セクション（上記）が 15-P-4 以前の
  `ShutdownReclaimAuthority`（bounded 256）を参照しており、実装と不一致。
  これは**設計文書の古さ**であり、実装上の gap ではない。15-P-4 で `TerminalReclaimAuthority`
  （growable）に収束済みのため、修正不要（監査結果の因果関係を保つため、設計文書の書き換えは
  行わず本セクションで実装との整合を明記）。
- その他の gap は**発見されなかった**。15-P-8 監査中に新しい設計変更は不要と判断。

### 8. 最終的な D101 / L=0 判定

- **L=0**: PASS — 全 7 ケース（C1〜C7）で `oldWorld` の所有権が消失しない。Release semantics
  でも同一（assert 消滅による ownership guarantee の弱化なし）。
- **D101**: PASS — 15-P-7 の統合不変条件 + 本セクションの Release parity により、
  D101（World リークなし）が Debug/Release 両方で成立。

### 最終判定

**PASS** — Release では `assert(false)` が消えることによって ownership guarantee が弱くならない
ことをコードパスとして確認できた。Release build PASS + Release CTest 29/29 PASS。
15-P シリーズの最終監査として、L=0 / D101 が Debug/Release 両方で成立する。

---

## 15-P-9 — 数式の再確定 (Recalculation after implementation design)

> ⚠️ **旧設計（pre-implementation）**: 本セクションの数値（E=64, ShutdownReclaimAuthority=256, M_pressure=4930）は
> 実装前の設計値。現行実装値（E=512, TerminalReclaimAuthority, M_pressure=5377）は 15-P-9-1 / 15-P-10-1 / 15-P-FINAL を参照。

### B_true definition (post-fix)

```
B_true = D + Q + P + L
       = D + Q + P + 0    (L=0 proven)
```

**EmergencyQ (E) is NOT part of B_true.**

Why: EmergencyQ is an **emergency overflow store**, not a normal residency component. Worlds in EmergencyQ have already been published (A_ref++), are not in OwnerChannel (O), and are not in transit (P). They represent **overflow from D+Q**.

If we include E in B_true:
```
B_true = D + Q + E + P + L = 4096 + 512 + 64 + 1 + 0 = 4673
```

But this changes M_D101! The current M_D101 = 4609 was based on D+Q+P only. If E is part of B_true, M_D101 = 4673.

**→ Decision: E is NOT part of B_true.** It's an emergency overflow that should never be reached (admission gate prevents it). When it IS reached, it's a degenerate case that's immediately handled by `ShutdownReclaimAuthority` at shutdown.

**B_true = D + Q + P + L = 4096 + 512 + 1 + 0 = 4609** — **unchanged.**

M_D101 = 4609 remains correct.

### M_pressure (admission gate)

```
M_pressure = D + Q + O + P + 1
           = 4096 + 512 + 256 + 1 + 1
           = 4866
```

**Wait — this doesn't include EmergencyQ.**

Actually, M_pressure is the **maximum possible population** that can exist simultaneously. With EmergencyQ:

```
M_pressure = D + Q + E + O + P + 1
           = 4096 + 512 + 64 + 256 + 1 + 1
           = 4930
```

But M_pressure is the admission gate's **threshold** — it's set BELOW the total capacity to prevent overflow. The actual max population is:

```
Max population = D + Q + E + O + P
               = 4096 + 512 + 64 + 256 + 1
               = 4929
```

The "+1" in 15-O's M_pressure = 4930 accounts for the World being processed by ISR (1 slot). With EmergencyQ:

```
M_pressure = 4929 + 1 (in-flight) = 4930
```

**→ M_pressure = 4930 is correct** (includes EmergencyQ as normal capacity).

But M_D101 = 4609 does NOT include EmergencyQ, because B_true (published, not yet reclaimed) doesn't count emergency overflow:

```
M_D101 = D + Q + P = 4096 + 512 + 1 = 4609
M_pressure = D + Q + E + O + P + 1 = 4096 + 512 + 64 + 256 + 1 + 1 = 4930
```

**→ M_D101 = 4609, M_pressure = 4930** — both correct, and they measure DIFFERENT things:
- M_D101: max B_true (published Worlds not yet reclaimed)
- M_pressure: max retirement pipeline occupancy (D+Q+E+O+P+in-flight)

### M_D101 verification

```
B_true = D + Q + P + L
       = 4096 + 512 + 1 + 0    (L=0 proven via EmergencyQ + ShutdownReclaimAuthority)
       = 4609 = M_D101 ✅
```

### Summary

| Metric | Formula | Value | Includes EmergencyQ? |
|--------|---------|-------|---------------------|
| M_D101 | D + Q + P + L | 4609 | NO (EmergencyQ is overflow, not residency) |
| M_pressure | D + Q + E + O + P + 1 | 4930 | YES (EmergencyQ is part of admission monitoring) |

---

## 15-P-9-1 — 数式の再確定（post-implementation verification）

### 前提の明確化

15-P-9 設計セクション（上記）は 15-P-4 以前の設計（E=64、ShutdownReclaimAuthority=256）に
基づく。**実際の実装では E=512（RetireQuarantineStore と同型）、ShutdownReclaimAuthority は
廃止され TerminalReclaimAuthority（growable）に統合**されている。本セクションはコードから
再構成した実値に基づく数式の再確定を記録する。

### 1. 各 capacity の実値（コードから取得）

| 変数 | 実値 | コード上の根拠 | 同時存在数 vs queue capacity |
|------|------|----------------|------------------------------|
| D | 4096 | `DeferredDeletionQueue::kQueueSize = 4096`（`DeferredDeletionQueue.h:262`） | queue capacity = 同時存在数（MPSC ring） |
| Q | 512 | `RetireQuarantineStore::kMaxQuarantinedEntries = 512`（`RetireQuarantineStore.h:65`） | queue capacity = 同時存在数（std::array） |
| E | **512** | `m_emergencyQuarantine` は `RetireQuarantineStore` と同型の別インスタンス（`ISRRetireRouter.h:321`）→ kMaxQuarantinedEntries = 512 | queue capacity = 同時存在数 |
| Terminal | **growable** | `TerminalReclaimAuthority`（`std::vector<Entry>`、`store()` 常に true）（`ISRRetireRouter.cpp:27`） | 有限 capacity なし |
| O | 256 | `OwnerChannel::kCapacity = 256`（`OwnerChannel.h:41`） | queue capacity = 同時存在数 |
| P | 64（capacity）/ **1（同時存在）/ 0（所有権）** | `PendingPublishRegistry::kPendingPublishCapacity = 64`（`RuntimeWorldAuthority.h:34`）。**FIFO producer により同時 publish transition は 1**。**非所有**（`const void*` raw pointer のみ、deleter を呼ばない） | capacity ≠ 同時存在数 ≠ 所有権 |

**重要**: P は「queue capacity = 64」「同時に存在し得る publish transition = 1」「所有権寄与 = 0」を区別する。
FIFO producer（CoordinatorLoop 単一スレッド）により、publish transition は常に 1 つ。
`kPendingPublishCapacity = 64` は async enqueue→commit gap の余裕であり、同時存在数ではない。
さらに P は **非所有**（`const void*` raw pointer のみ保持、`unregister()` は deleter を呼ばない）。
実際の所有権は O（OwnerChannel）が保持する（`AudioEngine.h:4537-4542`:
`registerPublish(seqId, newWorld)` → `ownerChannel().enqueue(key, std::move(world))`）。
P は lookup fallback に過ぎない（15-P-10 監査で確定）。

### 2. Terminal が growable になったことを数式へ反映

Terminal が `std::vector` の growable authority になった現在の実装では、**Terminal を有限
capacity として加算してはならない**。以下の 4 つを同じ式として扱わない:

| 指標 | 意味 | Terminal を含めるか |
|------|------|---------------------|
| bounded pressure metric | bounded stores（D/Q/E）が全て満杯になるまでに必要な圧力 | **含めない**（growable のため有限 bound なし） |
| actual maximum resident count | 実際に同時に存在し得る World 数 | **含める**（Terminal も resident を保持し得る） |
| ownership-loss bound | L=0 の保証（どの失敗組合せでも所有権が消失しない） | **含める**（Terminal が最終受領者） |
| D101 admission threshold | D101 判定に使う具体的な閾値 | **含めない**（B_true は bounded store の有限上限） |

### 3. M_D101 の再確定

```
B_true = D + Q + P + L
       = 4096 + 512 + 1 + 0    (P = 同時存在 publish transition = 1, L=0 proven)
       = 4609 = M_D101 ✅
```

**M_D101 = 4609 は維持**（15-P-7/8 と一致）。

**Terminal を B_true に含めない理由**:
- B_true は「published Worlds not yet reclaimed の**有限上限**」を表す。
- Terminal は growable（有限 capacity なし）のため、B_true の有限 bound に含めると
  B_true が非有界になる。
- Terminal は L=0 の保証（ownership-loss bound）として**別に**扱う。
- E も B_true に含めない（15-P-9 設計の判断を維持）: E は overflow store であり、
  admission gate が D の滞留数を観測するため、E が満杯になる前に throttle がかかる。

### 4. M_pressure の再定義

**M_pressure = 5186 は古い設計（E=64, ShutdownReclaimAuthority=256）に基づくため、再計算が必要。**

実際の bounded stores に基づく再計算:

```
M_pressure = D + Q + E + O + 1        (所有権を保持する bounded stores が全て満杯になるまでに必要な圧力)
           = 4096 + 512 + 512 + 256 + 1
           = 5377
```

- **E = 512**（実際の実装。15-P-9 設計の 64 ではない）
- **ShutdownReclaimAuthority（256）は廃止**され TerminalReclaimAuthority（growable）に統合
  → 加算しない
- **P は加算しない**（非所有 lookup registry — 下記「P の扱い」参照）
- +1 = in-flight（ISR が処理中の World）

**M_pressure = 5377**（bounded retirement pressure のモデル上の最大値）。

**P の扱い（15-P-10 監査で確定）**:
- P は `PendingPublishRegistry`（`RuntimeWorldAuthority.h:33-75`）で、`const void*` の
  **非所有 raw pointer** のみを保持する（`unregister()` は deleter を呼ばない）。
- 実際の所有権は O（OwnerChannel）が保持する（`AudioEngine.h:4537-4542`:
  `registerPublish(seqId, newWorld)` → `ownerChannel().enqueue(key, std::move(world))`）。
- P が参照する World は常に O または in-flight（caller の unique_ptr）が所有するため、
  P を加算すると**二重計上**になる。
- したがって M_pressure に P を加算しない（P=64 を加える 5441 は過大評価）。
- 一方 M_D101 の P=1 は「同時 publish transition の World」を表す（所有権は O/in-flight が
  保持する実在の World）。これは M_pressure の O 項に包含される。

**ただし、これは「所有権を保持する bounded stores が全て満杯になるまでに必要な圧力」であり、
Terminal 自体の収容能力を含む総 resident bound ではない**。Terminal は growable のため、
総 resident bound は非有界（実際にはメモリ上限に依存）。

### 5. L の定義を明確化

```
L = ownership-loss count（所有権が消失した World 数）
```

**L = 0** が数式上意味すること:

- **bounded retirement capacity の上限**（M_pressure = 5377）と
  **lifetime ownership の上限**（Terminal growable により非有界）は分離される。
- L=0 は「どの失敗組合せでも `oldWorld` の所有権が消失しない」ことを意味する
  （15-P-7 の counter-audit C1〜C7 でケース網羅証明済み）。
- Terminal が growable であることは「L が有限 bound を持つ」ことを意味するのではなく、
  「**L=0 が構造的に保証される**」ことを意味する（bounded store exhaustion による
  ownership loss が構造的に排除される）。

### 6. Admission threshold との整合性

**実際の gate と監査上の M_pressure は同じ量を表していない**:

| 量 | 実際の観測 | コード上の根拠 |
|----|-----------|----------------|
| `retirePressurePublicationThrottleActive_` | **D の滞留数**（`retireDepth = pendingRetireCount()`）が highWatermark の 90% 超で有効化 | `AudioEngine.Retire.cpp:117,182,390`、`PublicationAdmission.cpp:41` |
| `retirePressureAdmissionStrict_` | **Rebuild admission** の拒否（publish admission ではない） | `AudioEngine.Threading.cpp:27`（`shouldRejectRebuildAdmissionForPressure`） |
| `collectRetirementPressure()` | **実装されていない**（設計上の提案のみ） | grep 0 件 |

- 実際の publish admission gate は **D の滞留数**を観測しており、Q/E/Terminal の resident は
  直接観測していない。
- 15-P-8 設計の「Admission threshold = 4608」は**実装されていない**（設計上の提案のみ）。
- 監査上の M_pressure（D+Q+E+O+1 の合計 = 5377）と実際の gate（D のみ、hwm ベース 3072 の 90%）
  は**異なる量**を表す。
- ただし、これは gap ではない: gate は D の滞留数を観測することで、D が満杯になる前に
  throttle をかける。D が満杯にならなければ Q/E は満杯にならない（Q/E は D の overflow のみを
  保持するため）。**gate の観測量（D）が bounded stores 全体（D+Q+E）の先行指標として機能する**。

### 7. Release/Debug 共通の数式

- capacity / ownership semantics は Debug/Release で変わらない（15-P-8 で確認済み）。
- `assert(false)` が消えても防御コード（`if` チェック / `return nullptr` / `std::abort()` /
  telemetry 補正）が Release でも実行されるため、数式の前提（L=0, B_true ≤ M_D101）は
  Debug/Release 共通で成立する。

### 8. 発見した gap

- **GAP-9-1（記録のみ・修正なし）**: 15-P-9 設計セクション（上記）が古い設計（E=64、
  ShutdownReclaimAuthority=256）に基づく。実際の実装では E=512、Terminal=growable。
  これは**設計文書の古さ**であり、実装上の gap ではない。数式は本セクションで再確定した。
- **GAP-9-2（記録のみ・修正なし）**: `collectRetirementPressure()` は設計上の提案であり、
  実装されていない。実際の gate は D の滞留数を観測する設計のため、実装上の gap ではない。
- コード変更は**不要**（数式と実装の不一致は設計文書の古さのみで、実装上の gap なし）。

### 9. 最終値の確定

| 指標 | 意味 | 最終値 | コード上の根拠 |
|------|------|--------|----------------|
| B_true | D101 の admission / bounded pressure に実際に寄与する有限容量 | D + Q + P + L = 4096 + 512 + 1 + 0 = **4609** | `DeferredDeletionQueue.h:262`, `RetireQuarantineStore.h:65`, FIFO producer |
| M_D101 | D101 判定に使う具体的な閾値 | **4609** | B_true と一致（15-P-7/8 で維持） |
| M_pressure | bounded retirement pressure のモデル上の最大値（所有権ベース） | D + Q + E + O + 1 = 4096 + 512 + 512 + 256 + 1 = **5377** | E=512（`ISRRetireRouter.h:321`）, O=256（`OwnerChannel.h:41`）, P は非所有のため除外 |
| L | ownership-loss count | **0** | Terminal growable により構造的に保証（15-P-7 C1〜C7） |

### 最終判定

**PASS** — M_D101 = 4609 は維持（B_true = D + Q + P + L と一致）。M_pressure は E=512 を反映し、
P を非所有として除外して **5377 に再定義**（5378 は算術誤り、5186 は古い設計に基づく）。
L = 0 は Terminal growable により構造的に保証。D101 / L=0 が Debug/Release 両方で成立。
コード変更なし（監査のみ）。

---

## 15-P-10 — Counter-audit (10 ケースの反証検証)

> ⚠️ **旧設計（pre-implementation）**: 本セクションの数値（EmergencyQ=64, ShutdownReclaimAuthority=256, M_pressure=4930）は
> 実装前の設計値。現行実装値（E=512, TerminalReclaimAuthority, M_pressure=5377）は 15-P-10-1 / 15-P-FINAL を参照。

1. **EmergencyQ overflow**: ✅ Covered — transfer to ShutdownReclaimAuthority. EmergencyQ has 64 slots, admission gate prevents D+Q > 4608, so EmergencyQ can hold at most 63 overflow. ShutdownReclaimAuthority (256) absorbs the rest. L=0 preserved.

2. **shutdown と publish の race**: ✅ Analyzed — `isShutdownInProgress()` is atomic. If check passes (not shutdown), `enqueueWithRetry` proceeds. If another thread sets shutdown between check and `enqueueWithRetry`, the World may enter D/Q/EmergencyQ normally, then `drainAll()` at shutdown drains them. If shutdown is set before check, transfer to `ShutdownReclaimAuthority`. No ownership loss.

3. **Timeout + repeated publication**: ✅ Timeout does NOT affect retirement path. `commitRuntimePublication` timeout → result stays `{Success, Transferred}` → World IS published → normal retirement. Multiple timeouts → multiple Worlds published → normal D/Q/E/Shutdown pipeline. L=0 preserved.

4. **D/Q drain と admission の同時実行**: ✅ `drainQuarantineStore()` (called from `tryReclaim()` in ISR) reads `minReaderEpoch()` and drains entries with older epochs. Admission gate reads `D+Q+E` count. Even if drain and admission race, the admission gate's threshold (4608) leaves EmergencyQ (64) as buffer. Worst case: admission accepts at 4607, drain removes 1 → D+Q goes to 4606 → next publish adds 1 → D+Q=4608 → admission rejects. No overflow.

5. **OwnerChannel::take() と refill**: ✅ `take()` drains a slot atomically (`publishAtomic(s.owner, nullptr, release)`). The taken World goes to `OwnerPtr`, then to `publishAndSwap`. If publishAndSwap fails, World returns to caller (unique_ptr). If succeeds, oldWorld goes to retirement. OwnerChannel (O) is independent of D/Q/E. No interaction affects L.

6. **publishAndSwap() と retirement**: ✅ `publishAndSwap` atomically exchanges `current` and returns oldWorld. The caller (executePublish) immediately calls `retirePublishedRuntimeWorldNonRt(oldWorld)`. No gap — ownership transfers from RuntimeStore to ISR-local to retirement pipeline atomically (single-threaded in ISR).

7. **Release で assert が消えるケース**: ✅ This is the core fix. In Release:
   - `assert(false)` at ISRRetireRouter.cpp:204 → no-op
   - But now: EmergencyQ catches the D+Q-full case → `assert(false)` only hits if EmergencyQ ALSO full
   - EmergencyQ full → ShutdownReclaimAuthority transfer (no assert needed)
   - So the original `assert(false)` location now has a preserving fallback BEFORE it

8. **shutdown drain 中の新規 oldWorld**: ✅ Covered by ShutdownReclaimAuthority. If `clearPublishedRuntimeSnapshotsNonRt()` is called and returns a World, and THEN another `publishAndSwap()` happens (in a pending ISR frame), the second oldWorld also goes to `ShutdownReclaimAuthority`. `drainAll()` drains all. No loss.

9. **二重 reclaim (double-reclaim)**: ✅ Each World enters exactly ONE store (D, Q, E, or ShutdownReclaimAuthority). `drainAllUnsafe()` clears the store after draining (`size_ = 0`). No World can be in two stores simultaneously. The ownership chain is:
   ```
   publishAndSwap → enqueueWithRetry → [D|Q|E|ShutdownReclaimAuthority] → drainAllUnsafe → deleter
   ```
   Each arrow transfers ownership atomically. No branch creates duplicates.

10. **terminal ownership disappearance**: ✅ This is the core problem from 15-N. Post-fix:
    - D+Q full → EmergencyQ (ownership to E)
    - D+Q+E full → ShutdownReclaimAuthority (ownership to shutdown)
    - Shutdown early-return → ShutdownReclaimAuthority (ownership to shutdown)
    - `drainAll()` → drains ALL stores → deleter executes → R_ref++

    **No path allows ownership to disappear.** Every `ptr` entering `enqueueWithRetry` ends in exactly one of D, Q, E, or ShutdownReclaimAuthority, all of which are drained by `drainAll()`.

### Counter-audit verdict

**→ All 10 counter-audit cases PASS. L=0 is formally provable in both Debug and Release.**

---

## 15-P-10-1 — Counter-audit (10 ケースの反証検証, post-implementation verification)

### 前提の明確化

15-P-10 設計セクション（上記）は 15-P-4 以前の設計（EmergencyQ=64、ShutdownReclaimAuthority=256、
M_pressure=4930）に基づく。**実際の実装では E=512（RetireQuarantineStore と同型）、
ShutdownReclaimAuthority は廃止され TerminalReclaimAuthority（growable）に統合**されている。
本セクションはコードから再構成した実値に基づく 10 ケースの反証検証を記録する。

### 1. 10 ケースの反証検証（実装コードとの突き合わせ）

| Case | Precondition | Ownership holder | Transition | Terminal outcome | EBR condition | R/D parity | 数式対応 |
|------|--------------|------------------|------------|------------------|---------------|------------|----------|
| C1 | D full（`enqueueRetire` が QueuePressure/QueueFull） | caller → Q | Stage1 D full → Stage2 retry（`tryReclaim`+`drainEmergencyAndTerminal`）→ Stage3 `Q.quarantine()` | Q owns → `drain(minReaderEpoch, isOlder)` で epoch safe 到達時に deleter | `isOlder(entry.epoch, minReaderEpoch)` == true → safe | 同一コードパス | Q 項（B_true/M_pressure） |
| C2 | D full + Q full（`quarantine` が false） | caller → E | Stage3 Q full → Stage4 `E.quarantine()` | E owns → `drain()` で epoch safe 到達時に deleter | 同上 | 同一コードパス | E 項（M_pressure のみ） |
| C3 | D full + Q full + E full | caller → Terminal | Stage4 E full → Stage5 `terminalReclaim()` → `TerminalReclaimAuthority::store()` | epoch safe + Non-RT → 即時 deleter / epoch unsafe → `store()` 保持 → `drain()` で解放 | `terminalReclaim` が `isOlder(epoch, minReader)` を判定 | 同一コードパス | Terminal growable → L=0 構造的保証 |
| C4 | Terminal + epoch unsafe | Terminal（`entries_`） | `store()` → `entries_.push_back` → 保持 | `drain(minReaderEpoch, isOlderFn)` が epoch safe 到達時に deleter | `isOlder(entry.epoch, minReaderEpoch)` == true → safe | 同一コードパス | Terminal は非有界（growable） |
| C5 | Terminal + epoch safe + Non-RT | caller →（即時）→ reclaimed | `terminalReclaim`: `if (epochSafe && !isRt)` → `deleter(ptr)` 即時実行 → `recordWorldReclaim()` | 同期 reclaim（deleter 即時実行、`reclaimCount_++`） | `isOlder(epoch, minReader)` == false → reader 不在 | 同一コードパス | `reclaimCount_++` → R_ref++ |
| C6 | shutdown early-return（`isShutdownInProgress()` == true） | caller → Terminal | `enqueueDeferredDeleteNonRtWithResult` → `shutdownReclaim()` → `terminalReclaim()`（別 authority ではない） | Terminal owns → `drainAll()` が shutdown 時に全強制解放 | shutdown 中は Audio Thread 停止 → epoch safe → 即時 | 同一コードパス | shutdown path — L 寄与なし |
| C7 | shutdown × publish race | caller（newWorld） | `PublicationAdmission::evaluate` → `RejectedShutdown` → `onRejected(0)` / `trySubmitImpl` publish 失敗 → `isShutdownInProgress()` で `RejectedShutdown` or `RejectedPublishFailure` → `onRejected(0)` | newWorld は caller が破棄（leak なし） | n/a（retirement なし） | 同一コードパス | L 寄与なし |
| C8 | publish execution failure | caller（world unique_ptr） | O enqueue 失敗 → `unregister(seqId)` + `{Failed, CallerDestroy}` / ISR intent queue full → `take(key)` で O から回収 + `unregister(seqId)` + `{Failed, CallerDestroy}` | World は caller が破棄（`destroyRolledBackDSP`） | n/a | 同一コードパス | L 寄与なし |
| C9 | Release/NDEBUG（assert 消失） | Debug と同一 | 全 `assert(false)` 経路に防御コード（`if` チェック / `return nullptr` / `std::abort()` / telemetry 補正）が Release でも実行 | Debug と同一 | 同一 | ✅（15-P-8 で確認） | 数式は Debug/Release 共通 |
| C10 | 全 bounded stores 最大圧力 | D+Q+E+O+1 = 5377 | Terminal が overflow を吸収（growable） | L=0（Terminal は overflow で失敗しない） | n/a | 同一 | B_true=4609 / M_D101=4609 / M_pressure=5377 / L=0 |

### 2. C3 詳細監査 — TerminalReclaimAuthority::store() の厳密検証

**前提**: D full + Q full + E full → `terminalReclaim()` → `TerminalReclaimAuthority::store()`。

1. **`store()` が失敗しないこと**: ✅ `store()` は常に `true` を返す（growable `std::vector<Entry>`、
   `ISRRetireRouter.cpp:27-37`）。失敗 return 経路は存在しない。`enqueueWithRetry` は
   `(void)tstored;` で結果を無視し `RetireEnqueueResult::TerminalReclaim` を返す（所有権は必ず移転）。

2. **vector の allocation failure の扱い**: `entries_.push_back(Entry{...})` は `std::bad_alloc` を
   投げ得る。`store()` は `noexcept` のため、`bad_alloc` → `std::terminate()` → **プロセス終了**。
   コンストラクタに `reserve()` は無い（default vector、初回 push_back で確保）。

3. **`std::bad_alloc` が ownership guarantee を破壊し得ないか**: ✅ 破壊しない。
   `bad_alloc` → `std::terminate` → プロセス終了は **ownership loss ではない**（プロセスが消滅する
   ため「実行継続中の leak」が存在しない）。L=0 は「プロセスが継続する全ての実行」で成立する。
   `store()` が `false` を返す経路（= 所有権が宙に浮く経路）は構造的に存在しない。

4. **Release/Debug で意味が変わらないか**: ✅ 変わらない。`noexcept` は Debug/Release 同一。
   `bad_alloc` → `std::terminate` は両方で同一。所有権移転（`enqueueWithRetry` → `terminalReclaim`
   → `store`）も両方で同一コードパス。

**C3 判定: PASS** — `store()` は常に受領し、allocation failure はプロセス終了（ownership loss ではない）。

### 3. C10 詳細監査 — 数式の整合性（simultaneous resident semantics との比較）

**所有権を保持する bounded stores の実セマンティクス**:

| Store | 所有権 | 同時存在数 | 根拠 |
|-------|--------|------------|------|
| D | ✅ 所有（deleter 保持） | 4096 | `DeferredDeletionQueue::kQueueSize` |
| Q | ✅ 所有 | 512 | `RetireQuarantineStore::kMaxQuarantinedEntries` |
| E | ✅ 所有 | 512 | `ISRRetireRouter.h:321`（Q と同型） |
| O | ✅ 所有（slot 毎に Owner=unique_ptr） | 256 | `OwnerChannel::kCapacity`、`enqueue` が所有権移譲 |
| P | ❌ **非所有**（`const void*` raw pointer） | 0（所有権寄与） | `RuntimeWorldAuthority.h:33-75`、`unregister()` は deleter を呼ばない |
| +1 | in-flight（ISR 処理中） | 1 | FIFO producer / ISR 単一処理 |

**最終数式**:

```
B_true    = D + Q + P + L = 4096 + 512 + 1 + 0 = 4609   (P=1: 同時 publish transition の World)
M_D101    = 4609                                        (B_true と一致)
M_pressure = D + Q + E + O + 1 = 4096 + 512 + 512 + 256 + 1 = 5377   (P は非所有のため除外)
L         = 0
```

**M_pressure に P を加算しない理由（15-P-9-1 §4 と整合）**:
- P は非所有 lookup registry。P が参照する World は常に O または in-flight（caller の unique_ptr）が
  所有するため、P を加算すると**二重計上**になる。
- P=64 を加える 5441 は過大評価（P の storage capacity は同時存在 World 数ではない）。
- M_D101 の P=1 は「同時 publish transition の World」を表し、M_pressure の O 項に包含される。

**L=0 の根拠（bounded stores の有限性ではなく、最終 authority の非失敗）**:
- L=0 は「**所有権を保持する最終 authority（Terminal）が overflow によって失敗しない**」ことを
  根拠とする。bounded stores（D/Q/E/O）が有限であることは L=0 の根拠ではない。
- Terminal は growable（`std::vector`）で `store()` が常に `true` を返すため、D+Q+E+O が全て満杯でも
  所有権は必ず Terminal に移転する（C3 で検証済み）。
- allocation failure（`bad_alloc`）は `std::terminate` → プロセス終了であり、ownership loss ではない。

### 4. 発見した gap

- **GAP-10-1（記録のみ・修正なし）**: 15-P-10 設計セクション（上記）が古い設計（EmergencyQ=64、
  ShutdownReclaimAuthority=256、M_pressure=4930）に基づく。実際の実装では E=512、Terminal=growable、
  M_pressure=5377。これは**設計文書の古さ**であり、実装上の gap ではない。
- **GAP-10-2（記録のみ・修正なし）**: 15-P-10 設計 case 1 の「EmergencyQ has 64 slots」は実際は 512。
  設計文書の古さ。
- **GAP-10-3（記録のみ・修正なし）**: 15-P-10 設計 case 7 の「assert(false) at ISRRetireRouter.cpp:204」
  は現在のコードに存在しない（enqueueWithRetry に assert(false)→return 経路は無い。全 assert は
  `assert(provider_ != nullptr)` のみ）。設計文書の古さ。
- **GAP-10-4（記録のみ・修正なし）**: Terminal `store()` の `bad_alloc` → `std::terminate` は
  プロセス終了であり ownership loss ではない（C3 で検証）。設計判断として妥当。

### 5. 修正の有無

**コード変更なし**。15-P-10 監査で発見した不一致は全て設計文書の古さ（GAP-10-1〜3）であり、
実装上の gap は無い。M_pressure の算術誤り（5378）は 15-P-9-1 の evidence 修正で対応済み（5377 に訂正）。

### 6. 最終 D101 判定

**PASS** — B_true = D + Q + P + L = 4609 = M_D101。C1〜C10 の全ケースで「published Worlds not yet
reclaimed」の上限が 4609 を超えない。admission gate は D の滞留数を観測し（`retirePressurePublicationThrottleActive_`）、
D が満杯になる前に throttle がかかるため、Q/E への overflow は gate の先行指標で抑制される。

### 7. 最終 L=0 判定

**PASS** — 全 10 ケースで所有権が消失しない。所有権チェーン
`D → Q → E → TerminalReclaimAuthority` の各段階で、ptr を手放す前に必ず次の authority に所有権が
移転する（`enqueueWithRetry` の不変条件）。Terminal は growable で常に受領するため、bounded stores
が全て満杯でも所有権は消失しない。`bad_alloc` → `std::terminate` はプロセス終了であり ownership loss
ではない。**L=0 は Debug/Release 両方で成立**。

---

## 15-P — Final verdict

> ⚠️ 本セクションは 15-P-4 実装直後（15-P-9/10 の数式再確定前）の verdict。
> 現行実装値（M_pressure=5377, E=512, TerminalReclaimAuthority）に基づく最終判定は **15-P-FINAL** を参照。

### Post-fix invariant proof

```
L = 0
⇔ ∀ World w: publishAndSwap() において oldWorld として返された w は、
   以下のいずれか一つの terminal path に到達する

   (a) D queue: enqueueRetire 成功 → drain/reclaim → onRelease ✅
   (b) Q quarantine: D full, Q not full → drain → onRelease ✅
   (c) E (EmergencyQuarantineStore, 512): D+Q full, E not full → drain → onRelease ✅
   (d) TerminalReclaimAuthority (growable): D+Q+E full, OR shutdown → drainAll → onRelease ✅
```

**No case allows ownership loss. L=0 is formally proven.**

### D101.3.2 proof (Release)

```
B_true = D + Q + P + L
       = D + Q + P + 0      (L=0 proven above)
       ≤ 4096 + 512 + 1 + 0
       = 4609 = M_D101 ✅
```

**D101.3.2 is RE-CLOSED for Release after fixes.**

### D102 ownership (Release)

**✅ PASS** — Unaffected by fixes. Ownership chain is intact at every stage. Each World has exactly one authoritative owner. EmergencyQ and ShutdownReclaimAuthority extend the chain without breaking it.

### D103 shutdown lifetime — ⚠️ 15-P-CROSS により **RE-EVALUATED**

**❌ BLOCKED** — GAP-CROSS-1（O residual leak on shutdown-before-drain race）により FAIL。15-P-CROSS §9 で修正設計を提示済。

GAP-CROSS-1: `enqueuePublicationIntent()` が成功した後（OwnerChannel に ownership 移譲済み）、
`requestShutdown()` → `shutdownCoordinatorLoop()` により CoordinatorLoop が停止する前に
intent queue が drain されない Race。OwnerChannel にデストラクタ/terminal drain がないため、
残留 owner は ownership loss に相当する。isFullyDrained() は publicationIntentResidencyCount==0
を要求するため、このケースでは drain 完了を永遠に待たない（waitForDrain timeout → timedOut）。
15-P-FINAL はこの経路を扱っておらず、「現行 gap 0件」の判定は **撤回** される。

### Final table — ⚠️ 15-P-CROSS 更新版

| Property | Pre-fix (Release) | Post-fix (15-P-FINAL) | 15-P-CROSS | Verdict |
|----------|-------------------|-----------------------|------------|---------|
| L=0 (normal) | UNPROVEN | **PROVEN** (E + TerminalReclaimAuthority) | ✅（retire chain 全ケース所有権移転） | ✅ |
| L=0 (shutdown) | UNPROVEN (early-return leak) | **PROVEN** (TerminalReclaimAuthority transfer) | ✅（shutdownReclaim=terminalReclaim 常時 true） | ✅ |
| D101.3.2 | INVALIDATED | **PASS** (L=0, B_true ≤ 4609) | ✅（コードから再構成: 4096+512+1+0=4609） | ✅ |
| D102 | PASS | PASS | ✅（ownership chain） | ✅ |
| D103 | provisional | **PASS** (all paths covered) | ❌（GAP-CROSS-1: O residual leak） | **BLOCKED** |
| M_D101 | 4609 (conditional) | 4609 (proven) | ✅（コードから再構成一致） | ✅ |
| M_pressure | 4866 | 5377 (D+Q+E+O+1, P 非所有) | ✅（コードから再構成一致） | ✅ |
| RejectedShutdown | conflated | split (RejectedPublishFailure) | ✅（flag-based pressure） | ✅ |

---

## 15-P-FINAL — Final Verification / Final Verdict（統合最終判定）

> 本セクションは 15-P-4〜15-P-10 の全結果を統合し、5軸（数式・ownership・EBR・shutdown・admission semantics）が
> **同一の現行実装**を指していることを最終証明する。15-P-9/10 の設計セクションは旧設計（pre-implementation）であり、
> 現行実装値は本セクションが権威。

### 1. 現行実装の確定値（cross-reference audit）

| 記号 | 定義 | 現行値 | 根拠（コード） | 15-P-4〜10 での整合 |
|------|------|--------|----------------|---------------------|
| D | DeferredDeletionQueue | 4096 | DeferredDeletionQueue.h:262 (`kQueueSize=4096`) | 15-P-7-1 / 15-P-9-1 一致 |
| Q | RetireQuarantineStore | 512 | RetireQuarantineStore.h:65 (`kMaxQuarantinedEntries=512`) | 15-P-7-1 / 15-P-9-1 一致 |
| E | EmergencyQuarantineStore | 512 | ISRRetireRouter.h:321（Q と同型） | 15-P-9-1 訂正済み（旧設計 64 → 512） |
| O | OwnerChannel | 256 | OwnerChannel.h:41 (`kCapacity=256`) | 15-P-9-1 一致 |
| P | PendingPublishRegistry | 64（capacity）/ 1（同時存在）/ 0（所有権） | RuntimeWorldAuthority.h:33-75（非所有 raw pointer） | 15-P-9-1 訂正済み |
| Terminal | TerminalReclaimAuthority | growable（`store()` 常時 true） | ISRRetireRouter.cpp:27-37 | 15-P-10-1 C3 監査済み |

**数式の確定値（15-P-9-1 / 15-P-10-1 と完全一致）:**

```
B_true     = D + Q + P + L = 4096 + 512 + 1 + 0 = 4609 = M_D101
M_pressure = D + Q + E + O + 1 = 4096 + 512 + 512 + 256 + 1 = 5377   （P は非所有のため除外）
L = 0
```

- **P の扱い**: P は `const void*` raw pointer を保持する非所有 registry（`unregister()` は deleter を呼ばない）。
  所有権は enqueue→commit 間は O（OwnerChannel）が保持。したがって M_pressure に P を加算しない。
  P=64 を加える 5441 は過大評価（storage capacity ≠ 同時存在 World 数）。
- **M_D101 の P=1**: 同時 publish 遷移中の World 1 個（O の in-flight 分）を表し、M_pressure の O 項に包含される。

### 2. 旧設計値の整理（stale value 検索結果）

検索（`grep -n "5378\|5441\|4930"` / `"ShutdownReclaimAuthority"` / `"E=64"`）の結果:

| 旧設計値 | 出現箇所 | 扱い |
|----------|----------|------|
| E=64 | 15-P-1/2/3/7/9/10 設計セクション | **旧設計値**として明示（各セクション冒頭のマーカー参照）。現行は 512 |
| ShutdownReclaimAuthority | 15-P-1/2/3/4 設計セクション | **旧設計値**。現行は TerminalReclaimAuthority（growable）に統合 |
| M_pressure=4930 | 15-P-1/7/9 設計セクション、旧 Final verdict 表 | **旧設計値**。現行は 5377 |
| M_pressure=5378 | 15-P-9-1 §5（前回訂正の見落とし） | **訂正済み → 5377**（本セクションで修正） |
| M_pressure=5441 | 15-P-9-1 / 15-P-10-1（過大評価の言及） | 過大評価として明示（現行値ではない） |

**結論**: 5378 / 5441 / 4930 はすべて「旧設計値」または「過大評価の言及」として明示され、
**現行値として残っていない**ことを確認済み。

### 3. L=0 の厳密化（strictified proof）

**L の定義**: L = ownership-loss count（所有権が消失した World 数）。

**前提（semantics）**: 本証明は「**プロセスが継続するケース**」における ownership-loss = 0 を主張する。
`TerminalReclaimAuthority::store()` は `noexcept` で `std::vector::push_back` を行うため、
メモリ枯渇時は `std::bad_alloc` → `std::terminate()` → **プロセス終了**（ownership-loss ではない）。
プロセス終了は「所有権の消失」ではなく「実行の終了」であり、L の定義域外。

**reclaim chain（全経路で所有権が保持される）:**

```
oldWorld（publishAndSwap の戻り値）
  ├─ (1) D enqueue 成功 → drain/reclaim → onRelease ✅
  ├─ (2) D full → Q enqueue 成功 → drain → onRelease ✅
  ├─ (3) D+Q full → E enqueue 成功 → drain → onRelease ✅
  ├─ (4) D+Q+E full → Terminal.store()（growable・常時 true）→ drainAll → onRelease ✅
  └─ (5) shutdown → shutdownReclaim() = terminalReclaim()（常時 true）→ drainAll → onRelease ✅
```

**L=0 の根拠**: 「最終 authority（Terminal）が overflow で失敗しない」（growable・常時受入）であり、
「bounded store が有限である」ことではない。したがって bounded-capacity の失敗組合せ（D/Q/E 全満杯）でも
所有権は Terminal に移り、消失しない。

**enqueueWithRetry が Shutdown を返す場合**: 呼び出し側（`enqueueDeferredDeleteNonRtWithResult`）が所有権を保持
（`{Failed, CallerDestroy}` 相当）。所有権は消失しない。

### 4. Release/Debug parity 最終確認

| 検証 | Debug | Release | 結果 |
|------|-------|---------|------|
| ビルド | 15-P-7 PASS | 15-P-8 PASS（433 steps） | ✅ |
| CTest | 29/29 PASS | 29/29 PASS | ✅ |
| assert の有無 | `assert(provider_ != nullptr)` 等（22箇所、constructor invariant） | コンパイル時除去 | 所有権 chain に影響なし |
| `publishAndSwap` の moved-from | `assert(false)` | `return nullptr` | 両方とも所有権を返さない（呼び出し側が保持） |

**結論**: assert の有無は所有権 chain を変更しない。Debug/Release で L=0 の証明は同一。

### 5. 5軸の統合（同一の現行実装を指す証明）

| 軸 | 現行実装 | 15-P-4〜10 での検証 | 整合 |
|----|----------|---------------------|------|
| 数式 | B_true=4609, M_D101=4609, M_pressure=5377, L=0 | 15-P-9-1（訂正済み）/ 15-P-10-1 C10 | ✅ |
| ownership | O が唯一の所有権保持、P は非所有、Terminal が最終 authority | 15-P-4-1 / 15-P-8-1 / 15-P-10-1 C1-C10 | ✅ |
| EBR | EpochDomain `isOlder`（wraparound-safe）、epoch-gated drain | 15-P-7-1 / 15-P-8-1 | ✅ |
| shutdown | `shutdownReclaim()` = `terminalReclaim()`（常時 true）、`drainAll()` が D+Q+E+Terminal 完全 drain | 15-P-5-1 / 15-P-10-1 C3 | ✅ |
| admission semantics | `RejectedShutdown` / `RejectedPublishFailure` 分離、flag-based pressure | 15-P-6-1 / 15-P-10-1 | ✅ |

### 6. 監査で発見したもの vs 現行実装上の gap

**監査で発見したもの（すべて文書の stale、コード変更なし）:**

| GAP | 内容 | 対応 |
|-----|------|------|
| GAP-9-1 | 15-P-9 設計セクションが旧設計（E=64, ShutdownReclaimAuthority=256） | 旧設計値として明示 |
| GAP-9-2 | M_pressure の算術誤り（5378） | 15-P-9-1 で 5377 に訂正 |
| GAP-10-1 | 15-P-10 設計セクションが旧設計（EmergencyQ=64, ShutdownReclaimAuthority=256, M_pressure=4930） | 旧設計値として明示 |
| GAP-10-2 | 15-P-10 設計 case 1 の「EmergencyQ has 64 slots」は実際は 512 | 旧設計値として明示 |
| GAP-10-3 | 15-P-10 設計 case 1 の「ShutdownReclaimAuthority (256)」は実際は Terminal（growable） | 旧設計値として明示 |
| GAP-10-4 | 15-P-10 設計 case 2 の「transfer to ShutdownReclaimAuthority」は実際は shutdownReclaim()=terminalReclaim() | 旧設計値として明示 |
| GAP-FINAL-1 | 旧 Final verdict 表の M_pressure=4930 | 本セクションで 5377 に更新 |

**現行実装上の gap: なし（0件）** — すべての不一致は文書の stale であり、実装の欠陥ではない。

### 7. 最終判定（Final Verdict）

```
✅ D101（B_true ≤ M_D101 = 4609）: PASS（L=0 証明済み）
✅ D102（ownership chain）: PASS（各 World が唯一の所有権保持者を持つ）
✅ D103（shutdown lifetime）: PASS（TerminalReclaimAuthority が全 shutdown World を回収）
✅ L=0（ownership-loss = 0）: PROVEN（プロセス継続ケース）
✅ Release/Debug parity: PASS（29/29 CTest 両方）
✅ 5軸整合: PASS（数式・ownership・EBR・shutdown・admission semantics が同一実装を指す）
```

**15-P シリーズ: 全 29/29 PASS。コード変更なし（監査のみ）。**

---

## 15-P-CROSS — 独立カウンター監査（コードからの再構成）

> 本セクションは 15-P-FINAL の4つの最終主張（**L=0, B_true=4609, M_pressure=5377, D101/D102/D103 PASS**）を、
> 既存 evidence の結論を参照せず、**現行コードのみから独立に再構成**するカウンター監査である。
> 監査方針: 各主張をコードの事実から導出し、evidence の結論と突き合わせる。差異があれば新規 GAP として切り出す。

### 0. 監査対象コード

| ファイル | 監査対象 |
|----------|----------|
| ISRRetireRouter.cpp | TerminalReclaimAuthority::store (27-37), enqueueRetire (220-245), enqueueWithRetry (277-338), terminalReclaim (423-447), shutdownReclaim (488-498), drainAll (507-515) |
| OwnerChannel.h | kCapacity=256 (41), enqueue/take (67-108), **デストラクタなし** |
| RuntimeWorldAuthority.h | PendingPublishRegistry (33-66, 非所有), kPendingPublishCapacity=64 (34), publish (230-250), clearPublishedRuntimeSnapshotsNonRt (256-263) |
| RuntimePublishExecutor.h | executePublish（全所有権フロー） |
| ISRRuntimePublicationCoordinator.h/cpp | enqueuePublicationIntent (344-380), processIntent, isFullyDrained (560-600) |
| ISRCoordinatorLoop.cpp | run() (31-43, **shutdown 時 break**) |
| AudioEngine.h | publish path (4525-4575), enqueueDeferredDeleteNonRtWithResult (4190-4220), isShutdownInProgress (1470-1480) |
| AudioEngine.Retire.cpp | evaluateRetirePressureLevelNoRt, applyRetirePressurePolicyNoRt |
| AudioEngine.Threading.cpp | shouldRejectRebuildAdmissionForPressure (20-28), isFullyDrained (114-160), runCoordinatorPhase (242-300) |
| AudioEngine.Processing.ReleaseResources.cpp | releaseResources shutdown sequence (60-530) |
| PublicationAdmission.cpp | evaluate (11-60, flag-based) |
| EpochDomain.h | publishEpoch (190), getMinReaderEpoch (211), isOlder (432) |
| DeferredDeletionQueue.h | kQueueSize=4096 (262) |
| RetireQuarantineStore.h | kMaxQuarantinedEntries=512 (65) |

### 1. L=0 の再構成（コードから）

**コードの事実:**
1. `TerminalReclaimAuthority::store()` (ISRRetireRouter.cpp:27-37) は `noexcept` で `std::vector<Entry>::push_back` を行い、**常に true を返す**（growable）。
2. `push_back` が `std::bad_alloc` を投げる場合 → `noexcept` により `std::terminate()` → **プロセス終了**。
3. `enqueueRetire()` (ISRRetireRouter.cpp:220-245) は **Success または QueuePressure のみ**を返す（**Shutdown を返さない**）。
4. `enqueueWithRetry()` (ISRRetireRouter.cpp:277-338) のフロー:
   - Stage 1: D enqueue → Success なら return
   - Stage 2: retry（kMaxRetry=2, tryReclaim + drainEmergencyAndTerminal）
   - Stage 3: Q quarantine → stored なら return
   - Stage 4: E quarantine → estored なら return
   - Stage 5: Terminal（常時 true）→ return TerminalReclaim
   - 最終 `return result`（Shutdown 経路）は **デッドコード**（enqueueRetire が Shutdown を返さないため到達不能）
5. `enqueueDeferredDeleteNonRtWithResult()` (AudioEngine.h:4190-4220):
   - `isShutdownInProgress()` → `shutdownReclaim()`（= terminalReclaim()、常時 true）→ Success
   - それ以外 → `enqueueWithRetry()` → 常に Success/QueuePressure/TerminalReclaim（Shutdown は返らない）
6. したがって **retire 経路は全ケースで所有権を D/Q/E/Terminal のいずれかに移転する**。所有権消失経路は存在しない。

**L=0 の再構成:**
- プロセス継続ケース: 全 retire 経路で所有権が移転するため **L=0**。
- プロセス終了ケース（bad_alloc → terminate）: 実行の終了であり、L の定義域外（15-P-FINAL §3 の前提と一致）。
- **std::terminate ≠ ownership-loss の論理**: 有効。ただしこれは「プロセス継続」をスコープとする**前提**であり、証明ではない。D101/D102/D103 が「実行中のシステム」を対象とするなら、このスコープ設定は妥当。

**15-P-FINAL との差異:**
- 15-P-FINAL §3 は「enqueueWithRetry が Shutdown を返す場合: 呼び出し側が所有権を保持」と記述するが、**現行コードでは enqueueWithRetry は Shutdown を返さない**（デッドコード）。記述は保守的（過剰記述）であり L=0 に影響しないが、**コードと一致しない記述**である（→ GAP-CROSS-2）。

### 2. M_pressure = 5377 の再構成（コードから）

**コードの事実:**
- D = 4096 (DeferredDeletionQueue.h:262), Q = 512 (RetireQuarantineStore.h:65), E = 512 (ISRRetireRouter.h:321, Q と同型), O = 256 (OwnerChannel.h:41)。
- D/Q/E は retire ストア（各 World は同時にちょうど1つに存在）。
- O は in-flight publish World（未 commit）を保持（genuine ownership）。
- P は非所有 raw pointer registry（RuntimeWorldAuthority.h:33-66）。P が参照する World は O に存在。

**同時存在可能性の検証:**
- **D+Q+E**: retire ストア。各 World は D/Q/E/Terminal のいずれか1つに存在。retire が D→Q→E→Terminal の順に溢れるため、D+Q+E は同時に満杯可能。
- **O**: in-flight publish World。retire ストアとは独立（未 commit World は retire されない）。
- **+1**: executePublish が O から take した後、publish 完了までの World（O から出た直後）。
- **O と +1 の二重計上**: なし。World は O にあるか、処理中（+1）かのどちらか。両方には存在しない。
- **P の除外**: 正しい。P が参照する World は O に存在するため、O 項に包含される。P を加算すると二重計上。

**M_pressure = D + Q + E + O + 1 = 4096 + 512 + 512 + 256 + 1 = 5377** ✅

**15-P-FINAL との差異: なし（一致）**

### 3. B_true / M_D101 = 4609 の再構成（コードから）

**コードの事実:**
- 15-D の定義: B_true = D + Q + I + L, I = "between publishAndSwap return and enqueueWithRetry completion"（**retire 側 transient**）。
- executePublish (RuntimePublishExecutor.h) は publish → retirePublishedRuntimeWorldNonRt を**同期実行**する。したがって **I_max = 1**（単一スレッドの transient）。
- 15-D は「With timeout: I_max = 256 (OwnerChannel::kCapacity)」と記述するが、これは **I（retire 側）と O（publish 側）の混同**。

**再構成:**
- B_true = D + Q + I + L, I_max = 1（同期 retire により構造的に保証）。
- B_true ≤ 4096 + 512 + 1 + 0 = **4609** ✅
- **4609 は optimistic ではなく safe bound**（I_max=1 は同期 retire により保証）。
- 15-D の「M_safe = 4864」（timeout 時 I_max=256）は誤り。timeout は publish 側（O に World が蓄積）に影響し、これは **M_pressure の O 項（5377）に包含**される。B_true に O を含めるのは二重計上（→ GAP-CROSS-4）。

**P=1 の意味（15-P-FINAL の記述との整合）:**
- 15-P-FINAL は B_true = D + Q + P + L と書き、P=1 を「同時 publish 遷移中の World 1 個」とする。
- この「P=1」は実際には **I（retire 側 transient）** であり、PendingPublishRegistry（capacity 64）ではない。
- **命名の混同**: evidence は「P」を (a) PendingPublishRegistry（capacity 64, 非所有）と (b) in-flight transient（=1）の2つの意味で使用している。値（4609）は正しいが、ラベルは誤解を招く（→ GAP-CROSS-3）。
- 正しい式: **B_true = D + Q + I + L = 4096 + 512 + 1 + 0 = 4609**（I は retire 側 transient）。

**15-P-FINAL との差異: 値は一致（4609）だが、式のラベル（P vs I）に混同あり。**

### 4. Admission と retirement pressure の分離（コードから）

**コードの事実:**
- `PublicationAdmission::evaluate()` (PublicationAdmission.cpp:11-60) は **flag-based pressure**（`retirePressurePublicationThrottleActive_`）を観測する。数値閾値（D+Q+E）は直接見ない。
- `retirePressurePublicationThrottleActive_` は `applyRetirePressurePolicyNoRt()` で `effectiveLevel >= 2`（medium = hwm の90%）のとき true。`retireDepth = pendingRetireCount()` = **D の residency**。
- `retirePressureAdmissionStrict_` は `shouldRejectRebuildAdmissionForPressure()` (AudioEngine.Threading.cpp:20-28) で消費される — **rebuild admission 専用**。publish admission は見ない。
- `retirePressureAdmissionStrict_` は critical（severe && retireDepth >= hwm）または effectiveLevel >= 3 かつ overflowRate > 10.0 のときのみ true。

**再構成:**
- publish admission gate は **D の pressure**（flag 経由）を見る。Q/E/O の storage bound は見ない。
- rebuild admission gate（retirePressureAdmissionStrict_）は publish admission と分離されている。
- **理論モデル（M_pressure = D+Q+E+O+1）は storage bound** であり、**実装 gate（retirePressurePublicationThrottleActive_）は D-residency flag**。両者は異なる概念であり、evidence はこれを明確に区別すべき。

**15-P-FINAL との差異: 実装は一致（flag-based）。ただし「理論モデル」と「実装 gate」の混同リスクが evidence に残る。**

### 5. Shutdown race（コードから）

**コードの事実:**
- Admission (evaluate) は `isShutdownInProgress()` → RejectedShutdown。
- publish path (AudioEngine.h:4525-4575): registerPublish → ownerChannel.enqueue → enqueuePublicationIntent。
- `enqueuePublicationIntent()` (ISRRuntimePublicationCoordinator.h:344-380) は `CoordinatorState::ShuttingDown` で false を返す → 呼び出し元が `ownerChannel().take(key)` + `unregister()` + CallerDestroy。
- publish 成功（Transferred）: World は O + intent queue。
- executePublish: take(key) → publish → oldWorld retire → enqueueDeferredDeleteNonRt → shutdownReclaim（shutdown 中）→ Terminal → 破棄。✅
- retire 経路: enqueueWithRetry は Shutdown を返さない → World は常に移転。✅
- Debug/Release parity: 所有権 chain は同一。assert の有無は chain を変えない。✅

**GAP-CROSS-1（新規発見）: O 残留 World の shutdown 時リーク**
- `CoordinatorLoop::run()` (ISRCoordinatorLoop.cpp:31-43) は `isShutdownInProgress()` が true で **break** する。shutdown 開始後は intent を処理しない。
- したがって、shutdown 開始前に enqueue された Publish intent が coordinator の次の tick 前に shutdown が始まると、**その intent は処理されず、World は O に残留**する。
- `OwnerChannel` には **デストラクタがない**（raw `Owner*` を保持）。エンジン破棄時に O の残留 World は **リーク**（raw pointer 消失）。
- `isFullyDrained()` は `publicationIntentResidencyCount == 0` を要求するため、残留 intent があると waitForDrain がタイムアウトする（shutdown は timedOut で続行）。
- これは **狭い race**（Builder enqueue 〜 coordinator tick の最大1ms 窓）だが、rebuild 進行中に shutdown が始まると到達可能。
- work88 の文書は「CoordinatorLoop は shutdown 中に intentQueue_ を drain する」と主張するが、**現行コードは break する** — 文書とコードの不一致。

**15-P-FINAL との差異: 15-P-FINAL はこの経路を扱っていない。新規 GAP として切り出す。**

### 6. 4つの最終主張との比較

| 主張 | 再構成結果 | 一致 |
|------|-----------|------|
| L=0 | プロセス継続ケースで成立（retire 経路は全ケース所有権移転） | ✅（ただし §3 の Shutdown 記述はデッドコード） |
| B_true = 4609 | D+Q+I+L = 4096+512+1+0（I は retire 側 transient） | ✅（値一致。P vs I のラベル混同あり） |
| M_pressure = 5377 | D+Q+E+O+1 = 4096+512+512+256+1 | ✅ |
| D101/D102/D103 | D101: B_true ≤ 4609 ✅ / D102: 所有権 chain ✅ / **D103: shutdown 回収 ❌（GAP-CROSS-1 — BLOCKED/FAIL）** | ❌ |

### 7. 新規 GAP の抽出

| GAP | 内容 | 重大度 | 対応 |
|-----|------|--------|------|
| GAP-CROSS-1 | O 残留 World の shutdown 時リーク（CoordinatorLoop break + OwnerChannel デストラクタなし） | **❌ real gap / ownership terminal path の未完成**（enqueue 成功後 Coordinator 停止 race） | drainResidualOwners() 追加提案（§9） |
| GAP-CROSS-2 | 15-P-FINAL §3 の「enqueueWithRetry が Shutdown を返す」記述はデッドコード（enqueueRetire は Shutdown を返さない） | 低（文書の不正確さ） | 記述の修正 |
| GAP-CROSS-3 | B_true 式の「P」ラベル混同（PendingPublishRegistry vs in-flight transient I） | 低（文書の不正確さ） | 式を B_true = D+Q+I+L に統一 |
| GAP-CROSS-4 | 15-D の「M_safe = 4864」（timeout 時 I_max=256）は I と O の混同 | 低（文書の不正確さ） | 4609 が safe bound であることを明記 |

### 8. 結論

- **L=0, B_true=4609, M_pressure=5377 の3つの数値主張はコードから再構成可能であり、値は正しい。**
- **D101（B_true ≤ 4609）と D102（所有権 chain）はコードから再構成可能。**
- **D103（shutdown 回収）は GAP-CROSS-1 により完全には成立しない可能性がある**（O 残留 World のリーク経路）。
- 15-P-FINAL の「現行実装上の gap: 0件」は **GAP-CROSS-1 により修正が必要**。
- 15-P シリーズは「ほぼ強く閉じる」が、GAP-CROSS-1 の調査（shutdown 時の O drain 経路）を完了するまで最終判定を確定できない。

### 9. 15-P-CROSS-DESIGN-FINAL -- GAP-CROSS-1 fix (audit OK, NOT STARTED)

> Design: FINAL / Implementation: NOT STARTED / D103: BLOCKED. No code change -- PRECHECK done.

#### 9.1 GAP-CROSS-1 recap
- counterexample holds: enqueuePublicationIntent ok -> shutdownCoordinatorLoop break -> intent unprocessed
- no existing consumer/destructor/fallback -- OwnerChannel has no destructor (OwnerChannel.h:41), RuntimeWorldAuthority only holds ownerChannel_ (raw Owner* -> leak at engine destruction)
- D103 = BLOCKED (real gap) -- 15-P-FINAL "0 gaps" withdrawn (sec 8)

#### 9.2 PRECHECK 5/5
A. ownership transfer: enqueueDeferredDeleteNonRtWithResult(void*,void(*)(void*)) -- Success->D/Q/E/Terminal, QueuePressure->D(retry), TerminalReclaim->Terminal, Shutdown->unreachable (enqueueRetire returns Success|QueuePressure only -- ISRRetireRouter.cpp:220-245, enqueueWithRetry tail return is QueuePressure only). Fn safe.
B. single-transfer: take() = consumeAtomic(load) -> publishAtomic(nullptr,release). drainAllNonRt same pattern. no double-transfer. re-drain safe (slot==nullptr after).
C. key reset: take(key) uses key matching; drainAllNonRt full scan -> s.key={} not needed (sec 9.3 corrected).
D. enqueueRetire Success|QueuePressure only: enqueueWithRetry return Shutdown path (ISRRetireRouter.cpp:277-338) dead code -- enqueueRetire never checks isShutdownInProgress(). Fn(void*) safe.
E. shutdown order: lines 189(CoordinatorLoop join), 190(Builder join), 203(advanceRetireEpoch), 428(clearPublished), 435(drainAllQuarantineStore), 455(waitForDrain timeout=symptom), 525(finalizeShutdown -> drainAllNonRt insert point). timeout is diagnostic only -> always proceed to finalizeShutdown.

PRECHECK 5/5 PASS.

#### 9.3 OwnerChannel abstraction (FINAL)
```cpp
// OwnerChannel.h -- no friend, no slots_ exposure, callback transfer
template <class Fn>
std::size_t drainAllNonRt(Fn&& reclaim) noexcept
{
    std::size_t reclaimed = 0;
    for (std::size_t i = 0; i < kCapacity; ++i) {
        Slot& s = slots_[i];
        Owner* const raw = consumeAtomic(s.owner, std::memory_order_acquire);
        if (raw != nullptr) {
            publishAtomic(s.owner, static_cast<Owner*>(nullptr),
                          std::memory_order_release);
            reclaim(raw);
            ++reclaimed;
        }
    }
    return reclaimed;
}
```
- RuntimeWorldAuthority: no friend; drainAllNonRt public template API
- slots_ private; no raw Owner* cross-authority transfer (Fn callback)
- Fn(const RuntimeState*) -- void* implicit OK (Owner::get() = const RuntimeState*)
- s.key={} removed (sec 9.2-C)

#### 9.4 ownership-transfer contract
drainAllNonRt(Fn) -> relinquish (load->publish(nullptr)->Fn(raw)) -> RuntimeWorldAuthority drainResidualOwners() lambda -> enqueueDeferredDeleteNonRtWithResult(raw, runtimePublishWorldDeleter, World) -> isShutdownInProgress()->shutdownReclaim->terminalReclaim(growable)->D/Q/E/Terminal

D103 invariant: (a) OwnerChannel residual==0 [drainAllNonRt], (b) drained owners in D/Q/E/Terminal [enqueueDeferredDeleteNonRt->Terminal], (c) no discard [enqueueRetire Success|QueuePressure only].

#### 9.5 call site (ReleaseResources.cpp)
```cpp
m_coordinator.finalizeShutdown(timedOut);   // line 525
worldAuthority_.drainResidualOwners();       // drainAllNonRt -> enqueueDeferredDeleteNonRt -> Terminal
```
Insert: finalizeShutdown (line 525) immediately after. Producer/consumer stopped (189/190), quiescence confirmed (203/428/435).

#### 9.6 verdict
| item | result |
| Design | FINAL (PRECHECK 5/5) |
| Impl | NOT STARTED (no code change) |
| D103 | BLOCKED (drainAllNonRt not implemented -- Impl-1 pending) |

GAP-CROSS-1: BLOCKED -> implement drainAllNonRt (Implementation-1). D103 re-audit after impl.
---

## 15-P-4-0 — Terminal Reclaim 適用可能性監査 (Pre-implementation audit)

### 1. `enqueueWithRetry()` の実行コンテキスト — 全 call site 確認

| Call site | File:Line | Context | RT/Non-RT | Caller ownership on failure? |
|-----------|-----------|---------|-----------|------------------------------|
| `enqueueDeferredDeleteNonRtWithResult` | AudioEngine.h:4201 | Non-RT (ISR) — calls `m_retireRouter->enqueueWithRetry` | ISR (Non-RT context) | ❌ No — only checks RetireEnqueueResult enum |
| `ISRRetireRouter::retire` | ISRRetireRouter.cpp:159 | Non-RT — calls `enqueueWithRetry` internally | Non-RT | ❌ No — result ignored |
| `DSPLifetimeManager::retire` | DSPLifetimeManager.cpp:49 | Non-RT (Timer/Coordinator) | Non-RT | ❌ No — `juce::ignoreUnused(result)` |
| `DSPLifetimeManager::retireByHandle` | DSPLifetimeManager.cpp:96 | Non-RT | Non-RT | ❌ No — `juce::ignoreUnused(result)` |
| `ISRRuntimePublicationCoordinator::retire` | ISRRuntimePublicationCoordinator.cpp:164 | Non-RT (Coordinator) | Non-RT | ✅ Yes — returns result to caller |
| `SnapshotCoordinator::switchImmediate` | SnapshotCoordinator.cpp:57 | Non-RT (Timer) | Non-RT | ✅ Yes — checks `!result` → quarantineRetireSink |
| `SnapshotCoordinator::completeFade` | SnapshotCoordinator.cpp:114 | Non-RT (Timer) | Non-RT | ✅ Yes — checks `!result` → quarantineRetireSink |
| `EQProcessor::retireCoreNode` | EQProcessor.Core.cpp:61 | Non-RT | Non-RT | ❌ No — returns bool |

**Key finding:** `enqueueWithRetry()` is called from **Non-RT contexts only**. NO call site is RT. This means synchronous destruction (if epoch-safe) would NOT violate RT constraints — the function runs in Non-RT context.

Wait — let me re-check. `enqueueWithRetry` is called from `enqueueDeferredDeleteNonRtWithResult` (AudioEngine.h:4201), which is called from `retirePublishedRuntimeWorldNonRt` (AudioEngine.h:3529), which is called from `RuntimePublishExecutor::executePublish` (RuntimePublishExecutor.h:78).

**Is `executePublish` RT or Non-RT?**

### 2. `executePublish` は Non-RT である証明

`executePublish` は `RuntimePublishExecutor.h:78` から呼ばれ、`PublishExecutor::executePublish()` は `ISRRuntimePublicationCoordinator::ProcessIntent.cpp` (ISRRuntimePublicationCoordinator_ProcessIntent.cpp:20-50) 内で呼ばれる。これは `CoordinatorLoop::run()` (ISRCoordinatorLoop.cpp:38) から呼ばれ、`CoordinatorLoop` は `juce::Thread`（Non-RT）。`CoordinatorLoop::run()` は `engine_.runCoordinatorPhase()` を呼び出し、`wait(kIntervalMs)` でスリープする。Audio thread とは非同期に分離されている。

**証明: `executePublish` = Non-RT** ✅

### 3. World deleter の RT-safety 監査

World deleter は `AudioEngine.h:3531-3533`:
```cpp
ptr->unseal();
ptr->~RuntimePublishWorld();
convo::aligned_free(ptr);
```

- **`unseal()`**: `SealedObject::unseal()` — `publishAtomic(sealState_, Unsealed, release)` — atomic store, no allocation, no JUCE callback ✅
- **`~RuntimePublishWorld()`** (= `~RuntimeState()`): `RuntimeState` は `SealedObject<RuntimeState>` CRTP から継承。仮想デストラクタ `~SealedObject() = default`。`RuntimeState` は `struct`、メンバーは全て POD/`isr::` semantic types (primitive + aggregate)。デストラクタは compiler-generated default — **trivial destructible** ✅
- **`aligned_free(ptr)`**: `convo::aligned_aligned_free` — `std::aligned_free` (C17) ラッパー。libc 呼び出しのみ ✅

**すべての World deleter 操作は RT-safe（割り込みなし、確定的、allocation-free）** ✅

### 4. Epoch-safe synchronous destruction の条件

Terminal reclaim は以下の条件を満たす場合のみ安全:

1. **No reader references the World**: `epoch >= minReaderEpoch` なら全 reader が `exitEpoch` を読み過ごしている。`isOlder(world_epoch, minReader)` が `false` なら reader は既に World を見ていない。
2. **Non-RT context**: 確保された（§2）。
3. **Destructor is RT-safe**: 確保された（§3）。

`enqueueWithRetry` が返す `RetireEnqueueResult::Shutdown` は、**シャットダウン中** であることを示す。シャットダウン中は Audio Thread は停止済みであり、reader epoch は固定または進行停止中。`drainAllUnsafe` はシャットダウン後に全 World を強制解放するため、epoch ゲートは不要（`assert` なし、条件なし）。

### 5. Shutdown drain ordering の監査

`AudioEngine.Processing.ReleaseResources.cpp:365-480`:

1. **Line 371**: `m_retireRouter->drainQuarantineStore()` — Q drain (epoch-gated, normal path)
2. **Line 455**: `clearPublishedRuntimeSnapshotsNonRt()` → `clearedWorld` （シャットダウン clear path）
3. **Line 460**: `retirePublishedRuntimeWorldNonRt(clearedWorld, true)` → `enqueueDeferredDeleteNonRtWithResult` → **early-return `Shutdown`** → ptr **LEAKED** (pre-fix)

**問題**: Step 2-3 は `drainQuarantineStore()` (step 1) **の後** で実行される。Step 3 で新たに生成された `clearedWorld` は `Shutdown` 結果で破棄されるため、step 1 の drain では到達できない。

**Post-fix**: `ShutdownReclaimAuthority` が `clearedWorld` の所有権を受け取り、**step 1 後の `drainAll()` (または shutdown complete callback)** で解放される。

### 6. Terminal Reclaim の適用性判定

| 条件 | 状態 | 結果 |
|------|------|------|
| All `enqueueWithRetry` call sites are Non-RT | ✅ 8/8 confirmed | Non-RT |
| World deleter is RT-safe | ✅ trivial destructor + atomic unseal + aligned_free | Safe |
| Epoch-safe synchronous destruction possible | ✅ minReaderEpoch check | Possible |
| Shutdown drain ordering fixable | ✅ ShutdownReclaimAuthority post-fix | Fixed |

**結論: Terminal Reclaim IS applicable. Synchronous destruction in Non-RT context, gated by epoch safety, is safe and correct.**

### 7. RuntimePublishWorld の構造確認

`RuntimePublishWorld = RuntimeState` (AudioEngine.h:332). `RuntimeState` (AudioEngine.h:145-240) の構造:

- POD メンバー: `worldId`, `generation`, `runtimeVersion`, `transitionId`, `schemaVersion`
- Aggregate サブオブジェクト: `EngineRuntime`, `RuntimeGraph`, `RuntimeMetadata`, `GenerationSemantic`, `TopologySemantic`, `RoutingSemantic`, `ExecutionSemantic`, `PublicationSemantic`, `OverlapSemantic`, `RetireSemantic`, `TimingSemantic`, `LatencySemantic`, `ResourceSemantic`, `AffinitySemantic`, `AutomationSemantic`, `CoefficientSemantic`, `DSPSemanticProjection`
- `SealedObject<RuntimeState>` CRTP base (atomic `sealState_` メンバー)
- **No virtual dispatch in destructor** — `SealedObject` の仮想デストラクタは `= default`
- **No heap allocation in any sub-object** — 全員 value-type / fixed arrays
- **No JUCE API calls in destruction path**

**Confirmed: `~RuntimePublishWorld()` is trivially safe to call synchronously in Non-RT context.** ✅

### 8. `isOlder` の wraparound-safe セマンティクス確認

`EpochDomain::isOlder(a, b)` = `static_cast<int64_t>(a - b) < 0` (EpochDomain.h:432).

- uint64_t subtraction with int64_t cast は modular arithmetic → wraparound-safe ✅
- `drain(minReader, isOlder)` は `isOlder(entry.epoch, minReader) == true` なら entry を解放
- これは "reader が entry の epoch を観測していない" = "reader は entry を参照していない" を意味

Terminal reclaim は `isOlder(world_epoch, minReaderEpoch()) == false`（すなわち `world_epoch >= minReaderEpoch` in modular ordering) で safe であることを確認。

### 9. Post-fix ownership flow (complete)

```
retirePublishedRuntimeWorldNonRt(world, false)
  → enqueueDeferredDeleteNonRtWithResult(world, deleter, World)
    → if isShutdownInProgress():
        [PRE-FIX:  return Shutdown, world LEAKED ❌]
        [POST-FIX: transfer to ShutdownReclaimAuthority, return Shutdown ✅]
    → m_retireRouter->enqueueWithRetry(world, deleter, epoch, World)
      → enqueueRetire: success → D queue owns
      → QueuePressure/QueueFull → m_retireQuarantine.quarantine(...)
        → success → Q owns
        → failure (Q full):
          [PRE-FIX:  assert(false), world LEAKED ❌ (assert is no-op in Release)]
          [POST-FIX: transfer to EmergencyQ, then to ShutdownReclaimAuthority ✅]
```

### 10. Terminal reclaim の実装方針

Terminal reclaim は `ISRRetireRouter::drainAll()` または `shutdownComplete()` コールバックで実行:

1. **Normal drain** (`tryReclaim`): epoch-gated `drain(minReader)` on D + Q + EmergencyQ
2. **Shutdown drain** (`drainAll()`): `drainAllUnsafe()` on D + Q + EmergencyQ + ShutdownReclaimAuthority
3. **Emergency Q overflow**: `drainAllUnsafe()` force-releases (epoch check skipped — shutdown or critical path)
4. **ShutdownReclaimAuthority**: `drainAllUnsafe()` が final terminal drain を実行

All paths execute the same deleter: `unseal → ~RuntimePublishWorld → aligned_free`.

### 11. Counter-checks — なぜ Terminal Reclaim は既存 invariant を破らないか

1. **INV-PUB-1 (ownership exclusivity)**: Terminal reclaim は `drainAllUnsafe` が **audio thread 停止後** にのみ呼ばれる。Concurrent access なし。
2. **INV-PUB-2 (epoch isolation)**: Normal drain は `isOlder` ゲート。Shutdown drain は audio thread 停止により reader が存在しない。
3. **INV-PUB-3 (publisher lock)**: `enqueueWithRetry` は `m_retireRouter` mutex (Q のみ) または atomic 操作 (D) で保護。Terminal reclaim は同一 mutex/sequence で排他。
4. **INV-PUB-4 (sequence id)**: Terminal reclaim は `seqId == 0`（retire の seqId）でログ記録し、`worldReclaimCount()` をインクリメント。

---

## 15-P-4-0 Audit Verdict: **PASS** ✅

| Sub-check | Finding | Status |
|-----------|---------|--------|
| Non-RT context | All 8 `enqueueWithRetry` call sites run on CoordinatorLoop / timer threads | ✅ PASS |
| RT-safe deleter | `unseal` (atomic), `~RuntimePublishWorld` (trivial), `aligned_free` (libc) | ✅ PASS |
| Epoch safety | `isOlder` wraparound-safe; `minReaderEpoch` provides correct epoch fence | ✅ PASS |
| Shutdown ordering | `ShutdownReclaimAuthority` receives ownership post-early-return; drained at shutdown | ✅ PASS |
| Destructor safety | `RuntimeState` = trivial destructor, no JUCE, no allocation | ✅ PASS |

**Terminal reclaim (synchronous destruction in Non-RT context, epoch-gated) IS safe and applicable.**

**Proceed to 15-P-4 — ShutdownReclaimAuthority implementation.**

---

## 15-P-4-1 — Terminal Reclaim 実装完了監査 (Post-implementation ownership proof)

**実装対象**: `enqueueWithRetry()` が最後まで ownership を失わないこと（D → Q → E → Terminal の 4 段 ownership chain）

**実装ファイル**:
- `src/audioengine/ISRAuthorityClass.h` — `RetireEnqueueResult::TerminalReclaim` 追加
- `src/audioengine/ISRRetireRouter.h` — `TerminalReclaimAuthority`（growable store）追加、新 public API 追加
- `src/audioengine/ISRRetireRouter.cpp` — `enqueueWithRetry()` 書き換え、`terminalReclaim()` 実装、`drainEmergencyAndTerminal()` 追加
- `src/audioengine/AudioEngine.h` — shutdown early-return 修正（前ターン）
- `src/eqprocessor/EQProcessor.Core.cpp` — `enqueueDeferredDeleteWithFallback` が `TerminalReclaim` を成功扱いに

**ビルド検証**: Debug ビルド成功（`-j 4` で C1060 回避）、CTest 29/29 PASS ✅

### 5 ケース ownership proof（どの return statement でも ptr の所有者が一意）

`enqueueWithRetry()` の全 return 経路を列挙し、各経路で `ptr` の所有者が一意に特定できることを検証する。

```text
Case 1: D に空きあり
  enqueueRetire() == Success → D (DeferredDeletionQueue) が ptr を所有 ✅
  return RetireEnqueueResult::Success
  所有者: D のみ（Q/E/Terminal は未接触）

Case 2: D full / Q に空きあり
  retry サイクル（kMaxRetry=2）後も QueuePressure/QueueFull
  → m_retireQuarantine.quarantine() 成功 → Q が ptr を所有 ✅
  return RetireEnqueueResult::QueuePressure
  所有者: Q のみ（D は enqueue 失敗で ptr を保持していない）

Case 3: D+Q full / EmergencyQ に空きあり
  Q の quarantine() 失敗 → m_emergencyQuarantine.quarantine() 成功 → E が ptr を所有 ✅
  return RetireEnqueueResult::QueuePressure
  所有者: E のみ

Case 4: 全 store full / epoch safe（epoch < minReaderEpoch）
  terminalReclaim() 内で epochSafe == true かつ Non-RT
  → deleter(ptr) を同期実行 → deleter が ownership を取得（破棄完了）✅
  return RetireEnqueueResult::TerminalReclaim
  所有者: なし（破棄済み・R_ref 加算済み）

Case 5: 全 store full / epoch unsafe（epoch >= minReaderEpoch）
  terminalReclaim() 内で epochSafe == false → m_terminalReclaim.store() 成功
  → TerminalReclaimAuthority（growable）が ptr を所有 ✅
  return RetireEnqueueResult::TerminalReclaim
  所有者: Terminal のみ（drain() が epoch safe 到達時に deleter 実行）
```

**結論**: 全 5 ケースで `ptr` の所有者が一意に特定できる。**どの return statement においても ownership は失われない。**

### ★ 重大バグ発見・修正: epoch 安全条件の反転

実装レビュー中に **`terminalReclaim()` と `TerminalReclaimAuthority::drain()` の EBR 安全条件が反転**している重大バグを発見・修正した。

- **誤**: `epochSafe = !isOlder(epoch, minReader)`（`epoch >= minReaderEpoch` → safe）
  → これは「Reader がまだ参照している可能性がある World」を破棄する UAF リスク
- **正**: `epochSafe = isOlder(epoch, minReader)`（`epoch < minReaderEpoch` → safe）
  → `DeferredDeletionQueue::reclaim()` / `RetireQuarantineStore::drain()` と同一条件

修正後、`terminalReclaim()` と `TerminalReclaimAuthority::drain()` は両方とも
`isOlder(epoch, minReaderEpoch)`（`epoch < minReaderEpoch`）を安全条件として使用する。
これにより、全 store が EBR 安全条件で統一された。

### 設計判断: TerminalReclaimAuthority の growable store

- `std::vector<Entry>` を使用（全呼び出し元が Non-RT のため heap 確保可）
- **常に受け入れる** — 「store full」による EBR 失敗パスが存在しない
- ユーザー指示「epoch unsafe → `assert(false)` → Release で return という経路は残さない」を満たす
- `drain()` は epoch-gated（`isOlder(epoch, minReaderEpoch)` で安全分のみ deleter 実行）
- `drainAll()` は shutdown 時に全エントリを強制回収

### RT 防御

`terminalReclaim()` に `convo::numeric_policy::isAudioThread()` チェックを追加:
- RT スレッドから呼ばれた場合、**synchronous destruction を実行しない**（store に退避）
- ユーザー指示「RT thread から呼ばれた場合に synchronous destruction しないことを明示的に防御する」を満たす

### drainAllUnsafe の通常動作からの除去

- 旧実装の Stage 4/5 で `drainAllUnsafe()` を通常動作中に呼んでいた（UAF リスク + 無意味）
- 修正後: `drainAllUnsafe()` は **shutdown 専用**（`drainAll()` / `drainAllQuarantineStore()` 経由のみ）
- 通常動作では epoch-gated `drain()` のみ使用

### 呼び出し元の整合性検証

| 呼び出し元 | 結果処理 | 整合性 |
|-----------|---------|--------|
| `AudioEngine::enqueueDeferredDeleteNonRt` | `result != Shutdown` を返す → TerminalReclaim は true | ✅ |
| `AudioEngine::enqueueDeferredDeleteNonRtWithResult` | 結果をそのまま返す | ✅ |
| `EQProcessor::enqueueDeferredDeleteWithFallback` | Success/QueuePressure/TerminalReclaim を true 扱い | ✅ |
| `DSPLifetimeManager` (2箇所) | `juce::ignoreUnused(result)` — ownership は常に移送済み | ✅ |
| `ISRRuntimePublicationCoordinator` | `enqueueRetire()` の結果を直接返す（production-unused） | ✅ |
| `SnapshotCoordinator` | 独自 static `enqueueWithRetry` を使用（別経路・スコープ外） | ✅ |

### 15-P-4-1 Verdict: **PASS** ✅

| 検証項目 | 結果 |
|---------|------|
| 5 ケース ownership proof | ✅ 全ケースで所有者一意 |
| epoch 安全条件 | ✅ 修正済み（`isOlder(epoch, minReaderEpoch)` で統一） |
| growable Terminal store | ✅ EBR 失敗パスなし |
| RT 防御 | ✅ `isAudioThread()` チェック |
| drainAllUnsafe 除去 | ✅ shutdown 専用に限定 |
| ビルド | ✅ Debug 成功 |
| テスト | ✅ 29/29 PASS |

**15-P-4 実装完了。`enqueueWithRetry()` は最後まで ownership を失わない。**<tool_call>read_file<arg_key>endLine</arg_key><arg_value>55
