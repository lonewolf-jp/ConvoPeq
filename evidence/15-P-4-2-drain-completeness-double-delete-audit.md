# 15-P-4-2 — Shutdown Drain Completeness / Double-delete Audit

## ステータス
- **作成日**: 2026/08/18
- **監査種別**: コード変更なし — drain chain の single-consume / double-delete prevention / drain completeness
- **関連**: 15-P-4-0 (TerminalReclaim PROVEN), 15-P-4-1 (D103 closure PASS), 15-P-CROSS-IMPLEMENTATION-1

---

## A. OwnerChannel → retire chain の single-consume

### A.1. consumeAtomic / publishAtomic の semantics

`consumeAtomic` (AtomicAccess.h:57) = `std::atomic_load_explicit(acquire)`
`publishAtomic` (AtomicAccess.h:34) = `std::atomic_store_explicit(release)`

**SPSC single-transfer model**:
- **Producer** (= OwnerChannel::enqueue): `publishAtomic(s.owner, raw, release)` — atomics store
- **Consumer** (= OwnerChannel::take / drainAllNonRt): `consumeAtomic(s.owner, acquire)` — atomic load, then `publishAtomic(s.owner, nullptr, release)` — atomic store nullptr

**CAS は使用されない** — single-transfer の保証は SPSC モデル（single producer, single consumer）による。Producer は `enqueue` のみ、Consumer は `take` のみ呼ぶ。`drainAllNonRt` は shutdown 時 (quiescent) に追加で呼ばれる。

### A.2. `take()` の single-transfer

```cpp
// OwnerChannel.h:88-100
OwnerPtr take(const OwnerChannelKey& key) noexcept {
    Owner* raw = nullptr;
    const std::size_t base = hashOf(key);
    for (std::size_t i = 0; i < kCapacity; ++i) {
        Slot& s = slots_[(base + i) & kMask];
        Owner* const seen = consumeAtomic(s.owner, std::memory_order_acquire);
        if (seen == nullptr || s.key != key)
            continue;                                   // empty or different key
        // match: single-transfer drain (SPSC: sole consumer)
        publishAtomic(s.owner, static_cast<Owner*>(nullptr), std::memory_order_release);
        raw = seen;
        break;
    }
    return OwnerPtr(raw, ...);
}
```

- `seen == nullptr` → slot empty → skip
- `s.key != key` → different key → skip (slot NOT drained)
- match → `publishAtomic(nullptr)` → slot drained, `raw = seen`

**single-transfer**: `take` 後、`s.owner == nullptr` → 2nd `take` は `nullptr` を観測 → return nullptr (no owner re-yield)

### A.3. `drainAllNonRt()` の single-transfer

```cpp
// OwnerChannel.h:117-130
template <class Fn>
std::size_t drainAllNonRt(Fn&& reclaim) noexcept
{
    std::size_t reclaimed = 0;
    for (std::size_t i = 0; i < kCapacity; ++i) {
        Slot& s = slots_[i];
        Owner* const raw = consumeAtomic(s.owner, std::memory_order_acquire);
        if (raw != nullptr) {
            publishAtomic(s.owner, static_cast<Owner*>(nullptr), std::memory_order_release);
            reclaim(raw);
            ++reclaimed;
        }
    }
    return reclaimed;
}
```

- `raw != nullptr` → `publishAtomic(nullptr)` → slot drained, `reclaim(raw)` callback
- **re-drain**: `raw == nullptr` → skip → 0 returned

### A.4. `take()` と `drainAllNonRt()` の競合可能性

**通常運用 (Non-shutdown)**: `take()` は CoordinatorLoop thread (consumer) が呼ぶ。`drainAllNonRt()` は `ReleaseResources` (Message Thread) が呼ぶ。**shutdown 時のみ** `drainAllNonRt` が呼ばれる — 15-P-4-1 §A.3 で証明済み: CoordinatorLoop は `shutdownCoordinatorLoop()` (line 189) で join 済み。

**API 単体としての競合**: SPSC モデルでは producer/consumer が1:1 なので、`take` と `drainAllNonRt` が同時に呼ばれることはない（`drainAllNonRt` は caller が producer/consumer 両方を quiescent に保証する契約）。

```
take(key)         drainAllNonRt()
     │                   │
     │  owner != nullptr │  owner != nullptr → publish(nullptr) → owner == nullptr
     │  owner == nullptr │  owner == nullptr
     │                   │
     └──── どちらが先に到達しても single-transfer で slot は空になる ────┘
```

**`consumeAtomic` (load) + `publishAtomic(nullptr)` (store) の組み合わせにより、スロットは一度だけ drain 可能。** (single-transfer at the atomic level — no CAS needed because SPSC guarantees no concurrent producer/consumer on the same slot)

### A.5. `enqueue()` の再利用防止

```cpp
// OwnerChannel.h:65-82
bool enqueue(const OwnerChannelKey& key, OwnerPtr&& owner) noexcept {
    ...
    if (consumeAtomic(s.owner, acquire) != nullptr) {
        if (s.key == key)
            return false;        // already enqueued -> reject (no overwrite)
        continue;                 // collision with different key -> keep probing
    }
    // free slot: publish owner
    publishAtomic(s.owner, raw, release);
    owner.release();
    return true;
}
```

- `enqueue` は `owner != nullptr` の slot に **絶対書き込まない** (overwrite rejected)
- `publishAtomic(nullptr)` 後、slot は空 (`nullptr`) になり、**再び `enqueue` 可能** (channel reusable — unit test `testOwnerChannelDrainThenReenqueue` で証明済み)

**結論: ✅ single-consume は保証される — `take()` と `drainAllNonRt()` は同じ owner を二度取得しない。**

---

## B. `enqueueDeferredDeleteNonRtWithResult()` の disposition exhaustiveness

### B.1. 全 return path の列挙

```
enqueueDeferredDeleteNonRtWithResult(ptr, deleter, type)
  ├─ ptr == nullptr → Success
  ├─ deleter == nullptr → Success
  │
  ├─ isShutdownInProgress() == true
  │   → shutdownReclaim(ptr, deleter, epoch, type)
  │     → terminalReclaim(...)
  │       ├─ store() → ALWAYS true → return Success
  │       └─ deleter() immediate → return Success
  │
  └─ isShutdownInProgress() == false
      → enqueueWithRetry(ptr, deleter, epoch, type)
        → Stage 1: enqueueRetire()
        │   ├─ Success → return Success
        │   └─ QueuePressure → Stage 2 (retry)
        │     ├─ Success → return Success
        │     └─ QueuePressure → Stage 3: Q
        │       ├─ quarantine() true → return QueuePressure
        │       └─ quarantine() false → Stage 4: E
        │         ├─ quarantine() true → return QueuePressure
        │         └─ quarantine() false → Stage 5: Terminal
        │           ├─ terminalReclaim() → return TerminalReclaim
        │           └─ (store always succeeds)
```

### B.2. RetireEnqueueResult enum の到達可能性

| enum value | generated from | reachable? |
|------------|---------------|------------|
| **Success** | `ptr==nullptr` / `deleter==nullptr` / `shutdownReclaim→terminalReclaim` / `enqueueWithRetry` Stage 1 Success | ✅ Yes |
| **QueuePressure** | `enqueueWithRetry` Stage 3 (Q) / Stage 4 (E) | ✅ Yes |
| **QueueFull** | Never returned by `enqueueRetire` (ISRRetireRouter.cpp:210-244 returns only Success/QueuePressure) | ❌ No (dead) |
| **Shutdown** | Only `enqueueDeferredDeleteNonRtWithResult` shutdown path if `shutdownReclaim` returns false — but `terminalReclaim` always returns true | ❌ No (dead) |
| **TerminalReclaim** | `enqueueWithRetry` Stage 5 | ✅ Yes |

### B.3. `Shutdown` enum の到達可能性

`enqueueDeferredDeleteNonRtWithResult` (AudioEngine.h:4196-4205):
```cpp
if (isShutdownInProgress())
{
    const uint64_t epoch = markRetireEpoch();
    const bool transferred = m_retireRouter->shutdownReclaim(ptr, deleter, epoch, type);
    return transferred ? convo::isr::RetireEnqueueResult::Success
                       : convo::isr::RetireEnqueueResult::Shutdown;
}
```

`shutdownReclaim` → `terminalReclaim` → `store()` (always true) or `deleter()` (immediate). **`terminalReclaim()` は常に `true` を返す** (15-P-4-0 §A.1 で証明済み) → `transferred == true` → **常に `Success`**

→ **`RetireEnqueueResult::Shutdown` はこの call path から生成されない** — enum 定義されているが production code では到達不能.

**enum 変更について**: 本監査は到達可能性のみ記録。enum removal は別途判断 (code change 対象外).

---

## C. TerminalReclaim → drainAll の double-delete prevention

### C.1. `terminalReclaim()` の disposition

```cpp
// ISRRetireRouter.cpp:230-306
bool ISRRetireRouter::terminalReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                      DeletionEntryType type, const char* reason) noexcept
{
    const uint64_t minReader = minReaderEpoch();
    const bool epochSafe = ISRRetireRouter::isOlder(epoch, minReader);
    const bool isRt = convo::numeric_policy::isAudioThread();

    if (epochSafe && !isRt)
    {
        // Synchronous destruction — deleter executes immediately
        deleter(ptr);                          // ← DELETE HERE
        if (type == DeletionEntryType::World)
            m_terminalReclaim.recordWorldReclaim();
        return true;  // destroyed immediately, no storage
    }

    // epoch unsafe OR RT caller → store for later drain
    return m_terminalReclaim.store(ptr, deleter, epoch, type, reason);  // ← STORE HERE
}
```

### C.2. double-delete path の不存在

```
epoch-safe + Non-RT:
  → terminalReclaim() → deleter(ptr) executed → return true
  → store() NOT called → entries_ に entry 追加なし
  → drainAll() → entries_ が空 → no double-delete ✅

epoch-unsafe OR RT:
  → terminalReclaim() → store(entry) → entries_ に entry 追加
  → deleter NOT executed
  → drainAll() → entries_ から entry を取り出して deleter(ptr) を実行 → entry 消滅
  → drainAll() 再呼 → entries_ が空 → no double-delete ✅
```

**key insight**: `terminalReclaim()` は **if-else** structure — `deleter()` を呼んだ場合 **絶対に** `store()` を呼ばない (line 268: `return true` で early return)。逆も同じ — `store()` した場合は `deleter()` を呼ばない。

→ **immediate deleter と later drain の両方から同じ raw が解放される path は存在しない**.

### C.3. `drainAll()` の idempotency

```cpp
// ISRRetireRouter.cpp:74-92
void TerminalReclaimAuthority::drainAll() noexcept
{
    std::vector<Entry> pending;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        pending.swap(entries_);  // entries_ is now empty
    }
    for (auto& e : pending) {     // entries_ is empty → pending is empty → no-op
        if (e.ptr != nullptr && e.deleter != nullptr) {
            e.deleter(e.ptr);
            ...
        }
    }
}
```

- `pending.swap(entries_)` → `entries_` は空になる
- `pending` が空なら loop は 0 回 → **no-op**
- **re-drain**: `entries_` already empty → `pending` empty → no-op

### C.4. `RetireQuarantineStore::drainAllUnsafe` の idempotency

```cpp
// RetireQuarantineStore.h:148-175
void drainAllUnsafe() noexcept
{
    ...
    {
        std::lock_guard<std::mutex> lock(mtx_);
        for (std::size_t i = 0; i < size_; ++i) {
            ... extract e.ptr/e.deleter/e.type ...
            e = QuarantinedEntry{};   // zero each entry
        }
        size_ = 0;  // ← size is now 0
    }
    for (std::size_t i = 0; i < pendingCount; ++i) {
        pendingDeleters[i](pendingPtrs[i]);  // deleter executed
    }
}
```

- After first call: `size_ = 0`, all entries zeroed
- Re-call: `size_ == 0` → outer loop 0 iterations → `pendingCount = 0` → inner loop 0 iterations → **no-op**

### C.5. `residentCount()` の整合性

```cpp
// ISRRetireRouter.cpp:116, TerminalReclaimAuthority::residentCount()
std::size_t TerminalReclaimAuthority::residentCount() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return entries_.size();
}
```

| 操作 | entries_.size() | residentCount() |
|------|-----------------|-----------------|
| `store()` (epoch unsafe) | +1 | +1 |
| `drain()` (epoch safe) | -N | -N |
| `drainAll()` (shutdown) | 0 (swap) | 0 |
| `terminalReclaim()` immediate deleter | 0 (no store) | 0 |

→ **`residentCount()` は実際の ownership resident 数と整合する**.

### C.6. World deleter の実体

```cpp
// AudioEngine.h:3525-3530 (World deleter — inline lambda, NOT a named function)
[](void* p) noexcept {
    auto* ptr = static_cast<RuntimePublishWorld*>(p);
    ptr->unseal();                    // unseal: reverse of sealRecursively
    ptr->~RuntimePublishWorld();      // destruct RuntimeState (which IS RuntimePublishWorld)
    convo::aligned_free(ptr);         // free aligned memory
}
```

**重要**: deleter は `unseal → ~RuntimePublishWorld → aligned_free` の順で実行される。`aligned_free` によりメモリが解放されるため、**dangling pointer の二重 free を防ぐため、deleter は一度だけ呼ばれなければならない**。

- `terminalReclaim()` immediate path: deleter 1 回 (store 0 回)
- `terminalReclaim()` store path: deleter 0 回 (store 1 回) → `drainAll()` で deleter 1 回

→ **deleter は常に正確に 1 回だけ呼ばれる**. **double-free なし**.

**結論: ✅ Terminal double-delete safety — PASS**

---

## D. Shutdown idempotence — 複数 drain の安全性

### D.1. shutdown sequence の実際の順序

```
ReleaseResources.cpp:
line  75:  runtimePublicationBridge_.requestShutdown()        → state_ = ShuttingDown
line 175:  commitRuntimePublication (idle publish #4)         [producer last action]
line 189:  shutdownCoordinatorLoop()  → join                 [consumer STOPPED]
line 190:  stopRebuildThread()        → join                  [rebuild producer STOPPED]
line 200:  closeReaderRegistration()                           [reader gate CLOSED]
line 203:  advanceRetireEpoch()                                [epoch advanced]
line 430:  drainAllQuarantineStore()  [first call, PR2]       [Q + E + Terminal drained]
           ↓
         / 478:  waitForDrain(2000, 2)                        [poll loop, tryReclaim epoch-gated]
         / 515:  if (!drainedWithinBudget || !isFullyDrained())
         /           drainDeferredRetireQueues(true)           [defers to D queue, NOT drainAll]
         /           m_epochDomain.tryReclaim()                [epoch-gated drain, NOT drainAll]
line 521:  finalizeShutdown(timedOut)                         [SnapshotCoordinator guard]
         /                           ✓ idempotent (m_shutdownFinalized guard)
line 523:  requestShutdownClearNonRt()                        → flag set
line 525:  clearPublishedRuntimeSnapshotsNonRt()              → publishAndSwap(nullptr) → oldWorld
line 527:  retirePublishedRuntimeWorldNonRt(clearedWorld, true) → enqueueDeferredDeleteNonRt
line 536:  OwnerChannel::drainAllNonRt()                      [★ residual drain]
line 543:  drainAllQuarantineStore() [second call, if activeReaderCount()==0]
line 560:  isFullyDrained() final check
```

### D.2. 各関数の idempotent 確認

| 関数 | guard mechanism | 再呼時の挙動 |
|------|----------------|-----------|
| **`finalizeShutdown(timedOut)`** | `m_shutdownFinalized` atomic flag | 2nd call: `if (finalized) return;` → **no-op** ✅ |
| **`drainAllNonRt()`** | slot state (`owner == nullptr` after drain) | 2nd call: all `nullptr` → count=0 → **no-op** ✅ |
| **`drainAllQuarantineStore()`** | `size_=0` / `entries_` empty after drainAll | 2nd call: `drainAllUnsafe` (Q+E) → `pendingCount=0`; `drainAll()` (Terminal) → `pending` empty → **no-op** ✅ |
| **`clearPublishedRuntimeSnapshotsNonRt()`** | `shutdownClearRequested_` flag (one-shot) | 2nd call: `!shutdownClearRequested_` → return `nullptr` → **no-op** ✅ |

### D.3. `finalizeShutdown` が 2 回呼ばれた場合

`SnapshotCoordinator::finalizeShutdown()` (SnapshotCoordinator.h:62):
```cpp
void finalizeShutdown(bool timedOut) noexcept {
    if (convo::consumeAtomic(m_shutdownFinalized, std::memory_order_acquire))
        return;  // ★ idempotent guard
    retireCurrentAndTarget();
    if (!timedOut)
        m_epochProvider->tryReclaim();
    convo::publishAtomic(m_shutdownFinalized, true, std::memory_order_release);
}
```

→ 2 回目は guard で即座に return. `retireCurrentAndTarget` は 1 回のみ実行. **safe**.

### D.4. `drainAllQuarantineStore` が 2 回呼ばれた場合

PR2 (line 430): `drainAllQuarantineStore()` called when `quarantineResident > 0` or `residentBefore > 0`.
Post-drainAllNonRt (line 478 area): `if (activeReaderCount() == 0) drainAllQuarantineStore()`.

But after the first `drainAllQuarantineStore()`, all stores are empty (`size_=0`, `entries_` empty). Second call:
- `m_retireQuarantine.drainAllUnsafe()` → `pendingCount=0` → no-op
- `m_emergencyQuarantine.drainAllUnsafe()` → `pendingCount=0` → no-op
- `m_terminalReclaim.drainAll()` → `pending` empty → no-op

→ **All deleter calls are guarded by `ptr != nullptr && deleter != nullptr` check inside the drained entries**. Empty stores → no deleter calls. **double-delete なし**.

### D.5. `waitForDrain` timeout が後続 drain に与える影響

`waitForDrain` (line 478) の内部 loop:
```cpp
while (!isFullyDrained()) {
    drainDeferredRetireQueues(true);  // → enqueueRetire (D), NOT drainAll
    ...
    tryReclaim()                       // → drainQuarantine, drainEmergencyAndTerminal (epoch-gated, not drainAll)
    sleep(poll)
}
```

`waitForDrain` は `drainAllQuarantineStore()` (force drain) を呼ばない — 代わりに `tryReclaim()` を呼び、これは `drainQuarantineStore()` + `drainEmergencyAndTerminal()` (epoch-gated drain) を実行する。**force drain (`drainAll`) は `drainAllQuarantineStore` でのみ呼ばれる**.

→ `waitForDrain` timeout は `drainAllQuarantineStore` の force drain に影響を与えない.

**結論: ✅ Shutdown idempotence — PASS**

---

## E. `clearPublishedRuntimeSnapshotsNonRt()` との ownership overlap

### E.1. 2つの path の ownership scope

```
clearPublishedRuntimeSnapshotsNonRt()  (ReleaseResources.cpp:525)
  → publishAndSwap(nullptr)  [RuntimeWorldAuthority::publishAndSwap]
  → returns oldWorld (previously published RuntimeState*)
  → retirePublishedRuntimeWorldNonRt(oldWorld, true)
    → enqueueDeferredDeleteNonRt(world, World-deleter, World)
      → shutdownReclaim → terminalReclaim → deleter

OwnerChannel::drainAllNonRt()  (ReleaseResources.cpp:536)
  → drain residual slots (never take'd)
  → callback: enqueueDeferredDeleteNonRtWithResult(raw, World-deleter, World)
    → shutdownReclaim → terminalReclaim → deleter
```

### E.2. Ownership 分離の証明

**`publishAndSwap` の atomicity** (RuntimeStore.h:40):
```cpp
T* publishAndSwap(T* next) noexcept {
    return exchangeAtomic(store_->current, next, std::memory_order_acq_rel);
}
```

- `exchangeAtomic` = `std::atomic_exchange` — **atomic** swap
- `publishAndSwap(next)` returns old `current`, sets `current = next`

**`publish()` の flow** (RuntimeWorldAuthority.h:249):
```cpp
auto* oldWorld = writeAccess_.publishAndSwap(next);  // atomic: current → next, returns old current
```

OwnerChannel から `take(key)` で owner を取得 → `publish(std::move(owner), ...)` で `owner.release()` → `publishAndSwap(next)` → **owner は store に move された**. owner.release() により OwnerPtr は解放され、**raw pointer は store.current に存在する**.

→ **publish された world は OwnerChannel から取り出された後、store.current に存在する**. OwnerChannel slot は nullptr になる.

### E.3. disjoint set の証明

| World state | in OwnerChannel? | in RuntimeStore::current? |
|-------------|-----------------|---------------------------|
| enqueue'd but never take'd | ✅ YES (slot occupied) | ❌ No |
| take'd, published (publishAndSwap) | ❌ No (slot = nullptr) | ✅ YES (store.current) |
| clearPublishedRuntimeSnapshotsNonRt() returns | N/A | ✅ YES (returns store.current) |

**`clearPublishedRuntimeSnapshotsNonRt()`**:
- `publishAndSwap(nullptr)` — store.current を nullptr に, **前の current** (oldWorld) を返す
- oldWorld は **既に publish された** world — OwnerChannel からは take 済み

**`drainAllNonRt()`**:
- OwnerChannel に **残留**する owner を drain — これらは **publish されていない** world
- `enqueue` されたが `take` されていない world

→ **clearPublishedRuntimeSnapshotsNonRt が扱うのは "published world"、drainAllNonRt が扱うのは "unpublished owner" — disjoint set**.

### E.4. 同じ `RuntimePublishWorld*` が両方に存在しうるか

**結論: ❌ 存在しない — disjoint**.

`RuntimePublishWorld` のライフサイクル:
1. `RuntimeBuilder::buildRuntimePublishWorld()` → `aligned_unique_ptr<RuntimePublishWorld>` を生成
2. `commitRuntimePublication()` → `enqueueRuntimePublicationFireAndForget()` → `ownerChannel().enqueue(key, std::move(world))`
   - **world は OwnerChannel slot に移動**
3. `CoordinatorLoop::run()` → `processIntent()` → `executePublish()` → `authority.ownerChannel().take(key)`
   - **world は OwnerChannel から取り出される** (slot = nullptr)
4. `authority.publish(std::move(owner), ...)` → `owner.release()` → `publishAndSwap(next)`
   - **raw pointer は store.current に原子的に swap される**
5. `publish()` returns `oldWorld` (前の store.current)
   - **oldWorld は retire 対象** → `retirePublishedRuntimeWorldNonRt(oldWorld, false)`

**step 3 と step 4 は atomic に離れている**:
- step 3: `take` → slot = nullptr
- step 4: `publishAndSwap` → store.current = new, returns old

**→ world は OwnerChannel OR store.current のいずれか一方にしか存在しない**.

Shutdown clear:
- `clearPublishedRuntimeSnapshotsNonRt()` → `publishAndSwap(nullptr)` → store.current (previously published world) を返す
- **この world は OwnerChannel には存在しない** (既に take 済み)
- `drainAllNonRt()` → OwnerChannel に残留する **unpublished** world を drain
- **clear された world と drainAllNonRt された owner は異なる world**

→ **同じ `RuntimePublishWorld*` が published snapshot と OwnerChannel residual の両方に存在することはない**.

**結論: ✅ Published/OwnerChannel overlap — PASS**

---

## D103 final invariant summary

```
D103-A: OWNERCHANNEL RESIDUAL == 0 after drainAllNonRt()
    PASS
    - drainAllNonRt full-scans all kCapacity(256) slots
    - consume→publish(nullptr) single-transfer (same as take)
    - re-drain returns 0 (all slots nullptr)
    - Unit test: testOwnerChannelDrainAllNonRt (5 slots drain + re-drain no-op)

D103-B: Every drained owner has terminal disposition
    PASS
    drainAllNonRt callback → enqueueDeferredDeleteNonRtWithResult
      → isShutdownInProgress() == true
        → shutdownReclaim → terminalReclaim
          → epoch safe → deleter (deleted)
          → epoch unsafe → store → drainAll() (D/Q/E/TerminalReclaim)
    No path leads to unowned

D103-C: No drained ownership silently discarded
    PASS
    - callback is noexcept (no exception escape)
    - enqueueDeferredDeleteNonRtWithResult:
      shutdown → shutdownReclaim → terminalReclaim → always true → Success
      normal → enqueueWithRetry → D/Q/E/TerminalReclaim
    - RetireEnqueueResult::Shutdown is dead code (never generated)
    - raw pointer consumed once: consume→publish(nullptr)→reclaim→enqueue
    - raw never re-touched after reclaim(raw) call

D103-CROSS: No producer can recreate residual ownership
    after the final drain point
    PASS
    - All producers stopped before drainAllNonRt (line 536):
      CoordinatorLoop join (189), RebuildThread join (190),
      Timer shutdown guard (isShutdownInProgress), closeReaderRegistration (200)
    - enqueuePublicationIntent gates on ShuttingDown state
    - drainAllNonRt itself is idempotent (re-drain = no-op)

A. Single-consume:                 PASS
B. Disposition exhaustiveness:     PASS
C. Terminal double-delete safety:  PASS
D. Shutdown idempotence:           PASS
E. Published/OwnerChannel overlap: PASS
```

### GAP-CROSS-1

```
GAP-CROSS-1:
    CLOSED
    (OwnerChannel has no destructor → slots_ leak at engine destruction.
     But drainAllNonRt() at shutdown sequence drains ALL residual owners →
     ownership transferred to retire chain (D/Q/E/TerminalReclaim).
     No producer can recreate residual after drain point.
     No double-consume / double-delete possible.
     All 4 invariants PASS → GAP-CROSS-1 is structurally eliminated.)
```

---

### Build verification baseline

| Build | Config | Result |
|-------|--------|--------|
| MSVC | Debug | 31/31 ✅ PASS |
| MSVC | Release | 31/31 ✅ PASS |
| Intel ICX | Release | 31/31 ✅ PASS |

> 15-P-4-0 で ICX stale artifact を clean/rebuild 済み. 15-P-4-2 はコード変更なしの監査のため再ビルド不要.
