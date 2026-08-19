# 15-P-11: Pre-Publication / Non-Shutdown Residual-Owner Destruction Boundary Audit

## Scope

Resolve the three `INVESTIGATE` items from 15-P-10. Determine whether each non-shutdown destruction
path constitutes **pre-ownership destruction** (PASS) or **terminal-authority bypass after ownership
entry** (GAP).

**Method**: Full call-site enumeration with ownership-state tracing. Every suspect destruction path
was traced through the concrete type flow — `RuntimeState*` and `DSPCore*` — from construction to
final `::operator delete`. The ownership state machine is:

```text
Created → Caller-owned → OwnerChannel-owned → Published → Retire D → Q+E → Terminal → Destroyed
                    ↑                         ↑
         enqueue(成功)              publishAndSwap(成功)
```

**Key fact**: `AlignedObjectDeleter::operator()` calls `ptr->~T()` + `convo::aligned_free(ptr)` —
this is a **direct destruction** that **does not** route through the terminal-retire chain
(`enqueueDeferredDeleteNonRt → drainAllNonRt → terminalReclaim`). All `RuntimeState` and
`RuntimePublishWorld` objects are heap-allocated via `aligned_make_unique` and wrapped in
`aligned_unique_ptr<T>` whose deleter is `AlignedObjectDeleter<T>`. There is exactly **one**
mechanism for each: the deleter. There is **no** `delete T` in production source (verified by
workspace-wide grep: zero hits for `delete RuntimeState`, `delete DSPCore`, or
`delete RuntimePublishWorld` in `src/`).

## Audit Results

| # | Path / Object | Authority entered? | Owner at destruction | Destruction mechanism | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | `destroyRolledBackDSP(newDSPResolved)` — `RuntimePublicationOrchestrator.cpp:274` | ❌ Never retired | Caller (`trySubmitImpl` temporary via `lifetime_`) | `destroyDSPCoreNode` → direct `delete` via `AlignedObjectDeleter` on internal buffers + `aligned_free` | **PASS** — Pre-ownership |
| 2 | `CallerDestroy` → `enqueue` fails — `AudioEngine.h:4550` → parameter RAII in `enqueueRuntimePublicationFireAndForget` | ❌ Never entered OwnerChannel | `aligned_unique_ptr<const RuntimePublishWorld> world` parameter (RAII) | `AlignedObjectDeleter` → `~RuntimePublishWorld()` + `aligned_free` | **PASS** — Pre-ownership |
| 3 | `CallerDestroy` → `intentQueue_.try_enqueue` fails — `AudioEngine.h:4570` → `take()` discard → parameter RAII | ⚠️ Entered OwnerChannel (enqueue succeeded) then recalled via `take()` | `RuntimeOwner` temporary returned by `take()` (RAII) | `AlignedObjectDeleter` → `~RuntimePublishWorld()` + `aligned_free` | **PASS** — Ownership rollback |
| 4 | `FrozenRuntimeWorld::~FrozenRuntimeWorld()` — `state_` member (aligned_unique_ptr) | ❌ Never entered (releaseState always called first) | `aligned_unique_ptr<RuntimeState> state_` member (RAII) | Would be `AlignedObjectDeleter`, but `state_` is null after `releaseState()` | **PASS** — Destructor is no-op |
| 5 | `commitRuntimePublication` — `world` parameter after `CallerDestroy` return | ❌ (same as #2/#3) | `aligned_unique_ptr` parameter already destructed in callee | Already destroyed by #2/#3 callee | **PASS** — No residual owner |

## Detailed Trace per Path

### Path 1: `destroyRolledBackDSP` — `RuntimePublicationOrchestrator.cpp:274`

```cpp
// RuntimePublicationOrchestrator.cpp (trySubmitImpl, ~line 253-274)
auto frozen = convo::aligned_make_unique<convo::FrozenRuntimeWorld>(
    convo::aligned_unique_ptr<RuntimeState>(const_cast<RuntimeState*>(worldOwner.release())));

auto result = executor_.publish(engine_, std::move(frozen), req.newDSP, oldHandle);
if (result != PublishResult::Success) {
    if (newDSPResolved != nullptr)
        lifetime_.destroyRolledBackDSP(newDSPResolved);   // ← line 274
    ...
    return PublicationAdmission::Decision::RejectedNotFinalized;
}
```

**Ownership trace for `newDSPResolved`:**

1. `newDSPResolved` is `req.newDSP` (resolved earlier via `resolveDSPHandle`). This is a caller-owned
   pointer from the handle table — **never retired** (retire happens via `lifetime_.retire(handle)`,
   not by destroying the pointer).
2. `req.newDSP` is passed to `executor_.publish(engine_, frozen, req.newDSP, oldHandle)` → enters
   `PublicationExecutor::executePublish` (line 34-105).
3. Inside `executePublish`: `rawState = frozen->releaseState()` always executes (line 34). On
   `PublishResult::PublishFailed`, `executePublish` returns without calling `authority.publish()`
   (line 103: `result.stage != Committed` → early return at PublicationExecutor.cpp:70).
4. `newDSPResolved` is **NOT** retired on the publish failure path — the comment at line 272
   confirms: "work70 Phase2: commitRuntimePublication の ScopeExit が Handle を rollback 済み
  （Reclaimed）。したがって retireDSPHandleForRuntime は false を返す"
5. Therefore `destroyRolledBackDSP(newDSPResolved)` destroys a DSPCore that was **never enqueued
   into D→Q→E→T** — it is pre-authority.

**Implementation** (`DSPLifetimeManager.cpp:119`):

```cpp
void DSPLifetimeManager::destroyRolledBackDSP(void* dsp) noexcept {
    auto* core = static_cast<AudioEngine::DSPCore*>(dsp);
    core->~DSPCore();                          // direct destruction
    convo::aligned_free(core);
}
```

**Classification**: `newDSPResolved` was resolved from a handle (caller-owned), never retired into
the terminal chain. Destruction is direct but ownership never entered authority.

**Verdict: PASS — Pre-ownership destruction.**

---

### Path 2: `CallerDestroy` — `enqueue` fails — `AudioEngine.h:4550`

```cpp
// AudioEngine.h, enqueueRuntimePublicationFireAndForget (lines ~4525-4550)
auto result = enqueueRuntimePublicationFireAndForget(
    convo::aligned_unique_ptr<const RuntimePublishWorld>(
        static_cast<const RuntimePublishWorld*>(rawState)),  // stateOwner → callee
    ...
);
```

Inside `enqueueRuntimePublicationFireAndForget` (AudioEngine.h:4515-4572):

```cpp
// line 4548-4550:
if (!worldAuthority_.ownerChannel().enqueue(
        convo::isr::OwnerChannelKey{ seqId, epoch, mappedGen }, std::move(world)))
{
    worldAuthority_.registry().unregister(seqId);
    return { convo::PublishStageResult::Failed, OwnershipDisposition::CallerDestroy };
}
```

**Ownership trace:**

1. `rawState` was extracted via `frozen->releaseState()` (PublicationExecutor.cpp:34) — ownership
   transferred from FrozenRuntimeWorld to `stateOwner` (unique_ptr).
2. `stateOwner` is moved into `enqueueRuntimePublicationFireAndForget` as parameter `world`.
3. `enqueue(key, std::move(world))` — `OwnerPtr&& owner` is an rvalue reference to `world`
   (parameter). `enqueue` takes the reference, calls `enqueueInternal` which on failure returns false
   WITHOUT calling `owner.release()`.
4. `world` parameter still holds the pointer (rvalue ref doesn't destroy).
5. Function returns `CallerDestroy`. **`world` parameter destructs** → `AlignedObjectDeleter` fires
   → `~RuntimePublishWorld()` + `aligned_free`.

The RuntimeState was **never transferred to OwnerChannel** (enqueue failed). It was caller-owned
(the temporary parameter). Destruction via RAII is the `CallerDestroy` contract.

**Question**: Does `commitRuntimePublication` (the wrapper) also attempt destruction?

```cpp
// AudioEngine.h commitRuntimePublication (line ~4591):
auto result = enqueueRuntimePublicationFireAndForget(std::move(world), ...);
// result.ownership == CallerDestroy
// → no explicit destroy; just returns result
```

No — `commitRuntimePublication` moved `world` into the callee (already null after move). The callee
destroyed it via parameter RAII. No double-destroy, no leak.

**Verdict: PASS — Pre-ownership destruction.**

---

### Path 3: `CallerDestroy` — `intentQueue_.try_enqueue` fails — `AudioEngine.h:4570`

```cpp
// AudioEngine.h line 4562-4572:
if (!runtimePublicationBridge_.enqueuePublicationIntent(intent))
{
    // キュー full: 移譲した Owner を取り戻し、registry をクリアして rollback に委ねる。
    (void)worldAuthority_.ownerChannel().take(
        convo::isr::OwnerChannelKey{ seqId, epoch, mappedGen });
    worldAuthority_.registry().unregister(seqId);
    return { convo::PublishStageResult::Failed, OwnershipDisposition::CallerDestroy };
}
```

**Ownership trace:**

1. `enqueue(key, std::move(world))` **succeeded** at line 4548 → RuntimeState ownership **transferred
   to OwnerChannel** (D→Q→E→T chain entry point).
2. `intentQueue_.try_enqueue(intent)` fails → `take(key)` is called to **reclaim** the owner from
   the channel.
3. `take(key)` returns a `RuntimeOwner` temporary (line 86 of OwnerChannel.h: "take single-transfer:
   consume → publish nullptr"). The temporary holds the RuntimeState pointer.
4. `(void)worldAuthority_.ownerChannel().take(...)` — return value discarded → `RuntimeOwner`
   temporary destructs → `AlignedObjectDeleter` fires → `~RuntimePublishWorld()` + `aligned_free`.

The RuntimeState **was** in OwnerChannel (authority entered), but was **recalled** via `take()` and
then discarded by the caller. This is an **ownership rollback** — the authority entry was reversed
before any retire/D→Q→E→T processing occurred (the object was in the channel's enqueue slot, not
yet in the `mpsc` retire queue).

**Classification per user's framework**: The 15-P-10 conversation summary defines this as
"Ownership rollback" (entering authority then rolling back). The user's instruction states:
"did ownership enter OwnerChannel/Terminal before the direct destruction?" Answer: it entered
OwnerChannel but was recalled before entering Terminal (D→Q→E→T). The destruction mechanism
(AlignedObjectDeleter) bypasses the terminal chain, but the object was recalled to caller-owned
status by `take()`.

**Critical detail**: `take()` is the channel's documented rollback mechanism (single-transfer:
consume → publish nullptr). The caller is explicitly reclaiming ownership. The subsequent
discard is the caller's right. The terminal authority was not bypassed — the authority was
**never engaged** for this object (it sat in the channel slot, never enqueued into the retire
queue).

**Verdict: PASS — Ownership rollback** (not pre-ownership, but authority was reclaimed before D→Q→E→T).

---

### Path 4: `FrozenRuntimeWorld` destructor — `state_` member

```cpp
// FrozenRuntimeWorld.cpp
FrozenRuntimeWorld::~FrozenRuntimeWorld() {
    if (state_) {
        state_->unseal();
    }
    // state_ is aligned_unique_ptr<RuntimeState> — member destructor fires after body
}
```

**Construction site**: `RuntimePublicationOrchestrator.cpp:255` — the **only** production site.

```cpp
auto frozen = convo::aligned_make_unique<convo::FrozenRuntimeWorld>(
    convo::aligned_unique_ptr<RuntimeState>(const_cast<RuntimeState*>(worldOwner.release())));
auto result = executor_.publish(engine_, std::move(frozen), req.newDSP, oldHandle);
```

Inside `PublicationExecutor::executePublish` (PublicationExecutor.cpp:34):

```cpp
auto* rawState = frozen->releaseState();    // ALWAYS called before any return
if (rawState == nullptr) return PublishResult::PublishFailed;
auto stateOwner = convo::aligned_unique_ptr<RuntimeState>(rawState);
```

`releaseState()` (FrozenRuntimeWorld.h:56) sets `state_ = nullptr` and returns the pointer. Since
`executePublish` line 24 checks `if (!frozen) return PublishResult::PublishFailed;` **before**
calling `releaseState()`, and `frozen` is always non-null (constructed at line 255 with assertion),
`releaseState()` **always executes successfully** in production.

After `releaseState()`: `state_` is null → `FrozenRuntimeWorld::~FrozenRuntimeWorld()` body is
no-op (if-guard) → member `state_` destructor is no-op (null). The raw pointer was wrapped in
`stateOwner` and moved into `commitRuntimePublication`/`enqueueRuntimePublicationFireAndForget`.

There is **no** production path where `FrozenRuntimeWorld` is destroyed without `releaseState()`.
(Verified: only 1 construction site, `releaseState()` is unconditionally called at PublicationExecutor.cpp:34.)

**Verdict: PASS — No direct destruction path exists.**

---

### Path 5: `commitRuntimePublication` — residual `world` after `CallerDestroy`

```cpp
// AudioEngine.h commitRuntimePublication (line ~4591):
auto result = enqueueRuntimePublicationFireAndForget(std::move(world), regCtx, oldHandle);
if (result.stage == Failed && result.ownership == CallerDestroy) {
    // No explicit destroy of world — world in this scope is null (moved into callee)
}
return result;
```

After `std::move(world)` into the callee, `commitRuntimePublication`'s `world` is in moved-from
state (null). The callee (`enqueueRuntimePublicationFireAndForget`) already handled destruction via
parameter RAII (Path 2 or Path 3). `commitRuntimePublication` does NOT re-destroy.

**Verdict: PASS — No residual owner.**

## Non-Findings

These destruction mechanisms were investigated and confirmed NOT to be on non-shutdown paths:

| Mechanism | Status |
| --- | --- |
| `enqueueDeferredDeleteNonRtNonRt` (AudioEngine.h:3525) | Only called by `retirePublishedRuntimeWorldNonRt` (shutdown path — `AudioEngine.Processing.ReleaseResources.cpp:542`) and `retireRejectedRuntimeWorldNonRt` (bootstrap). Routes through `drainAllNonRt` → terminal. ✅ |
| `DSPLifetimeManager::retire` / `retireByHandle` (DSPLifetimeManager.cpp:50, 97) | Only destroys via `enqueueWithRetry` → terminal chain. ✅ |
| `destroyDSPCoreNode` (Threading.cpp:17) | Sole DSPCore runtime destruction — called by `retire` chain and `destroyRolledBackDSP`. Verified caller-owned in `destroyRolledBackDSP` path. ✅ |
| `clearPublishedRuntimeSnapshotsNonRt` | Shutdown-only (`requestShutdownClearNonRt` gate). ✅ |
| `isShutdownInProgress` branch in `enqueueDeferredDeleteNonRtWithResult` | Shutdown-only. ✅ |

## Conclusion

All three `INVESTIGATE` items from 15-P-10 are resolved:

1. **`destroyRolledBackDSP`**: PASS — DSPCore was never retired (caller-owned from handle table).
   Direct destruction is correct (no terminal-authority engagement occurred).

2. **`CallerDestroy` + enqueue failure (4550)**: PASS — RuntimeState never entered OwnerChannel.
   `enqueue` failed → owner remained caller-owned (parameter RAII). Terminal authority not
   applicable.

3. **`CallerDestroy` + intentQueue failure (4570)**: PASS — RuntimeState entered OwnerChannel but
   was **recalled** via `take()` before entering D→Q→E→T. The caller explicitly re-acquired
   ownership (rollback) and discarded. The `AlignedObjectDeleter` fires on a recalled owner, not a
   published one.

4. **`FrozenRuntimeWorld` destructor**: PASS — `releaseState()` always called before any failure
   return (PublicationExecutor.cpp:34, guarded by null-check at line 24). Destructor is no-op.

5. **`commitRuntimePublication` residual**: PASS — `world` moved into callee; no double-destroy.

**No GAPs found.** All non-shutdown destruction paths destroy objects that were either
(pre-ownership) never transferred to authority, or (post-recall) explicitly rolled back from
the channel before entering the terminal retire chain (D→Q→E→T).

### Note on `AlignedObjectDeleter` usage

The `AlignedObjectDeleter` performs direct `~T()` + `aligned_free` for `RuntimeState` and
`RuntimePublishWorld` in ALL non-shutdown paths. This is structurally correct because:

- Objects **never entering** OwnerChannel (Path 2) are caller-owned throughout.
- Objects **recalled** from OwnerChannel (Path 3) are caller-owned after `take()`.
- Objects **never created** by `destroyRolledBackDSP` target (Path 1) are DSPCore handle-table
  entries, not RuntimeState.

The terminal authority (`enqueueDeferredDeleteNonRt → drainAllNonRt → terminalReclaim`) is
**only** engaged for successfully published worlds (via `retirePublishedRuntimeWorldNonRt` in the
PublishExecutor execution tail, and shutdown/recall paths). The deleter does not bypass it because
it only fires outside the authority's jurisdiction.

**However**, one clarity gap exists: `commitRuntimePublication` (AudioEngine.h:~4591) moves
`world` into `enqueueRuntimePublicationFireAndForget` and ignores `result.ownership`. It relies
entirely on the callee's parameter RAII to handle destruction. While correct (verified: callee
owns the parameter through return), this implicit ownership-transfer-via-destructor is subtle and
warrants a code comment for maintainability. This is a documentation finding, not a GAP.
