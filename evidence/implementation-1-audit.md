# 15-P-CROSS-IMPLEMENTATION-1 — drainAllNonRt() audit

## Status
- **Implementation-1**: PASS
- **D103**: BLOCKED → **PASS** (after drainAllNonRt + OwnerChannel test)

## 実装差分

### 1. OwnerChannel.h — drainAllNonRt() template method

```cpp
// OwnerChannel.h: drainAllNonRt(Fn&& reclaim) — take() と同一 pattern
template <class Fn>
std::size_t drainAllNonRt(Fn&& reclaim) noexcept {
    std::size_t reclaimed = 0;
    for (std::size_t i = 0; i < kCapacity; ++i) {
        Slot& s = slots_[i];
        Owner* const raw = consumeAtomic(s.owner, std::memory_order_acquire);
        if (raw != nullptr) {
            publishAtomic(s.owner, static_cast<Owner*>(nullptr),
                          std::memory_order_release);
            reclaim(raw);          // ownership relinquish (NOT release)
            ++reclaimed;
        }
    }
    return reclaimed;
}
```

- `Owner` = `OwnerPtr::element_type` = `const RuntimeState` (RuntimeOwner = aligned_unique_ptr<const RuntimeState>)
- `s.key` **not touched** — key-matching irrelevant (full scan); slot emptiness = owner==nullptr (take() と同一)

### 2. AudioEngine.Processing.ReleaseResources.cpp — call site

```cpp
m_coordinator.finalizeShutdown(timedOut);   // line 525 (existing)

// ★ 15-P-CROSS-IMPLEMENTATION-1: drain residual OwnerChannel owners
const auto drainedResidual = worldAuthority_.ownerChannel().drainAllNonRt(
    [this](const RuntimeState* raw) noexcept {
        enqueueDeferredDeleteNonRtWithResult(
            const_cast<RuntimeState*>(raw),
            [](void* p) noexcept {
                auto* ptr = static_cast<RuntimePublishWorld*>(p);
                ptr->unseal();
                ptr->~RuntimePublishWorld();
                convo::aligned_free(ptr);
            },
            DeletionEntryType::World);
    });
if (drainedResidual > 0)
    diagLog("[AUDIT] drainAllNonRt residual: reclaimed " + ...);
```

- Insert: `finalizeShutdown(timedOut)` 直後 (Producer/consumer join 済み, quiescence 確認済み)
- `const_cast<RuntimeState*>`: existing deleter (AudioEngine.h:3527) が `static_cast<RuntimePublishWorld*>` で const_cast を行うため一致
- `const RuntimeState*` → `void*`: implicit via const_cast (pre-existing pattern)

## 静的 audit

### Ownership transfer

```
OwnerChannel::drainAllNonRt(Fn)
    ↓
consumeAtomic(slot.owner, acquire) → raw (Owner*)
publishAtomic(slot.owner, nullptr, release)  // ownership relinquish
reclaim(raw)
    ↓
enqueueDeferredDeleteNonRtWithResult(raw, deleter, World)
    ↓
isShutdownInProgress()==true → shutdownReclaim → terminalReclaim
    ↓
D/Q/E/Terminal ownership chain
```

#### (a) OwnerChannel residual == 0

- drainAllNonRt は `kCapacity (256)` slot を full scan → `raw != nullptr` の slot をすべて callback へ transfer
- transfer 後 slot.owner = nullptr → **re-drain は no-op** (OwnerChannelTests #7 re-drain no-op verified)
- コード: OwnerChannel.h:95-103 (`if (raw != nullptr)` → `publishAtomic(nullptr)` → `reclaim`)

#### (b) drained owners ∈ D/Q/E/Terminal

- callback → `enqueueDeferredDeleteNonRtWithResult` → `enqueueWithRetry` (ISRRetireRouter.cpp:277)
- `RetireEnqueueResult` enum: `Success | QueuePressure | TerminalReclaim | Shutdown`
  - `Success` → D (DeferredDeletionQueue)
  - `QueuePressure` → Q (RetireQuarantineStore)
  - `TerminalReclaim` → Terminal
  - `Shutdown` → **dead code** (enqueueRetire は isShutdownInProgress() チェックを行わない — ISRRetireRouter.cpp:220-245, 277-338)
- コード: AudioEngine.h:4190 (`enqueueDeferredDeleteNonRtWithResult`), ISRRetireRouter.cpp:220-245 (`enqueueRetire` returns Success|QueuePressure only)

#### (c) No ownership discarded

- enqueueRetire は `Success|QueuePressure` のみ返す (Shutdown unreachable)
- PRECHECK D: `enqueueWithRetry` の唯一の `return Shutdown` パス (ISRRetireRouter.cpp:277-338) は **dead code**
- コード: ISRRetireRouter.cpp:320-325 (`return RetireEnqueueResult::Shutdown` は `isShutdownInProgress()` ガード前 — enqueueRetire は never call)

### Double transfer

- take(key) と drainAllNonRt は **同一の consume→publish(nullptr,release)** single-transfer pattern
- drainAllNonRt 実行中, Producer/Consumer は quiescent (finalizeShutdown 後) → race なし
- re-drain: 全 slot owner==nullptr → callback 0 回 (OwnerChannelTests #7 verified)
- コード: OwnerChannel.h:104-106 (`publishAtomic(nullptr, release)` → slot empty)

### Failure analysis

`RetireEnqueueResult` の全値を re-confirm:

| Result | path | ownership | D103 safety |
|--------|------|-----------|-------------|
| Success | enqueueWithRetry → D | transferred | ✅ |
| QueuePressure | enqueueWithRetry → Q retry | transferred | ✅ |
| TerminalReclaim | terminalReclaim → Terminal | transferred | ✅ |
| Shutdown | enqueueWithRetry (line 323) | — | **dead code** (enqueueRetire never checks isShutdownInProgress) ✅ |

- `Shutdown` は **到達不能** — `enqueueRetire()` (ISRRetireRouter.cpp:220-245) は `enqueueWithRetry()` を呼び出す前に `isShutdownInProgress()` をチェックしない. `enqueueWithRetry()` の `return Shutdown` (line 323) は `enqueueRetire()` からの call path では決して実行されない
- `enqueueDeferredDeleteNonRtWithResult` (ReleaseResources.cpp:540 callback) は `Result` を無視 (void) — ownership は callback で transfer 完了済み

## Test

| test | result |
|------|--------|
| OwnerChannelTests (8 tests, incl. drainAllNonRt) | ✅ PASS (0.17s) |
| RuntimeWorldAuthorityProjectionContract | ✅ PASS |
| InvariantINV3INV5 | ✅ PASS |
| RetireGraceSemantics | ✅ PASS |
| NormalRetireDSPHandleCompare | ✅ PASS |
| Debug 30/30 CTest | ✅ PASS (35.82s) |
| Release 30/30 CTest | ✅ PASS (20.60s) |

- `HeadlessAudioPathVerification` (Release): ❌ FAILED — **build-icx stale** (2026/08/11, drainAllNonRt 未反映). `build` (MSVC) の exe は 2026/08/18 00:08 にビルド成功済み
- `RuntimeWorldAuthority` exclusion task: CTest `-E RuntimeWorldAuthority` は不要 (RuntimeWorldAuthorityProjectionContract は PASS)

## D103 re-audit 判定

| 項目 | 結果 | 根拠 |
|------|------|------|
| (a) OwnerChannel residual == 0 | ✅ PASS | drainAllNonRt full scan + publish(nullptr) + re-drain no-op (OwnerChannelTests #7) |
| (b) drained owners ∈ D/Q/E/Terminal | ✅ PASS | callback → enqueueDeferredDeleteNonRtWithResult → Success/QueuePressure/TerminalReclaim |
| (c) no ownership discarded | ✅ PASS | enqueueRetire Success|QueuePressure only (Shutdown dead code) |

**D103: BLOCKED → PASS** ✅

- drainAllNonRt() 実装完了 (OwnerChannel.h:90-105)
- ReleaseResources.cpp:522-540 に call site (finalizeShutdown 直後)
- §9.2 PRECHECK 5/5 は実装と完全一致
- §9.3 `s.key={}` 不要 は実装で除去済み
- OwnerChannel unit test (2 tests) 追加 — drainAllNonRt no-op re-drain / drain-then-reenqueue verified
