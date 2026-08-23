# D101-9 Phase 9-B Step 5 — K_terminal Sizing / Telemetry Sufficiency Audit

## 1. Purpose

> **K_terminal の「いくつ」を決めるのではなく、「いつから K_terminal を決定可能にする測定契紃が整っているか」を監査する。**

Step 4 concluded Candidate B = GO, with the caveat that `K_terminal` sizing is **telemetry-driven and measurement-gated**. This Step 5 performs a **telemetry sufficiency audit** — not a K_terminal value selection.

The audit proceeds through: `Telemetry sufficiency → Measurement protocol → Worst-case workload → Peak observation → K_terminal derivation`.

**No code changes in this step.** Pure audit.

---

## 2. Step 5-A — Latest-Source Production Call Graph Re-Audit

### 2.1 TerminalReclaimAuthority::Entry (ISRRetireRouter.h:62-70)

```cpp
struct Entry {
    void* ptr = nullptr;                    // 8 bytes
    void (*deleter)(void*) = nullptr;       // 8 bytes
    uint64_t epoch = 0;                    // 8 bytes
    DeletionEntryType type = DeletionEntryType::Generic;  // 1 byte (uint8_t)
    const char* reason = nullptr;          // 8 bytes
};  // Total: 33 bytes data → 40 bytes with padding
```

**sizeof(Entry) verification:**

| Field | Type | Size | Offset |
|---|---|---|---|
| `ptr` | `void*` | 8 | 0 |
| `deleter` | `void(*)(void*)` | 8 | 8 |
| `epoch` | `uint64_t` | 8 | 16 |
| `type` | `DeletionEntryType` (uint8_t) | 1 | 24 |
| (padding) | — | 7 | 25-31 |
| `reason` | `const char*` | 8 | 32 |
| **Total** | | **40** | |

**`sizeof(Entry) = 40 bytes` on x64/Itanium ABI.** This matches the Step 4 assumption. `alignof(Entry) = 8` (driven by `void*` and `uint64_t`).

**`static_assert` recommendation (Step 5-E):** `static_assert(sizeof(TerminalReclaimAuthority::Entry) == 40, ...)` should be verified at compile time, not assumed.

### 2.2 store() (ISRRetireRouter.cpp:27-53)

```cpp
bool TerminalReclaimAuthority::store(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                     DeletionEntryType type, const char* reason) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return true;  // no-op は成功扱い

    std::lock_guard<std::mutex> lock(mtx_);
    entries_.push_back(Entry{ptr, deleter, epoch, type, reason});
    residentAtomic_.fetch_add(1, std::memory_order_release);  // ← increment point
    return true;  // ★ P-4: growable store — ALWAYS accepts
}
```

**Atomic operation convention violation:** `residentAtomic_.fetch_add()` is called directly, NOT through the `convo::fetchAddAtomic()` wrapper. The codebase convention (`AtomicAccess.h`) requires wrapper usage. This is a **latent convention violation** that must be fixed if telemetry is added to this function.

### 2.3 drain() (ISRRetireRouter.cpp:39-75)

```cpp
void TerminalReclaimAuthority::drain(uint64_t minReaderEpoch,
                                     const std::function<bool(uint64_t, uint64_t)>& isOlderFn) noexcept
{
    std::vector<Entry> pending;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        // ... compact entries_, collect epoch-safe ones into `pending` ...
        entries_.resize(w);
    }
    residentAtomic_.fetch_sub(static_cast<uint32_t>(pending.size()), std::memory_order_release);  // ← decrement
    for (auto& e : pending) {
        e.deleter(e.ptr);
        if (e.type == DeletionEntryType::World) {
            ++reclaimCount_;  // ← NON-ATOMIC increment of atomic! BUG
            if (referenceObserver_ != nullptr)
                referenceObserver_->onRelease();
        }
    }
}
```

**BUG FOUND:** `++reclaimCount_` at line 70 is a non-atomic increment of an `std::atomic<uint64_t>`. While `++` on `std::atomic` is technically atomic (it's `operator++` which maps to `fetch_add`), it does NOT use the `convo::fetchAddAtomic()` wrapper, violating codebase convention. More importantly, there is a **race condition**: `reclaimCount_` is modified under `mtx_` (in drain/drainAll) AND via `recordWorldReclaim()` (which has NO lock). This is a pre-existing concern but relevant to Step 5-7 (drain counting).

### 2.4 drainAll() (ISRRetireRouter.cpp:77-98)

```cpp
void TerminalReclaimAuthority::drainAll() noexcept
{
    std::vector<Entry> pending;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        pending.swap(entries_);
        residentAtomic_.store(0, std::memory_order_release);  // ← reset to 0
    }
    for (auto& e : pending) {
        if (e.ptr != nullptr && e.deleter != nullptr) {
            e.deleter(e.ptr);
            if (e.type == DeletionEntryType::World) {
                ++reclaimCount_;  // ← same non-atomic increment issue
                if (referenceObserver_ != nullptr)
                    referenceObserver_->onRelease();
            }
        }
    }
}
```

### 2.5 tryReclaim() (ISRRetireRouter.cpp:99-119)

Calls `drain()` internally. No direct `residentAtomic_` access.

### 2.6 residentCount() / residentCountAtomic() (ISRRetireRouter.h:90-94, 117-121)

```cpp
[[nodiscard]] std::size_t residentCount() const noexcept;  // mutex-protected (line 121-124 .cpp)
[[nodiscard]] uint32_t residentCountAtomic() const noexcept  // lock-free
{
    return convo::consumeAtomic(residentAtomic_, std::memory_order_acquire);
}
```

✅ **Existing observation point for current resident count.** Uses `convo::consumeAtomic` wrapper (convention-compliant).

### 2.7 reclaimCount() (ISRRetireRouter.h:96-98)

```cpp
[[nodiscard]] std::uint64_t reclaimCount() const noexcept {
    return convo::consumeAtomic(reclaimCount_, std::memory_order_acquire);
}
```

✅ **Existing observation point for cumulative Terminal reclaim count.** Uses `convo::consumeAtomic` wrapper.

### 2.8 recordWorldReclaim() (ISRRetireRouter.h:101-103)

```cpp
void recordWorldReclaim() noexcept {
    ++reclaimCount_;  // ← NON-ATOMIC increment of atomic
    if (referenceObserver_ != nullptr)
        referenceObserver_->onRelease();
}
```

**Called from:** `terminalReclaim()` (ISRRetireRouter.cpp:505) — synchronous destruction path (epoch-safe + Non-RT).

⚠️ **BUG:** `++reclaimCount_` without `convo::fetchAddAtomic` wrapper. Also no lock, while `drain()`/`drainAll()` modify `reclaimCount_` under `mtx_`. Race condition exists but is low-impact since both paths only increment.

### 2.9 entries_ / residentAtomic_ / reclaimCount_

| Member | Type | Declared at | Increment/Decrement locations |
|---|---|---|---|
| `entries_` | `std::vector<Entry>` (growable) | ISRRetireRouter.h:115 | `push_back` (store:34), `clear/resize` (drain:62, drainAll:84) |
| `residentAtomic_` | `std::atomic<uint32_t>` | ISRRetireRouter.h:120 | `fetch_add` (store:35), `fetch_sub` (drain:65), `store(0)` (drainAll:84) |
| `reclaimCount_` | `std::atomic<uint64_t>` | ISRRetireRouter.h:117 | `++` (drain:70, drainAll:91), `++` (recordWorldReclaim:103) |

### 2.10 Router-level production paths to `store()`

**Path A — Normal operation escalation (enqueueWithRetry, Stage 5):**

```
enqueueWithRetry()
  → Stage 1: enqueueRetire() → D (DeferredDeletionQueue, capacity 4096)
  → Stage 2: retry loop (kMaxRetry=2): tryReclaim() + drainEmergencyAndTerminal()
  → Stage 3: m_retireQuarantine.quarantine() → Q (capacity 512)
  → Stage 4: m_emergencyQuarantine.quarantine() → E (capacity 512)
  → Stage 5: terminalReclaim() → TerminalReclaimAuthority::store()
             → residentAtomic_++ (always, since always stores)
```

**Callers of `enqueueWithRetry()`** (the only production path reaching Terminal via escalation):

1. `ISRRetireRouter::retire()` (ISRRetireRouter.cpp:275) — bool API, ignores result
2. `enqueueDeferredDeleteNonRtWithResult()` (AudioEngine.h:4230) — checks Success only
3. `RuntimeIntentCoordinator::enqueueRetire()` (ISRRuntimePublicationCoordinator.cpp:164) — delegates to router

**Path B — Shutdown path (shutdownReclaim):**

```
enqueueDeferredDeleteNonRtWithResult()
  → isShutdownInProgress() == true
  → shutdownReclaim() → terminalReclaim()
    → epoch safe → synchronous deleter + recordWorldReclaim()
    → epoch unsafe → m_terminalReclaim.store() → residentAtomic_++
```

**Callers of `shutdownReclaim()` (the only production path via shutdown):**

1. `enqueueDeferredDeleteNonRtWithResult()` (AudioEngine.h:4221)

### 2.11 Terminal resident increment/decrement/reset summary

| Operation | Location | residentAtomic_ | reclaimCount_ |
|---|---|---|---|
| store() (entry) | ISRRetireRouter.cpp:35 | `fetch_add(1)` | — |
| drain() (epoch-safe reclaim) | ISRRetireRouter.cpp:65 | `fetch_sub(n)` | `++` (non-atomic) |
| drainAll() (shutdown, all) | ISRRetireRouter.cpp:84 | `store(0)` | `++` (non-atomic) |
| recordWorldReclaim() | ISRRetireRouter.h:103 | — | `++` (non-atomic) |
| synchronous destroy in terminalReclaim() | ISRRetireRouter.cpp:505 | — | `recordWorldReclaim()` |

---

## 3. Terminal Occupancy — "Measurement Possible Points"

### 3.1 Current state audit

| Metric | Current API | Location | Convention-compliant? | Notes |
|---|---|---|---|---|
| Current Terminal resident | `residentCountAtomic()` | ISRRetireRouter.h:92 | ✅ (uses `consumeAtomic`) | Lock-free, used in `isFullyDrained()` |
| Terminal reclaim cumulative | `reclaimCount()` | ISRRetireRouter.h:96 | ✅ (uses `consumeAtomic`) | Counts World reclams via Terminal |
| **Terminal peak resident** | **NONE** | — | — | **MISSING** — primary gap |
| **Terminal store cumulative** | **NONE** | — | — | **MISSING** — no total store count |
| **Terminal drainAll count** | **NONE** | — | — | **MISSING** — no drainAll invocation count |
| **Terminal drain latency** | **NONE** | — | — | **MISSING** — no timing |
| Terminal occupancy sampling | `residentCountAtomic()` | ISRRetireRouter.h:92 | ✅ | Already available for sampling |
| **Peak observation persistence** | **NONE** | — | — | **MISSING** — no persistent max tracking |

### 3.2 Key finding: `residentCountAtomic()` aggregates ALL three stores

**IMPORTANT FINDING:** `ISRRetireRouter::residentCountAtomic()` (ISRRetireRouter.h:302-306) is:

```cpp
uint32_t residentCountAtomic() const noexcept {
    return m_retireQuarantine.residentCountAtomic()
         + m_emergencyQuarantine.residentCountAtomic()
         + m_terminalReclaim.residentCountAtomic();
}
```

This returns Q + E + T aggregate, NOT Terminal-only. For K_terminal sizing, we need the **Terminal-only** `residentAtomic_` accessed via `m_terminalReclaim.residentCountAtomic()`. The `TerminalReclaimAuthority::residentCountAtomic()` member (ISRRetireRouter.h:92-94) provides this.

### 3.3 Step 5-2 verdict

| Metric | Current | Verdict |
|---|---|---|
| Current Terminal resident | `residentCountAtomic()` (TerminalReclaimAuthority) | ✅ PASS |
| Terminal reclaim cumulative | `reclaimCount()` | ✅ PASS |
| Terminal peak | **NONE** | ❌ **MISSING — PRIORITY 1** |
| Terminal store cumulative | **NONE** | ❌ **MISSING — PRIORITY 2** |
| Terminal drainAll count | **NONE** | ❌ **MISSING — PRIORITY 3** |
| Terminal drain latency | **NONE** | ❌ MISSING (P1-low) |
| Terminal occupancy sampling | API exists | ✅ PASS |
| Peak persistent observation | **NONE** | ❌ **MISSING — PRIORITY 1** |

---

## 4. Step 5-3 — `terminalPeakResident` Measurement Location

### 4.1 Recommended measurement point: `store()` after `residentAtomic_++`

```cpp
// Proposed instrumentation in store():
std::lock_guard<std::mutex> lock(mtx_);
entries_.push_back(Entry{ptr, deleter, epoch, type, reason});
residentAtomic_.fetch_add(1, std::memory_order_release);
// ← NEW: peak update here
const uint32_t current = residentAtomic_.load(std::memory_order_acquire);
// atomic max-update pattern
```

### 4.2 Authoritative measurement source: `residentAtomic_` vs `entries_.size()`

The instruction asks to decide between `entries_.size()` and `residentAtomic_` as the peak authoritative source.

**Decision: `residentAtomic_` is the authoritative source.**

**Rationale:**
1. `residentCountAtomic()` is already used as the E-1.9-A lock-free occupancy predicate (ISRRetireRouter.h:92, AudioEngine.Threading.cpp:310-314)
2. `residentCountAtomic()` is read lock-free (no mutex) — can be sampled from any Non-RT context without blocking the store/drain path
3. `entries_.size()` requires `mtx_` lock — cannot be sampled without contention
4. `residentAtomic_` is incremented at the same point as `entries_.push_back()` in `store()` — the two are in sync by construction (both under `mtx_` for increment, `push_back` happens before `fetch_add`)
5. `residentAtomic_` is decremented/reset in `drain()`/`drainAll()` at the same logical points as `entries_` compaction/clear

**However:** `entries_.size()` is the "source of truth" for the actual vector content. The `residentAtomic_` counter is derived from it. For peak tracking, using `residentAtomic_` is sufficient because:
- It is incremented at the same code point as the actual entry insertion
- It is decremented at the same code point as entry removal
- The only divergence would be a bug where `residentAtomic_` and `entries_.size()` get out of sync — but that would itself be a correctness bug

### 4.3 Atomic wrapper convention requirement

The measurement MUST use `convo::fetchAddAtomic()` / `convo::consumeAtomic()` wrappers, NOT raw `std::atomic` operations. Current violations:

| Location | Current | Required fix |
|---|---|---|
| `store()` line 35 | `residentAtomic_.fetch_add(...)` | `convo::fetchAddAtomic(residentAtomic_, uint32_t{1}, std::memory_order_release)` |
| `drain()` line 65 | `residentAtomic_.fetch_sub(...)` | `convo::fetchSubAtomic(residentAtomic_, ...)` |
| `drainAll()` line 84 | `residentAtomic_.store(0, ...)` | `convo::publishAtomic(residentAtomic_, uint32_t{0}, std::memory_order_release)` |
| `recordWorldReclaim()` line 103 | `++reclaimCount_` | `convo::fetchAddAtomic(reclaimCount_, uint64_t{1}, std::memory_order_acq_rel)` |
| `drain()` line 70 | `++reclaimCount_` | `convo::fetchAddAtomic(...)` |
| `drainAll()` line 91 | `++reclaimCount_` | `convo::fetchAddAtomic(...)` |

### 4.4 Peak tracking mechanism

The peak update uses a **CAS loop with `fetch_max`** pattern:

```cpp
// In store(), after fetch_add:
const uint32_t current = convo::consumeAtomic(residentAtomic_, std::memory_order_acquire);
convo::publishAtomic(terminalPeakResident_,
    std::max(convo::consumeAtomic(terminalPeakResident_, std::memory_order_acquire), current),
    std::memory_order_release);
// OR more efficiently, atomic fetch_max:
uint32_t expected = convo::consumeAtomic(terminalPeakResident_, std::memory_order_acquire);
while (current > expected) {
    if (convo::compareExchangeAtomic(terminalPeakResident_, expected, current,
            std::memory_order_acq_rel, std::memory_order_acquire))
        break;
}
```

The `terminalPeakResident_` member would be `std::atomic<uint32_t>` added to `TerminalReclaimAuthority`.

---

## 5. Step 5-4 — K_terminal = 2 × peak is NOT yet adopted (2× coefficient audit)

### 5.1 What does `terminalPeakResident` represent?

The peak measurement must distinguish between:

| Scenario | Should peak include? | Rationale |
|---|---|---|
| Normal publish/rebuild load | YES (baseline) | Establishes minimum operating range |
| Transient burst (burst publish) | YES | Burst absorption requirement |
| Short reader stall (10-100ms) | YES | Q/E absorb, Terminal may spike |
| Long reader stall (1-10s) | YES | Terminal escalation; critical for sizing |
| Repeated publish under stall | YES | Worst-case accumulation |
| Shutdown transition | NO (drain, not growth) | Shutdown reduces resident to 0 |

**Step 5-4 finding:** Peak should capture ALL non-shutdown scenarios. Shutdown drain reduces Terminal to 0, so it does NOT inflate peak. The peak during shutdown drain is always 0 (drainAll resets), so including or excluding shutdown from peak observation does not matter — the peak is naturally a normal-operation concept.

### 5.2 Peak observation window

**Recommended: session peak (running max over process lifetime).**

| Window | Pros | Cons |
|---|---|---|
| Single run peak | Simple, deterministic | May miss worst-case across sessions |
| Session peak | Captures all scenarios in one run | Requires persistent storage across runs |
| N-run maximum | Statistical robustness | Complex, requires cross-run persistence |
| P99/P99.9 | Statistical rigor | Overkill for sizing; needs sampling infrastructure |
| Worst-case deterministic | Reproducible sizing | Requires artificial stress test |

**Recommendation: Start with session peak (single run max), with P99/peak across N runs as future enhancement.** The session peak is the minimum viable — it captures the actual operating envelope. If peak varies across runs, that indicates non-deterministic behavior that needs investigation anyway.

### 5.3 Burst absorption rationale for 2× coefficient

The 2× safety margin must be justified by a **rate-based derivation**, not an arbitrary heuristic:

```
K_terminal = (retire_arrival_rate × max_reader_stall_duration) × safety_factor
```

Where:
- `retire_arrival_rate`: entries/sec reaching Terminal (after Q/E absorption)
- `max_reader_stall_duration`: worst-case time readers hold epochs (from stuck reader detection: `kResidencyStuckUs=1s`, `kChronicResidencyUs=30s`)
- `safety_factor`: ≥ 2× (bursts can be 2× the steady-state rate)

**From EpochDomain.h (Step 1 evidence):**
- `kResidencyStuckUs = 1'000'000` (1 second — stuck reader threshold)
- `kChronicResidencyUs = 30'000'000` (30 seconds — chronic stuck)
- `kWarningResidencyUs = 10'000'000` (10 seconds — warning threshold)

The 2× coefficient is justified if:
1. Peak arrival rate during burst ≥ 2× steady-state arrival rate (empirically verifiable)
2. `max_reader_stall_duration` ≈ `kResidencyStuckUs` (1s) — the point where escalation to Terminal begins
3. Terminal drain rate ≥ arrival rate during normal (non-stuck) operation

**Step 5-4 verdict:** The 2× coefficient is a **starting hypothesis**, not a derived value. It must be validated against:
- Actual `retire_arrival_rate` at Terminal (requires telemetry)
- Actual `peak / steady-state` ratio (requires sampling)
- Actual `max_reader_stall_duration` distribution (requires health monitor data)

---

## 6. Step 5-5 — Terminal Growth Dominant Factors

### 6.1 The causal chain

```text
Reader stall (epoch not advancing)
    ↓
minReaderEpoch stagnation (stuck reader holds old epoch)
    ↓
Entries with epoch > minReaderEpoch cannot be reclaimed by drain()
    ↓
D fills (kQueueSize=4096) → QueuePressure
    ↓
Q fills (kMaxQuarantinedEntries=512) → quarantine() returns false
    ↓
E fills (kMaxQuarantinedEntries=512) → emergencyQuarantine() returns false
    ↓
Terminal escalation → store() → residentAtomic_++
    ↓
Terminal resident growth = retire arrival rate - epoch-safe reclaim rate
```

### 6.2 Measurement requirement: correlated time-series

To prove the causal chain, the following MUST be measured **at the same time** (same tick, same sample):

| Metric | Source API | Frequency |
|---|---|---|
| `activeReaderCount` | `m_retireRouter->activeReaderCount()` | Timer tick (100ms) |
| `minReaderEpoch` | `m_retireRouter->minReaderEpoch()` | Timer tick (100ms) |
| Terminal resident | `m_terminalReclaim.residentCountAtomic()` | Timer tick (100ms) |
| D resident | `m_retireRouter->pendingRetireCount()` | Timer tick (100ms) |
| Q resident | `m_retireQuarantine.residentCountAtomic()` | Timer tick (100ms) |
| E resident | `m_emergencyQuarantine.residentCountAtomic()` | Timer tick (100ms) |
| retire enqueue rate | delta of store count | Timer tick (100ms) |
| reclaim rate | delta of reclaimCount | Timer tick (100ms) |

### 6.3 Current gap

The existing `RuntimeBackpressureTelemetry` (`AudioEngine.h:1561-1585`) includes:
- `retireQueueDepth` — D only
- `quarantineResident` — Q + E aggregate (from `quarantineResidentCount()` + `getQuarantineResidentCount()`)
- `retireEscalationCount` — count of escalations (but NOT Terminal-specific)

**Missing:** Terminal resident, Terminal store count, Terminal reclaim count, minReaderEpoch, activeReaderCount in the telemetry struct.

The `RuntimeHealthMonitor::takeSnapshot()` (`RuntimeHealthMonitor.cpp:615-648`) includes:
- `pendingRetire` = `m_retireRouter->pendingRetireCount()` (D only)
- `activeReaderCount` = `m_retireRouter->activeReaderCount()`
- `readerStuckCount` from `detectStuckReaders(10)`
- `maxRetireAgeUs` (if refs set)

**Missing:** Terminal resident, minReaderEpoch, Q/E/T individual residents, store rate.

**Step 5-5 verdict: BLOCKED for K_terminal derivation.** The causal chain (reader stall → epoch stagnation → Terminal growth) cannot be measured today because Terminal resident and minReaderEpoch are not in any telemetry snapshot.

---

## 7. Step 5-6 — Worst-Case Workload Definition

### T1 — Normal operation

```text
Standard publish/rebuild cycle, no reader stalls.
Expected: Terminal resident ≈ 0 (entries reclaimed via drain() before accumulation).
```

**Measurement needed:** baseline Terminal occupancy under steady state.

### T2 — Short reader stall

```text
Reader stalls for 10-100ms (below kResidencyStuckUs=1s).
Q/E absorb burst, Terminal may briefly spike.
Recovery: reader advances → epoch safe → drain() reclaims.
```

**Measurement needed:** Terminal peak during short-stall recovery.

### T3 — Long reader stall

```text
Reader stalls for 1-30s (at/above kResidencyStuckUs → kChronicResidencyUs).
Q (512) + E (512) + D (4096) = 5120 entries fill.
Terminal escalation begins.
Expected: Terminal resident grows proportionally to (arrival rate × stall duration).
```

**Measurement needed:** Terminal peak during long-stall escalation. This is the **primary K_terminal sizing scenario**.

### T4 — Repeated publish under stall

```text
Reader stall + continuous publish/rebuild at maximum rate.
Each rebuild cycle may generate multiple retire entries.
Terminal resident can grow beyond single-publish stall.
```

**Measurement needed:** Terminal peak under sustained publish + stall. This is the **worst-case accumulation scenario**.

### T5 — Shutdown while Terminal resident > 0

```text
Normal operation → Terminal has entries → shutdown begins.
closeAdmission() + joinProducers() → no new entries.
Graceful drain: tryReclaim() + drainEmergencyAndTerminal() → epoch-safe entries reclaimed.
PR2: activeReaderCount()==0 → drainAllQuarantineStore() → Terminal drainAll().
Expected: Terminal resident → 0, deleter exactly once per entry.
```

**Measurement needed:** Verify `terminalResident == 0` after `drainAllQuarantineStore()`, and verify `reclaimCount` increments match entries stored.

### T6 — Stuck-reader shutdown

```text
Reader stuck → graceful drain timeout (5s) → EmergencyDrain → destructor fallback.
activeReaderCount > 0 → drainAll() NOT called in ReleaseResources PR2.
Destructor (AudioEngine.CtorDtor.cpp:251-257): activeReaderCount > 0 → drainAllQuarantineStore().
```

**Measurement needed:** Verify Terminal drainAll executes in destructor fallback. This is the **last-resort safety net**.

### T6 note: `sizeof(Entry)` real value

Verified in 2.1 above: `sizeof(TerminalReclaimAuthority::Entry) = 40 bytes` on x64.

Memory model:
- 1000 Terminal entries = 40 KB (fits L1/L2)
- 10000 Terminal entries = 400 KB (fits L3)
- 50000 Terminal entries = 2 MB (exceeds typical L3, but still manageable for Non-RT)
- 100000 Terminal entries = 4 MB (approaching practical limits for Non-RT heap)

**Not a correctness constraint** — Candidate B allows unbounded Terminal growth during normal operation. Only the **operational acceptability** threshold matters (health monitor escalation).

---

## 8. Step 5-7 — Minimum Telemetry Set

### 8.1 P0 (Required for K_terminal sizing)

| Metric | Current API | New API needed | Measurement location |
|---|---|---|---|
| `terminalResident` | `residentCountAtomic()` ✅ | Read through existing API | `store()` increment / `drain()` decrement |
| `terminalPeakResident` | **NONE** ❌ | `terminalPeakResident_` atomic member | `store()` after `residentAtomic_++` |
| `terminalStoreCount` | **NONE** ❌ | `terminalStoreCount_` atomic counter | `store()` (increment on every entry stored) |
| `terminalReclaimCount` | `reclaimCount()` ✅ | Existing (fix non-atomic `++` bug) | `drain()` / `drainAll()` / `recordWorldReclaim()` |
| `activeReaderCount` | `activeReaderCount()` ✅ | Existing | `ISRRetireRouter` / `EpochDomain` |
| `minReaderEpoch` | `minReaderEpoch()` ✅ | Existing | `ISRRetireRouter` → `EpochDomain` |
| `pendingRetireCount` | `pendingRetireCount()` ✅ | Existing | `ISRRetireRouter` → `EpochDomain` |
| `quarantineResident` | `quarantineResidentCount()` ✅ | Existing | Q + E aggregate |
| `emergencyQuarantineResident` | `emergencyQuarantineResidentCount()` ✅ | Existing | E only |
| `worldReclaimCount` | `worldReclaimCount()` ✅ | Existing (aggregates D+Q+E+T) | Router-level aggregate |

### 8.2 P1 (Required for operational monitoring)

| Metric | New API needed | Measurement location |
|---|---|---|
| `terminalDrainAllCount` | `terminalDrainAllCount_` atomic counter | `drainAll()` |
| `terminalDrainLatency` | timestamp diff in `drainAll()` | `drainAll()` start/end |

### 8.3 P1-low (Deferred)

| Metric | New API needed | Notes |
|---|---|---|
| `terminalDrainSuccessCount` / `terminalDrainEntryCount` | Per-drain entry count | Useful for verification |

### 8.4 Summary: What needs to be ADDED for K_terminal sizing

**Minimum new members in `TerminalReclaimAuthority`:**

```cpp
// New telemetry members (to be added):
std::atomic<uint32_t> terminalPeakResident_{0};    // B5-2: peak observation
std::atomic<uint64_t> terminalStoreCount_{0};     // B5-3: cumulative store count
std::atomic<uint64_t> terminalDrainAllCount_{0};  // B5-6: drainAll invocation count
```

**New API methods:**

```cpp
[[nodiscard]] uint32_t terminalPeakResident() const noexcept {
    return convo::consumeAtomic(terminalPeakResident_, std::memory_order_acquire);
}
[[nodiscard]] std::uint64_t terminalStoreCount() const noexcept {
    return convo::consumeAtomic(terminalStoreCount_, std::memory_order_acquire);
}
[[nodiscard]] std::uint64_t terminalDrainAllCount() const noexcept {
    return convo::consumeAtomic(terminalDrainAllCount_, std::memory_order_acquire);
}
```

**Instrumentation points in existing methods:**

1. `store()`: After `residentAtomic_.fetch_add(1)`:
   - `convo::fetchAddAtomic(terminalStoreCount_, uint64_t{1}, std::memory_order_release);`
   - Peak update: `uint32_t current = residentAtomic_.load(...); while (current > peak) CAS(peak, current)`

2. `drainAll()`: At entry:
   - `convo::fetchAddAtomic(terminalDrainAllCount_, uint64_t{1}, std::memory_order_acq_rel);`

3. Fix existing bugs:
   - `residentAtomic_.fetch_add(...)` → `convo::fetchAddAtomic(residentAtomic_, ...)`
   - `residentAtomic_.fetch_sub(...)` → `convo::fetchSubAtomic(residentAtomic_, ...)`
   - `++reclaimCount_` → `convo::fetchAddAtomic(reclaimCount_, ...)`

---

## 9. Step 5-8 — sizeof(Entry) Real Value Verification

### 9.1 Entry struct (ISRRetireRouter.h:64-70)

```cpp
struct Entry {
    void* ptr = nullptr;                    // 8 bytes
    void (*deleter)(void*) = nullptr;       // 8 bytes
    uint64_t epoch = 0;                    // 8 bytes
    DeletionEntryType type = DeletionEntryType::Generic;  // 1 byte (uint8_t)
    const char* reason = nullptr;          // 8 bytes
};
```

### 9.2 DeletionEntryType (DeferredDeletionQueue.h:21-22)

```cpp
enum class DeletionEntryType : uint8_t {
    Generic = 0,
    World   = 1
};
```

### 9.3 Layout calculation (x64/Itanium ABI)

| Field | Type | Size | Offset | Alignment |
|---|---|---|---|---|
| `ptr` | `void*` | 8 | 0 | 8 |
| `deleter` | function ptr | 8 | 8 | 8 |
| `epoch` | `uint64_t` | 8 | 16 | 8 |
| `type` | `uint8_t` | 1 | 24 | 1 |
| *padding* | — | 7 | 25–31 | — |
| `reason` | `const char*` | 8 | 32 | 8 |
| **Total size** | | **40** | | **alignof = 8** |

### 9.4 Compile-time verification needed

```cpp
static_assert(sizeof(TerminalReclaimAuthority::Entry) == 40,
    "Entry size assumptions (40 bytes for K_terminal cache-fit analysis) violated");
static_assert(alignof(TerminalReclaimAuthority::Entry) == 8,
    "Entry alignment assumptions violated");
```

**Step 5-8 verdict: CONFIRMED.** `sizeof(Entry) = 40 bytes` matches Step 4. But this must be verified with `static_assert` at compile time, not assumed from source inspection.

---

## 10. Step 5-9 — "L2 cache fit" Degraded to Operational Constraint

### 10.1 Candidate B: Terminal is GROWABLE during normal operation

This means:

```text
K_terminal × sizeof(Entry) <= L2
```

is **NOT** a correctness invariant. It is only an **operational performance heuristic**:
- If Terminal grows beyond L2, cache misses increase for `drain()` scanning
- But `drain()` is epoch-gated — it only runs when entries become safe, which is bounded by reader stall duration
- The `residentAtomic_` lock-free read (used by `isFullyDrained()`) does NOT require cache-line residency of `entries_`

### 10.2 Correct constraints for Candidate B

**Correctness constraint (hard):**
```text
shutdown completion ⇒ terminalResident == 0
```
This is already satisfied by `drainAllQuarantineStore()` (Step 4 proof).

**Operational constraint (soft):**
```text
terminalPeakResident × sizeof(Entry) should remain practically manageable
```
This is a health monitoring concern, not a correctness concern. If Terminal grows too large:
- HealthMonitor detects via `terminalPeakResident` telemetry
- PolicyEngine escalates (throttle, emergency drain, safe mode)
- User sees warning in diagnostics

### 10.3 Memory growth as operational risk

Under Candidate B, the risk is:

```text
Stuck reader (long stall)
+
High retire rate (burst publish)
+
Sustained operation (no recovery)
↓
std::vector push_back() → unbounded heap growth → OOM
```

This risk is **mitigated by** (but not eliminated by):
1. Reader stuck detection (`detectStuckReaders`, `kResidencyStuckUs=1s`)
2. Grace period (`kRetireGracePeriod`, maxRetireWallClockMs_=5000ms)
3. Emergency drain (EmergencyDrain phase)
4. Shutdown drain (drainAllQuarantineStore)

The risk is **NOT eliminated** because:
1. These are all reactive, not preventive
2. A reader stuck for 5s with high publish rate can still cause significant growth before detection triggers

**Step 5-9 verdict: CONFIRMED.** "L2 cache fit" must be degraded from a sizing criterion to an operational heuristic. The correctness criterion is `terminalResident == 0 at shutdown`.

---

## 11. B5 Gate Table

| Gate | Criterion | Status | Evidence |
|---|---|---|---|
| **B5-1** | Terminal current occupancy lock-free observable | ✅ PASS | `residentCountAtomic()` (ISRRetireRouter.h:92-94) |
| **B5-2** | Terminal peak occupancy observable | ❌ BLOCKED | No `terminalPeakResident` member. Must add to `TerminalReclaimAuthority`. |
| **B5-3** | Terminal arrival/reclaim rate measurable | ⚠️ PARTIAL | `reclaimCount()` exists. `terminalStoreCount` does NOT. Store rate requires new counter. |
| **B5-4** | activeReaderCount / minReaderEpoch correleasable with Terminal growth | ✅ PASS | `activeReaderCount()` (ISRRetireRouter.h:152) and `minReaderEpoch()` (ISRRetireRouter.h:272) both exist. Integration into snapshot is missing. |
| **B5-5** | Worst-case reader-stall workload defined | ✅ PASS | T1-T6 defined in Section 7. T3 (long stall) and T4 (repeated publish under stall) are primary sizing scenarios. |
| **B5-6** | Shutdown resident → 0 measurable | ✅ PASS | `isFullyDrained()` checks `terminalReclaimResident == 0` (AudioEngine.Threading.cpp:172). `drainAll()` resets `residentAtomic_` to 0. |
| **B5-7** | deleter exactly-once measurable | ⚠️ PARTIAL | `reclaimCount()` tracks World reclaims. But `terminalStoreCount - terminalReclaimResident - terminalDrainAllCount` can verify. No explicit "deleter call count" counter exists. |
| **B5-8** | `sizeof(Entry)` real value confirmed | ✅ PASS | Verified = 40 bytes (Section 9). Needs `static_assert`. |
| **B5-9** | K_terminal derivation model connected to measurement data | ❌ BLOCKED | Formula `K_terminal = 2 × peak` is a hypothesis. Needs: (a) terminalPeakResident telemetry, (b) arrival rate data, (c) stall duration distribution from health monitor. |
| **B5-10** | Telemetry schema design specified | ❌ BLOCKED | Section 8 specifies required new members and instrumentation points. Not yet implemented. |

---

## 12. Step 5-10 — Telemetry Implementation Plan (Design Only, Not Implemented)

### 5-A. New members to add to `TerminalReclaimAuthority` (ISRRetireRouter.h)

```cpp
// ★ Step 5: Telemetry for K_terminal sizing
std::atomic<uint32_t> terminalPeakResident_{0};     // B5-2: peak Terminal occupancy
std::atomic<uint64_t> terminalStoreCount_{0};      // B5-3: cumulative store() calls (entries stored)
std::atomic<uint64_t> terminalDrainAllCount_{0};   // B5-6: cumulative drainAll() calls
std::atomic<uint64_t> terminalDrainEntryCount_{0}; // B5-7: cumulative entries drained (drainAll)
```

### 5-B. New API methods (ISRRetireRouter.h)

```cpp
[[nodiscard]] uint32_t terminalPeakResident() const noexcept {
    return convo::consumeAtomic(terminalPeakResident_, std::memory_order_acquire);
}
[[nodiscard]] std::uint64_t terminalStoreCount() const noexcept {
    return convo::consumeAtomic(terminalStoreCount_, std::memory_order_acquire);
}
[[nodiscard]] std::uint64_t terminalDrainAllCount() const noexcept {
    return convo::consumeAtomic(terminalDrainAllCount_, std::memory_order_acquire);
}
```

### 5-C. Router-level delegation (ISRRetireRouter.h / ISRRetireRouter.cpp)

```cpp
// ISRRetireRouter.h — add methods:
[[nodiscard]] uint32_t terminalPeakResident() const noexcept;
[[nodiscard]] std::uint64_t terminalStoreCount() const noexcept;
[[nodiscard]] std::uint64_t terminalDrainAllCount() const noexcept;

// ISRRetireRouter.cpp — delegate to m_terminalReclaim:
uint32_t ISRRetireRouter::terminalPeakResident() const noexcept {
    return m_terminalReclaim.terminalPeakResident();
}
std::uint64_t ISRRetireRouter::terminalStoreCount() const noexcept {
    return m_terminalReclaim.terminalStoreCount();
}
std::uint64_t ISRRetireRouter::terminalDrainAllCount() const noexcept {
    return m_terminalReclaim.terminalDrainAllCount();
}
```

### 5-D. Instrumentation points

1. **`store()`** (ISRRetireRouter.cpp:27-53): After `residentAtomic_.fetch_add(1)`:
   - `convo::fetchAddAtomic(terminalStoreCount_, uint64_t{1}, std::memory_order_release);`
   - Peak update via CAS loop on `terminalPeakResident_`

2. **`drainAll()`** (ISRRetireRouter.cpp:77-98): At entry:
   - `convo::fetchAddAtomic(terminalDrainAllCount_, uint64_t{1}, std::memory_order_acq_rel);`
   - Inside loop: `convo::fetchAddAtomic(terminalDrainEntryCount_, uint64_t{1}, std::memory_order_release);` per entry

3. **Fix convention violations:**
   - All `residentAtomic_.fetch_add/fetch_sub/store` → `convo::fetchAddAtomic/fetchSubAtomic/publishAtomic`
   - All `++reclaimCount_` → `convo::fetchAddAtomic(reclaimCount_, ...)`

### 5-E. Telemetry integration into `RuntimeBackpressureTelemetry`

Add to `AudioEngine.h:1561-1585`:

```cpp
std::uint64_t terminalStoreCount = 0;       // from m_retireRouter->terminalStoreCount()
std::uint32_t terminalPeakResident = 0;     // from m_retireRouter->terminalPeakResident()
std::uint64_t terminalDrainAllCount = 0;    // from m_retireRouter->terminalDrainAllCount()
```

### 5-F. HealthMonitor integration

Add `checkTerminalPressure()` to `RuntimeHealthMonitor`:
- If `terminalResident > threshold` (e.g., 2× steady-state peak): Warning
- If `terminalResident > critical` (e.g., 4× peak or absolute 4096): Error
- Event: `EVENT_TERMINAL_OVERFLOW (1014)`

---

## 13. Step 5-11 — Step 5 GO/NO-GO

| Gate | Status |
|---|---|
| B5-1 | ✅ PASS |
| B5-2 | ❌ BLOCKED |
| B5-3 | ⚠️ PARTIAL |
| B5-4 | ✅ PASS |
| B5-5 | ✅ PASS |
| B5-6 | ✅ PASS |
| B5-7 | ⚠️ PARTIAL |
| B5-8 | ✅ PASS |
| B5-9 | ❌ BLOCKED |
| B5-10 | ❌ BLOCKED |

### Verdict: Candidate B = GO, but K_terminal value determination = BLOCKED

**Candidate B is NOT NO-GO.** All correctness invariants hold (B5-1, B5-4, B5-5, B5-6, B5-8 PASS). The blocking items are:

1. **B5-2 (terminalPeakResident):** Requires 4 new atomic members in `TerminalReclaimAuthority`
2. **B5-9 (K_terminal derivation):** Cannot derive K_terminal without peak telemetry

**This is expected and acceptable.** The Step 5 purpose is to audit measurement sufficiency, NOT to select K_terminal. The audit reveals:

> **Candidate B = GO** (correctness preserved)
> **K_terminal value = BLOCKED** (requires Step 5-A through 5-F instrumentation to be implemented first)

This aligns with the Step 4 guidance: "Candidate B = GO, K_terminal sizing = measurement pending."

---

## 14. No-code-change confirmation

**Zero code changes in this step.** This is a telemetry sufficiency audit only. The following files were READ but NOT modified:

- `src/audioengine/ISRRetireRouter.h` — Entry struct, store()/drain()/drainAll() declarations, residentCountAtomic(), reclaimCount()
- `src/audioengine/ISRRetireRouter.cpp` — store() impl, drain(), drainAll(), terminalReclaim(), drainAllQuarantineStore()
- `src/audioengine/RetireQuarantineStore.h` — drainAllUnsafe(), capacity constants
- `src/DeferredDeletionQueue.h` — kQueueSize=4096, DeletionEntryType
- `src/audioengine/AtomicAccess.h` — convo::fetchAddAtomic, consumeAtomic, publishAtomic
- `src/audioengine/AudioEngine.h` — RuntimeBackpressureTelemetry, isFullyDrained, enqueueDeferredDeleteNonRtWithResult
- `src/audioengine/AudioEngine.Threading.cpp` — isFullyDrained() terminal check
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` — shutdown drain sequence
- `src/audioengine/RuntimeHealthMonitor.cpp` — takeSnapshot(), checkRetireStall

### Cross-references

- Step 4 evidence: `evidence/phase-d101-9-step4-candidate-b-contract-audit.md`
- Step 3 evidence: `evidence/phase-d101-9-step3-candidate-a-feasibility-audit.md`
- Step 2 evidence: `evidence/phase-d101-9-step2-terminal-candidate-d-safety-audit.md`
- Step 1 census: `evidence/phase-d101-9-step9b-terminal-capacity-census.md`
- Design contract: `doc/work88/I4_DESIGN_CONTRACT.md` (D14.3, D15.2)
