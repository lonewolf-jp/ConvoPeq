# D101-9 Phase 9-B Step 5-F: K_terminal Derivation Model

## Status: COMPLETE ✅ (Step 5-F)

## Date
2026-08-23

## Context

Step 5-I telemetry instrumentation and Step 5-II telemetry snapshot integration are complete.
This document defines the K_terminal derivation model — the formula and measurement
protocol that will allow deriving a concrete K_terminal value once telemetry data is
collected from production runs.

**Explicitly NOT** selecting K_terminal value or modifying HealthMonitor thresholds.
The derivation model is defined; the actual value selection awaits measured data
(Step 5-V, pending).

## 1. sizeof(Entry) Verification

Verified via compiled test (`tools/verify_entry_sizeof.cpp`):

```
sizeof(TestEntry) = 40 bytes
offsetof ptr:     0
offsetof deleter: 8
offsetof epoch:   16
offsetof type:    24
offsetof reason:  32
alignof(TestEntry) = 8
```

| Field | Type | Size | Offset |
|-------|------|------|--------|
| `ptr` | `void*` | 8 | 0 |
| `deleter` | `void(*)(void*)` | 8 | 8 |
| `epoch` | `uint64_t` | 8 | 16 |
| `type` | `DeletionEntryType` (`uint8_t`) | 1 | 24 |
| *(padding)* | — | 7 | 25–31 |
| `reason` | `const char*` | 8 | 32 |
| **Total** | | **40** | |

Memory implication: 100,000 entries = 4 MB (Non-RT, acceptable).

## 2. Telemetry Implementation Status (Step 5-I + 5-II Complete)

### 4 new atomic members on TerminalReclaimAuthority (ISRRetireRouter.h):

| Member | Type | Purpose |
|--------|------|---------|
| `terminalPeakResident_` | `atomic<uint32_t>` | Peak Terminal resident (max of residentAtomic_) |
| `terminalStoreCount_` | `atomic<uint64_t>` | Cumulative store() entries (Generic + World) |
| `terminalDrainAllCount_` | `atomic<uint64_t>` | Cumulative drainAll() invocations |
| `terminalDrainEntryCount_` | `atomic<uint64_t>` | Cumulative entries drained by drainAll() |

### 4 new API methods on ISRRetireRouter:
- `terminalPeakResident()` → delegates to `m_terminalReclaim.terminalPeakResident()`
- `terminalStoreCount()` → delegates to `m_terminalReclaim.terminalStoreCount()`
- `terminalDrainAllCount()` → delegates to `m_terminalReclaim.terminalDrainAllCount()`
- `terminalDrainEntryCount()` → delegates to `m_terminalReclaim.terminalDrainEntryCount()`

### RuntimeBackpressureTelemetry integration (AudioEngine.h):
Added 5 new fields: `terminalStoreCount`, `terminalDrainAllCount`, `terminalDrainEntryCount`,
`terminalPeakResident`, `terminalResident` — all populated in `getRuntimeBackpressureTelemetry()`.

### Atomic convention fixes:
- `store()`: `fetch_add` → `convo::fetchAddAtomic`
- `drain()`: `fetch_sub` → `convo::fetchSubAtomic`, `++reclaimCount_` → `convo::fetchAddAtomic`
- `drainAll()`: `store(0)` → `convo::publishAtomic`, `++reclaimCount_` → `convo::fetchAddAtomic`
- `recordWorldReclaim()`: `++reclaimCount_` → `convo::fetchAddAtomic`

### Test verification:
- T-5.1 through T-5.6 all pass (exit code 0)
- No regressions in RetireGraceSemanticsTests, ShutdownRetireIntentDrainTests, StuckReaderFallbackDrainTests

## 3. K_terminal Two-Model Approach (to be compared in Step 5-V)

Step 5-F defines two independent sizing models. They will be compared
against measured data in Step 5-V. Neither is selected in this step.

### 3.1 Model A — Rate-based (theoretical)

```
K_rate = λ_terminal_peak × T_stall_design × S
```

| Symbol | Meaning | Source |
|--------|---------|--------|
| `λ_terminal_peak` | Peak rate of Terminal arrival (entries/sec) | `terminalStoreCount` delta over sampling window |
| `T_stall_design` | Design stall duration (seconds) | **MEASURED** in Step 5-III (NOT `kResidencyStuckUs`) |
| `S` | Safety factor (>= 2 for burst absorption) | Starting hypothesis: 2× |

**Important**: `T_stall_design` is a **measured value** from Step 5-III
(T3/T4 worst-case stall duration). `kResidencyStuckUs = 1s` is only the
stuck-detection threshold, NOT the actual stall duration that produces
peak Terminal occupancy.

#### λ_terminal (Terminal arrival rate)

```
λ_terminal(t) = ΔterminalStoreCount / Δt
λ_terminal_peak = max_over_t(λ_terminal(t))
```

- **Measurement**: Sample `terminalStoreCount` every 100ms (health timer tick).
- Entries/sec reaching Terminal after D (4096), Q (512), E (512) absorb.
- Under normal operation, λ_terminal ≈ 0.
- Under reader stall, λ_terminal > 0 as Q/E fill and entries escalate.

#### T_stall_design (actual stall duration — to be measured)

From `EpochDomain.h`:
- `kResidencyStuckUs = 1,000,000` (1 second) — stuck reader DETECTION threshold
- `kChronicResidencyUs = 30,000,000` (30 seconds) — chronic stuck threshold

**NOT to be confused with actual stall duration.** The actual stall duration
that produces peak Terminal occupancy must be **measured** in Step 5-III by
correlating `activeReaderCount > 0` + `minReaderEpoch が進まない` over time.

#### S (Safety factor)

Starting hypothesis: `S = 2`

Justification:
1. Burst arrival rate can be 2× steady-state (empirically verifiable)
2. Q (512) + E (512) = 1024 entries provide absorption before Terminal escalation
3. Terminal drain rate ≥ arrival rate during non-stuck operation

### 3.2 Model B — Observed peak (empirical)

```
K_observed = max_over_sessions(terminalPeakResident)
```

- **Measurement**: `terminalPeakResident()` returns running max (CAS-updated in `store()`).
- **No formula assumptions** — purely empirical from production data.

### 3.3 Step 5-V comparison

In Step 5-V, compare:

```
K_rate = λ_terminal_peak × T_stall_design × S
K_observed = max(terminalPeakResident)
```

- If `K_rate ≥ K_observed`: rate model is conservative → use `K_rate` (more principled).
- If `K_rate < K_observed`: rate model underestimates → investigate missing factors
  (burst spikes, multi-reader interaction, drain scheduling gaps).

### 3.4 What NOT to do in Step 5-F

- **Do NOT** set `K_terminal = K_rate` or `K_terminal = K_observed` — deferred to Step 5-V
- **Do NOT** fix `T_stall_max = 1s` — it's a measured variable, not a constant
- **Do NOT** fix `S = 2` — it's a hypothesis to be validated
- **Do NOT** decide which model (A or B) is authoritative — that's Step 5-V's job

## 4. Measurement Protocol

### 4.1 Sampling strategy

| Metric | Source API | Sampling frequency | Window |
|--------|-----------|--------------------|--------|
| `terminalResident` | `terminalReclaimResidentCount()` | 100ms (health timer tick) | Running max |
| `terminalPeakResident` | `terminalPeakResident()` | 100ms | Read-only (CAS-updated in store()) |
| `terminalStoreCount` | `terminalStoreCount()` | 100ms | Delta for rate calculation |
| `terminalDrainAllCount` | `terminalDrainAllCount()` | 100ms | Monotonic counter |
| `terminalDrainEntryCount` | `terminalDrainEntryCount()` | 100ms | Delta for drain rate |
| `activeReaderCount` | `activeReaderCount()` | 100ms | Correlation with stall |
| `minReaderEpoch` | `minReaderEpoch()` | 100ms | Epoch stagnation detection |
| `pendingRetireCount` | `pendingRetireCount()` | 100ms | D queue occupancy |
| `quarantineResidentCount` | `quarantineResidentCount()` | 100ms | Q + E occupancy |

### 4.2 Worst-case workload scenarios (from Step 5 audit)

| Scenario | T1-Normal | T2-Short-stall | T3-Long-stall | T4-Repeated-publish | T5-Shutdown | T6-Stuck-shutdown |
|----------|-----------|-----------------|---------------|---------------------|-------------|-------------------|
| Reader stall | 0ms | 10-100ms | 1-30s | 1-30s | 0 | 1-30s |
| Terminal expected | ≈0 | spike then drain | grows | accumulates | →0 | →0 |
| **Measures K_terminal?** | Baseline | Secondary | **PRIMARY** | **WORST** | Verification | Safety net |

### 4.3 K_terminal sizing data collection

```
K_terminal_candidate = 2 × max_over_all_sessions(terminalPeakResident_ observed)

// After collecting N sessions under T1-T4 scenarios:
K_terminal = 2 × max(terminalPeakResident_ values from all sessions)

// Validate:
// - Under T1: terminalPeakResident_ should be near 0
// - Under T3: terminalPeakResident_ should reflect (λ × T_stall)
// - Ratio: peak / steady-state should be ≤ S (validates safety factor)
// - T5/T6: terminalDrainEntryCount_ should equal prior terminalStoreCount_ (no leaks)
```

## 5. B5 Gate Table — K_terminal Specific

| B5-ID | Gate | Verification method | Status |
|-------|------|---------------------|--------|
| B5-1 | Entry struct layout | `verify_entry_sizeof.cpp` compiled test | ✅ PASS (40 bytes) |
| B5-2 | Peak observability | `terminalPeakResident()` returns 0 initially, tracks max | ✅ PASS (T-5.1, T-5.4) |
| B5-3 | Store count observability | `terminalStoreCount()` returns 0 initially, increments | ✅ PASS (T-5.1, T-5.2) |
| B5-4 | Drain all count observability | `terminalDrainAllCount()` returns 0 initially, increments | ✅ PASS (T-5.1, T-5.5) |
| B5-5 | Drain entry count observability | `terminalDrainEntryCount()` returns 0 initially, tracks drains | ✅ PASS (T-5.1, T-5.5) |
| B5-6 | Atomic convention compliance | All atomic ops use convo:: wrappers | ✅ PASS (all tests pass) |
| B5-7 | Telemetry snapshot integration | `RuntimeBackpressureTelemetry` includes 5 new fields | ✅ PASS (build OK) |
| B5-8 | No HealthMonitor modification | HealthMonitor thresholds unchanged | ✅ PASS (not modified) |
| B5-9 | No K_terminal value selection | K_terminal value NOT set (deferred to Step 5-V) | ✅ PASS (not selected) |

## 6. Next Steps

- **Step 5-III**: Correlated observation T1-T6 (requires production run with health monitor sampling)
- **Step 5-V**: K_terminal derivation (derive actual value from observed `terminalPeakResident_`)
- **Step 5-VI**: HealthMonitor thresholds (only AFTER K_terminal is derived)

---

## Summary

Two K_terminal sizing models are defined for comparison in Step 5-V:

**Model A (rate-based):**
$$K_{rate} = \lambda_{terminal\_peak} \times T_{stall\_design} \times S$$

**Model B (observed peak):**
$$K_{observed} = \max(\text{terminalPeakResident})$$

Where:
- `λ_terminal_peak` is the peak Terminal arrival rate measured from `terminalStoreCount` deltas (100ms sampling)
- `T_stall_design` is the **measured** stall duration from Step 5-III (NOT `kResidencyStuckUs = 1s`, which is only the detection threshold)
- `S = 2` is a starting hypothesis (burst absorption safety factor)

All telemetry instrumentation is complete and verified. Neither model is selected as
authoritative in Step 5-F — that decision is deferred to Step 5-V pending measurement
data from Step 5-III correlated observation.
