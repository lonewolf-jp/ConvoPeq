# Phase D101.9 Step 5-III-A: Observation Capture Readiness Audit

## Status

- **Step 5-III-A-1** ✅ COMPLETE — Contract audit table + Option A decision
- **Step 5-III-A-2** ✅ COMPLETE — `RuntimeBackpressureTelemetry` extended (Option A)
- **Step 5-III-A-3** ✅ COMPLETE — 100ms periodic capture implemented in `AudioEngine.Timer.cpp`
- **Step 5-III-A-4** ✅ COMPLETE — Scenario = external/manual annotation (no embedded detection)

## Audit Contract Table

10 observation fields × 4 assessment columns. Source: `RuntimeBackpressureTelemetry` (populated from `ISRRetireRouter` public APIs).

| Field | Obtainable? | Snapshot-Synchronized? | Loggable at 100ms? | Needs Modification? |
| ---- | ---- | ---- | ---- | ---- |
| `terminalStoreCount` | ✅ `ISRRetireRouter::terminalStoreCount()` | ✅ atomic load-acquire | ✅ `[D101_9_T5_OBS]` | ❌ (already in struct, Step 5-I) |
| `terminalDrainAllCount` | ✅ `ISRRetireRouter::terminalDrainAllCount()` | ✅ atomic load-acquire | ✅ `[D101_9_T5_OBS]` | ❌ (already in struct, Step 5-I) |
| `terminalDrainEntryCount` | ✅ `ISRRetireRouter::terminalDrainEntryCount()` | ✅ atomic load-acquire | ✅ `[D101_9_T5_OBS]` | ❌ (already in struct, Step 5-I) |
| `terminalPeakResident` | ✅ `ISRRetireRouter::terminalPeakResident()` | ✅ atomic load-acquire (CAS peak) | ✅ `[D101_9_T5_OBS]` | ❌ (already in struct, Step 5-I) |
| `terminalResident` | ✅ `ISRRetireRouter::terminalReclaimResidentCount()` | ✅ atomic load-acquire | ✅ `[D101_9_T5_OBS]` | ❌ (already in struct, Step 5-I) |
| `quarantineResident` | ✅ `quarantineResidentCount()` (Q+E aggregate) | ✅ atomic load-acquire | ✅ `[D101_9_T5_OBS]` | ❌ (already in struct) |
| `emergencyQuarantineResident` | ✅ `emergencyQuarantineResidentCount()` (E only) | ✅ atomic load-acquire | ✅ `[D101_9_T5_OBS]` (NEW) | ✅ Added to struct + getter |
| `activeReaderCount` | ✅ `ISRRetireRouter::activeReaderCount()` | ✅ atomic load-acquire | ✅ `[D101_9_T5_OBS]` (NEW) | ✅ Added to struct + getter |
| `minReaderEpoch` | ✅ `ISRRetireRouter::minReaderEpoch()` | ✅ atomic load-acquire | ✅ `[D101_9_T5_OBS]` (NEW) | ✅ Added to struct + getter |
| `pendingRetireCount` | ✅ `ISRRetireRouter::pendingRetireCount()` (D queue) | ✅ atomic load-acquire | ✅ `[D101_9_T5_OBS]` (NEW) | ✅ Added to struct + getter |

## Decision: Option A (Extend `RuntimeBackpressureTelemetry`)

### Rationale

1. **Single-snapshot consistency**: All 10 fields are obtainable from `m_retireRouter` public APIs in one call to `getRuntimeBackpressureTelemetry()`. No need for 4+ separate lock-free reads scattered across timerCallback.
2. **Existing design precedent**: Step 5-I already extended `RuntimeBackpressureTelemetry` with 5 Terminal fields and 4 non-Terminal fields already existed (`quarantineResident`). Adding 4 more Q/E/D fields is consistent with this pattern.
3. **No HealthMonitor modification**: `TrendSnapshot` in `RuntimePolicyEngine.h` is NOT modified (per instructions). The observation struct is `RuntimeBackpressureTelemetry`, not `TrendSnapshot`.
4. **ABI impact is acceptable**: `RuntimeBackpressureTelemetry` is an internal diagnostic struct, not part of any public plugin API or serialized format.
5. **Option B (separate calls) rejected**: Would introduce 4 additional `m_retireRouter` dereferences per 100ms tick, risking temporal inconsistency between fields (e.g., `pendingRetireCount` could be read before a drain, `activeReaderCount` after).

### Fields Added to `RuntimeBackpressureTelemetry`

```cpp
// ★ Step 5-III-A: Observation snapshot fields
uint32_t activeReaderCount = 0;
std::uint64_t minReaderEpoch = 0;
uint32_t pendingRetireCount = 0;
std::uint64_t emergencyQuarantineResident = 0;
```

### Fields Populated in `getRuntimeBackpressureTelemetry()`

```cpp
const auto readerCount = (m_retireRouter != nullptr)
    ? m_retireRouter->activeReaderCount() : 0u;
const auto minReader = (m_retireRouter != nullptr)
    ? m_retireRouter->minReaderEpoch() : std::uint64_t{0};
const auto pendingRetire = (m_retireRouter != nullptr)
    ? m_retireRouter->pendingRetireCount() : 0u;
const auto emergencyResident = (m_retireRouter != nullptr)
    ? static_cast<std::uint64_t>(m_retireRouter->emergencyQuarantineResidentCount()) : 0u;
// ...returned in aggregate initializer
```

## Periodic Capture Implementation

### Location

`src/audioengine/AudioEngine.Timer.cpp` — `timerCallback()`, before the existing change-triggered `[BACKPRESSURE]` block.

### Format

```text
[D101_9_T5_OBS] T_store=N T_drainAll=N T_drainEntry=N T_peak=N T_resident=N Q_resident=N E_resident=N activeReaders=N minEpoch=N pendingRetire=N pressureLevel=N
```

### Key Design Decisions

- **NOT change-triggered**: Captures every 100ms tick under `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`, independent of whether values changed.
- **Does NOT replace `[BACKPRESSURE]`**: The existing change-triggered `[BACKPRESSURE]` log remains for runtime diagnostics. The new `[D101_9_T5_OBS]` log is an additional measurement probe.
- **Single snapshot**: One `getRuntimeBackpressureTelemetry()` call per tick, ensuring snapshot-synchronized field capture.
- **Scenario = external/manual**: The `[D101_9_T5_OBS]` tag is a stable identifier for log parsing. Scenario annotation (e.g., "T1: idle_steady_state" or "T3: high_retire_load") is applied externally during log post-processing. No embedded scenario detection logic in `AudioEngine`.

## G6 Ownership Accounting Resolution

Per instructions, G6 uses **Option 1** (shutdown-only accounting via `drainAll()`). `terminalDrainEntryCount` only counts `drainAll()` entries, not `drain()` entries. This is sufficient for the T5/T6 measurement runs which focus on shutdown behavior. The `terminalDrainEntryCount` field in the observation captures cumulative drainAll count for drain accounting verification.

## Next Steps

- Step 5-III-B: T1-T6 measurement runs (idle steady-state, high retire load, etc.)
- Step 5-V: K_terminal selection — compare `K_rate` (λ_peak × T_stall_design × S) vs `K_observed` (max(terminalPeakResident))
- Step 5-VI: HealthMonitor threshold configuration (NOT before Step 5-V)
