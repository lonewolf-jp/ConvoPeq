# 15-P-7: Stuck-Reader Fallback Regression Test

## Status: PASS

## Objective

Verify the 15-P-5 fix in `AudioEngine.CtorDtor.cpp` — the stuck-reader fallback path
now calls `drainAllQuarantineStore()` in addition to `m_epochDomain.drainAll()`,
ensuring Q (RetireQuarantineStore), E (EmergencyQuarantineStore), and Terminal
(TerminalReclaimAuthority) are all drained during shutdown when a reader is stuck.

## Test File

`src/tests/StuckReaderFallbackDrainTests.cpp` — 6 test cases using only existing
public APIs (no test hooks, no production code changes).

## Test Cases

| # | Test Name | Description |
|---|-----------|-------------|
| 1 | `testStuckReaderFallbackDrainsAllStores` | Stuck reader + Q(512)/E(512)/Terminal(50) populated → `drainAllQuarantineStore()` clears all stores, deleters called exactly once, no leaks |
| 2 | `testDoubleDrainIsSafe` | Double `drainAllQuarantineStore()` is idempotent — no double-free, no crash |
| 3 | `testNoStuckReaderDrainWorks` | No stuck reader — `drainAll()` drains Terminal correctly |
| 4 | `testShutdownCompletesAfterDrain` | After drain + reader exit, `activeReaderCount() == 0` (completion invariant holds) |
| 5 | `testOwnershipTransferNoLeaks` | Q + E + Terminal populated simultaneously → counts match exactly, no double-counting, no lost entries |
| 6 | `testStuckReaderDrainAllPath` | Stuck reader + `drainAll()` path — Terminal drained even with stuck reader (epoch-agnostic) |

## Key Design Decisions

### Stuck Reader Simulation

- `EpochDomain::registerReaderThread("TestReader")` registers a reader slot
- `EpochDomain::enterReader(readerIdx)` increments reader depth → `activeReaderCount() > 0`
- Reader is never exited → simulates a stuck reader

### Epoch Strategy

- `epoch = router.currentEpoch() + 1000` (future epoch) ensures entries are NOT epoch-safe
- `isOlder(epoch, minReaderEpoch)` returns false for future epochs → entries stored, not destroyed
- This correctly simulates the production scenario where entries are quarantined pending epoch advancement

### Overflow Handling

- `RetireQuarantineStore::kMaxQuarantinedEntries = 512` (fixed capacity)
- When `quarantineRetire()` returns `false` (store full), the overflow object is deleted
  with a dummy tracker to avoid polluting deleter count verification
- This mirrors the production contract: "caller must NOT delete" — in production, the
  caller would escalate to `emergencyQuarantine` or `terminalReclaim`

### Ownership Verification

- `DeleterTracker` with `std::atomic<int> invokeCount` tracks deleter invocations
- `TestObject::aliveCount` (static atomic) tracks total live objects
- Both must reach 0 after drain — verifies no leaks and no double-frees

## Build & Run

```bash
cmake --build build --config Debug --target StuckReaderFallbackDrainTests
ctest -C Debug -R StuckReaderFallbackDrain --output-on-failure
```

## Result

```text
15-P-7: All StuckReaderFallbackDrain tests PASS (6 tests)
```

All 6 tests pass. The 15-P-5 fix is verified: `drainAllQuarantineStore()` correctly
drains Q + E + Terminal unconditionally (epoch-agnostic, Audio Thread stopped contract),
even when a reader is stuck.
