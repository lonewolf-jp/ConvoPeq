# Repair Plan — Final Report

## Session Summary

All workflow checks passed (3 failures fixed). This report consolidates:
1. **Unimplemented items** (未実装)
2. **Implementation gaps** (実装もれ)
3. **Newly discovered bugs** (新たに見つかったバグ)

---

## 1. Unimplemented Items (未実装)

| Item | Description | Location |
|------|-------------|----------|
| B4 pipeline flip | `enqueuePublicationIntent` is dormant — defined but not wired to live publish path | `RuntimePublicationOrchestrator.h:156` |
| `PublishDecisionSnapshot` | Defined but never generated — `enqueuePublicationIntent` is dormant | `ISRRuntimePublicationCoordinator.h` |
| `makePublishDecisionSnapshot` | Helper not yet created — decision generation not unified | `doc/work88/REPAIR_PLAN.md:2186` (serena memory B4 recommendation) |
| `ShutdownBlockingReason::ActiveBuilder` | Enum value missing — SHUTDOWN-7 confirmed gap | `ISRShutdown.h:46-57` |
| `static_assert(DispatcherHasNoDecision)` | Trait does not exist — only 1:1 size check implemented | `ISRIntentDispatcher.h` |
| FUTURE-10 | Observe/Recovery common `intentQueue_` integration — 50% done | `.h:376,394,399` |
| `hasDeferred_` → atomic | TODO at line 1112: `hasDeferred_` still non-atomic | `doc/work88/REPAIR_PLAN.md:1112` |

---

## 2. Implementation Gaps (実装もれ)

| Gap | Detail | Status |
|-----|--------|--------|
| `receiptReady_.store` relaxed ordering | `AudioEngine.Timer.cpp:1775,1829` — uses `relaxed` on `receiptReady_` | Checker undetected — fix needed |
| `isAllZero()` missing field | `RuntimeDrainAudit::isAllZero()` (`RuntimeDrainAudit.h:77-83`) audits 5 fields but **not** `overflowRingResident` — contradicts own comment at `:50` | SHUTDOWN-3/6 mismatch |
| `DispatcherHasNoDecision` | Static assert checks size 1:1 only; the trait/type itself **does not exist** | HANDLER-1 contract unenforced at compile time |
| `ActiveBuilder` enum | `ShutdownBlockingReason` has 7 values; `ActiveBuilder` missing required for SHUTDOWN-7 | Not added to enum |

---

## 3. Newly Discovered Bugs (BUG-047 — BUG-065)

19 bugs documented in `doc/work89/`. Summary:

### Concurrency / Data Race (5)
| BUG | Summary |
|-----|---------|
| **BUG-052** | `consumeDeferredRequest` accesses non-atomic `hasDeferred_` |
| **BUG-060** | `quarantineResidentCount_` TOCTOU race → unsigned underflow in `reclaim()` |
| **BUG-061** | `DSPQuarantineManager` — unprotected concurrent `std::vector` access (auditLog_) |
| **BUG-063** | `EpochDomain::ReaderSlot::ownerTag` (non-atomic `char[32]`) — data race in `detectStuckReaders` |
| **BUG-049** | Concurrent `quarantineFlags` store — immediate vs deferred quarantine conflict |

### Epoch / Reader Issues (2)
| BUG | Summary |
|-----|---------|
| **BUG-048** | `detectStuckReaders` breaks on first match, misses severe stuck readers |
| **BUG-050** | `enterReader` HB ordering — epoch store after `depth++`, `getMinReaderEpoch` may read stale epoch |

### DSP Transition / Crossfade (4)
| BUG | Summary |
|-----|---------|
| **BUG-051** | `exchangeFadingRuntimeDSP` sentinel mask — dead code |
| **BUG-054** | `onPublishCompleted` crossfade handle mismatch — uses `getActiveRuntimeDSPHandle()` as `oldHandle` but may not be correct when multiple fading DSPs exist |
| **BUG-055** | `runPublicationPrecheckNonRt` — unreachable `else if` branch (dead code) |
| **BUG-056** | `checkCrossfadeTimeout`/`checkCrossfadeEventDrop` — MonitorState stuck in Error, no recovery path |

### Health Monitor / State Management (5)
| BUG | Summary |
|-----|---------|
| **BUG-057** | `checkOverflowRate` — `RETIRE_STALL` eventCode misused (name misleading vs semantics) |
| **BUG-058** | `checkWorldConsistency` — reuses `m_prevConfigDivergenceState_` instead of dedicated MonitorState field |
| **BUG-059** | `reset()` — MonitorState field reset inconsistency (some reset to Normal, others left stale) |
| **BUG-062** | `checkRetireReclaimLatency` — uint64_t path missing normal recovery event (`EVENT_RETIRE_AGE_NORMAL`) |
| **BUG-003** | AGC reset path inconsistency (double path ordering) — related to BUG-065 |

### Audio Processing / Cache (3)
| BUG | Summary |
|-----|---------|
| **BUG-047** | `EQCoeffCache::computeParamsHash` — missing `sampleRate`/`maxBlockSize` → stale EQ coefficients after sample rate change |
| **BUG-053** | `stopNoiseShaperLearning` calls `stopLearning()` twice (queue via + direct) |
| **BUG-065** | `EQProcessor::reset()` — sets `agcResetSerial` to 0 instead of incrementing → AGC state not reset |

### Output Path (1)
| BUG | Summary |
|-----|---------|
| **BUG-064** | Float/Double output path ordering inconsistency — `applyFixedLatencyDelay` vs hard clamp ordering differs |

---

## Action Items (Priority)

### P0 — Fix immediately
1. **BUG-052**: Change `hasDeferred_` to `std::atomic<bool>` (per TODO at `doc/work88/REPAIR_PLAN.md:1112`)
2. **BUG-060**: Fix TOCTOU race on `quarantineResidentCount_` in `reclaim()`
3. **BUG-063**: Make `ownerTag` atomic or add synchronization
4. **BUG-065**: Increment `agcResetSerial` in `EQProcessor::reset()`
5. **BUG-064**: Unify Float/Double output path ordering
6. **SHUTDOWN-7**: Add `ActiveBuilder` enum + `VerifyDrained` check for `rebuildWorkerRunning`

### P1 — Fix soon
- **BUG-047**: Include `sampleRate`/`maxBlockSize` in `computeParamsHash`
- **BUG-048, BUG-050**: Fix `detectStuckReaders` break-on-first + `enterReader` HB ordering
- **BUG-054**: Crossfade handle selection under multiple fading DSPs
- **BUG-056**: Add Error→Normal recovery path in MonitorState
- **`isAllZero()`**: Add `overflowRingResident` check
- **`static_assert(DispatcherHasNoDecision)`**: Implement the trait
- **B4 flip**: Wire `enqueuePublicationIntent` to live path, create `makePublishDecisionSnapshot`

### P2 — Future / design
- **BUG-049, BUG-051, BUG-053, BUG-055, BUG-057, BUG-058, BUG-059, BUG-061, BUG-062**: Address per individual BUG files
- **FUTURE-10**: Complete Observe/Recovery common queue integration

---

## Appendix A: Implemented Items (実装済み事項)

Excerpted from `doc/work88/REPAIR_PLAN.md` — items marked ✅ at 2026-07-31 verification.

### A.1 Sanitizer / CI (ASan-CMAKE series)

| ID | Contract | Status |
|----|----------|--------|
| CMakeLists ENABLE_ASAN | ✅ 実装済み（line 1123） |
| CMakeLists ENABLE_TSAN | ✅ 実装済み（line 1159） |
| ASan ブロック — PGO 排他 | ✅ 実装済み（line 1127） |
| ASan ブロック — LTCG/IPO 無効化 | ✅ 実装済み（line 1077,1084） |
| ASan ブロック — 条件付き CRT フラグ | ✅ 実装済み（line 1068,1111,1147） |
| TSAN/ASAN 排他 | ✅ 実装済み（line 1161-1163） |
| TSAN MSVC 拒否 | ✅ 実装済み（line 1166-1167） |
| sanitizer-ci.yml | ✅ 実コード検証済み（2026-07-31） |
| debug-asan green | ✅ ctest 23/23 PASS |
| debug-tsan job 定義 | ✅ best-effort / graceful skip |
| ASan-CMAKE-1〜10 | ✅ All contracts implemented & verified |

### A.2 FUTURE Series

| FUTURE ID | Summary | Status |
|-----------|---------|--------|
| FUTURE-4 | Metadata Snapshot — `persistentState_` removed, `currentPublicationEpoch()` derives from `currentWorld_` | ✅ **Fully implemented** |
| FUTURE-3 | `submitRecoveryRequest()` / `popRecoveryRequest()` — rollback removed | ✅ **Fully implemented** |
| FUTURE-7 | `emitQuarantineIntent`→`submitQuarantine()`, `emitObserveIntent`→`submitObserve()` async | ✅ **Fully implemented** |
| FUTURE-8 | `observeDeferredRing_` + `drainObserveDeferred()` | ✅ **Fully implemented** |
| FUTURE-9 | Coordinator Loop + `hasDeferred_` atomic | ⚠️ **Partially** — `CoordinatorLoop` implemented, but `hasDeferred_` is still `bool` (BUG-052) |
| FUTURE-10 | Observe/Recovery common `intentQueue_` | ⚠️ **50%** — Queue/IntentType implemented, but Observe/Recovery still use dedicated queues |

### A.3 P0-P2 Implementation Items

| Item | Description | Status |
|------|-------------|--------|
| P0-1 | SafeStateSwapper tail 2-writer 解消 (head 専用化) | ✅ `tryReclaimSlot()` / `advanceHead()` / `ReclaimResult` enum |
| P0-2 | EQCoeffCache DSPHandleRuntime移行 | ✅ `RefCountedDeferred`継承削除, `CacheMap`→`DSPHandle` |
| P0-2b | PublishReceipt DSPCore*削除 | ✅ `PublishReceipt::dsp` removed |
| P0-3 | AudioSegmentBuffer 61MB ヒープ化 | ✅ `ScopedAlignedPtr` + Rule of Five |
| P0-4A | emitRetireIntent | ✅ `ISRRetire.h/cpp` implemented |
| P0-4B | Delete Authority — reclaim() Coordinator専用化 | ✅ `reclaim()` private, `shutdownReclaim()` added |
| P0-4C | Coordinator Interface拡充 | ✅ `emitObserveIntent()`, `emitQuarantineIntent()`, `requestReclaim()` |
| P0-5 | QuarantineService | ✅ `QuarantineService` class + `emitQuarantineIntent()` |
| P1-1 | FFT Backend Concept 全5Phase | ✅ `FFTBackend.h/cpp`, `FFTExecutionContext.h`, etc. |
| P1-2 | Receipt状態機械 | ✅ `resetReceipt()` + `QuarantineReason::ReceiptReset` |
| P2 | updateAudioSegmentBufferFade 削除 | ✅ Deleted |

### A.4 Design Contracts (OBSERVE-1〜10, etc.)

| Contract | Description | Status |
|----------|-------------|--------|
| OBSERVE-1 | Timer は ObserveIntent のみ発行 | ✅ 完了 |
| OBSERVE-3 | Coordinator Loop は processIntent() で処理 | ✅ 完了 |
| OBSERVE-7 | ACK は Epoch 安全確認後の通知 | ✅ 完了 |
| SHUTDOWN-2/3/4/6 | Drain: Queue空 / Retire Router / Epoch Complete / Verify Empty | ✅ 実装済み (SHUTDOWN-6: `isAllZero()` missing `overflowRingResident` — **SEE IMPL GAP**) |
| SHUTDOWN-7 | No Active Builder | ⚠️ **未実装** — `ActiveBuilder` enum value missing |
| QUEUE-21/22 | Intent variant + kDispatchTable | ✅ 実装済み (DispatcherHasNoDecision trait — **SEE UNIMPL**) |
| RECOVERY-7 | Recovery Coalescing in Builder PendingMap | ✅ Design finalized (code: design only) |
| BUILDER-STATE | PendingMap Build Session 限定 | ✅ Design finalized (code: design only) |

### A.5 Codebase Verification (2026-07-29)

| Check | Tool | Result |
|-------|------|--------|
| EQCoeffCache 継承関係 | grep | ✅ P0-2 完了 |
| DSPHandleRuntime 実装 | AiDex | ✅ create/resolve/retire/quarantine/reclaim 全API稼働 |
| getVersion() | grep | ✅ `ISRRuntimePublicationCoordinator.cpp:168-175` |
| kMaxFallback | grep | ✅ 4096 / 1024 |
| kMaxEpochDrift | grep | ✅ 10 |
| MMCSS例外登録簿 | grep/ls | ✅ `doc/exception_registry.md` exists |
| `[[deprecated]]` on atomic cache | grep | ✅ Not present — FUTURE-4 end-state directly (no transition cache) |

### A.6 Previously Fixed Bugs (work88 audit)

| BUG | Fix | Status |
|-----|-----|--------|
| BUG-011/012/013 | `sigma = std::clamp(s, sigmaMin, sigmaMax)` | ✅ Fixed in `CmaEsOptimizer.h:84` |
| BUG-015 | `enqueueWithRetry` リトライロジック | ✅ Fixed in `ISRRetireRouter.cpp:161` |
| BUG-028 | `complete()` で全フラグリセット | ✅ Verified |
| BUG-029 | Emergency Override で `exchangeFadingRuntimeDSP` | ✅ Verified |
| BUG-038 | `FFT_MAGNITUDE_SCALE = 2.0f / NUM_FFT_POINTS` | ✅ Fixed in `SpectrumAnalyzerComponent.h:74` |

---

*Generated: 2026-08-04*
```
