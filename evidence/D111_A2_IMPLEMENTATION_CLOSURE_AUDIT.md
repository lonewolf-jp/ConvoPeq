# D111 — A2 Implementation Closure / Build-Test Evidence Audit

**Status:** **D111 verdict: Case A — A2 IMPLEMENTATION CLOSED / IMPLEMENTATION ACCEPTED**
**Source changes:** 0
**I4 changes:** 0
**Test changes:** 0
**Build/Test:** Debug 40/40 PASS, Release 40/40 PASS

D111 is a read-only verification audit. It does **not implement A2** and does **not
modify production source**. D111 closes the A2 reclaim path by **converting the
D110 Case-A GO judgment into empirical build + test evidence**.

D110 concluded Case A based on static source review of `ConvoPeq.md`
(2026-08-27 14:33 + 2026-08-28 21:22). D111 re-derives the same conclusion from
the **live source tree** + **fresh Debug + Release builds** + **fresh CTest runs**.

---

## D111-1 — Baseline

- **Production source baseline:** `ConvoPeq.md` 2026-08-28 21:22 (size 4,633,246 B,
  104,118 lines) + live `src/` tree on 2026-08-28 (working tree, no uncommitted
  changes that affect A2 paths).
- **Build baseline:** MSVC (cl 14.51.36231), Ninja Multi-Config, IPP enabled, MKL
  enabled, PGO off, ASan off. Build wrapper: `tools\build_with_vcvars.bat`.
- **CTest baseline:** CMake 4.4.2, CTest 3.x, test driver `ctest -C <Config> --output-on-failure`.
- **Build output directory:** `build\` (created 2026-08-28 22:55; Debug + Release
  multi-config). Logs: `build_release_D111.log`, `evidence/D111_ctest_debug.log`,
  `evidence/D111_ctest_release.log`.

The production tree at audit time contains **0 uncommitted source changes**
that affect the A2 reclaim path. All A2 evidence below is derived from the live
source, not the stale `build-t1-test/` or `build-asan-msvc/` artefacts (those
date to 2026-07-31 / 2026-08-19 and were NOT used for D111 closure).

### I4 contract anchor (D110 hand-off)

| Anchor | Text | Status |
|---|---|---|
| `I4_DESIGN_CONTRACT.md:2439` | `reclaim: requestReclaim → reclaimNormal (retire 冪等 + epoch 確認 + reclaim)` | **MATCHES current source** (`reclaimNormal` body at `ISRRuntimePublicationCoordinator.cpp:684-713` calls `retire → waitReaders → reclaim`). |
| `I4_DESIGN_CONTRACT.md:2443` | `→ 各 handle を requestReclaim（epoch 不安全は push_back で再登録）` | **MATCHES** (deferred path at lines 695-703). |
| `I4_DESIGN_CONTRACT.md:2527` | `retire → requestReclaim → epoch 安全確認（retireEpoch < minReaderEpoch）→ reclaim` | **MATCHES**. |
| `D2_IMPL_CHECKLIST.md:84` | "Step 9-14 完了。production reclaim 接続済み (tryShutdownQuiescentReclaim が CacheMap/ReleaseResources に接続)" | **MATCHES** (3 production callers verified below). |

---

## D111-2 — Production Caller Closure

### 2.1 `tryShutdownQuiescentReclaim` (A2 Shutdown path entry) — 3 production callers

| # | File:line | Caller | Lifecycle phase | Notes |
|---|---|---|---|---|
| 1 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:457` | `activeHandle` reclaim during graceful drain | DrainRetire (post-`closeReaderRegistration`) | `dspHandleRuntime_.retire(activeHandle)` → `tryShutdownQuiescentReclaim(activeHandle)` → `jassert(reclaimed)`. |
| 2 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:464` | `fadingHandle` reclaim during graceful drain | DrainRetire | Same shape, guarded by `fadingHandle != activeHandle`. |
| 3 | `src/audioengine/AudioEngine.h:2106` | `CacheMap::~CacheMap` for each `EQCoeffCache` entry | `Destroy` shutdown phase (line 2090 gate) | `tryShutdownQuiescentReclaim(entry.second)` → on success `rt.resolve(...)` → `delete EQCoeffCache` (line 2106-2111). |

Test references (excluded from production count): 0 in `src/tests/`.

**Verdict: 3/3 production callers confirmed.** D109-D110 claim is empirically true.

### 2.2 `requestReclaim` → `reclaimNormal` (NORMAL EBR path) — split confirmed

| Aspect | Current source | I4 §D37.2 | Verdict |
|---|---|---|---|
| `requestReclaim` definition | `ISRRuntimePublicationCoordinator.cpp:668-676` | I4:2439 | **MATCH** |
| `requestReclaim` delegates to | `reclaimNormal(handle, handleRuntime, router)` (line 675) | (implicit) | **MATCH** |
| `reclaimNormal` body | `ISRRuntimePublicationCoordinator.cpp:684-713` | I4:2439 | **MATCH** |
| 1. executeRetire (line 690) | `handleRuntime.retire(handle)` | I4:2439 "retire 冪等" | **MATCH** |
| 2. waitReaders (line 693-703) | `retireEpoch >= minReaderEpoch` → deferred (`onReclaimBegin` → return false) | I4:2439 "epoch 確認" | **MATCH** |
| 3. executeReclaim (line 707) | `handleRuntime.reclaim(handle)` (slot 状態遷移) → `onReclaimEnd` | I4:2439 "reclaim" | **MATCH** |
| **Permit?** | None (NORMAL EBR, epoch-gated) | I4 design: NORMAL = no Permit | **MATCH** (D110 §D110-4 two-path) |

`requestReclaim` does **NOT** delegate to `reclaimShutdownQuiescent` (verified:
`requestReclaim` body is exactly `return reclaimNormal(...)`, line 675).

### 2.3 `tryShutdownQuiescentReclaim` → `reclaimShutdownQuiescent` (SHUTDOWN path) — split confirmed

`AudioEngine.h:4367-4402`:

1. Q0–Q7 observation collection (lines 4372-4384) — `outstanding()==0`, `readerRegistrationClosed()`, `activeReaderCount()==0`, `epochSettled=true`, `postStopEnqueueZero=true`, `noResurrection=!isAdmissionOpen()`, `epochGeneration`, `readerRegistrationGeneration`.
2. `shutdownRuntime_.tryMakeQuiescenceProof(obs)` (line 4386) — Q0–Q7 verified inside `ISRShutdown.cpp:351-398`; binds `reclaimAuthority_.bindShutdownIdentity(id)` (line 385) as friend.
3. `shutdownRuntime_.tryMakeReclaimPermit(*proof)` (line 4390) — `ISRShutdown.cpp:400-412` issues `ReclaimPermit(proof.identity())`.
4. `runtimePublicationBridge_.reclaimShutdownQuiescent(handle, dspHandleRuntime_, *m_retireRouter, std::move(*permit))` (line 4400-4401).

`reclaimShutdownQuiescent` body at `ISRRuntimePublicationCoordinator.cpp:743-773`:

```cpp
if (!shutdownIdentityBound())                 return false;  // (a) identity 未 bind
if (!(permit.identity() == currentShutdownIdentity_))  // (b) cross-runtime / stale reject
    return false;
if (!permit.consume())                        return false;  // (c) 二重使用 reject (INV-LIFE-7)
handleRuntime.retire(handle);                              // (d) retire
handleRuntime.reclaim(handle);                             // (e) reclaim (slot 状態遷移)
return true;
```

This is the **D110-5 / D111-3 expected ordering** (Proof → Permit → identity validate
→ consume → retire → reclaim). All 5 sub-steps verified.

### 2.4 Verdict: **PASS** (3 production callers, two-path split intact)

---

## D111-3 — NORMAL/SHUTDOWN Path Closure (G23 physical destruction ordering)

### 3.1 `reclaimShutdownQuiescent` execution order (D111-3 expected: Proof → Permit → identity → consume → retire → reclaim → physical destruction)

Live source confirms:

| Step | Site | Mechanism | Verified |
|---|---|---|---|
| 1. Proof collection | `AudioEngine.h:4372-4384` | Q0–Q7 observation struct | ✅ |
| 2. Proof generation | `ISRShutdown.cpp:351-398` (lines 354-365 Q check, 374-378 identity, 385 `bindShutdownIdentity`, 387-397 `ShutdownQuiescenceProof{id}`) | All Q conditions, identity bound to ReclaimAuthority | ✅ |
| 3. Permit generation | `ISRShutdown.cpp:400-412` | `if (!proof.valid()) return nullopt;` then `ReclaimPermit permit(proof.identity());` | ✅ |
| 4. identity validation | `ISRRuntimePublicationCoordinator.cpp:759` | `permit.identity() == currentShutdownIdentity_` (engineInstanceId + shutdownGeneration + epochGeneration + readerRegistrationGeneration) | ✅ |
| 5. single-use consume | `ISRRuntimePublicationCoordinator.cpp:761` | `permit.consume()` returns false if already consumed (CAS) | ✅ |
| 6. retire | `ISRRuntimePublicationCoordinator.cpp:765` | `handleRuntime.retire(handle)` (idempotent) | ✅ |
| 7. reclaim | `ISRRuntimePublicationCoordinator.cpp:771` | `handleRuntime.reclaim(handle)` (Reclaimed 状態遷移) | ✅ |
| 8. physical destruction (CacheMap path) | `AudioEngine.h:2106-2111` | on `reclaimed == true`: `rt.resolve(...)` → `delete static_cast<EQCoeffCache*>(resolved.instance)` | ✅ |

**No path performs `delete` before `permit.consume()`.** All 3 production paths
(activeHandle, fadingHandle, CacheMap) follow `Proof → Permit → identity → consume →
retire → reclaim → delete` (CacheMap) or `Proof → Permit → identity → consume →
retire → reclaim` (ReleaseResources; physical destruction is via `DSPHandleRuntime`
state transition, not a separate `delete`).

### 3.2 CacheMap destruction ordering (Step 13 fix)

`AudioEngine.h:2106-2111`:
```cpp
const bool reclaimed = owner->tryShutdownQuiescentReclaim(entry.second);
if (reclaimed)
{
    const auto resolved = rt.resolve(entry.second);
    if (resolved.instance != nullptr)
        delete static_cast<EQCoeffCache*>(resolved.instance);
}
jassert(reclaimed);
```

Sequence: `tryShutdownQuiescentReclaim` (= Proof + Permit + identity + consume + retire
+ reclaim) → `resolve()` → `delete EQCoeffCache`. Reclaim failure → no physical
destruction. This is the **A2 Step 13 / H.11.11.9.4 blocker fix** described in
D101-33. The post-fix code comment at `AudioEngine.h:2101-2105` documents this:

> "旧実装は『delete EQCoeffCache → reclaim(slot)』の順序で、reclaim 失敗時に
> object 消滅 + handle 未回収の状態を作り得た。本実装は reclaim（slot 状態遷移）
> を先に成功させ、成功時のみ EQCoeffCache を物理解放する。"

### 3.3 **Documentation/comment consistency issue (D111-3 flag)**

The class-header comment for `~CacheMap` at `AudioEngine.h:2085` still reads:

> `//   Shutdown: resolve → delete → reclaim（全マップ同時破棄のため安全）。`

This describes the **old ordering** (resolve → delete → reclaim). The actual code
at `AudioEngine.h:2106-2111` follows the **new ordering** (reclaim → resolve →
delete). The header comment was not updated when the Step 13 fix landed.

**Classification: documentation/comment consistency issue, NOT a safety blocker.**
The implementation is correct; only the class-header comment is stale. D111
records this as **D111-DOC-001** (minor, non-blocking, follow-up cleanup).

### 3.4 Verdict: **PASS** (G23 ordering correct, all 3 paths verified; D111-DOC-001
recorded as a non-blocking follow-up).

---

## D111-4 — Permit / Identity Closure (G15–G22)

### 4.1 G15–G22 evidence (live source)

| Gate | Source evidence (live) | Verdict |
|---|---|---|
| G15 Proof immutability | `ISRLifetimeProof.h:76-116` — `ShutdownQuiescenceProof` move-only struct, copy deleted, fields are const after construction (lines 103, 388) | **PASS** |
| G16 Permit immutability | `ISRLifetimeProof.h:126-171` — `ReclaimPermit` move-only, copy deleted, `identity_` const after ctor (line 168) | **PASS** |
| G17 Proof private ctor | `ISRLifetimeProof.h:103` `explicit ShutdownQuiescenceProof(ShutdownRuntimeIdentity id)` private; constructed only at `ISRShutdown.cpp:387` (friend `ShutdownRuntime`) | **PASS** |
| G18 generation match | `ISRShutdown.cpp:376` `id.generation = consumeAtomic(shutdownGeneration_, acquire)`; `reclaimShutdownQuiescent` at `ISRRuntimePublicationCoordinator.cpp:759` checks via `permit.identity() == currentShutdownIdentity_` which includes generation | **PASS** |
| G19 epoch generation | `ISRShutdown.cpp:377` `id.epochGeneration = observation.epochGeneration`; check at line 759 | **PASS** |
| G20 readerReg generation | `ISRShutdown.cpp:378` `id.readerRegistrationGeneration = observation.readerRegistrationGeneration`; check at line 759 | **PASS** |
| G21 stale Permit reject | `ISRRuntimePublicationCoordinator.cpp:759` `permit.identity() == currentShutdownIdentity_` includes `engineInstanceId` (cross-runtime) + 3 generations (stale); rejection = `return false` | **PASS** |
| G22 forged Permit impossible | `ReclaimPermit` has deleted default ctor (`ISRLifetimeProof.h`); only `tryMakeReclaimPermit` constructs (single authority, `ShutdownRuntime` friend); `proof.valid()` required | **PASS** |

### 4.2 Identity-binding authority (Authority Singularization, Step 14)

`ShutdownRuntime::tryMakeQuiescenceProof` (line 385) calls
`reclaimAuthority_.bindShutdownIdentity(id)`. This is the **single bind authority**
— `AudioEngine::tryShutdownQuiescentReclaim` (line 4400) does **NOT** bind (it only
transports the Permit). `ReclaimAuthority` then validates
`permit.identity() == currentShutdownIdentity_` (line 759) inside `reclaimShutdownQuiescent`.

`setReclaimAuthority` is **deleted** (D110-5 verified, `AudioEngine.h:4397` comment:
"ReclaimAuthority の wiring も constructor 固定注入（setReclaimAuthority 廃止）").
Authority association is **immutable** (constructor-only injection), preventing
runtime reconfiguration.

### 4.3 Verdict: **PASS** (all 8 G15–G22 gates verified in live source)

---

## D111-5 — G23 Physical Destruction Closure (already covered in D111-3)

Reaffirmed for completeness:
- `permit.consume()` is the **only** destruction authorization point.
- `delete EQCoeffCache` is gated by `reclaimed == true` (line 2107) — reclaim failure → no physical destruction.
- `DSPHandleRuntime::reclaim(handle)` is the slot state transition (post-`retire`).
- `destroyDSPCoreNode(void*)` is invoked only at: `RebuildDispatch.cpp:910` (build rollback, no Permit — unpublished), `RebuildDispatch.cpp:979, 1057` (unpublished), `DSPLifetimeManager.cpp:50, 97, 123` (registration-time binding, no Permit — pre-publish). All `destroyDSPCoreNode` callers operate on **unpublished** or **pre-publish** DSPCore nodes (no A2 contract applies).

**Verdict: PASS.**

---

## D111-6 — Old API Eradication Audit (D111-4)

### 6.1 `reclaim(ReclaimMode, ..., bool)` — production caller count

| Source location | Classification | Production caller? |
|---|---|---|
| `src/tests/invariant_INV3_INV5.cpp:771` | Test source — comment documenting compile-guard success | No (test-only) |
| `src/audioengine/ISRDSPHandle.h:174` | Production header — comment referencing old API for migration context | No (comment only) |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:642` | Production header — comment: "⚠️ Step 9: 旧 bool reclaim API（reclaim(ReclaimMode, ..., bool)）は削除済み（AC-1: ...）" | No (comment only) |
| `src/tests/invariant_INV3_INV5.cpp:204` | Test source — function name `testInvX3_4ReclaimModeQuiescent` (test, not API) | No (test-only) |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:636` | Production header — comment describing design migration | No (comment only) |

**Production caller count: 0.** Compile-guard at
`ISRRuntimePublicationCoordinator.cpp:657`: `// ⚠️ dash2 §2.2 (Phase A2 — Step 9): 旧 bool reclaim API は削除（compile guard）。`
The compile-guard is **in effect** (no production symbol `reclaim(ReclaimMode, ...)` exists
in current source — confirmed by `rg -n 'reclaim\(ReclaimMode' src/` returning only
comments and test files).

### 6.2 `ReclaimMode` enum — production caller count

5 references found in source, all are:
- `src/tests/invariant_INV3_INV5.cpp:204, 763, 770-771, 870` — test-only (`testInvX3_4ReclaimModeQuiescent` and compile-guard verification comments).
- `src/audioengine/ISRDSPHandle.h:174` — production comment.
- `src/audioengine/ISRRuntimePublicationCoordinator.h:636, 642` — production comments.

**No `enum class ReclaimMode` declaration exists in any production header or source.**
The type was removed (compile-guard enforced). Test `testInvX3_4ReclaimModeQuiescent`
verifies this at compile time.

### 6.3 `readerRegistrationClosed` — caller-side shutdown 判断 0 件 (AC-2)

| Location | Role | Caller-side shutdown 判断? |
|---|---|---|
| `src/core/EpochDomain.h:634` `bool readerRegistrationClosed() const` | EpochDomain observer API (Q3 source) | No (observation source) |
| `src/audioengine/ISRShutdown.h:267` `bool readerRegistrationClosed{false}` | `QuiescenceObservation` struct field (Q3 value) | No (data field) |
| `src/audioengine/AudioEngine.h:4377` `obs.readerRegistrationClosed = m_epochDomain.readerRegistrationClosed()` | observation collection (Q3 read) | No (observation read) |
| `src/audioengine/ISRLifetimeProof.h:92` `bool readerRegistrationClosed() const` | `ShutdownQuiescenceProof` getter (Q3 read-back) | No (observation read-back) |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:667` and `.cpp:719` | documentation comments | No |

**Caller-side shutdown 判断: 0 件.** (D110 AC-2 satisfied.) The `bool readerRegistrationClosed`
is **observation only**; the actual `ShutdownQuiescenceProof` carries
`qReaderRegClosed_` as one of 8 Q conditions, and `reclaimShutdownQuiescent` uses
**Permit identity**, not the bool, to authorize destruction.

### 6.4 `shutdownReclaim` (ISRRetireRouter method) — orthogonal to A2 contract

The `ISRRetireRouter::shutdownReclaim(void*, deleter, epoch, type)` method at
`src/audioengine/ISRRetireRouter.cpp:574-584` is a **void* deferred-deletion ownership
transfer** to `TerminalReclaimAuthority` (epoch-gated destruction, no Permit).
It is invoked from `AudioEngine.h:4219` inside `enqueueDeferredDeleteNonRtWithResult`
during shutdown-mode ownership transfer. This is **not the DSPHandle reclaim path**;
it is a separate concept (P-4 TerminalReclaimAuthority for `void*` retirement).

**A2 contract does not apply to this path.** A2 covers **DSPHandle reclaim
post-shutdown quiescence** (Permit-gated). `shutdownReclaim` covers **arbitrary
`void*` deferred-delete entries** (epoch-gated, separate authority). The two
systems coexist by design.

**Verdict: PASS** (D111-4 satisfied; orthogonal `shutdownReclaim` (void* system)
documented for clarity, not classified as A2 regression).

---

## D111-7 — Build Results (D111-6)

### 7.1 Build configuration

| Setting | Value |
|---|---|
| Compiler | MSVC 19.51 (`cl.exe` 14.51.36231) |
| Build wrapper | `tools\build_with_vcvars.bat` (vcvars64.bat + IPP include) |
| Generator | Ninja Multi-Config |
| Configurations | Debug, Release |
| IPP | Enabled (Intel oneAPI IPP) |
| MKL | Enabled |
| PGO | OFF (default) |
| ASan | OFF (default) |
| Build directory | `build\` (created 2026-08-28 22:55) |
| Test executables produced | 36 (Debug) + 36 (Release) |

### 7.2 Debug build + CTest

| Item | Result | Evidence |
|---|---|---|
| Debug build | **PASS** | `build_release_D111.log` (Debug section); `ConvoPeq.exe` produced at `build\ConvoPeq_artefacts\Debug\ConvoPeq.exe` |
| Debug CTest | **40/40 PASS** | `evidence/D111_ctest_debug.log` |
| Total time | 34.67 sec | CTest footer |
| A2 tests (InvariantINV3INV5) | **PASS** (0.25 sec) | Test #22 |
| A2 tests (AdmissionPackedState) | **PASS** (0.52 sec) | Test #23 |
| A2 tests (ISRSemanticValidationRejects) | **PASS** (0.33 sec) | Test #21 |
| A2 tests (ISRSoakTests) | **PASS** (0.67 sec) | Test #6 |
| A2 tests (RetireGraceSemantics) | **PASS** (0.24 sec) | Test #24 |

### 7.3 Release build + CTest

| Item | Result | Evidence |
|---|---|---|
| Release build | **PASS** | `build_release_D111.log` (Release section); `ConvoPeq.exe` produced at `build\ConvoPeq_artefacts\Release\ConvoPeq.exe` |
| Release CTest | **40/40 PASS** | `evidence/D111_ctest_release.log` |
| Total time | 26.63 sec | CTest footer |
| A2 tests (InvariantINV3INV5) | **PASS** (0.04 sec) | Test #22 |
| A2 tests (AdmissionPackedState) | **PASS** (0.20 sec) | Test #23 |
| A2 tests (ISRSemanticValidationRejects) | **PASS** (0.06 sec) | Test #21 |
| A2 tests (ISRSoakTests) | **PASS** (not shown in last 60 lines; passed in full run) | Test #6 |

### 7.4 Result matrix

| Configuration | Build | Tests | Result |
|---|---|---|---|
| Debug | PASS | 40/40 | **PASS** |
| Release | PASS | 40/40 | **PASS** |

All A2-related tests passed in both configurations. No pre-existing, environmental,
or A2-unrelated test failures observed.

### 7.5 A2 test inventory (live)

| Test name | File | Verifies | Status |
|---|---|---|---|
| `testInv3_1RetireEpochSafeReclaim` | `src/tests/invariant_INV3_INV5.cpp:131` | `requestReclaim` retire → epoch safe → reclaim 順序 | PASS |
| `testInv3_2ReclaimDeferredThenSucceeds` | `:168` | epoch 非安全 → pending 再登録 | PASS |
| `testInvX3_4ReaderRegistrationClosed` | `:447` | Q3 / readerRegistrationClosed | PASS |
| `testInvX3_4ReclaimModeQuiescent` | `:204` | ShutdownQuiescent reclaim precondition (旧 bool API compile-guard) | PASS |
| `testA2G19G20EpochGenerationSupply` | `:506` | G19/G20 epoch/readerReg generation 供給 | PASS |
| `testA2G10G13G21G22ProofPermit` | `:586` | G10/G13/G21/G22 Proof/Permit acceptance | PASS |
| `testA2Step14ConcurrentDoubleReclaim` | `:797` | T9 concurrent double consume → 1 success | PASS |
| `testA2Step14PermitABA` | `:586` | T10 Permit ABA (shutdown N → N+1 reject) | PASS |
| `testA2Step14SetterResurrection` | `:768` | T11 setReclaimAuthority 廃止・immutable binding | PASS |
| `testA2Step14DestructionOrderingAudit` | `:782` | T13 reclaim 成功後のみ physical destruction (CacheMap) | PASS (documented) |

All 10 A2-step / INV-3 / INV-X3-4 test cases pass.

### 7.6 `AdmissionPackedState` test inventory (G-H linearization, D101-31-B)

| Test name | File:line | Verifies | Status |
|---|---|---|---|
| `testOpenStateAdmit` | `src/tests/AdmissionPackedStateTests.cpp:52` | Open admit + outstanding++ | PASS |
| `testClosedRejects` | `:75` | Closed/Faulted reject | PASS |
| `testCloseAdmissionSetsState` | `:99` | Open → Closing → Closed | PASS |
| `testCloseAdmissionJoinsProducers` | `:159` | outstanding==0 → joinProducers OK | PASS |
| `testConcurrentTryAdmitCloseAdmission` | `:271` | G-H linearization (concurrent) | PASS |
| `testPostCloseAdmissionImpossible` | `:289` | post-Close reject | PASS |
| `testOutstandingRoundtrip` | `:314` | admit / release counter consistency | PASS |
| `testAdmitAfterRestart` | `:321` | post-Close re-admit impossible | PASS |

All 8 G-H linearization tests pass.

---

## D111-8 — Test Results Summary

**D111-8.1 Debug CTest: 40/40 PASS** (100%, 34.67 sec)
**D111-8.2 Release CTest: 40/40 PASS** (100%, 26.63 sec)

D110 hand-off expectation ("40/40 should still pass") **empirically met**. A2
implementation did not regress any test, and the new A2 tests
(`InvariantINV3INV5` test #22 in particular, covering G19/G20, G10/G13/G21/G22,
Step14 T9/T10/T11/T13) all pass under both Debug and Release.

---

## D111-9 — Final Regression Grep

### 9.1 Reclaim authority (production source)

| Token | Sites | Production caller count |
|---|---|---|
| `requestReclaim` | 60 (all comments + 1 production symbol at `ISRRuntimePublicationCoordinator.cpp:668`) | 1 production symbol; **delegates to `reclaimNormal` (line 675)** |
| `reclaimNormal` | 4 sites (1 symbol at `ISRRuntimePublicationCoordinator.cpp:684`, 3 comments) | 1 production symbol; **called by `requestReclaim` only** |
| `reclaimShutdownQuiescent` | 2 sites (1 symbol at `ISRRuntimePublicationCoordinator.cpp:743`, 1 call at `AudioEngine.h:4400`) | 1 production symbol; **called by `tryShutdownQuiescentReclaim` only** |
| `tryShutdownQuiescentReclaim` | 5 sites (1 symbol at `AudioEngine.h:4367`, 3 production callers, 1 call from definition) | **3 production callers** (D109/D110 confirmed) |

### 9.2 Destruction

| Token | Sites | Notes |
|---|---|---|
| `delete DSPCore` | 0 (no direct `delete DSPCore`; all via `destroyDSPCoreNode` or `delete EQCoeffCache`) | — |
| `delete EQCoeffCache` | 1 site (`AudioEngine.h:2111`) | gated by `reclaimed == true` and `resolved.instance != nullptr` |
| `destroyDSPCoreNode` | 5 sites (1 symbol + 4 callers: `RebuildDispatch.cpp:910, 979, 1057`; `DSPLifetimeManager.cpp:50, 97, 123`) | all callers operate on **unpublished** or **pre-publish** DSPCore; A2 contract does not apply |

### 9.3 Admission

| Token | Sites | Notes |
|---|---|---|
| `tryAdmit(1)` (production) | 4 sites: `RuntimePublicationOrchestrator.cpp:69` (Path A), `AudioEngine.RebuildDispatch.cpp:319` (Build), `AudioEngine.h:4444` (Path B), `AudioEngine.h:4542` (Recovery) | All 4 producer gates; D108 §G06–G09 / D110 §G07–G09 confirmed |
| `tryAdmit(1)` (test) | 8 sites in `src/tests/AdmissionPackedStateTests.cpp` | test-only |
| `closeAdmission` (production) | 1 site: `AudioEngine.Processing.ReleaseResources.cpp:87` | sole authority |
| `closeAdmission` (test) | 9 sites in `src/tests/invariant_INV3_INV5.cpp` | test-only |

### 9.4 Permit

| Token | Sites | Notes |
|---|---|---|
| `ReclaimPermit` (production) | 2 symbols (`ISRLifetimeProof.h:126`, `ISRRuntimePublicationCoordinator.cpp:747`); 5 references (construction, consume, identity) | single authority: `ShutdownRuntime::tryMakeReclaimPermit` |
| `permit.consume` (production) | 1 site: `ISRRuntimePublicationCoordinator.cpp:761` | single-use CAS |

### 9.5 D111-9 verdict

**No new reclaim authority / destruction / admission / permit path introduced
after D110.** All grep counts match D110-5 / D109-7 / D108-6 evidence. **A2
contract closure is empirically complete.**

---

## D111-10 — G01–G23 Final Matrix (Live Source + Live Build/Test)

| Gate | I4 requirement | Live source evidence | Verdict | Blocking? |
|---|---|---|---|---|
| G01 | external setter production caller = 0 | `setRetireBacklogCount`, `setPublicationBacklogCount`, `setPendingIntentCount` in `ISRRuntimePublicationCoordinator.h:147-149` (TEST-ONLY); 0 production callers (D110-1 verified, ISRSemanticValidationTests.cpp:326-437 are test-only) | **PASS** (T1 I4 contract met; header-cosmetic PARTIAL is non-blocking) | NO |
| G02 | counter mutation = single authority | All active production counters single-authority (D101-32-C/D) | **PASS** | NO |
| G03 | `isFullyDrained()` observational | D107 verified (16 conditions, read-only) | **PASS** | NO |
| G04 | `swapPending_` pre-check preserved | `ISRRuntimePublicationCoordinator.cpp:489, 507`; architectural separation from `packedState_` intentional (D110-2) | **PASS** | NO |
| G05 | Path A shutdown gate | `RuntimePublicationOrchestrator.cpp:69` `tryAdmit(1)` | **PASS** | NO |
| G06 | Path B admission linearization | `AudioEngine.h:4444` `tryAdmit(1)`; CAS on same `packedState_` as `closeAdmission()` (D101-31-B) | **PASS** | NO |
| G07 | Recovery enqueue gate | `AudioEngine.h:4542` `tryAdmit(1)` | **PASS** | NO |
| G08 | Build admission gate | `RebuildDispatch.cpp:319` `tryAdmit(1)` | **PASS** | NO |
| G09 | Publish gate | `RuntimePublicationOrchestrator.cpp:69` + `AudioEngine.h:4444` (Path A + B) | **PASS** | NO |
| G10 | Proof acceptance | D106 verified (Q0–Q7) | **PASS** | NO |
| G11 | Permit acceptance | D106 verified (proof.valid required) | **PASS** | NO |
| G12 | stale Proof reject | identity check at consume (line 759) | **PASS** | NO |
| G13 | Permit private ctor | `ReclaimPermit` deleted default ctor, friend `ShutdownRuntime` | **PASS** | NO |
| G14 | pending reclaim identity | `ReclaimIdentity{handle, retireSequence}` in source | **PASS** | NO |
| G15 | Proof immutability | move-only, copy deleted | **PASS** | NO |
| G16 | Permit immutability | move-only, copy deleted | **PASS** | NO |
| G17 | Proof private ctor | private member of `ShutdownRuntime` | **PASS** | NO |
| G18 | generation match | `ISRRuntimePublicationCoordinator.cpp:759` check | **PASS** | NO |
| G19 | epochGeneration match | line 759 check | **PASS** | NO |
| G20 | readerRegGeneration match | line 759 check | **PASS** | NO |
| G21 | stale Permit reject (engineInstanceId) | line 759 check (engineInstanceId + 3 generations) | **PASS** | NO |
| G22 | forged Permit compile error | `ReclaimPermit` deleted default ctor | **PASS** | NO |
| G23 | physical delete after `permit.consume()` | `reclaimShutdownQuiescent` line 761 consume → 765 retire → 771 reclaim; CacheMap dtor line 2106-2111: reclaim → resolve → delete | **PASS** | NO |

**All 23 gates PASS.** G01 has a cosmetic PARTIAL on header existence (3 test-only
setters in header), but T1 I4 contract is met (production caller = 0). Test
changes (G01 header cleanup) are explicitly **NOT required** for A2 closure (D110).

### 10.1 Live test corroboration

| Test category | Debug | Release |
|---|---|---|
| A2 contract tests (InvariantINV3INV5: G19/G20/G10/G13/G21/G22 + T9/T10/T11/T13) | PASS | PASS |
| G-H linearization (AdmissionPackedState: 8 tests) | PASS | PASS |
| Semantic validation (ISRSemanticValidationRejects) | PASS | PASS |
| Soak (ISRSoakTests) | PASS | PASS |
| Retirement semantics (RetireGraceSemantics) | PASS | PASS |
| Drain semantics (ShutdownRetireIntentDrain, StuckReaderFallbackDrain) | PASS | PASS |
| All 40 tests | PASS | PASS |

---

## D111-11 — Final Verdict

### D111 verdict: **Case A — A2 IMPLEMENTATION CLOSED / IMPLEMENTATION ACCEPTED**

**Rationale:**

1. **Static A2 path: PASS** (3 production callers of `tryShutdownQuiescentReclaim`;
   `requestReclaim → reclaimNormal` split intact; no cross-leak to `reclaimShutdownQuiescent`).
2. **G01–G23 contract: PASS** (all 23 gates verified in live source; G01 PARTIAL is
   header-cosmetic, non-blocking per D110-1).
3. **3 production callers: PASS** (ReleaseResources activeHandle:457, fadingHandle:464,
   CacheMap:2106).
4. **Old reclaim API: production caller = 0** (compile-guard enforced;
   `reclaim(ReclaimMode, ..., bool)` symbol not present in any production TU).
5. **Permit ordering: PASS** (Proof → Permit → identity → consume → retire → reclaim →
   delete; verified in `reclaimShutdownQuiescent` body and CacheMap dtor).
6. **Identity validation: PASS** (engineInstanceId + 3 generations; G21 stale reject
   confirmed; G22 forged Permit compile error).
7. **Physical destruction: PASS** (no `delete` before `permit.consume()`; CacheMap
   gated by `reclaimed == true`).
8. **Debug build: PASS** (40/40 tests, 34.67 sec).
9. **Release build: PASS** (40/40 tests, 26.63 sec).
10. **A2 tests: PASS** (10 A2-step / INV-3 / INV-X3-4 cases in `InvariantINV3INV5`).
11. **Regression tests: PASS** (no pre-existing, environmental, or A2-unrelated
    test failures).

### A2 implementation: **CLOSED**

The A2 production reclaim path is **operationally validated** (live Debug + Release
builds, 100% CTest pass rate, 0 regressions). D110 Case-A GO is **empirically
reaffirmed** by D111 build+test closure.

### Required changes before next phase: **0**

The implementation is complete. The only outstanding follow-up is **D111-DOC-001**
(class-header comment for `~CacheMap` at `AudioEngine.h:2085` is stale; describes
old `resolve → delete → reclaim` ordering instead of the Step-13 fixed
`reclaim → resolve → delete`). This is a **cosmetic documentation issue** and
**NOT blocking A2 closure** — it is recorded as a follow-up cleanup item.

### Open follow-ups (non-blocking)

| ID | Description | Severity | Action |
|---|---|---|---|
| **D111-DOC-001** | `~CacheMap` class-header comment at `AudioEngine.h:2085` still describes old `resolve → delete → reclaim` ordering; the actual code (lines 2106-2111) follows Step-13 fixed `reclaim → resolve → delete`. | Documentation / cosmetic | Optional cleanup; update comment to match implementation. **Not blocking A2 closure.** |
| D111-OPT-001 (D110 cosmetic) | G01 3 test-only setters in `ISRRuntimePublicationCoordinator.h:147-149` can be moved to a test-only header or annotated with `#ifdef CONVOPEQ_TEST_BUILD`. | Cosmetic | Optional cleanup; **NOT required by I4** (T1 satisfied). |
| D111-OPT-002 (D110 cosmetic) | I4 §D37.2 can be amended to explicitly mention the two-path model. | Documentation | Optional; **NOT required for A2 closure** (current I4:2439 already describes `requestReclaim → reclaimNormal` literally). |

### D111 → D112 hand-off

**D111 verdict: Case A (A2 IMPLEMENTATION CLOSED).**

A2 reclaim is operationally accepted. D112 (if any) can proceed to:
1. Apply D111-DOC-001 (comment cleanup) as a single trivial change.
2. Move to next-phase work (whatever follows A2 in the work88 plan).

No further read-only audit is required for A2 itself.

---

## File references (read-only, no source modification)

| File | Role |
|---|---|
| `ConvoPeq.md` (2026-08-28 21:22, 4,633,246 B) | runtime source baseline (read-only verification anchor) |
| `src/audioengine/AudioEngine.h:4367-4402` | `tryShutdownQuiescentReclaim` (3 production callers) |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:457, 464` | 2/3 production callers (graceful drain) |
| `src/audioengine/AudioEngine.h:2106-2111` | 1/3 production caller (CacheMap destroy) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:668-676` | `requestReclaim` → `reclaimNormal` delegation |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:684-713` | `reclaimNormal` body (NORMAL EBR, no Permit) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:743-773` | `reclaimShutdownQuiescent` body (Permit consume) |
| `src/audioengine/ISRShutdown.cpp:351-398` | `tryMakeQuiescenceProof` (Q0–Q7 + identity binding) |
| `src/audioengine/ISRShutdown.cpp:400-412` | `tryMakeReclaimPermit` (Proof.identity → Permit) |
| `src/audioengine/ISRLifetimeProof.h:76-171` | `ShutdownQuiescenceProof` + `ReclaimPermit` (move-only, single authority) |
| `src/tests/invariant_INV3_INV5.cpp:131-844` | A2 test cases (10 functions) |
| `src/tests/AdmissionPackedStateTests.cpp` | G-H linearization tests (8 functions) |
| `build/ConvoPeq_artefacts/Debug/ConvoPeq.exe` | Debug build output (PASS) |
| `build/ConvoPeq_artefacts/Release/ConvoPeq.exe` | Release build output (PASS) |
| `evidence/D111_ctest_debug.log` | Debug CTest 40/40 PASS |
| `evidence/D111_ctest_release.log` | Release CTest 40/40 PASS |
| `build_release_D111.log` | Full Debug + Release build log |
| `evidence/D108_A2_GATE_INVENTORY.md` | D108 G01–G23 inventory (corrected by D109/D110) |
| `evidence/D109_A2_RECLAIM_PATH_WIRING.md` | D109 Case B (3 callers verified) |
| `evidence/D110_A2_FINAL_CONTRACT_AUTHORIZATION.md` | D110 Case A (D111 closure baseline) |
| `doc/work88/I4_DESIGN_CONTRACT.md:2439, 2443, 2527` | I4 reclaim contract (literal two-path) |
| `doc/work88/D2_IMPL_CHECKLIST.md:84` | A2 reclaim wiring confirmation ("Step 9-14 完了") |

**No source files modified. No I4 files modified. No tests added.** D111 is a
read-only implementation closure audit: static source verification + live build +
live CTest.
