# D135-8/9 Gate B — Compile Audit (Debug + Release)

Date: 2026-08-30
Scope: post-Gate-A working tree (commit a65ace1 + 15 modified audioengine files).
Constraints honored: **production source 0 changes, test source 0 changes, CTest NOT run (deferred to Gate C).**

## Verdict: PASS — proceed to Gate C

| Config | Result | cmake exit | ConvoPeq.exe | Log |
| --- | --- | --- | --- | --- |
| Debug | PASS | 0 | 2026-08-30 15:05:59 (72,784,896 B) | `evidence/D135-8-9_GATE_B_DEBUG_BUILD_LOG.txt` |
| Release | PASS | 0 | 2026-08-30 15:20:15 (47,779,840 B) | `evidence/D135-8-9_GATE_B_RELEASE_BUILD_LOG.txt` |

Zero compile/link errors in both logs. Debug peak progress [521/522] (no FAILED lines),
Release [546/553]→complete (no FAILED lines). All test targets linked in both configs,
including `RuntimeHealthMonitorTierTests.exe`, `AudioEngineHarness.exe`, `DeferredDeletionQueueReclaimTests.exe`.

## Build environment (provenance)

- Tree: `build/` = **Ninja Multi-Config**, `CMAKE_C/CXX_COMPILER=cl`, pre-configured — no reconfigure needed.
- Toolchain: VS 2026 Enterprise v18.9.2 → `vcvarsall.bat x64` → `cl.exe` 14.51.36231 (Hostx64/x64), `cmake 4.4.3`, ninja 1.13.2.
- **Environment requirement (non-obvious)**: `CL=/I "C:\Program Files (x86)\Intel\oneAPI\2026.1\include"`
  so that `mkl.h` (DiagnosticsConfig.h:49, gated by `JUCE_DSP_USE_INTEL_MKL=1`; `CONVOPEQ_REQUIRE_MKL:BOOL=ON`
  in build/CMakeCache) resolves. This mirrors the project's own `tools/build_debug_with_vcvars.bat`.
  Note: the `ConvoPeq` target gets MKL includes from CMake itself (`-external:I"...mkl\2026.1\include"`),
  but some targets rely on the CL env include. The path MUST stay quoted inside `CL` — cl.exe splits
  unquoted spaces into phantom source-file tokens (C1083 'Files' / '(x86)\Intel\...').
- env provenance: `evidence/D135-8-9_GATE_B_VCVARSALL_diag.txt` (vcvarsall RC=0, cl found, cmake found).

## Failed attempts (recorded honestly)

1. `cmd.exe /c '...'` invocation from the agent shell is mangled (command after `/c` never reaches cmd.exe —
   interactive banner instead). Fix: run the helper .bat via `powershell.exe -NoProfile -Command "& '...bat'"`.
2. First real build failed: `C1083 'mkl.h'` at DiagnosticsConfig.h(49) — missing CL env include.
   **No D135-8/9 source errored**; the only failure was the health-monitor MKL path (environmental).
3. Second attempt failed: `CL` set with an unquoted spaced path → cl parsed `Files`/`(x86)\...` as source
   files. Fixed with the quoted form above.
4. First .bat used `echo ----- BUILD LOG TAIL (for quick triage) -----` inside an `else (` block — the
   unescaped `)` closed the block early and ` -----` became a phantom command (`の使い方が誤っています`),
   hiding the OK/FAIL markers. Wrapper rewritten without parenthesized echo text.
   A background run was also killed at its timeout leaving the log truncated at [521/522]; the re-run
   completed cleanly.

## The 5 Gate-B verification items — all PASS

1. **Header declarations** (`RuntimePublicationOrchestrator.h`): `requestDeferredClear()` :204,
   `drainDeferredClearIfRequested()` :205, `std::atomic<bool> deferredClearRequested_{false}` :296,
   `resetDeferredRetryBudget()` :171 (inline, carries rebuild-thread jassert),
   `processDeferredAdmission(bool wasRecoveryWake)` :194. ✓
2. **C1 fallback + RebuildThread jassert** (`RuntimePublicationOrchestrator.cpp:568-597`):
   `requestDeferredClear()` checks `rebuildThreadShouldExit` (acquire) → synchronous
   `clearDeferredForShutdown()`; live path latches + `rebuildCV.notify_one()`.
   `drainDeferredClearIfRequested()` carries `jassert(this_thread::get_id() == engine_.rebuildThreadId())`. ✓
3. **publishAtomic/exchangeAtomic API fit** (`AtomicAccess.h:52-73`): latch access is exclusively
   `convo::publishAtomic(deferredClearRequested_, true, release)` (cpp:580) and
   `convo::exchangeAtomic(deferredClearRequested_, false, acq_rel)` (cpp:591); grep confirms **zero raw
   `.store/.load/.exchange`** on the latch. All other latch-family sites (hasDeferred_, lastRecoveryPublishSeq_)
   likewise use the convo API. ✓
4. **RebuildDispatch.cpp wiring**: `drainDeferredClearIfRequested()` call at :906 (outside lock, before the
   deferred-publish handoff); `processDeferredAdmission(wasRecoveryWake)` at :920 with `wasRecoveryWake` a
   bool declared :848 and stamped from `convo::exchangeAtomic(recoveryRetryReady, false, acq_rel)` :888. ✓
5. **Timer.cpp producer sites**: exactly 3 `requestDeferredClear()` calls (:1642 publication-stall drain,
   :1809, :1829 recovery paths); **no** `resetDeferredRetryBudget()` call in Timer.cpp — the old blind
   producer-side reset is removed (comment :1724-1725) and the reset now lives rebuild-thread-side inside
   `processDeferredAdmission` (cpp:688), gated on the P3 provenance flag. ✓

## Artifacts created by this gate (build helpers only — not source/test)

- `tools/gateb_build.bat` — config-parameterized compile wrapper (vcvarsall x64 + CL oneAPI include +
  `cmake --build build --config <cfg>`, log to evidence/, prints `GATB_CMAKE_EXIT=<rc>`).
- `tools/gateb_vcdiag.bat` — vcvarsall/cl/cmake environment diagnostic (evidence capture).
- Build logs (above) + `evidence/D135-8-9_GATE_B_VCVARSALL_diag.txt`.
