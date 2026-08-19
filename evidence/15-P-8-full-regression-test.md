# 15-P-8: Full Regression Test Suite

## Status: PASS

## Objective

Run the full regression test suite after the 15-P-7 Stuck-Reader Fallback
Regression Test was added, verifying:

- No regression in existing tests (Debug / Release)
- New tests (`StuckReaderFallbackDrain`, `ShutdownRetireIntentDrain`) are
  correctly integrated into the normal Debug / Release / CTest pipeline
- No production code changes were made during 15-P-8

## Scope

| Item | Result |
| --- | --- |
| Build (Debug) | PASS |
| Build (Release) | PASS |
| CTest (Debug) | 33/33 PASS |
| CTest (Release) | 33/33 PASS |
| Shutdown/Retire regression | PASS |
| StuckReaderFallbackDrain | PASS |
| ASan | NOT RUN / BLOCKED (see below) |
| Production code changes | None |

## Build

Both Debug and Release configurations build cleanly with the standard
toolchain (MSVC `cl` + Intel oneAPI, Ninja Multi-Config).

```text
cmake --build build --config Debug
cmake --build build --config Release
```

Result: PASS (both configurations).

## CTest

### Debug

```text
100% tests passed out of 33
Total Test time (real) = 36.76 sec
```

All 33 tests pass in Debug.

### Release

```text
100% tests passed out of 33
Total Test time (real) = 32.70 sec
```

All 33 tests pass in Release.

## Test Count Comparison

The test count in `CMakeLists.txt` was compared before / after the 15-P-7
commit (`a07e994`):

| Metric | Before (HEAD~1) | After (HEAD) |
| --- | --- | --- |
| `add_test(NAME ...)` count | 31 | 33 |

New tests registered in CTest:

```text
ShutdownRetireIntentDrain
StuckReaderFallbackDrain
```

No existing tests were removed or renamed.

## Shutdown / Retire Regression

The shutdown / retire ownership tests pass in both configurations:

| Test | Debug | Release |
| --- | --- | --- |
| `RetireGraceSemantics` (#17) | PASS | PASS |
| `ShutdownRetireIntentDrain` (#18) | PASS | PASS |
| `PriorityIntegration` (#31) | PASS | PASS |

This confirms the 15-P-4-5-FIX (`drainPendingRetireIntentsForShutdown`) and
the System 1 / System 2 ownership shutdown paths remain intact.

## StuckReaderFallbackDrain

The 15-P-7 regression test passes in both configurations:

| Test | Debug | Release |
| --- | --- | --- |
| `StuckReaderFallbackDrain` (#19) | PASS | PASS |

This confirms the 15-P-5 fix (`drainAllQuarantineStore()` in the stuck-reader
fallback path) remains intact — Q + E + Terminal are all drained during
shutdown even when a reader is stuck.

## ASan

### Status: NOT RUN / BLOCKED

The ASan build procedure is documented in `BUILD_GUIDE_WINDOWS.md` section 9
(RelWithDebInfo config + `clang_rt.asan_dynamic-x86_64.dll` + `ASAN_OPTIONS=detect_leaks=0`).
The documented procedure was followed exactly:

```text
cmake --build build-asan-msvc --config RelWithDebInfo --target StuckReaderFallbackDrainTests
copy clang_rt.asan_dynamic-x86_64.dll <exe dir>
ASAN_OPTIONS=detect_leaks=0 StuckReaderFallbackDrainTests.exe
```

The ASan build **succeeded** (RelWithDebInfo, `/MD` + `/fsanitize=address`),
but the test **fails to start** with exit code `0xC0000139`
(`STATUS_ENTRYPOINT_NOT_FOUND`).

Diagnostics performed:

| Check | Result |
| --- | --- |
| ASan runtime DLL version vs compiler | Match (19.51.36256.0) |
| ASan DLL standalone load (`LoadLibrary`) | SUCCESS |
| ASan DLL exports (`__asan_init`, 683 total) | Present |
| ASan DLL dependencies | All standard (VCRUNTIME140, dbghelp, api-ms-win-*) |
| Exe imports (`clang_rt.asan_dynamic-x86_64.dll`) | Present |
| Debug config (`/MDd` + release DLL mismatch) | Also fails `0xC0000139` |
| RelWithDebInfo config (documented flow) | Also fails `0xC0000139` |

**Conclusion**: This is an environment / runtime issue, not a test failure.
The test itself passes in normal Debug / Release builds. The ASan runtime
cannot be executed in this environment despite following the documented
procedure. Per the 15-P-8 instruction ("ASan の実行方法が通常のプロジェクト手順として
確立していない場合、勝手に新しい環境構築を始めず NOT RUN / BLOCKED と記録"),
ASan is recorded as **NOT RUN / BLOCKED**.

## Production Code Changes

15-P-8 made **no production code changes**. The working tree shows no
uncommitted changes under `src/`. The only working-tree modifications are:

- `ConvoPeq.md` — auto-generated project extract (timestamp + new test file
  listing only)
- `tools/*.bat`, `tools/*.ps1` — temporary build/test scripts (cleaned up
  after this evidence is recorded)

## Verdict

**PASS** — Full regression suite passes 33/33 in both Debug and Release.
The new `StuckReaderFallbackDrain` and `ShutdownRetireIntentDrain` tests are
correctly integrated into the normal CTest pipeline. No existing test
regressed. No production code was changed during 15-P-8.

ASan execution is **NOT RUN / BLOCKED** due to a runtime DLL loading issue
(`0xC0000139`) in this environment, independent of the test code.
