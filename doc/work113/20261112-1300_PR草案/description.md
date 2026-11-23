fix(isr): restore the lint, atomic-access and proof-scope gates

## Background

The domain tag commit `173e6995` turned three CI checks red on `main`:

- `audioengine-lint` — LINT-AE-011 ×2 (`mutable` in `ISRDebugRuntime.h`, `thread_local` in `ISRRetireLatency.h`)
- `list-compliance` — 258 strict atomic dot-calls across 7 files
- `ISR Verification` (scheduled) — `isr-verify-proof-scope.ps1` rejected the B-11 `emitCIArtifacts`/`emitHBTrace` control flow

This branch fixes those three gates. Nothing else changes: the domain tag implementation, `RetireDomain` stamping, per-domain aggregation, thresholds and pass/fail decisions are untouched.

## Changes

- **LINT-AE-011**
  - drop the unnecessary `mutable` on `HBVerifierRuntime hbVerifier_` (only the non-const `emitFinalViolationReport()` and `emitCIArtifacts()` touch it)
  - keep the thread-local range assignment, with a `NOLINT(thread-local)` + `RT-SAFE:` justification (constant-initialized POD, no dynamic init/alloc/lock; MSVC `dumpbin` confirms no TLS guard, `.tls$` is 4 bytes)
- **Atomic access**
  - route 270 atomic operations through `convo::consumeAtomic` / `publishAtomic` / `fetchAddAtomic` / `compareExchangeAtomic`
  - add `compareExchangeWeakAtomic` so weak CAS keeps its spurious-failure semantics, and widen the CAS helpers to accept a convertible `desired`
  - the 23 counters in `RetireLatencyRecorder::reset()` move from the implicit `seq_cst` default to `release`; `reset()` runs only in a quiescent state (engine constructor, before `initialize()` and before the audio callback is registered, plus test setup). The call sites, lifecycle and maintenance conditions are in `reset_exclusivity.md` (delivered with the review evidence).
- **Proof scope**
  - rewrite `isr-verify-proof-scope.ps1` to check `emitCIArtifacts()` / `emitHBTrace()` per function (Release early return, force-evidence exception, emit position, its own suppression) and the A(D1) call **inside** `emitFinalViolationReport()`'s body
  - lexer that understands comments, normal/raw/prefixed-raw strings, character literals, escapes and LF/CRLF line continuations
  - fixture self-test (20 cases) and a lexer boundary test (38 checks, including three mutation controls)

## Verification

| Check | Result |
| --- | --- |
| `check-audioengine-lint.ps1` | PASS (LINT-AE-011 = 0) |
| `check-src-atomic-dotcall.ps1` | PASS (0 violations) |
| `isr-verify-proof-scope.ps1` | PASS |
| proof-scope self-test | 20/20 |
| lexer boundary test | 38/38 |
| `isr-run-tiered-verification.ps1 -Tier standard` | PASS (completed) |
| Release ctest | 57/57 |
| RelWithDebInfo ASan ctest | 57/57 |
| Debug ctest | 55/57 (the two known failures: `D8_2_B_2_Tests` stack overflow, `TailLayerIntegrityTests` RES-F budget) |
| TSan (WSL, clang 21.1.8) | 3 of 5 instrumented targets ran with 0 warnings; `ScopedEpochReaderTests` is blocked by a pre-existing Clang incompatibility (`ISRRetireLatency.h:1135`, `EpochDomain.h:767`, both unchanged by this branch), `ISRSemanticValidationTests` needs MKL on Linux |

This PR is what triggers `audioengine-lint`, `list-compliance`, `ISR Verification`, `Windows MSVC Build` and `Sanitizer CI` for the branch (a feature-branch push does not).

## Known limitations

- Remote CI has not run yet (no PR existed before this one).
- TSan coverage is partial; three targets without warnings are not a proof of thread safety for the whole program.
- The two Debug failures predate this branch and are out of scope.
