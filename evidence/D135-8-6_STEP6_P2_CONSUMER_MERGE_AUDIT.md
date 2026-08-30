# D135-8 — Step 6 (P2) Consumer Merge Audit (Read-Only Verification)

**Status:** READ-ONLY verification — no build / no test
**Date:** 2026-08-30
**Spec locked by:** `evidence/D135-8_IMPLEMENTATION_PREFLIGHT.md` §8 (P2: consumer-side provenance consume + wake-provenance handoff)
**Precondition:** Step 4 (P1-B) PASS — `recoveryRetryReady` writer in `AudioEngine.Timer.cpp` recovery handler; Step 5 PASS — blind `resetDeferredRetryBudget()` removed from producer.
**Scope:** One production file — `src/audioengine/AudioEngine.RebuildDispatch.cpp`. Two change spots (deferred-publish merge + `processDeferredAdmission` call). Signature/def reset **not** touched (intentional compile-blocked intermediate state — see §5).

---

## 0. Edits applied

| # | File:Loc | Before | After |
|---|----------|--------|-------|
| 1 | `AudioEngine.RebuildDispatch.cpp:843` | *(no `wasRecoveryWake`)* | `bool wasRecoveryWake = false;` declared in outer loop scope (next to `doDeferredPublish`) |
| 2 | `AudioEngine.RebuildDispatch.cpp:888` | *(no recovery consume)* | `wasRecoveryWake = convo::exchangeAtomic(recoveryRetryReady, false, std::memory_order_acq_rel);` inserted **before** `doDeferredPublish = publishRetryReady;` |
| 3 | `AudioEngine.RebuildDispatch.cpp:914` | `runtimeOrchestrator_->processDeferredAdmission();` | `runtimeOrchestrator_->processDeferredAdmission(wasRecoveryWake);` |

Also restored a self-introduced typo in the pre-existing `doDeferredPublish` comment (`ハンドオフ`, was erroneously `ハンドオff` in an intermediate edit) — verified back to original via direct Read at `:842`.

## 1. API verification (consume vs exchange — critical)

The Step 6 draft snippet used `convo::consumeAtomic(recoveryRetryReady, false, std::memory_order_acq_rel)`. **Verified against latest source** (`src/audioengine/AtomicAccess.h`):

- `consumeAtomic` (`:60`) is a **2-arg load-only** overload — `T consumeAtomic(const std::atomic<T>& src, std::memory_order order = acquire)` — backed by `std::atomic_load_explicit`. **No 3-arg overload exists.** The proposed call would not compile.
- The project's **read-and-clear** primitive is `exchangeAtomic` (`:68`):
  ```cpp
  template <typename T, typename U, typename = enable_if<is_convertible_v<U,T>>>
  inline T exchangeAtomic(std::atomic<T>& dst, U&& value,
                          std::memory_order order = std::memory_order_acq_rel) noexcept
  { return std::atomic_exchange_explicit(&dst, static_cast<T>(std::forward<U>(value)), order); }
  ```
  Returns the **old** value, writes `false`. This is exactly the Step 0–Step 4 specified intent:
  *"Preflight で確定している意図は `recoveryRetryReady.exchange(false, std::memory_order_acq_rel)` 相当の atomic read-and-clear"* and *"プロジェクトの atomic wrapper が exchange をどのように提供しているかを確認し、既存 API に合わせてください。"*
- `exchangeAtomicPtr` (`:142`) delegates to `exchangeAtomic` — the same idiom is already idiomatic in the codebase for read-and-clear.
- `recoveryRetryReady` is declared `std::atomic<bool>` (non-const lvalue member), so `exchangeAtomic` (which takes a non-const `std::atomic<T>&`) is applicable.

**Decision:** `convo::exchangeAtomic(recoveryRetryReady, false, std::memory_order_acq_rel)` — matching the existing API and the user's verified intent. Deviation from the illustrative snippet is the *name* (`exchangeAtomic` not `consumeAtomic`), forced by the API verification the user explicitly requested.

## 2. `wasRecoveryWake` const scoping note

The snippet declared `const bool wasRecoveryWake = ...` at the merge block (inside the `rebuildMutex`-locked scope, lines 849–900). The `processDeferredAdmission(wasRecoveryWake)` call is at `:914`, **outside** that scope (lock releases at the `}` on `:900`, per the existing design that submits without holding `rebuildMutex`). A `const` declared inside the locked scope cannot cross its closing brace to reach `:914`.

**Decision:** declared `bool wasRecoveryWake = false;` in the outer loop scope (at `:848`, alongside `doDeferredPublish`/`:842`), assigned once under lock (`=exchangeAtomic`, `:888`), read once after unlock (`:914`). Effective write-once/read-once const-ness is preserved; the relaxation of the `const` qualifier is a necessary C++ scoping concession, not a semantic change. The required ordering (consume recovery → consume `publishRetryReady` → unlock → call) is fully honored.

## 3. Read-only audit (10 criteria)

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | `recoveryRetryReady` reader = 1 in `RebuildDispatch.cpp` | **PASS** | `:888` `wasRecoveryWake = convo::exchangeAtomic(recoveryRetryReady, …)`; no other reader site |
| 2 | consume = atomic read-and-clear | **PASS** | `exchangeAtomic` (`AtomicAccess.h:68`) → `std::atomic_exchange_explicit`, returns old value |
| 3 | memory order = `acq_rel` | **PASS** | `std::memory_order_acq_rel` passed explicitly |
| 4 | `wasRecoveryWake` local variable | **PASS** | Declared `:848` (loop scope), assigned `:888`, read `:914` — single loop iteration |
| 5 | `publishRetryReady` mutex protection unchanged | **PASS** | Still consumed under `rebuildMutex` (`:889–890`, `unique_lock` from `:850`) |
| 6 | CV predicate unchanged (no `recoveryRetryReady`) | **PASS** | `rebuildCV.wait` predicate `:854-859` = `hasPendingTask \|\| publishRetryReady \|\| recoveryPending \|\| consumeAtomic(rebuildThreadShouldExit)` — `recoveryRetryReady` absent |
| 7 | caller = `processDeferredAdmission(wasRecoveryWake)` | **PASS** | `:914` |
| 8 | `Timer.cpp` writer (Step 4) untouched | **PASS** | `AudioEngine.Timer.cpp:1748` `convo::publishAtomic(recoveryRetryReady, true, std::memory_order_release)` |
| 9 | `resetDeferredRetryBudget()` NOT added by Step 6 | **PASS** | `grep resetDeferredRetryBudget src/audioengine/AudioEngine.RebuildDispatch.cpp` → no match |
| 10 | changed files = `AudioEngine.RebuildDispatch.cpp` only | **PASS** | `git diff --stat HEAD`: only `AudioEngine.RebuildDispatch.cpp` (+50/−6) |

### Ordering verification (direct Read, lines 848–915)

```
848:    bool wasRecoveryWake = false;          // (outer scope, survives unlock)
...
888:    wasRecoveryWake = convo::exchangeAtomic(recoveryRetryReady, false, std::memory_order_acq_rel);  // consume recovery provenance
889:    doDeferredPublish = publishRetryReady;  // consume wake trigger
890:    publishRetryReady = false;              // clear trigger
...
900:    }                                        // ← lock released here (unique_lock scope ends)
...
914:    runtimeOrchestrator_->processDeferredAdmission(wasRecoveryWake);  // call OUTSIDE lock
```

Sequence: consume `recoveryRetryReady` → consume `publishRetryReady` → unlock → `processDeferredAdmission(wasRecoveryWake)`. ✅ Matches the Step 6 required ordering exactly.

## 4. Intentional compile-blocked intermediate state

`processDeferredAdmission` signature is **unchanged** (still `void processDeferredAdmission() noexcept`):
- `RuntimePublicationOrchestrator.h:192` — `void processDeferredAdmission() noexcept;`
- `RuntimePublicationOrchestrator.cpp:635` — `void RuntimePublicationOrchestrator::processDeferredAdmission() noexcept`

Therefore `processDeferredAdmission(wasRecoveryWake)` at `:914` is currently a **compile error** (arg passed to no-arg fn). This is the **intended, user-mandated** intermediate state:
> "この時点ではヘッダ側の宣言・.cpp側の定義変更はまだ実施しないでください。…Step 6終了時点では一時的にコンパイル不能な中間状態になります。これは意図した実装順序です。"

Step 7 (P3) wires the `bool` signature and installs the provenance-gated `resetDeferredRetryBudget()` on the consumer side, resolving both the compile error and the budget-reset relocation. No build/test is run in this step (per spec).

## 5. Verdict

**Step 6 (P2) — PASS.** Recovery-wake provenance now flows end-to-end:
producer stamp (`Timer.cpp:1748`) → atomic read-and-clear with `acq_rel` (`RebuildDispatch.cpp:888`) → handoff to consumer (`processDeferredAdmission(wasRecoveryWake)` at `:914`).
The D135-5/D135-7 "blind `publishRetryReady` cannot distinguish producers" blocker is **resolved** by Steps 4+6: `wasRecoveryWake` is now a real discriminator the consumer can branch on.

**Next:** Step 7 (P3) — change `processDeferredAdmission()` → `processDeferredAdmission(bool wasRecoveryWake)` and relocate `resetDeferredRetryBudget()` to the START of `processDeferredAdmission`, gated on `wasRecoveryWake` (reset on recovery redrive; ordinary retry keeps the increment path). Ready to proceed.
