# ConvoPeq list.md Compliance Audit (2026-05-16)

## Scope

- Target ruleset: `doc/list.md`
- Target code area: `src/**` (focus on audio runtime path, publication/lifetime path, and queue/lifecycle path)
- Audit type: static source audit (grep-driven trace + focused code reading)

## Method Summary

- Searched RT entry points and transitive hot files.
- Searched prohibited patterns (allocation/lock/exception/libm/I/O/atomic write forms).
- Read key files around publish/retire/epoch/lifecycle transitions.
- Classified findings into: `Fail`, `Partial Pass`, `Unknown`.

## Executive Result

- Final verdict: **FAIL → REMEDIATED (2026-05-16)** — 全 P0/P1 タスク完了。CI スキャン (atomic-dotcall + audioengine-lint) PASS。Debug ビルド成功。

Primary blocking categories (all resolved):

- Atomic/memory model constraints in RT path (Section 3) — **FIXED (P0-1)**
- Publication sequence order violation (Section 5) — **FIXED (P0-3)**
- Queue architecture lock-free/wait-free requirement violation (Section 9) — **FIXED (P1-1)**
- Reclaim/lifetime determinism gaps (Section 6) — **FIXED (P0-2, P1-3)**
- Ownership rule mismatch due to raw delete usage (Section 7) — **FIXED (P1-4)**
- RT dispatch から listeners.call()/MessageManager アクセスが発生する違反 — **FIXED (P1-2 RT 側追加修正)**

## Confirmed Findings

### F-01 (Critical) RT path performs atomic exchange/fetch_add writes — **RESOLVED (P0-1)**

- Rule impact: list.md `3.1` (RT atomic restrictions)
- Evidence examples:
  - `src/convolver/ConvolverProcessor.Runtime.cpp`
    - exchangeAtomic in processing path (e.g. firstProcessCall/latency flags/mix reset flags)
    - fetch_add at latency clamp counter
  - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
    - `m_audioBlockCounter.fetch_add(...)`
  - `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
    - `m_audioBlockCounter.fetch_add(...)`
  - `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp`
    - `dropCount.fetch_add(...)`
  - `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`
    - `debugAppliedEqHashVersion.fetch_add(...)`
  - `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
    - `dropCount.fetch_add(...)`, `debugAppliedEqHashVersion.fetch_add(...)`

### F-02 (Critical) RT processing calls EQ setBypass (publishing atomic store) — **RESOLVED (P0-1)**

- Rule impact: list.md `3.1.1` (store prohibition in RT) and RT write constraints
- Evidence:
  - Call sites in RT processing:
    - `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`
    - `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
  - Setter implementation publishes atomic:
    - `src/eqprocessor/EQProcessor.h` (`setBypass`)

### F-03 (High) Publication sequence order violation (retire before epoch advance) — **RESOLVED (P0-3)**

- Rule impact: list.md `5.1` (required order: build -> warmup -> publishAtomic -> advanceEpoch -> retire)
- Evidence:
  - `src/convolver/ConvolverProcessor.LoadPipeline.cpp`
    - old engine retired before `EpochManager::instance().advanceEpoch()` in `switchEngineOnMessageThread`

### F-04 (High) RuntimeCommandQueue is not lock-free/wait-free — **RESOLVED (P1-1)**

- Rule impact: list.md `9.1.2`, `9.1.3`
- Evidence:
  - `src/audioengine/RuntimeCommandQueue.h`
    - enqueue path guarded by `std::lock_guard<std::mutex>`

### F-05 (High) UI -> Runtime direct mutate path exists — **RESOLVED (P1-2)**

- Rule impact: list.md `1.1.5` (UI->Runtime direct mutate prohibited; command queue path required)
- Evidence:
  - UI calls:
    - `src/ConvolverControlPanel.cpp` (setMix/setSmoothingTime/setTargetIRLength)
  - Runtime mutators with atomic publish:
    - `src/convolver/ConvolverProcessor.Runtime.cpp` (`setMix`, `setBypass`, `setTargetIRLength`, `setSmoothingTime`)

### F-06 (High) SnapshotCoordinator deferred deletion queue reclaim gap — **RESOLVED (P0-2)**

- Rule impact: list.md `6.2.3`, `6.3` (deterministic reclaim/safety)
- Evidence:
  - Enqueue to local deletion queue exists:
    - `src/core/SnapshotCoordinator.h`
    - `src/core/SnapshotCoordinator.cpp`
  - Reclaim implementation exists in DeletionQueue:
    - `src/core/DeletionQueue.cpp`
  - No clear call site found for `SnapshotCoordinator` internal `m_deletionQueue.reclaim(...)`

### F-07 (Medium) DeletionQueue is unbounded (vector push_back) — **RESOLVED (P1-3)**

- Rule impact: list.md `6.2.1` (bounded capacity)
- Evidence:
  - `src/core/DeletionQueue.h` (`std::vector<Entry> queue`)
  - `src/core/DeletionQueue.cpp` (`queue.push_back(...)`)

### F-08 (Medium) Ownership policy mismatch due to raw delete usage — **RESOLVED (P1-4)**

- Rule impact: list.md `7.1` (delete/manual free prohibition)
- Evidence:
  - Multiple raw delete call sites across `src/**` (including cache/lifecycle/thread/deferred cleanup paths)

## Confirmed Positives

### P-01 RT core files did not show lock/I-O/exception usage in processing bodies

- Checked files:
  - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
  - `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
  - `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp`
  - `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`
  - `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
  - `src/convolver/ConvolverProcessor.Runtime.cpp`
  - `src/eqprocessor/EQProcessor.Processing.cpp`

### P-02 Denormal/SIMD/cacheline precautions are present broadly

- Evidence types observed:
  - `juce::ScopedNoDenormals`
  - `_MM_SET_FLUSH_ZERO_MODE`, `_MM_SET_DENORMALS_ZERO_MODE`
  - `alignas(64)` on queue/index/state fields

## Section Status Snapshot

- 1 One-way Dataflow: **Pass** (P1-2 完了: UI 直参照ゼロ、RT dispatch の listeners.call() 違反修正済み)
- 2 Audio Thread Hard RT Safety: **Pass** (P0-1/P1-2 RT 側修正により mutex lock/I-O 排除)
- 3 Atomic/Memory Model: **Pass** (P0-1 で RT fetch_add/exchangeAtomic 違反解消)
- 4 Immutable RuntimeWorld: **Pass** (static mutable violations in RuntimeBuilder.cpp fixed 2026-05-16; no remaining static/thread_local mutable state found)
- 5 Publication: **Pass** (P0-3 で retire-after-epoch 順序修正、LINT-AE-007 CI 監視)
- 6 RCU/Lifetime: **Pass** (P0-2/P1-3 で DeletionQueue bounded + reclaim 確立)
- 7 Ownership: **Pass** (P1-4 で raw delete を deleter lambda 内に統一)
- 8 Blueprint: **Pass** (EngineCommand / BuildInput are trivially-copyable value types, no setters, no mutable, no runtime back-refs)
- 9 Command Queue: **Pass** (P1-1 で lock-free SPSC 化、LINT-AE-006 CI 監視)
- 10 Builder: **Pass** (single publisher path, RAII rollback, generation-check stale-build cancel)
- 11 Transition/Crossfade: **Pass** (independent DSPCore allocs, crossfade encapsulated in CrossfadeDSPState, fading DSPCore retired via exchangeFadingOutDSP+retireDSP in Timer.cpp)
- 12 Shutdown: **Pass** (shutdown/release 経路で `drainDeferredRetireQueues(true)` を実行し、join 後 reclaim の最終回収を明示保証。`reclaimAllIgnoringEpoch()` は LINT-AE-012 で誤用防止)
- 13 SIMD/DSP: **Pass-oriented**
- 14 Cache/False Sharing: **Pass-oriented**
- 15 AI-generated code checks: **Pass** (`src/**` のコメント行に禁止語 TODO/FIXME/workaround/quick fix/just for now/temporary の一致なし、LINT-AE-009 でCI監視。加えて実コード中の `thread_local` / `mutable` を LINT-AE-011 で監視し、`const_cast` を LINT-AE-013 で禁止。さらに `const` を含む危険な C 形式ポインタキャストを LINT-AE-014 で禁止)
- 16/17/18 Final invariant gate: **Pass** (Fail/Partial Pass 項目を解消し、lint + build で再検証済み)

## Limitations

- This report is from static source inspection and pattern tracing.
- AST/callgraph-level mandatory verification (Section 4/8/10/11) is completed for this audit cycle.

## Recommended Next Audit Actions

1. ~~Execute AST/callgraph checks for Section 4, 8, 10, 11 unknown areas.~~ **DONE (2026-05-16): all four sections verified and static mutable violations fixed.**
2. ~~Trace and fix publication ordering around convolver engine switch path.~~ **DONE (P0-3)**
3. ~~Replace mutex-based runtime command queue with lock-free design matching list.md constraints.~~ **DONE (P1-1)**
4. ~~Eliminate RT atomic write operations that violate Section 3.1.~~ **DONE (P0-1 + P1-2 RT-side fix)**
5. ~~Add explicit reclaim invocation path for SnapshotCoordinator internal deletion queue.~~ **DONE (P0-2)**
