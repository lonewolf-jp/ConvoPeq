# ConvoPeq Architecture (v0.6.9)

This document describes the internal architecture of **ConvoPeq**, a Windows-only standalone audio application built with **JUCE 8.0.12** and Intel oneMKL/IPP. It is intended for developers and contributors working on DSP, threading, state transitions, runtime governance, and ISR (Intelligent State Reconstruction) behavior.

For user-facing features and usage, see `README.md`.

---

## 1. System Goals and Non-Functional Priorities

ConvoPeq is organized around four priorities:

1. **Audio quality**
   - 64-bit double precision throughout the main DSP path.
   - IR convolution (MKL NUC) + 20-band parametric EQ (TPT SVF).
   - Output conditioning: output filter (HC/LC), musical soft clipping, and dither/noise shaping.

2. **Performance**
   - Optimized for Windows 11 x64 + AVX2-class CPUs.
   - Intel oneMKL for FFT/BLAS/VML paths (sequential, static link).
   - Intel IPP for optimized signal processing primitives.
   - Alignment-aware memory allocation (64-byte) for SIMD/MKL efficiency.
   - AVX2 intrinsics for FIR, Upsample, Tanh, EQ processes.

3. **Operational robustness**
   - UI/control logic is fully decoupled from Audio Thread DSP execution.
   - Heavy work (IR load/rebuild, NoiseShaper learning, CMA-ES optimization) is asynchronous.
   - ISR runtime governance ensures all state transitions are validated, authority-checked, and artifact-free.
   - RCU (Read-Copy-Update) / epoch-based reclamation prevents use-after-free and data races.

4. **Real-time safety**
   - Audio Thread: no allocations, no libm calls, no locks, no blocking, no exceptions.
   - All inter-thread state handoff uses RCU + atomic publish/consume patterns.
   - All temporary buffers are RAII-managed; no leaks on exceptions or early returns.

---

## 2. Adaptive Noise Shaper Learning (v0.5.8+)

- `NoiseShaperLearner` receives 256-sample `AudioBlock` structs from the Audio Thread via a `LockFreeRingBuffer<AudioBlock, 4096>`. CMA-ES optimization for 9th-order IIR noise shaper coefficients runs on a dedicated worker thread.
- Coefficient banks are managed per sample rate (10 banks) × bit depth (16/24/32) × learning mode (6 modes) = **180 total**.
- Three base learning modes (Short / Medium / Long) plus three spectral modes (Broadcast / Tonal / Custom).
- All inter-thread data transfer uses RCU/atomic/lock-free patterns.
- Typical convergence times: Short 10–20 min, Medium 20–40 min, Long 40–80 min.
- See `NoiseShaperLearner.h/.cpp` (68 KB, largest TU) and `README.md` for details.

---

## 3. Source Directory Structure (`src/` — 246 files, ~2.78 MB)

```
src/
├── [77 root files]          — Top-level DSP + UI + Framework adapters
├── audioengine/ (101 files) — ISR Runtime Governance + Orchestration + State Management
├── core/         (37 files) — RCU Foundation: EpochDomain, SnapshotCoordinator, Store
├── convolver/    (10 files) — Convolver Split (8 TU) + Internal Helpers
├── eqprocessor/   ( 6 files) — EQ Split (5 TU) + EQProcessor.h
└── tests/        (15 files) — CTest Regression Suite
```

### 3.1 `src/` Root — Core DSP / UI / Entry Points

| File(s) | Size | Role |
|---|---|---|
| `MainApplication.{h,cpp}` | 171 lines | JUCEApplication singleton. Initializes MKL, IPP, ProcessPriority, EcoQoS bypass, Denormal handling, and FileLogger. |
| `MainWindow.{h,cpp}` | 62.2 KB | JUCE DocumentWindow. Owns AudioEngine, AudioEngineProcessor, EQControlPanel, ConvolverControlPanel, SpectrumAnalyzer, DeviceSettings. |
| `AudioEngineProcessor.{h,cpp}` | 4.5 KB | `juce::AudioProcessor` adapter. Bridges `AudioProcessorPlayer` device callback into `AudioEngine::getNextAudioBlock()`. Supports both float and double paths. |
| `DeviceSettings.{h,cpp}` | 51.7 KB | ASIO/WASAPI persistence (`device_settings.xml`). Adaptive coefficient persistence. Channel mask auto-recovery. ASIO driver blacklist wrapper class. |
| `AsioBlacklist.h` | 1.5 KB | Compatibility guard for known-broken ASIO drivers. |
| `ConvolverProcessor.h` | 1180 lines, 60 KB | Public API header for IR convolution. BuildSnapshot, IRLoadPreview, PhaseMode, TailMode enums. |
| `MKLNonUniformConvolver.{h,cpp}` | 65.2 KB | Intel MKL-backed non-uniform partitioned convolution backend. |
| `CustomInputOversampler.{h,cpp}` | 33.5 KB | AVX2 multi-stage FIR/IIR oversampler (2x/4x/8x). IIRLike and LinearPhase presets. Corruption auto-detection and fallback. |
| `OutputFilter.{h,cpp}` | 20.2 KB | Biquad-based output conditioning (HPF, LPF, HC, LC). All coefficients pre-computed at prepare time. |
| `NoiseShaperLearner.{h,cpp}` | **68.4 KB** | CMA-ES-driven adaptive noise shaper learning (9th-order IIR). Largest TU in the project. |
| `EQControlPanel.{h,cpp}` | 26.0 KB | 20-band EQ user interface. |
| `SpectrumAnalyzerComponent.{h,cpp}` | 52.3 KB | Real-time FFT analyzer (MKL 4096-point). EQ overlay, peak hold, level meter bar rendering. |
| `AllpassDesigner.{h,cpp}` | 26.8 KB | All-pass filter design for mixed-phase decomposition. CMA-ES / AdaGrad optimization. |
| `CmaEsOptimizer{,.Dynamic}.{h,cpp}` | 7.7 KB | CMA-ES subspace optimizer. |
| `AlignedAllocation.h` | 6.0 KB | 64-byte SIMD-aligned `aligned_malloc` + `ScopedAlignedPtr`. |
| `DspNumericPolicy.h` | 13.5 KB | Single source of truth for DSP numeric constants and types. |
| `LockFreeRingBuffer.h` / `LockFreeAudioRingBuffer.h` | 12 KB | SPSC lock-free ring buffers for audio-thread-safe intra-thread communication. |
| `DeferredDeletionQueue.h` / `DeferredFreeThread.h` | 21 KB | Asynchronous object reclamation after RCU grace period. |
| `SafeStateSwapper.h` | 19.7 KB | RAII state swap with ownership transfer. |
| `EQEditProcessor.{h,cpp}` | 4.2 KB | UI/worker-side EQ editing interface. |
| `ConvolverState.{h,cpp}` / `IRConverter.{h,cpp}` / `IRDSP.{h,cpp}` | — | Convolver state serialization, IR normalization/preprocessing. |

### 3.2 `src/audioengine/` — ISR Runtime Governance (101 files, ~1.19 MB)

The architectural heart of ConvoPeq. `AudioEngine.h` alone is 5,600+ lines (207 KB).

**AudioEngine Split Translation Units** (PImpl-style responsibility split):

| File | Size | Responsibility |
|---|---|---|
| `AudioEngine.h` | 207 KB | All type definitions: `RuntimeState` (sealed via `BuilderToken`), `DSPCore`, `DiagEvent`, `EngineParameterSnapshot`, `RTLocalState`, `RTAuxMutable`, `EQCacheManager`, all atomic state variables. |
| `.CtorDtor.cpp` | 11.9 KB | Constructor / Destructor. ISRRetireRouter, RuntimePublicationOrchestrator, HealthMonitor, SnapshotWorker initialization. Shutdown sequence. |
| `.Init.cpp` | 4.7 KB | Post-construction initialization. |
| `.Parameters.cpp` | 32.5 KB | High-level UI parameters. |
| `.Processing.AudioBlock.cpp` | 32.8 KB | Audio Thread entry (float path). `getNextAudioBlock()`. |
| `.Processing.BlockDouble.cpp` | 28.7 KB | Audio Thread entry (double path). |
| `.Processing.DSPCoreFloat.cpp` | 15.3 KB | DSP core float processing. |
| `.Processing.DSPCoreDouble.cpp` | 36.0 KB | DSP core double processing. |
| `.Processing.DSPCoreLifecycle.cpp` | 17.2 KB | DSPCore prepare/reset lifecycle. |
| `.Processing.DSPCoreIO.cpp` | 18.8 KB | DSPCore I/O + crossfade delay gate. |
| `.Processing.PrepareToPlay.cpp` | 16.2 KB | Device callback start preparation. Lifecycle state transitions. |
| `.Processing.ReleaseResources.cpp` | 23.6 KB | Device stop resource release. |
| `.Commit.cpp` | 32.2 KB | Atomic RuntimeState commit/publish. `runPublicationPrecheckNonRt()`, `onRuntimePublishedNonRt()`, `onRuntimeRetiredNonRt()`. |
| `.RebuildDispatch.cpp` | 46.3 KB | Debounced rebuild dispatcher. `captureRuntimeBuildSnapshot()`, `equalsBuildParameterSnapshot()`, spell/rejection logic. |
| `.Timer.cpp` | **75.6 KB** | UI timer polling (100 ms). Transition verification, publication monitoring, memory tracking, learning dispatch, XRUN/crossfade/backpressure telemetry. Largest TU. |
| `.Retire.cpp` | 18.5 KB | Old `RuntimeState` retire-router logic. |
| `.Learning.cpp` | 26.0 KB | Adaptive noise shaper learning integration. |

**ISR Subsystem** (modular runtime governance):

| File | Size | Role |
|---|---|---|
| `ISRAuthorityClass.h` | 1.5 KB | `Authoritative/Derived/Diagnostic/ExecutorLocal` enum. |
| `ISRLifecycle.{h,cpp}` | 9.7 KB | Lifecycle scheduler (enter/leave audio callback). |
| `ISRRTExecution.{h,cpp}` | 4.5 KB | Real-time execution contract. |
| `ISRShutdown.{h,cpp}` | 17.4 KB | Shutdown FSM (10 states), `alignas(64) BlockingReasonStats`. |
| `ISRDSPHandle.{h,cpp}` | 9.3 KB | Handle-based DSP registry (`DSPHandleRuntime::MAX_DSP_SLOTS`). |
| `ISRDSPQuarantine.{h,cpp}` | 2.4 KB | Quarantine semantics for DSP objects. |
| `ISRClosure.{h,cpp}` | 3.1 KB | Reflective closure graph. |
| `ISRClosureGraphWalker.{h,cpp}` | 3.0 KB | Graph traversal. `validateGraph()`. |
| `ISRPayloadTier.{h,cpp}` | 2.6 KB | Payload tiering (`InlineImmutable` / `ImmutableShared`). |
| `ISRHB.{h,cpp}` | 9.2 KB | Heartbeat/hazard barrier. |
| `ISRRetire.{h,cpp}` | 9.8 KB | `RuntimeState` retirement. |
| `ISRRetireLane.h` | 0.2 KB | Retire lane classification. |
| `ISRRetireOverflowRing.h` | 4.8 KB | Overflow retirement ring. |
| `ISRRetireRouter.{h,cpp}` | 5.3 KB | Router for retirement entry + epoch coordination. |
| `ISRRetireRuntimeEx.{h,cpp}` | 21.3 KB | Extended retirement runtime (grace period, escalation, reclaim). |
| `ISRRuntimePublicationCoordinator.{h,cpp}` | 23.3 KB | Publication coordinator with overflow/deferred/shutdown schedulers. |
| `ISRRuntimeSemanticSchema.h` | 19.6 KB | Schema v9: single source of truth for authority class, ownership, mutability, visibility, and lifetime per field. |
| `ISRRuntimeIdentityGenerators.h` | 1.0 KB | Runtime/transition UUID generators. |
| `ISRSealedObject.h` | 2.9 KB | RAII seal wrapper (only Builder/Engine can construct). |
| `ISRDebugRuntime.{h,cpp}` | 6.0 KB | Debug runtime diagnostics (shadow compare, CI artifacts). |

**Runtime Publication Pipeline:**

| File | Size | Role |
|---|---|---|
| `RuntimeHealthMonitor.{h,cpp}` | 57.9 KB | Continuous runtime health/telemetry. Pull-based monitoring with 27+ monitor references. |
| `RuntimePolicyEngine.{h,cpp}` | 10.4 KB | Recovery action selection (6-level hierarchy: Observe → Throttle → Recover → Restore → Safe → Critical). |
| `RuntimePublicationOrchestrator.{h,cpp}` | 18.6 KB | Publish orchestration: Admission → Executor → DSPTransition. Deferred publish (30s TTL). |
| `RuntimePublicationValidator.{h,cpp}` | 7.8 KB | Validation pipeline (schema/authority/topology/transition). |
| `RuntimePublicationState.h` | 7.0 KB | Publication state owner + ledger. |
| `RuntimePublisher.{h,cpp}` | — | Publish executor (Coordinator-level). |
| `PublicationAdmission.{h,cpp}` | 2.4 KB | Admission evaluation (generation check, HealthState, shutdown check). |
| `PublicationExecutor.{h,cpp}` | 3.3 KB | Executor for publication (commit/dispatch). |
| `CrossfadeAuthority.{h,cpp}` | 2.3 KB | Crossfade decision authority (dspProjection-based, no DSPCore dependency). |
| `CrossfadeRuntime.h` | 9.0 KB | Crossfade executor runtime state. |
| `RuntimeBuilder.{h,cpp}` | 28.0 KB | Only entity that can construct `RuntimeState` (via `BuilderToken`). |
| `RuntimeBuildTypes.h` | 3.7 KB | Build snapshot and fingerprint types. |
| `RuntimeGraph.h` | 5.1 KB | Runtime graph representation. |
| `RuntimeTransition.h` | 2.3 KB | State transition description. |
| `FrozenRuntimeWorld.{h,cpp}` | 4.4 KB | Phase-4 frozen world concept. |
| `WorldLifecycleAudit.{h,cpp}` | 4.1 KB | World lifecycle audit trail. |
| `TelemetryRecorder.{h,cpp}` | 4.3 KB | Telemetry recording (progress, failure, correlation). |
| `RuntimeDrainAudit.h` | — | Drain audit for shutdown diagnostics. |
| `ISREvidenceExporter.{h,cpp}` | — | Evidence export for CI and auditing. |
| `AtomicAccess.h` | 5.8 KB | `consumeAtomic` / `publishAtomic` / `fetchAddAtomic` / `compareExchangeAtomic` API. Module-wide consistency for atomic operations. |
| `DSPLifetimeManager.h` / `DSPTransition.h` | 10 KB | DSP lifetime management and transition handling. |

### 3.3 `src/convolver/` — Convolver Split (10 files, ~251 KB)

8 feature flags (`CONVOPEQ_ENABLE_CONVOLVER_SPLIT_*`) control TU segmentation:

| File | Size | Responsibility |
|---|---|---|
| `ConvolverProcessor.Internal.h` | 5.5 KB | Split-internal helpers: `unwrapPhaseRadians`, `nextPow2`, `resampleIR`, `convertToMinimumPhase`. |
| `.Lifecycle.cpp` | 22.1 KB | Lifecycle management (RCU integration). |
| `.Rebuild.cpp` | 12.4 KB | Rebuild determination logic. |
| `.LoaderThread.cpp` | 29.5 KB | IR loading thread + `LoaderThreadInline.h`. |
| `.LoadPipeline.cpp` | 32.1 KB | Pipeline processing (load stages). |
| `.MixedPhase.cpp` | 38.9 KB | As-Is/Mixed/Minimum phase conversion. AllpassDesigner integration, disk cache, CMA-ES fallback. |
| `.ResampleAndFallback.cpp` | 18.0 KB | r8brain resampling and fallback paths. |
| `.Runtime.cpp` | 47.8 KB | Audio-thread runtime (process, bypass, latency). |
| `.StateAndUI.cpp` | 47.5 KB | Preset save/load, UI bridge, serialization. |

> Legacy monolithic `MKLNonUniformConvolver.cpp` (~65 KB) is kept compiled for backward compatibility, guarded by `#ifdef`.

### 3.4 `src/eqprocessor/` — 20-Band EQ Split (6 files, ~163 KB)

| File | Size | Responsibility |
|---|---|---|
| `EQProcessor.h` | 32.3 KB | `EQBandType`, `EQChannelMode`, `EQBandParams`, `EQCoeffsSVF`, `EQCoeffsBiquad`, `EQCoeffCache`, AGC constants. |
| `.Core.cpp` | 42.4 KB | Core initialization and public API. |
| `.Coefficients.cpp` | 19.3 KB | SVF and Biquad coefficient calculation (all 5 filter types). |
| `.Parameters.cpp` | 12.7 KB | Parameter update (RCU via `uintptr_t` atomic handles). |
| `.Processing.cpp` | **57.2 KB** | TPT SVF per-band processing (AVX2 FMA). Serial/Parallel structure, M/S mode, AGC, saturation. |
| `.ProcessingCache.cpp` | 2.7 KB | `EQCoeffCache` management. |

### 3.5 `src/core/` — RCU Foundation (37 files, ~118 KB)

Cross-cutting foundation delivered in phases (v13.0 redesign):

**Snapshot / RCU:**
| File | Size | Role |
|---|---|---|
| `EpochDomain.h` | **26.0 KB** | 64 named reader slots, `globalEpoch` management, quiescent-state-based reader registration/tracking. |
| `RCUReader.h` | 8.7 KB | RAII reader epoch enter/exit. |
| `SnapshotCoordinator.{h,cpp}` | 7.9 KB | Thread-safe snapshot publication and fade. |
| `SnapshotFactory.{h,cpp}` | 5.9 KB | Snapshot creation and destruction. |
| `SnapshotAssembler.{h,cpp}` | — | Snapshot assembly pipeline. |
| `RuntimeStore.h` | — | Internal store for `RuntimePublicationCoordinator`. |
| `RuntimeReaderContext.h` | — | Reader context (`ObserveChannel::Audio` / `Message` / `Publication`). |
| `ObservedRuntime.h` | — | Observed runtime abstraction (token-based). |
| `GlobalSnapshot.{h,cpp}` | — | Immutable snapshot base. |
| `ObserveChannel.h` | — | Observation channel classification. |
| `RebuildTypes.h` | — | Rebuild intent and classification types. |
| `Types.h` / `TimeUtils.h` | — | Common types, time measurement harness. |
| `EQParameters.h` | — | EQ parameter container. |

**Async Reclamation:**
| File | Role |
|---|---|
| `DeletionQueue.{h,cpp}` | Deferred object deletion queue. |
| `DeferredRetireFallbackQueue.h` | Overflow fallback for RetireRouter. |
| `WorkerThread.{h,cpp}` | Background snapshot worker thread. |
| `ThreadAffinityManager.h` | Thread affinity policy management. |
| `CommandBuffer.h` | Non-blocking command dispatch. |
| `FadeEngine.h` | Fade computation engine. |

### 3.6 `src/tests/` — CTest Regression (15 files, ~153 KB)

All tests registered via `add_test()` in CMakeLists.txt. Many are JUCE-independent.

| Test | KB | Focus |
|---|---|---|
| `RuntimePublicationCoordinatorTests` | 5.3 | Coordinator template contract. |
| `ISRSemanticValidationTests` | 19.2 | Semantic validation (schema v9). |
| `RuntimeSemanticSchemaValidationTests` | 26.7 | Field/authority invariants. |
| `RetireGraceSemanticsTests` | 12.7 | Grace period semantics. |
| `PartialPublicationRejectTests` | 21.2 | Partial publication rejection (MKL-linked). |
| `BuildInputSemanticContractTests` | 16.5 | Build input contract (large stack 8MB). |
| `RuntimeWorldAuthorityProjectionTests` | 13.1 | World authority projection invariants. |
| `OverlapAuthoritySingularTests` / `ShadowCompareContractTests` / `CrossfadeExecutorLocalContractTests` / `ObservePathSingleSourceTests` | 1–2 each | Singular authority boundaries. |
| `PublicationValidatorIsolationTests` / `RebuildAdmissionRegressionTests` / `PriorityIntegrationTests` | | Validator/release/priority tests. |
| `ISRRuntimeIdentityGeneratorsTests` | 1.5 | UUID/Generation generator correctness. |

+ External CI: `HeadlessAudioPathVerification` (PowerShell, gated by `$CONVO_CI_BUILD`).

### 3.7 `config/` — JSON Authority Manifests (4 files)

| File | Bytes | Role |
|---|---|---|
| `runtime_graph_baseline.json` | 97 | Baseline topology snapshot reference. |
| `publication_manifest.json` | 2,001 | Machine-readable publication inventory. |
| `authority_inventory.json` | 10,509 | Generated from `ISRRuntimeSemanticSchema.h` + `RuntimeGraph.h` + `AudioEngine.h`. Declares `Authoritative/Derived/Diagnostic` authority per field. |
| `pub_boundary_registry.json` | 2,255 | Publication-boundary registry (single source of publication transitions). |

Python verifiers in `tools/` cross-check source against these JSONs at build/commit time — guards against authority drift.

---

## 4. Runtime Topology and Data Flow

### 4.1 Logical Processing Chain

```
Audio Input
  → Input conditioning (DC removal, input headroom gain)
  → Oversampling (optional, 2x/4x/8x)
  → [EQ <-> Convolver] (order selectable)
  → Output Filter (HC/LC/HPF/LPF, mode-dependent)
  → Output Makeup Gain
  → Soft Clipping (optional, musical soft clip)
  → Downsampling (if oversampled)
  → Fixed Latency Delay (latency compensation)
  → Analyzer FIFO Tap (optional)
  → Audio Output (float back to device buffer)
```

### 4.2 Callback-Level Detailed Flow

```
AudioDeviceCallback → AudioProcessorPlayer → AudioEngineProcessor.getNextAudioBlock()
  → AudioEngine::getNextAudioBlock()                                                                   [Audio Thread]
    ├─ AudioCallbackRuntimeScope: lifecycle/firewall/allocator scope
    ├─ RuntimeWorld read (RCU): readAudioRuntimeView()
    │   ├─ audioThreadRcuReader.enter()
    │   ├─ RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore) → RuntimeState*
    │   ├─ resolveActiveRuntimeDSPFromRuntimeWorldOnly() → DSPCore*
    │   └─ resolveFadingRuntimeDSPFromRuntimeWorldOnly() → DSPCore*
    ├─ EngineParameterSnapshot = captureAudioThreadParameterSnapshot(runtimeWorld)
    ├─ Crossfade delay gate (if pending), arm crossfade
    ├─ DSPCore::process(bufferToFill, ...)                                                            [DSP Flow]
    │   ├─ processInput: headroom gain, DC remove, input level metering
    │   ├─ [if OS] processUp: multi-stage AVX2 FIR/IIR upsample
    │   │   └─ UltraHighRateDCBlocker.oversampledL/R
    │   ├─ route(order): EQThenConvolver → eqRt.process → convolverRt.process
    │   │                ConvolverThenEQ → convolverRt.process → eqRt.process
    │   ├─ outputFilter.process(HCMode/LCMode)
    │   ├─ scaleBlockFallback(outputMakeupGain) [AVX2]
    │   ├─ [if softClip] softClipBlockAVX2(fastTanh musical soft clip)
    │   ├─ [if OS] processDown: multi-stage AVX2 FIR downsampling
    │   ├─ pushToFifo(analyzerFifo) [if analyzer output tap]
    │   ├─ outputLevelLinear ← measureLevel(publishAtomic)
    │   └─ processOutput: DC remove, fixed latency delay, fade in ramp
    ├─ [if canCrossfade] runLatencyAlignedCrossfadeMixLoop (new/old equal-power blend)
    ├─ finish crossfade / cleanup
    └─ Diagnostic telemetry (CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS)
        ├─ CPU migration / callback sequence / DSP timing
        ├─ XRUN detection (interval > 1.5x expected || callback > 1.5x expected)
        ├─ CBSUMMARY (callback + interval max per second)
        └─ DiagEvent → LockFreeRingBuffer → Timer side drain
```

### 4.3 Inter-Thread Data Flow Architecture

```
┌─────────────────────┐     LockFreeRingBuffer<DiagEvent, 512>
│    Audio Thread     │─────→ Timer Thread (diag formatting + Logger write)
│ (DSP callback)      │
│                     │     LockFreeAudioRingBuffer (FIFO_SIZE = 1M samples)
│                     │─────→ Message Thread (SpectrumAnalyzer FFT + paint)
│                     │
│                     │     LockFreeRingBuffer<AudioBlock, 4096>
│                     │─────→ Worker Thread (NoiseShaperLearner CMA-ES)
│                     │
│                     │     publishAtomic / consumeAtomic (atomic variables)
│                     │─────→ Message/Timer Threads (all parameters + telemetry)
└─────────────────────┘

┌─────────────────────┐
│  Message Thread     │  submitRebuildIntent(Structural/...) + loadPreset + configure
│  (UI + Control)     │
│                     │─────────────────────────────────────────────↓
└─────────────────────┘                                           rebuildThreadLoop()
                                                                     ├─ buildNewDSP()
                                                                     ├─ enqueuePublicationIntent()
                                                                     └─ RuntimeBuilder.buildRuntimePublishWorld()
                                                                           → RuntimePublicationOrchestrator.submitPublishRequest()
                                                                             → PublicationAdmission.evaluate()
                                                                               → Accepted → RuntimePublicationCoordinator.publishWorld()
┌─────────────────────┐
│  Timer Thread       │  Orchestrator.tick() (100ms), HealthMonitor.tick(),
│  (100ms polling)    │  Telemetry drain, Evidence emit, Retire reclaim
└─────────────────────┘

┌─────────────────────┐     DeferredDeletionQueue → tryReclaim()
│ ISRRetireRouter     │     → old DSPCore/EQState/ConvolverState released
│ + DeferredFreeThread│     → aligned_free() / delete
└─────────────────────┘
```

### 4.4 Processing Order Routing Table

| order | convBypass | eqBypass | DSP Path |
|---|---|---|---|
| `ConvolverThenEQ` | false | false | `convolverRt.process` → `eqRt.process(eqParams, eqCache)` |
| `ConvolverThenEQ` | false | true | `convolverRt.process` → `eqRt.process()` (pass-through) |
| `ConvolverThenEQ` | true | false | bypass conv → `eqRt.process(eqParams, eqCache)` |
| `ConvolverThenEQ` | true | true | bypass conv → `eqRt.process()` (pass-through) |
| `EQThenConvolver` | false | false | `eqRt.process(eqParams, eqCache)` → `convolverInputTrimGain` → `convolverRt.process` |
| `EQThenConvolver` | true | false | `eqRt.process()` → `convolverRt.process` |

After core DSP: outputFilter always applied (convIsLast flag determines HC/LC/HPF/LPF selection). Makeup gain applied. Soft clip applied (saturation-dependent).

---

## 5. ISR Runtime Governance System

ConvoPeq implements a custom runtime governance layer (ISR) that treats every stateful field as belonging to exactly one AuthorityClass.

### 5.1 Authority Classification

| AuthorityClass | Meaning | Example Fields |
|---|---|---|
| **Authoritative** | Set in exactly one place; never derived. Mutations must be controlled. | `generation`, `topology`, `routing`, `execution`, `publication`, `overlap`, `metadata`, `retire`, `timing`, `latency` |
| **Derived** | Recomputed from authoritative fields. Not independently mutable. | `generationSemantic`, `graph`, `engine`, `resource`, `automation`, `coefficient`, `dspProjection` |
| **Diagnostic** | Observation only; must not drive runtime branching. | `worldId`, `affinity`, `projectionFreshness`, `semanticHash` |
| **ExecutorLocal** | Transient to one execution; not shared. | (none in current `RuntimeState`) |

Declared in `RuntimeState::kFieldDescriptors[21]` and ` RuntimeState::kRuntimeAuthorityInventory[21]`, verified against `config/authority_inventory.json`.

### 5.2 ISR 10-Layer Architecture

```
Layer 1    RuntimeGraph                 src/core/RuntimeGraph.h
             ├─ Active/fading node description
             └─ Contract: validateDecisionCoverageContract()

Layer 2    RuntimeState (sealed)        AudioEngine.h L133-299
             ├─ BuilderToken-protected construction
             ├─ 21 field descriptors, 21 authority inventory entries
             ├─ 10 read-authority inventory entries
             └─ Freeze/Seal for immutability post-publish

Layer 3    RuntimeBuilder               src/audioengine/RuntimeBuilder.h
             ├─ Only entity that can construct RuntimeState
             ├─ Populates all Semantic structs (topology, routing, execution, ...)
             └─ buildRuntimePublishWorld(dsp, oldDSP, policy, fadeSec, ...)

Layer 4    RuntimePublicationCoordinator  src/core/RuntimePublicationCoordinator.h
             ├─ Template <World, Handle, Bridge>
             ├─ publishWorld() — validates → commits → publishes
             └─ consumeWorldHandle() — RT read path (atomic observe)

Layer 5    RuntimePublicationValidator   src/audioengine/RuntimePublicationValidator.h
             └─ Validates schema/authority/topology/resource contracts

Layer 6    PublicationAdmission / Executor
             ├─ Admission.evaluate(healthState): generation check + shutdown + health
             └─ Executor: commit runtime world into store

Layer 7    CrossfadeAuthority            src/audioengine/CrossfadeAuthority.h
             ├─ evaluate(oldWorld, newWorld, policy) from dspProjection
             └─ Decision { needsCrossfade, fadeTimeSec }

Layer 8    ISRShutdown (FSM)             src/audioengine/ISRShutdown.h
             ├─ Running → AudioStopped → ObserverDrained → RetireClosed
             │   → EpochSettled → ReclaimComplete → [EmergencyDrain]
             │   → VerifyDrained → TimedOut|Failed → ShutdownComplete
             └─ BlockingReasonStats (alignas(64)) per-reason

Layer 9    ISRRetire / ISRRetireRouter / OverflowRing
             ├─ Router: epoch-coordinated retire entry
             ├─ OverflowRing: hardware-safe false sharing isolation
             └─ Grace period → reclaim: deferred deletion queue

Layer 10   RuntimeHealthMonitor + TelemetryRecorder + RuntimePolicyEngine
             ├─ Pull-based monitoring (27+ monitor references)
             ├─ ISRHealthState: Healthy → Degraded → Critical
             ├─ RecoveryAction hierarchy: Observe→Throttle→Recover→Restore→Safe→Critical
             └─ Telemetry/Evidence export (CI correlation)

Cross-layer    RuntimePublicationBridge
                 ├─ commit(PublishAuthority, RuntimeBoundary, newWorld, ver, seq, epoch, mappedGen)
                 └─ retire(RetireAuthority, RuntimeBoundary, oldWorld)
```

### 5.3 Publication Pipeline

```
Non-RT Path (Message/Worker/Timer Threads):
  submitRebuildIntent(kind) → RebuildDispatch.enqueueCommand()
    → rebuildThreadLoop() → buildNewDSP()
      → enqueuePublicationIntentForRuntimeCommit(newDSP, gen, sealedSnapshot)
        → RuntimePublicationOrchestrator.submitPublishRequest(req)
          ├─ admission_.evaluate(healthState, req) → Accepted
          ├─ RuntimeBuilder.buildRuntimePublishWorld()
          ├─ CrossfadeAuthority.evaluate(oldWorld, newWorld, policy)
          └─ Coordinator.publishWorld(worldOwner)
              ├─ runPublicationPrecheckNonRt(world)
              │   ├─ validateSemanticCompleteness()
              │   ├─ validateRuntimeGraphAuthorityContract()
              │   └─ precheckRuntimePublication(closure, descriptor)
              ├─ onRuntimePublishedNonRt(world)
              │   ├─ worldLifecycleAudit_.onWorldPublished()
              │   ├─ runtimePublicationBridge_.commit()
              │   ├─ lastCommittedRuntimeGeneration_ = world.generation
              │   └─ emitEvidenceTickNonRt()
              └─ RuntimeStore publish (atomic world* swap)

RT Path (Audio Thread):
  readAudioRuntimeView() → makeRuntimeReadHandle(audioCtx)
    → RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore)
      → RuntimeState* (atomic load)
        → resolveActiveRuntimeDSPFromRuntimeWorldOnly() → DSPCore*
```

### 5.4 Shutdown Sequence

```
~AudioEngine():
  1. ShutdownPhase::StopAcceptingWork    → lifecycleState=Releasing
  2. ShutdownPhase::StopAudio            → stopTimer()
  3. ShutdownPhase::StopWorkers          → stopRebuildThread()
                                        → retire active/fading DSP
                                        → shutdownWorkerThread()
  4. ShutdownPhase::ForceEpochAdvance    → m_retireRouter->publishEpoch()
  5. ShutdownPhase::DrainRetire          → poll up to 5 sec:
     while (pendingRetireCount > 0 || activeReaderCount > 0)
         m_retireRouter->publishEpoch() / tryReclaim()
  6. publishCoordinator.requestShutdownClearNonRt()
  7. runtimePublicationBridge_.markShutdownComplete()
  8. drainDeferredRetireQueues(true)
  9. m_epochDomain.drainAll()
 10. latencyBuf aligned_free
 11. lifecycleState = Destroyed
```

---

## 6. Subsystem Responsibilities

### 6.1 AudioEngine

- Owns the high-level runtime state exposed to UI.
- Bridges UI requests to DSP-safe update paths via `submitRebuildIntent()`.
- Coordinates processing order, bypass states, analyzer routing, device-driven prepare/reset, and rebuild staging.
- Owns `RuntimePublicationOrchestrator`, `RuntimeHealthMonitor`, `ISRRetireRouter`, `RuntimePublicationBridge`, `CrossfadeRuntime`, `EQCacheManager`, `WorkerThread`.
- Manages Adaptive Noise Shaper Learner lifecycle (start/stop learning, progress polling, error reporting).

### 6.2 EQProcessor

- 20-band parametric EQ in the real-time path using TPT SVF filters.
- RCU parameter updates via `uintptr_t`-backed atomic handles + `EpochDomain`.
- AGC (automatic gain control) with pre-computed attack/release/smooth coefficient tables.
- Nonlinear saturation via `fastTanh` approximation (AVX2).
- `EQCoeffCache` (RefCountedDeferred) for cross-snapshot coefficient sharing.
- Serial/Parallel filter structure with crossfade-able transition.

### 6.3 ConvolverProcessor

- IR-based convolution via Intel MKL Non-Uniform Partitioned Convolution (NUC).
- Asynchronous IR loading/rebuild on Worker Thread. `BuildSnapshot` + `StructuralHash` for rebuild decision.
- Configurable rebuild debounce (20 ms default, 10–3000 ms range).
- Crossfade-safe transitions (old/new DSP fade with latency compensation).
- Phase modes: As-Is / Mixed / Minimum. Mixed-phase uses `AllpassDesigner` (CMA-ES).
- Tail modes: AirAbsorption / LayerTailContouring / Bypass.
- Progressive FFT upgrade (background thread).

### 6.4 NoiseShaperLearner

- Dedicated worker thread for adaptive noise shaper learning.
- Audio thread pushes `AudioBlock` structs (256 samples, 2ch) to `LockFreeRingBuffer<AudioBlock, 4096>`.
- CMA-ES optimization of 9th-order IIR coefficients (180 coefficient banks).
- Multi-level normalization (4 target levels: -40/-30/-20/-10 dBFS).
- Progress, error, and best coefficients reported via atomic variables to engine/UI.
- All memory handoff and state transitions are real-time safe (RCU + lock-free).

### 6.5 SpectrumAnalyzerComponent

- Consumes `analyzerFifo` (LockFreeAudioRingBuffer) on the UI side.
- MKL 4096-point FFT. Hann windowing. Smoothing (α=0.15, 85% old retention).
- 1-second peak hold with decay. EQ overlay paths (L/R/Mid/Side individual curves).
- Adaptive timer rates: active analyzer 60 Hz, disabled-but-visible 15 Hz, hidden 5 Hz.

---

## 7. Threading Model

### Thread Classification

| Thread | Responsibility | Constraints |
|---|---|---|
| **Message Thread** (GUI) | UI rendering, event processing, user actions, device settings, dispatches async requests | Heavy work delegated to Worker thread |
| **Audio Thread** (RT Callback) | Block-based DSP processing only; always references pre-constructed state | **No** allocations, libm, locks, blocking, exceptions, I/O |
| **Timer Thread** (100ms) | Telemetry drain, rebuild dispatch, HealthMonitor polling, spectro-analysis trigger, Evidence export | |
| **Worker / Rebuild Thread** | IR parsing/loading/resampling/phase conversion, DSPCore construction, snapshot assembly | |
| **DeferredFree Thread** | Asynchronous object reclamation after RCU grace period | |
| **NoiseShaperLearner Thread** | CMA-ES optimization using recent AudioBlocks | |

### Thread-Safe Communication

| Pattern | Mechanism | Used For |
|---|---|---|
| RCU (Read-Copy-Update) | `EpochDomain` (64 slots) + `RCUReader` | EQ parameters, Convolver IR, NoiseShaper coefficients, RuntimeWorld |
| Atomic publish/consume | `publishAtomic` / `consumeAtomic` / `compareExchangeAtomic` | All scalar parameters (bypass, gain, order, mode, etc.) |
| Lock-Free SPSC Ring | `LockFreeRingBuffer<T,N>` | DiagEvent (512), XRunEvent, AudioBlock (4096) |
| Lock-Free Audio FIFO | `LockFreeAudioRingBuffer` | Spectrum analyzer (FIFO_SIZE = 1M samples) |
| Deferred Deletion | `DeferredDeletionQueue` + `DeferredFreeThread` | Old DSPCore, EQState, BandNode after grace period |

---

## 8. State and Transition Strategy

ConvoPeq follows a staged update model:

1. **Request phase (UI/control path)**
   - User or settings request a change (e.g., EQ band frequency, convolver IR file, oversampling factor).

2. **Prepare phase (non-real-time path)**
   - Expensive structures are built asynchronously:
     - DSPCore reconstruction (Convolver + EQ + Oversampler + OutputFilter)
     - Parameter snapshots (EQParameters, BuildSnapshot)
     - MKL NUC engine creation
   - `captureRuntimeBuildSnapshot()` captures sealed build fingerprint for admission comparison.

3. **Publish phase (ISR pipeline)**
   - `RuntimePublicationOrchestrator` executes admission evaluation (`PublicationAdmission::evaluate`):
     - Check HealthState (Healthy/Degraded/Critical)
     - Check shutdown state
     - Check generation monotonicity
     - Check structural hash equivalence
   - If accepted: `RuntimeBuilder.buildRuntimePublishWorld()` → `RuntimePublicationCoordinator.publishWorld()`.
   - `CrossfadeAuthority` determines crossfade type (smooth/hard reset).

4. **Apply/swap phase (real-time-safe boundary)**
   - `DSPTransition` atomically swaps active/fading DSPCore pointers.
   - Audio Thread observes new world via RCU read path on next callback.
   - Crossfade executes over fade duration (typically 30–80 ms, mode-dependent).

### Persistence Paths

- `device_settings.xml` (`DeviceSettings::saveSettings/loadSettings`)
  - Restores device state, ditherBitDepth, oversamplingFactor/Type, inputHeadroomDb, outputMakeupDb, adaptive noise shaper coefficients.
- Manual preset XML (`AudioEngine::getCurrentState()/requestLoadState()`)
  - Full processing-state portability: processing order, bypass, gain staging, filter modes, EQ 20-band parameters, Convolver params.
  - Load order is staged to prevent mode-dependent defaults from overwriting restored gain settings.

---

## 9. Crossfade System

### Crossfade Modes

| Trigger | fadeTimeSec | Source |
|---|---|---|
| Convolver bypass toggle | `m_irFadeTimeSec` (80 ms default) | atomic |
| IR length change | `m_irLengthFadeTimeSec` (50 ms) | atomic |
| Phase mode change | `m_phaseFadeTimeSec` (60 ms) | atomic |
| Direct head mode change | `m_directHeadFadeTimeSec` (10 ms) | atomic |
| NUC filter change | `m_nucFilterFadeTimeSec` (30 ms) | atomic |
| Tail mode change | `m_tailFadeTimeSec` (30 ms) | atomic |
| Oversampling change | `m_osFadeTimeSec` (30 ms) | atomic |

### Crossfade Execution

```
AudioBlock.cpp: getNextAudioBlock()
  ├─ processCrossfadeDelayGateIfPending(): delay-gate when LT delay has changed
  ├─ armCrossfadeIfPending(): activate crossfade tracking
  └─ if (canCrossfade):
      ├─ new DSP → current process path
      ├─ old DSP → dspCrossfadeFloatBuffer (fadingState, no analyzer)
      └─ runLatencyAlignedCrossfadeMixLoop(new, old, latencyDelay, gNew, gOld)
          └─ equal power sine mixing: out[i] = newL[i]*gNew + dryScaledL[i]*(1-gNew)
```

Crossfade runtime state: `CrossfadeRuntime` tracks `LinearRamp` gain (exponential ramp, 30 ms default), dry scale gain, crossfade arm status, event drop counter. `CrossfadeAuthority` determines whether crossfade is needed from `dspProjection` fields (irLoaded, structuralHash, oversamplingFactor) — no DSPCore dependency.

---

## 10. Memory and Alignment Discipline

- Main DSP path: 64-bit double precision.
- All large buffers (IR, FFT, workspaces): `convo::aligned_malloc` (64-byte alignment) + `ScopedAlignedPtr` (RAII).
- Audio Thread: allocations, libm calls, locks, exceptions, I/O **strictly prohibited**.
- `EpochDomain`: 64 named reader slots with per-slot epoch tracking and `alignas(64)` isolation.
- False-sharing prevention: critical atomics (`pendingLearningMode`, `globalCaptureSessionId`, `learningCommandWrite/Read`, et al.) use `alignas(64)`.
- All RAII-managed buffers; no leaks on exceptions or early returns.
- Denormal handling: DAZ/FTZ mode enabled at app startup + per-sample `killDenormal()` check in TPT SVF state variables.

---

## 11. Build and Runtime Context

| Aspect | Detail |
|---|---|
| OS | **Windows 11 x64** (Windows 7+ compatible API subset) |
| Framework | **JUCE 8.0.12** |
| C++ Standard | **C++20** |
| Compiler (primary) | **MSVC 19.44+ (Visual Studio 2022 17.11+)** |
| Compiler (alternative) | **Intel icx (oneAPI 2026.0)** |
| Build System | **CMake** with Ninja Multi-Config |
| Math Acceleration | **Intel oneMKL** (sequential, LP64, static) + **Intel IPP** |
| CRT | **Static** (`/MT` Release, `/MTd` Debug) |

### Build Presets (CMakePresets.json)

| Preset | Generator | Compiler | Output |
|---|---|---|---|
| `vs2026-x64` | Ninja Multi-Config | `cl` | `build/` |
| `icx-x64` | Ninja Multi-Config | `icx` | `build-icx/` |
| custom | Ninja (single) | auto | `out/build/${presetName}/` |

**Build Presets**: `debug` / `release` (configurePreset: `vs2026-x64`).

### Build Options

| Option | Default | Description |
|---|---|---|
| `CONVOPEQ_ENABLE_CLANG_TIDY` | OFF | Build-time clang-tidy analysis |
| `CONVOPEQ_ENABLE_ISR_TESTS` | ON | CTest regression suite (15 tests) |
| `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` | OFF | Runtime diagnostic logging (XRUN/MEM/VERIFY) |
| `ENABLE_ASAN` | OFF | AddressSanitizer (Debug only, forces /MDd) |
| `CONVOPEQ_PGO_INSTRUMENT` | OFF | PGO instrumentation |
| `CONVOPEQ_PGO_USE` | OFF | PGO optimized build |

### Convolver Split Feature Flags (all ON)

```
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LIFECYCLE=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_REBUILD=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOADER_THREAD=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_MIXED_PHASE=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RESAMPLE=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOAD_PIPELINE=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RUNTIME=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_STATE_UI=1
```

### Artifacts

| Configuration | Path |
|---|---|
| Debug (MSVC) | `build/ConvoPeq_artefacts/Debug/ConvoPeq.exe` |
| Release (MSVC) | `build/ConvoPeq_artefacts/Release/ConvoPeq.exe` |
| Release (icx) | `build-icx/ConvoPeq_artefacts/Release/ConvoPeq.exe` |

---

## 12. Dependency Boundaries

The following directories are external dependencies and must be treated as **strictly read-only** during normal development:

- `JUCE/` (JUCE 8.0.12, 3,056 files, 44 MB)
- `r8brain-free-src/` (r8brain sample-rate converter, 49 files, 14.5 MB)
- `.clang-format` / `.clang-tidy` / `.editorconfig` — coding style enforcement
- `.gitignore` — tracked exclusion rules

---

## 13. Development Notes

- Keep callback-time work deterministic and allocation-free.
- Treat convolver rebuilds, analyzer refresh, and NoiseShaper learning as separate burst-control problems.
- Prefer staging, debounce, and handoff over immediate heavy reconfiguration.
- All Atomic operations must use the `AtomicAccess.h` API (`publishAtomic` / `consumeAtomic` / `fetchAddAtomic` / `compareExchangeAtomic`). Direct `std::atomic::load/store` must be reviewed.
- Preserve the current read-only boundary for external dependencies.
- Runtime diagnostics (`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`) should remain OFF in Release builds; diagnostics use `LockFreeRingBuffer<DiagEvent>` to avoid Logger allocation on the Audio Thread.
- CI builds set `CONVO_CI_BUILD` environment variable to enable `NUC_DEBUG_GUARDS` and `HeadlessAudioPathVerification`.

---

## 14. Architectural Summary (Current)

ConvoPeq v0.6.9 uses a five-layer architecture:

```
┌────────────────────────────────────────────────────────────────┐
│                          UI Layer                               │
│  MainWindow, EQControlPanel, ConvolverControlPanel,             │
│  SpectrumAnalyzerComponent, DeviceSettings                      │
├────────────────────────────────────────────────────────────────┤
│                       Engine Layer                              │
│  AudioEngine (orchestration, state lifecycle, audio I/O bridge) │
├────────────────────────────────────────────────────────────────┤
│                       ISR Layer                                 │
│  RuntimePublicationOrchestrator, RuntimeHealthMonitor,           │
│  RuntimePolicyEngine, ISRShutdown, ISRRetireRouter,              │
│  CrossfadeAuthority, RuntimeBuilder, Publication Pipeline        │
├────────────────────────────────────────────────────────────────┤
│                        DSP Layer                                 │
│  EQProcessor (20-band TPT SVF), ConvolverProcessor (MKL NUC),    │
│  CustomInputOversampler (AVX2 FIR/IIR), OutputFilter (Biquad),   │
│  NoiseShaperLearner (CMA-ES), PsychoacousticDither, TruePeak     │
├────────────────────────────────────────────────────────────────┤
│                       Core Layer                                 │
│  EpochDomain (64-slot RCU), SnapshotCoordinator,                  │
│  RuntimeStore, DeferredDeletionQueue, AlignedAllocation          │
└────────────────────────────────────────────────────────────────┘
```

Design focus:

- **Strict real-time safety**: Audio Thread prohibitions enforced via Firewall (`ISRRTExecution`), zero-allocation path verification.
- **Asynchronous state construction**: Always on Worker/Message Threads; published atomically via ISR Publication Pipeline.
- **All inter-thread data transfer**: lock-free/RCU/atomic patterns; no mutex on Audio Thread.
- **All large buffers**: 64-byte aligned (`aligned_malloc` / MKL `DftiMalloc` / `PFFFT`).
- **Authority governance**: `authority_inventory.json` + `pub_boundary_registry.json` + Python verifiers → compile-time authority contract enforcement.
- **Dependency directories** (`JUCE/`, `r8brain-free-src/`): strictly read-only.
- **Schema versioning**: `ISRRuntimeSemanticSchema.h` schema v9 + `RuntimeState::kFieldDescriptors` + `RuntimeState::validateDescriptorSet()` → contract compiler-time verified.

---

*Version: v0.6.9 (Updated 2026-07-05)*
*Compiler: MSVC 19.44+ / Intel icx 2026.0*
*Platform: Windows 11 x64*
*JUCE: 8.0.12*
*MKL: oneAPI sequential (static)*
