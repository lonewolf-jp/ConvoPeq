# ConvoPeq Architecture (v0.6.10)

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
   - Build failures are classified (Permanent / Transient / Infrastructure / Fatal) and retried via `RetryScheduler` when disposition allows.

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
- See `NoiseShaperLearner.h/.cpp` (~79.8 KB) and `README.md` for details.

---

## 3. Source Directory Structure (`src/` — 336 source files, ~4.62 MB)

```
src/
├── [87 root files]          — Top-level DSP + UI + Framework adapters + FFT abstraction
├── audioengine/ (126 files) — ISR Runtime Governance + Orchestration + State Management
├── core/         (40 files) — RCU Foundation: EpochDomain, SnapshotCoordinator, Store
├── convolver/    (10 files) — Convolver Split (8 TU) + Internal Helpers
├── eqprocessor/  (17 files) — EQ Split (5 TU) + EQProcessor.h + Analysis Subsystem + EQEditProcessor
├── tests/        (55 files) — CTest Regression Suite (36 executables, ~943 KB)
├── dsp/math/     ( 1 file)  — FastTanhApprox.h (AVX2 tanh approximation)
└── tools/        ( 2 files) — Build identity gate + layout offset check (Python)
```

### 3.1 `src/` Root — Core DSP / UI / Entry Points

| File(s) | Size | Role |
|---|---|---|
| `MainApplication.{h,cpp}` | 12.6 KB | JUCEApplication singleton. Initializes MKL, IPP, ProcessPriority, EcoQoS bypass, Denormal handling, and FileLogger. |
| `MainWindow.{h,cpp}` | 70.3 KB | JUCE DocumentWindow. Owns AudioEngine, EQControlPanel, ConvolverControlPanel, SpectrumAnalyzer, DeviceSettings. |
| `DeviceSettings.{h,cpp}` | 59.3 KB | ASIO/WASAPI persistence (`device_settings.xml`). Adaptive coefficient persistence. Channel mask auto-recovery. ASIO driver blacklist wrapper class. |
| `AsioBlacklist.h` | 1.5 KB | Compatibility guard for known-broken ASIO drivers. |
| `ConvolverProcessor.h` | 66.0 KB | Public API header for IR convolution. BuildSnapshot, IRLoadPreview, PhaseMode, TailMode enums. |
| `ConvolverBuilder.h` | 3.1 KB | ISR Plan Builder (Non-RT Authority). Sole creator/destroyer of FFT Plans; injects Plan into FFTExecutionContext. |
| `FFTBackend.{h,cpp}` | 12.3 KB | FFT abstraction layer (C++20 Concept). `ProductionFft` (Intel IPP) + `TestFft` injectable backend. Plan lifecycle owned by Builder; RT calls are const & noexcept. |
| `FFTExecutionContext.{h,cpp}` | 5.8 KB | Receives const Plan& from Builder; must not own or extend Plan lifetime (EC-1/EC-2/PLAN-LT). |
| `MKLNonUniformConvolver.{h,cpp}` | 106.0 KB | Intel MKL-backed non-uniform partitioned convolution backend. Legacy monolithic (kept for backward compat). |
| `CustomInputOversampler.{h,cpp}` | 37.9 KB | AVX2 multi-stage FIR/IIR oversampler (2x/4x/8x). IIRLike and LinearPhase presets. Corruption auto-detection and fallback. |
| `OutputFilter.{h,cpp}` | 28.0 KB | Biquad-based output conditioning (HPF, LPF, HC, LC). All coefficients pre-computed at prepare time. |
| `NoiseShaperLearner.{h,cpp}` | 79.8 KB | CMA-ES-driven adaptive noise shaper learning (9th-order IIR). |
| `NoiseShaperLearnerTypes.h` | 2.3 KB | Learning mode, normalization level, error type enums. |
| `PsychoacousticDither.{h,cpp}` | 26.5 KB | Ultra Mastering dither engine: Xoshiro256** RNG, TPDF dither, 12th-order noise shaper, quantization (16/24/32-bit). |
| `FixedNoiseShaper.h` / `Fixed15TapNoiseShaper.h` / `LatticeNoiseShaper.h` | 42.9 KB | Fixed (4-tap / 15-tap) and adaptive lattice (9th-order AVX2) noise shapers. |
| `TruePeakDetector.{h,cpp}` | 14.4 KB | 4x oversampled (2-stage) true peak measurement. AVX2-optimized. |
| `LoudnessMeter.{h,cpp}` | 13.0 KB | ITU-R BS.1770 compliant loudness measurement. K-weighting pre-filter + RLB filter. |
| `ProgressiveUpgradeThread.{h,cpp}` | 8.7 KB | Background progressive FFT size upgrade for convolution quality. |
| `IRConverter.{h,cpp}` | 17.5 KB | Audio file → prepared IR state conversion. Configurable FFT/partition size, phase mode. |
| `IRAnalyzer.{h,cpp}` | 7.9 KB | FFT-based IR analysis: peak gain estimation, Tukey window, Gaussian interpolation. |
| `IRDSP.{h,cpp}` | 6.7 KB | High-quality IR resampling via r8brain library. |
| `ConvolverState.{h,cpp}` | 5.1 KB | Lightweight convolver state metadata (stateId, generationId, sampleRate). Used by SafeStateSwapper. |
| `CacheManager.{h,cpp}` / `MixedPhasePersistentCache.{h,cpp}` | 36.2 KB | IR disk cache management and mixed-phase persistent cache (LRU, SQLite-backed). |
| `MKLRealTimeSetup.{h,cpp}` | 1.4 KB | MKL real-time configuration: FTZ/DAZ/VML mode, error callback suppression. |
| `CpuFeatureCheck.{h,cpp}` | 6.6 KB | AVX2/FMA runtime CPU feature detection. |
| `MklFftEvaluator.h` | 37.2 KB | MKL FFT evaluator for CMA-ES spectral analysis. |
| `EQControlPanel.{h,cpp}` | 30.2 KB | 20-band EQ user interface. |
| `ConvolverControlPanel.{h,cpp}` | 66.2 KB | Convolver control panel: IR load, phase/tail mode, mix, HC/LC. |
| `ConvolverSettingsComponent.{h,cpp}` | 5.2 KB | Advanced convolver settings panel. |
| `SpectrumAnalyzerComponent.{h,cpp}` | 59.0 KB | Real-time FFT analyzer (MKL 4096-point). EQ overlay, peak hold, level meter bar rendering. |
| `NoiseShaperLearningComponent.{h,cpp}` | 24.1 KB | Noise shaper learning UI (progress, error metrics). |
| `MixedPhaseOptimizationComponent.{h,cpp}` | 5.3 KB | Mixed-phase optimization progress UI. |
| `AllpassDesigner.{h,cpp}` | 32.2 KB | All-pass filter design for mixed-phase decomposition. CMA-ES / AdaGrad optimization. |
| `CmaEsOptimizer.h` / `CmaEsOptimizerDynamic.{h,cpp}` | 17.0 KB | CMA-ES abstract optimizer and dynamic subspace optimizer. |
| `DspNumericPolicy.h` | 16.0 KB | Single source of truth for DSP numeric constants and types. |
| `LockFreeRingBuffer.h` / `LockFreeAudioRingBuffer.h` | 12.4 KB | SPSC lock-free ring buffers for audio-thread-safe intra-thread communication. |
| `MpscBoundedRing.h` | 11.5 KB | Bounded Multi-Producer Single-Consumer ring (Vyukov-style). Replaces SPSC `LockFreeRingBuffer` for `intentQueue_`, which is pushed from Builder/Rebuild, Timer, and CoordinatorLoop deferred resubmit threads. |
| `DeferredDeletionQueue.h` / `DeferredFreeThread.h` | 24.0 KB | Asynchronous object reclamation after RCU grace period. |
| `SafeStateSwapper.h` | 22.1 KB | RAII state swap with ownership transfer. |
| `EQEditProcessor.{h,cpp}` | 7.5 KB | UI/worker-side EQ editing interface. |
| `AlignedAllocation.h` | 8.5 KB | 64-byte aligned memory allocation (SIMD/MKL compatible). |
| `AudioSegmentBuffer.h` | 7.2 KB | Audio block segment buffer for thread-safe transfer. |
| `ConvolverRuntimeCompatAliases.h` | 0.2 KB | Runtime-compatible convolver type aliases. |
| `DftiHandle.h` | 1.6 KB | RAII wrapper for MKL DFTI descriptors. |
| `DiagnosticsConfig.h` | 12.7 KB | Runtime diagnostic configuration (sample masks, verbosity). |
| `GenerationManager.h` | 2.4 KB | Generation counter management for rebuild tracking. |
| `InputBitDepthTransform.h` | 5.0 KB | Input bit depth transformation utilities. |
| `PreparedIRState.h` | 3.2 KB | Prepared IR state container. |
| `RefCountedDeferred.h` | 3.6 KB | Ref-counted deferred deletion wrapper. |
| `StateKey.h` | 1.3 KB | State key type for state tracking. |
| `UltraHighRateDCBlocker.h` | 10.0 KB | DC blocker for oversampled paths (ultra-high-rate). |

> `AudioEngineProcessor.{h,cpp}` lives under `src/audioengine/` (see §3.2), not in `src/` root.

### 3.2 `src/audioengine/` — ISR Runtime Governance (126 files, ~1.91 MB)

The architectural heart of ConvoPeq. `AudioEngine.h` alone is 5,213 lines (280.5 KB).

**AudioEngine Split Translation Units** (PImpl-style responsibility split):

| File | Size | Responsibility |
|---|---|---|
| `AudioEngine.h` | 280.5 KB | All type definitions: `RuntimeState` (sealed via `BuilderToken`), `DSPCore`, `DiagEvent`, `EngineParameterSnapshot`, `RTLocalState`, `RTAuxMutable`, `EQCacheManager`, all atomic state variables. |
| `.CtorDtor.cpp` | 19.6 KB | Constructor / Destructor. ISRRetireRouter, RuntimePublicationOrchestrator, HealthMonitor, SnapshotWorker initialization. Shutdown sequence. |
| `.Init.cpp` | 10.7 KB | Post-construction initialization. |
| `.Parameters.cpp` | 32.4 KB | High-level UI parameters. |
| `.Processing.AudioBlock.cpp` | 35.9 KB | Audio Thread entry (float path). `getNextAudioBlock()`. |
| `.Processing.BlockDouble.cpp` | 33.2 KB | Audio Thread entry (double path). |
| `.Processing.DSPCoreFloat.cpp` | 17.4 KB | DSP core float processing. |
| `.Processing.DSPCoreDouble.cpp` | 29.2 KB | DSP core double processing. |
| `.Processing.DSPCoreLifecycle.cpp` | 21.4 KB | DSPCore prepare/reset lifecycle. |
| `.Processing.DSPCoreIO.cpp` | 20.0 KB | DSPCore I/O + crossfade delay gate. |
| `.Processing.DSPCoreToBuffer.cpp` | 1.7 KB | DSPCore to buffer transformation. |
| `.Processing.Latency.cpp` | 6.6 KB | Latency compensation processing. |
| `.Processing.PrepareToPlay.cpp` | 19.5 KB | Device callback start preparation. Lifecycle state transitions. |
| `.Processing.ReleaseResources.cpp` | 46.9 KB | Device stop resource release. |
| `.Processing.Snapshot.cpp` | 2.0 KB | Processing snapshot capture. |
| `.Commit.cpp` | 43.2 KB | Atomic RuntimeState commit/publish. `runPublicationPrecheckNonRt()`, `onRuntimePublishedNonRt()`, `onRuntimeRetiredNonRt()`. |
| `.RebuildDispatch.cpp` | 77.6 KB | Debounced rebuild dispatcher. `captureRuntimeBuildSnapshot()`, `equalsBuildParameterSnapshot()`, spell/rejection logic. |
| `.Timer.cpp` | **110.5 KB** | UI timer polling (100 ms). Transition verification, publication monitoring, memory tracking, learning dispatch, XRUN/crossfade/backpressure telemetry. Largest TU. |
| `.Retire.cpp` | 26.4 KB | Old `RuntimeState` retire-router logic. |
| `.Learning.cpp` | 25.4 KB | Adaptive noise shaper learning integration. |
| `.Cache.cpp` | 7.5 KB | EQ/convolver cache management. |
| `.EQResponse.cpp` | 10.0 KB | EQ frequency response computation. |
| `.Fifo.cpp` | 0.7 KB | FIFO initialization/shutdown. |
| `.Globals.cpp` | 0.2 KB | Global static initialization. |
| `.Mmcss.cpp` | 10.5 KB | MMCSS (Multimedia Class Scheduler Service) integration for audio thread priority. |
| `.Publication.cpp` | 4.3 KB | Publication orchestration bridge. |
| `.Reader.cpp` | 0.7 KB | Runtime reader initialization. |
| `.Snapshot.cpp` | 7.6 KB | Snapshot orchestration. |
| `.StateIO.cpp` | 11.9 KB | State save/load (preset XML I/O). |
| `.Threading.cpp` | 21.1 KB | Thread pool initialization and management. |
| `.Transition.cpp` | 1.3 KB | DSP transition management. |
| `.UIEvents.cpp` | 8.9 KB | UI event dispatch and batched updates. |

**Engine Support Files:**

| File | Size | Role |
|---|---|---|
| `AudioEngineProcessor.{h,cpp}` | 7.3 KB | JUCE AudioProcessor adapter for AudioProcessorPlayer bridge. |
| `AutoGainPlanner.{h,cpp}` | 9.3 KB | Automatic gain staging planner (v14.0). |
| `OversamplingPolicy.h` | 3.8 KB | Oversampling policy configuration. |
| `SimplePeakLimiter.h` | 3.3 KB | Simple peak limiter for output protection. |
| `ShutdownScope.h` | 3.2 KB | RAII scope guard for shutdown phases. |
| `RuntimePublicationSpecification.h` | 0.5 KB | Publication specification type. |
| `BuildErrorPolicy.h` | 8.4 KB | Extracted policy contract (`BuildError` / `FailureClassification` / `RetryDisposition` / `BuildOutcome`). Allows standalone contract tests without pulling AudioEngine.h / JUCE. |
| `RetryScheduler.{h,cpp}` | 4.6 KB | Minimal rebuild-retry scheduler. `RetryScheduleRequest` + `PendingRetry`; engine dispatch via injected callback (no AudioEngine.h dependency). |
| `RetrySchedulerTypes.h` | 1.8 KB | Telemetry reason/class/policy enums for retry scheduling. |
| `SequenceArithmetic.h` | 5.6 KB | Wraparound-safe modular sequence arithmetic (`isBefore` / `isAfter` / `isAtOrBefore` / `isCompleted`) per RFC 1982 serial-number semantics. |

**ISR Subsystem** (modular runtime governance):

| File | Size | Role |
|---|---|---|
| `ISRAuthorityClass.h` | 1.7 KB | `Authoritative/Derived/Diagnostic/ExecutorLocal` enum. |
| `ISRLifecycle.{h,cpp}` | 13.3 KB | Lifecycle scheduler (enter/leave audio callback). |
| `ISRRTExecution.{h,cpp}` | 10.1 KB | Real-time execution contract (firewall). |
| `ISRShutdown.{h,cpp}` | 47.0 KB | Shutdown FSM, `alignas(64) BlockingReasonStats`. |
| `ISRDSPHandle.{h,cpp}` | 24.4 KB | Handle-based DSP registry (`DSPHandleRuntime::MAX_DSP_SLOTS`). |
| `DSPHandleTable.h` | 7.0 KB | `DSPCore*` → `DSPHandle` O(1) open-addressing forward hash table (capacity 512 = 2× MAX_DSP_SLOTS). Replaces `std::unordered_map`; no rehash, no heap alloc on find/insert/erase. |
| `ISRDSPQuarantine.{h,cpp}` | 8.9 KB | Quarantine semantics for DSP objects. |
| `ISRClosure.{h,cpp}` | 4.2 KB | Reflective closure graph. |
| `ISRClosureGraphWalker.{h,cpp}` | 4.4 KB | Graph traversal. `validateGraph()`. |
| `ISRPayloadTier.{h,cpp}` | 4.1 KB | Payload tiering (`InlineImmutable` / `ImmutableShared`). |
| `ISRHB.{h,cpp}` | 11.2 KB | Heartbeat/hazard barrier. |
| `ISRRetire.{h,cpp}` | 20.3 KB | `RuntimeState` retirement. `LifetimeState` sole owner of lifetime/retire state. |
| `ISRRetireLane.h` | 0.2 KB | Retire lane classification. |
| `ISRRetireOverflowRing.h` | 4.6 KB | Overflow retirement ring. |
| `ISRRetireRouter.{h,cpp}` | 49.4 KB | Router for retirement entry + epoch coordination. |
| `ISRRetireRuntimeEx.{h,cpp}` | 25.3 KB | Extended retirement runtime (grace period, escalation, reclaim). |
| `RetireQuarantineStore.h` | 12.7 KB | Fallback store when `DeferredDeletionQueue::enqueue` is full (RT readers still holding refs). Holds objects without deleting until epoch-safe drain; allocation-free (`std::array` + index placement). |
| `ISRWorldRetirementTelemetry.h` | 17.6 KB | World retirement observation-window telemetry (`ObservationWindowTag`: Normal / Stall / Shutdown / Catastrophic). |
| `ISRWorldRetirementReference.h` | 6.3 KB | World retirement reference tracking. |
| `ISRRuntimePublicationCoordinator.{h,cpp}` | 173.8 KB | Publication coordinator with overflow/deferred/shutdown schedulers. |
| `ISRRuntimePublicationCoordinator_ProcessIntent.cpp` | 11.8 KB | Intent processing path extracted from coordinator. |
| `ISRRuntimeSemanticSchema.h` | 19.2 KB | Schema v9: single source of truth for authority class, ownership, mutability, visibility, and lifetime per field. |
| `ISRRuntimeIdentityGenerators.h` | 1.0 KB | Runtime/transition UUID generators. |
| `ISRRuntimeWorldAuthority.h` | 0.6 KB | `RuntimeWorldAuthority` forward declarations / holder types. |
| `ISRSealedObject.h` | 2.8 KB | RAII seal wrapper (only Builder/Engine can construct). |
| `ISRDebugRuntime.{h,cpp}` | 6.9 KB | Debug runtime diagnostics (shadow compare, CI artifacts). |
| `ISREvidenceExporter.{h,cpp}` | 18.8 KB | Evidence export for CI and auditing. |
| `ISRIntentDispatcher.h` | 5.1 KB | Intent dispatch context / handler boundary. |
| `ISRCoordinatorLoop.{h,cpp}` | 2.7 KB | Coordinator loop (deferred resubmit producer for intent queue). |
| `ISRLifetimeProof.h` | 10.4 KB | Shutdown lifetime proof / permit types (`ShutdownRuntimeIdentity`, `ShutdownQuiescenceProof`, `ReclaimPermit`, `ReclaimIdentity`). Currently type-only; production reclaim connection is a later phase. |

**Runtime Publication Pipeline:**

| File | Size | Role |
|---|---|---|
| `RuntimeHealthMonitor.{h,cpp}` | 90.4 KB | Continuous runtime health/telemetry. Pull-based monitoring with 27+ monitor references. |
| `RuntimePolicyEngine.{h,cpp}` | 26.1 KB | Recovery action selection (6-level hierarchy: Observe → Throttle → Recover → Restore → Safe → Critical). |
| `RuntimePublicationOrchestrator.{h,cpp}` | 73.3 KB | Publish orchestration: Admission → Executor → DSPTransition. Deferred publish (30s TTL). |
| `RuntimePublicationValidator.{h,cpp}` | 10.7 KB | Validation pipeline (schema/authority/topology/transition). |
| `RuntimePublicationState.h` | 7.3 KB | Publication state owner + ledger. |
| `PublicationAdmission.{h,cpp}` | 10.0 KB | Admission evaluation (generation check, HealthState, shutdown check). |
| `PublicationExecutor.{h,cpp}` | 6.6 KB | Executor for publication (commit/dispatch). |
| `RuntimePublishExecutor.h` | 7.4 KB | Sole Execution gateway to `RuntimeWorldAuthority::commit()` on the Publish path (ISR side). Reads only the publish payload fixed at enqueue; never re-decides. |
| `RuntimeWorldAuthority.h` | 21.4 KB | World authority + `PendingPublishRegistry` (owner of newWorld during async enqueue→commit gap). Lock-free; capacity 64. |
| `OwnerChannel.h` | 6.9 KB | Lock-free SPSC owner-transfer channel. Transfers sole ownership of a `RuntimeStateOwner` across the RT boundary. Key = (sequenceId, epoch, mappedGeneration). |
| `CrossfadeAuthority.{h,cpp}` | 4.0 KB | Crossfade decision authority (dspProjection-based, no DSPCore dependency). |
| `CrossfadeRuntime.h` | 12.7 KB | Crossfade executor runtime state. |
| `RuntimeBuilder.{h,cpp}` | 35.3 KB | Only entity that can construct `RuntimeState` (via `BuilderToken`). |
| `RuntimeBuildTypes.h` | 15.2 KB | Build snapshot and fingerprint types. |
| `RuntimeGraph.h` | 4.9 KB | Runtime graph representation. |
| `RuntimeTransition.h` | 2.1 KB | State transition description. |
| `FrozenRuntimeWorld.{h,cpp}` | 5.2 KB | Phase-4 frozen world concept. |
| `WorldLifecycleAudit.{h,cpp}` | 9.5 KB | World lifecycle audit trail. |
| `TelemetryRecorder.{h,cpp}` | 14.5 KB | Telemetry recording (progress, failure, correlation). |
| `RuntimeDrainAudit.h` | 4.3 KB | Drain audit for shutdown diagnostics. |
| `AtomicAccess.h` | 5.5 KB | `consumeAtomic` / `publishAtomic` / `fetchAddAtomic` / `compareExchangeAtomic` API. Module-wide consistency for atomic operations. |
| `DSPLifetimeManager.{h,cpp}` | 6.8 KB | DSP lifetime management. |
| `DSPTransition.h` | 10.8 KB | DSP transition handling (publish-completion facade). |

### 3.3 `src/convolver/` — Convolver Split (10 files, ~261 KB)

8 feature flags (`CONVOPEQ_ENABLE_CONVOLVER_SPLIT_*`) control TU segmentation:

| File | Size | Responsibility |
|---|---|---|
| `ConvolverProcessor.Internal.h` | 5.3 KB | Split-internal helpers: `unwrapPhaseRadians`, `nextPow2`, `resampleIR`, `convertToMinimumPhase`. |
| `.Lifecycle.cpp` | 23.7 KB | Lifecycle management (RCU integration). |
| `.Rebuild.cpp` | 12.2 KB | Rebuild determination logic. |
| `.LoaderThread.cpp` | 33.6 KB | IR loading thread + `LoaderThreadInline.h` (3.4 KB). |
| `.LoadPipeline.cpp` | 37.1 KB | Pipeline processing (load stages). |
| `.MixedPhase.cpp` | 37.1 KB | As-Is/Mixed/Minimum phase conversion. AllpassDesigner integration, disk cache, CMA-ES fallback. |
| `.ResampleAndFallback.cpp` | 17.1 KB | r8brain resampling and fallback paths. |
| `.Runtime.cpp` | 47.0 KB | Audio-thread runtime (process, bypass, latency). |
| `.StateAndUI.cpp` | 47.8 KB | Preset save/load, UI bridge, serialization. |

> Legacy monolithic `MKLNonUniformConvolver.cpp` (~80 KB) is kept compiled for backward compatibility, guarded by `#ifdef`.

### 3.4 `src/eqprocessor/` — 20-Band EQ Split + Analysis Subsystem (17 files, ~207 KB)

| File | Size | Responsibility |
|---|---|---|
| `EQProcessor.h` | 37.5 KB | `EQBandType`, `EQChannelMode`, `EQBandParams`, `EQCoeffsSVF`, `EQCoeffsBiquad`, `EQCoeffCache`, AGC constants. |
| `.Core.cpp` | 46.7 KB | Core initialization and public API. |
| `.Coefficients.cpp` | 20.6 KB | SVF and Biquad coefficient calculation (all 5 filter types). |
| `.Parameters.cpp` | 13.1 KB | Parameter update (RCU via `uintptr_t` atomic handles). |
| `.Processing.cpp` | **56.2 KB** | TPT SVF per-band processing (AVX2 FMA). Serial/Parallel structure, M/S mode, AGC, saturation. |
| `.ProcessingCache.cpp` | 3.1 KB | `EQCoeffCache` management. |
| `PeakEstimator.{h,cpp}` | 5.4 KB | Peak detection for EQ analysis. |
| `UpperBoundEstimator.{h,cpp}` | 1.2 KB | Upper bound estimation for EQ bands. |
| `EQResponseSampler.{h,cpp}` | 9.5 KB | Frequency response sampling (magnitude/phase). |
| `AnalysisMerge.h` | 2.7 KB | Merges multiple analysis results. |
| `BandHelper.{h,cpp}` | 2.3 KB | Band utility functions and helpers. |
| `EQAnalysisMath.h` | 4.0 KB | Mathematical formulas for EQ analysis. |
| `EQAnalysisTypes.h` | 4.6 KB | Analysis type definitions. |

### 3.5 `src/core/` — RCU Foundation (40 files, ~142 KB)

Cross-cutting foundation delivered in phases (v13.0 redesign):

**Snapshot / RCU / Publication:**
| File | Size | Role |
|---|---|---|
| `EpochDomain.h` | **32.2 KB** | 64 named reader slots, `globalEpoch` management, quiescent-state-based reader registration/tracking. |
| `RCUReader.h` | 8.8 KB | RAII reader epoch enter/exit. |
| `SnapshotCoordinator.{h,cpp}` | 13.3 KB | Thread-safe snapshot publication and fade. |
| `SnapshotFactory.{h,cpp}` | 8.4 KB | Snapshot creation and destruction. |
| `SnapshotAssembler.{h,cpp}` | 3.2 KB | Snapshot assembly pipeline. |
| `SnapshotSlotStore.h` | 2.3 KB | Slot-based atomic pointer storage for snapshot handles. |
| `SnapshotRetireManager.h` | 1.9 KB | Manages retirement of old snapshots after RCU grace period. |
| `SnapshotParams.h` | 1.7 KB | Parameter container for snapshot construction. |
| `SnapshotFadeState.h` | 5.3 KB | Crossfade state tracking for snapshot transitions. |
| `GlobalSnapshot.{h,cpp}` | 3.6 KB | Immutable snapshot base (DSP parameter container: EQ, gains, bypass, oversampling). |
| `RuntimeStore.h` | 3.6 KB | Internal store for `RuntimePublicationCoordinator`. |
| `RuntimeReaderContext.h` | 2.0 KB | Reader context (`ObserveChannel::Audio` / `Message` / `Publication`). |
| `RuntimePublicationCoordinator.h` | 5.6 KB | Template coordinator for atomic world publication. |
| `ObservedRuntime.h` | 2.4 KB | Observed runtime abstraction (token-based). |
| `ObserveChannel.h` | 3.3 KB | Observation channel classification. |
| `RebuildTypes.h` | 0.2 KB | Rebuild intent and classification types. |
| `Types.h` / `TimeUtils.h` | 1.8 KB | Common types, time measurement harness. |
| `EQParameters.h` | 1.7 KB | EQ parameter container. |
| `ConvolverRuntimeCompatTypes.h` | 0.1 KB | Runtime-compatible convolver type aliases. |

**Abstract Interfaces (Provider pattern):**
| File | Role |
|---|---|
| `IEpochProvider.h` | Abstract epoch provider interface. |
| `IPublicationProvider.h` | Abstract publication provider interface. |
| `IReaderEpochProvider.h` | Abstract reader epoch provider. |
| `IRetireProvider.h` | Abstract retire provider. |
| `IRetireRouter.h` | Abstract retire router interface. |

**Async Reclamation:**
| File | Role |
|---|---|
| `DeletionQueue.{h,cpp}` | Deferred object deletion queue. |
| `DeferredRetireFallbackQueue.h` | Overflow fallback for RetireRouter. |
| `WorkerThread.{h,cpp}` | Background snapshot worker thread. |
| `ThreadAffinityManager.h` | Thread affinity policy management. |
| `ThreadHash.h` | Thread hash computation utilities. |
| `CommandBuffer.h` | Non-blocking command dispatch. |
| `FadeEngine.h` | Fade computation engine. |

**Diagnostics & Utilities:**
| File | Role |
|---|---|
| `RetireBoundaryTelemetry.h` | Telemetry for retire boundary events. |
| `ScopedMXCSR.h` | RAII MXCSR state saver/restorer (FTZ/DAZ masking). |

### 3.6 `src/tests/` — CTest Regression (36 test executables, 55 source files, ~943 KB)

All tests registered via `add_test()` in CMakeLists.txt. Many are JUCE-independent.

| Test Executable (source) | KB | Focus |
|---|---|---|
| `ISRRuntimeIdentityTests` (`ISRRuntimeIdentityGeneratorsTests.cpp`) | 1.4 | UUID/Generation generator correctness. |
| `RuntimePublicationCoordinatorTests` | 6.2 | Coordinator template contract. |
| `PublicationAdmissionTests` | 8.6 | Publication admission evaluation. |
| `ISRSemanticValidationTests` | 150.4 | Semantic validation (schema v9). |
| `invariant_INV3_INV5Tests` (`invariant_INV3_INV5.cpp`) | 44.7 | INV-3 / INV-5 invariants. |
| `AdmissionPackedStateTests` | 22.0 | Admission packed-state access. |
| `RetireGraceSemanticsTests` | 29.7 | Grace period semantics. |
| `D8_1_WrapperCacheTests` | 7.4 | Wrapper cache (D8-1). |
| `D8_2_B_2_Tests` | 31.0 | D8-2-B-2 contract tests. |
| `TerminalTelemetryContractTests` | 8.5 | Terminal telemetry contract. |
| `RuntimeHealthMonitorTierTests` | 31.1 | Health monitor tier selection. |
| `ShutdownRetireIntentDrainTests` | 12.3 | Shutdown retire-intent drain. |
| `StuckReaderFallbackDrainTests` | 15.4 | Stuck-reader fallback drain. |
| `ISRSoakTests` | 22.0 | ISR soak / stress. |
| `OwnerChannelTests` | 8.4 | OwnerChannel SPSC transfer contract. |
| `NormalRetireDSPHandleCompareTests` | 6.4 | Normal-retire DSP handle comparison. |
| `RuntimeSemanticSchemaValidationTests` | 25.2 | Field/authority invariants. |
| `ObservePathSingleSourceTests` | 2.2 | Observe path single-source contract. |
| `OverlapAuthoritySingularTests` | 2.0 | Singular authority boundaries. |
| `ShadowCompareContractTests` | 1.2 | Shadow comparison contracts. |
| `CrossfadeExecutorLocalContractTests` | 2.3 | Crossfade executor-local contracts. |
| `RuntimeWorldAuthorityProjectionTests` | 12.9 | World authority projection invariants. |
| `PartialPublicationRejectTests` | 21.0 | Partial publication rejection (MKL-linked). |
| `RebuildAdmissionRegressionTests` | 3.2 | Rebuild admission regression. |
| `BuildInputSemanticContractTests` | 16.3 | Build input contract (large stack 8MB). |
| `BuildErrorClassificationTests` | 14.2 | BuildError / FailureClassification / RetryDisposition contract. |
| `RetrySchedulerTests` | 6.0 | RetryScheduler schedule/dispatch/collapse. |
| `DeferredDeletionQueueReclaimTests` | 28.9 | Deferred deletion queue reclaim with MPMC epoch stress. |
| `MpscBoundedRingTests` | 14.6 | MPSC bounded ring (Vyukov) contract. |
| `SequenceArithmeticTests` | 7.1 | Modular sequence arithmetic wraparound safety. |
| `DSPHandleTableTests` | 8.9 | DSPHandleTable O(1) map correctness. |
| `PriorityIntegrationTests` | 7.1 | Priority integration tests. |
| `GainStagingContractTests` | 14.8 | Auto gain staging contract (v14.0 Phase 8, JUCE/MKL independent). |
| `EQProcessorMaxGainTests` | 34.9 | EQ max gain response math contract (v14.0 Phase 8, JUCE/MKL independent). |
| `EQAnalysisUnitTests` | 39.2 | PeakEstimator/UpperBoundEstimator/AnalysisMerge/EQResponseSampler unit tests (v14.47, JUCE/MKL independent). |
| `FFTBackendTests` | 8.5 | FFTBackend abstraction (ProductionFft / TestFft). |
| `EQBoundExcessBenchmark` | 35.2 | boundExcessDb distribution benchmark (JUCE/MKL independent). |
| `MTNUPCMeasurement` (`MT-NUPC-Measurement.cpp`) | 8.7 | MT-NUPC measurement console app (Phase 1). |
| `AudioEngineHarness` (`src/tests/AudioEngineHarness/`) | ~7.1 | Audio engine integration harness. |

+ External CI: `HeadlessAudioPathVerification` (PowerShell, gated by `$CONVO_CI_BUILD`).

> CMake registers **40 `add_test()`** entries (some executables expose multiple named tests; `HeadlessAudioPathVerification` is PowerShell-gated).

### 3.7 `src/tools/` — Build Gate Scripts (2 files)

| File | Role |
|---|---|
| `build_identity_gate.py` | Build identity gate (pre-build verification). |
| `check_layout_offsets.py` | Struct layout / offset checker. |

### 3.8 `config/` — JSON Authority Manifests (4 files)

| File | Lines | Bytes | Role |
|---|---|---|---|
| `runtime_graph_baseline.json` | 5 | 93 | Baseline topology snapshot reference. |
| `publication_manifest.json` | 67 | 2,370 | Machine-readable publication inventory. |
| `authority_inventory.json` | 383 | 10,127 | Generated from `ISRRuntimeSemanticSchema.h` + `RuntimeGraph.h` + `AudioEngine.h`. Declares `Authoritative/Derived/Diagnostic` authority per field. |
| `pub_boundary_registry.json` | 66 | 2,678 | Publication-boundary registry (single source of publication transitions). |

Python verifiers in `tools/` (66 `.py` + 54 `.bat` scripts) cross-check source against these JSONs at build/commit time — guards against authority drift.

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
                                                                                 → RuntimePublishExecutor.executePublish()
                                                                                   → RuntimeWorldAuthority::commit()
┌─────────────────────┐
│  Timer Thread       │  Orchestrator.tick() (100ms), HealthMonitor.tick(),
│  (100ms polling)    │  Telemetry drain, Evidence emit, Retire reclaim
└─────────────────────┘

┌─────────────────────┐     DeferredDeletionQueue → tryReclaim()
│ ISRRetireRouter     │     → old DSPCore/EQState/ConvolverState released
│ + DeferredFreeThread│     → aligned_free() / delete
│ + RetireQuarantine  │     → (on enqueue-full) quarantine store, epoch-safe drain
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

Declared in `RuntimeState::kFieldDescriptors[21]` and `RuntimeState::kRuntimeAuthorityInventory[21]`, verified against `config/authority_inventory.json`.

### 5.2 ISR Architecture Layers

```
Layer 1    RuntimeGraph                 src/core/RuntimeGraph.h
             ├─ Active/fading node description
             └─ Contract: validateDecisionCoverageContract()

Layer 2    RuntimeState (sealed)        AudioEngine.h
             ├─ BuilderToken-protected construction
             ├─ 21 field descriptors, 21 authority inventory entries
             ├─ 10 read-authority inventory entries
             └─ Freeze/Seal for immutability post-publish

Layer 3    RuntimeBuilder               src/audioengine/RuntimeBuilder.h
             ├─ Only entity that can construct RuntimeState
             ├─ Populates all Semantic structs (topology, routing, execution, ...)
             └─ buildRuntimePublishWorld(dsp, oldDSP, policy, fadeSec, ...)

Layer 4    RuntimePublicationCoordinator  src/audioengine/ISRRuntimePublicationCoordinator.h
             ├─ publishWorld() — validates → commits → publishes
             ├─ ProcessIntent path (ISRRuntimePublicationCoordinator_ProcessIntent.cpp)
             └─ consumeWorldHandle() — RT read path (atomic observe)

Layer 5    RuntimePublicationValidator   src/audioengine/RuntimePublicationValidator.h
             └─ Validates schema/authority/topology/resource contracts

Layer 6    PublicationAdmission / RuntimePublishExecutor
             ├─ Admission.evaluate(healthState): generation check + shutdown + health
             ├─ RuntimePublishExecutor.executePublish(): sole gateway to
             │   RuntimeWorldAuthority::commit() on the Publish path
             └─ OwnerChannel: SPSC sole-ownership transfer across RT boundary

Layer 7    CrossfadeAuthority            src/audioengine/CrossfadeAuthority.h
             ├─ evaluate(oldWorld, newWorld, policy) from dspProjection
             └─ Decision { needsCrossfade, fadeTimeSec }

Layer 8    ISRShutdown (FSM)             src/audioengine/ISRShutdown.h
             ├─ Running → AudioStopped → ObserverDrained → RetireClosed
             │   → EpochSettled → ReclaimComplete → [EmergencyDrain]
             │   → VerifyDrained → TimedOut|Failed → ShutdownComplete
             ├─ BlockingReasonStats (alignas(64)) per-reason
             └─ ISRLifetimeProof: ShutdownQuiescenceProof / ReclaimPermit types

Layer 9    ISRRetire / ISRRetireRouter / OverflowRing / QuarantineStore
             ├─ Router: epoch-coordinated retire entry
             ├─ OverflowRing: hardware-safe false sharing isolation
             ├─ RetireQuarantineStore: fallback when enqueue is full (no UAF)
             └─ Grace period → reclaim: deferred deletion queue

Layer 10   RuntimeHealthMonitor + TelemetryRecorder + RuntimePolicyEngine + RetryScheduler
             ├─ Pull-based monitoring (27+ monitor references)
             ├─ ISRHealthState: Healthy → Degraded → Critical
             ├─ RecoveryAction hierarchy: Observe→Throttle→Recover→Restore→Safe→Critical
             ├─ BuildErrorPolicy: Permanent/Transient/Infrastructure/Fatal classification
             ├─ RetryScheduler: collapsed retry dispatch for Transient/Infrastructure
             └─ Telemetry/Evidence export (CI correlation)

Cross-layer    RuntimePublicationBridge + SequenceArithmetic
                 ├─ commit(PublishAuthority, RuntimeBoundary, newWorld, ver, seq, epoch, mappedGen)
                 ├─ retire(RetireAuthority, RuntimeBoundary, oldWorld)
                 └─ Modular sequence compare (wraparound-safe, RFC 1982)
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

ISR Publish Path (RuntimePublishExecutor):
  Intent payload (fixed at enqueue) → PublishExecutor.executePublish()
    → RuntimeWorldAuthority::commit() → PendingPublishRegistry lookup/unregister
    → onPublishCompleted → advanceRetireEpoch → completion-notify

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
- Owns `RuntimePublicationOrchestrator`, `RuntimeHealthMonitor`, `ISRRetireRouter`, `RuntimePublicationBridge`, `CrossfadeRuntime`, `EQCacheManager`, `WorkerThread`, `RetryScheduler`.
- Manages Adaptive Noise Shaper Learner lifecycle (start/stop learning, progress polling, error reporting).
- Manages MMCSS thread priority (via `AudioEngine.Mmcss.cpp`), state I/O (via `AudioEngine.StateIO.cpp`), and auto gain staging (via `AutoGainPlanner`).

### 6.2 EQProcessor

- 20-band parametric EQ in the real-time path using TPT SVF filters.
- RCU parameter updates via `uintptr_t`-backed atomic handles + `EpochDomain`.
- AGC (automatic gain control) with pre-computed attack/release/smooth coefficient tables.
- Nonlinear saturation via `fastTanh` approximation (AVX2).
- `EQCoeffCache` (RefCountedDeferred) for cross-snapshot coefficient sharing.
- Serial/Parallel filter structure with crossfade-able transition.
- Analysis subsystem: `PeakEstimator`, `UpperBoundEstimator`, `EQResponseSampler`, `AnalysisMerge`, `BandHelper`, `EQAnalysisMath`.

### 6.3 ConvolverProcessor

- IR-based convolution via Intel MKL Non-Uniform Partitioned Convolution (NUC).
- Asynchronous IR loading/rebuild on Worker Thread. `BuildSnapshot` + `StructuralHash` for rebuild decision.
- Configurable rebuild debounce (20 ms default, 10–3000 ms range).
- Crossfade-safe transitions (old/new DSP fade with latency compensation).
- Phase modes: As-Is / Mixed / Minimum. Mixed-phase uses `AllpassDesigner` (CMA-ES).
- Tail modes: AirAbsorption / LayerTailContouring / Bypass.
- Progressive FFT upgrade (background thread).
- Split across 8 TUs (`src/convolver/`) for compile-time efficiency; legacy monolithic `MKLNonUniformConvolver.cpp` retained for backward compat.
- FFT abstraction: `ConvolverBuilder` (Non-RT Plan factory) → `FFTBackend` (IPP ProductionFft / TestFft) → `FFTExecutionContext` (const Plan& consumer).

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
| **RetryScheduler Thread** | Collapsed retry dispatch for Transient/Infrastructure build failures | |

### Thread-Safe Communication

| Pattern | Mechanism | Used For |
|---|---|---|
| RCU (Read-Copy-Update) | `EpochDomain` (64 slots) + `RCUReader` | EQ parameters, Convolver IR, NoiseShaper coefficients, RuntimeWorld |
| Atomic publish/consume | `publishAtomic` / `consumeAtomic` / `compareExchangeAtomic` | All scalar parameters (bypass, gain, order, mode, etc.) |
| Lock-Free SPSC Ring | `LockFreeRingBuffer<T,N>` | DiagEvent (512), XRunEvent, AudioBlock (4096) |
| Lock-Free MPSC Ring | `MpscBoundedRing` | Intent queue (Builder/Rebuild + Timer + CoordinatorLoop producers) |
| Lock-Free Audio FIFO | `LockFreeAudioRingBuffer` | Spectrum analyzer (FIFO_SIZE = 1M samples) |
| Owner Transfer | `OwnerChannel` | Sole ownership of RuntimeState across RT boundary |
| Deferred Deletion | `DeferredDeletionQueue` + `DeferredFreeThread` | Old DSPCore, EQState, BandNode after grace period |
| Quarantine Fallback | `RetireQuarantineStore` | Objects that cannot be enqueued (queue full) |

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
   - Build failures are classified via `BuildErrorPolicy` and scheduled for retry when `RetryDisposition` allows.

3. **Publish phase (ISR pipeline)**
   - `RuntimePublicationOrchestrator` executes admission evaluation (`PublicationAdmission::evaluate`):
     - Check HealthState (Healthy/Degraded/Critical)
     - Check shutdown state
     - Check generation monotonicity
     - Check structural hash equivalence
   - If accepted: `RuntimeBuilder.buildRuntimePublishWorld()` → `RuntimePublicationCoordinator.publishWorld()`.
   - `CrossfadeAuthority` determines crossfade type (smooth/hard reset).
   - `RuntimePublishExecutor` commits via `RuntimeWorldAuthority` on the ISR path.

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
| Build System | **CMake** with Ninja Multi-Config (3.22+) |
| Math Acceleration | **Intel oneMKL** (sequential, LP64, static) + **Intel IPP** |
| CRT | **Static** (`/MT` Release, `/MTd` Debug; switches to `/MDd` when `ENABLE_ASAN=ON`) |

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
| `CONVOPEQ_ENABLE_CLANG_TIDY` | OFF | Build-time clang-tidy analysis (auto-ON in CI builds) |
| `CONVOPEQ_REQUIRE_MKL` | ON | Require Intel MKL for MSVC builds (OFF → system aligned allocator) |
| `CONVOPEQ_ENABLE_ISR_TESTS` | ON | CTest regression suite (36 test executables / 40 tests) |
| `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` | OFF | Runtime diagnostic logging (XRUN/MEM/VERIFY/WORLD/etc.) |
| `ENABLE_ASAN` | OFF | AddressSanitizer (Debug only, forces /MDd; mutually exclusive with PGO and TSAN) |
| `ENABLE_TSAN` | OFF | ThreadSanitizer (Clang only; mutually exclusive with ASAN) |
| `CONVOPEQ_PGO_INSTRUMENT` | OFF | PGO instrumentation (1st pass: /GENPROFILE) |
| `CONVOPEQ_PGO_USE` | OFF | PGO optimized build (2nd pass: /USEPROFILE) |

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

- `JUCE/` (JUCE 8.0.12)
- `r8brain-free-src/` (r8brain sample-rate converter)
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
- Intent queue producers are multi-threaded; use `MpscBoundedRing` (not SPSC `LockFreeRingBuffer`) for any queue with >1 producer.
- Sequence / epoch comparisons must use `SequenceArithmetic.h` helpers, never raw `<` on `uint64_t` counters.

---

## 14. Architectural Summary (Current)

ConvoPeq v0.6.10 uses a five-layer architecture:

```
┌────────────────────────────────────────────────────────────────┐
│                          UI Layer                               │
│  MainWindow, EQControlPanel, ConvolverControlPanel,             │
│  SpectrumAnalyzerComponent, DeviceSettings                      │
├────────────────────────────────────────────────────────────────┤
│                       Engine Layer                              │
│  AudioEngine (orchestration, state lifecycle, audio I/O bridge) │
│  AutoGainPlanner, OversamplingPolicy, SimplePeakLimiter,        │
│  BuildErrorPolicy, RetryScheduler                               │
├────────────────────────────────────────────────────────────────┤
│                       ISR Layer                                 │
│  RuntimePublicationOrchestrator, RuntimeHealthMonitor,           │
│  RuntimePolicyEngine, ISRShutdown, ISRRetireRouter,              │
│  CrossfadeAuthority, RuntimeBuilder, Publication Pipeline,       │
│  RuntimePublishExecutor, RuntimeWorldAuthority, OwnerChannel,    │
│  RetireQuarantineStore, ISRDSPQuarantine, ISREvidenceExporter,   │
│  FrozenRuntimeWorld, SequenceArithmetic, MpscBoundedRing         │
├────────────────────────────────────────────────────────────────┤
│                        DSP Layer                                 │
│  EQProcessor (20-band TPT SVF), ConvolverProcessor (MKL NUC),  │
│  ConvolverBuilder / FFTBackend / FFTExecutionContext,           │
│  CustomInputOversampler (AVX2 FIR/IIR), OutputFilter (Biquad),  │
│  NoiseShaperLearner (CMA-ES), PsychoacousticDither (TPDF),      │
│  TruePeakDetector (4x OS), LoudnessMeter (BS.1770)              │
├────────────────────────────────────────────────────────────────┤
│                       Core Layer                                 │
│  EpochDomain (64-slot RCU), SnapshotCoordinator,                  │
│  RuntimeStore, DeferredDeletionQueue, AlignedAllocation,          │
│  IEpochProvider (Provider pattern), RetireBoundaryTelemetry,      │
│  ScopedMXCSR, FadeEngine, CommandBuffer                           │
└────────────────────────────────────────────────────────────────┘
```

Design focus:

- **Strict real-time safety**: Audio Thread prohibitions enforced via Firewall (`ISRRTExecution`), zero-allocation path verification.
- **Asynchronous state construction**: Always on Worker/Message Threads; published atomically via ISR Publication Pipeline.
- **All inter-thread data transfer**: lock-free/RCU/atomic patterns; no mutex on Audio Thread.
- **All large buffers**: 64-byte aligned (`aligned_malloc` / MKL `DftiMalloc` / `PFFFT`).
- **Authority governance**: `authority_inventory.json` + `pub_boundary_registry.json` + Python verifiers → compile-time authority contract enforcement.
- **Dependency directories** (`JUCE/`, `r8brain-free-src/`): strictly read-only.
- **Schema versioning**: `ISRRuntimeSemanticSchema.h` schema v9 + `RuntimeState::kFieldDescriptors[21]` + `RuntimeState::validateDescriptorSet()` → contract compiler-time verified.
- **Build resilience**: `BuildErrorPolicy` classification + `RetryScheduler` collapsed retry for Transient/Infrastructure failures.
- **Ownership transfer**: `OwnerChannel` (SPSC sole-ownership) + `RuntimeWorldAuthority::PendingPublishRegistry` for the async enqueue→commit gap.

---

*Version: v0.6.10 (Updated 2026-09-10)*
*Compiler: MSVC 19.44+ / Intel icx 2026.0*
*Platform: Windows 11 x64*
*JUCE: 8.0.12*
*MKL: oneAPI sequential (static)*
