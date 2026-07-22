
# ConvoPeq   - powered by Vibe-Coding

---

## New in v0.6.10

- Enhanced Auto-Gain Functionality: Automatic gain calculation logic based on processing order (EQ/Convolver standalone, Conv→EQ, EQ→Conv) was added, along with heuristic-based Q-surge margins and safety margin calculations. A new `AutoGainPlanner` class was introduced to design the input/output gain auto-adjustment algorithm as a purely functional planner, improving clamping ranges and net 0dB alignment.
- Added Deferred Deletion Queue Reclaim Test: A dedicated 745-line test suite was added to verify the behavior of `DeferredDeletionQueue::reclaim()`. This suite comprehensively validates RCU mechanism safety, covering epoch progression, FIFO order guarantees, concurrent enqueue/reclaim operations, MPMC epoch correctness, and state invariants during reclamation.
- Improved ISR Runtime Governance: Enhancements were made to the ISR (Intelligent State Reconstruction) publication adjuster, retirement router, and runtime builder. Capabilities for state transition monitoring, publication behavior verification, and graph consistency checks were expanded; additionally, bug fixes for `RuntimeWorldAuthorityProjection` and fade state management using `fadingRuntimeUuid` were implemented.
- Enhanced Thread Affinity Management: The `ThreadAffinityManager` improved CPU affinity mask management for audio threads, adding support for the `AudioRealtime` thread type and dedicated mask settings introduced in "Work 64." Thread priority management on Windows was improved through MMCSS priority application capabilities and the `tryApplyMmcssForSelfManagedThread()` method.
- Eliminated Real-Time Blockers in Audio Threads: Non-real-time operations within audio threads (such as `Logger::writeToLog()`, `std::hash`, and the `GetCurrentProcessorNumber()` syscall) were eliminated or conditionally compiled out. Issues involving CRT function calls and false sharing were resolved; furthermore, interrupt-free processing for real-time threads was ensured by making `ScopedNoDenormals` thread-local and enhancing the safety of atomic operations.

ConvoPeq is a high-fidelity standalone audio processor for Windows 11 x64, combining IR convolution and a 20-band parametric EQ with a real-time analyzer.

## Overview

ConvoPeq v0.6.10 is built with JUCE 8.0.12 and is designed for low-latency, real-time-safe operation on Windows. All DSP runs in 64-bit double precision with AVX2 acceleration, backed by Intel oneMKL and IPP.

| Aspect | Detail |
|--------|--------|
| Platform | **Windows 11 x64** (standalone, not a plugin) |
| Framework | **JUCE 8.0.12** |
| Compilers | **MSVC 19.44+** (VS2022 17.11+ or VS2026) / **Intel icx** (oneAPI 2026.0) |
| DSP Precision | **64-bit double** throughout the main processing path |
| SIMD / Math | **AVX2** + **Intel oneMKL** (sequential, static link) + **Intel IPP** |
| Build System | **CMake 3.22+** + **Ninja Multi-Config** |
| Language | **C++20** |
| Source | **277 files** (~3.17 MB) across `src/` + 21 test files

---

## Documentation

- [README.md](README.md): User-facing overview, features, audio processing summary, and build entry points
- [ARCHITECTURE.md](ARCHITECTURE.md): Developer-facing architecture, threading, state flow, and subsystem design details
- [SOUND_PROCESSING.md](SOUND_PROCESSING.md): **In-depth, code-referenced technical documentation of the entire audio signal processing flow.**
  - Covers all DSP stages (input, conditioning, oversampling, main DSP chain, output, dither, etc.)
  - Includes mathematical formulas, buffer/parameter management, SIMD/real-time safety, and code path examples
  - Intended for international contributors and advanced users seeking a rigorous technical reference
- [BUILD_GUIDE_WINDOWS.md](BUILD_GUIDE_WINDOWS.md): Windows build instructions and troubleshooting
- [HOW_TO_USE.md](HOW_TO_USE.md): Practical usage guide — room correction via IR convolution with REW, and headphone EQ correction via AutoEq

### ISR Design & Audit Docs ([doc/work/](doc/work/))

The `doc/work/` directory contains ~82 ISR design documents covering the full ISR development lifecycle (Phases 0–4, Bridge Runtime, Rebuild Admission, EpochDomain migration, Shutdown State Machine, and formal compliance audits). Key entry points:

- [doc/work/ISR_Rebuild_Admission_最終計画書_2026-05-23.md](doc/work/ISR_Rebuild_Admission_最終計画書_2026-05-23.md): 実装計画と受入基準（8章）
- [doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md](doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md): 実測ログ付きのクローズ判定台帳
- [doc/work/R11-R25_Closed判定監査表_2026-05-21.md](doc/work/R11-R25_Closed判定監査表_2026-05-21.md): R11〜R25監査の正本

### Companion Analysis

- [doc/sourcecode_analysis_2026-07-03.md](doc/sourcecode_analysis_2026-07-03.md): Complete source code structure analysis (277 files, 25 sections, all data flows)

### Manuals ([manual/](manual/))

**English Manuals:**

- [manual/MAIN_WINDOW_EN.md](manual/MAIN_WINDOW_EN.md): Main window usage (English)
- [manual/AUDIO_SETTINGS_WINDOW_EN.md](manual/AUDIO_SETTINGS_WINDOW_EN.md): Audio settings window (English)
- [manual/IR_ADVANCED_WINDOW_EN.md](manual/IR_ADVANCED_WINDOW_EN.md): IR advanced window (English)
- [manual/ADAPTIVE_NOISE_SHAPER_LEARNING_WINDOW_EN.md](manual/ADAPTIVE_NOISE_SHAPER_LEARNING_WINDOW_EN.md): Adaptive Noise Shaper Learning (English)

**Japanese Manuals (in Japanese):**

- [manual/MAIN_WINDOW_JP.md](manual/MAIN_WINDOW_JP.md): Main window usage (Japanese)
- [manual/AUDIO_SETTINGS_WINDOW_JP.md](manual/AUDIO_SETTINGS_WINDOW_JP.md): Audio settings window (Japanese)
- [manual/IR_ADVANCED_WINDOW_JP.md](manual/IR_ADVANCED_WINDOW_JP.md): IR advanced window (Japanese)
- [manual/ADAPTIVE_NOISE_SHAPER_LEARNING_WINDOW_JP.md](manual/ADAPTIVE_NOISE_SHAPER_LEARNING_WINDOW_JP.md): Adaptive Noise Shaper Learning (Japanese)

### Directory Map (ASCII Tree)

```text
ConvoPeq/
├── src/                       # Main C++ source (277 files, ~3.17 MB)
│   ├── audioengine/           # ISR runtime governance, AudioEngine split TU (107 files)
│   │   ├── AudioEngine.Processing.*.cpp  # Audio thread core (DSPCore, Block, Latency)
│   │   ├── AudioEngine.*.cpp            # Lifecycle, Timer, Commit, Rebuild, Retire
│   │   └── ISR*.cpp                     # Closure, HB, Shutdown, Publication, Retire, etc.
│   ├── eqprocessor/           # 20-band EQ split + Analysis Subsystem (17 files)
│   │   ├── EQProcessor.{Core,Coefficients,Parameters,Processing,ProcessingCache}.cpp
│   │   └── PeakEstimator, UpperBoundEstimator, EQResponseSampler, BandHelper,
│   │        AnalysisMerge, EQAnalysisMath, EQAnalysisTypes (analysis subsystem)
│   ├── convolver/             # Convolution split TU (10 files)
│   │   └── ConvolverProcessor.{Lifecycle,Rebuild,LoaderThread,LoadPipeline,
│   │                           MixedPhase,ResampleAndFallback,StateAndUI,Internal}.cpp
│   ├── core/                  # RCU snapshot foundation (40 files)
│   │   ├── EpochDomain.h      # 64-slot named reader domain (26 KB)
│   │   ├── RCUReader.h, SnapshotCoordinator, GlobalSnapshot, DeletionQueue, FadeEngine
│   │   ├── WorkerThread, Types, CommandBuffer, SnapshotSlotStore, SnapshotRetireManager
│   │   └── IEpochProvider, IPublicationProvider, IRetireRouter (Provider pattern)
│   └── tests/                 # CTest regression suite (21 files)
├── config/                    # JSON authority manifests (4 files)
├── tools/                     # CodeGraph, CodeQL, CI verification scripts
├── doc/                       # Architecture work docs (~82 ISR design files)
│   ├── work/                  # ISR design, audit, compliance
│   └── sourcecode_analysis_2026-07-03.md  # Full source analysis
├── manual/                    # User manuals (EN/JP, 4 topics each)
├── .github/                   # CI workflows, ISR policies, scripts, prompts
├── .vscode/                   # tasks.json (22 tasks), launch.json (5 configs), mcp.json
├── resources/                 # App resources (icons, assets)
├── sampledata/                # Sample IR/EQ files
├── JUCE/                      # JUCE 8.0.12 framework source (in-tree)
├── r8brain-free-src/          # IR resampler (external dependency)
├── CMakeLists.txt             # Build configuration (1042 lines, v0.6.10)
├── CMakePresets.json          # 3 configure + 2 build presets
├── build.bat                  # Primary build script (MSVC + icx)
├── ProjectMetadata.cmake      # App name, version (v0.6.10), company
├── README.md                  # This file
├── ARCHITECTURE.md            # Architecture & ISR governance (v0.6.10)
├── SOUND_PROCESSING.md        # Complete signal processing guide
├── BUILD_GUIDE_WINDOWS.md     # Windows build instructions
└── HOW_TO_USE.md              # Practical usage guide (REW + AutoEq)
```

---

## Key Features

### Core DSP
- **20-band parametric EQ** (`EQProcessor`, split TU: 17 files in `src/eqprocessor/` incl. analysis subsystem)
  - TPT (Topology-Preserving Transform) State Variable Filters per band
  - `EQBandType`: LowShelf / Peaking / HighShelf / LowPass / HighPass
  - `EQChannelMode`: Stereo / Left / Right / Mid / Side (M/S processing)
  - Filter structure: **Serial** (default) or **Parallel**
  - Auto Gain Control (AGC): attack 0.2 s, release 2.0 s, smooth 0.2 s
  - Nonlinear saturation via `fastTanh` rational approximation (no libm)
  - `EQCoeffCache`: refcounted shared coefficient cache (v2.3)
- **IR convolution** (`ConvolverProcessor`, split TU: 10 files in `src/convolver/`)
  - Intel MKL **Non-Uniform Partitioned Convolution (NUC)** engine
  - Phase modes: **As-Is / Minimum / Mixed** with tunable transition (`f1`, `f2`, `tau`)
  - IR loading on dedicated background `LoaderThread` — no audio thread blocking
  - RCU (Read-Copy-Update) pattern for glitch-free IR handoff
  - Legacy `MKLNonUniformConvolver.cpp` retained for backward compatibility
- **Runtime-selectable processing order**: EQ→Convolver or Convolver→EQ
- **Input oversampling**: 2×/4×/8× via `CustomInputOversampler` (IIRLike / LinearPhase presets)
- **Output conditioning**: `OutputFilter` (HCF/LCF conditional on final processor), musical soft clipping with `fastTanh` (AVX2 vectorized), makeup gain

### Noise Shaping & Dithering
- **PsychoacousticDither**: 12th-order error-feedback (GUI: "9th-order"), MKL VSL RNG + TPDF, `kCoeffTable[6][3][12]` per sample rate and bit depth
- **FixedNoiseShaper**: 4th-order error-feedback, psychoacoustically tuned coefficients
- **Fixed15TapNoiseShaper**: 16th-order error-feedback (class name "15Tap" for legacy consistency, ORDER = 16)
- **Adaptive 9th-order** (`NoiseShaperLearner`, 68 KB): lattice-ladder noise shaper with CMA-ES optimization on a dedicated worker thread, RCU coefficient handoff, per-(sample rate, bit depth, mode) banks, 6 learning modes (Shortest–Ultra), converge in 5–160 minutes

### Analysis & Metering
- **Real-time spectrum analyzer** with EQ overlay (`SpectrumAnalyzerComponent`, 52 KB)
- **LoudnessMeter**: ITU-R BS.1770-4/5 K-weighting (2-stage biquad), lock-free ring buffer publish to worker thread
- **TruePeakDetector**: 4× oversampled true peak (63-tap linear phase FIR, ITU-R BS.1770-3)
- **DC blocking**: two-stage IIR per `UltraHighRateDCBlocker` (input + post-upsampling)

### I/O & Device Support
- Standalone runtime with **ASIO / WASAPI / DirectSound** device support
- Persistent device settings (`device_settings.xml`)
- `AsioBlacklist.h` for known broken ASIO drivers

### ISR Runtime Governance (107 files in `src/audioengine/`)
- **RCU + atomic** parameter handoff (`publishAtomic` / `consumeAtomic` primitives)
- **EpochDomain** (64 named reader slots) + `RCUReader` RAII pattern
- Publication choreography: `ISRRuntimePublicationCoordinator`, `PublicationAdmission`, `PublicationExecutor`
- Retire pipeline: `DSPLifetimeManager` → `ISRRetireRouter` → `DeletionQueue`
- Crossfade governance: `CrossfadeAuthority` / `CrossfadeRuntime` (Authority pattern)
- Health monitoring: `RuntimeHealthMonitor`, `RuntimePolicyEngine`
- Deferred garbage collection: `DeferredDeletionQueue`, `RefCountedDeferred`, `DeferredFreeThread`

### Build & Test Infrastructure
 - **CTest regression suite**: 21 test executables (ISR identity, publication coordinator, semantic validation, grace semantics, etc.)

---

## Audio Processing Method

This section is a user-facing summary of the current block processing strategy. For in-depth, code-referenced technical documentation (DSP stages, mathematical formulas, SIMD/real-time safety, code path examples), see `SOUND_PROCESSING.md`. For subsystem-level architectural details, threading model, RCU patterns, and ISR governance, see `ARCHITECTURE.md`.

### 1) Quality-Oriented Design Principles

ConvoPeq is designed to preserve fidelity under real-time conditions.

- **64-bit double-precision DSP** throughout the main processing path — reduces cumulative rounding error across cascaded stages.
- **All heavy preparation is moved off the audio thread**: filter coefficient computation (`std::sin`/`std::cos`), IR loading/resampling, noise shaper coefficient learning (CMA-ES), and state validation occur on message/worker threads.
- **SIMD (AVX2/FMA) + Intel oneMKL acceleration** are used where throughput matters: `scaleBlockFallback`, `softClipBlockAVX2`, FIR convolution in oversampling stages, MKL DFTI for IR convolution, VSL RNG for dither generation.
- **Transition-safe state changes**: all parameter handoff uses atomic publish/consume or RCU (Read-Copy-Update) with `EpochDomain` — no clicks, zipper noise, or abrupt latency jumps.

### 2) Block Entry and State Snapshot

For each callback block, `AudioEngine::processBlockDouble()` enters the audio thread with:

```cpp
AudioCallbackRuntimeScope (lifecycleToken + firewallToken)
ScopedNoDenormals              // flush denormals to zero
ThreadRole::AudioRealtime      // numeric policy verification
```

The `ProcessingState` snapshot is assembled from atomic reads and RCU-published shared state at block start. The entire block is then processed using only this **read-only, consistent snapshot** — no blocking operations, no parameter races, no allocation.

### 3) Main DSP Chain

Typical logical flow (exact sequence from `SOUND_PROCESSING.md`):

```
Input → Headroom Gain → DC Block → Oversampling (optional) →
[EQ ↔ Convolver] (order selectable) →
OutputFilter (HCF/LCF or HPF/LPF conditional) →
Soft Clip (musical, fastTanh, AVX2) → Makeup Gain →
Dither/Noise Shaping → Downsampling (if OS) → Output
```

- **Order is runtime-selectable**: EQ→Convolver or Convolver→EQ.
- Oversampling factor: 1×/2×/4×/8× (IIRLike or LinearPhase preset).
- Input headroom gain and output makeup gain are AVX2-optimized, gain values pre-converted to linear in the message thread (no `std::pow` on audio thread).
- Convolver input trim is applied only when processing order is **EQ→Convolver** and both processors are active.

### 4) Convolution Strategy

`ConvolverProcessor` (split TU, 10 files in `src/convolver/`) uses asynchronous IR preparation and RCU-based safe handoff:

- **IR load/rebuild** is handled by a dedicated `LoaderThread` (`std::thread`) — audio thread never blocks on file I/O or IR resampling.
- **Rebuild requests are debounced** (`REBUILD_DEBOUNCE_DEFAULT_MS`) to reduce burst load from rapid parameter changes.
- **RCU handoff**: new IR state atomically swapped; audio thread continues with old IR without interruption. Old engines retired via `ISRRetireRouter` → `DeletionQueue`.
- **Latency retargeting** is hysteresis-controlled to avoid frequent retriggers.
- User-facing controls are debounced to avoid unnecessary rebuild pressure during UI dragging.

Convolution algorithm: **Intel MKL NUC (Non-Uniform Partitioned Convolution)** with non-uniform block partitioning (shorter blocks near IR start for low latency, longer blocks toward the tail for efficiency).

Phase modes: **As-Is / Minimum / Mixed**. Mixed mode blends linear-phase (low frequencies) with minimum-phase (high frequencies) per `mixedTransitionStartHz` / `mixedTransitionEndHz` / `tau`.

### 5) EQ Strategy

`EQProcessor` (split TU, 17 files in `src/eqprocessor/`) applies per-band parametric filtering in real time:

- **20-band TPT (Topology-Preserving Transform) SVF** based on Vadim Zavalishin's "The Art of VA Filter Design" — smooth parameter modulation, low noise.
- **RCU-based coefficient handoff**: UI/worker thread creates new `EQState` / `BandNode`, publishes via `publishAtomic(currentStateBits, ...)`. Audio thread loads via `loadCurrentState(acquire)`.
- **EQCoeffCache**: refcounted shared coefficient cache — multiple snapshots share coefficients when parameters are identical.
- **Nonlinear saturation**: `fastTanh` rational polynomial approximation applied per-sample within the SVF loop (no `libm`, branchless for |x| < CLIP_THRESHOLD).
- **AGC** (Auto Gain Control): envelope-tracking automatic gain adjustment with precomputed attack/release/smooth coefficient lookup tables.
- **M/S (Mid-Side) processing**: supports per-band Left/Right/Mid/Side channel modes via `kFilterChannels=4`.

### 6) Oversampling, Output Conditioning, and Finalization

Additional quality-oriented stages applied around the core EQ/convolution chain:

- **Input oversampling** (`CustomInputOversampler`): 2×/4×/8× via multi-stage Kaiser-windowed FIR interpolation, AVX2/FMA SIMD, denormal flush. DC blocker applied after upsampling.
- **Output filtering** (`OutputFilter`): conditional on final processor — HCF+LCF when convolver is last (4th-order Butterworth/LR, 18–22 kHz), or HPF+LPF when EQ is last (Butterworth 2nd, 19–24 kHz). All coefficients precomputed in message thread; audio thread performs only biquad SOS evaluation (no `libm`).
- **Soft clipping** (`softClipBlockAVX2`): piecewise musical clip (linear → knee `t²(3−2t)` → `fastTanh` saturation), AVX2 vectorized (4 doubles/IP), `prevSampleInOut` scalar feedback across blocks. Anti-inter-sample-peak protection via smooth knee transition.
- **Dither/noise shaping**: 4 types (Psychoacoustic 12th-order / Fixed 4th-order / Fixed 15th-order / Adaptive 9th-order with CMA-ES). All TPDF dither, 64-byte aligned, lock-free audio thread path. RNG ring buffer pre-filled, worker thread refills.

### 7) Analyzer Path

Analyzer data is decoupled from output audio:

- Audio thread pushes analyzer source data to `LockFreeRingBuffer` (SPSC, 64-byte aligned, atomic head/tail).
- UI timer reads FIFO and runs FFT visualization (`SpectrumAnalyzerComponent`).
- Analyzer update rate is adaptive by state (active/disabled/hidden) to limit UI-thread load.

This separation ensures visualization quality does not compromise audio-thread safety.

### 8) Latency Reporting

Latency display uses a unified breakdown model from `AudioEngine::getCurrentLatencyBreakdown()`:

- **Oversampling latency**: base-rate estimated from FIR tap counts per stage (IIRLike: 511/127/31, LinearPhase: 1023/255/63).
- **Convolver algorithm latency** + **IR peak latency**: reported from `ConvolverProcessor::getLatencyBreakdown()`.
- **SoftClip local OS latency**: 15 base-rate samples (31-tap Halfband, 2 passes).
- All values reported in both `ms` and `samples` from `totalLatencyBaseRateSamples`.

### 9) State Persistence

Two persistence paths:

- **Auto-save** (`device_settings.xml`): device state + compact runtime settings (dither bit depth, oversampling factor/type, input headroom, output makeup).
- **Manual preset XML** (Save/Load Preset): full `AudioEngine` + `EQ` + `Convolver` state including phase/mixed parameters, Auto/Manual IR-length state.

### 10) Real-Time Safety Rules

The audio callback path avoids:

- file I/O, blocking locks/waits,
- heavy runtime allocations (`malloc`/`new`/`resize`),
- `libm` calls (`std::sin`/`std::cos`/`std::pow`/`std::log`/`std::exp`),
- exceptions, UI thread interactions,
- `std::condition_variable` waits.

Buffers and heavy state are prepared outside the callback (`prepareToPlay()`). Old resources are garbage collected asynchronously (`DeferredDeletionQueue`, `RefCountedDeferred`, `DeferredFreeThread`).

In practice, ConvoPeq aims for both:

- **high sound quality**: double-precision DSP, long-form convolution (MKL NUC), oversampling, TPT SVF EQ, psychoacoustic noise shaping, ITU-R BS.1770 metering, and
- **stable real-time behavior**: asynchronous preparation, debounce, RCU state handoff, ISR runtime governance, and callback-safe processing boundaries.

---

## Project Scope

- Standalone desktop application
- Windows-only runtime target
- Real-time audio processing with separate UI/analyzer pipeline

## Build Requirements

1. **Visual Studio 2022 (17.x) or 2026 (18.x)** with *Desktop development with C++* workload
   - Alternatively: **Intel icx** compiler (oneAPI 2026.0) — no VS required for icx mode
2. **CMake 3.22+**
3. **Ninja** (build system, used via `Ninja Multi-Config` generator)
4. **Intel oneAPI Base Toolkit** (MKL) — required for both MSVC and icx
5. Local `JUCE/` directory (JUCE 8.0.12 expected, containing `JUCE/CMakeLists.txt`)

For full instructions including PGO, icx compiler flags, CTest suite, ASan, and troubleshooting, see `BUILD_GUIDE_WINDOWS.md`.

---

## Quick Build

**Recommended**: use `build.bat` from the repository root:

```cmd
build.bat Release              # MSVC Release (default)
build.bat Debug                # MSVC Debug
build.bat Release icx          # Intel icx Release
build.bat Debug   icx          # Intel icx Debug
build.bat Release clean        # Clean + build
build.bat Release pgo-gen      # MSVC PGO instrumentation
build.bat Release pgo-use     # MSVC PGO optimization
```

What `build.bat` does:

1. Auto-detects Visual Studio via `vswhere` (or falls back to known VS17/VS18 paths)
2. Calls `vcvarsall.bat x64` (MSVC mode) — skipped for icx
3. Calls Intel `setvars.bat intel64` (both MSVC and icx)
4. Configures CMake with `Ninja Multi-Config` generator
5. Builds selected configuration
6. Retries once on RC1109 (common icx first-build issue)

**Output binaries:**

| Compiler | Build Dir | Binary |
|----------|-----------|--------|
| MSVC Debug | `build/` | `build\ConvoPeq_artefacts\Debug\ConvoPeq.exe` |
| MSVC Release | `build/` | `build\ConvoPeq_artefacts\Release\ConvoPeq.exe` |
| icx Debug | `build-icx/` | `build-icx\ConvoPeq_artefacts\Debug\ConvoPeq.exe` |
| icx Release | `build-icx/` | `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe` |

MSVC and icx use **completely separate build directories** — both can be kept simultaneously.

### Manual Build (MSVC Equivalent)

```cmd
call "C:\Program Files\Microsoft Visual Studio\[2022|2026]\VC\Auxiliary\Build\vcvarsall.bat" x64
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
cmake -S . -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cmake --build build --config Debug
```

For `CMakePresets.json` usage, VS Code task reference, CTest suite commands, and full icx configuration, see `BUILD_GUIDE_WINDOWS.md`.

## Notes

- Standalone app target (not a plugin target) — **Windows 11 x64 only**
- Default daily workflow: `build.bat` or VS Code tasks (22 tasks in `.vscode/tasks.json`)
- MSVC and icx build directories are fully isolated (`build/` vs `build-icx/`) — can coexist
- PGO (Profile-Guided Optimization) is MSVC-only; not supported for icx
- RNG ring buffer for dither is pre-filled by worker thread — no RNG generation on audio thread
- All coefficients (EQ SVF, filter biquad, AGC tables, noise shaper) precomputed in message thread
 - 21 CTest regression tests available (`cmake --build build --config Debug && cd build && ctest -C Debug`)
- Do not modify external dependency trees directly:
  - `JUCE/`
  - `r8brain-free-src/`

## License

- **ConvoPeq**: Copyright (c) 2024-2025 lonewolf-jp — source available, see `ProjectMetadata.cmake`
- **JUCE**: GPLv3 / Commercial
- **r8brain-free-src**: MIT
- **Intel oneMKL**: Intel Simplified Software License

![ConvoPeq screenshot](./image260315_214309.png)
