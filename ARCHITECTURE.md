# ConvoPeq Architecture (v0.4.4)

This document describes the current architecture of **ConvoPeq**, a Windows-only standalone audio application built with **JUCE 8.0.12**.

---

## 1. System Goals and Non-Functional Requirements

ConvoPeq is designed around four primary goals:

1. **Real-time safety**
   - The audio callback must avoid blocking operations.
   - No file I/O, no UI access, and no heavy reconfiguration in the real-time path.
   - Processing-critical memory is prepared before use.

2. **Audio quality**
   - Main DSP path is based on **64-bit double precision**.
   - High-quality convolution and parametric EQ are central processing elements.
   - Output conditioning and dither stages are included for final signal quality.

3. **Performance**
   - Optimized for Windows x64 and AVX2-capable CPUs.
   - Uses Intel oneMKL for performance-sensitive numerical workloads.
   - Alignment-aware allocation is used for SIMD/MKL efficiency.

4. **Operational robustness**
   - Clear separation between UI/control logic and audio processing logic.
   - Heavy tasks (e.g., IR load/preparation) are handled asynchronously.
   - Device compatibility handling includes ASIO blacklist support.

---

## 2. Source-Level Architecture (`src/`)

## 2.1 Application / UI Layer

- `MainApplication.h/.cpp`
  - JUCE application entry/lifecycle.
  - Creates and manages top-level window lifetime.

- `MainWindow.h/.cpp`
  - Main desktop window composition.
  - Hosts and wires control panels and global controls.
  - Bridges user actions to `AudioEngine`.

- `EQControlPanel.h/.cpp`
  - EQ-related controls and user parameter editing.
  - Dispatches control changes to engine/EQ processor.

- `ConvolverControlPanel.h/.cpp`
  - Convolver-related controls (IR operations, bypass/mix/order dependent controls).

- `SpectrumAnalyzerComponent.h/.cpp`
  - Real-time visualization component.
  - Reads analyzer-side data path and renders spectrum.

## 2.2 Engine / Runtime Coordination Layer

- `AudioEngine.h/.cpp`
  - Primary orchestration layer for audio modules.
  - Owns/coordinates EQ processor, convolver processor, utility DSP stages, and runtime options.
  - Exposes user-facing operations to UI (bypass/order/settings/quality toggles).

- `AudioEngineProcessor.h/.cpp`
  - Adapter between JUCE audio callback and `AudioEngine`.
  - Pulls blocks from device callback and executes processing chain.

- `DeviceSettings.h/.cpp`
  - Device configuration handling and persistence-related behavior.
  - Sample-rate/block-size driven reconfiguration coordination.

- `AsioBlacklist.h`
  - Static compatibility guard list for known-problematic ASIO configurations.

## 2.3 DSP Layer

- `EQProcessor.h/.cpp`
  - 20-band parametric EQ core processing.
  - Handles per-band parameter application and per-block filtering.

- `ConvolverProcessor.h/.cpp`
  - Convolution runtime processor.
  - Manages IR-related state transitions and async load/apply flow.
  - Contains latency/mix/bypass and transition-safe operational logic.

- `MKLNonUniformConvolver.h/.cpp`
  - oneMKL-backed non-uniform partitioned convolution backend.
  - Performance-critical convolution implementation.

- `CustomInputOversampler.h/.cpp`
  - Input-side oversampling stage for nonlinear-friendly processing quality.

- `OutputFilter.h/.cpp`
  - Output-stage filtering/conditioning.

- `PsychoacousticDither.h`
  - Dither/noise-shaping utilities for final output stage.

- `InputBitDepthTransform.h`
  - Input bit-depth/quantization-related transform utilities.

## 2.4 Utility / Memory

- `AlignedAllocation.h`
  - Alignment-aware memory allocation helpers.
  - Intended for SIMD/MKL-friendly aligned buffers.

---

## 3. Runtime Topology and Data Flow

At a high level, processing is coordinated by `AudioEngine` and executed through `AudioEngineProcessor` in the callback.

Typical logical chain:

`Audio Input -> (pre-conditioning) -> Oversampling (if enabled) -> [EQ <-> Convolver order selectable] -> Output filter / soft limiting-related stages -> Dither / final conditioning -> Audio Output`

Notes:

- Processing order between EQ and Convolver is configurable at runtime.
- UI controls modify engine state; engine applies changes in a callback-safe way.
- Analyzer data path is decoupled from final audio output path.

---

## 4. Thread Model

## 4.1 Message Thread (UI Thread)

Responsibilities:

- Window/control rendering and event handling.
- Device selection/configuration UI operations.
- Requesting parameter/state changes in engine.
- Triggering asynchronous heavy operations (e.g., IR load request).

## 4.2 Audio Thread (Real-Time Callback)

Responsibilities:

- Block-by-block DSP execution only.
- Uses already-prepared state and buffers.
- Must avoid:
  - blocking waits,
  - file I/O,
  - expensive re-initialization,
  - UI thread interaction.

## 4.3 Worker/Background Paths

Responsibilities:

- IR file parsing/loading/resampling/preparation.
- Convolver state build/update staging.
- Handoff into active state with glitch-safe timing strategy.

---

## 5. State Management Strategy

The codebase follows a staged update model:

1. **Request phase (UI/control path)**
   User or settings change requests a parameter/state update.

2. **Prepare phase (non-real-time path)**
   Expensive objects/state are prepared asynchronously when needed.

3. **Apply/swap phase (real-time-safe boundary)**
   Prepared state is made active with minimal callback disruption.

This pattern is especially relevant to convolution IR operations.

---

## 6. Device and Configuration Lifecycle

Device lifecycle is coordinated by `DeviceSettings` + engine:

- Device open/start => engine `prepare` path allocates/initializes processing buffers.
- Runtime operation => per-block process via `AudioEngineProcessor`.
- Device stop/restart/sample-rate change => engine reset/reprepare path.
- ASIO blacklist is used to reduce unstable driver scenarios.

---

## 7. Precision, Performance, and Memory Discipline

- Main DSP processing uses **double precision**.
- Spectrum visualization paths may use reduced precision where acceptable.
- Alignment-conscious allocation is used in performance-critical buffers.
- Real-time code path avoids dynamic growth/reallocation patterns.

---

## 8. Error Handling and Robustness

- User operations that can fail (e.g., IR load) are isolated from audio callback.
- Failures are reported through control/UI path without blocking audio processing.
- Defensive fallback behavior exists for invalid/unavailable resource conditions.

---

## 9. Build and Execution Context

- OS: **Windows 11 x64**
- Framework: **JUCE 8.0.12**
- Compiler: **MSVC (VS2022)**
- Build system: **CMake**
- Recommended generator: **Ninja Multi-Config**
- Math acceleration: **Intel oneMKL**

Artifacts:

- Debug: `build\ConvoPeq_artefacts\Debug\ConvoPeq.exe`
- Release: `build\ConvoPeq_artefacts\Release\ConvoPeq.exe`

---

## 10. Dependency Boundaries

The following directories are external dependencies and should be treated as read-only in normal development flow:

- `JUCE/`
- `r8brain-free-src/`

---

## 11. Current Architectural Summary

ConvoPeq uses a layered architecture:

- **UI Layer** for interaction and visualization,
- **Engine Layer** for lifecycle and orchestration,
- **DSP Layer** for quality/performance-critical audio processing.

The design emphasizes:

- strict real-time safety,
- asynchronous heavy-state preparation,
- modular processing components,
- and a Windows-optimized standalone runtime.
