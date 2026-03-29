
# ConvoPeq Architecture (v0.5.8+)

This document describes the internal architecture of **ConvoPeq**, a Windows-only standalone audio application built with **JUCE 8.0.12** and Intel oneMKL. It is intended for developers and contributors working on DSP, threading, state transitions, and runtime behavior.

For user-facing features and usage, see `README.md`.

---

## 1. System Goals and Non-Functional Priorities

ConvoPeq is organized around four priorities:

- Runtime-critical memory is pre-allocated and reused.

1. **Audio quality**
   - Core processors are high-quality convolution and 20-band parametric EQ.
   - Output conditioning includes output filtering, soft clipping (optional), and dither.

2. **Performance**
   - Optimized for Windows x64 + AVX2-class CPUs.
   - Uses Intel oneMKL where beneficial (FFT/BLAS/VML paths).
   - Uses alignment-aware memory allocation for SIMD/MKL efficiency.

3. **Operational robustness**
   - UI/control logic is decoupled from DSP execution.
   - Heavy work (IR load/rebuild) is asynchronous.
   - State transitions are staged to avoid audible artifacts.

## Adaptive Noise Shaper Learning (v0.5.8+)

- `NoiseShaperLearner` receives 256-sample AudioBlocks from the Audio Thread via a LockFreeRingBuffer and performs CMA-ES optimization (9th-order IIR noise shaper coefficients) on a dedicated worker thread.
- Coefficient banks are managed per sample rate and bit depth; progress, error, and coefficients are reported to the UI/Engine via atomic variables.
- All inter-thread data transfer is designed for real-time safety using RCU/atomic/lock-free patterns.
- Three learning modes (Short/Middle/Long) control convergence speed and stability.
- Typical convergence times: Short 10–20 min, Middle 20–40 min, Long 40–80 min.
- See README.md and NoiseShaperLearner.h/.cpp for details.

---

## 2. Design View of `src/`

### 2.1 Application / UI Layer

- `MainApplication.h/.cpp`
  - JUCE app entry and lifecycle management.

- `MainWindow.h/.cpp`
  - Main desktop window composition.
  - Wires UI controls to `AudioEngine` operations.

- `EQControlPanel.h/.cpp`
  - EQ parameter UI and user interaction.

- `ConvolverControlPanel.h/.cpp`
  - Convolver UI (IR load, phase mode, mix, smoothing, IR length, output filter modes).
  - Includes UI-side debounce for expensive convolver parameter changes.

- `SpectrumAnalyzerComponent.h/.cpp`
  - Real-time analyzer and EQ overlay rendering.
  - Uses lock-free FIFO input, FFT visualization pipeline, smoothing, peak hold, and adaptive update throttling.

### 2.2 Engine / Runtime Coordination Layer

- `AudioEngine.h/.cpp`
  - Main orchestration and runtime control layer.
  - Owns and coordinates EQ, convolver, utility DSP stages, and analyzer path.
  - Exposes high-level control API to UI.

- `AudioEngineProcessor.h/.cpp`
  - Adapter from device callback into `AudioEngine` processing.

  - Compatibility guard for known problematic ASIO scenarios.

### 2.3 DSP Layer

- `EQProcessor.h/.cpp`
  - 20-band parametric EQ core processing. Manages band/channel modes and real-time parameter application.

- `ConvolverProcessor.h/.cpp`
  - IR-based convolution runtime. Handles asynchronous IR loading/rebuild, crossfade-safe transitions, and latency/delay management.
  - Implements rebuild debounce, coalesced notifications, crossfade, and hysteresis to suppress clicks and bursty updates.
  - IR loading/rebuild is performed on the Message/Worker Thread; the Audio Thread only performs atomic reference switching.

- `MKLNonUniformConvolver.h/.cpp`
  - Intel oneMKL-backed non-uniform partitioned convolution backend.
  - All large buffers (IR, FFT, workspaces, etc.) are allocated/freed using convo::aligned_malloc (64-byte alignment) and ScopedAlignedPtr (RAII).
  - Memory allocation/deallocation and libm calls are strictly prohibited on the Audio Thread.

- `CustomInputOversampler.h/.cpp`
  - Input oversampling for quality/performance modes.

- `OutputFilter.h/.cpp`
  - Output filtering/conditioning stage.

- `PsychoacousticDither.h`
  - Final-stage dither/noise shaping.

- `NoiseShaperLearner.h/.cpp`
  - Adaptive Noise Shaper Learning (see above for details).

- `InputBitDepthTransform.h`
  - Input bit-depth/quantization utilities.

### 2.4 Utility / Memory

- `AlignedAllocation.h`
  - 64-byte alignment helpers for SIMD/MKL-friendly buffer allocation.
  - All buffers are managed with convo::aligned_malloc and ScopedAlignedPtr (RAII).

---

## 3. Runtime Topology and Data Flow

At runtime, `AudioEngine` coordinates all processing, including adaptive noise shaper learning, and `AudioEngineProcessor` invokes the main DSP chain from the device callback.

Typical logical chain:

`Audio Input -> Input conditioning -> Oversampling (optional) -> [EQ <-> Convolver (order selectable)] -> Output filter / soft clipping -> Dither / final conditioning (with optional Adaptive Noise Shaper) -> Audio Output`

## Detailed Data Processing Flow

The following describes the precise, step-by-step flow of audio data through ConvoPeq’s processing pipeline, referencing key classes and methods:

1. **Audio Input Reception**

- The audio device callback invokes `AudioEngineProcessor::getNextAudioBlock()`, which delegates to `AudioEngine::getNextAudioBlock()`.
- Input buffer is received as a `juce::AudioSourceChannelInfo` object.

1. **Oversampling (Optional)**

- If enabled, `CustomInputOversampler::processUp()` is called.
- Multi-stage FIR/IIR upsampling is performed (factor 2x/4x/8x), using AVX2-optimized routines.
- Input DC offset is removed by `UltraHighRateDCBlocker`.

1. **Processing Order Selection**

- The processing order is determined by `AudioEngine::ProcessingOrder` (Convolver→EQ or EQ→Convolver).
- This is set via the UI and stored atomically.

1. **EQ Processing**

- `EQProcessor::process()` is called.
- 20-band parametric EQ is applied using TPT SVF filters.
- All parameter/state updates are lock-free (RCU pattern).
- AGC (automatic gain control) is applied if enabled.

1. **Convolution Processing**

- `ConvolverProcessor::process()` is called.
- Performs high-performance convolution (linear/minimum phase, IR smoothing, output filter).
- IR and parameters are updated asynchronously and atomically.

1. **Output Conditioning**

- `OutputFilter::process()` applies high/low cut filtering.
- `UltraHighRateDCBlocker` removes output DC offset.
- If enabled, `AudioEngine::DSPCore::softClipBlockAVX2()` applies musical soft clipping (AVX2-optimized).

1. **Dither & Noise Shaping**

- Dither and noise shaping are applied according to user selection:
  - `PsychoacousticDither`, `FixedNoiseShaper`, `Fixed15TapNoiseShaper`, or `LatticeNoiseShaper` (adaptive, CMA-ES optimized).
- Adaptive coefficients are managed per sample rate/bit depth/mode and updated atomically.

1. **Downsampling (If Oversampled)**

- `CustomInputOversampler::processDown()` is called.
- Multi-stage FIR/IIR downsampling is performed, AVX2-optimized, with real-time safety.

1. **Output Buffer Delivery**

- The processed buffer is written back to the output channels in `juce::AudioSourceChannelInfo`.
- Level meters and spectrum analyzer taps are updated via lock-free FIFO.

1. **UI/Analyzer/Worker Thread Data Transfer**

- Analyzer and learning data are pushed to `LockFreeRingBuffer` for consumption by the UI and `NoiseShaperLearner` worker thread.
- All inter-thread communication uses atomic/lock-free/RCU patterns for real-time safety.

**See also:**

- `src/AudioEngine.cpp` (`getNextAudioBlock`, `DSPCore::process`)
- `src/EQProcessor.cpp`, `src/ConvolverProcessor.cpp`, `src/CustomInputOversampler.cpp`, `src/OutputFilter.cpp`
- `SOUND_PROCESSING.md` for mathematical and code-level details.

---

Key points:

- **EQ/Convolver order is runtime-selectable**.
- Analyzer source can be input or output path.
- Analyzer data flow is decoupled from final output and read through FIFO on the UI side.
- Gain staging is mode-aware:
  - input headroom is applied before core DSP,
  - convolver input trim is applied only in **EQ -> Convolver** when both stages are active,
  - output makeup is applied before optional soft clipping.
- **Adaptive Noise Shaper Learning**:
  - When enabled, the audio thread pushes audio blocks to a lock-free ring buffer for the learner.
  - The learner runs on a dedicated worker thread, asynchronously optimizing noise shaper coefficients using recent audio.
  - State handoff between audio and learner threads is RCU-style and lock-free for real-time safety.
  - UI progress and error state are exposed via atomic variables and polled by the engine/UI.

---

## 4. Subsystem Responsibilities

### 4.1 AudioEngine

- Owns the high-level runtime state exposed to UI.
- Bridges UI requests to DSP-safe update paths.
- Coordinates processing order, bypass states, analyzer routing, device-driven prepare/reset, and rebuild staging.

- Manages the lifecycle and control of the Adaptive Noise Shaper Learner, including starting/stopping learning, progress polling, and error reporting.

### 4.2 EQProcessor

- Applies the 20-band parametric EQ in the real-time path.
- Exposes parameter/state interfaces used by UI and engine.
- Keeps display-oriented EQ response work outside the callback path.

### 4.3 ConvolverProcessor

- Owns IR runtime state and rebuild lifecycle.
- Handles asynchronous IR preparation, debounce, crossfade-safe transitions, and notification coalescing.
- Separates heavy rebuild work from callback-time use.

### 4.4 NoiseShaperLearner

- Dedicated background worker thread for adaptive noise shaper learning.
- Audio thread pushes audio blocks to a lock-free ring buffer (256-sample blocks, real-time safe).
- Worker thread runs CMA-ES optimization using the most recent audio (up to 8 segments per generation).
- Three learning modes (Short/Middle/Long) select convergence speed and stability.
- Coefficient banks are saved/loaded per sample rate and bit depth.
- Progress, error, and best coefficients are reported via atomic variables and polled by engine/UI.
- All memory handoff and state transitions are real-time safe (no locks or blocking in the audio thread).
- Typical convergence time: Short 10–20min, Middle 20–40min, Long 40–80min.

### 4.5 SpectrumAnalyzerComponent

- Consumes analyzer FIFO data on the UI side.
- Runs FFT visualization, smoothing, peak hold, and EQ overlay drawing.
- Uses adaptive update rates to reduce UI/message-thread burst load.

---

## 5. Threading Model / Thread Classification and Responsibilities

### 5.1 Message Thread (UI Thread)

- Handles UI rendering, event processing, user actions, device settings, and dispatches asynchronous requests.
- Heavy operations such as IR loading/rebuild and NoiseShaper learning are delegated to the Worker Thread from here.

### 5.2 Audio Thread (Real-Time Callback)

- Performs only block-based DSP processing; always references pre-constructed state.
- Memory allocation/deallocation, libm calls, synchronization/communication, I/O, and UI access are strictly prohibited.
- For Adaptive Noise Shaper Learning, pushes AudioBlocks to the LockFreeRingBuffer (non-blocking, lock-free).

### 5.3 Worker / Background Thread

- Handles heavy tasks: IR parsing/loading/resampling/phase conversion, convolution state construction, engine rebuild, etc.
- Adaptive Noise Shaper Learning: CMA-ES optimization using recent AudioBlocks.
- All handoff to/from the Audio Thread is done with RCU/atomic/lock-free patterns for real-time safety.

---

## 6. State and Transition Strategy

ConvoPeq follows a staged update model:

1. **Request phase (UI/control path)**
   - User or settings request a change.

2. **Prepare phase (non-real-time path)**
   - Expensive structures are built asynchronously.

3. **Apply/swap phase (real-time-safe boundary)**
   - Prepared state is activated with minimal callback disruption.

This is especially important for convolver IR operations and engine rebuilds.

Persistence is split into two paths:

- `DeviceSettings::saveSettings/loadSettings`
  - Restores device state and a small set of session-like engine settings (`ditherBitDepth`, `oversamplingFactor`, `oversamplingType`, `inputHeadroomDb`, `outputMakeupDb`).
- `AudioEngine::getCurrentState()/requestLoadState()`
  - Manual preset XML path used by the main window save/load actions.
  - Restores processing order, bypass state, gain staging, analyzer routing, filter modes, dither, oversampling, EQ state, and Convolver state.
  - Load order is staged to prevent mode-dependent defaults from overwriting restored gain settings.

---

## 7. Convolver Runtime Design (Current)

`ConvolverProcessor` combines asynchronous rebuild with click-safe transitions:

- **Async IR loading/rebuilding** through worker paths.
- **Debounced rebuild scheduling** for bursty UI updates.
- **Configurable rebuild debounce** (`50..3000 ms`, default `400 ms`), persisted in state.
- **Coalesced change notifications** to reduce message-thread bursts.
- **Crossfade-aware processing** so old/new wet/delay paths overlap only where needed.
- **Latency retarget hysteresis** to reduce frequent retriggering.
- **Phase state persistence** for `AsIs / Mixed / Minimum` plus `mixedF1Hz / mixedF2Hz / mixedTau`.
- **IR length persistence** for both target length and auto/manual selection state.

UI integration details:

- `ConvolverControlPanel` applies a **3-second idle debounce** for selected expensive controls (mix/smoothing/IR length) to avoid repeated rebuild pressure during active dragging.
- Processor-side rebuild debounce (`50..3000 ms`) remains active as a second guard against bursty rebuild requests from any caller.
- Advanced IR settings are hosted in a separate dialog and include IR length, rebuild debounce, and Mixed-phase tuning controls.

Latency reporting details:

- `AudioEngine::getCurrentLatencyBreakdown()` publishes oversampling latency, convolver algorithm latency, IR peak latency, and total base-rate latency.
- `MainWindow::timerCallback()` renders latency from the single source `totalLatencyBaseRateSamples` to keep `ms` and `samples` numerically consistent.
- `AudioEngine::getCurrentLatencyMs()` also derives from total samples (single-source conversion), avoiding per-component rounding drift.

---

## 8. Analyzer and Visualization Path (Current)

`SpectrumAnalyzerComponent` uses a throttled UI pipeline:

- Input source is read from engine FIFO on timer callbacks.
- FFT path includes windowing, magnitude conversion, smoothing, and peak hold.
- EQ response overlays are computed and rendered separately from audio processing.

Performance controls:

- **Adaptive timer rates by state**:
  - active analyzer: `60 Hz`,
  - analyzer disabled but visible: `15 Hz`,
  - hidden component: `5 Hz`.
- **Coalesced EQ update requests** using dirty-flag + interval gating.
- Analyzer-off path avoids unnecessary repeated full visual updates.
- Input/output analyzer routing is selected in `AudioEngine`, while the UI consumes data through the engine FIFO without touching callback-time state.

---

## 9. Real-Time Safety and Memory Discipline

- The main DSP path uses 64-bit double precision.
- All large buffers (IR, FFT, workspaces, etc.) are allocated/freed using convo::aligned_malloc (64-byte alignment) and ScopedAlignedPtr (RAII).
- Memory allocation/deallocation, libm calls, synchronization/communication, exceptions, etc. are strictly prohibited on the Audio Thread.
- All temporary buffers are managed with RAII; no leaks even on exceptions or early returns.
- Dynamic growth patterns (e.g., std::vector) are allowed only outside the Audio Thread.
- Denormal handling and stable smoothing/crossfade transitions are enforced.
- Object retirement uses reference counting and deferred retirement for safety.

---

## 10. Device and Configuration Lifecycle

Device lifecycle is coordinated by `DeviceSettings` + `AudioEngine`:

- Device start/open -> prepare/init processing resources.
- Runtime -> per-block execution through `AudioEngineProcessor`.
- Device stop/restart/sample-rate change -> reset/reprepare flow.
- ASIO blacklist support reduces unstable driver scenarios.

Configuration lifecycle uses both persistence paths:

- `device_settings.xml` for device-centric auto-restore and a compact runtime subset.
- Manual preset XML for full processing-state portability (including EQ/Convolver internals).

---

## 11. Build and Runtime Context

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

## 12. Dependency Boundaries

The following directories are external dependencies and must be treated as strictly read-only during normal development:

- `JUCE/`
- `r8brain-free-src/`

---

## 13. Development Notes

- Keep callback-time work deterministic and allocation-free.
- Treat convolver rebuilds and analyzer refresh as separate burst-control problems.
- Prefer staging, debounce, and handoff over immediate heavy reconfiguration.
- Preserve the current read-only boundary for external dependencies.

---

## 14. Current Architectural Summary

ConvoPeq uses the following layered architecture:

- **UI Layer**: User interaction, visualization, and state display
- **Engine Layer**: Orchestration and lifecycle management
- **DSP Layer**: Quality/performance-focused signal processing

Design focus:

- Strict real-time safety (all Audio Thread prohibitions strictly enforced)
- Asynchronous and heavy state construction is always performed on Worker/Message Threads in advance
- All inter-thread data transfer uses lock-free/RCU/atomic patterns
- All buffers in the MKL path are strictly 64-byte aligned
- Dependency directories (JUCE/ and r8brain-free-src/) are strictly read-only

For details on the learning algorithm, convergence, and bit depth management, see `README.md` and `NoiseShaperLearner.h/.cpp`.
