# ConvoPeq Architecture

This document describes the current software architecture of **ConvoPeq v0.4.4**, a Windows-only standalone audio application built with JUCE 8.0.12.

## 1. Design Goals

ConvoPeq is designed around the following goals:

1. **Real-time safety**
   The audio callback path avoids blocking operations (no file I/O, no locks, no dynamic allocation on the audio thread).

2. **High performance**
   The DSP path is optimized for AVX2-capable CPUs and uses Intel oneMKL where appropriate.
   Performance-critical buffers are 64-byte aligned.

3. **Modular asynchronous updates**
   Expensive operations (for example IR load/prepare/rebuild) are handled off the audio thread and swapped into active processing state safely.

---

## 2. High-Level Module Map (`src/`)

- **Application/UI**
  - `MainApplication.*`
  - `MainWindow.*`
  - `EQControlPanel.*`
  - `ConvolverControlPanel.*`
  - `SpectrumAnalyzerComponent.*`

- **Audio engine / runtime orchestration**
  - `AudioEngine.*`
  - `AudioEngineProcessor.*`
  - `DeviceSettings.*`
  - `AsioBlacklist.h`

- **DSP core**
  - `ConvolverProcessor.*`
  - `MKLNonUniformConvolver.*`
  - `EQProcessor.*`
  - `CustomInputOversampler.*`
  - `OutputFilter.*`
  - `PsychoacousticDither.h`
  - `InputBitDepthTransform.h`

- **Memory/alignment utilities**
  - `AlignedAllocation.h`

---

## 3. Threading and Responsibilities

## 3.1 Message Thread (UI thread)

- Owns and updates GUI components.
- Handles user interactions and device changes.
- Initiates heavy DSP state changes asynchronously (for example IR load requests).

## 3.2 Audio Thread (real-time callback)

- Runs the per-block processing path only.
- Uses prepared DSP state and preallocated buffers.
- Must remain lock-free/wait-free from an application perspective.

## 3.3 Background worker paths

- Convolution/IR-related preparation is executed outside the audio callback.
- Prepared state is published back to active processing with safe handoff patterns to avoid glitches.

---

## 4. Audio Pipeline

The effective processing chain is organized around `AudioEngine` + DSP components:

`Input -> (optional conditioning / DC handling) -> Oversampling (up) -> Convolver/EQ chain -> Soft clip / output filter -> Oversampling (down) -> Dither / output conditioning -> Output`

Notes:

- Internal processing is centered on **double precision** for DSP quality.
- Convolver/EQ ordering is configurable in the UI.
- Nonlinear stages are protected with oversampling and output filtering to reduce aliasing.

---

## 5. DSP Components

## 5.1 `ConvolverProcessor` and `MKLNonUniformConvolver`

- Main IR convolution path.
- Uses partitioned convolution suitable for long IR workloads.
- Integrates IR loading/preparation logic and runtime-safe state application.

## 5.2 `EQProcessor`

- Multi-band parametric EQ processing.
- Designed for real-time parameter update behavior compatible with live audio use.

## 5.3 `CustomInputOversampler`

- Provides configurable oversampling around nonlinear stages.
- Used to improve spectral behavior during soft clipping / saturation-related processing.

## 5.4 `OutputFilter` and `PsychoacousticDither`

- Output-side cleanup and finalization.
- Dither/noise-shaping stage is included in final output conditioning.

---

## 6. UI and Data Flow

## 6.1 Window and control panels

- `MainWindow` hosts primary controls and status elements.
- `EQControlPanel` and `ConvolverControlPanel` map UI actions to `AudioEngine` APIs.
- `DeviceSettings` manages audio-device related configuration flow.

## 6.2 Spectrum analyzer path

- Audio analysis data is passed to `SpectrumAnalyzerComponent` through a decoupled path suitable for UI rendering cadence.
- UI visualization is isolated from real-time callback responsibilities.

---

## 7. Memory and Real-Time Constraints

- Audio-thread allocations are avoided.
- Alignment-sensitive buffers use dedicated allocation helpers (`AlignedAllocation.h`).
- Heavy initialization/reconfiguration occurs before activation or on worker paths, not inside the callback.

---

## 8. Build and Runtime Environment

- **OS**: Windows 11 x64
- **Framework**: JUCE 8.0.12
- **Language**: C++20
- **Build system**: CMake
- **Generator (recommended)**: Ninja Multi-Config
- **Math library**: Intel oneMKL
- **Resampling dependency**: r8brain-free-src

Typical output layout:

- `build\ConvoPeq_artefacts\Debug\ConvoPeq.exe`
- `build\ConvoPeq_artefacts\Release\ConvoPeq.exe`

---

## 9. Dependency and Source Boundaries

The following third-party source trees are external dependencies and should be treated as read-only project dependencies:

- `JUCE/`
- `r8brain-free-src/`

---

## 10. License

ConvoPeq project licensing is defined in repository license files.
Third-party dependencies (JUCE, oneMKL, r8brain-free-src) follow their own licenses.
