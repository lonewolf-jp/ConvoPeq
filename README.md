# ConvoPeq v0.4.4

High-fidelity standalone audio processor for Windows 11 x64, combining convolution processing and a 20-band parametric EQ.

## Overview

ConvoPeq is a JUCE 8.0.12 desktop application focused on real-time audio processing quality and stability on Windows.
The codebase is organized around a modular audio engine, dedicated DSP processors, and separate UI control panels.

- Platform: **Windows-only**
- Framework: **JUCE 8.0.12**
- DSP precision: **64-bit double** (except analyzer-oriented paths where appropriate)
- Performance targets: **AVX2 + Intel oneMKL**

---

## Source Layout (`src/`)

### Application / UI Layer

- `MainApplication.*`
  Application bootstrap and JUCE app lifecycle.
- `MainWindow.*`
  Main window composition, top-level UI wiring.
- `EQControlPanel.*`
  EQ-side parameter controls.
- `ConvolverControlPanel.*`
  Convolver-side parameter controls.
- `SpectrumAnalyzerComponent.*`
  Real-time spectrum visualization.

### Engine / Runtime Orchestration

- `AudioEngine.*`
  Central coordinator for DSP modules and runtime state.
- `AudioEngineProcessor.*`
  Audio callback-facing processing integration.
- `DeviceSettings.*`
  Device configuration and persistence-related handling.
- `AsioBlacklist.h`
  ASIO compatibility guard support.

### DSP Layer

- `ConvolverProcessor.*`
  Convolution processor state, IR loading/transition flow, runtime processing.
- `MKLNonUniformConvolver.*`
  oneMKL-accelerated non-uniform partitioned convolution core.
- `EQProcessor.*`
  20-band parametric EQ processing core.
- `CustomInputOversampler.*`
  Input-side oversampling stage.
- `OutputFilter.*`
  Output conditioning/filter stage.
- `PsychoacousticDither.h`
  Dither/noise-shaping utilities.
- `InputBitDepthTransform.h`
  Input bit-depth transform helpers.

### Memory / Utility

- `AlignedAllocation.h`
  Alignment-aware allocation helpers for SIMD/MKL-friendly memory usage.

---

## Functional Highlights (Current Codebase)

- 20-band parametric EQ processing (`EQProcessor`)
- Convolution processing with MKL-based non-uniform partitioning (`ConvolverProcessor`, `MKLNonUniformConvolver`)
- Configurable processing order support in UI/engine integration
- Oversampling and output conditioning stages (`CustomInputOversampler`, `OutputFilter`)
- Psychoacoustic dither support (`PsychoacousticDither`)
- Real-time analyzer component (`SpectrumAnalyzerComponent`)
- ASIO/WASAPI-oriented runtime operation and device management

---

## Build Requirements

1. **Visual Studio 2022** (17.11+) with Desktop C++ workload
2. **CMake 3.22+**
3. **Ninja** (recommended)
4. **Intel oneAPI Base Toolkit** (oneMKL required)
5. **JUCE 8.0.12** at:
   - `ConvoPeq/JUCE/...`

---

## Quick Build (Recommended)

Use the batch script from project root:

```cmd
build.bat Release
build.bat Debug
build.bat Release clean
```

### Output

- Debug:
  - `build\ConvoPeq_artefacts\Debug\ConvoPeq.exe`
- Release:
  - `build\ConvoPeq_artefacts\Release\ConvoPeq.exe`

---

## Manual Build (Equivalent)

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64

cmake -S . -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cmake --build build --config Debug
cmake --build build --config Release
```

---

## Notes

- This repository targets a **standalone app**, not a plugin target.
- Do not modify third-party dependency trees directly:
  - `JUCE/`
  - `r8brain-free-src/`

---

## License

- **ConvoPeq**: Copyright (c) lonewolf-jp
- **JUCE**: GPLv3 / Commercial
- **r8brain-free-src**: MIT
- **Intel oneMKL**: Intel Simplified Software License
