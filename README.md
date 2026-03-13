# ConvoPeq v0.4.4

High-Fidelity Convolution Reverb & Parametric EQ Application for Windows 11 x64

## Overview

ConvoPeq is a standalone desktop audio processor for Windows 11, designed for high-fidelity playback and mastering workflows.
It combines a low-latency convolution engine and a high-precision 20-band parametric EQ in a 64-bit double-precision signal path.

This project is **Windows-only** and optimized for **AVX2** CPUs with **Intel oneAPI MKL** acceleration.

---

## Core Features

### 20-Band Parametric EQ

- Topology-Preserving Transform (TPT) SVF design
- Filter types: Low Shelf, Peaking, High Shelf, Low Pass, High Pass
- Per-band channel mode control (Stereo / Left / Right)
- Auto Gain Control (AGC)
- Real-time response visualization
- Preset loading (XML and Equalizer APO `.txt`)

### Convolution Engine

- Custom low-latency non-uniform partitioned convolution (NUC)
- IR format support: WAV / AIFF / FLAC
- Linear phase / minimum phase switching
- Auto makeup gain normalization
- High-quality resampling via r8brain-free-src
- Adjustable mix, smoothing, and target IR length

### Audio Pipeline

- Full internal **64-bit double-precision** processing
- Oversampling up to 8x
- Psychoacoustic dither with noise shaping
- Input/output DC blocking
- Soft clipper with adjustable saturation
- Headroom control and anti-clipping safeguards
- Mono-to-stereo expansion
- Click-safe crossfades for parameter transitions
- Heavy operations offloaded asynchronously (e.g., IR loading)

### UI / Runtime Tools

- Real-time spectrum analyzer (with CPU-saving toggle)
- ASIO / WASAPI device support
- ASIO blacklist handling for unstable drivers
- Real-time CPU usage display
- Processing order switch (EQ -> Convolver / Convolver -> EQ)

---

## System Requirements

- **OS**: Windows 11 x64 (required)
- **CPU**: AVX2-capable Intel/AMD CPU
- **Audio**: ASIO device recommended for low-latency use

---

## Build Requirements

1. **Visual Studio 2022** (17.11+) with C++ desktop workload
2. **CMake** 3.22+
3. **Intel oneAPI Base Toolkit** (MKL required)
4. **JUCE 8.0.12** placed at project root:
   - `ConvoPeq/JUCE/...`

---

## Quick Build (Recommended)

Use `build.bat` from the project root:

```cmd
build.bat Release
build.bat Debug
build.bat Release clean
```

### Build Output

- Debug executable:
  - `build\ConvoPeq_artefacts\Debug\ConvoPeq.exe`
- Release executable:
  - `build\ConvoPeq_artefacts\Release\ConvoPeq.exe`

---

## Manual CMake Build (Equivalent)

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64

cmake -S . -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cmake --build build --config Release
cmake --build build --config Debug
```

---

## Notes

- JUCE and r8brain sources are external dependencies and should not be modified in-place.
- This application is a standalone app target (not a plugin target).
- For development in VS Code, task-based build flow is supported.

---

## License

- **ConvoPeq**: Copyright (c) lonewolf-jp
- **JUCE**: GPLv3 / Commercial
- **r8brain-free-src**: MIT
- **Intel oneMKL**: Intel Simplified Software License
