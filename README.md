# ConvoPeq

High-Fidelity Convolution Reverb & Parametric EQ Application.

## Overview

ConvoPeq is a standalone audio processing application designed for mastering and high-end audio playback. It features a zero-latency convolution engine and a high-precision parametric equalizer.

## Key Technologies

*   **Framework**: [JUCE 8.0.12](https://github.com/juce-framework/JUCE)
*   **Convolution Engine**:
    *   **WDL Convolution Engine** (by Cockos) for efficient non-uniform partitioned convolution.
    *   **Intel oneAPI MKL** (Math Kernel Library) integration for accelerated FFT and vector arithmetic.
*   **Resampling**: [r8brain-free-src](https://github.com/avaneev/r8brain-free-src) for high-quality sample rate conversion.
*   **Equalizer**: 20-band Parametric EQ using TPT (Topology-Preserving Transform) State Variable Filters (SVF).
*   **Dithering**: Custom Psychoacoustic Dither with 5th-order Noise Shaping.
*   **Oversampling**: Linear Phase / IIR polyphase oversampling support.

## Requirements

*   **OS**: Windows 10 / 11 (64-bit)
*   **CPU**: Intel/AMD processor with AVX2 support.
*   **Audio Interface**: ASIO compatible device recommended.

## Build Instructions

### Prerequisites

*   Visual Studio 2022 (17.11 or later)
*   CMake 3.22 or later
*   Intel oneAPI MKL (Optional, but recommended for performance)

### Building

Use the provided `build.bat` script:

```cmd
build.bat Release
```

Or manually via CMake:

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## License

*   **ConvoPeq**: Copyright (c) 2024 lonewolf-jp.