# ConvoPeq v0.3.5

High-Fidelity Convolution Reverb & Parametric EQ Application for Windows 11

## Overview

ConvoPeq is a standalone audio processing application designed for mastering and high-end audio playback. It integrates a zero-latency convolution engine and a high-precision 20-band parametric equalizer into a 64-bit double-precision audio pipeline.

Designed specifically for **Windows 11 x64**, it leverages **AVX2** instructions and **Intel oneAPI MKL** to deliver maximum performance and audio quality.

## Key Features

### 🎛️ 20-Band Parametric Equalizer

* **Algorithms**: TPT (Topology-Preserving Transform) State Variable Filters (SVF) for analog-matched curves without cramping at Nyquist.
* **Filter Types**: Low Shelf, Peaking, High Shelf, Low Pass, High Pass.
* **Channel Modes**: Stereo, Left-only, Right-only processing per band.
* **Auto Gain Control (AGC)**: Automatically matches output loudness to input.
* **Visualization**: Real-time frequency response curve overlay.
* **Preset Loading**: Supports standard XML presets and **Equalizer APO** format text files (`.txt`).

### 🔊 Convolution Engine

* **Engine**: Custom **Intel MKL Non-Uniform Partitioned Convolution (NUC)** engine.
* **Latency**: Zero-latency processing (suitable for real-time monitoring).
* **IR Support**: WAV, AIFF, FLAC formats.
* **Processing**:
  * **Phase Modes**: Switchable between Linear Phase and Minimum Phase.
  * **Auto-makeup gain**: Automatic energy normalization.
  * **Resampling**: High-quality resampling using **r8brain-free-src**.
  * **Adjustable**: Mix, Smoothing Time, and Target IR Length.

### 🚀 High-Fidelity Audio Pipeline

* **Precision**: Full **64-bit double-precision** signal path.
* **Oversampling**: Up to **8x** oversampling with selectable filters:
  * **Linear Phase**: Perfect phase response.
  * **IIR-Like**: Low latency "Intermediate" phase.
* **Dithering**: **Psychoacoustic Dither** with 5th-order Noise Shaping (MKL VSL based).
* **DC Blocking**: High-precision DC removal at input and output stages (UltraHighRateDCBlocker).
* **Soft Clipper**: "Musical" soft clipper with adjustable saturation to prevent digital overs.
* **Headroom**: Automatic -0.1dB input headroom to prevent inter-sample peaks.
* **Mono-to-Stereo**: Automatic expansion of mono inputs to stereo processing.
* **Seamless Switching**: Cross-fading on parameter changes to prevent clicks.
* **Asynchronous Architecture**: Heavy tasks (IR loading, resampling) are offloaded to background threads to ensure glitch-free audio.
* **Windows Optimization**: Automatic process priority elevation, Efficiency Mode (EcoQoS) disablement, and timer resolution adjustment.
* **Processing Order**: Switchable signal chain order (EQ → Convolver or Convolver → EQ).

### 📊 Visualization & Tools

* **Spectrum Analyzer**: Real-time FFT analyzer (60fps) with peak hold.
  * Switchable Input / Output source monitoring.
  * Toggleable to save CPU resources.
* **Device Management**: ASIO/WASAPI support with an **ASIO Blacklist** feature to exclude unstable drivers (e.g., single-client drivers).
* **CPU Monitor**: Real-time CPU usage display.

## System Requirements

* **OS**: Windows 11 x64 (Strictly required).
* **CPU**: Intel/AMD processor with **AVX2** instruction set support (Haswell/Zen or later).
* **Audio Interface**: ASIO compatible device recommended for low latency.

## Build Instructions

### Prerequisites

1. **Visual Studio 2022** (v17.11 or later) with C++ Desktop Development workload.
2. **CMake** 3.22 or later.
3. **Intel oneAPI Base Toolkit** (specifically Intel MKL).
4. **JUCE Framework** v8.0.12.

### Directory Structure

Ensure the `JUCE` folder is placed in the project root:

```text
ConvoPeq/
├── JUCE/              # JUCE 8.0.12 source code
├── src/               # Application source
├── ...
```

### Building

Use the provided helper script `build.bat` which automatically sets up the environment (including Intel MKL variables).

```cmd
:: Build Release version (Recommended)
build.bat Release

:: Build Debug version
build.bat Debug

:: Clean build
build.bat Release clean
```

Or manually via CMake:

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## License

* **ConvoPeq**: Copyright (c) 2024-2025 lonewolf-jp. CC BY-NC 4.0 (Attribution-NonCommercial).
* **JUCE**: GPLv3 / Commercial.
* **r8brain-free-src**: MIT License.
* **Intel oneMKL**: Intel Simplified Software License.
