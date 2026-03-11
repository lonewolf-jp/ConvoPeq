# ConvoPeq Architecture

This document outlines the software architecture of ConvoPeq, a high-performance audio convolution and equalization application.

## 1. Core Principles

The architecture is built upon three core principles:

1. **Real-time Safety**: The audio processing thread is designed to be lock-free and wait-free, avoiding operations that could lead to audio dropouts, such as memory allocation, locking, or file I/O.
2. **High Performance**: The signal processing pipeline is heavily optimized using Intel® oneAPI Math Kernel Library (oneMKL) and AVX2 intrinsics. Memory is managed with 64-byte alignment to maximize SIMD efficiency.
3. **Modularity and Asynchronicity**: Heavy tasks like loading impulse responses or reconfiguring the DSP chain are offloaded to background threads. A Read-Copy-Update (RCU) pattern is used extensively to update the state of the audio engine without blocking the real-time audio thread.

## 2. Threading Model

ConvoPeq employs a multi-threaded model to separate concerns and ensure a responsive UI and glitch-free audio playback.

* **Message Thread (UI Thread)**: The main application thread, managed by JUCE. It handles all user interface interactions, manages windowing, and dispatches events. When a user changes a parameter that requires a significant DSP reconfiguration (e.g., loading a new IR, changing the oversampling factor), this thread initiates a rebuild request.

* **Audio Thread**: A high-priority, real-time thread managed by the audio device driver (e.g., ASIO, WASAPI).
  * Its sole responsibility is to execute the `getNextAudioBlock()` callback.
  * It operates on a "snapshot" of the current DSP configuration by loading an atomic pointer to the active `DSPCore`. This is a wait-free operation.
  * It is strictly forbidden from performing any blocking operations.

* **Rebuild Thread (`AudioEngine::rebuildThreadLoop`)**: A dedicated background worker thread for preparing new DSP configurations.
  * When a rebuild is requested by the Message Thread, this thread creates and prepares a new `DSPCore` instance.
  * This is where all "heavy" non-real-time operations occur:
    * Memory allocation for audio buffers (`mkl_malloc`).
    * Initialization of DSP components (Convolver, EQ, Oversampler).
    * Synchronous rebuilding of the convolution engine if needed.
  * Once the new `DSPCore` is ready, it is handed back to the Message Thread to be atomically swapped into the `currentDSP` pointer.

* **Loader Thread (`ConvolverProcessor::LoaderThread`)**: A background thread specifically for loading and processing impulse response (IR) files.
  * Handles file I/O, audio format decoding, resampling, and minimum-phase transformation.
  * This ensures that loading large IR files does not block the UI or audio threads.

## 3. DSP Architecture

### 3.1. `AudioEngine` and the RCU Pattern

The `AudioEngine` is the central hub for all audio processing. It implements the `juce::AudioSource` interface and manages the lifecycle of the DSP chain.

The core of its design is a **Read-Copy-Update (RCU)** pattern for managing DSP state changes:

1. **Read**: The Audio Thread reads an atomic pointer (`currentDSP`) to get the currently active `DSPCore` instance for processing. This is a fast, lock-free operation.
2. **Copy**: When a setting is changed, the Message Thread requests a rebuild. The Rebuild Thread creates a *new* `DSPCore` instance and copies the state from the UI-facing processors (`uiEqProcessor`, `uiConvolverProcessor`).
3. **Update**: Once the new `DSPCore` is fully prepared, the Message Thread atomically swaps the `currentDSP` pointer to point to the new instance.

Old `DSPCore` instances are moved to a "trash bin" and are garbage-collected by a `juce::Timer` on the Message Thread after a safe delay. This ensures that the Audio Thread, which might still be using an old instance, can finish its processing cycle without encountering a dangling pointer.

### 3.2. `DSPCore`: The Processing Graph

The `DSPCore` struct encapsulates a complete, self-contained audio processing graph. This includes:

* Aligned memory buffers for audio data (`ScopedAlignedPtr<double>`).
* An instance of `ConvolverProcessor`.
* An instance of `EQProcessor`.
* A `CustomInputOversampler`.
* A `PsychoacousticDither` engine.
* Various DC blockers and state variables.

The processing order within the `DSPCore` is typically:
`Input -> DC Blocker -> Oversampling (Up) -> [Convolver <-> EQ] -> Soft Clipper -> Oversampling (Down) -> DC Blocker -> Dither -> Output`

The order of the Convolver and EQ is user-configurable.

## 4. Key DSP Components

### 4.1. `ConvolverProcessor` & `MKLNonUniformConvolver`

Convolution is handled by a highly optimized, custom engine built on Intel MKL.

* **Non-Uniform Partitioned Convolution**: The engine uses a 3-layer, non-uniform partitioning scheme to balance low latency with the ability to handle very long impulse responses efficiently.
  * **Layer 0 (Immediate)**: Processes the initial part of the IR with a small partition size for low latency. This is executed in every audio callback.
  * **Layer 1 & 2 (Delayed)**: Process the longer tail of the IR with progressively larger partition sizes. Their processing is distributed over multiple audio callbacks to prevent CPU spikes.
* **Optimization**: The core convolution is performed in the frequency domain using MKL's DFTI (FFT) and VML (vector math) functions, with complex multiplication optimized using AVX2 FMA intrinsics.

### 4.2. `EQProcessor`

The 20-band parametric equalizer also employs a lock-free RCU pattern for updating its filter coefficients.

* **Filter Topology**: It uses a **Topology-Preserving Transform (TPT) State-Variable Filter (SVF)**. This filter topology is known for its stability and robustness, especially when coefficients are being modulated.
* **Coefficient Updates**: When a user changes an EQ parameter, a new `BandNode` (containing the filter coefficients) is created on the Message Thread and atomically swapped in. The Audio Thread always reads the latest valid set of coefficients without locking.
* **Automatic Gain Control (AGC)**: An optional AGC feature is available to automatically compensate for gain changes introduced by the EQ, maintaining a consistent output level.

### 4.3. `CustomInputOversampler`

A custom polyphase FIR oversampler is used to run the core DSP at a higher sample rate, reducing aliasing from non-linear processes like soft clipping. It supports both IIR-like (low latency) and Linear Phase filter presets.

### 4.4. Memory Management

Performance-critical audio buffers are allocated using `mkl_malloc` to ensure 64-byte alignment, which is optimal for AVX/AVX2/AVX-512 instructions. The `convo::ScopedAlignedPtr` RAII wrapper is used to manage the lifetime of this memory safely.

## 5. UI Architecture

The UI is built with standard JUCE components.

* **`MainWindow`**: Owns all top-level UI components and the `AudioDeviceManager`.
* **Control Panels**: `EQControlPanel` and `ConvolverControlPanel` provide the user interface for manipulating DSP parameters. They communicate with the `AudioEngine` through its public API.
* **Data Flow for Visualization**:
  * The `AudioEngine`'s `DSPCore` pushes processed audio samples (either pre- or post-DSP, as selected by the user) into a lock-free, single-producer, single-consumer (SPSC) FIFO queue (`juce::AbstractFifo`).
  * The `SpectrumAnalyzerComponent`, running on the Message Thread via a `juce::Timer`, reads data from this FIFO to compute the FFT and draw the spectrum. This decouples the real-time audio processing from the UI rendering.

## 6. Build & Dependencies

* **Build System**: CMake
* **Language Standard**: C++20
* **Core Dependencies**:
  * JUCE 8.0.12
  * Intel® oneAPI Math Kernel Library (oneMKL)
  * r8brain-free-src (for high-quality IR resampling)
* **Compiler Optimizations**: The project is configured to build with high levels of optimization, including enabling the **AVX2 instruction set** (`/arch:AVX2` on MSVC) for significant performance gains in DSP code. MKL threading is explicitly disabled (`mkl_set_num_threads(1)`) to ensure predictable, low-latency performance suitable for real-time audio.

## 7. License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. Please see the `LICENSE.txt` file for details. Third-party libraries (JUCE, MKL, r8brain) are subject to their own respective licenses.
