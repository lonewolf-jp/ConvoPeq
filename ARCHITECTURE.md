# ConvoPeq v0.3.5 — Architecture Design Document

## Project Overview

**Name**: ConvoPeq (Convolution + Parametric EQ)
**Version**: v0.3.5
**Type**: Standalone audio application for Windows 11 x64 only
**Purpose**:

- Provide a mastering-grade audio processing environment that integrates a high-precision 20-band parametric equalizer (TPT SVF) and a zero-latency convolver (MKL NUC).
- Offer real-time correction for system audio and audio outside the DAW (room acoustic correction, headphone correction, etc.).

**Tech stack**:

- **Language**: C++20
- **Framework**: JUCE 8.0.12
- **Build system**: CMake 3.22+
- **Compiler**: MSVC 19.44+ (Visual Studio 2022)
- **Target OS**: Windows 11 x64 (AVX2 required)
- **External libraries**: Intel oneAPI Math Kernel Library (oneMKL) — FFT, vector math, RNG

## Core Design Principles

### 1. Strict real-time constraints and thread safety

To completely eliminate glitches on the audio processing thread (Audio Thread), the following rules are enforced:

- **Wait-free / Lock-free**: Blocking synchronization primitives such as `Mutex`, `CriticalSection`, `std::promise`, etc., are forbidden inside the Audio Thread.
- **No dynamic heap allocation**: Heap allocations such as `malloc`, `new`, `std::vector::resize` are forbidden. All buffers are preallocated during `DSPCore` construction (on the Message Thread / Worker Thread) using `ScopedAlignedPtr` (MKL allocator).
- **No system calls**: File I/O, console output, thread creation, and similar system calls are forbidden on the Audio Thread.
- **MKL configuration**: `mkl_set_num_threads(1)` and `mkl_set_dynamic(0)` are used to suppress MKL internal thread creation and prevent unpredictable latency.

### 2. State management using the RCU (Read-Copy-Update) pattern

A lock-free **RCU pattern** is used for parameter sharing between the UI/Worker threads and the Audio Thread.

- **Update (Writer)**: New state objects (`DSPCore`, `EQState`, `StereoConvolver`) are created on the heap and, after setup, the pointer is atomically swapped via `std::atomic<T*>` with `std::memory_order_release`.
- **Read (Reader / Audio Thread)**: The Audio Thread obtains a raw pointer via `std::atomic::load`. The Message Thread’s delayed-release mechanism (trash bin) guarantees the pointer remains valid during the Audio Thread’s processing cycles.
- **Delayed release (Garbage Collection)**: Old objects that are no longer referenced are moved to a `trashBin` list (managed by the Message Thread) and are safely destroyed after a configured delay (e.g., 2000 ms) based on timestamps.

### 3. Numerical stability and DSP quality

- **TPT SVF (Topology-Preserving Transform State Variable Filter)**: The EQ filters use the TPT SVF algorithm to eliminate high-frequency distortion and instability that can occur with conventional biquads during fast parameter modulation.
- **Denormal handling**: In addition to `juce::ScopedNoDenormals`, tiny values in IIR filter state variables are flushed to zero manually to prevent CPU load spikes. MKL VML mode is set to `VML_FTZDAZ_ON`.
- **NaN/Inf protection**: Detection and clamping are implemented to prevent invalid floating-point values (NaN/Inf) from propagating through the DSP chain, along with protection via an `UltraHighRateDCBlocker` (1st-order IIR).

### 4. Robust device management

- **ASIO blacklist**: ASIO drivers that are single-client only or otherwise unstable (e.g., BRAVO-HD, ASIO4ALL) are detected and automatically excluded.
- **Windows optimizations**: Timer precision is improved with `timeBeginPeriod(1)`, process priority is raised with `SetPriorityClass(HIGH_PRIORITY_CLASS)`, and Windows 11 efficiency modes (EcoQoS) are disabled to prevent audio dropouts.

## Component Design

### Overall structure diagram

```text
MainApplication
  │
  └─ MainWindow
       │
       ├─ AudioDeviceManager
       │    └─ AudioProcessorPlayer
       │         └─ AudioEngineProcessor (AudioProcessor)
       │              └─ AudioEngine (AudioSource)
       │         │
       │         ├─ DSPCore (Audio-thread processing container: RCU-managed)
       │         │    ├─ CustomInputOversampler (Polyphase IIR/FIR)
       │         │    ├─ UltraHighRateDCBlocker (DC removal)
       │         │    ├─ ConvolverProcessor (MKL NUC)
       │         │    ├─ EQProcessor (TPT SVF)
       │         │    ├─ SoftClipper (AVX2)
       │         │    └─ PsychoacousticDither (MKL VSL)
       │         │
       │         ├─ Rebuild Thread (Worker)
       │         │    └─ DSPCore construction and IR resampling
       │         │
       │         └─ UI State Instances (for Message Thread)
       │
       └─ UI Components (ConvolverControlPanel, EQControlPanel, SpectrumAnalyzer)
```

### Signal flow details

```text
Input Device
    ↓ (float) [IoCallback]
[IoCallback - Audio Thread]
    ↓ (convert to double + Headroom(-0.1dB) + Sanitize + Input DCBlocker: 3Hz)
AlignedBuffer (inside DSPCore)
    ↓
Input level measurement → inputLevelLinear (atomic)
    ↓
Oversampling (Up: 1x, 2x, 4x, 8x)
    ↓
Post-OS DCBlocker (1Hz)
    ↓
Analyzer Input Tap (Pre-DSP) → Lock-free FIFO
    ↓
┌──────────────────────────────────────┐
│  Processing order is configurable    │
│                                      │
│  Option 1: Conv → EQ                 │
│    ├─ ConvolverProcessor::process()  │
│    │    └─ MKLNonUniformConvolver    │ (Dry/Wet Mix, Latency Compensation)
│    │                                 │
│    └─ EQProcessor::process()         │
│         ├─ 20-band TPT SVF processing│
│         │   (AVX2 stereo optimized)  │
│         └─ Total Gain / AGC          │
│                                      │
│  Option 2: EQ → Conv                 │
│    (reverse order)                   │
└──────────────────────────────────────┘
    ↓
Soft Clipper (AVX2 optimized: tanh + polynomial)
    ↓
Analyzer Output Tap (Post-DSP) → Lock-free FIFO
    ↓
Oversampling (Down)
    ↓
Output level measurement → outputLevelLinear (atomic)
    ↓
Output DCBlocker (3Hz)
    ↓
Headroom(-0.1dB) + Psychoacoustic Dither (noise shaping)
    ↓
Convert to float + clamp
    ↓
Output Device
```

### AudioEngine & DSPCore

**Files**: `src/AudioEngine.cpp`, `src/AudioEngine.h`

#### DSPCore

A container for processing executed on the Audio Thread. Using the RCU pattern, when settings change a new `DSPCore` instance is constructed in the background and atomically swapped in.

- **Memory management**: Uses `ScopedAlignedPtr` to allocate memory with 64-byte alignment optimized for MKL/AVX2.
- **Buffer sizing**: Preallocates `SAFE_MAX_BLOCK_SIZE` (65536) * 8 (maximum oversampling factor) to eliminate runtime reallocations.

#### Rebuild Thread

Heavy operations such as sample-rate changes, buffer-size changes, oversampling configuration changes, and IR loading are executed in a dedicated `rebuildThreadLoop`.

1. The Message Thread issues `requestRebuild`.
2. The Worker Thread constructs a new `DSPCore`, performs memory allocation, IR resampling, and FFT plan creation.
3. Upon completion, `commitNewDSP` is called via the Message Thread to update the pointer.

### ConvolverProcessor

**Files**: `src/ConvolverProcessor.cpp`, `src/ConvolverProcessor.h`
**Engine**: `MKLNonUniformConvolver` (custom MKL implementation)

#### ConvolverProcessor features

1. **Impulse response loading**
   - Asynchronous loading via a `LoaderThread`.
   - **Preprocessing**:
     - Float → Double conversion (high quality)
     - Auto makeup gain (energy normalization)
     - Silence trimming (tail trimming)
     - Resampling (r8brain-free-src)
     - DC removal (1 Hz high-pass)
     - Asymmetric Tukey window (peak-referenced window)
     - Minimum-phase conversion (MKL FFT + cepstrum method, optional)

2. **MKL Non-Uniform Partitioned Convolution (NUC)**
   - Custom convolution engine using Intel MKL DFTI.
   - **Configuration**: Currently runs as a single-layer (uniform partitioned) configuration prioritizing stability.
   - **Optimizations**: Complex multiply-accumulate using AVX2 FMA.

3. **Latency compensation**
   - Dry-signal delay via ring buffer.
   - Crossfaded delay-time changes to avoid Doppler artifacts.

### EQProcessor

**Files**: `src/EQProcessor.cpp`, `src/EQProcessor.h`
**Filter type**: TPT SVF (Topology-Preserving Transform State Variable Filter)

#### EQProcessor features

1. **20-band parametric EQ**
   - LowShelf, Peaking, HighShelf, LowPass, HighPass.
   - Each band adjustable independently: frequency, gain, Q.

2. **AVX2 optimizations**
   - `processBandStereo`: Packs L/R channel coefficients and state variables into SIMD registers for simultaneous processing.

3. **Coefficient computation**
   - **Audio Thread**: Uses TPT SVF coefficients (`EQCoeffsSVF`) which are robust to time-varying changes and produce less automation noise.
   - **UI Thread**: Uses biquad coefficients (`EQCoeffsBiquad`) for display purposes.

4. **AGC (Auto Gain Control)**
   - Tracks input/output RMS levels and automatically adjusts total gain to maintain perceived loudness.

### CustomInputOversampler

**File**: `src/CustomInputOversampler.cpp`

- **Factors**: 1x, 2x, 4x, 8x (automatically limited according to sample rate).
- **Modes**: Selectable Polyphase IIR (low latency) or Linear Phase (linear-phase) implementations.
- **Implementation**: High-speed convolution using AVX2.

### PsychoacousticDither

**File**: `src/PsychoacousticDither.h`

- **RNG**: Intel MKL VSL (`VSL_BRNG_SFMT19937`) with `SplitMix64` seed generation.
- **Noise shaping**: 5th-order error-feedback topology (Lipshitz / Wannamaker style coefficients).
- **Processing**: Reduces audible quantization noise and shifts residual noise into ultrasonic bands.

### SpectrumAnalyzerComponent

**File**: `src/SpectrumAnalyzerComponent.cpp`

- **FFT**: Intel MKL DFTI (4096 points, single precision).
- **Rendering**: 60 fps timer-driven. Data is acquired via a lock-free FIFO.
- **Features**:
  - Switchable input/output source.
  - Peak hold and smoothing.
  - Overlay display of EQ response curves.

## Data Structures and Memory

### ScopedAlignedPtr

A smart pointer defined in `src/AlignedAllocation.h`.

- Wraps `mkl_malloc` / `mkl_free`.
- Guarantees 64-byte alignment (AVX-512 / AVX2 compatible).
- RAII-based automatic release.

### DeviceSettings

**File**: `src/DeviceSettings.cpp`
**Storage**: `%APPDATA%\ConvoPeq\device_settings.xml`

Stored settings include:

- Device type / ID
- Sample rate / buffer size
- Dither bit depth
- Oversampling settings
