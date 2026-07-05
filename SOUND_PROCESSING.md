# ConvoPeq Audio Signal Processing Guide

This document describes the complete audio signal processing flow in ConvoPeq v0.6.9, from external audio input to external audio output. It covers all major components, threading, buffer management, and real-time safety strategies.

**Project**: ConvoPeq v0.6.9 — IR Convolution + 20-band Parametric EQ + Real-Time Analyzer
**Stack**: JUCE 8.0.12 · Intel oneMKL (sequential) · Intel IPP · AVX2 · C++20
**Platform**: Windows 11 x64 · Real-time safe (no allocation/locks/libm/exceptions/I/O on audio thread)

---

## 1. Audio Input & Device Callback Entry Point

### 1.1 Platform Layer

- **JUCE Audio Device Manager** manages ASIO/WASAPI/DirectSound drivers and delivers audio via `AudioIODeviceCallback`.
- In ConvoPeq standalone mode, `AudioEngineProcessor` (a `juce::AudioProcessor` subclass) acts as the bridge.
- The audio device driver calls `AudioEngineProcessor::processBlock()` on its own high-priority thread (the **real-time audio thread**).

### 1.2 AudioEngineProcessor Entry

```cpp
// src/audioengine/AudioEngineProcessor.h
class AudioEngineProcessor final : public juce::AudioProcessor
{
    void processBlock(juce::AudioBuffer<float>& buffer,  juce::MidiBuffer&) override;
    void processBlock(juce::AudioBuffer<double>& buffer, juce::MidiBuffer&) override;
    bool supportsDoublePrecisionProcessing() const override { return true; }
};
```

- `AudioEngineProcessor::processBlock(float)` wraps the buffer in `juce::AudioSourceChannelInfo` and delegates to `AudioEngine::getNextAudioBlock()`.
- `AudioEngineProcessor::processBlock(double)` delegates to `AudioEngine::processBlockDouble()`.
- MIDI input is ignored (only audio is processed).

### 1.3 Buffer Structure & Lifetime

- Input buffer is a `juce::AudioBuffer<float>` or `juce::AudioBuffer<double>` — **planar (non-interleaved)**, one contiguous array per channel.
- Channels: typically 2 (stereo); mono is duplicated to both channels.
- **Buffer validity**: only for the duration of the callback. All processing must complete before return.
- The incoming buffer is **not guaranteed 64-byte aligned**, so internal processing copies samples into pre-allocated, 64-byte aligned double-precision working buffers (`alignedL`, `alignedR`).

### 1.4 Double-Precision Internals

- All internal DSP processing is performed in **double-precision (`double`)**, regardless of input format.
- If the input is `float`, samples are promoted to `double` during the copy to aligned buffers.
- Using `double` throughout ensures sufficient precision for the entire processing chain.

### 1.5 Real-Time Thread Safety at Entry

- `AudioEngine::processBlockDouble()` immediately acquires:
  - `lifecycleToken` from `lifecycleRuntime_.enterAudioCallback()` — engine lifecycle state gate
  - `firewallToken` from `rtCapabilityFirewall_.enter()` — RT capability verification
  - `ScopedNoDenormals` — flush denormals to zero for the entire callback duration
  - `ThreadRole::AudioRealtime` scoped role marker for numeric policy verification
- If `lifecycleState != EngineLifecycleState::Prepared`, the buffer is cleared and the function returns immediately.
- If shutdown is in progress, `shutdownRuntime_.markLateCallback()` is called and the buffer is cleared.

### 1.6 DSPCore Process Entry

- `DSPCore::process()` is the main audio-thread processing function.
- It receives the aligned double-precision buffers and a `ProcessingState` snapshot (all parameters atomically loaded at block start).
- All processing occurs within `DSPCore::process()` on the audio thread.

---

## 2. Input Conditioning

Input conditioning is the first DSP stage after buffer alignment. It prepares the signal without altering the signal content in a way that affects downstream processing fundamentally.

### 2.1 Headroom Gain

- **Purpose**: Prevents internal DSP clipping by scaling input samples by a configurable gain (typically -3 dB to -6 dB).
- **Implementation**: `scaleBlockFallback()` in `AudioEngine.Processing.DSPCoreDouble.cpp`:
  ```cpp
  inline void scaleBlockFallback(double* data, int numSamples, double gain) noexcept
  {
      // AVX2: 4 doubles per instruction
      const __m256d vGain = _mm256_set1_pd(gain);
      for (int i = 0; i < numSamples/4*4; i += 4)
          _mm256_storeu_pd(data+i, _mm256_mul_pd(_mm256_loadu_pd(data+i), vGain));
      // scalar tail
  }
  ```
- **No `libm` calls**: the gain is pre-converted from dB to linear in the message thread; the audio thread only performs multiplication.
- **No dynamic allocation**: all buffers are pre-allocated and 64-byte aligned.

### 2.2 Input DC Blocking — UltraHighRateDCBlocker

- **Purpose**: Removes DC offset from each channel using a **two-stage first-order IIR high-pass filter**.
- **Class**: `UltraHighRateDCBlocker` (header-only, `src/UltraHighRateDCBlocker.h`).
- **Structure**: Two cascaded 1st-order IIR sections with slightly different cutoff frequencies for minimal phase distortion.
- **Coefficients**: computed once in `prepareToPlay()` (message thread), using `std::sin`/`std::cos`. On the audio thread, only the difference equation is evaluated.
- **State**: maintained per channel, per block. Filter states are stored in aligned memory.
- **SIMD safety**: no `libm` calls, no dynamic allocation, no branching on audio thread.
- **Real-time guarantee**: all state variables are pre-allocated and initialized to zero in `prepareToPlay()`.

### 2.3 Analyzer Input Tap (Lock-Free FIFO)

- **Purpose**: Captures raw input samples (pre-gain, pre-DC block) for the UI spectrum analyzer.
- **Implementation**: `LockFreeRingBuffer<AudioBlock, 4096>` — a **SPSC (single-producer, single-consumer)** lock-free ring buffer.
  ```cpp
  // Audio thread: split into 256-sample blocks and push
  pushAdaptiveCaptureBlocks(captureQueue, alignedL, alignedR,
                              numSamples, sampleRate, bitDepth, coeffBankIndex);
  ```
- **Mechanism**:
  - `AudioBlock` contains 256 samples (double-precision, planar L/R).
  - `memcpy` is used to copy samples — no dynamic allocation.
  - Atomic head/tail counters ensure the SPSC property.
  - `readFromFifo()` in the UI thread consumes data — also lock-free for the audio thread.
- **Cache-line alignment**: ring buffer and control structures are `alignas(64)` to prevent false sharing.

---

## 3. Oversampling

### 3.1 Overview

Oversampling increases the internal sample rate (2×, 4×, or 8×) to reduce aliasing in subsequent DSP stages (particularly the EQ and convolution). It is implemented in `CustomInputOversampler`.

### 3.2 Structure & Initialization

- **Class**: `CustomInputOversampler` (`src/CustomInputOversampler.h/cpp`).
- **Stages**: Up to 3 cascaded 2× stages (maximum 8× oversampling).
- **Presets**:
  - `IIRLike`: lower latency, fewer FIR taps per stage.
  - `LinearPhase`: more taps, higher stopband attenuation.
- **All working buffers** (`workA`, `workB`, histories, coefficient arrays) are **pre-allocated and 64-byte aligned** in `prepare()`.
- `prepare()` (called from message thread before audio starts):
  - Computes FIR coefficients using a Kaiser windowed sinc kernel.
  - Determines tap count and attenuation based on preset and stage.
  - Normalizes coefficients for unity gain at DC.
  - Clears all history buffers.

### 3.3 Upsampling — processUp()

```cpp
juce::dsp::AudioBlock<double> processUp(juce::dsp::AudioBlock<double>& inputBlock,
                                        int numChannels) noexcept;
```

- For each 2× stage, calls `interpolateStage()` per channel:
  - Applies FIR interpolation using precomputed coefficients.
  - Uses **AVX2/FMA** SIMD for dot product operations.
  - Explicitly flushes denormal numbers to zero (`killDenormal`).
- Each stage doubles the sample count.
- Final output: double-precision, planar, 64-byte aligned buffer.

### 3.4 Post-Upsampling DC Blocker

After upsampling, `UltraHighRateDCBlocker` is applied to remove any DC introduced by interpolation. This is a separate instance from the input DC blocker and is configured for the higher sample rate.

### 3.5 Downsampling — processDown()

```cpp
void processDown(const juce::dsp::AudioBlock<double>& upsampledBlock,
                 juce::dsp::AudioBlock<double>& outputBlock,
                 int numChannels) noexcept;
```

- For each 2× stage (in reverse order), calls `decimateStage()` per channel:
  - Applies FIR decimation using precomputed coefficients.
  - Uses **AVX2/FMA** SIMD for convolution.
  - Explicitly flushes denormal numbers to zero.
- Each stage halves the sample count.
- **Fail-safe**: if input size exceeds pre-allocated capacity, output is zeroed and processing is skipped (no crash, no allocation).

### 3.6 Real-Time Safety

- **No allocation, no locks, no `libm` calls** on the audio thread.
- All buffer and history management is explicit and pre-allocated.
- Denormal handling: all stages flush denormals to zero for performance.
- All up/downsampling is performed in double-precision.

### 3.7 Corruption Detection

`CustomInputOversampler` maintains a `corruptionDetected` atomic flag:
- If any stage detects numerical corruption (e.g., NaN, Inf), it sets the flag.
- The flag can be consumed atomically by the UI thread via `consumeCorruptionFlag()`.
- Event counters track total corruption occurrences and auto-clear counts.

---

## 4. Main DSP Chain

### 4.1 Selectable Processing Order

The main chain can process in two orders, controlled by the atomic `ProcessingState` snapshot at block start:

- **EQ → Convolver**: EQ is applied first, then convolution.
- **Convolver → EQ**: Convolution is applied first, then EQ.
- **Mixed (Parallel)**: Both EQ and Convolver process the input independently in parallel; outputs are blended by a mix ratio α.

### 4.2 EQProcessor (20-Band Parametric EQ)

**Source**: `src/eqprocessor/` (6 files, split TU implementation)

| File | Size | Role |
|------|------|------|
| `EQProcessor.h` | 32.3 KB | Class definition, types, RCU handle |
| `EQProcessor.Core.cpp` | 42.4 KB | Core processing logic, M/S, AGC |
| `EQProcessor.Coefficients.cpp` | 19.3 KB | TPT SVF & biquad coefficient calculation |
| `EQProcessor.Parameters.cpp` | 12.7 KB | Parameter getters/setters |
| `EQProcessor.Processing.cpp` | 57.2 KB | **Largest TU** — AVX2 FMA TPT SVF processing |
| `EQProcessor.ProcessingCache.cpp` | 2.7 KB | EQCoeffCache management |

**Band configuration**:
- `NUM_BANDS = 20` (bands 0–19 with default frequencies from 25 Hz to 19.5 kHz)
- `kFilterChannels = 4` (L=0, R=1, Mid=2, Side=3 for M/S processing)
- Filter types: `LowShelf`, `Peaking`, `HighShelf`, `LowPass`, `HighPass`
- Channel modes: `Stereo`, `Left`, `Right`, `Mid`, `Side`
- Filter structures: `Serial` (default) or `Parallel`

**Filter implementation** — TPT (Topology-Preserving Transform) State Variable Filter:
- Based on Vadim Zavalishin's "The Art of VA Filter Design".
- Coefficients (`g`, `k`, `a1`, `a2`, `a3`, `m0`, `m1`, `m2`) preserve filter state topology under coefficient modulation, providing smooth parameter changes and low noise.
- For UI magnitude response display, separate **biquad coefficients** (RBJ Audio EQ Cookbook) are computed.
- SVF state: `ic1eq`, `ic2eq` (two integrator states) per band per channel, stored in `filterState[4][20][2]`.

**RCU + Atomic Parameter Update**:
- `currentStateBits`: `uintptr_t`-backed atomic handle for the whole `EQState`.
- `bandNodeBits[20]`: per-band atomic handles for individual `BandNode` objects.
- `publishAtomic` / `consumeAtomic` / `compareExchangeAtomic` primitives from `AtomicAccess.h`.
- Message thread: creates new `EQState`/`BandNode`, calls `publishCurrentState()`.
- Audio thread: `loadCurrentState()` with `acquire` order reads the latest snapshot.
- Old states retired via `retireEQStateDeferred()` / `retireBandNodeDeferred()` → `enqueueDeferredDeleteWithFallback()` → `DeletionQueue`.

**EQCoeffCache (Phase 1 v2.3)**:
- `EQCoeffCache` is a `RefCountedDeferred<EQCoeffCache>` — refcounted, shared coefficient cache.
- Multiple snapshots share the same `EQCoeffCache` if parameters are identical.
- Hash-based lookup via `computeParamsHash()`.
- Reduces coefficient recalculation when multiple `RuntimeState` snapshots share EQ parameters.

**Auto Gain Control (AGC)**:
- Attack time: **0.2 s**, release time: **2.0 s**, smooth time: **0.2 s**.
- AGC gain range: −24 dB to +24 dB (linear: 0.06 to 16.0).
- Attack/release/smooth coefficients precomputed as lookup tables (`agcAttackCoeffTable`, `agcReleaseCoeffTable`, `agcSmoothCoeffTable`) in `prepareToPlay()` — no `libm` calls on audio thread.
- Per-block: computes input/output envelope via RMS, applies adaptive gain.

**Nonlinear Saturation**:
- Uses `fastTanh` approximation (rational polynomial, branchless for |x| < CLIP_THRESHOLD).
- Applied per-sample within the SVF processing loop when `nonlinearSaturation > 0`.
- No `libm` calls, no branching on audio thread.

**Bypass Crossfade**:
- `BYPASS_FADE_TIME_SEC = 0.005` (5 ms) linear fade.
- `bypassFadeGain` (a `LinearRamp`) ramps from 0 to 1 on engage, 1 to 0 on disengage.

**Total Gain Smoothing**:
- `SMOOTHING_TIME_SEC = 0.05` (50 ms) exponential ramp.
- `totalGainTarget` stored as linear gain (converted in message thread — no `std::pow` on audio thread).

**M/S (Mid-Side) Processing**:
- Convert L/R to Mid/Side in the scratch buffer.
- Process each band with appropriate `EQChannelMode` (Stereo: both, Left: L only, Right: R only, Mid: M only, Side: S only).
- Convert back to L/R.

**Parallel Filter Structure**:
- When `FilterStructure::Parallel` is selected, bands process in parallel paths and are summed.
- `parallelInputBuffer`, `parallelWorkBuffer`, `parallelAccumBuffer` are pre-allocated.
- Crossfade between Serial and Parallel structures: `structureOldOutBuffer` / `structureNewOutBuffer` with `structureXfadeBufferCapacity`.

### 4.3 ConvolverProcessor (FFT Convolution Engine)

**Source**: `src/convolver/` (10 files, split TU implementation)

| File | Role |
|------|------|
| `ConvolverProcessor.Internal.h` | Helpers: `unwrapPhaseRadians`, `nextPow2`, `resampleIR`, `convertToMinimumPhase` |
| `ConvolverProcessor.Lifecycle.cpp` | Lifecycle, RCU integration, `ChangeBroadcaster` |
| `ConvolverProcessor.Rebuild.cpp` | Rebuild decision, debouncing (`REBUILD_DEBOUNCE_DEFAULT_MS`) |
| `ConvolverProcessor.LoaderThread.cpp` + `LoaderThreadInline.h` | Background IR loading, progress tracking |
| `ConvolverProcessor.LoadPipeline.cpp` | Pipeline processing, IR validation |
| `ConvolverProcessor.MixedPhase.cpp` | Phase modes, mixed-phase transition |
| `ConvolverProcessor.ResampleAndFallback.cpp` | r8brain resampling, hard fallback |
| `ConvolverProcessor.Runtime.cpp` | **Audio thread** — partitioned FFT convolution via MKL NUC |
| `ConvolverProcessor.StateAndUI.cpp` | Preset management, UI state |
| `ConvolverProcessor.h` (at `src/` root, 1180 lines) | Public API, `BuildSnapshot`, `PhaseMode`, `ResamplingPhaseMode` |

The **legacy monolithic** `MKLNonUniformConvolver.cpp` (~65 KB) is retained under `#ifdef` for backward compatibility.

**Algorithm** — Intel MKL Non-Uniform Partitioned Convolution (NUC):
- Partitions the impulse response into non-uniform blocks (shorter blocks near the start for low-latency, longer blocks toward the tail for efficiency).
- Uses **MKL DFTI** (Discrete Fourier Transform Interface) for forward/backward FFTs.
- All FFT and buffer operations use **AVX2/FMA SIMD**.
- 64-byte aligned buffers throughout.

**Phase Modes**:
- `PhaseMode::AsIs`: use IR as-is (linear phase).
- `PhaseMode::Minimum`: convert IR to minimum phase.
- `PhaseMode::Mixed`: blend linear-phase (low frequencies) with minimum-phase (high frequencies) — controlled by `mixedTransitionStartHz` and `mixedTransitionEndHz`.

**IR Loading (Background)**:
- `loadImpulseResponse()` is called from the **message thread**.
- A `LoaderThread` (`std::thread`) performs the actual file read, resampling (via `r8brain-free-src`), and phase conversion **asynchronously**.
- On completion, the new IR state is atomically swapped via **RCU** — audio thread continues processing with the old IR without interruption or blocking.
- Old convolution engines are retired via the ISR retire pipeline (`DSPLifetimeManager → ISRRetireRouter → EpochDomain`).

**Audio Thread Runtime** (`ConvolverProcessor.Runtime.cpp`):
```cpp
// At block start (audio thread): RCU load of latest IR state
auto* currentIR = irState.load(std::memory_order_acquire);
// For each channel: MKL NUC partitioned convolution
mklNUC.process(channelBuffer, currentIR, ...);
// Dry/wet mix, crossfade, latency compensation — all atomic, no allocation
```

**Parameters** (all atomic, snapshotted at block start):
- `mix` (dry/wet)
- `bypassed`
- `phaseMode`
- `smoothingTimeSec`
- `targetIRLengthSec`
- `mixedTransitionStartHz` / `mixedTransitionEndHz`

**Thread Safety**:
- No `malloc`/`new`/`resize` on audio thread.
- No `libm` calls on audio thread.
- No locks on audio thread.
- Old engines/IRs garbage collected asynchronously via the ISR retire pipeline.

### 4.4 Mixed (Parallel) Mode

When `processingMode == Mixed`, input is processed in parallel by EQ and Convolver, then blended:

```
y[n] = (1 − α) · EQ(x[n]) + α · Convolver(x[n])
```

- `α` (mix ratio): atomic variable, snapshotted at block start, smoothed for click-free transitions.
- EQ path: `eqBuffer` (separate pre-allocated working buffer).
- Convolver path: `convBuffer` (separate pre-allocated working buffer).
- Both paths operate on independent 64-byte aligned buffers.
- Blending: AVX2 SIMD on the output buffer.
- No dynamic allocation.

---

## 5. Output Conditioning

### 5.1 OutputFilter — Conditional on Final Processor

**Source**: `src/OutputFilter.h/cpp`

The output filter configuration depends on **which processor is last in the chain**:

#### Case ① — Convolver is Last

- **High-Cut Filter (HCF)**:
  - `Sharp`: Butterworth 4th-order cascaded (Q1=0.5412, Q2=1.3066) — steep, maintains sound pressure at cutoff
  - `Natural`: Linkwitz-Riley 4th-order (Q=0.7071 for both stages) — better phase response, default
  - `Soft`: 2nd-order (Q=0.5) — gentle, no time-domain smear
  - Cutoff: **19 kHz** (fs ≤ 48 kHz) / **22 kHz** (fs > 48 kHz)

- **Low-Cut Filter (LCF)**:
  - `Natural`: Butterworth 2nd-order **HPF, 18 Hz** — minimum low-frequency distortion
  - `Soft`: 2nd-order HPF, Q=0.5, **15 Hz** — gentler, more subsonic removal

#### Case ② — EQ is Last

- **High-Pass Filter (HPF)**: Fixed Butterworth 2nd-order, **20 Hz** (always).
- **Low-Pass Filter (LPF)**:
  - `Sharp`: Q=1.0 (two cascaded stages)
  - `Natural`: Q=0.7071 (two cascaded stages)
  - `Soft`: Q=0.5 (two cascaded stages)
  - Cutoff: **19 kHz** (fs ≤ 48 kHz) / **24 kHz** (fs > 48 kHz)

**Coefficient Precomputation** (`prepare()`, message thread only):
- Uses **RBJ Audio EQ Cookbook** biquad formulas with `std::sin`/`std::cos`.
- All coefficients for all modes and configurations are precomputed and stored in lookup tables.
- `process()` (audio thread): only performs the **Direct Form II Transposed** difference equation — no `libm` calls.

**Biquad Structure**:
```
y[n] = b0·x[n] + w1[n-1]
w1[n] = b1·x[n] − a1·y[n] + w2[n-1]
w2[n] = b2·x[n] − a2·y[n]
```
State (`w1`, `w2`) maintained per channel per stage, zeroed in `reset()`.

**SIMD**: all state and coefficient arrays are 64-byte aligned for AVX2 efficiency.

### 5.2 Soft Clipping

**Source**: `AudioEngine.Processing.DSPCoreDouble.cpp` — `softClipBlockAVX2()`, `musicalSoftClipScalar()`, `fastTanh()`

**Algorithm** — piecewise function (NOT simple cubic saturation):

For input `x` with `threshold`, `knee`, `asymmetry`:
- `clip_start = threshold − knee`
- If `|x| < clip_start`: **linear** — return `x` unchanged
- If `clip_start ≤ |x| < threshold + knee`: **knee region**
  - `t = (|x| − clip_start) / (2·knee)` → `t ∈ [0, 1]`
  - `knee_shape = t² · (3 − 2t)` — smooth S-curve
  - `clipped = threshold + knee · fastTanh((|x| − threshold) / knee)`
  - `mixed = |x| · (1 − knee_shape) + clipped · knee_shape`
- If `|x| ≥ threshold + knee`: **saturation region**
  - `mixed = threshold + knee · fastTanh((|x| − threshold) / knee)`
- `asymmetric_gain = 1 − asymmetry · (1 − sign) · 0.5 · knee_shape`
- `return sign · mixed · asymmetric_gain`

**fastTanh Approximation** (branchless, no `libm`):
```cpp
// Valid for |x| < CLIP_THRESHOLD (branchless)
const double x2 = x * x;
const double num = x * (NUM_A + x2 * (NUM_B + x2 * NUM_C));
const double den = DEN_A + x2 * (DEN_B + x2 * (DEN_C + x2));
return num / den;
```
- For `x ≥ CLIP_THRESHOLD`: returns ±1.0
- For `x ≤ −CLIP_THRESHOLD`: returns ∓1.0

**AVX2 Vectorized Version** (`softClipBlockAVX2`):
- Processes 4 doubles per instruction using `_mm256_*` intrinsics.
- `prevSampleInOut` scalar feedback carried across blocks (to maintain state across calls).
- `_mm256_set_pd` / `_mm256_mul_pd` / `_mm256_fnmadd_pd` / `_mm256_blendv_pd` etc.
- `_mm256_round_pd(..., _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)` for nearest-even quantization.

**Parameters** (all atomic, snapshotted at block start):
- `threshold` (default ~1.0)
- `knee` (default ~0.2)
- `asymmetry` (default ~0.0)

**Real-time safety**:
- No `libm` calls (fastTanh is a rational polynomial approximation).
- No dynamic allocation.
- Parameters pre-validated and clamped in the message thread.

### 5.3 Output Makeup Gain

- **Purpose**: Final gain adjustment after filtering and soft clipping to match target output level.
- **Implementation**: AVX2 SIMD scaling using the same `scaleBlockFallback()` function as input headroom gain.
- **Gain value**: atomic, converted from dB to linear in the message thread (no `std::pow` on audio thread).
- All buffers 64-byte aligned.

---

## 6. Dithering & Noise Shaping

All four noise shaper types share these properties:
- Double-precision processing throughout.
- All state and buffers **pre-allocated and 64-byte aligned** for SIMD.
- **No dynamic allocation, no locks** on the audio thread.
- **TPDF (Triangular Probability Density Function) dither** — two uniform random samples summed.
- **Error-feedback** topology: the quantization error is fed back and shaped by the noise shaper coefficients.

### 6.1 PsychoacousticDither (GUI: "9th-order")

**Source**: `src/PsychoacousticDither.h`

Despite the GUI label "9th-order", the actual order is **NS_ORDER = 12** (12-tap error-feedback). The discrepancy exists because the coefficient table (`kCoeffTable`) design uses only 9 independently variable coefficients per preset, while the remaining 3 coefficients are fixed for stability.

**Architecture**:
- **12th-order error-feedback** noise shaper (NS_ORDER = 12).
- **TPDF dither** via **MKL VSL** (`vdRngUniform`, BRNG = `VSL_BRNG_SFMT19937`) or **Xoshiro256\*\* fallback** if MKL VSL is unavailable.
- **RNG ring buffer**: 65,536 entries per channel, SPSC, pre-filled.
  - `rngRing[2][65536]` with atomic `rngReadPos[2]` / `rngWritePos[2]`.
  - Refilled by worker thread (non-RT) via `refillRandomRingNonRt()`.
  - `fillChunkForChannel()` uses `vdRngUniform` for bulk fill.

**Coefficient Table** (`kCoeffTable[6][3][12]`):
- **6 sample rate bands** × **3 bit depth presets** × **12 coefficients**.
- SR bands:
  - Band 0: 44.1 kHz
  - Band 1: 48 kHz
  - Band 2: 96 kHz
  - Band 3: 176.4 / 192 kHz
  - Band 4: 352.8 / 384 kHz
  - Band 5: 705.6 kHz+
- Bit depth presets:
  - 0: "16-bit strong" — pushes noise aggressively into ultrasonic range
  - 1: "24-bit standard" — balanced, POW-r #3 class
  - 2: "32-bit mild" — gentle shape since floor noise is already low

**Block Processing** (`processStereoBlock`):
- Unrolls 12 coefficients into local scalars (avoids repeated memory loads).
- Computes `shapedErrorL/R = Σ c_k · z_k[n−k−1]` for 12 taps.
- Generates TPDF dither: `dL = (u1 − 0.5) + (u2 − 0.5)` from the ring buffer.
- `tmp = (sample × headroom) + d + shapedError`.
- **SSE4.1 quantization**: `_mm_round_pd(v_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)` for nearest-even rounding.
- Error shift-register: `z_k[n+1] = z_{k−1}[n]` (12-tap shift left), `z_0[n+1] = error` with denormal kill.

**Error Feedback Stability**:
- Uses error-feedback (not direct-form IIR) topology — all poles are at z=0 → **always BIBO stable**.
- Maximum shaping error = `(scale/2) × Σ|c_k|` — worst case at 705.6 kHz / 16-bit ≈ −66 dBFS.

### 6.2 FixedNoiseShaper (GUI: "4th-order")

**Source**: `src/FixedNoiseShaper.h`

- **4th-order error-feedback** noise shaper.
- Coefficients: psychoacoustically tuned, sum to 1.0 for stability (e.g., `{0.46, 0.28, 0.17, 0.09}`).
- TPDF dither added before quantization.
- All state 64-byte aligned, pre-allocated.
- Diagnostics (RMS/peak error) computed in background thread.
- **SIMD fallback**: scalar processing when AVX2 is not beneficial.

### 6.3 Fixed15TapNoiseShaper (GUI: "15th-order")

**Source**: `src/Fixed15TapNoiseShaper.h`

- **15th-order error-feedback** noise shaper.
- Coefficients: `kFixed15TapNoiseShaperTunedCoeffs` (psychoacoustically optimized).
- TPDF dither added before quantization.
- All state 64-byte aligned, pre-allocated.
- Diagnostics (RMS/peak error) computed in background thread.

### 6.4 AdaptiveNoiseShaper — NoiseShaperLearner (GUI: "9th-order adaptive")

**Source**: `src/NoiseShaperLearner.h/cpp` (68.4 KB — largest source file in the project)

**Structure** — Lattice (ladder) filter:
- Order = 9 (`LatticeNoiseShaper::kOrder`).
- Lattice topology: reflection coefficients stay inside the unit circle by construction → unconditionally stable.
- Error feedback via lattice structure, not direct-form IIR.

**CMA-ES Optimization**:
- Covariance Matrix Adaptation Evolution Strategy running on a **dedicated worker thread** (not the audio thread).
- Receives audio blocks via `LockFreeRingBuffer<AudioSegment, N>` from the audio thread.
- `AudioSegment`: `double left[kFftLength]`, `double right[kFftLength]`, `std::array<double, kSpectrumBins> maskingThresholds`.
- For each generation: evaluates candidate coefficient sets by simulating the noise shaper and computing weighted error (psychoacoustic masking or A-weighting).
- Updates mean and covariance of the coefficient distribution.
- Converges over 10–80 minutes depending on mode (Short/Middle/Long).

**Coefficient Handoff** (RCU pattern):
- Best coefficient set published as `LearnedState` via atomic generation counter.
- Audio thread: `RCUReader` enters epoch, reads current `LearnedState`, processes, exits epoch.
- No blocking, no locks, no allocation on audio thread.

**Bank Management**:
- Coefficient banks keyed by `StateKey` (sample rate, bit depth, mode).
- Each bank stores current best coefficients, learning history, progress metrics.
- Resume / stop / save / load per bank.
- UI polls `Status`, `Progress`, `State` atomically.

**Stability Guarantee**:
- Reflection coefficients constrained to stay inside the unit circle.
- Lattice structure guarantees BIBO stability regardless of coefficient values.

---

## 7. Downsampling (Post-DSP)

If oversampling was enabled, `CustomInputOversampler::processDown()` is called after the main DSP chain and before output level measurement.

- **Multi-stage FIR decimation** in reverse order of upsampling.
- **Kaiser windowed sinc** kernel for each stage.
- **AVX2/FMA SIMD** for convolution (multiple accumulators to hide FMA latency).
- **Denormal flush to zero** in all stages.
- **Fail-safe**: if input size exceeds pre-allocated capacity, output is zeroed and processing is skipped.
- All buffers pre-allocated, 64-byte aligned.
- In-place operation — no extra memory allocation.

---

## 8. Output Level Measurement & Analysis

### 8.1 LoudnessMeter — ITU-R BS.1770-4/5

**Source**: `src/LoudnessMeter.h`

- **K-weighting**: two-stage biquad (pre-filter high-shelf + RLB high-pass) per ITU-R BS.1770-4 Table 1.
  - Pre-filter coefficients: `{1.535124859586970, −2.691696189406380, 1.198392810852850, −1.690659293182410, 0.732480774215850}`
  - RLB filter coefficients: `{1.0, −2.0, 1.0, −1.990047454833980, 0.990072250366210}`
- Channel weights: stereo = `{1.0, 1.0}`.
- **Block mean square** computed per channel, weighted and summed.
- Published to **worker thread** via `LockFreeRingBuffer<BlockPower, 4096>`.
- Worker thread aggregates Momentary / Short-term / Integrated loudness.
- All filter states per-channel, pre-allocated, 64-byte aligned.

### 8.2 TruePeakDetector — ITU-R BS.1770-4/5

**Source**: `src/TruePeakDetector.h`

- **4× oversampling** (2 stages of 2×) using linear-phase FIR interpolation.
- **63 taps** per stage (Kaiser window, Bessel I0 for side-lobe suppression) — exceeds ITU-R BS.1770-3 Example reference (48 taps).
- **AVX2 dot product** (`dotProductAvx2()`) for convolution.
- Peak hold published atomically for UI display.
- Measurement-only — no gain applied to the audio signal.

### 8.3 Spectrum Analyzer — UI Component

**Source**: `src/SpectrumAnalyzerComponent.h/cpp` (52.3 KB)

- FFT-based spectrum display.
- EQ response curve overlay.
- Peak hold and smoothing.
- Consumes data from the analyzer output tap (lock-free FIFO).

---

## 9. Real-Time Safety Architecture

### 9.1 Absolute Prohibitions on Audio Thread

ConvoPeq enforces these rules on every audio callback without exception:

| Prohibition | Rationale |
|-------------|-----------|
| No `malloc`, `new`, `std::vector::resize`, `unique_ptr` | Memory allocation may trigger OS lock or page fault |
| No `std::mutex`, `std::condition_variable`, or any blocking lock | May block indefinitely, causing audio dropout |
| No `std::log`, `std::exp`, `std::pow`, `std::sin`, `std::cos` (libm calls) | Variable-time execution; may trigger denormal flush |
| No exceptions | Stack unwinding is non-deterministic |
| No file I/O, network I/O | Unbounded latency |
| No `std::this_thread::sleep` or wait primitives | Blocking wait |

**Verification**: `ASSERT_AUDIO_THREAD()` and `convo::numeric_policy::ThreadRole::AudioRealtime` scope marker at the start of `processBlockDouble()`.

### 9.2 Atomic Parameter Updates

All parameter changes from the UI thread use **atomic publish/consume**:

```cpp
// Message thread: publish new value
convo::publishAtomic(param, newValue, std::memory_order_release);

// Audio thread: consume latest value
auto value = convo::consumeAtomic(param, std::memory_order_acquire);
```

Key atomics:
- `ProcessingState` snapshot: all parameters copied at block start into a struct — audio thread sees a consistent snapshot.
- `bypassRequested`, `totalGainTarget` (linear, pre-computed), `totalGainDbTarget`.
- `bandNodeBits[20]`, `currentStateBits` for EQ.
- `irState` generation counter for Convolver.
- `m_pendingAGCChange`, `agcResetSerial`, `bandResetPacked`.

### 9.3 RCU (Read-Copy-Update) Pattern

- **EpochDomain** (`src/core/EpochDomain.h`, 26 KB): manages named reader slots and a global epoch counter.
- **RCUReader** (`src/core/RCUReader.h`): RAII reader — enters epoch on construction, exits on destruction.
- **Publication flow**: message thread creates new object → `publishCurrentState()` → audio thread `loadCurrentState(acquire)` → old object retired via `enqueueRetire()`.
- **Retire pipeline**: `DSPLifetimeManager → ISRRetireRouter → EpochDomain →DeletionQueue`.
- Old objects are not deleted immediately — they are deferred until all in-flight readers exit the epoch.

### 9.4 Lock-Free Inter-Thread Communication

| Communication | Mechanism |
|---------------|-----------|
| UI parameter updates | `std::atomic` publish/consume |
| Analyzer data (audio → UI) | `LockFreeRingBuffer<SPSC>` |
| Loudness/TruePeak data (audio → UI) | `LockFreeRingBuffer<SPSC>` |
| Adaptive noise shaper (audio → worker) | `LockFreeRingBuffer<AudioSegment>` |
| Coefficient handoff (worker → audio) | RCU + atomic generation counter |
| IR handoff (loader → audio) | RCU + atomic `irState` |

### 9.5 Pre-Allocation Strategy

**All** buffers, states, and working memory are allocated in `prepareToPlay()`:

- `alignedL`, `alignedR` — 64-byte aligned input buffers.
- `eqBuffer`, `convBuffer`, `parallelInputBuffer`, `parallelWorkBuffer`, `parallelAccumBuffer`, `dryBypassBuffer`, `structureOldOutBuffer`, `structureNewOutBuffer`.
- `msWorkBuffer` (M/S processing).
- `scratchBuffer`, `filterState[4][20][2]` (EQ SVF integrators).
- `agcAttackCoeffTable`, `agcReleaseCoeffTable`, `agcSmoothCoeffTable`.
- All filter coefficients, lookup tables, and coefficient caches.

### 9.6 Asynchronous Garbage Collection

| Mechanism | Path |
|-----------|------|
| `DeferredDeletionQueue` | Thread-safe queue of delete requests |
| `RefCountedDeferred` | Reference-counted deferred delete |
| `DeferredFreeThread` | Dedicated background thread for actual deallocation |
| `enqueueDeferredDeleteWithFallback` | EO-style deferred delete with EpochDomain fallback |
| `retireEQStateDeferred` / `retireBandNodeDeferred` | Per-object EQ state retirement |
| `DSPLifetimeManager::retire()` | Orchestrates retire via `ISRRetireRouter` |
| `ISRRetireRouter::enqueueRetire()` | Queues retire to `EpochDomain` |

### 9.7 ISR Runtime Governance (101 files in `src/audioengine/`)

The ISR (Interrupt Service Routine-inspired) runtime governance layer manages DSP lifecycle, publication, crossfade, and health monitoring:

| Component | File | Role |
|-----------|------|------|
| `ISRLifecycle` | `ISRLifecycle.h/cpp` | Lifecycle state machine |
| `ISRRuntimePublicationCoordinator` | `ISRRuntimePublicationCoordinator.cpp` | Publication choreography |
| `ISRRetireRouter` | `ISRRetireRouter.h/cpp` | Unified retire API |
| `ISRRetireRuntimeEx` | `ISRRetireRuntimeEx.h/cpp` | Extended retire runtime |
| `RuntimeHealthMonitor` | `RuntimeHealthMonitor.h/cpp` | Continuous telemetry |
| `RuntimePolicyEngine` | `RuntimePolicyEngine.h/cpp` | Rebuild admission policy |
| `DSPLifetimeManager` | `DSPLifetimeManager.h` | DSP activation / retire / crossfade |
| `CrossfadeAuthority` | `CrossfadeAuthority.h/cpp` | Crossfade governance (Authority pattern) |
| `CrossfadeRuntime` | `CrossfadeRuntime.h` | Crossfade runtime state |
| `ISRHB` | `ISRHB.h/cpp` | Heartbeat and hazard barrier |
| `SnapshotCoordinator` | `SnapshotCoordinator.h/cpp` | Snapshot management |
| `SnapshotFactory` | `SnapshotFactory.h/cpp` | Snapshot creation |
| `CommandBuffer` | `CommandBuffer.h` | Debounced snapshot worker |
| `FadeEngine` | `FadeEngine.h` | Fade shape generation |

---

## 10. Complete Signal Flow Diagram

```
[JUCE Audio Device Callback]
        │
        ▼
[AudioEngineProcessor::processBlock (float or double)]
        │
        ▼
[AudioEngine::processBlockDouble]
        ├── AudioCallbackRuntimeScope (lifecycle + firewall token)
        ├── ScopedNoDenormals
        └── ThreadRole::AudioRealtime
        ▼
[DSPCore::process — Input Conditioning]
        │
        ├── 1. Copy to 64-byte aligned buffers (alignedL, alignedR)
        ├── 2. Headroom gain (AVX2 scaleBlockFallback)
        ├── 3. DC blocking (UltraHighRateDCBlocker × 2 stages)
        └── 4. Analyzer input tap (LockFreeRingBuffer SPSC)
        ▼
[Oversampling — CustomInputOversampler::processUp]
        (if enabled: 2×/4×/8× FIR, AVX2/FMA, Kaiser windowed sinc)
        ▼
[Main DSP Chain]
        │
        ├── [EQ → Convolver]
        │       ├── EQProcessor::process()
        │       │     (20-band TPT SVF, Serial/Parallel, M/S,
        │       │      AGC, nonlinearSaturation, fastTanh)
        │       │     RCU: load currentStateBits + bandNodeBits[20]
        │       │     EQCoeffCache (refcounted shared coefficient cache)
        │       │
        │       └── ConvolverProcessor::process()
        │             (MKL NUC partitioned FFT convolution)
        │             RCU: irState.load(acquire)
        │
        ├── [Convolver → EQ]
        │       ├── ConvolverProcessor::process()
        │       └── EQProcessor::process()
        │
        └── [Mixed — Parallel]
                ├── EQProcessor::process() → eqBuffer
                ├── ConvolverProcessor::process() → convBuffer
                └── y = (1−α)·eqBuffer + α·convBuffer (AVX2 SIMD)
        ▼
[OutputFilter (conditional on final processor)]
        │
        ├── Case ① (Convolver-last): HCF (Sharp/Natural/Soft 4th-order)
        │                          + LCF (Natural Butterworth 2nd HPF 18Hz
        │                              / Soft 2nd HPF Q=0.5 15Hz)
        │
        └── Case ② (EQ-last): HPF (Butterworth 2nd 20Hz, fixed)
        │                    + LPF (Sharp/Natural/Soft 2nd × 2 stages)
        │  All coefficients precomputed in prepare() (std::sin/cos),
        │  process() = table lookup only, no libm, in-place biquad SOS
        ▼
[Soft Clipping — softClipBlockAVX2]
        ├── piecewise: linear → knee (t²(3−2t)) → clipped (fastTanh)
        ├── fastTanh: rational polynomial approximation, branchless
        ├── AVX2: 4 doubles per instruction, _mm256_round_pd
        ├── prevSampleInOut scalar feedback across blocks
        └── Parameters: threshold, knee, asymmetry (atomic snapshot)
        ▼
[Output Makeup Gain]
        │  (AVX2 SIMD, atomic gain pre-converted to linear)
        ▼
[Dither / Noise Shaping]
        │
        ├── PsychoacousticDither: 12th-order (NS_ORDER=12)
        │     MKL VSL RNG ring buffer (65,536 × 2 ch) or Xoshiro fallback
        │     TPDF dither, kCoeffTable[6][3][12] (6 SR bands × 3 bitdepths)
        │     SSE4.1 stereo quantization, error shift-register
        │
        ├── FixedNoiseShaper: 4th-order error-feedback, psychoacoustic coeffs
        │
        ├── Fixed15TapNoiseShaper: 15th-order error-feedback
        │
        └── AdaptiveNoiseShaper (NoiseShaperLearner)
              LatticeNoiseShaper (9th-order, kOrder=9)
              CMA-ES optimizer on worker thread
              LockFreeRingBuffer<AudioSegment> for audio transfer
              RCU handoff of LearnedState to audio thread
        ▼
[Downsampling — CustomInputOversampler::processDown]
        (if oversampling: multi-stage FIR decimation, AVX2/FMA, fail-safe)
        ▼
[Output Level Measurement]
        │
        ├── LoudnessMeter: ITU-R BS.1770-4/5 K-weighting
        │     LockFreeRingBuffer publish to worker thread
        │
        └── TruePeakDetector: 4× OS, 63-tap FIR, AVX2 dot product
        ▼
[Analyzer Output Tap]
        │  (optional spectrum analysis via LockFreeRingBuffer)
        ▼
[DSPCore::processToBuffer — write to output buffer]
        │
        ▼
[JUCE Audio Device Output]
```

---

## 11. Key DSP Numeric Constants

| Constant | Value | Location |
|----------|-------|----------|
| `NUM_BANDS` | 20 | `EQProcessor.h` |
| `kFilterChannels` | 4 (L/R/Mid/Side) | `EQProcessor.h` |
| `NS_ORDER` | 12 | `PsychoacousticDither.h` |
| `LatticeNoiseShaper::kOrder` | 9 | `LatticeNoiseShaper.h` |
| `AGC_ATTACK_TIME_SEC` | 0.2 | `EQProcessor.h` |
| `AGC_RELEASE_TIME_SEC` | 2.0 | `EQProcessor.h` |
| `AGC_SMOOTH_TIME_SEC` | 0.2 | `EQProcessor.h` |
| `BYPASS_FADE_TIME_SEC` | 0.005 (5 ms) | `EQProcessor.h` |
| `SMOOTHING_TIME_SEC` | 0.05 (50 ms) | `EQProcessor.h` |
| `LoudnessMeter kMaxChannels` | 2 | `LoudnessMeter.h` |
| `TruePeakDetector kOversamplingRatio` | 4 | `TruePeakDetector.h` |
| `TruePeakDetector kDefaultTaps` | 63 | `TruePeakDetector.h` |
| `RNG_RING_SIZE` | 65,536 | `PsychoacousticDither.h` |
| `kDenormThresholdAudioState` | `~1e-300` | `DspNumericPolicy.h` |

---

## 12. Reference

### Core DSP Source Files

| File | Description |
|------|-------------|
| `src/audioengine/AudioEngineProcessor.{h,cpp}` | JUCE AudioProcessor entry, float/double dispatch |
| `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | Audio thread entry, lifecycle/firewall tokens |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | DSPCore, softClipBlockAVX2, scaleBlockFallback, fastTanh |
| `src/audioengine/AudioEngine.Processing.DSPCoreToBuffer.cpp` | Output buffer write |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | Float processing path |
| `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | Buffer allocation, filter initialization |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | Resource release |
| `src/audioengine/AudioEngine.Processing.Latency.cpp` | Latency reporting |
| `src/audioengine/AudioEngine.Processing.Snapshot.cpp` | Snapshot creation |
| `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` | DSP core lifecycle |
| `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` | I/O helpers |
| `src/eqprocessor/EQProcessor.h` | EQ class definition, 20-band, TPT SVF, RCU |
| `src/eqprocessor/EQProcessor.Core.cpp` | Core EQ logic, M/S, AGC |
| `src/eqprocessor/EQProcessor.Coefficients.cpp` | TPT SVF + RBJ biquad coefficient computation |
| `src/eqprocessor/EQProcessor.Parameters.cpp` | Parameter getters/setters |
| `src/eqprocessor/EQProcessor.Processing.cpp` | TPT SVF processing, AVX2 FMA (largest TU) |
| `src/eqprocessor/EQProcessor.ProcessingCache.cpp` | EQCoeffCache management |
| `src/convolver/ConvolverProcessor.Lifecycle.cpp` | RCU, lifecycle, ChangeBroadcaster |
| `src/convolver/ConvolverProcessor.Rebuild.cpp` | Rebuild decision, debouncing |
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | Background IR loading |
| `src/convolver/ConvolverProcessor.LoadPipeline.cpp` | Pipeline processing |
| `src/convolver/ConvolverProcessor.MixedPhase.cpp` | Phase modes, mixed-phase transition |
| `src/convolver/ConvolverProcessor.ResampleAndFallback.cpp` | r8brain resampling, hard fallback |
| `src/convolver/ConvolverProcessor.Runtime.cpp` | **Audio thread** MKL NUC partitioned convolution |
| `src/convolver/ConvolverProcessor.StateAndUI.cpp` | Preset management, UI state |
| `src/convolver/ConvolverProcessor.Internal.h` | Helper functions |
| `src/UltraHighRateDCBlocker.h` | Two-stage IIR DC blocker |
| `src/CustomInputOversampler.{h,cpp}` | FIR oversampler (2×/4×/8×) |
| `src/OutputFilter.{h,cpp}` | Output HCF/LCF/HPF/LPF (conditional) |
| `src/PsychoacousticDither.h` | 12th-order psychoacoustic dither |
| `src/FixedNoiseShaper.h` | 4th-order fixed noise shaper |
| `src/Fixed15TapNoiseShaper.h` | 15th-order fixed noise shaper |
| `src/LatticeNoiseShaper.h` | Lattice noise shaper structure (9th-order) |
| `src/NoiseShaperLearner.{h,cpp}` | CMA-ES adaptive learning (largest TU) |
| `src/LoudnessMeter.h` | ITU-R BS.1770-4/5 K-weighting |
| `src/TruePeakDetector.h` | 4× OS true peak, 63-tap FIR |
| `src/SpectrumAnalyzerComponent.{h,cpp}` | FFT spectrum UI |
| `src/MKLNonUniformConvolver.{h,cpp}` | Legacy MKL NUC (backward compat) |
| `src/AlignedAllocation.h` | 64-byte aligned `malloc`/`free` |
| `src/LockFreeRingBuffer.h` | SPSC lock-free ring buffer |
| `src/DspNumericPolicy.h` | Numeric constants, denorm thresholds |
| `src/audioengine/DSPLifetimeManager.h` | DSP lifecycle orchestration |
| `src/audioengine/CrossfadeAuthority.{h,cpp}` | Crossfade governance (Authority) |
| `src/audioengine/CrossfadeRuntime.h` | Crossfade runtime state |
| `src/audioengine/ISRRetireRouter.{h,cpp}` | Unified retire API |
| `src/audioengine/RuntimeHealthMonitor.{h,cpp}` | Telemetry / health monitoring |
| `src/audioengine/RuntimePolicyEngine.{h,cpp}` | Rebuild admission policy |
| `src/audioengine/AtomicAccess.h` | Atomic primitives (`publishAtomic`, etc.) |
| `src/core/EpochDomain.h` | RCU epoch domain (64 reader slots) |
| `src/core/RCUReader.h` | RAII RCU reader |
| `src/core/SnapshotCoordinator.{h,cpp}` | Snapshot coordination |
| `src/core/SnapshotFactory.{h,cpp}` | Snapshot creation |
| `src/core/FadeEngine.h` | Fade shape generation |
| `src/core/DeletionQueue.{h,cpp}` | Deferred deletion queue |

### Architecture Documents

- `ARCHITECTURE.md` — Overall system architecture, threading design
- `BUILD_GUIDE_WINDOWS.md` — Build instructions, toolchain setup
- `.github/copilot-instructions.md` — Coding standards, prohibitions
- `MEMORY_ALLOCATION_AUDIT.md` — Memory management design
- `doc/sourcecode_analysis_2026-07-03.md` — Detailed source code companion analysis

### External References

- **Vadim Zavalishin** — "The Art of VA Filter Design" (TPT SVF theory)
- **Robert Bristow-Johnson** — "Audio EQ Cookbook" (RBJ biquad formulas)
- **Intel oneMKL** — Non-Uniform Partitioned Convolution (NUC) engine
- **ITU-R BS.1770-4/5** — Loudness measurement (K-weighting)
- **ITU-R BS.1770-3** — True peak detection
- **Hansen, J. (2012)** — True peak detection literature
