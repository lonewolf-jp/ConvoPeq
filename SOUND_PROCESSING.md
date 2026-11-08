# ConvoPeq Audio Signal Processing Guide

This document describes the complete audio signal processing flow in ConvoPeq v0.6.11, from external audio input to external audio output. It covers all major components, threading, buffer management, and real-time safety strategies.

**Project**: ConvoPeq v0.6.11 — IR Convolution + 20-band Parametric EQ + Real-Time Analyzer
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

- `AudioEngineProcessor::processBlock(double)` delegates to `AudioEngine::processBlockDouble()`.
- The shipped app defines `CONVOPEQ_STANDALONE_ONLY`, and `MainWindow` calls `audioProcessorPlayer.setDoublePrecisionProcessing(true)` (MainWindow.cpp:282), so the double overload is always the one that runs. The float `processBlock(float)` is then a stub that clears the buffer (it delegates to `AudioEngine::getNextAudioBlock()` only when that macro is absent).
- MIDI input is ignored (only audio is processed).

### 1.3 Buffer Structure & Lifetime

- Input buffer is a `juce::AudioBuffer<float>` or `juce::AudioBuffer<double>` — **planar (non-interleaved)**, one contiguous array per channel.
- Channels: typically 2 (stereo); mono is duplicated to both channels.
- **Buffer validity**: only for the duration of the callback. All processing must complete before return.
- The incoming buffer is **not guaranteed 64-byte aligned**, so internal processing copies samples into pre-allocated, 64-byte aligned double-precision working buffers (`alignedL`, `alignedR`).

### 1.4 Double-Precision Internals

- All internal DSP processing is performed in **double-precision (`double`)**, regardless of input format.
- If the input is `float`, samples are promoted to `double` during the copy to the aligned buffers (AVX2 4-wide copy with a scalar tail).
- Using `double` throughout ensures sufficient precision for the entire processing chain.

### 1.5 Real-Time Thread Safety at Entry

- `AudioEngine::processBlockDouble()` immediately acquires:
  - `lifecycleToken` from `lifecycleRuntime_.enterAudioCallback()` — engine lifecycle state gate
  - `firewallToken` from `rtCapabilityFirewall_.enter()` — RT capability verification
  - `ensureThreadFloatingPointEnvironment()` — FTZ/DAZ is set once per thread (ADR-006); there is no per-block `ScopedNoDenormals` save/restore
  - `ThreadRole::AudioRealtime` scoped role marker for numeric policy verification
- If `lifecycleState != EngineLifecycleState::Prepared`, the buffer is cleared and the function returns immediately.
- If shutdown is in progress, `shutdownRuntime_.markLateCallback()` is called and the buffer is cleared.

### 1.6 DSPCore Process Entry

- `DSPCore::processDouble()` is the main audio-thread processing function for the double path (`DSPCore::process()` is the float/`AudioSourceChannelInfo` path).
- It receives the aligned double-precision buffers and a `ProcessingState` snapshot (all parameters atomically loaded at block start).
- All processing occurs within `DSPCore::processDouble()` on the audio thread.

---

## 2. Input Conditioning

Input conditioning is the first DSP stage after buffer alignment. It prepares the signal without altering the signal content in a way that affects downstream processing fundamentally.

### 2.1 Headroom Gain

- **Purpose**: Prevents internal DSP clipping by scaling input samples by a configurable gain (range −12…0 dB; −3 to −6 dB is typical).
- **Implementation**: `scaleBlockFallback()` in `AudioEngine.Processing.DSPCoreDouble.cpp`:
  ```cpp
  inline void scaleBlockFallback(double* data, int numSamples, double gain) noexcept
  {
      int i = 0;
      const int vEnd = numSamples / 4 * 4;
      const __m256d vGain = _mm256_set1_pd(gain);
      for (; i < vEnd; i += 4)
      {
          __m256d vData = _mm256_loadu_pd(data + i);
          _mm256_storeu_pd(data + i, _mm256_mul_pd(vData, vGain));
      }
      for (; i < numSamples; ++i)
          data[i] *= gain;
  }
  ```
- **No `libm` calls**: the gain is pre-converted from dB to linear in the message thread; the audio thread only performs multiplication.
- **No dynamic allocation**: all buffers are pre-allocated and 64-byte aligned.

### 2.2 Input DC Blocking — UltraHighRateDCBlocker

- **Purpose**: Removes DC offset from each channel using a **two-stage first-order IIR high-pass filter**.
- **Class**: `UltraHighRateDCBlocker` (header-only, `src/UltraHighRateDCBlocker.h`).
- **Structure**: Two cascaded 1st-order IIR sections with slightly different cutoff frequencies for minimal phase distortion.
- **Coefficients**: computed once in `prepareToPlay()` (message thread) as `alpha = 1 − exp(−ω)` using `std::expm1()` for precision at low cutoffs. On the audio thread, only the difference equation is evaluated.
- **State**: maintained per channel, per block. Filter states are stored in aligned memory.
- **SIMD safety**: no `libm` calls, no dynamic allocation, no branching on audio thread.
- **Real-time guarantee**: all state variables are pre-allocated and initialized to zero in `prepareToPlay()`.

### 2.3 Analyzer Input Tap (Lock-Free FIFO)

Two different lock-free queues are easy to confuse:

- **Analyzer FIFO** — `LockFreeAudioRingBuffer analyzerFifo` (SPSC, `FIFO_SIZE = 1048576` samples). The tap runs inside `processInputDouble()` (before headroom gain and the input DC blocker) when the analyzer source is `Input`, and the output-side tap pushes after the DSP chain, before the fixed latency delay.
- **Adaptive capture queue** — `LockFreeRingBuffer<AudioBlock, 4096>`. The audio thread splits its output into 256-sample `AudioBlock`s with `pushAdaptiveCaptureBlocks()` for the noise-shaper learner; this queue is not the analyzer's.

Both are SPSC with atomic head/tail counters, pre-allocated in `prepareToPlay()`, and consume nothing on the audio thread beyond the copy (AVX2 `_mm256_loadu_pd`/`_mm256_storeu_pd` plus a scalar tail). The UI/worker side reads with `readFromFifo()` / the learner's segment assembler.

---

## 3. Oversampling

### 3.1 Overview

Oversampling increases the internal sample rate (2×, 4×, or 8×) to reduce aliasing in subsequent DSP stages (particularly the EQ and convolution). It is implemented in `CustomInputOversampler`. The maximum oversampling factor is determined by `OversamplingPolicy` (`src/audioengine/OversamplingPolicy.h`, v14.27), which resolves the allowed factor per sample rate.

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
  - Flushes tiny accumulation results to zero (`fastAbs(acc) < kDenormThreshold`), and zeroes any non-finite result while setting the corruption flag.
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

### 4.2 EQProcessor (20-Band Parametric EQ)

**Source**: `src/eqprocessor/` (17 files, split TU implementation, ~213 KB)

| File | Size | Role |
|---|---|---|
| `EQProcessor.h` | 47.8 KB | Class definition, types, RCU handle |
| `EQProcessor.Core.cpp` | 41.2 KB | Core initialization and public API |
| `EQProcessor.Coefficients.cpp` | 21.0 KB | TPT SVF & biquad coefficient calculation |
| `EQProcessor.Parameters.cpp` | 15.2 KB | Parameter getters/setters |
| `EQProcessor.Processing.cpp` | **56.6 KB** — Largest TU | AVX2 FMA TPT SVF processing, M/S conversion, AGC |
| `EQProcessor.ProcessingCache.cpp` | 1.8 KB | EQCoeffCache management |
| `PeakEstimator.{h,cpp}` | 5.4 KB | Peak detection for EQ analysis |
| `UpperBoundEstimator.{h,cpp}` | 1.2 KB | Upper bound estimation for EQ bands |
| `EQResponseSampler.{h,cpp}` | 9.5 KB | Frequency response sampling (magnitude/phase) |
| `AnalysisMerge.h` | 2.7 KB | Merges multiple analysis results |
| `BandHelper.{h,cpp}` | 2.3 KB | Band utility functions and helpers |
| `EQAnalysisMath.h` | 4.0 KB | Mathematical formulas for EQ analysis |
| `EQAnalysisTypes.h` | 4.6 KB | Analysis type definitions |

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
- `EQCoeffCache` is an `alignas(64)` struct whose lifetime is owned by `DSPHandleRuntime` (the earlier `RefCountedDeferred` wrapper was replaced in v20.2.6).
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
- `SMOOTHING_TIME_SEC = 0.05` (50 ms) linear ramp (`convo::LinearRamp`).
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

**Source**: `src/convolver/` (12 files, split TU implementation, ~299 KB for the TU set; 372 KB including `ConvolverProcessor.h`)

| File | Size | Role |
|---|---|---|
| `ConvolverProcessor.Internal.h` | 5.3 KB | Helpers: `unwrapPhaseRadians`, `nextPow2`, `resampleIR`, `convertToMinimumPhase` |
| `ConvolverProcessor.Lifecycle.cpp` | 26.6 KB | Lifecycle, RCU integration, `ChangeBroadcaster` |
| `ConvolverProcessor.Rebuild.cpp` | 13.0 KB | Rebuild decision, debouncing (`REBUILD_DEBOUNCE_DEFAULT_MS`) |
| `ConvolverProcessor.LoaderThread.cpp` + `LoaderThreadInline.h` (3.9 KB) | 43.4 KB | Background IR loading, progress tracking |
| `ConvolverProcessor.LoadPipeline.cpp` | 41.6 KB | Pipeline processing, IR validation |
| `ConvolverProcessor.MixedPhase.cpp` | 36.8 KB | Phase modes, mixed-phase transition |
| `ConvolverProcessor.ResampleAndFallback.cpp` | 19.1 KB | r8brain resampling, hard fallback |
| `ConvolverProcessor.Runtime.cpp` | 53.3 KB | **Audio thread** — partitioned FFT convolution via MKL NUC |
| `ConvolverProcessor.StateAndUI.cpp` | 50.9 KB | Preset management, UI state |
| `ConvolverProcessor.h` (at `src/` root) | 72.7 KB | Public API, `BuildSnapshot`, `PhaseMode`, `ResamplingPhaseMode` |

`MKLNonUniformConvolver.{h,cpp}` (124.1 KB combined) is the live NUC engine and is compiled unconditionally. Only its diagnostic blocks are `#ifdef`-guarded; the class itself is not a legacy fallback.

**FFT Abstraction Layer** (Builder → Plan → ExecutionContext):
- `ConvolverBuilder.h` (3.1 KB) — Non-RT Plan factory; sole creator/destroyer of FFT Plans.
- `FFTBackend.{h,cpp}` (12.3 KB) — C++20 Concept abstraction. `ProductionFft` (Intel IPP) + injectable `TestFft`.
- `FFTExecutionContext.{h,cpp}` (5.8 KB) — Receives const Plan& from Builder; must not own Plan lifetime.

**Algorithm** — Intel MKL Non-Uniform Partitioned Convolution (NUC):
- Partitions the impulse response into non-uniform blocks (shorter blocks near the start for low-latency, longer blocks toward the tail for efficiency).
- Forward/backward FFTs use **Intel IPP** (`ippsFFTFwd_RToCCS_64f` / `ippsFFTInv_CCSToR_64f`) through `FFTBackend`; MKL DFTI is no longer used on the audio path.
- Buffers are 64-byte aligned and the hot loops use AVX2/FMA intrinsics; the FFT itself runs inside IPP.
- 64-byte aligned buffers throughout.

**Phase Modes**:
- `PhaseMode::AsIs`: use IR as-is (linear phase).
- `PhaseMode::Minimum`: convert IR to minimum phase.
- `PhaseMode::Mixed`: **minimum phase at low frequencies, linear phase at high frequencies** — the weight of the minimum-phase component is 1.0 at and below `mixedTransitionStartHz` and 0.0 at and above `mixedTransitionEndHz`.

**IR Loading (Background)**:
- `loadImpulseResponse()` is called from the **message thread**.
- A `LoaderThread` (`juce::Thread` subclass, thread names "IRLoader" / "IRRebuilder") performs the actual file read, resampling (via `r8brain-free-src`), and phase conversion **asynchronously**.
- On completion, the new IR state is atomically swapped via **RCU** — audio thread continues processing with the old IR without interruption or blocking.
- Old convolution engines are retired via the ISR retire pipeline (`DSPLifetimeManager → ISRRetireRouter → EpochDomain`).

**Audio Thread Runtime** (`ConvolverProcessor.Runtime.cpp`):
```cpp
// At block start (audio thread): scoped RCU read, then load the active engine
convo::RCUReaderGuard guard(runtimeRcuReader);
auto* conv = loadActiveEngine(std::memory_order_acquire);
// For each channel: StereoConvolver drives the MKL NUC partitions
conv->process(channelBuffer, ...);   // nucConvolvers[ch]->Add() / Get()
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

---

## 5. Output Conditioning

### 5.1 OutputFilter — Applied per Active Stage

**Source**: `src/OutputFilter.h/cpp`

`OutputFilter` is not a single "last processor" stage. `DSPCore::processDouble()` calls it twice per block: once on the convolver output whenever the convolver is active (HC/LC), and once on the EQ output whenever the EQ is active (HPF/LPF). With both processors active, both filter sets run, independent of the processing order.

All cutoffs below are keyed to the **internal processing rate** (device rate × oversampling factor), because `prepare()` receives the processing rate.

#### Stage ① — Convolver output (HC/LC)

- **High-Cut Filter (HCF)**:
  - `Sharp`: Butterworth 4th-order cascaded (Q1=0.5412, Q2=1.3066) — steep, maintains sound pressure at cutoff
  - `Natural`: Linkwitz-Riley 4th-order (Q=0.7071 for both stages) — better phase response, default
  - `Soft`: 2nd-order (Q=0.5) — gentle, no time-domain smear
  - Cutoff: **19 kHz** (processing rate ≤ 48 kHz) / **22 kHz** (above)

- **Low-Cut Filter (LCF)**:
  - `Natural`: Butterworth 2nd-order **HPF, 18 Hz** — minimum low-frequency distortion
  - `Soft`: 2nd-order HPF, Q=0.5, **15 Hz** — gentler, more subsonic removal

#### Stage ② — EQ output (HPF/LPF)

- **High-Pass Filter (HPF)**: Fixed Butterworth 2nd-order, **20 Hz** (always).
- **Low-Pass Filter (LPF)**:
  - `Sharp`: Q=1.0 (two cascaded stages)
  - `Natural`: Q=0.7071 (two cascaded stages)
  - `Soft`: Q=0.5 (two cascaded stages)
  - Cutoff: **19 kHz** (processing rate ≤ 48 kHz) / **24 kHz** (above)

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

**SIMD**: `process()` packs the left/right channels into 128-bit `__m128d` registers (SSE2 + FMA) and keeps the biquad state in plain structs; there is no explicit 64-byte alignment requirement on this path.

### 5.2 Soft Clipping

**Source**: `AudioEngine.Processing.DSPCoreDouble.cpp` — `softClipBlockAVX2()`, `musicalSoftClipScalar()`; `fastTanh()` lives in `src/dsp/math/FastTanhApprox.h` (`SoftClipPadePolicy`)

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

**fastTanh Approximation** (`SoftClipPadePolicy` in `src/dsp/math/FastTanhApprox.h`, no `libm`):
```cpp
const double x2 = x * x;
const double num = x * (NumA + x2 * (NumB + x2 * NumC));
const double den = DenA + x2 * (DenB + x2 * (DenC + x2));
return num / den;
```
- The polynomial evaluation itself is branch-free. Above `clipThreshold` (4.5) the scalar path returns ±1.0 with a branch, while the AVX2 path selects the result with a blend.
- For `x ≥ clipThreshold`: returns ±1.0; for `x ≤ −clipThreshold`: returns ∓1.0

**AVX2 Vectorized Version** (`softClipBlockAVX2`):
- Processes 4 doubles per instruction using `_mm256_*` intrinsics.
- `prevSampleInOut` is written back at the end of the block but is not read into the soft-clip math (write-only state).
- `_mm256_set_pd` / `_mm256_mul_pd` / `_mm256_fnmadd_pd` / `_mm256_blendv_pd` etc.
- With input oversampling at 1×, a local 2× halfband stage (`softClipOS`, 31 taps) wraps the clipper; that path adds 15 base-rate samples of latency, which `getCurrentLatencyBreakdown()` reports.

**Parameters** (snapshotted at block start): a single `saturationAmount` atomic drives all three shapes per block:
- `threshold = 0.95 − 0.45 · saturationAmount` (default 0.1 → 0.905)
- `knee = 0.05 + 0.35 · saturationAmount` (default → 0.085)
- `asymmetry = 0.10 · saturationAmount` (default → 0.01)

**Real-time safety**:
- No `libm` calls (fastTanh is a rational polynomial approximation).
- No dynamic allocation.
- Parameters pre-validated and clamped in the message thread.

### 5.3 Output Makeup Gain

- **Purpose**: Gain applied after the output filters and **before** soft clipping, so the clipper sees the final level (the chain in `DSPCore::processDouble()` is makeup gain → soft clip).
- **Implementation**: AVX2 SIMD scaling using the same `scaleBlockFallback()` function as input headroom gain.
- **Gain value**: atomic, converted from dB to linear in the message thread (no `std::pow` on audio thread).
- All buffers 64-byte aligned.

---

## 6. Dithering & Noise Shaping

All four noise shaper types share these properties:
- Double-precision processing throughout.
- All state and buffers are pre-allocated. The coefficient tables, the dither RNG ring, the 15-tap coefficient/error/RNG arrays and the lattice coefficient/state arrays are explicitly `alignas(64)`; the 4th-order fixed shaper's state is not.
- **No dynamic allocation, no locks** on the audio thread.
- **TPDF (Triangular Probability Density Function) dither** — two uniform random samples summed.
- **Error-feedback** topology: the quantization error is fed back and shaped by the noise shaper coefficients.

### 6.1 PsychoacousticDither (GUI: "12th-order")

**Source**: `src/PsychoacousticDither.h`

The device-settings entry is labelled **"12th-order"**, matching the implementation: **NS_ORDER = 12** (12-tap error-feedback). The separate "9th-order adaptive" entry selects the learned lattice shaper in §6.4.

**Architecture**:
- **12th-order error-feedback** noise shaper (NS_ORDER = 12).
- **TPDF dither** via **MKL VSL** (`vdRngUniform`, BRNG = `VSL_BRNG_SFMT19937`), or a **xorshift64\* fallback** (`fallbackUniform`) when MKL VSL is unavailable or the ring runs dry.
- **RNG ring buffer**: 65,536 entries per channel, SPSC, pre-filled.
  - `rngRing[MAX_CHANNELS][65536]` (8 channels) with atomic `rngReadPos[MAX_CHANNELS]` / `rngWritePos[MAX_CHANNELS]`.
  - Refilled from the audio engine's timer callback (message thread, non-RT) via `refillRandomRingNonRt()`.
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
- Maximum shaping error = `(scale/2) × Σ|c_k|`, where `scale = 1/2^(bitDepth−1)` (so `scale/2 = 2⁻¹⁶` for 16-bit). Worst case at 705.6 kHz / 16-bit: 65.18 × 2⁻¹⁶ ≈ 9.95e-4 ≈ −60 dBFS.

### 6.2 FixedNoiseShaper (GUI: "4th-order")

**Source**: `src/FixedNoiseShaper.h`

- **4th-order error-feedback** noise shaper (ORDER = 4).
- Coefficients: psychoacoustically tuned, sum to 1.0 for stability (e.g., `{0.46, 0.28, 0.17, 0.09}`).
- TPDF dither added before quantization.
- State is pre-allocated but not `alignas(64)`; quantization uses the SSE4.1 `_mm_round_sd`, and this shaper has no AVX2 path.
- Diagnostics (RMS/peak error) accumulate on the audio thread from each block and are published atomically for the UI.

### 6.3 Fixed15TapNoiseShaper (GUI: "15th-order")

**Source**: `src/Fixed15TapNoiseShaper.h`

- **16th-order error-feedback** noise shaper (class name retains "15Tap" for legacy consistency, but ORDER = 16).
- Coefficients: `kFixed15TapNoiseShaperTunedCoeffs` (psychoacoustically optimized).
- TPDF dither added before quantization.
- `coeffs`, `errors`, `rngState` and the diagnostic buffers are `alignas(64)`; quantization uses SSE4.1 rounding.
- Diagnostics (RMS/peak error) accumulate on the audio thread; the error FIFO has no reader in the current tree.

### 6.4 AdaptiveNoiseShaper — NoiseShaperLearner (GUI: "9th-order adaptive")

**Source**: `src/NoiseShaperLearner.{h,cpp}` (79.8 KB combined)

**Structure** — Lattice (ladder) filter:
- Order = 9 (`LatticeNoiseShaper::kOrder`).
- Lattice topology with reflection coefficients kept inside the unit circle by a ±0.85 clamp → stable by construction.
- Error feedback via lattice structure, not direct-form IIR.

**CMA-ES Optimization**:
- Covariance Matrix Adaptation Evolution Strategy running off the audio thread: one dedicated coordinator (`std::jthread`) fans candidate evaluation out to up to **6** worker threads (`kMaxParallelEvaluators`).
- Receives `AudioBlock` structs (256 samples, planar L/R) via `LockFreeRingBuffer<AudioBlock, 4096>` from the audio thread, assembled into `AudioSegment` (4096 samples + masking thresholds) for spectral evaluation.
  - `AudioBlock`: `double L[256]`, `double R[256]`, `numSamples`, `sampleRateHz`, `bitDepth`, `adaptiveCoeffBankIndex`, `sessionId` (defined in `AudioEngine.h`).
  - `AudioSegment`: `double left[4096]`, `double right[4096]`, masking thresholds over 2049 bins computed with an Intel IPP FFT (`MklFftEvaluator`; the class name keeps the MKL prefix, the backend is IPP). 64-byte aligned.
  - `AudioSegmentBuffer` (`src/AudioSegmentBuffer.h`) — heap-allocated aligned ring for segment assembly (5 s @ 768 kHz capacity).
  - `SpectralType`: `Broadband` / `Tonal` / `Transient` classification with per-segment gain normalization.
- Six learning modes: **Shortest, Short, Middle, Long, Ultra, Continuous** (gradated convergence quality).
- Multi-start: `NoiseShaperLearnerSettings` configures restart intervals, safety margins, and pole placement constraints.
- For each generation: evaluates candidate coefficient sets by simulating the noise shaper and computing weighted error (psychoacoustic masking or A-weighting).
- Updates mean and covariance of the coefficient distribution.

**Coefficient Handoff** (double-buffered `CoeffSet`):
- The learner writes `coeffSetA` / `coeffSetB` under `CoeffSetWriteLockGuard`, then publishes the active index with an atomic generation counter.
- The audio thread reads the active `CoeffSet` through the `ProcessingState` snapshot (`consumeAtomic` with acquire); no RCU reader is involved.
- `LearnedState` exists only for file persistence (`saveLearnedState` / `loadLearnedState`), not for the audio-thread handoff.
- No blocking, no locks, no allocation on audio thread.

**Bank Management**:
- Coefficient banks keyed by `StateKey` (sample rate, bit depth, mode).
- Each bank stores current best coefficients, learning history, progress metrics.
- Resume / stop / save / load per bank.
- UI polls `Status`, `Progress`, `State` atomically via `NoiseShaperLearnerProgress`.

**Stability Guarantee**:
- Reflection coefficients are clamped to `|k| ≤ 0.85` (`clampCoeff`) and the lattice state is clamped to ±2.0 (`kLatticeStateLimit`).
- Within those bounds the lattice is BIBO stable; stability is enforced by the clamps, not guaranteed for arbitrary coefficients.

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
- Block power is pushed into a `LockFreeRingBuffer<BlockPower, 4096>`. No consumer of `getRingBuffer()` exists in the current tree, so Momentary / Short-term / Integrated aggregation is not implemented yet.
- Filter states are per-channel and pre-allocated as plain structs (no explicit 64-byte alignment; only the ring-buffer storage is `alignas(64)`).

### 8.2 TruePeakDetector — ITU-R BS.1770-4/5

**Source**: `src/TruePeakDetector.h`

- **4× oversampling** (2 stages of 2×) using linear-phase FIR interpolation.
- **63 taps in the first stage and 31 in the second** (`max(15, taps / 2)`), Kaiser-windowed with Bessel I0. The first stage alone already exceeds the ITU-R BS.1770-3 example (48 taps).
- **AVX2 dot product** (`dotProductAvx2()`) for convolution.
- `peakHold` is a plain member updated by `processBlock`; there is no atomic publication and both call sites discard the return value.
- Measurement-only — no gain applied to the audio signal.

### 8.3 Spectrum Analyzer — UI Component

**Source**: `src/SpectrumAnalyzerComponent.{h,cpp}` (59.0 KB total)

- FFT-based spectrum display.
- EQ response curve overlay.
- Peak hold and smoothing.
- Consumes data from the analyzer output tap (lock-free FIFO).

---

## 9. Real-Time Safety Architecture

### 9.1 Absolute Prohibitions on Audio Thread

ConvoPeq enforces these rules on every audio callback without exception:

| Prohibition | Rationale |
|---|---|
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

- **EpochDomain** (`src/core/EpochDomain.h`, 35.6 KB): manages 64 named reader slots and a global epoch counter.
- **RCUReader** (`src/core/RCUReader.h`): RAII reader — enters epoch on construction, exits on destruction.
- **Publication flow**: message thread creates new object → `publishCurrentState()` → audio thread `loadCurrentState(acquire)` → old object retired via `enqueueRetire()`.
- **Retire pipeline**: `DSPLifetimeManager → ISRRetireRouter → EpochDomain → DeletionQueue`.
- Old objects are not deleted immediately — they are deferred until all in-flight readers exit the epoch.

### 9.4 Lock-Free Inter-Thread Communication

| Communication | Mechanism |
|---|---|
| UI parameter updates | `std::atomic` publish/consume |
| Analyzer data (audio → UI) | `LockFreeAudioRingBuffer` (SPSC, 1M samples) |
| Loudness block power (audio → main thread) | `LockFreeRingBuffer<BlockPower, 4096>` (producer only; no consumer yet) |
| Adaptive noise shaper (audio → worker) | `LockFreeRingBuffer<AudioBlock, 4096>` (SPSC) |
| Intent queue (Builder/Timer/CoordinatorLoop → ISR) | `MpscBoundedRing` (MPSC, Vyukov-style) |
| Coefficient handoff (worker → audio) | RCU + atomic generation counter |
| IR handoff (loader → audio) | RCU + atomic `irState` |
| RuntimeState ownership transfer | `OwnerChannel` (SPSC sole-ownership) |

### 9.5 Pre-Allocation Strategy

**All** buffers, states, and working memory are allocated in `prepareToPlay()`:

- `DSPCore`: `alignedL`, `alignedR` (input), `dryBypassBufferDoubleL/R` (bypass crossfade dry path).
- `EQProcessor`: `scratchBuffer`, `parallelInputBuffer`, `parallelWorkBuffer`, `parallelAccumBuffer`, `structureOldOutBuffer`, `structureNewOutBuffer` (Serial↔Parallel crossfade), `msWorkBuffer` (M/S processing), and `filterState[4][20][2]` (SVF integrators).
- `CustomInputOversampler`: `workA`, `workB`, per-stage histories and coefficient arrays.
- `PsychoacousticDither`: `rngRing[8][65536]` and the per-channel error shift registers.
- `agcAttackCoeffTable`, `agcReleaseCoeffTable`, `agcSmoothCoeffTable`.
- All filter coefficients, lookup tables, and coefficient caches.

### 9.6 Asynchronous Garbage Collection

| Mechanism | Path |
|---|---|
| `DeferredDeletionQueue` | Thread-safe queue of delete requests |
| `RefCountedDeferred` | Legacy ref-counted wrapper; no longer used for EQ lifetime (moved to `DSPHandleRuntime`) |
| `DeferredFreeThread` | Dedicated background thread for actual deallocation |
| `enqueueDeferredDeleteWithFallback` | EO-style deferred delete with EpochDomain fallback |
| `retireEQStateDeferred` / `retireBandNodeDeferred` | Per-object EQ state retirement |
| `DSPLifetimeManager::retire()` | Orchestrates retire via `ISRRetireRouter` |
| `ISRRetireRouter::enqueueRetire()` | Queues retire to `EpochDomain` |

### 9.7 ISR Runtime Governance (128 files in `src/audioengine/`)

The ISR (Intelligent State Reconstruction) runtime governance layer manages DSP lifecycle, publication, crossfade, and health monitoring:

| Component | File | Role |
|---|---|---|
| `ISRLifecycle` | `ISRLifecycle.{h,cpp}` | Lifecycle state machine |
| `ISRRTExecution` | `ISRRTExecution.{h,cpp}` | Real-time execution contract & firewall |
| `ISRRuntimePublicationCoordinator` | `ISRRuntimePublicationCoordinator.{h,cpp}` + `_ProcessIntent.cpp` | Publication choreography |
| `ISRRetireRouter` | `ISRRetireRouter.{h,cpp}` | Unified retire API |
| `ISRRetireRuntimeEx` | `ISRRetireRuntimeEx.{h,cpp}` | Extended retire runtime |
| `ISRShutdown` | `ISRShutdown.{h,cpp}` | Shutdown FSM |
| `ISRDSPHandle` | `ISRDSPHandle.{h,cpp}` | Handle-based DSP registry |
| `DSPHandleTable` | `DSPHandleTable.h` | O(1) open-addressing `DSPCore*` → `DSPHandle` map (capacity 512) |
| `ISRDSPQuarantine` | `ISRDSPQuarantine.{h,cpp}` | DSP quarantine for failing instances |
| `ISRClosure` / `ISRClosureGraphWalker` | `ISRClosure.{h,cpp}` / `ISRClosureGraphWalker.{h,cpp}` | Reflective closure graph & validation |
| `ISRPayloadTier` | `ISRPayloadTier.{h,cpp}` | Payload priority tiering |
| `ISRHB` | `ISRHB.{h,cpp}` | Heartbeat and hazard barrier |
| `ISRRetire` | `ISRRetire.{h,cpp}` | RuntimeState retirement |
| `ISRRetireLane` | `ISRRetireLane.h` | Retire lane classification |
| `ISRRetireOverflowRing` | `ISRRetireOverflowRing.h` | Overflow retirement ring |
| `RetireQuarantineStore` | `RetireQuarantineStore.h` | Fallback store when deferred-delete enqueue is full (no UAF) |
| `ISRWorldRetirementTelemetry` | `ISRWorldRetirementTelemetry.h` | World retirement observation-window telemetry |
| `ISRRuntimeSemanticSchema` | `ISRRuntimeSemanticSchema.h` | Schema v9: authority/permissions per field |
| `ISRRuntimeIdentityGenerators` | `ISRRuntimeIdentityGenerators.h` | Runtime/transition UUID generators |
| `ISRSealedObject` | `ISRSealedObject.h` | RAII seal wrapper |
| `ISRDebugRuntime` | `ISRDebugRuntime.{h,cpp}` | Debug runtime diagnostics |
| `ISREvidenceExporter` | `ISREvidenceExporter.{h,cpp}` | Evidence export for CI |
| `ISRIntentDispatcher` | `ISRIntentDispatcher.h` | Intent dispatch context / handler boundary |
| `ISRCoordinatorLoop` | `ISRCoordinatorLoop.{h,cpp}` | Coordinator loop (deferred resubmit) |
| `ISRLifetimeProof` | `ISRLifetimeProof.h` | Shutdown lifetime proof / permit types |
| `RuntimeHealthMonitor` | `RuntimeHealthMonitor.{h,cpp}` | Continuous telemetry |
| `RuntimePolicyEngine` | `RuntimePolicyEngine.{h,cpp}` | Recovery action (Observe→Throttle→Recover→Restore→Safe→Critical) |
| `RuntimePublicationOrchestrator` | `RuntimePublicationOrchestrator.{h,cpp}` | Publish orchestration |
| `RuntimePublicationValidator` | `RuntimePublicationValidator.{h,cpp}` | Validation pipeline |
| `RuntimePublicationState` | `RuntimePublicationState.h` | Publication state owner + ledger |
| `PublicationAdmission` | `PublicationAdmission.{h,cpp}` | Admission evaluation |
| `PublicationExecutor` | `PublicationExecutor.{h,cpp}` | Publication commit/dispatch |
| `RuntimePublishExecutor` | `RuntimePublishExecutor.h` | Sole ISR gateway to `RuntimeWorldAuthority::commit()` |
| `RuntimeWorldAuthority` | `RuntimeWorldAuthority.h` | World authority + PendingPublishRegistry |
| `OwnerChannel` | `OwnerChannel.h` | Lock-free SPSC sole-ownership transfer across RT boundary |
| `RuntimeBuilder` | `RuntimeBuilder.{h,cpp}` | Only entity that constructs RuntimeState |
| `RuntimeBuildTypes` | `RuntimeBuildTypes.h` | Build snapshot & fingerprint types |
| `RuntimeGraph` | `RuntimeGraph.h` | Runtime graph representation |
| `RuntimeTransition` | `RuntimeTransition.h` | State transition description |
| `BuildErrorPolicy` | `BuildErrorPolicy.h` | BuildError / FailureClassification / RetryDisposition contract |
| `RetryScheduler` | `RetryScheduler.{h,cpp}` | Collapsed retry dispatch for Transient/Infrastructure failures |
| `SequenceArithmetic` | `SequenceArithmetic.h` | Wraparound-safe modular sequence arithmetic |
| `DSPLifetimeManager` | `DSPLifetimeManager.{h,cpp}` | DSP activation / retire / crossfade |
| `DSPTransition` | `DSPTransition.h` | DSP transition handling |
| `CrossfadeAuthority` | `CrossfadeAuthority.{h,cpp}` | Crossfade governance (Authority pattern) |
| `CrossfadeRuntime` | `CrossfadeRuntime.h` | Crossfade runtime state |
| `FrozenRuntimeWorld` | `FrozenRuntimeWorld.{h,cpp}` | Frozen world for crash analysis |
| `WorldLifecycleAudit` | `WorldLifecycleAudit.{h,cpp}` | World lifecycle audit trail |
| `TelemetryRecorder` | `TelemetryRecorder.{h,cpp}` | Telemetry recording |
| `AutoGainPlanner` | `AutoGainPlanner.{h,cpp}` | Auto-gain staging |
| `OversamplingPolicy` | `OversamplingPolicy.h` | Oversampling factor resolution per sample rate |
| `AtomicAccess` | `AtomicAccess.h` | Atomic primitives (`publishAtomic`, `consumeAtomic`, etc.) |

---

## 10. Complete Signal Flow Diagram

```
[JUCE Audio Device Callback]
        │
        ▼
[AudioEngineProcessor::processBlock(double) — standalone build]
        │
        ▼
[AudioEngine::processBlockDouble]
        ├── AudioCallbackRuntimeScope (lifecycle + firewall token)
        ├── ensureThreadFloatingPointEnvironment() (FTZ/DAZ once per thread)
        └── ThreadRole::AudioRealtime
        ▼
[DSPCore::processDouble — Input Conditioning]
        │
        ├── 1. Copy to 64-byte aligned buffers (AVX2 4-wide + scalar tail)
        ├── 2. Analyzer input tap (when the analyzer source is Input)
        ├── 3. Headroom gain (AVX2 scaleBlockFallback, −12…0 dB)
        └── 4. Input DC blocking (UltraHighRateDCBlocker, 3 Hz)
        ▼
[Oversampling — CustomInputOversampler::processUp]
        (if enabled: 2×/4×/8× FIR, AVX2/FMA, Kaiser windowed sinc)
        (max factor resolved by OversamplingPolicy per sample rate)
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
        │             FFT: ConvolverBuilder → FFTBackend(IPP) → FFTExecutionContext
        │             RCU: irState.load(acquire)
        │
        ├── [Convolver → EQ]
        │       ├── ConvolverProcessor::process()
        │       └── EQProcessor::process()
        ▼
[OutputFilter — applied per active stage, not by routing position]
        │
        ├── Stage ① (convolver output, whenever the convolver is active):
        │     HCF Sharp/Natural 4th-order, Soft 2nd-order
        │     LCF Natural 2nd-order HPF 18 Hz / Soft 2nd-order Q=0.5 15 Hz
        │     cutoff 19 kHz (processing rate ≤ 48 kHz) / 22 kHz
        │
        └── Stage ② (EQ output, whenever the EQ is active):
              HPF 2nd-order 20 Hz, fixed
              LPF Sharp/Natural/Soft 2nd-order × 2 stages
              cutoff 19 kHz (processing rate ≤ 48 kHz) / 24 kHz
        │  All coefficients precomputed in prepare() (std::sin/cos),
        │  process() = table lookup only, no libm, in-place biquad SOS
        ▼
[Output Makeup Gain]
        │  (AVX2 SIMD, atomic gain pre-converted to linear)
        ▼
[Soft Clipping — softClipBlockAVX2]
        ├── piecewise: linear → knee (t²(3−2t)) → clipped (fastTanh)
        ├── local 2× halfband around the clipper when input OS = 1
        │     (31 taps; 15 base-rate samples of reported latency)
        ├── fastTanh: rational polynomial approximation, branchless
        └── Parameters: saturationAmount → threshold/knee/asymmetry
        ▼
[Downsampling — CustomInputOversampler::processDown]
        (if oversampling: multi-stage FIR decimation, AVX2/FMA, fail-safe)
        ▼
[Analyzer Output Tap]
        │  (LockFreeAudioRingBuffer analyzerFifo, taken before the fixed delay)
        ▼
[Output conditioning — processOutputDouble]
        ├── Output DC blocker (3 Hz)
        ├── Dither / Noise Shaping at −1 dBFS headroom
        │     (Psychoacoustic 12th / Fixed 4th / Fixed 16th / Adaptive 9th;
        │      plain −1 dBFS headroom when dither is off)
        ├── TruePeak + LUFS metering
        ├── peakLimiter + hard clamp to ±0.8913
        └── Fixed latency delay (latency compensation)
        ▼
[DSPCore::processToBuffer — write to output buffer]
        │
        ▼
[JUCE Audio Device Output]
```

---

## 11. Key DSP Numeric Constants

| Constant | Value | Location |
|---|---|---|
| `NUM_BANDS` | 20 | `EQProcessor.h` |
| `kFilterChannels` | 4 (L/R/Mid/Side) | `EQProcessor.h` |
| `NS_ORDER` | 12 | `PsychoacousticDither.h` |
| `SR_BANDS` | 6 | `PsychoacousticDither.h` |
| `LatticeNoiseShaper::kOrder` | 9 | `LatticeNoiseShaper.h` |
| `FixedNoiseShaper::ORDER` | 4 | `FixedNoiseShaper.h` |
| `FixedNoiseShaper::MAX_CHANNELS` | 8 | `FixedNoiseShaper.h` |
| `Fixed15TapNoiseShaper::ORDER` | 16 | `Fixed15TapNoiseShaper.h` |
| `AGC_ATTACK_TIME_SEC` | 0.2 | `EQProcessor.h` |
| `AGC_RELEASE_TIME_SEC` | 2.0 | `EQProcessor.h` |
| `AGC_SMOOTH_TIME_SEC` | 0.2 | `EQProcessor.h` |
| `BYPASS_FADE_TIME_SEC` | 0.005 (5 ms) | `EQProcessor.h` |
| `SMOOTHING_TIME_SEC` | 0.05 (50 ms) | `EQProcessor.h` |
| `LoudnessMeter::kMaxChannels` | 2 | `LoudnessMeter.h` |
| `TruePeakDetector::kOversamplingRatio` | 4 | `TruePeakDetector.h` |
| `TruePeakDetector::kDefaultTaps` | 63 | `TruePeakDetector.h` |
| `MklFftEvaluator::kFftLength` | 4096 | `MklFftEvaluator.h` |
| `MklFftEvaluator::kSpectrumBins` | 2049 | `MklFftEvaluator.h` |
| `AudioSegment::kLength` | 4096 (= `MklFftEvaluator::kFftLength`) | `NoiseShaperLearner.h` |
| `RNG_RING_SIZE` | 65,536 (1u << 16) | `PsychoacousticDither.h` |
| `kDenormThresholdAudioState` | `1.0e-20` | `DspNumericPolicy.h` |

---

## 12. Reference

### Core DSP Source Files

| File | Description |
|---|---|
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
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | DSP core (float path) |
| `src/audioengine/OversamplingPolicy.h` | Oversampling factor resolution per sample rate |
| `src/eqprocessor/EQProcessor.h` | EQ class definition, 20-band, TPT SVF, RCU |
| `src/eqprocessor/EQProcessor.Core.cpp` | Core EQ logic, M/S, AGC |
| `src/eqprocessor/EQProcessor.Coefficients.cpp` | TPT SVF + RBJ biquad coefficient computation |
| `src/eqprocessor/EQProcessor.Parameters.cpp` | Parameter getters/setters |
| `src/eqprocessor/EQProcessor.Processing.cpp` | TPT SVF processing, AVX2 FMA (largest EQ TU) |
| `src/eqprocessor/EQProcessor.ProcessingCache.cpp` | EQCoeffCache management |
| `src/eqprocessor/PeakEstimator.{h,cpp}` | Peak detection for EQ analysis |
| `src/eqprocessor/UpperBoundEstimator.{h,cpp}` | Upper bound estimation |
| `src/eqprocessor/EQResponseSampler.{h,cpp}` | Frequency response sampling |
| `src/eqprocessor/AnalysisMerge.h` | Analysis result merging |
| `src/eqprocessor/BandHelper.{h,cpp}` | Band utility functions |
| `src/eqprocessor/EQAnalysisMath.h` | EQ analysis math formulas |
| `src/eqprocessor/EQAnalysisTypes.h` | EQ analysis type definitions |
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
| `src/OutputFilter.{h,cpp}` | Output HC/LC (convolver stage) and HPF/LPF (EQ stage) |
| `src/PsychoacousticDither.{h,cpp}` | 12th-order psychoacoustic dither (26.5 KB) |
| `src/FixedNoiseShaper.h` | 4th-order fixed noise shaper (13.4 KB) |
| `src/Fixed15TapNoiseShaper.h` | 16th-order (ORDER=16, name legacy "15Tap") fixed noise shaper (19.0 KB) |
| `src/LatticeNoiseShaper.h` | Lattice noise shaper structure (9th-order, 10.6 KB) |
| `src/NoiseShaperLearner.{h,cpp}` | CMA-ES adaptive learning (79.8 KB combined) |
| `src/NoiseShaperLearnerTypes.h` | Learning mode, status, progress types (2.3 KB) |
| `src/LoudnessMeter.{h,cpp}` | ITU-R BS.1770-4/5 K-weighting (13.0 KB) |
| `src/TruePeakDetector.{h,cpp}` | 4× OS true peak, 63-tap FIR (16.2 KB) |
| `src/SpectrumAnalyzerComponent.{h,cpp}` | FFT spectrum UI (59.0 KB) |
| `src/NoiseShaperLearningComponent.{h,cpp}` | Noise shaper learning UI (24.1 KB) |
| `src/ConvolverControlPanel.{h,cpp}` | Convolver control panel (69.7 KB) |
| `src/ConvolverSettingsComponent.{h,cpp}` | Advanced convolver settings (5.2 KB) |
| `src/MixedPhaseOptimizationComponent.{h,cpp}` | Mixed-phase progress UI (5.3 KB) |
| `src/MKLNonUniformConvolver.{h,cpp}` | Active MKL NUC engine (124.1 KB) |
| `src/ConvolverBuilder.h` | ISR Plan Builder (Non-RT Authority, 3.1 KB) |
| `src/FFTBackend.{h,cpp}` | FFT abstraction layer (IPP ProductionFft / TestFft, 12.3 KB) |
| `src/FFTExecutionContext.{h,cpp}` | FFT execution context (const Plan& consumer, 5.8 KB) |
| `src/MpscBoundedRing.h` | Bounded MPSC ring for intent queue (11.5 KB) |
| `src/AudioSegmentBuffer.h` | Aligned heap ring for AudioSegment assembly (7.2 KB) |
| `src/AlignedAllocation.h` | 64-byte aligned `malloc`/`free` (8.5 KB) |
| `src/LockFreeRingBuffer.h` | SPSC lock-free ring buffer (5.1 KB) |
| `src/LockFreeAudioRingBuffer.h` | SPSC audio FIFO (7.3 KB) |
| `src/DspNumericPolicy.h` | Numeric constants, denorm thresholds (16.0 KB) |
| `src/audioengine/DSPLifetimeManager.{h,cpp}` | DSP lifecycle orchestration |
| `src/audioengine/CrossfadeAuthority.{h,cpp}` | Crossfade governance (Authority) |
| `src/audioengine/CrossfadeRuntime.h` | Crossfade runtime state |
| `src/audioengine/ISRRetireRouter.{h,cpp}` | Unified retire API |
| `src/audioengine/ISRRetireRuntimeEx.{h,cpp}` | Extended retire runtime |
| `src/audioengine/ISRShutdown.{h,cpp}` | Shutdown FSM |
| `src/audioengine/ISRLifecycle.{h,cpp}` | Lifecycle state machine |
| `src/audioengine/ISRRTExecution.{h,cpp}` | RT execution contract |
| `src/audioengine/ISRDSPHandle.{h,cpp}` | Handle-based DSP registry |
| `src/audioengine/DSPHandleTable.h` | O(1) DSPCore* → DSPHandle map (7.0 KB) |
| `src/audioengine/ISRDSPQuarantine.{h,cpp}` | DSP quarantine |
| `src/audioengine/ISRClosure.{h,cpp}` | Reflective closure graph |
| `src/audioengine/ISRClosureGraphWalker.{h,cpp}` | Closure graph traversal |
| `src/audioengine/ISRPayloadTier.{h,cpp}` | Payload tiering |
| `src/audioengine/ISRHB.{h,cpp}` | Heartbeat / hazard barrier |
| `src/audioengine/ISRRetire.{h,cpp}` | RuntimeState retirement |
| `src/audioengine/RetireQuarantineStore.h` | Fallback store for failed retire enqueue (12.7 KB) |
| `src/audioengine/ISRWorldRetirementTelemetry.h` | World retirement telemetry (17.6 KB) |
| `src/audioengine/ISRRuntimeSemanticSchema.h` | Schema v9 authority contract |
| `src/audioengine/ISRRuntimeIdentityGenerators.h` | Runtime/transition UUID generators |
| `src/audioengine/ISRSealedObject.h` | RAII seal wrapper |
| `src/audioengine/ISRDebugRuntime.{h,cpp}` | Debug diagnostics |
| `src/audioengine/ISREvidenceExporter.{h,cpp}` | Evidence export |
| `src/audioengine/ISRIntentDispatcher.h` | Intent dispatch context (5.1 KB) |
| `src/audioengine/ISRCoordinatorLoop.{h,cpp}` | Coordinator loop (2.7 KB) |
| `src/audioengine/ISRLifetimeProof.h` | Shutdown lifetime proof types (10.4 KB) |
| `src/audioengine/RuntimeHealthMonitor.{h,cpp}` | Telemetry / health monitoring |
| `src/audioengine/RuntimePolicyEngine.{h,cpp}` | Recovery action hierarchy |
| `src/audioengine/RuntimePublicationOrchestrator.{h,cpp}` | Publish orchestration |
| `src/audioengine/RuntimePublicationValidator.{h,cpp}` | Publication validation |
| `src/audioengine/RuntimePublicationState.h` | Publication state ledger |
| `src/audioengine/RuntimeBuilder.{h,cpp}` | RuntimeState construction |
| `src/audioengine/RuntimeGraph.h` | Runtime graph representation |
| `src/audioengine/RuntimePublishExecutor.h` | ISR publish execution gateway (7.4 KB) |
| `src/audioengine/RuntimeWorldAuthority.h` | World authority + PendingPublishRegistry (21.4 KB) |
| `src/audioengine/OwnerChannel.h` | SPSC sole-ownership transfer (7.4 KB) |
| `src/audioengine/BuildErrorPolicy.h` | BuildError / FailureClassification contract (8.9 KB) |
| `src/audioengine/RetryScheduler.{h,cpp}` | Rebuild retry scheduler (4.7 KB) |
| `src/audioengine/SequenceArithmetic.h` | Modular sequence arithmetic (5.6 KB) |
| `src/audioengine/PublicationAdmission.{h,cpp}` | Admission evaluation |
| `src/audioengine/PublicationExecutor.{h,cpp}` | Publication commit |
| `src/audioengine/FrozenRuntimeWorld.{h,cpp}` | Frozen world for crash analysis |
| `src/audioengine/WorldLifecycleAudit.{h,cpp}` | World lifecycle audit trail |
| `src/audioengine/TelemetryRecorder.{h,cpp}` | Telemetry recording |
| `src/audioengine/AutoGainPlanner.{h,cpp}` | Auto-gain staging |
| `src/audioengine/OversamplingPolicy.h` | Max oversampling factor per sample rate |
| `src/audioengine/DSPTransition.h` | DSP transition handling |
| `src/audioengine/AtomicAccess.h` | Atomic primitives (`publishAtomic`, etc.) |
| `src/core/EpochDomain.h` | RCU epoch domain (64 reader slots, 35.6 KB) |
| `src/core/RCUReader.h` | RAII RCU reader |
| `src/core/SnapshotCoordinator.{h,cpp}` | Snapshot coordination |
| `src/core/SnapshotFactory.{h,cpp}` | Snapshot creation |
| `src/core/SnapshotAssembler.{h,cpp}` | Snapshot assembly |
| `src/core/SnapshotSlotStore.h` | Slot-based atomic pointer storage |
| `src/core/SnapshotRetireManager.h` | Snapshot retirement management |
| `src/core/SnapshotParams.h` | Snapshot parameter container |
| `src/core/SnapshotFadeState.h` | Snapshot crossfade state |
| `src/core/GlobalSnapshot.{h,cpp}` | Immutable snapshot base |
| `src/core/ObservedRuntime.h` | Observed runtime abstraction |
| `src/core/ObserveChannel.h` | Observation channel classification |
| `src/core/RuntimeStore.h` | Runtime publication store |
| `src/core/RuntimeReaderContext.h` | Reader context |
| `src/core/RuntimePublicationCoordinator.h` | Publication coordinator template |
| `src/core/IEpochProvider.h` | Abstract epoch provider |
| `src/core/IPublicationProvider.h` | Abstract publication provider |
| `src/core/IReaderEpochProvider.h` | Abstract reader epoch provider |
| `src/core/IRetireProvider.h` | Abstract retire provider |
| `src/core/IRetireRouter.h` | Abstract retire router interface |
| `src/core/RetireBoundaryTelemetry.h` | Retire boundary telemetry |
| `src/core/ScopedMXCSR.h` | RAII MXCSR state saver |
| `src/core/ThreadAffinityManager.h` | CPU core affinity management |
| `src/core/ThreadHash.h` | Thread hash utilities |
| `src/core/CommandBuffer.h` | Non-blocking command dispatch |
| `src/core/WorkerThread.{h,cpp}` | Background snapshot worker |
| `src/core/DeletionQueue.{h,cpp}` | Deferred deletion queue |
| `src/core/DeferredRetireFallbackQueue.h` | Retire fallback queue |
| `src/core/FadeEngine.h` | Fade shape generation |
| `src/core/Types.h` | DSP core enum types |
| `src/core/TimeUtils.h` | Time measurement harness |
| `src/core/RebuildTypes.h` | Rebuild intent classification |
| `src/core/EQParameters.h` | EQ parameter container |
| `src/core/ConvolverRuntimeCompatTypes.h` | Convolver runtime compat types |

### Architecture Documents

- `ARCHITECTURE.md` — Overall system architecture, threading design, ISR governance
- `BUILD_GUIDE_WINDOWS.md` — Build instructions, toolchain setup
- `.github/copilot-instructions.md` — Coding standards, prohibitions
- `doc/sourcecode_analysis_2026-07-03.md` — Detailed source code companion analysis

### External References

- **Vadim Zavalishin** — "The Art of VA Filter Design" (TPT SVF theory)
- **Robert Bristow-Johnson** — "Audio EQ Cookbook" (RBJ biquad formulas)
- **Intel oneMKL** — Non-Uniform Partitioned Convolution (NUC) engine
- **ITU-R BS.1770-4/5** — Loudness measurement (K-weighting)
- **ITU-R BS.1770-3** — True peak detection
- **Hansen, J. (2012)** — True peak detection literature
