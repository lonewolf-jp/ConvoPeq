# Detailed Audio Signal Processing Flow in ConvoPeq

This document describes, in as much detail as possible, the entire audio signal processing flow in ConvoPeq, from external audio input to external audio output. The analysis covers all major components, including threading, buffer management, and real-time safety strategies.

---

## 1. Audio Input (External to Application)

### Platform Layer & Entry Point

- **JUCE Audio Device Callback**: Audio input is delivered by the JUCE audio engine via the `AudioIODeviceCallback` or, in plugin/standalone mode, via the `juce::AudioProcessor::processBlock()` method.
- **Entry Function**: In ConvoPeq, the main entry is `AudioEngineProcessor::processBlock()` (float/double), which is called by the audio device driver thread for each audio block.
  - For float: `void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)`
  - For double: `void processBlock(juce::AudioBuffer<double>& buffer, juce::MidiBuffer&)`
- **Thread Context**: This callback runs on the real-time audio thread, with strict timing and real-time safety requirements (no blocking, no allocation, no locks).

### Buffer Structure & Memory Layout

- **Buffer Type**: The input buffer is a `juce::AudioBuffer<float>` or `juce::AudioBuffer<double>`, which is a planar (non-interleaved) buffer:
  - Channels: Typically 2 (stereo), but can be mono.
  - Each channel is a contiguous array of samples.
  - The buffer is not guaranteed to be 64-byte aligned, so internal processing copies data to aligned buffers for SIMD/MKL efficiency.
- **Buffer Lifetime**: The buffer is only valid for the duration of the callback. All processing must be completed before returning.

### AudioEngineProcessor → AudioEngine

- **Delegation**: `AudioEngineProcessor::processBlock()` wraps the buffer in a `juce::AudioSourceChannelInfo` and calls `AudioEngine::getNextAudioBlock()`.
  - For double-precision, it calls `AudioEngine::processBlockDouble()`.
- **No MIDI**: MIDI input is ignored; only audio is processed.

### AudioEngine Input Handling

- **Aligned Buffering**: Input samples are copied into internal, 64-byte aligned buffers (see `AlignedAllocation.h`) to enable AVX2 and MKL operations.
  - This avoids any dynamic allocation in the audio thread; all buffers are pre-allocated during `prepareToPlay()`.
  - Copying is performed using `std::memcpy` or SIMD routines for maximum efficiency.
- **Channel Handling**: If the input is mono, the single channel is duplicated to both left and right internal buffers.
- **Sample Format**: All internal processing is performed in double-precision (`double`), regardless of input format.
  - If the input is float, samples are converted to double before further processing.

### Real-Time Safety & Synchronization

- **No Dynamic Allocation**: Absolutely no `malloc`, `new`, or `std::vector::resize` in the audio thread.
- **No Locks**: No mutexes or blocking synchronization. All parameter/state changes are handled via `std::atomic` or lock-free patterns.
- **Atomic State Snapshot**: At the start of each block, all runtime parameters (bypass, order, analyzer, gain, etc.) are snapshotted from atomics for thread safety and consistency.

### Input Dataflow Summary

1. **Audio device driver** fills a `juce::AudioBuffer<float/double>` and calls `AudioEngineProcessor::processBlock()`.
2. **AudioEngineProcessor** delegates to `AudioEngine`, passing the buffer.
3. **AudioEngine** copies input samples to internal, 64-byte aligned double-precision buffers.
4. **All further processing** (headroom, DC blocking, oversampling, DSP, etc.) operates on these aligned, double-precision buffers.

### FFT Convolver (ConvolverProcessor)

The ConvolverProcessor is a high-performance, real-time safe convolution engine designed for audio applications requiring long impulse responses (IRs), such as reverb, speaker simulation, and correction filters. It utilizes Intel MKL's Non-Uniform Partitioned Convolution (NUC) for efficient FFT-based processing, supporting stereo operation and seamless IR switching.

**Key architectural features:**

- **Thread Safety:** IR loading and switching are performed asynchronously on the message thread, using RCU (Read-Copy-Update) to ensure glitch-free operation. The audio thread never allocates memory or reloads IRs.
- **Real-Time Safety:** All buffers are pre-allocated and 64-byte aligned for SIMD/MKL efficiency. No dynamic allocation, locks, or I/O occur in the audio thread.
- **Stereo Processing:** Internally manages separate convolution engines for left and right channels, each with its own IR data and MKL NUC instance.
- **Parameter Management:** All parameters (mix, phase mode, smoothing time, IR length, etc.) are managed atomically for lock-free, thread-safe updates.
- **Visualization:** Generates IR waveform and frequency response snapshots for UI display, without impacting audio thread performance.
- **Garbage Collection:** Old convolution engines are safely garbage collected after IR switches, ensuring no memory leaks or thread hazards.

**Processing Flow:**

1. At block start, the audio thread atomically loads the current IR state.
2. For each channel, partitioned FFT convolution is performed using the MKL NUC engine.
3. Dry/wet mixing, latency compensation, and crossfading are handled in real time, with all operations performed on pre-allocated, aligned buffers.
4. All state changes (e.g., IR switch, parameter update) are applied atomically and safely, with no interruption to audio processing.

#### Code Path Example (ConvolverProcessor)

```cpp
// At block start (audio thread):
auto* currentIR = irState.load(std::memory_order_acquire);
for (int ch = 0; ch < numChannels; ++ch) {
  // Partitioned FFT convolution (MKL NUC)
  mklNUC.process(channelBuffer, currentIR, ...);
}
```cpp

### Mixed (Parallel) IR Mode: Signal Processing Flow

When the IR mode is set to "Mixed" (parallel), the input signal is split and processed in parallel by both the EQ and Convolver modules, then recombined. This enables hybrid processing, such as blending a clean EQ path with a colored convolution IR.

#### Signal Flow

1. **Input Buffer**: The aligned, double-precision input buffer $x[n]$ is prepared as usual.
2. **Parallel Processing**:

- **EQ Path**: $y_{\mathrm{EQ}}[n] = \mathrm{EQ}(x[n])$
- **Convolver Path**: $y_{\mathrm{Conv}}[n] = \mathrm{Convolver}(x[n])$

1. **Mixing**:

- The outputs are blended using a user-configurable mix ratio $\alpha$ (0 ≤ $\alpha$ ≤ 1):
    $$
    y[n] = (1 - \alpha) \cdot y_{\mathrm{EQ}}[n] + \alpha \cdot y_{\mathrm{Conv}}[n]
    $$
- $\alpha$ is typically set via the UI or preset, and may be smoothed atomically per block for click-free transitions.

1. **Post-Processing**: The mixed output $y[n]$ proceeds to output conditioning (filters, gain, soft clipping, dither, etc.).

#### Mathematical Formulation

- Let $x[n]$ be the input sample at time $n$.
- $y_{\mathrm{EQ}}[n]$ is the output of the parametric EQ (see EQProcessor section for details).
- $y_{\mathrm{Conv}}[n]$ is the output of the FFT-based convolution (see ConvolverProcessor section).
- The final output is:
  $$
  y[n] = (1 - \alpha) \cdot \mathrm{EQ}(x[n]) + \alpha \cdot \mathrm{Convolver}(x[n])
  $$

#### Buffer Management & SIMD

- Both EQ and Convolver operate on separate, 64-byte aligned double-precision working buffers.
- The mixing operation is performed using AVX2 SIMD for maximum throughput:
  $$
  y[n:n+3] = (1 - \alpha) \cdot y_{\mathrm{EQ}}[n:n+3] + \alpha \cdot y_{\mathrm{Conv}}[n:n+3]
  $$
- All buffers are pre-allocated in `prepareToPlay()`; no dynamic allocation occurs in the audio thread.

#### Parameter Management & Real-Time Safety

- The mix ratio $\alpha$ is stored as an atomic variable and snapshotted at block start.
- All parameter/state changes are atomic or lock-free.
- No locks, no dynamic allocation, and no blocking in the audio thread.

#### Code Path Example

```cpp
// In DSPCore::process() (pseudo-code):
if (processingMode == Mixed) {
   eq.process(inputBuffer, eqBuffer);
   convolver.process(inputBuffer, convBuffer);
   for (int n = 0; n < numSamples; ++n)
      outputBuffer[n] = (1 - alpha) * eqBuffer[n] + alpha * convBuffer[n];
}
```cpp


#### Summary

- Mixed IR mode enables flexible hybrid processing by blending EQ and convolution outputs in parallel.
- All operations are double-precision, SIMD-optimized, and real-time safe.
- The design ensures maximum fidelity and glitch-free transitions between processing modes.

---

### 9th-Order Adaptive Noise Shaper Learning (CMA-ES)

ConvoPeq features a 9th-order adaptive noise shaper whose coefficients are optimized in real time using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES). This enables the system to minimize perceived quantization noise for the current audio material and output bit depth.

#### Mathematical Model

The noise shaper is modeled as an IIR filter of the form:

$$
e[n] = x[n] - y[n]
$$
$$
y[n] = x[n] + \sum_{k=1}^{12} a_k \cdot e[n-k]
$$

where:
- $x[n]$: input sample (pre-quantization)
- $y[n]$: output sample (post-shaping, pre-quantization)
- $e[n]$: quantization error at time $n$
- $a_k$: adaptive feedback coefficients (to be optimized)

The goal is to find the set $\{a_1, ..., a_{12}\}$ that minimizes the weighted error energy in the output, subject to stability and real-time constraints.

#### Optimization Objective

The cost function $J$ is typically defined as:

$$
J = \sum_{n=0}^{N-1} w[n] \cdot (e[n])^2
$$

where $w[n]$ is a perceptual weighting function (e.g., A-weighting or psychoacoustic masking curve) and $N$ is the block length.

#### CMA-ES Learning Loop

1. **Initialization**: Start with a population of candidate coefficient vectors $\mathbf{a}^{(i)} = [a_1^{(i)}, ..., a_9^{(i)}]$.
2. **Evaluation**: For each candidate, run the noise shaper on the most recent audio block and compute $J^{(i)}$.
3. **Selection**: Rank candidates by $J^{(i)}$ and select the best-performing subset.
4. **Adaptation**: Update the mean and covariance of the coefficient distribution according to the CMA-ES algorithm.
5. **Repeat**: Iterate until convergence or for a fixed number of generations per block.

#### Real-Time & Threading Considerations

- The audio thread pushes recent audio blocks to a lock-free FIFO for the learner.
- The CMA-ES optimization runs on a dedicated worker thread, never blocking the audio thread.
- Coefficient updates are handed off using atomic/RCU patterns for glitch-free, real-time-safe application.
- The learning process is controlled by the UI (start/stop, mode selection) and progress is reported atomically.

#### Practical Notes

- The adaptive shaper can converge in 10–80 minutes depending on mode (Short/Middle/Long).
- Coefficient banks are saved/loaded per sample rate and bit depth.
- The system ensures stability by constraining the feedback polynomial roots inside the unit circle.
- All learning and application is performed in double-precision, SIMD-optimized code paths.

---
   // 3. Set up lock-free FIFOs for analyzer/UI
   // 4. No allocation in audio thread after this point
}

```cpp

### Key Internal Structures

- **DSPCore::ProcessingState**: Struct holding all parameters needed for one block, snapshotted from atomics.
- **ScopedAlignedPtr<double> alignedL/R**: 64-byte aligned input/output buffers for SIMD/MKL.
- **LockFreeRingBuffer**: Used for analyzer tap and inter-thread communication.
- **RCU/Atomic Generation**: Used for IR and adaptive coefficient handoff.

### Real-Time Safety Summary

- No dynamic allocation, no locks, no blocking in audio thread.
- All parameter/state changes are atomic or lock-free.
- All memory is pre-allocated in `prepareToPlay()`.
- Old DSP/IR/coefficients are garbage collected asynchronously.

## 3. Input Conditioning

### Input Conditioning Overview

Input conditioning is the first stage of internal processing after input buffer alignment. It prepares the signal for high-fidelity DSP by applying headroom gain, removing DC offset, and optionally capturing raw input for UI analysis.

### 1. Headroom Gain (SIMD Optimized)

- **Purpose**: Prevents internal DSP from clipping by scaling input samples down by a configurable gain (typically -3dB to -6dB).
- **Implementation**: SIMD-optimized scaling using AVX2 instructions for double-precision buffers.
  - Function: `scaleBlockFallback(double* data, int numSamples, double gain)`
  - All gain calculations and application are performed without libm calls in the audio thread.
- **No Dynamic Allocation**: All buffers are pre-allocated and aligned for SIMD.

### 2. Input DC Blocking (UltraHighRateDCBlocker)

- **Purpose**: Removes DC offset from each channel using a two-stage first-order IIR DC blocker.
- **Implementation**: `UltraHighRateDCBlocker` (see UltraHighRateDCBlocker.h)
  - Two cascaded 1st-order IIR filters with slightly different cutoff frequencies for minimal phase distortion.
  - SIMD-safe, no dynamic allocation, no libm calls in audio thread.
  - State is maintained per channel, per block.
- **Initialization**: All filter coefficients are set up in `prepareToPlay()` (never in audio thread).

### 3. Analyzer Tap (Lock-Free FIFO)

- **Purpose**: Captures raw input samples (pre-gain, pre-DC block) for UI spectrum analysis.
- **Implementation**:
  - Uses a lock-free, single-producer single-consumer ring buffer (`LockFreeRingBuffer<AudioBlock, 4096>`) for real-time safety.
  - Function: `pushAdaptiveCaptureBlocks()` splits the input into 256-sample blocks, copies with `memcpy`, and pushes to the FIFO.
  - No locks, no allocation, all operations are atomic and cache-line aligned.
- **Consumption**: UI thread reads from FIFO using `readFromFifo()`, which is also lock-free for the audio thread.

### SIMD & Real-Time Safety

- All conditioning steps use AVX2 SIMD for maximum throughput.
- No blocking, no locks, no dynamic allocation in the audio thread.
- All state (gain, DC blocker, FIFO pointers) is maintained per block and per channel.

### Example (Code Path)

```cpp
// Headroom Gain (SIMD)
scaleBlockFallback(alignedL, numSamples, headroomGain);
scaleBlockFallback(alignedR, numSamples, headroomGain);

// DC Blocking
inputDCBlockerL.process(alignedL, numSamples);
inputDCBlockerR.process(alignedR, numSamples);

// Analyzer Tap (if enabled)
if (analyzerEnabled)
   pushAdaptiveCaptureBlocks(captureQueue, alignedL, alignedR, numSamples, sampleRate, bitDepth, coeffBankIndex);
```cpp

### Key Internal Structures

- **UltraHighRateDCBlocker**: Two-stage IIR DC blocker, SIMD-safe, no libm, no allocation.
- **LockFreeRingBuffer**: SPSC, 64-byte aligned, atomic, no locks.
- **AudioBlock**: 256-sample, double-precision, planar, used for analyzer FIFO.

### Notes

- All conditioning is performed before oversampling or main DSP.
- Analyzer tap is optional and only enabled if UI requests spectrum data.
- All code paths are designed for maximum real-time safety and SIMD efficiency.

## 4. Oversampling (Optional)

### Oversampling Overview

Oversampling increases the internal sample rate (2x, 4x, or 8x) to reduce aliasing and improve DSP quality. It is implemented by the `CustomInputOversampler` class, which supports both IIR-like and linear-phase FIR upsampling.

### 1. Structure & Initialization

- **Class**: `CustomInputOversampler`
  - Supports up to 8x oversampling via cascaded stages (2x per stage, up to 3 stages).
  - Two presets: `IIRLike` (lower latency, fewer taps), `LinearPhase` (more taps, higher attenuation).
- **Buffering**: All working buffers (`workA`, `workB`, histories, coefficients) are pre-allocated and 64-byte aligned for SIMD.
- **prepare()**: Called from the message thread before audio starts. Sets up all stages, allocates buffers, computes FIR coefficients (Kaiser window, sinc), and clears histories.

### 2. Upsampling (processUp)

- **Entry**: `processUp(const juce::dsp::AudioBlock<double>& inputBlock, int numChannels)`
- **Flow**:
   1. For each stage, calls `interpolateStage()` per channel:
      - Applies FIR interpolation using precomputed coefficients.
      - Uses AVX2 SIMD for dot product and buffer operations.
      - Handles denormal numbers explicitly (flush-to-zero).
   2. Doubles the sample count at each stage.
   3. Final output is a double-precision, planar, aligned buffer.
- **SIMD Optimization**: All convolution and buffer ops use AVX2/FMA where available.
- **No Dynamic Allocation**: All memory is reserved in `prepare()`.

### 3. Downsampling (processDown)

- **Entry**: `processDown(const juce::dsp::AudioBlock<double>& upsampledBlock, juce::dsp::AudioBlock<double>& outputBlock, int numChannels)`
- **Flow**:
   1. For each stage (reverse order), calls `decimateStage()` per channel:
      - Applies FIR decimation using precomputed coefficients.
      - Uses AVX2 SIMD for convolution.
      - Handles denormal numbers explicitly.
   2. Halves the sample count at each stage.
   3. Final output is written to the aligned output buffer.
- **Safety**: If input size exceeds pre-allocated capacity, output is zeroed and processing is skipped.

### 4. FIR Filter Design

- **Coefficients**: Computed per stage using Kaiser windowed sinc, with tap count and attenuation depending on preset and stage.
- **Alignment**: All coefficient arrays are 64-byte aligned for SIMD.
- **Normalization**: Coefficients are normalized to ensure unity gain at DC.

### 5. Real-Time Safety & SIMD

- **No allocation, no locks, no libm calls** in the audio thread.
- **All buffer and history management** is explicit and pre-allocated.
- **Denormal Handling**: All stages flush denormals to zero for performance.
- **All up/downsampling is performed in double-precision** for maximum fidelity.

### 6. DC Blocker Integration

- **Post-Oversampling**: After upsampling, a DC blocker (`UltraHighRateDCBlocker`) is applied to remove any DC introduced by interpolation.
- **Pre-Downsampling**: After main DSP and before downsampling, another DC blocker may be applied to suppress DC artifacts.

### Example (Code Path)

```cpp
// Oversampling up (in AudioEngine/DSPCore)
if (oversamplingEnabled)
{
   auto upBlock = customOversampler.processUp(inputBlock, numChannels);
   dcBlocker.process(upBlock.getChannelPointer(0), upBlock.getNumSamples());
   // ... main DSP processing at high rate ...
   customOversampler.processDown(upBlock, outputBlock, numChannels);
}
```cpp

### Key Internal Structures

- **CustomInputOversampler::Stage**: Holds FIR coefficients, histories, tap counts, all 64-byte aligned.
- **workA/workB**: Double-precision, aligned working buffers for ping-pong processing.
- **All buffer sizes**: Determined at prepare-time, never resized in audio thread.

### Notes

- Oversampling is optional and only enabled if selected by the user.
- All up/downsampling is performed in-place, with no extra allocation or copying.
- All code paths are designed for maximum SIMD throughput and real-time safety.

## 5. Main DSP Chain

#### Code Path Example (ConvolverProcessor)

```cpp
// At block start (audio thread):
auto* currentIR = irState.load(std::memory_order_acquire);
for (int ch = 0; ch < numChannels; ++ch) {
  // Partitioned FFT convolution (MKL NUC)
  mklNUC.process(channelBuffer, currentIR, ...);
}
```cpp

### Main DSP Chain Overview

- **Processing Order**: The main DSP chain order is selectable at runtime: `EQ -> Convolver` or `Convolver -> EQ`. This is controlled by the atomic ProcessingState snapshot at the start of each block.
- **All processing is performed in double-precision, 64-byte aligned buffers, with AVX2/FMA SIMD optimization throughout.**

---

### Parametric Equalizer (EQProcessor)

- **Class**: `EQProcessor` ([src/EQProcessor.h](src/EQProcessor.h), [src/EQProcessor.cpp](src/EQProcessor.cpp))
- **Bands**: 20-band parametric EQ, each band independently configurable (frequency, gain, Q, type, channel mode, enabled).
- **Filter Structure**: Each band uses a TPT (Topology-Preserving Transform) State Variable Filter (SVF), based on Vadim Zavalishin's "The Art of VA Filter Design" for minimal noise and smooth modulation.
  - SVF coefficients: Calculated per band, per block, using sample-rate-correct formulas.
  - Biquad coefficients (for UI/magnitude response): Calculated using Audio EQ Cookbook (RBJ) formulas.
- **Parameter Management**:
  - All band parameters are stored in atomic structures (`EQBandParams`), ensuring lock-free, thread-safe updates.
  - UI thread updates parameters via `setBandXxx()` methods, which atomically swap in new BandNode objects (RCU pattern).
  - Audio thread always loads the latest parameters atomically at block start; no locks or blocking.
- **Processing**:
  - `process(juce::dsp::AudioBlock<double>& block)` applies all enabled bands in series (cascade) per channel.
  - SIMD (AVX2) is used for block processing where possible.
  - Bypass is handled with a short crossfade (bypassFadeGain) for click-free switching, using atomic flags.
  - AGC (auto gain control) is available, with attack/release/smoothing time constants, all atomic and SIMD-safe.
- **State/Memory**:
  - All filter states, coefficients, and working buffers are pre-allocated and 64-byte aligned.
  - No dynamic allocation, no locks, no libm calls in the audio thread.
  - All parameter/state changes are atomic or lock-free.
  - Old states are garbage collected asynchronously (trash bin pattern).

#### Signal Processing Flow (EQProcessor)

1. At block start, the audio thread atomically loads the current EQ state (all band parameters, types, channel modes).
2. For each channel (typically stereo):
   - For each band (up to 20):
      - If enabled, applies TPT SVF filtering to the channel buffer using the current band's parameters.
      - SVF state (integrators) is maintained per band, per channel, in aligned memory.
      - SIMD (AVX2) is used for blockwise multiply/adds.
3. If bypass is requested, a crossfade ramp is applied to smoothly transition in/out of EQ processing.
4. AGC (if enabled) computes input/output envelope, applies gain smoothing, and updates total gain atomically.

#### Key Internal Structures (EQProcessor)

- `EQBandParams`: Atomic struct for each band's frequency, gain, Q, enabled flag.
- `EQCoeffsSVF`: Struct holding TPT SVF coefficients (g, k, a1, a2, a3, m0, m1, m2).
- `EQState`: Ref-counted object holding all band parameters, types, channel modes, and total gain.
- `BandNode`: Per-band filter state, atomically swapped via RCU.
- All filter states and coefficients are 64-byte aligned for SIMD.

#### Code Path Example (EQProcessor)

```cpp
// At block start (audio thread):
auto* state = currentStateRaw.load(std::memory_order_acquire);
for (int ch = 0; ch < numChannels; ++ch) {
   for (int band = 0; band < NUM_BANDS; ++band) {
      if (state->bands[band].enabled) {
         // Apply TPT SVF filter to channel buffer
         processSVF(channelBuffer, state->bands[band], ...);
      }
   }
}
```cpp

---

### FFT Convolver (ConvolverProcessor)

- **Class**: `ConvolverProcessor` ([src/ConvolverProcessor.h](src/ConvolverProcessor.h), [src/ConvolverProcessor.cpp](src/ConvolverProcessor.cpp))
- **Algorithm**: FFT-based partitioned convolution using Intel oneMKL Non-Uniform Convolution (NUC) engine.
  - Supports very long IRs (impulse responses) with low latency via non-uniform partitioning.
  - All FFTs and buffer operations are performed with AVX2/FMA SIMD and MKL routines.
- **Impulse Response (IR) Management**:
  - IRs are loaded asynchronously on the message/worker thread via `loadImpulseResponse()`.
  - New IRs are atomically swapped in using the RCU pattern; audio thread always processes with the latest available IR, with no blocking or allocation.
  - Supports phase modes (as-is, minimum, mixed), IR length trimming, resampling, and windowing (Tukey, etc.).
  - All IR buffers are 64-byte aligned and pre-allocated.
- **Processing**:
  - `process(juce::dsp::AudioBlock<double>& block)` performs partitioned convolution per channel, using the current IR and NUC engine.
  - Dry/wet mix, bypass, and smoothing are all atomic and SIMD-optimized.
  - No dynamic allocation, no locks, no libm calls in the audio thread.
  - All parameter/state changes are atomic or lock-free.
- **Thread Safety**:
  - All IR/parameter updates use atomic variables and RCU for glitch-free, real-time-safe handoff.
  - Old IRs and convolution engines are garbage collected asynchronously.

#### Signal Processing Flow (ConvolverProcessor)

1. At block start, the audio thread atomically loads the current IR and convolution engine state.
2. For each channel:
   - Partitioned FFT convolution is performed using the Intel MKL NUC engine.
   - Input buffer is transformed to frequency domain, multiplied by IR partitions, and inverse-transformed.
   - All FFTs and buffer operations use AVX2/FMA SIMD and are 64-byte aligned.
   - Dry/wet mix is applied per sample, with atomic smoothing.
3. If bypass is requested, a crossfade ramp is applied to smoothly transition in/out of convolution.
4. All IR/engine state changes are handled via atomic swap (RCU), with no blocking or allocation in the audio thread.

#### Key Internal Structures (ConvolverProcessor)

- `MKLNonUniformConvolver`: Core engine for partitioned FFT convolution, using Intel oneMKL DFTI.
- `IRState`: Ref-counted object holding current IR, phase mode, windowing, and partitioning info.
- All FFT buffers, IRs, and working memory are 64-byte aligned and pre-allocated.
- All parameter/state changes are atomic or lock-free.

#### Code Path Example (ConvolverProcessor)

```cpp
// At block start (audio thread):
auto* currentIR = irState.load(std::memory_order_acquire);
for (int ch = 0; ch < numChannels; ++ch) {
  // Partitioned FFT convolution (MKL NUC)
  mklNUC.process(channelBuffer, currentIR, ...);
}
```cpp

### Mixed (Parallel) IR Mode: Signal Processing Flow

When the IR mode is set to "Mixed" (parallel), the input signal is split and processed in parallel by both the EQ and Convolver modules, then recombined. This enables hybrid processing, such as blending a clean EQ path with a colored convolution IR.

#### Signal Flow

1. **Input Buffer**: The aligned, double-precision input buffer $x[n]$ is prepared as usual.
2. **Parallel Processing**:

- **EQ Path**: $y_{\mathrm{EQ}}[n] = \mathrm{EQ}(x[n])$
- **Convolver Path**: $y_{\mathrm{Conv}}[n] = \mathrm{Convolver}(x[n])$

1. **Mixing**:

- The outputs are blended using a user-configurable mix ratio $\alpha$ (0 ≤ $\alpha$ ≤ 1):
    $$
    y[n] = (1 - \alpha) \cdot y_{\mathrm{EQ}}[n] + \alpha \cdot y_{\mathrm{Conv}}[n]
    $$
- $\alpha$ is typically set via the UI or preset, and may be smoothed atomically per block for click-free transitions.

1. **Post-Processing**: The mixed output $y[n]$ proceeds to output conditioning (filters, gain, soft clipping, dither, etc.).

#### Mathematical Formulation

- Let $x[n]$ be the input sample at time $n$.
- $y_{\mathrm{EQ}}[n]$ is the output of the parametric EQ (see EQProcessor section for details).
- $y_{\mathrm{Conv}}[n]$ is the output of the FFT-based convolution (see ConvolverProcessor section).
- The final output is:
  $$
  y[n] = (1 - \alpha) \cdot \mathrm{EQ}(x[n]) + \alpha \cdot \mathrm{Convolver}(x[n])
  $$

#### Buffer Management & SIMD

- Both EQ and Convolver operate on separate, 64-byte aligned double-precision working buffers.
- The mixing operation is performed using AVX2 SIMD for maximum throughput:
  $$
  y[n:n+3] = (1 - \alpha) \cdot y_{\mathrm{EQ}}[n:n+3] + \alpha \cdot y_{\mathrm{Conv}}[n:n+3]
  $$
- All buffers are pre-allocated in `prepareToPlay()`; no dynamic allocation occurs in the audio thread.

#### Parameter Management & Real-Time Safety

- The mix ratio $\alpha$ is stored as an atomic variable and snapshotted at block start.
- All parameter/state changes are atomic or lock-free.
- No locks, no dynamic allocation, and no blocking in the audio thread.

#### Code Path Example

```cpp
// In DSPCore::process() (pseudo-code):
if (processingMode == Mixed) {
   eq.process(inputBuffer, eqBuffer);
   convolver.process(inputBuffer, convBuffer);
   for (int n = 0; n < numSamples; ++n)
      outputBuffer[n] = (1 - alpha) * eqBuffer[n] + alpha * convBuffer[n];
}
```cpp

#### Summary

- Mixed IR mode enables flexible hybrid processing by blending EQ and convolution outputs in parallel.
- All operations are double-precision, SIMD-optimized, and real-time safe.
- The design ensures maximum fidelity and glitch-free transitions between processing modes.

### Input Trim & Processing Order

- When EQ precedes Convolver, an input trim gain may be applied before convolution to prevent IR overload.
- The chain order (`EQ -> Convolver` or `Convolver -> EQ`) is selected per block via atomic ProcessingState.
- All modules are designed for maximum SIMD throughput, real-time safety, and glitch-free parameter/IR handoff.

## 6. Output Conditioning

### Output Filter (High-Cut, Low-Cut, Low-Pass)

- **Component**: `OutputFilter` ([src/OutputFilter.h](src/OutputFilter.h), [src/OutputFilter.cpp](src/OutputFilter.cpp))
- **Purpose**: Applies high-cut (HCF), low-cut (LCF), and optionally low-pass filtering to the output signal, depending on user settings and processing order.
- **Filter Structure**:
  - Implements cascaded biquad IIR filters for both HCF and LCF.
  - Each filter is configured as a second-order section (SOS), with coefficients precomputed in the message thread and stored in aligned memory.
  - All filter states are maintained per channel, per block, and are 64-byte aligned for SIMD efficiency.
- **Mathematical Formulation**:
  - Each biquad section implements the standard difference equation:
      $$
      y[n] = b_0 x[n] + b_1 x[n-1] + b_2 x[n-2] - a_1 y[n-1] - a_2 y[n-2]
      $$
      where $b_0, b_1, b_2, a_1, a_2$ are the filter coefficients.
  - HCF and LCF cutoff frequencies are user-configurable (e.g., LCF: 10–40 Hz, HCF: 20–40 kHz).
  - Filter design uses Butterworth or Bessel topology for maximally flat or phase-linear response, as selected in code.
- **SIMD Optimization**:
  - All filtering is performed using AVX2 SIMD intrinsics for double-precision blocks.
  - Filter states and coefficients are 64-byte aligned to maximize throughput.
- **Real-Time Safety**:
  - No dynamic allocation or libm calls in the audio thread.
  - All coefficients are precomputed in `prepareToPlay()` or on parameter change (never in the audio thread).
  - All state updates are atomic or lock-free.
- **Code Path Example**:

   ```cpp
   // OutputFilter processing (per channel)
   outputFilterL.processBlock(alignedL, numSamples);
   outputFilterR.processBlock(alignedR, numSamples);
   ```

- **Parameter Management**:
  - Cutoff frequencies, filter order, and enable/disable flags are managed via atomic variables and updated by the UI thread.
  - Audio thread always uses the latest snapshot, with no locks or blocking.

### Output Makeup Gain

- **Purpose**: Applies a final gain adjustment after output filtering, before soft clipping, to compensate for any level loss or to match target output level.
- **Implementation**:
  - SIMD-optimized gain scaling using AVX2 for double-precision buffers.
  - Gain value is atomic and updated by the UI thread; audio thread reads the latest value per block.
  - No dynamic allocation or libm calls in the audio thread.
  - All buffers are 64-byte aligned for SIMD.

### Soft Clipping Saturation

- **Component**: Soft clipper implemented in [src/AudioEngine.cpp](src/AudioEngine.cpp) as `musicalSoftClipScalar()` and `softClipBlockAVX2()`.
- **Purpose**: Prevents digital clipping and shapes output dynamics by applying a smooth, anti-inter-sample-peak soft clipping curve.
- **Algorithm**:
  - The soft clipper uses a piecewise function:
      $$
      y = \begin{cases}
         x & |x| < t_1 \\
          ext{cubic curve} & t_1 \leq |x| < t_2 \\
          ext{sign}(x) \cdot s & |x| \geq t_2
      \end{cases}
      $$
      where $t_1$ and $t_2$ are threshold values, and $s$ is the saturation ceiling.
  - The cubic region ensures a smooth transition into limiting, minimizing odd-order distortion and aliasing.
  - The function is branchless and SIMD-friendly for maximum efficiency.
- **SIMD Optimization**:
  - `softClipBlockAVX2()` processes entire blocks using AVX2 intrinsics, handling 4–8 samples per instruction.
  - All buffers and states are 64-byte aligned.
- **Anti-Inter-Sample-Peak Logic**:
  - The soft clipper is designed to minimize inter-sample peaks by smoothing transitions and avoiding hard limiting.
  - This reduces the risk of DAC overload and preserves musicality.
- **Real-Time Safety**:
  - No dynamic allocation, no locks, no libm calls in the audio thread.
  - All parameters (thresholds, ceiling) are atomic and updated by the UI thread.
- **Code Path Example**:

   ```cpp
   // Soft clipping (per channel, after makeup gain)
   softClipBlockAVX2(alignedL, numSamples);
   softClipBlockAVX2(alignedR, numSamples);
   ```

- **Parameter Management**:
  - Enable/disable, thresholds, and ceiling are managed via atomic variables.
  - Audio thread always uses the latest snapshot, with no locks or blocking.

### Summary

- All output conditioning steps (filtering, gain, soft clipping) are performed in double-precision, SIMD-optimized, real-time-safe code paths.
- All parameters are atomic or lock-free, with no dynamic allocation or blocking in the audio thread.
- The design ensures maximum fidelity, safety, and musicality at the final output stage.

## 7. Dithering & Noise Shaping (Optional)

### Output Conditioning Overview

At the output stage, one of several dither/noise shaper algorithms can be selected. Each is implemented as a real-time-safe, allocation-free, double-precision processor. The available types and their GUI names are:

- **Psychoacoustic** (GUI: "9th-order")
- **Fixed4Tap** (GUI: "4th-order")
- **Fixed15Tap** (GUI: "15th-order")
- **Adaptive9thOrder** (GUI: "9th-order adaptive")

All types apply TPDF dither and error-feedback noise shaping to minimize quantization noise and maximize subjective audio quality at the target bit depth.

---

### Psychoacoustic (GUI: "9th-order")

- **Component**: `PsychoacousticDither` ([src/PsychoacousticDither.h](src/PsychoacousticDither.h))
- **Algorithm**:
  - Uses a 12th-order error-feedback noise shaper (NS_ORDER=12), but is labeled as "9th-order" in the GUI for user familiarity.
  - Applies true TPDF dither using a high-quality RNG (Xoshiro256** or MKL VSL).
  - Noise shaping coefficients are psychoacoustically optimized for each sample rate and bit depth (see `kCoeffTable`).
  - The core process for each sample is:
    $$
     ext{shapedError}[n] = \sum_{k=0}^{11} c_k \cdot z_k[n] \\
    d[n] = \text{TPDF dither} \\
    y[n] = x[n] + d[n] + \text{shapedError}[n] \\
    q[n] = \text{Quantize}(y[n]) \\
    e[n] = y[n] - q[n] \\
    z_0[n+1] = e[n],\ z_{k+1}[n+1] = z_k[n] \ (k=0..10)
    $$
    where $c_k$ are the shaping coefficients, $z_k$ are the error states, and $d[n]$ is TPDF dither.
  - SIMD (SSE4.1/AVX2) is used for block processing.
- **Parameter Management**:
  - Coefficients are selected per sample rate and bit depth.
  - All state is 64-byte aligned and pre-allocated.
- **Real-Time Safety**:
  - No dynamic allocation or locks in the audio thread.
  - All random number generation is performed in a background thread and buffered.

---

### Fixed4Tap (GUI: "4th-order")

- **Component**: `FixedNoiseShaper` ([src/FixedNoiseShaper.h](src/FixedNoiseShaper.h))
- **Algorithm**:
  - Implements a 4th-order error-feedback noise shaper:
    $$
    fb[n] = \sum_{k=0}^{3} c_k \cdot e_k[n] \\
    y[n] = x[n] - fb[n] \\
    q[n] = \text{Quantize}(y[n]) \\
    e_0[n+1] = q[n] - y[n],\ e_{k+1}[n+1] = e_k[n] \ (k=0..2)
    $$
    where $c_k$ are the shaping coefficients, $e_k$ are the error states.
  - Coefficients are psychoacoustically tuned and sum to 1.0 for stability (e.g., {0.46, 0.28, 0.17, 0.09}).
  - TPDF dither is added before quantization.
- **Parameter Management**:
  - Coefficients are interpolated per sample rate.
  - All state is pre-allocated and 64-byte aligned.
- **Real-Time Safety**:
  - No dynamic allocation or locks in the audio thread.
  - Diagnostics (RMS/peak error) are computed in the background.

---

### Fixed15Tap (GUI: "15th-order")

- **Component**: `Fixed15TapNoiseShaper` ([src/Fixed15TapNoiseShaper.h](src/Fixed15TapNoiseShaper.h))
- **Algorithm**:
  - Implements a 15th-order error-feedback noise shaper:
    $$
    fb[n] = \sum_{k=0}^{14} c_k \cdot e_k[n] \\
    y[n] = x[n] - fb[n] \\
    q[n] = \text{Quantize}(y[n]) \\
    e_0[n+1] = q[n] - y[n],\ e_{k+1}[n+1] = e_k[n] \ (k=0..13)
    $$
    where $c_k$ are the shaping coefficients, $e_k$ are the error states.
  - Coefficients are psychoacoustically optimized (see `kFixed15TapNoiseShaperTunedCoeffs`).
  - TPDF dither is added before quantization.
- **Parameter Management**:
  - Coefficients are fixed and precomputed.
  - All state is pre-allocated and 64-byte aligned.
- **Real-Time Safety**:
  - No dynamic allocation or locks in the audio thread.
  - Diagnostics (RMS/peak error) are computed in the background.

---

### Adaptive9thOrder (GUI: "9th-order adaptive")

- **Component**: `AdaptiveNoiseShaper` ([src/AdaptiveNoiseShaper.h](src/AdaptiveNoiseShaper.h), [src/NoiseShaperLearner.cpp](src/NoiseShaperLearner.cpp))
- **Algorithm**:
  - Implements a 9th-order error-feedback noise shaper with coefficients that are learned and updated in real time by a background worker thread.
  - The core process is:
    $$
    fb[n] = \sum_{k=0}^{8} c_k[n] \cdot e_k[n] \\
    y[n] = x[n] - fb[n] \\
    q[n] = \text{Quantize}(y[n]) \\
    e_0[n+1] = q[n] - y[n],\ e_{k+1}[n+1] = e_k[n] \ (k=0..7)
    $$
    where $c_k[n]$ are the adaptive coefficients, $e_k$ are the error states.
  - Coefficient adaptation is performed by a dedicated worker thread (see `NoiseShaperLearner`), which analyzes output error and updates the coefficient bank using lock-free RCU.
  - All coefficient updates are atomic and thread-safe.
- **Parameter Management**:
  - Coefficient banks are managed per sample rate and bit depth.
  - UI and worker thread communicate via lock-free buffers and atomic generation counters.
- **Real-Time Safety**:
  - No dynamic allocation or locks in the audio thread.
  - All coefficient updates are performed outside the audio thread and swapped in atomically.

---

### General Notes

- All noise shaper types are implemented as double-precision, SIMD-optimized, allocation-free processors.
- All state and buffers are pre-allocated and 64-byte aligned for maximum throughput.
- All parameter/state changes are atomic or lock-free.
- Diagnostics (RMS/peak error) are available for Fixed4Tap and Fixed15Tap.
- Adaptive9thOrder uses a background worker thread for real-time coefficient learning and RCU handoff.
- All algorithms are designed for maximum real-time safety and audio fidelity.

## 8. Downsampling (If Oversampling Was Used)

### Downsampling Overview

If oversampling was enabled, the processed audio block is downsampled back to the original sample rate using a multi-stage, double-precision, SIMD-optimized FIR filter chain implemented in `CustomInputOversampler::processDown()`.

### Component

- **Class**: `CustomInputOversampler` ([src/CustomInputOversampler.h](src/CustomInputOversampler.h), [src/CustomInputOversampler.cpp](src/CustomInputOversampler.cpp))
- **Key Methods**:
  - `processDown(const juce::dsp::AudioBlock<double>& upsampledBlock, juce::dsp::AudioBlock<double>& outputBlock, int numChannels)`
  - `decimateStage(const Stage& stage, const double* input, int inputSamples, double* output, int channel)`

### Algorithm & Mathematical Formulation

- Downsampling is performed in multiple stages (1–3), each halving the sample rate (e.g., 8x → 4x → 2x → 1x).
- Each stage applies a linear-phase FIR decimation filter, designed using a Kaiser-windowed sinc kernel:
  $$
  h[n] = \text{sinc}\left(\frac{n-M}{2}\right) \cdot w[n],\quad n=0..N-1
  $$
  where $w[n]$ is the Kaiser window, $M$ is the center tap, and $N$ is the number of taps.
- The output sample at each stage is computed as:
  $$
  y[m] = h_c \cdot x[mR - M] + \sum_{k=0}^{K-1} h_k \cdot x[mR - kS]
  $$
  where $h_c$ is the center coefficient, $h_k$ are the symmetric FIR coefficients, $R$ is the decimation ratio (2), $S$ is the stride, and $x[]$ is the input buffer.
- All coefficients and buffers are 64-byte aligned for SIMD.

### Buffer & State Management

- Each stage maintains per-channel history buffers for input and output, sized to accommodate the maximum block size and filter length.
- All working buffers (`workA`, `workB`) are double-precision and 64-byte aligned.
- No dynamic allocation occurs in the audio thread; all memory is pre-allocated in `prepare()`.

### SIMD Optimization

- The core convolution (dot product) in `decimateStage` uses AVX2/FMA intrinsics for maximum throughput:
  - Multiple accumulators are used to hide FMA latency and maximize instruction-level parallelism.
  - Unaligned loads are used for input history; aligned loads for coefficients.
- Scalar fallback is provided if AVX2 is not available.

### Real-Time Safety

- No dynamic allocation, locks, or libm calls in the audio thread.
- All state and buffers are pre-allocated and zeroed in `prepare()`.
- Denormal numbers are explicitly flushed to zero for performance.
- If input size exceeds pre-allocated capacity, output is zeroed and processing is skipped (fail-safe).

### Parameter Management

- The number of stages, taps per stage, and attenuation are determined by the oversampling ratio and preset (IIR-like or LinearPhase).
- All filter coefficients are computed in `prepareStage()` using the Kaiser window and normalized for unity gain at DC.
- All parameters are atomic or lock-free.

### Code Path Example

```cpp
// Downsampling after main DSP (in AudioEngine)
customOversampler.processDown(upsampledBlock, outputBlock, numChannels);
```cpp

### Summary

- Downsampling is performed in-place, double-precision, SIMD-optimized, and real-time-safe.
- All buffer management, filter design, and state handling are explicit and allocation-free in the audio thread.
- The design ensures maximum fidelity and glitch-free operation even at high oversampling ratios.

## 9. Output Level Measurement

- **Level Metering**: Output level is measured (RMS or peak) after all processing, and the value is stored atomically for UI display.

## 10. Analyzer Output Tap (Optional)

- **Analyzer Tap**: If enabled, post-DSP output samples are pushed to a FIFO for spectrum analysis in the UI thread.

## 11. Fade-In Ramp (On DSP Change)

- **Purpose**: When switching DSP configurations, a fade-in ramp is applied to the output to prevent clicks or pops.

## 12. Final Output (External)

- **Buffer Writeback**: The processed samples are written back to the output buffer provided by the audio device.
- **Thread**: The audio device driver reads the buffer and sends it to the physical output (speakers, headphones, etc.).

---

## Threading & Real-Time Safety

- **No Dynamic Allocation**: All memory is pre-allocated outside the audio thread. No malloc/new/vector::resize/etc. in the callback.
- **No Locks**: No mutexes or blocking synchronization in the audio thread. All parameter/state changes use atomics or lock-free patterns.
- **RCU/Atomic Patterns**: IR and coefficient updates use Read-Copy-Update and atomic generation counters for glitch-free handoff.
- **Analyzer/UI Communication**: All data for UI (spectrum, meters) is pushed to lock-free FIFOs for polling by the UI thread.

---

## Summary Diagram

```text
[Audio Input]
   |
   v
[Input Conditioning]
   |
   v
[Oversampling (optional)]
   |
   v
[EQ] <-> [Convolver] (order selectable)
   |
   v
[Output Filter]
   |
   v
[Output Makeup Gain]
   |
   v
[Soft Clipping (optional)]
   |
   v
[Dither/Noise Shaping (optional)]
   |
   v
[Downsampling (if used)]
   |
   v
[Output Level Metering]
   |
   v
[Analyzer Tap (optional)]
   |
   v
[Fade-In Ramp (on DSP change)]
   |
   v
[Audio Output]
```text

---

This flow ensures high-fidelity, real-time-safe audio processing with robust UI integration and flexible DSP configuration.

## Reference Materials & Related Files

- **Core Design Files**:
  - ARCHITECTURE.md — Overall architecture, dependencies, threading design
  - .github/copilot-instructions.md — Coding standards, prohibitions, design policies
  - src/AudioEngine.cpp, .h — Top-level processing, buffer management, DSPCore
  - src/EQProcessor.cpp, .h — 20-band PEQ implementation, parameter management
  - src/ConvolverProcessor.cpp, .h — FFT convolution, IR management, RCU
  - src/CustomInputOversampler.cpp, .h — Oversampling, FIR design
  - src/UltraHighRateDCBlocker.h — DC removal IIR
  - src/LockFreeRingBuffer.h — Lock-free FIFO implementation
  - src/AlignedAllocation.h — 64-byte aligned buffer
  - MEMORY_ALLOCATION_AUDIT.md — Memory allocation & release design
  - BUILD_GUIDE_WINDOWS.md — Build instructions & environment requirements

- **External References**:
  - JUCE Official API: <https://docs.juce.com/master/index.html>
  - Vadim Zavalishin "The Art of VA Filter Design" (TPT SVF theory)
  - Audio EQ Cookbook (RBJ) (Biquad coefficient design)
  - Intel oneMKL: <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>

- **Future Documentation Expansion Guidelines**:
  - Detailed algorithm explanations for each module (e.g., noise shaper, spectrum analyzer, adaptive learning)
  - Testing, debugging, and profiling methods
  - UI integration and external control API design
  - Design patterns and caveats for adding new DSP modules
  - Implementation examples and performance tuning tips

---

## Documentation Usage Guide & Contribution

- **Usage Guide**:
  - This document serves as a reference for understanding the overall signal processing flow, design philosophy, and real-time safety of ConvoPeq.
  - When adding new features, fixing bugs, or improving performance, always consult the relevant sections and strictly adhere to the design policies, prohibitions, buffer management, and thread safety requirements described herein.
  - For code reading and debugging, use this document to trace buffer flow, atomic/lock-free design, and UI integration patterns at each processing stage.

- **Contribution & Inquiry**:
  - Corrections, missing information, and improvement suggestions for this documentation are welcome via GitHub Issues or Pull Requests.
  - When adding new modules, ensure your design and implementation address "real-time safety," "atomic/lock-free design," "garbage collection," and "UI integration," and update this document accordingly.
  - For questions about core design or implementation policies, consult ARCHITECTURE.md and .github/copilot-instructions.md, and feel free to open an Issue for discussion.

- **FAQ / Troubleshooting**:
  - **Q: Crash or noise occurs in the audio thread**
      → Check for dynamic memory allocation, libm calls, locks, exceptions, or I/O in the audio thread. See copilot-instructions.md for prohibitions.
  - **Q: UI freezes or lags**
      → Ensure synchronization between audio and UI threads is only via lock-free FIFO or atomics. Heavy UI processing should be deferred or made asynchronous.
  - **Q: What should I watch out for when adding new DSP modules?**
      → Follow existing buffer management, atomic/lock-free design, garbage collection, and UI integration patterns. Never compromise real-time safety in the audio thread.
