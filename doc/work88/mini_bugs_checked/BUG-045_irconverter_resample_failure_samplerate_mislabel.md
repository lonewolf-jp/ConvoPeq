# BUG-045: IRConverter::convertFile — resample failure mislabels sample rate, corrupting downstream frequency analysis

**Severity:** Medium  
**Category:** Logic error (data correctness)  
**File:** `src/IRConverter.cpp`  
**Lines:** 258–281  

## Summary

When `IRDSP::resampleIR` fails (`converted.getNumSamples() <= 0`), the fallback path leaves the IR data at its **source** sample rate but labels it with `config.targetSampleRate` (line 271). The data/sample-rate mismatch propagates to downstream frequency-domain computations — UI spectrum display positions and IR frequency-peak-gain estimation will compute bins against the wrong Nyquist frequency.

## Details

In `convertFile` (`IRConverter.cpp:258-281`):

```cpp
juce::AudioBuffer<double> converted = ir;                   // [A] data at sourceRate
double actualSampleRate = sourceRate;
if (config.targetSampleRate > 0.0 && sourceRate > 0.0
    && std::abs(sourceRate - config.targetSampleRate) > 1.0e-6)
{
    converted = IRDSP::resampleIR(ir, sourceRate, ...);     // [B] try resample
    if (converted.getNumSamples() <= 0)
    {                                                       // [C] resample FAILED
        converted = ir;                                     //     data still at sourceRate
        actualSampleRate = config.targetSampleRate;         // [D] BUG: label as targetRate
    }
    else
    {
        actualSampleRate = config.targetSampleRate;         // [E] resample OK — correct
    }
}
```

The comment at line 264–266 claims this is intentional:  
> "Fall back to original IR. Report the IR at the target sample rate so the convolver engine uses the correct processing rate."

This reasoning is incorrect. The convolver engine expects the IR data to be at `sampleRate`; recording `targetSampleRate` when the data is actually at `sourceRate` causes all frequency-dependent analysis and rendering to misrepresent the IR's spectral content.

## Impact

`prepared->sampleRate` is consumed in `ConvolverProcessor.LoadPipeline.cpp`:

1. **Line 506** — `createFrequencyResponseSnapshot(*(prepared->timeDomainIR), prepared->sampleRate)`  
   → The UI spectrum plot interprets FFT bin frequencies using the wrong sample rate. A 1 kHz peak in a 48 kHz IR labeled as 96 kHz appears at 500 Hz on the UI.

2. **Line 512** — `updateIRState(*(prepared->timeDomainIR), prepared->sampleRate, ...)`  
   → Downstream state processing uses a mismatched sample rate for gain calculations.

3. **Line 460** — `publishAtomic(currentSampleRate, prepared->sampleRate, ...)`  
   → The published sample rate is wrong, potentially affecting other subsystems.

4. **IRConverter.cpp:362-369** — `IRAnalyzer::estimateMaxFrequencyResponseGain(scaledIR)`  
   → `IRAnalyzer` computes FFT on `scaledIR` (which is at `sourceRate`) but the logged `sampleRate` (line 369) reports `targetSampleRate`, making diagnostics misleading.

## Reproduction

1. Load an IR file at sample rate **R1** (e.g., 48000 Hz) into a session running at **R2** (e.g., 192000 Hz).
2. If `r8brain` resampling fails for any reason (corrupted file, extreme rate ratio, OOM), the fallback path activates.
3. The IR data is used as-is at 48000 Hz but labeled at 192000 Hz.
4. Frequency response display shows spectrum bins at 1/4× their correct frequency positions.

## Suggested Fix

When `resampleIR` fails, either:
- Set `actualSampleRate = sourceRate` (revert to actual data rate) and let the engine apply its own sample-rate conversion, or
- Zero-pad/interpolate the IR to genuinely match `targetSampleRate` duration, or
- Return `nullptr` (fail closed) instead of silently emitting mislabeled data.
