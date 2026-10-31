# Historical Closure Evidence — H-01 / H-02 / H-03 / C-01 resolved

```text
record type      closure evidence.  NOT a bug report.  NOT an authorization.
authority        ConvoPeq.md  5,551,216 B
                 6F1ED4815A4398FA3A5D34C6D72A914B63D9FDFC38E0BB2A6F2B47E743C90FB3
                 Generated 2026-09-27 20:33:59   HEAD e5e2f668
subjects modified  0
```

Owner instruction §2 defines these as **resolved evidence, not repair targets**.
Each entry records the current-authority locations that negate the historical
defect premise. No implementation follows from this record.

---

## H-02 — irPeakLatency energy centroid → true peak argmax: RESOLVED

Historical premise (`doc/audit/ConvoPeq_IR_Audio_Path_Audit_2026-09-13.md` L113):
`irPeakLatency` was the 99.9% energy centroid, overstating PDC.

Current-authority evidence:

```text
src\convolver\ConvolverProcessor.LoadPipeline.cpp
  L48423  int ConvolverProcessor::measureIrPeakLatencySamples(
              const juce::AudioBuffer<double>& ir, int targetLength) noexcept
  L48426  if (targetLength <= 0) return 0;
  L48431  if (channels <= 0 || samples <= 0) return 0;
  L48434  const int n = juce::jmin(samples, targetLength);
  L48438  for ch / L48441 for i < n:
  L48444      if (!std::isfinite(value)) continue;
  L48446      magnitude = abs(value);  if (magnitude > bestAbs) { bestAbs; bestIndex = i; }
  L48455  return bestIndex;   // true-peak argmax, non-finite guarded
```

Consumers: LoaderThread L47244, LoadPipeline L48617.
Tests assert argmax expectations (round-trip T1 sample0-peak ⇒ 0;
`irPeakLatency == 100` cases). The centroid mechanism is absent.

---

## H-01 — dry delay vs wet OLS alignment: RESOLVED

Historical premise (same audit, L89): dry delayed by `partSize` while wet OLS
has zero array delay → comb in mix middle.

Current-authority evidence:

```text
src\convolver\ConvolverProcessor.Runtime.cpp
  L50844  // ★ H-01 (H01-C1): bypass 経路の dry 遅延も internal alignment 基準
  L50970  // ★ H-01 (H01-C1): internal dry alignment 遅延値の単一変換関口
src\ConvolverProcessor.h
  L55639  // ★ H-01 (H01-C2): internal dry alignment 遅延基準のコンパイル時フラグ
tests (AudioEngineHarness, executed PASS in Dynamic Reproduction)
  L90838  // ★ H-01 (T-H01-1..6)
  L91045  // T-H01-3: bypass は dry/wet と同位置
  L92474  // ★ H-01 T-H01-1..6
  Dynamic run: checkH01DryWetAlignment: PASS (T-H01-1..6)
```

---

## H-03 — two IR paths, loadIR swaps no RT engine: RESOLVED

Historical premise: `loadIR` does not replace the RT engine.

Current-authority evidence:

```text
src\convolver\ConvolverProcessor.LoadPipeline.cpp
  L48305  void ConvolverProcessor::loadIR(const juce::File& irFile)
  L48336      applyComputedIR(std::move(directTarget));   // target-FFT cache hit
  L48349      applyComputedIR(std::move(cachedLow));      // low-res cache hit
```

Both cache-hit paths route through `applyComputedIR`, the engine-swap path.
The "no swap" premise does not hold.

---

## C-01 — non-power-of-2 Get clock divergence: RESOLVED by redesign

Historical premise (same audit, L62-85): L0 vs L1/L2 stream-clock divergence on
non-power-of-2 host blocks (shared-cursor `got` vs `numSamples`).

Current-authority evidence (I4 Get Clock + Repair Design Rev 4):

```text
src\MKLNonUniformConvolver.cpp
  L79445  t0 = m_outputSamplesProcessed            // absolute index at entry
  L79447  got = ringRead(output, numSamples)       // consumes min(avail, n)
  L79519  delayLineReadAdd(l, output, numSamples, t0, gain)  // ABSOLUTE stream-time read
  L79523  m_outputSamplesProcessed += got          // advance AFTER L1/L2 reads
  L79525  return got                               // caller knows valid count
  L79554  Phase-0 guard: t0 < outputDelaySamples → return
  L79567  I2 availability guard: unread computed content → counted, no add
src\convolver\ConvolverProcessor.Runtime.cpp
  L51917  got = nucConvolvers[channel]->Get(out, numSamples);
  L51919  bug3-8 guard: got < 0 → zero-fill path
  L51925  if (got < numSamples) memset(out + got, 0, ...)  // shortfall zero-filled
```

L1/L2 use absolute stream-time reads, not shared-cursor advancement; the
shortfall contract (return `got` + zero-fill) is honored by the production
caller. The historical divergence mechanism is absent.

---

## Non-targets (unchanged)

```text
D3/D4                    documentation/design only
preflight residual risk  DEFERRED
frozen lineage           unchanged (counter-evidence: none)
p1_5 §13.4               MEASUREMENT / OBSERVATION
N2-OPEN-1                CLOSED (formal acceptance)
D105-R5-9                ACCEPTED
```

---

## State

```text
TRUE OPEN BUG              = 0
CONTRACT VIOLATION         = 0
EVIDENCE GAP               = 0
Current Production Bug Queue = EMPTY (0 items)
Implementation             = NOT STARTED
Next cycle start condition = Owner-selected defect OR newly discovered
                             concrete defect
files created  1 (this record) / files modified 0
```
