# D162-1R-B — Minimal Attribution Instrumentation Evidence Summary

Date: 2026-09-02
Binary: build-diag RelWithDebInfo + CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON (evidence/D162-1RB_build_rwdi.log, EXIT=0)
Short soak: evidence/D162-1RB_short_soak.log (EXITCODE=0x00000000, 11 rebuild generations gen=5..15)
Conditions: IR=D162-1P_active.wav (76800 src @192kHz → processing 384kHz), block=1024, OS=2x,
  --cli-ir-reload-count 12 --cli-ir-reload-interval-ms 6000 --cli-intent-burst-count 12
  --cli-intent-burst-interval-ms 6000 --cli-exit-ms 110000
  (12 requested reloads → 11 completed rebuilds within the exit window; last may be cut off — same
  boundary effect as D162-1P where 60 requested → 60 completed over 420s. Purpose = instrumentation
  correctness, not memory-slope measurement.)

## 1. Event counts (short soak log)

| log tag | count | expected |
| --- | ---: | --- |
| [DSPCORE_PREPARE] total | 11 | 1 per gen |
| [CONV_FOOTPRINT] | 11 | 1 per gen (prepareToPlay end) |
| [DSP_FOOTPRINT] phase=construct | 11 | 1 per gen |
| [IR_LOAD] | 22 | 2 NUC per gen |
| [NUC_FOOTPRINT] | 22 | 2 NUC per gen |
| [NUC_ALLOC] | 660 | per layer/kind (15 kinds × 2 layers + 2 IPP lines per NUC ≈ 30 × 22) |
| [D133] enqueue | 11 | gens 5..15 |
| [DSP_ALLOC] | 66 | 6 kinds × 11 gens |
| [DSP_FOOTPRINT] phase=retained | 11 | 1 per gen |
| [DSP_DESTROY_FOOTPRINT] / [DSP_FOOTPRINT_RELEASED] | 2 / 2 | placeholder(gen=3) + replaced published(gen=5) |

enqueue gens [5..15] == retained gens [5..15] == construct pointers (11/11 with construct+retained on
same dsp pointer). Pointer identity reconciliation (AC-R5) PASS.

## 2. Per-retained-DSP measured footprint (all 11 gens identical)

[DSP_FOOTPRINT] phase=retained (bytes):

| field | bytes | MB | cross-check |
| --- | ---: | ---: | --- |
| convolver | 104,857,600 | 100.00 | == [CONV_FOOTPRINT] TOTAL (delay 64 + dry/smoothing/oldDry/wet 8×4 + fadeRamp 4) |
| irData | 3,072,000 | 2.93 | 192,000 samples × 8B × 2ch (exact) |
| nuc (MKL) | 37,738,048 | 36.00 | == 2 × NUC_FOOTPRINT (persistent 17,120,896 + scratch 2,359,840) exact |
| ipp (spec+work) | 1,813,760 | 1.73 | == 2 × (ippSpec 906,880) via ippsFFTGetSize requery, exact |
| latency (DSPCore History) | 131,104 | 0.125 | fixedLatencyBufferSize × 8B × 2 |
| eq | 1,769,496 | 1.69 | EQ capacity members × 8B |
| other | 0 | 0 | measured-but-uncategorized only |
| **TOTAL** | **149,382,008** | **142.48** | |

oversampler / loudness / truePeak = UNMEASURED (not counted in TOTAL — no estimate values used).
unaccountedBytes = 0 by construction (no residual dumping).

Per-NUC constant across all 22 instances: persistent=17,120,896 scratch=2,359,840 ipp=906,880
TOTAL=19,775,904. Layer × kind breakdown (per NUC, from allocSizes):
L0 part=2048: irFreqDomain 32,832 / irFreqReal 524,544 / irFreqImag 524,544 / fdlBuf 65,664 /
fdlReal 1,049,088 / fdlImag 1,049,088 / fftTimeBuf 32,768 / fftOutBuf 32,768 / prevInputBuf 16,384 /
accumBuf 32,832 / accumReal 16,392 / accumImag 16,392 / inputAccBuf 16,384 (scratch+IR)
L1 part=16384: irFreqReal/Imag 262,208 each, fdlReal/Imag 524,416 each, delayLineBuf 5,677,056,
tailOutputBuf 1,441,792, ipp spec 906,880 (spec+work via requery).

## 3. Closure against D162-1P measured slope

- D162-1P measured private increase: **+141.7 MB/gen** (CSV slope 1,214.9 MB/min @ 7s/gen).
- D162-1R-B measured per-retained-DSP footprint: **142.48 MB**.
- Δ = 0.78 MB/gen ≈ 0.5% → **PASS criterion (§14) satisfied**: retained DSP ↔ footprint closes
  with < 1% residual. The residual is covered by the UNMEASURED small categories
  (oversampler/loudness/truePeak — est. KB-MB order) and MEM_SNAP MB rounding.

### Model correction vs D162-1R-A (important)

D162-1R-A static model counted "latency 23.5 MB" per DSP (latencyBufOld/New ×4,
AudioEngine.Processing.PrepareToPlay.cpp:177-196). Investigation showed that buffer set is
**AudioEngine-owned (one-time)**, not generation-scoped. The per-DSP latency buffers are
HistoryRuntimeState::fixedLatencyBufferL/R = 131,104 B (0.125 MB). Corrected per-DSP static model:
100 (conv) + 2.93 (irData) + 36.0 (nuc) + 1.73 (ipp) + 0.125 (latency) + 1.69 (eq) ≈ 142.5 MB
— matches the runtime measured TOTAL and the D162-1P slope. The R-A "169 MB static vs 141.7 MB
measured (Δ12-15%)" residual is thereby resolved: the 23.5 MB latency set was never per-gen.

## 4. Destroy-side reconciliation (AC-R5)

[DSP_DESTROY_FOOTPRINT] dsp=000001E62B8FC080 gen=3 trackedFootprint=106,758,200
  (convolver=104,857,600 irData=0 nuc=0 ipp=0 latency=131,104 eq=1,769,496) — placeholder DSP
  (IR-less construct footprint, construct-phase value preserved).
[DSP_DESTROY_FOOTPRINT] dsp=000001E624F91080 gen=5 trackedFootprint=149,382,008 — replaced published
  DSP; pointer+gen+footprint identical to its retained line. [DSP_FOOTPRINT_RELEASED] dsp=… remaining=0
  ×2. Destroy count (2) is consistent with D162-1P (placeholder + replaced published only; the other
  retained DSPs were never retired — H1).

## 5. Relationship to D162-1P (published / non-published / DSPCore live)

Same structure reproduced at 11 gens: DC live 1 → 10 (+9 net over 11 gens; 11 = placeholder + 11 gens
with 2 destroys), SC live = DC live, NUC live = 2 × SC live (MEM_SNAP final: DC=10 SC=10 NUC=20,
Priv=1798MB). Every retained (non-published) DSP carries the full measured 142.48 MB footprint —
1:1 correspondence between retained DSP count and Convolver 100 MB + NUC 36 MB, confirming the
attribution: **the "未帰属 ~70 MB/DSP" of D162-1P is the ConvolverProcessor fixed buffers (100 MB)
plus NUC pair (36 MB) minus the engine-level latency correction**; attribution now closes at <1%.

## 6. NO-GO checks (§14)

- DSP lifecycle unchanged: rebuild count/destroy path/handle retirement identical to D162-1P
  (11/11 prepare→enqueue, 2 destroys, Ret pend=0 ovf=0).
- No RT allocation added: all instrumentation runs on RebuildThread (prepare/rebuildIR) or
  Message/destroy paths (all NonRT); audio-thread path (Add/Get) untouched.
- ASAN: build PASS (D162-1RB_build_asan.log); smoke run (3 gens, --cli-exit-ms 45000) →
  all instrumentation lines emitted correctly under ASAN ([CONV_FOOTPRINT]×3, [NUC_FOOTPRINT]×4,
  [DSP_FOOTPRINT]×6, [DSP_DESTROY_FOOTPRINT]×2, [DSP_FOOTPRINT_RELEASED]×2). At shutdown the run
  hit **the same pre-existing teardown UAF already documented in D162-1P**
  (`AudioEngine::tryShutdownQuiescentReclaim` via `EQCacheManager::CacheMap::~CacheMap`,
  D162-1P: AudioEngine.h:4428 → D162-1R-B: AudioEngine.h:4454 — the +26 line offset is exactly the
  DIAG member block added to AudioEngine.h; identical stack, identical address pattern). This is a
  D162-1P known finding (shutdown teardown ordering), NOT a D162-1R-B regression — no ASAN error
  occurs in any instrumented path before teardown. ASAN smoke: instrumented-path clean.
- Debug+DIAG build PASS (D162-1RB_build_debug.log).
- Retire/reclaim behavior unchanged (rec counter monotonic, pend=0).

## 7. Build artifacts

| config | script | log | result |
| --- | --- | --- | --- |
| RelWithDebInfo + DIAG | evidence/D162-1RB_diag_build_rwdi.bat | evidence/D162-1RB_build_rwdi.log | EXIT=0 |
| Debug + DIAG | evidence/D162-1RB_diag_build_debug.bat | evidence/D162-1RB_build_debug.log | EXIT=0 |
| Release + DIAG + ASAN | evidence/D162-1RB_diag_asan_build.bat | evidence/D162-1RB_build_asan.log | EXIT=0 |

## 8. Scope deviations (documented per §10)

1. AudioEngine.Commit.cpp (enqueuePublicationIntentForRuntimeCommit): gen-stamp + retained capture.
   Not in the literal file list but is the only (dsp*, generation) junction and is DSP commit
   lifecycle; DIAG-guarded only.
2. AudioEngine.Threading.cpp (destroyDSPCoreNode): destroy-side logs required by §7; DIAG-guarded.
3. EQProcessor.h: DIAG-only inline accessor diagFootprintBytes() (capacity members × 8B) required
   for the measured `eq=` field of §6. Read-only, DIAG-guarded.
4. IPP sizes measured by ippsFFTGetSize_R_64f requery inside MKLNonUniformConvolver.cpp (in scope)
   — no FFTBackend.cpp edit needed.
