# D101-9 Step 5-III-E — T4 Repeated-Publish Measurement Results

> **Status**: COMPLETE — 3 load cases executed (T4-A / T4-B / T4-C), all Gates T4-G1..G5 PASS.
> **Build**: build-icx Debug, `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON`, icx compiler
> **Binary**: `build-icx/Debug/AudioEngineHarness.exe --t4=<interval_us>`
> **Runner**: `run_t4_test.bat <a|b|c>` (a=5000us, b=3333us, c=1000us)
> **Raw outputs**: `evidence/t4-{a,b,c}-output.txt`
> **Scope guard**: No HealthMonitor changes, no K_terminal constant, no Terminal capacity
> limit were made in this step (per Step 5-III-E instructions). K selection is deferred to
> Step 5-V after T2/T3/T4 integration.

---

## 1. Purpose (restated from spec)

1. Verify that T3's ~300 worlds/s is workload-specific (publish-interval dependent).
2. Measure λ_terminal load dependence under increased publish load.
3. Provide Step 5-V inputs: `λ_terminal_peak`, `terminalPeakResident_max`, `T_stall_observed_max`,
   and validate `terminalPeakResident ≈ λ × T_stall − 5120` across publish rates.

## 2. Methodology

- **Stall**: fixed 30 s (T3-30s showed Terminal arrival at ~20 s ⇒ 30 s covers arrival →
  growth → peak). 60 s/120 s not required.
- **Reader slot**: MANDATORY slot 4 (`reserveReaderThread(4)`), hardened vs T3: on reserve
  failure the run is aborted as INVALID — **no fallback** to `registerReaderThread()`.
  All 3 runs logged `reader entered (index=4)`. `minEpoch` stagnated at 18 for every stall
  tick in every run (296/294/297 ticks, single-value set `{18}`).
- **Phases** (unified with T3): baseline 3 s @500 ms → stall 30 s repeated publish → exit →
  recovery ~10 s @100 ms.
- **Run-level accounting**: new machine-parseable line printed before recovery:
  `T4_SUMMARY: intervalUs=… stallSec=30 baselinePublishes=… stallPublishes=… actualStallUs=… lambdaPublish=…`
- **Primary metrics** per spec §11: `pendingRetire` / `terminalStoreCount` /
  `terminalResident` / `terminalPeakResident`. The OBS `Q_resident` aggregate was NOT used
  to infer capacities.

## 3. Results

### 3.1 Run accounting (ground-truth rates)

| Case | interval | stallPublishes | actualStall | **λ_publish** | total publishes |
| ---- | -------: | -------------: | ----------: | ------------: | --------------: |
| T4-A | 5000 us | 4100 | 30.005 s | **136.64 /s** | 4106 |
| T4-B | 3333 us | 4767 | 30.002 s | **158.89 /s** | 4773 |
| T4-C | 1000 us | 9212 | 30.000 s | **307.06 /s** | 9218 |

Note: target intervals ≠ actual rates (spec §2). The publish path itself costs ~2–3 ms per
world (RuntimeBuilder + commit + receipt wait), so even the 1 ms-interval case saturates at
~307 /s — reproducing T3's ~300 /s as the pipeline throughput ceiling under this workload.

### 3.2 Peak metrics

| Case | max D (pendingRetire) | max Q_resident* | max E | **T_peak (K_obs)** | arrival t | λ_terminal_peak | G_terminal_peak |
| ---- | --------------------: | --------------: | ----: | -----------------: | --------: | --------------: | ---------------: |
| T4-A | 4091 | 1 | 0 | **0** | n/a | 0 | 0 |
| T4-B | 4096 FULL | 663 | 151 | **0** | n/a | 0 | 0 |
| T4-C | 4096 FULL | 1025 | 512 FULL | **4092** | 16.4 s | 330.1 /s | 330.1 /s |
| (T3-30s ref) | 4096 FULL | 1025 | 512 FULL | **3989** | 16.6 s | 326.7 /s | 326.7 /s |

\* `Q_resident` is an OBS aggregate (store + auxiliary counter) — recorded but not used for
capacity inference (spec §11).

### 3.3 Model validation — the Step 5-V key table

Model: `K_predicted = λ × T_stall − 5120` with **λ = measured λ_publish** (ground truth):

| Case | λ_publish | T_stall | K_predicted | K_observed | Error |
| ---- | --------: | ------: | ----------: | ---------: | ----: |
| T4-A | 136.64 /s | 30 s | −1021 (→ 0 expected) | 0 | ✓ no Terminal |
| T4-B | 158.89 /s | 30 s | −353 (→ 0 expected) | 0 | ✓ no Terminal |
| T4-C | 307.06 /s | 30 s | **4092** | **4092** | **−0.0%** |
| T3-30s | 303.13 /s (recomputed post-hoc) | 30 s | **3974** | **3989** | **−0.4%** |

Two independent runs at ~300 /s agree with the linear model to **<0.5%**. The earlier 17%
error reported against sampled `λ_terminal_peak` was an OBS-discretization artifact
(100 ms sampling captures bursty spill intervals; peak-of-deltas overestimates the mean).
The correct estimator is the run-level `λ_publish`.

Terminal-engagement threshold for a 30 s stall:

$$\lambda^{*} = \frac{C_D + C_Q + C_E}{T_{stall}} = \frac{5120}{30} \approx 170.7 \text{ /s}$$

Observed ladder: T4-A (136.6) < T4-B (158.9) < **λ\* (170.7)** < T4-C (307.1) — and the
engagement states match exactly: A = D-only (D peaked at 4091, just under capacity; the
residual few entries were still in-flight in the CoordinatorLoop pipeline at exit),
B = D+Q+E without Terminal, C = full chain with Terminal.

### 3.4 State-ladder model (empirically closed by T3+T4)

For a stalled reader of duration $T$ and world-retire rate $\lambda$:

$$
K_{obs}(\lambda, T) =
\begin{cases}
0 & \lambda T \le C_D = 4096 \quad \text{(D only)} \\
0 & C_D < \lambda T \le 5120 \quad \text{(D + Q + E absorb)} \\
\lambda T - 5120 & \lambda T > 5120 \quad \text{(Terminal grows linearly)}
\end{cases}
$$

Validated data points: (136.6, 0), (158.9, 0), (303.1, 3989), (307.1, 4092).

## 4. Raw evidence excerpts

T4-C first Terminal admission:

```text
[14:11:42.554] Gen=5146 ... T_store=24 T_peak=24 T_resident=24 Q_resident=1025 E_resident=512 activeReaders=3 minEpoch=18 pendingRetire=4096 pressureLevel=3
```

T4-C final state (full closure):

```text
[14:12:06.028] Gen=9307 ... T_store=4092 T_peak=4092 T_resident=0 Q_resident=0 E_resident=0 activeReaders=2 minEpoch=18614 pendingRetire=0 pressureLevel=0
```

Run summaries:

```text
t4-a: T4_SUMMARY: intervalUs=5000 stallSec=30 baselinePublishes=6 stallPublishes=4100 actualStallUs=30005266 lambdaPublish=136.64
t4-b: T4_SUMMARY: intervalUs=3333 stallSec=30 baselinePublishes=6 stallPublishes=4767 actualStallUs=30002044 lambdaPublish=158.89
t4-c: T4_SUMMARY: intervalUs=1000 stallSec=30 baselinePublishes=6 stallPublishes=9212 actualStallUs=30000434 lambdaPublish=307.06
```

## 5. Gate results

| Gate | Criterion | Result |
| ---- | --------- | ------ |
| **T4-G1** Rate scaling | λ_publish ↑ → λ_terminal ↑ | **PASS** — λ_pub 136.6→158.9→307.1; engagement 0→0→4092 monotonic |
| **T4-G2** Growth scaling | λ_terminal ↑ → net growth ↑ | **PASS** — G_terminal 0→0→330.1 /s |
| **T4-G3** Model consistency | K_obs ≈ λ×30−5120 | **PASS** — T4-C err −0.0%; T3-30s err −0.4% (λ_publish estimator); non-engaged cases correctly predicted 0 |
| **T4-G4** Recovery | exit → minEpoch advance → drain | **PASS** — minEpoch 8412/9746/18616; D/Q/E/T all → 0 |
| **T4-G5** No ownership loss | terminalResident → 0, no residual | **PASS** — final T_resident=0 in all runs |

## 6. Findings & implications for Step 5-V

1. **λ ≈ 300 /s is a pipeline property, not a workload constant**: it is the idle-publish
   path throughput ceiling (publish cost ~2–3 ms/world dominates any sleep ≥1 ms).
   Load-dependence of Terminal growth is therefore driven by whatever feeds the publish
   path; the causal chain λ↑ → K↑ holds across the measured 136–307 /s range.
2. **The linear Terminal model is confirmed at <0.5% accuracy** with the ground-truth
   λ_publish estimator. The 5120 offset (= C_D+C_Q+C_E) behaved as a true step boundary,
   not an approximation: cases below it produced zero Terminal admissions.
3. **λ_terminal_steady ≈ 0 confirmed again** (no Terminal activity outside stalls); the
   original S=2 ratio λ_terminal_peak/λ_terminal_steady is undefined (0/0) — Step 5-V must
   select K from absolute bounds + the linear growth law, not from that ratio.
4. **Step 5-V input values (current maxima across T2/T3/T4)**:
   - `λ_terminal_peak` = 330.1 /s (sampled; use λ_publish 307.1 /s for modeling)
   - `terminalPeakResident_max` = **4092** (T4-C)
   - `T_stall_observed_max` = **30 s** (bounded by test design, not by system failure)
5. **Unboundedness caveat stands**: K grows without bound as λ·T grows; the only structural
   caps are D/Q/E (5120 combined). Any K_terminal constant must therefore be a policy
   choice (admission budget / alarm threshold), not a physical bound — decision deferred to
   Step 5-V per instructions.

## 7. Files changed (measurement infrastructure only)

- `src/tests/AudioEngineHarness/T4Measurement.cpp` — NEW (slot-4-mandatory harness)
- `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` — `--t4[=<interval_us>]` flag
- `CMakeLists.txt` — T4Measurement.cpp added to AudioEngineHarness sources
- `run_t4_test.bat` — NEW runner (a/b/c cases)

No production source files were modified.
