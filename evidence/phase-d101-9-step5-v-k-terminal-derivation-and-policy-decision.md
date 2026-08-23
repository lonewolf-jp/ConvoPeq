# D101-9 Step 5-V — K_terminal Derivation and Policy Decision

> **Status**: COMPLETE — policy semantics decided. **No code changes in this step.**
> **Inputs**: T2 (10/50/100ms), T3 (1/5/10/30s), T4-A/B/C repeated-publish measurements
> **Downstream**: Step 5-VI (HealthMonitor threshold implementation) is BLOCKED until this
> document's policy decision is reviewed. Step 5-VI must NOT start from "implement 4092".
> **Related evidence**: `phase-d101-9-step5-iii-c-t3-long-stall-measurement-results.md`,
> `phase-d101-9-step5-iii-e-t4-repeated-publish-results.md`

---

## 1. Measurement Inputs

### 1.1 Adopted values (observation facts)

| Input | Adopted value | Provenance / treatment |
| --- | ---: | --- |
| `λ_terminal_steady` | ≈ 0 /s | Observed fact — no Terminal activity outside reader stalls (T1/T2/T3 Phase-1/4 all zero) |
| `λ_publish_max` | 307.06 /s | T4-C run-level publish rate |
| `λ_terminal_peak` | 330.1 /s | 100 ms OBS-derived max ΔT_store/Δt (T4-C) — sampling artifact-prone, see §4 |
| `λ_model` | 307.06 /s | = T4-C run-level publish rate; adopted for model computation (see §4) |
| `T_stall_observed_max` | 30.0 s | Measurement ceiling of this campaign, NOT an observed system limit |
| `terminalPeakResident_max` | 4092 | T4-C observed peak |
| `C_D` | 4096 | DeferredDeletionQueue `kQueueSize` (implementation capacity) |
| `C_Q` | 512 | RetireQuarantineStore `kMaxQuarantinedEntries` (implementation capacity) |
| `C_E` | 512 | EmergencyQuarantineStore (same type as Q) |
| `C_total` | 5120 | C_D + C_Q + C_E |

### 1.2 λ_terminal_peak and λ_model are distinct estimators

They must not be conflated:

- `λ_terminal_peak = 330.1/s` — maximum of per-tick differences of the cumulative
  `T_store` counter over 100 ms OBS samples. Discretization + spill burstiness inflate it.
- `λ_model = λ_publish = 307.06/s` — ground truth from run-level counters
  (`stallPublishes / actualStallUs`), independent of telemetry sampling.

**Decision**: model computations use `λ_model`. Rationale in §4.

## 2. T2/T3/T4 Integrated Results

| Run | λ_publish (/s) | T_stall (s) | λT | max D | Terminal engaged | K_obs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| T2-10/50/100ms | ~20–40 | 0.01–0.1 | < 4 | ≤ 1 | No | 0 |
| T3-1s | ~137* | 1 | ~137 | 279 | No | 0 |
| T3-5s | ~300* | 5 | ~1500 | 1494 | No | 0 |
| T3-10s | ~303* | 10 | ~3030 | 3031 | No (PL=1 seen) | 0 |
| T3-30s | 303.13 | 30 | 9094 | 4096 FULL | Yes @16.6 s | 3989 |
| T4-A | 136.64 | 30 | 4099 | 4091 | No | 0 |
| T4-B | 158.89 | 30 | 4767 | 4096 FULL | No (Q+E absorbed) | 0 |
| T4-C | 307.06 | 30 | 9212 | 4096 FULL | Yes @16.4 s | 4092 |

\* T3-1s/5s/10s stall rates derived post-hoc from total publishes minus baseline/recovery
phases; T3-30s value cross-checked against its own totals.

Integrated reading:

1. Engagement boundary sits between λT = 4767 (no engagement) and λT = 9094 (engagement);
   the structural prediction 5120 is consistent with both brackets.
2. Below C_D=4096: D-only. Between 4096 and 5120: D+Q+E absorb, still zero Terminal
   (T4-B is the direct proof). Above 5120: Terminal grows linearly.
3. Recovery closure held in every run: minEpoch advance + D/Q/E/T → 0 + `T_resident = 0`.

## 3. Validated Growth Model

$$
\boxed{K_{growth}(\lambda, T) = \max(0,\ \lambda T - 5120)}
$$

Validation with ground-truth λ_publish:

| Run | Predicted | Observed | Error |
| --- | ---: | ---: | ---: |
| T4-C | 307.06 × 30 − 5120 = 4092 | 4092 | −0.0% |
| T3-30s | 303.13 × 30 − 5120 = 3974 | 3989 | −0.4% |

Engagement boundary:

$$
\boxed{\lambda T > 5120 \Rightarrow \text{Terminal growth starts}}
$$

These are **physical-model statements about the current implementation topology**
(D→Q→E→Terminal spill chain with fixed capacities). They describe what WILL happen; they
do not prescribe what SHOULD be allowed.

## 4. λ_terminal estimator decision

**Decision: adopt `λ_model = λ_publish` (run-level counter ratio). Do not use sampled
`λ_terminal_peak` for model computation.**

Rationale:

- Sampled peak overestimates by ≈ +7.5% here (330.1 vs 307.06) because ΔT_store across a
  100 ms tick includes whole spill bursts, and the first post-arrival tick absorbs the
  backlog spill transient. Using it in `λT − 5120` produced the misleading 17% error noted
  in the T4 analysis before the estimator was corrected.
- Run-level `stallPublishes / actualStallUs` is measurement-instrument-independent
  (harness counters, not engine telemetry) and reproduced the model to <0.5% on two
  independent runs.
- `λ_terminal_peak` remains useful as a *cheap online growth-rate signal* for
  HealthMonitor (§13), but carries a known positive bias and must not feed K math.

## 5. T_stall_design decision

**Decision: NOT fixed in this step.**

- The 30 s used in T3/T4 is the *measurement ceiling of this campaign*. No product
  requirement states "a 30 s reader stall is acceptable design operation".
- Candidate B (`K_B = λ_design × T_stall_design − 5120`) therefore remains BLOCKED on an
  explicit product/design input: the maximum stuck-reader duration the system is required
  to tolerate without escalation. That input does not exist yet in any document surveyed
  in this repository.
- Interim consequence: any K that presumes 30 s (e.g., "8184 = 2 × 4092") inherits an
  unjustified design assumption and is rejected for now (§12).

## 6. S-factor decision

**Decision: S = 2 is formally REJECTED as non-derivable — recorded as a measurement
result, not a failure to tune.**

Original plan hypothesis: `S = λ_terminal_peak / λ_terminal_steady`, adopt S = 2.

Measured reality:

1. `λ_terminal_steady ≈ 0` in every idle/recovery phase ⇒ the ratio is 0/0 undefined.
2. T4 demonstrated engagement is **threshold-like, not proportional**: λ_publish
   136.6 → 158.9 → 307.1 mapped to Terminal engagement 0 → 0 → 4092. A multiplicative
   safety factor over a steady-state rate has no purchase on a step-boundary phenomenon.
3. Therefore: *"S=2 を採用しない"* is upgraded to *"比率ベースの safety-factor derivation
   はこのシステムに適用不能"* — the derivation method itself is invalid here.

Replacement logic (this document): physical growth law (§3) + policy semantics (§11).

## 7. K_observed

```text
terminalPeakResident_max = 4092   (T4-C)
```

Status: **observed fact only**. It is the largest Terminal residency ever produced by this
campaign under λ≈307/s for 30 s. It is NOT `K_terminal`; treating it as a capacity would
be a category error (§9).

## 8. K_rate candidates

Derived from the validated growth law for illustration — each row is a *what-if*, not an
adopted value, because both λ and T inputs lack design authority today:

| Scenario | λ (/s) | T (s) | K_rate = λT − 5120 |
| --- | ---: | ---: | ---: |
| T4-B-like, longer stall | 158.89 | 60 | 4413 |
| T4-C replay | 307.06 | 30 | 4092 |
| T4-C, 60 s stall | 307.06 | 60 | 13304 |
| Hypothetical 600/s rebuild storm | 600 | 30 | 12880 |

Reading: K_rate is unbounded in (λ, T). Any finite adopted K is necessarily a policy line,
not a physics line.

## 9. Policy-vs-physical-boundary distinction

This is the central clarification of Step 5-V.

### Physical boundary (exists, measured)

- D, Q, E have hard capacities: 4096 / 512 / 512. When all are full, `enqueueWithRetry`
  spills to Terminal (ISRRetireRouter.cpp `enqueueWithRetry` Stage 3→4→5).
- Terminal admission is epoch-gated: `terminalReclaim()` destroys synchronously when
  `isOlder(epoch, minReaderEpoch)` and Non-RT; otherwise stores.
- Growth law §3 governs residency counts exactly (<0.5% error).

### Policy boundary (does not exist yet)

- `TerminalReclaimAuthority` is a growable store with **no capacity field anywhere**
  (verified: no `kMaxTerminalEntries` / `terminalCapacity` / `maxTerminalEntries` in src/).
- Design contract (P-4, cited from ISRRetireRouter.cpp): *"TerminalReclaimAuthority は
  growable store のため常に ownership を受領する… ptr が宙に浮くことはない"* — ownership
  always transfers; there is deliberately no store-failure path.
- Shutdown closure is unconditional: `drainAllQuarantineStore()` drains Q + E + Terminal
  after Audio Thread stop (ISRRetireRouter.cpp:446).

Consequence: `K_terminal` cannot be a physical bound — adding one (e.g.,
`if (terminalResident >= K) reject();`) would collide with the ownership-always-transfers
contract and reintroduce the L>0 leak class that P-4 eliminated. `K_terminal` can only be
a **policy quantity**: an observation/escalation threshold layered on top of the physical
model.

## 10. Candidate K values

| Candidate | Value | Meaning | Merits | Defects | Status |
| --- | --- | --- | --- | --- | --- |
| A — observed peak | 4092 | Max Terminal resident actually measured | Real data; zero assumptions | Guarantees nothing beyond λ≈307/s × 30 s; becomes stale with any workload change | Recorded as fact; **not adopted as K** |
| B — model + design stall | λ_design × T_design − 5120 | Extrapolation of validated law to design inputs | Uses the proven law; transparent | **Blocked**: no authoritative `T_stall_design` or `λ_design` exists (§5) | Deferred pending product input |
| C — policy budget | e.g., 8192 | Memory/ops budget line between "absorb" and "escalate now" | Directly actionable; tunable without re-deriving physics | Arbitrary unless anchored to a memory budget or ops requirement | Possible, but must be labeled *policy budget* and anchored (see §11) |

Memory anchor for Candidate C (empirical): during T3-30s, Private bytes grew
357 MB → 368 MB while ~9200 worlds were outstanding (peak ≈ 4000 Terminal-resident +
full D/Q/E) ⇒ **≈ 1.2 KB/world** in this harness configuration (idle publish, minimal
graph snapshot). Caveat: production worlds carry larger immutable snapshots (IR-bearing
graphs); a production bytes/world measurement is required before anchoring a numeric
budget on memory.

## 11. Recommended K_terminal policy

**Recommendation (to be ratified before Step 5-VI):**

> `K_terminal` is an **escalation-evidence threshold**, not an admission limit.
> Its primary semantic is boolean and rate-based, not a single magic constant:
>
> 1. **Terminal admission event** — first `ΔterminalStoreCount > 0` while a reader is
>    stalled ⇒ EMERGENCY-class evidence that D+Q+E are exhausted and unbounded growth is
>    active. This is the most information-dense signal in the chain: it fires exactly at
>    λT crossing 5120 and implies the stuck-reader condition has already survived ≥
>    (C_total/λ) seconds.
> 2. **Growth confirmation** — sustained `ΔterminalResident > 0` across consecutive
>    monitor ticks distinguishes "one straggler" from "linear growth at λ".
> 3. **Optional numeric budget (Candidate C)** — a `K_policy` alarm line may be added for
>    ops visibility, but it MUST be (a) labeled policy-budget, (b) anchored to a memory
>    budget via measured bytes/world, and (c) implemented as alert/telemetry only — never
>    as a rejection gate.

Supporting arguments:

- The existing pressure machinery is already D-centric and tiered
  (`evaluateRetirePressureLevelNoRt`: Mild ≥75% → coalescing, Medium ≥90% → publication
  throttle, Severe ≥95% → strict admission/emergency reclaim, Critical = Severe ∧ depth ≥
  dynamic hwm, default watermark 3072). Terminal admission is strictly *past* everything
  that machinery measures — it belongs one severity tier above Critical.
- The stuck-reader escalation path already exists
  (`EVENT_READER_STUCK` → `quarantineReader()` + high-priority retire intent,
  AudioEngine.Timer.cpp) and is the correct *remedy* channel; Terminal evidence should
  trigger/corroborate it, not replace it.
- Rejecting admissions at Terminal would violate P-4 and recreate the leak class it
  removed; alerting preserves the invariant while restoring observability.

## 12. Rejection criteria / rationale

| Rejected item | Reason |
| --- | --- |
| `K_terminal = 4092` (Candidate A as final K) | Observation ≠ decision; valid only for one (λ, T) point; no design authority |
| `K_terminal = 8184` (= 2 × 4092) | Doubles an arbitrary point; the "×2" has no derivation (S-factor route is invalid, §6) |
| S = 2 via peak/steady ratio | Non-derivable: steady ≈ 0 ⇒ 0/0; engagement is threshold-like (§6) |
| Admission rejection at Terminal (`reject()` when resident ≥ K) | Violates ownership-always-transfers contract (P-4); recreates L>0 leak class (§9) |
| Fixing `T_stall_design = 30 s` now | 30 s is a measurement ceiling, not a requirement (§5) |
| Using sampled `λ_terminal_peak` in K math | +7.5% discretization bias; caused the transient 17% model error (§4) |
| Inferring Q capacity from OBS `Q_resident` | Field is aggregate (store + auxiliary counter); T4-B/C showed 663/1025 against a 512-cap store |

## 13. Impact on HealthMonitor

No HealthMonitor code was changed in this step. The state ladder below fixes the intended
semantics so Step 5-VI implements thresholds rather than inventing them:

```text
D pressure          pendingRetire vs dynamic hwm      → EXISTING pressureLevel 0..3 (D-only view)
Q/E pressure        emergencyQuarantineResidentCount   → pure E-store gauge (cap 512)
                    quarantineOverflowCount()          → spill has begun (Q or E overflowed)
Terminal admission  ΔterminalStoreCount > 0            → NEW signal: D+Q+E exhausted (evidence)
Terminal growth     ΔterminalResident > 0 sustained    → NEW signal: unbounded growth active
Stuck reader        detectStuckReaders()               → EXISTING (residency 1s/10s/30s tiers)
                    EVENT_READER_STUCK → quarantine    → EXISTING remedy path
```

Key semantic points established by T3/T4 for Step 5-VI:

1. `pressureLevel = 3` fires on **D depth** (~95% of dynamic hwm) — i.e., *before*
   Terminal. It never sees Terminal. T4-C reached PL=3 at pendingRetire≈3306 while
   Terminal admission began later at λT>5120.
2. E-store count (`emergencyQuarantineResidentCount`) is the cleanest pre-Terminal
   precursor: E filling ⇒ D and Q-store are full ⇒ Terminal admission is next.
3. Terminal signals are the *corroboration* channel for stuck-reader escalation, not a
   separate fault domain: in all measurements, Terminal > 0 co-occurred with a
   test-induced stuck reader and vanished after `exitReader()` (79 ms full drain).
4. `quarantineReader()` exclusion semantics mean escalation restores forward progress even
   while the offending reader stays parked — the monitor can act without a code change to
   the reclaim path.

## 14. Step 5-VI inputs

PROPOSAL table (implementation deferred to Step 5-VI; values marked ⚠ need ratification):

| # | Signal | Source API | Condition | Proposed severity | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | D near-full | `pendingRetireCount()` | ≥ 0.75 × hwm | existing PL1 (coalesce) | unchanged |
| 2 | D critical | `pendingRetireCount()` | Severe ∧ ≥ hwm | existing Critical | unchanged |
| 3 | E engaging | `emergencyQuarantineResidentCount()` | > 0 | Warning+ | pure store count; earliest pre-Terminal precursor |
| 4 | Spill occurred | `quarantineOverflowCount()` | > 0 | Warning+ | Q/E overflowed at least once |
| 5 | Terminal admission | `terminalStoreCount()` delta | > 0 within window | **Emergency** | core new signal; implies λT crossed 5120 |
| 6 | Terminal growth | `terminalReclaimResidentCount()` delta | > 0 across 2 ticks | **Emergency (sustained)** | confirms linear growth |
| 7 | Numeric budget alarm | `terminalPeakResident()` | ≥ K_policy ⚠ | Alert only | Candidate C; anchor via memory budget (bytes/world TBD on production graphs); never a reject gate |
| 8 | Correlation tag | `activeReaderCount()` / `minReaderEpoch()` | snapshot with events | metadata | ties Terminal evidence to the stalled reader |

Open items carried into Step 5-VI:

- Ratify §11 recommendation (boolean/rate semantics first, optional numeric budget).
- Obtain product input for `T_stall_design` if Candidate B is ever wanted.
- Measure production bytes/world before anchoring any memory-based `K_policy`.
- Optional instrumentation split: expose Q-store count separately from the auxiliary
  component of the OBS `Q_resident` aggregate.

---

## Annex A — Confirmed facts at Step 5-V close (per instruction)

1. **Terminal growth model**: $K_{growth}=\max(0,\lambda T-5120)$ — measured, <0.5% error.
2. **Engagement boundary**: $\lambda T>5120$ starts Terminal growth.
3. **S=2**: not merely unadopted — the ratio-based derivation is invalid for this system
   (steady ≈ 0 ⇒ 0/0; threshold-like engagement).
4. **4092**: observed peak (`terminalPeakResident_max`), explicitly NOT the final
   `K_terminal`.
