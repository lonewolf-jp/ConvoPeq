# ConvoPeq ISR Bridge Runtime v5.5.1 コミットメッセージ案（P1-C01順）

Date: 2026-05-28
Base: `ConvoPeq_ISR_Bridge_Runtime_v5.5.1_実装タスク表_1コミット1責務_2026-05-28.md`
Rule: 1コミット=1責務

このファイルは、P1-C01 から X-C03 までの順序で、そのまま使えるコミットメッセージ案を提供する。

---

## P1-C01

Subject: `isr(p1): add acceptsRuntimePublication API and unified admission predicate`

Body: Add `acceptsRuntimePublication() noexcept` as single admission gate API. Bind decision to `lifecycleState / shutdownPhase / shutdownRuntime_`. Keep semantics strict so only Running accepts publication.

Rollback: Revert API addition in `AudioEngine.h` and related source.

## P1-C02

Subject: `isr(p1): gate requestRebuild and appendPublicationIntent at entry`

Body: Apply admission gate to `requestRebuild` and `appendPublicationIntent`. Enforce early return on reject to avoid side effects. Prevent new publication intent during shutdown phases.

Rollback: Revert guard blocks at function entry points.

## P1-C03

Subject: `isr(p1): gate prepareCommit and executeCommit for shutdown safety`

Body: Apply gate checks at `prepareCommit` and `executeCommit` boundaries. Ensure no commit-stage side effects when publication is rejected. Preserve deterministic shutdown transition behavior.

Rollback: Revert commit-stage guard insertion only.

## P1-C04

Subject: `isr(p1): normalize publication naming to appendPublicationIntent`

Body: Standardize publication enqueue terminology to `appendPublicationIntent`. Treat `enqueuePublication` as legacy alias in comments and docs only. Reduce naming ambiguity across admission and publication flow.

Rollback: Revert rename-only changes.

---

## P2-C01

Subject: `isr(p2): lock RuntimeBuildFingerprint field set for deterministic reuse`

Body: Fix fingerprint structure fields per v5.5.1 contract. Keep versioned identity based on semantic rebuild inputs. Align snapshot schema with FREEZE specification.

Rollback: Revert fingerprint struct definition changes.

## P2-C02

Subject: `isr(p2): split snapshot flow into capture finalize seal handoff`

Body: Refactor snapshot lifecycle into explicit staged pipeline. Enforce worker handoff of sealed snapshot only. Remove implicit mixed-stage transitions.

Rollback: Revert stage-splitting functions and call-site rewiring.

## P2-C03

Subject: `isr(p2): enforce finalize determinism by removing non-semantic inputs`

Body: Remove finalize dependencies on timing, thread ordering, allocation ordering, and pointer identity. Keep finalize as pure semantic normalization. Preserve stable fingerprint output for identical inputs.

Rollback: Revert finalize-internal deterministic filtering changes.

## P2-C04

Subject: `isr(p2): prohibit reuse on fingerprintVersion mismatch`

Body: Add strict mismatch handling with `reuse=false` for version divergence. Block partial equivalence ambiguity across version boundaries. Route mismatches to rebuild path.

Rollback: Revert version-check branch updates.

## P2-C05

Subject: `isr(p2): remove runtime pointer reads from rebuild worker`

Body: Eliminate worker direct reads of active and fading runtime pointers. Restrict worker inputs to sealed snapshot payload only. Align with Rule-4 snapshot-only rebuild contract.

Rollback: Revert worker-side pointer access removal.

---

## P3-C01

Subject: `isr(p3): fix backpressure thresholds to HWM3072 LWM1024`

Body: Set backpressure constants to v5.5 fixed values. Keep invariant `HWM > LWM` at all times. Stabilize saturation baseline across runs.

Rollback: Revert threshold constant updates.

## P3-C02

Subject: `isr(p3): apply uniform scale clamp 0.75 to 1.50`

Body: Apply clamp range to sampleRate, irComplexity, oversampling, and memoryPressure scales. Prevent unbounded scaling outside contract window. Keep saturation policy numerically stable.

Rollback: Revert clamp integration changes.

## P3-C03

Subject: `isr(p3): restrict memoryPressureScale to runtime-local metrics`

Body: Limit pressure inputs to approved runtime-local deterministic metrics. Remove OS/global/external allocator heuristic dependencies. Align pressure model with FREEZE Rule-Q.

Rollback: Revert pressure input source filtering.

## P3-C04

Subject: `isr(p3): enforce monotonic stabilization policy in saturation state`

Body: Forbid threshold relaxation during saturation. Allow only stabilization-direction updates. Prevent queue growth encouragement under pressure.

Rollback: Revert saturation state transition restrictions.

## P3-C05

Subject: `isr(p3): implement stepwise conservative recovery with hysteresis`

Body: Add recovery trigger on `queueDepth < LWM`. Apply conservative stepwise threshold relaxation. Avoid oscillating recovery behavior.

Rollback: Revert recovery policy implementation.

## P3-C06

Subject: `isr(p3): add required telemetry set for backpressure and shutdown diagnostics`

Body: Add fixed telemetry identifiers required by v5.5.1. Include saturation enter/exit and publication/rebuild counters. Keep collection non-RT and lightweight.

Rollback: Revert telemetry field additions and wiring.

---

## P4-C01

Subject: `isr(p4): codify execution-local mutable state boundary`

Body: Enforce Rule-3 and Rule-19 boundary in DSP state ownership. Keep mutable execution state local and non-publishable. Prevent shared mutable authority leakage.

Rollback: Revert DSP state boundary changes.

## P4-C02

Subject: `isr(p4): separate runtime visibility object from execution authority`

Body: Split visibility representation from execution control state. Keep RuntimeGraph free of execution mutable ownership. Reduce cross-thread authority drift risk.

Rollback: Revert visibility and authority separation changes.

## P4-C03

Subject: `isr(p4): wire CI fail-stop checks for RT worker resurrection rules`

Body: Enforce fail-stop checks for Rule-2, Rule-4, Rule-11, and Rule-23 in CI. Block merges on RT-path violations and resurrection paths. Convert critical invariants to automated gates.

Rollback: Revert CI script and wiring updates.

---

## P5-C01

Subject: `isr(p5): move crossfade preparation to message-thread prepared state`

Body: Prepare `CrossfadePreparedState` entirely on message thread. Fix prepared fields per design contract. Keep RT path free of initialization work.

Rollback: Revert prepared-state creation and publish changes.

## P5-C02

Subject: `isr(p5): make audio thread crossfade path activate-only`

Body: Remove RT-side reset and init operations from crossfade path. Keep audio thread limited to activate and progression behavior. Preserve RT boundedness and determinism.

Rollback: Revert RT helper changes for activate-only path.

## P5-C03

Subject: `isr(p5): remove cross-runtime mutable progression sharing`

Body: Eliminate shared mutable progression across runtimes. Keep progression execution-local only. Align crossfade authority isolation with Rule-19.

Rollback: Revert progression ownership changes.

---

## X-C01

Subject: `isr(x): enforce deterministic rebuild collapse and must-execute exclusions`

Body: Apply latest-generation-wins consistently. Enforce safe-to-collapse criteria and class/fingerprint match. Protect must-execute rebuild classes from collapse.

Rollback: Revert collapse decision logic changes.

## X-C02

Subject: `isr(x): enforce strict drained completion and ban post-drain resurrection`

Body: Require all five drained conditions before shutdown completion. Disallow enqueue, retry, and relaunch after drained state. Preserve deterministic shutdown finality.

Rollback: Revert drained-state policy changes.

## X-C03

Subject: `isr(x): add DoD automation scaffold for v5.5.1 acceptance`

Body: Add automated checks scaffold for 10-point DoD criteria. Tie static and runtime verifications to acceptance path. Improve objective completion judgment.

Rollback: Revert DoD scaffold and test-harness additions.

---

## すぐ着手用（最初の3本）

First: `isr(p1): add acceptsRuntimePublication API and unified admission predicate`
Second: `isr(p1): gate requestRebuild and appendPublicationIntent at entry`
Third: `isr(p1): gate prepareCommit and executeCommit for shutdown safety`

この3本を先に適用すると、shutdown中 publication 封止の骨格を最短で確立できる。
