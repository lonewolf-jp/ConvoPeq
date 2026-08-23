# D101-9 Step 5-VI-A — HealthMonitor Threshold Contract Audit / Policy Ratification

> **Status**: COMPLETE — contract ratified on code evidence. **Production code UNCHANGED**
> (zero diffs to `RuntimeHealthMonitor.*`, `AudioEngine.*`, `ISRRetireRouter.*`).
> **Baseline**: current `src/` tree (mirrored in latest `ConvoPeq.md`).
> **Inputs**: Step 5-V policy decision (`phase-d101-9-step5-v-k-terminal-derivation-and-policy-decision.md`),
> T2/T3/T4 measurements.
> **Next**: 5-VI-B (state/delta ownership design) → 5-VI-C (minimal implementation) →
> 5-VI-D (tests) → 5-VI-E (T1–T4 regression).

---

## 1. Code audit — signal inventory (enumerated from implementation)

Every row below was verified against the current source (file:line cited). The existing
D-centric pressure machinery and the Step 5-I Terminal telemetry are listed **separately**
and are not merged.

### 1.1 D-centric pressure machinery (existing, unchanged)

| # | Item | Implementation site | Verified semantics |
| --- | --- | --- | --- |
| A1 | `pendingRetireCount()` | `ISRRetireRouter.cpp:586` → `EpochDomain` → `deferredDeletionQueue.sizeApprox()` | Lock-free atomic pair (enqueue/dequeue positions). D-queue depth only. |
| A2 | `evaluateRetirePressureLevelNoRt()` | `AudioEngine.Retire.cpp:383`; thresholds `:20-22` (Mild ≥75% / Medium ≥90% / Severe ≥95%) | ratio = depth×100/hwm; returns 0..3. **D-only view — no Q/E/Terminal input.** |
| A3 | Dynamic HWM | `Retire.cpp:184-199` (`computeBackpressureScales`), default `retireHighWatermark_ {3072}` (`AudioEngine.h:4848`) | hwm scales up under sustained depth; Critical = Severe ∧ depth ≥ hwm. Explains T4-C PL=3 at depth≈3306. |
| A4 | Policy application | `applyRetirePressurePolicyNoRt` (`Retire.cpp:398`) | PL1→`retirePressureCoalescingActive_`, PL2→`...PublicationThrottleActive_`, PL3/Critical→`retirePressureAdmissionStrict_` + emergency reclaim boost. |
| A5 | Admission gate consumers | `Threading.cpp:27`, `PublicationAdmission.cpp:41` | Strict-admission / throttle flags consumed on publish admission path. |

### 1.2 Quarantine-tier telemetry (existing, semantics pinned)

| # | Item | Implementation site | Verified semantics |
| --- | --- | --- | --- |
| B1 | `quarantineResidentCount()` | `ISRRetireRouter.cpp:432` | **Q-store + E-store sum** (both mutex-guarded vector sizes). Pure store counts, but NOT Q-isolated. Pure-Q must be derived: `quarantineResidentCount() − emergencyQuarantineResidentCount()`. NonRT-only (mutex). |
| B2 | `emergencyQuarantineResidentCount()` | `ISRRetireRouter.cpp:503` | **Pure E-store resident count** (cap 512). Mutex-guarded read; NonRT-only. Cleanest pre-Terminal gauge. |
| B3 | `quarantineOverflowCount()` | `ISRRetireRouter.cpp:440` | **Cumulative Q+E overflow sum** (`m_retireQuarantine.overflowCount() + m_emergencyQuarantine.overflowCount()`). Historical evidence only — cannot attribute to E alone. ⚠ Must never be read as "E capacity gauge". |

### 1.3 Terminal telemetry (Step 5-I additions, currently zero HealthMonitor consumption)

| # | Item | Implementation site | Verified semantics |
| --- | --- | --- | --- |
| C1 | `terminalStoreCount()` | `ISRRetireRouter.h:117` → `terminalStoreCount_` (atomic u64, incremented in `store()`) | **Cumulative admissions** (Generic+World). Lock-free acquire read; RT-safe to read. Delta requires external stateful observation. |
| C2 | `terminalReclaimResidentCount()` | `ISRRetireRouter.cpp:536` → `TerminalReclaimAuthority::residentCount()` | Current resident entries (mutex-guarded; lock-free twin `residentCountAtomic_` exists for RT-side predicates). Current-state gauge. |
| C3 | `terminalPeakResident()` | `ISRRetireRouter.h:104` (CAS max update in `store()`) | **Running maximum** — persists after recovery; represents history, not current fault state. Diagnostics only (§7). |
| C4 | `terminalDrainAllCount()` / `terminalDrainEntryCount()` | `ISRRetireRouter.h:122/127` | Cumulative shutdown-drain counters (drainAll invocations / entries drained). Shutdown-closure evidence. |

### 1.4 Reader-stall detection & recovery (existing)

| # | Item | Implementation site | Verified semantics |
| --- | --- | --- | --- |
| D1 | `detectStuckReaders(threshold)` | `EpochDomain.h:459` (3-pass: Chronic residency>30s ∧ pending>0; Warning >10s ∧ pending>0; EpochGap gap>threshold ∧ residency>1s) | Returns `StuckReaderInfo{readerIndex, readerEpoch, enterCount, currentEpoch, minReaderEpoch, pendingRetireCount, residencyTimeUs, isStuck, isChronic}`. HealthMonitor calls it with threshold=10 (`RuntimeHealthMonitor.cpp:621`). |
| D2 | `quarantineReader(idx)` | `EpochDomain.h:276` | Immediate (depth==0) or deferred (`kPendingQuarantineFlag` promoted on exit). Excludes the reader from `getMinReaderEpoch()` ⇒ restores reclaim progress even while the reader stays parked. |
| D3 | `EVENT_READER_STUCK` | `RuntimeHealthMonitor.h:48` (= 3001) | Emitted by monitor stall diagnosis (`RuntimeHealthMonitor.cpp:487-518`; severe = pendingRetire>100 ∨ residency>30s). |
| D4 | Recovery dispatch | `AudioEngine.Timer.cpp:1705-1726` | `onHealthEvent(EVENT_READER_STUCK)` → log evidence → `m_retireRouter->quarantineReader(idx)` → if immediate ∧ slot valid → high-priority RetireIntent (dspSlot, generation=readerEpoch). **No new Terminal recovery path needed.** |

### 1.5 HealthMonitor structure (observation infrastructure)

| # | Item | Implementation site | Verified semantics |
| --- | --- | --- | --- |
| E1 | Tick cadence | `AudioEngine.Timer.cpp:1192-1195` — `m_healthMonitor.tick()` inside `timerCallback()` when `!isShutdownInProgress()` | **100 ms JUCE Timer on Message Thread** — identical cadence to the `[D101_9_T5_OBS]` probe. Single-threaded (NonRT) ⇒ stateful deltas are race-free by construction. |
| E2 | `TrendSnapshot` | `RuntimePolicyEngine.h:71` | Fields: pendingRetire, publicationSeq, maxRetireAgeUs, activeReaderCount, readerStuckCount, freezeDetected, activeFaultMask, restoreGeneration, epochAdvanceCount, lastCompletedEpoch, publicationGeneration, restorePhase. **No terminal fields today.** |
| E3 | `takeSnapshot()` | `RuntimeHealthMonitor.cpp:616` | Fills snapshot from router atomics + `detectStuckReaders(10)`. |
| E4 | `computeTrend(before, now)` | `RuntimeHealthMonitor.cpp:652` | Existing before/after **delta engine** (retireDelta/ageDelta/pubDelta → RecoveryOutcome). Natural home for cumulative-counter deltas. |
| E5 | Verification loop | `RuntimeHealthMonitor.cpp:61-62, 212` | `markForVerification(action, takeSnapshot())` keeps baseline/now snapshot pairs — precedent for paired-sample evaluation. |
| E6 | `BackpressureWindow` | `RuntimeHealthMonitor.h:411-437` | Existing peak+average+count windowed-statistics member — precedent for monitor-local windowed state. |

---

## 2. Threshold Contract (ratified)

Layering principle: **Tier 1 stays exactly as-is; Terminal sits strictly outside it.**
Each tier adds a distinct evidence class; no tier reinterprets another's counter.

### Tier 0 — Normal

```text
pendingRetire < Mild threshold (existing dynamic-hwm ratio < 75%)
AND emergencyQuarantineResidentCount() == 0
AND ΔterminalStoreCount == 0        (over the 100 ms sample)
AND ΔterminalReclaimResidentCount == 0
```

Behavior: unchanged from current production. No new event.

### Tier 1 — D pressure (UNCHANGED)

Existing `pressureLevel 0..3` from `evaluateRetirePressureLevelNoRt(pendingRetire, dynamicHwm)`
with its existing policy outputs (coalescing / throttle / strict admission / emergency
reclaim boost). **No modification.** Key ratified fact: Terminal is *outside* this ladder —
PL=3 fires on D depth (~95% of hwm) before any Q/E/Terminal engagement (measured: T4-C
reached PL=3 at depth≈3306 while Terminal admission began later at λT>5120).

### Tier 2 — E engaging (new observation, Warning)

```text
emergencyQuarantineResidentCount() > 0   →  Warning / escalation evidence
```

Semantics: pure E-store occupancy is the earliest *pre-Terminal* precursor (E filling ⇒
D full ∧ Q-store filling). It does **not** assert a Terminal fault — Terminal has not been
reached. Exit: returns to 0 (drain) without latching.

### Tier 3 — Spill occurred (new observation, Warning, historical)

```text
ΔquarantineOverflowCount > 0   (cumulative counter — delta over sample window)
```

Semantics: "Q or E capacity exhaustion occurred at least once". Because the counter is a
**Q+E sum**, the contract explicitly forbids interpreting it as "E became full".
Attribution requires pairing with Tier 2 (E_resident) at the same sample.

### Tier 4 — Terminal admission (new, Emergency — the core addition)

```text
ΔterminalStoreCount > 0   (over the 100 ms sample)
```

Semantics: **one single Terminal admission is itself proof that D+Q+E physical absorption
(5120 combined) was exceeded.** This is an evidence predicate about the spill chain, not a
capacity threshold — no `terminalResident >= K` comparison exists in this tier.
Severity: **Emergency**. Correlation with reader-stuck is mandatory (§3) before attributing
cause; the event carries the full snapshot either way.

### Tier 5 — Terminal sustained growth (new, Emergency-sustained)

```text
ΔterminalReclaimResidentCount > 0  for 2 consecutive 100 ms samples
   → Terminal sustained growth (Emergency, sustained)
```

Separates a one-off straggler (Tier 4 alone) from linear growth at λ. Explicitly excluded
from this tier: any `terminalPeakResident >= K` comparison — 4092 is this campaign's
observed maximum, not a policy boundary (Step 5-V §7/§12), and the peak counter is a
running maximum that outlives the fault (§7 below).

## 3. Reader-stuck correlation (mandatory)

A Terminal signal alone MUST NOT conclude cause=reader-stuck. Every Tier 2–5 event embeds
a same-sample correlation snapshot:

```text
activeReaderCount            (router)
minReaderEpoch               (router)
pendingRetireCount           (router)
emergencyQuarantineResidentCount
terminalStoreCount           (post-delta value)
terminalReclaimResidentCount
```

Classification:

```text
Terminal admission (Tier 4/5)
  + activeReaderCount > 0
  + minReaderEpoch stagnation (across samples)
  ⇒ Reader-stall-correlated Terminal escalation
otherwise ⇒ Terminal escalation (cause unattributed) — still Emergency, cause open
```

Recovery routing: **no new recovery mechanism.** Reader-stall-correlated escalations feed
the existing `EVENT_READER_STUCK` → `quarantineReader()` → high-priority retire path
(D3/D4). Terminal acts purely as an evidence channel that corroborates/triggers the
existing remedy.

## 4. Delta-window decision (100 ms) and state ownership investigation

**Ratified**: sampling interval = **100 ms**, matching the Step 5-III observation cadence
and the existing `tick()` cadence (E1). Both counters needing deltas
(`terminalStoreCount_`, and resident as a current-state series) are observed once per tick
on the Message Thread ⇒ delta state is single-writer/single-reader, race-free.

**State ownership — investigated, final design DEFERRED to 5-VI-B.** Options found in code:

| Option | Mechanism already present | Fit assessment |
| --- | --- | --- |
| (a) Extend `TrendSnapshot` + `computeTrend` | before/now paired snapshots already flow through verification (`E4`/`E5`) | Strongest structural fit: cumulative-counter deltas are exactly what computeTrend already computes for pendingRetire/publicationSeq. Adding `terminalStoreCount`/`terminalReclaimResidentCount` fields is additive and non-breaking. |
| (b) Monitor-local previous-sample members | `m_prev*State_` pattern used by every check; `BackpressureWindow` (E6) precedent for windowed stats | Needed regardless for the "2 consecutive samples" Tier-5 counter (consecutive-growth count cannot live in a stateless snapshot pair). |
| (c) PolicyEngine | owns escalation ladder / verification state machine | Wrong layer for raw observation: monitor observes→emits events; policy decides actions. Keeping raw deltas in PolicyEngine would invert the existing layering. |
| (d) AudioEngine timer state | OBS capture block reads telemetry every 100 ms but is `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` diagnostics-only | Rejected: production semantics must not live behind a diagnostics flag. |

**Provisional direction (to be finalized in 5-VI-B)**: (a) + (b) combination — extend
`TrendSnapshot` with the two terminal counters (gives delta via existing pairs) plus a
monitor-local consecutive-growth counter and last-sample cache for Tier 5. Decision points
remaining for 5-VI-B: whether Tier-2/3 checks join `tickFast`-style fast path, event
debounce/latch rules, and exit conditions for Emergency tiers.

## 5. `terminalPeakResident` — excluded from fault logic (ratified)

1. Step 5-V concluded `K_terminal` numeric budget is unset (Candidates B/C blocked or
   unanchored); there is no ratified K to compare against.
2. `terminalPeakResident_` is a CAS-max **running maximum**: after recovery it still holds
   the stale peak of the worst episode. `peak >= X` can therefore latch a fault condition
   that no longer exists — it represents history, not current state.
3. Primary fault signals are the **delta series** (`ΔterminalStoreCount`,
   `ΔterminalReclaimResidentCount`). Peak remains diagnostics/post-analysis only
   (already exported in `[D101_9_T5_OBS]`).

## 6. Ratified Threshold Contract Table (deliverable)

| Signal | 現行API | 現行判定 | 新Semantics | Severity | Recovery |
| --- | --- | --- | --- | --- | --- |
| D pressure | `pendingRetireCount()` → `evaluateRetirePressureLevelNoRt()` | existing PL 0..3 vs dynamic hwm | **unchanged** | existing (PL1/2/3/Critical) | existing (coalesce/throttle/strict/emergency-reclaim) |
| E engaging | `emergencyQuarantineResidentCount()` | none (telemetry only) | `> 0` | Warning | none (self-clears on drain) |
| Q/E overflow | `quarantineOverflowCount()` | none (telemetry only) | delta `> 0` (cumulative) | Warning | none (historical evidence) |
| Terminal admission | `terminalStoreCount()` | none | delta `> 0` per 100 ms sample | **Emergency** | reader-stuck path (`EVENT_READER_STUCK` → `quarantineReader()`) |
| Terminal growth | `terminalReclaimResidentCount()` | none | positive delta × 2 consecutive samples | **Emergency (sustained)** | reader-stuck path (same) |
| Reader stuck | `detectStuckReaders()` / `EVENT_READER_STUCK` | existing (1s/10s/30s tiers) | **unchanged** | existing | existing `quarantineReader()` + high-priority retire |
| Terminal peak | `terminalPeakResident()` | none | **diagnostics only** — never a fault predicate | — | — |

### Per-row engineering properties

| Signal | RT-safe read? | NonRT-only? | Atomic? | Snapshot source | Stateful delta required? | Existing event available? | Existing recovery available? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D pressure | Yes (lock-free sizeApprox) | Producer/consumer NonRT; PL evaluated NonRT | Yes | `TrendSnapshot.pendingRetire` (existing) | No (level computed per-tick) | Yes — retire-pressure events (work37/39 chain) | Yes — PL policy actions (A4/A5) |
| E engaging | Mutex-guarded read — **NonRT-only** | Yes (std::mutex in store) | Size from mutex state; overflow counters atomic | To be added to snapshot (5-VI-B) | No (absolute gauge) | No — new event code required | N/A (evidence only) |
| Q/E overflow | Atomic counters; router accessor non-mutex | Increment sites NonRT | Yes (cumulative) | To be added (delta needs prev-sample cache) | **Yes** (cumulative → delta) | No — new event code required | N/A (evidence only) |
| Terminal admission | Yes (atomic u64 acquire) | store()/drain() NonRT-only | Yes (cumulative) | To be added to `TrendSnapshot` (option a) | **Yes** (prev-sample cache, option b) | No — new event code required | Routed to existing reader-stuck path |
| Terminal growth | Mutex variant NonRT-only; lock-free twin exists (`residentCountAtomic_`) | Yes | Yes (current-state) | Same snapshot fields as above | **Yes** (consecutive-count ≥ 2) | No — new event code required | Routed to existing reader-stuck path |
| Reader stuck | Yes (atomics) | Monitor thread (Message) | Yes | `TrendSnapshot.readerStuckCount` + `StuckReaderInfo` (existing) | No | Yes — `EVENT_READER_STUCK` (3001) | Yes — `quarantineReader()` + RetireIntent |
| Terminal peak | Yes (atomic u32) | Updated in store() (NonRT) | Yes (CAS max) | `[D101_9_T5_OBS]` export (existing) | No | N/A — diagnostics only | N/A |

Legend: "NonRT-only" = safe to call only off the audio thread (mutex/allocation);
monitor tick runs on the Message Thread, satisfying this for all adopted signals.

## 7. Explicit non-goals ratified at this step

- `terminalPeakResident >= K` fault predicate — rejected (§5).
- Any numeric `K_terminal` (4092 / 8192) — not adopted (Step 5-V §10/§12).
- S=2 revival — invalid derivation (Step 5-V §6).
- `T_stall_design = 30 s` — not introduced (Step 5-V §5).
- Terminal admission-limit / reject gate — violates P-4 ownership-always-transfers.
- Reading `quarantineOverflowCount()` as an E-only capacity gauge — it is a Q+E sum (B3).
- Merging Terminal telemetry into the D-centric pressureLevel computation — layers stay
  separate (§2 Tier 1 note).

## 8. Completion checklist (Step 5-VI-A exit criteria)

- [x] 現行 HealthMonitor の実装を最新ソースで確認（§1.5: tick 100ms/Message Thread、TrendSnapshot、computeTrend、BackpressureWindow）
- [x] D/Q/E/Terminal の責務境界を確定（§1.1–1.3、§2 Tier 1 外側に Terminal）
- [x] `quarantineOverflowCount()` を E-only と誤認しない（B3 — Q+E 合算と明記）
- [x] Terminal を admission limit としない（§2 Tier 4、§7）
- [x] `4092` を threshold として採用しない（§5、§7）
- [x] `8192` を採用しない（§7）
- [x] `S=2` を復活させない（§7）
- [x] `T_stall_design=30s` を導入しない（§7）
- [x] Terminal admission と Terminal sustained growth を分離（Tier 4 / Tier 5）
- [x] Reader-stuck recovery との接続点を確定（§3 — 既存 EVENT_READER_STUCK 経路のみ）
- [x] cumulative counter の delta semantics を確定（§4 — 100ms 単一サンプル差分、単一スレッドで競合なし）
- [x] 100ms observation interval を維持（既存 tick() と同一周期）
- [x] **HealthMonitor production code は未変更**（本ステップの diff は本ドキュメントのみ）

## 9. Handoff to Step 5-VI-B

Scope inputs prepared by this audit:

1. Decide final state ownership (provisional: TrendSnapshot extension + monitor-local
   consecutive-growth counter — §4 options a+b).
2. Define new event codes (Warning for Tier 2/3, Emergency for Tier 4/5) following the
   existing numbering blocks (`3001` reader-stuck, `5001/5002` learner backpressure).
3. Define latch/exit rules for Emergency tiers (candidate: clear when
   `terminalReclaimResidentCount == 0` ∧ `ΔterminalStoreCount == 0` for N ticks, plus
   existing CriticalExitCondition stability gating).
4. Confirm snapshot field additions keep `takeSnapshot()` allocation-free and
   Message-Thread-confined.
