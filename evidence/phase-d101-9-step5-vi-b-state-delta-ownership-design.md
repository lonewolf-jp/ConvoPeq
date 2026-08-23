# D101-9 Step 5-VI-B — HealthMonitor State / Delta Ownership Design

> **Status**: COMPLETE — design ratified. **Production code UNCHANGED** (zero diffs in this step).
> **Inputs**: 5-VI-A contract audit, Step 5-V policy decision, current `src/` tree (re-verified).
> **Next**: 5-VI-C minimal implementation → 5-VI-D tests → 5-VI-E T1–T4 regression.

---

## B-1. Re-audit against current code (drift check vs 5-VI-A)

All 5-VI-A audit findings re-verified against the current source. **No drift found.**

| Item | Verified site | Consistent with 5-VI-A? |
| --- | --- | --- |
| `tick()` cadence | `AudioEngine.Timer.cpp:1192-1195` — 100 ms JUCE Timer, Message Thread, gated by `!isShutdownInProgress()` | ✅ |
| `takeSnapshot()` | `RuntimeHealthMonitor.cpp:616` — reads router atomics + `detectStuckReaders(10)` (`:621`) | ✅ |
| `computeTrend(before, now)` | `RuntimeHealthMonitor.cpp:652` — delta engine used ONLY by the verification loop (`:61-62`, `:212`) | ✅ |
| `TrendSnapshot` | `RuntimePolicyEngine.h:71` — no terminal fields today | ✅ |
| `MonitorState` | `RuntimeHealthMonitor.h:47` — `{Normal, Warning, Error}` | ✅ |
| `HealthEvent` | `RuntimeHealthMonitor.h:17-27` — `{timestampUs, severity(Info/Warning/Error), eventCode, value, slot, readerIndex, readerEpoch, readerDepth, residencyTimeUs}` | ✅ |
| `emitOnTransition()` | `RuntimeHealthMonitor.cpp:461-475` — fires only on state change; **returns silently when newState == Normal** (no recovery event); no-op without callback | ✅ |
| `EVENT_READER_STUCK` dual semantics | `RuntimeHealthMonitor.cpp:487-525` — transition fire once + **10 s periodic evidence** while stuck persists (`kStuckEvidenceIntervalUs = 10'000'000`, `h:245`; transition tick suppresses periodic timer) | ✅ |
| `markForVerification()` / `VerificationEntry` | `RuntimePolicyEngine.h:110-121` — paired `baselineSnapshot`/`lastSnapshot`, `stalledCount` cap 3, `verifyAfterUs` init 50 ms | ✅ |
| `EVENT_*` full enumeration | §B-8 below (all 24 codes listed) | ✅ |
| `AudioEngine::onHealthEvent()` reader-stuck handling | `AudioEngine.Timer.cpp:1705-1726` — quarantineReader + high-priority RetireIntent | ✅ |
| Admission-control flags | `retirePressureAdmissionStrict_` consumers: `Threading.cpp:27`, `PublicationAdmission.cpp:41` | ✅ |

Additional facts established this step:

- `emitOnTransition` precedent: **silent return to Normal** (learner backpressure resets
  `m_prevLearnerBackpressureState_` without emitting; retire-age is the exception that has
  an explicit `*_NORMAL` code). Two viable exit-notification styles exist in-tree.
- `CriticalExitCondition.stableDuration` is a **boolean flag** (`h:105`) set by the global
  critical-exit evaluator — no concrete stability duration constant exists in the monitor
  to inherit for episode-exit N (consequence in §B-6).

---

## B-2. TrendSnapshot extension — necessity-based field decision

Principle ratified: **Snapshot = raw observations only. Derived/latch state = HealthMonitor
local.** A field enters `TrendSnapshot` only if a tier predicate or correlation requirement
consumes its per-tick value (absolute gauge) or its paired-sample value (cumulative counter).

| Candidate field | Type | Verdict | Necessity rationale |
| --- | --- | --- | --- |
| `terminalStoreCount` | uint64 | **ADD** | Tier 4 primary input — cumulative; delta requires paired samples carried by snapshots. |
| `terminalReclaimResidentCount` | uint64 | **ADD** | Tier 5 primary input — current-state series; also the headline `value` for events. |
| `emergencyQuarantineResidentCount` | uint64 | **ADD** | Tier 2 absolute gauge evaluated every tick; also mandatory correlation evidence (5-VI-A §3). |
| `quarantineOverflowCount` | uint64 | **ADD** | Tier 3 cumulative-delta input; pairing via snapshots prevents ad-hoc second read sites. |
| `minReaderEpoch` | uint64 | **ADD** | Correlation requirement (5-VI-A §3): every Terminal event must record same-sample `minReaderEpoch(prev/now)`. Cheapest atomic read; core safety-invariant gauge. |

Rejected-for-now: none of the proposed candidates rejected; no other fields added
(payload discipline). `pendingRetire`/`activeReaderCount` already exist in the snapshot.

## B-3. Delta authority — single owner decision

**Ratified: HealthMonitor is the ONE AND ONLY owner of terminal/quarantine raw
observation deltas.**

Architecture:

```text
takeSnapshot()                      ← sole raw-read site (extended with 5 fields)
        │
        ▼
TrendSnapshot now   ──►  evaluateRetireChainTiers(now, m_prevTickSnapshot_)
        │                        │  Δstore, Δresident, Δoverflow  (signed i64)
        │                        │  Tier 2/3/4/5 predicates + episode latches
        ▼                        ▼
m_prevTickSnapshot_ = now   (rolling previous sample — authoritative prev-state holder)
```

- **New private method** `evaluateRetireChainTiers(const TrendSnapshot& now,
  const TrendSnapshot& prev)` — the single site computing these deltas and evaluating
  Tier 2–5. Called once per `tick()`.
- **`m_prevTickSnapshot_`** (new member, `TrendSnapshot`) is the authoritative
  previous-state holder for per-tick deltas.
- **`computeTrend()` remains untouched** — it stays verification-only and does NOT compute
  terminal deltas (avoids a second delta site for the same counters). Verification keeps
  consuming raw snapshot values only.
- First-tick guard: `m_prevTickSnapshotValid_{false}` until the first completed tick;
  tiers 3–5 are skipped on the bootstrap tick (no delta exists yet), Tier 2 evaluates
  immediately (absolute gauge).

Prohibitions honored (all ratified as design constraints):

1. No terminal delta state in `AudioEngine.Timer.cpp`.
2. No production semantics under `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` (OBS block stays
   diagnostics-only).
3. No raw-counter ownership in PolicyEngine (it consumes events/actions only).
4. Exactly one delta computation site per cumulative counter (the method above).

## B-4. Tier 5 — "2 consecutive ticks" explicit state machine

Resident is a **current-value series**, not cumulative: decrease = drain progress
(healthy), not anomaly. All deltas computed as **signed** `int64_t(now − prev)` — no
unsigned wraparound concern at observed magnitudes; a negative `Δstore` on the cumulative
counter would be a counter anomaly (logged, treated as no-admission).

```text
per tick (after first valid pair):
    dRes  = int64(now.resident − prev.resident)
    dStore= int64(now.store   − prev.store)

    case dRes > 0 :  growthTicks = min(growthTicks + 1, 2)
                     if growthTicks == 2 ∧ ¬sustainedLatched:
                         sustainedLatched = true
                         emit EVENT_TERMINAL_GROWTH_SUSTAINED (Error)      [once per episode]
    case dRes == 0:  growthTicks = 0          // plateau — arrival paused, epoch still unsafe
    case dRes < 0 :  growthTicks = 0          // draining — recovery progress (healthy)
                     drainObserved = true     // evidence for exit evaluation

    sustained re-arm: only after episode exit (§B-6)
```

Ratified refinement of the provisional `if (now > previous) ++c; else c = 0;`: the
three-case distinction is kept exactly as instructed — `Δ > 0` grows the counter,
`Δ == 0` and `Δ < 0` both reset it, with `Δ < 0` additionally recorded as drain evidence.
Latch threshold = **two consecutive positive inter-tick deltas** (three samples total):
one positive delta is already covered by Tier 4 admission; the second confirms
continuation (Step 5-V §11 semantics).

## B-5. Tier 4 retrigger semantics — episode latch

Precedent verified: `checkRetireStall` fires on transition once, then emits periodic
evidence every 10 s while the condition persists (`cpp:507-523`). Terminal admission
adopts the identical pattern:

```text
States: TerminalChainNormal ⇄ TerminalAdmissionLatched   (MonitorState-based)

Enter : dStore > 0 while Normal
        → latch Admission, emit EVENT_TERMINAL_ADMISSION (Error) ONCE
        → start episode evidence timer (suppress periodic on the transition tick — same
          double-fire guard as cpp:512)
While latched:
        - further dStore > 0 : NO re-emission (episode continuing)
        - periodic evidence  : re-emit EVENT_TERMINAL_ADMISSION every
          kStuckEvidenceIntervalUs (10 s) while latched — mirrors EVENT_READER_STUCK
          continuity semantics; own timer member (m_lastTerminalEvidenceUs_), NOT shared
          with the reader-stuck timer
Exit  : §B-6 condition met → unlatch silently (emitOnTransition Normal-return precedent),
        optional closure Info event deferred to 5-VI-C decision
```

Consistency note: the existing 10 s periodic evidence exists precisely to avoid silence
during persistent severe conditions; adopting it for Terminal episodes keeps operator
visibility uniform across severe channels without changing event spam characteristics.

## B-6. Emergency latch / exit conditions

Latch (Tier 4 and Tier 5 independently latched, both cleared by one episode exit):

```text
admissionLatched   := set on first dStore > 0            (Tier 4)
sustainedLatched   := set on growthTicks reaching 2      (Tier 5)
```

Exit predicate comparison:

| Candidate | Predicate | Assessment |
| --- | --- | --- |
| A — immediate | `resident == 0` | Rejected as sole condition: straggler admissions can still be in flight while resident momentarily drains → flap. |
| B — 1 tick stable | `resident == 0 ∧ dStore == 0` (one full tick) | **Adopted as the structural exit predicate** — requires both drained residency AND stopped admissions. |
| C — N tick stable | B held for N consecutive ticks | **N = UNDECIDED** — recorded as open tuning parameter. No concrete stability constant exists in-monitor to inherit (`CriticalExitCondition.stableDuration` is a boolean flag; the duration lives in the global critical-exit evaluator). Default implementation value N=1 (= Candidate B exactly) permitted for 5-VI-C; final N must be justified from 5-VI-E regression data or an inherited constant — never invented. |

Interaction with global Critical exit: episode exit is necessary-but-not-sufficient for
`ISRHealthState::Critical → Healthy` — the existing `CriticalExitCondition.canExit()`
stability gating remains the sole authority for overall Critical exit. Episode exit only
clears the Terminal-chain latches and stops their events.

## B-7. Reader-stuck correlation semantics

Principle: **Terminal ≠ reader-stuck. Correlation and causation are separated. The stuck
verdict belongs exclusively to the existing `detectStuckReaders(10)` diagnosis — no new
"N-tick stagnation" detector and no additional tick-count thresholds are introduced.**

At Terminal event emission (same tick):

```text
inputs (all already available):
    snap.activeReaderCount              (existing snapshot field)
    snap.minReaderEpoch                 (NEW snapshot field — prev/now pair)
    stuckInfo                           (existing detectStuckReaders(10) result, same tick)

classification emitted WITH the event:
    correlated        : stuckInfo.isStuck == true
                        → fill ev.readerIndex / ev.readerEpoch / ev.residencyTimeUs
                          from stuckInfo (existing HealthEvent fields)
    suspected         : stuckInfo.isStuck == false
                        ∧ activeReaderCount > 0
                        ∧ minReaderEpoch now == prev (stagnation observation only)
                        → reader fields left unset; stagnation recorded in evidence dump
    uncorrelated      : otherwise → cause open
```

The `minReaderEpochPrev/Now` pair is recorded as evidence only; it never produces a stuck
verdict by itself. Recovery routing is unchanged: correlated escalations flow through the
existing `EVENT_READER_STUCK` → `quarantineReader()` path; Terminal events never trigger
quarantine directly.

## B-8. Event codes — full enumeration and allocation

Existing codes (complete inventory, `RuntimeHealthMonitor.h:33-63`):

```text
1001 EVENT_RETIRE_STALL            1002 EVENT_RETIRE_STALL_WARNING
1009 EVENT_RETIRE_AGE_NORMAL       1010 EVENT_RETIRE_AGE_WARNING
1011 EVENT_RETIRE_AGE_CRITICAL     1012 EVENT_OVERFLOW_RATE_WARNING
1013 EVENT_OVERFLOW_RATE_CRITICAL
2001 EVENT_PUBLICATION_STALL       2002 EVENT_PUBLICATION_WARNING
3001 EVENT_READER_STUCK            3010 EVENT_READER_SLOT_USAGE
4001 EVENT_CROSSFADE_TIMEOUT       4002 EVENT_CROSSFADE_EVENT_DROP
4003 EVENT_CROSSFADE_ABORTED_EMERGENCY
5001 EVENT_LEARNER_BACKPRESSURE_WARNING  5002 EVENT_LEARNER_BACKPRESSURE_ERROR
6000-6003 EVENT_VALIDATION_*       7000-7002 EVENT_WORLD_CONSISTENCY_*
```

Family taxonomy: 1xxx = retire chain, 2xxx = publication, 3xxx = reader, 4xxx = crossfade,
5xxx = learner, 6xxx = validation, 7xxx = world consistency. Free: 1003-1008, 1014+ within
the retire family; 8000+ unallocated.

**Decision**: Terminal/quarantine signals are retire-chain phenomena (spill chain
D→Q→E→Terminal) → continue the 1xxx block at the first free number:

| Code | Name | Severity | Tier |
| --- | --- | --- | --- |
| **1014** | `EVENT_EMERGENCY_Q_ENGAGED` | Warning | Tier 2 |
| **1015** | `EVENT_QUARANTINE_OVERFLOW_DETECTED` | Warning | Tier 3 |
| **1016** | `EVENT_TERMINAL_ADMISSION` | Error | Tier 4 |
| **1017** | `EVENT_TERMINAL_GROWTH_SUSTAINED` | Error | Tier 5 |
| 1018 | `EVENT_TERMINAL_EPISODE_CLEARED` (Info) | Info | exit closure — OPTIONAL, decide in 5-VI-C (silent-exit precedent exists; retire-age NORMAL precedent exists) |

Severity mapping note: `HealthEvent::Severity` has no "Emergency" — the contract's
Emergency maps to `Severity::Error` (+ contributes to `ISRHealthState::Critical` via the
existing health-state aggregation).

## B-9. HealthEvent payload design

**Decision: NO HealthEvent struct change.** The struct is small POD copied into callbacks;
the required six-field evidence is obtainable without growth:

| Evidence item | Carrier |
| --- | --- |
| `terminalStoreCount` | `ev.value` on 1016 (cumulative at emission; Δ ≥ 1 implicit) |
| `terminalReclaimResidentCount` | `ev.value` on 1017 (current resident) |
| `pendingRetire` | same-tick telemetry dump (below) — also already in `TrendSnapshot` |
| `emergencyQuarantineResident` | same-tick telemetry dump |
| `activeReaderCount` | same-tick telemetry dump (also in snapshot) |
| `minReaderEpoch` (prev/now) | same-tick telemetry dump (prev/now from snapshot pair) |
| reader correlation | existing fields `readerIndex` / `readerEpoch` / `residencyTimeUs` (filled only when `correlated`, §B-7) |

Same-tick evidence dump: the event handler captures
`getRuntimeBackpressureTelemetry()` once at event time and logs a structured
`[TERMINAL_EVIDENCE]` line — the established pattern in this codebase (XRUN aggregation
and `[HEALTH]` reader-stuck logging already read backpressure telemetry at event time,
Timer.cpp). Single-snapshot consistency of that accessor was proven in Step 5-III-A.
Alternatives rejected: enlarging `HealthEvent` (payload bloat for all event types);
carrying a snapshot pointer (lifetime/ownership complexity).

---

## Ratified Design Table (deliverable)

| 項目 | 決定内容 |
| --- | --- |
| Snapshot fields | +5: `terminalStoreCount`, `terminalReclaimResidentCount`, `emergencyQuarantineResidentCount`, `quarantineOverflowCount`, `minReaderEpoch`（raw 値のみ・delta は持たない） |
| Delta authority | HealthMonitor 一箇所 — 新規 `evaluateRetireChainTiers(now, prev)` のみが Δ を計算 |
| Previous-state owner | `m_prevTickSnapshot_`（HealthMonitor メンバ、tick ごと更新の rolling previous sample）。`computeTrend` は verification 専用のまま触らない |
| Tier 2 state | absolute gauge（`emergencyQuarantineResidentCount > 0`、初回 tick から評価可） |
| Tier 3 state | cumulative delta（`quarantineOverflowCount` の符号付き差分 > 0、Q+E 合算である旨をイベント本文に明記） |
| Tier 4 state | cumulative delta + episode latch（`admissionLatched`、遷移時 1 回発火） |
| Tier 5 state | resident signed-delta × 2 連続（`growthTicks`、3-case 区別: >0 加算 / ==0 リセット / <0 ドレイン証拠＋リセット） |
| Tier 4 retrigger | episode 中は再発火なし。latched 継続中は 10 秒周期の periodic evidence 再送（`kStuckEvidenceIntervalUs` 準拠・専用タイマ） |
| Tier 5 retrigger | episode 中は再 latch なし。periodic evidence は Tier 4 と同一エピソードタイマで共有 |
| Emergency latch | `admissionLatched` / `sustainedLatched`（独立セット、単一 episode exit で両方クリア） |
| Emergency exit | 構造述語: `resident == 0 ∧ Δstore == 0`（Candidate B）。保持期間 N は **未決定**（既定実装値 N=1 till 5-VI-E 根拠。Global Critical exit は既存 `CriticalExitCondition` が唯一の権威） |
| Reader correlation | verdict は `detectStuckReaders(10)` 専属。correlated（isStuck）/ suspected（readers>0 ∧ minEpoch停滞の観測記録のみ）/ uncorrelated の 3 分類。新たな tick 数閾値は追加しない |
| Event codes | 1014 `EMERGENCY_Q_ENGAGED`(W) / 1015 `QUARANTINE_OVERFLOW_DETECTED`(W) / 1016 `TERMINAL_ADMISSION`(E) / 1017 `TERMINAL_GROWTH_SUSTAINED`(E) / 1018 `EPISODE_CLEARED`(Info, optional) |
| Event payload | HealthEvent 構造体は無変更。`value`=主要カウンタ、reader 相関は既存 3 フィールド、残り 6 項目は同 tick の `[TERMINAL_EVIDENCE]` ダンプ（`getRuntimeBackpressureTelemetry()` 1 回読み） |
| RT safety | Message Thread only（tick() と同一スレッド → 全状態非アトミックで競合なし。router 読み取りは既存 atomic/mutex API 経由） |
| Diagnostics dependency | なし — OBS ブロック（`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`）から production semantics は完全分離 |
| PolicyEngine responsibility | raw observation を持たない（event/action のみ）。`computeTrend` も terminal delta を計算しない |

---

## Gates self-assessment

| Gate | Criterion | Result |
| --- | --- | --- |
| **G1** Ownership | raw observation + delta 計算の所有者が 1 箇所 | **PASS** — takeSnapshot（唯一の raw-read）+ evaluateRetireChainTiers（唯一の delta site）。Timer.cpp/diagnostics/PolicyEngine への配置禁止を明記 |
| **G2** Snapshot | 既存 takeSnapshot→computeTrend 構造を壊さず terminal telemetry 取得 | **PASS** — 追加はフィールド 5 個のみ。computeTrend は未変更（verification 専用維持）、delta は新メソッドに分離し二重計上を回避 |
| **G3** Tier separation | D/E/overflow/Terminal/stuck が混線しない | **PASS** — Tier 1 は既存 PL を無変更、Tier 2-5 は独立した述語とイベントコード、stuck verdict は detectStuckReaders 専属 |
| **G4** Latch | 発火・再発火・解除条件が完全定義 | **PASS** — episode latch + 10s periodic evidence（既存 pattern 準拠）、exit 述語 Candidate B、N は未決定として明記（勝手な N 採用なし） |
| **G5** Correlation | Terminal ≠ reader-stuck、correlation/causation 分離 | **PASS** — 3 分類（correlated/suspected/uncorrelated）、verdict は既存診断専属、新閾値なし |
| **G6** No K | 4092 / 8192 / S=2 / T_stall=30s を使わない | **PASS** — 本設計の全述語は delta/gauge ベース。peak 比較も設計から除外（5-VI-A §5 引き継ぎ） |
| **G7** No production change | 本ステップのコード変更ゼロ | **PASS** — 本ドキュメントのみ作成 |

---

## Handoff to Step 5-VI-C (minimal implementation scope)

Implementation units implied by this design (each small, test-first):

1. `TrendSnapshot` +5 fields; `takeSnapshot()` fills them (router accessors already exist).
2. `RuntimeHealthMonitor` private state: `m_prevTickSnapshot_`, `m_prevTickSnapshotValid_`,
   `m_admissionLatched_`, `m_sustainedLatched_`, `m_growthTicks_`,
   `m_lastTerminalEvidenceUs_`, `m_prevMinReaderEpoch_`.
3. New `evaluateRetireChainTiers()` called from `tick()`; four new event-code constants
   (1014-1017; 1018 optional).
4. `[TERMINAL_EVIDENCE]` dump helper (handler-side or monitor-side — implementer's choice
   within the payload decision above).
5. Explicitly out of scope: any K comparison, peak-based predicates, quarantine actions,
   changes to `computeTrend`/verification, Timer.cpp state.
