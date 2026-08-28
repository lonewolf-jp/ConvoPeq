# D105-R23 — I4 Contract Closure / Obligation-Table Model Convergence

**Status:** **R23 PASS** ✅ (read-only + I4 amendments; Runtime source unchanged).
**Path A adopted**: episode abstraction deferred to Phase-II; I4 converges to the
**obligation-table model** that matches current production runtime. All 20 GO conditions
satisfied. R24 may now begin (contract-only final audit).

---

## R23-1 — Pre-edit I4 audit (read-only, before R23 changes)

R2 / R3 / R22 audits established these facts from current production runtime:

| Fact | Value | Source | Status |
|---|---|---|---|
| `RecoveryEpisodeId` (type) | 0 production hits | grep | design-only |
| `nextRecoveryEpisodeId_` (counter) | 0 production hits | grep | design-only |
| `EpisodeAdmissionState` (OPEN/CLOSED) | 0 production hits | grep | design-only |
| `recoveryClosedEpisodeRejectCount_` (telemetry) | 0 production hits | grep | design-only |
| `CoalesceIdentity` (struct) | `{handle, SemanticRecoveryTarget}` | ISRRuntimePublicationCoordinator.h:262-267 | R5-9 実装 |
| `Tentative` / `Owned` / `Released` (states) | 0 production hits | grep | design-only |
| `GlobalRecoveryBudget` API (Tentative/Owned) | 0 production hits | grep | design-only |
| `liveCount_` (single counter) | enforced `≤ 32` | ISRRuntimePublicationCoordinator.h:355-356 | R5-8 実装 |
| `Q_max` (quarantineActiveFlags_[256]) | 256 | ISRDSPQuarantine.h:68 | R2 確認 |
| `L_residency_max` (transport + durable) | 257 | ISRRuntimePublicationCoordinator.h:886 + PendingRecoveryAdmission | R2 確認 |
| `E_max` | UNDEFINED (no episode abstraction) | R2/R3 確認 | cannot derive |
| `O_max` | ≥ 2 (R3 counterexample still applies) | R3 §7.3 確認 | unbounded |
| `E×O ≤ 32` | arithmetically unsatisfiable | 256 × ≥2 = ≥512 ≠ 32 | R22 NO-GO |

## R23-2 — Classification of episode clauses

| I4 clause | Status under R23 | Reason |
|---|---|---|
| D13 (`RecoveryEpisodeId` 7-domain) | **Phase-II deferred** | 0 production hits |
| D13.1 / D13.2 (epoch ≠ `RecoveryEpisodeId`) | **Phase-II deferred** | no episode in code |
| D15.2 disappearance set (with `RetryExhaustion`) | **MAINTAINED** | R21 confirmed runtime consistency |
| D17.5 closure (`liveLogicalObligationCount == 0`) | **REWRITTEN** | clarify "logical obligation domain" not "episode" |
| D18.1 `CoalesceIdentity` (`{handle, episodeId, target}`) | **REWRITTEN** as D18.7 | match R5-9 implementation (`{handle, target}`) |
| D18.2 semantic supersession (equality) | **MAINTAINED** | runtime consistent |
| D18.3 ownership conservation | **REWRITTEN** | runtime-observable form only |
| D18.4 reservation semantics | **MAINTAINED** | R5-9 implements |
| D18.5 D13-D17 revisions | **MAINTAINED + annotations** | add Phase-II deferred notes |
| D18.6 test matrix | **MAINTAINED** (T15c/T15d/T15e/T15f preserved) | R21 consistent |
| D19.1 episode closure finality | **Phase-II deferred** | `RecoveryEpisodeId` 0 hits |
| D19.3 INV-CAP-1..4 | **INV-CAP-1 MAINTAINED, INV-CAP-2/3/4 REMOVED** | E/O unprovable |
| D20 episode closure linearization | **Phase-II deferred** | no `Closed` atomic in code |
| D21 backpressure liveness | **MAINTAINED** | runtime-observable |
| D22 capacity proof | **REWRITTEN** | `liveCount_ ≤ 32` direct form |
| D23 closure-aware CAS | **Phase-II deferred** | no `EpisodeAdmissionState` in code |
| D26 Tentative/Owned (capacity model) | **REWRITTEN** | `liveCount_` single counter model |
| D29.8 end-to-end state machine (with `MARK-TRANSIENT-FAILURE`) | **MAINTAINED** | R21 confirmed runtime consistency |

## R23-3 — D19.3 / D22.2 revision

**Before R23**: D19.3 stated `E_max × O_max ≤ kMaxLogicalRecoveryObligations` with INV-CAP-2/3/4
claims. D22.2 stated `Σ episode live obligations ≤ 32` with `E_max × O_max` decomposition.

**After R23 (D105-R23 amendment)**:

```
D19.3 (★ D105-R23 — Path A 適用):
    Episode-decomposition を Phase-I production invariant から除外

    kMaxLogicalRecoveryObligations = 32
    INV-CAP-1: liveLogicalObligationCount ≤ kMaxLogicalRecoveryObligations
               (直接 enforcement: RecoveryAdmissionTable::tryInsert の liveCount_ < kCapacity ゲート)

    INV-CAP-2: REMOVED (RecoveryEpisodeId 未実装)
    INV-CAP-3: REMOVED (episode 概念未実装、O_max 不定)
    INV-CAP-4: REMOVED (E×O≤32 arithmetically unsatisfiable)
```

```
D22.2 (★ D105-R23 改訂):
    kMaxLogicalRecoveryObligations = 32 = 【deliberate resource bound】
    admission authority が直接 enforcement:
        liveLogicalObligationCount ≤ 32
      (obligation table 内の single counter)
    per-episode 配分 / Σ episode bound は Phase-II に deferred
```

## R23-4 — Capacity name separation (D22.3 new)

```
Physical quarantine capacity:  Q_max = 256
Physical recovery residency:   L_residency_max = 257 (= 256 + 1)
Logical recovery obligation:   L_logical_max = 32
Episode capacity:              E_max = N/A
                               O_max = N/A

Q_max ≠ E_max
L_residency_max ≠ L_logical_max
E_max × O_max ≤ 32:  NOT A PHASE-I INVARIANT
```

## R23-5 — D18.1 / D18.7 / D18.8 (CoalesceIdentity + snapshot drift)

```
D18.7 (★ D105-R23 — Phase-I production CoalesceIdentity):
    CoalesceIdentity = { quarantinedHandle, SemanticRecoveryTarget }
                      (RecoveryEpisodeId 成分なし — Phase-II deferred)

D18.8 (★ D105-R23 — Snapshot drift contract):
    同一 handle H の episode lifetime 中に、
    `currentBuildSnapshot_` の値（→ buildSource.rebuildFingerprint.{3 hashes}）が変化し得る。

    snapshot drift による結果:
    - 同一 handle H + target_T1 → obligation_O1
    - 同一 handle H + target_T2 (≠ T1) → obligation_O2 (NEW, ΔL=+1)
    - 結果: 同一 handle H に紐づく Live obligation 数は 1 + (snapshot drift 数)

    O_max = 1 不成立 (Phase-I production では)
    E_max × O_max ≤ 32 不成立 (arithmetic)
```

## R23-6 — D20 / D23 closure deferred to Phase-II

```
D20/D23 deferred: Episode closure 仕様は Production fact として扱わない

Phase-I production closure:  liveLogicalObligationCount == 0 が全 Live logical obligation 消滅と等価
                              (RetryExhaustion ≠ EpisodeClosure は D20.5 audit verdict 維持)
```

## R23-7 — D18.3 conservation (runtime-observable form)

```
D18.3 (★ D105-R23 — Phase-I production observable form):
    RecoveryAdmissionTable.liveCount_ == 0
        iff
    for every admitted obligation: terminal transition has occurred

Runtime-observable counters:
    - liveCount_ (current Live count)
    - recoveryObligationShutdownDiscardCount_ (cumulative)
    - recoveryRetryExhaustedCount_ (cumulative)
    - recoveryCoalescedCount_ (cumulative)
    - recoveryCapacityExhaustedCount_ (cumulative)
    - recoveryRetryDeferredCount_, recoveryRetryRedriveCount_, recoveryRetryRedriveFailureCount_

NOT runtime-observable in Phase-I (reconstructed from derived values only):
    - successCount, supersededCount (no separate counter)
    - admittedLogicalObligationCount (not separately maintained)
    - admissionEventCount (not separately maintained)
```

## R23-8 / R23-9 — R21 amendments preserved

```
D14.3 transient failure: Live→Live 維持
D15.2 RetryExhaustion in disappearance set 維持
D18.3 retryExhaustedCount in terminalDispositionCount 維持
D18.6 T15c/T15d/T15e/T15f 維持
D20.5 RetryExhaustion ≠ EpisodeClosure 維持
D29.8 MARK-TRANSIENT-FAILURE 維持
```

## R23-11 — D26 capacity model alignment

```
D26 (★ D105-R23 改訂):
    Phase-I production の capacity model:
    - Tentative / Owned / Released 状態 → 実装なし
    - EpisodeAdmissionState (OPEN/CLOSED) → 実装なし
    - GlobalRecoveryBudget API → 実装なし
    - 単一 counter liveCount_ ≤ 32 のみが enforce される

    Tentative + Owned ≤ 32 → Phase-II deferred
    D26.6 resource adequacy analysis → Phase-II deferred
    T35/T36/T37 → Phase-II deferred
```

## R23-12 — GO conditions verification

| # | Condition | Status | Evidence |
|---|---|---|---|
| 1 | Runtime source unchanged | ✅ | R23 audit 期間中の source 変更: 0 |
| 2 | `RecoveryEpisodeId` Phase-I contract から除外 | ✅ | I4 §7.5 (R23) |
| 3 | `E_max` N/A として明示 | ✅ | I4 §D22.3 (R23) |
| 4 | `O_max` N/A として明示 | ✅ | I4 §D22.3 (R23) |
| 5 | `E×O ≤ 32` production invariant から除去 | ✅ | I4 §D19.3 / §D22.2 (R23) |
| 6 | `Q_max` = 256 として維持 | ✅ | I4 §D22.3 |
| 7 | physical residency = 257 として明示 | ✅ | I4 §D22.3 |
| 8 | logical capacity = 32 として明示 | ✅ | I4 §D22.3 |
| 9 | admission enforcement: `tryInsert` の 32 gate と対応 | ✅ | I4 §D22.3 |
| 10 | CoalesceIdentity: `{handle, target}` に一致 | ✅ | I4 §D18.7 (R23) |
| 11 | snapshot drift: new target = new obligation として明記 | ✅ | I4 §D18.8 (R23) |
| 12 | RetryExhaustion: R21 semantics 維持 | ✅ | I4 §D15.2 (R21) |
| 13 | transient failure: Live→Live 維持 | ✅ | I4 §D14.3 (R21) |
| 14 | closure: `liveCount 1→0` を logical domain に限定 | ✅ | I4 §D17.5 (R23) |
| 15 | D20/D23: Episode closure 仕様を Production fact として扱わない | ✅ | I4 §D20/D23 (R23) |
| 16 | D26: Runtime の actual capacity model と一致 | ✅ | I4 §D26 (R23) |
| 17 | conservation: Runtime で証明可能な形に限定 | ✅ | I4 §D18.3 (R23) |
| 18 | bidirectional trace: I4→Runtime / Runtime→I4 とも全項目対応 | ✅ | R22 確認 + R23 改訂 |
| 19 | unsupported claims: 0 件 | ✅ | E_max / O_max / E×O / episode abstraction を全部 deferred |
| 20 | I4 internal contradiction: 0 件 | ✅ | R21 + R23 改訂で contract 自己整合 |

**R23: 全 20 GO 条件 PASS。**

## Files changed (R23)

| File | Change |
|---|---|
| `doc/work88/I4_DESIGN_CONTRACT.md` | 7 amendments added (D14.3, D17.5, D18.3, D18.7, D18.8, D19.3, D20/D23, D22.2, D22.3 NEW, D26) + final summary section 7 (R23) |

**No runtime source changes.** Runtime source unchanged throughout R23.

## R23 prohibitions check

| # | Prohibition | Status |
|---|---|---|
| 1 | `RecoveryEpisodeId` を runtime に追加 | ✅ Avoided (R23 = Path A) |
| 2 | `Tentative`/`Owned` state を runtime に追加 | ✅ Avoided |
| 3 | `GlobalRecoveryBudget` を runtime に追加 | ✅ Avoided |
| 4 | Runtime source 変更 | ✅ Avoided (R23 = I4 only) |
| 5 | C8 変更 | ✅ Avoided |
| 6 | K=4 変更 | ✅ Avoided (R23 改訂は K に触れない) |
| 7 | `kMaxLogicalRecoveryObligations = 32` 変更 | ✅ Avoided (32 = deliberate bound 維持) |
| 8 | I4 internal contradiction | ✅ Avoided (R23 改訂で contract 自己整合) |
| 9 | Production で `ResolvedFailed` を generic failure terminal に | ✅ Avoided (R21 維持) |
| 10 | unsupported claim を production contract に残す | ✅ Avoided (全部 deferred) |

## R23 verdict: **PASS**

**D105-R23: I4 contract closure 達成**

- ✅ Runtime source unchanged
- ✅ I4 contract self-consistent (R21 amendments + R23 rewrites)
- ✅ All 20 GO conditions PASS
- ✅ All 10 prohibitions satisfied
- ✅ Path A selected (Episode abstraction deferred to Phase-II)
- ✅ Bidirectional I4↔Runtime trace complete (R22 + R23)

**R24 (I4 post-closure verification) may begin**: it should verify I4 全文 internal contradiction
= 0, latest `ConvoPeq.md` との最終 reverse/forward trace, `grep` による旧 Episode invariant 残存
チェック, `kMaxLogicalRecoveryObligations=32` の唯一性確認, R21/R22 の旧結論が R23 後に
矛盾なく整理されているかの確認。

After R24 PASS, R25 以降の Runtime 実装作業（episode 層の本格実装等）が許可される。
