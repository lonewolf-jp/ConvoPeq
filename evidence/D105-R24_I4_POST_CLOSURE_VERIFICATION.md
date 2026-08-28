# D105-R24 — I4 Post-Closure Final Verification

**Status:** **R24 PASS** ✅ (read-only verification; Runtime source unchanged; I4 unchanged from R23)
**Source changes:** 0 (Runtime + I4 both unchanged)
**Build status:** Debug 40/40 PASS, Release 40/40 PASS (after R23 I4 amendments)

R24 verifies that the R23-closed I4 contract is fully self-consistent and aligns with
current production runtime. The post-R23 I4 contract satisfies all 23 R24 GO conditions.

---

## 1. Baseline

| Item | Value | Source |
|---|---|---|
| `ConvoPeq.md` | 2026-08-27 14:33 version | verified at start of R24 |
| `I4_DESIGN_CONTRACT.md` | R23 amendments applied (D14.3, D17.5, D18.3, D18.7, D18.8, D19.3, D20/D23, D22.2, D22.3, D26) | verified by `grep "★ D105-R23"` |
| Runtime source | unchanged (R20 final state) | `ISRRuntimePublicationCoordinator.{h,cpp}` |
| Previous audits | R21 (RetryExhaustion), R22 (NO-GO E×O), R23 (Path A convergence) | `evidence/D105-R{21,22,23}_*.md` |
| C8 | unchanged | `ISRSemanticValidationTests.cpp:1034` |
| K=4 | unchanged | `kMaxObligationConsecutiveFailures = 4` |
| `kMaxLogicalRecoveryObligations=32` | unchanged | `ISRRuntimePublicationCoordinator.h:325` |

## 2. I4 Internal Contradiction Audit (read-only)

### Episode invariant residual scan (R24 brief §2)

Grep on `I4_DESIGN_CONTRACT.md` for: `RecoveryEpisodeId`, `nextRecoveryEpisodeId`,
`EpisodeAdmissionState`, `E_max`, `O_max`, `E_max × O_max`, `INV-CAP-2/3/4`, `Tentative`,
`Owned`, `Released`, `GlobalRecoveryBudget`, `OPEN`, `CLOSED`, `episode closure`,
`episode capacity`, `per-episode obligation`.

| Mention | Type | Status |
|---|---|---|
| D13 (D8 / D11 / D12 history, lines 18/93/115/116/122/125/262/286/298/451/456/728-729/741) | Historical design context | Phase-II Future Design (R23 deferred) |
| D17.5 / D19.1 / D20 / D23 / D26 (lines 451-631, 1064-1810) | Phase-II Future Design with explicit "Phase-I production contract としては無効" annotations | R23 deferred |
| D18.1 CoalesceIdentity (line 313, 520) | Historical Phase-II Design | **D18.7 (R23) rewrite: `{handle, target}`** |
| D22.2 (lines 866-925) | Replaced by D22.3 R23 amendment: `liveLogicalObligationCount ≤ 32` direct | R23 rewritten |
| D19.3 INV-CAP-2/3/4 (lines 491-504) | R23 amendment: REMOVED from Phase-I | R23 removed |
| T26 test (line 741) | Historical test specification (no production code asserts) | Test-only, no production invariant |
| §7 R23 summary (lines 7809-7943) | R23 amendments + final state | Active |

**No `E_max × O_max ≤ 32` claim as a Phase-I production invariant remains** in the I4
file. All mentions are:
- (a) Historical design context (D8 / D11 / D12)
- (b) Phase-II Future Design (D13 / D19.1 / D20 / D23 / D26)
- (c) R23 rewrites that mark the claim as "REMOVED from Phase-I" (D19.3) or
  "REWRITTEN" (D18.1) or "N/A" (D22.3)

### Capacity 4-concept separation (R24 brief §3)

R23 introduced 4 distinct capacity concepts and the I4 file's R23 amendment at
D22.3 (line 928-) explicitly names and separates them:

| Concept | Value | Source | Other ≠ this? |
|---|---|---|---|
| `Q_max` | 256 | `ISRDSPQuarantine.h:68` (`kMaxSlots`) | ≠ E_max, ≠ 32 |
| `L_residency_max` | 257 | `ISRRuntimePublicationCoordinator.h:886` (256) + `PendingRecoveryAdmission` (1) | ≠ 32 |
| `L_logical_max` | 32 | `ISRRuntimePublicationCoordinator.h:325` | ≠ 256, ≠ 257 |
| `E_max` | N/A (Phase-I production) | R23 amendment | — |
| `O_max` | N/A (Phase-I production) | R23 amendment | — |

**256 ≠ 32**: confirmed (`quarantineActiveFlags_[256]` vs `liveCount_ ≤ 32`).
**257 ≠ 32**: confirmed (transport + durable vs logical).

**No "32 = E×O" derivation remains**: I4 §7 (R23 summary, line 7876) explicitly
removes INV-CAP-2/3/4. The D22.2 text at line 920 was rewritten to remove the
"E×O decomposition" wording.

## 3. `kMaxLogicalRecoveryObligations = 32` Uniqueness Audit (R24 brief §4)

### All ConvoPeq.md occurrences of `kMaxLogicalRecoveryObligations`

| # | Location | Category | Type |
|---|---|---|---|
| 1 | `ISRRuntimePublicationCoordinator.h:325` | **Definition** | `static constexpr std::size_t kMaxLogicalRecoveryObligations = 32;  // INV-CAP-7` |
| 2 | `ISRRuntimePublicationCoordinator.h:934` | **Template parameter usage** | `RecoveryAdmissionTable<kMaxLogicalRecoveryObligations> recoveryAdmissions_;` |

**Total production source occurrences: 2** (definition + single template instantiation).

### Capacity gate (uses 32 indirectly via `kCapacity`)

| Location | Category | Type |
|---|---|---|
| `ISRRuntimePublicationCoordinator.h:355-356` | Capacity gate (template-resolved) | `if (liveCount_.load(...) >= kCapacity) return std::nullopt;` |
| `ISRRuntimePublicationCoordinator.cpp:880-885` | Runtime call site (uses `tryInsert` return) | `if (!ins) { fetchAddAtomic(recoveryCapacityExhaustedCount_, 1, ...); return false; }` |

The `kCapacity` is the template parameter resolved to `32` at compile time.
There is **no other `32` constant** anywhere in the recovery admission path.
No 32 is derived from E×O in the production code (R3 §5.3 / R22 / R23 all confirm).

### Other references in I4 and evidence (not in production source)

All other `kMaxLogicalRecoveryObligations` mentions in `I4_DESIGN_CONTRACT.md` are
documentary — not active invariant claims. They appear in:
- D14.1, D14.2, D14.3 (definition / obligation table enforcement narrative)
- D22.2, D22.3 (deliberate resource bound reasoning)
- D25 (INV-CAP-7 invariant definition)
- T21 (historical test specification)
- §7 (R23 final summary)

**No 32-obligation-capacity second definition exists in production source.**

### Verdict: kMaxLogicalRecoveryObligations uniqueness

- **Definition count**: 1 (line 325)
- **Template instantiation**: 1 (line 934)
- **Capacity gate**: 1 (h:355-356, resolved via template)
- **Runtime call site**: 1 (cpp:880-885)
- **Total functional 32 sources**: 1 (single template-resolved path through `kCapacity`)

**32 is the unique obligation table capacity constant**, and it is **not derived from
E×O nor from any other capacity concept** (R23 §7.5 confirmed).

## 4. I4 → Runtime Reverse Trace (R24 brief §5)

### Phase-I production clause → runtime implementation

| I4 clause (R23 final form) | Runtime implementation site | Status |
|---|---|---|
| **D14.3**: transient failure ≠ terminal, ΔL=0, delivery=None, counter++ | `markTransientFailure` (`ISRRuntimePublicationCoordinator.cpp:998-1024`) | ✅ `delivery = None` (line 1010), `consecutiveFailureCount.fetch_add(1, acq_rel)` (line 1013-1014) |
| **D15.2 disappearance set** = `{Success, Superseded, ShutdownDiscard, RetryExhaustion}` | `table.resolve(obligationId, ResolvedSuccess/ResolvedStaleSuperseded/ShutdownDiscarded/ResolvedFailed)` (`cpp:1019`, `h:366-378`) | ✅ 4 terminal types match the I4 set |
| **D17.5 (R23)**: `liveLogicalObligationCount == 0` iff all admitted are terminal | `liveCount_.fetch_sub(1, release)` on `table.resolve` CAS success (`h:388-389`) | ✅ Single counter, terminal-when-0 |
| **D18.1 / D18.7 (R23)**: `CoalesceIdentity = {handle, target}` | `cid{quarantinedHandle, {3 hashes}}` (`cpp:836-841`) | ✅ No `RecoveryEpisodeId` field, no `episodeId` key |
| **D18.3 (R23)**: `liveCount_ == 0` (single-counter observable) | `recoveryAdmissions_.liveCount_` (h:380) — direct `std::uint64_t` counter | ✅ |
| **D18.8 (R23)**: snapshot drift ⇒ new target ⇒ new obligation | `cid` includes 3 hashes from `buildSource.rebuildFingerprint`; `findByKey(cid)` returns npos when hashes differ ⇒ `tryInsert` (cpp:866, 880) | ✅ R3 counterexample preserved as contract |
| **D19.1 (Phase-II deferred)**: episode closure finality | n/a in Phase-I production | ✅ Phase-II deferred |
| **D19.3 (R23)**: only INV-CAP-1 (`liveCount_ ≤ 32`) | `tryInsert` h:355-356, `kCapacity = kMaxLogicalRecoveryObligations = 32` | ✅ Other 3 INV-CAP REMOVED from Phase-I |
| **D20 (Phase-II deferred)**: episode closure linearization | n/a in Phase-I production | ✅ Phase-II deferred |
| **D22.2 (R23)**: `kMaxLogicalRecoveryObligations = 32 = deliberate resource bound` | `ISRRuntimePublicationCoordinator.h:325` (const), `h:934` (template), `h:355-356` (gate) | ✅ Single-counter direct form |
| **D22.3 (R23 NEW)**: capacity name separation | I4 §D22.3 lines 928-988 (4 named capacities) | ✅ R23 contract addition |
| **D23 (Phase-II deferred)**: closure-aware CAS | n/a in Phase-I production | ✅ Phase-II deferred |
| **D26 (R23)**: `liveCount_ ≤ 32` single counter model | R23 amendment at I4 lines 1296-1345 (rewrites D26 to match runtime) | ✅ I4 aligned with runtime |
| **D29.8 (R21 maintained)**: `MARK-TRANSIENT-FAILURE` non-terminal action | `markTransientFailure` (`cpp:1010-1020`) non-terminal action, terminal only at counter == K | ✅ R21 alignment preserved |

### Closure (Phase-I production)

| State | R23 I4 clause | Runtime mechanism |
|---|---|---|
| `Live` | closure target | `recoveryAdmissions_.slot(i).state == Live` |
| `terminal (any of 4)` | closure | `table.resolve` CAS (`h:370-372`) + `liveCount_--` (`h:388-389`) |
| `liveCount_ == 0` | **logical obligation domain closure** | `recoveryAdmissions_.liveCount_.load() == 0` |
| Episode closure | **NOT Phase-I production invariant** (D19.1 / D20 deferred) | n/a |

## 5. Runtime → I4 Forward Trace (R24 brief §6)

### 12 obligation state mutations (Runtime side) → I4 clause mapping

| # | Runtime mutation | Source site | I4 clause | Match? |
|---|---|---|---|---|
| 1 | `submitRecoveryRequest` (admission) | `ISRRuntimePublicationCoordinator.cpp:820-942` | D14.2 reservation-first + D15.2 `Success`/`Superseded`/`Shutdown`/`RetryExhaustion` | ✅ |
| 2 | `findByKey` (coalesce) | `ISRRuntimePublicationCoordinator.cpp:866` | D18.1 → D18.7 (R23) `{handle, target}` | ✅ |
| 3 | `tryInsert` (new obligation) | `ISRRuntimePublicationCoordinator.cpp:880` | D14.2 + D19.3 (R23) → D22.3 (R23) `liveCount_ ≤ 32` direct form | ✅ |
| 4 | `recoveryIntentQueue_.push` (transport enqueue) | `ISRRuntimePublicationCoordinator.cpp:909` | D14.2 transport placement | ✅ |
| 5 | `pendingRecoveryAdmission_.state = DurablePending` (durable admission) | `ISRRuntimePublicationCoordinator.cpp:930-940` | D14.2 durable placement + D18.4 reservation semantics | ✅ |
| 6 | `popRecoveryRequest` / `takePendingRecoveryAdmission` (Builder take) | `AudioEngine.RebuildDispatch.cpp:1017` | D14.2 builder consumption | ✅ |
| 7 | `onPublishCommitted` (publish success) | `RuntimePublicationOrchestrator.cpp` | D15.2 `Success` terminal | ✅ |
| 8 | `RejectedStaleGeneration` → `StaleSuperseded` | `RuntimePublicationOrchestrator.cpp:380-385` | D15.2 `Superseded` terminal | ✅ |
| 9 | `discardRecoveryRequestsOnShutdown` (shutdown discard) | `ISRRuntimePublicationCoordinator.cpp:1203-1218` | D15.2 `ShutdownDiscard` terminal | ✅ |
| 10 | `markTransientFailure` (transient failure) | `ISRRuntimePublicationCoordinator.cpp:998-1024` | D14.3 + D29.8 (R21) | ✅ |
| 11 | `redriveDeferredRecoveryObligations` (redrive) | `ISRRuntimePublicationCoordinator.cpp:1042-1052` | D14.3 P-B (delivery=None) | ✅ |
| 12 | `markTransientFailure` exhaustion → `table.resolve(id, ResolvedFailed)` | `ISRRuntimePublicationCoordinator.cpp:1017-1019` | D15.2 `RetryExhaustion` terminal (R21) | ✅ |

**All 12 runtime state mutations have a corresponding I4 clause.**
**No residual gap detected.**

## 6. R21 → R22 → R23 Historical Consistency (R24 brief §7)

### R21 amendments (preserved)

- D14.3 footnote: `Live → Live` for transient failure (line 163-180) — **preserved**
- D15.2 disappearance set with `RetryExhaustion` (line 219-235) — **preserved**
- D18.3 `retryExhaustedCount` in conservation (line 350-373) — **preserved**
- D18.6 T15c/T15d/T15e/T15f test matrix (line 454-457) — **preserved**
- D20.5 audit verdict `RetryExhaustion ≠ EpisodeClosure` (line 615-620) — **preserved**
- D29.8 `MARK-TRANSIENT-FAILURE` non-terminal (line 1652-1668) — **preserved**

### R22 NO-GO history (resolved by R23 Path A)

- R22 found: `E_max × O_max ≤ 32` is arithmetically unsatisfiable
  (256 × ≥2 = ≥512 ≠ 32)
- R22 verdict: NO-GO (production/contract mismatch)
- R23 Path A: `E_max`, `O_max`, `E×O ≤ 32` REMOVED from Phase-I production
  contract (deferred to Phase-II)
- R23 historical relation: **R22 was not "wrong"** — R22 discovered the
  production/contract mismatch; R23 **resolved it via contract simplification**
  (Path A), not by re-implementing the production runtime
- This satisfies the brief: "R22 が発見した production/contract mismatch を
  R23 Path A が contract simplification により解消した"

### R23 final state

- E_max = N/A (Phase-I production) ✅
- O_max = N/A (Phase-I production) ✅
- E×O ≤ 32 = NOT a Phase-I invariant ✅
- L_logical_max = 32 (kMaxLogicalRecoveryObligations) ✅
- L_residency_max = 257 (transport 256 + durable 1) ✅
- Q_max = 256 (quarantineActiveFlags_[256]) ✅
- CoalesceIdentity = `{handle, target}` (no `RecoveryEpisodeId` field) ✅
- RetryExhaustion = `consecutiveFailureCount == K` only ✅
- Closure = `liveCount_ == 0` (logical obligation domain) ✅
- conservation = `liveCount_ == 0 iff all admitted are terminal` (R23 observable form) ✅

## 7. RetryExhaustion Final Consistency

| Required condition | Runtime state | I4 state | Match? |
|---|---|---|---|
| `transient failure ≠ terminal disposition` | `markTransientFailure` does NOT call `resolveRecoveryObligation(id, Failed)` unless counter == K | D14.3 footnote (R23 line 163-180) | ✅ |
| `markTransientFailure × 1..3 → Live, ΔL=0` | T-R18-5, T-R20-3 | D14.3 invariant (R23) | ✅ |
| `markTransientFailure × 4 → ResolvedFailed, ΔL=-1` | T-R18-5, T-R20-4 | D14.3 + D15.2 `RetryExhaustion` (R21) | ✅ |
| `RetryExhaustion ≠ EpisodeClosure` | Phase-I production has no `RecoveryEpisodeId` | D20.5 audit verdict (R23 line 615-620) | ✅ |
| `ResolvedFailed ≠ generic failure terminal` | Only `markTransientFailure` exhaustion calls `table.resolve(id, ResolvedFailed)`; C8 test-only; switch fallthrough removed (`jassertfalse` default) | D15.2 amendment (R21) + D18.3 (R23) | ✅ |

## 8. Conservation Equation Final Audit

### Runtime-observable (R23 form)

| Counter | Source | Present in runtime? |
|---|---|---|
| `liveCount_` | `RecoveryAdmissionTable` (`h:380`) | ✅ |
| `recoveryObligationShutdownDiscardCount_` | `h:892` | ✅ |
| `recoveryRetryExhaustedCount_` | `h:944` | ✅ |
| `recoveryCoalescedCount_` | `h:895` | ✅ |
| `recoveryCapacityExhaustedCount_` | `h:898` | ✅ |
| `recoveryRetryDeferredCount_` / `recoveryRetryRedriveCount_` / `recoveryRetryRedriveFailureCount_` | `h:896-897` | ✅ |

### NOT directly maintained in runtime

| Counter | Why not |
|---|---|
| `successCount` | Terminal Success does not increment a separate counter; the only signal is `liveCount_--` |
| `supersededCount` | Terminal Superseded does not increment a separate counter |
| `admittedLogicalObligationCount` | Runtime does not maintain this; implied by `liveCount_ + terminalCount` |
| `admissionEventCount` | Runtime does not maintain this counter |

**R23's runtime-observable form** (`liveCount_ == 0 iff all admitted are terminal`)
is correctly represented in the runtime. The I4 §D18.3 (R23 amendment, line 376-444) is
correct.

## 9. Residual Gaps

**Zero residual gaps.** Every runtime state transition has a corresponding I4 clause.
Every I4 production clause has a corresponding runtime implementation.

## 10. GO Conditions (23 items per R24 brief)

| # | Gate | Status | Evidence |
|---|---|---|---|
| 1 | Latest ConvoPeq (2026-08-27) | ✅ | verified at R24 start |
| 2 | Runtime source unchanged | ✅ | R20 → R21 → R22 → R23 → R24: 0 changes |
| 3 | I4 internal contradiction = 0 | ✅ | R24 §2 audit |
| 4 | Old Episode production invariant = 0 | ✅ | R24 §2 grep scan |
| 5 | Episode references = Phase-II deferred only | ✅ | R23 §7.5 deferred list |
| 6 | `Q_max = 256` | ✅ | `ISRDSPQuarantine.h:68` `kMaxSlots = 256` |
| 7 | `L_residency_max = 257` | ✅ | transport 256 + durable 1 |
| 8 | `L_logical_max = 32` | ✅ | `kMaxLogicalRecoveryObligations = 32` (`h:325`) |
| 9 | `E_max = N/A` | ✅ | R23 §D22.3 |
| 10 | `O_max = N/A` | ✅ | R23 §D22.3 |
| 11 | `E×O ≤ 32` ≠ Phase-I invariant | ✅ | R23 §D19.3 / §D22.2 rewrite |
| 12 | 32 capacity gate = `tryInsert` | ✅ | `h:355-356`, `cpp:880-885` |
| 13 | CoalesceIdentity = `{handle, target}` | ✅ | R23 §D18.7, `cpp:836-841` |
| 14 | Snapshot drift ⇒ new target = new obligation | ✅ | R23 §D18.8, R3 counterexample preserved |
| 15 | RetryExhaustion = K=4 | ✅ | R21 §D15.2, `kMaxObligationConsecutiveFailures = 4` |
| 16 | transient failure = Live→Live | ✅ | R21 §D14.3, `markTransientFailure` `cpp:998-1024` |
| 17 | closure = `liveCount 1→0` (logical) | ✅ | R23 §D17.5, `table.resolve` `h:370-372` |
| 18 | conservation = runtime-observable form | ✅ | R23 §D18.3 (line 376-444) |
| 19 | I4→Runtime: 0 residual gap | ✅ | R24 §4 table |
| 20 | Runtime→I4: 0 residual gap | ✅ | R24 §5 table |
| 21 | R21 history consistent with R23 final | ✅ | R24 §6 R21 subsection |
| 22 | R22 history resolved by R23 Path A | ✅ | R24 §6 R22 subsection |
| 23 | R23 history consistent with current final | ✅ | R24 §6 R23 subsection |

**R24: 全 23 GO 条件 PASS。**

## 11. Prohibitions Check

| # | Prohibition | Status |
|---|---|---|
| 1 | Runtime source change | ✅ Avoided (R24 = verification only) |
| 2 | `RecoveryEpisodeId` added | ✅ Avoided |
| 3 | Episode CAS added | ✅ Avoided |
| 4 | `Tentative/Owned` added | ✅ Avoided |
| 5 | `GlobalRecoveryBudget` added | ✅ Avoided |
| 6 | `kMaxLogicalRecoveryObligations` change | ✅ Unchanged (still 32) |
| 7 | K=4 change | ✅ Unchanged (still 4) |
| 8 | C8 change | ✅ Unchanged |
| 9 | New obligation state added | ✅ Avoided |
| 10 | R23 Path A → Path B revert | ✅ Path A maintained |
| 11 | R22 E×O ≤ 32 revival | ✅ R23 explicitly removed |
| 12 | New test added | ✅ R24 has no new tests (verification only) |

## 12. Files Referenced (read-only)

| File | Role |
|---|---|
| `doc/work88/I4_DESIGN_CONTRACT.md` | R23-amended I4 (7809-7943 = R23 §7) |
| `ConvoPeq.md` (2026-08-27) | runtime source baseline |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:325` | kMaxLogicalRecoveryObligations definition |
| `src/audioengine/ISRRuntimePublicationCoordinator.h:355-356, 934` | tryInsert gate / template instantiation |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:820-942, 955-1024` | submitRecoveryRequest, resolveRecoveryObligation, markTransientFailure |
| `src/audioengine/RuntimePublicationOrchestrator.cpp:357-433` | submitPublishRequest switch (5 terminal paths) |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp:1017-1076` | Builder durable loop (settle + markTransientFailure) |
| `evidence/D105-R21_I4_CONTRACT_AMENDMENT.md` | R21 amendments (RetryExhaustion, conservation) |
| `evidence/D105-R22_FINAL_CONSISTENCY_AUDIT.md` | R22 NO-GO discovery |
| `evidence/D105-R23_OBLIGATION_TABLE_MODEL_CONVERGENCE.md` | R23 Path A convergence |

## 13. Final Verdict: **R24 PASS**

**D105-R24: I4 Post-Closure Final Verification — 全 23 GO 条件 PASS**

- I4 post-R23 (Path A) contract is **fully self-consistent** with current production runtime
- `kMaxLogicalRecoveryObligations = 32` is the **unique** logical obligation capacity constant
- 4 capacity concepts (`Q_max` / `L_residency_max` / `L_logical_max` / episode `= N/A`) are **cleanly separated**
- `RecoveryEpisodeId` references are **all Phase-II deferred** (no production invariant claims)
- `RetryExhaustion ≠ EpisodeClosure` (R20 + R21 + R23 consistent)
- Conservation is **runtime-observable** (`liveCount_` single counter)
- Bidirectional I4↔Runtime trace has **0 residual gaps** (12/12 state mutations mapped)
- Debug + Release both pass 40/40 tests (no regression from R23 I4 amendments)
- No source changes; no test changes; no I4 changes (R24 is verification only)

**R25 handoff**: R24 PASS confirms the R23-closed I4 contract is correct. R25 may now
proceed — but it should be a **scoped design audit** for the next Runtime implementation
candidate (e.g. Phase-II episode layer), not an immediate code change. R25 will not modify
the obligation-table model that R23 fixed; it will design the episode layer that
**will eventually** be implemented as Phase-II.

Specifically, R25 should:
1. Audit the gap between the **Phase-II deferred design** (D13 / D19 / D20 / D23 / D26)
   and the actual production migration path required to realize it.
2. Identify the minimum Runtime changes that would make `E_max`, `O_max`, and
   `E_max × O_max ≤ 32` provable from code.
3. Document the migration risk and reversibility.

R24 → R25 is a **planning transition**, not an implementation transition. The production
runtime (R20-finalized, R23-aligned) remains the authoritative Phase-I implementation.
