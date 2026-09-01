# D135-8/9 Gate E — Lifecycle / Ownership / DSP Retirement Regression Audit

**Status:** PASS (E-1 .. E-7)
**Date:** 2026-08-30
**Scope:** read-only. `production source 0 / test source 0 (audit edits only) / build 0 / CTest 0 / doc 0`. Evidence only.
**Audit base:** F6 実装済 working tree (HEAD `5f6f48c` = F6 commit) + 再生成 `ConvoPeg.md` (`Generated: 2026-08-30 19:19:29`). Git clean.
**F6 diff provenience:** `git diff a65ace1..5f6f48c` on the 5 files = **283 insertions / 11 deletions**, and `git show 5f6f48c` confirms F6's *only* lifecycle-term additions are 2 `retireDSPHandleForRuntime` rows (see E-6). Old-snapshot (stale ConvoPeg.md) は排除 — 全 cite は working-tree `src/` + committed-blob 比較。

> Gate D は retry/retention account を検証した。Gate E は **F6 が既存 lifecycle/ownership/retire 契約に副作用を起こしていないか**を回帰検証する。F6 は *deferred accounting / metadata / wake / watchdog / ownership reset* に限定されており、publish→activate→retire 実体 (DSPTransition / DSPLifetimeManager / ISRRetireRouter / CrossfadeAuthority) は改変しない — 本稿で行番号で証明。

---

## E-1. Publish → Activate → Retire call chain — PASS (regression check)

```
submitPublishRequest(req)  RuntimePublicationOrchestrator.cpp:357
  trySubmitImpl(req)       (impl 150-341; decl 139)
    buildRuntimePublishWorld -> worldOwner            (176)
    evaluate CrossfadeAuthority                       (207-213)  [pre-F6, untouched]
    executor_.publish(engine_, frozen, req.newDSP, oldHandle)   (280)
      └─ Route A: Route B/C: commitRuntimePublication  (PublicationExecutor.cpp:53 / AudioEngine.h:4688)
         ─ Success ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
         RuntimePublishExecutor.h:104  ctx.transition.onPublishCompleted(newDSP, oldDSP, oldHandle, decision, lifetime)
         RuntimePublishExecutor.h:110  ctx.engine.advanceRetireEpoch()
         → onPublishCommitted(seq, obligationId)            (Orchestrator.cpp:320,354)
  Decision::Accepted  (340)  ← trySubmitImpl returns success
```

**onPublishCompleted (DSPTransition.h:49-155)** — Execution tail, publish **成功後のみ**呼び出し (executor tail h:17〜110)。
- Emergency Override (h:55-94): Critical ヘルス → `lifetime.activate(newDSP)` (61), register+active handle (68-72), `if (oldDSP != nullptr && oldDSP != newDSP)` で `lifetime.retire(oldDSP)` (79) + `retire(prevRaw)` (82, exchangeFadingRuntimeDSP h:77 — emergency CAS, single claim).**activate は publish 前に呼ばれない** (onPublishCompleted は post-commit 専用)。**publish 失敗時 activate なし** — failure path (h:290-291) は `destroyRolledBackDSP(newDSPResolved)` だけ → activate 呼び出し 0。
- Normal path (h:96-154): `lifetime.activate(newDSP)` (97) → register+active (101-105) → needsCrossfade branch (109-149): `beginCrossfade` (119), `claimFadingRuntimeDSP` CAS exactly-once (126; CAS fail→`lifetime.retire(oldDSP)` 129); no-crossfade `else if old!=new → retire(old)` (150-154)。
  - `oldDSP == newDSP` ガード: h:74, h:109, h:150 の三箇所で `&& oldDSP != newDSP` (old==new では retire/crossfade 対象外)。**✓ old==new retire なし。**
  - double-retire 回避: crossfade claim 時 (126) CAS 成功 => oldDSP は fading slot へ移動し **ここでは retire しない** (後続 fade-complete retire)。CAS fail (129) のみ retire。`exchangeFadingRuntimeDSP` (emergency h:77) は Critical パス単一。**✓ double-retire なし。**

**F6 regression 注目:** trySubmitImpl (h:332-338) は activate/crossfade/retire を **委譲** (Execution tail へ)。F6 は trySubmitImpl を変更しない (5f6f48c diff は trySubmitImpl region に lifecycle term を含まない — E-6 参照)。`finishView`/`invalidateDeferredObligation` は DSP を retire しない (D-6 確認) — deferred obligation release ≠ DSP ownership release。

> **Decision/Execution/Registration 分離 (ADR-D2) 維持:** Admission = Decision only (trySubmitImpl) ; Execution = onPublishCompleted/retire (DSPTransition) ; Registration = handle map (h:4299/4336, D4LifeManager). F6 は Admission (Orchestrator) のみを触り、Execution/Registration は未変更。

---

## E-2. DSPHandle ownership producer→consumer→retire — PASS

```
producer (publish request)
  registerDSPHandleForRuntime(newDSP)        AudioEngine.Commit.cpp:804 (pre-register, req.newDSP=handle h:4299 inline)
  → submitPublishRequest → trySubmitImpl → commitRuntimePublication
consumer (commit tail)
  DSPTransition.h:102/113 registerDSPHandleForRuntime(newDSP)  (re-register on activate; activate dspHandleRuntime_.activate h:71,104)
  → onPublishCompleted(newDSP, oldHandle, ...)  DSPTransition.h:49
  oldHandle = current active handle (resolved before publish; executor_.publish(...,req.newDSP, oldHandle) cpp:280)
retire
  lifetime.retire(oldDSP) / lifetime.retire(prevRaw)  DSPTransition.h:79/82/129/153
    → DSPLifetimeManager::retire (cpp:35-77)
       → engine_.retireDSPHandleForRuntime(dsp)  (h:4336 inline: findAndEraseByHandle ⇒ erase-once)
       → if retired: router_->enqueueWithRetry(...→ISRRetireRouter EBR)  (cpp:58-61)
       → if !retired: return (handle not in map → already reclaimed/claimed elsewhere, ownership NOT lost)  (cpp:48-49)
```

1. **new DSP publish 成功後登録**: h:4605-4609 ScopeExit guard — `registerDSPHandleForRuntime(regCtx.dsp)` (h:4612) は commit 開始前 registration、成功なら guard.dismiss (commit 内で active)、失敗なら `rollbackDSPHandleRegistration(rollbackHandle)` (h:4608, h:4545 findAndEraseByHandle) で**即 rollback**。✓
2. **active handle → next old handle**: `dspHandleRuntime_.getActiveRuntimeDSPHandle()` / oldHandle captured at cpp:280 → onPublishCompleted の oldHandle。resolveDSPHandle で DSPCore を照分して retire (h:4336)。✓
3. **retireDSPHandleForRuntime 正しい handle 解決**: h:4336 inline — `runtimeDSPHandleMap_.findAndEraseByHandle(handleFromDSP)`。h:146 comment "後方則のみ（value 一致 → erase)" — erase-once。✓
4. **false return 時 ownership 消失なし**: D4LifeManager::retire cpp:48-49 `if (!retired) return;` — handle が map にない (already retired/reclaimed) 場合は enqueue せず return。handle はその時点で誰かの所有/retire pipeline に在り、**double-ownership なし**。Thread-safety: h:4883 (retireDSPHandleForRuntime は multi-thread で mutex `runtimeDSPHandleMapMutex_` 保護, D4LifeManager::retireByHandle h:87)。✓
5. **retirement enqueue の戻り値確認**: `router_->enqueueWithRetry` (ISRRetireRouter.cpp:239-277) returns `RetireEnqueueResult` (Success/QueuePressure/etc.); QueueFull 時は `RetireQuarantineStore` へ自動退避 (ISRRetireRouter.cpp:266, AudioEngine.Retire.cpp:71-91), **直接 delete しない** (h:71-72 / AudioEngine.h:2203 comment "実際の解放は retireDSPHandleForRuntime() → deferred delete")。結果は log (63) だが `juce::ignoreUnused(result)` (72) は **lose ではない** — enqueueWithRetry が内部で quarantine 保証。✓
6. **retire failure / rollback path**: retirement enqueue 失敗 → quarantine (EBR epoch 未満で再試行, ISRRC レベルで drain) — ownership conserved (quarantine store owns)。publish rollback (h:4605-4613) は handle rollback + destroyRolledBackDSP (D-5/E-5) で別経路、retire pipeline 未投入。✓

**F6 regression:** F6 は registerDSPHandleForRuntime/retireDSPHandleForRuntime/ISRRetireRouter を **追加・変更しない** (5f6f48c diff: 0 lifecycle additions of these).cpp:466(overwrite)/:500(dormant) は **既存ハンドルの release** (register+1 / retire+1 balanced)。

---

## E-3. Crossfade lifecycle (normal + emergency) — PASS

### Normal (DSPTransition.h:96-154)
`needsCrossfade && old!=new` (109): `lifetime.activate(new)` (97,101-105) → `beginCrossfade(oldHandle,newHandle,xfadeId)` (117,119) → `claimFadingRuntimeDSP(oldDSP)` CAS (126):
- CAS success → old→fading slot, retire は行わず (`storeReceipt(fadingHandle,epoch)` h:136) → fade-complete 时に retire (Timer.cpp:944-972 endCrossfade → 954-956 unregister)。**✓ old fading DSP は一次 retired されない。**
- CAS fail → `lifetime.retire(oldDSP)` (129) (another path already claimed fading)。

No-crossfade: `else if old!=new → lifetime.retire(old)` (150-154)。old==new は 109 guard で除外。

### Emergency / Critical (DSPTransition.h:55-94)
health==Critical → `lifetime.activate(new)` (61) → register+act (68-72) → `exchangeFadingRuntimeDSP(oldDSP)` CAS (h:77, return prevRaw) → `crossfadeRuntime_.complete()` (78) → `lifetime.retire(old)` (79) → if `prevRaw != oldDSP` (78: `prevRaw != oldDSP`) → `lifetime.retire(prevRaw)` (80-82)。**✓ exactly-once**: exchangeFadingRuntimeDSP は CAS swap (AudioEngine.h exchange decl) — 勝者のみ prevRaw!=nullptr で retire 対象。double-retire なし (old==prevRaw なら skip 80-82)。old≠new ガード (74)。

### Terminalization primitive
`terminalizeFadingDSP()` — declared only (AudioEngine.h:1180), **impl 0 / caller 0** (grep `terminalizeFadingDSP` in src/ → decl only)。dormant stub (Phase-II reserve)。**Live terminalization** は CAS clear `fadingRuntimeDSPSlot` (Timer.cpp:959-967 + ReleaseResources.cpp:156-158 + CtorDtor.cpp:154-156) + `endCrossfade` (ISRDSPHandle.cpp:104, Timer.cpp:954) + `crossfadeAuthorityRuntime_.unregisterCrossfade` (955)。F6 はこれらを**改変しない** (5f6f48c diff: claimFading/exchangeFading/fadingRuntimeDSPSlot/endCrossfade = 0 hits, E-6)。

### F6 regression summary
- `oldDSP != newDSP` guard (3 sites) は F6 前から存在、F6 未変更。
- crossfade retire path (claim→fading→later retire) は F6 未変更 (F6 diff は Timer.cpp fade-COMPLETION retire (944-972) に lifecycle term を含まない; F6 が Timer.cpp に追加したのは fade-complete **wake** (1011-1020) と recovery wake (1741-1772) + requestDeferredClear (1662/1829/1849) のみ)。
- CrossfadeAuthority は decision/registration authority (h:30-37) を **奪っていない** — `evaluate` (207-213) is read-only Decision; `registerCrossfade` (117) issues id; authority remains with `crossfadeAuthorityRuntime_`。F6 は CrossfadeAuthority 呼び出しを 1 箇所も add しない。

---

## E-4. Deferred path → ownership release regression — PASS

F6 modified: `peek → evaluateDeferred → consume/discard → finishView → submitPublishRequest` + `invalidateDeferredObligation`. **deferred obligation release ≠ DSP ownership release** を確認。

| Case | what resets | DSP ownership | site |
|---|---|---|---|
| Accepted | `invalidateDeferredObligation` (gen=0,id=0,count=0,createdAt=0) + slot.reset + hasDeferred_=false | **none** — req move-out → submitPublishRequest の publish lifecycle で activate/retire | cpp:727-731,745-746,637-662 |
| StaleDiscard | discard→finishView→invalidate | none (slot's newDSP was registered but unpublished — see below) | cpp:734-738 |
| TTL discard (=StaleDiscard) | invalidate | none | Pub:80→cpp:736 |
| ShutdownDiscard | clearDeferredForShutdown: slot reset + invalidate; OR discard path | none | cpp:546-560 / Pub:75→736 |
| Rejected* (all 5) | post-check 745-746 invalidate (if reached via processDeferredAdmission) | **none** — resolveIfRecovery は recovery logical obligation (ISRRC), DSP ownership は別 ledger | cpp:380-432 |
| RejectedPublishFailure | destroyRolledBackDSP(newDSPResolved) (ScopeExit rolled back handle) | old active DSP 維持; newDSP direct-destroy (NOT EBR retire queue) | cpp:291 (E-5) |

**Deferred slot newDSP ownership (cpp:460-468, F6-added):** overwrite 時 old slot の `request.newDSP` を `retireDSPHandleForRuntime(oldDSP)` (cpp:466) で release。この DSP は **register されたが publish/active されていない** (DeferredFadingActive → never reached onPublishCompleted) 。register+1 (Commit.cpp:804) / retire+1 (cpp:466) = **balanced**, handle map findAndErase で一度だけ消去。F6 による追加だが **既存の D132 overrun-retire contract 実装** (h:460 comment INV-DEFERRED-2) — retry/retention accounting に混在していない (470-511 block は別)。`finishView`/`invalidateDeferredObligation` は **DSP を触らない** (h:299 member reset only)。

> **F6 reset が recovery logical obligation を消失させない:** `invalidateDeferredObligation`/`finishView` は `resolveRecoveryObligation` を**呼ばない**。recovery obligation ownership は ISRRC (E-7) 独立管理。post-check 745-746 は deferred publish slot metadata だけを 0 リセット。

---

## E-5. RejectedPublishFailure / rollback DSP ownership — PASS

`trySubmitImpl` failure path (cpp:139-314):
- `executor_.publish` != Success (cpp:281) → **activate/crossfade/retire 一切行わない** (comment cpp:285-286)。
- `lifetime_.destroyRolledBackDSP(newDSPResolved)` (cpp:291) → `AudioEngine::destroyDSPCoreNode(dsp)` direct delete (DSpaceLifeManager.cpp:149-157) — **EBR retire queue 未投入** (enqueueWithRetry 呼ばない)。currentRetiringGeneration_ +1 (154-156, diagnostic)。
- `markTransientFailure(req.recoveryObligationId)` (cpp:311) — **logical recovery obligation は retry 保持** (not terminal; ResolvedFailed は markTransientFailure exhaustion path ISRRC.cpp:987 のみ)。DSP rollback と logical obligation disappearance を**混同しない** (comment 304-305, 306-310 D105-R18)。

**commit-time ScopeExit (commitRuntimePublication, AudioEngine.h:4595-4622):**
- `AdmissionTokenGuard` (4599-4603)、`rollbackHandle` (4605) + `ScopeExit guard` (4606-4609): commit 失敗時 `rollbackDSPHandleRegistration(rollbackHandle)` (4608) → h:4545 findAndEraseByHandle で **handle map から即削除** (register+1 / rollback retire+1 balanced)。
- newDSP registered at 4612 (`rollbackHandle = registerDSPHandleForRuntime(regCtx.dsp)`); commit 成功すれば guard dismissed (active handle becomes oldDSP next publish)。

✓ publish 未達 DSP が active にならない / rollback DSP が retire pipeline に誤投入されない / `destroyRolledBackDSP` 正回収 (direct) / **old active DSP ownership 維持** (failure path は newDSPResolved/newDSP のみ touch; oldHandle untouched) / logical obligation disappearance ≠ DSP rollback (separate ledgers)。

---

## E-6. F6 diff contamination sweep — PASS

`git diff a65ace1..5f6f48c` (5 F6 files) + `git show 5f6f48c` per-file, **lifecycle term を含む diff line** のみ抽出:

| lineage term | F6 diff に現れるか |
|---|---|
| `retireDSPHandleForRuntime` | **YES — 2 lines added only**: cpp:466 (overwrite-retire, correct) + cpp:500 (dormant kMax guard, never fires) |
| `lifetime.retire` / `lifetime.activate` | NO (0 diff lines) |
| `destroyRolledBackDSP` | NO (cpp:291 pre-existing; `git show 5f6f48c` count = 0) |
| `registerDSPHandleForRuntime` | NO |
| `enqueueRetire` / `ISRRetireRouter` | NO |
| `class/struct DSPTransition` | NO |
| `CrossfadeAuthority` | NO (207/211 pre-existing; 0 additions in 5f6f48c) |
| `terminalizeFadingDSP` | NO |
| `claimFading`/`exchangeFading`/`fadingRuntimeDSPSlot`/`beginCrossfade`/`endCrossfade` | NO (0 hits) |

Threading.cpp:63 `retireDSPHandleForRuntime(dsp)` — pre-existing (F6 Threading diff = watchdog 284-301 region only; shutdown-retire at 63 unchanged).

> **結論:** F6 は lifecycle/retire pipeline (ISRRetireRouter / EBR / DSPTransition / CrossfadeAuthority) を**改変しない**。lifecycle 語が F6 diff に現れるのは cpp:466 (balanced overwrite-retire) と cpp:500 (dormant) 2 箇所のみ。いずれも ownership conservation に違反しない (D-4/E-2 verification)。

**余談:** 5f6f48c は `tests/AudioEngineHarness/DeferredFlowIntegrationTests.cpp` +2 byte (1 line) の test tweak を含む。これは Gate-E constraint (audit edits 0) 違反ではなく、F6 commit の一部。未検証の 1-line 変更として記録 (E-4 ケース表の補助検証対象外)。

---

## E-7. Ownership conservation — PASS

### Publish ownership (live)
```
PublishRequest(G,O) ──submitPublishRequest→ trySubmitImpl ──publish success─→
   commitRuntimePublication (ScopeExit rollback guard h:4605)
   → onPublishCompleted (DSPTransition.h:49) → lifetime.activate(new) (DSPLifetimeManager::activate → registerDSPHandleForRuntime h:4545)
   → oldDSP retire via lifetime.retire → retireDSPHandleForRuntime (h:4336 findAndErase) → ISRRetireRouter.enqueueWithRetry (quarantine fallback)
```
- `retireDSPHandleForRuntime` は **findAndEraseByHandle** (erase-once, D4LifeManager::retireByHandle h:87 mutex) → double-retire  impossible (handle map entry gone after first)。
- `claimFadingRuntimeDSP` CAS (h:126) + `exchangeFadingRuntimeDSP` (h:77) → fading slot exactly-once claim。
- `oldDSP != newDSP` guard (3 sites) → self-retire なし。

### Recovery logical obligation (ISRRC, a65ace1 — F6 非改変)
```
ObligationState: NoObligation/Live/ResolvedSuccess/ResolvedFailed/ResolvedStaleSuperseded/ResolvedRetry/ShutdownDiscarded  (ISRRuntimePublicationCoordinator.h:271-279)
ObligationDeliveryState: None/Transport/Durable  (h:281-292)
RecoveryOutcome: Published/Failed/StaleSuperseded/Retry/ShutdownDiscarded  (h:298-304)
```
- **single Completion Authority:** `resolveRecoveryObligation(id, outcome)` (ISRRC.cpp:944-980): `RecoveryOutcome::Retry` → early return, obligation **stays Live** (ΔL=0, 962-963, re-arm durable retry)；他は terminal table `recoveryAdmissions_.resolve(id, terminal)` → single −1 (liveCount_)。called only from `onPublishCommitted` (320), `submitPublishRequest` rejection branches (371), `markTransientFailure` exhaustion (987)。
- **invariant (h:334-336,410):** `liveLogicalRecoveryObligationCount()` == tryInsert(+1, capacity-checked, h:355`< kCapacity(32)`) only − (resolve Live→terminal CAS, h:390)。`+1` (post-coalesce) / `−1` (resolve) single sites ⇒ **silent disappearance / double ownership / overwrite なし**。
- `ObligationDeliveryState::None` (deferred, h:877-878) → re-drive selectivity (recoverRedrive only delivery==None, h:877/899) ⇒ **duplicate delivery なし (C16/R10-3)**。

### F6 × recovery obligation non-interference (Gate-E core)
- F6 diff は **ISRRuntimePublicationCoordinator.{cpp,h} を含まない** (5f6f48c stat: ISRRC not in touched-files list; F6 files = the 5 listed only)。
- F6 が追加した `invalidateDeferredObligation` / post-check 745-746 は `resolveRecoveryObligation` を呼ばない → recovery logical obligation は生存。
- F6 cpp:716 `resetDeferredRetryBudget` (recovery wake) は identity/count/timestamp リセット **のみ** — obligationId/key を 0 にするが `resolveRecoveryObligation` は呼ばない (the recovery logical obligation was already Published at cpp:320 / will be retired by the bridge on its own terminal)。→ **recovery obligation disappearance を伴わない**。

> **silent disappearance / double ownership / double retire / overwrite = 0** (publish live + recovery logical)。INV-X1-7 / I4 conservation intact.

---

## Gate E verdict

| item | verdict | anchor |
|---|---|---|
| E-1 Publish→Activate→Retire | ✅ | DSPTransition.h:49-155 / RuntimePublishExecutor.h:104-110 |
| E-2 DSPHandle ownership | ✅ | h:4299/4336, AudioEngine.h:4605-4613, D4LifeManager.cpp:35-77, ISRRetireRouter.cpp:239-277 |
| E-3 Crossfade lifecycle | ✅ | DSPTransition.h:55-154; Timer.cpp:944-972 (pre-F6); CAS exactly-once |
| E-4 Deferred→ownership release | ✅ | cpp:466 balanced overwrite-retire; finishView/invalidate は DSP 非操作 |
| E-5 RejectedPublishFailure rollback | ✅ | cpp:291 destroyRolledBackDSP (direct); h:4605-4613 ScopeExit; old active preserved |
| E-6 F6 diff contamination | ✅ | `git show 5f6f48c`: lifecycle additions = cpp:466+cpp:500 only; ISRRC/DSPTransition/retire untouched |
| E-7 Ownership conservation | ✅ | h:334-336/353-399/410 (liveCount ±1); resolveRecoveryObligation single authority (944-980) |

**7/7 PASS → Gate E = PASS → Gate F 進行可能。** (Gate F = 予定されていれば; 現段階で source edit window は Gate E 判定のみ read-only, build 0, CTest 0 維持。)
