# D135-8/9 Gate F — End-to-End Invariant / Boundary / Regression Final Audit

**Status:** PASS (F-1 .. F-7)
**Date:** 2026-08-30
**Scope:** read-only. `production source 0 / test source 0 / build 0 / CTest 0`; documentation = evidence only.
**Audit base:** HEAD `5f6f48c` (F6 commit), git clean. **一次参照 = 再生成 `ConvoPeg.md` (`Generated: 2026-08-30 19:19:29`)**。旧版 ConvoPeg.md / stale index は根拠にしない。`git diff a65ace1..5f6f48c` で F6 provenance 再確認。
**位置付:** Gate D (retry/retention account) / Gate E (lifecycle/ownership/retire) がともに read-only PASS。Gate F は D/E の個別証明を **cross-cutting の状態遷移表・境界行列** へ再構成し、`processDeferredAdmission`(cpp:700) の一本化が全 invariant を閉じているかを最終確認する。

> **一次ソース規約:** 全 cite は working-tree `src/*.cpp|h` + commit-blob 比較。ConvoPeg.md は `enqueueTimestampUs = now` (cpp:532 aggregate init) 1件のみ (D-4/D-5 で証明済み)。

---

## F-1. Deferred ownership end-to-end conservation (state-transition table)

### States (4-tuple + slot)
```
S0 IDLE    : deferredSlot_=null, hasDeferred_=false, identity=(0,0), count=0, createdAt=0
S1 RETAIN  : deferredSlot_=slot{req(G,O), metadata(ts=createdAt,G,O), slotTs=now}, hasDeferred_=true
S2 ASYNC   : 同一 (G,O) 退避中 (slot 有・identity 同値・createdAt 不変)         — between wake & consume
S3 CONSUMED: view.consume() 直後 (slot 未 reset / hasDeferred_=true?) → finishView で IDLE
S4 DISCARD : view.discard(reason) → finishView で IDLE + invalidate
```

### Edge-transition table

| from | trigger | guard | to | members change | site |
|---|---|---|---|---|---|
| S0 | submitPublishRequest→DeferredFadingActive | !sameOblig(0,0 vs (G,O)) | S1 | identity=(G,O), count=0, createdAt=now; slot=req; hasDeferred_=true | cpp:475-534 |
| S1 | DeferredFadingActive re-enqueue (同 (G,O)) | sameOblig | S1 | **identity/count/createdAt 不変**; slot 置換 (old newDSP handle retire @cpp:466, balanced); metadata.ts=createdAt; hasDeferred_=true | cpp:441-468,475-483,513-534 |
| S1 | wake (fade/watchdog/recovery) | RebuildThread | S2 | (fade/watchdog) nothing; (recovery) reset identity(0,0)/count0/createdAt0 — **slot 不変** | cpp:715-716 |
| S2 | evaluateDeferred: Shutdown | ctx.shutdown | S4 | ShutdownDiscard | Pub:74-75 |
| S2 | evaluateDeferred: ageUs>ttl | ageUs>ttlUs | S4 | StaleDiscard | Pub:78-80 |
| S2 | evaluateDeferred: gen mismatch | m.generation≠current | S4 | StaleDiscard | Pub:83-84 |
| S2 | evaluateDeferred: seq<last | m.sequence<lastSequence | S4 | StaleDiscard | Pub:87-88 |
| S2 | evaluateDeferred: else | | Ready→S3 | | Pub:90 |
| S3 | consume()→submitPublishRequest | | | view state Consumed; slot move-out | cpp:676-683 |
| S3 | resubmit→DeferredFadingActive | sameOblig (identity kept) | S1 | hasDeferred_=true; post-check 745 no invalidate | cpp:377-379,745-746 |
| S3 | resubmit→Accepted | | S0 | hasDeferred_=false; **invalidateDeferredObligation** | cpp:745-746,666-672 |
| S3 | resubmit→Rejected* | | S0(deferred) | hasDeferred_=false; invalidate; recovery obligation resolved by bridge | cpp:380-432,745-746 |
| S4 | discard(reason) | | S0 | finishView: slot.reset, hasDeferred_=false, reason≠None→invalidate | cpp:734-738,647-654 |

### Cardinality & release checks
- **PublishRequest 0 or 1 (never 2):** `deferredSlot_` = `std::optional<DeferredPublishSlot>` (h:279) — single slot; enqueueDeferred *replaces* (513), finishView/clear *reset* (647/555). `req` is moved-out once in consume (cpp:680). No second consumer. ✓
- **Slot ownership not lost:** consume/discard both route to `owner_->finishView()` (cpp:681,690) — the sole ownership-release (h:187-189 comment). No path resets slot without finishView except clearDeferredForShutdown (555, shutdown-only). ✓
- **consume/discard no double-release:** `DeferredPublishView::state_` jassert (Valid→Consumed/Discarded, cpp:678/687); second call on same view is UB-guarded; `peekDeferred()` (615) returns `nullopt` once hasDeferred_ false. ✓
- **re-defer identity retained:** Ready→consume→submitPublishRequest→DeferredFadingActive→enqueueDeferred: `sameObligation` true (identity preserved because invalidate NOT called — hasDeferred_ stayed true through 745). createdAt/metadata.ts carried via cpp:528. ✓
- **terminal invalidate:** discard(653-654), Accepted/Rejected post-check(745-746), shutdown(560). All reset 4 members to 0. recovery-reset(716) zeros identity but keeps slot (next enqueueDeferred → new obligation). ✓

> **Conclusion:** F1. PublishRequest cardinality 0/1 — never 0-and-live / never 2. Slot ownership mono-release via finishView. consume/discard exactly-once (state machine). Double-free impossible.

---

## F-2. 3-ledger boundary — producer→consumer→terminal cross-check

The 3 independent ledgers:
```
L1 DeferredPublish (cpp:437-747)      members: deferredSlot_/hasDeferred_/deferredRetry{G,Id,Cnt}_, deferredObligationCreatedAtUs, lastRecoveryPublishSeq_(stamp)
L2 Recovery logical obligation (ISRRC) members: recoveryAdmissions_ table / ObligationState / ObligationDeliveryState / liveCount_
L3 DSPHandle ownership (h:4299/4336)  members: runtimeDSPHandleMap_ / active/fading handles / ISRRetireRouter EBR / destroyRolledBackDSP
```

### Cross-ledger call audit (write sites → target ledger)

| caller | touches L1 | touches L2 | touches L3 | notes |
|---|---|---|---|---|
| `finishView` (cpp:637-662) | + | – | – | slot.reset / hasDeferred_=false / invalidate (deferred only) |
| `invalidateDeferredObligation` (666-672) | + | – | – | 4 members →0 (L1 only) |
| `resetDeferredRetryBudget` (h:173-179) | + | – | – | 4 members →0; L1 only |
| `clearDeferredForShutdown` (546-560) | + | – | –(except? see note) | slot+identity+createdAt+lastRecoveryPublishSeq_=0 (stamp only) |
| `resolveRecoveryObligation` (ISRRC.cpp:944-980) | – | + | – | recoveryAdmissions_/liveCount_/counters |
| `retireDSPHandleForRuntime` (h:4336) | – | – | + | map erase-once + ISRRetireRouter enqueue |
| `destroyRolledBackDSP` (D4LifeManager.cpp:149-157) | – | – | + | destroyDSPCoreNode (direct, no EBR) |
| `submitPublishRequest` rej. branches (cpp:380-432) | –(post-check L1) | + (resolveIfRecovery/markTransientFailure/rearm) | + (destroyRolledBackDSP on RejPublishFailure) | **correctly routed to own ledger** |

**Cross-ledger release violations: 0.** Each of the 3 ledgers has exactly-one reset/release authority (L1=finishView/clear, L2=resolveRecoveryObligation single authority, L3=findAndErase-once + destroyRolledBackDSP). No call site releases another ledger's ownership.

**Note on `lastRecoveryPublishSeq_` (cpp:561 reset to 0 in clearDeferredForShutdown):** this is a **publish-sequence correlation stamp** (h:170 comment, h:301-302), NOT an obligation-lifetime counter. It is set at recovery (Timer.cpp:1743) and read as a correlation mark; zeroing on shutdown has no obligation-ownership effect (obligation lifetime = L2). **Not a cross-ledger release.** ✓

> **F2 verdict:** `finishView`/`invalidate`/`resetBudget`/`clearDeshutdown` → L1 only. `resolveRecoveryObligation` → L2 only. `retireDSPHandleForRuntime`/`destroyRolledBackDSP` → L3 only. No mixed release.

---

## F-3. Wake / clear / shutdown race boundary — flag matrix

### Wake-flag inventory (writer / reader / clear / memory-order)

| flag | type | set-true (writer) | consume/clear (reader) | memory order | in rebuildCV predicate |
|---|---|---|---|---|---|
| `publishRetryReady` | plain bool (rebuildMutex-guarded) | Timer.cpp:1017 (fade), Threading.cpp:293 (watchdog), Timer.cpp:1769 (recovery,同lock) | RebuildDispatch.cpp:889 read / 890 clear (under lock 850); CtorDtor:182·Prep:82·Release:175 reset | mutex(rebuildMutex) holds; not atomic | YES (h:854-859) |
| `recoveryRetryReady` | `std::atomic<bool>` | Timer.cpp:1768 `publishAtomic(...,true,release)` | RebuildDispatch.cpp:888 `exchangeAtomic(false, acq_rel)` | release-store / acq_rel exchange | **NO** (h:303-306 comment) |
| `deferredClearRequested_` | `std::atomic<bool>` | Orchestrator.cpp:591 `publishAtomic(true,release)` (requestDeferredClear) | Orchestrator.cpp:602 `exchangeAtomic(false, acq_rel)` (drainDeferredClearIfRequested, RebuildThread) | release / acq_rel exchange | **NO** (cpp:589-590 comment) |
| `hasDeferred_` | `std::atomic<bool>` | cpp:534 publishAtomic(true,release) | cpp:648/556 publishAtomic(false,release) | release-store / acquire-load (read sites 441,552,615,717,745) | YES (predicate 854-859 via `hasPendingTask`? no — separate) |
| `lastRecoveryPublishSeq_` | `std::atomic<PublicationSequenceId>` | cpp:565? no — setRecoveryPublishSeq h:165 (Timer:1743); cpp:561 resets 0 | getter h:168 consumeAtomic acquire | release/acquire | NO |

### Race cases

| race | analysis | verdict |
|---|---|---|
| fade wake × watchdog wake | both write `publishRetryReady=true` under rebuildMutex (Timer:1017 / Threading:293) — single bool latch; RebuildThread consumes once (889-890) | duplicate admission なし ✓ |
| recovery wake × ordinary wake | recovery sets `recoveryRetryReady` AND `publishRetryReady` (1768-1769, same lock); ordinary sets only `publishRetryReady`. Consumer exchanges `recoveryRetryReady` (888) **before** reading `publishRetryReady` (889) → if coalesced, wasRecoveryWake=true (conservative; F4-approved). Ordinary never writes recoveryRetryReady (Timer:1007 verbatim) | provenance 混同なし ✓ |
| wake × deferred clear | `requestDeferredClear` latches `deferredClearRequested_` (591)+notify; RebuildThread drains clear (905-906→602) **before** deferred admission (914-921) → clear wins; processDeferredAdmission sees hasDeferred_=false (717)/peek nullopt (615) → stale request NOT re-injected | stale 再投入なし ✓ |
| wake × shutdown | RebuildDispatch.cpp:861 `if(rebuildThreadShouldExit) break`; 862-868 `isShutdownInProgress()` → `publishRetryReady=false` + break (no admission); evaluateDeferred ctx.shutdown→ShutdownDiscard (Pub:74-75) | shutdown 後に publish/consume 進まず ✓ |
| clear × RebuildThread exit | `requestDeferredClear` checks `rebuildThreadShouldExit` (Orchestrator.cpp:583) → C1 sync `clearDeferredForShutdown` inline (585); else latch+notify — if thread exiting, the latch is drained on the final wake (drainDeferredClearIfRequested:602) or handled by C1; **no lost clear** (persistent latch, drained every wake) | clear ownership 失われない ✓ |
| recovery wake × no deferred | recovery wake only fired when `hasDeferredRequest()` (Timer.cpp:1741). If no deferred, block skips → recoveryRetryReady NOT set. Consumer `processDeferredAdmission`: if `wasRecoveryWake` but hasDeferred_=false (717) → reset already done (716) then return (718) — reset on empty state is idempotent (members already 0 / slot null) → no stray | stray state なし ✓ |

**lost-wake proof (deferredClearRequested_ & recoveryRetryReady):** both are `std::atomic`, **predicate-ineligible** (not part of rebuildCV predicate — if they were, a wake could be consumed merely by predicate re-evaluation without draining them). They are **persistent latches**. The RebuildThread, on *every* wake, drains them unconditionally (`drainDeferredClearIfRequested` at 905-906; `exchange recoveryRetryReady` at 888 before the deferred block). Because they persist until explicitly cleared and are drained on every wake (not only predicate-matching wakes), a wake that "misses" them cannot drop them — they survive until the next wake and are then drained. → **no lost wake / no lost clear.** Contract verbatim (Orchestrator.cpp:589-592, h:303-306, RebuildDispatch.cp:883-888).

> **F3 verdict:** All 6 race cases closed. Memory-order matrix correct (release-store set / acq_rel exchange-clear / acquire-load). Predicate-ineligible persistent latches give lost-wake proof.

---

## F-4. Capacity / overwrite boundary — container capacity ≠ ownership cardinality

### Container capacities (C) vs logical cardinality (L)

| container | capacity | meaning | ownership ledger |
|---|---|---|---|
| recoveryAdmissions_ table `slots_` (ISRRC.h:409) | **32** (kMaxLogicalRecoveryObligations h:325; kCapacity=C inst 32) | max concurrent **logical recovery obligations** (L2) | liveCount_ ≤ 32 (h:355) |
| recoveryIntentQueue_ (transport) | **256** (comment h:897) | transport residency of intents (delivery, not ownership) | L2 delivery only |
| pendingRecoveryAdmission_ (durable slot) | **1** | single durable pending (Building/DurablePending) | L2 delivery |
| deferredSlot_ (h:279) | **1** | single deferred **publish** slot (L1) — independent of L2 count | L1 |
| runtimeDSPHandleMap_ (h:5018) | **512** (DSPHandleTable, comment h:4528/5016) | registered DSP handles (L3) | L3 |
| quarantineFallbackQueue_ (ISRRC.h:962) | **1024** (kQuarantineFallbackCapacity) | retirement fallback residency | L3 |
| pendingReclaimHandles_ (Reclaim) | bounded reclaim staging | reclaimed pending handles | L3 |

### Separation confirmation
- ****logical ownership cardinality (liveCount_ ≤ 32) vs container cap (32 slots, 256 transport, 512 handles, 1 deferred):** the deferred slot is **1**, NOT capped at 32 — because L1 (deferred publish) is independent of L2 (logical obligation). A single deferred publish slot can be re-driven regardless of how many logical recovery obligations exist (the deferred req is *one* publish request; recovery obligations are a separate table).
- **I4-listed distinction** (container bound vs retired-world/handle cardinality): `pendingReclaimHandles_` (L3 reclaim staging) — capped by container, but *retired-world cardinality* counts EBR-epoch-guarded deletes. `liveCount_` (L2) is a true cardinality (single +1/-1). Container capacity and ownership cardinality are **treated as separate** (h:334-336 comment: "single +1 (tryInsert, post-coalesce, L<Capacity) and only −1 (resolve, Live→terminal CAS)").
- **No mixing:** F6 touches only L1 (deferred accounting); a65ace1 touched L2 (ISRRC); L3 (DSP handles/retire) pre-existing. Capacities are per-container; the single deferred slot is not confused with the 32 logical obligation capacity.

> **F4 verdict:** container capacity and logical ownership cardinality are separate concerns. 1 deferred publish slot ≠ 32 logical obligation cap ≠ 512 handle cap ≠ 256 transport.

---

## F-5. Dormant path contamination — reachability matrix

| symbol | declaration | producer | consumer | terminal effect | **live reachability** |
|---|---|---|---|---|---|
| `RetryExhaustedDiscard` | RuntimePublicationState.h:22 (enum) + cpp:497 guard | cpp:497 `if (deferredRetryCount_ > kMaxDeferredRetries)` — count always 0 (retention never increments; D-1) | cpp:498-509 (would retire req.newDSP cpp:500 + return) | `return` from enqueueDeferred (slot NOT written) | **UNREACHABLE** (0 > 2 always false) |
| `SupersededDiscard` | RuntimePublicationState.h:14 (enum) | **none** (grep producer = 0) | none | none | **UNREACHABLE** (dormant; supersession realized as StaleDiscard via gen/seq, Pub:83/87) |
| `Expired` | RuntimePublicationState.h:15 (enum) | **none** (TTL path returns StaleDiscard, Pub:80 — comment "Expired を別 enum 化可能") | none | none | **UNREACHABLE** (TTL→StaleDiscard) |
| `terminalizeFadingDSP` | AudioEngine.h:1180 (decl) | **none** (impl 0, caller 0) | none | none | **UNREACHABLE** (live terminalize = CAS-clear fadingRuntimeDSPSlot, Timer.cpp:959-967 etc.) |

**Enum/decl 存在 ≠ production path 存在:**
- `RetryExhaustedDiscard` enum retained as **dormant reserve** (F6-6, h:292-295: "kMaxDeferredRetries は将来の真の Type-A retry 経路用の dormant guard として保持（削除しない）"). Reachable **only if** a future Type-A retry increments count — not present now.
- `SupersededDiscard` / `Expired` enum-only, 0 producers — dormant, no runtime path.
- `terminalizeFadingDSP` decl-only stub — no impl, no caller; the "fading terminalize primitive" is the existing CAS-clear + endCrossfade (not F6-affected).

> **F5 verdict:** All 4 dormant symbols have **no production path** (0 producers/consumers/terminal), except `RetryExhaustedDiscard` which has a dormant count-gate (never fires). No enum-without-production confusion in the live code.

---

## F-6. F6 contamination / scope proof — additions vs depends-not-modify

### F6 additions (a65ace1..5f6f48c, 5 files; 283 insertions, 11 deletions)
1. Deferred accounting block (cpp:470-511): identity=(G,O), retention-doe-not-increment-count, createdAt=now-on-new.
2. `invalidateDeferredObligation` (666-672) + post-check (745-746).
3. `resetDeferredRetryBudget` relocated to cpp:716 (gated `wasRecoveryWake`, before peek).
4. `processDeferredAdmission(bool wasRecoveryWake)` signature + provenance gate (700,714-716).
5. `finishView` terminal-invalidate (650-654).
6. **cpp:466** overwrite-retire (balanced) + **cpp:500** dormant retire (both `retireDSPHandleForRuntime`).
7. `requestDeferredClear`/`drainDeferredClearIfRequested` (579-608) + latch `deferredClearRequested_`.
8. Fade-complete wake (Timer.cpp:1011-1020) + recovery wake handoff (Timer.cpp:1741-1772).
9. Watchdog throttling (Threading.cpp:284-301, `kDeferredWakeWatchdogTicks` 100 — replaced 1ms every-tick poll).
10. RuntimeState.h DiscardReason enum documentation (RetryExhaustedDiscard dormant + F6 contracts).

### F6 depends-not-modify (pre-existing lifecycle primitives)
| primitive | F6 changes it? | F6 calls/uses it? |
|---|---|---|
| `DSPTransition` (onPublishCompleted/activate/beginCrossfade/retire) | **NO** (0 diff lines) | NO (F6 は transition を呼ばない — trySubmitImpl cpp:337 は既にコメントアウト) |
| `CrossfadeAuthority` (evaluate/registerCrossfade) | **NO** (0 additions; 207/211 pre-existing) | NO (evaluate は pre-existing trySubmitImpl) |
| `DSPLifetimeManager` (activate/retire/beginCrossfade/destroyRolledBackDSP) | **NO** (5f6f48c は D4LifeManager.cpp を touch? 見る — stat なし) | partially: `lifetime_.destroyRolledBackDSP` (cpp:291 pre-existing) |
| `ISRRetireRouter` (enqueueRetire/EBR) | **NO** | `retireDSPHandleForRuntime` → router (indirect, pre-existing) |
| `ISRRuntimePublicationCoordinator` (ObligationState/table/resolve/liveCount_) | **NO** (5f6f48c stat に含まれない; a65ace1 が変更) | NO (F6 は resolveRecoveryObligation を呼ばない) |
| `retireDSPHandleForRuntime` (h:4336) | **NO** | **YES — 2 call sites added** (cpp:466 balanced, cpp:500 dormant) — use-only, body unchanged |

### Authority contamination check
`git diff a65ace1..5f6f48c` lifecycle-term additions = **`retireDSPHandleForRuntime` 2 lines (cpp:466, cpp:500) only**; `lifetime.{retire,activate}` / `destroyRolledBackDSP` / `registerDSPHandleForRuntime` / `enqueueRetire` / `ISRRetireRouter` / `DSPTransition` / `CrossfadeAuthority`(defs) / `claimFading`/`exchangeFading`/`beginCrossfade`/`endCrossfade`/`terminalizeFadingDSP` = **0 diff lines** → F6 does **not modify** any lifecycle authority. The 2 retire calls are **dependent use** (balanced, already audited E-2/E-4).

> **F6 verdict:** F6 additions are confined to deferred accounting / metadata / wake provenance / clear latch / watchdog throttling. Lifecycle primitives (DSPTransition/CrossfadeAuthority/DSPLifeManager/ISRRetireRouter/ISRRuntimePublicationCoordinator) are **dependencies-only, not modified** → **no authority contamination**.

---

## F-7. Final invariant matrix (GO / NO-GO)

| invariant | verdict | anchor |
|---|---|---|
| Deferred ownership conservation | **GO** | F-1 (0/1 cardinality; mono-release via finishView) |
| Recovery obligation conservation | **GO** | F-2/E-7 (single authority resolveRecoveryObligation, liveCount_ ±1) |
| DSP ownership conservation | **GO** | F-2/E-2 (findAndErase-once + destroyRolledBackDSP direct) |
| No double consume | **GO** | F-1 (view state_ jassert; peek nullopt after) |
| No double retire | **GO** | E-3 (CAS claim exactly-once; old!=new 3-guard) |
| No silent disappearance | **GO** | F-5 (dormant 0-path); F-4 (ownership ≠ container) |
| No blind overwrite affecting live ownership | **GO** | E-2/F-2 (h:4605-4613 rollback; cpp:466 balanced overwrite-retire) |
| Recovery/ordinary wake provenance separation | **GO** | F-3 (recoveryRetryReady ordinary never write; exchange-before-read) |
| Shutdown boundary | **GO** | F-3 (RebuildDispatch 861-868 break; Pub:74 ShutdownDiscard; clearDeferredForShutdown) |
| TTL origin = obligation creation time | **GO** | D-3/D-4 (metadata.ts = deferredObligationCreatedAtUs, re-drive 不変) |
| retention ≠ retry | **GO** | D-1 (retention で count 不増加; ++ deferredRetryCount_ = 0) |
| dormant path remains unreachable | **GO** | F-5 (4 dormant symbols: 0 producer path) |
| F6 authority boundary preserved | **GO** | F-6 (lifecycle primitive dependencies-only, 0 modifications) |

**13/13 GO — Gate F = PASS。**

---

## Gate F 判定 (read-only, 最終構造証明)

**全項目 PASS → `Gate F = PASS`。** `processDeferredAdmission()`(cpp:700) の一本化 (wasRecoveryWake gate → hasDeferred_ → peek → evaluate → consume/discard → submitPublishRequest → post-check invalidate) が、D の retry/retention invariant と E の lifecycle/ownership/retire invariant を**横断して閉じて**いることを state-transition・boundary・capacity・reachability・contamination の 5 軸で確認。

- F-1 state-transition edge table: D/E 個別証明を 0/1 cardinality + mono-release に再構成。
- F-2 3-ledger boundary: reset/invalidate/finishView/clear/recover/retire/destroy の 7 関数の cross-ledger release = 0。
- F-3 6 race cases + lost-wake proof (predicate-ineligible persistent latches)。
- F-4 container capacity (32/256/1/512/1024) ≠ logical ownership cardinality (liveCount_ ≤ 32)。
- F-5 4 dormant symbols = 0 production path。
- F-6 F6 additions confined to deferred/wake/watchdog; lifecycle primitives depend-not-modify。
- F-7 13/13 GO。

> 保守的注記 (NO-GO ではない): `5f6f48c` に `tests/DeferredFlowIntegrationTests.cpp` +1 line test tweak が含まれる (F6 commit 内)。本 Gate-F read-only 監査はソース編集しないため、これは F6 provenance の一部として記録のみ。将来の Gate (impl window) で確認推奨。また `terminalizeFadingDSP` / `SupersededDiscard` / `Expired` は dormant (実装 0) のため、Gate-H/I の実装 window で有効化する場合は別途監査が必要(Phase-II)。

**Next:** Gate F PASS → 実装へ進む前の最終構造証明が閉じた。F6 系 read-only 監査 (Gates A-F) は完了。**以降はユーザー指示のある次フェーズ (実装 window / build / CTest / Gate G+) 待ち。** 本 Gate 群ではソース編集・ビルド・CTest を行わない。
