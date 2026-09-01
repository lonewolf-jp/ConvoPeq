# D135-8/9 Gate D — Retry / Retention State-Machine Audit

**Status:** PASS
**Date:** 2026-08-30
**Scope:** `production source 0 changes / test source 0 changes / build 0 / CTest 0 / doc 0`
**Audit base (per Gate-D 制約):** F6-実装済み working tree (HEAD `5f6f48c`, 2026-08-30 19:30:11) + 再生成済み `ConvoPeq.md` (`Generated: 2026-08-30 19:19:29`, mtime 19:19:37).

## 0. Baseline & old-snapshot exclusion

- `git status`: clean. HEAD `5f6f48c` は F6 実装 + `output_sourcecode_markdown.py` 再生成 (19:19:29) + F6-SYNC-B0 restore PASS (Gate D 解錠前提) 後のユーザー commit。working tree = F6 実装 = regenerated ConvoPeq.md 内容一致（F6-SYNC-B0 audit 参照）。
- **旧版スナップショット排除:** 検索 indexes (cocoindex/sem/AiDex 等) に残る旧 `ConvoPeg.md` は旧実装を含み得る。今回の D-1〜D-12 の全 cite は **live `src/*.cpp / *.h`** (tree 一致確認済) に基づく。`enqueueTimestampUs = now` は ConvoPeq.md:66244 / src `RuntimePublicationOrchestrator.cpp:532` の **aggregate 初期化** (`slot.enqueueTimestampUs`) 1件のみ。旧実装の `metadata.enqueueTimestampUs = now` 毎リドライブ更新は**0件**。
- F6 実装は ConvoPeq.md 2026-08-30 版と byte-identical。監査の一義性を確保。

---

## D-1. retention と retry accounting の完全分離 — PASS

**定義 (h:292-295, cpp:470-473):** DeferredFadingActive の再 enqueue は *retention*（保持）。obligation identity が変わらない間は count/createdAt は不変。

`deferredRetryCount_` **全 write site** (working tree):

| site | 文脈 |
|---|---|
| `RuntimePublicationOrchestrator.cpp:480` | 新 obligation (`!sameObligation` branch): `= 0` |
| `RuntimePublicationOrchestrator.cpp:670` | `invalidateDeferredObligation`: `= 0` |
| `RuntimePublicationOrchestrator.h:177` | `resetDeferredRetryBudget`: `= 0` |
| `RuntimePublicationOrchestrator.h:298` | 定義フィールド: `uint8_t deferredRetryCount_{0};` |

**`++deferredRetryCount_` 残存検査:** `grep -rn "++deferredRetryCount_" src/` → **0 件**。
`deferredRetryCount_ =` (reset 以外)：`grep "deferredRetryCount_ ="` → **0 件**（唯一の `=` 系は `= 0`）。

Dormant guard (cpp:495-510 / h:300):
```cpp
if (deferredRetryCount_ > kMaxDeferredRetries) {   // cpp:497 — canonical `>`
    ... retire DSP ...
    return;  // RetryExhaustedDiscard (dormant)
}
```

> **判定:** ordinary retention/re-drive によって `deferredRetryCount_` が増加する経路は **0**。`RetryExhaustedDiscard` は Type-A retry 経路がない現行 production からは到達不能（h:287-294 / RuntimePublicationState.h:16-22 dormant 文書済み）。

---

## D-2. identity tuple `(generation, recoveryObligationId)` — PASS

**比較 (cpp:475-476):** 両軸必須
```cpp
const bool sameObligation = (req.generation == deferredRetryGeneration_
                             && req.recoveryObligationId == deferredRetryObligationId_);
```

**write (pair, never single axis) (cpp:478-479):**
```cpp
deferredRetryGeneration_ = req.generation;            // cpp:478
deferredRetryObligationId_ = req.recoveryObligationId; // cpp:479
```

**reset (pair) (cpp:668-669, h:175-176):** `generation=0; obligationId=0;` 同時。

**generation-only 残存検査:** `deferredRetryGeneration_` は cpp:475（tuple 比較）と `getRetryGeneration` (h:160, test-observe only) 以外に**読まれない**。`guard.generation` (cpp:515, Guard 構築) は**読み取られない** (`grep -rn "guard\.generation" src/` → 0; guard は h:273-277 の `DeferredGuard` で staleness 参照用意、identity 判定には未使用)。

**snapshot-staleness は identity と分離:** `evaluateDeferred` の `m.generation != ctx.currentGeneration` (PublicationAdmission.cpp:83) は**slot snapshot の鮮度**判定（recovery は generation を再利用 G=G なので spurious discard せず）。obligation identity とは別 ledger。h:287-288 参照。

**区別可能性 (動的):**
- `(G,0) != (G,O)`: tuple 比較で recoveryObligationId が 0↔O を区別 → yes。
- `(G,O1) != (G,O2)`: recovery は `recovery->obligationId` ごとに新 ID (RebuildDispatch.cpp:1040/1120) → yes。
- `(G1,O) != (G2,O)`: generation も比較 → yes。
- recovery は `recoveryGeneration = currentGeneration` (RebuildDispatch.cpp:1036-1037) → G を再利用しながら O を新規 → F4 要求 (generation-alone insufficient) 確認済み。

**call-chain (enqueue → re-drive → eval → consume → submit → re-defer) identity 流在:**
1. `enqueueDeferred(req)` (cpp:437): req.generation/req.recoveryObligationId は Commit.cpp:808/813 (`req.generation=generation; req.recoveryObligationId=recoveryObligationId`) から来る。
2. recovery publish: `enqueuePublicationIntentForRuntimeCommit(dspToCommit, recoveryGeneration, ..., recovery->obligationId)` (RebuildDispatch.cpp:1042) → 全 caller 3件が `rebuildThreadLoop()` 内 (1040/1120/1326)。
3. `submitPublishRequest` caller: Orchestrator.cpp:731 (processDeferredAdmission, RebuildThread) + AudioEngine.Commit.cpp:822 (enqueuePublicationIntentForRuntimeCommit, RebuildThread)。→ `enqueueDeferred` は **RebuildThread 単一スレッドからのみ**到達 (h:291 single-owner 契約成立)。

---

## D-3. createdAt lifetime proof — PASS

`deferredObligationCreatedAtUs` (h:299, Orchestrator **member**) — slotフィールドではない。slot lifetime ≠ obligation lifetime 構造的保証 (h:289-291)。

| ケース | コード | 証 |
|---|---|---|
| 初回 `(G,O)` | cpp:477-481 `!sameObligation` (stored key 0,0 → differ) | `deferredObligationCreatedAtUs = now` (481) |
| 同一 `(G,O)` re-drive | cpp:477 `!sameObligation` = false | 480-481 skip → **不変** (483 comment) |
| 別 `(G,O)` / `(G1,O1)→(G2,O2)` | cpp:477 `!sameObligation` = true | cpp:481 `= now` **新 ts** |
| terminal | cpp:654 (`invalidateDeferredObligation` 4-site: 560, 654, 716→via reset, 746) | `= 0` invalidate |

**consume() 後の保持根拠:** `DeferredPublishView::consume()` (cpp:676-683) → `owner_->finishView()` (cpp:637) → slot reset + `hasDeferred_=false` (cpp:647-648), `lastDiscardReason==None` なので `invalidateDeferredObligation` **呼ばない** (cpp:653-654)。したがって `deferredObligationCreatedAtUs` (member) は生存。次ロード: Ready→submitPublishRequest→DeferredFadingActive→`enqueueDeferred(req)` → `!sameObligation` が false (key match) → createdAt 維持 → metadata construction cpp:528 `.enqueueTimestampUs = deferredObligationCreatedAtUs` で slot に再搭乗。

> **slot lifetime ≠ obligation lifetime** 実証: slot は `finishView`/`clearDeferredForShutdown` で reset されるが、identity/createdAt/count は Orchestrator member で存続 → consume 後も re-drive で correct ts 引き継ぐ。

---

## D-4. timestamp source-of-truth — PASS

3 階層は物理的分離:

- **obligation source-of-truth**: `deferredObligationCreatedAtUs` (h:299; writes 481, 655-reset via 671, 178).
- **metadata transport**: `DeferredPublishMetadata.enqueueTimestampUs` (PublicationAdmission.h:89 / slot field h:37) — `cpp:528 = deferredObligationCreatedAtUs` (re-drive も含め常に obligation ts から派生)。**metadata は独立更新されない** (480-481 ブロックの外でもう一度 `= now` する箇所は 0; 再生成前の `metadata.enqueueTimestampUs = now` 毎リドライブ刷新は 0 件 — D-4 NG-2/-3 回避)。
- **slot age**: `DeferredPublishSlot::enqueueTimestampUs` (RuntimePublicationOrchestrator.h:37, cpp:532 `= now`) — **唯一の使用先** cpp:449 overwrite-age (maxDeferredAgeMs, F5-4 diagnostic)。TTL/evaluate では**一切参照されない**。

TTL 読取: `PublicationAdmission.cpp:78` `ageUs = ctx.nowUs - m.enqueueTimestampUs` — `m` は `view->metadata()` (cpp:724) = obligation createdAt を輸した metadata。slot.enqueueTimestampUs は TTL 算出に介入しない (D-4 NG-1 違反なし)。

---

## D-5. TTL state transition — PASS

`**processDeferredAdmission` (cpp:700) 順序:**
```
715-716  reset(if wasRecoveryWake)
717-718   hasDeferred_ check (no-op guard)
720-722   peekDeferred()  (jassert RebuildThread @614)
724-725   evaluateDeferred(metadata, snapshot)
    ↓  PublicationAdmission.cpp:69-91
74-75  Shutdown → ShutdownDiscard
78-80  TTL: ageUs = nowUs - m.enqueueTimestampUs ; if (ageUs > ctx.ttlUs) → StaleDiscard  (strict `>`)
83     Generation: m.generation != currentGeneration → StaleDiscard
87     Sequence:  m.sequence  < lastSequence      → StaleDiscard
90     else → Ready
727-732 consume → submitPublishRequest
734-738 discard(reason)
745-746  post-check: !hasDeferred_ → invalidateDeferredObligation
```
`>`. 判定: `ageUs > ctx.ttlUs` (PublicationAdmission.cpp:79) **strict**。`>=` 残存検査 (`grep -rn "ageUs >=\|>= ctx.ttlUs" src/`) → **0 件**。retry-cap も `deferredRetryCount_ > kMaxDeferredRetries` (cpp:497) strict `>`。

`kDeferredPublishTTLUs = 30'000'000` (h:125)。ageUs は `now − createdAt` (D-4)。**obligation dwell 上限 30s**: metadata は re-drive 毎に `deferredObligationCreatedAtUs` (unchanged by retention) から再派生するため、`(G,O)` 保持期間は 30s を越えて破棄 (F3 churn/lost-expiry 修正済み)。

---

## D-6. terminal reset completeness — PASS (F5-10 表不変確認)

6 メンバ: `deferredSlot_`, `hasDeferred_`, `deferredRetryGeneration_`, `deferredRetryObligationId_`, `deferredRetryCount_`, `deferredObligationCreatedAtUs`

| path | site | slot | hasDeferred | gen/id/cnt/createdAt (invalidate @666-672) |
|---|---|---|---|---|
| Accepted | cpp:727-731 consume→finishView(647-648) → 745-746 invalidate | reset | false | 0 (746) |
| StaleDiscard (gen/seq mismatch) | Pub:84 → cpp:736 discard→finishView(654) | reset | false | 0 |
| TTL discard (StaleDiscard) | Pub:80 → cpp:736 | reset | false | 0 |
| ShutdownDiscard (eval) | Pub:75 → cpp:736 | reset | false | 0 |
| ShutdownDiscard (clear) | clearDeferredForShutdown cpp:553-560 | reset(555) | false(556) | 0(560) |
| recovery reset | processDeferredAdmission cpp:715-716 | **untouched** | **true** | 0 (resetDeferredRetryBudget h:173-179) — slot 残し，next deferral を new dwell |
| RejectedStaleGeneration | cpp:380-386 → resolveIfRecovery(StaleSuperseded) — 後 745-746 invalidate | reset | false | 0 |
| RejectedNotFinalized | cpp:389-402 → markTransientFailure (non-terminal/bridge) — 745-746 | reset | false | 0 |
| RejectedPressure | cpp:403-414 → resolve(Retry)+rearm (non-terminal/bridge) — 745-746 | reset | false | 0 |
| RejectedShutdown | cpp:415-421 — 745-746 | reset | false | 0 |
| RejectedPublishFailure | cpp:427-432 (no 2nd resolve; trySubmitImpl 既解決) — 745-746 | reset | false | 0 |
| Superseded (generation advance) | `StaleDiscard` via Pub:83 | reset | false | 0 |
| **RetryExhaustedDiscard** | cpp:497 dormant — 到達不能 | n/a | n/a | n/a |

**F5-10 表不変 (evidence/D135-8-9_GATE_C_F5_FINAL_BOUNDARY_AUDIT.md:150-160):**
- RejectedStaleGeneration: 非 rec 終端 / rec resolve(StaleSuperseded) → cpp:385 ✓
- RejectedNotFinalized: rec markTransientFailure (non-terminal) → cpp:400-401 ✓ (ΔL=0/Delivery=None counter+1/枯渇→ResolvedFailed, F4-7 訂正反映)
- RejectedPressure: rec resolve(Retry)+guarded rearm → cpp:411-413 ✓
- RejectedShutdown: resolve(ShutdownDiscarded) → cpp:420 ✓
- RejectedPublishFailure: trySubmitImpl 既解決, DSP `destroyRolledBackDSP` 回収済み (cpp:425-426コメント, RuntimePublicationOrchestrator.cpp:304-305) → cpp:427-432 ✓

> ** Rejected* は bridge (recovery obligation) の終局を定義する。 deferred obligation metadata はすべて 745-746 の post-check で invalidate される。F5-10 の "終端/非終端" は recovery obligation lifecycle を指すため、D-6 の deferred-member tracking とは独立 ledger。いずれも F5-10 表を変更せず一致。**

**SupersededDiscard** (RuntimePublicationState.h:14): `grep` producer → **0** (dormant enum)。supersession は generation/seq mismatch → StaleDiscard (Pub:83/87) で実現。D-6 に影響なし (到達不能のため reset 追跡不要)。将来 Type-A 用 reserve。

---

## D-7. recovery wake provenance — PASS

**発火点の唯一性:** `resetDeferredRetryBudget()` call sites = **cpp:716** (processDeferredAdmission, `if (wasRecoveryWake)`) 唯一。h:173-179 定義のみ他に。→ "ordinary fade/watchdog wake で reset しない" 厳守。

**wake provenance 3 path:**
- **fade-complete** (Timer.cpp:1011-1020): `publishRetryReady=true` のみ，`recoveryRetryReady` は未触 (1007 comment)。→ `wasRecoveryWake=false` (RebuildDispatch.cpp:888 exchange=false)。→ reset 不発。
- **watchdog** (Threading.cpp:284-301): `publishRetryReady=true` のみ，`recoveryRetryReady` 未触。→ `wasRecoveryWake=false`。→ reset 不発。
- **recovery** (Timer.cpp:1766-1771): lock 下 `recoveryRetryReady=true` (1768, release-store) **+** `publishRetryReady=true` (1769) → `notify_one` (1771)。→ `wasRecoveryWake = exchange(recoveryRetryReady, false)` (RebuildDispatch.cpp:888) = true → cpp:715-716 reset。

**blind producer-side reset removed (D135-5 blocker 解消):** Timer.cpp:1744-1748 comment『D135-5/8: blind producer-side reset removed. Recovery-retry budget reset is gated on recoveryRetryReady provenance in processDeferredAdmission (P3).』旧コードで Timer が直接 `resetDeferredRetryBudget()` 呼んでいた箇所は**0** (grep `resetDeferredRetryBudget` Timer.cpp → 0)。

---

## D-8. fade-complete wake (producer→consumer) — PASS

`AudioEngine::timerCallback` (Timer.cpp):
1. `fadeCompleted = m_coordinator.tryCompleteFade()` (Timer.cpp:931) — **fade 完了 tick のみ**进入本ブロック (D-8 "fade前にwakeしない" ✓)。
2. idle world publish: `publishIdleWorldOnly`→`commitRuntimePublication` (984-997) — **idle publish 完了後** (D-8 "idle publish 完了後" ✓)。
3. `sendChangeMessage()` (1000)。
4. F6-4 wake block (Timer.cpp:1011-1020):
   - `!isShutdownInProgress()` guard (1011) ✓
   - `runtimeOrchestrator_ != nullptr && hasDeferredRequest()` guard (1012-1013) ✓ ("deferred 不在なら no-op")
   - `recoveryRetryReady` 非変更 (1007 comment + no write site in block) ✓
5. lock(rebuildMutex) → `publishRetryReady=true` (1016-1017) → unlock → `rebuildCV.notify_one()` (1019)。CV predicate 未変更 (h:2718 comment / RebuildDispatch.cpp:854-859: `hasPendingTask || publishRetryReady || recoveryPending || rebuildThreadShouldExit`)。

consumer side (RebuildDispatch.cpp):
- predicate wake → `doDeferredPublish = publishRetryReady` (889); `publishRetryReady=false` (890)。
- `processDeferredAdmission(wasRecoveryWake=false)` (920) → peek(720) → evaluateDeferred(724) → Ready→consume→submitPublishRequest(731)。

---

## D-9. watchdog contract — PASS

`runCoordinatorPhase` (Threading.cpp:258-301), producer-side (CoordinatorLoop thread):
```
284  if (!isShutdownInProgress() && orchestrator && hasDeferredRequest()) {
288      if (++coordinatorDeferredWatchdogTicks_ >= kDeferredWakeWatchdogTicks) {
290          coordinatorDeferredWatchdogTicks_ = 0;
292          lock(rebuildMutex); publishRetryReady=true; unlock;   // ordinary provenance
295          rebuildCV.notify_one();
296      }
298-301 } else { coordinatorDeferredWatchdogTicks_ = 0; }
```

- **通常 tick → notify しない**: counter インクリメントのみ (`++`)；notify は `>=` branch に**限定**。100tick 到達前は CPU 0 waste。旧 1ms 毎 tick publishRetryReady set (Gate-C F3 churn root cause) は**削除済み**。
- **lost-wake self-repair**: Timer.cpp:280-283 + h:2718-2724 comment『現行 P1（coordinator poll）は毎 tick publishRetryReady を立てていたが、F6 では fade-complete wake が正常経路、本 poll は lost-wake 自己修復専用。周期は fallback tuning 値であり設計契約ではない（F5-6）』。
- **period**: `kDeferredWakeWatchdogTicks = 100` (h:2724) ≈ 100ms @1ms tick. **F4 'watchdog 周期未確定' open item 解決**: 100ms は仕様契約ではなく named constant (tunable fallback)，F5-6 により fixed-as-spec ではない。

> watchdog は processDeferredAdmission を起動せず (RebuildThread が実体) → poll churn 再発なし。

---

## D-10. duplicate / lost wake — PASS

- **Case A (fade wake + watchdog wake → bool latch):** 二者とも `publishRetryReady` (rebuildMutex 保護 bool) set + notify. consumer は `doDeferredPublish = publishRetryReady; publishRetryReady=false` (RebuildDispatch.cpp:889-890) で **1 回のみ** consume → 1 admission。predicate `publishRetryReady` が false になるまで再睡眠。二重 consume 不可 (View state jassert cpp:678)。
- **Case B (wake → deferred clear → consume no-op):** `requestDeferredClear()` (Timer.cpp:1662/1829/1849, C2/C3/C4) → `deferredClearRequested_=true`(release, Orch:591) + `notify_one`(592) → RebuildThread wake → `drainDeferredClearIfRequested()`(905-906) → `clearDeferredForShutdown()`(604→546): `slot.reset()(555)` + `hasDeferred_=false`(556) + `invalidateDeferredObligation()`(560) → 次の `processDeferredAdmission`: `hasDeferred_==false`→return(717-718) / `peekDeferred()`→`nullopt`(615-616) → **no-op**。latch は毎 wake drain (905) → lost clear なし。
- **Case C (wake → shutdown):** RebuildDispatch.cpp:861 `if(rebuildThreadShouldExit) break`; 862-868 `if(isShutdownInProgress){ publishRetryReady=false; break; }` (no admit)。runtime: `evaluateDeferred ctx.shutdown → ShutdownDiscard`(Pub:74-75)→discard→finishView→invalidate。double-consume impossible。
- **Case D (wake + recovery wake provenance mixing):** exchange under lock single-shot (cpp:888), `wasRecoveryWake` は coalescing wake 全体で 1-bit。ordinary producers (fade 1017, watchdog 293) は `recoveryRetryReady` を**絶対**書かない (Timer.cpp:1007 verbatim comment)。recovery は両フラグを同一 lock 下に set (1768-1769)。→ provenance 極薄なし。coalesce 時は保守的に recovery 側 (budget reset → fresh dwell) 取り扱い — F4 承認済み。

---

## D-11. `kMaxDeferredRetries` 意義 — PASS

- `kMaxDeferredRetries` **存在** (h:300, `= 2`)。**未削除**。
- `retention では increment されない` : D-1 で `++` 0 件 / `count_ =` 0 件 (reset-only) 証明。→ `RetryExhaustedDiscard` は retention 経路 **未到達** (RuntimePublicationState.h:16-22 dormant)。
- **判定 `>` (canonical)**: cpp:497 `deferredRetryCount_ > kMaxDeferredRetries` (F5-9/F6-6 確定)。<`=` 不在 (grep `>= kMaxDeferredRetries` → 0)。
- **future Type-A contract**: h:292-295 / cpp:495-496 / RuntimePublicationState.h:16-22 に dormant guard + enum で文書済み。"2回で discard するか" をテストしていない — retention≠retry の分離自体を検査 (D-11 再定義通り) 。

---

## D-12. state-machine consistency map (line numbers)

```
PublishRequest(G,O)
       │  enqueuePublicationIntentForRuntimeCommit (Commit.cpp:808-813; RebuildDispatch.cpp:1040/1120/1326)
       ▼
submitPublishRequest(req)  RuntimePublicationOrchestrator.cpp:357
  trySubmitImpl → Decision
       │
 ┌─────┼────────────────────────────┐
 │     │                            │
 ▼     ▼                            ▼
Accepted│  cpp:375-376                  Deferred│ cpp:377-378 -> enqueueDeferred (cpp:437, retention accounting :470-511)
         (Consumed via processDeferredAdmission:727-731 consume→finishView:637-662)        │
         │  post-check invalidate(745-746) ← hasDeferred_==false                           │
         ▼                                                                                    │
Success  (onPublishCommitted :343, seq recorded :348-349)                                  RETAIN
         │                                                                                   │
         │                                                                                   ▼
 │       │ ┌───┴──────────────┐
 │       │ │                  │
 │   fade-complete (Timer.cpp:931→1011-1020, wasRecoveryWake=false, PubRetryReady only)  watchdog (Threading.cpp:288-295, wasRecoveryWake=false)
 │       │ │                  │
 │       └──────┬────────────┘
 │          ▼
 │     RebuildDispatch.cpp:914 processDeferredAdmission(wasRecoveryWake) :700
 │          │
 │     ┌────┼─────┐
 │     │    │     │
 │    TTL   Gen   Seq  (evaluateDeferred PublicationAdmission.cpp:69-91; order Shutdown→TTL→Gen→Seq→Ready)
 │     │
 │   Discard (StaleDiscard/TTL→Pub:80; Gen→Pub:83; Seq→Pub:87) -> view->discard(:736)->finishView->invalidate(:653-654)
 │
 ▼
terminal: invalidateDeferredObligation(:666-672) {gen=0,id=0,count=0,createdAt=0}; slot.reset(:647); hasDeferred_=false(:648)
         │
         ▼
retire (submitPublishRequest:375-376 Accepted → DSPTransition/retire in Execution tail :332-336; Recovery DSP retire via destroyRolledBackDSP)
```

**per-arrow ledger map:**
- `enqueue → submit`: identity `(G,O)` stamped (Commit.cpp:808/813 / RebuildDispatch.cpp:1036-1042); `createdAt=now` at new identity (cpp:477-481); `count=0` (cpp:480)。
- `Deferred → RETAIN`: retention, key/createdAt/count **unchanged** (cpp:483); metadata derives createdAt (cpp:528)。
- `→ fade-complete`: prov `publishRetryReady only`, `wasRecoveryWake=false` (Timer.cpp:1007,1017; RebuildDispatch:888)。
- `→ watchdog`: prov `publishRetryReady only`, `wasRecoveryWake=false` (Threading.cpp:293; RebuildDispatch:888)。
- `evaluate`: Shutdown→TTL(ageUs>ttlUs, Pub:79)>>Gen(Pub:83)>>Seq(Pub:87)>>Ready(Pub:90); `m` = metadata w/ obligation createdAt (cpp:724-725,D-4).
- `TTL/Gen/Seq → Discard`: `discard(reason)`→`finishView`→`invalidate` (cpp:734-738→637-662,653-654)。
- `Ready → consume`: move req + `finishView` (cpp:676-683) → `submitPublishRequest` (cpp:731).
- `→ Accepted`: post-check `!hasDeferred_`→`invalidate` (cpp:745-746): **full identity+ts+count invalidate**。
- `→ terminal`: 4-reset sites (clearDeferredForShutdown:560, finishView:654, resetDeferredRetryBudget h:173-179/@:716, post-check:746) — all zero gen,id,count,createdAt。
- `retire`: non-recovery Accepted → Execution tail retire (cpp:332-336); RejectedPublishFailure DSP → `destroyRolledBackDSP` (h:273-277 comment, cpp:420-421 note, RuntimePublicationOrchestrator.cpp:304-305)。

---

## Gate D 判定 — PASS

| # | criterion | verdict | cite |
|---|---|---|---|
| 1 | retention で retry count が増えない | ✅ `++deferredRetryCount_` 0件, `=` sites 皆 `=0`/reset | cpp:480/670, h:177 |
| 2 | identity = (generation, recoveryObligationId) | ✅ tuple compare & paired writes | cpp:475-476,478-479 |
| 3 | 同一 identity で createdAt 不変 | ✅ `!sameObligation` else hold | cpp:483 comment |
| 4 | identity 変更時のみ createdAt 更新 | ✅ same | cpp:477-481 |
| 5 | TTL が obligation dwell を測定 | ✅ ageUs = now − metadata.enqueueTimestampUs(=createdAt) | Pub:78-79, cpp:528 |
| 6 | terminal で metadata 完全 invalidate | ✅ 4-site zero-reset | cpp:560/654/666-672/746 |
| 7 | recovery wake のみ budget reset | ✅ resetDeferredRetryBudget 唯一 call = cpp:716 `(wasRecoveryWake)` | cpp:715-716 |
| 8 | fade/watchdog が ordinary provenance | ✅ publishRetryReady only, recoveryRetryReady untouch | Timer.cpp:1007/1017, Threading.cpp:293 |
| 9 | watchdog が poll churn を再発させない | ✅ counter-only on usual tick, notify は threshold のみ; 旧毎.tick set removed | Threading.cpp:284-301 |
| 10 | kMaxDeferredRetries が retention から切離 | ✅ exists h:300; `=2`; retention increment 0; 到達不能 | h:300, cpp:497, RuntimePublicationState.h:16-22 |

**10/10 PASS.** → Gate D = PASS → Gate E 進行可。
