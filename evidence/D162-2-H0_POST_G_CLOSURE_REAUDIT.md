# D162-2-H0 — Post-G Closure / Residual Open-Item Re-audit（read-only）

```text
Date:        2026-09-04
Type:        read-only audit / production source 変更 0 / Test 0 / Build 0 / CTest 0
Baseline:    ConvoPeq.md Generated 2026-09-04 20:28:50（G2 適用済み working tree・G3/G4 で変更 0 のまま）
             ※ 9/4 18:29:10 版に対し G1/G2 分が反映された現行最新 snapshot。唯一の production baseline とする。
Position:    D162-2-G4 PASS 後の正式 closure 監査。次期実装の根拠を「9/1 inventory の記憶」ではなく
             現行 source から再確定する。
参照:        doc/work88/PLAN_DOCS_UNIMPLEMENTED_INVENTORY_20260901.md（9/1 07:20 inventory）/
             doc/work88/ND01-04_CW8_*.md / doc/work88/ND07_CRALPHA_RETRY_CONTRACT_REPORT.md /
             doc/work88/CRALPHA1-6_*.md / doc/work88/CRBETA0_RESIDUAL_TRIAGE_REPORT.md /
             evidence/D162-2G0..G4（G シリーズ全証跡）
```

---

## 0. 総合判定

> ## **H0 = PASS / GO**
>
> 1. **D162-2-G の authority 統一は現行 source に完全に残存**（静的 closure 証明成立）。
> 2. **INV-D162-1〜9 の新規違反なし** — 全 anchor が現行 source に実在し、G4 の実測と矛盾なし。
> 3. **9/1 inventory の「未実装 2 系統」は現行 source ではいずれも未実装ではない**:
>    CW-8 = **実装・検証済み**（ND01-04 / `PublishedWorldObservation` + `observePublishedObservation`
>    + T-CW8-1〜7）、BuildError retry = **CR-α として実装・CLOSED**（CRALPHA1-6 /
>    `warmupRetryDecision` + exponential backoff 10→80ms + Exhausted one-shot telemetry）。
>    → **CW-8 を新規 CR にすることは不可（本監査で正式に「実装済み」へ更新）。**
> 4. **新たな authority / lifetime violation は検出されなかった**（direct destroy caller・
>    authority 外 retire・stale map・EBR ownership ambiguity の全走査で 0 件）。
> 5. 次の実装対象は **OPEN 0 件**。残置は DEFER（trigger 待ち）または観測のみの
>    residual register 項目。→ **新規 CR 起こしは不適・Phase-I は現行構成で closure。**

---

## 1. Baseline

- 基準: `ConvoPeq.md Generated: 2026-09-04 20:28:50`（行番号は実ファイル現行のものを併記）
- G4 完了後の working tree に追加の production 変更なし（git diff = G2 状態と同一・6 files / +216 −71 は G1/G2 + E 系の既存分）
- 本監査での production / test / CMake / build 変更: **0**

## 2. D162-2-G closure verification（H0-1）

### 2.1 S3（shutdown clear disposition）

| 項目 | 現行 source 実測 | 判定 |
| --- | --- | --- |
| authority retire | `RuntimePublicationOrchestrator.cpp:630-633` `retireRegisteredDSP(req, "shutdown-clear")` | ✓ |
| 実行順序 | `deferredSlot_.reset()`（:634）の**前**に retire block | ✓ |
| DIAG | `event=CLEAR`（文言修正済み・`no disposition` 表記なし）+ `event=CLEAR_SHUTDOWN_DISPOSITION` | ✓ |

### 2.2 V-D（VerifyDrained 最終 active/fading）

| 項目 | 現行 source 実測 | 判定 |
| --- | --- | --- |
| active | `AudioEngine.Processing.ReleaseResources.cpp:543` `lifetimeMgrForFinalDSP.retire(activeDSPToDestroy)` | ✓ |
| fading | `:552` `retire(fadingDSPToDestroy)`（`!= activeDSPToDestroy` ガード付き） | ✓ |
| **V-D の `destroyRolledBackDSP` caller** | **0 件**（ReleaseResources.cpp 内はコメント 1 件のみ） | ✓ |
| `if (false && ...)` | **src 全走査 0 件** | ✓ |

### 2.3 authority / EBR 単一路線

| 項目 | 現行 source 実測 | 判定 |
| --- | --- | --- |
| `retireRegisteredDSP` 全 call site | 9 箇所 = S4 ×4（Orchestrator.cpp:396/:416/:433/:444）+ S1（:503）+ dormant（:536）+ S3（:630）+ E-4d（:700）+ S2（:889）— 全て authority 経由 | ✓ |
| map erase 位置 | `AudioEngine.h:4367`（`retireDSPHandleForRuntime` 内 = authority 内のみ） | ✓ |
| EBR enqueue → reclaim → destroy | `DSPLifetimeManager::retire` → `enqueueWithRetry(dsp, destroyDSPCoreNode, epoch)` 単経路（DSPLifetimeManager.cpp:124-161） | ✓ |
| `destroyRolledBackDSP` 正当 caller | rollback 経路のみ（Orchestrator.cpp:292 — publish 失敗 rollback・registry rollback済み） | ✓ |
| その他の direct destroyDSPCoreNode caller | DSPGuard 契約のみ（RebuildDispatch.cpp:1036/:1118 — 未コミット recovery DSP・**未登録** = EBR 保護不要の既存正当契約。INV-D162-1 の規律対象外） | ✓（新規 caller なし） |
| authority 外 retire | なし — `DSPLifetimeManager::retire/retireByHandle` caller は Timer（4）・RebuildDispatch（1）・CtorDtor（2+pendingTask）・Threading quarantine（1）・ProcessIntent（1）・ReleaseResources V-D（2）の全てが想定 site | ✓ |

### 2.4 G4 実測との整合

静的構造（上記）と G4 の動的実測（60-gen × 2: exit 0x0 / dump 0 / residual 0 / generation 1:1 /
E-4 収支 / stale MISS / direct destroy 0）は完全に一致。**G4 evidence と現行 source の不整合なし。**

## 3. INV-D162-1〜9 status（H0-2）

| INV | 内容 | 現行 source anchor | 状態 |
| --- | --- | --- | --- |
| 1 | registered DSP terminal = 単一 authority | `retireRegisteredDSP` 9 site → `DSPLifetimeManager::retire`。T2 direct は未登録 DSP（DSPGuard / rollback）のみ | **成立** |
| 2 | eventual destruction（orphan 禁止） | 上記 authority + G4: 構築 gens と破壊 gens の 1:1（両 run）・residual 0 | **成立** |
| 3 | 二重処分禁止 | map erase 先行構造（h:4367）+ resolve nullptr no-op + G4 duplicate-gen 0 件 | **成立** |
| 4 | RT 不変 | G1〜G4 の変更は Orchestrator / ReleaseResources の NonRT path のみ | **成立** |
| 5 | published 破壊は epoch-safe | published old DSP の既存 EBR 経路は未改変・S1-S4/V-D/S3 は未 publish or quiescence 証明後 | **成立** |
| 6 | member teardown 中の reclaim 禁止 | `drainForShutdown`（Cache.cpp:169・CtorDtor 呼出）+ `~CacheMap` no-op | **成立** |
| 7 | dangling slot 値の map lookup 禁止 | dtor は `retireByHandle`（CtorDtor.cpp:208/:211・generation 検証付き）のみ | **成立** |
| 8 | shutdown 破壊は EBR 単経路 + dtor body 内 drain | E-3 assert（CtorDtor.cpp:293-301）+ G4 両 run の最終 destroy が `~AudioEngine: enter` 後 | **成立** |
| 9 | deferred slot 出口の全数カバレッジ | DIAG 7 イベント（CREATE/CONSUME/DISCARD/OVERWRITE/CLEAR/CLEAR_MIDRUN/CLEAR_SHUTDOWN）+ G4 収支 2 run 完全 | **成立** |

**新規違反: 0 件。**

## 4. S3 standalone retired=1 の分類（H0-3）

```text
S3 block 到達            = 実証済み（G1-G4 累積 14 回）
S3 no-op 帰還            = 実証済み（全 14 回・INV-D162-3 防護動作として正しい）
S3 standalone retired=1  = 未観測
```

- 通常 production path では Timer C2/C3/C4 → `requestDeferredClear` → latch → RebuildThread
  `drainDeferredClearIfRequested` → **E-4d が必ず先行 disposition** するため、S3 block 到達時点で
  map erase 済み → no-op が正しい挙動。
- EmergencyDrain / C1 経由で S3 が retired=1 になるには「RebuildThread 停止後に deferred 保持
  DSP が残留する」production 到達不能な異常系が必要。
- **分類: 「未観測 ≠ 欠陥」→ residual risk register（観測のみ・対応不要）に登録。**
  人工的異常系の生成は実施しない（指示どおり）。将来この変異を実測する場合の手がかり:
  emergency drain 要求 + RebuildThread 停止後の deferred 残留を同時に成立させる
  harness 異常系テスト。

## 5. 2026-09-01 inventory delta（H0-4）

9/1 07:20 の inventory（`PLAN_DOCS_UNIMPLEMENTED_INVENTORY_20260901.md`）は「未実装 2 系統」を
報告したが、**同日中に両系統とも実装/決着が完了している**（inventory 時点から状況が進展）。
本監査で現行 source に照合し直した結果:

| inventory 項目 | 9/1 時点 | その後の工程 | 現行 source（9/4 20:28:50）実測 | 分類更新 |
| --- | --- | --- | --- | --- |
| **CW-8** PublishedWorldObservation pair-snapshot | src 0 件・設計のみ | **ND01-04**（07:49-10:32）で実装・検証 | `RuntimeWorldAuthority.h:90-109` 型（private ctor + friend 構造遮断）・`:240-248` factory `observePublishedObservation(ReadToken)`（`observe()` **1 回 acquire load** で `{world, &world->publication}` 同時確定）・`testCW8_PublishedWorldObservation`（ISRSemanticValidationTests.cpp:930・T-CW8-1/2/3/4/6/7・runner :2607 接続） | **実装済み・CLOSED**（新規 CR 不適） |
| **CR 候補 α** BuildError retry backoff / count / telemetry | `schedule(req, 0ms)` delay 0 固定・backoff 未実装 | **ND07 → CRALPHA1-6**（11:01-13:19）で実装・CLOSED（15/15 closure 条件成立） | `BuildErrorPolicy.h:92` `kMaxWarmupConsecutiveRetries=3`・`:97-104` `RetryBackoffPolicy{10,80,2}`・`:109` `retryBackoffDelayMs`（saturation 付き exponential）・`:143` `warmupRetryDecision`（純関数）・`RebuildDispatch.cpp:1272-1312` 呼出 site（`schedule(req, decision.delayMs)` — **delay=0 固定は解消**）+ Exhausted one-shot terminal telemetry（:1304-1312） | **実装済み・CLOSED**（CR-α-6 closure） |

**CRBETA0（9/1 13:33 residual triage）の判定と現行 source は一致:**
- 候補 A（CW-8）= ALREADY COVERED（枠組み実装 + production caller 未接続は保守的構成）
- 候補 B（build-error telemetry 強化）= DEFER / NO TRIGGER

**→ 9/1 inventory の「新規 CR 候補 2 系統」は、いずれも現行 source を根拠にした CR 起こし対象から除外。**

## 6. CW-8 current status（H0-4 補足）

- 型: `PublishedWorldObservation` — non-owning borrow pair。private member + private ctor +
  friend class RuntimeWorldAuthority により aggregate init / 独立 pair 捏造を型レベルで遮断
  （T-CW8-7 が compile-time で表明）。
- factory: `observePublishedObservation(const ReadToken&)` — 単一 `runtimeStore_.observe()`
  acquire load のみから pair を形成（二段 read / 別 atomic 不使用）。
- identity: `&world->publication`（同一オブジェクト内部 pointer・独立 storage 不存在）—
  CW-5 / INV-X4-6/7/A と構造的に同根。
- production caller 接続: 未接続（既存 `observePublishedWorld` / `consumeWorldHandle` 経路が
  現用）— CRBETA0 が「保守的構成」として正常と判定済み。強制接続は不要。
- **結論: CW-8 = 実装済み・テスト済み。inventory を「実装済み」に更新して closure。**

## 7. BuildError / Retry current status（H0-5）

### 7.1 全経路追跡（現行 source）

```text
BuildError classification
    ↓
classifyBuildError（BuildErrorPolicy.h:66・constexpr descriptor table §1.8.10.3）
    ├─ Site: build failure（RebuildDispatch.cpp:1197）→ 分類を diagLog 記録 → continue
    │        （build failure の retry は recovery 層の obligation 契約が管理・Site 3 scope 外）
    └─ Site: warmup failure（:1273）
         ↓ ++warmupRetryCount（failure 回数・schedule 前に ++）
warmupRetryDecision（:1274 / BuildErrorPolicy.h:143・純関数）
    条件（AND）: contextRetryable AND !obsolete AND attempt ≤ 3 AND disposition ≠ NoRetry
    ├─ Schedule: schedule(req, decision.delayMs)（:1294）
    │     delayMs = RetryBackoff → 10/20/40/80ms（exponential・saturation 80ms 上限）
    │              RetryImmediate → 0ms（無条件 retry ではない・bounded）
    │     schedule reject（capacity 満杯 / shutdown）= attempt 消費の retry drop（巻き戻しなし・ND-07 §7）
    ├─ Exhausted: one-shot terminal diagLog（:1304-1312・exhausted flag で再発火なし）
    └─ NoRetry: telemetry なし
    fallback: scheduler 未生成の異常系では submitRebuildIntent 直接（:1296-1302）
```

### 7.2 9/1 記載からの変化

| 9/1 inventory 記載 | 現行状態 |
| --- | --- |
| `schedule(req, 0ms)` delay 0 固定 | **解消** — `decision.delayMs`（backoff 10→20→40→80ms / RetryImmediate 0） |
| exponential backoff 未実装 | **実装済み**（`retryBackoffDelayMs`・`RetryBackoffPolicy{10,80,2}`） |
| retry count / exhaustion telemetry 未実装 | **実装済み**（counter 意味論 + Exhausted one-shot terminal log・attempt 付き逐次 log） |
| `RetryBackoffPolicy` src 0 件 | **実装済み**（BuildErrorPolicy.h:97-104・単体 test 10 箇所: BuildErrorClassificationTests.cpp） |
| retry telemetry（`retryCount` 集約 / `retryLatency` / `retryStormDetected`） | 未実装 — **CRBETA0 候補 B が DEFER / NO TRIGGER 判定済み**（diagLog で復元可能・`rejectCount_`/`pendingCount()` 実装済み） |
| `buildErrorCount_` | 未実装（RuntimeBuilder.h:124 コメントのみ）— **同上 DEFER** |
| MKLFailure / ConvolverFailure / PrepareFailure の生成経路 | **依然なし**（RuntimeBuilder.cpp:417/:443/:448 は InvalidInput/ResourceUnavailable/InternalError のみ・validateWarmup は WarmupFailed のみ・enum/table/test のみ存在）— **CRBETA0 B-3 が DEFER 判定済み**（実 failure 観測 + subsystem 別 retry 判定が必要になる設計確定時が trigger・RuntimeBuilder.h 既決監査 E-NEXT-6 / Phase D2-0 NO-GO） |

### 7.3 判定

- Site 3 warmup retry 契約（backoff / count / exhaustion telemetry）は **CR-α として CLOSED**。
- 残る sub-item（telemetry counter 集約・MKL/Convolver/Prepare 生成経路）は **DEFER / NO TRIGGER**
  （CRBETA0 判定を現行 source で再確認・変化なし）。
- **「9/1 時点で未実装だった」ことを理由にした CR α 起こしは禁止どおり不発。**

## 8. Remaining OPEN items

**現行 source に基づく OPEN（実装待ち・trigger 成立済み）項目: 0 件。**

## 9. DEFER items（trigger 待ち・着手不適）

| 項目 | 凍結根拠 | trigger |
| --- | --- | --- |
| build-error telemetry counter 集約（`buildErrorCount_` / `retryLatency` / `retryStormDetected`） | CRBETA0 候補 B DEFER | long-running 統計の実要件発生時 |
| MKLFailure / ConvolverFailure / PrepareFailure 生成経路 | CRBETA0 B-3 + RuntimeBuilder.h 既決監査（E-NEXT-6 / Phase D2-0 NO-GO） | convolver/prepare の実 failure 観測 + subsystem 別 retry 判定の設計確定 |
| I4 Phase-II（RecoveryEpisodeId / Supersession / E_max 等） | D159 D2/D3 凍結 | Phase-II 開始判断 |
| dash2 §5.14 ConvolverFailure/PrepareFailure 生成 | inventory 1-C-1 と同根 | 同上 |

## 10. Recommended next work

1. **新規実装 CR の起こしは不適**（OPEN 0 件・2 候補とも既に CLOSED/DEFER）。
2. D162-2-G 完了後の標準フローとして **実運用検証系の次工程**（D116 系 operational validation
   の再実施・長時間 soak / device cycle 等）を推奨。S3/V-D authority 化の効果は
   shutdown 時 DSP 破壊の確率的 AV class の消滅として運用中に観測されるはず。
3. 残置観測（residual register）: S3 standalone retired=1（§4）・Release CTest AudioEngineHarness
   pre-existing 0xC0000374（別 track）・cli-smoke-test.ps1 の build-icx 旧 binary 自動選択
   （dump 混入防止の候補リスト更新は低コスト改善候補だが trigger 判断は運用側）。
4. `ConvoPeq.md` は 20:28:50 版が FRESH（G3/G4 で source 変更なし）— 再生成不要。

## 11. GO / NO-GO

| NO-GO 条件 | 該当 |
| --- | --- |
| 新しい direct destroy caller | なし（DSPGuard 既存契約のみ） |
| authority 外の retire | なし |
| stale map HIT | なし（G4 MISS 実測と整合） |
| EBR ownership ambiguity | なし |
| terminal path の orphan | なし（generation 1:1） |
| E-3 violation | なし |
| D162-2 INV violation | なし |
| G4 evidence と現行 source の不一致 | なし |
| inventory と現行 source の重大な不整合 | なし（不整合は「inventory が古い」方向のみ・本監査で解決） |

# **判定: GO — D162-2-G closure を正式確定。新規実装対象なし・Phase-I は現行構成で完了。**
