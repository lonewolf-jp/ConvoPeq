# D163 — OPEN CR Read-only Boundary / Priority Audit (CR-α / CR-β)

```text
Date:        2026-09-06
Type:        read-only audit (D163-A + D163-B) — production/test/CMake/build/build.bat/tool 変更すべて 0
Baseline:    HEAD 9cacee1f + H5 CMakeLists (+10/-0) のみ。ConvoPeq.md (09-05 生成) は stale のため
             source authority に使用しない (A0 指示どおり・src/** 生ファイルを authority とする)
判定:        **CR-α = REJECT (Case D — 既存 infrastructure が契約を充足。CR-α-1..6 として 2026-09-01
             実装・CLOSED済みでインベントリ記載は stale)。CR-β = REJECT (ALREADY COVERED —
             PublishedWorldObservation 型 + factory + T-CW8-1..7 test 実装済み・commit 0aeb22ca)。
             両 CR とも新規実装対象なし → 次の実装タスクは「なし」(インベントリ更新と運用検証へ)。**
```

---

## 0. 要旨

> インベントリ (PLAN_DOCS_UNIMPLEMENTED_INVENTORY_20260901.md) の OPEN 2 系統は、**どちらも現行
> snapshot では「未実装」ではない**。CR-α (BuildError retry backoff/count/telemetry) は
> 2026-09-01 に CR-α-1〜6 (anchor audit → implementation → verification → build → CTest 40/40×2 →
> retry source audit 12/12 → closure 15/15) として実装・CLOSED 済みであり、現行 source に
> `kMaxWarmupConsecutiveRetries=3`・`RetryBackoffPolicy{kDefaultWarmupRetryBackoff={10,80,2}}`・
> `warmupRetryDecision()`・`retryBackoffDelayMs()` が実装・配線されている (唯一の production
> call site は `schedule(req, decision.delayMs)` — delay は policy 由来で接続済み)。
> CR-β (CW-8 PublishedWorldObservation) は型 (private ctor + friend 構造遮断) + factory
> (`observePublishedObservation` — 単一 acquire load から {world, &world->publication} 同時確定) +
> test (T-CW8-1..7) が commit 0aeb22ca で導入済み (production caller 0 = 保守的休止)。
> D163 の実 work は「監査でこの事実を確定し、インベントリを更新対象として印を付けること」である。

---

# D163-A — CR-α BuildError Retry Backoff / Count / Telemetry Audit

## A0 — baseline

```text
authority            src/** 生ファイル (HEAD 9cacee1f)。ConvoPeq.md は H5 前の stale → 不使用
変更                 production/test/CMake/build/build.bat/tools = 0
先行確定報告          doc/work88/CRALPHA6_CLOSURE_REPORT.md (2026-09-01): CR-α = CLOSED / ACCEPTED
                     doc/work88/CRBETA0_RESIDUAL_TRIAGE_REPORT.md (2026-09-01)
git -S warmupRetryDecision → 0aeb22ca (CR-α 実装 commit)
```

## A1 — retry path 完全追跡 (caller → policy → scheduler → retry completion)

```text
[Site 3 = main rebuild warmup (AudioEngine.RebuildDispatch.cpp rebuildThreadLoop)]
warmup failure (runtimeBuilder.validateWarmup → BuildError::WarmupFailed 等)
  → ++warmupRetryCount (RebuildThread 関数スコープ state・generation rebind :1177-1182)
  → classifyBuildError(error) → kBuildErrorDefaultTable → BuildOutcome{classification, retry}
  → warmupRetryDecision(attempt, kMaxWarmupConsecutiveRetries=3, retryable=shouldRetryWarmupFailure
     (=isLoadingIR), isObsolete(), disposition, kDefaultWarmupRetryBackoff)   [純関数 BuildErrorPolicy.h:143]
      Schedule: disposition==RetryBackoff → retryBackoffDelayMs = 10/20/40ms (saturation 80)
                disposition==RetryImmediate → 0ms (bounded immediate)
      Exhausted: attempt > 3 → one-shot terminal diagLog (generation rebind まで発火しない)
      NoRetry:   NoRetry / 非 retryable / obsolete → telemetry なし
  → retryScheduler_->schedule(req{Structural, RebuildThreadWarmupRetry, Structural, Replaceable},
                               decision.delayMs)                      [:1294・唯一の production caller]
  → RetryScheduler worker (deadline-sorted deque・待機は worker 内) → DispatchFn
     = submitRebuildIntent(kind, reason, class, collapsePolicy)          [CtorDtor.cpp:98-100]
  → 次の rebuild cycle で再 build (completion)

[Recovery Site 1/2 (RebuildDispatch.cpp durable admission loop)]
build/warmup failure → settlePendingRecoveryAdmission(true) (DurablePending へ戻す = retry 構造保証)
  → postRecoveryFailureSignal(obligationId)                            [:1100/:1124]
  → obligation-level counter (RuntimeIntentCoordinator / kMaxObligationConsecutiveFailures=4)
     到達で ResolvedFailed (terminal)
  → Builder-local spin guard kMaxRecoveryConsecutiveFailures=4 (連続 failure で cycle break) [:1076/:1102/:1126]
markTransientFailure / rearmRecoveryRetry / redriveDeferredRecovery(Obligations) は
  RuntimeIntentCoordinator (ISRRuntimePublicationCoordinator.cpp:1153/1176/1190/1209) が権限を持つ
  (T1 producer = RebuildThread、durable state と obligation counter は独立 linearization — D105-R18)
```

**結論: A1 の全要素 (BuildErrorPolicy・RetryDisposition・classifyBuildError・K=4 ×2・RetryScheduler・
markTransientFailure・rearmRecoveryRetry・redriveDeferredRecovery・warmup/recovery 境界) は
すべて現行 source に実装・接続済み。**

## A2 — retry domain 分離証明

統合**禁止が妥当** (コード上も構造的に分離済み):

| 観点 | Site 3 warmup | Recovery Site 1/2 |
| --- | --- | --- |
| 上限 | kMaxWarmupConsecutiveRetries = 3 | obligation-level K=4 + Builder-local guard 4 |
| failure 意味 | IR loading transient (latency-sensitive・RetryImmediate) | obligation lifecycle (durable state 保持) |
| counter 所在 | RebuildThread 関数スコープ (generation rebind) | obligation-level atomic + builder-local int |
| exhaustion 意味 | one-shot terminal log・generation 変更で再開 | ResolvedFailed (terminal) |
| backoff | {10,80,2} 適用 (RetryImmediate は 0ms) | backoff なし (settle→next cycle retry) |

`BuildErrorPolicy.h:86-91` に「意図的に別値 (3 vs 4) にして grep/telemetry 読解時の混同を検出可能に
する」と明記。ND-07 §3 (絶対遵守) どおりであり、D163 で統合を検討する余地はない。

## A3 — telemetry gap 実測 (production src/**)

| 項目 | 実測 | 判定 |
| --- | --- | --- |
| buildErrorCount_ | RuntimeBuilder.h:124 の将来拡張コメントのみ・実体 0 件 | 未実装 (意図的保留) |
| retryStormDetected | 0 件 | 未実装 (契約外) |
| retryLatency | 0 件 (buildElapsedMs は build 時間計測) | 未実装 (契約外) |
| retryExhaustedCount | 実体なし (test の log 文字列/コメントのみ) | recovery は obligation counter→ResolvedFailed で表現 |
| Site 3 telemetry | diagLog: generation/attempt/limit/delayMs/error + one-shot exhausted | **実装済み** (CR-α) |
| RetryScheduler.rejectCount_ / pendingCount() | atomic 実装済み (retry drop 観測先) | **実装済み** |
| recoveryRetryDeferredCount_ / IntentDrop / ShutdownDiscard / Coalesced / CapacityExhausted | coordinator 内 atomic 5 種 + getter | **実装済み** |

意味的重複: buildErrorCount_ は既存 recovery*Count_ と母集団が別 (build failure vs obligation transport)
だが、Site 3 の failure は diagLog が error 名付きで逐次記録し、exhausted は one-shot log で復元可能。
counter 集約の欠落は「long-running 統計需要」が発生するまで実障害ではない
(RuntimeBuilder.h:120-124 の既決監査 E-NEXT-6/Phase D2-0 NO-GO 2026-08-19 が trigger 条件まで明示)。

## A4 — RetryScheduler::schedule() 全 call site audit (CR-α の核心)

```text
production call site = 1 件のみ: AudioEngine.RebuildDispatch.cpp:1294
    retryScheduler_->schedule(req, std::chrono::milliseconds(decision.delayMs));

判定:
  × 「backoff 未実装」ではない — delay は warmupRetryDecision → RetryDisposition →
    retryBackoffDelayMs(kDefaultWarmupRetryBackoff) で完全接続済み (10/20/40ms・saturation 80)
  ○ delay = 0 は RetryDisposition::RetryImmediate (WarmupFailed) の場合のみ —
    latency-sensitive な warmup retry の intentional immediate retry (bounded・counter 管理)
    = RetryBackoff との矛盾ではなく policy table の設計どおり
  ○ RetryScheduler 自体は delay を受容する設計 (deadline-sorted deque) で、caller policy は接続済み
  ○ fallback: retryScheduler_ == nullptr の異常系のみ submitRebuildIntent 直接 (immediate・既存経路維持)

→ インベントリの「schedule(..., 0ms) = backoff 未接続」記述は CR-α 実装 (2026-09-01) 以前の
  状態を反映した stale 情報であることが code-level で証明された。
```

## A5 — RT safety

```text
RT thread (audio ISR)
  → 待機しない (retry/backoff に RT は関与しない)
NonRT RebuildThread
  → warmupRetryDecision (純関数・JUCE/Scheduler 非依存) → schedule() enqueue (mutex + bounded deque)
RetryScheduler 専用 worker thread (plain std::thread・RT affinity なし)
  → cv_.wait_until(deadline) で待機 (backoff 時間はここで消化) → dispatch_ = submitRebuildIntent
  → kCapacity=8 超過 / shutdown で reject (rejectCount_++) = attempt 消費の retry drop
     (巻き戻しなし — ND-07 §7 契約・RebuildDispatch.cpp:1269-1270 に明記)
shutdown: CtorDtor StopWorkers で RetryScheduler を先に停止 (accept→clear/discard→join)
```

責務境界「RT は待機しない・NonRT scheduler が bounded delayed retry を担う」は**維持されている**。
backoff 導入済みの現行でも RT path に新規待機・新規 queue は存在しない。

## A6 — failure taxonomy gap

```text
BuildErrorPolicy.h: FailureClassification = Permanent/Transient/Infrastructure/Fatal
                    RetryDisposition     = NoRetry/RetryBackoff/RetryImmediate
production 生成経路 (grep histogram, tests/policy 除外):
  BuildError::None ×5 / WarmupFailed ×1 (RuntimeBuilder.cpp:458 validateWarmup)
  / ResourceUnavailable ×1 (:443) / InvalidInput ×1 (:417) / InternalError ×1 (:448)
  MKLFailure / ConvolverFailure / PrepareFailure = 生成 site 0 件 (enum+toString+table のみ)
RuntimeBuilder.h:120-124 に既決監査記録:
  「MKLFailure/ConvolverFailure/PrepareFailure は enum+toString のみ (保険分類)。
   prepare→build→caller→publish の伝播は現行 4 failure モードで semantic loss なし。
   PrepareResult 導入は現行の利益ゼロ。将来 subsystem 別 retry 判定が必要になる設計確定時のみ
   最小 wiring (status 伝播 + caller failure check 強化 + buildErrorCount_ telemetry) で対応」
→ D163 では failure generation を実装しない (指示どおり)。
```

## A7 — CR-α 判定

> **Case D — existing infrastructure already satisfies contract。**
> backoff ({10,80,2} + retryBackoffDelayMs)・count (warmupRetryCount + generation rebind + one-shot
> exhausted)・telemetry (diagLog 逐次 + one-shot terminal + rejectCount_) はすべて実装・gate 済み
> (CR-α-1..6: implementation PASS / verification V1-V16 PASS / CTest 40/40×2 PASS / retry source
> audit 12/12 PASS / closure 15/15 PASS — 2026-09-01 CLOSED)。
> **CR-α を実装する価値 = 現時点でなし** (既に実装済みのため)。residual の buildErrorCount_ 集約
> counter は observability-only で trigger 条件付き DEFER (既決監査どおり)。

---

# D163-B — CR-β CW-8 PublishedWorldObservation Audit

## B0 — baseline

A0 と同一。先行確定: CRBETA0 (2026-09-01) = 「候補 A (CW-8) = ALREADY COVERED」— 本監査で
code-level に再検証した。

## B1 — PublishedWorldObservation provenance

```text
src/audioengine/RuntimeWorldAuthority.h:
  :73-107   class PublishedWorldObservation — private member (world_/identity_) + private ctor +
            friend RuntimeWorldAuthority → aggregate init / 独立 pair 捏造を構造的に遮断
            (world(): const RuntimeState* / identity(): const PublicationSemantic*)
  :239-249  observePublishedObservation(const ReadToken&) — member function template
            (RuntimeState 完全型を呼び出し点で要求) → runtimeStore_.observe() 1 回 acquire load のみ
            から {world, &world->publication} を同時確定。world==nullptr → {nullptr, nullptr}
導入 commit: 0aeb22ca (git -S PublishedWorldObservation / observePublishedObservation)
production caller: 0 件 (休止 — read path 不変の保守的構成)
test: ISRSemanticValidationTests.cpp:916-1066 testCW8_PublishedWorldObservation (T-CW8-1..7・5 call sites)
```

## B2 — RuntimeStore read path

```text
単一物理 source = RuntimeStore::current (INV-X4-A/B — 旧 currentWorld_ は CW-3c で削除・実参照 0)
read API (RuntimeWorldAuthority):
  observePublishedWorld() / consumeWorldHandle() → runtimeStore_.observe() (単一 acquire load)
production callers of observePublishedWorld:
  AudioEngine.Publication.cpp:68 (oldWorld = retire 対象) / RuntimePublicationOrchestrator.cpp:120,
  :202 (spec.currentRuntimeWorld / oldWorld) / AudioEngine.h:1173 (epoch 読み) / :3658-3666 (facade)
identity は別 store に存在しない — RuntimeState::publication のみ (bake-before-swap・INV-X4-6)
```

## B3 — identity/world consistency

三層で成立 (CRBETA0 A-2 の検証を code-level で再確認):

1. **型構造**: {world N, identity N+1} 混在ペアは型システム上生成不能 (T-CW8-7 = compile-time 表明)
2. **単一物理 read**: factory は observe() 1 回のみ (二段 read / 別 atomic は :225-227 で禁止明記)
3. **identity 導出**: `&world->publication` は同一オブジェクト内部 pointer — CW-5
   (RuntimeStore::current.identity == RuntimeState::publication.identity) と構造的に同根 +
   INV-X4-6/7/A (bake-before-swap・単一 read source)

## B4 — atomicity requirement

**充足済み**。単一 acquire load で {world, identity} が同時確定する read-contract は
`observePublishedObservation` 内に実装済み。新 atomic / 新 wrapper / topology 変更は不要
(かつ禁止どおり追加していない)。production caller が存在しないため接続も不要 (consumer 出現時のみ
call site で instantiation — 完全型要求はその時点で自然に満たされる)。

## B5 — invariant impact

observation は非所有 borrow で publish/retire に参加しない (:81-89 契約明記)。新 authority・
新 atomic・read topology 変更なし。INV-X4-1..C / INV-ISR-06/07 / CW-5 に触れない。
production caller 0 = 「読む API が増えない」保守的状態。

## B6 — minimal implementation boundary

**実装境界 = なし (既に境界内で完結)**。将来の consumer 接続時に必要なのは call site 追加のみ
(ObservePathSingleSource 契約の read-side 強化として既存 read API と同一制約)。contract 変更・
RuntimeStore 変更・Authority 変更は不要。

## B7 — CR-β 判定

> **ALREADY COVERED — 実装済み (型 + factory + T-CW8-1..7・commit 0aeb22ca / ND-01..04・
> CRBETA0 triage 2026-09-01 も同結論)。** インベントリの「単一 acquire load 保証は未実装」記述は
> ND-01..04 作業前の状態を反映した stale。新規 CR 起票の必要なし。

---

# Final — D163 総合判定

| 項目 | 判定 |
| --- | --- |
| **CR-α** (BuildError retry backoff/count/telemetry) | **REJECT** — Case D: 既存 infrastructure が契約を充足。CR-α-1..6 (2026-09-01) で実装・CLOSED済み。インベントリ記載は stale |
| **CR-β** (CW-8 PublishedWorldObservation) | **REJECT** — ALREADY COVERED: 型+factory+test 実装済み (0aeb22ca)。production caller 未接続は保守的構成 |
| **priority ordering** | 実装候補としての順位付け対象 = 0 件。2 系統とも「OPEN から削除 (インベントリ更新)」が正しい後処理 |
| **exact next implementation task** | **なし**。次の実 work は (1) インベントリ更新: CR-α → CLOSED (CR-α-6 参照) / CR-β → ALREADY COVERED (ND-01..04 + CRBETA0 参照) の注記追記、(2) 運用検証系 (D116 系 operational validation) への復帰。将来実装候補は D159 冻結中の Phase-II recovery episode 層のみ (trigger gated・着手不適を再確認) |

### 監査で確定した補足事項 (棚卸し)

1. **buildErrorCount_ 集約 counter** — 唯一の残存 residual。observability-only・trigger =
   「subsystem 別 retry 判定の設計確定」または「長時間運用での統計実需要」
   (RuntimeBuilder.h:120-124 既決監査 + CRBETA0 B-7 DEFER と二重記録済み)。CRBETA0 の
   「freeze register 補助 trigger 登録 (次の編集ウィンドウ)」は未実施のまま — 次回編集 window の
   doc-only 作業候補。
2. **インベントリ日付の教訓**: PLAN_DOCS_UNIMPLEMENTED_INVENTORY_20260901.md の OPEN 2 系統は
   同日内 (16:20→19:31) の ND-01..04 / CR-α 作業で陳腐化していた。以後の棚卸しは
   「inventory 日付 < 対象 report 日付」の確認を必須とする (D163 の起点になった事例)。
3. **禁止事項遵守**: production/test/CMake/build/build.bat/tool 変更 0。新規 RetryBackoffPolicy 作成 × /
   retry domain 統合 × / MKLFailure 等の生成実装 × / PublishedWorldObservation 実装 × /
   D159 Phase-II × / I3-4-J × / I3-4-K-impl × — すべて実施せず。

## 添付 evidence (evidence/D163/)

```text
D163_OPEN_CR_BOUNDARY_PRIORITY_AUDIT.md   本書
d163_a0_freeze.json                        A0 baseline freeze
d163_a3_telemetry_census.txt               A3 telemetry census 表
```
