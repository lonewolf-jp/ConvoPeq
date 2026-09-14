# work94 — Governance Expiry Batch Re-approval（allowlist/trigger-policy 2026-08-31 期限切れ一括再承認）

- **日付**: 2026-09-14
- **Type**: governance data re-approval（JSON のみ・検査スクリプト/production code 変更 0）
- **発見元**: standard PUSH GATE（work93 check 同期後）の `isr-verify-trigger-symbol-usage.ps1`、
  続いて `isr-verify-trigger-policy.ps1`（5 violations）
- **原則**: work93 と同一 —「実装・契約が正しく、免除/期限データだけが古い」場合はデータ側を再レビューで
  同期する。production code を旧免除形へ戻す方向は禁止。
- **対象限定**: 2026-08-31 失効分 + trigger-symbol-allowlist の top-level expiry
  （旧 2026-09-30・rules と同一レビュー単位のため一括）を再承認（2026-12-31・四半期レビュー）。
  その他の未失効（2026-09-30 以降）top-level 期限は**一切変更しない**（将来のレビューは別サイクル）。

## 1. 対象 6 ファイルと根拠

`.github/isr-trigger-symbol-allowlist.json`（BRIDGE-SYMBOL-GOV-001）:
- rules 13 条すべて expiry = **2026-08-31**（gate 実行時点で約 2 週間失効済み → 失効は check 仕様上無条件 violation。13 violations）

word-boundary 監査（usage check と同一 semantics: `grep -rnw`、scan root = src/audioengine）:

| symbol | src/ 内 \b ヒット |
|---|---|
| runtimePublicationCoordinator_ / RuntimePublicationCoordinator::create / RuntimeExecutionView / getActiveDSP / resolveActiveDSPFromRuntimeWorldOnly / resolveFadingDSPFromRuntimeWorldOnly / exchangeFadingOutDSP / getFadingDSP / fadingDSP_ / activeDSP_ / getRuntimeExecutionViewForAudioThread / getRuntimeExecutionViewForControlThread（12 条） | **0 件**（移行完了 = 退行ゼロを再確認） |
| activeDSP | 8 件・**全件が pathRegex 許容ファイル内**（Timer.cpp accessor 結果ローカル / RuntimeBuilder / Orchestrator / DSPTransition.h）、範囲外 0 件 |

- 当初案「死んだ免除 12 条を削除」は**不採用**。`isr-trigger-audit.ps1` が usage レポートの
  symbolStats に activeDSP / runtimePublicationCoordinator_ /
  RuntimePublicationCoordinator::create / RuntimeExecutionView の entries を必須要求
  （retire facade metrics 証拠源）しており、rule 削除は監査証拠系を壊す（実測で audit throw を確認）。
- 採用:**13 条すべて存続 + expiry 再承認（2026-12-31・四半期）**。rationale に再レビュー記録を追記、
  pathRegex・owner・issue は不変。監視 anchor（範囲外使用 = 即 blocked）の設計は同一。

その他の 2026-08-31 失効バッチ（同一 owner ブロック BRIDGE-* / 同一四半期サイクル）:

| ファイル | 再承認した期限項目 | targeted 検証 |
|---|---|---|
| .github/isr-trigger-policy.json | entries 5（activeDspDeletionStart / fadingOutDspDeletionStart / retireFacadeRemovalStart / observeShimRemovalStart / runtimeExecutionViewConvergence） | **PASS** |
| .github/isr-clang-tidy-rule-registry.json | top-level expiry 1 | **PASS**（clang-tidy-readiness） |
| .github/isr-flag-dependency-graph.json | top-level expiry 1 | **PASS** |
| .github/isr-metric-governance.json | metrics 4（xrunDelta / callbackJitter / retireLatency / crossfadePeak） | **PASS** |
| .github/isr-rollback-compatibility-matrix.json | subsystemFlags 3 + compatibility 3（計 6） | **PASS** |

## 2b. isr-ai-governance-policy.json — v7.3 admission funnel の policy data 未登録 2 件（同期）

standard gate 次段 `isr-verify-v73-admission-funnel.ps1` violations=2。blame 確定: 両件とも
commit 335240470（D167・2026-09-07・本セッション外）で導入済み・policy データ側が未追従。

| checkId | 内容 | 施置（policy JSON のみ） |
|---|---|---|
| CI-ADMISSION-004 | RebuildDispatch.cpp:327 `RebuildTelemetryReason::AdmissionClosed`（D167 AdmissionClosed telemetry）が suppressionReasonAllowlist 未記載 | list に `AdmissionClosed` 追加 |
| CI-ADMISSION-001 | PublishPipelineIntegrationTests.cpp:1039 `e.requestRebuild(Structural)`（D167 テlemetry vehicle テスト）が requestRebuildDirectCall.allowlist 未記載（DeferredFlowIntegrationTests の同型 entry は既存） | 同型 entry 追加（owner audio-runtime / BRIDGE-ISR-AI-GOV-001 / expiry 2027-12-31・既存と同一書式、rationale に commit 335240470 由来明記） |

targeted: **isr-verify-v73-admission-funnel PASS**（14 checks / violationCount=0）。

## 2c. isr-verify-v73-retire-pressure-contract.ps1 — v7.3 retire pressure の検査形同期

次段 violations=4。blame/provenance 確定: すべて本セッション外の実装進化に対する検査側未追従。

| checkId | 旧検査形 | 现行契約（実測） | 同期 |
|---|---|---|---|
| CI-RETIREPRESS-004 ×2 | Threading 側 `setRetireBacklogCount(retireDepth)` / `setDeferredRetireResidencyCount(fallbackDepth)` 絶対値 setter publish を要求 | **dash2 §1.4 B0-4（3b43a35d・09-10）で external setter 廃止**、AudioEngine.Retire.cpp:139-140 の実測 publishAtomic(retireQueueDepth_ = pendingRetireCount() / fallbackQueueDepth_) に収束（2820dfe7 系・Layer1 実測判定） | publishAtomic 実測形を要求 + **setter 復活を negative invariant で検出**（退行防止は強化方向） |
| CI-RETIREPRESS-006 enum | RetireEnqueueResult を 4 分岐 exact-match | 55aa7e96（2026-08-18）で `TerminalReclaim` 増設（優先度レーン結果） | Success/QueuePressure/QueueFull/Shutdown + optional TerminalReclaim を許可（必須 4 分岐要求は不変） |
| CI-RETIREPRESS-006 helper | 2 引数 noexcept 形 | 同一 commit 系で `DeletionEntryType` 既定引数付き | 3 引数形を許可 |

targeted: **isr-verify-v73-retire-pressure PASS**（violationCount=0・実測）。

## 2d. isr-verify-v73-shutdown-reclaim.ps1 + policy shutdownReclaimChecks — 5 violations 同期

| checkId | 内容 | 根拠（pre-session 確定） | 同期 |
|---|---|---|---|
| CI-RECLAIM-003 ×2 | required pattern / application が `RuntimeIntentCoordinator::reclaim(` 旧 API を要求 | work88 Step 9 / dash2 §2.2 で bool reclaim API 削除・`requestReclaim / reclaimNormal / reclaimShutdownQuiescent`（ReclaimPermit consume）へ収束（coordinator cpp:668-743） | policy の patterns + applications lineRegex を 3 API 選択形へ更新（単一 authority 要求は維持） |
| CI-SHUTDOWN-005 | ReleaseResources.cpp:253 `waitForDrain(100, 1)` 未 allowlist | commit 5c84ec9d（2026-08-25・D169-2 reconfigure 有界 drain） | allowlist entry 追加（bounded・non-RT・budget 明示 — 既存 2000ms entry と同型） |
| CI-SHUTDOWN-006 ×2 | Threading.cpp:186 / ISRShutdown.cpp:345 の **コメント文中** `isFullyDrained()` 言及が callsite 扱い | 55aa7e96（08-18）/ f2c1bf53（08-15）のコメント | scanner に注釈行 skip（`^\s*(//|/\*|\*)`）追加 — allowlist の対象は実 callsite という設計義へ同期。callsite 検出能力は不変 |

targeted: **isr-verify-v73-shutdown-reclaim PASS**（violationCount=0・実測）。

## 2e. isr-verify-v73-residency-telemetry.ps1 + policy residencyTelemetryChecks — 5 violations 同期

| checkId | 旧検査形 | 现行契約（pre-session 実装） | 同期 |
|---|---|---|---|
| CI-RESIDENCY-001 | `std::deque<PendingRetry> queue_`（RetryScheduler.h:55・fd42796b 2026-08-24）が allowlist 不在 | 有界 retry キュー（retirement 系ではないが name heuristic 'queue' に該当） | residencyContainerAllowlist へ declaration-site entry 追加 |
| CI-TELEMETRY-002 ×3 | required applications が `runtimePublicationBridge_.set{FallbackBacklogCount,RetireBacklogCount,DeferredRetireResidencyCount}`（Retire.cpp）を要求 | dash2 §1.4 B0-4 で setter API 自体削除、実測 publishAtomic(retireQueueDepth_/fallbackQueueDepth_) に収束（Retire.cpp:139-140・B0-7 コメント AudioEngine.h:4371） | policy requiredTelemetryApplications から setter 3 条を除去（実測 publish の 2 条は既に登録済・検出能力維持） |
| CI-TELEMETRY-009 | fallback enqueue 即時同期に setRetireBacklogCount を要求 | 现行同期 = `drainDeferredRetireQueues(false);` + `publishAtomic(retireQueueDepth_, retireDepth, release)`（AudioEngine.h:4368-4370） | スクリプト requiredImmediateSyncPatterns を现行 2 パターンへ・policy へ AudioEngine.h drain entry 追加 |

targeted: **isr-verify-v73-residency-telemetry PASS**（violationCount=0・実測）。

## 2f. isr-verify-p3-governance.ps1 R14 — retire-intent API 命名整合（D132 M2 同期）

- 旧要求: Commit.cpp に `lifetime().emitRetireIntentRT(` 必須。现行（D132 M2・v7 R9 と同一進化）:
  commit 経路は NonRT 供給で `lifetime().emitRetireIntentNonRT(intent)`（Commit.cpp:485）。
- 同期: Assert-Match → NonRT 名、NotMatch に **RT 名復活禁止 + 命名なし旧 API 禁止**を追加
  （検出方向は同一でむしろ強化）。
- targeted: **P3 governance PASS**（R13/R14/R19-R23）。

## 2g. publication-atomicity / runtime-view-lifetime / aba-hazard — 同型 drift 3 件同期

| script | 旧検査形 | 现行契約（provenance 済） | 同期 |
|---|---|---|---|
| isr-verify-publication-atomicity.ps1 | Commit.cpp 内 `runtimePublicationBridge_.commit(` anchor + ordering | dash2 §1.7 CW-3a で callback 内 commit #2 除去（単一 authority commit = RuntimeWorldAuthority::publish→coordinator_.commit。C4/single-path が検証済） | anchor を post-publish callback `onRuntimePublishedNonRt` 頭部へ。ordering（lastCommitted* が callback 入口後）維持 + **callback 内 bridge commit 再出現を negative で禁止**（CW-3a 退行検出・強化方向）。requiredPatterns からも bridge commit を除去 |
| isr-verify-runtime-view-lifetime.ps1 | Commit.cpp に `readControlRuntimeView(` 要求 | 同 API は 9e0db5f7（2026-06-02）系で消滅、control read は D170 handle 一本路 `worldAuthority_.consumeWorldHandle()`（Commit.cpp:587） | handle-scoped read 要求へ置換（意図同一） |
| isr-verify-aba-hazard.ps1 | prevWorld=static_cast(currentWorld_) + seq/epoch raw static_cast> 比較 | CW-3b 明示 prevWorld baseline + §1.6.1 isAfter modular（memory-ordering と同一進化） | coordinatorPatterns を现行 3 形（prevSeqId baseline / isAfter×2）へ（mappedGeneration/bake は不変） |

targeted: **3/3 PASS**。

## 3. 検証

| ゲート | 結果 |
|---|---|
| targeted 6 check（trigger-policy / trigger-symbol-usage / metric-governance / rollback-matrix / flag-dependency-graph / clang-tidy-readiness） | **全 PASS**（usage: totalMatches=8 blockedMatches=0） |
| standard PUSH GATE（-Tier standard・完走） | **PASS**（2026-09-14 / 160 [PASS]・FAIL/ERROR 0・`tiered verification completed. tier=standard`） |
| push | **実行**（本 commit を含む H-02/SR-03/work93/work94 series を origin/main へ通常 push） |

## 4. 意味

本 work は gate の緩和ではない。expiry 再承認は「四半期レビューを実施した」ことを示す管理レコードであり、
検出設計（zero-regression: 範囲外・新 path の出現は即 blocked）は不変。

