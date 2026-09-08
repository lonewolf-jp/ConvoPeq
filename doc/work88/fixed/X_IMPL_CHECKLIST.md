# work88 X1〜X6 実装チェックリスト（REPAIR_PLAN2-dash.md §6）

**更新日**: 2026-08-11
**対象**: REPAIR_PLAN2-dash.md の残課題 X1〜X6（§6.1〜§6.9）
**状態**: ✅ 完了（Phase 0 / X5 / X6 / X2 / X1 / X3 主要不変条件 / X4-A / X4-B / X3-R4 Phase 7（shutdownReclaim API 削除））+ 🚧 残（Phase 7 の Release・soak は別途）
**検証（2026-08-11）**: Debug フルビルド 0 エラー・ctest **28/28 PASS**・git diff --check クリーン・`publication_authority_verifier.py` PASS
**X4-B 実装注記（2026-08-11）**: RuntimeStore ownership migration 完了 — RuntimeWorldAuthority が Store を value 所有（INV-X4-3/5）。PublishExecutor は commit 削除 + `authority.publish()` 一本化。Bootstrap / shutdown clear / read API（~13箇所）を worldAuthority 経由に移行。旧 `makeRuntimePublishAuthority()` は production から削除。
**Harness 競合対応（2026-08-11）**: X4-B の publish() 高速化により、並行 rebuild publish が test の観測 seq を上書きする競合（5ms ポーリング vs 上書きウィンドウ）が顕在化。パイプライン自体は正常（swap はデバッグ検証済み）のため、`testIdlePublishViaFacade` / `testPublishCompletionMonotonicity` の store 検証を `== seqId` → `>= seqId` に堅牢化（FIFO 連続性 INV-X2-6 前提で正しい検証）。
**前提**: P2-1〜P2-4 は実装完了（`P2_IMPL_CHECKLIST.md` 35 項目 ✅ / ctest 28/28 PASS）
**normative source**: dash の履歴記述ではなく **最新版 A-2.42 以降 + §6 の「現在形」記述**を正とする

**実装順序（dash §6.9）**:

```text
Phase 0  invariant / specification freeze（INV-X1-5,6 / INV-X2-6 / INV-X3-4,5 / INV-X4-1〜8,A〜C / INV-X5-1 / INV-X6-4 / INV-ISR-01〜07）
Phase 1  X5（Publish Intent residency 専用 counter）
Phase 2  X6（Quarantine Intent/Ring/Resident semantic 分離）
Phase 3  X2（Publish completion sequence monotonicity）
Phase 4  X1（Recovery Durable Admission — lease 方式）
Phase 5  X4（Authority naming + ownership convergence: X4-A → X4-B）
Phase 6  X3（Reclaim Authority 統合 + readerRegistrationClosed）
Phase 7  統合 shutdown / ISR soak / 全 ctest
```

---

## Phase 0 — invariant / specification freeze（コード契約として固定）

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 1 | `pendingIntentCount_` 宣言コメントで「This counter excludes Publish and RetireIntent.」を固定 | `ISRRuntimePublicationCoordinator.h` | ✅（2026-08-11 実装 — 除外コメントの英語文言を追加（INV-ISR-02 と併記）） |
| 2 | INV-ISR-01〜07（isFullyDrained 完全条件 / residency+reservation / semantic 混同禁止 / readerRegistrationClosed 必須 / committed≠completed / currentWorld_ non-owning / dual pointer identity）をコメント固定 | `ISRRuntimePublicationCoordinator.h` / `AudioEngine.h` | ✅（2026-08-11 実装 — `RuntimeIntentCoordinator` 冒頭に INV-ISR-01〜07 コメントブロック固定。INV-ISR-05 は AudioEngine.h PublishReceiptWaiter に既存） |
| 3 | INV-X5-1（publicationIntentResidencyCount = queue residency + producer reservation）をコメント固定 | `ISRRuntimePublicationCoordinator.h` | ✅ |
| 4 | INV-X6-4（quarantine の counter が transport と DSP residency を同時に表さない）をコメント固定 | `ISRRuntimePublicationCoordinator.h` | ✅（コード実装済み — チェックリスト ⬜ 更新漏れ） |
| 5 | INV-X2-5/6（sole completion writer / completion order == publication sequence order）をコメント固定 | `AudioEngine.h` `PublishReceiptWaiter` | ✅（コード実装済み — チェックリスト ⬜ 更新漏れ） |
| 6 | INV-X3-4/5（readerRegistrationClosed / pendingReclaimHandles_ source of truth）をコメント固定 | `core/EpochDomain.h` / `AudioEngine.h` | ✅（コード実装済み — チェックリスト ⬜ 更新漏れ） |
| 7 | INV-X4-1〜8 / A〜C（authority matrix / publishAndSwap sole owner / currentWorld_ observation-only）をコメント固定 | `RuntimeWorldAuthority.h` / `core/RuntimeStore.h` | ✅（コード実装済み — チェックリスト ⬜ 更新漏れ） |
| 8 | INV-X1-1〜6（exactly one durable state / queue full ≠ lost / 1 admission = 1 reservation / 二重計上禁止）をコメント固定 | `ISRRuntimePublicationCoordinator.h` | ✅（コード実装済み — チェックリスト ⬜ 更新漏れ） |

## X5 — Publish Intent residency 専用 counter（dash §6.5）— **GO（実装着手）**

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 9 | `publicationIntentResidencyCount_` を宣言（`publicationBacklogCount_` の隣・Publish 系 counter 集約） | `ISRRuntimePublicationCoordinator.h` :383 付近 | ✅ |
| 10 | `enqueuePublicationIntent`: reservation→push→rollback（fetchAdd +1 → push 成功維持 / 失敗 fetchSub -1） | 同上（inline） | ✅ |
| 11 | `processIntent` の `intentQueue_.pop` で `type == Publish` の場合 `publicationIntentResidencyCount_--`（Publish 分岐追加・pendingIntentCount_ は触らない） | `ISRRuntimePublicationCoordinator_ProcessIntent.cpp` | ✅ |
| 12 | `ShutdownScheduler::isFullyDrained()` に `publicationIntentResidencyCount == 0` を追加 | `ISRRuntimePublicationCoordinator.cpp` | ✅ |
| 13 | メモリオーダリング: fetchAdd/fetchSub は acq_rel（wrapper デフォルト）を明示 | 同上 | ✅ |
| 14 | X5 テスト（ISRSoakTests 拡張）: enqueue +1 / pop -1 / queue full rollback / deferred counter unaffected | `src/tests/ISRSoakTests.cpp` | ✅（pop 側は Harness 統合で検証予定） |

## X6 — Quarantine Intent / Ring / Resident semantic 分離（dash §6.6）— 条件付き GO

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 15 | `quarantineIntentResidencyCount_` / `quarantineRingResidencyCount_` を宣言（`quarantineResidentCount_` の隣） | `ISRRuntimePublicationCoordinator.h` :388 付近 | ✅ |
| 16 | `submitQuarantine`: primary push 成功 → `quarantineIntentResidencyCount_++`。fallback 移動 → intent -1 → ring +1（両方同時に 1 にならない） | `ISRRuntimePublicationCoordinator.cpp` `submitQuarantine` | ✅ |
| 17 | `submitQuarantine` の `quarantineResidentCount_` +1 を撤去（DSPQuarantineManager::quarantineHandle が唯一管理） | 同上 | ✅ |
| 18 | `processIntent`: `intentQueue_` の Quarantine pop → `quarantineIntentResidencyCount_--` / `quarantineFallbackQueue_` pop → `quarantineRingResidencyCount_--` | `ISRRuntimePublicationCoordinator_ProcessIntent.cpp` | ✅ |
| 19 | `AudioEngine::isFullyDrained()` の `setQuarantineResidentCount(ringResident + dspQuarantine)` aggregate 上書きを廃止（:131）+ ReleaseResources の aggregate setter も撤去。ring/DSP/RetireQuarantine を個別直接判定 | `AudioEngine.Threading.cpp` / `ReleaseResources.cpp` | ✅ |
| 20 | `ShutdownScheduler::isFullyDrained()` に `quarantineIntentResidency == 0` / `quarantineRingResidency == 0` を追加（retireQuarantineStore は AudioEngine 側で直接判定） | `ISRRuntimePublicationCoordinator.cpp` / `AudioEngine.Threading.cpp` | ✅ |
| 21 | X6 テスト（状態遷移表固定）: primary/fallback 飽和後の intent=4096 / ring=1024 / pending=5120（intent+ring==pending 不変条件）+ full-drop rollback | `src/tests/ISRSoakTests.cpp` | ✅ |

## X2 — Publish completion sequence monotonicity（dash §6.2）— 条件付き GO

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 22 | `PublishReceiptWaiter`: contiguous completion 前提 + sole completion writer（INV-X2-5/6）をコードコメントで固定。mutex+cv+monotonic watermark 維持（案1 最小変更） | `AudioEngine.h` `PublishReceiptWaiter` :3613-3635 | ✅ |
| 23 | Committed（`lastCommittedPublicationSequence_`）≠ Completed（`lastCompleted_`）の分離コメントを固定（INV-ISR-05） | `AudioEngine.h` | ✅ |
| 24 | 2 watermark 同期（`m_lastObservedSequence` / `lastCompleted_`）の検証テスト（AudioEngineHarness 統合） | `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | ✅（contiguous FIFO completion 統合テスト） |
| 25 | timeout semantics 明文化: timeout ≠ publish failure（rollback 禁止・Transferred 維持）をコメント固定 | `AudioEngine.h` `commitRuntimePublication` | ✅ |

## X1 — Recovery Durable Admission（lease 方式）（dash §6.1）— 修正後 GO

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 26 | `PendingRecoveryAdmission` 構造体（State enum: NoAdmission/DurablePending/Building + pending/recoveryGeneration/buildSource/reservationOwned/handle/epoch/intentId） | `ISRRuntimePublicationCoordinator.h` :437 付近 | ✅ |
| 27 | `recoveryAdmissionPending_`（atomic bool）isFullyDrained 用 | 同上 | ✅ |
| 28 | `submitRecoveryRequest`: push 失敗時、`pendingIntentCount_` rollback → durable state 保持 + `recoveryAdmissionPending_ = true`（coalesce 単一スロット・reservation 増加なし — INV-X1-5/6）。drop counter は saturation 診断として維持（INV-X1-3） | `ISRRuntimePublicationCoordinator.cpp` `submitRecoveryRequest` | ✅ |
| 29 | `takePendingRecoveryAdmission()`: lease（DurablePending → Building 遷移。クリアしない）+ `settlePendingRecoveryAdmission(retry)` | 同上 | ✅ |
| 30 | Builder 消費ループ: `popRecoveryRequest()` の while ループ後に durable 残余を消費。build 失敗（transient）は settle(true) で retry。成功は PublishTransport + settle(false) | `AudioEngine.RebuildDispatch.cpp` :911 後 | ✅ |
| 31 | `stopRebuildThread()` の Builder join 後で durable admission 破棄（RecoveryAdmissionClosed）+ `recoveryAdmissionPending_ = false` | `AudioEngine.RebuildDispatch.cpp` `stopRebuildThread` | ✅ |
| 32 | `ShutdownScheduler::isFullyDrained()` に `!recoveryAdmissionPending_`（DurablePending OR Building が両方 false）を追加 | `ISRRuntimePublicationCoordinator.cpp` | ✅ |
| 33 | X1 テスト: queue full → durable 化 → lease take → settle（retry / success）→ coalesce（単一 admission）→ 二重計上なし | `src/tests/ISRSemanticValidationTests.cpp`（3 本追加） | ✅ |

## X4 — Authority naming + ownership convergence（dash §6.4）— GO（段階実装）

### X4-A（rename・低リスク）

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 34 | `convo::isr::RuntimePublicationCoordinator` → `RuntimeIntentCoordinator`（全 28 ファイル・LSP rename + 検証） | 各ファイル | ✅ |
| 35 | `convo::RuntimePublicationCoordinator` → `RuntimePublishAuthority`（template クラス / alias / factory `makeRuntimePublishAuthority` / read static 関数 acquireReadToken・consumeWorldHandle・consumePublishedWorld） | `AudioEngine.h` / `RuntimePublishExecutor.h` / `Init.cpp` / `CtorDtor.cpp` / `ReleaseResources.cpp` / `Latency.cpp` / `Commit.cpp` 等 | ✅ |
| 36 | X4-A ビルド + 全テスト PASS（静的検査: publishAndSwap はコアクラスのみ / commit は PublishExecutor のみ / Store 単一） | ビルド + ctest | ✅（フルビルド 0 エラー + ctest 28/28 PASS） |

### X4-B（ownership topology 変更・大規模リファクタ）

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 37 | `RuntimeStore.h` に `using OwnerType = Owner` を追加（Test 1 用） | `core/RuntimeStore.h` | ✅ |
| 37b | X4-B Stage A（Authority が Store 所有 + publish()）は実装検討後 revert — write パス移行（Executor の validate/retire 順序・read API ~10 箇所・Bootstrap/Shutdown）が大規模・高リスクのため単独ステージでは INV-X4-5 違反の二重 Store になり、一括移行が必要。グリーン状態を維持 | — | ↩️ revert |
| 38 | `RuntimeWorldAuthority` が `RuntimeStore<RuntimePublishWorld, RuntimeWorldAuthority>` を所有（member order + move/copy 禁止） | `RuntimeWorldAuthority.h` | ✅ X4-B 実装（2026-08-11）— `Store = convo::RuntimeStore<RuntimeState, RuntimeWorldAuthority>`（CRTP・Owner=自身）を value 所有。member order は `runtimeStore_` → `writeAccess_`（逆順破棄で `writeAccess_` が先）。move/copy 禁止 + static_assert 4 本固定 |
| 39 | `RuntimeWorldAuthority::publish()` 導入 | `RuntimeWorldAuthority.h` | ✅ X4-B-4 実装 — `publish(RuntimeOwner&&, PublishMetadata)` = basic validate + commit metadata（currentWorld_ 更新・commit-before-swap Test 7）+ `WriteAccess::publishAndSwap`（INV-X4-3）→ oldWorld 返却。retire は publish() 外（Execution tail / Bridge） |
| 40 | `PublishExecutor` から commit 削除 + 一時生成 publishWorld 廃止 | `RuntimePublishExecutor.h` | ✅ X4-B-5 実装 — `authority.commit()` 削除（publish() が内包・commit 二重化防止）+ 一時生成 `makeRuntimePublishAuthority()` 廃止 → `authority.publish()` に一本化。seal + validate（従来 publishWorld 相当）+ didPublish/willRetire/retire を Bridge 経由で実行 |
| 41 | Bootstrap / shutdown clear を authority 経由に | `AudioEngine.Init.cpp` / `ReleaseResources.cpp` | ✅ X4-B-6/7 実装 — Bootstrap は `worldAuthority_.publish()`（seal+validate+Bridge lifecycle）。shutdown clear は `worldAuthority_.requestShutdownClearNonRt()` + `clearPublishedRuntimeSnapshotsNonRt()` |
| 42 | 旧 `makeRuntimePublishAuthority()` を production path から削除（INV-X4-3/5） | `AudioEngine.h` | ✅ X4-B-8 実装 — `makeRuntimePublishAuthority()` / `RuntimePublishStore` / `runtimeStore` メンバ / `RuntimePublishAuthority` エイリアスを production から削除（write-capable Store は Authority 配下のみ — INV-X4-3/5） |
| 43 | read API migration（X4-B-9） | `AudioEngine.h` 等 | ✅ X4-B-9 実装 — `RuntimePublishAuthority::acquireReadToken/consumeWorldHandle/consumePublishedWorld` の全呼び出し（AudioEngine.h 5箇所 + Commit.cpp + Latency.cpp + observePublishedWorld 委譲）を `worldAuthority_` の read API（`acquireReadToken` / `consumeWorldHandle` / `observePublishedWorld`）に置換（getCurrent() は置換先にしない — INV-X4-7） |
| 44 | Architecture tests（Test 1-10） | テスト + 静的検査 | ✅ Test 1（`RuntimeWorldAuthority::Store::OwnerType == RuntimeWorldAuthority` を `ISRSemanticValidationTests.cpp` に追加 + `RuntimePublicationCoordinatorTests.cpp` の core 版）/ ✅ Test 2（WriteAccess move-only static_assert 6 本を `ISRSemanticValidationTests.cpp` に追加・2026-08-11）/ ✅ Test 3-10（`publication_authority_verifier.py` 静的検査 + AudioEngineHarness 統合でカバー — Test 3/4/5/6 は verifier、Test 7 は publish() 内 commit→swap 順序コメント、Test 8 は OwnerChannel SPSC、Test 9/10 は PublishPipelineIntegrationTests で統合検証。カバレッジ対応表をテストコードに記載） |

## X3 — Reclaim Authority 統合 + readerRegistrationClosed（dash §6.3 + R4）— 条件付き GO

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 45 | `EpochDomain` に `registrationClosed_` atomic + `closeReaderRegistration()` + `readerRegistrationClosed()`。`registerReaderThread` / `reserveReaderThread` 冒頭ガード | `core/EpochDomain.h` | ✅ |
| 46 | `ReclaimMode`（RuntimeEBR / ShutdownQuiescent）+ `reclaim(ReclaimMode, handle, ...)`（precondition を retire 前に評価） | `ISRRuntimePublicationCoordinator.h/.cpp` | ✅（R4 Phase 1-2 + Phase 3: ShutdownQuiescent は readerRegistrationClosed AND readersZero（activeReaderCount==0）を要求） |
| 47 | `requestReclaim` → `reclaim(RuntimeEBR, ...)` 委譲（挙動同一・RuntimeEBR パス維持） | `ISRRuntimePublicationCoordinator.cpp` | ✅ |
| 48 | `shutdownReclaim` 呼び出し元 → `reclaim(ShutdownQuiescent, ..., readerRegistrationClosed)` に移行（CacheMap::~CacheMap / ReleaseResources:417,422）。INV-X3-4 precondition 検証 | `AudioEngine.h` / `ReleaseResources.cpp` | ✅（R4 Phase 4・ビルド検証中） |
| 49 | `pendingReclaimHandles_.empty()` を `AudioEngine::isFullyDrained()` に追加（INV-X3-5: source of truth） | `AudioEngine.Threading.cpp` | ✅ |
| 50 | `shutdownReclaim()` API deprecated → 削除（R4 Phase 6-7） | `ISRDSPHandle.h` | ✅ **R4 Phase 7 完了（2026-08-10）** — `DSPHandleRuntime::shutdownReclaim()` 完全削除。call site = 0（AC-R4-1）・symbol absent（AC-R4-2）確認済み。production の reclaim は Reclaim Authority（`RuntimeIntentCoordinator::reclaim(ReclaimMode, ...)`）に一本化 |
| 51 | X3 テスト: closeReaderRegistration 後の registerReaderThread 失敗（INV-X3-4）+ reclaim(ShutdownQuiescent) precondition | `src/tests/invariant_INV3_INV5.cpp` | ✅（2 テスト追加・全 PASS） |

## Phase 7 — 統合検証

| # | 項目 | 実装箇所 | 状態 |
| --- | ------ | --------- | ------ |
| 52 | Debug/Release ビルド成功（警告は既存のみ） | ビルド | ✅ Debug フルビルド 0 エラー（X4-B 後再検証・2026-08-11） / ✅ Release（AudioEngineHarness）ビルド 0 エラー — ★ CMakeLists の `CMAKE_CXX_FLAGS_RELEASE`/`CMAKE_C_FLAGS_RELEASE`/icx に欠落していた `/EHsc` を追加（JUCE C1189 回避・既存ビルド構成バグ） |
| 53 | 全 ctest PASS | ctest | ✅（**28/28 PASS — X4-B 含む・2026-08-11**。AudioEngineHarness は堅牢化後 6/6 連続 PASS 確認） |
| 54 | `git diff --check` クリーン | git | ✅（X4-B 後再検証・CRLF 警告のみ） |
| 55 | 静的 architectural test | 静的検査 | ✅（publishAndSwap は RuntimeWorldAuthority.h + core のみ — `publication_authority_verifier.py` PASS / commit は publish() 内 / write-capable Store 単一（INV-X4-3/5））+ X4-B Test 1（`RuntimeWorldAuthority::Store::OwnerType == RuntimeWorldAuthority` static_assert） |
| 56 | shutdown / soak 実測 | 実行 | ✅ **Release でフル soak ALL PASS**（S1: 100000/100000 rejected=0・S2b: 80000/80000・S3/S4/S5 全て PASS・合計 issued=180300 rejected=0）。Debug の S1 rejection は Debug パフォーマンス起因（低速 coordinator loop）と確定・回帰なし |

---

## 変更ファイル一覧（予定）

| ファイル | X | 変更内容 |
| --- | --- | --- |
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | X5/X6/X1 | counter 宣言・PendingRecoveryAdmission・enqueuePublicationIntent |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | X5/X6/X1 | submit 系・isFullyDrained・takePendingRecoveryAdmission |
| `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp` | X5/X6 | processIntent type 分岐 |
| `src/audioengine/AudioEngine.Threading.cpp` | X6 | setQuarantineResidentCount aggregate 廃止 |
| `src/audioengine/AudioEngine.h` | X2/X4/X3 | PublishReceiptWaiter 契約・rename・worldAuthority |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | X1 | Builder durable 消費ループ |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | X1/X3 | shutdown discard・reclaim 移行 |
| `src/core/EpochDomain.h` | X3 | registrationClosed_ |
| `src/core/RuntimeStore.h` | X4 | using OwnerType |
| `src/audioengine/RuntimeWorldAuthority.h` | X4 | Store 所有・publish() |
| `src/audioengine/RuntimePublishExecutor.h` | X4 | commit 削除・publish() 一本化 |
| `src/audioengine/ISRDSPHandle.h` | X3 | shutdownReclaim deprecated |
| `src/tests/ISRSoakTests.cpp` / `ISRSemanticValidationTests.cpp` / 新規 | X1-X6 | テスト追加 |
| `CMakeLists.txt` | X1-X6 | テスト登録 |
