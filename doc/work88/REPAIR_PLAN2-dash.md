# REPAIR_PLAN2 残課題・改修設計（REPAIR_PLAN2-dash）

**★ レビュー判定（2026-08-10・外部レビュー反映）:**

- **GO（実装済み）**: P2-1〜P2-4 + P2-4 shutdown 補正（Step A/B/C: Recovery shutdown admission closure + ShutdownDiscard）
- **GO（設計確定）**: X5
- **条件付き GO（設計確定・Acceptance Criteria テスト化後に段階実装）**: X1 / X2 / X3 / X4 / X6
- **一括実装は NO-GO**（Phase 0 の invariant freeze → 各 Acceptance Criteria をテスト化 → 段階実装。推奨順序: P2→X5→X6→X2→X1→X4→X3）

**★ normative source 注意（2026-08-10・外部レビュー §22）:**
dash 内には過去レビュー記録（例: A-2.17「X4 NO-GO / X1 条件付き GO」）と最新版（例: A-2.42「X1 lease 方式で修正」）が混在する。
**実装者は履歴記述ではなく、最新版 A-2.42 以降 + §6 の「現在形」記述を normative source として扱うこと。**

**作成日:** 2026-08-09（同日 三次レビュー反映・構成再編・A-1〜A-4 設計化・四次〜十六次レビュー反映・X1-X6 詳細設計精緻化版）
**対象:** `doc/work88/REPAIR_PLAN2.md`（2026-08-08 更新版, 1309行）
**検証方法:** 実コード照合（六次レビュー）・semble/cocoindex/graphify/serena/AiDex による検索・構造検証・`build.bat Release icx` 実ビルド
**目的:** 未修正バグ・修正漏れに対する**詳細改修設計（実装対象）**を提供する。
**構成:** 前半 = 改修内容の**設計**（§1 実装対象 P2 / §2 追加対応設計）。後半 = 優先度・レビュー記録の**Appendix**。

**★ 実装対象（P2）:** §1.1（pendingIntentCount_ residency accounting 再設計）+ §1.2（isFullyDrained queue emptiness）+ §1.3（MpscBoundedRing テスト固定）+ §1.4（INV-3/INV-5 テスト）。**§2 は追加対応設計**（P3 降格・撤回案の代替・既知事項・残余リスクの対応方法を詳細設計）。

**★ 四次レビュー反映（2026-08-09）:** P2-1 は **GO**（convo wrapper 統一・cur>0 ガード削除・reservation invariant 明記）。**NO-GO 3件**: ①quarantineResidentCount_ の pop 減算（別の意味のカウンタ）、②publicationBacklogCount_ = Publish residency（現コードと不一致）、③**Recovery coalesce**（正当な Recovery を silent loss するバグ — §2.2 全面修正）。**CONDITIONAL**: PublishReceiptWaiter（completion seq monotonicity）。

**★ 五次レビュー反映（2026-08-09・実装可能性検証）:** 改修設計は**実装可能**で API 実在性を全て確認。修正5点: ①**2つの RuntimePublicationCoordinator の区別**（本設計は convo::isr 側が対象・queue emptiness は ShutdownScheduler::isFullyDrained に追加）、②**PublishStageResult は Success/Rejected/Failed の3値**、③既存 testProducerHoleDoesNotJumpAhead との命名衝突回避、④INV-3 テストの ISRRetireRouter/DSPHandleRuntime スタブ要件、⑤test hook は push 内部の #ifdef ブロック。

**★ 六次レビュー反映（2026-08-09・最終整合）:** **P2-1/P2-2/P2-3/P2-4 は実装 GO**。実装前に2点を明文化: **A. `pendingIntentCount_` の基本 invariant は `counter >= actual residency`（`==` は producer quiescence 後に限定）**、**B. queue emptiness は `AdmissionClosed → producer停止/join → queue観測` の順序保証をコード上で固定**。INV-5 は「**Recovery Intent の silent loss 禁止**」と再定義（drop ≠ memory corruption）。

**★ 七次レビュー反映（2026-08-09・実装詳細の完全性）:** **重大発見 W2 — `intentQueue_` に Publish が混在するため、`intentQueue_.pop` で Publish intent を pop した際に `fetchSub(1)` すると非対称 -1 を生じる**（pendingIntentCount_ の過小評価 → isFullyDrained が見逃し）。**processIntent は `commonIntent.type != Publish` の場合のみ fetchSub する**（§1.1.6）。その他: waitForDrain 時点の Coordinator/Builder 停止を実コード照合、queue emptiness は pendingIntentCount_ と独立に Publish 残留も捕捉、CONVO_TESTING は production レイアウト不変、INV-5 の drop テスト方法（256回 submit → full → drop）を確定。

**★ 八次レビュー反映（2026-08-09・実装完了条件の確定）:** **実装着手可能（概ね90%以上の設計精度）**。P2-1〜P2-4 は GO。実装完了条件として、以下の3点は**絶対に崩さない**（コードとテスト双方で固定）:
1. **`pendingIntentCount_ == actual residency` を常時要求しない**（基本 invariant は `counter >= actual residency`。`==` は producer quiescence 後のみ）
2. **Publish pop では decrement しない**（W2 — 七次レビュー）
3. **queue emptiness は producer quiescence 後のみ authoritative**（AdmissionClosed + all producers joined をコードで assert / phase guard）

条件A（八次 §22）: reservation counter は「成功した push の数」ではなく **`queue residency + producer-side enqueue reservation`**。コードコメント・テスト名・設計資料で同一定義にする。詳細は A-2.7 八次レビュー総括を参照。

**★ 九次レビュー反映（2026-08-09・残課題の明確化）:** **「P2-1〜P2-4 は実装 GO。ただし dash を『完全修正版』とは扱わない」**。実装完了条件4条件（八次3条件 + 九次条件1: reservation 意味のコードコメント固定）を確定。**残課題6件を §5 に追記**（X1 Recovery full-drop そのもの〔最優先〕/ X2 Publish completion monotonicity / X3 shutdownReclaim 二系統 / X4 authority 二重化 / X5 Publish intent residency 専用 counter 未導入 / X6 quarantine intent/resident semantic 分離）。実装順序を九次推奨 Phase 1-8 から Phase 1-10 に展開。詳細は A-2.8 九次レビュー総括を参照。

**★ 十次レビュー反映（2026-08-09・具体的コード変更の実行可能性）:** 実装コードとの整合を確認し、修正3点: ①`observeOverflowCounter_`（:559）は reservation とは独立に既存位置（primary push 失敗直後）へ維持、②`submitQuarantine` の `quarantineResidentCount_` +1 は現行維持（reservation は pendingIntentCount_ のみ。P3 で別カウンタ化）、③`drainObserveDeferred` の fetchSub は pop 成功直後・skip 前に挿入し、ObserveIntentHandler では fetchSub しない（二重計上防止）。Recovery reservation は CoordinatorLoop 内・ShutdownScheduler は nested class で private queue アクセス可（U4/U5/U6 は正確）。詳細は A-2.9 十次レビュー総括を参照。

**★ 十一次レビュー反映（2026-08-09・残課題 X1-X6 の詳細設計）:** **P2-1〜P2-4 は実装 GO。ただし「ISR Runtime 全体を完全に健全化する改修案ではない」**。acceptance criteria 3条件（C1: pendingIntentCount_ の意味をコード上で固定 / C2: queue emptiness は phase-gated / C3: Recovery full-drop を成功扱いにしない）を追記。**§6 に残課題 X1-X6 の詳細設計を追加**（X1 Recovery Durable Admission / X2 completion monotonic watermark / X3 Reclaim Authority 単一 + mode 2種 / X4 IntentCoordinator と RuntimePublishAuthority の明示分離 / X5 publicationIntentResidencyCount_ 新設 / X6 quarantineIntentResidencyCount_ 新設）。実装順序: **X5/X6 → X2 → X4 → X1 → X3**。中心原則: **Queue residency / deferred state / committed state / resident object / reclaimable object を別 semantic state として扱う**。詳細は A-2.10 十一次レビュー総括を参照。

**★ 十二次レビュー反映（2026-08-09・X1-X6 の具体的コード挿入位置）:** §6 の X1-X6 に**実装コードの具体的挿入位置**を追記: X3（requestReclaim → reclaim(ReclaimMode) 拡張・shutdownReclaim 廃止・呼び出し元4箇所）、X5（enqueuePublicationIntent が inline のため reservation→push→rollback をここに追加）、X6（submitQuarantine に quarantineIntentResidencyCount_ 追加・quarantineResidentCount_ +1 撤去）、X1（submitRecoveryRequest の push 失敗時に durable Pending state）、X2（PublishReceiptWaiter::complete を monotonic watermark + CAS に変更）、X4（AudioEngine.h:3509 の using エイリアス + 7ファイルの型参照を変更）。**P2 と X1-X6 は競合しない（X1-X6 は P2 後の独立タスク）**。詳細は A-2.11 十二次レビュー総括を参照。

**★ X1-X6 詳細設計精緻化（2026-08-09・実コード調査）:** §6 の X1-X6 を**実装可能な詳細設計レベル**に精緻化: X1（RebuildDispatch.cpp:911 の Builder 消費ループ後の durable state 処理 / recoveryPending との整合 / takePendingRecoveryAdmission 新設）、X2（m_lastObservedSequence と PublishReceiptWaiter::lastCompleted_ の2箇所の watermark + waitFor の cv 整合）、X3（requestReclaim:589 の epoch 判定を踏まえた reclaim(ReclaimMode) の具体的シグネチャ）、X4（PublishExecutor:63 での core Coordinator 一時生成の特定）、X5（processIntent:36 の type 分岐による3系統 counter 減算）、X6（DSPQuarantineManager::residentCount が source of truth）。テスト計画（§6.8）も既存テストの拡張位置を確定。詳細は A-2.12 を参照。

**★ 十四次レビュー反映（2026-08-09・P2 GO・X1-X6 要修正）:** **P2-1〜P2-4 は GO。X1〜X6 は「このまま実装してよい完成設計」とするのは NO-GO**。修正6点: ①X2 は「completion を何と定義するか」から再設計（contiguous completion 前提で store で十分・wraparound は案A「architecturally impossible」・INV-X2-5 sole completion writer 追加）、②X1 は single slot をやめ `PendingRecoveryAdmission`（pending/recoveryGeneration/buildSource）に再設計・reservation を push 前に、③X3 は precondition を retire 前に評価・`reader re-entry impossible` を追加、④X4 は RuntimeStore の `friend Owner`（:81）構造を確認し `RuntimeWorldAuthority` を publication authority surface に、⑤X5 は GO 承認、⑥X6 は `quarantineIntentResidencyCount_`/`quarantineRingResidencyCount_`/`quarantineResidentCount_` の3分離。詳細は A-2.13 十四次レビュー総括を参照。

**★ 十六次レビュー反映（2026-08-09・最終設計レビュー・X4 NO-GO）:** **P2-1〜P2-4 は GO。X1/X2/X3/X6 は条件付き GO。X4 は現状 NO-GO**。追加 INV: INV-X1-5/INV-X1-6（Recovery reservation 二重計上防止・state machine）/ INV-X2-6（completion order == publication sequence order）/ INV-X3-4（reader registration permanently closed・shutdown state machine 統合）/ INV-X4-3（publishAndSwap は RuntimeWorldAuthority-owned WriteAccess のみ）/ INV-X6-4（no counter may represent both transport and DSP residency）。X6 は RetireQuarantineStore を含む4分離。詳細は A-2.17 十六次レビュー総括を参照。

**★ 十七次追加調査反映（2026-08-09・X4 詳細改修計画確定・X4-A/X4-B 分離）:** **X4 は「命名の二重化を解消するだけ」の旧案を採用しない。`X4 = naming convergence + physical ownership convergence` として設計固定**。実コード検証で確定: ①**二重の write surface**（ISR commit は currentWorld_ の metadata 更新のみ / core publishWorld が唯一の Store swap）、②publishWorld 直接呼び出し元3箇所（PublishExecutor / Bootstrap / shutdown clear）、③Store は AudioEngine のメンバで Owner は core Coordinator、④RuntimeWorldAuthority は既に ownerChannel_ と lifetime_ を値保持、⑤`testRuntimeWorldAuthorityAdapter` は pure delegate 検証で X4-B で仕様変更。**X4-A（rename・低リスク）→ X4-B（ownership topology 変更・大規模リファクタ）の二段階**。INV-X4-1〜5 を確定（INV-X4-3 が最終合格条件）。テスト再設計8本 + 実装順序 X4-0〜X4-10。X6 の quarantineRing 実体も確定（quarantineFallbackQueue_）。詳細は §6.4-X4 と A-2.20 を参照。

**★ 十八次別視点調査反映（2026-08-09・スレッド所有権・外部 setter 干渉・メモリオーダリング）:** X1〜X6 を別視点から検証し確定: ①**`AudioEngine::isFullyDrained()`（Threading.cpp:117,131）の外部 setter が X1/X5/X6 と干渉** — `setPendingIntentCount(hasDeferredCommit ? 1 : 0)` と `setQuarantineResidentCount(ring+resident)` の外部上書きは X1〜X6 実装時に廃止し Coordinator 内部の純粋 accounting に一本化（§6.7）、②**X2 complete() の thread 所有権**: CoordinatorLoop 単一スレッド・Producer は waitFor・shutdown 中は timeout で Transferred、③**X3 の2つの ShutdownPhase enum**（AudioEngine/ISRShutdown）: CloseReaderRegistration は StopAudio 完了時の副作用として set、④**X4 friend 関係**: PublishExecutor は worldAuthority() 経由に置換・WriteAccess の friend Owner 変更は PublishExecutor に非影響、⑤**メモリオーダリング**: fetchAdd/fetchSub は acq_rel（X5/X6 新設 counter も踏襲・2スレッド間 RMW で race なし）、⑥**X6 residentCount() は quarantineActiveFlags_ 走査**（X6 は走査維持・isFullyDrained は NonRT で直接読む）。詳細は A-2.22 / INTEGRATED 9-34 を参照。

**★ 十八次別視点3調査反映（2026-08-09・実装詳細・Producer 前提・reclaim 管理）:** ①**X6 の `retireQuarantineResidentCount_` は新 counter 不要（設計変更）** — RetireQuarantineStore は既に `size_` + `residentCount()` を持ち、既存を source of truth に（§6.6/§6.7/A-2.18 D3）、②**X1 lost-wakeup 整合**: takePendingRecoveryAdmission 消費中も recoveryPending 再 set 機構が機能（§6.1）、③**X3 shutdownReclaim 呼び出し元の順序**: CacheMap は delete 先 / ReleaseResources は retire 先（§6.3）、④**X3 reclaimInFlightCount_ 管理**: +1 は遅延 / 0 は完了の簡略設計・ShutdownQuiescent はカウンタを呼ばない（§6.3）、⑤**X2 sequence 採番は RebuildThread**（RuntimeBuilder.cpp:81,183。Producer serialization 成立）（§6.2）、⑥**X1 Producer 単一スレッド前提の完全確認**: submitRecoveryRequest の呼び出し元は 1 箇所・RecoveryIntentHandler は dead code（§6.1）。詳細は A-2.24 / INTEGRATED 9-36 を参照。

**★ 十九次レビュー反映（2026-08-09・X4-B の currentWorld_ 意味論修正・条件付き GO → 実装着手可能 GO）:** X4 の方向性は正しいが、**`currentWorld_` は metadata cache ではなく第二の publication/read surface** であり、**`getCurrent()` を `consumeWorldHandle(runtimeStore)` の置換先にすることは NO-GO**（別 read source）。反映: ①**INV-X4-6 / INV-X4-7 追加**（publish 完了後 dual-pointer 同一 identity / 独立 source として扱う API 禁止）、②**`publish()` を「一つの atomic publication」と定義しない**（semantic transaction の唯一の execution boundary）、③**X4/X2 境界明確化**: commit-before-swap ordering は X4、completion monotonicity は X2、④**RuntimeWorldAuthority の move/copy 禁止**（Store+WriteAccess 所有）、⑤**Bootstrap/shutdown は「例外」でなく「lifecycle-controlled publish」**、⑥**Test 7 → commit-before-swap ordering / Test 9（dual-pointer consistency）/ Test 10（INV-X4-7）追加**（8本→10本）、⑦**実装順序 X4-0〜X4-10 → X4-0〜X4-11**（X4-7 publish() 導入を独立・X4-9 全 direct publishWorld caller 排除）。詳細は A-2.25 / §6.4-X4 を参照。

**★ 二十次レビュー反映（2026-08-09・X4-B の Store ownership・publish() 責務・INV-X4-8・Test 定義修正・実装 GO）:** 「Authority が Store を所有」と「AudioEngine の中で安全に所有」は別問題。反映: ①**コンストラクタは外部 Store を受け取らず、Authority 自身が Store identity を形成**（`runtimeStore_()` + `writeAccess_(runtimeStore_.acquireWriteAccess())`）、②**publish() の責務を「commit + physical swap」までに限定**（現行 publishWorld :100-141 の didPublish/willRetire/retire は PublishExecutor → completion → LifetimeState へ委譲）、③**read API を getCurrent() と分離し、physical RuntimeStore read 専用 API を新設**（observePublishedWorld / acquireReadToken / consumeWorldHandle(ReadToken)）、④**Test 7 は commit-before-swap ordering**（X2 側は completion monotonicity と分離）、⑤**INV-X4-8 追加**（source-role separation: currentWorld_ = metadata observation alias / RuntimeStore::current = physical publication source。delete/retire/unique_ptr/shared_ptr 変換を禁止）、⑥**INV-X4-6 の identity 構成要素確定**（sequenceId + publicationEpoch + mappedGeneration）、⑦**Test 3 厳密化**（allowed = RuntimeWorldAuthority / forbidden = RuntimeIntentCoordinator 等）/ **Test 6 修正**（write-capable のみ禁止・read-only 参照は許容）、⑧**RuntimePublishAuthority は WriteAccess を所有してはいけない**（二階層化禁止・INV-X4-3 強化）、⑨**Bootstrap/shutdown clear は publish() と統合しない**（clearPublishedRuntimeSnapshotsNonRt は別 API・shutdown semantic は X3）、⑩**実装順序を X4-B-0〜X4-B-11 に細分化**（rollback point 多数確保）。**X4-B = 実装 GO**。詳細は A-2.27 / §6.4-X4 を参照。

**★ 二十一次レビュー反映（2026-08-09・X4-B の型・constructor・member declaration 固定 + CRTP コンパイル検証・設計 GO）:** 実装開始前に `RuntimeWorldAuthority` の現行定義と `RuntimeStore` の型依存を実コードで固定。反映: ①**member declaration order**: runtimeStore_ → writeAccess_ → ownerChannel_ → lifetime_ → registry_（逆順破棄で WriteAccess が生きている間に Store を破棄しない）、②**CRTP 的 template 依存の実コンパイル検証（g++ -std=c++20）**: `RuntimeStore<World, Self>` の CRTP は既存実績あり（RuntimePublicationCoordinator.h:34）・`friend Owner` は incomplete でも動作・`static_assert(is_class_v)` は incomplete でも well-formed・**`Store::Owner` はコンパイル不可 → `using OwnerType = Owner` を RuntimeStore.h に追加**（Test 1 用・COMPILE_OK 検証済み）・循環依存なし、③**publish() の失敗経路・null経路**（null owner → Failed / null→null swap → Failed / null 公開は shutdown clear が担当）、④**Bootstrap / shutdown clear の初期化・破棄順序**（ctor → Bootstrap publish → shutdown clear → 逆順破棄）。**X4-B = 設計 GO・実装開始可能**（`using OwnerType` 追加が唯一の前提）。詳細は A-2.29 / §6.4-X4-B を参照。

**★ 二十二次レビュー反映（2026-08-09・X4-B の commit ownership・Test 9 identity 限定・RuntimePublishAuthority 一切所有禁止・設計方針 GO）:** 必須修正3点を反映。①**commit 二重化の危険（必須修正1）**: 現行 PublishExecutor（RuntimePublishExecutor.h:42-57）は既に authority.commit() を実行。**案A（publish() が transaction boundary）を採用** — PublishExecutor から authority.commit() を完全削除し、publish() 内部（validate → commit → owner.release() → publishAndSwap() → return previous）に内包、②**Test 9 は identity equality に限定（必須修正2）**: pointer equality（currentWorld_.load() == runtimeStore.current.load()）を要求しない。PublicationIdentity（sequenceId + publicationEpoch + mappedGeneration）の一致のみ検証、③**RuntimePublishAuthority は一切所有しない（必須修正3・INV-X4-3 強化）**: Store / WriteAccess / publishAndSwap 直接呼 / 代替 authority / Store に対する write capability を一切所有しない。**production code から RuntimePublishAuthority::create() 自体を削除**、④**Test 6 の write-capable 条件厳密化**: `RuntimeStore<RuntimePublishWorld, Owner>` の write-capable instance は `Owner == RuntimeWorldAuthority` のみ、⑤**旧記述「getCurrent() が consumeWorldHandle() の置換先になり得る」は削除済み**（NO-GO 規範のみ維持）。**X4-B = 設計方針 GO・実装着手可能**。詳細は A-2.31 / §6.4-X4-B を参照。

**★ 二十三次レビュー反映（2026-08-09・RecoveryAdmissionClosed・X2 timeout semantics・X4 swap failure・Phase 0・必須 Acceptance Criteria・条件付き GO）:** 総合判定: **P2-1〜P2-4 = 実装GO / X5 = 実装GO / X1/X2/X3/X6 = invariant 固定後実装GO / X4 = 設計GO（段階実装必須）**。「そのまま一括実装してよい完成改修案」ではない。反映: ①**X1 の `RecoveryAdmissionClosed` を shutdown state machine に追加**（`AdmissionClosed + RecoveryAdmissionClosed + BuilderStopped`。build gap 中の isFullyDrained 早期 true 防止）、②**X2 の timeout semantics 明文化**: `timeout ≠ publish failure`（timeout を failure と誤解して rollback すると double ownership / double publish。lifecycle = Allocated → Transferred → Committed → Completed）、③**X4 の `swap failure is architecturally impossible / handled` を acceptance criterion に**（publishAndSwap は単一原子 exchange で失敗しない・null→null は validate で事前検出）、④**Phase 0（invariant/specification freeze）を最優先**（実装順序を Phase 0〜Phase 7 に明確化。X4 は X2/X1/X3 の意味確定後に触る）、⑤**必須 Acceptance Criteria 表を §6.8 に追加**（X1-5/X1-6/X2-6/X3-4×2/X4-3/X4 3項目/X5/X6 2項目）。最上位 invariant = 「Intent → Admission → Transport Residency → Execution → Committed → Completed → Resident → Retired → Reclaimable → Deleted」の状態分離。詳細は A-2.33 / §6.8 / §6.9 を参照。

**★ 二十四次レビュー反映（2026-08-09・isFullyDrained measurement predicate・pendingIntentCount_ 命名・X4 dual-pointer 暫定正常状態・条件付き承認）:** 総合判定: **P2-1〜P2-4 = GO / X5 = GO / X1/X2/X3/X6 = 条件付きGO / X4-A = GO / X4-B = GO（段階実装）**。一括実装 = NO-GO。Phase 0 の invariant freeze を実施してから P2→X5→X6→X2→X1→X4→X3 の順に実装（条件付き承認）。新規反映3点: ①**`isFullyDrained()` を単独の truth source にしない**: `ShutdownPhase + ProducerQuiescence + AdmissionClosed + RecoveryAdmissionClosed + BuilderStopped + isFullyDrained()` を組み合わせる。**isFullyDrained() は measurement predicate であり shutdown authority そのものではない**（§6.7）、②**`pendingIntentCount_` の命名・コメント固定**: Observe+Quarantine+Recovery の queue residency + producer reservation（Publish と RetireIntent 除外）。将来は `transportIntentResidency_` 等への改名を検討。**コードコメントで「This counter excludes Publish and RetireIntent.」を固定**（§1.1）、③**X4 dual-pointer を「暫定正常状態」として明示**: `X4-B: write authority singularization / Future: read-source singularization` と分離。publish transaction 完了後 INV-X4-6（同一 PublicationIdentity）を保証するため正常動作として許容（§6.4-X4-B）。詳細は A-2.35 / §6.7 / §1.1 / §6.4-X4-B を参照。

**★ 二十五次レビュー反映（2026-08-09・INV-ISR-01〜07・X1 ShutdownDiscard・X3 意味論先行固定・最終判定）:** 総合判定: **P2-1〜P2-4 = GO（そのまま実装してよい）/ X1/X2/X6 = 条件付きGO / X3 = GO / X4 = GO（段階実装）/ X5 = GO**。Recovery coalesce / force reclaim = NO-GO（撤回が正しい）。一括実装 = NO-GO。「P2 を先行実装し、X1〜X6 は設計ゲートとして固定してから実装」を推奨。新規反映: ①**INV-ISR-01〜07（§23・最上位 ISR 不変条件）**: INV-ISR-01（isFullyDrained 完全条件）/ 02（pendingIntentCount_ = residency+reservation）/ 03（semantic 混同禁止）/ 04（ShutdownQuiescent は readerRegistrationClosed 必須）/ 05（committed ≠ completed）/ 06（currentWorld_ は non-owning）/ 07（dual pointer identity consistency 検証可能）（§6 冒頭）、②**X1 shutdown discard を「ShutdownDiscard」として明示（§8.1）**: `Recovery lost` と `ShutdownDiscard` を同じ意味にしない。Telemetry 上で2つを分ける（§6.1）、③**X3 の意味論を X4 より先に固定（§22）**: Lifetime correctness（X3）を先に固定してから Publication authority topology（X4）を変更。X3 の意味論（INV-X3-4 / INV-ISR-04）は Phase 0 で X4 より先に固定（§6.9）。残余リスク: X4-B 完了後も dual publication/read surface は残る（X4-B = write authority Singularization）。**最終評価: 「P2-1〜P2-4 は実装GO。X1〜X6 は段階実装が必要。特に X3 の lifetime closure と X4 の dual publication surface を最終的な ISR Architecture Review で再確認すること」**。詳細は A-2.38 / §6 冒頭 / §6.1 / §6.9 を参照。

**★ 二十六次レビュー反映（2026-08-10・X1 lease 方式・X3 INV-X3-5・X4 INV-X4-A/B/C・修正後 GO）:** 総合判定: **P2-1〜P2-4 = GO / X5 = GO / X6 = 条件付きGO / X2 = 条件付きGO / X3 = 条件付きGO / X4-B = GO（strict acceptance criteria）/ X1 = 修正後GO**。必須修正2点・強く推奨1点を反映: ①**X1 の Pending/Building 矛盾（必須修正1）**: `takePendingRecoveryAdmission()` を lease（state transition）に変更 — PendingRecoveryAdmission に `State` enum（NoAdmission/DurablePending/Building）を追加。take は DurablePending → Building へ遷移（クリアしない）。build 失敗（transient）は Building → DurablePending へ戻す。obsolete は Discarded。build success は PublishTransport。**INV-X1-1（exactly one durable state）が lease 方式で常に成立**（§6.1）、②**X3 の reclaimInFlightCount_ 近似 counter（必須修正2）**: `reclaimInFlightCount_ == 0` だけで shutdown drain を判定しない。**INV-X3-5 を追加**: ShutdownQuiescent completion requires `pendingReclaimHandles_.empty() AND reclaimInFlight == 0`。`pendingReclaimHandles_`（:4616）が reclaim pending の source of truth。isFullyDrained に pendingReclaimHandles.empty() 追加（§6.3 + §6.7）、③**X4 の INV-X4-A/B/C（強く推奨）**: `currentWorld_ = observation-only` / `RuntimeStore::current = sole physical RuntimeWorld source` / `No RT API may derive RuntimeWorld ownership/lifetime from currentWorld_`。**Audio Thread は currentWorld_ を RuntimeWorld 取得元として使わない**（§6.4-X4 INV-X4-8 に追記）。**最終評価: 「P2 と X5 は実装開始可能。X4-B も実装GO まで到達。ただし X1 の Pending/Building 状態矛盾と X3 の pending reclaim accounting は実装着手前に必ず修正する」**。詳細は A-2.42 / §6.1 / §6.3 / §6.4-X4 / §6.7 を参照。

---

# 設計（改修内容）

## 1. 実装対象の改修設計

> **★ 実装状況（2026-08-10）: P2-1〜P2-4 はすべて実装完了。**
> ビルド成功（AudioEngineHarness までリンク）+ ctest 28/28 合格 + `doc/work88/P2_IMPL_CHECKLIST.md`（35 項目）で検証済み。
> **変更内容**: 第三者別視点監査（2026-08-10）に基づき、§1.2 関連として「Recovery shutdown admission closure + ShutdownDiscard（Step A/B/C）」を**追加実装**（下記 §1.2 参照）。

### 1.1 🔴 pendingIntentCount_ の residency accounting 再設計 — **最優先（P2-1）**

> **✅ 実装完了（2026-08-10）** — §1.1.1〜§1.1.6 の設計どおり実装（P2_IMPL_CHECKLIST.md #1-11）。
>
> - **reservation-before-push**: `submitObserve` / `submitRecoveryRequest` / `submitQuarantine` とも `fetchAdd(+1)` → push 成功で維持 → 全段失敗 `fetchSub` rollback + drop カウンタ
> - **pop 成功時 fetchSub**: `processIntent`（quarantineFallbackQueue_ 無条件 / intentQueue_ は Publish 以外 — 七次 W2）/ `drainObserveDeferred`（pop 直後・skip 前）/ `popRecoveryRequest`（cur>0 ガード削除）
> - **`setPendingIntentCount(0)` ハードリセット廃止** + `AudioEngine::isFullyDrained()` / `AudioEngine.Commit.cpp`（:462, :607）の外部絶対値上書き廃止（RetireIntent 混入排除）
> - **メモリオーダリング**: `fetchAdd(release)` → push release-seq publish → pop acquire → `fetchSub` の happens-before チェーンで **+1 は -1 より必ず先行 → underflow 不可**（C++ [atomics.order] / cppreference memory_order で裏付け）
> - **⚠️ 変更内容（実装との乖離）**: `publicationBacklogCount_` は本番で**誰もインクリメントしない（デッドカウンタ化）**。dash §1.1.5 の「Coordinator 内部で実測維持される」は実態と不一致。ただし P2-4 の `intentQueue_.sizeApprox()==0` が Publish 残留を独立捕捉するため機能回帰なし。X5（`publicationIntentResidencyCount_` 新設）で解消予定。

#### 1.1.1 現状と問題

**現状**
`processIntent`（ISRRuntimePublicationCoordinator_ProcessIntent.cpp:43）末尾で `setPendingIntentCount(0)` を呼ぶ。producer 側（submitObserve / submitQuarantine / submitRecoveryRequest）は `setPendingIntentCount(pendingIntentCount_.load() + 1)` の **load → store 非RMW** で +1 する。

**🔴🔴🔴 二十四次レビュー（2026-08-09）— 命名の意味をコードコメントで固定（§27-B）**:
`pendingIntentCount_` という名前は「全 Intent の pending 数」と誤解されやすい。**実際の意味**:
```
pendingIntentCount_ = Observe + Quarantine + Recovery の queue residency + producer reservation
                    （Publish と RetireIntent は除外）
```
- **将来の名前変更を検討**: `transportIntentResidency_` など semantic な名称に変更すると事故を減らせる（本改修では名前変更しないが、コメントで固定）
- **コードコメントで固定（必須）**: 宣言箇所に以下を記述する:
  ```cpp
  // This counter tracks transport residency of Observe/Quarantine/Recovery intents
  // plus producer-side enqueue reservations. It EXCLUDES Publish and RetireIntent.
  std::atomic<std::uint64_t> pendingIntentCount_;
  ```
- **§1.1.1 以降の全セクションでこの意味定義を前提とする**（七次レビューの前提と整合）

**🔴 七次レビュー（2026-08-09）— 前提の明確化**: `intentQueue_` には **Observe / Publish / Recovery / Quarantine の4種が混在**するが、**`pendingIntentCount_` は Observe / Quarantine / Recovery の3種のみをカウント**する（Publish は publicationBacklogCount_ / deferredPublicationCount_ 側。enqueuePublicationIntent は pendingIntentCount_ に触れない）。この前提は本設計の全セクションに一貫して適用される。

**問題**
1. processIntent 実行中に submitXxx が push した intent の pending count が 0 にリセットされ、残留 intent が反映されない
2. producer の load → store は MPSC 下で **lost-update**（P1: load=10 / P2: load=10 / P1: store 11 / P2: store 11 → +2 が +1 になる）
3. **単純な `fetch_add after push` では underflow する**:

```
P: intentQueue_.push(intent)      ← item は publication 済み
        ↓
C: intentQueue_.pop(intent)       ← consumer が pop できる
C: pendingIntentCount_.fetch_sub(1)
        ↓
count = 0 → UINT64_MAX へ underflow   ← P の fetch_add より先に実行される可能性
        ↓
P: pendingIntentCount_.fetch_add(1)
```

- **「push 成功 → fetch_add が pop より先」という保証はない**。queue の publication と counter の RMW は**同一の同期変数ではない**ため、`memory_order` だけでは順序を強制できない
- **正しい線形化点は「queue への enqueue より前」に置く必要がある**

#### 1.1.2 改修設計 — residency reservation → push → failure rollback

**🔴 四次レビュー（2026-08-09）— 設計は GO、ただし実装方法に修正あり**:
- **基本原理（reservation-before-publish）は正しい**。counter は「numeric residency accounting」（同期 payload ではない）のため relaxed で良いが、**プロジェクト規約により `convo::fetchAddAtomic()/fetchSubAtomic()` に統一する**（`std::atomic::fetch_add()` の直接使用は避ける — Practical Stable ISR Runtime 原則）
- **`quarantineResidentCount_` の pop 減算は NO-GO**（§1.1.5 参照 — 別の意味のカウンタ）

**🔴🔴 五次レビュー（2026-08-09）— memory_order の扱いを確定**:
- `convo::fetchAddAtomic/fetchSubAtomic`（AtomicAccess.h:91,100）は **default `memory_order_acq_rel`**。dash の `memory_order_relaxed` 明示は「counter は numeric accounting（同期 payload ではない）」ため**技術的に正しい**
- **ただし実装簡潔性のため、明示的 memory_order を省略して wrapper のデフォルト（acq_rel）に任せることも可**（acq_rel でも性能影響は軽微 — cache-line bouncing は全 producer NonRT のため実質無し）
- **推奨**: producer 側も `memory_order_acq_rel`（デフォルト）に統一し、**明示的 relaxed を避ける**（規約の「明示的 ordering は同期ポイントにのみ」という思想と整合）。実装時はどちらかに統一し、部分的な混在を避けること

```cpp
// 改修: counter の線形化点を「enqueue 試行前」に置く（residency reservation 方式）
//   プロジェクト規約: convo::fetchAddAtomic / fetchSubAtomic を使用（std::atomic 直接使用を避ける）
//   memory_order: デフォルト（acq_rel）に統一（§五次レビュー）
//   producer 側（submitObserve / submitQuarantine / submitRecoveryRequest）:
//     convo::fetchAddAtomic(pendingIntentCount_, 1);  // ① reservation
//     if (intentQueue_.push(intent)) { ... return; }                 // ② enqueue（success → residency 確定）
//     convo::fetchSubAtomic(pendingIntentCount_, 1);  // ③ push 失敗 → rollback
//     // ④ fallback / drop 処理（fallback 成功なら ① の reservation をそのまま維持）
//   consumer 側（processIntent の3キュー drain / popRecoveryRequest）:
//     convo::fetchSubAtomic(pendingIntentCount_, 1);  // pop 成功時（skip 含む）
//   ※ setPendingIntentCount(0) の hard reset は廃止（残留を失わない）
```

**なぜこれで underflow しないか**: ①の +1 が ②の enqueue（publication）より**必ず先行**するため、consumer が pop できる時点では必ず対応する +1 が存在する。②が失敗した場合のみ ③で -1（rollback）する。①〜②の間に consumer が来ても、queue に item が存在しないので pop できない → underflow 発生不可。

**🔴 reservation invariant（六次レビュー確定 — テスト仕様に明記すべき）**:
```
基本不変条件:  pendingIntentCount_ >= actual queue residency
              （= actual queue residency + outstanding enqueue reservations）
quiescent point: pendingIntentCount_ == total queue residency
              （producer stopped AND no enqueue in flight でのみ成立）
```
**🔴🔴 六次レビュー（2026-08-09）— 「push完了後 == actual」は並行実行中は強すぎる**:
Producer が `push()` から戻った時点で Consumer が既に pop 済みの可能性がある（`publish直後 → consumer pop → counter -1`）。したがって **「push operation 完了後 = actual」は一般の並行実行では成立しない**。正しいのは:
- **基本不変条件 = `counter >= actual residency`**（in-flight reservation を含む）
- **`==`（equality）は quiescent point（producer stopped AND no enqueue in flight）でのみ成立** と定義する

テストでは:
- 並行実行中: `pendingIntentCount_ >= actual queue residency`（>= の検証）
- quiescent point（全 producer 停止・全 enqueue 完了後）: `pendingIntentCount_ == total queue residency`（== の検証）

この定義は「counter = queue residency + in-flight reservations」という意味論と形式的に整合する（§1.1.5）。

#### 1.1.3 対象キューの型と producer 数（実装時の前提）

| キュー | 型 | コンカレンシー | 扱い |
|--------|-----|---------------|------|
| `intentQueue_` | `MpscBoundedRing<Intent, kIntentQueueCapacity=4096>`（ISRRuntimePublicationCoordinator.h:445-446） | **MPSC**（Observe/Quarantine/Publish の複数 producer） | reservation→push→rollback 必須 |
| `quarantineFallbackQueue_` | `MpscBoundedRing<Intent, kQuarantineFallbackCapacity=1024>`（:453-454） | **MPSC** | 同上（submitQuarantine の2段目） |
| `observeDeferredRing_` | `LockFreeRingBuffer<ObserveIntent, kObserveDeferredRingCapacity=1024>`（:429-430） | **SPSC** | reservation は intentQueue_ 失敗後の2段目で維持 |
| `recoveryIntentQueue_` | `LockFreeRingBuffer<RecoveryIntent, 256>`（:433-434） | **SPSC**（Producer=CoordinatorLoop 単一） | ①reservation→push → popRecoveryRequest で fetch_sub |

- reservation 方式は全キューに適用可能（MPSC は fetch_add の原子性で安全、SPSC も同一パターンで一貫）
- SPSC キュー（observeDeferredRing_ / recoveryIntentQueue_）は「1 producer」前提が契約。同一スレッド（Timer / CoordinatorLoop）が逐次実行するため成立
- drop 経路は各 submitXxx の既存 drop カウンタ（observeFallbackOverflowCounter_ / quarantineFallbackDropCount_ / recoveryIntentDropCount_）を流用

#### 1.1.4 fallback 経路の注意

`submitObserve`（intentQueue_ → observeDeferredRing_）と `submitQuarantine`（intentQueue_ → quarantineFallbackQueue_）は**2段階 enqueue**。①の reservation は「1 件の intent の residency」を表すため、fallback 成功時は ① の +1 を**そのまま維持**（rollback しない）。全段失敗時のみ rollback + drop カウンタ。

#### 1.1.5 pendingIntentCount_ の意味論（四次レビューで確定）

**🔴🔴 四次レビュー指摘 — 3カウンタは分離すべき**:

```
pendingIntentCount_        = Observe + Quarantine + Recovery の queue residency counter
                            （in-flight reservation 含む）
publicationBacklogCount_   ≠ Publish intent residency（現コードでは hasDeferredCommit 由来）
deferredPublicationCount_  = RuntimePublicationOrchestrator の deferred 状態（hasDeferredRequest）
```

**🔴🔴🔴 八次レビュー（2026-08-09）— 条件A: reservation counter は「成功した push の数」ではない**:
正確な定義は:
```
pendingIntentCount_ = queue residency + producer-side enqueue reservation
（成功した push の数ではなく、reservation を含む累計）
```
したがって **concurrent invariant は `counter >= actual queue residency`**。この定義を **コードコメント・テスト名・設計資料のすべてで同一に**する（実装完了条件）。「成功した push の数」と誤解する記述は避ける。

- **`publicationBacklogCount_` を「Publish residency」と解釈するのは誤り**（四次レビュー NO-GO）。現在の `AudioEngine::isFullyDrained()`（Threading.cpp:118）は `setPublicationBacklogCount(hasDeferredCommit ? 1u : 0u)` であり、**実際の intentQueue_ 内 Publish 件数ではない**。`enqueuePublicationIntent` は pendingIntentCount_ にも publicationBacklogCount_ にも触れない
- **Publish Intent Queue residency** を監視するなら **`publicationIntentResidencyCount_` を新設**する（本改修の範囲外・P3）。`deferredPublicationCount_`（Orchestrator の deferred 状態）とは明確に分離する
- **混入を廃止**: `AudioEngine::isFullyDrained()`（Threading.cpp:117）の `setPendingIntentCount(hasDeferredCommit ? 1u : 0u)`（Publish 系）と、`AudioEngine.Commit.cpp:462,604` の `setPendingIntentCount(lifetime().pendingIntentCount())`（RetireIntent 系）の**2つの混入上書きを廃止**する
- **🔴 五次レビュー確定 — isFullyDrained の返り値 `!hasDeferredCommit` は維持**: 上書き廃止後も `AudioEngine::isFullyDrained()` の返り値 `return !hasDeferredCommit && runtimePublicationBridge_.isFullyDrained();`（Threading.cpp:135）の **`!hasDeferredCommit` 条件は維持**する。deferred commit の判定は「pendingIntentCount への混入」ではなく「返り値の独立条件」として保持する（`publicationBacklogCount_` への上書きも同様に廃止し、deferred 判定は `!hasDeferredCommit` で担保）

**🔴🔴 quarantineResidentCount_ は変更しない（四次レビュー NO-GO）**:
- `submitQuarantine` が `quarantineResidentCount_` を +1 するのは**意味論が混在**するが、四次レビューでは「**既存の意味を再定義しない**」と判断。`quarantineResidentCount_` は **「実際に quarantine lane に存在する DSP 数」**（`DSPQuarantineManager::residentCount()` 由来、Threading.cpp:131 で `ringResident + dspQuarantineResident` に設定）であり、**Quarantine Intent の queue residency ではない**
- **正しい扱い**: `submitQuarantine` の `quarantineResidentCount_` +1（:713,723）は **P3 リファクタで撤去**し、`quarantineResidentCount_` は isFullyDrained の実測設定のみに限定する。Quarantine intent queue residency を個別監視したければ **`quarantineIntentResidencyCount_` を新設**する
- **本改修（P2）では `quarantineResidentCount_` に触れない**（現状維持）。§1.1.6 の不変条件から quarantineResidentCount_ の reservation/rollback を削除

#### 1.1.6 不変条件（テストで固定）

**🔴🔴 七次レビュー（2026-08-09）— Publish intent の pop は fetchSub 対象外**:
`intentQueue_` には **Observe / Publish / Recovery / Quarantine の4種が混在**（ISRRuntimePublicationCoordinator.h:201-206）。**Publish は pendingIntentCount_ に +1 されない**（publicationBacklogCount_ 側）ため、**Publish intent を pop した際に fetchSub(1) すると非対称な -1 を生じる**（過小評価 → isFullyDrained が残留を見逃す可能性）。**processIntent の `intentQueue_.pop` は `commonIntent.type == Publish` の場合 fetchSub をスキップする**必要がある:
```
while (intentQueue_.pop(commonIntent)) {
    if (commonIntent.type != IntentType::Publish)  // Publish は pendingIntentCount_ 対象外
        convo::fetchSubAtomic(pendingIntentCount_, 1);
    kDispatchTable[...]->handle(commonIntent, ctx);
}
```
（Publish の drain 判定は publicationBacklogCount_ / deferredPublicationCount_ 側で担保）

```
submitObserve   : ①fetchAdd(1) → intentQueue_/observeDeferredRing_ push 成功 → 維持
                  → 全段失敗 → ③fetchSub(1) + drop カウンタ
submitQuarantine: ①fetchAdd(1) → intentQueue_/quarantineFallbackQueue_ push 成功 → 維持
                  → 全段失敗 → ③fetchSub(1) + drop カウンタ
                  ※ quarantineResidentCount_ は本改修で触らない（P3 で別カウンタ化）
submitRecoveryRequest: ①fetchAdd(1) → recoveryIntentQueue_ push 成功 → 維持
                       → 失敗 → ③fetchSub(1) + recoveryIntentDropCount++（drop は telemetry 有り）
popRecoveryRequest（Builder Loop）: pop 成功 → fetchSub(1)
                  （cur>0 ガード削除 — reservation invariant が成立するため不要。
                   四次レビュー: むしろガードは counter 不整合を silently hide するため危険。
                   テストでは「pop 成功時 counter==0」を assert failure にすべき）
processIntent   : intentQueue_（**Publish 以外**）/ quarantineFallbackQueue_ / observeDeferredRing_ の
                  pop 成功（epoch-FIFO skip 含む）ごとに fetchSub(1)
```

**🔴🔴 十次レビュー（2026-08-09）— 観測カウンタと現行二重カウンタの扱いを確定**:

**① `observeOverflowCounter_`（ISRRuntimePublicationCoordinator.cpp:559）は reservation とは独立に既存位置へ維持**:
`submitObserve` の `observeOverflowCounter_` は「intentQueue_ full → observeDeferredRing_ 退避」の**観測カウンタ**であり、reservation の成功・失敗とは無関係に、**既存の位置（primary push 失敗直後）に維持**する:
```
convo::fetchAddAtomic(pendingIntentCount_, 1);      // ① reservation
if (intentQueue_.push(intent)) return;               // ② primary 成功 → 維持
observeOverflowCounter_.fetch_add(1, relaxed);      // ★ 観測（既存位置のまま）
if (observeDeferredRing_.push(fallback)) return;     // ④ fallback 成功 → ① の維持
convo::fetchSubAtomic(pendingIntentCount_, 1);      // ③ 全段失敗 → rollback
observeFallbackOverflowCounter_.fetch_add(1, relaxed);  // ★ drop 観測
```
`observeOverflowCounter_` / `observeFallbackOverflowCounter_` は**診断カウンタ**であり、residency に影響しない（§1.1.5 の意味論に含めない）。

**② `submitQuarantine` の `quarantineResidentCount_` +1（:713,723）は現行維持のため、reservation とは別に primary/fallback 成功時に +1**:
reservation（①）は `pendingIntentCount_` のみに対して行う。`quarantineResidentCount_` の +1 は**既存の意味論（実在 DSP 数の近似）を維持**するため、push 成功時（primary または fallback）に**既存どおり実行**する:
```
convo::fetchAddAtomic(pendingIntentCount_, 1);        // ① reservation（pendingIntentCount_ のみ）
if (intentQueue_.push(intent)) {
    setQuarantineResidentCount(load + 1);  // ★ 現行維持（pending の reservation とは別）
    return;
}
if (quarantineFallbackQueue_.push(intent)) {
    setQuarantineResidentCount(load + 1);  // ★ 現行維持
    return;
}
convo::fetchSubAtomic(pendingIntentCount_, 1);        // ③ rollback
convo::fetchAddAtomic(quarantineFallbackDropCount_, 1, release);  // drop
```
- **注意**: quarantineResidentCount_ の +1 は「1 intent = 1 reservation」とは独立（§1.1.5 の意味論に含めない。P3 で `quarantineIntentResidencyCount_` に分離）。P2 では**既存動作を維持**

- **epoch-FIFO skip の扱い**: `ObserveIntentHandler`（ProcessIntent.cpp:65）と `drainObserveDeferred`（:52）は、`epoch < currentEpoch` または `handle.isNull()` の intent を skip する。skip された intent は**キューから除去される**ため、**pendingIntentCount は pop 済みとして減算すべき**（「semantic processing されたか」ではなく「queue residency が終了したか」を accounting）
- **Recovery の扱い**: `RecoveryIntent` は `recoveryIntentQueue_`（Builder Work Queue）経由で `processIntent` の3キューに含まれない。`popRecoveryRequest`（Builder Loop 側）が独自に `fetchSub(1)` する

**🔴🔴 十次レビュー（2026-08-09）— fetchSub の挿入位置（pop 成功直後・skip 前）を確定**:
`drainObserveDeferred`（ProcessIntent.cpp:47-56）と `ObserveIntentHandler`（:59-69）の fetchSub は、**`pop` 成功直後・epoch-FIFO skip 判定の前に**挿入する（pop 済みなら skip でも decrement）:
```cpp
// drainObserveDeferred（ProcessIntent.cpp:50-55）:
while (observeDeferredRing_.pop(deferred)) {
    convo::fetchSubAtomic(pendingIntentCount_, 1);   // ★ pop 成功直後・skip 前（十次）
    const auto currentEpoch = currentPublicationEpoch();
    if (deferred.epoch < currentEpoch || deferred.handle.isNull())
        continue;                                    // skip でも decrement 済み
    lifetimeMgr.retireByHandle(deferred.handle);
}
```
```cpp
// ObserveIntentHandler::handle（ProcessIntent.cpp:60-68）:
//   intentQueue_ 経由の Observe は processIntent の while ループで fetchSub 済み
//   （commonIntent.type != Publish の場合のみ）。ハンドラ内では fetchSub しない。
```
- **intentQueue_ 経由**の Observe は、processIntent の `while (intentQueue_.pop)` で `commonIntent.type != Publish` の場合に fetchSub 済み（§1.1.6 の Publish 除外）。
- **observeDeferredRing_ 経由**の Observe は、`drainObserveDeferred` 内で fetchSub（上記）。
- **二重 fetchSub を防ぐ**: Observe が intentQueue_ 経由か observeDeferredRing_ 経由かは排他（どちらか一方のみ）なので、fetchSub は各経路で1回のみ。

#### 1.1.7 対象ファイル・影響範囲・検証

**対象ファイル**:
- `ISRRuntimePublicationCoordinator.cpp`（submitObserve / submitQuarantine / submitRecoveryRequest / popRecoveryRequest の reservation→push→rollback 化。**quarantineResidentCount_ は触らない**）
- `ISRRuntimePublicationCoordinator_ProcessIntent.cpp`（setPendingIntentCount(0) 廃止・pop 時 fetchSub。**Publish intent の pop は fetchSub 対象外** — §1.1.6）
- `AudioEngine.Threading.cpp`（isFullyDrained の pendingIntentCount / publicationBacklogCount 上書き廃止。**返り値の `!hasDeferredCommit` は維持**）
- `AudioEngine.Commit.cpp`（setPendingIntentCount の RetireIntent 混入廃止）

**影響範囲**: `processIntent` と `isFullyDrained` の整合性。`pendingIntentCount_` の意味が「Observe+Quarantine+Recovery residency（in-flight reservation 含む）」に確定（Publish / RetireIntent 除外）。**`quarantineResidentCount_` は現状維持**（P3 で別カウンタ化）。

**検証**:
- shutdown 時に残留 intent がある場合、`isFullyDrained` が false を返すこと（INV-6）
- `fetchSub` が `fetchAdd` に先行しないこと（residency reservation の線形化順序テスト）
- MPSC マルチ producer 下でカウンタが正確であること
- **reservation invariant**: `pendingIntentCount_ >= actual queue residency`（並行中）→ `== actual`（quiescent point）をテスト
- **Publish intent の pop で pendingIntentCount_ が減算されないこと**（非対称 -1 の防止。七次レビュー）
- Publish の deferred commit が pendingIntentCount に混入しないこと（deferredPublicationCount 側で判定）
- RetireIntent が pendingIntentCount に混入しないこと（retireBacklogCount で判定）
- **popRecoveryRequest 成功時 counter==0 は assert failure**（ガード撤去の検証）
- admission closed 後の queue emptiness が drain 判定に反映されること

---

### 1.2 🟠 isFullyDrained の queue emptiness 検証（P2-4）

> **✅ 実装完了（2026-08-10）** — §1.2.2 の設計どおり、`ShutdownScheduler::isFullyDrained()` に 4 キュー空判定（消費なし）を追加（P2_IMPL_CHECKLIST.md #12-15）。
>
> - `intentQueue_` / `quarantineFallbackQueue_`（`sizeApprox`）+ `observeDeferredRing_` / `recoveryIntentQueue_`（`size`）+ 既存 7 カウンタ == 0。phase-gated コメント付き。
> - **🔴 変更内容（監査補正 — 追加実装 2026-08-10）**: P2-4 の queue emptiness は「検出」として正しい（queue が実際に non-empty なら true-positive）。問題は「なぜ shutdown 完了時点で queue が non-empty か」= **shutdown admission closure の穴**。修正対象は前段の shutdown admission / Recovery lifecycle であり、**「Recovery shutdown admission closure + ShutdownDiscard（Step A/B/C）」を追加実装**した:
>   - **Step A**: `requestShutdown()`（ReleaseResources.cpp:75 / CtorDtor.cpp:102）で `state=ShuttingDown` を確定 → Recovery admission を閉じる（既存機構を利用）
>   - **Step B**: `submitRecoveryRequest()` 先頭に `state==ShuttingDown` gate（**reservation 前評価で counter 非接触**）。閉鎖後の submit は enqueue せず `recoveryShutdownDiscardCount_++`（ShutdownDiscard — silent loss 禁止 INV-5）
>   - **Step C**: `stopRebuildThread()` の Builder join 後に `discardRecoveryRequestsOnShutdown()` で残留 Recovery を**明示 discard**（`popRecoveryRequest` が fetchSub するため counter 整合・Producer join 済みで決定的）
>   - **telemetry**: `recoveryShutdownDiscardCount_` を新設（queue full による drop と区別 — dash §8.1 の ShutdownDiscard 分離方針）。getter 公開・Critical 昇格対象外
>   - **テスト**: INV-5-3（shutdown 後 submit → ShutdownDiscard）/ INV-5-4（残留 discard）を追加
>   - **残余**: admission check → reservation の atomicity は X1（`RecoveryAdmissionClosed` + `RecoveryDurableAdmission` lease 方式）で完全化（§6.1）。P2-4 の queue observation は維持

#### 1.2.1 現状

`AudioEngine::isFullyDrained()`（AudioEngine.Threading.cpp:114-136）は pendingIntentCount_ だけでなく、**以下の全カウンタを「現在の実測値」に上書きしてから** `runtimePublicationBridge_.isFullyDrained()` を呼ぶ:

```
setPendingIntentCount(hasDeferredCommit ? 1 : 0)      ← :117
setPublicationBacklogCount(hasDeferredCommit ? 1 : 0)  ← :118
setFallbackBacklogCount(fallbackDepth)                 ← :122（fallbackQueueDepth_）
setRetireBacklogCount(retireDepth)                     ← :123（retireQueueDepth_）
setDeferredRetireResidencyCount(fallbackDepth)         ← :124
setQuarantineResidentCount(ringResident + dspQuarantine) ← :131（OverflowRing + DSPQuarantineManager 実測）
```

`isFullyDrained` は「カウンタの整合性を検証」しているのではなく、各所で load→store 加算されたカウンタを「その瞬間の実測値」で上書きし、その上書き値を自分で読んでいる。カウンタの増減は isFullyDrained 呼び出し時に「消去」される。

#### 1.2.2 改修設計

**🔴 四次レビュー指摘 — queue emptiness は「観測」であって「消費」ではない**:
- 単純に `while (!queue.pop(...))` を `isFullyDrained()` から行っては**いけない**（消費になる）
- `sizeApprox() == 0` は通常動作中は完全な source of truth ではないが、**shutdown 時（producer 停止後）は強い意味を持つ**（MpscBoundedRing の sizeApprox は単一 producer/consumer で正確とテスト済み）

**🔴🔴 五次レビュー（2026-08-09）— queue emptiness の実装 API を確定（両キューとも size 読取りで消費しない）**:
| キュー | API | 実装 | 消費の有無 |
|--------|-----|------|-----------|
| `intentQueue_` / `quarantineFallbackQueue_` | `sizeApprox()`（MpscBoundedRing.h:115） | `enqueuePos_ - dequeuePos_` の acquire 読取り | **消費なし**（pop しない） |
| `observeDeferredRing_` / `recoveryIntentQueue_` | `size()`（LockFreeRingBuffer.h:76） | `writeIndex - readIndex` の acquire 読取り | **消費なし**（pop しない） |

- **両 API とも消費（pop）を伴わない**ため、`isFullyDrained()` から安全に呼べる
- `LockFreeRingBuffer::size()` は SPSC 前提だが、**producer 停止後**は正確な占有数（w - r）を返す

**カウンタは audit / diagnostic / invariant counter として扱い、完全 drain の正判定は queue 自体の空（queue emptiness）を source of truth に追加する**。ただし **producer が停止（admission closed）後にのみ有効**。

**対象キュー（明示）**:
```
intentQueue_            （MpscBoundedRing, MPSC — sizeApprox()）
observeDeferredRing_    （LockFreeRingBuffer, SPSC — size()）
quarantineFallbackQueue_（MpscBoundedRing, MPSC — sizeApprox()）
recoveryIntentQueue_    （LockFreeRingBuffer, SPSC — size()）
```

**🔴🔴 七次レビュー（2026-08-09）— queue emptiness と pendingIntentCount_ の独立判定**:
`intentQueue_` には Publish intent も混在するため、**queue emptiness は pendingIntentCount_ と独立に intentQueue_ 全体（全4種）を空にすること**を要求する。つまり:
- **pendingIntentCount_ == 0** は「Observe/Quarantine/Recovery の residnet なし」（Publish は対象外）
- **queue emptiness == 全キュー空** は「Observe/Publish/Quarantine/Recovery の全 transport 残留なし」（Publish を含む）
- **両者が独立に成立して初めて drain 完了**（カウンタと transport の二重化）

これは六次レビューの「accounting が0でも、実際の transport queue が空であることを確認する」方針と整合し、**Publish intent が pendingIntentCount_ 対象外であることが原因で漏れる残留を queue emptiness が捕捉**する。

**🔴 shutdown ordering の強化（四次レビュー）** — queue emptiness は「最後の防衛線」であり、単独では shutdown を完成させない（queue non-empty かつ Coordinator/Builder 停止後は誰も consume しない → waitForDrain timeout）。以下を不変条件にする:

**🔴🔴 七次レビュー（2026-08-09）— waitForDrain 時点で Coordinator/Builder は停止済み（実コード照合）**:
ReleaseResources.cpp の shutdown 順序を実コードで確認:
```
setShutdownPhase(StopWorkers)          ← :188
shutdownCoordinatorLoop()              ← :189（Coordinator join）
stopRebuildThread()                    ← :190（Builder join）
...
waitForDrain(2000, 2)                  ← :447（観測ループ）
```
- **waitForDrain 時点で CoordinatorLoop（processIntent）と RebuildThread（Builder）は両方停止済み**
- `waitForDrain()`（AudioEngine.Threading.cpp:158）は `while (!isFullyDrained())` ループ内で **`drainDeferredRetireQueues(true)` を呼ぶ**（観測 + reclaim 促進）。**queue emptiness を追加しても、Coordinator/Builder 停止後のため queue を消費する者は存在せず、残留 intent は正しく検出される**
- つまり「queue non-empty が waitForDrain で検出される = Coordinator/Builder 停止後の残留」であり、**shutdown ordering が queue emptiness の前提を保証する**

```
StopAcceptingWork
↓
全 Producer 停止（Timer / Transition / Commit 経路）
↓
Producer join
↓
CoordinatorLoop 停止・join（既存: ReleaseResources.cpp:189 shutdownCoordinatorLoop）
↓
RebuildThread 停止・join（既存: :190 stopRebuildThread）
↓
waitForDrain（:447）→ queue occupancy 確認（4キュー）→ counter 確認 → reclaim 確認
↓
ShutdownComplete
```

**🔴🔴 五次レビュー（2026-08-09）— queue emptiness の実装場所と Coordinator 特定を確定**:
- `AudioEngine::isFullyDrained()`（Threading.cpp:114-136）は `runtimePublicationBridge_`（**`convo::isr::RuntimePublicationCoordinator`** — ISRRuntimePublicationCoordinator.h:68）の setter 群 + `isFullyDrained()` を呼ぶ
- **queue emptiness は `ShutdownScheduler::isFullyDrained()`（ISRRuntimePublicationCoordinator.cpp:470）に追加**する。ShutdownScheduler は `RuntimePublicationCoordinator& coordinator_`（:351）を保持し、**intentQueue_ / observeDeferredRing_ / quarantineFallbackQueue_ / recoveryIntentQueue_ に直接アクセス可能**
- **注意 — 同名 Coordinator が2つ存在**: `convo::isr::RuntimePublicationCoordinator`（Intent authority, ISRRuntimePublicationCoordinator.h:68）と `convo::RuntimePublicationCoordinator`（Publish authority, src/core/RuntimePublicationCoordinator.h:24）は別クラス。本設計（§1.1/§1.2）は**前者（ISR）**が対象。core 側は makeRuntimePublicationCoordinator() で一時生成され publishWorld のみに使用（AudioEngine.h:3646）

**🔴🔴🔴 六次レビュー（2026-08-09）— shutdown ordering はコード上の invariant として固定**:
queue emptiness の成立条件（Admission Closed → producer 停止/join → queue 観測）は**設計書だけでなく、コード上の invariant として保証**する。具体的には:
- **`ShutdownScheduler::isFullyDrained()` は「admission closed 状態でのみ queue emptiness を評価する」**ことをコードで固定（`shutdownPhase_` が AudioStopped 以降 かつ producer 経路が閉じたことを前提）
- **queue emptiness の観測は `isFullyDrained()` 呼び出し時点のスナップショット**であり、「queue empty を確認した瞬間に producer が enqueue する」競合は、**shutdown ordering（producer join 後）で排除**する
- 六次レビュー推奨の完全な shutdown invariant:
```
1. StopAcceptingWork
2. AudioStopped
3. producer 停止
4. producer join
5. Coordinator/Builder の処理完了
6. queue occupancy == 0（4キュー）
7. counters == 0
8. EBR/reclaim conditions == 0
9. ShutdownComplete
```
- これを **`waitForDrain()` の不変条件テスト**（Phase 6）で固定する

**範囲確定**: 1.1 の混入廃止（Publish / RetireIntent 上書き）と processIntent の hard reset 廃止のみを本改修で実施。isFullyDrained の**他のカウンタの実測上書き**（fallbackBacklog / retireBacklog / deferredRetire / quarantineResident）は**現状維持**とし、queue emptiness を加える形で強化する。実測上書きの全廃は別タスク（上書きが「drain 判定の正しさ」を担保している面があり、廃止には全カウンタの正確な増減管理が必要で P2 範囲を超える）。

**対象ファイル**: `ISRRuntimePublicationCoordinator.cpp`（ShutdownScheduler::isFullyDrained に queue emptiness 追加）/ `AudioEngine.Threading.cpp`（pendingIntentCount 上書き廃止）
**検証**: admission closed + producer join 後の queue occupancy が drain 判定に反映されること。queue non-empty 時に isFullyDrained が false を返すこと（consumption なしで観測）。既存 Phase 6 テスト回帰。

---

### 1.3 🟡 MpscBoundedRing の producer hole テスト固定（P2-3）

> **✅ 実装完了（2026-08-10）** — §1.3.2 / §1.3.3 の設計どおり実装（P2_IMPL_CHECKLIST.md #16-20）。
>
> - `MpscBoundedRing.h` に `#ifdef CONVO_TESTING` フック（位置ベース `testHoleBlockPos_`、push の CAS 予約成功直後・payload 書込み前。production レイアウト不変）
> - public test API（`testSetHoleBlock` / `testReleaseHole` / `testResetHole` 等）を `CONVO_MPSC_TEST_HOOKS` 内に定義。CMake で `MpscBoundedRingTests` のみ `CONVO_TESTING=1`
> - `MpscBoundedRingTests` に 4 本追加（delayed publication / FIFO order / empty pop false / payload publication ordering）— 「reservation order ≠ publication visibility order」を 2 スレッド deterministic に検証。計 10 本 PASS

#### 1.3.1 現状と設計方針

**現状**: `processIntent`（ISRRuntimePublicationCoordinator_ProcessIntent.cpp:36）の `while (intentQueue_.pop(commonIntent))` は、producer hole（別 producer が slot 予約後未書き込み）に遭遇すると `pop` が false を返し、**ループを途中終了**する。後続 intent は次 CoordinatorLoop tick（1ms）まで遅延。

**★ 検証 — Empty と producer hole は同じ観測値**: Capacity=256, 初期状態 `dequeuePos=0, sequences_[0]=0`:
```
pop 時: diff = sequences_[0] - (0+1) = 0 - 1 = -1
```
1. **空キュー**: `dequeuePos=0, sequences_[0]=0` → diff = -1
2. **producer hole**: Producer が `CAS(enqueuePos_, 0, 1)` 成功直後、`entries_[0]` 書き込み前。`sequences_[0]` はまだ 0 → diff = -1

→ **両者とも diff = -1 で、sequence protocol だけでは空キューと unpublished producer reservation を区別できない**。seq 遷移: 初期化 `sequences_[i]=i`(:54) → push 成功 `seq=pos+1`(:81) → pop 成功 `seq=pos+Capacity`(:109)。`diff < 0` は「まだ pos+1 になっていない」だけで、それが「空」か「予約中」かは判定不能。

**設計方針（確定）**: `PopStatus` API は追加しない。**現行 `bool pop(T&)` の契約を明文化**する:
```
bool pop(T& item) noexcept:
  true  = fully published item acquired
  false = queue empty または producer reservation exists but publication not yet visible
```
- `false` は「安全に停止すべき」を意味する（torn payload を読まない / 未 publication データを読まない / busy-spin しない）
- `processIntent` の `while (pop)` は false で一旦停止し、次 tick で再試行 → **現行実装を維持**
- **cross-type FIFO は「壊れていない」**: reservation order を FIFO order として扱う。A が予約(10)・未 publish、B が予約(11)・publish 済みなら、consumer は 11 を先に処理しない（10 が publish されるまで待つ）。これは **FIFO-preserving backpressure / head-of-line blocking** であり、FIFO violation ではない。

#### 1.3.2 test seam（必須）

現在の `MpscBoundedRing::push()` は **CAS(enqueuePos_) → entries_[slot] = item → publish(sequence) を一つの呼び出しに内包**している。つまり「CAS 成功後・publication 前に別スレッドを停止する」というテストポイントが**公開 API には存在しない**。テスト1/2 を実装するには test-only hook が必要:

**🔴🔴 五次レビュー（2026-08-09）— フックは push() 内部の #ifdef ブロックとして挿入**:
フックは「public メソッド」ではなく、**`push()` の CAS 成功直後・`entries_[pos & kMask] = item` の前に `#ifdef CONVO_TESTING` ブロック**として挿入する（CAS 成功時点で producer hole が形成されるため、その位置が正確）:

```cpp
// MpscBoundedRing.h — 本番コードは変更しない。テスト専用フック。
//   ★ フック用メンバは private セクション末尾（dequeuePos_ の後）に置く。
//     production ビルドでは #ifdef が消滅し、メモリレイアウトは不変（alignas(64) 配列の配置に影響なし）。
#ifdef CONVO_TESTING
    // push の CAS 成功直後・publication 前（entries_ 書込み前）にブロック
    //   テストスレッド（producer）を gate で停止し、consumer が hole を観測する時間を作る
    std::atomic<bool> testHoleGate { false };
    std::atomic<bool> testHoleReady { false };
    // push() 内部で: CAS 成功 → testHoleReady = true → testHoleGate が true になるまで spin
#endif
```

**🔴🔴 七次レビュー（2026-08-09）— メモリレイアウトと cache-line の整合**:
- MpscBoundedRing のメンバは **`alignas(64)`** で cache-line 分離（sequences_ / entries_ / enqueuePos_ / dequeuePos_）される
- **フック用 atomic は production ビルドでは消滅**するため、**production のレイアウト・パディングは不変**（C4324 警告抑制の `#pragma warning(pop)` にも影響なし）
- テストビルドのみフックメンバが private 末尾に追加される（cache-line 境界への影響はテスト専用のため無害）
- **テストの gate spin は production に存在しない**ため、RT コードパスは完全に元のまま（五次レビュー確認済み）

```cpp
// push() 内部（CAS 成功直後）:
if (convo::compareExchangeAtomic(enqueuePos_, pos, static_cast<uint32_t>(pos + 1),
                                 std::memory_order_acq_rel, std::memory_order_acquire))
{
#ifdef CONVO_TESTING
    testHoleReady.store(true, std::memory_order_release);
    while (!testHoleGate.load(std::memory_order_acquire)) { /* spin: consumer が hole 観測 */ }
#endif
    entries_[pos & kMask] = item;  // payload 書込み（publication）
    convo::publishAtomic(seq_atom, static_cast<uint32_t>(pos + 1), std::memory_order_release);
    return true;
}
```
- 本番ビルド（CONVO_TESTING 未定義）では `#ifdef` ブロックが消滅し、**RT コードパスは完全に元のまま**（コードサイズ・実行時間とも無影響）
- **CONVO_TESTING は新規マクロ**: 現在のソース・CMake に存在しない。CMakeLists.txt の MpscBoundedRingTests に `target_compile_definitions(MpscBoundedRingTests PRIVATE CONVO_TESTING=1)` を追加。本番ターゲット（ConvoPeq / AudioEngineHarness）には定義しないこと。

**🔴 四次レビュー指摘 — テストは「単一スレッド」ではなく「2スレッド deterministic」必須**:
`push()` の内部（CAS reserve → hook → payload write → sequence publish）で hook が producer を停止している間に consumer を動かすには**別スレッドが必要**。単一スレッドでは「producer が停止中に consumer が pop する」を実現できない。

```
Producer thread:                    Consumer thread:
  CAS reservation
       ↓
  test gate wait  ──────────────→   pop() → false（hole を観測）
       ↓
  gate release
       ↓
  payload write
  sequence release
                                    pop() → true（publication 後）
```

**対象ファイル**: `src/MpscBoundedRing.h`（CONVO_TESTING フックのみ）/ `src/tests/MpscBoundedRingTests.cpp`（テスト追加）/ `CMakeLists.txt`（CONVO_TESTING 定義）
**検証**: 上記4テストを**実際に別スレッドで**実行し、producer hole の FIFO-preserving と payload publication ordering を確認。`processIntent` の cross-type FIFO 順序保証テスト（REPAIR_PLAN2.md:1240 の INV-7）。

#### 1.3.3 テスト仕様（4本・2スレッド deterministic）

**🔴 五次レビュー（2026-08-09）— 既存テストとの関係を確定**:
- 既存 `testProducerHoleDoesNotJumpAhead()`（MpscBoundedRingTests.cpp:238）は「空状態で pop が false（穴を跨がない）」を検証するが、**真の producer hole（CAS 予約後に publication 前の遅延）を生成していない**。dash §1.3 のテスト1/2 は依然として必要
- **既存テストと命名衝突しないこと**: 既存名 `testProducerHoleDoesNotJumpAhead` と区別するため、新テストは `testProducerHoleWithDelayedPublication` / `testProducerHoleFifoOrder` 等に命名する
- 既存テストは既に `<thread>` を使用（:44）し、`testMultiProducerNoLoss`（:123）で並行 push を実装済み。**2スレッド deterministic test は既存フレームワークと自然に統合できる**

```
1. testProducerHoleWithDelayedPublication:
   Producer reserves slot → publication 意図的に遅延（gate で停止）→ consumer pop == false
   → producer release → consumer pop == true → payload intact
2. testProducerHoleFifoOrder:
   A reserves N, B publishes N+1, A delayed → consumer は B を消費しない
   → A publishes → A → B 順序保持（FIFO invariant 直接検証）
3. Empty queue で pop == false（連続 pop で false）— 既存 testProducerHoleDoesNotJumpAhead と重複確認
4. payload publication ordering: payload write → release sequence publish の
   memory-order を検証（consumer が acquire sequence 後に pop した payload が intact）。
   ISR memory-order invariant — entries_[pos]=item; publishAtomic(seq, pos+1, release) の構造を直接検証
```

全テストで **producer と consumer を別スレッド**にし、gate（条件変数）で deterministic に制御する。単一スレッドでは producer hole 中の consumer 動作を検証できない。

---

### 1.4 🟡 invariant_INV3_INV5 テスト追加（P2-2）

> **✅ 実装完了（2026-08-10）** — §1.4.2 / §1.4.3 の設計どおり実装（P2_IMPL_CHECKLIST.md #21-29）。
>
> - `src/tests/invariant_INV3_INV5.cpp` 新規（INV-3-1 / INV-3-2 / INV-5-1 / INV-5-2）
> - `TestEpochProvider`（IEpochProvider スタブ）で router の `currentEpoch` / `minReaderEpoch` を制御し、INV-3 の安全/非安全を deterministic 化
> - CMake: `invariant_INV3_INV5Tests` を ISRSemanticValidationTests と同一パターンで追加（add_test `InvariantINV3INV5`）
> - **🔴 変更内容（監査補正 — 追加テスト 2026-08-10）**: §1.2 の監査補正（Step B/C）に伴い **INV-5-3 / INV-5-4 を追加**:
>   - INV-5-3: `requestShutdown()`（AdmissionClosed）後の submit → enqueue されず `recoveryShutdownDiscardCount+1`・pending 不変・drop と区別
>   - INV-5-4: `discardRecoveryRequestsOnShutdown()` が残留を明示 discard（queue empty + pending 0 + discard カウント）

#### 1.4.1 現状

計画書（REPAIR_PLAN2.md:1181）は「`src/tests/invariant_*.cpp` 形式で各 invariant のテストを追加」と要求。しかし `invariant_*.cpp` は存在しない。以下の recent fixes がテスト・ハーネスで一切参照されていない:
- (a) `requestReclaim` bool 化 + `pendingReclaimHandles_` 再試行（INV-3 retire 順序）
- (b) `retireByHandle` への `requestReclaimHandle` 追加（Observe 経路 slot リーク）
- (c) Recovery 発行経路（`QuarantineIntentHandler` → `submitRecoveryIntent`）
- (d) `recoveryIntentDropCount` 監視（INV-5）

#### 1.4.2 テスト仕様

**🔴🔴 五次レビュー（2026-08-09）— INV-3 テストの依存関係を確定**:
`requestReclaim`（ISRRuntimePublicationCoordinator.cpp:573）は **`DSPHandleRuntime&` / `ISRRetireRouter&` を引数に取り**、`router.currentEpoch()` / `router.minReaderEpoch()` / `handleRuntime.retire()` / `handleRuntime.reclaim()` に依存する。**単体テストでは epoch の安全/不安全を制御できるスタブが必要**:
- `ISRRetireRouter` スタブ: `currentEpoch()` / `minReaderEpoch()` を制御可能にし、「epoch 不安全→ false → pending 再登録」を deterministic に検証
- `DSPHandleRuntime` スタブ: `retire()` / `reclaim()` を記録し、「Retired → Reclaimed 遷移」「Quarantined を Retired に上書きしない（isRetired ガード）」を検証
- `pendingReclaimHandles_` は **AudioEngine のメンバ（AudioEngine.h:4616）** のため、**AudioEngineHarness 経由**（`requestReclaimHandle` 呼び出し → pending 登録 → drainDeferredRetireQueues 再試行）か、**requestReclaim の戻り値検証**で間接的に固定する

**🔴🔴 七次レビュー（2026-08-09）— INV-5 の drop テスト方法を確定**:
- 既存 `testRecoveryRequestEnqueueAndPop()`（ISRSemanticValidationTests.cpp:608）は `submitRecoveryRequest → popRecoveryRequest` の 1-hop 輸送を検証済み。INV-5 はこれを拡張して **recoveryIntentQueue_ を意図的に full にする**必要がある
- **recoveryIntentQueue_ は SPSC（LockFreeRingBuffer, 256 容量）**。full にする方法:
  1. `submitRecoveryRequest` を 256 回呼び、consumer（popRecoveryRequest）を呼ばない → queue full
  2. 257 回目の `submitRecoveryRequest` → push 失敗 → `recoveryIntentDropCount_++` + `pendingIntentCount_` 不変
- **注意**: 256 回の submit は `pendingIntentCount_` を 256 まで増やす。INV-5-1 の「pendingIntentCount が増えないこと」は **drop 時の増加なし**（257 回目）を検証する
- テストは HealthMonitor tick（AudioEngine.h:1546,1620 の drop counter → Critical）まで通す

新規テスト `src/tests/invariant_INV3_INV5.cpp` を追加:

```cpp
// invariant_INV3_INV5.cpp — INV-3（retire 順序）と INV-5（Intent loss）の回帰テスト
//   CMakeLists.txt に add_executable / add_test を追加
//   ★ 依存: ISRRetireRouter / DSPHandleRuntime のスタブ（epoch 安全制御用）

// INV-3-1: retire → epoch 安全確認 → reclaim の順序（requestReclaimHandle 経由）
//   - retireDSPHandleForRuntime 後に handle が Retired → Reclaimed に遷移すること
//   - epoch 不安全時に pendingReclaimHandles_ に登録され、drain で再試行されること
//   - quarantineSlot 経路で Quarantined に遷移した handle を requestReclaim が
//     Retired に上書きしないこと（isRetired ガード）— ABA/state-ownership
// INV-3-2: requestReclaim が false を返した場合、handle が保留リストに再登録されること（TOCTOU）
// INV-5-1: submitRecoveryRequest push 失敗時に recoveryIntentDropCount_ が増え、
//          pendingIntentCount が増えないこと
// INV-5-2: QuarantineIntentHandler が quarantine 成功（stateChanged=true）時のみ
//          Recovery を発行すること
```

**実装形式（確定）**: `invariant_*.cpp` は現状存在しない。既存テストは `*Tests.cpp` + CMake の `add_executable` + `add_test` パターン（例: MpscBoundedRingTests = CMakeLists.txt:322-326）。invariant_INV3_INV5.cpp も**同パターンで追加**（`add_executable(invariant_INV3_INV5Tests src/tests/invariant_INV3_INV5.cpp)` + `add_test`）。テストハーネス（JUCEDummy 等）は `ISRSemanticValidationTests` の構成を踏襲。

#### 1.4.3 INV-5 の定義（確定）

現在のコードは「★ INV-5: Recovery drop 禁止」（ISRRuntimePublicationCoordinator.cpp:661）とコメントしながら、実際は full 時に `recoveryIntentDropCount_++` で drop + telemetry する。**これは「drop 禁止」という仕様と「現実は drop する」という実装の mismatch** である。INV-5 を以下のように明確化する:

```
INV-5（確定）:
  Recovery request loss must never be silent.

  Normal:      enqueue success → Builder consumes
  Saturation:  enqueue failure → recoveryIntentDropCount_++ + Critical health（telemetry 必須）
  Forbidden:   enqueue failure → no telemetry
               enqueue failure → pendingIntentCount_++  （residency 不正）
               enqueue failure → false success
```

- テストは「Recovery が静かに消えない」ことを固定する
- **🔴 四次レビュー指摘 — Critical 昇格は HealthMonitor 経由で検証**: 「drop → Critical」は `submitRecoveryRequest()` の同期的な戻り値ではない。実際には **HealthMonitor の tick() が `recoveryIntentDropCount` の delta を監視し、Critical 相当へ昇格**する（AudioEngine.h:1546,1620 で `recoveryIntentDropCount()` が HealthMonitor 入力となる）。したがってテストは:

```
queue full
→ recoveryIntentDropCount_++
→ HealthMonitor tick（drop counter delta 監視）
→ ISRHealthState::Critical（または Critical 昇格経路）
```

まで通す必要がある（`submitRecoveryRequest` の戻り値検証だけでは不十分）。
- **🔴🔴 六次レビュー（2026-08-09）— drop を memory corruption と同一視しない**: Recovery queue full による drop は **runtime safety と functional recovery guarantee を分けて考える**べき:
```
Recovery lost
→ current DSP remains quarantined
→ no unsafe DSP access（runtime safety は維持）
→ but recovery is not performed（functional/availability は喪失）
```
HealthMonitor の escalation は必要だが、**`UAF` と同一レベルの memory safety violation として扱わない**方が semantic に正確。INV-5 の仕様名は「**Recovery Intent の silent loss 禁止**」とし、「drop == memory corruption」とは定義しない。
- **真に「drop 絶対禁止」が必要なら、bounded 256 SPSC ring 自体の再設計が必要**（単なる P2 修正ではない — 別タスク）。現状は「silent loss 禁止」で合意する

**対象ファイル**: `src/tests/invariant_INV3_INV5.cpp`（新規）/ `CMakeLists.txt`
**検証**: `ctest` で INV-3/INV-5 テストが green。INV-5 は HealthMonitor tick を通した Critical 昇格まで検証（drop は memory corruption ではなく functional loss として扱う）。既存 `ISRSemanticValidationTests` / `MpscBoundedRingTests` の回帰。

---

## 2. 追加対応の改修設計（旧 Appendix A-1〜A-4 を設計化）

本セクションは、旧 Appendix A-1（P3 降格）・A-2（撤回案の代替）・A-3（既知事項）・A-4（残余リスク）を**設計**として取り込み、各項目の**対応方法を詳細設計**したものである。

### 2.1 per-type admission policy の統一機構（旧 A-1.1）— **P3 対応設計**

#### 2.1.1 現状

`submitObserve` / `submitQuarantine` / `submitRecoveryRequest` / `enqueuePublicationIntent` の4関数に per-type の overflow 処理が**個別実装**されている。統一 `AdmissionPolicy` エンジンは存在しない。

**🔴 別視点調査（2026-08-10）— 実コード照合**: 4 関数の現状は実コードと整合（submitObserve: intentQueue_→observeDeferredRing_→drop / submitQuarantine: intentQueue_→quarantineFallbackQueue_→drop / submitRecoveryRequest: recoveryIntentQueue_→drop〔fallback ring なし〕/ enqueuePublicationIntent: intentQueue_→ownerChannel 回収）。P3 で統一 `IntentAdmissionPolicy`（Decision-only・副作用は type 固有）を導入。本項は P3 アーキテクチャ整理（既存バグの修正ではない）。

| 関数 | 現在の overflow 処理 |
|------|---------------------|
| `submitObserve` | intentQueue_ → observeDeferredRing_ → drop カウンタ（条件付き drop / coalesce 可） |
| `submitQuarantine` | intentQueue_ → quarantineFallbackQueue_ → drop カウンタ + Critical 昇格 |
| `submitRecoveryRequest` | recoveryIntentQueue_ → drop カウンタ + Critical 昇格（**fallback ring なし**） |
| `enqueuePublicationIntent` | intentQueue_ → ownerChannel 回収 + Failed |

#### 2.1.2 設計方針

**判断**: P3 アーキテクチャ整理（既存バグの修正ではない）。現在の各 submitXxx は計画書の per-type admission policy を**機能的に満たしている**。「統一」は決定のみ共通化し、副作用（ownerChannel 回収 / Critical 昇格 / drop カウンタ）は type 固有のまま。

**🔴 四次レビュー指摘 — 単一 action ではなく「staged admission state machine」が必要**:
Observe は `primary intentQueue → observeDeferredRing → drop` という**段階的 admission** であり、単一の `OverflowAction` 3分類（FallbackRing / OwnerReclaim / DropWithCounter）では正確に表現できない。type → 単一 action ではなく、**type → admission state machine** とする。

#### 2.1.3 対応方法（詳細設計）

**① ヘルパー導入**: `IntentAdmissionPolicy`（`ISRRuntimePublicationCoordinator.h` に追加）。Decision-only（副作用なし）、呼出し元が policy に従って退避する。**staged admission を表現するため、段階ごとの action を保持**:

```cpp
struct IntentAdmissionPolicy {
    enum class AdmissionStep : uint8_t {
        Primary,    // 一次キュー（intentQueue_）
        Fallback,   // 二次キュー（observeDeferredRing_ / quarantineFallbackQueue_）
        Drop        // 全段失敗 → drop + telemetry
    };
    enum class OverflowAction : uint8_t {
        FallbackRing,   // 専用 fallback ring へ退避
        OwnerReclaim,   // 呼出し元が所有権回収（Publish）
        DropWithCounter // drop カウンタ + HealthEvent/Critical 昇格
    };
    // 各 AdmissionStep で type が取る action を返す（staged admission）
    static constexpr OverflowAction actionFor(IntentType type, AdmissionStep step) noexcept {
        switch (type) {
            case IntentType::Observe:
                return (step == AdmissionStep::Primary) ? OverflowAction::FallbackRing
                                                        : OverflowAction::DropWithCounter;
            case IntentType::Publish:
                return OverflowAction::OwnerReclaim;   // primary/full とも owner 回収
            case IntentType::Quarantine:
                return (step == AdmissionStep::Primary) ? OverflowAction::FallbackRing
                                                        : OverflowAction::DropWithCounter;
            case IntentType::Recovery:
                return OverflowAction::DropWithCounter; // fallback ring なし（既定）
        }
        return OverflowAction::DropWithCounter;
    }
};
```

**② 4関数のリファクタ**（内部ロジック変更のみ、公開 API 不変）:
- `submitObserve`: `actionFor(Observe, Primary)` = FallbackRing → observeDeferredRing_ へ退避。`actionFor(Observe, Fallback)` = DropWithCounter → drop カウンタ
- `submitQuarantine`: `actionFor(Quarantine, Primary)` = FallbackRing → quarantineFallbackQueue_ へ退避。`actionFor(Quarantine, Fallback)` = DropWithCounter → drop + Critical 昇格
- `submitRecoveryRequest`: `actionFor(Recovery, _)` = DropWithCounter → drop カウンタ + Critical 昇格（fallback なし）
- `enqueuePublicationIntent`: `actionFor(Publish, _)` = OwnerReclaim → ownerChannel 回収

**③ 監視統合**: drop カウンタ → HealthEvent を統一ヘルパー `IntentDropMonitor` に集約（drop 時に `fetchAddAtomic` + HealthMonitor 昇格の定型処理）。

**対象ファイル**: `src/audioengine/ISRRuntimePublicationCoordinator.h` / `.cpp`
**検証**: 単体テストで4 type × 各 AdmissionStep の overflow action が計画書表（REPAIR_PLAN2.md:858-860）と一致すること。既存 `MpscBoundedRingTests` / `ISRSemanticValidationTests` の回帰。

### 2.2 Recovery の coalesce（マージ）実装（旧 A-1.2）— **四次レビューで NO-GO（実装削除）**

#### 2.2.1 現状

`submitRecoveryRequest`（ISRRuntimePublicationCoordinator.cpp:643-669）は、同一 quarantinedHandle の重複 Recovery を**マージせず**個別に push する。計画書（REPAIR_PLAN2.md:860）は「Recovery は drop 禁止。ただし coalesce 可能」と定義。

#### 2.2.2 🔴🔴 四次レビュー — 現提案は NO-GO（正当な Recovery を silent loss する）

`submitRecoveryRequest`（ISRRuntimePublicationCoordinator.cpp:643-669）は、同一 quarantinedHandle の重複 Recovery を**マージせず**個別に push する。計画書（REPAIR_PLAN2.md:860）は「Recovery は drop 禁止。ただし coalesce 可能」と定義。

**現提案（lastRecoveryHandle_ tracking）は以下のバグを持つため NO-GO**:

```
A を push
↓
Builder が A を pop
↓
A が再び Recovery 要求
↓
lastRecoveryHandle_ == A が残ったまま
↓
if (lastRecoveryHandle_ == quarantinedHandle) return;   ← 正当な Recovery を破棄！
```

- **Producer は Consumer の pop を検知できない**。`lastRecoveryHandle_` が A のまま残るため、**A が pop 後に再 Recovery 要求された場合も coalesce 扱いで破棄**される
- これは最適化上の問題ではなく、**正当な Recovery Intent の silent loss**（INV-5 違反）
- **二次レビューの「Producer は Consumer の pop を検知できない」という認識から、それでも lastRecoveryHandle_ を保持して coalesce するという結論は論理矛盾**
- **latest-wins も証明不足**: 同一 handle の Recovery A, B で「B の buildSource が必ず正しい」は現在のソースから保証できない（Recovery は quarantined DSP を除外した現在の authoritative configuration の再 build。coalesce するなら `same handle + same semantic generation + same recovery reason` を条件に定義すべき）

#### 2.2.3 対応方法（詳細設計）— **coalesce は今回は削除**

**🔴 四次レビュー推奨（採用）: 今回の改修では Recovery coalesce を実装しない**。正確な FIFO transport を先に完成させる。

候補3案（将来の P3 で選択）:
- **案 A: producer-visible consumption sequence 追加**: `lastProducedRecoverySerial` / `lastConsumedRecoverySerial` を Producer が観測して tracking を reset。ただし設計が重くなる
- **案 B: queue に peek を追加して Consumer 側で coalesce**: `pop A → peek next → if (next.handle == A) discard/merge`。ただし「latest buildSource wins」の意味論を明確化する必要
- **案 C: coalesce 自体をやめる（推薦）**: Recovery は 256 件の bounded queue で、通常の Recovery 頻度が高くなければ、**まず正確な FIFO transport を保証し、coalesce は別最適化として後回し**が ISR として最も安全

**今回の設計（確定）**: coalesce は実装しない。現行の「重複 Recovery をそのまま push」を維持（正確な FIFO transport）。

**対象ファイル**: 変更なし
**検証**: 既存 `ISRSemanticValidationTests.cpp:609-622`（submitRecoveryRequest → popRecoveryRequest 1-hop 輸送）の維持。coalesce 実装は別タスク（P3）として記録。

---

## 3. 撤回案の代替対応設計（旧 Appendix A-2 を設計化）

### 3.1 PopStatus API の代替 — **現行 pop() 契約の維持とテスト固定（§1.3 と一体）**

**撤回理由**: 旧案は `diff = seq - (pos+1)` の符号で `Empty`/`Hole` を区別しようとしたが、**sequence protocol だけでは区別不能**（§1.3 の検証参照）。`diff < 0` は「まだ pos+1 になっていない」だけで、それが「空」か「予約中」かは判定不能。**`PopStatus` API の導入は不可**。

**代替対応（§1.3 に設計済み）**: `PopStatus` を追加せず、現行 `bool pop(T&)` の契約（false = empty または unpublished reservation）を明文化 + CONVO_TESTING test hook + テスト4本。**§1.3 を参照**。

### 3.2 force reclaim の代替 — **正常 reclaim パイプライン維持と Faulted 可視化**

#### 3.2.1 現状

`~AudioEngine`（CtorDtor.cpp:185-224）の DrainRetire フェーズで `drainDeferredRetireQueues(true)`（:222）が保留 reclaim 再試行を実行する。Audio Thread 停止後は `activeReaderCount()==0`（:202 で待機）→ `minReaderEpoch` が最新 epoch に進むため、通常は `requestReclaim` が成功し `reclaimInFlightCount_` が 0 に戻る。

**実コード照合 — 既存の 5 秒タイムアウト forcing drain フォールバック**: `~AudioEngine`（CtorDtor.cpp:194-221）は既に **Graceful Drain ポーリング（最大 5000ms / 10ms 間隔）** を実装している:
```cpp
// CtorDtor.cpp:200-217
while (waitedMs < kGracefulDrainMaxMs)  // 5000ms
{
    if (m_retireRouter->pendingRetireCount() == 0
        && m_retireRouter->activeReaderCount() == 0) break;
    m_retireRouter->publishEpoch();   // epoch 前進で reclaim 促進
    m_retireRouter->tryReclaim();     // 正常 reclaim 試行
    waitedMs += kGracefulDrainPollMs;
}
if (waitedMs >= kGracefulDrainMaxMs)
    diagLog("[AUDIT] Graceful drain timeout ... forcing drain");
```
この「forcing drain」は **`publishEpoch` + `tryReclaim` の促進**であり、`requestReclaim` を強制するものではない（**EBR 安全を壊さない**）。「force しない」方針は既にコードで実践されている。

#### 3.2.2 対応方法（詳細設計）

**判断**: force reclaim は**実装しない**（EBR 安全を壊す）。既存の Graceful Drain フォールバック + `requestReclaim` の戻り値尊重を維持。

- **force reclaim は実装しない**。現状の「`requestReclaim` の戻り値を尊重し、false なら Faulted」は EBR 安全を正しく保つ
- `pendingReclaimHandles_` は**正常パイプラインで処理できた分だけ**削除し、処理できなかった分は残す（または診断ログに記録）
- stuck Reader の検出は既存 `detectStuckReaders` / Faulted 遷移に委ねる
- **Faulted ≠ memory safety 保証**: `Faulted` は「正常 shutdown invariant を満たせなかった」という**診断状態**。テストでは `Faulted 遷移 + pendingReclaimHandles_ を無条件 clear していない + 未 reclaim handle を再利用可能状態に戻していない` まで確認する

**🔴 四次レビュー指摘 — `shutdownReclaim()` の残存は別問題（二系統の正確な認識）**:
「force reclaim は存在しない」と完全には言えない。最新ソースでは releaseResources の VerifyDrained 段階で:
```
dspHandleRuntime_.retire(handle)
dspHandleRuntime_.shutdownReclaim(handle)   // ReleaseResources.cpp:415,420
```
が存在する。正確には:
```
通常 reclaim:    EBR safety 確認（requestReclaim）
shutdown-specific reclaim: shutdown フェーズで reader 停止後に許可（shutdownReclaim）
```
という**二系統**。これは R4 認識（§5「shutdown 専用経路は残る」）と整合する。**P2 で無理に統合しない判断は正しい**（shutdownReclaim の全廃は別タスク）。

**🔴🔴 八次レビュー（2026-08-09）— shutdown lifetime contract を将来タスクとして明文化**:
八次レビュー §18 の推奨により、**shutdown 側の「本当に reader が存在しないことを何によって保証しているか」を将来タスクとして明文化**する。二系統の現状を維持しつつ、以下の shutdown lifetime contract を別タスクで確定する:
```
Shutdown reclaim の安全性保証:
  - Audio Thread 停止（activeReaderCount()==0 待機）は「当該 handle の reclaim 安全性」の十分条件か？
    （四次レビュー: 否 — 非 Reader スレッドが参照し得る）
  - shutdownReclaim が安全に呼べるのは「全 Reader 停止 + epoch settle + 全 producer 停止」の後
  - この契約を明文化し、shutdownReclaim の使用箇所（ReleaseResources.cpp:415,420）に注記を追加
```
**将来タスク記録**: shutdownReclaim の全廃 or shutdown lifetime contract の明文化（P3/R4 相当）。P2 では現状の二系統を維持。

**★ R4 詳細設計との関連（2026-08-10・外部レビュー反映）**: 本項の「shutdown lifetime contract の将来タスク」は、**R4 詳細設計（§6.3 末尾）の Phase 0/3 で確定された `ShutdownQuiescenceProof`** がその答え。Audio Thread 停止（activeReaderCount()==0）は十分条件ではなく、**全条件（admissionClosed / producersJoined / coordinatorStopped / builderStopped / audioStopped / readerRegistrationClosed / readersZero / epochSettled）の proof が必要**（R4 NG4 と整合 — `activeReaderCount()==0` だけでは不十分）。shutdownReclaim の二系統は R4 の Phase 1-7（ReclaimMode 導入 → RuntimeEBR 一本化 → ShutdownQuiescent → deprecated → API 削除）で統合。**R4 実装時に本項の契約を Phase 0 として固定**する。

**対象ファイル**: `AudioEngine.CtorDtor.cpp`（変更なし or 診断ログ追加のみ）
**検証**: shutdown 時に pendingReclaimHandles_ が残った場合、**Faulted に遷移 + 未 reclaim handle を clear/再利用しない**ことをテストで固定（Phase 6 テスト）。正常系では ShutdownComplete に到達すること。shutdownReclaim 二系統の存在をテスト・文書で認識。

---

## 4. 既知事項の対応設計（旧 Appendix A-3 を設計化）

### 4.1 2つの ShutdownPhase enum の非1:1対応（旧 A-3.1）— **P3 対応設計**

#### 4.1.1 現状

`AudioEngine::ShutdownPhase`（AudioEngine.h:2521, `enum class : int` — **7値: Running/StopAcceptingWork/StopAudio/StopWorkers/ForceEpochAdvance/DrainRetire/Destroy**）と `convo::isr::ShutdownPhase`（ISRShutdown.h:25, `uint8_t` — 11値）の2種類が共存。**非1:1**（`StopWorkers` が isr の3フェーズを駆動）。

**🔴 別視点調査（2026-08-10）— enum 値数の確定**: `AudioEngine::ShutdownPhase` は **7 値**（`Running` を含む。dash 旧記述「6値」は Running を数え漏れ）。`convo::isr::ShutdownPhase` は **11 値**（`Running` 含む）。対応表の実コード検証（ReleaseResources.cpp）: 行番号は正確（:73→:74, :115 遷移なし, :188→:191, :194→:196/:197, :199→:308→:407→:537）。**追加確認**: `EmergencyDrain`（:315）は `CONVOPEQ_EMERGENCY_DRAIN` 定義時のみ遷移（ISRShutdown.h:29-31 — デフォルトではスキップ）。四次レビューの Normal/Failure/Emergency 3 系統テストと整合。

対応表（実測 ReleaseResources.cpp）:
```cpp
//   AudioEngine::ShutdownPhase  →  convo::isr::ShutdownPhase
//   StopAcceptingWork (:73)     →  AudioStopped (:74)
//   StopAudio (:115)            →  （isr を直接遷移させない）
//   StopWorkers (:188)          →  ObserverDrained (:191) → RetireClosed (:196) → EpochSettled (:197)
//   ForceEpochAdvance (:194)    →  EpochSettled（既に :197 で遷移済み）
//   DrainRetire (:199)          →  ReclaimComplete (:308)
//   （isr 独自・Emergency）     →  EmergencyDrain (:315) — CONVOPEQ_EMERGENCY_DRAIN 時のみ（デフォルトではスキップ）
//   （isr 独自）                →  VerifyDrained (:407) / ShutdownComplete (:537)
//   Destroy                     →  ShutdownComplete（~AudioEngine 側）
```

**対応の曖昧さ**: `ForceEpochAdvance` は isr の追加遷移を行わない。**1対多の対応**であり、`switch` の網羅だけでは不整合を検出できない。

#### 4.1.2 対応方法（詳細設計）

**① 対応表の明示**: `AudioEngine.h` に上記の対応表をコメントとして明示（`ShutdownPhase` enum 定義直後に配置）。enum 追加時は対応表の更新を強制する。

**② 遷移シーケンス invariant テスト**: `switch` 網羅では検出不可なため、**transition history**（AudioEngine の各 setShutdownPhase と isr の各 transitionTo の実測順序）をテストで固定:

```cpp
// invariant テスト: 遷移シーケンスの固定（実測 ReleaseResources.cpp:73-537）
//   AudioEngine::StopAcceptingWork → isr::AudioStopped
//   AudioEngine::StopWorkers → isr::ObserverDrained → RetireClosed → EpochSettled
//   AudioEngine::DrainRetire → isr::ReclaimComplete → VerifyDrained → ShutdownComplete
```

テストは `shutdownRuntime_.getPhase()` の遷移履歴（または transitionTo のログ）を検証し、上記シーケンスが不変であることを固定する。

**🔴 四次レビュー指摘 — 正常系だけでなく Timeout / Emergency も分離してテスト**:
`waitForDrain()`（AudioEngine.Threading.cpp:138-152）は `EmergencyDrain` / `TimedOut` / `Failed` / `ShutdownComplete`（ISRShutdown.h:36-40）の各フェーズを許容する。テストは **Normal / Failure / Emergency の3系統に分離**する:

```
Normal:    StopAcceptingWork → AudioStopped → ObserverDrained → RetireClosed
           → EpochSettled → ReclaimComplete → VerifyDrained → ShutdownComplete
Failure:   ... → VerifyDrained → TimedOut
Emergency: ... → EmergencyDrain → ...
```

**対象ファイル**: `AudioEngine.h` / `AudioEngine.CtorDtor.cpp` / `AudioEngine.Processing.ReleaseResources.cpp`
**検証**: invariant テストで遷移シーケンスが実測順序どおりであること（enum 値の1:1対応ではなく遷移順序を検証）。Normal / Failure / Emergency の3系統を分離して検証。

### 4.2 PublishReceiptWaiter の high-water mark（旧 A-3.2）— **P3 対応設計（現状維持 + 不変条件テスト）**

#### 4.2.1 現状

AudioEngine.h:3607（実測は complete() 内の `if (seqId > lastCompleted_) lastCompleted_ = seqId`）の high-water mark。後の seqId が先に完了すると、先の seqId の `waitFor` が即 true になる。

**実コード照合（2026-08-09）**: `PublishReceiptWaiter::complete` は `std::lock_guard<std::mutex>` 保護下で `if (seqId > lastCompleted_) lastCompleted_ = seqId;` を実行し `cv_.notify_all()`。`waitFor` は `unique_lock` + cv 待機。**mutex 保護の high-water mark** であり、SPSC 前提で正しい。

**🔴🔴 四次レビュー指摘 — MPSC ring の FIFO と PublicationSequenceId の順序は同一ではない**:
- `publicationSequenceCounter_`（AudioEngine.h:2189）は `fetchAddAtomic`（:3412）で seqId を割り当てる。**MPSC ring は reservation order で FIFO** だが、seqId allocation が別の atomic で行われるため:
```
Producer A: seq=10 を取得 → pause
Producer B: seq=11 を取得 → enqueue
→ queue reservation order: 11, 10（A の enqueue が遅れる）
```
- つまり「**MPSC なので high-water mark は安全**」とは言えない。正確には「**実際の Producer serialization によって completion sequence が単調増加することが保証されているなら安全**」
- **現状の SPSC（single consumer = CoordinatorLoop）+ intentQueue_ FIFO 処理により completion が seqId 順に発生する**ため high-water mark は正しいが、これは**不変条件として明示的に固定すべき前提**である

#### 4.2.2 対応方法（詳細設計）

**判断**: 現状維持。ただし **completion sequence の単調性（monotonicity）を明示的な invariant として固定**する。

**🔴🔴 八次レビュー（2026-08-09）— Producer serialization による seqId monotonicity の保証をテスト固定**:
- seqId は `fetchAddAtomic(publicationSequenceCounter_, 1)`（AudioEngine.h:3412）で割り当てられ、Publish intent に `intent.sequenceId = seqId` として入る（:4405）
- コメントにも「**executePublish は intentQueue_ を FIFO で処理するため seqId は単調増加で完了する（順序性前提）**」（AudioEngine.h:3605）と明記
- **今すぐ API 変更は不要**。ただし、`submit → queue → processIntent → completion` の実際の **Producer serialization が seqId monotonicity を保証していること**をテストで固定する（八次レビュー §19）

**対応**:
- **不変条件テスト固定**: 「completion が seqId 順に発生する」ことを `ISRSoakTests` 相当で検証。**10 complete → 11 complete** の順序に加え、**11 complete → 10 not complete** を意図的に作るテストを追加（high-water mark なら `11 complete → lastCompleted=11 → waitFor(10)=true` になる。これが仕様上許されるかを明確にする）
- **FIFO completion invariant が保証される限り high-water mark は正しい**（`waitFor(10)` が 11 完了時に true になるのは「10 も完了した」を意味するため安全）。保証されない場合は per-seqId FIFO 化が必要
- **Producer serialization テスト**: 実際の Publish 経路（`reserveNextRuntimeGraphGeneration` → seqId 割当 → enqueuePublicationIntent → processIntent → executePublish → onPublishCommitted → notifyPublishReceipt）が seqId 順に completion を発生させることを固定
- **将来の再順序化設計（deferred commit 拡張・per-type admission が Publish 実行順序を seqId 順から逸脱させる場合）を入れる場合のみ**、per-seqId FIFO（完了済み seqId の集合）への拡張を検討

```cpp
// 将来拡張案（reorder 設計導入時のみ）: high-water mark → per-seqId FIFO
//   現状: lastCompleted_（単調増加, mutex 保護）
//   拡張案: 完了済み seqId の集合（bitmask または ordered set）+ waitFor は集合確認
```

**対象ファイル**: `AudioEngine.h`（PublishReceiptWaiter）
**検証**: completion seq monotonicity invariant（10→11 順序 + 11 先行完了時の waitFor(10) 仕様確認）+ Producer serialization テスト（:3412 の seqId 割当 → completion の順序）。既存テスト回帰。

### 4.3 ConvolverProcessor の LinearRamp 分離（旧 A-3.3）— **四次レビュー承認（対象外・文書化）**

`ConvolverProcessor.h:910,935,945` の `latencySmoother`/`crossfadeGain`/`mixSmoother` は CrossflareRuntime の `gain_`/`dryScaleGain_` と別個。設計判断（対象外）。将来の独立 RT-safety 検証を文書化:

```cpp
// ConvolverProcessor.h — 将来検証メモ
//   mixSmoother / latencySmoother / crossfadeGain は NonRT（prepareToPlay）で
//   reset()/setCurrentAndTargetValue() される。RT は getNextValue()/isSmoothing() で読む。
//   CrossflareRuntime の gain_/dryScaleGain_ と同様の RT-only ownership 契約を
//   将来独立に検証する（ConvolverProcessor 固有の課題）。
```

**対象ファイル**: `ConvolverProcessor.h`（検証メモのみ）
**検証**: Phase 6（soak）で ConvolverProcessor の LinearRamp に RT↔NonRT race がないこと。

### 4.4 BlockDouble の finalizeCrossfadeMixPath(..., false)（旧 A-3.4）— **四次レビュー承認（対象外・実測後判断）**

BlockDouble.cpp:434 は `finalizeCrossfadeMixPath(dsp, fading, false)`、AudioBlock.cpp:458 は `true`。double パスで dryScaleGain_ がリセットされない。**BlockDouble の false は「意図的な差異」である可能性が高い**（double パスは異なる crossfade 完了条件を持つ）。単純な true 統一は**推奨しない**。

**四次レビュー指摘 — Authority の観点で判断すべき**: 「似たコードだから同じ状態遷移にする」ではなく、**その状態を誰が Authority として管理しているか**で判断すべき。現ソース上、SnapshotCoordinator と CrossfadeRuntime は独立した機構として扱われている。**今回の P2 改修とは分離**し、Phase 6 soak の実測で判断する。

**対応**: BUG-028 修正後の整合を Phase 6 soak で**実測**してから判断（挙動変更を伴うため、実測なしの変更は危険）。

```cpp
// AudioEngine.Processing.BlockDouble.cpp:434 — 検証メモ
//   resetDryScaleGain=false のため、double パスでは dryScaleGain_ が crossfade 完了後に
//   リセットされない。AudioBlock パス（true）と動作が異なる。
//   ★ 注意: BlockDouble の false は「意図的な差異」である可能性が高い。単純な true 統一は推奨しない。
//   BUG-028 修正後の整合を Phase 6 soak で実測してから判断。
```

**対象ファイル**: `AudioEngine.Processing.BlockDouble.cpp`
**検証**: Phase 6（soak）で double パスと float パスの crossfade 挙動を実測し、差異が「意図的」か「バグ」かを確定してから対処。

**🔴 別視点調査（2026-08-10）— resetDryScaleGain の実コード照合**: `finalizeCrossfadeMixPath`（AudioEngine.h:3962）の第 3 引数 `resetDryScaleGain` は、crossfade 完了後（`isSmoothing()==false`）に `dryScaleGain` を 1.0 にリセットする（current/target/step/remaining を 1.0/1.0/0.0/0 に）。AudioBlock（:458）は true（リセット）、BlockDouble（:434）は false（リセットしない）。**差異の判断基準**: 次の crossfade 開始時（armCrossfade）に `dryScaleGain` が正しく初期化される前提なら false は「意図的」、初期化されず前回の残留値を使うなら「バグ」。Phase 6 soak での実測判断を維持（dash §4.4）。

### 4.5 bootstrap publishWorld 失敗の ignoreUnused（旧 A-3.5）— **四次レビュー承認（診断のみ）**

> **★ 実コード照合確定（2026-08-10）**: `PublishStageResult` は 3 値（Success / Rejected / Failed — RuntimePublicationCoordinator.h:14-19）を確認。`AudioEngine.Init.cpp:53-56` は `juce::ignoreUnused(result)` のまま（jassert 未実装）。下記 jassert 追加案を**確定**（実装は別途）。**★ 2026-08-11 実装済み: X4-B 後は `worldAuthority_.publish()` 一本化。bootstrap の validate 失敗分岐に `jassertfalse` を実装（AudioEngine.Init.cpp）。**

Init.cpp:55 `juce::ignoreUnused(result)`。`coordinator.publishWorld` は `PublishStageResult` を返す（AudioEngine.h:3654-3656 に「Coordinator の publishWorld() が PublishStageResult を返す」と明記）。四次レビューで `PublishStageResult::Success` を正常完了値として確認済み。失敗時の早期診断として Debug 用 `jassert` を追加:

**🔴 五次レビュー（2026-08-09）— `PublishStageResult` は3値（Success/Rejected/Failed）**:
`PublishStageResult` は `src/core/RuntimePublicationCoordinator.h:15-19` で **`Success, Rejected, Failed` の3値**。`publishWorld` は store 失敗時に `Failed`（:103）、reject 時に `Rejected`（:115）を返す。したがって**失敗検出は `result != Success` が正しい**（`Rejected` も失敗扱い）。

```cpp
// AudioEngine.Init.cpp:52-56 — 改修案（Debug のみ）
const auto result = coordinator.publishWorld(std::move(bootstrapWorld));
jassert(result == convo::PublishStageResult::Success);  // Debug のみ。Rejected/Failed を検出
juce::ignoreUnused(result);
```

**対象ファイル**: `AudioEngine.Init.cpp`
**検証**: 既存テスト回帰。

---

## 5. 残余リスクの対応設計（旧 Appendix A-4 を設計化）

| # | 残余リスク | 対応方法（詳細設計） |
|---|-----------|---------------------|
| R1 | `recoveryIntentQueue_` は SPSC（:434）。将来 Timer 等から直接呼ぶ場合は MPSC 化が必要 | **Phase 5 将来拡張**。対応手順: ①`LockFreeRingBuffer<RecoveryIntent, 256>` → `MpscBoundedRing<RecoveryIntent, 256>` へ置換 ②Producer 側 `submitRecoveryRequest` を §1.1 の reservation→push→rollback に変更（MPSC 対応） ③coalesce は `hasDuplicates` API 新設で「保留中全マージ」へ拡張。**四次レビュー: coalesce の実装は別タスク（§2.2 のとおり現方式は NO-GO）**。現状 Producer は CoordinatorLoop のみで SPSC 成立のため緊急性なし |
| R4 | retire 順序逆転は残るが quarantine fallback で UAF/リーク排除。requestReclaim 一本化は保留 | **runtime 経路対応済み**。`retireDSPHandleForRuntime` / `retireByHandle` は `requestReclaimHandle` 経由。`shutdownReclaim` は shutdown 専用（AudioEngine.h:2027 CacheMap::dtor, ReleaseResources.cpp:415,420）。**★ 2026-08-10 確定: R4 詳細設計（§6.3 末尾）として Phase 0-7 の段階的リファクタを確定**（ReclaimMode/ReclaimRequest 導入 → RuntimeEBR 一本化 → ShutdownQuiescent + ShutdownQuiescenceProof → releaseResources/CacheMap 移行 → deprecated → API 削除。**X3 実装時に R4-0〜R4-12 の順序で実施**。pendingReclaimHandles_ 再試行・isRetired()・RetireQuarantineStore との相互作用を先に固定） |
| R5 | bootstrap ignoreUnused による null world リスク | **要対応なし**（§4.5 で jassert 提案）。bootstrap は稀にしか失敗せず、次の操作（submitRebuildIntent）が検出。対応手順: §4.5 の jassert 追加のみ |
| R6 | BlockDouble finalizeCrossfadeMixPath(false) で dryScaleGain_ 未リセット | **Phase 6 確認**（§4.4）。対応手順: Phase 6 soak で double/float パスの crossfade 挙動を実測 → 差異が「意図的」なら文書化、「バグ」なら修正 |

**🔴🔴 九次レビュー（2026-08-09）— 残課題6件（P2 後検証対象・優先度高）**:

| # | 残課題 | 現状 | P2 後の検証優先度 |
|---|--------|------|-----------------|
| X1 | **Recovery Intent の full-drop そのもの** | recoveryIntentQueue_（SPSC, 256）が full で drop + Critical telemetry（AudioEngine.Retire.cpp:192-196, :223）。**「Recovery 保証」にはなっていない**。INV-5 は「drop を正しく記録できるか」の検証であって「絶対 drop しない」保証ではない | **最優先**（Recovery 保証に直結） |
| X2 | **Publish completion sequence monotonicity の実装保証** | 現状は Producer serialization で成立（§4.2.2）。コード上の不変条件として固定必要 | **高**（Shutdown/Receipt correctness に直結） |
| X3 | **shutdownReclaim の二系統** | requestReclaim（runtime, EBR）+ shutdownReclaim（shutdown, reader 停止後）の二系統（§3.2） | 中（別タスク） |
| X4 | **RuntimePublicationCoordinator の authority 二重化** | `convo::isr::`（Intent authority）と `convo::`（Publish authority, core）が同名共存（§1.2） | 中（Authority Singularization） |
| X5 | **Publish Intent residency の専用 counter 未導入** | `publicationIntentResidencyCount_` が未導入。publicationBacklogCount_ は hasDeferredCommit 由来（§1.1.5） | 中（P3） |
| X6 | **quarantine intent residency と quarantine resident の semantic 分離** | `quarantineResidentCount_` は実在 DSP 数。`quarantineIntentResidencyCount_` は未導入（§1.1.5） | 中（P3） |

- **X1 は「後で考えればよい改善」ではなく、Recovery 保証に直結するため最優先**。将来的には `Recovery 専用の durable admission state`（primary queue → retry/coalescing state → Critical failure only when recovery guarantee itself is impossible）が望ましい（九次 §17）。P2 の範囲外として別タスク化
- **X2 は Shutdown/Receipt correctness に直結**。§4.2.2 の Producer serialization テストを P2 後の検証対象に

**実装完了条件（九次 §21 — 4条件 + 十一次 §21 — C1/C2/C3 をコードレビュー項目として固定）**:
```
条件1: pendingIntentCount_ = queue residency + producer-side enqueue reservation
       （successful push count ではない。コードコメントで固定）
条件2: 1 Intent = 1 reservation（fallback で二重 +1 しない）
条件3: Publish pop は pending counter を触らない（if (type != Publish) decrement）
条件4: queue empty は AdmissionClosed + all producers joined + Coordinator stopped
       + Builder stopped の後だけ drain 判定に使う（shutdown phase guard）
```

**🔴🔴🔴 十一次レビュー §21 — 必須 acceptance criteria 3条件**:
```
C1: pendingIntentCount_ の意味をコード上で固定
    （Observe + Quarantine + Recovery の queue residency + producer-side enqueue reservations。
      Publish excluded / Retire excluded / Quarantine resident excluded。コメントを単に
      "pending intent count" としない。将来の誤修正防止に必須）
C2: queue emptiness は phase-gated にする
    （単なる queue.empty() ではなく、AdmissionClosed + all producers joined
      + Coordinator stopped + Builder stopped を assert できる形にする。
      ISR ではなく shutdown correctness の問題だが、最終的に lifetime safety に直結）
C3: Recovery full-drop を「成功扱い」にしない
    （INV-5 定義では drop + Critical telemetry まで許容するが、呼び出し側が
      submitRecoveryRequest() → 成功したと思う API 設計は避ける。将来 Recovery は
      Accepted / Queued / Dropped を明確に区別するか、少なくとも drop を
      Health/diagnostic layer まで確実に伝播できる構造にする。X1 の durable admission 導入までの中間対応）
```

---

## 6. 残課題 X1〜X6 の詳細設計（十一次レビュー反映）

> **★ 実コード照合確定（2026-08-10・第三者別視点監査 + 追加調査）**: X1〜X6 は全て「設計確定済み・実装未着手」**（★ 2026-08-11: 全 X1〜X6 実装済み — X_IMPL_CHECKLIST.md で検証）**（X5 は実装 GO / X1〜X4,X6 は条件付き GO）。実コード照合で主要点を確定:
>
> - **X1**: recoveryGeneration = rebuildRequestGeneration（AudioEngine.h:2423, RebuildDispatch.cpp:643 で増加 / :973-977 で消費時現在値）。**Step A/B/C（Recovery shutdown admission closure + ShutdownDiscard）を 2026-08-10 に最小限先行実装済み**（§1.2 監査補正 — X1 の shutdown semantics の前倒し）
> - **X2**: PublishReceiptWaiter（AudioEngine.h:3613-3635）の mutex+cv+monotonic watermark を確認。単一 completion writer（PublishExecutor sole gateway + intentQueue_ FIFO）→ CAS 不要。2 箇所 watermark（m_lastObservedSequence:246 / lastCompleted_:3634）同期
> - **X3**: shutdownReclaim 呼び出し元 3 箇所（AudioEngine.h:2027 CacheMap::dtor / ReleaseResources.cpp:415,420）+ requestReclaim 呼び出し元（AudioEngine.h:4248 / Retire.cpp:83）を確認。**INV-X3-5**: pendingReclaimHandles_（:4616）を source of truth
> - **X4**: read API（observePublishedWorld / acquireReadToken / consumeWorldHandle）は **RuntimeWorldAuthority.h に未実装**（X4-B-9 で新設予定）。現行 read path は `RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore)` 直接呼び（AudioEngine.h:1331/2119/3116/3383/3691 等）。**★ 2026-08-11 実装済み: X4-B-9 完了 — observePublishedWorld / acquireReadToken / consumeWorldHandle は RuntimeWorldAuthority.h に存在し、全 read path は worldAuthority_ 経由に移行。**
> - **X5**: `enqueuePublicationIntent`（ISRRuntimePublicationCoordinator.h:286 — dash 記載 :273 と行番号差）に reservation→push→rollback。全 3 経路（通常 rebuild / Recovery publish / deferred 再 enqueue）が**単一箇所に集約** → reservation は :286 のみで完結
> - **X6**: `quarantineResidentCount_` の実測上書き（AudioEngine.Threading.cpp:131）を確認。3 層分離（Intent / Ring / Resident）で aggregate 混入を解消
> - **§4.5**: `PublishStageResult` は 3 値（Success / Rejected / Failed — RuntimePublicationCoordinator.h:14-19）。bootstrap jassert 追加案を**確定**（実装は別途）。**★ 2026-08-11 実装済み（bootstrap validate 失敗分岐に `jassertfalse` を実装）。**
> - **§5 R4**: shutdownReclaim 全廃は **2026-08-10 に R4 詳細設計（§6.3 末尾・Phase 0-7）として確定**（X3 実装時に R4-0〜R4-12 で実施）。**R5**: §4.5 の jassert 追加で確定

> **★ 最上位 state transition table（2026-08-10・外部レビュー §24 反映 — Phase 0 で invariant として固定）**:
> 「Intent → Admission → Transport Residency → Execution → Committed → Completed → Resident → Retired → Reclaimable → Deleted」の状態分離の正式 table。counter の意味の混在を防ぐ（INV-ISR-03 / INV-X6-4 と整合）:
>
> | State | owner | residency counter | shutdown時 |
> |---|---|---|---|
> | Intent | producer | pendingIntent | discard/consume |
> | Queue | queue | pendingIntent | drain/discard |
> | DurablePending | Recovery admission | recoveryAdmissionPending | ShutdownDiscard |
> | Building | Builder | recoveryAdmissionPending | Builder join |
> | PublishTransport | publish queue | publicationIntentResidency | drain |
> | Committed | RuntimeWorld | なし | retain current semantics |
> | Completed | Receipt watermark | なし | wait/observe |
> | Resident | DSP runtime | quarantineResident | reclaim |
> | Retired | retire domain | retire counters | reclaim |
> | Reclaimable | Epoch | pending reclaim | delete |
> | Deleted | none | 0 | terminal |
>
> **★ Source of Truth / Diagnostic の区別（2026-08-10・外部レビュー §26 反映 — counter を増やしすぎない）**:
>
> - **Source of Truth（正判定の根拠）**: `pendingReclaimHandles_`（reclaim pending）/ Recovery admission state（PendingRecoveryAdmission）/ `RuntimeStore::current`（physical publication source）/ Quarantine resident state（DSPQuarantineManager::residentCount）
> - **Diagnostic / derived（補助・監視）**: `reclaimInFlightCount_`（近似・secondary guard）/ backlog metrics / watermarks（m_lastObservedSequence・lastCompleted_）
> - counter を増やし続けると counter 間の整合性が新たな bug source になる。Source of Truth を明確にし、Diagnostic は derived として扱う（INV-ISR-03 と整合）
>
> **★ Atomic wrapper 規約の AC（2026-08-10・外部レビュー §25/§27 反映）**:
>
> - 新 counter を追加する際、`.load()` / `.store()` / `.exchange()` を**直接書かない**。既存 atomic wrapper（`consumeAtomic` / `publishAtomic` / `fetchAddAtomic` / `fetchSubAtomic` / `compareExchangeAtomic`）を維持する（Practical Stable ISR の atomic abstraction を壊さない）
> - 適用対象: X5（publicationIntentResidencyCount_）/ X6（quarantineIntentResidencyCount_ / quarantineRingResidencyCount_）/ X3（registrationClosed_）/ R4（ReclaimPermit 関連）の新規 atomic
>
**🔴🔴🔴 十一次レビュー（2026-08-09）— 残課題6件は「個別バグの修正」ではなく、ISR の「唯一の Authority / Intent residency / Publish completion / shutdown-reclaim」の意味論を最終的に閉じるための設計**。実装順序: **X5/X6 → X2 → X4 → X1 → X3**。

**🔴🔴🔴 十五次レビュー（2026-08-09）— 実装開始前の最終 acceptance criteria**:
**INV-X1〜INV-X6 をコードレベルの不変条件として固定**し、特に X1/X2/X3 について「正常系」だけでなく **queue-full / out-of-order / shutdown race / reader re-entry の adversarial test** を通すことを必須にする:
```
INV-X1: Recovery が絶対に silent-loss しない（queue full ≠ lost・durable admission 保証）
INV-X2: completion の意味が数学的に閉じている（contiguous FIFO・wraparound 不可・Committed/Completion/Receipt 分離）
INV-X3: shutdown 後に reader が再侵入できない（readerRegistrationClosed）
INV-X4: RuntimeStore の write authority が実体として一本（RuntimeWorldAuthority のみ）
INV-X5: publicationIntentResidencyCount_ = Publish intent residency（deferred/commit と分離）
INV-X6: quarantine の queue / ring / resident が完全に別 semantic（aggregate 混入禁止）
```

**🔴🔴🔴 十七次レビュー（2026-08-09）— 「ISR安全」と「ISR意味論」の2層を区別**:

ISR には2つの層がある。X1〜X6 は主に **B を閉じる改修**:
```
A. realtime safety:
   - lock-free / no allocation / no delete / no blocking / bounded operation
B. realtime semantic correctness:
   - counter の意味が明確
   - ownership が明確
   - lifetime が明確
   - state transition が一意
   - RT が policy decision をしない
```
- 例: X5 の `fetchAddAtomic(publicationIntentResidencyCount_, 1)` は **RT-safe（A）** かもしれないが、**counter の意味（B）** が曖昧なら ISR architecture としては未完成
- **X1〜X6 は B（semantic correctness）を閉じる改修**。A（realtime safety）は既存の lock-free / bounded 設計を維持
- この2層を区別しないと、RT-safe な atomic 追加だけで「ISR 完了」と誤認する危険がある

**🔴🔴🔴 二十五次レビュー（2026-08-09）— INV-ISR-01〜07（最上位 ISR 不変条件・§23）**:

X1〜X6 の INV に加えて、**ISR 全体の最上位不変条件**として以下を追加する:

```
INV-ISR-01: isFullyDrained == true は以下を意味する:
    all producers stopped AND all producer joins completed
    AND all transport queues empty AND all deferred state empty
    AND all reclaim-in-flight == 0 AND all reader inactive
    AND reader registration closed

INV-ISR-02: pendingIntentCount_ は queue size ではなく
    transport residency + producer reservation である

INV-ISR-03: 異なる semantic state を一つの counter で表現しない。
    特に Intent / DSP resident / Retire resident を混ぜない

INV-ISR-04: ShutdownQuiescent reclaim は readerRegistrationClosed なしでは
    絶対に許可しない

INV-ISR-05: completion watermark を publication committed と同一視しない

INV-ISR-06: currentWorld_ を ownership source として扱わない

INV-ISR-07: currentWorld_ と RuntimeStore::current が存在する間は、
    両者の identity consistency を検証可能にする（Test 9 で検証）
```

- **INV-ISR-01**: isFullyDrained の完全条件（§6.7 + 二十四次 §27-A の measurement predicate と整合）
- **INV-ISR-02**: pendingIntentCount_ の定義（§1.1 + 二十四次 §27-B の命名と整合）
- **INV-ISR-03**: semantic 混同禁止（§6.6 X6 の4層分離と整合）
- **INV-ISR-04**: ShutdownQuiescent reclaim の必須前提（§6.3 X3 と整合）
- **INV-ISR-05**: committed ≠ completed（§6.2 X2 と整合）
- **INV-ISR-06**: currentWorld_ は non-owning observation alias（§6.4-X4 INV-X4-8 と整合）
- **INV-ISR-07**: dual pointer の identity consistency（§6.4-X4 Test 9 / INV-X4-6 と整合）

### 6.1 X1 — Recovery Durable Admission（P0 優先）

> **★ 実コード照合確定（2026-08-10）**: recoveryGeneration = rebuildRequestGeneration（AudioEngine.h:2423, RebuildDispatch.cpp:643 で増加 / :973-977 で消費時現在値 + sealed=true）。**Step A/B/C（Recovery shutdown admission closure + ShutdownDiscard）を 2026-08-10 に最小限先行実装済み**（§1.2 監査補正）: submitRecoveryRequest の shutdown gate（Step B）+ stopRebuildThread 後の discardRecoveryRequestsOnShutdown（Step C）。残余の durable admission（PendingRecoveryAdmission + lease 方式）は本項目の実装対象。

**現状**: Recovery は `RecoveryIntent`（POD/trivially-copyable transport payload）を `recoveryIntentQueue_`（256, SPSC）に投入し、Builder が消費。queue full で drop + `recoveryIntentDropCount_++` + Critical telemetry。**「loss を silent にしない」にはなるが、Recovery を保証しない**。

**🔴 設計方針**: queue を 256 → 4096 にするのは **NO-GO**。必要なのは **Recovery Intent を transient queue ではなく、durable な admission state として一度受理**すること:

```text
Recovery event → Recovery Admission（state = Pending, recoveryGeneration, buildSnapshot）
    ↓
recoveryIntentQueue_
    ├─ success → Builder
    └─ full    → retry/coalesce（durable Pending state に残す）→ 次 Builder wake → retry
```

**🔴🔴🔴 十四次レビュー（2026-08-09）— single slot をやめ、Recovery generation 単位の durable admission に再設計（必須）**:

dash の旧案（`std::atomic<RecoveryIntent> pendingRecoveryAdmission_` single slot）は危険:
```
Recovery A, B, C が queue full の間に来た場合:
  pending = A → pending = B → pending = C   ← A/B が消える
```
- **異なる target の扱いが未定義**。「同一 target は coalesce」だけでは、異なる handle の Recovery が消える
- **Recovery は「handle 単位 coalesce」より「Recovery generation 単位 coalesce」が自然**: ConvoPeq の Recovery semantic は「quarantined DSP を除外した現在の authoritative configuration から build」であり、A,B,C を個別に3回 build する必要はない

**推奨設計（Recovery rebuild request の durable admission）**:
```cpp
struct PendingRecoveryAdmission {
    bool pending;                        // durable state 有効
    uint64_t recoveryGeneration;         // Recovery generation（build 世代）
    RuntimeBuildSnapshot buildSource;    // latest（coalesce で更新）
    // ★ DSPHandle を必ず保持する必要はない — Builder が quarantine manager の
    //   authoritative state を参照できる場合は、recoveryGeneration のみで十分
};
```
- **coalesce semantics**: 同一 recoveryGeneration（または quarantine 状態の変化）の複数 Recovery は、**latest buildSource で上書き**（Pending=1 を維持）。異なる generation は最新を採用
- **reservation は push 前**（十四次指摘 — P2 の reservation 原則と競合させない）:
```cpp
convo::fetchAddAtomic(pendingIntentCount_, 1);   // ① reservation（push 前）
if (recoveryIntentQueue_.push(intent))
    return;                                      // ② success → 維持
storePendingRecovery(intent);                    // ③' durable state に保持（reservation 維持）
// push 失敗でも rollback しない — durable state に存在するため residency は継続
```
- **Builder が durable state から消費した時点で fetchSub**（queue pop と同様に pendingIntentCount_--）

**Coalescing が必要**: Recovery は「現在の authoritative configuration から quarantined DSP を除外して再構築」（`getCurrentBuildSnapshotForRecovery()` — AudioEngine.h:4265）の semantic。複数 Recovery を3件処理する必要はない:
```
RecoveryAdmissionState { Pending, Building, Completed, Failed }
same recovery generation → coalesce → Pending = 1（latest buildSource で上書き）
```

**🔴🔴🔴 十八次別視点8調査（2026-08-09）— currentBuildSnapshot_ の供給と初期状態を確定**:
- **currentBuildSnapshot_（AudioEngine.h:4619-4623, mutex 保護）** は `enqueuePublicationIntentForRuntimeCommit` が sealedSnapshot を受け取る時点で更新（Commit.cpp:704-705）
- **`getCurrentBuildSnapshotForRecovery()`（:4265）** は currentBuildSnapshot_ を返す。QuarantineIntentHandler が submitRecoveryIntent(handle, buildSource) を呼ぶ（ProcessIntent.cpp:101-102）
- **初期状態（sealed=false）の挙動**（REPAIR_PLAN2.md:173 で確定）: `currentBuildSnapshot_` の初期値は `RuntimeBuildSnapshot{}`（sealed=false）。初回 publish 前に quarantine が発生した場合、`getCurrentBuildSnapshotForRecovery()` は sealed=false を返し、**Builder の `recovery->buildSource.sealed` チェック（RebuildDispatch.cpp:917）で Recovery が skip** される（正しい動作 — build できる構成がない）
- **X1 の durable admission との整合**: sealed=false の buildSource を持つ Recovery は、**durable admission に入る前に検証**すべき（`takePendingRecoveryAdmission` または `submitRecoveryRequest` で sealed チェック）。無効な durable admission を保持しない（reservation の無駄）
- **軽微な無駄**: 初回 publish 前の quarantine で submitRecoveryIntent が recoveryPending=true + notify を実行し無意味な起床が発生するが、機能上の問題はない（Phase 5 の最適化候補）

**🔴🔴🔴 追加調査（2026-08-09）— recoveryGeneration の実体を確定（rebuildRequestGeneration）**:
`rebuildRequestGeneration`（AudioEngine.h:2423）は:
- `requestRebuild`（RebuildDispatch.cpp:643）で `++rebuildRequestGeneration`（rebuild 要求ごとに増加）
- `isRebuildObsolete(generation)`（AudioEngine.h:2464）: `generation != currentBuildGeneration()` で obsolete 判定
- Builder の Recovery 消費（RebuildDispatch.cpp:965-967）: `recoveryGeneration = consumeAtomic(rebuildRequestGeneration, acquire)` を取得し、`recoverySnapshot.generation = recoveryGeneration`（:969）で publish

**X1 の recoveryGeneration は、この `rebuildRequestGeneration` を durable state に保持**する:
- **同一 generation の複数 Recovery**: 最新 buildSource で上書き（coalesce, Pending=1）
- **generation が進んだ場合**（新たな rebuild 要求が発生）: 古い Recovery は stale（`isRebuildObsolete` と整合）。新しい generation の Recovery として再受理
- **🔴🔴🔴 十八次別視点5調査（2026-08-09）— buildSource.generation との関係を確定**: `RuntimeBuildSnapshot`（RuntimeBuildTypes.h:48-66）は `generation` フィールド（:50）を持つ。**durable admission の recoveryGeneration と buildSource.generation は別物**:
  ```
  PendingRecoveryAdmission.recoveryGeneration = 入ってきた時点の rebuildRequestGeneration（coalesce 判定用）
  PendingRecoveryAdmission.buildSource.generation = buildSource が運ぶ generation（build 入力の世代）

  消費時（RebuildDispatch.cpp:968-969）:
    recoverySnapshot = recovery->buildSource   // 値コピー
    recoverySnapshot.generation = currentRebuildRequestGeneration  // ★ 消費時に現在値で上書き
    recoverySnapshot.sealed = true
  ```
  - **coalesce の generation 判定は `PendingRecoveryAdmission.recoveryGeneration` を使う**（buildSource.generation ではない）
  - **buildSource.generation は消費時に現在の rebuildRequestGeneration で上書きされる**ため、durable state の buildSource.generation は「入ってきた時点の値」であり、消費時には意味を持たない（上書きされるため）
  - **isRebuildObsolete 判定（coalesce の obsolete チェック）は recoveryGeneration（durable 側）に適用**する。buildSource.generation は obsolete 判定に使わない
- **🔴🔴🔴 INV-X1-4 の DSPHandle 保持（実コード検証で確定）**: Builder は build 入力を buildSource から引当するため、**build 実行自体には handle 不要**。**ただし消費時検証（RebuildDispatch.cpp:917 `qHandle.isNull()`）には handle が必要**。durable state には `handle` を保持する（RecoveryIntent :166-179 と一致）。**INV-X1-4 は「World ownership を持たない」の意味**（`RuntimePublishWorld*` 等の World ポインタを持たない。DSPHandle は非所有）

**不変条件**:
```
INV-X1-1: Recovery accepted ⇒ exactly one durable Pending/Building state exists
INV-X1-2: queue full ≠ Recovery lost
INV-X1-3: Recovery drop telemetry は削除せず、queue saturation の診断として残す
INV-X1-4: Recovery durable state は World ownership を持たない
          （DSPHandle / epoch / intentId / RuntimeBuildSnapshot のみ。ISR 的に重要）
INV-X1-5: One logical Recovery admission owns at most one pendingIntent reservation.
          Coalescing does not create another reservation.
          （1 logical Recovery admission = at most 1 reservation。coalesce で reservation を増やさない）
INV-X1-6: A durable Recovery admission is not simultaneously counted as both
          queue residency and a second independent admission.
          （durable admission は queue residency と二重計上しない）
```

**🔴🔴🔴 十六次レビュー（2026-08-09）— X1 の reservation / coalesce state machine を確定（最重要）**:

`pendingIntentCount_` と durable admission の関係は、**「counter 増加」ではなく「reservation ownership の移動」**として扱う:
```
                ┌──────────────┐
                │ No Admission │
                └──────┬───────┘
                       │ reserve（pendingIntentCount_++）
                       ▼
                ┌──────────────┐
                │   Reserved   │
                └──────┬───────┘
                       │
              ┌────────┴────────┐
              │                 │
           push OK          push full
              │                 │
              ▼                 ▼
       Queue Residency    Durable Pending（pendingIntentCount_ rollback → recoveryAdmissionPending_ へ）
              │                 │
           pop             retry push
              │                 │
              └────────┬────────┘
                       ▼
                 Reservation released
```
- **1 Recovery admission = 1 reservation**（INV-X1-5）
- **coalesced admission が既に存在するなら reservation は増やさない**（`Recovery A, A, A` で pendingIntentCount_ が 1→2→3 にならない）
- **X1 の修正案**: `PendingRecoveryAdmission` に `reservationOwned` を追加
```
PendingRecoveryAdmission
 ├─ pending
 ├─ recoveryGeneration
 ├─ buildSource
 └─ reservationOwned
coalesce:
    same generation → buildSource update only（reservationOwned は変更なし）
    newer generation → replace admission（reservation 移動）
    obsolete generation → discard before queue insertion（isRebuildObsolete 再利用）
```

**🔴🔴🔴 十六次レビュー（2026-08-09）— RecoveryAdmissionKey（rebuildGeneration を基本キー）**:
coalesce の基本キーを明示する:
```cpp
struct RecoveryAdmissionKey {
    uint64_t rebuildGeneration;   // requestRebuild で increment（RebuildDispatch.cpp:643）
};
```
- **`recoveryGeneration` は `rebuildRequestGeneration`（AudioEngine.h:2423）に対応**し、`isRebuildObsolete(generation)`（:2464）が obsolete 判定
- **異なる generation の「最新を採用」は obsolete policy と明示的に結び付ける**: `G10 pending → G11 arrives` のとき、G10 を捨てて G11 に更新してよいかは `isRebuildObsolete(G10)` の結果に従う
- **🔴🔴🔴 十七次レビュー（2026-08-09）— `pending = newest` は禁止**: 「G10 が obsolete だから捨ててよい」ことを **`isRebuildObsolete(G10)` で判定**する。単純な `pending = newest` 置換は禁止（G10 がまだ obsolete でない場合、G10 を維持すべき）
```
coalesce（G10 pending, G11 arrives）:
  if (isRebuildObsolete(G10)) → G10 discard → G11 採用（reservation 移動）
  else                        → G10 維持（G11 は stale として破棄 or 新規 admission 扱い）
```
- **`pendingIntentCount_` は admission event 数ではなく transport reservation 数**と固定（INV-X1-5/INV-X1-6）

**ISR 制約**: **Audio Thread に durable store を作らない**。Audio Thread は `quarantine detected → Recovery Intent → bounded enqueue` まで。queue full でも Audio Thread で mutex / allocation / retry loop / blocking しない。**durable state は NonRT/Coordinator 側**。

**🔴🔴 十二次レビュー（2026-08-09）— 具体的な実装位置を確定**:
`submitRecoveryRequest`（ISRRuntimePublicationCoordinator.cpp:647-673）は現在 push 失敗時に drop する。X1 では **push 失敗時に RecoveryIntent を破棄せず、Coordinator 側の durable admission state（Pending）に保持**する:
```cpp
void RuntimePublicationCoordinator::submitRecoveryRequest(...) noexcept
{
    // ... intent 生成 ...
    convo::fetchAddAtomic(pendingIntentCount_, 1);        // ① reservation（push 前 — 十四次指摘）
    if (recoveryIntentQueue_.push(intent))
        return;                                            // ② success → 維持
    // X1: queue full ≠ Recovery lost（INV-X1-2）
    //   durable Pending state に保持（Coordinator 側・NonRT）
    //   → 次 Builder wake 時に recoveryIntentQueue_ へ再 push（retry）
    //   → 同一 recovery generation は coalesce（Pending = 1, latest buildSource で上書き）
    //   → drop は「永続的 failure」時のみ（Recovery guarantee 自体が不可能な場合）
    //   → recoveryIntentDropCount_++ は queue saturation の診断として維持（INV-X1-3）
    storePendingRecovery(intent);   // durable state（PendingRecoveryAdmission）
}
```

**🔴🔴🔴 十五次レビュー（2026-08-09）— pendingIntentCount_ と durable admission の semantic boundary を明確化（必須）**:

dash 旧案は「durable state に移った場合も `pendingIntentCount_` の reservation を rollback しない」としていた。しかし `pendingIntentCount_` の定義は **queue residency + producer-side reservation** であり、durable state に移った瞬間は **transport queue residency ではなく durable admission residency** になる。**semantic ambiguity が残る**。

**推奨（十五次 §14）: `pendingIntentCount_` を transport residency に限定し、`recoveryAdmissionPending_` を独立させる**:
```
pendingIntentCount_       = transport queue residency（queue に実際に存在する intent）
recoveryAdmissionPending_ = durable Recovery state（PendingRecoveryAdmission が有効）
```
- **Producer**: `pendingIntentCount_` の reservation（①）は **push 成功時のみ維持**。push 失敗（durable 化）時は **`pendingIntentCount_` から rollback（fetchSub）し、`recoveryAdmissionPending_ = true` に切替**
```cpp
convo::fetchAddAtomic(pendingIntentCount_, 1);   // ① reservation（push 前）
if (recoveryIntentQueue_.push(intent))
    return;                                      // ② success → pendingIntentCount_ 維持
convo::fetchSubAtomic(pendingIntentCount_, 1);   // ③ transport residency から rollback
recoveryAdmissionPending_.store(true, release);  // ④ durable state に切替（recoveryAdmissionPending_ で追跡）
storePendingRecovery(intent);
```
- **Consumer（Builder）**: durable state から消費した時点で `recoveryAdmissionPending_.store(false)`。transport queue 経由なら `pendingIntentCount_--`
- **isFullyDrained**: 両方を見る（`pendingIntentCount_ == 0 AND !recoveryAdmissionPending_`）— §6.7 の `recoveryAdmissionPending == false` と整合
- **durable state の実装**: `PendingRecoveryAdmission`（pending / recoveryGeneration / buildSource）。SPSC（Producer=CoordinatorLoop）なので競合なし。Builder wake は既存 `rebuildMutex` / `rebuildCV` 経路（AudioEngine.h:4274-4282 の submitRecoveryIntent が起床）

**🔴🔴🔴 追加調査（2026-08-09）— 宣言位置を確定（ISRRuntimePublicationCoordinator.h:434 の recoveryIntentQueue_ の近く）**:
`recoveryAdmissionPending_` と `PendingRecoveryAdmission` は、`recoveryIntentQueue_`（ISRRuntimePublicationCoordinator.h:434）の直後に宣言する:
```cpp
// ISRRuntimePublicationCoordinator.h — Recovery Intent Queue（:431-437）
static constexpr size_t kRecoveryIntentQueueCapacity = 256;
LockFreeRingBuffer<RecoveryIntent, kRecoveryIntentQueueCapacity> recoveryIntentQueue_;  // :434
std::atomic<uint64_t> nextRecoveryIntentId_{0};                                        // :435
std::atomic<uint64_t> recoveryIntentDropCount_{0};                                     // :437

// ★ X1 新設（recoveryIntentQueue_ の隣）
struct PendingRecoveryAdmission {
    // 🔴🔴🔴 二十六次レビュー（2026-08-09）— state enum を追加（必須修正1）
    enum class State : uint8_t {
        NoAdmission = 0,   // durable なし
        DurablePending,    // durable 保持（queue に無い Recovery）
        Building           // Builder が build 中（lease 中）
    };
    State state = State::NoAdmission;
    bool pending = false;                 // durable state 有効（state != NoAdmission）
    uint64_t recoveryGeneration = 0;      // rebuildRequestGeneration（AudioEngine.h:2423）
    RuntimeBuildSnapshot buildSource{};   // latest（coalesce で更新）
    bool reservationOwned = false;        // 1 admission = 1 reservation（INV-X1-5）
    // ★ 🔴🔴🔴 実コード整合で追加（RecoveryIntent 構造 :166-179 と一致させる）
    DSPHandle handle{};                   // recovery 対象（quarantined DSPHandle）— 消費時 :917 の isNull 検証に使用
    PublicationEpoch epoch{0};            // emit 時 publicationEpoch（FIFO/epoch 検証用）
    uint64_t intentId{0};                 // 診断・モニタリング用シーケンス番号
};
PendingRecoveryAdmission pendingRecoveryAdmission_;   // SPSC（Producer=CoordinatorLoop 専有）
std::atomic<bool> recoveryAdmissionPending_{false};    // durable 有効フラグ（isFullyDrained 用）
```
- `pendingRecoveryAdmission_` は **plain 構造体**（SPSC のため atomic 不要）
- `recoveryAdmissionPending_` は **atomic<bool>**（isFullyDrained が NonRT から読むため）。**state == DurablePending or Building の間 true**
- **🔴🔴🔴 十八次別視点14調査（2026-08-10）— epoch の取得元（currentWorld_ 由来）を確定**: `submitRecoveryRequest`（ISRRuntimePublicationCoordinator.cpp:650-652）は **`consumeAtomic(currentWorld_)` → `world->publication.epoch`** で epoch を取得する:
  ```cpp
  const auto world = static_cast<const RuntimeState*>(
      convo::consumeAtomic(currentWorld_, std::memory_order_acquire));   // :650-651
  const auto currentEpoch = world ? world->publication.epoch : PublicationEpoch{0};  // :652
  RecoveryIntent intent{ quarantinedHandle, currentEpoch, ... };        // :654-659
  ```
  - **X1 の `PendingRecoveryAdmission.epoch` は submitRecoveryRequest 時に currentWorld_ から取得**される（:652）
  - **X4 の INV-X4-A（currentWorld_ is observation-only）との整合**: submitRecoveryRequest は **NonRT（CoordinatorLoop）** から呼ばれるため、currentWorld_ を metadata observation（epoch 取得）として使用するのは**正当**（RT から currentWorld_ を RuntimeWorld 取得元にしない INV-X4-C に違反しない）
  - **coalesce 時の epoch 更新**: 新規 Recovery が durable 化する際、latest buildSource と同様に epoch も currentWorld_ から再取得する（最新 epoch を保持）
- **🔴🔴🔴 二十六次レビュー（2026-08-09）— take を「lease（state transition）」に変更（必須修正1）**: `takePendingRecoveryAdmission()` を **destructive dequeue（state 消去）ではなく、DurablePending → Building の state transition** にする。build 失敗時に **Building → DurablePending へ戻す**ことで、一時的 failure の retry を durable state で構造的に保証する。
- **🔴🔴🔴 十八次別視点10調査（2026-08-09）— intentId 採番と IR 転送の整合を確定**:
  - **`nextRecoveryIntentId_`（ISRRuntimePublicationCoordinator.h:435, atomic）は `submitRecoveryRequest`（:657）で fetch_add(1, relaxed)**。**durable 化後も採番は継続**（queue に入った Recovery も durable に入った Recovery も nextRecoveryIntentId_ から採番）
  - **`PendingRecoveryAdmission.intentId` は採番済みの値**を保持（durable 化時に submitRecoveryRequest 内で採番）
  - **IR 転送（RuntimeBuilder.cpp:447 `transferIRStateFrom(engine.getConvolverProcessor())`）**: Recovery build は現在の UI processor から IR を転送（Recovery semantic = 現在のユーザー構成を再 build + quarantined 除外）。**buildSource（RuntimeBuildSnapshot）は IR データを内包しない**（値コピーは metadata/fingerprint のみ — ISRRuntimePublicationCoordinator.h:174-177）ため、durable admission の buildSource は軽量
  - **coalesce 後の buildSource**: 最新 buildSource で上書きされても IR 実体は build 時に転送されるため、**stale IR の懸念なし**
- **🔴🔴🔴 十八次別視点3調査（2026-08-09）— Producer 単一スレッド前提の完全確認**: `submitRecoveryRequest` の呼び出し元は **1箇所のみ**（AudioEngine.h:4277 の `submitRecoveryIntent` 内）:
  ```
  submitRecoveryRequest の Producer 経路（実コード検証）:
    QuarantineIntentHandler::handle（ProcessIntent.cpp:102）→ engine.submitRecoveryIntent（AudioEngine.h:4274）
      → runtimePublicationBridge_.submitRecoveryRequest（:4277）  ← 唯一の Producer
  RecoveryIntentHandler::handle（ProcessIntent.cpp:126-132）は DEAD CODE
    （現状誰も intentQueue_ に Recovery Intent を push しない — :123-125 のコメント確認）
  ```
  - **Producer = CoordinatorLoop 単一スレッド**（QuarantineIntentHandler 経由）。SPSC 前提が完全に成立
  - **RecoveryIntentHandler 経路（将来拡張）**が追加された場合も、`submitRecoveryRequest` は CoordinatorLoop 経由で呼ばれる（intentQueue_ → processIntent → RecoveryIntentHandler → submitRecoveryIntent）。**Producer は常に CoordinatorLoop スレッド**（非同期・別スレッドからの直接 push はない）
- **🔴🔴🔴 handle 保持は必要（実コード検証で確定）**: RebuildDispatch.cpp:913-917 の消費ループは `qHandle.isNull()`（:917）で無効ハンドルを検証するため、durable state にも `handle` を保持する。**INV-X1-4 は「durable state は World ownership を持たない」の意味**（DSPHandle は非所有ハンドルであり、`RuntimePublishWorld*` 等の World ポインタを持たない）。DSPHandle は保持してよい
- **🔴🔴🔴 十八次調査（2026-08-09）— `AudioEngine::isFullyDrained()` の外部 setter との干渉**: Threading.cpp:117 の `setPendingIntentCount(hasDeferredCommit ? 1u : 0u)` は、X1 の durable 化（pendingIntentCount_ rollback → recoveryAdmissionPending_ = true）後の pendingIntentCount_ = 0 を**さらに上書き**しないが、`recoveryAdmissionPending_` を認識しないため **isFullyDrained が durable Recovery の存在を見落とす**可能性がある。X1 実装時は **:117 の外部上書きを廃止**し、isFullyDrained は `pendingIntentCount_ == 0 AND !recoveryAdmissionPending_` を Coordinator 内部で評価する（§6.7）

**🔴🔴🔴 追加調査（2026-08-09）— recoveryAdmissionPending_ と submitRecoveryIntent の起床統合**:
`submitRecoveryIntent`（AudioEngine.h:4274-4287）は:
```cpp
inline void submitRecoveryIntent(quarantinedHandle, buildSource) {
    runtimePublicationBridge_.submitRecoveryRequest(quarantinedHandle, buildSource);  // :4278
    { std::lock_guard<std::mutex> lock(rebuildMutex); recoveryPending = true; }       // :4283-4285
    rebuildCV.notify_all();                                                          // :4286
}
```
**X1 の recoveryAdmissionPending_ は、`submitRecoveryRequest` の push 失敗時（durable 化）に set する**。rebuild スレッドの起床は **`recoveryPending = true`（既存 :4283）が既に担保**する（durable state への移送後も rebuild スレッドが wake して takePendingRecoveryAdmission を消費する）:
```
submitRecoveryIntent
  → submitRecoveryRequest
      ├─ push 成功 → pendingIntentCount_ 維持（transport queue）
      └─ push 失敗 → pendingIntentCount_ rollback + recoveryAdmissionPending_ = true（durable）
  → recoveryPending = true（既存）→ rebuildCV.notify_all()（既存）
  → Builder: popRecoveryRequest ループ → takePendingRecoveryAdmission（durable 消費）→ recoveryAdmissionPending_ = false
```

**🔴🔴🔴 十一次/十二次レビュー追記 — Builder 消費ループ（RebuildDispatch.cpp:898-947）との整合を確定**:

X1 は、**Producer 側（submitRecoveryRequest）だけでなく、Builder 側の消費ループも変更**する。既存構造:
```cpp
// AudioEngine.RebuildDispatch.cpp:901-918（現行）
if (!convo::consumeAtomic(rebuildThreadShouldExit, acquire)) {
    { std::lock_guard<std::mutex> lock(rebuildMutex); recoveryPending = false; }  // :905
    while (auto recovery = runtimePublicationBridge_.popRecoveryRequest()) {        // :911
        // handle 実在性・buildSource.sealed 検証 → build → publish
    }
}
```

**X1 での変更**:
1. **Producer 側**: `submitRecoveryRequest` は **reservation（fetchAdd）を push 前に行い**、push 成功時は `pendingIntentCount_` 維持、push 失敗時は **`pendingIntentCount_` から rollback（fetchSub）して durable `PendingRecoveryAdmission` へ保持（`recoveryAdmissionPending_ = true`）**（十五次 §14 で確定）。`recoveryPending` フラグ（AudioEngine.h:2581,4283）は **set を維持**（Builder 起床）
2. **Builder 側**: while ループ（:911）の**後**に、durable Pending から残余 Recovery を取り出して処理。**実コードの既存消費パス（RebuildDispatch.cpp:911-972）と同一処理を再利用する**:
```cpp
// AudioEngine.RebuildDispatch.cpp — X1 追記（while ループ :911 の後、:973 の isObsolete ラムダの前）
if (!convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire))
{
    // durable admission の残余を処理（queue には無いが durable state に存在する Recovery）
    while (auto recovery = runtimePublicationBridge_.takePendingRecoveryAdmission())
    {
        const auto& qHandle = recovery->handle;
        if (qHandle.isNull() || !recovery->buildSource.sealed)
            continue;                                   // :917 と同じ無効ハンドル検証
        auto convolverSnapshot = uiConvolverProcessor.captureBuildSnapshot();   // :923
        convo::BuildResult recoveryResult = runtimeBuilder.build(
            recovery->buildSource.buildInput, convolverSnapshot);               // :925-926
        if (recoveryResult.runtime == nullptr)
        {
            diagLog("[DIAG] rebuildThreadLoop: recovery(durable) build failed error="
                + juce::String(convo::toString(recoveryResult.error)));
            continue;   // durable state は Pending のまま（次サイクル retry。reservation 維持）
        }
        dspGuard.ptr = recoveryResult.runtime;
        auto* recoveryDSP = recoveryResult.runtime;
        if (recoveryDSP->convolverRt().getIRLength() > 0)
            recoveryDSP->convolverRt().rebuildAllIRsSynchronous(isRecoveryAborted);  // :937
        const auto recoveryWarmup = runtimeBuilder.validateWarmup(*recoveryDSP);
        if (recoveryWarmup != convo::BuildError::None)
        {
            if (dspGuard.ptr != nullptr) { AudioEngine::destroyDSPCoreNode(dspGuard.ptr); dspGuard.ptr = nullptr; }
            continue;   // :949-954 と同じ（失敗 DSP 破棄 → durable Pending 維持）
        }
        recoveryDSP->convolverRt().refreshLatency();
        recoveryDSP->ramps().fadeInSamplesLeft = DSPCore::FADE_IN_SAMPLES;      // :957-958
        DSPCore* dspToCommit = dspGuard.ptr;
        dspGuard.ptr = nullptr;
        const int recoveryGeneration =
            convo::consumeAtomic(rebuildRequestGeneration, std::memory_order_acquire);  // :966-967
        auto recoverySnapshot = *recovery->buildSourcePtr;   // 最新 buildSource を採用
        recoverySnapshot.generation = recoveryGeneration;
        recoverySnapshot.sealed = true;
        enqueuePublicationIntentForRuntimeCommit(dspToCommit, recoveryGeneration, recoverySnapshot);  // :971
        // ★ X1: durable state から消費成功 → recoveryAdmissionPending_ = false（take 時に解除）
    }
```
- **🔴🔴🔴 十八次別視点3調査（2026-08-09）— `takePendingRecoveryAdmission` 消費と `recoveryPending` フラグの lost-wakeup 整合を確定**:
  実コード検証（RebuildDispatch.cpp:898-911）: Builder は `recoveryPending` クリア（:905）後に `popRecoveryRequest()` ループ（:911）を実行する。**X1 の `takePendingRecoveryAdmission()` はこの while ループの後に追加される**:
  ```
  RebuildDispatch.cpp（現行 :901-911）:
    recoveryPending = false;                        // :905（クリア）
    while (auto recovery = popRecoveryRequest())    // :911（transport queue 消費）
        ...
    // ★ X1 追記（while ループ後）:
    while (auto admission = takePendingRecoveryAdmission())  // durable 消費
        ...
  ```
  - **lost-wakeup の安全性**: 消費中に新規 Recovery 要求が来た場合、`submitRecoveryIntent`（AudioEngine.h:4283）が `recoveryPending = true` を再 set + `rebuildCV.notify_all()` する（既存 :899-900 の設計）。**takePendingRecoveryAdmission の消費中も同じ機構が機能**する（recoveryPending は transport queue の有無と独立に set される）
  - **ただし注意**: `recoveryPending` は `:905` で一度クリアされるため、**durable admission の消費開始後**に新規 durable 化が起きた場合、`recoveryPending` は再 set され、次サイクルで takePendingRecoveryAdmission が再び消費する。**takePendingRecoveryAdmission 消費ループが空になるまで recoveryPending の再確認は不要**（`recoveryAdmissionPending_` が durable 有無の真実）
  - **`takePendingRecoveryAdmission()` は `recoveryAdmissionPending_` が false なら即座に nullopt** を返すため、ループは durable が空で自然終了する（:1284 の実装）
- **🔴🔴🔴 十八次別視点調査（2026-08-09）— Recovery publish と X5 counter の相互作用を確定**: Recovery publish（:971 → enqueuePublicationIntentForRuntimeCommit → submitPublishRequest → enqueuePublicationIntent）は、通常 rebuild と**同一の Publish enqueue 経路**（経路2）。したがって:
  - **Recovery publish も X5 の `publicationIntentResidencyCount_` を +1 する**（正常な Publish として扱われる）
  - **X1 の durable admission（recoveryAdmissionPending_）と X5 の counter は独立**（X1 は「Recovery 要求」の durable 化、X5 は「Publish Intent」の transport residency）
  - **Recovery build が publish に到達する前の「build 中」状態はどちらの counter にも含まれない**: recoveryAdmissionPending_ は take でクリア済み（:1278）、publicationIntentResidencyCount_ は enqueue 前。この「build gap」は isFullyDrained の観点で注意（shutdown 中に Recovery build が途中で止まる可能性 → 既存の Builder join で回収）

**🔴🔴🔴 十八次別視点6調査（2026-08-09）— shutdown 中の durable Recovery admission の処理を確定**:
実コード検証（stopRebuildThread, RebuildDispatch.cpp:771-784 + Builder 消費ループ :901-973）:
```
stopRebuildThread（:771-784）:
  rebuildThreadShouldExit = true（:776）→ rebuildCV.notify_all()（:779）→ join（:782）
Builder 消費ループ（:901-973）:
  if (!rebuildThreadShouldExit) {         // :901 ← shutdown 中は false → ループ全体をスキップ
      recoveryPending = false;            // :905
      while (popRecoveryRequest()) { ... }
      // ★ X1: while ループ後に takePendingRecoveryAdmission を消費
  }
```
- **shutdown 中の durable admission の扱い（確定）**:
  ```
  case A: shutdown 前に durable admission が消費済み → recoveryAdmissionPending_ == false（正常）
  case B: shutdown 中に durable admission が残っている（Builder が :901 でスキップ）
          → recoveryAdmissionPending_ は true のまま
          → isFullyDrained の `recoveryAdmissionPending == false` が false → shutdown 完了不可
  ```
- **X1 での対応（確定）**: **shutdown 時は durable admission を破棄（クリア）する**。Recovery は「現在の構成の再 build」であり、shutdown 中は publish も commit も実行されないため、**durable admission を保持しても意味がない**。`requestShutdown()` 時に:
  ```
  requestShutdown():
    shutdownScheduler_.requestShutdown()   // 既存
    recoveryAdmissionPending_.store(false, release)   // ★ X1: durable admission を破棄
    pendingRecoveryAdmission_ = PendingRecoveryAdmission{}   // クリア
  ```
- **isFullyDrained の整合**: shutdown で durable admission を破棄すれば、`recoveryAdmissionPending == false` が成立し、shutdown 完了（markShutdownComplete）が可能
- **🔴🔴🔴 二十五次レビュー（2026-08-09）— shutdown discard を「ShutdownDiscard」として明示（§8.1）**: `Recovery lost` と `ShutdownDiscard` を同じ意味にしない:
  ```
  Running 中の queue full → durable pending = loss ではない（INV-5 の保証が機能）
  Shutdown 中の durable pending → explicit shutdown discard = 意図的な lifecycle discard
  ```
  - **Telemetry 上で2つを分ける**: `recoveryIntentDropCount_`（queue full の診断）とは別に、**shutdown discard カウンタ**（または telemetry 区分）を設ける
  - **INV-5 との整合**: shutdown discard は「意図的な lifecycle 破棄」であり、INV-5 の「silent loss 禁止」に違反しない（shutdown 後は publish/commit が実行されないため、破棄が正当）
  - **実装**: `requestShutdown()` の durable 破棄時に、discard を telemetry に記録する（shutdown 専用カウンタ or ログ）
- **🔴🔴🔴 二十三次レビュー（2026-08-09）— `RecoveryAdmissionClosed` を shutdown state machine に追加（§4.3）**: `recoveryAdmissionPending_` を追加するだけでは不十分。**build gap 中の isFullyDrained 早期 true を防ぐため、shutdown state machine は以下を含める必要がある**:
  ```
  AdmissionClosed
  + RecoveryAdmissionClosed      ← ★ X1 追加（Recovery の新規受付が閉じた）
  + BuilderStopped               ← ★ X1 追加（Builder join 完了）
  ```
  - **RecoveryAdmissionClosed**: `requestShutdown()` 時（:1352-1358 の durable 破棄）に、**Recovery の新規受付も閉じる**（`submitRecoveryRequest` が以後受理しない）
  - **BuilderStopped**: `stopRebuildThread` の join 完了（:782）を state machine に含める。**isFullyDrained が BuilderStopped を確認する前に true になってはいけない**（build gap の Recovery DSP が残っている可能性）
  - **isFullyDrained の完全条件**: `AdmissionClosed AND RecoveryAdmissionClosed AND BuilderStopped AND pendingIntentCount_ == 0 AND !recoveryAdmissionPending_ AND ...`（§6.7 に反映）
- **注意**: shutdown 中の Recovery build（build gap にある DSP）は `stopRebuildThread` の join で回収される（:782）。build 途中の DSP は DSPGuard が破棄する（既存 :949-954 の dspGuard 契約）
**`takePendingRecoveryAdmission()` の戻り値設計（実コードの RecoveryIntent 構造に整合）**:
```cpp
// RuntimePublicationCoordinator に追加
[[nodiscard]] std::optional<RecoveryIntent> takePendingRecoveryAdmission() noexcept
{
    // 🔴🔴🔴 二十六次レビュー（2026-08-09）— lease（state transition）に変更（必須修正1）
    //   destructive dequeue ではなく、DurablePending → Building へ遷移させる。
    //   build 失敗時は Building → DurablePending へ戻して retry を構造的に保証する。
    if (pendingRecoveryAdmission_.state != PendingRecoveryAdmission::State::DurablePending)
        return std::nullopt;
    // SPSC（Producer=CoordinatorLoop, Consumer=Builder Loop）なので競合なし
    RecoveryIntent intent{
        pendingRecoveryAdmission_.handle,
        pendingRecoveryAdmission_.epoch,
        pendingRecoveryAdmission_.intentId,
        pendingRecoveryAdmission_.buildSource
    };
    pendingRecoveryAdmission_.state = PendingRecoveryAdmission::State::Building;  // ★ クリアしない
    // recoveryAdmissionPending_ は Building 中も true を維持（build gap を isFullyDrained が検出）
    return intent;
}
```
- **注意**: durable 化した時点で `pendingIntentCount_` は rollback 済み（十五次 §14）。`takePendingRecoveryAdmission()` は `pendingIntentCount_` を触らない（transport residency ではないため）
- **🔴🔴🔴 二十六次レビュー（2026-08-09）— build 失敗時の state transition（必須修正1）**: take は lease（DurablePending → Building）のため、**build 失敗時は Building → DurablePending へ戻す**（destructive dequeue ではない）:
  ```
  NoAdmission → reserve → DurablePending → take(lease) → Building
      Building + transient failure（ResourceUnavailable/MKLFailure/PrepareFailure/WarmupFailed）→ DurablePending（retry）
      Building + obsolete（isRebuildObsolete）→ Discarded（state クリア）
      Building + build success → PublishTransport（enqueuePublicationIntentForRuntimeCommit）
  ```
  - **transient failure は durable state を保持したまま retry**（Building → DurablePending に戻し、recoveryAdmissionPending_ は true 維持）。**次サイクルの Builder が再 take する**
  - **obsolete（isRebuildObsolete）**: recoveryGeneration が古い → Discarded（state = NoAdmission にクリア）
  - **build success**: PublishTransport（enqueuePublicationIntentForRuntimeCommit :971）→ state = NoAdmission にクリア + recoveryAdmissionPending_ = false
  - **INV-X1-1（exactly one durable state exists）の整合**: lease 方式では「accepted ⇒ exactly one durable state（DurablePending or Building）exists」が常に成立（destructive dequeue では build 失敗時に durable が消えて不成立）
- **🔴🔴🔴 十八次別視点11調査（2026-08-09）— BuildError 種類と再試行方針を確定**: `RuntimeBuilder::build` は `BuildError` enum（RuntimeBuilder.cpp:55-71）を返す:
  ```
  BuildError::InvalidInput       → build 入力不正（回復不能。Discarded）
  BuildError::ResourceUnavailable → メモリ/リソース不足（一時的。Building → DurablePending）
  BuildError::MKLFailure         → MKL 失敗（一時的。Building → DurablePending）
  BuildError::ConvolverFailure   → convolver 失敗（★2026-08-10 確定: 一時的。Building → DurablePending）
  BuildError::PrepareFailure     → prepare 失敗（一時的。Building → DurablePending）
  BuildError::WarmupFailed       → warmup 失敗（:476。一時的。Building → DurablePending）
  BuildError::InternalError      → 内部エラー（永続。Discarded）
  ```
  - **一時的 failure（ResourceUnavailable / MKLFailure / PrepareFailure / WarmupFailed / ★ConvolverFailure）**: Building → DurablePending に戻し、**次サイクルで retry**（durable state は保持）
  - **永続 failure（InvalidInput / InternalError）**: Discarded（state クリア + recoveryAdmissionPending_ = false）。drop 相当（INV-X1-2/INV-X1-3）
  - **🔴🔴🔴 別視点調査（2026-08-10）— ConvolverFailure の分類を確定（未確定解消）**: ConvolverFailure を**「一時的 failure」に確定**（Building → DurablePending に戻す）。根拠: 一時的と分類して retry しても durable state は保持され、回復不能なら最終的に Critical telemetry で検出される。一方「永続」と分類すると、一時的な convolver 問題で正当な Recovery が即 Discarded になる（Recovery 保証の喪失）。安全側は一時的。
  - **🔴🔴🔴 別視点調査（2026-08-10）— build パスで実際に返る BuildError の実コード照合**: `RuntimeBuilder.cpp` の build パスで実際に返るのは `InvalidInput`（:435）/ `ResourceUnavailable`（:461）/ `InternalError`（:466）/ `WarmupFailed`（:476）/ `None`（:478）のみ。**MKLFailure / ConvolverFailure / PrepareFailure は enum 定義のみで build パスでは未使用**（将来 convolver build 失敗時に備えた保険分類として維持 — 一時的扱いで設計整合）。
- **🔴🔴🔴 十八次別視点7調査（2026-08-09）— 通常 rebuild と Recovery の相互作用（isObsolete の2箇所チェック）**: RebuildDispatch.cpp:975-1138 の**通常 rebuild** は `isObsolete`（:976-978 = `isRebuildObsolete(task.generation) || rebuildThreadShouldExit`）を **build 前（:980）と build 後（:1011）の2箇所**でチェックする。X1 の durable Recovery 消費（RebuildDispatch.cpp:901-973 の後に追加）は、この通常 rebuild の**前**に実行されるため:
  ```
  Builder Loop の各反復:
    1. Recovery 消費（:911 popRecoveryRequest + ★X1 takePendingRecoveryAdmission）← 先に実行
    2. 通常 rebuild タスク処理（:975-1138）
  ```
  - **Recovery 消費は isObsolete チェックの対象外**（Recovery は quarantine された DSP の復旧であり、rebuild generation の obsolete 判定は Recovery 消費時の `recoveryGeneration = currentRebuildRequestGeneration`（:966-967）で暗黙的に最新化）
  - **通常 rebuild の isObsolete チェックは X1 の設計に影響しない**（X1 は Recovery 消費ループ内で完結）
  - **ただし**: Recovery 消費中に新規 rebuild 要求が来た場合、`rebuildRequestGeneration` が進み、`recoverySnapshot.generation = currentRebuildRequestGeneration`（:969）で**最新 generation に更新**される。isRebuildObsolete 誤判定は回避される（既存 :965-967 の設計）
- **🔴🔴🔴 十八次別視点13調査（2026-08-09）— 通常 rebuild の後半処理（IR rebuild / warmup / commit）を確定**: RebuildDispatch.cpp:1024-1138 の通常 rebuild 後半は:
  ```
  IR rebuild（:1025-1041）: newDSP->convolverRt().getIRLength() > 0 なら rebuildAllIRsSynchronous（:1039）
      → isObsolete チェック（:1027）後
  Warmup（:1051-1070）: runtimeBuilder.validateWarmup(*newDSP)
      → warmupError != None なら retryable 判定（:1054, shouldRetryWarmupFailure）
      → retryable なら submitRebuildIntent（:1064）
  refreshLatency + fadeIn（:1085,1088）
  投影値更新（:1104-1115）: oversamplingFactor を DSP 解決値で上書き
  Commit（:1138）: enqueuePublicationIntentForRuntimeCommit(dspToCommit, ...)
  ```
  - **通常 rebuild と Recovery は同一の `enqueuePublicationIntentForRuntimeCommit`（:1138 vs :971）で publish**（X5 の経路1/経路2）
  - **X1 の Recovery 消費はこの後半処理の前に実行**（Builder Loop の各反復で Recovery 消費 → 通常 rebuild の順）
  - **warmup 失敗の retryable 判定（:1054）**: Recovery build でも同様の retryable 判定が可能（一時的 failure は次サイクル再試行 — 別視点11で確定済み）
3. **Coalesce**: `pendingRecoveryAdmission_` の store 時に、同一 handle の既存 Pending があれば buildSource を更新（latest 採用）し、Pending=1 を維持
4. **pendingIntentCount_ との整合（🔴🔴🔴 十五次 §14 の最新判断に統一）**: `pendingIntentCount_` は **transport queue residency** と定義。push 失敗で durable 化した場合は **`pendingIntentCount_` から rollback（fetchSub）し、`recoveryAdmissionPending_ = true` に切替**（INV-X1-5: 1 logical admission = 1 reservation。但し reservation の**追跡先**が transport → durable に移動する）。Builder が durable state から消費した時点で `recoveryAdmissionPending_ = false`

**新規メソッド（RuntimePublicationCoordinator に追加）**:
```
submitRecoveryRequest(handle, buildSource)     // 既存改修: push 失敗時 durable 化
takePendingRecoveryAdmission() → optional<RecoveryIntent>   // 新設: Builder が消費
hasPendingRecoveryAdmission() → bool            // 新設: isFullyDrained 用
```
- **P2 では X1 を実装しない**（P2 は「drop を正しく記録できるか」の INV-5 テストまで。Recovery 保証は X1 で実装）

### 6.2 X2 — Publish completion sequence monotonicity（P1 優先）— **十四次レビューで再設計（最優先）**

> **★ 実コード照合確定（2026-08-10）**: PublishReceiptWaiter（AudioEngine.h:3613-3635）の mutex+cv+monotonic watermark（lastCompleted_:3634）を確認。wait_until の predicate（:3628 `seqId <= lastCompleted_`）で lost wakeup 安全。単一 completion writer（PublishExecutor sole gateway + intentQueue_ FIFO）→ CAS 不要。2 箇所 watermark（m_lastObservedSequence:246 / lastCompleted_）の同期をテストで検証。StateOwner ledger は completion と独立（診断）。
> **★ 別視点調査追記（2026-08-10）— Committed/Completed の実コード照合**: `lastCommittedPublicationSequence_`（AudioEngine.h:2191、Commit.cpp:398 で commit 成功時に publishAtomic）と `PublishReceiptWaiter::lastCompleted_`（:3634）が**別 semantic** であることを実コード確認。**Committed**（publication state transition 成立）≠ **Completed**（executor 側 completion 成立）の分離を裏付け（INV-ISR-05 と整合）。

**現状**: `onPublishCommitted(seqId)`（RuntimePublicationOrchestrator.h:146 / RuntimePublishExecutor.h:84）→ notifyPublishReceipt。`PublishExecutor` が sole execution gateway。**「Producer serialization によってたまたま単調になる」では不十分**。

**設計方針（十四次レビューで確定 — contiguous completion 前提）**: `PublicationSequenceId` を **completion sequence として明示的に monotonic** にする。ただし **PublishExecutor が sole gateway（RuntimePublishExecutor.h:19-20）+ intentQueue_ FIFO のため、completion は seqId 順に発生**し、**CAS は不要**（単一 completion writer）:
```cpp
std::atomic<PublicationSequenceId> lastCompletedSequence_;
// 更新規則（contiguous completion 前提）: store(seq, release) のみ
//   PublishExecutor が sole completion writer（INV-X2-5）のため、CAS 不要
```
- **CAS が不要な理由**: completion が seqId 順に発生するため、`seq > lastCompletedSequence_` の比較すら不要。単一 writer の `store(seq, release)` で十分
- **将来 MPSC completion を許可する場合のみ**、CAS max に変更（後述 sparse completion）
```cpp
// 将来 MPSC completion 許容時のみ（現在は不必要 — sole completion writer）
bool publishCompletion(seq) {
    auto current = lastCompleted.load(acquire);
    while (seq > current) {
        if (lastCompleted.compare_exchange_weak(current, seq, acq_rel, acquire))
            return true;
    }
    return seq == current;
}
```
**現在の実装（contiguous completion）**: `lastCompletedSequence_.store(seq, release)` のみ。CAS の publishCompletion は将来の MPSC 化時のみ。
**本質**: out-of-order を潰すことではなく、**out-of-order が起きても state が rollback しないこと**。

**Receipt の意味も分離**: Global completion watermark（`lastCompletedSequence`）と Per-request receipt（`seq → completed`）を分離。

**🔴🔴🔴 十五次レビュー（2026-08-09）— 4層の sequence 分離を確定（X2 最優先）**:

X2 の最終形は、**4層の sequence を完全に区別**する:
```
① publicationSequenceCounter_      = identity allocation（seqId 採番）
② lastCommittedPublicationSequence_ = committed state（RuntimeWorld が commit 済み）
③ lastCompletedSequence_           = contiguous completion watermark（execution tail 完了）
④ per-request receipt             = producer acknowledgement（自分の request 完了確認）
```
**意味の遷移**:
```
commit成功 → ② Committed
execution tail 完了 → ③ Completion
Producer が自分の request 完了を確認 → ④ Receipt
```
- **① と ② の関係**: `reserveRuntimePublicationIdentity`（AudioEngine.h:3406）で採番 → commit 成功時 Commit.cpp:398 で `lastCommittedPublicationSequence_` 更新
- **🔴🔴🔴 十八次別視点3調査（2026-08-09）— sequence 採番の thread 所有権を確定**:
  ```
  reserveRuntimePublicationIdentity（AudioEngine.h:3407-3416）の呼び出し元:
    RuntimeBuilder.cpp:81（buildRuntimePublishWorld 内）
    RuntimeBuilder.cpp:183（buildRuntimePublishWorld 内・再 build）
  → 採番は RebuildThread 内（RuntimeBuilder が RebuildThread で実行される）
  ```
  - **① sequence 採番は RebuildThread（Builder）で行われる**（`fetchAddAtomic(publicationSequenceCounter_, 1, acq_rel) + 1`、:3412-3414）
  - **② lastCommittedPublicationSequence_ の更新は Commit.cpp:398**（RebuildThread の executor_.publish 経由 → commitRuntimePublication）
  - **同一スレッド（RebuildThread）内で採番 → commit が進行するため、採番順序と commit 順序は同一**（RebuildThread が逐次処理）
  - **Producer serialization は RebuildThread 単一スレッドで成立**（X2 の INV-X2-5「sole completion writer」の前提の一部）
- **③ と ④ の関係**: `waitFor(seq)` は ③（contiguous watermark）で判定。**contiguous completion 前提では `seq <= lastCompletedSequence_` が正しい**（FIFO invariant が保証されるため ③ は「1..N 全て完了」を意味）
- **per-request receipt（④）**: 現在は `waitForPublishReceipt(seqId, timeout)`（AudioEngine.h:3641）が Producer の完了確認。contiguous 前提では ③ で代用可。**将来 out-of-order completion を許す場合のみ** ④ を sparse set に変更
- **INV-X2-2 の明確化**: `completion(seq) implies commit(seq) succeeded`（② → ③ の順序）

**🔴🔴🔴 追加調査（2026-08-09）— per-request receipt の呼び出し元（Producer 側）を確定**:
`waitForPublishReceipt(seqId, kPublishReceiptWaitTimeoutMs)` は **`commitRuntimePublication`（AudioEngine.h:4450）内で呼ばれる**:
```cpp
// AudioEngine.h:4445-4454
auto result = enqueueRuntimePublicationFireAndForget(...);
// 4. 完了通知を待つ（executePublish → orchestrator.onPublishCommitted → notifyPublishReceipt）
if (PublishStageResultTraits::isCommitted(result.stage) && seqId != 0) {
    if (!waitForPublishReceipt(seqId, kPublishReceiptWaitTimeoutMs))
        juce::Logger::writeToLog("[DIAG] commitRuntimePublication: receipt timeout seq=...");
}
```
- **seqId は Producer 自身の割当**（`reserveRuntimePublicationIdentity` で採番）
- **タイムアウトしても Transferred 扱い**（所有権は executePublish が後続で commit するため移譲済み — 呼び出し元は world/DSP を破棄してはならない）
- **🔴🔴🔴 二十三次レビュー（2026-08-09）— timeout semantics を明文化（§6.1）**: `timeout ≠ publish failure`。Producer が receipt timeout しても**所有権は既に Transferred**（executePublish が後続で commit する）。**timeout を publish failure と誤解して rollback すると、実際には Coordinator が publish するため double ownership / double publish が発生**する。したがって:
  ```
  Allocated → Transferred → Committed → Completed
  Producer の lifecycle:
    Allocated:   reserveRuntimePublicationIdentity で採番
    Transferred: OwnerChannel::enqueue で所有権移譲（commitRuntimePublication）
    Committed:   executePublish → publishAndSwap（CoordinatorLoop）
    Completed:   onPublishCommitted → complete()（CoordinatorLoop）
  receipt timeout: Transferred のまま（rollback 禁止・world/DSP 破棄禁止）
  ```
- **X2 の per-request receipt（④）はこの呼び出し元を対象**にする。contiguous completion 前提では ③（watermark）で判定可能。`kPublishReceiptWaitTimeoutMs` は「③ が seqId を超えるまで待つ」タイムアウト

**🔴🔴🔴 十八次別視点4調査（2026-08-09）— deferred publish と contiguous completion の整合（REPAIR_PLAN2.md:914 で確定）**:

REPAIR_PLAN2.md:914 は「**Publish は deferred へ退避できないかぎり completion order = seqId order が保たれる必要あり**」と明記。**deferred publish が contiguous completion 前提に例外を生む可能性**を検証:
- **deferred は単一スロット**（Orchestrator.cpp:360-409 の `deferredSlot_`）: deferred 中に後続 publish が来ると `deferredSlot_` を上書き（:383）。**deferred 中の publish は連続して deferred に落ちる**
- **🔴🔴🔴 十八次別視点13調査（2026-08-09）— trySubmitImpl の crossfade decision と deferred の関係を確定**: `trySubmitImpl`（Orchestrator.cpp:186-218）は:
  ```
  cfDecision = CrossfadeAuthority::evaluate(*oldWorld, *worldOwner, policy)（:193-206）
    → cfDecision.needsCrossfade → spec.execution.transitionActive = true（:221-227）
  → executor_.publish（:263）→ commitRuntimePublication → DeferredFadingActive（admission が hasFading 判定）
  ```
  - **crossfade decision（CrossfadeAuthority）は Deferred の発生源**: `hasFadingRuntimeInWorld`（admission :55-56）が true なら DeferredFadingActive → deferredSlot_ に退避
  - **deferred は「crossfade 中の publish」**（fading runtime が存在する間に新 publish が来ると Deferred）
  - **X2 の completion との関係**: deferred が解消（crossfade 完了）後に re-enqueue → PublishExecutor → commit → completion。**deferred 中は新 publish の completion は発生しない**（re-enqueue 後）
  - **INV-X2-6 の deferred 例外**: deferred 中は最新のみ re-enqueue されるため、cancel された古い seqId の completion は発生しない（contiguous の例外 — 既に §6.2 で明示済み）
- **🔴🔴🔴 十八次別視点15調査（2026-08-10）— evaluateDeferred の stale-discard 判定を確定**: `evaluateDeferred`（PublicationAdmission.cpp:69-91）は deferred の re-enqueue 可否を4段階で判定する:
  ```
  1. Shutdown（:74-75）→ Discard（ShutdownDiscard）     — shutdown 中は publish しない
  2. TTL 超過（:78-80, 30s）→ Discard（StaleDiscard）   — 滞留しすぎた deferred は破棄
  3. Generation 不一致（:83-84）→ Discard（StaleDiscard）— rebuild 世代が変わった
  4. Sequence 後戻り（:87-88）→ Discard（StaleDiscard） — 履歴の後戻り禁止
  → それ以外 → Ready（:90）→ re-enqueue（processDeferredAdmission :525）
  ```
  - **deferred の cancel 条件が確定**: ShutdownDiscard / StaleDiscard（TTL・Generation・Sequence）の3種類
  - **X2 の completion との関係**: StaleDiscard された deferred の seqId は re-enqueue されない → **completion は発生しない**（cancel 相当）。INV-X2-6 の deferred 例外（cancel された古い seqId の completion なし）と整合
  - **ShutdownDiscard（:75）は deferred の shutdown 破棄**（clearDeferredForShutdown :412 の補完）
- **completion の順序**: deferred が解消（crossfade 完了 → `processDeferredAdmission` :502-535 → `submitPublishRequest` 再 enqueue → PublishExecutor）後に、**順次処理**される。deferred 内の publish は単一スロットのため**同時に複数は滞留しない**（上書きで最新のみ残る）
- **X2 の contiguous completion 前提との整合（確定）**:
  ```
  case 1: deferred なし → completion order == seqId order（contiguous 成立）
  case 2: deferred あり → deferred 中の publish は上書きされ、最新のみが re-enqueue される
          → completion は re-enqueue 後に順次発生。contiguous は seqId の連続性でなく
            「re-enqueue された最新 publish が次に commit/completion する」意味に
  ```
- **X2 の INV-X2-6（completion order == publication sequence order）は、deferred 非発生時を前提とした invariant として明示**する。deferred 発生時は「deferred 上書き → 最新のみ commit」のため、**未完の古い seqId は cancel 相当**（receipt は来ない）
- **waitFor の整合**: deferred により古い seqId の publish が cancel されると、その seqId の receipt は来ない。`commitRuntimePublication` の waitForPublishReceipt はタイムアウト（250ms）で Transferred 扱い（既存 :4448-4454）。**これは X2 の contiguous completion 前提と整合**（cancel された publish は Producer 側でタイムアウト処理）

**🔴🔴🔴 十六次レビュー（2026-08-09）— `waitForPublishReceipt()` は削除しない。API semantic は別物**:
`waitForPublishReceipt(seq)` は「**自分の要求が完了したか**」を待つ Producer 側 API（`commitRuntimePublication` が enqueue 後に receipt を待つ同期ラッパ）。`lastCompletedSequence_`（watermark）と **内部実装として同じ watermark に実装してよい**が、**API semantic としては別物として残す**:
```
lastCompletedSequence_   = contiguous completion watermark（③）
waitForPublishReceipt() = per-request producer acknowledgement（④）
```
- 内部実装は同じ watermark を参照してよい（contiguous completion 前提）
- **API の削除・統合はしない**（Producer は自分の seqId の完了を待つ必要があるため）

**🔴🔴🔴 追加調査（2026-08-09）— Committed state と Completion watermark の区別（既存の2つの sequence）**:
`AudioEngine` には **2つの独立した sequence 変数**が存在する:
```cpp
// AudioEngine.h:2189 — seqId 割当（reserveRuntimePublicationIdentity で fetchAddAtomic+1）
std::atomic<PublicationSequenceId> publicationSequenceCounter_ { 0 };
// AudioEngine.h:2191 — Committed state（Commit.cpp:398 で world.publication.sequenceId に更新）
std::atomic<PublicationSequenceId> lastCommittedPublicationSequence_ { 0 };
```
- **`lastCommittedPublicationSequence_`（Committed state）**: RuntimeWorld が commit された時点の sequence（Commit.cpp:398）
- **`lastCompletedSequence_`（Completion watermark）**: onPublishCommitted → notifyPublishReceipt の完了（PublishReceiptWaiter）

**X2 の設計では両者を明確に区別**する:
```
Committed:  commit(seq) succeeded → lastCommittedPublicationSequence_ 更新
Completion: onPublishCommitted → lastCompletedSequence_ 更新（waitFor の source）
```
現行は `onPublishCommitted`（:305）が両方の意味を兼ねる（`m_lastObservedSequence` 更新 + `notifyPublishReceipt`）。X2 では **`lastCommittedPublicationSequence_`（Committed）と `lastCompletedSequence_`（Completion）を分離**し、INV-X2-2（completion implies commit succeeded）を明示。

**🔴🔴🔴 十四次レビュー（2026-08-09）— X2 は「CAS を入れるか」ではなく「completion を何と定義するか」から再設計（最優先）**:

**根本的な意味論の問題**: 現行 `waitFor(seq) := seq <= lastCompleted_`（AudioEngine.h:3628）は、**「最大完了 sequence」と「seqId の publish が完了した」を同一視**している。例えば:
```
seq=10 commit, seq=11 commit, completion が 11 のみ（10 未完了）
→ lastCompleted=11
→ waitFor(10) = (10 <= 11) = true   ← 誤り（10 は未完了）
```
これは `PublishReceiptWaiter` の high-water mark が **contiguous completion 前提**（10 が完了 ⇒ 1..10 全て完了）であるため。

**ただし現行は FIFO completion invariant で正しい**: `PublishExecutor` が sole gateway（RuntimePublishExecutor.h:19-20）で `intentQueue_` を FIFO 処理（コメント :3605「executePublish は intentQueue_ を FIFO で処理するため seqId は単調増加で完了する」）するため、**現状では completion は seqId 順に発生**し、`waitFor(10)` が 11 完了時に true なのは「10 も完了済み」を意味して正しい。

**したがって X2 の本質は**: CAS や sparse completion の導入ではなく、**Publish completion order を architectural invariant として固定**すること。PublishExecutor を単一 serialized path に固定できる限り:
```cpp
lastCompletedSequence_.store(seq, release);   // monotonic store で十分（CAS 不要）
```
`waitFor(seq) := seq <= lastCompletedSequence_` は contiguous completion 前提のまま正しい。

**推奨設計（contiguous vs sparse の決定）**:
| 方式 | 前提 | 実装 |
|------|------|------|
| **contiguous completion**（採用） | PublishExecutor が sole gateway + FIFO → completion は seqId 順 | `lastCompletedSequence_.store(seq, release)` + `waitFor(seq) := seq <= lastCompleted` |
| sparse completion（将来 MPSC completion 許容時） | completion が out-of-order | `completedThrough_`（contiguous frontier）+ `completedOutOfOrder_`（sparse set）。`waitFor(seq)` は frontier と sparse を併用 |

**wraparound semantics（十四次指摘 — 未確定を確定）**:
`seq > current` の unsigned 比較は **wraparound に不正**（`current=UINT64_MAX-1, new=0` で `0 > MAX-1 == false`）。以下を明示:
- **案A（採用）: sequence は実質 wrap しない契約** — `uint64_t overflow is architecturally impossible`（実用上成立。PublicationSequenceId は単調増加で、現行設計では永続的に使用する世界の数が wrap を起こさない）。**wrap テストをやめる**
- 案B: serial number arithmetic（`isNewer(a,b)` を定義）— 将来 wrap が必要になった場合のみ

**⚠️ 実装上の注意**: `waitFor` を lock-free にすると、**complete の notify を失う**可能性がある（notify 前に waitFor 開始すれば問題なし、逆順は deadline まで spin）。cv を維持する場合は、**monotonic watermark の更新後も cv_.notify_all() を呼ぶ**（既存 :3620 の動作を維持）。

**不変条件**:
```
INV-X2-1: lastCompletedSequence never decreases
INV-X2-2: completion(seq) implies commit(seq) succeeded
INV-X2-3: receipt(seq) implies completion watermark >= seq
INV-X2-4: stale completion cannot overwrite newer completion
INV-X2-5: PublishExecutor は sole completion writer（architectural invariant — sparse 不要の根拠）
INV-X2-6: completion writer is unique AND completion order is identical to
          publication sequence order（十六次 §5.1 追加）
          → Intent FIFO → PublishExecutor FIFO → commit FIFO → completion FIFO
```
**🔴🔴🔴 十六次レビュー（2026-08-09）— FIFO completion invariant を実装契約として強制**:
`INV-X2-5`（sole completion writer）だけでなく、**`completion order == publication sequence order`（INV-X2-6）をコード上で固定**する。将来 MPSC completion / parallel publish / async completion を許した瞬間、`seq <= lastCompletedSequence_` の意味は壊れる。**source-level architectural test** で `PublishExecutor` 以外から completion が発生しないことを固定する。
**必須テスト**: 正常系 10→11→12（FIFO completion invariant の検証）。out-of-order テストは **PublishExecutor が sole gateway である限り不要**（将来 sparse 化する場合のみ 11→10 / duplicate / wraparound を追加）。

**🔴🔴🔴 追加調査（2026-08-09）— `PublishReceiptWaiter::complete()` の現行実装と X2 の整合（確定）**:

現行 `complete()`（AudioEngine.h:3614-3621）を実コードで検証:
```cpp
// AudioEngine.h:3614-3621（現行）
void complete(convo::isr::PublicationSequenceId seqId) noexcept
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (seqId > lastCompleted_) { lastCompleted_ = seqId; }   // ← mutex 下の read-modify-write
    }
    cv_.notify_all();
}
```
- **現行は mutex ガード付きの `if (seqId > lastCompleted_)`**（guard: stale completion が新しい watermark を上書きしない INV-X2-4）
- **X2 の contiguous completion 前提では**（PublishExecutor が sole completion writer で seqId 順に complete が来る）:
  - `seqId > lastCompleted_` は常に成立（単調増加のみ）
  - **mutex の要否**: `complete()` は CoordinatorLoop の単一スレッド（PublishExecutor）からのみ呼ばれるが、`waitFor()` は Producer スレッド（RebuildThread 等）から呼ばれるため、**cv との同期に mutex は必要**
- **🔴🔴🔴 十八次調査（2026-08-09）— complete() の thread 所有権を確定（CoordinatorLoop 実装検証）**:
  ```
  CoordinatorLoop::run()（ISRCoordinatorLoop.cpp:31-43）:
    while (!threadShouldExit()) {
        if (engine_.isShutdownInProgress()) break;
        engine_.runCoordinatorPhase();   // → processIntent → PublishExecutor::executePublish → onPublishCommitted → complete()
        wait(kIntervalMs);               // 1ms 周期（kIntervalMs）
    }
  ```
  - **complete() の呼び出しスレッド = CoordinatorLoop スレッド（単一）**
  - **waitFor() の呼び出しスレッド = Producer（複数スレッド）**: `commitRuntimePublication`（waitForPublishReceipt を含む同期 publish）の呼び出し元は **5箇所**:
    ```
    1. PrepareToPlay.cpp:155,277（Bootstrap/idle publish）
    2. ReleaseResources.cpp:175（shutdown idle publish）
    3. Timer.cpp:918（crossfade 完了 publish・MessageThread Timer）
    4. Transition.cpp:25（transition 完了 publish）
    5. PublicationExecutor.cpp:53（Orchestrator 経由・RebuildThread）
    ```
  - **複数 Producer スレッド**（MessageThread Timer / RebuildThread / PrepareToPlay 等）から waitFor される。mutex は**複数 Producer 間の cv 同期にも必要**
  - **mutex は cv との同期に必要**（CoordinatorLoop と複数 Producer の間）。`lastCompleted_` の更新自体は単一 writer だが、`waitFor` の predicate 評価（:3628 `seqId <= lastCompleted_`）が mutex 下で行われるため、**data race は構造的に排除されている**（複数 waitFor は mutex で直列化）
  - **shutdown 時の注意**: `isShutdownInProgress()` で CoordinatorLoop が break する（:35-36）ため、**shutdown 中は complete() が呼ばれない**。Producer の waitFor はタイムアウト（250ms）で Transferred 扱いになる（既存 :4448-4454 の設計が正しい）
- **X2 の実装選択（確定）**:
  ```
  案1（推奨・最小変更）: 現行構造を維持し、`lastCompleted_` の更新を monotonic watermark として明示。
      mutex 下の `if (seqId > lastCompleted_)` を残す（INV-X2-4 を満たす）。
      セマンティクスをコードコメントで固定（contiguous completion 前提・sole writer）。
  案2（将来 MPSC 時）: 案1 に加え CAS max（sparse completion）。
  ```
- **`lastCompleted_` の初期値**: 0（AudioEngine.h:3634）。**初回 publish の seqId=1** のため `1 > 0` で正常。`waitFor(0)` は即 true（seqId=0 は無効とみなす — commitRuntimePublication は `seqId != 0` ガード済み :4448）
- **`markReceiptReclaimComplete()`（ProcessIntent.cpp:41）との関係**: processIntent 末尾で呼ばれるが、これは receipt とは独立（reclaim 完了のマーキング）。X2 の `lastCompleted_` とは別物
- **🔴🔴🔴 十八次別視点9調査（2026-08-09）— `pendingReceipt_` と `PublishReceiptWaiter` の区別を確定**: AudioEngine には **2種類の「receipt」** が存在する:
  ```
  receipt #1: pendingReceipt_（AudioEngine.h:4683, optional<PublishReceipt>）
    = Timer の retire 用 PublishReceipt（handle + publicationEpoch）
    = storeReceipt（:1157）/ resetReceipt（:1176-1178）/ retirePublishedDSP（Timer.cpp:1774-1780）で管理
    = markReceiptReclaimComplete（ProcessIntent.cpp:41）で解放（Epoch Safe 通知）
    = ★ X2 の completion watermark とは無関係（Timer の retire epoch 伝搬用）

  receipt #2: PublishReceiptWaiter（AudioEngine.h:3613-3635）
    = X2 の completion watermark（lastCompleted_）
    = onPublishCommitted → notifyPublishReceipt → complete() で管理
    = Producer の waitForPublishReceipt（自分の publish 完了確認）
  ```
  - **receipt #1（pendingReceipt_）は X2 のスコープ外**: Timer の retire epoch 伝搬用であり、X2 の completion ordering と無関係
  - **X2 の設計は receipt #2（PublishReceiptWaiter）のみを対象**にする
  - **X2 テストでは receipt #1 / receipt #2 を混同しない**（pendingReceipt_ は retire 同期、PublishReceiptWaiter は completion watermark）

**X2 の completion 経路（実コード検証・確定）**:
```
PublishExecutor::executePublish（RuntimePublishExecutor.h:84）
  → orchestrator.onPublishCommitted(seqId)（RuntimePublicationOrchestrator.cpp:305）
      ├─ m_lastObservedSequence = seqId（Orchestrator.h:246 更新 :310）
      ├─ m_lastProgressTimestampUs = now（:311）
      └─ engine_.notifyPublishReceipt(seqId)（:313）
          → publishReceiptWaiter_.complete(seqId)（AudioEngine.h:3639）
              → mutex 下で lastCompleted_ 更新（:3618）
              → cv_.notify_all()（:3620）
Producer 側:
  commitRuntimePublication（AudioEngine.h:4450）
  → waitForPublishReceipt(seqId, 250ms)（:3641）
      → publishReceiptWaiter_.waitFor(seqId, 250ms)（:3623-3630）
          → cv_.wait_until(lock, deadline, [&]{ return seqId <= lastCompleted_; })
```
- **🔴🔴🔴 十八次別視点11調査（2026-08-09）— cv 動作の詳細（PublishReceiptWaiter :3613-3635）を確定**:
  ```
  complete(seqId):
    lock(mutex_) → if (seqId > lastCompleted_) lastCompleted_ = seqId → unlock
    → cv_.notify_all()                                                    // :3620

  waitFor(seqId, timeoutMs):
    unique_lock(mutex_)
    deadline = now + timeoutMs
    cv_.wait_until(lock, deadline, [&]{ return seqId <= lastCompleted_; })  // :3628 predicate
    return seqId <= lastCompleted_                                          // :3629
  ```
  - **wait_until の predicate（:3628）は `seqId <= lastCompleted_`**（contiguous completion 前提: watermark が seqId を超えれば seqId も完了）
  - **notify_all（:3620）は complete のたびに実行**（複数 Producer の waitFor を全て起こす）
  - **lost wakeup の安全性**: wait_until は predicate 付き（:3628）のため、**notify 前に waitFor が始まっても predicate 評価で即復帰**する。notify を逃しても timeout までに predicate が再評価される
  - **deadline 到達後**: predicate が false なら false を返す（:3629）→ commitRuntimePublication は timeout を Transferred 扱い（既存 :4448-4454）
  - **X2 の案1（最小変更）はこの構造を維持**（mutex + cv + monotonic watermark）
- **`m_lastObservedSequence`（Orchestrator.h:246）と `lastCompleted_`（AudioEngine.h:3634）の2箇所の watermark は同期**する（両方 onPublishCommitted から更新）。**X2 テストでは両者の同期を検証**（十六次 U4 / §6.8）
- **🔴🔴🔴 十八次別視点8調査（2026-08-09）— StateOwner の ledger との関係を確定**: `RuntimePublicationStateOwner`（RuntimePublicationState.h:100-159）は **State + Ledger のカウント**（submitted/built/validated/published/retired/reclaimed/rejected/executorFailed）を記録する。X2 の completion との関係:
  ```
  StateOwner ledger: onSubmitted → onBuilt → onValidated → onPublished（trySubmitImpl 内）
                    onRetired / onReclaimed（retire/reclaim path）
  X2 completion:     onPublishCommitted → notifyPublishReceipt → complete() → lastCompleted_
  ```
  - **StateOwner の ledger は X2 の completion watermark と独立**（診断 ledger）。`onPublished` は trySubmitImpl（:289）で記録され、`onPublishCommitted`（CoordinatorLoop）とは別タイミング
  - **X2 の completion ordering は StateOwner の ledger に依存しない**（ledger は published 数カウントのみ、completion sequence の ordering は PublishReceiptWaiter が管理）
  - **X2 テストでは両者を混同しない**: ledger の publishedCount は「publish 試行数」、lastCompleted_ は「completion watermark」

### 6.3 X3 — shutdownReclaim 二系統の統合（P2 優先）

> **★ 実コード照合確定（2026-08-10）**: shutdownReclaim 呼び出し元 3 箇所（AudioEngine.h:2027 CacheMap::dtor / ReleaseResources.cpp:415,420）+ requestReclaim 呼び出し元（AudioEngine.h:4248 / Retire.cpp:83）を確認。2 つの ShutdownPhase enum（AudioEngine:2521-2530 / ISRShutdown:25-41）の整合はテストで固定。**INV-X3-5**: pendingReclaimHandles_（:4616）を reclaim pending の source of truth にし、isFullyDrained に pendingReclaimHandles.empty() を追加。closeReaderRegistration は `setShutdownPhase(StopAudio)` 完了時に set（enum 変更なし）。
> **★ 別視点調査追記（2026-08-10）— readerRegistrationClosed の実装対象を実コードで確認**: 実装対象は **`src/core/EpochDomain.h`**（dash :2233-2267 に実装詳細済み）。実コード確認: `registerReaderThread`（:44-60）と `reserveReaderThread`（:79-103）は**両方とも registrationClosed_ ガード未実装（★ 2026-08-11 実装済み: EpochDomain.h に registrationClosed_ / closeReaderRegistration() / readerRegistrationClosed() を実装）**（X3 で導入）。ISRRetireRouter（:71 委譲）・RCUReader（acquireThreadSlot）経由の登録も EpochDomain 委譲で自動的に封じられる（dash :2266-2271 済み）。

**現状**: runtime（requestReclaim → epoch safety）と shutdown（shutdownReclaim → reader 停止後）の二系統。**無理に一本化しない**。

**設計方針**: **Reclaim Authority は一つ、Safety Precondition が二種類**に変える:
```cpp
enum class ReclaimMode { RuntimeEBR, ShutdownQuiescent };
struct ReclaimRequest { DSPHandle handle; ReclaimMode mode; };
```
- **RuntimeEBR**: reader epoch check → safe? yes → reclaim / no → pending
- **ShutdownQuiescent**: AdmissionClosed + all producers joined + Coordinator joined + Builder joined + Audio reader stopped → reclaim
- `shutdownReclaim()` bypass API を残さず、**同じ Reclaim Authority に通す**（`reclaim(Granted, ShutdownQuiescent, handle)`）

**最大の注意点**: 「shutdown だから reclaim() してよい」ではない。**明示的な shutdown phase assertion**（ShutdownPhase >= AudioStopped AND all producers joined AND Coordinator stopped AND Builder stopped）を満たして初めて ShutdownQuiescent を許可。満たさなければ **NO reclaim → Faulted**。

**🔴🔴 十二次レビュー（2026-08-09）— 既存呼び出し元への影響を確定**:
ReclaimMode 導入は、既存の2つの呼び出し元を Reclaim Authority に統合する:
| 既存呼び出し元 | 現在の経路 | X3 での対応 |
|---------------|-----------|------------|
| `AudioEngine.h:4248`（requestReclaimHandle） | `requestReclaim(handle, dspHandleRuntime_, *m_retireRouter)` | `reclaim(ReclaimMode::RuntimeEBR, handle)` に変更（epoch safety は内部で維持） |
| `AudioEngine.Retire.cpp:83`（retireDSPHandleForRuntime） | 同上 | 同上（RuntimeEBR） |
| `AudioEngine.Retire.cpp:41`（drainDeferredRetireQueues） | `setReclaimInFlightCount(1)` → `tryReclaim()` → `m_coordinator.reclaim()` → `setReclaimInFlightCount(0)` + pendingReclaimHandles_ 再試行 | `reclaim(ReclaimMode::RuntimeEBR, ...)` を経由（保留再試行機構を維持） |

**🔴🔴🔴 十八次別視点6調査（2026-08-09）— drainDeferredRetireQueues の保留再試行機構（pendingReclaimHandles_）を確定**:
実コード検証（AudioEngine.Retire.cpp:41-114）:
```cpp
void AudioEngine::drainDeferredRetireQueues(bool allowDuringShutdown) noexcept
{
    ...
    runtimePublicationBridge_.setReclaimInFlightCount(1);       // :48
    m_retireRouter->tryReclaim();                               // :49
    m_coordinator.reclaim(m_retireRouter->getMinReaderEpoch()); // :51
    runtimePublicationBridge_.setReclaimInFlightCount(0);       // :52

    // 保留中 reclaim の再試行（slot リーク防止）
    std::vector<convo::isr::DSPHandle> pending;
    { std::lock_guard<std::mutex> lock(pendingReclaimHandlesMutex_); pending.swap(pendingReclaimHandles_); }  // :60-64
    for (const auto& handle : pending) {
        if (dspHandleRuntime_.isRetired(handle)) {              // :75（Quarantined/Reclaimed は他経路が所有）
            const auto retireEpoch = m_retireRouter->currentEpoch();
            const auto minReaderEpoch = m_retireRouter->minReaderEpoch();
            if (retireEpoch < minReaderEpoch) {                 // :79 epoch 安全
                if (!runtimePublicationBridge_.requestReclaim(handle, dspHandleRuntime_, *m_retireRouter))
                    pendingReclaimHandles_.push_back(handle);   // :83-87 再登録（TOCTOU 対策）
            } else {
                pendingReclaimHandles_.push_back(handle);       // :91-92 未安全 → 再登録
            }
        }
    }
}
```
- **X3 の `reclaim(ReclaimMode::RuntimeEBR, ...)` はこの保留再試行機構と整合**する: `drainDeferredRetireQueues` が `reclaim(RuntimeEBR, handle)` を呼び、epoch 不安全なら false（再登録）
- **X3 での変更点**: `requestReclaim`（:83）を `reclaim(ReclaimMode::RuntimeEBR, handle, ...)` に置換（内部で retire + epoch 確認 + reclaim を実行）
- **pendingReclaimHandles_ の登録/再登録ロジックは不変**（slot リーク防止の core 機構）
- **isRetired ガード（:75）**: Quarantined/Reclaimed 状態の handle は他経路が所有するため、X3 の reclaim もこのガードを維持する（quarantine lifecycle との二重管理防止）
| `AudioEngine.h:2027`（CacheMap::dtor） | `rt.shutdownReclaim(entry.second)` | `reclaim(ReclaimMode::ShutdownQuiescent, handle)` に変更（phase assertion 必須） |
| `ReleaseResources.cpp:415,420`（active/fading handle） | `dspHandleRuntime_.shutdownReclaim(handle)` | 同上（ShutdownQuiescent + phase assertion） |

- **🔴🔴🔴 十八次別視点3調査（2026-08-09）— shutdownReclaim 呼び出し元の実装順序を確定**:
  ```
  CacheMap::~CacheMap（AudioEngine.h:2015-2037）:
    if (shutdownPhase >= Destroy) {
        for (entry) {
            delete EQCoeffCache（:2026）      ← ★ delete を先に実行
            rt.shutdownReclaim(entry)（:2027） ← その後 slot reclaim
        }
    }
  ReleaseResources.cpp:410-421（VerifyDrained）:
    retire(activeHandle)（:414）→ shutdownReclaim(activeHandle)（:415）  ← retire 先・reclaim 後
    retire(fadingHandle)（:419）→ shutdownReclaim(fadingHandle)（:420）
  ```
  - **CacheMap::dtor は `delete` を先に実行**してから `shutdownReclaim` を呼ぶ（DSPCore ではなく EQCoeffCache の削除。slot reclaim はその後）
  - **X3 の `reclaim(ReclaimMode::ShutdownQuiescent, ...)` 移行時はこの順序を維持**する: delete（物理解放）→ reclaim（slot 状態遷移）。phase assertion は `shutdownPhase >= Destroy`（CacheMap）と VerifyDrained（ReleaseResources）が満たす
  - **ReleaseResources は `retire` → `shutdownReclaim` の順**（reclaim は retire 済み slot を遷移）。X3 の reclaim は内部で retire を実行する設計（:1555）だが、**既に retire 済みの handle への再 retire は冪等**（ISRDSPHandle の retire が冪等 — REPAIR_PLAN2.md:156 参照）

- **`requestReclaim`（ISRRuntimePublicationCoordinator.cpp:573）を `reclaim(ReclaimMode, ...)` に拡張**し、RuntimeEBR では既存の epoch 再確認ロジックを維持、ShutdownQuiescent では phase assertion を追加
- **`shutdownReclaim`（ISRDSPHandle.h:171）は bypass API として廃止**（または内部で Reclaim Authority を呼ぶ薄いラッパーに変更）
- **X3 の実装は P2（§1.1-1.4）と同時には行わない**（§6.9 のとおり X3 は最後）。P2 完了後に独立タスクとして実施

**🔴🔴🔴 十一次/十二次レビュー追記 — 具体的な設計（既存コード構造）**:

`requestReclaim`（ISRRuntimePublicationCoordinator.cpp:573-608）の既存構造:
```cpp
bool requestReclaim(handle, handleRuntime, router) {
    handleRuntime.retire(handle);                              // 1. executeRetire
    const auto retireEpoch = router.currentEpoch();           // 2. waitReaders
    const auto minReaderEpoch = router.minReaderEpoch();
    if (retireEpoch >= minReaderEpoch) {                       // :589 epoch 不安全
        setReclaimInFlightCount(load+1);
        return false;                                          // → pending 再試行
    }
    handleRuntime.reclaim(handle);                             // 3. executeReclaim
    setReclaimInFlightCount(0);
    return true;
}
```
- **🔴🔴🔴 十八次別視点3調査（2026-08-09）— `reclaimInFlightCount_` の管理を確定**:
  - **`setReclaimInFlightCount(+1)`（:592）は「epoch 不安全で遅延」を表す**: `pendingReclaimHandles_` に再登録される handle の数と対応。isFullyDrained の `reclaimInFlightCount == 0` は「遅延 reclaim なし」を保証
  - **`setReclaimInFlightCount(0)`（:606）は「reclaim 完了」**: 単一のカウンタを 0 にリセットする設計。**複数 handle の並行 pending を正確に数えていない**（+1 して 0 リセットの累積近似）。既存の簡略設計であり、X3 の RuntimeEBR モードはこのまま維持
  - **X3 の ShutdownQuiescent モード**: epoch 判定をスキップするため `setReclaimInFlightCount` は**呼ばない**（:592 相当なし）。precondition（isQuiescent）成立後に `handleRuntime.reclaim(handle)` を直接実行
  - **`reclaimInFlightCount_` と X3 の整合**: ShutdownQuiescent は Reader 停止済み（activeReaderCount==0 + readerRegistrationClosed）のため遅延が発生せず、`reclaimInFlightCount_` は 0 のまま。isFullyDrained の `reclaimInFlightCount == 0` チェックと整合
- **🔴🔴🔴 二十六次レビュー（2026-08-09）— `reclaimInFlightCount_` を単独の truth source にしない（必須修正2・INV-X3-5）**: `reclaimInFlightCount_` は「+1 → 0 reset」の近似カウンタで、複数 pending reclaim を正確に数えない（handle A → pending, handle B → pending, A → success で count=0 なのに B が pending という状態を作り得る）。**isFullyDrained() が `reclaimInFlightCount_ == 0` だけを見ると、reclaim pending なのに drained と誤判定**する。したがって:
  ```
  INV-X3-5: ShutdownQuiescent completion requires:
      pendingReclaimHandles_.empty()
      AND reclaimInFlight == 0
  ```
  - **`pendingReclaimHandles_`（AudioEngine.h:4616, mutex 保護）が reclaim pending の実際の source of truth**（retry 対象を保持）
  - **推奨**: `isFullyDrained()` の shutdown 条件に `pendingReclaimHandles_.empty()` を追加（既存 architecture を大きく変更しない）。または `reclaimInFlightCount_` を正確な residency count へ変更（変更面積大）
  - **`reclaimInFlightCount_` は診断値に降格**するか、`pendingReclaimHandles_` と併用する（source of truth は pendingReclaimHandles_）
  - **実装**: `isFullyDrained()` は `pendingReclaimHandles_.empty() AND reclaimInFlightCount_ == 0` を評価する（mutex 保護で pendingReclaimHandles_ を読む）

**X3 での設計（ReclaimMode 導入）**:
```cpp
enum class ReclaimMode { RuntimeEBR, ShutdownQuiescent };

bool reclaim(ReclaimMode mode, const DSPHandle& handle,
             DSPHandleRuntime& handleRuntime, ISRRetireRouter& router,
             const ShutdownContext* shutdownCtx = nullptr) noexcept
{
    // 0. validate reclaim precondition（retire 前 — 十四次指摘）
    //    ★ 旧案は「retire 実行後に phase assertion」で、phase 不正時に
    //      state transition が既に発生する問題があった。正しくは precondition を先に評価。
    if (mode == ReclaimMode::ShutdownQuiescent) {
        if (!shutdownCtx || !shutdownCtx->isQuiescent())
            return false;   // NO reclaim（retire も実行しない）→ Faulted（slot リーク防止）
    }

    handleRuntime.retire(handle);   // 1. executeRetire（共通・precondition 通過後）

    if (mode == ReclaimMode::RuntimeEBR) {
        // 2. epoch 判定（RuntimeEBR のみ）
        const auto retireEpoch = router.currentEpoch();
        const auto minReaderEpoch = router.minReaderEpoch();
        if (retireEpoch >= minReaderEpoch) {
            setReclaimInFlightCount(load+1);
            return false;  // pending 再試行（既存 :589-597）
        }
    }
    // ShutdownQuiescent: epoch 判定をスキップ — phase assertion が reader 停止を保証

    handleRuntime.reclaim(handle);   // 3. executeReclaim（共通）
```
- **🔴🔴🔴 十八次別視点5調査（2026-08-09）— `DSPHandleRuntime::reclaim()` の free list ロックを確認**: `reclaim()`（ISRDSPHandle.cpp:129-148）は `std::lock_guard<std::mutex> lock(freeListMutex_)`（:133）で **free list（freeSlots_ / freeSize_）をロック**する:
  ```cpp
  void DSPHandleRuntime::reclaim(DSPHandle handle) {
      if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS) return;
      std::lock_guard<std::mutex> lock(freeListMutex_);      // :133 ← mutex ロック
      auto& reg = registry_[handle.slot];
      if (consumeAtomic(reg.generation, acquire) != handle.generation) return;  // :137 stale 検出
      if (consumeAtomic(reg.state, acquire) == DSPState::Reclaimed) return;      // :141 二重 reclaim 防止
      reg.instance = nullptr;
      publishAtomic(reg.state, DSPState::Reclaimed, release);
      if (handle.slot != 0 && freeSize_ < MAX_DSP_SLOTS)
          freeSlots_[freeSize_++] = handle.slot;             // :146-147 free list 返却
  }
  ```
  - **X3 の reclaim(ReclaimMode) はこの reclaim() を呼ぶ**ため、**freeListMutex_ をロック**する（NonRT のため許容）
  - **RuntimeEBR / ShutdownQuiescent の両モードとも同一の reclaim() を使用**（free list の扱いは共通）
  - **RT スレッドからは reclaim() を呼ばない**（INV-X4-4 と整合 — reclaim は NonRT）
  - **free list の slot 再利用（FUTURE-5）**: reclaim で Reclaimed 化された slot は free list に戻り、次回 register で再利用される。X3 の reclaim はこの free list 返却を維持する（既存動作）

**🔴🔴🔴 十八次別視点8調査（2026-08-09）— reclaim と物理削除の分離（enqueueWithRetry 経由）を確定**:
実コード検証（DSPLifetimeManager.cpp:45-100）:
```cpp
void DSPLifetimeManager::retire(void* dsp, uint64_t publicationEpoch) noexcept {
    ...
    const auto result = router_->enqueueWithRetry(            // :49
        dsp, &AudioEngine::destroyDSPCoreNode, epoch, DeletionEntryType::Generic);
    // enqueue 失敗は enqueueWithRetry 内部で RetireQuarantineStore へ移送済み
}
void DSPLifetimeManager::retireByHandle(convo::isr::DSPHandle handle) noexcept {
    ...
    engine_.dspHandleRuntime_.retire(handle);                 // :89（slot 状態遷移のみ）
    // requestReclaimHandle → requestReclaim（reclaim slot）
}
```
- **物理削除（DSPCore* delete）は retire path の `enqueueWithRetry`（:49）が担当**（destroyDSPCoreNode を deferred delete）
- **reclaim（slot 状態遷移）と物理削除（DSPCore* delete）は分離**: reclaim は slot を Reclaimed に遷移するだけで、物理削除は enqueueWithRetry が epoch 安全に実行（既存 :602-603 のコメントと整合）
- **X3 の reclaim(ReclaimMode) は物理削除を含まない**（reclaim は slot 遷移のみ）。物理削除は DSPLifetimeManager::retire/retireByHandle が enqueueWithRetry で行う
- **🔴🔴🔴 十八次別視点11調査（2026-08-09）— ISRRetireRouter::retire の実装詳細を確定**: `ISRRetireRouter::retire`（ISRRetireRouter.cpp:149-158）は:
  ```cpp
  void ISRRetireRouter::retire(void* ptr, void (*deleter)(void*)) noexcept {
      if (ptr == nullptr || deleter == nullptr) return;
      const auto result = enqueueWithRetry(ptr, deleter,
          provider_->currentEpoch(), DeletionEntryType::Generic);   // :154 epoch = currentEpoch
      if (result != RetireEnqueueResult::Success) { /* Future: HealthMonitor 通知 */ }
  }
  RetireEnqueueResult ISRRetireRouter::enqueueWithRetry(...) noexcept {
      auto result = enqueueRetire(ptr, deleter, epoch, type);        // :167 通常 enqueue
      if (result == Success) return result;
      for (int attempt = 0; attempt < kMaxRetry; ++attempt) {        // :172 最大2回リトライ
          provider_->tryReclaim();                                   // :174
          result = enqueueRetire(ptr, deleter, epoch, type);
          if (result == Success) return result;
          if (result != QueuePressure) break;                        // :178 QueuePressure 以外は即終了
      }
      return result;
  }
  ```
  - **retire は `provider_->currentEpoch()`（:154）を epoch として enqueueWithRetry に渡す**（reclaim の epoch 判定とは独立）
  - **enqueueWithRetry は最大2回の tryReclaim → enqueue リトライ**（:172-178）。QueuePressure 以外（Shutdown 等）は即時終了
  - **🔴🔴🔴 十八次別視点15調査（2026-08-10）— QueuePressure 時の RetireQuarantineStore 移送を確定**: `enqueueWithRetry`（ISRRetireRouter.cpp:182-203）は全リトライ失敗時（QueuePressure/QueueFull）に:
    ```cpp
    if (result == QueuePressure || result == QueueFull) {           // :188
        const bool stored = m_retireQuarantine.quarantine(          // :190
            ptr, deleter, epoch, type, "enqueueWithRetry:QueuePressure",
            /*publicationSequenceId=*/0, /*generation=*/0);
        if (!stored) assert(false && "RetireQuarantineStore capacity exhaustion");  // :199
    }
    // ★ Future: runtimeHealth_->notifyQueuePressure(QueuePressureInfo{...});   // :202
    ```
    - **queue full は RT 参照中の可能性が高いため、即時解放は UAF を生む** → RetireQuarantineStore で安全保持（:190-192）
    - **store full 時は delete を絶対しない**（UAF 構造的排除・:195-199）。capacity exhaustion は health escalation（quarantineOverflowCount 監視）で先行検知
    - **X3 の reclaim との関係**: QueuePressure 時の退避移送は既存機構（X6 の RetireQuarantineStore::residentCount() が source）。X3 の reclaim はこれに影響しない
  - **X3 の reclaim との関係**: reclaim（slot 遷移）と retire（物理削除予約）は独立。X3 は reclaim 側のみを変更し、retire の enqueueWithRetry リトライ機構は不変
- **INV-X3 の整合**: X3 の reclaim は「slot 状態遷移 + free list 返却」に限定。DSPCore* の物理削除は X3 の責務外（retire path の enqueueWithRetry）

**🔴🔴🔴 十四次レビュー（2026-08-09）— shutdown precondition の十分条件を強化**:
「Audio reader stopped」だけでは不十分。**ShutdownQuiescent の precondition は以下を含む**:
```
ShutdownPhase >= AudioStopped
AND all producers joined
AND Coordinator joined
AND Builder joined
AND Audio processing stopped
AND activeReaderCount == 0
AND reader re-entry impossible   ← 追加（registration cannot resume）
```
- `activeReaderCount == 0` に加えて **`reader registration cannot resume`**（新しい reader が登録できない）を保証して初めて、EBR を bypass する ShutdownQuiescent に意味がある
- これは shutdown ordering（§1.2）の「全 Producer 停止 → join」で担保される

**🔴🔴🔴 十六次レビュー（2026-08-09）— readerRegistrationClosed を shutdown state machine に統合（INV-X3-4）**:
`readerRegistrationClosed` は単なる bool ではなく、**shutdown phase と同じ lifetime state machine に組み込む**:
```
INV-X3-4: ShutdownQuiescent reclaim is legal only after
          reader registration has been permanently closed for that shutdown.
```
**state machine**:
```
Running → StopAdmission → StopAudio → CloseReaderRegistration
  → activeReaderCount == 0 → JoinCoordinator → JoinBuilder
  → ShutdownQuiescent → Reclaim
```
- **CloseReaderRegistration** フェーズで `EpochDomain::registerReaderThread`（kMaxReaders=64 slot 確保）が以後失敗することを保証
- **INV-X3-4 のテスト**: `activeReaderCount = 0, readerRegistrationClosed = false` → reclaim forbidden / `readerRegistrationClosed = true` → reclaim allowed / **shutdown reclaim 開始後に registerReaderThread() が必ず失敗すること**

**🔴🔴🔴 十八次別視点6調査（2026-08-09）— CloseReaderRegistration の実コード挿入位置（2系統の shutdown シーケンスで確定）**:

実コード検証で、**shutdown には2系統のシーケンス**が存在する:
```
系統1: releaseResources（AudioEngine.Processing.ReleaseResources.cpp:34-404）
  StopAcceptingWork(:73) → requestShutdown(:75) → StopAudio(:115) → idle publish(:175)
  → StopWorkers(:188) → shutdownCoordinatorLoop(:189) + stopRebuildThread(:190)
  → ObserverDrained(:191) → ForceEpochAdvance(:194) → advanceRetireEpoch(:195)
  → RetireClosed/EpochSettled(:196-197) → DrainRetire(:199) → escalateAllRetires(:202)
  → Graceful Drain（:210-241, pendingRetireCount==0 && activeReaderCount==0 待ち）
  → drainDeferredRetireQueues(:307) → ReclaimComplete(:308) → EmergencyDrain(:315)
  → RetireQuarantineStore drainAllUnsafe(:376) → DSPQuarantine destroyForShutdown(:387)

系統2: ~AudioEngine（AudioEngine.CtorDtor.cpp:92-231・releaseResources 未実行の異常系用）
  StopAcceptingWork(:99) → requestShutdown(:102) → StopAudio(:105) → stopTimer(:106)
  → StopWorkers(:108) → shutdownCoordinatorLoop(:110) + stopRebuildThread(:111)
  → ForceEpochAdvance(:189) → publishEpoch(:190) → DrainRetire(:194)
  → Graceful Drain（:199-209, pendingRetireCount==0 && activeReaderCount==0 待ち）
  → clearPublishedRuntimeSnapshotsNonRt(:221) → drainDeferredRetireQueues(true)(:222)
  → m_epochDomain.drainAll(:223) → markShutdownComplete(:224)
```
**CloseReaderRegistration の挿入位置（確定）**:
```
系統1: releaseResources — DrainRetire フェーズ開始前（:194 ForceEpochAdvance の後）に closeReaderRegistration()
系統2: ~AudioEngine — DrainRetire フェーズ開始前（:189 ForceEpochAdvance の後）に closeReaderRegistration()
```
- **両系統とも graceful drain（activeReaderCount==0 待ち）の前に readerRegistrationClosed を確立**する
- **理由**: graceful drain は `activeReaderCount() == 0` を待つが、reader 登録を封じないと「0 に達した後に新しい reader が登録される」可能性がある。**readerRegistrationClosed を先に設定**すれば、graceful drain 完了時点で再登録が構造的に不可能
- **isQuiescent の評価タイミング**: graceful drain 完了（activeReaderCount==0）かつ readerRegistrationClosed == true の時点で ShutdownQuiescent reclaim が許可される

**🔴🔴🔴 追加調査（2026-08-09）— reader re-entry impossible の実装根拠（ISRRetireRouter API）**:
`ISRRetireRouter`（ISRRetireRouter.h:67-75）の reader 系 API:
```cpp
uint32_t activeReaderCount() const noexcept;          // :67
int readerCapacity() const noexcept;                  // :68
int registerReaderThread() noexcept;                  // :71（reader スレッド登録）
bool reserveReaderThread(int readerIndex) noexcept;   // :72（slot 予約）
void enterReader(int readerIndex) noexcept;           // :73（epoch 進入）
void exitReader(int readerIndex) noexcept;            // :74（epoch 退出）
uint64_t minReaderEpoch() const noexcept;             // :75（最小 reader epoch）
```
- **`reader re-entry impossible` の実装**: shutdown で `registerReaderThread()`（:71）が以後呼ばれないこと、または `reserveReaderThread`（:72）が失敗すること（capacity 満杯 or shutdown フラグ）を保証
- **`ShutdownContext::isQuiescent()` の実装**:
```
isQuiescent() :=
  ShutdownPhase >= AudioStopped
  AND activeReaderCount() == 0
  AND readerRegistrationClosed()   // registerReaderThread が不可（shutdown フラグ）
```

**🔴🔴🔴 追加調査（2026-08-09）— reader 登録の実体（EpochDomain / RCUReader）を確定**:
Reader 登録の実体は `EpochDomain`（core/EpochDomain.h）と `RCUReader`（core/RCUReader.h）:
```cpp
// core/EpochDomain.h:22 — reader スロット容量
static constexpr int kMaxReaders = 64;
// core/EpochDomain.h:45-65 — registerReaderThread: kMaxReaders の slot を CAS で確保
//   （失敗 = 全 slot が Reserved で埋まる or shutdown フラグ）

// core/RCUReader.h:65 — enter(): acquireThreadSlot() → registerReaderThread()
//   :171 — reservedTid = epochProvider->registerReaderThread()
// AudioEngine.h:4529 — audioThreadRcuReader { m_epochDomain }（Audio callback 用）
```
- **Audio callback の reader**: `audioThreadRcuReader`（AudioEngine.h:4529）が BlockDouble.cpp:151 の `RuntimeReaderContext audioCtx{audioThreadRcuReader, Audio}` で enter/exit される
- **readerRegistrationClosed の実装（🔴🔴🔴 実コード検証で確定）**: `EpochDomain` に shutdown フラグを追加し、`registerReaderThread` がフラグ設定後に `-1`（失敗）を返す。**「Audio Thread 停止後に kMaxReaders を全消費」案は NO-GO**（他の reader 登録を塞ぎ、slot 枯渇診断を壊す）:
```cpp
// core/EpochDomain.h — 追加メンバ
std::atomic<bool> registrationClosed_{false};   // ★ X3: CloseReaderRegistration フェーズで true

// core/EpochDomain.h:45-77 — registerReaderThread の冒頭にガード追加
int registerReaderThread(const char* tag) noexcept
{
    if (registrationClosed_.load(std::memory_order_acquire))
        return -1;                                  // ★ X3: reader registration permanently closed
    for (int i = 0; i < kMaxReaders; ++i) {
        // ... 既存の CAS slot 確保 ...
    }
    return -1;
}

// 新設 accessor
void closeReaderRegistration() noexcept { registrationClosed_.store(true, std::memory_order_release); }
[[nodiscard]] bool readerRegistrationClosed() const noexcept { return registrationClosed_.load(std::memory_order_acquire); }
```
- **🔴🔴🔴 十八次別視点7調査（2026-08-09）— ISRRetireRouter 経由の伝播を確定**: ISRRetireRouter（ISRRetireRouter.h:52-99）は `IEpochProvider& provider`（:56）を受け取り、`registerReaderThread()`（:71）が**内部で EpochDomain::registerReaderThread を委譲**する（.cpp で EpochDomain を include、:49-51）。したがって:
  - **EpochDomain::registrationClosed_ を設定すれば、ISRRetireRouter::registerReaderThread（:71）経由の登録も自動的に拒否される**（委譲先で -1 が返る）
  - **RCUReader::acquireThreadSlot → epochProvider->registerReaderThread()（RCUReader.h:171）も同じ経路**（EpochDomain か ISRRetireRouter のどちらを provider にしても registrationClosed_ が効く）
  - **X3 の closeReaderRegistration() は EpochDomain に追加すれば十分**（ISRRetireRouter / RCUReader 経由の全登録が自動的に封じられる）
  - **Audio Thread の audioThreadRcuReader（AudioEngine.h:4529）は EpochDomain を直接 provider に持つ**ため、EpochDomain へのフラグ追加で直接効く
- **🔴🔴🔴 十八次別視点10調査（2026-08-09）— ReaderSlot 構造と readerRegistrationClosed の関係を確定**: `ReaderSlot`（EpochDomain.h:531-547）は:
  ```
  struct ReaderSlot {
      std::atomic<uint64_t> epoch;                    // kInactiveEpoch = max
      std::atomic<uint32_t> depth;                    // ネスト深度（0 = 非アクティブ）
      std::atomic<uint64_t> enterCount;               // enter 回数（軽量カウント）
      std::atomic<uint64_t> residencyStartTimestampUs;// steady_clock 滞留開始
      std::atomic<uint64_t> ownerThreadId;            // thread::id ハッシュ
      char ownerTag[32];                              // "AudioThread" / "TimerThread" 等
      std::atomic<uint8_t> quarantineFlags;           // 0x01 quarantined / 0x02 pending
  };
  ```
  - **closeReaderRegistration() は新規登録のみを拒否**し、**既存の ReaderSlot は解放しない**（exitReader で epoch = kInactiveEpoch に戻るのみ）
  - **graceful drain は `activeReaderCount() == 0`（全 slot の depth == 0）を待つ**。readerRegistrationClosed 後も既存 slot の exit は継続可能
  - **quarantineFlags（0x01）の Reader は getMinReaderEpoch から除外**（:211-215 既確認）— detectStuckReaders が stuck Reader を quarantine すると reclaim が進む
  - **X3 の isQuiescent は「readerRegistrationClosed AND activeReaderCount == 0」**: 新規登録封鎖 + 既存 reader 全 exit で成立
- **🔴🔴🔴 十八次別視点13調査（2026-08-09）— quarantineReader / unquarantineAllReaders と X3 の関係を確定**:
  - **quarantineReader（:264-311）**: depth==0 で即座 quarantine（CAS で flags 0x00→0x01）/ depth>0 で pending（0x02）設定 → exitReader で 0x02→0x01 昇格。**quarantined Reader は getMinReaderEpoch から除外**（safe-epoch 計算から）
  - **unquarantineAllReaders（:313-321）**: 全 slot の quarantineFlags を 0 に戻す（shutdown 時）
  - **verifyReaderInvariants（:338-367）**: quarantined は epoch==kInactiveEpoch（:354-357）/ pending は depth>0（:360-363）/ quarantined と pending は同時成立しない（:366）
  - **X3 の shutdown での流れ**: detectStuckReaders が stuck Reader を quarantine → getMinReaderEpoch が進む → reclaim が進む → graceful drain 完了後 unquarantineAllReaders（ReleaseResources.cpp:367）で flags クリア
  - **X3 の readerRegistrationClosed は quarantine と独立**: quarantine は「既存 reader の stuck 解除」、registrationClosed は「新規登録封鎖」。両者は直交する
  - **INV-ISR-04 の整合**: ShutdownQuiescent reclaim は readerRegistrationClosed AND activeReaderCount==0 が前提。quarantined Reader は activeReaderCount から除外済み（depth==0 のため）
- **🔴🔴🔴 十八次別視点14調査（2026-08-10）— enterReader / exitReader の詳細を確定**:
  - **enterReader（:106-130）**: **epoch を depth++ より先に store（:115-116, BUG-050）** → depth++（:119-121, acq_rel）。**ネスト時（previousDepth > 0）は epoch 再設定なし（:122-123）**（epoch は active Reader を反映済み）。初回 enter 時に residencyStartTimestampUs 記録（:125-129）
  - **exitReader（:133-168）**: depth--（:141）→ 0 なら epoch = kInactiveEpoch（:157）→ **pending quarantine（0x02）昇格（:163-168, CAS で 0x02→0x01）**
  - **X3 の readerRegistrationClosed との関係**: enter/exit は registrationClosed の影響を受けない（**既存 Reader の enter/exit は shutdown 中も継続可能**）。graceful drain は activeReaderCount()==0（全 slot depth==0）を待つ
  - **X3 の isQuiescent**: readerRegistrationClosed（新規登録不可）+ activeReaderCount==0（既存 Reader 全 exit）で成立。**exit 後の pending quarantine 昇格は getMinReaderEpoch から除外**されるため reclaim が進む
- **🔴🔴🔴 十八次別視点12調査（2026-08-09）— DeferredDeletionQueue::reclaim の epoch 安全削除を確定**: `EpochDomain::tryReclaim`（:371-381）は `deferredDeletionQueue.reclaim(getMinReaderEpoch())` を呼ぶ。`DeferredDeletionQueue::reclaim`（DeferredDeletionQueue.h:108-119）は:
  ```
  reclaim(minReaderEpoch):
    deqPos = dequeuePos（acquire）
    scanPos = deqPos
    while (scanned < kMaxScan) {            // :118 kMaxScan=1024
        seq_atom = sequences[scanPos & kMask]
        // epoch < minReaderEpoch（isOlder）のエントリのみ deleter 実行
        if (entry.epoch isOlder(entry.epoch, minReaderEpoch)) {
            deleter(ptr) → 解放
            seq を次世代に更新（scanPos + kQueueSize）
            ++reclaimed
        } else {
            break;                           // FIFO 前提: 先頭が不安全なら以後も不安全
        }
    }
  ```
  - **reclaim は FIFO 前提**（先頭が epoch 不安全なら break、:119 相当）— 古い epoch の Retire が新しい epoch の後ろに残ることはない（enqueue 順で epoch 単調）
  - **`isOlder(a, b)`（:399-402）** = `static_cast<int64_t>(a - b) < 0`（wraparound 対応）
  - **X3 の reclaim(ReclaimMode::RuntimeEBR) はこの epoch 安全削除と整合**（reclaim は slot 状態遷移、物理削除は DeferredDeletionQueue::reclaim が epoch 安全に実行）
  - **ShutdownQuiescent では**: reader 停止済み（activeReaderCount==0 + readerRegistrationClosed）のため minReaderEpoch が最新 epoch に進み、全 Retire が安全判定される
- **`closeReaderRegistration()` は shutdown state machine の `CloseReaderRegistration` フェーズで呼ぶ**（AudioEngine の shutdown 制御側。§1.2 / A-3.1 の ShutdownPhase と連動）
- **🔴🔴🔴 十八次調査（2026-08-09）— 2つの ShutdownPhase enum の整合を確定**: プロジェクトには **2つの独立した ShutdownPhase enum** が存在する:
  ```
  AudioEngine::ShutdownPhase（AudioEngine.h:2521-2530）: Running / StopAcceptingWork / StopAudio / StopWorkers / ForceEpochAdvance / DrainRetire / Destroy
  ISRShutdown::ShutdownPhase（ISRShutdown.h:25-41）: Running / AudioStopped / ObserverDrained / RetireClosed / EpochSettled / ReclaimComplete / EmergencyDrain / VerifyDrained / TimedOut / Failed / ShutdownComplete
  ```
  - **CloseReaderRegistration は既存の列挙値のどちらにも存在しない**（新設）
  - **X3 の実装選択（確定）**: `closeReaderRegistration()` は **`AudioEngine::setShutdownPhase(ShutdownPhase::StopAudio, ...)` の完了時（Audio Thread 停止確認後）に呼ぶ**。これは AudioEngine 側の enum に新規値を追加せず、**既存フェーズ遷移の副作用として registrationClosed_ を set** する方法（enum 変更の影響を最小化）
  - **isQuiescent の precondition**: `ShutdownContext::isQuiescent()` は `AudioEngine::ShutdownPhase >= StopAudio AND activeReaderCount() == 0 AND readerRegistrationClosed()`（AudioEngine 側 enum を基準）
  - **ISRShutdown::ShutdownPhase との対応**: `StopAudio`（AudioEngine）≈ `AudioStopped`（ISR）。両 enum の対応表はテストで固定（REPAIR_PLAN2.md:1267 の既存指摘）
- **既存の登録済み reader への影響**: フラグは**新規登録のみ**を拒否。登録済み slot（audioThreadRcuReader / messageThreadRcuReader）は exit まで動作継続（既存 reader の epoch 安全性は維持）
- **`reserveReaderThread`（EpochDomain.h:79-103）にも同様のガードを追加**（将来 reserve 経由の登録も封じる）:
```cpp
bool reserveReaderThread(int readerIndex) noexcept override
{
    if (registrationClosed_.load(std::memory_order_acquire))
        return false;
    // ... 既存の CAS slot 予約 ...
}
```
- **isQuiescent の readerRegistrationClosed()**: `m_epochDomain` が reader 登録を拒否する状態（shutdown フラグ）を確認
    setReclaimInFlightCount(0);
    return true;
}
```

**呼び出し元の対応**:
| 既存呼び出し元 | 変更 |
|---------------|------|
| `AudioEngine.h:4248` / `AudioEngine.Retire.cpp:83` | `requestReclaim(handle, rt, router)` → `reclaim(ReclaimMode::RuntimeEBR, handle, rt, router)` |
| `AudioEngine.h:2027`（CacheMap::dtor） | 既存 `shutdownPhase >= Destroy` チェックに加えて `reclaim(ReclaimMode::ShutdownQuiescent, ...)`。phase assertion は `shutdownPhase >= Destroy` が満たす |
| `ReleaseResources.cpp:415,420` | `dspHandleRuntime_.shutdownReclaim(handle)` → `reclaim(ReclaimMode::ShutdownQuiescent, ...)`（releaseResources の VerifyDrained 段階は phase 保証済み） |

**`shutdownReclaim`（ISRDSPHandle.h:171）の扱い**: bypass API として**廃止**。`reclaim` は private（:188）で `friend class RuntimePublicationCoordinator` のみアクセス可のため、**Coordinator 経由の `reclaim(ReclaimMode, ...)` に一本化**。

### ★ R4 詳細設計（2026-08-10・外部レビュー反映 — X3 実装時の実施順序として固定）

R4 は X3 と同じ内容。**`shutdownReclaim()` API を消すのではなく、「reclaim authority と shutdown safety proof を一本化」**する（単なる API 置換ではない）。最終目標は 4 者分離:
```
ReclaimAuthority   ← safety policy / shutdown phase validation / pending handling / reclaim orchestration
  ├── RuntimeEBR        （epoch safety → unsafe → pendingReclaimHandles_）
  └── ShutdownQuiescent （quiescence proof → safe → Reclaim）
DSPHandleRuntime   ← physical slot/state operation のみ（Retired → Reclaimed の primitive は private/internal API）
RetireRouter       ← EBR / retire / quarantine
ShutdownRuntime    ← ShutdownQuiescenceProof
```

**Phase 0-7（実装順序）**:
- **Phase 0**: 現状の安全性契約を固定（コード変更なし）。RuntimeEBR = `retire → epoch check → safe? No → pendingReclaimHandles_ / Yes → reclaim`。ShutdownQuiescent = `AdmissionClosed + all producers stopped + Coordinator stopped + Builder stopped + Audio reader stopped + reader registration closed + epoch settled`
- **Phase 1**: `shutdownReclaim()` の中身を Authority へ移す（`ReclaimMode{ RuntimeEBR, ShutdownQuiescent }` / `ReclaimRequest{ handle, mode }` 導入、`reclaim(ReclaimRequest)` を唯一の entry point に）
- **Phase 2**: RuntimeEBR を先に一本化（`requestReclaim` / `requestReclaimHandle` / `drainDeferredRetireQueues` → `reclaim(RuntimeEBR)`）。**`pendingReclaimHandles_` 再試行は維持**（slot leak 防止の核心）
- **Phase 3**: ShutdownQuiescent を実装。**`ShutdownQuiescenceProof` を独立オブジェクト化**（admissionClosed / producersJoined / coordinatorStopped / builderStopped / audioStopped / readerRegistrationClosed / readersZero / epochSettled + `valid()`）。boolean 直渡し禁止（`true` が何を意味するか不明瞭）→ `ReclaimPermit`（caller cannot manufacture）
- **Phase 4**: `releaseResources` 移行（retire は VerifyDrained で実行 → **reclaim(ShutdownQuiescent) は quiescence 確立後（waitForDrain 相当の後）に移動** — 外部レビュー重要問題 1 / INV-R4-15）。retire と reclaim は別操作に維持（Retired ≠ Reclaimed）。
  現行: `VerifyDrained(:407) → retire + shutdownReclaim(:412-421) → waitForDrain(:447)`（shutdownReclaim が waitForDrain より前 — R4 と矛盾）
  変更後: `VerifyDrained(:407) → retire(:412-421) → stop/join 全 worker → close reader registration → readers==0 → drain RuntimeEBR pending → ShutdownQuiescenceProof 取得 → reclaim(ShutdownQuiescent, permit)（waitForDrain 相当の後）`
- **Phase 5**: CacheMap destructor 移行（`owner.reclaimCacheEntry(handle)` — handle ownership release 通知）。**delete 順序（delete EQCoeffCache → reclaim slot）は変更しない**
- **Phase 6**: `shutdownReclaim()` を deprecated 化（call site=0 確認。この段階では API 削除しない — external/hidden call site 検出のため）
- **Phase 7**: `DSPHandleRuntime::shutdownReclaim()` 完全削除

**NG（やってはいけないこと）**:
- NG1: 名前だけ ReclaimAuthority にする（authority duality 残存）
- NG2: shutdown なら無条件 reclaim（EBR safety violation）
- NG3: `pendingReclaimHandles_` を shutdown 時に無条件 clear（slot leak / lifecycle inconsistency）
- NG4: `activeReaderCount()==0` だけで shutdown reclaim（不十分 — Audio Thread 停止だけでは当該 handle の reclaim 安全性の十分条件ではない）
- NG5: Faulted になったら pending handle を破棄（**絶対にしない**。Faulted ≠ memory safe）

**実コード照合（2026-08-10）**:
- `shutdownReclaim`（ISRDSPHandle.h:171）= `reclaim(handle)` のラッパー（bypass authority）
- 呼び出し元: CacheMap::~CacheMap（AudioEngine.h:2015-2027: resolve→delete→shutdownReclaim）/ releaseResources（ReleaseResources.cpp:415 active, :420 fading）
- 通常系: retireDSPHandleForRuntime → requestReclaimHandle（AudioEngine.h:4248）→ requestReclaim → epoch check → reclaim/pending — 二重 authority

**テスト（R4-T1〜T7）**: T1 RuntimeEBR（reader active → retire → reclaim(false) → pending → reader exit → drain → reclaim）/ T2 Shutdown happy path（全 proof 成立 → reclaim 成功）/ T3 Shutdown premature（proof incomplete → Faulted + non-Reclaimed）/ T4 stale reader（activeReaderCount>0 → proof 生成不可）/ T5 pending handle（shutdown 移行でも pending を silently clear しない）/ T6 CacheMap（delete→reclaim 順序固定）/ T7 API architectural test（`shutdownReclaim(` が 0・`DSPHandleRuntime::reclaim(` の caller が ReclaimAuthority のみ）

**Acceptance Criteria（AC-R4-1〜10）**: AC-R4-1 shutdownReclaim() call sites==0 / AC-R4-2 symbol absent / AC-R4-3 DSPHandleRuntime::reclaim() の production caller が ReclaimAuthority のみ / AC-R4-4 両 mode で same physical reclaim primitive / AC-R4-5 ShutdownQuiescent に quiescence proof 必須 / AC-R4-6 epoch unsafe な RuntimeEBR handle は pendingReclaimHandles_ に残る / AC-R4-7 Faulted で pending を clear しない / AC-R4-8 Audio Thread から reclaim authority を呼ばない / AC-R4-9 reclaim 自体は RT thread で実行しない / AC-R4-10 isFullyDrained() は reclaim authority にならない

**実装順序 R4-0〜R4-12**: R4-0 現状 call graph + safety proof 固定 → R4-1 ReclaimMode/ReclaimRequest 導入 → R4-2 RuntimeEBR 移動 → R4-3 pendingReclaimHandles_ 再試行接続 → R4-4 ShutdownQuiescenceProof 導入 → R4-5 releaseResources 移行 → R4-6 CacheMap 移行 → R4-7 deprecated → R4-8 call-site=0 → R4-9 API 削除 → R4-10 architectural test → R4-11 Debug/Release/CTest → R4-12 shutdown/soak。**R4-5 以前には shutdownReclaim() を削除しない**。

> **★ 重要な相互作用（外部レビュー §25 追記）**: `pendingReclaimHandles_` / `isRetired()` / `RetireQuarantineStore` との相互作用を先に固定しないと、現在解消済みの slot leak / quarantine race を再導入する可能性がある。R4 は X3 実装時に上記の順序で実施するのが安全。

**🔴🔴🔴 外部レビュー反映（2026-08-10）— ShutdownQuiescent 入口条件を Proof ベースに厳密化（最も重要な修正）**:

`shutdownPhase >= Destroy` は **ShutdownQuiescent reclaim の安全性証明にならない**（Destroy phase 到達 ≠ 当該 handle の reclaim 安全性）。Phase assertion ではなく **Proof を渡す**設計に変更:

```cpp
// ❌ 不十分: shutdownPhase >= Destroy は proof の代替にならない（現行 CacheMap::~CacheMap の実装）
if (shutdownPhase >= Destroy) { delete; shutdownReclaim(handle); }

// ✅ 正しい: ShutdownRuntime だけが生成する ReclaimPermit を渡す（caller cannot manufacture）
auto permit = shutdownRuntime.acquireShutdownReclaimPermit();
if (!permit.valid()) return false;
reclaim(ReclaimMode::ShutdownQuiescent, handle, permit);
```

- **`ShutdownReclaimPermit` は `ShutdownRuntime` のみが生成**（friend class で caller cannot manufacture — AC-X3-11）。CacheMap / ReleaseResources 自身は proof を作らない（Authority Singularization）
- **ShutdownQuiescent の完全条件（Phase 0 の厳密化）**: `admissionClosed + producersJoined + coordinatorStopped + builderStopped + audioStopped + readerRegistrationClosed + activeReaderCount==0 + pendingReclaimHandles.empty() + reclaimInFlight==0 + epochSettled`。**`pendingReclaimHandles_.empty()` は all reclaim producers joined 後にのみ評価**（AC-X3-14 — 観測直後の push を防ぐため、quiescence 後に empty を評価）
- **証明可能な一方向の shutdown protocol**:

```text
ShutdownRuntime → close admission → stop producers → join producers → close reader registration
→ stop audio/builder/coordinator → activeReaderCount==0 → pendingReclaimHandles.empty()
→ reclaimInFlight==0 → create immutable ReclaimPermit → reclaim(ShutdownQuiescent, permit)
```
- **実コード照合（2026-08-10）**: `CacheMap::~CacheMap`（AudioEngine.h:2015-2032）は `shutdownPhase >= Destroy` で `delete EQCoeffCache + shutdownReclaim(handle)` を実行（レビュー指摘どおり不十分）。**R4 Phase 5 で `owner.reclaimCacheEntry(handle)` + ShutdownRuntime の Permit 取得に置換**（CacheMap 自身は shutdown policy を持たない — AC-X3-12）
- **retire の idempotency（AC-X3-16）**: `ReleaseResources: retire(handle) → reclaim(ShutdownQuiescent)` の際、R4 の `reclaim()` が内部で `retire()` を実行する場合の二重 retire を防ぐ。`retire(handle) × 2` が double enqueue / double ownership transition を起こさないことを unit test で固定。`retire → reclaim(RuntimeEBR)` と `retire → reclaim(ShutdownQuiescent)` の両方を検証
- **physical DSPCore destruction の独立性（AC-X3-18）**: R4 の handle reclaim（slot state transition + free-list return）は DSPCore の物理 delete とは独立（物理 delete は Retire/Epoch path）。両者を同一概念にしない

**追加 Acceptance Criteria（AC-X3-11〜18）**:

- AC-X3-11: ShutdownReclaimPermit は ShutdownRuntime 以外生成不可
- AC-X3-12: shutdownPhase >= Destroy だけでは ShutdownQuiescent reclaim を許可しない
- AC-X3-13: ShutdownQuiescent reclaim の直前に readerRegistrationClosed == true を証明
- AC-X3-14: pendingReclaimHandles_.empty() は all reclaim producers joined 後にのみ評価
- AC-X3-15: registerReaderThread() は CloseReaderRegistration 後に必ず失敗
- AC-X3-16: retire(handle) → reclaim(handle) による二重 retire が double enqueue を起こさない
- AC-X3-17: ShutdownQuiescent reclaim は Audio Thread から呼べない
- AC-X3-18: physical DSPCore destruction は R4 の handle reclaim とは独立して Retire/Epoch path を通る

**🔴🔴🔴 外部レビュー反映（2026-08-10）— releaseResources の順序再構成・Proof 循環解消・epochSettled 定義固定（最重要 3 問題）**:

**実コード照合（releaseResources の現行順序）**:
```text
VerifyDrained (:407)
  ↓
retire(active) + shutdownReclaim(active) (:412-416)
retire(fading) + shutdownReclaim(fading) (:417-421)
  ↓
waitForDrain (:447)
```
現行は **shutdownReclaim を waitForDrain より前**に実行しており、R4 の Proof 設計（quiescence 確立後に reclaim）と**矛盾**する。

**重要問題 1 — releaseResources の順序再構成（必須修正）**: reclaim(ShutdownQuiescent) を quiescence 確立後に移動する:
```text
retire(active) / retire(fading)      ← VerifyDrained（現行どおり。retire は独立）
  ↓ stop/join all producers → stop/join coordinator → stop/join builder
  ↓ close reader registration → readers == 0 → drain RuntimeEBR pending
  ↓ construct ShutdownQuiescenceProof
  ↓ reclaim(ShutdownQuiescent, permit)   ← quiescence 確立後（waitForDrain 相当の後）
```
**INV-R4-15**: releaseResources の shutdown reclaim は waitForDrain / producer quiescence より前に実行しない。

**重要問題 2 — pending/Proof の循環解消**: `pendingReclaimHandles_.empty()` を Proof 生成条件にするが、**Proof 取得後に新規 reclaim producer が存在しないこと**を保証する（INV-R4-12/13）:
```text
ShutdownQuiescenceProof is constructed only after:
  AdmissionClosed AND ProducersJoined AND CoordinatorStopped AND BuilderStopped
  AND AudioStopped AND ReaderRegistrationClosed AND ReadersZero
  AND RuntimeEBRPending == empty AND ReclaimInFlight == 0
```
Proof 取得後、新しい reclaim producer は存在しない（INV-R4-13）。

**重要問題 3 — epochSettled の定義固定**: `currentEpoch == minReaderEpoch` のような epoch counter 比較にしない。epochSettled は **lifetime condition** として定義（INV-R4-16）:
```text
epochSettled = no active reader AND no reader registration possible AND no pending epoch-protected reclaim
```

**追加 Invariant（INV-R4-11〜17）**:
- INV-R4-11: ShutdownReclaimPermit は ShutdownRuntime だけが生成できる
- INV-R4-12: ShutdownQuiescenceProof は RuntimeEBR pending が空になるまで生成されない
- INV-R4-13: Proof 生成後、新規 reclaim producer は存在しない
- INV-R4-14: shutdownPhase >= Destroy だけでは ShutdownQuiescent reclaim を許可しない
- INV-R4-15: releaseResources の shutdown reclaim は waitForDrain / producer quiescence より前に実行しない
- INV-R4-16: epochSettled は epoch counter 値だけではなく、reader registration closure + readers zero + pending reclaim drain を包含する
- INV-R4-17: Faulted は memory-safe / reclaim-safe を意味しない（NG5 と整合）

**最終 R4 プロトコル（推奨・外部レビュー §30）**:
```text
ShutdownRuntime → StopAcceptingWork → Close Admission → Stop Coordinator → Stop Builder
→ Stop Audio Reader → Close Reader Registration → readers == 0 → Drain RuntimeEBR Pending
→ pendingReclaimHandles == 0 → reclaimInFlight == 0 → ShutdownQuiescenceProof
→ ReclaimAuthority（active/fading handle）→ DSPHandleRuntime::reclaim → Reclaimed
```
**Retire は前段階として独立**: `Current/Fading → Retire → RuntimeEBR pending → Quiescent → Reclaim`（Retired ≠ Reclaimed — 分離維持）。

**🔴🔴🔴 外部レビュー反映（2026-08-10）— ReclaimProducersClosed state + R4 atomic state transition（実装注意の確定）**:

**`ReclaimProducersClosed` を追加**（INV-R4-13 を runtime state として表現）: Proof 生成前に「reclaim producer が閉じた」ことを **state として固定**する。`pendingReclaimHandles_.empty()` の観測直後に新 handle が pending へ追加される race を、**単なるコメントではなく shutdown state machine に組み込んで**排除する。

**R4 の atomic state transition（推奨・外部レビュー §20）**:
```text
Running → StopAccepting → AdmissionClosed → CoordinatorStopping → CoordinatorStopped
→ BuilderStopped → AudioStopped → ReaderRegistrationClosed → ReadersZero
→ ReclaimProducersClosed → DrainRuntimeEBR → PendingReclaim == 0
→ ReclaimInFlight == 0 → ShutdownQuiescenceProof → ShutdownReclaim → Destroyed
```

**releaseResources は単なる API 置換ではなく制御フロー改修**（外部レビュー §21・Phase 4 の本質）:
```text
現行: VerifyDrained → retire → shutdownReclaim → UI release → shutdown clear → waitForDrain
変更後: VerifyDrained → retire → UI/worker shutdown → Coordinator join → Builder join
      → reader registration close → reader drain → RuntimeEBR pending drain
      → Proof → ShutdownQuiescent reclaim
```

**実装前に絶対固定すべき 4 invariant（外部レビュー最終評価）**:
- INV-R4-12: Proof 生成前に pending reclaim を空にする
- INV-R4-13: Proof 生成後、新規 reclaim producer は存在しない（**ReclaimProducersClosed で実装**）
- INV-R4-15: waitForDrain / producer quiescence 前に shutdown reclaim しない
- INV-R4-16: epochSettled は epoch counter 値ではなく reader registration closure + readers zero + pending drain

**外部レビュー反映（2026-08-10）— 追加 Acceptance Criteria（AC-1〜AC-3）**:

**AC-1 — Shutdown proof の monotonicity（irreversible state transition）**:
`ShutdownQuiescenceProof.valid() == true` になった後、`readerRegistrationClosed` / `producersClosed` / `readersZero` などが **false に戻ることがない**こと。Proof は単なる snapshot ではなく、**irreversible shutdown state transition の証拠**として扱う（state machine の monotonicity を固定）。

**AC-2 — Reclaim 後の physical slot identity（lifetime correctness）**:
`Retired → Reclaimable → Reclaimed` 後に、**same slot が再利用されても old RuntimeWorld からアクセスできない**ことを architectural test で固定（slot reuse 後の lifetime 分離 — INV-X4-6 / INV-X4-8 と整合）。

**AC-3 — Publish / Completion / Reclaim の cross-domain test**:
`Publish(seq=N) → Commit(N) → Complete(N) → Retire(old) → Reclaim(old)` を**単一の test scenario として追跡**する。個別 test（X2 OK / X3 OK / X4 OK）だけでは **X2→X3 の境界**で壊れる可能性があるため、cross-domain の一貫性を検証（§6.8 の統合テストに追加）。

**AC-R4-Y — Proof 後の producer 禁止テスト（ReclaimProducersClosed の実装検証）**:
`ShutdownQuiescenceProof.valid() == true` 後に `requestReclaim()`（または `reclaim(ReclaimMode::RuntimeEBR, ...)`）が**成功したらテスト失敗**。これは **ReclaimProducersClosed を実装上も検証**するテスト（INV-R4-13 の直接検証）。Proof 生成後は新規 reclaim producer が存在しないことを、実行時テストで固定する（単なるコメントではなく shutdown state machine + 実行時テストで検証）。

### 6.4 X4 — RuntimePublicationCoordinator authority 二重化（P1 優先・構造的に最重要）

> **★ 実コード照合確定（2026-08-10）**: read API（observePublishedWorld / acquireReadToken / consumeWorldHandle）は **RuntimeWorldAuthority.h に未実装**（X4-B-9 で新設予定）。現行 read path は `RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore)` を直接呼ぶ（AudioEngine.h:1331/2119/3116/3383/3691 等）。publishAndSwap は RuntimeWorldAuthority-owned WriteAccess のみ（INV-X4-3）。X4-A（rename）/ X4-B（ownership topology）の二段階実装（X4-0〜X4-11）。**★ 2026-08-11 実装済み: X4-B 完了（read API は worldAuthority_ 経由に移行済み）。**
> **★ 別視点調査追記（2026-08-10）— X4-B の実装対象を実コード確認**: `using RuntimePublicationCoordinator`（AudioEngine.h:3509）/ `using RuntimePublishStore`（:3544）/ `makeRuntimePublicationCoordinator()`（:3646 inline factory）/ `RuntimePublishExecutor.h:55`（`auto coordinator = ctx.engine.makeRuntimePublicationCoordinator()` 一時生成 → `publishWorld` — X4-B で削除）。X4-A の rename 対象（:3509 エイリアス + 7 ファイルの型参照）と X4-B-9 の read API 置換対象（:2702 一覧）の実コード裏付け。
> **★ 別視点調査追記2（2026-08-10）— WriteAccess の実コード照合**: `RuntimeStore::WriteAccess`（RuntimeStore.h:18）/ `publishAndSwap`（:35）/ `acquireWriteAccess`（:83）を確認。**INV-X4-3**（publishAndSwap は RuntimeWorldAuthority-owned WriteAccess のみ）の実装対象を裏付け。X4-B で WriteAccess を RuntimeWorldAuthority が move-only 保持（core Coordinator は publishWorld の一時生成をやめる — :3646 / RuntimePublishExecutor.h:55 削除）。
> **★ 別視点調査追記3（2026-08-10）— authority topology test の既存実装を確認**: `tools/publication_authority_verifier.py` が存在し、`RuntimeStore::publishAndSwap bypass detected` を検出（**INV-X4-3 の architectural test が既にツール化済み**）。`ALLOWED_PUBLISH_AND_SWAP_FILES` で許可ファイルを管理。X4-B 実装時はこの verifier を拡張・維持する（Test 3 / AC-X4-3 の静的検査の基盤）。

**現状**: `convo::isr::RuntimePublicationCoordinator`（Intent authority）と `convo::RuntimePublicationCoordinator`（Publish authority, core）が同名共存。型名だけではどちらが本当の Coordinator か分からない。

**🔴 設計方針**: クラス統合は **NO-GO**（責務が異なる）。**Authority を明示的に命名・分離**:
```
convo::isr::RuntimeIntentCoordinator（旧 convo::isr::RuntimePublicationCoordinator）
convo::RuntimePublishAuthority    （旧 convo::RuntimePublicationCoordinator, core）
```

**Authority matrix（明文化）**:
| 操作 | 唯一の Authority |
|------|----------------|
| Intent enqueue / dispatch | IntentCoordinator |
| Admission | PublicationAdmission |
| RuntimeWorld commit / RuntimeStore swap | RuntimeWorldAuthority |
| Publish completion | PublishCompletionAuthority |
| Retire | LifetimeState / Retire Authority |
| Reclaim | Reclaim Authority |
| Shutdown phase | ShutdownRuntime |

**禁止**: Coordinator A → commit() と Coordinator B → commit() の二重 execution path。`PublishExecutor`（RuntimePublishExecutor.h:19-20）が sole gateway の状態を維持し、`IntentCoordinator → PublishIntent → PublishExecutor → RuntimeWorldAuthority → RuntimeStore` を一本に。

**🔴🔴 十二次レビュー（2026-08-09）— 影響範囲（型エイリアスと使用箇所）を確定**:
| 対象 | 現状 | X4 での変更 |
|------|------|------------|
| `AudioEngine.h:3509` | `using RuntimePublicationCoordinator = convo::RuntimePublicationCoordinator<RuntimePublishWorld, DSPCore*, RuntimePublicationBridge>` | `using RuntimePublishAuthority = convo::RuntimePublicationCoordinator<...>` に改名（Publish authority） |
| `convo::isr::RuntimePublicationCoordinator` | 7ファイルで使用（ConvolverProcessor.h / EQProcessor.h / SafeStateSwapper.h / AudioEngine.h / AudioEngine.Publication.cpp / テスト2件） | `convo::isr::RuntimeIntentCoordinator` に改名（Intent authority） |
| `makeRuntimePublicationCoordinator()`（AudioEngine.h:3646） | Publish authority の一時生成 | `makeRuntimePublishAuthority()` に改名 |

- **X4 は大規模リネームを伴う**ため、P2 完了後の独立タスク（§6.9 のとおり X4 は4番目）として実施。リネームは `serena` / `coccinelle` 相当の reference-aware ツールで一括適用
- **P2 では X4 を実施しない**（§1.1-1.4 の対象は `convo::isr::` 側のみで、core 側は publishWorld のみ）。但し、**コメントで Authority matrix の明文化**は P2 でも推奨（将来の誤配線防止）

**🔴🔴🔴 十一次/十二次レビュー追記 — core Coordinator の使用箇所（PublishExecutor 内部）を確定**:
`PublishExecutor::executePublish`（RuntimePublishExecutor.h:19-90）内で、core `convo::RuntimePublicationCoordinator` は **一時生成されて publishWorld のみに使用**される:
```cpp
// RuntimePublishExecutor.h:63-66
if (owner) {
    auto coordinator = ctx.engine.makeRuntimePublicationCoordinator();  // core convo::RuntimePublicationCoordinator
    (void)coordinator.publishWorld(std::move(owner));                  // sole RuntimeStore store-swap
}
```
- **X4 の `RuntimePublishAuthority` 改名対象**: この `makeRuntimePublicationCoordinator()`（AudioEngine.h:3646 定義 / RuntimePublishExecutor.h:63 使用）と、AudioEngine.h:3509 の using エイリアス
- **`convo::isr::RuntimeIntentCoordinator` 改名対象**: 7ファイル（ConvolverProcessor.h / EQProcessor.h / SafeStateSwapper.h / AudioEngine.h / AudioEngine.Publication.cpp / ISRSoakTests.cpp / ISRSemanticValidationTests.cpp）
- **Authority Singularization の現状**: `PublishExecutor`（sole gateway）→ `RuntimeWorldAuthority::commit` → core `RuntimePublishAuthority::publishWorld` が唯一の publish execution path。X4 はこれを**名前で明示**する（クラス統合はしない）

**🔴🔴🔴 十四次レビュー（2026-08-09）— 命名だけでなく RuntimeStore write authority の実体を一本化**:
`RuntimeStore<World, Owner>`（core/RuntimeStore.h:12-83）は `friend Owner`（:81）のみが `acquireWriteAccess()`（:83）を呼べ、`WriteAccess` は move-only（:21-28）で `publishAndSwap`（:35）。**Owner が WriteAccess の取得を一意に制御**する:
```cpp
// core/RuntimePublicationCoordinator.h:34-51
using Store = RuntimeStore<World, RuntimePublicationCoordinator<World, Handle, Bridge>>;  // Owner = 自身
static create(...) { ... store.acquireWriteAccess() ... }   // friend Owner で取得
```
- **現状**: RuntimeStore の Owner は core Coordinator 自身で、WriteAccess は `create` 時に1回取得。**これは既に write authority が一本化されている**
- **X4 での実体一本化**: `RuntimeWorldAuthority` を**本当の publication authority surface** にする（十四次推奨）:

**🔴🔴🔴 追加調査（2026-08-09）— RuntimeWorldAuthority の commit 委譲構造を確定**:
`RuntimeWorldAuthority`（RuntimeWorldAuthority.h:78-123）は **`coordinator_`（core `RuntimePublicationCoordinator&`）への参照**を保持し、`commit` を委譲する:
```cpp
// RuntimeWorldAuthority.h:81-123（現行）
class RuntimeWorldAuthority {
    explicit RuntimeWorldAuthority(RuntimePublicationCoordinator& coordinator) noexcept
        : coordinator_(coordinator) {}
    void commit(PublishAuthority auth, RuntimeBoundary boundary, const void* newWorld,
                std::uint64_t version) noexcept {
        coordinator_.commit(auth, boundary, newWorld, version);   // :112 委譲
    }
    // ...
    RuntimePublicationCoordinator& coordinator_;   // core Coordinator への参照
};
```

**X4 の最終形（RuntimeWorldAuthority を write authority にする）の実装詳細**:
1. **`coordinator_` 参照を `RuntimeStore::WriteAccess` に置き換え**（大規模リファクタ）:
```cpp
class RuntimeWorldAuthority {
    // 現行: RuntimePublicationCoordinator& coordinator_（commit 委譲 + read source）
    // X4 後: RuntimeStore<World, RuntimeWorldAuthority>::WriteAccess writeAccess_;
    //        （RuntimeStore の Owner を core Coordinator から RuntimeWorldAuthority に変更）
    //        → commit は writeAccess_.publishAndSwap(newWorld) を直接実行
};
```
2. **RuntimeStore の Owner 変更**: `RuntimeStore<World, RuntimePublicationCoordinator<...>>` → `RuntimeStore<World, RuntimeWorldAuthority>`（template パラメータ変更）
3. **`RuntimeWorldAuthority` が Owner になると**: `acquireWriteAccess()`（friend Owner）を RuntimeWorldAuthority が呼び、WriteAccess を move-only で保持。core Coordinator は publishWorld の一時生成をやめる
4. **影響範囲**: `RuntimeWorldAuthority` / `core/RuntimeStore.h` / `core/RuntimePublicationCoordinator.h` / `RuntimePublishExecutor.h`（:63-66 の一時生成を削除）

**⚠️ 注意**: Owner 変更は `RuntimeStore<World, Owner>` の template パラメータと friend 関係の変更を伴う**大規模リファクタ**。P2 後・X4 の独立タスクとして実施。**名前変更のみ（RuntimePublishAuthority）でも write authority 一本化は満たす**（十四次も命名整理は GO）。

**🔴🔴🔴 十七次レビュー（2026-08-09）— Physical write owner と Architectural authority surface を区別（必須）**:

「現状でも write authority は一本化されている」という評価は**狭義には正しい** — `RuntimeStore` の物理 WriteAccess を持つ Owner は core Coordinator 一つだから。しかし**architectural authority surface が一本化されているとは言えない**:

```
Physical write owner:      RuntimeStore の WriteAccess を物理的に持つ者
                           = core Coordinator（現在）→ RuntimeWorldAuthority（X4 後）
Architectural authority surface: 外部から見た publish の権威点
                           = RuntimeWorldAuthority（commit の入口）だが、
                             実際の Store write は core Coordinator に委譲中
```
- **この2つを区別しないと X4 の必要性が曖昧になる**（十七次 §7）
- **X4 の本質**: Physical write owner を Architectural authority surface（RuntimeWorldAuthority）に一致させる
- **現在**: Physical = core Coordinator / Architectural = RuntimeWorldAuthority（乖離）
- **X4 後**: Physical = RuntimeWorldAuthority / Architectural = RuntimeWorldAuthority（一致）

**🔴🔴🔴 十六次レビュー（2026-08-09）— X4 は現状 NO-GO。rename だけでなく ownership topology の再設計が必要（INV-X4-3）**:

`WriteAccess` は特定の `RuntimeStore` インスタンスへのアクセス権。**`RuntimeWorldAuthority` が WriteAccess を持つだけでは不十分** — **`RuntimeStore` そのものがどこに存在するか**を変更しなければならない:
```
現状: core RuntimePublicationCoordinator
        └── RuntimeStore
             └── WriteAccess

X4 最終形: RuntimeWorldAuthority
        ├── RuntimeStore
        │    └── WriteAccess
        ├── OwnerChannel
        └── LifetimeState
```
**物理所有関係まで移動させる**（Store の Owner を core Coordinator から RuntimeWorldAuthority に変更）。最終 invariant:
```
INV-X4-3: RuntimeStore::publishAndSwap() is reachable through exactly one
          RuntimeWorldAuthority-owned WriteAccess.
```
- **source-level architectural test**: `publishAndSwap` の caller を静的検査し、RuntimeWorldAuthority の WriteAccess のみから到達すること + `PublishExecutor` 以外から commit が発生しないこと（INV-X2-6 と併せて固定）
- **`RuntimePublishAuthority` は必要なら内部実装名として残してよい**が、`RuntimeWorldAuthority` 以外から `RuntimeStore::publishAndSwap()` に到達できないことを最終 invariant にする
RuntimeIntentCoordinator → PublicationAdmission → PublishExecutor
    → RuntimeWorldAuthority（authority surface）
        ├── RuntimeStore::WriteAccess（Owner = RuntimeWorldAuthority に変更）
        └── LifetimeState
```
- **注意**: Owner 変更は `RuntimeStore<World, Owner>` の template パラメータ変更を伴う大規模リファクタ。**P2 後・X4 の独立タスクとして実施**。現状の「core Coordinator が Owner」は write authority 一本化を満たしており、**名前変更のみでも意味はある**（十四次も命名整理は GO）

---

## 6.4-X4 詳細改修計画（十七次レビュー追加調査・X4-A/X4-B 分離）（2026-08-09）

**🔴🔴🔴 十七次レビュー（2026-08-09）— X4 は「命名の二重化を解消するだけ」の旧案は採用しない。最新レビューで INV-X4-3 が追加され、`RuntimeStore` の物理的 ownership topology まで変更する改修として設計固定する。**

### 実コード検証結果（2026-08-09・十七次調査で確定）

X4 計画の前提を実コードで検証した結果、**現状の authority topology は「二重の write surface」**であることが確定:

```
write surface #1: ISR commit()   → currentWorld_ の atomic 更新（メタデータのみ）
                    ISRRuntimePublicationCoordinator.cpp:80-115（publishAtomic(currentWorld_, newWorld)）
write surface #2: core publishWorld() → RuntimeStore::WriteAccess::publishAndSwap()
                    RuntimePublishExecutor.h:53-57（一時生成 coordinator → publishWorld）
```

**現状の publish execution path（唯一の store-swap は #2）**:
```
PublishExecutor::executePublish（RuntimePublishExecutor.h:20-85）
  ├─ ownerChannel().take()                # Owner 取得（:30-33）
  ├─ authority.commit(...)                # write surface #1（ISR メタデータ commit）:42-48
  ├─ makeRuntimePublicationCoordinator()  # 一時生成（:55）
  ├─ coordinator.publishWorld(owner)      # write surface #2（唯一の Store swap）:56
  ├─ authority.registry().unregister()    # :59
  └─ 以降: onPublishCompleted / advanceRetireEpoch / onPublishCommitted（:74-84）
```
- **ISR `commit()` は `RuntimeStore` を触らない**。`currentWorld_`（ISRRuntimePublicationCoordinator.h:380）の atomic を publish するだけ。X4 計画 §14「commit ≠ publishAndSwap」は実コードと一致
- **実際の Store swap は一時生成 coordinator のみ** — `RuntimePublishExecutor.h:55-56` の `ctx.engine.makeRuntimePublicationCoordinator()` → `publishWorld()`

**publishWorld の直接呼び出し元（X4-B で全て authority 経由に一本化）**:
| 箇所 | 用途 |
|------|------|
| `RuntimePublishExecutor.h:55-56` | PublishIntent 実行（**唯一の runtime store-swap**） |
| `AudioEngine.Init.cpp:53-54` | **Bootstrap World** の同期 publish（CoordinatorLoop 起動前） |
| `AudioEngine.Processing.ReleaseResources.cpp:436-438` | **shutdown clear**（requestShutdownClearNonRt + clearPublishedRuntimeSnapshotsNonRt） |

**Store の物理所有（AudioEngine が保持）**:
- `AudioEngine.h:3509-3511`: `using RuntimePublicationCoordinator = convo::RuntimePublicationCoordinator<RuntimePublishWorld, DSPCore*, RuntimePublicationBridge>`
- `AudioEngine.h:3546`: `RuntimePublishStore runtimeStore;` — **Store 自体は AudioEngine のメンバ**
- `AudioEngine.h:3646-3651`: `makeRuntimePublicationCoordinator()` が `RuntimePublicationCoordinatorFactory::create(bridge, runtimeStore)` で一時生成
- **Store の Owner は AudioEngine ではなく `convo::RuntimePublicationCoordinator`**（Store の template Owner = Coordinator 自身。AudioEngine は Store を値保持しているだけで friend Owner ではない）

**RuntimeWorldAuthority の現状（delegate）**:
- `RuntimeWorldAuthority.h:78-157`: `coordinator_`（ISR `RuntimePublicationCoordinator&`）参照を保持し commit/read を委譲（:81-124）
- **既に `ownerChannel_`（:154）と `lifetime_`（:152）を値保持** — X4-B の最終形（ownerChannel + lifetimeState）は既にこの Authority にある
- `AudioEngine.CtorDtor.cpp:28`: `worldAuthority_(runtimePublicationBridge_)` で初期化

**既存テスト（X4-B で仕様変更が必要）**:
- `ISRSemanticValidationTests.cpp:641-658` `testRuntimeWorldAuthorityAdapter`: **「pure delegate（coordinator と同じ epoch/sequence/version を返す、シャドウ状態なし）」を検証** — X4-B で「Authority owns Store」に変更
- `RuntimeWorldAuthorityProjectionTests.cpp:105`: `testRuntimeWorldAuthorityProjectionContract`

---

### X4 を二段階に分離（十七次で確定）

いきなり ownership topology を変更するのは危険。**X4-A（semantic/name convergence）と X4-B（physical ownership convergence）に分離**:

```
X4-A: 名前・責務の完全分離（rename のみ。低リスク）→ ビルド/テスト
X4-B: RuntimeStore ownership 移動（template Owner + friend 変更。大規模リファクタ）→ authority topology 完成
```

**X4-B は `RuntimeStore<World, Owner>` の template パラメータと `friend Owner` の構造を変更するため、単純な rename として扱ってはならない**。`RuntimeStore` / `RuntimeWorldAuthority` / `RuntimePublishAuthority` / `RuntimePublishExecutor` の4点を一つの ownership migration として実装・検証する。

---

### 6.4-X4-A — Authority naming の分離

**現状**: 二つの異なる型が同じ `RuntimePublicationCoordinator` 名を持つ（semantic ambiguity の根源）。

| 型 | 現状の名前 | 実体 |
|----|-----------|------|
| ISR 側 | `convo::isr::RuntimePublicationCoordinator` | Intent enqueue / dispatch / runtime coordination |
| core 側 | `convo::RuntimePublicationCoordinator<World, Handle, Bridge>` | RuntimeStore / publishWorld / publication implementation |

**X4-A の新しい名前**:
```cpp
convo::isr::RuntimeIntentCoordinator   // 旧 convo::isr::RuntimePublicationCoordinator（Intent 責務）
convo::RuntimePublishAuthority         // 旧 convo::RuntimePublicationCoordinator<...>（Publish 実装、内部実装名）
RuntimeWorldAuthority                  // 最終的な外部 semantic surface（不変）
```
- `RuntimePublishAuthority` は**最終 authority surface ではない**。内部実装として残してよい
- **🔴🔴🔴 二十次レビュー（2026-08-09）— `RuntimePublishAuthority` は `RuntimeStore::WriteAccess` を所有してはいけない（§7）**: X4-B 後に `RuntimeWorldAuthority → RuntimePublishAuthority → RuntimeStore` と二階層化すると、**physical write authority が二階層化する危険**。明確に:
  ```
  RuntimeWorldAuthority
      │
      └── owns Store
              └── WriteAccess
  ```
  `RuntimePublishAuthority` が残る場合でも、**algorithm / helper / factory / internal implementation に限定**し、`RuntimeStore::WriteAccess` を所有してはならない（INV-X4-3 を強化）
- 最終的な外部 semantic surface は `RuntimeWorldAuthority`。最新レビューも「`RuntimePublishAuthority` は内部実装名として残してよいが、`RuntimeWorldAuthority` 以外から `RuntimeStore::publishAndSwap()` に到達できない」方向

**X4-A の変更対象（実コードで確定）**:

core 側 `AudioEngine.h`:
```cpp
// 現行: using RuntimePublicationCoordinator = convo::RuntimePublicationCoordinator<RuntimePublishWorld, DSPCore*, RuntimePublicationBridge>;
// 変更: using RuntimePublishAuthority = convo::RuntimePublicationCoordinator<RuntimePublishWorld, DSPCore*, RuntimePublicationBridge>;
// 現行: makeRuntimePublicationCoordinator()（:3646）
// 変更: makeRuntimePublishAuthority()（:3646）
```

ISR 側 `convo::isr::RuntimePublicationCoordinator` → `RuntimeIntentCoordinator`:
| ファイル | 使用箇所 |
|---------|---------|
| `ConvolverProcessor.h` | :215 `setRetireCoordinator` |
| `EQProcessor.h` | :429, :466 |
| `SafeStateSwapper.h` | :79, :454 |
| `AudioEngine.h` | :1196, :4403-4404, :4667 |
| `AudioEngine.Publication.cpp` | :50, :55 |
| `ISRSoakTests.cpp` | :14, :39, :274, :325 |
| `ISRSemanticValidationTests.cpp` | 多数（:21-672） |

**🔴🔴🔴 core 側の読み取り static 関数呼び出し（X4-A の rename は write 側だけでなく読み取り側も含む）**:

X4-A で `convo::RuntimePublicationCoordinator → RuntimePublishAuthority` に rename すると、**write 側だけでなく、AudioEngine 全体の読み取り static 関数呼び出しも改名対象**になる:
| 呼び出し | 用途 | 箇所 |
|---------|------|------|
| `RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore)` | 現在の publish world 参照取得 | AudioEngine.h:1331-1332, 2119-2120, 3116-3117, 3383-3384, 3691-3692 / AudioEngine.Commit.cpp:559 / AudioEngine.Processing.Latency.cpp:91-92 |
| `RuntimePublicationCoordinator::acquireReadToken(runtimeStore)` | 読み取りトークン取得 | 同上（consumeWorldHandle と対） |
| `RuntimePublicationCoordinator::consumePublishedWorld(runtimeStore)` | publish world 取得 | AudioEngine.h:3551（observePublishedWorld 内部） |
| `observePublishedWorld()` ラッパー | Store 読み取りの getter | AudioEngine.Publication.cpp:68 / RuntimeBuilder.h:189 / RuntimePublicationOrchestrator.cpp:108,187 / テスト（PublishPipelineIntegrationTests / DeferredFlowIntegrationTests / SoakPublishIntegrationTests） |

- **X4-A の rename 対象 = write 側（一時生成 publishWorld）+ 読み取り側（上記 static 関数呼び出し）**
- **🔴🔴🔴 十九次レビュー（2026-08-09）— `getCurrent()` を `consumeWorldHandle(runtimeStore)` の置換先にすることは NO-GO**: `RuntimeWorldAuthority::getCurrent()`（RuntimeWorldAuthority.h:96-99）は `coordinator_.getCurrent()` を委譲し、**ISR 側 `currentWorld_` を読む**。一方 `consumeWorldHandle(runtimeStore)` は **core 側 `RuntimeStore::current` を読む**。**両者は別の atomic source** であり、単純置換は read source の意味を変更する（rename/refactor ではない）:
  ```
  getCurrent()                → currentWorld_（ISR metadata/pointer source）
  consumeWorldHandle(store)   → RuntimeStore::current（physical pointer source）
  ```
  - **X4-B の read path は「write topology migration と同時に全面変更しない」**（レビュー §16）。段階的に:
    - X4-B-1: Store ownership 移動
    - X4-B-2: write path 移動（publishAndSwap → authority）
    - X4-B-3: read path を Authority へ段階的移行（`consumeWorldHandle(runtimeStore)` → Authority の read API。`getCurrent()` ではなく**新設の read authority API**）
  - **Read Authority と Write Authority を分けて設計**する（レビュー §8）:
    ```
    RuntimeWorldAuthority
      ├── Publication metadata authority → commit()
      ├── Physical publication authority → publish()
      ├── Runtime read authority         → acquireReadToken() / consumeWorldHandle() 相当
      ├── Owner authority                → OwnerChannel
      └── Lifetime authority             → LifetimeState
    ```
  - **`getCurrent()` は「Publication metadata の取得（currentWorld_ 由来）」であり、「physical world 参照の一般取得」に流用しない**

**🔴🔴🔴 二十次レビュー（2026-08-09）— Read Authority API を新設（§21）**:
`getCurrent()`（ISR metadata source）と `consumeWorldHandle(store)`（physical source）の意味が異なるため、**`getCurrent()` を万能 read API にしない**。Authority に **physical RuntimeStore read 専用 API** を新設:
```cpp
class RuntimeWorldAuthority {
public:
    // ★ ISR metadata source（currentWorld_ 由来）
    [[nodiscard]] PublicationEpoch currentEpoch() const noexcept;            // 既存
    [[nodiscard]] PublicationSequenceId sequence() const noexcept;           // 既存
    [[nodiscard]] std::uint64_t getVersion() const noexcept;                 // 既存（currentWorld_ 由来）

    // ★ physical RuntimeStore read source（RuntimeStore::current 由来）— 新設
    [[nodiscard]] const RuntimePublishWorld* observePublishedWorld() const noexcept;
    [[nodiscard]] ReadToken acquireReadToken() const noexcept;
    [[nodiscard]] const RuntimePublishWorld* consumeWorldHandle(const ReadToken&) const noexcept;
};
```
- **`getCurrent()` = ISR metadata source** / **`observePublishedWorld()` 等 = physical published world source** を**型/API レベルで分離**
- これは **INV-X4-7 / INV-X4-8 を強く**する（source-role separation を API 契約として固定）
- **🔴🔴🔴 十八次別視点15調査（2026-08-10）— X4-B-9 の read API 置換対象を確定（実コード検証）**: `RuntimeWorldAuthority.h` には read API（observePublishedWorld / acquireReadToken / consumeWorldHandle）は**未実装**（X4-B-9 で新設予定）。現状の read path は全て `RuntimePublicationCoordinator` の static 関数を直接呼ぶ:（★ 2026-08-11 実装済み: X4-B-9 で専用 read API に置換済み）
  ```
  RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore)        → RuntimeWorldAuthority::consumeWorldHandle(ReadToken)
  RuntimePublicationCoordinator::acquireReadToken(runtimeStore)          → RuntimeWorldAuthority::acquireReadToken()
  RuntimePublicationCoordinator::consumePublishedWorld(runtimeStore)     → RuntimeWorldAuthority::observePublishedWorld()
  AudioEngine::observePublishedWorld()（:3550）                          → worldAuthority().observePublishedWorld()
  ```
  - **置換対象の呼び出し元**: AudioEngine.h:1331/2119/3116/3383/3691 / AudioEngine.Commit.cpp:559 / AudioEngine.Processing.Latency.cpp:91 / observePublishedWorld 経由（RuntimeBuilder.h:189 / Orchestrator.cpp:108,187 等）
  - **X4-B-9 はこれらを `worldAuthority().readAPI()` に一括置換**する（getCurrent() は置換先にしない — 別 source）
  - **単調性監視（observeLastSeenGeneration_ / observeLastSeenSequenceId_）は AudioEngine 側で維持**（別視点7で確認済み）
- **X4-A では読み取り経路も「型参照単位」で rename するだけで、Store の物理位置は変えない**（AudioEngine が Store を保持したまま）
- テスト側: `PartialPublicationRejectTests.cpp` / `RuntimePublicationCoordinatorTests.cpp` の `Coordinator::consumeWorldHandle(store)` も改名対象（core 型の直接参照）

- **X4-A は文字列置換ではなく、型参照単位の rename**（serena / coccinelle 相当の reference-aware ツールで一括適用）

**X4-A で絶対に維持する経路**（ownership topology は変更しない）:
```
RuntimeIntentCoordinator → PublicationAdmission → PublishExecutor
  → RuntimeWorldAuthority::commit() → RuntimePublishAuthority → RuntimeStore::WriteAccess::publishAndSwap()
```
- `PublishExecutor` が sole execution gateway（`ownerChannel().take()` → `commit()` → `publishWorld()`）であることを X4-A では壊さない

---

### 6.4-X4-B — ownership topology の変更（X4 の本体）

**現状の topology**（実コード検証済み）:
```cpp
// core/RuntimePublicationCoordinator.h:34
using Store = RuntimeStore<World, RuntimePublicationCoordinator<World, Handle, Bridge>>;  // Owner = 自身
// core/RuntimeStore.h:81
friend Owner;                                    // Owner のみ acquireWriteAccess()
// core/RuntimeStore.h:83-86
[[nodiscard]] WriteAccess acquireWriteAccess() noexcept { return WriteAccess(*this); }
// core/RuntimeStore.h:35-45
[[nodiscard]] T* publishAndSwap(T* next) noexcept { return exchangeAtomic(store_->current, next, std::memory_order_acq_rel); }
```

**X4-B の最終 topology**:
```cpp
RuntimeWorldAuthority
    ├── RuntimeStore<RuntimePublishWorld, RuntimeWorldAuthority>   // Owner = RuntimeWorldAuthority に変更
    │       └── WriteAccess
    ├── OwnerChannel
    └── LifetimeState
```
```cpp
RuntimeStore<RuntimePublishWorld, RuntimeWorldAuthority>  // template Owner を変更
friend RuntimeWorldAuthority;                             // friend 関係も変更
RuntimeWorldAuthority::acquireWriteAccess() だけが write capability を取得できる
```

**🔴🔴🔴 二十一次レビュー（2026-08-09）— X4-B 実装前に固定すべき4点（型・constructor・member declaration・初期化順序）**:

X4-B 実装開始前に、`RuntimeWorldAuthority` の完全な現行定義と `RuntimeStore` の型依存関係を実コードで固定する:

**① member declaration order（確定）**:
```
runtimeStore_     // 先: 初期化
writeAccess_      // 次: runtimeStore_ を参照
ownerChannel_     // 次
lifetime_         // 次
registry_         // 後
```
- C++ 逆順破棄により writeAccess_ → runtimeStore_ の順で破棄（WriteAccess が生きている間に Store が破棄されない）
- 現行（RuntimeWorldAuthority.h:151-154）は `coordinator_ / lifetime_ / registry_ / ownerChannel_` の順 → X4-B で `runtimeStore_ / writeAccess_` を追加し、`coordinator_` を削除

**② `RuntimeStore<World, RuntimeWorldAuthority>` の CRTP 的 template 依存（コンパイル可能性）**:
実コード検証で確定:
```
現行 RuntimePublicationCoordinator（core/RuntimePublicationCoordinator.h:34）:
  using Store = RuntimeStore<World, RuntimePublicationCoordinator<World, Handle, Bridge>>;
  // ★ 既に「自分自身を Owner にした CRTP」が存在し、コンパイル実績あり
```
- **`RuntimeStore<World, Self>` のパターンは既存実績あり**（RuntimePublicationCoordinator が自身の Store の Owner）。X4-B の `RuntimeStore<RuntimePublishWorld, RuntimeWorldAuthority>` も同一パターンでコンパイル可能
- **`friend Owner`（RuntimeStore.h:81）は incomplete type でも動作**（friend 宣言は前方宣言で成立）
- **`static_assert(std::is_class_v<Owner>)`（RuntimeStore.h:16）は incomplete type でも well-formed**（`std::is_class` は incomplete でも true を返す）
- **`RuntimePublishWorld = RuntimeState`**（AudioEngine.h:329）。RuntimeWorldAuthority.h:18 の `struct RuntimeState;` 前方宣言で pointer 参照は成立
- **include 依存**: RuntimeWorldAuthority.h は `core/RuntimeStore.h` を include する必要がある（WriteAccess は nested class のため complete type が必要）。RuntimeStore.h は `audioengine/AtomicAccess.h` のみ include（:8）するため、**循環依存は発生しない**
- **⚠️ 実装時の検証ポイント**: `RuntimeStore<RuntimePublishWorld, RuntimeWorldAuthority>::WriteAccess` を member として持つには RuntimeStore.h の complete type が必要。ただし **RuntimeState は forward decl で十分**（WriteAccess の member は `RuntimeStore* store_` / `std::atomic<T*> current` — pointer のみ）
- **🔴🔴🔴 二十一次レビュー + 実コンパイル検証（2026-08-09・g++ -std=c++20）で確定**:
  ```
  検証コード: RuntimeStore<RuntimeState, RuntimeWorldAuthority> を RuntimeWorldAuthority が
              member として所有（CRTP）。owner = 自身。
  結果: COMPILE_OK / RUN_OK
  検証した static_assert:
    - is_same_v<Store::OwnerType, RuntimeWorldAuthority>   → OK（using OwnerType 追加時）
    - !is_copy_constructible_v<RuntimeWorldAuthority>      → OK
    - !is_move_constructible_v<RuntimeWorldAuthority>      → OK
    - is_move_constructible_v<Store::WriteAccess>          → OK
    - !is_copy_constructible_v<Store::WriteAccess>         → OK
  発見: `Store::Owner` はコンパイル不可（RuntimeStore に using Owner が無い）
        → `using OwnerType = Owner` を RuntimeStore.h に追加（Test 1 用）
  ```
  **結論**: X4-B の CRTP パターンは `using OwnerType` 追加で完全にコンパイル可能。**二十一次レビュー §② の懸念（incomplete-type / header include / nested WriteAccess の定義位置）は全て解消済み**

**③ `RuntimeWorldAuthority::publish()` の ownership transfer（失敗経路・null経路）**:
```cpp
PublishResult publish(RuntimeStateOwner&& owner, const PublishMetadata& metadata) noexcept;
// 1. validate: owner が null なら Failed / metadata が不正なら Rejected
// 2. commit metadata（ISR currentWorld_ 更新）
// 3. RuntimePublishWorld* next = owner.release();        // null owner → ここで Failed
// 4. auto* previous = writeAccess_.publishAndSwap(next);
// 5. if (previous == nullptr && next == nullptr) return Failed;   // null→null 異常
// 6. return Success（previous を caller へ）
```
- **null owner**: `owner.release()` が nullptr を返す → `publishAndSwap(nullptr)` は Store を null にする。**通常 publish では owner は非 null 前提**（null は shutdown clear が担当 — X4-B-7 の clearPublishedRuntimeSnapshotsNonRt に分離）
- **失敗経路**: validate 失敗 → Rejected / owner null → Failed / null→null swap → Failed
- **🔴🔴🔴 二十三次レビュー（2026-08-09）— `commit-before-swap` と `swap failure` を acceptance criterion に（§20）**: `commit ≠ publishAndSwap` を「2段階だから OK」とするだけでは不十分。**partial publication state（commit success → publishAndSwap failure）の扱いを明文化**する:
  ```
  acceptance criterion:
    commit-before-swap ordering（commit(N) happens-before publishAndSwap(N)）— Test 7
    swap failure is architecturally impossible OR handled:
      - publishAndSwap は単一原子 exchange（RuntimeStore.h:35-45）で失敗しない（atomic）
      - 唯一の失敗経路: null→null swap（:124-126 相当）— 異常として Failed を返す
      - owner null は publish() の validate（:2338）で事前検出し、swap 前に Failed
  ```
  - **publishAndSwap 自体は単一の atomic exchange であり、CPU レベルで失敗しない**。したがって「commit 成功後に swap が失敗する」は**architecturally impossible**（swap は常に成功する）
  - **唯一の例外的状態**: null→null swap（異常）→ Failed。これは validate 段階で事前検出可能（owner null チェック）
  - **X2 との境界**: commit-before-swap ordering は X4（Test 7）、completion ordering は X2（INV-X2-6）

**④ Bootstrap / shutdown clear の初期化・破棄順序**:
```
初期化順序:
  AudioEngine ctor → worldAuthority_（X4-B: runtimeStore_ 構築 + writeAccess_ acquire）
  → CoordinatorLoop 起動 → Bootstrap publish（X4-B-6）
破棄順序:
  shutdown clear（clearPublishedRuntimeSnapshotsNonRt）→ CoordinatorLoop join
  → Builder join → Audio 停止 → worldAuthority_ 破棄（writeAccess_ → runtimeStore_）
```
- **Bootstrap は CoordinatorLoop 起動前に authority.publish() で同期 publish**（AudioEngine.Init.cpp:53-54 相当）
- **shutdown clear は authority.clearPublishedRuntimeSnapshotsNonRt()**（publish() と分離・X4-B-7）
- **worldAuthority_ の破棄順序**: AudioEngine のメンバとして値保持されるため、AudioEngine の破棄時に C++ の逆順破棄に従う。**writeAccess_ → ownerChannel_ → lifetime_ → registry_ → runtimeStore_ の順で破棄**（X4-B の member order と整合）

**`RuntimeWorldAuthority` の設計変更**:
```cpp
// 現行: RuntimePublicationCoordinator& coordinator_;   // delegate（RuntimeWorldAuthority.h:151）
// X4-B 後:
class RuntimeWorldAuthority {
    ...
private:
    RuntimeStore<RuntimePublishWorld, RuntimeWorldAuthority> runtimeStore_;
    RuntimeStore<RuntimePublishWorld, RuntimeWorldAuthority>::WriteAccess writeAccess_;
    OwnerChannel ownerChannel_;
    LifetimeState lifetimeState_;
};
```

**🔴🔴🔴 二十次レビュー（2026-08-09）— Store ownership と constructor の topology を確定（§3/§6）**:

「**Authority が Store を所有する**」ことと「**AudioEngine の lifetime/topology の中で Store を安全に所有できる**」ことは**別問題**。X4-B で Store を Authority に埋め込む際、**AudioEngine における runtimeStore の object identity が変わる**（現在は AudioEngine 直下 → X4-B 後は Authority 内部）ことを踏まえる:
- **X4-B の変更は単なる ownership 移動ではない**。Store の物理位置（object identity）が変わるため、AudioEngine の全 `consumeWorldHandle(runtimeStore)` 参照（10箇所以上）が影響を受ける
- **コンストラクタは「外部 Store を受け取る」形を残してはいけない**:
  ```cpp
  // ✗ 禁止: RuntimeWorldAuthority(RuntimeStore& store, ...)  ← ownership ambiguity が残る
  // ✓ 推奨: Authority 自身が Store identity を形成する
  RuntimeWorldAuthority(/* 依存 ... */)
      : runtimeStore_()
      , writeAccess_(runtimeStore_.acquireWriteAccess())   // Authority が Store を構築し WriteAccess を取得
      , ownerChannel_()
      , lifetimeState_()
  {
  }
  ```
- **外部 Store 受け取りを残すと**: `Authority → 外部 Store` という ownership ambiguity が残り、X4 の「Authority が物理 write owner」という目的が曖昧になる
- **WriteAccess は Store より後に宣言**（C++ 逆順破棄で Store より先に破棄）。`writeAccess_(runtimeStore_.acquireWriteAccess())` の初期化順序はメンバ宣言順序に従う（runtimeStore_ → writeAccess_）

**🔴🔴🔴 十九次レビュー（2026-08-09）— コンストラクタ変更の推奨順序（§10）**:
現行の `RuntimeWorldAuthority(runtimePublicationBridge_)`（delegate-oriented, CtorDtor.cpp:28）を Store 所有の形に変更する際、**コンストラクタを先に変更しない**方が安全。推奨順序:
```
1. RuntimeStore 型変更（Store<World, RuntimeWorldAuthority>）
2. Factory 変更（makeRuntimePublishAuthority / create）
3. Authority 内部に Store 追加
4. WriteAccess 取得（acquireWriteAccess）
5. constructor 接続（依存を Authority が直接受ける構造へ）
```
- **理由**: コンストラクタ先行変更は、Store 型・Factory の変更が未完了の間にコンパイルエラーや中間状態を生む。型変更 → 所有 → 接続の順で依存を解決する

**🔴🔴🔴 `WriteAccess` と `RuntimeStore` の寿命関係（静的レビュー項目として固定）**:
`WriteAccess` は内部に `RuntimeStore* store_`（非所有参照）を保持する。**`RuntimeStore` より先に `WriteAccess` を破壊してはいけない**。
```cpp
RuntimeStore runtimeStore_;   // 先に宣言
WriteAccess  writeAccess_;    // 後に宣言 → C++ の逆順破棄により writeAccess_ が先に破棄される
```

**🔴🔴🔴 十九次レビュー（2026-08-09）— `RuntimeWorldAuthority` 自体の move/copy 禁止（追加）**:
`RuntimeWorldAuthority` は `RuntimeStore` と `WriteAccess` を**内部所有**するため、**Authority 自体を move/copy 可能にすると topology が破綻**する。宣言順序に加えて、**Authority 自体の非 movable/non-copyable 契約を static_assert で固定**する:
```cpp
class RuntimeWorldAuthority {
public:
    RuntimeWorldAuthority(const RuntimeWorldAuthority&) = delete;
    RuntimeWorldAuthority& operator=(const RuntimeWorldAuthority&) = delete;
    RuntimeWorldAuthority(RuntimeWorldAuthority&&) = delete;
    RuntimeWorldAuthority& operator=(RuntimeWorldAuthority&&) = delete;
    // ...
private:
    RuntimeStore runtimeStore_;   // 先に宣言
    WriteAccess  writeAccess_;    // 後に宣言
    OwnerChannel ownerChannel_;
    LifetimeState lifetime_;
    PendingPublishRegistry registry_;
};
```
- **static_assert による契約固定**:
  ```cpp
  static_assert(!std::is_copy_constructible_v<RuntimeWorldAuthority>);
  static_assert(!std::is_copy_assignable_v<RuntimeWorldAuthority>);
  static_assert(!std::is_move_constructible_v<RuntimeWorldAuthority>);
  static_assert(!std::is_move_assignable_v<RuntimeWorldAuthority>);
  ```
- **理由**: WriteAccess は Store への非所有参照を持ち、move すると参照先が分離する。Store は non-copyable/non-movable（RuntimeStore.h:66-70）のため、Authority を move すると Store の stable address 前提が壊れる
- **AudioEngine 側**: `worldAuthority_`（AudioEngine.h:4669）は値保持されるため、**move 不要**（CtorDtor.cpp:28 で初期化のみ）。コンストラクタで Store を構築し、WriteAccess を acquire する構造に変更

**🔴🔴🔴 追加調査（2026-08-09）— `currentWorld_` と `runtimeStore.current` の二重管理を解消（未確定事項の確定）**:

**実コード検証で判明した二重 atomic 構造**:
```
publish world の pointer を保持する atomic が2つ存在する:
  atomic #1: ISRRuntimePublicationCoordinator::currentWorld_（ISRRuntimePublicationCoordinator.h:380）
             ← ISR commit() が更新（ISRRuntimePublicationCoordinator.cpp:112）
  atomic #2: RuntimeStore::current（core/RuntimeStore.h:88）
             ← publishWorld() → WriteAccess::publishAndSwap() が更新（core/RuntimePublicationCoordinator.h:121）
```
- **同じ publish に対して2つの atomic が更新される**（ISR commit → currentWorld_ / PublishExecutor → runtimeStore.current）
- ISR 側の read（currentPublicationEpoch / currentPublicationSequenceId / getCurrent / getVersion）は **currentWorld_ から導出**（ISRRuntimePublicationCoordinator.cpp:89-191）
- core 側の read（consumeWorldHandle / acquireReadToken / observePublishedWorld）は **runtimeStore.current から取得**（AudioEngine.h:1331-3692）

**X4-B 後の解消方針（未確定 → 確定）**:
```
案1（推奨・最小変更）: 二重管理を維持したまま、Store の Owner だけ RuntimeWorldAuthority に移す
  - currentWorld_（ISR metadata source）は RuntimeWorldAuthority::commit() が引き続き更新
  - runtimeStore.current（physical swap）は writeAccess_.publishAndSwap() が更新
  - X4-B のスコープを「write authority の一本化」に限定（二重管理の解消は X4 のスコープ外）
案2（将来・optional）: currentWorld_ を廃止し、読み取りを全て runtimeStore.current に一本化
  - ISR 側 read（currentEpoch/sequence/version）を RuntimeState::publication から直接導出（既に FUTURE-4 で
    persistentState_ を削除済み → currentWorld_ 自体も削除可能な方向）
  - ただしこれは ISR 側 commit の意味論変更を伴うため、X4 のスコープ外・独立タスク
```
- **X4-B のスコープは案1（write authority 一本化）に限定**し、案2（二重 atomic 解消）は将来タスクとして A-2.20 に記録する
- **ISR 側 read は RuntimeWorldAuthority::commit() の currentWorld_ 更新に依存するため、X4-B で ISR commit を消すことはできない**（ISR のメタデータ source は残す）
- **🔴🔴🔴 二十四次レビュー（2026-08-09）— dual-pointer を「暫定正常状態」として明示（§27-C）**: `currentWorld_` と `RuntimeStore::current` が2つ存在する状態は**architecture として理想ではない**が、現段階で無理に一本化するのも危険。**「暫定正常状態」として明示的に定義**する:
  ```
  X4-B: write authority singularization（write capability を RuntimeWorldAuthority に一意化）
  Future: read-source singularization（currentWorld_ を廃止し read を runtimeStore.current に一本化）
  ```
  - **X4-B の目的は「write authority singularization」であり、「read-source singularization」は別問題として扱う**
  - **dual-pointer 状態は「暫定正常状態」（temporary but valid）**: publish transaction 完了後は INV-X4-6（同一 PublicationIdentity）を保証するため、正常動作として許容
  - **X4-B で read-source singularization を同時にやらない**（publication semantics・read source・completion ordering・lifetime の複数問題を同時変更しない — 二十次レビューと同じ原則）

**重要な区別（X4-B で確定）**:
```
ISR commit()  → currentWorld_ 更新 = ISR metadata write（RuntimeWorldAuthority が継続所有）
publishWorld() → runtimeStore.current 更新 = physical store swap（RuntimeWorldAuthority-owned WriteAccess に移動）
この2つは別の write であり、X4-B は後者の authority を RuntimeWorldAuthority に一本化する
```

**`RuntimePublishAuthority` の役割縮小**:
```cpp
// 現行（RuntimePublishExecutor.h:55-56）:
auto coordinator = ctx.engine.makeRuntimePublicationCoordinator();
(void)coordinator.publishWorld(std::move(owner));
// X4-B 後: この一時生成を廃止し、authority.commit(...) / authority.publish(...) 内部で
//           唯一の WriteAccess を使用して publish する。
```

**🔴🔴🔴 X4 の責務範囲（やらないことを固定）**:
```
X4 = ownership topology convergence
not X4 = all runtime lifecycle convergence
```
- **publish 後の completion / retire / reclaim まで RuntimeWorldAuthority に再集約しない**（X2/X3 の責務）
- **X4 は authority topology だけを変更し、drain semantics は変更しない**（shutdown drain は swapPending / retireBacklog / publicationBacklog / pendingIntent / fallback / reclaimInFlight / deferredRetireResidency / quarantineResident を見る現状を維持）

**X4 で「やらないこと」6項目（十七次計画 §23 確定）**:
| # | やらないこと | 責務の持ち主 |
|---|-------------|-------------|
| ① | `RuntimeIntentCoordinator` と `RuntimeWorldAuthority` を統合しない | 責務が異なる（Intent vs write authority） |
| ② | `RuntimePublishAuthority` と `RuntimeWorldAuthority` を無理に同一クラスへ統合しない | `RuntimePublishAuthority` は core implementation abstraction として残す余地 |
| ③ | X4 で completion semantics を変更しない | **X2** |
| ④ | X4 で reclaim path を変更しない | **X3** |
| ⑤ | X4 で residency counter を変更しない | **X5** |
| ⑥ | X4 で Quarantine/Intent semantics を変更しない | **X6** |

**推奨する最終 API（X4-B 後・🔴🔴🔴 十九次 §9 + 二十次 §8 で内部仕様を厳密化）**:
```cpp
PublishResult publish(RuntimeStateOwner&& owner, const PublishMetadata& metadata) noexcept;
// 内部:
//   1. validate precondition（metadata / boundary / authority チェック）
//   2. commit metadata（ISR currentWorld_ 更新 — semantic commit）
//   3. RuntimePublishWorld* next = owner.release();
//      auto* previous = writeAccess_.publishAndSwap(next);   // 唯一の Store swap（physical）
//   4. return previous（oldWorld を caller へ返す）
//   ★ retire はここに戻さない（Lifetime の責務 — X3/retire と分離）
```
- **🔴🔴🔴 二十次レビュー（2026-08-09）— publish() の責務限定（単純置換では成立しない）**: 現行 `publishWorld()`（core/RuntimePublicationCoordinator.h:100-141）は **seal（:107）→ validate（:111）→ release（:119）→ publishAndSwap（:121）→ didPublish（:130）→ willRetire（:135）→ retire（:138）** まで担う。X4 計画の「retire は publish() に戻さない」は、**この現行責務を単純に publish() へ移すだけでは成立しない**。したがって:
  ```
  X4-B の publish() に含める:   validate / commit metadata / owner.release / publishAndSwap / return oldWorld
  X4-B の publish() に含めない: didPublishRuntimeNonRt / willRetireRuntimeNonRt / retireRuntimePublishWorldNonRt
  ```
  - **didPublish / willRetire / retire は publish() の後、PublishExecutor の Execution tail → completion → LifetimeState へ委譲**する（既存 onPublishCompleted / advanceRetireEpoch 経路）
  - **🔴🔴🔴 十八次別視点8調査（2026-08-09）— Bridge（RuntimePublicationBridge）の役割を確定**: `RuntimePublicationBridge`（AudioEngine.h:3446-3500）は validate / didPublish / willRetire / retire を担う:
    ```
    validatePublicationNonRt（:3458）: Validator + engine precheck（publish() の validate に使用）
    didPublishRuntimeNonRt（:3473）: onRuntimePublishedNonRt（publish() 後）
    willRetireRuntimeNonRt（:3478）: onRuntimeRetiredNonRt（publish() 後・shutdown 中は skip）
    retireRuntimePublishWorldNonRt（:3489）: unseal → デストラクタ → aligned_free（物理削除・publish() 後）
    ```
    - **X4-B 後も Bridge は残る**（didPublish/willRetire/retire を PublishExecutor の Execution tail から呼ぶ）
    - **publish() は validate に Bridge を使う**（publish() の内部で validatePublicationNonRt を呼ぶ）が、didPublish/willRetire/retire は publish() の**外**（PublishExecutor の Execution tail）で Bridge を呼ぶ
    - **Bridge は AudioEngine の内部参照（engine_ / validator_）を持つ**ため、X4-B の Store 移動後も AudioEngine 側に残る（RuntimeWorldAuthority に移さない）
  - **owner の型**: `RuntimeStateOwner&&`（aligned_unique_ptr<const RuntimeState> 相当）— `owner.get()` は non-owning read、`owner.release()` のみ ownership transfer
  - **返却値**: `previous`（oldWorld）を返し、retire 対象を caller（PublishExecutor）に明示する
- **`publish()` は semantic publication transaction の唯一の execution boundary**（commit + publishAndSwap を束ねる）だが、**一つの atomic operation ではない**（2つの atomic が存在 — レビュー §10）
- **🔴🔴🔴 二十二次レビュー（2026-08-09）— commit 二重化の危険を明示（必須修正1）**: 現行 `PublishExecutor`（RuntimePublishExecutor.h:42-57）は既に `authority.commit(...)` を実行している。X4-B-4 で `authority.publish()` を導入する際、**PublishExecutor 側の `authority.commit()` を削除しないと commit が二重化する**:
  ```
  ✗ 危険: PublishExecutor → authority.commit(...) → authority.publish(...)  // commit 二重化
  ✓ 案A（採用）: publish() が transaction boundary。PublishExecutor から authority.commit() を完全に削除
      authority.publish(std::move(owner), PublishMetadata{...})
      内部: validate → commit → owner.release() → publishAndSwap() → return previous
  ✓ 案B（非推奨）: authority.commit(...) + authority.publishPhysical(...) の別API
      → 「semantic publication transaction の唯一の execution boundary」設計を弱める
  ```
  **X4-B-4 で commit の責務を publish() に移すことを明示**し、X4-B-5 で PublishExecutor から `authority.commit()` を削除する
- **retire は publish() に戻さない**: 内部の `lifetimeState_.retire(previous)` は**削除**し、oldWorld を caller に返す（PublishExecutor の Execution tail が onPublishCompleted で retire を処理する既存設計を維持）
- **X4 で新規に責務を広げすぎない**。write authority を移すことを主目的とし、publish 後の complete/retire/reclaim 再集約はしない（X2/X3 の責務）

**`PublishExecutor` の役割（X4 後も残す）**:
```
Intent payload → fixed snapshot を読む → OwnerChannel から ownership 取得
  → RuntimeWorldAuthority::publish() → WriteAccess::publishAndSwap()
```
- **PublishExecutor 自身が RuntimeStore を直接触らない**こと
- **🔴🔴🔴 十九次レビュー（2026-08-09）— sole gateway の定義を明確化**: 「PublishExecutor が sole gateway」ではなく **「RuntimeWorldAuthority が sole physical publish gateway」**。PublishExecutor は normal publish initiator、Bootstrap/ShutdownRuntime は lifecycle-controlled initiator。両者とも最終的な Store write は `RuntimeWorldAuthority::publish()` 経由（INV-X4-2 と整合）

**🔴🔴🔴 十八次調査（2026-08-09）— friend 関係の影響を確定**:
実コードで PublishExecutor のアクセス経路を確認:
```cpp
// AudioEngine.h:3580 — PublishExecutor は AudioEngine の private メンバにアクセス可
friend struct convo::isr::PublishExecutor;
```
- **現行**: `PublishExecutor`（friend struct）が `ctx.engine.makeRuntimePublicationCoordinator()` を直接呼び、AudioEngine の private `runtimeStore`（:3546）へアクセスしている（RuntimePublishExecutor.h:55-56）
- **X4-B 後のアクセス経路**: PublishExecutor は `ctx.engine.worldAuthority()`（AudioEngine.h:4749）経由で `RuntimeWorldAuthority::publish()` を呼ぶ。**friend 関係は維持される**が、`runtimeStore` への直接アクセスは X4-B で `worldAuthority()` 経由に置換される:
```cpp
// RuntimePublishExecutor.h — X4-B 後の executePublish（概念）
void executePublish(RuntimeWorldAuthority& authority, const Intent& intent, IntentHandlerContext& ctx) const noexcept
{
    auto owner = authority.ownerChannel().take(...);      // Owner 取得（不変）
    authority.publish(std::move(owner), ...);             // ★ X4-B: 一時生成 coordinator を廃止
    authority.registry().unregister(intent.sequenceId);
    // Execution tail（onPublishCompleted / advanceRetireEpoch / onPublishCommitted）は不変
}
```
- **🔴🔴🔴 十八次別視点12調査（2026-08-09）— Execution tail の構成要素を確定**: Execution tail は3つの独立した処理からなる:
  ```
  tail-1: ctx.transition.onPublishCompleted(...)（DSPTransition.h:49-90）
          = DSP activate / crossfade / retire（publish 成功後の DSP lifetime 操作）
          = Crossfade Registration Authority（registerCrossfade は DSPTransition のみ — CI gate: grep "registerCrossfade(" → DSPTransition only）
  tail-2: ctx.engine.advanceRetireEpoch()
          = retire epoch 前進（EBR）
  tail-3: ctx.engine.runtimeOrchestrator_->onPublishCommitted(intent.sequenceId)
          = X2 の completion（m_lastObservedSequence + notifyPublishReceipt → complete()）
  ```
  - **tail-1（DSPTransition）は X2 の completion と独立**: onPublishCompleted は DSP activate の execution、onPublishCommitted は completion watermark。**両者を混同しない**
  - **X4-B の publish() は tail-1〜3 を含まない**（publish() は validate + commit + swap + return oldWorld まで）。tail は PublishExecutor が実行
  - **X2 の completion（tail-3）は DSPTransition（tail-1）の後に発生**（RuntimePublishExecutor.h:74-84 の順序）
- **friend 関係の変更は不要**（PublishExecutor は引き続き AudioEngine の private メンバを参照するが、`runtimeStore` 直接アクセスを `worldAuthority()` 経由に変える）
- **X4-B の friend 影響**: `RuntimeStore::WriteAccess` の `friend Owner` が `RuntimeWorldAuthority` に変わる（core/RuntimeStore.h:81）。PublishExecutor は WriteAccess に直接アクセスせず `authority.publish()` 経由のため、**friend 変更の影響を受けない**

**`commit()` と `publishAndSwap()` の意味を分離**:
```
commit        = semantic publication metadata の確定（epoch / sequence / generation / version / boundary / authority）
publishAndSwap = physical RuntimeWorld pointer の公開（RuntimeStore.current への atomic exchange）
commit ≠ publishAndSwap
```

---

### 6.4-X4 の INV（ISR 観点・実コード検証済み）

| INV | 内容 | 検証 |
|-----|------|------|
| **INV-X4-1** | Intent authority singularity: Intent enqueue/dispatch → RuntimeIntentCoordinator only | 実コード: Intent enqueue は `runtimePublicationBridge_.enqueuePublicationIntent`（AudioEngine.h:4413）のみ |
| **INV-X4-2** | Publish execution singularity: PublishIntent → PublishExecutor → RuntimeWorldAuthority 以外から publish execution を開始しない。**ただし「例外」ではなく「lifecycle-controlled publish」と定義する（🔴 十九次レビュー §13）**: <br>Publish execution authority = RuntimeWorldAuthority<br>Normal publish initiator = PublishExecutor<br>Lifecycle-controlled publish = Bootstrap / ShutdownRuntime → RuntimeWorldAuthority<br>**正確には「PublishExecutor が sole gateway」ではなく「RuntimeWorldAuthority が sole physical publish gateway」**。Bootstrap / shutdown clear も最終的な Store write は Authority 経由 | 実コード: `PublishIntentHandler → PublishExecutor{}.executePublish`（ISRRuntimePublicationCoordinator_ProcessIntent.cpp:111）のみ。Bootstrap / shutdown clear の直接 publishWorld は X4-B で authority 経由に一本化 |
| **INV-X4-3** | **RuntimeStore write singularity（最重要）**: `RuntimeStore::publishAndSwap()` に到達可能なのは RuntimeWorldAuthority-owned WriteAccess のみ。**🔴 二十次レビュー（§7）で強化**: `RuntimePublishAuthority` は `RuntimeStore::WriteAccess` を**所有してはいけない**（algorithm/helper/factory に限定。二階層化の禁止）。**🔴 二十二次レビュー（§6）でさらに強化（必須修正3）**: `RuntimePublishAuthority` は以下を**一切所有してはいけない**:
```
RuntimePublishAuthority
    ├─ RuntimeStore を所有しない
    ├─ WriteAccess を保持しない
    ├─ publishAndSwap() を直接呼ばない
    ├─ RuntimeWorldAuthority の代替 authority にならない
    └─ RuntimeWorldAuthority が所有する Store に対する
       write capability を取得できない
```
**production code から `RuntimePublishAuthority::create()` 自体を削除**（factory が Store を生成できると INV-X4-3/X4-5 を破る） | X4-B の核心。現在は一時生成 coordinator.publishWorld が実行 |
| **INV-X4-4** | No RT ownership transfer: ISR/Audio callback から `RuntimeStateOwner` / `unique_ptr` / `shared_ptr` / `RuntimeStore::WriteAccess` を新規取得・破棄しない | Practical Stable ISR Runtime の原則（RT thread は実行主体でなく観測主体） |
| **INV-X4-5** | No second store: X4-B 後、RuntimeWorldAuthority::RuntimeStore 以外の `RuntimeStore` を作ってはいけない。`makeRuntimePublishAuthority()` が内部で別 Store を作る構造を残さない | X4-B の検証対象 |

**🔴🔴🔴 十九次レビュー（2026-08-09）— INV-X4-6 / INV-X4-7 を追加（currentWorld_ 二重管理の暫定許容と authority 分離）**:

`currentWorld_` と `RuntimeStore::current` の二重 atomic は**暫定的に残す**（実装スコープとして妥当）。ただし「INV-X4-3（publishAndSwap 単一 authority）」だけでは X4 の architecture correctness を完全に表せない。追加 invariant:
```
INV-X4-6: currentWorld_ and RuntimeStore::current
          must refer to the same publication identity
          after publish transaction completion.
          （publish transaction 完了後、両者が同じ publication identity を指す）

INV-X4-7: No API may treat currentWorld_ and RuntimeStore::current
          as independent authoritative RuntimeWorld sources.
          （currentWorld_ と RuntimeStore::current を独立した
            authoritative source として扱う API を禁止）
```
- **INV-X4-6**: publish 完了後、両者の publication identity が一致することを保証（Test 9 で検証）
- **🔴🔴🔴 二十次レビュー（2026-08-09）— INV-X4-6 の PublicationIdentity 構成要素を確定（§11）**: identity の構成要素を曖昧なまま Test 9 にしない:
  ```
  PublicationIdentity = sequenceId + publicationEpoch + mappedGeneration   ← 主 identity
  version / boundary は publication metadata として別扱い（identity の構成要素ではない）
  ```
  Test 9 は `currentWorld_.publication.sequenceId == runtimeStore.current.publication.sequenceId` かつ epoch/generation が一致することを検証（identity = sequence + epoch + mappedGeneration に限定）
- **INV-X4-7**: 二重 atomic を暫定的に残しても、**二つの Authority を許してしまうことを防ぐ**（`getCurrent()` を `consumeWorldHandle(runtimeStore)` の置換先にしない — 前述の NO-GO と整合。Test 10 で検証）
- **`currentWorld_` の lifetime 契約（レビュー §16）**: `currentWorld_` は **non-owning observation alias**（ownership source ではない）。World の実所有権は `RuntimeStateOwner → RuntimeStore → retire/reclaim` 系。`currentWorld_` を delete source や retire source に使用するのは禁止（誤配線防止のコメント契約）

**🔴🔴🔴 二十次レビュー（2026-08-09）— INV-X4-8 を追加（source-role separation・強く推奨）**:
```
INV-X4-8: currentWorld_ は metadata observation alias、
          RuntimeStore::current は physical publication source。
          Neither API may treat them as interchangeable.
          （currentWorld_ と RuntimeStore::current を
            交換可能な source として扱う API を禁止）
```
- **currentWorld_ = metadata observation alias**（getCurrent/getVersion/currentPublicationEpoch/currentPublicationSequenceId の source）
- **RuntimeStore::current = physical publication source**（observePublishedWorld / consumeWorldHandle の source）
- **禁止変換（architecture check で排除）**: `delete currentWorld` / `retire(currentWorld)` / `unique_ptr(currentWorld)` / `shared_ptr(currentWorld)` — currentWorld_ は ownership を持たない
- **INV-X4-8 は INV-X4-7 より強い**: 単に「独立 source として扱う API を禁止」でなく、**各 source の役割（metadata observation vs physical publication）を固定**する
- **🔴🔴🔴 二十六次レビュー（2026-08-09）— INV-X4-A/B/C を追加（強く推奨・§29）**: X4-B 後も `currentWorld_` / `RuntimeStore::current` の dual atomic が残るため、**コード/API レベルで禁止事項として固定**する:
  ```
  INV-X4-A: currentWorld_ is observation-only
            （currentWorld_ は観測専用。RuntimeWorld 取得元として使わない）

  INV-X4-B: RuntimeStore::current is sole physical RuntimeWorld source
            （RuntimeStore::current が唯一の物理 RuntimeWorld source）

  INV-X4-C: No RT API may derive RuntimeWorld ownership/lifetime from currentWorld_
            （RT API は currentWorld_ から RuntimeWorld の ownership/lifetime を導出しない）
  ```
  - **特に重要なのは**: **Audio Thread は `currentWorld_` を RuntimeWorld 取得元として使わない**（§25）。currentWorld_ は metadata observation のみ
  - **INV-X4-A/B/C は INV-X4-8 を補強**: INV-X4-8 は「交換可能と扱う API を禁止」、INV-X4-A/B/C は「各 source の役割を固定的に定義」（observation-only / sole source / RT 導出禁止）
  - **Test 10 は INV-X4-7 + INV-X4-8 + INV-X4-A/B/C を検証**する

**🔴🔴🔴 追加調査（2026-08-09）— 2段階 write の一貫性（X4-B の注意点として確定）**:

X4-B 後も publish は2段階で行われる（案1・最小変更）。PublishExecutor（RuntimePublishExecutor.h:42-57）は:
```
① authority.commit(...)          → ISR currentWorld_ 更新（metadata write）
② writeAccess_.publishAndSwap()  → runtimeStore.current 更新（physical store swap）
```
- **この2段階の間に、Audio thread の read が「古い core world + 新しい ISR metadata」を観測する可能性**がある（ISR 側 read と core 側 read が別 atomic を読むため）
- **ただし現状も同じ2段階構造**（PublishExecutor が commit → publishWorld の順で呼ぶ）。X4-B は「誰が write するか」を変えるだけで、2段階の順序・間隔は変えない
- **🔴🔴🔴 十九次レビュー（2026-08-09）— `publish()` を「一つの atomic publication」と定義してはいけない**: `RuntimeWorldAuthority::publish()` API を導入しても、commit（currentWorld_）と publishAndSwap（runtimeStore.current）の**2つの atomic operation が存在する限り、CPU レベルで atomic な一括 operation にはならない**。仕様としては:
  ```
  publish() = semantic publication transaction の唯一の execution boundary
  であるが、
  currentWorld_ と runtimeStore.current が同時に atomic に更新される
  とは定義しない（後者を保証するには別設計が必要）
  ```
- **X4 のスコープ**: 2段階 write の一貫性（同一 publish 内の metadata/pointer 原子性）は X2（completion ordering）の責務。X4 は authority の一本化のみ
- **🔴🔴🔴 十九次レビュー（2026-08-09）— X4 と X2 の境界を明確化**: `commit → publishAndSwap` の **ordering そのものは X4 で保証**する（X4-B で `authority.publish()` へ再構成するため）:
  ```
  X4 が保証: commit precedes publishAndSwap / publishAndSwap is singular / ownership transfer exactly once
  X2 が保証: completion sequence is monotonic / completion order == publication sequence order / committed != completed
  ```
- **X4-B での不変条件**: ①commit と ②publishAndSwap の順序は維持（先 metadata、後 physical swap）。逆転禁止（Test 7 commit-before-swap ordering で検証）

---

### 6.4-X4-B のテスト再設計（既存テスト仕様の変更）

| # | テスト | 内容 |
|---|--------|------|
| Test 1 | **Authority owns Store** | compile-time invariant: `static_assert(std::is_same_v<RuntimeWorldAuthority::Store::OwnerType, RuntimeWorldAuthority>)`。**🔴 二十一次レビュー + コンパイル検証（2026-08-09）で確定**: `RuntimeStore.h` には `using Owner` 型エイリアスが**存在しない**（Owner は template パラメータのみ）。**Test 1 を成立させるには `RuntimeStore` に `using OwnerType = Owner;` の公開型エイリアスを追加**する必要がある（g++ -std=c++20 でコンパイル検証済み・`Store::Owner` はコンパイル不可 / `Store::OwnerType` は COMPILE_OK） |
| Test 2 | **WriteAccess move-only** | 既存 invariant 維持（RuntimeStore.h:58-64 の static_assert を確認）。`!copy_constructible / !copy_assignable / move_constructible / move_assignable / nothrow move` |
| Test 3 | **publishAndSwap 唯一性** | source-level architectural test: `publishAndSwap(` の caller を列挙 → RuntimeWorldAuthority の write path のみ。**🔴 二十次レビュー（§18）で検証対象を厳密化**: 単純 grep でなく **`RuntimeStore<RuntimePublishWorld, ...>::WriteAccess` まで追う**。allowed / forbidden を明示:
```
allowed:   RuntimeWorldAuthority
forbidden: RuntimeIntentCoordinator / PublishExecutor / RuntimePublishAuthority
           / AudioEngine / Builder / DSPTransition
``` |
| Test 4 | **PublishExecutor bypass 禁止** | `runtimeStore.publishAndSwap(...)` を PublishExecutor が直接呼んでいないこと |
| Test 5 | **Coordinator bypass 禁止** | RuntimeIntentCoordinator から RuntimeStore / commit / publishAndSwap へ直接到達しないこと |
| Test 6 | **二重 Store 検出** | 静的検索で `RuntimeStore<` の RuntimePublishWorld 用インスタンスを列挙 → RuntimeWorldAuthority 配下のみが write-side Store。**🔴 二十次レビュー（§19）で定義を修正**: 禁止するのは **write-capable RuntimeStore**。**read-only Store reference（`const RuntimeStore&`）は read API が持ってよい**（これを違反とすると X4 の read migration を過剰に拘束）。**🔴 二十二次レビュー（§7）で write-capable の条件を厳密化**: `RuntimeStore<RuntimePublishWorld, Owner>` の write-capable instance について **`Owner == RuntimeWorldAuthority` を要求**する:
```
禁止: write-capable RuntimeStore instance（Owner != RuntimeWorldAuthority）が存在
      例: RuntimeStore<RuntimePublishWorld, SomeHelper> ← 潜在的に作れてしまうため検出
許容: const RuntimeStore<RuntimePublishWorld, ...>& を read API が保持（read migration 用）
要求: write-capable instance は Owner == RuntimeWorldAuthority のみ
      （「Store object の存在」と「write capability の存在」を分離）
``` |
| Test 7 | **commit-before-swap ordering**（🔴 十九次レビューで Test 7 の意味を明確化） | `commit(newWorld)` が `publishAndSwap(newWorld)` より**前**に発生すること。逆順を拒否。**sequence monotonicity は X2 のテストスイートに残す**（X4 の Test 7 は ordering 検証） |
| Test 8 | **ownership transfer exactly once** | Producer → OwnerChannel → Executor → Authority → Store の ownership transfer が exactly once（テスト: `std::unique_ptr` の move 回数を検証）。**🔴 OwnerChannel の SPSC 実装（十八次別視点4調査で確定）**: OwnerChannel（OwnerChannel.h:38-118）は **SPSC**（enqueue = Non-RT publish thread 単一 / take = ISR/audio thread 単一）、capacity 256。key = (sequenceId, epoch, mappedGeneration)。`enqueue`（:67-87）は key 重複 reject（no overwrite）/ `take`（:92-108）は single-transfer drain（slot は1回のみ）。**`owner.get()` は non-owning read / `std::move(owner)` のみ ownership transfer** の区別が型で保証される |
| Test 9 | **dual-pointer semantic consistency**（🔴 十九次レビューで追加） | publish 完了後、**PublicationIdentity（sequenceId + publicationEpoch + mappedGeneration）が一致**することを確認。**publish transaction の途中では transitional mismatch が発生し得る**ことをテスト仕様に明記（「途中状態を許容する」ことを確定）。**🔴 二十二次レビュー（§4）で pointer equality を禁止**: `currentWorld_.load() == runtimeStore.current.load()` のような **pointer equality を要求しない**（currentWorld_ は metadata observation alias / RuntimeStore::current は physical publication source のため、同一 pointer を要求すると INV-X4-8 の意味論を過剰に固定する）。**Test 9 は identity equality に限定** |
| Test 10 | **INV-X4-7 の検証**（🔴 十九次レビューで追加） | `currentWorld_` と `RuntimeStore::current` を独立した authoritative RuntimeWorld source として扱う API が存在しないこと（source-level 検査: `getCurrent()` が `consumeWorldHandle(runtimeStore)` の置換先として使われていないこと） |

**🔴🔴🔴 既存テストの仕様変更**:
- `testRuntimeWorldAuthorityAdapter`（ISRSemanticValidationTests.cpp:641）: **「pure delegate」を検証する現在の仕様は残してはいけない**。X4-B では Authority owns Store に変わるため、テスト仕様自体を変更する（Test 1 の compile-time invariant + Authority が own した Store の observe を検証）

---

### 6.4-X4 の実装順序（X4-0〜X4-11・🔴🔴🔴 十九次レビュー §19 反映）

| Step | 内容 | 検証 |
|------|------|------|
| X4-0 | **Baseline**: Debug/Release/CTest/ISR semantic/ISR soak を全 PASS | 変更前の状態 |
| X4-1 | **Authority contract / invariants 固定**: INV-X4-1〜INV-X4-7 をコードコメントとして固定（**INV-X4-6 / INV-X4-7 を含む**）。RuntimeIntentCoordinator / PublicationAdmission / PublishExecutor / RuntimeWorldAuthority / RuntimeStore / LifetimeState の責務を明文化 | コード変更前に |
| X4-2 | **ISR Coordinator rename**: `RuntimePublicationCoordinator → RuntimeIntentCoordinator`（全参照） | ビルド |
| X4-3 | **core Coordinator rename**: `RuntimePublicationCoordinator → RuntimePublishAuthority`（型 alias / factory / local variable / **読み取り static 関数呼び出し consumeWorldHandle / acquireReadToken / consumePublishedWorld の10箇所以上**） | ビルド |
| X4-4 | **compile/test**: Store Owner = core `RuntimePublishAuthority` のまま。**semantic/name convergence の検証** | 全テスト PASS |
| X4-5 | **RuntimeStore physical ownership migration**: `RuntimePublishAuthority └── RuntimeStore` → `RuntimeWorldAuthority └── RuntimeStore` へ移動。**X4-A では読み取り経路は rename のみ（X4-B で全面変更しない）** | ビルド |
| X4-6 | **WriteAccess acquisition**: `RuntimeWorldAuthority::WriteAccess` を保持。`coordinator_` delegate を削除。**RuntimeWorldAuthority に `Store` 型を導入し、`friend Owner = RuntimeWorldAuthority` に変更**。<br>**RuntimeWorldAuthority の初期化変更**: `AudioEngine.CtorDtor.cpp:28` の `worldAuthority_(runtimePublicationBridge_)` を Store 所有の形に変更 | ビルド |
| X4-7 | **`RuntimeWorldAuthority::publish()` 導入（🔴 十九次レビューで独立させた重要ステップ）**: commit + publishAndSwap を束ね、**commit-before-swap ordering をここで固定**。retire は publish() に戻さない（Lifetime の責務） | ビルド |
| X4-8 | **`publishWorld()` 一時生成削除 + PublishExecutor 接続**: `makeRuntimePublicationCoordinator().publishWorld()`（RuntimePublishExecutor.h:55-56）を廃止し `authority.publish()` に一本化。<br>**Bootstrap**（AudioEngine.Init.cpp:53-54）: 同期 publish を authority 経由に。<br>**shutdown clear**（ReleaseResources.cpp:436-438）: `authority.clearPublishedRuntimeSnapshotsNonRt()` へ（🔴 二十次 §15: `publish()` と統合せず、**同じ physical write authority の下に別 API として置く**） | ビルド |

**🔴🔴🔴 二十次レビュー（2026-08-09）— X4-B 実装を細分化（§20・rollback point 多数確保）**:
X4-5〜X4-8 は大きすぎる。**X4-B をさらに細分化**し、各ステップを rollback point にする:
```
X4-B-0  Baseline 固定（Debug/Release/CTest/ISR semantic/ISR soak/shutdown soak 全 PASS）
X4-B-1  RuntimeStore<World, Owner> の Owner を RuntimeWorldAuthority に変更
        （まだ Store の物理位置は移さない。コンパイル可能な最小構造）
        🔴 二十一次レビュー + コンパイル検証: RuntimeStore に `using OwnerType = Owner;`
        を追加（Test 1 の static_assert 用）。`RuntimeStore<World, Self>` の CRTP は
        既存（RuntimePublicationCoordinator.h:34）でコンパイル実績あり。g++ -std=c++20
        で検証済み（friend Owner / is_class_v / member WriteAccess 全て COMPILE_OK）
X4-B-2  Authority に Store を移動（AudioEngine └── worldAuthority └── runtimeStore）
X4-B-3  Authority が WriteAccess を取得（writeAccess_ = runtimeStore_.acquireWriteAccess()）
X4-B-4  RuntimeWorldAuthority::publish() を導入（既存 publishWorld() と比較できる状態を維持）
X4-B-5  PublishExecutor を切替（makeRuntimePublicationCoordinator() → worldAuthority.publish()）
X4-B-6  Bootstrap を切替
X4-B-7  Shutdown clear を切替（clearPublishedRuntimeSnapshotsNonRt → authority 経由）
X4-B-8  旧 makeRuntimePublicationCoordinator() を production path から削除
X4-B-9  Read API migration（getCurrent() でなく専用 read API — observePublishedWorld / acquireReadToken / consumeWorldHandle(ReadToken)）
X4-B-10 Architecture tests（Test 1-10）
X4-B-11 Full regression / soak
```
- **各ステップでビルド・テストを通過**させる（X4-B-1 はコンパイル最小・X4-B-4 は既存 publishWorld と並存）
- **🔴🔴🔴 十八次別視点7調査（2026-08-09）— X4-B-9 の read migration 対象を実コードで確定**: `makeRuntimeReadHandle`（AudioEngine.h:3099-3139）の read path は **`acquireReadToken(runtimeStore)` + `consumeWorldHandle(runtimeStore, readToken)`（:3116-3117）** で physical Store read を行う。加えて **generation/sequence の単調性監視**（:3128-3135）を実行する:
  ```cpp
  // AudioEngine.h:3116-3139（makeRuntimeReadHandle 内）
  const auto readToken = RuntimePublicationCoordinator::acquireReadToken(runtimeStore);   // :3116
  const auto* world = RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore, readToken);  // :3117
  ...
  // 単調性監視（observeLastSeenGeneration_[slot] / observeLastSeenSequenceId_[slot]）
  const bool generationBackward = (previousGeneration != 0 && currentGeneration < previousGeneration);  // :3128
  const bool sequenceBackward = (previousSequence != 0 && currentSequence < previousSequence);          // :3129
  if (generationBackward || sequenceBackward) { observeMonotonicViolationCount_++; ... }                 // :3131-3135
  ```
  - **X4-B-9 の置換対象**: `acquireReadToken(runtimeStore)` / `consumeWorldHandle(runtimeStore, readToken)` を `worldAuthority().acquireReadToken()` / `worldAuthority().consumeWorldHandle(token)` に置換（専用 read API）
  - **単調性監視（observeLastSeenGeneration_ / observeLastSeenSequenceId_ / observeMonotonicViolationCount_）は AudioEngine 側の state で維持**（X4-B の read migration でも AudioEngine に残る。Authority に移さない）
  - **呼び出し元**: makeRuntimeReadHandle は BlockDouble.cpp:152 / Snapshot.cpp:27 / Timer.cpp:374,1593 / CtorDtor.cpp:72,121 / PrepareToPlay.cpp:136 / ReleaseResources.cpp:128 / Learning.cpp:127 など多数。X4-B-9 はこれらを一括置換する
- **X4-B-9 の read migration は `getCurrent()` でなく専用 API**（レビュー §21）

**🔴🔴🔴 二十次レビュー（2026-08-09）— Bootstrap / shutdown clear は `publish()` と統合しない（§15）**:
shutdown clear（`clearPublishedRuntimeSnapshotsNonRt`）は内部的に `writeAccess_.publishAndSwap(nullptr)` を実行するが、これは通常の「owner → new RuntimeWorld → publish」とは**意味が異なる**（null 公開 = world クリア）。したがって:
```
RuntimeWorldAuthority
  ├─ publish()                          ← 通常 publish（owner → swap → return oldWorld）
  └─ clearPublishedRuntimeSnapshotsNonRt() ← shutdown clear（null swap・別 semantic）
```
- **`authority.publish(owner, metadata)` に shutdown clear を無理に統合しない**（null owner の publish は意味が曖昧）
- **同じ physical write authority（RuntimeWorldAuthority）の下に置く**が、semantic operation は分離（shutdown semantic は X3 の責務）
- **Bootstrap**（AudioEngine.Init.cpp:53-54）は「owner → new World → publish」の通常 semantic のため `authority.publish()` を使用してよい
| X4-9 | **全 direct `publishWorld()` caller 排除（🔴 十九次レビューで追加）**: Bootstrap / shutdown clear を含む全 publishWorld 直接呼び出しが Authority 経由であることを確認 | ビルド |
| X4-10 | **architectural tests**: `publishAndSwap` caller 検証 + **Test 7（commit-before-swap）/ Test 9（dual-pointer consistency）/ Test 10（INV-X4-7）**（Test 1-10） | CTest |
| X4-11 | **full regression + shutdown soak**: P2 → X5 → X6 → X2 → X1 → X4 → X3 の累積状態で検証 | 全テスト PASS |

---

### 6.4-X4 の最終形（17次レビュー確定）

```
                         ┌─────────────────────┐
                         │ RuntimeIntent        │
                         │ Coordinator          │
                         │ (Intent only)        │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ PublicationAdmission│
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ PublishExecutor     │
                         │ (ownership transfer)│
                         └──────────┬──────────┘
                                    ▼
                    ╔══════════════════════════════╗
                    ║ RuntimeWorldAuthority       ║
                    ║   SINGLE WRITE AUTHORITY    ║
                    ║ ┌────────────────────────┐ ║
                    ║ │ RuntimeStore           │ ║
                    ║ │ Owner = this           │ ║
                    ║ │ ┌────────────────────┐ │ ║
                    ║ │ │ WriteAccess        │ │ ║
                    ║ │ │ publishAndSwap()   │ │ ║
                    ║ │ └────────────────────┘ │ ║
                    ║ └────────────────────────┘ ║
                    ║ OwnerChannel  LifetimeState ║
                    ╚══════════════════════════════╝
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ RuntimeWorld        │
                         │ atomic publication  │
                         └─────────────────────┘
```

**最終判定**: 「名前の二重化を解消するだけ」の旧案は採用しない。**X4 = naming convergence + physical ownership convergence** として扱う。最終合格条件:
```
INV-X4-3: RuntimeStore::publishAndSwap()
    ↓ exactly one
    ↓ RuntimeWorldAuthority-owned WriteAccess
```
ISR の「Authority Singularization」と一致。RT 側に新たな write authority を増やさず、Publish execution を `PublishExecutor → RuntimeWorldAuthority` に一本化するため、Practical Stable ISR Runtime の原則（Coordinator が唯一の Authority / RT で publish/retire/delete を増やさない）とも整合。

**実装上の最大リスク**: X4-B の `RuntimeStore<World, Owner>` Owner 変更。`RuntimeStore` / `RuntimeWorldAuthority` / `RuntimePublishAuthority` / `RuntimePublishExecutor` の4点を一つの ownership migration として実装・検証する。

### 6.5 X5 — Publish Intent residency 専用 counter（P1 優先）— **十四次レビュー GO（そのまま採用可能）**

> **★ 実コード照合確定（2026-08-10）**: `enqueuePublicationIntent` は ISRRuntimePublicationCoordinator.h:286（dash 記載 :273 と行番号差）。reservation→push→rollback をここに追加（全 3 経路が単一箇所に集約）。processIntent の intentQueue_ Publish pop で fetchSub（quarantineFallbackQueue_ は Quarantine 専用 → Publish 分岐不要）。admission（PublicationAdmission::evaluate）は counter 非影響（Accepted 後の enqueue で増加）。isFullyDrained に == 0 追加。INV-X5-1: residency + producer reservation。

**現状**: `publicationBacklogCount_` は hasDeferredCommit 由来で、実際の intentQueue_ 内 Publish 件数ではない。

**🔴🔴🔴 十四次レビュー（2026-08-09）— GO 判定**: X 中最も完成度が高い。`publicationBacklogCount_` が Publish Intent Queue の件数ではないことの認識と、`publicationIntentResidencyCount_` の新設方針は正しい。`publicationIntentResidencyCount_ ≠ deferredPublicationCount_ ≠ hasDeferredCommit` の分離も ISR semantic model と一致。**そのまま採用可能**。

**设计方針**: `publicationIntentResidencyCount_` を新設:
```
Publish producer: reserve → publicationIntentResidencyCount_++ → intentQueue_.push
                  success → 維持 / failure → publicationIntentResidencyCount_--
Publish consumer: intentQueue_.pop(Publish) → publicationIntentResidencyCount_--
```
**完全分離**:
```
Observe → pendingIntentCount_ / Quarantine → pendingIntentCount_ / Recovery → pendingIntentCount_
Publish → publicationIntentResidencyCount_
```

**🔴🔴🔴 追加調査（2026-08-09）— 宣言位置を確定（ISRRuntimePublicationCoordinator.h:383 の隣）**:
`publicationIntentResidencyCount_` は、既存の `publicationBacklogCount_`（ISRRuntimePublicationCoordinator.h:383）の直後に宣言する:
```cpp
// ISRRuntimePublicationCoordinator.h — atomic counter メンバ群（:380-388）
std::atomic<std::uint64_t> retireBacklogCount_;            // :382
std::atomic<std::uint64_t> publicationBacklogCount_;       // :383
std::atomic<std::uint64_t> publicationIntentResidencyCount_; // ★ X5 新設（publicationBacklogCount_ の隣）
std::atomic<std::uint64_t> pendingIntentCount_;            // :384
```
- 宣言位置は `publicationBacklogCount_`（:383）の隣（Publish 系 counter を一箇所に集約）
- `enqueuePublicationIntent`（:273-278）で reservation→push→rollback（下記 十二次レビュー）

**🔴🔴🔴 十八次別視点調査（2026-08-09）— Publish intent の全 enqueue 経路（3経路・実コード検証）を確定**:

Publish Intent は `enqueuePublicationIntent`（ISRRuntimePublicationCoordinator.h:273）に到達する前に、**3つの Producer 経路**がある。X5 の `publicationIntentResidencyCount_` は全経路で正しく +1 されることを確認:
```
経路1（通常 rebuild・RebuildThread）:
  RebuildDispatch.cpp:1138 → enqueuePublicationIntentForRuntimeCommit（AudioEngine.Commit.cpp:688）
    → runtimeOrchestrator_->submitPublishRequest（Orchestrator.cpp:316）
    → trySubmitImpl（Orchestrator.cpp:40, RebuildThread 上で同期 build/publish）
    → executor_.publish（PublicationExecutor.cpp:8 → commitRuntimePublication :4450）
    → enqueueRuntimePublicationFireAndForget → enqueuePublicationIntent（AudioEngine.h:4413）

経路2（Recovery publish・Builder Loop）:
  RebuildDispatch.cpp:971 → enqueuePublicationIntentForRuntimeCommit（:688）
    → 経路1 と同じ（submitPublishRequest → trySubmitImpl → executor_.publish → enqueuePublicationIntent）

経路3（deferred 再 enqueue・RebuildThread）:
  Orchestrator.cpp:525（processDeferredAdmission）→ submitPublishRequest（:316）
    → trySubmitImpl → executor_.publish → enqueuePublicationIntent
```
- **X5 の counter は `enqueuePublicationIntent` の単一箇所で reservation される**（3経路すべてがここに集約）。**二重計上なし**（各 publish は 1 回だけ enqueue）
- **`enqueuePublicationIntentForRuntimeCommit` は Publish intent を直接 push しない**（submitPublishRequest 経由）。X5 の reservation は `enqueuePublicationIntent`（:273）に追加するため、この共通パスに手を入れる必要はない
- **`commitRuntimePublication`（AudioEngine.h:4431-4458）**: `enqueueRuntimePublicationFireAndForget`（:4443）を呼び、内部で `enqueuePublicationIntent`（:4413）→ queue full なら ownerChannel().take で回収。**X5 の reservation は :4413 で実施され、queue full 時に :4419 の rollback と整合**する

**🔴🔴 十二次レビュー（2026-08-09）— 具体的な挿入位置（inline 関数の変更）を確定**:
`enqueuePublicationIntent`（ISRRuntimePublicationCoordinator.h:273-278）は **inline 関数**で `intentQueue_.push(prepared)` のみを行う。X5 ではここに reservation→push→rollback を追加:
```cpp
[[nodiscard]] bool enqueuePublicationIntent(const Intent& intent) noexcept
{
    Intent prepared = intent;
    prepared.type = IntentType::Publish;
    convo::fetchAddAtomic(publicationIntentResidencyCount_, 1);  // ① reservation
    if (intentQueue_.push(prepared))
        return true;                                            // ② success → 維持
    convo::fetchSubAtomic(publicationIntentResidencyCount_, 1);  // ③ failure → rollback
    return false;
}
```
- **🔴🔴🔴 十八次調査（2026-08-09）— メモリオーダリングの規約（AtomicAccess.h 検証）**: `fetchAddAtomic` / `fetchSubAtomic` は **default `std::memory_order_acq_rel`**（AtomicAccess.h:91-105）。X5 の新設 counter も **default の acq_rel を明示指定**する:
  ```
  enqueuePublicationIntent（Producer = NonRT publish スレッド）:
    fetchAddAtomic(publicationIntentResidencyCount_, 1, std::memory_order_acq_rel)
    → push 前に reservation を公開（consumer の acquire と HB）
  processIntent pop（Consumer = CoordinatorLoop）:
    fetchSubAtomic(publicationIntentResidencyCount_, 1, std::memory_order_acq_rel)
    → pop 後に reservation を解放（次の enqueue の acquire と HB）
  isFullyDrained 読み（NonRT）:
    consumeAtomic(publicationIntentResidencyCount_, std::memory_order_acquire)
  ```
- **ordering の理由**: reservation（+1）を release で公開し、pop 減算（-1）を acquire で観測することで、**「queue 内 Publish 数」の一貫した view** が得られる。relaxed では isFullyDrained が古い値を読む可能性
- **fetchAdd/fetchSub は RMW のため、Producer（NonRT publish）と Consumer（CoordinatorLoop）の2スレッド間でも race なし**
- **🔴🔴🔴 十八次別視点5調査（2026-08-09）— producer hole と counter の整合（MpscBoundedRing 検証）**:
  `intentQueue_` は `MpscBoundedRing`（MpscBoundedRing.h:43-136）。**push() は reservation（CAS :76）→ payload 書込み（:80）→ seq release（:81）の2段階**で、reservation と publication の間に**別 Producer が割り込む（producer hole）**可能性がある。X5/X6 counter との整合:
  ```
  enqueuePublicationIntent:
    fetchAdd(publicationIntentResidencyCount_, +1)   // ① counter 増加（push 前）
    intentQueue_.push(prepared)                       // ② push（reservation→publication）
    → success: counter 維持 / failure: fetchSub(-1)  // ③ rollback
  ```
  - **counter は「reservation 先行」で増加**するが、push は **publication 完了後に return true/false** する（MpscBoundedRing.h:80-82）。したがって:
    - **push 成功時点で producer hole は解消済み**（publication 完了）→ counter は「queue 内実要素数」と同期
    - **push 失敗時**: counter を rollback（fetchSub）→ 一時的な過大計上は解消
  - **REPAIR_PLAN2.md:174(c) の整合確認**: 「submitXxx は push() 成功後に pendingIntentCount_ + 1 する」という既存記述は、**X5 の fetchAdd 先行設計とは順序が異なる**。X5 は「fetchAdd 先行 → push → 失敗 rollback」のため、**producer hole 中に counter が一瞬過大**になるが、push の return 後は必ず収束する（成功: 要素数と一致 / 失敗: rollback）
  - **isFullyDrained の観点**: producer hole による一時的過大計上は、CoordinatorLoop の pop（次 tick）で解消される。**push が return する前に isFullyDrained が評価されても、producer hole は push 内で完結**するため（push は同期）、永続的な不整合は生じない
- consumer 側: processIntent の `while (intentQueue_.pop)` で `commonIntent.type == Publish` の場合に `publicationIntentResidencyCount_--`（pendingIntentCount_ とは独立）
- **Publish の reservation と pendingIntentCount_ の reservation は完全独立**（Publish pop は pendingIntentCount_ を触らない — 七次 W2 / §1.1.6 と整合）
**deferred Publish との分離**: `publicationIntentResidencyCount_`（Intent Queue 内の Publish Intent）≠ `deferredPublicationCount_`（Orchestrator の deferred state）≠ `hasDeferredCommit`（Commit 未完了の logical state）。**Queue residency / Deferred state / Commit completion の3つを混ぜない**。

**🔴🔴🔴 十八次別視点4調査（2026-08-09）— deferred は単一スロット（実コード検証で確定）**:
`RuntimePublicationOrchestrator::enqueueDeferred`（Orchestrator.cpp:360-409）は **`deferredSlot_`（単一スロット）+ `hasDeferred_`（atomic<bool>）** で管理する:
```cpp
// RuntimePublicationOrchestrator.cpp:383-400
deferredSlot_ = DeferredPublishSlot{ .request = req, .guard = {...}, .metadata = {...}, ... };
convo::publishAtomic(hasDeferred_, true, std::memory_order_release);
```
- **deferred publish は同時に1つだけ**（単一スロット）。上書き時は `deferredOverwriteCount_++`（:365）
- **`deferredPublicationCount_` は 0 or 1 のみ**（単一スロットのため）
- **X5 の分離設計の確定**: `publicationIntentResidencyCount_`（複数あり得る）と `deferredPublicationCount_`（0/1）は**完全に別 semantic**。`hasDeferred_`（atomic<bool>）も別（logical state）
- **deferred 再 enqueue 経路**: `processDeferredAdmission`（Orchestrator.cpp:502-535）が `submitPublishRequest`（:525）で再 enqueue → `trySubmitImpl` → `executor_.publish` → `enqueuePublicationIntent`（経路3）。**deferred の publish intent も X5 の counter を +1 する**（再 enqueue 時に）

**🔴🔴🔴 十一次/十二次レビュー追記 — processIntent の Publish pop 減算の具体的実装**:
processIntent（ISRRuntimePublicationCoordinator_ProcessIntent.cpp:36-37）の while ループで、Publish の pop 減算を追加:
```cpp
while (intentQueue_.pop(commonIntent)) {
    if (commonIntent.type == IntentType::Publish)
        convo::fetchSubAtomic(publicationIntentResidencyCount_, 1);  // X5: Publish 専用 counter 減算
    else
        convo::fetchSubAtomic(pendingIntentCount_, 1);               // 既存 §1.1: Observe/Quarantine/Recovery
    kDispatchTable[static_cast<std::size_t>(commonIntent.type)]->handle(commonIntent, ctx);
}
```
- **🔴🔴🔴 追加調査（2026-08-09）— fallback queue の Publish 有無を確定**: `quarantineFallbackQueue_`（:32-33）は **Quarantine Intent 専用**（submitQuarantine の intentQueue_ full 時の退避先、ISRRuntimePublicationCoordinator.cpp:722）。**Publish Intent は fallback queue に入らない**。したがって:
  - `publicationIntentResidencyCount_` の減算は **intentQueue_ の while ループ（:36-37）のみ**
  - `quarantineFallbackQueue_` の while ループ（:32-33）では Publish 分岐不要（Quarantine のみ → pendingIntentCount_ 減算）
- **Publish の PublishExecutor 実行**（PublishIntentHandler → executePublish → commit → onPublishCommitted）は、pop 減算とは独立。`publicationIntentResidencyCount_` は「queue 内の Publish Intent 数」であり、completion（X2）とは別 semantic
- **🔴🔴🔴 十八次別視点9調査（2026-08-09）— admission 判定と counter の関係を確定**: `PublicationAdmission::evaluate`（PublicationAdmission.cpp:6-61）は publish enqueue 前に5段階の admission を実行する:
  ```
  1. Shutdown check（:11）→ RejectedShutdown
  2. Generation staleness（:15-18）→ RejectedStaleGeneration
  3. DSP finalized check（:21-22）→ RejectedNotFinalized
  4. HealthState check（:25-37）→ RejectedPressure（Critical/Degraded）
  5. Pressure/throttle（:40-48）→ RejectedPressure
  6. Fading active → Deferred（:51-58）→ DeferredFadingActive
  ```
  - **X5 の counter は admission Accepted 後の enqueue で増加**する（enqueuePublicationIntent）
  - **Rejected / Deferred の publish は counter に影響しない**（enqueuePublicationIntent に到達しないため）
  - **Deferred は単一スロット（Orchestrator.cpp:360-409）**: deferred 化された publish は再 enqueue（processDeferredAdmission :525）時に counter +1。**admission 判定（DeferredFadingActive）は counter に直接影響しない**（再 enqueue 後の counter 増加のみ）
- **`publicationIntentResidencyCount_` のクリア**: isFullyDrained では `== 0` を要求（§6.7）。producer quiescence 後は queue 内 Publish が無くなり 0 に収束
- **🔴🔴🔴 十八次別視点6調査（2026-08-09）— shutdown 時の収束を確定**: CoordinatorLoop（ISRCoordinatorLoop.cpp:31-43）は `isShutdownInProgress()` が false の間 processIntent を続行する。requestShutdown（releaseResources:75 / ~AudioEngine:102）から join（shutdownCoordinatorLoop）までの間に intentQueue_ を drain するため、**publicationIntentResidencyCount_ は shutdown 中に 0 へ収束**する。shutdownCoordinatorLoop の join（:189/:110）後に queue 内 Publish が残っていれば isFullyDrained の `publicationIntentResidencyCount == 0` が false → shutdown 完了不可（正常動作：残留 Publish は drain 待ち）

**不変条件**: `INV-X5-1: publicationIntentResidencyCount = Publish intent queue residency + producer reservation`（並行中は `>=` 物理可視、producer quiescence 後は `==`）。

### 6.6 X6 — Quarantine Intent / Resident semantic 分離（P1 優先）

> **★ 実コード照合確定（2026-08-10）**: quarantineResidentCount_ の実測上書き（AudioEngine.Threading.cpp:131: ringResident + dspQuarantineResident）を確認 — drain aggregate としての現状。3 層分離（quarantineIntentResidencyCount_ / quarantineRingResidencyCount_ / quarantineResidentCount_）で aggregate 混入を解消。retireQuarantineResidentCount_ は新 counter 不要（既存 RetireQuarantineStore::residentCount() を使用 — 十八次別視点3）。DSPQuarantineManager::residentCount() が唯一 source。

**現状**: `quarantineResidentCount_` は「実際に quarantine lane に存在する DSP 数」（ISRRetireRuntimeEx.cpp:219,222,237）だが、submitQuarantine（:713,723）が Intent の意味で +1 している（semantic contamination）。

**🔴🔴🔴 十四次レビュー（2026-08-09）— 3 semantic に分離（source-of-truth 記述の矛盾を解消）**:

dash 旧案は「`DSPQuarantineManager::residentCount()` が唯一の source of truth」としながら、isFullyDrained で `quarantineResidentCount_ = ringResident + dspQuarantineResident`（Threading.cpp:131）と設定する矛盾があった。**厳密には `quarantineResidentCount_` は drain aggregate であり、`DSPQuarantineManager::residentCount()` 単独ではない**。

**推奨: 3 semantic に完全分離**:
```cpp
std::atomic<uint64_t> quarantineIntentResidencyCount_;  // Quarantine Intent が transport lane（intentQueue_/quarantineFallbackQueue_）に存在する数
std::atomic<uint64_t> quarantineRingResidencyCount_;    // quarantine fallback / ring の残留（transport）
std::atomic<uint64_t> quarantineResidentCount_;         // DSPQuarantineManager が実際に保持する DSP 数（唯一 source）
```
**`quarantineResidentCount_` という名前を aggregate drain counter として使うのは避ける**（十四次指摘）。名前は「実在 DSP 数」に限定し、ring residency は別 counter に。

**🔴🔴🔴 追加調査（2026-08-09）— 宣言位置を確定（ISRRuntimePublicationCoordinator.h:388 の隣 + RetireQuarantineStore）**:
`quarantineIntentResidencyCount_` / `quarantineRingResidencyCount_` は、既存の `quarantineResidentCount_`（ISRRuntimePublicationCoordinator.h:388）の直後に宣言する:
```cpp
// ISRRuntimePublicationCoordinator.h — atomic counter メンバ群（:380-388）
std::atomic<std::uint64_t> quarantineResidentCount_;      // :388 既存（Coordinator 側の実測カウンタ）
std::atomic<std::uint64_t> quarantineIntentResidencyCount_; // ★ X6 新設（Intent lane residency）
std::atomic<std::uint64_t> quarantineRingResidencyCount_;   // ★ X6 新設（ring/fallback 残留）
```
`retireQuarantineResidentCount_` は **`RetireQuarantineStore`（RetireQuarantineStore.h:60）の既存 `residentCount()` を source of truth に使用**する:
```cpp
// RetireQuarantineStore.h — 4 semantic の4番目
// 🔴🔴🔴 十八次別視点3調査（2026-08-09）: 新 atomic counter は追加しない（確定）
//   RetireQuarantineStore は既に size_（:175, mutex 保護）を持ち residentCount()（:157-161）で公開。
//   quarantine()（:69-93）で ++size_ / drain()（:98-131）で size_=w / drainAllUnsafe()（:134-154）で size_=0。
//   → 既存 residentCount() を source of truth にし、新 counter 不要。
// isFullyDrained では: ISRRetireRouter::retireQuarantineStore().residentCount() == 0 を評価
```
- **Coordinator 側（ISRRuntimePublicationCoordinator）**: quarantineIntentResidencyCount_ / quarantineRingResidencyCount_（新設）
- **RetireQuarantineStore 側**: **新 counter は追加しない**。既存 `residentCount()`（:157）を source of truth にする（`quarantine()` で ++size_ / `drain()` で size_ 減少 / `drainAllUnsafe()` で size_=0）
- **🔴🔴🔴 十八次別視点3調査（2026-08-09）— RetireQuarantineStore の実装詳細（既存 size_ で十分）**:
  ```
  quarantine()（:69-93）: 退避成功時 ++size_（:91）。store full 時 ++overflowCount_（:82, deleter 実行禁止）
  drain()（:98-131）: epoch 安全（isOlderFn）なエントリのみ deleter 実行。size_ = w（:126）
  drainAllUnsafe()（:134-154）: size_ = 0（:150）。Audio Thread 停止後のみ
  residentCount()（:157-161）: size_ を返す（mutex 保護・NonRT のみ）
  ```
  - **RT スレッドからは参照されない**（RetireQuarantineStore.h:52-54「全操作は NonRT」）。`residentCount()` を isFullyDrained（NonRT）で読むのは安全
  - **isFullyDrained の評価**: `m_retireRouter->retireQuarantineStore().residentCount() == 0`（ISRRetireRouter.h:198 の m_retireQuarantine を介して）
  - **新 counter を追加しない理由**: RetireQuarantineStore は mutex 保護の `size_` を唯一の実体数として持つ。atomic counter を追加すると**2重の source になり不整合リスク**（X6 の INV-X6-4「no counter may represent both」の精神に反する）。既存 `residentCount()` が唯一の source

**🔴🔴🔴 十六次レビュー（2026-08-09）— 4 semantic に完全分離（対象コンテナを確定）**:

`quarantineRingResidencyCount_` の対象コンテナは **追加調査（2026-08-09）で確定: `quarantineFallbackQueue_`（ISRRuntimePublicationCoordinator.h:453-454, MpscBoundedRing, kQuarantineFallbackCapacity=1024）**。**以下の4つのコンテナを別 semantic として扱う**:
```
quarantineIntentResidency   = intentQueue_ + quarantineFallbackQueue_（Quarantine Intent transport）
quarantineRingResidency     = quarantine 専用 transport ring（RetireQuarantineStore とは別）
quarantineResident          = DSPQuarantineManager が実際に保持する DSP オブジェクト
retireQuarantineResident    = RetireQuarantineStore（RetireQuarantineStore.h:60, kMaxQuarantinedEntries=512）
                              に退避された retire 対象オブジェクト
```
- **`quarantineFallbackQueue_` と `RetireQuarantineStore` を同じ counter に入れてはいけない**（十六次 §17）
- `RetireQuarantineStore::quarantine()`（:69）は retire 対象を退避、`drain()`（:77）で epoch 安全後に削除、`residentCount()`（:157）を保持

**対象コンテナのマッピング（🔴🔴🔴 追加調査 2026-08-09 で quarantine 専用 ring の実体を確定）**:
| コンテナ | 意味 | 管理 counter |
|---------|------|-------------|
| `intentQueue_`（MpscBoundedRing, kIntentQueueCapacity=4096） | Quarantine Intent transport（primary） | `quarantineIntentResidencyCount_` |
| `quarantineFallbackQueue_`（MpscBoundedRing, kQuarantineFallbackCapacity=1024, ISRRuntimePublicationCoordinator.h:453-454） | **quarantine 専用 transport ring（実体確定）** — intentQueue_ full 時の退避先 | `quarantineRingResidencyCount_` |
| `DSPQuarantineManager`（AudioEngine.h:4698） | 実在 quarantine DSP | `quarantineResidentCount_`（source: DSPQuarantineManager::residentCount(), ISRDSPQuarantine.cpp:103） |
| `RetireQuarantineStore`（ISRRetireRouter.h:198） | retire 対象の quarantine 退避 | **新 counter 不要（十八次別視点3確定）**: 既存 `residentCount()`（RetireQuarantineStore.h:157, size_ :175）を source of truth に。isFullyDrained は `retireQuarantineStore().residentCount() == 0` を評価 |

- **🔴🔴🔴 十八次別視点14調査（2026-08-10）— 全 transport queue の capacity を確定（実コード検証）**:
  ```
  intentQueue_                 MpscBoundedRing, kIntentQueueCapacity = 4096        （:445-446）
  quarantineFallbackQueue_     MpscBoundedRing, kQuarantineFallbackCapacity = 1024 （:453-454）
  recoveryIntentQueue_         LockFreeRingBuffer, kRecoveryIntentQueueCapacity = 256 （:433-434）
  observeDeferredRing_         LockFreeRingBuffer, kObserveDeferredRingCapacity = 1024（:429-430）
  ```
  - **intentQueue_（4096）は Observe/Publish/Recovery/Quarantine の共通 queue**（MPSC）
  - **quarantineFallbackQueue_（1024）は Quarantine 専用退避**（X6 の quarantineRingResidencyCount_ 対象）
  - **recoveryIntentQueue_（256）は Recovery 専用 SPSC**（X1 の durable admission 対象）
  - **observeDeferredRing_（1024）は Observe 専用 overflow**（P2 の pendingIntentCount_ 減算対象 — 十八次別視点2）

- **`quarantineRingResidencyCount_` の対象 = `quarantineFallbackQueue_`（:454）と確定**。`intentQueue_` 残留は quarantineIntentResidencyCount_、fallback 残留は quarantineRingResidencyCount_ に分離する
- **processIntent の drain 順序**: quarantineFallbackQueue_ を intentQueue_ より先に drain（ISRRuntimePublicationCoordinator_ProcessIntent.cpp:32-33）— fallback 残留は pop された時点で quarantineRingResidencyCount_--、その後 handler 実行で quarantineResidentCount_++（manager 内部）
- **submitQuarantine の enqueue 経路**（ISRRuntimePublicationCoordinator.cpp:712-726）: intentQueue_ 成功 → quarantineIntentResidencyCount_++ / full → quarantineFallbackQueue_ → quarantineRingResidencyCount_++

**状態遷移**:
```
submitQuarantine → quarantineIntentResidencyCount_++（intent queue へ）
    → pop（invalid/stale → intent counter-- / handler → QuarantineService）
    → stateChanged=true → DSPQuarantineManager::quarantineHandle → quarantineResidentCount_++（manager 内部）
reclaim → quarantineResidentCount_--（manager 内部）
retire 対象の quarantine 退避 → RetireQuarantineStore::quarantine() 成功 → size_++（residentCount() で観測）
RetireQuarantineStore drain（epoch 安全） → size_ = w（residentCount() で観測。deleter 実行）
RetireQuarantineStore drainAllUnsafe（Audio 停止後） → size_ = 0
```
- **🔴🔴🔴 十八次別視点6調査（2026-08-09）— shutdown 時の quarantine counter の整合（releaseResources 検証）**: releaseResources（ReleaseResources.cpp:363-404）の shutdown quarantine 全解放を実コード検証:
  ```
  unquarantineAllReaders(:367) → RetireQuarantineStore drainAllUnsafe(:376, size_=0)
  → for slot: dspQuarantineManager_.destroyForShutdown(slot)(:387)
    → dspHandleRuntime_.destroyQuarantineSlot(slot,0)(:390)
    → worldAuthority_.lifetime().reclaim(slot)(:392, quarantineResidentCount-- 相当)
  ```
  - **X6 の quarantineResidentCount_（= DSPQuarantineManager::residentCount()）は destroyForShutdown で自然に 0** になる（quarantineActiveFlags_ がクリアされる）
  - **RetireQuarantineStore::residentCount() は drainAllUnsafe（:376）で 0**（size_ = 0）
  - **isFullyDrained の整合**: shutdown 後、`quarantineResidentCount == 0`（DSPQuarantineManager）/ `retireQuarantineStore.residentCount() == 0`（RetireQuarantineStore）が成立
  - **X6 counter は shutdown で自然収束**する（新たな shutdown 専用 counter 処理は不要）
- **🔴🔴🔴 十八次別視点10調査（2026-08-09）— destroyForShutdown の実装詳細を確定**: `DSPQuarantineManager::destroyForShutdown`（ISRDSPQuarantine.cpp:130-155）は:
  ```cpp
  bool DSPQuarantineManager::destroyForShutdown(uint32_t slot) {
      if (slot >= kMaxSlots) return false;
      bool active = consumeAtomic(quarantineActiveFlags_[slot], acquire);  // :135
      if (!active) return false;
      publishAtomic(quarantineActiveFlags_[slot], false, release);         // :141 RT 側フラグ解除
      { std::lock_guard<std::mutex> lock(auditMutex_);                     // :145
        for (auto& entry : auditLog_) {                                    // :146
            if (entry.slot == slot && !entry.resolved) { entry.resolved = true; break; }
        }
        compactAuditLogLocked();                                           // :152
      }
      return true;
  }
  ```
  - **quarantineActiveFlags_[slot] を false にすることで residentCount()（:103-111 の走査）が自然に減る**（X6 の quarantineResidentCount_ = DSPQuarantineManager::residentCount() が 0 へ）
  - **auditLog の未解決エントリを resolved に + compactAuditLogLocked（:152）で compaction**（memory 管理）
  - **🔴🔴🔴 十八次別視点12調査（2026-08-09）— compactAuditLogLocked の実装詳細を確定**: `compactAuditLogLocked`（ISRDSPQuarantine.cpp:158-172）は:
    ```cpp
    void DSPQuarantineManager::compactAuditLogLocked() noexcept {
        constexpr size_t kCompactThreshold = 1024;      // :161 compaction 閾値
        if (auditLog_.size() < kCompactThreshold) return;  // :162 1024未満は skip
        auto it = auditLog_.begin();
        while (it != auditLog_.end() && it->resolved) ++it;  // :167-169 先頭の resolved を走査
        if (it != auditLog_.begin())
            auditLog_.erase(auditLog_.begin(), it);      // :170-171 先頭の resolved 連続を削除
    }
    ```
    - **compaction は resolved エントリが1024超えた場合のみ**（:161-162）実行。メモリ管理の効率化
    - **auditLog_ は vector**（append-only、resolved マーク + compaction で削除）
    - **X6 の設計で auditLog への新規介入は不要**（destroyForShutdown / compactAuditLog が既存で管理）
  - **X6 の設計で新たな処理は不要**: destroyForShutdown が quarantineActiveFlags_ と auditLog の両方をクリアするため、quarantineResidentCount_ は自然収束
**4つの counter は同じイベントで同時に増えるとは限らない**（Intent pending で resident=0、Intent=0 で resident=1 の両状態が可能）。絶対に一つの counter に統合しない。

**🔴🔴🔴 十六次レビュー（2026-08-09）— semantic contamination の是正を明示**:
現行 `submitQuarantine()`（:713,723）が Intent を queue に入れる段階で `quarantineResidentCount_++` するのは、**「Intent exists」と「DSP is actually quarantined」を同じ counter にしているため誤り**。X6 では:
```
submitQuarantine       → quarantineIntentResidencyCount_++（Intent counter）
QuarantineService success → DSPQuarantineManager::quarantineHandle → resident++（manager 内部）
reclaim                → DSPQuarantineManager resident--（manager 内部）
```
- **`quarantineResidentCount_` は `DSPQuarantineManager::residentCount()`（source of truth）のみ**が管理
- **`submitQuarantine` の `quarantineResidentCount_` +1（:713,723）は X6 で撤去**（Intent counter と分離）
- **🔴🔴🔴 十八次調査（2026-08-09）— residentCount() の実装と整合を確定**: `DSPQuarantineManager::residentCount()`（ISRDSPQuarantine.cpp:103-111）は **`quarantineActiveFlags_[kMaxSlots]` を for ループで走査**して真数:
  ```cpp
  size_t DSPQuarantineManager::residentCount() const noexcept {
      size_t count = 0;
      for (const auto& flag : quarantineActiveFlags_) {
          if (convo::consumeAtomic(flag, std::memory_order_acquire)) ++count;
      }
      return count;
  }
  ```
  - **`quarantineActiveFlags_[slot]` は `quarantineHandle`（:30, release）で true、`reclaimSlot` で false**（ISRDSPQuarantine.cpp:49-80）。**Coordinator の quarantineResidentCount_ とは独立した Epoch ドメイン内フラグ**（ISRRetireRuntimeEx 側の quarantineResidentCount_ とも別系統）
  - **X6 の整合**: `quarantineResidentCount_`（Coordinator 側）を廃止 or 完全に DSPQuarantineManager の管理に委譲。**isFullyDrained は `DSPQuarantineManager::residentCount()` を直接読む**（走査コストは kMaxSlots=256 の for ループで NonRT では許容）
  - **走査 vs atomic counter の選択（確定）**: 既存の `quarantineActiveFlags_` 走査を**維持**（新しい atomic counter を追加しない）。isFullyDrained は NonRT で `DSPQuarantineManager::residentCount()` を呼び、Coordinator の quarantineResidentCount_ とは分離。X6 の新設 counter は intent/ring/retireQuarantine の transport residency に限定
- **🔴🔴🔴 十八次別視点14調査（2026-08-10）— kMaxSlots と constructor を確定**: `DSPQuarantineManager`（ISRDSPQuarantine.h:34-36）は:
  ```
  explicit DSPQuarantineManager(std::size_t maxSlots = 256);   // :36 kMaxSlots = 256
  QuarantineReason enum（:12-21）: GenerationMismatch / ResolveFailure / PublishViolation
      / CrossfadeViolation / ShutdownViolation / RetireDeferralTimeout / ReceiptReset / Unknown
  ```
  - **kMaxSlots = 256**（DSPHandleRuntime::MAX_DSP_SLOTS と同数）。residentCount() の走査は 256 要素の for ループ（NonRT で許容）
  - **QuarantineReason::ReceiptReset**（:19）: pendingReceipt_ reset → quarantine（X2 の receipt #1 と関係 — 別視点9で確認済み）
  - **X6 の設計で kMaxSlots / constructor の変更は不要**（既存 256 で充足。DSPHandleRuntime の slot 数と一致）

**🔴🔴🔴 十六次レビュー（2026-08-09）— INV-X6-4（二重 semantic 禁止）**:
```
INV-X6-4: No quarantine counter may represent both transport residency
          and actual DSP residency.
```
- **二重計上ではないが意味が違う**: `pendingIntentCount_` と `quarantineIntentResidencyCount_` が同じ Quarantine Intent を数えても二重計上ではない（transport の視点が異なる）。状態遷移表で固定:
```
Quarantine Intent = 1 → pendingIntentCount_ = 1, quarantineIntentResidencyCount_ = 1, quarantineResidentCount_ = 0
処理開始後 → pendingIntentCount_ = 0, quarantineIntentResidencyCount_ = 0, quarantineResidentCount_ = 1
```
- **状態遷移表（§6.8 テスト）**:
| 状態 | Intent | Ring | Resident |
|------|-------:|-----:|---------:|
| submit 後 | 1 | 0 | 0 |
| fallback 後 | 1 | 1 | 0 |
| pop 直後 | 0 | 0/1 | 0 |
| quarantine 成功 | 0 | 0 | 1 |
| reclaim 後 | 0 | 0 | 0 |

**🔴🔴 十二次レビュー（2026-08-09）— submitQuarantine の具体的な挿入位置を確定（🔴🔴🔴 追加調査 2026-08-09 で quarantineRingResidencyCount_ の増減を追加確定）**:
X6 実装時の `submitQuarantine`（ISRRuntimePublicationCoordinator.cpp:690-732）の変更:
```cpp
void RuntimePublicationCoordinator::submitQuarantine(...) noexcept
{
    // ... intent 生成 ...
    convo::fetchAddAtomic(pendingIntentCount_, 1);                    // ① reservation（既存 §1.1）
    convo::fetchAddAtomic(quarantineIntentResidencyCount_, 1);        // ①' X6: intent residency（新設）
    if (intentQueue_.push(intent))
    {
        // ② primary 成功 → quarantineIntentResidencyCount_ 維持（queue 内 Quarantine）
        return;
    }
    // ★ X6: fallback へ移動した時点で「primary intent residency」→「ring residency」に移る
    convo::fetchSubAtomic(quarantineIntentResidencyCount_, 1);        // ②' primary から減算
    if (quarantineFallbackQueue_.push(intent))
    {
        convo::fetchAddAtomic(quarantineRingResidencyCount_, 1);      // ④' X6: ring residency（fallback 残留）
        return;
    }
    convo::fetchSubAtomic(pendingIntentCount_, 1);                    // ③ rollback（全段失敗）
```
- **🔴🔴🔴 十八次調査（2026-08-09）— メモリオーダリングの規約（X5 と同一）**: `pendingIntentCount_` / `quarantineIntentResidencyCount_` / `quarantineRingResidencyCount_` の増減は **`std::memory_order_acq_rel`**（fetchAdd/fetchSub の default。AtomicAccess.h:91-105）。Producer（Timer/CoordinatorLoop）と Consumer（CoordinatorLoop）の2スレッド間 RMW のため race なし
- **🔴🔴🔴 十八次別視点5調査（2026-08-09）— producer hole との整合（X5 と同一の考察）**: submitQuarantine の `fetchAdd(quarantineIntentResidencyCount_, +1)` は push 先行。intentQueue_（MpscBoundedRing）の push は publication 完了後に return true/false するため、**producer hole は push 内で完結**し counter は収束する。fallback 移動（intent-- → ring++）も同様。**quarantineFallbackQueue_ も MpscBoundedRing**（ISRRuntimePublicationCoordinator.h:454）で同一の producer hole 性質を持つ
- **quarantineResidentCount_ は fetchAdd しない**（DSPQuarantineManager::quarantineHandle 内部で管理。ISRDSPQuarantine.cpp:17-48 の audit 成功時）
    convo::fetchAddAtomic(quarantineFallbackDropCount_, 1, release);  // drop
}
```
- **🔴🔴🔴 追加調査（2026-08-09）— intent/ring counter の移動を確定**: `quarantineIntentResidencyCount_` と `quarantineRingResidencyCount_` は**同時に 1 にならない**。fallback に移動した Quarantine Intent は「primary intent residency」を離れ「ring residency」に入る（前回調査の対象マッピング表 :2305-2308 と整合）
- **`quarantineResidentCount_` の +1（:713,723）は X6 実装で撤去**し、`DSPQuarantineManager::quarantineHandle()`（QuarantineService 実行時）が実在 DSP 数の管理に一本化
- **`quarantineIntentResidencyCount_` / `quarantineRingResidencyCount_` の pop 減算**: processIntent の `intentQueue_.pop`（Quarantine 時 → intent counter--）と `quarantineFallbackQueue_.pop`（:32-33 → ring counter--）で実施（Publish とは独立 — Quarantine は pendingIntentCount_ にも含まれるため二重管理だが、**pendingIntentCount_ と quarantine*ResidencyCount_ は別 semantic**）
- **P2 実装（§1.1）では quarantineIntentResidencyCount_ を追加しない**（現行の quarantineResidentCount_ +1 を維持）。X6 は P2 後に独立タスクとして実施

**🔴🔴🔴 十一次/十二次レビュー追記 — quarantineResidentCount_ の source of truth を確定（十四次で3分離に修正）**:
- **`DSPQuarantineManager::residentCount()`（ISRDSPQuarantine.h:50）が `quarantineResidentCount_` の唯一の source of truth**
- `quarantineRingResidencyCount_` は quarantine fallback / ring の残留を別に管理（`isFullyDrained` の `ringResident` 相当）
- **isFullyDrained の実測設定（Threading.cpp:131）は `ringResident + dspQuarantineResident` の drain aggregate として維持**するが、これは `quarantineResidentCount_` に混ぜず、**`quarantineRingResidencyCount_` と `quarantineResidentCount_` を個別に評価**
- **🔴🔴🔴 十八次調査（2026-08-09）— `AudioEngine::isFullyDrained()` の aggregate setter を X6 で廃止**: Threading.cpp:131 の `setQuarantineResidentCount(ringResident + dspQuarantineResident)` は、**X6 の分離と直接矛盾**する（`quarantineResidentCount_` に overflow ring 滞留を混ぜる）。X6 実装時は:
  - **:131 の aggregate setter を廃止**（`quarantineResidentCount_` = `DSPQuarantineManager::residentCount()` のみ）
  - **ringResident（overflow ring 滞留）は `quarantineRingResidencyCount_` に分離**して個別評価
  - isFullyDrained は §6.7 の最終形（quarantineIntentResidency / quarantineRing / quarantineResident / retireQuarantine を個別に == 0 評価）に拡張

**X6 の pop 減算位置（processIntent の while ループ + QuarantineIntentHandler）**:
```cpp
// processIntent（ProcessIntent.cpp:32-37）— X6 追記
// fallback ring の drain（quarantineFallbackQueue_）: ring counter 減算
while (quarantineFallbackQueue_.pop(commonIntent)) {
    if (commonIntent.type == IntentType::Quarantine)
        convo::fetchSubAtomic(quarantineRingResidencyCount_, 1);   // X6: ring residency 減算
    else
        convo::fetchSubAtomic(pendingIntentCount_, 1);             // （fallback は Quarantine 専用のため実際は Quarantine のみ）
    kDispatchTable[static_cast<std::size_t>(commonIntent.type)]->handle(commonIntent, ctx);
}
// メイン intentQueue_ の drain: intent counter 減算（Publish は X5 で別減算）
while (intentQueue_.pop(commonIntent)) {
    switch (commonIntent.type) {
        case IntentType::Publish:
            convo::fetchSubAtomic(publicationIntentResidencyCount_, 1);   // X5
            break;
        case IntentType::Quarantine:
            convo::fetchSubAtomic(quarantineIntentResidencyCount_, 1);    // X6: intent residency 減算
            convo::fetchSubAtomic(pendingIntentCount_, 1);                // 既存 §1.1
            break;
        default:  // Observe / Recovery
            convo::fetchSubAtomic(pendingIntentCount_, 1);                // 既存 §1.1
            break;
    }
    kDispatchTable[static_cast<std::size_t>(commonIntent.type)]->handle(commonIntent, ctx);
}
```
```cpp
// QuarantineIntentHandler::handle（ProcessIntent.cpp:73-104）— X6 追記
void QuarantineIntentHandler::handle(const Intent& intent, IntentHandlerContext& ctx) const noexcept
{
    // processIntent の while ループで quarantineIntentResidencyCount_/quarantineRingResidencyCount_ は減算済み
    //   （この handler は intent counter を触らない — pop 減算は processIntent が一元管理）
    const auto qResult = ctx.quarantine.executeQuarantine(...);
    // quarantineResidentCount_ は DSPQuarantineManager::quarantineHandle 内部で管理
    //   （audit 成功 = 実在 DSP として residentCount に反映）
    if (qResult.stateChanged && !request.handle.isNull())
        ctx.engine.submitRecoveryIntent(request.handle, buildSource);  // 既存 Recovery 発行
}
```
- **🔴🔴🔴 追加調査（2026-08-09）— pop 減算の一元管理を確定**: Quarantine の intent/ring counter 減算は **processIntent の while ループで実施**し、handler では行わない（X5 の Publish 減算と同じパターン）。handler は純 routing（HANDLER-1）のまま
- **fallback queue の Quarantine 以外**: 実コードで fallback は Quarantine 専用（submitQuarantine のみが push）。将来他 type が入る場合も `if type==Quarantine` 分岐で安全

**🔴🔴🔴 十八次別視点調査（2026-08-09）— observeDeferredRing_ の pendingIntentCount_ 減算（P2 と X6 の接続点）**:

実コード検証: `submitObserve`（ISRRuntimePublicationCoordinator.cpp:561-562）は intentQueue_ full 時に **observeDeferredRing_ にも push し `pendingIntentCount_ +1`** する。一方 `drainObserveDeferred`（ProcessIntent.cpp:47-56）は observeDeferredRing_ を pop して retireByHandle するだけで **pendingIntentCount_ を減算しない**:
```cpp
// ProcessIntent.cpp:47-56（現行）
void RuntimePublicationCoordinator::drainObserveDeferred(DSPLifetimeManager& lifetimeMgr) noexcept
{
    ObserveIntent deferred{};
    while (observeDeferredRing_.pop(deferred)) {
        const auto currentEpoch = currentPublicationEpoch();
        if (deferred.epoch < currentEpoch || deferred.handle.isNull())
            continue;
        lifetimeMgr.retireByHandle(deferred.handle);   // ← pendingIntentCount_ を減算しない
    }
}
```
- **現状は processIntent 末尾の `setPendingIntentCount(0)`（:43）でリセットされるため整合**している
- **ただし §1.1（P2）で `setPendingIntentCount(0)` を廃止する場合、`drainObserveDeferred` の pop でも `pendingIntentCount_` を減算する必要がある**（deferred として退避した Observe も pop 時に -1）
- **X6 との整合**: observeDeferredRing_ は Observe 専用（X6 の Quarantine intent/ring counter の対象外）。`pendingIntentCount_` のみが対象
- **P2 実装時の追記（確定）**:
> **★ 実装済み（2026-08-10・P2-1 で反映）**: `drainObserveDeferred`（ProcessIntent.cpp:68）に pop 成功直後の `fetchSubAtomic(pendingIntentCount_, 1)` を実装済み（十次レビュー ③ どおり、skip 判定前に減算 — pop 成功数 == push 成功数）。
```cpp
while (observeDeferredRing_.pop(deferred)) {
    convo::fetchSubAtomic(pendingIntentCount_, 1);   // ★ P2: deferred ring pop でも減算
    ...
}
```

**quarantineResidentCount_ の整合**: X6 後、`submitQuarantine` の +1 を撤去し、`quarantineResidentCount_` は **isFullyDrained の実測設定（Threading.cpp:131）と DSPQuarantineManager 内部管理のみ**に依存。QuarantineIntentHandler は intent counter 減算のみに関与（resident には触らない）。

### 6.7 X1-X6 統合後の isFullyDrained

**🔴🔴🔴 十六次レビュー（2026-08-09）— 全体 semantic state machine（設計の核心）**:
今回の修正の核心は、個々の atomic や counter ではなく、以下の **semantic state machine を一つずつ別の状態として閉じること**:
```
Intent → Admission → Transport Residency → Execution → Committed → Completed
→ Resident → Retired → Reclaimable → Deleted
```
- **X1**: Recovery admission（Intent → Admission → Transport/Durable Residency）の境界を閉じる
- **X2**: Committed → Completed → Receipt の境界を閉じる
- **X3**: Reclaimable → Deleted の境界（Reclaim Authority）を閉じる
- **X4**: Execution → Committed の write authority（RuntimeWorldAuthority）を閉じる
- **X6**: Transport Residency → Resident の境界（Intent/Ring/Resident/RetireQuarantine）を閉じる
- **🔴🔴🔴 十八次調査（2026-08-09）— X1 と X4 の state machine 境界は設計確定済み**:
  - **X1: reservation ownership 移動** = transport → durable の移動（十五次 §14 で rollback + recoveryAdmissionPending_ 切替に確定。§6.1）
  - **X4: ownership topology** = RuntimeWorldAuthority が Store を所有（§6.4-X4 で X4-A/X4-B 分離設計固定）
  - **両者とも実装開始可能なレベルに確定**（A-2.21）

**🔴🔴🔴 二十四次レビュー（2026-08-09）— `isFullyDrained()` を単独の truth source にしない（§27-A）**:
`isFullyDrained() == true` だから shutdown できる、ではない。**`isFullyDrained()` は measurement predicate であり、shutdown authority そのものではない**。shutdown は以下を組み合わせて判断する:
```
ShutdownPhase
+ ProducerQuiescence
+ AdmissionClosed
+ RecoveryAdmissionClosed
+ BuilderStopped
+ isFullyDrained()          ← measurement predicate（最後に評価）
```
- **isFullyDrained() は「観測」であって「決定」ではない**: 単独で true でも shutdown は開始しない。ShutdownPhase / ProducerQuiescence / AdmissionClosed / RecoveryAdmissionClosed / BuilderStopped が先に成立し、その後に isFullyDrained() を measurement として評価する
- **これは X1 の build gap（recoveryAdmissionPending_ = false だが build 中の gap）の安全性**と整合する: BuilderStopped が先に成立すれば、build gap の DSP は join で回収済み
- **実装**: isFullyDrained() は「観測関数」として純粋に保ち、shutdown 判断ロジック（ShutdownPhase 遷移）からは分離する

最終的な `isFullyDrained()`:
```
AdmissionClosed AND all producers joined AND Coordinator stopped AND Builder stopped
AND intentQueue empty AND observeDeferredRing empty
AND quarantineFallbackQueue empty AND recoveryIntentQueue empty
AND publicationIntentResidencyCount == 0 AND pendingIntentCount == 0
AND deferredPublicationCount == 0
AND quarantineIntentResidencyCount == 0          // X6: Intent lane 残留
AND quarantineRingResidencyCount == 0            // X6: ring/fallback 残留（十四次で分離）
AND quarantineResidentCount == 0                 // X6: 実在 quarantine DSP（DSPQuarantineManager が source）
AND retireQuarantineStore.residentCount == 0      // X6: RetireQuarantineStore 退避（既存 residentCount() を source。十八次別視点3で新 counter 不要を確定）
AND recoveryAdmissionPending == false            // X1: durable Recovery admission が空（🔴 二十六次: DurablePending OR Building の両方が false であること。lease 方式では Building 中も true を維持）
AND pendingReclaimHandles.empty()                 // X3: 🔴 二十六次 INV-X3-5（pendingReclaimHandles_ が reclaim pending の source of truth。近似 counter の reclaimInFlightCount_ だけでは不十分）
AND reclaimInFlightCount == 0
AND retire/deferred retire == 0
```
- **🔴🔴🔴 二十六次レビュー（2026-08-09）— lease 方式と isFullyDrained の整合**: `recoveryAdmissionPending == false` は「durable Recovery admission が空」を意味するが、**lease 方式では DurablePending OR Building の両方が false であること**を要求する。**Building 中は recoveryAdmissionPending_ が true を維持**するため、isFullyDrained は build gap を検出する（BuilderStopped と併用 — 二十三次レビュー §4.3 と整合）

**🔴🔴🔴 十四次レビュー（2026-08-09）— X6 の3分離を反映**: `quarantineRingResidencyCount_` と `quarantineResidentCount_` を**個別に評価**する（旧案の aggregate `ringResident + dspQuarantineResident` を混ぜない）。`quarantineResidentCount_` は `DSPQuarantineManager::residentCount()`（source of truth）と一致することを検証。
**🔴🔴🔴 十六次レビュー（2026-08-09）— X6 の4分離を反映**: `retireQuarantineResidentCount_` を追加（`RetireQuarantineStore::residentCount()` が source）。`quarantineResidentCount_` に RetireQuarantineStore の退避分を混ぜない。

**🔴🔴🔴 十八次調査（2026-08-09）— `AudioEngine::isFullyDrained()` の外部 setter が X1/X5/X6 と干渉（重要）**:

実コード検証で確定: **`AudioEngine::isFullyDrained()`（AudioEngine.Threading.cpp:114-136）は、Coordinator の counter を外部 setter で強制上書き**する:
```cpp
// AudioEngine.Threading.cpp:114-136（現行）
bool AudioEngine::isFullyDrained() noexcept
{
    const bool hasDeferredCommit = (runtimeOrchestrator_ != nullptr && runtimeOrchestrator_->hasDeferredRequest());
    runtimePublicationBridge_.setPendingIntentCount(hasDeferredCommit ? 1u : 0u);     // :117 ← X1/X6 と衝突
    runtimePublicationBridge_.setPublicationBacklogCount(hasDeferredCommit ? 1u : 0u); // :118
    ...
    runtimePublicationBridge_.setQuarantineResidentCount(
        static_cast<std::uint64_t>(ringResident + dspQuarantineResident));            // :131 ← X6 と衝突
    return !hasDeferredCommit && runtimePublicationBridge_.isFullyDrained();          // :135
}
```
**X1/X5/X6 との干渉**:
| setter | 現行の上書き | X との衝突 |
|--------|-------------|-----------|
| `setPendingIntentCount(hasDeferredCommit ? 1u : 0u)`（:117） | pendingIntentCount_ を 0/1 へ強制 | **X1**（recoveryAdmissionPending_ 切替後に pendingIntentCount_ は 0 だが durable state が存在し得る — isFullyDrained は両方を見る必要）/ **X6**（quarantineIntentResidencyCount_ の reservation を無視） |
| `setPublicationBacklogCount(hasDeferredCommit ? 1u : 0u)`（:118） | publicationBacklogCount_ を 0/1 へ強制 | **X5**（publicationIntentResidencyCount_ は新設で別 counter。publicationBacklogCount_ とは独立） |
| `setQuarantineResidentCount(ringResident + dspQuarantineResident)`（:131） | quarantineResidentCount_ を aggregate で上書き | **X6**（quarantineRingResidencyCount_ と quarantineResidentCount_ の分離と矛盾） |

**🔴🔴🔴 十八次別視点4調査（2026-08-09）— `LifetimeState::pendingIntentCount()` の混入（RetireIntent 系・Commit.cpp:462,604）**:
`AudioEngine.Commit.cpp:462,604` は `runtimePublicationBridge_.setPendingIntentCount(worldAuthority_.lifetime().pendingIntentCount())` を呼ぶ。`LifetimeState::pendingIntentCount()`（ISRRetire.cpp:182-189）は **retire intent の pending 数**:
```cpp
// ISRRetire.cpp:182-189
std::uint64_t LifetimeState::pendingIntentCount() const noexcept
{
    const uint64_t enqueued = convo::consumeAtomic(enqueueTicket_, acquire);
    const uint64_t consumed = convo::consumeAtomic(dequeuePos_, acquire);
    const uint64_t mainPending = (enqueued > consumed) ? (enqueued - consumed) : 0;
    const uint64_t fbPending = convo::consumeAtomic(fallbackCount_, relaxed);
    return mainPending + fbPending;   // ← retire intent の pending 数
}
```
- **`LifetimeState::pendingIntentCount()` は「retire intent（RetireIntentQueue）の pending 数」**であり、Coordinator の `pendingIntentCount_`（transport intent）とは**完全に別 semantic**
- **X5/X6 の counter 分離で重要**: LifetimeState の retire intent pending は Coordinator の transport intent に**混ぜてはならない**。Commit.cpp:462,604 の `setPendingIntentCount(lifetime().pendingIntentCount())` は §1.1（P2）で廃止対象（RetireIntent 系混入）
- **X5 の publicationIntentResidencyCount_ / X6 の quarantine 系 counter は LifetimeState の retire intent pending と独立**（retire intent は publish/quarantine intent ではない）
- **`setRetireBacklogCount` はこの LifetimeState pending と同値を設定**（Commit.cpp:463,605 は `setRetireBacklogCount(lifetime().pendingIntentCount())`）— retire backlog の実測として妥当。`setPendingIntentCount` への混入のみが問題

**X1〜X6 実装時の対応（確定）**:
```
① setPendingIntentCount の外部上書き（:117）は廃止
   → X1/X6 の reservation accounting と衝突するため
   → pendingIntentCount_ は Coordinator 内部の純粋 accounting（§1.1 P2）に一本化
   → hasDeferredCommit の判定は queue emptiness / deferredPublicationCount で代替
② setQuarantineResidentCount の aggregate 上書き（:131）は廃止
   → X6 の「quarantineResidentCount_ = DSPQuarantineManager::residentCount() のみ」に統一
   → ringResident（overflow ring 滞留）は quarantineRingResidencyCount_ に分離
③ isFullyDrained は「新設 counter を含む全項目」を Coordinator 内部で評価（§6.7 の最終形）
   → ShutdownScheduler::isFullyDrained（ISRRuntimePublicationCoordinator.cpp:470-482）を拡張
```
- **§1.1（P2）で①の一部（pendingIntentCount 混入廃止）は既に対応済み**。X1/X5/X6 実装時に②③も含めて完結させる
- **`ShutdownScheduler::isFullyDrained()`（:470-482）は現在 X1〜X6 の新設 counter を含まない**（swapPending / retireBacklog / publicationBacklog / pendingIntent / fallbackBacklog / reclaimInFlight / deferredRetireResidency / quarantineResident の8項目のみ）。X1〜X6 実装後は §6.7 の最終形（publicationIntentResidency / quarantineIntentResidency / quarantineRing / retireQuarantine / recoveryAdmissionPending を追加）に拡張する

### 6.8 X1-X6 のテスト計画（十一次レビュー §25 + 実コード調査で精緻化）

**🔴🔴🔴 二十三次レビュー（2026-08-09）— 必須 Acceptance Criteria（§23・architectural test として固定）**:

「完了」と認定するには Debug/Release build green では不十分。**以下を architectural test に必須化**:

| ID | 必須 invariant |
|----|----------------|
| X1-5 | **1 logical Recovery admission = exactly one reservation** |
| X1-6 | durable Recovery は transport residency と二重計上しない |
| X2-6 | **completion order == publication sequence order** |
| X3-4 | `readerRegistrationClosed == false` → shutdown reclaim forbidden |
| X3-4 | shutdown reclaim 開始後の `registerReaderThread()` は必ず失敗 |
| X4-3 | `publishAndSwap()` の write-capable Owner は `RuntimeWorldAuthority` のみ |
| X4 | production code に別 `RuntimeStore<..., SomeOtherOwner>` が存在しない |
| X4 | `RuntimePublishAuthority` は WriteAccess を所有しない |
| X4 | `getCurrent()` を physical read API の代替にしない |
| X5 | enqueue +1 / pop -1 / rollback -1 が完全一致 |
| X6 | Intent residency と DSP residency が同一 counter に入らない |
| X6 | `quarantineResidentCount_ == DSPQuarantineManager::residentCount()` |

| 対象 | 必須テスト | 実装位置 |
|------|-----------|---------|
| X1 | 256 enqueue → 257th → durable pending remains / Recovery A,A,A coalescing / queue full → Builder wakes → eventual recovery execution（soak） | 既存 `testRecoveryRequestEnqueueAndPop`（ISRSemanticValidationTests.cpp:608）を拡張。`takePendingRecoveryAdmission` の新規テスト。**🔴 テスト基盤（十八次調査確定）**: `takePendingRecoveryAdmission` / `PendingRecoveryAdmission` は Coordinator のメンバに追加されるため、**ISRSemanticValidationTests（ヘッドレス）で直接テスト可能**。Builder 連携（RebuildDispatch.cpp:911 後の消費）は AudioEngineHarness の統合テスト（queue full → Builder wake → Recovery publish → isFullyDrained）で検証。**🔴 既存テストの実装詳細（十八次別視点9調査で確定）**: `testRecoveryRequestEnqueueAndPop`（:609-624）は `submitRecoveryRequest(handle, buildSource)` → `popRecoveryRequest()` の **1-hop transport のみ検証**（buildSource.sealed=true を渡し、2回目の pop が null = 1-hop 保証を検証 :621-622）。**X1 の拡張はこの基盤に追加**:
```cpp
// X1 追加テスト（queue full → durable 化 → takePendingRecoveryAdmission）
// 1. recoveryIntentQueue_（256）を満杯に enqueue
// 2. 257th submitRecoveryRequest → durable PendingRecoveryAdmission に保持
// 3. popRecoveryRequest() で 256 件消費 → durable は残る（pop されない）
// 4. takePendingRecoveryAdmission() → durable を取得 → recoveryAdmissionPending_ == false
// 5. coalesce: 同一 handle の重複 submit → Pending=1（reservation 増加なし）
``` |
| X2 | 10→11→12（正常）/ 11→10（out-of-order）/ 10→10（duplicate）/ UINT64_MAX 近傍（wraparound） | 新規 `PublishCompletionMonotonicityTests`（AudioEngine.h の PublishReceiptWaiter を直接検証）。**🔴 テスト基盤（十八次調査確定）**: PublishReceiptWaiter は AudioEngine の private メンバ（:3637）のため、**AudioEngineHarness（AudioEngineHarness.h）経由で統合テスト**する。完全な publish パイプライン（commitRuntimePublication → OwnerChannel → IntentQueue → CoordinatorLoop → executePublish → onPublishCommitted → receipt）を実スレッドで通し、waitForPublishReceipt の戻り値と m_lastObservedSequence の同期を検証 |
| X3 | reader active → shutdownReclaim forbidden / reader stopped → allowed / shutdown phase invalid → no reclaim | 既存 `ISRSemanticValidationTests` の requestReclaim テストを拡張。`reclaim(ReclaimMode, ...)` のモード分岐検証。**🔴 テスト基盤（十八次調査確定）**: `closeReaderRegistration()` / `readerRegistrationClosed()` は EpochDomain のメソッドなので、**ISRSemanticValidationTests（ヘッドレス）で直接テスト可能**。`registerReaderThread()` が `-1` を返すこと + 既存登録済み reader の enter/exit 継続を検証。AudioEngineHarness で shutdown 中の reader re-entry 不可（統合）を検証 |
| X4 | `RuntimeWorldAuthority::commit()` caller を1箇所に限定（静的検査）/ `RuntimeStore::publishAndSwap()` caller を1つに固定（regression） | **静的検査（rg で caller を列挙）**。既存 `RuntimeWorldAuthorityProjectionTests` を regression に。**🔴 静的検査コマンド（十八次調査確定）**:
```bash
# publishAndSwap の caller（Test 3）: RuntimeWorldAuthority-owned WriteAccess のみ期待
rg -n "publishAndSwap\(" src/ | grep -v "\.h:35\|\.h:44\|\.h:121"   # WriteAccess 定義以外
# ISR commit() の caller（Test 5）: PublishExecutor のみ期待
rg -n "\.commit\((PublishAuthority|Granted)" src/audioengine/
# 二重 Store 検出（Test 6）: RuntimePublishWorld 用 RuntimeStore の実体化
rg -n "RuntimeStore<RuntimePublishWorld|RuntimeStore<.*RuntimePublishWorld" src/
```
**Expected**: publishAndSwap は RuntimeWorldAuthority-owned WriteAccess のみ / commit は PublishExecutor のみ / Store は AudioEngine::runtimeStore 1箇所のみ。**🔴 十九次レビューで Test 7/9/10 を追加**: Test 7（commit-before-swap ordering）/ Test 9（dual-pointer semantic consistency: publish 完了後 currentWorld_ と runtimeStore.current の sequence/epoch/generation 一致。途中の transitional mismatch を明記）/ Test 10（INV-X4-7: getCurrent() を consumeWorldHandle(runtimeStore) の置換先にしない） |
| X5 | Publish enqueue +1 / pop -1 / queue full rollback / deferred counter unaffected | 既存 `ISRSoakTests.cpp:70` のコメント（enqueuePublicationIntent は pendingIntentCount_ を更新しない）を拡張し、publicationIntentResidencyCount_ の検証を追加。**🔴 テスト基盤（十八次調査確定）**: publicationIntentResidencyCount_ は Coordinator のメソッドなので、**ISRSoakTests（ヘッドレス）で直接テスト可能**。既存 `testIntentQueueSaturation`（:75-114）の enqueuePublicationIntent ループで +1/pop -1/rollback を検証 |
| X6 | QuarantineIntent enqueue intent+1 / pop intent-1 / Quarantine success resident+1 / Reclaim resident-1 / Intent pending で resident=0 / Intent=0 で resident=1 | 既存 `ISRSemanticValidationTests` の quarantine テストを拡張。`quarantineIntentResidencyCount_` / `quarantineResidentCount_` の分離検証。**🔴 テスト基盤（十八次調査確定）**: quarantine counter は Coordinator のメソッドなので、**ISRSemanticValidationTests（ヘッドレス）で直接テスト可能**。DSPQuarantineManager の resident 状態遷移（quarantineHandle / reclaimSlot）は ISRDSPQuarantine 単体で検証 |

**🔴🔴🔴 十六次レビュー §21 — テスト計画の補強（adversarial test）**:
| 対象 | 追加テスト要件 |
|------|----------------|
| X1 | **queue full → Recovery A → Recovery A coalesce → Recovery B → Builder wake → exactly one valid recovery admission → counter == 0**（coalesce で reservation が増えないこと INV-X1-5 の直接検証）<br>**A(G10) / B(G10) / C(G11) → G11 のみが必要**（obsolete policy 再利用: G10 は `isRebuildObsolete` と整合して破棄） |
| X2 | **seq 10 complete → seq 11 complete → seq 10 duplicate → lastCompleted == 11**<br>**commit 10 → commit 11 → completion 11 の人工的 out-of-order test**: これは**現在の architecture（PublishExecutor sole gateway + FIFO）では発生不能**であることをテスト側で明示（INV-X2-6）。「CAS なら安全」ではなく、architectural invariant の固定 |
| X3 | **activeReaderCount = 0, readerRegistrationClosed = false → reclaim forbidden**<br>**activeReaderCount = 0, readerRegistrationClosed = true → reclaim allowed**<br>**shutdown reclaim 開始後に registerReaderThread() が必ず失敗すること**（INV-X3-4） |
| X4 | **RuntimeWorldAuthority → RuntimeStore::WriteAccess → publishAndSwap 以外の経路が存在しない**ことを source-level architectural test で固定（INV-X4-3） |
| X5 | **Publish commit → counter unchanged / Publish completion → counter unchanged**（reservation は queue residency + producer reservation のみ。commit/completion で decrement しない） |
| X6 | **状態遷移表をテストで固定**: submit 後 (1,0,0) → fallback 後 (1,1,0) → pop 直後 (0,0/1,0) → quarantine 成功 (0,0,1) → reclaim 後 (0,0,0)。`pendingIntentCount_` と `quarantineIntentResidencyCount_` が同じ Intent を数えても二重計上でないこと（INV-X6-4） |

**🔴 実コード調査による追加テスト要件**:
- **X1**: `takePendingRecoveryAdmission()` は RebuildDispatch.cpp:911 の while ループ**後**に残余を処理する。テストは「while ループ終了後に durable state が空であること」を検証
- **X1（追加）**: `PendingRecoveryAdmission` の handle/epoch/intentId が消費時に正しく復元されること（`takePendingRecoveryAdmission()` が `RecoveryIntent` と同フィールドを返す）。coalesce で latest buildSource が採用され、同一 generation で handle が変わらないこと
- **X2**: `m_lastObservedSequence`（Orchestrator.h:246）と `PublishReceiptWaiter::lastCompleted_`（AudioEngine.h:3634）の**2箇所の watermark が同期**することを検証（onPublishCommitted 経由の統合テスト）
- **X3（追加）**: `EpochDomain::closeReaderRegistration()` 呼び出し後に `registerReaderThread()` が `-1` を返し、**既存登録済み reader（audioThreadRcuReader）は enter/exit を継続できる**こと（新規登録のみ拒否）。`reserveReaderThread()` も false を返すこと
- **X5/X6**: processIntent の while ループ（ProcessIntent.cpp:36）の type 分岐で、**Publish は publicationIntentResidencyCount_--、Quarantine/Recovery/Observe は pendingIntentCount_--、Quarantine は quarantineIntentResidencyCount_-- の3系統が正しく分岐**することを検証
- **X6（追加）**: quarantineFallbackQueue_（:32-33）の drain で `quarantineRingResidencyCount_--` が実施されること。submitQuarantine の fallback 移動時に「intent→ring」の counter 移動が正しいこと（両方が同時に 1 にならない）

### 6.9 X1-X6 の実装順序（十一次 §24 + 十五次 §25 + 十七次 §13 反映）

**🔴🔴🔴 十五次レビュー（2026-08-09）— X1 を X4 より先に設計固定**: X1（Recovery 保証）と X2（completion 意味論）は **correctness の根幹**。X1 は X4（authority naming）より先に設計固定すべき。

**🔴🔴🔴 十七次レビュー（2026-08-09）— 実装と設計固定を分ける**:
実装順序は「P2 → X5 → X6 → X2 → X1 → X4 → X3」を支持するが、**設計固定と実装を分離**する:
- **設計固定（先に invariant を決定）**: X2 / X1 / X4 / X3 の invariant（INV-X2-6 / INV-X1-5,6 / INV-X4-3 / INV-X3-4）を先にコード契約として決定
- **実装（その後）**: P2 → X5 → X6 → X2 → X1 → X4 → X3 の順で実装
- **理由**: X4 を先に実装すると、**ownership topology の変更と X1/X2 の semantic 変更が同時に入り、デバッグ範囲が広がる**。X1/X2 の semantic invariant を先に固定し、X4 は独立リファクタとして扱う

**🔴🔴🔴 二十三次レビュー（2026-08-09）— Phase 0（invariant/specification freeze）を最優先（§22）**:
X2 の invariant を実装前に固定することを最優先とする。**実装順序を Phase 0〜Phase 7 に明確化**:
```
Phase 0  invariant / specification freeze（INV-X1-5,6 / INV-X2-6 / INV-X3-4 / INV-X4-1〜8 / INV-X5-1 / INV-X6-4 をコード契約として固定）
Phase 1  P2-1 → P2-2 → P2-3 → P2-4
Phase 2  X5 → X6
Phase 3  X2（completion ordering — 最優先）
Phase 4  X1（Recovery durable admission）
Phase 5  X4-A → X4-B-0 ... X4-B-11
Phase 6  X3（reclaim authority）
Phase 7  integrated shutdown / ISR soak
```
- **X4 は X2/X1/X3 の意味が確定してから触る**（X4-B は ownership topology 変更のため、X1/X2/X3 の semantic 変更と同時に実装するとデバッグ範囲が広がる）
- **🔴🔴🔴 二十五次レビュー（2026-08-09）— X3 の意味論を X4 より先に固定すべき（§22）**: X3 は reader active / reader registration / shutdown phase / reclaim を結ぶ **memory lifetime の最終安全境界**。一方 X4 は主として **publication authority topology**。したがって:
  - **Lifetime correctness（X3）を先に固定してから、Publication authority topology（X4）を変更する**方が安全
  - **X3 の実装（Phase 6）は X4 の実装（Phase 5）より後でもよいが、X3 の「意味論」（readerRegistrationClosed と reclaim の関係・INV-X3-4 / INV-ISR-04）は Phase 0 で X4 より先に固定する**
  - **設計固定順序**: X2 → X1 → **X3（意味論）** → X4（意味論）→ 実装順序（P2 → X5 → X6 → X2 → X1 → X4 → X3）
- **Phase 0 で invariant を freeze する理由**: 「counter の値」ではなく「counter が何を意味しているか」を先に固定する（ISR の lifetime proof の前提）

| Phase | 対象 | 優先度 | 理由 |
|-------|------|--------|------|
| 設計固定 | X2 / X1 / X4 / X3 | — | invariant を先にコード契約として決定 |
| 1 | **P2-1〜P2-4** | P0 | pendingIntentCount_ accounting + shutdown drain（§1.1-1.4） |
| 2 | X5 | P1 | residency semantics の基礎を確定（Publish residency） |
| 3 | X6 | P1 | queue / ring / resident の意味論を完全分離 |
| 4 | X2 | P1 | **Publish completion の意味論を数学的に閉じる（最優先）** |
| 5 | X1 | P0 | **Recovery durable admission（correctness の根幹・X4 より先に設計固定）** |
| 6 | **X4** | P1 | **二段階: X4-A（rename・低リスク）→ X4-B（ownership topology 再設計・大規模リファクタ）**。詳細は §6.4-X4（X4-0〜X4-10） |
| 7 | X3 | P2 | reclaim authority を最終統合 |
| 8 | — | — | 統合 shutdown / soak |

**X4 の内部順序（十七次確定・X4-A/X4-B 分離）**:
```
設計固定: INV-X4-1〜INV-X4-5 をコード契約として決定
X4-A: rename（RuntimeIntentCoordinator / RuntimePublishAuthority）→ ビルド/テスト → リスク低
X4-B: RuntimeStore Owner 変更（core Coordinator → RuntimeWorldAuthority）→ 大規模リファクタ
     → 一時生成 publishWorld 削除 → PublishExecutor/Bootstrap/shutdown clear を authority 経由に一本化
     → static architectural tests（publishAndSwap caller / 二重 Store 検出）
```
- **X4-A だけなら比較的低リスクだが、X4-B は `RuntimeStore<World, Owner>` の template パラメータと `friend Owner` の構造を変更する大規模リファクタ**
- **X4-B は単純な rename として扱わない**: `RuntimeStore` / `RuntimeWorldAuthority` / `RuntimePublishAuthority` / `RuntimePublishExecutor` の4点を一つの ownership migration として実装・検証

**中心原則**: **Queue residency / deferred state / committed state / resident object / reclaimable object を、それぞれ別の semantic state として扱う**。これが完成すると、`pendingIntentCount_` 周辺の「counter は何を数えているのか」問題が Publish / Quarantine / Recovery まで含めて閉じ、ISR の安全性・Shutdown correctness・Authority Singularization まで一貫したモデルになる。

---

# Appendix（優先度・レビュー記録）

## A-1. 優先度別サマリ・実装順序

### 優先度サマリ（十六次レビュー + X1-X6 精緻化・宣言位置確定版）

| 優先度 | 項目 | 状態 | 設計参照 |
|--------|------|------|----------|
| **P2-1** | pendingIntentCount_ accounting 再設計（residency reservation → push → rollback。**Publish pop は fetchSub 対象外 — 七次レビュー W2**） | 実装 | §1.1 |
| **P2-2** | invariant_INV3_INV5 テスト | 実装 | §1.4 |
| **P2-3** | MpscBoundedRing producer hole テスト（CONVO_TESTING hook） | 実装 | §1.3 |
| **P2-4** | isFullyDrained queue emptiness 検証（admission closed 後・Publish 残留も捕捉） | 実装 | §1.2 |
| **P3** | §2.1 admission 統一（staged admission）/ §4.1 ShutdownPhase / §4.2 ReceiptWaiter / §4.3 LinearRamp / §4.4 BlockDouble / §4.5 bootstrap | 将来最適化 | §2 / §4 |
| **撤回** | PopStatus API / force reclaim / **Recovery coalesce（四次レビュー NO-GO）** | 設計不成立 | §3.1 / §3.2 / §2.2 |

### 実装順序（推奨・十一次レビュー反映 Phase 1-10 + X1-X6）

**P2（Phase 1-10）**: 九次レビュー推奨の Phase 分けを反映（変更による因果関係を追いやすい順序）:

- **Phase 1 — counter semantics 固定**: コードコメント/設計契約として `pendingIntentCount_ = Observe residency + Quarantine residency + Recovery residency + enqueue reservations` を固定。`setPendingIntentCount(...)` による lifecycle counter の再設定を producer/consumer path から排除（§1.1）
- **Phase 2 — Publish pop exclusion**: `commonIntent.type != Publish` の場合のみ fetchSub（Observe/Recovery/Quarantine）。**九次 §21 条件3**（§1.1.6）
- **Phase 3 — reservation implementation**: 各 producer（submitObserve / submitQuarantine / submitRecoveryRequest）を `reserve → primary push → fallback push → rollback only if all enqueue failed` に統一。**1 Intent = 1 reservation（九次 §21 条件2）**（§1.1.2）
- **Phase 4 — consumer decrement + setPendingIntentCount(0) 全廃**: 全 transport endpoint で decrement。skip / invalid / stale intent でも pop した時点で decrement（§1.1.6）
- **Phase 5 — isFullyDrained() queue emptiness**: `ShutdownScheduler::isFullyDrained()` に4キュー emptiness を追加（§1.2）
- **Phase 6 — shutdown phase assertion**: AdmissionClosed + all producers joined + Coordinator stopped + Builder stopped をコードで assert / phase guard（**九次 §21 条件4**）（§1.2）
- **Phase 7 — MPSC producer-hole deterministic test**: CONVO_TESTING 限定で gate を入れ、Producer A reserve → pause → Producer B publish → Consumer pop false → A publish → Consumer pop A → B を固定（§1.3）
- **Phase 8 — INV-3 / INV-5**: lifecycle invariants を固定（§1.4）
- **Phase 9 — seqId completion monotonicity test**: PublishReceiptWaiter の Producer serialization をテスト固定（§4.2.2。九次 §22 Phase 7）
- **Phase 10 — Recovery drop behavior 別タスク**: X1（Recovery full-drop そのもの）を明示的な残課題として別タスク化（九次 §22 Phase 8）

**残課題 X1-X6（十一次レビュー §24 推奨順序）**:
| Phase | 対象 | 優先度 | 理由 |
|-------|------|--------|------|
| 1 | X5 | P1 | residency semantics の基礎を確定（§6.5） |
| 2 | X6 | P1 | queue と resident の意味論を完全分離（§6.6） |
| 3 | X2 | P1 | Publish completion の ordering を固定（§6.2） |
| 4 | X4 | P1 | **二段階: X4-A（Authority naming 分離・rename・低リスク）→ X4-B（RuntimeStore ownership topology 変更・大規模リファクタ）**。詳細は §6.4-X4（X4-0〜X4-10） |
| 5 | X1 | P0 | Recovery 保証を実装（設計自体は先に決める。§6.1） |
| 6 | X3 | P2 | reclaim authority を最終統合（§6.3） |

軽微改善（bootstrap jassert）は §4.5。将来拡張（recoveryIntentQueue_ MPSC 化 / coalesce 別設計 / 複数 DSP world）は R1 として記録。

---

## A-2. レビュー記録（一次〜十六次 + 追加調査）

### A-2.1 一次・二次レビュー総括（2026-08-09）

| 項目 | 二次レビュー判定 | 反映 |
|------|-------------|----------|
| per-type admission 統一 | 妥当。P3 アーキテクチャ整理 | **P3 に降格**（§2.1） |
| PopStatus 導入 | **設計不成立**（Empty と producer hole は diff 同一値で区別不能） | **撤回**（§3.1）。テスト固定のみ |
| producer hole テスト | **方向は正しいが現行 API では生成不能** | **CONVO_TESTING test hook 必須**。テスト4本（§1.3） |
| pendingIntentCount | **fetch_add-after-push は underflow する** | **residency reservation → push → rollback に再設計**（§1.1） |
| 意味論 | **pendingIntentCount_ は Publish を含まない** | **意味論を修正**。isFullyDrained の Publish 混入上書きを廃止 |
| queue emptiness | 正しいが **admission closed 後に限定** | **shutdown 順序 invariant 化**（§1.2） |
| Recovery coalesce | 連続同一 handle のみで要件不満。**連続 coalesce も今は入れない** | **四次レビューで NO-GO**（silent loss バグ）。実装は別設計（§2.2 参照） |
| INV-3/INV-5 テスト | 妥当。優先度高 | **承認**（§1.4）。INV-5 を「silent loss 禁止」に再定義 |
| force reclaim | **明確に正しい撤回**。ISR の Authority Singularization と整合 | **撤回**（§3.2）。Faulted ≠ memory safety 保証を明記 |
| ShutdownPhase | P3 で妥当。遷移シーケンス invariant | **承認**（§4.1） |
| PublishReceiptWaiter | 現状維持妥当。**危険の本質は reorder を許す admission/deferred 設計** | **承認**（§4.2） |
| LinearRamp / BlockDouble / bootstrap | 対象外・実測してから判断・補助的 | **承認**（§4.3/§4.4/§4.5） |

### A-2.2 三次レビュー（別視点・実コード照合）（2026-08-09）

**レビュー観点**: dash の行番号・実装可能性・既存コードとの整合・ISR/RT 前提の実在性を**別視点**から実コードで検証した。

| # | 検証項目 | 検証結果（実コード照合） | 反映 |
|---|---------|------------------------|----------|
| T1 | MpscBoundedRing の seq 遷移・diff 解釈 | `sequences_[i]=i`(:54) → push `pos+1`(:81) → pop `pos+Capacity`(:109)。pop の diff(:103) は `diff!=0` で false。empty と producer hole の区別不能・正常動作で diff>0 は発生しない — **分析は正確** | 修正不要 |
| T2 | `processIntent` の行番号 | `quarantineFallbackQueue_`(:32) → `intentQueue_`(:36) → `drainObserveDeferred`(:39) → `setPendingIntentCount(0)`(:43) — **正確** | 修正不要 |
| T3 | `pendingIntentCount_` の書き込み元 | 5箇所3系統（intent系/Publish系 Threading.cpp:117/RetireIntent系 Commit.cpp:462,604）— **3系統混在は実在** | 修正不要 |
| T4 | **isFullyDrained は全カウンタを実測値に上書き** | Threading.cpp:117-131 は pendingIntentCount 以外も publicationBacklog / fallbackBacklog / retireBacklog / deferredRetire / quarantineResident を**全て実測値で上書き**。`retireBacklogCount_` も load→store 非RMW（:136-160）。「3系統混在」は全カウンタに及ぶ | **§1.1/§1.2 に反映**（上書きは現状維持、混入廃止のみ） |
| T5 | **producer は全て NonRT** | submitObserve（DSPTransition.h:156 / Timer.cpp:896,1029,1568）、submitQuarantine（Timer.cpp:1788,1826）、submitRecoveryRequest（AudioEngine.h:4277）、enqueuePublicationIntent（AudioEngine.h:4413）— **全て Timer/Transition/Commit（NonRT）**。CoordinatorLoop 周期 = 1ms（kIntervalMs=1）。**RT producer の cache-line bouncing 懸念は現状では成立しない** | **ISR/RT 注意を明記**（将来の RT enqueue のみ注意） |
| T6 | force reclaim の ~AudioEngine 構造 | Graceful Drain ポーリング（最大 5000ms, publishEpoch+tryReclaim）が既存（CtorDtor.cpp:200-217） | **§3.2 に反映** |
| T7 | ShutdownPhase | AudioEngine.h:2521（6値, enum class:int）/ ISRShutdown.h:25。遷移シーケンス（ReleaseResources.cpp:73-537）は dash の記述と**完全一致** | **行番号・値数を修正**（§4.1） |
| T8 | bootstrap / R4 行番号 | Init.cpp:54-55、AudioEngine.h:2027、ReleaseResources.cpp:415,420 — **正確** | 修正不要 |
| T9 | recoveryIntentQueue_ 型 | `LockFreeRingBuffer<RecoveryIntent, 256>`（ISRRuntimePublicationCoordinator.h:433-434）SPSC — **正確** | 修正不要 |

**三次レビュー総合判定**: dash の技術的根拠は**約 90% 正確**。修正を要したのは **T4（全カウンタ上書き）・T5（RT 経路の不存在）・T7（ShutdownPhase 行番号）** の3点。

**★ 追加確定事項（三次レビュー）**:
- **isFullyDrained は全カウンタを実測値に上書き**: 本改修は混入廃止（Publish/RetireIntent）と hard reset 廃止のみ。他の実測上書きは現状維持 + queue emptiness で補強
- **全 producer は NonRT**: RT 経路の cache-line bouncing 懸念は現状では不成立
- **CONVO_TESTING は新規マクロ**: CMakeLists.txt で MpscBoundedRingTests に定義が必要
- **invariant_*.cpp は未存在**: 既存 `*Tests.cpp` パターンで追加
- **キュー容量**: intentQueue_=4096 / quarantineFallback=1024 / observeDeferred=1024 / recovery=256

### A-2.3 四次レビュー（ISR/MPSC/shutdown 観点）（2026-08-09）

**レビュー観点**: ISR（Immutable Snapshot Runtime）・MPSC/SPSC メモリモデル・shutdown/drain・ownership/lifetime の観点から、改修案を検証した。最新ソースで確認した事実: `ShutdownScheduler::isFullyDrained()` は7カウンタのみ、`AudioEngine::isFullyDrained()` は pendingIntentCount_/publicationBacklogCount_ を hasDeferredCommit で上書き、`publicationSequenceCounter_`（AudioEngine.h:2189）は fetchAddAtomic で seqId 割当、`shutdownReclaim` は ReleaseResources.cpp:415,420 に残存、`dspQuarantineManager_.residentCount()` は quarantine lane の実在 DSP 数。

| # | 検証項目 | 四次レビュー判定 | dash 反映 |
|---|---------|----------------|----------|
| F1 | P2-1 reservation→push→rollback | **GO**（基本原理は正しい） | §1.1 維持 |
| F2 | load→store lost update 廃止 | **GO**（必須） | §1.1 維持 |
| F3 | push後 fetch_add の underflow 回避 | **GO**（reservation-before-publish が正しい） | §1.1 維持 |
| F4 | fallback reservation 維持 | **GO** | §1.1 維持 |
| F5 | setPendingIntentCount(0) 廃止 | **GO**（必須） | §1.1 維持 |
| F6 | Publish/Retire の混入廃止 | **GO** | §1.1 維持 |
| F7 | **quarantineResidentCount_ を pop で減算** | **NO-GO**（別の意味のカウンタ — 実際の quarantine lane DSP 数） | **§1.1.5 で削除**（P3 で別カウンタ化） |
| F8 | **publicationBacklogCount_ = Publish residency** | **NO-GO**（現コードでは hasDeferredCommit 由来） | **§1.1.5 で修正**（publicationIntentResidencyCount_ 新設は P3） |
| F9 | queue emptiness を drain 判定に追加 | **GO**（ただし consumption 禁止・shutdown ordering 条件付き） | §1.2 修正 |
| F10 | MPSC producer-hole 契約明文化 | **GO** | §1.3 維持 |
| F11 | PopStatus 導入撤回 | **GO** | §3.1 維持 |
| F12 | producer-hole test hook | **GO with modification**（2スレッド必須） | §1.3.2 修正 |
| F13 | INV-3 test | **GO** | §1.4 維持 |
| F14 | INV-5 silent-loss test | **GO**（Critical は HealthMonitor 経由で検証） | §1.4.3 修正 |
| F15 | IntentAdmissionPolicy | **P3としてGO**（staged admission に要修正） | §2.1 修正（AdmissionStep） |
| F16 | **Recovery coalesce** | **NO-GO**（正当な Recovery を永久抑制するバグ） | **§2.2 全面修正**（削除・代替3案） |
| F17 | force reclaim を実装しない | **GO**（EBR上正しい） | §3.2 維持 |
| F18 | Faulted と reclaim safety の分離 | **GO**（重要） | §3.2 維持 |
| F19 | ShutdownPhase 対応表 | **GO**（正常系だけでは不十分） | §4.1 修正（Normal/Timeout/Emergency） |
| F20 | PublishReceiptWaiter 現状維持 | **CONDITIONAL**（seq 完了順序の証明が必要） | §4.2 修正（completion monotonicity） |
| F21 | LinearRamp 文書化 | **GO** | §4.3 維持 |
| F22 | BlockDouble false | **GO**（実測前に変更しない） | §4.4 維持 |
| F23 | bootstrap jassert | **GO**（Success 判定確認済み） | §4.5 維持 |
| F24 | memory_order（relaxed） | **GO**（counter は numeric accounting）だが **convo wrapper に統一** | §1.1.2 修正（fetchAddAtomic/fetchSubAtomic） |
| F25 | fetch_sub の cur>0 ガード削除 | **GO**（reservation invariant 成立後。ガードは不整合を隠す） | §1.1.6 修正 |
| F26 | shutdownReclaim 残存 | **確認**（二系統の認識。P2 で統合しない判断は正しい） | §3.2 追記 |

**四次レビュー総合判定**: **部分採用 / 一部 NO-GO**。
- **GO**: P2-1 residency accounting（要 convo wrapper 統一・cur>0 ガード削除）、P2-4 queue emptiness（要 shutdown ordering）、P2-3 producer-hole test（要 2スレッド化）、P2-2 INV-3/INV-5、force reclaim 撤回、ShutdownPhase 対応表
- **NO-GO（修正必須）**: ①quarantineResidentCount_ の pop 減算（別の意味のカウンタ）、②publicationBacklogCount_ = Publish residency（現コードと不一致）、③Recovery coalesce（silent loss バグ）
- **CONDITIONAL**: PublishReceiptWaiter（completion seq monotonicity の証明が必要）

**ISR 上の優先順位（四次レビュー推奨）**:
```
① pendingIntentCount_ の正確な residency accounting
② queue emptiness を shutdown source of truth へ追加
③ shutdown producer 停止順序を固定
④ MPSC hole test
⑤ INV-3 / INV-5
⑥ Admission Policy 整理
⑦ Recovery coalesce（別設計）
```

### A-2.4 五次レビュー（実装可能性・API 実在性・既存テスト整合）（2026-08-09）

**レビュー観点**: これまでのレビューが検証していない「実装可能性・API 実在性・既存テストとの整合」を実コードで検証した。

| # | 検証項目 | 検証結果（実コード照合） | dash 反映 |
|---|---------|------------------------|----------|
| V1 | `convo::fetchAddAtomic/fetchSubAtomic` の存在 | AtomicAccess.h:91,100 に定義。`std::atomic<T>&, U value, memory_order order = acq_rel`。dash の `fetchAddAtomic(pendingIntentCount_, 1, relaxed)` はコンパイル可能 | **§1.1.2 で memory_order の扱いを確定**（wrapper デフォルト acq_rel に統一推奨） |
| V2 | queue emptiness の API 実在 | MpscBoundedRing::`sizeApprox()`（:115）= enqueuePos-dequeuePos の acquire 読取り（消費なし）。LockFreeRingBuffer::`size()`（:76）= writeIndex-readIndex（消費なし） | **§1.2.2 で4キュー×API を確定** |
| V3 | **2つの RuntimePublicationCoordinator の区別** | `convo::isr::RuntimePublicationCoordinator`（ISRRuntimePublicationCoordinator.h:68, Intent authority）と `convo::RuntimePublicationCoordinator`（src/core/RuntimePublicationCoordinator.h:24, Publish authority）は**別クラス**。本設計は前者（ISR）が対象。core 側は makeRuntimePublicationCoordinator()（AudioEngine.h:3646）で一時生成され publishWorld のみ | **§1.2 に実装場所（ShutdownScheduler::isFullyDrained）と Coordinator 特定を追記** |
| V4 | `PublishStageResult` の値 | src/core/RuntimePublicationCoordinator.h:15-19 で **Success/Rejected/Failed の3値**。`publishWorld` は store 失敗→Failed(:103)、reject→Rejected(:115) | **§4.5 で3値と `result != Success` を追記** |
| V5 | 既存 `testProducerHoleDoesNotJumpAhead`（MpscBoundedRingTests.cpp:238） | **既存**だが「空状態で pop が false」のみ検証し、真の producer hole（publication 遅延）は生成しない。新テストは命名衝突に注意 | **§1.3.3 で命名・重複を確定** |
| V6 | 既存テストのスレッド使用 | MpscBoundedRingTests.cpp は `<thread>` 使用（:44）・`testMultiProducerNoLoss`（:123）で並行 push 実装済み | **§1.3.3 で 2スレッド deterministic の統合性を確認** |
| V7 | `requestReclaim` の依存 | ISRRuntimePublicationCoordinator.cpp:573 は `DSPHandleRuntime&`/`ISRRetireRouter&` に依存。`pendingReclaimHandles_` は AudioEngine.h:4616 のメンバ | **§1.4.2 で INV-3 テストのスタブ要件を追記** |
| V8 | test hook の挿入位置 | MpscBoundedRing::push() の CAS 成功直後（entries_ 書込み前）に `#ifdef CONVO_TESTING` ブロックを挿入可能。本番ビルドでは消滅 | **§1.3.2 で push 内部の #ifdef ブロックとして確定** |
| V9 | `isFullyDrained` の返り値 `!hasDeferredCommit` | Threading.cpp:135 `return !hasDeferredCommit && ...`。上書き廃止後も返り値条件は**維持**すべき | **§1.1.5 で返り値維持を追記** |
| V10 | `submitObserve` の2段階 fallback | intentQueue_ → observeDeferredRing_ の2段階。reservation は fallback 成功時に維持（再 +1 しない） | §1.1.6 の不変条件と整合 |

**五次レビュー総合判定**: 改修設計は**実装可能**であり、API 実在性は全て確認できた。修正を要したのは **V3（2つの Coordinator 区別）・V4（PublishStageResult 3値）・V5（既存テスト衝突）・V7（INV-3 スタブ要件）・V8（test hook 挿入位置）** の5点。設計の中心（P2-1 residency accounting）は**変更なしで GO**。

### A-2.5 六次レビュー（最終整合・ISR/RCU/MPSC/shutdown 観点）（2026-08-09）

**レビュー観点**: `REPAIR_PLAN2-dash.md` と `Practical Stable ISR Bridge Runtime.md` を突き合わせ、ISR/RCU/MPSC/shutdown の観点から最終整合を検証した。

| # | 検証項目 | 六次レビュー判定 | dash 反映 |
|---|---------|----------------|----------|
| S1 | P2-1 pendingIntentCount_ residency accounting | **GO**（MPSC lost-update / reset / underflow を正しく解消） | §1.1 維持 |
| S2 | **invariant「push完了後 == actual」は強すぎる** | **修正必要**（Producer が push() から戻った時点で Consumer が既に pop 済みの可能性） | **§1.1.2 で修正**（基本不変条件 = `counter >= actual residency`、`==` は quiescent point のみ） |
| S3 | P2-4 isFullyDrained() queue emptiness | **GO**（drain の source of truth 強化として妥当） | §1.2 維持 |
| S4 | **queue emptiness は shutdown ordering が必須** | **コード上の invariant として固定**（AdmissionClosed → producer停止/join → queue観測） | **§1.2 でコード invariant 化を追記** |
| S5 | P2-3 MPSC producer-hole test | **GO**（MPSC publication ordering の回帰として必要） | §1.3 維持 |
| S6 | P2-2 INV-3 / INV-5 tests | **GO**（Retire safety / silent loss の保証として妥当） | §1.4 維持 |
| S7 | **INV-5: drop == memory corruption と定義しない** | **修正推奨**（runtime safety と functional recovery を分離。Recovery lost でも runtime safety は維持） | **§1.4.3 で追記**（「Recovery Intent の silent loss 禁止」と再定義） |
| S8 | quarantineResidentCount_ を pending counter にしない | **GO**（意味論が異なる） | §1.1.5 NO-GO 維持 |
| S9 | publicationBacklogCount_ を Publish residency と解釈しない | **GO**（現コードと意味が一致しない） | §1.1.5 NO-GO 維持 |
| S10 | Recovery coalesce 撤回 | **GO**（正当な Recovery の silent loss） | §2.2 NO-GO 維持 |
| S11 | force reclaim 撤回 | **GO**（EBR/RCU safety を破壊） | §3.2 NO-GO 維持 |
| S12 | shutdownReclaim 残存は別問題 | **妥当**（二系統。P2-1/P2-4 に混ぜない） | §3.2 追記済み |
| S13 | PublishReceiptWaiter | **CONDITIONAL**（completion sequence の単調性証明が必要。P2 counter bug とは別問題のため今は触らない） | §4.2 CONDITIONAL 維持 |
| S14 | ShutdownPhase 強化 | **GO（P3）**（正常/timeout/emergency の区別） | §4.1 維持 |
| S15 | IntentAdmissionPolicy 統一 | **GO（P3）**（現時点では P2 から分離） | §2.1 P3 維持 |
| S16 | producer は全て NonRT | **確認**（fetchAddAtomic の cache-line contention は RT deadline を直接侵害しない） | §1.1 維持 |
| S17 | memory_order relaxed vs acq_rel | **wrapper default（acq_rel）に統一を支持**（ただし acq_rel が queue ordering を解決するわけではない） | §1.1.2 反映済み |

**六次レビュー総合判定**: **P2-1 / P2-2 / P2-3 / P2-4 は実装 GO**。実装開始前に2点を明文化することを推奨:
- **A. `pendingIntentCount_` の基本 invariant は `counter >= actual residency`。`==` は producer quiescence 後に限定**
- **B. queue emptiness は `AdmissionClosed → producer停止/join → queue観測` の順序保証を前提条件**

**dash の評価点（六次レビュー）**: counter を「状態の再計算値」から「residency accounting」へ / load/store 廃止 / reservation-before-publication / fallback 含めて一件一予約 / queue emptiness を二次的な実測 source に / quarantine resident と intent residency 分離 / Publish backlog と deferred publish の混同解消 / Recovery coalesce 撤回 / force reclaim 撤回 / MPSC producer-hole の実再現テスト。これらは Practical Stable ISR Bridge Runtime の原則（RT は待たない・所有しない・判断しない / Retire は Epoch を通る / Shutdown は完全 Drain / Overflow は silent loss にしない / Authority を一箇所に集約）と整合。

### A-2.6 七次レビュー（実装詳細の完全性・call site 網羅・RT 経路最終確認）（2026-08-09）

**レビュー観点**: 実装直前の「実装詳細の完全性」を検証 — reservation→push→rollback の全 call site 網羅、RT 経路の最終確認、CONVO_TESTING の実装影響、staged admission と dispatch の整合、INV-5 の drop テスト方法。

| # | 検証項目 | 検証結果（実コード照合） | dash 反映 |
|---|---------|------------------------|----------|
| W1 | setPendingIntentCount 全 call site | :553,562（Observe）:667（Recovery）:714,724（Quarantine）の +1 / :686（popRecoveryRequest）の -1 / processIntent の 0 / Threading:117・Commit:462,604 の上書き — **dash §1.1.6 と一致** | 修正不要 |
| W2 | **intentQueue_ に Publish が混在 → pop 時 fetchSub の非対称** | intentQueue_ は Observe/Publish/Recovery/Quarantine の4種混在（:201-206）。Publish は pendingIntentCount_ 対象外のため、**Publish pop で fetchSub(1) すると非対称 -1**（過小評価 → isFullyDrained が見逃し） | **§1.1.6 で「Publish pop は fetchSub 対象外」を追記**（重大発見） |
| W3 | RT 経路の最終確認 | processBlockDouble（BlockDouble.cpp:27）内に submitXxx / setPendingIntentCount なし。**全 producer は NonRT 確定** | 修正不要 |
| W4 | waitForDrain 時点の Coordinator/Builder 停止 | ReleaseResources.cpp:189（shutdownCoordinatorLoop）:190（stopRebuildThread）→ :447（waitForDrain）。**waitForDrain 時点で両方停止済み** | **§1.2 で shutdown ordering の実コード照合を追記** |
| W5 | queue emptiness と pendingIntentCount_ の独立判定 | intentQueue_ は Publish 含む全4種。queue emptiness は Publish 残留も捕捉（pendingIntentCount_ とは独立） | **§1.2 で独立判定を追記** |
| W6 | CONVO_TESTING のメモリレイアウト | MpscBoundedRing は alignas(64) の cache-line 分離。フック用 atomic は #ifdef で消滅 → **production レイアウト不変** | **§1.3.2 で private 末尾配置を追記** |
| W7 | staged admission と dispatch 整合 | kDispatchTable は4種 1:1 網羅 + static_assert（ISRIntentDispatcher.h:58-68）。§2.1 の AdmissionStep（Primary/Fallback/Drop）は4種を表現 | 修正不要 |
| W8 | INV-5 の drop テスト方法 | 既存 testRecoveryRequestEnqueueAndPop（ISRSemanticValidationTests.cpp:608）は 1-hop 輸送検証済み。INV-5 は 256 回 submit → full → 257 回目 drop を検証 | **§1.4.2 で drop テスト方法を追記** |
| W9 | processIntent の3キュー drain | quarantineFallbackQueue_(:32) → intentQueue_(:36) → drainObserveDeferred(:39)。Publish pop 除外を追加 | §1.1.6 で反映 |

**七次レビュー総合判定**: 実装詳細は**ほぼ完全**。修正を要したのは **W2（Publish pop の fetchSub 非対称 — 重大発見）・W4（shutdown ordering 実コード照合）・W5（queue emptiness の独立判定）・W6（CONVO_TESTING レイアウト）・W8（INV-5 drop テスト方法）** の5点。**P2-1/P2-2/P2-3/P2-4 は引き続き実装 GO**。

### A-2.7 八次レビュー（コード責務・スレッドモデル・実装完了条件）（2026-08-09）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準実装として、コード上の責務・スレッドモデル・ISR/RCU・MPSC・shutdown/lifetime の観点から再検証。`Practical Stable ISR Bridge Runtime.md` も照合。

**実装完了条件（八次レビュー §25 — 3点の絶対条件）**:
| # | 絶対条件 | 崩すと | dash 反映 |
|---|---------|--------|----------|
| C1 | `pendingIntentCount_ == actual residency` を**常時要求しない**（基本 invariant は `counter >= actual residency`。`==` は producer quiescence 後のみ） | 並行中の counter 不整合を誤検出 | §1.1.2（六次反映済み） |
| C2 | **Publish pop では decrement しない**（intentQueue_ に Publish 混在のため） | 非対称 -1 → isFullyDrained が見逃し | §1.1.6（七次 W2 反映済み） |
| C3 | **queue emptiness は producer quiescence 後のみ authoritative**（AdmissionClosed + all producers joined をコードで assert / phase guard） | 通常動作中の誤った drain 判定 | §1.2（六次/七次反映済み） |

**追加要求（八次レビュー §22-23）**:
| # | 要求 | dash 反映 |
|---|------|----------|
| 条件A | reservation counter は「成功した push の数」ではなく **`queue residency + producer-side enqueue reservation`**。コードコメント・テスト名・設計資料で同一に | **§1.1.5 で追記** |
| 条件B | fallback 含めて「1 intent = 1 reservation」（二重 +1 禁止） | §1.1.4（既存反映済み） |
| §19 | PublishReceiptWaiter は Producer serialization が seqId monotonicity を保証することをテストで固定（今すぐ API 変更不要） | **§4.2.2 で追記** |
| §18 | shutdown lifetime contract（「本当に reader が存在しないことの保証」）を将来タスクとして明文化 | **§3.2 で追記** |

**実コード検証（八次レビュー）**:
- `PublishReceiptWaiter::complete`（AudioEngine.h:3604-3614）は mutex 保護の high-water mark。コメントに「executePublish は intentQueue_ を FIFO で処理するため seqId は単調増加で完了する（順序性前提）」（:3605）と明記
- seqId は `fetchAddAtomic(publicationSequenceCounter_, 1)`（:3412）で割り当て、Publish intent に `intent.sequenceId = seqId`（:4405）
- 八次レビュー判定: **P2-1/P2-2/P2-3/P2-4 は実装 GO。概ね 90% 以上の設計精度**。実装完了条件として上記3点（C1/C2/C3）をコードとテストの双方で固定する
- **ISR 整合性**: residency と semantic state の分離 / reservation-before-publication / Publish を pendingIntentCount から除外 / queue emptiness を shutdown 後の transport-level source of truth に / EBR/Retire authority を迂回しない — いずれも Practical Stable ISR Bridge Runtime と整合。RT は待たない・所有しない・判断しないを維持

### A-2.8 九次レビュー（残課題の明確化・完全修正版ではないとの判断）（2026-08-09）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準コードとして、`REPAIR_PLAN2-dash(2)`（八次反映版）と `Practical Stable ISR Bridge Runtime.md` を突き合わせて検証。

**総合判定**: **「P2-1〜P2-4 は実装 GO。ただし REPAIR_PLAN2-dash(2) を『完全修正版』とは扱わない」**。

**実装完了条件（九次 §21 — 4条件をコードレビュー項目として固定）**:
| # | 条件 | dash 反映 |
|---|------|----------|
| 条件1 | `pendingIntentCount_` = `queue residency + producer-side enqueue reservation`（successful push count ではない。コードコメントで固定） | §1.1.5（八次反映済み） |
| 条件2 | 1 Intent = 1 reservation（fallback で二重 +1 しない） | §1.1.4（反映済み） |
| 条件3 | **Publish pop は pending counter を触らない**（`if (type != Publish) decrement`） | §1.1.6（七次 W2 反映済み） |
| 条件4 | queue empty は AdmissionClosed + all producers joined + Coordinator stopped + Builder stopped の後だけ drain 判定に使う | §1.2（反映済み） |

**残課題6件（九次 §23 — P2 後検証対象・§5 に反映）**:
1. **X1: Recovery Intent の full-drop そのもの**（最優先 — Recovery 保証に直結）。現状は drop + Critical telemetry（AudioEngine.Retire.cpp:192-196, :223）で「Recovery 保証」ではない。INV-5 は「drop を正しく記録できるか」の検証であり「絶対 drop しない」保証ではない
2. **X2: Publish completion sequence monotonicity の実装保証**（Shutdown/Receipt correctness に直結）
3. X3: shutdownReclaim の二系統
4. X4: RuntimePublicationCoordinator の authority 二重化
5. X5: Publish Intent residency の専用 counter 未導入
6. X6: quarantine intent residency と quarantine resident の semantic 分離

**評価点（九次 §20,23）**: load/store counter → residency accounting / push→increment → reservation→push→rollback / counter only → counter + actual queue occupancy / Recovery coalesce → NO-GO / Publish pop → pending counter から除外 / queue emptiness → producer quiescence 後のみ authority。**RT/audio callback に新しい lock・allocation・ownership・decision path を導入するものではなく、ISR の境界を悪化させない**。

**推奨実装順序（九次 §22 Phase 1-8 → dash では Phase 1-10 に展開）**: counter semantics → Publish pop exclusion → reservation → consumer decrement + hard reset 全廃 → queue emptiness → shutdown phase assertion → producer-hole test → INV-3/INV-5 → seqId monotonicity test → Recovery drop 別タスク。

### A-2.9 十次レビュー（具体的コード変更の実行可能性・実装コードとの整合）（2026-08-09）

**レビュー観点**: 実装直前の「具体的コード変更が実装コードの構造と整合するか」を検証 — reservation→push→rollback の観測カウンタとの関係、quarantineResidentCount_ の現行維持、fetchSub 挿入位置、SPSC/private アクセス。

| # | 検証項目 | 検証結果（実コード照合） | dash 反映 |
|---|---------|------------------------|----------|
| U1 | **`observeOverflowCounter_`（:559）の扱い** | 「intentQueue_ full → observeDeferredRing_ 退避」の観測カウンタ。reservation の成功・失敗とは独立に、**既存位置（primary push 失敗直後）に維持**すべき。診断カウンタとして residency に含めない | **§1.1.6 で追記** |
| U2 | **`submitQuarantine` の `quarantineResidentCount_` +1（:713,723）の現行維持** | reservation は `pendingIntentCount_` のみ。quarantineResidentCount_ の +1 は既存の意味論（実在 DSP 数の近似）を維持するため、push 成功時（primary または fallback）に**既存どおり実行**。P3 で `quarantineIntentResidencyCount_` に分離 | **§1.1.6 で追記** |
| U3 | **fetchSub の挿入位置（pop 成功直後・skip 前）** | `drainObserveDeferred`（ProcessIntent.cpp:50-55）は `pop(deferred)` 成功直後・epoch-FIFO skip 判定前に fetchSub。`ObserveIntentHandler`（intentQueue_ 経由）は processIntent の while ループで fetchSub 済み（ハンドラ内では fetchSub しない）→ **二重 fetchSub を防止** | **§1.1.6 で追記** |
| U4 | **Recovery reservation のスレッド** | submitRecoveryIntent（AudioEngine.h:4274）は QuarantineIntentHandler（ProcessIntent.cpp:102）から呼ばれ、**CoordinatorLoop 内**。popRecoveryRequest（Builder Loop）との SPSC 契約維持 | 修正不要 |
| U5 | **ShutdownScheduler の private queue アクセス** | `ShutdownScheduler` は RuntimePublicationCoordinator の **nested class**。C++ では nested class は外側の private メンバにアクセス可能 → `coordinator_.intentQueue_` 等に直接アクセス可 | 修正不要（§1.2 の記述は正確） |
| U6 | **submitRecoveryRequest の SPSC** | recoveryIntentQueue_ は SPSC（Producer=CoordinatorLoop 単一）。reservation は CoordinatorLoop 内で行われ、競合なし | §1.1.3 の記述は正確 |

**十次レビュー総合判定**: 実装コードとの整合は**完全**。修正を要したのは **U1（observeOverflowCounter_ の位置維持）・U2（quarantineResidentCount_ の現行維持）・U3（fetchSub 挿入位置・二重計上防止）** の3点（いずれも dash §1.1.6 に追記）。**P2-1〜P2-4 は実装 GO**。

### A-2.10 十一次レビュー（残課題 X1-X6 の詳細設計・ISR 意味論の最終閉包）（2026-08-09）

**レビュー観点**: `REPAIR_PLAN2-dash(3)`（十次反映版）を最新案として、`ConvoPeq(20260809-022629).md` と `Practical Stable ISR Bridge Runtime.md` を突き合わせて検証。

**総合判定**: **P2-1〜P2-4 は基本的に妥当・実装 GO**。ただし「この改修案は pending intent の accounting と shutdown drain の健全化としては妥当だが、ConvoPeq の ISR Runtime 全体を完全に健全化する改修案ではない」。残課題 X1〜X6 は個別バグの修正ではなく、ISR の「唯一の Authority / Intent residency / Publish completion / shutdown-reclaim」の意味論を最終的に閉じるための設計。

**実装完了条件（十一次 §21 — 必須 acceptance criteria 3条件）**:
```
C1: pendingIntentCount_ の意味をコード上で固定
    （Observe + Quarantine + Recovery の queue residency + producer-side enqueue reservations。
      Publish excluded / Retire excluded / Quarantine resident excluded）
C2: queue emptiness は phase-gated（AdmissionClosed + all producers joined + Coordinator stopped
    + Builder stopped を assert できる形）
C3: Recovery full-drop を「成功扱い」にしない（drop を Health/diagnostic layer まで確実に伝播。
    将来は Accepted / Queued / Dropped を区別）
```

**X1-X6 詳細設計（§6 に反映）**:
| # | 対象 | 設計方針 | 不変条件 |
|---|------|---------|---------|
| X1 | Recovery Durable Admission | durable Pending state + retry/coalesce（queue 拡張は NO-GO） | INV-X1-1〜4 |
| X2 | Publish completion monotonicity | CAS による monotonic watermark | INV-X2-1〜4 |
| X3 | shutdownReclaim 二系統 | Reclaim Authority は一つ、Safety Precondition が二種類（RuntimeEBR / ShutdownQuiescent） | phase assertion |
| X4 | authority 二重化 | クラス統合 NO-GO。IntentCoordinator / RuntimePublishAuthority に明示命名・分離 | Authority matrix |
| X5 | Publish residency 専用 counter | `publicationIntentResidencyCount_` 新設（deferred と分離） | INV-X5-1 |
| X6 | Quarantine Intent/Resident 分離 | `quarantineIntentResidencyCount_` 新設（resident 専用 counter と分離） | 状態遷移 |

**実コード検証（十一次）**: `PublishExecutor::executePublish`（RuntimePublishExecutor.h:19-20）が sole gateway / `getCurrentBuildSnapshotForRecovery()`（AudioEngine.h:4265）/ `onPublishCommitted`（RuntimePublishExecutor.h:84 → Orchestrator.h:146）/ `quarantineResidentCount_` は ISRRetireRuntimeEx（:219,222,237）と Coordinator（:713,723）の2系統。

**RT 安全性（十一次 §13）**: RT allocation / delete / mutex / wait / World mutation / publish decision / ownership transfer / crossfade decision / Epoch bypass は全て「追加なし」→ **ISR 境界を悪化させない**。

**実装順序（十一次 §24）**: X5/X6 → X2 → X4 → X1 → X3（X1 の設計自体は先に決める）。

**中心原則**: **Queue residency / deferred state / committed state / resident object / reclaimable object を、それぞれ別の semantic state として扱う**。これが完成すると、`pendingIntentCount_` 周辺の「counter は何を数えているのか」問題が Publish / Quarantine / Recovery まで含めて閉じ、ISR の安全性・Shutdown correctness・Authority Singularization まで一貫したモデルになる。

### A-2.11 十二次レビュー（X1-X6 の具体的コード挿入位置・P2 との競合検証）（2026-08-09）

**レビュー観点**: §6 の X1-X6 詳細設計が、実装コードのどこに挿入されるか（具体的コード挿入位置）と、P2 実装（§1.1-1.4）と競合しないかを検証。

| # | 検証項目 | 検証結果（実コード照合） | dash 反映 |
|---|---------|------------------------|----------|
| V11 | **X3: 既存呼び出し元への影響** | `requestReclaim`（AudioEngine.h:4248, AudioEngine.Retire.cpp:83）→ RuntimeEBR / `shutdownReclaim`（AudioEngine.h:2027, ReleaseResources.cpp:415,420）→ ShutdownQuiescent。`requestReclaim`（ISRRuntimePublicationCoordinator.cpp:573）を `reclaim(ReclaimMode, ...)` に拡張、`shutdownReclaim`（ISRDSPHandle.h:171）を廃止 | **§6.3 で追記** |
| V12 | **X5: enqueuePublicationIntent は inline** | `enqueuePublicationIntent`（ISRRuntimePublicationCoordinator.h:273-278）は inline で `intentQueue_.push` のみ。X5 はここに reservation→push→rollback を追加 | **§6.5 で追記** |
| V13 | **X6: submitQuarantine の挿入位置** | `submitQuarantine`（:690-732）に quarantineIntentResidencyCount_ の reservation/rollback を追加。quarantineResidentCount_ の +1（:713,723）は X6 で撤去 | **§6.6 で追記** |
| V14 | **X1: submitRecoveryRequest の durable state** | `submitRecoveryRequest`（:647-673）の push 失敗時に durable Pending state へ保持。`pendingRecoveryAdmission_`（atomic）を Coordinator 側に追加。Builder wake は既存 rebuildMutex/rebuildCV | **§6.1 で追記** |
| V15 | **X2: PublishReceiptWaiter との統合** | `complete`（AudioEngine.h:3604-3614）を monotonic watermark + CAS に変更。`onPublishCommitted`（Orchestrator.cpp:305）が publishCompletion を呼ぶ形に | **§6.2 で追記** |
| V16 | **X4: 型エイリアスと使用箇所** | `convo::RuntimePublicationCoordinator`（core）は AudioEngine.h:3509 の using エイリアス / `convo::isr::RuntimePublicationCoordinator` は7ファイルで使用。リネームは大規模のため P2 後の独立タスク | **§6.4 で追記** |

**十二次レビュー総合判定**: X1-X6 の詳細設計は**実装可能**で、挿入位置は全て実コードと整合。修正を要したのは **V11（X3 呼び出し元）・V12（X5 inline）・V13（X6 submitQuarantine）・V14（X1 durable state）・V15（X2 統合）・V16（X4 影響範囲）** の6点（いずれも dash §6 に追記）。**P2-1〜P2-4 は実装 GO。X1-X6 は P2 後に独立タスクとして実施（P2 と競合しない）**。

### A-2.12 実コード調査による X1-X6 詳細設計の精緻化（2026-08-09）

**調査観点**: X1〜X6 の実装対象コードを詳細に調査し、各 X の詳細設計を実装可能なレベルまで精緻化した。

| # | 対象 | 実コード調査結果 | §6 反映 |
|---|------|----------------|---------|
| R1 | **X1: Builder 消費ループとの整合** | `popRecoveryRequest` は RebuildDispatch.cpp:911 の `while (auto recovery = ...popRecoveryRequest())` で消費。`recoveryPending` フラグ（AudioEngine.h:2581,4283）で lost-wakeup 防止（:905 でクリア）。`RecoveryIntent` は handle/epoch/intentId/buildSource の POD | **§6.1 で Builder 側の durable state 消費処理を追記** |
| R2 | **X2: 3箇所の同期変数** | `m_lastObservedSequence`（Orchestrator.h:246, cpp:310）と `PublishReceiptWaiter::lastCompleted_`（AudioEngine.h:3634）の2箇所の watermark + `waitFor`（:3628 cv 待機）。`onPublishCommitted`（:305）→ notifyPublishReceipt（:313） | **§6.2 で3箇所の変更を追記** |
| R3 | **X3: requestReclaim の epoch 判定構造** | `requestReclaim`（:573-608）: retire → `retireEpoch < minReaderEpoch` 判定（:589）→ reclaim。AudioEngine.h:2027 は `shutdownPhase >= Destroy` で shutdownReclaim。`reclaim` は private（ISRDSPHandle.h:188）+ friend Coordinator | **§6.3 で reclaim(ReclaimMode) の具体的設計と呼び出し元対応を追記** |
| R4 | **X4: core Coordinator の一時生成** | `PublishExecutor::executePublish`（RuntimePublishExecutor.h:63-66）で `makeRuntimePublicationCoordinator().publishWorld()` を一時生成。ownerChannel().take → commit → publishWorld が sole execution path | **§6.4 で PublishExecutor 内部の使用を追記** |
| R5 | **X5: processIntent の type 分岐** | processIntent（ProcessIntent.cpp:36-37）の while ループで、Publish は publicationIntentResidencyCount_-- / 他は pendingIntentCount_-- に type 分岐 | **§6.5 で processIntent の分岐を追記** |
| R6 | **X6: quarantineResidentCount_ の source of truth** | `DSPQuarantineManager::residentCount()`（ISRDSPQuarantine.h:50）が唯一の source of truth。`quarantineHandle` は実際に適用時のみ true。QuarantineIntentHandler（ProcessIntent.cpp:73-104）は quarantine 実行 + Recovery 発行 | **§6.6 で resident source of truth と handler の整合を追記** |

**精緻化の結果**: X1-X6 は全て**実装可能な詳細設計**に到達。特に X1（Builder durable state 消費）、X2（3箇所の同期変数）、X3（reclaim(ReclaimMode) の具体的シグネチャ）、X6（residentCount の source of truth）は、実装時にそのまま利用できるレベルの精度になった。テスト計画（§6.8）も既存テストの拡張位置を確定。

### A-2.13 十四次レビュー（P2 GO・X1-X6 要修正）（2026-08-09）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準コードとして、`REPAIR_PLAN2-dash(4)` を ISR/Audio Thread 境界・Immutable Snapshot Runtime・Authority Singularization・ownership/lifetime/reclaim・queue residency と semantic state の分離・shutdown 安全性・publication/completion ordering まで含めて評価。

**総合判定**: **P2-1〜P2-4 は GO。X1〜X6 は「このまま実装してよい完成設計」とするのは NO-GO（一段修正してから実装）**。

| X | 判定 | 必須修正 |
|---|------|---------|
| X1 Recovery | 🟡 要修正 | single slot をやめ、Recovery generation / durable state に再設計。reservation を push 前に行う |
| X2 Completion | 🔴 要修正（最優先） | watermark と per-request receipt を分離。wraparound semantics を決定 |
| X3 Reclaim | 🟡 要修正 | shutdown precondition を retire 前に評価。reader re-entry 不可まで保証 |
| X4 Authority | 🟡 要修正 | rename は有効。RuntimeWorldAuthority → RuntimeStore write path を実際に一本化 |
| X5 Publish residency | 🟢 GO | そのまま採用可能 |
| X6 Quarantine | 🟡 要修正 | Intent/Ring/Resident を3分離。`quarantineResidentCount_` の意味を再定義 |

**dash §6 への反映**:
- **X2**: 「CAS を入れるか」ではなく「completion を何と定義するか」から再設計。contiguous completion 前提（PublishExecutor sole gateway + FIFO）で `lastCompletedSequence_.store(seq, release)` で十分。sparse completion は将来 MPSC 許容時のみ。wraparound は「architecturally impossible」契約に（案A 採用）。INV-X2-5（sole completion writer）追加
- **X1**: `PendingRecoveryAdmission`（pending / recoveryGeneration / buildSource）に再設計。reservation を push 前に行い、push 失敗時は rollback しない（durable state に存在するため）
- **X3**: precondition を retire 前に評価。`activeReaderCount == 0` に加えて `reader re-entry impossible` を precondition に追加
- **X4**: RuntimeStore の `friend Owner`（core/RuntimeStore.h:81）構造を確認。`RuntimeWorldAuthority` を publication authority surface にする設計を追記（Owner 変更は大規模リファクタのため P2 後）
- **X5**: GO 判定を反映（見出しに承認）
- **X6**: `quarantineIntentResidencyCount_` / `quarantineRingResidencyCount_` / `quarantineResidentCount_` の3 semantic に分離。`quarantineResidentCount_` は `DSPQuarantineManager::residentCount()` のみ

**最優先修正（十四次 §最終判定）**: 1. X2（completion watermark と receipt の完全分離）2. X1（single-slot 再設計）3. X1（reservation を push 前）4. X6（3分離）5. X3（precondition を retire 前）6. X4（write authority 実体一本化）。

### A-2.14 追加調査による X1-X6 詳細設計の精緻化（2026-08-09）

**調査観点**: X1-X6 の設計根拠となる実装コードをさらに深く調査し、未確定事項を確定した。

| # | 対象 | 追加調査結果 | §6 反映 |
|---|------|-------------|---------|
| T1 | **X1: recoveryGeneration の実体** | `rebuildRequestGeneration`（AudioEngine.h:2423）は requestRebuild（RebuildDispatch.cpp:643）で `++`。`isRebuildObsolete(generation)`（:2464）で obsolete 判定。Builder の Recovery 消費（:965-967）で `recoveryGeneration = consumeAtomic(rebuildRequestGeneration)` を取得し snapshot に設定 | **§6.1 に追記**（coalesce は generation 単位） |
| T2 | **X2: Committed state と Completion の区別** | `lastCommittedPublicationSequence_`（AudioEngine.h:2191）は Commit.cpp:398 で更新（Committed state）。`PublishReceiptWaiter::lastCompleted_`（:3634）は completion。2つを分離し INV-X2-2 を明示 | **§6.2 に追記** |
| T3 | **X3: reader re-entry impossible の API** | `ISRRetireRouter`（ISRRetireRouter.h:71-74）: `registerReaderThread` / `reserveReaderThread` / `enterReader` / `exitReader` / `activeReaderCount`（:67）/ `minReaderEpoch`（:75） | **§6.3 に追記**（isQuiescent は readerRegistrationClosed を含む） |
| T4 | **X4: RuntimeStore の Owner 実体** | `RuntimeStore<World, Owner>`（core/RuntimeStore.h:12）は `friend Owner`（:81）のみ `acquireWriteAccess`（:83）可。WriteAccess move-only（:21-28）。core Coordinator が Owner（core/RuntimePublicationCoordinator.h:34） | §6.4 反映済み（十四次） |
| T5 | **X5: seqId 割当** | `reserveRuntimePublicationIdentity`（AudioEngine.h:3406-3414）で `fetchAddAtomic(publicationSequenceCounter_, 1) + 1` | §6.5 反映済み |

**確定結果**: X1 の coalesce は `rebuildRequestGeneration` 単位（T1）、X2 は Committed（`lastCommittedPublicationSequence_`）と Completion（`lastCompleted_`）の2 sequence を分離（T2）、X3 の isQuiescent は `readerRegistrationClosed` を含む（T3）。全て実装時に利用可能な精度に確定。

### A-2.15 十五次レビュー（P2/X5 GO・X1-X6 条件付き GO・acceptance criteria）（2026-08-09）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準コードとして、`REPAIR_PLAN2-dash(5)` を ISR/Immutable Snapshot Runtime/Authority Singularization/ownership・lifetime/shutdown/completion ordering の観点から再検証。

**総合判定**: **P2 = GO、X5 = GO。X1/X2/X3/X4/X6 = 設計方向 GO、実装は条件反映後 GO**。dash(5) は dash(4) までの問題点をかなり正確に潰している。**ただし過去レビュー由来の記述が一部残っており、最新版の結論と旧版の記述が同一文書内で混在**（実装時は A-2.14 以降の記述を正とする）。

| X | 判定 | 実装条件 |
|---|------|---------|
| X1 Recovery | 🟡 条件付き GO | **pendingIntentCount_ と durable Recovery admission の semantic boundary を明文化**（transport residency と recoveryAdmissionPending_ を分離） |
| X2 Completion | 🔴 最優先 | **4層の sequence 分離**（publicationSequenceCounter_ / lastCommittedPublicationSequence_ / lastCompletedSequence_ / per-request receipt）+ completion is FIFO を architectural invariant に |
| X3 Reclaim | 🟡 条件付き GO | **readerRegistrationClosed** + all producers stopped + Audio stopped まで shutdown precondition |
| X4 Authority | 🟡 条件付き GO | rename は GO。**最終目標は RuntimeWorldAuthority を actual RuntimeStore write authority に** |
| X5 Publish residency | 🟢 GO | そのまま採用可能 |
| X6 Quarantine | 🟡 条件付き GO | 3 semantic 厳密固定。**quarantineResidentCount_ に aggregate 値を絶対に入れない** |

**dash §6 への反映**:
- **X1**: `pendingIntentCount_` を transport residency に限定し、`recoveryAdmissionPending_` を独立させる（§6.1 のコード例を修正）。push 失敗時は `pendingIntentCount_` から rollback し、`recoveryAdmissionPending_` に切替
- **X2**: 4層の sequence 分離（§6.2）。contiguous completion 前提で `store(seq, release)` で十分。wraparound は案A（architecturally impossible）
- **X6**: 3 semantic 厳密固定。`quarantineResidentCount_` = `DSPQuarantineManager::residentCount()` のみ
- **実装順序**: X1 を X4 より先に設計固定（§6.9）。X1/X2 は correctness の根幹

**acceptance criteria（§6 冒頭）**: INV-X1〜INV-X6 をコードレベル不変条件として固定 + X1/X2/X3 は正常系だけでなく queue-full / out-of-order / shutdown race / reader re-entry の adversarial test を必須。

### A-2.16 追加調査による X1-X6 詳細設計の更なる精緻化（2026-08-09）

**調査観点**: X1-X6 の実装対象コードの正確な構造（EpochDomain / RCUReader / RuntimeWorldAuthority / waitForPublishReceipt 呼び出し元）を深く調査し、実装詳細を確定した。

| # | 対象 | 追加調査結果 | §6 反映 |
|---|------|-------------|---------|
| U1 | **X3: reader 登録の実体** | `EpochDomain::kMaxReaders = 64`（core/EpochDomain.h:22）。`registerReaderThread`（:45-65）は slot を CAS 確保。`RCUReader::enter`（core/RCUReader.h:65）→ `acquireThreadSlot` → `registerReaderThread`。`audioThreadRcuReader`（AudioEngine.h:4529）が Audio callback（BlockDouble.cpp:151）で enter/exit | **§6.3 に追記**（readerRegistrationClosed は EpochDomain の shutdown フラグで実現） |
| U2 | **X1: submitRecoveryIntent の起床統合** | `submitRecoveryIntent`（AudioEngine.h:4274-4287）: submitRecoveryRequest → `recoveryPending = true`（:4283）→ `rebuildCV.notify_all()`（:4286）。recoveryAdmissionPending_ は submitRecoveryRequest 内で set し、起床は既存 recoveryPending が担保 | **§6.1 に追記** |
| U3 | **X4: RuntimeWorldAuthority の commit 委譲** | `RuntimeWorldAuthority`（RuntimeWorldAuthority.h:78-123）は `coordinator_`（core Coordinator&）参照に commit を委譲（:112）。X4 最終形は `coordinator_` を `RuntimeStore::WriteAccess` に置換（Owner 変更は大規模リファクタ） | **§6.4 に追記** |
| U4 | **X2: per-request receipt の呼び出し元** | `waitForPublishReceipt(seqId, timeout)` は `commitRuntimePublication`（AudioEngine.h:4450）で呼ばれる。seqId は Producer 自身の割当。タイムアウトしても Transferred 扱い（所有権は移譲済み） | **§6.2 に追記** |

**確定結果**: X3 の readerRegistrationClosed は `EpochDomain` の shutdown フラグで registerReaderThread が失敗することを保証（U1）。X1 の recoveryAdmissionPending_ は submitRecoveryIntent の既存起床経路（recoveryPending + rebuildCV）と統合（U2）。X4 は coordinator_ 参照の WriteAccess 置換（U3）。X2 の per-request receipt は commitRuntimePublication:4450 を対象（U4）。全て実装時に利用可能な精度に確定。

### A-2.17 十六次レビュー（dash(6) 最終設計レビュー・X4 NO-GO）（2026-08-09）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準コードとして、`REPAIR_PLAN2-dash(6)` を ISR/RCU・ownership/lifetime・publication ordering・queue residency・shutdown・Authority Singularization の観点から検証。

**総合判定**: **P2-1〜P2-4 は GO。X1/X2/X3/X6 は条件付き GO。X4 は現状 NO-GO（rename だけでなく RuntimeStore ownership topology の再設計が必要）**。

| X | 判定 | 実装条件 |
|---|------|---------|
| X1 Recovery | 🟠 条件付き GO | **reservation / coalesce state machine を確定**。1 logical Recovery admission = exactly one reservation（INV-X1-5）。durable admission を queue residency と二重計上しない（INV-X1-6） |
| X2 Completion | 🟠 条件付き GO / 最優先 | **FIFO completion invariant を実装契約として強制**（INV-X2-6: completion order == publication sequence order） |
| X3 Reclaim | 🟠 条件付き GO | **readerRegistrationClosed を shutdown state machine に統合**（INV-X3-4: permanently closed） |
| X4 Authority | 🔴 現状 NO-GO | **rename だけでなく RuntimeStore ownership topology を再設計**（INV-X4-3: publishAndSwap は RuntimeWorldAuthority-owned WriteAccess のみ） |
| X5 Publish residency | 🟢 GO | そのまま採用可能 |
| X6 Quarantine | 🟠 条件付き GO | **Ring/Intent/Resident/RetireQuarantine を4分離**（INV-X6-4: no counter may represent both transport and DSP residency） |

**追加 INV（dash §6 に反映）**: INV-X1-5 / INV-X1-6 / INV-X2-6 / INV-X3-4 / INV-X4-3 / INV-X6-4。

**dash §6 への反映**:
- **X1**: reservation ownership の移動 state machine（No Admission → Reserved → Queue Residency / Durable Pending → Reservation released）を追記。`PendingRecoveryAdmission` に `reservationOwned` を追加
- **X2**: INV-X2-6（completion order == publication sequence order）を追記
- **X3**: readerRegistrationClosed を shutdown state machine（Running → StopAdmission → StopAudio → CloseReaderRegistration → ... → Reclaim）に統合。INV-X3-4
- **X4**: 現状 NO-GO。RuntimeStore の物理所有関係（RuntimeWorldAuthority が Store を所有）まで変更。INV-X4-3
- **X6**: RetireQuarantineStore（RetireQuarantineStore.h:60, kMaxQuarantinedEntries=512）を含む4分離 + 状態遷移表

**実コード検証**: RetireQuarantineStore（:69 quarantine / :77 drain / :157 residentCount）が retire 対象の quarantine 退避を管理。RuntimeWorldAuthority は coordinator_ 参照に commit 委譲（:112）し、本当の Store owner ではない。

**最終評価**: dash(6) は「完成設計」ではなく「実装直前の最終設計レビュー版」。核心は **Intent → Admission → Transport Residency → Execution → Committed → Completed → Resident → Retired → Reclaimable → Deleted** の semantic state machine を一つずつ別の状態として閉じること。現時点では X1 と X4 の state machine 境界がまだ完全には閉じていない。

### A-2.18 X1-X6 新設 counter の宣言位置確定（2026-08-09）

**調査観点**: X1〜X6 の新設 counter / durable state の**宣言位置**（どのクラスのメンバとして追加するか）を、実装対象クラスのメンバ構造から確定した。

| # | 対象 | 宣言位置 | dash §6 反映 |
|---|------|---------|-------------|
| D1 | **X5: `publicationIntentResidencyCount_`** | ISRRuntimePublicationCoordinator.h:383（`publicationBacklogCount_` の隣） | **§6.5 に追記** |
| D2 | **X6: `quarantineIntentResidencyCount_` / `quarantineRingResidencyCount_`** | ISRRuntimePublicationCoordinator.h:388（`quarantineResidentCount_` の隣） | **§6.6 に追記** |
| D3 | **X6: `retireQuarantineResidentCount_`** | **新 counter は追加しない**（十八次別視点3で確定）。RetireQuarantineStore の既存 `size_`（:175）/ `residentCount()`（:157）を source of truth に。isFullyDrained は `m_retireRouter->retireQuarantineStore().residentCount() == 0` を評価 | **§6.6 に追記** |
| D4 | **X1: `PendingRecoveryAdmission` / `recoveryAdmissionPending_`** | ISRRuntimePublicationCoordinator.h:437（`recoveryIntentDropCount_` の隣） | **§6.1 に追記** |

**確定結果**: X5 の publicationIntentResidencyCount_ は publicationBacklogCount_（:383）の隣（Publish 系 counter 集約）。X6 の intent/ring counter は quarantineResidentCount_（:388）の隣、retireQuarantineResidentCount_ は RetireQuarantineStore 側。X1 の PendingRecoveryAdmission は plain 構造体（SPSC のため atomic 不要）、recoveryAdmissionPending_ は atomic<bool>（isFullyDrained が NonRT から読むため）。全て実装時の宣言位置が確定。

### A-2.19 十七次レビュー（dash(7) 追加調査指示・X4 NO-GO 継続）（2026-08-09）

**レビュー観点**: `ConvoPeq(20260809-104710).md` を基準コードとして、dash(6)（十六次反映版）の追加調査指示を確認。判定は **X5 GO / X1/X2/X3/X6 条件付きGO / X4 NO-GO** を維持しつつ、以下の追加の詳細設計調査指示が付された。

**新規指示・反映状況**:

| # | 指示内容 | 反映先 |
|---|---------|--------|
| 1 | **X1**: `PendingRecoveryAdmission` に `reservationOwned`。同一 generation の coalesce では reservation を増やさない | ✅ 反映済み（§6.1, :1099-1107） |
| 2 | **X1**: 単純な `pending = newest` は禁止。G10 を捨ててよいかは `isRebuildObsolete(G10)` で判定 | ✅ **本セッションで追記**（§6.1, :1121-1130） |
| 3 | **X2**: `waitForPublishReceipt()` は削除しない。username API semantic（全体 watermark vs 自分の request 完了）を統合してはいけない | ✅ 反映済み（§6.2, :1314-1318） |
| 4 | **X3**: `readerRegistrationClosed` を shutdown state machine に組み込み、re-entry を永久禁止 | ✅ 反映済み（§6.3, :1479-1492） |
| 5 | **X4**: **Physical write owner と Architectural authority surface を区別**。現状の「write authority は一本化されている」は狭義のみ正しく、architectural authority surface は一本化されていない | ✅ **本セッションで追記**（§X4, :1631-1642） |
| 6 | **X5**: INV-X5-1 の定義（queue residency + producer reservation、producer quiescence 後に ==） | ✅ 反映済み（§6.5） |
| 7 | **X6**: Intent/Ring/Resident/RetireQuarantine の4層分離 | ✅ 反映済み（§6.6） |
| 8 | **ISR の2層**: realtime safety（lock-free/bounded）と realtime semantic correctness（counter の意味/ownership/lifetime）を区別。X1〜X6 は B を閉じる改修 | ✅ **本セッションで追記**（§6 冒頭, :977-992） |
| 9 | **実装順序**: 設計固定と実装を分離。先に X2/X1/X4/X3 の invariant を決定 → 実装は P2→X5→X6→X2→X1→X4→X3 | ✅ **本セッションで追記**（§6.9, :1932-1950） |
| 10 | **acceptance test を必須化**: X1（exactly one valid Recovery admission → reservation == 0）、X2（commit order == completion order を architectural test）、X3（readerRegistrationClosed == false → reclaim forbidden）、X4（publishAndSwap の caller が RuntimeWorldAuthority 以外に存在しない）、X5（enqueue→+1/pop→-1/full→rollback/deferred→unaffected）、X6（Intent enqueue→intent+1/pop→intent-1/success→resident+1/reclaim→resident-1/Intent pending で resident==0/Intent==0 で resident==1） | ✅ 反映済み（§6.8, :1909-1928） |

**最終評価**: dash(6) の結論（X1 と X4 の state machine 境界が未完成）は十七次でも維持。**追加の詳細設計調査指示は本セッションで全件反映済み**。X1〜X6 は全 counter の宣言位置（D1-D4）・型・挿入点・state machine・不変条件・テスト計画が確定し、**設計固定（X2/X1/X4/X3 invariant）→ 実装（P2→X5→X6→X2→X1→X4→X3）** の手順で開始可能なレベルに到達。

**次回の実装手順**:
1. **設計固定**: X2 / X1 / X4 / X3 の invariant（INV-X2-6 / INV-X1-5,6 / INV-X4-3 / INV-X3-4）をコード契約として決定
2. **Phase 1**: P2-1〜P2-4（pendingIntentCount_ accounting + shutdown drain）
3. **Phase 2**: X5 → X6（residency semantics の基礎）
4. **Phase 3**: X2 → X1（completion / Recovery の correctness の根幹）
5. **Phase 4**: X4（ownership topology 再設計・独立リファクタ）→ X3（reclaim authority）
6. **最終**: 統合 shutdown / soak

### A-2.20 X4 詳細改修計画（十七次追加調査・X4-A/X4-B 分離確定）（2026-08-09）

**レビュー観点**: 最新の X4 検討結果を dash に反映。**「X4 = naming convergence + physical ownership convergence」として設計固定**。INV-X4-3 が追加され、従来案の「名前を分けるだけ」では不十分と判断。

**実コード検証で確定した事実**:
1. **二重の write surface が存在**:
   - `ISR commit()`（ISRRuntimePublicationCoordinator.cpp:80-115）: `currentWorld_` の atomic 更新のみ（メタデータ）。**RuntimeStore を触らない**
   - `core publishWorld()`（RuntimePublishExecutor.h:55-56 経由）: `RuntimeStore::WriteAccess::publishAndSwap()` — **唯一の store-swap**
2. **publishWorld の直接呼び出し元3箇所**: PublishExecutor（:55-56）/ Bootstrap（AudioEngine.Init.cpp:53-54）/ shutdown clear（AudioEngine.Processing.ReleaseResources.cpp:436-438）
3. **Store は AudioEngine のメンバ**（AudioEngine.h:3546 `RuntimePublishStore runtimeStore;`）。Store の Owner = `convo::RuntimePublicationCoordinator`（template パラメータ）
4. **RuntimeWorldAuthority は既に ownerChannel_ と lifetime_ を値保持**（RuntimeWorldAuthority.h:152-154）。X4-B の最終形（ownerChannel + LifetimeState）は既に Authority にある
5. **`testRuntimeWorldAuthorityAdapter`（ISRSemanticValidationTests.cpp:641）は「pure delegate」検証** — X4-B で仕様変更が必要（Authority owns Store に）

**X4 の二段階分離（十七次確定）**:
- **X4-A（semantic/name convergence）**: `convo::isr::RuntimePublicationCoordinator → RuntimeIntentCoordinator`（7ファイル）、`convo::RuntimePublicationCoordinator → RuntimePublishAuthority`（AudioEngine.h:3509 alias / :3646 factory）。rename のみ・低リスク・ownership topology 不変
- **X4-B（physical ownership convergence）**: `RuntimeStore<World, Owner>` の Owner を core Coordinator から **RuntimeWorldAuthority に変更**。`friend Owner` 変更、`WriteAccess` の値保持、`coordinator_` delegate 削除、一時生成 publishWorld 廃止。**最大リスク・単純な rename として扱わない**

**INV 確定（実コード検証済み）**: INV-X4-1（Intent authority singularity）/ INV-X4-2（Publish execution singularity、Bootstrap・shutdown clear は例外）/ INV-X4-3（**publishAndSwap は RuntimeWorldAuthority-owned WriteAccess のみ**）/ INV-X4-4（No RT ownership transfer）/ INV-X4-5（No second store）

**テスト再設計（8本）**: Test1 Authority owns Store（compile-time）/ Test2 WriteAccess move-only / Test3 publishAndSwap 唯一性 / Test4 PublishExecutor bypass 禁止 / Test5 Coordinator bypass 禁止 / Test6 二重 Store 検出 / Test7 publish sequence monotonicity / Test8 ownership transfer exactly once。**既存 `testRuntimeWorldAuthorityAdapter` は仕様変更（pure delegate → owns Store）**

**実装順序**: X4-0 Baseline → X4-1 Authority map コメント固定 → X4-2 ISR rename → X4-3 core rename → X4-4 compile/test → X4-5 Store ownership migration → X4-6 WriteAccess migration → X4-7 publishWorld 一時生成削除 → X4-8 PublishExecutor 接続 → X4-9 static architectural tests → X4-10 full regression

**追加調査で確定した未確定事項（本セッション）**:
1. **読み取り側の影響範囲**: core rename（X4-3）は write 側だけでなく、`RuntimePublicationCoordinator::consumeWorldHandle / acquireReadToken / consumePublishedWorld` の**読み取り static 関数呼び出し10箇所以上**（AudioEngine.h:1331/2119/3116/3383/3691、AudioEngine.Commit.cpp:559、AudioEngine.Processing.Latency.cpp:91、observePublishedWorld 経由多数）も改名対象。**🔴 十九次レビューで確定**: X4-B 後の read path は `getCurrent()`（currentWorld_ 由来）に単純置換してはいけない（別 read source）。read path は Authority へ段階的移行（X4-B-3）し、`consumeWorldHandle(runtimeStore)` は Store の read API を維持 or Authority の read authority API を新設
2. **currentWorld_ と runtimeStore.current の二重管理**: publish world pointer の atomic が2つ（ISR currentWorld_ :380 / RuntimeStore current :88）。X4-B は案1（二重管理維持・write authority 一本化のみ）に限定し、案2（currentWorld_ 廃止）は将来タスクとして記録
3. **2段階 write の一貫性**: X4-B 後も commit（ISR metadata）→ publishAndSwap（physical swap）の順序は維持（逆転禁止・Test 7 commit-before-swap ordering で検証）。**🔴 十九次レビュー §17 で境界を明確化**: commit-before-swap ordering は X4 の責務（authority.publish() で固定）、completion monotonicity は X2 の責務（X4 の Test 7 は ordering 検証、sequence monotonicity は X2 test suite に残す）。2段階の原子性は X2（completion ordering）の責務

**最終合格条件**: INV-X4-3（RuntimeStore::publishAndSwap() → exactly one → RuntimeWorldAuthority-owned WriteAccess）。

**最大リスク**: X4-B の `RuntimeStore<World, Owner>` Owner 変更。RuntimeStore / RuntimeWorldAuthority / RuntimePublishAuthority / RuntimePublishExecutor の4点を一つの ownership migration として実装・検証する。

**dash §6 への反映**: §6.4-X4 詳細改修計画セクション（6.4-X4 冒頭〜最終形）に全文反映済み。詳細は §6.4-X4 を参照。

### A-2.21 X1-X6 詳細設計の実コード精緻化（十八次調査）（2026-08-09）

**調査観点**: X1〜X6 の詳細設計を、実装対象コードの**正確な実装**（関数シグネチャ・挿入位置・counter 更新箇所・テスト対象）と突き合わせて精緻化。未確定事項を確定した。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X1 | **矛盾解消**: 「push 失敗時に pendingIntentCount_ を rollback しない（reservation 維持）」の旧記述を、**十五次 §14 の最新判断（rollback + recoveryAdmissionPending_ 切替）に統一**。旧記述（十一次/十二次追記 :1229,1243）を修正 | §6.1 :1229-1243 |
| 2 | X1 | **PendingRecoveryAdmission に handle/epoch/intentId を追加**（RecoveryIntent :166-179 と一致）。消費時 RebuildDispatch.cpp:917 の `qHandle.isNull()` 検証に必要。INV-X1-4 は「World ポインタを持たない」意味（DSPHandle は非所有）と確定 | §6.1 :1184-1195 |
| 3 | X1 | **takePendingRecoveryAdmission の実装詳細**を確定: SPSC（Producer=CoordinatorLoop, Consumer=Builder Loop）で競合なし。durable 化済みのため pendingIntentCount_ を触らない。build 失敗時は durable は消費済み（take でクリア）→ 次サイクル再発行 | §6.1 :1279-1299 |
| 4 | X1 | **Builder 消費ループの追記**を確定: RebuildDispatch.cpp:911 の while ループ**後**に takePendingRecoveryAdmission の処理を追加。既存 :911-972 と同じ処理パス（build/IR rebuild/warmup/publish）を再利用 | §6.1 :1231-1277 |
| 5 | X2 | **PublishReceiptWaiter::complete() の現行実装を検証**: mutex ガード付き `if (seqId > lastCompleted_)`。contiguous completion 前提では単調増加のみ。案1（最小変更・mutex 維持）を確定。`m_lastObservedSequence` と `lastCompleted_` の2 watermark 同期をテスト対象に | §6.2 :1447-1491 |
| 6 | X2 | **completion 経路を完全追跡**: PublishExecutor:84 → onPublishCommitted（Orchestrator.cpp:305）→ m_lastObservedSequence + notifyPublishReceipt → complete() → lastCompleted_。Producer 側 waitForPublishReceipt との対応を確定 | §6.2 :1471-1491 |
| 7 | X3 | **readerRegistrationClosed の実装を確定**: EpochDomain に `registrationClosed_` atomic<bool> + `closeReaderRegistration()`。registerReaderThread 冒頭で `-1` を返す。**「kMaxReaders 全消費」案は NO-GO**。reserveReaderThread にもガード追加 | §6.3 :1633-1660 |
| 8 | X3 | **既存 reader への影響**: フラグは新規登録のみ拒否。登録済み slot（audioThreadRcuReader/messageThreadRcuReader）は exit まで動作継続 | §6.3 :1652-1655 |
| 9 | X5 | **fallback queue に Publish が入らないことを確定**: quarantineFallbackQueue_ は Quarantine 専用（submitQuarantine のみ push）。publicationIntentResidencyCount_ の減算は intentQueue_ の while ループ（:36-37）のみ | §6.5 :2238-2251 |
| 10 | X6 | **submitQuarantine の intent/ring counter 移動を確定**: fallback 移動時は quarantineIntentResidencyCount_-- して quarantineRingResidencyCount_++（両方が同時に 1 にならない）。十二次レビューの旧コードを修正 | §6.6 :2354-2381 |
| 11 | X6 | **pop 減算の一元管理を確定**: quarantine の intent/ring counter 減算は processIntent の while ループで実施（handler では行わない）。X5 の Publish 減算と同じパターン。handler は純 routing（HANDLER-1）維持 | §6.6 :2388-2433 |
| 12 | 全体 | **Publish intent は pendingIntentCount_ に含まれない**（ISRSoakTests.cpp:70 コメント確認）。§1.1（P2）の「Publish pop は pendingIntentCount_ から除外」方針と整合 | §6.5 |

**結果**: X1〜X6 は**実装可能な精度に到達**。counter の宣言位置（D1-D4）・更新箇所・関数シグネチャ・テスト対象・既存コードとの整合が全て確定。残る未確定事項なし（保留事項は §6.9 の実装順序と A-2.20 の将来タスクとして明記）。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B）→ X3 → 統合 shutdown/soak

### A-2.22 X1-X6 別視点調査（スレッド所有権・外部 setter 干渉・メモリオーダリング）（2026-08-09 十八次・別視点）

**調査観点**: 前回（A-2.21）の関数シグネチャ・挿入位置に加え、**スレッド所有権・外部 setter との干渉・メモリオーダリング・isFullyDrained 実装・RCUReader 経路**の別視点から X1〜X6 を検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 別視点の確定事項 | §6 反映 |
|---|------|-----------------|---------|
| 1 | 全体 | **`AudioEngine::isFullyDrained()`（Threading.cpp:114-136）の外部 setter が X1/X5/X6 と干渉**: `setPendingIntentCount(hasDeferredCommit ? 1u : 0u)`（:117）と `setQuarantineResidentCount(ringResident + dspQuarantineResident)`（:131）は、X1/X6 の新設 counter を認識しない。X1〜X6 実装時は外部上書きを廃止し、Coordinator 内部で純粋 accounting | §6.7 |
| 2 | X1 | **`recoveryAdmissionPending_` と Threading.cpp:117 の干渉**: durable 化後に pendingIntentCount_=0 でも recoveryAdmissionPending_=true の状態が isFullyDrained で見落とされる可能性。:117 廃止 + isFullyDrained は両方見る | §6.1 |
| 3 | X2 | **complete() の thread 所有権**: CoordinatorLoop::run（ISRCoordinatorLoop.cpp:31-43）の単一スレッドから complete() が呼ばれる。waitFor() は Producer スレッド。mutex は cv 同期に必要（data race は構造的に排除）。shutdown 中は CoordinatorLoop が break し complete() が来ない → Producer の waitFor は timeout で Transferred（既存設計が正しい） | §6.2 |
| 4 | X3 | **2つの ShutdownPhase enum**: AudioEngine（:2521-2530）と ISRShutdown（:25-41）が独立。CloseReaderRegistration はどちらにも存在しない。実装は `setShutdownPhase(StopAudio)` 完了時に registrationClosed_ を副作用として set（enum 変更最小化） | §6.3 |
| 5 | X4 | **friend 関係**: PublishExecutor（friend struct, AudioEngine.h:3580）は runtimeStore に直接アクセス。X4-B で worldAuthority() 経由に置換。WriteAccess の friend Owner 変更（core/RuntimeStore.h:81）は PublishExecutor に影響しない（authority.publish() 経由のため） | §6.4-X4-B |
| 6 | X5 | **メモリオーダリング**: fetchAdd/fetchSub は default acq_rel（AtomicAccess.h:91-105）。X5 の新設 counter も acq_rel を明示。Producer（NonRT）と Consumer（CoordinatorLoop）の2スレッド間 RMW のため race なし | §6.5 |
| 7 | X6 | **メモリオーダリング**: X6 の counter も acq_rel。quarantineResidentCount_ は fetchAdd しない（DSPQuarantineManager 内部管理） | §6.6 |
| 8 | X6 | **residentCount() の実装**: DSPQuarantineManager::residentCount()（ISRDSPQuarantine.cpp:103-111）は quarantineActiveFlags_ を走査。X6 は走査を維持し（新 atomic 追加しない）、isFullyDrained は NonRT で直接読む。kMaxSlots=256 の走査は NonRT で許容 | §6.6 |

**結果**: 別視点（スレッド所有権・外部 setter・メモリオーダリング）からも X1〜X6 の詳細設計が確定。**外部 setter 干渉（#1/#2）は X1/X6 実装時の必須対応**。残る未確定事項なし。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B）→ X3 → 統合 shutdown/soak

### A-2.23 X1-X6 共通パス・テスト基盤調査（2026-08-09 十八次・別視点2）

**調査観点**: 前回までと異なり、**Publish の共通 enqueue 経路（enqueuePublicationIntentForRuntimeCommit → submitPublishRequest）、deferred 再 enqueue、observeDeferredRing_ の counter 管理、テスト基盤（CMakeLists ・既存テスト配置）**から X1〜X6 を検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X5 | **Publish intent の全 enqueue 経路（3経路）**: 経路1（RebuildThread・通常 rebuild）/ 経路2（Builder Loop・Recovery publish）/ 経路3（RebuildThread・deferred 再 enqueue）。すべて `enqueuePublicationIntent`（:273）に集約。X5 の counter は単一箇所で reservation され二重計上なし。`enqueuePublicationIntentForRuntimeCommit`（:688）は直接 push しない | §6.5 |
| 2 | X1 | **Recovery publish と X5 counter の相互作用**: Recovery publish（RebuildDispatch.cpp:971 → 経路2）も X5 の counter を +1。X1 の durable admission と X5 は独立。Recovery build 中の「build gap」はどちらの counter にも含まれない（isFullyDrained で注意） | §6.1 |
| 3 | 全体 | **observeDeferredRing_ の pendingIntentCount_ 減算（P2 との接続点）**: submitObserve（:561-562）は deferred ring にも +1 するが、drainObserveDeferred（ProcessIntent.cpp:47-56）は減算しない。現状は setPendingIntentCount(0)（:43）で整合。**§1.1（P2）で setPendingIntentCount(0) を廃止する場合、drainObserveDeferred の pop でも -1 が必要** | §6.6 |
| 4 | 全体 | **テスト基盤（CMakeLists 確認）**: ISRSemanticValidationTests（:181）/ ISRSoakTests（:219）/ RuntimeWorldAuthorityProjectionTests（:295）が登録済み。X1/X3/X5/X6 は Coordinator/EpochDomain/DSPQuarantineManager のメソッドなのでヘッドレスで直接テスト可能。X2 の PublishReceiptWaiter は AudioEngine の private メンバのため AudioEngineHarness で統合テスト | §6.8 |
| 5 | X2 | **PublishReceiptWaiter のテスト基盤**: AudioEngineHarness（AudioEngineHarness.h）で publish パイプライン全体（commitRuntimePublication → OwnerChannel → IntentQueue → CoordinatorLoop → executePublish → onPublishCommitted → receipt）を実スレッドで通し、waitForPublishReceipt と m_lastObservedSequence の同期を検証 | §6.8 |
| 6 | X4 | **静的検査コマンド確定**: `rg -n "publishAndSwap\(" src/`（WriteAccess 定義以外が RuntimeWorldAuthority-owned のみ）/ `rg -n "\.commit\((PublishAuthority\|Granted)" src/audioengine/`（PublishExecutor のみ）/ `rg -n "RuntimeStore<RuntimePublishWorld" src/`（Store 1箇所のみ） | §6.8 |

**結果**: Publish 共通パス・テスト基盤の観点からも X1〜X6 が確定。X5 は全3 enqueue 経路で counter が正しく機能すること、X2 は AudioEngineHarness で統合テスト可能なことを確認。残る未確定事項なし。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B）→ X3 → 統合 shutdown/soak

### A-2.24 X1-X6 実装詳細・Producer 前提・reclaim 管理調査（2026-08-09 十八次・別視点3）

**調査観点**: RetireQuarantineStore 実装・shutdownReclaim 呼び出し元・requestReclaim 内部・sequence 採番スレッド・submitRecoveryRequest Producer 前提を実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X6 | **`retireQuarantineResidentCount_` は新 counter 不要（設計変更）**: RetireQuarantineStore は既に `size_`（:175, mutex 保護）+ `residentCount()`（:157-161）を持つ。`quarantine()`（:69-93）で ++size_ / `drain()`（:98-131）で size_=w / `drainAllUnsafe()`（:134-154）で size_=0。**新 atomic counter を追加せず、既存 residentCount() を source of truth にする**（2重 source の不整合リスク回避。INV-X6-4 の精神）。isFullyDrained は `m_retireRouter->retireQuarantineStore().residentCount() == 0` を評価 | §6.6 / §6.7 / A-2.18 D3 |
| 2 | X1 | **lost-wakeup 整合（takePendingRecoveryAdmission と recoveryPending）**: Builder は recoveryPending クリア（:905）後に popRecoveryRequest（:911）→ X1 追記で takePendingRecoveryAdmission。消費中に新規 durable 化があれば recoveryPending 再 set（:4283）+ notify で次サイクル再消費。recoveryAdmissionPending_ が durable 有無の真実（take が false なら nullopt で自然終了） | §6.1 |
| 3 | X3 | **shutdownReclaim 呼び出し元の順序**: CacheMap::~CacheMap（AudioEngine.h:2015-2037）は `delete` 先（:2026）→ shutdownReclaim（:2027）。ReleaseResources.cpp:410-421 は retire（:414,419）→ shutdownReclaim（:415,420）。X3 移行時もこの順序を維持（delete→reclaim / retire→reclaim）。再 retire は冪等 | §6.3 |
| 4 | X3 | **reclaimInFlightCount_ の管理**: setReclaimInFlightCount(+1)（:592）は「epoch 不安全で遅延」、setReclaimInFlightCount(0)（:606）は「完了」。複数 handle の並行 pending を正確に数えない簡略設計。ShutdownQuiescent は epoch 判定スキップのためカウンタを呼ばない（Reader 停止済みで遅延なし） | §6.3 |
| 5 | X2 | **sequence 採番の thread 所有権**: reserveRuntimePublicationIdentity（AudioEngine.h:3407）は RuntimeBuilder.cpp:81,183（RebuildThread）で呼ばれる。採番→commit が同一スレッド（RebuildThread）で進行するため、Producer serialization が成立（INV-X2-5 の前提） | §6.2 |
| 6 | X1 | **Producer 単一スレッド前提の完全確認**: submitRecoveryRequest の呼び出し元は 1 箇所（AudioEngine.h:4277）。RecoveryIntentHandler（ProcessIntent.cpp:126-132）は dead code。Producer = CoordinatorLoop 単一スレッド（QuarantineIntentHandler 経由）。SPSC 前提が完全成立 | §6.1 |

**結果**: X6 の retireQuarantineResidentCount_ 設計を「新 counter 追加」から「既存 residentCount() 使用」に変更。X1 の Producer 前提・X3 の reclaim 管理・X2 の採番スレッドを完全確定。残る未確定事項なし。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B）→ X3 → 統合 shutdown/soak

### A-2.25 十九次レビュー反映（X4-B の currentWorld_ 意味論修正・条件付き GO から実装着手可能 GO へ）（2026-08-09）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準コードとして、dash(8) の X4 を実装可能性・ownership/lifetime・publication ordering・RCU/ISR・Authority Singularization・shutdown の観点から再検証。

**総合判定**: **X4-A = GO。X4-B = 条件付き GO（ただし修正3点を入れれば実装着手可能な GO）**。`currentWorld_` の「metadata write」整理と `getCurrent()` の core read 置換先案に意味論上の未解決問題。

**レビュー指摘・反映状況**:

| # | 指摘 | 反映先 |
|---|------|--------|
| 1 | **`currentWorld_` は metadata cache ではなく、RuntimeWorld pointer の第二の publication/read surface**。`commit()`（:109-112）は pubWorld->publication を書込み currentWorld_ を更新。`getCurrent()`（:169-171）/ `getVersion()`（:173-178）/ `currentPublicationEpoch()`（:180-185）/ `currentPublicationSequenceId()`（:189-193）は currentWorld_ から導出 | ✅ **実コード検証で確定**（§6.4-X4-B 二重管理セクション） |
| 2 | **`getCurrent()` を `consumeWorldHandle(runtimeStore)` の置換先にすることは NO-GO**。getCurrent() は currentWorld_、consumeWorldHandle は RuntimeStore::current と**別の atomic source**。単純置換は read source の意味を変更する | ✅ **§6.4-X4-B に NO-GO を追記**（read/write authority 分離 + 段階移行 X4-B-1/2/3） |
| 3 | **`commit → publishAndSwap` を「一つの atomic publication」とみなすのは NO-GO**。`publish()` は semantic publication transaction の唯一の execution boundary と定義し、currentWorld_ と runtimeStore.current の同時 atomic 更新は保証しない | ✅ **§6.4-X4-B に追記**（レビュー §10） |
| 4 | **X4 と X2 の境界明確化**: commit-before-swap ordering は X4 の責務（authority.publish() で固定）、completion monotonicity は X2 の責務 | ✅ **§6.4-X4-B と A-2.20 に追記** |
| 5 | **INV-X4-6 / INV-X4-7 を追加**: 二重 atomic 暫定許容でも二つの Authority を許さない | ✅ **§6.4-X4 INV に追加** |
| 6 | **`currentWorld_` は non-owning observation alias**（ownership source でない）。delete/retire source に使用禁止 | ✅ **INV-X4-6/7 と共に追記** |
| 7 | **RuntimeWorldAuthority 自体の move/copy 禁止**（Store + WriteAccess 所有のため） | ✅ **§6.4-X4-B に追記**（static_assert 含む） |
| 8 | **Bootstrap/shutdown を「例外」でなく「lifecycle-controlled publish」と定義**。sole gateway は「PublishExecutor」でなく「RuntimeWorldAuthority」 | ✅ **INV-X4-2 と PublishExecutor 役割に追記** |
| 9 | **Test 7 は「publish sequence monotonicity」→「commit-before-swap ordering」に変更**。sequence monotonicity は X2 に残す | ✅ **§6.4-X4-B テスト表と §6.8 に反映** |
| 10 | **Test 9（dual-pointer semantic consistency）/ Test 10（INV-X4-7）を追加**（8本→10本） | ✅ **§6.4-X4-B テスト表と §6.8 に反映** |
| 11 | **実装順序 X4-0〜X4-10 → X4-0〜X4-11 に拡張**（X4-7 publish() 導入を独立、X4-9 全 direct publishWorld caller 排除を追加） | ✅ **§6.4-X4 実装順序に反映** |

**優先すべき追加修正3点（レビュー最終結論）**:
1. `getCurrent()` ≠ `consumeWorldHandle(runtimeStore)` を明文化 — ✅ 反映済み（§6.4-X4-B）
2. `commit-before-publishAndSwap` を X4 invariant として追加 — ✅ 反映済み（Test 7 + §6.4-X4-B）
3. `currentWorld_` / `RuntimeStore::current` の dual-pointer consistency test を追加 — ✅ 反映済み（Test 9 + INV-X4-6）

**最終評価**: 上記3点を反映した dash(8) は **X4-B = 実装着手可能な GO**。X4 は「publish を誰が決定するか」ではなく「publish の physical write capability を誰が所有するか」を厳密に一本化する改修として閉じる。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant・INV-X4-1〜7 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B）→ X3 → 統合 shutdown/soak

### A-2.26 X1-X6 deferred 経路・OwnerChannel・LifetimeState・overflow ring 調査（2026-08-09 十八次・別視点4）

**調査観点**: deferred publish 経路の詳細・OwnerChannel 実装・LifetimeState::pendingIntentCount と X5/X6 の関係・overflow ring と X6 の関係・deferred と X2 completion の整合を実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X4 | **OwnerChannel は SPSC**（OwnerChannel.h:38-118）: enqueue = Non-RT publish thread 単一 / take = ISR/audio thread 単一。capacity 256。key = (sequenceId, epoch, mappedGeneration)。`enqueue`（:67-87）は key 重複 reject / `take`（:92-108）は single-transfer drain。**owner.get() は non-owning read / std::move(owner) のみ transfer** が型で保証（X4 Test 8 の根拠） | §6.4-X4-B Test 8 |
| 2 | X5/X6 | **LifetimeState::pendingIntentCount() は retire intent の pending 数**（ISRRetire.cpp:182-189: enqueueTicket_ - dequeuePos_ + fallbackCount_）。Commit.cpp:462,604 が Coordinator の pendingIntentCount_ に混入（RetireIntent 系）。**X5/X6 の transport counter と完全に別 semantic**。setRetireBacklogCount への同値設定は妥当（:463,605） | §6.7 |
| 3 | X5 | **deferred は単一スロット**（Orchestrator.cpp:360-409: `deferredSlot_` + `hasDeferred_`）。deferredPublicationCount は 0/1 のみ。上書き時 deferredOverwriteCount_++ | §6.5 |
| 4 | X6 | **overflow ring は retire 系**（RetireOverflowRing / coordinatorDeferredRing_ / lastResortQueue_ は retire intent の overflow）。**X6 の quarantineRingResidencyCount_ = quarantineFallbackQueue_ と独立**（設計変更なし）。drainOverflowRing の再注入は LifetimeState 側（emitRetireIntent）で Coordinator pendingIntentCount_ と独立 | §6.6 |
| 5 | X2 | **deferred publish と contiguous completion の整合（REPAIR_PLAN2.md:914）**: deferred は単一スロット（上書きで最新のみ残る）。INV-X2-6 は「deferred 非発生時」を前提とした invariant と明示。deferred で cancel された古い seqId の receipt は来ず、waitFor はタイムアウト（250ms）で Transferred 扱い | §6.2 |
| 6 | X1 | **recoveryIntentQueue_ の SPSC 確認**（ISRRuntimePublicationCoordinator.h:434, kRecoveryIntentQueueCapacity=256）: Producer=CoordinatorLoop / Consumer=Builder Loop。X1 の durable admission 設計と完全に整合 | §6.1 |

**結果**: deferred 経路・OwnerChannel・LifetimeState・overflow ring の観点からも X1〜X6 が確定。X5 の deferredPublicationCount は 0/1（単一スロット）、X2 の INV-X2-6 は deferred 非発生前提と明示。残る未確定事項なし。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant・INV-X4-1〜7 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B）→ X3 → 統合 shutdown/soak

### A-2.27 二十次レビュー反映（X4-B の Store ownership・publish() 責務・INV-X4-8・Test 定義修正）（2026-08-09）

**レビュー観点**: X4 詳細改修計画（十七次〜十九次反映版）を `ConvoPeq(20260809-022629).md` と `Practical Stable ISR Bridge Runtime.md` を基準に再検証。

**総合判定**: **X4-A / X4-B 分離と「RuntimeWorldAuthority を物理 write authority にする」方針は妥当**。ただし「Authority が Store を所有する」ことと「AudioEngine の lifetime/topology の中で安全に所有できる」ことは**別問題**。必須修正4点を反映すれば実装 GO に引き上げ可能。

**レビュー指摘・反映状況**:

| # | 指摘 | 反映先 |
|---|------|--------|
| 1 | **Store ownership と constructor/topology を確定**: 「Authority が Store を所有」と「AudioEngine の中で安全に所有」は別問題。**コンストラクタは外部 Store を受け取らず、Authority 自身が Store identity を形成**する（`runtimeStore_()` + `writeAccess_(runtimeStore_.acquireWriteAccess())`） | ✅ **§6.4-X4-B に反映** |
| 2 | **`publish()` の責務を「commit + physical swap」までに限定**: 現行 `publishWorld()`（:100-141）は seal/validate/release/swap/didPublish/willRetire/retire まで担うため、単純置換では成立しない。didPublish/willRetire/retire は publish() の後、PublishExecutor → completion → LifetimeState へ委譲 | ✅ **§6.4-X4-B 推奨 API に反映** |
| 3 | **read API を `getCurrent()` と分離し、RuntimeStore::current 専用の Authority read API を新設**: `observePublishedWorld()` / `acquireReadToken()` / `consumeWorldHandle(const ReadToken&)` | ✅ **§6.4-X4-A に反映** |
| 4 | **Test 7 を commit-before-swap ordering に、X2 側を completion monotonicity と明確分離** | ✅ **既反映（§6.4-X4-B Test 7 + テスト帰属表）** |
| 5 | **INV-X4-8 を追加（source-role separation・強く推奨）**: currentWorld_ = metadata observation alias / RuntimeStore::current = physical publication source。Neither API may treat them as interchangeable。delete/retire/unique_ptr/shared_ptr 変換を禁止 | ✅ **INV セクションに反映** |
| 6 | **INV-X4-6 の PublicationIdentity 構成要素を確定**: sequenceId + publicationEpoch + mappedGeneration を主 identity、version/boundary は metadata | ✅ **INV セクションに反映** |
| 7 | **Test 3 の検証対象を厳密化**: `RuntimeStore<RuntimePublishWorld, ...>::WriteAccess` まで追う。allowed = RuntimeWorldAuthority / forbidden = RuntimeIntentCoordinator, PublishExecutor, RuntimePublishAuthority, AudioEngine, Builder, DSPTransition | ✅ **Test 3 に反映** |
| 8 | **Test 6 の定義修正**: write-capable Store のみ禁止。read-only Store reference（`const RuntimeStore&`）は read API が保持してよい | ✅ **Test 6 に反映** |
| 9 | **`RuntimePublishAuthority` は WriteAccess を所有してはいけない**（二階層化の禁止） | ✅ **§6.4-X4-A + INV-X4-3 に反映** |
| 10 | **Bootstrap / shutdown clear は publish() と統合しない**: clearPublishedRuntimeSnapshotsNonRt は同じ physical write authority の下に別 API として置く（shutdown semantic は X3） | ✅ **§6.4-X4-B X4-8 に反映** |
| 11 | **実装順序を細分化**: X4-B-0〜X4-B-11（rollback point 多数確保） | ✅ **§6.4-X4 実装順序に反映** |

**必須修正4点（レビュー最終結論）**:
1. RuntimeWorldAuthority の Store ownership と constructor/lifetime topology 確定 — ✅ 反映
2. publish() の責務を「commit + physical swap」までに限定、retire/reclaim を入れない — ✅ 反映
3. read API を getCurrent() と分離、RuntimeStore::current 専用の Authority read API 新設 — ✅ 反映
4. Test 7 を commit-before-swap ordering、X2 側を completion monotonicity と明確分離 — ✅ 反映

**最終評価**: 上記4点を反映した dash(8) は **X4-B = 実装 GO**。X4 の目的は「二つの atomic を一つにすること」ではなく「**physical write capability を RuntimeWorldAuthority に一意化すること**」と固定。currentWorld_ / RuntimeStore::current の二重性自体を X4 で解消しない（publication semantics・read source・completion ordering・lifetime の複数問題を同時変更しない）。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant・INV-X4-1〜8 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.28 X1-X6 キュー基盤・generation・free list・複数 Producer 調査（2026-08-09 十八次・別視点5）

**調査観点**: X5/X6 の core である `intentQueue_` の基盤（MpscBoundedRing）の内部構造・producer hole・RuntimeBuildSnapshot の generation・X2 receipt の複数 Producer・X3 reclaim の free list ロックを実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X5/X6 | **MpscBoundedRing の producer hole と counter の整合**: `intentQueue_` は MpscBoundedRing（:43-136）。push() は reservation（CAS :76）→ payload 書込み（:80）→ seq release（:81）の2段階。**push は publication 完了後に return true/false するため、producer hole は push 内で完結**。X5 の fetchAdd 先行設計（counter +1 → push → 失敗 rollback）は、producer hole 中に counter が一瞬過大になるが、push の return 後は必ず収束（成功: 要素数と一致 / 失敗: rollback）。REPAIR_PLAN2.md:174(c) の「push 成功後に +1」とは順序が異なる点を明記 | §6.5 / §6.6 |
| 2 | X1 | **RuntimeBuildSnapshot.generation と recoveryGeneration の関係**: buildSource.generation は消費時（RebuildDispatch.cpp:968-969）に現在の rebuildRequestGeneration で上書きされる。**coalesce の generation 判定は PendingRecoveryAdmission.recoveryGeneration を使う**（buildSource.generation ではない）。isRebuildObsolete は recoveryGeneration に適用 | §6.1 |
| 3 | X2 | **waitFor の複数 Producer（5箇所）**: commitRuntimePublication（waitForPublishReceipt を含む同期 publish）の呼び出し元は PrepareToPlay.cpp:155,277 / ReleaseResources.cpp:175 / Timer.cpp:918 / Transition.cpp:25 / PublicationExecutor.cpp:53。複数スレッドから waitFor される。mutex は複数 Producer 間の cv 同期にも必要（data race は構造的に排除） | §6.2 |
| 4 | X3 | **DSPHandleRuntime::reclaim() の free list ロック**: reclaim()（ISRDSPHandle.cpp:129-148）は freeListMutex_（:133）をロックし、freeSlots_ に slot 返却（:146-147）。X3 の reclaim(ReclaimMode) は両モードとも同一の reclaim() を使用（free list の扱い共通）。RT スレッドからは呼ばない（INV-X4-4 と整合） | §6.3 |
| 5 | 全体 | **kDispatchTable の 1:1 mapping 確認**（ISRIntentDispatcher.h:60-65）: Observe/Publish/Recovery/Quarantine が各 Handler に 1:1 でルーティング。QuarantineIntentHandler は intentQueue_ と quarantineFallbackQueue_ の両方から dispatch される（ProcessIntent.cpp:32-33 + :36-37）。X6 の counter 減算は両ループで正しく分岐（§6.6 反映済み） | §6.6 |

**結果**: キュー基盤（MpscBoundedRing）の producer hole・generation の扱い・free list ロック・複数 Producer の観点からも X1〜X6 が確定。X5/X6 counter は producer hole に影響されず収束することを確認。残る未確定事項なし。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant・INV-X4-1〜8 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.29 二十一次レビュー反映（X4-B の型・constructor・member declaration 固定 + CRTP コンパイル検証）（2026-08-09）

**レビュー観点**: X4 計画を `ConvoPeq(20260809-022629).md` を基準に照合。**X4-A = GO / X4-B = 設計としては GO 寄り**。実装開始前に `RuntimeWorldAuthority` の完全な現行定義と `RuntimeStore` の型依存関係を詳細に照合する必要あり。

**指摘・反映状況**:

| # | 指摘 | 反映先 |
|---|------|--------|
| 1 | **① member declaration order を固定**: runtimeStore_ → writeAccess_ → ownerChannel_ → lifetime_ → registry_（WriteAccess が生きている間に Store が破棄されない） | ✅ **§6.4-X4-B 4点セクション** |
| 2 | **② RuntimeStore の World/Owner template dependency**: `RuntimeStore<World, RuntimeWorldAuthority>` の CRTP 的参照のコンパイル可能性を確認 | ✅ **§6.4-X4-B ② + 実コンパイル検証** |
| 3 | **③ publish() の ownership transfer（失敗経路・null経路）** | ✅ **§6.4-X4-B ③** |
| 4 | **④ Bootstrap / shutdown clear の初期化・破棄順序** | ✅ **§6.4-X4-B ④** |

**②の実コンパイル検証（g++ -std=c++20）で確定した重要事実**:
1. **`RuntimeStore<World, Self>` の CRTP は既存実績あり**（RuntimePublicationCoordinator.h:34 — 自身を Owner に）
2. **`friend Owner`（:81）は incomplete type でも動作** / **`static_assert(is_class_v<Owner>)`（:16）は incomplete でも well-formed**
3. **`Store::Owner` はコンパイル不可** — RuntimeStore.h に `using Owner` が存在しないため。**`using OwnerType = Owner` を追加**すれば Test 1 の static_assert が成立（検証済み: COMPILE_OK / RUN_OK）
4. **member としての WriteAccess は Store の complete type が必要**（RuntimeWorldAuthority.h が core/RuntimeStore.h を include）。RuntimeState は forward decl で十分（pointer のみ）
5. **循環依存なし**（RuntimeStore.h は AtomicAccess.h のみ include）

**結論**: X4-B の CRTP 懸念（incomplete-type / header include / nested WriteAccess）は全て解消済み。**X4-B は設計 GO・実装開始可能**。`RuntimeStore.h` への `using OwnerType = Owner` 追加が唯一の前提追加。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant・INV-X4-1〜8 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.30 X1-X6 shutdown 順序・保留再試行・shutdown 相互作用調査（2026-08-09 十八次・別視点6）

**調査観点**: shutdown シーケンスの実コード詳細（releaseResources / ~AudioEngine の2系統）・X3 の pendingReclaimHandles_ 再試行機構・ShutdownRuntime FSM と AudioEngine shutdownPhase の同期・shutdown と X1/X5/X6 の相互作用を実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X3 | **CloseReaderRegistration の挿入位置（2系統の shutdown）**: <br>系統1 releaseResources（ReleaseResources.cpp:34-404）: StopAcceptingWork→StopAudio→StopWorkers→ForceEpochAdvance→DrainRetire→GracefulDrain→ReclaimComplete→EmergencyDrain→RetireQuarantineStore drainAllUnsafe<br>系統2 ~AudioEngine（CtorDtor.cpp:92-231）: 同様の順序 + clearPublishedRuntimeSnapshotsNonRt<br>**両系統とも graceful drain（activeReaderCount==0 待ち）の前に closeReaderRegistration() を呼ぶ**（DrainRetire フェーズ開始前） | §6.3 |
| 2 | X3 | **drainDeferredRetireQueues の保留再試行機構**（Retire.cpp:41-114）: setReclaimInFlightCount(1)→tryReclaim→coordinator.reclaim→setReclaimInFlightCount(0) + pendingReclaimHandles_ 抽出（:60-64）→ isRetired ガード（:75）→ epoch 確認（:79）→ requestReclaim（:83）→ 失敗時再登録（:85-87）。**X3 の reclaim(RuntimeEBR) はこの機構を維持**（requestReclaim を reclaim に置換） | §6.3 |
| 3 | X1 | **shutdown 中の durable Recovery admission の破棄**: Builder 消費ループ（:901）は shutdown 中スキップされるため、durable admission が残る。**requestShutdown() 時に recoveryAdmissionPending_ = false + pendingRecoveryAdmission_ をクリア**（shutdown 中は publish/commit が実行されないため保持しても無意味）。isFullyDrained の `recoveryAdmissionPending == false` を成立させる | §6.1 |
| 4 | X6 | **shutdown 時の quarantine counter 自然収束**（ReleaseResources.cpp:363-404）: destroyForShutdown（:387）→ destroyQuarantineSlot（:390）→ lifetime().reclaim（:392）。quarantineResidentCount_（= DSPQuarantineManager::residentCount()）は destroyForShutdown で 0、RetireQuarantineStore::residentCount() は drainAllUnsafe（:376）で 0。**X6 counter は shutdown で自然収束**（新たな shutdown 専用処理不要） | §6.6 |
| 5 | X5 | **shutdown 時の counter 収束**: CoordinatorLoop（ISRCoordinatorLoop.cpp:31-43）は isShutdownInProgress() が false の間 processIntent を続行。requestShutdown から join までの間に intentQueue_ を drain し publicationIntentResidencyCount_ は 0 へ収束。join 後に残留があれば isFullyDrained が false（正常） | §6.5 |

**結果**: shutdown 順序の実コード詳細・保留再試行機構・shutdown 相互作用の観点からも X1〜X6 が確定。X3 の CloseReaderRegistration 挿入位置（2系統）・X1 の durable admission 破棄・X6/X5 の counter 自然収束を確定。残る未確定事項なし。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant・INV-X4-1〜8 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.31 二十二次レビュー反映（X4-B の commit ownership・Test 9 identity 限定・RuntimePublishAuthority 一切所有禁止）（2026-08-09）

**レビュー観点**: 6.4-X4 詳細改修計画を `ConvoPeq.md` を基準に一次レビュー。**X4-A/X4-B 分離・RuntimeStore 物理 ownership 移動・getCurrent() 非流用は妥当**。実装前に必須修正3点を反映。

**指摘・反映状況**:

| # | 指摘 | 反映先 |
|---|------|--------|
| 1 | **commit 二重化の危険（必須修正1）**: 現行 PublishExecutor（RuntimePublishExecutor.h:42-57）は既に authority.commit() を実行。X4-B-4 で publish() を導入する際、PublishExecutor の authority.commit() を削除しないと commit が二重化。**案A（publish() が transaction boundary）を採用**: PublishExecutor から authority.commit() を完全削除 | ✅ **§6.4-X4-B 推奨最終 API に追記** |
| 2 | **Test 9 は identity equality に限定（必須修正2）**: pointer equality（currentWorld_.load() == runtimeStore.current.load()）を要求しない。currentWorld_ は metadata observation alias / RuntimeStore::current は physical publication source のため、同一 pointer を要求すると INV-X4-8 の意味論を過剰に固定 | ✅ **Test 9 に追記** |
| 3 | **RuntimePublishAuthority は一切所有しない（必須修正3）**: Store / WriteAccess / publishAndSwap 直接呼び / 代替 authority / Store に対する write capability を一切所有しない。**production code から RuntimePublishAuthority::create() 自体を削除**（factory が Store を生成できると INV-X4-3/X4-5 を破る） | ✅ **INV-X4-3 に追記** |
| 4 | **Test 6 の write-capable 条件厳密化**: `RuntimeStore<RuntimePublishWorld, Owner>` の write-capable instance について `Owner == RuntimeWorldAuthority` を要求（RuntimeStore<RuntimePublishWorld, SomeHelper> の潜在生成を検出） | ✅ **Test 6 に追記** |
| 5 | **旧記述「getCurrent() が consumeWorldHandle() の置換先になり得る」を削除**: 一つの規範（NO-GO）だけを残す | ✅ **既に削除済み**（全「置換先」記述は NO-GO として維持） |

**その他確認（変更不要で採用）**: X4-A/X4-B 分離 / Owner template migration / Authority 内 Store ownership / WriteAccess 宣言順序 / Authority non-copy/non-move / getCurrent() と physical read API の分離 / currentWorld_ 二重管理を X4 スコープ外 / shutdown clear を publish() と分離 / B-1〜B-11 rollback point 化。

**最終評価**: 設計方針 GO。必須修正3点を反映した dash(8) は **X4-B 実装着手可能**。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant・INV-X4-1〜8 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.32 X1-X6 read path・epoch 判定・rebuild 相互作用調査（2026-08-09 十八次・別視点7）

**調査観点**: X3 の core（EpochDomain の getMinReaderEpoch / tryReclaim / detectStuckReaders）、X1 の isRebuildObsolete と通常 rebuild の相互作用、X2/X4 の read path（makeRuntimeReadHandle / observePublishedWorld）、X3 の ISRRetireRouter 経由の reader 登録を実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X3 | **EpochDomain::getMinReaderEpoch（:199-233）の詳細**: quarantined Reader は safe-epoch 計算から除外（:211-215）/ depth==0 は除外（:220-221）/ 最小 epoch を取る（:228-229）。X3 の reclaim 判定（retireEpoch < minReaderEpoch）の核心。readerRegistrationClosed で新規登録を封じれば minReaderEpoch が新規 reader で下がらない | §6.3 |
| 2 | X3 | **tryReclaim（:371-381）**: `deferredDeletionQueue.reclaim(getMinReaderEpoch())`。minReaderEpoch で安全判定。**detectStuckReaders（:426-470）**: 3パス評価（Chronic→Warning→EpochGap）で stuck Reader 検出 → quarantine → getMinReaderEpoch から除外 | §6.3 |
| 3 | X3 | **ISRRetireRouter 経由の readerRegistrationClosed 伝播**: ISRRetireRouter::registerReaderThread（:71）は EpochDomain に委譲（.cpp で include）。**EpochDomain::registrationClosed_ 設定で router / RCUReader 経由の全登録が自動的に封じられる**。audioThreadRcuReader は EpochDomain を直接 provider に持つため直接効く | §6.3 |
| 4 | X1 | **通常 rebuild と Recovery の相互作用（isObsolete の2箇所チェック）**: RebuildDispatch.cpp:976-978 の isObsolete は build 前（:980）と build 後（:1011）の2箇所。Recovery 消費（:901-973）は通常 rebuild の**前**に実行され、isObsolete チェックの対象外。Recovery は recoveryGeneration = currentRebuildRequestGeneration（:966-967）で暗黙的に最新化 | §6.1 |
| 5 | X4 | **makeRuntimeReadHandle の read path（AudioEngine.h:3099-3139）**: acquireReadToken(runtimeStore) + consumeWorldHandle(runtimeStore, readToken)（:3116-3117）+ generation/sequence 単調性監視（:3128-3135）。X4-B-9 の置換対象は read token 2箇所、単調性監視は AudioEngine 側で維持。呼び出し元多数（BlockDouble:152 / Snapshot:27 / Timer:374,1593 等） | §6.4-X4-B |
| 6 | X1 | **isRebuildObsolete（AudioEngine.h:2464）**: `generation != currentRebuildRequestGeneration` の単純不一致判定。X1 の coalesce の obsolete 判定に使用（既反映） | §6.1 |

**結果**: read path・epoch 判定・rebuild 相互作用の観点からも X1〜X6 が確定。X3 の readerRegistrationClosed は EpochDomain へのフラグ追加で ISRRetireRouter / RCUReader 経由も自動的に封じられることを確認。残る未確定事項なし。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant・INV-X4-1〜8 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.33 二十三次レビュー反映（RecoveryAdmissionClosed・X2 timeout semantics・X4 swap failure・Phase 0・必須 Acceptance Criteria）（2026-08-09）

**レビュー観点**: 最新ソース `ConvoPeq(20260809-022629).md` を基準に、dash の最新版設計（X1〜X6 / X4-A・X4-B）を `Practical Stable ISR Bridge Runtime.md` の設計原則と照合。

**総合判定**: **P2-1〜P2-4 = 実装GO / X5 = 実装GO / X1/X2/X3/X6 = invariant 固定後実装GO / X4 = 設計GO（段階実装必須）**。「そのまま一括実装してよい完成改修案」ではない。

**新規・強調ポイントの反映**:

| # | 指摘 | 反映先 |
|---|------|--------|
| 1 | **X1: `RecoveryAdmissionClosed` を shutdown state machine に追加（§4.3）**: `recoveryAdmissionPending_` だけでは不十分。`AdmissionClosed + RecoveryAdmissionClosed + BuilderStopped` を shutdown state machine に含める（build gap 中の isFullyDrained 早期 true 防止） | ✅ **§6.1 に追記** |
| 2 | **X2: timeout semantics を明文化（§6.1）**: `timeout ≠ publish failure`。timeout を failure と誤解して rollback すると double ownership / double publish が発生。lifecycle を `Allocated → Transferred → Committed → Completed` として固定 | ✅ **§6.2 に追記** |
| 3 | **X4: `swap failure is architecturally impossible / handled` を acceptance criterion に（§20）**: publishAndSwap は単一原子 exchange で CPU レベルで失敗しない。null→null swap は異常として Failed（validate で事前検出可能） | ✅ **§6.4-X4-B に追記** |
| 4 | **Phase 0（invariant/specification freeze）を最優先（§22）**: X2 の invariant を実装前に固定。実装順序を Phase 0〜Phase 7 に明確化（X4 は X2/X1/X3 の意味確定後に触る） | ✅ **§6.9 に追記** |
| 5 | **必須 Acceptance Criteria 表（§23）**: X1-5 / X1-6 / X2-6 / X3-4 ×2 / X4-3 / X4（他 Owner Store 禁止・RuntimePublishAuthority 非所有・getCurrent 非流用）/ X5 / X6（同一 counter 禁止・residentCount equality）を architectural test に必須化 | ✅ **§6.8 に追記** |
| 6 | **NO-GO 6点（§21）**: X1 単純 coalesce / X2 monotonic のみ / X3 activeReaderCount のみ / X4 rename-only / X4-B-9 getCurrent 一括置換 / X6 aggregate counter — 全て dash で既に NO-GO として固定済み（変更なし） | ✅ 既反映 |
| 7 | **ISR semantic correctness の最上位 invariant**: 「Intent → Admission → Transport Residency → Execution → Committed → Completed → Resident → Retired → Reclaimable → Deleted」の状態分離を最上位 invariant として採用（§6.7 に既反映） | ✅ 既反映 |

**最終評価**: 改修案は採用可能。**X1/X2/X3/X4/X6 の invariant をコード化・テスト化してから実装する**という条件付き GO。dash(8) は上記5点の新規反映により、この条件を満たす実装準備完了状態。

**次回の実装手順**: Phase 0（invariant/specification freeze）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.34 X1-X6 Bridge 型・物理削除・StateOwner ledger・buildSource 供給調査（2026-08-09 十八次・別視点8）

**調査観点**: X4 の core Coordinator の Bridge 型（RuntimePublicationBridge）、X3 の DSPLifetimeManager 物理削除（enqueueWithRetry / destroyRolledBackDSP）、X2 の StateOwner ledger（RuntimePublicationStateOwner）、X1 の currentBuildSnapshot_ 供給を実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X4 | **RuntimePublicationBridge（AudioEngine.h:3446-3500）の役割を確定**: validatePublicationNonRt（:3458, publish() の validate に使用）/ didPublishRuntimeNonRt（:3473）/ willRetireRuntimeNonRt（:3478, shutdown 中 skip）/ retireRuntimePublishWorldNonRt（:3489, unseal→dtor→aligned_free）。**X4-B 後も Bridge は残る**（didPublish/willRetire/retire を PublishExecutor の Execution tail から呼ぶ）。publish() は validate に Bridge を使うが、didPublish/willRetire/retire は publish() の外。Bridge は AudioEngine の内部参照を持つため AudioEngine 側に残る | §6.4-X4-B |
| 2 | X3 | **reclaim と物理削除の分離（DSPLifetimeManager.cpp:45-100）**: 物理削除（DSPCore* delete）は retire path の `enqueueWithRetry`（:49, destroyDSPCoreNode を deferred delete）が担当。**reclaim（slot 状態遷移）と物理削除は分離**。X3 の reclaim(ReclaimMode) は物理削除を含まない（slot 遷移 + free list 返却のみ） | §6.3 |
| 3 | X2 | **StateOwner ledger（RuntimePublicationState.h:100-159）との関係**: onSubmitted/onBuilt/onValidated/onPublished/onRetired/onReclaimed/onRejected/onExecutorFailed を記録。**X2 の completion watermark と独立**（診断 ledger）。onPublished は trySubmitImpl（:289）で記録、onPublishCommitted（CoordinatorLoop）とは別タイミング。X2 テストでは両者を混同しない | §6.2 |
| 4 | X1 | **currentBuildSnapshot_ 供給と初期状態**: currentBuildSnapshot_（:4619-4623, mutex 保護）は enqueuePublicationIntentForRuntimeCommit が sealedSnapshot 受領時に更新（Commit.cpp:704-705）。getCurrentBuildSnapshotForRecovery（:4265）が返す。**初期状態 sealed=false の Recovery は Builder の :917 チェックで skip**（正しい動作）。X1 の durable admission は sealed=false を入れる前に検証すべき | §6.1 |

**結果**: Bridge 型・物理削除・StateOwner ledger・buildSource 供給の観点からも X1〜X6 が確定。X4 の Bridge は X4-B 後も残る、X3 の reclaim は物理削除を含まない、X2 の ledger は completion と独立、X1 の buildSource は sealed=false 検証が必要。残る未確定事項なし。

**次回の実装手順**: Phase 0（invariant/specification freeze）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.35 二十四次レビュー反映（isFullyDrained measurement predicate・pendingIntentCount_ 命名・X4 dual-pointer 暫定正常状態）（2026-08-09）

**レビュー観点**: 情報源 `ConvoPeq(20260809-022629).md` を基準コードとして再参照し、dash(10) を ISR/Immutable Snapshot Runtime/RCU/ownership/lifetime/publication ordering/shutdown の観点から検証。

**総合判定**: **P2-1〜P2-4 = GO / X5 = GO / X1/X2/X3/X6 = 条件付きGO / X4-A = GO / X4-B = GO（段階実装）**。一括実装 = NO-GO。Phase 0 の invariant freeze を実施してから P2→X5→X6→X2→X1→X4→X3 の順に実装（条件付き承認）。

**新規反映3点（§27）**:

| # | 指摘 | 反映先 |
|---|------|--------|
| A | **`isFullyDrained()` を単独の truth source にしない**: `ShutdownPhase + ProducerQuiescence + AdmissionClosed + RecoveryAdmissionClosed + BuilderStopped + isFullyDrained()` を組み合わせる。**isFullyDrained() は measurement predicate であり shutdown authority そのものではない** | ✅ **§6.7 に追記** |
| B | **`pendingIntentCount_` の命名・コメント固定**: 実際の意味は Observe+Quarantine+Recovery の queue residency + producer reservation（Publish と RetireIntent 除外）。将来は `transportIntentResidency_` 等への改名を検討。**コードコメントで「This counter excludes Publish and RetireIntent.」を固定** | ✅ **§1.1 に追記** |
| C | **X4 dual-pointer を「暫定正常状態」として明示**: `X4-B: write authority singularization / Future: read-source singularization` と分離。dual-pointer 状態は publish transaction 完了後 INV-X4-6（同一 PublicationIdentity）を保証するため正常動作として許容 | ✅ **§6.4-X4-B に追記** |

**その他確認（変更不要・既反映）**: pendingIntentCount_ の reservation + residency 再定義 / Publish 除外（W2）/ isFullyDrained の producer quiescence 後 authoritative / Recovery durable admission + RecoveryAdmissionClosed / timeout ≠ publish failure / readerRegistrationClosed / quarantine 4層分離 / X4 physical ownership convergence / RuntimePublishAuthority に write capability を持たせない / PublicationIdentity（sequenceId+epoch+mappedGeneration）整合 — 全て既に dash に反映済み。

**評価できる11点（§28）**: pendingIntentCount_ の reservation+residency 再定義 / Publish 除外 / isFullyDrained の quiescence 後限定 / Recovery durable admission / Committed≠Completed 分離 / timeout≠failure / reader registration 閉鎖 / quarantine 分離 / X4 physical ownership convergence / RuntimePublishAuthority に write capability を持たせない / PublicationIdentity 整合。

**最終評価**: dash(10) は改修方向として妥当。**「設計方針として採用可能。Phase 0 の invariant freeze を実施してから P2→X5→X6→X2→X1→X4→X3 の順に実装する」という条件付き承認**。

**次回の実装手順**: Phase 0（invariant/specification freeze）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.36 X1-X6 receipt 状態・テスト基盤・admission 判定調査（2026-08-09 十八次・別視点9）

**調査観点**: X2 の pendingReceipt_ / markReceiptReclaimComplete の関係、X1 の既存テスト基盤（testRecoveryRequestEnqueueAndPop）、X5 の PublicationAdmission::evaluate（admission 判定と counter の関係）を実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X2 | **`pendingReceipt_` と `PublishReceiptWaiter` の区別**: AudioEngine に2種類の receipt が存在。<br>receipt #1 `pendingReceipt_`（:4683, optional<PublishReceipt>）= Timer の retire 用（handle+epoch）。storeReceipt（:1157）/ resetReceipt（:1176）/ retirePublishedDSP（Timer.cpp:1774）で管理。markReceiptReclaimComplete（ProcessIntent.cpp:41）で解放（Epoch Safe 通知）。**X2 の completion watermark と無関係**。<br>receipt #2 `PublishReceiptWaiter`（:3613-3635）= X2 の completion watermark（lastCompleted_）。onPublishCommitted → notifyPublishReceipt → complete() で管理。Producer の waitForPublishReceipt が使用。<br>**X2 の設計は receipt #2 のみを対象** | §6.2 |
| 2 | X1 | **testRecoveryRequestEnqueueAndPop の実装詳細（:609-624）**: `submitRecoveryRequest(handle, buildSource)` → `popRecoveryRequest()` の 1-hop transport のみ検証。buildSource.sealed=true を渡し、2回目の pop が null（1-hop 保証）を検証。**X1 の拡張は queue full → durable 化 → takePendingRecoveryAdmission の流れを追加**（256 満杯 → 257th durable → pop 256 消費 → durable 残存 → take → recoveryAdmissionPending_ == false → coalesce で reservation 増加なし） | §6.8 |
| 3 | X5 | **admission 判定と counter の関係**: PublicationAdmission::evaluate（:6-61）は5段階の admission（Shutdown/Generation/HealthState/Pressure/Fading→Deferred）を実行。**X5 の counter は admission Accepted 後の enqueue で増加**。Rejected/Deferred は counter に影響しない。Deferred は単一スロット（再 enqueue 時に counter +1） | §6.5 |

**結果**: receipt 状態・テスト基盤・admission 判定の観点からも X1〜X6 が確定。X2 は receipt #1（pendingReceipt_）と receipt #2（PublishReceiptWaiter）を区別、X1 は既存 1-hop テストの拡張詳細、X5 は admission Accepted 後に counter 増加。残る未確定事項なし。

**次回の実装手順**: Phase 0（invariant/specification freeze）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.37 X1-X6 ReaderSlot 構造・destroyForShutdown・intentId 採番・IR 転送調査（2026-08-09 十八次・別視点10）

**調査観点**: X3 の ReaderSlot 構造（EpochDomain.h:531-547）、X6 の DSPQuarantineManager::destroyForShutdown（ISRDSPQuarantine.cpp:130-155）、X1 の nextRecoveryIntentId_ 採番（ISRRuntimePublicationCoordinator.cpp:657）と RuntimeBuilder の IR 転送（RuntimeBuilder.cpp:447）を実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X3 | **ReaderSlot 構造（EpochDomain.h:531-547）**: epoch / depth / enterCount / residencyStartTimestampUs / ownerThreadId / ownerTag / quarantineFlags（0x01 quarantined / 0x02 pending）。**closeReaderRegistration() は新規登録のみ拒否し、既存 ReaderSlot は解放しない**（exitReader で epoch = kInactiveEpoch に戻るのみ）。graceful drain は activeReaderCount()==0（全 slot depth==0）を待つ。quarantineFlags(0x01) の Reader は getMinReaderEpoch から除外（:211-215）。X3 の isQuiescent = readerRegistrationClosed AND activeReaderCount == 0 | §6.3 |
| 2 | X6 | **destroyForShutdown（ISRDSPQuarantine.cpp:130-155）の詳細**: quarantineActiveFlags_[slot] を false（:141）→ auditLog の未解決エントリを resolved に（:146-151）→ compactAuditLogLocked（:152）。**quarantineResidentCount_（= residentCount() の走査）が自然に 0 へ**。X6 の新たな shutdown 処理は不要 | §6.6 |
| 3 | X1 | **nextRecoveryIntentId_ 採番と IR 転送の整合**: nextRecoveryIntentId_（:435, atomic）は submitRecoveryRequest（:657）で fetch_add(1, relaxed)。**durable 化後も採番は継続**（queue も durable も同じ採番源）。IR 転送（RuntimeBuilder.cpp:447 transferIRStateFrom(engine.getConvolverProcessor())）は build 時に現在の UI processor から取得。**buildSource は IR データを内包しない**（metadata/fingerprint のみ）ため durable admission は軽量。coalesce 後の buildSource も IR 実体は build 時に転送されるため stale IR の懸念なし | §6.1 |

**結果**: ReaderSlot 構造・destroyForShutdown・intentId 採番・IR 転送の観点からも X1〜X6 が確定。X3 の closeReaderRegistration は既存 slot を解放しない、X6 の destroyForShutdown は counter を自然収束させる、X1 の durable admission は軽量（IR 非内包）。残る未確定事項なし。

**次回の実装手順**: Phase 0（invariant/specification freeze）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.38 二十五次レビュー反映（INV-ISR-01〜07・X1 ShutdownDiscard・X3 意味論先行固定・最終判定）（2026-08-09）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準コードとして、REPAIR_PLAN2-dash を L1〜L4（C++/Thread Safety → Queue/Counter/State Machine → ISR/RCU/Lifetime → Architectural Authority Singularization）の4層で検証。

**総合判定**: **P2-1〜P2-4 = GO（そのまま実装してよい）/ X1/X2/X6 = 条件付きGO / X3 = GO / X4 = GO（段階実装）/ X5 = GO**。Recovery coalesce / force reclaim = NO-GO（撤回が正しい）。一括実装 = NO-GO。「P2 を先行実装し、X1〜X6 は設計ゲートとして固定してから実装」を推奨。

**新規反映**:

| # | 指摘 | 反映先 |
|---|------|--------|
| 1 | **INV-ISR-01〜07（§23・最上位 ISR 不変条件）**: INV-ISR-01（isFullyDrained 完全条件）/ 02（pendingIntentCount_ = residency+reservation）/ 03（semantic 混同禁止）/ 04（ShutdownQuiescent は readerRegistrationClosed 必須）/ 05（committed ≠ completed）/ 06（currentWorld_ は non-owning）/ 07（dual pointer identity consistency 検証可能） | ✅ **§6 冒頭に追記** |
| 2 | **X1 shutdown discard を「ShutdownDiscard」として明示（§8.1）**: `Recovery lost` と `ShutdownDiscard` を同じ意味にしない。Running 中の queue full → durable pending = loss ではない（INV-5 保証が機能）。Shutdown 中の durable pending → explicit lifecycle discard。**Telemetry 上で2つを分ける**（recoveryIntentDropCount_ とは別に shutdown discard を記録） | ✅ **§6.1 に追記** |
| 3 | **X3 の意味論を X4 より先に固定（§22）**: X3 は memory lifetime の最終安全境界、X4 は publication authority topology。**Lifetime correctness（X3）を先に固定してから Publication authority topology（X4）を変更**。X3 の実装（Phase 6）は X4 の後でもよいが、**意味論（INV-X3-4 / INV-ISR-04）は Phase 0 で X4 より先に固定** | ✅ **§6.9 に追記** |
| 4 | **X2 の「CAS 化すれば安全」は誤り**: completion semantic が重要（§9.1）。既に INV-X2-6 で固定済み（変更不要） | ✅ 既反映 |
| 5 | **評価できる12点（§25）**: pendingIntentCount_ の queue size 誤認修正 / Publish/Observe/Recovery/Quarantine 分離 / producer reservation 明示 / producer hole テスト契約固定 / Recovery coalesce 撤回 / force reclaim 撤回 / ShutdownQuiescent に reader re-entry 禁止 / completion order と publication order 分離 / Authority の物理 ownership 検討 / dual publication/read surface 発見 / quarantine 4種分離 / 既存 source of truth への counter 追加撤回 | ✅ 全て既反映 |

**残余リスク（§14）**: X4-B 完了後も `currentWorld_` / `RuntimeStore::current` の dual publication/read surface は残る。**X4-B 完了 = publication write authority の Singularization（read-source singularization は Future）**。

**最終 ISR 到達点**: `currentWorld_` と `RuntimeStore::current` を最終的に「どちらが canonical publication state か」一つに収束させる必要があるが、今回の X4-B に無理に含めない判断は正しい（大規模 semantic 変更を一改修に混ぜない）。

**最終評価**: **「P2-1〜P2-4 は実装GO。X1〜X6 は設計として概ね正しいが、段階実装が必要。特に X3 の lifetime closure と X4 の dual publication surface を最終的な ISR Architecture Review で再確認すること」**。

**次回の実装手順**: Phase 0（invariant/specification freeze・INV-ISR-01〜07 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.39 X1-X6 cv 動作・retire 実装・BuildError 種類調査（2026-08-09 十八次・別視点11）

**調査観点**: X2 の PublishReceiptWaiter の cv 動作詳細（AudioEngine.h:3613-3635）、X3 の ISRRetireRouter::retire 実装（ISRRetireRouter.cpp:149-179）、X1 の RuntimeBuilder::build の BuildError 種類（RuntimeBuilder.cpp:55-71）、X4 の read API 現状（★ 2026-08-11: X4 read API 実装済み）（RuntimeWorldAuthority に read API 未実装）を実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X2 | **cv 動作の詳細（PublishReceiptWaiter :3613-3635）**: complete（:3614-3621）は mutex 下で lastCompleted_ 更新 → cv_.notify_all()（:3620）。waitFor（:3623-3630）は `cv_.wait_until(lock, deadline, [&]{ return seqId <= lastCompleted_; })`（:3628）。**wait_until は predicate 付きのため、notify 前に waitFor が始まっても即復帰**（lost wakeup 安全）。deadline 到達後 predicate が false なら false（:3629）。X2 の案1はこの構造を維持 | §6.2 |
| 2 | X3 | **ISRRetireRouter::retire の実装詳細（:149-158）**: `enqueueWithRetry(ptr, deleter, provider_->currentEpoch(), Generic)` に委譲。enqueueWithRetry（:161-179）は通常 enqueue（:167）→ 失敗時 tryReclaim → enqueue を最大2回リトライ（:172-178）。QueuePressure 以外（Shutdown 等）は即時終了（:178）。**reclaim（slot 遷移）と retire（物理削除予約）は独立**。X3 は reclaim 側のみ変更し、retire の enqueueWithRetry リトライ機構は不変 | §6.3 |
| 3 | X1 | **BuildError 種類と再試行方針（RuntimeBuilder.cpp:55-71）**: InvalidInput / ResourceUnavailable / MKLFailure / ConvolverFailure / PrepareFailure / WarmupFailed / InternalError。**一時的 failure（ResourceUnavailable/MKLFailure/PrepareFailure）は次サイクルで自然に再試行**（新たな quarantine → submitRecoveryIntent）。永続的 failure（InvalidInput/InternalError）は drop 相当（INV-X1-2/INV-X1-3）。**X1 の新たな retry ループは不要**（retry は新たな Recovery 要求で自然発生） | §6.1 |
| 4 | X4 | **read API の現状**: `RuntimeWorldAuthority.h` には read API（observePublishedWorld / acquireReadToken / consumeWorldHandle）は**未実装**。現状の read path は全て `RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore)` を直接呼ぶ（AudioEngine.h:1331/2119/3116/3383/3691 等）。**X4-B-9 で新設予定の専用 read API に置換（★ 2026-08-11 実装済み）**（§6.4-X4-A で design 済み） | §6.4-X4-B |

**結果**: cv 動作・retire 実装・BuildError 種類の観点からも X1〜X6 が確定。X2 の wait_until は lost wakeup 安全、X3 の retire は enqueueWithRetry リトライ機構を維持、X1 の build 失敗は一時的/永続で扱いが分かれる、X4 の read API は X4-B-9 で新設。残る未確定事項なし。

**次回の実装手順**: Phase 0（invariant/specification freeze・INV-ISR-01〜07 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.40 X1-X6 Execution tail・compactAuditLog・DeferredDeletionQueue reclaim 調査（2026-08-09 十八次・別視点12）

**調査観点**: X2 の DSPTransition（publish-completion の execution tail）、X6 の compactAuditLogLocked（ISRDSPQuarantine.cpp:158-172）、X3 の DeferredDeletionQueue::reclaim（DeferredDeletionQueue.h:108-119）を実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X4 | **Execution tail の3構成要素を確定**: <br>tail-1 `ctx.transition.onPublishCompleted(...)`（DSPTransition.h:49-90）= DSP activate/crossfade/retire（publish 成功後の DSP lifetime 操作）。Crossfade Registration Authority（registerCrossfade は DSPTransition のみ — CI gate）<br>tail-2 `ctx.engine.advanceRetireEpoch()` = retire epoch 前進（EBR）<br>tail-3 `ctx.engine.runtimeOrchestrator_->onPublishCommitted(seqId)` = X2 の completion。<br>**tail-1 と tail-3 は独立**（onPublishCompleted は DSP activate、onPublishCommitted は completion watermark）。X4-B の publish() は tail を含まない | §6.4-X4-B |
| 2 | X6 | **compactAuditLogLocked（ISRDSPQuarantine.cpp:158-172）の実装詳細**: `kCompactThreshold = 1024`（:161）、resolved エントリが1024超えた場合のみ compaction（:162）。先頭の resolved 連続を削除（:167-171）。**auditLog_ は vector**（append-only + resolved マーク + compaction）。X6 の新規介入は不要 | §6.6 |
| 3 | X3 | **DeferredDeletionQueue::reclaim（DeferredDeletionQueue.h:108-119）の epoch 安全削除**: `reclaim(minReaderEpoch)` は `isOlder(entry.epoch, minReaderEpoch)` のエントリのみ deleter 実行。**FIFO 前提**（先頭が不安全なら break、:119）。`isOlder(a,b)`（:399-402）= `static_cast<int64_t>(a-b) < 0`（wraparound 対応）。X3 の reclaim(ReclaimMode::RuntimeEBR) は slot 状態遷移、物理削除は DeferredDeletionQueue::reclaim が epoch 安全に実行。ShutdownQuiescent では minReaderEpoch が最新に進み全 Retire が安全判定 | §6.3 |

**結果**: Execution tail・compactAuditLog・DeferredDeletionQueue reclaim の観点からも X1〜X6 が確定。X4 の tail は DSPTransition/advanceRetireEpoch/onPublishCommitted の3構成、X6 の compactAuditLog は1024閾値、X3 の reclaim は FIFO epoch 安全削除。残る未確定事項なし。

**次回の実装手順**: Phase 0（invariant/specification freeze・INV-ISR-01〜07 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.41 X1-X6 通常 rebuild 後半・quarantineReader・crossfade decision 調査（2026-08-09 十八次・別視点13）

**調査観点**: X1 の通常 rebuild 後半処理（RebuildDispatch.cpp:1024-1138）、X3 の quarantineReader / unquarantineAllReaders / verifyReaderInvariants（EpochDomain.h:264-367）、X2 の trySubmitImpl の crossfade decision と deferred の関係（RuntimePublicationOrchestrator.cpp:186-218）を実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X1 | **通常 rebuild の後半処理**: IR rebuild（:1025-1041, rebuildAllIRsSynchronous :1039）→ Warmup（:1051-1070, validateWarmup + retryable 判定 :1054）→ refreshLatency + fadeIn（:1085,1088）→ 投影値更新（:1104-1115）→ Commit（:1138, enqueuePublicationIntentForRuntimeCommit）。**通常 rebuild と Recovery は同一の commit 関数（:1138 vs :971）**。X1 の Recovery 消費はこの後半の前に実行 | §6.1 |
| 2 | X3 | **quarantineReader / unquarantineAllReaders / verifyReaderInvariants**: quarantineReader（:264-311, depth==0 即座 quarantine / depth>0 pending→exitReader で昇格）。unquarantineAllReaders（:313-321, 全 slot flags 0 に）。verifyReaderInvariants（:338-367, quarantined は epoch==inactive / pending は depth>0 / quarantined と pending 同時不可）。**X3 の readerRegistrationClosed は quarantine と独立**（quarantine は stuck 解除、registrationClosed は新規登録封鎖）| §6.3 |
| 3 | X2 | **trySubmitImpl の crossfade decision と deferred の関係**: cfDecision = CrossfadeAuthority::evaluate（:193-206）→ needsCrossfade なら spec.transitionActive=true（:221-227）→ executor_.publish（:263）。**crossfade decision は Deferred の発生源**（hasFadingRuntimeInWorld → DeferredFadingActive → deferredSlot_ 退避）。**deferred 中は新 publish の completion は発生しない**（re-enqueue 後）。INV-X2-6 の deferred 例外と整合 | §6.2 |

**結果**: 通常 rebuild 後半・quarantineReader・crossfade decision の観点からも X1〜X6 が確定。X1 は通常 rebuild と Recovery が同一 commit 経路、X3 は quarantine と registrationClosed が直交、X2 は crossfade decision が Deferred の発生源。残る未確定事項なし。

**次回の実装手順**: Phase 0（invariant/specification freeze・INV-ISR-01〜07 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.42 二十六次レビュー反映（X1 lease 方式・X3 INV-X3-5・X4 INV-X4-A/B/C）（2026-08-10）

**レビュー観点**: `REPAIR_PLAN2-dash(20260810-002710)` を最新版として、`ConvoPeq.md` の最新ソースコードと照合して再評価。前回版よりかなり完成度が上がり、**X4-B は条件を満たせば実装GO できる設計に到達**。ただし実装前に修正すべき重要論点2つ。

**総合判定**: **P2-1〜P2-4 = GO / X5 = GO / X6 = 条件付きGO / X2 = 条件付きGO / X3 = 条件付きGO / X4-B = GO（strict acceptance criteria）/ X1 = 修正後GO**。実装方針（P2→X5→X6→X2→X1→X4→X3）は妥当だが、X1 を現在の文章どおり実装してはいけない。

**必須修正2点・強く推奨1点の反映**:

| # | 指摘 | 反映先 |
|---|------|--------|
| 1 | **X1 の Pending/Building 矛盾（必須修正1・§9-13）**: `takePendingRecoveryAdmission()` が state クリア（destructive dequeue）なのに build 失敗で「Pending 維持」という矛盾。**take を lease（state transition）に変更**: PendingRecoveryAdmission に `State` enum（NoAdmission/DurablePending/Building）を追加。take は DurablePending → Building へ遷移（クリアしない）。build 失敗（transient）は Building → DurablePending へ戻す（retry 構造的保証）。obsolete は Discarded。build success は PublishTransport。**INV-X1-1（exactly one durable state）が lease 方式で常に成立** | ✅ **§6.1 PendingRecoveryAdmission 定義 + take 実装 + build 失敗時 state transition に反映** |
| 2 | **X3 の reclaimInFlightCount_ 近似 counter（必須修正2・§16-17）**: `reclaimInFlightCount_ == 0` だけで shutdown drain を判定しない。**INV-X3-5 を追加**: ShutdownQuiescent completion requires `pendingReclaimHandles_.empty() AND reclaimInFlight == 0`。`pendingReclaimHandles_`（:4616, mutex 保護）が reclaim pending の実際の source of truth。`reclaimInFlightCount_` は診断値に降格 or 併用 | ✅ **§6.3 INV-X3-5 + §6.7 isFullyDrained に pendingReclaimHandles.empty() 追加** |
| 3 | **X4 の INV-X4-A/B/C（強く推奨・§29）**: `currentWorld_ = observation-only` / `RuntimeStore::current = sole physical RuntimeWorld source` / `No RT API may derive RuntimeWorld ownership/lifetime from currentWorld_`。**Audio Thread は currentWorld_ を RuntimeWorld 取得元として使わない** | ✅ **§6.4-X4 INV-X4-8 に INV-X4-A/B/C 追記** |

**その他確認（変更不要・既反映）**: P2 accounting / queue direct observation / W2（Publish 除外）/ X5 residency / X6 4層分離 / X2 FIFO invariant（INV-X2-5/6 + CAS 追加しない）/ reader registration closure / RuntimePublishAuthority 非所有 / currentWorld_ 意味論分離 / publish() transaction boundary / shutdown clear 分離。

**最終判定**: **「設計の骨格は妥当で、P2 と X5 は実装開始可能。X4-B も今回の修正で実装GO まで到達。ただし X1 の Pending/Building 状態矛盾と X3 の pending reclaim accounting は、ISR shutdown correctness に直接関係するため、実装着手前に必ず修正する」**。

**次回の実装手順**: Phase 0（invariant/specification freeze・INV-ISR-01〜07 / INV-X3-5 / INV-X4-A〜C 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1（lease 方式）→ X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.43 X1-X6 epoch 取得元・queue capacity・enterReader 詳細・kMaxSlots 調査（2026-08-10 十八次・別視点14）

**調査観点**: X1 の submitRecoveryRequest の epoch 取得（currentWorld_ 由来）、X5/X6 の全 transport queue capacity、X3 の enterReader/exitReader 詳細、X6 の kMaxSlots を実コードで検証。

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X1 | **submitRecoveryRequest の epoch 取得元（:650-652）**: `consumeAtomic(currentWorld_)` → `world->publication.epoch`。**PendingRecoveryAdmission.epoch は submitRecoveryRequest 時に currentWorld_ から取得**。X4 の INV-X4-A（currentWorld_ observation-only）との整合: submitRecoveryRequest は NonRT（CoordinatorLoop）から呼ばれるため、currentWorld_ を metadata observation（epoch 取得）として使用するのは正当（INV-X4-C に違反しない）。coalesce 時の epoch は latest buildSource と同様に currentWorld_ から再取得 | §6.1 |
| 2 | X6 | **全 transport queue capacity（実コード検証）**: intentQueue_ = 4096（:445-446, MPSC）/ quarantineFallbackQueue_ = 1024（:453-454）/ recoveryIntentQueue_ = 256（:433-434, SPSC）/ observeDeferredRing_ = 1024（:429-430）。各 X の対象 queue と capacity が確定 | §6.6 |
| 3 | X3 | **enterReader / exitReader の詳細**: enterReader（:106-130）は epoch を depth++ より先に store（:115-116, BUG-050）→ depth++（:119-121）。ネスト時（previousDepth > 0）は epoch 再設定なし（:122-123）。exitReader（:133-168）は depth-- → 0 で epoch = kInactiveEpoch（:157）→ pending quarantine（0x02）昇格（:163-168, CAS）。**enter/exit は registrationClosed の影響を受けない**（既存 Reader の enter/exit は shutdown 中も継続可能） | §6.3 |
| 4 | X6 | **kMaxSlots = 256**（ISRDSPQuarantine.h:36）: residentCount() の走査は 256 要素の for ループ（NonRT で許容）。QuarantineReason enum（:12-21, ReceiptReset 含む）は X2 の receipt #1 と関係。X6 の設計で kMaxSlots / constructor の変更は不要 | §6.6 |

**結果**: epoch 取得元・queue capacity・enterReader 詳細・kMaxSlots の観点からも X1〜X6 が確定。X1 の epoch は currentWorld_ から NonRT で取得（INV-X4-A/C と整合）、X6 の queue capacity は 4096/1024/256/1024、X3 の enter/exit は registrationClosed と独立、X6 の kMaxSlots は 256。残る未確定事項なし。

**次回の実装手順**: Phase 0（invariant/specification freeze・INV-ISR-01〜07 / INV-X3-5 / INV-X4-A〜C 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1（lease 方式）→ X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak

### A-2.44 X1-X6 evaluateDeferred・QueuePressure 移送・read API 置換対象調査（2026-08-10 十八次・別視点15）

**調査観点**: X2 の evaluateDeferred の stale-discard 判定（PublicationAdmission.cpp:69-91）、X3 の enqueueWithRetry の QueuePressure 移送（ISRRetireRouter.cpp:182-203）、X4 の read API 置換対象一覧（RuntimeWorldAuthority に read API 未実装）を実コードで検証。**（★ 2026-08-11: X4 read API 置換実装済み）**

**確定事項（§6 に反映済み）**:

| # | 対象 | 確定事項 | §6 反映 |
|---|------|---------|---------|
| 1 | X2 | **evaluateDeferred の stale-discard 判定（:69-91）**: 1. Shutdown（:74-75）→ Discard（ShutdownDiscard）/ 2. TTL 超過（:78-80, 30s）→ Discard（StaleDiscard）/ 3. Generation 不一致（:83-84）→ Discard（StaleDiscard）/ 4. Sequence 後戻り（:87-88）→ Discard（StaleDiscard）→ それ以外 Ready（:90）。**deferred の cancel 条件が確定**（ShutdownDiscard / StaleDiscard の3種類）。StaleDiscard された deferred の seqId は re-enqueue されない → **completion は発生しない**（INV-X2-6 の deferred 例外と整合） | §6.2 |
| 2 | X3 | **enqueueWithRetry の QueuePressure 移送（:182-203）**: QueuePressure/QueueFull 時は `m_retireQuarantine.quarantine(ptr, deleter, epoch, type, "enqueueWithRetry:QueuePressure")`（:190-192）で RetireQuarantineStore へ移送。**queue full は RT 参照中の可能性が高いため即時解放は UAF**。store full 時は delete を絶対しない（:195-199, assert + health escalation 監視）。Future: runtimeHealth_->notifyQueuePressure（:202）。X3 の reclaim はこれに影響しない | §6.3 |
| 3 | X4 | **read API 置換対象一覧**: RuntimeWorldAuthority に read API は未実装（X4-B-9 で新設）。現状は `RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore)` / acquireReadToken / consumePublishedWorld を直接呼ぶ（AudioEngine.h:1331/2119/3116/3383/3691 / Commit.cpp:559 / Latency.cpp:91 / observePublishedWorld :3550 経由）。**X4-B-9 はこれらを worldAuthority().readAPI() に一括置換**（getCurrent() は置換先にしない）。単調性監視は AudioEngine 側で維持 | §6.4-X4-B（★ 2026-08-11 実装済み） |

**結果**: evaluateDeferred・QueuePressure 移送・read API 置換対象の観点からも X1〜X6 が確定。X2 の deferred cancel 条件（ShutdownDiscard/StaleDiscard）、X3 の QueuePressure 時の RetireQuarantineStore 移送、X4 の read API 置換対象一覧を確定。残る未確定事項なし。

**次回の実装手順**: Phase 0（invariant/specification freeze・INV-ISR-01〜07 / INV-X3-5 / INV-X4-A〜C 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1（lease 方式）→ X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak
