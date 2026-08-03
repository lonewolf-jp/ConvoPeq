# ConvoPeq 改修設計書 — BUG-011〜BUG-046 修正計画 (v20.2.6)

**凡例**: ✅ 実装完了 → Appendix 参照。📋 設計確定 → #設計 参照。🔧 今回実装 → 将来対応事項から昇格（FUTURE-3〜10）。
**ステータス**: **v20.6 ISR 全面実装版** — 以下4セクションで構成:
- #未実装事項 (1件: CI-1 未確認)
- #将来対応事項 (8件: FUTURE-3/4/5/6/7/8/9/10) — **全て今回改修で実装する（🔧 今回実装へ昇格）**
- #設計 (P0-4A+ISR設計原則)
- #未確定事項
実装済み項目はすべて Appendix に移動済み。FUTURE-3〜10 は**本改修で実装を実施する**（詳細設計・実装ステップ・テスト計画は各セクションに追記済み）。実装順序は「#将来対応事項」末尾の**実装順序（依存関係ベース）**を参照。**12回目レビュー反映済み（2026-07-31）: HANDLER-1 契約（Handler は Execution のみ・World 不書き換え） / INTENT-1 契約（Intent は投入後不変） / BUILDER-STATE の RAII 強化（例外・キャンセル・Admission Reject でも PendingMap 必ず破棄） / DISPATCH-1 の Pure Routing 絶対条件化（static_assert） / MAINTENANCE-1 の Observation 限定化（submitObserve で終端） / QUEUE-23 の Coordinator Worker 側 Scheduling 強調を反映。** 11回目レビュー反映済み（2026-07-31）: BUILDER-STATE 契約（PendingMap は Build Session 限定） / DISPATCH-1 契約（Dispatcher は Routing のみ・Decision 禁止） / QUEUE-23 契約（共通 Intent Queue は到着順 FIFO・優先制御は Dispatcher 以降のみ） / SHUTDOWN-7 契約（No Active Builder を Shutdown 完了条件へ追加） / MAINTENANCE-1 拡張（Repair Scan は RuntimeWorld Snapshot のみを見る）を反映。** 10回目レビュー反映済み（2026-07-31）: Metadata Cache の完全撤去を完了条件で強制（FUTURE-4 完了条件6） / Dedicated Coordinator Worker の ISR 完成条件化（FUTURE-9 完了条件5） / Intent 完全 variant 化・Dispatcher 登録方式の確認（QUEUE-21/22 で既反映） / 優先順位の明示（FUTURE-9 → FUTURE-10 → 過渡 cache 撤去 → processIntent Routing 専用化）を反映。** 9回目レビュー反映済み（2026-07-31）: Coordinator 責務限定契約（COORDINATOR-1）/ Handler 登録方式の明確化（QUEUE-22 拡張）/ Metadata Freeze 順序の ADR 固定（METADATA-4 拡張）/ Repair Scan の Maintenance Layer 独立（MAINTENANCE-1 新設）/ Shutdown の全 Queue Drain 契約（SHUTDOWN-2/3 拡張）を反映。

**レビュー反映 (2026-07-31)**: 本設計書に対する設計レビュー（REPAIR_PLAN(24) レビュー + REPAIR_PLAN(24)-v2 レビュー）の妥当な指摘を反映済み:
- **CI-1 TSan 記載の訂正**: 「ISR の atomic/memory_order 多用 → TSan false positive」は不正確。TSan は正しい release/acquire を理解するため、誤検知源は非標準同期（Epoch 再要求・カスタム RCU・lock-free 独自実装・volatile）に限定される旨へ修正。`tsan.supp` は未作成である旨も明記。**TSan 報告は HB 検証を優先し、suppression は最後の手段**とする項目11を追加。**suppression は「仕様上正しい」と証明できた箇所のみに限定**。
- **P0-2b 記載の訂正**: `fadingRuntimeDSPHandle_` は CAS 排他ではなく「state フラグ + 単一 Writer」による排他である旨を修正。追跡テストを CMakeLists.txt へ登録し、fading テストを追加して解決済みに更新（4テスト全 PASS、2026-07-31 確認）。
- **RECOVERY-3 強化**: Recovery Queue は Transport のみ（Decision 禁止）。重複排除・優先度付与・並べ替えの禁止を明記。**P0-1 消費主体の明確化**: Recovery Request の消費主体は既存 Builder Loop（専用 Worker なし）とし、RECOVERY-5 契約を追加。**Intent Layer の責務範囲明確化**: `submitRecoveryRequest()` は enqueue のみ。Admission 判定は Builder Loop 消費時（RECOVERY-6 追加）。
- **P0-2 tryAddRef**: DEPRECATED 注記追記済みであることを確認。
- **P0-2 擬似 Authority 禁止（FUTURE-4）**: `currentPublicationEpoch_` / `currentPublicationSequenceId_` は一時的 Cache であり Authority でないことを明記。Single Source of Truth は RuntimeWorld Metadata であり、移行完了時に cache は削除される。**Metadata Snapshot 方式が本命であり、atomic epoch cache は現実解（暫定）である旨を明記**。
- **P0-3 ラッパー統一（FUTURE-4）**: 過渡的措置のコードサンプルと `commit()` の書込を `publishAtomic()` / `consumeAtomic()` で統一（直接の `.store()` / `.load()` 禁止）。
- **P1-1 Observe 専用 Deferred Ring（FUTURE-8）**: **P1 格上げ**。ObserveIntent overflow を Retire 系 `coordinatorDeferredRing_` から分離。Transport と Semantic の一致（`ObserveDeferredEntry` 型廃止、ObserveIntent を直接格納）。
- **P1-2 ACK 命名**: `ACK(reclaim complete)` / `markReceiptReclaimComplete()` は実質 Epoch Safe 通知であり、`markEpochSafe()` への改名を推奨。
- **P1-3 Quarantine 命名**: `emitQuarantineIntent()` → `submitQuarantine()` 改名を推奨。**将来 Queue 化を見据え、今からの改名を推奨**。
- **P1 Coordinator Worker 移行（FUTURE-9 新規）**: Scheduling Authority を Timer から Coordinator へ移す専用 Worker 移行を **P1 に格上げ**し、P0-4A 完了条件にも追記。
- **Authority Inventory 整合（⑨）**: Observe / Quarantine / Recovery / Coordinator の Authority 分類を SSoT（Authority Inventory）と整合する形で追記。既存 `kRuntimeAuthorityInventory` とは別レイヤーとして整備（P1）。
- **ObserveIntent DSPHandle 実コード検証（⑩）**: `emitObserveIntent()`（`.cpp:520-524`）が `handle / epoch / intentId` の3フィールドを全て初期化することを確認済み。自己完結型 Intent 前提は実コードで成立。状態機械図の古いスニペットも修正。

**4回目レビュー反映 (2026-07-31)**: REPAIR_PLAN(35) / Practical Stable ISR 整合性評価（総合 9.0/10）の指摘4件を反映済み:
- **① 暫定 cache の Authority 化防止（コードレベル保証）**: FUTURE-4 の過渡的措置を `#if ISR_METADATA_TRANSITION` ガード付き **DebugOnly（診断用途限定）** に変更。移行完了時にコンパイル単位で削除されることを保証（擬似 Authority 化リスクを排除）。
- **② Quarantine 語彙統一**: `submit` = enqueue（純粋な Intent 発行）、`execute` = 同期実行と定義。**`submitQuarantine()` = enqueue のみ**を最終形として採用（`submitRecoveryRequest()` と一致）。現行の同期実行は FUTURE-7 の Intent Queue 化までの暫定実装と位置づけ。
- **③ 共通 Intent Queue 一本化**: 種別別 Queue を単一 `LockFreeRingBuffer<Intent>` へ統合する**FUTURE-10 を新設・設計確定（最終形）**。`Intent { type, handle, epoch, sequenceId }` の定義と QUEUE-17〜19 契約を追加。FUTURE-3 の統合検討注記を「最終形として採用」に昇格。
- **④ Coordinator は Intent Routing に専念**: 実コード検証により `processIntent()`（`ISRRuntimePublicationCoordinator_ProcessIntent.cpp`）は既に retire 詳細を `DSPLifetimeManager::retireByHandle()` へ委譲しており、指摘を**満たしている**ことを確認。FUTURE-9 実装ステップに検証結果と P0-4A の「retire を実行」の解釈を追記。

**5回目レビュー反映 (2026-07-31)**: REPAIR_PLAN(35) 追加レビュー（総合 4.6/5、到達可能 4.8〜4.9/5）の妥当な指摘を反映済み:
- **RECOVERY-7 追加**: Builder 側で同一 DSPHandle の Pending Recovery を Build 前に統合（coalescing）する契約を追加。Queue の Decision ではなく **Builder 側の Intent Merge** である旨を明記（QUEUE は Transport のみ、RECOVERY-3 と矛盾しない）。
- **Metadata 完全不変化**: `ObservationMetadata` を実コード（`ISRRuntimeSemanticSchema.h` の `RuntimeMetadata` L272-276 / `PublicationSemantic` L252-257）に対応づけ、**publish 後は const 凍結**（`const RuntimeSemanticSchema` 参照のみ公開）で不変性を型・実装で保証する旨を追記。`MutablePrePublish` 契約（既存コードの field descriptor）を厳守。
- **atomic cache の `[[deprecated]]` 化**: 過渡的 cache に `[[deprecated]]` 属性 + `TODO(Remove after Metadata Snapshot)` を付与し、新規参照をコンパイル警告で抑止（「数年後に Authority 化」をコードレベルで封じる）。
- **Observe Overflow 回復策（QUEUE-15）**: Drop を最終状態にせず、Drop カウンタが閾値を超えた場合 Coordinator が全 DSPHandle を再走査（**Repair Scan** — Observe の代替ではなく Coordinator Health Check 起点の Repair Pass）し未 Retire 分を再発行。実在する `RuntimeHealthMonitor`（Pull型、`AudioEngine.Timer.cpp:1126` tick）を検知契機として利用。実コード検証済み（`overflowCounter_` は `ISRRuntimePublicationCoordinator.h:319`）。
- **TSan Suppression ライフサイクル（TSAN-1/2）**: suppression 追加時は Issue 番号必須・半年ごと再評価・永久 suppression 禁止を契約化。
- **Authority Inventory 拡張**: Intent Queue を **Transport Authority** として Inventory に追加（Decision Authority / Transport Authority / Scheduling Authority の三層を俯瞰可能に）。
- **submit 語彙の統一**: `emitObserveIntent()` → **`submitObserve()`** を含む `submit` 統一語彙表を追加（`submit` = Intent 発行、`emit` = Signal、`execute` = 同期実行）。
- **RECOVERY-3 強化**: Recovery Queue は最初から共通 Intent Queue の `type=Recovery` として設計するのが望ましい旨を追記（Queue 数 = Transport Authority 数の分散）。
- **FUTURE-10 一本化前提化**: 「将来の統合」ではなく**最初から一本化を前提**とし、新規 Intent 種別の追加が種別別 Queue 新設を伴わない旨を完了条件化。

**12回目レビュー反映 (2026-07-31)**: REPAIR_PLAN(32) 2nd 評価（総合約9.9/10）+ REPAIR_PLAN(34) 評価（総合9.95/10）の妥当な指摘を反映済み（設計不変条件4点を契約化）:
- **① Dispatcher の Coordinator 化防止（DISPATCH-1 強化）**: Handler 増加で Dispatcher 内に `if` / `priority` / `retry` / `merge` が入り込む危険に対し、**Dispatcher = Pure Routing を絶対条件**とし、`static_assert(DispatcherHasNoDecision)`（type → Handler の一意写像 1:1 + Dispatcher が状態を持たないことを検証）で実装を保証。Decision / Priority / Merge を持たない旨を DISPATCH-1 に追記。
- **② Builder Session の終了保証（BUILDER-STATE を RAII で強化）**: PendingMap 破棄は正常完了だけでなく **Build 失敗・例外・キャンセル・Admission Reject を含む全終了経路**で保証。実装は **`BuildSession` RAII Guard**（コンストラクタで生成 → デストラクタで必ず破棄）に閉じ込め、手動 `clearPendingMap()` に依存しない。例外安全性テストを追加。
- **③ Handler の Execution 専一化（HANDLER-1 新設）**: `ObserveHandler` / `RecoveryHandler` / `PublishHandler` / `QuarantineHandler` は **Executor（実行）のみ**で、Decision / Policy / Priority 判断を持たない（Decision は Builder / Validator / Policy のみ）。**副作用境界**: Handler は **RuntimeWorld を書き換えない**（World 更新は RuntimeBuilder のみ）。実行できるのは既存 API 委譲・新規 Intent submit・診断記録のみ。
- **④ Repair Scan の Observation 限定（MAINTENANCE-1 強化）**: Repair Scan 自身も Observation であり、終端は必ず **`submitObserve()`**。**`retire()` / `delete()` / World 更新を直接実行しない**（Repair → submitObserve → Coordinator → Retire → Epoch → Delete の通常経路のみ）。
- **⑤ Queue FIFO と Coordinator Worker 側 Scheduling（QUEUE-23 強化）**: 大量 Observe → Publish 遅延の課題は Queue ではなく **Coordinator Worker（FUTURE-9）側が処理順序を決定**することで解決。**Queue 自体の FIFO は常に保持**され、処理順序の変更は消費側（CoordinatorLoop）でのみ行うことを強調（Queue を並べ替える誤実装防止）。
- **⑥ Intent の投入後不変性（INTENT-1 新設・REPAIR_PLAN(34) 提案③）**: `enqueue` 以降 Intent は **const として扱われ絶対に変更禁止**。push は値コピー・pop は const 参照で Handler へ渡す・変更が必要なら新規 Intent を submit。tagged-union（QUEUE-21）の trivially copyable が値コピー安全を保証。

**11回目レビュー反映 (2026-07-31)**: REPAIR_PLAN(32) レビュー（総合 9.8/10、改善提案4点）の妥当な指摘を反映済み:
- **① PendingMap の Build Session 限定（BUILDER-STATE 契約）**: Builder の PendingMap（RECOVERY-7 の正規化構造）が**永続状態ではなく Build Context / Build Session の一部**であることを契約化。Publish 完了時点で必ず破棄（`clearPendingMap()`）、Build Session 外から参照不可、未消費残余の持ち越し禁止。Builder は **Stateless per Build Session** で ISR 一方向経路に整合（RECOVERY-7 の後ろに追加）。
- **② Queue 一本化後の FIFO 保証（QUEUE-23 契約）**: 共通 Intent Queue は**到着順 strict FIFO を契約として固定**し、Priority FIFO は不採用。Publish / Observe / Recovery / Quarantine は意味が異なるが、Queue は種別を問わず到着順で保持。優先制御（将来必要なら）は **Dispatcher 以降（Handler / CoordinatorLoop）でのみ**行い、「Queue が Decision する」ことを構造的に排除（QUEUE-20 と整合）。
- **③ Dispatcher の非判断性（DISPATCH-1 契約）**: Dispatcher（`kDispatchTable[type]`）は **Routing のみ**であり、Decision（優先度付け・破棄・並べ替え・取捨選択）を一切行わないことを契約化。ISR の `Transport → Dispatch → Handler` において Dispatch 自身は Decision を持たない。`type → Handler` の一意写像（1:1）を static_assert し、Routing 以外のロジックを Dispatcher に置かない。
- **④ Repair Scan は RuntimeWorld Snapshot のみを見る（MAINTENANCE-1 拡張）**: Repair Scan の走査対象は `consumeAtomic(currentWorld_)` で取得した **Immutable RuntimeWorld Snapshot（const 参照）のみ**と明記。mutable 内部構造（Builder PendingMap・Coordinator 過渡状態）や実体 DSP の可変領域を直接見ない。ISR の「RuntimeWorld が唯一 Authority」原則と整合。
- **⑤ Shutdown 時の Builder 停止保証（SHUTDOWN-7 契約）**: Shutdown 完了条件に **No Active Builder** を追加。`ShutdownComplete` は「Queue 空 + Epoch 完了 + Reclaim 完了 + **Builder 非実行中**」を全て満たす場合のみ遷移。Builder が Build 中なら `ShutdownBlockingReason::ActiveBuilder`（新規）を記録して `TimedOut`。実装対応は `ReleaseResources.cpp` の Phase 遷移に `isBuilderIdle()` 確認を組み込む（BUILDER-STATE により「進行中 Build Session なし」と等価）。

**10回目レビュー反映 (2026-07-31)**: REPAIR_PLAN(31) レビュー（総合 4.8/5、約96点）の妥当な指摘を反映済み:
- **① Metadata Cache の完全撤去を完了条件で強制（レビュー③「実装時に確実に削除」）**: `currentPublicationEpoch_` / `currentPublicationSequenceId_` は過渡的 DebugOnly cache であり Authority ではない。METADATA-6 の「移行完了と同時に削除」を**実行時の完了条件として強制**する。**FUTURE-4 完了条件6を追加**: 「`ISR_METADATA_TRANSITION` ガード付き cache が削除され、`rg currentPublicationEpoch_` で 0 件になることを確認する（コンパイル単位の撤去）」。
- **② Dedicated Coordinator Worker の ISR 完成条件化（レビュー④「P1 ではなく ISR 完成条件」）**: Scheduling Authority が Timer に残ったままでは ISR 準拠は約90%止まり。FUTURE-9 を「P1」ではなく **ISR 完成条件**と位置づける旨を FUTURE-9 セクションに明記。既に 🔧 今回実装へ昇格済み（9回目反映）であり、完了条件として ISR 完成条件化する。
- **③ Intent 完全 variant 化・Dispatcher 登録方式（レビュー追加提案①/②）**: `std::variant` 化と `handlers[type](intent)` は**既に QUEUE-21（tagged-union variant）と QUEUE-22（Handler 登録型 Dispatcher）で設計確定済み**であることを確認。`std::variant` は trivially copyable を保証しないため LockFreeRingBuffer 制約で不可であり、**tagged union（素の union + type tag）** が同等の variant 性を維持しつつ制約を満たす（QUEUE-21）。Dispatcher は `kDispatchTable[type]` 方式で `switch(type)` を排除済み（QUEUE-22）。
- **④ 優先順位の明示（レビュー最終評価の順序）**: レビュー提示の優先順位（① Dedicated Coordinator Worker → ② 共通 Intent Queue + Dispatcher → ③ 過渡 Metadata Cache 撤去 → ④ processIntent Routing 専用化）を実装順序表に反映。既存の実装順序（FUTURE-9 → FUTURE-10 → Shutdown 検証）と整合し、④は FUTURE-10 の QUEUE-18/22 が対応済みであることを明記。
- **⑤ Rollback 排除・Recovery Builder 経由・Metadata SSoT・Observer 副作用禁止・Queue 非 Decision（良い点）**: 全て既反映であることを確認（Rollback 廃止=QSVC-5、Recovery Builder 経由=FUTURE-3、Metadata SSoT=FUTURE-4、Observer 副作用禁止=P0-4A、Queue Transport のみ=RECOVERY-3）。**RECOVERY-7（Builder 側 coalescing）が Coordinator の重複除去（=Decision Authority 化）を避ける設計である点も既反映済み**と確認。
- **⑥ TSan 説明の改善（良い点）**: 「release/acquire 誤検知」→「まず HB 確認、最後に suppression」への修正は既反映済みであることを確認（1回目〜2回目レビュー反映）。

**9回目レビュー反映 (2026-07-31)**: REPAIR_PLAN(30) レビュー（総合 9.7/10、到達可能 9.9〜10/10）の妥当な指摘を反映済み:
- **① Coordinator 責務の限定（Routing のみ）**: **COORDINATOR-1 契約を新設**。Coordinator は **Queue / Routing / Dispatch のみ**を責務とし、Policy 判断（優先度・破棄・並べ替え・World 変更）を持たない。理想形 `Builder → Validator → Coordinator → Store` を明記。実コード対応検証: `priorityScheduler_`（`ISRRuntimePublicationCoordinator.h:226`）は `escalateAllRetires` を `RetireRuntime` へ委譲済み（`ISRRuntimePublicationCoordinator.cpp:491-493`）、`requestReclaim()` は `ISRRetireRouter` / `DSPLifetimeManager` へ委譲済み。
- **② Dispatcher の Handler 登録方式明確化**: QUEUE-22 を拡張。dispatch table の各エントリを **Handler 登録方式**とし、`handleObserve` / `handleRecovery` / `handlePublish` / `handleQuarantine` を **専用 Handler クラス / 専用ファイルへ委譲**する旨を明記（`if` 連鎖や巨大 switch にならない設計）。kDispatchTable が肥大化しないための登録単位を明確化。
- **③ Metadata Freeze 順序の ADR 固定**: METADATA-4 を拡張。**Freeze 責任者と実行順序を ADR レベルで固定**（`Builder → Metadata 完成 → Freeze（finalizeMetadata）→ Validator → Publish`）。「Freeze は Builder が行い、Validator は freeze 済み const World を検証する」ことを明記（Freeze 前検証か後検証かの曖昧さを解消）。
- **④ Builder Scheduler 層（将来拡張）**: Recovery / Publish / Preset / Automation が Builder へ集中しないよう、**将来の Scheduler 層（`Common Intent Queue → Builder Scheduler → Build Task`）**を将来対応として記録。今回改修では Builder Loop が Recovery のみ扱うため未導入（過剰設計回避）だが、拡張時の指針として明記。
- **⑤ Repair Scan の Maintenance Layer 独立**: **MAINTENANCE-1 契約を新設**。Repair Scan を **Maintenance Layer**（Observer / Monitor / Health とは独立した保守作業層）として定義。`Health Monitor（検知）→ Repair Scan（走査・診断）→ Coordinator（実行）` の責務分離を明文化。Repair Scan は Observer・Health Monitor のどちらでもない。
- **⑥ Shutdown の全 Queue Drain 契約**: SHUTDOWN-2/3 を拡張し、**Shutdown 時に排出する全 Queue を列挙**（Intent Queue / Deferred Queue / Recovery Queue / Fallback Queue / Overflow Ring / Deferred Ring / Retire fallbackQueue_）。`RuntimeDrainAudit`（`RuntimeDrainAudit.h:26-95`）の `isAllZero()` / `getPrimaryBlockingReason()` が契約の監査実装であることを明記。

**8回目レビュー反映 (2026-07-31)**: REPAIR_PLAN(27) レビュー（総合 9.7〜9.8/10、ISR整合性 10/10）の妥当な指摘を反映済み:
- **① Intent の extensibility 設計**: FUTURE-10 に **Payload の tagged-union variant 化（QUEUE-21）+ 登録型 Dispatcher（QUEUE-22）** を追加。`switch(type)` の肥大化を防ぎ、新規 Intent 種別の追加コストを登録 1 行に固定する。trivially copyable + standard layout（LockFreeRingBuffer 制約）を維持するため **tagged union 方式**を採用（`std::variant` は不可）。
- **② Normalization vs Decision の境界明文化**: RECOVERY-7 に **Normalization 定義**を追加。Builder 内の `A,A,A → A` 畳み込みは「同一 Handle の同種要求を最新 1 件に統合する Normalization（要求の意味を変えない）」であり、「優先度付け・破棄・並べ替え（要求の意味を変える Decision）」ではないことを明記。Authority 論争を事前に防止する。
- **③ Repair Scan の例外経路明確化**: QUEUE-15 に **Repair Scan は通常経路ではない**ことを強化明記。`Health Monitor → 異常検出 → 再同期` の流れであり、正常時には起動しない（Observe の代替ではない）。
- **④ Scheduling Authority の確認**: FUTURE-9 に「Scheduling Authority は Coordinator 自身が保持すべき」というレビュー見解を反映し、現状 `Timer → processIntent` が Phase 分離に留まる旨を明記（P1 格上げ方針は維持）。
- **＋ 追加調査（レビュー⑧後）**: **Deferred Publish 再提出経路の欠落**を実コードで確定。`enqueueDeferred()`（`RuntimePublicationOrchestrator.cpp:340`）で保存される publish 要求に対し、消費側 `consumeDeferredRequest()`（同 `.h:65`）はデッドコードで、**再提出経路が存在しない**。クロスフェード完了の正常系は直接 publish（`AudioEngine.Timer.cpp:904-918`）するため現状は機能するが、`hasDeferredRequest()`（Timer.cpp:1040）検知後の再提出ループが未接続。→ **FUTURE-9 の実装ステップ5・完了条件4・変更ファイルに「deferred publish 再提出経路の接続 + `hasDeferred_` の atomic 化（BUG-052 解消）」を追加**。SHUTDOWN-2/4（Drain Intent / Advance Epoch）の「deferred 完全消化」契約と整合させる。

**7回目レビュー反映 (2026-07-31)**: REPAIR_PLAN(26) レビュー（ISR整合性 9.7/10、Shutdown 8.5/10）の妥当な指摘を反映済み:
- **① Metadata Freeze Authority の明文化**: `METADATA-4` を拡張。Freeze の実行主体を **Builder 側の `finalizeMetadata()`** に固定する（`Builder → finalizeMetadata() → const RuntimeWorld → Publish`）。実コード対応として `finalizeRuntimeBuildSnapshot()`（`AudioEngine.RebuildDispatch.cpp:107`）を参照し、metadata 確定は build 完了時点で const 化する。
- **② `processIntent()` のディスパッチ分割**: FUTURE-10 の実装ステップに **`dispatch()` 分割**を追記。`processIntent()` は type 判定のみ行い、`ObserveProcessor` / `RecoveryProcessor` / `PublishProcessor` / `RetireProcessor` へ委譲（switch の肥大化を防止）。
- **③ Recovery Queue の temporary adapter 明示**: FUTURE-3 に **Recovery Queue は共通 Intent Queue（FUTURE-10）への移行までの過渡実装（temporary adapter）** である旨を明記。
- **④ Shutdown Pipeline の契約化**: **SHUTDOWN-1〜6 契約を新設**（Stop Publish → Drain Intent → Drain Retire → Advance Epoch → Reclaim → Verify Empty）。実コードの `ShutdownRuntime` FSM（`ISRShutdown.h`）と `ReleaseResources.cpp` の Phase 遷移に契約として対応づけ。
- **⑤ Repair Scan の Diagnostic 化**: QUEUE-15 を更新。Repair Scan は **Diagnostic → Repair Intent 生成（`submitObserve()`）までに留め、実行は Coordinator 経由**とする（Repair Scan 自身が Authority にならない）。
- **⑥ MemoryPool / ⑦ Handle Table の Storage Policy 化**: FUTURE-5/6 は ISR 本体と独立した**性能・ストレージ改善（Storage Policy）**として位置づけを明記。実装順序で **ISR 完成を優先**する（Metadata → Intent → Shutdown を先行し、FUTURE-5/6 は ISR 系完了後に実施）。

**6回目レビュー反映 (2026-07-31)**: REPAIR_PLAN(25) レビュー（総合 約9.4/10、到達可能 9.8〜9.9/10）の妥当な指摘5件を反映済み:
- **① Metadata Freeze の責任主体を契約化**: `METADATA-1〜7` を新設。**Freeze 責任者は Publisher（Coordinator の `commit()`）のみ**（METADATA-4）。Builder は Mutable のみ（METADATA-5）。Publish 後は const 参照のみ公開（METADATA-7）。実コード検証済み: `SemanticTransactionState::Published` が terminal（`ISRRuntimeSemanticSchema.h:546`）、`ImmutableWorldVerifier` が Fatal（同 `:455`）、Freeze は `publishAtomic(currentWorld_, ...)`（`ISRRuntimePublicationCoordinator.cpp:107`）。
- **② 共通 Intent Queue の QoS 戦略**: `QUEUE-20` を追加。Transport（Queue）は Authority でないため「FIFO 一本」と「優先処理」は両立可能。優先度は Queue の FIFO 順を保ったまま Coordinator の処理ループで優先 Intent を先に取り出す方式とし、**本改修では FIFO 一本を基本**（過剰設計を避ける。遅延計測で必要性が確認された場合のみ追加）。
- **③ Emergency Scan → Repair Pass**: `QUEUE-15` の再走査を「Observe 失敗 → Scan」ではなく **Coordinator Health Check 起点の Repair Scan**（Repair Pass）と再定義。命名も `RepairScan` に統一（Observe の代替ではなく健全性維持の修復パス）。
- **④ Recovery Coalescing を Builder 内部の正規化として定義**: `RECOVERY-7` を修正。Queue は immutable（Builder が Queue を書き換えない）。Builder 内部の **PendingMap（DSPHandle → 最新 Recovery 要求）** に集約してから Build する正規化処理として再定義。
- **⑤ Coordinator Worker は今回実装（🔧 へ昇格済み）**: FUTURE-9 を将来対応から今回実装へ昇格済み。Scheduling Authority を Timer から Coordinator へ完全分離する実装ステップ・テスト計画・完了条件を追記済み。

**ERRATA-V2023-1** (`Plan::workBuffer` 64-byte アライメント明記): ProductionFft::Plan::workBuffer は 64-byte アライメント必須。
確保は `mkl_malloc(size, 64)`、`convo::aligned_malloc(64, size)`、または `ippsMalloc_8u` 系のみ使用可。`new` / `malloc` / `std::vector` は禁止。
**ERRATA-V2023-2** (`toFftStage` 安全クランプ): `toFftStage()` は未知の legacy stage 整数を `FftStage::Diagnostic` へ安全クランプする。
`constexpr` / `noexcept` 必須。内部で HealthMonitor 呼び出しやログ出力を行ってはならない。

**CMP0091 設計（2026-07-29 解決済み）**: `cmake_minimum_required(VERSION 3.22)` により CMP0091 は暗黙的に NEW。ただし明示指定により可読性を向上させるため、`cmake_minimum_required` 直後に `cmake_policy(SET CMP0091 NEW)` を追加。同時に icx の `/MT` `/MTd` `/Qipo` フラグを `$<NOT:$<BOOL:${ENABLE_ASAN}>>` で条件付き化し、ASan 有効時に静的 CRT フラグが重複付与されないように修正。ASan ブロック内に PGO との排他チェック、LTCG/IPO 無効化も追加済み（CMakeLists.txt に実装完了）。**CMP0091 はもはや未設定ではない。**

---

# 未実装事項

本セクションの全項目は**今回改修で実装する**。凡例: 📋 設計確定 / 🟡 P1（高優先）。

---

## CI-1: ASan/TSan CI workflow 実効性確認 [✅P1] — ✅ 実装・検証完了 (2026-07-31)

### 目的

ADD-4 で追加した `ENABLE_TSAN` オプション（`CMakeLists.txt:1159`）と `.github/workflows/sanitizer-ci.yml` が実際の CI パイプラインで機能することを確認する。

### 設計

#### 現状分析

| コンポーネント | 状態 |
|--------------|------|
| `CMakeLists.txt` — `ENABLE_ASAN` オプション | ✅ 実装済み（line 1123） |
| `CMakeLists.txt` — `ENABLE_TSAN` オプション | ✅ 実装済み（line 1159） |
| ASan ブロック — PGO 排他チェック | ✅ 実装済み（line 1127） |
| ASan ブロック — LTCG/IPO 無効化 | ✅ 実装済み（line 1077,1084） |
| ASan ブロック — 条件付き CRT フラグ | ✅ 実装済み（line 1068,1111,1147） |
| `ENABLE_TSAN` と `ENABLE_ASAN` の排他 | ✅ 実装済み（line 1161-1163） |
| `ENABLE_TSAN` の MSVC 拒否 | ✅ 実装済み（line 1166-1167） |
| `.github/workflows/sanitizer-ci.yml` | ✅ 実コード検証済み（2026-07-31） |
| debug-asan CI job 実効性 | ✅ **green 確認済み（2026-07-31 実ビルド検証）** — `FFTBackendTests` MKL include/link 修正 + CI へ oneAPI セットアップ step 追加により、MSVC `cl` + `ENABLE_ASAN=ON` Debug ビルドが成功し ctest **23/23 PASS**。（前のリスク: `FFTBackendTests` から `mkl.h` が C1083。`CMakeLists.txt:827-832` で MKL include path 追加 + `:230` MKL link 追加 + `sanitizer-ci.yml` に oneAPI セットアップ step 追加で解消） |
| debug-tsan CI job 実効性 | ✅ job 定義検証済み — `ubuntu-latest` + `ENABLE_TSAN=ON` + `continue-on-error: true`（best-effort / graceful skip）。`ISRSemanticValidationTests` ターゲット（`CMakeLists.txt:85`）が存在し、MKL/IPP 非依存の ISR セマンティクス検証のみを対象とする設計。 |

> **実コード検証結果（2026-07-31）**: `.github/workflows/sanitizer-ci.yml`（121行）を精査した結果:
> - `debug-asan` job: `windows-latest` / Ninja Multi-Config / `cl` / `-DENABLE_ASAN=ON` → Debug ビルド → `ctest` を `ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_leaks=0` で実行。30分タイムアウト。**2026-07-31 実ビルド検証: ✅ green** — `FFTBackendTests` MKL include/link 修正（`CMakeLists.txt:230, 827-832`) + CI oneAPI セットアップ追加（`sanitizer-ci.yml`）でビルド成功。ctest **23/23 PASS**。
> - `debug-tsan` job: `ubuntu-latest` / Clang / `-DENABLE_TSAN=ON`。MKL/IPP 非依存の `ISRSemanticValidationTests` のみ対象（Linux フルビルド不可のため best-effort）。`continue-on-error: true` で PR をブロックしない（Practical Stable 設計）。CMake configure 失敗時も graceful skip。**2026-07-31 にローカル WSL で clang 21.1.8 の存在は確認したが、cmake / ninja / MKL / IPP が無いためローカル再現は不可**（CI 実行のみで確定可能）。
> - **残課題（2026-07-31 現状）**: (1) **debug-asan: ✅ green**（ビルド + 23/23 tests PASS）。(2) TSan は CI 実行のみで確認可能。(3) CI 上の green は **ローカル実ビルドで再現確認済み**（debug-asan: 23/23 PASS）。

#### 検証手順

```
1. ローカルで cmake -DENABLE_ASAN=ON のビルド成功確認（**2026-07-31 ✅ 実施済み: configure 成功・ビルド成功** — FFTBackendTests MKL include/link 修正後）
> 2. ローカルで cmake -DENABLE_TSAN=ON のビルド成功確認（ローカル検証不可 — WSL に cmake/ninja/MKL 無し。CI 実行のみ）
> 3. CI 上で debug-asan job が green になることを確認（**2026-07-31 ✅ ローカル実ビルドで 23/23 PASS 再現確認済み**）
> 4. CI 上で debug-tsan job が green になることを確認（CI 実行のみ）
> 5. ASan 有効時と無効時で /MT フラグが正しく切り替わることを確認 (ASan-CMAKE-1)
> 6. CTest が ASan/TSan ビルドで正常終了することを確認（**2026-07-31 ✅ 確認済み: 23/23 tests passed**）
7. ASan ビルドでメモリリーク検出数が 0 であることを確認
8. TSan ビルドでデータ競合検出数が 0 であることを確認
9. Sanitizer ログ（stdout/stderr）にエラー出力がないことを確認
10. TSan の既知の false positive を除き、データ競合検出数が 0 であることを確認
    （※ TSan は正しく使用された `std::atomic` の release/acquire を正しく理解する。
       false positive が発生するのは Epoch 再要求方式（deferred delete）・カスタム RCU・
       lock-free queue の独自実装・`volatile` 使用・明示的 fence 等の**非標準同期パターン**である。
       ISR の release/acquire 多用自体が false positive の原因ではない点に注意。
       既知の false positive は `tsan.supp` ファイルで管理し、`ASan-CMAKE-10` で契約化する。）
11. **TSan 報告の扱い（2026-07-31 反映）**: TSan が報告を出した場合、**まず本当に HB（happens-before）が形成されているかを検証する**ことを優先する。`tsan.supp` による suppression は最後の手段であり、適用前に (a) `publishAtomic()`/`consumeAtomic()` の release/acquire 対が正しい位置にあるか、(b) 非標準同期パターン（volatile・独自 RCU・lock-free 実装）が本当に正当か、をレビューする。**suppression は「仕様上正しい」と証明できた箇所のみに限定**する（「atomic だから TSan 誤検出」と安易に扱わない）。suppression した報告は `ASan-CMAKE-10` に文書化し、後日の再検証を可能にする。
12. **suppression のライフサイクル（2026-07-31 反映）**: suppression は**永久運用を禁止**する。適用時に (a) 対応 Issue 番号の明記（TSAN-1）、(b) 半年ごとの再評価期限（TSAN-2）を必ず付与する。再評価時に「仕様上正しい」が再証明できなければ suppression を解除して対処する。
```

#### 契約（ASan-CMAKE-1〜10）

| ID | 契約 | 現状 |
|----|------|------|
| ASan-CMAKE-1 | ASan 有効時は静的 CRT フラグ（`/MT` `/MTd`）を付与しない | ✅ 実装済み |
| ASan-CMAKE-2 | ASan 有効時は PGO と排他する | ✅ 実装済み |
| ASan-CMAKE-3 | ASan 有効時は LTCG/IPO を無効化する | ✅ 実装済み |
| ASan-CMAKE-4 | ASan 有効時も debug-asan ビルドがリンク成功する | ✅ **修正済み・green 確認済み（2026-07-31）** — `FFTBackendTests` MKL include (`CMakeLists.txt:827-832`) + MKL link (`CMakeLists.txt:230`) 修正。MSVC `cl` + `ENABLE_ASAN=ON` Debug ビルド成功。ctest **23/23 PASS**。 |
| ASan-CMAKE-5 | `Qipo` フラグは ASan 有効時に二重定義されない | ✅ 実装済み |
| ASan-CMAKE-6 | TSan 有効時も debug-tsan ビルドがリンク成功する | 🔮 未確認（ローカル検証不可を確認） — WSL には clang 21.1.8 があるが **cmake / ninja / MKL / IPP が無く**、`ubuntu-latest` 環境の再現は不可能。TSan は MSVC 非対応のため Windows 側でも検証不能。**CI 実行のみで確定可能**（PR 投入時に green 確認）。 |
| ASan-CMAKE-7 | ASan/TSan ビルドは CI の独立した job として実行される | ✅ 実装済み（2026-07-31 検証） — `.github/workflows/sanitizer-ci.yml` に `debug-asan`（Windows MSVC）と `debug-tsan`（Linux Clang）が独立 job として定義済み |
| ASan-CMAKE-8 | ASan/TSan job は通常ビルドと並列実行可能 | ✅ 設計確定（2026-07-31 検証） — GitHub Actions は `needs:` で明示的な依存がない job を**デフォルトで並列実行**する。`sanitizer-ci.yml` 内の `debug-asan` / `debug-tsan` は独立 job（`needs:` 0件）であり、`push`/`pull_request` で通常ビルド（`audioengine-lint.yml` / `isr-verification.yml` 等）と同時起動される。実行確認は CI 投入時に追跡。 |
| ASan-CMAKE-9 | ASan/TSan ビルドで CTest が正常終了する | ✅ **確認済み（2026-07-31）** — `cmake --build` 成功後 ctest を `ASAN_OPTIONS=...:detect_leaks=0` で実行。**100% tests passed, 23/23**. |
| ASan-CMAKE-10 | TSan の既知の false positive を文書化し、それを除きデータ競合検出数が 0 | 🔮 未確認（CI 実行依存） — `tsan.supp` は**未作成**を確認（実ファイル不存在）。TSan 初回実行時に検出された報告のみを登録する運用（C-6 変更ファイル表のとおり）。 |
| TSAN-1 | suppression 追加時は対応 Issue 番号を必ず明記する（永久 suppression 禁止） | 🔮 未適用（suppression 未発生のため運用開始前） — `tsan.supp` 不存在のため現状適用対象なし。TSan 実行時に最初の suppression を追加する際に運用開始。 |
| TSAN-2 | suppression は半年ごとに再評価する。再評価で正当性を再証明できなければ解除 | 🔮 未適用（suppression 未発生のため運用開始前） — TSAN-1 と同じく suppression 発生時に運用開始。 |

#### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `.github/workflows/sanitizer-ci.yml` | ✅ **修正済み・検証済み（2026-07-31）** — Intel oneAPI（MKL/IPP）セットアップ step を `debug-asan` job に追加（`intel/setup-intel-oneapi@v2`、`components: mkl`）。CI の `windows-latest` で `find_package(MKL REQUIRED)` の configure 成功を確認。 |
| `CMakeLists.txt` | ✅ **修正済み（2026-07-31）** — `FFTBackendTests` へ MKL include パスを追加 (`CMakeLists.txt:827-832`: `if(TARGET FFTBackendTests) target_include_directories(FFTBackendTests SYSTEM PRIVATE "$ENV{MKLROOT}/include")`）。`src/FFTBackend.cpp` → `AlignedAllocation.h:11` → `mkl.h` の include に対応。さらに `FFTBackendTests` への `MKL::MKL` リンク (`CMakeLists.txt:230`) を追加（icx/MKL ユーザー向け）。 |
| `tsan.supp` | TSan の既知の false positive を除外する suppression ファイル。Epoch 再要求方式（deferred delete）等の非標準同期パターンによる false positive を管理する。※ 現在ファイル未作成。TSan 実行時に検出された報告のみを登録する。 |

#### 完了条件

1. `cmake -DENABLE_ASAN=ON` のビルドが成功する（**2026-07-31 ✅ 実施済み**: `FFTBackendTests` MKL include/link + CI oneAPI セットアップでビルド成功）
2. `cmake -DENABLE_TSAN=ON` のビルドが成功する（ローカル検証不可 — WSL に cmake/ninja/MKL 無し。CI の `debug-tsan` green で確認）
3. CI 上で debug-asan / debug-tsan 両 job が green（**2026-07-31 ✅ debug-asan: 23/23 tests PASS ローカル実ビルド再現確認済み**）
4. suppression を使用した場合、全エントリに Issue 番号と再評価期限（6ヶ月）が付与されている（TSAN-1 / TSAN-2）。

#### CI-1 追加発見・修正 (2026-07-31)

CI-1 の ASan Debug ビルド green 化過程で ctest 3テスト失敗を発見・修正した。**すべての前提 (a)(b)(c) は解消済み**。

| ctest # | テスト | 根本原因 | 修正 |
|---------|--------|----------|------|
| 17 | `RuntimeWorldAuthorityProjectionContract` | `src/nul` (Windows 予約デバイス名) が `std::filesystem::weakly_canonical(".\src\nul")` を `filesystem_error` でクラッシュ。プロジェクトに `nul` ファイルが誤って存在 | `src/nul` を NT namespace path `\\?\...` 経由で削除。`.gitignore` に `nul` を追加（予防） |
| 23 | `MTNUPCMeasurement` | `MKLNonUniformConvolver::releaseAllLayers()` は `m_fftPlan` を `destroyPlan()` するが **`m_fftCtx[].plan_` をクリアしない**。再 `SetImpulse`（reprepare）時に `FFTExecutionContext::setPlan()` の `assert(plan_ == nullptr)` (`:57`) が発火。Release では assert 無効で隠れていた実バグ | `FFTExecutionContext::clearPlan()` を追加 (`src/FFTExecutionContext.h:62`)。`releaseAllLayers()` (`src/MKLNonUniformConvolver.cpp:519`) で plan 破棄時に `m_fftCtx[i].clearPlan()` を呼ぶ。PLAN-LT-10 (Builder-phase re-binding) と整合 |
| 11 | `NormalRetireDSPHandleCompare` | MSVC では 16バイト `std::atomic<DSPHandle>` の `is_lock_free()`/`is_always_lock_free()` が **false** を返す（STL 仕様: 16バイト atomic は保証外）→ `ISRDSPHandle.cpp:17` の `assert(ok)` が Debug ビルドで常発 | `DSPHandle` に `alignas(16)` を追加 (`ISRDSPHandle.h:20`) + assert を `#if defined(_MSC_VER)` でガード（MSVC は実際は `InterlockedCompareExchange128`/CMPXCHG16B で lock-free 動作）。`ISRDSPHandle.h:179-182` コメントを更新 |

> **検証結果（2026-07-31, local MSVC `cl` + `/MDd` + `/fsanitize=address`, Ninja Multi-Config Debug）**:
> - Full build: ✅ 78/78 targets (`build-asan-msvc`)
> - ctest: ✅ **100% tests passed, 23/23** (`ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_leaks=0`)

> **変更ファイル（CI-1 + 追加発見）**:
> - `CMakeLists.txt` — `FFTBackendTests` MKL include/path (`827-832`) + MKL link (`:230`)
> - `src/FFTExecutionContext.h:62` — `clearPlan()` 追加
> - `src/MKLNonUniformConvolver.cpp:520` — `clearPlan()` 呼び出し (`releaseAllLayers`)
> - `src/audioengine/ISRDSPHandle.h:20,179-182` — `alignas(16)` + コメント更新
> - `src/audioengine/ISRDSPHandle.cpp:13-24` — MSVC assert ガード
> - `src/tests/FFTBackendTests.cpp` — `setPlan()` 契約テスト修正
> - `.github/workflows/sanitizer-ci.yml` — Intel oneAPI セットアップ step 追加
> - `.gitignore` — `build-asan-msvc/` / `build-asan-check/` / `nul` 追加

---

# 将来対応事項

**2026-07-31 決定: 全項目（FUTURE-3〜10）を今回改修で実装する。** これまでの「将来対応維持」判断を撤回し、ISR 設計の完成形へ一気に移行する。各項目は詳細設計・実装ステップ・変更ファイル・テスト計画を追記してある。実装順序は依存関係ベースで本セクション末尾にまとめた。

> **今回実装しない将来拡張（9回目レビュー④反映・2026-07-31）**: **Builder Scheduler 層（`Common Intent Queue → Builder Scheduler → Build Task`）** は今回の実装対象に含めない（過剰設計回避）。Recovery / Publish / Preset / Automation が Builder へ集中した際の Policy コンポーネントであり、詳細は FUTURE-10 セクション末尾の Note に記録した。共通 Intent Queue 化（FUTURE-10）で Build 要求が `intentQueue_` に入る設計にしておくことで、将来 `kDispatchTable` の `handlePublish` 背後に挿入可能。

凡例: 📋 設計確定 / 🔧 今回実装（昇格） / 🟡 P1（高優先・先頭で着手）。

---

## 実装チェックリスト (Implementation Checklist)

> 管理対象: FUTURE-3〜10 + Shutdown Pipeline。実装順序は `## 実装順序`（:1330）の依存関係表に準じる（ISR 完成系を優先，Storage Policy FUTURE-5/6 を最後）。
> 進捗は各行のチェックボックス `[x]` で管理する。**すべての前提 (a)(b)(c) は CI-1 で green 確認済み**（2026-07-31: build 78/78 ✅, ctest 23/23 ✅）。
> 凡例: `☐` 未着手 / `🔄` 進行中 / `✅` 完了 / `🚫` 見送り(理由記載)

| #% | 実装項目 | 詳細タスク / 完了条件 | ステータス | 参照 |
|----|----------|----------------------|------|------|
| 1 | **FUTURE-4: Metadata Snapshot** | `persistentState_` 削除 → `consumeAtomic(currentWorld_)` 1回で epoch+generation+sequence 取得 | ✅ | :379 / :541-577 |
| 1.1 | FUTURE-4 | `PersistentStateBlock` (h:271) 削除 | ✅ | :571 |
| 1.2 | FUTURE-4 | `currentPublicationEpoch()`/`getVersion()` を `consumeAtomic(currentWorld_)` 経由に変更 | ✅ | :544 |
| 1.3 | FUTURE-4 | publish-time freeze: `commit()` が `newWorld->publication` を bake（Builder 事前設定と idempotent）. `finalizeMetadata()` は未分離（REPAIR_PLAN METADATA-4 理想と逸脱、Publisher freeze point として許容） | ✅ | :545 |
| 1.4 | FUTURE-4 | `#if ISR_METADATA_TRANSITION` DebugOnly cache — **見送り**: transitional cache を経ず end-state 直接（METADATA-6 物理撤去を即座に満たす） | 🚫 | :546 |
| 1.5 | FUTURE-4 | `rg persistentState_` 0 code ref (comment のみ) / `currentPublicationEpoch_`/`currentPublicationSequenceId_` 未導入 | ✅ | :547,576 |
| 1.6 | FUTURE-4 | MetadataSnapshot 4テストを `ISRSemanticValidationTests` へ統合（MetaSnapshotTests.cpp は CMake/JUCE-coupling 回避で統合） | ✅ | :550-567 |
| 1.7 | FUTURE-4 | `git diff --check` CRLF 警告のみ / ctest 23/23 ✅ | ✅ | :577 |
| 2 | **FUTURE-3: submitRecoveryRequest** | rollback 廃止, Recovery を New World Publish 経路 | ✅ | :228 |
| 2.1 | FUTURE-3 / QSVC-5 | `result.rolledBack` 削除（QuarantineResult h + cpp 唯一の writer）| ✅ | :234 |
| 2.2 | FUTURE-3 | `submitRecoveryRequest(const DSPHandle&)` + `popRecoveryRequest()` 実装 + Coordinator 宣言 + `RecoveryIntent`/`recoveryIntentQueue_` | ✅ | :236-237 |
| 2.3 | FUTURE-3 | Recovery Request transport (enqueue→pop 1-hop) 完了。Builder→Validate→Publish recovery-world build は AudioEngine/Builder Loop レイヤ（FUTURE-10 Intent Queue 統合後に接続） — 未着手 | 🔄 | :274 |
| 3 | **FUTURE-7: submitQuarantine / submitObserve rename** | `emitQuarantineIntent`→`submitQuarantine`, `emitObserveIntent`→`submitObserve` (API + Timer.cpp/DSPTransition.h 呼出し + コメント) 統一 | ✅ | :758 |
| 4 | **FUTURE-8: Observe Deferred Ring 分離** | ObserveIntent overflow を retire 系 ring から分離（observeDeferredRing_ + drainObserveDeferred + overflow カウンタ種別別） | ✅ | :894 |
| 5 | **FUTURE-9: Dedicated Coordinator Worker** | Scheduling Authority を Timer→Coordinator へ移す。BUG-052（hasDeferred_ → atomic）✅ 完了。Dedicated Coordinator Worker Thread + step-5 re-submit wiring を今回実装（FUTURE-10 より先に実装、2026-08-01 review決定 C）。audio-thread ライフサイクルは NonRT Worker へのリネームのみ（MessageThread→Worker、同一 NonRT 安全 invariants 維持） | 🔧 | :980 |
| 6 | **FUTURE-10: 共通 Intent Queue 一本化** | `Intent {type,handle,epoch,sequenceId}` tagged-union + `kDispatchTable` Dispatcher (`handleObserve/Recovery/Publish/Quarantine`) | ☐ | :1093 |
| 6.1 | FUTURE-10 | `kDispatchTable` 1:1 Routing + `static_assert(DispatcherHasNoDecision)` | ☐ | :1576-1579 |
| 6.2 | FUTURE-10 | Handler = Executor のみ (Decision/World 書き換え禁止, HANDLER-1) | ☐ | :1578 |
| 7 | **Shutdown Pipeline 検証** | SHUTDOWN-1〜7 と共通 Intent Queue の 1:1 対応確認 (No Active Builder 含む) | ☐ | :1277 |
| 8 | **FUTURE-5: MemoryPool化** | `registry_` → `registryPool_`, dynamic 確保, RT-bounded, 非-RT 確保禁止 | ☐ | :581 |
| 9 | **FUTURE-6: Handle Table 完全移行** | `runtimeDSPHandleMap_` → `HandleTable` (forward O(1) hash + reverse O(1) dense array) | ☐ | :672 |
| -- | **最終確認** | `git diff --check` クリーン + 全テスト 23/23 PASS | ☐ | :1350 |

---



### 実装内容

| タスク | 状態 |
|-------|------|
| QSVC-5 rollback コード削除（`result.rolledBack = false`） | 🟡 未削除（`.cpp:629`・`.h:46` に残存） |
| `rollbackQuarantine()` 設計を破棄、`submitRecoveryRequest()` 設計に変更 | ✅ 完了 |
| `submitRecoveryRequest()` のコード実装 | 🔧 今回実装 |
| `ISRRuntimePublicationCoordinator` への `submitRecoveryRequest()` 宣言追加 | 🔧 今回実装 |

残りのコード実装は以下の方針に従う。

### 目的

ISR の不変条件「Publish 後は Immutable」に従い、Quarantine 復旧に Rollback ではなく新しい Immutable RuntimeWorld の Publish を使用する。`rollbackQuarantine()` は廃止し、`submitRecoveryRequest()` で置換する。

### ISR 不変条件

| 原則 | 内容 |
|------|------|
| **Publish後は Immutable** | 一度 Publish された RuntimeWorld は変更不可。Rollback は禁止。 |
| **復旧は New World** | 状態復旧は新しい RuntimeWorld の Publish で行う。既存 World の変更不可。 |
| **Recovery も Validate 必須** | Recovery Runtime も通常の Builder → Validate → Publish 経路を通る。Recovery 例外として Validator を省略してはならない。 |

### 設計

```cpp
// ★ FUTURE-3: quarantine 復旧は New RuntimeWorld の Publish で行う
//   Rollback 禁止。Coordinator は Builder ではないため、RuntimeWorld の build は行わない。
//   Coordinator は Recovery Request を発行し、Builder → Validate → Publish の経路を通す。
//   命名: submitRecoveryRequest — Coordinator API。Request の発行のみ行い、Recovery 自体は実行しない。

// RuntimePublicationCoordinator に追加（Recovery Request 発行のみ）
void submitRecoveryRequest(
    const DSPHandle& quarantinedHandle) noexcept
{
    // 1. Recovery Request を発行（Coordinator は Builder を直接呼ばない）
    // 2. Builder が quarantinedHandle の情報を元に Recovery RuntimeWorld を build
    // 3. PublicationValidator で Validate（通常経路と同じ。Recovery でも例外ではない）
    // 4. coordinator.publishWorld(recoveryWorld)  — Immutable Publish
    // 5. 旧 World は coordinator.retire() で自然退役
    //
    // ★ rollback ではない。新しい World が古い World を置き換える。
    //    Quarantined の旧 Handle は EpochDomain が削除するのを待つだけ。
    // ★ Coordinator は Builder を知らない。Request を発行するのみ。
}
```

### QSVC-5 契約の修正

| 契約ID | 旧内容 | 新内容 |
|--------|--------|--------|
| QSVC-5 | Audit失敗時、State + Audit + Receipt の3状態をロールバック | **Audit失敗時は診断カウンタ更新のみ。State は変更しない。rollback 禁止。** |
| QSVC-5a | `quarantine()` 実行時に `previousState` を保存 | **削除。previousState 不要。** |
| QSVC-5b | `rollbackQuarantine()` で State 復元 | **削除。`submitRecoveryRequest()` で代替。** |
| QSVC-5c | rollback 完了後 Receipt 状態も戻す | **削除。Receipt は Epoch 完了後に解放。** |

### Recovery Intent 契約

`submitRecoveryRequest()` は Recovery Runtime を構築してはならない。**Recovery Request の enqueue のみ**を責務とする。Builder → Validate → Publish の責務境界を維持する。
Recovery Queue は**単なる Transport** であり、Decision Authority を持たない。**Queue 側で重複排除・優先度付与・並べ替え等の Decision を行ってはならない**（push/pop 以外の意味を持たせない。重複排除を始めると Queue が Decision Authority になり、ISR の責務分離を破る）。

**Intent Layer の責務範囲（2026-07-31 明確化）**: `submitRecoveryRequest()` は ISR の **Intent Layer** に属する。すなわち **Request enqueue のみ**であり、**Admission 判定（受け入れ可否の評価）は行わない**。Admission は Builder Loop 側（消費時に `quarantinedHandle` の実在性・Quarantine 状態を検証）が担当する。これにより `submitRecoveryRequest()` は「何も判断しない」純粋な発行関数となり、ISR の「Intent は未来に処理される要求」という定義に完全一致する。

**Recovery Request の消費主体（2026-07-31 明確化）**: Recovery Request は**既存の Builder Loop が消費する**。専用の Recovery Worker は設けない。ISR では `Intent → Builder` の唯一経路が原則であり、Recovery は「Builder への別種の Intent 入力」と位置づける。すなわち:
```
Coordinator --submitRecoveryRequest()--> Recovery Queue
                                          ↓ (pop)
                                   Builder Loop（既存）
                                          ↓
                                   Validate → Publish New RuntimeWorld
                                          ↓
                                   Old Runtime Retire（EpochDomain が処理）
```
Recovery Request は `quarantinedHandle`（復旧対象 DSPHandle）のみをペイロードに持つ自己完結型 Intent とし、Builder Loop はこの Handle から Recovery RuntimeWorld を構築する。専用 Worker を設けない理由: (1) Builder → Validate → Publish の唯一経路を維持できる、(2) 追加スレッドによるスケジューリング Authority の分散を避ける、(3) 既存の Intent Queue 統合（後述の共通 Intent Queue 化）への移行障壁を下げる。

| ID | 契約 |
|----|------|
| RECOVERY-1 | `submitRecoveryRequest()` は Recovery Request の **enqueue のみ**行う。RuntimeWorld の構築・Validate・**Admission 判定**は行わない（純粋な Intent 発行関数）。 |
| RECOVERY-2 | Recovery Runtime は通常の Builder → Validate → Publish 経路を通る。Recovery 例外として Validator を省略してはならない。 |
| RECOVERY-3 | Coordinator は Builder を直接呼ばない。Recovery Request は Queue を経由して Builder Loop へ渡される。Recovery Queue は単なる Transport であり、Decision Authority を持たない。**Queue 数が増えるほど Transport Authority が散らばるため、Recovery Queue は最初から共通 Intent Queue の `type=Recovery` として設計するのが望ましい（FUTURE-10 で一本化）**。 |
| RECOVERY-4 | `submitRecoveryRequest()` は NonRT（MessageThread）からのみ呼び出し可能。 |
| RECOVERY-5 | Recovery Request の消費主体は**既存の Builder Loop**。専用 Recovery Worker は設けない。Recovery Request は `quarantinedHandle` のみをペイロードに持つ自己完結型 Intent。 |
| RECOVERY-6 | Admission 判定（`quarantinedHandle` の実在性・Quarantine 状態検証）は **Builder Loop の消費時**に行う。`submitRecoveryRequest()` は何も判断しない。 |
| RECOVERY-7 | **Recovery Coalescing は Builder 内部の Normalization 処理として定義する（レビュー⑥⑧反映）**: 同一 DSPHandle に対する Pending Recovery を Build 前に統合する。**Queue は immutable であり、Builder が Queue 内容を書き換えることはない**。実装は「Builder 内部に PendingMap（DSPHandle → 最新 Recovery 要求）を保持し、消費時に集約（正規化）→ 最新 1 件を Build」とし、残余は同一 World として消化する。**Normalization と Decision の境界（レビュー⑧で明文化）**: `A, A, A → A` の畳み込みは、同一 Handle に対する**同種・同値の要求を最新 1 件に統合する Normalization** であり、「要求の意味を変えない」操作である。一方 Decision は「優先度付け・破棄・並べ替え」のように**要求の意味を変える**操作を指す。Builder の PendingMap 集約は Normalization に該当し、RECOVERY-3（Queue は Decision しない）とも Builder（New World の生成 Authority）が要求の統合を行うとも矛盾しない。 |
| BUILDER-STATE | **PendingMap は Build Session 限定の内部状態（11回目レビュー①反映・12回目レビューで RAII 強化）**: Builder の PendingMap（RECOVERY-7 の正規化構造）は **Builder の永続状態ではなく Build Session（Build Context）の一部**である。ISR では Builder は「New World の生成」という一過性の作業のみを行い、恒久的な可変状態を保持しない。**契約**: (1) PendingMap は 1 回の Build Session 内でのみ生存する、(2) 当該 Recovery の New World が Publish 完了した時点で**必ず破棄**する（`clearPendingMap()` 等で明示的にリセット）、(3) PendingMap は Build Session 外（Coordinator / Timer / Monitor）から参照・変更されない、(4) 未消費の PendingMap 残余が次の Build Session へ持ち越されない。これにより Builder が「Mutable state を永続保持する Authority」になることを防ぎ、Builder は **Stateless per Build Session** のまま ISR の一方向経路に整合する。**RAII BuildSession による終了保証（12回目レビュー②・最終評価2反映）**: PendingMap の破棄は**正常完了時だけでなく、Build 失敗・例外・キャンセル・Admission Reject を含む全終了経路**で保証される必要がある。実装は **`BuildSession` RAII Guard**（コンストラクタで PendingMap 生成 → デストラクタで必ず破棄）に閉じ込め、手動の `clearPendingMap()` 呼び出しに依存しない。RAII により「PendingMap が例外伝播や早期 return で残留する」ことを構造的に排除する。static_assert 又はテストで「全終了経路後に PendingMap が空」を検証する（例外安全性テスト含む）。 |

> **注 — Recovery Coalescing（2026-07-31 反映）**: Recovery Queue が FIFO であるため、同一 DSPHandle に対する Recovery Request が大量に滞留する可能性がある（`A, A, A, A, A`）。Queue 側は Decision 禁止（RECOVERY-3）のため、**同一 Handle の統合は Builder 内の PendingMap 正規化として行う**（RECOVERY-7）。**Queue 自体は immutable（Transport のみ）であり、Builder が Queue を書き換える実装にはしない**。Builder は Queue から Intent を pop し、自身の PendingMap に集約（同一 Handle は最新 1 件に正規化）してから Build する。これは Queue の Decision でも変更でもなく、**Intent Merge（複数 Intent を 1 つの Build に統合する Normalization）**であり、ISR の「Queue は Transport のみ」原則を守りつつ Builder の無駄な再 Build を防ぐ。**Normalization が Decision でない理由（レビュー⑧で明文化）**: 正規化は「同一実体（DSPHandle）の重複要求を束ねる」だけで要求内容を変えない。Decision（優先度付け・破棄・並べ替え）は要求間の序列や取捨を決め、要求の意味を変える。**PendingMap 集約は後者を行わない**ため Authority の侵害にならない。

> **注 — 共通 Intent Queue への統合（最終形・FUTURE-10 参照）**: RECOVERY-3 の Recovery Queue は、Observe Queue・Publish Intent・Quarantine Request と統合した**共通 Intent Queue** として設計するのが ISR の最終形である。ISR では Intent は統一的なイベント（`Intent { type, payload }`）として扱え、`LockFreeRingBuffer<Intent>` 一本で FIFO 保証・Authority 単一化・SPSC を実現できる。現時点では種別別 Queue としているが、**共通 Intent Queue 一本化を最終形として採用する**（FUTURE-10）。

> **注 — Recovery Queue は temporary adapter（レビュー⑦反映・2026-07-31）**: 種別別 Recovery Queue は ISR の最終形ではなく、**FUTURE-10 の共通 Intent Queue が完成するまでの過渡実装（temporary adapter）**である。したがって (1) 本 Queue への依存は最低限に留め、(2) 独自機能（Decision・重複排除・優先度等）を追加しない（RECOVERY-3 の Transport 原則に従う）、(3) FUTURE-10 統合時に機械的置換で `intentQueue_` へ移行できる**最小 API のみ**（enqueue/pop）を公開する。新規 Intent は最初から共通 `Intent` 型で定義し、temporary adapter 期間が短くなるようにする。

### 実装ステップ（🔧 今回実装）

1. `ISRRuntimePublicationCoordinator.h` に `submitRecoveryRequest(const DSPHandle& quarantinedHandle) noexcept` を宣言
2. 実装は **Recovery Queue への enqueue のみ**（FUTURE-10 で共通 `intentQueue_` の `type=Recovery` へ統合する。今回は将来の統合を見据え `LockFreeRingBuffer<RecoveryRequest>` を単一定義）。**temporary adapter として最小 API（enqueue/pop）のみ公開**し、独自機能を追加しない（レビュー⑦反映）
3. 既存の `QuarantineService::executeQuarantine()` 内で `result.rolledBack = false` 相当の rollback 経路は **実コード上は未削除**（`ISRRuntimePublicationCoordinator.cpp:629`・`.h:46` に残存。動作は `rolledBack=false` のまま恒常的に無効化されているが、フィールドと代入文は残る）
4. **Admission 判定は Builder Loop 消費時に実装**（`quarantinedHandle` の実在性・Quarantine 状態検証）。Builder は自身の **PendingMap**（DSPHandle → 最新 Recovery 要求）に pop した Intent を集約し、同一 Handle を正規化してから Build する（RECOVERY-7。Queue は immutable）。**RAII BuildSession で実装（12回目レビュー②反映・BUILDER-STATE）**: PendingMap の生成・破棄は `BuildSession` RAII Guard に閉じ込め、Build 成功 / 失敗 / 例外 / キャンセル / Admission Reject の全終了経路でデストラクタが必ず PendingMap を破棄する（手動 `clearPendingMap()` に依存しない）
5. FUTURE-10 統合時に `RecoveryRequest` を共通 `Intent{type=Recovery}` へ変換（contract は RECOVERY-1〜7 を継承）

### 変更ファイル（🔧 今回実装）

| ファイル | 変更内容 |
|---------|---------|
| `ISRRuntimePublicationCoordinator.h` | `submitRecoveryRequest()` 宣言。`LockFreeRingBuffer<RecoveryRequest, kRecoveryQueueCapacity> recoveryQueue_` 追加。 |
| `ISRRuntimePublicationCoordinator.cpp` | `submitRecoveryRequest()` 実装（enqueue のみ）。 |
| Builder Loop（`AudioEngine.h` 内 build 経路） | Recovery 消費時の Admission 判定 + RECOVERY-7 coalescing。 |

### テスト計画（🔧 今回実装）

```cpp
// tests/RecoveryIntentTests.cpp
TEST(RecoverySubmitEnqueueOnly) {
    // submitRecoveryRequest() が enqueue のみ（build しない）ことを検証
    coordinator.submitRecoveryRequest(handle);
    ASSERT_EQ(coordinator.recoveryQueueCount(), 1);
}
TEST(RecoveryCoalescingBuilderSide) {
    // 同一 Handle の Recovery が複数滞留 → Builder の PendingMap で 1 件に正規化
    // （Queue 自体は immutable: pop のみで書き換えないことを同時に検証）
    // → Queue 内の Intent 数は不変、Build は 1 回のみ実行
}
TEST(RecoveryQueueImmutable) {
    // Builder が Queue 内容を書き換えない（RECOVERY-7: Queue は Transport のみ）
}
TEST(RecoveryAdmissionAtConsume) {
    // 実在しない Handle / 非 Quarantine 状態 → Builder 消費時に拒否
}
TEST(BuildSessionRAIIExceptionSafety) {
    // ★ 12回目レビュー②反映（BUILDER-STATE）: Build 中に例外が発生しても
    //   BuildSession デストラクタが PendingMap を必ず破棄する（例外安全性）
    //   → 例外後も PendingMap が空であることを検証
}
TEST(BuildSessionRAIICancelAndReject) {
    // ★ 12回目レビュー②反映（BUILDER-STATE）: キャンセル / Admission Reject でも
    //   BuildSession デストラクタが PendingMap を必ず破棄する
}
```

### 完了条件（🔧 今回実装）

1. `submitRecoveryRequest()` が enqueue のみを実行し、Build / Validate / Admission を行わない（RECOVERY-1/6）
2. Builder Loop が Recovery を通常の Builder → Validate → Publish 経路で処理する（RECOVERY-2）
3. Builder の PendingMap による同一 Handle coalescing（正規化）が動作し、Queue は immutable（RECOVERY-7）
4. **BuildSession RAII により PendingMap が全終了経路で破棄される（BUILDER-STATE・12回目レビュー②反映）**: 成功・失敗・例外・キャンセル・Admission Reject の各経路後に PendingMap が空であることをテストで検証
5. FUTURE-10 統合時に `type=Recovery` への変換が contract 変更なしで可能

---

## FUTURE-4: persistentState_ の廃止と RuntimeWorld Metadata Snapshot 統合 [🔧 今回実装] — 📋 設計確定

### 目的

`persistentState_`（plain struct, 3×uint64_t）は MessageThread-only の前提だが、`emitObserveIntent()` が Timer Thread から読んでいる。ISR の Single Source of Truth 原則に従い、RuntimeWorld を唯一の Metadata Authority とする。

### 設計方針

`persistentState_` を廃止し、全メタデータ（epoch + generation + sequence）を RuntimeWorld 内の ObservationMetadata 構造体として統合する。Timer Thread は `const RuntimeWorld*` を1回読み取るだけで全メタデータを取得する。

```
// Before: 3 sources of truth
persistentState_.publicationEpoch       ← plain struct, cross-thread unsafe
persistentState_.mappedRuntimeGeneration ← same struct
persistentState_.publicationSequenceId   ← same struct

// After: 1 source of truth
RuntimeWorld::metadata::ObservationMetadata
  ├── epoch          ← RuntimeWorld publish時に atomically に設定
  ├── generation     ← RuntimeWorld publish時に atomically に設定
  └── sequence       ← 同上。DeferredDeletionQueue 等からは world 経由で参照
  │
  └── Timer Thread: const RuntimeWorld* world = consumeAtomic(currentWorld_)
       → world->metadata.epoch, world->metadata.generation, world->metadata.sequence
       → 1回の atomic load で全メタデータが一貫性をもって取得可能 ("RuntimeWorld Metadata Snapshot")
```

### ISR 設計判断

| 方式 | 問題点 | 採用 |
|------|--------|------|
| `persistentState_` 廃止、RuntimeWorld Metadata Snapshot 統合（**本設計で採用**） | `currentWorld_` の atomic load 1回で全メタデータを一貫性をもって取得。epoch/generation/sequence 間の inconsistency が原理的に発生しない。Single Source of Truth。 | ✅ **本設計** |
| atomic epoch cache + RuntimeWorld generation（過渡的措置） | epoch だけ atomic cache、generation は World から別途取得。epoch==N, generation==N-1 の inconsistency が理論上発生する。Metadata Authority が3箇所に分散。本設計への移行までの暫定措置としてのみ許容。 | ⏳ 過渡的措置 |
| `std::atomic<PersistentStateBlock>` | lock-free 非保証（icx）。mutex リスク。 | ❌ |
| plain struct 維持 | cross-thread の一貫性未保証。既知の技術負債。 | ❌ |

### 設計

```cpp
// RuntimeWorld 内で新たに定義される ObservationMetadata
// ★ 実コード対応（2026-07-31 反映）: 既存の ISRRuntimeSemanticSchema.h には
//   RuntimeMetadata（schemaVersion + publicationSequence, L272-276）と
//   PublicationSemantic（sequenceId + epoch + mappedRuntimeGeneration + previousSequenceId, L252-257）が存在。
//   本設計の ObservationMetadata はこれらを「publish 確定後に const で凍結する」形へ拡張する。

// ★ Metadata 完全不変化（2026-07-31 反映）: レビュー指摘「metadata 自体も Publish 後は
//   変更不能であることを型・実装で保証すべき」。ISR では「Publish 後は Immutable」は
//   RuntimeWorld の全フィールド（metadata 含む）に適用される。
//   実装は「const 凍結」を採用する:
//   - RuntimeWorld は build → publish の間に metadata を確定し、publish 以降は const で凍結
//   - 凍結は型レベル（`const RuntimeSemanticSchema` 参照経由のみ公開）で保証
//   - MutablePrePublish（`ISRRuntimeSemanticSchema.h:35` の `MutabilityClass` enum、`:260-263` の `kFieldDescriptors` で使用）を厳守し、
//     publish 境界を越えた書き込みをコンパイル時・レビュー時に排除

struct ObservationMetadata {
    PublicationEpoch epoch{0};            // = PublicationSemantic.epoch
    uint64_t generation{0};               // = PublicationSemantic.mappedRuntimeGeneration
    uint64_t publicationSequence{0};      // = PublicationSemantic.sequenceId
};
// static_assert: ObservationMetadata は trivially copyable（RuntimeWorld の atomically publish 用）

// ISRRuntimePublicationCoordinator.h — persistentState_ 完全削除
// Before:
//   PersistentStateBlock persistentState_{};   // 削除（3フィールドのplain struct）
// After:
//   persistentState_ は完全廃止。
//   全スレッドは consumeAtomic(currentWorld_) から ObservationMetadata を取得。

// MessageThread 書き込み（commit 内）:
//   1. RuntimeWorld を build（この時点で metadata は確定。以後 const 凍結）
//   2. atomic<const void*>::store(currentWorld_, newWorld, release)
//      → RuntimeWorld + 全メタデータが atomically に公開される
//      ※ publish 後は const RuntimeSemanticSchema 参照のみ公開する（Builder のみ
//        mutable アクセス可。publish 境界を越えた書き込みはコンパイルエラーで防止）

// Timer Thread 読み取り（emitObserveIntent 内）:
//   const auto* world = static_cast<const RuntimeState*>(
//       convo::consumeAtomic(currentWorld_, std::memory_order_acquire));
//   if (world != nullptr) {
//       ObserveIntent intent{
//           fadingHandle,
//           world->metadata.epoch,            // const 経由読み取り（変更不可）
//           world->metadata.publicationSequence
//       };
//   }
//   → 1回の atomic load で epoch + generation + sequence を一貫性をもって取得
//   → epoch/generation/sequence 間の inconsistency は原理的に発生しない
//   → metadata は const 凍結のため、読み取り側からは変更不可能（SSoT 不変）
```

### 過渡的措置（atomic epoch cache）

> **最終形は Metadata Snapshot 方式（2026-07-31 反映）**: Practical Stable ISR では `RuntimeWorld → Metadata → epoch/generation/sequence` が**一体（Snapshot）**である。設計上の本命は上記の「RuntimeWorld Metadata Snapshot 方式」であり、**atomic epoch cache は現実解（暫定措置）であり最終形ではない**。本 cache は移行までの橋渡しとしてのみ存在し、Metadata Snapshot 方式の実装完了と同時に**削除する**。削除までの間も、cache と RuntimeWorld Metadata の不一致（epoch=N, generation=N-1 等）が発生し得ることを許容し、その旨をコードコメントに明記する。

RuntimeWorld Metadata Snapshot 方式への完全移行までの暫定措置として、`atomic<PublicationEpoch> currentPublicationEpoch_` のみを追加する。この方式では epoch と generation の inconsistency が理論上発生するが、ObserveIntent の世代逆転検出は epoch のみで動作するため実用上問題ない。

> **設計上の注意（2026-07-31 反映）**: `currentPublicationEpoch_` / `currentPublicationSequenceId_` は**あくまで一時的な Cache であり、Authority ではない**。これを独立した「擬似 Authority」として扱わないこと。設計上の Single Source of Truth は RuntimeWorld Metadata（`RuntimeWorld::metadata`）であり、本 cache は Metadata Snapshot 方式への移行が完了した時点で**削除される**。本 cache の存続期間中も、読み取り側（Observer）は必ず **RuntimeWorld 経由**で epoch を取得できる構造を維持する（cache 読み取りは診断・Monitor 用途に限定し、制御フローを cache に依存させない）。
>
> **DebugOnly 限定（2026-07-31 反映）**: 暫定 cache は**診断用途に限定し、コードレベルでも Authority として扱われないことを保証**する。実装は `#if ISR_METADATA_TRANSITION` ガードで囲み、RuntimeWorld Metadata Snapshot 方式への移行完了時にコンパイル単位で削除されるようにする。これにより「コード上に atomic が存在する」こと自体が将来の誤用（cache 読み取りによる制御フロー依存）を誘発するリスクを排除する。さらに **`[[deprecated]]` 属性を付与**し、新規参照をコンパイル警告で抑止する（既存参照は `TODO(Remove after Metadata Snapshot)` を併記）。「一時的」と注記するだけでなく、**型・属性レベルで再利用不能にする**ことが本指摘の趣旨である。

```cpp
// ★ 過渡的措置: RuntimeWorld Metadata Snapshot 移行までの暫定 atomic cache
//   ※ 本 cache は一時的措置。Metadata Snapshot 方式へ統一後は削除する。
//   ※ 診断・Monitor 用途のみ。制御フローを本 cache に依存させてはならない。
//   ※ #if ISR_METADATA_TRANSITION ガードで囲み、移行完了時にコンパイル単位で削除される。
//   ★ 削除必須マーカー（2026-07-31 反映）: 本 cache は [[deprecated]] を付与し、
//     いかなる新規参照もコンパイル警告で抑止する。既存参照はすべて
//     TODO(Remove after Metadata Snapshot) コメントを併記する。
//     これにより「数年後に Authority 化」するリスクをコードレベルで封じる。
#if defined(ISR_METADATA_TRANSITION)
[[deprecated("Temporary diagnostic cache — remove after RuntimeWorld Metadata Snapshot")]]
std::atomic<PublicationEpoch> currentPublicationEpoch_{0};      // TODO(Remove after Metadata Snapshot)
[[deprecated("Temporary diagnostic cache — remove after RuntimeWorld Metadata Snapshot")]]
std::atomic<uint64_t> currentPublicationSequenceId_{0};         // TODO(Remove after Metadata Snapshot)

// HB契約（過渡的措置中使用）:
//   Writer: publishAtomic(currentWorld_) → publishAtomic(currentPublicationEpoch_)
//   → RuntimeWorld 公開が epoch 更新より先行することを保証
//   ※ 直接の .store() / .load() は使用せず、必ず publishAtomic() / consumeAtomic() で統一する
//     （Practical Stable ISR 運用規約: load/store 直接利用は禁止）
#endif // ISR_METADATA_TRANSITION
```

### 削除されるメンバ

| 現状メンバ | 移行先 | 理由 |
|-----------|--------|------|
| `persistentState_.publicationEpoch` | `RuntimeWorld::metadata.epoch`（過渡的: `#if ISR_METADATA_TRANSITION` ガード付き DebugOnly cache） | Single Source of Truth: RuntimeWorld |
| `persistentState_.mappedRuntimeGeneration` | `RuntimeWorld::metadata.generation`（`consumeAtomic(currentWorld_)` 経由） | RuntimeWorld は publish 後に Immutable |
| `persistentState_.publicationSequenceId` | `RuntimeWorld::metadata.publicationSequence`（過渡的: `#if ISR_METADATA_TRANSITION` ガード付き DebugOnly cache） | 複数ファイル参照のため過渡的に atomic cache を許容（診断用途限定） |

#### トレードオフ

| 方式 | メリット | デメリット |
|------|---------|-----------|
| **RuntimeWorld 統合（採用）** | ISR 完全準拠。atomic 1個。lock-free 保証。 | `getVersion()` の実装変更が必要（world→generation）。 |
| atomic 2個 + Seqlock | 2変数の論理的一貫性を保証しようとする。 | Seqlock として不完全（writer が seq++ を1回のみ）。epoch だけ古い状態を検出不可。 |
| plain struct 維持 | 変更ゼロ。 | cross-thread の一貫性未保証。既知の技術負債。 |

#### リスク

`currentPublicationEpoch_` が単一 `std::atomic<uint64_t>` であるため、C++ メモリモデル上の問題は完全に解決される。`getVersion()` の実装が `persistentState_.mappedRuntimeGeneration` から `world->generation` に変わるが、`currentWorld_` の読み取りは既存の `consumeAtomic` パターンと同一。**本 cache は一時的措置であり、`#if ISR_METADATA_TRANSITION` ガード付き DebugOnly としてのみ存在し、Metadata Snapshot 方式への移行時にコンパイル単位で削除する（擬似 Authority 化しない）。**

#### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ISRRuntimePublicationCoordinator.h` | `persistentState_` を削除。`#if ISR_METADATA_TRANSITION` ガード付きで `std::atomic<PublicationEpoch> currentPublicationEpoch_` + `std::atomic<uint64_t> currentPublicationSequenceId_` を追加（DebugOnly・診断用途）。`getVersion()` の実装を `currentWorld_` 経由の RuntimeWorld 読み取りに変更。 |
| `ISRRuntimePublicationCoordinator.cpp` | 全 `persistentState_` 参照を `currentPublicationEpoch_` / `currentPublicationSequenceId_` / `currentWorld_`（RuntimeWorld generation）に変更。`commit()` 内の3フィールド書込を各 `publishAtomic()` 呼び出しに分割（直接の `.store()` は使用しない）。 |

### 契約（METADATA-1〜7）

| ID | 契約 |
|----|------|
| METADATA-1 | RuntimeWorld が唯一の Metadata Authority。`persistentState_` は廃止し、全メタデータ（epoch + generation + sequence）を `RuntimeWorld::metadata` に統合する。 |
| METADATA-2 | 全スレッドは `consumeAtomic(currentWorld_)` の1回の atomic load で全メタデータを一貫取得する（Snapshot 方式）。epoch/generation/sequence の不一致は原理的に発生しない。 |
| METADATA-3 | `MutablePrePublish` を厳守する（実コード: `PublicationSemantic::kFieldDescriptors` の `MutabilityClass::MutablePrePublish`）。Build 前のフィールドのみ Mutable であり、Publish 境界を越えた書き込みは排除する。 |
| METADATA-4 | **Freeze Authority は Builder のみ（レビュー⑦⑨反映）**: Metadata の確定（freeze）は **Builder 側の `finalizeMetadata()`** が実行する。**実行順序を ADR レベルで固定（9回目レビュー③反映）**: `Builder → Metadata 完成 → Freeze（finalizeMetadata, build 完了時点）→ Validator（freeze 済み const World を検証）→ Publish`。**Freeze は Validator より前**である。Validator（`ImmutableWorldVerifier`）は「freeze 済み const World が本当に不変であるか」を検証するのであり、freeze 前の可変 World を検証するのではない。Freeze 前検証（Builder → Validator → Freeze → Publish）は不採用。Coordinator（Publisher）は freeze 済みの const RuntimeWorld を公開するのみであり、metadata の書き込みは行わない。実コード対応: `finalizeRuntimeBuildSnapshot()`（`AudioEngine.RebuildDispatch.cpp:107`）が build 完了時に snapshot を seal（freeze）する既存パターンに合わせ、`RuntimeMetadata` / `PublicationSemantic`（`ISRRuntimeSemanticSchema.h:252-276`）の確定も build 完了時点で const 化する。`SemanticTransactionState::Published` は terminal 状態（同 `:546`）であり、`ImmutableWorldVerifier` が Fatal verifier（同 `:455`）として不変性を検証する。 |
| METADATA-5 | **Builder のみが Mutable アクセスを許可され、`finalizeMetadata()` で metadata を const 凍結する**。`publishAtomic(currentWorld_, newWorld, release)`（`ISRRuntimePublicationCoordinator.cpp:107`）の呼び出し以降、RuntimeWorld は const 凍結され、`const RuntimeSemanticSchema` 参照のみ公開する。Coordinator（Publisher）は freeze 済み World の公開のみ行い、Metadata を変更できない。 |
| METADATA-6 | 過渡的 atomic cache（`#if ISR_METADATA_TRANSITION`）は Authority ではなく診断用途限定。`[[deprecated]]` 付与 + `TODO(Remove after Metadata Snapshot)` 併記。移行完了と同時に削除する。**完全移行後の物理削除を完了条件で強制（10回目レビュー③反映・2026-07-31）**: Debug 用途であっても過渡 cache が Authority 化するのを防ぐため、FUTURE-4 完了条件6 として `rg "currentPublicationEpoch_|currentPublicationSequenceId_"` 0件を確認する。 |
| METADATA-7 | Publish 後の不変性保証は型（const 参照のみ公開）+ 実装（`SemanticTransactionState::Published` が terminal）+ 検証（`ImmutableWorldVerifier` が Fatal）の3層で担保する。この不変性を破るコードはコンパイルエラーまたは Fatal verifier で阻止される。 |

### 実装ステップ（🔧 今回実装）

1. **実コード調査（2026-07-31 確認済み）**: 現状の参照箇所は `ISRRuntimePublicationCoordinator.h:98`（`currentPublicationEpoch()`）、`.cpp:87,101,174,522`（`persistentState_` 読書）、`_ProcessIntent.cpp:13,21`（世代逆転検出）、`.h:271`（メンバ定義）の計8箇所。`commit()` 内（`.cpp:87-101`）で3フィールドが書込まれる。
2. **`persistentState_` 削除**: `PersistentStateBlock`（`.h:271`）を削除し、`currentPublicationEpoch()` / `getVersion()` を `consumeAtomic(currentWorld_)` 経由の RuntimeWorld Metadata 読み取りに変更
3. **Metadata 書き込み経路（Builder freeze 方式・レビュー⑦⑨反映）**: **Builder が build 完了時に `finalizeMetadata()` で metadata（epoch/generation/sequence）を確定し const 凍結**する（`finalizeRuntimeBuildSnapshot()` の既存パターン `AudioEngine.RebuildDispatch.cpp:107` に合わせる）。**順序の固定（9回目レビュー③反映）**: `Builder → Metadata 完成 → Freeze → Validator → Publish`。Freeze は build 完了時点で実行され、**Validator は freeze 済み const World を検証する**（Freeze 前検証は不採用、METADATA-4 参照）。`commit()` は freeze 済み const RuntimeWorld を `publishAtomic(currentWorld_, newWorld, release)` で原子公開するのみ。**Metadata の完全不変化（const 凍結）**を build 完了時点で成立させ、以後 const 参照のみ（METADATA-4/5）
4. **過渡的措置**: `#if ISR_METADATA_TRANSITION` ガード + `[[deprecated]]` 付き DebugOnly atomic cache を追加（診断用途限定）。既存の全参照に `TODO(Remove after Metadata Snapshot)` を併記
5. **過渡的 cache の物理撤去（10回目レビュー③反映・2026-07-31）**: Metadata Snapshot 統合完了後、`#if defined(ISR_METADATA_TRANSITION)` ブロック全体と `currentPublicationEpoch_` / `currentPublicationSequenceId_` を削除する。`rg "currentPublicationEpoch_|currentPublicationSequenceId_"` で 0 件になることを確認（完了条件6）。「一時的」という注記に留めず、**物理削除で Authority 化の芽を断つ**
6. **検証**: `getVersion()` が `world->metadata.generation` を返すことを確認。`emitObserveIntent()`（Timer Thread）が `consumeAtomic(currentWorld_)` 1回で epoch+generation+sequence を一貫取得することを確認。`finalizeMetadata()` が build 完了時に1回だけ呼ばれ、以後 Mutable アクセスが型レベルで拒否されることを確認

### テスト計画（🔧 今回実装）

```cpp
// tests/MetadataSnapshotTests.cpp
TEST(MetadataSnapshotSingleAtomicLoad) {
    // 1回の consumeAtomic(currentWorld_) で epoch/generation/sequence が一貫取得できる
    // （persistentState_ の 3 フィールド読取が消えたことを検証）
}
TEST(MetadataImmutableAfterPublish) {
    // publish 後の RuntimeSemanticSchema が const 経由でしか参照できない（コンパイル時保証）
}
TEST(MetadataSnapshotEpochGenerationConsistency) {
    // epoch==N かつ generation==N-1 の不一致が原理的に発生しないこと
}
TEST(TransitionCacheDiagnosticOnly) {
    // #if ISR_METADATA_TRANSITION 時のみ atomic cache が存在し、診断用途に限定されること
}
```

### 完了条件（🔧 今回実装）

1. `persistentState_`（`.h:271`）が完全削除される
2. 全スレッドが `consumeAtomic(currentWorld_)` 経由で Metadata を一貫取得する（Snapshot 方式）
3. Metadata が publish 後 const 凍結され、書き込み経路が型レベルで排除される（METADATA-4/5/7: Freeze 責任者は Publisher のみ）
4. 過渡的 atomic cache が `#if ISR_METADATA_TRANSITION` + `[[deprecated]]` で限定される（METADATA-6）
5. METADATA-1〜7 の契約が全て満たされる（`SemanticTransactionState::Published` terminal / `ImmutableWorldVerifier` Fatal をテストで確認）
6. **過渡的 cache の完全撤去（10回目レビュー③反映・2026-07-31）**: `currentPublicationEpoch_` / `currentPublicationSequenceId_` を **Metadata Snapshot 統合完了時にコンパイル単位で削除**する。**`rg "currentPublicationEpoch_|currentPublicationSequenceId_"` で 0 件になることを確認**し、`ISR_METADATA_TRANSITION` ガードの削除と `#if defined(ISR_METADATA_TRANSITION)` ブロック全体の除去を実施する。Debug 用途であっても過渡 cache は Authority になり得るため、**「一時的」という認識に留めず、実装時に確実に物理削除する**（レビュー「完全移行後には削除必須」）
7. `git diff --check` クリーン + 既存テスト全件通過

---

## FUTURE-5: MemoryPool化 [🔧 今回実装] — 📋 設計確定

### 目的

DSPHandle の内部ストレージを動的メモリプールに移行する。**2026-07-31 決定により今回改修で実装する**（旧判断「将来対応維持」を撤回）。固定256スロット（`std::array`）の上限撤廃と、将来のスロット拡張時における RT-bounded な確保経路の確立が目的。

> **位置づけ（レビュー⑦反映・2026-07-31）**: MemoryPool 化は **ISR 本体（Coordinator / Intent / Metadata / Shutdown）と独立した Storage Policy 改善**である。ISR の正しさ（Authority ・不変条件）には影響せず、**性能・スケーラビリティ改善（P1〜P2 相当）**として扱う。優先度は ISR 完成系（FUTURE-3/4/7/8/9/10 + Shutdown Pipeline）より下位とし、**実装順序では ISR 系完了後に実施**する。ただし RT-bounded な確保経路の確立は ISR のリアルタイム性の前提となるため、**FUTURE-5 実装時点で「非 RT コンテキストからの確保禁止」契約（後述）を必ず含める**。

### 実コード調査結果（2026-07-31）

- `struct DSPRegistrySlot` — `ISRDSPHandle.h:91`
- `class DSPHandleRuntime` — `ISRDSPHandle.h:104`
- `static constexpr size_t MAX_DSP_SLOTS = 256` — `ISRDSPHandle.h:107`
- `std::array<DSPRegistrySlot, MAX_DSP_SLOTS> registry_{}` — `ISRDSPHandle.h:168`

### 設計方針

```
現状:
  std::array<DSPRegistrySlot, 256> registry_;        // ISRDSPHandle.h:168
  → 256 スロット固定。compile-time 確保。

今回実装:
  MemoryPool<DSPRegistrySlot> registryPool_;         // ページ単位・動的拡張
  → 初期ページ = 256 スロット（MEMPOOL-4: 後方互換）
  → ページ拡張は NonRT（MessageThread）でのみ実行（MEMPOOL-1）
  → RT パスのスロット確保は O(1) bounded（MEMPOOL-2: フリースロットリングバッファ）
  → 縮小は明示的 shrink のみ（MEMPOOL-3）
  → スロットアドレスはページ内連続（cache locality 維持: ページ単位で dense array を保持）
```

### 契約

| ID | 契約 |
|----|------|
| MEMPOOL-1 | プール拡張は NonRT（MessageThread）でのみ実行する |
| MEMPOOL-2 | RT パスでのスロット確保は O(1) bounded とする |
| MEMPOOL-3 | プール縮小は明示的な shrink 操作のみ |
| MEMPOOL-4 | プールの初期容量は 256 スロット（後方互換） |

### 実装ステップ（🔧 今回実装）

1. **`MemoryPool<DSPRegistrySlot>` を新規作成**（`ISRMemoryPool.h`）。ページ管理: `std::vector<std::unique_ptr<DSPRegistrySlot[]>>` + フリースロットインデックスの `LockFreeRingBuffer<size_t>`（RT 側 O(1) 確保/解放）
2. **`DSPHandleRuntime::registry_`（`.h:168`）を `registryPool_` に置換**。全アクセスを `slotHandle->...` から `registryPool_[slot]` に変更（インデックス経由で既存契約を維持）
3. **スロット取得経路を分離**: 通常取得（NonRT）はページ拡張を許容。RT パスの取得（fading チェック等）はフリースロットからの O(1) bounded 確保のみに制限
4. **解放経路**: `DSPHandleRuntime::release()` 相当でフリースロットへ戻す + 該当ページが空なら shrink 候補として NonRT に通知
5. **page 上限を導入**: `kMaxPages` を設け、ページ上限到達時に生成を `DSPHandle::Null` で返す（RT bounded 維持）。現行の 256 スロット時と同一の挙動を初期状態で再現
6. **検証**: 256 スロットを超える DSP 生成（テスト用）が動的確保で成功すること、RT パスに malloc/new が含まれないことを確認

### 変更ファイル（🔧 今回実装）

| ファイル | 変更内容 |
|---------|---------|
| `ISRMemoryPool.h`（新規） | `MemoryPool<DSPRegistrySlot>`: ページ配列 + フリースロット `LockFreeRingBuffer` + NonRT 拡張 API |
| `ISRDSPHandle.h` | `registry_`（:168）を `registryPool_` に置換。全メソッドのスロットアクセスをプール経由に変更 |
| `AudioEngine.h` | `dspHandleRuntime_` の生成/解放経路（:4017-4120）をプール API に追従 |
| テスト | `tests/MemoryPoolTests.cpp`（下記） |

### テスト計画（🔧 今回実装）

```cpp
// tests/MemoryPoolTests.cpp
TEST(MemoryPoolInitialCapacity) {
    // 初期 256 スロットが利用可能（MEMPOOL-4 後方互換）
}
TEST(MemoryPoolDynamicExpansionNonRT) {
    // 256 超の確保が NonRT ページ拡張で成功する（上限撤廃）
}
TEST(MemoryPoolRTAllocationBounded) {
    // RT パスの確保が O(1) でページ拡張を行わない（MEMPOOL-1/2）
}
TEST(MemoryPoolSlotReuse) {
    // 解放スロットがフリースロットに戻り、再利用される
}
TEST(MemoryPoolShrinkOnlyExplicit) {
    // 縮小が明示的 shrink 呼び出しでのみ発生（MEMPOOL-3）
}
TEST(MemoryPoolPageCap) {
    // kMaxPages 到達時に DSPHandle::Null が返る（RT bounded 維持）
}
```

### 完了条件（🔧 今回実装）

1. 256 スロット制限が撤廃され、動的確保に移行（MEMPOOL-1〜4 全て満たす）
2. RT パスのパフォーマンスが現状と同等（初期 256 スロット時）
3. スロット再利用（フリースロット）が generation と整合する（FUTURE-6 の HTABLE-3 と併用時も矛盾しない）
4. 既存テスト全件通過 + 上記 MemoryPool テスト合格

---

## FUTURE-6: Handle Table 完全移行 [🔧 今回実装] — 📋 設計確定

### 目的

`std::unordered_map<DSPCore*, DSPHandle>` を Handle Table に移行する。**2026-07-31 決定により今回改修で実装する**（旧判断「将来対応維持」を撤回）。`eraseByHandle` の linear scan（O(n)）を双方向 O(1) に置換し、FUTURE-5（MemoryPool）のスロットインデックスと密結合で dense array アクセスを成立させる。

> **位置づけ（レビュー⑦反映・2026-07-31）**: Handle Table 移行も FUTURE-5 と同様、**ISR 本体とは独立した性能改善（Storage Policy）**である。**Handle（DSPHandle）は ISR の唯一の識別子であり、その検索/削除の実装（unordered_map → Table）が ISR の Authority や不変条件に影響しない**ことを原則とする。優先度は FUTURE-5 と合わせて **ISR 完成系の後に実施**（P1〜P2 相当）。なお双方向 O(1) 化は `eraseByHandle` の線形走査を解消する確実な改善であり、FUTURE-5 実装時（スロット dense array 化）に同時実施するのが効率的。

### 実コード調査結果（2026-07-31）

- `std::unordered_map<DSPCore*, convo::isr::DSPHandle> runtimeDSPHandleMap_;` — `AudioEngine.h:4420`
- forward（`DSPCore* → DSPHandle`）: `runtimeDSPHandleMap_.find/emplace/erase` — `AudioEngine.h:4017-4058`
- reverse（`DSPHandle → DSPCore*`）: `eraseByHandle` の linear scan — `AudioEngine.h:4091-4098`（`for (it = begin; it != end; ++it)`）
- `eraseByHandle` の呼び出し元: `AudioEngine.h:4119`（`rollbackDSPHandleRegistration`（:4112 定義）内）

### 設計方針

```
現状:
  std::unordered_map<DSPCore*, DSPHandle> runtimeDSPHandleMap_;   // AudioEngine.h:4420
  DSPHandle → DSPCore* の逆引きは linear scan（eraseByHandle: O(n)）

今回実装:
  HandleTable<DSPHandle, DSPCore*> handleTable_;
  → forward（DSPCore* → DSPHandle）: O(1) hash または直接ポインタインデックス
  → reverse（DSPHandle → DSPCore*）: O(1) スロット配列の逆エントリ（FUTURE-5 の slot index と 1:1 対応）
  → メモリアクセスパターンの改善（密配列）
```

### 契約

| ID | 契約 |
|----|------|
| HTABLE-1 | forward map（DSPCore* → DSPHandle）は O(1) |
| HTABLE-2 | reverse map（DSPHandle → DSPCore*）は O(1) |
| HTABLE-3 | スロット再利用は generation で ABA 防止 |
| HTABLE-4 | 全操作は lock-free または bounded mutex |

### 実装ステップ（🔧 今回実装）

1. **`HandleTable<DSPHandle, DSPCore*>` を新規作成**（`ISRHandleTable.h`）。forward は `std::unordered_map<DSPCore*, uint32_t slot>` のまま（ハッシュは O(1)）だが、**reverse は `std::array<DSPCore*, MAX_DSP_SLOTS> reverseSlot_` の密配列**（FUTURE-5 のプールページと 1:1 対応）。`DSPHandle{slot, generation}` の slot が直接配列インデックスになる
2. **`runtimeDSPHandleMap_`（:4420）を `handleTable_` に置換**。`find/emplace/erase`（:4017-4058）は `handleTable_.forward_` 相当 API に変更
3. **`eraseByHandle`（:4091-4098）を O(1) に変更**: linear scan を `reverseSlot_[handle.slot]` の直接参照 + generation 検証（HTABLE-3）に置換。`reverseSlot_[slot] == dsp` かつ generation 一致時のみ erase
4. **`rollbackDSPHandleRegistration`（:4112）を `handleTable_.removeByHandle()` に変更**。generation 検証は `DSPHandleRuntime::registry_`（または `registryPool_`）の slot generation と照合
5. **generation の一元管理**: ABA 防止のため generation は `DSPRegistrySlot`（FUTURE-5 のプールスロット）と Handle Table で同一ソースから取得（`DSPHandleRuntime::allocateSlot()` が発行）
6. **検証**: `eraseByHandle` の計算量が O(1) になったことを benchmark で確認。線形探索コードの完全削除

### 変更ファイル（🔧 今回実装）

| ファイル | 変更内容 |
|---------|---------|
| `ISRHandleTable.h`（新規） | `HandleTable<DSPHandle, DSPCore*>`: forward（unordered_map）+ reverse（密配列 `reverseSlot_`）+ generation 検証 |
| `AudioEngine.h` | `runtimeDSPHandleMap_`（:4420）を `handleTable_` に置換。`eraseByHandle`（:4091-4098）の linear scan を O(1) に変更。`rollbackDSPHandleRegistration`（:4112）を `removeByHandle` に変更 |
| `ISRDSPHandle.h` | generation 発行を `allocateSlot()` に一元化（FUTURE-5 と共有） |
| テスト | `tests/HandleTableTests.cpp`（下記） |

### テスト計画（🔧 今回実装）

```cpp
// tests/HandleTableTests.cpp
TEST(HandleTableForwardO1) {
    // DSPCore* → DSPHandle の取得が O(1)（unordered_map 経由）
}
TEST(HandleTableReverseO1) {
    // DSPHandle → DSPCore* の逆引きが O(1)（linear scan なし）
    // eraseByHandle がスロット直接参照で動作
}
TEST(HandleTableGenerationGuard) {
    // スロット再利用後、旧 generation の Handle では erase が拒否される（HTABLE-3 ABA 防止）
}
TEST(HandleTableSlotIndexMapping) {
    // reverseSlot_ のインデックス == DSPHandle.slot（FUTURE-5 のプールインデックスと 1:1）
}
```

### 完了条件（🔧 今回実装）

1. `eraseByHandle` の linear scan（`AudioEngine.h:4091-4098`）が完全に削除され、O(1) になる
2. forward / reverse の双方向 O(1) が成立（HTABLE-1/2）
3. スロット再利用時に generation 検証で ABA が防止される（HTABLE-3）
4. 既存テスト全件通過 + 上記 HandleTable テスト合格

> **FUTURE-5/6 の統合関係**: FUTURE-6 の reverse 配列（`reverseSlot_[slot]`）は FUTURE-5 のプールスロットインデックスに直接対応する。**FUTURE-5 → FUTURE-6 の順に実装する**（逆順だとスロットインデックスの再設計が発生する）。両者で generation 発行を一元化するため、FUTURE-6 の実装は FUTURE-5 完了後とする。両者は **Storage Policy（ISR 本体と独立）**であり、実装順序は ISR 系（FUTURE-4 → FUTURE-3/7 → FUTURE-8/9 → FUTURE-10 → Shutdown Pipeline 検証）完了後に実施する（レビュー⑦反映・`#実装順序` 参照）。

---

## FUTURE-7: AudioEngine.Threading.cpp — emitQuarantineIntent → submitQuarantine 統合 [🔧 今回実装] — 📋 設計確定

### 目的

`AudioEngine::quarantineSlot()`（Threading.cpp:36-65）内の直接 `dspQuarantineManager_.quarantineHandle()` 呼び出しを `emitQuarantineIntent()` 経由に変更する。

### 現状

```cpp
// AudioEngine.Threading.cpp:36-65
bool AudioEngine::quarantineSlot(uint32_t slot, uint64_t generation,
                                  convo::isr::QuarantineReason reason) noexcept
{
    // Step 1: Truth store
    const bool applied = dspQuarantineManager_.quarantineHandle(slot, generation, reason);
    // Step 2-3: retire + Projection 更新
    // ...
}
```

### 変更後

```cpp
bool AudioEngine::quarantineSlot(uint32_t slot, uint64_t generation,
                                  convo::isr::QuarantineReason reason) noexcept
{
    const convo::isr::DSPHandle handle{slot, generation};
    // Coordinator 経由で quarantine を実行（QSVC-2）
    runtimePublicationBridge_.emitQuarantineIntent(
        handle, reason, dspHandleRuntime_, dspQuarantineManager_);

    // Step 2-3: retire + Projection 更新（現状維持）
    // ...
}
```

### トレードオフ

| 利点 | 欠点 |
|------|------|
| QSVC-2 完全遵守。全 quarantine が Coordinator 経由に。 | 追加の関数呼び出しオーバーヘッド。 |
| Authority 一元化が完全に達成。 | Threading.cpp:42 の直接呼び出しが無くなることで変更範囲が広い。 |

### 同期性分析

`quarantineSlot()` の現状は **同期実行**（`dspQuarantineManager_.quarantineHandle()` を直接呼び、即座に結果が返る）である。一方 `emitQuarantineIntent()` 経由に変更すると、Intent Queue → Coordinator → QuarantineService の経路を経由するため、**呼び出しから quarantine 確定までに遅延が発生する可能性がある**。

ただし、以下の理由で影響は限定的:

1. `quarantineSlot()` の呼び出し元（`AudioEngine.Commit.cpp:578,598`）は **NonRT（MessageThread）** である。RT パスからの呼び出しではない。
2. `emitQuarantineIntent()` 内の `QuarantineService::executeQuarantine()` は直ちに State + Audit を実行する（Intent Queue を経由しない）。したがって**同期性は維持される**。
3. `emitQuarantineIntent()` 自体が `DSPHandleRuntime::quarantine()` と `DSPQuarantineManager::quarantineHandle()` の両方を同期的に呼び出すため、`quarantineSlot()` が期待する「即時隔離」のセマンティクスは変わらない。

**結論**: `emitQuarantineIntent()` への置換は同期性を維持するため、安全に実施できる。

### 保留理由

~~設計書が「将来のリファクタリング候補」と明記。現在の直接呼び出しでも機能的正しさは維持されている。~~ **2026-07-31 決定により今回改修で実装する**。`submit` 語彙統一（下記）と Intent Queue 化を同時に実施し、QSVC-2 完全遵守を達成する。

### 命名に関する注意

`emitQuarantineIntent()` は現状**同期実行**（Intent Queue を経由せず、直ちに `QuarantineService::executeQuarantine()` を呼ぶ）である。ISR 的には「Intent 発行」という命名と「同期実行」という動作に乖離がある。ISR では Intent は「未来に処理される要求」を意味し、同期実行する関数は Intent ではない。

以下のいずれかに統一すべき:

- **`submitQuarantine()`**: **推奨 — 将来の Queue 化（非同期化）を見据えた最も ISR らしい命名**。ISR では `submit` は「要求の発行（未来に処理）」を意味し、`emitQuarantineIntent()` の現在の責務（Quarantine 要求の受け付け）と将来の非同期化の両方に整合する。
- **`executeQuarantine()`**: 現状の同期実行のみを正確に表す（同期実装のままで完結させる場合の代替案）
- ~~`emitQuarantineIntent()`~~: 命名と動作の乖離あり（ISR語彙として曖昧）

> **語彙体系の統一（2026-07-31 反映）**: レビュー指摘「Recovery は `submit` → Builder、Quarantine は `submit` → execute（同期）で語彙体系が崩れている」。ISR 語彙では **`submit` = enqueue（純粋な Intent 発行）**、**`execute` = 同期実行**である。したがって以下**①**を最終形とする:
>
> **①（採用）: `submitQuarantine()` = enqueue のみ** — `submitRecoveryRequest()` と完全に一致させる。Quarantine 実行は Intent Queue → Coordinator → QuarantineService の経路で行う。将来の共通 Intent Queue（FUTURE-10）で Observe / Recovery / Quarantine が同じ語彙になる。
>
> ②（不採用）: `executeQuarantine()` = 同期実行 — 現状の動作を正確に表すが、Future の非同期化を見据えると改名コストが2回（submit への再改名）になる。
>
> **実装方針**: `submitQuarantine()` は現状では「enqueue のみ」の責務を実装する（FUTURE-7 で Intent Queue 化と同時に実施）。現行の `emitQuarantineIntent()` 同期実行は、FUTURE-7 の Intent Queue 化までの**暫定実装**として位置づけ、`submitQuarantine()` への改名と同時に enqueue 化する。

> **submit 語彙の統一（2026-07-31 反映）**: `emit` は Signal / Notification 寄りの語彙であり、Intent を表す関数名としては `submit` が適切である。ISR の Intent 発行 API は**全て `submit` に統一**する:
>
> | 関数 | 語彙 | 意味 |
> |------|------|------|
> | `submitRecoveryRequest()` | `submit` = enqueue | Recovery Intent の発行（FUTURE-3） |
> | `submitQuarantine()` | `submit` = enqueue | Quarantine Intent の発行（FUTURE-7） |
> | `submitObserve()` | `submit` = enqueue | Observe Intent の発行（P0-4A の `emitObserveIntent` の改名先） |
>
> `emitObserveIntent()` → **`submitObserve()`** への改名も、FUTURE-7 の ISR 純化フェーズで併せて実施する。これにより `submitXxx()` が「Intent 発行」で統一され、`emit`（Signal）と `execute`（同期実行）が語彙体系から明確に分離される。

今回の改修では `emitQuarantineIntent()` のまま維持するが、次回の ISR 純化フェーズで **`submitQuarantine()` への改名**を行うことを推奨する。`submit` を第一候補とする理由: (1) ISR の Intent 語彙（`submit` = 要求発行、`process` = 処理実行）に適合する、(2) 将来の Intent Queue 化で関数シグネチャ変更なしで非同期化できる、(3) 同期性維持を「現状実装の一時的性質」として明確化できる。

> **今からの改名を推奨（2026-07-31 反映）**: レビュー指摘「将来 Queue 化する予定なら、今から `submitQuarantine()` へ改名しておく方が長期保守性が高い」。Queue 化予定がある以上、命名は実装より先に確定させておく方が、後の rename コスト（呼び出し箇所の一括変更）を回避できる。**推奨: 本改修の範囲で `emitQuarantineIntent()` → `submitQuarantine()` の rename を実施する**（FUTURE-7 実装時に併せて実施、または独立した rename コミット）。rename 時は呼び出し元（`AudioEngine.Commit.cpp:578,598` 等）とヘッダ宣言を更新し、動作変更は行わない。

### 実装ステップ（🔧 今回実装）

1. **rename コミット（動作変更なし）**: `emitQuarantineIntent()` → `submitQuarantine()`。呼び出し元 `AudioEngine.Threading.cpp:36-65`（`quarantineSlot()`）、`AudioEngine.Commit.cpp:578,598`、ヘッダ宣言を一括更新
2. **`submitQuarantine()` を enqueue 化**: 現行の同期 `executeQuarantine()` 呼び出しを Quarantine Intent Queue（`LockFreeRingBuffer<QuarantineRequest>`）への enqueue に変更。FUTURE-10 で共通 `intentQueue_`（`type=Quarantine`）へ統合するための単一 Queue 定義
3. **消費経路を Coordinator Loop に統合**: Coordinator の `processIntent` サイクル（`AudioEngine.Timer.cpp:1032`）で Quarantine 消費 → `QuarantineService::executeQuarantine()` 実行。**QSVC-2（全 quarantine が Coordinator 経由）を完全達成**
4. **`submitObserve()` への改名を同時実施**（`emitObserveIntent` → `submitObserve`）: submit 語彙統一の一環。`AudioEngine.Timer.cpp:1025` の呼び出しも更新
5. **同期性の検証**: `quarantineSlot()` の呼び出し元が NonRT（MessageThread）であることを確認済み（Commit.cpp:578,598）。enqueue 化後も quarantine 確定タイミングが変わらないことを統合テストで確認

### 変更ファイル（🔧 今回実装）

| ファイル | 変更内容 |
|---------|---------|
| `ISRRuntimePublicationCoordinator.h` | `submitQuarantine()` 宣言。`LockFreeRingBuffer<QuarantineRequest, kQuarantineQueueCapacity> quarantineQueue_` 追加。 |
| `ISRRuntimePublicationCoordinator.cpp` | `submitQuarantine()` 実装（enqueue のみ）。Coordinator Loop 内の Quarantine 消費。 |
| `AudioEngine.Threading.cpp` | `quarantineSlot()`（:36-65）を `submitQuarantine()` 経由に変更。 |
| `AudioEngine.Commit.cpp` | `emitQuarantineIntent()` 呼び出し（:578,597）を `submitQuarantine()` に変更。 |
| `AudioEngine.Timer.cpp` | `emitObserveIntent`（:1025）→ `submitObserve()`。`processIntent`（:1032）は Quarantine 消費を含むため維持。 |

### テスト計画（🔧 今回実装）

```cpp
// tests/QuarantineSubmitTests.cpp
TEST(SubmitQuarantineEnqueuesOnly) {
    // submitQuarantine() が enqueue のみ（同期実行しない）ことを検証
}
TEST(QuarantineProcessedByCoordinator) {
    // enqueue 後、Coordinator の processIntent サイクルで executeQuarantine が実行される（QSVC-2）
}
TEST(QuarantineSyncSemanticsPreserved) {
    // quarantineSlot() → submit → 次の Timer サイクルまでに確定（NonRT 呼び出し元で問題なし）
}
TEST(SubmitVocabularyUnified) {
    // submitRecoveryRequest / submitQuarantine / submitObserve が同一語彙で宣言されている
}
```

### 完了条件（🔧 今回実装）

1. `emitQuarantineIntent()` / `emitObserveIntent` がコードベースから消滅し、`submitQuarantine()` / `submitObserve()` に統一される
2. 全 quarantine が Coordinator 経由（QSVC-2 完全遵守）で実行される
3. `quarantineSlot()` の同期性セマンティクスが NonRT 呼び出し元で維持される
4. 既存テスト全件通過 + 上記 QuarantineSubmit テスト合格

---

## FUTURE-8: ObserveIntent 専用 Deferred Ring の分離 [🔧 今回実装] — 📋 設計確定

### 目的

P0-4A の Intent Queue Overflow Policy の Deferred 層で共用している `coordinatorDeferredRing_`（本来は `RetireOverflowEntry` 用）と、ObserveIntent の overflow 経路を責務分離する。ISR では Observe / Retire / Publish / Recovery は Intent 種別は同じでも Payload が異なるため、Deferred Ring を共用すると責務が曖昧になる。Observe 専用 Deferred Ring を用意する。

**P1 格上げ理由（2026-07-31 反映）**: レビュー指摘により、Deferred 層の共用は「Transport と Semantic の不一致」であり、ISR 的には責務混在に当たる。overflow 発生頻度は低いが、将来の共通 Intent Queue 化（FUTURE-10）に移行する際に、Retire 用 Ring と Observe 用 Ring が分離されていないと移行障壁となる。よって次フェーズ（P1）で実装する。**→ 2026-07-31 決定により今回改修で実装する（🔧 今回実装へ昇格）。**

### 設計方針

```
現状（Overflow 時）:
  observeIntentQueue_ (ObserveIntent)
    └── full → observeFallbackQueue_ → full → coordinatorDeferredRing_ (RetireOverflowEntry に変換・共用) ← 責務混在

将来:
  observeIntentQueue_ (ObserveIntent)
    └── full → observeFallbackQueue_ → full → observeDeferredRing_ (ObserveIntent 専用, LockFreeRingBuffer<ObserveIntent, 1024>)
```

**ObserveDeferredEntry 型**: Retire 系の `RetireOverflowEntry` に変換して投入する現状（`deferredEntry{}` 変換）を廃止し、**ObserveIntent をそのまま格納する**（Transport と Semantic の一致）。ObserveIntent はすでに trivially copyable + standard layout（`ISRRuntimePublicationCoordinator.h:306-309` の static_assert 済み）であり、LockFreeRingBuffer に直接格納可能。

> **実コード検証（2026-07-31）**: 現状の Deferred 変換（`ISRRuntimePublicationCoordinator.cpp:549-555`）は `deferredEntry.intent.dspSlot = 0` と**ObserveIntent の `handle` 情報を破棄**しており、drainOverflowRing で回収された際に正しい retire 対象を特定できない構造になっている（`RetireOverflowEntry` は `RetireIntent`（dspSlot ベース）を保持するため）。overflow 発生頻度は低いため実運用上の影響は限定的だが、**自己完結型 Intent の原則に反する**。FUTURE-8 で ObserveIntent をそのまま Deferred Ring に格納することで解消される。

### 契約

| ID | 契約 |
|----|------|
| QUEUE-15 | ObserveIntent の overflow は Observe 専用 Deferred Ring（`observeDeferredRing_`）へ流し、Retire 系の `coordinatorDeferredRing_` とは分離する。 |
| QUEUE-16 | Deferred Ring の回収（drain）は Coordinator Phase（`processIntent()`）内で行う。Observe 用と Retire 用で drain ルーチンを分離する。 |

### 完了条件

1. `coordinatorDeferredRing_` から ObserveIntent 由来のエントリが除去される
2. Observe overflow が Observe 専用 Ring で完結し、Retire 系データと混在しない
3. **ObserveIntent の `handle` が Deferred Ring を経由しても保持される**（`dspSlot = 0` 変換の廃止。回収時に正しい retire 対象を特定可能）
4. Overflow カウンタ（`overflowCounter_` / `fallbackOverflowCounter_`）の診断が種別別に分離される

### 実装ステップ（🔧 今回実装）

1. **`observeDeferredRing_` を追加**: `LockFreeRingBuffer<ObserveIntent, 1024> observeDeferredRing_`。ObserveIntent は trivially copyable + standard layout（`.h:306-309` の static_assert 済み）のため直接格納可能
2. **overflow 経路を分離**: `ISRRuntimePublicationCoordinator.cpp:549-555` の `deferredEntry{}` 変換（`intent.dspSlot = 0` による **handle 情報の破棄**）を廃止し、ObserveIntent をそのまま `observeDeferredRing_` に格納
3. **drain ルーチンを分離**: `processIntent()` 内で `observeDeferredRing_` の drain（Observe 用）と `coordinatorDeferredRing_` の drain（Retire 用）を別関数に分割（QUEUE-16）
4. **overflow カウンタを種別別に**: `overflowCounter_` / `fallbackOverflowCounter_` を Observe 用と Retire 用に分離
5. **回収時の retire 対象特定**: drain で回収した ObserveIntent の `handle` をそのまま `retireByHandle(intent.handle)` に渡す（`ISRRuntimePublicationCoordinator_ProcessIntent.cpp:17,25` と同じ経路）
6. **検証**: overflow 強制テスト（Queue 容量を小さくして発火）で handle 情報が保持されることを確認

### 変更ファイル（🔧 今回実装）

| ファイル | 変更内容 |
|---------|---------|
| `ISRRuntimePublicationCoordinator.h` | `observeDeferredRing_` メンバ追加。Observe / Retire 別の overflow カウンタ宣言。 |
| `ISRRuntimePublicationCoordinator.cpp` | `:549-555` の deferredEntry 変換を廃止し ObserveIntent 直接格納に変更。drain ルーチンを種別別に分離。 |
| `ISRRuntimePublicationCoordinator_ProcessIntent.cpp` | `observeDeferredRing_` の drain → `retireByHandle()` 経路を追加。 |
| テスト | `tests/DeferredRingTests.cpp`（下記） |

### テスト計画（🔧 今回実装）

```cpp
// tests/DeferredRingTests.cpp
TEST(ObserveOverflowGoesToOwnRing) {
    // observeIntentQueue_ full → observeFallbackQueue_ full → observeDeferredRing_（QUEUE-15）
    // coordinatorDeferredRing_ に ObserveIntent が混入しない
}
TEST(ObserveHandlePreservedThroughOverflow) {
    // overflow 経由で Deferred された ObserveIntent の handle が保持される
    // （dspSlot=0 変換の廃止後、drain で正しい retire 対象を特定できる）
}
TEST(DrainRoutinesSeparated) {
    // processIntent() 内で Observe drain と Retire drain が別関数で実行される（QUEUE-16）
}
TEST(OverflowCountersPerType) {
    // overflowCounter_ が Observe 用 / Retire 用に分離され、種別別に診断可能
}
```

### 完了条件（🔧 今回実装）

1. `coordinatorDeferredRing_` から ObserveIntent 由来のエントリが除去される
2. Observe overflow が Observe 専用 Ring で完結し、Retire 系データと混在しない
3. **ObserveIntent の `handle` が Deferred Ring を経由しても保持される**（`dspSlot = 0` 変換の廃止）
4. Overflow カウンタの診断が種別別に分離される
5. 既存テスト全件通過 + 上記 DeferredRing テスト合格

---

## FUTURE-9: Dedicated Coordinator Worker 移行 [🔧 今回実装] — 📋 設計確定

### 目的

P0-4A の「Timer callback 内で `processIntent()` を呼ぶ暫定実装」から、**専用の Coordinator Worker / Loop で `processIntent()` を実行する完全 ISR 形へ移行**する。これにより Scheduling Authority が Timer から Coordinator 自身に移り、Authority 分離 + Execution Context 分離が完成する。

**P1 格上げ理由（2026-07-31 反映）**: レビュー指摘「完了条件に Coordinator Worker 移行がなく、ISR 完成形としては Future でなく P1 程度に格上げすべき」。Scheduling Authority が Timer 側に残ったままでは ISR 準拠は約90%止まりであり、これは完了条件として追跡する。**→ 2026-07-31 決定により今回改修で実装する（🔧 今回実装へ昇格）。**

> **ISR 完成条件としての位置づけ（10回目レビュー④反映・2026-07-31）**: 10回目レビューは「Dedicated Coordinator Worker は P1 というより **ISR 完成条件**」と指摘。本設計もこれに同意し、FUTURE-9 を **P1 ではなく ISR 完成条件（Priority 0）** と位置づける。レビューが提示した優先順位の**第1位（Scheduling Authority の完全分離）**である。Timer に Scheduling Authority が残る限り ISR 準拠は 90% 止まりであり、FUTURE-9 は ISR 準拠 100% への到達条件である。

> **Scheduling Authority の原則（レビュー⑧確認・2026-07-31）**: ISR 完成形では **Scheduling Authority（いつ Intent を処理するかの決定権）は Coordinator 自身が保持する**べきである。現状の `Timer → emitObserveIntent() → processIntent()` は、同じ Timer callback 内で実行コンテキストを分離しているに過ぎず、**Phase 分離**（時系列的な分離）であって **Authority 分離**ではない。Phase 分離は暫定的な実装であり、Dedicated Coordinator Worker 移行で初めて Authority 分離が完成する。本設計はこの認識を明示し、FUTURE-9 を P1 格上げ（今回実装）で追跡する。

### 設計方針

```
現状（暫定）:
  Timer callback → emitObserveIntent() → processIntent()   ★ Scheduling Authority = Timer
                                                              （Phase 分離のみ = 時系列分離）
  レビュー⑧: これは Authority 分離ではなく、Timer が Scheduling Authority を保持したままの暫定形。

将来（完成形）:
  Timer: emitObserveIntent() のみ（即座復帰）
  Dedicated Coordinator Worker / MessageThread 定期タスク:
    while (true) { processIntent(); }  ★ Scheduling Authority = Coordinator（Authority 分離完成）
```

### 実装ステップ

1. Timer callback から `processIntent()` 呼び出しを削除
2. `CoordinatorLoop`（MessageThread の定期タスクまたは専用 Worker）を新設
3. `processIntent()` の呼び出し元を Loop に変更（ObserveIntent は自己完結型のため**インターフェース変更ゼロ** — `ISRRuntimePublicationCoordinator_ProcessIntent.cpp:17,25` の `lifetimeMgr.retireByHandle(intent.handle)` は既に外部状態に依存しない。ただし `processIntent()` 自体は世代逆転検出のため `persistentState_.publicationEpoch` を参照する（同 `:13,21`）が、呼び出し元の変更のみで完了する）
4. P0-4A 完了条件 1〜3 が引き続き成立することを確認

> **Coordinator は Intent Routing に専念（2026-07-31 反映・実コード検証済み）**: 実コードの委譲構造を検証した結果、Coordinator は retire/reclaim の実行詳細を一切保持しない。具体的には:
>
> | 経路 | 委譲先 | 実コード |
> |------|--------|---------|
> | Observe Intent 処理 | `DSPLifetimeManager::retireByHandle()` | `ISRRuntimePublicationCoordinator_ProcessIntent.cpp:17,25` |
> | Reclaim 要求 | `DSPHandleRuntime::retire()` → `ISRRetireRouter::currentEpoch()/minReaderEpoch()` で epoch 安全確認 → `DSPHandleRuntime::reclaim()` | `ISRRuntimePublicationCoordinator.cpp:568-599` |
>
> `requestReclaim()`（`.cpp:568-599`）は DELETE-2/3 に従い (1) `handleRuntime.retire()` 委譲 → (2) `router.currentEpoch()/minReaderEpoch()` で安全確認 → (3) `handleRuntime.reclaim()` 遷移のみを行い、物理削除（`DSPCore*` delete）は DSPLifetimeManager 経由で別途実行する。これはレビュー④の指摘「Coordinator は Intent Routing に専念し、Retire 詳細（epoch 安全確認含む）は専用コンポーネント（ISRRetireRouter / DSPLifetimeManager）へ委譲」を**既に満たしている**。P0-4A の「Coordinator は retire を実行」という記述は「Retire の Routing（Intent → Router 委譲）」を意味し、実行そのものを意味しない点を明記する。

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `AudioEngine.Timer.cpp` | Timer callback から `processIntent()` 呼び出しを削除。`emitObserveIntent()` のみに純化。 |
| `ISRRuntimePublicationCoordinator.h/.cpp` | `CoordinatorLoop` のための公開 API（`processIntent()` の定期実行インターフェース）を整備。 |
| 該当スレッド管理 | MessageThread への定期タスク登録、または専用 Worker の起動・停止。 |

### 完了条件

1. Timer callback が `emitObserveIntent()` のみを実行する（副作用なし）
2. `processIntent()` が Coordinator 自身の Scheduling Authority 下で定期実行される
3. 実行コンテキスト分離（Observer Phase / Coordinator Phase）がスレッドレベルで達成される
4. **Deferred Publish が再提出される**（`consumeDeferredRequest()` のデッドコード解消。`hasDeferred_` の atomic 化）。SHUTDOWN-2/4 の「deferred 完全消化」と整合（8回目レビュー後・調査確定）
5. **ISR 完成条件を満たす（10回目レビュー④反映）**: Scheduling Authority が Timer から Coordinator へ完全分離され、ISR 準拠が 100% に到達する（Scheduling Authority = Coordinator のみ）。この完了条件は **Priority 0（ISR 完成条件）**として追跡する

### 実装ステップ（🔧 今回実装）

1. **Timer callback の純化**: `AudioEngine.Timer.cpp:1029-1033` の `processIntent()` 呼び出しを削除し、`emitObserveIntent()`（:1025、FUTURE-7 で `submitObserve()` に改名）のみに純化
2. **CoordinatorLoop を新設**: MessageThread の定期タスクとして `processIntent()` を実行するループを追加。頻度は現在の Timer サイクルと同等（Audio 処理の区切りに合わせる）
3. **Scheduling Authority の移管**: Timer は「Observe 発行」のみの責務となり、Coordinator の処理スケジューリングは CoordinatorLoop に一元化（Authority 分離の完成）
4. **シャットダウン連携**: `isShutdownInProgress()`（Timer.cpp:1035）と同じ条件を CoordinatorLoop にも適用し、停止時に残存 Intent を安全に消化
5. **Deferred Publish 再提出経路の接続（8回目レビュー後・調査確定）**: 現状 `consumeDeferredRequest()`（`RuntimePublicationOrchestrator.h:65`）はデッドコードであり、`enqueueDeferred()` で保存された publish 要求が**再提出される経路が存在しない**。FUTURE-9 の CoordinatorLoop 実装時に、**Timer の `hasDeferredRequest()` 検知（`AudioEngine.Timer.cpp:1040`）→ `consumeDeferredRequest()` → `submitPublishRequest()` の再提出ループ**を接続する。`hasDeferred_` は `std::atomic<bool>` に変更し、データ競合（`doc/work89/BUG-052.md` 記載）を解消する。これにより SHUTDOWN-2/4（Drain Intent / Advance Epoch）の「deferred の完全消化」契約が成立する
6. **P0-4A 完了条件 1〜3 の回帰確認**: processIntent の移設後も retire/reclaim タイミングが変わらないことを統合テストで確認
7. **優先順位の検討**: CoordinatorLoop は FUTURE-10（共通 Intent Queue）実装後の Routing 主体になるため、**FUTURE-9 は FUTURE-10 の直前に実装**する（共通 Queue の消費主体として CoordinatorLoop が機能する）
8. **`processIntent()` の Routing 専用化（10回目レビュー②反映・2026-07-31 確認済み）**: Coordinator の `processIntent()` は **Routing のみ**を行い、各 Intent 種別の処理詳細は専用 Handler（`RetireHandler` / `RecoveryHandler` / `PublishHandler` / `ObserveHandler`）へ委譲する。これは QUEUE-22 の `kDispatchTable[type]` 登録方式で**既に設計確定済み**であり、FUTURE-10 実装で `switch(type)` は完全に排除される。レビューが提案する `Dispatcher → RetireHandler / RecoveryHandler / PublishHandler` の分割構造と等価である

### 変更ファイル（🔧 今回実装）

| ファイル | 変更内容 |
|---------|---------|
| `AudioEngine.Timer.cpp` | `:1029-1033` の `processIntent()` 呼び出しを削除。`emitObserveIntent()` のみに純化。deferred publish 再提出ループの接続。 |
| `ISRRuntimePublicationCoordinator.h/.cpp` | `CoordinatorLoop` のための公開 API（`processIntent()` の定期実行インターフェース）を整備。 |
| `RuntimePublicationOrchestrator.h/.cpp` | `hasDeferred_` を `std::atomic<bool>` 化（BUG-052 解消）。`consumeDeferredRequest()` の再提出経路接続。 |
| 該当スレッド管理（MessageThread / Worker） | 定期タスク登録、または専用 Worker の起動・停止。シャットダウン連携。 |

### テスト計画（🔧 今回実装）

```cpp
// tests/CoordinatorWorkerTests.cpp
TEST(TimerCallbackSideEffectFree) {
    // Timer callback が observe 発行のみを実行（processIntent 副作用なし）
}
TEST(CoordinatorLoopProcessesIntents) {
    // CoordinatorLoop が processIntent() を定期実行し Intent を消費する
}
TEST(CoordinatorLoopShutdownDrains) {
    // シャットダウン時に残存 Intent が安全に消化される
}
TEST(DeferredPublishResubmitted) {
    // enqueueDeferred された publish 要求が CoordinatorLoop 経由で再提出される（完了条件4）
    // hasDeferredRequest → consumeDeferredRequest → submitPublishRequest の経路を検証
}
TEST(HasDeferredAtomic) {
    // hasDeferred_ が std::atomic<bool> でデータ競合がない（BUG-052 解消）
}
TEST(RetireTimingPreserved) {
    // processIntent 移設後も retire/reclaim タイミングが従来と同等（P0-4A 回帰）
}
```

### 完了条件（🔧 今回実装）

1. Timer callback が `emitObserveIntent()`（→ `submitObserve()`）のみを実行する（副作用なし）
2. `processIntent()` が Coordinator 自身の Scheduling Authority 下で定期実行される
3. 実行コンテキスト分離（Observer Phase / Coordinator Phase）がスレッドレベルで達成される
4. シャットダウン時に残存 Intent が安全に消化される
5. 既存テスト全件通過 + 上記 CoordinatorWorker テスト合格

---

## FUTURE-10: 共通 Intent Queue 一本化 [🔧 今回実装] — 📋 設計確定（最終形）

### 目的

Observe / Recovery / Publish / Quarantine の各 Intent を**共通の `Intent` 型へ統合**し、種別別 Queue（`observeIntentQueue_` / Recovery Queue / Quarantine Request）を**単一の `LockFreeRingBuffer<Intent>`** に一本化する。Practical Stable ISR では Intent は「未来に処理される要求」であり、**Queue は一種類**であることが最も自然な姿である。**2026-07-31 決定により今回改修で実装する（🔧 今回実装へ昇格）。** FUTURE-3/7 の新規 Queue は共通 `intentQueue_` への統合を前提に定義し、FUTURE-8/9 はこの一本化の土台となる。

**一本化を前提とした設計（2026-07-31 反映）**: レビュー指摘「Queue 数が増えるほど Transport Authority が散らばる」。本設計は「将来の統合」ではなく**最初から一本化を前提**とする。新規の Intent（Recovery / Quarantine）を導入する際も、**種別別 Queue を新設せず共通 `Intent` 型 + `intentQueue_` 一本に追加する**。既存の `observeIntentQueue_` / `observeFallbackQueue_` / Deferred Ring は共通 Queue への移行対象として明示的に管理し、追加の種別別 Queue は作らない（例外: 実装上の過渡的容量分離のみ許容し、FUTURE-10 で必ず一本化する）。

> **ISR 観点の評価（10回目レビュー①反映・2026-07-31）**: 10回目レビューは「Intent Queue が型ごとに分かれている」点を現状 ★★★★☆ と評価しつつ、FUTURE-10 の共通 Intent Queue により **将来 ★★★★★ に到達**すると明言している。本設計はこの認識を共有する。**本改修（FUTURE-10 実装）完了時点で ISR 観点の「Intent Queue」項目が満点となる**ことを明記する。

### 設計方針

```cpp
// ISR 最終形: 共通 Intent 型（レビュー⑧反映: Payload は tagged-union variant）
enum class IntentType : std::uint8_t {
    Observe,
    Publish,
    Recovery,
    Quarantine
};

// 各種別の Payload（種別ごとに必要なデータのみ保持。型安全に増殖可能）
struct ObservePayload { DSPHandle handle; PublicationEpoch epoch; };
struct PublishPayload { DSPHandle handle; };
struct RecoveryPayload { DSPHandle quarantinedHandle; };
struct QuarantinePayload { DSPHandle handle; QuarantineReason reason; };

struct Intent {
    IntentType type;                 // Intent 種別（Routing キー）
    union {                          // ★ tagged-union variant（QUEUE-21）
        ObservePayload    observe;
        PublishPayload    publish;
        RecoveryPayload   recovery;
        QuarantinePayload quarantine;
    } payload;
    uint64_t sequenceId;             // 診断・モニタリング用
};
// static_assert: trivially copyable + standard layout（LockFreeRingBuffer 制約）
//   ★ std::variant は trivially copyable を保証しないため不可（LockFreeRingBuffer 要件）。
//     素の union + tag による tagged union で両立させる（QUEUE-21）。

// ★ FUTURE-10/HANDLER-2（2026-08-01 レビュー決定）: Execution dependencies を明示的に注入する HandlerContext。
//   Service Locator（AudioEngine 経由 getRuntime()/getQuarantine()）ではなく、Coordinator が構築し
//   Handler へ一方向注入する。Authority 境界を明示的に維持し、将来 Metrics/Logger/Diagnostics 追加で
//   processIntent シグネチャを変更せずに済む。Context は Runtime Snapshot ではなく Execution Context。
struct HandlerContext {
    AudioEngine& engine;
    DSPLifetimeManager& lifetime;
    DSPHandleRuntime& runtime;
    QuarantineService& quarantine;      // ★ A(2026-08-01): Handler→Service→Domain。DSPQuarantineManager は Service 背後の Domain。handleQuarantine は ctx.quarantine.execute(ctx.runtime, request) を呼ぶ。
};

// Queue: 単一
LockFreeRingBuffer<Intent, kIntentQueueCapacity> intentQueue_;
// ★ INTENT-1（12回目レビュー反映）: enqueue 後の Intent は const として扱われ、絶対に変更されない。
//    push は値コピー（tagged-union が trivially copyable のため安全）、pop は const 参照で Handler へ渡す。

// ★ 登録型 Dispatcher（QUEUE-22 / DISPATCH-1）: switch(type) を肥大化させない。Pure Routing（Decision なし）。
// ★ FUTURE-10/HANDLER-1/QUEUE-22: Handler = Execution のみ（World 書き換え / Decision 禁止）。
//   Coordinator は Routing のみ。dispatch() → Handler → 専用コンポーネントへ委譲。
void handleObserve(const Intent& intent, HandlerContext& ctx) noexcept;           // → ctx.lifetime.retireByHandle
void handlePublish(const Intent& intent, HandlerContext& ctx) noexcept;            // → ctx.engine 経由 commit/publishAtomic
void handleRecovery(const Intent& intent, HandlerContext& ctx) noexcept;            // → enqueue（Builder Loop が pop）
void handleQuarantine(const Intent& intent, HandlerContext& ctx) noexcept;          // → ctx.quarantine.execute（QuarantineService）
// ★ QUEUE-22 / DISPATCH-1 / HANDLER-2（2026-08-01 レビュー決定）: Pure Routing Dispatcher。switch(type) 肥太化を防ぐ。
//   IntentHandler は (const Intent&, HandlerContext&) を受け取る — Dependency は HandlerContext で明示的注入（Service Locator 回避）。
using IntentHandler = void (RuntimePublicationCoordinator::*)(const Intent&, HandlerContext&) noexcept;
static constexpr std::array<IntentHandler, kIntentTypeCount> kDispatchTable = {
    &RuntimePublicationCoordinator::handleObserve,     // Observe    → ObserveProcessor   （HANDLER-1: Execution のみ）
    &RuntimePublicationCoordinator::handlePublish,     // Publish    → PublishProcessor   （HANDLER-1: Execution のみ）
    &RuntimePublicationCoordinator::handleRecovery,    // Recovery   → RecoveryProcessor  （HANDLER-1: Execution のみ）
    &RuntimePublicationCoordinator::handleQuarantine,  // Quarantine → QuarantineProcessor
};
static_assert(kDispatchTable.size() == kIntentTypeCount, "QUEUE-22/DISPATCH-1: kDispatchTable must be a 1:1 total mapping over IntentType (Pure Routing, DispatcherHasNoDecision)");
// ★ DISPATCH-1 追加: type → Handler の 1:1 onto をコンパイル時検証（Dispatcher が Decision を持たないこと保証）。
// 新規 IntentType 追加時: enum 追加 + Payload 追加 + kDispatchTable に 1 行登録 のみ。
// switch の肥大化なし。FIFO（QUEUE-17）と Routing の責務を維持する。
```

> **Builder Scheduler 層（9回目レビュー④反映・将来拡張指針）**: Recovery / Publish / Preset / Automation の各 Build 要求が直接 Builder へ集中すると Builder Queue が肥大化する。ConvoPeq の規模では当面発生しないが、将来 Automation（BGM 自動切替等）が加わると **`Common Intent Queue → Builder Scheduler → Build Task`** の Scheduler 層が必要になる。Builder Scheduler は「どの Build 要求をどの優先度で次に build するか」を判断する **Policy コンポーネント**であり、Builder 本体（build 実行）と分離する。**今回改修では Builder Loop が Recovery 要求のみを扱うため Scheduler 層を導入しない（過剰設計回避）**。ただし共通 Intent Queue 化（FUTURE-10）で Build 要求が `intentQueue_` に入る設計（`PublishPayload`）にしておけば、将来 Scheduler 層を `kDispatchTable` の `handlePublish` 背後に挿入するだけで済む。

### 実装内容

1. 種別別 Queue を単一 `intentQueue_` に統合（`observeIntentQueue_` / `observeFallbackQueue_` / Recovery Queue / Quarantine Request を置換）
2. `processIntent()` を共通化 — `intent.type` で Routing し、各 Intent ハンドラへ委譲
3. 4層 Overflow Policy は種別に依存しない形に一般化（Primary → Fallback → Deferred → Drop）
4. FUTURE-8 の Observe 専用 Deferred Ring は共通 Intent 用 Deferred Ring に吸収

### 契約

| ID | 契約 |
|----|------|
| QUEUE-17 | Intent は `{ type, payload }` で統一する。Queue は `LockFreeRingBuffer<Intent>` 一本。 |
| QUEUE-18 | `processIntent()` は `intent.type` で Routing する。Coordinator は Intent Routing に専念し、各処理の詳細は専用コンポーネントへ委譲する。 |
| QUEUE-19 | 4層 Overflow Policy は種別非依存に一般化する。 |
| QUEUE-20 | **QoS は Transport とは分離した問題として扱う（レビュー⑥反映）**: Transport（Queue）は Authority ではない。したがって「FIFO 一本」と「Intent 種別の優先処理」は両立可能である。優先度が必要な場合（例: Publish が Observe に遅延される状況）、実装は **Queue 内の FIFO 順を保ったまま Coordinator の処理ループで優先 Intent を先に取り出す**（scan による priority 参照。Queue の順序は変えない）方式を採る。**本改修の基本は FIFO 一本（QUEUE-17）**とし、優先度方式は実際の遅延計測で必要性が確認された場合に限り追加する（過剰設計を避ける）。優先度を追加する場合も Transport の FIFO 保証は不変に保つ。 |
| QUEUE-21 | **Intent Payload は tagged-union variant で定義する（レビュー⑧反映）**: `Intent` のペイロードを種別別 struct の素の union で保持し、`type` タグで判別する。`std::variant` は trivially copyable を保証しないため LockFreeRingBuffer 制約を満たせない。**tagged union であれば trivially copyable + standard layout を維持**でき、新規 Intent 種別の追加は「enum 追加 + Payload struct 追加」のみで済む（Intent 本体のフィールド追加が不要になる）。 |
| QUEUE-22 | **Dispatcher は Handler 登録方式（dispatch table）で実装する（レビュー⑧⑨反映）**: `switch(type)` や `if` 連鎖の肥大化を防ぐため、`IntentType` ごとのハンドラ登録配列 `kDispatchTable[]` を用意し、`processIntent()` は `kDispatchTable[type]` へ委譲する。**各エントリは専用 Handler（`ObserveHandler` / `RecoveryHandler` / `PublishHandler` / `QuarantineHandler`）への委譲**であり、Coordinator 本体にハンドラ実装を置かない（Handler 登録方式・レビュー⑨反映）。新規 Intent 種別の追加は**テーブルへの登録 1 行 + 専用 Handler 実装**で完了し、`processIntent()` 本体の変更を伴わない。`static_assert(kDispatchTable.size() == kIntentTypeCount)` で登録漏れをコンパイル時に検出する。 |
| DISPATCH-1 | **Dispatcher は Routing のみ（Decision 禁止）（11回目レビュー③反映・12回目レビューで Pure Routing を絶対条件化）**: Dispatcher（`kDispatchTable[type]`）は Intent を **種別に応じて対応する Handler へ転送（Routing）するのみ**であり、**Decision（優先度付け・破棄・並べ替え・取捨選択）・Priority・Merge・Retry を一切持たない**。ISR の `Transport → Dispatch → Handler` の流れにおいて、Dispatch 自身は Decision を持たない（Decision は Handler 以降の各 Authority のみが行う）。Dispatcher が「どの Intent を先に処理するか」「どの Intent を無視するか」を判断することは禁止される（それは QoS 判断であり QUEUE-20/QUEUE-23 に委ねる）。**Dispatcher の Coordinator 化防止（12回目レビュー①反映）**: Handler が増加しても Dispatcher 内に `if` / `priority` / `retry` / `merge` が入り込まないよう、**Dispatcher = Pure Routing を絶対条件**とし、実装時に `static_assert(DispatcherHasNoDecision)`（例: `kDispatchTable[type]` が「type から Handler への一意写像」であること 1:1 を static_assert し、Dispatcher が状態を持たないこと・ループ内で Routing 以外の分岐を持たないことを検証）で保証する。Routing 以外のロジックを Dispatcher に置かない。 |
| HANDLER-1 | **各 Handler は Execution のみ（Policy / Authority を持たない）（12回目レビュー③・最終評価3・REPAIR_PLAN(34)提案①④反映）**: `ObserveHandler` / `RecoveryHandler` / `PublishHandler` / `QuarantineHandler` は **Executor（実行）のみ**を担当し、**Decision / Policy / Priority 判断を持たない**。Decision は **Builder / Validator / Policy 層のみ**が行う（ISR の「Handler = Executor、Decision = Builder / Validator / Policy」）。**副作用境界（REPAIR_PLAN(34) 提案④反映）**: Handler は **RuntimeWorld を書き換えない**。Handler が実行できるのは (1) 既存 RuntimeWorld への委譲（`retireByHandle` 等の既存 API 呼び出し）、(2) 新規 Intent の submit（`submitObserve()` 等）、(3) 診断情報の記録のみであり、(4) `publishAtomic(currentWorld_, ...)` による World 更新を直接実行しない（World 更新は RuntimeBuilder のみが行う）。これにより Authority が RuntimeBuilder へ限定される。Handler の追加時は「Decision を持たない」「World を書き換えない」ことをコードレビュー + 設計契約で検証する。 |
| INTENT-1 | **Intent は投入後不変（enqueue 後変更禁止）（REPAIR_PLAN(34) 提案③反映）**: 共通 Intent Queue に投入された `Intent` は**絶対に変更してはならない**。`enqueue` 以降、Intent は const として扱われ、Dispatcher / Handler / CoordinatorLoop のいずれも Intent の内容（`type` / `payload`）を書き換えない。**実装契約**: (1) `LockFreeRingBuffer<Intent>` は push 時に Intent を値コピーする（ポインタ共有・参照共有をしない）、(2) pop された Intent は const 参照で Handler へ渡す、(3) Handler が Intent を書き換えたい場合は**新規 Intent を submit する**（既存 Intent の変更ではなく新しい Intent の発行）、(4) tagged-union（QUEUE-21）が trivially copyable であるため値コピーは安全。これにより「Queue 内の Intent が途中で書き換えられる」ことが構造的に排除され、ISR の「Intent は未来に処理される要求」という意味論が保たれる。 |
| QUEUE-23 | **共通 Intent Queue は到着順 FIFO を維持し、優先制御は Dispatcher 以降でのみ行う（11回目レビュー②反映・12回目レビューで Coordinator Worker 側の Scheduling を強調）**: 共通 Intent Queue（`intentQueue_`）は **到着順 strict FIFO** を契約として固定する（QUEUE-17 の FIFO 保証を強化）。**Priority FIFO は採用しない**。Publish / Observe / Recovery / Quarantine は意味が異なる Intent だが、Queue は種別を問わず到着順で保持し、優先制御（もし将来必要になれば）は **Dispatcher 以降の段（Handler / CoordinatorLoop の処理ループ）でのみ**行う。これにより「Queue が Decision する」ことを構造的に排除する（QUEUE-20 の QoS 検討と整合）。実装契約: (1) `intentQueue_` は push 順に pop される（LockFreeRingBuffer の FIFO 保証をそのまま維持）、(2) Queue には種別優先度・期限・並べ替えロジックを持たない、(3) 優先度が必要な場合は CoordinatorLoop が「FIFO を保ったまま scan で優先 Intent を先に取り出す」方式（QUEUE-20）のみ許容する。**処理順序の決定権（12回目レビュー⑤反映）**: 大量 Observe による Publish 遅延等の課題は、Queue ではなく **Coordinator Worker（FUTURE-9）側が「どの Intent をどの順序で処理するか」を決定**することで解決する（ISR の `Queue = FIFO → Handler 側で Scheduling` 設計）。**Queue 自体の FIFO は常に保持され、処理順序の変更は消費側（CoordinatorLoop）でのみ行う**ことを強調する。これにより「Queue を並べ替えて優先処理する」誤実装を防ぐ。 |

> **QoS 検討メモ（レビュー⑥反映・2026-07-31）**: 単一 FIFO の懸念は「Observe が大量に詰まると Publish / Recovery が遅延する」こと。ただし ConvoPeq(35) では (1) Observe は self-contained で各 DSPHandle 1回、非定型の遅延は次サイクルで回収される、(2) **Publish は Intent Queue を経由しない直接 publish 経路を持つ（実コード検証済み）**: `AudioEngine.Commit.cpp:391` → `runtimePublicationBridge_.commit(...)` → `publishAtomic(currentWorld_, newWorld, release)`（`ISRRuntimePublicationCoordinator.cpp:107`）。Publish Intent が FIFO で遅延する構成には最初からなっていない、(3) 実際の overflow 頻度は低い（QUEUE-15 の Repair Scan で回復）。したがって**現時点で priority 化は不要**。ただし FUTURE-10 の共通 Queue 化後、Observe/Publish/Recovery/Quarantine の流量特性を計測し、Publish の遅延が観測された場合のみ QUEUE-20 の priority 方式を導入する。

### トレードオフ

| 利点 | 欠点 |
|------|------|
| Authority 単一化。FIFO 保証が容易。 | 種別ごとの容量調整ができない（最大種別に合わせる）。 |
| SPSC 前提の維持。RT-safe。 | 統合に伴うリファクタリングコスト。 |
| ISR 最終形として最も自然。 | 現行の種別別 Queue 実装からの移行作業。 |

### 2026-08-01 レビュー決定（Authority 境界固定 — 再変更不可）

FUTURE-10 は Authority 分離フェーズ。3 点を最終確定:

1. **Handler plumbing = 依存明示注入（HANDLER-2）**: `HandlerContext{engine,lifetime,runtime,quarantine}` を `processIntent(const Intent&, HandlerContext&)` へ受け渡す。`AudioEngine` は Facade とする — Service Locator 化しない。Coordinator が HandlerContext を構築し Handler へ一方向注入。呼び出し元は `AudioEngine.Timer.cpp:1032` **1箇所のみ**（影響域限小）。
2. **Overflow = Intent種別非依存統一**: `enqueue(intent)` → Primary → Fallback → Deferred → Drop を **1 実装**。Queue は `Intent` だけ知る（種別不問）。FUTURE-8 Observe 専用 Deferred Ring は `OverflowPolicy` 実装として吸収。
3. **Quarantine = Coordinator Routing のみ**: `submitQuarantine()` は Intent enqueue。`processIntent` → `handleQuarantine(ctx)` → `DSPQuarantineManager`（State: `DSPHandleRuntime::quarantine` / Audit: `quarantineManager.quarantineHandle`）。QSVC-2 は「Coordinator 経由到達」で満たす（Coordinator 自身は quarantine state を触らない）。**Authority 分離**: Coordinator=routing / Handler=execution / DSPQuarantineManager=quarantine-state。

#### 追加確定（2026-08-01 レビュー — A/B/C）

- **A（QuarantineService 維持）**: `HandlerContext.quarantine` は `DSPQuarantineManager&` ではなく **`QuarantineService&`**。`handleQuarantine(ctx)` は `ctx.quarantine.execute(ctx.runtime, request)` を呼び、Service が State（`DSPHandleRuntime::quarantine`）+ Audit（`DSPQuarantineManager::quarantineHandle`）を単一トランザクションで実行する（Handler → Service → Domain）。HandlerContext フィールドを `QuarantineService&` へ変更済み。
- **B（IntentOverflowPolicy 分離）**: `RetireOverflowScheduler`（Epoch-aware）と `IntentOverflowPolicy`（FIFO: Primary → Fallback → Deferred → Drop）を**別実装**とする。Transport(Queue)は統一しても Policy は Intent 種別ごとに異なる意味（Epoch vs FIFO）を持つため。
- **C（順序）**: FUTURE-9（Dedicated Coordinator Worker / Scheduling Authority 確立） → FUTURE-10（HandlerContext + kDispatchTable + 共通 Queue）。Worker が完成して初めて Handler群を最終形で実装できる。
- **ISR Authority 構造（完成形)**: `RT → Intent Queue（Transport Authority） → Coordinator Worker（Scheduling Authority） → kDispatchTable（Routing） → Handler（Execution） → Domain Service（QuarantineService） → DSP`。

#### 未実装インベントリ（棚卸し 2026-08-01 — 現コード 2026-08-01 時点）

| 項目 | 現コード | 設計 / レビュー決定 | ステータス |
|------|------|----------------|------|
| `kDispatchTable` / `handle*` 宣言 | `ISRRuntimePublicationCoordinator.h:168-196` に `IntentType`/`Intent`/`intentQueue_`/`nextIntentId_` (`.h:355`) 追加済（Phase A, build 63/64 ✅）。`handle*`/`kDispatchTable` は**ソース未実装**（REPAIR_PLAN 設計のみ） | QUEUE-22/HANDLER-1 | 未実装 |
| `processIntent` routing | Observeのみ (`ISRRuntimePublicationCoordinator_ProcessIntent.cpp:7-33`: `observeIntentQueue_`+`observeFallbackQueue_`+`drainObserveDeferred`)。Recovery/Quarantine/Publishは未経由 | FUTURE-10 統合 (`kDispatchTable[type](intent,ctx)`) | 未実装 |
| `submitQuarantine` | **同期** (`QuarantineService::executeQuarantine` 直接呼び, `ISRRuntimePublicationCoordinator.cpp:676`) | enqueue + `handleQuarantine` 委譲 | 未実装（決定済） |
| Intent overflow (4層) | 未実装。現 Overflow は Retire 専用 (`OverflowScheduler`, `coordinatorDeferredRing_`/`.h:310`, `lastResortQueue_`/`.h:313`, `OverflowDrainResult`/`.h:215`) | QUEUE-19 (種別非依存) | 未実装 |
| `QuarantineService` 配置 | `ISRRuntimePublicationCoordinator.cpp:610-635` 実装済（State `DSPHandleRuntime::quarantine`+Audit `DSPQuarantineManager::quarantineHandle`）。`HandlerContext.quarantine = QuarantineService&` なので `handleQuarantine` へ**inlineまたはService背後** | HANDLER-2 | 解決済（2026-08-01 A 確定: keep QuarantineService as Handler backend。Handler→Service→Domain。handleQuarantine(ctx) → ctx.quarantine.execute(ctx.runtime, request)） |
| 旧 Queue 同居 | `observeIntentQueue_`/`.h:331`, `observeFallbackQueue_`/`.h:334`(`kObserveFallbackCapacity`)`, `observeDeferredRing_`/`.h:345`, `recoveryIntentQueue_`/`.h:349`(kRecoveryIntentQueueCapacity256) 仍存在 | FUTURE-10 で `intentQueue_`(`.h:353`,4096)へ統合置換 | 未実装 |

#### スタブ参照の不一致（棚卸し済み 2026-08-01）

- REPAIR_PLAN 内の行参照 `.h:311`/`.h:315`/`.cpp:527-531` 等は **stale**（FUTURE-7/8 の追記で行がずれ）。現コード: `kObserveIntentQueueCapacity` `.h:330`, `observeFallbackQueue_`/`.h:334`, `observeDeferredRing_`/`.h:345`, `kIntentQueueCapacity`/`.h:353`, `nextIntentId_`/`.h:355`。
- doc:852-908/1244/1254/1256 の `emitQuarantineIntent()`/`emitObserveIntent()` は FUTURE-7 で `submitQuarantine()`/`submitObserve()` に改名済み（現コード `AudioEngine.Timer.cpp:1809,1847` は `submitQuarantine`）。
- `submitQuarantine` 呼び出し: `AudioEngine.Timer.cpp:1809,1847`（両点で `dspQuarantineManager_` を既に渡しているため、HandlerContext plumbing は既存引数の再利用で済む）。

### 完了条件

1. 全 Intent（Observe / Publish / Recovery / Quarantine）が単一 `intentQueue_` を経由する

2. `processIntent()` が `intent.type` で Routing し、各ハンドラへ委譲する
3. 4層 Overflow が種別非依存で動作する
4. 新規 Intent 種別の追加が種別別 Queue の新設を伴わない（共通 Queue への追加のみ）
5. 優先度（QUEUE-20）は FIFO を維持したまま追加可能である（必要時のみ。計測で必要性が確認されるまで実装しない）
6. **Intent が tagged-union variant（QUEUE-21）かつ登録型 Dispatcher（QUEUE-22）で実装されている**（`std::variant` 不使用・`processIntent()` に switch なし）
7. **Dispatcher が Pure Routing である（DISPATCH-1）**: `static_assert(DispatcherHasNoDecision)` により type → Handler の一意写像（1:1）と Dispatcher の無状態性がコンパイル時に検証される
8. **各 Handler が Execution のみ（HANDLER-1）**: Handler が Decision / Policy / World 書き換えを持たないことをコードレビューで確認（retire は既存 API 委譲・World 更新なし）
9. **Intent が投入後不変（INTENT-1）**: Queue 投入後の Intent が const で扱われることをテストで検証（pop 後の Intent を書き換えるコードが存在しない）

### 実装ステップ（🔧 今回実装）

1. **`Intent` 型を定義（tagged-union variant・レビュー⑧反映）**: `enum class IntentType`（Observe / Publish / Recovery / Quarantine）+ 種別別 `Payload` struct + `union { observe; publish; recovery; quarantine; }` を持つ `struct Intent`（QUEUE-21）。trivially copyable + standard layout を static_assert（`std::variant` は不可）
2. **`intentQueue_` に一本化**: `observeIntentQueue_` / `observeFallbackQueue_` / Recovery Queue / Quarantine Request を単一 `LockFreeRingBuffer<Intent, kIntentQueueCapacity>` に置換。**実コード確認（2026-07-31）**: 現行は Primary 1024（`kObserveIntentQueueCapacity`, `.h:311`）+ Fallback 2048（`kObserveFallbackCapacity`, `.h:315`）の2層。`kIntentQueueCapacity` は現行合計（約3072）を基準に設定し、overflow 頻度が低いことを確認済み（`emitObserveIntent` の overflow 経路は `.cpp:527-531`）
3. **`processIntent()` を共通化 + 登録型 Dispatcher（レビュー⑦⑧反映）**: `processIntent()` は **`kDispatchTable[intent.type]` への委譲のみ**を行い、各 Intent Processor へ分岐する:
   ```
   processIntent()
     └── pop → (this->*kDispatchTable[intent.type])(intent)   ← テーブル参照のみ（switch なし）
           ├── handleObserve    → ObserveProcessor    (retireByHandle 委譲)
           ├── handlePublish    → PublishProcessor    (commit/publishAtomic 委譲)
           ├── handleRecovery   → RecoveryProcessor   (Builder Loop 経由)
           └── handleQuarantine → QuarantineProcessor (executeQuarantine 委譲)
   ```
   これにより Intent 種別が増えても `processIntent()` 自体は肥大化せず、**新規種別の追加は enum + Payload + テーブル登録 1 行**で完了する（QUEUE-22）。実コード現状: `ISRRuntimePublicationCoordinator_ProcessIntent.cpp`（34行）は Observe 2層（Primary + Fallback）の pop → `retireByHandle` のみ。共通 Intent 化時に dispatch table 方式へ再構成する（QUEUE-18）
4. **4層 Overflow を一般化**: Primary → Fallback → Deferred → Drop を種別非依存に（QUEUE-19）。FUTURE-8 の Observe 専用 Deferred Ring は共通 Intent 用 Deferred Ring に吸収
5. **FUTURE-3/7 との統合**: `submitRecoveryRequest()` / `submitQuarantine()` は `intentQueue_.enqueue(Intent{Recovery/Quarantine, ...})` に直結（FUTURE-3/7 実装時にこの形で作成）
6. **容量検証**: 全 Intent 種別のピーク流量を合算し、`kIntentQueueCapacity` の妥当性を負荷テストで確認

### 変更ファイル（🔧 今回実装）

| ファイル | 変更内容 |
|---------|---------|
| `ISRRuntimePublicationCoordinator.h` | `Intent` / `IntentType` / 種別別 Payload（tagged union）定義。`intentQueue_` 追加。種別別 Queue メンバを置換。`kDispatchTable` 定義。ハンドラ宣言（`handleObserve` 等）。 |
| `ISRRuntimePublicationCoordinator.cpp` | `processIntent()` の dispatch table 委譲。enqueue 経路の一本化。 |
| `ISRRuntimePublicationCoordinator_ProcessIntent.cpp` | `handleObserve` / `handlePublish` / `handleRecovery` / `handleQuarantine` 実装（`kDispatchTable` 登録）。 |
| `AudioEngine.Timer.cpp` | `submitObserve()` が `intentQueue_` に enqueue（FUTURE-7/9 と整合）。 |
| FUTURE-3/7 の新規 Queue | 個別 Queue 定義を破棄し、共通 `intentQueue_` の `type=Recovery/Quarantine` に直結。 |
| テスト | `tests/CommonIntentQueueTests.cpp`（下記） |

### テスト計画（🔧 今回実装）

```cpp
// tests/CommonIntentQueueTests.cpp
TEST(SingleQueueAllIntentTypes) {
    // Observe / Publish / Recovery / Quarantine が単一 intentQueue_ を経由（QUEUE-17）
    // 種別別 Queue メンバが残存しないことをコンパイル時/実行時で検証
}
TEST(RoutingByType) {
    // processIntent() が intent.type で正しいハンドラへ委譲（QUEUE-18）
}
TEST(TaggedUnionTriviallyCopyable) {
    // Intent が trivially copyable + standard layout（static_assert）（QUEUE-21）
    // std::variant を使わず素の union + tag であること
}
TEST(PayloadVariantTypeSafety) {
    // type==Recovery のとき recovery ペイロードのみ有効にアクセスできる（QUEUE-21）
    // type と payload の不整合（例: type==Observe で quarantine にアクセス）を診断検出
}
TEST(DispatchTableRegistration) {
    // kDispatchTable が IntentTypeCount と 1:1 対応（static_assert で保証）（QUEUE-22）
    // processIntent() に switch 文が存在しないこと
}
TEST(NewIntentTypeNoQueueNoSwitch) {
    // 新規 IntentType 追加が (1) Queue 新設 (2) switch 肥大化 (3) processIntent 本体変更
    // を伴わない（完了条件4・QUEUE-22）
}
TEST(OverflowTypeAgnostic) {
    // 4層 Overflow が種別に依存せず動作（QUEUE-19）
}
TEST(NewIntentTypeNoNewQueue) {
    // 新規 IntentType 追加が Queue 新設を伴わない（完了条件4）
}
TEST(FUTURE10MigrationOrder) {
    // FUTURE-3/7 の submit 関数が intentQueue_ に直結している
}
```

### 完了条件（🔧 今回実装）

1. 全 Intent（Observe / Publish / Recovery / Quarantine）が単一 `intentQueue_` を経由する
2. `processIntent()` が `intent.type` で Routing し、各ハンドラへ委譲する
3. 4層 Overflow が種別非依存で動作する
4. 新規 Intent 種別の追加が種別別 Queue の新設を伴わない（共通 Queue への追加のみ）
5. **新規 Intent 種別の追加が `switch` 肥大化と `processIntent()` 本体変更を伴わない（QUEUE-21/22）**
6. 既存テスト全件通過 + 上記 CommonIntentQueue テスト合格

> **実装順序上の位置づけ**: FUTURE-10 は全 Intent の最終統合点であり、**FUTURE-3/7（新規 Intent 導入）→ FUTURE-8（Deferred Ring 分離）→ FUTURE-9（Coordinator Loop）→ FUTURE-10（一本化）** の順で実装する。新規 Intent は最初から共通 `Intent` 型で定義することで、移行コストを最小化する。

---

## Shutdown Pipeline 契約（SHUTDOWN-1〜7・2026-07-31 新設） — レビュー⑦反映

### 目的

Shutdown は「Publish を止めてから Intent Queue を空にし、最終的に何も残らない」ことを**契約として固定**する。これまでの設計は各 Phase の遷移が実装主導で、**「意図的に何を止めて、何を空にしているのか」が外から見えない**という弱点があった（REPAIR_PLAN(26) レビュー Shutdown 8.5/10）。Shutdown を上から下への一方向パイプラインとして契約化し、各ステップの完了条件を明示する。

### 現行実装との対応（実コード検証済み・2026-07-31）

Shutdown FSM は**実コードに既に存在**する。本契約はこれを「規格」として文書化し、実装との整合を確認するものである:

| 契約ステップ | 実コード対応 |
|---|---|
| ① Stop Publish | `ShutdownPhase::AudioStopped`（`ISRShutdown.h:28`）。`AudioEngine.Processing.ReleaseResources.cpp:74` で `transitionTo(ShutdownPhase::AudioStopped)` |
| ② Drain Intent | `ShutdownPhase::ObserverDrained`（`ISRShutdown.h:29`）。`ReleaseResources.cpp:187` で遷移 |
| ③ Drain Retire | `ShutdownPhase::RetireClosed`（`ISRShutdown.h:30`）。`ReleaseResources.cpp:192` で遷移 |
| ④ Advance Epoch | `ShutdownPhase::EpochSettled`（`ISRShutdown.h:31`）。`ReleaseResources.cpp:193` で遷移 |
| ⑤ Reclaim | `ShutdownPhase::ReclaimComplete`（`ISRShutdown.h:32`）。`ReleaseResources.cpp:304` で遷移 |
| ⑥ Verify Empty | `ShutdownPhase::VerifyDrained`（`ISRShutdown.h:37`）。`ReleaseResources.cpp:393` で遷移 |

- `ShutdownResult`（`ISRShutdown.h:135-144`）: `finalPhase` / `blockingReason` / `transitionViolations` / `lateCallbackCount` / `postStopEnqueueCount` を構造化して記録する。
- `ShutdownBlockingReason`（`ISRShutdown.h:46-57`）: `PendingPublication` / `PendingRetire` / `ActiveCrossfade` / `DeferredPublish` / `QuarantineResident` / `RouterPendingRetire` / `ReaderActive` / `Unknown`。各理由を `BlockingReasonStats` で個別計測（`ISRShutdown.h:66-70`）。
- `finalizeShutdown(timedOut)`（`ReleaseResources.cpp:467`）: 最終確定処理。
- `markLateCallback()` / `markPostStopEnqueue()`（`ISRShutdown.h:200-201`）: 契約違反（停止後の遅延コールバック / 停止後の enqueue）を記録。

### 契約

| ID | 契約 |
|----|------|
| SHUTDOWN-1 | **Stop Publish**: オーディオ停止後、新規 Publish を受け付けない（`ShutdownPhase::AudioStopped`）。この後 `markPostStopEnqueue()` が記録する post-stop enqueue が発生しないことを Verify で監査する。 |
| SHUTDOWN-2 | **Drain Intent（9回目レビュー⑥で全 Queue を列挙）**: **Shutdown 時に排出する全 Queue を以下のとおり明確に列挙し、空にする**（`ShutdownPhase::ObserverDrained`）:
  - **Intent Queue**（`observeIntentQueue_` / 共通 Intent Queue）: ObserveIntent の未処理分を全て消化
  - **Deferred Queue / Deferred Ring**（`coordinatorDeferredRing_` / `observeDeferredRing_`）: Overflow により deferred された Intent を全て再発行・消化
  - **Recovery Queue**（FUTURE-3）: Recovery Request を全て消化
  - **Fallback Queue**（`observeFallbackQueue_` / Retire `fallbackQueue_`）: Fallback 済み Intent を全て消化
  - **Deferred Publish Slot**（`RuntimePublicationOrchestrator` の `enqueueDeferred()` で保存された publish 要求）: **FUTURE-9 の再提出経路接続後に `consumeDeferredRequest()` → `submitPublishRequest()` で必ず消化する**（8回目調査で確定したデッドコード解消と整合）
  **監査実装**: `RuntimeDrainAudit`（`RuntimeDrainAudit.h:26-95`）が `pendingPublication` / `pendingRetire` / `deferredPublish` / `quarantineResident` / `overflowRingResident` / `blockingReasons` として全 Queue の残量を監査する。**停止後の enqueue は P0 契約違反**（`postStopEnqueueCount` で検出）。 |
| SHUTDOWN-3 | **Drain Retire（9回目レビュー⑥で Retire 系 Queue を列挙）**: Retire Router の Pending Retire を全て処理する（`ShutdownPhase::RetireClosed`）。排出対象は **Retire Router**（`m_retireRouter->pendingRetireCount()`）/ **Fallback Queue**（`fallbackQueueDepth_`）/ **Overflow Ring**（`getOverflowRing()` の滞留分）であり、`RuntimeDrainAudit` の `routerPendingRetire` / `fallbackQueueDepth_` / `overflowRingResident` で監査する。`RouterPendingRetire` 残留はブロッキング理由として記録（`BlockingReasonStats`）。 |
| SHUTDOWN-4 | **Advance Epoch**: 最終 Epoch を確定し、`EpochSettled` にする。Active Crossfade / Deferred Publish があれば先に完結させる（`ActiveCrossfade` / `DeferredPublish` はブロッキング理由）。 |
| SHUTDOWN-5 | **Reclaim**: リソース（DSPHandle / 参照）を回収する（`ShutdownPhase::ReclaimComplete`）。`QuarantineResident` 残留はブロッキング理由。 |
| SHUTDOWN-6 | **Verify Empty**: **全 Queue / Router / World が空であることを最終監査**する（`ShutdownPhase::VerifyDrained`）。残留がなければ `ShutdownComplete`、あれば `TimedOut` / `Failed` と `ShutdownResult.blockingReason` で報告。`transitionViolations` / `lateCallbackCount` / `postStopEnqueueCount` が全て 0 であることを確認する。**実コードギャップ（9回目レビュー⑥で検出・2026-07-31）**: `RuntimeDrainAudit::isAllZero()`（`RuntimeDrainAudit.h:77-83`）は `pendingPublication / pendingRetire / activeCrossfadeCount / deferredPublish / routerPendingRetire` の5項目のみ監査しており、**`overflowRingResident`（Overflow Ring 滞留数・SHUTDOWN-3 の排出対象）が監査に含まれていない**。コメント（同 `:50`「OverflowRing 滞留数（Drain 完了判定用）」）とは矛盾する。→ **実装時に `isAllZero()` に `overflowRingResident == 0` を追加する**（SHUTDOWN-3 の Overflow Ring 排出と整合）。`getPrimaryBlockingReason()` にも Overflow Ring 残留をブロッキング理由として追加するかは、Overflow Ring が排出可能な設計（FUTURE-9 の再提出経路）実装後に判断する。 |
| SHUTDOWN-7 | **No Active Builder（11回目レビュー⑤反映）**: Shutdown 完了条件に **Builder の進行中ジョブが存在しないこと**を追加する。Queue 空（SHUTDOWN-2/3）・Epoch 完了（SHUTDOWN-4）だけでは、**Builder が Build 中の RuntimeWorld が途中で残る**可能性があり、その状態は `ShutdownComplete` ではない。**契約**: (1) `ShutdownComplete` は「Queue 空 + Epoch 完了 + Reclaim 完了 + **Builder 非実行中**」をすべて満たした場合のみ遷移する、(2) Builder が Build 中（Recovery / Publish の build 進行中）ならば `ShutdownBlockingReason::ActiveBuilder`（新規）として記録し、`TimedOut` で報告する、(3) Build の完了待ちは有限時間内に限り許容し、タイムアウト時は残 Build を破棄して `TimedOut` とする。**実装対応**: Shutdown Phase 遷移（`ReleaseResources.cpp`）に Builder 実行状態の確認を組み込む。**実コード調査結果（2026-07-31）**: Builder は非同期 rebuild ワーカーであり、状態確認は `rebuildThreadShouldExit`（`AudioEngine.h:2502`）+ `rebuildWorkerRunning`（`ISRRuntimeSemanticSchema.h:324`）で行える。`releaseResources()`（`AudioEngine.Processing.ReleaseResources.cpp:185-187` の StopWorkers → ObserverDrained 遷移）がワーカー停止を先に実行する既存順序に合わせ、**`VerifyDrained` 遷移時（同 `:393`）に `rebuildWorkerRunning == false` を確認**する。`ShutdownBlockingReason`（`ISRShutdown.h:46-57`）には `ActiveBuilder` が未追加のため、実装時に `ReaderActive` の後ろへ追加する。Builder は BUILDER-STATE 契約により Build Session 単位で Stateless であり、Publish 完了と同時に PendingMap を破棄するため、Builder 非実行中の判定は「進行中の Build Session なし」と等価である。 |

### 完了条件

1. 7 契約（SHUTDOWN-1〜7）が実コードの Phase 遷移（`ReleaseResources.cpp`）と 1:1 に対応している
2. `ShutdownResult` が全 Phase の完了/失敗を構造化して記録する
3. Verify Empty で残留が検出された場合、`ShutdownBlockingReason` で原因が特定可能（**SHUTDOWN-7 の `ActiveBuilder` を含む**）
4. 既存 Shutdown テスト（`tests/`）全件通過
5. **全 Queue 排出契約が `RuntimeDrainAudit` と整合する（9回目レビュー⑥反映）**: SHUTDOWN-2/3 の列挙（Intent / Deferred / Recovery / Fallback / Overflow Ring / Deferred Publish）が `RuntimeDrainAudit` の監査項目と対応する。`isAllZero()` の `overflowRingResident` 追加（SHUTDOWN-6 記載の実コードギャップ解消）が実装されている
6. **Builder 非実行中の監査（11回目レビュー⑤反映）**: SHUTDOWN-7 の通り、ShutdownComplete 判定に「Builder 進行中ジョブなし」が含まれる（`isBuilderIdle()` 相当の確認 + `BlockingReason::ActiveBuilder` 記録）

---

## 実装順序（FUTURE-3〜10 依存関係ベース・2026-07-31 確定）

全 FUTURE を今回改修で実装する。依存関係とリスク最小化を考慮した実装順序は以下の通り。

| 順序 | 項目 | 依存 | 根拠 |
|------|------|------|------|
| 1 | **FUTURE-4**（Metadata Snapshot） | なし | 他項目と独立。実コード箇所が明確（7箇所）でリスク最小。先に片付ける。 |
| 2 | **FUTURE-3**（submitRecoveryRequest） | —（新規 Intent 導入） | 共通 `Intent` 型で最初から定義し、FUTURE-10 の移行コストを最小化。 |
| 3 | **FUTURE-7**（submitQuarantine / submitObserve） | —（新規 Intent 導入 + rename） | submit 語彙統一を早期に確定。FUTURE-3 と同一パターン。 |
| 4 | **FUTURE-8**（Observe Deferred Ring 分離） | FUTURE-10 の前提 | Deferred Ring を種別別に分離しておかないと共通 Queue 移行が困難。 |
| 5 | **FUTURE-9**（Coordinator Loop） | FUTURE-10 の前提 | 共通 Queue の消費主体として CoordinatorLoop が機能する。FUTURE-10 の直前に実装。 |
| 6 | **FUTURE-10**（共通 Intent Queue 一本化） | FUTURE-3/7/8/9 全て | 全 Intent の最終統合点。土台が完成した後に実行。 |
| 7 | **Shutdown Pipeline 検証**（SHUTDOWN-1〜7 整合確認） | FUTURE-10 完了後 | 共通 Intent Queue 化で Drain Intent の対象が一本化された後に、SHUTDOWN 契約との 1:1 対応を検証する（SHUTDOWN-7 の No Active Builder 確認を含む）。 |
| 8 | **FUTURE-5**（MemoryPool） | FUTURE-6 の前提 | スロットインデックスを確定させ、FUTURE-6 の reverse 配列と整合させる。**Storage Policy（ISR 本体と独立）のため ISR 系完了後に実施**（レビュー⑦反映）。 |
| 9 | **FUTURE-6**（Handle Table） | FUTURE-5 完了後 | reverse 配列がプールスロットインデックスに 1:1 対応する。逆順だと再設計が発生。**ISR 完成後の中長期的性能改善（レビュー⑦反映）**。 |

**方針**:
- **ISR 完成系を優先**: 独立・低リスク（FUTURE-4）→ 新規 Intent（FUTURE-3/7）→ 分離/移行（FUTURE-8/9）→ 統合（FUTURE-10）→ Shutdown Pipeline 検証 → Storage Policy（FUTURE-5/6）の順（レビュー⑦反映）。ISR 本体と独立な FUTURE-5/6 は ISR 系完了後に実施する。
- 各項目は単独コミットで実装し、実装ごとに当該テスト + 既存テスト全件を実行する。
- FUTURE-3/7 の新規 Queue は最初から共通 `Intent` 型で定義する（FUTURE-10 の移行を前提とした設計）。
- 各項目の完了条件は各セクション内に記載済み。全項目完了時点で `git diff --check` + 全テストを最終確認する。

> **10回目レビュー優先順位との対応（2026-07-31 反映）**: 10回目レビュー最終評価の優先順位は以下のとおり本実装順序と整合する:
>
> | レビュー優先順位 | 対応する実装項目 | 状態 |
> |---|---|---|
> | 1. Dedicated Coordinator Worker 移行（Scheduling Authority 完全分離） | **FUTURE-9**（順序5） | 🔧 今回実装・**ISR 完成条件**として完了条件5を追加 |
> | 2. 共通 Intent Queue + Dispatcher 一本化 | **FUTURE-10**（順序6） | 🔧 今回実装・QUEUE-17/18/21/22 |
> | 3. 過渡的 Metadata Cache 完全撤去 | **FUTURE-4 完了条件6**（順序1） | `rg currentPublicationEpoch_` 0件で物理撤去を確認 |
> | 4. `processIntent()` の Routing 専用化 | **FUTURE-10**（QUEUE-18 Routing + QUEUE-22 Dispatcher） | 既に設計確定。`kDispatchTable[type]` で switch 排除 |

> レビュー②（`processIntent()` の `RetireHandler` / `RecoveryHandler` / `PublishHandler` 分割）は QUEUE-22 の Handler 登録方式で対応済みである（`kDispatchTable` の各エントリが専用 Handler クラスへ委譲）。

> **11回目レビュー改善提案との対応（2026-07-31 反映）**: 11回目レビューが提案した4契約は以下の実装項目に反映済み:
>
> | 提案 | 反映先 | 内容 |
> |---|---|---|
> | BUILDER-STATE 契約 | **RECOVERY-7 直後** | PendingMap は Build Session 限定。Publish 完了後に必ず破棄。 |
> | DISPATCH-1 契約 | **FUTURE-10 契約表** | Dispatcher は Routing のみ。Decision 禁止。 |
> | QUEUE-23 契約 | **FUTURE-10 契約表** | 共通 Intent Queue は到着順 strict FIFO。優先制御は Dispatcher 以降のみ。 |
> | SHUTDOWN-7 契約 | **Shutdown Pipeline 契約表** | No Active Builder を Shutdown 完了条件へ追加。`ActiveBuilder` ブロッキング理由を新設。 |

> **12回目レビュー改善提案との対応（2026-07-31 反映）**: REPAIR_PLAN(32) 2nd（①②③④）+ REPAIR_PLAN(34)（HANDLER-1 / RAII / INTENT-1 / 副作用境界）の指摘は以下の実装項目に反映済み:
>
> | 提案 | 反映先 | 内容 |
> |---|---|---|
> | ① Dispatcher の Coordinator 化防止（Pure Routing 絶対条件） | **DISPATCH-1 契約 + FUTURE-10 完了条件7** | `static_assert(DispatcherHasNoDecision)` で一意写像（1:1）・無状態性をコンパイル時検証。Decision / Priority / Merge 禁止。 |
> | ② Builder Session 終了保証（RAII） | **BUILDER-STATE 契約 + FUTURE-3 実装ステップ4・テスト・完了条件4** | `BuildSession` RAII Guard で全終了経路（成功/失敗/例外/キャンセル/Admission Reject）の PendingMap 破棄を保証。`BuildSessionRAIIExceptionSafety` / `BuildSessionRAIICancelAndReject` テスト追加。 |
> | ③ Handler の Execution 専一化 | **HANDLER-1 契約（FUTURE-10 契約表）** | Handler = Executor のみ。Decision / Policy / World 書き換え禁止。既存 API 委譲・新規 Intent submit・診断記録のみ。 |
> | ④ Repair Scan の Observation 限定 | **MAINTENANCE-1 契約** | 終端は必ず `submitObserve()`。`retire()` / `delete()` / World 更新を直接実行しない。 |
> | ⑤ Intent の投入後不変性（REPAIR_PLAN(34) ③） | **INTENT-1 契約（FUTURE-10 契約表）** | enqueue 後 const 扱い。push 値コピー・pop const 参照・変更は新規 submit のみ。 |
> | Handler 副作用境界（REPAIR_PLAN(34) ④） | **HANDLER-1 契約** | `publishAtomic` 直接実行は RuntimeBuilder のみ。Handler は World 書き換え不可。 |

---

# 設計

本セクションは**今回改修で実装する**（ISR 設計原則に基づく設計確定項目。コードは Appendix 参照）。凡例: 📋 設計確定 / 🔴 P0（最優先）。

### ISR 設計原則（本設計書全体に適用）

| 原則 | 内容 | 根拠 |
|------|------|------|
| **Observer 副作用禁止** | Observer（Timer）は Intent Queue への push のみ。Retire/State Transition 不可。 | ISR Runtime 不変条件 |
| **Coordinator 唯一 Authority** | `processIntent()` は RuntimePublicationCoordinator のみが実行する。 | ISR Runtime 不変条件 |
| **Observe ≠ Retire** | `emitObserveIntent()` は Observation Intent。`processIntent()` は Retire Coordination。別関数・別責務。 | ISR Runtime 不変条件 |
| **Publish後は Immutable** | 一度 Publish された RuntimeWorld は変更不可。Rollback 禁止。復旧は New World の Publish。 | ISR Runtime 不変条件 |
| **Epoch 安全確認後 Ownership Release** | ACK は Epoch 完了通知であり、解放契機ではない。解放は `getMinReaderEpoch() > retireEpoch` の安全確認後にのみ実行する。 | Practical Stable ISR |
| **実行コンテキスト分離** | Observer Phase（`emitObserveIntent`）と Coordinator Phase（`processIntent`）は同一スレッド上でも明確に分離された Phase として実行する。Timer callback 内では Observer→Coordinator の順序を保証する。 | ISR Execution Context Separation |

## 🔴 P0-4A: Observe Authority — Observe Intent Queue + Timer→Coordinator 委譲 — 📋 設計確定

### アーキテクチャ

```
Timer callback（暫定実装 — Authority分離ではなく実行コンテキスト分離）
  │
  ├── [Observer Phase] emitObserveIntent()
  │     └── Intent Queue push — 即座復帰
  │
  ├── [Coordinator Phase] processIntent()  ★ 暫定: 同一 callback 内で時系列分離
  │     ├── Intent Queue pop → retire要求 → EpochDomain委譲
  │     ├── Coordinator は retire の開始のみ行う。物理削除（delete）は行わない。
  │     └── EpochDomain が唯一の Delete Authority
  │
  └── return

将来の理想:
  Timer: emitObserveIntent() のみ（即座復帰）
  Dedicated Coordinator Worker / MessageThread:
    while(...) { processIntent(); }  ★ Authority分離 + Execution Context分離
```

> **Important — 暫定実装について**: 現状の Timer callback 内での `processIntent()` 呼び出しは、**Authority 分離ではなく実行コンテキストの時系列分離（Phase 分離）**である。Observer と Coordinator の Authority はコード上分離されているが、実行主体（Timer callback）は同一であるため、**完全な Coordinator Authority 分離とは言えない。** 特に以下の2点の制約がある:
> - **Scheduling Authority は依然 Timer 側**: Coordinator Loop の実行開始を Timer が決定しており、Coordinator 自身が自律的に動作しているわけではない。
> - **Retire 遅延は 1ms 保証ではない**: Queue backlog + Coordinator 処理時間 + Epoch 待ち の総和となり、負荷状況により変動する。processIntent が毎 callback 実行されることで実用上のレイテンシは bounded だが、理論上の worst-case にはキューイング遅延が加わる。
>
> ただし、ObserveIntent は DSPHandle を保持する自己完結型 Intent であるため、processIntent は外部状態（`lifetimeMgr.getActive()`）に依存しない。**ただし `persistentState_.publicationEpoch` は世代逆転検出のために参照する**（`ISRRuntimePublicationCoordinator_ProcessIntent.cpp:13,21`。FUTURE-4 で Metadata Snapshot 統合により置換対象）。これは専用の Coordinator Worker / Loop が未整備であるための暫定措置であり、以下の理由で許容される:
>
> 1. `emitObserveIntent()` → `processIntent()` の順序が Timer callback 内で保証されており、publish interleaving が発生しない
> 2. `processIntent()` 自体は Coordinator の public メソッドであり、Observer が直接 retire を実行しているわけではない
> 3. ObserveIntent は自己完結型（DSPHandle 保持）のため、将来 Dedicated Coordinator Worker へ移行する際のインターフェース変更はゼロ（`processIntent()` の呼び出し元を変えるだけ）
> 4. 本改修のコードベースは ISR 準拠度 **約85〜90%** を達成しており、Scheduling Authority は依然 Timer 側にある。残る Scheduling Authority と Execution Context 分離（Dedicated Coordinator Loop への移行）は将来対応とする
>
> **将来の完全 ISR 移行パス**:
> 1. ✅ ObserveIntent に `DSPHandle` フィールド追加（今回実装済み）
> 2. Timer callback から `processIntent()` 呼び出しを削除
> 3. 専用の `CoordinatorLoop`（MessageThread の定期タスクまたは Worker）で `processIntent()` を実行（Coordinator 自身が Scheduling Authority を持つ）
> 4. これにより Authority 分離 + Execution Context 分離 + Scheduling Authority の完全 ISR が達成される
>
> **P1 格上げ（2026-07-31 反映）**: 上記2〜4（Dedicated Coordinator Worker への移行）は、ISR 完成形の必須要件であり、**将来対応（FUTURE）ではなく P1 として追跡する**（FUTURE-9）。完了条件に「Coordinator Worker 移行」を含め、次フェーズで実装する。

### 目的

`retirePublishedDSP()` が Timer から直接呼ばれる現状を改め、Timer は Observe Intent のみを発行し、Coordinator が retire を実行する設計に変更する。これにより以下の向上を達成する:

- **RT レイテンシ低減**: Timer callback は `emitObserveIntent()` の1命令で即座復帰
- **Coordinator Authority 一元化**: 全寿命管理（Observe/Delete/Quarantine）が Coordinator 経由に統一
- **ISR パイプライン整合**: `Publish → Observe → Retire → Epoch → Delete` に完全準拠

### データ構造

#### ObserveIntent（✅ 実装済み）

```cpp
// ISRRuntimePublicationCoordinator.h 🔬 確認済み
struct ObserveIntent {
    DSPHandle handle;           // ★ 観測対象の DSPHandle（自己完結型 Intent）。ISR: Coordinator は handle のみで retire 対象を識別可能。
    PublicationEpoch epoch;     // emit 時の publicationEpoch（FIFO順序保証、世代逆転検出用）
    uint64_t intentId;          // 診断・モニタリング用途専用。Coordinator は handle と epoch のみで処理可能。
};
```

> **Note**: ObserveIntent は DSPHandle を保持する**自己完結型（self-contained）Intent** である。コード実装でも `processIntent()` は `retireByHandle(intent.handle)` を使用しており（`ISRRuntimePublicationCoordinator_ProcessIntent.cpp:17,25`）、`lifetimeMgr.getActive()` には依存していない。Coordinator は Intent 内の `handle` のみで retire 対象を一意に識別できる。これにより、将来の専用 Coordinator Worker への移行がコード変更なく実現可能。`intentId` は診断・モニタリング用途に限定される。
>
> **実コード検証（2026-07-31）**: レビューにて「ObserveIntent に `DSPHandle` が設定されていないスニペットがある」との指摘があったが、実コードでは **`handle` は正しく設定されている**ことを確認済み。`emitObserveIntent()`（`ISRRuntimePublicationCoordinator.cpp:520-524`）は `ObserveIntent intent{ handle, persistentState_.publicationEpoch, nextObserveIntentId_.fetch_add(...) }` と 3 フィールドすべてを初期化しており、構造体定義（`.h:301-305`）も `handle / epoch / intentId` の3フィールド構成。レビューが参照したスニペットは古い記述であり、P0-4A の自己完結型 Intent 前提は実コードで成立している。
>
> **Queue 責務分離に関する注意**: Overflow Policy の Deferred 層は `coordinatorDeferredRing_` を使用しているが、このリングは本来 `RetireOverflowEntry`（Retire 系統）を保持するものであり、ObserveIntent とは異なる責務のデータを同一経路で扱う設計になっている。ObserveIntent の overflow が Retire queue の経路に流入することは、ISR 上の責務分離を曖昧にする可能性がある。ISR では Observe / Retire / Publish / Recovery は Intent 種別は同じでも Payload が異なるため、Deferred Ring の共用は責務を曖昧にする。
> 現在は overflow 発生頻度が極めて低いため問題にはならないが、**将来の完全 ISR 化では ObserveIntent 専用の Deferred Ring（例: `LockFreeRingBuffer<ObserveIntent, 1024> observeDeferredRing_`）を別途用意し、Retire 系の `coordinatorDeferredRing_` と責務を分離する**こと。これは将来対応事項として追跡する（FUTURE-8）。

**設計判定**: `LockFreeRingBuffer<ObserveIntent, 1024>` を使用。既存の `coordinatorDeferredRing_`（`LockFreeRingBuffer<RetireOverflowEntry, 1024>`）と同じパターン。SPSC なので atomic オーバーヘッドなし。Capacity 1024 は Timer 周期（1ms）× 1秒間のバッファに十分。

#### Authority Inventory 整合（SSoT）

> **Authority Inventory 分類の更新（2026-07-31 反映）**: コードベースの Authority Inventory（`AudioEngine.h:253` `kRuntimeAuthorityInventory`、`RuntimeGraph.h:41` `kAuthorityInventory`）は `Authoritative / Derived / Diagnostic` 分類で構成されている。本設計書の新規 Authority（Observe / Quarantine / Recovery）を Inventory に追加・分類すると以下のとおり。**Inventory は SSoT であり、本設計書の Authority は必ず Inventory と整合させる**:

| Authority | 分類 | 根拠 |
|-----------|------|------|
| **Observe**（`emitObserveIntent`） | **Authoritative** | 観測イベントの発行元。Retire の起動契機であり、Runtime の決定に直接影響する。 |
| **Quarantine**（`emitQuarantineIntent` / FUTURE-7 の `submitQuarantine`） | **Authoritative** | DSP の隔離判定。Runtime 挙動を変更する Authority。 |
| **Recovery**（`submitRecoveryRequest` / FUTURE-3） | **Authoritative** | 新しい RuntimeWorld の Publish 契機。Runtime の決定に直接影響する。 |
| **Coordinator**（`processIntent`） | **Authoritative** | Intent Queue の消費・Retire 委譲の唯一実行主体。 |
| **Intent Queue**（`observeIntentQueue_` / Recovery Queue / FUTURE-10 の共通 Intent Queue） | **Transport Authority** | 単なる Transport（push/pop のみ）であり Decision を持たないが、**Intent が通る唯一の経路**として Authority の一種である（FUTURE-10 で一本化し、Transport Authority を単一化する）。 |
| Overflow 診断カウンタ（`overflowCounter_` 等） | **Diagnostic** | 診断・監視のみ。Runtime 分岐を駆動しない。 |

> **Intent Queue = Transport Authority（2026-07-31 反映）**: ISR では Queue も Authority である（Decision はないが、Transport として唯一）。上記 Inventory に「Intent Queue = **Transport Authority**」を明示的に追加する。これにより「Decision Authority（各 Intent）」「Transport Authority（Queue）」「Scheduling Authority（Coordinator）」の三層が Inventory 上で俯瞰でき、設計監査が容易になる。FUTURE-10（共通 Intent Queue 一本化）により Transport Authority を単一化するのが最終形。

> これらの追加は既存の `kRuntimeAuthorityInventory`（RuntimeWorld フィールドの Authority 分類）とはレイヤーが異なる（Runtime 制御フローの Authority vs World データの Authority）ため、**同一配列には混ぜず、別途「Intent/Coordinator Authority Inventory」として整備する**。実装時に `config/authority_inventory.json` と verifier（`tools/authority_inventory_verifier.py`）の拡張で対応する（P1, FUTURE-9 と同フェーズ）。

#### ACK 定義（4種のイベント分離）

ISR では Publish / Observe / Retire / Epoch / Delete は別イベントであり、ACK もこれに対応して分離される。

| イベント | ACK種別 | 意味 | 発行タイミング |
|---------|---------|------|--------------|
| **Publish** | `ACK (published)` | RuntimeWorld が Publish された。Observer が次回の Observe で検出可能。 | RuntimePublicationCoordinator::publishWorld() 完了後 |
| **Observe** | `ACK (queued)` | Intent がキューに受理された。Timer は即座に復帰可能。 | emitObserveIntent() 直後 |
| **Retire/Epoch** | `ACK (reclaim complete)` | Epoch 安全確認が完了し、Ownership Release が可能になった。**ACK自体は解放契機ではなく、Epoch安全確認の完了通知である。** 実際の解放契機は EpochDomain::getMinReaderEpoch() > retireEpoch。 | processIntent() 完了後、EpochDomain の安全確認後 |
| **Delete** | （ACK なし） | 物理削除は EpochDomain の内部処理。Coordinator は関知しない。 | — |

> **ISR Note**: ACK(reclaim complete) は Epoch が安全を保証した証拠であり、単なる ACK ではない。以下の4イベントは別々の意味を持ち、混同してはならない:
> - **Publish Receipt**: RuntimeWorld が Publish された証拠（`RuntimePublicationCoordinator::publishWorld` が返す）
> - **Retire Receipt**: DSP が Retire キューに投入された証拠（`DSPLifetimeManager::retire` が返す）
> - **Epoch Complete**: 全 Reader が epoch を通過した証拠（`EpochDomain::getMinReaderEpoch() > retireEpoch`）
> - **Delete**: 物理削除（EpochDomain の内部。Coordinator は関知しない）
>
> 現設計では Epoch 安全確認を `processIntent()` 完了時に暗黙的に行い、`markReceiptReclaimComplete()` は Epoch 完了通知として機能する。ただし `markReceiptReclaimComplete()` の名称は「Receipt 解放完了」ではなく「Epoch 安全確認完了通知」の意味であり、命名の再検討余地がある。
>
> **命名の見直し推奨（2026-07-31 反映）**: `ACK(reclaim complete)` / `markReceiptReclaimComplete()` は「Reclaim 完了」と読めるが、実際は **Epoch Safe 通知**である。ISR 語彙に合わせ **`markEpochSafe()`** への改名を推奨する（または `notifyEpochSafe()`）。改名に伴い、状態機械上の意味も「Receipt 解放完了」→「Epoch 安全確認完了通知」に明確化する。

### 状態機械

```
emitObserveIntent() (Timer Thread, RT)
  │
  ├── observeIntentQueue_.push({handle, epoch, intentId})  ← SPSC, lock-free, RT-safe
  │     └── full → Fallback→Deferred→Drop（4層Overflow）
  │
  └── return ACK (queued) — Timer は即座に復帰

processIntent() (Coordinator Phase, MessageThread/NonRT)
  │
  ├── observeIntentQueue_.pop(intent) — SPSC, lock-free
  │     ├── empty → return (no-op)
  │     └── intent取得
  │
  ├── OBSERVE-10: intent.epoch < currentEpoch → skip (世代逆転検出)
  │
  └── DSPLifetimeManager::retire(currentDSP)
        ├── ISRRetireRouter → EpochDomain (deferred deletion)
        ├── Epoch安全確認: getMinReaderEpoch() > retireEpoch
        │   （Coordinator は delete を実行しない。物理削除は EpochDomain の責務）
        └── ✅ 安全確認後に engine.markReceiptReclaimComplete()
              → ACKは「Epoch安全確認完了通知」であり、解放契機ではない
              → 実際の解放契機は「Epoch Complete」である
              → Coordinator は pendingReceipt_ の解放契機を通知するが、delete は行わない
```

### 契約一覧

| ID | 契約 | 対象 | 実装状態 |
|----|------|------|---------|
| OBSERVE-1 | Timer は ObserveIntent のみ発行し、Retire Authority を直接実行しない | `AudioEngine.Timer.cpp` | ✅ 完了（3箇所＋DSPTransition全置換） |
| OBSERVE-2 | Coordinator は ObserveIntent を Intent Queue に追加し、即時復帰する（Timer をブロックしない） | `emitObserveIntent()` | ✅ 完了 |
| OBSERVE-3 | Coordinator Loop は Intent Queue から取り出した Intent を `processIntent()` で処理する | `processIntent()` | ✅ 完了（毎 Timer callback 実行） |
| OBSERVE-7 | Timer は `ACK(reclaim complete)` = Epoch 安全確認完了通知を受信後、`pendingReceipt_` を安全に解放する。ACK は解放契機ではなく、Epoch Complete の通知である。 | `markReceiptReclaimComplete()` | ✅ 完了（Epoch 安全確認後） |
| OBSERVE-8 | ObserveIntent は NonRT パス（Timer Thread）からのみ発行可能 | — | ✅ 完了 |
| OBSERVE-9 | ObserveIntent は Publish 順序を保持する。FIFO で Coordinator へ渡される | `LockFreeRingBuffer` | ✅ 完了 |
| OBSERVE-10 | Coordinator は古い PublicationGeneration の ObserveIntent を実行してはならない | `processIntent()` | ✅ 完了 |
| COORDINATOR-1 | **Coordinator は Queue / Routing / Dispatch のみを責務とする（9回目レビュー反映）**: 理想形は `Builder → Validator → Coordinator → Store`。Coordinator は Intent の Routing / Dispatch に専念し、**Policy 判断（優先度付け・破棄・並べ替え・World 変更・Metadata 変更）を持たない**。Policy 判断が必要な場合は専用コンポーネント（Builder / Validator / Router / Scheduler）へ委譲する。実コード検証済み: `priorityScheduler_`（`.h:226`）の `escalateAllRetires` は `RetireRuntime` へ委譲（`.cpp:491-493`）、`requestReclaim()` は `ISRRetireRouter` / `DSPLifetimeManager` へ委譲（`.cpp:568-599`）。 | Coordinator 全体 | 📋 設計確定（実装済み構造と整合） |

### 変更ファイル一覧

| ファイル | 変更内容 | 状態 |
|---------|---------|------|
| `ISRRuntimePublicationCoordinator.h` | `ObserveIntent` 構造体, `observeIntentQueue_`, `nextObserveIntentId_`, `overflowCounter_`, `processIntent()` 宣言 | ✅ 実装済み |
| `ISRRuntimePublicationCoordinator.cpp` | `emitObserveIntent()` → LockFreeRingBuffer push。`processIntent()` 実装（FIFO pop, 世代逆転検出, retire委譲） | ✅ 実装済み |
| `AudioEngine.Timer.cpp:893,1025,1591` | `retirePublishedDSP()` 直接呼出 → `runtimePublicationBridge_.emitObserveIntent()` のみ（3箇所） | ✅ 実装済み |
| `AudioEngine.Timer.cpp` 末尾 | `runtimePublicationBridge_.processIntent()` 定期実行追加 | ✅ 実装済み |
| `DSPTransition.h:146` | `engine_.retirePublishedDSP(...)` → `engine_.runtimePublicationBridge_.emitObserveIntent()` のみ | ✅ 実装済み |

### Intent Queue Overflow Policy（4層設計）

`LockFreeRingBuffer<ObserveIntent, 1024>` の `push()` は容量満杯時に `false` を返す（spin-wait しない）。この場合、Timer 側は復帰するが Intent が失われる。以下の4層ポリシーで対処する。

| 層 | 容量 | 動作 | 状態 |
|----|------|------|------|
| **Primary** | 1024 | `LockFreeRingBuffer` push 正常完了 | ✅ 実装済み |
| **Fallback** | 2048 | `LockFreeRingBuffer<ObserveIntent, 2048>` Secondary キュー | ✅ 実装済み |
| **Deferred** | 1024 | `coordinatorDeferredRing_` → `drainOverflowRing` で定期回収 | ✅ 実装済み |
| **Drop** | ∞ | `overflowCounter_` / `fallbackOverflowCounter_` increment | ✅ 実装済み |

**Overflow 状態機械**:
```
observeIntentQueue_.push({intent})
  ├── success → ACK(queued)、正常復帰
  │
  └── false (full) → observeFallbackQueue_.push({intent})     ← Fallback層
        ├── success → ACK(queued-fallback)
        │
        └── false (full) → coordinatorDeferredRing_.push({entry})  ← Deferred層
              ├── success → coordinatorDeferredCount_++
              │   （次回 drainOverflowRing で回収）
              │
              └── false (full) → overflowCounter_++ / fallbackOverflowCounter_++  ← Drop
                    └── HealthMonitor へ通知（QUEUE-15）→ Repair Scan 契機（Coordinator Health Check 起点）
```

| 契約ID | 契約 |
|--------|------|
| QUEUE-11 | Intent Queue が満杯の場合、Fallback → Deferred → Quarantine の4段階で安全側へ倒す |
| QUEUE-12 | Fallback Queue と Quarantine 発行は RT-safe（lock-free or atomic increment） |
| QUEUE-13 | Overflow 発生時は診断カウンタ（`overflowCounter_` / `fallbackOverflowCounter_`）を atomic increment する |
| QUEUE-14 | Overflow 発生を Coordinator が診断可能なイベントとして扱えるよう、HealthMonitor へ非同期通知する（将来対応） |
| QUEUE-15 | **Drop は最終状態にしない（2026-07-31 反映）**: Drop すると ObserveIntent が消え、対応 DSPHandle が Retire されない危険がある。Drop カウンタが閾値（例: 100回）を超えた場合、Coordinator は全 DSPHandle を再走査し、未 Retire の Handle を新規 ObserveIntent として再発行する。**Repair Pass としての位置付け（レビュー⑥反映）**: 本再走査は Observe の代替ではなく **Coordinator Health Check 起点の Repair Scan** である。ISR では「Observe 失敗 → Scan」ではなく「**Coordinator Health Check → Repair Scan**」とし、健全性維持の修復パスとして扱う（命名も `RepairScan` とする）。これは ISR を壊さない Health Recovery であり、`Coordinator → RetireRouter → Epoch` の通常経路に復帰させる。**例外経路の明文化（レビュー⑧反映）**: Repair Scan は**正常時の通常経路（`Timer → submitObserve → intentQueue_ → processIntent → retireByHandle`）ではない**。Repair Scan は **`Health Monitor → 異常検出（overflow 閾値超過） → 再同期（Repair Intent 生成）`** の**例外時のみ起動する修復経路**であり、通常経路が正常に機能している間は決して起動しない。設計上も Repair Scan の存在は通常経路の代替を意味せず、通常経路の健全性を損なわない（Scan 起動条件は overflow 閾値のみ）。**Diagnostic 化（レビュー⑦反映）**: Repair Scan 自身は**常に Diagnostic（走査結果の報告）に留まり、実行は行わない**。修復（未 Retire の再発行）は Repair Scan が直接実行するのではなく、**走査で検出した未 Retire Handle を `submitObserve()` で Repair Intent として enqueue し、Coordinator の通常経路（`processIntent` → `retireByHandle`）経由で実行**する。これにより Repair Scan 自身が Authority になることがなく（Scan が直接 World を書き換えない）、修復実行は常に Coordinator の Intent 経路を通過する。**実コード対応（2026-07-31 検証）**: 既存の `RuntimeHealthMonitor`（`RuntimeHealthMonitor.h:124`、Pull型監視エンジン）が `AudioEngine.Timer.cpp:1126` で tick され、`overflowCounter_` の読み取りに利用できる。Repair Scan の実行主体は Coordinator（`processIntent` 内の専用パス）とし、HealthMonitor は検知・通知（`onHealthEvent`）のみ行う（RuntimeHealthMonitor は Decision しない）。 |
| MAINTENANCE-1 | **Repair Scan は Maintenance Layer として独立させる（9回目レビュー⑤反映）**: Repair Scan（`RepairScan`）は **Observer / Monitor / HealthMonitor のいずれでもない、独立した保守作業層（Maintenance Layer）**に属する。責務分離は `Health Monitor（検知）→ Repair Scan（走査・診断）→ Coordinator（実行）` の3段階で、それぞれが独立した関心を持つ: **Health Monitor** は異常の検知のみ（`onHealthEvent`）、**Repair Scan** は走査・診断のみ（未 Retire Handle の列挙と Repair Intent 生成）、**Coordinator** は実行のみ（`processIntent` 経由の Retire）。Repair Scan はオブザーバブルなヘルスチェックの副産物ではなく、**意図的に起動される保守作業**であり、その起動契機・頻度・対象は Monitor 層のそれとは独立に設計・計測される。実装上は独立したクラス（例: `DSPMaintenanceScan`）として実装し、Coordinator のメンバ関数や HealthMonitor のサブステップに埋め込まない。**Repair Scan は RuntimeWorld Snapshot のみを見る（11回目レビュー④反映）**: Repair Scan の走査対象は **`consumeAtomic(currentWorld_)` で取得した Immutable RuntimeWorld Snapshot（const 参照）のみ**であり、mutable な内部構造（Builder の PendingMap・Coordinator の過渡状態等）や実体 DSP の可変領域を直接見ない。これは ISR の「RuntimeWorld が唯一の Authority」原則と整合する。Repair Scan が出力するのは「Snapshot 上の未 Retire Handle の列挙」であり、これが `submitObserve()` による Repair Intent 生成の入力となる。**Repair Scan は Observation の範囲に限定（12回目レビュー④・最終評価4反映）**: Repair Scan 自身も **Observation である**。その終端は必ず **`submitObserve()`（Repair Intent の生成・enqueue）** であり、**`retire()` / `delete()` / World 更新を直接実行しない**。Repair → submitObserve() →（Coordinator 経由）→ Retire → Epoch → Delete の通常経路のみを通過する。これにより Repair Scan が「直接 retire する Maintenance Authority」に化けることを防ぎ、ISR の一方向経路（`Intent → Coordinator → Retire → Epoch → Delete`）が常に維持される。 |

### 完了条件

1. ✅ `retirePublishedDSP()` が Timer から直接呼ばれず、Coordinator 経由になった（4箇所全置換完了）
2. ✅ Observer（Timer）は `emitObserveIntent()` のみ発行。Retire は Coordinator の責務。
3. ✅ processIntent が毎 Timer callback で定期実行される（MessageThread 保証）
4. 🟡 P1（次フェーズ）: **Dedicated Coordinator Worker 移行** — Timer から `processIntent()` 呼び出しを削除し、専用 `CoordinatorLoop` で実行する（FUTURE-9）。これにより Scheduling Authority が Coordinator 側に移り、ISR 完成形に到達する。**完了条件に含め、P1 として追跡する。**

### テスト計画

```cpp
// tests/ObserveIntentTests.cpp
TEST(ObserveIntentTimerFlow) {
    coordinator.emitObserveIntent();
    ASSERT_EQ(coordinator.getPendingIntentCount(), 1);
    coordinator.processIntent(lifetimeMgr, handleRuntime, engine);
    ASSERT_EQ(coordinator.getPendingIntentCount(), 0);
}
TEST(ObserveIntentFIFOOrder) {
    coordinator.emitObserveIntent(); coordinator.emitObserveIntent(); coordinator.emitObserveIntent();
    // processIntent は FIFO で処理
}
TEST(ObserveIntentGenerationReversal) {
    // OBSERVE-10: 古い PublicationGeneration の Intent を破棄
}
```

---

## 設計上の注意点（要点抽出）

| # | 項目 | 重要度 | 状態 | 説明 |
|---|------|--------|------|------|
| 1 | kMaxMismatch Timer周期依存 | ✅ 解決済み | FIX-D1 対応済み | `kMaxEpochDrift=10` に移行。周期依存から epoch 差分ベース検出へ変更。 |
| 2 | Emergency Override後の stale receipt | 🟡 LOW | P1-2 対応済み | `resetReceipt()` で quarantine Intent 発行。基盤実装済み。 |
| 3 | onTransitionComplete/notifyTransitionComplete | ✅ 実コード検証済み（2026-07-31） | 設計上の統合フック。実コードでは `notifyTransitionComplete`（`RuntimePublicationOrchestrator.cpp:392`）は現状**呼び出し元が存在しない**が、4責務（Transition Completion / Shutdown Guard / Stale Discard / Deferred Publish Submit）を定義して保持。`transition_.onTransitionComplete(currentAfterFade)`（同 `:398`）を呼び、その中で `retirePublishedDSP` へ到達する。FUTURE-9（Coordinator Loop）導入時に Coordinator 経由の統合フックとして再設計する。 |
| 4 | release/acquire + External Serialization二層依存 | ✅ 実コード検証済み（2026-07-31） | 二層の整合性は Coordinator Authority の外部 Serialization に依存。実コードでは `timerCallback()`（`AudioEngine.Timer.cpp:370`）が **Message Thread 単一スレッドで実行**され、Observer Phase → Coordinator Phase を直列化している（同 `:1051`「timerCallback は Message Thread で実行されるため RT 制約に抵触しない」）。この直列化が release/acquire 二層の整合を保証する。設計上の制約として許容し、FUTURE-9（Coordinator Loop）導入後も Phase 分離を維持する。 |
| 5 | Fatal時の pendingReceipt_ 診断用保持 | ✅ 実コード検証済み（2026-07-31） | Fatal 状態でも `pendingReceipt_` を残し、診断情報として活用。リークではなく意図的設計。実コード: `AudioEngine.Timer.cpp:1782` の Emergency Retire 分岐で「fatal → runtimeEpoch（pendingReceipt_ は診断用に保持）」を確認。`pendingReceipt_`（`AudioEngine.h:4364`）は `resetReceipt()`（`AudioEngine.h:1161`）で次回出版時に安全に解放される。 |
| 6 | MMCSS AvRevertのRT性 | ✅ 実コード検証済み（2026-07-31） | `AudioEngine.Mmcss.cpp:204` AvRevertMmThreadCharacteristics 呼び出し。MMCSS-EX-1〜5 契約策定（ADD-2）。Audio Thread が次回 callback で AvRevert を実行する MSDN same-thread 要求を遵守。 |
| 7 | ASan/TSan CI job分離 | ✅ 実コード検証済み（2026-07-31） | `ENABLE_ASAN`（CMakeLists.txt:1123）/ `ENABLE_TSAN`（:1159）オプション追加済み。`.github/workflows/sanitizer-ci.yml` に `debug-asan`（Windows MSVC）/ `debug-tsan`（Linux Clang best-effort）の独立 job を定義済み（CI-1 参照）。 |
| 8 | Coordinator 唯一 Authority 原則 | 🔴 P0 | P0-4 対応完了 | Observe(P0-4A)/Delete(P0-4B)/Quarantine(P0-5) 全3 Authority を Coordinator に一元化。 |
| 9 | FFTExecutionContext 分離 | ✅ 本設計で採用 | P1-1 実装完了 | Layer が FFT を知らない設計。`FFTExecutionContext` が仲介。 |
| 10 | ISR Coordinator 経由寿命管理 | 🔴 P0 | P0-4 対応完了 | Observe/Delete の Coordinator Authority 経由化。 |

> 詳細なコードベース検証結果、調査結果詳細、レビュー履歴、付属文書、Errata 運用については Appendix を参照。

**11回目レビュー後・追加調査（2026-07-31）**: レビュー⑪の指摘（PendingMap ライフサイクル / Dispatcher 非判断性 / FIFO 契約 / Shutdown 時の Builder 停止保証）に関連する未確定事項を実コードで調査・確定済み:
- **`processIntent()` の現在のディスパッチ構造** → `ISRRuntimePublicationCoordinator_ProcessIntent.cpp:7-31` は現状 **ObserveIntent のみ**を処理する（`observeIntentQueue_` + `observeFallbackQueue_` の二重 pop で、switch なし）。Recovery / Publish / Quarantine の Intent 種別が増える FUTURE-10 で `kDispatchTable[type]`（QUEUE-22）と tagged-union（QUEUE-21）を導入する。**現在の構造は単一種別のため switch 不存在であり、DISPATCH-1（Dispatcher は Routing のみ）は将来種別が増えても成立する**ことを確認。
- **Builder 実行状態の確認方法（SHUTDOWN-7 の実装対応）** → Builder は非同期 rebuild ワーカー。`rebuildThreadShouldExit`（`AudioEngine.h:2502`）+ `rebuildWorkerRunning`（`ISRRuntimeSemanticSchema.h:324`）で進行中 Build の有無を確認可能。`releaseResources()`（`AudioEngine.Processing.ReleaseResources.cpp:185-187`）が StopWorkers → ObserverDrained の順でワーカー停止を先に実行する既存順序と整合する形で、`VerifyDrained` 遷移時（同 `:393`）に `rebuildWorkerRunning == false` を確認する。`ShutdownBlockingReason`（`ISRShutdown.h:46-57`）に `ActiveBuilder` が未追加であることを確認（SHUTDOWN-7 で追加対象として記録）。
- **PendingMap の現状** → 実コードに未実装（設計段階）。FUTURE-3 実装時に Builder 内部の Build Session 限定構造として導入し、BUILDER-STATE 契約でライフサイクルを固定する。実装時に `clearPendingMap()`（Publish 完了後破棄）をテストで検証する。
- **共通 Intent Queue の FIFO** → 現状の `LockFreeRingBuffer<ObserveIntent, 1024>`（`ISRRuntimePublicationCoordinator.h:311`）は FIFO 保証を実装済み。FUTURE-10 の `LockFreeRingBuffer<Intent>` も同一の FIFO 保証を維持し、QUEUE-23（到着順 strict FIFO）が成立する（Priority FIFO は不採用）ことを確認。
**12回目レビュー後・追加調査（2026-07-31）**: REPAIR_PLAN(32) 2nd（Handlers 一覧・Pure Routing 絶対条件・RAII 終了保証）+ REPAIR_PLAN(34)（HANDLER-1 / INTENT-1 / Handler 副作用境界）の指摘に関連する未確定事項を実コードで調査・確定済み:
- **INTENT-1（投入後不変）の実現可否** → `LockFreeRingBuffer` の `push(const T&)`（`src/LockFreeRingBuffer.h:33-42`）は**値コピーでバッファへ格納**（`buffer[w & MASK] = item`）、`pop(T&)`（同 `:54-67`）は**コピーアウト**。両方向とも参照の保持・共有がなく、T の trivially copyable（QUEUE-21 の tagged-union は static_assert で保証）が前提。**投入後の Intent が Queue 内部・消費側のいずれでも書き換えられない構造であり、INTENT-1 が成立**することを確認。Handler には pop でコピーされた const 参照（`const Intent&`）のみが渡る。
- **HANDLER-1（Handler = Execution のみ）の実現可否** → 既存の Intent 処理は `processIntent()` 単一 → `retireByHandle`（`ISRRuntimePublicationCoordinator.cpp`）/ `requestReclaim`（同 `:568-599`）へ委譲し、Coordinator 自身が retire 詳細・Decision を保持しない既存構造（5回目調査で確認済み）と整合。FUTURE-10 で `kDispatchTable[type]`（1:1 一意写像）を導入しても Handler は**既存 API 委譲・新規 Intent submit・診断記録のみ**に留められ、World 書き換え（`publishAtomic` 直接実行）を Handler に持たせない契約（HANDLER-1）が実装可能であることを確認。
- **DISPATCH-1（Pure Routing）の検証手段** → `static_assert(DispatcherHasNoDecision)` の検証は (1) `kDispatchTable` が `IntentType` から Handler への**一意写像（1:1・重複なし）**であること、(2) Dispatcher（`processIntent()`）が可変状態（`Intent` 以外のメンバ変数）を参照しないこと、の2点をコンパイル時に検証する設計とする。Dispatcher 自体が状態を持つ構造は現行の `processIntent()`（`ISRRuntimePublicationCoordinator_ProcessIntent.cpp:7-31`、二重 pop + 世代逆転検出のみ）と整合し、追加可能であることを確認。**注意（2026-07-31 実コード検証）**: `processIntent()` は世代逆転検出のため `persistentState_.publicationEpoch` を参照する（同 `:13,21`）。これは「Decision を持たない」こととは両立する（読み取り専用の state 参照）が、**FUTURE-4 で Metadata Snapshot 統合により `consumeAtomic(currentWorld_)` 経由の読み取りへ置換する**ことで、Dispatcher の状態参照をさらに削減できる。
- **BUILDER-STATE RAII 化の影響範囲** → Builder は非同期 rebuild ワーカー（`rebuildThreadShouldExit` / `rebuildWorkerRunning`）で **noexcept バウンド**（`processIntent` も `noexcept`）のため、例外は Build 内の検証・診断関数（`ImmutableWorldVerifier` 等）から来る可能性に限定される。RAII `BuildSession` は例外安全性テスト（`BuildSessionRAIIExceptionSafety`）で検証し、`noexcept` 境界では早期 return・キャンセル経路がデストラクタで必ず PendingMap を破棄することを保証する。

---

---

# 未確定事項

本セクションも**今回改修で調査・解決する**。凡例: ⚡ 設計方針確定（実装未着手） / ✅ 確認済み / 🔷 調査中。
**2026-07-31 現在: 全項目調査・解決済み**（MemoryPool / Handle Table は延期判断妥当とレビューで確認していたが、**2026-07-31 決定により FUTURE-5 / FUTURE-6 として今回改修で実装する** — 下記 Appendix は調査記録として維持し、実装詳細は FUTURE-5/6 参照、P0-2 / P0-2b は解決済み）。
**6回目レビュー後・追加調査（2026-07-31）**: レビュー⑥の指摘に関連する未確定事項を実コードで調査・確定済み:
- **Metadata Freeze 責任者** → `SemanticTransactionState::Published` が terminal（`ISRRuntimeSemanticSchema.h:546`）、`ImmutableWorldVerifier` が Fatal verifier（同 `:455`）、Freeze は `publishAtomic(currentWorld_, ...)`（`ISRRuntimePublicationCoordinator.cpp:107`）= **Publisher（Coordinator commit()）のみ**（METADATA-4）。
- **Publish は Intent Queue 非経由（QoS 検証）** → `AudioEngine.Commit.cpp:391` → `commit()` → `publishAtomic(currentWorld_, ...)` の直接経路。Publish Intent は FIFO 遅延の影響を受けない（QUEUE-20 メモ）。
- **Observe Queue 容量** → Primary 1024（`.h:311`）+ Fallback 2048（`.h:315`）。FUTURE-10 の `kIntentQueueCapacity` 基準は約3072。
- **notifyTransitionComplete** → `RuntimePublicationOrchestrator.cpp:392`。現状呼び出し元なし（設計上の統合フックとして4責務を保持）。FUTURE-9 で再設計。
- **Timer の External Serialization** → `timerCallback()`（`AudioEngine.Timer.cpp:370`）が Message Thread 単一スレッドで Observer→Coordinator Phase を直列化。
- **CI workflow** → `.github/workflows/sanitizer-ci.yml`（121行）に debug-asan / debug-tsan の独立 job を確認（CI-1 に反映済み）。
**8回目レビュー後・追加調査（2026-07-31）**: レビュー⑧後に未確定事項（デッドコード・再提出経路）を実コードで調査・確定済み:
- **Deferred Publish の再提出経路（要調査→確定）**: `enqueueDeferred()`（`RuntimePublicationOrchestrator.cpp:340`）で保存されるが、**消費側 `consumeDeferredRequest()`（同 `.h:65`）は呼び出し元なし（デッドコード）**。Timer の `hasDeferredRequest()` 検知（`AudioEngine.Timer.cpp:1040`）→ `triggerAsyncUpdate()` は rebuild 処理のみで deferred publish の再提出は行わない（`AudioEngine.RebuildDispatch.cpp:375` の `handleAsyncUpdate()` に deferred commit 消費なし）。**クロスフェード完了の正常系は `notifyTransitionComplete()` を経由せず直接 publish する**（`AudioEngine.Timer.cpp:904-918`: `publishWorld()` + `commitRuntimePublication()`）。したがって **deferred publish の再提出は現状「次の publish 契機で上書き・置換」されるか、Shutdown 時に `clearDeferredForShutdown()`（同 `.cpp:456`）で破棄される**のみ。→ **FUTURE-9 の再設計対象として確定**（SHUTDOWN-2/4 の Drain Intent / Advance Epoch で「deferred の完全消化」を保証するため）。
- **notifyTransitionComplete / consumeDeferredRequest のデッドコード** → `doc/work89/BUG-052.md` に記録済み（LOW）。`hasDeferred_`（plain bool）のデータ競合リスクは現状デッドコードのため発現しないが、**FUTURE-9 でこれらを有効化する際は atomic 化（`std::atomic<bool>`）が必要**。
- **`finalizeRuntimeBuildSnapshot`** → `AudioEngine.RebuildDispatch.cpp:107` に実在（FUTURE-4 実装ステップ3 の Builder freeze 参照先として整合確認済み）。
- **`handleAsyncUpdate` の実体** → `AudioEngine.RebuildDispatch.cpp:375` に実在（rebuild 要求の消費のみ。deferred publish 消費はなし）。
**5回目レビュー反映（2026-07-31）**: 追加契約（RECOVERY-7 / QUEUE-15 / TSAN-1〜2 / Metadata 完全不変化）に対応する実コードの存在を検証済み:
- `overflowCounter_` / `fallbackOverflowCounter_`（`ISRRuntimePublicationCoordinator.h:319-320`）✅ 実在
- `RuntimeHealthMonitor`（`RuntimeHealthMonitor.h:124`、Pull型、`AudioEngine.Timer.cpp:1126` で tick）✅ 実在 — QUEUE-15 の Repair Scan 契機として利用可能
- `requestReclaim()` の委譲構造（`ISRRuntimePublicationCoordinator.cpp:568-599`）✅ 実コード検証済み — Coordinator は retire 詳細を保持しない
- `ISRRetireRouter`（epoch 確認 + enqueueWithRetry）✅ 実在 — Coordinator の epoch 安全確認委譲先

## ✅ MemoryPool化 — 調査記録（実装は FUTURE-5 で今回実施）

| 項目 | 内容 |
|------|------|
| **設計方針** | 確定（FUTURE-5）。コードには未完全実装。 |
| **内容** | DSPHandle の内部ストレージを動的メモリプールに移行。現在の256固定スロット（`std::array<DSPRegistrySlot, MAX_DSP_SLOTS>`）を動的確保に変更する。 |
| **メリット** | 256スロット制限の解消。メモリ使用量の動的最適化。 |
| **トレードオフ** | 動的確保によるRT安全性の低下。プール管理の複雑性増加。 |
| **現状の制限** | `MAX_DSP_SLOTS=256` は実運用で十分だが、理論上の上限が存在する。 |
| **推奨** | ~~延期（3回目レビューでも妥当と確認）~~ → **2026-07-31 決定: 今回改修で実装（FUTURE-5 参照）。** 初期256スロット + NonRT拡張 + RT O(1) bounded でトレードオフを解決。 |

## ✅ Handle Table完全移行 — 調査記録（実装は FUTURE-6 で今回実施）

| 項目 | 内容 |
|------|------|
| **設計方針** | 確定（FUTURE-6）。コードには未完全実装。 |
| **内容** | 現在の `std::unordered_map<DSPCore*, DSPHandle> runtimeDSPHandleMap_` を Handle Table（密なスロット配列 + 逆引き index）に移行する。 |
| **メリット** | O(1) lookup（現在は linear scan `eraseByHandle`）。メモリアクセスパターンの改善。 |
| **トレードオフ** | 移行に伴うリファクタリングコスト。現在の `MAX_DSP_SLOTS=256` では linear scan でも実用上問題ない。 |
| **推奨** | ~~延期（3回目レビューでも妥当と確認）~~ → **2026-07-31 決定: 今回改修で実装（FUTURE-6 参照）。** reverse を密配列 `reverseSlot_[slot]` で O(1) 化し、FUTURE-5 のプールスロットと 1:1 対応させる。 |

## ✅ P0-2: tryAddRef Dead Code — 解決済み

| 項目 | 内容 |
|------|------|
| **確認日** | 2026-07-29 |
| **確認ツール** | WSL grep, serena MCP |
| **発見内容** | `RefCountedDeferred.h:51` で定義されている `tryAddRef()` メソッドが、全 `.cpp`/`.h` ファイルから一度も呼び出されていない。 |
| **対応状況（2026-07-31 確認）** | `RefCountedDeferred.h:4` に `★ P0-2: DEPRECATED. EQCoeffCache は DSPHandleRuntime に移行済み。` の DEPRECATED 注記を追記済み。関数本体は将来のリファクタリング再利用可能性のため現状維持。 |
| **リスク** | なし（Dead Code）。削除しても安全だが、将来のリファクタリングで再利用される可能性があるため現状維持。 |

## ✅ P0-2b: retirePublishedDSP 比較ロジックの完全性検証待ち

| 項目 | 内容 |
|------|------|
| **内容** | P0-2b で `current == pendingReceipt_->dsp` から `currentHandle == pendingReceipt_->handle` に変更したが、`getFadingRuntimeDSPHandle()` が常に正しい Handle を返すとは限らない。複数の fading が同時に存在する場合の動作が未検証。 |
| **調査結果（2026-07-31）** | `beginCrossfade()` は `fadingRuntimeDSPHandle_` へ **store（`publishAtomic`）** で書き込む（CAS ではない）。保護の実体は `DSPState::CrossfadingOut/CrossfadingIn` フラグによる同時 crossfade 防止と、単一 Writer（NonRT）前提。複数 fading は設計上防止されているため、`getFadingRuntimeDSPHandle()` は最後に開始した crossfade の Handle を返す。 |
| **リスク** | LOW（従来評価を維持）。CAS 排他という従来記載は不正確であり、正しくは「state フラグ + 単一 Writer」による排他。実装上は非 Null の fading Handle が Retire 対象に混入することはない（`activate()` 時に `null` へリセット、`endCrossfade()` 時にも `null` へリセット）。 |
| **追跡** | ✅ 解決済み（2026-07-31）: `NormalRetireDSPHandleCompareTests.cpp` を CMakeLists.txt へ登録（`add_executable` + `add_test(NAME NormalRetireDSPHandleCompare)`、icx 用 LTCG 無効化・`/utf-8`・`/EHsc`・WIN32 defs を含む）。`testFadingRuntimeDSPHandle()` を新規追加し、`beginCrossfade` で from を返す・`activate`/`endCrossfade` で null にリセットされることを検証。Release ビルドで4テスト全 PASS（ctest 確認済み）。※ Debug ビルドは 16-byte `atomic<DSPHandle>` のロックフリー検証 assert が icx で失敗する既知の制約（ISRDSPHandle.h:174-178 参照）。 |

---

# Appendix: 実装済み事項一覧

## A-1: v20.2.6 新規追加実装済み事項（全10件）

| ID | 内容 | 成果ファイル | 確認内容 |
|----|------|-------------|----------|
| ✅ **P1-1** | FFT Backend Concept 全5Phase | `FFTBackend.h/cpp`, `FFTExecutionContext.h`, `ConvolverBuilder.h`, `MKLNonUniformConvolver.h/cpp` | `FftStatus`/`FftStage` enum, `FftBackendConcept`, `ProductionFft`, `TestFft`, `FFTExecutionContext`, `ConvolverBuilder`, Layer `m_fftPlan`/`m_fftCtx`統合, 6FFT呼出全置換, `releaseAllLayers`, `FFTBackendTests`(7テスト) |
| ✅ **P0-2** | EQCoeffCache DSPHandleRuntime移行 | `EQProcessor.h`, `AudioEngine.h`, `AudioEngine.Cache.cpp`, `RefCountedDeferred.h` | `EQCoeffCache`→`RefCountedDeferred`継承削除, `CacheMap`→`DSPHandle`化, `getOrCreate()`/`get()`→`DSPHandleRuntime::create()`/`resolve()`統合 |
| ✅ **P1-2** | Receipt状態機械 | `AudioEngine.h`, `AudioEngine.Timer.cpp`, `ISRDSPQuarantine.h` | `resetReceipt()`実装, `QuarantineReason::ReceiptReset`追加 |
| ✅ **ADD-2** | MMCSS例外登録簿 | `doc/coding_rule_jp.txt` | MMCSS-EX-1〜5契約, 例外登録簿テーブル追加 |
| ✅ **ADD-4** | ASan/TSan CI設定 | `CMakeLists.txt`, `.github/workflows/sanitizer-ci.yml` | `ENABLE_TSAN`オプション追加, debug-asan+debug-tsan CI workflow |
| ✅ **P0-4B** | Delete Authority — reclaim() Coordinator専用化 | `ISRDSPHandle.h`, `ISRRuntimePublicationCoordinator.h/cpp` | `reclaim()` private + friend。`shutdownReclaim()` 追加（DELETE-7）。`requestReclaim()` に executeRetire→waitReaders→executeReclaim 実装。全4箇所の直接 reclaim() 呼び出しを shutdownReclaim() に置換。 |
| ✅ **P0-2b** | PublishReceipt DSPCore*削除 | `AudioEngine.h`, `AudioEngine.Timer.cpp`, `DSPTransition.h` | `PublishReceipt::dsp` 削除。`storeReceipt()` DSPCore*引数削除。retirePublishedDSPのNormal Retire判定をDSPHandle比較に変更。 |
| ✅ **P0-5** | QuarantineService | `ISRRuntimePublicationCoordinator.h/cpp` | `QuarantineService` クラス新規追加。`emitQuarantineIntent()` → QuarantineService 経由の単一Authority。Timer内の直接 quarantine/quarantineHandle 呼び出しを emitQuarantineIntent に置換。 |
| ✅ **CACHE-LT-1** | キャッシュライフタイム契約 | `doc/work88/REPAIR_PLAN.md` | 通常時`retire()`のみ/Shutdown時`resolve→delete→reclaim`の契約明文化 |
| ✅ **P0-4C** | Coordinator Interface拡充 | `ISRRuntimePublicationCoordinator.h/cpp` | `emitObserveIntent()`, `emitQuarantineIntent()`, `requestReclaim()` 実装完了（P0-4A/B/5 で本実装に置換）。プレースホルダからの昇格完了。 |

## A-2: v12 新規追加実装済み事項（6件）

| ID | 内容 | ファイル | 確認内容 |
|----|------|----------|----------|
| ✅ **P0-1** | SafeStateSwapper tail 2-writer 解消（head 専用化） | `SafeStateSwapper.h` | `tryReclaimSlot()` + `advanceHead()` + `ReclaimResult` enum 実装済み。`publishAtomic(tail)` は `swap()` のみ |
| ✅ **P0-3** | AudioSegmentBuffer 61MB ヒープ化 | `AudioSegmentBuffer.h`, `NoiseShaperLearner.h` | `ScopedAlignedPtr` heap + factory + Rule of Five + `static_assert(sizeof<1024)` |
| ✅ **P2** | updateAudioThreadSnapshotFade 削除 | `AudioEngine.h:3738`, `src/core/SnapshotCoordinator.h:111` | DELETED コメント確認 |
| ✅ **ADD-1** | fallbackQueue bounded化 | `SafeStateSwapper.h:448` | `kMaxFallback=1024` + overflow counter 実装済み |
| ✅ **ADD-3** | DeferredFreeThread Logger rate limit | `DeferredFreeThread.h:169,184-185` | `kLogInterval=5s` + `lastLogTime_` 実装済み |
| ✅ **FIX-D1** | kMaxMismatch epochベース化 | `AudioEngine.h:1133,4372`, `AudioEngine.Timer.cpp:1820` | `kMaxEpochDrift=10` + `publicationEpochDistance()` |

## A-3: 実装済み事項一覧（全37件）

### HW-1: Publication Metadata Propagation ✅ 完了

| 項目 | 内容 |
|------|------|
| **対応バグ** | BUG-030（拡張） |
| **重要度** | 🔴 HIGH |
| **関連ファイル** | 7ファイル |
| **ステータス** | ✅ **実装完了・テスト通過（19/19）** |

**実装ファイル**: `ISRRuntimeSemanticSchema.h`, `ISRRuntimePublicationCoordinator.h`, `AudioEngine.h`, `AudioEngine.Timer.cpp`, `DSPLifetimeManager.h`, `DSPTransition.h`, `RuntimePublicationOrchestrator.cpp`

### グループA: バグ修正（13件完了）

| ID | バグ | ファイル | 修正内容 |
|----|------|----------|----------|
| A-1 | BUG-038 | `SpectrumAnalyzerComponent.h:74` | `FFT_MAGNITUDE_SCALE = 2.0f / NUM_FFT_POINTS` |
| A-2 | BUG-035 | `ConvolverProcessor.LoadPipeline.cpp` | RAII `ApplyComputedIRLoadingGuard` 導入 |
| A-3 | BUG-036 | `ConvolverProcessor.LoadPipeline.cpp` | `irL.release()`/`irR.release()` を init 成功時に移動 |
| A-4 | BUG-034 | `MKLNonUniformConvolver.cpp`（6箇所） | `clearFFTOutputOnError()` ヘルパー導入 |
| A-5 | BUG-011/012/013 | `CmaEsOptimizer.h/Dynamic.h/Dynamic.cpp` | `sigma = std::clamp(s, sigmaMin, sigmaMax)` 5箇所 |
| A-6 | BUG-029 | `DSPTransition.h` | Emergency Override で `exchangeFadingRuntimeDSP` を使用 |
| A-7 | BUG-028 | `CrossfadeRuntime.h` | `complete()` で全フラグリセット |
| A-8 | BUG-015 | `ISRRetireRouter.cpp` | `enqueueWithRetry`（`.h:96` / `.cpp:161`）でリトライロジック内蔵＋戻り値確認 |
| A-9 | BUG-016 | `CmaEsOptimizer.h/Dynamic.h` | `sanitize()` で NaN/Inf→0.0 クランプ |
| A-10 | BUG-042/044/046 | 各クラス | Rule of Five（`=delete`/`=default`） |
| A-11 | BUG-045 | `IRConverter.cpp` | resample 失敗時に `actualSampleRate = sourceRate` |
| A-12 | BUG-039 | `CustomInputOversampler.cpp` | `std::min(targetSamples, ...)` |
| A-13 | BUG-040 | `NoiseShaperLearner.cpp` | `sampleRateHz > 0 ? ... : 48000` フォールバック |

### グループB: 設計確定済み（4件完了）

| ID | バグ | ファイル | 修正内容 |
|----|------|----------|----------|
| B-1 | BUG-030 | `AudioEngine.h`, `DSPTransition.h`, `AudioEngine.Timer.cpp` | `claimFadingRuntimeDSP` CAS-only 実装 |
| B-4 | BUG-032 | `SnapshotCoordinator.h:122` | `getCurrentSnapshot()` インターフェース追加 |
| B-5 | BUG-024 | `SnapshotFadeState.h` | `fadeGeneration_` ABA 対策 |
| B-6 | BUG-037 | `ConvolverProcessor.h:883`, `ConvolverProcessor.Lifecycle.cpp:107` | `loaderGeneration_` UAF 防止 |

### グループC: 計画的対応（7件完了）

| ID | バグ | ファイル | 修正内容 |
|----|------|----------|----------|
| C-1 | BUG-033 | `AudioEngine.Processing.BlockDouble.cpp:421` | `dryScale` ラムダキャプチャ追加 |
| C-2 | BUG-025 | `SnapshotCoordinator.cpp:38` | `enqueueWithRetry()` 化（リトライ込み retire 委譲） |
| C-3 | BUG-018 | 3ファイル | `!=1.0` → `std::abs(x-1.0)>1e-5f` |
| C-4 | BUG-019 | `TruePeakDetector.cpp:102-111` | `int` → `size_t` |
| C-5 | BUG-020 | `ConvolverProcessor.LoaderThread.cpp:151-152` | `if(targetLength<=0)return 0;` |
| C-6 | BUG-021/022 | `ConvolverProcessor.Lifecycle.cpp:147-150` | RCU `GlobalGuard` 追加 |
| C-7 | BUG-026 | `ObservedRuntime.h:49` | `rootEnterSucceeded()`確認 |

### グループD: 余裕時（4件完了）

| ID | バグ | ファイル | 修正内容 |
|----|------|----------|----------|
| D-1 | BUG-041 | `NoiseShaperLearner.cpp:649` | VLA→`makeAlignedArray` ヒープ割当 |
| D-2 | BUG-043 | `IRConverter` | パラメータ名修正 |
| D-3 | BUG-027 | `SnapshotCoordinator.cpp:15` | `target==null` 時 state 再確認 |
| D-4 | BUG-046 | `PsychoacousticDither.h` | A-10 に含む（Rule of Five） |

## A-4: 修正案詳細 (FIX) — 実装済み

### FIX-P0-1: SafeStateSwapper — Option A（head 専用化）✅ 実装済み

**変更内容**: `tryReclaim()` 内の tail 回転コード削除, head 専用 reclaim, null slot skip を bounded loop で実装。
**CI 3層化**: L1: rg `publishAtomic(tail` → swap() 内のみ / L2: ast-grep `tryReclaim` 内の `publishAtomic.*tail` 禁止 / L3: `SafeStateSwapperTailWriterSingleTests`
**テスト追加（8件）**: `SafeStateSwapperTailWriterSingleTests`, `SafeStateSwapperHeadOnlyReclaimTests`, `SafeStateSwapperNullSlotSkipTests`, `SafeStateSwapperEpochOrderTests`, `SafeStateSwapperHeadBlockingTests`, `SafeStateSwapperFullFallbackTests`, `SafeStateSwapperFallbackOverflowTests`, `SafeStateSwapperReaderStuckTests`

### FIX-P2: updateAudioThreadSnapshotFade 削除 ✅ 実装済み

### FIX-D1: kMaxMismatch Epoch ベース検出への移行 ✅ 実装済み

### FIX-ADD-1: fallbackQueue bounded 化 ✅ 実装済み

### FIX-ADD-3: DeferredFreeThread Logger rate limit ✅ 実装済み

### FIX-P1-2: Stale receipt quarantine 状態機械 ✅ 実装済み

### FIX-ADD-2: MMCSS AvRevert 例外登録 ✅ 設計完了（文書のみ）

### FIX-ADD-4: ASan / TSan CI job 分離 ✅ 設計完了

---

# Appendix B: コードベース検証結果（2026-07-29）

## B-1: コードベース検証結果（全ツール使用）

| 調査項目 | ツール | 結果 |
|---------|--------|------|
| EQCoeffCache 継承関係 | WSL grep / serena | ✅ P0-2 完了。`EQProcessor.h:119` の `EQCoeffCache` は素の struct（`RefCountedDeferred` 継承は削除済み）。`DSPHandleRuntime` 管理に移行。 |
| DSPHandleRuntime 実装状況 | WSL grep / AiDex | ✅ `ISRDSPHandle.h/cpp` に完全実装（create/resolve/retire/quarantine/reclaim 全API稼働） |
| emitRetireIntent 有無 | WSL grep | ✅ `ISRRetire.h/cpp` に実装済み |
| emitObserveIntent 有無 | WSL grep / semble | ✅ Queue push + DSPHandle 実装済み。processIntent は retireByHandle で自己完結動作。P0-4A 完了。 |
| emitQuarantineIntent 有無 | WSL grep / semble | ✅ QuarantineService 経由で実装済み（P0-5完了） |
| QuarantineService 有無 | WSL grep / semble | ✅ 実装済み（P0-5完了） |
| ProductionFft / TestFft | WSL grep / AiDex / semble | ✅ P1-1 実装完了 |
| MMCSS例外登録簿 別ファイル | grep / ls | ✅ `doc/exception_registry.md` が独立ファイルとして存在（MMCSS-EX 契約記載）。`doc/coding_rule_jp.txt` にも MMCSS-EX-1〜5 を記載。 |
| ENABLE_TSAN | WSL grep | ✅ CMakeLists.txt:1159 実装済み |
| TODO(ADR-010) | WSL grep | ✅ `getVersion()` が実装済み（`ISRRuntimePublicationCoordinator.cpp:168-175`）。`persistentState_.mappedRuntimeGeneration`（read-only 単調増加 uint64）を返す。`assert(true)` プレースホルダではなく、コメント注記として ADR-010 を記録。※FUTURE-4 で Metadata Snapshot 統合により置換対象。 |
| FallbackQueue bounded化 | WSL grep / AiDex | ✅ `kMaxFallback=1024` |
| DeferredFreeThread Logger rate limit | WSL grep | ✅ `kLogInterval=5s` |
| kMaxMismatch epochベース化 | WSL grep / AiDex | ✅ `kMaxEpochDrift=10` |
| RetireRuntime fallbackQueue 容量 | WSL grep | ✅ `FALLBACK_QUEUE_CAPACITY=4096` |

## B-2: v20.4 ISR Design Refinements（本版で反映）

| # | 改善内容 | 反映先 |
|---|---------|--------|
| 1 | **単一削除 Authority 確定**: EpochDomain を唯一の削除 Authority | P0-4B, DELETE-8 |
| 2 | **ACK 定義拡張 — Receipt 完全ライフサイクル**: 文書化 | P0-4A ISR Note |
| 3 | **processIntent() 将来方向**: markReceiptReclaimComplete 過渡的措置を明記 | P0-4A コメント |
| 4 | **Overflow Policy 4層化**: Deferred 層追加 | P0-4A §6 |

## B-3: 調査で使用したツール

grep/ast-grep/rg/sed/awk/fdfind/fzf（WSL）, serena MCP, AiDex MCP, cocoindex, semble, graphify

---

# Appendix C: 補完セクション

## C-1: 拡張 Enum 定義

### AckResult — Intent Queue ACK 用

```cpp
enum class AckResult : int {
    Accepted = 0,     // Intent がキューに受理された
    QueueFull,         // Intent Queue が満杯
    ShuttingDown       // Shutdown 中で新規 Intent を受付不可
};
```

### EnqueueResult — 汎用 Enqueue 結果

```cpp
enum class EnqueueResult : int {
    Success = 0,
    QueueFull,
    QueueFullCritical,
    Shutdown,
    InvalidArgument,
    NotReady,
    Duplicate,
    RejectedByPolicy,
    RejectedByAdmission,
    InternalError
};
```

**契約（ENQUEUE-1〜9）**: 全 enqueue 関数は `[[nodiscard]] noexcept`。`QueueFullCritical` は critical command が reserved slot にも enqueue 不可。`Shutdown` は終端状態。`InternalError` は HealthMonitor へ報告。

## C-2: Shutdown 状態機械

```cpp
enum class ShutdownState : int {
    Running = 0, ShutdownRequested, Draining, EpochWaiting,
    Reclaiming, Quarantined, ShutdownCompleted, Faulted
};
```

**遷移**: `Running → ShutdownRequested → Draining → EpochWaiting → Reclaiming → ShutdownCompleted`, EpochWaiting から timeout → `Quarantined → ShutdownCompleted`

**閾値**: `EPOCH_WAIT_NORMAL_MS=100`, `EPOCH_WAIT_SHUTDOWN_MS=1000`, `EPOCH_WAIT_HARD_LIMIT_MS=3000`

**契約（SHUTDOWN-1〜7）**: Shutdown は最高優先度。冪等。新規 enqueue 拒否。Faulted 遷移可能。

## C-3: HealthMonitor イベント種別

```
EVENT_FFT_ERROR, EVENT_QUEUE_FULL, EVENT_QUEUE_FULL_CRITICAL,
EVENT_EPOCH_WAIT_TIMEOUT, EVENT_QUARANTINE_ENTERED, EVENT_QUARANTINE_RECLAIMED,
EVENT_QUARANTINE_ABANDONED, EVENT_QUARANTINE_LIMIT_EXCEEDED,
EVENT_QUARANTINE_SERVICE_FAILURE, EVENT_READER_SLOT_USAGE,
EVENT_PUBLICATION_MISMATCH, EVENT_RETIRE_OVERFLOW,
EVENT_ADMISSION_STOPPED, EVENT_SHUTDOWN_REQUESTED,
EVENT_SHUTDOWN_COMPLETED, EVENT_FAULTED
```

**契約（HEALTH-1〜7）**: bounded enqueue, RT non-blocking, NonRT 集計, pull 型 UI, 診断ログは `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガード。

## C-4: Traceability Matrix

| BUG/リスク | 設計項目 | 契約 | テスト | 完了条件 |
|---|---|---|---|---|
| FFT 異常系未検証 | P1-1 | FFT-PROD / FFT-TEST / FFT-FAIL | FftErrorInjectionTests | clearFFTOutputOnError 発火、silent output |
| ASan CRT 衝突 | ADD-4 | ASan-CMAKE-1〜8 | debug-asan build | link success、ASan clean |
| TSan 非現実性 | ADD-4 | TSAN-ALT-1〜7 | stress / audit | race 代替検証 |
| critical drop 矛盾 | QUEUE-9 | QUEUE-9-1〜10 | queue stress | QueueFullCritical、critical drop なし |
| epoch timeout 未定義 | Epoch/Quarantine | EPOCH-1〜9 / QUARANTINE-1〜10 | shutdown drain | quarantine or reclaim |
| MMCSS RT 呼び出し | ADD-2 | MMCSS-EX-1〜5 | doc + audit | exception registry |
| DSPState UAF | DSPState/P0-4 | DSPSTATE-1〜8 / RETIRE-1〜6 | publication tests | UAF なし |
| Receipt stale | P1-2 | RECEIPT-1〜8 | receipt state test | quarantine/reset via intent |
| Handle Authority 分散 | P0-4/P0-5 | ISR-AUTH-1〜6 | coordinator tests | Authority 単一化 |
| Qipo 二重定義 | ADD-4 | ASan-CMAKE-5 | CMake inspect | 条件付き単一定義 |
| clearFFTOutputOnError 移行 | P1-1 | FFT-STAGE / FFT-STATUS | migration test | legacy stage 互換 |
| Quarantine 二重 Authority | P0-5 | QSVC-1〜4 | quarantine tests | State+Audit 単一管理 |
| workBuffer alignment | P1-1 | FFT-PROD-11〜14 | debug assert / ASan | 64-byte alignment |
| toFftStage 範囲外 | P1-1 | FFT-STAGE-6〜9 | unit test | Diagnostic clamp |

## C-5: 付属文書

```text
doc/exception_registry.md            — MMCSS 例外登録簿（未作成）
doc/health_monitor_events.md         — HealthMonitor イベント定義
doc/fft_backend_concept.md           — FftBackendConcept / FftStatus / FftStage 完全仕様
doc/quarantine_lifecycle.md          — Quarantine ライフサイクル詳細
doc/ci_asan_matrix.md                — ASan CI 設定マトリックス
doc/errata/v20.2-errata.md           — 設計と実装の乖離を記録する errata
```

## C-6: Errata 運用

実装中に乖離が見つかった場合、コードを無理に設計書へ合わせず以下を行う:
1. 事実を実測する
2. errata を追記する
3. 契約番号を振る
4. テストを追加する
5. 実装へ反映する

ERRATA 命名規則: `ERRATA-{Phase}-{番号}`（例: `ERRATA-PHASE0-1`）

---

*本設計書は ISR Runtime OS 設計原則に基づく。v20.5（確定版）: v20.2.6 + ISR Review 4件反映 + Phase2 Coordinator整備(4件) + REPAIR_PLAN(19) ISRレビュー対応（Observer純化・Coordinator専用processIntent・EpochベースOwnership Release・publishRecoveryRuntime・Runtime Version API分離・MMCSS例外登録簿・Full FUTURE-1実装）。全26最終受け入れ条件中20実装済み・6設計確定。「一部実装済み」全解消。*
