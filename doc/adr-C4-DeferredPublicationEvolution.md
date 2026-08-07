# ADR-C4 — Deferred Publication Evolution (notifyTransitionComplete 撤去と stale discard の位置づけ)

**Status:** Accepted (implemented)
**Date:** 2026-08-05（第11回 ISR レビュー反映: 2026-08-06 更新）
**Context:** ISR publish 経路の同期一本化リファクタリング。旧設計の `RuntimePublicationOrchestrator::notifyTransitionComplete` はクロスフェード完了時に保留中の deferred publish を再送する統合フックだったが、Builder/Coordinator 分離によりその責務はすべて別層へ移譲され、関数は dead code 化した。

## Decision
- **`notifyTransitionComplete()` は削除する。** 呼び出し元ゼロ (grep 確認済み)。
- **関数を置換した経路を唯一の正式経路とする:**
  ```
  CoordinatorLoop (Decision/Notify)
      ↓ publishRetryReady (rebuildMutex 保護 bool)
  RebuildThread (Builder)
      ↓ consumeDeferredRequest() → submitPublishRequest()
  ```
- **stale discard の設計知見はコードコメントではなく本 ADR に移管する。**
  - 半年後の読者にとって「なぜ dead code があるか」より「なぜこの層にあるべきか」の方が価値がある。
- **stale discard (generation + publication sequence guard) は Transition Completion の責務ではなく Admission Policy の責務として整理する。**

## 旧責務の分解と移譲先

`notifyTransitionComplete` が担っていた 4 責務の行き先:

| # | 旧責務 | 移譲先 | 状態 |
|---|--------|--------|------|
| 1 | Transition Completion (`transition_.onTransitionComplete`) | `AudioEngine::publishIdleWorldOnly()` (Timer が直接呼ぶ) | 移譲済み |
| 2 | Shutdown Guard (deferred キャンセル) | `clearDeferredForShutdown()` | 分離済み |
| 3 | Stale Discard (generation + sequence 二重ガード) | → **Admission Policy へ再配置 (下記)** | アルゴリズムのみ再設計対象 |
| 4 | Deferred Publish Submit | `consumeDeferredRequest()` → `submitPublishRequest()` (RebuildThread) | 完全移行済み |

## stale discard の設計知見（資産）

以降の Layer 2/3 統合では関数 API ではなく下記アルゴリズムを再利用する。

### 判定フロー（旧 `notifyTransitionComplete` 内）
TTL 超過 → generation 検査 → publication sequence 検査 → 有効なら submit。

**TTL 超過 (最優先):**
```
deferred.enqueueTimestampUs != 0
&& (nowUs - enqueueTimestampUs) > kDeferredPublishTTLUs
→ DiscardReason::Expired
```

**Generation Guard:**
```
deferred.guard.generation != 0
&& deferred.guard.generation != currentGen (engine_.rebuildRequestGeneration)
→ DiscardReason::StaleDiscard
```

**Publication Sequence Guard:**
```
deferred.guard.sequence < currentPubSeq (getLastCommittedPublicationSequence)
→ DiscardReason::StaleDiscard
```

### 設計指針（本 ADR による確定）
- ISR では `Transition` / `Admission` / `Publication` は**別 Layer**。
- したがって stale discard を Transition Completion に置くのは設計上の偶然であり、Layer 分離後は不適切。
- generation + sequence による拒否判定は **`submitPublishRequest()` の Admission (`PublicationAdmission::evaluate`) 層**に属する。
- **`evaluateDeferred` は AudioEngine に依存しない（第11回 ISR レビュー反映・2026-08-06 追加）。**
  Policy は Engine 全体を参照せず、**`DeferredAdmissionSnapshot`**（`currentGeneration` /
  `lastSequence` / `shutdownInProgress` / `nowUs` の4値のみ）を受け取って判定する。
  エンジン状態の取得（`rebuildRequestGeneration` / `getLastCommittedPublicationSequence` /
  `isShutdownInProgress`）と現在時刻（`getCurrentTimeUs()`）は呼び出し元（RebuildThread）
  がコンテキストに詰めて渡す（第12回反映で nowUs も完全スナップショット化）。
  これにより ISR の「Policy は RuntimeState / Snapshot のみ読む」を満たし、
  Policy → Engine の依存肥大化を防ぐ。※ 既存 `evaluate` は `AudioEngine&` +
  `RuntimeReaderContext&` を取るが、これは既存契約であり新規 API（evaluateDeferred）の
  Engine 非依存を妨げない。
- **「Policy はスナップショットのみを受け取る」を設計原則として確定（第13回 ISR レビュー反映・
  2026-08-06 追加）。** 層分離は `Engine ──► Snapshot ──► Policy`。Policy への入力は
  ① `DeferredPublishMetadata`（enqueue 時点の immutable スナップショット）と
  ② `DeferredAdmissionSnapshot`（peek 時点の Observation Snapshot）の**全てスナップショット**
  であり、Policy が Engine の生状態・生時刻・Slot 構造を直接読むことは決してない。
  現行 `metadata()` の const& は「スナップショットへの const 参照」（下記契約に基づく）。
  **将来スレッドモデルが拡張されても「Policy はスナップショットのみ」は維持する** —
  値返しにする等、渡し方だけ変更する。
- **`DeferredAdmissionSnapshot` は唯一の Observation Snapshot PODとする（第13回反映①）。**
  将来 engine-state（maintenance mode / publish paused / validation mode 等）を Policy へ
  渡す場合は**この1つの POD へフィールドを拡張**する。`getCurrentTimeUs()` の散在（全体95箢）
  は防避可能 — Policy は渡された snapshot のみ読む。enqueue-timestamp（`DeferredPublishMetadata`）
   と evaluate-nowUs（`DeferredAdmissionSnapshot`）は**意図的に2つの immutable 取得時点**である。
   ※ **Observation Entry-Point 契約（将来拡張用）**: 新しい Engine-state（publish-paused / maintenance / offlineRendering / validation-mode 等）を Policy へ渡す場合は**必ずこの `DeferredAdmissionSnapshot` へフィールドを追加**すること。Policy は「渡された snapshot のみ読む」を維持するため、別の入口を増やすと Snapshot 整合性が分裂し ISR 原則が崩れる。`getCurrentTimeUs()` の散在（全体95箆、design-D4 D-3 棚却）はここで一点解決する。
- **TTL（30秒）は Phase 0/1 は `constexpr` 固定（第13回反映②）。** 将来的な `PolicyTTLUs`
  （Offline / Realtime / Network-Sync での設定可能 TTL）への拡張余地を残す。
  現行 `kDeferredPublishTTLUs` は `.h:48` 定義のみ（実強制は design-D4 A-4 / D-9）。
- **`DeferredPublishView` は move-only とする。** View は stack 上で短命に使用し、consume/discard 完了後は再利用されない（`std::vector<DeferredPublishView>` のような誤用防止）。
- **`metadata()` は const 参照を返す（第7回レビュー反映③）。** DeferredSlot は **Single Thread Owner（RebuildThread）**契約（enqueue / peek / consume / discard が同一スレッドに閉じる）。peek → evaluate → consume の窓で他スレッドが slot を改変する経路はなく、値コピー（スナップショット保証）は不要。**契約: peek/consume/discard は RebuildThread 専用・他スレッド参照禁止**。契約を破る場合は値返しに戻すこと。
  - **※ 第11回反映（2026-08-06）: 本契約は「現行スレッドモデル限定」であることを明文化する。**
    const& 返却の安全根拠は Single Thread Owner だけではない: **(a) View が slot の
    寿命を保証する（View = slot 所有権 Authority）+ (b) consume/discard 後の metadata
    呼び出し禁止（state_ ガード）+ (c) Single Thread Owner 契約（View 寿命中の slot
    非変更）** の組合せで成立する。**将来スレッドモデルが変わる場合（Coordinator /
    Timer が slot を読む等）は、metadata() を値返しに戻すことを設計契約として固定する。**
  - **※ 第13回反映（2026-08-06）: `metadata()` は上記スナップショット原則のもと
    「enqueue 時点の immutable Snapshot（`DeferredPublishMetadata`）への const 参照」で
    ある。** 渡し方（const& か値か）は契約で変わり得るが、**Policy に渡るものが
    Snapshot であることは将来も不変**。
- **Single Thread Owner 関係図（第14回 ADR 追記）:**
  Single Thread Owner (RebuildThread)
    ↓ 設計契約: peek/consume/discard は RebuildThread 専用・他スレッド参照禁止
    ↓ Debug Assert: `jassert(std::this_thread::get_id() == engine_.rebuildThreadId())`
    ↓ Release assumes contract (const& 返却 / non-atomic slot / atomic 不要)
  - Phase 1 **必須** (debug code ではなく、設計契約を実行時検証する Guard)。Timer/Coordinator/Recovery-worker が peek を始める = 契約違反 = fail-fast。
- **Slot 所有権プロトコル（第11回反映・2026-08-06 追加）:** 有効な `DeferredPublishView`
  は同時に高々1つ。所有権状態は Orchestrator 側の **DEBUG 専用
  `DeferredSlotOwnership slotOwnership_`**（enum: `Released` / `Borrowed`）で管理する。
  bool ではなく enum にするのは assert メッセージ・ログで状態を判別するため
  （Semantic Single Source）。`atomic` は不要（Single Thread Owner 契約で並行経路なし）。
  解放は consume()/discard() または ~DeferredPublishView()（state_==Valid のときのみ）。
  この所有権状態は「View が slot 寿命を保証する」(a) の実装側裏付けである。
  ※ **Ownership Protocol は Single-Slot-by-design。** 現行は 1:1 `DeferredPublishView`↔`DeferredSlot`。Multi-slot Queue 化（複数 slot 並列）の場合は 1:1 プロトコルを捨て、**generation-keyed iterator**（View は `generation`+`index` で参照し lock/atomic を伴う）へ切り替えること。Single-slot 前提の `metadata() const&` / `non-atomic slot` は Multi-slot 化の瞬間に即座に無効化されるため、プロトコル切り替えは**設計契約違反ではなく**architectural boundary として明示すること。
- **`DeferredPublishView` のデストラクタは暗黙 discard しない（第7回レビュー反映⑥）。** consume / discard の呼び出しは Caller の責務。どちらも呼ばずに**Valid**のまま破棄された場合は**DEBUG assert**（fail-fast・第13回反映③）—`peek()→evaluate()→consume/discard()` 以外のルートはないため peek-only は意図的欠陥。slot の所有権は destructor 内で reset される（リークなし）が、Valid 寿命はバグとして即検出する。（※ §99 旧方針「slot 残留・再 peek」を fail-fast へ更新。ADR-C4 104 Consequences 参照。）
- **Slot 所有権遷移は原子的（Atomic ownership transition・第13回反映④）。** `consume()`/`discard()` は `finishView()` を終端で内部原子呼出しし、state_ 遷移と ownership Release を**1つの不可分操作**として返す。caller（RebuildThread）は `view->consume()` **一回**で済み、中間状態を観測できない。（現行ソース `RebuildDispatch.cpp:844-848` の `consumeDeferredRequest()→submitPublishRequest()` 2段は legacy two-step。新設計（design-D4 A-3）で不変分けを閉じる。）

## Consequences
- **ADR design principle — Admission は判定専属 (Decision Authority は Admission / Store mutation は View):** `PublicationAdmission` は「判定」を担い、Deferred Storage の状態変更は `DeferredPublishView`（または Queue）側が実施する。Admission は Storage の Authority を持たず、**Discard 理由のみを決定**する。`evaluateDeferred()` は `DeferredAdmissionResult{decision, discardReason}` を返し Store を触らない -- Store 変更は `view->consume()` / `view->discard(reason)` で View が実施する（design-D4 A-4 / D-13.6 確認済み）。将来 Admission が `discard()`/`consume()` を直接呼ぶコードが混入した場合は**設計契約違反**として fail-fast assert で検出する。
- `notifyTransitionComplete` / `DSPTransition::onTransitionComplete` は呼び出し元ゼロ (dead code 化; `onTransitionComplete` は今後の Layer 2/3 統合で整理予定)。**ソース確認:** `onTransitionComplete`（`DSPTransition.h:131-164`）はコメントで「呼び出し元ゼロ・publishIdleWorldOnly() に置換済み」と自記。crossfade 完了 publish 経路は `publishIdleWorldOnly()` → `submitObserve`→Intent であり、`IntentHandlerContext`（`ISRIntentDispatcher.h:21`）＋**無状態 handler singleton** `g_*IntentHandler`＋`kDispatchTable`（`static_assert` ガード, HANDLER-1）により **handler は publish を直接実行しない** — `CrossfadeCompleteIntentHandler` も委譲のみ（第13回反映⑤）。
- `RuntimePublicationOrchestrator::transition_` / `transition()` は ProcessIntent から使用継続 (削除しない)。
- stale discard 実装を Admission 層へ統合する際は、本 ADR の判定フローを再実装の仕様とする。
- `processDeferredAdmission()`（旧 `consumeDeferredRequest() → submitPublishRequest()` ハンドオフ）は **Coordinator/RebuildThread 専用** (Single Thread Owner, 第13回確認①)。Phase 1 実装では `peekDeferred`/`processDeferredAdmission`/`releaseDeferredSlot` に `jassert(std::this_thread::get_id() == engine_.rebuildThreadId())`（=`engine_.rebuildThread.get_id()` / AudioEngine.h:2525）スレッドガードを付与し、Timer/Coordinator/Recovery-worker が peek を始めることを**コンパイル/assertレベルで禁止**する。`hasDeferred_`(atomic) → `deferredSlot_`(non-atomic) の handshake はこの前提の上成り立つ。`enqueueDeferred`/`submitPublishRequest` は Audio Thread（trySubmit）からも到達し得るため**意図的にガード対象外**（lock-free/atomic）とする。`recordProgress` の generation uint64_t↔int テレメトリ widening は ISR-Authority と無関係のため**この変更セットには含めない**（TODO は design-D4 D-13.6 TODO へ）。詳細は design-D4 D-13.6。

## Testing Principle（第13回 ISR レビュー反映・2026-08-07）

**Objects that enforce RebuildThread ownership shall be verified inside AudioEngineHarness.
Standalone unit tests are reserved for pure policy components.**

- RebuildThread ownership（jassert スレッドガード付き）を強制するオブジェクト
  （`DeferredPublishView` / `RuntimePublicationOrchestrator` の deferred 系 API）は
  **AudioEngineHarness 上の Integration Test** で検証する（実経路 + 未ガード atomic
  観測 API + 型レベル static_assert）。
- **standalone `main()` 単体テスト**（`add_executable` / `add_test`）は**純粋 Policy**
  （`PublicationAdmissionTests` の `evaluateDeferred()` 等、入力を入れれば出力が決まる
  副作用なしコンポーネント）のみに予約する。
- 根拠: `DeferredPublishView` は `finishView()` → Orchestrator → AudioEngine の
  Authority チェーンの一部であり、Fake Orchestrator / Fake AudioEngine で単独化すると
  **Authority を偽装**することになる。また consume/discard/finishView を他スレッドから
  呼ぶことは Single Thread Owner 契約違反そのものであり、standalone テストは最初から
  契約違反を前提とする。
- 実装: `src/tests/AudioEngineHarness/DeferredPublishViewStateMachineTests.cpp`
  （`runDeferredPublishViewStateMachineTests()`）+ `DeferredFlowIntegrationTests.cpp`。
  設計: design-D4 不変条件8の検証マッピング表（第13回反映）を参照。