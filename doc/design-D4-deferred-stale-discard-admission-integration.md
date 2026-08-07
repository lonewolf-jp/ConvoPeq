# 設計: stale discard の Admission 統合 & onTransitionComplete 整理

**Status:** Design (not implemented)
**Date:** 2026-08-06（第11回レビュー反映・ISR 観点の3点修正・総合 A）
**前提:** ADR-C4 (DeferredPublicationEvolution) — `notifyTransitionComplete` 撤去・stale discard は Admission Policy 責務と確定済み。

---

## 0. 設計概要（実装者向け）

### 目的

- **Part A**: 現状 `consumeDeferredRequest` が判定を持たない deferred publish の
  stale discard（TTL / generation / sequence）を `PublicationAdmission`（Policy 層）へ
  正式統合する。ADR-C4 で確定済みの責務割当（「stale discard は Admission Policy 責務」）の実装。
- **Part B**: 呼び出し元ゼロの `onTransitionComplete` を整理し、最終形は Intent 化
  （CrossfadeCompleteIntent）で Coordinator へ一元化する。

### 変更の要点

| 項目 | 設計 | 実装種別 |
|------|------|---------|
| 判定 | `PublicationAdmission::evaluateDeferred(const DeferredPublishMetadata&, const DeferredAdmissionSnapshot&)` → `DeferredAdmissionResult { decision, discardReason }`（Decision は動作指示のみ・理由は DiscardReason が分離。**Policy はスナップショットのみを受け取る・第11回反映・第12回で完全スナップショット化（nowUs 含む）・第13回で `DeferredAdmissionSnapshot` へ命名（Observation Snapshot）**） | 新規 |
| Queue API | `peekDeferred()` → `std::optional<DeferredPublishView>`。consume / discard は View 経由のみ。**View は move-only + `State` enum（Valid/Consumed/Discarded/MovedFrom）+ Slot 所有権ライフサイクル（第10回・第11回反映・第12回で `finishView()` により Authority を Orchestrator 側に集約）**。`metadata()` は **const 参照**（Single Thread Owner 契約・第7回反映③。ADR-C4 に契約明記） | 新規 |
| Storage | `DeferredPublishTypes.h`（`DeferredPublishSlot`（request + metadata）/ `DeferredPublishMetadata` / `DeferredDiscardInfo` 独立 POD）。**`lastDiscardReason` は持たない** | 新規ヘッダ |
| Telemetry | `DeferredDiscardInfo` → TelemetryRecorder の Adapter（`recordDeferredDiscard`）で `DeferredHealth` へ変換・記録。enqueue 時は `recordDeferredEnqueue`。※ Event 化（Coordinator → Event → Recorder）は将来課題（第7回反映④・Appendix D-7） | 新規（Adapter）/ 将来（Event） |
| Part B (Phase 1) | `finalizeFadingSlot()` ヘルパー（内部で Step 1: CAS 終端 / Step 2: observe 投入 に2段分離） | 新規 |
| Part B (Phase 2/3) | `CrossfadeCompleteIntent`（IntentType 追加 + dispatch 登録 + 無状態 Handler。Payload は `DSPHandle` + `CrossfadeId`） | 将来 |

### 実装フェーズ

| Phase | 内容 | 今回スコープ |
|-------|------|-------------|
| Phase 0 | `onTransitionComplete` 削除（呼び出し元ゼロ確認済み） | 対象外 |
| Phase 1 | `finalizeFadingSlot()` 抽出、Timer の CAS ボイラープレート置換 | 対象 |
| Phase 2/3 | CrossfadeCompleteIntent 導入・publish 集約 | 将来 |

### 不変条件（実装時に必ず守る）

1. **consume / discard は `peekDeferred()` が返す View からのみ呼ぶ** — 順序は型で強制。
   `peek` なしに consume するコードは書けない。
2. **consume = 所有権移転・slot 即解放**。consume 後に slot を保持しない（「2つの真実」禁止）。
3. **`evaluateDeferred` は副作用なし**。metadata は const 参照（安全性は
   **(a) View が slot 所有権を握る + (b) consume/discard 後の metadata 禁止 + (c)
   Single Thread Owner（RebuildThread）契約**の組合せ、第7回反映③・第9回反映①）で
   渡り、Queue 内部構造（Slot）を知らない。
   ★ **Policy は AudioEngine 全体に依存しない（第11回反映）**: `evaluateDeferred` は
   Engine 参照ではなく `DeferredAdmissionSnapshot`（currentGeneration /
   lastSequence / shutdownInProgress / **nowUs** の4値のみ）を受け取る。
   ★ **TTL 判定も完全スナップショット化（第12回反映）**: 現在時刻 `nowUs` もコンテキスト
   に含め、Policy は `getCurrentTimeUs()` を直接呼ばない。これにより Policy は
   **決定論的**（渡された値のみで判定）になり、ISR の「Policy は Runtime State /
   Snapshot のみ読む」を完全に満たす。時刻取得は呼び出し元（RebuildThread）が
   `core/TimeUtils.h::getCurrentTimeUs()` で行いコンテキストに詰める。
   現行 `evaluate` が `AudioEngine&` + `RuntimeReaderContext&` を取るのは既存契約であり、
   `evaluateDeferred` は新規 API として「Engine 非依存・完全スナップショット」を
   最初から満たす（ISR の Policy = RuntimeState のみ読む、を先取り）。
   ★ **Policy はスナップショットのみを受け取る（第13回反映・設計原則）**: Policy への
   入力は**全てスナップショット**である。① `DeferredPublishMetadata`（enqueue 時点の
   immutable スナップショット）と ② `DeferredAdmissionSnapshot`（peek 時点の Observation
   Snapshot）。層分離は `Engine ──► Snapshot ──► Policy` であり、Policy が Engine の
   生状態・生時刻・Slot 構造を直接読むことは決してない。現行 `metadata()` の const& は
   (a)(b)(c) 契約に基づく**スナップショットへの const 参照**であり、将来スレッドモデルが
   拡張されても「Policy はスナップショットのみ」は維持する（値返しにする等、渡し方だけ
   変える）。→ ADR-C4 に同旨を記録。
4. **Queue は Telemetry 型を知らない**。`view.discard` は `DeferredDiscardInfo`（純粋データ）を
   返すのみ。`DeferredHealth` への変換・記録は TelemetryRecorder 側の Adapter が担う。
5. **`DeferredDecision` は「動作指示」のみ**（Ready / Discard）。破棄理由は
   `DiscardReason`（既存 enum）が分離して保持する。将来の Decision 追加で enum を肥大化しない。
   `RetryLater` は利用経路ゼロ（YAGNI）のため導入見送り（第6回レビュー反映③）。
6. **`DeferredPublishView` は move-only** — copy 禁止・stack 上で短命に使用。
   consume/discard 後は再利用しない。View の状態は `bool` ではなく
   `State` enum（`Valid` / `Consumed` / `Discarded` / `MovedFrom`）で管理する
   （MovedFrom は move 後の状態。第9回反映）。
7. **Storage は Telemetry 用フィールドを持たない**。`DeferredPublishSlot` は
   request + metadata のみ。破棄理由は `DeferredDiscardInfo` にのみ載る。
8. **`DeferredPublishView` のデストラクタは暗黙 discard しない**（第7回反映⑥）。
   View 生成後に consume も discard も呼ばれずに寿命を迎えた場合、slot は**そのまま
   残る**（＝次回 peek で再評価される）。「デストラクタが破棄を肩代わりする」暗黙の
   ライフサイクルは持たない。ただし **Slot 所有権（Ownership）はデストラクタが解除する**
   （第10回反映・第11回で所有権プロトコルへ整理。下記状態遷移図・不変条件 9）。
   ライフサイクルは下記の状態遷移図のとおり。
   ```text
   View 生成（peekDeferred()）— Slot 所有権取得（Ownership = Borrowed）
        │
        ▼
      Valid ── consume() ──► Consumed  （request 取出し・slot 解放・Ownership = Released）
        │
        ├── discard(reason) ──► Discarded（DeferredDiscardInfo 生成・slot 解放・Ownership = Released）
        │
        ├── move される ──► MovedFrom（owner_ = nullptr。公開API呼び出し禁止）
        │                      （Ownership は移転先 View が引き継ぎ = Borrowed のまま）
        │
        └── 何も呼ばず破棄（デストラクタ）
            └─► slot は残る（次回 peek で再評価）。Ownership のみ解除（Released）
   ```
    各メソッドは `state_ == State::Valid` のときのみ有効。`Consumed` / `Discarded` /
   `MovedFrom` 後の呼び出しは nullopt を返すか assert で検出する
   （多重呼び出し・consume→discard の両立を防御）。

   **状態遷移表（全遷移の自動検証対象・第12回反映）:**

   | 現在状態 | 操作 | 次状態 | slot の寿命 | Ownership | 結果 / 契約 |
   |---|---|---|---|---|---|
   | —（Released） | `peekDeferred()` | Valid（新 View） | 保持 | Released → Borrowed | 成功。slot 無しなら nullopt |
    | Valid | `consume()` | Consumed+Released（**atomic**） | 解放（request 取出し） + ownership Released を1操作 | `PublishRequest` を返す。`finishView()` は consume() の終端原子操作として内部呼出し（第13回④）。呼び出し側に中間状態の隙はない（no caller-visible gap） |
    | Valid | `discard(reason)` | Discarded+Released（**atomic**） | 解放（reset） + ownership Released を1操作 | `DeferredDiscardInfo` を返す。finishView() は discard() の終端原子操作として内部呼出し |
   | Valid | `metadata()` | Valid（不変） | 保持 | 不変 | `const&` を返す（寿命は View 生存に一致） |
   | Valid | move コンストラクタ | self → MovedFrom / dest → Valid | dest へ移転 | Borrowed のまま（dest へ） | dest のみ有効。self の公開 API は禁止 |
   | Valid | move 代入（dest が Valid） | dest 旧 slot は Consumed 相当で解放 | dest 旧 slot 解放 | dest 旧: → Released | 自己代入は `this != &other` でガード |
    | Valid | デストラクタ | （終了） | 解放（reset） | → Released | **DEBUG assert（state_==Valid）** — peek 取得後 consume/discard 忘れを即検出（fail-fast・第13回③）。slot は reset するが、**Valid のままでの寿命は意図的欠陥**。`peek()→evaluate()→consume/discard()` 以外のルートはないため、peek-only は合法でない |
   | Consumed | `consume()` / `discard()` | Consumed（不変） | — | — | nullopt |
   | Consumed | `metadata()` | — | — | — | 契約違反（DEBUG で assert） |
   | Consumed | デストラクタ | （終了） | — | 何もしない（解除済み） | OK |
   | Discarded | `consume()` / `discard()` | Discarded（不変） | — | — | nullopt |
   | Discarded | `metadata()` | — | — | — | 契約違反（DEBUG で assert） |
   | Discarded | デストラクタ | （終了） | — | 何もしない（解除済み） | OK |
   | MovedFrom | `metadata()` / `consume()` / `discard()` | — | — | — | 契約違反（DEBUG で assert・release は UB） |
   | MovedFrom | デストラクタ | （終了） | — | 何もしない（所有権は移転先） | OK |

   ★ **この表を Phase 1 実装時にテストへ落とし込む**（第12回反映）: 行の各遷移を
    自動検証する。**★ 第13回反映（2026-08-07）: 検証方式を修正。** 本表は
    `DeferredPublishView`（→ `finishView()` → `RuntimePublicationOrchestrator` →
    `AudioEngine`）の Authority チェーンを構成し、さらに peek/consume/discard/
    finishView に `jassert(thread == rebuildThreadId())` のスレッドガードがあるため、
    **standalone `main()` 単体テストは成立しない**。したがって:
    - **AudioEngineHarness 上の Integration Test として検証する。**
      `src/tests/AudioEngineHarness/DeferredPublishViewStateMachineTests.cpp`
      （`runDeferredPublishViewStateMachineTests()`）を AudioEngineHarness
      （CMake の `add_executable(AudioEngineHarness ...)`）に登録し、既存
      `DeferredFlowIntegrationTests.cpp` と同一 exe で実行する。
    - **RebuildThread 専有の遷移**（Released→Valid: peek / Valid→Consumed+Released:
      consume / Valid→Discarded+Released: discard）は実経路（requestRebuild →
      CoordinatorLoop → RebuildThread.processDeferredAdmission）を駆動する
      `DeferredFlowIntegrationTests.cpp` が実行時検証し、本ファイルでは重複しない。
    - **本ファイルの検証範囲**: 型レベル契約（copy 禁止 / move-only の static_assert）、
      default ctor → MovedFrom、MovedFrom 間の move 系、Valid 行の観測
      （`metadata()` const& 同一性 / state / isValid）。Valid View は fail-fast
      デストラクタのためテストスレッドから破棄不可 → **heap リークで観測**
      （意図的・tiny・プロセス終了時に OS 回収）。契約違反行
      （Consumed/Discarded/MovedFrom 後の公開 API 呼び出し）と Valid のままの
      デストラクタ（fail-fast）は **コード検査で担保**（各 jassert 実装を検査）。
    - ★ 特に「Consumed/Discarded/MovedFrom 後の公開 API 呼び出し = 契約違反」と
      「Valid のままのデストラクタ = slot は残る + Ownership のみ Released」は
      不変条件の要（2つの真実禁止）。**Testing Principle（ADR-C4 追記・第13回反映）**:
      RebuildThread ownership を強制するオブジェクトは AudioEngineHarness 上で検証する。
      Standalone main() 単体テスト（add_executable/add_test）は純粋 Policy
      （`PublicationAdmissionTests` = `evaluateDeferred`）のみに予約する。
9. **有効な `DeferredPublishView` を同時に複数保持しない**（第9回反映・第10回で借用
   ライフサイクルを明文化・第11回で**Slot 所有権プロトコル**へ整理）。`auto a =
   peekDeferred(); auto b = peekDeferred();` は契約違反 — single slot の所有権が分かれる。
   所有権解放規約:
    ```text
    peek 取得成功             : Ownership = Borrowed（Orchestrator 側 slotOwnership_ を検査→設定）
    consume() / discard()     : Ownership = Released（owner_->finishView() を呼ぶ・第12回反映）
    ~DeferredPublishView()    : state_ == Valid のときのみ Ownership = Released
                                （Consumed/Discarded は consume/discard が既に解除、
                                  MovedFrom は移転先が引き継いでいるため何もしない）
    ```
    ★ **Authority 集約（第12回反映）**: View は `owner_->finishView()` を呼ぶだけであり、
    所有権フラグの書き換えは Orchestrator 側の `finishView()`（内部で
    `releaseSlotOwnership()`）が行う。「View が Owner を書き換える」のではなく
    「Owner が Owner を管理する」（ISR の Authority Singularization）。
    実装は Orchestrator 側に **DEBUG 専用の `DeferredSlotOwnership slotOwnership_`**
    （enum: `Released` / `Borrowed`）を持ち、View 生成時に検査して2枚目（所有中の再
    peek）を assert で検出する。bool ではなく enum にする理由は、assert メッセージ・
    ログで Slot の所有権状態が判別できるようにするため（**Semantic Single Source**）。
    View は Slot の**所有権 Authority** を握る（Ownership も Authority・ISR 第11回観点。
    `DeferredToken` 命名は将来検討・現行は View で十分）。**atomic は不要**（Single
    Thread Owner 契約で並行アクセス経路が存在しない）。release ビルドでは存在しない
    （契約違反 = UB・既存方針と同一）。
10. **`DeferredPublishView` に状態照会 API（`valid()` 等）を追加しない**（第9回反映）。
    公開 API は `metadata()` / `consume()` / `discard()` の3つのみ。
    状態分岐を書かせずプロトコル中心の設計を維持する。状態確認は内部実装と
    assert に留める。

### 主な実装影響ファイル

| ファイル | 変更 |
|---------|------|
| `RuntimePublicationTypes.h` (新規) | `PublishRequest` を独立定義（PublicationAdmission ネスト型から分離・第8回反映①） |
| `DeferredPublishTypes.h` (新規) | Storage POD（`DeferredPublishSlot`（request + metadata）/ `DeferredPublishMetadata` / `DeferredDiscardInfo`） |
| `PublicationAdmission.h/.cpp` | `DeferredDecision` / `DeferredAdmissionResult` / `evaluateDeferred` / `kDeferredPublishTTL` 移設（chrono 型） |
| `TelemetryRecorder.h/.cpp` | `recordDeferredDiscard(DeferredDiscardInfo)` / `recordDeferredEnqueue(overwriteCount)` Adapter（DeferredHealth 変換） |
| `RuntimePublicationOrchestrator.h/.cpp` | `peekDeferred` / `DeferredPublishView`（move-only + State enum）/ `clearDeferredForShutdown`・`enqueueDeferred` の Telemetry を Adapter へ移譲 |
| `AudioEngine.RebuildDispatch.cpp` | A-5 の consume 箇所を peek → evaluate → consume に |
| `AudioEngine.h` / `AudioEngine.Timer.cpp` 他 | Part B: `finalizeFadingSlot`（Step 1 は `finalizeFadingTermination`・第8回反映②） |
| `src/tests/AudioEngineHarness/DeferredPublishViewStateMachineTests.cpp` (新規) | 不変条件 8 の状態遷移表を自動検証（第12回反映）。**第13回反映（2026-08-07）: standalone main() ではなく AudioEngineHarness 上の Integration Test として実装**（`runDeferredPublishViewStateMachineTests()` → AudioEngineHarness exe に登録）。型レベル契約（static_assert）+ default/MovedFrom 状態機械 + Valid 行観測（metadata const& 同一性）を検証。RebuildThread 専有遷移は DeferredFlowIntegrationTests が実経路で検証（重複なし）。契約違反行・fail-fast 行はコード検査で担保 |

## Part A. stale discard の Admission 層への正式統合設計

> **現状分析・設計方針の詳細**（欠けている3点・「peek → evaluate → consume」の判断根拠・
> evaluateDeferred の Entry Point 判断）は **Appendix B-1 / B-2** を参照。

### A-3. 提案する構造（Storage は独立POD、判定は PublicationAdmission、consume は純粋な Queue）

```
// DeferredPublishTypes.h — 共通POD（Storage 型。Queue のデータ構造）
// ★ Policy の所有物にしない。Admission と Orchestrator の両方が include する。
//   将来 Deferred Queue を拡張（Persistent/Priority）しても Storage 型はここに集まる。
//
// ★ Storage は Policy（PublicationAdmission）を知らない（第8回レビュー反映①）。
//   DeferredPublishSlot が保持する PublishRequest は PublicationAdmission のネスト型
//   ではなく、RuntimePublicationTypes.h の独立型を参照する（依存方向は
//   Storage → RuntimePublicationTypes、Policy → RuntimePublicationTypes）。
//   これにより「Storage が Policy を所有する」見た目の依存が構造的に排除され、
//   将来の Persistent Queue 化で Storage が Policy から独立したまま拡張できる。

// 判定に必要な最小限のメタデータ（PublishRequest 自体は含まない）
struct DeferredPublishMetadata {
    // ★ generation は int（第13回調査確定）: PublishRequest.generation（int）と
    //   rebuildRequestGeneration（std::atomic<int>・AudioEngine.h:2372）が同一型であり、
    //   ctx.currentGeneration（int）との比較で符号付き/なし警告（C4018/-Wsign-compare）を
    //   出さない。現行 DeferredGuard の uint64_t はキャストの産物で、int に統一して
    //   static_cast を廃止する（enqueue 側の現行コードは
    //   `static_cast<uint64_t>(req.generation)` で保存しているが新設計では不要）。
    int generation = 0;                 // enqueue 時点の generation（int・req と同一型）
    PublicationSequenceId sequence = 0; // enqueue 時点の commit 済み sequence
    uint64_t enqueueTimestampUs = 0;    // TTL 判定用
};

// Queue の格納型（request は PublishRequest、判定用情報は metadata）
// ★ Telemetry 用フィールドを持たない: 破棄理由は discard() が生成する
//   DeferredDiscardInfo にのみ載る。slot は discard 直後に reset() される（→ Appendix B-2）。
// ★ PublishRequest は RuntimePublicationTypes.h の独立型
//   （PublicationAdmission のネスト型ではない。第8回レビュー反映①）。
struct DeferredPublishSlot {
    PublishRequest request;          // convo::isr::PublishRequest（RuntimePublicationTypes.h）
    DeferredPublishMetadata metadata;
};
```

```
// RuntimePublicationTypes.h（新規）— PublishRequest の独立定義
// ★ PublicationAdmission のネスト型から分離（第8回レビュー反映①）。
//   PublicationAdmission（Policy）・DeferredPublishTypes（Storage）の両方が参照する
//   中立の型定義。Policy が Storage を持つ依存も、Storage が Policy を持つ依存も
//   発生しない（いずれも本ヘッダへ依存）。
// ★ DSPHandle / RuntimeBuildSnapshot / BuildAnalysis / OversamplingResult /
//   BuildDiagnostics は既存ヘッダ（ISRDSPHandle.h / RuntimeBuildTypes.h）から include。
namespace convo::isr {
struct PublishRequest {
    DSPHandle newDSP;
    int generation = 0;
    RuntimeBuildSnapshot sealedSnapshot;
    BuildAnalysis buildAnalysis {};           // ★ v14.0: Auto Gain 解析値
    OversamplingResult oversamplingResult {}; // ★ v14.38
    BuildDiagnostics buildDiagnostics {};     // ★ v14.37
};
}
```

```
// PublicationAdmission.h — Policy 層に判定を集約（evaluate と同一インスタンス）
// ★ metadata のみを受け取る。Admission は Queue 内部構造（Slot）を知らない。
// ★ 二層構造: Decision は「どうするか」（動作指示）のみ。破棄理由は既存の
//   DiscardReason（RuntimePublicationState.h:10-16）が分離して保持する（→ Appendix B-2）。
// ★ PublishRequest は RuntimePublicationTypes.h の独立型（第8回レビュー反映①）。
//   PublicationAdmission は型の定義を所有せず、参照のみ。
// ★ 将来リネーム余地（第12回反映）: 本クラスは publish 判定全般（evaluate /
//   evaluateDeferred / 将来的な evaluateRecovery 等）を担うため、責務が増えるに
//   つれて `PublicationPolicy` へリネームする余地を残す。内部 Policy 分割
//   （evaluate 系 = PublicationAdmission、他 = 別 Policy）が必要になった場合は
//   この境界から分割する。**現時点ではリネームしない**（既存コードの
//   `PublicationAdmission` 参照を増やさない。Phase 1 実装中に境界が固まったら
//   単独コミットで実施）。

// DeferredAdmissionSnapshot — evaluateDeferred が参照する最小 Runtime 状態（Observation Snapshot）
// ★ 第13回レビュー反映: 「Snapshot」という命名・意味付けを明確化。ISR の層分離では
//     Engine ──► Snapshot ──► Policy であり、Policy が読むのは**観測時点の不変な
//     Snapshot** のみ（Engine の生状態ではない）。
// ★ Policy が AudioEngine 全体へ依存しないための読み取り専用コンテキスト
//   （第11回レビュー反映）。evaluateDeferred が読むのは「現在 generation / 最後の
//   commit sequence / shutdown 中か / 現在時刻」の4つのみ。これらを呼び出し元（RebuildThread）が
//   engine のアクセサ + getCurrentTimeUs() で都度構築して渡す（peek → evaluate の窓で
//   単一スレッド内・atomic 読みで整合。peek 時点の観測スナップショット）。
//   ★ nowUs も含めることで Policy は getCurrentTimeUs() を直接呼ばない＝完全スナップショット
//     判定（決定論的。第12回レビュー反映）。ISR の「Policy は RuntimeState / Snapshot のみ
//     読む」を完全に満たす。時刻取得の Authority は呼び出し元側に置く。
//   ★ 現行 evaluate は AudioEngine& + RuntimeReaderContext& を取るが、これは既存契約。
//     evaluateDeferred は新規 API として Engine 非依存を最初から満たす（ISR の
//     「Policy は RuntimeState / Snapshot のみ読む」を先取り。ADR-C4 に決定を記録）。
struct DeferredAdmissionSnapshot {
    int currentGeneration = 0;                   // engine.rebuildRequestGeneration（AudioEngine.h:2372）
    PublicationSequenceId lastSequence = 0;      // engine.getLastCommittedPublicationSequence()（AudioEngine.h:1592）
    bool shutdownInProgress = false;             // engine.isShutdownInProgress()（AudioEngine.h:1460）
    uint64_t nowUs = 0;                          // 判定時点の現在時刻（getCurrentTimeUs() を
                                                 //   呼び出し元が詰める。TTL 判定に使用）
};

enum class DeferredDecision {
    Ready,       // 有効 → view.consume() へ進む
    Discard,     // 破棄 → view.discard(reason) へ（理由は evaluateDeferred が返す）
    // ★ Ready / Discard の2値のみ（RetryLater は利用経路ゼロ・YAGNI のため見送り。
    //   将来 Queue 多段化等で必要になった時点で追加。→ Appendix B-2）
};

// 判定結果: 動作指示 + 破棄理由（Discard 時のみ discardReason が有効）。
// ★ 責務分離: この struct は「判定」のみ。Store 変更は Caller が view.consume() / view.discard(reason) で実施する
//   （ADR principle: Admission = Decision only, View = Store mutation; ADR-C4 §Consequences）。Admission は
//   Storage Authority を持たない。→ design-D4 A-4 フロー (evaluate → view.consume/discard) と対応。
struct DeferredAdmissionResult {
    DeferredDecision decision{DeferredDecision::Discard};  // 明示的に設定される（初期値は Discard）
    DiscardReason discardReason{DiscardReason::None};  // decision==Discard のとき有効
};

// 既存 evaluate() に追加（同じ Policy 層、ただし Engine 非依存）
// evaluate(req, engine&, ctx) とは異なり、AudioEngine 参照を取らない — 最小コンテキスト
// （DeferredAdmissionSnapshot）のみで判定する（第11回反映）。
// ★ 実装は private helper evaluateDeferredImpl() に置き、本 API は薄く公開する
//   （Policy は1つ・Entry Point は複数可。→ Appendix B-2）。
[[nodiscard]] DeferredAdmissionResult evaluateDeferred(const DeferredPublishMetadata& metadata,
                                                       const DeferredAdmissionSnapshot& ctx) noexcept;
```

```
// RuntimePublicationOrchestrator.h — Queue 操作（純粋。判定・Policy・Telemetry を持たない）
[[nodiscard]] bool hasDeferredRequest() const noexcept;   // CoordinatorLoop 通知用（維持）

// ★ Slot 所有権プロトコル（第10回反映・第11回で所有権概念へ整理）: 有効な
//   DeferredPublishView は同時に高々1つ（不変条件 9）。View 自身は owner_ と state_
//   しか持たないため、Slot の所有権状態は Orchestrator が保持する。
//   ★ bool ではなく enum にする理由: 「借用中か」を bool で持つと assert メッセージ・
//     ログが「0/1」しか出ず、Slot の所有権状態（Released / Borrowed）が判別できない。
//     enum は Semantic Single Source に近い（ISR・第11回観点）。View は Slot の
//     所有権 Authority を握る（Ownership も Authority。DeferredToken 命名は将来検討・
//     現行は View で十分）。
//   ★ Single Thread Owner（RebuildThread）契約により並行アクセス経路が存在しないため
//     plain enum で十分（atomic 不要）。DEBUG ビルドでのみ存在: release では契約違反を
//     検出しない（違反 = UB・既存方針と同一）。
enum class DeferredSlotOwnership {
    Released,   // いずれの View も slot を所有していない（peek 可能）
    Borrowed,   // 有効な View（高々1つ）が slot を所有中
};
#ifndef NDEBUG
DeferredSlotOwnership slotOwnership_ = DeferredSlotOwnership::Released;
#endif

// Slot 所有権解除（private メンバー。呼び出しは Orchestrator の公開 API `finishView()` 経由。
//   `DeferredPublishView` は `owner_->finishView()` を呼ぶだけで、所有権フラグの書き換えは
//   行わない。★ 第12回反映: Authority を Orchestrator（Owner）側に集約する。
//   「View が Owner を書き換える」のではなく「Owner が Owner を管理する」。
//   ISR の Authority Singularization（ある資源の状態遷移は 1 つの Authority のみが握る）に一致。
//   アクセスは Orchestrator 側で `friend class DeferredPublishView;` を宣言して許可する。
//   方向は DeferredPublishView 内の `friend class RuntimePublicationOrchestrator;`（View の
//   private コンストラクタを Orchestrator が呼ぶ）と対を成す相互 friend）。
void releaseSlotOwnership() noexcept
{
#ifndef NDEBUG
    slotOwnership_ = DeferredSlotOwnership::Released;
#endif
}

// View のライフサイクル終了を Orchestrator へ通知する内部 API（第12回反映・第13回④ 強化）。
//   ★ Authority の方向: View → Orchestrator は「終了した」という通知のみ。
//     実際の所有権フラグ更新（releaseSlotOwnership()）は Orchestrator 自身が行う。
//   ★ **Atomic terminal step（第13回④）**: この関数は consume()/discard() の終端で
//     **内部原子呼出し**される。caller が finishView() を直接呼ぶことはない（public ではなく
//     View 専用）。state_ 遷移と ownership Release が1関数内で完結し、中間状態を外部に露出しない。
//   ★ consume / discard 後・および Valid のままのデストラクタで呼ばれる。
//     呼び出しは Single Thread Owner（RebuildThread）内に限る（既存契約）。
void finishView() noexcept
{
    releaseSlotOwnership();
}

// peek: 単一の DeferredPublishView を返す（slot が無ければ nullopt）。
//   ★ View パターン: consume/discard は peek で得た View にしか存在しないため、
//     「必ず peek → consume の順で呼ぶ」が API 型レベルで強制される。
//   ★ 同時取得禁止（第9回レビュー反映・第10回で所有権ライフサイクル確定・第11回で
//     所有権プロトコルへ整理）: 有効な View を複数同時に保持してはならない。
//     auto a = peekDeferred(); auto b = peekDeferred(); は契約違反 — single slot の
//     所有権が2つの View に分かれ、consume/discard の一意性が壊れる。
//     ★ Slot 所有権管理: 所有権状態は View ではなく Orchestrator 側の
//       `DeferredSlotOwnership slotOwnership_`（DEBUG 専用 enum・atomic 不要）が持つ。
//       取得成功時 = Borrowed に設定し、解放は consume()/discard()（→ Released）または
//       ~DeferredPublishView()（state_==Valid のときのみ → Released）のいずれかで行う
//       （規約詳細は不変条件 9）。2枚目（所有中の再 peek）は assert で検出
//       （release は契約違反 = UB）。
//   ★ スレッド所有権契約: peekDeferred()/metadata()/consume()/discard() は
//     RebuildThread 専用。DeferredSlot は他スレッドから参照してはならない
//     （理由・将来スレッド増時の値返し復帰条件は下記コメント）。
[[nodiscard]] std::optional<DeferredPublishView> peekDeferred() noexcept;

// DeferredPublishView — peek で得た slot への一時アクセス（所有権は view が握る）
//
// ★ Protocol View: 「peek → evaluate → consume」という API 利用順序
//   （Ownership Transition Protocol）を型で表現するための View。データアクセス用の
//   Storage View ではない（slot 個数には依存しない — 単一 slot でも「peek せずに
//   consume する」「consume を2回呼ぶ」がコンパイル時に書けない価値は不変。
//   → 判断根拠は Appendix B-2）。
//
// ★ スレッド所有権契約（metadata() const& 化の前提）:
//   deferred slot（DeferredSlot）は **Single Thread Owner（RebuildThread）**。
//   - enqueue（submitPublishRequest → enqueueDeferred）: RebuildThread のみ
//     （CoordinatorLoop は hasDeferredRequest()（atomic）で通知するのみ。slot に触れない）
//   - peek / metadata / consume / discard: RebuildThread のみ
//   - clearDeferredForShutdown: RebuildThread 停止後（Timer / CtorDtor / ReleaseResources）
//   → enqueue と peek/consume は同一スレッドで完結し、peek → evaluate → consume の
//     窓で他スレッドが slot を改変する経路は存在しない。
//   ★ const& 化の根拠はこの契約だけではない（第9回反映①）: 「View が slot 所有権を
//     握る（(a)）」+「consume/discard 後の metadata 禁止（(b)）」が前提にあり、
//     本契約 (c) は「View 寿命中に他スレッドが slot を変えない」部分を保証する。
//   ★ この契約を破ると（Coordinator / Timer から peekDeferred() を呼ぶ等）UB。
//     将来スレッドを増やす場合は metadata() を値返しに戻し、
//     Single Thread Owner 前提の解除を設計レビューで行うこと。
//
// ★ move-only: copy を禁止し、stack 上で短命に使い切り、consume/discard 完了後は
//   再利用しない。これにより std::vector<DeferredPublishView> のような誤用
//   （View の長期保持・複製）を防ぐ。
// ★ valid() を公開しない（第9回レビュー反映）: 状態照会 API（bool valid() 等）は
//   追加しない。if (view.valid()) のような状態分岐を書かせると、consume() / discard()
//   のプロトコル中心設計が崩れる。状態確認は内部実装と assert に留め、公開 API は
//   metadata() / consume() / discard() の3つのみ（Protocol Object としての一貫性）。
class DeferredPublishView {
public:
    // ★ metadata() が「const 参照」を返す理由（第9回レビュー反映①で根拠を明確化）:
    //   安全性は次の3点の組合せで成立する。
    //     (a) View が slot の所有権を握っている — Protocol View として consume/discard
    //         は View にしか存在せず、参照対象 slot の寿命は View の生存期間に対応する。
    //     (b) consume()/discard() 後の metadata() 呼び出しは契約違反 — state_ ガード +
    //         assert で検出（→ 下記「呼び出し条件」）。
    //     (c) Single Thread Owner（RebuildThread）契約 — View の寿命中、slot が他スレッド
    //         から変更される経路がないため、const 参照がダングリングしない。
    //   → 「Single Thread Owner だけが理由」ではない。(a)+(b) で「参照が無効化されない
    //     使い方のみ許可」され、(c) が「参照中の変更なし」を保証する。
    //   「値コピー（スナップショット保証）」は将来スレッド増加への防御だったが、
    //   現行スレッドモデル（単一所有者）では不要なコピーであるため廃止。
    //   ※ 値返しに戻す条件: Single Thread Owner 契約を解除する場合のみ（上記コメント参照）。
    //   ※ 参照の有効期間は View の生存期間に一致する（consume/discard で slot 解放
    //      後の参照使用は state_ ガードとデストラクタ契約で禁止）。
    //   ★ ADR 契約（第11回反映）: 本 const& 返却は「**View が slot 寿命を保証する**」こと
    //     が前提であり、現行スレッドモデル（Single Thread Owner）**限定**の契約。
    //     将来スレッドモデルが変わる場合（Coordinator / Timer が slot を読む等）は
    //     値返しへ戻すことを設計契約として ADR-C4 に固定した（→ Appendix D-11）。
    // ★ [[nodiscard]]（第8回レビュー反映③）: 戻り値を捨てるだけの呼び出しは
    //   意味がないため、結果破棄をコンパイル時に警告する。
    // ★ 呼び出し条件（第8回反映④・第9回反映・第10回反映）: **State::Valid のときのみ
    //   呼べる**。Consumed / Discarded / MovedFrom 後に metadata() を呼んではならない
    //   （consume 後の slot は解放済みで、参照はダングリングする。MovedFrom は
    //   owner_ が nullptr で参照不可）。
    //   ★ assert は二重防御（第10回反映）: state_ ガード（state_ == State::Valid）に
    //     加え、owner_ != nullptr も assert する。実装が owner_->slot_ へアクセスする
    //     ため、MovedFrom（state = MovedFrom / owner = nullptr）の状態を確実に検出する。
    //     DEBUG: assert(owner_ != nullptr); assert(state_ == State::Valid);
    //   「view.consume() の後に view.metadata()」を書くのは不変条件違反。
    [[nodiscard]] const DeferredPublishMetadata& metadata() const noexcept;

    // consume: Ready 確定後の所有権移転。request を取り出し slot 解放。
    //   ★ consume 後の slot 保持なし（「2つの真実」を構造的に排除）。
    //   多重呼び出し・consume→discard の両立は state_ で防御。
    //   ★ Slot 所有権解除（第10回反映・第11回で所有権概念へ整理・第12回で Authority 集約）:
    //     呼び出し成功時に owner_->finishView() を呼び、state_ を State::Consumed に遷移
    //     させる（所有権フラグ更新は Orchestrator 側の finishView() が行う）。
    //     ※ View は所有権フラグを直接書き換えない（Owner が Owner を管理する）。
    //   ★ 失敗理由の表現（第9回レビュー反映）: 現行は nullopt で「state invalid /
    //     slot empty / double consume」の全失敗を同一表現にする（デバッグ情報は失われる）。
    //     Phase 1 は単一正当経路（peek → evaluate → consume）しかなく、呼び出しは
    //     assert で防げるため nullopt で十分。将来 ConsumeResult
    //     （例: std::expected<PublishRequest, ConsumeError>）へ拡張する余地を残す
    //     （→ Appendix D-9）。※ std::expected は C++23 導入のため、拡張時は
    //     C++23 昇格 or tl::expected 等のポリフィルが前提（第13回棚卸し確定）。
    [[nodiscard]] std::optional<PublishRequest> consume() noexcept;

    // discard: 破棄情報 DeferredDiscardInfo を生成して slot 解放。
    //   ★ slot に理由を記録しない — reason は DeferredDiscardInfo にのみ載る
    //     （Storage は Telemetry 用フィールドを持たない。→ Appendix B-2）。
    //   ★ Telemetry は呼ばない（Queue は DeferredHealth を知らない）— 後述の
    //     TelemetryRecorder::recordDeferredDiscard（Telemetry Adapter）が受け取って記録する。
    //   ★ ライフサイクル契約: デストラクタは暗黙 discard しない。consume() /
    //     discard() のどちらかを呼ぶのは Caller の責務。どちらも呼ばずに View が破棄
    //     された場合は slot が残り、次回 peekDeferred() で再評価される（不変条件 8）。
    //   ★ Slot 所有権解除（第10回反映・第11回で所有権概念へ整理・第12回で Authority 集約）:
    //     呼び出し成功時に owner_->finishView() を呼び、state_ を State::Discarded に遷移
    //     させる（Ownership = Released。フラグ更新は Orchestrator 側が行う）。slot は
    //     残らず解放されるため、後続のデストラクタは所有権解除を行わない（state != Valid）。
    //   ★ 失敗理由の表現（第9回レビュー反映）: consume() と同様、nullopt は全失敗を
    //     同一表現にする。将来 DiscardResult（expected 系）拡張の余地を残す
    //     （→ Appendix D-9。std::expected は C++23 前提・第13回棚卸し確定）。
    [[nodiscard]] std::optional<DeferredDiscardInfo> discard(DiscardReason reason) noexcept;

    // ~DeferredPublishView（第10回レビュー反映・第11回で所有権概念へ整理・第12回で Authority 集約）:
    //   **暗黙 discard はしない**（slot は残る・不変条件 8）が、**Slot 所有権は解除する**。
    //   state_ == State::Valid のときのみ owner_->finishView() を呼ぶ（フラグ更新は
    //   Orchestrator 側。View は「終了した」と通知するのみ）。
    //   Consumed / Discarded は consume/discard が既に解除済み、MovedFrom は所有権を
    //   移転先に引き継いでいるため何もしない。
    //   → 「auto view = peek(); return;」のように peek しただけで破棄しても、次回
    //     peekDeferred() が所有権検査で assert にならない（slot は残ったまま再評価）。
    ~DeferredPublishView() noexcept;

    // move-only: copy 禁止・move 許可（stack 上の返却値としてのみ使用）
    //   ★ move セマンティクス（第9回レビュー反映・第10回で所有権引継ぎを確定）:
    //     move 後、other は State::MovedFrom + owner_ = nullptr に遷移する。MovedFrom
    //     の View に対する metadata()/consume()/discard() は契約違反（DEBUG で assert
    //     検出、release では未定義動作）。MovedFrom の View は破棄するのみ。
    //   ★ Slot 所有権は**移転先が引き継ぐ**（Orchestrator 側 slotOwnership_ は Borrowed
    //     のまま。other のデストラクタは state != Valid のため解除しない）。move 代入は、
    //     代入先が Valid だった場合に**先に自身の Slot 所有権を解除**してから移転を受ける
    //     （自己代入は this != &other でガード）。
    DeferredPublishView(DeferredPublishView&& other) noexcept;
    DeferredPublishView& operator=(DeferredPublishView&& other) noexcept;
    DeferredPublishView(const DeferredPublishView&) = delete;
    DeferredPublishView& operator=(const DeferredPublishView&) = delete;

private:
    // peek 専用コンストラクタ（RuntimePublicationOrchestrator のみ生成可能）
    friend class RuntimePublicationOrchestrator;
    explicit DeferredPublishView(RuntimePublicationOrchestrator* owner) noexcept;

    RuntimePublicationOrchestrator* owner_ = nullptr;

    // ★ View の状態: bool ではなく状態を表す enum。bool では「false」が Consumed /
    //   Discarded のどちらか判別できず、ログ・assert が曖昧になる。
    //   State なら assert(state_ == State::Valid) と書ける（→ Appendix B-2）。
    // ★ MovedFrom（第9回レビュー反映）: move 後の View の状態。owner_ = nullptr と
    //   併用し、MovedFrom 状態での公開API呼び出しを assert で検出する。
    enum class State { Valid, Consumed, Discarded, MovedFrom };
    State state_ = State::Valid;
};

// DeferredDiscardInfo — 破棄情報（Telemetry 入力。Queue は記録しない・するとも知らない）
struct DeferredDiscardInfo {
    DiscardReason reason{DiscardReason::None};
    uint64_t ageMs{0};            // enqueue からの経過時間
    uint64_t overwriteCount{0};   // 累積上書き回数
};
```

> **View パターンの意義（Protocol View）:** 従来の `peekDeferredMetadata()` +
> `consumeDeferredRequest()` ペアは「必ずこの順で呼ぶ」という**プロトコル**が API の
> 外側に暗黙に存在した。`peekDeferred()` が View を返す設計では consume/discard は
> **view にしか生えない**ため、「peek せずに consume する」「consume を2回呼ぶ」コードが
> **コンパイル時に書けなくなる**。将来の複数呼び出し元追加時のプロトコル破綻リスクを
> 型で防ぐ。判断根拠の詳細は **Appendix B-2**。

**責務分担の明確化:**
| 層 | 責務 | 実体 |
|----|------|------|
| **Storage (POD)** | データ構造のみ。Policy・Queue ロジックを持たない | `DeferredPublishTypes.h` (`DeferredPublishSlot` / `DeferredPublishMetadata` / `DeferredDiscardInfo`) |
| **Queue** | slot の格納・取出しのみ。判定・破棄理由の記録判断を持たない | `enqueueDeferred` / `peekDeferred` / `DeferredPublishView` (`metadata` / `consume` / `discard`) / `hasDeferred_` |
| **Policy (Admission)** | 有効性判定（TTL → generation → sequence） | `PublicationAdmission::evaluateDeferred` |
| **Telemetry Adapter** | `DeferredDiscardInfo` → `DeferredHealth` 変換・記録（第4回反映②） | `TelemetryRecorder::recordDeferredDiscard` |
| **Orchestrator (調整)** | Policy の結果に応じた再送・破棄の調整（破棄情報を Telemetry Adapter へ渡すのみ） | `submitPublishRequest` 内の分岐 / `view.discard` → `telemetryRecorder().recordDeferredDiscard` |

> **依存方向:** `DeferredPublishTypes.h` → `PublicationAdmission.h`（PublishRequest 参照）。
> `PublicationAdmission.h` は `DeferredPublishMetadata` を前方宣言のみ参照し、
> Storage 型の完全定義を所有しない。これにより「Policy が Storage を持つ」依存が
> 発生しない。

### A-4. 判定ロジック（`evaluateDeferred` 内、現 notifyTransitionComplete のフローを踏襲）

```
PublicationAdmission::evaluateDeferred(const DeferredPublishMetadata& m,
                                       const DeferredAdmissionSnapshot& ctx):
  // 判定順序 TTL → Generation → Sequence:
  //   「安価 → 高価」「局所 → 全体」の順。いずれのチェックも短絡（early return）。
  //   判定順序の根拠は Appendix B-2。
  // ★ Policy は AudioEngine を参照しない（第11回反映・第12回で完全スナップショット化）。
  //   エンジン状態と現在時刻（nowUs）は呼び出し元（RebuildThread）が
  //   DeferredAdmissionSnapshot に詰めて渡す。Policy は実行中に何も読まない。
  // 起動停止時は再送不要
  if (ctx.shutdownInProgress)
      return { Discard, DiscardReason::StaleDiscard };
  // TTL チェック（chrono リテラル採用・現在時刻は ctx.nowUs（第12回反映で完全
  //   スナップショット化。Policy は getCurrentTimeUs() を直接呼ばない））
  if (m.enqueueTimestampUs != 0
      && (ctx.nowUs - m.enqueueTimestampUs) > kDeferredPublishTTL.count())
      return { Discard, DiscardReason::Expired };
  // Generation チェック
  if (m.generation != 0
      && m.generation != ctx.currentGeneration)
      return { Discard, DiscardReason::StaleDiscard };
  // Sequence チェック
  if (m.sequence < ctx.lastSequence)
      return { Discard, DiscardReason::StaleDiscard };
  return { Ready, DiscardReason::None };
```

> **依存関係の整理:** `kDeferredPublishTTL` は現行
> `RuntimePublicationOrchestrator.h:48` の `kDeferredPublishTTLUs`（`uint64_t`、
> マイクロ秒リテラル）を移設しつつ、**chrono 型へ置換**する:
> ```cpp
> // PublicationAdmission.h（Policy 定数）
> inline constexpr std::chrono::microseconds kDeferredPublishTTL =
>     std::chrono::seconds(30);
> ```
> `getCurrentTimeUs()` は既存 `core/TimeUtils.h` を使用。マイクロ秒を返すため
> `.count()` で比較する。★ **時刻取得は呼び出し元が行い `ctx.nowUs` に詰める**
> （第12回反映）: Policy は ctx の値のみで判定するため、同一コンテキストで何度
> 呼んでも同じ結果を返す（決定論的・テスト容易）。
> `DeferredPublishMetadata` は `DeferredPublishTypes.h` で定義。
> Admission は `DeferredPublishSlot`（Queue 構造）を**知らない** — metadata のみで
> 判定できる。

**ライフサイクル（peek → evaluate → consume）:**

```
RebuildThread (A-5):
  // ① peek: View を取得（deferred が無ければ nullopt で終了）。所有権は動かない
  viewOpt = orchestrator_.peekDeferred()
  if (!viewOpt) return                        // deferred なし

  // ② evaluate: Policy が metadata の const 参照で判定。副作用なし
  //    ★ Policy は Engine 非依存 — 判定に必要な4値をコンテキストに詰めて渡す（第11回反映・
  //      第12回で nowUs 追加）:
  ctx = DeferredAdmissionSnapshot {
      .currentGeneration   = engine_.rebuildRequestGeneration,
      .lastSequence        = engine_.getLastCommittedPublicationSequence(),
      .shutdownInProgress  = engine_.isShutdownInProgress(),
      .nowUs               = getCurrentTimeUs(),   // TTL 判定用・呼び出し元で取得
  }
  result = admission_.evaluateDeferred(viewOpt->metadata(), ctx)

  switch (result.decision):
    Ready:
      // ③ 消費: Ready 確定後のみ所有権移転（consume = request を取り出し slot 解放）
      reqOpt = viewOpt->consume()
      if (reqOpt) submitPublishRequest(*reqOpt)
    Discard:
      // 破棄理由は evaluateDeferred が決定（Expired / StaleDiscard）。Caller は mapping 不要
      if (discardInfo = viewOpt->discard(result.discardReason))
          telemetryRecorder_.recordDeferredDiscard(*discardInfo)   // Telemetry Adapter
      // ★ 第6回レビュー反映③: RetryLater は導入見送り（Ready / Discard の2値のみ）
```

`DeferredPublishView::consume()` は**判定を持たず**、slot の取出しのみ:

```
RuntimePublicationOrchestrator::DeferredPublishView::consume():
  1. state_ != State::Valid → return nullopt（二重呼び出しガード）
  2. hasDeferred_ == false → return nullopt
  3. deferredSlot_ が有れば request を取り出して返す
  4. slot を解放 (reset) + hasDeferred_ = false + state_ = State::Consumed
     ← consume = 所有権移転・即解放
```

> **「2つの真実」を排除（判断根拠は Appendix B-2）:** consume（Ready 時）と
> discard（破棄時）のどちらか一方だけが slot を解放する。consume 後に slot を保持する
> ことは**ない**（request と slot の owner が分かれる設計は採用しない）。判定
> （peek/evaluate）と所有権移転（consume）が完全に分離される。

**破棄の実装（`discard` と telemetry の責務分離・Adapter 化）:**

`DeferredHealth`（`TelemetryRecorder.h:122-128`）は `lastDiscardReason` /
`lastDiscardTimestampUs` を保持し、`recordDeferredHealth` が `lastDeferredHealth_`
(atomic) に格納 → `ISREvidenceExporter::buildDeferredHealthJson`
（`ISREvidenceExporter.cpp:231-244`）が `deferred_health.json` に出力する。

```
// Queue 責務（Telemetry を呼ばない・知らない）
RuntimePublicationOrchestrator::DeferredPublishView::discard(DiscardReason reason):
  1. state_ != State::Valid → return nullopt（二重呼び出しガード）
  2. DeferredDiscardInfo info;                    // ★ slot に理由を記録しない
     info.reason         = reason;                //   Storage は Telemetry 用
     info.ageMs          = (now - slot.enqueueTimestampUs) / 1000;  // フィールドを持たない
     info.overwriteCount = deferredOverwriteCount();
  3. deferredSlot_.reset(); hasDeferred_ = false; state_ = State::Discarded;  // slot 解放
  4. return info

// TelemetryRecorder（Telemetry Adapter）
// ★ DeferredDiscardInfo → DeferredHealth 変換は Telemetry 層の責務。
//   Coordinator は Telemetry 型（DeferredHealth）を知る必要がなくなる。
//   （責務の流れ・判断根拠は Appendix B-2）
TelemetryRecorder::recordDeferredDiscard(const DeferredDiscardInfo& info):
  1. DeferredHealth dh;
     dh.deferredCount        = 0;                   // 解放後は 0
     dh.oldestDeferredAgeMs  = info.ageMs;
     dh.overwriteCount       = info.overwriteCount;
     dh.lastDiscardReason    = info.reason;
     dh.lastDiscardTimestampUs = nowUs();
  2. recordDeferredHealth(dh);                      // Evidence 出力
```

### A-5. RebuildThread 側の消費

```
if (doDeferredPublish && runtimeOrchestrator_ != nullptr)
{
    // ① peek: View を取得。所有権は動かない（deferred が無ければ nullopt）
    auto viewOpt = runtimeOrchestrator_->peekDeferred();
    if (!viewOpt) return;

    // ② evaluate: Policy (Admission) が metadata const 参照で判定。副作用なし
    //    ★ Policy は Engine 非依存（第11回反映）: 判定に必要な4値をコンテキストで渡す
    //      （第12回反映で nowUs 追加。時刻取得はここで行い Policy には渡さない）
    const convo::isr::DeferredAdmissionSnapshot ctx {
        /* currentGeneration  */ rebuildRequestGeneration,
        /* lastSequence       */ getLastCommittedPublicationSequence(),
        /* shutdownInProgress */ isShutdownInProgress(),
        /* nowUs              */ getCurrentTimeUs(),
    };
    const auto result = runtimeOrchestrator_->admission().evaluateDeferred(
                            viewOpt->metadata(), ctx);

    // ③ 所有権移転: Ready → consume / Discard → discard
    //   （★ 第6回レビュー反映③: RetryLater は導入見送り。Ready / Discard の2値のみ）
    switch (result.decision)
    {
        case DeferredDecision::Ready:
            if (const auto reqOpt = viewOpt->consume())
                runtimeOrchestrator_->submitPublishRequest(*reqOpt);  // 再送
            break;
        case DeferredDecision::Discard:
            if (const auto discardInfo = viewOpt->discard(result.discardReason))
                runtimeOrchestrator_->telemetryRecorder().recordDeferredDiscard(*discardInfo);
            break;
    }
}
```

> **責務の流れ:** peek（参照のみ）→ evaluate（判定のみ）→ consume or discard
> （所有権移転・slot 解放）。consume 後に slot を保持する**2つの真実**状態は発生しない。
> `view.consume()` / `view.discard()` は判定を持たないため、Queue の責務が純粋に保たれる。
> Telemetry 記録は **TelemetryRecorder の Adapter**（`recordDeferredDiscard`）が一元化し、
> Coordinator は `DeferredHealth` 型を知らない（判断根拠は Appendix B-2）。

- 破棄時は `view.discard` → `telemetryRecorder().recordDeferredDiscard` の流れで
  `lastDiscardReason` を telemetry に反映。
- 現行 `enqueueDeferred` の `maxDeferredAgeMs` / `deferredOverwriteCount` 監査は維持。

**shutdown 時（`clearDeferredForShutdown`）の新設計:**

```
RuntimePublicationOrchestrator::clearDeferredForShutdown():
  1. hasDeferred_ == false → return（何もしない）
  2. // 破棄情報は reset 前に slot から採る（★ reset 後の slot 参照はダングリング・第13回反映）
     DeferredDiscardInfo info;
     info.reason         = DiscardReason::ShutdownDiscard;
     info.ageMs          = (nowUs - deferredSlot_->enqueueTimestampUs) / 1000;
     info.overwriteCount = deferredOverwriteCount();
  3. // slot 解放（shutdown は評価不要。Slot に残っているなら直ちに破棄）
     deferredSlot_.reset(); hasDeferred_.store(false, release);
  4. telemetryRecorder().recordDeferredDiscard(info);   // Telemetry Adapter
```

> **一貫性:** shutdown も stale discard も「slot 解放 + 破棄情報の記録」という
> 同じ形になり、Telemetry 記録は常に `recordDeferredDiscard`（Telemetry Adapter）を
> 通る。現行の「`clearDeferredForShutdown` が slot に reason を書く」特例は消える
> （現行実装との対応は Appendix C）。

**enqueue 時の滞留状態記録:**

```
TelemetryRecorder::recordDeferredEnqueue(uint64_t overwriteCount):
  1. DeferredHealth dh;
     dh.deferredCount        = 1;      // 新規 enqueue
     dh.oldestDeferredAgeMs  = 0;
     dh.overwriteCount       = overwriteCount;
     dh.lastDiscardReason    = DiscardReason::None;
  2. recordDeferredHealth(dh);
```

> enqueue は破棄ではないため `recordDeferredDiscard` は使わず、専用 Adapter
> `recordDeferredEnqueue` を追加する。これで Queue（`enqueueDeferred`）は
> `DeferredHealth` の生成・記録から解放され、Orchestrator の `enqueueDeferred` は
> slot への格納と `deferredOverwriteCount_` / `maxDeferredAgeMs_` 監査のみを担う。

### A-6. 利点・懸念（概要）

> **利点・懸念の詳細（Generation 二重チェック・Sequence Guard 実効性・TTL 妥当性・
> peek/consume の atomicity と Single Thread Owner 契約・破棄理由の記録タイミング・
> Telemetry Event 化の保留判断）は Appendix B-2 に移動済み。**

### A-6b. 実装影響ファイル

| ファイル | 変更 |
|---------|------|
| `RuntimePublicationTypes.h` (新規) | `PublishRequest` を `PublicationAdmission` のネスト型から独立定義（第8回反映①）。Storage / Policy 双方が参照する中立型 |
| `DeferredPublishTypes.h` (新規) | `DeferredPublishSlot`（request + metadata）/ `DeferredPublishMetadata`（generation + sequence + enqueueTimestampUs、`DeferredGuard` と `enqueueTimestampUs` を統合）/ `DeferredDiscardInfo` を独立PODとして定義。`lastDiscardReason` は Storage に持たない。`PublishRequest` は `RuntimePublicationTypes.h` を参照 |
| `PublicationAdmission.h` | `DeferredDecision` / `DeferredAdmissionResult` / `DeferredAdmissionSnapshot` / `evaluateDeferred(metadata, ctx)` 宣言、`kDeferredPublishTTL` 移設（chrono 型）。ネスト型 `PublishRequest` は削除し `RuntimePublicationTypes.h` を include（第8回反映①） |
| `PublicationAdmission.cpp` | `evaluateDeferred`（→ private `evaluateDeferredImpl`）実装（TTL / generation / sequence 判定） |
| `TelemetryRecorder.h/.cpp` | `recordDeferredDiscard(DeferredDiscardInfo)` / `recordDeferredEnqueue(overwriteCount)` Adapter（DeferredHealth 変換 + 記録）追加 |
| `RuntimePublicationOrchestrator.h` | `peekDeferred` / `DeferredPublishView`（`metadata` / `consume` / `discard`、move-only + State enum）追加、`kDeferredPublishTTLUs` / `DeferredGuard` / `lastDiscardReason` 廃止 |
| `RuntimePublicationOrchestrator.cpp` | `DeferredPublishView` の各メソッド実装、`clearDeferredForShutdown` を `recordDeferredDiscard` 経由に統一、`enqueueDeferred` の `DeferredHealth` 生成を `recordDeferredEnqueue` へ移譲 |
| `AudioEngine.RebuildDispatch.cpp` | 消費箇所の変更 (`A-5`: peek → evaluate → consume) |
| テスト | deferred 破棄経路の回帰テスト（TTL / StaleGeneration / StaleSequence / Ready / ShutdownDiscard） |

---

## Part B. DSPTransition::onTransitionComplete の整理設計（Layer 2/3 統合時）

> **現状分析（onTransitionComplete の3責務・重複実装箇所の詳細）と設計方針
> （削除方針・ヘルパー抽出方針）は Appendix B-4 を参照。**

### B-3. 整理対象: 重複パターンの一元化

Timer.cpp 内の5箇所以上で繰り返される:

```cpp
DSPCore* current = consumeAtomic(fadingRuntimeDSPSlot, acquire);
if (current != nullptr
    && compareExchangeAtomic(fadingRuntimeDSPSlot, current, nullptr, acq_rel, acquire))
{
    DSPLifetimeManager lifetime(*this);
    const auto fadingHandle = dspHandleRuntime_.getFadingRuntimeDSPHandle();
    if (!fadingHandle.isNull())
        runtimePublicationBridge_.submitObserve(fadingHandle);
}
```

これを AudioEngine の単一ヘルパーに抽出:

```cpp
// AudioEngine.h (private)
// ★ クロスフェード完了時: fading slot を CAS クリアし、observe を非同期投入する。
//   全 Timer 経路がこのヘルパーを通る（重複除去）。
//   命名: 「fading slot の後始末を完了させる」意図で finalize を採用
//   （単なる clear ではなく、CAS クリア + observe 送出までを1操作にまとめる）。
//
// ★ 第4回レビュー対応（粒度）: 内部を2段に分離する。これにより Phase 2/3 で
//   CrossfadeCompleteIntent 導入時に submitObserve が不要になっても、
//   「CAS による終端処理」だけを再利用でき、変更範囲が抑えられる。
void finalizeFadingSlot() noexcept;

// 内部実装は2段構造（Phase 2/3 への移行を考慮）:
//
//   Step 1 — 終端処理（CAS クリア + fading handle 取得）
//     DSPCore* current = consumeAtomic(fadingRuntimeDSPSlot, acquire);
//     DSPHandle fadingHandle{};
//     if (current != nullptr
//         && compareExchangeAtomic(fadingRuntimeDSPSlot, current, nullptr, acq_rel, acquire))
//     {
//         DSPLifetimeManager lifetime(*this);
//         fadingHandle = dspHandleRuntime_.getFadingRuntimeDSPHandle();
//     }
//     // fadingHandle が null なら observe も publish も不要
//
//   Step 2 — observe 投入（終端処理の結果に依存）
//     if (!fadingHandle.isNull())
//         runtimePublicationBridge_.submitObserve(fadingHandle);
//
// ★ Phase 2/3 の CrossfadeCompleteIntent ハンドラは Step 1 を流用し、
//   Step 2 を Intent 経由（Coordinator の処理）へ置き換える（B-4 参照）。
//   Step 1/2 の分離単位は private メソッド（例: finalizeFadingTermination() /
//   finalizeFadingSlotObserve()）として実装し、統合ヘルパー finalizeFadingSlot() が
//   両方を順に呼ぶ形にする。
//   ★ Step 1 の命名（第8回レビュー反映②）: finalizeFadingTermination() —
//     CAS は実装手段であり、メソッド名に CAS を入れない（API に実装手段を漏らさない。
//     将来 compare_exchange を使わなくなっても API 名は変わらない）。
```

> **2段構造の粒度判断（設計メモ）:** Step 1/2 のメソッド分離は Phase 2/3 の
> CrossfadeCompleteIntent 導入時に Step 1 のみを再利用するため。`bool submitObserve`
> 引数による分岐より変更範囲が小さい（判断根拠は Appendix B-4）。

### B-4. Layer 2/3 統合時の最終形（将来）

Layer 2/3 統合（Coordinator がすべての Decision を一元化し、ISR への委譲が完全化された
段階）では、クロスフェード完了は Intent として Coordinator に通知されるべき。

**実現可能性（調査確定）:** 既存の Intent パイプラインが完全に確立済み。
`IntentType`（`ISRRuntimePublicationCoordinator.h:172-177`）= `Observe` / `Publish` /
`Recovery` / `Quarantine`。`kDispatchTable`（`ISRIntentDispatcher.h:58-66`）は
IntentType → IntentHandler の 1:1 total mapping で、`static_assert` により新しい
IntentType のテーブル漏れをコンパイル時に検出。`processIntent`
（`ISRRuntimePublicationCoordinator_ProcessIntent.cpp:10-52`）が pop → dispatch。
**CrossfadeComplete の追加は、enum 列挙 + payload 定義 + Handler 登録の3点で
既存パターンに完全に適合する**。

```
AudioThread / Timer
    └─ fade complete 検知
        └─ Intent: CrossfadeCompleteIntent (handle, id) を enqueue   // id = CrossfadeId（B-4 参照）
            └─ ISR CoordinatorLoop (processIntent)
                └─ CrossfadeCompleteHandler (新規, IntentType::CrossfadeComplete)
                    ├─ CrossfadeAuthority: unregisterCrossfade
                    ├─ DSPHandleRuntime: endCrossfade
                    ├─ finalizeFadingTermination()  // B-3 の Step 1（終端処理）のみ流用
                    │   ※ Step 2 の observe 投入は不要（Intent 経路に置換）
                    └─ PublicationExecutor へ委譲（★ 第6回レビュー反映）:
                        publishIdleWorldOnly を Handler が直接呼ばず、
                        PublicationExecutor（publish 一元実装）へ委譲
```

> **第6回レビュー反映（Part B）: Handler は publish を直接実行せず PublicationExecutor へ委譲。**
> 上図の最終形では Handler が `publishIdleWorldOnly(currentAfterFade, HardReset)` を
> 直接呼ぶのではなく、**`PublicationExecutor`**（`PublicationExecutor.h:17`、
> validate → publishAndSwap → retire old を一元実行）へ委譲する。理由:
> Handler が publish 実装を知ると、将来の Publication Policy 共有（正常完了 SmoothOnly /
> タイムアウト HardReset 等の選択）が Handler に散らばる。`PublicationExecutor` は
> `RuntimePublicationOrchestrator.h:177` で `executor_` として既に実在し、
> ISR の Authority Singularization（Decision Authority は Coordinator・publish 実行は
> Executor）に適合する。今回 Phase 1 スコープ外のため、B-4 最終形として記録する。

**追加実装要素（現行 Intent パイプラインへの適合）:**
| 要素 | 現行 | 追加 |
|------|------|------|
| `IntentType` enum | 4種 (`ISRRuntimePublicationCoordinator.h:172-177`) | `CrossfadeComplete` を追加、`kIntentTypeCount`=5 に |
| `kDispatchTable` | 4登録 (`ISRIntentDispatcher.h:58-63`) | `CrossfadeCompleteIntentHandler` を追加（static_assert 自動検証） |
| Payload | `ObservePayload` / `PublishPayload` / `RecoveryPayload` / `QuarantinePayload` | `CrossfadeCompletePayload { DSPHandle handle; CrossfadeId id; }` |
| Handler 規約 | `IntentHandler` は無状態 singleton（HANDLER-1、`DispatcherHasNoDecision` assert で強制） | `CrossfadeCompleteIntentHandler` も無状態で実装 |

> **CrossfadeId の要否（第5回レビュー指摘への調査確定）:**
> 結論: **必要（維持）。** 理由は、Handler が実行する crossfade 終了処理が
> `DSPHandleRuntime::endCrossfade(CrossfadeId)`（`ISRDSPHandle.h:132`）と
> `CrossfadeAuthorityRuntime::unregisterCrossfade(CrossfadeId)`（`ISRDSPHandle.h:215`）で、
> いずれも **CrossfadeId を必須引数**とするため。DSPHandle（`{slot, generation}`、
> `ISRDSPHandle.h:25-49`）は CrossfadeId を内包せず、id は
> `getActiveCrossfades()` の `CrossfadeRecord` でのみ対応付けられる。
> 現行の fade 完了処理（`Timer.cpp:864-877`）も `records.front().id` →
> `notifyFadeComplete(xfadeId)` → `consumeCompletedFade(ev).id` で CrossfadeId を
> 取り回しており、Phase 2/3 で Handler へ移す際も Payload に CrossfadeId が必要。
> なお Payload は**自己完結**とし、Handler 側で `getActiveCrossfades()` 検索をしない
> （無状態規約 HANDLER-1 を維持するため、必要な id は enqueue 側で確定して渡す）。

これにより:
- Timer から publish 実行が消え、ISR の **Builder 責務一元化**（ADR-C4 方針）と整合。
- 重複していた「publish」ボイラープレートが Intent Handler に集約。
- `onTransitionComplete` が持っていた3責務が Intent ハンドラへ完全移行。
- Handler の無状態規約（`DispatcherHasNoDecision`）が維持される限り、
  Decision Authority は Coordinator に留まる（Authority Singularization と整合）。
- **Handler は publish を直接実行しない**（第13回⑤ 確認）: `CrossfadeCompleteIntentHandler` も
  HANDLER-1 遵守。crossfade 完了は `publishIdleWorldOnly()`（Timer 直呼→現行 `.h:131-137` 自記の dead code 化）または
  `PublishExecutor` へ**委譲**する。Handler は「publish 要求」ではなく「Coordinator への
  **状態通知**」として位置づける（ISR Observer = write 禁止・Intent enqueue のみ）。

### B-5. 段階整理（推奨手順）

| Phase | 内容 |
|-------|------|
| **Phase 0 (今回は実装しない)** | `onTransitionComplete` 本体を削除（呼び出し元ゼロ確認済み） |
| **Phase 1** | `finalizeFadingSlot()` ヘルパー抽出、Timer.cpp の CAS ボイラープレートを置換 |
| **Phase 2 (Layer 2/3)** | CrossfadeCompleteIntent 導入、Timer の inline publish を Intent 化 |
| **Phase 3 (Layer 2/3)** | 上記 Intent ハンドラへ publish を集約、`onTransitionComplete` の役割を完全に閉じる |

### B-6. 実装影響ファイル

| ファイル | Phase | 変更 |
|---------|-------|------|
| `DSPTransition.h` | 0 | `onTransitionComplete` 削除 |
| `AudioEngine.h` | 1 | `finalizeFadingSlot` 宣言 |
| `AudioEngine.Timer.cpp` | 1 | CAS ボイラープレートをヘルパー呼び出しへ |
| `AudioEngine.CtorDtor.cpp` / `ReleaseResources.cpp` | 1 | 同左（破棄時は Step 1 `finalizeFadingTermination()` のみ・observe は不要） |
| `ISRRuntimePublicationCoordinator*` | 2/3 | CrossfadeCompleteIntent 追加 |

> **設計判断メモ**（「onTransitionComplete を Timer から呼んで一元化」しない理由・
> SmoothOnly / HardReset をヘルパーに含めない判断）は **Appendix B-4** を参照。

---

# Appendix

> **構成:** Appendix A（詳細調査）/ B（設計判断詳細: Part A の現状分析・設計方針・利点懸念、
> Part B の現状分析・判断メモ）/ C（現行コード対応表）/ D（改訂履歴・レビュー反映メモ）。

## Appendix A. 詳細調査結果（コード照合済み・調査確定事項）

> 旧「Appendix B. 詳細調査結果」から改名（本文の実装契約から参照される調査項目のみを集約）。

1. **`admission_.evaluate` は既に generation 比較を持つ**（`PublicationAdmission.cpp:6-57`、
   2. Generation staleness check が `req.generation != engine.rebuildRequestGeneration`）。
   ※ **sequence 比較は現行 evaluate には存在しない**（第9回調査確定）: evaluate は
   Shutdown → Generation → DSP finalized → HealthState → Pressure → Fading の順で判定し、
   `getLastCommittedPublicationSequence()`（`AudioEngine.h:1592`）は参照しない。
   sequence guard は deferred 専用の**新規ロジック**として `evaluateDeferred` に
   追加する（generation と同型の Policy 層判定であり、重複実装ではなく「同じ Policy
   層への集約」。`rebuildRequestGeneration` は `AudioEngine.h:2372`）。
2. **`hasDeferred_` は既に atomic**（`RuntimePublicationOrchestrator.h:81-88` で
   `load/store`）。work88 REPAIR_PLAN 記載の「非 atomic のまま」は**解消済み**。
   BUG-052 は本設計と独立に修正済み。
3. **`submitPublishRequest` の呼び出し元は RebuildThread のみ**
   (`AudioEngine.Commit.cpp:709` ← `AudioEngine.RebuildDispatch.cpp:847`)。
   consume → submit は同一スレッド連続実行。deferred slot へのデータ競合リスクは
   `hasDeferred_` atomic 化で解消済み。
4. **TTL 30秒は妥当**: crossfade の fade 時間は `CrossfadePolicy` の max で実測上限
   ~0.080s（`m_irFadeTimeSec=0.080`、`AudioEngine.h:2187`）。TTL は約375倍のマージン。
   Timer は 1ms tick ポーリング（`AudioEngine.Timer.cpp`）。→ 現行値維持。
5. **Sequence guard 追い越しは成立しうる**: sequence は
   `reserveRuntimePublicationIdentity()`（`AudioEngine.h:3349`）で採番、commit 成功時のみ
   `lastCommittedPublicationSequence_` 更新（`AudioEngine.Commit.cpp:397-398`）。
   deferred 滞留中に非 deferred 経路（`publishIdleWorldOnly` 等）が commit すれば
   guard.sequence < committed で検出される。→ 防御的チェックとして維持。
6. **deferred 破棄の telemetry 経路は完成済み**: `DeferredHealth`
   （`TelemetryRecorder.h:122-128`）の `lastDiscardReason` / `lastDiscardTimestampUs` が
   `recordDeferredHealth` → `lastDeferredHealth_`(atomic) →
   `ISREvidenceExporter::buildDeferredHealthJson`（`ISREvidenceExporter.cpp:231-244`）
    → `deferred_health.json` に出力される。`view.discard(reason)` が reason を slot に記録し、
    `TelemetryRecorder::recordDeferredDiscard`（Telemetry Adapter）が `DeferredDiscardInfo`
    を `DeferredHealth` に変換して `recordDeferredHealth` を呼べば、破棄理由が Evidence
    に反映される。
7. **`onTransitionComplete` は呼び出し元ゼロを確定**: `DSPTransition.h:138-165` の定義のみで
   参照は宣言のみ（grep 結果）。B-1 の「呼び出し元ゼロ」は正確。Phase 0 削除が有効。
8. **Layer 2/3 の CrossfadeCompleteIntent は既存 Intent パイプラインに完全適合**:
   `IntentType`（`ISRRuntimePublicationCoordinator.h:172-177`）= Observe/Publish/Recovery/
   Quarantine の4種。`kDispatchTable`（`ISRIntentDispatcher.h:58-66`）は 1:1 total mapping +
   static_assert。Handler は無状態 singleton 規約（`DispatcherHasNoDecision`）。
   CrossfadeComplete 追加は enum + payload + Handler 登録の3点で実現可能（B-4 参照）。
9. **crossfade timeout の検知と回復は分離**: 検知は `RuntimeHealthMonitor::
   checkCrossfadeTimeout`（30秒、`RuntimeHealthMonitor.cpp:530-547`）が
   `EVENT_CROSSFADE_TIMEOUT` を emit。回復は `onHealthEvent` → `Timer.cpp:1547-1598` の
   CAS クリア + complete + `publishIdleWorldOnly(HardReset)`。B-1 の記述と一致。
10. **Sequence Guard の実効性を確定（第3回調査）**: deferred 滞留中に非 deferred 経路が
    commit して sequence を進める経路を実コードで確認。`publishIdleWorldOnly`
    （`AudioEngine.Transition.cpp:10-30`）は `commitRuntimePublication`（`Transition.cpp:25`、
    内部で `AudioEngine.Commit.cpp:397` の `publishAtomic` により sequence 更新）を呼ぶ。
    呼び出し元は `Timer.cpp:1593`（timeout recovery: HardReset）と
    `CtorDtor.cpp:72`（RestoreStep2: HardReset）、さらに fade 正常完了の inline
    build→commit も `Timer.cpp:915` で `commitRuntimePublication` を呼ぶ。
    → deferred 滞留中にこれらが実行されれば `metadata.sequence < committed` で
    StaleSequence 破棄が成立する。**防御的チェックとして維持（確定）**。
11. **CrossfadeId は Payload に必須（第5回調査確定）**: `CrossfadeId` は
    `uint32_t`（`ISRDSPHandle.h:79`）。`DSPHandle`（`{slot, generation}`、
    `ISRDSPHandle.h:25-49`）は CrossfadeId を内包せず、対応は
    `CrossfadeAuthorityRuntime::records_`（`CrossfadeRecord`）でのみ分かる。
    現行の fade 完了処理（`Timer.cpp:865-876`）は `getActiveCrossfades()` の
    `records.front().id` → `notifyFadeComplete(xfadeId)` →
    `consumeCompletedFade(ev).id` で取り回し、`endCrossfade(ev.id)` /
    `unregisterCrossfade(ev.id)`（いずれも CrossfadeId 必須引数）を実行する。
    → Phase 2/3 の CrossfadeCompleteHandler が同じ終了処理を行うなら Payload に
    CrossfadeId が必要（B-4 参照）。Payload は自己完結（enqueue 側で id 確定）。
12. **`clearDeferredForShutdown` の DeferredHealth 記録（第5回調査確定）**:
    現行は slot に `lastDiscardReason = ShutdownDiscard` を書き、`DeferredHealth` を
    Orchestrator が生成して `recordDeferredHealth` を呼ぶ
    （`RuntimePublicationOrchestrator.cpp:404-424`）。新設計では slot 記録を廃止し、
    `DeferredDiscardInfo{ShutdownDiscard, ageMs, overwriteCount}` →
    `recordDeferredDiscard`（Telemetry Adapter）に統一する。
13. **enqueue 時の DeferredHealth 記録（第5回調査確定）**: 現行
    `enqueueDeferred`（`RuntimePublicationOrchestrator.cpp:394-400`）は Orchestrator が
    `DeferredHealth` を生成・記録する。新設計では `TelemetryRecorder::recordDeferredEnqueue`
    （専用 Adapter）が生成を担い、Queue は slot 格納 + `deferredOverwriteCount_` /
    `maxDeferredAgeMs_` 監査のみに徹する（A-4 参照）。

## Appendix B. 設計判断詳細（本文から移動）

### B-1. Part A: 現状分析（リファクタ後の実態）

`notifyTransitionComplete` 削除後、deferred publish の再送経路は:

```
CoordinatorLoop (Decision/Notify)
    ↓ publishRetryReady
RebuildThread (Builder)
    ↓ consumeDeferredRequest() → submitPublishRequest()
```

`submitPublishRequest` → `trySubmitImpl` → `admission_.evaluate(req, engine_, ctx)` と進む。

**重要な実態:** 現行 `PublicationAdmission::evaluate` (`PublicationAdmission.cpp:6-57`) は
生成世代チェックを**既に実施している**:

```
2. Generation staleness check
   const int currentGen = consumeAtomic(engine.rebuildRequestGeneration, acquire);
   if (req.generation != currentGen)
       return Decision::RejectedStaleGeneration;
```

すなわち ADR-C4 で「Admission へ統合すべき」とした **generation guard はリファクタ後に
暗黙的に Admission 内に存在済み**。deferred 再送時に世代が進んでいれば
`RejectedStaleGeneration` で拒否される。

**欠けている3点:**
1. **TTL 超過チェック** (`kDeferredPublishTTLUs = 30秒`) — `enqueueTimestampUs` は slot に
   記録されるが、TTL 比較は `notifyTransitionComplete` 内にのみ存在し削除済み。
   現在は `getMaxDeferredAgeMs()` 監査値のみで、TTL 破棄は行われない。
2. **Publication Sequence Guard** — `DeferredGuard.sequence` は enqueue 時に記録されるが
   (`RuntimePublicationOrchestrator.cpp:385`)、**読み出される箇所が存在しない**。
   consume 時に slot から request だけ取り出し guard は捨てられる
   (`RuntimePublicationOrchestrator.h:82-88`)。
3. **DiscardReason の正確な記録** — `Expired`/`StaleDiscard` の破棄理由が telemetry に
   記録されない（`ShutdownDiscard` のみ `clearDeferredForShutdown` が記録）。

### B-2. Part A: 設計方針・利点・懸念

**原則 (ADR-C4 再確認):**
- stale discard は Transition 責務ではなく **Admission Policy** 責務。
- 判定アルゴリズム（TTL → generation → sequence）は再利用する。API は再利用しない。

**方針: 「consume が判定を持つ」ではなく「consume は純粋な Queue、判定は
`PublicationAdmission`（Policy 層）が担う」。**

**ライフサイクル順序: 「peek → evaluate → consume」を厳守。**

```
peekDeferred() → DeferredPublishView（Queue: slot の参照のみ。所有権は動かない）
  ↓
evaluate (Policy: view.metadata() の const 参照で有効/破棄を判定。副作用なし)
  ↓
view.consume() (Queue: Ready の場合のみ request を取り出す = 所有権移転)
    or
view.discard(reason) (Queue: 破棄理由を slot に記録して slot 解放 + DeferredDiscardInfo を返す)
```

**理由（レビュー指摘反映）:**
- **consume → peek は所有権遷移が二意的**（consume 後も slot が残るなら、request と
  slot のどちらが owner か曖昧になる）。**peek → evaluate → consume** なら
  「判定までは参照のみ、Ready 確定後に初めて所有権が動く」ため一意。
- stale discard は ADR-C4 で「**Admission Policy 責務**」と確定済み。Queue
  （`consumeDeferredRequest`）が判定を持つのは、Queue が Policy を持つことになり
  責務分離に逆行する。
- 判定ロジックは現行 `admission_.evaluate`（`PublicationAdmission.cpp:6-57`）と
  **generation 比較が同義**（`engine_.rebuildRequestGeneration`、`AudioEngine.h:2372`）。
  **sequence 比較は現行 evaluate には無い**ため、`evaluateDeferred` で
  `ctx.lastSequence` を読む**新規ロジック**として追加する（第9回調査確定・第11回反映で
  Engine 参照ではなく呼び出し元が `getLastCommittedPublicationSequence()`（
  `AudioEngine.h:1592`）からコンテキストに詰めて渡す）。generation と同型の Policy 層
  判定であり、同一 Policy 層に集約するのが自然。
- `PublishRequest` は audio thread から封入され、ISR 側へ POD で渡る契約
  (`RuntimePublicationOrchestrator.h:17-24`)。guard を request に載せる変更は
  この POD 契約を破るため行わない。guard は **slot に保持したまま**、
  Admission の判定メソッドへ metadata 参照を渡す。

> **第4回レビュー反映:** `evaluateDeferred` は `PublicationAdmission` の
> **Deferred 専用 Policy エントリポイント**であり、`evaluate`（PublishRequest 用）と
> 対になるものではない。実装は `evaluate` と同じクラスの private helper
> （`evaluateDeferredImpl`）として配置し、公開 API はこの1つだけを薄く公開する
> （改善提案③: `DeferredAdmission` 分離までは不要と判断、現在案を採用）。
>
> **第6回レビュー反映①（evaluateDeferred の公開 API 維持判断・ISR 観点）:**
> **Policy は1つだが、Policy Entry Point は1つである必要はない。** ISR の
> Authority Singularization が要求するのは「Decision Authority が一か所」「Policy
> 実装が一か所」であり、「Public API が1個」ではない。`evaluate(PublishRequest)` と
> `evaluateDeferred(DeferredPublishMetadata)` は**異なる入力契約を持つ同一 Authority の
> 別 Entry Point** であり、Authority Singularization を損なわない。
> variant 統合は「API は1つ、実装は2つ」の**見た目だけの単一化**であり、`visit` 内に
> Publish 用判定と Deferred 用判定の巨大 switch を生む。戻り値も非対称（`Decision` vs
> `DeferredAdmissionResult { decision, discardReason }`）のため、variant 化すると
> 第4回で確定した「Decision は動作のみ・理由は別」という二層構造を壊す。→ **現行案を維持**。

**判定順序の根拠（TTL → Generation → Sequence）:**
「安価 → 高価」「局所 → 全体」の順。TTL は metadata のみで完結（最安・過半数ケースを
打ち切れる）、Generation は Runtime 状態（atomic 読み1回）、Sequence は Commit 履歴
（最も観測コストが高い）。いずれも early return で以後の状態参照は行われない。
★ **判定は完全スナップショット（第12回反映）**: 全入力（metadata + ctx の4値）が
Policy に渡る時点で固定され、TTL 用の現在時刻 `nowUs` も ctx に含まれる。Policy は
実行中に何も読まない（決定論的・テスト容易）。

**責務分担の明確化（依存方向）:**
| 層 | 責務 | 実体 |
|----|------|------|
| **Storage (POD)** | データ構造のみ。Policy・Queue ロジックを持たない | `DeferredPublishTypes.h` (`DeferredPublishSlot` / `DeferredPublishMetadata` / `DeferredDiscardInfo`) |
| **Queue** | slot の格納・取出しのみ。判定・破棄理由の記録判断を持たない | `enqueueDeferred` / `peekDeferred` / `DeferredPublishView` (`metadata` / `consume` / `discard`) / `hasDeferred_` |
| **Policy (Admission)** | 有効性判定（TTL → generation → sequence） | `PublicationAdmission::evaluateDeferred` |
| **Telemetry Adapter** | `DeferredDiscardInfo` → `DeferredHealth` 変換・記録（第4回反映②） | `TelemetryRecorder::recordDeferredDiscard` |
| **Orchestrator (調整)** | Policy の結果に応じた再送・破棄の調整（破棄情報を Telemetry Adapter へ渡すのみ） | `submitPublishRequest` 内の分岐 / `view.discard` → `telemetryRecorder().recordDeferredDiscard` |

> **依存方向:** `DeferredPublishTypes.h` → `PublicationAdmission.h`（PublishRequest 参照）。
> `PublicationAdmission.h` は `DeferredPublishMetadata` を前方宣言のみ参照し、
> Storage 型の完全定義を所有しない。これにより「Policy が Storage を持つ」依存が
> 発生しない。
>
> **Policy の将来境界（第12回反映）:** `PublicationAdmission` は publish 判定全般
> （`evaluate` / `evaluateDeferred` / 将来の `evaluateRecovery` 等）を担うため、
> 責務が増えた時点で **`PublicationPolicy` へリネームする余地**と、内部 Policy 分割
> （evaluate 系 = PublicationAdmission・他 = 別 Policy）の境界を確保する。
> **現時点ではリネームしない**（既存参照を増やさない。Phase 1 実装中に境界が固まったら
> 単独コミットで実施）。A-3 PublicationAdmission.h ブロックに同じ旨を明記。

**利点:**
- TTL / sequence ガードが復活し、30秒超の滞留や sequence 追い越しで確実に破棄される。
- 判定が `evaluate` と同じ **PublicationAdmission（Policy 層）** に集約され、
  ADR-C4 の「stale discard は Admission Policy 責務」に完全一致。
- Queue（`DeferredPublishView`）は純粋になり、Queue が Policy を持つという
  責務分離違反が解消される。
- `DeferredPublishSlot` が Storage（POD）として独立し、Policy が Storage を持つ
  依存が発生しない。将来の Queue 拡張（Persistent/Priority）時に Storage 型が
  `PublicationAdmission.h` に増殖しない。
- **peek → evaluate → consume** の順序が**型レベルで強制**される（consume/discard は
  `DeferredPublishView` にしか存在しない）。「peek せずに consume する」「consume を
  2回呼ぶ」コードがコンパイル時に書けないため、将来の複数呼び出し元追加時も
  プロトコル破綻リスクがない。**View は Storage View ではなく Protocol View** —
  データアクセスのためではなく **Ownership Transition Protocol を型で表現するため**に
  存在し、slot 個数に依存しない（第6回レビュー反映②）。
- `metadata()` が **const 参照**を返すため、値コピー（16バイト）が毎回発生しない。
  安全性は **(a) View が slot 所有権を握る + (b) consume/discard 後の metadata()
  禁止（state_ ガード）** の前提の上で、**Single Thread Owner（RebuildThread）契約 (c)**
  が「View 寿命中の slot 非変更」を保証することで成立（第7回反映③・第9回反映①）:
  peek と consume の間に他スレッドが slot を改変する経路は存在しない
  （CoordinatorLoop は `hasDeferredRequest()`（atomic）のみ・slot に触れない）。
  この契約を守らない場合（将来の他スレッドからの peek）は metadata() を値返しに
  戻すこと（本文 A-3 の View コメントに明記）。
- **Telemetry 責務の分離（Adapter 化・第4回反映②）**: Queue（`view.discard`）は
  `DeferredHealth` を知らず、`DeferredDiscardInfo`（純粋データ）を返すのみ。
  `DeferredHealth` への変換・記録は TelemetryRecorder の `recordDeferredDiscard`
  （Telemetry Adapter）が担う。Coordinator は `DeferredHealth` 型を知らない。
- **Decision の二層構造（第4回反映①）**: `DeferredDecision` は動作指示（Ready / Discard）のみ。
  破棄理由は既存 `DiscardReason` が分離保持し、
  `DeferredAdmissionResult { decision, discardReason }` で返す。将来
  Backpressure 等の Decision が増えても enum が肥大化しない（`RetryLater` は
  第6回レビュー反映③で利用経路ゼロのため導入見送り）。
- RebuildThread の既存ループ構造をほぼ変えずに済む。

**懸念・判断材料:**
- **Generation 二重チェック:** `evaluateDeferred` の generation チェックと
  `admission_.evaluate`（再送時の最終防御）の generation チェックが重複する。
  → `evaluateDeferred` は「破棄理由の明確化 + 再送前の早期遮断」、
  `evaluate` は「不変条件の最終防御」と住み分ける（二重でも安全、コスト無視可）。
- **Sequence Guard の実効性（調査確定）:** deferred は単一 slot で上書きされるため、
  sequence 追い越しが実運用で発生するか検証した。結果:
  - sequence は `reserveRuntimePublicationIdentity()`（AudioEngine.h:3349）で採番、
    commit 成功時のみ `lastCommittedPublicationSequence_` が進む
    （`AudioEngine.Commit.cpp:397` の `publishAtomic`）。
  - **deferred 滞留中に非 deferred 経路が commit する経路を実コードで確認:**
    `publishIdleWorldOnly`（`AudioEngine.Transition.cpp:10-30`）は
    `commitRuntimePublication`（`Transition.cpp:25`）を呼び sequence を進める。
    呼び出し元は `Timer.cpp:1593`（crossfade timeout recovery: HardReset）と
    `CtorDtor.cpp:72`（RestoreStep2: HardReset）。
  - さらに `Timer.cpp:915`（fade 正常完了の inline build→commit: SmoothOnly）も
    `commitRuntimePublication` を呼ぶ。
  - → deferred 滞留中に上記のいずれかが実行されれば
    `metadata.sequence < lastCommittedPublicationSequence` となり追い越し検出は
    **成立する**（例: crossfade timeout 回復時は HardReset の publishIdleWorldOnly が
    先に commit → その後 consume した deferred は StaleSequence で破棄される）。
  - → **防御的チェックとして維持推奨**（metadata 情報が現在未使用であること自体がバグ）。
- **TTL 30秒の妥当性（調査確定）:** crossfade の fade 時間は `CrossfadePolicy` の
  最大値で、実測上限は ~0.080s（`m_irFadeTimeSec=0.080` が最大、
  AudioEngine.h:2187-2193）。TTL 30秒はその約375倍のマージンがあり、Timer ポーリング
  （1ms tick）や crossfade 完了検知遅延を考慮しても十分。→ **現行値で妥当。**
- **peek と consume の atomicity:** peek → evaluate → consume の間に deferred slot が
  変わらない前提。調査確定: `submitPublishRequest`（enqueue 元）は RebuildThread のみ
  （`AudioEngine.Commit.cpp:709` ← `RebuildDispatch.cpp:847`）で、consume も同スレッド。
  CoordinatorLoop は `hasDeferredRequest()`（atomic）で通知するのみ。→ **同一スレッド内
  で完結し、peek と consume の間に他スレッドが slot を変更する経路はない。**
  View は stack 上の一時オブジェクトとして使い切り、`State` enum（`Valid` / `Consumed` /
  `Discarded` / `MovedFrom`、第9回反映）で多重呼び出し・二重解放を防ぐ。
  ★ **Single Thread Owner 契約（第7回反映③・const& 化の根拠）:** 上記調査は
  `metadata()` を const 参照で返すための正当性根拠である。**enqueue / peek /
  evaluate / consume の全操作が RebuildThread に閉じ、slot は Single Thread Owner
  （Single Writer / Single Reader より強い契約）**。`clearDeferredForShutdown` は
  RebuildThread 停止後（`Timer.cpp:1521/1652`・`CtorDtor.cpp`・`ReleaseResources.cpp`）の
  ため、並行改変はない。この契約を将来破る場合（Coordinator / Timer からの peek 等）は
  値返しに戻し、設計レビューで契約変更を確定すること（本文 A-3 コメント参照）。
  ※ const& 化の根拠は本契約のみではない（第9回反映①）: 「View が slot 所有権を
  握る」(a) と「consume/discard 後の metadata 禁止」(b) が前提で、本契約 (c) は
  「View 寿命中の slot 非変更」を保証する部分。
- **Slot 所有権プロトコル（第10回レビュー反映・第11回で所有権概念へ整理・第12回で
  Authority 集約）:** 同時複数 View の禁止を実現するため、所有権状態は **Orchestrator 側
  の DEBUG 専用 `DeferredSlotOwnership slotOwnership_`**（enum: `Released` / `Borrowed`）
  が保持する（View 自身は owner_ と state_ のみ）。bool ではなく enum にするのは、assert
  メッセージ・ログで Slot の所有権状態が判別できるようにするため（**Semantic Single
  Source**・ISR 第11回観点）。**atomic にしない** — Single Thread Owner 契約で並行
  アクセス経路が存在しないため、plain enum で十分。解放規約は「peek 取得成功 =
  Borrowed → consume/discard = Released、~DeferredPublishView() は state_ == Valid
  のときのみ Released、move は移転先へ引継ぎ」の3経路（不変条件 8・9 に図示）。
  **peek 取得後に consume/discard せずに破棄すると DEBUG assert**（fail-fast・第13回③）。
  `peek()→evaluate()→consume/discard()` 以外のルートはないため peek-only は合法ではない。
  （所有権は destructor 内で reset されるためリークはしないが、意図的欠陥として検出する。）
  現行 View は Slot の**所有権 Authority** を握る形（Ownership も Authority）。
  `DeferredToken` 命名は将来検討（実装は View で十分・第11回観点）。
  ★ **Authority Singularization（第12回反映・第13回④ 強化）**: View は所有権フラグを直接書き換えず、
  `owner_->finishView()` を呼ぶのみ。フラグ更新は Orchestrator 側の `finishView()`（内部で
  `releaseSlotOwnership()`）が行う。**しかし finishView() は caller-visible な2段階ではない** —
  consume()/discard() は終端で finishView() を内部原子呼出しし、state_ 遷移と ownership
  Release を**1つの不可分操作**として返す。caller（RebuildThread）は `view->consume()` **一回**で
  済み、中間状態を観測できない。（現行ソース `RebuildDispatch.cpp:844-848` の
  `consumeDeferredRequest()→submitPublishRequest()` 2段は legacy two-step であり、
  新設計でこの隙を閉じる。）「View が Owner を書き換える」のではなく「Owner が Owner を管理する」。
  所有権状態遷移（不変条件 8）は `DeferredPublishViewStateMachineTests` で全遷移を自動検証する。
- **破棄理由の記録タイミング:** `view.discard(reason)` は slot 解放時に
  `DeferredDiscardInfo`（reason / ageMs / overwriteCount）を生成して返す
  （slot には理由を記録しない。第5回レビュー反映①）。Telemetry 反映は
  TelemetryRecorder の `recordDeferredDiscard`（Adapter）が `view.discard` の直後に
  受けて記録する。
- **Coordinator が TelemetryRecorder を直接知る（第6回レビュー指摘・第7回で保留確定）:**
  RebuildThread（Coordinator 側）が `telemetryRecorder().recordDeferredDiscard(...)` を
  直接呼ぶのは「Coordinator → TelemetryRecorder」の直接依存で、責務分離としては
  「Coordinator → Telemetry Event → Recorder」の間接化が理想（第6回レビュー Part A 指摘）。
  **第7回レビュー判断（反映④）: Telemetry Event 化は今回は見送り。** 理由:
  (a) Part A の本題は stale discard であり、Telemetry Event 化は**独立した Layer 変更**
  （責務境界そのものを変える）。(b) 既に `recordDeferredDiscard` / `recordDeferredEnqueue`
  （Telemetry Adapter）で「Coordinator は `DeferredHealth` 型を知らない」を達成済みで、
  依存は「Coordinator → DeferredDiscardInfo → TelemetryRecorder」。
  → 今回の Phase 1 は現行を維持し、Event 化（「Coordinator → TelemetryEvent →
  Recorder」）は将来スコープとして Appendix D-7 に記録。

### B-4. Part B: 現状分析・設計判断メモ

**現状分析:**

`onTransitionComplete` (`DSPTransition.h:131-165`) は呼び出し元ゼロの legacy。
`notifyTransitionComplete` 削除後、唯一の到達経路が消えた。

関数の3責務:
1. **CAS-based fading slot clear** (`fadingRuntimeDSPSlot` → nullptr)
2. **crossfade snapshot 更新** (`setDryHoldSamples(0)` + `refreshCrossfadePreparedSnapshotFromAtomics`)
3. **idle world publish** (`publishIdleWorldOnly(currentAfterFade, HardReset)`)

**既に重複実装されている場所:**
- `AudioEngine.Timer.cpp:880-919` — fade complete 時:
  CAS クリア + `complete()` + `setDryHoldSamples(0)` + `refresh...` + inline build→commit (SmoothOnly)
- `AudioEngine.Timer.cpp:1547-1598` — crossfade timeout recovery:
  CAS クリア + `complete()` + `setDryHoldSamples(0)` + `refresh...` + `publishIdleWorldOnly(HardReset)`
  ※ timeout **検知**は `RuntimeHealthMonitor::checkCrossfadeTimeout`（30秒、
  `RuntimeHealthMonitor.cpp:530-547`）が `EVENT_CROSSFADE_TIMEOUT` を emit →
  `onHealthEvent` がこの分岐へ誘導。**回復**は Timer 側。
- `AudioEngine.Timer.cpp:1011-1027` — fade 完了監視: CAS クリア + observe submit
- `AudioEngine.CtorDtor.cpp:135` — 破棄時: CAS クリア
- `AudioEngine.Processing.ReleaseResources.cpp:144` — shutdown: CAS クリア

**設計方針:**

**onTransitionComplete 自体は「削除する」。**
その処理は Timer の各経路に既に展開済み（上記）。ただし**重複するボイラープレート
（CAS クリア + snapshot 更新）は残っており、これを一元化するのが整理の本題**。

**設計判断メモ:**
- **「onTransitionComplete を Timer から呼んで一元化」はしない。** fade 完了検知は
  Timer の複数経路（正常完了・タイムアウト・監視）に分散しており、呼び出し時点を
  1箇所に絞れない。むしろヘルパー抽出 + 将来の Intent 化が正しい方向。
- **publish の平滑遷移 (SmoothOnly vs HardReset) は経路で異なる**（正常完了=SmoothOnly、
  タイムアウト=HardReset）。ヘルパー抽出時は publish を含めず、各経路の判断を残す。
  ヘルパーは「CAS クリア（Step 1）+ observe 投入（Step 2）」まで
  （publish は呼び出し側で実行）。Step 1/2 は private メソッドに分離し、
  Phase 2/3 の Intent 化では Step 1 のみ再利用する（本文 B-3 参照）。
- **finalizeFadingSlot の粒度（第4回レビュー対応）:** 従来案は CAS クリア +
  `submitObserve` を1操作に束ねていたが、Phase 2/3 で Intent 化すると
  `submitObserve` 自体が不要になる可能性がある。**「CAS によるフェーディングスロットの
  終端処理」と「observe 投入」を内部で分離**しておくことで、Phase 2/3 移行時は
  Step 1 を再利用し Step 2 を差し替えるだけで済み、Helper の責務再定義や呼び出し元の
  一括変更を回避できる。引数 `bool submitObserve` による分岐よりも、メソッド分離の方が
  Phase 2/3 の変更範囲が小さい（Intent ハンドラは Step 1 のみを呼ぶ）。

## Appendix D. 改訂履歴（レビュー反映メモ）

> 旧「Appendix A. 改訂履歴」から改名（本文再編に伴う番号整理）。

### D-1. 第1回レビュー反映

| 指摘 | 反映内容 |
|------|---------|
| Part A: 判定を consumeDeferredRequest ではなく **PublicationAdmission** へ | A-2/A-3/A-4/A-5 を全面改訂。`evaluateDeferred` を Admission に、consume は純粋な Queue に |
| Part B: ヘルパー名を **finalizeFadingSlot()** に | B-3/B-4/B-5/B-6 の命名を統一 |

### D-2. 第2回レビュー反映

| 指摘 | 反映内容 |
|------|---------|
| ① `peekDeferredSlot()` と `consumeDeferredRequest()` のライフサイクルが曖昧（consume→peek は所有権遷移が二意的） | **peek → evaluate → consume** の順序に統一（A-3/A-4/A-5 改訂）。consume は判定後・Ready 時のみ呼ぶ |
| ② `consumeDeferredRequest` 後に slot を保持するのは「2つの真実」で Authority が曖昧 | **consume = 所有権移転**。Ready 時に consume → submit で slot は即解放。破棄時は discard で slot 解放。保持期間をゼロにする |
| ③ `DeferredPublishSlot` を `PublicationAdmission.h` に置くと Policy が Storage を持つ依存になる | **`DeferredPublishTypes.h` に独立 POD として分離**。Admission と Orchestrator の両方が include（Storage → Policy の依存方向） |
| ④ `evaluateDeferred(const DeferredPublishSlot&, ...)` は Queue 内部構造を知りすぎる | **`evaluateDeferred(const DeferredPublishMetadata&, ...)`** — generation / sequence / enqueue time のみ受け取る。Admission は Queue 構造を知らない |

### D-3. 第3回レビュー反映

| 指摘 | 反映内容 |
|------|---------|
| ① `peekDeferredMetadata()` / `consumeDeferredRequest()` が二つの API で「必ず順番で呼ぶ」プロトコルに依存している | **`DeferredPublishView` 型を導入**（A-3/A-4/A-5 改訂）。`peekDeferred()` が View を返し、`consume()` / `discard()` は View にしか存在しない。順序を API 型レベルで強制。copy 禁止 + `active_` ガードで二重呼び出しも防止 |
| ② `discardDeferred()` が `recordDeferredHealth` を知りすぎる（Queue が Telemetry を持つ） | **Queue から Telemetry 知識を排除**。`view.discard(reason)` は slot 解放 + 破棄情報 `DeferredDiscardInfo`（純粋データ）を返すのみ。Telemetry 記録は TelemetryRecorder の Adapter `recordDeferredDiscard` が担う（※当初は Orchestrator の Coordinator 責務としたが、第4回レビュー改善②で TelemetryRecorder へ Adapter 化） |
| ③ `peekDeferredMetadata()` の返却形式（参照 or 値）が未定義 | **`metadata()` は値コピーを返す契約に確定**。peek 時点のスナップショットで評価し、consume 前に参照が無効化される懸念を排除 |

### D-4. 第4回レビュー反映

| 指摘 | 反映内容 |
|------|---------|
| ① `DeferredDecision` に DiscardReason が混在（enum 肥大化） | **二層構造に分離**（A-3 改訂）: `DeferredDecision` は動作指示のみ（`Ready` / `Discard` / `RetryLater`〔※後述の第6回反映③で RetryLater は削除・2値化〕）。破棄理由は既存 `DiscardReason` が分離保持。`evaluateDeferred` は `DeferredAdmissionResult { decision, discardReason }` を返す |
| ② `recordDeferredDiscard` を Coordinator 責務にすると Orchestrator が Telemetry 型（`DeferredHealth`）を知る | **TelemetryRecorder の Adapter に変更**（A-4/A-5/Appendix 改訂）: `DeferredDiscardInfo → DeferredHealth` 変換は `TelemetryRecorder::recordDeferredDiscard` が担う。Coordinator は破棄結果を Adapter へ渡すだけ（`telemetryRecorder().recordDeferredDiscard(*discardInfo)`） |
| ③ `evaluateDeferred` の公開 API 位置 | **`PublicationAdmission` の private helper `evaluateDeferredImpl()`** として配置。公開 API は薄く1つだけ公開（A-3 で既反映・第4回で再確認） |
| ④ TTL 定数が constexpr リテラル | **chrono 型に置換**（A-4 反映）: `kDeferredPublishTTLUs`（uint64_t）→ `kDeferredPublishTTL`（`std::chrono::microseconds`、`30s`）。`getCurrentTimeUs()` と `.count()` で比較 |
| Part B. `finalizeFadingSlot()` が CAS クリア + `submitObserve` を1操作に束ねる | **内部を2段に分離**（B-3 反映）: Step 1「CAS による終端処理」（`finalizeFadingTermination()`）/ Step 2「observe 投入」（`finalizeFadingSlotObserve()`）を private メソッド分離。統合ヘルパーが両方を順に呼ぶ。Phase 2/3 で CrossfadeCompleteIntent 導入時に Step 2 を Intent へ差し替え可能 |

### D-5. 第5回レビュー反映

| 指摘 | 反映内容 |
|------|---------|
| ① `DeferredPublishSlot::lastDiscardReason` は slot 寿命より長く使われない（discard 直後に reset される）ので Storage の責務ではない | **`lastDiscardReason` を Storage から削除**（A-3 改訂）。破棄理由は `view.discard(reason)` が生成する `DeferredDiscardInfo` にのみ載せる。Storage は純粋なデータ（request + metadata）に保つ。波及として `clearDeferredForShutdown` の slot への `ShutdownDiscard` 記録も廃止し、`DeferredDiscardInfo{ShutdownDiscard}` → `recordDeferredDiscard` に統一（A-4 / Appendix C） |
| ② `metadata()` が値コピーを返す理由をコメントで明記すべき（レビュー時に「なぜコピー？」と言われる） | **スナップショット保証の理由を明記**（A-3 改訂）: peek → evaluate の間に slot が解放/上書きされても評価値が壊れないため。const 参照だとダングリングリスク。POD 3値（16バイト程度）でコピーコスト無視可 |
| ③ `DeferredPublishView` の `active_`（bool）は false が Consumed / Discarded のどちらか区別できない | **`State` enum（`Valid` / `Consumed` / `Discarded`）に変更**（A-3 改訂）。`assert(state_ == State::Valid)` が書け、ログが判別可能に |
| ④ `DeferredPublishView` は move-only とすべき（`vector<DeferredPublishView>` のような誤用防止） | **copy 禁止・move 許可**（A-3 改訂）。stack 上で短命に使用し、consume/discard 後は再利用しない旨を ADR-C4 にも追記 |
| ⑤ `evaluateDeferred` の判定順序（TTL → Generation → Sequence）に理由コメントを | **判定順序の理由を明記**（A-4 改訂）: 「安価 → 高価」「局所 → 全体」の順。TTL は metadata のみで完結（最安）、Generation は Runtime 状態（atomic 読み）、Sequence は Commit 履歴（最も観測コストが高い）。いずれも early return |
| ⑥ `CrossfadeCompletePayload` の CrossfadeId が本当に必要か | **調査確定: 必要**（B-4 反映）: `endCrossfade(CrossfadeId)` / `unregisterCrossfade(CrossfadeId)` の必須引数。DSPHandle は CrossfadeId を内包せず、現行 fade 完了処理（Timer.cpp:864-877）も id を取り回している。Payload は自己完結とし Handler で検索しない（HANDLER-1 維持） |
| 棚卸し: `clearDeferredForShutdown` / `enqueueDeferred` の `DeferredHealth` 記録経路 | **Telemetry Adapter へ統一**（A-4 反映）: shutdown 時は `recordDeferredDiscard`、enqueue 時は新設 `recordDeferredEnqueue` が `DeferredHealth` を生成。Queue / Orchestrator は `DeferredHealth` 型から解放 |

### D-6. 第6回レビュー反映（ISR アーキテクチャ観点・総合 8.8/10）

| 指摘 | 対応判断 | 反映内容 |
|------|---------|---------|
| ① `evaluateDeferred()` を独立公開 API にせず既存 `evaluate()` へ variant 統合できないか | **現行案を維持**（不採用） | A-2 に判断根拠を明記: **Policy は1つだが Policy Entry Point は1つである必要はない**。Authority Singularization が要求するのは「Decision Authority が一か所」「Policy 実装が一か所」であり「Public API が1個」ではない。`evaluate(PublishRequest)` と `evaluateDeferred(DeferredPublishMetadata)` は異なる入力契約を持つ同一 Authority の別 Entry Point。variant 統合は「API 1つ・実装2つ」の見た目だけの単一化で、`visit` 内に巨大 switch を生み、戻り値が非対称（`Decision` vs `DeferredAdmissionResult{decision, discardReason}`）なため第4回確定の二層構造を壊す |
| ② `DeferredPublishView` は単一 slot に対して設計が重い（ハンドル型で十分では） | **現行案を維持**（不採用） | A-3 に判断根拠を明記: View は **Storage View（データ保持）ではなく Protocol View（API 利用順序の型化）**。peek → evaluate → consume の Ownership Transition Protocol を型で表現するもので、slot 個数に依存しない。ハンドル型は consume()/discard() を持つ限り結局 View と等価 |
| ③ `RetryLater` は現時点で利用経路がないため導入見送り（Ready/Discard で十分） | **採用** | `DeferredDecision` を **Ready / Discard の2値のみ**に縮小（A-3 改訂）。`DeferredAdmissionResult` の初期値も Discard に変更。A-4/A-5 の switch から RetryLater 分岐を削除。将来 Queue 多段化時に再追加（ISR: 単純で予測可能な状態機械） |

> **Part B（9.5/10）への反映:** CrossfadeCompleteIntent Handler が `publishIdleWorldOnly()`
> を直接呼ぶより **`PublicationExecutor` へ委譲**した方が良いという指摘は、Phase 2/3
> の設計判断として B-4 に記録する。Handler は publish 実装を知らず、`PublicationExecutor`
> （実在コード `PublicationExecutor.h:17`、`RuntimePublicationOrchestrator.h:177` で
> `executor_` として使用）へ委譲することで将来の Publication Policy 共有に備える
> （Authority Singularization の完成形）。今回の Phase 1 スコープ外のため設計メモとして反映。
>
> **TelemetryRecorder 責務（8.8/10 の指摘）:** Orchestrator（Coordinator）が
> `TelemetryRecorder` を直接知るのは責務漏れという指摘は、理想形「Coordinator →
> Telemetry Event → Recorder」へ向けた将来課題として A-6 懸念に記録済み。今回の
> Phase 1 では `recordDeferredDiscard` / `recordDeferredEnqueue`（Telemetry Adapter）による
> 「Coordinator は `DeferredHealth` 型を知らない」までを達成し、Recorder 自体の間接化は
> 将来スコープとする。

### D-8. 第8回レビュー反映（実装前最終確認・2026-08-06）

| 指摘 | 対応判断 | 反映内容 |
|------|---------|---------|
| ① `PublishRequest` を `PublicationAdmission` のネスト型から独立させる（`DeferredPublishSlot` が直接参照できるように） | **採用** | `RuntimePublicationTypes.h` を新設し `PublishRequest` を独立型として定義（A-3）。`PublicationAdmission` はネスト型を削除して本ヘッダを include、`DeferredPublishSlot` は `PublicationAdmission` を介さず `PublishRequest` を参照。依存方向は Storage → RuntimePublicationTypes / Policy → RuntimePublicationTypes となり、「Storage が Policy を所有する」見た目の依存が構造的に排除される。A-6b / 0章 / Appendix C に追記 |
| ② `finalizeFadingSlotCas()` の命名（CAS は実装手段） | **採用**（第7回⑦の再検討） | `finalizeFadingSlotCas()` → `finalizeFadingTermination()` へ改名（B-3 Step 1）。API に実装手段を漏らさない方針。将来 compare_exchange を使わなくなっても API 名は変わらない。`finalizeFadingSlotObserve()` は残置（B-3/B-4/B-6 参照を更新） |
| ③ `metadata()` に `[[nodiscard]]` を付与 | **採用** | A-3 `DeferredPublishView::metadata()` に `[[nodiscard]]` を付与（consume/discard と同等の扱い）。戻り値を捨てるだけの呼び出しをコンパイル時に警告 |
| ④ `state_` ガードの明文化（Consumed/Discarded 後の metadata() 呼び出し禁止） | **採用** | A-3 `metadata()` コメントに **State::Valid のときのみ呼べる** 旨と、Consumed / Discarded 後に呼んだ場合に参照がダングリングすることを明記。`view.consume()` の後に `view.metadata()` を書くのは不変条件違反 |
| ⑤ `DeferredDiscardInfo` に `enqueueTimestamp` を持たせない（破棄理由は既に十分） | **妥当・維持** | 変更なし。`DeferredPublishMetadata.enqueueTimestampUs` は判定専用・`DeferredDiscardInfo` には ageMs / overwriteCount / reason のみ（Telemetry に必要十分な情報量）。Appendix B-2 の責務分担どおり |
| ⑥ Telemetry を Adapter でなく Event 化する | **今回見送り**（第7回④と同結論） | 変更なし。Phase 1 は Adapter（`recordDeferredDiscard` / `recordDeferredEnqueue`）で「Coordinator は `DeferredHealth` 型を知らない」を達成済み。Event 基盤は既存 `HealthEvent` / `RebuildTelemetryEvent` と統合できるタイミングで導入（D-7 将来課題を継承） |
| ⑦ `clearDeferredForShutdown` は `discard()` を再利用しない（独立実装） | **妥当・維持** | 変更なし。shutdown は全 slot 一括・理由固定（`ShutdownDiscard`）で、個別 slot の Protocol View 経由とは経路が異なるため独立実装を維持。A-5 の設計どおり |
| ⑧ `evaluateDeferred` の Entry Point | **妥当・維持** | 変更なし。専用 Entry Point + private helper `evaluateDeferredImpl()`（Policy は1つ）の構造を維持。A-2 / A-6 / Appendix B-2 の判断根拠どおり |

**第8回での更新箇所一覧（設計本文）:** A-3（`RuntimePublicationTypes.h` 新設・`metadata()` `[[nodiscard]]` + ガード明文化・`PublishRequest` 独立参照）/ B-3（Step 1 改名）/ B-4（Intent ハンドラ参照）/ B-6・0章・A-6b（影響ファイル表に `RuntimePublicationTypes.h` 追加・改名反映）/ Appendix C（`PublishRequest` 行追加）。

### D-9. 第9回レビュー反映（総合評価 A・実装前最終調整・2026-08-06）

> 第9回レビュー: 総合評価 **A（設計として実装可能）**。Storage / Policy 分離・
> PublishRequest 独立・metadata 渡し・Decision/Reason 二層・View パターン・move-only・
> 暗黙 discard なし の各点は賛成。実装前に調整を勧める5点（下記①〜⑤）を反映。

| 指摘 | 対応判断 | 反映内容 |
|------|---------|---------|
| ① `metadata()` の const& 安全性コメントが「Single Thread Owner だけが理由」と読める | **採用** | 安全根拠を3点 (a)(b)(c) に明確化: (a) View が slot 所有権を握る（Protocol View）+ (b) consume/discard 後の metadata 禁止（state_ ガード）+ (c) Single Thread Owner 契約で View 寿命中の slot 非変更を保証。A-3 View コメント・スレッド所有権契約ブロック・B-2 利点/懸念・不変条件 3 を修正 |
| ② `consume()` / `discard()` の失敗理由が全て `nullopt` で消える | **採用（Phase 1 は現行維持・将来余地を記録）** | コメントに「nullopt は state invalid / slot empty / double consume の全失敗を同一表現にする」旨と、将来 `std::expected<PublishRequest, ConsumeError>` / `DiscardResult` へ拡張する余地を明記（A-3）。Phase 1 は単一正当経路（peek → evaluate → consume）しかなく assert で防御可能なため nullopt で十分。**（第13回棚卸し確定）`std::expected` は C++23 導入（`__cpp_lib_expected` = 202202L）であり、本プロジェクトは C++20（`cxx_std_20`）のため、Result 型拡張の実現には C++23 昇格または tl::expected / kz::expected 等のポリフィル（C++20 で動作・`kz::expected` 等）が前提となる点を制約として記録** |
| ③ move 後の `DeferredPublishView` の状態仕様が曖昧 | **採用** | `State` enum に **`MovedFrom`** を追加。move 後は `owner_ = nullptr` + `State::MovedFrom`。MovedFrom に対する metadata()/consume()/discard() は契約違反（DEBUG assert、release は UB）。move セマンティクスをコメントで明文化（A-3・不変条件 6/8） |
| ④ 同時に複数の View 取得（`auto a=peek(); auto b=peek();`）を防ぐ契約 | **採用** | `peekDeferred()` に「有効な View を複数同時保持禁止」の契約を明記。実装は DEBUG ビルドで View 生成時に借用カウンタを検査し、2枚目（借用中の再 peek）を assert で検出（release は契約違反 = UB）。不変条件 9 として追加 |
| ⑤ `valid()` 等の状態照会 API を公開しない | **採用** | `DeferredPublishView` の公開 API は `metadata()` / `consume()` / `discard()` の3つのみに確定。`bool valid()` 等は追加しない（if (view.valid()) の状態分岐で Protocol 中心設計が崩れるため）。状態確認は内部実装と assert に留める。不変条件 10 として追加 |

**第9回での更新箇所一覧（設計本文）:** A-3（metadata() 安全根拠 (a)(b)(c)・consume/discard 将来 Result 余地・`State::MovedFrom` 追加・move セマンティクス・`valid()` 不公開・peekDeferred 同時取得禁止契約）/ 0章 不変条件 3・6・8・9・10 / B-2（metadata const& の根拠記述）。

**棚卸し検証（第9回・ソースコード照合結果）:** 設計書が引用する全 `ファイル:行番号` を
WSL rg / sed で実コードと照合し、次の9件の行番号ずれと1件の事実誤認を修正した。

- 行番号ずれ（修正済み）:
  - `RebuildDispatch.cpp:1050` → **`:847`**（`submitPublishRequest` の deferred 再送呼び出し）
  - `AudioEngine.Commit.cpp:394-397` → **`:397-398`**（`lastCommittedPublicationSequence_` 更新）
  - `AudioEngine.h:3345` → **`:3349`**（`reserveRuntimePublicationIdentity`）
  - `Transition.cpp:24` → **`:25`**（`commitRuntimePublication` 呼び出し）
  - `CtorDtor.cpp:71` → **`:72`**（`publishIdleWorldOnly(HardReset)`）
  - `Timer.cpp:1592` → **`:1593`**（timeout recovery の `publishIdleWorldOnly`）
  - `Timer.cpp:914` → **`:915`**（fade 完了 inline build→commit）
  - `Timer.cpp:849-899` → **`:865-876`**（fade 完了処理ブロック）
  - `RuntimePublicationOrchestrator.cpp:387` → **`:385`**（`DeferredGuard` 初期化）
- 事実誤認（修正済み）: **「`admission_.evaluate` は既に generation + sequence 比較を持つ」は誤り**。
  現行 `evaluate`（`PublicationAdmission.cpp:6-57`）は Shutdown → Generation → DSP finalized →
  HealthState → Pressure → Fading の判定で **generation 比較のみ**を持ち、sequence 比較は存在しない
  （`getLastCommittedPublicationSequence()` は `AudioEngine.h:1592` に存在するが evaluate は参照しない）。
  → Appendix A ①・B-2 方針を修正: sequence guard は deferred 専用の**新規ロジック**として
  `evaluateDeferred` に集約する（generation と同型の Policy 層判定。重複実装ではない）。

**整合が確認できた主な参照（照合済み）:** `PublicationAdmission.h:17-24`（PublishRequest）/
`:53`（evaluate）/ `RuntimePublicationOrchestrator.h:21-24`（DeferredGuard）/ `:27-32`
（DeferredPublishSlot）/ `:48`（kDeferredPublishTTLUs）/ `:81-88`（hasDeferredRequest/
consumeDeferredRequest）/ `:162`（hasDeferred_ member）/ `RuntimePublicationOrchestrator.cpp:
360-401`（enqueueDeferred）/ `:390`（enqueueTimestampUs）/ `:404-424`
（clearDeferredForShutdown）/ `AudioEngine.Commit.cpp:701`（PublishRequest 構築）/ `:709`
（submitPublishRequest）/ `DSPTransition.h:138-165`（onTransitionComplete、呼び出し元ゼロを
再確認）/ `TelemetryRecorder.h:122-128`（DeferredHealth）/ `ISREvidenceExporter.cpp:231-244`
（buildDeferredHealthJson）/ `ISRRuntimePublicationCoordinator.h:172-177`（IntentType 4種）/
`ISRIntentDispatcher.h:58-66`（kDispatchTable + static_assert）/ `RuntimeHealthMonitor.cpp:
530-547`（checkCrossfadeTimeout）/ `AudioEngine.Transition.cpp:10-30`（publishIdleWorldOnly）/
`AudioEngine.h:2187`（m_irFadeTimeSec=0.080）/ `:2372`（rebuildRequestGeneration）/
`ISRDSPHandle.h:79`（CrossfadeId）/ `RuntimePublicationState.h:10-16`（DiscardReason =
None/ShutdownDiscard/StaleDiscard/SupersededDiscard/Expired）/ `Timer.cpp:1521/1652`
（clearDeferredForShutdown）/ `CtorDtor.cpp:135`・`ReleaseResources.cpp:144`（CAS 終端）。

### D-10. 第10回レビュー反映（総合評価 A・borrow ライフサイクル確定・2026-08-06）

> 第10回レビュー: 総合評価 **A（実装可能）**。Storage / Queue / Policy / Telemetry の4層
> 分離・Protocol View・Decision/Reason 分離・Part B 段階移行・evaluateDeferred の
> TTL→Generation→Sequence 順・`finalizeFadingTermination`/`submitObserve` 分離の各点は
> 一貫していると評価。実装前に勧める補強は **`DeferredPublishView` の借用（borrow）
> ライフサイクル仕様の明文化**のみ。推奨2点（下記①②）を反映。それ以外は変更なし。

| 指摘 | 対応判断 | 反映内容 |
|------|---------|---------|
| ① View の「借用状態」の持ち場所と解放タイミングが未確定（デストラクタが borrow を解除するか曖昧） | **採用** | 借用状態は **Orchestrator 側の DEBUG 専用 `bool viewBorrowed_`** が保持（View 自身は owner_ / state_ のみ。**atomic 不要** — Single Thread Owner 契約で並行アクセス経路なし・plain bool で十分。release は契約違反 = UB の既存方針と同一）。解放規約を明文化: **peek 取得成功 = true → consume()/discard() = false、`~DeferredPublishView()` は state_ == Valid のときのみ false**（Consumed/Discarded は consume/discard が既に解除・MovedFrom は移転先が引継ぎ）。`releaseBorrow()` ヘルパーで集約。→ A-3（Orchestrator 借用管理ブロック・peekDeferred コメント・View デストラクタ宣言）/ 不変条件 8・9 / B-2（借用管理）に反映 |
| ② 状態遷移図に borrow を追加して実装漏れを防ぐ | **採用** | 不変条件 8 の状態遷移図を更新: Valid 生成時 = 借用開始（true）、consume/discard = false、move = 移転先へ引継ぎ（true のまま）、デストラクタ = 状態によらず borrow のみ返却（Valid のとき false）。「auto view = peek(); return;」でも次回 peek が assert にならない旨を明記 |
| ③（参考）`metadata()` は `assert(state_ == Valid)` だけでなく `assert(owner_ != nullptr)` も入れる | **採用** | MovedFrom は state = MovedFrom / owner = nullptr の二重防御だが、実装が `owner_->slot_` へアクセスするため owner チェックも併記。A-3 metadata() 呼び出し条件に「DEBUG: `assert(owner_ != nullptr); assert(state_ == State::Valid);`」を明記 |
| ④（参考）`view.discard` が slot に理由を記録しない点・`DeferredAdmissionResult { decision, discardReason }` の拡張性・Telemetry Adapter の依存方向・`DeferredPublishTypes` が `RuntimePublicationTypes` に依存する構造 | **評価のみ・変更なし** | 変更不要（現設計のまま）。RetryLater / Throttle / Blocked 等の Decision 追加時に DiscardReason を増やさず済む点、Queue → POD → Telemetry の依存方向、`RuntimePublicationTypes ↑ Storage / Policy` の依存が素直である点は現行を維持 |
| ⑤（参考）`finalizeFadingTermination()` / `submitObserve` の分離（CrossfadeCompleteIntent 導入時に Step 1 だけ流用可能）・API 名から CAS を除去した点 | **評価のみ・変更なし** | Part B の Phase 1 構成は現行のまま。Phase 2/3 で CrossfadeCompleteIntent 導入時に Step 1（終端検出）のみ再利用する方針を Appendix B-4 が既に記述済み |

**第10回での更新箇所一覧（設計本文）:** 0章 変更の要点（State enum に MovedFrom を補完）/
不変条件 8（状態遷移図に borrow を追加）・9（借用解放規約を明文化）/ A-3（Orchestrator 側
`viewBorrowed_` + `releaseBorrow()` 追加・peekDeferred 借用管理コメント・`~DeferredPublishView()`
宣言と借用解除契約・consume/discard の releaseBorrow ・move の borrow 引継ぎ・metadata() の
owner_ assert）/ B-2（借用管理の設計判断を追記）。

**棚卸し補足（第10回・ソース照合）:** 既存の `AudioCallbackAuthorityView`
（`AudioEngine.h:2146`）は `CrossfadePreparedSnapshot` を保持する**値スナップショット型**
（借用概念なし・owner ポインタなし）であり、借用型 View である `DeferredPublishView` とは
別概念であることを確認。borrow / lease の既存前例はコードベースに存在しないため、
新規導入の用語として `viewBorrowed_` / `releaseBorrow()` を採用（命名衝突なし）。

### D-11. 第11回レビュー反映（ISR アーキテクチャ観点・総合 A／3点の実装前修正・2026-08-06）

> 第11回レビュー: ConvoPeq.md（ISR 設計文書）前提での評価。Part A（Admission 統合）A・
> View 化 A-・Telemetry Adapter A・Storage 分離 A・Part B A・CrossfadeCompleteIntent A
> と総じて高評価。ただし実装前に修正を推奨された3点（下記①〜③）を反映。これは
> 「方向性は非常に良いが、そのまま実装するには ISR 設計上いくつか修正すべき点がある」と
> いう結論に対する対応。**評価の低い2点は ①（B-・責務漏れ）と ②（B・実装都合）で、
> ③（B・ISR 的注意点）は ADR レベルの契約明文化。**

| 指摘 | 対応判断 | 反映内容 |
|------|---------|---------|
| ① `evaluateDeferred(AudioEngine&)` は Policy が Engine 全体へ依存する危険（評価 **B-**） | **採用** | 新規 `DeferredAdmissionSnapshot`（`currentGeneration` / `lastSequence` / `shutdownInProgress` の3値のみ）を導入し、`evaluateDeferred(const DeferredPublishMetadata&, const DeferredAdmissionSnapshot&)` に変更。TTL 用現在時刻は Engine 非依存の `core/TimeUtils.h::getCurrentTimeUs()` を Policy が直接読む。呼び出し元（RebuildThread / A-5）が engine の3アクセサ（`rebuildRequestGeneration`:2372 / `getLastCommittedPublicationSequence`:1592 / `isShutdownInProgress`:1460）でコンテキストを構築して渡す。→ A-3（PublicationAdmission.h ブロック）/ A-4（判定ロジック）/ A-5（呼び出し側）/ B-2 / 不変条件 3 / 実装影響ファイル表 / ADR-C4 に反映。※ 現行 `evaluate` は `AudioEngine&` + `RuntimeReaderContext&` の既存契約であり、新規 API のみ Engine 非依存を先取り |
| ② Borrow 管理が「実装都合」（DEBUG bool）で Ownership 概念でない（評価 **B**） | **採用** | `bool viewBorrowed_` を **`enum class DeferredSlotOwnership { Released, Borrowed }` の `slotOwnership_`**（Orchestrator 側・DEBUG 専用・atomic 不要）へ昇格。bool → enum で assert メッセージ・ログの状態判別を可能にし、Slot の所有権状態を Semantic Single Source で表現。`releaseBorrow()` → `releaseSlotOwnership()`。View は Slot の**所有権 Authority** を握る旨を明記。`DeferredToken` 命名は将来検討（現行 View で十分）。→ A-3（Orchestrator ブロック・peekDeferred・View コメント）/ 不変条件 8・9 / B-2 に反映 |
| ③ `metadata()` const& を「現状限定」として ADR レベルで固定（評価 **B**） | **採用** | 安全根拠を「(a) **View が slot 寿命を保証する** + (b) consume/discard 後禁止 + (c) Single Thread Owner」の3点として ADR-C4 に契約として明文化。**将来スレッドモデルが変わる場合（Coordinator / Timer が slot を読む等）は値返しへ戻す**ことを設計契約として固定。→ A-3 metadata() コメント・不変条件 3・ADR-C4 に反映 |
| ④（参考）`DeferredPublishView` は Queue API と Ownership API を兼ねる。ISR では Ownership も Authority なので `DeferredToken` がより理想（命名レベル） | **一部採用（将来検討）** | 現行は View で十分（レビューも「実装上は View でも十分」と明記）。A-3 コメント・不変条件 9・ADR-C4 に「View は Slot の所有権 Authority を握る・DeferredToken 命名は将来検討」を明記。名称変更は行わない（Phase 1 は View で実装） |
| ⑤（参考）現行 `evaluate` も ReaderContext 経由である点 | **調査確定** | ソース照合の結果、現行 `evaluate` は `AudioEngine&` + `const RuntimeReaderContext&` の両方を受け取る（`PublicationAdmission.h:55`・`RuntimeReaderContext.h` は RCUReader + ObserveChannel を束縛する軽量コンテキスト）。ctx は RCU 読み取りハンドルの生成に使用され、Engine 参照も併用する。→ ①で `evaluateDeferred` は Engine 非依存として新設し、既存 `evaluate` は現行契約を維持 |

**第11回での更新箇所一覧（設計本文）:** 0章 変更の要点（判定シグネチャ・Queue API の
所有権ライフサイクル表記）/ 不変条件 3（Policy Engine 非依存）・8（状態遷移図を所有権
表現に）・9（`DeferredSlotOwnership` enum） / A-3（`DeferredAdmissionSnapshot` 追加・
evaluateDeferred シグネチャ変更・`DeferredSlotOwnership slotOwnership_` / `releaseSlotOwnership()`
・View コメントの所有権化・metadata() に ADR 契約参照）/ A-4（ctx ベース判定）/ A-5
（コンテキスト構築）/ B-2（所有権プロトコル・sequence 参照の ctx 化）/ 実装影響ファイル表。
**ADR-C4 を更新:** evaluateDeferred の Engine 非依存・metadata() const& の現状限定契約・
Slot 所有権プロトコルを「設計指針」として確定記録。

**棚卸し検証（第11回・ソース照合）:** `PublicationAdmission.h:55` の現行 `evaluate` が
`AudioEngine&` + `RuntimeReaderContext&` を取ることを照合（レビュー主張「ReaderContext 経由」
の妥当性確認）。`core/RuntimeReaderContext.h` が RCUReader + ObserveChannel の束縛
コンテキストであることを確認。`getCurrentTimeUs()` が `core/TimeUtils.h` の独立ユーティリティ
（Engine 非依存・Policy から直接使用可）であることを確認。`rebuildRequestGeneration`
（`AudioEngine.h:2372`・`std::atomic<int>`）・`getLastCommittedPublicationSequence`
（`AudioEngine.h:1592`）・`isShutdownInProgress`（`AudioEngine.h:1460`）の3アクセサを照合し、
`DeferredAdmissionSnapshot` の3値が全て実在することを確定。

### D-12. 第12回レビュー反映（ConvoPeq.md 比較・ISR 設計純度・4点修正・2026-08-06）

> 第12回レビュー: ConvoPeq.md（ISR 設計文書）との照合で総合 8.5-9/10。Direction は
> 良好だが「ISR 設計の純度」の観点から4点の修正を推奨された。推奨4点（下記①〜④）を
> 全て反映。

| 指摘 | 対応判断 | 反映内容 |
|------|---------|---------|
| ① `evaluateDeferred()` が `getCurrentTimeUs()` を直接呼ぶのは「Policy が Runtime 外（時刻）を読む」点で完全スナップショット判定でない | **採用** | `DeferredAdmissionSnapshot` に **`nowUs`（uint64_t）を追加**し4値化。TTL 判定を `ctx.nowUs` ベースに変更。時刻取得の Authority は呼び出し元（RebuildThread / A-5）に置き、Policy は渡された値のみで**決定論的**に判定（同一 ctx で何度呼んでも同一結果）。→ 不変条件 3 / A-3（ctx 定義）/ A-4（TTL 判定）/ A-5（ctx 構築2箇所）に反映 |
| ② `DeferredPublishView` が `owner_->releaseSlotOwnership()` を直接呼ぶのは「View が Owner を書き換える」逆転（Authority が曖昧） | **採用** | Orchestrator に**公開 API `finishView()`** を追加。View の consume / discard / デストラクタは `owner_->finishView()` を呼ぶのみで、所有権フラグ更新は Orchestrator 自身が行う（内部で `releaseSlotOwnership()`）。「Owner が Owner を管理する」（Authority Singularization）を明文化。→ A-3（finishView 追加・View コメント3箇所）/ 不変条件 9 に反映 |
| ③ `DeferredPublishView` のライフサイクル（Valid/Consumed/Discarded/MovedFrom）が複雑で実装漏れの恐れ | **採用** | 不変条件 8 に**全遷移の状態遷移表**（16行）を追加。Phase 1 実装時に `src/tests/DeferredPublishViewStateMachineTests.cpp` へ1遷移1ケースで自動検証（既存 `src/tests/*Tests.cpp` と同じ `main()` 単体テスト + `add_executable` / `add_test` 形式）。特に「非 Valid 状態後の公開 API = 契約違反」と「Valid のままのデストラクタ = slot は残る + Ownership のみ Released」を固定 |
| ④ `PublicationAdmission` は今後責務が増える（evaluateDeferred 以外の Policy 系） | **採用（将来余地のみ）** | A-3 PublicationAdmission.h ブロックに「**将来 `PublicationPolicy` へのリネーム余地**・内部 Policy 分割（evaluate 系 / 他）を見据えた構成」を明記。**現時点ではリネームしない**（既存参照を増やさない。境界が固まったら単独コミットで実施） |

**第12回での更新箇所一覧（設計本文）:** 0章 変更の要点（判定4値化・Queue API の finishView）/
不変条件 3（完全スナップショット化）・8（状態遷移表 + テストファイル）・9（finishView + Authority 集約）/
A-3（ctx に nowUs・finishView 追加・View コメント・PublicationPolicy 将来余地）/ A-4（TTL 判定 ctx.nowUs）/
A-5（ctx 構築2箇所に nowUs）/ 実装影響ファイル表（テストファイル追加）。

### D-13. 第13回レビュー反映（ConvoPeq(38) 照合・総合 A／ISR 純度2点の改善・2026-08-06）

> 第13回レビュー: 設計書（第12回反映版）と現行 ConvoPeq(38) ソースの双方を照合。
> 責務分離 A・ISR整合性 A-・Single Source of Truth A・Realtime安全性 A・Ownership A-・
> 将来保守性 A・実装容易性 A で「**採用して問題ない設計**」。良い点6つ（stale discard の
> Admission 復帰・Queue の Storage 化・peek→evaluate→consume・Protocol View・metadata のみ
> 受け渡し・Engine 非依存 Context）を全て再評価。**ISR 純度改善 concern 5 点**（Snapshot
> 完全性 / TTL 可変性 / View 寿命 assert / finishView 原子性 / CrossfadeCompleteIntent
> 位置づけ）についてソース照合・確定・反映を実施（詳細は D-13.5）。改善案は**必須修正ではない**
> が設計純度を高めるため採用。ソース引用の棚卸し検証も実施。

| 指摘 | 対応判断 | 反映内容 |
|------|---------|---------|
| ① `DeferredAdmissionContext` は「Snapshot」であることが明確な命名・意味付けにする（例: `DeferredAdmissionSnapshot`） | **採用（命名・意味付け）** | **`DeferredAdmissionContext` → `DeferredAdmissionSnapshot` へ一括リネーム**（設計未実装のため安全）。意味付けを「**Observation Snapshot**（peek 時点の観測スナップショット）」と明文化。ISR の層分離 `Engine ──► Snapshot ──► Policy` を A-3 コメントに図示。→ A-3（定義コメント）/ 0章 / 不変条件 3 / ADR-C4 に反映 |
| ② `metadata()` const& は現契約では妥当だが、将来スレッドモデル拡張に備え「Policy はスナップショットのみを受け取る」原則を明文化 | **採用（設計原則）** | 不変条件 3 に**「Policy はスナップショットのみを受け取る」を設計原則として確定**。Policy への入力は全てスナップショット（① `DeferredPublishMetadata` = enqueue 時点の immutable Snapshot、② `DeferredAdmissionSnapshot` = peek 時点の Observation Snapshot）。`metadata()` const& は「スナップショットへの const 参照」であり、将来スレッドモデルが拡張されても「Policy はスナップショットのみ」を維持（渡し方だけ変える）。→ 不変条件 3 / ADR-C4（設計原則として確定記録）に反映 |
| ③（棚卸し）設計書が引用するソース行番号の再検証 | **調査確定（全引用正確）** | ConvoPeq(38) ソースで再照合。`AudioEngine.h:2372/1592/1460`（rebuildRequestGeneration / getLastCommittedPublicationSequence / isShutdownInProgress）は一致。`AudioEngine.h:2146`（AudioCallbackAuthorityView の完全定義・前方宣言 1209 とは別）・`DSPTransition.h:138-165`（onTransitionComplete 実体定義）・`RuntimePublicationOrchestrator.cpp:360/390/404`（enqueueDeferred / enqueueTimestampUs / clearDeferredForShutdown）・`AudioEngine.Commit.cpp:709`（submitPublishRequest）・`RuntimePublicationOrchestrator.h:48`（kDeferredPublishTTLUs）全て正確。行番号修正は不要 |
| ④（棚卸し確定）`clearDeferredForShutdown` の疑似コードが `reset()` 後に `slot.enqueueTimestampUs` を読む順序バグ | **修正** | `DeferredDiscardInfo` の構築（reason / ageMs / overwriteCount）を `deferredSlot_.reset()` **より前**に移動（A-5 shutdown フローを修正）。reset 後の slot 参照はダングリング。破棄情報の取得は discard() と同様に「reset 前」を明記 |
| ⑤（棚卸し確定）`DeferredPublishMetadata.generation` の型（uint64_t）と `ctx.currentGeneration`（int）の不一致で C4018/-Wsign-compare 警告 | **修正** | `metadata.generation` を **`int` に統一**（`PublishRequest.generation`・`rebuildRequestGeneration`（`std::atomic<int>`）と同一型）。現行 `DeferredGuard.generation`（uint64_t）は enqueue 時の `static_cast<uint64_t>(req.generation)` の産物であり、新設計ではキャスト廃止。A-4 の `m.generation != ctx.currentGeneration` が同型比較になる |

**第13回での更新箇所一覧（設計本文）:** 0章 変更の要点（判定行）/ 不変条件 3（スナップショット原則・ObservationSnapshot 1 点統一）/
A-2（StaleGuard の生成タイミング・Context 4値）/ A-3（`DeferredAdmissionSnapshot` へリネーム・
Observation Snapshot の意味付け・`DeferredPublishMetadata.generation` の int 統一・
state table 106/107 atomic consume/discard・111 destructor **fail-fast assert**・
finishView コメント・Part B 位置づけ）/ A-4（TTL 判定 ctx.nowUs）/
A-5（shutdown フローの reset 順序修正）/
ADR-C4（リネーム + スナップショット原則 + §99 fail-fast + Atomic ownership transition +
Part B Consequences の source-confirmation）。

**棚卸し確定事項（第13回・追加調査）:** ① `hasDeferred_` のライフサイクルは設計済み
（consume:600 / discard:624 / shutdown で false）。**Valid のままの破棄は
DEBUG assert（fail-fast・第13回③）**—「ture 維持＝re-peek」の緩価値を破棄。
② `recordDeferredDiscard` / `recordDeferredEnqueue` は**新規**
Adapter（現行は `recordDeferredHealth(DeferredHealth)` のみ。`TelemetryRecorder.h:121-128`）。
③ `DeferredHealth` は `TelemetryRecorder.h:122`（設計引用どおり）。④ `DiscardReason`
（`RuntimePublicationState.h:10-16`）= None / ShutdownDiscard / StaleDiscard /
SupersededDiscard / Expired で設計使用分と一致。⑤ `nowUs()` は `TelemetryRecorder.cpp:8`
の既存 static ヘルパー（`recordDeferredDiscard` 疑似コードの参照は妥当）。
⑥ A-5 統合点（`AudioEngine.RebuildDispatch.cpp:844-848`）は現行の
consume→submit フロー（two-step, legacy）と一致し、新設計の peek→evaluate→consume
へ置換可能（詳細は D-13.5 ④）。
⑦（D-9 ②補完）`std::expected` は **C++23** 導入（`__cpp_lib_expected` = 202202L）。
本プロジェクトは **C++20**（`CMakeLists.txt` の `cxx_std_20`）のため、Result 型
（ConsumeResult / DiscardResult）拡張は C++23 昇格 or ポリフィル（tl::expected /
kz::expected、C++20 で動作）が前提。A-3 コメント・D-9 表 ② に制約として記録。
⑧（実装前反映）D-13 ⑤ の「現行 `DeferredGuard.generation`（uint64_t）」を**ソース列レベルで先行統一**。
`RuntimePublicationOrchestrator.h:22` の `DeferredGuard.generation` を `uint64_t`
→ `int` に変更し、`RuntimePublicationOrchestrator.cpp:386` の
`static_cast<uint64_t>(req.generation)` を `req.generation` に簡素化。根拠:
AiDex により `guard.generation` は**全コードベースで0読み取り**（書き込みのみ）確認済み。
`PublishRequest.generation`（`int`）・`rebuildRequestGeneration`（`std::atomic<int>`）と
同一型になり、設計 A-4 の `m.generation != ctx.currentGeneration` 比較で
`-Wsign-compare` を予防。セマンティクスは int→int 同型変換（旧 uint64_t への widening
は読み手零のため no-op）。※ sandbox LSP は JUCE ヘッダを未解決（環境的。実ビルドは要 JUCE）。

### D-13.5. 第13回レビュー（ConvoPeq(38) 照合）追伸 ― 5 つのISR純度改善案の調査・確定

> レビュー:「Snapshot 完全性 / TTL 可変性 / View 寿命 / finishView 原子性 / CrossfadeCompleteIntent」の 5 concerns について、ConvoPeq(38) ソースに照合して確定。

| # | Concern（ISR視点） | ソース照合結果 | 確定・反映 |
|---|---|---|---|
| ① | `DeferredAdmissionSnapshot` 4値だけで十分か — 将来 engine-state 増に弱いのでは | `getCurrentTimeUs()` が**全体で95箇所**散在（`enqueueDeferred:368`, `onPublishCommitted:311` 等）。Deferred フローの時間取得は 2 点に集中（enqueue-timestamp vs evaluate-nowUs） | **確定**: `DeferredAdmissionSnapshot` を**唯一の Observation Snapshot POD**として固定。将来フィールドはこの1つの POD へ拡張（散乱させない）。enqueue-timestamp（metadata）と evaluate-nowUs（snapshot）は**意図的に2つの immutable 取得時点**。0章/不変条件 3 / ADR-C4 に「Observation Snapshot は1つ」原則を追記（A-3 0章参照）。 |
| ② | TTL 30秒がハードコード。Offline/Realtime/Network-Sync で異なり得る | `kDeferredPublishTTLUs` = `.h:48` で**定義のみ**（AiDex: 1件＝コメント）。`StaleDiscard`/`Expired` は enum (`.h:13`) **定義のみ**。実隈の TTL 強制評価は**未実装** | **確定**: Phase 0/1 は `constexpr 30s` 維持（ソース一致）。将来 `PolicyTTLUs`（config/policy-driven）への拡張を A-4 / D-9 に**Future note**として記録。 |
| ③ | `peek()` した View を `consume/discard` せず破棄するバグを即検出したい | ソース `consumeDeferredRequest()`（`.h:82`）は**View すらなく** bare `std::optional<PublishRequest>` を返す。State enum / destructor 保護は未実装 | **採用（fail-fast）**: `DeferredPublishView` デストラクタに `DEBUG assert(state_ != Valid)` を追加。peek 後 decide 忘れを即検出。state table 行 111 / ~1205を更新。（所有権リークはない—reset されるが、Valid 寿命は意図的欠陥。） |
| ④ | `finishView()` が caller-visible 2段階（`consume()→finishView()`）なのでは | ソース `RebuildDispatch.cpp:844-848` は**確かに two-step** `consumeDeferredRequest()→submitPublishRequest()`（hasDeferred_ false 済みで slot は reset されないままという実質的ギャップあり） | **確定**: 設計は既に `consume()` 内で `finishView()` を終端原子呼出ししているが、これを**明文化・強化**。state table 106/107 を `Consumed+Released（atomic）` に更新。`finishView()` は **private / View 専用 / caller 非公開**にし、終端で不可分に呼ぶ。`1208-1212`/A-3 finishView コメントを更新。 |
| ⑤ | `CrossfadeCompleteIntent` が「publish 要求」になりがち — ISRでは state notify であるべき | `onTransitionComplete`（`DSPTransition.h:138`）は**呼び出し元ゼロ**（`.h:131-137` コメント自記）。publish 経路は `publishIdleWorldOnly()` → `submitObserve`→Intent。既に `IntentHandlerContext`（`ISRIntentDispatcher.h:21`）＋**無状態 handler singleton** `g_*IntentHandler`＋`kDispatchTable[4]`（`static_assert` ガード, HANDLER-1）により **handler は publish を直接しない**（委譲のみ） | **確認（既構築済み）**: Part B は既に正しい — `CrossfadeCompleteIntentHandler` は HANDLER-1 遵守・`publishIdleWorldOnly()`/`PublishExecutor`へ委譲。`onTransitionComplete` は Phase 0 で削除済み（`.h:131-135`）。B-4/B-5 に「handler は publish 直接実行禁止（委譲のみ）」を明記。 |

**追記 — ソース照合の行番号メモ（第13回 D-13 ③ の微誤再検証）:**
- `rebuildRequestGeneration` は `AudioEngine.h:2371`（設計 D-13 ③ の `:2372` は off-by-1）。`std::atomic<int>` であり `PublishRequest.generation`（`int`, `PublicationAdmission.h:19`）と同型。→ `DeferredGuard.generation` を `int` に統一した**ソース先行反映（D-13 ⑧）**は型不一致の根拠一致。
- `DiscardReason` 列挙は `RuntimePublicationState.h:13-19`（D-13 ④ の `:10-16` は off-by-3）。値は 5 種一致。
- `nowUs()` は `TelemetryRecorder.cpp:8`（anon namespace `uint64_t nowUs() noexcept`）。確定。

### D-13.6. 第13回レビュー（2） ― 実装前確認事項4

> レビュー concern:「peek→evaluate→consume 競合 / Snapshot immutable / submitPublishRequest Admission 再実行 / TTL config」の4点をソース照合して確定。

| # | 確認事項 | ソース照合結果 | 確定（実装指示） |
|---|---|---|---|
| (1) | peek→evaluate→consume 競合 — Single Thread Owner 契約維持 | `enqueueDeferred` は `submitPublishRequest(.cpp:328)` 経由のみ。`submitPublishRequest` は `RebuildDispatch.cpp:846` 内（processDeferredAdmission, RebuildThread）+ `AudioEngine.Commit.cpp:709` の2箇所。`processDeferredAdmission()` は `RebuildDispatch.cpp:846` のみ（旧 `consumeDeferredRequest`）。`hasDeferred_`(atomic) が `deferredSlot_`(non-atomic)のhandshake。`trySubmit`(.cpp:34, audio)は `DeferredFadingActive`を返すが音声スレッドで enqueue しない（design .h:47-49）。 | **契約維持**: `peekDeferred`/`processDeferredAdmission`/`finishView` は Coordinator/RebuildThread のみ。**実装ガード**: 同3関数に `jassert(std::this_thread::get_id() == engine_.rebuildThreadId())` を付与。※ `Commit.cpp:709` の呼び出しスレッドをコメントで明示し handshake 前提を守る。将来 Timer/Coordinator/Recovery-worker が `peek` を始めると handshake 崩壊 → **禁止事項**。State table 106/107 (atomic) はこの前提の上成り立つ。 |
| (2) | Snapshot immutable | `enqueueDeferred(.cpp:368)` で `now`/`sequence`/`generation` を取得し `DeferredPublishSlot` へ格納。design ctx は peek 時点で4値取得。 | **確定**: immutable は Single Thread Owner 契約（peek→evaluate→consume 同一スレッド同期）で保たれる。**実装**: `DeferredAdmissionSnapshot` を `const`値/const-refで渡し、Snapshot 取得後の Engine 再参照をコンパイル/assertで禁止。不変条件 3 に「Snapshot 取得後は Engine を読まない」を追記。 |
| (3) | Ready path が Admission を再実行（二段防御） | `submitPublishRequest(.cpp:316)` → `trySubmitImpl(req)`(.cpp:40, build+execute+admission) → `DeferredFadingActive`なら`enqueueDeferred`。Ready は `trySubmitImpl` で `admission.evaluate` 済みの上 `executor_.publish`。 | **確認OK**: `view.consume() → submitPublishRequest() → trySubmitImpl → admission.evaluate()` 二段防御。design A-4 フロー 588-592 と整合。 |
| (4) | TTL future-config | D-13.5 ② 調査済み。`kDeferredPublishTTLUs(.h:48)` 定義のみ、未強制。 | Future note 済み（A-4/D-9）。Phase 1実装時に `PolicyTTLUs` の extension point を確保。 |

**実装フェーズ (Phase 1) 事前条件:** 上記 (1) の `jassert` スレッドガード + (2)の Snapshot const 強制は、Phase 1 実装の**必須前提**。(3)二段防御は design-D4 A-4 フロー 591-592 が `submitPublishRequest(*req)` を呼ぶことで自動成立。

### D-7. 第7回レビュー反映（2026-08-06）

| 指摘 | 対応判断 | 反映内容 |
|------|---------|---------|
| ① `evaluateDeferred` を維持（専用 Entry Point で十分） | **妥当・維持** | 変更なし。A-2 / A-6 の判断根拠（Policy は1つ・Entry Point は複数可、Authority Singularization との整合）を再確認。第6回反映①と同じ結論 |
| ② `DeferredPublishView`（Protocol View）を維持 | **妥当・維持** | 変更なし。A-3 / A-6 の判断根拠（slot 個数ではなく Ownership Transition Protocol の型化が価値）を再確認。第6回反映②と同じ結論 |
| ③ `metadata()` を const 参照で返す（値コピーは不要では） | **採用（契約明文化付き）** | A-3 改訂: `metadata()` を `const DeferredPublishMetadata&` に変更。第5回の「値コピー=スナップショット保証」は防御的だったが、**DeferredSlot は Single Thread Owner（RebuildThread）**であり（A-6 懸念の調査確定・Appendix A 詳細調査 3）、peek → evaluate → consume の窓で他スレッドによる改変経路がないため不要。**ただし所有権契約を明文化**（peek/consume/discard は RebuildThread 専用・他スレッド参照禁止）。契約を破る（Coordinator / Timer から peek 等）場合のみ値返しに戻す旨をコメントに記載 |
| ④ Telemetry Event 化（Coordinator → Event → Recorder）を今やるべき | **今回見送り・将来課題として記録** | A-6 懸念を更新。理由: (a) 独立した Layer 変更（責務境界の変更）で Part A の本題（stale discard）と異なる Phase。(b) Adapter 化で「Coordinator は `DeferredHealth` 型を知らない」は達成済み。→ 本セクション下の「将来課題」に記録 |
| ⑤ View 簡素化（単一 slot なら重い） | **不採用** | 第6回反映②と同じ結論を再確認。Protocol View の価値（型による順序強制）は slot 個数に依存しない。簡素化は ISR 的利益なし |
    | ⑥ `discard()` ライフサイクル・不変条件の明文化 | **採用（第13回③ で fail-fast へ更新）** | 不変条件 8 を追加: **View のデストラクタは暗黙 discard しない**。consume / discard の呼び出しは Caller の責務。※ **第13回レビュー③ により、Valid のままでの破棄は DEBUG assert（fail-fast）へ更新**（以前の「slot 残留＝re-peek」緩価値は破棄）。slot は destructor 内で reset される（リークなし）が Valid 寿命は意図的欠陥。詳細は D-13.5 ③。状態遷移図（Valid → Consumed / Discarded / 何もしない）を不変条件 8 に記載。`discard()` のコメントにも追記 |
| ⑦ `finalizeFadingSlotCas()` の命名変更 | **任意・今回は現行維持** | ISR 上の意味は不変（命名のみ）。`finalizeFadingTermination()` 等への変更は Phase 1 実装時に検討可。本設計は現行名を維持。※ **第8回で再検討し採用**（→ D-8 ②） |

**将来課題（第7回レビュー反映④・Telemetry Event 化）:**

```
理想形（将来）:
Coordinator ── emitDeferredTelemetry(event) ──► TelemetryRecorder
                  （Telemetry Event / 軽量 Event Seam）
現行（Phase 1）:
Coordinator ── recordDeferredDiscard(DeferredDiscardInfo) ──► TelemetryRecorder（Adapter）
```

- **対象:** `view.discard` 後の破棄通知（`recordDeferredDiscard`）と enqueue 通知
  （`recordDeferredEnqueue`）を、Coordinator から直接の Adapter 呼び出しではなく
  **イベント発行 → 購読**に変更する。
- **着手条件:** 既存 `HealthEvent`（`RuntimeHealthMonitor` → `onHealthEvent` コールバック）
  や `RebuildTelemetryEvent`（`emitRebuildTelemetry`）と整合する Event 基盤の導入時。
  単発の Adapter 呼び出しに Event 基盤を持ち込むのは過剰設計のため、既存 Event 経路と
  統合できるタイミングで実施する。
- **見送り理由（第7回確定）:** Part A の本題（stale discard の Admission 統合）と
  Telemetry Event 化は独立した変更。責務境界そのものを変える Layer 変更であり、
  Phase が異なる。Adapter 化により「Coordinator は `DeferredHealth` 型を知らない」は
  既に達成済みのため、Event 化の遅延による設計負債はない。


## Appendix C. 現行コード対応表

| 概念 | 現行実装 | 設計での扱い |
|------|---------|-------------|
| PublishRequest | `PublicationAdmission.h:17-24`（ネスト型） | `RuntimePublicationTypes.h` へ独立（第8回反映①）。`DeferredPublishSlot`（Storage）は `PublicationAdmission` を介さず本型を参照。Policy も Storage も中立型へ依存 |
| DeferredGuard | `RuntimePublicationOrchestrator.h:21-24`（記録のみ・未読） | `DeferredPublishMetadata`（`DeferredPublishTypes.h`）へ統合。`evaluateDeferred`（Admission）で判定に使用（復活） |
| enqueueTimestampUs | `RuntimePublicationOrchestrator.cpp:390` | `DeferredPublishMetadata.enqueueTimestampUs`。`evaluateDeferred` TTL 判定に使用 |
| kDeferredPublishTTLUs | `RuntimePublicationOrchestrator.h:48`（未使用） | `kDeferredPublishTTL` へ改名し `PublicationAdmission.h` へ移設・chrono 型化（`30s`）し `evaluateDeferred` で有効化（30秒・妥当性確認済み） |
| consumeDeferredRequest | `RuntimePublicationOrchestrator.h:82-88` | `DeferredPublishView::consume()` へ。純粋 Queue 化（判定を Admission へ移動、consume = 所有権移転で即 slot 解放） |
| peekDeferredMetadata | — | `peekDeferred()` → `DeferredPublishView` へ。metadata は const 参照（Single Thread Owner 契約、第7回反映③）で返し、consume/discard は View 経由でしか呼べない |
| discardDeferred | — | `DeferredPublishView::discard(reason)`（slot 解放 + `DeferredDiscardInfo` 返却）+ `TelemetryRecorder::recordDeferredDiscard`（Telemetry Adapter が telemetry 記録） |
| DeferredRetrieveResult | — | `DeferredDecision`（PublicationAdmission）へ改称・配置 |
| DeferredPublishSlot | `RuntimePublicationOrchestrator.h:27-32` | `DeferredPublishTypes.h`（独立 POD）へ移動。`lastDiscardReason` を削除し request + metadata のみに（第5回反映①） |
| lastDiscardReason | `RuntimePublicationOrchestrator.h:30`（discard 時に slot へ記録） | Storage から削除。破棄理由は `DeferredDiscardInfo` にのみ保持 |
| clearDeferredForShutdown | `RuntimePublicationOrchestrator.cpp:404-424`（slot に `ShutdownDiscard` を記録して reset） | `DeferredDiscardInfo{ShutdownDiscard, ageMs, overwriteCount}` 生成 → `recordDeferredDiscard`（Telemetry Adapter）に統一。slot への書き込みは廃止 |
| enqueueDeferred | `RuntimePublicationOrchestrator.cpp:360-401`（`DeferredHealth` を直接生成・記録） | slot 格納 + `deferredOverwriteCount_` / `maxDeferredAgeMs_` 監査のみ。`DeferredHealth` 生成・記録は `recordDeferredEnqueue`（Telemetry Adapter）へ移譲 |
| onTransitionComplete | `DSPTransition.h:138-165`（呼び出し元ゼロ） | Phase 0 で削除 |
| publishIdleWorldOnly | `AudioEngine.Transition.cpp:10` | 維持（各経路の publish 一元） |
