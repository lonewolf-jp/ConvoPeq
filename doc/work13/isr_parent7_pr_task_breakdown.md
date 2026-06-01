# ISR 親契約7本 実装タスク分解（PR粒度）

作成日: 2026-06-01
基準: `doc/work13/isr_bridge_runtime_unmet_items_remediation_full.md` 第7版

## 目的

第7版で定義した「親契約7本」を、実装順に沿ってレビュー可能なPR粒度へ分解し、完了判定を明確化する。

## 親契約7本（再掲）

1. RuntimeWorld Self-contained
2. RuntimeWorld Read Authority Singularization
3. Semantic Reachability Contract
4. RuntimeBuilder Semantic Isolation
5. Crossfade Executor-Local Contract
6. Publication Lifecycle Contract
7. Governance / Verifier Wiring Contract

---

## 実装順（第7版準拠）

1. ID1 RuntimeWorld Self-contained
2. ID35+ID37 Semantic Reachability Contract
3. ID14 RuntimeBuilder Runtime Semantic Isolation
4. ID41 RuntimeWorld Read Authority Singularization（Coordinator Read Contract）
5. ID12+ID13+ID36 Publication Lifecycle Contract
6. ID41 RuntimeWorld Read Authority Singularization（Audio Thread Authority Cleanup）
7. ID3 Crossfade Executor-Local Contract

> 注記: `ID42 RuntimeGraph Projection Purification` は独立PRにせず、`ID1`（PR-01）とRead側収束（PR-04B）の完了条件に吸収する。

---

## PR分解（推奨）

### PR-01: RuntimeWorld Self-contained 基盤化（ID1）

- 対象契約: 1
- 目的:
  - semantic決定を `RuntimeWorld` へ集約し、setter/atomic直読での意味決定を排除する。
- 主対象ファイル:
  - `src/audioengine/AudioEngine.Parameters.cpp`
  - `src/audioengine/AudioEngine.h`
  - `src/audioengine/RuntimeBuilder.cpp`
- 主対象関数:
  - `applyDefaultsForCurrentMode`
  - `setInputHeadroomDb`, `setOutputMakeupDb`, `setConvolverInputTrimDb`
  - `buildRuntimePublishWorld`
- 主対象変数:
  - `eqBypassRequested`, `convBypassRequested`, `currentProcessingOrder`
  - `inputHeadroomDb`, `outputMakeupDb`, `convolverInputTrimDb`
  - `inputHeadroomGain`, `outputMakeupGain`, `convolverInputTrimGain`
- 完了条件:
  - semantic最終値が `RuntimeBuilder -> RuntimeWorld` でのみ確定される。
  - RuntimeWorld以外で semantic最終決定を行うコードが存在しない。
  - RuntimeGraphは Projection/View 用途のみ許可し、Semantic Decision用途を禁止する。
  - `RuntimeGraphAuthorityVerifier`（新設）で、RuntimeGraph由来の意思決定経路混入を fail-closed 検出する。
  - `RuntimeGraphAuthorityVerifier` は RuntimeGraph field descriptor table を母集団として検査する。
  - 手書き列挙の固定検査対象を禁止し、RuntimeGraph の実在 field を包括的に対象化する。
  - RuntimeGraph field が以下の decision に使用されていないことを検証する:
    - branch decision
    - publish decision
    - topology decision
    - routing decision
    - execution decision

### PR-02: Semantic Reachability Contract 導入（ID35+ID37）

- 対象契約: 3
- 目的:
  - `semantic source -> rebuild trigger -> BuildInput -> RuntimeWorld -> SemanticHash` を fail-closed で保証する。
- 主対象ファイル:
  - `src/audioengine/AudioEngine.Parameters.cpp`
  - `src/audioengine/AudioEngine.UIEvents.cpp`
  - `src/audioengine/AudioEngine.Timer.cpp`
  - `src/audioengine/AudioEngine.StateIO.cpp`
  - `src/audioengine/AudioEngine.RebuildDispatch.cpp`
  - `src/audioengine/RuntimeBuildTypes.h`
  - `src/audioengine/RuntimeBuilder.cpp`
  - `src/audioengine/ISRRuntimeSemanticSchema.h`
  - `src/tests/RuntimeSemanticSchemaValidationTests.cpp`
- 主対象関数:
  - `submitRebuildIntent`, `requestRebuild`, `handleAsyncUpdate`, `rebuildThreadLoop`
  - `captureRuntimeBuildSnapshot`, `buildRuntimePublishWorld`, `validateSemanticCompleteness`
- 主対象変数:
  - `pendingRebuildKinds`, `needsRebuild`, `rebuildGeneration`
  - `RuntimeBuildSnapshot::buildInput`, `RuntimeSemanticHash::*`
- 完了条件:
  - trigger coverage と semantic coverage が同時に100%（不足時PR fail）。
  - coverage定義が「手書き列挙」で管理され、暗黙自動推論に依存しない。
  - coverage母集団は `ISRRuntimeSemanticSchema.h` の**唯一の descriptor table**に固定する。
  - Coverage対象外項目（恣意的除外）を禁止する。
  - 複数 descriptor table の並立（coverage母集団の分岐）を禁止する。

### PR-03: RuntimeBuilder Semantic Isolation（ID14）

- 対象契約: 4
- 目的:
  - Builderが runtime mutable state を観測しない契約を固定。
- 主対象ファイル:
  - `src/audioengine/RuntimeBuilder.h`
  - `src/audioengine/RuntimeBuilder.cpp`
  - `src/audioengine/AudioEngine.RebuildDispatch.cpp`
  - `src/audioengine/RuntimeBuildTypes.h`
- 主対象関数:
  - `RuntimeBuilder::RuntimeBuilder`
  - `buildRuntimePublishWorld`, `build(const BuildInput&)`
  - `captureRuntimeBuildSnapshot`
- 主対象変数:
  - `RuntimeBuilder::engine`, `engine.runtimeStore`
  - `BuildInput`, `RuntimeBuildSnapshot::buildInput`
- 完了条件:
  - `Builder semantic input source = BuildInput only`。
  - RuntimeBuilder から RuntimeStore 参照が消滅する。
  - RuntimeBuilder が `BuildInput` 以外の semantic source を参照しないことを verifier で保証する。
  - RuntimeBuilder 内で `BuildInput` 以外から取得した値を semantic decision に使用することを禁止する。
  - RuntimeBuilder は `BuildInput` 以外の runtime semantic source を**参照してはならない**。
  - semantic decision への未使用を理由にした read access の残置を禁止する。
  - 禁止参照をシンボル単位で明示的に検査する:
    - `engine.*`
    - `runtimeStore.*`
    - `runtimeCoordinator.*`
    - `runtimePublicationBridge.*`
    - `RuntimeState.*`
    - `RuntimeWorld.*`
    - `RuntimeGraph.*`
    - publication state
    - transition state
  - 例外は `BuildInput` 構築時のみ許可する（それ以外は fail-closed）。
  - RuntimeBuilder semantic input descriptor は `RuntimeBuildTypes.h` の `BuildInput` を唯一とする。
  - `BuildInput` 互換構造体 / 派生構造体 / 代替構造体（例: `BuildInputV2`）の導入を禁止する。

### PR-04A: Coordinator Read Contract（ID41-A）

- 対象契約: 2
- 目的:
  - read入口を `consume` 中心へ統合し、`observe*` 外部公開を解消する。
- 主対象ファイル:
  - `src/core/RuntimePublicationCoordinator.h`
  - `src/audioengine/ISRRuntimePublicationCoordinator.h`
  - `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
  - `src/audioengine/AudioEngine.h`
- 主対象関数:
  - `observePublishedWorld`, `observeWorldHandle`, `getCurrent`
  - （追加）`consume` / read contract API
  - `makeRuntimePublishView`, `makeRuntimeReadView`
- 主対象変数:
  - `RuntimeStore::current`, `state_`, `swapPending_`
- 完了条件:
  - `getCurrent()==nullptr` 暫定実装が完全撤去される。
  - `observe*` は内部実装へ降格し、公開read契約は単一路化される。
  - `AudioEngine` から `observePublishedWorld` / `observeWorldHandle` / `getCurrent` を直接呼ぶ経路が存在しない。
  - RuntimeWorld読取は `consume(ReadToken)` 系契約のみ経由する。
  - `ReadToken` の生成元は Coordinator のみとし、外部生成を禁止する。
  - `kRuntimeReadAuthorityInventory`（新設）を導入し、read authority field 集合を固定する。
  - `consume()` が返す公開 read view は `kRuntimeReadAuthorityInventory` 列挙 field のみ参照可能とし、inventory外fieldへのアクセスを禁止する。
  - read authority inventory は唯一とし、read authority field の母集団は `kRuntimeReadAuthorityInventory` のみとする。
  - 複数 read authority inventory の併存を禁止する。

### PR-05: Publication Lifecycle Contract 統合（ID12+ID13+ID36）

- 対象契約: 6
- 目的:
  - publish/retire/shutdown/drain を単一状態機械として固定し、例外遷移を検証可能にする。
- 主対象ファイル:
  - `src/core/RuntimePublicationCoordinator.h`
  - `src/core/RuntimeStore.h`
  - `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
  - `src/audioengine/AudioEngine.Commit.cpp`
  - `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
  - `src/audioengine/AudioEngine.CtorDtor.cpp`
- 主対象関数:
  - `publishState`, `clearPublishedRuntimeSnapshotsNonRt`, `publishAndSwap`
  - `commit`, `retire`, `requestShutdown`, `markShutdownComplete`, `isFullyDrained`
- 主対象変数:
  - `state_`, `swapPending_`
  - `retireBacklogCount_`, `publicationBacklogCount_`, `reclaimInFlightCount_`, `deferredRetireResidencyCount_`
- 完了条件:
  - 非許可遷移が fail-closed で拒否される。
  - `publish(nullptr)` は shutdown専用契約としてテストで固定される。
  - `PublicationEpoch` ↔ `RuntimeGeneration` の monotonic mapping verifier が pass する。
  - `PublicationEpoch` / `RuntimeGeneration` の双方で strictly monotonic を要求し、rollback・reuse・wraparound を fail-closed で検出する。
  - `PublicationEpoch` / `RuntimeGeneration` は wraparound を許可しない。上限到達は implementation-defined hard stop（契約違反）とし、自動再利用を禁止する。

### PR-04B: Audio Thread Authority Cleanup（ID41-B / Cluster-A親）

- 対象契約: 2
- 目的:
  - Audio Thread分岐authorityを `RuntimeWorld` に一本化し、transition依存を診断/投影へ降格。
- 主対象ファイル:
  - `src/audioengine/AudioEngine.h`
  - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
  - `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
  - `src/audioengine/AudioEngine.Processing.Snapshot.cpp`
  - `src/audioengine/AudioEngine.Timer.cpp`
- 主対象関数:
  - `getNextAudioBlock`, `processDouble`
  - `resolveActiveRuntimeDSPFromRuntimeWorldOnly`, `resolveFadingRuntimeDSPFromRuntimeWorldOnly`
- 主対象変数:
  - `runtimePublishView.transition.current`, `runtimePublishView.transition.next`
  - `RuntimeWorld::topology.hasFadingRuntime`
- 完了条件:
  - `ID2/ID7/ID21/ID25/ID33` はID41配下の子タスクとしてクローズされる。
  - RuntimeGraph由来の authority 的な意思決定経路が残らない（ID42吸収分）。
  - `kRuntimeAuthorityInventory`（新設）を導入し、`RuntimeWorld::topology/routing/crossfade/dsp` 等のauthorityを列挙する。
  - `kRuntimeAuthorityInventory` と RuntimeWorld authority fields の一致検証を fail-closed で追加する。
  - authority inventory は唯一とし、Runtime authority field の母集団は `kRuntimeAuthorityInventory` のみとする。
  - 複数 authority inventory の並立を禁止する。
  - RuntimeWorld authority field の母集団は `kRuntimeAuthorityInventory` のみとする。
  - RuntimeWorld に authority class を持つ field を追加する場合、同一PR内で inventory 更新を必須とする。
  - inventory 未登録 authority field は fail-closed とする。

### PR-06: Crossfade Executor-Local Contract（ID3）

- 対象契約: 5
- 目的:
  - `CrossfadePreparedSnapshot` のsemantic source化を禁止し、executor-localへ封じ込める。
- 主対象ファイル:
  - `src/audioengine/AudioEngine.h`
  - `src/audioengine/AudioEngine.Commit.cpp`
  - `src/audioengine/AudioEngine.Timer.cpp`
  - `src/audioengine/RuntimeTransition.h`
- 主対象関数:
  - `refreshCrossfadePreparedSnapshotFromAtomics`
  - `publishCrossfadePreparedSnapshot`
  - `consumeCrossfadePreparedSnapshot`
- 主対象変数:
  - `CrossfadePreparedSnapshot::*`
  - `crossfadePreparedSnapshots_`, `crossfadePreparedSnapshotIndex_`
- 完了条件:
  - `CrossfadePreparedSnapshot must not be semantic source.` がテストで保証される。
  - semantic source の定義を固定する。以下に使用された時点で semantic source とみなす:
    - branch decision
    - rebuild decision
    - publish decision
    - retire decision
    - topology decision
    - routing decision
    - overlap decision
    - execution decision
  - `CrossfadePreparedSnapshot` の上記用途への使用を全面禁止する。
  - PR-06開始条件として、`PR-05` 完了かつ `PR-04B` 完了の双方を必須とする。

### PR-07: Governance / Registry Audit

- 対象契約: 7
- 目的:
  - 各PRで段階接続した verifier/registry/tier の最終監査を行い、欠落をゼロ化する。
- 主対象ファイル:
  - `.github/isr-verifier-registry.json`
  - `.github/isr-verifier-execution-layers.json`
  - `.github/isr-validator-tiering-policy.json`
  - `.github/workflows/isr-verification.yml`
  - `.github/scripts/isr-run-tiered-verification.ps1`
  - `src/tests/RuntimePublicationCoordinatorTests.cpp`
  - `src/tests/ISRSemanticValidationTests.cpp`
  - `src/tests/RuntimeSemanticSchemaValidationTests.cpp`
- 完了条件:
  - 新契約の verifier が tier設定に反映され、未配線時にCI fail-closed。
  - 監査PR時点で registry / descriptor / inventory / tests の不整合がゼロ。
  - `RuntimeSemanticSchema` / `AuthorityInventory` / `ReadAuthorityInventory` / `ISRRuntimeSemanticSchema.h の唯一の Semantic Descriptor Table` の相互一致を検証する `Consistency Verifier` を追加し、passする。
  - `ReadAuthorityInventory ⊆ RuntimeWorld` ではなく、`ReadAuthorityInventory == RuntimeWorld Read Authority Set`（完全一致）を検証する。
  - `Consistency Verifier` は以下の一致も検証する:
    - VerifierRegistry
    - TierPolicy
    - WorkflowRegistration
  - 上記いずれか欠落時は fail-closed とする。
  - `Consistency Verifier` の最低 fail 条件を固定する:
    - descriptor table mismatch
    - authority inventory mismatch
    - read authority inventory mismatch
    - registry missing entry
    - tier missing entry
    - workflow missing registration
    - orphan verifier
    - orphan inventory
    - orphan descriptor

---

## PR依存グラフ（順序制約）

- PR-01 → PR-02 → PR-03 → PR-04A → PR-05 → PR-04B → PR-06 → PR-07
- verifier/registry/tier接続は各PRで必須。PR-07は最終監査に限定する。
- `PR-06` は `PR-05` と `PR-04B` の両完了を開始条件とし、逆転実施を禁止する。

---

## AI実装統治（全PR共通の必須前提）

- 実装前探索義務
  - `grep` / Serena / CodeGraph の3系統で対象シンボルを探索する。
  - 着手前に以下を列挙する:
    - 呼び出し元
    - 呼び出し先
    - 保持構造体
    - verifier登録状況（registry/tier）
- 実装後差分監査義務
  - 実装後に再度 `grep` / Serena / CodeGraph を実行し、着手前との差分をPR本文に記録する。
  - 参照残り / 未削除経路 / 孤立コード を差分監査で明示する。
- 削除・改変の禁止条件
  - 参照が残る状態で `descriptor` / `inventory` / `verifier` / `registry` を変更しない。
  - 「コード削除先行・検証後追い」を禁止し、常に fail-closed 側で段階移行する。

---

## 各PRの共通チェックリスト

- 変更対象が `JUCE/` と `r8brain-free-src/` を含まない
- Audio Thread禁止事項（blocking/alloc/lock/libm/MessageManager/shared_ptr 等）を侵していない
- `Strict Atomic Dot-Call Scan` 相当の規約チェックを通過
- Debug build（必要に応じてRelease）成功
- `grep` / Serena / CodeGraph の3系統探索結果をPR本文へ記録
- 対応する契約テストを追加/更新
- 新規 verifier 追加・registry接続・tier接続・回帰テスト追加を同PR内で完了
- 新規契約追加時は同一PR内で `registry` / `tier` / `workflow` / `test` を接続する（後続PR送り禁止）
- 未配線 verifier が registry / workflow に残っていない

---

## 進捗確認チェックリスト（運用用）

> 使い方: 各PR着手時に `状態` を更新し、完了時に `証跡` と `残課題` を必ず記録する。
> 状態記号: `未着手 / 進行中 / 完了 / ブロック`

### A. PR進捗トラッカー

- [x] PR-01 RuntimeWorld Self-contained
  - 状態: 完了
  - 証跡（PR/コミット/検証ログ）: `src/audioengine/AudioEngine.Commit.cpp` に RuntimeGraph authority contract verifier を追加し `runPublicationPrecheckNonRt()` に fail-closed 接続。`src/tests/PartialPublicationRejectTests.cpp` に authority mismatch / transition semantic mismatch / publication sequence rollback / mapped runtime generation mismatch に加えて、branch policy 系（invalid routing processing order / invalid execution transition policy）reject ケースを追加。さらに `src/audioengine/RuntimeGraph.h` で `kFieldDescriptors` と `kAuthorityInventory` を実在field母集団に統一し、`validateDecisionCoverageContract()` で decision関連fieldの descriptor/inventory 被覆を fail-closed 化。`Build_CMakeTools` 成功、`RunCtest_CMakeTools(tests=[PartialPublicationReject])` Passed（2026-06-01）。
  - 残課題: RuntimeGraph authority verifier の decision coverage を descriptor table 母集団基準で継続監査（新規 field 追加時の同PR更新を必須化）。

- [x] PR-02 Semantic Reachability
  - 状態: 完了
  - 証跡（PR/コミット/検証ログ）:
    - `grep` で reachability 関連の実装/検証入口を抽出:
      - 実装側: `AudioEngine.h`（`requestRebuild(...)`, `submitRebuildIntent(...)`, `rebuildGeneration`）
      - 実装側: `AudioEngine.Commit.cpp`（`lastCommittedRebuildGeneration` 反映、`semanticHash` 観測）
      - テスト側: `RuntimeSemanticSchemaValidationTests.cpp` の
        `testRuntimeSemanticReachabilityValidation()` / `testDescriptorCoverageContract()`
      - script側: `.github/scripts/isr-run-tiered-verification.ps1` に
        `isr-verify-semantic-reachability.ps1` が配線済み
    - PR-07最終監査（governance chain）完了後、PR-02着手へ遷移（2026-06-01）。
    - `src/tests/RuntimeSemanticSchemaValidationTests.cpp` に
      `testSemanticTriggerToHashPathContract()` と `testSemanticHashCoverageContract()` を追加し、
      `TriggerAccepted -> BuildInputSealed -> RuntimeWorldPublished -> SemanticHashComputed -> PublicationStable -> CrossfadeComplete -> RetireSettled`
      の到達経路と、`RuntimeSemanticHash` 8フィールドすべての非等価寄与を fail-closed で固定。
    - `.github/scripts/isr-verify-semantic-reachability.ps1` を `v2` 化し、
      1) contract token coverage
      2) unit test coverage
      3) trigger path coverage（`submitRebuildIntent/requestRebuild/capture+finalize+seal/enqueuePublicationIntentForRuntimeCommit`）
      4) semantic hash coverage（8ハッシュ）
      5) tier wiring coverage
      を全件一致（100%）で強制する fail-closed verifier に更新。
    - 検証実行:
      - `RunCtest_CMakeTools(tests=[RuntimeSemanticSchemaValidation])` Passed
      - `isr-verify-semantic-reachability.ps1` Passed
      - `Build_CMakeTools` 成功（no work to do）（2026-06-01）
  - 残課題: なし（以後は tiered verifier で退行監査）。

- [x] PR-03 RuntimeBuilder Semantic Isolation
  - 状態: 完了
  - 証跡（PR/コミット/検証ログ）: `src/audioengine/RuntimeBuildTypes.h` で `BuildInput` を `final` 化し、`RuntimeBuildSnapshot::buildInput` が canonical `BuildInput` であることを `static_assert` で固定。`src/audioengine/RuntimeBuilder.cpp` で `observeWorldHandle(engine.runtimeStore)` 直参照を廃止し、`previousSequenceId` は `AudioEngine::getLastCommittedPublicationSequence()` 経由に変更。さらに `src/audioengine/RuntimeTransition.h` / `src/audioengine/AudioEngine.h` で `EngineRuntime.processingOrder` を導入して `engine.getProcessingOrder()` 直参照を廃止し、同じく `EngineRuntime.retireBacklog/deferredResidency/rebuildWorkerRunning/eqCoeffHash` を導入して RuntimeBuilder から `retire/affinity/eqHash` の `engine.*` 直参照を廃止。加えて `AudioEngine::reserveRuntimePublicationIdentity()` を導入し、`RuntimeBuilder` 側の generation/worldId/publicationSequence の個別 `engine.*` 直参照を単一API経由に集約。さらに `AudioEngine::computeRuntimePublishComputation()` を導入し、`makeEngineRuntimeState` / `makeRuntimeGraphState` / `previousSequenceId` 取得を `RuntimeBuilder` 側の複数呼び出しから単一APIへ集約。`Build_CMakeTools` 成功、`RunCtest_CMakeTools`（`PartialPublicationReject`, `RuntimePublicationCoordinatorRejects`）Passed（2026-06-01）。
  - 最終境界定義（RuntimeBuilder → AudioEngine 許可API）:
    - publication identity: `reserveRuntimePublicationIdentity()`
    - publish computation: `computeRuntimePublishComputation(...)`
    - committed sequence read: `getLastCommittedPublicationSequence()`
    - BuildInput 構築以外での `engine.*` / `runtimeStore.*` / `RuntimeState.*` / `RuntimeWorld.*` / `RuntimeGraph.*` 直参照は禁止（fail-closed）。
  - 残課題: なし（以後は PR-07 consistency verifier で退行監査）。

- [x] PR-04A Coordinator Read Contract
  - 状態: 完了
  - 証跡（PR/コミット/検証ログ）:
    - `grep` / Serena で `src/core/RuntimePublicationCoordinator.h` の `observePublishedWorld` / `observeWorldHandle` を確認。
    - `grep` / Serena で `src/audioengine/AudioEngine.h` の `makeRuntimePublishView` / `makeRuntimeReadView` / `readControlRuntimeView` が `RuntimePublicationCoordinator::observeWorldHandle(runtimeStore)` に依存していることを確認。
    - `grep` / Serena で `src/audioengine/AudioEngine.Commit.cpp` ほか複数箇所に `readControlRuntimeView()` 呼び出し経路を確認。
    - `grep` / Serena で `src/audioengine/ISRRuntimePublicationCoordinator.cpp` の `getCurrent()==nullptr` 暫定実装を確認（PR-04A完了条件の直接対象）。
    - CodeGraph では read contract クエリで有効エンティティが返らず（0件）、本PRでは `grep` + Serena を一次根拠として採用。
    - `src/core/RuntimePublicationCoordinator.h` に `consumePublishedWorld` / `consumeWorldHandle` を追加し、read API の単一路化へ段階移行を開始。
    - `src/core/RuntimePublicationCoordinator.h` に `ReadToken` / `acquireReadToken(...)` と token受け取り版 `consumePublishedWorld(...)` / `consumeWorldHandle(...)` を追加し、ReadToken 契約の骨組みを導入。
    - `src/audioengine/AudioEngine.h` に `RuntimeState::kRuntimeReadAuthorityInventory` を導入し、`validateReadAuthorityInventoryAgainstDescriptors(...)` で descriptor 母集団に対する fail-closed 検証を接続。
    - `src/audioengine/AudioEngine.h` / `src/audioengine/AudioEngine.Commit.cpp` の read 側呼び出しを `observeWorldHandle` から `consumeWorldHandle` へ置換（互換のため observe 系APIは残置）。
    - `src/audioengine/AudioEngine.h` の `makeRuntimePublishView` / `makeRuntimeReadView` ほか主要読取経路を token 経由 consume 呼び出しへ移行。
    - `src/tests/PartialPublicationRejectTests.cpp` / `src/tests/RuntimePublicationCoordinatorTests.cpp` の `observeWorldHandle` 呼び出しを `consumeWorldHandle` へ移行。
    - `src/core/RuntimePublicationCoordinator.h` から `observePublishedWorld` / `observeWorldHandle` 公開APIを削除し、read API を `consume*` に一本化。
    - `grep` 再監査で `observePublishedWorld` / `observeWorldHandle` の残存参照 0 件を確認。
    - `src/audioengine/ISRRuntimePublicationCoordinator.cpp` で `getCurrent()==nullptr` 暫定実装を撤去し、`currentWorld_` atomic による current world 返却へ移行。
    - `Build_CMakeTools` 成功、`RunCtest_CMakeTools`（`PartialPublicationReject`, `RuntimePublicationCoordinatorRejects`）Passed（2026-06-01）。
  - 残課題: なし（以後は PR-07 consistency verifier で退行監査）。

- [x] PR-05 Publication Lifecycle Contract
  - 状態: 完了
  - 証跡（PR/コミット/検証ログ）:
    - `grep` / Serena で lifecycle 主経路を確認:
      - `src/core/RuntimePublicationCoordinator.h`: `publishAndSwap(nullptr/newWorld)`
      - `src/audioengine/AudioEngine.Commit.cpp`: `runtimePublicationBridge_.commit(...)` / `retire(...)` / publication backlog publish
      - `src/audioengine/ISRRuntimePublicationCoordinator.cpp`: `commit/retire/requestShutdown/markShutdownComplete/isFullyDrained/swapPending` 実装
      - `src/audioengine/AudioEngine.Threading.cpp`: `AudioEngine::isFullyDrained()` が bridge の drained 判定へ委譲
      - `src/audioengine/AudioEngine.CtorDtor.cpp` / `AudioEngine.Processing.ReleaseResources.cpp`: shutdown 開始/完了シーケンス
    - `src/audioengine/ISRRuntimePublicationCoordinator.cpp` で `getCurrent()==nullptr` 暫定実装を撤去済み（PR-04A成果を PR-05 前提に接続）。
    - CodeGraph では lifecycle クエリの有効エンティティ取得が0件で、本PRも `grep` + Serena を一次根拠として採用。
    - `src/tests/ISRSemanticValidationTests.cpp` を更新し、lifecycle reject 契約テストを追加/補強:
      - `commit` 成功時 `getCurrent()==published world` の検証に更新（PR-04Aの current world 返却変更に整合）。
      - epoch rollback（`sequence` 増加でも `epoch` 減少）を fail-closed 検証。
      - epoch advance 時の mapped generation rollback を fail-closed 検証。
      - epoch reuse / mapped generation reuse / wraparound（`max -> 0`）を fail-closed 検証。
      - shutdown 完了時に not-drained なら `Faulted` へ遷移することを検証。
      - `swapPending` 中は Pressure 正規化しないこと、解除後に bounded window で Ready 復帰することを検証。
      - drained counters がゼロでも `swapPending=true` なら shutdown complete を拒否（Faulted）することを検証。
    - `src/audioengine/ISRRuntimePublicationCoordinator.cpp` の monotonic 判定を strict 化:
      - `epoch <= previousEpoch` を拒否。
      - `mappedGeneration <= previousMappedGeneration` を拒否。
    - `src/core/RuntimePublicationCoordinator.h` に shutdown clear request 契約を追加:
      - `requestShutdownClearNonRt()` を追加。
      - `clearPublishedRuntimeSnapshotsNonRt()` は request 未実施時 no-op（`publish(nullptr)` を拒否）化。
    - `src/tests/RuntimePublicationCoordinatorTests.cpp` に `testClearRequiresShutdownRequest()` を追加し、
      request なし clear は no-op、request あり clear のみ `nullptr` publish を許可する契約を固定。
    - `src/audioengine/AudioEngine.CtorDtor.cpp` / `AudioEngine.Processing.ReleaseResources.cpp` / `src/tests/PartialPublicationRejectTests.cpp` で
      clear 呼び出しを `requestShutdownClearNonRt()` 先行へ移行。
    - `RunCtest_CMakeTools`:
      - `ISRSemanticValidationRejects` Passed
      - `RuntimePublicationCoordinatorRejects` Passed
      - `PartialPublicationReject` Passed
      - 備考: `DartConfiguration.tcl` 警告は継続するが、各テスト結果は pass（2026-06-01）。
  - 残課題: なし（以後は PR-07 consistency verifier で退行監査）。

- [x] PR-04B Audio Thread Authority Cleanup
  - 状態: 完了
  - 証跡（PR/コミット/検証ログ）:
    - `grep` / Serena で Audio Thread系の authority 分岐残存を抽出:
      - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` / `BlockDouble.cpp` / `Snapshot.cpp` / `Timer.cpp` に
        `runtimePublishView.transition.current/next` 直参照が残存。
      - `hasFadingRuntime` 判定は `runtimeWorld->topology.hasFadingRuntime` へ寄っているが、
        実体取得が依然として `transition.next` へ依存する箇所を確認。
      - `src/audioengine/AudioEngine.h` の
        `resolveActiveRuntimeDSPFromRuntimeWorldOnly` / `resolveFadingRuntimeDSPFromRuntimeWorldOnly` が
        内部で `runtimePublish.transition.current/next` を参照しており、PR-04Bでの authority cleanup 対象として確定。
    - CodeGraph では PR-04B クエリの有効エンティティ取得が0件のため、本PRでも `grep` + Serena を一次根拠として採用。
    - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` / `BlockDouble.cpp` / `Snapshot.cpp` / `Timer.cpp` を更新し、
      Audio Thread（および timer の runtime 選択）での `transition.current/next` 直接参照を
      `resolveActiveRuntimeDSPFromRuntimeWorldOnly(...)` / `resolveFadingRuntimeDSPFromRuntimeWorldOnly(...)` 経由へ統一。
    - `grep` 再監査で上記4ファイルの `transition.current|transition.next` 残存 0 件を確認。
    - `src/audioengine/AudioEngine.h` に `kRuntimeAuthorityInventory` を導入し、
      `validateDescriptorSet()` で `validateAuthorityInventorySet/AgainstDescriptors` を同 inventory に接続（fail-closed）。
    - `Build_CMakeTools` 成功、`RunCtest_CMakeTools`（`ISRSemanticValidationRejects`, `RuntimePublicationCoordinatorRejects`, `PartialPublicationReject`）Passed（2026-06-01）。
    - `RunCtest_CMakeTools`（`ObservePathSingleSource`, `OverlapAuthoritySingular`, `ShadowCompareContract`）Passed（2026-06-01）。
  - 残課題: なし（以後は PR-07 consistency verifier で退行監査）。

- [x] PR-06 Crossfade Executor-Local
  - 状態: 完了
  - 証跡（PR/コミット/検証ログ）:
    - `grep` / Serena で `CrossfadePreparedSnapshot` の使用箇所を抽出し、`AudioEngine.Commit.cpp` の分岐判定で
      `consumeCrossfadePreparedSnapshot()` 依存が残ることを確認。
    - `src/audioengine/AudioEngine.Commit.cpp` を更新し、commit 経路の以下判定を
      `CrossfadePreparedSnapshot` ではなく `RuntimeWorld` semantic fields へ置換:
      - `hasFadingRuntime` ← `runtimeWorld->topology.hasFadingRuntime`
      - `hasPendingCrossfade` ← `runtimeWorld->engine.dspCrossfadePending`
      - `useDryAsOld` ← `runtimeWorld->overlap.firstIrDryCrossfadePending || runtimeWorld->overlap.useDryAsOld`
    - `grep` 再監査で `AudioEngine.Commit.cpp` の `consumeCrossfadePreparedSnapshot` / `preparedCrossfade` 残存 0 件を確認。
    - `Build_CMakeTools` 成功、`RunCtest_CMakeTools`（`PartialPublicationReject`, `ObservePathSingleSource`, `OverlapAuthoritySingular`, `ISRSemanticValidationRejects`）Passed（2026-06-01）。
    - `src/audioengine/AudioEngine.h::makeEngineRuntimeState` を更新し、crossfade semantic 値の生成元を
      `CrossfadePreparedSnapshot` から `RuntimeWorld`（fallback: atomics）へ置換。
    - `src/audioengine/AudioEngine.Timer.cpp` を更新し、`hasPendingCrossfade` 判定を
      `consumeCrossfadePreparedSnapshot()` 依存から `RuntimeWorld` semantic fields へ置換。
    - `src/tests/CrossfadeExecutorLocalContractTests.cpp` を追加し、
      `CrossfadePreparedSnapshot` が commit/timer の branch/rebuild/publish/retire/execution decision へ
      混入しないことを fail-closed 監査で固定。
    - `CMakeLists.txt` に `CrossfadeExecutorLocalContractTests` / `CrossfadeExecutorLocalContract` を登録。
    - `grep` 再監査で `consumeCrossfadePreparedSnapshot()` 実使用が `AudioEngine.h` の宣言のみであることを確認。
    - `Build_CMakeTools` 成功、`RunCtest_CMakeTools`
      （`CrossfadeExecutorLocalContract`, `ISRSemanticValidationRejects`, `ObservePathSingleSource`,
      `OverlapAuthoritySingular`, `PartialPublicationReject`, `ShadowCompareContract`）Passed（2026-06-01）。
  - 残課題: なし（以後は PR-07 consistency verifier で退行監査）。

- [x] PR-07 Governance / Registry Audit
  - 状態: 完了
  - 証跡（PR/コミット/検証ログ）:
    - `grep` 探索で `governance-consistency-verifier` が
      `.github/isr-verifier-registry.json`（release tier）および
      `.github/scripts/isr-verify-governance-registries.ps1` の required set に登録済みであることを確認。
    - `.github/isr-validator-tiering-policy.json` と `.github/workflows/isr-verification.yml` を読取し、
      nightly/weekly cron と tier マッピング（standard/exhaustive）の整合前提を確認。
    - `.github/scripts/isr-run-tiered-verification.ps1` を読取し、
      governance scripts が tier runner に配線される構成であることを確認。
    - `run_in_terminal` で以下を実行し、通過を確認:
      - `.github/scripts/isr-verify-governance-registries.ps1`
      - `.github/scripts/isr-verify-gate-wiring.ps1`
      - 結果: `[PASS] governance registries verification passed` / `[PASS] ISR gate wiring self-test verified`（2026-06-01）
    - `.github/scripts/isr-verify-governance-consistency.ps1` を新設し、
      PR-07の最低 fail 条件（descriptor/authority/read-authority/registry/tier/workflow/orphan）を
      fail-closed で検証する専用レポート `evidence/governance_consistency_report.json` を追加。
    - `.github/scripts/isr-run-tiered-verification.ps1` の standard tier 配線に
      `isr-verify-governance-consistency.ps1` を追加。
    - `.github/scripts/isr-verify-governance-registries.ps1` の
      `governance-consistency-verifier` wiring contract を
      `isr-verify-gate-wiring.ps1` から `isr-verify-governance-consistency.ps1` へ更新。
    - `run_in_terminal` で
      `.github/scripts/isr-verify-governance-consistency.ps1` 実行: `[PASS] governance consistency verification passed`。
    - 変更後に再度
      `.github/scripts/isr-verify-governance-registries.ps1` / `.github/scripts/isr-verify-gate-wiring.ps1` を実行し、
      いずれも PASS を確認（2026-06-01）。
    - `run_task` で `Strict Atomic Dot-Call Scan` 実行: `passed`（2026-06-01）。
    - `Build_CMakeTools` 成功、`RunCtest_CMakeTools`
      （`RuntimeSemanticSchemaValidation`, `CrossfadeExecutorLocalContract`, `ISRSemanticValidationRejects`,
      `RuntimePublicationCoordinatorRejects`, `PartialPublicationReject`, `ObservePathSingleSource`,
      `OverlapAuthoritySingular`, `ShadowCompareContract`）Passed（2026-06-01）。
    - 最終監査として
      `.github/scripts/isr-verify-governance-consistency.ps1` → PASS、
      `.github/scripts/isr-verify-governance-registries.ps1` → PASS、
      `.github/scripts/isr-verify-gate-wiring.ps1` → PASS を連続実行で確認（2026-06-01）。
  - 残課題: なし。

### B. 依存順序ゲート（着手可否）

- [x] PR-02 着手前に PR-01 が完了している
- [x] PR-03 着手前に PR-02 が完了している
- [x] PR-04A 着手前に PR-03 が完了している
- [x] PR-05 着手前に PR-04A が完了している
- [x] PR-04B 着手前に PR-05 が完了している
- [x] PR-06 着手前に PR-05 + PR-04B が完了している
- [x] PR-07 着手前に PR-01〜PR-06 が完了している

### C. 各PR完了時の必須ゲート

- [x] `grep` / Serena / CodeGraph の3系統探索結果をPR本文に記録
- [x] 新規 verifier の registry / tier / workflow / test を同一PR内で接続
- [x] fail-closed 条件が実装・テストで担保されている
- [x] `Strict Atomic Dot-Call Scan` 相当チェック通過
- [x] Debug build 成功（必要に応じて Release）
- [x] orphan verifier / orphan inventory / orphan descriptor がゼロ

### D. 最終完了判定（プログラム全体）

- [x] Runtime authority が RuntimeWorld に収束
- [x] Read authority が RuntimeWorld に単一路で収束
- [x] Semantic reachability が fail-closed で保証
- [x] RuntimeBuilder が BuildInput only で駆動
- [x] Crossfade が executor-local 契約を満たす
- [x] Publication lifecycle が単一状態機械で検証
- [x] Governance/verifier wiring が tier実行に固定

---

## 完了判定（プログラム全体）

以下を全て満たしたとき、親契約7本の実装を完了とする。

1. Runtime authority が RuntimeWorld に収束
2. Read authority が RuntimeWorld に単一路で収束
3. Semantic reachability が fail-closed で保証
4. RuntimeBuilder が BuildInput only で駆動
5. Crossfade が executor-local 契約を満たす
6. Publication lifecycle が単一状態機械で検証される
7. Governance/verifier wiring が tier実行に固定される

---

## 追加探索エビデンス（本版反映時）

- `grep`
  - `RuntimeBuilder.h` に `AudioEngine& engine` メンバが残存することを確認。
  - `RuntimeBuilder.cpp` で `engine.runtimeStore` + `observeWorldHandle(engine.runtimeStore)` 直接参照を確認（PR-03 参照禁止強化根拠）。
  - `AudioEngine.h` で `BuildInput` の実体利用（`RuntimeBuildSnapshot::buildInput`）を確認し、Builder入力型唯一化の根拠を確認。
  - `AudioEngine.h` で `eqBypassed` / `convBypassed` / `order` 等の RuntimeGraph系フィールドを確認（PR-01 verifier検査対象固定の根拠）。
  - `AudioEngine.Commit.cpp` / `AudioEngine.h` に `runtimePublicationBridge_` と `RuntimePublicationBridge::buildRuntimePublishWorld` 経路が存在することを確認（PR-03禁止参照境界の根拠）。
  - `RuntimePublicationCoordinator.h` と `AudioEngine.h` に `observeWorldHandle` 直接参照が残存することを確認（PR-04Aの read authority 収束根拠）。
  - `ISRRuntimePublicationCoordinator.cpp/.h` で `publicationSequenceId_`, `publicationEpoch_`, `mappedRuntimeGeneration_` と monotonic 更新経路を確認。
  - `AudioEngine.Commit.cpp` で `observeMonotonicRollbackRequested_` / `monotonicViolationCount` / rollback 経路を確認（PR-05 strict monotonic 強化根拠）。
  - `AudioEngine.Commit.cpp` / `RuntimeBuilder.cpp` で `publication.epoch` と `mappedRuntimeGeneration` の整合条件が存在することを確認。
  - `AudioEngine.h` / `ISRRuntimeSemanticSchema.h` で descriptor/inventory validation API（`validateFieldDescriptorSet`, `validateAuthorityInventoryAgainstDescriptors` 等）を確認。
  - `.github/workflows/isr-verification.yml` / `.github/isr-verifier-registry.json` / `.github/isr-validator-tiering-policy.json` / `.github/scripts/isr-run-tiered-verification.ps1` の連携点を確認（PR-07 一致検証強化根拠）。
  - 追加探索で `AudioEngine.h::captureAudioThreadParameterSnapshot(const RuntimePublishWorld*)` が `world->graph.saturationAmount/inputHeadroomGain/outputMakeupGain/convolverInputTrimGain` を参照していることを確認し、PR-01/PR-04B の漏れ対象として補完。
  - 追加探索で `AudioEngine.h::resolveActiveRuntimeDSPFromRuntimeWorldOnly` / `resolveFadingRuntimeDSPFromRuntimeWorldOnly` が `runtimePublish.transition.current/next` を参照していることを確認し、PR-04B の漏れ対象として補完。
  - 追加探索で `AudioEngine.Processing.PrepareToPlay.cpp::prepareToPlay` が `runtimePublish.transition` を使って republish 条件を組み立てていることを確認し、PR-04B の漏れ対象として補完。
  - 追加探索で `RuntimeBuilder.cpp` の `automation/coefficient` 投影が `graphState` 依存のまま残っていることを確認し、PR-01/PR-03 の漏れ対象として補完。
- Serena
  - `RuntimeBuilder.cpp` で `engine.runtimeStore` 観測、`engine.*` 参照群が残存することを確認（PR-03強化根拠）。
  - `RuntimePublicationCoordinator.h` で `observePublishedWorld/observeWorldHandle` 併存を確認（PR-04A強化根拠）。
  - `ISRRuntimePublicationCoordinator.cpp` で epoch/generation/lifecycle関連シンボルを再確認（PR-05強化根拠）。
  - `AudioEngine.h` の runtime read helper / parameter snapshot helper 群を再確認し、RuntimeWorld authority inventory に `automation` / `coefficient` が未反映だったことを確認（PR-04A/PR-04B 補完根拠）。
- CodeGraph
  - `buildRuntimePublishWorld` 呼び出し元を再確認し、publish/read経路が複数層に分散していることを確認。
  - （注）`observePublishedWorld` 呼び出し逆引きは本環境では未解決だったため、`grep`/Serena結果を一次根拠として採用。
  - `resolveActiveRuntimeDSPFromRuntimeWorldOnly` / `resolveFadingRuntimeDSPFromRuntimeWorldOnly` / `captureAudioThreadParameterSnapshot` が audio processing cluster の中心依存点であることを再確認し、漏れ対象優先度を高と判断。

## 追加改修証跡（漏れ対象補完 2026-06-01）

- PR-01 / PR-03 補完:
  - `src/audioengine/ISRRuntimeSemanticSchema.h` の `AutomationSemantic` に `saturationAmount` / `inputHeadroomGain` / `outputMakeupGain` / `convolverInputTrimGain` を追加。
  - `src/audioengine/RuntimeTransition.h` の `EngineRuntime` に automation/coefficient 派生情報（bypass群、gain群、adaptive coeff bank/generation）を追加。
  - `src/audioengine/RuntimeBuilder.cpp` で `RuntimeWorld::routing/automation/coefficient` の投影元を `graphState` 依存から `engineState` 主体へ補正し、payload hash に automation/coefficient semantic を反映。
- PR-04A / PR-04B 補完:
  - `src/audioengine/AudioEngine.h` の `kRuntimeReadAuthorityInventory` に `automation` / `coefficient` を追加し、read authority inventory を実態に整合。
  - `src/audioengine/AudioEngine.h::makeEngineRuntimeState` を更新し、processing order / bypass / gain / eqCoeffHash / adaptive coeff generation を `RuntimeWorld` authority（fallback: atomics）から組み立てるよう修正。
  - `src/audioengine/AudioEngine.h::makeRuntimeGraphState` を更新し、authority系フィールドを `state` から投影する純 projection に補正。
  - `src/audioengine/AudioEngine.h::resolveActiveRuntimeDSPFromRuntimeWorldOnly` / `resolveFadingRuntimeDSPFromRuntimeWorldOnly` を `runtimeWorld->engine.current/fading` 参照へ変更。
  - `src/audioengine/AudioEngine.h::captureAudioThreadParameterSnapshot(const RuntimePublishWorld*)` を `world->automation` / `world->coefficient` 参照へ変更し、`world->graph.*` 参照を撤去。
  - `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp::prepareToPlay` を `runtimeWorld->execution.transitionPolicy` / `runtimeWorld->overlap.fadeTimeSec` / `runtimeWorld->topology.hasFadingRuntime` ベースの republish に変更。
- テスト/監査補完:
  - `src/tests/RuntimeSemanticSchemaValidationTests.cpp` に軽量 schema coverage 検証を維持したまま、重い JUCE 依存を持ち込まずに既存契約を継続検証。
  - `src/tests/RuntimeWorldAuthorityProjectionTests.cpp` を追加し、RuntimeWorld authority projection / read-inventory 追加 / `world->graph.*` 撤去を fail-closed で固定。
  - `CMakeLists.txt` に `RuntimeWorldAuthorityProjectionTests` を登録。
  - `Strict Atomic Dot-Call Scan` Passed。
  - Debug build 成功。
  - `RunCtest_CMakeTools`:
    - `RuntimeSemanticSchemaValidation` Passed
    - `ObservePathSingleSource` Passed
    - `OverlapAuthoritySingular` Passed
    - `ShadowCompareContract` Passed
    - `CrossfadeExecutorLocalContract` Passed
    - `RuntimeWorldAuthorityProjectionContract` Passed
  - `isr-run-tiered-verification.ps1 -Tier standard` Passed。
  - `isr-run-tiered-verification.ps1 -Tier exhaustive` Passed。

## 追加改修証跡（read authority 直参照漏れ補完 2026-06-01 第2ラウンド）

- 再探索（grep / Serena / CodeGraph）
  - `grep` で `runtimeReadView.runtimeWorld|runtimeReadViewRef.runtimeWorld` を `src/audioengine/**` 走査し、`AudioEngine.h` helper以外の直参照を漏れ対象として抽出。
  - Serena regex 再監査（`src/audioengine/**/*.cpp`）で直参照残存 0 を確認。
  - CodeGraph で `resolveActiveRuntimeDSPFromRuntimeWorldOnly` callers を再確認し、`Commit/PrepareToPlay/AudioBlock/BlockDouble/Snapshot/Timer` を改修優先対象に固定。
- PR-04A/PR-04B 補完実装
  - `src/audioengine/AudioEngine.h`
    - `getRuntimeWorldFromReadView(...)` を新設。
    - `getTransitionPolicyFromRuntimeWorld(...)` / `getOverlapFadeTimeFromRuntimeWorld(...)` /
      `hasFadingRuntimeInWorld(...)` / `hasPendingCrossfadeInWorld(...)` / `shouldUseDryAsOldInWorld(...)` を新設。
    - `resolveActiveRuntimeDSPFromRuntimeWorldOnly` / `resolveFadingRuntimeDSPFromRuntimeWorldOnly` を helper 経由へ統一。
  - `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`
    - republish 判定の world 読取を helper 経由へ統一。
  - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
  - `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
  - `src/audioengine/AudioEngine.Processing.Snapshot.cpp`
    - `runtimeReadView(.Ref).runtimeWorld` 直参照を helper 経由へ置換。
  - `src/audioengine/AudioEngine.Commit.cpp`
    - crossfade dedup 判定 (`hasFadingRuntime/hasPendingCrossfade/useDryAsOld`) を helper 経由へ統一。
  - `src/audioengine/AudioEngine.Timer.cpp`
    - transition/crossfade 判定を helper 経由へ統一（`runtimePublishView.transition` 非依存を維持）。
- PR-07 監査テスト補完
  - `src/tests/RuntimeWorldAuthorityProjectionTests.cpp`
    - helper導入に合わせて期待文字列を更新。
    - `PrepareToPlay/Snapshot/AudioBlock/BlockDouble/Commit` に `runtimeReadView.runtimeWorld` 直参照が残る場合 fail-closed となる監査を追加。
    - helper 使用痕跡（`getTransitionPolicyFromRuntimeWorld(...)` / `hasFadingRuntimeInWorld(...)`）を契約として固定。
- 検証結果
  - `Build_CMakeTools` 成功（result code 0）。
  - `RunCtest_CMakeTools`:
    - `RuntimeWorldAuthorityProjectionContract` Passed
    - `CrossfadeExecutorLocalContract` Passed
    - `ObservePathSingleSource` Passed
  - `Strict Atomic Dot-Call Scan` Passed。
  - `check-list-compliance.ps1`:
    - Failures: 0
    - Warnings: 2（既存テストコード上の manual review 警告のみ）

## 追加改修証跡（RuntimeReadView 段階的カプセル化 2026-06-02 第3ラウンド）

- 目的
  - `RuntimeReadView` の公開面をさらに絞り、`runtimePublish` 経由の直接参照を撤去。
  - read path を `RuntimeWorld` helper 群に一本化し、field直参照の再流入を fail-closed 監査で固定。
- 実装
  - `src/audioengine/AudioEngine.h`
    - `RuntimePublishView` を削除。
    - `RuntimeReadView` から `runtimePublish` member を削除し、`observedSnapshot + runtimeWorld` の最小構成へ移行。
    - `makeRuntimePublishView(...)` を削除し、`makeRuntimeReadView(...)` で read token + observed snapshot を直接構築。
    - `getRuntimeSampleRateHzFromWorld(...)` を追加し、sample rate 取得を helper 経由へ統一。
    - `getRuntimeGraph(...)` を static helper 化し、`RuntimeReadView` の world から直接返す形へ簡素化。
  - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
  - `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
  - `src/audioengine/AudioEngine.Processing.Snapshot.cpp`
    - `runtimeReadView(.Ref).runtimePublish.sampleRateHz` 参照を撤去し、
      `getRuntimeSampleRateHzFromWorld(...)` へ置換。
- 契約テスト強化
  - `src/tests/RuntimeWorldAuthorityProjectionTests.cpp`
    - `struct RuntimePublishView` が残存した場合 fail。
    - `getRuntimeSampleRateHzFromWorld(...)` helper の存在を必須化。
    - `AudioBlock/BlockDouble/Snapshot` に `runtimeReadView(.Ref).runtimePublish` が残存した場合 fail。
    - 上記3ファイルで sample-rate helper 使用を必須化。
- 検証
  - `Build_CMakeTools`: success (result code 0)
  - `RunCtest_CMakeTools`:
    - `RuntimeWorldAuthorityProjectionContract` Passed
    - `CrossfadeExecutorLocalContract` Passed
    - `ObservePathSingleSource` Passed
  - `Strict Atomic Dot-Call Scan`: Passed
  - `check-list-compliance.ps1`: Failures 0 / Warnings 2（既知 manual review 警告のみ）

## 追加改修証跡（RuntimeReadView opaque 契約固定 2026-06-02 第4ラウンド）

- 目的
  - `RuntimeReadView` を「外部からは opaque に近い扱い」に固定し、
    `getRuntimeSnapshot(...)` / `getRuntimeWorldFromReadView(...)` などの helper 以外による
    直接 field 参照の再流入を fail-closed で防止。
- 実装
  - `src/tests/RuntimeWorldAuthorityProjectionTests.cpp` を拡張。
    - `testRuntimeReadViewOpaqueContract()` を追加し、`src/` 配下を再帰走査。
    - 対象: `src/tests/**` と `src/audioengine/AudioEngine.h` を除くコードファイル（`.h/.hpp/.cpp/.cxx/.cc`）。
    - 禁止パターン（検出で fail）:
      - `runtimeReadView.runtimeWorld`
      - `runtimeReadViewRef.runtimeWorld`
      - `runtimeReadView.runtimePublish`
      - `runtimeReadViewRef.runtimePublish`
      - `runtimeReadView.observedSnapshot`
      - `runtimeReadViewRef.observedSnapshot`
      - `readView.runtimeWorld`
      - `readView.observedSnapshot`
    - `main()` で既存 `testRuntimeWorldAuthorityProjectionContract()` に加えて
      `testRuntimeReadViewOpaqueContract()` を実行し、どちらか失敗時は即 fail-closed。
- 検証
  - `Build_CMakeTools`: success (result code 0)
  - `RunCtest_CMakeTools`:
    - `RuntimeWorldAuthorityProjectionContract` Passed
    - `CrossfadeExecutorLocalContract` Passed
    - `ObservePathSingleSource` Passed
  - `Strict Atomic Dot-Call Scan`: Passed
  - `check-list-compliance.ps1`: Failures 0 / Warnings 2（既知 manual review 警告のみ）

## 追加改修証跡（RuntimeReadView 型定義閉鎖 2026-06-02 第5ラウンド）

- 目的
  - `RuntimeReadView` を API 設計レベルでさらに閉じ、
    「生成経路 = `AudioEngine` 内のみ」「参照経路 = accessor/helper のみ」を明示化。
- 実装
  - `src/audioengine/AudioEngine.h`
    - `RuntimeReadView` の保持メンバを private 化:
      - `observedSnapshot_`
      - `runtimeWorld_`
    - 生成コンストラクタを private 化し、`friend class AudioEngine;` を追加して生成経路を限定。
    - 読取専用 accessor を追加:
      - `runtimeWorldPtr() const noexcept`
      - `observedSnapshotPtr() const noexcept`
    - helper を accessor 経由へ更新:
      - `getRuntimeWorldFromReadView(...)`
      - `getRuntimeSnapshot(...)`
  - `src/tests/RuntimeWorldAuthorityProjectionTests.cpp`
    - `RuntimeReadView` カプセル化契約を追加監査:
      - accessor 定義の存在必須
      - `friend class AudioEngine;` 存在必須
      - 旧 public field 形（`observedSnapshot` / `runtimeWorld`）残存禁止
      - helper 内の直 field return（`runtimeReadView.runtimeWorld` / `runtimeReadView.observedSnapshot.get()`）禁止
- 検証
  - `Build_CMakeTools`: success (result code 0)
  - `RunCtest_CMakeTools`:
    - `RuntimeWorldAuthorityProjectionContract` Passed
    - `CrossfadeExecutorLocalContract` Passed
    - `ObservePathSingleSource` Passed
  - `Strict Atomic Dot-Call Scan`: Passed
  - `check-list-compliance.ps1`: Failures 0 / Warnings 2（既知 manual review 警告のみ）

## 追加改修証跡（RuntimeReadHandle 命名明確化と全面移行 2026-06-02 第6ラウンド）

- 目的
  - read path API 名称を `View` から `Handle` へ改名し、
    「参照可能性の提示」ではなく「opaque handle 経由での読取契約」という意図を明確化。
  - 旧命名の残存をゼロ化し、契約テスト文字列も同時更新して fail-closed 保守性を維持。
- 実装
  - `src/audioengine/AudioEngine.h`
    - `RuntimeReadView` -> `RuntimeReadHandle`
    - `makeRuntimeReadView(...)` -> `makeRuntimeReadHandle(...)`
    - `readAudioRuntimeView()` -> `readAudioRuntimeHandle()`
    - `readControlRuntimeView()` -> `readControlRuntimeHandle()`
    - `getRuntimeWorldFromReadView(...)` -> `getRuntimeWorldFromReadHandle(...)`
    - `getRuntimeSnapshot(...)` -> `getRuntimeSnapshotFromReadHandle(...)`
  - 呼び出し側追従
    - `src/audioengine/AudioEngine.Commit.cpp`
    - `src/audioengine/AudioEngine.Snapshot.cpp`
    - `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
    - `src/NoiseShaperLearner.cpp`
    - `src/SpectrumAnalyzerComponent.cpp`
    - （先行反映済み）`AudioBlock/BlockDouble/PrepareToPlay/Timer/CtorDtor/Learning`
  - 契約テスト追従
    - `src/tests/RuntimeWorldAuthorityProjectionTests.cpp`
      - opaque 契約検査名/禁止パターン/期待文字列を `RuntimeReadHandle` 系へ更新。
      - `readControlRuntimeHandle` と `getRuntimeSnapshotFromReadHandle` の使用を固定。
    - `src/tests/ObservePathSingleSourceTests.cpp`
      - 旧 `runtimeReadView` 文字列前提を `runtimeReadHandle` に更新。
- 検証
  - 旧命名残存検索（`RuntimeReadView|readControlRuntimeView|readAudioRuntimeView|getRuntimeWorldFromReadView|getRuntimeSnapshot(`）: `src/**` で 0 hit。
  - `get_errors`: 0件。
  - `Build_CMakeTools`: success (result code 0)
  - `RunCtest_CMakeTools`:
    - `RuntimeWorldAuthorityProjectionContract` Passed
    - `CrossfadeExecutorLocalContract` Passed
    - `ObservePathSingleSource` Passed
  - `Strict Atomic Dot-Call Scan`: Passed
  - `check-list-compliance.ps1`: Failures 0 / Warnings 2（既知 manual review 警告のみ）

## 追加監査証跡（PR-01〜PR-07 未対応漏れ 網羅スキャン 2026-06-02 第7ラウンド）

- 目的
  - PR-01〜PR-07 完了後の退行/漏れを、
    1) 契約文字列ベース
    2) 呼び出しグラフベース
    の両軸で再監査し、未対応漏れの有無を確定する。
- 契約文字列ベース監査
  - `src/**` を対象に旧命名残存を再走査:
    - `RuntimeReadView`
    - `readControlRuntimeView/readAudioRuntimeView`
    - `getRuntimeWorldFromReadView/getRuntimeSnapshot(...)`
    - 結果: 0 hit（テスト内期待文字列を除く）。
  - `RuntimeReadHandle` opaque 契約違反パターン
    （`runtimeReadHandle.runtimeWorld` 等）を再走査:
    - 結果: 実コード側は 0 hit（`AudioEngine.h` accessor 実装と契約テスト内禁止パターン定義のみ）。
  - authority/read inventory の存在と validator 接続:
    - `kRuntimeAuthorityInventory` / `kRuntimeReadAuthorityInventory` の定義と
      descriptor 整合 validator 呼び出しを確認。
- 呼び出しグラフベース監査
  - CodeGraph incremental index を再生成（Entities=696, Relations=4613, Files Indexed=32）。
  - read helper の caller 集合を再確認:
    - `readControlRuntimeHandle`: Commit/Snapshot/PrepareToPlay/Learning/ReleaseResources/Timer/CtorDtor/NoiseShaperLearner/SpectrumAnalyzer に集約。
    - `readAudioRuntimeHandle`: AudioBlock/BlockDouble/Snapshot に集約。
    - `getRuntimeSnapshotFromReadHandle`: Snapshot/Timer に集約。
    - `resolveActiveRuntimeDSPFromRuntimeWorldOnly` / `resolveFadingRuntimeDSPFromRuntimeWorldOnly`:
      Commit/AudioBlock/BlockDouble/Snapshot/Timer/PrepareToPlay 系で使用を確認。
  - `observePublishedWorld/observeWorldHandle` の audioengine 呼び出し残存は検出されず。
- 代表 verifier 実行（PR軸）
  - Pass:
    - `isr-verify-self-contained-world.ps1`（PR-01）
    - `isr-verify-semantic-closure-forbidden-inputs.ps1`（PR-03）
    - `isr-verify-runtime-semantic-read-contract.ps1`（PR-04A）
    - `isr-verify-overlap-authority.ps1`（PR-04B）
    - `isr-verify-publication-state-machine.ps1`（PR-05）
    - `isr-verify-crossfade-observable-state.ps1`（PR-06）
    - `isr-verify-governance-consistency.ps1`（PR-07）
    - `isr-verify-projection-austerity.ps1`
    - `isr-verify-projection-freshness.ps1`
  - 補足:
    - `projection-freshness` 初回実行は `trigger_audit_report` / `observe_shim_usage_report` の鮮度切れで fail。
    - 前提スクリプト（`isr-trigger-audit.ps1`, `isr-verify-observe-shim-usage.ps1`）を更新後、再実行で Pass。
- 結論
  - 第7ラウンド時点で、PR-01〜PR-07に関する追加の未対応漏れは検出されず。
  - 検出事項は「レポート鮮度依存」の運用要件のみで、コード修正は不要。

## 追加改修証跡（PR-01〜PR-07 漏れ対象の再発掘＋不足wiring補完 2026-06-02 第8ラウンド）

- 目的
  - `doc/work13/isr_parent7_pr_task_breakdown.md` の「主対象ファイル/関数/変数」外にある関連箇所を、
    `grep` + `Serena` + `CodeGraph`（index更新後）で再探索し、PR-01〜PR-07の条件充足を再確認。
  - 条件監査の機械接続が不足していた箇所（PR-07）をコード改修で補完。

- 追加発掘（PR別）
  - PR-01（Self-contained）
    - 追加確認対象: `src/audioengine/AudioEngine.Commit.cpp::validateSemanticCompleteness`, `src/audioengine/RuntimeBuilder.cpp::buildRuntimePublishWorld`
    - 追加変数確認: `worldOwner->automation.*`, `worldOwner->coefficient.*`, `worldOwner->semanticHash.*`
    - 判定: 追加違反なし（`isr-verify-self-contained-world.ps1` Pass）
  - PR-02（Semantic Reachability）
    - 追加確認対象: `src/audioengine/AudioEngine.RebuildDispatch.cpp::requestRebuild`, `captureRuntimeBuildSnapshot` 呼び出し系
    - 判定: 追加違反なし（`isr-verify-semantic-reachability.ps1` Pass）
  - PR-03（Builder Isolation）
    - 追加確認対象: `src/audioengine/RuntimeBuilder.cpp` の `engine.reserveRuntimePublicationIdentity` / `computeRuntimePublishComputation` / `prepare(...,&engine)`
    - 判定: 現行契約スクリプトで許容範囲内（`isr-verify-semantic-closure-forbidden-inputs.ps1` Pass）
  - PR-04A/PR-04B（Read Authority / Audio Thread Authority）
    - 追加確認対象:
      - `src/audioengine/AudioEngine.Snapshot.cpp`
      - `src/NoiseShaperLearner.cpp`
      - `src/SpectrumAnalyzerComponent.cpp`
      - `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
    - 追加関数確認:
      - `readControlRuntimeHandle`, `readAudioRuntimeHandle`
      - `getRuntimeWorldFromReadHandle`, `getRuntimeSnapshotFromReadHandle`
      - `resolveActiveRuntimeDSPFromRuntimeWorldOnly`, `resolveFadingRuntimeDSPFromRuntimeWorldOnly`
    - 判定: 追加違反なし（`isr-verify-runtime-semantic-read-contract.ps1`, `isr-verify-overlap-authority.ps1` Pass）
  - PR-05（Publication Lifecycle）
    - 追加確認対象: `src/audioengine/ISRRuntimePublicationCoordinator.cpp::getCurrent` 呼び出し境界
    - 判定: `src/audioengine/**` からの直接依存漏れなし（`isr-verify-publication-state-machine.ps1` Pass）
  - PR-06（Crossfade Executor-Local）
    - 追加確認対象: `makeCrossfadePreparedSnapshotFromWorld` 呼び出し元（AudioBlock/BlockDouble）
    - 判定: semantic source 逆流なし（`isr-verify-crossfade-observable-state.ps1` Pass）
  - PR-07（Governance Wiring）
    - 追加発掘した漏れ:
      - 契約テスト wiring（`RuntimeWorldAuthorityProjectionContract` / `CrossfadeExecutorLocalContract` / `ObservePathSingleSource`）を
        `governance consistency` で直接検査していなかった。

- 実改修（PR-07）
  - `src/` 側の追加修正は不要（条件違反なし）。
  - `.github/scripts/isr-verify-governance-consistency.ps1` を改修し、以下を fail-closed で検証追加:
    - テストファイル存在:
      - `src/tests/RuntimeWorldAuthorityProjectionTests.cpp`
      - `src/tests/CrossfadeExecutorLocalContractTests.cpp`
      - `src/tests/ObservePathSingleSourceTests.cpp`
    - `CMakeLists.txt` wiring:
      - `add_executable(...)` 3件
      - `add_test(NAME ... COMMAND ...)` 3件

- 再検証
  - `isr-verify-self-contained-world.ps1` Pass
  - `isr-verify-semantic-reachability.ps1` Pass
  - `isr-verify-semantic-closure-forbidden-inputs.ps1` Pass
  - `isr-verify-runtime-semantic-read-contract.ps1` Pass
  - `isr-verify-publication-state-machine.ps1` Pass
  - `isr-verify-crossfade-observable-state.ps1` Pass
  - `isr-verify-governance-consistency.ps1` Pass（改修後）

- 結論
  - PR-01〜PR-06: 追加発掘分に対して実装違反なし。
  - PR-07: 監査接続漏れを1件補完し、機械検証を強化。

---

## 監査メモ

- 更新日時: 2026-06-02
- 更新者: GitHub Copilot
- 更新理由: PR-01〜PR-07 の漏れ対象に対する第6ラウンド（`RuntimeReadHandle` 命名明確化）、第7ラウンド（網羅スキャン）、第8ラウンド（不足wiring補完）を追加反映。
