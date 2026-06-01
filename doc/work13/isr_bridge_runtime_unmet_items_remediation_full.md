# Practical Stable ISR Bridge Runtime 未達項目 改修完全一覧

作成日: 2026-06-01
対象: ConvoPeq `main`

## 目的

未達項目ごとに、対象ファイル・対象関数・対象変数/フィールド・改修内容を漏れなく整理する。

## 改修マトリクス（未達項目別）

| ID | 未達項目 | 対象ファイル | 対象関数 | 対象変数/フィールド | 改修内容 |
| --- | --- | --- | --- | --- | --- |
| 1 | RuntimeWorld Self-contained 化未達 | `AudioEngine.Parameters.cpp`, `AudioEngine.h`, `RuntimeBuilder.cpp`, `RuntimeGraph.h`, `ISRRuntimeSemanticSchema.h` | `applyDefaultsForCurrentMode`, `setEqBypassRequested`, `setConvolverBypassRequested`, `setProcessingOrder`, `setInputHeadroomDb`, `setOutputMakeupDb`, `setConvolverInputTrimDb`, `makeEngineRuntimeState`, `makeRuntimeGraphState`, `buildRuntimePublishWorld` | `eqBypassRequested`, `convBypassRequested`, `currentProcessingOrder`, `inputHeadroomDb`, `outputMakeupDb`, `convolverInputTrimDb`, `inputHeadroomGain`, `outputMakeupGain`, `convolverInputTrimGain`, `m_currentInputHeadroomDb`, `m_currentOutputMakeupDb`, `m_currentConvInputTrimDb`, `RuntimeGraph::eqBypassed`, `RuntimeGraph::convBypassed`, `RuntimeGraph::inputHeadroomGain`, `RuntimeGraph::outputMakeupGain`, `RuntimeGraph::convolverInputTrimGain` | atomic直読でのsemantic決定を排除し、`RuntimeBuilder`にauthorityを集約。setterは入力受付/再ビルド指示のみ。 |
| 2 | Transition Semantic Leakage | `AudioEngine.h`, `AudioEngine.Processing.AudioBlock.cpp`, `AudioEngine.Processing.BlockDouble.cpp`, `AudioEngine.Processing.Snapshot.cpp`, `AudioEngine.Timer.cpp`, `RuntimeTransition.h` | `makeRuntimePublishView`, `makeRuntimeReadView`, `resolveActiveRuntimeDSPFromRuntimeWorldOnly`, `resolveFadingRuntimeDSPFromRuntimeWorldOnly`, AudioBlock/BlockDoubleのtransition参照ブロック | `runtimePublishView.transition.current/next/active`, `TransitionState::current/next/active`, `RuntimeWorld::topology.hasFadingRuntime`, `runtimeUuid`, `fadingRuntimeUuid` | DSP分岐authorityを`transition.*`から`topology.hasFadingRuntime`へ移行。transitionはexecutor-local/diagnostic限定。 |
| 3 | Crossfade semantic executor-local 化未達 | `AudioEngine.h`, `RuntimeTransition.h`, `RuntimeBuilder.cpp`, `AudioEngine.Commit.cpp`, `AudioEngine.Processing.AudioBlock.cpp`, `AudioEngine.Processing.BlockDouble.cpp`, `AudioEngine.Timer.cpp` | `refreshCrossfadePreparedSnapshotFromAtomics`, `publishCrossfadePreparedSnapshot`, `consumeCrossfadePreparedSnapshot`, `makeCrossfadePreparedSnapshotFromWorld`, crossfade実行ブロック | `latencyDelayOld`, `latencyDelayNew`, `dspCrossfadePending`, `dspCrossfadeUseDryAsOld`, `firstIrDryCrossfadePending`, `dspCrossfadeStartDelayBlocks`, `dspCrossfadeDryHoldSamples`, `latencyResetPending`, `dspCrossfadeDryScaleTarget`, `CrossfadePreparedSnapshot::*` | crossfade実行状態をRuntime semanticから切り離し、executor-local snapshotへ封じ込める。 |
| 4 | Legacy Build Generation 二重系統 | `RuntimeBuildTypes.h`, `AudioEngine.RebuildDispatch.cpp`, `RuntimeBuilder.cpp`, `RuntimeGraph.h`, `ISRRuntimeSemanticSchema.h` | `captureRuntimeBuildSnapshot`, `finalizeRuntimeBuildSnapshot`, `sealRuntimeBuildSnapshot`, `isRuntimeBuildSnapshotSealedAndCompatible`, `buildRuntimePublishWorld` | `RuntimeBuildSnapshot::generation`, `snapshot.generation`, `RuntimeGraph::generation`, `RuntimePublishWorld::generation`, `runtimeVersion`, `PublicationSemantic::mappedRuntimeGeneration` | build用generationを削減/診断化し、authoritative generationをRuntimeWorld/Publicationに単一化。 |
| 5 | Runtime Identity/Timeline singularization 未完 | `RuntimeBuilder.cpp`, `RuntimeGraph.h`, `RuntimeTransition.h`, `ISRRuntimeSemanticSchema.h`, `ISRRuntimePublicationCoordinator.cpp` | `buildRuntimePublishWorld`, `RuntimePublicationCoordinator::commit`, `getVersion` | `world.generation`, `world.runtimeVersion`, `world.transitionId`, `world.publication.sequenceId/epoch/mappedRuntimeGeneration`, `runtimeUuid`, `fadingRuntimeUuid`, `transitionCurrentRuntimeUuid`, `transitionNextRuntimeUuid` | identity/time軸をauthority/derived/diagnosticで固定し、`runtimeVersion`は診断ミラーに限定。 |
| 6 | RuntimeCoordinator consume 完成形未達 | `RuntimePublicationCoordinator.h`, `ISRRuntimePublicationCoordinator.h/.cpp`, `AudioEngine.h` | `observePublishedWorld`, `observeWorldHandle`, `getCurrent`, （追加）`consume` | `RuntimeStore::current`, coordinator state (`state_`, `swapPending_`) | observe散在をconsume中心へ統合。`getCurrent()==nullptr`暫定実装を解消。 |
| 7 | Runtime topology authority split 未解消 | `RuntimeBuilder.cpp`, `AudioEngine.h`, `AudioEngine.Timer.cpp`, `AudioEngine.Processing.AudioBlock.cpp`, `AudioEngine.Processing.BlockDouble.cpp` | `buildRuntimePublishWorld`, `makeEngineRuntimeState`, `resolveFadingRuntimeDSPFromRuntimeWorldOnly` | `topology.hasFadingRuntime`, `engine.transition.current/next/active` | topologyを唯一authorityに固定し、transitionは投影/観測用途へ限定。 |
| 8 | Legacy mutable runtime semantic removal 未完 | `AudioEngine.h`, `AudioEngine.Parameters.cpp`, `ISRRuntimeSemanticSchema.h`, `.github/isr-legacy-temporary.json` | bypass/order/gain/oversampling/noise-shaper系 setter/getter, `getCurrentState`, `requestLoadState` | `eqBypassRequested`, `convBypassRequested`, `currentProcessingOrder`, `inputHeadroomDb`, `outputMakeupDb`, `convolverInputTrimDb`, `manualOversamplingFactor`, `oversamplingType`, `noiseShaperType` | `Authoritative/Derived/Diagnostic/ExecutorLocal/LegacyTemporary`へ再分類し、manifest期限でfail-closed化。 |
| 9 | Shadow Compare 実運用化不足 | `AudioEngine.Commit.cpp`, `ISRDebugRuntime.cpp`, `ShadowCompareContractTests.cpp`, `RuntimeSemanticSchemaValidationTests.cpp`, `.github/scripts/isr-verify-shadow-compare-cadence.ps1`, `.github/scripts/isr-verify-shadow-compare-coverage.ps1` | `classifySemanticEquivalence`, `validateSemanticCompleteness`, commit/publish経路 | `RuntimeSemanticHash::*`, `observeMonotonicRollbackRequested_`, `publicationRejectCount_`, `debugRuntime_` | compare結果をtelemetry/rollback triggerへ接続し、publish pathで常時運用。 |
| 10 | Authority Duplication Verifier 実接続不足 | `ISRRuntimeSemanticSchema.h`, `AudioEngine.Commit.cpp`, `RuntimeBuilder.cpp` | `validateAuthorityInventoryAgainstDescriptors`, `validateFieldDescriptorSet`, `validateSemanticCompleteness` | `RuntimeAuthorityInventoryPolicy::kExhaustivenessEnforced`, `kSchemaInventoryMismatchFails`, `kFieldDescriptors` | schema/inventory mismatchをbuild/publish gateでfail-closed。 |
| 11 | Diagnostic-only boundary verifier 強化 | `ISRRuntimeSemanticSchema.h`, `AudioEngine.Timer.cpp`, `AudioEngine.Commit.cpp` | `ObserveForbiddenTypeVerifier::isForbiddenTypeName`, timer diagnostics path | `TransitionState*` forbidden list, transition診断参照群 | 診断参照のauthority分岐混入を明示ガードで防止。 |
| 12 | RuntimeStore ownership model 明確化不足 | `RuntimeStore.h`, `RuntimePublicationCoordinator.h`, `ISRRuntimePublicationCoordinator.cpp`, `AudioEngine.Commit.cpp` | `RuntimeStore::observe`, `publishAndSwap`, `publishState`, `clearPublishedRuntimeSnapshotsNonRt`, `retire` | `RuntimeStore::current`, `WriteAccess::store_`, `oldWorld/newWorld` swap pointer, `retireBacklogCount_`, `reclaimInFlightCount_` | ownership transfer matrixをpre/post checkへ反映し、retire/ABA監視をreject/rollback接続。 |
| 13 | Publication Atomicity 例外経路 | `RuntimePublicationCoordinator.h` | `clearPublishedRuntimeSnapshotsNonRt` | `publishAndSwap(nullptr)` | shutdown専用例外contractとして明文化し、通常publish pathと分離。 |
| 14 | RuntimeBuilder pure semantic builder 化未完 | `RuntimeBuilder.h`, `RuntimeBuilder.cpp` | `RuntimeBuilder::RuntimeBuilder`, `buildRuntimePublishWorld`, `build` | `AudioEngine& engine` | builder入力をnarrow interface化し、`AudioEngine`依存を縮小。 |
| 15 | Verifier/Governance 名称・実体整合 | `ISRRuntimeSemanticSchema.h`, `.github/isr-verifier-registry.json`, `.github/scripts/isr-verify-governance-registries.ps1` | registry validation処理 | `kRequiredVerifierTable`, registry tiers (`pr/nightly/release`) | C++ verifier名とregistry名の対応を明示し、未配線verifierを補完。 |
| 16 | Rollback governance state machine 統合不足 | `AudioEngine.Commit.cpp`, `AudioEngine.h`, `ISRDebugRuntime.cpp`, `.github/isr-rollback-compatibility-matrix.json` | rollback request/escalation判定群, `DebugRuntime::escalationCount` | `observeMonotonicRollbackRequested_`, `retireEscalationCount_`, `retireRuntimeEx_` | rollback/cooldown/escalation条件を単一state machineへ統合。 |
| 17 | Publication monotonicity/sequence 運用接続 | `ISRRuntimePublicationCoordinator.cpp`, `RuntimePublicationCoordinatorTests.cpp`, `.github/isr-verifier-registry.json`, `.github/scripts/isr-verify-governance-registries.ps1` | `RuntimePublicationCoordinator::commit`, `testRejectRepublishAndRollback` | `publicationSequenceId_`, `publicationEpoch_`, `mappedRuntimeGeneration_` | 単調性チェックをtier実行配線で必須化。 |
| 18 | Semantic hash equivalence 運用接続 | `RuntimeBuilder.cpp`, `AudioEngine.Commit.cpp`, `ShadowCompareContractTests.cpp`, `.github/isr-verifier-registry.json` | `classifySemanticEquivalence`, publish precheck/commit path | `RuntimeSemanticHash::*` | hash生成→比較→mismatch telemetry→rollbackの連鎖を必須化。 |
| 19 | RuntimeSemanticSchema 完全分類証跡 | `ISRRuntimeSemanticSchema.h`, `RuntimeGraph.h`, `RuntimeTransition.h`, `RuntimeSemanticSchemaValidationTests.cpp` | descriptor validation群, schema validation tests | `kFieldDescriptors`, semantic descriptor fields | 全runtime field分類をdescriptor+testでfail-closed化。 |
| 20 | Nightly/Release/Fail-Closed/Soak/Manifest 運用固定 | `.github/workflows/isr-verification.yml`, `.github/isr-validator-tiering-policy.json`, `.github/isr-verifier-execution-layers.json`, `.github/isr-verifier-registry.json`, `.github/isr-legacy-temporary.json`, `.github/isr-8_1-close-policy.json`, `.github/scripts/isr-run-tiered-verification.ps1`, `.github/scripts/isr-verify-soak-governance.ps1`, `.github/scripts/isr-verify-authority-inventory.ps1` | tier scheduling, expiry guard, soak matrix, fail-closed throw条件 | `nightlyCron`, `weeklyCron`, `nightlyTier`, `weeklyTier`, `SoakMinutes`, `expiry`, `expiryGuardDaysByTier` | tier別必須検証を強制し、expiry超過legacy manifestをCI fail-closedで拒否。 |
| 33 | Runtime Publication Read Consistency Contract | `RuntimePublicationCoordinator.h`, `RuntimeStore.h`, `AudioEngine.h`, `ISRRuntimePublicationCoordinator.h/.cpp` | `observePublishedWorld`, `observeWorldHandle`, `makeRuntimePublishView`, `makeRuntimeReadView`, `getCurrent`, （追加）`consume` | `RuntimeStore::current`, `RuntimePublishView::runtimeWorld`, `RuntimePublishView::transition`, `swapPending_` | publish→consume→read を単一路へ統一し、observe系は内部実装へ降格する。 |
| 34 | RuntimeWorld Completeness Verification | `ISRRuntimeSemanticSchema.h`, `RuntimeBuilder.cpp`, `RuntimeSemanticSchemaValidationTests.cpp`, `AudioEngine.Commit.cpp` | `buildRuntimePublishWorld`, `validateSemanticCompleteness`, （追加）`RuntimeWorldCompletenessVerifier` | `BuildInput`, `RuntimeWorld.*`, `RuntimeSemanticHash::*`, `kFieldDescriptors` | Builder入力→RuntimeWorld→SemanticHash 到達性を fail-closed で検証し、未到達をCIで拒否する。 |
| 35 | Runtime Rebuild Trigger Authority | `AudioEngine.Parameters.cpp`, `AudioEngine.RebuildDispatch.cpp`, `AudioEngine.UIEvents.cpp`, `AudioEngine.Timer.cpp`, `AudioEngine.StateIO.cpp` | `submitRebuildIntent`, `requestRebuild`, `handleAsyncUpdate`, `rebuildThreadLoop`, `endBulkParameterRestore`, `requestLoadState`, `convolverParamsChanged` | `RebuildTelemetryReason::*`, `pendingRebuildKinds`, `needsRebuild`, `rebuildGeneration` | semantic変更点が rebuild トリガに100%到達することを verifier で保証する。 |
| 36 | Runtime Publish→Retire Full Lifecycle Contract | `RuntimeStore.h`, `RuntimePublicationCoordinator.h`, `ISRRuntimePublicationCoordinator.cpp`, `AudioEngine.Processing.ReleaseResources.cpp`, `AudioEngine.CtorDtor.cpp` | `publishAndSwap`, `publishState`, `retire`, `isFullyDrained`, `requestShutdown`, `markShutdownComplete`, （追加）`LifecycleStateMachineVerifier` | `retireBacklogCount_`, `publicationBacklogCount_`, `reclaimInFlightCount_`, `deferredRetireResidencyCount_`, `state_`, `swapPending_` | Construct→Seal→Publish→Observe→Consume→Retire→Reclaim→Destroy の遷移を契約化し、逸脱を fail-closed にする。 |
| 37 | Runtime Semantic Coverage Closure | `RuntimeBuilder.cpp`, `RuntimeBuildTypes.h`, `ISRRuntimeSemanticSchema.h`, `AudioEngine.Parameters.cpp`, `AudioEngine.RebuildDispatch.cpp` | `submitRebuildIntent`, `captureRuntimeBuildSnapshot`, `buildRuntimePublishWorld`, `validateSemanticCompleteness`, （追加）`SemanticCoverageVerifier` | semantic source fields, `BuildInput`, `RuntimeBuildSnapshot::buildInput`, `RuntimeWorld::*`, `RuntimeSemanticHash::*` | `semantic source -> rebuild trigger -> BuildInput -> RuntimeWorld -> SemanticHash` の閉路を fail-closed で検証する。 |
| 38 | RuntimeWorld Projection Consistency | `AudioEngine.h`, `RuntimePublicationCoordinator.h`, `AudioEngine.Commit.cpp`, `RuntimeSemanticSchemaValidationTests.cpp` | `makeRuntimePublishView`, `makeRuntimeReadView`, `readAudioRuntimeView`, `readControlRuntimeView`, （追加）`ProjectionConsistencyVerifier` | `RuntimePublishView`, `RuntimeReadView`, `runtimeWorld`, `transition`, `projectionFreshness` | RuntimeWorld→PublishView→ReadView の投影欠落・不整合を機械検証し、分離経路による乖離を検出する。 |
| 39 | RuntimeHash Authority Freeze | `AudioEngine.h`, `ISRRuntimeSemanticSchema.h`, `RuntimeBuilder.cpp` | `validateRuntimeSemanticSchemaContract`, `validateAuthorityInventoryAgainstDescriptors`, hash生成部 | `kFieldDescriptors[semanticHash]`, `kAuthorityInventory[semanticHash]`, `RuntimeSemanticHash::*` | `semanticHash` を non-authoritative (`Diagnostic`) として固定し、authority昇格の回帰を拒否する。 |
| 40 | RuntimeWorld Mutation Boundary Freeze | `ISRSealedObject.h`, `RuntimeBuilder.cpp`, `AudioEngine.Commit.cpp`, `RuntimeSemanticSchemaValidationTests.cpp` | `createForBuilder`, `assertMutable`, `freeze`, `isSealedRecursively`, （追加）`RuntimeSealVerifier` | `sealState_`, `sealViolationCount`, `RuntimeBuildSnapshot::sealed` | `freeze()` 後 mutation を禁止し、sealed object 境界違反を CI で fail-closed にする。 |

## 推奨実装順

1. Runtime authority 単一化（ID: 1,2,5,7）
2. Crossfade executor-local 化（ID: 3）
3. Generation/Identity 単一化（ID: 4,5）
4. Publication/Ownership/Consume API 整理（ID: 6,12,13）
5. Verifier 実接続（ID: 8,10,11,15,17,18,19）
6. Governance/CI 運用固定（ID: 20）
7. Builder 純化（ID: 14）

## 完了条件（DoD）

- Runtime authority が `RuntimeWorld` に収束
- audio thread branching authority が topology中心
- generation/sequence/epoch/version の役割分離が完了
- shadow compare が publish runtime path で稼働
- ownership/ABA/retire が fail-closed で検証
- nightly/release/soak/manifest expiry が CI 機械検証で担保

## レビュー反映判定（2026-06-01）

### 採用（妥当・第2版）

- RuntimeWorld Self-contained 化未達
- Transition Semantic Leakage
- Crossfade executor-local 化未達
- Legacy Build Generation / Identity / Timeline の単一化未完
- RuntimeCoordinator consume 完成形未達（`observe*` 依存、`getCurrent()==nullptr`）
- RuntimeBuilder の pure semantic builder 化未完（`AudioEngine& engine` 依存）

### 条件付き採用（実装は存在するが運用接続の強化が必要）

- Shadow Compare（契約テストは存在、publish runtime path の常時運用接続を強化）
- Publication monotonicity / sequence（実装・テストは存在、tier wiring の固定化を強化）
- Deterministic Build / SemanticDependencyGraph（verifier登録・scriptは存在、実行経路の固定化を強化）

### 不採用（「不在」指摘は誤り）

- Nightly/Release tier governance 不在
- Fail-Closed governance 不在
- Soak validation infrastructure 不在
- Legacy manifest expiry system 不在

上記は `.github` 配下（`isr-verifier-registry.json`, `isr-validator-tiering-policy.json`, `isr-verifier-execution-layers.json`, `workflows/isr-verification.yml`, `scripts/isr-run-tiered-verification.ps1` ほか）に実装が存在する。

## 追補：改修対象の具体化（grep / Serena / CodeGraph 追加探索）

### P1. RuntimeCoordinator Centralization（consume収束）

- 対象ファイル: `src/core/RuntimePublicationCoordinator.h`, `src/audioengine/ISRRuntimePublicationCoordinator.h`, `src/audioengine/ISRRuntimePublicationCoordinator.cpp`, `src/audioengine/AudioEngine.h`
- 対象関数: `RuntimePublicationCoordinator::observePublishedWorld(...)`, `RuntimePublicationCoordinator::observeWorldHandle(...)`, `RuntimePublicationCoordinator::getCurrent() const noexcept`, `AudioEngine::makeRuntimePublishView(...)`, `AudioEngine::makeRuntimeReadView(...)`
- 対象変数: `RuntimeStore::current`, `RuntimePublicationCoordinator::state_`, `RuntimePublicationCoordinator::swapPending_`
- 改修方針: `observe*` 呼び出し散在を `consume()`（新設または等価API）に集約し、`getCurrent()==nullptr` 暫定実装を廃止する。

### P2. RuntimeStore Ownership / retire 安全性の実装接続強化

- 対象ファイル: `src/core/RuntimeStore.h`, `src/core/RuntimePublicationCoordinator.h`, `src/audioengine/ISRRuntimePublicationCoordinator.cpp`, `src/audioengine/AudioEngine.Commit.cpp`
- 対象関数: `RuntimeStore::observe() const noexcept`, `RuntimeStore::WriteAccess::publishAndSwap(T* next) noexcept`, `RuntimePublicationCoordinator::publishState(...)`, `RuntimePublicationCoordinator::retire(...)`, `RuntimePublicationCoordinator::clearPublishedRuntimeSnapshotsNonRt()`
- 対象変数: `RuntimeStore::current`, `RuntimePublicationCoordinator::retireBacklogCount_`, `publicationBacklogCount_`, `reclaimInFlightCount_`, `deferredRetireResidencyCount_`
- 改修方針: `publish(nullptr)` を shutdown 専用例外契約として明文化し、ownership transfer matrix（publish→observe→retire）を verifier/precheck に接続する。

### P3. Transition / topology 二重authorityの解消

- 対象ファイル: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`, `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`, `src/audioengine/AudioEngine.Processing.Snapshot.cpp`, `src/audioengine/AudioEngine.Timer.cpp`, `src/audioengine/RuntimeTransition.h`, `src/audioengine/RuntimeGraph.h`
- 対象関数: `AudioEngine::getNextAudioBlock(...)`, `AudioEngine::processDouble(...)`, `AudioEngine::makeRuntimeGraphState(...)`, `AudioEngine::resolveFadingRuntimeDSPFromRuntimeWorldOnly(...)`
- 対象変数: `runtimePublishView.transition.current`, `runtimePublishView.transition.next`, `runtimePublishView.transition.active`, `RuntimeGraph::runtimeUuid`, `RuntimeGraph::fadingRuntimeUuid`, `RuntimeGraph::transitionCurrentRuntimeUuid`, `RuntimeGraph::transitionNextRuntimeUuid`, `RuntimeWorld.topology.hasFadingRuntime`
- 改修方針: audio-thread branching を `topology.hasFadingRuntime` 中心に統一し、`transition.*` は executor-local/diagnostic として隔離する。

### P4. Deterministic Build の入力境界強化

- 対象ファイル: `src/audioengine/RuntimeBuilder.h`, `src/audioengine/RuntimeBuilder.cpp`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`, `src/ConvolverProcessor.h`
- 対象関数: `RuntimeBuilder::RuntimeBuilder(AudioEngine& owner) noexcept`, `RuntimeBuilder::buildRuntimePublishWorld(...)`, `RuntimeBuilder::build(const BuildInput& in) noexcept`, `captureRuntimeBuildSnapshot(...)`, `ConvolverProcessor::captureBuildSnapshot() const`
- 対象変数: `RuntimeBuilder::engine`, `RuntimeBuildSnapshot::buildInput`, `RuntimeBuildSnapshot::rebuildFingerprint`, `RuntimeBuildFingerprint::dspParameterHash`
- 改修方針: builder 入力境界を `BuildInput + immutable snapshot` に収束し、`AudioEngine` 直接依存の mutable state 参照を縮小する。

### P5. RuntimeWorld Self-contained の最終収束（追加明確化）

- 対象ファイル: `src/audioengine/AudioEngine.Parameters.cpp`, `src/audioengine/AudioEngine.h`, `src/audioengine/RuntimeBuilder.cpp`
- 対象関数: `AudioEngine::applyDefaultsForCurrentMode()`, `AudioEngine::setInputHeadroomDb(float)`, `AudioEngine::setOutputMakeupDb(float)`, `AudioEngine::setConvolverInputTrimDb(float)`, `RuntimeBuilder::buildRuntimePublishWorld(...)`
- 対象変数: `eqBypassRequested`, `convBypassRequested`, `currentProcessingOrder`, `inputHeadroomDb`, `outputMakeupDb`, `convolverInputTrimDb`, `inputHeadroomGain`, `outputMakeupGain`, `convolverInputTrimGain`
- 改修方針: `applyDefaultsForCurrentMode()` の authority を廃止し、semantic 決定を builder に移送。setter は入力受付 + rebuild trigger のみに制限する。

## 探索エビデンス（要約）

- `grep`
  - `AudioEngine.Parameters.cpp` に `applyDefaultsForCurrentMode`, `eqBypassRequested`, `convBypassRequested`, `currentProcessingOrder` を確認。
  - `AudioEngine.Processing.AudioBlock.cpp` / `BlockDouble.cpp` / `Snapshot.cpp` / `Timer.cpp` に `runtimePublishView.transition.current/next` を確認。
  - `ISRRuntimePublicationCoordinator.cpp` で `getCurrent() const noexcept { return nullptr; }` を確認。
  - `RuntimePublicationCoordinator.h` で `publishAndSwap(nullptr)` 経路を確認。
- Serena
  - pattern 検索で上記シンボルの位置を再確認（`AudioEngine.h`, `AudioEngine.RebuildDispatch.cpp`, `RuntimeBuildTypes.h` など）。
- CodeGraph
  - `mcp_codegraph_get_file_structure` で `AudioEngine.Parameters.cpp`, `AudioEngine.Processing.AudioBlock.cpp`, `ISRRuntimePublicationCoordinator.cpp`, `RuntimeBuildTypes.h` の構造を確認。
  - NLクエリ（`mcp_codegraph_query_codebase`）は 0 件だったため、構造解析結果を一次エビデンスとして採用。

## 追加レビュー反映判定（2026-06-01 第2版）

### 採用（妥当）

- ID21: Read-side Authority Freeze 未完
  - 根拠: `AudioEngine.Processing.AudioBlock.cpp`, `BlockDouble.cpp`, `Snapshot.cpp`, `Timer.cpp` で `runtimePublishView.transition.current/next` を参照し、実行側分岐に関与している。
- ID22: Projection Purity 契約の未完
  - 根拠: `AudioEngine.h` の `makeRuntimePublishView` / `makeRuntimeReadView` が `observeWorldHandle(runtimeStore)` と transition pointer を直接束ねる。
- ID23: RuntimeBuildSemanticInput 境界の未完
  - 根拠: `RuntimeBuilder::buildRuntimePublishWorld` が `AudioEngine& engine` 依存で `runtimeStore` 観測を行う。
- ID24: State I/O の authority 混在
  - 根拠: `AudioEngine.StateIO.cpp` の `requestLoadState` / `getCurrentState` が複数 atomic を直接 read/write。
- ID25: Transition pointer authority の実行混入
  - 根拠: audio 実行経路で `transition.current/next` から DSP を直接解決する箇所が残存。

### 条件付き採用（第2版・実装あり運用強化）

- ID26: Construction Exclusivity / Publication Seal
  - 根拠: `AudioEngine.h` に `BuilderToken`, `createForBuilder`、`ISRSealedObject.h` に `assertMutable` / `freeze`、`RuntimeBuilder.cpp` に `worldOwner->assertMutable(); ... worldOwner->freeze();` は存在。
  - 追加改修: 「freeze 後の mutation 禁止」を verifier と CI gate に接続。
- ID27: Ownership State Machine の形式化
  - 根拠: `ISRRuntimePublicationCoordinator.cpp` に `markTransitionStart`, `markTransitionCommitted`, `requestShutdown`, `markShutdownComplete`, `swapPending_` が存在。
  - 追加改修: publish/retire/shutdown の遷移表を fail-closed precheck に統合。
- ID28: Retire Ordering / Reentrancy 監視の強制
  - 根拠: `publicationSequenceId_`, `publicationEpoch_`, `mappedRuntimeGeneration_`, `swapPending_` の単調監視はある。
  - 追加改修: reentrant publish 拒否条件を registry tier 実行へ昇格。
- ID29: Authority Inventory Freeze / Derived Dependency Freeze
  - 根拠: `ISRRuntimeSemanticSchema.h` に `validateAuthorityInventoryAgainstDescriptors`, `SemanticDependencyGraphVerifier`, `DeterministicBuildVerifier` は存在。
  - 追加改修: build/publish の必須前段チェックとして wiring 固定。
- ID30: Verifier Self-Verification の運用固定
  - 根拠: `.github/isr-contract-registry.json`, `.github/isr-verifier-registry.json`, `.github/isr-verifier-execution-layers.json`, `.github/scripts/isr-verify-governance-registries.ps1` が存在。
  - 追加改修: PR tier でも registry 不整合を fail-closed に固定。

### 不採用（第2版・不在指摘は誤り）

- ID31: UUID一意性保護が未実装
  - 反証: `AudioEngine.h` / `AudioEngine.Processing.DSPCoreLifecycle.cpp` に `runtimeUuidCounterStorage_`, `reserveNextRuntimeUuid()` が存在。
- ID32: Governance registry / expiry guard が未実装
  - 反証: `.github/workflows/isr-verification.yml` に expiry guard と tier 解決ロジック、`.github/*policy*.json` に `expiry` が多数存在。

## 追補 P6-P10（第2版レビュー対応の具体改修）

### P6. Read-side Authority Freeze（実行分岐の最終収束）

- 対象ファイル: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`, `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`, `src/audioengine/AudioEngine.Processing.Snapshot.cpp`, `src/audioengine/AudioEngine.Timer.cpp`, `src/audioengine/AudioEngine.h`
- 対象関数: `getNextAudioBlock`, `processDouble`, runtime snapshot 取得系、`makeRuntimePublishView`
- 対象変数: `runtimePublishView.transition.current`, `runtimePublishView.transition.next`, `RuntimeWorld.topology.hasFadingRuntime`
- 改修方針: 実行authorityを `runtimeWorld` 側へ固定し、transition pointer は診断/補助に限定する。

### P7. Projection Purity 契約の明示化

- 対象ファイル: `src/audioengine/AudioEngine.h`, `src/core/RuntimePublicationCoordinator.h`
- 対象関数: `makeRuntimePublishView`, `makeRuntimeReadView`, `observeWorldHandle`
- 対象変数: `RuntimePublishView::runtimeWorld`, `RuntimePublishView::transition`
- 改修方針: projection は immutable world projection のみ返す契約へ縮約し、transition を別診断ビューへ分離する。

### P8. RuntimeBuildSemanticInput 契約（Builder入力境界）

- 対象ファイル: `src/audioengine/RuntimeBuilder.h`, `src/audioengine/RuntimeBuilder.cpp`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`
- 対象関数: `RuntimeBuilder::RuntimeBuilder`, `buildRuntimePublishWorld`, `build(const BuildInput&)`
- 対象変数: `RuntimeBuilder::engine`, `BuildInput`, `RuntimeBuildSnapshot::buildInput`
- 改修方針: `AudioEngine& engine` の直接依存を縮小し、builder入力を semantic snapshot に閉じる。

### P9. State I/O authority 混在の解消

- 対象ファイル: `src/audioengine/AudioEngine.StateIO.cpp`, `src/audioengine/AudioEngine.Parameters.cpp`, `src/audioengine/RuntimeBuilder.cpp`
- 対象関数: `requestLoadState`, `getCurrentState`, `applyDefaultsForCurrentMode`, 各 setter
- 対象変数: `eqBypassRequested`, `convBypassRequested`, `currentProcessingOrder`, `inputHeadroomDb`, `outputMakeupDb`, `convolverInputTrimDb`
- 改修方針: State I/O は「入力復元要求」のみ行い、最終semantic確定は builder 側に一本化する。

### P10. Construction/Seal/Ownership の verifier 接続

- 対象ファイル: `src/audioengine/ISRRuntimeSemanticSchema.h`, `src/audioengine/ISRRuntimePublicationCoordinator.cpp`, `.github/isr-verifier-registry.json`, `.github/isr-verifier-execution-layers.json`, `.github/scripts/isr-run-tiered-verification.ps1`
- 対象関数: authority/descriptor validation 群、publish precheck 群
- 対象変数: verifier registry entries, `swapPending_`, `publicationSequenceId_`, `publicationEpoch_`
- 改修方針: Construction/Seal/Ownership の契約を verifier と tier 実行に配線し、条件違反は fail-closed で拒否する。

## 第2版 追加探索エビデンス（要約）

- `grep`
  - `RuntimeBuilder.cpp` で `createForBuilder` → `assertMutable` → `freeze` の構築封印フローを確認。
  - `AudioEngine.h` で `makeRuntimePublishView` / `makeRuntimeReadView` と `observeWorldHandle` の集中点を確認。
  - 実行系（`AudioBlock.cpp`, `BlockDouble.cpp`, `Snapshot.cpp`, `Timer.cpp`）で `runtimePublishView.transition.current/next` 参照を確認。
  - `AudioEngine.StateIO.cpp` と `AudioEngine.Parameters.cpp` で state I/O と parameter atomic の直接 read/write を確認。
  - `.github` で verifier registry / execution layers / expiry guard を確認。
- Serena
  - `BuilderToken`, `createForBuilder`, `assertMutable`, `freeze`, `observeWorldHandle`, `publishAndSwap(nullptr)`, `getCurrent()` の出現位置を再確認。
- CodeGraph
  - `RuntimeBuilder.cpp`, `ISRRuntimePublicationCoordinator.cpp`, `RuntimePublicationCoordinator.h`, `AudioEngine.StateIO.cpp` の構造を取得し、改修対象関数の所在を再確認。

## 追加レビュー反映判定（2026-06-01 第3版 / ID33-ID36）

### 採用（妥当・第3版）

- ID33: Runtime Publication Read Consistency Contract
  - 根拠: `RuntimePublicationCoordinator.h` に `observePublishedWorld` / `observeWorldHandle` が併存し、`AudioEngine.h` 側では `makeRuntimePublishView` / `makeRuntimeReadView` が直接 `observeWorldHandle(runtimeStore)` を呼ぶ。
  - 判定理由: Publication Contract / Read Contract / Projection Contract が分離したままで、`publish -> consume -> read` の一本化が未完。

- ID35: Runtime Rebuild Trigger Authority
  - 根拠: 実装上の rebuild 中核は `scheduleRebuild/dispatchRebuild` ではなく、`submitRebuildIntent` / `handleAsyncUpdate` / `rebuildThreadLoop`（`AudioEngine.RebuildDispatch.cpp`）。
  - 補助根拠: `AudioEngine.Parameters.cpp`, `AudioEngine.UIEvents.cpp`, `AudioEngine.Timer.cpp`, `AudioEngine.StateIO.cpp` で `submitRebuildIntent(...)` 呼び出しが散在。
  - 判定理由: semantic変更点と rebuild trigger の完全対応を機械検証していないため、将来の「再構築されないパラメータ」リスクが残る。

- ID36: Runtime Publish→Retire Full Lifecycle Contract
  - 根拠: `ISRRuntimePublicationCoordinator.cpp` に backlog/pressure/shutdown 管理（`retireBacklogCount_`, `publicationBacklogCount_`, `reclaimInFlightCount_`, `deferredRetireResidencyCount_`, `isFullyDrained`, `requestShutdown`, `markShutdownComplete`）はある。
  - 判定理由: ただし Construct/Seal/Publish/Observe/Consume/Retire/Reclaim/Destroy を単一状態機械として検証する契約は未整備。

### 条件付き採用（第3版・既存実装を拡張）

- ID34: RuntimeWorld Completeness Verification
  - 根拠: `validateSemanticCompleteness`、`validateAuthorityInventoryAgainstDescriptors`、`SemanticDependencyGraphVerifier`、`DeterministicBuildVerifier` は既存。
  - 不足点: しかし「Builder入力 -> RuntimeWorld field -> SemanticHash」到達性を1本で検査する専用 verifier は未実装。
  - 判定理由: 新規 `RuntimeWorldCompletenessVerifier` の追加は妥当だが、既存 verifier 群の拡張として実装するのが安全。

## 追補 P11-P14（第3版レビュー対応の具体改修）

### P11. ID33 Runtime Publication Read Consistency Contract

- 対象ファイル: `src/core/RuntimePublicationCoordinator.h`, `src/core/RuntimeStore.h`, `src/audioengine/AudioEngine.h`, `src/audioengine/ISRRuntimePublicationCoordinator.h`, `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
- 対象関数: `observePublishedWorld`, `observeWorldHandle`, `makeRuntimePublishView`, `makeRuntimeReadView`, `getCurrent`（暫定）, 追加 `consume`
- 対象変数: `RuntimeStore::current`, `RuntimePublishView::runtimeWorld`, `RuntimePublishView::transition`, `swapPending_`
- 改修方針:
  - 外部公開の read API を `consume()` 系に統一。
  - `observePublishedWorld` / `observeWorldHandle` は coordinator 内部ヘルパへ降格。
  - `makeRuntimePublishView` / `makeRuntimeReadView` は consume 結果を受ける projection 専用関数へ縮約。

### P12. ID34 RuntimeWorld Completeness Verification

- 対象ファイル: `src/audioengine/ISRRuntimeSemanticSchema.h`, `src/audioengine/RuntimeBuilder.cpp`, `src/tests/RuntimeSemanticSchemaValidationTests.cpp`, `src/audioengine/AudioEngine.Commit.cpp`
- 対象関数: `buildRuntimePublishWorld`, `validateSemanticCompleteness`, 追加 `RuntimeWorldCompletenessVerifier`
- 対象変数: `BuildInput` 各フィールド, `RuntimeWorld` の authoritative fields, `RuntimeSemanticHash::*`, `kFieldDescriptors`
- 改修方針:
  - `BuildInput -> RuntimeWorld field` の到達マップを静的記述し、未到達を fail-closed。
  - `RuntimeWorld field -> RuntimeSemanticHash` の到達マップも同時検証。
  - `RuntimeSemanticSchemaValidationTests.cpp` に未到達ケース（故意の欠落）で失敗する回帰テストを追加。

### P13. ID35 Runtime Rebuild Trigger Authority

- 対象ファイル: `src/audioengine/AudioEngine.Parameters.cpp`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`, `src/audioengine/AudioEngine.UIEvents.cpp`, `src/audioengine/AudioEngine.Timer.cpp`, `src/audioengine/AudioEngine.StateIO.cpp`, `src/audioengine/RuntimeBuilder.cpp`
- 対象関数: `submitRebuildIntent`, `requestRebuild`, `handleAsyncUpdate`, `rebuildThreadLoop`, `endBulkParameterRestore`, `requestLoadState`, `convolverParamsChanged`
- 対象変数: `RebuildTelemetryReason::*`, `pendingRebuildKinds`, `needsRebuild`, `rebuildGeneration`
- 改修方針:
  - semantic field 単位で「変更 -> `submitRebuildIntent` 到達」を宣言する `RebuildTriggerCoverageVerifier` を追加。
  - `scheduleRebuild/dispatchRebuild` という名称依存ではなく、現実装の主経路（submit/async/loop）を authority 経路として固定。
  - 100% coverage 未達時は PR tier で fail-closed。

### P14. ID36 Runtime Publish→Retire Full Lifecycle Contract

- 対象ファイル: `src/core/RuntimeStore.h`, `src/core/RuntimePublicationCoordinator.h`, `src/audioengine/ISRRuntimePublicationCoordinator.cpp`, `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`, `src/audioengine/AudioEngine.CtorDtor.cpp`
- 対象関数: `publishAndSwap`, `publishState`, `retire`, `isFullyDrained`, `requestShutdown`, `markShutdownComplete`, 追加 `LifecycleStateMachineVerifier`
- 対象変数: `retireBacklogCount_`, `publicationBacklogCount_`, `reclaimInFlightCount_`, `deferredRetireResidencyCount_`, `state_`, `swapPending_`
- 改修方針:
  - Lifecycle状態を `Constructed/Sealed/Published/Observed/Consumed/Retired/Reclaimed/Destroyed` に正規化。
  - 非許可遷移（例: 未Sealed publish, shutdown後 publish, reclaim前 destroy）を verifier で拒否。
  - `isFullyDrained` の条件と shutdown 完了条件を状態機械上の終端条件として結び付ける。

## 第3版 追加探索エビデンス（要約）

- `grep`
  - Read契約: `observePublishedWorld`, `observeWorldHandle`, `makeRuntimePublishView`, `makeRuntimeReadView` の併存を確認。
  - Rebuild契約: `submitRebuildIntent`, `handleAsyncUpdate`, `rebuildThreadLoop` が中核で、`Parameters/UIEvents/Timer/StateIO` から流入することを確認。
  - Lifecycle契約: `requestShutdown`, `markShutdownComplete`, `isFullyDrained`, backlog/reclaim系カウンタ群を確認。
  - Completeness契約: `validateSemanticCompleteness`, `validateAuthorityInventoryAgainstDescriptors`, `SemanticDependencyGraphVerifier`, `DeterministicBuildVerifier` の既存実装を確認。
- Serena
  - `submitRebuildIntent` への流入点（`endBulkParameterRestore`, `requestLoadState`, `convolverParamsChanged` など）と read API 群の位置を再確認。
- CodeGraph
  - `AudioEngine.RebuildDispatch.cpp`, `RuntimeStore.h`, `RuntimePublicationCoordinator.h`, `ISRRuntimeSemanticSchema.h`, `AudioEngine.Parameters.cpp` の構造を取得し、改修対象の関数境界を特定。

## 追加レビュー反映判定（2026-06-01 第4版 / ID37-ID40）

### 採用（妥当・第4版）

- ID37: Runtime Semantic Coverage Closure
  - 根拠: `submitRebuildIntent` は中核経路として存在し、`captureRuntimeBuildSnapshot` / `RuntimeBuildSnapshot::buildInput` / `RuntimeBuilder::buildRuntimePublishWorld` への経路も存在する。
  - 不足点: ただし「semantic source から SemanticHash までの全経路到達性」を一体で検証する verifier は未実装。

- ID38: RuntimeWorld Projection Consistency
  - 根拠: `makeRuntimePublishView` と `makeRuntimeReadView` は別関数で、双方が `observeWorldHandle(runtimeStore)` を独立に呼ぶ構造。
  - 補助根拠: `projectionFreshness` の検査はあるが、World→PublishView→ReadView 全体整合を網羅する専用 verifier は未確認。

### 条件付き採用（第4版・実装あり運用強化）

- ID40: RuntimeWorld Mutation Boundary Freeze
  - 根拠: `ISRSealedObject.h` に `assertMutable` / `freeze` / `isSealedRecursively` があり、`RuntimeBuilder.cpp` でも `assertMutable(); ... freeze();` の順序を実装。`AudioEngine.Commit.cpp` で `isSealedRecursively()` チェックも存在。
  - 不足点: freeze後mutation禁止を「独立 verifier とCIゲート」で常時保証する項目は明示されていない。

### 不採用（第4版・未達指摘は誤り）

- ID39: RuntimeHash Authority Freeze
  - 反証: `AudioEngine.h` の `kFieldDescriptors` で `semanticHash` は `SemanticCategory::Diagnostic`、`kAuthorityInventory` で `RuntimeAuthorityClass::Diagnostic` に既に固定されている。
  - 補助反証: `ISRRuntimeSemanticSchema.h` の `Derived Semantic Non-Persistence` / `NonDeterministicSourcesMustBeDiagnosticOnly` ポリシーとも整合。

## 追補 P15-P18（第4版レビュー対応の具体改修）

### P15. ID37 Semantic Coverage Closure の追加

- 対象ファイル: `src/audioengine/RuntimeBuilder.cpp`, `src/audioengine/RuntimeBuildTypes.h`, `src/audioengine/ISRRuntimeSemanticSchema.h`, `src/audioengine/AudioEngine.Parameters.cpp`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`
- 対象関数: `submitRebuildIntent`, `captureRuntimeBuildSnapshot`, `buildRuntimePublishWorld`, `validateSemanticCompleteness`
- 対象変数: semantic source fields, `BuildInput`, `RuntimeBuildSnapshot::buildInput`, `RuntimeWorld::*`, `RuntimeSemanticHash::*`
- 改修方針:
  - `SemanticCoverageVerifier` を追加し、semantic source→trigger→BuildInput→RuntimeWorld→SemanticHash の到達グラフを定義。
  - 未到達ノード・経路断絶を PR tier で fail-closed。

### P16. ID38 Projection Consistency の追加

- 対象ファイル: `src/audioengine/AudioEngine.h`, `src/core/RuntimePublicationCoordinator.h`, `src/audioengine/AudioEngine.Commit.cpp`, `src/tests/RuntimeSemanticSchemaValidationTests.cpp`
- 対象関数: `makeRuntimePublishView`, `makeRuntimeReadView`, `readAudioRuntimeView`, `readControlRuntimeView`
- 対象変数: `RuntimePublishView::transition`, `RuntimeReadView::runtimePublish`, `RuntimeReadView::runtimeWorld`, `projectionFreshness`
- 改修方針:
  - `ProjectionConsistencyVerifier` を追加し、RuntimeWorldの主要semanticが PublishView/ReadView で欠落・矛盾しないことを検証。
  - `projectionFreshness` と `publication.mappedRuntimeGeneration` の一致を単独条件でなく投影整合条件に昇格。

### P17. ID39 RuntimeHash Authority Freeze（現状維持の固定化）

- 対象ファイル: `src/audioengine/AudioEngine.h`, `src/audioengine/ISRRuntimeSemanticSchema.h`, `src/tests/RuntimeSemanticSchemaValidationTests.cpp`
- 対象関数: `validateRuntimeSemanticSchemaContract`, authority/descriptor validation 群
- 対象変数: `kFieldDescriptors[semanticHash]`, `kAuthorityInventory[semanticHash]`
- 改修方針:
  - 新規未達としては扱わず、既存分類の「回帰防止テスト」だけを追加して固定化する。
  - `semanticHash` が `Diagnostic` 以外へ変更された場合にテスト失敗させる。

### P18. ID40 RuntimeWorld Mutation Boundary Freeze（運用固定）

- 対象ファイル: `src/audioengine/ISRSealedObject.h`, `src/audioengine/RuntimeBuilder.cpp`, `src/audioengine/AudioEngine.Commit.cpp`, `src/tests/RuntimeSemanticSchemaValidationTests.cpp`, `.github/isr-verifier-registry.json`
- 対象関数: `assertMutable`, `freeze`, `isSealedRecursively`, `buildRuntimePublishWorld`
- 対象変数: `sealState_`, `sealViolationCount`, `RuntimeBuildSnapshot::sealed`
- 改修方針:
  - `RuntimeSealVerifier` を追加し、freeze後mutation API 呼び出しを検出したら fail。
  - sealed違反件数（`sealViolationCount`）をテレメトリ化し、閾値>0で CI 不合格にする。

## 優先度再評価（第4版反映後）

- Tier-A（設計破綻防止）
  - `ID1`, `ID2`, `ID5`, `ID7`, `ID21`, `ID25`, `ID33`
- Tier-B（ISR完成条件）
  - `ID3`, `ID6`, `ID12`, `ID13`, `ID36`
- Tier-C（検証完成）
  - `ID10`, `ID17`, `ID18`, `ID19`, `ID29`, `ID34`
- Tier-D（今回追加）
  - `ID37`, `ID38`, `ID40`（`ID39` は既存実装で満たしているため未達管理対象外）

## 第4版 追加探索エビデンス（要約）

- `grep`
  - `submitRebuildIntent` を中心に `Parameters/UIEvents/Timer/StateIO` からの流入を確認。
  - `captureRuntimeBuildSnapshot` と `RuntimeBuildSnapshot::buildInput`、`RuntimeBuilder::buildRuntimePublishWorld` の連結を確認。
  - `makeRuntimePublishView` / `makeRuntimeReadView` の別経路実装と `observeWorldHandle` 依存を確認。
  - `semanticHash` が `Diagnostic` として descriptor/inventory へ登録済みであることを確認。
  - `freeze/assertMutable/isSealedRecursively` の seal境界実装を確認。
- Serena
  - `kFieldDescriptors`, `kAuthorityInventory`, `semanticHash`, `submitRebuildIntent`, `buildRuntimePublishWorld`, `freeze/assertMutable` の位置を再確認。
- CodeGraph
  - `RuntimeBuilder.cpp`, `AudioEngine.RebuildDispatch.cpp`, `ISRSealedObject.h`, `AudioEngine.h`, `ISRRuntimeSemanticSchema.h` の構造を取得し、ID37-40の対象関数境界を確定。

## 追加レビュー反映判定（2026-06-01 第5版 / 優先度見直し）

### 採用（妥当・第5版）

- ID6 / ID33 RuntimeCoordinator consume未完
  - 根拠: `RuntimePublicationCoordinator::getCurrent() const noexcept` が `return nullptr;` の暫定実装。
  - 根拠: `RuntimePublicationCoordinator.h` に `observePublishedWorld` / `observeWorldHandle` が併存。
  - 反映: 「publish→consume→read 一本化未完」の判定を維持。

- ID12 / ID36 Ownership・Lifecycle未完
  - 根拠: `retireBacklogCount_`, `publicationBacklogCount_`, `reclaimInFlightCount_`, `deferredRetireResidencyCount_`, `swapPending_`, `state_` は存在し、`isFullyDrained` / shutdown経路も実装済み。
  - 不足点: `Construct -> Seal -> Publish -> Observe -> Consume -> Retire -> Reclaim -> Destroy` を単一契約として検証する verifier は未確認。
  - 反映: ID36 を妥当として維持。

- ID17 Publication Monotonicity（実装あり・運用配線不足）
  - 根拠: `commit()` 内で `publicationSequenceId_` / `publicationEpoch_` / `mappedRuntimeGeneration_` の単調性チェックを実装。
  - 反映: 「実装あり、tier wiring 強化が必要」の整理を維持。

- ID35 Rebuild Trigger Authority
  - 根拠: setter/UI/timer/state復元経路が `submitRebuildIntent` に収束し、`requestRebuild` / `handleAsyncUpdate` / `rebuildThreadLoop` で rebuild 実行へ接続。
  - 反映: 未達管理は妥当、かつ優先度を引き上げる（下記の優先度改定）。

- ID40 Seal Boundary（機能あり・検証弱い）
  - 根拠: `createForBuilder -> assertMutable -> freeze` 系および `isSealedRecursively()` チェックを確認。
  - 反映: 機能は存在するが verifier 接続不足という条件付き採用を維持。

### 優先度改定（第5版）

- Tier-A（設計破綻防止）
  - `ID1`, `ID2`, `ID5`, `ID7`, `ID21`, `ID25`, `ID33`, **`ID35`**
- Tier-B（ISR完成条件）
  - `ID3`, `ID6`, `ID12`, `ID13`, `ID36`, **`ID37`**
- Tier-C（検証完成）
  - `ID10`, `ID17`, `ID18`, `ID19`, `ID29`, `ID34`
- Tier-D（残課題）
  - `ID38`, `ID40`（`ID39` は既存実装で満たしているため未達管理対象外）

## 追補 P19-P20（第5版レビュー対応の具体改修）

### P19. 最大リスク束（ID1 + ID35 + ID37）を先行閉鎖

- 対象ファイル: `src/audioengine/AudioEngine.Parameters.cpp`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`, `src/audioengine/RuntimeBuildTypes.h`, `src/audioengine/RuntimeBuilder.cpp`, `src/audioengine/ISRRuntimeSemanticSchema.h`
- 対象関数: `applyDefaultsForCurrentMode`, 各setterの rebuild 委譲点, `submitRebuildIntent`, `captureRuntimeBuildSnapshot`, `buildRuntimePublishWorld`, `validateSemanticCompleteness`
- 対象変数: `eqBypassRequested`, `convBypassRequested`, `currentProcessingOrder`, `BuildInput`, `RuntimeBuildSnapshot::buildInput`, `RuntimeSemanticHash::*`
- 改修方針:
  - `SemanticCoverageVerifier` を ID35 の trigger coverage と連動させ、`source -> trigger -> BuildInput -> RuntimeWorld -> SemanticHash` を1つの閉路として fail-closed 検証。
  - coverage 未達は PR tier で即失敗にする。

### P20. 実装順の安全化（第5版）

- 推奨実装順（更新）
  1. `ID1` RuntimeWorld Self-contained
  2. `ID35` Runtime Rebuild Trigger Authority
  3. `ID37` Runtime Semantic Coverage Closure
  4. `ID2` Transition Leakage
  5. `ID7` Topology authority split
  6. `ID33` Publication Read Consistency
  7. `ID3` Crossfade executor-local
- 理由:
  - authority 集約の成立条件は「semantic が build 経路に必ず到達すること」であり、ID35/37 が欠けると ID1 完了判定自体が不安定になるため。

## 第5版 追加探索エビデンス（要約）

- `grep`
  - `ISRRuntimePublicationCoordinator.cpp` で `getCurrent() const noexcept { return nullptr; }` を確認。
  - `RuntimePublicationCoordinator.h` で `observePublishedWorld` / `observeWorldHandle` 併存を確認。
  - coordinator の lifecycle監視フィールド（`retireBacklogCount_` 等）と `isFullyDrained` / shutdown 経路を確認。
  - `submitRebuildIntent` を中心に setter/UI/timer/state復元の流入と `BuildInput` 生成経路を確認。
  - `semanticHash` の Diagnostic 分類、`freeze` / `assertMutable` / `isSealedRecursively` を確認。
- Serena
  - `observe*` 群、`submitRebuildIntent`、`captureRuntimeBuildSnapshot`、`buildRuntimePublishWorld`、`freeze/isSealedRecursively` の位置と関連を再確認。
- CodeGraph
  - `ISRRuntimePublicationCoordinator.cpp` と `AudioEngine.RebuildDispatch.cpp` の構造解析で、monotonic/lifecycle と rebuild authority の関数境界を再確認。

## 追加レビュー反映判定（2026-06-01 第6版 / 管理単位再編）

### 採用（妥当・第6版）

- ID41（新規）: RuntimeWorld Read Authority Singularization
  - 判定: **採用**。
  - 根拠: audio実行系で `runtimePublishView.transition.current/next` を直接参照する経路が複数残存（`AudioEngine.Processing.AudioBlock.cpp`, `AudioEngine.Processing.BlockDouble.cpp`, `AudioEngine.Processing.Snapshot.cpp`, `AudioEngine.Timer.cpp`）。
  - 根拠: `RuntimePublicationCoordinator` の read入口が `observePublishedWorld` / `observeWorldHandle` で分散し、`getCurrent()==nullptr` 暫定も残る。
  - 補足: 既存の `ID2/ID7/ID21/ID33` は「同一未達の下位症状」としてぶら下げ管理する。

- ID42（新規）: RuntimeGraph Projection Purification
  - 判定: **採用**。
  - 根拠: `RuntimeGraph.h` に `eqBypassed`, `convBypassed`, `inputHeadroomGain`, `outputMakeupGain`, `convolverInputTrimGain`, `runtimeUuid`, `fadingRuntimeUuid`, `transitionCurrentRuntimeUuid`, `transitionNextRuntimeUuid` が残存。
  - 根拠: 同ファイルの descriptor 定義で `runtimeUuid`/`fadingRuntimeUuid` が `SemanticCategory::Authority` 扱いとなっており、projection純化の完了条件に未達。

- ID14（再定義）: RuntimeBuilder Runtime Semantic Isolation
  - 判定: **採用（定義強化）**。
  - 根拠: `RuntimeBuilder.cpp` で `AudioEngine::RuntimePublicationCoordinator::observeWorldHandle(engine.runtimeStore)` を参照しており、Builderが runtime mutable 観測可能な構造。
  - 反映: 「依存縮小」ではなく「`Builder semantic input source = BuildInput only`」をDoDに昇格。

- ID3（定義補強）: Crossfade semantic executor-local
  - 判定: **採用（対象明確化）**。
  - 根拠: `refreshCrossfadePreparedSnapshotFromAtomics()` と `consumeCrossfadePreparedSnapshot()` が現存し、`CrossfadePreparedSnapshot` の取得経路が残る。
  - 反映: `CrossfadePreparedSnapshot must not be semantic source.` を明示。

- ID35 + ID37（統合管理）: Semantic Reachability Contract
  - 判定: **採用（管理統合）**。
  - 方針: 実装上は `ID35/ID37` を子IDとして残しつつ、完了判定は上位契約 `Semantic Reachability Contract` で一体評価する。

### 不採用維持（第6版）

- ID39 RuntimeHash Authority Freeze の「新規未達化」
  - 判定: **不採用維持**（`semanticHash` は Diagnostic 固定で妥当）。
  - 対応: 回帰防止テストのみ維持。

## 追補 P21-P24（第6版レビュー対応の具体改修）

### P21. ID41 RuntimeWorld Read Authority Singularization

- 対象ファイル: `src/audioengine/AudioEngine.h`, `src/core/RuntimePublicationCoordinator.h`, `src/audioengine/ISRRuntimePublicationCoordinator.h`, `src/audioengine/ISRRuntimePublicationCoordinator.cpp`, `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`, `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`, `src/audioengine/AudioEngine.Processing.Snapshot.cpp`, `src/audioengine/AudioEngine.Timer.cpp`
- 対象関数: `makeRuntimePublishView`, `makeRuntimeReadView`, `observePublishedWorld`, `observeWorldHandle`, `getCurrent`, `getNextAudioBlock`, `processDouble`
- 対象変数: `runtimePublishView.transition.current`, `runtimePublishView.transition.next`, `RuntimeWorld::topology.hasFadingRuntime`, `RuntimeStore::current`, `swapPending_`
- 改修方針:
  - 公開read APIを `consume()` 単一路へ集約。
  - Audio Thread の semantic branching source を `RuntimeWorld` のみに固定。
  - `transition.*` は executor-local / diagnostic 投影へ降格。

### P22. ID42 RuntimeGraph Projection Purification

- 対象ファイル: `src/audioengine/RuntimeGraph.h`, `src/audioengine/AudioEngine.h`, `src/audioengine/RuntimeBuilder.cpp`, `src/audioengine/ISRRuntimeSemanticSchema.h`, `src/tests/RuntimeSemanticSchemaValidationTests.cpp`
- 対象関数: `makeRuntimeGraphState`, `buildRuntimePublishWorld`, descriptor/inventory validation 群
- 対象変数: `RuntimeGraph::eqBypassed`, `RuntimeGraph::convBypassed`, `RuntimeGraph::inputHeadroomGain`, `RuntimeGraph::outputMakeupGain`, `RuntimeGraph::convolverInputTrimGain`, `RuntimeGraph::runtimeUuid`, `RuntimeGraph::fadingRuntimeUuid`, `RuntimeGraph::transitionCurrentRuntimeUuid`, `RuntimeGraph::transitionNextRuntimeUuid`
- 改修方針:
  - authoritative semantic を `RuntimeWorld` 側へ移送し、`RuntimeGraph` は projection/diagnostic のみ保持。
  - `RuntimeGraph` descriptor を `Authority` から `Derived/Diagnostic` へ再分類（必要なもののみ残存）。
  - `RuntimeGraph contains no authoritative semantic.` をテストで固定。

### P23. ID14 RuntimeBuilder Runtime Semantic Isolation（DoD強化）

- 対象ファイル: `src/audioengine/RuntimeBuilder.h`, `src/audioengine/RuntimeBuilder.cpp`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`, `src/audioengine/RuntimeBuildTypes.h`
- 対象関数: `RuntimeBuilder::RuntimeBuilder`, `buildRuntimePublishWorld`, `build(const BuildInput&)`, `captureRuntimeBuildSnapshot`
- 対象変数: `RuntimeBuilder::engine`, `engine.runtimeStore`, `BuildInput`, `RuntimeBuildSnapshot::buildInput`
- 改修方針:
  - Builder本体から `runtimeStore` / transition / atomic 直接観測を禁止。
  - semantic入力は `BuildInput` のみ許可。
  - `RuntimeBuilder cannot observe runtime mutable state.` を verifier化して fail-closed。

### P24. Semantic Reachability Contract（ID35+ID37統合）

- 対象ファイル: `src/audioengine/AudioEngine.Parameters.cpp`, `src/audioengine/AudioEngine.UIEvents.cpp`, `src/audioengine/AudioEngine.Timer.cpp`, `src/audioengine/AudioEngine.StateIO.cpp`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`, `src/audioengine/RuntimeBuildTypes.h`, `src/audioengine/RuntimeBuilder.cpp`, `src/audioengine/ISRRuntimeSemanticSchema.h`, `src/tests/RuntimeSemanticSchemaValidationTests.cpp`
- 対象関数: `submitRebuildIntent`, `requestRebuild`, `handleAsyncUpdate`, `rebuildThreadLoop`, `captureRuntimeBuildSnapshot`, `buildRuntimePublishWorld`, `validateSemanticCompleteness`
- 対象変数: `pendingRebuildKinds`, `needsRebuild`, `rebuildGeneration`, `RuntimeBuildSnapshot::buildInput`, semantic source fields, `RuntimeSemanticHash::*`
- 改修方針:
  - `semantic source -> rebuild trigger -> BuildInput -> RuntimeWorld -> SemanticHash` を単一契約として検証。
  - `RebuildTriggerCoverageVerifier` + `SemanticCoverageVerifier` を統合した到達性検証で未到達をPR tierで fail-closed。
  - `ID35`（trigger coverage）と`ID37`（semantic closure）は子項目としてのみ残し、完了判定は上位契約で一本化。

## 優先度再編（第6版）

- Tier-A（最優先）
  - `ID1`, `ID41`, `ID35+ID37(統合: Semantic Reachability Contract)`
- Tier-B（高優先）
  - `ID2`, `ID7`, `ID33`, `ID14(再定義)`, `ID42`
- Tier-C（実装境界の安定化）
  - `ID3`（CrossfadePreparedSnapshot 非semantic source 固定）, `ID36`, `ID38`, `ID40`
- Tier-D（運用固定）
  - `ID10`, `ID17`, `ID18`, `ID19`, `ID29`, `ID34`, `ID39(回帰防止のみ)`

## 第6版 推奨実装順（再編）

1. `ID1` RuntimeWorld Self-contained
2. `ID41` RuntimeWorld Read Authority Singularization
3. `ID35+ID37` Semantic Reachability Contract
4. `ID14` RuntimeBuilder Runtime Semantic Isolation
5. `ID42` RuntimeGraph Projection Purification
6. `ID3` Crossfade executor-local（補強DoD適用）
7. `ID2` / `ID7` / `ID33` の残存枝刈り（ID41配下）

## 第6版 追加探索エビデンス（要約）

- `grep`
  - `ISRRuntimePublicationCoordinator.cpp`: `getCurrent() const noexcept { return nullptr; }`。
  - `RuntimePublicationCoordinator.h`: `observePublishedWorld` / `observeWorldHandle` 併存、`clearPublishedRuntimeSnapshotsNonRt` の `publishAndSwap(nullptr)`。
  - `AudioEngine.RebuildDispatch.cpp`: `submitRebuildIntent`, `captureRuntimeBuildSnapshot`。
  - `RuntimeGraph.h`: `eqBypassed`, `convBypassed`, gain群, UUID群が残存し、descriptorで authority 扱いの項目あり。
  - `AudioEngine.h`: `refreshCrossfadePreparedSnapshotFromAtomics`, `consumeCrossfadePreparedSnapshot` が現存。
- Serena
  - `AudioEngine.*` 実行経路で `transition.current/next` 参照、および `RuntimeBuilder.cpp` で `observeWorldHandle(engine.runtimeStore)` 参照を再確認。
- CodeGraph
  - 構造取得APIは現セッションで無効化されていたため、`find_callers(submitRebuildIntent)` と `global_search` で補完。
  - `submitRebuildIntent` 呼び出し元として `Parameters/UIEvents/Timer/StateIO/RebuildDispatch` 系が検出され、中核経路の妥当性を確認。

## 追加レビュー反映判定（2026-06-01 第7版 / 契約クラスタ再編）

### 採用（妥当・第7版）

- 妥当性評価（85-90%）
  - `ID1`, `ID2`, `ID3`, `ID6`, `ID14`, `ID35`, `ID37`, `ID41` は現行コード根拠と整合するため採用維持。
  - 例: `RuntimePublicationCoordinator::getCurrent()` は現状も `return nullptr;`。
  - 例: Audio Thread 側で `runtimePublishView.transition.current/next` 直接参照が残存。

- Cluster-A（Read-side authority 非収束）の親子化
  - 親契約: `ID41 RuntimeWorld Read Authority Singularization`
  - 子項目: `ID2`, `ID7`, `ID21`, `ID25`, `ID33`
  - 反映: 上記は独立完了ではなく、`ID41` 完了条件への従属として扱う。

- Cluster-B（Reachability）の統合維持
  - 親契約: `Semantic Reachability Contract`
  - 子項目: `ID35`（Trigger Authority）, `ID37`（Coverage Closure）
  - 反映: 第6版方針を継続。

- Cluster-C（Publication Lifecycle）の統合
  - 親契約: `Publication Lifecycle Contract`
  - 子項目: `ID12`, `ID13`, `ID36`
  - 反映: ownership / shutdown例外 / lifecycle を単一状態機械として一体管理。

- 新規追加: ID43 Coordinator Duality Removal
  - 判定: **採用**。
  - 根拠: `src/core/RuntimePublicationCoordinator.h` 側に `observePublishedWorld/observeWorldHandle/publishState`、`src/audioengine/ISRRuntimePublicationCoordinator.cpp` 側に `commit/retire/state machine` が併存。
  - 反映: coordinator責務の二重化を独立未達として管理。

### 補正採用（第7版）

- ID42 RuntimeGraph Projection Purification（説明修正）
  - 旧説明（第6版）: 「RuntimeGraph が Authority を持つ」
  - 補正: `AudioEngine.h` の runtime state descriptor/inventory では `graph` と `engine` は `Derived`。
  - 現行未達の本質:
    - 「RuntimeGraphが定義上Authorityかどうか」ではなく、
    - 「RuntimeGraphが**実装上の意味決定経路**として残存し得ること」。
  - よってID42は維持しつつ、DoDを「権威分類」から「実参照経路の浄化」に補正する。

## 追補 P25-P28（第7版レビュー対応の具体改修）

### P25. Cluster-A 親子化（ID41配下の完了判定統一）

- 対象ファイル: `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`, `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`, `src/audioengine/AudioEngine.Processing.Snapshot.cpp`, `src/audioengine/AudioEngine.Timer.cpp`, `src/core/RuntimePublicationCoordinator.h`
- 対象関数: `makeRuntimePublishView`, `makeRuntimeReadView`, `getNextAudioBlock`, `processDouble`, `observePublishedWorld`, `observeWorldHandle`
- 対象変数: `runtimePublishView.transition.current`, `runtimePublishView.transition.next`, `RuntimeWorld::topology.hasFadingRuntime`
- 改修方針:
  - `ID2/7/21/25/33` を `ID41` の子項目として定義し、個別完了を禁止。
  - DoDは `Audio Thread semantic branching source = RuntimeWorld only` に一本化。

### P26. Cluster-C 統合（Publication Lifecycle Contract）

- 対象ファイル: `src/core/RuntimePublicationCoordinator.h`, `src/core/RuntimeStore.h`, `src/audioengine/ISRRuntimePublicationCoordinator.cpp`, `src/audioengine/AudioEngine.Commit.cpp`, `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`, `src/audioengine/AudioEngine.CtorDtor.cpp`
- 対象関数: `publishState`, `clearPublishedRuntimeSnapshotsNonRt`, `publishAndSwap`, `commit`, `retire`, `requestShutdown`, `markShutdownComplete`, `isFullyDrained`
- 対象変数: `state_`, `swapPending_`, `retireBacklogCount_`, `publicationBacklogCount_`, `reclaimInFlightCount_`, `deferredRetireResidencyCount_`
- 改修方針:
  - `ID12/ID13/ID36` を単一状態機械（publish/retire/shutdown/drain）へ統合。
  - shutdown専用 `publishAndSwap(nullptr)` は lifecycle契約上の例外遷移として明文化。

### P27. ID42 補正（Projection Purification の焦点修正）

- 対象ファイル: `src/audioengine/AudioEngine.h`, `src/audioengine/RuntimeGraph.h`, `src/audioengine/RuntimeBuilder.cpp`, `src/tests/RuntimeSemanticSchemaValidationTests.cpp`
- 対象関数: `makeRuntimeGraphState`, `buildRuntimePublishWorld`, schema/inventory validation 群
- 対象変数: `RuntimeState.graph`, `RuntimeState.engine`, `RuntimeGraph::*`, `kFieldDescriptors`, `kAuthorityInventory`
- 改修方針:
  - descriptor分類の修正ではなく、`RuntimeGraph` を semantic branching source に使う経路を禁止。
  - DoD: `RuntimeGraph contains no authoritative decision path.`

### P28. ID43 Coordinator Duality Removal

- 対象ファイル: `src/core/RuntimePublicationCoordinator.h`, `src/audioengine/ISRRuntimePublicationCoordinator.h`, `src/audioengine/ISRRuntimePublicationCoordinator.cpp`, `src/audioengine/AudioEngine.h`, `src/tests/RuntimePublicationCoordinatorTests.cpp`, `src/tests/ISRSemanticValidationTests.cpp`
- 対象関数: `publishState`, `observePublishedWorld`, `observeWorldHandle`, `commit`, `retire`, `commitRuntimePublication`
- 対象変数: `RuntimeStore::current`, `state_`, `swapPending_`, publication sequence/epoch counters
- 改修方針:
  - template coordinator と ISR coordinator の責務境界を再定義し、外部公開契約を単一 coordinator API に集約。
  - 二重経路（publish/read/lifecycle）の同時存在をテストで禁止。

## 第7版 推奨実装順（更新）

1. `ID1` RuntimeWorld Self-contained
2. `ID35+ID37` Semantic Reachability Contract
3. `ID14` RuntimeBuilder Runtime Semantic Isolation
4. `ID41` RuntimeWorld Read Authority Singularization（Cluster-A親）
5. `ID3` Crossfade Executor-Local Contract
6. `ID42` RuntimeGraph Projection Purification（補正DoD）
7. `Publication Lifecycle Contract`（`ID12+ID13+ID36`）

## 第7版 親契約7本（管理単位）

1. RuntimeWorld Self-contained
2. RuntimeWorld Read Authority Singularization
3. Semantic Reachability Contract
4. RuntimeBuilder Semantic Isolation
5. Crossfade Executor-Local Contract
6. Publication Lifecycle Contract
7. Governance / Verifier Wiring Contract

## 第7版 追加探索エビデンス（要約）

- `grep`
  - `AudioEngine.h`: `kFieldDescriptors/kAuthorityInventory` で `graph` / `engine` は `Derived`。
  - `RuntimeGraph.h`: `runtimeUuid/fadingRuntimeUuid/generation` など Authority 記述が残る。
  - `RuntimePublicationCoordinator.h`: `observePublishedWorld/observeWorldHandle/publishState` と `publishAndSwap(nullptr)`。
  - `ISRRuntimePublicationCoordinator.cpp`: `commit/retire/requestShutdown/markShutdownComplete/isFullyDrained` と state machine 変数群。
  - Audio実行系: `runtimePublishView.transition.current/next` の直接参照。
- Serena
  - `AudioEngine.h` で `graph`/`engine` の `Derived` 分類、`RuntimePublicationCoordinator.h` の observe/publish API、`ISRRuntimePublicationCoordinator.cpp` の lifecycle state machine を再確認。
- CodeGraph
  - `find_callers(publishState)` で `AudioEngine.Commit.cpp` など publish系呼び出しを確認。
  - `find_callers(commit)` で `AudioEngine.Commit.cpp`/`AudioEngine.h`/テスト群からの呼び出しを確認。
  - `find_callers(submitRebuildIntent)` で `Parameters/UIEvents/Timer/StateIO/RebuildDispatch` からの流入を再確認。
