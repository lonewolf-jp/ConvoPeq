# ISR Bridge Runtime 影響範囲シンボル判定・Data Flow（2026-05-30）

## 1) 主要シンボル判定（修正/修正不要）

| ファイル | シンボル | 判定 | 根拠 |
| --- | --- | --- | --- |
| `src/audioengine/AudioEngine.h` | `runtimeWorldIdGenerator_` | 修正済み/適合 | worldId発番を generator 単一化。 |
| `src/audioengine/AudioEngine.h` | `runtimeGenerationGenerator_` | 修正済み/適合 | generation発番を generator 単一化。 |
| `src/audioengine/AudioEngine.h` | `publicationSequenceCounter_` | 修正済み/適合 | publicationSequence を単調カウンタ化。 |
| `src/audioengine/AudioEngine.h` | `buildRuntimePublishWorld` | 修正済み/適合 | metadata/publication semantic を集約設定。 |
| `src/audioengine/AudioEngine.Commit.cpp` | `runPublicationPrecheckNonRt` | 修正済み/適合 | schema/sequence/generation rollback fail-closed。 |
| `src/audioengine/AudioEngine.Commit.cpp` | `onRuntimePublishedNonRt` | 修正済み/適合 | committed generation/sequence 更新。 |
| `src/core/RuntimePublicationCoordinator.h` | `observeWorldHandle` | 修正済み/適合 | `[[nodiscard]]` 適用。 |
| `src/audioengine/ISRRetire.*` | `overflow/dropped` metrics | 修正済み/適合 | overflow/drop 可観測化追加。 |
| `src/audioengine/RuntimeGraph.h` | 実行進捗状態 | 修正済み/適合 | crossfade/latency進捗を分離済み。 |

## 2) Data Flow 列挙

### publish

1. `AudioEngine::commitNewDSP` -> `makeRuntimePublicationCoordinator().publishState(...)`
2. `RuntimePublicationCoordinator::publishState` -> `buildRuntimePublishWorld`
3. `runPublicationPrecheckNonRt` 通過時のみ `RuntimeStore::publishAndSwap`
4. `onRuntimePublishedNonRt` で committed generation/sequence を確定

### observe

1. `RuntimePublicationCoordinator::observeWorldHandle(runtimeStore)`
2. `AudioEngine::makeRuntimePublishView/makeRuntimeReadView` で `const RuntimeWorld*` を借用
3. RT処理は world->graph の読み取りのみ（所有権移譲なし）

### retire

1. publish swap 後 old world を `willRetireRuntimeNonRt` へ通知
2. `onRuntimeRetiredNonRt` で retire intent を emit/dequeue/ack
3. retire runtime ex による enqueue/settle/reclaim へ移送

### generation

1. `runtimeGenerationGenerator_.next()` が唯一採番
2. `buildRuntimePublishWorld` で `world.generation` と semantic に反映
3. `runPublicationPrecheckNonRt` が monotonic check

### worldId

1. `runtimeWorldIdGenerator_.next()` が唯一発番
2. `buildRuntimePublishWorld` で `world.worldId` に設定

### publicationSequence

1. `publicationSequenceCounter_` fetch_add で採番
2. `buildRuntimePublishWorld` で metadata/publication に同期設定
3. precheck で previous/committed との単調性検証
4. publish成功時のみ `lastCommittedPublicationSequence_` 更新

## 3) 実装後再監査（要点）

- `observeWorldHandle` 利用箇所は `AudioEngine` 周辺の限定箇所へ収束。
- legacy候補シンボル（`publishLegacy` など）は未検出。
- worldId/generation/publicationSequence の writer は現行実装で単一化を維持。
