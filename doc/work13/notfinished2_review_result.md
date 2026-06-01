# notfinished2.md 検証結果レポート

## 結論

`doc/work13/notfinished2.md` のレビューは、**コア実装の未達指摘については概ね妥当**です。特に以下は現行ソースで確認できる未達です。

- RuntimeWorld の self-contained 化未完
- Transition / Crossfade の executor-local 化未完
- generation / identity / timeline の単一化未完
- RuntimeBuilder の `AudioEngine&` 依存残存
- transition.current / transition.next の直接参照残存

一方で、**governance / nightly / release / fail-closed / manifest / soak について「不在」と断定している箇所は過大評価または誤認**が含まれます。これらは `.github` 配下の policy / workflow / script で実装・運用されています。

---

## 検証方法

- `grep` による広域検索
- Serena の pattern search による対象関数の局所確認
- CodeGraph のインデックス更新後、依存関係と関連シンボルを確認

---

## 妥当と判定した未達項目

### 1. RuntimeWorld Self-contained 化未達

**妥当**。

`AudioEngine::applyDefaultsForCurrentMode()` が以下を直接読んでいます。

- `eqBypassRequested`
- `convBypassRequested`
- `currentProcessingOrder`

その結果として

- `inputHeadroomDb`
- `outputMakeupDb`
- `convolverInputTrimDb`

を決定しています。

関連ファイル:

- `src/audioengine/AudioEngine.Parameters.cpp`
- `src/audioengine/AudioEngine.h`
- `src/audioengine/RuntimeBuilder.cpp`

---

### 2. Transition Semantic Leakage

**妥当**。

Audio thread 側の処理が `runtimePublishView.transition.current` / `next` を直接参照して DSP を選択しています。

関連ファイル:

- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
- `src/audioengine/AudioEngine.Timer.cpp`
- `src/audioengine/AudioEngine.h`

---

### 3. Legacy Build Generation の残存

**妥当**。

以下が並立しています。

- `RuntimeBuildSnapshot.generation`
- `RuntimeGraph.generation`
- `world.generation`
- `world.runtimeVersion`
- `world.publication.sequenceId`
- `world.publication.epoch`

関連ファイル:

- `src/audioengine/RuntimeBuildTypes.h`
- `src/audioengine/RuntimeGraph.h`
- `src/audioengine/AudioEngine.RebuildDispatch.cpp`
- `src/audioengine/RuntimeBuilder.cpp`

---

### 4. Crossfade Semantic の完全分離未達

**妥当**。

`latencyDelayOld / latencyDelayNew / dspCrossfadePending / dspCrossfadeUseDryAsOld / firstIrDryCrossfadePending / dspCrossfadeStartDelayBlocks / dspCrossfadeDryHoldSamples` が runtime 構築時に注入されています。

関連ファイル:

- `src/audioengine/AudioEngine.h`
- `src/audioengine/RuntimeTransition.h`
- `src/audioengine/RuntimeBuilder.cpp`
- `src/audioengine/AudioEngine.Commit.cpp`

---

### 5. Runtime Coordinator の `consume()` 完成形未達

**妥当**。

現状は `observeWorldHandle()` が主経路で、`RuntimePublicationCoordinator::getCurrent()` は `nullptr` を返しています。

関連ファイル:

- `src/core/RuntimePublicationCoordinator.h`
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
- `src/audioengine/AudioEngine.h`

---

### 6. RuntimeTopologyAuthoritySplit / Runtime Activity 重複

**妥当**。

`topology.hasFadingRuntime` と `transition.current/next` が並存し、意図の源が重複しています。

関連ファイル:

- `src/audioengine/AudioEngine.h`
- `src/audioengine/RuntimeGraph.h`
- `src/audioengine/RuntimeTransition.h`
- `src/audioengine/AudioEngine.Timer.cpp`

---

### 7. Deterministic Build の完全証明不足

**部分妥当**。

契約・テストはありますが、`RuntimeBuilder` が `AudioEngine&` に強く依存しており、完全に純粋な semantic builder にはまだ到達していません。

関連ファイル:

- `src/audioengine/RuntimeBuilder.h`
- `src/audioengine/RuntimeBuilder.cpp`
- `src/audioengine/ISRRuntimeSemanticSchema.h`
- `src/tests/RuntimeSemanticSchemaValidationTests.cpp`

---

## Governance / Soak / Manifest についての修正点

### 不正確だった指摘

以下は「未確認」や「不在」と断定されていますが、実際には存在します。

- Nightly / Release の tiering
- Fail-closed governance
- Soak validation infrastructure
- Legacy manifest / expiry system
- Shadow compare の運用層
- Publication monotonicity の検証

### 確認できた実装

- `.github/isr-verifier-registry.json`
- `.github/isr-validator-tiering-policy.json`
- `.github/isr-8_1-close-policy.json`
- `.github/isr-legacy-temporary.json`
- `.github/workflows/isr-verification.yml`
- `.github/scripts/isr-verify-governance-registries.ps1`
- `.github/scripts/isr-run-tiered-verification.ps1`
- `.github/scripts/isr-verify-shadow-compare-cadence.ps1`
- `.github/scripts/isr-verify-shadow-compare-coverage.ps1`
- `.github/scripts/isr-verify-soak-governance.ps1`

---

## 個別に確認できた実装例

### Publication monotonicity / rollback reject

- `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
- `src/tests/RuntimePublicationCoordinatorTests.cpp`

### Shadow compare contract

- `src/tests/ShadowCompareContractTests.cpp`
- `src/tests/RuntimeSemanticSchemaValidationTests.cpp`

### Authority / ABA / ownership 系の契約テーブル

- `src/audioengine/ISRRuntimeSemanticSchema.h`
- `.github/isr-verifier-registry.json`
- `.github/scripts/isr-verify-aba-hazard.ps1`

---

## 総合評価

`notfinished2.md` は、**runtime semantic の本体側の未達を指摘する文書としては有効**です。特に「意味状態の単一化がまだ終わっていない」という主張は、現行コードの実態と一致します。

ただし、**運用・統治・検証基盤が存在するにもかかわらず『不在』と断じている箇所は修正が必要**です。

つまり、このレビューは

- **実装本体の未達の指摘**: 概ね正しい
- **governance / CI / soak / manifest の不在主張**: 過大評価あり

という評価になります。

---

## 補足

必要であれば次に、このレポートを元にして

1. Critical / High / Medium に再分類
2. 修正対象ファイルごとの一覧化
3. 実装着手順のタスク分解

まで `doc/work13/` に追記できます。
