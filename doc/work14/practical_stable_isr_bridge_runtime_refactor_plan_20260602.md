# Practical Stable ISR Bridge Runtime 改修計画（2026-06-02）

- 対象根拠: `doc/work14/notfinished3.md`、`doc/work14/notfinished3_validation_20260602.md`
- 目的: 監査で妥当と判定された論点（特に `#5/#7/#16/#17`、次点 `#18/#19`、継続監査 `#21/#4`）を、実装可能な改修タスクへ分解する。
- 探索手段: `grep_search` + `mcp_oraios_serena_search_for_pattern` + CodeGraph（`shell: CodeGraph Full Index` 実行後）。

---

## 0. 探索結果サマリ（拾い漏れ抑止）

### 0.1 Publication API (`publishState`) 系の主要ヒット

- `src/core/RuntimePublicationCoordinator.h`
  - `RuntimePublicationCoordinator::publishState(...)`
- `src/audioengine/AudioEngine.h`
  - `makeRuntimePublicationCoordinator()`
  - `publishRuntimeStateNonRt(DSPCore* current, DSPCore* next, TransitionPolicy, double, bool, const RuntimeBuildSnapshot*)`
- `src/audioengine/AudioEngine.Commit.cpp`
  - `applyRuntimeCommitFromIntent(...)` 内の `publishRuntimeStateNonRt(...)` 呼び出し
- 呼び出し元群
  - `src/audioengine/AudioEngine.Commit.cpp`
  - `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`
  - `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
  - `src/audioengine/AudioEngine.Timer.cpp`
- テスト群
  - `src/tests/PartialPublicationRejectTests.cpp`
  - `src/tests/RuntimePublicationCoordinatorTests.cpp`

### 0.2 `#16/#17`（意味重複）関連ヒット

- `src/audioengine/ISRRuntimeSemanticSchema.h`
  - `struct ExecutionSemantic`
    - `transitionActive`
    - `crossfadeStartDelayBlocks`
    - `crossfadeDryHoldSamples`
  - `struct SchedulingSemantic`
    - `transitionActive`
    - `crossfadeStartDelayBlocks`
    - `crossfadeDryHoldSamples`
  - `struct GenerationSemantic::activationEpoch`
  - `struct TimingSemantic::activationEpoch`
- `src/audioengine/RuntimeBuilder.cpp`
  - `worldOwner->generationSemantic.activationEpoch = ...`
  - `worldOwner->timing.activationEpoch = ...`

### 0.3 `#18/#19`（Hash/Equivalence）関連ヒット

- `src/audioengine/ISRRuntimeSemanticSchema.h`
  - `struct RuntimeSemanticHash`
  - `classifySemanticEquivalence(...)`
- `src/audioengine/RuntimeBuilder.cpp`
  - `worldOwner->semanticHash.<...> = ...` の構築
- `src/audioengine/ISRDebugRuntime.cpp` / `ISRDebugRuntime.h`
  - semantic hash 比較・観測

### 0.4 `#21`（Engine Hook依存）関連ヒット

- `src/audioengine/AudioEngine.h`
  - `class RuntimePublicationBridge`
    - `validatePublicationNonRt(...) -> engine_->runPublicationPrecheckNonRt(...)`
    - `didPublishRuntimeNonRt(...) -> engine_->onRuntimePublishedNonRt(...)`
    - `willRetireRuntimeNonRt(...) -> engine_->onRuntimeRetiredNonRt(...)`
- `src/audioengine/AudioEngine.Commit.cpp`
  - `runPublicationPrecheckNonRt(...)`
  - `onRuntimePublishedNonRt(...)`
  - `onRuntimeRetiredNonRt(...)`
- `src/core/RuntimePublicationCoordinator.h`
  - Bridge concept の `validatePublicationNonRt` 呼び出し

### 0.5 既存契約テスト（改修時の回帰防止アンカー）

- `src/tests/ObservePathSingleSourceTests.cpp`（#3 達成寄りの根拠）
- `src/tests/CrossfadeExecutorLocalContractTests.cpp`（#4 継続監査の根拠）
- `src/tests/BuildInputSemanticContractTests.cpp`（publish 経路の semantic 契約）
- `src/tests/PartialPublicationRejectTests.cpp`（fail-closed reject 契約）

---

## 1. 改修方針（優先順位）

### P1（最優先）

1. `#5` Legacy Runtime Semantic Removal
2. `#7` Publication API Zero-Call
3. `#16` Execution/Scheduling 重複解消
4. `#17` activationEpoch 重複解消

### P2（次点監査）

1. `#18` Semantic Hash Coverage 明確化/拡張
2. `#19` Semantic Equivalence 対象範囲明確化

### P3（継続監査）

1. `#21` Engine Hook依存の縮退（要件依存、段階的に）
2. `#4` Crossfade authority 再混線監視（テスト強化）

---

## 2. 具体改修計画（関数・クラス・変数単位）

## 2.1 P1-A: Publication API の単一化（`publishState` 脱却）

### 対象（P1-A）

- `src/core/RuntimePublicationCoordinator.h`
  - `void publishState(...)`
- `src/audioengine/AudioEngine.h`
  - `publishRuntimeStateNonRt(...)`
  - `makeRuntimePublicationCoordinator()`
  - `RuntimePublicationBridge::buildRuntimePublishWorld(...)`
- 呼び出し元
  - `src/audioengine/AudioEngine.Commit.cpp`
  - `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`
  - `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
  - `src/audioengine/AudioEngine.Timer.cpp`

### 改修内容（P1-A）

1. Coordinator API を二層化
   - 追加: `publishWorld(convo::aligned_unique_ptr<RuntimePublishWorld>)`
   - 既存 `publishState(...)` は一時的に wrapper 化し、内部で `buildRuntimePublishWorld(...)` + `publishWorld(...)` を呼ぶ。
2. AudioEngine 側 publish 経路を world 単位へ移行
   - `publishRuntimeStateNonRt(...)` を縮退:
     - 直接 `publishState(...)` 呼び出しをやめ、`RuntimeBuilder` で world を先に生成
     - 生成 world を `publishWorld(...)` へ委譲
3. call site を段階的移行
   - `AudioEngine.Commit.cpp` の主要 publish 呼び出し点から先行移行
   - Prepare/Release/Timer は第二段階で移行
4. 完了条件
   - production code における `publishState(current,next,...)` 呼び出しゼロ（テスト内は移行過渡期許容）

### 影響テスト（P1-A）

- 更新
  - `src/tests/PartialPublicationRejectTests.cpp`
  - `src/tests/RuntimePublicationCoordinatorTests.cpp`
- 追加候補
  - `src/tests/RuntimePublicationWorldApiTests.cpp`（world直接publish契約）

---

## 2.2 P1-B: Semantic 重複の単一 authority 化（#16/#17）

### 対象（P1-B）

- `src/audioengine/ISRRuntimeSemanticSchema.h`
  - `ExecutionSemantic`
  - `SchedulingSemantic`
  - `GenerationSemantic.activationEpoch`
  - `TimingSemantic.activationEpoch`
- `src/audioengine/RuntimeBuilder.cpp`
  - `worldOwner->execution.*`
  - `worldOwner->scheduling.*`
  - `worldOwner->generationSemantic.activationEpoch`
  - `worldOwner->timing.activationEpoch`
- `src/audioengine/AudioEngine.Commit.cpp`
  - `isValidExecutionSemantic(...)` 前提の precheck ロジック

### 改修内容（P1-B）

1. `SchedulingSemantic` の重複3フィールドを Derived 化（または削除）
   - 第一段階（安全）: field残置 + `Derived` 扱い明示 + `ExecutionSemantic` を唯一 authority として参照
   - 第二段階: 参照箇所ゼロ化後に削除
2. `activationEpoch` の単一化
   - Authority を `GenerationSemantic.activationEpoch` に統一
   - `TimingSemantic.activationEpoch` は削除または `Derived` に変更
3. Verifier 追加
   - 例: `SemanticAliasVerifier` を実効化し、
     - `ExecutionSemantic.*` と `SchedulingSemantic.*` の不一致を build-fail
     - `activationEpoch` 二重定義を build-fail
4. RuntimeBuilder 同期
   - `worldOwner->scheduling.* = worldOwner->execution.*` の鏡写しを廃止
   - `worldOwner->timing.activationEpoch` 設定を削除/派生計算化

### 影響テスト（P1-B）

- 更新
  - `src/tests/BuildInputSemanticContractTests.cpp`
- 追加候補
  - `src/tests/SemanticSingleAuthorityTests.cpp`

---

## 2.3 P2-A: Semantic Hash Coverage の仕様化（#18）

### 対象（P2-A）

- `src/audioengine/ISRRuntimeSemanticSchema.h`
  - `RuntimeSemanticHash`
- `src/audioengine/RuntimeBuilder.cpp`
  - `worldOwner->semanticHash.*` 構築
- `src/audioengine/ISRDebugRuntime.cpp`

### 改修内容（P2-A）

1. 「authority class のみ hash 対象」か「全 semantic 対象」かを仕様固定
2. 仕様に合わせて `RuntimeSemanticHash` を再編
   - authority-only 方針なら、authority inventory から hash対象を自動列挙
   - full-schema 方針なら timing/resource/automation 等を追加
3. hash coverage verifier を追加
   - authority inventory と hash項目の差分検知

### 影響テスト（P2-A）

- 追加候補
  - `src/tests/SemanticHashCoverageContractTests.cpp`

---

## 2.4 P2-B: Equivalence 判定を inventory 駆動へ（#19）

### 対象（P2-B）

- `src/audioengine/ISRRuntimeSemanticSchema.h`
  - `classifySemanticEquivalence(...)`

### 改修内容（P2-B）

1. 互換判定の層を明文化
   - `Equivalent`: authority 全一致
   - `Compatible`: 非破壊差分のみ（定義表を別途持つ）
2. 比較ロジックを固定列挙から inventory 駆動へ
   - authority class を基に比較対象を算出
   - 固定8項目比較のハードコードを縮退

### 影響テスト（P2-B）

- 追加候補
  - `src/tests/SemanticEquivalencePolicyTests.cpp`

---

## 2.5 P3-A: Engine Hook依存の段階的縮退（#21）

### 対象（P3-A）

- `src/audioengine/AudioEngine.h`
  - `RuntimePublicationBridge`
- `src/audioengine/AudioEngine.Commit.cpp`
  - `runPublicationPrecheckNonRt`
  - `onRuntimePublishedNonRt`
  - `onRuntimeRetiredNonRt`
- `src/core/RuntimePublicationCoordinator.h`
  - bridge 呼び出しポイント

### 改修内容（P3-A）

1. Publication validator を独立コンポーネント化
   - 新規候補: `src/audioengine/RuntimePublicationValidator.{h,cpp}`
   - `runPublicationPrecheckNonRt` の pure validation 部分を移管
2. Bridge の責務を転換
   - 現状: `engine_` へ委譲
   - 目標: validator + retire policy の compose
3. 段階移行
   - Step1: `runPublicationPrecheckNonRt` 内部で validator 呼び出しに置換
   - Step2: Bridge 直接委譲を validator 参照へ置換
   - Step3: Engine hook は telemetry/side-effect のみに限定

### 影響テスト

- 更新
  - `src/tests/PartialPublicationRejectTests.cpp`
  - `src/tests/RuntimePublicationCoordinatorTests.cpp`
- 追加候補
  - `src/tests/PublicationValidatorIsolationTests.cpp`

---

## 2.6 P3-B: Crossfade 継続監査の強化（#4）

### 対象

- `src/tests/CrossfadeExecutorLocalContractTests.cpp`
- `src/tests/ObservePathSingleSourceTests.cpp`
- `src/audioengine/AudioEngine.h`
  - `makeCrossfadePreparedSnapshotFromWorld(...)`
  - `makeEngineRuntimeState(...)`

### 改修内容

1. 禁止トークンテストの範囲拡張
   - `Commit/Timer` に加えて、branch系ファイルの明示検査を追加
2. authority/projection 混線の検知追加
   - `preparedCrossfade` と `overlap semantic` の二重参照を CI fail

---

## 3. 実施順（推奨スプリント分割）

1. **Sprint-1**: P1-A（publish API 単一化の骨格）
2. **Sprint-2**: P1-B（#16/#17 重複解消）
3. **Sprint-3**: P2-A/P2-B（hash/equivalence 仕様固定）
4. **Sprint-4**: P3-A/P3-B（validator分離 + 監査強化）

---

## 4. Definition of Done（DoD）

- D1: production path で `publishState(current,next,...)` 呼び出しゼロ
- D2: `ExecutionSemantic` / `SchedulingSemantic` の重複 authority ゼロ
- D3: `activationEpoch` authority 単一化
- D4: hash/equivalence の対象範囲が inventory と一致
- D5: `PartialPublicationRejectTests` / `RuntimePublicationCoordinatorTests` / `ObservePathSingleSourceTests` / `CrossfadeExecutorLocalContractTests` が全通過

---

## 5. 注意点

- `#21` は「事実」だが「必ず不正」ではない（要件依存）。
  したがって一括削除ではなく、validator分離を優先。
- `#3/#10/#11/#14/#22` は未達扱い不可のため、今回計画では新規の未達課題として扱わない。
- CodeGraph は本リポジトリで一部 unresolved が残るため、最終判定は grep/Serena の一致を優先する。
