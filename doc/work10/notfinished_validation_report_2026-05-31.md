# notfinished.md 妥当性検証レポート（2026-05-31）

## 1. 目的と結論

`doc/work10/notfinished.md` に記載された「Practical Stable ISR Bridge Runtime 未達箇所」の指摘について、現行ソースを横断して妥当性を検証した。

**総合結論**:

- 指摘は全体として **妥当**。
- 特に「Publication/Retire は高成熟、ただし Authority 一本化（Execution/Observe/Crossfade）と Semantic Closure 証明が未完」という評価は、実コードと整合する。

---

## 2. 検証方法（実施ログ要約）

- `orios/serena` による構文/パターン探索
- `codegraph` による構造確認（ファイル構造・抜粋）
- 主要対象:
  - `src/audioengine/AudioEngine.h`
  - `src/audioengine/AudioEngine.Commit.cpp`
  - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
  - `src/audioengine/AudioEngine.Timer.cpp`
  - `src/audioengine/RuntimeGraph.h`
  - `src/audioengine/ISRBarrierOptimizer.cpp`
  - `src/tests/RuntimeSemanticSchemaValidationTests.cpp`

---

## 3. 妥当と判断した主要指摘（証拠つき）

### 3.1 RuntimeWorld の自己完結性未達（外部状態依存）

**指摘妥当**。`buildRuntimePublishWorld(...)` が RuntimeWorld 構築時に外部状態を参照している。

- `readControlRuntimeView()`
- `getRuntimeSnapshot(runtimeReadView)`
- fallback で `consumeAtomic(currentProcessingOrder, ...)`

該当: `src/audioengine/AudioEngine.h`（`buildRuntimePublishWorld` 周辺）

### 3.2 Descriptor Inventory と RuntimeState 実体の不一致

**指摘妥当**。`RuntimeState` は多数の semantic field を保持する一方、`kFieldDescriptors` は 9 件。

- 実体: generation/topology/routing/execution/publication/overlap/retire/timing/latency/scheduling/resource/affinity/automation/coefficient/projectionFreshness/semanticHash ほか
- descriptor: 9 件

該当: `src/audioengine/AudioEngine.h`（`struct RuntimeState`）

### 3.3 Precheck の Semantic/Projection 混在

**指摘妥当**。`runPublicationPrecheckNonRt(...)` 内で semantic 検証に加え、`world.graph`/`world.engine` の要素を分岐条件として使用。

例:

- `world.graph.activeNode`
- `world.graph.fadingNode`
- `world.engine.transition.next`

該当: `src/audioengine/AudioEngine.Commit.cpp`（`runPublicationPrecheckNonRt`）

### 3.4 RuntimeGraph 直観測経路の残存

**指摘妥当**。`RuntimeReadView`/`RuntimePublishView` が `const RuntimeGraph* graph` を保持し、Audio/Timer 側で直接参照。

- `RuntimeReadView`/`RuntimePublishView` 定義
- `AudioEngine.Processing.AudioBlock.cpp` で `getRuntimeGraph(runtimeReadViewRef)`
- `AudioEngine.Timer.cpp` で `runtimeGraph->generation/runtimeUuid/...` を監視

該当:

- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- `src/audioengine/AudioEngine.Timer.cpp`

### 3.5 active/fading DSP slot と commit 系の並列権威

**指摘妥当**。`activeRuntimeDSPSlot` / `fadingRuntimeDSPSlot` が現存し、`commitNewDSP(...)` では slot 更新と別権威系（handle/crossfade authority）を併用。

- `setActiveRuntimeDSP(newDSP)`
- `dspHandleRuntime_.beginCrossfade(...)`
- `crossfadeAuthorityRuntime_.registerCrossfade(...)`
- `activeCrossfadeId_` publish

該当:

- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Commit.cpp`

### 3.6 PublicationIntent / backlog 系 state machine 残存

**指摘妥当**。`PublicationIntent`, `PublicationLog`, `publicationBacklog_` など Engine 側 state が存在。

該当:

- `src/audioengine/AudioEngine.Commit.cpp`
- `src/audioengine/AudioEngine.h`

### 3.7 BarrierOptimizer の実効性不足

**指摘妥当**。`optimizeBarriers()` は分岐のみで実処理なし。

該当: `src/audioengine/ISRBarrierOptimizer.cpp`

### 3.8 RuntimeGraphRevision と generation の二重管理傾向

**指摘妥当（少なくとも要監査）**。`runtimeGraphRevision` の publish/consume が RT 実行フレームで利用される。

該当:

- `src/audioengine/AudioEngine.h`（publish）
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`（consume）

---

## 4. 追加監査による確定判定（2026-05-31 追補）

### 4.1 LegacyTemporary の運用統治

#### 確定判定（LegacyTemporary）

問題としては未確定（懸念は解消寄り）。

追加監査で、`LegacyTemporary` は manifest/expiry/deadline/owner/replacement_authority を含む統治経路が実装され、検証スクリプトで強制されていることを確認した。

- manifest 実体: `.github/isr-legacy-temporary.json`
  - `schema`, `issue`, `expiry`, `owner`, `replacement_authority`, `removal_phase`, `deadline` を保持
- 検証スクリプト: `.github/scripts/isr-verify-authority-inventory.ps1`
  - manifest の必須フィールド検証
  - `expiry` / `deadline` の日付妥当性・期限切れ検証
  - `LegacyTemporary` state と manifest の対応（欠落/ stale）検証
  - `replacement_authority == Authoritative`、`removal_phase == Phase 3` の契約検証
- 配線: `.github/scripts/isr-run-tiered-verification.ps1` に `isr-verify-authority-inventory.ps1` が含まれる

実行結果:

- `isr-verify-authority-inventory.ps1` は **manifest統治項目では失敗せず**、今回の fail は `Authority source growth detected: addedCount=4`（別論点）によるもの。

補足: `isr-verifier-registry.json` 上の verifier 名は `legacy-manifest-expiry-verifier` だが、実際の enforcement は `isr-verify-authority-inventory.ps1` 側で吸収されている。

### 4.2 Shadow Compare の release gate 統合

#### 確定判定（Shadow Compare）

問題としては否（統合済み）。

追加監査で、Shadow Compare は契約・cadence・tier 配線・gate wiring まで実装済みであることを確認した。

- 契約検証スクリプト: `.github/scripts/isr-verify-shadow-compare-contract.ps1`
  - `recordShadowCompareObservation(...)` 呼出し
  - `RuntimeSemanticHash` カバレッジ
  - `semanticHashEquals(...)` 実装
- cadence 検証スクリプト: `.github/scripts/isr-verify-shadow-compare-cadence.ps1`
  - `evidence/shadow_compare_cadence.json` の schema/閾値/必須フィールド検証
- tier 配線: `.github/scripts/isr-run-tiered-verification.ps1` に両スクリプトが登録
- gate wiring 自己検証: `.github/scripts/isr-verify-gate-wiring.ps1` が PASS

実行結果:

- `isr-verify-shadow-compare-contract.ps1` : PASS
- `isr-verify-shadow-compare-cadence.ps1` : PASS
- `isr-verify-governance-registries.ps1` : PASS
- `isr-verify-gate-wiring.ps1` : PASS

---

## 5. 補足で発見した追加論点

### 5.1 テストの統治カバレッジ不足

`RuntimeSemanticSchemaValidationTests.cpp` は主に Routing/Execution 範囲検証で、Descriptor 網羅性や Authority 一本化の不変条件を直接テストしていない。

該当: `src/tests/RuntimeSemanticSchemaValidationTests.cpp`

### 5.2 publication single-path 検証スクリプトの適用範囲

`.github/scripts/isr-verify-publication-single-path.ps1` は publication path 契約の検証として有効だが、Execution/Observe/Crossfade authority の単一化まで包括的に担保するものではない。

該当: `.github/scripts/isr-verify-publication-single-path.ps1`

---

## 6. 監査結論（再掲）

現状は以下の評価が妥当:

- **達成済み寄り**: Publication / Retire / Freeze / Sequence/Governance の骨格
- **未達中心**: Authority Collapse（Execution/Observe/Crossfade）、Semantic Closure 証明、Descriptor 完全性統治

したがって、`notfinished.md` の主張は「過剰評価」ではなく、**構造監査として有効**。

---

## 7. 次アクション案（任意）

1. Descriptor 完全網羅テスト追加（RuntimeState 実体と Descriptor の一致検証）
2. RuntimeGraph 直参照禁止/限定ルールを lint/gate 化
3. Crossfade authority を RuntimeWorld semantic に収束させる段階計画化
4. LegacyTemporary manifest 運用の CI 強制
