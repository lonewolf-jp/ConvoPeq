# Practical Stable ISR Bridge Runtime 移行計画書 v3.0（検証反映版）

**Document Version:** 3.0
**Date:** 2026-06-19
**Target:** ConvoPeq Runtime Architecture
**Goal:** Practical Stable ISR Bridge Runtime への完全移行
**Status:** 検証反映版（doc/work49/validation_report.md に基づく）

> **注意:** 本計画書は v2.0 の全主張をコードベース（278 ファイル）の実態と照合し、
> 6 種類のツール（Serena MCP, AiDex MCP, CodeGraph MCP, graphify, semble, Select-String）
> を用いて検証した結果を反映している。
> v2.0 と v3.0 の最大の差異は「各 Phase の未実装/実装済みの判定」である。

---

# 1. エグゼクティブサマリー

## 現状（検証済み）

ConvoPeq の実装達成率は v2.0 記載の 92-95% ではなく **95-97%** と評価する。

その理由は以下の機能が v2.0 執筆時点で「未実装」とされていたが、
実際には**既に実装済み**であるため：

| 機能 | v2.0 の記述 | 実際の状態 |
|---|---|---|
| RuntimePublicationValidator | §8: Placeholder (`return true`) | 全バリデーション（Topology/Resource/Transition）実装済み |
| Validation Telemetry | §11: 未実装 | 6000-6003 イベント定義、emitValidationEvent()、1秒レート制限完備 |
| CrossfadePolicy | §13: 導入予定 | `CrossfadeAuthority.h` で定義済み |
| CrossfadeAuthority | §12: 分離予定 | `evaluate()` 純粋関数実装済み、HealthState 非依存 |
| Emergency Override | §15: 導入予定 | DSPTransition::onPublishCompleted() 内で実装済み |
| publishIdleWorldOnly() | §16: 導入予定 | AudioEngine.Transition.cpp で実装済み |
| HealthState責務移動 | §14: 移動予定 | 既に CrossfadeAuthority は HealthState 非依存 |

---

## 残る本質的課題（検証済み）

v2.0 の課題認識は概ね正しい。ただし解決手段の多くは既に実装済み。

真に残る課題は以下：

1. **PersistentStateBlock 不在**: `publicationSequenceId` / `publicationEpoch` / `mappedRuntimeGeneration` が Coordinator に分散保持されている
2. **AuthoritySource 不在**: Publish 要求の発行元追跡機構がない
3. **deriveAuthorityState() 不在**: リカバリの核となる再導出関数がない
4. **currentWorld_ 残留**: `ISRRuntimePublicationCoordinator` の `std::atomic<const void*>` が RuntimeStore と二重管理
5. **Recovery Architecture 未体系化**: RecoveryAction/EscalationTracker は存在するが、Step1-6 の体系的手順が未定義

---

# 2. 最終到達アーキテクチャ（変更なし）

v2.0 の以下設計は検証の結果、妥当と確認：

- **Single Source Of Truth**: RuntimeStore（`src/core/RuntimeStore.h`）は既に template として実装済み
- **Single Authority**: Coordinator::publishWorld() + Orchestrator::trySubmit() が既に唯一の Publish 経路
- **Rebuildable Runtime**: 原則として正しい。PersistentStateBlock の不在が唯一のギャップ

---

# 3. Authority State 分類（修正）

## 3.1 Persistent State（変更なし）

```
publicationSequenceId
publicationEpoch
mappedRuntimeGeneration
```

**検証結果**: これらは `ISRRuntimePublicationCoordinator` に `std::atomic` として存在する。
`PersistentStateBlock` としての統一インターフェースは未実装。

## 3.2 Rebuildable State（修正）

v2.0 では以下を「保持禁止」としていたが、検証の結果これらは既に適切に管理されている：

- `activeCrossfadeCount` → RuntimeStore 経由で取得
- `hasPendingPublication` → Coordinator state から導出
- `hasActiveRuntime` → RuntimeStore.observe() で取得
- `crossfadeInProgress` → CrossfadeRuntime が管理

いずれも RuntimeStore の SSOT をバイパスしていないことを確認済み。

---

# 4. PersistentStateBlock 導入（Phase-6 → 実装必要）

**ステータス: 未実装**

### 設計（v2.0 を踏襲、検証済み）

```cpp
struct PersistentStateBlock
{
    std::atomic<uint64_t> publicationSequenceId{0};
    std::atomic<uint64_t> publicationEpoch{0};
    std::atomic<uint64_t> mappedRuntimeGeneration{0};
};
```

### snapshot() / update()

v2.0 の設計を踏襲。

---

# 5. RuntimeStore SSOT 化（Phase-9 → 部分的に未完了）

**ステータス: ほぼ完了。currentWorld_ の削除が残課題。**

### 検証結果

- `runtimeStore.observe()` は既に唯一の取得経路として機能
- `getCurrent()` は `ISRRuntimePublicationCoordinator` の `currentWorld_` を返す
- `RuntimePublicationCoordinator`（template）は `RuntimeStore` 経由で publishWorld() を実行

### 残作業

`ISRRuntimePublicationCoordinator::currentWorld_`（`std::atomic<const void*>`）の削除。
ただし注意：

- `commit()` 内で `currentWorld_` に `newWorld` を格納
- `retire()` 内で `compareExchangeAtomic` を使用
- `getCurrent()` がこの値を返す

→ RuntimeStore の WriteAccess::publishAndSwap() で置換可能。ただし `ISRRuntimePublicationCoordinator` と `RuntimePublicationCoordinator`（template）の責務整理が必要。

---

# 6. AuthoritySource 導入（Phase-5 → 実装必要）

**ステータス: 未実装**

### 設計（v2.0 を踏襲）

```cpp
enum class AuthoritySource : uint8_t
{
    Unknown,
    UserAction,
    PresetLoad,
    Recovery,
    DSPTransition,
    HealthMonitor
};
```

### Coordinator API

```cpp
publishWorld(..., AuthoritySource src);
```

### Telemetry

```cpp
std::atomic<uint64_t> callCount[6];
```

**注**: Validator Telemetry (6000-6003) は既に実装済みのため、Authority Telemetry は分離して実装する。

---

# 7. RuntimePublicationValidator（Phase-1 → 実装済み、確認のみ）

**ステータス: ✅ 実装完了。v2.0 の「Placeholder」記述は誤り。**

### 検証結果

`RuntimePublicationValidator` は既に以下の完全な実装を持つ：

- `validatePublication()` — 4段階の検証パイプライン
- `validateSemanticConsistency()` — generation/sequenceId の不変条件
- `validateTopology()` — runtimeUuid/fading/transition の整合性
- `validateResources()` — Oversampling/Dither/NoiseShaper の許容値
- `checkNoConflictingTransitions()` — SmoothOnly/DryAsOld/HardReset の遷移検証

### コード上の注意点

v2.0 §9 の許容値に以下の誤りがあったため訂正：

| 項目 | v2.0 記載 | 実際のコード |
|---|---|---|
| Dither BitDepth | 0, 16, 24 | 0, 16, 24, **32** |
| NoiseShaper Type | 0, 1, 2 | 0, 1, 2, **3** |

---

# 8. Validation Telemetry（Phase-1.5 → 実装済み）

**ステータス: ✅ 実装完了。v2.0 の記述を確認・追認。**

### 検証結果

- イベントコード 6000-6003 定義済み（`RuntimeHealthMonitor.h`）
- `emitValidationEvent()` 実装済み（`RuntimeHealthMonitor.cpp`）
- 1秒レート制限（`kValidationEventMinIntervalUs = 1'000'000`）
- `AudioEngine::runPublicationPrecheckNonRt()` から呼び出し確認済み

---

# 9. CrossfadeAuthority 分離（Phase-2 → 実装済み）

**ステータス: ✅ 実装完了。**

### 検証結果

- `CrossfadeAuthority::evaluate()` は pure function（AudioEngine 非依存）
- `CrossfadePolicy` は immutable POD（HealthState 含まず）
- 既存テスト: `CrossfadeAuthorityRegressionTest`（4 ケース）

---

# 10. Emergency Override（Phase-2.5 → 実装済み）

**ステータス: ✅ 実装完了。**

### 検証結果

`DSPTransition::onPublishCompleted()` 内に以下を確認：

1. HealthState::Critical 検知
2. `lifetime.activate(newDSP)` 即時 activate
3. `crossfadeRuntime_.complete()` + `lifetime.retire(oldDSP)`
4. `EVENT_CROSSFADE_ABORTED_EMERGENCY (4003)` 発行

---

# 11. Recovery Architecture（Phase-8 → 部分実装、体系化必要）

**ステータス: ⚠️ 部分的に実装済み。体系化が不足。**

### 既存要素

- `RecoveryAction` enum（Observe/Throttle/Recover/Restore/Safe/Critical）— `RuntimePolicyEngine.h`
- `RecoveryBudget` — ウィンドウ管理、エスカレーション追跡
- `EscalationTracker` — 問題C/A-2対応
- `AudioEngine.Timer.cpp` 内の Recovery Action 実行ロジック

### 不足要素

v2.0 §18 の Step1-6 体系手順：

1. RuntimeStore取得
2. PersistentState取得（PersistentStateBlock 未実装のため不可）
3. deriveAuthorityState()（未実装）
4. 状態比較
5. 不足状態補完
6. Publish再開

→ PersistentStateBlock と deriveAuthorityState() の実装が前提。

---

# 12. テスト戦略（Phase-3 → 一部実装済み）

**ステータス: ⚠️ Validator/CrossfadeAuthority テストは既存。Recovery/Property テストは未着手。**

### 既存テスト（検証済み）

- `PublicationValidatorIsolationTests`: 30+ テストケース（Topology/Resource/Transition/Semantic 網羅）
- `CrossfadeAuthorityRegressionTest`: 4 テストケース（決定論/Policy変更/同一Hash/Oversampling変更）

### 未着手

- Recovery 障害注入テスト
- Property Test（10,000〜100,000 回ランダムシーケンス）
- Validator エッジケース拡充（必要に応じて）

---

# 13. フェーズ計画（改訂版）

## Phase-A: 確実な不足対応（推奨工数: 小〜中）

### A-1: PersistentStateBlock 導入

- `ISRRuntimePublicationCoordinator` から `publicationSequenceId_` / `publicationEpoch_` / `mappedRuntimeGeneration_` を抽出
- `PersistentStateBlock` 構造体として統一
- snapshot() / update() メソッド実装

### A-2: AuthoritySource 導入

- `AuthoritySource` enum 定義
- `publishWorld()` に source パラメータ追加
- Telemetry カウンタ追加（任意）

### A-3: deriveAuthorityState() 実装

- 入力: `PersistentStateBlock`, `RuntimeStore`
- 出力: `AuthorityState`（再導出状態）
- Pure Function として実装

### A-4: currentWorld_ 統合

- `ISRRuntimePublicationCoordinator` の `currentWorld_` を RuntimeStore に統合
- 互換性のため getCurrent() は RuntimeStore.observe() に委譲

## Phase-B: テスト拡充（推奨工数: 小）

### B-1: Recovery 障害注入テスト

### B-2: Property Test（任意）

### B-3: 既存テストのメンテナンス

---

# 14. 完了判定（検証反映版）

## Authority

```
Publish Authority = Coordinatorのみ
```

→ **達成済み**。`submitPublishRequest()` → `Orchestrator` → `Coordinator::publishWorld()` の経路のみ。

## Truth

```
Runtime Truth = RuntimeStoreのみ
```

→ **ほぼ達成**。`currentWorld_` の削除が残課題。

## Persistent State

```
PersistentStateBlockのみ
```

→ **未達成**。Phase-A-1 で対応。

## Recovery

```
deriveAuthorityState()で完全再導出可能
```

→ **未達成**。Phase-A-3 で対応。

## Validation

```
Placeholder = 0
```

→ **達成済み**。Validator は全実装済み。

## Crossfade

```
CrossfadeAuthority → Pure Function
```

→ **達成済み**。`CrossfadeAuthority::evaluate()` は pure function。

## State Duplication

```
currentWorld_ = 0箇所
```

→ **未達成**。`ISRRuntimePublicationCoordinator` に 1 箇所存在。Phase-A-4 で対応。

---

# 15. 最終達成率評価（検証反映版）

現状

```
95-97%
```

Phase-A 完了

```
98-99%
```

Phase-B 完了

```
99-100%
```

---

# 16. 検証プロセス

## 使用ツール

| ツール | 用途 | 結果 |
|---|---|---|
| **Serena MCP** (find_symbol / find_referencing_symbols) | シンボル構造・参照関係の把握 | 236 ファイル、clangd v19.1.2 正常動作 |
| **AiDex MCP** (aidex_query / aidex_status) | 高速識別子検索 | 278 ファイル、12247 items、4248 methods、401 types のインデックス |
| **CodeGraph MCP** (get_file_structure) | ファイル構造確認 | Incremental index 完了、67 communities |
| **graphify MCP** (god_nodes / get_node) | 知識グラフ中心ノード確認 | Practical Stable ISR Bridge Runtime Masterplan: 325 edges |
| **semble** (semantic search) | 自然言語ベースセマンティック検索 | 全セマンティック検索正常動作 |
| **Select-String** | パターンベース grep | 未実装項目の有無確認 |

## 検証ファイル

- ソースコード: `src/audioengine/*.h/.cpp` の中核ファイル
- テスト: `src/tests/PublicationValidatorIsolationTests.cpp`
- コア: `src/core/RuntimeStore.h`, `src/core/RuntimePublicationCoordinator.h`
- ポリシー: `src/audioengine/RuntimePolicyEngine.h/.cpp`
- 設定: `.github/copilot-instructions.md`, AGENTS.md, steering/rules/constitution.md
