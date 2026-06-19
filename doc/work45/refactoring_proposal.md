# ConvoPeq ISR Bridge Runtime 改修計画

> 当初提案（2026-06-18）
> 出典: 改修計画「ConvoPeq.md全体を「Practical Stable ISR Bridge Runtime（実運用で破綻しにくい設計）」の観点で再評価」

---

## 1. 未達成な内容

### (A) ISR Bridgeの「単一真実性（Single Source of Truth）」が破れている

**問題概要**

RuntimeWorld / dspProjection / AudioEngine / CrossfadeAuthority がそれぞれ「状態判断の起点」を持っている。

これにより：

- crossfade判断が複数箇所に分散
- retirement条件とpublication条件が別経路で評価
- Snapshotと実DSP状態の乖離リスク

**該当コード**

#### CrossfadeAuthority

```cpp
const auto oldHash = oldWorld.dspProjection.structuralHash;
const auto newHash = newWorld.dspProjection.structuralHash;
```

#### RuntimeHealthMonitor

```cpp
checkCrossfadeTimeout();
checkCrossfadeEventDrop();
checkWorldConsistency();
```

#### RuntimePublicationValidator

```cpp
// Placeholder for actual conflict detection logic
return true;
```

**問題の本質**

- 「crossfade必要性」が複数モジュールで独立計算
- しかもValidatorは未実装（常にtrue）

---

### (B) RT/Non-RT境界の「責務分離が論理的に破綻」

**問題概要**

設計上：

- RT: atomic / snapshotのみ
- Non-RT: validation / rebuild / allocation

しかし現状：

- RuntimePublicationCoordinator が RT寄りロジックを保持
- validatorが空実装
- crossfade decisionがRT側に影響する経路が存在

**該当コード**

```cpp
if (!bridge_.validatePublicationNonRt(*worldOwner))
```

→ "NonRt" と名前が付いているが呼び出しタイミングは実質RT遷移直前

さらに：

```cpp
std::atomic_thread_fence(std::memory_order_release);
auto* oldWorld = writeAccess_.publishAndSwap(newWorld);
```

→ fence + swap の順序保証が「設計依存」になっている

---

### (C) Snapshot / RCU設計の「寿命管理が不完全」

**問題概要**

RCUReader + SnapshotCoordinator があるが：

- retire経路の保証が弱い
- DeletionQueueとRetireQueueの関係が曖昧
- oldWorldの所有権が分散

**該当コード**

```cpp
bridge_.retireRuntimePublishWorldNonRt(oldWorld, false);
```

→ retireの意味が「即時破棄なのか遅延なのか不明」

また：

```cpp
bridge_.willRetireRuntimeNonRt(oldWorld);
```

→ observer hook依存で寿命制御している

---

### (D) Validatorが「形式的に存在するだけ」

**問題概要**

RuntimePublicationValidator:

```cpp
bool validateTopology(...) const { return true; }
bool validateResources(...) const { return true; }
```

→ 実質 no-op

**影響**

- ISR Bridgeの安全性が「テスト依存」
- 実運用で破綻検出不能

---

### (E) Crossfade Runtime と DSP state の整合性保証が弱い

**問題概要**

Crossfade決定は存在するが：

- 実DSP適用との結合点が不明確
- fadeTimeSecが複数ソースからmax取られているだけ

```cpp
ctx.fadeTimeSec = std::max(ctx.fadeTimeSec,
    consumeAtomic(engine.m_irFadeTimeSec));
```

→ policyではなく "accumulation heuristic"

---

### (F) HealthMonitor が「観測系」と「制御系」を混在

**問題概要**

RuntimeHealthMonitor::tick()

```cpp
checkRetireStall();
checkPublicationStall();
diagnoseRetireStall();
```

さらに：

```cpp
auto trend = computeTrend(...)
m_policyEngine_.resetVerification();
```

→ 観測オブジェクトが状態遷移を実行している

---

## 2. 該当ソースコード箇所（要約付き）

| 問題 | 該当箇所 |
|------|----------|
| crossfade分散 | CrossfadeAuthority.cpp / RuntimeHealthMonitor.cpp |
| validator無力化 | RuntimePublicationValidator.cpp |
| RT/NonRT混在 | RuntimePublicationCoordinator.cpp |
| retire寿命曖昧 | retireRuntimePublishWorldNonRt() |
| snapshot責務曖昧 | SnapshotCoordinator / RCUReader |
| state mutation in monitor | RuntimeHealthMonitor::tick() |

---

## 3. 本来あるべき姿（Practical Stable ISR Bridge Runtime）

### (A) Single Decision Authority Model

**原則**

```
Crossfade Decision = 1箇所のみ
Retire Decision    = 1箇所のみ
Publication Decision = 1箇所のみ
```

**理想構造**

```
RuntimePolicyEngine
    ├── CrossfadePolicy
    ├── RetirePolicy
    └── PublicationPolicy
```

CrossfadeAuthorityは削除 or pure projection関数化

---

### (B) RTスレッドは「完全受動化」

RT側は禁止：

- policy evaluation
- decision making
- logging

許可：

- atomic snapshot read
- DSP execution

---

### (C) Validatorは必須・非空実装

```cpp
validateTopology → must reject cycles
validateResources → must estimate DSP load
validateSemanticConsistency → must enforce ISR schema
```

---

### (D) Snapshot/RCUの単一寿命管理

理想：

```
writeAccess.swap()
    ↓
RetireQueue.enqueue(oldWorld)
    ↓
DeferredDeletionThread owns lifecycle
```

現状の問題は「bridgeが寿命を持つ」点 → 必ず external lifecycle manager に移譲すべき

---

### (E) Crossfadeは deterministic state machine

現状：heuristic accumulation

あるべき：

```
CrossfadeStateMachine
  INPUT: (oldWorld, newWorld, policy)
  OUTPUT: (fadeType, duration, curve)
```

engine値の直接参照は禁止

---

### (F) Monitorは純観測化

```cpp
RuntimeHealthMonitor
    → metrics export only
    → no state mutation
```

policyEngineに統合すべき

---

## 4. 改修方法（具体）

### 修正1：CrossfadeAuthorityの削除または縮退

**方針**

- evaluate() → pure function化
- engine参照禁止

```cpp
Decision evaluate(
    const RuntimeWorldDiff& diff,
    const CrossfadePolicy& policy);
```

---

### 修正2：RuntimePublicationValidatorの完全実装

最低限：

```cpp
if (world.routing.hasCycle())
    return false;
if (world.resource.dspLoad > threshold)
    return false;
```

---

### 修正3：PolicyEngine統合（最重要）

**追加構造**

```cpp
class ISRPolicyEngine {
    CrossfadePolicy crossfade;
    RetirePolicy retire;
    PublicationPolicy publication;
};
```

すべての decision をここへ集約

---

### 修正4：RTスレッドから分岐ロジック除去

削除対象：

- checkCrossfadeTimeout()
- checkConfigurationDrift()
- diagnoseRetireStall()
→ すべて metric exportへ

---

### 修正5：Retire lifecycleの単一化

**現状**：bridge + queue + hook + coordinator

**修正後**：

```
RuntimeStore owns lifecycle
DeferredDeletionThread owns destruction
```

bridgeは禁止：delete / free / retire timing control

---

### 修正6：Snapshot責務整理

RCUReaderは read-onlyに限定し：

- snapshot生成 → SnapshotFactory
- snapshot配布 → SnapshotCoordinator
- snapshot破棄 → DeletionQueue

---

## 5. 総合評価

ConvoPeqのISR Bridgeは：

- 「設計思想」は高度（RCU / snapshot / tiered payload）
- しかし **decision authority が分散しすぎている**

結果として：
> "Practical Stable" の核心条件（decision determinism）が未達

未達の本質はバグではなく：
> **ISR Bridge が「データ駆動」ではなく「ロジック分散型」になっている**

---

## 6. 現状評価（最新版ベース）

### 既に達成済み

#### RuntimeStore

単一路線化済み

```cpp
return exchangeAtomic(store_->current, next, std::memory_order_acq_rel);
```

#### RuntimeWorld

Semantic Schema化済み（generation / routing / execution / publication / overlap / retire / timing）

#### RuntimePublicationCoordinator

単一路線Publish

```cpp
coordinator.publishWorld(...)
```

#### CrossfadeAuthority

DSP直読排除済み

```cpp
oldWorld.dspProjection
newWorld.dspProjection
```

#### Retire

DeferredDelete化済み

```cpp
enqueueDeferredDeleteNonRt(...)
```

**Practical Stable達成率：85〜90%**

---

## 7. 最小修正で収束させる方針（優先順位順）

```text
P1 Validator完成
P2 Crossfade Policy分離
P3 HealthMonitor純観測化
P4 Retire Authority整理
P5 RuntimeWorld Freeze強化
```

---

### P1 RuntimePublicationValidator完成

現状最大の未完成部分。現在は Placeholder。

**修正**：最低限 `transitionActive`, `hasFadingRuntime`, `fadeTimeSec` の整合性を検証。

```cpp
if (world.execution.transitionActive != world.topology.hasFadingRuntime)
    return false;
```

さらに `fadeTimeSec > 0` 必須化。

期待効果：Crossfade不整合検出、Publish前Fail Closed

---

### P2 Crossfade Policy抽出

現状、CrossfadeAuthority が `engine.m_irFadeTimeSec`, `engine.m_phaseFadeTimeSec`, `engine.m_tailFadeTimeSec` などを直接参照。

**修正**：

```cpp
struct CrossfadePolicy {
    double irFade;
    double phaseFade;
    double tailFade;
};
```

Authority は `Decision evaluate(oldWorld, newWorld, policy)` へ変更。

効果：CrossfadeAuthorityが純粋関数になる。

---

### P3 HealthMonitor純観測化

現状、Crossfade timeout時に `unregisterCrossfade()`, `crossfadeRuntime_.complete()`, `publishIdleWorldOnly()` を実行。

**修正**：HealthMonitorは `EVENT_CROSSFADE_TIMEOUT` のみ発行。回復処理は `PolicyEngine` へ移動。

理想：

```text
HealthMonitor → Event → PolicyEngine → Recovery
```

効果：監視系と制御系分離

---

### P4 Retire Authority整理

現状、退役責務が `DSPLifetimeManager / RetireRouter / Coordinator / DeferredDelete` へ分散。

**修正**：`RetireAuthority` を追加し `scheduleRetire()` のみを責務とする。内部は既存実装を呼ぶだけ。

効果：将来の保守性向上

---

### P5 RuntimeWorld Freeze強化

現状、Builder側コメント「freeze は caller が行う」。Publish前に `assertMutable()` が残っている。

**修正**：Publish直前 `world->sealRecursively()` 強制。Debug限定で `assertMutable()` 失敗を強化。

効果：RuntimeWorld Immutable保証強化

---

## 8. 実施しない方が良い改修

- RuntimeStore全面刷新（既に十分良い）
- RuntimePublicationCoordinator除去（既に権限集中できている）
- Snapshot全面再設計（既に存在）
- CrossfadeAuthority削除（むしろ残すべき、DSP直読排除済み）

---

## 9. 推奨ロードマップ

```text
Phase 1: RuntimePublicationValidator完成
Phase 2: CrossfadePolicy抽出
Phase 3: HealthMonitor → Event化
Phase 4: RetireAuthority導入
Phase 5: RuntimeWorld Freeze強化
```

---

## 10. PolicyEngine 統合設計（完全版）

### 完成形アーキテクチャ

```text
                       Runtime Intent
                              │
                              ▼
                    RuntimeWorldBuilder
                              │
                              ▼
                     RuntimeWorld(New)
                              │
                              ▼
                  RuntimePublicationValidator
                              │
                              ▼
                    RuntimePolicyEngine
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
 PublicationDecision   CrossfadeDecision    RetireDecision
        │                     │                     │
        └─────────────┬───────┴─────────────┬──────┘
                      ▼                     ▼
              RuntimePublicationCoordinator
                      │
                      ▼
                 RuntimeStore
                      │
             oldWorld ▼
                  RetireRouter
                      │
                      ▼
                 DeleteQueue
```

### RuntimePolicyEngine クラス構成

```cpp
class RuntimePolicyEngine {
public:
    PolicyDecision evaluate(
        const RuntimeWorld& oldWorld,
        const RuntimeWorld& newWorld,
        const RuntimeMetrics& metrics) const;
private:
    PublicationPolicy publicationPolicy_;
    CrossfadePolicy crossfadePolicy_;
    RetirePolicy retirePolicy_;
    RecoveryPolicy recoveryPolicy_;
};
```

### PolicyDecision

```cpp
struct PolicyDecision {
    PublicationDecision publication;
    CrossfadeDecision crossfade;
    RetireDecision retire;
    RecoveryDecision recovery;
};
```

### Coordinatorの役割

Coordinatorは馬鹿でよい。PolicyDecision を受け取り、その通り実行する。
禁止：`if (needCrossfade)`, `if (retireNow)`。全て `decision.*` に従う。

---

## 11. 削除候補

### レベルA（実質削除推奨）

- RuntimePublicationValidator内のPlaceholder Validator群（実装するか削除するか）

### レベルB（モジュール削除ではなく縮退）

- HealthMonitor内のRecovery実装
- Observer系の制御ロジック

### レベルC（名前変更推奨）

- CrossfadeAuthority → CrossfadeEvaluator または CrossfadePolicyEvaluator
- Coordinator内部の判断ロジック

### レベルD

- RetireRouterの判断部分

### 削除してはいけないモジュール

- RuntimeStore / RuntimePublicationCoordinator / SnapshotCoordinator / RetireRouter / RuntimeWorld / DeferredDeleteQueue
