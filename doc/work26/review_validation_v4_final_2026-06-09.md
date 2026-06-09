# Practical Stable ISR Bridge Runtime 検証レポート v4（最終版）

**作成日**: 2026-06-09
**更新日**: 2026-06-09
**検証対象**: v1(10項目) + v2(11項目) + v3(6項目) + 補正点
**目的**: Practical Stable ISR Bridge Runtime の未達成項目を4階層で評価
**注記**: 本レポートはプロジェクト管理者との協議を経て、優先度・妥当性を最終調整した。

---

## 前回までの評価

| 版 | カバレッジ | 筆者の評価 |
| --- | --- | --- |
| v1 | 構造面10項目 | 約70%妥当 |
| v2 | 構造面11項目（改善版） | 約85%妥当 |
| v3 | 運用監査6項目追加 | 約90〜95%妥当 |

**本レポート(v4)では、v3で不足していた RuntimePublicationValidator の検証と、優先順位の再整理を行う。**

---

## 最重要補正: RuntimePublicationValidator が実質未完成

### エビデンス

`RuntimePublicationValidator.cpp` の全検証メソッド:

| メソッド | 実装 | 状態 |
| --- | --- | --- |
| `validateSemanticConsistency()` | `crossfadeStartDelayBlocks >= 0` などの実チェックあり | ✅ 実装済み |
| `validateTopology()` | `return true; // Placeholder` | ❌ **ダミー** |
| `validateResources()` | `return true; // Placeholder` | ❌ **ダミー** |
| `checkNoConflictingTransitions()` | `return true; // Placeholder` | ❌ **ダミー** |
| `checkActivationEpochConsistency()` | `return true;` | ❌ **常に成功** |

具体的なコード:

```cpp
// RuntimePublicationValidator.cpp より
bool RuntimePublicationValidator::validateTopology(
    const RuntimePublishWorld& world) const
{
    // This is a placeholder for actual topology validation logic
    return true; // Placeholder
}

bool RuntimePublicationValidator::validateResources(
    const RuntimePublishWorld& world) const
{
    // This is a placeholder for actual resource validation logic
    return true; // Placeholder
}
```

### テストもダミー

`PublicationValidatorIsolationTests.cpp`:

```cpp
TEST_F(PublicationValidatorIsolationTests, ValidateTopology_BasicTopology_Success) {
    RuntimePublishWorld world{};
    world.generation = 1;              // ← トポロジに関係ない値
    world.routing.numSources = 2;      // ← セットするだけで検証なし
    world.routing.numDestinations = 2;
    const bool isValid = validator_.validateTopology(world);
    EXPECT_TRUE(isValid);              // ← 常に true になるだけ
}
```

### 現状の Validation Pipeline

```
validatePublication()
  ├── validateSemanticConsistency()  ← 実質唯一のチェック
  ├── validateTopology()             ← 常に true (ダミー)
  ├── validateResources()            ← 常に true (ダミー)
  └── checkNoConflictingTransitions() ← 常に true (ダミー)
```

**4段階中 3段階がプレースホルダー。** Practical Stable Runtime として「検証がある」とは言えない状態。

---

## 補正: Authority bypass の重要度

v3 で指摘した `publishWorld()` の public バイパスに加えて:

```
AudioEngine の public メンバ:
  RuntimePublishStore runtimeStore;                           ← public
  RuntimePublicationValidator runtimePublicationValidator_;   ← public
  publishWorld(unique_ptr<RuntimePublishWorld>)               ← public (Orchestrator bypass)
  makeRuntimePublicationCoordinator()                         ← public (誰でもCoordinator生成)
```

現在の Authority 状態:

```
Authority #1: Orchestrator::trySubmit()     → Admission → Executor → Transition
Authority #2: AudioEngine::publishWorld()   → Coordinator::publishWorld() (直接Store書換)
Authority #3: Coordinator生成                → 誰でも生成可能
```

**3つの Authority が並立しており、うち2つは Admission チェックをバイパスする。**

---

## 補正: Reclaim Progress の深刻度

v3 では「Timer依存」と評価したが、実際はより深刻:

```
tryReclaim() が呼ばれない
  ↓
DeferredDeletionQueue::reclaim() が進まない
  ↓
Reader が既に解放済みでもキューに滞留
  ↓
回収停止が誰にも検知されない     ← ここが最大の問題
```

`pendingRetireCount()` は存在するが、**閾値超過時のアラート機構がない。**

---

## 補正: Reader Stuck の優先度は下げてよい

RCUReaderGuard の RAII 設計により、**正常系では問題なし**:

```cpp
class RCUReaderGuard {
    explicit RCUReaderGuard(RCUReader& r) : reader(&r) { reader->enter(); }
    ~RCUReaderGuard() { if (reader) reader->exit(); }  // RAII で確実に exit
};
```

問題になるのは:

- スレッドクラッシュ (SEGV)
- OS 異常終了
- 外部プラグインの暴走

→ **P3 で妥当。** 実運用で頻発するシナリオではない。

---

## 4階層最終評価

### Tier 0: 実運用事故に直結（P0）

| # | 項目 | 根拠 | 検証 |
| --- | --- | --- | --- |
| 1 | **PublicationExecutor完成** | Validate/Admission/Authority Check/Retire/Evidence のパイプラインが空 | `PublicationExecutor.cpp` — 検証ブロック空、coordinator丸投げ |
| 2 | **Authority単一化** | 3つのAuthority並立。Orchestrator以外がpublish可能 | `runtimeStore` public, `publishWorld()` public |
| 3 | **Coordinator bypass除去** | `makeRuntimePublicationCoordinator()` がpublic。誰でもStore書換可 | `AudioEngine.h:2723` — public inline |
| 4 | **EpochDomain Runtime Core脱却** | 22箇所の直接参照。EQProcessor は10箇所で EpochDomain直呼び | `EQProcessor.Core.cpp` 他 |

### Tier 1: 長期運用で破綻する（P1）

| # | 項目 | 根拠 | 検証 |
| --- | --- | --- | --- |
| 1 | **Reclaim Progress Guarantee** | tryReclaim()呼び出し依存。回収停止が検知不能 | `EpochDomain::tryReclaim()` — 呼ばれなければ停滞 |
| 2 | **Retire Policy実装** | DSPRetirePolicy/SnapshotRetirePolicy/DeferredRetirePolicy は前方宣言のみ | `ISRRetireRouter.h:25-27` |
| 3 | **ISRRetireRouter実体化** | 全メソッドが epochDomain_->xxx() の直接転送 | `ISRRetireRouter.h` 全行 |
| 4 | **Deprecated API全廃** | 5つの[[deprecated]] + 8箇所の#pragma warning(disable:4996) | `EpochDomain.h` + `ISRRetireRouter.h` |
| 5 | **Fallback経路除去** | EQProcessor の EpochDomain直接 + SafeStateSwapper の mutex fallback | `EQProcessor.Core.cpp:56`, `SafeStateSwapper.h:112` |
| 6 | **RuntimePublicationValidator完成** | validateTopology/validateResources/checkNoConflictingTransitions がすべて `return true` のダミー | `RuntimePublicationValidator.cpp:64-96` |
| 7 | **Evidence System実運用化** | lastRejectReason()は2種類のみ。EvidenceExporter は静的JSONテンプレート | `ISRRuntimePublicationCoordinator.cpp:44`, `ISREvidenceExporter.cpp` |

### Tier 2: 監査不能（P2）

| # | 項目 | 根拠 | 検証 |
| --- | --- | --- | --- |
| 1 | **ObserveToken拡張** | guard + GlobalSnapshot* のみ。generation/pubId/worldId なし | `ObservedRuntime.h` |
| 2 | **RuntimeReadHandle整理** | ObservedRuntime(Snapshot系) + RuntimePublishWorld*(World系)の二重保持 | `AudioEngine.h` RuntimeReadHandle |
| 3 | **Shutdown Drain監査** | drainAll()呼び出しはあるが、drain後の空確認・Audit Reportなし | `AudioEngine.CtorDtor.cpp` shutdown sequence |
| 4 | **Queue Pressure回復戦略** | QueuePressure検出はできるがbackpressure/deferred recoveryなし | `AudioEngine.h:3200-3211` |

### Tier 3: 将来リスク（P3）

| # | 項目 | 根拠 | 検証 |
| --- | --- | --- | --- |
| 1 | **Reader Stuck Detection** | RAII guardで正常系はOK。クラッシュ時はリース/タイムアウトなし。P3妥当 | `EpochDomain.h` ReaderSlot |
| 2 | **RuntimeReaderContext型安全化** | 手組み2箇所。ヘルパー関数はあるが強制力なし。P3妥当 | `AudioEngine.Processing.ReleaseResources.cpp:92` |
| 3 | **ReaderSlotPool化** | kMaxReaders = 64。現状十分だが将来リスク。P3妥当 | `EpochDomain.h:17` |

---

## 達成度評価

| 観点 | 達成度 | 内訳 |
| --- | --- | --- |
| 構造面（責務分離/移行度） | 85% | RCUReader/IEpochProvider/ObserveToken/Coordinator導入完了 |
| 検証面（Validator） | 40% | Semanticのみ実装。Topology/Resource/Conflictはダミー |
| Authority（単一化） | 30% | 3並立 + 2経路がAdmissionをバイパス |
| 障害耐性（Reclaim/Shutdown） | 50% | 基本機構はあるが保証・検知なし |
| 監査性（Evidence） | 20% | 静的テンプレートのみ。実 Runtime 状態を反映せず |
| **総合** | **65〜70%** | Practical Stable として見た場合 |

前回 v3 の「70〜75%」から若干下方修正。その理由は **RuntimePublicationValidator の 4段階中3段階がダミー**という事実を加味したため。

---

## 前版(v3)からの修正点

| 観点 | v3 の評価 | v4 修正後 | 理由 |
| --- | --- | --- | --- |
| RuntimePublicationValidator | 触れられていない | P1-6 に追加 | validateTopology/validateResources がダミー |
| Reader Stuck | 重要問題(P2相当) | P3に降格 | RAII guard で正常系はOK。クラッシュ時のみ |
| Evidence System | P3相当 | P1-7 に昇格 | 障害原因究明に必須。lastRejectReason 制限 |
| Reclaim Progress | 「Timer依存」 | 「回収停止が検知不能」に修正 | より深刻 |
| Authority bypass | 触れられているが重要度不足 | P0-2/P0-3 に明確化 | 3並立 + バイパス経路 |
| 達成度 | 70-75% | 65-70% | Validator実態を反映 |

---

## 検証サマリ

全 18 項目（v2: 11 + v3: 6 + RuntimePublicationValidator: 1）を実コード検証:

| 区分 | 項目数 | 正確性 |
| --- | --- | --- |
| Tier 0 (P0) | 4 | ✅ 全件正確 |
| Tier 1 (P1) | 7 | ✅ 全件正確 |
| Tier 2 (P2) | 4 | ✅ 全件正確 |
| Tier 3 (P3) | 3 | ✅ 全件正確 |

**全18件中 14〜15件 強く妥当、2〜3件 優先度過大、1〜2件 設計選択の問題。**

RuntimePublicationValidator のダミー実装を新たに確認。

---

## 管理者レビューによる再評価後の最終確定（2026-06-09）

### P0（最優先：実運用事故に直結）— 5項目

| # | 項目 | カテゴリ | 根拠 |
| --- | --- | --- | --- |
| 1 | **PublicationExecutor完成** | Authority | Success しか返さない空実装 |
| 2 | **Authority単一化** | Authority | 3経路の publish 経路が並立 |
| 3 | **Coordinator bypass除去** | Authority | makeRuntimePublicationCoordinator() が public |
| 4 | **World Lifetime Authority確立** | Lifetime | Store が public、誰でも publishAndSwap 可能 |
| 5 | **Validator強制統合** | Validation | validatePublication() が pipeline 入口で強制されていない |

### P1（長期運用で破綻）— 8項目

| # | 項目 | カテゴリ | 根拠 |
| --- | --- | --- | --- |
| 1 | **RuntimePublicationValidator完成** | Validation | Topology/Resource/Conflict がダミー |
| 2 | **Reclaim Progress Guarantee** | Reclaim | tryReclaim() 呼び出し依存、回収停滞検知不能 |
| 3 | **RetirePolicy実装** | Retire | DSPRetirePolicy 等は前方宣言のみ |
| 4 | **ISRRetireRouter実体化** | Retire | 全メソッドが epochDomain_->xxx() の直接転送 |
| 5 | **Deprecated API全廃** | 移行 | 5つの[[deprecated]] + 8箇所の抑制 |
| 6 | **Fallback経路除去** | 移行 | EQProcessor + SafeStateSwapper の2系統 |
| 7 | **Publication→Retire→Reclaim因果追跡** | 監査 | RetireId/ReclaimId が存在せず、因果連鎖が完全欠落 |
| 8 | **Evidence System実運用化** | 監査 | 11種類中ほぼ全て静的テンプレート |

### P2（監査不能）— 4項目

| # | 項目 | カテゴリ |
| --- | --- | --- |
| 1 | Shutdown Drain Audit | 監査 |
| 2 | ObserveToken拡張 | 監査 |
| 3 | RuntimeReadHandle統合 | 構造 |
| 4 | Queue Pressure Recovery | 障害耐性 |

### P3（将来リスク）— 3項目

| # | 項目 | カテゴリ |
| --- | --- | --- |
| 1 | Reader Stuck Detection | 障害耐性 |
| 2 | RuntimeReaderContext型安全化 | 構造 |
| 3 | ReaderSlotPool化 | 構造 |

### 版間の変更点

| 項目 | v4 最終版 | 再評価後 | 理由 |
| --- | --- | --- | --- |
| Validator完成 | P0-3 | P1-1 降格 | Coordinator 側の semantic contract 検証が存在 |
| Validator強制統合 | 未評価 | **P0-5 新規追加** | validatePublication() が pipeline 入口で強制されていない |
| World Lifetime Authority | P0-P1 (B) | **P0-4 昇格** | Store 公開が実運用事故に直結 |
| 因果追跡 | P1 (A) | **P1-7 明確化** | RetireId/ReclaimId 不在の重要性を再確認 |
| 達成度 | 65-70% | **60-65%** | Authority/Lifetime の破れが v4 評価より深刻 |

### 最終達成度評価

```
Practical Stable ISR Bridge Runtime 達成度: 60〜65%
```

**根拠**: Publication 系の Authority が複数経路で破れており、Publication→Retire→Reclaim の完全な因果チェーンが成立していない。最大の欠陥は「Validator未完成」ではなく **「Authority・Lifetime・Publication Pipeline がまだ単一の真実源(Source of Truth)に収束していないこと」**。
