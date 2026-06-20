# Practical Stable ISR Bridge Runtime — 最終設計書 v4.0

**Document Version:** 4.0
**Date:** 2026-06-19
**Based on:** basic_plan.md v2.0 → validation_report.md → design_deep_investigation_report.md
**Status:** 3回の検証サイクル完了。全未確定事項を確定。

---

## 検証プロセスサマリー

| サイクル | 成果物 | 確認事項 |
|---|---|---|
| 1st (v2.0→v3.0) | validation_report.md | 12の「未実装」項目が実際は実装済みと判明 |
| 2nd (v3.0→deep) | design_deep_investigation_report.md | 7つの未確定事項を全ツールで調査し確定 |
| 3rd (本ドキュメント) | **basic_plan.v4.md** | 最終設計。全項目が実装レベルで確定 |

### 使用ツール（6種）

| ツール | 使用方法 | 成果 |
|---|---|---|
| **Serena MCP** (oraios/serena) | find_symbol, find_referencing_symbols, get_symbols_overview | シンボル構造・参照関係の完全把握 |
| **AiDex MCP** (aidex_query) | 278ファイル/12247アイテムのインデックスから高速検索 | currentWorld_, RecoveryAction, publishWorld 等の全出現箇所特定 |
| **CodeGraph MCP** | get_file_structure, query_codebase (67 communities) | ファイル構造把握 |
| **graphify MCP** | god_nodes, get_node | アーキテクチャ中心ノード確認（Masterplan: 325 edges） |
| **semble** (CLI) | semantic search | 自然言語ベース検索（AuthorityTelemetry, Recovery Architecture） |
| **Select-String** (grep) | パターン検索 | 未実装項目の不在確認 |

---

## 第1章: 現状アーキテクチャの完全解明

### 1.1 2つの Coordinator の関係（最重要発見）

ConvoPeq には **2種類の `RuntimePublicationCoordinator`** が存在する：

```
┌──────────────────────────────────────────────────────────┐
│ AudioEngine                                               │
│                                                          │
│  runtimeStore (RuntimeStore) ──── SSOT (world pointer)    │
│       ↑                                                    │
│  makeRuntimePublicationCoordinator()                       │
│       ↓                                                    │
│  RuntimePublicationCoordinator<World,Handle,Bridge>        │
│  (template, core/RuntimePublicationCoordinator.h)          │
│       ↓                                                    │
│  publishAndSwap() → oldWorld → retireRuntimePublishWorld()│
│                                                          │
│  runtimePublicationBridge_ (ISRRuntimePublicationCoord.)   │
│  ─ metadata tracking (backlog, state, pressure)            │
│  ─ currentWorld_ (auxiliary pointer, REDUNDANT)            │
│  ─ commit() / retire() called as side-effect              │
└──────────────────────────────────────────────────────────┘
```

**結論**: `currentWorld_` は `ISRRuntimePublicationCoordinator` 内の補助ポインタ。
真の SSOT は `runtimeStore`（`RuntimeStore` template）。
→ `currentWorld_` の削除は可能であり、削除すべき。

### 1.2 唯一のPublish経路（確定）

```
AudioEngine.Commit.cpp:683  →  Orchestrator::submitPublishRequest()
  → trySubmit() → Admission → Builder → Executor → Coordinator::publishWorld()
     → Bridge.validatePublicationNonRt() → RuntimeStore::publishAndSwap()
     → Bridge.didPublishRuntimeNonRt() → Bridge.willRetireRuntimeNonRt()
     → Bridge.retireRuntimePublishWorldNonRt()

AudioEngine.Timer.cpp:425   →  makeRuntimePublicationCoordinator().publishWorld()
AudioEngine.Transition.cpp:26 → makeRuntimePublicationCoordinator().publishWorld()
PublicationExecutor.cpp:25   →  coordinator.publishWorld()
```

**確認**: DSPTransition/HealthMonitor/CrossfadeRuntime による直接 Publish は存在しない。

---

## 第2章: 拡張設計（Phase-A: 不足機能の実装）

### A-1: PersistentStateBlock 導入

#### ファイル構成

**新規**: `src/core/PersistentStateBlock.h`

```cpp
#pragma once
#include <atomic>
#include <cstdint>
#include "AtomicAccess.h"

namespace convo {

struct PersistentStateBlock {
    std::atomic<uint64_t> publicationSequenceId{0};
    std::atomic<uint64_t> publicationEpoch{0};
    std::atomic<uint64_t> mappedRuntimeGeneration{0};

    struct Snapshot {
        uint64_t sequenceId;
        uint64_t epoch;
        uint64_t mappedGeneration;
    };

    Snapshot snapshot() const noexcept {
        return Snapshot{
            convo::consumeAtomic(publicationSequenceId, std::memory_order_acquire),
            convo::consumeAtomic(publicationEpoch, std::memory_order_acquire),
            convo::consumeAtomic(mappedRuntimeGeneration, std::memory_order_acquire)
        };
    }

    void update(const Snapshot& s) noexcept {
        convo::publishAtomic(publicationSequenceId, s.sequenceId, std::memory_order_release);
        convo::publishAtomic(publicationEpoch, s.epoch, std::memory_order_release);
        convo::publishAtomic(mappedRuntimeGeneration, s.mappedGeneration, std::memory_order_release);
    }
};

} // namespace convo
```

#### 組み込み先

`ISRRuntimePublicationCoordinator.h` の private メンバとして追加：

```cpp
// 変更前
std::atomic<PublicationSequenceId> publicationSequenceId_;
std::atomic<PublicationEpoch> publicationEpoch_;
std::atomic<std::uint64_t> mappedRuntimeGeneration_;

// 変更後
PersistentStateBlock persistentState_;
```

#### 既存APIとの互換性

既存の `commit()` オーバーロードは `PersistentStateBlock::update()` を内部で呼ぶよう変更。
外部からのAPI変更は不要。

---

### A-2: AuthoritySource 導入

#### ファイル構成

**新規**: `src/core/AuthoritySource.h`

```cpp
#pragma once
#include <atomic>
#include <cstdint>
#include "AtomicAccess.h"

namespace convo {

enum class AuthoritySource : uint8_t {
    Unknown        = 0,
    UserAction     = 1,
    PresetLoad     = 2,
    Recovery       = 3,
    DSPTransition  = 4,
    HealthMonitor  = 5,
    _Count
};

struct AuthorityTelemetry {
    std::atomic<uint64_t> callCount[6]{};

    void record(AuthoritySource src) noexcept {
        auto idx = static_cast<size_t>(src);
        if (idx < 6)
            convo::fetchAddAtomic(callCount[idx], 1u, std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t getCount(AuthoritySource src) const noexcept {
        auto idx = static_cast<size_t>(src);
        return (idx < 6)
            ? convo::consumeAtomic(callCount[idx], std::memory_order_acquire)
            : 0;
    }
};

} // namespace convo
```

#### Coordinator API 拡張

`core/RuntimePublicationCoordinator.h` の `publishWorld()` にオプショナル引数追加：

```cpp
[[nodiscard]] PublishStageResult publishWorld(
    convo::aligned_unique_ptr<World> worldOwner,
    AuthoritySource src = AuthoritySource::Unknown) noexcept;
```

内部で `AuthorityTelemetry::record(src)` を呼ぶ。
既存の呼び出し元は変更不要（デフォルト引数で Unknown）。

---

### A-3: deriveAuthorityState() 実装

#### ファイル構成

**新規**: `src/core/AuthorityState.h`

```cpp
#pragma once
#include <cstdint>
#include "PersistentStateBlock.h"

namespace convo {

struct AuthorityState {
    // PersistentStateBlock 由来
    uint64_t publicationSequenceId{0};
    uint64_t publicationEpoch{0};
    uint64_t mappedRuntimeGeneration{0};

    // RuntimeStore 由来
    bool hasActiveRuntime{false};

    // 導出状態
    bool hasPendingPublication{false};
    bool hasActiveCrossfade{false};
};

// Pure Function — 内部状態を一切参照しない
template <typename World>
[[nodiscard]] AuthorityState deriveAuthorityState(
    const PersistentStateBlock::Snapshot& persistentState,
    const World* runtimeWorld) noexcept
{
    AuthorityState result;
    result.publicationSequenceId = persistentState.sequenceId;
    result.publicationEpoch = persistentState.epoch;
    result.mappedRuntimeGeneration = persistentState.mappedGeneration;
    result.hasActiveRuntime = (runtimeWorld != nullptr);
    result.hasPendingPublication =
        (persistentState.sequenceId > 0 && runtimeWorld == nullptr);

    if (runtimeWorld != nullptr) {
        result.hasActiveCrossfade = runtimeWorld->execution.transitionActive;
    }

    return result;
}

} // namespace convo
```

#### Recovery 統合

`AudioEngine.Timer.cpp` の `executeRecoveryAction()` 内での使用：

```cpp
// Restore アクション内（概念コード）
case convo::RecoveryAction::Restore: {
    // Step1: RuntimeStore取得
    const auto* runtimeWorld = observePublishedWorld();
    // Step2: PersistentState取得
    const auto persistentState = persistentState_.snapshot();
    // Step3: AuthorityState導出
    const auto state = deriveAuthorityState(persistentState, runtimeWorld);
    // Step4: 状態比較（期待状態との差異）
    const bool needsIdlePublish = state.hasPendingPublication && !state.hasActiveRuntime;
    // Step5: 不足状態補完
    if (needsIdlePublish) {
        // ...Idle World 発行
    }
    // Step6: Publish再開（既存のRestoreロジック内で実施済み）
    break;
}
```

---

### A-4: currentWorld_ 削除

#### 削除手順（詳細）

**Step 1**: `PersistentStateBlock` 導入（A-1）後、ISR coordinator の3フィールドを置き換え

**Step 2**: `currentWorld_` の使用を1箇所ずつ置き換え

| 置き換え対象 | 変更内容 |
|---|---|
| `commit()` 内の `publishAtomic(currentWorld_, newWorld, ...)` | **削除**（RuntimeStore が既に保持） |
| `retire()` 内の `compareExchangeAtomic(currentWorld_, ...)` | **削除**（RuntimeStore が管理） |
| `getCurrent()` | `RuntimeStore::observe()` に委譲するよう実装変更（ISR coordinator に RuntimeStore 参照を注入） |

**Step 3**: テスト修正（ISRSemanticValidationTests.cpp の 17 箇所）

```cpp
// 変更前
if (coordinator.getCurrent() != &world1)
    return false;

// 変更後（RuntimeStore をテストで保持し、observe を使用）
if (RuntimePublicationCoordinator::consumePublishedWorld(store) != &world1)
    return false;
```

#### RuntimeStore 参照の注入方法

`ISRRuntimePublicationCoordinator` に `const void*` または `RuntimeStore` へのポインタを保持させる案と、
`getCurrent()` を削除してしまい、全テストを `consumePublishedWorld()` に移行させる案がある。

**推奨**: `getCurrent()` を削除し、全テストを `consumePublishedWorld(store)` に移行。
ISR coordinator は Pure Metadata Manager として純化する。

---

## 第3章: 運用設計（既存確認）

### 3.1 Emergency Override（確認完了）

```
発動条件: HealthState::Critical
ファイル: DSPTransition.h:55-75
動作:
  1. HealthState::Critical 検知
  2. activate(newDSP) — 即時 activate
  3. crossfadeRuntime_.complete() — クロスフェード強制完了
  4. retire(oldDSP) — 旧 DSP 退役
  5. incrementEmergencyAbortCount() — カウンタ増加
  6. enqueueHealthEvent(EVENT_CROSSFADE_ABORTED_EMERGENCY, 4003)
```

### 3.2 Recovery Architecture（確認完了）

```
6階層 RecoveryAction:
  Observe  (L0): HealthEvent記録のみ
  Throttle (L1): retirePressureAdmissionStrict_ + suppressionStartUs_
  Recover  (L2): tryReclaimResources + drainDeferredRetireQueues + clearDeferredForShutdown
  Restore  (L3): Epoch Recovery + Learner Rollback + restoreGeneration++
  Safe     (L4): stopNoiseShaperLearning + retirePressureAdmissionStrict_ = false
  Critical (L5): retirePressureAdmissionStrict_ = true + requestEmergencyDrain

閉ループ制御:
  RecoveryOutcome (Improving/Recovered/Stalled/Worsening)
  TrendSnapshot (12フィールド)
  EscalationTracker (ストーム検出、発振検出)
  RecoveryBudget (ウィンドウ管理)
```

### 3.3 RuntimePublicationValidator（確認完了）

```
validatePublication():
  1. validateSemanticConsistency() — generation/sequenceId 不変条件
  2. validateTopology() — runtimeUuid/fading/transition 整合性
  3. validateResources() — Oversampling(1/2/4/8/16) Dither(0/16/24/32) NS(0/1/2/3)
  4. checkNoConflictingTransitions() — SmoothOnly/DryAsOld/HardReset

Validation Telemetry (6000-6003): 実装済み、emitValidationEvent() 完備
```

### 3.4 CrossfadeAuthority（確認完了）

```
evaluate(oldWorld, newWorld, policy) → Decision
  Pure Function (AudioEngine 非依存)
  HealthState Critical 抑制: Orchestrator レベル（evaluate 後、Decision 上書き）
```

---

## 第4章: テスト計画

### 4.1 既存テスト（39ケース）

| Test Group | テスト数 | カバレッジ |
|---|---|---|
| Semantic Consistency | 3 | Success / InvalidExecution / NegativeDryHold |
| Topology | 7 | Basic / NoUuid / FadingMismatch / Bootstrap / Transition付き |
| Resource | 9 | Oversampling / Dither(0,16,24,32) / NoiseShaper(0,2,3) |
| Transition | 8 | SmoothOnly / DryAsOld / HardReset / Inactive / Unknown |
| Publication統合 | 4 | Topology / Resource / Transitionからreject |
| CrossfadeAuthority | 4 | Deterministic / PolicyChange / SameHash / OSChange |
| Coordinator契約 (ISR) | 12 | Monotonicity / Epoch / Generation / Wraparound / Reject保全 |
| Coordinator契約 (Template) | 5 | RejectRepublish / ClearRequiresShutdown |
| **全体** | **~52** | |

### 4.2 追加テスト計画

#### Phase-B-1: Validator エッジケース（優先: 中）

| # | テスト名 | 検証内容 |
|---|---|---|
| 1 | generation>0 + sequenceId==0 → reject | Semantic |
| 2 | generation>0 + runtimeGeneration==0 → reject | Topology |
| 3 | HardReset + fade==0 + useDryAsOld=true → reject | Transition |
| 4 | DryAsOld + fade==0 + useDryAsOld=true → accept | Transition |
| 5 | oversampling=16 → accept | Resource（上限） |
| 6 | dither=0 → accept | Resource（未設定） |
| 7 | noiseShaper=0 → accept | Resource（未設定） |

#### Phase-B-2: Recovery 障害注入テスト（優先: 高）

| # | シナリオ | 注入方法 | 検証内容 |
|---|---|---|---|
| 1 | HealthState::Critical 発動 | HealthMonitor から Critical 状態を設定 | Emergency Override 発動→abortCount増加 |
| 2 | Crossfade Timeout | 長時間 fade で timeout 発生 | publishIdleWorldOnly 呼び出し確認 |
| 3 | Retire Stall | Retire backlog 蓄積 | Throttle→Recover の昇格確認 |
| 4 | Publication Stall | sequenceId 停止 | 閉ループ制御の昇格確認 |

#### Phase-B-3: Property Test（優先: 中）

```cpp
// 10,000回ランダム publish シーケンス
for (int i = 0; i < 10000; i++) {
    auto world = createRandomWorld(rng);
    auto result = coordinator.publishWorld(std::move(world));
    // 単調増加契約を検証
    assert(result == PublishStageResult::Success
        || result == PublishStageResult::Rejected);
    // Reject 後も直前の world が維持されていることを確認
    if (result == PublishStageResult::Rejected) {
        assert(consumePublishedWorld(store) == lastPublished);
    }
}
```

---

## 第5章: 実装順序と依存関係

### Phase-A 実装順序（依存関係あり）

```
A-1: PersistentStateBlock
     └─ 依存: なし（独立して導入可能）

A-2: AuthoritySource
     └─ 依存: なし（独立して導入可能）

A-3: deriveAuthorityState
     └─ 依存: A-1（PersistentStateBlock の型が必要）

A-4: currentWorld_ 削除
     └─ 依存: A-1（PersistentStateBlock 導入後、ISR coordinator の純化）

A-5: Recovery 統合（deriveAuthorityState 接続）
     └─ 依存: A-3（deriveAuthorityState が必要）
```

### Phase-B 実装（Phase-A と並行可能）

```
B-1: Validator エッジケース追加 → 依存: なし
B-2: Recovery 障害注入テスト → 依存: A-5（Recovery 統合後）
B-3: Property Test → 依存: なし
```

### 推奨着手順

```
Week 1: A-1 + A-2（並行、独立）
Week 2: A-3（A-1依存）+ B-1（並行）
Week 3: A-4 + A-5（A-1/A-3依存）+ B-3（並行）
Week 4: B-2（A-5依存）+ 全体検証
```

---

## 第6章: 完了条件と検証方法

### 完了条件

| 条件 | 確認方法 |
|---|---|
| PersistentStateBlock 導入 | `aidex_query PersistentStateBlock` で確認 |
| AuthoritySource 導入 | `aidex_query AuthoritySource` で確認 |
| currentWorld_ 削除 | `aidex_query currentWorld_` で0件 |
| deriveAuthorityState 実装 | `aidex_query deriveAuthorityState` で確認 |
| Recovery 統合済み | Restore アクション内で deriveAuthorityState 呼び出し確認 |
| Validator テスト 45+ | `Select-String TEST_F\|TEST\(` でカウント |
| 障害注入テスト 4+ | 全シナリオパス確認 |
| Property Test 10,000+ | 全ランダムシーケンスパス確認 |

### 検証コマンド

```powershell
# 削除確認
Select-String -Path "src\audioengine\ISRRuntimePublicationCoordinator.h","src\audioengine\ISRRuntimePublicationCoordinator.cpp" -Pattern "currentWorld_"

# 導入確認
Select-String -Path "src\core" -Pattern "PersistentStateBlock|AuthoritySource|AuthorityState|deriveAuthorityState"

# テストカウント
Select-String -Path "src\tests\*.cpp" -Pattern "TEST_F|TEST\(" | Measure-Object -Line
```
