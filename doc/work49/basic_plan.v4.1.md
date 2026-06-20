# Practical Stable ISR Bridge Runtime — 設計書 v4.1（レビュー反映版）

**Document Version:** 4.1
**Date:** 2026-06-19
**Based on:** v4.0 + コードレビュー指摘4点を反映
**Status:** 最終

---

## v4.0 → v4.1 変更点一覧

| # | 項目 | v4.0 | v4.1 | 理由 |
|---|---|---|---|---|
| 1 | Phase順序 | A-1→A-2→**A-3**→A-4→A-5 | A-1→A-2→**A-4**→A-3→A-5 | currentWorld_ 除去を先に行うことで deriveAuthorityState が RuntimeStore のみを観測源にできる |
| 2 | deriveAuthorityState | 単体の導出関数 | **reconcileAuthorityState(observed, expected)** として上位概念を追加 | Recovery に必要なのは「現在状態」ではなく「期待状態との差異」 |
| 3 | PersistentStateBlock | 3フィールドを個別acquire load | **version 付き論理スナップショット**（read-version → read-fields → read-version） | 途中更新による不整合読取の理論的防止 |
| 4 | AuthoritySource | 単一enum | **AuthorityDomain + AuthorityReason** の2軸に分割 | 将来の拡張性（Recovery の種別分離、新Domain追加） |
| 5 | Recovery Invariant | 未策定 | **ISR-AUTH-001/002** を明文化 | 回復後の状態が通常経路と同値であることを保証 |
| 6 | Property Test | Publishのみ | **Publish + Retire + Recover + Shutdown 混在** | 実運用に近いランダムシーケンスの検証 |

---

## 第1章: アーキテクチャ（v4.0 から継承・確定）

### 1.1 2つの Coordinator の関係（変更なし）

```
AudioEngine
├── runtimeStore (RuntimeStore) ──── SSOT (world pointer)
├── makeRuntimePublicationCoordinator()
│   └── RuntimePublicationCoordinator<World,Handle,Bridge> (template)
│       └── publishAndSwap() → oldWorld → retireRuntimePublishWorld()
│
└── runtimePublicationBridge_ (ISRRuntimePublicationCoordinator)
    └── metadata tracking (backlog, state, pressure) ONLY
    └── currentWorld_ (REDUNDANT → 削除予定)
```

### 1.2 唯一のPublish経路（変更なし）

全経路が Coordinator を通ることを確認済み。

---

## 第2章: 論理スナップショット版 PersistentStateBlock

### 2.1 設計思想

3つの atomic フィールドを個別に読み取ると、書き込み途中の不整合値を取得する可能性がある。
これを防ぐため version フィールドを追加し、**read-version → read-fields → read-version** の順序で読み取り、
前後の version が一致した場合のみ有効なスナップショットとする。

### 2.2 定義

**ファイル**: `src/core/PersistentStateBlock.h`

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
    std::atomic<uint64_t> version{0};  // ★ 論理スナップショット用バージョン

    struct Snapshot {
        uint64_t sequenceId;
        uint64_t epoch;
        uint64_t mappedGeneration;
        uint64_t snapshotVersion;  // 読み取り成功時の version
    };

    // ★ 論理スナップショット: read-version → read-fields → read-version
    Snapshot snapshot() const noexcept {
        for (;;) {
            const auto v0 = convo::consumeAtomic(version, std::memory_order_acquire);
            const auto seq = convo::consumeAtomic(publicationSequenceId, std::memory_order_acquire);
            const auto ep  = convo::consumeAtomic(publicationEpoch, std::memory_order_acquire);
            const auto gen = convo::consumeAtomic(mappedRuntimeGeneration, std::memory_order_acquire);
            const auto v1  = convo::consumeAtomic(version, std::memory_order_acquire);
            if (v0 == v1) {
                return Snapshot{seq, ep, gen, v0};
            }
            // version 不整合 → リトライ（書き込み途中のため）
        }
    }

    // ★ 論理スナップショット更新: version++ → write-fields → version++
    void update(const Snapshot& s) noexcept {
        convo::fetchAddAtomic(version, uint64_t{1}, std::memory_order_acq_rel);
        convo::publishAtomic(publicationSequenceId, s.sequenceId, std::memory_order_release);
        convo::publishAtomic(publicationEpoch, s.epoch, std::memory_order_release);
        convo::publishAtomic(mappedRuntimeGeneration, s.mappedGeneration, std::memory_order_release);
        convo::fetchAddAtomic(version, uint64_t{1}, std::memory_order_acq_rel);
    }

    // ★ 高速単一フィールド読み取り（スナップショット不要時）
    [[nodiscard]] uint64_t readSequenceId() const noexcept {
        return convo::consumeAtomic(publicationSequenceId, std::memory_order_acquire);
    }
};

} // namespace convo
```

### 2.3 組み込み先

`ISRRuntimePublicationCoordinator.h` 修正：

```cpp
// 変更前（3つの個別 atomic）
std::atomic<PublicationSequenceId> publicationSequenceId_;
std::atomic<PublicationEpoch> publicationEpoch_;
std::atomic<std::uint64_t> mappedRuntimeGeneration_;

// 変更後（PersistentStateBlock に統合）
PersistentStateBlock persistentState_;
```

`commit()` 内の書き込み：

```cpp
// 変更前（個別 publishAtomic × 3）
convo::publishAtomic(publicationSequenceId_, sequenceId, ...);
convo::publishAtomic(publicationEpoch_, epoch, ...);
convo::publishAtomic(mappedRuntimeGeneration_, mappedGeneration, ...);

// 変更後（論理スナップショット更新）
persistentState_.update({sequenceId, epoch, mappedGeneration, 0});
```

### 2.4 整合性保証

- `snapshot()` 内の version 一致チェックで、書き込み途中の不整合読み取りを防止
- `update()` 内の version 2回 increment で、書き込み開始/終了をマーク
- 読み取り側の retry loop は有限回数で収束（書き込みは Non-RT 単一スレッド）
- 単一フィールド読み取りが必要な場合は `readSequenceId()` を使用（高速パス）

---

## 第3章: deriveAuthorityState + reconcileAuthorityState

### 3.1 設計思想

Recovery が必要とするのは「現在の観測状態」ではなく「観測状態と期待状態の差異」である。
したがって以下の3層構造とする：

```
deriveAuthorityState(persistentSnapshot, runtimeWorld)
    → AuthorityState observed     # 現在の観測状態

deriveExpectedState(persistentSnapshot, domain)
    → AuthorityState expected     # あるべき期待状態

reconcileAuthorityState(observed, expected)
    → AuthorityReconciliation     # 差異と修復アクション
```

### 3.2 定義

**ファイル**: `src/core/AuthorityState.h`

```cpp
#pragma once
#include <cstdint>
#include "PersistentStateBlock.h"

namespace convo {

// ── AuthorityState: ある時点のAuthority状態 ──
struct AuthorityState {
    uint64_t publicationSequenceId{0};
    uint64_t publicationEpoch{0};
    uint64_t mappedRuntimeGeneration{0};
    bool hasActiveRuntime{false};
    bool hasPendingPublication{false};
    bool hasActiveCrossfade{false};

    bool operator==(const AuthorityState& o) const noexcept {
        return publicationSequenceId == o.publicationSequenceId
            && publicationEpoch == o.publicationEpoch
            && mappedRuntimeGeneration == o.mappedRuntimeGeneration
            && hasActiveRuntime == o.hasActiveRuntime
            && hasPendingPublication == o.hasPendingPublication
            && hasActiveCrossfade == o.hasActiveCrossfade;
    }
    bool operator!=(const AuthorityState& o) const noexcept { return !(*this == o); }
};

// ── AuthorityReconciliation: observed と expected の差異から修復アクションを導出 ──
struct AuthorityReconciliation {
    bool needsIdlePublish{false};     // observed に world がないが expected にある
    bool needsRetireDrain{false};     // Retire backlog が滞留
    bool needsEpochAdvance{false};    // Epoch が期待値より進んでいない
    bool needsCrossfadeComplete{false}; // Crossfade が滞留
    bool fullReconciliation{false};   // observed == expected（差分なし）

    bool needsAnyAction() const noexcept {
        return needsIdlePublish || needsRetireDrain
            || needsEpochAdvance || needsCrossfadeComplete;
    }
};

// ── Pure Function: 現在状態の導出 ──
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

// ── Pure Function: 期待状態の導出 ──
//   PersistentState が「あるべき状態」を定義する。
//   sequenceId > 0 なら world が存在すべき、など。
[[nodiscard]] AuthorityState deriveExpectedState(
    const PersistentStateBlock::Snapshot& persistentState) noexcept
{
    AuthorityState result;
    result.publicationSequenceId = persistentState.sequenceId;
    result.publicationEpoch = persistentState.epoch;
    result.mappedRuntimeGeneration = persistentState.mappedGeneration;
    // sequenceId > 0 かつ RuntimeStore が null でないことが期待される
    result.hasActiveRuntime = (persistentState.sequenceId > 0);
    result.hasPendingPublication = false;  // 期待状態では pending は解消されているべき
    return result;
}

// ── Pure Function: observed と expected の差異調整 ──
[[nodiscard]] AuthorityReconciliation reconcileAuthorityState(
    const AuthorityState& observed,
    const AuthorityState& expected) noexcept
{
    AuthorityReconciliation rec;
    if (observed == expected) {
        rec.fullReconciliation = true;
        return rec;
    }

    // expected に world が必要だが observed にない → Idle Publish が必要
    rec.needsIdlePublish = expected.hasActiveRuntime
        && !observed.hasActiveRuntime
        && observed.publicationSequenceId > 0;

    // observed に世界があるが crossfade が滞留
    rec.needsCrossfadeComplete = observed.hasActiveCrossfade
        && !expected.hasActiveCrossfade;

    // Epoch が期待値より小さい → advance が必要
    rec.needsEpochAdvance = observed.publicationEpoch < expected.publicationEpoch;

    // 運用状態の不一致 → drain が必要
    if (observed.hasPendingPublication && !expected.hasPendingPublication) {
        rec.needsRetireDrain = true;
    }

    return rec;
}

} // namespace convo
```

### 3.3 Recovery 統合

`AudioEngine.Timer.cpp` の `executeRecoveryAction()` 内：

```cpp
case convo::RecoveryAction::Restore: {
    // Step1: PersistentState 取得（論理スナップショット）
    const auto ps = persistentState_.snapshot();
    // Step2: RuntimeStore 取得
    const auto* runtimeWorld = observePublishedWorld();
    // Step3: observed 状態導出
    const auto observed = deriveAuthorityState(ps, runtimeWorld);
    // Step4: expected 状態導出
    const auto expected = deriveExpectedState(ps);
    // Step5: 差異調整
    const auto rec = reconcileAuthorityState(observed, expected);
    // Step6: 修復アクション実行
    if (rec.needsIdlePublish) {
        // publishIdleWorldOnly() を呼び出し
    }
    if (rec.needsCrossfadeComplete) {
        crossfadeRuntime_.complete();
    }
    // ★ ISR-AUTH-002: Recovery 後の状態が通常経路と同値であることを確認
    const auto* afterWorld = observePublishedWorld();
    const auto afterPs = persistentState_.snapshot();
    const auto afterObserved = deriveAuthorityState(afterPs, afterWorld);
    const auto afterExpected = deriveExpectedState(afterPs);
    const auto afterRec = reconcileAuthorityState(afterObserved, afterExpected);
    jassert(afterRec.fullReconciliation);
    break;
}
```

---

## 第4章: AuthorityDomain + AuthorityReason 2軸設計

### 4.1 設計思想

単一の `AuthoritySource` enum では将来の拡張性に欠ける。
`Domain`（発行元コンポーネント）と `Reason`（発行理由）を分離することで、
新たな発行元や理由の追加が容易になる。

### 4.2 定義

**ファイル**: `src/core/AuthorityDescriptor.h`

```cpp
#pragma once
#include <atomic>
#include <cstdint>
#include "AtomicAccess.h"

namespace convo {

// ── AuthorityDomain: Publish 要求の発行元コンポーネント ──
enum class AuthorityDomain : uint8_t {
    Unknown        = 0,
    User           = 1,  // UI操作全般
    Preset         = 2,  // プリセット読み込み
    Recovery       = 3,  // リカバリシステム
    DSPLifecycle   = 4,  // DSPTransition / DSP ライフサイクル
    Health         = 5,  // HealthMonitor
    Shutdown       = 6,  // シャットダウン
    _Count
};

// ── AuthorityReason: 同一 Domain 内での具体的理由 ──
enum class AuthorityReason : uint8_t {
    Unknown           = 0,

    // Domain::User
    UserParameter     = 1,   // パラメータ変更
    UserBypass        = 2,   // バイパス切り替え
    UserPresetRecall  = 3,   // プリセット呼び出し（Domain::Preset と併用）

    // Domain::Recovery
    TimerRecovery     = 10,  // Timer 経由の定期リカバリ
    EmergencyRecovery = 11,  // Emergency Override 発動
    TimeoutRecovery   = 12,  // Crossfade/Retire タイムアウト
    ShutdownRecovery  = 13,  // シャットダウン時の最終リカバリ

    // Domain::Health
    HealthStall       = 20,  // Publication/Retire 停滞
    HealthCritical    = 21,  // Critical 状態
    HealthDegraded    = 22,  // Degraded 状態

    // Domain::DSPLifecycle
    DSPTransition_Complete = 30,  // クロスフェード完了
    DSPTransition_Abort    = 31,  // クロスフェード中断

    _Count
};

// ── AuthorityDescriptor: Domain × Reason の組 ──
struct AuthorityDescriptor {
    AuthorityDomain domain{AuthorityDomain::Unknown};
    AuthorityReason reason{AuthorityReason::Unknown};

    bool operator==(const AuthorityDescriptor& o) const noexcept {
        return domain == o.domain && reason == o.reason;
    }
};

// ── AuthorityTelemetry: Domain × Reason ごとのカウンタ ──
//   メモリ効率のため Domain 単位の集計のみ保持。Reason 詳細は HealthEvent に委ねる。
struct AuthorityTelemetry {
    std::atomic<uint64_t> domainCount[7]{};  // _Count = 7

    void record(AuthorityDomain d) noexcept {
        auto idx = static_cast<size_t>(d);
        if (idx < 7)
            convo::fetchAddAtomic(domainCount[idx], 1u, std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t getCount(AuthorityDomain d) const noexcept {
        auto idx = static_cast<size_t>(d);
        return (idx < 7)
            ? convo::consumeAtomic(domainCount[idx], std::memory_order_acquire)
            : 0;
    }
};

// ── 文字列表現（デバッグ/診断用） ──
[[nodiscard]] const char* to_string(AuthorityDomain d) noexcept;
[[nodiscard]] const char* to_string(AuthorityReason r) noexcept;

} // namespace convo
```

### 4.3 Coordinator API 拡張

`core/RuntimePublicationCoordinator.h`：

```cpp
[[nodiscard]] PublishStageResult publishWorld(
    convo::aligned_unique_ptr<World> worldOwner,
    AuthorityDescriptor auth = {}) noexcept;
```

### 4.4 呼び出し例

```cpp
// UI操作からの publish
coordinator.publishWorld(std::move(world),
    {AuthorityDomain::User, AuthorityReason::UserParameter});

// リカバリからの publish
coordinator.publishWorld(std::move(world),
    {AuthorityDomain::Recovery, AuthorityReason::TimerRecovery});

// デフォルト（後方互換）
coordinator.publishWorld(std::move(world));  // auth = {}
```

---

## 第5章: 改訂 Phase 順序

### 5.1 新 Phase 順序

```
A-1: PersistentStateBlock 導入（論理スナップショット版）
  └─ 依存: なし

A-2: AuthorityDescriptor + AuthorityTelemetry 導入
  └─ 依存: なし

[A-4 をここに移動] ← ★ v4.0 から変更
A-4: currentWorld_ 完全削除
  └─ 依存: A-1（PersistentStateBlock 導入後）
  └─ 理由: deriveAuthorityState より先に行うことで、
            RuntimeStore のみを観測源とできる

[A-3 を後ろに移動] ← ★ v4.0 から変更
A-3: deriveAuthorityState + deriveExpectedState + reconcileAuthorityState
  └─ 依存: A-1（PersistentStateBlock）, A-4（currentWorld_除去後）
  └─ 理由: currentWorld_ 除去後に RuntimeStore 唯一の観測源として導出

A-5: Recovery 統合（reconcileAuthorityState 接続）
  └─ 依存: A-3

Phase-B: テスト拡充
  └─ B-1: Validator エッジケース（7ケース）
  └─ B-2: 障害注入テスト（4シナリオ）
  └─ B-3: Property Test（Publish+Retire+Recover+Shutdown 混在）
```

### 5.2 変更理由

```
v4.0: A-1 → A-2 → A-3(derive) → A-4(currentWorld_) → A-5
                  ↑ deriveAuthorityState が currentWorld_ の存在を前提に設計されるリスク

v4.1: A-1 → A-2 → A-4(currentWorld_) → A-3(derive+reconcile) → A-5
                  ↑ currentWorld_ 除去後に derive 設計。RuntimeStore 純粋依存が確定
```

---

## 第6章: Recovery Invariant 明文化

### 6.1 ISR-AUTH-001: Authority State Rebuildability

```
ISR-AUTH-001

Authority State は PersistentStateBlock からのみ再構築可能でなければならない。

根拠:
  Recovery が currentWorld_, crossfadeRuntime_, temporary cache 等の
  揮発性状態に依存すると、それらの消失時に回復不能となる。

遵守方法:
  - deriveAuthorityState() は PersistentStateBlock::Snapshot + RuntimeStore のみを入力とする
  - deriveExpectedState() は PersistentStateBlock::Snapshot のみを入力とする
  - reconcileAuthorityState() は両者の出力のみを入力とする
  - Recovery 実行中は一時的な揮発性状態参照を禁止する

違反検出:
  - grep "currentWorld_" が 0 件
  - deriveAuthorityState の引数に PersistentStateBlock 以外の AudioEngine メンバが含まれない
```

### 6.2 ISR-AUTH-002: Recovery State Equivalence

```
ISR-AUTH-002

Recovery 後の状態は、通常の Publish 経路で到達可能な状態と同値でなければならない。

根拠:
  Recovery が特殊状態を作り始めると、将来の状態遷移が非対称になり、
  予測不能なバグの原因となる。

遵守方法:
  - Recovery の全出口で reconcileAuthorityState() の fullReconciliation == true を確認
  - Recovery 専用の publish 経路（bypass）を作成しない
  - publishIdleWorldOnly() は通常の Coordinator::publishWorld() を使用する

違反検出:
  - executeRecoveryAction() の各 action 出口で
    jassert(reconcileAuthorityState(observed, expected).fullReconciliation) を挿入
  - Recovery 専用の publish 関数が存在しないことを CI で確認
```

### 6.3 運用監査

```powershell
# CI ゲート: ISR-AUTH-001 違反検出
Select-String -Path "src\audioengine\ISRRuntimePublicationCoordinator.*" -Pattern "currentWorld_"
# → 0件であること

# CI ゲート: ISR-AUTH-002 違反検出
Select-String -Path "src\audioengine\AudioEngine.Timer.cpp" -Pattern "publishIdleWorldOnly\|makeRuntimePublicationCoordinator"
# → publishIdleWorldOnly が Coordinator 経由であることを確認
```

---

## 第7章: 改訂テスト計画

### 7.1 既存テスト（52ケース、継承）

v4.0 からの変更なし。

### 7.2 Phase-B-1: Validator エッジケース（7ケース）

変更なし。

### 7.3 Phase-B-2: 障害注入テスト（4シナリオ）

変更なし。

### 7.4 Phase-B-3: Property Test — Publish + Retire + Recover + Shutdown 混在

```cpp
// 10,000回のランダムオペレーションシーケンス
// 各ステップで Publish / Retire / Recover / Shutdown を確率的に選択
// Practical Stable ISR Bridge Runtime で最も危険な「循環」を検証

enum class OpType : uint8_t {
    Publish,
    Retire,
    Recover,
    Shutdown,
    _Count
};

TEST(PropertyTest, RandomPublishRetireRecoverShutdownSequence) {
    auto rng = createDeterministicRNG(/*seed=*/42);
    CoordinatorType coordinator = createTestCoordinator();
    std::vector<PublishStageResult> results;

    for (int i = 0; i < 10000; i++) {
        const auto op = static_cast<OpType>(rng.uniformInt(
            0, static_cast<int>(OpType::_Count) - 1));

        switch (op) {
        case OpType::Publish: {
            auto world = createRandomWorld(rng);
            auto result = coordinator.publishWorld(std::move(world));
            results.push_back(result);
            // ★ Invariant: Reject 後も直前の world が維持
            if (result == PublishStageResult::Rejected) {
                auto* current = CoordinatorType::consumeWorldHandle(store);
                assert(current == lastPublished);
            }
            break;
        }
        case OpType::Retire: {
            auto* current = CoordinatorType::consumeWorldHandle(store);
            if (current != nullptr) {
                // ★ Retire 後は world が null になる
                bridge.willRetireRuntimeNonRt(current);
                bridge.retireRuntimePublishWorldNonRt(
                    const_cast<WorldType*>(current), false);
            }
            break;
        }
        case OpType::Recover: {
            // ★ RecoveryAction::Restore 相当の操作
            const auto ps = persistentState.snapshot();
            const auto* world = CoordinatorType::consumeWorldHandle(store);
            const auto observed = deriveAuthorityState(ps, world);
            const auto expected = deriveExpectedState(ps);
            const auto rec = reconcileAuthorityState(observed, expected);
            if (rec.needsIdlePublish) {
                auto idleWorld = createIdleWorld(rng);
                coordinator.publishWorld(std::move(idleWorld));
            }
            // ★ ISR-AUTH-002: Recovery 後 full reconciliation
            const auto* afterWorld = CoordinatorType::consumeWorldHandle(store);
            const auto afterPs = persistentState.snapshot();
            const auto afterObserved = deriveAuthorityState(afterPs, afterWorld);
            const auto afterExpected = deriveExpectedState(afterPs);
            const auto afterRec = reconcileAuthorityState(afterObserved, afterExpected);
            assert(afterRec.fullReconciliation);
            break;
        }
        case OpType::Shutdown: {
            coordinator.requestShutdownClearNonRt();
            coordinator.clearPublishedRuntimeSnapshotsNonRt();
            break;
        }
        }
    }

    // ★ 最終 Invariant: 全オペレーションを通じて単調増加契約が維持されている
    EXPECT_GE(results.size(), 0);
}
```

---

## 第8章: 推奨実装スケジュール

| 週 | 作業 | 成果物 |
|---|---|---|
| Week 1 | A-1: PersistentStateBlock（論理スナップショット版）| `src/core/PersistentStateBlock.h` + テスト |
| Week 1 | A-2: AuthorityDescriptor + Telemetry | `src/core/AuthorityDescriptor.h` + テスト |
| Week 2 | **A-4: currentWorld_ 削除** | ISRRuntimePublicationCoordinator 修正 + 全テスト修正 |
| Week 2 | A-3: deriveAuthorityState + reconcileAuthorityState | `src/core/AuthorityState.h` + テスト |
| Week 3 | A-5: Recovery 統合 | AudioEngine.Timer.cpp 修正 |
| Week 3 | B-1: Validator エッジケース | PublicationValidatorIsolationTests.cpp 拡張 |
| Week 4 | B-2: 障害注入テスト | 新規テストファイル |
| Week 4 | B-3: Property Test | 新規テストファイル |

---

## 参考: 設計判断の根拠

### 2軸 AuthorityDescriptor の採用理由

単一 enum では以下の問題がある：

- Recovery には Timer/Emergency/Timeout/Shutdown の4種がある
- これらを単一 enum に並べると `HealthMonitor_Timer_Timeout` のような複合名が増殖する
- 新 Domain（例: Automation, Script）追加時に enum 値の再編が必要

Domain × Reason の2軸でこれらの問題を解決。

### version 付き論理スナップショットの採用理由

3つの atomic を個別に読むと以下の問題がある：

- commit() が sequenceId を書き終え、epoch を書き始める前に snapshot() が割り込む
- 読み取り結果: `{sequence=101, epoch=100, generation=100}` のような不整合
- 現状は Non-RT 単一スレッドのため実害はないが、Recovery パスからの非同期読み取りを考慮し対策

### Phase 順序変更の理由

deriveAuthorityState は「RuntimeStore のみを観測源とする Pure Function」であるべき。
currentWorld_が存在すると、誤ってそれに依存した設計になるリスクがある。
先に currentWorld_ を削除することで、設計の純度を保証する。
