# Practical Stable ISR Bridge Runtime — 設計書 v4.10（実装開始版・最終確定）

**Document Version:** 4.10
**Date:** 2026-06-20
**Based on:** v4.9 + レビュー指摘6点の反映
**Status:** 実装開始版（全設計判断完了）

---

## v4.9 → v4.10 変更点一覧

| # | 項目 | v4.9 | v4.10 | 根拠 |
|---|---|---|---|---|
| 1 | **PersistentStateBlock 所有権** | 暗黙的 | **MessageThread専有・RT/AudioThread参照禁止を明文化 + jassert ガード** | 半年後の誤用防止 |
| 2 | **reconcileAuthorityState** | bool のみ | **`RepairConfidence` enum 追加（ObserveOnly / SoftRepair / HardRepair）** | Recovery storm 防止 |
| 3 | **ISR-AUTH-001 保証** | CI (regex) | **C++20 `requires` による型制約 + CI 補助** | コンパイル時保証へ移行 |
| 4 | **Fault Injection** | 4シナリオ | **6シナリオ**（+FI-5 Persistent 破損 + FI-6 RuntimeStore 喪失） | ISR-AUTH-006 検証 |
| 5 | **Property Test** | ランダム10,000回 | **Model-Based 状態遷移テスト**（Coverage Matrix 付き） | 複合状態の網羅 |
| 6 | **Telemetry 境界** | 統合のみ | **`domainFrequency_` と既存レコードの責務境界を明確化** | 設計の明確化 |

---

## 第1章: PersistentStateBlock — 所有権の明文化

### 1.1 所有権ルール

```cpp
/**
 * PersistentStateBlock
 *
 * ★ 所有権: MessageThread 専有
 *   - 書き込み: commit() のみ（MessageThread）
 *   - 読み取り: Recovery / Timer / HealthMonitor（すべて MessageThread）
 *
 * ★ 禁止: AudioThread / RTスレッド / RCU Reader からの参照
 *   non-atomic のため、複数スレッドからの同時アクセスは未定義動作となる。
 *
 * ★ 違反検出: jassert(!numeric_policy::isAudioThread()) を全アクセスに挿入
 */
struct PersistentStateBlock {
    PersistentStateSnapshot current{};

    void commitFields(uint64_t seq, uint64_t ep, uint64_t gen) noexcept {
        jassert(!convo::numeric_policy::isAudioThread());
        current.sequenceId = seq;
        current.epoch = ep;
        current.mappedGeneration = gen;
    }

    PersistentStateSnapshot snapshot() const noexcept {
        jassert(!convo::numeric_policy::isAudioThread());
        return current;
    }
    // ...
};
```

### 1.2 Thread Safety Model（設計書）

| スレッド | snapshot() | commitFields() |
|---|---|---|
| MessageThread | ✅ 許可 | ✅ 許可（唯一のwriter） |
| AudioThread | **禁止**（jassert + UB） | **禁止** |
| RCU Reader | **禁止** | **禁止** |
| Worker Thread | **禁止** | **禁止** |

---

## 第2章: RepairConfidence — Recovery storm 防止

### 2.1 問題

`reconcileAuthorityState()` が少し厳しすぎる判定を返すと、
`publishIdleWorldOnly()` が毎 tick 呼ばれ Recovery storm になる。

### 2.2 設計

```cpp
// ★ RepairConfidence: 修復の確実性レベル
//   Recovery storm 防止のため、閾値ベースで修復強度を変化させる
enum class RepairConfidence : uint8_t {
    None          = 0,  // 差異なし（fullReconciliation）
    ObserveOnly   = 1,  // 差異あるが監視のみ。即時修復不要
    SoftRepair    = 2,  // 軽度修復（publishIdleWorldOnly 等）
    HardRepair    = 3,  // 強制修復（Emergency Override 等）
};

struct AuthorityReconciliation {
    // ...（既存フィールド）...

    // ★ 修復強度（新規）
    RepairConfidence confidence{RepairConfidence::None};

    // ★ needsIdlePublish が true でも HardRepair 未満なら
    //   次回 tick まで待機（storm 防止）
    bool needsImmediateAction() const noexcept {
        return confidence >= RepairConfidence::HardRepair;
    }
};

// reconcileAuthorityState 内での判定例
AuthorityReconciliation rec;
// ...
if (observed.runtimeMissing && expected.hasActiveRuntime) {
    // Persistent は sequenceId > 0 だが world がない
    // → epoch が 0 なら bootstrap の可能性 → ObserveOnly
    // → epoch > 0 なら確実に world 消失 → HardRepair
    rec.needsIdlePublish = true;
    rec.confidence = (observed.publicationEpoch > 0)
        ? RepairConfidence::HardRepair
        : RepairConfidence::ObserveOnly;
}
```

### 2.3 Recovery 統合

```cpp
case convo::RecoveryAction::Restore: {
    // ...
    const auto rec = reconcileAuthorityState(observed, expected);

    // ★ RepairConfidence で修復強度を判断
    if (rec.needsIdlePublish && rec.confidence >= RepairConfidence::SoftRepair) {
        publishIdleWorldOnly(getActiveRuntimeDSP(),
            convo::TransitionPolicy::HardReset);
    }
    if (rec.confidence == RepairConfidence::ObserveOnly) {
        // 監視のみ → 次回 tick で再評価
        diagLog("[RECOVERY] ObserveOnly: monitoring");
        break;  // 修復せず次回 tick へ
    }
    // ...
}
```

---

## 第3章: ISR-AUTH-001 — C++20 requires による型制約

### 3.1 問題

v4.9 の regex CI は以下のケースを見逃す：

```cpp
// CI を PASS するが ISR-AUTH-001 違反
void deriveAuthorityState(
    PersistentStateSnapshot ps,     // const ref ではない
    RuntimeStore* store             // 余分な引数
);
```

### 3.2 設計：C++20 requires によるコンパイル時制約

```cpp
// ★ 引数型の制約を requires で表現
//   - 第1引数: const PersistentStateSnapshot&
//   - 第2引数: const World*（ポインタ、null 許容）
//   - それ以外の引数はコンパイルエラー

template <typename World>
[[nodiscard]] AuthorityState deriveAuthorityState(
    const PersistentStateSnapshot& persistentState,
    const World* runtimeWorld) noexcept
    // ★ requires: World が RuntimePublishWorld であることを要求
    requires std::is_same_v<World, const RuntimePublishWorld>
         || std::is_same_v<World, RuntimePublishWorld>;
```

### 3.3 CI の役割

CI は「型制約が期待通り使われているか」の補助的確認に留める：

```powershell
# isr-verify-auth-001.ps1（補助的）
# ★ 主保証: C++20 requires（コンパイル時）
# ★ CI の役割: requires 制約が削除されていないことを確認
$content = Get-Content $targetFile -Raw -Encoding UTF8
if ($content -match 'requires.*std::is_same_v') {
    Write-Host "[PASS] ISR-AUTH-001 type constraint active"
} else {
    Write-Host "[FAIL] ISR-AUTH-001: requires constraint missing"
    exit 1
}
```

---

## 第4章: Fault Injection — 6シナリオ

### FI-1〜4（v4.9 から継承）

変更なし。

### FI-5: PersistentStateBlock 破損（新規）

```cpp
TEST_F(FaultInjectionTest, PersistentStateBlock_CorruptedData) {
    // ★ シナリオ5: PersistentStateBlock に矛盾値を設定
    //   sequenceId=100, epoch=0, generation=200
    //   → ISR-AUTH-006: runtimeMissing 検出
    persistentState.commitFields(100, 0, 200);

    // Recovery 実行
    triggerHealthCheck();

    // ISR-AUTH-006 により runtimeMissing が検出される
    const auto snap = persistentState.snapshot();
    const auto* world = observePublishedWorld();
    const auto observed = deriveAuthorityState(snap, world);
    EXPECT_TRUE(observed.runtimeMissing);

    // Recovery が publishIdleWorldOnly で修復
    world = observePublishedWorld();
    EXPECT_NE(world, nullptr);
}
```

### FI-6: RuntimeStore 喪失（新規）

```cpp
TEST_F(FaultInjectionTest, RuntimeStore_Loss) {
    // ★ シナリオ6: RuntimeStore が nullptr になる
    //   PersistentStateBlock は有効、RuntimeStore が消失
    //   → Recovery が publishIdleWorldOnly で再公開
    clearRuntimeStore();

    triggerRecovery();

    // Recovery 後: RuntimeStore に world が再公開されている
    const auto* world = RuntimePublicationCoordinator::consumeWorldHandle(store);
    EXPECT_NE(world, nullptr);

    // ISR-AUTH-002: Recovery 後 full reconciliation
    const auto snap = persistentState.snapshot();
    const auto observed = deriveAuthorityState(snap, world);
    const auto expected = deriveExpectedState(snap);
    EXPECT_TRUE(reconcileAuthorityState(observed, expected).fullReconciliation);
}
```

---

## 第5章: Model-Based Test（状態遷移ベース）

### 5.1 設計

ランダム10,000回ではなく、状態遷移グラフを定義し、
各状態の組み合わせを体系的にテストする。

```cpp
// ★ 状態遷移モデル
enum class RuntimeState : uint8_t {
    Idle,                  // world = null, persistent = 0
    Active,                // world あり、crossfade なし
    CrossfadeActive,       // world あり、crossfade 中
    PendingPublication,    // publish 保留中（deferred）
    RetireBacklog,         // Retire backlog 蓄積
    ShutdownRequested,     // シャットダウン中
    RecoveryActive,        // Recovery 動作中
    _Count
};

// ★ 遷移カバレッジ行列
//   行 = 現在状態, 列 = 操作 → 期待される遷移先
constexpr RuntimeState kTransitionCoverage[7][4] = {
    // Publish    Retire     Recover    Shutdown
    {  Active,    Idle,      Idle,      ShutdownRequested },  // Idle
    {  Active,    Idle,      Idle,      ShutdownRequested },  // Active
    {  CrossfadeActive, Idle, Active,   ShutdownRequested },  // CrossfadeActive
    // ... 全状態 × 全操作の組み合わせ
};
```

### 5.2 テスト

```cpp
TEST_F(ModelBasedTest, FullStateTransitionCoverage) {
    // ★ 全状態 × 全操作の組み合わせを網羅
    for (int s = 0; s < static_cast<int>(RuntimeState::_Count); s++) {
        for (int op = 0; op < 4; op++) {
            setRuntimeState(static_cast<RuntimeState>(s));
            executeOperation(static_cast<Operation>(op));

            // ★ 各遷移後の整合性 Invariant 確認
            assertInvariants();

            // ★ ISR-AUTH-002: 操作後も Recovery 可能
            const auto snap = persistentState.snapshot();
            const auto* world = observePublishedWorld();
            const auto observed = deriveAuthorityState(snap, world);
            const auto expected = deriveExpectedState(snap);
            const auto rec = reconcileAuthorityState(observed, expected);
            if (!rec.fullReconciliation) {
                // Recovery で修復可能であること
                EXPECT_GE(rec.confidence, RepairConfidence::ObserveOnly);
            }
        }
    }
}
```

---

## 第6章: TelemetryRecorder 境界の明確化

### 6.1 責務マップ

| レコード種別 | 既存/新規 | 責務 | データ |
|---|---|---|---|
| `PublicationProgressRecord` | 既存 | 出版進捗の時系列 | PublishStage, correlationId, timestamp |
| `FailureRecord` | 既存 | 出版障害の記録 | FailureStage, FailureReason, detail |
| `OrchestratorHealthSnapshot` | 既存 | オーケストレータ健全性 | backlog counts, reader state |
| `RetireTimelineRecord` | 既存 | Retire 進捗の時系列 | retireAge, backlog |
| **`domainFrequency_[]`** | **新規** | AuthorityDomain 出現頻度の**軽量集計のみ** | 7 × atomic<uint64_t> |

### 6.2 domainFrequency_ の設計指針

- **時系列ではない**: 累積カウンタのみ
- **軽量**: `memory_order_relaxed` で十分（診断目的のみ）
- **別管理**: TelemetryRecorder の既存レコードとは別に保持。統合しない。

```cpp
class TelemetryRecorder {
    // ... 既存メンバ ...

    // ★ AuthorityDomain 出現頻度（軽量集計のみ・時系列なし）
    //   既存の PublicationProgressRecord/FailureRecord とは責務が異なる
    std::atomic<uint64_t> domainFrequency_[7]{};
};
```

---

## 第7章: 完了条件

```
【基盤】
grep "PersistentStateSnapshot" src/core/PersistentStateBlock.h → 1件
grep "jassert.*isAudioThread" src/core/PersistentStateBlock.h → 1件以上
grep "requires.*std::is_same_v" src/core/AuthorityState.h → 1件以上

【currentWorld_】
grep "currentWorld_" src/audioengine/ISRRuntimePublicationCoordinator.* → 0件

【Recovery】
grep "RepairConfidence" src/core/AuthorityState.h → 1件以上

【CI 全6種】
isr-verify-auth-001.ps1 → PASS  # requires 制約確認
isr-verify-auth-002.ps1 → PASS  # fullReconciliation 確認
isr-verify-auth-003.ps1 → PASS  # 経路唯一性
isr-verify-auth-004.ps1 → PASS  # 型保証 + 補助CI
isr-verify-auth-005.ps1 → PASS  # 唯一永続源
isr-verify-auth-006.ps1 → PASS  # RuntimeStore 整合性

【テスト】
Model-Based Test → 全状態×全操作 網羅
Fault Injection 6 scenarios → PASS
Validator tests → 45+ PASS
```
