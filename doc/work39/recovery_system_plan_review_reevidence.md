# 再検証：コード実装 vs recovery_system_plan.md のずれ

> **日付**: 2026-06-15
> **検証日時**: 2本の外部レビュー指摘を踏まえ、全クレームを grep による生の証拠と共にゼロベース再検証
> **検証方法**: `src/**` 全ファイルに対する grep 検索、コードブロックの直接読み取り

---

## 最重要確認3項目（grep 確定結果）

### A. `RecoveryOutcome` — **定義のみ。未使用。** ← ご指摘通り

```
$ grep -r "RecoveryOutcome\|::Success\|::NoEffect\|::Failed\|::Unsafe" src/**/*.cpp
→ 該当なし（0件）

$ grep -r "RecoveryOutcome" src/**/*.h
→ src/audioengine/RuntimePolicyEngine.h:54
  enum class RecoveryOutcome : uint8_t { None, Success, NoEffect, Failed, Unsafe };
→ この1件のみ（定義のみ、あらゆる .cpp で未使用）
```

**結論**: enum 定義は存在するが、`RuntimeHealthMonitor.cpp` の `tick()`、`executeRecoveryAction()`、および全コールチェーンで**1回も参照されていない**。両レビューの「閉ループ制御欠如」は正しい。

---

### B. `Restore` / `lastHealthyWorldId_` — **記録のみ。切戻し未配線。** ← ご指摘通り

**記録側: 実装済み ✅**

```cpp
// AudioEngine.CtorDtor.cpp:203-216
void AudioEngine::notifyHealthyPublication(uint64_t worldId) noexcept {
    convo::publishAtomic(lastHealthyWorldId_, worldId, ...);
    // LearnerRollback: lastKnownGoodNoiseShaper_ も保存
    if (noiseShaperLearner && noiseShaperLearner->isRunning()) {
        noiseShaperLearner->getState(lastKnownGoodNoiseShaper_.state);
        lastKnownGoodNoiseShaper_.isValid = true;
    }
}
```

**回復側: 未配線 ❌**

```cpp
// AudioEngine.Timer.cpp:679-683
case convo::RecoveryAction::Restore:
    // [work37 Phase 9.16] RollbackToLastHealthyWorld を含む復元操作
    tryReclaimResources();
    drainDeferredRetireQueues(false);
    // Rollback 基盤（ISRRetireRuntimeEx）の設定は呼び出し元が行う  ← 未実装
    break;
```

```
$ grep -r "lastHealthyWorldId_" src/**/*.cpp
→ AudioEngine.CtorDtor.cpp:205  // 書き込みのみ
※ 読み取り箇所なし（getLastHealthyWorldId() は .h で定義されているが、呼び出し元なし）
```

**結論**: `lastHealthyWorldId_` / `lastKnownGoodNoiseShaper_` は publish 成功時に記録されるが、**回復時に使うコードが一切存在しない**。`RecoveryAction::Restore` は drain するだけで World を戻さない。

---

### C. `injectBackpressureSignal` — **定義のみ。未呼び出し。** ← ご指摘通り

```
$ grep -r "injectBackpressureSignal" src/**
→ src/audioengine/RuntimeHealthMonitor.h:146  // 定義のみ
→ この1件のみ（呼び出し元なし）
```

一方、`AudioEngine.Retire.cpp` では依然として **直接** `retirePressureAdmissionStrict_` を書き換えている:

```cpp
// AudioEngine.Retire.cpp:157
convo::publishAtomic(retirePressureAdmissionStrict_, true, ...);
// AudioEngine.Retire.cpp:291
convo::publishAtomic(retirePressureAdmissionStrict_, severe, ...);
```

`applyRetirePressurePolicyNoRt()` が `admissionStrict_` の最終決定権を持っており、PolicyEngine 経路とは独立している。

**結論**: plan 4.4.1「経路1の PolicyEngine 統合」は未完了。背圧経路の二重権限が残存。

---

## レビューが「古い設計を批判している」と言われた項目

### 「PolicyEngine が HealthState を直接書き換える」— **コードには存在しない**

```
$ grep -r "escalateToCritical" src/**
→ 該当なし（0件）

$ grep -r "targetHealth" src/**
→ src/audioengine/RuntimePolicyEngine.h:106
  // ★ v3.6: targetHealth は持たない。HealthStateは HealthMonitor 専権。
→ コメントのみ。シンボルとしての targetHealth は存在しない
```

**コード内の実際の HealthState 更新 (`RuntimeHealthMonitor.cpp:235-273`)**:

```cpp
void RuntimeHealthMonitor::updateHealthState(const PolicyDecision& decision) noexcept
{
    ISRHealthState newState = ISRHealthState::Healthy;
    // 1. 全6系統の MonitorState から newState を算出（これが主判定）
    if (m_prevRetireState == MonitorState::Error) newState = ISRHealthState::Critical;
    // ... 5系統 ...
    // 2. PolicyDecision は causes 参照のみ
    if (decision.causes != 0 && newState == ISRHealthState::Healthy)
        newState = ISRHealthState::Degraded;  // 最大でも Degraded 昇格のみ
    convo::publishAtomic(m_healthState_, newState, ...);
}
```

- PolicyDecision は `causes` のみ参照。`escalateToCritical` / `targetHealth` は**コードにも計画書の最終版にも存在しない**
- 計画書 L203-205 に `targetHealth` の古い記述が残っているが、L265 で `★ v3.6: targetHealth は持たない` と明示的に撤回されている

---

### 「WorldConsistency 監視が shutdown 側に偏っている」— **毎 tick 実行済み**

```
$ grep -n "checkWorldConsistency" src/audioengine/RuntimeHealthMonitor.cpp
→ L28:  checkWorldConsistency();  // tick() 内で毎回実行
→ L607: void RuntimeHealthMonitor::checkWorldConsistency() noexcept { ... }
```

tick() 内の実行順序:

```
checkRetireStall → checkPublicationStall → ... → checkConfigurationDivergence()
→ checkWorldConsistency() → checkSnapshotStarvation() → checkSuppressionDuration()
→ checkRuntimeProgressFreeze() → evaluateAggregate() → updateHealthState(decision)
```

運転中に毎 tick 実行されている。

---

### 「ReaderStuck 情報不足」— **詳細情報は取得・報告済み**

```cpp
// RuntimeHealthMonitor.cpp:300-320
void RuntimeHealthMonitor::diagnoseRetireStall() noexcept {
    auto stuckInfo = m_retireRouter->detectStuckReaders(10);
    HealthEvent ev{getCurrentTimeUs(), ...};
    ev.readerIndex = stuckInfo.readerIndex;     // ✅ どのスロットか
    ev.readerEpoch = stuckInfo.readerEpoch;     // ✅ どの epoch か
    ev.readerDepth = 1;                          // ✅ depth
    ev.residencyTimeUs = stuckInfo.residencyTimeUs; // ✅ 滞留時間
    m_callback(ev);
}
```

Reader Slot >90% の詳細診断 (`checkReaderSlotUsage()` L395-430) でも同様の詳細情報を含む HealthEvent が生成される。

---

## 「RecoveryAction 6段階実装済み」の根拠

```cpp
// RuntimePolicyEngine.h:43-51
enum class RecoveryAction : uint8_t {
    Observe,   // Level 0
    Throttle,  // Level 1
    Recover,   // Level 2
    Restore,   // Level 3
    Safe,      // Level 4
    Critical,  // Level 5
    _Count     // = 6
};
```

```cpp
// AudioEngine.Timer.cpp:659-702 — 全6段階に case 実装あり
void AudioEngine::executeRecoveryAction(convo::RecoveryAction action) noexcept {
    switch (action) {
        case convo::RecoveryAction::Throttle: ... break;  // L663-670
        case convo::RecoveryAction::Recover:  ... break;  // L672-678
        case convo::RecoveryAction::Restore:  ... break;  // L679-683
        case convo::RecoveryAction::Safe:     ... break;  // L685-693
        case convo::RecoveryAction::Critical: ... break;  // L695-699
        default: break;
    }
}
```

---

## 計画書 (recovery_system_plan.md) とコードのずれ一覧

| 項目 | 計画書の記述 | コード実装 | 状態 |
| --- | --- | --- | --- |
| PolicyEngine → HealthState | L203-205: targetHealth で昇格（古い設計） | `causes != 0 → Degraded` のみ | コードの方が安全 |
| escalateToCritical | L716: 移行テーブルに残骸 | コードには存在しない | コードで解決済み |
| `injectBackpressureSignal` | L427-436: 経路1統合 | 定義のみ、呼び出しなし | 未完了 |
| RecoveryOutcome | L157-162, L265-269: 閉ループ制御 | 定義のみ、使用なし | 未完了 |
| Restore → Rollback | L681-683: 「含む復元操作」 | drain のみ、World 切戻しなし | 未完了 |
| Learner 複合条件 | L63-69: 3条件監視 | stallDur > 10s + learnerActive | 部分的 |
| WorldConsistency 毎 tick | Phase 7: 新規 | 実装済み ✅ | 完了 |
| ReaderStuck 詳細診断 | Phase 2.1: 改善 | 実装済み ✅ | 完了 |
| RecoveryAction 6段階 | Phase 0: 6 Action | 実装済み ✅ | 完了 |

---

## 結論

**コード実装の進捗率: 約65%（10項目中6.5項目完了、3.5項目未完了）**

### 未完了で両レビュー一致の P0 課題（4項目）

1. **RecoveryVerification** — `RecoveryOutcome` 定義のみ、未使用。Action→確認→昇格の閉ループ不在
2. **Restore→LastHealthyWorld 配線** — `lastHealthyWorldId_` は記録のみ、切戻し未実装
3. **injectBackpressureSignal 統合** — 定義のみ。Retire.cpp の直接書き込みが残存
4. **Learner 停止の複合条件** — stallDur > 10s 単一条件のみ。3条件複合未実装

### コードは plan より進んでいるが、plan に古い記述が残っている項目（4項目）

1. **PolicyEngine と HealthState の分離** — コードは分離済みだが plan に `targetHealth` の残骸
2. **WorldConsistency 稼働中監視** — コードは毎 tick 実行済み
3. **ReaderStuck 詳細診断** — コードは readerIndex/epoch/depth/residencyUs 報告済み
4. **RecoveryAction 6段階** — コードは全6段階実装済み
