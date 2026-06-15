# 第二回 外部レビュー コード突合検証報告

> **日付**: 2026-06-15
> **検証方法**: doc/work37/recovery_system_plan.md (v7.1最終版) + 実装コード 全ファイル突合
> **検証範囲**: RuntimeHealthMonitor.h/.cpp, RuntimePolicyEngine.h/.cpp, AudioEngine.Timer.cpp, AudioEngine.Retire.cpp, AudioEngine.h, AudioEngine.CtorDtor.cpp, RefCountedDeferred.h, SnapshotCoordinator.h/.cpp, DSPLifetimeManager.h, DeferredRetireFallbackQueue.h, RuntimeDrainAudit.h
> **焦点**: リーク/Shutdown/HealthMonitor ではなく、**「音が壊れたまま戻らない」実障害**の観点

---

## 0. 前提

本レビューは前回より実用的で、recovery_system_plan v7.1 の現状に比較的近い分析をしている。ただし依然として一部の指摘はコード実装上既に対応済みであり、また一部は実装が不完全な状態である。

以下の各項目について、「レビューの主張」「コード実装状態」「検証結果」の3点で整理する。

---

## P0-1: Retire Stall Recovery Ladder

### レビューの主張 (P0-1)

enum RecoveryPhase { None, ReclaimRetry, FlushDeferredPublish, ReleaseDeferredStructural, PublishLastKnownGood, AdmissionStop } のような段階的リカバリーラダーが無い。改善確認→効果なければ次段階 の機構が無い。

### コード実装状態 (P0-1)

`RecoveryAction` は6段階として既に実装済み:

```text
Observe (L0) → Throttle (L1) → Recover (L2) → Restore (L3) → Safe (L4) → Critical (L5)
```

各 Action の実装内容 (`executeRecoveryAction()` in `AudioEngine.Timer.cpp`):

| Action | 実装内容 | レビューの要求との対応 |
| --- | --- | --- |
| Throttle | `admissionStrict_ = true` | (抑制フェーズ) |
| Recover | `tryReclaimResources()` + `drainDeferredRetireQueues()` + `clearDeferredForShutdown()` | ReclaimRetry + FlushDeferredPublish |
| Restore | `tryReclaimResources()` + `drainDeferredRetireQueues()` (Rollback未配線) | PublishLastKnownGood (未配線) |
| Safe | `stopNoiseShaperLearning()` + `admissionStrict_ = false` | (安全確保) |
| Critical | `admissionStrict_ = true` + `requestEmergencyDrain()` | AdmissionStop |

### 検証結果 (P0-1): ⚠️ 部分的に妥当

**ラダーの段階自体は既に存在する。しかし「逐次エスカレーション」が無い。**

- `RuntimePolicyEngine::evaluateAggregate()` は毎 tick 独立して MonitorState → RecoveryAction を選択
- Cooldown 制御はあるが、「Throttle 実行 → 5秒後確認 → 改善なければ Recover」という**閉ループ逐次昇格の保証がない**
- `RecoveryOutcome` (Success/NoEffect/Failed) は `RuntimePolicyEngine.h` で定義されているが、`RuntimeHealthMonitor.cpp` では**全く使用されていない**

**核心の欠落**: `executeRecoveryAction()` の後、その Action が有効だったかを確認し、無効なら次段階へ昇格する機構がない。

---

## P0-2: Rebuild Suppression が片方向

### レビューの主張 (P0-2)

`retirePressureAdmissionStrict_` の解除条件が弱い。「いつ解除するのか」が見えない。

### コード実装状態 (P0-2)

**解除経路1**: `applyRetirePressurePolicyNoRt()` (`AudioEngine.Retire.cpp:278-315`)

```cpp
convo::publishAtomic(retirePressureAdmissionStrict_, severe, std::memory_order_release);
```

`severe = (retirePressureLevel >= 3)`。retireDepth/hwm 比率が 95% 未満に下がれば自動的に `severe=false` となり `admissionStrict_` は false に戻る。

**解除経路2**: `executeRecoveryAction(RecoveryAction::Safe)` (`AudioEngine.Timer.cpp:691`)

```cpp
convo::publishAtomic(retirePressureAdmissionStrict_, false, std::memory_order_release);
```

Safe Mode 進入時に明示的に解除。

**問題**: 経路1は retireDepth が高止まりしている場合（実ログの tryReclaim 失敗時）**永遠に解除されない**。経路2は Safe Mode 進入という非常に重い判断が必要。中間的な解除条件（「retireDepth が改善していないが、一定時間経過したので suppression を解除してみる」）が存在しない。

### 検証結果: ✅ 妥当

- `applyRetirePressurePolicyNoRt()` は retireDepth/HWM 比率ベースの自動解除を持つ
- しかし **tryReclaim 失敗時に retireDepth が高止まりするシナリオでは解除されない**
- `RecoveryAction::Safe` が唯一の強制解除だが、発動条件が厳しい（60秒 Cooldown）
- 中間解除条件（ex: 30秒経過したら admissionsStrict を一時解除してリトライ）が不在

---

## P0-3: Last Known Good World

### レビューの主張 (P0-3)

PublishedWorld, PreviousWorld, LastKnownGoodWorld の3世代保持が必要。

### コード実装状態 (P0-3)

**インフラは既に存在する**:

| コード | 役割 |
| --- | --- |
| `AudioEngine.h:L1607` `lastHealthyWorldId_` | 最終健全 World ID |
| `AudioEngine.h:L1608` `lastHealthyPublicationTimestampUs_` | 最終健全発行タイムスタンプ |
| `AudioEngine.CtorDtor.cpp:203` `notifyHealthyPublication()` | 正常 publish 完了時に呼ばれ、`lastHealthyWorldId_` と `lastKnownGoodNoiseShaper_` を更新 |
| `AudioEngine.h:L1899` `lastKnownGoodNoiseShaper_` | 最終安定 Learner 状態スナップショット |
| `AudioEngine.Timer.cpp:681` `Restore` case | **コメントに "RollbackToLastHealthyWorld" と記載あり** |

**しかし**:

```cpp
case convo::RecoveryAction::Restore:
    // [work37 Phase 9.16] RollbackToLastHealthyWorld を含む復元操作
    tryReclaimResources();
    drainDeferredRetireQueues(false);
    // Rollback 基盤（ISRRetireRuntimeEx）の設定は呼び出し元が行う
    break;
```

コメントで「呼び出し元が行う」と書かれている通り、**実際の Rollback 処理は `executeRecoveryAction()` に実装されていない**。World の切戻し機構が未配線。

### 検証結果 (P0-3): ⚠️ 部分的に妥当

- `lastHealthyWorldId_` / `lastKnownGoodNoiseShaper_` は更新されている
- しかし `RecoveryAction::Restore` が実際にそれらを使って World を戻すコードが未実装
- **インフラは整っているが、回復パスが配線されていない**

---

## P0-4: NoiseShaper Learner 独立暴走

### レビューの主張 (P0-4)

LearningActive && RetireStall && RebuildSuppressed の複合条件で pauseLearning() すべき。

### コード実装状態 (P0-4)

**既存の Learner 保護 (`RuntimeHealthMonitor.cpp:tick() L63-69`)**:

```cpp
const uint64_t stallDur = getRetireStallDurationUs();
const bool learnerActive = (m_learnerRunningRef != nullptr)
    && convo::consumeAtomic(*m_learnerRunningRef, std::memory_order_acquire);
if (stallDur > 10'000'000 && learnerActive) {
    decision.actions |= toBit(RecoveryAction::Throttle);
}
```

**`RecoveryAction::Safe` での学習停止** (`AudioEngine.Timer.cpp:688`):

```cpp
case convo::RecoveryAction::Safe:
    diagLog("[RECOVERY] EnterSafeMode: stopping learner");
    stopNoiseShaperLearning();
```

### 検証結果 (P0-4): ⚠️ 部分的に妥当

- ✅ stallDur > 10s && learnerActive → Throttle (抑制)
- ✅ Safe Mode 進入時は `stopNoiseShaperLearning()` を実行
- ❌ レビューが要求する **3条件複合 (LearningActive && RetireStall && RebuildSuppressed) の陽な監視は未実装**
- ❌ Phase 9.4 (Learner Backpressure Monitor: bufferedSamples > 90% + RetireStall → pause) の FIFO 使用率監視はコード上確認できず
- `RecoveryAction::Throttle` は `admissionStrict_ = true` のみで、学習自体は停止しない。学習停止は Safe まで昇格が必要

---

## P0-5: Deferred Publish 滞留診断不足

### レビューの主張 (P0-5)

TTL 破棄だけでなく、滞留している publish の詳細（DeferredPublishAge, DeferredPublishCount, OldestDeferredPublishAge）を HealthMonitor に出せ。

### コード実装状態 (P0-5)

**既存の監視**:

| 監視項目 | コード | 状態 |
| --- | --- | --- |
| maxDeferredAgeMs (10s Warning, 30s Error) | `checkSnapshotStarvation()` | ✅ 実装済み |
| deferredPublish (件数) | RuntimeDrainAudit.h | ✅ 監査用に保持 |
| oldestPendingAgeMs | RuntimeDrainAudit.h | ✅ 監査用に保持 |
| suppression 継続時間 (30/60/120/180s 段階的) | `checkSuppressionDuration()` | ✅ 実装済み |

### 検証結果 (P0-5): ❌ 古い情報に基づく指摘

- `checkSnapshotStarvation()` (Phase 9.7) がまさに「DeferredPublishAge を監視して 10s/30s 閾値で Warning/Error を発行」するもの
- `checkSuppressionDuration()` (Phase 9.29) が suppression 継続時間に応じた段階的エスカレーションを行う
- レビューよりも **すでに多くの診断が実装されている**

**ただし**: `oldestPendingAgeMs` は RuntimeDrainAudit に含まれているが、HealthMonitor の独立した `check*()` 関数としては監視されていない。`checkSnapshotStarvation()` が `m_orchestrator->getMaxDeferredAgeMs()` を読んでいるが、これは `RuntimeDrainAudit::oldestPendingAgeMs` とは別経路かもしれない。

---

## P1-6: PolicyEngine が Recovery と Health を混在

### レビューの主張 (P1-6)

`decision.escalateToCritical` で直接 `m_healthState_` を書いている。`PolicyDecision → updateHealthState()` に統一すべき。

### コード実装状態 (P1-6)

**計画書の記述**:

```markdown
// ★ v3.0: PolicyEngine は HealthState を直接書き換えない
// ❌ 禁止: PolicyEngine が直接 m_healthState_ を publish
```

**コード実装 (`RuntimePolicyEngine.h:37`)**:

```cpp
// ★ v3.0: PolicyEngine は HealthState を直接書き換えない
```

**実際の更新パス** (唯一 `RuntimeHealthMonitor.cpp`) にて:

```cpp
void RuntimeHealthMonitor::updateHealthState(const PolicyDecision& decision) noexcept
{
    ISRHealthState newState = ISRHealthState::Healthy;
    // ... 全6系統の MonitorState から newState を算出 ...
    // PolicyDecision の causes を考慮
    if (decision.causes != 0 && newState == ISRHealthState::Healthy)
        newState = ISRHealthState::Degraded;
    convo::publishAtomic(m_healthState_, newState, std::memory_order_release);
}
```

`PolicyDecision` には `escalateToCritical` も `targetHealth` も存在しない。`decision.causes != 0 → Degraded` 昇格のみ。

### 検証結果 (P1-6): ❌ 古い設計案に対する指摘

- 計画書（v3.0 以降）もコードも、PolicyEngine が HealthState を直接書き換える設計を**明確に禁止している**
- `updateHealthState(const PolicyDecision&)` が唯一の決定権限
- レビューの指摘は v3.0 以前の古い設計案に基づく

---

## P1-7: Reader Stuck 根因追跡

### レビューの主張 (P1-7)

どの reader か (readerSlot, readerDepth, readerEpoch, ownerThread) が見えない。

### コード実装状態 (P1-7)

**Reader Stuck 検出 (`RuntimeHealthMonitor.cpp:294-345`)**:

`diagnoseRetireStall()` は `m_retireRouter->detectStuckReaders(10)` を呼び出し、以下の情報を取得:

```cpp
auto stuckInfo = m_retireRouter->detectStuckReaders(10);
// stuckInfo には以下が含まれる
// - readerIndex
// - readerEpoch
// - residencyTimeUs
// - readerDepth (>0)
```

HealthEvent は以下のように生成される:

```cpp
HealthEvent ev{getCurrentTimeUs(), ...};
ev.readerIndex = stuckInfo.readerIndex;
ev.readerEpoch = stuckInfo.readerEpoch;
ev.readerDepth = 1;
ev.residencyTimeUs = stuckInfo.residencyTimeUs;
m_callback(ev);
```

**10秒間隔の定期 Evidence 出力**も同様の詳細情報を含む。
**Reader Slot >90% の詳細診断**でも同じく full detail が含まれる。

### 検証結果 (P1-7): ❌ 不正確

- `readerSlot` (readerIndex), `readerDepth`, `readerEpoch`, `residencyTimeUs` は**全て報告されている**
- `onHealthEvent(EVENT_READER_STUCK)` のログ出力でも `readerIndex`, `residencyTimeUs`, `pendingRetire` を出力
- ただし `ownerThread` は報告されていない（ISR 設計上、Reader Slot はスレッド所有権を持たない）
- `emitOnTransition()` 経由の標準パスでは詳細情報が欠落するが、これは Warning レベル（Reader Slot 75-90%）のみ

---

## P1-8: RouterPendingRetire 独立監視

### レビューの主張 (P1-8)

pendingRetireCount だけでなく `routerOldestRetireAgeMs` を独立監視すべき。

### コード実装状態 (P1-8)

**既存の監視**:

| 監視項目 | コード | 状態 |
| --- | --- | --- |
| pendingRetireCount | `checkRetireStall()` | ✅ |
| maxRetireAge | `checkRetireReclaimLatency()` via `m_maxRetireAgeRef` | ✅ |
| oldestPendingAge | `RuntimeDrainAudit.h:L34` | ✅ 監査用 |
| oldestPendingGeneration_ | `AudioEngine.h:L1600` | ✅ |

**ただし**: `oldestPendingAgeMs` は `RuntimeDrainAudit` の監査構造体に含まれているが、`RuntimeHealthMonitor` の独立した `check*()` 関数としては監視されていない。`checkRetireReclaimLatency()` は `m_maxRetireAgeRef` 経由で Retire Age を監視しているが、これが `oldestPendingAgeMs` と同じ値かどうかは確認が必要。

### 検証結果 (P1-8): ⚠️ 部分的に妥当

- pendingRetireCount の監視は十分
- `maxRetireAge`（最古 retire の経過時間）は `checkRetireReclaimLatency()` で監視済み
- ただし `oldestPendingAgeMs` が HealthMonitor の独立 metric として陽に監視されているわけではない
- 監査 (RuntimeDrainAudit) と HealthMonitor の間で metric の重複/分断がある

---

## P1-9: WorldConsistency が shutdown 側に偏っている

### レビューの主張 (P1-9)

稼働中にも published-retired, activeWorlds, routerPendingRetire を定期監査すべき。

### コード実装状態 (P1-9)

`RuntimeHealthMonitor::tick()` には `checkWorldConsistency()` が含まれている:

```cpp
void RuntimeHealthMonitor::checkWorldConsistency() noexcept
{
    if (!m_worldConsistencyCheck_)
        return;
    const auto consistencyState = m_worldConsistencyCheck_();
    if (consistencyState >= 2) {
        emitOnTransition(m_prevConfigDivergenceState_, MonitorState::Error,
            HealthEvent::Severity::Error, 5001,
            static_cast<uint64_t>(consistencyState));
    }
}
```

この `m_worldConsistencyCheck_` は `AudioEngine` が `RuntimeDrainAudit::verifyWorldConsistency()` をラップしたコールバックである。つまり**毎 tick、運転中に World 整合性を監査している**。

### 検証結果 (P1-9): ❌ 不正確

- **既に運転中に `checkWorldConsistency()` が実行されている**
- `EVENT_WORLD_CONSISTENCY` は HealthMonitor 経由で PolicyEngine に通知される
- レビューの「shutdown 側に偏っている」という認識は、v7.1 の実装では解消済み

---

## P2-10: HealthMonitor が「検出器」でしかない

### レビューの主張 (P2-10)

音質劣化（音が悪い → 構成更新停止 → 学習継続）を検出できない。将来的に AudioQualityDegradationMonitor が必要。

### コード実装状態 (P2-10)

- ✅ `checkConfigurationDivergence()` — 設定世代乖離の検出
- ✅ `checkSuppressionDuration()` — 抑制継続時間の段階的エスカレーション
- ✅ `checkRuntimeProgressFreeze()` — 3軸統合進行凍結検出（Publish/Retire/Rebuild）
- ✅ `checkPendingStructuralDeployment()` — Rebuild generation gap
- ✅ `checkSnapshotStarvation()` — Deferred publish 滞留
- ❌ **音響品質の直接監視は未実装**（DC Offset, Peak Clipping, RMS Jump, Noise Floor は `PolicySource::AudioOutputAnomaly` として定義されているが、対応する `check*()` 関数は未実装）

### 検証結果 (P2-10): ✅ 妥当（ただし進行中）

- 検出器としての機能は Phase 9 で大幅に強化されている
- `RuntimeProgressFreeze`（3軸統合）はレビューが指摘する「音が悪い→構成更新停止→学習継続」に最も近い検出器
- しかし **音響品質そのものの監視**（DC Offset, Clipping 等）は未実装
- レビューが言う AudioQualityDegradationMonitor は P2 として妥当な将来課題

---

## まとめ: 優先5項目のコード検証結果

| # | 項目 | レビュー優先度 | コード対応度 | 検証結果 |
| --- | --- | --- | --- | --- |
| 1 | Retire Stall Recovery Ladder | **P0** | 50% | ラダーは6段階存在するが、逐次エスカレーション + RecoveryVerification が未実装 |
| 2 | Recovery Verification（回復確認） | **P0** | 0% | `RecoveryOutcome` は定義されているが全く未使用。最も深刻なギャップ |
| 3 | Last Known Good World | **P0** | 40% | インフラ（`lastHealthyWorldId_`/`lastKnownGoodNoiseShaper_`）は整備済み。しかし回復パスが未配線 |
| 4 | NoiseShaper Learning Pause 条件 | **P0** | 60% | stallDur > 10s + learnerActive の基本保護は実装済み。3条件複合監視と FIFO 使用率監視は未実装 |
| 5 | Rebuild Suppression 自動解除条件 | **P0** | 50% | retireDepth ベースの自動解除はあるが、tryReclaim 失敗時の Alternative 解除条件が不在 |

### 補足: 前回の検証結果との差分

前回の検証で指摘した2つの最重要ギャップは、今回のレビューでも同様に指摘されている:

1. **RecoveryVerification 不足** ← 両レビューで共通の最重要指摘
2. **`injectBackpressureSignal` 未統合** ← 今回のレビューでは直接言及されていないが、Rebuild Suppression 片方向問題の一部

### 総合評価

| レビューの主張 | 検証結果 |
| --- | --- |
| P0-1: Retire Stall Recovery Ladder 不在 | ⚠️ ラダー段階は存在。逐次エスカレーションと閉ループ検証が欠落 |
| P0-2: Rebuild Suppression 片方向 | ✅ 妥当。tryReclaim 失敗時の自動解除条件が弱い |
| P0-3: Last Known Good World 不在 | ⚠️ インフラはあるが未配線。Restore Action に Rollback 実装が必要 |
| P0-4: Learner 独立暴走 | ⚠️ 基本保護（10s+learnerActive）は実装済み。複合条件とFIFO監視は未実装 |
| P0-5: Deferred Publish 滞留診断不足 | ❌ 不正確。checkSnapshotStarvation() と checkSuppressionDuration() が既に監視済み |
| P1-6: PolicyEngine 混在 | ❌ 不正確。計画書もコードも既に分離済み |
| P1-7: Reader Stuck 根因追跡不足 | ❌ 不正確。readerIndex/readerEpoch/readerDepth/residencyUs は全報告済み |
| P1-8: RouterPendingRetire 独立監視不足 | ⚠️ 部分的に妥当。maxRetireAge は監視済みだが、oldestPendingAge は独立 metric ではない |
| P1-9: WorldConsistency 偏り | ❌ 不正確。checkWorldConsistency() は毎 tick 稼働中に実行済み |
| P2-10: HealthMonitor が検出器でしかない | ✅ 妥当。音響品質直接監視は未実装 |

### 実際に動作確認すべきギャップ（トップ3）

1. **RecoveryVerification の実装** — `RecoveryOutcome` を `RuntimeHealthMonitor::tick()` に組み込み、Action 実行→N秒後評価→昇格 の閉ループ制御を追加
2. **`RecoveryAction::Restore` への Rollback 配線** — `lastHealthyWorldId_` と `lastKnownGoodNoiseShaper_` を使って実際の World 切戻しを実装
3. **Suppression の中間解除条件** — retireDepth が高止まりしていても、一定時間経過後に admissionStrict を一時解除してリトライする機構
