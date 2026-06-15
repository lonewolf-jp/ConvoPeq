# recovery_system_plan.md 外部レビュー コード突合検証報告

> **日付**: 2026-06-15
> **検証方法**: doc/work37/recovery_system_plan.md (v7.1最終版) と実装コード12ファイルの突合
> **検証ファイル**: RuntimePolicyEngine.h/.cpp, RuntimeHealthMonitor.h/.cpp, DSPLifetimeManager.h, SnapshotCoordinator.h/.cpp, RefCountedDeferred.h, DeferredRetireFallbackQueue.h, AudioEngine.Timer.cpp, AudioEngine.Retire.cpp, AudioEngine.h

---

## 0. 前提：重要な発見

**このレビューは、recovery_system_plan.md の「v5.1〜v6.2 系」の古いバージョンを評価対象としている**可能性が極めて高い。現時点のコードベースには **v7.1 最終版のほぼ全内容が実装済み**であり、レビューが指摘する問題の大半は既に解消されている。

検証で確認した実装ファイル:

| ファイル | 対応Phase | 状態 |
| --- | --- | --- |
| `RuntimePolicyEngine.h/.cpp` | Phase 0 | ✅ 実装済み |
| `RuntimeHealthMonitor.h/.cpp` | Phase 2/9 全監視 | ✅ 実装済み |
| `DSPLifetimeManager.h` | Phase 1.1 | ✅ tryReclaim+再試行実装済み |
| `SnapshotCoordinator.h/.cpp` | Phase 1.2 | ✅ `enqueueWithRetry()` 実装済み |
| `RefCountedDeferred.h` | Phase 1.3 | ✅ `canBlock()` 分岐実装済み |
| `DeferredRetireFallbackQueue.h` | Phase 8.1 | ✅ SoftLimit/HardLimit実装済み |
| `AudioEngine.Timer.cpp` | Phase 4.1/9 | ✅ `executeRecoveryAction()` 6段階実装済み |
| `AudioEngine.Retire.cpp` | Phase 4.4 | ⚠️ 背圧統合未完了 |

---

## 1. レビュー指摘の各項目に対する検証結果

### 1.1 妥当と評価された部分

| レビューの評価 | 検証結果 |
| --- | --- |
| **Phase 1 enqueueRetire 戻り値修復: 非常に妥当** | ✅ **完全に正しい**。DSPLifetimeManager, SnapshotCoordinator, RefCountedDeferred は全て戻り値チェック+tryReclaim再試行を実装済み |
| **ReaderStuck 改善 (epochGap + residency): 妥当** | ✅ **正しい**。ただしコード上は plan の「10秒滞留」条件のみ部分的に実装済み。完全な複合判定（epochGap AND residency）は EpochDomain 側の実装修正が必要 |
| **PolicyEngine 縮退版: 大規模案より良い** | ✅ **正しい**。実際の `RuntimePolicyEngine` は MonitorState → RecoveryAction のシンプルな選択器。Severity/Persistence/BlastRadius 体系は導入されていない |

### 1.2 問題として指摘されたが、すでに解消済みの項目

#### 問題①「HealthState 直接書き換え」— **誤った指摘**

レビュー:
> `if (decision.escalateToCritical) { publishAtomic(m_healthState_, ISRHealthState::Critical); }` は危険

**コード検証結果: このコードは現行実装に存在しない。**

- `RuntimePolicyEngine.h` L37: `// ★ v3.0: PolicyEngine は HealthState を直接書き換えない`
- `RuntimeHealthMonitor::updateHealthState(const PolicyDecision&)` が唯一の権限
- PolicyDecision は `targetHealth` フィールドを持たない（`/* ★ v3.6: targetHealth は持たない */`）
- 代わりに `decision.causes != 0` → `Degraded` 昇格の形で PolicyEngine の意見を参照しているのみ

**評価: レビューは古い設計案を批判しており、現状コードでは改善済み。**

#### 問題②「HealthEvent 再利用」— **誤った指摘**

レビュー:
> RecoveryAction を HealthEvent に変換して onHealthEvent() に流すのは設計的に後退

**コード検証結果: 現行実装では完全に分離されている。**

- `RuntimeHealthMonitor::tick()` L70-86: RecoveryAction は `m_actionCallback(action)` で直接発火
- `AudioEngine::onHealthEvent()` と `AudioEngine::executeRecoveryAction()` は**独立した関数**
- plan 文書: `// ★ v3.1: HealthEvent の再利用をやめ、executeRecoveryAction() を新設`

**評価: レビューは古い v5.1 案の問題点を指摘しているが、v7.1 実装では解決済み。**

#### 問題③「フォールバックキュー Critical 強制昇格」— **実装より安全**

レビュー:
> `forceHealthState(Critical)` は原因情報が失われる

**コード検証結果: 現行実装では `requestEmergencyDrain()` を使用。**

- `RuntimeHealthMonitor::requestEmergencyDrain()` は単なる `atomic<bool>` フラグ設定
- HealthState は `updateHealthState()` が引き続き統括
- Emergency Drain 要求は HealthCauseBits で原因追跡可能

**評価: 問題の核心は既に回避されている。**

---

## 2. 部分的に妥当な指摘

### 2.1 `injectBackpressureSignal` が未統合

**レビューが間接的に指摘する「背圧の二重権限問題」は現状も残っている。**

`AudioEngine.Retire.cpp` の `drainDeferredRetireQueues()` は依然として **直接** `retirePressureAdmissionStrict_` を書き換えている（L157, L291）。一方 `injectBackpressureSignal()` は `RuntimeHealthMonitor.h` で定義済みだが、**どこからも呼び出されていない**（grep 一致: 定義箇所のみ1件）。

つまり plan の「4.4.1 経路1の PolicyEngine 統合」は**未完了**。`applyRetirePressurePolicyNoRt()` の `admissionStrict_` 直接設定は残存している。

**実害**: PolicyEngine が Throttle を解除しようとしても、Retire.cpp が再設定するため競合状態が発生し得る。

### 2.2 RecoveryVerification（回復確認）が未実装 ← **最も重要な指摘**

**コード検証: 完全に正しい指摘。**

- `RecoveryOutcome` enum（Success/NoEffect/Failed/Unsafe）は `RuntimePolicyEngine.h` で定義されている
- しかし `RuntimeHealthMonitor.cpp` で `RecoveryOutcome` を**一切使用していない**
- `executeRecoveryAction()` の後、tick() は Action の効果を確認せずに次のサイクルへ進む
- 実ログの「tryReclaim → 失敗 → その後ずっと suppression」を防ぐ閉ループ制御（Action実行→N秒後評価→効果なければ昇格）は存在しない

**これが最も深刻なギャップであり、レビューの「55〜60点」評価はここに起因する。**

### 2.3 Retire Stall Recovery Ladder — **実は既に存在する**

レビュー: 「RetireRecoveryLevel の段階的ラダーが必要」

**コード検証: `RecoveryAction` が6段階（Observe→Throttle→Recover→Restore→Safe→Critical）で既にラダーを形成。**

ただし、レビューが求める「Level1 tryReclaim → Level2 flushDeferredPublish → Level3 releaseDeferredStructuralRebuild → Level4 publishLastKnownGoodWorld → Level5 admissionStop」という**段階的リカバリーの自動エスカレーション**は、`RuntimePolicyEngine` の `evaluateAggregate()` では MonitorState ベースでしか Action を選択しておらず、「Action A を試した → 効果なし → Action B へ昇格」という**逐次エスカレーション**は実装されていない。

`evaluateAggregate()` は毎 tick 独立して最高優先度 Action を選ぶため、Cooldown 制御以外に昇格シーケンスの保証がない。

### 2.4 NoiseShaper/LearningDivergence 対策 — **部分的に実装済み**

レビュー: 「LearningDivergence 監視と pause learning が必要」

**コード検証:**

- ✅ `RuntimeHealthMonitor::tick()` L66-69: `stallDur > 10s && learnerActive → Throttle`
- ✅ `RecoveryAction::Safe` に `stopNoiseShaperLearning()` が実装済み
- ✅ Phase 9.4 (Learner Backpressure Monitor) の設計は存在
- ❌ しかし `learnerFifoUsage`（AudioSegmentBuffer の使用率）の実監視はコード上確認できず

**評価: 基本対策は実装済み。レビューで言及された「bufferedSamples > 90% AND RetireStall Error → pause learning」という複合条件は未実装。**

---

## 3. 総合評価マトリクス

| 評価軸 | レビューの評価 | 実コード検証後の補正評価 | 補正理由 |
| --- | --- | --- | --- |
| 実運用適合性 | A- | **A** | HealthState直接書き換え・HealthEvent再利用の懸念は既に解消 |
| 実装容易性 | A | **A** | 大半が既に実装済み。残りは背圧統合のみ |
| ISR安全性 | A- | **A** | 問題①〜③は既に安全な設計に修正済み |
| 障害回復能力 | B | **B+** | RecoveryVerification 不足分は B- だが、SuppressionDuration/ProgressFreeze 監視の充実度を加味 |
| 音質破綻回復能力 | C+ | **B-** | Learner Health Policy は実装済みだが、bufferedSamples 複合監視は未実装 |
| Retire Stall耐性 | B | **B+** | エスカレーションラダーは既に存在。閉ループ検証がないため上限 B+ |
| 過剰設計リスク | 低 | **低** | 特に問題なし |

### 評価スコア補正

```text
レビュー評価: リーク対策85-90点, HealthMonitor整理80点, 実障害対策55-60点
コード突合後: リーク対策95点,  HealthMonitor整理90点, 実障害対策70-75点
```

---

## 4. 結論

### 4.1 レビューが的確だった指摘（トップ3）

1. **RecoveryVerification 不足** — 🥇 最も重要。`RecoveryOutcome` enum はあるが使われていない。閉ループ制御の欠如は実障害再発の最大リスク。
2. **`injectBackpressureSignal` 未統合** — 背圧経路の二重権限（PolicyEngine + Retire.cpp の直接書き込み）が残存。
3. **逐次エスカレーションの不在** — `evaluateAggregate()` が MonitorState ベースで独立判断するため、「Throttle → 効果なし → Recover」のような段階的昇格が保証されない。

### 4.2 レビューが誤っていた／古い設計を批判した指摘

1. **HealthState 直接書き換え** — 現状コードでは PolicyEngine は HealthState を直接書き換えない。v3.0 で修正済み。
2. **HealthEvent 再利用** — `executeRecoveryAction()` が独立して存在。v3.1 で修正済み。
3. **Retire Stall Recovery Ladder 不在** — `RecoveryAction` 6段階として既に実装。ただし逐次エスカレーション機構は未実装。
4. **NoiseShaper 対策なし** — Phase 9.1 の Learner Health Policy は実装済み。

### 4.3 総評

レビューの核心（RecoveryVerification 不足）は正しい指摘である。ただし、レビューの批判の大半は v5.1〜v6.x 系の古い設計に対するものであり、v7.1 最終版として実装されたコードはそれらの問題をほぼ全て解決している。実スコアはレビュー評価より約15ポイント高い。

**現時点で最も対応すべき2項目**:

1. `injectBackpressureSignal` を `AudioEngine.Retire.cpp` から呼び出し、背圧経路を PolicyEngine 単一権限に統一
2. `RecoveryOutcome` を `RuntimeHealthMonitor::tick()` に組み込み、Action実行後の回復確認と逐次エスカレーションを実装
