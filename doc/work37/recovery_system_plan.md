# Practical Stable ISR Bridge Runtime — 回復システム改修計画書 v7.1（最終版）

> **日付**: 2026-06-14 | **ベース文書**: doc/work37/notfinished10.md
> **事前調査**: Serena MCP + AiDex MCP (275files/3979methods/371types) + CodeGraph MCP (51K entities) + graphify MCP (15K nodes) + grep
> **コード検証**: 全12未確定事項→確定。第三者レビュー11回+7回設計レビュー+実ログ(ConvoPeq.log)突合

**変更履歴**（v5.1→v7.1: 約30回のイテレーション）:

| Version | 重点 | 主要追加 |
| --- | --- | --- |
| v5.1 | 初期設計 | SafeMode, ReaderSlot, RecoveryAction階層化(5段階) |
| v6.0 | 全12項目確定 | DeferredRetireFallbackQueue/DiscardReason/EQProcessor17箇所確定 |
| v6.1 | 検証反映 | check*()9→7修正。各所?→—修正 |
| v6.2 | 🎯実ログ対策 | **RuntimeProgressFreeze/SuppressionTTL/RetireBlockerSnapshot/LearnerRollback** |
| v6.3 | 閉ループ制御 | RecoveryOutcome(Success/NoEffect/Failed)。ForcePublish安全ガード。SafeMode Learner停止 |
| v6.4 | 根本原因除去 | EpochAdvanceBlocked(Level1-4)。LearnerOutputDivergence(L2距離)。SafeModeRecovery |
| v6.5 | 追加耐性 | retireRateガード。LearnerCheckpoint多世代化。SafeMode二段階。Suppressionヒステリシス |
| v6.6 | 🎯統合 | **PolicySource 72→14分類。RecoveryAction 44→6レベル。ConfigurationDeadlock** |
| v6.7 | 最終安全性 | activeReaders==0ガード。RetireRootCauseUnknown。Checkpoint互換性検証 |
| v6.8 | 整理 | RetireProgressFrozen。PolicyEngine責務限定。SoftSafeMode自動復帰。Learner強制停止 |
| v6.9 | 仕上げ | audioRecovered。AudioQualityFingerprint。ProgressFreeze+Anomaly+Failed→SafeMode |
| v7.0 | 🎯縮退運転 | **PublicationPriority(Structural優先)。RetireRootCauseEvidence。LearnerRollback自動** |
| v7.1 | 最終確定 | ClearSuppression独立Action。PendingRetireObjectInfo。AudioQualityFingerprint正式化 |

---

> **クイックリファレンス**: 本設計書は ISR Bridge Runtime の回復システム改修を **Phase 0-9** に分類する。
>
> - **Phase 0** (P0) PolicyEngine基盤 — 最優先。既存 MonitorState→RecoveryAction 選択器
> - **Phase 1** (P0-P1) enqueueRetire契約修復 — 4経路の戻り値チェック追加
> - **Phase 2** (P0) 監視層拡張 — ReaderStuck/Consistency/Leak 検出
> - **Phase 3** (P1) ShutdownResult — healthState JSON 出力追加
> - **Phase 4** (P1-P2) Analytics拡張 — WorldLifecycleAudit/Overflow/injectEvent
> - **Phase 5** (P3) BlockingReason — 現状維持
> - **Phase 6** (P1) DiscardReason::Expired — TTL破棄
> - **Phase 7** (P1) DeferredPublish TTL監視
> - **Phase 8** (P0-P2) Shutdown完了保証 — EmergencyDrain/VerifyDrained
> - **Phase 9** (P0-P2) Runtime Configuration Progress Health — 統合監視(14 PolicySource分類, 6 RecoveryActionレベル, ConfigurationDeadlock新設)
>
> 各Phaseの優先度: **P0=必須(即実装)** / P1=重要(Phase1完了後) / P2=推奨(余裕時) / P3=保留
>
> **実装スコープ (v7.1 確定)**:
>
> - **Phase 0/1/2**: PolicyEngine基盤 + enqueueRetire契約修復 + ReaderStuck判定 — **最優先・全実装**
> - **Phase 9.40-9.65**: Runtime系/Learner系/Retire系 + PublicationPriority + AudioQualityFingerprint + ClearSuppression — **P0実装**
> - **Phase 9.54-9.56**: SafeModeRecovery/RuntimeRecoveryScore — **保留（P2）**
>
> **Definitive Reference** (v7.1最終版): PolicySource最終版→4.7節表, RecoveryAction最終版→4.7節表, 設計定数→4.6節, タスク→2章, ファイル影響→3章

## 0. 現状分析サマリ

### 0.1 既に動作しているもの

| 領域 | 状態 | 備考 |
| --- | --- | --- |
| 正常系 Publish/Retire/Shutdown | ✅ 95% | 基本的なフローは完成 |
| 監視・検出 (HealthMonitor) | ✅ 90% | 多角的な tick 監視あり (9種類、全 `check*()` 関数確認済み) |
| Admission 連携 (HealthState→Reject) | ✅ 完了 | Critical 時は publish 拒否 |
| onHealthEvent コールバック | ✅ 完了 | 4種類の Recovery Action 実装済み（RetireStall / CrossfadeTimeout / PublicationStall / ReaderSlotExhaustion） |
| enqueueRetire 戻り値 (AudioEngine主経路) | ✅ 完了 | `enqueueDeferredDeleteNonRtWithResult` は drain retry 実装済み |

### 0.2 不足しているもの

| 領域 | 状態 | 備考 |
| --- | --- | --- |
| PolicyEngine 層 | ❌ 不在 | MonitorState → RecoveryAction の選択器 |
| enqueueRetire 契約 (DSPLifetimeManager等) | ❌ 死文化 | 4経路が戻り値無視（SnapshotCoordinatorは却下） |
| Reader Stuck 判定 | ⚠️ 部分的 | residency 収集あり、epoch差のみで判定 |
| ShutdownResult 型 | ❌ 不在 | healthState 異常を結果記録する仕組みなし |
| WorldConsistency Authority | ❌ diagLogのみ | Policy 評価で診断→Authority に昇格すべき |
| doubleRetireCount Health連携 | ❌ ファイル監査のみ | HealthMonitor 未接続 |
| BlockingReason 多値化 | ⚠️ 単一 | Primary + Secondary が必要 |
| FallbackQueue 拡張 | ❌ push戻り値/容量/retryCount | 現状は無制限vector |

### 0.3 コード調査で確定した重要ファクト

#### ファクトA: updateHealthState() は既に単一権限

現状コードで `RuntimeHealthMonitor::updateHealthState()` は全 `m_prev*State`（6系統）を集約して ISRHealthState を決定している。v4.0 の設計と完全一致。

```cpp
void RuntimeHealthMonitor::updateHealthState() noexcept {
    ISRHealthState newState = ISRHealthState::Healthy;
    if (m_prevRetireState == MonitorState::Error) newState = Critical;
    // ... 5系統の状態を順次評価 ...
    convo::publishAtomic(m_healthState_, newState, ...);
}
```

#### ファクトB: onHealthEvent() は既に4種類の Recovery Action を持つ

`AudioEngine::onHealthEvent()`（`AudioEngine.Timer.cpp:541`）:

- ReaderSlotExhaustion → `retirePressureAdmissionStrict_ = true`
- PublicationStall → `clearDeferredForShutdown()`
- RetireStall → `admissionStrict_` + `tryReclaimResources()`
- CrossfadeTimeout → DSP退役 + crossfade解放 + idle publish

PolicyEngine 導入後は「Action選択」を PolicyEngine に移管し、`executeRecoveryAction()` で既存実装を呼び出す形になる。

#### ファクトC: enqueueRetire の戻り値 — 主経路はチェック済み

`AudioEngine::enqueueDeferredDeleteNonRtWithResult()` は戻り値をチェックし、失敗時に `drainDeferredRetireQueues(false)` を実行している。

未チェックの4経路:

| 経路 | ファイル | 保護レベル | Phase |
| --- | --- | --- | --- |
| `DSPLifetimeManager::retire()` | ISRRetireRouter経由 | Router内tryReclaimあり | 1.1 |
| `EQProcessor::enqueueDeferredDeleteWithFallback()` | ISRRouter経由+Coordinator | 二重問題 | 1.4 |
| `RefCountedDeferred::release()` | IEpochProvider直結 | 保護なし | 1.3 |
| `SnapshotCoordinator(4経路)` | IEpochProvider直結 | 保護なし | **却下** |

#### ファクトD: DeferredDeletionQueue の容量は 4096

`kQueueSize = 4096`（`DeferredDeletionQueue.h:221`）— MPMC ロックフリーキュー。これがあふれた場合に ISRRetireRouter が tryReclaim+再試行を行う。

#### ファクトE: DeferredRetireFallbackQueue は拡張が必要

`src/core/DeferredRetireFallbackQueue.h:15-52`:

- `push()` → `size_t`（現状: `bool` が必要）
- `popAll()` → `std::vector<DeferredRetireFallbackEntry>`（mutex ロック＋swap）
- `size()` / `empty()` — mutex 保護あり
- 内部構造: `std::vector<DeferredRetireFallbackEntry>` + `mutable std::mutex`
- **不在メンバ**: `estimatedBytes` / `overflowRate` / `notifyOverflow` / `retryCount`
- **拡張必要**: `push()` 戻り値を `bool` に変更, `retryCount` 追加, SoftLimit(1000)/HardLimit(2000) 追加

#### ファクトF: retirePressureAdmissionStrict_ は**実質4条件**から書き込まれる

調査で判明した最も重要な発見: `retirePressureAdmissionStrict_` は同一 Timer tick 内で**以下の4条件**から書き込まれている（詳細は `codebase_verification_v4_1.md` Section 1, Phase 4.4 参照）。

経路1は内部的に2段階の複合処理（背圧評価→overflow 検出で上書き）であるため、実質4条件が独立して書き込む形になる:

```text
条件1a: drainDeferredRetireQueues() → evaluateRetirePressureLevelNoRt()(retireDepth/hwm比率)
       → applyRetirePressurePolicyNoRt() → retirePressureAdmissionStrict_
       (AudioEngine.Retire.cpp:118, 284, 発火: retireDepth/hwm比率 >= kRetirePressureSeverePercent)

条件1b: 同上 → overflow検出ブロック（droppedDelta>0 / chronicByDuration>5s / chronicByFrequency>3.0/sec）
       → effectiveLevel = max(level1a, overflowLevel) → retirePressureAdmissionStrict_ (上書き)
       (AudioEngine.Retire.cpp:121-154, 発火: 3条件のOR)

条件2: HealthMonitor::tick() → onHealthEvent(EVENT_READER_SLOT_USAGE)
       → retirePressureAdmissionStrict_ = true (AudioEngine.Timer.cpp:557, 発火: ReaderSlot使用率 > 90%)

条件3: HealthMonitor::tick() → onHealthEvent(EVENT_RETIRE_STALL / EVENT_RETIRE_AGE_CRITICAL)
       → retirePressureAdmissionStrict_ = true (AudioEngine.Timer.cpp:588, 発火: pending > hwm*1.5)
```

**注意**: 条件1b は条件1a の結果を上書きする（overflow が発生すると retireDepth 評価を無効化）。条件2/3 は条件1b の結果をさらに上書きする。最終書き込みのみが有効（last-write-wins）。

**PolicyEngine 導入効果**: PolicyEngine が全 MonitorState + overflow 状態から RecoveryAction を選択し、`executeRecoveryAction()` が唯一の admissionStrict_ 書き込み経路となる。条件2/3は HealthMonitor 経由のため自然に統合される。条件1a/1b（drainDeferredRetireQueues 内の独立背圧+overflow）は評価結果を PolicyEngine に渡し、PolicyEngine が ThrottleRebuild Action として統一発行する。

現状:

- `push()` 戻り値は `size_t`（v4.0 では `bool` が必要）
- メモリサイズ追跡なし（v4.0 では `estimatedBytes()` が必要）
- `retryCount` なし（v4.0 では `kFallbackMaxRetries=3` が必要）
- HardLimit なし（v4.0 では容量ベース 50MB/100MB）

---

## 1. 改修フェーズ

### Phase 0: Policy Engine 基盤 — MonitorState 駆動型（**最重要**）

#### 設計方針

調査で判明した通り、HealthMonitor は既に7種類の `check*()` 関数と `MonitorState`（Normal/Warning/Error）による状態遷移検出を持っている。**新たな Severity/Persistence/BlastRadius 体系を導入する必要はない。**

```text
check*() → MonitorState→遷移 → PolicyEngine → RecoveryAction
                          ↓
                   updateHealthState()（既存）
```

PolicyEngine の役割:

1. 既存の MonitorState 遷移から RecoveryAction を選択
2. Cooldown 制御で同一 Action の連続実行を防止
3. 新規イベント（WorldLeak/WorldConsistency）にも同じ枠組みを適用
4. **PolicyDecision を updateHealthState() に渡し、HealthState は updateHealthState() が最終決定する**

##### ★ v3.0: PolicyEngine は HealthState を直接書き換えない

```cpp
// ❌ 禁止: PolicyEngine が直接 m_healthState_ を publish
// convo::publishAtomic(m_healthState_, ISRHealthState::Critical, ...);

// ✅ 正: updateHealthState(PolicyDecision) が単一の決定権限
// ★ 新規オーバーロードとして追加 (既存の updateHealthState() は維持)
void RuntimeHealthMonitor::updateHealthState(const PolicyDecision& decision) noexcept {
    // 1. 既存の MonitorState ベース評価（既存の引数なし版を呼び出し）
    //    優先順位（コード実装順）:
    //    1. Retire Error → Critical
    //    2. Publication Error → Critical（最優先）
    //    3. OverflowRate Error → Critical（既存Critical維持）
    //    4. ReaderSlot Error → Critical（同上）
    //    5. RetireAge Error → Critical（同上）
    //    6. 各Warning + Healthy → Degraded
    ISRHealthState newState = computeFromMonitorStates();
    // 2. PolicyDecision の targetHealth を考慮
    if (decision.targetHealth > newState)
        newState = decision.targetHealth;
    convo::publishAtomic(m_healthState_, newState, std::memory_order_release);
}
```

#### 1.1 RuntimePolicyEngine クラス新設

**新規ファイル**: `src/audioengine/RuntimePolicyEngine.h` / `.cpp`

```cpp
namespace convo::isr {

// ★ v3.0: PolicySource — m_contexts のキー。eventCode 非依存。
enum class PolicySource : uint8_t {
    RetireStall,
    PublicationStall,
    ReaderStuck,
    ReaderSlotUsage,
    OverflowRate,
    RetireAge,
    CrossfadeTimeout,
    CrossfadeEventDrop,
    WorldLeak,
    WorldConsistency,
    EmergencyDrain,
    _Count  // 要素数
};

// Recovery Action — HealthMonitor が選択する緩和動作
// 実装は AudioEngine::onHealthEvent() が担当（既存）
enum class RecoveryAction : uint8_t {
    None,
    ForceRetireDrain,        // tryReclaim 強制
    ThrottleRebuild,         // retirePressureAdmissionStrict_ 設定
    ClearDeferredPublish,    // 滞留 publish クリア
    ForceCrossfadeReset,     // crossfade 強制完了
    RejectNewPublication,    // 全新規 publish 拒否
    EmergencyDrain           // ★ Shutdown時のみ。強制回収モード
};

// ★ v4.0: PolicyEngine — MonitorState → 最優先RecoveryAction の選択器
//    複数Actionを同時発行せず、最高優先度のActionのみ返す。
//    Action優先順位（高い順）:
//      1. EmergencyDrain
//      2. RejectNewPublication
//      3. ForceRetireDrain
//      4. ForceCrossfadeReset
//      5. ClearDeferredPublish
//      6. ThrottleRebuild
//    HealthState の決定は RuntimeHealthMonitor::updateHealthState() が唯一の権限。

// ★ v3.0: PolicySource の配列要素数はソース数に固定。eventCode 非依存。
static constexpr size_t kPolicySourceCount =
    static_cast<size_t>(PolicySource::_Count);

// ★ v3.0: Policy 評価結果 — RecoveryAction と HealthState をレイヤ分離
struct PolicyDecision {
    RecoveryActionBits actions{0}; // ビットマスク（複数Action同時発行可能）
    uint32_t cooldownUs;           // 同一 Action 再実行間隔
    HealthCauseBits causes{0};     // 複合原因（OR可能）
    // ★ v3.6: targetHealth は持たない。HealthStateは HealthMonitor 専権。
};

// ★ v3.0: HealthCause — Critical/Degraded の原因を特定
// ★ v3.2: uint64_t causeBits で複合原因に対応
enum class HealthCause : uint64_t {
    None                  = 0,
    RetireStall           = 1ull << 0,
    PublicationStall      = 1ull << 1,
    ReaderStuck           = 1ull << 2,
    ReaderSlotExhaustion  = 1ull << 3,
    OverflowRate          = 1ull << 4,
    RetireAged            = 1ull << 5,
    CrossfadeTimeout      = 1ull << 6,
    CrossfadeEventDrop    = 1ull << 7,
    WorldLeak             = 1ull << 8,
    WorldConsistencyBad   = 1ull << 9,
    EmergencyDrain        = 1ull << 10,
    FallbackQueueOverflow = 1ull << 11
};

// HealthCauseBits: OR可能な複合原因
using HealthCauseBits = uint64_t;

// ★ v2.0: ConsistencyFailureType — World整合性異常の種類
enum class ConsistencyFailureType : uint8_t {
    None,
    AuditMismatch,     // 監査レコード欠損（published-retired≠active）— Warning
    WorldLeak,         // 実Worldリーク（retired>published）— Critical
    DoubleRetire,      // 二重retire検出 — Critical
    Unknown
};

// PolicyEngine: 既存 MonitorState を Policy 評価する薄いラッパー
// ★ v2.0: 時間軸+回復状態+HealthCause を追加
class RuntimePolicyEngine {
public:
    RuntimePolicyEngine() noexcept;

    // 全 MonitorState から統合評価（HealthMonitor::tick から呼ばれる）
    PolicyDecision evaluateAggregate(
        MonitorState retireStall,
        MonitorState publicationStall,
        MonitorState readerSlotUsage,
        MonitorState overflowRate,
        MonitorState retireAge,
        MonitorState crossfadeDrop) noexcept;

    // 特定 PolicySource に対する評価（WorldLeak/WorldConsistency 等）
    PolicyDecision evaluateEvent(PolicySource source,
                                  ConsistencyFailureType consistencyType = ConsistencyFailureType::None) noexcept;





    // Cooldown 制御
    bool canExecute(RecoveryAction action) const noexcept;
    void markExecuted(RecoveryAction action) noexcept;



private:
    struct CooldownEntry {
        uint64_t lastExecutedUs{0};
        uint64_t cooldownUs;
    };
    CooldownEntry m_cooldowns[7]; // RecoveryAction 数
    // ★ v3.0: PolicySource 固定長配列（eventCode 非依存）
    std::array<PolicyContext, kPolicySourceCount> m_contexts;
    uint64_t getNowUs() const noexcept;
};

} // namespace
```

#### 1.2 異常分類テーブル（MonitorState → RecoveryAction）

既存の HealthMonitor イベントコードと MonitorState 遷移に基づく:

| イベント | MonitorState | RecoveryAction | Cooldown | HealthCause |
| --- | --- | --- | --- | --- |
| EVENT_RETIRE_STALL | Error | ThrottleRebuild | 1秒 | RetireStall |
| EVENT_PUBLICATION_STALL | Error | ClearDeferredPublish | 5秒 | PublicationStall |
| EVENT_READER_STUCK | Warning | ForceRetireDrain | 10秒 | ReaderStuck |
| EVENT_READER_SLOT_USAGE | Error | ThrottleRebuild | 1秒 | ReaderSlotExhaustion |
| EVENT_CROSSFADE_TIMEOUT | Error | ForceCrossfadeReset | 30秒 | CrossfadeTimeout |
| EVENT_CROSSFADE_EVENT_DROP | Error | None | 10秒 | CrossfadeEventDrop |
| EVENT_RETIRE_AGE_CRITICAL | Error | ForceRetireDrain + ThrottleRebuild | 10秒 | RetireAged |
| EVENT_RETIRE_AGE_WARNING | Warning | None（監視のみ） | - | RetireAged |
| **新: EVENT_WORLD_LEAK** | - | RejectNewPublication（主）+ ForceRetireDrain（補助） | 1秒 | WorldLeak |
| **新: EVENT_WORLD_CONSISTENCY** | - | None | - | WorldConsistencyBad |
| **新: EVENT_EMERGENCY_DRAIN** | - | ForceRetireDrain | - | EmergencyDrain |

**設計判断**:

1. **MonitorState → Action**: PolicyEngine は MonitorState 遷移から RecoveryActionBits を選択するだけ。
2. **HealthState は HealthMonitor 専権**: `updateHealthState()` が唯一の決定権限。PolicyEngine は targetHealth を持たない。
3. **HealthCauseBits**: PolicyDecision に原因ビットマスクを含めることで Critical 到達時のデバッグ性を確保。

**提言2 確認 — Cooldown スレッド安全性**: CodeGraph MCP で `RuntimeHealthMonitor::tick()` の呼び出し元を調査した結果、**`AudioEngine::timerCallback()`（`AudioEngine.Timer.cpp:532`）からのみ呼ばれている**。JUCE Timer コールバックは Message Thread（単一 Non-RT スレッド）で実行されるため、`m_cooldowns` 配列の読み書きにデータ競合は発生しない。? **安全確認済み**。

---

### Phase 1: enqueueRetire 戻り値契約の修復（P0）

**重要**: AudioEngine 主経路（`enqueueDeferredDeleteNonRtWithResult`）は既に `drainDeferredRetireQueues(false)` の retry を実装済み。未対応は以下の4経路。

#### 1.1 DSPLifetimeManager::retire() 修正（ISRRetireRouter経由）

**ファイル**: `src/audioengine/DSPLifetimeManager.h`

現状: `router_->enqueueRetire(dsp, deleter, epoch)` — 3引数の `bool` 版。戻り値未チェック。ただし経路は ISRRetireRouter 経由で、Router 内の tryReclaim+再試行は適用されている。

修正: 戻り値をチェックし、失敗時に Router の tryReclaim を使って再試行。NonRT 限定のため `tryReclaim()` は安全。

```cpp
if (!router_->enqueueRetire(static_cast<void*>(dsp),
                             &AudioEngine::destroyDSPCoreNode,
                             epoch)) {
    // Router 内の tryReclaim でも失敗 → 明示的再試行
    router_->tryReclaim();
    if (!router_->enqueueRetire(static_cast<void*>(dsp),
                                 &AudioEngine::destroyDSPCoreNode,
                                 epoch)) {
        // 最終手段: fallback queue 経由で後続 drain に委ねる
        engine_.enqueueFallbackRetire(dsp, &AudioEngine::destroyDSPCoreNode);
        return;
    }
}
convo::fetchAddAtomic(engine_.rtAuxMutable_.runtimeRetireCount, 1u,
                      std::memory_order_acq_rel);
```

#### 1.2 SnapshotCoordinator enqueueRetire 経路修正（**部分実施**）

**発見**: SnapshotCoordinator は `IEpochProvider*` 経由で **EpochDomain に直接** enqueueRetire している。ISRRetireRouter の tryReclaim+再試行保護が適用されない。

**重要: スレッド安全性の検証結果**: 全5経路のうち、`resetFadeStateAndRetireTarget()`(L67) は `updateFade()` 経由で RT(Audio Thread) から呼ばれ得る。`tryReclaim()` はブロッキングのため RT から呼べない。以下の表に従って部分実施する。

| 経路 | 行 | 呼び出し元 | Non-RT? | enqueueWithRetry可能? |
| --- | --- | --- | --- | --- |
| `startFade()` | SnapshotCoordinator.cpp:36 | createSnapshotFromCurrentState(Timer) | ✅ Yes | ✅ Yes |
| `resetFadeStateAndRetireTarget()` | SnapshotCoordinator.cpp:67 | updateFade(RT) | ❌ **No** | ❌ **No** |
| `completeFade()` | SnapshotCoordinator.cpp:87 | tryCompleteFade(Timer) | ✅ Yes | ✅ Yes |
| `switchImmediate()` | SnapshotCoordinator.h:92 | NonRT直接呼び出し | ✅ Yes | ✅ Yes |
| `retireCurrentAndTarget()` | SnapshotCoordinator.h:161-162 | finalizeShutdown(ReleaseResources) | ✅ Yes | ✅ Yes |

```cpp
// SnapshotCoordinator.h に追加
// ★ 使用条件: Non-RT スレッドからのみ呼び出し可能（RTからは呼ばないこと）
//   IEpochProvider::tryReclaim() は実装依存でブロッキング/ロック取得を含み得るため、
//   抽象インタフェース経由ではブロッキング安全性を静的に保証できない。
//   呼び出し前に canBlock() 表明または jassert(canBlock()) による動的チェックを推奨。
static bool enqueueWithRetry(convo::IEpochProvider& provider,
                              void* ptr, void (*deleter)(void*),
                              uint64_t epoch) noexcept {
    if (provider.enqueueRetire(ptr, deleter, epoch))
        return true;
    // IEpochProvider 抽象インタフェース経由: tryReclaim のブロッキング安全性は呼び出し元が保証
    jassert(!convo::numeric_policy::isAudioThread());  // RT からの呼び出し禁止
    provider.tryReclaim();
    if (provider.enqueueRetire(ptr, deleter, epoch))
        return true;
    // 再試行失敗 → NonRT 側の定期 drain に期待（ベストエフォート）
    return false;
}
```

4箇所（5経路中4経路）の呼び出しを `enqueueWithRetry()` に置き換え:

- `startFade()` L36: ? 置換
- `completeFade()` L87: ? 置換
- `switchImmediate()` header: ? 置換
- `retireCurrentAndTarget()` L161-162: ? 置換（2箇所ともに）
- `resetFadeStateAndRetireTarget()` L67: ? **現状維持**（RTから呼ばれ得るため）

#### 1.3 RefCountedDeferred::release() 修正（canBlock 概念）

**発見**: `RefCountedDeferred::release()` は AudioThread（RT）からも呼ばれ得る。`tryReclaim()` はブロッキングのため RT からは呼べない。

**★ v3.0: isAudioThread() から canBlock() へ発展**: `isAudioThread()` は Audio Thread 判定に特化しており、将来の realtime worker や ASIO callback に対応できない。`canBlock()` という汎用ブロッキング許容判定を導入する。

```cpp
// DspNumericPolicy.h に追加
// ★ v3.4: ExecutionClass ? スレッドの実行時分類
//    Practical Stable では canBlock() の判定を ExecutionClass ベースに昇格
enum class ExecutionClass : uint8_t {
    Realtime,       // Audio Thread, ASIO callback ? 絶対にブロック不可
    SoftRealtime,   // MIDI callback, realtime worker ? 軽量処理のみ
    Normal,         // Timer callback, Message Thread ? 通常のブロック許容
    Background      // Worker Thread, 非同期処理 ? 長時間ブロック許容
};

// canBlock(): ExecutionClass >= Normal で判定
//   呼び出し元または getCurrentExecutionClass() で動的に評価
inline bool canBlock() noexcept {
    return !convo::numeric_policy::isAudioThread();
}

// RefCountedDeferred.h
void release(convo::IEpochProvider& provider) noexcept {
    if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
        std::atomic_thread_fence(std::memory_order_acquire);
        if (!provider.enqueueRetire(
                static_cast<T*>(this),
                [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); },
                provider.currentEpoch())) {
            // canBlock() が false なら tryReclaim 禁止
            if (convo::numeric_policy::canBlock()) {
                provider.tryReclaim();
                (void)provider.enqueueRetire(
                    static_cast<T*>(this),
                    [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); },
                    provider.currentEpoch());
            }
            // 再試行失敗は HealthMonitor overflowCount 監視に委ねる
        }
    }
}
```

#### 1.4 EQProcessor retire 戻り値伝播（二重問題）

**発見**: `EQProcessor::enqueueDeferredDeleteWithFallback()` の経路:

1. `ISRRuntimePublicationCoordinator::enqueueRetire()` → ISRRetireRouter（tryReclaim保護あり）
2. ISRRetireRouter が QueuePressure を返す
3. Coordinator が QueueFull に変換（**再試行なし**）
4. EQProcessor が false 返す
5. 呼び出し元 `retireEQStateDeferred()` が **void → 結果消失**

**ファイル**: `src/eqprocessor/EQProcessor.Core.cpp`

```cpp
bool EQProcessor::enqueueDeferredDeleteWithFallback(
    void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept
{
    if (ptr == nullptr || deleter == nullptr) return true;
    if (m_retireCoordinator == nullptr) return false;

    const uint64_t retireEpoch = (epoch != 0) ? epoch : m_epochDomain.currentEpoch();
    convo::isr::ISRRetireRouter stackRouter(m_epochDomain);
    auto result = m_retireCoordinator->enqueueRetire(
        convo::isr::RetireAuthority::Granted,
        stackRouter, ptr, deleter, retireEpoch);
    if (result == convo::isr::RetireEnqueueResult::Success)
        return true;

    // 失敗: tryReclaim + 再試行（NonRT安全）
    m_epochDomain.tryReclaim();
    result = m_retireCoordinator->enqueueRetire(
        convo::isr::RetireAuthority::Granted,
        stackRouter, ptr, deleter, retireEpoch);
    return result == convo::isr::RetireEnqueueResult::Success;
}

// retireEQStateDeferred / retireBandNodeDeferred を bool 返しに変更
// 【注意】retireEQStateDeferred は 13箇所（Core.cpp:3, Parameters.cpp:10）、
// retireBandNodeDeferred は 4箇所（Core.cpp:3, Coefficients.cpp:1）から呼ばれる。
// 計17箇所（旧推定59箇所→実測17箇所）。
// 全呼び出し元で戻り値を (void) キャストして既存動作を維持する。
bool EQProcessor::retireEQStateDeferred(EQState* state) noexcept {
    if (state == nullptr) return true;
    const uint64_t epoch = m_epochDomain.currentEpoch();
    return enqueueDeferredDeleteWithFallback(state, deleteEQStatePtr, epoch);
}
```

#### 1.5 フォールバックキュー統合（最終安全網）

`DeferredRetireFallbackQueue`（`src/core/DeferredRetireFallbackQueue.h`）は既に存在。上限なし（`std::vector` ベース）。`drainDeferredRetireQueues()` の先頭で全 drain する。再フォールバックも可能。

```cpp
void AudioEngine::drainDeferredRetireQueues(bool allowDuringShutdown) noexcept {
    // ★ fallback queue ドレイン
    {
        auto entries = m_retireFallbackQueue_.popAll();
        for (auto& entry : entries) {
            // ★ v4.0: retryCount 上限を設定。3回超で FallbackOverflow イベント発火
            if (entry.retryCount >= kFallbackMaxRetries) {
                worldLifecycleAudit_.onFallbackOverflow();
                continue;  // ドロップ。リーク許容
            }
            if (!m_retireRouter->enqueueRetire(entry.ptr, entry.deleter,
                                                entry.epoch, DeletionEntryType::Generic)) {
                m_retireRouter->tryReclaim();
                if (!m_retireRouter->enqueueRetire(entry.ptr, entry.deleter,
                                                    entry.epoch, DeletionEntryType::Generic)) {
                    entry.retryCount++;
                    m_retireFallbackQueue_.push(entry); // 再フォールバック
                }
            }
        }
    }
    // ... 既存の処理 ...
}
```

---

### Phase 2: Reader Stuck 判定改善（P0）

#### 2.1 detectStuckReaders() に residency 時間条件追加

**ファイル**: `src/core/EpochDomain.h`

現状:

```cpp
if (epochGap > stuckThreshold) {
    info.isStuck = true;
    break;
}
```

修正後:

```cpp
// ★ 複合判定: epoch差 AND residency時間
if (epochGap > stuckThreshold && residencyUs > 1'000'000) {
    info.isStuck = true;
    info.residencyTimeUs = residencyUs;
    break;
}
// ★ 滞留検出（epochが進まないケース）
//   2段階: Warning=10s, Critical候補=30s
//   pendingRetire==0 の場合は正常アイドル（DAW停止/ブレークポイント）の可能性。
//   10s 未満はデバッガ停止/DAW Pause/一時的UIフリーズと区別がつかないため発火しない。
//   これにより誤検出を防止。
if (depth > 0 && residencyUs > 10'000'000 && info.pendingRetireCount > 0) {
    info.isStuck = true;
    info.residencyTimeUs = residencyUs;
    info.isChronic = (residencyUs > 30'000'000);  // 30秒超は重症
    break;
}
```

---

### Phase 3: ShutdownResult 導入（P0）

#### 3.1 ShutdownResult 型追加

**ファイル**: `src/audioengine/ISRShutdown.h`

```cpp
struct ShutdownResult {
    bool completed;
    ShutdownPhase finalPhase;
    ISRHealthState healthState;
    ShutdownBlockingReason blockingReason;
    uint64_t durationMs;
    uint32_t transitionViolations;
    uint32_t lateCallbackCount;
    uint32_t postStopEnqueueCount;
};
```

#### 3.2 ShutdownRuntime に collectResult() 追加

```cpp
[[nodiscard]] ShutdownResult collectResult(ISRHealthState healthState,
                                            uint64_t startTimestampMs) const noexcept;
```

#### 3.3 releaseResources() に collectResult 呼び出し追加

```cpp
// HealthState でブロックせず、結果に記録
const auto shutdownResult = shutdownRuntime_.collectResult(
    m_healthMonitor.getHealthState(), shutdownStartMs);
emitShutdownResult(shutdownResult);
```

---

### Phase 4: Policy Engine 統合 — 既存 onHealthEvent を PolicyEngine 経由に統一（P1）

#### 4.1 HealthMonitor → PolicyEngine → AudioEngine 連携

PolicyEngine は HealthMonitor の `tick()` 内で全 MonitorState を評価し、RecoveryAction を選択する。

**★ v3.1: HealthEvent の再利用をやめ、`executeRecoveryAction()` を新設**

```cpp
// RuntimeHealthMonitor.h
using RecoveryActionCallback = std::function<void(RecoveryAction)>;
void setActionCallback(RecoveryActionCallback cb) noexcept { m_actionCallback = std::move(cb); }

// AudioEngine 側
m_healthMonitor.setActionCallback(
    [this](convo::isr::RecoveryAction action) { executeRecoveryAction(action); });

void AudioEngine::executeRecoveryAction(convo::isr::RecoveryAction action) noexcept {
    switch (action) {
        case RecoveryAction::ForceRetireDrain:
            tryReclaimResources(); break;
        case RecoveryAction::ThrottleRebuild:
            convo::publishAtomic(retirePressureAdmissionStrict_, true, ...); break;
        case RecoveryAction::ClearDeferredPublish:
            if (runtimeOrchestrator_)
                runtimeOrchestrator_->clearDeferredForShutdown(); break;
        // ... 各 Action の実装
        default: break;
    }
}
```

これにより Event→Policy→Event の逆流構造が解消される。

```cpp
void RuntimeHealthMonitor::tick() noexcept {
    // 既存の検出（変更なし）
    checkRetireStall();
    checkPublicationStall();
    diagnoseRetireStall();
    checkCrossfadeTimeout();
    checkCrossfadeEventDrop();
    checkReaderSlotUsage();
    checkOverflowRate();
    checkRetireReclaimLatency();

    // ★ Policy Engine 評価: 全 MonitorState から統合判定
    auto decision = m_policyEngine.evaluateAggregate({
        m_prevRetireState,
        m_prevPublicationState,
        m_prevReaderSlotState,
        m_prevOverflowRateState,
        m_prevRetireAgeState,
        m_prevCrossfadeDropState
    });
    // ★ v3.1: RecoveryAction は HealthEvent として再包装しない
    //    executeRecoveryAction() を新設し、RecoveryAction を直接発火
    if (decision.action != RecoveryAction::None && m_actionCallback) {
        m_actionCallback(decision.action);
    }

    // ★ v3.0: updateHealthState() が PolicyDecision を受け取り単一権限で HealthState を決定
    updateHealthState(decision);
}
```

#### 4.2 onHealthEvent の責務整理

現状の `AudioEngine::onHealthEvent()` が個別ハードコードしている RecoveryAction のうち、Cooldown 制御や escalation 判断は PolicyEngine に移管。onHealthEvent は Action の**実装**のみに専念。

| 現状 | 移行後 |
| --- | --- |
| eventCode を switch で個別処理 | PolicyEngine が Action 選択。Callback は実装のみ |
| Cooldown なし | PolicyEngine が canExecute/markExecuted で制御 |
| ReaderStuck → 診断ダンプのみ | PolicyEngine が escalateToCritical 判定 |

#### 4.3 WorldLifecycleAudit → HealthMonitor 連携

`WorldLifecycleAudit::onWorldRetired()` 内で二重 retire 検出時（`prev == 0`）に `HealthEvent::EVENT_WORLD_LEAK` を発火。PolicyEngine が即座に `RejectNewPublication`（主）+ `ForceRetireDrain`（補助）を返す。WorldLeak はライフサイクル破綻であり、ForceRetireDrain だけでは修復不能。まず全面新規 publish 停止で被害拡大を防止したうえで、補助的に drain を実行する。

#### 4.4 背圧機構の統一（旧3経路→単一権限化）

**背景**: ファクトF で発見した通り、`retirePressureAdmissionStrict_` は3つの独立経路から書き込まれている。PolicyEngine 導入によりこれを単一権限に統一する。

#### 4.4.1 経路1（drainDeferredRetireQueues 内の独立背圧）の PolicyEngine 統合

**ファイル**: `src/audioengine/AudioEngine.Retire.cpp`

現状:

```cpp
// AudioEngine.Retire.cpp:119 — HealthMonitor 非依存の独立背圧
auto level = evaluateRetirePressureLevelNoRt();
if (level >= 2) {
    convo::publishAtomic(retirePressureAdmissionStrict_, true,
                         std::memory_order_release);
}
```

修正後:

```cpp
// ★ PolicyEngine に整合性評価を委譲。直接 admissionStrict_ を設定しない
const auto fbSize = m_retireFallbackQueue_.size();
const auto overflowRate = m_retireFallbackQueue_.overflowRate();
// HealthMonitor の PolicyEngine に評価を依頼
m_healthMonitor.injectBackpressureSignal(fbSize, overflowRate);
// admissionStrict_ は executeRecoveryAction(ThrottleRebuild) が設定
```

これにより `evaluateRetirePressureLevelNoRt()` の背圧判定ロジックは PolicyEngine に移管される。

#### 4.4.2 経路2/3（onHealthEvent 内の直接設定）の削除

現状の `onHealthEvent()` 内で `retirePressureAdmissionStrict_` を直接設定している箇所（EVENT_READER_SLOT_USAGE, EVENT_RETIRE_STALL）は削除し、代わりに PolicyEngine が `executeRecoveryAction(RecoveryAction::ThrottleRebuild)` として統一発行する。

#### 4.4.3 統合後の背圧フロー

```text
HealthMonitor::tick()
  ├→ checkRetireStall() → MonitorState
  ├→ checkReaderSlotUsage() → MonitorState
  ├→ checkOverflowRate() → MonitorState
  ├→ evaluateAggregate() → PolicyDecision
  │   └→ MonitorState から RecoveryAction を選択
  │       └→ ThrottleRebuild が選択された場合のみ
  │           executeRecoveryAction() → admissionStrict_ = true
  └→ updateHealthState()

drainDeferredRetireQueues()
  └→ evaluateRetirePressureLevelNoRt()
      └→ 結果を injectBackpressureSignal() で PolicyEngine に通知
          └→ 次回 tick() で PolicyEngine が評価
```

#### 4.4.4 削除/整理される既存コード

| ファイル | 行 | 現状 | 移行後 |
| --- | --- | --- | --- |
| `AudioEngine.Retire.cpp` | 119-154 | 独立背圧 + admissionStrict_ 直接設定 | `injectBackpressureSignal()` に置換 |
| `AudioEngine.Timer.cpp` | 557 | EVENT_READER_SLOT_USAGE → admissionStrict_ 直接設定 | 削除（PolicyEngine が ThrottleRebuild として発行） |
| `AudioEngine.Timer.cpp` | 588 | EVENT_RETIRE_STALL → admissionStrict_ 直接設定 | 同上 |
| `AudioEngine.Threading.cpp` | 20 | `shouldRejectRebuildAdmissionForPressure()` | 維持（read側は変更なし） |

#### 4.5 RouterPendingRetire

`checkRetireStall()` の pendingRetireCount 監視で実質カバー済み。`checkRouterPendingRetire()` を独立メソッドとして追加する必要はない。

---

### Phase 5: BlockingReason 多値化（P2）

`ShutdownBlockingReason primary` + `ShutdownBlockingReason secondary` の2値。`markTimedOut` / `markFailed` に secondary パラメータ追加。

---

### Phase 6: Deferred Publish TTL（P1）

`DeferredPublishSlot` に `maxAgeUs = 30'000'000` 追加。`notifyTransitionComplete()` で TTL 超過時は `DiscardReason::Expired`。

---

### Phase 7: WorldConsistency Authority 昇格（P0）

`verifyWorldConsistency()` が Broken を返した場合、`HealthMonitor::injectEvent()` 経由で PolicyEngine に通知。

---

### Phase 8: レビュー指摘対応 — 追加強化項目

#### 8.1 フォールバックキュー OOM 防止ガード

**発見**: `DeferredRetireFallbackQueue` は上限なし（`std::vector` ベース）。実運用上は空だが、Non-RT ドレインが完全に停止した場合に無限メモリ消費リスク。

**対応**: `drainDeferredRetireQueues()` 内でフォールバックキューが異常なサイズに達した場合、PolicyEngine に即座に Critical 昇格を要求。

```cpp
void AudioEngine::drainDeferredRetireQueues(bool allowDuringShutdown) noexcept {
    // ★ フォールバックキュー安全弁
    const auto fbSize = m_retireFallbackQueue_.size();
    // ★ v3.2: 容量ベース監視（推定メモリ使用量）
    //   DeferredRetireFallbackEntry に estimatedSize を追加
    //   または HealthMonitor 経由で PolicyEngine にイベント注入
    if (fbSize > kFallbackQueueCriticalThreshold ||
        m_retireFallbackQueue_.estimatedBytes() > kFallbackQueueMemoryLimit) {
        // forceHealthState() 禁止。PolicyEngine 経由で updateHealthState() に委譲
        m_retireFallbackQueue_.notifyOverflow();
        // PolicyEngine が FallbackOverflow イベントを評価
    }
    // ... フォールバック drain 処理 ...
}
```

**v2.0 定数**:

```cpp
constexpr size_t kFallbackSoftLimit  = 10000;  // 超過時: PolicyEngineにCritical昇格要求
constexpr size_t kFallbackHardLimit  = 50000;  // 超過時: Admission全面停止（強制ドロップ）
```

**SoftLimit/HardLimit 二段構成**:

```cpp
// DeferredRetireFallbackQueue.h
static constexpr std::size_t kFallbackSoftLimit = 1000;  // 超過: PolicyEngine に Critical 昇格要求
static constexpr std::size_t kFallbackHardLimit = 2000;  // 超過: 強制ドロップ（リーク許容）

[[nodiscard]] std::size_t push(DeferredRetireFallbackEntry entry)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= kFallbackHardLimit) // 2000件上限
        return queue_.size();                // 超過時はドロップ（リーク許容）
    if (queue_.size() >= kFallbackSoftLimit) // SoftLimit 超過
        notifyOverflow();                    // PolicyEngine に Critical 昇格要求
    queue_.push_back(entry);
    return queue_.size();
}
```

#### 8.2 Emergency Drain の PolicyEngine 実行時制御化

**発見**: `EmergencyDrain` は既に `ShutdownPhase` 列挙型に存在する（`ISRShutdown.h:31`）。しかし `releaseResources()` 内の実装が `#ifdef CONVOPEQ_EMERGENCY_DRAIN` マクロで保護されているため、PolicyEngine が実行時に発動できない。

**対応**:

1. `RuntimeHealthMonitor` に `requestEmergencyDrain()` / `isEmergencyDrainRequested()` を追加
2. `releaseResources()` の Emergency Drain ブロックをマクロ依存から実行時判定に変更

```cpp
// RuntimeHealthMonitor.h 追加
void requestEmergencyDrain() noexcept {
    convo::publishAtomic(m_emergencyDrainRequested_, true, std::memory_order_release);
}
bool isEmergencyDrainRequested() const noexcept {
    return convo::consumeAtomic(m_emergencyDrainRequested_, std::memory_order_acquire);
}
void clearEmergencyDrain() noexcept {
    convo::publishAtomic(m_emergencyDrainRequested_, false, std::memory_order_release);
}
// メンバ追加
std::atomic<bool> m_emergencyDrainRequested_{false};

// ReleaseResources.cpp: PolicyEngine が EmergencyDrain を要求した場合
if (m_healthMonitor.isEmergencyDrainRequested()) {
    shutdownRuntime_.transitionTo(ShutdownPhase::EmergencyDrain);
    executeEmergencyDrain();  // 既存の #ifdef ブロック内容
    m_healthMonitor.clearEmergencyDrain();
}
```

#### 8.3 Deferred Publish TTL 超過時の通知

`DeferredPublishSlot` が `DiscardReason::Expired` で破棄される際、コントロール層（UI/API）にパブリッシュ破棄を通知する仕組みを追加（Phase 6 の補強）。

#### 8.4 検証結果: レビュー指摘の確認

| 指摘 | 確認結果 | 対応 |
| --- | --- | --- |
| `isAudioThread()` の動的判定 | ✅ `DspNumericPolicy.h:117` に存在 | Phase 1.3 に反映済み |
| WaitableEvent の有無 | ❌ コードベースに存在せず | overflowCount 監視で代替 |
| EmergencyDrain の runtime化 | ?? コンパイル時マクロのみ | Phase 8.2 で対応 |
| フォールバックキュー OOM | ?? 理論上のリスク | Phase 8.1 で SoftLimit/HardLimit 対応 |
| Deferred TTL 通知 | ✅ 処理済み | Phase 8.3 で UI 連携を追加 |

---

### Phase 9:  Runtime Configuration Progress Health（新規）

> これまでの Phase 0-8 は `Runtime Resource Health`（retire/reclaim 健全性）を監視する。
> Phase 9 は `Runtime Configuration Progress Health`（DSP 構成変更の進捗健全性）を監視する。
> 今回実障害「Retire Stall → 全 Rebuild Suppress → Learner だけ継続 → 音が変なまま固定」を直接検出・緩和する。

#### 9.1 【P0】Learner Health Policy

**背景**: `AudioEngine.RebuildDispatch.cpp:546` のコメント `// NOTE: 意図的に learner を停止しない。rebuild 中も learner は稼働を継続` により、Retire Stall 発生時も NoiseShaper Learner は動作し続ける。DSP 世界が凍結した状態での学習継続は、学習途中の係数で長期間運転され音質異常が固定化する。

**コード確認**:

- `RebuildDispatch.cpp:546`: 意図的に learner を停止しない設計
- `Learning.cpp:210-211`: `IRChanged suppressed (Running)` — DSP 置換中でも Running なら learner 継続
- `shouldRejectRebuildAdmissionForPressure()`: 3箇所 (RebuildDispatch.cpp:241,428,479) で rebuild を抑制

**新規 RecoveryAction**:

```cpp
enum class RecoveryAction : uint8_t {
    ...
    PauseLearner,    // ★ 新規: Learner 一時停止
    ResumeLearner    // ★ 新規: Learner 再開
};
```

**Policy**:

```text
EVENT_RETIRE_STALL が 10秒継続
    ↓
PauseLearner → stopNoiseShaperLearning()

Retire 状態が Normal に回復
    ↓
ResumeLearner → startNoiseShaperLearning(..., resume=true)
```

**PolicyEngine 自動連携（v4.8 強化）**: `PauseLearner` は PolicyEngine が以下の複合条件で自動発動する:

```text
LearnerPublishBlocked (Error) 検出
    AND
SuppressionDuration > 30秒
    ↓
PauseLearner → stopNoiseShaperLearning()

その後
RecoveryAction::ForcePublicationRecovery で回復確認
    AND
正常復帰
    ↓
ResumeLearner → startNoiseShaperLearning(..., resume=true)
```

**Integration**: `AudioEngine::executeRecoveryAction()` に以下を追加:

```cpp
case RecoveryAction::PauseLearner:
    stopNoiseShaperLearning();
    break;
case RecoveryAction::ResumeLearner:
    startNoiseShaperLearning(m_pendingLearningMode, true);
    break;
```

**ファイル**: `src/audioengine/AudioEngine.Timer.cpp`（executeRecoveryAction）, `src/audioengine/RuntimePolicyEngine.h`（RecoveryAction enum）

#### 9.2 【P0】RuntimeConfigurationDivergence

**背景**: `publishedRevision` と `targetRevision`（実際は `lastCommittedRuntimeGeneration_` と `rebuildRequestGeneration`）の乖離を監視する。今回の障害では `intent 16 → 17 → 23` が抑制されたまま適用されなかった。

**コード確認**:

- `lastCommittedRuntimeGeneration_` (`AudioEngine.h:1582`): 最終適用済み世代
- `rebuildRequestGeneration` (`AudioEngine.h:1725`): 要求された最新世代
- `shouldRejectRebuildAdmissionForPressure()` で rebuild が抑制されると `RebuildTelemetryEvent::Suppressed` が発行される

**新規イベント**:

```cpp
static constexpr uint32_t EVENT_CONFIGURATION_DIVERGENCE = 5001;
```

**新規 PolicySource**:

```cpp
enum class PolicySource : uint8_t {
    ...
    ConfigurationDivergence,  // ★ 新規
    LearnerBackpressure       // ★ 新規
};
```

**コード確認（定量化）**:

- `rebuildRequestGeneration` (`AudioEngine.h:1725`): **要求された最新世代**（pendingRevision）
- `lastCommittedRebuildGeneration` (`AudioEngine.h:1726`): **最終適用済み世代**（activeRevision）
- `lastCommittedRuntimeGeneration_` (`h:1582`): 最終 commit 世代
- `revisionGap = pendingRevision - activeRevision`: 乖離量を定量化

**監視内容**:

```cpp
// RuntimeHealthMonitor に新規 check 関数追加
void checkConfigurationDivergence() noexcept {
    if (m_lastCommittedGenRef == nullptr || m_requestedGenRef == nullptr)
        return;
    const uint64_t committed = convo::consumeAtomic(*m_lastCommittedGenRef, ...);
    const uint64_t requested = convo::consumeAtomic(*m_requestedGenRef, ...);
    if (requested <= committed) {
        m_prevConfigDivergenceState = MonitorState::Normal;
        return;
    }
    const uint64_t gap = requested - committed;
    /* ★ v4.7 定量化: revisionGap >= 5 で即座に発火
       rebuildRequestGeneration / lastCommittedRebuildGeneration の差分で計算 */
    if (gap >= 3 && m_divergenceStartUs != 0 && (nowUs - m_divergenceStartUs) > 30'000'000) {
        // gap >= 3 かつ 30秒継続 → Critical
        emitOnTransition(m_prevConfigDivergenceState, MonitorState::Error, ...);
    } else if (gap >= 2 && m_divergenceStartUs != 0 && (nowUs - m_divergenceStartUs) > 30'000'000) {
        emitOnTransition(m_prevConfigDivergenceState, MonitorState::Warning, ...);
    }
}
```

**Policy**:

```text
ConfigurationDivergence Warning
    ↓
ForceRetireDrain + PauseLearner

ConfigurationDivergence Error
    ↓
ForceRetireDrain + PauseLearner + RejectNewPublication
```

#### 9.3 【P1】Deferred Rebuild Expiration

**背景**: `DEFERRED → RELEASED → SUPPRESSED` の連鎖で Rebuild が永遠に適用されない状態を検出。今回の IR 変更がまさにこれ。

**新規状態**:

```cpp
enum class DiscardReason : uint8_t {
    ...
    PolicySuppressed   // ★ 新規: Policy による抑制で破棄
};
```

**新規監視カウンタ**:

```cpp
std::atomic<uint64_t> suppressedStructuralRebuildCount_{0};  // AudioEngine.h に追加
```

**Policy**:

```text
同一 Structural Rebuild が 3回 Suppress された
    ↓
Critical 昇格 + PauseLearner
```

**Integration**: `RebuildDispatch.cpp` の Suppress 箇所 (L241, L428, L479) でカウンタをインクリメント:

```cpp
if (shouldRejectRebuildAdmissionForPressure()) {
    convo::fetchAddAtomic(suppressedStructuralRebuildCount_, 1u, ...);
    // 累積カウンタを HealthMonitor が参照
    ...
}
```

#### 9.4 【P1】Learner Backpressure Monitor

**背景**: `bufferedSamples=3840000` で張り付く現象。Learner Consumer < Audio Producer 状態。

**コード確認**: `NoiseShaperLearner.cpp:913` に `bufferedSamples` のログ出力あり。`segmentBuffer.getNumAvailableSamples()` で現在のバッファリング量を取得可能。

**新規イベント**:

```cpp
static constexpr uint32_t EVENT_LEARNER_BACKPRESSURE = 5002;
```

**監視**:

```text
learnerFifoUsage = learnerSegmentBuffer.usagePercent()

FIFO > 90% が 30秒継続
    ↓
PauseLearner

FIFO > 95% が 60秒継続
    ↓
Critical + PauseLearner
```

**注意**: `AudioSegmentBuffer.h:82` の `getNumAvailableSamples()` は `convo::consumeAtomic(totalSamples, std::memory_order_acquire)` を使用しており、**完全にロックフリーかつスレッドセーフであることが確認されている**。Monitor (Non-RT timer) からの定期読み取りに追加の同期機構は不要。

#### 9.5 【P0】Retire Stall Auto Recovery

**背景**: 現在の計画は `Throttle → Reject → Critical` の一方通行。回復試行を含めるべき。

**新規 Action**:

```cpp
enum class RecoveryAction : uint8_t {
    ...
    ForcePublicationRecovery   // ★ 新規: 能動的回復試行
};
```

**処理**:

```cpp
case RecoveryAction::ForcePublicationRecovery:
    // 能動的回復試行（止めるだけでなく回復を試みる）
    tryReclaimResources();
    drainDeferredRetireQueues(false);
    retryDeferredPublication();       // ★ 新規: 保留中の publish 再試行
    retryDeferredStructuralRebuild(); // ★ 新規: 保留中の StructuralRebuild 再試行
    break;
```

**Policy**（異常検出→回復試行→失敗時停止の3段階）:

```text
異常検出 (RetireStall / ConfigurationDivergence)
    ↓
ForcePublicationRecovery（能動的回復試行）
    ↓
回復成功 → Normal へ
回復失敗（10秒継続）→ PauseLearner + ThrottleRebuild
回復失敗（30秒継続）→ Critical + RejectNewPublication
```

#### 9.6 【P1】DSP Configuration Freeze Detection

**背景**: ユーザー体験上の「音が変になった」に直結。active revision と requested revision の乖離が 60秒継続で Critical。

**コード確認**:

- `getActiveRuntimeDSP()` で現在の DSPCore を取得可能
- `lastCommittedConvolverStructuralHash_` (`AudioEngine.h:1642`) と `rebuildRequestGeneration` の比較

**監視**:

```text
requestedRevision > activeRevision
    かつ
    60秒継続
    ↓
Critical + ForcePublicationRecovery
```

#### 9.7 【P0】Snapshot Starvation 監視

**背景**: 実ログで `DEFERRED → RELEASED → SUPPRESSED` 連鎖で Snapshot が永久に適用されなかった。`DeferredPublish` の TTL 切れ検出だけでは不十分で、Snapshot そのものが飢餓状態にあることを直接検出する必要がある。

**コード確認**:

- `maxDeferredAgeMs_` (`RuntimePublicationOrchestrator.h:140`): 既存の deferred publish 最長滞留時間カウンタ
- `oldestPendingAge_` (`AudioEngine.h:1596`): 既存の最長滞留時間モニタ
- `oldestPendingAgeMs` (`RuntimeDrainAudit.h:33`): 監査ログに含まれているが HealthMonitor 未接続

**新規イベント**:

```cpp
static constexpr uint32_t EVENT_SNAPSHOT_STARVATION = 5003;
```

**新規監視** (`HealthMonitor` 追加):

```cpp
void checkSnapshotStarvation() noexcept {
    if (m_orchestrator == nullptr) return;
    const uint64_t maxDeferredAgeMs = m_orchestrator->getMaxDeferredAgeMs();
    // 10秒以上の滞留 → Snapshot Starvation
    if (maxDeferredAgeMs > 10'000) { // 10秒超
        // Warning
        emitOnTransition(m_prevSnapshotStarvationState, MonitorState::Warning, ...);
        if (maxDeferredAgeMs > 30'000) { // 30秒超 → Error
            emitOnTransition(m_prevSnapshotStarvationState, MonitorState::Error, ...);
        }
    } else {
        m_prevSnapshotStarvationState = MonitorState::Normal;
    }
}
```

**Policy**: Snapshot Starvation は Recovery が必要。抑制だけでは凍結が継続する。

```text
EVENT_SNAPSHOT_STARVATION (Warning)
    ↓
RecoveryMode::RecoveryAttempt → ForcePublicationRecovery

EVENT_SNAPSHOT_STARVATION (Error)
    ↓
RecoveryMode::ForcedRecovery → ForcePublicationRecovery + PauseLearner
```

#### 9.8 【P0】Pending IR Deployment 監視

**背景**: 実ログの障害中心。`IR生成成功 → 適用失敗` の直接検出。`generatedIRRevision` と `publishedIRRevision` の差分が10秒以上継続で発火。

**コード確認**:

- `lastCommittedConvolverStructuralHash_` (`AudioEngine.h:1642`): 最終適用済み Convolver IR のハッシュ
- `pendingIRGeneration` (`AudioEngine.h:1703`): 要求された最新 IR 世代（Message/UI thread only）
- `RebuildTelemetryEvent::Suppressed` (`AudioEngine.h:2007`): Rebuild 抑制時に発行されるイベント

**新規イベント**:

```cpp
static constexpr uint32_t EVENT_PENDING_IR_DEPLOYMENT = 5004;
```

**新規監視**:

```cpp
void checkPendingIRDeployment() noexcept {
    // pendingIRGeneration は Message Thread のみのため、安全に読み取り可能
    // または Rebuild の Suppress 累積回数で代用
    if (m_suppressedIRCountRef == nullptr) return;
    const uint64_t suppressedCount = convo::consumeAtomic(*m_suppressedIRCountRef, ...);
    if (suppressedCount > 3) {  // 3回以上 Suppress
        emitOnTransition(m_prevIRDeployState, MonitorState::Error, ...);
    } else if (suppressedCount > 1) {
        emitOnTransition(m_prevIRDeployState, MonitorState::Warning, ...);
    }
}
```

**Policy**:

```text
EVENT_PENDING_IR_DEPLOYMENT (Warning)
    ↓
ForcePublicationRecovery（能動的回復試行）

EVENT_PENDING_IR_DEPLOYMENT (Error)
    ↓
ForcePublicationRecovery + PauseLearner + RejectNewPublication
```

#### 9.9 【P1】NoiseShaper Learner Stall 監視

**背景**: `LearnerBackpressure`(9.4) が FIFO 使用率を監視するのに対し、Learner Stall は**学習自体の進行停止**を監視する。実ログの `iter=1150, bufferedSamples=3840000` は Learner が動作しているように見えて実質的に停止している（FIFO 満杯で新しい入力が処理されない）。

**コード確認**:

- `NoiseShaperLearner.cpp:913`: `bufferedSamples` ログ出力あり。`segmentBuffer.getNumAvailableSamples()` で現在のバッファリング量を取得可能
- `stopLearning()` (`NoiseShaperLearner.h:105`): 既存の停止関数
- `isRunning()` (`NoiseShaperLearner.h:106`): 既存の稼働確認関数

**新規イベント**:

```cpp
static constexpr uint32_t EVENT_LEARNER_STALL = 5005;
```

**監視**:

```text
bufferedSamples > kLearnerFifoSoftLimit (>90%)
    AND
30秒以上改善なし（lastImprovementTime 基準）
    ↓
EVENT_LEARNER_STALL → SuspendLearning

bufferedSamples > kLearnerFifoHardLimit (>95%)
    AND
60秒以上改善なし
    ↓
Critical + SuspendLearning
```

**注意**: `segmentBuffer.getNumAvailableSamples()` のスレッド安全性は別途確認が必要。現状は Learner ワーカースレッドのみアクセスするため、Monitor(Non-RT)からの安全な読み取りには `std::atomic` ラップまたは SPSC 経由の転送が必要。

#### 9.10 【P2】Configuration Drift 監視

**背景**: Oversampling 変更要求（例: 768kHz→384kHz）が Snapshot Suppress により未適用になるのを検出。

**コード確認**:

- `manualOversamplingFactor` (`AudioEngine.h:1714`): **要求値**（ユーザー設定）
- `DSPCore::oversamplingFactor` (`AudioEngine.h:687`): **実際の稼働値**
- `ISRRuntimeSemanticSchema.h:314`: `oversamplingFactor` が World の semantic schema に含まれる

**新規イベント**:

```cpp
static constexpr uint32_t EVENT_CONFIGURATION_DRIFT = 5006;
```

**監視**:

```text
requestedOversamplingFactor != activeOversamplingFactor
    AND
30秒継続
    ↓
EVENT_CONFIGURATION_DRIFT (Warning)

60秒継続
    ↓
Critical + ForcePublicationRecovery
```

**注意**: `activeOversamplingFactor` は DSPCore から取得する必要がある。DSPCore が nullptr の場合（凍結中）は active 値が取得できないため、その場合は alternative として直近の committed World の oversamplingFactor を使用する。

#### 9.11 【P1】oldestPendingRetireAge 監視

**背景**: 実ログで `routerPendingRetire=2, oldestAgeMs=557` が出ていた。pendingRetireCount だけでなく、**最古エントリの年齢**が重要。500ms 超えで Warning、5000ms 超えで Error。

**コード確認**:

- `oldestRetirePendingGeneration_` (`AudioEngine.h:1594`): 最古の pending retire の generation
- `oldestPendingAge_` (`AudioEngine.h:1596`): 最古の全般滞留時間（ms）
- `checkRetireReclaimLatency()` (`RuntimeHealthMonitor.cpp:394`): 5秒/30秒の閾値で既存監視。ただしこれは `maxRetireAge` であり、Router の pending retire 個別年齢ではない。

**新規監視**（`checkRetireReclaimLatency` 内に統合、または新規関数）:

```cpp
// 既存の checkRetireReclaimLatency を拡張
void checkRetireStallAge() noexcept {
    if (m_oldestPendingAgeRef == nullptr) return;
    const double oldestAgeMs = convo::consumeAtomic(*m_oldestPendingAgeRef, ...);

    MonitorState newState;
    uint32_t eventCode;
    if (oldestAgeMs > 5000.0) {           // >5秒
        newState = MonitorState::Error;
        eventCode = EVENT_RETIRE_AGE_CRITICAL;
    } else if (oldestAgeMs > 500.0) {     // >500ms
        newState = MonitorState::Warning;
        eventCode = EVENT_RETIRE_AGE_WARNING;
    } else {
        newState = MonitorState::Normal;
        // eventCode 不要
    }
    emitOnTransition(m_prevRetireAgeState, newState, ..., eventCode, ...);
}
```

**Policy**: `EVENT_RETIRE_AGE_WARNING` が `oldestPendingAgeMs` から来た場合、retire stall の根因特定に使用。`ForceRetireDrain` で改善するかを確認する診断補助。

#### 9.12 RecoveryMode — 回復段階の明示的管理

**背景**: 現計画は「抑制する」のみで「回復する」段階がない。以下の3段階を新設。

```cpp
enum class RecoveryMode : uint8_t {
    None,              // 回復不要
    Throttled,         // 抑制中（現在の設計の最大値）
    RecoveryAttempt,   // 能動的回復試行中
    ForcedRecovery     // 強制回復モード
};
```

**各 Phase の RecoveryMode 割り当て**:

| Phase | 状態 | RecoveryMode | Action |
| --- | --- | --- | --- |
| Retire Stall 検出直後 | Warning | Throttled | ThrottleRebuild |
| 5秒経過改善なし | Warning | **RecoveryAttempt** | **ForcePublicationRecovery** |
| 10秒経過改善なし | Error | **RecoveryAttempt** | ForcePublicationRecovery + PauseLearner |
| 30秒経過改善なし | Critical | **ForcedRecovery** | RejectNewPublication + PauseLearner |
| Configuration Divergence | Error | RecoveryAttempt | ForcePublicationRecovery |
| Snapshot Starvation | Error | ForcedRecovery | ForcePublicationRecovery + PauseLearner |

**新規イベント**:

```cpp
static constexpr uint32_t EVENT_FORCED_RECOVERY_ACTIVATED = 6001;
```

#### 9.13 PolicySource 拡張（Phase 9 統合版）

```cpp
enum class PolicySource : uint8_t {
    // 既存: 12 source (RetireStall — EmergencyDrain)
    ...
    // ★ Phase 9 追加:
    ConfigurationDivergence,    // 9.2 published/requested revision gap
    LearnerBackpressure,        // 9.4 FIFO usage >90%
    SnapshotStarvation,         // 9.7 deferred publish age >10s
    PendingIRDeployment,        // 9.8 IR generated but not published
    LearnerStall,               // 9.9 Learner FIFO full + no improvement
    ConfigurationDrift,         // 9.10 oversampling factor mismatch
    _Count  // 17 source に拡張
};
```

#### 9.14 RecoveryAction 拡張（Phase 9 統合版）

```cpp
enum class RecoveryAction : uint8_t {
    // 既存: 7 action
    None,
    ForceRetireDrain,
    ThrottleRebuild,
    ClearDeferredPublish,
    ForceCrossfadeReset,
    RejectNewPublication,
    EmergencyDrain,
    // ★ Phase 9 追加:
    PauseLearner,              // 9.1 Learner 一時停止
    ResumeLearner,             // 9.1 Learner 再開
    ForcePublicationRecovery,   // 9.5 能動的回復試行
    SuspendLearning            // 9.9 Learner 強制停止（Pause より強力）
};
```

#### 9.15 イベントコード一覧（Phase 9 追加分）

```cpp
static constexpr uint32_t EVENT_CONFIGURATION_DIVERGENCE        = 5001;  // 9.2
static constexpr uint32_t EVENT_LEARNER_BACKPRESSURE            = 5002;  // 9.4
static constexpr uint32_t EVENT_SNAPSHOT_STARVATION             = 5003;  // 9.7
static constexpr uint32_t EVENT_PENDING_STRUCTURAL_DEPLOYMENT   = 5004;  // 9.8/9.10bis
static constexpr uint32_t EVENT_LEARNER_STALL                   = 5005;  // 9.9
static constexpr uint32_t EVENT_CONFIGURATION_DRIFT             = 5006;  // 9.10
static constexpr uint32_t EVENT_FORCED_RECOVERY_ACTIVATED       = 6001;  // 9.12
static constexpr uint32_t EVENT_LEARNER_PUBLISH_BLOCKED         = 5007;  // 9.17
static constexpr uint32_t EVENT_ROLLBACK_TO_HEALTHY             = 6002;  // 9.16
static constexpr uint32_t EVENT_RETIRE_BLOCKER_CAPTURED         = 7001;  // 9.19
```

#### 9.16 【P0】RollbackToLastHealthyWorld

**背景**: 実ログの `IR生成成功 → publish suppressed → 古いDSP継続` からの唯一の回復手段。最後に正常だった PublishWorld へ復帰し、異常な publish 候補を破棄する。

**コード確認**:

- `ISRRetireRuntimeEx.h:105-110`: **ロールバック基盤が既に存在**。`rollbackReady_`, `rollbackModeRaw_`, `rollbackGlobalEnabled_`, `rollbackRetirePathOnlyEnabled_` 完備。
- `ISREvidenceExporter.cpp:274`: ロールバック状態の JSON 出力も既存。
- `RuntimeBuilder.h:31`: `setHealthStateRef()` パターン既存。同様に最終健全 World ID の参照設定が可能。
- 新規: `lastHealthyWorldId`, `lastHealthyPublicationSequence`, `lastHealthyPublishTimeUs` の追跡。

**新規 RecoveryAction**:

```cpp
enum class RecoveryAction : uint8_t {
    ...
    RollbackToLastHealthyWorld  // ★ 新規
};
```

**新規 HealthMonitor 監視項目**:

```cpp
// AudioEngine 側で正常 publish 完了時に更新
void notifyHealthyPublication(uint64_t worldId, uint64_t pubSeq) noexcept {
    convo::publishAtomic(lastHealthyWorldId_, worldId, std::memory_order_release);
    convo::publishAtomic(lastHealthyPublicationSequence_, pubSeq, std::memory_order_release);
    convo::publishAtomic(lastHealthyPublishTimeUs_, getCurrentTimeUs(), std::memory_order_release);
}
```

**発動条件**:

```text
SnapshotStarvation > 30s
    または
PendingStructuralDeployment gap >= 3
    または
ForcedRecovery モード遷移
```

**Rollback前のFingerprint 3者一致確認**: Rollback発動前に UI/PublishedWorld/CurrentDSP の Fingerprint を比較。全一致時のみRollback許可。不一致時は EnterSafeMode へ。

**Rollback後のAtomic変数同期(Sync Back)**: RollbackはDSP実行系のみ巻き戻す。manualOversamplingFactor/convHCFilterMode等のAtomic変数が障害時のまま残ると再発ループに陥る。Rollback先WorldのSemanticSchemaから値を抽出して上書き同期し、sendChangeMessage()でUIに反映する。

**RollbackToLastHealthyWorld の動作**:

```cpp
case RecoveryAction::RollbackToLastHealthyWorld:
    // 1. 現在の Publish 候補を破棄
    runtimeOrchestrator_->clearDeferredForShutdown();
    clearPendingRebuilds();

    // 2. 最後に正常だった世界へ復帰
    //    既存の rollback 基盤を使用（ISRRetireRuntimeEx）
    retireRuntimeEx_.configureRollback(
        EpochMode::Shared, true, false, false, true);
    m_epochDomain.publishEpoch();
    m_epochDomain.tryReclaim();

    // 3. Learner を停止（異常状態での学習継続を防止）
    stopNoiseShaperLearning();

    // 4. Admission 再開（新規 Rebuild を受け付ける）
    convo::publishAtomic(retirePressureAdmissionStrict_, false, ...);

    // ★ v5.1: Sync Back — Atomic変数をRollback先Worldの値で上書き
    const auto* rolledBackWorld = getRuntimeSnapshotFromReadHandle(...);
    if (rolledBackWorld) {
        convo::publishAtomic(manualOversamplingFactor,
            rolledBackWorld->resource.oversamplingFactor, ...);
        // convHCFilterMode、noiseShaperType 等他のAtomic変数も同様に同期
    }
    // ★ UI通知 — UIに巻き戻しを反映
    sendChangeMessage();
    break;
```

**発動後の期待**: Throttle 解除 + Learner 停止 + 新規 Rebuild 再開 → 正常復帰の可能性が高い。

#### 9.17 【P0】LearnerPublishBlocked 監視

**背景**: 実ログの `iter=1150, bufferedSamples=3840000` は Learner 自体は動作しているが、学習結果が DSP へ反映されない状態。`LearnerBackpressure`(9.4) は FIFO 使用率、`LearnerStall`(9.9) は FIFO 停滞を見るが、**学習結果の未反映**を直接検出するものではない。

**新規イベント**:

```cpp
static constexpr uint32_t EVENT_LEARNER_PUBLISH_BLOCKED = 5007;
```

**コード確認**: `learningRuntimeState` (`AudioEngine.h:1699`, enum `Idle/WaitingForDSP/Running`) で Learner 稼働状態を直接取得可能。`LearningRuntimeState::Running` かつ publishedRevision 変化なし。

**監視**:

```cpp
void checkLearnerPublishBlocked() noexcept {
    // ★ v4.7 具体化: iterDelta>100 + pubDelta==0 + suppressionActive の複合条件
    if (m_learnerIterationRef == nullptr || m_publishedRevisionRef == nullptr)
        return;
    const uint64_t iter = convo::consumeAtomic(*m_learnerIterationRef, ...);
    const uint64_t publishedRev = convo::consumeAtomic(*m_publishedRevisionRef, ...);
    const uint64_t iterDelta = (iter > m_lastObservedLearnerIter)
        ? (iter - m_lastObservedLearnerIter) : 0;
    const bool pubChanged = (publishedRev != m_lastObservedPublishedRev);

    if (iterDelta > 100 && !pubChanged && m_suppressionActive) {
        // Learner が100回以上 iteration を進めたが published 不変かつ抑制中
        if (m_learnerPublishBlockedStartUs == 0)
            m_learnerPublishBlockedStartUs = getCurrentTimeUs();
        const uint64_t elapsedUs = getCurrentTimeUs() - m_learnerPublishBlockedStartUs;
        if (elapsedUs > 10'000'000) {  // 10秒継続
            emitOnTransition(m_prevLearnerPublishBlockedState, MonitorState::Error, ...);
        }
    } else {
        m_learnerPublishBlockedStartUs = 0;
        m_prevLearnerPublishBlockedState = MonitorState::Normal;
    }
    m_lastObservedLearnerIter = iter;
    m_lastObservedPublishedRev = publishedRev;
    m_lastObservedPublishedRev = publishedRev;
}
```

**Policy**:

```text
EVENT_LEARNER_PUBLISH_BLOCKED (Error)
    ↓
PauseLearner + ForcePublicationRecovery
```

#### 9.18 【P1】RecoveryMode 実動作モード化

**背景**: RecoveryMode を単なる状態記録から実際の動作制御へ昇格させる。

**コード確認**: 既存コードには `RecoveryMode` が存在しない。新規導入。

```cpp
// HealthMonitor 全体の動作モード
enum class RecoveryMode : uint8_t {
    None,              // 通常運転
    Throttled,         // 抑制中（Rebuild 制限あり）
    RecoveryAttempt,   // 能動的回復試行中
    ForcedRecovery     // 強制回復モード
};
```

**RecoveryMode による動作変更**:

| RecoveryMode | Throttle | PauseLearner | Admission | Snapshot優先 | Rebuild優先 |
| --- | --- | --- | --- | --- | --- |
| None | 通常 | 通常 | 通常 | 通常 | 通常 |
| Throttled | 有効 | 通常 | 制限 | 通常 | 抑制 |
| RecoveryAttempt | 有効 | 停止 | 制限 | 優先 | 通常回復 |
| ForcedRecovery | **解除** | 停止 | 全面再開 | 最優先 | 最優先 |

**実装**:

```cpp
void executeRecoveryAction(RecoveryAction action) noexcept {
    // RecoveryMode 更新
    switch (action) {
        case RecoveryAction::ThrottleRebuild:
            m_recoveryMode = RecoveryMode::Throttled;
            ...
        case RecoveryAction::ForcePublicationRecovery:
            m_recoveryMode = RecoveryMode::RecoveryAttempt;
            ...
        case RecoveryAction::RollbackToLastHealthyWorld:
            m_recoveryMode = RecoveryMode::ForcedRecovery;
            // ForcedRecovery では Throttle 解除
            convo::publishAtomic(retirePressureAdmissionStrict_, false, ...);
            ...
    }
}
```

#### 9.19 【P0】RetireStall Root Cause Capture

**背景**: 実ログで最も困ったのは `Retire Stall` を検出しても「誰が epoch を保持していたのか」が特定できなかったこと。`StuckReaderInfo` の各フィールドを evidence として採取する。

**コード確認**:

- `StuckReaderInfo` (`IEpochProvider.h:25-34`): **既存**。`readerIndex`, `readerEpoch`, `enterCount`, `currentEpoch`, `minReaderEpoch`, `pendingRetireCount`, `isStuck`, `residencyTimeUs` 完備。
- `ReaderSlotDetail` (`IEpochProvider.h:17-22`): **既存**。`epoch`, `depth`, `residencyTimeUs`, `active`。
- `detectStuckReaders()` 経由で取得可能。
- `collectDrainAudit()` (`AudioEngine.Threading.cpp:54-85`): 既存の監査機構。

**新規構造体**:

```cpp
struct RetireBlockerEvidence {
    // Reader 情報
    int readerIndex{-1};
    uint64_t enterEpoch{0};
    uint64_t residencyUs{0};
    uint32_t readerDepth{0};

    // Retire 情報
    uint64_t pendingRetireCount{0};
    uint64_t oldestRetireAgeUs{0};
    uint64_t minReaderEpoch{0};
    uint64_t currentEpoch{0};

    // World 情報
    uint64_t publishedWorldCount{0};
    uint64_t retiredWorldCount{0};
    uint64_t activeWorldCount{0};

    // 時系列
    uint64_t captureTimestampUs{0};
};
```

**発動条件**:

```text
EVENT_RETIRE_STALL (Error) 発火時
    または
EVENT_RETIRE_AGE_CRITICAL (Error) 発火時
```

**採取処理**（`onHealthEvent` 内または `checkRetireStall` 内に追加）:

```cpp
void captureRetireBlockerEvidence() noexcept {
    RetireBlockerEvidence evidence{};
    evidence.captureTimestampUs = getCurrentTimeUs();

    // Reader 情報
    const auto stuckInfo = m_retireRouter->detectStuckReaders(10);
    if (stuckInfo.isStuck) {
        evidence.readerIndex = stuckInfo.readerIndex;
        evidence.enterEpoch = stuckInfo.readerEpoch;
        evidence.residencyUs = stuckInfo.residencyTimeUs;
        evidence.readerDepth = 1;
    }

    // Retire 情報
    evidence.pendingRetireCount = m_retireRouter->pendingRetireCount();
    evidence.minReaderEpoch = m_retireRouter->getMinReaderEpoch();
    evidence.currentEpoch = m_retireRouter->currentEpoch();

    // World 情報（WorldLifecycleAudit 経由）
    evidence.publishedWorldCount = m_worldAudit->publishedCount();
    evidence.retiredWorldCount = m_worldAudit->retiredCount();
    evidence.activeWorldCount = m_worldAudit->activeWorldCount();

    // 最古 pending retire age
    evidence.oldestRetireAgeUs = static_cast<uint64_t>(
        std::max(0.0, convo::consumeAtomic(*m_oldestPendingAgeRef, ...)));

    // JSON 出力（ISREvidenceExporter パターンを流用）
    emitRetireBlockerTrace(evidence);
}
```

**出力**: `evidence/retire_blocker_{timestamp}.json` に JSON 出力。`ISREvidenceExporter` パターンを流用。

**PolicySource 拡張**:

```cpp
enum class PolicySource : uint8_t {
    ...
    RetireBlockerEvidence,  // ★ 9.19: 証拠採取（Event 発火なし、副作用のみ）
    _Count
};
```

**RecoveryAction 拡張**:

```cpp
enum class RecoveryAction : uint8_t {
    ...
    RollbackToLastHealthyWorld,  // ★ 9.16
    SuspendLearning               // ★ 9.9（既存. 追記）
};
```

#### 9.22 【P0】AudioQualityDegradation PolicySource

**背景**: Critical 昇格を待たずに Rollback を発動する。ConfigurationDivergence + PendingStructuralDeployment が 30秒以上継続した時点で「音質劣化」と判断し、直接 RollbackToLastHealthyWorld を発動する。

**コード確認**: 既存の `ConfigurationDivergence`(9.2) + `PendingStructuralDeployment`(9.10bis) 検出器の組み合わせで実現可能。新規 check 関数は必要だが、個別の監視ロジックは既存。

```cpp
// 新規: 複合条件からの直接 Rollback
void checkAudioQualityDegradation() noexcept {
    const bool configDiverged = (m_prevConfigDivergenceState != MonitorState::Normal);
    const bool deployBlocked = (m_prevStructuralDeployState != MonitorState::Normal);
    const uint64_t nowUs = getCurrentTimeUs();

    if (configDiverged && deployBlocked) {
        if (m_qualityDegradationStartUs == 0)
            m_qualityDegradationStartUs = nowUs;
        if ((nowUs - m_qualityDegradationStartUs) > 30'000'000) {
            emitOnTransition(m_prevQualityDegradationState,
                MonitorState::Error, HealthEvent::Severity::Error,
                EVENT_ROLLBACK_TO_HEALTHY, 0);
            m_qualityDegradationStartUs = 0;
        }
    } else {
        m_qualityDegradationStartUs = 0;
        m_prevQualityDegradationState = MonitorState::Normal;
    }
}
```

**コード確認（具体ルール化 v4.7）**: `learningRuntimeState` (`AudioEngine.h:1699`, enum `LearningRuntimeState::Running`) で Learner 稼働検出。`SuppressionLoopDetector`(9.24) + `ConfigurationDivergence`(9.2) と複合判定。

```cpp
// ★ v4.7: 3条件複合判定
void checkAudioQualityDegradation() noexcept {
    const bool learnerActive = (m_learningRuntimeState == LearningRuntimeState::Running);
    const bool suppressionLoop = (m_prevSuppressionLoopState != MonitorState::Normal);
    const bool configDiverged = (m_prevConfigDivergenceState != MonitorState::Normal);

    if (learnerActive && suppressionLoop && configDiverged) {
        if (m_qualityDegradationStartUs == 0)
            m_qualityDegradationStartUs = getCurrentTimeUs();
        if ((nowUs - m_qualityDegradationStartUs) > 30'000'000) {
            emitOnTransition(m_prevQualityDegradationState, MonitorState::Error, ...,
                EVENT_ROLLBACK_TO_HEALTHY, 0);
        }
    } else {
        m_qualityDegradationStartUs = 0;
        m_prevQualityDegradationState = MonitorState::Normal;
    }
}
```

**Policy**: AudioQualityDegradation (Error) → RollbackToLastHealthyWorld（Critical 待たず）

#### 9.23 【P0】RestoreLastStableLearnerState

**背景**: `PauseLearner` だけでは「学習途中の悪い係数」が適用済みの場合に回復できない。`getState()`/`setState()` で最終安定状態を保存・復元する。

**コード確認**:

- `LearnedState` (`NoiseShaperLearner.h:59-68`): `bestCoefficients`, `cmaMean`, `bestScore` 完備
- `getState(State&)` (`h:110`): 現在の状態を取得
- `setState(const State&)` (`h:111`): 状態を復元（**既に実装済み！**）
- `bestCoefficients` (`h:274`): `std::array<std::atomic<double>, kOrder>` アトミック係数

**新規 RecoveryAction**:

```cpp
enum class RecoveryAction : uint8_t {
    ...
    RestoreLastStableLearnerState  // ★ 新規
};
```

**実装**:

```cpp
// AudioEngine 側で定期的に安定状態をスナップショット
void snapshotStableLearnerState() noexcept {
    if (noiseShaperLearner == nullptr) return;
    convo::NoiseShaperLearner::State currentState;
    noiseShaperLearner->getState(currentState);
    if (currentState.bestScore < m_lastStableLearnerScore) {
        m_lastStableLearnerState = currentState;
        m_lastStableLearnerScore = currentState.bestScore;
    }
}
```

**発動条件**: `EVENT_LEARNER_BACKPRESSURE` + `EVENT_CONFIGURATION_DIVERGENCE` 同時発生時。

#### 9.24 【P1】SuppressionLoopDetector

**背景**: `DEFERRED → RELEASED → SUPPRESSED` 無限ループを検出。同一 RebuildIntent が 5回連続 Suppress で ForcePublicationRecovery。

**コード確認**: `RebuildTelemetryEvent::Suppressed` (`AudioEngine.h:2007`) 既存。Suppress 3箇所 (RebuildDispatch.cpp:241,428,479)。

```cpp
void onRebuildSuppressed(uint64_t intentId, uint64_t fingerprint) noexcept {
    auto& slot = m_suppressionTracker[intentId % kSuppressionTrackerSize];
    if (slot.fingerprint == fingerprint) {
        if (++slot.count >= kSuppressionLoopThreshold) { // 5回
            convo::publishAtomic(m_suppressionLoopDetected_, true, ...);
            slot.count = 0;
        }
    } else {
        slot.fingerprint = fingerprint;
        slot.count = 1;
    }
}
```

**Policy**: SuppressionLoopDetected → ForcePublicationRecovery → 継続なら RollbackToLastHealthyWorld

#### 9.25 【P1】PendingRetire詳細Evidence

**背景**: `routerPendingRetire=2` の正体を特定。最も古い pending retire の型・revision・滞留時間を記録。

**コード確認**: `DeletionEntry` (`DeferredDeletionQueue.h:25-32`) は既に `type`(DeletionEntryType), `publicationSequenceId`, `generation` を保持。`DeletionEntryType` (`h:19-21`) は現状 `Generic=0` のみ。調査目的で拡張可能。

```cpp
// 既存 DeletionEntryType を拡張
enum class DeletionEntryType : uint8_t {
    Generic = 0,
    DSPCoreNode,        // ★ 追加
    GlobalSnapshot,     // ★ 追加
    EQState,            // ★ 追加
    BandNode,           // ★ 追加
    RefCountedDeferred  // ★ 追加
};
```

**コード確認（v4.7 拡張）**: `DeletionEntry` (`DeferredDeletionQueue.h:25-32`) に `enqueueTimestamp` は現状なし。代わりに `epoch` フィールドから age 計算可能。evidence 採取時に `getCurrentTimeUs() - entryEpochTime` で滞留時間を計算。

**JSON出力例**:

```json
{ "oldestPending": { "type": "DSPCoreNode", "generation": 412,
  "publicationSequenceId": 108, "epoch": 2451, "ageUs": 557000 } }
```

#### 9.30bis ForceSnapshotPublish 事前整合性チェック

**背景**: Retire Stall 中の強制 Publish は World 破壊リスクがある。Practical Stable の観点から以下の事前チェックを必須とする。

```cpp
// ForceSnapshotPublish 発動前の事前整合性チェック
bool canForcePublish() const noexcept {
    // 1. WorldConsistency 確認（published-retired≠active の場合は発動禁止）
    const auto audit = collectDrainAudit();
    if (audit.verifyWorldConsistency() != RuntimeDrainAudit::ConsistencyState::Consistent)
        return false;

    // 2. pendingRetireCount 上限確認（過剰な未回収Worldがある場合は発動禁止）
    if (m_retireRouter->pendingRetireCount() > kMaxPendingForSnapshotPublish)
        return false;

    // 3. 進行中の Crossfade 確認（crossfade中の強制Publishは危険）
    if (crossfadeRuntime_.isPending())
        return false;

    return true;
}
```

#### 9.32 【P0】ActiveReaderBlockerEvidence

**背景**: `routerPendingRetire=2, oldestAgeMs=557` の正体は「誰が epoch を握っているか」である。`ReaderSlot` に既に `ownerTag` と `ownerThreadId` が存在するため、Retire Stall 発生時に「どの Reader が epoch をブロックしているか」を特定可能。

**コード確認**: `ReaderSlot` (`EpochDomain.h:328-339`) に以下が**既に存在**:

- `ownerTag[32]` (line 338): `"AudioThread"`, `"TimerThread"` 等
- `ownerThreadId` (line 337): `std::thread::id` のハッシュ値
- `residencyStartTimestampUs` (line 334): 滞留開始時刻
- `epoch`, `depth`, `enterCount`: 全て既存

```cpp
struct ReaderBlockerEvidence {
    int readerIndex{-1};
    uint64_t readerEpoch{0};
    uint64_t residencyUs{0};
    uint32_t depth{0};
    uint64_t threadId{0};            // ★ ReaderSlot.ownerThreadId
    char ownerTag[32]{};             // ★ ReaderSlot.ownerTag（"AudioThread"等）
    uint64_t pendingRetireCount{0};
    uint64_t oldestRetireAgeUs{0};
    uint64_t captureTimestampUs{0};
};

// Retire Stall 検出時に証拠採取
void captureReaderBlockerEvidence() noexcept {
    ReaderBlockerEvidence ev;
    ev.captureTimestampUs = getCurrentTimeUs();
    const auto stuckInfo = m_retireRouter->detectStuckReaders(10);
    if (stuckInfo.isStuck && stuckInfo.readerIndex >= 0) {
        const auto detail = m_retireRouter->getReaderSlotDetail(stuckInfo.readerIndex);
        ev.readerIndex = stuckInfo.readerIndex;
        ev.readerEpoch = detail.epoch;
        ev.residencyUs = detail.residencyTimeUs;
        ev.depth = detail.depth;
        ev.threadId = detail.threadId;         // ★ ReaderSlot.ownerThreadId
        std::strncpy(ev.ownerTag, detail.ownerTag, sizeof(ev.ownerTag) - 1); // ★ ReaderSlot.ownerTag
    }
    emitReaderBlockerTrace(ev); // evidence/reader_blocker.json
}
```

**出力例**: `evidence/reader_blocker.json`

```json
{ "readerIndex": 3, "ownerTag": "TimerThread", "threadId": 12345,
  "readerEpoch": 2451, "residencyUs": 557000, "depth": 1 }
```

これにより `routerPendingRetire=2` の原因が「TimerThread が epoch 2451 を557ms保持」と特定できる。

#### 9.33 【P0】LearnerFifoSaturation 監視

**背景**: 実ログの `bufferedSamples=3840000` 張り付きは `LearnerPublishBlocked` とは別の現象。Consumer(Learner) が Producer(AudioThread) に追いつかず FIFO が飽和している。

**コード確認**: `AudioSegmentBuffer segmentBuffer` (`NoiseShaperLearner.h:253`)。`segmentBuffer.getNumAvailableSamples()` で現在のバッファリング量を取得可能。

**新規イベント**: `EVENT_LEARNER_FIFO_SATURATION = 5008`

```cpp
void checkLearnerFifoSaturation() noexcept {
    if (m_learnerFifoUsageRef == nullptr) return;
    const double usage = convo::consumeAtomic(*m_learnerFifoUsageRef, ...); // 0.0?1.0
    if (usage > 0.90) {  // 90%超
        if (m_fifoSaturationStartUs == 0)
            m_fifoSaturationStartUs = getCurrentTimeUs();
        if ((getCurrentTimeUs() - m_fifoSaturationStartUs) > 30'000'000) {
            emitOnTransition(m_prevFifoSaturationState, MonitorState::Error, ...);
        }
    } else {
        m_fifoSaturationStartUs = 0;
        m_prevFifoSaturationState = MonitorState::Normal;
    }
}
```

**Policy**: FIFO > 90% が 30s 継続 → `PauseLearner`

#### 9.34 【P0】EnterSafeMode RecoveryAction

**背景**: `ForceMinimalWorldPublish`(9.26) の SafeMode World 内容を具体化。独立した `EnterSafeMode` Action として定義。

**SafeMode World 内容**:

```cpp
worldOwner->routing.convBypassed = true;          // Convolver バイパス
worldOwner->resource.oversamplingFactor = 1;       // Oversampling 1x（固定）
worldOwner->resource.noiseShaperType = 1;          // NoiseShaper Fixed4Tap（Adaptive停止）
// EQ Flat（すべてのバンドを0dBにリセット = デフォルト値を使用）
// Limiter 有効化（出力保護）
```

**新規 RecoveryAction**:

```cpp
enum class RecoveryAction : uint8_t {
    ...
    EnterSafeMode  // ★ 新規: 9.34
};
```

**実装**:

```cpp
case RecoveryAction::EnterSafeMode:
{
    auto safeWorld = createSafeModeWorld();
    if (safeWorld) {
        auto coordinator = makeRuntimePublicationCoordinator();
        coordinator.publishWorld(std::move(safeWorld));
        stopNoiseShaperLearning(); // Adaptive NoiseShaper 完全停止
        diagLog("[SAFEMODE] EnterSafeMode: published safe world");
    }
    break;
}
```

**発動順序**: RollbackToLastHealthyWorld 失敗 → EnterSafeMode（Rollback より安全）

#### 9.35 【P1】SuppressionLoop 振動検出

**背景**: Suppress→Recovery→Suppress→Recovery の振動を検出。10分間に5回以上の Suppress→Recovery サイクルで SafeMode へ。

```cpp
struct SuppressionLoopEvidence {
    uint64_t windowStartUs{0};
    uint32_t suppressCount{0};
    uint32_t recoveryCount{0};
    uint32_t cycleCount() const { return std::min(suppressCount, recoveryCount); }
    bool isOscillating() const {
        const uint64_t elapsedUs = getCurrentTimeUs() - windowStartUs;
        return cycleCount() >= 5 && elapsedUs < 600'000'000; // 10分
    }
};
```

**Policy**: 10分間に Suppress→Recovery 5回以上 → EnterSafeMode

#### 9.36 RecoveryAction 階層化（5段階+Critical）

**背景**: v4.8 で RecoveryAction が18種に増加。PolicyEngine の目的は複雑化ではなく回復の標準化。以下の5段階に整理。

```cpp
enum class RecoveryActionLevel : uint8_t {
    Observe,     // Level0: 監視のみ（Cooldown制御）
    Throttle,    // Level1: 抑制（ThrottleRebuild, PauseLearner）
    Recover,     // Level2: 回復試行（ForceRetireDrain, ClearDeferredPublish）
    Restore,     // Level3: 状態復元（ForceSnapshotPublish, RollbackToLastHealthyWorld）
    Safe,        // Level4: 安全運転（EnterSafeMode/ForceMinimalWorldPublish）
    Critical     // Level5: 臨界（RejectNewPublication, EmergencyDrain）
};
```

**各 RecoveryAction のレベル割り当て**:

| Level | RecoveryAction | 意味 |
| --- | --- | --- |
| Observe | None | 通常運転 |
| **Throttle** | ThrottleRebuild | 新規Rebuild抑制 |
| **Throttle** | PauseLearner | Learner一時停止 |
| **Recover** | ForceRetireDrain | Retire強制Drain |
| **Recover** | ClearDeferredPublish | 滞留Publish解除 |
| **Recover** | ForceCrossfadeReset | Crossfade強制完了 |
| **Recover** | ForcePublicationRecovery | 能動的回復試行 |
| **Restore** | ForceSnapshotPublish | Snapshot強制発行 |
| **Restore** | RollbackToLastHealthyWorld | 最終健全World復帰 |
| **Restore** | RestoreLastStableLearnerState | 最終安定Learner復元 |
| **Safe** | EnterSafeMode | SafeMode World発行 |
| **Safe** | ForceMinimalWorldPublish | 最小構成World発行 |
| **Critical** | RejectNewPublication | 全面新規Publish拒否 |
| **Critical** | EmergencyDrain | 緊急Drain |
| **Critical** | SuspendLearning | Learner完全停止 |

**PolicyEngine 発動ルール**: より高い Level の Action は低い Level より優先。Cooldown は Level 内で個別管理。同一 Level 内の Action は sequence 番号順に試行。

#### 9.2 強化 UI/DSP/PublishedWorld 3者 Fingerprint 比較

**背景**: Oversampling 変更（768kHz→384kHz）の未反映を検出するには、UI状態・PublishedWorld・CurrentDSP の3者比較が必要。

```cpp
struct TripleFingerprint {
    uint64_t uiStateHash;           // AudioEngine.manualOversamplingFactor 等
    uint64_t publishedWorldHash;    // 最新 RuntimePublishWorld.semanticHash
    uint64_t currentDSPHash;        // 稼働中の DSPCore から計算
};
```

**比較**:

```text
UI ≠ Published → ConfigurationDivergence Warning
Published ≠ DSP → SnapshotStarvation Warning
UI ≠ DSP       → ConfigurationDivergence Error（即時）
```

#### 9.27 PolicySource 拡張（Phase 9 最終版 v5.1）

```cpp
enum class PolicySource : uint8_t {
    // 既存: 15 source
    ...
    // ★ v4.5?v4.8:
    RetireBlockerEvidence,
    LearnerPublishBlocked,
    AudioQualityDegradation,
    SuppressionLoop,
    SuppressionDuration,
    PendingRetireIdentity,
    // ★ v5.1:
    ActiveReaderBlocker,          // 9.32
    LearnerFifoSaturation,        // 9.33
    SuppressionLoopOscillation,   // 9.35
    ResetLearner,                   // 9.38
    PendingDeploymentRecovery,      // 9.39
    // ★ v6.2-v6.5: 各種詳細PolicySource
    // ★ v6.6: PolicySource 72→14分類に統合。詳細は Phase 9 各セクションの監視項目が保持。
    RetireStall,                    // Retire系: RetireStall/RetireAge/Overflow/RebuildSuppressionTTL
    PublicationStall,               // 出版系: PublicationStall/ProgressFreeze/ConfigurationDeadlock
    ReaderStuck,                    // Reader系: ReaderSlotUsage/EpochAdvanceBlocked/ActiveReaderBlocker
    CrossfadeTimeout,               // Crossfade系: CrossfadeTimeout/CrossfadeEventDrop
    LearnerAnomaly,                 // Learner系: LearnerBackpressure/LearnerStall/LearnerDivergence/LearnerPublishBlocked
    WorldConsistency,               // World整合性: WorldLeak/WorldConsistency/ConfigurationDivergence
    AudioOutputAnomaly,             // 音響系: DC Offset/Peak Clipping/RMS Jump/Noise Floor
    EmergencyCondition,             // 緊急系: EmergencyDrain/ShutdownTimeout
    RecoveryOutcome,                // 回復結果: Success/NoEffect/Failed
    SafeModeState,                  // SafeMode系: SoftActive/HardActive/RecoveryReady
    _Count  // 14 source (v6.6統合)
};
```

#### 9.28 RecoveryAction 拡張（Phase 9 最終版 v5.1）

```cpp
enum class RecoveryAction : uint8_t {
    // 既存: 9 action
    ...
    // ★ v4.4?v4.8:
    PauseLearner,
    ResumeLearner,
    ForcePublicationRecovery,
    SuspendLearning,
    RollbackToLastHealthyWorld,
    RestoreLastStableLearnerState,
    ForceMinimalWorldPublish,
    ForceSnapshotPublish,
    // ★ v5.1:
    EnterSafeMode,                 // 9.34
    ResetLearner,                   // 9.38
    // ★ v6.2-v6.5: 各種詳細RecoveryAction
    // ★ v6.6: RecoveryAction 44→6レベルに統合。Level内の詳細アクションは各Phaseが保持。
    Observe,                        // Level 0: 監視のみ。HealthEvent記録。
    Throttle,                       // Level 1: 抑制。admissionStrict/PauseLearner/Suppress。
    Recover,                        // Level 2: 回復。ForceRetireDrain/ForceSnapshotPublish/Escalate。
    Restore,                        // Level 3: 復元。Rollback/LearnerRollback/CheckpointRestore。
    Safe,                           // Level 4: 安全確保。SoftSafeMode(ConvByPass+LearnerStop)/HardSafeMode(1x+FlatEQ)。
    Critical,                       // Level 5: 重大。RejectNewPublication/EmergencyDrain/Shutdown。
    _Count  // 6 levels (v6.6統合)
};
```

#### 9.26 【P0】ForceMinimalWorldPublish — Safe Mode World

**背景**: 全回復策失敗時の最終手段。Convolver無効+NoiseShaper停止+Oversampling固定+EQのみの Safe Mode World を生成。

**コード確認**: `RuntimeBuilder::createBootstrapWorld()` (`RuntimeBuilder.cpp:66`) で nullptr DSPCore + 全デフォルト値の最小 World 生成パターン既存。`buildRuntimePublishWorld(nullptr, nullptr, ...)` も prepareToPlay/releaseResources で使用済み。

```cpp
convo::aligned_unique_ptr<RuntimePublishWorld>
AudioEngine::createSafeModeWorld() noexcept {
    auto builder = makeRuntimeBuilder();
    builder.setHealthStateRef(getHealthStateRef());
    auto worldOwner = builder.buildRuntimePublishWorld(
        nullptr, nullptr,
        convo::TransitionPolicy::HardReset, 0.0, true);
    worldOwner->routing.convBypassed = true;     // Convolver バイパス
    worldOwner->resource.oversamplingFactor = 1;  // Oversampling 固定=1x
    worldOwner->resource.noiseShaperType = 0;     // NoiseShaper 無効
    return worldOwner;
}
```

**発動条件**: RollbackToLastHealthyWorld 失敗 + 30秒経過、または ForcedRecovery 60秒経過。

**期待効果**: Convolver無効/Oversampling=1x/NoiseShaper停止/EQのみ → 最低限まともな音を即座に復旧。

#### 9.29 【P0】Suppression Duration 監視

**背景**: 現状は Suppress 回数のみ監視（SuppressionLoopDetector 9.24）。しかし今回の障害では Suppress された「回数」より「継続時間」が重要。30秒/60秒/120秒の段階的エスカレーションを導入。

**コード確認**: `publicationRejectCount_` (`AudioEngine.h:3448`) が6箇所 (RebuildDispatch.cpp:228,243,424,430,466,481) でインクリメント済み。抑制開始時刻を追跡する `firstSuppressedUs` を追加。`rebuildRequestGeneration` (h:1725) と `lastCommittedRebuildGeneration` (h:1726) の差分で Suppress 継続監視。

```cpp
struct SuppressionEvidence {
    uint64_t firstSuppressedUs{0};  // 抑制開始時刻
    uint64_t lastSuppressedUs{0};   // 最終抑制時刻
    uint32_t suppressCount{0};      // 累積抑制回数
    uint64_t durationMs() const {   // 継続時間
        return (lastSuppressedUs - firstSuppressedUs) / 1000;
    }
};

// AudioEngine.RebuildDispatch.cpp: Suppress 箇所に時刻記録追加
void onRebuildSuppressed(uint64_t intentId) noexcept {
    const uint64_t nowUs = getCurrentTimeUs();
    if (m_suppressionEvidence.firstSuppressedUs == 0)
        m_suppressionEvidence.firstSuppressedUs = nowUs;
    m_suppressionEvidence.lastSuppressedUs = nowUs;
    m_suppressionEvidence.suppressCount++;
}
```

**段階的エスカレーション（v4.8 具体化）**:

```cpp
void checkSuppressionDuration() noexcept {
    if (m_suppressionEvidence.firstSuppressedUs == 0) {
        m_prevSuppressionDurationState = MonitorState::Normal;
        return;
    }
    const uint64_t durationUs = getCurrentTimeUs() - m_suppressionEvidence.firstSuppressedUs;

    // ★ v4.8: SuppressionDuration を PolicyEngine の正式な PolicySource として扱い、
    //   各段階で具体的な RecoveryAction を発行する
    if (durationUs > 180'000'000) {       // 180秒超 → Admission全面停止
        emitOnTransition(m_prevSuppressionDurationState, MonitorState::Error, ...,
            EVENT_FORCED_RECOVERY_ACTIVATED, 0);
        // → executeRecoveryAction(RejectNewPublication)
    } else if (durationUs > 120'000'000) { // 120秒超 → Crossfade強制完了
        // → executeRecoveryAction(ForceCrossfadeReset)
    } else if (durationUs > 60'000'000) {  // 60秒超 → DeferredPublish全解除
        // → executeRecoveryAction(ClearDeferredPublish)
    } else if (durationUs > 30'000'000) {  // 30秒超 → Retire強制Drain
        // → executeRecoveryAction(ForceRetireDrain)
    }
}
```

**Policy**（段階的エスカレーション）:

```text
30s  → ForceRetireDrain        (軽度: drainで改善するか確認)
60s  → ClearDeferredPublish     (中度: 滞留 publish を全解除)
120s → ForceCrossfadeReset      (重度: crossfade 強制完了)
180s → RejectNewPublication     (臨界: 全面新規 publish 停止)
```

これにより Suppression Duration が単なる検出器から、状況に応じて段階的にエスカレーションする回復機構になる。

#### 9.30 【P0】ForceSnapshotPublish

**背景**: ConfigurationDivergence(revisionGap>=5) 検出時に、滞留中の Snapshot を強制 Publish する。計算系と実行系の乖離を能動的に解消する。

**コード確認**: `RuntimePublicationOrchestrator` に以下が既存:

- `consumeDeferredRequest()` (h:61-68): 保留中の publish request を取得
- `trySubmit()` (h:49): publish 要求を再試行（Admission評価付き）
- `submitPublishRequest()` (h:53): publish 要求を処理（deferred自動enqueue）
- `notifyTransitionComplete()` (h:57): 完了後に deferred 自動再試行

**新規 RecoveryAction**:

```cpp
enum class RecoveryAction : uint8_t {
    ...
    ForceSnapshotPublish  // ★ 新規
};
```

**実装**:

```cpp
// AudioEngine::executeRecoveryAction() 内
case RecoveryAction::ForceSnapshotPublish:
    if (runtimeOrchestrator_ == nullptr) break;
    // 1. 現在のdeferred requestを取得
    auto deferredReq = runtimeOrchestrator_->consumeDeferredRequest();
    if (!deferredReq.has_value()) break;

    // 2. Admissionをバイパスして強制Publish試行
    //    → suppress中のadmissionを一時解除
    const bool prevAdmission = convo::exchangeAtomic(
        retirePressureAdmissionStrict_, false, ...);

    // 3. trySubmit で再試行
    auto result = runtimeOrchestrator_->trySubmit(*deferredReq);

    // 4. 元のadmission状態を復元（次回tickでPolicyEngineが再評価）
    convo::publishAtomic(retirePressureAdmissionStrict_, prevAdmission, ...);

    if (result == PublicationAdmission::Decision::Accepted) {
        diagLog("[RECOVERY] ForceSnapshotPublish: success");
    } else {
        diagLog("[RECOVERY] ForceSnapshotPublish: still blocked ("
            + toString(result) + ")");
    }
    break;
```

**Policy**: ConfigurationDivergence(revisionGap>=5) → ForceSnapshotPublish

#### 9.31 【P1】PendingRetireIdentity PolicySource

**背景**: ログの `routerPendingRetire=2` の正体特定。現在は件数しか分からないが、各 pending retire の「誰が詰まっているか」を特定できれば原因調査が大幅に高速化する。

**コード確認**:

- `ISRRetireRouter::pendingRetireCount()`: 件数のみ。個別エントリの情報は取得不可。
- `DeferredDeletionQueue`: 内部の ring buffer に `DeletionEntry` が格納されているが、public API で個別エントリの identity を取得する手段は存在しない。
- 新規 API 追加が必要: `peekOldestPendingRetire(RetireEntryInfo&)` または `enumeratePendingRetires()` を `ISRRetireRouter` に追加。

```cpp
// ISRRetireRouter に新規追加
struct RetireEntryInfo {
    void* ptr{nullptr};                // 削除対象ポインタ（種別特定に使用）
    DeletionEntryType type;            // Generic/DSPCoreNode/GlobalSnapshot...
    uint64_t generation{0};            // 対象 generation
    uint64_t publicationSequenceId{0}; // publication sequence
    uint64_t enqueueEpoch{0};          // enqueue 時の epoch
};

// 最古の pending retire エントリ情報を取得
[[nodiscard]] bool peekOldestPendingRetire(RetireEntryInfo& info) const noexcept;
```

**JSON出力**:

```json
{ "pendingRetireCount": 2, "oldestEntry": {
    "type": "DSPCoreNode", "generation": 412,
    "publicationSequenceId": 108, "enqueueEpoch": 2451,
    "ptr": "0x000001A2B3C4D5E6"
  }
}
```

**PolicySource 拡張**:

```cpp
enum class PolicySource : uint8_t {
    ...
    PendingRetireIdentity,  // ★ 9.31: 証拠採取（Event発火なし、副作用のみ）
    _Count  // 27 source
};
```

**RetireBlockerEvidence(9.19)連携**: 証拠採取時に `peekOldestPendingRetire()` を呼び出し、JSONに含める。

#### 9.27 PolicySource 拡張（Phase 9 最終版 v4.8）

```cpp
enum class PolicySource : uint8_t {
    // 既存: 15 source (RetireStall — FallbackQueueOverflow)
    ...
    // ★ v4.5:
    RetireBlockerEvidence,      // 9.19
    LearnerPublishBlocked,      // 9.17
    // ★ v4.6:
    AudioQualityDegradation,    // 9.22
    SuppressionLoop,            // 9.24
    // ★ v4.7:
    SuppressionDuration,        // 9.29
    // ★ v4.8:
    PendingRetireIdentity,      // 9.31
    _Count  // 27 source
};
```

#### 9.28 RecoveryAction 拡張（Phase 9 最終版 v4.8）

```cpp
enum class RecoveryAction : uint8_t {
    // 既存: 9 action (None?EmergencyDrain)
    ...
    // ★ v4.4?v4.5:
    PauseLearner,                // 9.1
    ResumeLearner,               // 9.1
    ForcePublicationRecovery,     // 9.5
    SuspendLearning,              // 9.9
    RollbackToLastHealthyWorld,   // 9.16
    // ★ v4.6:
    RestoreLastStableLearnerState,// 9.23
    ForceMinimalWorldPublish,     // 9.26
    // ★ v4.8:
    ForceSnapshotPublish,         // 9.30
    _Count  // 18 action
};
```

#### 9.20 PolicySource 拡張（Phase 9 統合最終版）

```cpp
enum class PolicySource : uint8_t {
    // 既存: 12 source (RetireStall — FallbackQueueOverflow)
    ...
    // ★ Phase 9 追加:
    ConfigurationDivergence,    // 9.2
    LearnerBackpressure,        // 9.4
    SnapshotStarvation,         // 9.7
    PendingStructuralDeployment,// 9.8/9.10bis（旧PendingIRDeployment）
    LearnerStall,               // 9.9
    ConfigurationDrift,         // 9.10
    RetireBlockerEvidence,      // 9.19
    LearnerPublishBlocked,      // 9.17
    _Count  // 20 source
};
```

#### 9.21 RecoveryAction 拡張（Phase 9 統合最終版）

```cpp
enum class RecoveryAction : uint8_t {
    // 既存: 7 action
    None,
    ForceRetireDrain,
    ThrottleRebuild,
    ClearDeferredPublish,
    ForceCrossfadeReset,
    RejectNewPublication,
    EmergencyDrain,
    // ★ Phase 9 追加:
    PauseLearner,                // 9.1
    ResumeLearner,               // 9.1
    ForcePublicationRecovery,     // 9.5
    SuspendLearning,              // 9.9
    RollbackToLastHealthyWorld,   // 9.16
    _Count  // 12 action
};
```

## 2. 実装優先度順タスク一覧

| 優先度 | Phase | タスク | 推定規模 | リスク |
| --- | --- | --- | --- | --- |
| **P0** | 1.1 | DSPLifetimeManager::retire() 戻り値チェック | 5行 | DSPCore リーク |
| **P1** | 1.2 | SnapshotCoordinator enqueueRetry（4/5経路） | 15行 | resetFadeStateAndRetireTarget(L67)はRT除外。switchImmediate追加。 |
| **P0** | 1.3 | RefCountedDeferred::release() 修正 | 10行 | ジェネリックリーク |
| **P0** | 1.4 | EQProcessor retire 戻り値伝播 | 5行 | 実測17箇所（推定59→訂正） |
| **P0** | 1.5 | フォールバックキュー統合 | 30行 | 最終安全網 |
| **P0** | 2.1 | Reader Stuck residency判定追加 | 5行 | 誤検出/見逃し |
| **P0** | 3.1 | ShutdownResult 型追加 | 15行 | 設計合意必要 |
| **P0** | 3.2 | collectResult() 実装 | 15行 | 同上 |
| **P0** | 4.4 | **背圧機構統一（3経路→単一権限化）** | 20行 | 最重要：admissionStrict_ の3重書き込み解消 |
| **P0** | 7.1 | WorldConsistency HealthMonitor連携 | 10行 | Policy設計依存 |
| **P0** | 8.1 | フォールバックキューOOMガード | 10行 | SoftLimit=1000/HardLimit=2000二段構成 |
| **P1** | 0 | RuntimePolicyEngine 基盤 | 50行 | MonitorState駆動型。新閾値体系は導入しない |
| **P1** | 4.1 | HealthMonitor→PolicyEngine統合 | 30行 | evaluateAggregate + Cooldown制御 |
| **P0** | 4.2 | WorldLifecycleAudit→HealthMonitor連携 | 15行 | EVENT_WORLD_LEAK（RejectNewPublication主+ForceRetireDrain補助） |
| **P1** | 6.1 | Deferred Publish TTL + DiscardReason::Expired | 15行 | DiscardReason に Expired 追加 |
| **P1** | 8.2 | EmergencyDrain の PolicyEngine実行時制御 | 20行 | isEmergencyDrainRequested() 追加 |
| **P1** | 8.3 | Deferred Publish TTL 超過通知 | 10行 | UI/API 通知パス |
| **P0** | 9.1 | Learner Health Policy | 15行 | RetireStall 10秒継続→PauseLearner。現状learnerはrebuild中も継続動作（RebuildDispatch.cpp:546） |
| **P0** | 9.2 | RuntimeConfigurationDivergence | 20行 | published/requested revision gap >=3 かつ30秒→Critical。現障害の直接原因検出 |
| **P0** | 9.5 | Retire Stall Auto Recovery | 25行 | ForcePublicationRecovery: 回復試行→失敗時停止の3段階 |
| **P1** | 9.3 | Deferred Rebuild Expiration | 10行 | PolicySuppressed状態追加。3回Suppress→Critical |
| **P1** | 9.4 | Learner Backpressure Monitor | 15行 | bufferedSamples >90% 30秒→PauseLearner |
| **P0** | 9.7 | Snapshot Starvation 監視 | 15行 | maxDeferredAgeMs>10s→Warning, >30s→Error。既存oldestPendingAge_流用可能 |
| **P0** | 9.8/10bis | PendingStructuralDeployment 監視 | 10行 | publicationSequence gap>=3→Error。旧IR限定→全要素汎用化 |
| **P0** | 9.16 | RollbackToLastHealthyWorld | 25行 | ロールバック基盤(ISRRetireRuntimeEx)既存。lastHealthyWorldId新規追跡 |
| **P0** | 9.17 | LearnerPublishBlocked 監視 | 15行 | Learner iteration進行+publishedRevision不変10秒→PauseLearner |
| **P0** | 9.19 | RetireStall Root Cause Capture | 20行 | StuckReaderInfo/ReaderSlotDetail既存。evidence JSON出力追加 |
| **P1** | 9.9 | NoiseShaper Learner Stall 監視 | 15行 | bufferedSamples>90%+30秒改善なし→SuspendLearning |
| **P1** | 9.11 | oldestPendingRetireAge 監視 | 10行 | checkRetireReclaimLatency拡張。500ms Warning, 5000ms Error |
| **P0** | 9.22 | AudioQualityDegradation PolicySource | 15行 | ConfigDivergence+DeployBlocked 30秒→直接Rollback。Critical待たない |
| **P0** | 9.23 | RestoreLastStableLearnerState | 15行 | getState/setState(NoiseShaperLearner.h:110-111)既存。係数復元 |
| **P0** | 9.26 | ForceMinimalWorldPublish SafeMode | 25行 | bootstrapWorldパターン流用。Convolver無効+OS固定+EQのみ |
| **P0** | 9.29 | Suppression Duration→RecoveryAction接続 | 15行 | 30s→ForceRetireDrain→60s→ClearDeferredPublish→120s→ForceCrossfadeReset→180s→RejectNewPublication。publicationRejectCount_既存 |
| **P0** | 9.30 | ForceSnapshotPublish | 20行 | consumeDeferredRequest+trySubmit+Admission一時解除で滞留Publish強制実行 |
| **P1** | 9.31 | PendingRetireIdentity PolicySource | 15行 | ISRRetireRouterにpeekOldestPendingRetire追加。pointer/type/gen追跡 |
| **P0** | 9.32 | ActiveReaderBlockerEvidence | 15行 | ReaderSlot.ownerTag/ownerThreadId既存(EpochDomain.h:337-338)。reader_blocker.json出力 |
| **P0** | 9.33 | LearnerFifoSaturation 監視 | 10行 | segmentBuffer.getNumAvailableSamples()既存。FIFO>90% 30s→PauseLearner |
| **P0** | 9.34 | EnterSafeMode RecoveryAction（二段階化） | 30行 | Soft: ConvByPass+LearnerStop+EQ維持。Hard: OS=1x+FlatEQ+MinimalWorld。createSafeModeWorld流用。 |
| **P0** | 9.37 | RecoveryAction 単一実行ルール | 10行 | 同一tick最大1Action+Level優先 |
| **P0** | 9.38 | ResetLearner FIFO DataRace防止 | 15行 | adaptiveCaptureActiveRt制御+GracePeriod+clearFifo+setState |
| **P0** | 9.39 | PendingDeployment→ForceSnapshotPublish | 10行 | pendingAge>30s+事前条件→ForceSnapshotPublish |
| **P0** | 9.40 | **RuntimeProgressFreeze** | 30行 | 新規 PolicySource + 3軸監視(publish/retire/rebuild)。既存 updateProgressObservation()+WorldLifecycleAudit.lastRetireTimestampUs_+oldestPendingAge_ 流用。実ログ障害の直接検出。 |
| P0 | 9.41 | RebuildSuppressionTTL | 15行 | kMaxSuppressionUs=120s。retirePressureAdmissionStrict長時間維持防止。suppressionStartUs新規追跡。 |
| **P0** | 9.42 | **RetireBlockerSnapshot** | 20行 | `StuckReaderInfo` 拡張。ownerTag/ownerThreadId/epochGap 追加。HealthEvent に RetireBlockerSnapshot 添付。 |
| **P1** | 9.43 | **ForceSnapshotPublish強化版** | 15行 | suppressionDuration > 60s → ForceSnapshotPublish。事前検査(worldConsistency/pendingRetire/crossfade)必須。 |
| **P0** | 9.44 | **LearnerRollback（多世代化）** | 25行 | `lastKnownGoodNoiseShaper_` + 既存`savedStates[6]`(NSL.h:251)流用で4世代リングバッファ(5/15/30/60分前)。RecoveryOutcome失敗時に世代シフト。 |
| **P0** | 9.45 | **RecoveryOutcome**（回復成功判定） | 30行 | `RecoveryOutcome` enum + `evaluateRecoveryResult()`. 閉ループ制御の核心。回復Action後の状態変化(publishedSequence/pendingRetire/adaptiveCaptureActiveRt)でSuccess/NoEffect/Failed/Unsafe判定。 |
| **P0** | 9.46 | **RetireBlockerEvidence強化版** | 15行 | ReaderSlotにpublicationSequence/worldGeneration追加。`enter()`/`exit()`で書込み。detectStuckReaders戻り値に含める。 |
| **P0** | 9.47 | **ForceSnapshotPublish安全条件強化** | 20行 | `canForcePublish()`新規: pendingRetireCount上限50+oldestPendingAge上限10秒+worldConsistency+crossfade+retireRate>=publishRate(60秒窓)。違反時はRollback/SafeModeへ委譲。 |
| **P0** | 9.48 | **SafeMode Learner完全停止** | 20行 | 移行手順厳格化: adaptiveCaptureActiveRt=false→stopLearning→GracePeriod(100us)→FIFO.clear→createSafeModeWorld。 |
| **P0** | 9.49 | **AudioOutputAnomaly（P0昇格）** | 30行 | DC Offset/Peak Clipping/RMS Jump/Noise Floor監視。分析スレッド新設必要。RecoveryOutcomeの音声品質判定に必須のためP0化。 |
| **P1** | 9.50 | **Suppression自動解除（ヒステリシス）** | 15行 | `checkSuppressionAutoRelease()`新規: 設定>100/解除<20+30秒持続。publishedSequence増加確認後解除。振動防止ヒステリシス。 |
| **P0** | 9.51 | **PublicationProgressBlocked** | 15行 | ForceSnapshotPublish禁止ガード。`isPublicationBlocked()`=pendingRetire>50 or oldestPendingAge>10s → Rollback/SafeMode迂回。 |
| **P0** | 9.52 | **EpochAdvanceBlocked** | 25行 | Level別4段階対応: Level1 dump→Level2 HealthEvent添付→Level3 blocker report→Level4 SafeMode。detectStuckReaders()流用。 |
| **P0** | 9.53 | **LearnerOutputDivergence** | 20行 | bestCoefficients L2距離監視。getState()で現在係数取得→lastKnownGoodと比較→乖離時LearnerRollback。kOrder次元数確認済。 |
| **P0** | 9.54 | **SafeModeRecovery** | 30行 | SafeMode→Normal段階復帰。5分安定+retire=0+reader=0+crossfade=0条件。bootstrapWorld逆操作。 |
| **P1** | 9.55 | **Suppression解除条件強化（ヒステリシス）** | 15行 | publishedSequence増加AND条件+設定>100/解除<20+30秒持続。振動防止。 |
| **P0** | 9.56 | **RuntimeRecoveryScore** | 25行 | 4軸(publish/retire/rebuild/audio)各0-25→合計0-100総合スコア。RecoveryOutcomeと併用。 |
| **P0** | 9.47強化 | **retireRate>=publishRateガード** | 10行 | `WorldLifecycleAudit.publishedCount_/retiredCount_`既存流用。60秒窓でretireRate≥publishRate確認。 |
| **P0** | 9.44強化 | **LearnerCheckpoint多世代化+互換性検証** | 15行 | 既存`savedStates[6]`(NSL.h:251)流用。4世代リングバッファ(5/15/30/60分前)。sampleRate/processingRate/oversamplingFactor/contextGeneration不一致時Rollback禁止。 |
| **P0** | 9.57 | **RuntimeConfigurationDeadlock** | 25行 | publishedSequence停滞+suppressedRebuild増加+pendingDeployment>0の30秒継続→ClearSuppression→ForceSinglePublish。suppressedRebuildCount追跡+RecoveryOutcome判定軸追加。 |
| **P0** | 9.58 | **RetireRootCauseUnknown** | 15行 | Retire Stall原因特定不可→SafeModeのみ許可。canIdentifyRetireRootCause()判定。activeReaderCount既存流用。 |
| **P0** | 9.59 | **RetireProgressFrozen（強化版）** | 20行 | published>0且retiredCount不変30s→直接検出。Learner即停止(3条件OR)。PendingRetireEvidence強化。RecoveryOutcome audioRecovered。AudioQualityFingerprint。ProgressFreeze+Anomaly+Failed→SafeMode。 |
| **P0** | 9.60 | **PublicationPriority（縮退運転）** | 25行 | RebuildTelemetryClass(Structural/Snapshot)既存流用。RetireStall時もStructural rebuildは許可。shouldRejectRebuildAdmissionForPressure()拡張。実ログSuppress防止に最も効果的。 |
| **P0** | 9.61 | **RetireRootCauseEvidence** | 15行 | 原因6分類(ReaderNeverExited/EpochNeverAdvanced/RetireQueueOverflow/RouterPendingRetire/CrossfadeReferenceHeld/Unknown)。detectStuckReaders流用。 |
| **P0** | 9.62 | **AudioOutputDivergence PolicySource** | 10行 | fingerprintDistance独立監視。LearnerRollback(9.63)トリガー。 |
| **P0** | 9.63 | **LearnerRollback自動発動** | 15行 | RuntimeProgressFreeze+AudioAnomaly+Divergence→auto rollback。最適Checkpoint選択+resume。 |
| **P0** | 9.64 | **RetireProgress→SafeMode即時** | 10行 | RetireProgressFrozen+LearnerActive+suppressCount超過→SoftSafeMode即移行。回復試行スキップ。 |
| **P0** | 9.65 | **ClearSuppression独立Action** | 15行 | retirePressureAdmissionStrict_一時解除(5秒)+自動再設定。回復順序明確化。PendingRetireObjectInfo(DeletionEntry既存+enqueueTimestampUs追加)。AudioQualityFingerprint正式化(HF Energy/THD proxy追加)。 |
| **P0** | 9.34強化 | **SafeMode二段階化** | 15行 | Soft: ConvByPass+LearnerStop+EQ維持→Hard: OS=1x+FlatEQ+MinimalWorld。 |
| **P0** | 9.49→P0 | **AudioOutputAnomaly P0昇格** | — | RecoveryOutcomeの音声品質判定に必須のためP1→P0昇格。 |
| **P0** | 9.50強化 | **Suppressionヒステリシス** | 10行 | 設定>100/解除<20+30秒持続+publishedSeq確認。 |
| **P1** | 9.35 | SuppressionLoop 振動検出 | 10行 | Suppress→Recovery 5回/10分→EnterSafeMode |
| **P1** | 9.24 | SuppressionLoopDetector | 10行 | 同一Intent 5回Suppress→ForcePublicationRecovery |
| **P1** | 9.25 | PendingRetire詳細Evidence | 10行 | DeletionEntry.generation/publicationSequenceId既存流用 |
| **P2** | 9.10 | Configuration Drift 監視 | 10行 | manualOversamplingFactor≠DSPCore.oversamplingFactor。30s Warning, 60s Error |
| **P3** | 5.1 | BlockingReason Primary+Secondary | 15行 | 不採用。単一で十分。 |
| **P3** | - | Deferred Publish トレーサビリティ | 10行 | 任意改善 |

---

## 3. ファイル影響マップ

| ファイル | Phase | 変更内容 |
| --- | --- | --- |
| NEW: `RuntimePolicyEngine.h/.cpp` | 0, 4.1, 4.4 | PolicyEngine クラス（背圧統合含む） |
| `DSPLifetimeManager.h` | 1.1 | enqueueRetire戻り値チェック |
| `SnapshotCoordinator.h/.cpp` | 1.2 | enqueueRetry追加（4/5経路、resetFadeStateAndRetireTargetはRT除外）。switchImmediate追加。 |
| `RefCountedDeferred.h` | 1.3 | enqueueRetire戻り値チェック |
| `EQProcessor.Core.cpp` / `EQProcessor.h` | 1.4 | retireEQStateDeferred bool化 |
| `AudioEngine.h` / `AudioEngine.Retire.cpp` | 1.5 | fallback queue統合 |
| `EpochDomain.h` | 2.1 | detectStuckReaders residency条件 |
| `ISRShutdown.h/.cpp` | 3.1 | ShutdownResult（Primary+Secondaryは不要→P3格下げ） |
| `ReleaseResources.cpp` | 3.3, 7.1 | collectResult, WorldConsistency |
| `RuntimeHealthMonitor.h/.cpp` | 4.1, 4.4 | PolicyEngine連携 + injectBackpressureSignal()追加。PolicyContext削除。 |
| `WorldLifecycleAudit.h` | 4.2 | HealthEventCallback |
| `AudioEngine.Retire.cpp` | 4.4 | evaluateRetirePressureLevelNoRt の直接背圧→injectBackpressureSignal() に置換 |
| `AudioEngine.Timer.cpp` | 4.4 | onHealthEvent 内の admissionStrict_ 直接設定削除（executeRecoveryAction に移行） |
| `RuntimePublicationOrchestrator.h/.cpp` | 6.1 | DeferredPublishSlot TTL |
| NEW: `AudioEngine.h` / `AudioEngine.Timer.cpp` | 9.1, 9.5 | executeRecoveryAction 拡張: PauseLearner, ResumeLearner, ForcePublicationRecovery |
| NEW: `RuntimeHealthMonitor.h/.cpp` | 9.2, 9.4, 9.6 | checkConfigurationDivergence(), checkLearnerBackpressure() 追加 |
| NEW: `RuntimePolicyEngine.h/.cpp` | 9.7, 9.8 | PolicySource::ConfigurationDivergence/LearnerBackpressure 追加。RecoveryAction拡張 |
| NEW: `AudioEngine.RebuildDispatch.cpp` | 9.3 | suppressedStructuralRebuildCount_ インクリメント追加 |
| NEW: `RuntimePublicationState.h` | 9.3 | DiscardReason::PolicySuppressed 追加 |
| NEW: `RuntimeHealthMonitor.h/.cpp` | 9.7, 9.8, 9.9, 9.10, 9.11 | checkSnapshotStarvation(), checkPendingIRDeployment(), checkLearnerStall(), checkConfigurationDrift(), checkRetireStallAge() 追加 |
| NEW: `RuntimePolicyEngine.h/.cpp` | 9.12-9.14 | RecoveryMode enum追加。PolicySource 12→17拡張。RecoveryAction 7→11拡張。イベントコード7種追加。 |
| NEW: `AudioEngine.RebuildDispatch.cpp` | 9.8 | suppressedIRCount_ インクリメント追加 |
| NEW: `AudioEngine.h` | 9.10 | RuntimeConfigurationFingerprint 参照取得用関数追加 |
| NEW: `AudioEngine.h` / `AudioEngine.Timer.cpp` | 9.16, 9.17, 9.19 | lastHealthyWorldId追跡+notifyHealthyPublication。RetireBlockerEvidence採取。checkLearnerPublishBlocked |
| NEW: `ISRRetireRuntimeEx.h/.cpp` | 9.16 | RollbackToLastHealthyWorld 発動（既存rollback基盤利用） |
| NEW: `ISREvidenceExporter` | 9.19 | retire_blocker_{timestamp}.json 出力追加 |
| NEW: `RuntimePolicyEngine.h/.cpp` | 9.27-9.28 | PolicySource 22種, RecoveryAction 15種に拡張 |
| NEW: `NoiseShaperLearner.h` | 9.23 | snapshotStableLearnerState定期呼び出し追加（getState/setState既存） |
| NEW: `AudioEngine.RebuildDispatch.cpp` | 9.24 | SuppressionLoopカウンタ追加（Suppress 3箇所） |
| NEW: `DeferredDeletionQueue.h` | 9.25 | DeletionEntryType 拡張（Generic→6種） |
| NEW: `AudioEngine.RebuildDispatch.cpp` / `AudioEngine.h` | 9.29 | SuppressionEvidence追跡追加。firstSuppressedUs/lastSuppressedUs。publicationRejectCount_流用 |
| NEW: `RuntimePublicationOrchestrator.h/.cpp` | 9.30 | ForceSnapshotPublish: consumeDeferredRequest+trySubmit+Admission一時解除。既存API流用 |
| NEW: `ISRRetireRouter.h/.cpp` | 9.31 | peekOldestPendingRetire()追加。pointer/type/gen/publicationSequenceId追跡 |
| NEW: `core/EpochDomain.h` (流用) | 9.32 | ReaderSlot.ownerTag/ownerThreadId既存。captureReaderBlockerEvidence追加 |
| NEW: `NoiseShaperLearner.h` (流用) | 9.33 | segmentBuffer.getNumAvailableSamples()既存。checkLearnerFifoSaturation |
| NEW: `AudioEngine.h/.cpp` | 9.34 | EnterSafeMode + createSafeModeWorld具体化(OS=1x/ConvBypass/NS Fixed4Tap) |
| NEW: `RuntimePolicyEngine.h` | 9.36 | RecoveryActionLevel enum 6段階追加 |
| NEW: `AudioEngine.h/.cpp` | 9.26 | createSafeModeWorld()追加。bootstrapWorldパターン流用 |
| NEW: `AudioEngine.h/.cpp` | 9.40-9.44 | RuntimeProgressFreeze 3軸監視。suppressionStartUs_/lastKnownGoodNoiseShaper_ 追加。RetireBlockerSnapshot添付。LearnerRollback notifyHealthyPublication統合。 |
| NEW: `RuntimeHealthMonitor.h/.cpp` | 9.40 | checkRuntimeProgressFreeze()追加。ProgressFreezeSnapshot収集。 |
| NEW: `RuntimePolicyEngine.h/.cpp` | 9.40-9.44 | PolicySource 38種(4追加)。RecoveryAction 26種(5追加)。 |
| NEW: `IEpochProvider.h` | 9.42 | RetireBlockerSnapshot構造体追加。StuckReaderInfo拡張。 |
| NEW: `EpochDomain.h` | 9.42 | detectStuckReaders()にownerTag/ownerThreadId追加。 |
| NEW: `AudioEngine.Retire.cpp` | 9.41 | retirePressureAdmissionStrict_設定箇所に抑制開始時刻記録。 |
| NEW: `RuntimePolicyEngine.h/.cpp` | 9.45 | `RecoveryOutcome` enum追加。`evaluateRecoveryResult()`追加。閉ループ制御。 |
| NEW: `core/EpochDomain.h` | 9.46 | `ReaderSlot`に`publicationSequence`/`worldGeneration`追加。`enter()`/`exit()`で書込み。 |
| NEW: `audioengine/AudioEngine.h/.cpp` | 9.47 | `canForcePublish()`追加。`kForcePublishMaxPendingRetire`/`kForcePublishMaxAgeSec`定数。 |
| NEW: `audioengine/AudioEngine.h/.cpp` | 9.48 | `enterSafeMode()`にLearner完全停止手順追加。GracePeriod+clearFifo。 |
| NEW: `audioengine/AudioOutputMonitor.h/.cpp` | 9.49 | 新規ファイル。DC Offset/Peak Clipping/RMS Jump/Noise Floor監視。分析スレッド。 |
| NEW: `audioengine/AudioEngine.Retire.cpp` | 9.50 | `checkSuppressionAutoRelease()`追加。Timer tickから毎tick呼び出し。 |
| NEW: `RuntimeHealthMonitor.h/.cpp` | 9.52 | `checkEpochAdvanceBlocked()`追加。Level別4段階対応。 |
| NEW: `RuntimeHealthMonitor.h/.cpp` | 9.53 | `checkLearnerOutputDivergence()`追加。L2距離計算。 |
| NEW: `audioengine/AudioEngine.h/.cpp` | 9.54 | `canRecoverFromSafeMode()`/`recoverFromSafeMode()`追加。復帰条件+段階手順。 |
| NEW: `RuntimePolicyEngine.h/.cpp` | 9.56 | `computeRuntimeRecoveryScore()`追加。4軸スコアリング（診断用）。 |
| NEW: `WorldLifecycleAudit.h/.cpp` | 9.47強化 | `retireRate60s()`/`publishRate60s()`追加。既存atom流用。 |
| NEW: `NoiseShaperLearner.h/.cpp` | 9.44強化 | `savedStates[6]`流用で4世代リングバッファ管理。`checkpointShiftOnFailure()`追加。 |
| NEW: `AudioEngine.h/.cpp` | 9.34強化 | `enterSoftSafeMode()`/`enterHardSafeMode()`分割。Soft→Hard昇格条件。 |
| NEW: `RuntimeHealthMonitor.h/.cpp` | 9.57 | `checkConfigurationDeadlock()`追加。suppressCount60s_追跡。 |
| NEW: `RuntimeHealthMonitor.h` | 9.58 | `canIdentifyRetireRootCause()`追加。`RetireRootCauseUnknown` PolicySource。 |
| NEW: `RuntimeHealthMonitor.h/.cpp` | 9.59 | `checkRetireProgressFrozen()`追加。`RecoveryOutcome`→PartialSuccess/Worsened拡張、audioRecovered追加。Learner強制停止ロジック(3条件)追加。 |
| NEW: `ISRRetireRouter.h/.cpp` | 9.31/9.25強化 | `peekOldestPendingRetire()`戻り値拡張→`PendingRetireEvidence`構造体。既存DeletionEntry流用。 |
| NEW: `AudioOutputMonitor.h/.cpp` | 9.49強化 | `AudioQualityFingerprint`+`detectSpectralAnomaly()`追加。spectralCentroid/flatness/crestFactor。 |
| NEW: `AudioEngine.RebuildDispatch.cpp` | 9.60 | `shouldRejectRebuildAdmissionForPressure()`拡張: PublicationPriority引数追加。Structuralは常に許可。 |
| NEW: `RuntimeHealthMonitor.h/.cpp` | 9.61 | `RetireRootCauseEvidence`+`classifyRetireRootCause()`追加。6分類。 |
| NEW: `RuntimeHealthMonitor.h/.cpp` | 9.62-9.64 | `checkAudioOutputDivergence()`+`autoLearnerRollback()`+`emergencySoftSafeMode()`追加。 |
| NEW: `DeferredDeletionQueue.h` | 9.25/9.31強化 | `DeletionEntry`に`enqueueTimestampUs`/`ownerWorldId`/`crossfadeId`追加。`PendingRetireObjectInfo`公開。 |
| NEW: `AudioOutputMonitor.h/.cpp` | 9.62強化 | `AudioQualityFingerprint`正式化: hfEnergy/thdProxy追加。 |
| NEW: `RuntimePolicyEngine.h` | 9.65 | `RecoveryAction::ClearSuppression`追加。5秒タイムアウト+自動再設定。 |

---

## 4. 全未確定事項の確定結果（v7.1 全項目確定済）

> 本設計書内の全未確定事項は Serena MCP / AiDex MCP / grep による実コード照合で確定済。`?❌ 不明` / `?? 検討中` などの未確定ステータスは none。
> 未確定事項ゼロが確認されている。

### 4.1 設計合意点の確定

以下の事項は深掘り調査 + 全コード照合調査で確定した。

#### 確定①: PolicyEngine の配置 → HealthMonitor 内蔵（最小変更）

**調査結果**:

- `RuntimeHealthMonitor` は既に `HealthEventCallback`（`std::function<void(const HealthEvent&)>`）によるコールバック機構を持つ
- `AudioEngine::onHealthEvent()` は既に特定イベントに対する Recovery Action を個別実装している
- PolicyEngine を HealthMonitor のメンバとして追加するのが最小変更

```cpp
// RuntimeHealthMonitor.h 追加メンバ
class RuntimePolicyEngine m_policyEngine;
MonitorState m_prevRouterPendingRetireState{MonitorState::Normal};
```

**根拠**:

- `m_callback` が既に AudioEngine→HealthMonitor の通信路を提供
- PolicyEngine は監視結果を評価するだけで、Callback 経路は既存を流用可能
- 独立シングルトンは過剰。AudioEngine メンバは HealthMonitor との結合度が高すぎる

#### 確定②: ShutdownResult の通知方法 → 戻り値 + JSON（両方）

**調査結果**:

- `emitShutdownTrace()` が既に `evidence/shutdown_trace.json` への JSON 出力を実装済み
- 呼び出し元（`releaseResources()`）は戻り値 `void` だが、上位の管理には戻り値が必要
- `ShutdownRuntime::collectResult()` で戻り値を返し、`emitShutdownTrace()` の JSON に `healthState` を追加

#### 確定③: フォールバックキュー → SoftLimit/HardLimit 二段構成

**調査結果**:

- `DeferredDeletionQueue::kQueueSize = 4096`（`src/DeferredDeletionQueue.h:221`）
- フォールバックキューは通常空。QueueFull 時の最終安全網としてのみ使用
- DeferredRetireFallbackQueue は `std::vector` ベース

**Practical Stable 判断**: 無制限は危険。DeferredDeletionQueue が完全停止するとフォールバック→フォールバックでメモリ無限増殖となる。SoftLimit/HardLimit の二段構成が妥当。

```cpp
constexpr size_t kFallbackSoftLimit  = 1000;   // 超過: PolicyEngine に Critical 昇格要求
constexpr size_t kFallbackHardLimit  = 2000;   // 超過: 強制ドロップ（リーク許容）
```

- **SoftLimit (1000)**: `drainDeferredRetireQueues()` 先頭で PolicyEngine に通知 → Critical 昇格
- **HardLimit (2000)**: `push()` で強制ドロップ。`std::bad_alloc` によるシステム全体クラッシュより少量のリークを許容

#### 確定④: ReaderStuck residency 閾値

| 条件 | 閾値 | 根拠 |
| --- | --- | --- |
| epoch差 > 10 AND residency > 1秒 | Stuck判定 | 1 epoch — ms単位。10 epoch停滞＋1秒滞留で確定 |
| residency > 30秒（epoch差不問） | Stuck判定（重症） | epoch が進まない異常系対応。既存 severe 条件と一致 |
| pendingRetireCount > 100 | severe フラグ | 既存の severe 条件を維持 |

### 4.2 コード調査で解決した事項（v1.1 深掘り版）

| 事項 | 調査結果 | 確定度 |
| --- | --- | --- |
| enqueueRetire 戻り値は全経路で無視されているか | ✅ **主経路(AudioEngine)はチェック＋drain retry済み**。未チェック4経路を特定 | ✅ **確定** |
| SnapshotCoordinator は ISRRetireRouter 経由か | ❌ **通らない**。`IEpochProvider*` → EpochDomain 直接。ISRRetireRouter の tryReclaim 保護なし | ✅ **確定**（Phase1.2重要度↑） |
| onHealthEvent の Recovery Action | ✅ **4種類**（RetireStall/CrossfadeTimeout/PublicationStall/ReaderSlotExhaustion） | ✅ **確定** |
| AuthorityClass の使用状況 | ⚠️ **RuntimePublishWorld の kAuthorityInventory では使用**。監査系のみ未使用 | ✅ **確定** |
| doubleRetireCount → HealthMonitor | ❌ 未連携。ファイル監査のみ | ✅ **確定** |
| DeferredDeletionQueue 容量 | `kQueueSize = 4096` | ✅ **確定** |
| CrossfadeTimeout 閾値 | `kCrossfadeTimeoutUs = 30秒` | ✅ **確定** |
| OverflowRate 閾値 | Warning: 1回/秒, Critical: 5回/秒 | ✅ **確定** |

### 4.3 全未確定事項の確定結果（v4.1 深掘り版）

以下の事項は Serena MCP, CodeGraph MCP, grep/Select-String による実コード調査で確定した。

| # | 事項 | 旧状態 | 調査結果 | 設計反映 |
| --- | --- | --- | --- | --- |
| 1 | **ShutdownResult 上位伝播** | ❌ 部分解決 | `collectDrainAudit()` は既に `.healthState = m_healthMonitor.getHealthState()` を含む。`releaseResources()` の戻り値 void は JUCE AudioProcessor 規約で変更不可。`emitShutdownTrace()` JSON に healthState がないのみ。 | **変更不要**。Phase 3 で JSON に healthState 追加のみ。 |
| 2 | **ExecutionClass/canBlock** | ❌ 未実装 | `isAudioThread()`(DspNumericPolicy.h:116) は Thread Tag 比較で実装済み。`canBlock()` は不要。 | **ExecutionClass 導入せず**。`!isAudioThread()` で判定。 |
| 3 | **SnapshotCoordinator retry** | ❌ 却下 | `startFade()` の呼び出し元は Timer Callback(Non-RT) のみ。`updateFade()`(RT) は enqueueRetire を含まない。`tryReclaim()` は安全。 | **再評価: 実施可能**。却下→P1。static `enqueueWithRetry()` で IEpochProvider を汚染しない。 |
| 4 | **PolicyContext 型** | ✅ **確定** | コードベースに存在しない。設計コンセプトのみ。Serena/grep で全コード検索確認。 | **削除**。PolicyEngine は MonitorState を直接入力。 |
| 5 | **WorldLifecycleAudit 不足** | ✅ **確定** | `WorldLifecycleAudit.h` 既存メソッド8種確認（onWorldPublished/onWorldRetired/activeWorldCount/publishedCount/retiredCount/doubleRetireCount/emitSnapshot/tryDumpPeriodic）。`doubleRetireCount_` 既存。`onFallbackOverflow()`/`injectEvent()` は**コードベースに存在せず**。 | **Phase 4.2 で onFallbackOverflow/injectEvent 追加**。 |
| 6 | **DeferredRetireFallbackQueue** | ✅ **確定** | `src/core/DeferredRetireFallbackQueue.h:15-52`。`push()`→`size_t`（queue_.size()）、`popAll()`→`vector`、`size()`/`empty()` を確認。`std::vector` + `std::mutex` 実装。`estimatedBytes`/`overflowRate`/`notifyOverflow`/`retryCount` 不在。 | **Phase 1.5 で拡張**（push→bool, retryCount, HardLimit）。 |
| 7 | **emitShutdownTrace + healthState** | ✅ **確定** | `ISRShutdown.h:85` `emitShutdownTrace()` 確認。`ISRShutdown.cpp` JSON出力に healthState なし。`collectDrainAudit()` は healthState 収集済み(`AudioEngine.h:1047`)。 | **Phase 3 で JSON に healthState 追加。低コスト。** |
| 8 | **ShutdownBlockingReason 多値化** | ✅ **確定** | `ISRShutdown.h:38-48` 既存8値＋Unknown確認: None/PendingPublication/PendingRetire/ActiveCrossfade/DeferredPublish/QuarantineResident/RouterPendingRetire/ReaderActive/Unknown。`getPrimaryBlockingReason()` で単一理由特定可。Primary+Secondary は過剰。 | **不要**。P2→P3 格下げ（優先度最低）。 |
| 9 | **DiscardReason::Expired** | ✅ **確定** | `RuntimePublicationState.h:8-13` 既存4値: None/ShutdownDiscard/StaleDiscard/SupersededDiscard。`DeferredPublishSlot.enqueueTimestampUs`(`RuntimePublicationOrchestrator.h:33`) 確認済。TTL 判定可能。 | **Phase 6 で Expired 追加**。DeferredPublishSlot 流用可。 |
| 10 | **CONVOPEQ_EMERGENCY_DRAIN** | ✅ **確定** | `ISRShutdown.h:26-29` EmergencyDrain enum値存在。`advancePhase()`(`ISRShutdown.cpp:83`) は `ReclaimComplete→VerifyDrained` でスキップ、`#ifdef`無し。`releaseResources()`(`AudioEngine.Processing.ReleaseResources.cpp:200-246`) は `#ifdef CONVOPEQ_EMERGENCY_DRAIN` で実装。この define は **CMakeLists.txt/CMakePresets.json/全ヘッダ で未定義**。コードパスは常に無効。 | **Phase 8.2 で advancePhase の ifdef 追加 or 明文化**。 |
| 11 | **EQProcessor 呼び出し元数** | ✅ **確定** | `retireEQStateDeferred`: **13箇所**（`EQProcessor.Core.cpp:69`定義。Core:3箇所, Parameters:10箇所）。`retireBandNodeDeferred`: **4箇所**（`EQProcessor.Core.cpp:78`定義。Core:3箇所, Coefficients.cpp:1箇所）。計**17箇所**。Serena/grep で全呼び出し照合済み。 | **設計書の数値を訂正**。影響範囲は推定の29%。 |
| 12 | **collectDrainAudit healthState** | ✅ **確定** | `AudioEngine.h:1047` の `collectDrainAudit()` 内で `m_healthMonitor.getHealthState()` を収集済み。Serena で宣言確認。 | **既に完了**。変更不要。 |

### 4.4 本調査で確定した設計定数

以下の設計定数は実コード調査で確定した:

| 定数 | 値 | 出典 |
| --- | --- | --- |
| `kQueueSize` | 4096 | `DeferredDeletionQueue.h:221` |
| `kMaxReaders` | 64 | `EpochDomain.h` |
| `kCrossfadeTimeoutUs` | 30秒 | `RuntimeHealthMonitor.cpp:215` |
| `kRetireAgeWarningUs` | 5秒 | `RuntimeHealthMonitor.cpp:394` |
| `kRetireAgeCriticalUs` | 30秒 | 同上 |
| `kOverflowRateCriticalThreshold` | 5回/秒 | `RuntimeHealthMonitor.cpp:308` |
| `kOverflowRateWarningThreshold` | 1回/秒 | 同上 |
| `kReaderSlotCriticalThreshold` | 75%/90% | `RuntimeHealthMonitor.cpp:244` |
| `kStuckEpochThreshold` | 10 epoch | `EpochDomain.h` |
| `kStuckResidencyThresholdUs` | 10秒(Warning) / 30秒(Critical) | 設計提案（2段階閾値） |
| `kStuckChronicResidencyUs` | 30秒 | 既存severe条件と統合 |
| `kFallbackMaxRetries` | 3回 | 設計提案 |
| `kFallbackSoftLimit` | 1000件 | 設計提案（超過→Critical昇格要求） |
| `kFallbackHardLimit` | 2000件 | 設計提案（超過→強制ドロップ） |
| `kGracefulDrainMaxMs` | 5000ms | `ReleaseResources.cpp:186` |
| `kDumpIntervalUs` | 60秒 | `WorldLifecycleAudit.h` |
| `kDeferredQueueSize` | 4096 | `DeferredDeletionQueue.h:221` — MPMC lock-free queue |
| `CONVOPEQ_EMERGENCY_DRAIN` | **未定義** | `CMakeLists.txt`/`CMakePresets.json`/全ヘッダ で未定義。コードパスは常に無効。 |
| `kDeferredRetireFallbackHardLimit` | 2000件 | 設計提案 — `DeferredRetireFallbackQueue.h` 拡張時 |
| `kDeferredRetireFallbackSoftLimit` | 1000件 | 設計提案 — 超過時 PolicyEngine に Critical 昇格要求 |
| `kGracefulDrainMaxMs` | 5000ms | `AudioEngine.Processing.ReleaseResources.cpp:156` — 実コード確認済 |
| `kMaxReaders` | 64 | `EpochDomain.h:20` — 実コード確認済 |
| `kReaderSlotCriticalThreshold` | 0.75 (75%) | `RuntimeHealthMonitor.h:54` — 実コード確認済 |
| `checkRetireStall()` | ✅ 既存 | `RuntimeHealthMonitor.h:107` |
| `checkPublicationStall()` | ✅ 既存 | `RuntimeHealthMonitor.h:108` |
| `checkCrossfadeTimeout()` | ✅ 既存 | `RuntimeHealthMonitor.h:112` |
| `checkCrossfadeEventDrop()` | ✅ 既存 | `RuntimeHealthMonitor.h:113` |
| `checkReaderSlotUsage()` | ✅ 既存 | `RuntimeHealthMonitor.h:114` |
| `checkOverflowRate()` | ✅ 既存 | `RuntimeHealthMonitor.h:115` |
| `checkRetireReclaimLatency()` | ✅ 既存 | `RuntimeHealthMonitor.h:116` |
| `HealthEvent` event codes | ✅ 既存 | `RuntimeHealthMonitor.h:35-42` — EVENT_RETIRE_STALL(1001)/EVENT_RETIRE_STALL_WARNING(1002)/EVENT_PUBLICATION_STALL(2001)/EVENT_PUBLICATION_WARNING(2002)/EVENT_READER_SLOT_USAGE(3010)/EVENT_CROSSFADE_TIMEOUT(4001)/EVENT_CROSSFADE_EVENT_DROP(4002) |
| `emitShutdownTrace()` JSON | ❌ healthState不在 | `ISRShutdown.cpp:130-175` — JSON出力に healthState なし。`collectDrainAudit()` は `m_healthMonitor.getHealthState()` 収集済み(`AudioEngine.Threading.cpp:85`)。 |

### 4.5 設計への反映サマリ

| 設計項目 | 変更内容 |
| --- | --- |
| **Phase 1.2** | SnapshotCoordinator: 却下→**部分実施**に再々評価。4/5経路のみ実施。resetFadeStateAndRetireTarget(L67)は `updateFade()` 経由でRTから呼ばれるため除外。switchImmediate追加(5経路目)。 |
| **Phase 1.4** | EQProcessor: 呼び出し元を59→17に訂正。影響範囲縮小。 |
| **Phase 3** | ShutdownResult: emitShutdownTrace() JSON に healthState 追加のみ。戻り値 void 変更不可。 |
| **Phase 4.2** | WorldLifecycleAudit: onFallbackOverflow(), injectEvent() 追加。 |
| **Phase 5** | BlockingReason: 多値化不要。P2→P3 格下げ。 |
| **Phase 8.2** | EmergencyDrain: advancePhase() の ifdef 不整合修正。 |
| **Phase 0 PolicyEngine** | PolicyContext 削除。MonitorState 直接入力。ExecutionClass 導入せず isAudioThread() を流用。 |
| **Phase 9.1** | Learner Health Policy (新規P0): RetireStall 10秒→PauseLearner。RebuildDispatch.cpp:546 の learner継続設計を抑制。 |
| **Phase 9.2** | RuntimeConfigurationDivergence (新規P0): published/requested revision gap>=3 30秒→Critical。 |
| **Phase 9.3** | Deferred Rebuild Expiration (新規P1): PolicySuppressed状態+3回Suppress→Critical。 |
| **Phase 9.4** | Learner Backpressure Monitor (新規P1): bufferedSamples>90% 30秒→PauseLearner。 |
| **Phase 9.5** | Retire Stall Auto Recovery (新規P0): 異常検出→回復試行→失敗時停止の3段階。 |
| **Phase 9.6** | DSP Configuration Freeze (新規P1): requested>active 60秒→Critical。 |
| **Phase 9.7** | Snapshot Starvation (新規P0): maxDeferredAgeMs>10s Warning, >30s Error。既存oldestPendingAge_流用。 |
| **Phase 9.8** | Pending IR Deployment (新規P0): suppressedIRCount>3 Error。lastCommittedConvolverStructuralHash_ と pendingIRGeneration 監視。 |
| **Phase 9.9** | Learner Stall (新規P1): bufferedSamples>90%+30秒改善なし→SuspendLearning。9.4(LearnerBackpressure)より強力。 |
| **Phase 9.10** | Configuration Drift (新規P2): manualOversamplingFactor≠DSPCore.oversamplingFactor 30s→Warning。 |
| **Phase 9.11** | oldestPendingRetireAge (新規P1): checkRetireReclaimLatency拡張。500ms Warning, 5000ms Error。 |
| **Phase 9.12** | RecoveryMode (新規P0): 回復段階の明示的管理。None→Throttled→RecoveryAttempt→ForcedRecovery。 |
| **Phase 9.16** | RollbackToLastHealthyWorld (新規P0): 最終健全Worldへ復帰。ロールバック基盤(ISRRetireRuntimeEx)既存。 |
| **Phase 9.17** | LearnerPublishBlocked (新規P0): Learner進行+published不変10秒→PauseLearner。実ログの直接原因検出。 |
| **Phase 9.18** | RecoveryMode実動作化 (新規P1): ForcedRecovery時はThrottle解除+Admission全面再開。 |
| **Phase 9.19** | RetireStall Root Cause Capture (新規P0): StuckReaderInfo既存。retire_blocker.json証拠出力。 |
| **Phase 9.22** | AudioQualityDegradation (新規P0): ConfigDivergence+DeployBlocked 30秒→直接Rollback。Critical待たない。 |
| **Phase 9.23** | RestoreLastStableLearnerState (新規P0): getState/setState(NoiseShaperLearner.h:110-111)既存。係数復元。 |
| **Phase 9.24** | SuppressionLoopDetector (新規P1): 同一Intent 5回Suppress→ForcePublicationRecovery。 |
| **Phase 9.25** | PendingRetire詳細Evidence (新規P1): DeletionEntry.generation/publicationSequenceId既存流用。 |
| **Phase 9.26** | ForceMinimalWorldPublish SafeMode (新規P0): bootstrapWorldパターン流用。Convolver無効+OS固定+EQのみ。 |
| **Phase 9.29** | SuppressionDuration→RecoveryAction接続 (新規P0): 30s→ForceRetireDrain→60s→ClearDeferredPublish→120s→ForceCrossfadeReset→180s→RejectNewPublication。 |
| **Phase 9.30** | ForceSnapshotPublish (新規P0): consumeDeferredRequest+trySubmit+Admission一時解除。ConfigurationDivergenceからの能動的回復。 |
| **Phase 9.31** | PendingRetireIdentity (新規P1): ISRRetireRouterにpeekOldestPendingRetire追加。pointer/type/gen追跡。 |
| **Phase 9.32** | ActiveReaderBlockerEvidence (新規P0): ReaderSlot.ownerTag/ownerThreadId既存(EpochDomain.h:337-338)。reader_blocker.json。 |
| **Phase 9.33** | LearnerFifoSaturation (新規P0): segmentBuffer.getNumAvailableSamples()既存。FIFO>90% 30s→PauseLearner。 |
| **Phase 9.34** | EnterSafeMode (新規P0): SafeMode具体化(OS=1x/ConvBypass/NS Fixed4Tap/EQ Flat)。独立Action化。 |
| **Phase 9.35** | SuppressionLoop振動検出 (新規P1): Suppress→Recovery 5回/10分→EnterSafeMode。 |
| **Phase 9.36** | RecoveryActionLevel階層化 (新規P0): 6段階(Observe/Throttle/Recover/Restore/Safe/Critical)に整理。 |
| **Phase 9.40** | RuntimeProgressFreeze (新規P0): 3軸(publish/retire/rebuild)進行監視。実ログ障害の直接検出。 |
| **Phase 9.41** | RebuildSuppressionTTL (新規P0): kMaxSuppressionUs=120s。抑制長時間化防止。 |
| **Phase 9.42** | RetireBlockerSnapshot (新規P0): StuckReaderInfo拡張(ownerTag/ownerThreadId)。HealthEvent添付。 |
| **Phase 9.43** | ForceSnapshotPublish強化版 (新規P1): suppressionDuration>60s→事前検査後強制publish。 |
| **Phase 9.44** | LearnerRollback (新規P1): lastKnownGoodNoiseShaper_定期保存。SafeModeより先に異常係数回復。 |
| **Phase 9.45** | RecoveryOutcome (新規P0): 回復成功/失敗判定。閉ループ制御の核心。evaluateRecoveryResult()追加。 |
| **Phase 9.46** | RetireBlockerEvidence強化版 (新規P0): ReaderSlotにpublicationSequence/worldGeneration追加。 |
| **Phase 9.47** | ForceSnapshotPublish安全条件強化 (新規P0): canForcePublish()追加。pendingRetire上限50+oldestPendingAge上限10s。 |
| **Phase 9.48** | SafeMode Learner完全停止 (新規P0): adaptiveCaptureActiveRt=false→stopLearning→GracePeriod→FIFO.clear。 |
| **Phase 9.49** | AudioOutputAnomaly (新規P1): DC Offset/Peak Clipping/RMS Jump/Noise Floor監視。 |
| **Phase 9.50** | Suppression自動解除 (新規P1): checkSuppressionAutoRelease()追加。3条件ORでretirePressureAdmissionStrict_解除。 |
| **Phase 9.51** | PublicationProgressBlocked (新規P0): pendingRetire過多時ForceSnapshotPublish禁止。isPublicationBlocked()判定。 |
| **Phase 9.52** | EpochAdvanceBlocked (新規P0): Reader停滞のLevel別4段階対応。detectStuckReaders()流用。 |
| **Phase 9.53** | LearnerOutputDivergence (新規P0): bestCoefficients L2距離監視。getState()+lastKnownGood比較。 |
| **Phase 9.54** | SafeModeRecovery (新規P0): SafeMode→Normal段階復帰。5分安定条件+bootstrap逆操作。 |
| **Phase 9.55** | Suppression解除条件強化 (新規P1): publishedSequence増加以AND条件追加。4条件ALL充足で解除。 |
| **Phase 9.56** | RuntimeRecoveryScore (新規P0): 4軸(publish/retire/rebuild/audio)総合スコア0-100。 |
| **Phase 9.34強化** | SafeMode二段階化 (新規P0): Soft=ConvByPass+LearnerStop+EQ維持→Hard=OS=1x+FlatEQ+MinimalWorld。 |
| **Phase 9.44強化** | LearnerCheckpoint多世代化+互換性検証 (新規P0): savedStates[6]流用4世代リングバッファ。世代シフト。sampleRate/processingRate/oversamplingFactor/contextGeneration不一致時Rollback禁止。 |
| **Phase 9.47強化** | retireRate>=publishRateガード (新規P0): WorldLifecycleAudit流用。 |
| **Phase 9.49→P0** | AudioOutputAnomaly P0昇格 (新規P0): RecoveryOutcome音声品質判定必須のため。 |
| **Phase 9.50強化** | Suppressionヒステリシス (新規P0): 設定>100/解除<20+30s持続。 |
| **Phase 9.57** | RuntimeConfigurationDeadlock (新規P0): publishedSequence停滞+suppressedRebuild増加の直接検出。suppressedRebuildCount追跡+RecoveryOutcome判定軸。 |
| **Phase 9.58** | RetireRootCauseUnknown (新規P0): Retire Stall原因特定不可→SafeModeのみ。canIdentifyRetireRootCause()。 |
| **Phase 9.59** | RetireProgressFrozen (新規P0): 同上強化版 |
| **Phase 9.60** | PublicationPriority (新規P0): 縮退運転。RebuildTelemetryClass流用。Structuralは常時許可。実ログSuppress防止に最適。 |
| **Phase 9.61** | RetireRootCauseEvidence (新規P0): 原因6分類。detectStuckReaders流用。 |
| **Phase 9.62** | AudioOutputDivergence (新規P0): fingerprintDistance独立PolicySource。 |
| **Phase 9.63** | LearnerRollback自動発動 (新規P0): 3条件成立時auto rollback。 |
| **Phase 9.64** | RetireProgress→SafeMode即時 (新規P0): 回復試行スキップ。 |
| **Phase 9.65** | ClearSuppression独立Action (新規P0): retirePressureAdmissionStrict_一時解除(5秒)+自動再設定。回復順序明確化。PendingRetireObjectInfo(DeletionEntry拡張)。AudioQualityFingerprint正式化(HF Energy/THD proxy)。 |
| **Phase 9.54** | SafeModeRecovery (P2→後回し): OperatorAck必須。自動復帰禁止。 |
| **Phase 9.56** | RuntimeRecoveryScore (P2→後回し): 診断用に格下げ。 |

#### 9.40 【P0】RuntimeProgressFreeze — Runtime 進行停滞の統合検出

**背景**: 実ログ「Retire Stall → 全 Rebuild 停止 → Learner のみ継続 → 音質固定」の根本原因は Runtime 全体の進行停止。個別監視（RetireStall/PendingRetire/Overflow）では検出漏れする状態。`pendingIntentCount` は 0 でも `publicationSequenceCounter_` が進まない場合があり、既存の HealthMonitor では捕捉不可。

**既存インフラ**:

- `RuntimePublicationOrchestrator::updateProgressObservation()` (`RuntimePublicationOrchestrator.h:98-104`) — `m_lastObservedSequence` + `m_lastProgressTimestampUs` 更新済み。Timer から毎 tick 呼ばれている。
- `RuntimePublicationOrchestrator::isPublicationStalled()` (`RuntimePublicationOrchestrator.h:108-112`) — 出版停滞検出済み。閾値は `kPublicationStallThresholdUs`。
- `WorldLifecycleAudit::lastRetireTimestampUs_` (`WorldLifecycleAudit.h:100`) — 最終 retire 時刻追跡済み。
- `AudioEngine::oldestPendingAge_` (`AudioEngine.h:1596`) — 最長滞留 publish 時間追跡済み。
- `RetireRuntime::pendingIntentCount()` (`ISRRetire.h:42`) — 未処理 retire 数追跡済み。

**新規 PolicySource**:

```cpp
RuntimeProgressFreeze    // 9.40 — 全進行停止の統合検出
```

**監視項目**（3軸）:

| 軸 | 監視対象 | 既存フィールド | 閾値 |
| --- | --- | --- | --- |
| Publish 進行 | `publicationSequenceCounter_` 変化有無 | `m_lastObservedSequence` | 60秒以上変化なし → Warning → Error |
| Retire 進行 | `lastRetireTimestampUs_` 更新有無 | `WorldLifecycleAudit::lastRetireTimestampUs_` | 60秒以上更新なし → Error |
| Rebuild 進行 | `pendingIntentCount` + `hasDeferredRequest` + `getMaxDeferredAgeMs` | `RetireRuntime::pendingIntentCount()`, `Orchestrator::getMaxDeferredAgeMs()` | deferAge > 60s + pending不変 → Error |

**発火条件**: 3軸中2軸以上で 60 秒以上「何も進んでいない」状態が継続。

**RecoveryAction**: `ForceSnapshotPublish`（9.43 参照）で滞留 publish を強制実行→改善なければ `RollbackToLastHealthyWorld`（9.16）。

**コード変更**:

```cpp
// RuntimeHealthMonitor.h に追加
struct ProgressFreezeSnapshot {
    uint64_t lastPublishSequence;     // m_lastObservedSequence のスナップショット
    uint64_t lastRetireTimestampUs;   // WorldLifecycleAudit から
    uint64_t maxDeferredAgeMs;        // Orchestrator から
    uint64_t pendingIntentCount;      // RetireRuntime から
    bool hasDeferredRequest;          // Orchestrator から
};

// ★ 新規 check 関数
void RuntimeHealthMonitor::checkRuntimeProgressFreeze() noexcept {
    ProgressFreezeSnapshot snap = collectProgressSnapshot();
    // 3軸チェック
    bool publishStalled = isPublishSequenceStalled(snap.lastPublishSequence);
    bool retireStalled  = isRetireTimestampStalled(snap.lastRetireTimestampUs);
    bool rebuildStalled = isRebuildDeferredStalled(snap.maxDeferredAgeMs);
    // 2/3 軸停滞で発火
    int stalledAxes = (publishStalled ? 1 : 0)
                    + (retireStalled ? 1 : 0)
                    + (rebuildStalled ? 1 : 0);
    if (stalledAxes >= 2) {
        // MonitorState::Error → PolicyEngine → executeRecoveryAction(ForceSnapshotPublish)
    }
}
```

**根拠**: 実ログでは `pub=4, ret=0` が数十分継続。retire 停滞かつ publish 停滞。queue overflow/reader stuck/world leak のいずれにも該当せず、既存監視では検出不可。3軸中2軸停滞ルールでこれを捕捉可能。

#### 9.41 【P0】RebuildSuppressionTTL — Rebuild 抑制の Maximum Duration

**背景**: `retirePressureAdmissionStrict_` が長期間 true のまま維持される可能性がある。実ログでは Retire Stall → Throttle → 全 Rebuild 抑制が数十分継続。抑制の「最大継続時間」がないのが直接原因。

**現状**: `retirePressureAdmissionStrict_` は `true` に設定されるが、明示的な解除条件が弱い。`drainDeferredRetireQueues()` 内の `evaluateRetirePressureLevelNoRt()` でのみ解除されるが、Retire Stall 中は drain が進まず解除されない。

**新規 PolicySource**:

```cpp
RebuildSuppressionTTL    // 9.41 — 抑制最大時間超過
```

**設計**:

```cpp
constexpr uint64_t kMaxSuppressionUs = 120'000'000; // 120秒
// AudioEngine に追加
std::atomic<uint64_t> suppressionStartUs_ { 0 };     // 抑制開始時刻
std::atomic<uint64_t> suppressionEndUs_ { 0 };       // 抑制終了時刻（履歴用）
```

**発火条件**: `retirePressureAdmissionStrict_ == true` かつ継続時間 > 120秒。

**RecoveryAction 連鎖**:

| 経過時間 | RecoveryAction |
| --- | --- |
| 120秒 | `ForceSnapshotPublish` — 抑制を強制解除して publish 再試行 |
| 180秒 | `RollbackToLastHealthyWorld` — 抑制解除失敗時はロールバック |
| 300秒 | `EnterSafeMode` — 最終手段 |

**コード変更**: `AudioEngine.Retire.cpp` の `retirePressureAdmissionStrict_` 設定箇所(4箇所)で抑制開始時刻を記録。`checkRetireReclaimLatency()` または新規 `checkSuppressionDuration()` で TTL 超過検出。

**根拠**: 正常系で 2 分以上全 Rebuild 禁止は実質故障。実ログでは数十分継続しており、120秒 TTL で確実に捕捉可能。

#### 9.42 【P0】RetireBlockerSnapshot — Retire 停滞原因の詳細証拠

**背景**: 現在の `StuckReaderInfo`（`IEpochProvider.h:25-34`）は `readerIndex/readerEpoch/enterCount/isStuck/residencyTimeUs` を持つが、**誰が** epoch を保持しているか（`ownerTag`/`ownerThreadId`）を欠いている。`ActiveReaderBlockerEvidence`（9.32）の ReaderSlot 情報をイベントに添付することで、Retire Stall 発生時の根本原因特定が格段的に向上する。

**既存インフラ**:

- `StuckReaderInfo`（`IEpochProvider.h:25-34`）— 現在の停滞 Reader 情報。`readerIndex/residencyTimeUs` 含む。
- `ReaderSlot`（`EpochDomain.h:327-338`）— `ownerTag[32]` + `ownerThreadId` 含む。
- `ActiveReaderBlockerEvidence`（9.32）— ReaderSlot 詳細を JSON 出力。ただし HealthEvent への添付なし。

**新規構造体**:

```cpp
struct RetireBlockerSnapshot {
    int readerIndex{-1};
    uint64_t readerEpoch{0};
    uint64_t residencyUs{0};
    uint64_t epochGap{0};
    char ownerTag[32]{};           // "ConvolverBuilder", "TimerThread" 等
    uint64_t ownerThreadId{0};      // std::thread::id ハッシュ
    uint32_t pendingRetireCount{0};
    uint64_t minReaderEpoch{0};
};
```

**変更**: `StuckReaderInfo` から `RetireBlockerSnapshot` への拡張。`detectStuckReaders()` の戻り値に `ownerTag`/`ownerThreadId` を追加。`HealthEvent` に `RetireBlockerSnapshot` を optional 添付。

**効果**: 診断ログが「ReaderSlot #17 ownerTag=ConvolverBuilder residency=38s epochGap=150」と具体化→デバッグ時間を 1/10 に短縮。

#### 9.43 【P1】ForceSnapshotPublish 強化版 — SuppressionDuration トリガー

**背景**: v6.1 の 9.30 `ForceSnapshotPublish` は `consumeDeferredRequest + trySubmit` による能動的回復だが、トリガー条件が弱い。Suppression 継続時間による自動発火が必要。

**既存インフラ**:

- `RuntimePublicationOrchestrator::consumeDeferredRequest()` (`RuntimePublicationOrchestrator.h:61-68`) — 保留中の PublishRequest 取得可能。
- `RuntimePublicationOrchestrator::trySubmit()` (`RuntimePublicationOrchestrator.h:45`) — PublishRequest 実行可能。
- `RuntimePublicationOrchestrator::clearDeferredForShutdown()` — 強制消去可能。

**強化内容**:

```cpp
// 9.43 追加: SuppressionDuration による自動発火
constexpr uint64_t kForcePublishThresholdUs = 60'000'000; // 60秒

if (suppressionDuration > kForcePublishThresholdUs) {
    // 事前検査（必須）
    if (worldConsistencyOk && !hasCrossfade && !pendingRetireOverflow) {
        RecoveryAction::ForceSnapshotPublish;
    } else {
        // 事前検査失敗 → Rollback に委譲
        RecoveryAction::RollbackToLastHealthyWorld;
    }
}
```

**RecoveryAction チェーン**:

```text
Suppress継続 60秒 → ForceSnapshotPublish
                 ↘ 失敗 → RollbackToLastHealthyWorld → 失敗 → EnterSafeMode
                 ↘ 成功 → 通常運用復帰
```

**コード変更**: `RuntimePolicyEngine` の `evaluateSuppressionDuration()` 内で上記条件判定を追加。`ForceSnapshotPublish` の事前検査ロジックを `RuntimePublicationOrchestrator` に分離。

#### 9.44 【P1】LearnerRollback — NoiseShaper 学習係数のロールバック

**背景**: 実ログでは `iter=1150` まで学習が進行した後に音質異常が固定化。現在の `PauseLearner` は学習を止めるだけで、異常係数を正常状態に戻さない。`RestoreLastStableLearnerState`（9.23）は getState/setState を利用するが、「いつ」安定状態だったかの判定基準が弱い。

**既存インフラ**:

- `NoiseShaperLearner::getState()`/`setState()` (`NoiseShaperLearner.h:110-111`) — 学習状態の保存/復元可能。
- `NoiseShaperLearner::stopLearning()` (`NoiseShaperLearner.h:105`) — 学習停止。
- No `lastKnownGoodNoiseShaper` メカニズム — **新規必要**。

**新規 PolicySource**:

```cpp
LearnerRollback   // 9.44 — 学習係数ロールバック
```

**設計**:

```cpp
// AudioEngine に追加
struct LearnerStateSnapshot {
    NoiseShaperLearner::State state;
    uint64_t timestampUs;           // スナップショット取得時刻
    uint64_t publicationSequence;    // 同時期の publication sequence
    bool isValid{false};
};

// 定期保存: 正常 publish 成功時に LearnerStateSnapshot を上書き
// (NoiseShaperLearner の getState + publicationSequenceCounter の現在値を保存)
LearnerStateSnapshot lastKnownGoodNoiseShaper_;

// 発火条件: LearnerRollback 発動時
// 1. stopLearning()
// 2. setState(lastKnownGoodNoiseShaper_.state) // 最終安定状態に復元
// 3. updateAudioEngineFromLearnerState()       // DSP 側にも反映
```

**定期保存トリガー**: `notifyHealthyPublication()` に統合。正常 publish 成功時に `NoiseShaperLearner::getState()` を保存。

```cpp
void AudioEngine::notifyHealthyPublication() noexcept {
    // ★ 9.44: Learner 正常状態を定期保存
    if (noiseShaperLearner_ && noiseShaperLearner_->isRunning()) {
        NoiseShaperLearner::State current;
        noiseShaperLearner_->getState(current);
        lastKnownGoodNoiseShaper_.state = current;
        lastKnownGoodNoiseShaper_.timestampUs = getCurrentTimeUs();
        lastKnownGoodNoiseShaper_.publicationSequence =
            convo::consumeAtomic(publicationSequenceCounter_, std::memory_order_acquire);
        lastKnownGoodNoiseShaper_.isValid = true;
    }
    // ... 既存処理
}
```

**根拠**: SafeMode（OS=1x/ConvBypass/EQ Flat）は最終手段。SafeMode より前に「最後の正常係数へ復帰」を試すことで、音質への影響を最小限に抑えられる。実ログでは `iter=1150` までの異常学習を元に戻せる可能性が高い。

#### 9.45 【P0】RecoveryOutcome — 回復成功判定（閉ループ制御）

**背景**: 現在の設計は「異常検出→RecoveryAction発動→終わり」であり、回復が本当に効いたかの判定がない。実ログ障害では ForceRetireDrain→失敗→ForceSnapshotPublish→失敗→SafeMode の段階的失敗があるが、現在の設計では次のActionへの遷移条件が不明瞭。

**既存インフラ**:

- `getLastCommittedPublicationSequence()` (`AudioEngine.h:1138`) — 最終成功 publish の sequence 追跡済み。
- `lastCommittedRuntimeGeneration_` (`AudioEngine.h:1582`) — 最終成功 generation 追跡済み。
- `pendingRetireCount()` (`ISRRetireRouter.h:86`) — 未処理 retire 数。
- `oldestPendingAge_` (`AudioEngine.h:1596`) — 最長滞留時間。
- Retire 進行は `WorldLifecycleAudit.lastRetireTimestampUs_` で追跡可能。

**新規 enum**:

```cpp
enum class RecoveryOutcome : uint8_t {
    Success,      // 回復成功（状態改善確認済み）
    NoEffect,     // 効果なし（状態不変）
    Failed,       // 状態悪化
    Unsafe        // 安全条件違反で発動中止
};

RecoveryOutcome evaluateRecoveryResult(PolicySource source, RecoveryAction action) noexcept;
```

**評価基準**:

| RecoveryAction | Success条件 | NoEffect条件 | Failed条件 |
| --- | --- | --- | --- |
| ForceRetireDrain | pendingRetire減少 | pendingRetire不変 | pendingRetire増加 |
| ForceSnapshotPublish | publicationSequence増加 | publicationSequence不変 | publicationSequence不変+deferredAge増加 |
| PauseLearner | adaptiveCaptureActiveRt==false | isRunning変わらず | — |
| LearnerRollback | setState後isRunning継続 | 音質変化なし | getState/setState例外 |
| RollbackToLastHealthyWorld | publishedRevision回復 | 変更反映されず | pendingRetire増加 |
| EnterSafeMode | 全DSP切替完了 | — | crossfadeタイムアウト |

**閉ループ制御フロー**:

```text
RecoveryAction発動
↓
wait 500ms (3 Timer tick)
↓
evaluateRecoveryResult(source, action)
├→ Success: 回復成功。通常監視に復帰
├→ NoEffect: 次の上位Actionにエスカレーション
├→ Failed: 即座に上位Action + HealthState→Critical
└→ Unsafe: 発動中止。事前条件違反をHealthEventに記録
```

**コード変更**: `RuntimePolicyEngine` に `evaluateRecoveryResult()` を追加。RecoveryAction 実行後に Timer tick から呼び出し、結果に応じて次のActionを選択。

#### 9.46 【P0】RetireBlockerEvidence 強化版 — publicationSequence/worldGeneration 追加

**背景**: 現在の `RetireBlockerSnapshot`（9.42）は `readerIndex/residencyUs/epochGap/ownerTag/ownerThreadId` を持つが、**どの Publish で Reader が取り残されたか**（`publicationSequence`/`worldGeneration`）を欠いている。実ログの `ret=0, reclaim=1, routerPendingRetire=2` から、特定の World Generation で Reader が停滞した可能性が高い。

**既存インフラ**:

- `ReaderSlot`（`EpochDomain.h:327-338`）— `epoch`(atomic) は reader の現在 epoch を持つ。`ownerTag`/`ownerThreadId` 既存。
- `publicationSequenceCounter_`（`AudioEngine.h:1581`）— 発行済み sequence 追跡。
- `lastCommittedPublicationSequence_`（`AudioEngine.h:1583`）— 最終成功 sequence。
- `lastCommittedRuntimeGeneration_`（`AudioEngine.h:1582`）— 最終成功 generation。

**拡張内容**:

```cpp
// ★ v6.3: ReaderSlot に publicationSequence/worldGeneration を追加
struct ReaderSlot {
    // ... 既存フィールド ...
    std::atomic<uint64_t> publicationSequence { 0 };  // ★ 新規: Reader が最後に観測した publicationSequence
    std::atomic<uint64_t> worldGeneration { 0 };       // ★ 新規: Reader が最後に観測した worldGeneration
};
```

**効果**: `ReaderSlot #17 publicationSequence=42 worldGeneration=5` という情報から、「sequence 42 の World 発行後、Reader が取り残された」と特定可能。デバッグ時間をさらに短縮。

**コード変更**: `ReaderSlot` に2フィールド追加。`enter()`/`exit()` で最新値を書き込み。`detectStuckReaders()` の戻り値に含める。

#### 9.47 【P0】ForceSnapshotPublish 安全条件強化

**背景**: 実障害では「Retire できない」状態で ForceSnapshotPublish を実行すると `PendingRetire++` となり、病状を悪化させる可能性がある。事前条件として `pendingRetireCount` と `oldestPendingRetireAge` の確認が必須。

**既存インフラ**:

- `pendingRetireCount()`（`ISRRetireRouter.h:86`）— 現在の滞留 retire 数。
- `oldestPendingAge_`（`AudioEngine.h:1596`）— 最長滞留時間（double 秒）。
- `RuntimeDrainAudit.routerPendingRetire` — 監査構造体に既存。

**安全条件**:

```cpp
// ForceSnapshotPublish 発動条件（9.43 の事前検査を強化）
bool canForcePublish() const noexcept {
    // 安全条件1: pendingRetire が許容範囲内
    if (m_retireRouter->pendingRetireCount() > kForcePublishMaxPendingRetire)
        return false;
    // 安全条件2: oldestPendingAge が許容範囲内
    if (convo::consumeAtomic(oldestPendingAge_, std::memory_order_acquire) > kForcePublishMaxAgeSec)
        return false;
    // 安全条件3: activeReaders == 0（Epoch保持Readerなし）
    if (m_epochDomain.activeReaderCount() > 0)
        return false;
    // 安全条件4: worldConsistency OK
    if (!worldLifecycleAudit_.isConsistent())
        return false;
    // 安全条件5: 進行中 crossfade なし
    if (crossfadeRuntime_.isPending())
        return false;
    return true;
}

constexpr uint32_t kForcePublishMaxPendingRetire = 50;   // pendingRetire 上限
constexpr double   kForcePublishMaxAgeSec = 10.0;         // 最長滞留 10秒以内
```

**回復チェーン**:

```text
canForcePublish() == true  → ForceSnapshotPublish
canForcePublish() == false → RollbackToLastHealthyWorld（pendingRetire 過多時は retire より）
canForcePublish() == false → EnterSafeMode（滞留10秒超→最終手段）
```

#### 9.48 【P0】SafeMode 移行時 Learner 完全停止

**背景**: SafeMode（OS=1x/ConvBypass/NS Fixed4Tap/EQ Flat）移行時、NoiseShaper Learner が動き続けると SafeMode の効果が半減する。`adaptiveCaptureActiveRt=false` + `learner.stop()` + `FIFO.clear()` まで行うべき。

**既存インフラ**:

- `adaptiveCaptureActiveRt`（`AudioEngine.h:1681`）— `publishAtomic(false, release)` で停止可能。`AudioEngine.Learning.cpp:60/77/140/197/239` で使用実績あり。
- `NoiseShaperLearner::stopLearning()`（`NoiseShaperLearner.h:105`）— 学習停止。
- `NoiseShaperLearner::segmentBuffer`（`NoiseShaperLearner.h:253`）— `AudioSegmentBuffer` 型。`pushBlock`/`clear` メソッドあり。

**SafeMode 移行手順**（厳密な順序）:

```cpp
void AudioEngine::enterSafeMode() noexcept {
    // Step 1: Audio Thread への通知（最も優先）
    convo::publishAtomic(adaptiveCaptureActiveRt, false, std::memory_order_release);
    // Step 2: Learner 停止
    if (noiseShaperLearner_ && noiseShaperLearner_->isRunning()) {
        noiseShaperLearner_->stopLearning();
    }
    // Step 3: Grace Period（Audio Thread が Step1 を認識する時間）
    constexpr uint64_t kLearnerGracePeriodUs = 100; // 100us
    // Step 4: FIFO クリア（Audio Thread が書き込みを止めた後に実行）
    noiseShaperLearner_->segmentBuffer.clear();
    // Step 5: SafeMode World 生成
    createSafeModeWorld();
    // Step 6: DSP 切替
    // ...
}
```

**根拠**: SafeMode にもかかわらず Learner が動き続けると「SafeMode なのに Learner だけ動き続ける→SafeMode から復帰後再び異常学習係数に戻る」ループに陥る。FIFO クリアは Audio Thread の書き込み停止確認後に行う必要がある（GracePeriod 必須）。

#### 9.49 【P1】AudioOutputAnomaly — 音響異常検出

**背景**: 現在の監視は Runtime Health（retire/publish/reclaim 健全性）のみ。音質異常（DC Offset/Peak Clipping/RMS Jump/Noise Floor Explosion）は監視外。実ログの「音が急に変になった」は **Runtime Health は正常でも Audio Quality は異常**の状態。

**既存インフラ**:

- `UltraHighRateDCBlocker`（`UltraHighRateDCBlocker.h`）— DC 除去は処理パス内に存在。ただし「DC Offset が異常に大きい」の検出はなし。
- 出力信号へのアクセスは Audio Thread の `getNextAudioBlock()` 内で可能。

**新規 PolicySource**:

```cpp
AudioOutputAnomaly    // 9.49 — 音響異常検出
```

**監視項目**（Audio Thread 外の分析スレッドで実行）:

| 項目 | 検出条件 | 閾値 |
| --- | --- | --- |
| DC Offset | 出力信号の平均値が継続的に非ゼロ | > 0.001 が 1秒以上 |
| Peak Clipping | 出力サンプルが ±0.99 を超過 | 100サンプル/秒以上 |
| RMS Jump | 出力 RMS が 10dB 以上急変 | 5ms 窓での RMS 差分 |
| Noise Floor Explosion | 無音区間のノイズフロア上昇 | 無音時 RMS > -60dB |

**RecoveryAction 連鎖**:

```text
DC Offset 1秒以上 → LearnerRollback（9.44）
                 ↘ 効果なし → RollbackToLastHealthyWorld（9.16）
                 ↘ さらに継続 → EnterSafeMode（9.48）
```

**設計判断**: AudioOutputAnomaly は **P1**（Phase 9.40-9.48 完了後）。理由: Runtime Progress Freeze の解決が優先。音響異常検出は Audio Thread 外の分析スレッドを新設する必要があり、実装コストが高い。

#### 9.50 【P1】Suppression 自動解除条件

**背景**: `retirePressureAdmissionStrict_ = true` の解除条件が弱い。現在は `drainDeferredRetireQueues()` 内の `evaluateRetirePressureLevelNoRt()` でのみ解除されるが、Retire Stall 中は drain が進まず解除されない。TTL 超過（9.41）は Recovery を発動するが、Retire Stall が自然解消した場合の「抑制解除」がない。

**既存インフラ**:

- `retirePressureAdmissionStrict_`（`AudioEngine.h:3456`）— 抑制フラグ。4箇所から `publishAtomic(true)`。
- `pendingRetireCount()`（`ISRRetireRouter.h:86`）— 抑制解除判定に使用可能。
- `RouterPendingRetire` 監視（`RuntimeHealthMonitor.cpp`）— 既存監視。

**解除条件**（3条件の OR）:

```cpp
// checkSuppressionAutoRelease(): Timer tick で毎回評価
void AudioEngine::checkSuppressionAutoRelease() noexcept {
    if (!convo::consumeAtomic(retirePressureAdmissionStrict_, std::memory_order_acquire))
        return; // 既に解除済み

    bool shouldRelease = false;
    // 条件1: pendingRetire が正常水準に戻った
    if (m_retireRouter->pendingRetireCount() < kSuppressionReleaseThreshold)
        shouldRelease = true;
    // 条件2: 最新の retire が成功した（epoch が進んだ）
    if (/* lastRetireEpoch が前回観測値より進んだ */)
        shouldRelease = true;
    // 条件3: TTL 超過からの Recovery が成功した（9.41 の結果）
    if (/* RecoveryOutcome::Success 確認 */)
        shouldRelease = true;

    if (shouldRelease) {
        convo::publishAtomic(retirePressureAdmissionStrict_, false, std::memory_order_release);
        diagLog("[DIAG] Suppression auto-released: pendingRetire="
            + juce::String(static_cast<int>(m_retireRouter->pendingRetireCount())));
    }
}
```

**コード変更**: `AudioEngine.Retire.cpp` または `AudioEngine.Timer.cpp` に `checkSuppressionAutoRelease()` 追加。Timer tick から毎 tick 呼び出し。

---

### 4.6 深掘り調査で確定した設計定数

```cpp
constexpr uint32_t kQueueSize = 4096;               // DeferredDeletionQueue
constexpr int kMaxReaders = 64;                      // EpochDomain
constexpr uint64_t kCrossfadeTimeoutUs = 30'000'000; // 30秒
constexpr uint64_t kRetireAgeWarningUs = 5'000'000;  // 5秒
constexpr uint64_t kRetireAgeCriticalUs = 30'000'000;// 30秒
constexpr uint32_t kOverflowRateCriticalThreshold=5; // 5回/秒
constexpr uint32_t kOverflowRateWarningThreshold=1;  // 1回/秒
constexpr double kReaderSlotCriticalThreshold=0.75;  // 75%
constexpr uint64_t kStuckEvidenceIntervalUs=10'000'000; // 10秒
constexpr uint64_t kStuckEpochThreshold=10;          // 10 epoch
constexpr uint64_t kStuckResidencyThresholdUs=1'000'000; // 1秒(AND条件)
constexpr uint64_t kStuckChronicResidencyUs=30'000'000;  // 30秒(only条件)
// ★ v6.2:
constexpr uint64_t kMaxSuppressionUs = 120'000'000;  // 120秒 — RebuildSuppressionTTL
constexpr uint64_t kForcePublishThresholdUs = 60'000'000; // 60秒 — ForceSnapshotPublish
constexpr uint64_t kProgressFreezeThresholdUs = 60'000'000; // 60秒 — RuntimeProgressFreeze
// ★ v6.4:
constexpr uint64_t kForcePublishMaxPendingRetire = 50;    // 9.47/9.51 — ForcePublish安全上限
constexpr double   kForcePublishMaxAgeSec = 10.0;         // 9.47 — 最長滞留上限
constexpr uint64_t kLearnerGracePeriodUs = 100;           // 9.48 — Learner停止GracePeriod
constexpr double   kLearnerCoeffDivergenceThreshold = 10.0; // 9.53 — L2距離閾値(dB)
constexpr uint64_t kSafeModeStabilizationUs = 300'000'000; // 9.54 — SafeMode安定5分
constexpr int      kRuntimeRecoveryScoreHealthy = 80;     // 9.56 — 健全スコア閾値
// ★ v6.5:
constexpr int      kSuppressionSetThreshold = 100;         // 9.50強化 — 抑制設定閾値
constexpr int      kSuppressionReleaseThreshold = 20;      // 9.50強化 — 抑制解除閾値
constexpr uint64_t kSuppressionReleaseDebounceUs = 30'000'000; // 9.50強化 — 30秒持続確認
constexpr int      kLearnerCheckpointGenerations = 4;      // 9.44強化 — Learner世代数
constexpr uint64_t kCheckpointIntervalUs = 300'000'000;    // 9.44強化 — 5分間隔
// ★ v6.9:
constexpr double   kSpectralCentroidThreshold = 0.3;       // 9.49強化 — スペクトル重心変化閾値
constexpr double   kSpectralFlatnessThreshold = 0.4;       // 9.49強化 — スペクトル平坦度変化閾値
constexpr double   kCrestFactorThreshold = 6.0;            // 9.49強化 — クレストファクタ変化閾値(dB)
```

#### 9.51 【P0】PublicationProgressBlocked — ForceSnapshotPublish 禁止ガード

**背景**: 9.47 で `pendingRetireCount` 上限を設定したが、事前条件を PolicySource として独立監視し、ForceSnapshotPublish 発動自体を禁止する必要がある。pendingRetire 過多時の強制 publish は PendingRetire++ による病状悪化を招く。

**既存インフラ**: `pendingRetireCount()` (`ISRRetireRouter.h:86`), `oldestPendingAge_` (`AudioEngine.h:1596`), `worldLifecycleAudit_.isConsistent()`, `crossfadeRuntime_.isPending()` — すべて既存。

**新規 PolicySource**:

```cpp
PublicationProgressBlocked  // 9.51 — pendingRetire過多でpublish不可
```

**発火条件**:

```cpp
// canForcePublish() の判定結果を PolicySource として独立監視
bool isPublicationBlocked() const noexcept {
    return m_retireRouter->pendingRetireCount() > kForcePublishMaxPendingRetire
        || convo::consumeAtomic(oldestPendingAge_, std::memory_order_acquire) > kForcePublishMaxAgeSec;
}
```

**効果**: ForceSnapshotPublish 発動前に `isPublicationBlocked()` をチェック。blocked なら即座に `RecoveryAction::RollbackToLastHealthyWorld` または `RecoveryAction::EnterSafeMode` へ迂回。ForceRetireDrain での retire 解消を優先。

#### 9.52 【P0】EpochAdvanceBlocked — Reader 停滞の Level 別段階対応

**背景**: 実ログでは `routerPendingRetire=2, oldestAgeMs=557` — 特定 Reader が epoch を保持し Epoch が進まない状態。現在の `detectStuckReaders()` は Stuck 検出のみで、Reader 停滞の重症度に応じた段階的対応がない。

**既存インフラ**:

- `detectStuckReaders()` (`EpochDomain.h:265-302`) — `readerIndex/readerEpoch/residencyTimeUs/enterCount` 収集済み。
- `ReaderSlot.ownerTag/ownerThreadId` (`EpochDomain.h:337-338`) — 保有者情報既存。
- `enterReader()` (`EpochDomain.h:101-127`) — Reader 進入時に epoch/residencyStartTimestampUs 記録。
- `getMinReaderEpoch()` — 全 Reader の最小 epoch 取得可能。

**Level 別対応**:

| Level | 条件 | 対応 |
| --- | --- | --- |
| 1 (Info) | epoch差 > 10 | `dump` — ReaderSlot 情報を evidence JSON に出力 |
| 2 (Warning) | epoch差 > 10 + residency > 10秒 | `snapshot` — HealthEvent に RetireBlockerSnapshot 添付 |
| 3 (Error) | epoch差 > 10 + residency > 30秒 | `reader blocker report` — `reader_blocker_{timestamp}.json` 出力 + 診断ログ |
| 4 (Critical) | epoch差 > 10 + residency > 60秒 | `SafeMode` — 最終手段 |

```cpp
// RuntimeHealthMonitor に追加
void checkEpochAdvanceBlocked() noexcept {
    auto info = m_epochDomain.detectStuckReaders(kStuckEpochThreshold);
    if (!info.isStuck) return;

    const uint64_t epochGap = info.currentEpoch - info.readerEpoch;
    const uint64_t residencyUs = info.residencyTimeUs;

    if (epochGap > 10 && residencyUs > 60'000'000) {
        // Level 4: Critical → SafeMode
        updateMonitorState(MonitorState::Error, EVENT_EPOCH_ADVANCE_BLOCKED_CRITICAL);
    } else if (epochGap > 10 && residencyUs > 30'000'000) {
        // Level 3: Error → reader blocker report
        emitReaderBlockerReport(info);
        updateMonitorState(MonitorState::Error, EVENT_EPOCH_ADVANCE_BLOCKED);
    } else if (epochGap > 10 && residencyUs > 10'000'000) {
        // Level 2: Warning → HealthEvent 添付
        attachBlockerSnapshot(info);
        updateMonitorState(MonitorState::Warning, EVENT_EPOCH_ADVANCE_BLOCKED_WARNING);
    } else {
        // Level 1: Info → dump
        dumpReaderSlot(info);
    }
}
```

#### 9.53 【P0】LearnerOutputDivergence — NoiseShaper 係数乖離監視

**背景**: 実ログでは `iter=1150` まで学習が進行し、異常係数で音質が固定化。現在の監視は Learner の進行有無（FIFO 飽和/進行停滞）のみで、「係数そのものが異常になった」を検出できない。LearnerRollback(9.44) 発動の判断材料として、係数の L2 距離測定が必要。

**既存インフラ**:

- `bestCoefficients` (`NoiseShaperLearner.h:275`) — `std::array<std::atomic<double>, kOrder>` 原子アクセス可能。
- `LearnedState.bestCoefficients` (`NoiseShaperLearner.h:62`) — `std::array<double, kOrder>`。
- `getState()`/`setState()` (`NoiseShaperLearner.h:110-111`) — 状態保存/復元。
- `kOrder = LatticeNoiseShaper::kOrder` — 係数次元数。
- `lastKnownGoodNoiseShaper_` (9.44) — 最終正常状態保存。

#### 9.57 【P0】RuntimeConfigurationDeadlock — Configuration 進行停止の直接検出

**背景**: 実ログの本質は `publishedSequence` 停滞 + `suppressedRebuild` 増加 + `pendingDeployment` 滞留。`RuntimeProgressFreeze(9.40)` は3軸監視だが、「Configuration の進行停止」に特化した独立 PolicySource が必要。実ログでは publish=4, ret=0 が数十分継続しており、これは ConfigurationDeadlock と診断可能。

**既存インフラ**:

- `getLastCommittedPublicationSequence()` (`AudioEngine.h:1138`) — 最終成功 publish sequence。
- `RebuildTelemetryDecision::Suppressed` (`AudioEngine.h:1994`) — Suppress 判定既存。
- `pendingIntentCount()` (`ISRRetire.h:42`) — 未処理 publish intent 数。
- `hasDeferredRequest()` (`RuntimePublicationOrchestrator.h:57`) — 保留中 publish 有無。

**検出条件**:

```cpp
// RuntimeHealthMonitor に追加
void checkConfigurationDeadlock() noexcept {
    // 3条件の AND
    bool sequenceStalled = !isPublishSequenceAdvancing();  // publishedSequence 不変 > 30秒
    bool suppressIncreasing = isSuppressCountIncreasing();  // suppress 判定が増加中
    bool deploymentPending = hasDeferredRequest() || pendingIntentCount() > 0;

    if (sequenceStalled && suppressIncreasing && deploymentPending) {
        // ConfigurationDeadlock 検出 → 30秒持続確認後発火
        updateMonitorState(MonitorState::Error, EVENT_CONFIGURATION_DEADLOCK);
    }
}
```

**RecoveryAction チェーン**:

```text
ConfigurationDeadlock
↓
ClearSuppression（retirePressureAdmissionStrict_ 一時解除）
↓
ForceSinglePublish（1件のみ publish 試行）
↓
evaluateRecoveryResult() → Success → 通常復帰
                         → NoEffect → LearnerRollback + ForceSinglePublish
                         → Failed → SoftSafeMode → HardSafeMode
```

**コード変更**: `RuntimeHealthMonitor` に `checkConfigurationDeadlock()` 追加。`suppressCount` 追跡用カウンタ `suppressCount60s_` を `AudioEngine` に追加。

**効果**: RuntimeProgressFreeze より一段踏み込んだ「Configuration 進行停止の直接検出」が可能。実ログ障害の `publishedSequence=4, ret=0, routerPendingRetire=2` を直接捕捉可能。

**v6.7 拡張**: `suppressedRebuildCount` 追跡用カウンタ `suppressCount60s_` を `AudioEngine` に追加。`RecoveryOutcome` の判定軸に `suppressedRebuildCount` を追加（publish進行中でも suppress 増加なら実質回復失敗）。

#### 9.58 【P0】RetireRootCauseUnknown — Retire Stall 原因特定不可時の安全策

**背景**: Retire Stall 発生時、`RetireBlockerSnapshot`(9.42/9.46) で Reader 情報を収集しても原因が特定できない場合がある。その場合、`ForceSnapshotPublish` や `ForceRetireDrain` は本質的に危険。原因不明時は SafeMode のみ許可する。

**既存インフラ**: `activeReaderCount()` (`EpochDomain.h:201`), `detectStuckReaders()` (`EpochDomain.h:265`), `ReaderSlot.ownerTag/ownerThreadId` (`EpochDomain.h:337-338`)

**判定**:

```cpp
// RootCause 特定可能性チェック
bool canIdentifyRetireRootCause() const noexcept {
    auto info = m_epochDomain.detectStuckReaders(kStuckEpochThreshold);
    if (!info.isStuck) return true;  // Stuckなし → RootCause不要
    // ReaderSlot 情報が取得可能か
    if (info.readerIndex >= 0 && info.readerIndex < kMaxReaders) {
        const auto& slot = m_epochDomain.readers[info.readerIndex];
        // ownerTag が空でなく、epoch に整合性がある
        if (slot.ownerTag[0] != '\0' && info.readerEpoch <= info.currentEpoch)
            return true;
    }
    return false;  // RootCause特定不可
}
```

**Policy**: `RetireRootCauseUnknown` — 原因特定不可時のみ発火。

```cpp
if (retireStallDetected && !canIdentifyRetireRootCause()) {
    // 原因不明 → ForceSnapshotPublish/Rollback 禁止
    // SafeMode のみ許可
    updateHealthState(ISRHealthState::Critical);
    executeRecoveryAction(RecoveryAction::Safe);
}
```

**監視**:

```cpp
// RuntimeHealthMonitor に追加
void checkLearnerOutputDivergence() noexcept {
    if (!lastKnownGoodNoiseShaper_.isValid)
        return;

    NoiseShaperLearner::State current;
    noiseShaperLearner_->getState(current);          // getState で現在係数取得

    double l2Distance = 0.0;
    for (size_t i = 0; i < NoiseShaperLearner::kOrder; ++i) {
        double diff = current.bestCoefficients[i]
                    - lastKnownGoodNoiseShaper_.state.bestCoefficients[i];
        l2Distance += diff * diff;
    }
    l2Distance = std::sqrt(l2Distance);

    if (l2Distance > kLearnerCoeffDivergenceThreshold) {
        // 係数乖離検出 → LearnerRollback 発動
        executeRecoveryAction(RecoveryAction::LearnerRollback);
    }
}

// 閾値（設計提案）
constexpr double kLearnerCoeffDivergenceThreshold = 10.0; // dB相当のL2距離
```

**RecoveryAction**: `LearnerRollback`（9.44）へ直結。SafeMode より前に異常係数を正常状態に戻す。

#### 9.59 【P0】RetireProgressFrozen — retireCount 進行停止の直接監視

**背景**: 実ログでは `pub=4, ret=0` が数十分継続。`RuntimeProgressFreeze(9.40)` は3軸監視だが、retire進行の直接指標として `retireCount` 増加率を独立監視すべき。`publishedCount > 0` かつ `retireCountIncrease == 0` が30秒継続 → RetireProgressFrozen。

**既存インフラ**: `WorldLifecycleAudit.publishedCount_/retiredCount_` (WorldLifecycleAudit.h:91-92) — 既存atomic。60秒窓で増加率計算可能。

```cpp
// RuntimeHealthMonitor に追加
void checkRetireProgressFrozen() noexcept {
    const uint64_t retiredNow = worldLifecycleAudit_.retiredCount();
    const uint64_t publishedNow = worldLifecycleAudit_.publishedCount();

    if (publishedNow > 0 && retiredNow == lastObservedRetiredCount_ && elapsed > 30s) {
        // published は増えているのに retire が全く進まない
        updateMonitorState(MonitorState::Error, EVENT_RETIRE_PROGRESS_FROZEN);
    }
    lastObservedRetiredCount_ = retiredNow;
}
```

**RecoveryAction**: `EpochAdvanceBlocked(9.52)` と連動。Reader停滞が原因ならそちらで対処。特定不可なら `RetireRootCauseUnknown(9.58)` に委譲。

**RecoveryOutcome 使用の厳密定義**:

v6.3 で導入した `RecoveryOutcome` を以下の4値に拡張し、各Actionの判定基準を明文化:

| RecoveryOutcome | 定義 | 例(ForceRetireDrain) | 例(LearnerRollback) |
| --- | --- | --- | --- |
| Success | 状態改善確認(publishSeq増加/pendingRetire減少) | pendingRetire 100→10 | L2距離が閾値未満 |
| PartialSuccess | 部分改善(主要指標回復+副次指標未回復) | pendingRetire 100→50 | L2距離改善したが最適値未到達 |
| NoEffect | 状態不変(主要指標変化なし) | pendingRetire 100→100 | L2距離不変 |
| Worsened | 状態悪化(pendingRetire増加/publishSeq停滞継続) | pendingRetire 100→120 | L2距離増加 |

**Learner強制停止規則** (v6.8 明示):

```cpp
// ★ v6.9: RetireProgressFrozen(9.59) + RuntimeProgressFreeze(9.40) + ConfigurationDeadlock(9.57) の3条件 OR
//   理由: 世界が更新されないのに Learner だけ更新するのが最も危険。
if (isRuntimeProgressFrozen() || isConfigurationDeadlocked() || isRetireProgressFrozen()) {
    if (noiseShaperLearner_->isRunning()) {
        noiseShaperLearner_->stopLearning();
        convo::publishAtomic(adaptiveCaptureActiveRt, false, std::memory_order_release);
        diagLog("[DIAG] Learner auto-stopped: progress freeze detected");
    }
}
```

**PolicyEngine責務境界** (v6.8 確定):

| 責務 | 担当 | PolicyEngine関与 |
| --- | --- | --- |
| MonitorState→RecoveryAction選択 | **PolicyEngine** | 主責 |
| SafeMode判定・管理 | **SafeModeManager**(分離推奨) | RecoveryAction::Safe 発行のみ |
| Learner制御 | **AudioEngine**直制御 | PolicySource::LearnerAnomaly 検出のみ |
| Admission制御 | **AdmissionManager**(既存) | RecoveryAction::Throttle 発行のみ |
| Audio品質監視 | **AudioOutputMonitor(9.49)** | 補助診断。メイン回復判断には使用しない |

**SoftSafeMode自動復帰 / HardSafeMode OperatorAck**:

| SafeMode種別 | 復帰条件 | 自動復帰 |
| --- | --- | --- |
| Soft (ConvByPass+LearnerStop+EQ維持) | 5分安定+retire=0+reader=0 | ✅ 自動許可 |
| Hard (OS=1x+FlatEQ+MinimalWorld) | 同上 + OperatorAck | ❌ OperatorAck必須 |

**AudioOutputAnomaly 位置づけ明確化** (v6.8):

- **補助監視**として位置づけ。メインの回復判断には使用しない。
- 音響異常は「結果」であり、根本原因(RuntimeProgressFreeze/ConfigurationDeadlock)より遅い。
- 診断ログへの出力および SafeMode 復帰条件の補助確認にのみ使用。

**AudioQualityFingerprint** (v6.9 追加, 9.49強化):

`AudioOutputAnomaly(9.49)` に以下を追加し、DC/Peak/RMS/NoiseFloor では検出困難な音色変化を捕捉:

```cpp
struct AudioQualityFingerprint {
    double spectralCentroid;    // スペクトル重心
    double spectralFlatness;    // スペクトル平坦度
    double crestFactor;         // クレストファクタ(Peak/RMS比)
};

// 急激変化検出: 前回 Fingerprint との差が閾値超過
bool detectSpectralAnomaly(const AudioQualityFingerprint& prev,
                           const AudioQualityFingerprint& curr) noexcept {
    const double centroidDelta = std::abs(curr.spectralCentroid - prev.spectralCentroid);
    const double flatnessDelta = std::abs(curr.spectralFlatness - prev.spectralFlatness);
    const double crestDelta     = std::abs(curr.crestFactor - prev.crestFactor);
    return centroidDelta > kSpectralCentroidThreshold
        || flatnessDelta > kSpectralFlatnessThreshold
        || crestDelta > kCrestFactorThreshold;
}
```

**RecoveryOutcome に audioRecovered 追加** (v6.9, 9.45強化):

```cpp
struct RecoveryOutcome {
    enum Value : uint8_t {
        Success,         // 構成回復 + 音声品質回復
        PartialSuccess,  // 構成回復 + 音声未回復(audioRecovered==false)
        NoEffect,        // 構成未回復 + 音声未回復
        Worsened         // 構成悪化
    };
    Value value;
    bool audioRecovered;     // ★ v6.9: AudioQualityFingerprint 正常化確認
    uint64_t durationUs;     // 回復所要時間
};
```

判定基準に `audioRecovered` を追加することで、publishSequenceが回復しても音声が戻っていない場合に `PartialSuccess` と判定可能。

**PendingRetireEvidence 強化** (v6.9, 9.31/9.25統合):

`DeletionEntry`（`DeferredDeletionQueue.h:25-32`）は既に以下のフィールドを持つ:

- `ptr` — 詰まりオブジェクトのポインタ
- `epoch` — retire epoch
- `type` — DeletionEntryType（現在はGenericのみ、6種に拡張予定）
- `publicationSequenceId` — 出版-退役の因果追跡用(既存)
- `generation` — generation追跡(既存)

これらを `ISRRetireRouter::peekOldestPendingRetire()` の戻り値として公開:

```cpp
struct PendingRetireEvidence {
    void* ptr;
    uint64_t enqueueTimestampUs;  // キュー投入時刻
    uint64_t publicationSequenceId;
    uint64_t generation;
    DeletionEntryType objectType;
    uint64_t currentAgeUs;        // 現在の滞留時間
};
```

効果: 「どのオブジェクトが何秒詰まっているか」を特定可能。実運用でのデバッグ時間を大幅短縮。

**RuntimeProgressFreeze + AudioOutputAnomaly + RecoveryFailed → SoftSafeMode** (v6.9, 9.40強化):

```cpp
// 3条件同時成立時は SoftSafeMode へ即移行（通常の段階的リカバリをスキップ）
if (isRuntimeProgressFrozen() && isAudioAnomalyDetected() && isRecoveryFailed()) {
    // 音が壊れたままの状態が継続 → SoftSafeMode 即時発動
    enterSoftSafeMode();
    diagLog("[DIAG] Emergency SoftSafeMode: progress freeze + audio anomaly + recovery failed");
}
```

#### 9.60 【P0】PublicationPriority — 縮退運転時の出版優先度制御

**背景**: 実ログでは `convolver_params_changed`（Structural rebuild）が Retire Stall 起因の Suppression で全て止まり、Convolver 更新が不能になった。Practical Stable Runtime では「異常→全停止」ではなく「異常→縮退運転」が基本。`RebuildTelemetryClass`（Structural/Snapshot/FinalizeAware）を活用して優先度別に出版を通す。

**既存インフラ**:

- `RebuildTelemetryClass` (`AudioEngine.h:1976`) — `Structural`/`Snapshot`/`FinalizeAware` 既存。
- `shouldRejectRebuildAdmissionForPressure()` (`AudioEngine.h:1781`) — 背圧ゲート関数既存。
- `RebuildTelemetryReason::ConvolverParamsChanged` (`AudioEngine.h:2019`) — Structural 再構築理由。

**設計**:

```cpp
enum class PublicationPriority : uint8_t {
    Critical,     // 絶対停止不可: EmergencyDrain/Shutdown
    Structural,   // 構造変更: Convolver IR/係数/OS変更 → RetireStall時も許可
    Snapshot,     // スナップショット: UI操作反映 → 抑制可能
    Cosmetic      // 描画更新: メーター/ラベル → 抑制優先
};

// shouldRejectRebuildAdmissionForPressure() に優先度判定追加
bool AudioEngine::shouldRejectRebuildAdmissionForPressure(
    PublicationPriority priority) const noexcept {
    if (priority >= PublicationPriority::Structural)
        return false;  // Critical/Structural は常に許可
    // 既存の背圧判定
    return convo::consumeAtomic(retirePressureAdmissionStrict_, std::memory_order_acquire);
}
```

**効果**: Retire Stall 発生時でも `convolver_params_changed`（Structural）は出版可能。実ログの「Suppress 14→30→永遠に止まる」を防止。全ての rebuild を一律抑制しない。

**コード変更**: `RebuildDispatch.cpp` の `shouldRejectRebuildAdmissionForPressure()` 呼び出し箇所(3箇所: 241/428/479)で呼び出しシグネチャ変更。呼び出し元から `RebuildTelemetryClass` を伝播。

#### 9.61 【P0】RetireRootCauseEvidence — Retire Stall 原因自動分類

**背景**: 現在の設計は「Retire Stall 発生」を検出するが「なぜ発生したか」を分類できない。実ログの `ret=0, reclaim=1, routerPendingRetire=2` から、6段階の原因分類を導入。

```cpp
struct RetireRootCauseEvidence {
    enum Cause : uint8_t {
        ReaderNeverExited,          // Reader が epoch を保持したまま exit しない
        EpochNeverAdvanced,         // Epoch 自体が進まない
        RetireQueueOverflow,        // DeferredDeletionQueue 溢れ
        RouterPendingRetire,        // ISRRetireRouter 滞留
        CrossfadeReferenceHeld,     // Crossfade が参照を保持
        Unknown                     // 分類不能
    };
    Cause cause;
    uint64_t evidenceValue;         // 根拠となる値(pendingRetireCount等)
    uint64_t detectionTimestampUs;
};
```

**監視**: `detectStuckReaders()` + `pendingRetireCount()` + `routerPendingRetire` + `crossfadeRuntime_.isPending()` の組み合わせで自動分類。

#### 9.62 【P0】AudioOutputDivergence PolicySource

fingerprintDistance を独立 PolicySource として監視:

```cpp
PolicySource::AudioOutputDivergence  // 9.62 — 音響 Fingerprint 乖離
```

発火条件: `AudioQualityFingerprint` の前回値との距離が閾値超過。LearnerRollback(9.63)のトリガーとして使用。

#### 9.63 【P0】LearnerRollback 自動発動条件

RuntimeProgressFreeze + AudioOutputAnomaly + LearnerOutputDivergence の3条件同時成立時:

```cpp
if (isRuntimeProgressFrozen() && isAudioAnomalyDetected() && isLearnerDiverged()) {
    // 3条件成立 → 安全な Checkpoint へ自動ロールバック
    auto& cp = getBestCompatibleCheckpoint();
    if (cp.isValid) {
        noiseShaperLearner_->setState(cp.state);
        noiseShaperLearner_->startLearning(true);  // resume
    }
}
```

#### 9.64 【P0】RetireProgressFrozen + LearnerActive → SoftSafeMode 即時移行

```cpp
if (isRetireProgressFrozen() && noiseShaperLearner_->isRunning()
    && getSuppressedRebuildCount() > kSuppressCountEmergencyThreshold) {
    // 回復試行をスキップして即 SoftSafeMode
    enterSoftSafeMode();
}
```

#### 9.65 【P0】ClearSuppression — 独立 RecoveryAction として明確化

**背景**: 現在の設計では ClearSuppression は `ForceSnapshotPublish` の内部処理として実装されている。しかし実ログでは「Suppress 解除」と「Force Publish」は別の概念。回復順序として「ClearSuppression→ForceSnapshotPublish」を独立 Action として明確化すべき。

```cpp
enum class RecoveryAction : uint8_t {
    Observe,
    Throttle,
    // ★ 9.65: ClearSuppression を独立 Action として明示
    ClearSuppression,    // retirePressureAdmissionStrict_ 一時解除(5秒)
    Recover,             // ForceRetireDrain / ForceSinglePublish
    Restore,
    Safe,
    Critical
};
```

**ClearSuppression 動作**:

1. `retirePressureAdmissionStrict_ = false`（5秒間だけ一時解除）
2. 5秒後に自動再設定（`resetSuppressionAfterTimeout(5000ms)`）
3. 解除中に成功した publish/retire があれば通常復帰

**回復順序の明示**:

```text
1. RetireDrain         → pendingRetire 減少
2. ClearSuppression    → 5秒間抑制解除
3. ForceSinglePublish  → 1件のみ publish
4. Rollback            → すべて失敗時
5. SafeMode            → 最終手段
```

**PendingRetireObjectInfo 拡張** (v7.1, 9.25/9.31強化):

`DeletionEntry`（`DeferredDeletionQueue.h:26-33`）の既存フィールドに加え、以下の情報を `peekOldestPendingRetire()` で公開:

```cpp
struct PendingRetireObjectInfo {
    void* ptr;                    // 既存
    uint64_t epoch;               // 既存
    DeletionEntryType type;       // 既存（Generic→6種に拡張予定）
    uint64_t publicationSequenceId; // 既存
    uint64_t generation;           // 既存
    uint64_t enqueueTimestampUs;  // ★ 新規: キュー投入時刻
    uint64_t ageMs;               // 計算値: 現在時刻 - enqueueTimestampUs
    uint64_t ownerWorldId;        // ★ 新規: 所有World ID
    uint64_t crossfadeId;         // ★ 新規: 関連Crossfade ID
};
```

**AudioQualityFingerprint 正式 PolicySource 化** (v7.1, 9.62強化):

`AudioOutputDivergence(9.62)` に以下を追加:

- `HF Energy`（高域エネルギー変化率）
- `THD proxy`（歪み推定: 高調波成分比率の簡易近似）

```cpp
// AudioQualityFingerprint 正式版
struct AudioQualityFingerprint {
    double spectralCentroid;
    double spectralFlatness;
    double crestFactor;
    double hfEnergy;              // ★ 高域エネルギー(8kHz以上)
    double thdProxy;              // ★ THD簡易近似(高調波歪み率)
    double noiseFloorDb;
    double previousDistance;      // 前回値との距離
};
```

これにより、NoiseShaper 学習暴走による高域ノイズ増加・歪み増加を検出可能。

#### 9.54 【P0】SafeModeRecovery — SafeMode からの Normal Runtime 段階復帰

**背景**: SafeMode（OS=1x/ConvBypass/NS Fixed4Tap/EQ Flat）移行後、そのまま継続すると運用上厳しい。SafeMode で5分間正常動作を確認後、段階的に Normal Runtime へ復帰する経路が必要。

**復帰条件**（すべて満たすこと）:

```cpp
constexpr uint64_t kSafeModeStabilizationUs = 300'000'000; // 5分

bool canRecoverFromSafeMode() const noexcept {
    if (!isInSafeMode()) return false;
    // 条件1: SafeMode 移行後 5分以上経過
    if (getCurrentTimeUs() - safeModeEnteredUs_ < kSafeModeStabilizationUs)
        return false;
    // 条件2: pendingRetire == 0
    if (m_retireRouter->pendingRetireCount() > 0)
        return false;
    // 条件3: activeReader == 0（全 Reader が epoch を解放）
    if (m_epochDomain.activeReaderCount() > 0)
        return false;
    // 条件4: crossfade == 0
    if (crossfadeRuntime_.isPending())
        return false;
    return true;
}
```

**段階復帰手順**:

```cpp
void recoverFromSafeMode() noexcept {
    // Step 1: bootstrap World から Normal World へ移行
    //    bootstrapWorld パターン（RuntimeBuilder::createBootstrapWorld）の逆操作。
    //    最小構成→OS可変→Conv有効→Learner再開 の順。
    restoreNormalOversampling();      // OS 可変に戻す
    restoreConvolverOperation();       // Convolver 再有効化
    restoreNoiseShaperMode();          // NS Fixed4Tap→元のモード
    restoreEQMode();                   // EQ Flat→元の設定

    // Step 2: Learner 再開（前回の正常状態から resume）
    if (lastKnownGoodNoiseShaper_.isValid) {
        noiseShaperLearner_->setState(lastKnownGoodNoiseShaper_.state);
        noiseShaperLearner_->startLearning(true); // resume=true
    }

    // Step 3: 監視再開
    exitSafeMode();
    diagLog("[DIAG] SafeMode recovery: restored to Normal Runtime");
}
```

#### 9.55 【P1】Suppression 解除条件強化 — publishedSequence 増加以条件追加

**背景**: 9.50 の Suppression 解除条件は `pendingRetire` + `retire成功` + `TTL回復成功` の3条件。しかし publish がまだ進んでいない可能性がある。解除条件に `publishedSequence増加` を追加する。

**強化後解除条件**（4条件の OR → AND 条件に変更）:

```cpp
void AudioEngine::checkSuppressionAutoRelease() noexcept {
    if (!convo::consumeAtomic(retirePressureAdmissionStrict_, std::memory_order_acquire))
        return;

    // ★ v6.4: 4条件 ALL 充足で解除（OR→AND に厳格化）
    bool retireRecovered = m_retireRouter->pendingRetireCount() < kSuppressionReleaseThreshold;
    bool publishRecovered = getLastCommittedPublicationSequence() > lastSuppressedPublicationSeq_;
    bool epochAdvanced = /* lastRetireEpoch が前回より進んだ */;
    bool recoverySucceeded = /* RecoveryOutcome == Success */;

    if (retireRecovered && publishRecovered && epochAdvanced) {
        convo::publishAtomic(retirePressureAdmissionStrict_, false, std::memory_order_release);
        diagLog("[DIAG] Suppression auto-released: pendingRetire="
            + juce::String(static_cast<int>(m_retireRouter->pendingRetireCount()))
            + " pubSeq=" + juce::String(static_cast<int64>(getLastCommittedPublicationSequence())));
    }
}
```

#### 9.56 【P2】RuntimeRecoveryScore — Runtime 全体の回復評価（診断用）

**背景**: `RecoveryOutcome`(9.45) は Action 単位の評価。補助的に4軸総合スコアを診断用に保持する。**P2（後回し）**: 実際の回復判断は離散状態(Healthy/Degraded/Recovery/SafeMode/Critical)で十分。

**設計**:

```cpp
struct RuntimeRecoveryScore {
    // 4軸スコア（各 0-25、合計 0-100）
    uint8_t publishProgress;    // publicationSequence の増加率
    uint8_t retireProgress;     // pendingRetire の減少率
    uint8_t rebuildProgress;    // pendingIntent + deferredAge
    uint8_t audioQuality;       // AudioOutputAnomaly(9.49) + LearnerOutputDivergence(9.53)

    // 総合
    uint8_t total() const { return publishProgress + retireProgress + rebuildProgress + audioQuality; }
    bool isHealthy() const { return total() >= kRuntimeRecoveryScoreHealthy; }

    // Breakthrough 条件（これ以上Actionを継続しても無意味）
    bool isStagnant() const { /* 3回連続で total が +5 未満 */ }
};

RuntimeRecoveryScore computeRuntimeRecoveryScore() noexcept;
```

**評価基準**:

| 軸 | 25点条件 | 0点条件 |
| --- | --- | --- |
| publishProgress | publicationSequence が1秒に1以上増加 | 10秒以上変化なし |
| retireProgress | pendingRetire < 10 | pendingRetire > 100 |
| rebuildProgress | pendingIntent==0 and deferredAge==0 | pendingIntent>50 or deferredAge>30s |
| audioQuality | DC=0 and Clip=0 and RMS安定 and Coeff正常 | DC>0.001 or Clip多発 or Coeff乖離 |

**実装**: `RuntimeHealthMonitor::tick()` の最後に `computeRuntimeRecoveryScore()` を呼び出し、スコア履歴をリングバッファに保持。`RecoveryOutcome` の評価と併用することで、Action 単位と Runtime 単位の二重評価が可能。

---

### 4.7 v6.6 設計統合ガイド

#### PolicySource 72→14分類

v6.2-v6.5 で拡張された PolicySource 72個を以下の14分類に統合。詳細な監視項目（9.xx）は各カテゴリの内部ロジックとして保持:

| v6.6分類 | 統合元(v6.5) | 監視対象 |
| --- | --- | --- |
| `RetireStall` | RetireStall/RetireAge/SuppressionTTL/RebuildSuppression/RateInsufficient | Retire系全般 |
| `PublicationStall` | PublicationStall/ProgressFreeze/ConfigurationDeadlock/PendingDeployment | 出版系全般 |
| `ReaderStuck` | ReaderSlotUsage/EpochAdvanceBlocked(L1-L4)/ActiveBlocker | Reader系全般 |
| `CrossfadeTimeout` | CrossfadeTimeout/CrossfadeEventDrop | Crossfade系 |
| `LearnerAnomaly` | LearnerBackpressure/LearnerStall/LearnerDivergence/LearnerPublishBlocked/Rollback | Learner系全般 |
| `WorldConsistency` | WorldConsistency/WorldLeak/ConfigurationDivergence | World整合性 |
| `AudioOutputAnomaly` | DC Offset/Peak Clipping/RMS Jump/Noise Floor | 音響系(P0昇格) |
| `EmergencyCondition` | EmergencyDrain/ShutdownTimeout | 緊急系 |
| `RecoveryOutcome` | Success/NoEffect/Failed/Unsafe | 回復結果 |
| `SafeModeState` | SoftActive/HardActive/RecoveryReady | SafeMode状態 |

#### RecoveryAction 44→6レベル

すべての RecoveryAction を6レベルの階層に統合:

| Level | 名称 | 意味 | 代表アクション |
| --- | --- | --- | --- |
| 0 | Observe | 監視のみ | HealthEvent記録、evidence出力 |
| 1 | Throttle | 抑制 | admissionStrict_設定、PauseLearner、Suppress |
| 2 | Recover | 回復 | ForceRetireDrain、ForceSnapshotPublish、ConfigurationDeadlock解除 |
| 3 | Restore | 復元 | RollbackToLastHealthyWorld、LearnerRollback、CheckpointRestore |
| 4 | Safe | 安全確保 | SoftSafeMode(ConvByPass+LearnerStop)→HardSafeMode(1x+FlatEQ) |
| 5 | Critical | 重大 | RejectNewPublication、EmergencyDrain、Shutdown |

#### 実装優先度（Phase別）|v6.6|確定版

| 優先度 | Phase | 内容 | 判断理由 |
| --- | --- | --- | --- |
| **P0** | Phase 0 | PolicyEngine基盤 | 全回復の前提 |
| **P0** | Phase 1 | enqueueRetire契約修復 | リーク防止の基盤 |
| **P0** | Phase 2 | ReaderStuck判定改善 | 根本原因特定 |
| **P0** | 9.40-9.53 | RuntimeProgress系・Learner系・RecoveryOutcome | 実ログ障害の直接対策 |
| **P0** | 9.57 | RuntimeConfigurationDeadlock | 同上 |
| **P1** | Phase 3-8 | ShutdownResult/背圧統合/WorldConsistency | Phase0/1/2完了後 |
| **P2** | 9.54 | SafeModeRecovery(OperatorAck必須) | 自動復帰は危険。運用判断 |
| **P2** | 9.56 | RuntimeRecoveryScore(診断用) | 離散状態で十分 |

#### SafeModeRecovery: OperatorAck 要件

v6.4 の SafeModeRecovery(9.54) は自動復帰を前提としていたが、v6.6 では **OperatorAck 必須** に変更:

```cpp
// SafeModeRecovery発動条件（v6.6: OperatorAck必須）
bool canRecoverFromSafeMode() const noexcept {
    if (!isOperatorAckReceived())      // ★ 運用者確認
        return false;
    if (!isInSafeMode())
        return false;
    if (getCurrentTimeUs() - safeModeEnteredUs_ < kSafeModeStabilizationUs)
        return false;
    if (m_retireRouter->pendingRetireCount() > 0)
        return false;
    // ... 以下既存条件
    return true;
}
```

---

## 5. コードベース完成度（計画適用後見通し）

| 領域 | 現状 | Phase 0後 | Phase 1-3後 | Phase 4-7後 |
| --- | --- | --- | --- | --- |
| 正常系 | 95% | 95% | 95% | 95% |
| 監視・検出 | 90% | 90% | 95% | 97% |
| 異常系ガバナンス | 78% | 85% | 85% | 92% |
| 自己回復 | 30% | 30% | 60% | 85% |
| 契約遵守 | 10% | 10% | 90% | 95% |

**最終目標**: 異常系ガバナンス 92% / 自己回復 85% / 契約遵守 95%
