# Recovery System 実装計画書 — AI実行可能版（再構成版）

> **日付**: 2026-06-15（第3版）
> **検証済みコードファクト**: 27件（全件コード突合確認済み）
> **棚卸し結果**: 全55項目確定済み（第1版〜第10版: 2026-06-15）
> **実装指示**: 各 Phase は独立して実装可能。依存関係は「実装順序」セクションを参照。

---

## 目次

1. [実装前確認事項](#1-実装前確認事項)
2. [Phase 1: Restore 維持 + Epoch Recovery 実装](#2-phase-1-restore-維持--epoch-recovery-実装)
3. [Phase 2: 背圧統合](#3-phase-2-背圧統合)
4. [Phase 3: 閉ループ制御](#4-phase-3-閉ループ制御)
5. [Phase 4: RecoveryBudget](#5-phase-4-recoverybudget)
6. [Phase 5: Learner FIFO 監視](#6-phase-5-learner-fifo-監視)
7. [Phase 6: Suppression Probe](#7-phase-6-suppression-probe)
8. [Phase 7: Critical 出口](#8-phase-7-critical-出口)
9. [ファイル影響マップ](#9-ファイル影響マップ)
10. [実装順序](#10-実装順序)

---

## 1. 実装前確認事項

### 1.1 コードファクト一覧

全27件(F01-F27)は Serena MCP/CodeGraph MCP/grep で実コード確認済み。

| # | ファクト | 確認ソース |
| --- | --- | --- |
| F01-F18 | 基本ファクト（RecoveryAction enum, executeRecoveryAction, evaluateAggregate 等） | 各ソースファイル |
| F19 | **requestRollback() = setEpochMode(getRollbackMode())**: atomic storeのみ。pipeline非消費。診断専用。 | ISRRetireRuntimeEx.cpp:280-285 |
| F20 | **publishIdleWorldOnly()**: Shutdown guard有り。再入保護なし。 | AudioEngine.Transition.cpp:9-27 |
| F21 | **getLastExecutedAction/VerificationEntry/RecoveryBudget**: 現コードに未存在。新規設計。 | ソース調査 |
| F22 | **CrossfadeTimeout Recovery**: HealthEvent callback発火。publishIdleWorldOnly既実行。 | AudioEngine.Timer.cpp:576-640 |
| F23 | **二重背圧系**: admissionStrict_(全体) と publicationThrottle_(局所) の2系。 | AudioEngine.h:3478 |
| F24 | **publicationSeq**: `PublicationSequenceId`=`uint64_t`, monotonic, no wrap. | ISRRuntimeSemanticSchema.h:195 |
| F25 | **epochAdvanceCount**: `audioCallbackEpochCounter`=atomic<uint64_t>, Audio Threadのみincrement, reset無し。 | AudioEngine.h:1185 |
| F26 | **pendingRetireCount**: ISRRetireRouter委譲値。Deferred含む代表値。 | ISRRetireRouter.cpp:139 |
| F27 | **Learner EMAリセット**: `m_learnerWasActive_` 実装確認済。 | 6.2節参照 |

### 1.2 設計上の重要制約

| # | 制約 | 背景 |
| --- | --- | --- |
| C1 | **requestRollback() は NoPipelineEffect**: epochModeRaw_ は pipeline非消費。Restoreの実効果は tryReclaim/drain/clearDeferred の副次効果のみ。 | F19 |
| C2 | **epochAdvanceCount は相関指標**: Restore効果の因果指標ではない。pipelineがepochModeRaw_を消費する未来実装まで真の効果検証不可。 | F19+F25 |
| C3 | **publishIdleWorldOnly() 再入保護なし**: RestorePhase状態機械で二重発行防止必須。 | F20 |
| C4 | **CrossfadeTimeout回復が優先**: 即時callback, Restore Step2より先に実行。 | F22 |
| C5 | **二重背圧 Authority**: admissionStrict_=Master, publicationThrottle_=Sub（変更禁止）。 | F23 |
| C6 | **RestoreCapability**: 現状 `NoPipelineEffect`。将来 `PartialPipelineEffect`→`FullPipelineEffect` へ拡張可能。 | F19 |

### 1.3 共通参照テーブル

| データ | ソース | 型 |
|--------|--------|----|
| publicationSeq | `getLastCommittedPublicationSequence()` (AudioEngine.h:1146) | `atomic<uint64_t>` |
| pendingRetireCount | `ISRRetireRouter::pendingRetireCount()` | method(uint32_t) |
| maxRetireAgeUs | `m_maxRetireAgeRef` (setter injection) | `atomic<uint64_t>` |
| epochAdvanceCount | `rtLocalState_.audioCallbackEpochCounter` (AudioEngine.h:1185) | `atomic<uint64_t>` |
| readerStuckCount | `ISRRetireRouter::readerStuckCount()` | method |
| activeReaderCount | `ISRRetireRouter::activeReaderCount()` | method |
| healthState | `RuntimeHealthMonitor::m_healthState_` | `atomic<ISRHealthState>` |

### 1.3 設計上の注意（全Phase共通）

| # | 注意事項 | 対象Phase |
| --- | --- | --- |
| N1 | `retirePressureAdmissionStrict_` が Master Authority。 `retirePressurePublicationThrottleActive_` は Sub Authority（変更禁止）。 | 2 |
| N2 | `publishIdleWorldOnly()` は再入保護なし。RestorePhase 状態機械で二重発行を防止。 | 1, 6 |
| N3 | CrossfadeTimeout 回復が publishIdleWorldOnly(HardReset) を先に呼ぶ可能性あり。 | 1, 3, 6 |
| N4 | VerificationEntry は昇格時に上書き。明示的 cleanup は不要（markForVerificationが上書き）。 | 3, 7 |
| N5 | `verifyAfterUs` 初期値は最低50ms推奨（requestRollback効果発現の遅延を考慮）。 | 3 |
| N6 | closed-loop 評価は tick() 内で PolicyEngine 評価より前に実行。 | 3 |
| N7 | Critical からの自動復帰は updateHealthState() が担当。CriticalExitCondition は監視のみ。 | 7 |

### 1.4 VerificationEntry ライフサイクル

```text
[Action 発火] → markForVerification() → [PendingVerification]
    ↓ computeTrend() 評価（verifyAfterUs 経過後）
    ├─ Recovered → resetVerification() → [Idle]
    ├─ Improving → stalledCount==3 → nextAction() → [新規PendVerif]
    │                stalledCount<3 → verifyAfterUs 延長 → [継続]
    ├─ Stalled   → nextAction() → [新規PendVerif]
    └─ Worsening → nextAction() → [新規PendVerif]

[Critical時]: markExecuted(Critical) → hasPendingVerification()=false
[Shutdown時]: resetVerification() 必須
```

### 1.5 事前準備（既存コード・変更しない）

```cpp
enum class RecoveryAction : uint8_t {
    Observe, Throttle, Recover, Restore, Safe, Critical, _Count
};
enum class MonitorState : uint8_t { Normal, Warning, Error };
```

## 2. Phase 1: Restore 維持 + Epoch Recovery 実装（P0-B）

### 目的
`RecoveryAction::Restore` に Epoch Recovery + Learner Rollback + Idle World 発行を追加する。
Restore は削除せず維持（Recovery Ladder: Throttle→Recover→Restore→Safe→Critical）。

**Restore 発動条件（問題G）**: Restore は以下の要因で発生した問題に対してのみ有意味:
- `kFaultEpochAdvance` (Epoch Advance Blocked): EpochMode 切替で epoch advance 回復を期待
- `kFaultReader` (ReaderStuck): Reader 解放の副次効果を期待
- `kFaultConfigDiverg` (ConfigurationDivergence): Learner Rollback で回復を期待

上記以外の要因（RetireStall/PublicationStall/OverflowRate）では Recover で十分であり、
Restore に昇格しても追加効果は期待できない（`requestRollback()` が pipeline 非消費のため）。
その場合、`nextAction()` は通常通り Restore へ昇格する（問題A-2: Ladder維持）。ただし Restore の実効性が低いことは認識した上で昇格させる。

### 2.1 Recovery Ladder（確定）

```text
Throttle (L1) → Recover (L2) → Restore (L3) → Safe (L4) → Critical (L5)
```

### 2.2 executeRecoveryAction() の Restore case 置換

**ファイル**: `src/audioengine/AudioEngine.Timer.cpp:679-683`

**BEFORE**:
```cpp
case convo::RecoveryAction::Restore:
    tryReclaimResources();
    drainDeferredRetireQueues(false);
    break;
```

**AFTER**:
```cpp
case convo::RecoveryAction::Restore:
    // Step1: Epoch Recovery
    if (retireRuntimeEx_.canRollback()) {
        retireRuntimeEx_.setRollbackMode(convo::isr::EpochMode::Split);
        retireRuntimeEx_.requestRollback();
        ++m_restoreGeneration_;
    }
    // 強制回復（問題A-1: Recover との差別化）
    tryReclaimResources();
    drainDeferredRetireQueues(false);
    // Learner Rollback
    if (lastKnownGoodNoiseShaper_.isValid)
        noiseShaperLearner->setState(lastKnownGoodNoiseShaper_.state);
    // DeferredPublicationFlush
    if (runtimeOrchestrator_) {
        runtimeOrchestrator_->clearDeferredForShutdown();
        runtimeOrchestrator_->forceSnapshotPublish();  // ★ ForceSnapshotPublish（問題1: Recoverとの差別化）
    }
    // Step2（publishIdleWorldOnly）は閉ループ制御後
    break;
```

> **注意**: `requestRollback()` の実効性は限定的（1.2節参照）。将来的に EpochDomain/RetireRouter での `epochModeRaw_` 消費実装まで「rollback」は名目上の動作。

### 2.3 RestorePhase 状態機械

**更新タイミング（問題D）**:
- `restoreGeneration++` は `executeRecoveryAction(Restore)` の先頭でのみ実行
- Step2完了やSafe移行では増加しない
- これにより `restoreGeneration` は「Restore開始回数」を一意に識別する

```cpp
enum class RestorePhase : uint8_t {
    None,                    // Restore未実行
    EpochRecoveryIssued,     // Step1完了
    LearnerRollbackDone,     // Learner復元完了
    IdleWorldPublished       // Step2完了
};
```

### 2.4 TrendSnapshot への restorePhase 追加

```cpp
struct TrendSnapshot {
    // ...（既存メンバ。詳細は Phase 3）...
    uint64_t restoreGeneration{0};
    RestorePhase restorePhase{RestorePhase::None};
};
```

### 2.5 AudioEngine フィールド追加

```cpp
std::atomic<uint64_t> m_restoreGeneration_{0};      // Restore効果識別用（問題A-1: uint64_t, wraparoundなし）
std::atomic<RestorePhase> m_restorePhase_{RestorePhase::None};  // atomic（問題A-2: 複数スレッド参照）
```

### 2.6 PolicySource::AudioOutputAnomaly マッピング

変更不要。`AudioOutputAnomaly→Restore` は維持。

**TEST**: `grep "RecoveryAction::Restore" src/**/*.cpp` → 3件。

---

## 3. Phase 2: 背圧統合（P1-A）

### 目的
`injectBackpressureSignal()` 未使用(F04)を修正。`retirePressureAdmissionStrict_` の直接書き込みを PolicyEngine 経由に変更。Critical 水準のみ緊急直接経路を残す。

### 3.1 Authority 定義

| 背圧系 | 制御変数 | ロール | 変更 |
| --- | --- | --- | --- |
| Master | `retirePressureAdmissionStrict_` | 全体遮断 (RecoveryAction直結) | **拡張** |
| Sub | `retirePressurePublicationThrottleActive_` | Publication局所抑制 | **変更禁止** |

### 3.2 injectBackpressureSignal() に最大値保持追加

**ファイル**: `src/audioengine/RuntimeHealthMonitor.h:146-151`

```cpp
void injectBackpressureSignal(std::size_t fallbackSize, double overflowRate) noexcept {
    // atomic CAS max update
    uint64_t current = convo::consumeAtomic(m_maxFallbackSize_, std::memory_order_acquire);
    while (fallbackSize > current) {
        if (convo::compareExchangeAtomic(m_maxFallbackSize_, current,
            static_cast<uint64_t>(fallbackSize), std::memory_order_acq_rel,
            std::memory_order_acquire)) break;
        current = convo::consumeAtomic(m_maxFallbackSize_, std::memory_order_acquire);
    }
    double rateCurrent = convo::consumeAtomic(m_maxOverflowRate_, std::memory_order_acquire);
    while (overflowRate > rateCurrent) {
        if (convo::compareExchangeAtomic(m_maxOverflowRate_, rateCurrent,
            overflowRate, std::memory_order_acq_rel, std::memory_order_acquire)) break;
        rateCurrent = convo::consumeAtomic(m_maxOverflowRate_, std::memory_order_acquire);
    }
    m_backpressureInjected_ = true;
}
```

**フィールド追加**:
```cpp
std::atomic<uint64_t> m_maxFallbackSize_{0};
std::atomic<double> m_maxOverflowRate_{0.0};

// BackpressureWindow（問題F: peak + average + count の統計モデル）
struct BackpressureWindow {
    uint64_t maxSize{0};
    uint64_t sumSize{0};
    uint32_t sampleCount{0};
    double   maxRate{0.0};
    double   sumRate{0.0};

    void record(std::size_t size, double rate) noexcept {
        if (size > maxSize) maxSize = size;
        maxRate = (rate > maxRate) ? rate : maxRate;
        sumSize += size;
        sumRate += rate;
        ++sampleCount;
    }

    [[nodiscard]] double averageSize() const noexcept {
        return sampleCount > 0
            ? static_cast<double>(sumSize) / sampleCount : 0.0;
    }

    [[nodiscard]] double averageRate() const noexcept {
        return sampleCount > 0 ? sumRate / sampleCount : 0.0;
    }

    void reset() noexcept {
        maxSize = 0; sumSize = 0; sampleCount = 0;
        maxRate = 0.0; sumRate = 0.0;
    }
};
BackpressureWindow m_backpressureWindow_;  // tick() の exchange(0) と併用
```

### 3.3 tick() の背圧評価 → exchange(0) でリセット

```cpp
if (m_backpressureInjected_) {
    const auto maxFb = convo::exchangeAtomic(m_maxFallbackSize_, uint64_t{0},
                                              std::memory_order_acq_rel);
    const auto maxOr = convo::exchangeAtomic(m_maxOverflowRate_, 0.0,
                                              std::memory_order_acq_rel);
    if (maxFb > 500) decision.actions |= toBit(RecoveryAction::Throttle);
    if (maxOr > 5.0) decision.actions |= toBit(RecoveryAction::Critical);
    m_backpressureInjected_ = false;
}
```

### 3.4 AudioEngine.Retire.cpp 直接書き込み置換

**変更箇所1** (L155-161): Critical 水準のみ緊急直接、通常は injectBackpressureSignal 経由。
**変更箇所2** (L291): publishAtomic → injectBackpressureSignal に置換。

**TEST**: `grep -n "retirePressureAdmissionStrict_" src/audioengine/AudioEngine.Retire.cpp` → 1件のみ。

---

## 4. Phase 3: 閉ループ制御（P0-A）

### 目的
Open Loop → Closed Loop。Action実行後の効果を傾向判定し、改善なければ次段階へ昇格。

### 4.1 RecoveryOutcome enum 再定義

**ファイル**: `src/audioengine/RuntimePolicyEngine.h:54-60`

```cpp
enum class RecoveryOutcome : uint8_t {
    None,          // 未評価
    Improving,     // 改善傾向 — 維持、昇格禁止
    Recovered,     // 正常復帰 — Observe へ移行
    Stalled,       // 停滞 — 次段階へ昇格
    Worsening      // 悪化 — 即時昇格
};
```

### 4.2 TrendSnapshot 構造体

**ファイル**: `src/audioengine/RuntimePolicyEngine.h`

```cpp
struct TrendSnapshot {
    uint64_t pendingRetire{0};
    uint64_t publicationSeq{0};
    uint64_t maxRetireAgeUs{0};
    uint32_t activeReaderCount{0};
    uint32_t readerStuckCount{0};
    bool     freezeDetected{false};
    uint32_t activeFaultMask{0};
    uint64_t restoreGeneration{0};
    uint64_t epochAdvanceCount{0};       // Epoch advance 累積回数（問題A-1: Restore効果測定用）
    uint64_t lastCompletedEpoch{0};      // 最終完了Epoch ID
    uint64_t publicationGeneration{0};   // Publication世代（問題A-1: Epoch+Publish+Retire三点セット評価用）
    RestorePhase restorePhase{RestorePhase::None};
};

// activeFaultMask ビット定義
static constexpr uint32_t kFaultRetire       = 1u << 0;
static constexpr uint32_t kFaultPublication  = 1u << 1;
static constexpr uint32_t kFaultReader       = 1u << 2;
static constexpr uint32_t kFaultOverflow     = 1u << 3;
static constexpr uint32_t kFaultCrossfade    = 1u << 4;
static constexpr uint32_t kFaultLearner      = 1u << 5;
static constexpr uint32_t kFaultConfigDiverg = 1u << 6;
static constexpr uint32_t kFaultEpochAdvance  = 1u << 7;  // Epoch Advance Blocked（問題A-1/G）

// EpochAdvanceHealth（問題G: EpochAdvanceBlocked定量定義）
//   epochAdvanceCount が kEpochAdvanceStallWindowUs 秒間増加しない場合に Blocked と判定
struct EpochAdvanceHealth {
    uint64_t lastAdvanceUs{0};        // 最後に epochAdvanceCount が増加した時刻
    uint64_t currentEpoch{0};         // 現在の epochAdvanceCount
    uint64_t completedEpoch{0};       // 最終完了Epoch ID

    static constexpr uint64_t kEpochAdvanceStallWindowUs = 1 * 1'000'000;  // 1秒間進まなければStall

    [[nodiscard]] bool isBlocked(uint64_t nowUs) const noexcept {
        return nowUs - lastAdvanceUs > kEpochAdvanceStallWindowUs;
    }
};
```

### 4.3 VerificationEntry 構造体

```cpp
enum class VerificationState : uint8_t { Idle, PendingVerification };

struct VerificationEntry {
    VerificationState state{VerificationState::Idle};
    uint64_t executedAtUs{0};
    uint64_t verifyAfterUs{0};       // 初期値50ms推奨
    uint64_t restoreGeneration{0};   // 発行時の restoreGeneration（問題A-3: 世代追跡用、uint64_tでwraparoundなし）
    TrendSnapshot baselineSnapshot;
    TrendSnapshot lastSnapshot;
    uint8_t stalledCount{0};         // 上限3回

    [[nodiscard]] bool isIdle() const noexcept {
        return state == VerificationState::Idle;
    }
};
```

### 4.4 RuntimePolicyEngine 拡張

```cpp
class RuntimePolicyEngine {
public:
    void markForVerification(RecoveryAction action, const TrendSnapshot& snapshot) noexcept;
    VerificationEntry& getEntry(RecoveryAction action) noexcept;
    const VerificationEntry& getEntry(RecoveryAction action) const noexcept;
    void resetVerification() noexcept;
    [[nodiscard]] bool hasPendingVerification() const noexcept;
    RecoveryAction getLastExecutedAction() const noexcept { return m_lastAction_; }
    RecoveryBudget& getBudget() noexcept { return m_budget_; }
    const RecoveryBudget& getBudget() const noexcept { return m_budget_; }

private:
    RecoveryAction m_lastAction_{RecoveryAction::Observe};
    std::array<VerificationEntry, static_cast<size_t>(RecoveryAction::_Count)> m_verificationEntries_;
    RecoveryBudget m_budget_;
};
```

### 4.5 computeTrend() — 評価順序（確定版）

評価は以下の優先順位で実行：

```cpp
RecoveryOutcome RuntimeHealthMonitor::computeTrend(...) const noexcept
{
    // Step 0: 主要delta計算
    const int64_t retireDelta = static_cast<int64_t>(now.pendingRetire)
                              - static_cast<int64_t>(before.pendingRetire);
    const int64_t ageDelta    = static_cast<int64_t>(now.maxRetireAgeUs)
                              - static_cast<int64_t>(before.maxRetireAgeUs);
    const int64_t pubDelta    = static_cast<int64_t>(now.publicationSeq)
                              - static_cast<int64_t>(before.publicationSeq);
    const bool faultMaskIncreased = (now.activeFaultMask > before.activeFaultMask);

    // Step 1: ProgressFreeze 監視（最優先、多軸評価）（問題3）
    //   5%改善のみでは不十分。reader/epochAdvance/publication も確認する。
    if (now.freezeDetected) {
        const bool retireProgress = before.pendingRetire > 0
            && (now.pendingRetire * 100) < (before.pendingRetire * 95);
        const bool readerProgress = (now.readerStuckCount < before.readerStuckCount);
        const bool epochProgress = (now.epochAdvanceCount > before.epochAdvanceCount);
        const bool pubProgress = (now.publicationSeq > before.publicationSeq);
        const bool multiAxisImprovement = retireProgress
            || (readerProgress && epochProgress)
            || (pubProgress && readerProgress);
        if (!multiAxisImprovement)
            return RecoveryOutcome::Worsening;
    }

    // Step 2: Recovered（全軸正常, 静的安定状態も含む）（問題B/2）
    //   pendingRetire 閾値: 固定256に加え、健全な最小値ならRecovered認定
    constexpr uint64_t kRecoveredRetireLimit = 256;
    const bool idleRecovered = (now.pendingRetire == 0 && now.maxRetireAgeUs == 0);
    const bool retireWithinLimit = now.pendingRetire <= kRecoveredRetireLimit;
    const bool retireTrendImproving = (retireDelta < 0);
    if (!faultMaskIncreased
        && (pubDelta > 0 || idleRecovered)
        && (retireTrendImproving || retireWithinLimit)
        && ageDelta <= 0
        && now.readerStuckCount == 0 && now.activeReaderCount < 64)
        return RecoveryOutcome::Recovered;

    // Step 3: reader異常（Improvingより優先）
    if (now.readerStuckCount > before.readerStuckCount)
        return RecoveryOutcome::Worsening;
    if (now.pendingRetire > 0 && now.activeReaderCount == 0 && before.activeReaderCount > 0)
        return RecoveryOutcome::Worsening;

    // Step 4: Worsening
    if (faultMaskIncreased || retireDelta > 0 || ageDelta > 0)
        return RecoveryOutcome::Worsening;

    // Step 5: Improving
    // (a) retire改善: pendingRetire 減少
    const bool retireImproving = (retireDelta < -2);
    // (b) reader改善: readerStuckCount 減少（問題B）
    const bool readerImproving = (now.readerStuckCount < before.readerStuckCount);
    // (c) Restore専用: epochAdvanceCount 増加確認（問題A-1）
    //    Restore の効果は epochAdvance の促進で測定する。
    //    epochAdvanceCount増加なしでも retire/reader 改善があれば Improving とするが、
    //    閉ループ制御の Restore Step2 発行条件で epochAdvancing を必須とする。
    const bool epochAdvancing = (now.epochAdvanceCount > before.epochAdvanceCount);
    if ((retireImproving || readerImproving) && !faultMaskIncreased)
        return RecoveryOutcome::Improving;

    // epochAdvancing は Improving 判定に直接使用しない（問題A-1）:
    //   computeTrend() では retire/reader の改善を中心に評価する。
    //   epochAdvancing は Restore Step2 発行条件でのみ必須とする。
    //   これにより「epochAdvanceは停滞しているがretireは改善」というケースを
    //   Improving として扱える（Restoreが不要な軽度改善に対応）。

    // Step 6: Stalled
    return RecoveryOutcome::Stalled;
}
```

### 4.6 tick() 閉ループ制御組み込み

```cpp
// closed-loop control (before PolicyEngine evaluation)
{
    const auto lastAction = m_policyEngine_.getLastExecutedAction();
    if (lastAction > RecoveryAction::Observe) {
        auto& entry = m_policyEngine_.getEntry(lastAction);
        if (entry.state == VerificationState::PendingVerification) {
            const uint64_t nowUs = getCurrentTimeUs();
            if (nowUs - entry.executedAtUs >= entry.verifyAfterUs) {
                const auto nowSnapshot = takeSnapshot();
                const auto trend = computeTrend(entry.baselineSnapshot, nowSnapshot);

                switch (trend) {
                    case RecoveryOutcome::Recovered:
                        m_policyEngine_.resetVerification();
                        break;

                    case RecoveryOutcome::Improving: {
                        // Improving 時の retireReduction を計算（閉ループ共通）
                        const uint64_t retireReduction =
                            entry.lastSnapshot.pendingRetire > nowSnapshot.pendingRetire
                            ? entry.lastSnapshot.pendingRetire - nowSnapshot.pendingRetire : 0;
                        // stall判定: retireReductionRatio ベース（問題B-2: 絶対値<2→比率<1%に変更）
                        const uint64_t baselineRetire = entry.baselineSnapshot.pendingRetire;
                        const double reductionRatio = baselineRetire > 0
                            ? static_cast<double>(retireReduction) / baselineRetire : 0.0;
                        if (reductionRatio < 0.01)  // 1%未満の改善は停滞扱い
                            ++entry.stalledCount;
                        else
                            entry.stalledCount = 0;
                        entry.lastSnapshot = nowSnapshot;

                        if (entry.stalledCount >= 3) {
                            auto next = nextAction(lastAction);
                            if (m_policyEngine_.canExecute(next)) {
                                m_actionCallback(next);
                                m_policyEngine_.markExecuted(next);
                                m_policyEngine_.markForVerification(next, nowSnapshot);
                            }
                        } else {
                            entry.verifyAfterUs = std::min(
                                entry.verifyAfterUs * 2, uint64_t{30'000'000});
                            entry.executedAtUs = nowUs;
                        }
                        break;
                    }

                    case RecoveryOutcome::Stalled:
                    case RecoveryOutcome::Worsening:
                        // escalate: nextAction(lastAction)
                        break;
                }
            }
        }
    }
}
```

### 4.7 nextAction() — Restore を含む Ladder

```cpp
RecoveryAction nextAction(RecoveryAction current) noexcept {
    switch (current) {
        case RecoveryAction::Throttle: return RecoveryAction::Recover;
        case RecoveryAction::Recover:  return RecoveryAction::Restore;
        case RecoveryAction::Restore:  return RecoveryAction::Safe;
        case RecoveryAction::Safe:     return RecoveryAction::Critical;
        default:                       return RecoveryAction::Critical;
    }
}
```

---

## 5. Phase 4: RecoveryBudget（P0-D）

### 5.1 toRecoveryLevel()

```cpp
[[nodiscard]] constexpr uint8_t toRecoveryLevel(RecoveryAction action) noexcept {
    switch (action) {
        case RecoveryAction::Observe:  return 0;
        case RecoveryAction::Throttle: return 1;
        case RecoveryAction::Recover:  return 2;
        case RecoveryAction::Restore:  return 3;
        case RecoveryAction::Safe:     return 4;
        case RecoveryAction::Critical: return 5;
        default:                       return 0;
    }
}
```

#### 5.2 EscalationTracker 構造体（問題C/A-2: RecoveryStorm repeatCount + 同Action再突入→Critical固定）

```cpp
struct EscalationTracker {
    // action→index 明示的マッピング（問題C）:
    //   Observe=0, Throttle=1, Recover=2, Restore=3, Safe=4, Critical=5
    static constexpr uint8_t toIndex(RecoveryAction action) noexcept {
        switch (action) {
            case RecoveryAction::Observe:  return 0;
            case RecoveryAction::Throttle: return 1;
            case RecoveryAction::Recover:  return 2;
            case RecoveryAction::Restore:  return 3;
            case RecoveryAction::Safe:     return 4;
            case RecoveryAction::Critical: return 5;
            default:                       return 0;
        }
    }

    uint64_t lastActionUs[6]{0,0,0,0,0,0};  // per-action last exec time (6段階)
    uint32_t repeatCount[6]{0,0,0,0,0,0};   // per-action repeat count within 30s
    uint8_t  lastLevel{0};                   // 最後に記録した Recovery Level
    uint32_t levelOscillationCount{0};       // Level振動回数（問題C: Throttle↔Recover往復検出）
    uint32_t transitionPairCount[3]{0,0,0}; // TransitionPair: [Tr↔Rv]=0, [Rv↔Rs]=1, [Rs↔Sf]=2

    static constexpr uint64_t kStormWindowUs = 30 * 1'000'000;
    static constexpr uint32_t kMaxRepeatBeforeStorm = 3;
    static constexpr uint32_t kMaxOscillationBeforeStorm = 4;  // 4回往復→Critical

    [[nodiscard]] bool isStormDetected(RecoveryAction action, uint64_t nowUs) const noexcept {
        const auto idx = toIndex(action);
        if (lastActionUs[idx] == 0) return false;
        // repeatCount: 初回=0, 2回目=1, 3回目=2, 4回目=3 → >=3 で4回目トリガ（問題B-1）
        // 厳密に3回目でトリガする場合は kMaxRepeatBeforeStorm=2 に設定
        return (nowUs - lastActionUs[idx]) < kStormWindowUs
            && repeatCount[idx] >= kMaxRepeatBeforeStorm;
    }

    void record(RecoveryAction action, uint64_t nowUs) noexcept {
        const auto idx = toIndex(action);
        if (nowUs - lastActionUs[idx] < kStormWindowUs)
            ++repeatCount[idx];
        else
            repeatCount[idx] = 0;
        lastActionUs[idx] = nowUs;

        // Level振動検出: 1段差の往復（問題B-2）
        const uint8_t currentLevel = idx;
        if (lastLevel != 0 && currentLevel != lastLevel
            && std::abs(static_cast<int>(currentLevel) - static_cast<int>(lastLevel)) == 1) {
            ++levelOscillationCount;
            // TransitionPair 単位の監視: [Tr↔Rv]=0, [Rv↔Rs]=1, [Rs↔Sf]=2
            const uint8_t pairIdx = std::min(lastLevel, currentLevel);  // 低い方のLevelがpair index
            if (pairIdx >= 1 && pairIdx <= 3)
                ++transitionPairCount[pairIdx - 1];
        } else {
            levelOscillationCount = 0;
        }
        lastLevel = currentLevel;
    }

    void reset() noexcept {
        for (auto& tc : lastActionUs) tc = 0;
        for (auto& rc : repeatCount) rc = 0;
    }
};
```

**同Action再突入→Critical固定（問題A-2）**:
```cpp
// tick() 内、Budget 評価後:
if (m_recoveryBudget_.escalationTracker.isStormDetected(lastAction, nowUs)) {
    // 30秒以内に同一Actionが3回以上再突入 → Critical に固定
    diagLog("[RECOVERY] Storm detected: action=" + std::to_string(static_cast<int>(lastAction)));
    m_actionCallback(RecoveryAction::Critical);
    m_policyEngine_.markExecuted(RecoveryAction::Critical);
    m_policyEngine_.getBudget().reset();
}
```
```

#### 5.3 RecoveryBudget 構造体（修正版）

```cpp
struct RecoveryBudget {
    uint32_t cycleCountInWindow{0};
    uint32_t criticalCount{0};
    uint32_t recoverCount{0};
    uint8_t  ladderStep{0};
    uint64_t windowStartUs{0};
    uint64_t lastRecoverySuccessUs{0};
    uint64_t lastEscalationUs{0};
    EscalationTracker escalationTracker;  // 独立した Storm 検出器
    bool     latched{false};

    static constexpr uint64_t kBudgetWindowUs = 10 * 60 * 1'000'000;
    static constexpr uint64_t kStableResetUs  = 15 * 60 * 1'000'000;
    static constexpr uint32_t kMaxCyclesPerWindow = 3;
    static constexpr uint32_t kMaxCriticalCount = 5;
    static constexpr uint32_t kMaxRecoverCount = 20;

    [[nodiscard]] bool isExhausted(uint64_t nowUs) const noexcept;
    [[nodiscard]] bool isStormDetected(RecoveryAction action, uint64_t nowUs) const noexcept;
    void record(RecoveryAction action, uint64_t nowUs) noexcept;
    void recordCycleCompletion(uint64_t nowUs) noexcept;
    void recordHeavyReach(uint64_t nowUs) noexcept;
    void reset() noexcept;
};
```

### 5.3 Recovered 時の Budget 方針（問題D: 確定版）

RecoveryOutcome::Recovered 検出時は以下の順で Budget 処理する:

```cpp
case RecoveryOutcome::Recovered:
    m_policyEngine_.resetVerification();
    m_policyEngine_.getBudget().recordCycleCompletion(getCurrentTimeUs());
    break;
```

`recordCycleCompletion()` の実装方針:
- `cycleCountInWindow++`: 1サイクル完了としてカウント（根本原因未解消の兆候として残す）
- `ladderStep = 0`: ラダー初期化（次回は Throttle から再開）
- `repeatCount[] 維持`: ストーム検出用カウンタはクリアしない（高速再発検出のため）
- `latched = false`: Budget 回復

```
つまり「成功したが根本原因は残っている」という状態を repeatCount で検出可能にする。
```

### 5.4 Budget 記録（tick() 内 Action 発火箇所）

```cpp
if (m_policyEngine_.canExecute(action)) {
    m_actionCallback(action);
    m_policyEngine_.markExecuted(action);
    m_policyEngine_.getBudget().record(action, getCurrentTimeUs());

    if (m_policyEngine_.getBudget().isExhausted(getCurrentTimeUs())) {
        m_actionCallback(RecoveryAction::Critical);
        m_policyEngine_.markExecuted(RecoveryAction::Critical);
    }
}
```

---

## 6. Phase 5: Learner FIFO 監視（P1-B）

### 6.1 フィールド

```cpp
MonitorState m_prevLearnerBackpressureState_{MonitorState::Normal};
bool     m_learnerWasActive_{false};
double   m_fifoEma_{-1.0};        // -1.0 = uninitialized
double   m_lastFifoEma_{0.0};
uint64_t m_lastFifoTickUs_{0};
uint64_t m_learnerFifoHighSinceUs_{0};
uint64_t m_learnerSegmentBuffer_;  // AudioSegmentBuffer* (setter injection)
std::atomic<bool>* m_learnerRunningRef_{nullptr};
// epochAdvance / lastCompletedEpoch（問題A-1: Restore効果測定用）
std::atomic<uint64_t>* m_epochAdvanceCountRef_{nullptr};
std::atomic<uint64_t>* m_lastCompletedEpochRef_{nullptr};
```

### 6.2 checkLearnerBackpressure()

```cpp
void RuntimeHealthMonitor::checkLearnerBackpressure() noexcept {
    if (m_learnerRunningRef_ == nullptr) return;
    const bool learnerActive = convo::consumeAtomic(*m_learnerRunningRef_,
                                                     std::memory_order_acquire);

    // Learner restart detection → EMA reset
    if (!learnerActive) { m_learnerWasActive_ = false; return; }
    if (!m_learnerWasActive_) {
        m_fifoEma_ = -1.0;  m_lastFifoEma_ = 0.0;
        m_learnerFifoHighSinceUs_ = 0;  m_learnerWasActive_ = true;
    }

    const int available = m_learnerSegmentBuffer_
        ? m_learnerSegmentBuffer_->getNumAvailableSamples() : 0;
    constexpr int kCapacity = 3'840'000;
    const double fifoUsage = static_cast<double>(available) / kCapacity;

    // EMA (alpha=0.3)
    constexpr double kEmaAlpha = 0.3;
    if (m_fifoEma_ < 0.0) m_fifoEma_ = fifoUsage;
    m_fifoEma_ = kEmaAlpha * fifoUsage + (1.0 - kEmaAlpha) * m_fifoEma_;

    // Time-normalized slope
    const uint64_t nowUs = getCurrentTimeUs();
    const double elapsedSec = (m_lastFifoTickUs_ > 0)
        ? (nowUs - m_lastFifoTickUs_) / 1'000'000.0 : 1.0;
    const double slope = (m_fifoEma_ - m_lastFifoEma_) / std::max(elapsedSec, 0.001);
    m_lastFifoEma_ = m_fifoEma_;  m_lastFifoTickUs_ = nowUs;

    // 2-stage thresholds
    if (fifoUsage > 0.95 && slope >= 0.0)
        emitOnTransition(..., MonitorState::Error, ...);
    else if (fifoUsage > 0.85 && slope >= 0.0)
        emitOnTransition(..., MonitorState::Warning, ...);
    else if (fifoUsage <= 0.80 || slope < -0.01)
        m_prevLearnerBackpressureState_ = MonitorState::Normal;
}
```

---

## 7. Phase 6: Suppression Probe（P0-C）

### 7.1 フィールド

```cpp
std::atomic<uint32_t> m_probeBudget_{0};
std::atomic<uint64_t> m_lastProbeUs_{0};
std::atomic<bool>     m_suppressionActive_{false};

struct ProbeState {
    uint64_t publishSeqBefore{0};
    uint64_t pendingRetireBefore{0};
    uint64_t retireAgeBefore{0};       // Probe発行時点の maxRetireAge（問題E: 悪化検出用）
    uint64_t startedUs{0};
    // ProbeState 状態機械化（問題E: bool reserved は二重返却を許す）
    enum class ReserveState : uint8_t { Idle, Reserved, Committed, RolledBack };
    ReserveState reserveState{ReserveState::Idle};
    uint32_t failureCount{0};
};
ProbeState m_probeState_;

### 7.1a HardReset Publication Authority 一本化（問題E）

`publishIdleWorldOnly(HardReset)` は CrossfadeTimeout 回復と Restore Step2 の両方から呼ばれる。
競合を防ぐため、HardReset publish の Authority を世代番号で一本化する。

```cpp
// AudioEngine.h に追加
std::atomic<uint64_t> m_lastHardResetGeneration_{0};  // 最後に HardReset publish を行った restoreGeneration
```

**HardReset 発行時の排他ルール**:
1. CrossfadeTimeout 回復: `m_lastHardResetGeneration_` を参照せず常時発行（即時性優先）
2. Restore Step2: `m_lastHardResetGeneration_ == m_restoreGeneration_` の場合、既に発行済みとしてスキップ
3. 発行後: `m_lastHardResetGeneration_ = m_restoreGeneration_` を設定

```cpp
// Restore Step2 発行時:
if (m_lastHardResetGeneration_ != m_restoreGeneration_) {
    publishIdleWorldOnly(currentDSP, convo::TransitionPolicy::HardReset);
    m_lastHardResetGeneration_ = m_restoreGeneration_;
    m_restorePhase_ = RestorePhase::IdleWorldPublished;
}
```

### 7.2 ProbeBudget reserve/commit/rollback

```cpp
// CAS reserve: 1→0 only
uint32_t expected = 1;
if (convo::compareExchangeAtomic(m_probeBudget_, expected, uint32_t{0},
                                 std::memory_order_acq_rel, std::memory_order_acquire)) {
    m_probeState_.reserved = true;

    bool publishSucceeded = doPublish();

    // commit: publicationSeq増加のみ（問題D-1: pendingRetire/retireAgeは閉ループ制御側で評価）
    const uint64_t seqAfter = getLastCommittedPublicationSequence();
    if (publishSucceeded
        && seqAfter > m_probeState_.publishSeqBefore) {
    } else {
        // rollback: refund budget
        convo::fetchAddAtomic(m_probeBudget_, uint32_t{1}, std::memory_order_acq_rel);
        m_probeState_.reserved = false;
    }
}
```

### 7.3 Restore Step2 実行条件（問題A-1/A-2: 強化版）

Restore Step1 実行後、閉ループ制御の Improving 分岐で以下を確認:

```cpp
// In closed-loop control, Improving branch:
if (lastAction == RecoveryAction::Restore
    && nowSnapshot.restorePhase == RestorePhase::EpochRecoveryIssued)
{
    // A-1: epochAdvanceCount 増加確認（Restore が実際に効果を持ったかの判定）
    const bool epochAdvancing = (nowSnapshot.epochAdvanceCount
        > entry.baselineSnapshot.epochAdvanceCount);
    // A-2: 改善率が20%未満または age悪化している場合は Step2 を保留
    const uint64_t retireReduction = entry.lastSnapshot.pendingRetire > nowSnapshot.pendingRetire
        ? entry.lastSnapshot.pendingRetire - nowSnapshot.pendingRetire : 0;
    const uint64_t retireBaseline = entry.baselineSnapshot.pendingRetire;
    const double reductionRate = retireBaseline > 0
        ? static_cast<double>(retireReduction) / retireBaseline : 0.0;
    const int64_t ageDelta = static_cast<int64_t>(nowSnapshot.maxRetireAgeUs)
        - static_cast<int64_t>(entry.baselineSnapshot.maxRetireAgeUs);

    // 絶対量条件も併用（問題A: 小規模backlog対応のためOR条件）
    constexpr uint64_t kAbsoluteReductionMin = 10;
    const bool absoluteEnough = retireReduction >= kAbsoluteReductionMin;
    constexpr uint64_t kHealthyThreshold = 256;
    const bool nearlyHealthy = nowSnapshot.pendingRetire <= kHealthyThreshold;
    if (epochAdvancing
        && (reductionRate >= 0.20 || absoluteEnough || nearlyHealthy)
        && ageDelta <= 0) {
        publishIdleWorldOnly(currentDSP, convo::TransitionPolicy::HardReset);
        m_lastHardResetGeneration_ = m_restoreGeneration_;
        m_restorePhase_ = RestorePhase::IdleWorldPublished;
    }
    // 条件不成立: Step2 をスキップ、closed-loop は Improving として継続
}
```

> **注意**: CrossfadeTimeout 回復との競合(F22): 両者とも publishIdleWorldOnly(HardReset) を呼ぶ。CrossfadeTimeout の即時 callback が先に実行される。RestorePhase が IdleWorldPublished の場合、Step2 の publish をスキップする。

---

## 8. Phase 7: Critical 出口（P0-E）

### 8.1 CriticalExitCondition

```cpp
// Critical 出口ブロック理由（問題E-1: 診断用。pendingRetire超過も追加）
enum class CriticalExitBlocker : uint8_t {
    None,
    MonitorNotNormal,
    SuppressionActive,
    RecoveryRunning,
    StableDurationInsufficient,
    PendingRetireExceeded,
    RetireAgeExceeded
};

struct CriticalExitCondition {
    bool allMonitorsNormal{false};
    bool suppressionInactive{false};
    bool noRecoveryActionRunning{false};
    bool stableDuration{false};  // 60秒
    CriticalExitBlocker blocker{CriticalExitBlocker::None};

    [[nodiscard]] bool canExit() noexcept {
        // const 非対応: blocker 書き換えのため mutable または非const（問題A-4）
        if (!allMonitorsNormal) { blocker = CriticalExitBlocker::MonitorNotNormal; return false; }
        if (!suppressionInactive) { blocker = CriticalExitBlocker::SuppressionActive; return false; }
        if (!noRecoveryActionRunning) { blocker = CriticalExitBlocker::RecoveryRunning; return false; }
        if (!stableDuration) { blocker = CriticalExitBlocker::StableDurationInsufficient; return false; }
        blocker = CriticalExitBlocker::None;
        return true;
    }
};
```

### 8.2 tick() 評価

```cpp
if (m_healthState_ == ISRHealthState::Critical) {
    CriticalExitCondition exitCond;

    // 条件1: 全 MonitorState が Normal（Learner含む）
    exitCond.allMonitorsNormal = (m_prevRetireState == MonitorState::Normal)
        && (m_prevPublicationState == MonitorState::Normal)
        && (m_prevReaderSlotState == MonitorState::Normal)
        && (m_prevOverflowRateState == MonitorState::Normal)
        && (m_prevRetireAgeState == MonitorState::Normal)
        && (m_prevConfigDivergenceState_ == MonitorState::Normal)
        && (m_prevLearnerBackpressureState_ == MonitorState::Normal);

    // 条件2: Suppression 非アクティブ
    exitCond.suppressionInactive = ...;

    // 条件3: 閉ループ制御 Idle
    exitCond.noRecoveryActionRunning =
        m_policyEngine_.getEntry(m_policyEngine_.getLastExecutedAction()).isIdle()
        && !m_policyEngine_.hasPendingVerification();

    // 条件4: RetireDepth + RetireAge 実メトリクス確認
    const uint64_t retirePending = m_retireRouter ? m_retireRouter->pendingRetireCount() : 0;
    const uint64_t retireAge = m_maxRetireAgeRef
        ? convo::consumeAtomic(*m_maxRetireAgeRef, std::memory_order_acquire) : 0;
    // 条件4b: ReaderStuck なし（問題D-1: MonitorState は閾値ベースのため補完）
    const bool readerHealthy = (m_retireRouter == nullptr
        || m_retireRouter->readerStuckCount() == 0);
    constexpr uint64_t kHealthyThreshold = 256;
    constexpr uint64_t kHealthyAgeUs = 3 * 1'000'000;
    // 条件4c: RetireAge 明示確認（問題F: MonitorState が Normal でも retireAge が高い可能性）
    const bool retireAgeHealthy = retireAge < kHealthyAgeUs;
    exitCond.allMonitorsNormal = exitCond.allMonitorsNormal
        && retirePending < kHealthyThreshold
        && retireAgeHealthy
        && readerHealthy;
    if (!retireAgeHealthy)
        exitCond.blocker = CriticalExitBlocker::RetireAgeExceeded;

    // 条件5: 安定60秒継続
    if (exitCond.allMonitorsNormal) { /* stable duration tracking */ }

    if (exitCond.canExit()) {
        // Critical exit detected (monitoring only)
        // 実際の HealthState 復帰は updateHealthState() が担当:
        //   全 MonitorState が Normal に戻ると ISRHealthState::Healthy へ自動遷移
        //   CriticalExitCondition は監視専用。canExit() 後に強制遷移は行わない。
        //   遷移保証: updateHealthState() は tick() 末尾で常時実行される。
        //   canExit()==true なら次 tick の updateHealthState() で Healthy 復帰が期待できる。
        diagLog("[RECOVERY] Critical exit conditions met");
    }
}
```

---

## 9. ファイル影響マップ

| Phase | ファイル | 操作 | 内容 |
| --- | --- | --- | --- |
| 1 | AudioEngine.Timer.cpp | 修正 | Restore case 置換 |
| 1 | AudioEngine.h | 追加 | m_restoreGeneration_, m_restorePhase_ |
| 1 | RuntimePolicyEngine.h | 追加 | TrendSnapshot.restoreGeneration/restorePhase |
| 2 | RuntimeHealthMonitor.h | 修正 | injectBackpressureSignal() 拡張 |
| 2 | RuntimeHealthMonitor.h | 追加 | m_maxFallbackSize_, m_maxOverflowRate_ |
| 2 | RuntimeHealthMonitor.cpp | 修正 | tick() 背圧評価 exchange(0) |
| 2 | AudioEngine.Retire.cpp | 修正 | L155-161/L291 injectBackpressureSignal化 |
| 3 | RuntimePolicyEngine.h | 修正 | RecoveryOutcome enum |
| 3 | RuntimePolicyEngine.h | 追加 | TrendSnapshot, VerificationEntry, Budget |
| 3 | RuntimePolicyEngine.h | 追加 | RuntimePolicyEngine 拡張メソッド群 |
| 3 | RuntimeHealthMonitor.h | 追加 | computeTrend(), takeSnapshot() |
| 3 | RuntimeHealthMonitor.cpp | 追加 | computeTrend(), closed-loop |
| 4 | RuntimePolicyEngine.h | 追加 | toRecoveryLevel(), RecoveryBudget |
| 4 | RuntimeHealthMonitor.cpp | 修正 | Budget記録 |
| 5 | RuntimeHealthMonitor.h | 追加 | Learner FIFO フィールド |
| 5 | RuntimeHealthMonitor.cpp | 追加 | checkLearnerBackpressure() |
| 5 | RuntimeHealthMonitor.cpp | 修正 | tick() 呼出追加 |
| 6 | AudioEngine.h | 追加 | Probe fields, ProbeState |
| 6 | AudioEngine.RebuildDispatch.cpp | 修正 | ProbeBudget CAS消費 |
| 6 | RuntimeHealthMonitor.cpp | 修正 | RestoreStep2 発行 |
| 7 | RuntimeHealthMonitor.h | 追加 | CriticalExitCondition |
| 7 | RuntimeHealthMonitor.cpp | 修正 | Critical出口評価 |

---

## 10. 実装順序

```text
Phase 2 (背圧統合)     ← 独立。最優先実装
    ↓
Phase 5 (Learner FIFO) ← 独立。Phase 2と並行可
    ↓
Phase 1 (Restore)      ← Phase 2完了後が望ましい
    ↓
Phase 3 (閉ループ制御)  ← Phase 1,2完了後
    ↓
Phase 4 (Budget)       ← Phase 3完了後
    ↓
Phase 6 (Probe)        ← Phase 2,3完了後
    ↓
Phase 7 (Critical出口)  ← Phase 3,5完了後
```

### 検証コマンド

| Phase | 検証 |
| --- | --- |
| 2 | `grep -c "retirePressureAdmissionStrict_" AudioEngine.Retire.cpp` = 1 |
| 3 | ログ `[RECOVERY] ... outcome=Improving/Stalled/Worsening` |
| 4 | ログ `[RECOVERY] Budget exhausted` |
| 5 | ログ `[HEALTH] eventCode=5002` |
| 6 | ログ `[RECOVERY] Suppression probe` |
| 7 | ログ `[RECOVERY] Critical exit conditions met` |
