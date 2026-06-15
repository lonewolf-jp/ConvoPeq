# Recovery System 改修計画書 — 確定版

> **日付**: 2026-06-15
> **ベース文書**: doc/work37/recovery_system_plan.md (v7.1最終版)
> **事前調査ツール**: grep/Select-String, Serena MCP, CodeGraph MCP (51K entities), AiDex MCP (275 files)
> **調査確定事項**: 全9項目を grep + Serena + CodeGraph で確定。未確定事項ゼロ。

---

## 0. 前提

本計画は Practical Stable ISR Bridge Runtime の観点で作成する。「未実装項目を埋める」ことと「実運用で破綻しない回復系を作る」ことは別であり、後者を優先する。

### 0.1 調査ツールによる確定事項

以下の調査は全て grep/Select-String, Serena MCP, CodeGraph MCP (51K entities), AiDex MCP (275 files) で確定済み:

| ファクト | 確定内容 | 確定方法 |
| --- | --- | --- |
| `RecoveryOutcome` | 定義のみ、全 .cpp で未使用（参照元ゼロ） | Serena find_referencing_symbols + grep |
| `computeRuntimeRecoveryScore()` | 定義+実装あり、しかしどの .cpp からも未呼び出し | Serena search_for_pattern |
| `lastHealthyWorldId_` | 書き込みのみ、読み取り元がコードベースに存在しない | grep (書き込み1件、読み取り0件) |
| `injectBackpressureSignal` | 定義のみ、呼び出し元がコードベースに存在しない | grep (定義1件のみ) |
| `ISRRetireRuntimeEx::requestRollback()` | 引数なし（worldId 不要）。内部 Rollback | Serena find_symbol + body |
| `ISRRetireRuntimeEx::canRollback()` | `rollbackReady_` を返す | コード読み取り |
| `NoiseShaperLearner::setState()` | 存在する（Learner Rollback に使用可） | Serena find_symbol + body (L111) |
| `AudioSegmentBuffer` | `kCapacity = 3,840,000`, `getNumAvailableSamples()` あり | コード読み取り |
| `segmentBuffer.getNumAvailableSamples()` | NoiseShaperLearner.cpp:913 で既にログ出力中 | grep |
| `shouldRejectRebuildAdmissionForPressure()` | 3箇所 (L241, L428, L479) で rebuild 抑制。`retirePressureAdmissionStrict_` + `HealthState::Critical` の2重 | grep |

## 0.2 現状診断サマリ

### 0.1 既に動作しているもの（変更不要）

| 領域 | 状態 | 確認方法 |
| --- | --- | --- |
| RecoveryAction 6段階 enum (Observe/Throttle/Recover/Restore/Safe/Critical) | ✅ 実装済み | RuntimePolicyEngine.h:43-51 |
| executeRecoveryAction() switch (全6段階) | ✅ 実装済み | AudioEngine.Timer.cpp:659-702 |
| RuntimePolicyEngine (MonitorState→Action選択器) | ✅ 実装済み | RuntimePolicyEngine.cpp:58-110 |
| checkConfigurationDivergence() (世代乖離監視) | ✅ 実装済み | RuntimeHealthMonitor.cpp:573-614 |
| checkSuppressionDuration() (抑制時間段階的昇格) | ✅ 実装済み | RuntimeHealthMonitor.cpp:704-728 |
| checkRuntimeProgressFreeze() (3軸統合) | ✅ 実装済み | RuntimeHealthMonitor.cpp:732-780 |
| checkWorldConsistency() (毎tick実行) | ✅ 実装済み | RuntimeHealthMonitor.cpp:618-630 |
| ReaderStuck詳細診断 (readerIndex/epoch/depth/residencyUs) | ✅ 実装済み | RuntimeHealthMonitor.cpp:295-347 |
| `notifyHealthyPublication()` — lastHealthyWorldId_ の記録 | ✅ 実装済み | AudioEngine.CtorDtor.cpp:203-216 |
| `stopNoiseShaperLearning()` — Safe Mode 経由の学習停止 | ✅ 実装済み | AudioEngine.Learning.cpp:39-49 |

### 0.2 不足しているもの（改修対象）

| # | 項目 | 優先度 | コード状態 | 実装度 |
| --- | --- | --- | --- | --- |
| **P0-A** | RecoveryOutcome 閉ループ制御 | **P0** | 定義のみ、全 .cpp で未使用 | **0%** |
| **P0-B** | Restore→LastHealthyWorld Rollback | **P0** | 記録のみ、切戻し未配線 | **40%** |
| **P0-C** | Suppression Escape 機構 | **P0** | 未実装 | **0%** |
| **P1-A** | injectBackpressureSignal 統合 | **P1** | 定義のみ、未呼び出し | **0%** |
| **P1-B** | LearnerBackpressure FIFO 監視 | **P1** | PolicySource/HealthCause 定義のみ | **30%** |
| **P2-A** | ForcePublicationRecovery | **P2** | 未実装 | **0%** |
| **P2-B** | AudioQualityFingerprint | **P2** | 未実装 | **0%** |
| **P2-C** | ClearSuppression | **P2** | 未実装 | **0%** |

### 0.3 調査で確定した重要ファクト

#### ファクトA: RecoveryOutcome は定義のみ — grep 確定

```
$ grep -r "RecoveryOutcome" src/**/*.cpp
→ 0件
$ grep -r "RecoveryOutcome" src/**
→ src/audioengine/RuntimePolicyEngine.h:36  (PolicySource コメント内)
→ src/audioengine/RuntimePolicyEngine.h:52-60 (enum 定義)
```

Serena `find_referencing_symbols(convo/RecoveryOutcome)` → 空結果（参照元ゼロ）。

#### ファクトB: computeRuntimeRecoveryScore() は未使用

- `RuntimeHealthMonitor.h:293` で宣言
- `RuntimeHealthMonitor.cpp:818-856` で実装
- **どの .cpp からも呼び出されていない**
- 4軸スコアを計算するが、RecoveryAction の選択や閉ループ制御には使われていない

#### ファクトC: Restore は drain のみ。Rollback なし

```cpp
// AudioEngine.Timer.cpp:679-683
case convo::RecoveryAction::Restore:
    tryReclaimResources();
    drainDeferredRetireQueues(false);
    // Rollback 基盤（ISRRetireRuntimeEx）の設定は呼び出し元が行う  ← 呼び出し元なし
    break;
```

`ISRRetireRuntimeEx` には `setRollbackMode()` / `requestRollback()` / `canRollback()` が存在するが、**executeRecoveryAction() から呼ばれていない**。`lastHealthyWorldId_` は書き込みのみで、読み取り箇所がコードベースに存在しない。

#### ファクトD: injectBackpressureSignal は定義のみ

```cpp
// RuntimeHealthMonitor.h:146-151 — 定義のみ
void injectBackpressureSignal(std::size_t fallbackSize, double overflowRate) noexcept {
    m_injectedFallbackSize_ = fallbackSize;
    m_injectedOverflowRate_ = overflowRate;
    m_backpressureInjected_ = true;
}
```

`RuntimeHealthMonitor::tick()` L48-56 で `m_backpressureInjected_` を読み取って Throttle 判定に反映するコードはあるが、**`AudioEngine.Retire.cpp` からの呼び出しがない**。Retire.cpp は依然として直接 `retirePressureAdmissionStrict_` を書き換えている (L157, L291)。

#### ファクトE: LearnerBackpressure FIFO 監視は未実装

- `HealthCause::LearnerBackpressure` は定義済み ✅
- `PolicySource::LearnerAnomaly → Throttle` マッピングは PolicyEngine に実装済み ✅
- しかし `checkLearnerBackpressure()` 関数が存在しない ❌
- 現在の Learner 保護は `stallDur > 10s && learnerActive → Throttle` の単一条件のみ

#### ファクトF: 直接書き込み経路の残存

`retirePressureAdmissionStrict_` への書き込み経路（全11箇所）:

| 経路 | ファイル | 行 | PolicyEngine経由？ |
| --- | --- | --- | --- |
| overflow → strict | AudioEngine.Retire.cpp | 157 | ❌ 直接 |
| retirePressureLevel ≥3 → strict | AudioEngine.Retire.cpp | 291 | ❌ 直接 |
| EVENT_READER_SLOT_USAGE → strict | AudioEngine.Timer.cpp | 558 | ❌ onHealthEvent 内 |
| EVENT_RETIRE_STALL → strict | AudioEngine.Timer.cpp | 592 | ❌ onHealthEvent 内 |
| executeRecoveryAction(Throttle) → strict | AudioEngine.Timer.cpp | 663 | ✅ PolicyEngine |
| executeRecoveryAction(Safe) → false | AudioEngine.Timer.cpp | 691 | ✅ PolicyEngine |
| executeRecoveryAction(Critical) → strict | AudioEngine.Timer.cpp | 697 | ✅ PolicyEngine |

3経路が PolicyEngine 非経由で直接書き込みを行っている。

---

## 1. 改修フェーズ

### Phase P0-A: RecoveryOutcome 閉ループ制御（改善設計版）

#### 1.1 設計方針

現在の Recovery System は「Action を発火する」だけの Open Loop。RecoveryAction 実行後、N秒後に効果を**傾向で判定**し、改善傾向なら維持、停滞なら昇格する閉ループ制御を追加する。

**重要な設計判断**: 固定閾値（80%/120%等）で Success/Failed を判定すると、「ゆっくり改善中」を NoEffect と誤判定して誤昇格する。代わりに**傾向ベース（Improving/Stalled/Worsening/Recovered）** の4状態を用いる。

**問題2対応: Improving に上限を設定** — 極めて遅い改善が永久維持されるのを防ぐ。`improvingCount` が 3 回に達したら Stalled 扱い。

```
RecoveryAction 実行
    ↓
N秒待機（verifyAfterUs）
    ↓
改善傾向を評価（勾配ベース）
    ├→ Improving (count<3) → 維持（verifyAfterUs *= 2）
    ├→ Improving (count≥3) → Stalled 扱い（上限到達）
    ├→ Recovered → Normal 復帰 + Budgetリセット
    ├→ Stalled   → 次段階へ昇格
    └→ Worsening → 即時昇格（待機時間無視）
```

**問題3対応: TrendSnapshot に3軸を含める** — publication だけでなく retireProgress と readerHealth を追加。

```cpp
struct TrendSnapshot {
    uint64_t pendingRetire{0};
    uint64_t publicationSeq{0};
    uint64_t maxRetireAgeUs{0};
    uint64_t retireHwm{0};
    // ★ 追加: ProgressFreeze 3軸
    uint32_t activeReaderCount{0};
    uint32_t readerStuckCount{0};
    bool     freezeDetected{false};  // checkRuntimeProgressFreeze の結果
};
```

#### 1.2 RecoveryOutcome の再定義

**ファイル**: `src/audioengine/RuntimePolicyEngine.h`

```cpp
// ★ 修正: 傾向ベース4状態。固定閾値は使わない。
enum class RecoveryOutcome : uint8_t {
    None,          // 未評価
    Improving,     // 改善傾向（勾配負）— 昇格禁止
    Recovered,     // 正常復帰 — Observe へ
    Stalled,       // 停滞（勾配ゼロ付近）— 昇格
    Worsening      // 悪化傾向（勾配正）— 即昇格
};
```

#### 1.3 傾向評価 computeTrend()（修正版: 問題1/2対応）

**問題1対応**: `pubDelta` 単独での Improving 判定を禁止。改善の主指標を retireDelta とし、publication は補助指標にする。freezeDetected が true なら即 Worsening。

**問題2対応**: `freezeDetected` を実際に使用する。

```cpp
// ★ 新規: 傾向ベース回復評価（問題4対応: Recovered条件緩和）
RecoveryOutcome RuntimeHealthMonitor::computeTrend(
    RecoveryAction action,
    const TrendSnapshot& before,
    const TrendSnapshot& now) const noexcept
{
    // ProgressFreeze 検出中は即 Worsening
    if (now.freezeDetected)
        return RecoveryOutcome::Worsening;

    const int64_t retireDelta = static_cast<int64_t>(now.pendingRetire)
                              - static_cast<int64_t>(before.pendingRetire);
    const int64_t ageDelta    = static_cast<int64_t>(now.maxRetireAgeUs)
                              - static_cast<int64_t>(before.maxRetireAgeUs);
    const int64_t pubDelta    = static_cast<int64_t>(now.publicationSeq)
                              - static_cast<int64_t>(before.publicationSeq);

    // ★ 問題A-1: Recovered は全主要監視軸の正常を要求
    //   readerStuckCount==0 かつ activeReaderCount < kReaderHealthyLimit
    constexpr uint64_t kHealthyRetireDepth = 256;
    constexpr uint32_t kReaderHealthyLimit = 64;
    if (retireDelta < 0 && pubDelta > 0 && ageDelta <= 0
        && now.pendingRetire <= kHealthyRetireDepth
        && now.readerStuckCount == 0
        && now.activeReaderCount < kReaderHealthyLimit)
        return RecoveryOutcome::Recovered;

    if (retireDelta > 0 || ageDelta > 0)
        return RecoveryOutcome::Worsening;

    const bool retireImproving = (retireDelta < -2);
    const bool publicationAlive = (pubDelta > 0);
    if (retireImproving && publicationAlive)
        return RecoveryOutcome::Improving;

    return RecoveryOutcome::Stalled;
}

// ★ 新規: スナップショット構造体（問題3反映: freezeDetected を含む全3 axis）
struct TrendSnapshot {
    uint64_t pendingRetire{0};
    uint64_t publicationSeq{0};
    uint64_t maxRetireAgeUs{0};
    uint64_t retireHwm{0};
    uint32_t activeReaderCount{0};
    uint32_t readerStuckCount{0};
    bool     freezeDetected{false};  // ProgressFreeze 検出結果
};
```

#### 1.4 閉ループ制御の昇格ロジック

```cpp
// tick() 内
{
    const auto lastAction = m_policyEngine_.getLastExecutedAction();
    if (lastAction > RecoveryAction::Observe) {
        const auto& entry = m_verificationTracker_.getEntry(lastAction);
        if (entry.state == RecoveryVerificationState::PendingVerification) {
            const uint64_t nowUs = getCurrentTimeUs();
            if (nowUs - entry.executedAtUs >= entry.verifyAfterUs) {
                const auto trend = computeTrend(lastAction,
                    entry.snapshotBefore, takeSnapshot());

                // ★ 問題3: verifyAfterUs 上限 30秒
                entry.verifyAfterUs = std::min(
                    entry.verifyAfterUs, uint64_t{30'000'000});

                switch (trend) {
                    case RecoveryOutcome::Recovered:
                        // ★ 問題3: Budget 全リセット禁止。ladderStep のみ初期化
                        m_verificationTracker_.reset();
                        m_recoveryBudget_.ladderStep = 0;
                        break;

                    case RecoveryOutcome::Improving:
                        // ★ 問題A-2: baselineSnapshot（累積）/ lastSnapshot（直前）の二系統
                        //   baselineSnapshot との累積改善率を計算
                        const double totalImproveRatio =
                            static_cast<double>(entry.baselineSnapshot.pendingRetire
                                - nowSnapshot.pendingRetire)
                            / static_cast<double>(std::max<uint64_t>(
                                entry.baselineSnapshot.pendingRetire, 1));
                        // ★ 問題A-3: 改善率≥15%なら count 増加なし（正当な改善）
                        if (totalImproveRatio < 0.15) {
                            ++entry.improvingCount;
                        }
                        entry.lastSnapshot = nowSnapshot;
                        if (entry.improvingCount >= 3) {
                            escalateAndMark(nextAction(lastAction));
                        } else {
                            entry.verifyAfterUs = std::min(
                                entry.verifyAfterUs * 2,
                                uint64_t{30'000'000});
                        }
                        break;

                    case RecoveryOutcome::Stalled:
                        // 停滞 → 次段階へ昇格
                        escalateAndMark(nextAction(lastAction));
                        break;

                    case RecoveryOutcome::Worsening:
                        // 悪化 → 即時昇格（待機時間を無視）
                        escalateAndMark(nextAction(lastAction));
                        break;
                }
            }
        }
    }
}
```

**影響ファイル**: `RuntimePolicyEngine.h`, `RuntimePolicyEngine.cpp`, `RuntimeHealthMonitor.h`, `RuntimeHealthMonitor.cpp`

---

### Phase P0-B: Restore → LastHealthyWorld Rollback

#### 2.1 設計方針

`RecoveryAction::Restore` が現在 `tryReclaim()` + `drainDeferredRetireQueues()` しか実行していないのを、実際の World Rollback に変更する。

**Serena/CodeGraph/AiDex 調査で確定したファクト**:

- `ISRRetireRuntimeEx::requestRollback()` → **実体は `setEpochMode(getRollbackMode())`**（ISRRetireRuntimeEx.cpp:280-283）
  → **World Rollback ではなく Epoch Mode 切替**であることが確定
- `ISRRetireRuntimeEx::canRollback()` → `rollbackReady_` atomic（`ISRRetireRuntimeEx.h:34`, ISRRetireRuntimeEx.cpp:274-276）
- `NoiseShaperLearner::setState(const State&)` → L111 に存在（Learner Rollback に使用可）
- `publishIdleWorldOnly()` → `AudioEngine.h:2334` に存在

**重要**: `requestRollback()` は `lastHealthyWorldId_` を参照しない。従って本設計における Restore は「復元」ではなく「Epoch Recovery + Learner Rollback + Idle World 発行」の複合動作となる。真の World Rollback は将来の新規機能として要検討。

```cpp
case convo::RecoveryAction::Restore:
{
    // 1. Epoch Rollback 要求（Epoch Mode 切替）
    if (!retireRuntimeEx_.canRollback()) {
        break;  // 閉ループ制御が Safe へ昇格
    }
    retireRuntimeEx_.setRollbackMode(convo::isr::EpochMode::Split);
    retireRuntimeEx_.requestRollback();  // setEpochMode(rollbackMode)

    // 2. Learner Rollback（setState 確認済み）
    if (lastKnownGoodNoiseShaper_.isValid) {
        noiseShaperLearner->setState(lastKnownGoodNoiseShaper_.state);
    }

    // 3. Idle World 発行（現在の DSP 状態で）
    {
        const convo::RuntimeReaderContext messageCtx{
            messageThreadRcuReader, convo::ObserveChannel::Message };
        const auto runtimeReadHandle = makeRuntimeReadHandle(messageCtx);
        auto* currentDSP =
            resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        (void)publishIdleWorldOnly(currentDSP,
            convo::TransitionPolicy::HardReset);
    }
    break;
}
```

#### 2.2 制約と将来課題

- `lastHealthyWorldId_` / `getLastHealthyWorldId()` は **P0-B では使用しない**（`requestRollback()` が worldId を受け付けないため）
- 代わりに `notifyHealthyPublication()` による記録は維持し、将来の真の World Rollback 機構に備える
- `lastKnownGoodNoiseShaper_.state` は Learner Rollback に使用する（`setState()` 確認済み）
- Epoch Mode 切替後の実際の効果検証は閉ループ制御（P0-A）の `computeTrend()` で行う

#### 2.3 Restore 成功判定（問題5/9対応）

Epoch Mode 切替は即座に成功/失敗が確定しない。成功判定は閉ループ制御に委ね、`computeTrend()` が pendingRetire 減少を検出した時点で Recovered とする。

```cpp
// AudioEngine.Timer.cpp — Restore 実行後は閉ループ制御に委ねる
case convo::RecoveryAction::Restore:
{
    if (!retireRuntimeEx_.canRollback()) break;
    retireRuntimeEx_.setRollbackMode(convo::isr::EpochMode::Split);
    retireRuntimeEx_.requestRollback();
    if (lastKnownGoodNoiseShaper_.isValid)
        noiseShaperLearner->setState(lastKnownGoodNoiseShaper_.state);
    {
        // ... publishIdleWorldOnly ...
    }
    // ★ 成功判定は閉ループ制御の computeTrend() が担当
    //   pendingRetire 減少を検出したら Recovered、不変なら Stalled→Safe へ
    break;
}
```

**問題6対応**: `requestRollback()` の実体が Epoch Mode 切替のみであるため、Restore の価値は限定的。
Practical Stable の観点から **案B（単純化）を採用** する。
`Throttle → Recover → Safe` の3段階。Restore は削除し、Safe への移行時間を短縮する。

```text
Throttle (L1) → Recover (L2) → Safe (L3) → Critical (L4)
```

**影響ファイル**: `AudioEngine.Timer.cpp`（Restore case 削除、-15行）

---

### Phase P0-C: Suppression Probe（改善設計版）

#### 3.1 設計方針

Suppression を完全解除すると、原因が残ったまま再び Publish が流入し振動系になる。代わりに **Probe（探索的1回だけ許可）** で回復の可能性を探る。

**なぜ Probe が必要か**: `shouldRejectRebuildAdmissionForPressure()`（`AudioEngine.Threading.cpp:20`）は `retirePressureAdmissionStrict_` と `HealthState::Critical` の二重判定をしている。`admissionStrict_` を単に false にしても、HealthState が Critical のままなら rebuild は抑制され続ける。

```
Suppression 継続中
    ↓
30秒経過
    ↓
Probe: 1 publication だけ許可
    ├→ 成功（retireDepth 減少）→ 全解除
    └→ 失敗（retireDepth 不変）→ 再 Suppression（+ HealthState Critical 維持）
```

**問題5対応: bool → `std::atomic<uint32_t>` に変更**。仮に複数箇所から同時に publication が来ても、probeBudget のデクリメントで正確に1回だけ許可する。

```cpp
// ★ Probe カウンタ: 正確に1 publication だけ許可する atomic カウンタ
static constexpr uint64_t kSuppressionProbeIntervalUs = 30'000'000; // 30秒
std::atomic<uint32_t> m_probeBudget_{0};
```

**問題6対応**: 消費位置を admission check から publish 成功直前に移動。
現在の設計は「1 admission check 通過」であり「1 publication 実行」ではない。

**問題5対応**: ABA 競合防止のため CAS 方式を使用（fetch_sub は2スレッド同時通過のリスク）。

```cpp
// ★ publish 成功直前で CAS 消費。ABA 安全。
// AudioEngine の publish 実行箇所（例: submitRebuildIntent 内）:
uint32_t budget = convo::consumeAtomic(m_probeBudget_,
                                        std::memory_order_acquire);
while (budget > 0) {
    if (convo::compareExchangeAtomic(m_probeBudget_, budget, budget - 1,
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
        // 1 消費成功 → publish 許可
        break;
    }
    budget = convo::consumeAtomic(m_probeBudget_,
                                   std::memory_order_acquire);
}
if (budget == 0) {
    // 通常の admission チェック
    if (shouldRejectRebuildAdmissionForPressure())
        return;
}
```

```cpp
// tick() 内 — 30秒ごとに probeBudget を設定（問題C-3: 0の時のみ設定）
if (m_suppressionActive_
    && convo::consumeAtomic(m_probeBudget_, std::memory_order_acquire) == 0) {
    const uint64_t nowUs = convo::getCurrentTimeUs();
    if (nowUs - m_lastProbeUs_ >= kSuppressionProbeIntervalUs
        && nowUs - m_suppressionStartUs_ >= kSuppressionProbeIntervalUs) {
        convo::publishAtomic(m_probeBudget_, uint32_t{1}, std::memory_order_release);
        m_retireBeforeProbe_ = m_retireRouter
            ? m_retireRouter->pendingRetireCount() : 0;
        m_lastProbeUs_ = nowUs;
        diagLog("[RECOVERY] Suppression probe budget=1");
    }
}
```

**問題C-2対応**: Probe 成功条件（tick() 内で評価）

```cpp
// ★ Probe 成功判定: 5%以上改善で成功
static constexpr double kProbeSuccessRatio = 0.05;
if (m_lastProbeUs_ > 0 && m_suppressionActive_) {
    const uint64_t retireNow = m_retireRouter
        ? m_retireRouter->pendingRetireCount() : 0;
    const uint64_t retiredDelta = m_retireBeforeProbe_ > retireNow
        ? m_retireBeforeProbe_ - retireNow : 0;
    const double reductionRatio = m_retireBeforeProbe_ > 0
        ? static_cast<double>(retiredDelta) / static_cast<double>(m_retireBeforeProbe_)
        : 0.0;
    if (reductionRatio >= kProbeSuccessRatio) {
        diagLog("[RECOVERY] Probe succeeded: reduction="
            + juce::String(reductionRatio));
        // 閉ループ制御が Recovered を判定
    }
}
```

#### 3.2 既存の checkSuppressionDuration() との共存

`checkSuppressionDuration()` は段階的エスカレーション（30s→60s→120s→180s）を HealthEvent として発火する。Probe はこれと独立して動作する。Probe が成功しても HealthEvent 通知は継続される。

**影響ファイル**: `AudioEngine.h`（`m_probeRequested_`, `m_suppressionActive_`, `m_lastProbeUs_` 追加）, `AudioEngine.Threading.cpp`（`shouldRejectRebuildAdmissionForPressure()` 先頭に Probe チェック追加）, `RuntimeHealthMonitor.cpp`（tick() に Probe 設定ロジック）

---

### Phase P0-D: RecoveryBudget（新規追加）

#### 3A.1 問題

閉ループ制御を実装すると、別の永久ループが発生し得る:

```text
Throttle → Recover → Restore → Safe → (再び) Throttle → Recover → ...
```

これを防ぐために、各 RecoveryAction の累積実行回数に上限を設ける。

#### 3A.2 設計

```cpp
**問題D-1対応**: 累積回数固定からレート制限型に変更。

```cpp
struct RecoveryBudget {
    uint32_t cycleCountInWindow{0}; // 窓内の cycle 回数
    uint32_t criticalCount{0};
    uint8_t  ladderStep{0};
    uint64_t windowStartUs{0};

    static constexpr uint64_t kBudgetWindowUs = 10 * 60 * 1'000'000; // 10分窓
    static constexpr uint32_t kMaxCyclesPerWindow = 3;  // 10分間に3回まで
    static constexpr uint32_t kMaxCriticalCount = 5;

    [[nodiscard]] bool isExhausted(uint64_t nowUs) const noexcept {
        if (nowUs - windowStartUs > kBudgetWindowUs)
            return false;  // 窓外
        return cycleCountInWindow >= kMaxCyclesPerWindow
            || criticalCount >= kMaxCriticalCount;
    }

    void record(RecoveryAction action, uint64_t nowUs) noexcept {
        // 窓リセット
        if (nowUs - windowStartUs > kBudgetWindowUs) {
            cycleCountInWindow = 0;
            windowStartUs = nowUs;
        }
        const uint8_t step = toRecoveryLevel(action);  // ★ D-2: 関数化
        if (step <= 1 && ladderStep >= static_cast<uint8_t>(RecoveryAction::Safe))
            ++cycleCountInWindow;
        ladderStep = step;
        if (action == RecoveryAction::Critical)
            ++criticalCount;
    }

    void reset() noexcept {
        cycleCountInWindow = 0;
        criticalCount = 0;
        ladderStep = 0;
        windowStartUs = 0;
    }
};

// ★ 問題D-2: enum 順序非依存の level 取得関数
[[nodiscard]] constexpr uint8_t toRecoveryLevel(RecoveryAction action) noexcept {
    switch (action) {
        case RecoveryAction::Observe:  return 0;
        case RecoveryAction::Throttle: return 1;
        case RecoveryAction::Recover:  return 2;
        case RecoveryAction::Safe:     return 3;
        case RecoveryAction::Critical: return 4;
        default:                       return 0;
    }
}
```
```

#### 3A.3 Budget リセット条件（問題1対応）

Budget がリセットされないまま数時間固定化するのを防ぐため、以下の2条件でリセットする:

1. **Recovered 判定時**: 閉ループ制御が Recovered を検出したら Budget を全リセット
2. **長時間正常運転**: `m_lastCriticalExitUs_` からの経過時間が 30分を超えたらリセット

```cpp
// RuntimePolicyEngine.h に追加
static constexpr uint64_t kBudgetResetAfterUs = 30 * 60 * 1'000'000; // 30分
```

```cpp
// tick() 内: Budget 自動リセット
{
    const auto& entry = m_verificationTracker_.getEntry(lastAction);
    if (entry.outcome == RecoveryOutcome::Recovered) {
        m_recoveryBudget_.reset();  // 回復成功時にリセット
    }
    // 長時間正常運転でもリセット
    if (convo::getCurrentTimeUs() - m_lastCriticalExitUs_ >= kBudgetResetAfterUs) {
        m_recoveryBudget_.reset();
    }
}
```

```cpp
// markExecuted 時に Budget を記録
if (m_policyEngine_.canExecute(action)) {
    m_actionCallback(action);
    m_policyEngine_.markExecuted(action);
    m_recoveryBudget_.record(action);

    if (m_recoveryBudget_.isExhausted()) {
        m_actionCallback(RecoveryAction::Critical);
        m_policyEngine_.markExecuted(RecoveryAction::Critical);
    }
}
```

**影響ファイル**: `RuntimePolicyEngine.h`（RecoveryBudget struct + `kBudgetResetAfterUs` 追加）, `RuntimeHealthMonitor.cpp`（tick() に Budget 記録 + リセット追加）, `RuntimeHealthMonitor.h`（`m_lastCriticalExitUs_` 追加）

---

### Phase P1-A: injectBackpressureSignal 統合

#### 4.1 設計方針

`AudioEngine.Retire.cpp` の直接背圧制御を `injectBackpressureSignal()` 経由に置換する。

**変更前** (`AudioEngine.Retire.cpp:157`):

```cpp
convo::publishAtomic(retirePressureAdmissionStrict_, true, std::memory_order_release);
```

**変更後**:

**問題8対応**: tick 間に複数の背圧イベントが来ると最後の値のみ残るため、最大値を保持する。

```cpp
// RuntimeHealthMonitor.h の injectBackpressureSignal を修正
void injectBackpressureSignal(std::size_t fallbackSize, double overflowRate) noexcept {
    // ★ 最大値保持（atomic max update）
    uint64_t current = convo::consumeAtomic(m_maxFallbackSize_,
                                             std::memory_order_acquire);
    while (fallbackSize > current) {
        if (convo::compareExchangeAtomic(m_maxFallbackSize_, current,
                                         static_cast<uint64_t>(fallbackSize),
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
            break;
        current = convo::consumeAtomic(m_maxFallbackSize_,
                                        std::memory_order_acquire);
    }
    // overflowRate も同様に最大値更新
    double rateCurrent = convo::consumeAtomic(m_maxOverflowRate_,
                                               std::memory_order_acquire);
    while (overflowRate > rateCurrent) {
        if (convo::compareExchangeAtomic(m_maxOverflowRate_, rateCurrent,
                                         overflowRate,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
            break;
        rateCurrent = convo::consumeAtomic(m_maxOverflowRate_,
                                            std::memory_order_acquire);
    }
    m_backpressureInjected_ = true;
}
```

```cpp
// tick() 内 — 最大値を回収後リセット
if (m_backpressureInjected_) {
    const auto maxFb = convo::exchangeAtomic(m_maxFallbackSize_, uint64_t{0},
                                              std::memory_order_acq_rel);
    const auto maxOr = convo::exchangeAtomic(m_maxOverflowRate_, 0.0,
                                              std::memory_order_acq_rel);
    if (maxFb > kFallbackQueueCriticalThreshold)
        decision.actions |= toBit(RecoveryAction::Throttle);
    if (maxOr > kOverflowRateCriticalThreshold)
        decision.actions |= toBit(RecoveryAction::Critical);
    m_backpressureInjected_ = false;
}
```

```cpp
// ★ PolicyEngine に背圧評価を委譲
m_healthMonitor.injectBackpressureSignal(
    m_retireFallbackQueue_.size(),
    m_retireFallbackQueue_.overflowRate());
// admissionStrict_ は次回 tick() の PolicyEngine 評価で設定
```

#### 4.2 tick() 内の背圧評価強化

`RuntimeHealthMonitor::tick()` 内の既存の背圧評価コード（L48-56）は以下のように強化:

```cpp
if (m_backpressureInjected_) {
    if (m_injectedFallbackSize_ > kFallbackQueueCriticalThreshold) {
        decision.actions |= toBit(RecoveryAction::Throttle);
    }
    if (m_injectedOverflowRate_ > kOverflowRateCriticalThreshold) {
        decision.actions |= toBit(RecoveryAction::Critical);
    }
    m_backpressureInjected_ = false;
}
```

#### 4.3 Critical 水準の緊急経路（問題6対応）

`injectBackpressureSignal()` は次回 tick() まで PolicyEngine 評価が遅れる。tick が 1秒周期の場合、その間に fallback queue が溢れるリスクがある。

**対策**: Critical 水準（overflowRate > 閾値）の場合は `injectBackpressureSignal()` と同時に、
`retirePressureAdmissionStrict_` の即時設定を残す。ただしこれは**緊急経路限定**とし、通常の背圧評価は PolicyEngine 経由に統一する。

```cpp
// AudioEngine.Retire.cpp: overflow 検出ブロック
const bool overflowCritical = (overflowRate > kOverflowRateCriticalThreshold);
if (overflowCritical) {
    // ★ Critical 水準のみ即時設定（緊急）
    convo::publishAtomic(retirePressureAdmissionStrict_, true,
                         std::memory_order_release);
}
// ★ 通常の背圧は PolicyEngine 経由
m_healthMonitor.injectBackpressureSignal(
    m_retireFallbackQueue_.size(),
    m_retireFallbackQueue_.overflowRate());
```

**設計判断**: `retirePressureAdmissionStrict_` の完全削除は危険。**PolicyEngine 経由を主経路としつつ、Critical 水準のみ緊急直接設定を残す**ハイブリッド方式を採用。

**影響ファイル**: `AudioEngine.Retire.cpp`（緊急経路 + injectBackpressureSignal 併用）, `RuntimeHealthMonitor.cpp`（tick() 背圧評価）

---

### Phase P1-B: LearnerBackpressure FIFO 監視

#### 5.1 設計方針

`checkLearnerBackpressure()` を新規追加。`AudioSegmentBuffer` の FIFO 使用率 + 勾配（slope）を監視する。

**Serena/CodeGraph 調査で確定したファクト**:

- `AudioSegmentBuffer::kCapacity = 3,840,000`（5秒 x 768kHz）
- `AudioSegmentBuffer::getNumAvailableSamples()` → `totalSamples` atomic を acquire 読み取り
- `NoiseShaperLearner` の `segmentBuffer` は L913 で既にログ出力中
- Learner FIFO usage = `getNumAvailableSamples() / kCapacity`

```cpp
// ★ 新規: RuntimeHealthMonitor に追加
void RuntimeHealthMonitor::checkLearnerBackpressure() noexcept {
    if (m_learnerRunningRef == nullptr)
        return;

    const bool learnerActive = convo::consumeAtomic(*m_learnerRunningRef,
                                                     std::memory_order_acquire);
    if (!learnerActive)
        return;

    // FIFO 使用率 = getNumAvailableSamples() / kCapacity
    const int available = m_learnerSegmentBuffer_
        ? m_learnerSegmentBuffer_->getNumAvailableSamples() : 0;
    constexpr int kCapacity = AudioSegmentBuffer::kCapacity;
    const double fifoUsage = static_cast<double>(available)
                           / static_cast<double>(kCapacity);  // 0.0 ~ 1.0

    const uint64_t nowUs = getCurrentTimeUs();

    // ★ 問題7対応: EMA 平滑化（単純差分はノイズに弱い）
    //    α=0.3: 最新値を30%反映、70%は過去EMAを維持
    constexpr double kEmaAlpha = 0.3;
    if (m_fifoEma_ < 0.0) m_fifoEma_ = fifoUsage;  // 初回初期化
    m_fifoEma_ = kEmaAlpha * fifoUsage + (1.0 - kEmaAlpha) * m_fifoEma_;

    // ★ 問題7対応: 時間正規化した勾配（tick 周期依存除去）
    const uint64_t nowUs = getCurrentTimeUs();
    const double elapsedSec = (m_lastFifoTickUs_ > 0)
        ? static_cast<double>(nowUs - m_lastFifoTickUs_) / 1'000'000.0
        : 1.0;
    m_lastFifoTickUs_ = nowUs;
    const double slope = (m_fifoEma_ - m_lastFifoEma_) / std::max(elapsedSec, 0.001);
    m_lastFifoEma_ = m_fifoEma_;

    // ★ 問題E-1: 2段階閾値（warning=85%, error=95%）
    if (fifoUsage > 0.95 && slope >= 0.0) {
        // 95%超 + 増加傾向 → Error (PauseLearner)
        emitOnTransition(m_prevLearnerBackpressureState_, MonitorState::Error,
            HealthEvent::Severity::Error, 5002,
            static_cast<uint64_t>(fifoUsage * 100));
    } else if (fifoUsage > 0.85 && slope >= 0.0) {
        // 85%超 + 増加傾向 → Warning
        if (m_learnerFifoHighSinceUs_ == 0)
            m_learnerFifoHighSinceUs_ = nowUs;
        const uint64_t elapsed = nowUs - m_learnerFifoHighSinceUs_;
        if (elapsed > 30'000'000) {
            emitOnTransition(m_prevLearnerBackpressureState_, MonitorState::Warning,
                HealthEvent::Severity::Warning, 5002,
                static_cast<uint64_t>(fifoUsage * 100));
        }
    } else if (fifoUsage <= 0.80 || slope < -0.01) {
        // 85%以下 または 明確な減少傾向 → 正常復帰
        m_learnerFifoHighSinceUs_ = 0;
        m_prevLearnerBackpressureState_ = MonitorState::Normal;
    }
}
```

#### 5.2 PolicyEngine との連携

`evaluateEvent(PolicySource::LearnerAnomaly)` が既に `Throttle` を返す実装は完了している（`RuntimePolicyEngine.cpp:136`）。`checkLearnerBackpressure()` が `emitOnTransition()` 経由で HealthEvent を発行すれば、既存の PolicyEngine マッピングが動作する。

**影響ファイル**: `RuntimeHealthMonitor.h`, `RuntimeHealthMonitor.cpp`

---

### Phase P0-E: Critical 出口（問題8対応）

#### 5A.1 問題

`updateHealthState()` は MonitorState から HealthState を**一方向に昇格**するのみ。`updateHealthState(const PolicyDecision&)` は毎 tick `ISRHealthState::Healthy` から再計算するため、全 MonitorState が Normal に戻れば自動的に Critical は解除される。

しかし、全 MonitorState が Normal に戻っても、`checkSuppressionDuration()` が `suppressionStartUs_` の残存により発火し続ける可能性がある。

#### 5A.2 Critical 自動解除条件

```text
全 MonitorState が Normal
AND
suppressionStartUs_ == 0 (suppression 解除済み)
AND
直近の閉ループ昇格から 60秒経過
```

上記を満たした場合、`updateHealthState()` は `ISRHealthState::Healthy` を publish する（既存コードで対応済み — MonitorState ベース評価のため）。

追加で必要なのは:

```cpp
// RuntimeHealthMonitor::tick() 内 — Critical 長期滞留時にも Budget リセット
if (m_healthState_ == ISRHealthState::Critical) {
    const uint64_t criticalDuration = convo::getCurrentTimeUs() - m_criticalEnteredUs_;
    if (criticalDuration > kBudgetResetAfterUs) {
        // Critical が30分以上継続しても Budget をリセット
        // （MonitorState が Normal なら HealthState が自動復帰するが、
        //   MonitorState が Warning 継続で Degraded にしか戻れない場合の安全策）
        m_recoveryBudget_.reset();
    }
}
```

**問題7対応**: `CriticalExitCondition` を診断イベント発行に使用する（監視専用ではない）。

```cpp
// ★ RuntimeHealthMonitor に追加
struct CriticalExitCondition {
    bool allMonitorsNormal{false};    // 全 MonitorState が Normal
    bool suppressionInactive{false};  // suppressionStartUs_ == 0
    bool noRecoveryActionRunning{false}; // 閉ループが Idle
    bool stableDuration{false};       // 安定状態が 60秒継続

    [[nodiscard]] bool canExit() const noexcept {
        return allMonitorsNormal && suppressionInactive
            && noRecoveryActionRunning && stableDuration;
    }
};
```

```cpp
// tick() 内 — Critical 出口評価
if (m_healthState_ == ISRHealthState::Critical) {
    CriticalExitCondition exitCond;
    exitCond.allMonitorsNormal =
        (m_prevRetireState == MonitorState::Normal)
        && (m_prevPublicationState == MonitorState::Normal)
        && (m_prevReaderSlotState == MonitorState::Normal)
        && (m_prevOverflowRateState == MonitorState::Normal)
        && (m_prevRetireAgeState == MonitorState::Normal)
        // ★ 問題F-1: 追加監視軸
        && (m_prevLearnerBackpressureState_ == MonitorState::Normal)
        && (m_prevConfigDivergenceState_ == MonitorState::Normal);
    exitCond.suppressionInactive =
        (m_suppressionStartRef_ == nullptr
         || convo::consumeAtomic(*m_suppressionStartRef_,
                                 std::memory_order_acquire) == 0);
    exitCond.noRecoveryActionRunning =
        (m_policyEngine_.getLastExecutedAction()
         <= RecoveryAction::Observe);
    if (exitCond.allMonitorsNormal) {
        if (m_criticalStableSinceUs_ == 0)
            m_criticalStableSinceUs_ = getCurrentTimeUs();
        exitCond.stableDuration =
            (getCurrentTimeUs() - m_criticalStableSinceUs_) > 60'000'000;
    } else {
        m_criticalStableSinceUs_ = 0;
    }

    if (exitCond.canExit()) {
        m_criticalStableSinceUs_ = 0;
        diagLog("[RECOVERY] Critical exit conditions met, "
            "awaiting MonitorState recovery");
        // HealthEvent 発行（診断用）
        if (m_callback) {
            HealthEvent ev{getCurrentTimeUs(),
                HealthEvent::Severity::Info, 6002, 0, 0};
            m_callback(ev);
        }
        // updateHealthState() は MonitorState ベースで自動的に
        // Healthy を publish するため、明示的な HealthState 操作は不要
    }

    // ★ Budget リセット（Critical 解除とは独立）
    if (convo::getCurrentTimeUs() - m_criticalEnteredUs_
        >= kBudgetResetAfterUs) {
        m_recoveryBudget_.reset();
    }
}
```

**設計判断**: `updateHealthState()` は MonitorState ベースで自動復帰する。CriticalExitCondition は診断・監視用であり、HealthState の直接操作は行わない。Budget リセットは CriticalExitCondition とは独立して動作する。

**影響ファイル**: `RuntimeHealthMonitor.h`（`CriticalExitCondition` struct, `m_criticalEnteredUs_`, `m_criticalStableSinceUs_` 追加）, `RuntimeHealthMonitor.cpp`（tick() に Critical 出口評価追加）

---

### Phase P2: 将来拡張項目

#### 6.1 ForcePublicationRecovery

`executeRecoveryAction()` に `retryDeferredPublication()` と `retryDeferredStructuralRebuild()` を追加。

#### 6.2 AudioQualityFingerprint

現在の `computeRuntimeRecoveryScore()` は `const` メソッドで未使用。これを `RecoveryOutcome` 判定に活用する。

#### 6.3 ClearSuppression

`RecoveryAction::ClearSuppression` を新設（または Throttle の解除フラグとして実装）。

---

## 2. ファイル影響マップ（修正版）

| Phase | ファイル | 変更種別 | 変更内容 |
| --- | --- | --- | --- |
| P0-B | `AudioEngine.Timer.cpp` | 修正 | `RecoveryAction::Restore` case に World Rollback + Learner Rollback 実装 |
| P1-A | `AudioEngine.Retire.cpp` | 修正 | `retirePressureAdmissionStrict_` 直接書き込みを `injectBackpressureSignal()` に置換 |
| P1-A | `RuntimeHealthMonitor.cpp` | 強化 | tick() 内の背圧評価ロジック強化 |
| P0-A | `RuntimePolicyEngine.h` | 修正 | `RecoveryOutcome` enum 再定義（Improving/Recovered/Stalled/Worsening） |
| P0-A | `RuntimePolicyEngine.h` | 修正 | `TrendSnapshot` struct, `RecoveryBudget` struct 追加 |
| P0-A | `RuntimePolicyEngine.h` | 修正 | `m_verificationStates_`, `m_recoveryBudget_` 追加 |
| P0-A | `RuntimePolicyEngine.cpp` | 修正 | 傾向評価ロジック実装 |
| P0-A | `RuntimeHealthMonitor.h` | 修正 | `computeTrend()`, `takeSnapshot()`, `TrendSnapshot` 型追加 |
| P0-A | `RuntimeHealthMonitor.cpp` | 修正 | `tick()` に閉ループ制御追加、`computeTrend()` 実装 |
| P0-D | `RuntimeHealthMonitor.cpp` | 修正 | tick() に Budget 記録追加 |
| P0-C | `AudioEngine.h` | 修正 | `m_probeRequested_`, `m_suppressionActive_`, `m_lastProbeUs_` 追加 |
| P0-C | `AudioEngine.Threading.cpp` | 修正 | `shouldRejectRebuildAdmissionForPressure()` 先頭に Probe チェック追加 |
| P0-C | `RuntimeHealthMonitor.cpp` | 修正 | tick() に Probe 設定ロジック追加 |
| P1-B | `RuntimeHealthMonitor.h` | 修正 | `checkLearnerBackpressure()`, `m_lastFifoUsage_`, `m_learnerFifoHighSinceUs_` 追加 |
| P1-B | `RuntimeHealthMonitor.cpp` | 修正 | `checkLearnerBackpressure()` 実装 + slope 計算、tick() に呼び出し追加 |

### 総変更ファイル数: 7ファイル

| ファイル | Phase | 行数見積 |
| --- | --- | --- |
| `AudioEngine.Timer.cpp` | P0-B | +25行 |
| `AudioEngine.Retire.cpp` | P1-A | -8行（削除）+5行（追加） |
| `AudioEngine.Threading.cpp` | P0-C | +3行 |
| `AudioEngine.h` | P0-C | +5行 |
| `RuntimePolicyEngine.h` | P0-A, P0-D | +60行 |
| `RuntimePolicyEngine.cpp` | P0-A | +70行 |
| `RuntimeHealthMonitor.h` | P0-A, P1-B | +45行 |
| `RuntimeHealthMonitor.cpp` | P0-A, P0-C, P0-D, P1-A, P1-B | +160行 |

**合計見積: 約370行追加 / 8行削除**

---

## 3. 実装順序と依存関係（修正版: Practical Stable 優先順位）

```
Step 1: P0-B (Restore→Rollback)  — 単一ファイル、他に依存せず、最大の障害回復効果
Step 2: P1-A (背圧統合)           — PolicyEngine 権限一元化、admissionStrict_ の直接書き込み削除
Step 3: P0-A (RecoveryVerification) — 閉ループ制御基盤。Improving/Stalled/Worsening 傾向評価
Step 4: P0-D (RecoveryBudget)      — 永久ループ防止。P0-A 完了後でないと意味がない
Step 5: P1-B (Learner FIFO)        — P0-A と独立。slope ベース FIFO 監視
Step 6: P0-C (Suppression Probe)   — 最後。Probe 方式は他 Phase の閉ループが整ってから効果的
```

### 推奨実装順

| Step | Phase | 推定工数 | 備考 |
| --- | --- | --- | --- |
| 1 | P0-A (RecoveryVerification基盤) | 3-5h | enum/struct 追加 + tick() 組み込み |
| 2 | P0-C (Suppression Escape) | 1-2h | P0-A と独立、並行可能 |
| 3 | P0-B (Restore Rollback) | 2-3h | P0-A の閉ループが Restore へ昇格するようになった後に実装 |
| 4 | P1-A (背圧統合) | 1-2h | Retire.cpp の直接書き込み削除 |
| 5 | P1-B (Learner FIFO) | 2-3h | AudioSegmentBuffer API の確認が必要 |
| 6 | P2 (将来拡張) | - | P0/P1 完了後 |

---

## 4. 検証方法

### 4.1 単体テスト観点

| Phase | 検証項目 | 確認方法 |
| --- | --- | --- |
| P0-A | Throttle 実行→5秒後 pendingRetire 減少で Success | ログ `[RECOVERY] outcome=Success` 確認 |
| P0-A | Throttle 実行→5秒後 pendingRetire 不変で Recover へ昇格 | ログ `[RECOVERY] escalate Throttle→Recover` 確認 |
| P0-B | Restore 実行→lastHealthyWorldId_ が使われる | `publishIdleWorldOnly` 呼び出しログ確認 |
| P0-C | Suppression 30秒経過→一時解除 | ログ `[RECOVERY] Suppression escape` 確認 |
| P1-A | Retire.cpp からの admissionStrict_ 直接書き込みが0件 | grep 確認 |
| P1-B | Learner FIFO >90% 30秒継続→Warning HealthEvent | ログ `[HEALTH] eventCode=5002` 確認 |

### 4.2 実ログ再現テスト

実障害ログの「Retire stall → tryReclaim → 失敗 → 永久 Suppression」を再現するシナリオ:

1. `retireDepth` を高く維持するテスト用フックを有効化
2. Retire Stall 検出 → Throttle 発火 → 5秒後確認→Recover
3. Recover（tryReclaim + drain）→ 10秒後確認→Restore
4. Restore（World Rollback）→ 15秒後確認→Safe
5. Safe（admissionStrict 解除 + Learner停止）
6. 最終的に音が正常に戻ることを確認

---

## 5. リスクと注意点

### 5.1 閉ループ制御による Cooldown との競合

`RuntimePolicyEngine` の Cooldown 制御（`canExecute()` / `markExecuted()`）と閉ループ昇格が競合する可能性がある。`verifyAndEscalate()` では Cooldown を無視して昇格させる（Cooldown は同一 Action の連続実行防止用であり、昇格は別 Action のため競合しない）。

### 5.2 Restore 失敗時の Safe 移行

World Rollback が失敗した場合（healthyWorldId == 0 など）、閉ループ制御が自動的に Safe へ昇格する。`RecoveryAction::Safe` は既に `stopNoiseShaperLearning()` + `admissionStrict_ = false` を実装済み。

### 5.3 AudioSegmentBuffer API 確認

`getLearnerFifoUsage()` の実装には `AudioSegmentBuffer` の容量取得 API が必要。`getNumAvailableSamples()` / `getCapacity()` の有無を確認する必要がある（Phase 9.4 の設計注記参照）。
