# RecoveryAction 実装状況 確定検証 — 設計 vs コード最終突合

> **日付**: 2026-06-15
> **検証方法**: 最新版 `src/**` 全ファイルに対する grep 確定。設計書 (recovery_system_plan.md v7.1) 記載の全 RecoveryAction / 監視項目と実コードの突合
> **注意**: 以下の「設計上の記載」は最新版 recovery_system_plan.md の Phase 9（v7.1最終版）に基づく

---

## P0: 最優先確認3項目

### ① RecoveryOutcome — **未実装（定義のみ）**

```
$ grep -r "RecoveryOutcome" src/**/*.cpp
→ 該当なし

$ grep -r "RecoveryOutcome" src/**
→ src/audioengine/RuntimePolicyEngine.h:37  コメント中の文字列として
→ src/audioengine/RuntimePolicyEngine.h:53  コメント // [work37 Phase 9.45] RecoveryOutcome
→ src/audioengine/RuntimePolicyEngine.h:54  enum class RecoveryOutcome : uint8_t { ... };
```

**設計**: Phase 9.45 — `RecoveryOutcome` enum (Success/NoEffect/Failed/Unsafe) + 閉ループ制御
**コード**: enum 定義のみ。`RuntimeHealthMonitor.cpp` の `tick()` / `executeRecoveryAction()` / 全コールチェーンで**0回参照**。
**結論**: ❌ **未実装。Action→効果確認→昇格の閉ループが完全に欠落。**

---

### ② executeRecoveryAction() 結果検証 — **未実装**

実装 (`AudioEngine.Timer.cpp:659-702`):

```cpp
void AudioEngine::executeRecoveryAction(convo::RecoveryAction action) noexcept {
    switch (action) {
        case ...: // Action 実行
            break;
        // ...
    }
    diagLog("[RECOVERY] execute action=" + ...);  // ログのみ
    // ★ 戻り値なし。Action の効果を検証するコードなし
}
```

`executeRecoveryAction()` は `void` 戻り値。Action 実行後、その Action が効果を発揮したか（例: pendingRetireCount が減少したか、suppression が解除されたか）を確認するコードは一切存在しない。

**結論**: ❌ **未実装。RecoveryAction は発火するが「効果があったか」を確認しない。**

---

### ③ Restore → World Rollback — **未実装（記録のみ）**

**記録側 ✅** (`AudioEngine.CtorDtor.cpp:203-216`):

```cpp
void AudioEngine::notifyHealthyPublication(uint64_t worldId) noexcept {
    convo::publishAtomic(lastHealthyWorldId_, worldId, ...);
    // Learner state も保存
    lastKnownGoodNoiseShaper_.state = current;
    lastKnownGoodNoiseShaper_.isValid = true;
}
```

**回復側 ❌** (`AudioEngine.Timer.cpp:679-683`):

```cpp
case convo::RecoveryAction::Restore:
    tryReclaimResources();
    drainDeferredRetireQueues(false);
    // Rollback 基盤（ISRRetireRuntimeEx）の設定は呼び出し元が行う
    break;
```

```
$ grep -r "getLastHealthyWorldId\|lastHealthyWorldId_" src/**/*.cpp
→ src/audioengine/AudioEngine.CtorDtor.cpp:205  // publishAtomic 書き込みのみ
→ 読み取り箇所なし
```

`getLastHealthyWorldId()` は `.h` でインライン定義されているが、**どの .cpp からも呼ばれていない**。従って World Rollback は行われていない。

**結論**: ❌ **未実装。健全部の World ID / Learner state は保存されるが、回復時に使われず、切戻しが発生しない。**

---

## P1: 重要確認3項目

### ④ injectBackpressureSignal — **未使用（定義のみ）**

```
$ grep -r "injectBackpressureSignal" src/**
→ src/audioengine/RuntimeHealthMonitor.h:146  // 定義のみ
```

定義:

```cpp
void injectBackpressureSignal(std::size_t fallbackSize, double overflowRate) noexcept {
    m_injectedFallbackSize_ = fallbackSize;
    m_injectedOverflowRate_ = overflowRate;
    m_backpressureInjected_ = true;
}
```

しかし `AudioEngine.Retire.cpp` からこの関数を呼び出していない。

**結論**: ❌ **設計上は PolicyEngine 統合が計画されているが、実コードでは未呼び出し。**

---

### ⑤ retirePressureAdmissionStrict_ 直接書き込み — **残存**

```
$ grep -n "retirePressureAdmissionStrict_" src/audioengine/AudioEngine.Retire.cpp
→ L157: convo::publishAtomic(retirePressureAdmissionStrict_, true, ...);
→ L291: convo::publishAtomic(retirePressureAdmissionStrict_, severe, ...);
```

両方とも `drainDeferredRetireQueues()` / `applyRetirePressurePolicyNoRt()` 内の**直接書き込み**。PolicyEngine を経由していない。

同様に `AudioEngine.Timer.cpp` の `onHealthEvent()` 内でも直接書き込みが残っている（L558, L592）。

**結論**: ❌ **3経路（Retire.cpp の背圧評価, Timer.cpp の onHealthEvent 内の ReaderSlotExhaustion, 同 RetireStall）から直接書き込みが残存。二重権限は解消されていない。**

---

### ⑥ LearnerBackpressure FIFO 監視 — **未実装**

```
$ grep -r "checkLearnerBackpressure\|learnerFifo\|bufferedSamples\|segmentBuffer.*usage\|FIFO.*learner" src/**
→ 該当なし
```

設計上の記述（Phase 9.4）:
> `learnerFifoUsage = learnerSegmentBuffer.usagePercent()` を監視し、
> FIFO > 90% が 30秒継続 → PauseLearner
> FIFO > 95% が 60秒継続 → Critical + PauseLearner

**現状の Learner 保護は tick() 内の以下のみ** (`RuntimeHealthMonitor.cpp:63-69`):

```cpp
const uint64_t stallDur = getRetireStallDurationUs();
const bool learnerActive = (m_learnerRunningRef != nullptr) && ...;
if (stallDur > 10'000'000 && learnerActive) {
    decision.actions |= toBit(RecoveryAction::Throttle);
}
```

- `LearnerBackpressure` は `HealthCause` として定義済み ✅
- `LearnerAnomaly` は `PolicySource` として定義済み ✅
- `PolicySource::LearnerAnomaly → Throttle` のマッピングは PolicyEngine に実装済み ✅
- しかし**FIFO 使用率を実際に読み取る `checkLearnerBackpressure()` が未実装** ❌

**結論**: ⚠️ **部分的に準備（enum/PolicySource）は整っているが、FIFO 使用率監視の実装は未着手。現在は stallDur > 10s の単一条件のみ。**

---

## P2: 参考確認3項目

### ⑦ AudioQualityFingerprint — **未実装**

```
$ grep -ri "audioquality\|fingerprint\|AudioQuality" src/**/*.h
→ 該当なし
```

設計 Phase 9.40 に記載あり。コード上に一切存在しない。

### ⑧ ForcePublicationRecovery — **未実装**

```
$ grep -ri "ForcePublication\|forcePublication\|retryDeferredPublication\|retryDeferredStructural" src/**
→ 該当なし
```

設計 Phase 9.5 に記載あり。コード上に存在しない。現在の `RecoveryAction::Recover` は `tryReclaimResources()` + `drainDeferredRetireQueues()` + `clearDeferredForShutdown()` を実行するが、retryPublication や retryRebuild は含まない。

### ⑨ ClearSuppression — **未実装**

```
$ grep -ri "ClearSuppression\|clearSuppression" src/**
→ 該当なし
```

設計 Phase v7.0/v7.1 に記載あり。コード上に存在しない。現在 suppression を解除できるのは `RecoveryAction::Safe`（`admissionStrict_ = false`）のみ。

---

## 総合評価テーブル

| # | 項目 | 優先度 | 設計記載 | コード状態 | 実装度 |
| --- | --- | --- | --- | --- | --- |
| ① | RecoveryOutcome 実使用 | **P0** | Phase 9.45 | 定義のみ、全 .cpp で未使用 | **0%** |
| ② | executeRecoveryAction 結果検証 | **P0** | Phase 9.45 | void 戻り値、効果確認なし | **0%** |
| ③ | Restore → World Rollback | **P0** | Phase 9.16/9.44 | 記録のみ。Rollback 未配線 | **40%** |
| ④ | injectBackpressureSignal 呼び出し | **P1** | Phase 4.4.1 | 定義のみ、未呼び出し | **0%** |
| ⑤ | admissionStrict_ 直接書き込み残存 | **P1** | Phase 4.4 | Retire.cpp (L157,L291) + Timer.cpp (L558,L592) で残存 | **50%** |
| ⑥ | LearnerBackpressure FIFO監視 | **P1** | Phase 9.4 | PolicySource/HealthCause は定義済み。check*() 未実装 | **30%** |
| ⑦ | AudioQualityFingerprint | **P2** | Phase 9.40 | コード上に存在せず | **0%** |
| ⑧ | ForcePublicationRecovery | **P2** | Phase 9.5 | コード上に存在せず | **0%** |
| ⑨ | ClearSuppression | **P2** | v7.0/v7.1 | コード上に存在せず | **0%** |

---

## 最終結論

### 設計がコードより進んでいる項目（6件）

① RecoveryOutcome, ④ injectBackpressureSignal, ⑥ LearnerBackpressure FIFO,
⑦ AudioQualityFingerprint, ⑧ ForcePublicationRecovery, ⑨ ClearSuppression

→ 設計書には存在するが、コード実装が追い付いていない。

### 設計とコードが両方とも未完了（3件）

② executeRecoveryAction 結果検証, ③ Restore → World Rollback, ⑤ admissionStrict_ 直接書き込み

→ 設計にもコードにも「完全版」が存在しないギャップ領域。

### あなたのご認識との一致
>
> **「RecoveryActionを発火できるか」ではなく「RecoveryActionが失敗したことを検知して次段階へ進めるか」**

✅ **完全に正しい。現在のコードは Action を発火できるが、失敗検知→次段階昇格ができない。**

> **「Last Known Good Worldへ実際に戻せるか」**

✅ **完全に正しい。保存はされているが、戻すコードが存在しない。**

**両者とも、あなたが実際に遭遇した「Retire Stall → tryReclaim → 失敗 → 永久 Suppression」の再発防止に直結する。**

---

## 補足：実障害ログとの対応分析

### 現状の Recovery System は実質 Open Loop

```
障害検出 → RecoveryAction 発火 → (効果確認なし) → 終了
```

実ログ「Retire stall → tryReclaim → 失敗 → 永久 Suppression」は、tryReclaim の成否をシステムが認識しないことに起因する。`RecoveryOutcome` が 0% 使用であるため、Recover の効果確認が行われず、失敗しても次の Action (Restore/Safe) へ進まない。

### 実効 Recovery Ladder は設計より1段少ない

設計上の Ladder:

```
Observe → Throttle → Recover → Restore → Safe → Critical (6段階)
```

Restore が空実装（tryReclaim + drain のみ、Rollback なし）であるため、実質的な Ladder:

```
Observe → Throttle → Recover → Safe → Critical (5段階)
```

Restore は Recover の強化版でしかなく、Last Known Good World への切戻しが行われない。

### 現時点の Recovery System の位置づけ

```
「障害を検出してログを出すシステム」—— ここまで達成
「障害から復旧するシステム」———— 未到達
```

RecoveryVerification と Rollback の2つが実装されて初めて後者に到達する。

### 推奨優先順位（実障害再発防止の観点）

| 順位 | 項目 | コード状態 | 効果 |
| --- | --- | --- | --- |
| **P0-A** | RecoveryVerification (`RecoveryOutcome` 実使用) | 定義のみ、0% | Throttle→効果確認→Recover→... の閉ループ。実障害を直接潰す |
| **P0-B** | Restore→LastHealthyWorld Rollback | 記録のみ、40% | Convolver IR/EQ/Oversampling 未適用を戻せる |
| **P0-C** | Suppression Escape（一定時間後一時解除と再試行） | 未実装、0% | intent 23/24/25... の永久 suppression 連鎖を防止 |
| **P1** | injectBackpressureSignal 統合 + LearnerBackpressure | 定義のみ/30% | 背圧二重権限の解消。実障害への直接効果は P0 群より小 |
