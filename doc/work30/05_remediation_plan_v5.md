# ISR Bridge Runtime Practical Stable 改修計画 v5（最終版）

**作成日**: 2026-06-11
**v4→v5 差分**:

- P1-A: `atomic<bool>` → 通常 `bool`（thread confinement 確認済み）
- P1-C: SPSC push 失敗時の drop telemetry 追加 → Practical-6 へ昇格
- P-2: 固定30秒タイムアウト → 期待fade長ベースの動的タイムアウト
- P3-B: `fetchSub` → `compare_exchange_weak` CAS ループ
- Practical-6: Crossfade Event Drop Telemetry（新規）
**推定工数**: P0=2h / P1=5日 / P2=2.5日 / P3=1週間

**コード調査で確認した事実**（全ツール使用）:

- RCUReaderGuard の `enter()/exit()` は同一スレッドのスタック RAII パターン → `bool` で安全
- `SPSCRingBuffer::push()` / `LockFreeRingBuffer::push()` は `bool` 返却
- Crossfade duration 設定: 最大 `m_irFadeTimeSec` = 80ms（最小 10ms → 最大 80ms）
- `DeferredDeletionQueue::getMaxRetireAgeUs()` は既存 API だが呼び出し元なし（dead API）

---

## 優先度定義（v3 と同様）

---

## Phase 0: 即時修正（P0、総工数〜2h）

### P0-A: IEpochProvider に pendingRetireCount/drainAll 追加（〜1h）◎

v4 計画を踏襲。`IRetireProvider` に純粋仮想関数追加。

### P0-B: EQProcessor Fallback 削除（〜1h）◎

v4 計画を踏襲。

---

## Phase 1: 監視・制御閉ループ（P1、総工数〜5日）

### P1-A: RCUReader per-enter Fail-Closed 化 — non-atomic bool（〜4h）△→◎

**v4→v5 変更点**:

- `std::atomic<bool>` → 通常 `bool`（thread confinement 確認済み）
- 名前: `lastEnterSucceeded_` → `rootEnterSucceeded_`（ネストRCUの意味を明確化）

**根拠（コード調査結果）**:

- `RCUReader::enter()` / `RCUReaderGuard::valid()` / `RCUReader::exit()` は
  **全て同一スレッド**から RAII スタックパターンで呼ばれる
- `RCUReaderGuard` コンストラクタで `reader->enter()`、デストラクタで `reader->exit()`
- したがって `enter()` で設定したフラグは `exit()` まで他スレッドから触られない
- `std::atomic` にする必要はなく、通常 `bool` で十分

```cpp
class RCUReader {
public:
    void enter() noexcept
    {
        const uint32_t previousDepth = convo::fetchAddAtomic(...);
        if (previousDepth > 0)
        {
            // ★ ネスト: rootEnterSucceeded_ は変更しない（最外層の結果を維持）
            return;
        }

        // ... CASロジック ...

        const int tid = acquireThreadSlot();
        if (tid >= 0)
        {
            epochProvider->enterReader(tid);
            rootEnterSucceeded_ = true;  // ★ non-atomic: thread confined
        }
        else
        {
            rootEnterSucceeded_ = false;
            // ... 後片付け ...
        }
    }

    /** ★ 最外層 enter() が成功したか（ネスト時は最外層の結果を返す） */
    [[nodiscard]] bool rootEnterSucceeded() const noexcept
    {
        return rootEnterSucceeded_;
    }

private:
    // ...
    bool rootEnterSucceeded_ = false;  // ★ non-atomic: thread confined
};
```

**ネスト動作**:

```
enter() → 成功 → rootEnterSucceeded_ = true
  enter() → ネスト → 何もしない
  exit() → ネスト → 何もしない
exit() → rootEnterSucceeded_ = false（次の enter に備えリセット）

enter() → 失敗 → rootEnterSucceeded_ = false
  enter() → ネスト → 何もしない（rootEnterSucceeded_ は false のまま）
exit() → ネスト → 何もしない
valid() → false（正しく Fail-Closed）
```

---

### P1-B: HealthMonitor → Admission 閉ループ（〜2日）○

v4 計画を踏襲。`ISRHealthState` 導入。

---

### P1-C: Crossfade 完了イベント SPSC キュー化（〜1日）△→◎

**v4→v5 変更点**:

- SPSC push 失敗時の drop telemetry を追加（Practical-6 と統合）
- `completedFadeQueue_.push(ev)` の戻り値をチェック

```cpp
// CrossfadeRuntime.h
#include "core/CommandBuffer.h"  // SPSCRingBuffer

struct CompletedFadeEvent {
    CrossfadeId id;
    uint64_t    completedTimestampUs;
};

class CrossfadeRuntime {
public:
    void notifyFadeComplete(CrossfadeId id) noexcept
    {
        CompletedFadeEvent ev{id, getCurrentTimeUs()};
        // ★ SPSC push: 戻り値をチェック
        if (!completedFadeQueue_.push(ev))
        {
            // ★ Queue full → drop telemetry
            convo::fetchAddAtomic(crossfadeEventDropCount_, uint64_t{1},
                std::memory_order_release);
        }
    }

    /** ★ 超過イベント破棄数を取得 */
    [[nodiscard]] uint64_t crossfadeEventDropCount() const noexcept
    {
        return convo::consumeAtomic(crossfadeEventDropCount_, std::memory_order_acquire);
    }

    // ... 残りは v4 と同じ ...

private:
    SPSCRingBuffer<CompletedFadeEvent, 32> completedFadeQueue_;
    std::atomic<uint64_t> crossfadeEventDropCount_{0};  // ★ P1-C/Practical-6
    std::atomic<uint64_t> fadeStartTimestampUs_{0};
};
```

---

## Phase 2: 防御的改善（P2、総工数〜2.5日）

### P2-A: Deprecated API 完全除去（〜2h）◎

### P2-B: Shutdown VerifyDrained + BlockingReason（〜4h）○

### P2-C: HealthState 単一 enum 共有（〜1日）○

---

## Phase 3: 中長期改善（P3、総工数〜1週間）

### P3-A: HealthMonitor ↔ RetirePressure 統合（〜3日）○

### P3-B: WorldLifecycleAudit — CAS 式二重 retire 防御（〜2日）○

**v4→v5 変更点**: `fetchSub` → `compare_exchange_weak` CAS ループ

```cpp
void onWorldRetired(uint64_t worldId, uint64_t epoch) noexcept
{
    // ★ P3-B: CAS ループで安全に減算
    uint64_t current = convo::consumeAtomic(activeWorldCount_, std::memory_order_acquire);
    while (true)
    {
        if (current == 0)
        {
            jassertfalse;  // ★ 二重 retire 検出
            return;        //   カウンタ不変維持
        }

        if (convo::compareExchangeWeakAtomic(activeWorldCount_,
                current, current - 1,
                std::memory_order_acq_rel,
                std::memory_order_acquire))
            break;
        // ★ CAS 失敗: 他スレッドが更新 → latest で再試行
    }
    convo::fetchAddAtomic(retiredCount_, uint64_t{1}, std::memory_order_release);
}
```

**CAS の根拠**:

- 現在は単一 retire 経路だが、将来のマルチスレッド化に備える
- `compare_exchange_weak` は SPSC ではないが、retire は NonRT 限定かつ低頻度
- CAS 失敗時の retry も NonRT 側であり、RT 制約外

---

## Practical-1: Retire Queue High Watermark（P2、〜2h）◎

## Practical-2: Crossfade Timeout — 期待fade長ベース（P1、〜4h）△→◎

**v4→v5 変更点**: 固定30秒 → `expectedFadeDurationUs * 4` の動的タイムアウト

```cpp
class CrossfadeRuntime {
public:
    void start(double fadeTimeSec, double sampleRate) noexcept
    {
        // ... 既存 ...
        convo::publishAtomic(fadeStartTimestampUs_, getCurrentTimeUs(), std::memory_order_release);
        // ★ Practical-2: 期待fade長を記録
        uint64_t expectedUs = static_cast<uint64_t>(fadeTimeSec * 1'000'000);
        convo::publishAtomic(expectedFadeDurationUs_, std::max(expectedUs, uint64_t{1'000'000}),
            std::memory_order_release);
    }

    void complete() noexcept
    {
        // ... 既存 ...
        convo::publishAtomic(fadeStartTimestampUs_, 0, std::memory_order_release);
        convo::publishAtomic(expectedFadeDurationUs_, 0, std::memory_order_release);
    }

    /** ★ 動的タイムアウト値（期待fade長×4、最低30秒） */
    [[nodiscard]] uint64_t timeoutUs() const noexcept
    {
        uint64_t expected = convo::consumeAtomic(expectedFadeDurationUs_, std::memory_order_acquire);
        return std::max(expected * 4, uint64_t{30'000'000});  // max(4x, 30s)
    }

    /** ★ 開始からの経過時間 */
    [[nodiscard]] uint64_t getFadeAgeUs() const noexcept
    {
        uint64_t start = convo::consumeAtomic(fadeStartTimestampUs_, std::memory_order_acquire);
        if (start == 0) return 0;
        return getCurrentTimeUs() - start;
    }

private:
    std::atomic<uint64_t> fadeStartTimestampUs_{0};
    std::atomic<uint64_t> expectedFadeDurationUs_{0};  // ★ 期待fade長
};

// HealthMonitor 側
void RuntimeHealthMonitor::checkCrossfadeTimeout() noexcept {
    if (!m_crossfadeRuntime) return;
    if (!m_crossfadeRuntime->isPending()) return;

    uint64_t ageUs = m_crossfadeRuntime->getFadeAgeUs();
    uint64_t timeout = m_crossfadeRuntime->timeoutUs();  // ★ 動的タイムアウト

    if (ageUs > timeout) {
        emitOnTransition(m_prevCrossfadeState, MonitorState::Error,
            HealthEvent::Severity::Error, EVENT_CROSSFADE_TIMEOUT,
            ageUs / 1000);
    }
}
```

**動作例**:

| fadeTimeSec | expectedFadeDurationUs | timeoutUs | 実効タイムアウト |
|-------------|----------------------|-----------|----------------|
| 0.080 (IR) | 80,000 | max(320,000, 30,000,000) = **30秒** |
| 0.010 (directHead) | 10,000 | max(40,000, 30,000,000) = **30秒** |
| 5.0 (超長) | 5,000,000 | max(20,000,000, 30,000,000) = **30秒** |
| 10.0 (超々長) | 10,000,000 | max(40,000,000, 30,000,000) = **40秒** |

→ 通常運転では`fadeTimeSec * 4` ≪ 30秒なので実質30秒固定相当。
→ 超長時間 crossfade 時のみ自動延長される。

## Practical-3: Shutdown Audit Event（P2、〜2h）◎

## Practical-4: Reader Slot Usage Telemetry（P1、〜4h）◎

v4 計画を踏襲。

## Practical-5: Retire Reclaim Latency（P2、〜4h）◎

v4 計画を踏襲。既存 `DeferredDeletionQueue::getMaxRetireAgeUs()`、
`AudioEngine::reclaimLatency_` を HealthMonitor へ接続。

## Practical-6: Crossfade Event Drop Telemetry（P2、〜2h）【新規】

**目的**: AudioThread → Timer の crossfade 完了通知経路における
イベント欠落を監視する。P1-C の SPSC overflow 時に drop を検出。

**既存資産**: P1-C で `crossfadeEventDropCount_` を追加済み。

**HealthMonitor 接続**:

```cpp
// RuntimeHealthMonitor.h
static constexpr uint32_t EVENT_CROSSFADE_EVENT_DROP = 4002;

void setCrossfadeEventDropRef(const std::atomic<uint64_t>* ref) noexcept {
    m_crossfadeEventDropRef = ref;
}

// tick() 内
void checkCrossfadeEventDrop() noexcept {
    if (!m_crossfadeEventDropRef) return;
    uint64_t drops = convo::consumeAtomic(*m_crossfadeEventDropRef, std::memory_order_acquire);
    if (drops > 0) {
        emitOnTransition(m_prevCrossfadeDropState, MonitorState::Warning,
            HealthEvent::Severity::Warning, EVENT_CROSSFADE_EVENT_DROP, drops);
    }
}
```

**診断価値**:

- SPSC サイズ 32 に対して crossfade 完了は高々 1〜2 なので、通常は drop 発生しない
- drop 発生 = SPSC の処理能力不足（Timer がイベントを消費しきれていない）
- → Timer interval の調整または SPSC サイズ拡大の判断材料

---

## マイルストーン

| M | 内容 | フェーズ | 時期 |
|---|------|---------|------|
| M1 | IEpochProvider 拡張 + Fallback削除 | P0-A, P0-B | Day 1 AM |
| M2 | RCUReader per-enter Fail-Closed | P1-A | Day 1 PM |
| M3 | Crossfade SPSC + Drop監視 + Timeout改善 | P1-C, P-2, P-6 | Day 2 |
| M4 | 閉ループ完成 (HealthState→Admission + ReaderSlot + RetireAge) | P1-B, P2-C, P-4, P-5 | Day 3-4 |
| M5 | Shutdown強化 + Deprecated整理 | P2-A, P2-B, P-3 | Day 4-5 |
| M6 | WorldLifecycleAudit + RetireHWM | P3-B, P-1 | Day 5 |
| M7 | HealthMonitor↔RetirePressure統合 | P3-A | Week 2-3 |

---

## v2→v3→v4→v5 改善サマリ

| 項目 | v2 | v3 | v4 | v5 | v5 方針 |
|------|-----|-----|-----|-----|---------|
| P0-A | △ | ○ | ◎ | ◎ | IRetireProvider拡張 |
| P1-A | △ 危険 | △ m_valid永続 | △ atomic<bool> | ◎ | non-atomic bool + per-enter |
| P1-C | △ bool | △ atomic<Id> | ○ SPSC | ◎ | SPSC + drop telemetry |
| P-2 | — | △ completeTs | △ 固定30s | ◎ | 期待fade長×4 動的 |
| P3-B | △ vector | ○ RingBuffer | ○ fetchSub | ◎ | CASループ化 |
| P-4 | — | — | 新規 | ◎ | Reader Slot監視 |
| P-5 | — | — | 新規 | ◎ | Retire Age監視 |
| P-6 | — | — | — | **新規** | Crossfade Drop監視 |

---

## 修正対象ファイル一覧（v5 最終）

| ファイル | P | 変更概要 |
|----------|---|----------|
| `src/core/IRetireProvider.h` | P0-A | pendingRetireCount/drainAll 純粋仮想関数 |
| `src/core/EpochDomain.h` | P0-A, P2-A | override, deprecated private化 |
| `src/audioengine/ISRRetireRouter.h` | P0-A | 関数ポインタ削除 |
| `src/audioengine/ISRRetireRouter.cpp` | P0-A | dynamic_cast削除 |
| `src/eqprocessor/EQProcessor.Core.cpp` | P0-B | Fallback削除 |
| `src/core/RCUReader.h` | P1-A | bool rootEnterSucceeded_ (non-atomic) |
| `src/audioengine/CrossfadeRuntime.h` | P1-C, P-2, P-6 | SPSC + dropCount + expectedDuration |
| `src/audioengine/RuntimeHealthMonitor.h` | P1-B, P2-C, P-2, P-4, P-5, P-6 | 全監視統合 |
| `src/audioengine/RuntimeHealthMonitor.cpp` | 同上 | tick()拡張 |
| `src/audioengine/PublicationAdmission.h` | P1-B, P2-C | HealthStateRef |
| `src/audioengine/PublicationAdmission.cpp` | P1-B | evaluate()拡張 |
| `src/audioengine/AudioEngine.h` | P1-B, P-1, P-5 | 配線 |
| `src/audioengine/DSPTransition.h` | P1-C | notifyFadeComplete |
| `src/audioengine/AudioEngine.Timer.cpp` | P1-C | SPSC consumer |
| `src/audioengine/ISRShutdown.h` | P-3 | ShutdownBlockingReason |
| `src/audioengine/ISRShutdown.cpp` | P2-B, P-3 | VerifyDrained |
| `src/audioengine/WorldLifecycleAudit.h` | P3-B | 新規: CAS式二重retire防御 |
| `src/audioengine/AudioEngine.Retire.cpp` | P3-A, P-1 | HWM + Monitor連携 |
