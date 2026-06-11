# ISR Bridge Runtime Practical Stable 改修計画 v4

**作成日**: 2026-06-11
**根拠文書**: `01_review_validation_report.md` / 第三者フィードバック v2
**v3→v4 差分**:

- P1-A: `m_valid`（永続）→ `lastEnterSucceeded_`（enter 単位）に変更
- P1-C: `atomic<CrossfadeId>` → `SPSCRingBuffer<CompletedFadeEvent, 32>`（既存流用）
- Practical-2: `completeTimestampUs_` → `fadeStartTimestampUs_` に修正
- P3-B: `activeWorldCount` に二重 retire 防御 (`jassert`) 追加
- Practical-4: Reader Slot Usage Telemetry（新規）
- Practical-5: Retire Reclaim Latency（新規、既存API流用）

---

## 優先度定義

| 優先度 | 定義 |
|--------|------|
| **P0** | Practical Stable に必須。未対処で運用破綻リスク |
| **P1** | 長時間運用の安定性に寄与。監視・制御の閉ループ化 |
| **P2** | 防御的改善。現状でも運用可能だが強化が望ましい |
| **P3** | 中長期的改善。Phase-C/D 以降の検討事項 |

---

## Phase 0: 即時修正（P0、総工数〜2h）

### P0-A: IEpochProvider に pendingRetireCount/drainAll を追加（〜1h）◎

**方針**: `IRetireProvider` に純粋仮想関数として追加。`ISRRetireRouter` の
`dynamic_cast` と未初期化関数ポインタを完全撤廃。

**注意**: `drainAll()` は shutdown 専用 API。理論上は `IShutdownDrainable` 等への
分離も考えられるが、工数対効果と Practical Stable の完了優先の観点から
`IRetireProvider` への追加で許容する。

**変更ファイル**:

- `src/core/IRetireProvider.h` — `pendingRetireCount()`, `drainAll()` 純粋仮想関数追加
- `src/core/EpochDomain.h` — `override` 明示
- `src/audioengine/ISRRetireRouter.h` — 関数ポインタ削除
- `src/audioengine/ISRRetireRouter.cpp` — dynamic_cast 削除、委譲に変更

---

### P0-B: EQProcessor Fallback 経路削除 + Releaseガード（〜1h）◎

**方針**: Fallback 経路を完全削除。`if (m_retireCoordinator == nullptr) return false;`
で Release ビルドでも安全を確保。

**変更ファイル**:

- `src/eqprocessor/EQProcessor.Core.cpp` — Fallback 削除、nullptr ガード

---

## Phase 1: 監視・制御閉ループ（P1、総工数〜4.5日）

### P1-A: RCUReader の enter 単位 Fail-Closed 化（〜4h）△→◎

**問題**: v3案の `std::atomic<bool> m_valid`（永続フラグ）は、RCUReader が
**クラスのメンバ変数として長期間再利用**される設計のため、一度の slot 枯渇で
永続的に invalid になる。これは一時的な枯渇を永続障害へ変換する。

**RCUReader の実際の使用パターン**（コード調査結果）:

- `NoiseShaperLearner` のメンバ変数として `rcuReader(engineRef.getRetireRouter())`
- `RuntimePublicationOrchestrator` のメンバ変数として `publicationReader`
- 各クラスのコンストラクタで一度構築され、その後 `enter()/exit()` を繰り返す
- `RCUReaderGuard` が RAII ラッパーとして `enter()/exit()` を呼ぶ

**修正方針**: 永続 `m_valid` → **enter 単位の `lastEnterSucceeded_`** に変更。
呼び出し側は `enter()` 成功直後にのみ `lastEnterSucceeded_` が true になる。

#### A-1: RCUReader に per-enter tracking 追加

```cpp
class RCUReader {
public:
    void enter() noexcept
    {
        const uint32_t previousDepth = convo::fetchAddAtomic(nestingDepth, ...);
        if (previousDepth > 0)
        {
            // ★ ネスト時は直前の enter 結果を引き継ぐ
            return;
        }

        // ... CASロジック ...

        const int tid = acquireThreadSlot();
        if (tid >= 0)
        {
            epochProvider->enterReader(tid);
            // ★ 今回の enter 成功を記録
            convo::publishAtomic(lastEnterSucceeded_, true, std::memory_order_release);
        }
        else
        {
            // ★ 今回の enter 失敗を記録（以降の exit まで有効）
            convo::publishAtomic(lastEnterSucceeded_, false, std::memory_order_release);
            // 後片付け（nestingDepth / ownerThreadToken の解放）
            // ...既存の解放ロジック...
        }
    }

    /** ★ 今回の enter() が成功し、epoch 保護下有効かを返す */
    [[nodiscard]] bool lastEnterSucceeded() const noexcept
    {
        return convo::consumeAtomic(lastEnterSucceeded_, std::memory_order_acquire);
    }

    void exit() noexcept
    {
        // ...既存の exit ロジック...
        // ★ exit 完了時にフラグをリセット（次の enter に備える）
        //   注: exit は enter 成功/失敗に関わらず呼ばれる
    }

private:
    // ...
    std::atomic<bool> lastEnterSucceeded_{false};  // ★ per-enter tracking
};
```

#### A-2: RCUReaderGuard / ObservedRuntime でチェック

```cpp
class RCUReaderGuard {
public:
    explicit RCUReaderGuard(RCUReader& r) noexcept : reader(&r) {
        reader->enter();
    }
    [[nodiscard]] bool valid() const noexcept {
        return reader && reader->lastEnterSucceeded();
    }
    // ...
};

// ObservedRuntime または makeRuntimeReadHandle() 内
if (!guard.valid())
    return nullptr;  // Fail-Closed: 保護無し読み取りを拒否
```

**v3→v4 改善点**:

- ❌ `m_valid`（永続、一度失敗すると永久に invalid）
- ✅ `lastEnterSucceeded_`（enter 単位、次の enter で再試行可能）

---

### P1-B: HealthMonitor → Admission 閉ループ完成（〜2日）○

v3計画を踏襲（変更なし）。`ISRHealthState { Healthy, Degraded, Critical }` 導入。
`std::atomic<ISRHealthState>` は 1byte で lock-free 保証。

---

### P1-C: Crossfade 完了イベント SPSC キュー化（〜1日）△→◎

**問題**: v3案の `std::atomic<CrossfadeId> completedFadeId_` は連続完了
（fade 101 → 102）で中間 ID (101) が消失する。ConvoPeq の
`CrossfadeAuthorityRuntime` が複数 crossfade の同時追跡を想定している設計と矛盾する。

**修正方針**: 既存の `SPSCRingBuffer<T, Capacity>`（`CommandBuffer.h` に実装済み、
SPSC lock-free、RT-safe）を流用し、`CompletedFadeEvent` の SPSC キューとする。

#### C-1: CompletedFadeEvent 定義 + CrossfadeRuntime 拡張

```cpp
// CompletedFadeEvent: AudioThread → Timer へ渡す完了イベント
struct CompletedFadeEvent {
    CrossfadeId id;
    uint64_t    completedTimestampUs;
};
static_assert(std::is_trivially_copyable_v<CompletedFadeEvent>,
    "Must be trivially copyable for SPSC queue");

// CrossfadeRuntime.h
#include "core/CommandBuffer.h"  // SPSCRingBuffer

class CrossfadeRuntime {
public:
    void start(double fadeTimeSec, double sampleRate) noexcept
    {
        // ...既存...
        // ★ P1-C: 開始タイムスタンプを記録（Practical-2 Timeout監視用）
        convo::publishAtomic(fadeStartTimestampUs_, getCurrentTimeUs(), std::memory_order_release);
    }

    // ★ P1-C: AudioThread から呼ばれる完了通知
    void notifyFadeComplete(CrossfadeId id) noexcept
    {
        CompletedFadeEvent ev{id, getCurrentTimeUs()};
        completedFadeQueue_.push(ev);  // SPSC: wait-free bounded push
    }

    // ★ P1-C: Timer 側で完了イベントを消費
    [[nodiscard]] bool consumeCompletedFade(CompletedFadeEvent& ev) noexcept
    {
        return completedFadeQueue_.pop(ev);
    }

    // ★ Practical-2: 開始からの経過時間（Timeout 監視用）
    [[nodiscard]] uint64_t getFadeAgeUs() const noexcept
    {
        uint64_t start = convo::consumeAtomic(fadeStartTimestampUs_, std::memory_order_acquire);
        if (start == 0) return 0;
        return getCurrentTimeUs() - start;
    }

    void complete() noexcept
    {
        // ...既存...
        convo::publishAtomic(fadeStartTimestampUs_, 0, std::memory_order_release);
    }

    void reset() noexcept
    {
        // ...既存...
        convo::publishAtomic(fadeStartTimestampUs_, 0, std::memory_order_release);
        // ★ キューを空に（DropAll 相当）
        CompletedFadeEvent dummy;
        while (completedFadeQueue_.pop(dummy)) {}
    }

private:
    // ...既存メンバ...
    SPSCRingBuffer<CompletedFadeEvent, 32> completedFadeQueue_;  // ★ P1-C
    std::atomic<uint64_t> fadeStartTimestampUs_{0};              // ★ P1-C/Practical-2
};
```

#### C-2: Timer 側 consumer 化

```cpp
// AudioEngine.Timer.cpp
CompletedFadeEvent ev;
while (crossfadeRuntime_.consumeCompletedFade(ev)) {
    // ★ SPSC から全イベントを消費（中間消失なし）
    //   101, 102 の両方とも Timer に届く
    dspHandleRuntime_.endCrossfade(ev.id);
    crossfadeAuthorityRuntime_.unregisterCrossfade(ev.id);
    // ... cleanup ...
}
```

**SPSCキュー(32) の根拠**:

- 1回の Timer callback 間隔（JUCE default ~30Hz ≈ 33ms）に発生する
  crossfade 完了は高々 1 〜 2 個（連続 publish でも最大数個）
- 32エントリは十分なマージン
- `SPSCRingBuffer` は RT-safe（wait-free bounded push）

---

## Phase 2: 防御的改善（P2、総工数〜2.5日）

### P2-A: Deprecated API 完全除去（〜2h）◎

v3計画を踏襲。`EpochDomain.h` の deprecated メソッドを private 化。

---

### P2-B: Shutdown VerifyDrained + BlockingReason（〜4h）○

v3計画を踏襲。`ShutdownRuntime` に `ShutdownBlockingReason` 追加。
`RuntimeDrainAudit::getPrimaryBlockingReason()` の結果を `markTimedOut()` 時に保存。

---

### P2-C: HealthState 単一 enum 共有化（〜1日）○

v3計画を踏襲。`std::atomic<AdmissionHealthContext>` を廃止し、
`std::atomic<ISRHealthState>` 単一 enum のみ共有。

---

## Phase 3: 中長期改善（P3、総工数〜1週間）

### P3-A: HealthMonitor ↔ RetirePressure 統合（〜3日）○

v3計画を踏襲。

---

### P3-B: WorldLifecycleAudit — RingBuffer + 二重 retire 防御（〜2日）○

**v3→v4**: `activeWorldCount` の二重 retire 対策を追加。

```cpp
class WorldLifecycleAudit {
public:
    void onWorldPublished(uint64_t worldId, uint64_t epoch, CorrelationId cid) noexcept
    {
        ringBuffer_.tryPush(WorldLifecycleRecord{...});
        convo::fetchAddAtomic(publishedCount_, 1u, std::memory_order_release);
        convo::fetchAddAtomic(activeWorldCount_, 1u, std::memory_order_release);
    }

    void onWorldRetired(uint64_t worldId, uint64_t epoch) noexcept
    {
        // ★ P3-B: 二重 retire 防御
        //   activeWorldCount が 0 での retire は不整合を意味する。
        //   Debug: jassert で即座に検出
        //   Release: saturated subtraction で負数化を防止
        uint64_t prevActive = convo::consumeAtomic(activeWorldCount_, std::memory_order_acquire);
        if (prevActive == 0) {
            jassertfalse;  // 二重 retire 検出
            return;        // カウンタ不変を維持
        }

        convo::fetchSubAtomic(activeWorldCount_, 1u, std::memory_order_release);
        convo::fetchAddAtomic(retiredCount_, 1u, std::memory_order_release);
    }

    [[nodiscard]] bool allWorldsRetired() const noexcept {
        return convo::consumeAtomic(activeWorldCount_, std::memory_order_acquire) == 0;
    }

private:
    FixedRingBuffer<WorldLifecycleRecord, 4096> ringBuffer_;
    std::atomic<uint64_t> activeWorldCount_{0};
    std::atomic<uint64_t> publishedCount_{0};
    std::atomic<uint64_t> retiredCount_{0};
};
```

**二重 retire の検出価値**:

- Debug: `jassertfalse` で即時発見 → テストで捕捉
- Release: `return` でカウンタ不変維持 → システム全体の不整合を防止
- 将来の拡張: Debug 限定の `activeWorldIds` 小さな集合でどの worldId が二重 retire されたか追跡可能

---

## Practical-1: Retire Queue High Watermark 観測（P2、〜2h）◎

v3計画を踏襲。`AudioEngine::maxPendingRetireObserved_` を追加。

---

## Practical-2: Crossfade Timeout 監視 — 開始時刻基準に修正（P1、〜4h）△→◎

**問題**: v3案の `getFadeAgeUs()` は `completeTimestampUs_`（完了時刻）から
経過時間を計算していた。これでは fade が完了しない場合に `completeTimestampUs_ == 0`
となり、age = 0 で Timeout を検出できない。

**修正方針**: `fadeStartTimestampUs_`（開始時刻）を基準に Timeout 判定。

```cpp
// CrossfadeRuntime.h（P1-C と統合）
std::atomic<uint64_t> fadeStartTimestampUs_{0};  // ★ 開始時刻基準

// HealthMonitor 側
void RuntimeHealthMonitor::checkCrossfadeTimeout() noexcept {
    if (!m_crossfadeRuntime) return;
    if (!m_crossfadeRuntime->isPending()) return;

    uint64_t ageUs = m_crossfadeRuntime->getFadeAgeUs();  // ★ now - fadeStartTimestampUs_
    if (ageUs > kCrossfadeTimeoutUs) {                     // ★ 開始から30秒経過
        emitOnTransition(m_prevCrossfadeState, MonitorState::Error,
            HealthEvent::Severity::Error, EVENT_CROSSFADE_TIMEOUT, ageUs / 1000);
        // HealthState → Critical 昇格
    }
}
```

**v3→v4 修正点**:

- ❌ `completeTimestampUs_` (完了時刻基準) → fade 未完了時に age=0
- ✅ `fadeStartTimestampUs_` (開始時刻基準) → 正しく経過時間を計測

---

## Practical-3: Shutdown Audit Event 拡充（P2、〜2h）◎

v3計画を踏襲。`ShutdownBlockingReason` 追加。

---

## Practical-4: Reader Slot Usage Telemetry（P1、〜4h）【新規】

**目的**: Reader slot の使用率を監視し、枯渇予兆を事前に検出する。

**既存 API**: `EpochDomain::activeReaderCount()` / `ISRRetireRouter::activeReaderCount()`
が現在の使用 slot 数を返す。`kMaxReaders = 64`。

**修正**:

```cpp
// RuntimeHealthMonitor.h に追加
static constexpr uint32_t EVENT_READER_SLOT_USAGE_WARNING  = 3010;
static constexpr uint32_t EVENT_READER_SLOT_USAGE_CRITICAL = 3011;

static constexpr double kReaderSlotWarningThreshold  = 0.50;  // 50% (32/64)
static constexpr double kReaderSlotCriticalThreshold = 0.75;  // 75% (48/64)
// ★ 90%/100% は HealthState → Critical へ直結

void setReaderSlotRef(const std::atomic<uint32_t>* ref) noexcept {
    m_readerSlotRef = ref;
}

// RuntimeHealthMonitor::tick() 内
void checkReaderSlotUsage() noexcept {
    if (!m_readerSlotRef) return;
    uint32_t activeCount = convo::consumeAtomic(*m_readerSlotRef, std::memory_order_acquire);
    constexpr uint32_t kMaxSlots = EpochDomain::kMaxReaders;  // 64

    double usage = static_cast<double>(activeCount) / kMaxSlots;

    if (usage >= 0.90) {  // 90%以上 → Critical
        emitOnTransition(m_prevReaderSlotState, MonitorState::Error,
            HealthEvent::Severity::Error, EVENT_READER_SLOT_USAGE_CRITICAL,
            activeCount, kMaxSlots);
    } else if (usage >= kReaderSlotCriticalThreshold) {  // 75%以上 → Warning
        emitOnTransition(m_prevReaderSlotState, MonitorState::Warning,
            HealthEvent::Severity::Warning, EVENT_READER_SLOT_USAGE_WARNING,
            activeCount, kMaxSlots);
    } else if (usage >= kReaderSlotWarningThreshold) {  // 50%以上 → Info
        // 情報提供のみ、状態遷移不要
    }
}
```

**診断価値**:

- 50% 超過: リソース計画の見直しタイミング
- 75% 超過: スレッド増殖の可能性あり、調査推奨
- 90% 超過: 枯渇リスク高い、HealthState → Critical
- 100%: `registerReaderThread()` が -1 を返し、P1-A の Fail-Closed が発動

---

## Practical-5: Retire Reclaim Latency 監視（P2、〜4h）【新規】

**目的**: `pendingRetireCount` だけでなく、**retire されてから回収まで**
の所要時間を監視する。長時間滞留は Reader stuck や Epoch 進行停止の指標。

**既存資産**（コード調査結果）:

- `DeferredDeletionQueue::getMaxRetireAgeUs()` — 既に実装済み（ただし呼び出し元なし＝dead API）
- `AudioEngine::reclaimLatency_` — 既に実装済み

**修正**: HealthMonitor へ接続する。

```cpp
// RuntimeHealthMonitor.h に追加
static constexpr uint32_t EVENT_RETIRE_AGE_WARNING = 1010;
static constexpr uint32_t EVENT_RETIRE_AGE_CRITICAL = 1011;

static constexpr uint64_t kRetireAgeWarningUs  = 5'000'000;   // 5秒
static constexpr uint64_t kRetireAgeCriticalUs = 30'000'000;  // 30秒

void setMaxRetireAgeRef(const std::atomic<uint64_t>* ref) noexcept {
    m_maxRetireAgeRef = ref;
}

// RuntimeHealthMonitor::tick() 内
void checkRetireReclaimLatency() noexcept {
    if (!m_maxRetireAgeRef) return;
    uint64_t maxAgeUs = convo::consumeAtomic(*m_maxRetireAgeRef, std::memory_order_acquire);

    if (maxAgeUs > kRetireAgeCriticalUs) {
        emitOnTransition(m_prevRetireAgeState, MonitorState::Error,
            HealthEvent::Severity::Error, EVENT_RETIRE_AGE_CRITICAL,
            maxAgeUs / 1000);
    } else if (maxAgeUs > kRetireAgeWarningUs) {
        emitOnTransition(m_prevRetireAgeState, MonitorState::Warning,
            HealthEvent::Severity::Warning, EVENT_RETIRE_AGE_WARNING,
            maxAgeUs / 1000);
    }
}
```

**配線**:

- `EpochDomain` の `DeferredDeletionQueue` に `getMaxRetireAgeUs()` が既存
- `IEpochProvider` に `getMaxRetireAgeUs()` を追加（または P0-A の pendingRetireCount と同様に追加）
- HealthMonitor へ注入

---

## マイルストーン

| M | 内容 | フェーズ | 時期 |
|---|------|---------|------|
| M1 | IEpochProvider 拡張 + dynamic_cast 撤廃 | P0-A, P0-B | Day 1 AM |
| M2 | RCUReader per-enter Fail-Closed | P1-A | Day 1 PM |
| M3 | Crossfade SPSC キュー + Timeout修正 | P1-C, P-2 | Day 2 |
| M4 | 閉ループ完成 (HealthState→Admission + ReaderSlot + RetireAge) | P1-B, P2-C, P-4, P-5 | Day 3-4 |
| M5 | Shutdown 強化 + Deprecated整理 | P2-A, P2-B, P-3 | Day 4-5 |
| M6 | WorldLifecycleAudit + RetireHWM | P3-B, P-1 | Day 5 |
| M7 | HealthMonitor↔RetirePressure統合 | P3-A | Week 2-3 |

---

## 修正対象ファイル一覧（v4）

| ファイル | P | 変更概要 |
|----------|---|----------|
| `src/core/IRetireProvider.h` | P0-A | pendingRetireCount/drainAll 純粋仮想関数追加 |
| `src/core/EpochDomain.h` | P0-A, P2-A | override 明示, deprecated private化 |
| `src/audioengine/ISRRetireRouter.h` | P0-A | 関数ポインタ削除 |
| `src/audioengine/ISRRetireRouter.cpp` | P0-A | dynamic_cast 削除 |
| `src/eqprocessor/EQProcessor.Core.cpp` | P0-B | Fallback削除 + nullptr guard |
| `src/core/RCUReader.h` | P1-A | m_valid→lastEnterSucceeded_ per-enter |
| `src/audioengine/CrossfadeRuntime.h` | P1-C, P-2 | SPSC queue, fadeStartTimestampUs_ |
| `src/audioengine/RuntimeHealthMonitor.h` | P1-B, P2-C, P-2, P-4, P-5 | 全監視追加 |
| `src/audioengine/RuntimeHealthMonitor.cpp` | P1-B, P-2, P-4, P-5 | tick()拡張 |
| `src/audioengine/PublicationAdmission.h` | P1-B, P2-C | HealthStateRef |
| `src/audioengine/PublicationAdmission.cpp` | P1-B | evaluate()拡張 |
| `src/audioengine/AudioEngine.h` | P1-B, P-1, P-5 | HealthState, HWM, RetireAge配線 |
| `src/audioengine/DSPTransition.h` | P1-C | notifyFadeComplete呼び出し |
| `src/audioengine/AudioEngine.Timer.cpp` | P1-C | SPSC consumer化 |
| `src/audioengine/ISRShutdown.h` | P-3 | ShutdownBlockingReason |
| `src/audioengine/ISRShutdown.cpp` | P2-B, P-3 | VerifyDrained + BlockingReason |
| `src/audioengine/WorldLifecycleAudit.h` | P3-B | 新規: RingBuffer + 二重retire防御 |
| `src/audioengine/AudioEngine.Retire.cpp` | P3-A, P-1 | HWM, Monitor連携 |

---

## v2→v3→v4 改善サマリ

| 項目 | v2 | v3 | v4 | v4 方針 |
|------|-----|-----|-----|---------|
| P0-A | △ dynamic_cast隠蔽 | ○ IEProvider拡張 | ◎ | IRetireProviderに純粋仮想関数追加 |
| P1-A | △ 一部危険 | △ m_valid永続 | ◎ | lastEnterSucceeded_ per-enter |
| P1-C | △ bool設計不足 | △ atomic<Id>取こぼし | ◎ | SPSCRingBuffer<Event,32>流用 |
| P-2 | — | △ completeTs基準 | ◎ | fadeStartTs基準に修正 |
| P3-B | △ vector無制限 | ○ RingBuffer+count | ○ | 二重retire jassert追加 |
| P-4 | — | — | 新規 | Reader Slot監視(50/75/90/100%) |
| P-5 | — | — | 新規 | Retire Age監視(既存API流用) |
