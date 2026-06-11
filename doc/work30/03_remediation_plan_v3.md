# ISR Bridge Runtime Practical Stable 改修計画 v3

**作成日**: 2026-06-11
**根拠文書**: `01_review_validation_report.md` / 第三者フィードバック
**v2→v3 主な変更**:

- P0-A: dynamic_cast隠蔽→IEpochProvider拡張に変更
- P0-B: Release安全の `return false` ガード追加
- P1-A: Fail-Open→Fail-Closed 化
- P1-C: `bool`→`CrossfadeId` ベースのイベントID設計
- P2-C: `atomic<struct>`→`ISRHealthState` enum 単一共有
- P3-B: 固定長 RingBuffer + カウンタ方式
- Practical-1〜3: 追加項目
**推定工数**: P0=2h / P1=4日 / P2=2日 / P3=1週間

---

## 優先度定義

| 優先度 | 定義 | 完了基準 |
|--------|------|----------|
| **P0** | Practical Stable に必須。未対処で運用破綻リスク | コード修正 + build green + CI gate通過 |
| **P1** | 長時間運用の安定性に寄与。監視・制御の閉ループ化 | コード修正 + build green + CI gate通過 |
| **P2** | 防御的改善。現状でも運用可能だが強化が望ましい | コード修正 + build green |
| **P3** | 中長期的改善。Phase-C/D 以降の検討事項 | 設計書 + 承認待ち |

---

## Phase 0: 即時修正（P0、総工数〜2h）

### P0-A: IEpochProvider に pendingRetireCount/drainAll を追加（〜1h）

**問題の本質**: `ISRRetireRouter` が `pendingRetireCount()` / `drainAll()` で
`dynamic_cast<EpochDomain*>(provider_)` を使っている理由は、これらのメソッドが
`IEpochProvider` インターフェースに存在しないためである。

旧計画の方針「dynamic_castをdynamic_castで隠蔽（関数ポインタ）」は本質的解決ではなく
**技術的負債を温存する**だけである。

**修正方針**: `IEpochProvider` / `IRetireProvider` に純粋仮想関数として追加する。

#### A-1: IRetireProvider に pendingRetireCount / drainAll を追加

`src/core/IRetireProvider.h`:

```cpp
class IRetireProvider
{
public:
    virtual ~IRetireProvider() = default;

    virtual bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept = 0;
    virtual void tryReclaim() noexcept = 0;

    // ★ P0-A: EpochDomain 固有メソッドをインターフェースに昇格
    //   pendingRetireCount / drainAll は EpochDomain にしか実装がないが、
    //   Router が dynamic_cast でアクセスするより Interface で宣言する方が
    //   型安全であり、Practical Stable として正当。
    /** Return approximate count of pending retire entries. */
    virtual uint32_t pendingRetireCount() const noexcept = 0;

    /** Drain all pending retire entries (unsafe; shutdown only). */
    virtual void drainAll() noexcept = 0;
};
```

#### A-2: EpochDomain で override を明示

`src/core/EpochDomain.h`:

```cpp
uint32_t pendingRetireCount() const noexcept override { ... }
void drainAll() noexcept override { ... }
```

（既存実装をそのまま利用、override キーワードを追加）

#### A-3: ISRRetireRouter の dynamic_cast を削除し、委譲に変更

`src/audioengine/ISRRetireRouter.cpp`:

```cpp
uint32_t ISRRetireRouter::pendingRetireCount() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->pendingRetireCount();
}

void ISRRetireRouter::drainAll() noexcept
{
    assert(provider_ != nullptr);
    provider_->drainAll();
}
```

`src/audioengine/ISRRetireRouter.h` の関数ポインタ宣言も削除:

```cpp
// 削除:
// uint32_t (*pendingRetireFn_)(void*) = nullptr;
// void     (*drainAllFn_)(void*)      = nullptr;
// void*    epochContext_              = nullptr;
```

**確認方法**: build green + `grep "dynamic_cast<EpochDomain\*>" src/` が 0 になること。

**注意**: `SnapshotCoordinator` も `IEpochProvider` 経由で `enqueueRetire()` を
呼んでいるが、こちらは Interface 経由で問題ない。

---

### P0-B: EQProcessor Fallback 経路削除 + Releaseガード（〜1h）

**現状**: `EQProcessor.Core.cpp` に direct EpochDomain fallback が生存。
coordinator 未設定時に発動し、Authority bypass となる。

**修正内容**:

`enqueueDeferredDeleteWithFallback()` を修正:

```cpp
bool EQProcessor::enqueueDeferredDeleteWithFallback(void* ptr,
                                                    void (*deleter)(void*),
                                                    uint64_t epoch) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return true;

    // [Phase-B] coordinator は常時設定済み。
    // ★ Release build でも安全なように nullptr ガードを残す。
    if (m_retireCoordinator == nullptr)
        return false;

    const uint64_t retireEpoch = (epoch != 0) ? epoch : m_epochDomain.currentEpoch();
    convo::isr::ISRRetireRouter stackRouter(m_epochDomain);
    auto result = m_retireCoordinator->enqueueRetire(
        convo::isr::RetireAuthority::Granted,
        stackRouter,
        ptr, deleter, retireEpoch);
    if (result == convo::isr::RetireEnqueueResult::Success)
        return true;

    // [P0-5] enqueue failure -> drop + telemetry (RT-safe).
    return false;
}
```

**ポイント**:

- `jassert` のみでは Release build で unprotected access になるため、
  `if (m_retireCoordinator == nullptr) return false;` を残す
- Fallback 経路（`m_epochDomain.enqueueRetire()` 直接呼び出し）は**完全削除**

**確認方法**: build green + `grep "Fallback.*direct EpochDomain"` が src/ で 0。

---

## Phase 1: 監視・制御閉ループ（P1、総工数〜4日）

### P1-A: RCUReader Fail-Closed 化（〜4h）

**問題**: 現在の `RCUReader::enter()` は slot 取得失敗時に「保護なし読取」へ
進んでしまう（Fail-Open）。Practical Stable では Fail-Closed が必須。

**修正方針**: `RCUReader` に `m_valid` フラグを追加し、enter() 失敗時は
`m_valid = false` に設定。呼び出し側は `reader.valid()` をチェックして
`nullptr` を返す。

#### A-1: RCUReader に valid() 追加

`src/core/RCUReader.h`:

```cpp
class RCUReader
{
public:
    // ... 既存 ...

    void enter() noexcept
    {
        // ★ Fail-Closed: 一度 invalid になった reader は enter をブロック
        if (!m_valid)
            return;

        const uint32_t previousDepth = convo::fetchAddAtomic(nestingDepth, ...);
        if (previousDepth > 0)
            return;

        // ... 既存のCASロジック ...

        const int tid = acquireThreadSlot();
        if (tid >= 0)
        {
            epochProvider->enterReader(tid);
        }
        else
        {
            // ★ P1-A: slot 取得失敗 → Fail-Closed
            //   以降この reader は epoch 保護なしとマーク。
            //   呼び出し側が valid() で検知し、観測をスキップする。
            convo::publishAtomic(m_valid, false, std::memory_order_release);

            // 後片付け（nestingDepth / ownerThreadToken の解放）
            convo::fetchSubAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
            uint64_t expectedOwnerOnRelease = threadToken;
            convo::compareExchangeAtomic(ownerThreadToken,
                                         expectedOwnerOnRelease,
                                         static_cast<uint64_t>(0),
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire);
        }
    }

    /** ★ P1-A: Reader が有効な epoch 保護下にあるか確認 */
    [[nodiscard]] bool valid() const noexcept
    {
        return convo::consumeAtomic(m_valid, std::memory_order_acquire);
    }

    // ... exit() は既存のまま ...

private:
    // ... 既存メンバ ...
    std::atomic<bool> m_valid{ true };   // ★ P1-A: 追加
};
```

#### A-2: ObservedRuntime / 呼び出し側での valid() チェック

```cpp
// ObservedRuntime または各 makeRuntimeReadHandle() 内
if (!reader.valid())
    return nullptr;  // または空の読み取り結果
```

**注意**: RCUReaderGuard のコンストラクタも同様に修正:

```cpp
class RCUReaderGuard {
public:
    explicit RCUReaderGuard(RCUReader& reader) noexcept : reader_(&reader) {
        reader_->enter();  // ← 内部で m_valid チェック済み
    }
    [[nodiscard]] bool valid() const noexcept { return reader_ && reader_->valid(); }
};
```

---

### P1-B: HealthMonitor → Admission 閉ループ完成（〜2日）

**修正内容**: v2計画を踏襲（評価: ○）。

- `ISRHealthState { Healthy, Degraded, Critical }` 導入
- `RuntimeHealthMonitor` が HealthState を更新
- `PublicationAdmission::evaluate()` が HealthState を参照
- AudioEngine で配線

詳細は v2計画の P1-B を参照。以下の点に注意:

- **`std::atomic<ISRHealthState>` は lock-free 保証あり**（1byte enum）
- HealthMonitor の各 `checkXxx()` 結果を統合して HealthState を決定
- Admission は HealthState のみ読み、詳細値は Monitor 所有

---

### P1-C: Crossfade 完了の RT 権威化 — Event ID ベース（〜1日）

**問題**: v2計画の `std::atomic<bool> fadeComplete_` では連続 publish で
「どの fade が完了したか」を識別できない。

**修正方針**: `CrossfadeId`（既存: `using CrossfadeId = uint32_t`）を使用。
`CrossfadeRuntime` に `std::atomic<CrossfadeId> completedFadeId_` を追加。
AudioThread 側で完了検出時に ID を書き込み、Timer 側で消費する。

#### C-1: CrossfadeRuntime に completedFadeId 追加

`src/audioengine/CrossfadeRuntime.h`:

```cpp
class CrossfadeRuntime {
public:
    void start(double fadeTimeSec, double sampleRate) noexcept
    {
        // ... 既存 ...
        // ★ P1-C: 新しい fade 開始時に completedFadeId をリセット
        convo::publishAtomic(completedFadeId_, CrossfadeId{0}, std::memory_order_release);
    }

    // ★ P1-C: AudioThread から呼ばれる完了通知
    //   DSPTransition または fade advancement ロジック内で、
    //   クロスフェード完了を検出したらこのメソッドを呼ぶ。
    void notifyFadeComplete(CrossfadeId id) noexcept
    {
        convo::publishAtomic(completedFadeId_, id, std::memory_order_release);
        convo::publishAtomic(completeTimestampUs_, getCurrentTimeUs(), std::memory_order_release);
    }

    // ★ P1-C: Timer 側で完了した fade の ID を消費（read-then-reset）
    //   戻り値: 完了した CrossfadeId（0 = 未完了）
    [[nodiscard]] CrossfadeId consumeCompletedFadeId() noexcept
    {
        CrossfadeId id = convo::exchangeAtomic(completedFadeId_, CrossfadeId{0}, std::memory_order_acq_rel);
        return id;
    }

    [[nodiscard]] uint64_t getFadeAgeUs() const noexcept
    {
        uint64_t ts = convo::consumeAtomic(completeTimestampUs_, std::memory_order_acquire);
        if (ts == 0) return 0;
        return getCurrentTimeUs() - ts;
    }

private:
    // ... 既存メンバ ...
    std::atomic<CrossfadeId> completedFadeId_{0};  // ★ P1-C: 追加
    std::atomic<uint64_t> completeTimestampUs_{0};  // ★ P1-C: 完了タイムスタンプ
};
```

#### C-2: Timer 側 consumer 化

```cpp
// AudioEngine.Timer.cpp
// 従来: m_coordinator.tryCompleteFade() ← Authority が Timer にある
// 修正後: completedFadeId を消費するのみ
CrossfadeId completedId = crossfadeRuntime_.consumeCompletedFadeId();
if (completedId != 0) {
    // ★ Authority は AudioThread 側にある。Timer は cleanup のみ。
    // DSP retire, publish idling world 等...
    dspHandleRuntime_.endCrossfade(completedId);
    crossfadeAuthorityRuntime_.unregisterCrossfade(completedId);
    // ... cleanup ...
}
```

**IDベースの利点**:

- 連続 publish（fade A → fade B → fade C）でどの完了が無視されたか追跡可能
- `completedFadeId_` を `exchangeAtomic(0)` で消費するため、複数回の完了通知が
  あっても最後の1つだけが Timer に届く（stale を自然にハンドリング）
- タイムスタンプでタイムアウト監視が可能（Practical-2と連携）

---

## Phase 2: 防御的改善（P2、総工数〜2日）

### P2-A: Deprecated API 完全除去（〜2h）

v2計画を踏襲（評価: ○）。`EpochDomain.h` の deprecated メソッドを private 化:

- `advanceEpoch()` → private
- `reclaimRetired()` → private
- `enqueueRetire()` 4-param → private

CI gate: `advanceEpoch` 呼び出し元の許容数を 0 に変更。

---

### P2-B: Shutdown VerifyDrained と DrainAudit 接続（〜4h）

v2計画を踏襲（評価: ○）。`VerifyDrained` phase で `RuntimeDrainAudit::isAllZero()` を使用。

**補足**: `RuntimeDrainAudit` の `getPrimaryBlockingReason()` も利用し、
Practical-3（Shutdown Audit Event）に接続する。

---

### P2-C: HealthState 単一 enum 共有化（〜1日）

**問題**: v2計画で提案した `std::atomic<AdmissionHealthContext>` は
複合型のため lock-free 保証がなく、Audio 系で内部 mutex が入る危険がある。

**修正方針**: 共有するのは `ISRHealthState` enum 1つだけ。
詳細な診断値は `RuntimeHealthMonitor` 内部に閉じ込める。

#### C-1: ISRHealthState 定義

```cpp
// RuntimeHealthMonitor.h または RuntimeBuildTypes.h
enum class ISRHealthState : uint8_t {
    Healthy = 0,    // 正常
    Degraded,       // 軽度障害（Reader枯渇/Retire backlog増加）
    Critical        // 重度障害（Publication stall/Reader stuck/Timout）
};
// static_assert: std::atomic<ISRHealthState> は lock-free（1byte）
```

#### C-2: PublicationAdmission は HealthState のみ参照

```cpp
// PublicationAdmission.h
class PublicationAdmission {
public:
    void setHealthStateRef(const std::atomic<ISRHealthState>* ref) noexcept {
        m_healthStateRef = ref;
    }

private:
    const std::atomic<ISRHealthState>* m_healthStateRef = nullptr;
};

// PublicationAdmission::evaluate() 内
if (m_healthStateRef) {
    auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
    switch (health) {
        case ISRHealthState::Critical:
            return Decision::RejectedPressure;
        case ISRHealthState::Degraded:
            if (/* low priority heuristic */)
                return Decision::RejectedLowPriority;
            break;
        case ISRHealthState::Healthy:
        default:
            break;
    }
}
```

**ポイント**:

- `std::atomic<ISRHealthState>` は 1byte のため、全環境で lock-free 保証
- Admission は詳細な診断値を知る必要がない（関心の分離）
- 詳細値（retire backlog, reader count 等）は HealthMonitor 内部のみ

---

## Phase 3: 中長期改善（P3、総工数〜1週間）

### P3-A: HealthMonitor ↔ RetirePressure 統合（〜3日）

v2計画を踏襲（評価: ○）。HealthMonitor の診断結果を
`AudioEngine::applyRetirePressurePolicyNoRt()` へフィードバックする経路を追加。

---

### P3-B: WorldLifecycleAudit — 固定長 RingBuffer + カウンタ方式（〜2日）

**問題**: v2計画の `std::vector<WorldLifecycleRecord>` は無制限に増加し、
長時間運用でメモリ肥大が発生する。

**修正方針**: 既存の `FixedRingBuffer<T, N>`（`TelemetryRecorder.h` に実装済み）を
再利用し、固定長 RingBuffer + カウンタ方式に変更。
Shutdown 完了判定はカウンタのみで行う。

#### B-1: LifecycleRecord 定義

```cpp
// src/audioengine/WorldLifecycleAudit.h
#include "TelemetryRecorder.h" // FixedRingBuffer
#include "ISRRuntimeIdentityGenerators.h"

namespace convo::isr {

struct WorldLifecycleRecord {
    uint64_t worldId;
    uint64_t publishEpoch;
    uint64_t retireEpoch;        // 0 = 未退役
    uint64_t publishTimestampUs;
    uint64_t retireTimestampUs;
    CorrelationId correlationId;
};

// ★ 固定長 RingBuffer（4096エントリ = 約256KB）
//   長時間運用でも溢れたら古いレコードから上書き（overwrite方式）
//   監査目的の最新4096件を保持すれば十分。
//   Shutdown 判定はカウンタで行う。
class WorldLifecycleAudit {
public:
    void onWorldPublished(uint64_t worldId, uint64_t epoch, CorrelationId cid) noexcept
    {
        ringBuffer_.tryPush(WorldLifecycleRecord{
            .worldId = worldId,
            .publishEpoch = epoch,
            .retireEpoch = 0,
            .publishTimestampUs = getCurrentTimeUs(),
            .retireTimestampUs = 0,
            .correlationId = cid
        });
        convo::fetchAddAtomic(publishedCount_, uint64_t{1}, std::memory_order_release);
        convo::fetchAddAtomic(activeWorldCount_, uint64_t{1}, std::memory_order_release);
    }

    void onWorldRetired(uint64_t worldId, uint64_t epoch) noexcept
    {
        // RingBuffer 内の該当レコードを検索して retireEpoch を設定（古いものは上書き済みの可能性あり）
        // ★ 正確な追跡が必要な shutdown 時は activeWorldCount カウンタが source-of-truth
        convo::fetchAddAtomic(retiredCount_, uint64_t{1}, std::memory_order_release);
        convo::fetchSubAtomic(activeWorldCount_, uint64_t{1}, std::memory_order_release);
    }

    // ★ Shutdown 完了判定用: カウンタのみ参照（RingBuffer は監査用）
    [[nodiscard]] bool allWorldsRetired() const noexcept {
        return convo::consumeAtomic(activeWorldCount_, std::memory_order_acquire) == 0;
    }

    [[nodiscard]] uint64_t activeWorldCount() const noexcept { ... }
    [[nodiscard]] uint64_t publishedCount() const noexcept { ... }
    [[nodiscard]] uint64_t retiredCount() const noexcept { ... }

    void emitSnapshot() const noexcept;  // 診断用ダンプ

private:
    FixedRingBuffer<WorldLifecycleRecord, 4096> ringBuffer_;
    std::atomic<uint64_t> activeWorldCount_{0};
    std::atomic<uint64_t> publishedCount_{0};
    std::atomic<uint64_t> retiredCount_{0};
};

} // namespace convo::isr
```

**設計根拠**:

- 1000 publish/day × 30日 = 30,000 → RingBuffer 4096 では古いものが上書きされる
- しかし監査目的の「直近4096件の完全な発行→退役ペア」は常に保持
- Shutdown 判定は `activeWorldCount == 0` のカウンタ方式で正確
- 長期トレンド分析は `publishedCount_` / `retiredCount_` の累積カウンタで代用

---

## Practical-1: Retire Queue High Watermark 観測（P2、〜2h）

**現状**: `pendingRetireCount()` で現在値は取得できるが、長期運用での
最大滞留数を記録していない。

**修正**: AudioEngine.Retire.cpp の HWM 更新ロジックに `maxPendingRetireObserved` を追加:

```cpp
// AudioEngine.h に追加
std::atomic<uint64_t> maxPendingRetireObserved_{0};

// AudioEngine.Retire.cpp の retire queue 監視ループ内
uint64_t current = retireRuntime_.pendingRetireCount();
uint64_t prevMax = convo::consumeAtomic(maxPendingRetireObserved_, std::memory_order_acquire);
while (current > prevMax) {
    if (convo::compareExchangeAtomic(maxPendingRetireObserved_, prevMax, current,
                                     std::memory_order_acq_rel, std::memory_order_acquire))
        break;
}
```

**診断価値**: リリース後の運用データから「本当に必要な HWM」を決定できる。
現状の固定値 3072 が適正かどうかの検証に使用。

---

## Practical-2: Crossfade Timeout 監視（P1、〜4h）

**問題**: 現在の計画には「crossfade が timeout したらどうするか」の規定がない。

**修正**: HealthMonitor に crossfade timeout 検出を追加:

```cpp
// RuntimeHealthMonitor.h に追加
static constexpr uint32_t EVENT_CROSSFADE_TIMEOUT = 4001;
static constexpr uint64_t kCrossfadeTimeoutUs = 30'000'000;  // 30秒

void setCrossfadeRuntime(const isr::CrossfadeRuntime* rt) noexcept {
    m_crossfadeRuntime = rt;
}

// RuntimeHealthMonitor::tick() 内
void RuntimeHealthMonitor::checkCrossfadeTimeout() noexcept {
    if (!m_crossfadeRuntime) return;
    if (!m_crossfadeRuntime->isPending()) return;  // 進行中でなければ対象外

    uint64_t ageUs = m_crossfadeRuntime->getFadeAgeUs();
    if (ageUs > kCrossfadeTimeoutUs) {
        emitOnTransition(m_prevCrossfadeState, MonitorState::Error,
            HealthEvent::Severity::Error, EVENT_CROSSFADE_TIMEOUT, ageUs / 1000);
        // ★ HealthState を Critical に昇格（Admission が publish をブロック）
    }
}
```

**HealthState との連携**:

- Crossfade timeout 検出 → `ISRHealthState::Critical`
- → `PublicationAdmission` が全 publish を拒否
- → システム全体の安全停止へ誘導

---

## Practical-3: Shutdown Audit Event 拡充（P2、〜2h）

**問題**: `ShutdownRuntime::markTimedOut()` / `markFailed()` は現在
原因（どのリソースが完了を阻害しているか）を記録しない。

**修正**:

```cpp
// ISRShutdown.h
enum class ShutdownBlockingReason : uint8_t {
    None = 0,
    PendingPublication,
    PendingRetire,
    ActiveCrossfade,
    DeferredPublish,
    QuarantineResident,
    RouterPendingRetire,
    ReaderActive,
    Unknown
};

class ShutdownRuntime {
    // ... 既存 ...
    void markTimedOut(ShutdownBlockingReason reason = ShutdownBlockingReason::Unknown) noexcept;
    void markFailed(ShutdownBlockingReason reason = ShutdownBlockingReason::Unknown) noexcept;
    ShutdownBlockingReason getBlockingReason() const noexcept;

private:
    // ...
    std::atomic<ShutdownBlockingReason> blockingReason_{ShutdownBlockingReason::None};
};
```

`advancePhase()` の `VerifyDrained` → `TimedOut` 遷移時に
`RuntimeDrainAudit::getPrimaryBlockingReason()` の結果を保存。

**診断価値**: 障害解析時に「なぜ shutdown が完了しなかったか」が即座に特定可能。
`emitShutdownTrace()` の JSON 出力にも反映。

---

## マイルストーン

| マイルストーン | 内容 | フェーズ | 時期目安 |
|---|---|---|---|
| **M1: IEpochProvider 拡張** | pendingRetireCount/drainAll を Interface に追加、dynamic_cast削除 | P0-A | Day 1 AM |
| **M2: Fallback削除 + RetireHWM** | EQProcessor Fallback削除 + maxPendingRetireObserved | P0-B, P-1 | Day 1 PM |
| **M3: RCUReader Fail-Closed** | valid()追加 + 呼び出し側修正 | P1-A | Day 2 |
| **M4: Crossfade ID 化 + Timeout** | completedFadeId + fade timeout監視 | P1-C, P-2 | Day 3 |
| **M5: 閉ループ完成** | HealthMonitor → HealthState → Admission | P1-B, P2-C | Day 4-5 |
| **M6: Shutdown強化** | VerifyDrained + BlockingReason | P2-B, P-3 | Day 5 |
| **M7: Deprecated整理** | EpochDomain private化 | P2-A | Day 5 |
| **M8: 長期運用対応** | RingBuffer WorldLifecycleAudit + HealthMonitor↔RetirePressure | P3-A, P3-B | Week 2-3 |

---

## リスクと注意点（v3更新）

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| IEpochProvider 拡張による既存実装者への影響 | EpochDomain と ISRRetireRouter のみに影響 | 両方に override 追加。他の IEpochProvider 実装がないことを確認 |
| RCUReader Fail-Closed による読み取り欠落 | Epoch保護なしでも「読めてしまっていた」動作が「読めなくなる」 | Degraded 状態で Admission が publish を抑制するため、データ欠落よりシステム安定優先 |
| CrossfadeId の atomic exchange 競合 | completedFadeId の取りこぼし | exchangeAtomic(0) で消費するため、最悪でも最後の1つは届く。複数連続の場合は中間が欠けても問題なし |
| FixedRingBuffer 4096 の溢れ | 古い WorldLifecycleRecord の消失 | Shutdown 判定はカウンタで行うため問題なし。長期傾向は累積カウンタで代用 |
| Crossfade Timeout 30秒の妥当性 | 長い IR で 30秒以上かかる可能性 | 設定可能な定数化。kCrossfadeTimeoutUs をコンフィグ化してもよい |

---

## 付録: 修正対象ファイル一覧（v3）

| ファイル | 関連P | 変更概要 |
|----------|-------|----------|
| `src/core/IRetireProvider.h` | P0-A | pendingRetireCount/drainAll 純粋仮想関数追加 |
| `src/core/EpochDomain.h` | P0-A, P2-A | override 明示, deprecated private化 |
| `src/audioengine/ISRRetireRouter.h` | P0-A | 関数ポインタ削除 |
| `src/audioengine/ISRRetireRouter.cpp` | P0-A | dynamic_cast 削除、委譲に変更 |
| `src/eqprocessor/EQProcessor.Core.cpp` | P0-B | Fallback経路削除、nullptrガード |
| `src/core/RCUReader.h` | P1-A | m_valid追加、Fail-Closed化 |
| `src/audioengine/RuntimeHealthMonitor.h` | P1-B, P-2 | ISRHealthState, CrossfadeTimeout |
| `src/audioengine/RuntimeHealthMonitor.cpp` | P1-B, P-2 | tick()拡張 |
| `src/audioengine/PublicationAdmission.h` | P1-B, P2-C | HealthStateRef追加 |
| `src/audioengine/PublicationAdmission.cpp` | P1-B | evaluate()拡張 |
| `src/audioengine/AudioEngine.h` | P1-B, P-1 | HealthState, maxPendingRetireObserved |
| `src/audioengine/CrossfadeRuntime.h` | P1-C, P-2 | completedFadeId, timeout |
| `src/audioengine/DSPTransition.h` | P1-C | notifyFadeComplete呼び出し |
| `src/audioengine/AudioEngine.Timer.cpp` | P1-C | Timer consumer化 |
| `src/audioengine/ISRShutdown.h` | P-3 | ShutdownBlockingReason |
| `src/audioengine/ISRShutdown.cpp` | P2-B, P-3 | VerifyDrained 強化 |
| `src/audioengine/RuntimeDrainAudit.h` | P2-B | (変更なし) |
| `src/audioengine/WorldLifecycleAudit.h` | P3-B | 新規: RingBuffer方式 |
| `src/audioengine/AudioEngine.Retire.cpp` | P3-A, P-1 | maxPendingRetireObserved, Monitor連携 |
| `.github/scripts/check-work21-epochdomain-gates.ps1` | P0-A, P2-A | dynamic_cast検出追加 |

---

## 付録: 評価サマリ（v2→v3差分）

| 項目 | v2評価 | v3評価 | v3変更点 |
|------|--------|--------|----------|
| P0-A | △ 改修方針は正しいが実装方法は再検討 | ○ IEpochProvider拡張 | dynamic_cast隠蔽→Interface昇格 |
| P0-B | ○ 妥当 | ○ 妥当 | nullptr guard 追加で Release安全化 |
| P0-C | ○ 妥当 | ○ 妥当 | 変更なし |
| P1-A | △ 一部危険 | ○ Fail-Closed | m_valid + valid() チェック追加 |
| P1-B | ○ 方向性は良い | ○ 方向性は良い | 変更なし |
| P1-C | △ 設計不足 | ○ IDベース | bool→CrossfadeId に変更 |
| P2-A | ○ 妥当 | ○ 妥当 | 変更なし |
| P2-B | ○ 妥当 | ○ 妥当 | 変更なし |
| P2-C | △ Atomic設計に問題 | ○ enum単一 | atomic<struct>→atomic<enum> |
| P3-A | ○ 妥当 | ○ 妥当 | 変更なし |
| P3-B | △ Audit設計を修正推奨 | ○ RingBuffer | vector→FixedRingBuffer<4096>+counter |
| P-1 | — | 新規 | maxPendingRetireObserved |
| P-2 | — | 新規 | Crossfade Timeout 監視 |
| P-3 | — | 新規 | ShutdownBlockingReason |
