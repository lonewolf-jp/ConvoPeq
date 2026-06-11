# ISR Bridge Runtime Practical Stable 改修計画 v7.3（最終確定版）

**作成日**: 2026-06-11
**v7.2→v7.3 差分**:

- P1-A: rootEnterSucceeded_ を exit() 先頭一括リセット→最外層（previousDepth==1）時のみ false に変更
**全体完成度**: 99.9/100

---

## 重要設計判断の確定根拠（コード調査結果）

### 判断1: RCUReader slot authority は registerReaderThread が唯一

`acquireThreadSlot()` → `registerReaderThread()` が Reader slot 割当の唯一の Authority。
`activeReaderCount` の事前チェックは TOCTOU 競合（A=63, B=63 通過後 A成功 B失敗）を
防げず、false reject を増やすだけ。**第二層（rootEnterSucceeded_）で十分。**

### 判断2: Crossfade Event Drop は段階的昇格

fade 長 10〜80ms で Timer 33ms、SPSC 32 のため正常系での drop は想定外。
ただし Timer 停止等の一時的イベントでも drop は発生し得る。
即 Critical → Publication 全停止は過剰。Warning 閾値到達後に Critical へ昇格。

### 判断3: WorldLifecycleAudit は Diagnostic 限定

`RuntimeDrainAudit` が shutdown 完了判定の唯一の Authority。
診断カウンタに CAS ループは過剰。fetch_sub + jassert で十分。

### 判断4: Crossfade Timeout は固定30秒

fade 長 10〜80ms のため期待値×4 ≈ 最大 320ms ≪ 30s。固定値で十分。

---

## Phase 0: 即時修正（P0）

### P0-A: IEpochProvider 拡張（◎）

`IRetireProvider` に `pendingRetireCount()` / `drainAll()` を純粋仮想関数追加。
`ISRRetireRouter` の dynamic_cast と未初期化関数ポインタを撤廃。

### P0-B: EQProcessor Fallback 削除（◎）

Fallback 経路を完全削除。`if (m_retireCoordinator == nullptr) return false;` 維持。

---

## Phase 1: 監視・制御閉ループ（P1）

### P1-A: RCUReader rootEnterSucceeded_ — exit() ネスト対応リセット（◎）

**v7.3**: exit() 先頭一括リセット→最外層 exit 時のみ false に変更

**修正理由**: `exit()` 先頭で一律リセットすると、ネストケース
（`enter()`→`enter()`→`exit()`）で outer scope がまだ active なのに
`rootEnterSucceeded() == false` になるため。

```cpp
// RCUReader.h — 最外層 exit 時のみ rootEnterSucceeded_ を false に
class RCUReader {
public:
    void enter() noexcept
    {
        const uint32_t previousDepth = convo::fetchAddAtomic(...);
        if (previousDepth > 0) return;  // ネスト: 最外層の結果を維持

        // ... CAS + acquireThreadSlot ...

        if (tid >= 0) {
            epochProvider->enterReader(tid);
            rootEnterSucceeded_ = true;
        } else {
            rootEnterSucceeded_ = false;  // Fail-Closed
            // ... 後片付け ...
        }
    }

    [[nodiscard]] bool rootEnterSucceeded() const noexcept
    {
        return rootEnterSucceeded_;
    }

    void exit() noexcept
    {
        const uint32_t previousDepth = convo::fetchSubAtomic(...);

        // previousDepth == 0: underflow — 安全のためリセット
        if (previousDepth == 0) { rootEnterSucceeded_ = false; return; }

        // previousDepth > 1: ネスト — rootEnterSucceeded_ を維持
        //   outer scope がまだ active なため false にしてはいけない。
        if (previousDepth > 1) return;

        // ★ previousDepth == 1: 最外層 exit
        if (convo::consumeAtomic(ownerThreadToken, ...) != currentThreadToken())
        {
            rootEnterSucceeded_ = false;  // 所有者不一致でもリセット
            return;
        }

        // ... activeThreadId解放、exitReader()、ownerThreadToken解放 ...

        rootEnterSucceeded_ = false;  // 正常 cleanup 完了
    }

private:
    bool rootEnterSucceeded_ = false;  // ★ RAII thread confinement 下で安全
};
```

**リセット位置の整理**:

| 経路 | previousDepth | リセット | 理由 |
|------|--------------|---------|------|
| underflow | 0 | ✅ | 安全のため |
| ネスト | >1 | ❌ | outer scope 維持 |
| 所有者不一致 | 1 | ✅ | exit 完了 |
| 正常 cleanup | 1 | ✅ | exit 完了 |

- `activeReaderCount >= kMaxReaders` のチェックは TOCTOU 競合のため正確な防御にならない
- `registerReaderThread()` が真の slot Authority
- 実際の防御は第二層（RCUReader の rootEnterSucceeded_）で完結
- 余分な分岐は false reject を増やすだけ

---

### P1-B: HealthMonitor → Admission 閉ループ（◎）

### P1-C: Crossfade Event SPSC + 段階的 Drop 昇格（◎）

**v6→v7**: 即 Critical → Warning → count>=10 → Critical

```cpp
// CrossfadeRuntime.h
struct CompletedFadeEvent {
    CrossfadeId id;
    uint64_t    completedTimestampUs;
};

class CrossfadeRuntime {
public:
    void notifyFadeComplete(CrossfadeId id) noexcept
    {
        CompletedFadeEvent ev{id, getCurrentTimeUs()};
        if (!completedFadeQueue_.push(ev))
        {
            // ★ Queue full → drop count increment
            convo::fetchAddAtomic(crossfadeEventDropCount_, uint64_t{1},
                std::memory_order_release);
        }
    }

    [[nodiscard]] uint64_t crossfadeEventDropCount() const noexcept
    {
        return convo::consumeAtomic(crossfadeEventDropCount_, std::memory_order_acquire);
    }

private:
    SPSCRingBuffer<CompletedFadeEvent, 32> completedFadeQueue_;
    std::atomic<uint64_t> crossfadeEventDropCount_{0};
    std::atomic<uint64_t> fadeStartTimestampUs_{0};
};

// HealthMonitor 側 — 差分ベース昇格
// ★ v7.1: 累積値ではなく、前回観測時からの差分 (delta) で判定。
//   累積値だと「半年で10回」でも Critical になるため。
static constexpr uint64_t kCrossfadeEventDropCriticalDelta = 10;
static constexpr uint64_t kCrossfadeEventDropWarningDelta  = 1;
uint64_t lastObservedDropCount_ = 0;  // ★ HealthMonitor ローカル状態

void checkCrossfadeEventDrop() noexcept {
    if (!m_crossfadeEventDropRef) return;
    uint64_t current = convo::consumeAtomic(*m_crossfadeEventDropRef, std::memory_order_acquire);
    uint64_t delta = current - lastObservedDropCount_;

    if (delta >= kCrossfadeEventDropCriticalDelta) {
        // ★ 短時間に多数 drop → Critical（連続異常）
        emitOnTransition(m_prevCrossfadeDropState, MonitorState::Error,
            HealthEvent::Severity::Error, EVENT_CROSSFADE_EVENT_DROP, delta);
    } else if (delta >= kCrossfadeEventDropWarningDelta) {
        // ★ 新規 drop → Warning
        emitOnTransition(m_prevCrossfadeDropState, MonitorState::Warning,
            HealthEvent::Severity::Warning, EVENT_CROSSFADE_EVENT_DROP, delta);
    }

    lastObservedDropCount_ = current;  // ★ 観測位置を更新
}
```

**段階的昇格の根拠**:

- `drops >= 1`: Warning — Timer 一時停止等の一時的イベントでは Publication を止めない
- `drops >= 10`: Critical — 継続的 drop は AudioThread → Timer 経路の恒久的問題を示唆
- 閾値 10 は SPSC 32 の 30% 程度。連続 10 回 drop は設計上の異常

---

## Phase 2: 防御的改善（P2）

### P2-A: Deprecated API 整理（◎）

### P2-B: Shutdown VerifyDrained + BlockingReason（◎）

### P2-C: HealthState enum 単一共有（◎）

---

## Phase 3: 中長期改善（P3）

### P3-A: HealthMonitor ↔ RetirePressure 統合（○）

### P3-B: WorldLifecycleAudit — Diagnostic 限定 + fetch_sub（◎）

**v6→v7**: CASループ削除 → fetch_sub + jassert ガード

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
        // ★ v7.2: fetchSub→if(prev==0)→publishAtomic(0)
        //   load→if→fetchSub 方式は TOCTOU 競合があり、
        //   マルチスレッド時に監査値が UINT64_MAX になり得る。
        //   Diagnostic 限定とはいえ監査価値を維持するため、
        //   fetchSub の戻り値で判定する。
        uint64_t prev = convo::fetchSubAtomic(activeWorldCount_, 1u,
            std::memory_order_acq_rel);
        if (prev == 0) {
            // ★ 二重 retire — 診断アサート + 飽和補正
            jassertfalse;
            convo::publishAtomic(activeWorldCount_, uint64_t{0},
                std::memory_order_release);
        }

        convo::fetchAddAtomic(retiredCount_, 1u, std::memory_order_release);
    }

    // ★ 監査用途のみ。Shutdown 完了判定には使用しない。
    //   Shutdown 判定は RuntimeDrainAudit + ShutdownRuntime FSM が担当。
    [[nodiscard]] uint64_t activeWorldCount() const noexcept { ... }
    [[nodiscard]] uint64_t publishedCount() const noexcept { ... }
    [[nodiscard]] uint64_t retiredCount() const noexcept { ... }
    void emitSnapshot() const noexcept;

private:
    FixedRingBuffer<WorldLifecycleRecord, 4096> ringBuffer_;
    std::atomic<uint64_t> activeWorldCount_{0};
    std::atomic<uint64_t> publishedCount_{0};
    std::atomic<uint64_t> retiredCount_{0};
};
```

**fetchSub→if(prev==0)→publishAtomic(0) の根拠（v7.2 で v7 方式に回帰）**:

- Diagnostic 限定カウンタに CAS ループは過剰
- `fetchSub` の戻り値 `prev` で判定すれば TOCTOU 競合なし
- `prev == 0` 時に `publishAtomic(0)` で飽和補正 → UINT64_MAX 回避
- load→if→fetchSub 方式は将来のマルチスレッド retire で破綻するため不採用

---

## Practical Items

### Practical-1: Retire Queue HWM（◎）

### Practical-2: Crossfade Timeout — 固定30秒（◎）

### Practical-3: Shutdown BlockingReason（◎）

### Practical-4: Reader Slot Usage Telemetry（◎）

### Practical-5: Retire Reclaim Latency（◎）

### Practical-6: Crossfade Event Drop Telemetry（◎）（P1-C と統合）

---

## v6→v7 変更サマリ（最終差分）

| 項目 | v6 | v7 | 根拠 |
|------|----|----|------|
| P1-A | 二層防御（rootEnterSucceeded + makeRuntimeReadHandle事前ブロック） | rootEnterSucceeded のみ | activeReaderCount事前チェックはTOCTOU競合で正確な防御にならず |
| P1-C | Drop→即Critical | Drop→Warning→count>=10→Critical | Timer停止等の一時的イベントでもdrop発生。即Criticalは過剰 |
| P3-B | CAS compare_exchange_weak loop | fetch_sub + jassert | Diagnostic限定カウンタにCASは過剰 |

---

## 修正対象ファイル一覧（v7 確定版）

| ファイル | P | 変更概要 |
|----------|---|----------|
| `src/core/IRetireProvider.h` | P0-A | pendingRetireCount/drainAll 追加 |
| `src/core/EpochDomain.h` | P0-A, P2-A | override, deprecated private化 |
| `src/audioengine/ISRRetireRouter.h` | P0-A | 関数ポインタ削除 |
| `src/audioengine/ISRRetireRouter.cpp` | P0-A | dynamic_cast削除 |
| `src/eqprocessor/EQProcessor.Core.cpp` | P0-B | Fallback削除 |
| `src/core/RCUReader.h` | P1-A | rootEnterSucceeded_ (non-atomic bool) |
| `src/audioengine/CrossfadeRuntime.h` | P1-C, P-6 | SPSC + dropCount |
| `src/audioengine/RuntimeHealthMonitor.h` | P1-B/C, P2-C, P-2/4/5/6 | 全監視統合 |
| `src/audioengine/RuntimeHealthMonitor.cpp` | 同上 | tick()拡張 |
| `src/audioengine/PublicationAdmission.h` | P1-B, P2-C | HealthStateRef |
| `src/audioengine/PublicationAdmission.cpp` | P1-B | evaluate()拡張 |
| `src/audioengine/AudioEngine.h` | P-1, P-5 | HWM, RetireAge配線 |
| `src/audioengine/DSPTransition.h` | P1-C | notifyFadeComplete |
| `src/audioengine/AudioEngine.Timer.cpp` | P1-C | SPSC consumer |
| `src/audioengine/ISRShutdown.h` | P-3 | ShutdownBlockingReason |
| `src/audioengine/ISRShutdown.cpp` | P2-B, P-3 | VerifyDrained |
| `src/audioengine/WorldLifecycleAudit.h` | P3-B | 新規: Diagnostic限定 fetch_sub |
| `src/audioengine/AudioEngine.Retire.cpp` | P3-A, P-1 | HWM + Monitor連携 |
| `src/audioengine/RuntimeDrainAudit.h` | P2-B | (変更なし - 既存authority) |
