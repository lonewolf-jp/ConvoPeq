# ISR Bridge Runtime Practical Stable 改修計画 v6（最終版）

**作成日**: 2026-06-11
**v5→v6 差分**:

- P1-A: makeRuntimeReadHandle() レベルで activeReaderCount 事前チェック追加（二層防御）
- P1-C: SPSC drop → HealthState::Critical（Warning→Critical 昇格）
- P-2: expectedFadeDurationUs_ 削除 → 固定30秒（現在のfade長は10-80msで常に30秒相当）
- P3-B: WorldLifecycleAudit を Diagnostic 限定に変更。Shutdown 判定は既存 RuntimeDrainAudit

---

## 優先度定義

| 優先度 | 定義 |
|--------|------|
| **P0** | Practical Stable に必須。未対処で運用破綻リスク |
| **P1** | 長時間運用の安定性に寄与。監視・制御の閉ループ化 |
| **P2** | 防御的改善。現状でも運用可能だが強化が望ましい |
| **P3** | 中長期的改善。Phase-C/D 以降の検討事項 |

---

# 重要設計判断（コード調査結果に基づく）

## 判断1: WorldLifecycleAudit は Diagnostic 限定

**既存コードの事実**:

- `RuntimeDrainAudit.h` の冒頭コメント:

  ```
  // isAllZero() は監査ログ出力専用。shutdown 完了判定の authority にはしない。
  ```

- Shutdown 完了条件は `ShutdownRuntime` の `VerifyDrained` phase が管理
- `RuntimeDrainAudit` は監査ログ出力のみ（shutdown 判定に未使用）

**v6 方針**: WorldLifecycleAudit は `Diagnostic` 扱いに限定。
`activeWorldCount_` の shutdown 判定接続は行わない。
Shutdown 判断は引き続き既存の `RuntimeDrainAudit` + `ShutdownRuntime` FSM が担当。

## 判断2: Crossfade Event Drop → HealthState::Critical

**既存コードの事実**:

- ConvoPeq の crossfade duration 設定: 最大 `m_irFadeTimeSec` = 80ms（最小 10ms）
- Timer 周期: JUCE default ~30Hz ≈ 33ms
- SPSC 容量: 32

正常動作では 1 Timer 周期内に 32 個の fade 完了が発生する想定はない。
したがって drop 発生 = 設計上の異常。Warning ではなく Critical が適切。

## 判断3: makeRuntimeReadHandle レベルでの二層防御

RCUReader Fail-Closed を二層で実現:

- 第一層（Authority）: `makeRuntimeReadHandle()` が `activeReaderCount >= kMaxReaders` を事前チェック
- 第二層（Safety）: RCUReader の `rootEnterSucceeded_`（non-atomic bool）で per-enter 成否を追跡

## 判断4: Crossfade Timeout は固定30秒で十分

fade 時間は 10〜80ms の範囲で、`*4` しても 320ms。実効的には常に `max(320ms, 30s)` = 30s。
`expectedFadeDurationUs_` の保存コストに対してリターンが極めて小さいため固定値で十分。

---

## Phase 0: 即時修正（P0）

### P0-A: IEpochProvider 拡張（◎）

### P0-B: EQProcessor Fallback 削除（◎）

---

## Phase 1: 監視・制御閉ループ（P1）

### P1-A: RCUReader Fail-Closed — 二層防御（○）

**第一層: makeRuntimeReadHandle() での事前チェック**

```cpp
// AudioEngine.h の makeRuntimeReadHandle() 内
[[nodiscard]] inline RuntimeReadHandle makeRuntimeReadHandle(
    const convo::RuntimeReaderContext& ctx) noexcept
{
    // ★ P1-A: 第一層防御 — Reader slot 枯渇時は観測を生成しない
    //   activeReaderCount >= kMaxReaders なら Fail-Closed。
    //   ObservedRuntime を作らず、呼び出し側は空の handle を受ける。
    if (m_retireRouter->activeReaderCount() >= EpochDomain::kMaxReaders)
    {
        // ★ Reader slot 枯渇 — HealthMonitor が監視（Practical-4）
        return {};  // 空の RuntimeReadHandle
    }

    switch (ctx.channel) { ... }
    // ... 既存ロジック ...
}
```

**第二層: RCUReader per-enter tracking**

```cpp
// RCUReader.h — enter() 内で slot 取得失敗時に設定
class RCUReader {
public:
    void enter() noexcept
    {
        const uint32_t previousDepth = convo::fetchAddAtomic(...);
        if (previousDepth > 0)
        {
            // ネスト: 最外層の結果を維持
            return;
        }

        // ... CAS + acquireThreadSlot ...

        if (tid >= 0)
        {
            epochProvider->enterReader(tid);
            rootEnterSucceeded_ = true;
        }
        else
        {
            rootEnterSucceeded_ = false;  // ★ 第二層防御
            // ... 後片付け ...
        }
    }

    [[nodiscard]] bool rootEnterSucceeded() const noexcept
    {
        return rootEnterSucceeded_;
    }

private:
    bool rootEnterSucceeded_ = false;  // ★ non-atomic: thread confined
};
```

**二層防御の動作**:

| シナリオ | 第一層 | 第二層 | 結果 |
|----------|--------|--------|------|
| 正常時 | pass | pass | 通常観測 |
| slot 枯渇（事前検出） | **block** | — | handle 空、ObservedRuntime 未生成 |
| slot 枯渇（競合） | pass | **fail** | handle 生成されるが ptr==null |
| 部分回復後 retry | 次回は pass | 次の enter で再試行 | 正常復帰可能 |

---

### P1-B: HealthMonitor → Admission 閉ループ（○）

### P1-C: Crossfade Event SPSC + Drop → Critical（○）

**v5→v6修正**: drop を Warning から Critical に昇格

```cpp
void notifyFadeComplete(CrossfadeId id) noexcept
{
    CompletedFadeEvent ev{id, getCurrentTimeUs()};
    if (!completedFadeQueue_.push(ev))
    {
        convo::fetchAddAtomic(crossfadeEventDropCount_, uint64_t{1},
            std::memory_order_release);
        // ★ drop 発生 = 正常系では想定外 → HealthMonitor で Critical 通知
    }
}

// HealthMonitor 側
void checkCrossfadeEventDrop() noexcept {
    if (!m_crossfadeEventDropRef) return;
    uint64_t drops = convo::consumeAtomic(*m_crossfadeEventDropRef, std::memory_order_acquire);
    if (drops > 0) {
        emitOnTransition(m_prevCrossfadeDropState, MonitorState::Error,  // ★ Warning→Error
            HealthEvent::Severity::Error, EVENT_CROSSFADE_EVENT_DROP, drops);
        // ★ HealthState → Critical 昇格（Admission が publish ブロック）
    }
}
```

**SPSC 設計判断の根拠**: 現在の CrossfadeRuntime は完了通知経路を持たない。
P0-C 設計図で「CrossfadeFinishedEvent FIFO」を想定しており、SPSC 導入は設計と一致する。
Queue overflow / reset / lifecycle の管理負荷は極めて低い（正常系で overflow は発生しない想定のため）。

---

## Phase 2: 防御的改善（P2）

### P2-A: Deprecated API 整理（◎）

### P2-B: Shutdown VerifyDrained + BlockingReason（○）

### P2-C: HealthState enum 単一共有（○）

---

## Phase 3: 中長期改善（P3）

### P3-A: HealthMonitor ↔ RetirePressure 統合（○）

### P3-B: WorldLifecycleAudit — Diagnostic 限定（△→○）

**v5→v6 変更点**:

- `allWorldsRetired()` を削除（Shutdown 判定に接続しない）
- 監査目的の `FixedRingBuffer<4096>` + カウンタのみ維持
- Shutdown 判定は既存 `RuntimeDrainAudit` + `ShutdownRuntime` FSM が担当

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
        // CAS ループで安全に減算
        uint64_t current = convo::consumeAtomic(activeWorldCount_, std::memory_order_acquire);
        while (true) {
            if (current == 0) { jassertfalse; return; }
            if (convo::compareExchangeWeakAtomic(activeWorldCount_,
                    current, current - 1,
                    std::memory_order_acq_rel, std::memory_order_acquire))
                break;
        }
        convo::fetchAddAtomic(retiredCount_, 1u, std::memory_order_release);
    }

    // ★ 監査用途のみ。Shutdown 完了判定には使用しない。
    // Shutdown 判定は RuntimeDrainAudit + ShutdownRuntime FSM が担当。
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
```

**Diagnostic 限定とする根拠**:

- `RuntimeDrainAudit.h` に既に明記: 監査ログ出力専用、shutdown 完了判定の authority ではない
- `ShutdownRuntime` の `VerifyDrained` phase + `RuntimeDrainAudit` が shutdown 判定の source-of-truth
- WorldLifecycleAudit は診断補助としての価値に徹する

---

## Practical-1: Retire Queue HWM（◎）

## Practical-2: Crossfade Timeout — 固定30秒（○）

**v5→v6**: `expectedFadeDurationUs_` を削除。コード調査の結果、
crossfade duration は 10〜80ms であり、期待値×4 は最大 320ms ≪ 30秒のため。
固定 `constexpr uint64_t kCrossfadeTimeoutUs = 30'000'000;` で十分。

```cpp
// CrossfadeRuntime.h
std::atomic<uint64_t> fadeStartTimestampUs_{0};

// HealthMonitor 側
static constexpr uint64_t kCrossfadeTimeoutUs = 30'000'000;  // 30秒固定

void checkCrossfadeTimeout() noexcept {
    if (!m_crossfadeRuntime) return;
    if (!m_crossfadeRuntime->isPending()) return;
    uint64_t ageUs = m_crossfadeRuntime->getFadeAgeUs();
    if (ageUs > kCrossfadeTimeoutUs) {
        emitOnTransition(m_prevCrossfadeState, MonitorState::Error,
            HealthEvent::Severity::Error, EVENT_CROSSFADE_TIMEOUT, ageUs / 1000);
    }
}
```

## Practical-3: Shutdown BlockingReason（◎）

## Practical-4: Reader Slot Usage Telemetry（○）

## Practical-5: Retire Reclaim Latency（○）

## Practical-6: Crossfade Event Drop Telemetry（○）

---

## v5→v6 変更サマリ

| 項目 | v5 | v6 | v6 方針 |
|------|----|----|---------|
| P1-A | ○ RCUReader per-enter | ◎ | makeRuntimeReadHandle で事前チェック追加（二層防御） |
| P1-C | ○ drop→Warning | ◎ | drop→Critical 昇格 |
| P-2 | ○ expectedFadeDurationUs | ◎ | 固定30秒（調査: 10-80msのため常に30s相当） |
| P3-B | △ activeWorldCount shutdown接続 | ◎ | Diagnostic限定、RuntimeDrainAuditがauthority |

---

## 修正対象ファイル一覧（v6 最終）

| ファイル | P | 変更概要 |
|----------|---|----------|
| `src/core/IRetireProvider.h` | P0-A | pendingRetireCount/drainAll 追加 |
| `src/core/EpochDomain.h` | P0-A, P2-A | override, deprecated private化 |
| `src/audioengine/ISRRetireRouter.h` | P0-A | 関数ポインタ削除 |
| `src/audioengine/ISRRetireRouter.cpp` | P0-A | dynamic_cast削除 |
| `src/eqprocessor/EQProcessor.Core.cpp` | P0-B | Fallback削除 |
| `src/core/RCUReader.h` | P1-A | rootEnterSucceeded_ (non-atomic bool) |
| `src/audioengine/AudioEngine.h` | P1-A | makeRuntimeReadHandle 事前チェック |
| `src/audioengine/CrossfadeRuntime.h` | P1-C, P-6 | SPSC + dropCount |
| `src/audioengine/RuntimeHealthMonitor.h` | 全P1/P2/Prac | 監視統合 |
| `src/audioengine/RuntimeHealthMonitor.cpp` | 同上 | tick()拡張 |
| `src/audioengine/PublicationAdmission.h` | P1-B, P2-C | HealthStateRef |
| `src/audioengine/PublicationAdmission.cpp` | P1-B | evaluate()拡張 |
| `src/audioengine/DSPTransition.h` | P1-C | notifyFadeComplete |
| `src/audioengine/AudioEngine.Timer.cpp` | P1-C | SPSC consumer |
| `src/audioengine/ISRShutdown.h` | P-3 | ShutdownBlockingReason |
| `src/audioengine/ISRShutdown.cpp` | P2-B, P-3 | VerifyDrained |
| `src/audioengine/WorldLifecycleAudit.h` | P3-B | 新規: Diagnostic限定 |
| `src/audioengine/AudioEngine.Retire.cpp` | P3-A, P-1 | HWM + Monitor連携 |
| `src/audioengine/RuntimeDrainAudit.h` | P2-B | (変更なし - 既存authority) |
