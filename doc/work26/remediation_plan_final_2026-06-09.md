# 未確定事項 最終確定レポート v2

**作成日**: 2026-06-09
**調査手段**: CodeGraph MCP (12,739 entities), Serena MCP, grep, 直接読取

---

## 調査結果

| # | 項目 | 確定内容 | 補正要否 |
| --- | --- | --- | :---: |
| 1 | DSPTransition の Coordinator 生成 | `engine_.makeRuntimePublicationCoordinator()` を呼ぶ | ✅ 改修計画に反映 |
| 2 | markShutdownComplete / isFullyDrained | **存在する。** 6条件をチェック、Faulted/Bootstrapping 遷移 | ✅ 既存の正しい指摘 |
| 3 | SafeStateSwapper kMaxRetired | **256** (推定64は誤り) | ⚠️ 軽微な数値誤差 |
| 4 | RuntimeDrainAudit 構造 | 7フィールド + BlockingReason 列挙 + isAllZero() | ✅ 確認済み |
| 5 | EQProcessor retireCoordinator | `setRetireCoordinator()` で設定、設定漏れ時は EpochDomain 直接 | ✅ 既存の正しい指摘 |
| 6 | RuntimePublicationBridge構造 | validate→didPublish→willRetire→retireの4メソッド | ✅ 確認済み |
| 7 | CrossfadeAuthority | dspProjection投影値のみで判断。DSPCore直読なし | ✅ 確認済み |

---

## 詳細

### 1. DSPTransition の Coordinator 生成 — 確認

`DSPTransition.h:115`:

```cpp
auto coordinator = engine_.makeRuntimePublicationCoordinator();
auto worldBuilder = convo::RuntimeBuilder(engine_);
auto worldOwner = worldBuilder.buildRuntimePublishWorld(currentAfterFade, nullptr,
    convo::TransitionPolicy::HardReset, 0.0, false);
if (worldOwner) {
    coordinator.publishWorld(std::move(worldOwner));
}
```

DSPTransition は Orchestrator のメンバ (`transition_`) であり、`Orchestrator::trySubmit()` 内から `transition_.onPublishCompleted()` 経由で呼ばれる。しかしここで再び `makeRuntimePublicationCoordinator()` を呼び、さらに publish している。

**問題**: この publish は Orchestrator の制御外で行われ、Admission チェックを経由しない (Coordinator 直接経路 = 経路C)。

→ **改修計画への反映が必要**: DSPTransition::onTransitionComplete() を Orchestrator 経由に変更。

### 2. markShutdownComplete / isFullyDrained — 確認 (重要)

`ISRRuntimePublicationCoordinator.cpp:257`:

```cpp
bool RuntimePublicationCoordinator::isFullyDrained() const noexcept {
    if (swapPending_) return false;
    return retireBacklogCount_ == 0
        && publicationBacklogCount_ == 0
        && pendingIntentCount_ == 0
        && fallbackBacklogCount_ == 0
        && reclaimInFlightCount_ == 0
        && deferredRetireResidencyCount_ == 0;
}
```

`ISRRuntimePublicationCoordinator.cpp:298`:

```cpp
void RuntimePublicationCoordinator::markShutdownComplete() noexcept {
    if (state != ShuttingDown) return;
    if (isFullyDrained())
        state_ = Bootstrapping;
    else
        state_ = Faulted;  // ← Drain 不完全を検出
}
```

`AudioEngine::isFullyDrained()` はさらに Orchestrator の deferred commit もチェック:

```cpp
bool AudioEngine::isFullyDrained() noexcept {
    const bool hasDeferredCommit = runtimeOrchestrator_->hasDeferredRequest();
    runtimePublicationBridge_.setPendingIntentCount(hasDeferredCommit ? 1u : 0u);
    // ... backlog counts を bridge に反映 ...
    return !hasDeferredCommit && runtimePublicationBridge_.isFullyDrained();
}
```

**結論**: `isFullyDrained()` + `markShutdownComplete()` の Drain 判定機構は**既に実装済み**。不足は Evidence 連携と自動実行。

### 3. SafeStateSwapper kMaxRetired = 256

`SafeStateSwapper.h:55`:

```cpp
static constexpr size_t kMaxRetired = 256;
```

**補正**: 改修計画書で「推定64」と書いていたが実際は256。256エントリで溢れることは稀。

### 4. RuntimeDrainAudit 構造

`RuntimeDrainAudit.h`:

```cpp
struct RuntimeDrainAudit {
    uint64_t pendingPublication;
    uint64_t pendingRetire;
    uint64_t activeCrossfadeCount;
    uint64_t routerPendingRetire;
    uint64_t maxDeferredAgeMs;
    uint64_t deferredPublish;
    uint64_t quarantineResident;
    uint64_t oldestPendingAgeMs;
    uint64_t maxQuarantineAgeSec;

    enum class BlockingReason : uint8_t {
        None, PendingPublication, PendingRetire, ActiveCrossfade,
        DeferredPublish, QuarantineResident, RouterPendingRetire, Unknown
    };

    BlockingReason getPrimaryBlockingReason() const noexcept;
    bool isAllZero() const noexcept;
};
```

確認済み。既に詳細な監査構造体が実装されている。

### 5. RuntimePublicationBridge 構造

`AudioEngine.h` (RuntimePublicationBridge):

```cpp
class RuntimePublicationBridge final {
    bool validatePublicationNonRt(const RuntimePublishWorld& world) noexcept {
        const auto result = validator_->validatePublication(world);
        if (!result.isValid) return false;
        return engine_->runPublicationPrecheckNonRt(world);
    }
    void didPublishRuntimeNonRt(const RuntimePublishWorld& world) noexcept;
    void willRetireRuntimeNonRt(const RuntimePublishWorld* world) noexcept;
    void retireRuntimePublishWorldNonRt(RuntimePublishWorld* world, bool resetRevision) noexcept;
};
```

Bridge は Coordinator に注入され、`publishWorld()` 内で validate → publish → retire を実行する。設計としては正しい。

### 6. CrossfadeAuthority

`CrossfadeAuthority.h`:

```cpp
class CrossfadeAuthority {
    struct Decision { bool needsCrossfade; bool oldHasIR; bool newHasIR; double fadeTimeSec; };
    Decision evaluate(const AudioEngine&, const RuntimePublishWorld& old, const RuntimePublishWorld& newWorld) noexcept;
};
```

Orchestrator::trySubmit() 内で呼ばれる:

```cpp
auto cfDecision = crossfade.evaluate(engine_, *oldWorld, *worldOwner);
```

DSPCore 直読なし。dspProjection 投影値のみで判断。設計として正しい。

---

## 改修計画への補正反映

| 計画書の記述 | 補正 | 影響 |
| --- | --- | --- |
| SafeStateSwapper kMaxRetired = 64(推定) | **kMaxRetired = 256** | 影響なし (数値訂正) |
| DSPTransition の publish 経路 | **DSPTransition::onTransitionComplete() も要移行** | 改修手順に追加必要 |
| isFullyDrained() / markShutdownComplete() | **既に実装済み** | P2-1 の優先度を再確認 |

### DSPTransition の移行を改修計画に追加

**P0-2 (Admission bypass除去) の改修手順に追加**:

```
6. DSPTransition::onTransitionComplete() の Coordinator 直接生成を Orchestrator 経由に変更
   - notifyTransitionComplete() が既に Orchestrator に存在
   - onTransitionComplete 内の publish を Orchestrator::notifyTransitionComplete() 経由に変更
```

---

## 総合確定

以上で未確定事項は全て解消した。改修計画書の全体構造 (P0〜P3 の 4フェーズ 17項目) は妥当であり、SafeStateSwapper の数値誤差 (64→256) と DSPTransition の移行追加のみが補正点。
