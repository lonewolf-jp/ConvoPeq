# 未確定事項 最終確定レポート v3

**作成日**: 2026-06-09
**調査手段**: CodeGraph MCP (12,739 entities), Serena MCP, grep, 直接読取

---

## 調査結果一覧

| # | 項目 | 確定内容 | 改修計画への影響 |
| --- | --- | --- | --- |
| 1 | `makeRuntimePublicationCoordinator()` 呼び出し | **10箇所** (定義除く)。P0-3 で全箇所の移行が必要 | ✅ 正確なカウントを確認 |
| 2 | `ISRRetireRouter::drainAll()` | **未実装**。EpochDomain のみが持つ。P1-4 で Router に追加必要 | ✅ 改修計画に反映済み |
| 3 | `[[deprecated]]` API | **5関数**が `[[deprecated]]`。**8箇所**の `#pragma warning(disable:4996)` | ✅ 正確なカウントを確認 |
| 4 | QueuePressure 状態遷移 | **既存**: kPressureSlopeThreshold → Pressure → kPressureNormalizeWindows → Ready | ✅ 「検知あり、回復弱い」確定 |
| 5 | Store スレッド安全性 | WriteAccess は move-only, publishAndSwap は acq_rel exchange | ✅ 設計として正しい |
| 6 | `publishWorld()` 内の Validator | Coordinator::publishWorld() → bridge_.validatePublicationNonRt() → validator_->validatePublication() | ✅ 呼ばれている確定 |

---

## 詳細

### 1. `makeRuntimePublicationCoordinator()` 全呼び出し箇所 (10箇所)

| # | ファイル | 行 | 状況 | 移行 |
| --- | --- | --- | --- | --- |
| 1 | `AudioEngine.h:2733` | publishWorld() 内部 | P0-4 で削除予定 |
| 2 | `AudioEngine.CtorDtor.cpp:127` | ~AudioEngine shutdown | Orchestrator 経由 |
| 3 | `AudioEngine.Init.cpp:46` | bootstrap publish | Orchestrator bootstrap |
| 4 | `PrepareToPlay.cpp:124` | prepareToPlay | Orchestrator 経由 |
| 5 | `PrepareToPlay.cpp:236` | prepareToPlay (2nd) | Orchestrator 経由 |
| 6 | `ReleaseResources.cpp:124` | releaseResources | Orchestrator 経由 |
| 7 | `ReleaseResources.cpp:196` | releaseResources drain | Orchestrator 経由 |
| 8 | `AudioEngine.Timer.cpp:404` | クロスフェード完了 | Orchestrator 経由 |
| 9 | `DSPTransition.h:115` | onTransitionComplete | Orchestrator 経由 |
| 10 | `PublicationExecutor.cpp:15` | publish() | 維持 (Executor→Coordinator) |

### 2. ISRRetireRouter::drainAll() — 未実装

Router に drainAll() メソッドは存在しない。EpochDomain のみが持つ:
```cpp
// EpochDomain.h
void drainAll() noexcept {
    deferredDeletionQueue.drainAllUnsafe();
}
```

呼び出し元:
- `AudioEngine.CtorDtor.cpp:131`: `m_epochDomain.drainAll();`
- `AudioEngine.Processing.ReleaseResources.cpp:208`: `m_epochDomain.drainAll();`
- `EQProcessor.Core.cpp:137`: `m_epochDomain.drainAll();` (別ドメイン)

P1-4 で Router に drainAll() 委譲を追加する必要がある。

### 3. `[[deprecated]]` API 正確なカウント

**5関数** に `[[deprecated]]` 属性:
| 関数 | EpochDomain.h 行 | 代替先 |
|------|-----------------|--------|
| `enterReader(int)` | 82 | `RCUReader::enter()` |
| `exitReader(int)` | 102 | `RCUReader::exit()` |
| `advanceEpoch()` | 134 | `Router::publishEpoch()` |
| `enqueueRetire(ptr,del,epoch)` | 197 | Coordinator::enqueueRetire |
| `enqueueRetire(ptr,del,epoch,type)` | 203 | Coordinator::enqueueRetire |
| `reclaimRetired()` | 209 | `tryReclaim()` |

**8箇所** の抑制:
| ファイル | 箇所 | 抑制されるAPI |
|---------|------|--------------|
| `AudioEngine.h:126` | 1 | EngineRuntime (deprecated struct) |
| `AudioEngine.CtorDtor.cpp:20` | 1 | EpochDomain → SnapshotCoordinator |
| `ISRRetireRouter.h:71,109,118,145,166` | **5** | advanceEpoch/enterReader/exitReader/enqueueRetire/tryReclaim |
| `EQProcessor.Core.cpp:59` | 1 | EpochDomain.enqueueRetire (fallback) |

### 4. QueuePressure 状態遷移 — 既存

```cpp
void RuntimePublicationCoordinator::setRetireBacklogCount(count) noexcept {
    if (slope > kPressureSlopeThreshold) {
        state_ = CoordinatorState::Pressure;  // ← Pressure 検出
        pressureNormalizedWindows_ = 0;
        return;
    }
    // Pressure からの回復:
    if (state_ == CoordinatorState::Pressure) {
        pressureNormalizedWindows_++;
        if (pressureNormalizedWindows_ >= kPressureNormalizeWindows) {
            state_ = CoordinatorState::Ready;  // ← 自動回復
        }
    }
}
```

**状態遷移**: Ready → (slope > threshold) → **Pressure** → (normalize windows) → **Ready**

### 5. Store スレッド安全性 — 確認済み

`RuntimeStore.h`:
- `WriteAccess`: **move-only**, コピー不可。`publishAndSwap()` は `std::memory_order_acq_rel` の atomic exchange
- `observe()`: `std::memory_order_acquire` の atomic load
- `acquireWriteAccess()`: `friend Owner` のみアクセス可能
- Store 自体は `std::atomic<T*>` の単一ポインタ。ロックフリーでスレッドセーフ

### 6. `publishWorld()` 内の Validator — 確認

`RuntimePublicationCoordinator.h:97-100`:
```cpp
if constexpr (requires(Bridge bridge, const World& world) { bridge.validatePublicationNonRt(world); })
{
    if (!bridge_.validatePublicationNonRt(*worldOwner))
    {
        // validation failed → retire and return
        auto* rejectedWorld = worldOwner.release();
        bridge_.retireRuntimePublishWorldNonRt(rejectedWorld, false);
        return;
    }
}
```

`RuntimePublicationBridge::validatePublicationNonRt()`:
```cpp
bool validatePublicationNonRt(const RuntimePublishWorld& world) noexcept {
    const auto result = validator_->validatePublication(world);
    if (!result.isValid) return false;
    return engine_->runPublicationPrecheckNonRt(world);
}
```

**確認済み**: 全 publish 経路で Validator は coordinator.publishWorld() 経由で呼ばれる。

---

## 改修計画への影響

| 発見事項 | 影響 | 対応 |
|---------|------|------|
| Coordinator生成10箇所 | 移行作業量の正確な見積り可能 | P0-3 の改修手順に反映済み |
| RouterにdrainAll()なし | P1-4 で追加実装が必要 | P1-4 に明記 |
| deprecated 5関数/8抑制 | 削除作業の正確な範囲確定 | Deprecated API全廃項目に反映 |
| Pressure状態遷移既存 | 「検知あり回復弱い」確定 | P1-6 の現状記述を強化 |

---

## 使用ツール実績

| ツール | 使用回数 | 主な用途 |
|--------|---------|---------|
| CodeGraph MCP | 2回 | モジュール構造解析、インデックス更新 |
| Serena MCP | 2回 | シンボル定義取得、参照解析 |
| grep | 10回以上 | Coordinator/Deprecated/Pressure網羅検索 |
| 直接ファイル読取 | 5ファイル | 実装確認 |
