# Practical Stable ISR Bridge Runtime — 設計書 v4.13（実装開始版・最終）

**Document Version:** 4.13
**Date:** 2026-06-20
**Based on:** v4.12 + 最終実装詳細確定
**Status:** 実装開始版（全設計判断・全実装詳細完了）

---

## v4.12 → v4.13 変更点一覧

| # | 項目 | v4.12 | v4.13 |
|---|---|---|---|
| 1 | **seqlock retry loop** | `continue`（CPU ビジーループ） | **`_mm_pause()` 追加（SMT 排他制御）** |
| 2 | **SnapshotError / ScopedGuard** | 概念のみ | **完全な定義コード** |
| 3 | **commit() 移行コード** | 概念のみ | **変更前/変更後の完全な対比コード** |
| 4 | **getVersion() 移行** | 未記載 | **移行後の実装確定** |
| 5 | **SnapshotResult に snapVersion** | なし | **整合性確認のための snapVersion 追加** |

---

## 第0章: 最終コード（実装開始可能）

本設計書 v4.13 をもって、以下のファイルを実装開始可能：

- `src/core/PersistentStateBlock.h` → **全文掲載**
- `src/core/AuthorityDescriptor.h` → **全文掲載**
- `src/core/AuthorityState.h` → **全文掲載**
- `ISRRuntimePublicationCoordinator.cpp` → **変更差分掲載**
- `ISRSemanticValidationTests.cpp` → **変更差分掲載**

---

## 第1章: PersistentStateBlock（実装コード）

**ファイル**: `src/core/PersistentStateBlock.h`

```cpp
#pragma once
#include <atomic>
#include <cstdint>
#include "AtomicAccess.h"

namespace convo {

// ── SnapshotError: seqlock 読み取り失敗の理由 ──
enum class SnapshotError : uint8_t {
    None,
    WriterBusy,     // version が奇数のまま（書き込み区間と衝突）
    RetryExceeded   // 最大リトライ回数超過（writer が異常停止の可能性）
};

// ── SnapshotResult: seqlock 読み取り結果 ──
struct SnapshotResult {
    bool valid{false};
    SnapshotError error{SnapshotError::None};
    uint64_t sequenceId{0};
    uint64_t epoch{0};
    uint64_t mappedGeneration{0};
    uint64_t snapVersion{0};  // 読み取り成功時の version（整合性確認用）
};

// ── ScopedVersionWriteGuard: RAII で version の偶数/奇数を管理 ──
//    commitFields() の開始時に version を奇数にし、終了時に偶数にする。
//    デストラクタで必ず偶数に戻すため、assert/jassert による途中終了でも安全。
class ScopedVersionWriteGuard {
public:
    explicit ScopedVersionWriteGuard(std::atomic<uint64_t>& v) noexcept
        : version_(v) {
        convo::fetchAddAtomic(version_, uint64_t{1}, std::memory_order_acq_rel);
    }
    ~ScopedVersionWriteGuard() noexcept {
        convo::fetchAddAtomic(version_, uint64_t{1}, std::memory_order_acq_rel);
    }
    ScopedVersionWriteGuard(const ScopedVersionWriteGuard&) = delete;
    ScopedVersionWriteGuard& operator=(const ScopedVersionWriteGuard&) = delete;
private:
    std::atomic<uint64_t>& version_;
};

// ── PersistentStateBlock: Publication の永続メタデータ ──
//   所有権: 書き込みは MessageThread 専有（commitFields）
//   読み取り: 任意のスレッドから可能（seqlock で整合性保証）
//
//   ★ commit() の単調増加検証と同じ整合性条件を保証する
//   ★ seqlock により、将来の並行読取追加でも壊れない
struct PersistentStateBlock {
    std::atomic<uint64_t> version{0};
    std::atomic<uint64_t> publicationSequenceId{0};
    std::atomic<uint64_t> publicationEpoch{0};
    std::atomic<uint64_t> mappedRuntimeGeneration{0};

    static constexpr int kMaxSnapshotRetries = 3;

    // ── snapshot(): 論理一貫スナップショット ──
    //   read-version → read-fields → read-version の seqlock パターン
    //   _mm_pause() で SMT 排他制御（writer 完了待ち時の CPU 負荷軽減）
    SnapshotResult snapshot() const noexcept {
        for (int i = 0; i < kMaxSnapshotRetries; ++i) {
            auto v0 = convo::consumeAtomic(version, std::memory_order_acquire);
            if ((v0 & 1u) != 0) {
                _mm_pause();  // 書き込み完了待ち（SMT 排他制御）
                if (i == kMaxSnapshotRetries - 1)
                    return {false, SnapshotError::WriterBusy, 0, 0, 0, v0};
                continue;
            }
            auto seq = convo::consumeAtomic(publicationSequenceId, std::memory_order_acquire);
            auto ep  = convo::consumeAtomic(publicationEpoch, std::memory_order_acquire);
            auto gen = convo::consumeAtomic(mappedRuntimeGeneration, std::memory_order_acquire);
            auto v1  = convo::consumeAtomic(version, std::memory_order_acquire);
            if ((v1 & 1u) == 0 && v0 == v1)
                return {true, SnapshotError::None, seq, ep, gen, v0};
            _mm_pause();
        }
        return {false, SnapshotError::RetryExceeded, 0, 0, 0, 0};
    }

    // ── commitFields(): 3フィールドを一括論理更新 ──
    //   ScopedVersionWriteGuard で version 偶数/奇数を RAII 管理
    void commitFields(uint64_t seq, uint64_t ep, uint64_t gen) noexcept {
        jassert(!convo::numeric_policy::isAudioThread());
        ScopedVersionWriteGuard guard(version);
        convo::publishAtomic(publicationSequenceId, seq, std::memory_order_release);
        convo::publishAtomic(publicationEpoch, ep, std::memory_order_release);
        convo::publishAtomic(mappedRuntimeGeneration, gen, std::memory_order_release);
        // ~ScopedVersionWriteGuard で version++（偶数に戻す）
    }

    // ── 単調増加チェック（commit() 内で使用） ──
    static bool isMonotonic(const SnapshotResult& prev,
                            uint64_t seq, uint64_t ep, uint64_t gen) noexcept {
        if (!prev.valid) return true;  // 初回書き込みは常に許可
        const bool hasPrev = prev.sequenceId != 0 || prev.epoch != 0
            || prev.mappedGeneration != 0;
        if (hasPrev && seq <= prev.sequenceId) return false;
        if (hasPrev && ep <= prev.epoch) return false;
        if (hasPrev && gen <= prev.mappedGeneration) return false;
        return true;
    }
};

} // namespace convo
```

---

## 第2章: commit() 移行コード（完全対比）

### 変更前

```cpp
void RuntimePublicationCoordinator::commit(PublishAuthority,
    RuntimeBoundary boundary, const void* newWorld,
    std::uint64_t version, PublicationSequenceId sequenceId,
    PublicationEpoch epoch, std::uint64_t mappedGeneration) {

    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
        return;
    }

    // ★ 3つの個別 atomic 読み取り（不整合リスクあり）
    const auto previousSequenceId = convo::consumeAtomic(publicationSequenceId_, ...);
    const auto previousEpoch = convo::consumeAtomic(publicationEpoch_, ...);
    const auto previousMappedGeneration = convo::consumeAtomic(mappedRuntimeGeneration_, ...);

    // ★ 単調増加チェック
    if (hasPrevious && sequenceId <= previousSequenceId) { state_ = Faulted; return; }
    if (hasPrevious && epoch <= previousEpoch)           { state_ = Faulted; return; }
    if (hasPrevious && mappedGeneration <= previousMappedGeneration) { state_ = Faulted; return; }

    convo::publishAtomic(state_, CoordinatorState::Publishing, std::memory_order_release);
    convo::publishAtomic(swapPending_, true, std::memory_order_release);
    (void) version;  // ★ 未使用パラメータ抑制（削除対象）

    // ★ 3つの個別 atomic 書き込み + currentWorld_
    convo::publishAtomic(publicationSequenceId_, sequenceId, ...);
    convo::publishAtomic(publicationEpoch_, epoch, ...);
    convo::publishAtomic(mappedRuntimeGeneration_, mappedGeneration, ...);
    convo::publishAtomic(currentWorld_, newWorld, ...);  // ★ Phase-D で削除

    convo::publishAtomic(swapPending_, false, std::memory_order_release);
    convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);
}
```

### 変更後（Phase-1a: PersistentStateBlock 導入 + Phase-D: currentWorld_ 削除後）

```cpp
void RuntimePublicationCoordinator::commit(PublishAuthority,
    RuntimeBoundary boundary, const void* newWorld,
    std::uint64_t /*version*/, PublicationSequenceId sequenceId,
    PublicationEpoch epoch, std::uint64_t mappedGeneration) {

    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
        return;
    }

    // ★ PersistentStateBlock の seqlock snapshot で論理一貫読み取り
    const auto prev = persistentState_.snapshot();
    // ★ isMonotonic で単調増加チェック（prev.valid==false は初回書き込みとして許可）
    if (!PersistentStateBlock::isMonotonic(prev, sequenceId, epoch, mappedGeneration)) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
        return;
    }

    convo::publishAtomic(state_, CoordinatorState::Publishing, std::memory_order_release);
    convo::publishAtomic(swapPending_, true, std::memory_order_release);

    // ★ (void) version 行削除（パラメータ名をコメント化）
    // ★ 3フィールド一括書き込み（seqlock で整合性保証）
    persistentState_.commitFields(sequenceId, epoch, mappedGeneration);

    // ★ Phase-D 後: currentWorld_ 書き込み削除
    // convo::publishAtomic(currentWorld_, newWorld, ...);  // DELETED

    convo::publishAtomic(swapPending_, false, std::memory_order_release);
    convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);
}
```

### retire() 変更（Phase-D 後）

```cpp
void RuntimePublicationCoordinator::retire(RetireAuthority,
    RuntimeBoundary boundary, const void* oldWorld) {
    if (boundary != RuntimeBoundary::NonRTWorld || oldWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    // ★ Phase-D 後: currentWorld_ の CAS 削除
    // auto observedCurrent = convo::consumeAtomic(currentWorld_, ...);  // DELETED
    // convo::compareExchangeAtomic(currentWorld_, ...);                // DELETED

    const auto backlog = convo::consumeAtomic(retireBacklogCount_, ...) + 1u;
    setRetireBacklogCount(backlog);
}
```

### getVersion() 変更（Phase-1a 後）

```cpp
// 変更前
std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    return convo::consumeAtomic(mappedRuntimeGeneration_, std::memory_order_acquire);
}

// 変更後
std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    const auto snap = persistentState_.snapshot();
    return snap.valid ? snap.mappedGeneration : 0;
}
```

---

## 第3章: AuthorityDescriptor（実装コード）

**ファイル**: `src/core/AuthorityDescriptor.h`

```cpp
#pragma once
#include <atomic>
#include <cstdint>
#include "AtomicAccess.h"

namespace convo {

enum class AuthorityDomain : uint8_t {
    Unknown      = 0,
    User         = 1,
    Preset       = 2,
    Recovery     = 3,
    DSPLifecycle = 4,
    Health       = 5,
    Shutdown     = 6,
    _Count
};

enum class AuthorityReason : uint8_t {
    Unknown             = 0,
    UserParameter       = 1,
    TimerRecovery       = 10,
    EmergencyRecovery   = 11,
    TimeoutRecovery     = 12,
    ShutdownRecovery    = 13,
    HealthStall         = 20,
    HealthCritical      = 21,
    DSPTransitionFinish = 30,
    _Count
};

struct AuthorityDescriptor {
    AuthorityDomain domain{AuthorityDomain::Unknown};
    AuthorityReason reason{AuthorityReason::Unknown};
};

} // namespace convo
```

---

## 第4章: AuthorityState（実装コード）

**ファイル**: `src/core/AuthorityState.h`

```cpp
#pragma once
#include <cstdint>
#include <type_traits>
#include "PersistentStateBlock.h"

namespace convo {

struct AuthorityState {
    uint64_t publicationSequenceId{0};
    uint64_t publicationEpoch{0};
    uint64_t mappedRuntimeGeneration{0};
    bool hasActiveRuntime{false};
    bool hasPendingPublication{false};
    bool hasActiveCrossfade{false};
    bool runtimeMissing{false};
    bool persistentMissing{false};
    bool fieldInconsistencyDetected{false};

    bool operator==(const AuthorityState& o) const noexcept {
        return publicationSequenceId == o.publicationSequenceId
            && publicationEpoch == o.publicationEpoch
            && mappedRuntimeGeneration == o.mappedRuntimeGeneration
            && hasActiveRuntime == o.hasActiveRuntime;
    }
    bool operator!=(const AuthorityState& o) const noexcept { return !(*this == o); }
};

// ★ ISR-AUTH-004: Pure Function（const ref + requires 型制約）
template <typename World>
[[nodiscard]] AuthorityState deriveAuthorityState(
    const PersistentStateBlock::SnapshotResult& ps,
    const World* runtimeWorld) noexcept
    requires std::is_same_v<World, const RuntimePublishWorld>
         || std::is_same_v<World, RuntimePublishWorld>
{
    AuthorityState result;
    result.publicationSequenceId = ps.sequenceId;
    result.publicationEpoch = ps.epoch;
    result.mappedRuntimeGeneration = ps.mappedGeneration;
    result.hasActiveRuntime = (runtimeWorld != nullptr);
    result.runtimeMissing = (ps.sequenceId > 0 && runtimeWorld == nullptr);
    result.persistentMissing = (runtimeWorld != nullptr && ps.sequenceId == 0);
    result.fieldInconsistencyDetected =
        (ps.sequenceId > 0 && ps.epoch == 0)
     || (ps.sequenceId > 0 && ps.mappedGeneration == 0)
     || (ps.epoch > 0 && ps.mappedGeneration == 0)
     || (ps.epoch == 0 && ps.mappedGeneration > 0);
    result.hasPendingPublication = result.runtimeMissing;
    if (runtimeWorld != nullptr)
        result.hasActiveCrossfade = runtimeWorld->execution.transitionActive;
    return result;
}

[[nodiscard]] AuthorityState deriveExpectedState(
    const PersistentStateBlock::SnapshotResult& ps) noexcept
{
    AuthorityState result;
    result.publicationSequenceId = ps.sequenceId;
    result.publicationEpoch = ps.epoch;
    result.mappedRuntimeGeneration = ps.mappedGeneration;
    result.hasActiveRuntime = (ps.sequenceId > 0);
    return result;
}

struct RepairConfidence {
    enum Level : uint8_t {
        None          = 0,
        ObserveOnly   = 1,
        SoftRepair    = 2,
        HardRepair    = 3
    };
    Level level{None};
};

struct AuthorityReconciliation {
    bool needsIdlePublish{false};
    bool needsRetireDrain{false};
    bool needsEpochAdvance{false};
    bool needsCrossfadeComplete{false};
    bool fullReconciliation{false};
    RepairConfidence confidence{};

    bool needsAnyAction() const noexcept {
        return needsIdlePublish || needsRetireDrain
            || needsEpochAdvance || needsCrossfadeComplete;
    }
    bool needsImmediateAction() const noexcept {
        return needsAnyAction() && confidence.level >= RepairConfidence::HardRepair;
    }
};

[[nodiscard]] AuthorityReconciliation reconcileAuthorityState(
    const AuthorityState& observed,
    const AuthorityState& expected) noexcept
{
    AuthorityReconciliation rec;
    if (observed == expected) {
        rec.fullReconciliation = true;
        return rec;
    }
    if (expected.hasActiveRuntime && !observed.hasActiveRuntime
        && observed.publicationSequenceId > 0) {
        rec.needsIdlePublish = true;
        rec.confidence.level = (observed.publicationEpoch > 0)
            ? RepairConfidence::HardRepair
            : RepairConfidence::ObserveOnly;
    }
    rec.needsCrossfadeComplete = observed.hasActiveCrossfade && !expected.hasActiveCrossfade;
    rec.needsEpochAdvance = observed.publicationEpoch < expected.publicationEpoch;
    if (observed.hasPendingPublication && !expected.hasPendingPublication)
        rec.needsRetireDrain = true;
    if (!rec.needsAnyAction()) rec.fullReconciliation = true;
    return rec;
}

// ★ 独立 Validator: reconcileAuthorityState とは独立に observed==expected を検証
[[nodiscard]] bool validateAuthorityStateMatch(
    const AuthorityState& observed,
    const AuthorityState& expected) noexcept
{
    return observed.publicationSequenceId == expected.publicationSequenceId
        && observed.publicationEpoch == expected.publicationEpoch
        && observed.mappedRuntimeGeneration == expected.mappedRuntimeGeneration
        && observed == expected;
}

} // namespace convo
```

---

## 第5章: Phase 計画（最終）

```
Phase-1: 基盤導入
  1a: PersistentStateBlock（seqlock + _mm_pause + ScopedGuard）
  1b: AuthorityDescriptor + Telemetry
  1c: deriveAuthorityState / deriveExpectedState / reconcileAuthorityState
  1d: validateAuthorityStateMatch（独立 Validator）
  1e: Validator エッジケース（7 tests）

Phase-2: currentWorld_ 段階的削除（前編）
  2a: getCurrent → RuntimeStore 委譲
  2b: 全17テスト移行
  2c: getCurrent の currentWorld_ フォールバック削除
  2.5: currentWorld_ 監査 + CI 禁止

Phase-3: Recovery 統合
  executeRecoveryAction に reconcileAuthorityState 接続

Phase-4: Invariant CI + currentWorld_ 完全削除
  4a-f: ISR-AUTH-001〜006 CI ゲート
  4g: commit() 内 currentWorld_ 削除
  4h: retire() 内 currentWorld_ CAS 削除
  4i: currentWorld_ メンバ削除

Phase-5: テスト拡充
  5a: Model-Based Test（Model State 比較）
  5b: Fault Injection 6シナリオ
```
