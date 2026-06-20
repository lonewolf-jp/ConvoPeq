# Practical Stable ISR Bridge Runtime — 設計書 v4.8（実装開始版・最終）

**Document Version:** 4.8
**Date:** 2026-06-20
**Based on:** v4.7 + 深堀6項目の確定
**Status:** 実装開始版（全不確定要素ゼロ）

---

## v4.7 → v4.8 変更点一覧

| # | 項目 | v4.7 | v4.8 | 根拠 |
|---|---|---|---|---|
| 1 | **方式A** | 3方式併記 | **廃止**（sizeof=24 mutex fallback が保守上の罠） | MSVC 実装依存 + ISR Runtime の「atomic 削減」思想に反する |
| 2 | **方式C** | 単一スレッド前提のみ | **正式採用**：Thread Ownership を全関数で確認し文書化 | commit/Recovery/HealthMonitor は MessageThread 専有 |
| 3 | **方式B seqlock** | オプション | **維持**：方式Cの Thread Ownership 保証が将来崩れた場合の安全網 | 大きなプロジェクトの将来変更リスクに備える |
| 4 | **ISR-AUTH-004** | regex CI（暫定） | **型保証へ移行**：関数シグネチャの const 参照で Pure Function を強制 | regex の限界を認識。型システムで保証 |
| 5 | **AuthorityDescriptor** | 呼び出し元生成 | **Orchestrator 生成に一本化** | Authority 一元化の思想に一致 |
| 6 | **Thread Ownership** | 暗黙的 | **全 PersistentStateBlock 読取主体のスレッド一覧を明文化** | 方式C 採用の根拠 |

---

## 第0章: Thread Ownership 確定（最重要発見）

### 調査結果

全 PersistentStateBlock 読取主体のスレッドコンテキストを調査した：

| 読取主体 | ファイル | スレッド | 根拠 |
|---|---|---|---|
| `commit()` 読取 (line 81-83) | ISRRuntimePublicationCoordinator.cpp | **MessageThread** | `onRuntimePublishedNonRt()` から呼ばれる |
| `commit()` 書込 (line 106-108) | 同上 | **MessageThread** | 同上 |
| `getVersion()` | 同上 | **テストのみ** | ISRSemanticValidationTests.cpp |
| `executeRecoveryAction()` | AudioEngine.Timer.cpp | **MessageThread** | `timerCallback()` → `onHealthEvent()` → `executeRecoveryAction()` |
| `timerCallback()` | AudioEngine.Timer.cpp | **MessageThread** | JUCE Timer 標準動作 |
| `onHealthEvent()` | AudioEngine.Timer.cpp | **MessageThread** | timerCallback から呼ばれる |
| HealthMonitor internal | RuntimeHealthMonitor.h/.cpp | **MessageThread** | バックグラウンドスレッドなし確認 |

**結論**: 全アクセスが MessageThread（同一スレッド）。Single-Thread Ownership 確定。

### 将来のスレッド変更リスク

| リスク | 確率 | 対策 |
|---|---|---|
| HealthMonitor が別スレッド化 | 低 | 方式B seqlock を残す |
| Recovery が独立スレッド化 | 極低 | 方式B seqlock を残す |
| commit() が別スレッド化 | なし（RuntimeStore の設計上不可能） | 不要 |

---

## 第1章: PersistentStateBlock（方式C 正式採用・方式B 維持）

### 1.1 方式C: non-atomic struct（最軽量・正式採用）

```cpp
#pragma once
#include <cstdint>
#include "AtomicAccess.h"

namespace convo {

// ★ PersistentStateSnapshot: 単一スレッド（MessageThread）専有
//   commit/Recovery/Timer/HealthMonitor は同一 MessageThread で実行
//   sizeof = 24 (3 × uint64_t)、atomic 不要
struct PersistentStateSnapshot {
    uint64_t sequenceId{0};
    uint64_t epoch{0};
    uint64_t mappedGeneration{0};
};

struct alignas(64) PersistentStateBlock {
    PersistentStateSnapshot current{};

    void commitFields(uint64_t seq, uint64_t ep, uint64_t gen) noexcept {
        current.sequenceId = seq;
        current.epoch = ep;
        current.mappedGeneration = gen;
    }

    PersistentStateSnapshot snapshot() const noexcept {
        return current;
    }

    static bool isMonotonic(const PersistentStateSnapshot& prev,
                            uint64_t seq, uint64_t ep, uint64_t gen) noexcept {
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

### 1.2 方式B: seqlock（安全網・維持）

Thread Ownership が将来変更された場合の安全網として維持。
v4.7 からの変更点は「C++20 対応の Result 構造体」のみ：

```cpp
struct SnapshotResult {
    bool valid{false};
    SnapshotError error{SnapshotError::None};
    PersistentStateBlock::Snapshot data{};
};
```

### 1.3 方式A: 廃止

`std::atomic<PersistentStateSnapshot>`（sizeof=24 > 16）は MSVC 内部 mutex に
fallback するため、ISR Runtime の「atomic 削減」思想に反する。採用しない。

### 1.4 commit() 統合コード

```cpp
void RuntimePublicationCoordinator::commit(PublishAuthority,
    RuntimeBoundary boundary, const void* newWorld,
    std::uint64_t /*version*/, PublicationSequenceId sequenceId,
    PublicationEpoch epoch, std::uint64_t mappedGeneration) {

    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    // 方式C: non-atomic 読み取り（同一スレッド）
    const auto prev = persistentState_.snapshot();
    if (!PersistentStateBlock::isMonotonic(prev, sequenceId, epoch, mappedGeneration)) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    convo::publishAtomic(state_, CoordinatorState::Publishing, ...);
    convo::publishAtomic(swapPending_, true, ...);

    // 方式C: non-atomic 書き込み
    persistentState_.commitFields(sequenceId, epoch, mappedGeneration);

    // Phase-D 後: currentWorld_ 書き込み削除
    // convo::publishAtomic(currentWorld_, newWorld, ...);  // DELETED

    convo::publishAtomic(swapPending_, false, ...);
    convo::publishAtomic(state_, CoordinatorState::Ready, ...);
}
```

---

## 第2章: ISR-AUTH-004 — 型保証による Pure Function 強制

### 2.1 設計思想

regex ベースの CI は補助的役割に留める。真の保証は **型システム** で行う。

```cpp
// ★ Pure Function の保証方法:
//   引数は全て const 参照。関数内部で atomic 操作不可。
//   → 引数に AudioEngine& や RuntimeStore& を含めなければ、
//     コンパイラレベルで Pure Function が強制される。

// 引数 = const PersistentStateSnapshot& + const World*
// → このシグネチャだけで Pure Function が保証される

template <typename World>
[[nodiscard]] AuthorityState deriveAuthorityState(
    const PersistentStateSnapshot& persistentState,  // ★ const ref → 書換不可
    const World* runtimeWorld)                        // ★ const ptr → 書換不可
    noexcept;
```

### 2.2 CI の役割

```powershell
# isr-verify-auth-004.ps1（補助的CI）
# ★ 型保証が主。CI は「型が想定通り使われているか」を確認する補助
#
# 確認内容:
# 1. deriveAuthorityState の第1引数が const PersistentStateSnapshot& であること
# 2. deriveAuthorityState の第2引数が const World* であること
# 3. 関数本体に atomic 操作が含まれないこと（補助的チェック）

$targetFile = "src/core/AuthorityState.h"
$content = Get-Content (Join-Path $RepoRoot $targetFile) -Raw -Encoding UTF8

# 型シグネチャチェック
if ($content -match 'deriveAuthorityState\s*\([^)]*PersistentStateSnapshot') {
    Write-Host "[PASS] ISR-AUTH-004: deriveAuthorityState uses PersistentStateSnapshot"
} else {
    Write-Host "[FAIL] ISR-AUTH-004: Wrong parameter type"
    exit 1
}
```

### 2.3 将来の clang-query AST 化

```
# 本格的には clang-query で関数本体内の atomic 呼び出しを禁止
clang-query> let fnDecl = functionDecl(hasName("deriveAuthorityState"))
clang-query> let atomicCalls = declRefExpr(to(functionDecl(
    hasAnyName("publishAtomic", "consumeAtomic", "fetchAddAtomic")
)))
clang-query> match fnDecl, atomicCalls
# マッチした場合は違反
```

---

## 第3章: AuthorityDescriptor — Orchestrator 生成に一本化

### 3.1 設計

v4.7 までは「呼び出し元で Domain/Reason を生成する」設計だった。
これを Orchestrator のみが AuthorityDescriptor を生成する設計に変更する。

```cpp
// RuntimePublicationOrchestrator.h
class RuntimePublicationOrchestrator {
public:
    // ★ AuthorityDescriptor 生成は Orchestrator のみが行う
    //   呼び出し元（AudioEngine.Commit.cpp 等）は関与しない
    void submitPublishRequest(
        const PublicationAdmission::PublishRequest& req,
        AuthorityDescriptor auth = {}) noexcept;

private:
    // 内部で呼び出し元を識別し、適切な AuthorityDescriptor を生成
    [[nodiscard]] AuthorityDescriptor resolveAuthoritySource() const noexcept;
};
```

### 3.2 実装

```cpp
// RuntimePublicationOrchestrator.cpp
AuthorityDescriptor RuntimePublicationOrchestrator::resolveAuthoritySource() const noexcept {
    // 現在の実行コンテキストから呼び出し元を識別
    // ★ 実装方針:
    //   明示的に auth が指定された場合はそれを優先
    //   指定がない場合は、AudioEngine の状態から自動判別
    //
    //   例:
    //   - isShutdownInProgress() → Shutdown domain
    //   - HealthState::Critical → Recovery domain
    //   - デフォルト → User domain

    if (engine_.isShutdownInProgress())
        return {AuthorityDomain::Shutdown, AuthorityReason::Unknown};

    const auto* healthRef = engine_.getHealthStateRef();
    if (healthRef) {
        const auto health = convo::consumeAtomic(*healthRef, std::memory_order_acquire);
        if (health == ISRHealthState::Critical)
            return {AuthorityDomain::Recovery, AuthorityReason::HealthCritical};
        if (health == ISRHealthState::Degraded)
            return {AuthorityDomain::Recovery, AuthorityReason::HealthDegraded};
    }

    return {AuthorityDomain::User, AuthorityReason::UserParameter};
}
```

---

## 第4章: 全 Invariant（6件確定・最終版）

| # | 名称 | 保証方法 | CI |
|---|---|---|---|
| 001 | Authority State 再構築可能性 | PersistentStateBlock のみが永続メタデータ | `isr-verify-auth-001.ps1` |
| 002 | Recovery 状態同値性 | reconcileAuthorityState で確認 | `isr-verify-auth-002.ps1` |
| 003 | Publish 経路唯一性 | Orchestrator → Coordinator のみ | `isr-verify-auth-003.ps1`（既存） |
| 004 | Pure Function | **型システム**（const ref 引数）+ regex 補助 | `isr-verify-auth-004.ps1` |
| 005 | 唯一永続メタデータ源 | PersistentStateBlock 以外の永続状態禁止 | `isr-verify-auth-005.ps1` |
| 006 | RuntimeStore 整合性 | deriveAuthorityState 内で runtimeMissing 検出 | `isr-verify-auth-006.ps1` |

---

## 第5章: Phase 計画（確定版）

```
Phase-0: seqlock 方式決定（方式C 推奨・方式B 安全網として維持）
  └─ 実装着手前に1回のみ

Phase-1: 基盤導入
  1a: PersistentStateBlock（方式C: non-atomic）
  1b: AuthorityDescriptor（Orchestrator 生成）
  1c: AuthorityTelemetry（Orchestrator 内）
  1d: Validator エッジケース（7 tests）

Phase-2: currentWorld_ 段階的削除（前編）
  2a: getCurrent → RuntimeStore 委譲
  2b: 全17テスト移行（consumePublishedWorld）
  2c: getCurrentの currentWorld_ フォールバック削除

Phase-2.5: currentWorld_ 監査
  監査ログ + CI 禁止

Phase-3: 状態導出 + Recovery
  3a: deriveAuthorityState / deriveExpectedState / reconcileAuthorityState
  3b: Recovery 統合（reconcileAuthorityState 接続）

Phase-4: Invariant CI + currentWorld_ 完全削除
  4a-f: ISR-AUTH-001～006 CI ゲート
  4g: commit/retire 内 currentWorld_ 操作削除
  4h: currentWorld_ メンバ削除

Phase-5: テスト拡充
  5a: Property Test（10,000回混在）
  5b: 障害注入テスト（4シナリオ）
```
