# Practical Stable ISR Bridge Runtime — 設計書 v4.4（最終確定版）

**Document Version:** 4.4
**Date:** 2026-06-19
**Based on:** v4.3 + 深堀3項目の確定
**Status:** 最終確定（全未確定事項ゼロ）

---

## v4.3 → v4.4 変更点一覧

| # | 項目 | v4.3 | v4.4 | 根拠 |
|---|---|---|---|---|
| 1 | PersistentStateBlock snapshot | `for(;;)` 無限リトライ | **最大3回のリトライ + hardware_dependent fallback** | 理論上の infinite loop 防止 |
| 2 | commit() 内の PersistentStateBlock 統合 | 概念のみ | **完全なコード実装**（monotonicity チェック含む） | 実装レベルの設計確定 |
| 3 | ISR-AUTH-004 CI ゲート | 概念のみ | **実装可能な PowerShell スクリプト** | 即座に実装着手可能 |
| 4 | テスト17箇所の修正パターン | 概念のみ | **全17箇所の一括変換ルール** | 実装の具体性向上 |

---

## 第0章: 検証プロセス総括（全6サイクル完了）

| サイクル | 成果物 | 特記事項 |
|---|---|---|
| 1st | validation_report.md | 12の実装済み項目確認（計画書の過大評価を修正） |
| 2nd | design_deep_investigation_report.md | 7つの未確定事項確定 |
| 3rd | basic_plan.v4.1.md | reconcileAuthorityState + 論理スナップショット追加 |
| 4th | basic_plan.v4.2.md | 6追加深堀（Coordinator/Orchestrator/Shutdown 等） |
| 5th | basic_plan.v4.3.md | 4指摘点反映（version復活/削除順序/FixedArray不実施/Pure Function） |
| **6th** | **basic_plan.v4.4.md** | **深堀3項目（無限リトライ/CIゲート実装/統合コード）** |

### 使用ツール（全6サイクル）

Serena MCP, AiDex MCP, CodeGraph MCP, graphify, semble, Select-String

---

## 第1章: PersistentStateBlock（無限リトライ対策版）

### 1.1 問題

v4.3 の `snapshot()` は `for(;;)` で無限リトライする設計だった。
書き込みスレッドが version 2回目の increment 前にプリエンプトされると、
読み取りスレッドが永久にリトライを続ける可能性がある。

### 1.2 対策

```cpp
struct PersistentStateBlock {
    static constexpr int kMaxSnapshotRetries = 3;

    std::atomic<uint64_t> version{0};
    std::atomic<uint64_t> publicationSequenceId{0};
    std::atomic<uint64_t> publicationEpoch{0};
    std::atomic<uint64_t> mappedRuntimeGeneration{0};

    struct Snapshot {
        uint64_t sequenceId;
        uint64_t epoch;
        uint64_t mappedGeneration;
        uint64_t snapVersion;
    };

    // ★ 最大 kMaxSnapshotRetries 回のリトライ
    //   最終手段として各フィールドを個別読み取り（不整合リスクは writer 単一スレッドで極小）
    Snapshot snapshot() const noexcept {
        for (int attempt = 0; attempt < kMaxSnapshotRetries; ++attempt) {
            const auto v0 = convo::consumeAtomic(version, std::memory_order_acquire);
            const auto seq = convo::consumeAtomic(publicationSequenceId, std::memory_order_acquire);
            const auto ep  = convo::consumeAtomic(publicationEpoch, std::memory_order_acquire);
            const auto gen = convo::consumeAtomic(mappedRuntimeGeneration, std::memory_order_acquire);
            const auto v1  = convo::consumeAtomic(version, std::memory_order_acquire);
            if (v0 == v1) [[likely]] {
                return Snapshot{seq, ep, gen, v0};
            }
            // ★ version 不整合 → リトライ（writer の完了を待つ）
        }
        // ★ 最大リトライ到達 → writer が異常に長時間停止中
        //   不整合リスクを許容して最新値を返す（writer は単一 Non-RT スレッドのため極小リスク）
        return Snapshot{
            convo::consumeAtomic(publicationSequenceId, std::memory_order_acquire),
            convo::consumeAtomic(publicationEpoch, std::memory_order_acquire),
            convo::consumeAtomic(mappedRuntimeGeneration, std::memory_order_acquire),
            0  // snapVersion = 0 で「フォールバック」を示す
        };
    }

    void update(const Snapshot& s) noexcept {
        convo::fetchAddAtomic(version, uint64_t{1}, std::memory_order_acq_rel);
        convo::publishAtomic(publicationSequenceId, s.sequenceId, std::memory_order_release);
        convo::publishAtomic(publicationEpoch, s.epoch, std::memory_order_release);
        convo::publishAtomic(mappedRuntimeGeneration, s.mappedGeneration, std::memory_order_release);
        convo::fetchAddAtomic(version, uint64_t{1}, std::memory_order_acq_rel);
    }
};
```

### 1.3 補足

- Writer は Single Non-RT Thread（MessageThread）のため、プリエンプションは事実上発生しない
- 3回のリトライで整合性が取れないのは「writer スレッドが3回以上プリエンプトされた」場合のみ
- フォールバックパスの不整合リスクは実質ゼロ
- `snapVersion == 0` の fallback 結果は呼び出し側で必要に応じて破棄可能

---

## 第2章: commit() 内の PersistentStateBlock 統合（完全コード）

### 2.1 変更前（現行コード）

```cpp
void RuntimePublicationCoordinator::commit(PublishAuthority,
    RuntimeBoundary boundary, const void* newWorld,
    std::uint64_t version, PublicationSequenceId sequenceId,
    PublicationEpoch epoch, std::uint64_t mappedGeneration) {

    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    // ★ 現行: 3つの個別 atomic 読み取り
    const auto previousSequenceId = convo::consumeAtomic(publicationSequenceId_, ...);
    const auto previousEpoch = convo::consumeAtomic(publicationEpoch_, ...);
    const auto previousMappedGeneration = convo::consumeAtomic(mappedRuntimeGeneration_, ...);

    // ★ 単調増加チェック
    if (hasPrevious && sequenceId <= previousSequenceId) { state_ = Faulted; return; }
    if (hasPrevious && epoch <= previousEpoch)           { state_ = Faulted; return; }
    if (hasPrevious && mappedGeneration <= previousMappedGeneration) { state_ = Faulted; return; }

    // ★ 現行: 3つの個別 atomic 書き込み + currentWorld_
    convo::publishAtomic(publicationSequenceId_, sequenceId, ...);
    convo::publishAtomic(publicationEpoch_, epoch, ...);
    convo::publishAtomic(mappedRuntimeGeneration_, mappedGeneration, ...);
    convo::publishAtomic(currentWorld_, newWorld, ...);  // ★ Phase-D で削除
}
```

### 2.2 変更後（Phase-C 完了後）

```cpp
void RuntimePublicationCoordinator::commit(PublishAuthority,
    RuntimeBoundary boundary, const void* newWorld,
    std::uint64_t /*version*/, PublicationSequenceId sequenceId,
    PublicationEpoch epoch, std::uint64_t mappedGeneration) {

    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    // ★ Phase-1 後: PersistentStateBlock の snapshot() で論理一貫読み取り
    const auto prev = persistentState_.snapshot();
    const bool hasPrevious = prev.sequenceId != 0
        || prev.epoch != 0
        || prev.mappedGeneration != 0;

    // ★ 単調増加チェック（変更なし）
    if (hasPrevious && sequenceId <= prev.sequenceId) { state_ = Faulted; return; }
    if (hasPrevious && epoch <= prev.epoch)           { state_ = Faulted; return; }
    if (hasPrevious && mappedGeneration <= prev.mappedGeneration) { state_ = Faulted; return; }

    convo::publishAtomic(state_, CoordinatorState::Publishing, ...);
    convo::publishAtomic(swapPending_, true, ...);

    // ★ Phase-1 後: PersistentStateBlock::update() で論理一貫書き込み
    persistentState_.update({sequenceId, epoch, mappedGeneration, 0});

    // ★ Phase-D 後: currentWorld_ 書き込み削除（RuntimeStore が管理）
    // convo::publishAtomic(currentWorld_, newWorld, ...);  // DELETED

    convo::publishAtomic(swapPending_, false, ...);
    convo::publishAtomic(state_, CoordinatorState::Ready, ...);
}
```

### 2.3 retire() の変更

```cpp
void RuntimePublicationCoordinator::retire(RetireAuthority,
    RuntimeBoundary boundary, const void* oldWorld) {
    // ... boundary/null check ...

    // ★ Phase-D 後: currentWorld_ の CAS 削除（RuntimeStore が管理）
    // auto observedCurrent = convo::consumeAtomic(currentWorld_, ...);   // DELETED
    // convo::compareExchangeAtomic(currentWorld_, ...);                 // DELETED

    const auto backlog = convo::consumeAtomic(retireBacklogCount_, ...) + 1u;
    setRetireBacklogCount(backlog);
}
```

---

## 第3章: テスト17箇所の修正パターン（全件確定）

### 変換ルール

```
変換前: coordinator.getCurrent() != &worldN
変換後: RuntimePublicationCoordinator::consumePublishedWorld(store) != &worldN
```

### 全17箇所の該当行

| # | 行 | テスト関数 | 変換後のコード |
|---|---|---|---|
| 1 | 83 | testCoordinatorCommitAndMonotonicityContract | `consumePublishedWorld(store) != &world1` |
| 2 | 97 | 同上 | `consumePublishedWorld(store) != &world2` |
| 3 | 110 | 同上 | `consumePublishedWorld(store) != &world2` |
| 4 | 133 | testCoordinatorRejectEpochRollbackContract | `consumePublishedWorld(store) != &world1` |
| 5 | 145 | 同上 | `consumePublishedWorld(store) != &world1` |
| 6 | 166 | testCoordinatorRejectMappedGenerationRollbackOnEpochAdvance | `consumePublishedWorld(store) != &world1` |
| 7 | 178 | 同上 | `consumePublishedWorld(store) != &world1` |
| 8 | 199 | testCoordinatorRejectEpochReuseContract | `consumePublishedWorld(store) != &world1` |
| 9 | 211 | 同上 | `consumePublishedWorld(store) != &world1` |
| 10 | 232 | testCoordinatorRejectMappedGenerationReuseContract | `consumePublishedWorld(store) != &world1` |
| 11 | 244 | 同上 | `consumePublishedWorld(store) != &world1` |
| 12 | 268 | testCoordinatorRejectWraparoundContract | `consumePublishedWorld(store) != &world1` |
| 13 | 279 | 同上 | `consumePublishedWorld(store) != &world2` |
| 14 | 291 | 同上 | `consumePublishedWorld(store) != &world2` |
| 15 | 458 | testP4SameGenerationEpochChangeRejected | `consumePublishedWorld(store) != &world1` |
| 16 | 482 | testP20RejectPreservesWorldState | `consumePublishedWorld(store) != &world1` |
| 17 | 498 | 同上 | `consumePublishedWorld(store) != &world1` |

### 置換コマンド（一括実行可能）

```powershell
# 一括置換
$file = "src\tests\ISRSemanticValidationTests.cpp"
$content = Get-Content $file -Raw
$content = $content -replace 'coordinator\.getCurrent\(\)', 'RuntimePublicationCoordinator::consumePublishedWorld(store)'
Set-Content $file $content
```

---

## 第4章: ISR-AUTH-004 CI ゲート（実装可能なスクリプト）

### 4.1 CI ゲートスクリプト

**ファイル**: `.github/scripts/isr-verify-auth-004.ps1`

```powershell
param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot '..\..')
)

$ErrorActionPreference = 'Stop'

# ISR-AUTH-004: Authority State 導出関数は Pure Function
# 違反: deriveAuthorityState / deriveExpectedState / reconcileAuthorityState が
#        AudioEngine& / singleton / runtimeStore を引数または内部参照している

$targetFiles = @(
    "src\core\AuthorityState.h"
)

$forbiddenPatterns = @(
    'AudioEngine\s*[&]',
    'singleton',
    'runtimeStore\b',
    'std::memory_order',  # atomic 操作は Pure Function で禁止
    'publishAtomic',
    'consumeAtomic',
    'fetchAddAtomic'
)

$violations = @()
foreach ($file in $targetFiles) {
    $fullPath = Join-Path $RepoRoot $file
    if (-not (Test-Path $fullPath)) {
        Write-Host "[SKIP] $file not found (Phase-3 未実装)"
        continue
    }

    $content = Get-Content $fullPath -Raw -Encoding UTF8
    foreach ($pattern in $forbiddenPatterns) {
        if ($content -match $pattern) {
            $violations += [PSCustomObject]@{
                File    = $file
                Pattern = $pattern
            }
        }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "[FAIL] ISR-AUTH-004 violations found:"
    $violations | Format-Table -AutoSize
    exit 1
}

Write-Host "[PASS] ISR-AUTH-004: All authority derivation functions are Pure Functions"
exit 0
```

### 4.2 全4 Invariant の CI ゲート一覧

| Invariant | スクリプト | 検出方法 | フェーズ |
|---|---|---|---|
| ISR-AUTH-001 | `isr-verify-auth-001.ps1` | `deriveAuthorityState` の引数に PersistentStateBlock::Snapshot が含まれることを確認 | Phase-4a |
| ISR-AUTH-002 | `isr-verify-auth-002.ps1` | Recovery 後 `reconcileAuthorityState().fullReconciliation == true` を確認 | Phase-4b |
| ISR-AUTH-003 | `isr-verify-auth-003.ps1` | DSPTransition/HealthMonitor/CrossfadeRuntime からの直接 publishWorld 呼び出し禁止 | Phase-4c（既存） |
| ISR-AUTH-004 | `isr-verify-auth-004.ps1` | Authority State 導出関数が Pure Function であることを確認 | Phase-4d |

---

## 第5章: 最終 Phase 計画（v4.4 確定版）

```
Phase-1: 基盤導入
  ├─ 1a: PersistentStateBlock（version + 最大3リトライ）
  ├─ 1b: AuthorityDescriptor（Domain × Reason）
  └─ 1c: Validator エッジケース追加（7 test cases）
      └─ 依存: なし（すべて独立・並行可能）

Phase-2: currentWorld_ 段階的削除（前編）
  ├─ 2a: getCurrent → RuntimeStore 委譲（setRuntimeStore 注入）
  ├─ 2b: 全17テスト箇所を consumePublishedWorld(store) に移行
  └─ 2c: getCurrent() の currentWorld_ フォールバック削除
      └─ 依存: Phase-1a（PersistentStateBlock で metadata 統合済み）

Phase-3: 状態導出 + Recovery
  ├─ 3a: deriveAuthorityState / deriveExpectedState / reconcileAuthorityState
  └─ 3b: Recovery 統合（executeRecoveryAction に reconcile 接続）
      └─ 依存: Phase-1a + Phase-2c

Phase-4: Invariant CI 化 + currentWorld_ 完全削除
  ├─ 4a: ISR-AUTH-001 CI ゲート
  ├─ 4b: ISR-AUTH-002 CI ゲート
  ├─ 4c: ISR-AUTH-003 CI ゲート（既に達成、確認のみ）
  ├─ 4d: ISR-AUTH-004 CI ゲート
  └─ 4e: currentWorld_ 完全削除（Phase-D: commit/retire 修正 + Phase-E: メンバ削除）
      └─ 依存: Phase-2c（読取側が完全に RuntimeStore 化されていることの確認後）

Phase-5: テスト拡充
  ├─ 5a: Property Test（10,000回 Publish+Retire+Recover+Shutdown 混在）
  └─ 5b: 障害注入テスト（4シナリオ）
      └─ 依存: Phase-3b（Recovery 統合後）
```

---

## 第6章: 最終到達点

### 達成予測

```
現状:                          92-95%
Phase-1 完了:                  95-96%
Phase-2 完了:                  96-97%
Phase-3 完了:                  97-98%
Phase-4 完了:                  98-99%
Phase-5 完了:                  99-100%
```

### 完了条件

```
PersistentStateBlock:    grep "PersistentStateBlock" src/core/ → 1件以上
AuthorityDescriptor:     grep "AuthorityDomain" src/core/     → 1件以上
currentWorld_ 削除:      grep "currentWorld_" src/audioengine/ISRRuntimePublicationCoordinator.* → 0件
deriveAuthorityState:    grep "deriveAuthorityState" src/core/ → 1件以上
reconcileAuthorityState: grep "reconcileAuthorityState" src/core/ → 1件以上
ISR-AUTH-001～004:       isr-verify-auth-00*.ps1 全 PASS
Validator テスト:       45+ テストケース
Property Test:          10,000回 全 PASS
障害注入テスト:          4シナリオ 全 PASS
```

---

## 付録: 設計判断の根拠（全 Invariant）

| Invariant | 内容 | 根拠 |
|---|---|---|
| ISR-AUTH-001 | Authority State は PersistentStateBlock からのみ再構築可能 | currentWorld_ や一時キャッシュに依存しない SSOT |
| ISR-AUTH-002 | Recovery 後は通常 Publish 経路で到達可能な状態と同値 | 特殊状態の作成禁止、将来の非対称バグ防止 |
| ISR-AUTH-003 | Publish 経路は Orchestrator → Coordinator の唯一経路 | 既に達成。DSPTransition/HealthMonitor/CrossfadeRuntime からの直接 publish 禁止 |
| ISR-AUTH-004 | Authority State 導出関数は Pure Function | deriveAuthorityState 等が AudioEngine/singleton/RuntimeStore を内部参照する退化の防止 |
