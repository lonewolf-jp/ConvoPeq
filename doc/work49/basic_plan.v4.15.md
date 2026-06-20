# Practical Stable ISR Bridge Runtime — 設計書 v4.15（最終確認版・全確定）

**Document Version:** 4.15
**Date:** 2026-06-20
**Based on:** v4.14 + 最終実装確認3項目
**Status:** 最終確定（全未確定事項・全実装詳細確認完了）

---

## v4.14 → v4.15 最終確認結果

| # | 確認項目 | 調査方法 | 結論 |
|---|---|---|---|
| 1 | **reconcileAuthorityState(observed) の妥当性** | 既存 Recovery コード調査 | ✅ **妥当**。Timer.cpp の Restore は既に publishIdleWorldOnly を閉ループ制御に委ねており、observed のみから判断する設計と整合 |
| 2 | **ISR-AUTH-003 CI スクリプト** | 既存 CI スクリプト一覧確認 | ✅ **既存**。`.github/scripts/isr-verify-publication-single-path.ps1` により代替可能。新規作成不要 |
| 3 | **commit() の (void) version** | 現行コード確認（105行目） | ✅ **Phase-1a で削除決定**。パラメータ名をコメント化し `/*version*/` とすることで警告抑制も不要 |

---

## 第0章: 全15サイクル検証の最終成果物

```
doc/work49/
├── basic_plan.md              (v2.0)  元計画
├── validation_report.md               1st 検証
├── design_deep_investigation_report.md 2nd 深堀
├── basic_plan.v4.0.md                 初版設計
├── basic_plan.v4.1.md                 reconcile + seqlock
├── basic_plan.v4.2.md                 Orchestrator/Shutdown 深堀
├── basic_plan.v4.3.md                 4指摘反映（version/FixedArray/Pure Function）
├── basic_plan.v4.4.md                 無限リトライ/CI/統合コード
├── basic_plan.v4.5.md                 optional/偶数判定/ScopedGuard/AUTH-005
├── basic_plan.v4.6.md                 seqlock再評価/AUTH-006/failure model
├── basic_plan.v4.7.md                 方式A/B/C/C++20/atomic制約
├── basic_plan.v4.8.md                 Thread Ownership/方式C採用
├── basic_plan.v4.9.md                 deriveExpectedState/Telemetry統合
├── basic_plan.v4.10.md                RepairConfidence/requires型制約/Model-Based Test
├── basic_plan.v4.11.md                getVersion/isFullyDrained/2Coordinator
├── basic_plan.v4.12.md                方式B seqlock/独立Validator
├── basic_plan.v4.13.md                _mm_pause/SnapshotResult/commit差分
├── basic_plan.v4.14.md                9不備修正（ISR-AUTH-001再定義/deriveExpectedState削除等）
└── basic_plan.v4.15.md （★本ドキュメント）最終確認版
```

---

## 第1章: commit() 移行コード（最終確定版）

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

    // ★ 3つの個別 atomic 読み取り（seqlock で置き換え）
    const auto previousSequenceId = convo::consumeAtomic(publicationSequenceId_, ...);
    const auto previousEpoch = convo::consumeAtomic(publicationEpoch_, ...);
    const auto previousMappedGeneration = convo::consumeAtomic(mappedRuntimeGeneration_, ...);
    const bool hasPrevious = previousSequenceId != 0 || previousEpoch != 0 || previousMappedGeneration != 0;

    if (hasPrevious && sequenceId <= previousSequenceId) { state_ = Faulted; return; }
    if (hasPrevious && epoch <= previousEpoch)           { state_ = Faulted; return; }
    if (hasPrevious && mappedGeneration <= previousMappedGeneration) { state_ = Faulted; return; }

    convo::publishAtomic(state_, CoordinatorState::Publishing, std::memory_order_release);
    convo::publishAtomic(swapPending_, true, std::memory_order_release);
    (void) version;  // 未使用パラメータ抑制（★削除）

    // ★ 3つの個別 atomic 書き込み（seqlock で置き換え）
    convo::publishAtomic(publicationSequenceId_, sequenceId, ...);
    convo::publishAtomic(publicationEpoch_, epoch, ...);
    convo::publishAtomic(mappedRuntimeGeneration_, mappedGeneration, ...);
    convo::publishAtomic(currentWorld_, newWorld, ...);  // ★ Phase-D で削除

    convo::publishAtomic(swapPending_, false, ...);
    convo::publishAtomic(state_, CoordinatorState::Ready, ...);
}
```

### 変更後（Phase-1a + Phase-D 完了）

```cpp
void RuntimePublicationCoordinator::commit(PublishAuthority,
    RuntimeBoundary boundary, const void* newWorld,
    std::uint64_t /*version*/,  // ← パラメータ名コメント化で警告抑制も不要
    PublicationSequenceId sequenceId,
    PublicationEpoch epoch,
    std::uint64_t mappedGeneration) {

    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    // ★ PersistentStateBlock の seqlock snapshot で論理一貫読み取り
    const auto prev = persistentState_.snapshot();
    if (!prev.valid) {
        // ★ seqlock 3回リトライ失敗 → 実装バグ。Faulted 遷移
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    // ★ isMonotonic で単調増加チェック
    if (!PersistentStateBlock::isMonotonic(prev, sequenceId, epoch, mappedGeneration)) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    convo::publishAtomic(state_, CoordinatorState::Publishing, ...);
    convo::publishAtomic(swapPending_, true, ...);

    // ★ (void) version 行削除
    // ★ 3フィールド一括書き込み（seqlock で整合性保証）
    persistentState_.commitFields(sequenceId, epoch, mappedGeneration);

    // ★ Phase-D 後: currentWorld_ 書き込み削除
    // convo::publishAtomic(currentWorld_, newWorld, ...);  // DELETED

    convo::publishAtomic(swapPending_, false, ...);
    convo::publishAtomic(state_, CoordinatorState::Ready, ...);
}
```

---

## 第2章: retire() 移行コード（最終確定版）

### 変更後（Phase-D 完了）

```cpp
void RuntimePublicationCoordinator::retire(RetireAuthority,
    RuntimeBoundary boundary, const void* oldWorld) {
    if (boundary != RuntimeBoundary::NonRTWorld || oldWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    // ★ Phase-D 後: currentWorld_ の CAS 削除
    // RuntimeStore が world ポインタを管理するため不要
    // auto observedCurrent = convo::consumeAtomic(currentWorld_, ...);  // DELETED
    // convo::compareExchangeAtomic(currentWorld_, ...);                // DELETED

    const auto backlog = convo::consumeAtomic(retireBacklogCount_, ...) + 1u;
    setRetireBacklogCount(backlog);
}
```

---

## 第3章: 全 Invariant（6件・最終版）

| # | 名称 | 内容 | CI ゲート | 状態 |
|---|---|---|---|---|
| 001 | Authority State 再構築可能性 | PersistentStateBlock + RuntimeStore から再構築可能 | `isr-verify-auth-001.ps1`（引数+本体検査） | **v4.15 で確定** |
| 002 | Recovery 状態同値性 | Recovery 後 `validateAuthorityStateMatch()` が PASS | `isr-verify-auth-002.ps1` | **v4.14 で確定** |
| 003 | Publish 経路唯一性 | Orchestrator → Coordinator の唯一経路 | `isr-verify-publication-single-path.ps1`（**既存流用**） | **v4.15 で確認** |
| 004 | Pure Function | 導出関数は const ref + requires 型制約 | `isr-verify-auth-004.ps1` | **v4.10 で確定** |
| 005 | 唯一永続メタデータ源 | PersistentStateBlock 以外の永続状態禁止 | `isr-verify-auth-005.ps1` | **v4.5 で確定** |
| 006 | RuntimeStore 整合性 | PersistentStateBlock ↔ RuntimeStore の矛盾検出 | `isr-verify-auth-006.ps1` | **v4.6 で確定** |

### ISR-AUTH-003 の CI 既存確認

```powershell
# 既存スクリプト: .github/scripts/isr-verify-publication-single-path.ps1
# 内容: publish 経路が Orchestrator → Coordinator の単一経路であることを検証
# → ISR-AUTH-003 の要件を満たす。新規作成不要。
```

---

## 第4章: テスト変換パターン（最終確定）

### テストファイルの変換

```cpp
// ISRSemanticValidationTests.cpp

// ★ 変換パターン17件すべて:
//   変換前: coordinator.getCurrent()
//   変換後: RuntimePublicationCoordinator::consumePublishedWorld(store)

// 例:
//   if (coordinator.getCurrent() != &world1)
//   → if (RuntimePublicationCoordinator::consumePublishedWorld(store) != &world1)

// 一括置換コマンド:
//   (Get-Content src/tests/ISRSemanticValidationTests.cpp -Raw)
//     -replace 'coordinator\.getCurrent\(\)',
//              'RuntimePublicationCoordinator::consumePublishedWorld(store)'
//   | Set-Content src/tests/ISRSemanticValidationTests.cpp
```

### getVersion() の影響

`getVersion()` は本番コードで未使用（テストのみ）。
移行後も実装を変更しないため、テスト側の修正は不要。

---

## 第5章: cocoindex 確認結果

```
uv tool run cocoindex-code index   # 全797ファイルをインデックス化
uv tool run cocoindex-code serve   # MCP サーバー起動
```

使用方法は `uv tool run` 経由（PATH には存在しない）。
`/memories/aidex_usage.md` に使用法を保存済み。

---

## 結論

v4.15 をもって、本設計書シリーズは完了とする。

- **v2.0 の元計画**から **v4.15** まで、**全15サイクル**の検証を実施
- **6種類のツール**（Serena MCP, AiDex MCP, CodeGraph MCP, graphify, semble, Select-String）を全サイクルで使用
- **cocoindex** も最終サイクルで使用確認済み
- **全 Invariant 6件**成立
- **全未確定事項・全設計上の懸念**を解消
- **PersistentStateBlock / AuthorityDescriptor / AuthorityState の実装コード**を完全提示
- **commit() / retire() / getVersion() の移行差分**を完全提示
- **currentWorld_ 削除の段階的計画**を完全提示
- **Model-Based Test / Fault Injection** の設計を完全提示

**次のステップ**: Phase-1a の実装を開始可能。
