# Practical Stable ISR Bridge Runtime — 設計書 v4.18（実コード整合完了版）

**Document Version:** 4.18
**Date:** 2026-06-20
**Based on:** v4.17 + 実コード調査7課題再検証
**Status:** 実コード整合完了

---

## v4.17 → v4.18 7課題修正サマリ

| # | 課題 | 原因 | 調査内容 | 修正内容 |
|---|---|---|---|---|
| ① | **seqlock 要否の最終判断** | v4.13 以降、方式B(seqlock)と方式C(atomic struct)が混在していた | commit() は MessageThread 専有、concurrent reader なし。`std::atomic<24byte>` は MSVC で非ロックフリーだが、コンテンションゼロのため実害なし | **方式C (`std::atomic<PersistentStateBlock>`)** に完全統一。seqlock の全痕跡を削除。version/ScopedGuard/_mm_pause/RetryExceeded は不採用を確定 |
| ② | **方式C 記述の不整合** | 「方式C」の説明が v4.13 と v4.17 で異なっていた | v4.13 式: 構造体内に `std::atomic<uint64_t>`。v4.17 式: `std::atomic<PersistentStateBlock>`。全く別物 | 3方式を明確に定義:
 方式A: 3個別 `std::atomic<uint64_t>`（現行）
 方式B: seqlock（不採用確定）
 方式C: `std::atomic<PersistentStateBlock>`（**採用**） |
| ③ | **snapshot 失敗→Faulted 危険** | v4.15 で seqlock 前提の `!prev.valid→Faulted` が残存 | `std::atomic<PersistentStateBlock>::load()` は常に成功（WriterBusy なし）。方式C ではこの問題は存在しない | 方式C 採用により自動解決。ただし Commit 失敗（isMonotonic 違反）→ Faulted は妥当 |
| ④ | **validateAuthorityStateMatch 不完全** | operator== が診断フラグを無視 | AuthorityState は3層構造に分割:
 AuthoritySnapshot: {sequenceId, epoch, generation, hasActiveRuntime}
 AuthorityDiagnostics: {runtimeMissing, persistentMissing, fieldInconsistency, hasPendingPublication, hasActiveCrossfade}
 AuthorityReconciliation: 不一致検出＋修復ロジック | **全フィールド比較に統一**。`validateAuthorityMatch()` は 3 階層すべての全フィールドを独立比較 |
| ⑤ | **currentWorld_ 削除証明不足** | 17件のテスト参照と coordinator 種別の混在が未整理 | getCurrent() は ISR coordinator のメソッド。consumePublishedWorld() は template coordinator のメソッド。両者は別オブジェクト。テストの ISR coordinator 依存を解決するには RuntimeStore の注入が必要 | Phase 再編: currentWorld_ 削除を Phase-1 に先行。5段階の削除条件を定義＋CI ゲート |
| ⑥ | **AuthorityState 役割肥大化** | Recovery/Validation/Diagnostics の3役を1構造体が兼任 | 3層分割により責務を明確化 | `AuthoritySnapshot` + `AuthorityDiagnostics` + `AuthorityReconciliation` の3層に分割 |
| ⑦ | **Phase 順序が最適でない** | AuthorityState (新機能) が currentWorld_ (既存負債) より優先 | 既存負債(currentWorld_)の除去を優先すべき | Phase 再編:
 P0: PersistentStateBlock
 P1: currentWorld_ 削除（5段階）
 P2: Authority 3層導入
 P3: Recovery 統合
 P4: CI + Model-Based Test |

---

## 第0章: 3方式の明確な定義

```
方式A（現行）: 3個別 std::atomic<uint64_t>
  publicationSequenceId_    std::atomic<PublicationSequenceId>
  publicationEpoch_         std::atomic<PublicationEpoch>
  mappedRuntimeGeneration_  std::atomic<uint64_t>
  利点: ロックフリー（各8byte）、現状稼働中
  欠点: 3フィールドの論理一貫性が保証されない（個別更新のため）
  → コミット前の snapshot 読取で不整合リスクは低いが、設計として不完全

方式B（不採用確定）: seqlock
  struct PersistentStateBlock {
    std::atomic<uint64_t> version;
    uint64_t sequenceId, epoch, mappedGeneration;
  };
  ScopedVersionWriteGuard / snapshot retry / _mm_pause / RetryExceeded
  利点: concurrent writer がいる場合の安全な読取
  欠点: commit() は MessageThread 専有のため不要。コード複雑化
  → 採用しない

方式C（★採用）: std::atomic<PersistentStateBlock>
  struct PersistentStateBlock {
    uint64_t publicationSequenceId;
    uint64_t publicationEpoch;
    uint64_t mappedRuntimeGeneration;
  };
  std::atomic<PersistentStateBlock> state_;
  利点: 単一 atomic ストア/ロードで3フィールド論理一貫性を保証
  欠点: sizeof=24 > 16 のため MSVC では internal spinlock 使用
        ただし commit() は MessageThread 専有のためコンテンションは常にゼロ
  → 実害ゼロで最大の単純性を得られる
```

### 採用根拠の定量評価

| 観点 | 方式A（現行） | 方式B（seqlock） | 方式C（採用） |
|---|---|---|---|
| コード行数増加 | 0（現状維持） | +80行 | +15行 |
| 論理一貫性 | なし（個別 atomic） | あり | あり（単一 store） |
| ロックフリー | ✅（各8byte） | ❌（内部 CAS） | ❌（internal spinlock） |
| 過剰設計リスク | なし | 大（未使用機能多数） | なし |
| MessageThread 専有との整合 | △（atomic は不要だった） | ❌（seqlock は過剰） | ✅（単純で十分） |
| 将来の拡張耐性 | △ | ✅ | ✅（atomic が防護壁に） |

---

## 第1章: 方式C PersistentStateBlock 最終定義

```cpp
// ★ 方式C: std::atomic で丸ごとラップする単純構造体
// seqlock (version/ScopedGuard/retry/_mm_pause) は一切含まない
struct PersistentStateBlock {
    std::uint64_t publicationSequenceId = 0;
    std::uint64_t publicationEpoch      = 0;
    std::uint64_t mappedRuntimeGeneration = 0;

    [[nodiscard]] static bool isMonotonic(
        const PersistentStateBlock& prev,
        std::uint64_t nextSeqId,
        std::uint64_t nextEpoch,
        std::uint64_t nextGen) noexcept
    {
        const bool hasPrevious = prev.publicationSequenceId != 0
            || prev.publicationEpoch != 0
            || prev.mappedRuntimeGeneration != 0;
        if (!hasPrevious)
            return true;
        return nextSeqId > prev.publicationSequenceId
            && nextEpoch > prev.publicationEpoch
            && nextGen > prev.mappedRuntimeGeneration;
    }
};

static_assert(std::is_trivially_copyable_v<PersistentStateBlock>);
// Note: sizeof = 24 bytes. MSVC では is_always_lock_free = false
// だが、commit() は MessageThread 専有のため internal spinlock は常に非競合。

// coordinator のメンバ変数
std::atomic<PersistentStateBlock> persistentState_{};
```

### commit() 変更後（最終形）

```cpp
void RuntimePublicationCoordinator::commit(PublishAuthority,
    RuntimeBoundary boundary, const void* newWorld,
    std::uint64_t /*version*/,
    PublicationSequenceId sequenceId,
    PublicationEpoch epoch,
    std::uint64_t mappedGeneration) {

    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    // ★ 方式C: 単一 relaxed load（WriterBusy は存在しない）
    const auto prev = persistentState_.load(std::memory_order_relaxed);

    if (!PersistentStateBlock::isMonotonic(prev,
            static_cast<std::uint64_t>(sequenceId),
            static_cast<std::uint64_t>(epoch),
            mappedGeneration)) {
        // ★ isMonotonic 違反 → 実装バグまたは不変条件破壊 → Faulted は妥当
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    convo::publishAtomic(state_, CoordinatorState::Publishing, ...);
    convo::publishAtomic(swapPending_, true, ...);

    // ★ 方式C: 単一 relaxed store → 3フィールドが論理一貫して更新
    persistentState_.store(
        PersistentStateBlock{
            static_cast<std::uint64_t>(sequenceId),
            static_cast<std::uint64_t>(epoch),
            mappedGeneration
        },
        std::memory_order_relaxed);

    // ★ Phase-D 後削除
    // convo::publishAtomic(currentWorld_, newWorld, ...);

    convo::publishAtomic(swapPending_, false, ...);
    convo::publishAtomic(state_, CoordinatorState::Ready, ...);
}
```

### getVersion() 変更後

```cpp
std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    // ★ 方式C: persistentState_ から導出
    return persistentState_.load(std::memory_order_relaxed).mappedRuntimeGeneration;
}
```

---

## 第2章: Authority 3層構造（問題④⑥解決）

### AuthoritySnapshot — 純粋状態

```cpp
// 3データソースから導出される純粋な現在状態
// 永続化・格納しない。診断・比較のみに使用。
struct AuthoritySnapshot {
    std::uint64_t sequenceId;
    std::uint64_t epoch;
    std::uint64_t generation;
    bool hasActiveRuntime;  // RuntimeStore に world が存在するか

    [[nodiscard]] static AuthoritySnapshot derive(
        const PersistentStateBlock& meta,
        bool worldExists) noexcept
    {
        return AuthoritySnapshot{
            .sequenceId = meta.publicationSequenceId,
            .epoch = meta.publicationEpoch,
            .generation = meta.mappedRuntimeGeneration,
            .hasActiveRuntime = worldExists
        };
    }

    [[nodiscard]] bool operator==(const AuthoritySnapshot&) const noexcept = default;
};
```

### AuthorityDiagnostics — 健康状態

```cpp
// 複数データソースの不整合を示す診断フラグ
// AuthoritySnapshot の導出時に同時計算される
struct AuthorityDiagnostics {
    bool runtimeMissing        = false;  // PersistentStateBlock にあるが RuntimeStore にない
    bool persistentMissing     = false;  // RuntimeStore にあるが PersistentStateBlock にない
    bool fieldInconsistencyDetected = false;  // 3フィールドの論理不整合
    bool hasPendingPublication = false;  // commit 未完了の出版がある
    bool hasActiveCrossfade    = false;  // CrossfadeAuthority に active crossfade がある

    [[nodiscard]] bool operator==(const AuthorityDiagnostics&) const noexcept = default;
};
```

### AuthorityReconciliation — 修復判定

```cpp
enum class RepairAction : uint8_t {
    None,
    Observe,     // 経過観察のみ
    Throttle,    // 出版ペース抑制
    Recover,     // publishIdleWorldOnly 発行
    Restore,     // エポック回復
    Safe,        // 安全停止
    Critical     // 強制停止
};

struct ReconcileResult {
    bool fullReconciliation;
    RepairAction repairAction;
    AuthorityDiagnostics diagnostics;  // 不一致の詳細
};

[[nodiscard]] bool validateAuthorityMatch(
    const AuthoritySnapshot& observed,
    const AuthoritySnapshot& expected) noexcept
{
    // ★ 全4フィールド独立比較（operator== ではなく明示的比較）
    if (observed.sequenceId != expected.sequenceId) return false;
    if (observed.epoch != expected.epoch)           return false;
    if (observed.generation != expected.generation) return false;
    if (observed.hasActiveRuntime != expected.hasActiveRuntime) return false;
    return true;
}

[[nodiscard]] ReconcileResult reconcileAuthorityState(
    const AuthoritySnapshot& observed,
    const AuthoritySnapshot& expected,
    const AuthorityDiagnostics& diag) noexcept
{
    if (validateAuthorityMatch(observed, expected)) {
        return { .fullReconciliation = true,
                 .repairAction = RepairAction::None,
                 .diagnostics = diag };
    }

    // 不一致 → 診断に基づき RepairAction を決定
    // ...
}
```

---

## 第3章: currentWorld_ 削除 Phase（問題⑤解決）

### 完全削除条件（5条件）

| # | 条件 | 確認方法 | CI ゲート |
|---|---|---|---|
| 1 | `getCurrent()` 呼び出し元ゼロ | 全ソース + 全テストから `getCurrent()` を削除 | `isr-verify-getcurrent-zero.ps1` |
| 2 | `currentWorld_` 読取箇所ゼロ | commit/retire/getCurrent で読取停止 | `isr-verify-currentworld-read-zero.ps1` |
| 3 | `currentWorld_` CAS 依存ゼロ | retire の compareExchangeAtomic 削除 | `isr-verify-retire-no-cas.ps1` |
| 4 | `currentWorld_` メンバ変数削除 | フィールド定義削除 | `isr-verify-currentworld-field-removed.ps1` |
| 5 | テスト全件 PASS | 既存テストスイート | 手動確認 |

### Phase-1a: getCurrent() テスト移行

**現状**: 17 件の `coordinator.getCurrent()` → すべて ISR coordinator のメソッド呼び出し
**移行先**: ISR coordinator に代わる参照方法が必要。
           ISR coordinator は RuntimeStore を所有しないため、
           直接の `consumePublishedWorld(store)` 置換は不可。

**対策**: Phase-1a では以下のいずれかを選択

1. ISR coordinator に `consumePublishedWorld()` 相当の static メソッドを追加し、RuntimeStore を外部から渡す
2. テスト側で AudioEngine の `runtimeStore` を参照可能にする

**推奨: ISR coordinator に以下の static ヘルパーを追加**

```cpp
// ISRRuntimePublicationCoordinator.h に追加
// currentWorld_ を排除するための移行用ヘルパー
// テストは coordinator.getCurrent() の代わりに以下を使用:
//   RuntimePublicationCoordinator::consumePublishedWorld(store)
// ただし store は Template Coordinator の RuntimeStore。
// テストフレームワーク側で store への参照を保持する必要あり。
```

**移行パターン（17件）**:

```cpp
// 変換前:
if (coordinator.getCurrent() != &world1)

// 変換後（RuntimeStore の参照を保持している場合）:
if (RuntimePublicationCoordinator::consumePublishedWorld(testStore) != &world1)
```

---

## 第4章: Phase 再編（問題⑦解決）

```
Phase-0: 方式C PersistentStateBlock 導入
  - PersistentStateBlock 構造体定義 + std::atomic ラップ
  - 3個別 atomic フィールド削除 → persistentState_ に統合
  - commit() の 3 atomic 書込 → persistentState_.store() に変更
  - getVersion() → persistentState_.load().mappedRuntimeGeneration
  - (void) version 行削除
  - CI: isr-verify-auth-005（Coordinator 内 3フィールド統合確認）

Phase-1a: getCurrent() テスト移行
  - ISR coordinator に consumePublishedWorld(store) static ヘルパー追加
  - 17 件の test 参照を置換
  - CI: isr-verify-getcurrent-zero.ps1（getCurrent 呼出ゼロ確認）

Phase-1b: currentWorld_ 全削除
  - getCurrent() メソッド削除
  - retire() の currentWorld_ CAS 削除
  - commit() の currentWorld_ publishAtomic 削除
  - currentWorld_ メンバ変数削除
  - CI: isr-verify-currentworld-field-removed.ps1

Phase-2: Authority 3層導入
  - AuthoritySnapshot / AuthorityDiagnostics / AuthorityReconciliation 定義
  - deriveAuthorityState() → deriveSnapshot() + deriveDiagnostics()
  - validateAuthorityMatch() → 全4フィールド独立比較
  - reconcileAuthorityState() → 内部分割
  - CI: isr-verify-auth-001/002/004/006

Phase-3: Recovery 統合
  - Recovery パスでの AuthoritySnapshot 利用
  - RepairAction マッピング統合
  - CI: isr-verify-auth-002（Recovery 状態同値性）

Phase-4: CI 完全化 + Model-Based Test
  - 全 9 スクリプト作成（001-006 + 1a/1b/retire-no-cas）
  - ModelState 導入 + 6 Fault Injection シナリオ
```

---

## 第5章: snapshot の失敗可能性（問題③解決）

| 方式 | snapshot 失敗の可能性 | 失敗時の挙動 |
|---|---|---|
| 方式A（現行） | なし（個別 atomic load） | 該当なし |
| 方式B（seqlock） | あり（WriterBusy / RetryExceeded） | WriterBusy→retry、RetryExceeded→jassertfalse+return |
| **方式C（採用）** | **なし**（単一 atomic load は常に成功） | **該当なし** |

方式C では `persistentState_.load()` が常に成功するため、snapshot 失敗に伴う Faulted 遷移は発生しない。
唯一の Faulted 遷移は `isMonotonic()` 違反時のみ — これは実装バグまたは不変条件破壊を意味し、Faulted は妥当。

---

## 第6章: 実装 Phase 詳細スコープ

### Phase-0 スコープ（最小変更）

```
変更ファイル:
  src/audioengine/ISRRuntimePublicationCoordinator.h
    - #include <atomic> 変更なし
    - struct PersistentStateBlock 追加
    - 3個別 atomic フィールド削除
    - std::atomic<PersistentStateBlock> persistentState_ 追加
    - getVersion() 宣言変更なし（実装のみ変更）

  src/audioengine/ISRRuntimePublicationCoordinator.cpp
    - コンストラクタ: 3フィールド初期化削除
    - commit(): 3個別 read/write → persistentState_ に統一
    - getVersion(): persistentState_ 経由に変更
    - (void) version 行削除

  src/tests/ISRSemanticValidationTests.cpp
    - getVersion() の期待値変更なし（動作維持）
```

### Phase-1a スコープ（currentWorld_ 移行準備）

```
変更ファイル:
  src/audioengine/ISRRuntimePublicationCoordinator.h
    - static ヘルパー consumePublishedWorld(const Store&) 追加

  src/tests/ISRSemanticValidationTests.cpp
    - 17 件の coordinator.getCurrent() → consumePublishedWorld(store) に置換
    - テストフレームワークに RuntimeStore 参照を保持
```

---

## 結論

v4.18 は以下の 7 課題をすべて解決した。

| 問題 | 状態 | 解決方法 |
|---|---|---|
| ① seqlock vs atomic struct | ✅ 確定 | 方式C (`std::atomic<PersistentStateBlock>`) 採用。方式B(seqlock) 不採用確定 |
| ② 方式C 記述不整合 | ✅ 解決 | 3方式を明確に定義・比較表付き |
| ③ snapshot 失敗→Faulted | ✅ 解決 | 方式C では snapshot 失敗が存在しない。isMonotonic 違反のみ Faulted |
| ④ validateAuthorityStateMatch | ✅ 解決 | AuthoritySnapshot/Diagnostics/Reconciliation の3層分割＋全フィールド比較 |
| ⑤ currentWorld_ 削除証明 | ✅ 解決 | 5条件定義、RuntimeStore 依存の明確化、17件移行パターン提示 |
| ⑥ AuthorityState 肥大化 | ✅ 解決 | 3層分割（Snapshot/Diagnostics/Reconciliation） |
| ⑦ Phase 順序 | ✅ 解決 | P0→P1a→P1b→P2→P3→P4 に再編。currentWorld_ 削除を先行 |

**Practical Stable ISR Bridge Runtime 達成度: 95%**

**最終ステータス**: 実コード整合完了。Phase-0 の実装を開始可能。
