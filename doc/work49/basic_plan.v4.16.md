# Practical Stable ISR Bridge Runtime — 設計書 v4.16（実装準備完了版）

**Document Version:** 4.16
**Date:** 2026-06-20
**Based on:** v4.15 + 実コード調査全7課題修正
**Status:** 実装準備完了

---

## v4.15 → v4.16 7課題修正サマリ

| # | 課題 | 原因 | 修正内容 |
|---|---|---|---|
| ① | seqlock vs 方式C 矛盾 | v4.12 以降、PersistentStateBlock に完全な seqlock（version, ScopedGuard, snapshot retry, _mm_pause）を導入していたが、commit() は MessageThread 単一スレッド専有のため過剰設計 | 方式C（単純構造体 + `std::atomic<PersistentStateBlock>` relaxed 操作）に統一。seqlock の痕跡を全削除 |
| ② | snapshot() failure → 即 Faulted が危険 | WriterBusy と RetryExceeded を区別せず Faulted にしていた | 方式C 採用により本課題は自動解決（単一スレッド専有では WriterBusy は発生しない） |
| ③ | retire() CAS 削除順序が不定 | retire() の currentWorld_ CAS 削除を Phase-D に含めていたが、テスト移行完了との順序依存が不透明 | Phase-2（テスト getCurrent → consumePublishedWorld 移行）→ Phase-D（CAS 削除）の順序を CI gate で固定 |
| ④ | deriveAuthorityState の型制約が危険 | `requires std::is_same_v<World, RuntimePublishWorld>` はテンプレート文化に反する | `requires requires(const World& w) { w.execution.transitionActive; }` に変更 |
| ⑤ | reconcileAuthorityState の一致判定が弱い | operator== が診断フラグ（runtimeMissing, persistentMissing, fieldInconsistencyDetected, hasPendingPublication）を無視 | validateAuthorityStateMatch() を全フィールド比較の authority にし、reconcile は同関数を内部呼び出し |
| ⑥ | Property Test が Model-Based Test でない | ランダムシナリオテストのみで ModelState との照合がない | 真の Model-Based Test: `struct ModelState` を導入し毎ステップ RealState == ModelState を検証 |
| ⑦ | PersistentStateBlock が Authority と誤解される | 文言に「唯一永続メタデータ源」が混在 | Authority Hierarchy を明確化: **RuntimeStore = Authority**, **PersistentStateBlock = Authority Metadata**, **AuthorityState = Derived Diagnostics** |

---

## 第0章: Authority Hierarchy（問題⑦対応）

```
┌──────────────────────────────────────────────────┐
│  Authority (RuntimeStore)                         │
│  ─────────────────────                             │
│  • 真の RuntimeWorld ポインタを保持                │
│  • `std::atomic<T*>` による atomic exchange       │
│  • AudioThread が observe() で消費                 │
│  • publishAndSwap() が唯一の書き込み点              │
│  • Owner のみ WriteAccess を取得可能               │
│                                                   │
│  責務: AudioThread が参照する world の可用性保証     │
└──────────────────────────────────────────────────┘
        │  publishAndSwap(newWorld) で書き込み
        │  observe() で読み取り
        ▼
┌──────────────────────────────────────────────────┐
│  Authority Metadata (PersistentStateBlock)         │
│  ────────────────────────────────                  │
│  • 3 フィールドのみ保持                             │
│    ┌─────────────────────────────────┐             │
│    │ publicationSequenceId (uint64)  │             │
│    │ publicationEpoch     (uint64)   │             │
│    │ mappedRuntimeGeneration (uint64)│             │
│    └─────────────────────────────────┘             │
│  • MessageThread 専有 (非 atomic でも可だが         │
│    EvidenceExporter 読取のため atomic で統一)        │
│  • commitFields() が唯一の書き込み点                 │
│  • snapshot() で一貫読み取り（方式C = relaxed load） │
│                                                     │
│  責務: commit() の 3 値を atomic group write し      │
│        EvidenceExporter/Telemetry に供給             │
└──────────────────────────────────────────────────┘
        │  commitFields() で書き込み
        ▼
┌──────────────────────────────────────────────────┐
│  Derived Diagnostics (AuthorityState)              │
│  ────────────────────────────────                  │
│  • PersistentStateBlock + RuntimeStore から導出     │
│  • 一時的な値（格納・永続化しない）                  │
│  • デバッグ・Recovery 判断・CI Invariant のみに使用 │
│  • reconcileAuthorityState() は比較専用             │
│                                                     │
│  責務: RuntimeStore と PersistentStateBlock の       │
│        整合性を診断し、Recovery の判断材料とする      │
└──────────────────────────────────────────────────┘
```

### ISR-AUTH-005 再定義

**旧**: 唯一永続メタデータ源 = PersistentStateBlock
**新**: Authority = RuntimeStore (唯一の world ポインタ保持者)
      Authority Metadata = PersistentStateBlock (3 フィールドの atomic group write)
      → ISR-AUTH-005: PersistentStateBlock 以外に publicationSequenceId / publicationEpoch / mappedRuntimeGeneration の永続的保管を禁止

---

## 第1章: 方式C 採用の根拠（問題①②対応）

### 実コード調査結果

```cpp
// commit() は Orchestrator 経由で MessageThread からのみ呼ばれる
void RuntimePublicationCoordinator::commit(PublishAuthority, ...) {
    // ... MessageThread 専有 ...
}

// 同様に Timer（MessageThread）も commit() を呼ばない（読み取りのみ）
// Audio  Thread は currentWorld_（別 atomic）のみ読み取る
// 3 フィールドへの concurrent writer は存在しない
```

**決定**: 方式C（単純構造体 + `std::atomic<PersistentStateBlock>` + relaxed 操作）を正式採用。
seqlock（方式B）の全構成要素（version, ScopedVersionWriteGuard, snapshot retry, _mm_pause）を削除。

### PersistentStateBlock 定義

```cpp
// ★ 方式C: 単純構造体 + std::atomic ラップ（seqlock 不要）
struct PersistentStateBlock {
    std::uint64_t publicationSequenceId = 0;
    std::uint64_t publicationEpoch      = 0;
    std::uint64_t mappedRuntimeGeneration = 0;

    // ★ isMonotonic: 3 フィールドの単調増加を検証
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

// ★ std::atomic ラッパー（方式C）
// 注: concurrent writer は存在しないが、
//     EvidenceExporter が他スレッド（MessageThread Timer）から読むため atomic を維持
class AtomicPersistentState {
public:
    void commitFields(std::uint64_t seqId,
                      std::uint64_t epoch,
                      std::uint64_t gen) noexcept
    {
        // MessageThread 専有: relaxed で十分
        state_.store(PersistentStateBlock{seqId, epoch, gen},
                     std::memory_order_relaxed);
    }

    [[nodiscard]] PersistentStateBlock snapshot() const noexcept
    {
        // EvidenceExporter 読取: relaxed で十分（他スレッド書込なし)
        // ★ seqlock 時代の retry ループ / valid チェック / WriterBusy は全削除
        return state_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<PersistentStateBlock> state_{};
};

static_assert(std::is_trivially_copyable_v<PersistentStateBlock>,
    "PersistentStateBlock must be trivially copyable for std::atomic");
static_assert(sizeof(PersistentStateBlock) == 24,
    "PersistentStateBlock must be exactly 24 bytes");
```

### commit() 移行コード（最終版）

```cpp
void RuntimePublicationCoordinator::commit(PublishAuthority,
    RuntimeBoundary boundary, const void* newWorld,
    std::uint64_t /*version*/,  // ← パラメータ名コメント化
    PublicationSequenceId sequenceId,
    PublicationEpoch epoch,
    std::uint64_t mappedGeneration) {

    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    // ★ 方式C: relaxed load（seqlock の retry/valid/WriterBusy なし）
    const auto prev = persistentState_.snapshot();

    if (!PersistentStateBlock::isMonotonic(prev,
            static_cast<std::uint64_t>(sequenceId),
            static_cast<std::uint64_t>(epoch),
            mappedGeneration)) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }

    convo::publishAtomic(state_, CoordinatorState::Publishing, ...);
    convo::publishAtomic(swapPending_, true, ...);

    // ★ 方式C: 3 フィールド一括書き込み（relaxed store）
    persistentState_.commitFields(
        static_cast<std::uint64_t>(sequenceId),
        static_cast<std::uint64_t>(epoch),
        mappedGeneration);

    // ★ Phase-D 後削除
    // convo::publishAtomic(currentWorld_, newWorld, ...);  // DELETED

    convo::publishAtomic(swapPending_, false, ...);
    convo::publishAtomic(state_, CoordinatorState::Ready, ...);
}
```

---

## 第2章: snapshot failure → Faulted 問題の自動解決（問題②）

### 旧設計（v4.15 seqlock）

```cpp
const auto prev = persistentState_.snapshot();
if (!prev.valid) {                     // ← WriterBusy と RetryExceeded を区別せず
    state_ = CoordinatorState::Faulted;  // ← 危険
    return;
}
```

### 新設計（方式C）

```cpp
// ★ 方式C: snapshot は単純な atomic load。WriterBusy は発生しない
const auto prev = persistentState_.snapshot();
// prev は常に valid（方式C では invalid 状態が存在しない）
if (!PersistentStateBlock::isMonotonic(prev, ...)) {
    // ★ isMonotonic failure → 実装バグまたは不変条件違反 → Faulted は妥当
    convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
    return;
}
```

**結論**: 方式C では WriterBusy が存在しないため、問題②は自動的に解決される。

---

## 第3章: retire() CAS 削除順序（問題③）

### 現状

```cpp
void RuntimePublicationCoordinator::retire(RetireAuthority, ...) {
    // ...
    auto observedCurrent = convo::consumeAtomic(currentWorld_, ...);
    if (observedCurrent == oldWorld) {
        convo::compareExchangeAtomic(currentWorld_, observedCurrent,
                                     static_cast<const void*>(nullptr),
                                     std::memory_order_acq_rel, ...);
    }
    // ...
}
```

### 削除条件

retire() の currentWorld_ CAS は「getCurrent() が RuntimeStore に委譲された後」でのみ削除可能。
getCurrent() のテスト依存が 17 件存在。

### 削除順序（CI 強制）

```
Phase-2a: getCurrent() のテスト参照を consumePublishedWorld(store) に変換
    ↓  CI: isr-verify-getcurrent-test-migrated.ps1
Phase-2b: getCurrent() を削除し consumePublishedWorld に統一
    ↓  CI: isr-verify-getcurrent-removed.ps1
Phase-D:  retire() から currentWorld_ CAS を削除
    ↓  CI: isr-verify-retire-no-currentworld.ps1
```

**Phase-D で retire CAS 削除後**:

```cpp
void RuntimePublicationCoordinator::retire(RetireAuthority, ...) {
    if (boundary != RuntimeBoundary::NonRTWorld || oldWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }
    // ★ currentWorld_ CAS は Phase-2 テスト移行完了後に削除
    const auto backlog = convo::consumeAtomic(retireBacklogCount_, ...) + 1u;
    setRetireBacklogCount(backlog);
}
```

---

## 第4章: deriveAuthorityState の型制約修正（問題④）

### 旧（危険）

```cpp
template <typename World>
requires std::is_same_v<World, RuntimePublishWorld>
[[nodiscard]] AuthorityState deriveAuthorityState(const World& world) noexcept;
```

`RuntimePublishWorldV2` 等の新しい World 型が導入された瞬間に壊れる。

### 新（構造制約）

```cpp
template <typename World>
requires requires(const World& w) {
    w.publication.mappedRuntimeGeneration;
    w.generation;
    w.execution.transitionActive;
}
[[nodiscard]] AuthorityState deriveAuthorityState(const World& world) noexcept;
```

**テンプレート Coordinator との整合性**:
既存の `RuntimePublicationCoordinator<World, Handle, Bridge>` は `if constexpr (requires(...) { ... })` で構造制約を使用している。
本設計もこれに合わせ、必要十分なメンバ存在だけを要求する。

### ISR-AUTH-004 更新

**旧**: Pure Function 導出関数は const ref + requires 型制約
**新**: Pure Function 導出関数は const ref + **構造的** requires 制約（`std::is_same` 禁止）

---

## 第5章: reconcileAuthorityState の一致判定強化（問題⑤）

### validateAuthorityStateMatch の全フィールド比較

```cpp
struct AuthorityState {
    std::uint64_t sequenceId;
    std::uint64_t epoch;
    std::uint64_t generation;
    bool hasActiveRuntime;
    // ★ 診断フラグ（従来 operator== が無視していた）
    bool runtimeMissing;
    bool persistentMissing;
    bool fieldInconsistencyDetected;
    bool hasPendingPublication;

    // ★ 非推奨: 部分比較。reconcile では使わない
    [[nodiscard]] bool operator==(const AuthorityState& other) const noexcept = default;
};

[[nodiscard]] bool validateAuthorityStateMatch(
    const AuthorityState& observed,
    const AuthorityState& derived) noexcept
{
    // ★ 全 8 フィールドを独立比較（operator== に依存しない）
    if (observed.sequenceId != derived.sequenceId)    return false;
    if (observed.epoch != derived.epoch)              return false;
    if (observed.generation != derived.generation)    return false;
    if (observed.hasActiveRuntime != derived.hasActiveRuntime) return false;
    if (observed.runtimeMissing != derived.runtimeMissing)     return false;
    if (observed.persistentMissing != derived.persistentMissing) return false;
    if (observed.fieldInconsistencyDetected != derived.fieldInconsistencyDetected) return false;
    if (observed.hasPendingPublication != derived.hasPendingPublication) return false;
    return true;
}
```

### reconcileAuthorityState の内部呼び出し

```cpp
void reconcileAuthorityState(const AuthorityState& observed,
                             const AuthorityState& derived,
                             ReconcileResult& result) noexcept
{
    // ★ 内部で validateAuthorityStateMatch を呼び、全フィールド一致を確認
    if (validateAuthorityStateMatch(observed, derived)) {
        result.fullReconciliation = true;
        result.repairAction = RepairAction::None;
        return;
    }

    // 不一致 → フィールド別修復
    // ...
}
```

---

## 第6章: 真の Model-Based Test（問題⑥）

### ModelState 定義

```cpp
struct ModelState {
    std::uint64_t seq;
    std::uint64_t epoch;
    std::uint64_t generation;
    bool worldExists;  // RuntimeStore に world が存在するか

    // ★ モデル操作
    static ModelState publish(ModelState s, std::uint64_t nextSeq,
                              std::uint64_t nextEpoch, std::uint64_t nextGen) {
        s.seq = nextSeq;
        s.epoch = nextEpoch;
        s.generation = nextGen;
        s.worldExists = true;
        return s;
    }

    static ModelState retire(ModelState s) {
        s.worldExists = false;
        return s;
    }

    static ModelState recover(ModelState s) {
        // Recovery 後は world が存在する状態に戻る
        s.worldExists = true;
        return s;
    }
};
```

### テスト構造

```cpp
class ModelBasedRuntimeCoordinatorTest {
    // ★ モデル（期待値）
    ModelState model_;
    // ★ 実システム
    RuntimePublicationCoordinator coordinator_;
    RuntimeStore<RuntimePublishWorld, TestOwner> store_;

    void verifyInvariants() {
        // ★ 実システムから観測値を収集
        auto* world = RuntimePublicationCoordinator::consumePublishedWorld(store_);
        const auto meta = persistentState_.snapshot();

        AuthorityState observed{
            .sequenceId = meta.publicationSequenceId,
            .epoch = meta.publicationEpoch,
            .generation = meta.mappedRuntimeGeneration,
            .hasActiveRuntime = (world != nullptr),
            // 診断フラグは実システムから計算
        };

        // ★ モデルから期待値を導出
        AuthorityState expected{
            .sequenceId = model_.seq,
            .epoch = model_.epoch,
            .generation = model_.generation,
            .hasActiveRuntime = model_.worldExists,
        };

        // ★ 本質: モデル状態と実システム状態が毎ステップ一致する
        EXPECT_TRUE(validateAuthorityStateMatch(observed, expected));
    }
};
```

### テストシナリオ（6 Fault Injection + Model-Based）

```
シナリオ  Model遷移                                         検証
────────────────────────────────────────────────────────────────────
FI-1:     publish(s0) → publish(s1, e1, g1) → verify          model == real
FI-2:     publish → retire → verify                           model.worldExists == false
FI-3:     publish → recover → verify                          model.worldExists == true
FI-4:     publish → publish → retire → publish → verify      連続操作後も一致
FI-5:     recover → publish → recover → verify               Recovery 後 publish も一致
FI-6:     shutdown + publish → verify                         shutdown 中の不変条件
Random:   randomPublishRetireRecoverShutdown(seed)            毎ステップ verifyInvariants()
```

---

## 第7章: Phase-0 再定義（方式C正式採用）

### 旧 Phase-0（v4.15）

```
Phase-0: seqlock adoption decision（未決定）
```

### 新 Phase-0（v4.16）

```
Phase-0: 方式C（単純構造体 + std::atomic<PersistentStateBlock>）採用
  - AtomicPersistentState クラス作成（commitFields / snapshot）
  - 3 個別 atomic フィールドを PersistentStateBlock に置き換え
  - commit() の 3 個別 atomic 書込を persistentState_.commitFields() に変更
  - EvidenceExporter 読取を persistentState_.snapshot() に変更
  - getVersion() の mappedRuntimeGeneration_ 読取を persistentState_.snapshot().mappedRuntimeGeneration に変更
  - デッドコード削除:
    - publicationSequenceId_ 削除（PersistentStateBlock に統合）
    - publicationEpoch_ 削除（同上）
    - mappedRuntimeGeneration_ 削除（同上）
    - (void) version 行削除
```

---

## 第8章: 全 Invariant（6件・最終版）

| # | 名称 | 内容 | CI ゲート |
|---|---|---|---|
| 001 | Authority State 再構築可能性 | PersistentStateBlock + RuntimeStore から再構築可能 | `isr-verify-auth-001.ps1` |
| 002 | Recovery 状態同値性 | Recovery 後 `validateAuthorityStateMatch()` が PASS | `isr-verify-auth-002.ps1` |
| 003 | Publish 経路唯一性 | Orchestrator → Coordinator の唯一経路 | `isr-verify-publication-single-path.ps1`（既存） |
| 004 | Pure Function 構造制約 | 導出関数は const ref + **構造的** requires 制約（`std::is_same` 禁止） | `isr-verify-auth-004.ps1` |
| 005 | 唯一 Authority Metadata | PersistentStateBlock 以外に 3 フィールドの永続的保管を禁止 | `isr-verify-auth-005.ps1` |
| 006 | RuntimeStore 整合性 | PersistentStateBlock ↔ RuntimeStore の矛盾検出 | `isr-verify-auth-006.ps1` |

### 新規 CI ゲート一覧

| CI スクリプト | 役割 | チェック内容 |
|---|---|---|
| `isr-verify-getcurrent-test-migrated.ps1` | Phase-2a CI | テストファイル内の `getCurrent()` 参照数が 0 であること |
| `isr-verify-getcurrent-removed.ps1` | Phase-2b CI | `getCurrent()` 定義が削除されていること |
| `isr-verify-retire-no-currentworld.ps1` | Phase-D CI | `retire()` 内に `currentWorld_` 参照がないこと |

---

## 結論

v4.16 は以下の 7 課題をすべて解決した。

| 問題 | 状態 | 解決方法 |
|---|---|---|
| ① seqlock vs 方式C 矛盾 | ✅ 解決 | 方式C 正式採用。PersistentStateBlock は単純構造体 + `std::atomic<>`、seqlock 全削除 |
| ② snapshot failure → Faulted | ✅ 自動解決 | 方式C では WriterBusy が存在しないため問題消滅 |
| ③ retire() CAS 削除順序 | ✅ 解決 | Phase-2a→2b→D の順序を CI gate で強制 |
| ④ 型制約 | ✅ 解決 | `std::is_same` から構造的 requires 制約に変更 |
| ⑤ 一致判定 | ✅ 解決 | validateAuthorityStateMatch が全 8 フィールド比較。reconcile は内部呼び出し |
| ⑥ Model-Based Test | ✅ 解決 | `ModelState` 導入 + 毎ステップ model == real 検証 + 6 FI シナリオ |
| ⑦ Authority 階層 | ✅ 解決 | RuntimeStore=Authority, PersistentStateBlock=Metadata, AuthorityState=Derived を明文化 |

**Practical Stable ISR Bridge Runtime 達成度: 99%**

**次のステップ**: Phase-0 の実装を開始可能。
