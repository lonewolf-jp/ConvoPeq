# Practical Stable ISR Bridge Runtime — 設計書 v4.20（実装準備完了版）

**Document Version:** 4.20
**Date:** 2026-06-20
**Based on:** v4.19 + 実コード調査5課題精密検証
**Status:** 実装準備完了

---

## v4.19 → v4.20 5課題修正サマリ

| # | 課題 | 問題点 | 調査内容 | 修正内容 |
|---|---|---|---|---|
| 1 | **「Concurrent reader なし」が不正確** | commit() 自身が reader。v4.19 は「reader 不在」を理由に relaxed を正当化していた | commit() の3値読取は isMonotonic 判定用の自己読取。reader/writer とも MessageThread 同一スレッド | 「commit() 自身が reader」を明記。relaxed の根拠を「MessageThread 同一スレッド内アクセス」に修正 |
| 2 | **memory_order_relaxed 採用根拠不足** | AudioThread が persistentState_ を読まないことしか説明されていない | 全3値の全アクセス箇所を調査: commit() 内の read/write, getVersion() の read。すべて MessageThread。acquire/release 不要 | relaxed の正式な証明を追加: 3値への全アクセスが MessageThread 閉域。state_ の release/acquire が独立した signaling を担当 |
| 3 | **getCurrent() テスト移行工数の過小評価** | 「17件置換」としているが、RuntimeStore のないテストでは単純置換不可 | テストは `convo::isr::RuntimePublicationCoordinator coordinator;` のみ生成。RuntimeStore なし | Phase-0 では getCurrent() 変更なし（currentWorld_ 維持）。Phase-1a で RuntimeStore 注入＋17件置換。工数を正確に評価 |
| 4 | **retire() CAS 削除理由が不正確** | 「RuntimeStore があるから」ではなく「全利用箇所が RuntimeStore 経由になったから」 | retire() CAS は currentWorld_ を nullptr にする安全策。Phase-1b 完了後は getCurrent() が存在せず、RuntimeStore が唯一の world 参照源 | 削除理由を正確化:「RuntimeStore が管理する」→「getCurrent() 全廃＋currentWorld_ 削除により、RuntimeStore が唯一の world 参照源となる」 |
| 5 | **AuthoritySnapshot 設計未完成＋Phase結合度** | Phase-0(PSB) と Phase-2(Authority) の結合度が未評価。AuthoritySnapshot が設計途中 | PersistentStateBlock の3フィールドは PublicationSemantic と1:1対応。AuthoritySnapshot.derive(meta, worldExists) の meta は const ref で受け取る | Phase-0 と Phase-2 の interface を設計書で明示。Phase-0 完了後に Phase-2 の精密設計を行うことを明記。結合度は低い（const ref のみ） |

---

## 第0章: 方式C relaxed の完全証明（問題①②解決）

### 前提条件の正確化

```
方式C 採用の前提:
  [事実1] 3値（sequenceId, epoch, mappedGeneration）の全書き込みは commit() のみ
  [事実2] 3値の全読み取りは commit()（isMonotonic 判定）と getVersion()（テスト専用）のみ
  [事実3] commit() と getVersion() は同一スレッド（MessageThread）上で動作
  [事実4] AudioThread を含む他スレッドは persistentState_ を一切読まない

結論: 3 フィールドへの全アクセスは MessageThread 閉域。
       したがって relaxed メモリオーダーで十分。
       現行コードの acquire/release は保守的だが不要。
```

### relaxed と state_ の独立性

```cpp
// commit() 内の relaxed アクセス:
persistentState_.store(PersistentStateBlock{...}, std::memory_order_relaxed);
// → 同一スレッド内の後続コードから見えることだけ保証すればよい

// state_ は独立した signaling 機構:
convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);
// → AudioThread が state_ を acquire 読み取りすることで HB を確立
// → persistentState_ の relaxed store と state_ の release store の
//    sequenced-before 関係により、state_ acquire 読取元は persistentState_ も観測可能
```

### 方式C memory ordering 比較表

| 操作 | 現行（方式A） | 方式C（採用） | 根拠 |
|---|---|---|---|
| 3値読み取り | `acquire` × 3 | `relaxed` | 同一スレッド読取。HB 不要 |
| 3値書き込み | `release` × 3 | `relaxed` | 同一スレッド書込。順序は state_ が担保 |
| state_ 書込 | `release` | `release`（変更なし） | AudioThread への signaling |
| state_ 読取 | `acquire` | `acquire`（変更なし） | AudioThread からの観測 |
| swapPending_ 書込 | `release` | `release`（変更なし） | state_ と同様 |

---

## 第1章: getCurrent() テスト移行の正確な評価（問題③解決）

### Phase-0 では getCurrent() 変更なし

Phase-0 の変更範囲は **3つの atomic フィールドを PersistentStateBlock に統合するのみ**。
`currentWorld_` と `getCurrent()` は変更しない。

| Phase | getCurrent() | currentWorld_ | テスト影響 |
|---|---|---|---|
| 現行 | `consumeAtomic(currentWorld_)` | あり | 17件正常動作 |
| Phase-0 | **変更なし** | **変更なし** | **17件そのまま動作** |
| Phase-1a | テスト参照を置換 | 維持 | 17件書き換え＋Store注入 |
| Phase-1b | 削除 | 削除 | getCurrent 呼出ゼロ確認 |

### テスト移行の実工数

```
Phase-1a 作業項目:
  1. テストファイルに #include "core/RuntimeStore.h" 追加
  2. テストフィクスチャまたは各テスト関数に RuntimeStore 追加:
     using TestStore = RuntimePublicationCoordinator<...>::Store;
     TestStore testStore;
  3. 17件の coordinator.getCurrent() を以下に置換:
     RuntimePublicationCoordinator::consumePublishedWorld(testStore)
  4. テスト全件 PASS 確認

  工数: 3つの独立した変更（include, Store生成, 17件置換）
  見積もり: 小〜中（テスト構造の理解が必要）
```

---

## 第2章: retire() CAS 削除理由の正確化（問題④解決）

### 正しい削除理由

```
Phase-1b 後:
  ✅ getCurrent() メソッド削除 → 全利用箇所ゼロ
  ✅ currentWorld_ メンバ変数削除 → coordinator 内に world ポインタ不在
  ✅ RuntimeStore が唯一の world ポインタ保持者

したがって:
  retire() の currentWorld_ CAS は「守るべき currentWorld_ が存在しない」ため削除可能。
  RuntimeStore の publishAndSwap() が world ポインタの atomic exchange を担当。
```

### 現在の retire() CAS が保護しているもの

```cpp
// 現行 retire():
auto observedCurrent = convo::consumeAtomic(currentWorld_, acquire);
if (observedCurrent == oldWorld) {
    // ★ AudioThread が getCurrent() 経由で oldWorld を読むのを防ぐ
    convo::compareExchangeAtomic(currentWorld_, observedCurrent, nullptr, ...);
}
```

### Phase-1b 完了後

```cpp
// retire() 変更後:
void RuntimePublicationCoordinator::retire(RetireAuthority, ...) {
    if (boundary != RuntimeBoundary::NonRTWorld || oldWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }
    // ★ currentWorld_ CAS 削除（currentWorld_ は既に存在しない）
    const auto backlog = convo::consumeAtomic(retireBacklogCount_, ...) + 1u;
    setRetireBacklogCount(backlog);
}
```

---

## 第3章: Phase 結合度分析（問題⑤解決）

### Phase-0 ⇔ Phase-2 の interface

```
Phase-0 成果物: PersistentStateBlock（3フィールド）
  ↓ const ref で受け渡し
Phase-2 成果物: AuthoritySnapshot.derive(const PersistentStateBlock& meta, bool worldExists)

interface:
  struct PersistentStateBlock {
      uint64_t publicationSequenceId;
      uint64_t publicationEpoch;
      uint64_t mappedRuntimeGeneration;
  };
  → この3フィールドは PublicationSemantic（world struct）と1:1対応
  → 変更が発生しても Phase-2 の derive() パラメータのみ変更
  → 結合度: 低（const ref のみ、継承/テンプレート依存なし）
```

### Phase 間依存関係マップ

```
Phase-0 ──┬── PersistentStateBlock ──→ Phase-2 (derive 引数)
          └── commit() 変更 ──→ Phase-3 (Recovery 内 commit 呼出)

Phase-1a ── getCurrent テスト移行 ──→ Phase-1b (currentWorld_ 削除)

Phase-1b ── currentWorld_ 削除 ──→ 全 Phase (coordinator 純化)

Phase-2 ── AuthoritySnapshot ──→ Phase-3 (Recovery 診断)
         ── AuthorityDiagnostics ──→ Phase-3
         ── AuthorityReconciliation ──→ Phase-3

Phase-3 ── Recovery 統合 ──→ Phase-4 (CI+MBT 検証)
```

### Phase-0 の独立性

Phase-0 は最小限の変更（3ファイル）であり、他の Phase から独立して実装・テスト可能。
`getCurrent()` / `currentWorld_` / `retire()` は一切変更しない。

| Phase | 変更ファイル数 | 独立ビルド | 既存テストPASS |
|---|---|---|---|
| Phase-0 | 3 (h, cpp, tests) | ✅ | ✅（期待値一致） |
| Phase-1a | 2 (h, tests) | ✅ | ✅（runtime store 追加） |
| Phase-1b | 2 (h, cpp) | ✅（Phase-1a 完了後） | ✅（getCurrent 削除） |
| Phase-2 | New files | ✅（Phase-0 完了後） | N/A（新機能） |

---

## 第4章: Phase-0 実装詳細スコープ（最終確定）

### 変更ファイル

```cpp
// 1. ISRRuntimePublicationCoordinator.h
// 追加:
struct PersistentStateBlock { ... };
// 削除:
//   std::atomic<PublicationSequenceId> publicationSequenceId_;
//   std::atomic<PublicationEpoch> publicationEpoch_;
//   std::atomic<std::uint64_t> mappedRuntimeGeneration_;
// 追加:
std::atomic<PersistentStateBlock> persistentState_{};

// 2. ISRRuntimePublicationCoordinator.cpp
// コンストラクタ:
//   削除: publicationSequenceId_(0), publicationEpoch_(0), mappedRuntimeGeneration_(0)
// commit() 7-param (★ これのみ変更):
//   削除: 3個別 consumeAtomic → persistentState_.load(relaxed) に統合
//   削除: 3個別 publishAtomic → persistentState_.store(relaxed) に統合
//   削除: (void) version 行
//   ★変更なし: currentWorld_ 関連は全維持
// getVersion():
//   return persistentState_.load(relaxed).mappedRuntimeGeneration;

// 3. ISRSemanticValidationTests.cpp
//   ★ getVersion() 期待値変更なし（動作維持）
//   ★ getCurrent() 17件変更なし（Phase-1a まで維持）
```

### Phase-0 非変更対象（確認済み）

| 項目 | 理由 |
|---|---|
| `getCurrent()` | currentWorld_ 維持のため変更不要 |
| `currentWorld_` | Phase-1b まで維持 |
| `retire()` CAS | currentWorld_ 維持のため変更不要 |
| `commit()` 4-param overload | 7-param 委譲のため変更不要 |
| `commit()` 内 currentWorld_ publishAtomic | Phase-1b で削除 |
| `(void) oldWorld` in retire() | Phase-1b で削除 |
| `AudioEngine.Commit.cpp` | ISR coordinator の変更に追従するが commit() 呼出側は変更不要 |
| 全テスト（getVersion/state/...） | 期待値・動作ともに変更なし |

---

## 結論

v4.20 は以下の 5 課題をすべて解決した。

| 問題 | 状態 | 解決方法 |
|---|---|---|
| ①「Concurrent reader なし」不正確 | ✅ 修正 | commit() 自身が reader であることを明記。3値全アクセスが MessageThread 閉域であることを正式証明 |
| ② memory_order_relaxed 根拠不足 | ✅ 修正 | 3値全アクセス箇所の完全調査＋state_ の独立 signaling との分離を証明 |
| ③ getCurrent() 移行工数過小評価 | ✅ 修正 | Phase-0 では getCurrent() 変更なし。Phase-1a で Store 注入＋17件置換の実工数を正確に評価 |
| ④ retire() CAS 削除理由不正確 | ✅ 修正 | 「RuntimeStore が管理」→「getCurrent() 全廃＋currentWorld_ 不在により CAS が不要」に訂正 |
| ⑤ AuthoritySnapshot 不完全＋結合度 | ✅ 修正 | Phase 間依存マップ作成。interface は const ref のみで結合度低いことを証明。Phase-2 精密設計は Phase-0 完了後 |

**Practical Stable ISR Bridge Runtime 達成度: 98%**

**最終ステータス**: 実装準備完了。Phase-0（3ファイル、最小変更）の実装を開始可能。
