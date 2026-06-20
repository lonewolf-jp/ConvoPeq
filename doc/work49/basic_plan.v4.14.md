# Practical Stable ISR Bridge Runtime — 設計書 v4.14（最終確定版・全不備修正済）

**Document Version:** 4.14
**Date:** 2026-06-20
**Based on:** v4.13 + レビュー指摘9点の反映
**Status:** 最終確定（全設計判断・全不備修正完了）

---

## v4.13 → v4.14 修正一覧

| # | 指摘 | v4.13 | v4.14 |
|---|---|---|---|
| 1 | **ISR-AUTH-001 未成立** | PersistentStateBlock のみで再構築可能と規定 | **「PersistentStateBlock + RuntimeStore から再構築可能」に修正** |
| 2 | **deriveExpectedState 単純化** | `sequenceId>0 ⇒ hasActiveRuntime` | **削除。RuntimeStore の現実（shutdown/restore 途中）を直視し、期待状態導出は reconcileAuthorityState に統合** |
| 3 | **snapshot 失敗時** | Recovery 経由 | **Faulted 状態遷移。Recovery は使わない** |
| 4 | **\_mm\_pause()** | 3回のループに導入 | **削除。3回では効果なし** |
| 5 | **ScopedGuard 説明** | 「assert/jassert でも安全」 | **「通常スコープ離脱時に偶数へ戻す」に修正。terminate/AV は対象外** |
| 6 | **operator==** | 4フィールドのみ | **全フィールドを比較** |
| 7 | **validateAuthorityStateMatch** | operator== 再利用 | **独立したフィールド単位の比較** |
| 8 | **getVersion()** | snapshot 失敗時 0 返却 | **変更なし。直接 mappedRuntimeGeneration を返す** |
| 9 | **3系統の状態管理** | RuntimeStore + Persistent + Authority | **AuthorityState は導出値（キャッシュではない）。真の状態は RuntimeStore のみ** |

---

## 第0章: アーキテクチャ（最重要の整理）

### 状態の種類

```
真の状態（唯一）:
  RuntimeStore （world ポインタ）
  PersistentStateBlock （永続メタデータ: seq/epoch/gen）

導出値（キャッシュではない）:
  AuthorityState = deriveAuthorityState(PersistentState, RuntimeStore)
  → この値は保存されず、必要な都度導出される

Recovery アクション:
  reconcileAuthorityState(observed, expected)
  → 導出値の比較から修復アクションを決定
```

**最重要**: `AuthorityState` は保存されない。状態は `RuntimeStore` と `PersistentStateBlock` のみ。

---

## 第1章: ISR-AUTH-001 再定義

```
ISR-AUTH-001（修正版）

Authority State は PersistentStateBlock の永続メタデータと
RuntimeStore の現在状態から再構築可能でなければならない。

成立条件:
  - deriveAuthorityState() の入力は PersistentStateBlock::SnapshotResult + RuntimeStore::observe()
  - deriveAuthorityState() は Pure Function（副作用なし）
  - Recovery は deriveAuthorityState() + reconcileAuthorityState() で修復アクションを決定

補足:
  - currentWorld_ からの再構築は禁止（削除予定）
  - AuthorityState は保存されない（毎回導出）
```

---

## 第2章: deriveExpectedState 削除

v4.13 の `deriveExpectedState` は `sequenceId>0 ⇒ hasActiveRuntime` と単純化し過ぎていた。
shutdown/restore/idle publish 前など `sequenceId>0 && world==nullptr` は正常状態になり得る。

**判定**: `deriveExpectedState` を削除し、`reconcileAuthorityState` が直接 observed から
修復要否を判断する方式に変更する。

```cpp
struct AuthorityReconciliation {
    bool needsIdlePublish{false};
    bool needsRetireDrain{false};
    bool needsCrossfadeComplete{false};
    bool fullReconciliation{false};
};

// ★ deriveExpectedState は削除。
//   reconcileAuthorityState が observed のみから修復要否を判断する。
[[nodiscard]] AuthorityReconciliation reconcileAuthorityState(
    const AuthorityState& observed) noexcept
{
    AuthorityReconciliation rec;

    // runtimeMissing: Persistent は世界を示す (sequenceId>0) が world がない
    //   ただし shutdown/restore 途中もあり得るため epoch も確認
    if (observed.runtimeMissing && observed.publicationEpoch > 0) {
        // epoch>0 かつ world なし → 確実に world 消失（bootstrap ではない）
        rec.needsIdlePublish = true;
    }

    if (observed.hasActiveCrossfade) {
        // crossfade が滞留 → epoch/generation の整合性を確認
        if (observed.fieldInconsistencyDetected)
            rec.needsCrossfadeComplete = true;
    }

    if (!rec.needsIdlePublish && !rec.needsCrossfadeComplete)
        rec.fullReconciliation = true;

    return rec;
}
```

---

## 第3章: snapshot() 失敗時の扱い

```
v4.13: RetryExceeded → Recovery → IdleWorld Publish
v4.14: RetryExceeded → Faulted（プログラムバグとして扱う）

理由:
  - commit と snapshot は同一 MessageThread で動作
  - 3回のリトライで成功しないのは実装バグ
  - Recovery で修復する状態ではない
```

```cpp
// commit() 内
const auto prev = persistentState_.snapshot();
if (!prev.valid) {
    // ★ seqlock が 3回リトライで成功しない = 実装バグ
    //   Faulted に遷移し、回復不能状態として扱う
    convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
    return;
}
```

---

## 第4章: _mm_pause() 削除

```
kMaxSnapshotRetries = 3 では _mm_pause() の効果は無視できる。
複雑性だけが増加するため導入しない。

v4.13 の _mm_pause() 行を削除。
```

---

## 第5章: ScopedVersionWriteGuard 説明修正

```
v4.13:
  「デストラクタで必ず偶数に戻すため、assert/jassert による途中終了でも安全」

v4.14:
  「通常のスコープ離脱時に version を偶数に戻す。
   jassert は Release ビルドで消滅するため Release では動作しない。
   std::terminate / access violation ではデストラクタは呼ばれないため、
   これらのケースでは version が奇数のまま残る可能性がある。
   これは seqlock の既知の制限であり、システム再起動まで snapshot() が
   WriterBusy を返し続けることを意味する」

  // 補足: WriterBusy が永続化した場合の回復手段
  //   commitFields() は MessageThread 専有であるため、
  //   terminate/AV 後の回復は Process 再起動に依存する。
  //   これは許容範囲内のリスクである。
```

---

## 第6章: operator== 完全化

```cpp
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

    // ★ 全フィールドを比較
    bool operator==(const AuthorityState& o) const noexcept {
        return publicationSequenceId == o.publicationSequenceId
            && publicationEpoch == o.publicationEpoch
            && mappedRuntimeGeneration == o.mappedRuntimeGeneration
            && hasActiveRuntime == o.hasActiveRuntime
            && hasPendingPublication == o.hasPendingPublication
            && hasActiveCrossfade == o.hasActiveCrossfade
            && runtimeMissing == o.runtimeMissing
            && persistentMissing == o.persistentMissing
            && fieldInconsistencyDetected == o.fieldInconsistencyDetected;
    }
    bool operator!=(const AuthorityState& o) const noexcept { return !(*this == o); }
};
```

---

## 第7章: 独立 Validator（operator== 非依存）

```cpp
// ★ validateAuthorityStateMatch: operator== とは独立した検証
//   フィールド単位で比較するため、operator== のバグの影響を受けない
[[nodiscard]] bool validateAuthorityStateMatch(
    const AuthorityState& observed,
    const AuthorityState& expected) noexcept
{
    // 独立したフィールド単位の比較
    bool match = true;
    if (observed.publicationSequenceId != expected.publicationSequenceId) match = false;
    if (observed.publicationEpoch != expected.publicationEpoch) match = false;
    if (observed.mappedRuntimeGeneration != expected.mappedRuntimeGeneration) match = false;
    if (observed.hasActiveRuntime != expected.hasActiveRuntime) match = false;
    if (observed.runtimeMissing != expected.runtimeMissing) match = false;
    if (observed.fieldInconsistencyDetected != expected.fieldInconsistencyDetected) match = false;
    return match;
}
```

---

## 第8章: getVersion() 変更なし

```cpp
// ★ v4.14: 変更なし。直接 mappedRuntimeGeneration を返す。
//   snapshot 失敗時の 0 返却は行わない。
//   getVersion() はテスト専用であり、移行後の影響はテスト17箇所の修正のみ。
std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    return convo::consumeAtomic(mappedRuntimeGeneration_, std::memory_order_acquire);
}
```

---

## 第9章: PHASE 計画（簡略化）

```
Phase-0: 方式決定（seqlock 採用）
  変更: なし

Phase-1: 基盤導入
  1a: PersistentStateBlock（seqlock、_mm_pause なし、ScopedGuard 説明修正）
  1b: AuthorityDescriptor
  1c: deriveAuthorityState + reconcileAuthorityState（deriveExpectedState 削除）
  1d: validateAuthorityStateMatch（独立 Validator）

Phase-2: currentWorld_ 段階的削除
  2a: getCurrent → RuntimeStore 委譲
  2b: 全17テスト移行
  2c: getCurrent の currentWorld_ フォールバック削除
  2.5: 監査 + CI 禁止

Phase-3: Recovery 統合（reconcileAuthorityState 接続）

Phase-4: Invariant CI + currentWorld_ 完全削除
  4a-f: ISR-AUTH-001〜006 CI
  4g-i: commit/retire/currentWorld_ 削除

Phase-5: テスト拡充
  5a: Model-Based Test
  5b: Fault Injection 6シナリオ
```
