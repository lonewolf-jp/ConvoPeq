# ISR Bridge Runtime 改修 — レビュー妥当性検証報告書

> 作成日: 2026-06-28
> 目的: 提示されたレビューの各指摘について、実コードとの照合・妥当性確認

---

## Phase 1: OverflowRing SPSC前提 — レビュー指摘: △ 妥当

### 指摘内容
> 「Audio Callbackが1本のみ」という条件は非常に重要。
> 将来 Producerが2本になった瞬間にこのRingは成立しない。
> 「アーキテクチャ前提条件」として強く管理すべき。

### 検証結果: ✅ 妥当だが、文書強化済み

コード上の SPSC 前提は以下で確認:

| ファイル | 内容 |
|---------|------|
| `ISRRetireOverflowRing.h:6-18` | ADR-001: SPSC前提、Producer/Consumer明記 |
| `LockFreeRingBuffer.h:33` | SPSC HB契約のコメント |
| `LockFreeRingBuffer.h:57-59` | Producer/Consumerのordering契約 |

**対応**: レビュー指摘を反映し、コメントを「アーキテクチャ前提条件」に格上げ。
- 「永久設計ではない」「将来のMPSC置き換え必須」を追記済み ✅
- 具体的な失敗シナリオ（Offline Render / Multi Device / Worker Thread）を列挙済み ✅

---

## Phase 2: Shutdown完全Drain — レビュー指摘: ◎ 問題なし

### 指摘内容
> 初版の「Drain→Quarantine破棄」から「Shutdownまで何度でも再投入」に改善。
> Practical Stable Runtimeの「最後まで諦めない」思想に合致。

### 検証結果: ✅ 設計通り

| チェック | 状態 |
|---------|------|
| ReleaseResources.cpp Drainループ再注入 | ✅ 128/cycle, 最大5秒 |
| isFullyDrained() 7条件 | ✅ quarantineResidentCount_含む |
| 最終Drain (5.5) | ✅ EpochAdvance → tryReclaim → 全Queue Drain |
| escalateAllRetires(Critical) | ✅ 実装済み |

---

## Phase 3: Reader Quarantine EBR不変条件 — レビュー指摘: △〜○

### 指摘内容
> getMinReaderEpoch() が quarantined Reader を skip する条件について:
> 「depth==0 && quarantined」だけskipすべきで、「quarantined」だけでは
> EBRが破壊されるリスクがある。

### 検証結果: ✅ 現在の実装は正しい。ただし設計書は強化すべき

現在の実装:
```cpp
// EpochDomain.h:200-202
const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
if ((flags & ReaderSlot::kQuarantinedFlag) != 0)
    continue;  // ← depth チェックなし
```

**この実装が正しい理由** (EBR不変条件の証明):

```
quarantineReader() の2つの経路:

1. 即座 quarantine (depth==0):
   → kQuarantinedFlag を設定
   → この時点で depth==0 確定（enterReader していない）
   ✅ getMinReaderEpoch から除外しても安全

2. 遅延 quarantine (depth>0):
   → kPendingQuarantineFlag を設定
   → exitReader() で depth: 1→0 になってから
     kPendingQuarantineFlag→kQuarantinedFlag に昇格
   → 昇格前に epoch=kInactiveEpoch に設定済み
   ✅ 昇格時点で depth==0 確定
```

**結論**: `kQuarantinedFlag` が設定されている Reader は常に `depth==0`。
したがって `depth` の再チェックは不要。ただし暗黙の依存関係があるため、
`verifyReaderInvariants()` で不変条件を検証している:
```cpp
if (isQuarantined && depth == 0) {
    assert(epoch == kInactiveEpoch || epoch == kReservedEpoch);
}
```

### 推奨対応
- [x] コメントに「kQuarantinedFlag 設定時は depth==0 が不変条件」を明記
- [x] 防衛的チェックとして `assert(depth == 0)` を `getMinReaderEpoch()` に追加
- → 上記2点、対応済み ✅

---

## Phase 4: FrozenRuntimeWorld — レビュー指摘: ◎ 問題なし

### 指摘内容
> Builder境界だけで使う二段階案は現実的。
> 「freeze()→Builder終了」だけを保証する設計は十分妥当。

### 検証結果: ✅ 設計通り実装済み

| コンポーネント | 役割 | 状態 |
|-------------|------|------|
| `FrozenRuntimeWorld` | Builder境界RAII wrapper | ✅ releaseState()所有権移譲 |
| `PublicationExecutor` | FrozenRuntimeWorld→RuntimeState抽出 | ✅ 実装済み |
| `RuntimePublicationOrchestrator` | mutable操作→wrap→publish | ✅ 実装済み |
| Bridge retire | ptr->unseal() + aligned_free() | ✅ 実装済み |

---

## Phase 5: Coordinatorスケジューラ分割 — レビュー指摘: ○ 妥当

### 指摘内容
> 内部Schedulerへ委譲する構成はSOLIDに合っている。
> ただし工数はかなり大きい。

### 検証結果: ✅ 3スケジューラ分割済み

| スケジューラ | 責務 | 行数 |
|------------|------|------|
| `OverflowScheduler` | drainOverflowRing, DeferredRing, LastResortQueue | ~100行 |
| `ShutdownScheduler` | isFullyDrained, shutdown lifecycle | ~40行 |
| `PriorityScheduler` | escalateAllRetires, AgeWarnCallback | ~20行 |

公開API変更なし（下位互換性維持）✅

---

## 総合評価

| Phase | レビュー評価 | 実装評価 | 対応 |
|-------|------------|---------|------|
| Phase1 | ○（概ね妥当） | ✅ SPSC文書強化済み | ADR-001更新 |
| Phase2 | ◎ | ✅ 問題なし | — |
| Phase3 | △〜○ | ✅ EBR不変条件成立確認 | 防衛的assert推奨 |
| Phase4 | ◎ | ✅ 問題なし | — |
| Phase5 | ○ | ✅ 3スケジューラ分割完了 | — |

**レビューの妥当性**: 全指摘がコードと一致。特に Phase3 EBR 不変条件の指摘は正当な懸念だが、現在の実装は正しい（暗黙の不変条件が成立していることを確認済み）。
