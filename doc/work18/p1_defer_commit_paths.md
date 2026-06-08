# P1 Phase1-A 前準備: Defer Commit 経路全列挙

**AUDIT_DATE**: 2026-06-05
**AUDITOR**: GitHub Copilot (AI Assistant)
**SEARCH_METHOD**: grep (src/**) for appendPublicationIntentForCommit, commitNewDSP, defer
**STATUS**: 完了

---

## 概念

現在の commit は 2 段階化されている:

1. **Producer 経路**: `enqueuePublicationIntentForRuntimeCommit` → `appendPublicationIntentForCommitProducer` → PublicationLog に enqueue
2. **Consumer 経路**: `commitNewDSP` 内で crossfade アクティブ検出時 → `appendPublicationIntentForCommitConsumer` → PublicationLog に enqueue（defer）
3. **Drain 経路**: `drainPublicationIntentsForRuntimeCommit` が PublicationLog を消費 → `applyRuntimeCommitFromIntent` を実行

## 経路詳細

### 経路 A: Producer → Consumer（通常 publish）

```
rebuildThreadLoop()
  → enqueuePublicationIntentForRuntimeCommit(dspToCommit, generation, sealedSnapshot)
    → appendPublicationIntentForCommitProducer(newDSP, targetWorldId, sealedSnapshot)
      → appendPublicationIntentForCommitSlot(..., CommitReaderSlot::Producer, ...)
        → PublicationLog (CAS linked-list enqueue)
    → triggerAsyncUpdate()
      → handleAsyncUpdate()
        → drainPublicationIntentsForRuntimeCommit()
          → PublicationLog を消費
          → applyRuntimeCommitFromIntent(next->newDSP, generation, sealedSnapshot)
            → commitNewDSP(newDSP, generation, sealedSnapshot)
              → setActiveRuntimeDSP(newDSP)
              → buildRuntimePublishWorld + coordinator.publishWorld()
```

### 経路 B: Consumer → 再Consumer（defer commit）

```
commitNewDSP(newDSP, generation, sealedSnapshot) 内
  → crossfade アクティブ検出
    → appendPublicationIntentForCommitConsumer(newDSP, generation, sealedSnapshot)
      → appendPublicationIntentForCommitSlot(..., CommitReaderSlot::Consumer, ...)
        → PublicationLog (CAS linked-list enqueue)
    → return (publish せず defer)
```

### 経路 C: drainPublicationLogForShutdown（シャットダウン時）

```
drainPublicationLogForShutdown()
  → PublicationLog の全エントリを取得
  → 各エントリの retireDSP(next->newDSP) または applyRuntimeCommitFromIntent
  → PublicationLog をクリア
```

### PublicationLog 構造

```cpp
struct PublicationLog {
    std::atomic<PublicationIntent*> head;       // 先頭ポインタ
    std::atomic<PublicationIntent*> consumedTail; // 消費済みテール
    std::atomic<PublicationIntent*> retiredHead;  // 退役済み先頭
};

struct PublicationIntent {
    DSPCore* newDSP;
    std::uint64_t targetWorldId;
    std::uint64_t requestId;
    std::int64_t enqueueTimeTicks;
    convo::RuntimeBuildSnapshot runtimeBuildSnapshot;
    std::atomic<PublicationIntent*> next;  // CAS でつなぐ単方向リンクリスト
};
```

### 関連メトリクス

- `publicationBacklog_` (atomic uint64): バックログ有無
- `commitDrainInProgress` (atomic bool): 排他制御
- `RuntimePublicationCoordinator::publicationBacklogCount_`: 監視用コピー
- `RuntimePublicationCoordinator::pendingIntentCount_`: 監視用コピー
- `setPublicationBacklogCount()`, `setPendingIntentCount()`: bridge 更新

---

## 総括

- Defer commit 経路（Consumer）は `commitNewDSP` 内で crossfade アクティブ時にのみ発行される（1箇所: AudioEngine.Commit.cpp:1184）。
- PublicationLog は MPMC CAS linked-list で実装され、Producer スレッド（RebuildDispatch）と Consumer スレッド（Message Thread, handleAsyncUpdate）の間で commit を安全に受け渡す。
- Phase1-A ではこの 2 段階 commit 構造を維持したまま、`enqueuePublicationIntentForRuntimeCommit` を coordinator.publishWorld() への委譲ラッパーに変更する。PublicationLog 経路は Phase1-B で完全削除予定。
