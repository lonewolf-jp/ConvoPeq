# P1 Phase1-A 前準備: enqueuePublicationIntentForRuntimeCommit 呼び出し元全列挙

**AUDIT_DATE**: 2026-06-05
**AUDITOR**: GitHub Copilot (AI Assistant)
**SEARCH_METHOD**: grep (src/**), codegraph (caller graph)
**STATUS**: 完了

---

## 定義

`AudioEngine.h:1737` — `AudioEngine::enqueuePublicationIntentForRuntimeCommit(DSPCore*, int, const convo::RuntimeBuildSnapshot&)`

`AudioEngine.Commit.cpp:749` — 実装。現在の動作:

1. null DSP チェック
2. `acceptsRuntimePublication()` で reject 可能性を判定 → reject 時は `publicationRejectCount_` を増加し `retireDSP(newDSP)` で退役
3. `appendPublicationIntentForCommitProducer(newDSP, generation, sealedSnapshot)` で PublicationLog に enqueue
4. `triggerAsyncUpdate()` でメッセージスレッドの `drainPublicationIntentsForRuntimeCommit()` をトリガ

## 呼び出し元一覧

### 1. `AudioEngine.RebuildDispatch.cpp:824`

```cpp
enqueuePublicationIntentForRuntimeCommit(dspToCommit, task.generation, task.runtimeBuildSnapshot);
```

コンテキスト: `rebuildThreadLoop()` 内。リビルドスレッドで新しい DSP をビルド後、コミットのために enqueue。

---

## 関連する PublicationLog 経路

### `appendPublicationIntentForCommitSlot` の呼び出し元

- `appendPublicationIntentForCommitProducer()` → `appendPublicationIntentForCommitSlot(..., CommitReaderSlot::Producer, ...)`
  - → `enqueuePublicationIntentForRuntimeCommit()` からのみ呼ばれる
- `appendPublicationIntentForCommitConsumer()` → `appendPublicationIntentForCommitSlot(..., CommitReaderSlot::Consumer, ...)`
  - → `AudioEngine.Commit.cpp:1184` から呼ばれる（commitNewDSP 内、defer commit 経路）

### `drainPublicationIntentsForRuntimeCommit` の呼び出し元

- `handleAsyncUpdate()` 経由（AsyncUpdater コールバック）
- 明示的呼び出しはなし

### `applyRuntimeCommitFromIntent` の呼び出し元

- `drainPublicationIntentsForRuntimeCommit()` 内のループからのみ呼ばれる

---

## 総括

- `enqueuePublicationIntentForRuntimeCommit` の直接呼び出しは **1箇所**（RebuildDispatch.cpp:824）のみ。
- これはリビルドスレッドから非同期コミットを発行する唯一の経路。
- Phase1-A でこの関数を委譲ラッパ化する場合、以下の変更が必要:
  - `enqueuePublicationIntentForRuntimeCommit` 内で `coordinator.publishWorld()` を呼び出す（直接 publish に変更）
  - PublicationLog, appendPublicationIntentForCommitSlot, applyRuntimeCommitFromIntent の連鎖は Phase1-B で削除
  - 当面は既存コードとの互換性維持のため、PublicationLog を維持しつつ coordinator.publishWorld() への委譲を追加
