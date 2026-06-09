# 未確定事項 最終確定レポート v5

**作成日**: 2026-06-09
**調査手段**: CodeGraph MCP, Serena MCP, grep, 直接読取

---

## 調査結果一覧

| # | 項目 | 確定内容 | 改修影響 |
| --- | --- | --- | --- |
| 1 | Pressure定数 | `kPressureSlopeThreshold = 8`, `kPressureNormalizeWindows = 3` | P1-6 基礎データ確定 |
| 2 | Sequence監視現状 | **monotonicity violation 検出なし**。Orchestrator の stale discard は deferred 専用 | P1-9 新規追加の必要性確認 |
| 3 | DeletionEntry enqueue | `DeferredDeletionQueue::enqueue(ptr, deleter, epoch, type)` — publicationSequenceId なし | P2-2 の改修範囲確定 |
| 4 | ReaderSlot | `epoch` + `depth` のみ。`enterTimestamp` なし | P3-1 の改修範囲確定 |
| 5 | runtimeStore 全アクセス | observe() 9箇所, publishAndSwap は Coordinator 経路のみ。Store自体は public | P0-4 の Store封鎖範囲確定 |

---

## 詳細

### 1. Pressure定数

```cpp
// ISRRuntimePublicationCoordinator.h:103-104
static constexpr std::uint64_t kPressureSlopeThreshold = 8;       // backlog増加傾斜閾値
static constexpr std::uint32_t kPressureNormalizeWindows = 3;     // Pressure→Ready復帰ウィンドウ数
```

状態遷移: `Ready → (slope > 8) → Pressure → (3回のnormalize window経過) → Ready`

### 2. Sequence Integrity — 現状

`publicationSequenceCounter_` と `lastCommittedPublicationSequence_` は存在する:

```cpp
// AudioEngine.h:1568-1570
std::atomic<PublicationSequenceId> publicationSequenceCounter_ { 0 };
std::atomic<PublicationSequenceId> lastCommittedPublicationSequence_ { 0 };
```

発番は `reserveRuntimePublicationIdentity()`:

```cpp
identity.publicationSequence = fetchAddAtomic(publicationSequenceCounter_, 1, acq_rel) + 1;
```

確定は `AudioEngine.Commit.cpp:377`:

```cpp
publishAtomic(lastCommittedPublicationSequence_, world.publication.sequenceId, release);
```

**しかし newSequence <= lastCommittedSequence の検出は存在しない。**

Orchestrator の stale discard は deferred publish 専用であり、一般 publish の monotonicity を監視していない。

### 3. DeletionEntry

```cpp
// DeferredDeletionQueue.h:25-30
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
};
```

publicationSequenceId フィールドは存在しない。enqueue 時に記録する方法がない。

### 4. ReaderSlot

```cpp
// EpochDomain.h:237-240
struct ReaderSlot {
    std::atomic<uint64_t> epoch { kInactiveEpoch };
    std::atomic<uint32_t> depth { 0 };
};
```

`enterTimestamp` なし。Reader がいつ入ったかの追跡不可。

### 5. runtimeStore アクセスパターン

Store へのアクセス:

- **読取 (observe)**: 9箇所 — すべて static メソッド `RuntimePublicationCoordinator::consumeWorldHandle()` / `consumePublishedWorld()` 経由
- **書換 (publishAndSwap)**: Coordinator::publishWorld() 内部のみ。WriteAccess は friend Owner で制限
- **Store自体**: `RuntimePublishStore runtimeStore;` は public セクション

問題の本質: Store 自体の保護機構 (friend Owner) は正しいが、Coordinator が誰でも生成可能なため実質的な保護になっていない。

---

## 改修計画への反映

| 発見事項 | 反映内容 |
| --- | --- |
| Pressure定数確定 | P1-6 の設計パラメータとして利用可能 |
| Sequence監視不在 | P1-9 の必要性確認 |
| DeletionEntry構成確定 | P2-2 の改修範囲確定 |
| ReaderSlot構成確定 | P3-1 の改修範囲確定 |
| Storeアクセス全容確定 | P0-4 の移行範囲確定 |
