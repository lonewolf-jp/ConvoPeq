# P7 – Retire Ordering Contract（退役順序契約）

**最終更新**: 2026-06-05
**ステータス**: 文書化完了（実装確認済み）

---

## 1. 契約概要

RuntimePublishWorld の退役（retire）は以下の順序制約に従う：

1. **先入先出（FIFO）**: 古い World から順に退役される。
2. **生成元紐付け**: 各 World は `generation` / `activationEpoch` / `publicationSequenceId` で一意に識別され、退役時にこれらの値が不変であることが保証される。
3. **callback 契約**: `willRetireRuntimeNonRt` → `retireRuntimePublishWorldNonRt` の順で呼ばれる。

## 2. 実装確認

### 2.1 RuntimePublicationCoordinator (template) — `core/RuntimePublicationCoordinator.h`

```cpp
void publishWorld(aligned_unique_ptr<World> worldOwner) noexcept
{
    // 1. validate
    // 2. publishAndSwap (newWorld を current に、oldWorld を取得)
    // 3. didPublishRuntimeNonRt(newWorld)
    // 4. willRetireRuntimeNonRt(oldWorld)
    // 5. retireRuntimePublishWorldNonRt(oldWorld, false)
}
```

退役順序:

- `publishAndSwap()` がアトミックに oldWorld を取得
- `willRetireRuntimeNonRt` コールバック（pre-retire hook）
- `retireRuntimePublishWorldNonRt` で実際のメモリ解放（post-retire）

### 2.2 ISR RuntimePublicationCoordinator — `ISRRuntimePublicationCoordinator.cpp`

```cpp
void retire(RetireAuthority, RuntimeBoundary boundary, const void* oldWorld)
{
    // 1. currentWorld が oldWorld と一致する場合のみ CAS で null に
    // 2. retireBacklogCount を +1
}
```

退役順序:

- CAS で currentWorld のクリア（安全な所有権移譲）
- カウンタ更新（非同期退役パイプライン制御）

### 2.3 RetireRuntime — `ISRRetire.h/.cpp`

```cpp
RetireEnqueueResult enqueueRetire(RetireAuthority, EpochDomain& domain, void* ptr,
                                   void (*deleter)(void*), std::uint64_t epoch) noexcept
{
    // 1. retireAuthorityCount を +1
    // 2. domain.enqueueRetire() で EpochDomain に登録
    // 3. retireBacklogCount を +1
}
```

退役順序:

- EpochDomain への enqueue（指定 epoch まで退役を遅延可能）
- バックログカウンタ更新

## 3. 順序保証

| レイヤー | 順序保証 | 備考 |
|---------|---------|------|
| Coordinator (template) | publishAndSwap → didPublish → willRetire → retire | アトミックに oldWorld 確定後に callback |
| Coordinator (ISR) | CAS clear → backlog++ | CAS が成功した world のみ退役 |
| EpochDomain | epoch 順に処理 | 非同期退役のタイミング制御 |

## 4. 制約と注意点

- **同一 World の二重退役防止**: CAS 機構により、`currentWorld` に設定されている World のみ退役可能。
- **非同期退役**: `RetireAuthority` 経由の退役は `EpochDomain` に defer され、epoch 条件が満たされた後に実行される。
- **シャットダウン時**: `drainPublicationLogForShutdown()` で全未処理 Intent の退役を強制実行。
