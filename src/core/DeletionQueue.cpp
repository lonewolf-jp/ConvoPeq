//==============================================================================
// DeletionQueue.cpp
//==============================================================================
#include "../DeferredDeletionQueue.h"   // 先頭付近に追加
#include "DeletionQueue.h"

namespace convo {

void DeletionQueue::enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (count >= kCapacity)
    {
        // 容量超過: テールのエントリを強制実行してキューを空ける
        // (安全側に倒れる前に deleter を呼び出す)
        if (deleter && ptr)
            deleter(ptr);
        return;
    }
    queue[count++] = {ptr, deleter, epoch, type};
}

void DeletionQueue::reclaim(uint64_t minReaderEpoch)
{

    std::lock_guard<std::mutex> lock(mutex);

    size_t write = 0;
    for (size_t i = 0; i < count; ++i)
    {
        Entry& e = queue[i];
        // [P1-21] isOlder をインライン展開 (EpochDomain非依存)
        const bool safeToDelete = static_cast<int64_t>(e.epoch - minReaderEpoch) < 0;
        if (safeToDelete)
        {
            // 安全に解放可能
            if (e.deleter && e.ptr)
                e.deleter(e.ptr);
            // write を進めない（スロットを空ける）
        }
        else
        {
            // まだ保持中: 可能なら先頭へ密辺める
            if (write != i)
                queue[write] = e;
            ++write;
        }
    }
    // 使用済みスロットをクリア
    for (size_t i = write; i < count; ++i)
        queue[i] = {};
    count = write;
}

} // namespace convo
