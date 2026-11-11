//==============================================================================
// DeletionQueue.cpp
//==============================================================================
#include "../DeferredDeletionQueue.h"   // 先頭付近に追加
#include "DeletionQueue.h"
#include "../audioengine/ISRRetireLatency.h"   // ★ A(D1): 退役要求→解放完了の測定

namespace convo {

void DeletionQueue::enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (count >= kCapacity)
    {
        // 容量超過: テールのエントリを強制実行してキューを空ける
        // (安全側に倒れる前に deleter を呼び出す)
        if (deleter && ptr)
        {
            // ★ domain tag: レガシー DeletionQueue は本設計の対象外（到達しない）。
            //   万一到達した場合は「ドメイン未特定」として Unknown を記録する。
            convo::isr::RetireLatencyRecorder::instance().recordComplete (
                ptr, convo::isr::RetireLatencyRecorder::Path::QueueOverflow,
                convo::RetireDomainId::Unknown);   // ★ A(D1)
            deleter(ptr);
        }
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
            {
                // ★ domain tag: レガシー経路（対象外・到達しない）。未特定として Unknown。
                convo::isr::RetireLatencyRecorder::instance().recordComplete (
                    e.ptr, convo::isr::RetireLatencyRecorder::Path::Queue,
                    convo::RetireDomainId::Unknown);   // ★ A(D1)
                e.deleter(e.ptr);
            }
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
