//==============================================================================
// DeletionQueue.cpp
//==============================================================================
#include "DeletionQueue.h"
#include "../ConvolverState.h"
#include <algorithm>

namespace convo {

void DeletionQueue::enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type)
{
    std::lock_guard<std::mutex> lock(mutex);
    queue.push_back({ptr, deleter, epoch, type});
}

void DeletionQueue::enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch)
{
    enqueue(ptr, deleter, epoch, DeletionEntryType::Generic);
}

void DeletionQueue::reclaim(uint64_t minEpoch)
{
    std::lock_guard<std::mutex> lock(mutex);
    auto it = std::stable_partition(queue.begin(), queue.end(),
        [minEpoch](Entry& e) -> bool {
            if (e.epoch >= minEpoch)
                return true;

            if (e.type == DeletionEntryType::ConvolverState)
            {
                auto* state = static_cast<ConvolverState*>(e.ptr);
                if (state != nullptr
                    && state->snapshotRefCount.load(std::memory_order_relaxed) > 0)
                {
                    return true;
                }
            }

            return false;
        }
    );

    for (auto iter = it; iter != queue.end(); ++iter) {
        if (iter->deleter && iter->ptr) {
            iter->deleter(iter->ptr);
        }
    }

    queue.erase(it, queue.end());
}

void DeletionQueue::reclaimAllIgnoringEpoch()
{
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& e : queue) {
        if (e.deleter && e.ptr) {
            e.deleter(e.ptr);
        }
    }
    queue.clear();
}

} // namespace convo
