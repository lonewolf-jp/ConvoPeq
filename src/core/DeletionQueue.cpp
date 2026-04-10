//==============================================================================
// DeletionQueue.cpp
//==============================================================================
#include "DeletionQueue.h"
#include <algorithm>

namespace convo {

void DeletionQueue::enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch)
{
    std::lock_guard<std::mutex> lock(mutex);
    queue.push_back({ptr, deleter, epoch});
}

void DeletionQueue::reclaim(uint64_t minEpoch)
{
    std::lock_guard<std::mutex> lock(mutex);
    auto it = std::remove_if(queue.begin(), queue.end(),
        [minEpoch](const Entry& e) { return e.epoch < minEpoch; }
    );

    // 削除対象のエントリを実際に解放
    for (auto iter = it; iter != queue.end(); ++iter) {
        if (iter->deleter && iter->ptr) {
            iter->deleter(iter->ptr);
        }
    }

    queue.erase(it, queue.end());
}

} // namespace convo
