#pragma once

#include <vector>
#include <functional>
#include <mutex>
#include "EpochManager.h"

namespace convo {

class EBRQueue
{
public:
    struct Retired
    {
        void* ptr;
        std::function<void(void*)> deleter;
        uint64_t epoch;
    };

    void retire(void* p, std::function<void(void*)> d)
    {
        if (p == nullptr) return;
        
        std::lock_guard<std::mutex> lock(queueMutex);
        uint64_t e = EpochManager::instance().currentEpoch();
        retired.push_back({p, d, e});
    }

    void tryReclaim()
    {
        uint64_t minEpoch = EpochManager::instance().minActiveEpoch();

        std::lock_guard<std::mutex> lock(queueMutex);
        size_t write = 0;

        for (size_t i = 0; i < retired.size(); ++i)
        {
            if (EpochManager::isOlder(retired[i].epoch, minEpoch))
            {
                retired[i].deleter(retired[i].ptr);
            }
            else
            {
                if (write != i)
                    retired[write] = std::move(retired[i]);
                write++;
            }
        }

        retired.resize(write);
    }
    
    static EBRQueue& instance()
    {
        static EBRQueue inst;
        return inst;
    }

private:
    std::mutex queueMutex;
    std::vector<Retired> retired;
};

// Global shorthand
inline void retireObject(void* p, std::function<void(void*)> d)
{
    EBRQueue::instance().retire(p, d);
}

} // namespace convo
