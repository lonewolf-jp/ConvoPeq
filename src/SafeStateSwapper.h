// src/SafeStateSwapper.h
#pragma once
#include "ConvolverState.h"
#include <atomic>
#include <array>
#include <queue>
#include <mutex>
#include <algorithm>
#include <cstdint>

class SafeStateSwapper {
public:
    static constexpr size_t kMaxRetired = 64;
    static constexpr int kMaxReaders = 8; // 十分なマージン

    SafeStateSwapper() : globalEpoch(0) {
        for (auto& e : readerEpochs) e.store(0, std::memory_order_relaxed);
    }

    // スワップ処理（非RTスレッド用）
    void swap(ConvolverState* newState, uint64_t newEpoch) {
        ConvolverState* oldState = activeState.exchange(newState, std::memory_order_acq_rel);
        if (!oldState) return;

        size_t t = tail.load(std::memory_order_relaxed);
        size_t next = (t + 1) % kMaxRetired;
        
        if (next == head.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(fallbackMutex);
            fallbackQueue.push({ oldState, newEpoch });
            return;
        }

        retiredBuffer[t].state.store(oldState, std::memory_order_relaxed);
        retiredBuffer[t].epoch.store(newEpoch, std::memory_order_release);
        tail.store(next, std::memory_order_release);
    }

    // Reader 登録（Audio Thread 用）
    void enterReader(int readerIndex) {
        if (readerIndex >= 0 && readerIndex < kMaxReaders) {
            readerEpochs[readerIndex].store(
                globalEpoch.load(std::memory_order_acquire),
                std::memory_order_release
            );
        }
    }

    // Reader 解除（Audio Thread 用）
    void exitReader(int readerIndex) {
        if (readerIndex >= 0 && readerIndex < kMaxReaders) {
            readerEpochs[readerIndex].store(0, std::memory_order_release);
        }
    }

    // 状態取得（Audio Thread 用）
    ConvolverState* getState() const {
        return activeState.load(std::memory_order_acquire);
    }

    // 解放可能エントリの取得（Reclaimer Thread 用）
    ConvolverState* tryReclaim(uint64_t minReaderEpoch) {
        {
            std::lock_guard<std::mutex> lock(fallbackMutex);
            if (!fallbackQueue.empty()) {
                auto& entry = fallbackQueue.top();
                if (entry.epoch < minReaderEpoch) {
                    auto* ptr = entry.state;
                    fallbackQueue.pop();
                    return ptr;
                }
            }
        }

        size_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire)) return nullptr;

        auto& entry = retiredBuffer[h];
        uint64_t entryEpoch = entry.epoch.load(std::memory_order_acquire);
        
        if (entryEpoch < minReaderEpoch) {
            ConvolverState* ptr = entry.state.load(std::memory_order_acquire);
            head.store((h + 1) % kMaxRetired, std::memory_order_release);
            return ptr;
        }
        return nullptr;
    }

    // エポック進捗
    uint64_t bumpEpoch() {
        return globalEpoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    }

    // 最小 Reader エポック計算
    uint64_t getMinReaderEpoch() const {
        uint64_t minEpoch = UINT64_MAX;
        for (int i = 0; i < kMaxReaders; ++i) {
            uint64_t e = readerEpochs[i].load(std::memory_order_acquire);
            if (e != 0 && e < minEpoch) {
                minEpoch = e;
            }
        }
        return (minEpoch == UINT64_MAX) ? globalEpoch.load() : minEpoch;
    }

private:
    struct RetiredEntry {
        std::atomic<ConvolverState*> state{nullptr};
        std::atomic<uint64_t> epoch{0};
    };

    struct FallbackEntry {
        ConvolverState* state;
        uint64_t epoch;
        bool operator<(const FallbackEntry& other) const {
            return epoch > other.epoch; // Min-Heap 用
        }
    };

    std::atomic<ConvolverState*> activeState{nullptr};
    std::array<RetiredEntry, kMaxRetired> retiredBuffer;
    std::atomic<size_t> head{0};
    std::atomic<size_t> tail{0};
    
    std::atomic<uint64_t> globalEpoch{0};
    std::array<std::atomic<uint64_t>, kMaxReaders> readerEpochs;

    std::mutex fallbackMutex;
    std::priority_queue<FallbackEntry> fallbackQueue;
};
