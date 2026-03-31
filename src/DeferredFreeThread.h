// src/DeferredFreeThread.h
#pragma once
#include "SafeStateSwapper.h"
#include <thread>
#include <atomic>
#include <chrono>

class DeferredFreeThread {
public:
    DeferredFreeThread(SafeStateSwapper& swapper) : swapperRef(swapper), running(true) {
        thread = std::thread([this]() { run(); });
    }

    ~DeferredFreeThread() {
        running = false;
        if (thread.joinable()) thread.join();
        // 残りの強制解放
        while (auto* ptr = swapperRef.tryReclaim(UINT64_MAX)) {
            delete ptr;
        }
    }

private:
    void run() {
        while (running) {
            uint64_t minEpoch = swapperRef.getMinReaderEpoch();
            if (auto* ptr = swapperRef.tryReclaim(minEpoch)) {
                delete ptr;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    SafeStateSwapper& swapperRef;
    std::atomic<bool> running;
    std::thread thread;
};
