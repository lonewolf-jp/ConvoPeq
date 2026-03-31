// src/GenerationManager.h
#pragma once
#include <atomic>
#include <cstdint>

class GenerationManager {
public:
    // 設定変更時に呼ぶ（世代を進める）
    uint64_t bumpGeneration() { 
        return ++currentGeneration; 
    }
    
    // タスク開始時に自分の世代を取得（キャプチャ）
    uint64_t getCurrentGeneration() const { 
        return currentGeneration.load(std::memory_order_acquire); 
    }
    
    // タスク完了時にチェック（現在の世代と一致するか）
    bool isCurrentGeneration(uint64_t taskGen) const { 
        return taskGen == currentGeneration.load(std::memory_order_acquire); 
    }

private:
    std::atomic<uint64_t> currentGeneration{0};
};
