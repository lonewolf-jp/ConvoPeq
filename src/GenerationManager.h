// src/GenerationManager.h
// Phase 0: タスク世代（バージョン）管理クラス
//
// 設計思想:
//   - 設定変更のたびに世代番号を単調増加させ、古いバックグラウンドタスクの
//     結果が新しい状態に上書きされることを防ぐ。
//   - スレッドセーフ: 全操作が atomic なので、どのスレッドからも安全に呼び出せる。
//
// 使用パターン:
//   Writer側: bumpGeneration() で世代を進め、戻り値をタスクに渡す。
//   Task側:  タスク完了時に isCurrentGeneration() で陳腐化チェック。
//
// スレッド安全性: ✅ Lock-free / Wait-free（全操作 atomic）
// Audio Thread 使用: ✅ 安全（atomic load のみ）
#pragma once
#include <atomic>
#include <cstdint>

class GenerationManager
{
public:
    GenerationManager() = default;
    ~GenerationManager() = default;

    // 設定変更時に呼ぶ（世代を進める）
    // @return 新しい世代番号（タスク起動時に保存しておく）
    uint64_t bumpGeneration() noexcept
    {
        return ++currentGeneration;
    }

    // タスク開始時に自分の世代を取得（キャプチャ）
    uint64_t getCurrentGeneration() const noexcept
    {
        return currentGeneration.load(std::memory_order_acquire);
    }

    // タスク完了時にチェック（現在の世代と一致するか）
    // @param taskGen  タスク起動時に bumpGeneration() から得た世代番号
    // @return true = まだ有効（最新世代）、false = 陳腐化済み（破棄すべき）
    bool isCurrentGeneration(uint64_t taskGen) const noexcept
    {
        return taskGen == currentGeneration.load(std::memory_order_acquire);
    }

private:
    std::atomic<uint64_t> currentGeneration{0};

    // コピー・ムーブ禁止（所有権が明確な単一インスタンスで使用する想定）
    GenerationManager(const GenerationManager&)            = delete;
    GenerationManager& operator=(const GenerationManager&) = delete;
    GenerationManager(GenerationManager&&)                 = delete;
    GenerationManager& operator=(GenerationManager&&)      = delete;
};
