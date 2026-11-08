//==============================================================================
// LoaderCurrencyContractTests.cpp — 2026-11-08
//
// 目的:
//   ConvolverProcessor のローダー世代契約を、実際に使用されている
//   GenerationManager（src/GenerationManager.h）に対して固定する。
//
// 背景（調査結果）:
//   LoaderThread.cpp の旧コメントは「ロードループ内の generation check は実装待ち」と
//   書いていたが、stale completion の正当性は commit 直前の
//   convolverStateGeneration.isCurrentGeneration(requestGenId) が担っている
//   （ConvolverProcessor.LoadPipeline.cpp: bumpGeneration / applyComputedIR /
//     executePendingCommit / updateConvolverState）。
//   ConvolverProcessor::loaderGeneration_ は加算のみで読み出しが無く、この契約に
//   寄与していなかったため削除した（ConvolverProcessor.h / .Lifecycle.cpp）。
//
//   本テストは「新しい要求が古い要求を無効化する」という GenerationManager の
//   セマンティクスを固定し、将来 loaderGeneration_ のような別カウンタを再導入して
//   二重管理に戻ることを防ぐ。engine を起動しないため RT / MKL に依存しない。
//==============================================================================
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "GenerationManager.h"

namespace {

[[nodiscard]] bool testBumpInvalidatesOlderGeneration()
{
    GenerationManager manager;
    // 初期世代（0）をタスクが捕捉
    const std::uint64_t taskGen = manager.getCurrentGeneration();
    if (!manager.isCurrentGeneration(taskGen))
        return false;

    // 新しい要求 → 古いタスクは陳腐化
    const std::uint64_t newGen = manager.bumpGeneration();
    if (newGen == taskGen)
        return false;
    if (manager.isCurrentGeneration(taskGen))
        return false;                       // stale は拒否される
    if (!manager.isCurrentGeneration(newGen))
        return false;                       // 最新は有効

    // さらに新しい要求 → 直前の世代も陳腐化（単調増加の連鎖）
    const std::uint64_t newest = manager.bumpGeneration();
    if (manager.isCurrentGeneration(newGen))
        return false;
    if (!manager.isCurrentGeneration(newest))
        return false;
    return true;
}

// 複数スレッドが bump しても、最後に確定した世代以外は必ず stale と判定される。
// （commit 側の isCurrentGeneration が「勝者 1 つだけ」を許すことの固定）
[[nodiscard]] bool testConcurrentBumpLeavesSingleWinner()
{
    GenerationManager manager;
    constexpr int kThreads = 8;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::uint64_t> generations(static_cast<std::size_t>(kThreads), 0);

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(kThreads));
    for (int i = 0; i < kThreads; ++i)
    {
        workers.emplace_back([&manager, &ready, &go, &generations, i]() {
            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire))
                std::this_thread::yield();
            generations[static_cast<std::size_t>(i)] = manager.bumpGeneration();
        });
    }
    while (ready.load(std::memory_order_acquire) != kThreads)
        std::this_thread::yield();
    go.store(true, std::memory_order_release);
    for (auto& worker : workers)
        worker.join();

    int currentCount = 0;
    for (const auto gen : generations)
    {
        if (manager.isCurrentGeneration(gen))
            ++currentCount;
    }
    // 全世代が一意（bump は原子的）で、最新の 1 つだけが current
    return currentCount == 1;
}

// タスクが走っている間に新しい要求が来た場合、途中の結果は採用されない。
// （applyComputedIR → executePendingCommit の 2 段ガードと同じ判定を模す）
[[nodiscard]] bool testInFlightTaskSeesStaleAfterNewRequest()
{
    GenerationManager manager;
    const std::uint64_t requestGen = manager.bumpGeneration();

    // ロード中の段階（変換完了）
    bool intermediateAccepted = manager.isCurrentGeneration(requestGen);
    if (!intermediateAccepted)
        return false;

    // ユーザーが別の IR を要求（= 新しい bump）
    const std::uint64_t nextRequest = manager.bumpGeneration();

    // commit 直前の判定: 途中結果は破棄、新しい要求だけが採用される
    if (manager.isCurrentGeneration(requestGen))
        return false;
    if (!manager.isCurrentGeneration(nextRequest))
        return false;
    return true;
}

} // namespace

int main()
{
    struct Case
    {
        const char* name;
        bool (*fn)();
    };
    const Case cases[] = {
        { "bump invalidates older generation", &testBumpInvalidatesOlderGeneration },
        { "concurrent bump leaves a single winner", &testConcurrentBumpLeavesSingleWinner },
        { "in-flight task becomes stale after a new request", &testInFlightTaskSeesStaleAfterNewRequest },
    };

    int failed = 0;
    for (const auto& testCase : cases)
    {
        if (!testCase.fn())
        {
            std::fprintf(stderr, "FAIL: %s\n", testCase.name);
            ++failed;
        }
        else
        {
            std::printf("PASS: %s\n", testCase.name);
        }
    }

    if (failed != 0)
    {
        std::fprintf(stderr, "LoaderCurrencyContractTests: %d case(s) failed\n", failed);
        return 1;
    }
    std::printf("LoaderCurrencyContractTests: ALL TESTS PASSED\n");
    return 0;
}
