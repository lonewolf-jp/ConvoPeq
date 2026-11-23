//============================================================================
// LearnerStateAuthorityTests.cpp — B-07 の同期契約テスト
//
// 目的:
//   * 復旧要求は「学習スレッドが動作中」のときだけキューに残り、境界で 1 回だけ
//     取り出されること（取りこぼし・二重適用が無いこと）。
//   * 停止中の投入は保持されないこと。
//   * 公開スナップショットが中途半端な状態（torn read）を返さないこと。
//   * withExclusive() の中で積んだ要求が、その後に取り出せること。
//
// 本テストは JUCE に依存しない（実際に複数スレッドを動かす）。
//============================================================================
#include "learner/LearnerStateAuthority.h"
#include "NoiseShaperLearnerTypes.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

namespace
{
using State = convo::NoiseShaperLearnerState;

int gFailures = 0;

void check (bool ok, const char* id, const char* detail)
{
    std::printf ("[%s] %s %s\n", ok ? "PASS" : "FAIL", id, detail);
    if (! ok)
        ++gFailures;
}

State makeState (int marker)
{
    State s {};
    for (int i = 0; i < 9; ++i)
    {
        s.mean[i] = static_cast<double> (marker) + 0.25 * i;
        s.bestCoefficients[i] = -static_cast<double> (marker) - 0.5 * i;
    }
    for (int i = 0; i < 45; ++i)
        s.covarianceUpperTriangle[i] = static_cast<double> (marker * 3 + i);
    s.sigma = 0.1 + 0.001 * marker;
    s.elapsedPlaybackSeconds = 1.5 * marker;
    s.currentPhase = marker % 3 + 1;
    s.iteration = marker * 7;
    s.bestScore = -static_cast<double> (marker);
    s.processCount = marker;
    s.totalGenerations = static_cast<std::uint64_t> (marker) * 11;
    s.learningMode = marker % 6;
    return s;
}

bool sameState (const State& a, const State& b)
{
    for (int i = 0; i < 9; ++i)
        if (a.mean[i] != b.mean[i] || a.bestCoefficients[i] != b.bestCoefficients[i])
            return false;

    for (int i = 0; i < 45; ++i)
        if (a.covarianceUpperTriangle[i] != b.covarianceUpperTriangle[i])
            return false;

    return a.sigma == b.sigma
        && a.elapsedPlaybackSeconds == b.elapsedPlaybackSeconds
        && a.currentPhase == b.currentPhase
        && a.iteration == b.iteration
        && a.bestScore == b.bestScore
        && a.processCount == b.processCount
        && a.totalGenerations == b.totalGenerations
        && a.learningMode == b.learningMode;
}

// R-07-A: 停止中は保持しない
void testStoppedDoesNotQueue()
{
    convo::LearnerStateAuthority<State> auth;
    const auto s = makeState (1);

    const bool queued = auth.enqueueRestoreIfRunning (s, false);
    check (! queued && ! auth.hasPendingRestore() && auth.submittedCount() == 0,
           "R-07-A", "stopped submit is not queued");
}

// R-07-B: 動作中は最新 1 件だけ保持し、境界で 1 回だけ取り出せる
void testLatestWinsAndSingleTake()
{
    convo::LearnerStateAuthority<State> auth;

    for (int v = 1; v <= 5; ++v)
        (void) auth.enqueueRestoreIfRunning (makeState (v), true);

    check (auth.submittedCount() == 5 && auth.hasPendingRestore(),
           "R-07-B1", "five submits collapse into one pending slot");

    State taken {};
    const bool got = auth.takePendingRestore (taken);
    check (got && sameState (taken, makeState (5)),
           "R-07-B2", "the newest submit is taken");

    State again {};
    check (! auth.takePendingRestore (again) && auth.appliedCount() == 1,
           "R-07-B3", "second take at the same boundary returns nothing");
}

// R-07-C: 2 スレッドから大量に投入し、境界で取り出しても取りこぼしが無い
void testConcurrentSubmits()
{
    convo::LearnerStateAuthority<State> auth;
    constexpr int kPerThread = 2000;
    std::atomic<bool> writersDone { false };
    std::atomic<int>  applied { 0 };

    auto writer = [&auth] (int base)
    {
        for (int i = 0; i < kPerThread; ++i)
            (void) auth.enqueueRestoreIfRunning (makeState (base + i + 1), true);
    };

    std::thread w1 (writer, 0);
    std::thread w2 (writer, 100000);

    std::thread boundary ([&]
    {
        while (! writersDone.load (std::memory_order_acquire))
        {
            State s {};
            if (auth.takePendingRestore (s))
                applied.fetch_add (1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
    });

    w1.join();
    w2.join();
    writersDone.store (true, std::memory_order_release);
    boundary.join();

    // 最後に 1 回だけ境界処理
    State last {};
    if (auth.takePendingRestore (last))
        applied.fetch_add (1, std::memory_order_relaxed);

    const auto submitted = auth.submittedCount();
    const auto appliedCount = auth.appliedCount();

    check (submitted == 2 * static_cast<std::uint64_t> (kPerThread),
           "R-07-C1", "all submits reached the authority");
    check (appliedCount >= 1 && appliedCount <= submitted && ! auth.hasPendingRestore(),
           "R-07-C2", "every applied restore was taken exactly once and nothing is left pending");
    check (static_cast<int> (appliedCount) == applied.load(),
           "R-07-C3", "boundary thread and final drain agree on the applied count");
}

// R-07-D: 公開スナップショットは torn read を返さない
void testSnapshotIsConsistent()
{
    convo::LearnerStateAuthority<State> auth;
    std::atomic<bool> stop { false };
    std::atomic<int> reads { 0 };
    std::atomic<int> torn { 0 };

    auto isValid = [] (const State& s)
    {
        // makeState() が作る値の関係（mean[0] = marker、sigma = 0.1 + 0.001 * marker）
        const double marker = s.mean[0];
        if (marker < 0.0 || marker > 100.0)
            return false;

        const bool sigmaOk = std::fabs (s.sigma - (0.1 + 0.001 * marker)) < 1e-12;
        const bool iterOk = s.iteration == static_cast<int> (marker) * 7;
        const bool coeffOk = s.bestCoefficients[0] == -marker;
        return sigmaOk && iterOk && coeffOk;
    };

    std::thread writer ([&]
    {
        for (int v = 0; v < 20000 && ! stop.load (std::memory_order_acquire); ++v)
            auth.publishSnapshot (makeState (v % 100));
    });

    std::thread reader ([&]
    {
        State s {};
        for (int i = 0; i < 200000; ++i)
        {
            if (! auth.readSnapshot (s))
                continue;

            reads.fetch_add (1, std::memory_order_relaxed);
            if (! isValid (s))
                torn.fetch_add (1, std::memory_order_relaxed);
        }
    });

    writer.join();
    stop.store (true, std::memory_order_release);
    reader.join();

    check (reads.load() > 0 && torn.load() == 0,
           "R-07-D", "published snapshot is never observed torn");
}

// R-07-E: withExclusive の中で積んだ要求が取り出せる
void testExclusiveEnqueue()
{
    convo::LearnerStateAuthority<State> auth;
    bool enqueued = false;

    auth.withExclusive ([&]
    {
        auth.enqueueRestoreLocked (makeState (42));
        enqueued = true;
    });

    State taken {};
    const bool got = auth.takePendingRestore (taken);
    check (enqueued && got && sameState (taken, makeState (42)),
           "R-07-E", "enqueue under withExclusive is visible to the boundary");
}

} // namespace

int main()
{
    testStoppedDoesNotQueue();
    testLatestWinsAndSingleTake();
    testConcurrentSubmits();
    testSnapshotIsConsistent();
    testExclusiveEnqueue();

    std::printf ("[SUMMARY] failures=%d\n", gFailures);
    return gFailures == 0 ? 0 : 1;
}
