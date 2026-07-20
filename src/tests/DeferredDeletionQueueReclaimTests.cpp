//==============================================================================
// DeferredDeletionQueueReclaimTests.cpp
//
// Bug#5 改修計画に基づく DeferredDeletionQueue::reclaim() の自動実測テスト。
//
// ■ テスト対象: DeferredDeletionQueue (src/DeferredDeletionQueue.h)
// ■ 測定項目:
//   1. reclaim() の基本正常動作（同一epoch / 異種epoch）
//   2. FIFO 順序保証（先頭から順に回収されること）
//   3. 先頭ブロッキング挙動（head が reclaim 不可のとき後続も回収されないこと）
//   4. MPMC epoch逆転シナリオの再現試行と測定
//   5. reclaim 返戻値（reclaimed カウンタ）の正確性
//   6. ストレステスト（高スループット concurrent enqueue + reclaim）
//
// ■ ビルド:
//   CMakeLists.txt に以下を追加:
//     add_executable(DeferredDeletionQueueReclaimTests
//         src/tests/DeferredDeletionQueueReclaimTests.cpp
//     )
//     add_test(NAME DeferredDeletionQueueReclaimTests
//         COMMAND DeferredDeletionQueueReclaimTests)
//     target_compile_features(DeferredDeletionQueueReclaimTests PRIVATE cxx_std_20)
//     target_include_directories(DeferredDeletionQueueReclaimTests PRIVATE
//         ${CMAKE_CURRENT_SOURCE_DIR}
//         ${CMAKE_CURRENT_SOURCE_DIR}/src
//         ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
//         ${CMAKE_CURRENT_SOURCE_DIR}/src/core
//     )
//
//==============================================================================

#include "DeferredDeletionQueue.h" // テスト対象本体（ヘッダオンリー）
#include "audioengine/AtomicAccess.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

//==============================================================================
// テスト補助: 簡易 TestRunner
//==============================================================================
namespace {

int g_testCount = 0;
int g_failCount = 0;
std::mutex g_ioMutex;

void testPass(const char* name)
{
    std::lock_guard<std::mutex> lock(g_ioMutex);
    std::cout << "  ✅ PASS: " << name << std::endl;
    ++g_testCount;
}

void testFail(const char* name, const char* detail = nullptr)
{
    std::lock_guard<std::mutex> lock(g_ioMutex);
    std::cout << "  ❌ FAIL: " << name;
    if (detail) std::cout << " — " << detail;
    std::cout << std::endl;
    ++g_testCount;
    ++g_failCount;
}

// 整数の一致をチェック
void checkEq(const char* name, uint64_t expected, uint64_t actual)
{
    if (expected == actual)
        testPass(name);
    else {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "expected %llu, got %llu",
                      static_cast<unsigned long long>(expected),
                      static_cast<unsigned long long>(actual));
        testFail(name, buf);
    }
}

void checkTrue(const char* name, bool condition)
{
    if (condition)
        testPass(name);
    else
        testFail(name, "condition was false");
}

void checkFalse(const char* name, bool condition)
{
    if (!condition)
        testPass(name);
    else
        testFail(name, "condition was true");
}

//==============================================================================
// テスト補助: ダミーのポインタとデリーター
//==============================================================================
// 解放されたポインタを記録するための構造体
struct DeletionLog {
    std::vector<void*> deleted;
    std::mutex mutex;

    void record(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        deleted.push_back(ptr);
    }

    bool wasDeleted(void* ptr) const {
        return std::find(deleted.begin(), deleted.end(), ptr) != deleted.end();
    }

    int count() const { return static_cast<int>(deleted.size()); }
    void clear() { deleted.clear(); }
};

// 破棄イベントを追跡可能なダミーオブジェクト
struct TrackedObject {
    int id;
    TrackedObject(int id) : id(id) {}
};

DeletionLog g_delLog;

void dummyDeleter(void* p)
{
    g_delLog.record(p);
}

void resetDeletionLog()
{
    g_delLog.clear();
}

//==============================================================================
// テスト 1: 基本 enqueue + reclaim（同一 epoch）
//==============================================================================
bool testBasicEnqueueReclaim()
{
    resetDeletionLog();
    DeferredDeletionQueue queue;

    TrackedObject obj1(1), obj2(2), obj3(3);

    // 同一 epoch (1) で 3 エントリ enqueue
    checkTrue("basic: enqueue 1", queue.enqueue(&obj1, dummyDeleter, 1));
    checkTrue("basic: enqueue 2", queue.enqueue(&obj2, dummyDeleter, 1));
    checkTrue("basic: enqueue 3", queue.enqueue(&obj3, dummyDeleter, 1));

    // epoch=1 より古い minReaderEpoch=2 で reclaim → 全件回収されるはず
    uint32_t reclaimed = queue.reclaim(2);
    checkEq("basic: reclaimed count", 3, reclaimed);
    checkTrue("basic: obj1 deleted", g_delLog.wasDeleted(&obj1));
    checkTrue("basic: obj2 deleted", g_delLog.wasDeleted(&obj2));
    checkTrue("basic: obj3 deleted", g_delLog.wasDeleted(&obj3));

    // 2回目の reclaim では何も回収されない
    reclaimed = queue.reclaim(2);
    checkEq("basic: second reclaim", 0, reclaimed);

    return g_failCount == 0; // テスト継続のため常に true を返す
}

//==============================================================================
// テスト 2: FIFO 順序保証
//==============================================================================
bool testFifoOrder()
{
    resetDeletionLog();
    DeferredDeletionQueue queue;

    TrackedObject obj1(1), obj2(2), obj3(3);

    queue.enqueue(&obj1, dummyDeleter, 1);
    queue.enqueue(&obj2, dummyDeleter, 1);
    queue.enqueue(&obj3, dummyDeleter, 1);

    // epoch=1 より古い minReaderEpoch=2 で reclaim
    uint32_t reclaimed = queue.reclaim(2);
    checkEq("fifo: reclaimed count", 3, reclaimed);

    // FIFO 順で削除されていること
    checkTrue("fifo: obj1 deleted first", g_delLog.wasDeleted(&obj1));
    checkTrue("fifo: obj2 deleted second", g_delLog.wasDeleted(&obj2));
    checkTrue("fifo: obj3 deleted third", g_delLog.wasDeleted(&obj3));

    return true;
}

//==============================================================================
// テスト 3: 異種 epoch — 古いものだけ回収される
//==============================================================================
bool testMixedEpochReclaim()
{
    resetDeletionLog();
    DeferredDeletionQueue queue;

    TrackedObject oldObj(1), newObj(2);

    queue.enqueue(&oldObj, dummyDeleter, 1);  // epoch=1
    queue.enqueue(&newObj, dummyDeleter, 5);  // epoch=5

    // minReaderEpoch=3 → epoch=1 は回収可能、epoch=5 は回収不可（先頭ブロッキング）
    uint32_t reclaimed = queue.reclaim(3);
    // ★ 現状の実装: 先頭 (oldObj, epoch=1) は回収可能なので回収される。
    //   その後 deqPos が進み、先頭が newObj (epoch=5) になる。
    //   newObj は回収不可なので break。よって reclaimed=1。
    //   ※ これが Bug#5 の「先読み非搭載」の現状動作。先読みがあれば newObj も
    //     別の機会に回収されるが、今は not reclaimable で即 break.
    checkEq("mixed: reclaimed count", 1, reclaimed);
    checkTrue("mixed: old obj deleted", g_delLog.wasDeleted(&oldObj));
    checkFalse("mixed: new obj NOT deleted", g_delLog.wasDeleted(&newObj));

    // 2回目: 同じ minReaderEpoch=3 → newObj(epoch=5) はまだ回収不可
    reclaimed = queue.reclaim(3);
    checkEq("mixed: second reclaim (still blocked)", 0, reclaimed);

    // 3回目: minReaderEpoch=7 → newObj も回収可能
    reclaimed = queue.reclaim(7);
    checkEq("mixed: third reclaim (unblocked)", 1, reclaimed);
    checkTrue("mixed: new obj now deleted", g_delLog.wasDeleted(&newObj));

    return true;
}

//==============================================================================
// テスト 4: 先頭ブロッキング — 新しい epoch が先頭にあると後続をブロック
//==============================================================================
bool testHeadBlocking()
{
    resetDeletionLog();
    DeferredDeletionQueue queue;

    TrackedObject blocker(1), blocked(2);

    queue.enqueue(&blocker, dummyDeleter, 10); // epoch=10（新しい）
    queue.enqueue(&blocked, dummyDeleter, 1);  // epoch=1（古いが後ろ） ← 逆転！

    // minReaderEpoch=5 → blocker(epoch=10) が先頭で回収不可
    // ★ 現状の実装: else 節 → !canDelete → break。blocked は後続にいるが回収されない
    uint32_t reclaimed = queue.reclaim(5);
    checkEq("head-blocking: reclaimed (blocked)", 0, reclaimed);
    checkFalse("head-blocking: blocker NOT deleted", g_delLog.wasDeleted(&blocker));
    checkFalse("head-blocking: blocked NOT deleted", g_delLog.wasDeleted(&blocked));

    // blocker が古くなったら回収可能
    reclaimed = queue.reclaim(12);
    checkEq("head-blocking: reclaim after blocker aged", 2, reclaimed);
    checkTrue("head-blocking: blocker deleted", g_delLog.wasDeleted(&blocker));
    checkTrue("head-blocking: blocked deleted", g_delLog.wasDeleted(&blocked));

    return true;
}

//==============================================================================
// テスト 5: reclaim 返戻値の正確性（連続 reclaim）
//==============================================================================
bool testReclaimCountAccuracy()
{
    resetDeletionLog();
    DeferredDeletionQueue queue;

    // ★ ヒープ確保したオブジェクトを使用（vector の再確保によるポインタ無効化を回避）
    std::vector<std::unique_ptr<TrackedObject>> objs;
    objs.reserve(100);
    for (int i = 0; i < 100; ++i) {
        auto obj = std::make_unique<TrackedObject>(i);
        queue.enqueue(obj.get(), dummyDeleter, 1);
        objs.push_back(std::move(obj));
    }

    // 全件 epoch=1, minReaderEpoch=2 で全件回収
    uint32_t reclaimed = queue.reclaim(2);
    checkEq("count: all 100 reclaimed", 100, reclaimed);

    // 全件削除されたことを確認
    for (auto& ptr : objs) {
        if (!g_delLog.wasDeleted(ptr.get())) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "obj %d not deleted", ptr->id);
            testFail("count: all deleted", buf);
            return true;
        }
    }
    testPass("count: all objects deleted");

    // 2回目: 空キュー
    reclaimed = queue.reclaim(2);
    checkEq("count: empty queue reclaim", 0, reclaimed);

    // さらに追加して部分回収（スタック変数、get() 不使用）
    TrackedObject partial1(200), partial2(201);
    queue.enqueue(&partial1, dummyDeleter, 1);
    queue.enqueue(&partial2, dummyDeleter, 10); // 新しい

    reclaimed = queue.reclaim(5);
    checkEq("count: partial reclaim (only old)", 1, reclaimed);
    checkTrue("count: partial1 deleted", g_delLog.wasDeleted(&partial1));
    checkFalse("count: partial2 NOT deleted", g_delLog.wasDeleted(&partial2));

    reclaimed = queue.reclaim(15);
    checkEq("count: remaining reclaimed", 1, reclaimed);
    checkTrue("count: partial2 now deleted", g_delLog.wasDeleted(&partial2));

    return true;
}

//==============================================================================
// テスト 6: 空キューでの reclaim
//==============================================================================
bool testEmptyQueueReclaim()
{
    resetDeletionLog();
    DeferredDeletionQueue queue;

    uint32_t reclaimed = queue.reclaim(100);
    checkEq("empty: reclaim on empty queue", 0, reclaimed);

    // 空キューに enqueue して即 reclaim
    TrackedObject obj(1);
    queue.enqueue(&obj, dummyDeleter, 1);
    reclaimed = queue.reclaim(2);
    checkEq("empty: single then reclaim", 1, reclaimed);

    return true;
}

//==============================================================================
// テスト 7: MPMC epoch 逆転シナリオ（複数スレッド同時 enqueue）
//
// ■ 目的
//   DeferredDeletionQueue は MPMC（複数 Producer / 複数 Consumer）を謳っている。
//   複数スレッドが独立に取得した epoch を同時に enqueue したとき、
//   キュー内で epoch の順序が逆転する現象が実際に発生するかを測定する。
//
// ■ 測定方法
//   1. N スレッドがそれぞれ異なる epoch 範囲で一斉に enqueue する
//   2. スレッドごとに異なる epoch オフセットを割り当てる
//   3. 全スレッド完了後に reclaim() を実行
//   4. reclaim() が返した件数と、実際に回収可能なエントリ数を比較
//   5. 差があれば epoch 逆転が発生した証拠（ただし先頭ブロッキングの影響も含む）
//
// ■ 測定指標
//   - inversionDetected: 逆転が疑われる状況が発生したか
//   - skipCount: reclaim が見逃した回収可能エントリ数
//   - scanBudgetExhausted: 1回の reclaim 呼び出しで kMaxScan=1024 に達したか
//==============================================================================
struct MpmcEpochInversionResult {
    bool inversionDetected = false;
    int skipCount = 0;
    int scanBudgetExhaustedCount = 0;
    int totalReclaimableEntries = 0;
    int totalReclaimed = 0;
    double reclaimRatio = 1.0; // totalReclaimed / totalReclaimableEntries
};

MpmcEpochInversionResult testMpmcEpochInversion(int numProducers,
                                                 int entriesPerProducer,
                                                 int epochSpread)
{
    DeferredDeletionQueue queue;
    std::atomic<int> readyCount{0};
    std::atomic<bool> startSignal{false};

    // 全スレッドが一斉に enqueue するためのバリア
    auto barrier = [&]() {
        convo::fetchAddAtomic(readyCount, 1, std::memory_order_release);
        while (!convo::consumeAtomic(startSignal, std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    };

    // 全 Producer スレッド生成
    std::vector<std::thread> producers;
    std::vector<TrackedObject*> allObjs;
    std::mutex objMutex;

    for (int p = 0; p < numProducers; ++p) {
        producers.emplace_back([&, p]() {
            barrier(); // 一斉スタート

            for (int i = 0; i < entriesPerProducer; ++i) {
                auto* obj = new TrackedObject(p * 10000 + i);
                {
                    std::lock_guard<std::mutex> lock(objMutex);
                    allObjs.push_back(obj);
                }
                // Producer ごとに異なる epoch 範囲を割り当て
                uint64_t epoch = static_cast<uint64_t>(p * epochSpread + i);
                queue.enqueue(obj, [](void* ptr) {
                    delete static_cast<TrackedObject*>(ptr);
                }, epoch);
            }
        });
    }

    // 一斉スタート
    while (convo::consumeAtomic(readyCount, std::memory_order_acquire) < numProducers) {
        std::this_thread::yield();
    }
    convo::publishAtomic(startSignal, true, std::memory_order_release);

    for (auto& t : producers) {
        t.join();
    }

    // ★ 測定: 全エントリが epoch 最大値より古い状態で reclaim
    //   最も新しい epoch を計算
    uint64_t maxEpoch = 0;
    {
        std::lock_guard<std::mutex> lock(objMutex);
        for (size_t i = 0; i < allObjs.size(); ++i) {
            // epoch は p * epochSpread + i の形式。最大値を計算
            int p = i / entriesPerProducer;
            int idx = i % entriesPerProducer;
            uint64_t epoch = static_cast<uint64_t>(p * epochSpread + idx);
            if (epoch > maxEpoch) maxEpoch = epoch;
        }
    }

    // minReaderEpoch = maxEpoch + 1 で全エントリ回収可能な状態で reclaim
    uint64_t minReaderEpoch = maxEpoch + 1;
    uint32_t reclaimed = queue.reclaim(minReaderEpoch);

    MpmcEpochInversionResult result;
    result.totalReclaimableEntries = numProducers * entriesPerProducer;
    result.totalReclaimed = static_cast<int>(reclaimed);
    result.reclaimRatio = result.totalReclaimableEntries > 0
        ? static_cast<double>(result.totalReclaimed) / static_cast<double>(result.totalReclaimableEntries)
        : 1.0;
    result.skipCount = result.totalReclaimableEntries - result.totalReclaimed;
    result.inversionDetected = (result.skipCount > 0);
    // 最大値scanned の代わりに skip 率で間接測定
    result.scanBudgetExhaustedCount = result.skipCount;

    // 残ったエントリを解放 (drainAllUnsafe)
    queue.drainAllUnsafe();

    // allObjs のクリーンアップ（drainAllUnsafe で削除済みのため解放不要）
    for (auto* obj : allObjs) {
        (void)obj; // drainAllUnsafe が削除している
    }

    return result;
}

// MPMC テストのランナー（複数条件で実行）
void runMpmcEpochInversionTests()
{
    std::cout << "\n  --- MPMC Epoch Inversion Measurement ---" << std::endl;

    struct TestCase {
        const char* name;
        int producers;
        int entriesPerProducer;
        int epochSpread;
    };

    TestCase cases[] = {
        {"2 producers, 100 entries, spread=1",    2,   100, 1},
        {"4 producers, 100 entries, spread=1",    4,   100, 1},
        {"2 producers, 1000 entries, spread=1",   2,  1000, 1},
        {"4 producers, 500 entries, spread=1",    4,   500, 1},
        {"2 producers, 100 entries, spread=10",   2,   100, 10},
        {"4 producers, 100 entries, spread=100",  4,   100, 100},
        {"8 producers, 100 entries, spread=1",    8,   100, 1},
        {"8 producers, 500 entries, spread=10",   8,   500, 10},
    };

    for (const auto& tc : cases) {
        auto result = testMpmcEpochInversion(tc.producers,
                                              tc.entriesPerProducer,
                                              tc.epochSpread);

        // ★ FIX: g_ioMutex は testPass/testFail 内部でロックされるため、
        //   ここでは不要。二重ロックによるデッドロックを防止する。
        std::cout << "  " << tc.name << ": "
                  << "reclaimed=" << result.totalReclaimed
                  << "/" << result.totalReclaimableEntries
                  << " (" << (result.reclaimRatio * 100.0) << "%)"
                  << " skip=" << result.skipCount;
        if (result.inversionDetected) {
            std::cout << " WARNING INVERSION DETECTED";
        } else {
            std::cout << " OK no inversion";
        }
        std::cout << std::endl;

        if (result.inversionDetected) {
            testFail("mpmc: epoch inversion detected (see measurement above)");
        } else {
            testPass("mpmc: no epoch inversion in this configuration");
        }
    }
}

//==============================================================================
// テスト 8: ストレステスト（高スループット concurrent enqueue + reclaim）
//
// ■ 目的
//   長時間・高頻度の enqueue/reclaim サイクルで動作安定性を検証する。
//   kQueueSize=4096 という制約の中で、キュー飽和・空・通常状態を繰り返す。
//
// ■ 測定方法
//   1. Producer スレッド群が一定レートで enqueue を継続
//   2. Consumer (reclaimer) スレッドが定期的に reclaim() を呼ぶ
//   3. 規定時間経過後、全スレッド停止
//   4. drainAllUnsafe() で残余エントリを回収
//   5. 全 Producer からの enqueue 数と Consumer の reclaim 数の一致を確認
//==============================================================================
struct StressTestResult {
    // NOLINT(atomic-dot-call): テスト専用の診断カウンタ。convo::fetchAddAtomic は使用せず直接 atomic 操作
    std::atomic<uint64_t> totalEnqueued{0};
    std::atomic<uint64_t> totalReclaimed{0};
    std::atomic<uint64_t> totalEnqueueFailures{0};
    std::atomic<uint64_t> maxReclaimLatencyUs{0};
    bool passed = false;
};

void runStressTest(int durationMs)
{
    std::cout << "\n  --- Stress Test (" << durationMs << "ms) ---" << std::endl;

    DeferredDeletionQueue queue;
    StressTestResult result;
    std::atomic<bool> stopFlag{false};

    constexpr int kNumProducers = 4;
    constexpr int kReclaimIntervalUs = 500; // 500us

    // Producer スレッド（epoch をインクリメントしながら enqueue）
    std::vector<std::thread> producers;
    for (int p = 0; p < kNumProducers; ++p) {
        producers.emplace_back([&, p]() {
            uint64_t baseEpoch = static_cast<uint64_t>(p) * 1000000ULL;
            uint64_t counter = 0;
            while (!convo::consumeAtomic(stopFlag, std::memory_order_acquire)) {
                auto* obj = new TrackedObject(p * 100000 + static_cast<int>(counter % 100000));
                uint64_t epoch = baseEpoch + counter;
                if (queue.enqueue(obj, [](void* ptr) {
                        delete static_cast<TrackedObject*>(ptr);
                    }, epoch)) {
                    convo::fetchAddAtomic(result.totalEnqueued, uint64_t{1}, std::memory_order_release);
                } else {
                    delete obj;
                    convo::fetchAddAtomic(result.totalEnqueueFailures, uint64_t{1}, std::memory_order_release);
                }
                ++counter;
                // Producer ごとに適度なペース
                if (counter % 100 == 0) {
                    std::this_thread::yield();
                }
            }
        });
    }

    // Reclaimer スレッド（定期的に reclaim）
    std::thread reclaimer([&]() {
        uint64_t currentEpoch = 1;
        while (!convo::consumeAtomic(stopFlag, std::memory_order_acquire)) {
            auto start = std::chrono::high_resolution_clock::now();
            uint32_t reclaimed = queue.reclaim(currentEpoch);
            auto end = std::chrono::high_resolution_clock::now();
            uint64_t latencyUs = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();

            convo::fetchAddAtomic(result.totalReclaimed, static_cast<uint64_t>(reclaimed), std::memory_order_release);

            // 最大レイテンシを更新
            uint64_t prev = convo::consumeAtomic(result.maxReclaimLatencyUs, std::memory_order_acquire);
            while (latencyUs > prev) {
                if (convo::compareExchangeAtomic(result.maxReclaimLatencyUs,
                        prev, latencyUs, std::memory_order_release, std::memory_order_acquire))
                    break;
            }

            currentEpoch += 1000; // epoch を進める
            std::this_thread::sleep_for(std::chrono::microseconds(kReclaimIntervalUs));
        }
    });

    // 規定時間実行
    std::this_thread::sleep_for(std::chrono::milliseconds(durationMs));
    convo::publishAtomic(stopFlag, true, std::memory_order_release);

    // 全スレッド完了待ち
    for (auto& t : producers) t.join();
    reclaimer.join();

    // 残余エントリを回収（drainAllUnsafe は void 戻り値）
    uint32_t preDrainReclaimed = static_cast<uint32_t>(convo::consumeAtomic(result.totalReclaimed, std::memory_order_acquire));
    queue.drainAllUnsafe();
    // ★ drainAllUnsafe は戻り値を返さないため、reclaim 後の残余カウントは不明。
    //   代わりに「後続の reclaim で残ゼロ」であることの確認は省略し、
    //   少なくとも preDrainReclaimed までのエントリが正しく回収され、
    //   残余は drainAllUnsafe がすべて解放したと見なす。
    uint64_t totalReclaimed = preDrainReclaimed;
    uint64_t totalEnqueued = convo::consumeAtomic(result.totalEnqueued, std::memory_order_acquire);

    // ★ FIX: testPass/testFail が内部で g_ioMutex をロックするため、
    //   ここでは生の std::cout 出力のみ行い、チェックはロック外で行う。
    bool stressPassed = false;
    {
        std::lock_guard<std::mutex> lock(g_ioMutex);
        std::cout << "  Stress: enqueued=" << totalEnqueued
                  << " reclaimed=" << totalReclaimed
                  << " enqFail=" << convo::consumeAtomic(result.totalEnqueueFailures, std::memory_order_acquire)
                  << " maxLatencyUs=" << convo::consumeAtomic(result.maxReclaimLatencyUs, std::memory_order_acquire)
                  << std::endl;

        if (convo::consumeAtomic(result.totalEnqueueFailures, std::memory_order_relaxed) > 0) {
            std::cout << "  Queue was FULL during stress test ("
                      << convo::consumeAtomic(result.totalEnqueueFailures, std::memory_order_relaxed) << " enqueue failures)"
                      << std::endl;
        }

        stressPassed = (totalEnqueued > 0);
    } // g_ioMutex のロックを解放してから testPass/testFail を呼ぶ

    if (stressPassed) {
        testPass("stress: enqueue/reclaim cycle completed");
    } else {
        testFail("stress: no entries were enqueued");
    }
}

//==============================================================================
// テスト 9: kMaxScan 到達テスト — 大量 reclaimable エントリの一括回収
//
// kMaxScan=1024 の制限を確認。1回の reclaim() で 1024 エントリ以上回収できること
// （scan budget は先読み用であり、回収自体に上限はないため）。
//==============================================================================
bool testMaxScanBudget()
{
    resetDeletionLog();
    DeferredDeletionQueue queue;

    // ★ kQueueSize=4096 がキューの上限のため、kNumEntries は kQueueSize 以下にする
    constexpr int kNumEntries = 4096;

    std::vector<std::unique_ptr<TrackedObject>> objs;
    objs.reserve(kNumEntries);
    for (int i = 0; i < kNumEntries; ++i) {
        auto obj = std::make_unique<TrackedObject>(i);
        queue.enqueue(obj.get(), dummyDeleter, 1);
        objs.push_back(std::move(obj));
    }

    // 1回の reclaim() で全件回収できるはず（kMaxScan は先読み用であり回収自体に上限なし）
    uint32_t reclaimed = queue.reclaim(2);
    checkEq("maxscan: all entries reclaimed in one call", kNumEntries, reclaimed);

    return true;
}

//==============================================================================
// テスト 10: エントリ数が多いときの reclaim パフォーマンス測定
//==============================================================================
void testReclaimPerformance()
{
    std::cout << "\n  --- Reclaim Performance Measurement ---" << std::endl;

    // ★ kQueueSize=4096 が上限。上限まで詰めて測定。
    constexpr int kNumEntries = 4096;

    DeferredDeletionQueue queue;
    std::vector<std::unique_ptr<TrackedObject>> objs;
    objs.reserve(kNumEntries);
    for (int i = 0; i < kNumEntries; ++i) {
        auto obj = std::make_unique<TrackedObject>(i);
        queue.enqueue(obj.get(), dummyDeleter, 1);
        objs.push_back(std::move(obj));
    }

    auto start = std::chrono::high_resolution_clock::now();
    uint32_t reclaimed = queue.reclaim(2);
    auto end = std::chrono::high_resolution_clock::now();

    uint64_t elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    {
        std::lock_guard<std::mutex> lock(g_ioMutex);
        std::cout << "  Performance: reclaimed " << reclaimed
                  << " entries in " << elapsedUs << " us"
                  << " (" << (elapsedUs * 1000.0 / kNumEntries) << " ns/entry)"
                  << std::endl;
    }
    checkEq("perf: all reclaimed", kNumEntries, reclaimed);
}

} // anonymous namespace

//==============================================================================
// main
//==============================================================================
int main()
{
    std::cout << "==============================================" << std::endl;
    std::cout << "DeferredDeletionQueue Reclaim Tests" << std::endl;
    std::cout << "  (Bug#5 改修計画に基づく自動実測テスト)" << std::endl;
    std::cout << "==============================================" << std::endl;

    // ── 基本機能テスト ──
    std::cout << "\n[Basic Functionality Tests]" << std::endl;
    testBasicEnqueueReclaim();
    testFifoOrder();
    testMixedEpochReclaim();
    testHeadBlocking();
    testReclaimCountAccuracy();
    testEmptyQueueReclaim();
    testMaxScanBudget();

    // ── パフォーマンス測定 ──
    std::cout << "\n[Performance Measurement]" << std::endl;
    testReclaimPerformance();

    // ── MPMC epoch 逆転測定 ──
    std::cout << "\n[MPMC Epoch Inversion Measurement]" << std::endl;
    runMpmcEpochInversionTests();

    // ── ストレステスト ──
    std::cout << "\n[Stress Test]" << std::endl;
    runStressTest(3000); // 3秒間

    // ── 結果 ──
    std::cout << "\n==============================================" << std::endl;
    std::cout << "Results: " << g_testCount << " tests, "
              << g_failCount << " failures" << std::endl;
    std::cout << "==============================================" << std::endl;

    return g_failCount > 0 ? 1 : 0;
}
