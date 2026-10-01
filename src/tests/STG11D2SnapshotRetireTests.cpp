// STG11D2SnapshotRetireTests.cpp — STG-11-D2 / R-A regression (TD2-1 .. TD2-3).
//
// 対象 defect（STG-11-D2）:
//   SnapshotCoordinator::quarantineRetireSink() が Q（RetireQuarantineStore）へ
//   直接移送するだけで、E / T へ昇格しなかった。Q 満杯時は Release で no-op となり
//   GlobalSnapshot* が終端解決なしに失われた。
//   R-A: Q-full 時に E → T へ昇格し、retire ownership を終端まで保持する。
//
// Test contract:
//   TD2-1: D + Q 満杯後の retire が E に格納される（E == 3 正確、合計一致、drop 0）
//   TD2-2: D + Q + E 満杯後の retire が T に格納される（T == 3 正確、合計一致）
//   TD2-3: drain / shutdown 後に全 counts 0・alive 0・drop 相当 0
//   TD2-4: 既存非回帰（D8_2 T2 / StuckReader / harness / TD1。改変 0、CTest で実施）
//
// 方針:
//   ・単一スレッド・決定論的。race 再現は不要。
//   ・容量は固定値（D=4096 / Q=512 / E=512）に依存するが、これは production の
//     公開 constexpr であり、test 側のマジックナンバーではない。
//   ・D / Q / E の pre-fill は counted local object で行い、coordinator 駆動分は
//     実 GlobalSnapshot（SnapshotFactory::create）で本経路を通す。
//     pre-fill object と coordinator object を混ぜない。
//   ・private epoch は test 内で明示的に進める（reader なし → minReader 追従）。
//     fill 中は全 entry が同一 epoch で reclaim 不可 — 決定論的 fill が成立する。
//   ・fixture は heap 確保する（RuntimeIntentCoordinator は使わないが、
//     EpochDomain 約 213KB のため。TD1 の教訓）。
//   ・既存 oracle の改変 = 0。production ロジックの変更は R-A のみ。
//
// 注意:
//   - standalone CTest target（STG11D2SnapshotRetireTests）としても登録する
//     （AudioEngineHarness の深いコールチェイン外で実行するため）。
//     harness 側にも runSTG11D2SnapshotRetireTests() として配線する。
// =============================================================================

#pragma warning(push)
#pragma warning(disable : 4996) // enterReader/exitReader は deprecated だが
                                // reader-pinned fill に直接使用する（production 経路と同一 API）
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>

#include "core/EpochDomain.h"
#include "core/SnapshotCoordinator.h"
#include "core/SnapshotFactory.h"
#include "audioengine/ISRRetireRouter.h"
#pragma warning(pop)

// 公開 constexpr 容量（production の契約値）。
// DeferredDeletionQueue.h:262 / RetireQuarantineStore.h:65 と対応。
static constexpr std::uint32_t kDQueueSize = 4096;
static constexpr std::size_t kQStoreSize = 512;
static constexpr std::size_t kEStoreSize = 512;

namespace {

// pre-fill 用の counted object（coordinator 駆動の GlobalSnapshot とは別管理）。
struct TD21Tracker
{
    std::atomic<int> invokeCount { 0 };
    std::atomic<int> aliveCount { 0 };
};

struct TD21Object
{
    TD21Tracker* tracker;
    explicit TD21Object(TD21Tracker* t) : tracker(t) { ++tracker->aliveCount; }
    ~TD21Object() { --tracker->aliveCount; }
};

void td21Deleter(void* p) noexcept
{
    auto* obj = static_cast<TD21Object*>(p);
    ++obj->tracker->invokeCount;
    delete obj;
}

// D を容量まで pre-fill する（counted local objects）。
bool prefillD(convo::isr::ISRRetireRouter& router, TD21Tracker& tracker,
              std::uint64_t epoch, const char* id)
{
    for (std::uint32_t i = 0; i < kDQueueSize; ++i)
    {
        auto* obj = new TD21Object(&tracker);
        if (!router.enqueueRetire(obj, &td21Deleter, epoch))
        {
            std::fprintf(stderr, "%s: D pre-fill failed at %u\n", id, i);
            delete obj;
            return false;
        }
    }
    return true;
}

// Q を容量まで pre-fill する（counted local objects）。
bool prefillQ(convo::isr::ISRRetireRouter& router, TD21Tracker& tracker,
              std::uint64_t epoch, const char* id)
{
    for (std::size_t i = 0; i < kQStoreSize; ++i)
    {
        auto* obj = new TD21Object(&tracker);
        if (!router.quarantineRetire(obj, &td21Deleter, epoch,
                                     ::DeletionEntryType::Generic, "td2-prefix"))
        {
            std::fprintf(stderr, "%s: Q pre-fill failed at %llu\n", id,
                         static_cast<unsigned long long>(i));
            delete obj;
            return false;
        }
    }
    return true;
}

// E を容量まで pre-fill する（counted local objects）。
bool prefillE(convo::isr::ISRRetireRouter& router, TD21Tracker& tracker,
              std::uint64_t epoch, const char* id)
{
    for (std::size_t i = 0; i < kEStoreSize; ++i)
    {
        auto* obj = new TD21Object(&tracker);
        if (!router.emergencyQuarantine(obj, &td21Deleter, epoch,
                                        ::DeletionEntryType::Generic, "td2-prefix", 0, 0))
        {
            std::fprintf(stderr, "%s: E pre-fill failed at %llu\n", id,
                         static_cast<unsigned long long>(i));
            delete obj;
            return false;
        }
    }
    return true;
}

// 実 GlobalSnapshot を coordinator 経由で retire する（本経路）。
// 1 回の switchImmediate は current 1 個を retire する（priming 後）。
bool driveCoordinatorRetire(convo::SnapshotCoordinator& coord, const char* id)
{
    auto* snap = convo::SnapshotFactory::create(convo::SnapshotParams{});
    if (snap == nullptr)
    {
        std::fprintf(stderr, "%s: SnapshotFactory::create returned null\n", id);
        return false;
    }
    coord.switchImmediate(snap);
    return true;
}

bool checkAllDrained(convo::EpochDomain& dom, convo::isr::ISRRetireRouter& router,
                     TD21Tracker& tracker, const char* id)
{
    if (router.pendingRetireCount() != 0 || router.quarantineResidentCount() != 0
        || router.emergencyQuarantineResidentCount() != 0
        || router.terminalReclaimResidentCount() != 0)
    {
        std::fprintf(stderr,
                     "%s: not drained (pending=%u Q+E=%llu E=%llu T=%llu)\n",
                     id, router.pendingRetireCount(),
                     static_cast<unsigned long long>(router.quarantineResidentCount()),
                     static_cast<unsigned long long>(router.emergencyQuarantineResidentCount()),
                     static_cast<unsigned long long>(router.terminalReclaimResidentCount()));
        return false;
    }
    if (tracker.aliveCount != 0 || tracker.invokeCount == 0)
    {
        // invokeCount == 0 は「何も解放されなかった」異常も検出する。
        std::fprintf(stderr, "%s: lifetime mismatch (alive=%d invoked=%d)\n",
                     id, tracker.aliveCount.load(), tracker.invokeCount.load());
        return false;
    }
    return true;
}

} // namespace

// =============================================================================
// TD2-1: D + Q 満杯後の retire が E に格納される
//   D(4096) + Q(512) を pre-fill し、coordinator 駆動 3 回で E == 3（正確）。
//   旧 sink（Q のみ）では 3 個が失われた（Release）／assert 発火（Debug）。
// =============================================================================
static bool checkTD21QFullEscalatesToEmergency()
{
    auto dom = std::make_unique<convo::EpochDomain>();
    auto router = std::make_unique<convo::isr::ISRRetireRouter>(*dom);
    auto coord = std::make_unique<convo::SnapshotCoordinator>(*dom);
    coord->setRetireSink(router.get());
    TD21Tracker tracker;

    const std::uint64_t epoch = dom->currentEpoch();
    if (!prefillD(*router, tracker, epoch, "TD2-1"))
        return false;
    if (!prefillQ(*router, tracker, epoch, "TD2-1"))
        return false;
    // ★ reader を滞留させて minReader を固定する。
    //   coordinator retire は内部で publishEpoch() するため、reader なしでは
    //   pre-fill が tryReclaim で全解放されてしまう（決定論的 fill が崩れる）。
    //   reader epoch での entry は isOlder が偽となり reclaim 不可。
    const int readerIdx = dom->registerReaderThread("td2-1");
    if (readerIdx < 0)
    {
        std::fprintf(stderr, "TD2-1: reader registration failed\n");
        return false;
    }
    dom->enterReader(readerIdx);
    if (tracker.aliveCount != static_cast<int>(kDQueueSize + kQStoreSize))
    {
        std::fprintf(stderr, "TD2-1: pre-fill accounting mismatch (alive=%d)\n",
                     tracker.aliveCount.load());
        return false;
    }

    // prime: current slot に snap1 を据える（retire なし）。
    if (!driveCoordinatorRetire(*coord, "TD2-1"))
        return false;
    // 測定 3 回: 各回 old current 1 個が D-full → Q-full → E へ。
    for (int i = 0; i < 3; ++i)
    {
        if (!driveCoordinatorRetire(*coord, "TD2-1"))
            return false;
    }
    if (router->emergencyQuarantineResidentCount() != 3)
    {
        std::fprintf(stderr, "TD2-1: E residency=%llu expected 3 (D=%u Q+E=%llu T=%llu)\n",
                     static_cast<unsigned long long>(router->emergencyQuarantineResidentCount()),
                     router->pendingRetireCount(),
                     static_cast<unsigned long long>(router->quarantineResidentCount()),
                     static_cast<unsigned long long>(router->terminalReclaimResidentCount()));
        return false;
    }
    if (router->terminalReclaimResidentCount() != 0)
    {
        std::fprintf(stderr, "TD2-1: unexpected T residency\n");
        return false;
    }

    // reader 退出＋epoch 前進で全 entry を reclaim 可能にする。
    dom->exitReader(readerIdx);
    (void) dom->publishEpoch();
    router->tryReclaim();
    router->drainAllQuarantineStore();
    (void) dom->publishEpoch();
    router->tryReclaim();
    if (!checkAllDrained(*dom, *router, tracker, "TD2-1"))
        return false;

    std::printf("STG11D2SnapshotRetireTests: TD2-1 PASS (Q-full escalates to E, E=3 drained)\n");
    return true;
}

// =============================================================================
// TD2-2: D + Q + E 満杯後の retire が T に格納される
//   D(4096) + Q(512) + E(512) を pre-fill し、coordinator 駆動 3 回で T == 3（正確）。
// =============================================================================
static bool checkTD22EFullEscalatesToTerminal()
{
    auto dom = std::make_unique<convo::EpochDomain>();
    auto router = std::make_unique<convo::isr::ISRRetireRouter>(*dom);
    auto coord = std::make_unique<convo::SnapshotCoordinator>(*dom);
    coord->setRetireSink(router.get());
    TD21Tracker tracker;

    const std::uint64_t epoch = dom->currentEpoch();
    if (!prefillD(*router, tracker, epoch, "TD2-2"))
        return false;
    if (!prefillQ(*router, tracker, epoch, "TD2-2"))
        return false;
    if (!prefillE(*router, tracker, epoch, "TD2-2"))
        return false;
    // ★ TD2-1 と同様に reader を滞留させて minReader を固定する。
    const int readerIdx = dom->registerReaderThread("td2-2");
    if (readerIdx < 0)
    {
        std::fprintf(stderr, "TD2-2: reader registration failed\n");
        return false;
    }
    dom->enterReader(readerIdx);

    if (!driveCoordinatorRetire(*coord, "TD2-2"))
        return false;
    for (int i = 0; i < 3; ++i)
    {
        if (!driveCoordinatorRetire(*coord, "TD2-2"))
            return false;
    }
    if (router->terminalReclaimResidentCount() != 3)
    {
        std::fprintf(stderr, "TD2-2: T residency=%llu expected 3 (D=%u Q+E=%llu E=%llu)\n",
                     static_cast<unsigned long long>(router->terminalReclaimResidentCount()),
                     router->pendingRetireCount(),
                     static_cast<unsigned long long>(router->quarantineResidentCount()),
                     static_cast<unsigned long long>(router->emergencyQuarantineResidentCount()));
        return false;
    }

    dom->exitReader(readerIdx);
    (void) dom->publishEpoch();
    router->tryReclaim();
    router->drainAllQuarantineStore();
    (void) dom->publishEpoch();
    router->tryReclaim();
    if (!checkAllDrained(*dom, *router, tracker, "TD2-2"))
        return false;

    std::printf("STG11D2SnapshotRetireTests: TD2-2 PASS (E-full escalates to T, T=3 drained)\n");
    return true;
}

// =============================================================================
// TD2-3: drain / shutdown 後に全 counts 0・alive 0
//   finalizeShutdown(false) 経路（production の shutdown と同一入口）を通し、
//   engine 既存 drain（drainAllQuarantineStore＋drainAll）で完全解決する。
// =============================================================================
static bool checkTD23ShutdownDrainCompletes()
{
    auto dom = std::make_unique<convo::EpochDomain>();
    auto router = std::make_unique<convo::isr::ISRRetireRouter>(*dom);
    auto coord = std::make_unique<convo::SnapshotCoordinator>(*dom);
    coord->setRetireSink(router.get());
    TD21Tracker tracker;

    const std::uint64_t epoch = dom->currentEpoch();
    if (!prefillD(*router, tracker, epoch, "TD2-3"))
        return false;
    if (!prefillQ(*router, tracker, epoch, "TD2-3"))
        return false;
    // ★ TD2-1 と同様に reader を滞留させて minReader を固定する。
    const int readerIdx = dom->registerReaderThread("td2-3");
    if (readerIdx < 0)
    {
        std::fprintf(stderr, "TD2-3: reader registration failed\n");
        return false;
    }
    dom->enterReader(readerIdx);
    if (!driveCoordinatorRetire(*coord, "TD2-3"))
        return false;
    if (!driveCoordinatorRetire(*coord, "TD2-3"))
        return false;

    // shutdown 入口（production: ReleaseResources.cpp:664 と同一）。
    coord->finalizeShutdown(false);
    // reader 退出＋epoch 前進で全 entry を reclaim 可能にする。
    dom->exitReader(readerIdx);
    (void) dom->publishEpoch();
    router->tryReclaim();
    router->drainAllQuarantineStore();
    dom->drainAll();
    (void) dom->publishEpoch();
    router->tryReclaim();
    if (!checkAllDrained(*dom, *router, tracker, "TD2-3"))
        return false;
    // finalizeShutdown 済みのため dtor は no-op（scope exit で実行される）。

    std::printf("STG11D2SnapshotRetireTests: TD2-3 PASS (shutdown drain completes)\n");
    return true;
}

// harness main（PublishPipelineIntegrationTests.cpp）から呼ばれるエントリ。
int runSTG11D2SnapshotRetireTests()
{
    bool ok = true;
    if (!checkTD21QFullEscalatesToEmergency())
    {
        std::fprintf(stderr, "FAIL: TD2-1 Q-full escalation\n");
        ok = false;
    }
    if (!checkTD22EFullEscalatesToTerminal())
    {
        std::fprintf(stderr, "FAIL: TD2-2 E-full escalation\n");
        ok = false;
    }
    if (!checkTD23ShutdownDrainCompletes())
    {
        std::fprintf(stderr, "FAIL: TD2-3 shutdown drain\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11D2SnapshotRetireTests: PASS (TD2-1/TD2-2/TD2-3)\n");
    return ok ? 0 : 1;
}

#ifdef STG11D2_STANDALONE_MAIN
// ★ standalone CTest target 用 main（STG10ReaderQuarantineTests と同じ形）。
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    return runSTG11D2SnapshotRetireTests();
}
#endif
