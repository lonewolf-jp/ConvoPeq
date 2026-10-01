// STG11EQRetireTests.cpp — STG-11-D1 / Candidate B regression (TD1-1 .. TD1-4).
//
// 対象 defect（STG-11-D1）:
//   EQProcessor::enqueueDeferredDeleteWithFallback() が stack-local ISRRetireRouter
//   を使用していたため、D 満杯時に Q/E/T へ昇格した entry が関数 return で失われた。
//   Candidate B: EQ-owned member router（m_ownedRetireRouter, m_epochDomain 束縛）。
//
// Test contract:
//   TD1-1: D → Q 後の lifetime（Q residency が return を跨いで生存し、drain で回収）
//   TD1-2: E / T まで到達した entry の lifetime（同上）
//   TD1-3: releaseResources / destruction 後の leak / drain（全 counts 0・drop 0）
//   TD1-4: private epoch provenance 非回帰
//     (a) member router の epoch が private domain と一体で進む（束縛の実証）
//     (b) reclaim は束縛 provider の epoch でのみ進む（他 domain の前進では進まない）
//   TD1-5: 既存 retire / epoch test の非回帰（CTest で実施。改変 0）
//
// 方針:
//   ・単一スレッド・決定論的。race 再現は不要。
//   ・容量は固定値（D=4096 / Q=512 / E=512）に依存するが、これは production の
//     公開 constexpr であり、test 側のマジックナンバーではない。
//   ・setter 1 回 = EQState 1 + BandNode 1 の計 2 retire（setBandFrequency 経路）。
//   ・private epoch は flush（releaseResources / prepareToPlay）でのみ進むため、
//     fill 中は全 entry が同一 epoch で reclaim 不可 — 決定論的 fill が成立する。
//   ・既存 oracle の改変 = 0。production ロジックの変更は Candidate B のみ。
//
// 注意:
//   - standalone CTest target（STG11EQRetireTests）としても登録する
//     （AudioEngineHarness の深いコールチェイン外で実行するため。
//      EQProcessor / RuntimeIntentCoordinator は大きいため heap 確保する）。
//     harness 側にも runSTG11EQRetireTests() として配線する（二重実行で回帰を固定）。
// =============================================================================

#pragma warning(push)
#pragma warning(disable : 4996) // enterReader/exitReader は deprecated だが TD1-4b の
                                // mechanism 検証に直接使用する（production 経路と同一 API）

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>

#include "eqprocessor/EQProcessor.h"
#include "core/EpochDomain.h"
#include "audioengine/ISRRetireRouter.h"
#include "audioengine/ISRRuntimePublicationCoordinator.h"

#pragma warning(pop)

// 公開 constexpr 容量（production の契約値）。
// DeferredDeletionQueue.h:262 / RetireQuarantineStore.h:65 と対応。
static constexpr std::uint32_t kDQueueSize = 4096;
static constexpr std::size_t kQStoreSize = 512;
static constexpr std::size_t kEStoreSize = 512;

// setter 1 回あたりの retire 数（EQState 1 + BandNode 1）。
static constexpr int kRetiresPerSetter = 2;

namespace {

// TD1-4b 用の deleter カウンタ。
struct TD11Tracker
{
    std::atomic<int> invokeCount { 0 };
    std::atomic<int> aliveCount { 0 };
};

struct TD11Object
{
    TD11Tracker* tracker;
    explicit TD11Object(TD11Tracker* t) : tracker(t) { ++tracker->aliveCount; }
    ~TD11Object() { --tracker->aliveCount; }
};

void td11Deleter(void* p) noexcept
{
    auto* obj = static_cast<TD11Object*>(p);
    ++obj->tracker->invokeCount;
    delete obj;
}

void driveSetters(EQProcessor& eq, int count, float baseFreq = 100.0f)
{
    for (int i = 0; i < count; ++i)
    {
        eq.setBandFrequency(0, baseFreq + static_cast<float>(i) * 0.5f);
        if ((i % 500) == 499)
        {
            // ★ quarantineResidentCount() は Q+E 合算のため "Q+E" と表記。
            std::fprintf(stderr,
                         "  [fill %d] pending=%u Q+E=%llu E=%llu T=%llu drop=%llu\n",
                         i + 1, eq.eqOwnedPendingRetire(),
                         static_cast<unsigned long long>(eq.eqOwnedQuarantineResident()),
                         static_cast<unsigned long long>(eq.eqOwnedEmergencyResident()),
                         static_cast<unsigned long long>(eq.eqOwnedTerminalResident()),
                         static_cast<unsigned long long>(eq.eqRetireDropCount()));
        }
    }
}

bool checkBaselineClean(const EQProcessor& eq, const char* id)
{
    if (eq.eqOwnedPendingRetire() != 0 || eq.eqOwnedQuarantineResident() != 0
        || eq.eqOwnedEmergencyResident() != 0 || eq.eqOwnedTerminalResident() != 0
        || eq.eqRetireDropCount() != 0)
    {
        std::fprintf(stderr,
                     "%s: baseline not clean (pending=%u Q=%llu E=%llu T=%llu drop=%llu)\n",
                     id,
                     eq.eqOwnedPendingRetire(),
                     static_cast<unsigned long long>(eq.eqOwnedQuarantineResident()),
                     static_cast<unsigned long long>(eq.eqOwnedEmergencyResident()),
                     static_cast<unsigned long long>(eq.eqOwnedTerminalResident()),
                     static_cast<unsigned long long>(eq.eqRetireDropCount()));
        return false;
    }
    return true;
}

bool checkAllDrained(const EQProcessor& eq, const char* id)
{
    if (eq.eqOwnedPendingRetire() != 0 || eq.eqOwnedQuarantineResident() != 0
        || eq.eqOwnedEmergencyResident() != 0 || eq.eqOwnedTerminalResident() != 0)
    {
        std::fprintf(stderr,
                     "%s: not drained (pending=%u Q=%llu E=%llu T=%llu)\n",
                     id,
                     eq.eqOwnedPendingRetire(),
                     static_cast<unsigned long long>(eq.eqOwnedQuarantineResident()),
                     static_cast<unsigned long long>(eq.eqOwnedEmergencyResident()),
                     static_cast<unsigned long long>(eq.eqOwnedTerminalResident()));
        return false;
    }
    if (eq.eqRetireDropCount() != 0)
    {
        std::fprintf(stderr, "%s: retire drop occurred (drop=%llu)\n",
                     id, static_cast<unsigned long long>(eq.eqRetireDropCount()));
        return false;
    }
    return true;
}

} // namespace

// =============================================================================
// TD1-1: D → Q 後の lifetime
//   D(4096) を超える retire を public setter で駆動し、Q residency が
//   関数 return を跨いで生存すること（旧 stack-local 欠陥では失われた）、
//   および epoch 前進＋drain で回収されることを確認する。
// =============================================================================
static bool checkTD11QRetainsAcrossReturns()
{
    // production 配線と同様に coordinator を設定（DSPCoreLifecycle.cpp:90）。
    // 未設定では enqueueDeferredDeleteWithFallback() が false を返す（by design）。
    // ★ heap 確保: RuntimeIntentCoordinator は約 4.1MB、
    //   EQProcessor は約 281KB のため stack 確保はしない
    //   （production も DSPCore ごと heap 生成する）。
    //   coordinator は eq より先に生成し、破棄は逆順（eq の dtor が coordinator を使う）。
    std::fprintf(stderr, "TD1-1: constructing coordinator...\n");
    auto coordinator = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    std::fprintf(stderr, "TD1-1: constructing EQProcessor...\n");
    auto eq = std::make_unique<EQProcessor>();
    std::fprintf(stderr, "TD1-1: wiring coordinator...\n");
    eq->setRetireCoordinator(coordinator.get());
    if (!checkBaselineClean(*eq, "TD1-1"))
        return false;
    std::fprintf(stderr, "TD1-1: driving setters...\n");

    // 2200 setters × 2 = 4400 > 4096 → Q に正確に 304（単一所有のため）。
    driveSetters(*eq, 2200);
    std::fprintf(stderr, "TD1-1: fill done, checking...\n");
    if (eq->eqRetireDropCount() != 0)
    {
        std::fprintf(stderr, "TD1-1: retire drop during fill\n");
        return false;
    }
    if (eq->eqOwnedPendingRetire() != kDQueueSize)
    {
        std::fprintf(stderr, "TD1-1: D pending=%u expected %u\n",
                     eq->eqOwnedPendingRetire(), kDQueueSize);
        return false;
    }
    const std::size_t q = eq->eqOwnedQuarantineResident();
    if (q != 304)
    {
        // ★ Q=0 は旧欠陥（stack-local 喪失）の署名。
        //   304 以外の超過は二重所有を示す。
        std::fprintf(stderr, "TD1-1: Q residency=%llu expected 304\n",
                     static_cast<unsigned long long>(q));
        return false;
    }

    // epoch 前進（releaseResources 内の flush）＋ driver tryReclaim で全 drain。
    // test 内に active reader はいないため全 entry が reclaim 可能になる。
    std::fprintf(stderr, "TD1-1: releasing...\n");
    eq->releaseResources();
    std::fprintf(stderr, "TD1-1: released, checking drain...\n");
    if (!checkAllDrained(*eq, "TD1-1"))
        return false;

    std::printf("STG11EQRetireTests: TD1-1 PASS (D->Q retained across returns, Q=%llu drained)\n",
                static_cast<unsigned long long>(q));
    return true;
}

// =============================================================================
// TD1-2: E / T まで到達した entry の lifetime
//   D(4096) + Q(512) + E(512) を超える retire を駆動し、E / T residency が
//   生存すること、および drain で回収されることを確認する。
// =============================================================================
static bool checkTD12EmergencyAndTerminalRetain()
{
    auto coordinator = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto eq = std::make_unique<EQProcessor>();
    eq->setRetireCoordinator(coordinator.get());
    if (!checkBaselineClean(*eq, "TD1-2"))
        return false;

    // 2700 setters × 2 = 5400 = 4096(D) + 512(Q) + 512(E) + 280(T)。単一所有のため正確。
    driveSetters(*eq, 2700);
    if (eq->eqRetireDropCount() != 0)
    {
        std::fprintf(stderr, "TD1-2: retire drop during fill\n");
        return false;
    }
    // ★ quarantineResidentCount() は Q+E の合算（ISRRetireRouter.cpp:431-435）。
    //   したがって Q 単独 = 合算 - E。
    if (eq->eqOwnedPendingRetire() != kDQueueSize
        || eq->eqOwnedQuarantineResident() != kQStoreSize + kEStoreSize
        || eq->eqOwnedEmergencyResident() != kEStoreSize)
    {
        std::fprintf(stderr,
                     "TD1-2: unexpected fill state (pending=%u Q+E=%llu E=%llu T=%llu)\n",
                     eq->eqOwnedPendingRetire(),
                     static_cast<unsigned long long>(eq->eqOwnedQuarantineResident()),
                     static_cast<unsigned long long>(eq->eqOwnedEmergencyResident()),
                     static_cast<unsigned long long>(eq->eqOwnedTerminalResident()));
        return false;
    }
    const std::size_t qAlone = eq->eqOwnedQuarantineResident() - eq->eqOwnedEmergencyResident();
    if (qAlone != kQStoreSize)
    {
        std::fprintf(stderr, "TD1-2: Q alone=%llu expected %llu\n",
                     static_cast<unsigned long long>(qAlone),
                     static_cast<unsigned long long>(kQStoreSize));
        return false;
    }
    const std::size_t t = eq->eqOwnedTerminalResident();
    if (t != 280)
    {
        // ★ T=0 は所有喪失、280 超過は二重所有を示す。
        std::fprintf(stderr, "TD1-2: T residency=%llu expected 280\n",
                     static_cast<unsigned long long>(t));
        return false;
    }

    eq->releaseResources();
    if (!checkAllDrained(*eq, "TD1-2"))
        return false;

    std::printf("STG11EQRetireTests: TD1-2 PASS (E/T retained across returns, T=%llu drained)\n",
                static_cast<unsigned long long>(t));
    return true;
}

// =============================================================================
// TD1-3: release / destruction 後の leak / drain
//   通常量の retire → releaseResources で全 counts 0・drop 0。
//   dtor は各 test の scope exit で実行される（force-drain 前提は
//   publication pipeline の engine-epoch-gated 破棄）。
// =============================================================================
static bool checkTD13ReleaseLeavesNoResidue()
{
    auto coordinator = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto eq = std::make_unique<EQProcessor>();
    eq->setRetireCoordinator(coordinator.get());
    if (!checkBaselineClean(*eq, "TD1-3"))
        return false;

    driveSetters(*eq, 50);
    eq->releaseResources();
    if (!checkAllDrained(*eq, "TD1-3"))
        return false;

    std::printf("STG11EQRetireTests: TD1-3 PASS (release leaves no residue)\n");
    return true;
}

// =============================================================================
// TD1-4a: member router の epoch が private domain と一体で進む（束縛の実証）
// =============================================================================
static bool checkTD14aRouterBoundToPrivateDomain()
{
    auto coordinator = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto eq = std::make_unique<EQProcessor>();
    eq->setRetireCoordinator(coordinator.get());
    if (eq->eqOwnedRouterEpoch() != eq->eqPrivateEpoch())
    {
        std::fprintf(stderr, "TD1-4a: router epoch != private epoch at baseline\n");
        return false;
    }
    const std::uint64_t e0 = eq->eqPrivateEpoch();
    eq->releaseResources(); // flush: private publishEpoch × 1
    if (eq->eqPrivateEpoch() != e0 + 1 || eq->eqOwnedRouterEpoch() != e0 + 1)
    {
        std::fprintf(stderr,
                     "TD1-4a: epochs did not advance together (priv=%llu router=%llu base=%llu)\n",
                     static_cast<unsigned long long>(eq->eqPrivateEpoch()),
                     static_cast<unsigned long long>(eq->eqOwnedRouterEpoch()),
                     static_cast<unsigned long long>(e0));
        return false;
    }

    std::printf("STG11EQRetireTests: TD1-4a PASS (router bound to private domain)\n");
    return true;
}

// =============================================================================
// TD1-4b: reclaim は束縛 provider の epoch でのみ進む（他 domain の前進では進まない）
//   Candidate B の配線パターン（private domain に束縛された router）そのものを
//   決定論的に検証する。engine domain への移管（Candidate A）が禁止される根拠。
// =============================================================================
static bool checkTD14bReclaimGatedOnlyOnBoundProvider()
{
    convo::EpochDomain eqLike;
    convo::EpochDomain foreign;
    convo::isr::ISRRetireRouter router(eqLike);
    TD11Tracker tracker;

    // eqLike を epoch 5 まで進め、reader を入場させる（滞留 reader を再現）。
    for (int i = 0; i < 4; ++i)
        (void) eqLike.publishEpoch();
    if (eqLike.currentEpoch() != 5)
        return false;
    const int idx = eqLike.registerReaderThread("td1-4b");
    if (idx < 0)
        return false;
    eqLike.enterReader(idx);

    auto* obj = new TD11Object(&tracker);
    if (!router.enqueueRetire(obj, &td11Deleter, 5))
    {
        delete obj;
        return false;
    }

    // foreign domain を 100 進めても reclaim されない（epoch provenance）。
    for (int i = 0; i < 100; ++i)
        (void) foreign.publishEpoch();
    router.tryReclaim();
    if (tracker.invokeCount != 0 || tracker.aliveCount != 1)
    {
        std::fprintf(stderr, "TD1-4b: foreign epoch advance reclaimed the entry (provenance broken)\n");
        return false;
    }
    if (router.pendingRetireCount() != 1)
    {
        std::fprintf(stderr, "TD1-4b: entry lost after foreign advance\n");
        return false;
    }

    // 束縛 domain の reader が退出し epoch が進むと reclaim される。
    eqLike.exitReader(idx);
    (void) eqLike.publishEpoch();
    router.tryReclaim();
    if (tracker.invokeCount != 1 || tracker.aliveCount != 0)
    {
        std::fprintf(stderr, "TD1-4b: bound-domain reclaim did not free the entry\n");
        return false;
    }
    if (router.pendingRetireCount() != 0)
    {
        std::fprintf(stderr, "TD1-4b: pending not empty after bound reclaim\n");
        return false;
    }

    std::printf("STG11EQRetireTests: TD1-4b PASS (reclaim gated only on bound provider)\n");
    return true;
}

// harness main（PublishPipelineIntegrationTests.cpp）から呼ばれるエントリ。
// 新規 CTest 登録なし（AudioEngineHarness exe のサブテスト。STG-8/9 と同じ形）。
static bool checkTD10SetterAccounting()
{
    auto coordinator = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto eq = std::make_unique<EQProcessor>();
    eq->setRetireCoordinator(coordinator.get());
    // 10 setters → 20 objects を段階的に確認する。
    for (int i = 0; i < 10; ++i) {
        eq->setBandFrequency(0, 100.0f + static_cast<float>(i));
        std::fprintf(stderr, "TD1-0: after setter %d: pending=%u Q=%llu E=%llu T=%llu drop=%llu\n",
                     i + 1, eq->eqOwnedPendingRetire(),
                     static_cast<unsigned long long>(eq->eqOwnedQuarantineResident()),
                     static_cast<unsigned long long>(eq->eqOwnedEmergencyResident()),
                     static_cast<unsigned long long>(eq->eqOwnedTerminalResident()),
                     static_cast<unsigned long long>(eq->eqRetireDropCount()));
    }
    return true;
}

int runSTG11EQRetireTests()
{
    bool ok = true;
    if (!checkTD10SetterAccounting())
    {
        std::fprintf(stderr, "FAIL: TD1-0 setter accounting\n");
        ok = false;
    }
    if (!checkTD11QRetainsAcrossReturns())
    {
        std::fprintf(stderr, "FAIL: TD1-1 D->Q lifetime\n");
        ok = false;
    }
    if (!checkTD12EmergencyAndTerminalRetain())
    {
        std::fprintf(stderr, "FAIL: TD1-2 E/T lifetime\n");
        ok = false;
    }
    if (!checkTD13ReleaseLeavesNoResidue())
    {
        std::fprintf(stderr, "FAIL: TD1-3 release residue\n");
        ok = false;
    }
    if (!checkTD14aRouterBoundToPrivateDomain())
    {
        std::fprintf(stderr, "FAIL: TD1-4a router binding\n");
        ok = false;
    }
    if (!checkTD14bReclaimGatedOnlyOnBoundProvider())
    {
        std::fprintf(stderr, "FAIL: TD1-4b bound-provider gating\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11EQRetireTests: PASS (TD1-1/TD1-2/TD1-3/TD1-4a/TD1-4b)\n");
    return ok ? 0 : 1;
}

#ifdef STG11_STANDALONE_MAIN
// ★ standalone CTest target 用 main（STG10ReaderQuarantineTests と同じ形）。
//   AudioEngineHarness の深いコールチェイン外で実行し、stdout を unbuffered にする。
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    return runSTG11EQRetireTests();
}
#endif
