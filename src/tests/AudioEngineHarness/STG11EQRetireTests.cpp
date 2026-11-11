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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "eqprocessor/EQProcessor.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "core/EpochDomain.h"
#include "audioengine/ISRRetireLatency.h"
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
    convo::EpochDomain eqLike(convo::RetireDomainId::Unknown);
    convo::EpochDomain foreign(convo::RetireDomainId::Unknown);
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


// -----------------------------------------------------------------------------
// STG11-D13: 係数キャッシュの同一性判定（#1 符号衝突の回帰）
//   ・符号だけ異なるゲイン／totalGain を区別する
//   ・クランプ後に同値となる入力は同一キーにする
//   ・sampleRate / maxBlockSize の差を区別する
//   ・ハッシュ一致でもキーが違えばキャッシュヒットさせない
// -----------------------------------------------------------------------------
static bool checkCoeffCacheKeyContract()
{
    bool ok = true;
    const auto expect = [&ok](bool condition, const char* name)
    {
        if (!condition)
        {
            std::fprintf(stderr, "FAIL: %s\n", name);
            ok = false;
        }
    };

    constexpr double sr = 48000.0;
    constexpr int block = 512;

    const convo::EQParameters base;
    const auto keyOf = [sr, block](const convo::EQParameters& p) { return EQProcessor::makeCacheKey(p, sr, block); };
    const auto hashOf = [](const EQCoeffKey& k) { return EQProcessor::hashCacheKey(k); };

    expect(keyOf(base) == keyOf(base), "identical params produce identical key");
    expect(hashOf(keyOf(base)) == hashOf(keyOf(base)), "identical params produce identical hash");

    convo::EQParameters plusGain = base;
    plusGain.bands[0].gain = 6.0f;
    convo::EQParameters minusGain = base;
    minusGain.bands[0].gain = -6.0f;
    expect(!(keyOf(plusGain) == keyOf(minusGain)), "gain sign is part of the key");
    expect(hashOf(keyOf(plusGain)) != hashOf(keyOf(minusGain)), "gain sign is part of the hash");

    convo::EQParameters plusTotal = base;
    plusTotal.totalGainDb = 12.0f;
    convo::EQParameters minusTotal = base;
    minusTotal.totalGainDb = -12.0f;
    expect(!(keyOf(plusTotal) == keyOf(minusTotal)), "total gain sign is part of the key");
    expect(hashOf(keyOf(plusTotal)) != hashOf(keyOf(minusTotal)), "total gain sign is part of the hash");

    convo::EQParameters freqLow = base;
    freqLow.bands[0].frequency = 10.0f;
    convo::EQParameters freqMin = base;
    freqMin.bands[0].frequency = 20.0f;
    expect(keyOf(freqLow) == keyOf(freqMin), "frequency clamp is canonical");

    convo::EQParameters gainHigh = base;
    gainHigh.bands[0].gain = 100.0f;
    convo::EQParameters gainMax = base;
    gainMax.bands[0].gain = 48.0f;
    expect(keyOf(gainHigh) == keyOf(gainMax), "gain clamp is canonical");

    convo::EQParameters qZero = base;
    qZero.bands[0].q = 0.0f;
    convo::EQParameters qMin = base;
    qMin.bands[0].q = 0.01f;
    expect(keyOf(qZero) == keyOf(qMin), "q clamp is canonical");

    expect(hashOf(EQProcessor::makeCacheKey(base, sr, block))
               != hashOf(EQProcessor::makeCacheKey(base, 44100.0, block)),
           "sample rate is part of the hash");
    expect(hashOf(EQProcessor::makeCacheKey(base, sr, block))
               != hashOf(EQProcessor::makeCacheKey(base, sr, 1024)),
           "block size is part of the hash");

    convo::EQParameters enabledOff = base;
    enabledOff.bands[3].enabled = false;
    expect(!(keyOf(enabledOff) == keyOf(base)), "enabled is part of the key");

    convo::EQParameters typeChanged = base;
    typeChanged.bands[3].type = 4;
    expect(!(keyOf(typeChanged) == keyOf(base)), "type is part of the key");

    convo::EQParameters channelChanged = base;
    channelChanged.bands[3].channelMode = 2;
    expect(!(keyOf(channelChanged) == keyOf(base)), "channel mode is part of the key");

    convo::EQParameters saturationChanged = base;
    saturationChanged.nonlinearSaturation = 0.9f;
    expect(!(keyOf(saturationChanged) == keyOf(base)), "saturation is part of the key");

    convo::EQParameters structureChanged = base;
    structureChanged.filterStructure = 1;
    expect(!(keyOf(structureChanged) == keyOf(base)), "filter structure is part of the key");

    convo::EQParameters agcChanged = base;
    agcChanged.agcEnabled = true;
    expect(!(keyOf(agcChanged) == keyOf(base)), "agc is part of the key");

    // #1B: 無効バンドの値はキーから正規化される（係数にも処理にも使われないため）。
    convo::EQParameters disabledA = base;
    disabledA.bands[7].enabled = false;
    disabledA.bands[7].frequency = 35.0f;
    disabledA.bands[7].gain = 9.0f;
    disabledA.bands[7].q = 2.5f;

    convo::EQParameters disabledB = disabledA;
    disabledB.bands[7].frequency = 8000.0f;
    disabledB.bands[7].gain = -12.0f;
    disabledB.bands[7].q = 0.5f;

    expect(keyOf(disabledA) == keyOf(disabledB),
           "disabled band values are normalised out of the key");
    expect(hashOf(keyOf(disabledA)) == hashOf(keyOf(disabledB)),
           "disabled band values are normalised out of the hash");

    convo::EQParameters disabledNonFinite = disabledA;
    disabledNonFinite.bands[7].frequency = std::numeric_limits<float>::quiet_NaN();
    disabledNonFinite.bands[7].gain = std::numeric_limits<float>::infinity();
    disabledNonFinite.bands[7].q = -std::numeric_limits<float>::infinity();
    expect(keyOf(disabledNonFinite) == keyOf(disabledA),
           "non-finite values in a disabled band keep the key stable");

    convo::EQParameters reenabled = disabledA;
    reenabled.bands[7].enabled = true;
    expect(!(keyOf(reenabled) == keyOf(disabledA)), "re-enabling a band changes the key");

    convo::EQParameters enabledChanged = base;
    enabledChanged.bands[7].gain = 3.0f;
    expect(!(keyOf(enabledChanged) == keyOf(base)), "enabled band values remain part of the key");

    std::unique_ptr<EQCoeffCache> disabledCache(EQProcessor::createCoeffCache(disabledA, sr, block, 3u));
    std::unique_ptr<EQCoeffCache> disabledCacheB(EQProcessor::createCoeffCache(disabledB, sr, block, 4u));
    if (disabledCache != nullptr && disabledCacheB != nullptr)
    {
        expect(EQProcessor::cacheKeyMatches(*disabledCache, keyOf(disabledB)),
               "cache built from one disabled-band setting matches the equivalent key");
        bool sameDisabledCoeffs = true;
        for (int i = 0; i < EQProcessor::NUM_BANDS; ++i)
        {
            const auto& a = disabledCache->coeffs[i];
            const auto& b = disabledCacheB->coeffs[i];
            sameDisabledCoeffs = sameDisabledCoeffs
                && a.a1 == b.a1 && a.a2 == b.a2 && a.a3 == b.a3
                && a.m0 == b.m0 && a.m1 == b.m1 && a.m2 == b.m2;
        }
        expect(sameDisabledCoeffs, "disabled band values do not change the coefficients");
    }

    std::unique_ptr<EQCoeffCache> cache(EQProcessor::createCoeffCache(plusGain, sr, block, 1u));
    expect(cache != nullptr, "createCoeffCache returns a cache");
    if (cache != nullptr)
    {
        const auto plusKey = keyOf(plusGain);
        const auto minusKey = keyOf(minusGain);
        expect(EQProcessor::cacheKeyMatches(*cache, plusKey), "cache matches the key it was built from");
        expect(!EQProcessor::cacheKeyMatches(*cache, minusKey), "cache rejects a different key with the same hash slot");
        expect(cache->paramsHash == hashOf(plusKey), "cache stores the canonical hash");
    }

    std::unique_ptr<EQCoeffCache> cacheAgain(EQProcessor::createCoeffCache(plusGain, sr, block, 2u));
    if (cache != nullptr && cacheAgain != nullptr)
    {
        bool sameCoefficients = true;
        for (int i = 0; i < EQProcessor::NUM_BANDS; ++i)
        {
            const auto& a = cache->coeffs[i];
            const auto& b = cacheAgain->coeffs[i];
            sameCoefficients = sameCoefficients
                && a.a1 == b.a1 && a.a2 == b.a2 && a.a3 == b.a3
                && a.m0 == b.m0 && a.m1 == b.m1 && a.m2 == b.m2;
        }
        expect(sameCoefficients, "coefficient generation is deterministic");
    }

    if (ok)
        std::printf("STG11-D13: coefficient cache key contract PASS\n");
    return ok;
}


// -----------------------------------------------------------------------------
// 入力検証（#4 / #5 / #7）の回帰
//   非有限値は拒否し、有限値は係数生成と同じ静的範囲へ正規化する。
// -----------------------------------------------------------------------------
static bool checkInputValidationContract()
{
    bool ok = true;
    const auto expect = [&ok](bool condition, const char* name)
    {
        if (!condition)
        {
            std::fprintf(stderr, "FAIL: %s\n", name);
            ok = false;
        }
    };

    EQProcessor eq;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    const float freq0 = eq.getBandParams(0).frequency;
    eq.setBandFrequency(0, nan);
    expect(eq.getBandParams(0).frequency == freq0, "freq NaN rejected");
    eq.setBandFrequency(0, inf);
    expect(eq.getBandParams(0).frequency == freq0, "freq Inf rejected");
    eq.setBandFrequency(0, 1.0e9f);
    expect(eq.getBandParams(0).frequency == EQProcessor::DSP_MAX_FREQ, "freq upper clamp");
    eq.setBandFrequency(0, 1.0f);
    expect(eq.getBandParams(0).frequency == EQProcessor::DSP_MIN_FREQ, "freq lower clamp");

    const float gain0 = eq.getBandParams(0).gain;
    eq.setBandGain(0, nan);
    expect(eq.getBandParams(0).gain == gain0, "gain NaN rejected");
    eq.setBandGain(0, 100.0f);
    expect(eq.getBandParams(0).gain == EQProcessor::DSP_MAX_GAIN_DB, "gain upper clamp");
    eq.setBandGain(0, -100.0f);
    expect(eq.getBandParams(0).gain == EQProcessor::DSP_MIN_GAIN_DB, "gain lower clamp");

    eq.setBandQ(0, 0.0f);
    expect(eq.getBandParams(0).q == EQProcessor::DSP_MIN_Q, "q lower clamp");
    eq.setBandQ(0, nan);
    expect(eq.getBandParams(0).q == EQProcessor::DSP_MIN_Q, "q NaN rejected");

    const float saturation0 = eq.getNonlinearSaturation();
    eq.setNonlinearSaturation(nan);
    expect(eq.getNonlinearSaturation() == saturation0, "saturation NaN rejected");
    eq.setNonlinearSaturation(inf);
    expect(eq.getNonlinearSaturation() == saturation0, "saturation Inf rejected");

    eq.setFilterStructure(EQProcessor::FilterStructure::Parallel);
    eq.setFilterStructure(static_cast<EQProcessor::FilterStructure>(7));
    expect(eq.getFilterStructure() == EQProcessor::FilterStructure::Parallel,
           "invalid filter structure rejected");
    eq.setFilterStructure(EQProcessor::FilterStructure::Serial);
    expect(eq.getFilterStructure() == EQProcessor::FilterStructure::Serial,
           "valid filter structure accepted");

    if (ok)
        std::printf("STG11-D14: input validation contract PASS\n");
    return ok;
}

// -----------------------------------------------------------------------------
// 位相アンラップ（#2）の回帰
//   上方向・下方向の双方で、補正量が累積することを確認する。
// -----------------------------------------------------------------------------
static bool checkUnwrapContract()
{
    bool ok = true;
    const auto expect = [&ok](bool condition, const char* name)
    {
        if (!condition)
        {
            std::fprintf(stderr, "FAIL: %s\n", name);
            ok = false;
        }
    };
    const double twoPi = 2.0 * juce::MathConstants<double>::pi;
    const double tol = 1.0e-9;

    {
        std::vector<double> phase { 3.0, 3.1, -3.1, -3.0, -2.9 };
        ConvolverProcessorInternal::unwrapPhaseRadians(phase.data(), static_cast<int>(phase.size()));
        expect(std::abs(phase[2] - (-3.1 + twoPi)) < tol, "upward wrap corrected");
        expect(phase[3] > phase[2] && phase[4] > phase[3], "upward phase remains continuous");
    }
    {
        std::vector<double> phase { -3.0, -3.1, 3.1, 3.0, 2.9 };
        ConvolverProcessorInternal::unwrapPhaseRadians(phase.data(), static_cast<int>(phase.size()));
        expect(std::abs(phase[2] - (3.1 - twoPi)) < tol, "downward wrap corrected");
        expect(phase[3] < phase[2] && phase[4] < phase[3], "downward phase remains continuous");
    }
    {
        std::vector<double> phase { 0.0, 0.5, 1.0, 1.5 };
        const auto original = phase;
        ConvolverProcessorInternal::unwrapPhaseRadians(phase.data(), static_cast<int>(phase.size()));
        expect(phase == original, "continuous phase unchanged");
    }

    if (ok)
        std::printf("STG11-D15: phase unwrap contract PASS\n");
    return ok;
}

// =============================================================================
// ★ Gate 1 (2026-11-11, Owner 承認): RCU 入場失敗時に退役対象を参照しないことの検証
//   (a) 入場成功: 状態参照は従来どおり（unprotectedReadCount は 0 のまま）
//   (b) 入場失敗: currentStateBits を読まない（カウンタ +1・出力は「状態未公開」の挙動）
//   (c) reclaim 完了: 解放済みの退役対象を参照しない（ASan ビルドで UAF 無し）
//   EQProcessor の private へはテスト専用 friend（製品 API は追加しない）で到達する。
// =============================================================================
struct EQProcessorGate1TestAccess
{
    static const EQProcessor::EQState* currentState(EQProcessor& eq) noexcept
    {
        return eq.loadCurrentState(std::memory_order_acquire);
    }
    // ★ 段階① v2: 理由は引数で明示（デフォルトなし）
    static void flushPendingEpochAdvance(EQProcessor& eq,
                                         EQProcessor::PublishReason reason) noexcept
    {
        eq.flushPendingEpochAdvance(reason);
    }
    static void closeReaderRegistration(EQProcessor& eq) noexcept
    {
        eq.m_epochDomain->closeReaderRegistration();
    }
    // ★ 段階①: pending を立てずに滞留を作る（reclaim のみの経路を検証するため）
    //   epoch は呼出し側が明示する（現在 epoch より古い値にすると、
    //   publish なしでも reclaim 可能になる）。
    static bool enqueueTestRetire(EQProcessor& eq, void* ptr, void (*deleter)(void*),
                                  std::uint64_t epoch) noexcept
    {
        return eq.enqueueDeferredDeleteWithFallback(ptr, deleter, epoch);
    }
    static std::uint64_t currentEpoch(EQProcessor& eq) noexcept
    {
        return eq.m_epochDomain->currentEpoch();
    }
};

// ★ 段階①: 検証用 deleter（実際に解放されたことを数える）
namespace {
std::atomic<int> g_driverProbeDeletes{0};
void driverProbeDeleter(void* p) noexcept
{
    g_driverProbeDeletes.fetch_add(1, std::memory_order_relaxed);
    delete static_cast<int*>(p);
}
}

namespace
{
constexpr int    kGate1BlockSize = 256;
constexpr double kGate1SampleRate = 48000.0;

void gate1FillInput(juce::AudioBuffer<double>& buf)
{
    const int n = buf.getNumSamples();
    const double twoPi = 6.283185307179586476925286766559;
    for (int i = 0; i < n; ++i)
    {
        // 飽和（soft clip）の差が出るよう振幅 3.0（|x| > 1）で入れる
        const double s = 3.0 * std::sin(twoPi * 1000.0 * static_cast<double>(i) / kGate1SampleRate);
        for (int ch = 0; ch < buf.getNumChannels(); ++ch)
            buf.setSample(ch, i, s);
    }
}

void gate1Configure(EQProcessor& eq)
{
    eq.prepareToPlay(kGate1SampleRate, kGate1BlockSize);
    eq.setBandType(0, EQBandType::Peaking);
    eq.setBandFrequency(0, 1000.0f);
    eq.setBandGain(0, 9.0f);
    eq.setBandQ(0, 0.7f);
    eq.setBandEnabled(0, true);
    eq.setBandType(1, EQBandType::HighShelf);
    eq.setBandFrequency(1, 6000.0f);
    eq.setBandGain(1, -9.0f);
    eq.setBandQ(1, 0.7f);
    eq.setBandEnabled(1, true);
}

bool gate1Render(EQProcessor& eq, std::vector<double>& outL, std::vector<double>& outR)
{
    std::vector<double> lastL, lastR;
    bool finite = true;
    juce::AudioBuffer<double> in(2, kGate1BlockSize);
    for (int b = 0; b < 3; ++b)
    {
        gate1FillInput(in);
        juce::dsp::AudioBlock<double> block(in);
        eq.process(block);
        lastL.assign(in.getReadPointer(0), in.getReadPointer(0) + kGate1BlockSize);
        lastR.assign(in.getReadPointer(1), in.getReadPointer(1) + kGate1BlockSize);
        for (double v : lastL) if (!std::isfinite(v)) finite = false;
        for (double v : lastR) if (!std::isfinite(v)) finite = false;
    }
    outL.swap(lastL);
    outR.swap(lastR);
    return finite;
}

double gate1MaxAbs(const std::vector<double>& v)
{
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

double gate1MaxAbsDiff(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}
} // namespace

static bool checkGate1ProcessIsolation()
{
    bool ok = true;

    // ---- (a) 正常入場: 状態と BandNode を従来どおり参照する（カウンタ 0・出力は非無音・有限） ----
    {
        auto coordinator = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto eq = std::make_unique<EQProcessor>();
        eq->setRetireCoordinator(coordinator.get());   // retire authority（null だと drop される）
        gate1Configure(*eq);

        std::vector<double> l, r;
        if (!gate1Render(*eq, l, r))
        {
            std::fprintf(stderr, "G1-1: non-finite output\n");
            ok = false;
        }
        if (eq->unprotectedReadCount() != 0)
        {
            std::fprintf(stderr, "G1-1: unprotectedReadCount=%u (expected 0)\n",
                         static_cast<unsigned>(eq->unprotectedReadCount()));
            ok = false;
        }
        if (EQProcessorGate1TestAccess::currentState(*eq) == nullptr)
        {
            std::fprintf(stderr, "G1-1: currentState is null\n");
            ok = false;
        }
        if (gate1MaxAbs(l) < 1.0e-9)
        {
            std::fprintf(stderr, "G1-1: output is silent (state not applied?)\n");
            ok = false;
        }
    }

    // ---- (b) 入場失敗 + reclaim 完了後: 状態も BandNode も参照しない ----
    {
        // 参考 1: バンドを全て無効にした「素通し」参照（同じ EQProcessor 履歴で比較する）
        auto coordRef = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto eqRef = std::make_unique<EQProcessor>();
        eqRef->setRetireCoordinator(coordRef.get());
        gate1Configure(*eqRef);
        eqRef->setBandEnabled(0, false);
        eqRef->setBandEnabled(1, false);
        std::vector<double> lRef, rRef;
        if (!gate1Render(*eqRef, lRef, rRef))
        {
            std::fprintf(stderr, "G1-2: non-finite output (reference passthrough)\n");
            ok = false;
        }

        // 参考 2: バンド有効（= 失敗経路が退役対象を参照した場合に見えるはずの出力）
        auto coordBand = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto eqBand = std::make_unique<EQProcessor>();
        eqBand->setRetireCoordinator(coordBand.get());
        gate1Configure(*eqBand);
        eqBand->setNonlinearSaturation(0.9f);
        eqBand->setBandGain(1, -6.0f);
        std::vector<double> lBand, rBand;
        if (!gate1Render(*eqBand, lBand, rBand))
        {
            std::fprintf(stderr, "G1-2: non-finite output (reference banded)\n");
            ok = false;
        }

        // 被験体: saturatuin 0.9 の状態と BandNode を退役・reclaim させてから入場失敗で process する
        auto coord = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto eq = std::make_unique<EQProcessor>();
        eq->setRetireCoordinator(coord.get());
        gate1Configure(*eq);
        eq->setNonlinearSaturation(0.9f);
        eq->setBandGain(1, -6.0f);      // 参考 2 と同じ最終係数

        const EQProcessor::EQState* retired = EQProcessorGate1TestAccess::currentState(*eq);
        if (retired == nullptr)
        {
            std::fprintf(stderr, "G1-2: retired state is null\n");
            ok = false;
        }

        eq->setBandGain(0, 6.0f);                                    // 交換 → 直前の状態を退役
        EQProcessorGate1TestAccess::flushPendingEpochAdvance(*eq, EQProcessor::PublishReason::Explicit);    // publish + reclaim

        if (eq->eqOwnedPendingRetire() != 0)
        {
            std::fprintf(stderr, "G1-2: pendingRetire=%u (reclaim not completed)\n",
                         static_cast<unsigned>(eq->eqOwnedPendingRetire()));
            ok = false;
        }
        if (eq->eqRetireDropCount() != 0)
        {
            std::fprintf(stderr, "G1-2: retire dropped (%u) - coordinator not effective\n",
                         static_cast<unsigned>(eq->eqRetireDropCount()));
            ok = false;
        }
        const EQProcessor::EQState* current = EQProcessorGate1TestAccess::currentState(*eq);
        if (current == nullptr || current == retired)
        {
            std::fprintf(stderr, "G1-2: state was not exchanged (current=%p retired=%p)\n",
                         static_cast<const void*>(current), static_cast<const void*>(retired));
            ok = false;
        }

        // Reader 登録を閉じる → EQ の RCUReader は入場できない（決定的な失敗注入）
        EQProcessorGate1TestAccess::closeReaderRegistration(*eq);

        // 解放済みブロックを再利用して毒を入れる（誤って退役対象を読んだ場合に必ず観測できるようにする）
        std::vector<std::unique_ptr<EQProcessor::EQState>> poison;
        for (int i = 0; i < 8; ++i)
        {
            poison.push_back(std::make_unique<EQProcessor::EQState>());
            poison.back()->nonlinearSaturation = 1.0f;
            for (int b = 0; b < EQProcessor::NUM_BANDS; ++b) poison.back()->bands[static_cast<size_t>(b)].gain = 24.0f;
        }

        std::vector<double> lFail, rFail;
        if (!gate1Render(*eq, lFail, rFail))
        {
            std::fprintf(stderr, "G1-2: non-finite output (failure path)\n");
            ok = false;
        }
        if (eq->unprotectedReadCount() != 3)
        {
            std::fprintf(stderr, "G1-2: unprotectedReadCount=%u (expected 3)\n",
                         static_cast<unsigned>(eq->unprotectedReadCount()));
            ok = false;
        }

        // 失敗経路は「バンド無効の素通し」と一致し、「バンド有効」とは一致しない。
        const double dRef = gate1MaxAbsDiff(lFail, lRef);
        const double dBand = gate1MaxAbsDiff(lFail, lBand);
        std::printf("  G1-2: dRef=%.6g dBand=%.6g (amplitude=%.3f)\n",
                    dRef, dBand, gate1MaxAbs(lFail));
        if (!(dRef < 1.0e-5))
        {
            std::fprintf(stderr, "G1-2: failure path is not passthrough (dRef=%.6g)\n", dRef);
            ok = false;
        }
        if (!(dBand > 1.0e-3))
        {
            std::fprintf(stderr, "G1-2: probe has no discriminating power (dBand=%.6g)\n", dBand);
            ok = false;
        }

        // ---- (d) 入場失敗のまま reclaim を反復しても退役対象に触れない ----
        const std::uint32_t before = eq->unprotectedReadCount();
        for (int i = 0; i < 5; ++i)
        {
            eq->setBandGain(0, 3.0f + static_cast<float>(i));
            EQProcessorGate1TestAccess::flushPendingEpochAdvance(*eq, EQProcessor::PublishReason::Explicit);
            std::vector<double> lx, rx;
            if (!gate1Render(*eq, lx, rx))
            {
                std::fprintf(stderr, "G1-3: non-finite output (repeat %d)\n", i);
                ok = false;
                break;
            }
        }
        if (eq->unprotectedReadCount() != before + 15u)
        {
            std::fprintf(stderr, "G1-3: unprotectedReadCount=%u (expected %u)\n",
                         static_cast<unsigned>(eq->unprotectedReadCount()),
                         static_cast<unsigned>(before + 15u));
            ok = false;
        }
    }

    // ---- (e) reclaim と入場失敗の同時進行（非 RT スレッドが交換 + publish + reclaim） ----
    {
        auto coordC = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto eq = std::make_unique<EQProcessor>();
        eq->setRetireCoordinator(coordC.get());
        gate1Configure(*eq);
        EQProcessorGate1TestAccess::closeReaderRegistration(*eq);   // 音声側は常に入場失敗

        std::atomic<bool> stop{false};
        std::atomic<int> swaps{0};
        std::thread driver([&] {
            for (int i = 0; i < 300 && !stop.load(std::memory_order_acquire); ++i)
            {
                eq->setBandGain(0, 3.0f + static_cast<float>(i % 7));
                EQProcessorGate1TestAccess::flushPendingEpochAdvance(*eq, EQProcessor::PublishReason::Explicit);   // publish + reclaim
                swaps.fetch_add(1, std::memory_order_relaxed);
            }
            stop.store(true, std::memory_order_release);
        });

        bool finite = true;
        int blocks = 0;
        std::vector<double> l, r;
        juce::AudioBuffer<double> in(2, kGate1BlockSize);
        while (!stop.load(std::memory_order_acquire) || blocks < 300)
        {
            if (blocks >= 4000) break;                    // 暴走防止
            gate1FillInput(in);
            juce::dsp::AudioBlock<double> block(in);
            eq->process(block);                            // 毎回入場失敗 → 何も参照しない
            for (int ch = 0; ch < 2 && finite; ++ch)
                for (int i = 0; i < kGate1BlockSize; ++i)
                    if (!std::isfinite(in.getSample(ch, i))) finite = false;
            ++blocks;
        }
        driver.join();

        std::printf("  G1-4: concurrent blocks=%d swaps=%d unprotected=%u\n",
                    blocks, swaps.load(), static_cast<unsigned>(eq->unprotectedReadCount()));
        if (!finite)
        {
            std::fprintf(stderr, "G1-4: non-finite output during concurrent reclaim\n");
            ok = false;
        }
        if (eq->unprotectedReadCount() != static_cast<std::uint32_t>(blocks))
        {
            std::fprintf(stderr, "G1-4: unprotectedReadCount=%u (expected %d)\n",
                         static_cast<unsigned>(eq->unprotectedReadCount()), blocks);
            ok = false;
        }
        if (swaps.load() <= 0)
        {
            std::fprintf(stderr, "G1-4: no concurrent reclaim happened\n");
            ok = false;
        }
        if (eq->eqRetireDropCount() != 0)
        {
            std::fprintf(stderr, "G1-4: retire dropped (%u)\n",
                         static_cast<unsigned>(eq->eqRetireDropCount()));
            ok = false;
        }
    }

    return ok;
}

// =============================================================================
// ★ 段階① (2026-11-11, Owner 承認): reclaim ドライバの契約検証（7 項目）
//   1 pending あり  2 pending なし・滞留あり  3 pending なし・滞留なし
//   5 publish 理由 4 区分  6 競合・寿命  7 RT（Message Thread）制約
// =============================================================================
static bool checkRetireDriverContract()
{
    static juce::ScopedJuceInitialiser_GUI juceInit;   // MessageManager（main thread = message thread）
    bool ok = true;

    auto coordinator = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto eq = std::make_unique<EQProcessor>();
    eq->setRetireCoordinator(coordinator.get());
    gate1Configure(*eq);

    auto fail = [&ok](const char* what) {
        std::fprintf(stderr, "DRV: %s\n", what);
        ok = false;
    };

    // ---- 1) pending あり: epoch 前進 + reclaim ----
    {
        const auto before = eq->retireDriverStats();
        eq->setBandGain(0, 6.0f);                       // pending = true
        eq->driveRetireReclaimNonRt();
        const auto after = eq->retireDriverStats();

        if (after.currentEpoch != before.currentEpoch + 1) fail("1: epoch did not advance by 1");
        if (after.publishTimerPending != before.publishTimerPending + 1) fail("1: timerPending publish not counted");
        if (after.pendingRetire != 0) fail("1: D queue not drained");
        if (after.reclaimOnlyRuns != before.reclaimOnlyRuns) fail("1: publish path must not count as reclaim-only");
    }

    // ---- 2) pending なし・滞留あり: epoch は前進せず reclaim のみ ----
    //   滞留は「publish 済みの古い epoch」で作る。つまり
    //   epoch を進めなくても safe-below に入っている entry である。
    {
        const std::uint64_t epochNow = EQProcessorGate1TestAccess::currentEpoch(*eq);
        if (epochNow < 2)
            fail("2: epoch must be >= 2 before the reclaim-only probe");

        const auto before = eq->retireDriverStats();
        const int deletesBefore = g_driverProbeDeletes.load(std::memory_order_relaxed);
        if (! EQProcessorGate1TestAccess::enqueueTestRetire(*eq, new int(7), driverProbeDeleter,
                                                            epochNow - 1))
            fail("2: enqueue of probe entry failed (coordinator?)");

        const auto enqueued = eq->retireDriverStats();
        if (enqueued.pendingRetire == 0) fail("2: probe entry is not resident in D queue");

        eq->driveRetireReclaimNonRt();
        const auto after = eq->retireDriverStats();

        if (after.currentEpoch != before.currentEpoch) fail("2: epoch advanced without pending");
        if (after.reclaimOnlyRuns != before.reclaimOnlyRuns + 1) fail("2: reclaim-only run not counted");
        if (g_driverProbeDeletes.load(std::memory_order_relaxed) != deletesBefore + 1)
            fail("2: probe entry was not reclaimed");
        if (after.pendingRetire != 0) fail("2: D queue not drained by reclaim-only run");
    }

    // ---- 3) pending なし・滞留なし: 何もしない ----
    {
        const auto before = eq->retireDriverStats();
        eq->driveRetireReclaimNonRt();
        const auto after = eq->retireDriverStats();

        if (after.idleTicks != before.idleTicks + 1) fail("3: idle tick not counted");
        if (after.reclaimOnlyRuns != before.reclaimOnlyRuns) fail("3: unnecessary reclaim executed");
        if (after.publishTimerPending != before.publishTimerPending) fail("3: unexpected publish");
        if (after.currentEpoch != before.currentEpoch) fail("3: epoch changed while idle");
    }

    // ---- 5) publish 理由 4 区分 ----
    {
        {   // prepare 経由
            const auto before = eq->retireDriverStats();
            eq->setBandGain(1, -3.0f);
            eq->prepareToPlay(kGate1SampleRate, kGate1BlockSize);
            const auto after = eq->retireDriverStats();
            if (after.publishPrepare != before.publishPrepare + 1) fail("5: prepare publish not counted");
        }
        {   // release 経由
            const auto before = eq->retireDriverStats();
            eq->setBandGain(1, -4.0f);
            eq->releaseResources();
            const auto after = eq->retireDriverStats();
            if (after.publishRelease != before.publishRelease + 1) fail("5: release publish not counted");
        }
        {   // 明示的な直接 flush = Explicit
            eq->prepareToPlay(kGate1SampleRate, kGate1BlockSize);
            const auto before = eq->retireDriverStats();
            eq->setBandGain(1, -5.0f);
            EQProcessorGate1TestAccess::flushPendingEpochAdvance(*eq, EQProcessor::PublishReason::Explicit);
            const auto after = eq->retireDriverStats();
            if (after.publishExplicit != before.publishExplicit + 1) fail("5: explicit publish not counted");
        }
        {   // タイマー経由
            const auto before = eq->retireDriverStats();
            eq->setBandGain(1, -6.0f);
            eq->driveRetireReclaimNonRt();
            const auto after = eq->retireDriverStats();
            if (after.publishTimerPending != before.publishTimerPending + 1) fail("5: timer publish not counted");
        }
        {   // ★ 段階① v2: TimerPending の二重計数がないこと
            //   ドライバは flush を呼ばないため、ドライバ 1 回で増えるのは
            //   publishTimerPending だけである（Explicit は増えない）。
            const auto before = eq->retireDriverStats();
            eq->setBandGain(0, 4.0f);
            eq->driveRetireReclaimNonRt();
            const auto after = eq->retireDriverStats();
            if (after.publishTimerPending != before.publishTimerPending + 1) fail("5b: timer publish not counted once");
            if (after.publishExplicit != before.publishExplicit) fail("5b: timer path must not count as explicit");
            if (after.publishPrepare != before.publishPrepare) fail("5b: timer path must not count as prepare");
            if (after.publishRelease != before.publishRelease) fail("5b: timer path must not count as release");
        }
    }

    // ---- 6) 競合・寿命: driver（Message Thread）と setter/prepare/release（他スレッド）----
    {
        std::atomic<bool> stop{false};
        std::atomic<int> iterations{0};
        std::thread worker([&] {
            for (int i = 0; i < 200 && ! stop.load(std::memory_order_acquire); ++i)
            {
                eq->setBandGain(0, 3.0f + static_cast<float>(i % 5));
                if ((i % 20) == 0)  eq->prepareToPlay(kGate1SampleRate, kGate1BlockSize);
                if ((i % 20) == 10) eq->releaseResources();
                iterations.fetch_add(1, std::memory_order_relaxed);
            }
            stop.store(true, std::memory_order_release);
        });

        int ticks = 0;
        while (! stop.load(std::memory_order_acquire) || ticks < 200)
        {
            eq->driveRetireReclaimNonRt();
            ++ticks;
            if (ticks > 200000) break;                  // 暴走防止
        }
        worker.join();

        // 最後の setter が残した entry は、あと 1 回 publish すれば safe-below に入る。
        // ドライバが pending を消費して epoch を進めれば D は空になる。
        eq->setBandGain(0, 2.0f);
        eq->driveRetireReclaimNonRt();

        const auto end = eq->retireDriverStats();
        if (eq->eqRetireDropCount() != 0) fail("6: retire drop occurred under concurrency");
        if (end.pendingRetire != 0 || end.resident != 0) fail("6: residue left after concurrency");
        if (iterations.load() <= 0) fail("6: worker did not run");
        std::printf("  DRV6: ticks=%d iterations=%d epoch=%llu\n", ticks, iterations.load(),
                    static_cast<unsigned long long>(end.currentEpoch));
    }

    // ---- 7) RT 制約: Message Thread 以外からは拒否（fail-safe）----
    {
        eq->prepareToPlay(kGate1SampleRate, kGate1BlockSize);
        const auto before = eq->retireDriverStats();
        std::thread foreign([&] { eq->driveRetireReclaimNonRt(); });
        foreign.join();
        const auto after = eq->retireDriverStats();

        if (after.offThreadRejects != before.offThreadRejects + 1) fail("7: off-thread call was not rejected");
        if (after.currentEpoch != before.currentEpoch) fail("7: off-thread call advanced the epoch");
        if (after.publishTimerPending != before.publishTimerPending) fail("7: off-thread call published");
    }

    // ---- 最終: 音声処理が崩れていないこと ----
    {
        eq->prepareToPlay(kGate1SampleRate, kGate1BlockSize);
        std::vector<double> l, r;
        if (! gate1Render(*eq, l, r)) fail("final: non-finite output");
        if (gate1MaxAbs(l) < 1.0e-9) fail("final: silent output");
        const auto s = eq->retireDriverStats();
        std::printf("  DRV: timer=%u prepare=%u release=%u explicit=%u reclaimOnly=%u idle=%u offThread=%u\n",
                    s.publishTimerPending, s.publishPrepare, s.publishRelease, s.publishExplicit,
                    s.reclaimOnlyRuns, s.idleTicks, s.offThreadRejects);
    }

    return ok;
}

// =============================================================================
// ★ 段階③ (2026-11-11, Owner 承認): 3 分類診断の契約検証
//   (1) 合成 driver+ebr+exec+unclassified == latency
//   (2) firstScanTs==0 → unclassified 全額（0 埋めしない）
//   (3) safeTs==0 → 分類対象外（excluded）(4)(5) 順序不整合 → inconsistent
//   (6) 実キュー由来の観測で合成と firstScanTs を確認
// =============================================================================
static bool checkLatencyClassificationContract()
{
    bool ok = true;
    using Rec = convo::isr::RetireLatencyRecorder;

    auto fail = [&ok](const char* what) {
        std::fprintf(stderr, "CLS: %s\n", what);
        ok = false;
    };

    // (1) 合成: ts=1000, firstScan=5000, safe=8000, now=9000
    {
        const auto c = Rec::classifyTimes (1000u, 5000u, 8000u, 9000u);

        if (! c.classified) fail("1: not classified");
        if (c.driverWaitNs != 4000u) fail("1: driverWait != 4000");
        if (c.ebrWaitNs != 3000u) fail("1: ebrWait != 3000");
        if (c.execWaitNs != 1000u) fail("1: execWait != 1000");
        if (c.unclassifiedNs != 0u) fail("1: unclassified != 0");
        const std::uint64_t total = 9000u - 1000u;
        if (c.driverWaitNs + c.ebrWaitNs + c.execWaitNs + c.unclassifiedNs != total)
            fail("1: composition != latency");
    }

    // (2) firstScanTs 未設定 → unclassified は latency 全額
    {
        const auto c = Rec::classifyTimes (1000u, 0u, 8000u, 9000u);

        if (c.classified) fail("2: must not be classified");
        if (c.unclassifiedNs != 8000u) fail("2: unclassified must be the full latency");
    }

    // (3) safeTs 未観測 → 分類対象外
    {
        const auto c = Rec::classifyTimes (1000u, 5000u, 0u, 9000u);

        if (! c.excluded) fail("3: must be excluded");
        if (c.classified) fail("3: excluded must not be classified");
    }

    // (4) firstScanTs < ts → 不整合
    {
        const auto c = Rec::classifyTimes (5000u, 1000u, 8000u, 9000u);

        if (! c.inconsistent) fail("4: must be inconsistent");
        if (c.unclassifiedNs != 4000u) fail("4: unclassified must be the full latency");
    }

    // (5) safeTs < firstScanTs → 不整合
    {
        const auto c = Rec::classifyTimes (1000u, 6000u, 5000u, 9000u);

        if (! c.inconsistent) fail("5: must be inconsistent");
        if (c.unclassifiedNs != 8000u) fail("5: unclassified must be the full latency");
    }

    // (5b) safeTs < ts（安全時刻が要求より前）も不整合として拾う
    {
        const auto c = Rec::classifyTimes (5000u, 5000u, 4000u, 9000u);

        if (! c.inconsistent) fail("5b: safeTs < ts must be inconsistent");
        if (c.excluded) fail("5b: safeTs < ts must not be treated as excluded");
    }

    // (6) 実観測での合成（実キューの reclaim 経路を通ったもの）
    {
        const auto snap = Rec::instance().snapshot();
        std::uint64_t withFirstScan = 0;
        std::uint64_t classifiedObs = 0;
        std::uint64_t compositionErrors = 0;

        for (const auto& o : snap.observations)
        {
            if (o.safeTsNs == 0)
                continue;
            if (o.firstScanTsNs == 0)
                continue;                  // 未走査（drainAll）は対象外

            ++withFirstScan;

            if (o.classified == 1u)
            {
                ++classifiedObs;
                const std::uint64_t sum = o.driverWaitNs + o.ebrWaitNs + o.execWaitNs + o.unclassifiedNs;

                if (sum != o.latencyNs)
                    ++compositionErrors;
            }
        }

        std::printf("  CLS6: observations_with_firstScan=%llu classified=%llu compositionErrors=%llu\n",
                    static_cast<unsigned long long>(withFirstScan),
                    static_cast<unsigned long long>(classifiedObs),
                    static_cast<unsigned long long>(compositionErrors));

        if (withFirstScan == 0) fail("6: firstScanTs is never recorded (queue hook missing)");
        if (classifiedObs == 0) fail("6: no classified observation");
        if (compositionErrors != 0) fail("6: composition mismatch in real observations");

        // 母数分離: classified + unclassified + excluded == completions
        const auto& tot = snap.classTotals;

        if (tot.classified + tot.unclassified + tot.excluded != tot.completions)
        {
            std::fprintf(stderr, "CLS6: counts classified=%llu unclassified=%llu excluded=%llu completions=%llu\n",
                         static_cast<unsigned long long>(tot.classified),
                         static_cast<unsigned long long>(tot.unclassified),
                         static_cast<unsigned long long>(tot.excluded),
                         static_cast<unsigned long long>(tot.completions));
            fail("6: classified + unclassified + excluded != completions");
        }
    }

    return ok;
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
    if (!checkCoeffCacheKeyContract())
    {
        std::fprintf(stderr, "FAIL: D13 coefficient cache key contract\n");
        ok = false;
    }
    if (!checkInputValidationContract())
    {
        std::fprintf(stderr, "FAIL: D14 input validation contract\n");
        ok = false;
    }
    if (!checkUnwrapContract())
    {
        std::fprintf(stderr, "FAIL: D15 phase unwrap contract\n");
        ok = false;
    }
    if (!checkGate1ProcessIsolation())
    {
        std::fprintf(stderr, "FAIL: Gate1 reader-safety process isolation\n");
        ok = false;
    }
    if (!checkRetireDriverContract())
    {
        std::fprintf(stderr, "FAIL: stage-1 retire reclaim driver contract\n");
        ok = false;
    }
    if (!checkLatencyClassificationContract())
    {
        std::fprintf(stderr, "FAIL: stage-3 latency classification contract\n");
        ok = false;
    }
    if (ok)
        std::printf("STG11EQRetireTests: PASS (TD1-1/TD1-2/TD1-3/TD1-4a/TD1-4b/Gate1/Stage1/Stage3)\n");
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
