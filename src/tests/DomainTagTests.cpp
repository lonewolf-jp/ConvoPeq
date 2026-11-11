//==============================================================================
// DomainTagTests.cpp
//
// domain tag 設計差分 v2.9 の §7.2 検証のうち、レコーダ／型の契約を単体で固定する。
//   #15-a 値域制約（共通変換ロジック）
//   #15-b Release の実 ctor（不正値 → Unknown・カウンタ +1・停止しない）
//   #15-c Debug の実 ctor（--assert-probe 指定時。デバッガ接続時のみブレーク）
//   #6/#8/#13 ドメイン別集計の不変条件・Unknown の非混入・未刻印の不在
//   #9  retireLatencyD1 の互換（閾値・パス数の固定）
//   #10 サイズ・型（DeferredDeletionQueue.h の static_assert と一致）
//
// ビルド: CMakeLists.txt の DomainTagTests ターゲット（JUCE 非依存）。
//==============================================================================

#include "core/RetireDomain.h"
#include "core/EpochDomain.h"
#include "DeferredDeletionQueue.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <type_traits>

#if defined (_WIN32)
    #include <windows.h>
#endif

namespace
{
int g_failures = 0;

void check (bool cond, const char* what)
{
    std::printf ("  %s: %s\n", cond ? "ok  " : "FAIL", what);

    if (! cond)
        ++g_failures;
}

std::uint64_t nowNs()
{
    using namespace std::chrono;
    return static_cast<std::uint64_t> (
        duration_cast<nanoseconds> (steady_clock::now().time_since_epoch()).count());
}

// #15-c 用: 実 ctor に不正値を渡し、SEH で捕捉できるかを見る
bool g_caught = false;
unsigned long g_code = 0;

void constructInvalidInSeh()
{
    // EpochDomain は仮想デストラクタを持つため、自動変数として __try 内に置くと C2712 になる。
    // 静的ストレージへ placement new して、巻き戻し対象のオブジェクトを作らない。
    alignas (convo::EpochDomain) static unsigned char storage[sizeof (convo::EpochDomain)];

    __try
    {
        ::new (static_cast<void*> (storage)) convo::EpochDomain (convo::RetireDomainId::Count);
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        g_caught = true;
        g_code = GetExceptionCode();
    }
}

void testValueRange()   // #15-a
{
    std::printf ("[#15-a] 値域制約（共通変換ロジック）\n");

    check (convo::EpochDomain::isAssignableDomain (convo::RetireDomainId::Unknown), "Unknown is assignable");
    check (convo::EpochDomain::isAssignableDomain (convo::RetireDomainId::EngineISR), "EngineISR is assignable");
    check (convo::EpochDomain::isAssignableDomain (convo::RetireDomainId::EQ), "EQ is assignable");
    check (! convo::EpochDomain::isAssignableDomain (convo::RetireDomainId::Unstamped), "Unstamped is not assignable");
    check (! convo::EpochDomain::isAssignableDomain (convo::RetireDomainId::Count), "Count is not assignable");
    check (! convo::EpochDomain::isAssignableDomain (static_cast<convo::RetireDomainId> (5)), "5 is not assignable");
    check (! convo::EpochDomain::isAssignableDomain (static_cast<convo::RetireDomainId> (255)), "255 is not assignable");

    check (convo::EpochDomain::sanitizeDomain (static_cast<convo::RetireDomainId> (5))
               == convo::RetireDomainId::Unknown, "sanitize(5) == Unknown");
    check (convo::EpochDomain::sanitizeDomain (convo::RetireDomainId::Unstamped)
               == convo::RetireDomainId::Unknown, "sanitize(Unstamped) == Unknown");
    check (convo::EpochDomain::sanitizeDomain (convo::RetireDomainId::EQ)
               == convo::RetireDomainId::EQ, "sanitize(EQ) == EQ");
}

void testConstructorContract()   // #15-b（Release の実 ctor。Debug でも同じ結果になる）
{
    std::printf ("[#15-b] 実 ctor の値域契約\n");

    const auto base = convo::EpochDomain::invalidDomainAssignments();

    // EpochDomain は 4096 エントリのキューを内包し 1 個で約 230 KB。スタックに複数置かない。
    auto engine = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::EngineISR);
    auto eq = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::EQ);
    auto unknown = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unknown);
    check (engine->domainId() == convo::RetireDomainId::EngineISR, "EngineISR adopted");
    check (eq->domainId() == convo::RetireDomainId::EQ, "EQ adopted");
    check (unknown->domainId() == convo::RetireDomainId::Unknown, "Unknown adopted");
    check (convo::EpochDomain::invalidDomainAssignments() == base, "valid values do not bump the counter");

    // 不正値は「採用しない」。Debug では assertion が走るため、ここは Release でのみ実行する。
#if defined (NDEBUG)
    auto bad5 = std::make_unique<convo::EpochDomain> (static_cast<convo::RetireDomainId> (5));
    auto badCount = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Count);
    auto badUnstamped = std::make_unique<convo::EpochDomain> (convo::RetireDomainId::Unstamped);

    check (bad5->domainId() == convo::RetireDomainId::Unknown, "5 falls back to Unknown");
    check (badCount->domainId() == convo::RetireDomainId::Unknown, "Count falls back to Unknown");
    check (badUnstamped->domainId() == convo::RetireDomainId::Unknown, "Unstamped falls back to Unknown");
    check (convo::EpochDomain::invalidDomainAssignments() == base + 3, "counter +3 for invalid values");
    check (bad5->domainId() != convo::RetireDomainId::EngineISR
               && bad5->domainId() != convo::RetireDomainId::EQ, "invalid value never becomes a real domain");
#else
    std::printf ("  (skip: Debug では assertion が走るため、不正値の生成は --assert-probe で行う)\n");
#endif
}

void testRecorderInvariants()   // #6 / #8 / #13
{
    std::printf ("[#6/#8/#13] ドメイン別集計の不変条件\n");

    auto& rec = convo::isr::RetireLatencyRecorder::instance();
    rec.reset();

    // ドメイン別に 4 完了を作る（EngineISR=classified, EQ=excluded, Unknown=unclassified, Unstamped=excluded）
    static int objs[4];

    struct Case { int* obj; convo::RetireDomainId domain; convo::isr::RetireLatencyRecorder::Path path; int mode; };
    const Case cases[] = {
        { &objs[0], convo::RetireDomainId::EngineISR, convo::isr::RetireLatencyRecorder::Path::Queue,      2 },
        { &objs[1], convo::RetireDomainId::EQ,        convo::isr::RetireLatencyRecorder::Path::Drain,      0 },
        { &objs[2], convo::RetireDomainId::Unknown,   convo::isr::RetireLatencyRecorder::Path::Sync,       1 },
        { &objs[3], convo::RetireDomainId::Unstamped, convo::isr::RetireLatencyRecorder::Path::Quarantine, 0 },
    };

    for (const auto& c : cases)
    {
        rec.recordRequest (c.obj, 1);
        std::uint64_t firstScan = 0;

        if (c.mode >= 1)
        {
            const std::uint64_t t = nowNs();
            rec.recordSafe (c.obj);
            if (c.mode == 2)
                firstScan = t;
        }

        rec.recordComplete (c.obj, c.path, c.domain, firstScan);
    }

    const auto snap = rec.snapshot();
    const auto& tot = snap.classTotals;

    std::uint64_t sumCount = 0, sumClassified = 0, sumUnclassified = 0, sumExcluded = 0, sumConsistency = 0;

    for (int d = 0; d < convo::kRetireDomainCount; ++d)
    {
        sumCount += snap.domains[d].count;
        sumClassified += snap.domains[d].classified;
        sumUnclassified += snap.domains[d].unclassified;
        sumExcluded += snap.domains[d].excluded;
        sumConsistency += snap.domains[d].consistencyErrors;
    }

    check (sumCount == tot.completions, "sum(domains.count) == totals.completions");
    check (sumClassified == tot.classified, "sum(domains.classified) == totals.classified");
    check (sumUnclassified == tot.unclassified, "sum(domains.unclassified) == totals.unclassified");
    check (sumExcluded == tot.excluded, "sum(domains.excluded) == totals.excluded");
    check (sumConsistency == tot.consistencyErrors, "sum(domains.consistencyErrors) == totals.consistencyErrors");
    check (tot.classified + tot.unclassified + tot.excluded == tot.completions,
           "classified + unclassified + excluded == completions");

    // 原因の分離
    check (snap.domains[static_cast<int> (convo::RetireDomainId::Unstamped)].count == snap.stampMissing,
           "domains[Unstamped].count == stampMissing");
    check (snap.domains[static_cast<int> (convo::RetireDomainId::Unknown)].count
               == snap.providerUnknown + snap.providerAbsent,
           "domains[Unknown].count == providerUnknown + providerAbsent");
    check (snap.providerAbsent == 0, "providerAbsent == 0 (防御的・到達不能)");

    // Unknown の非混入（EngineISR / EQ はそれぞれ 1 件だけ）
    check (snap.domains[static_cast<int> (convo::RetireDomainId::EngineISR)].count == 1,
           "EngineISR count == 1 (Unknown が混入していない)");
    check (snap.domains[static_cast<int> (convo::RetireDomainId::EQ)].count == 1,
           "EQ count == 1 (Unknown が混入していない)");
    check (snap.stampMissing == 1, "stampMissing counts the Unstamped completion");

    // パス別の成分も一致するか（JSON の domainSumsConsistent と同じ検査）
    bool pathSumsOk = true;

    for (std::size_t p = 0; p < convo::isr::RetireLatencyRecorder::kPathCount; ++p)
    {
        std::uint64_t driver = 0, ebr = 0, exec = 0, unclassified = 0, excluded = 0;

        for (int d = 0; d < convo::kRetireDomainCount; ++d)
        {
            driver += snap.domains[d].paths[p].driverWait.count;
            ebr += snap.domains[d].paths[p].ebrWait.count;
            exec += snap.domains[d].paths[p].execWait.count;
            unclassified += snap.domains[d].paths[p].unclassifiedCount;
            excluded += snap.domains[d].paths[p].excludedCount;
        }

        if (driver != snap.classifiedPaths[p].driverWait.count
            || ebr != snap.classifiedPaths[p].ebrWait.count
            || exec != snap.classifiedPaths[p].execWait.count
            || unclassified != snap.classifiedPaths[p].unclassifiedCount
            || excluded != snap.classifiedPaths[p].excludedCount)
            pathSumsOk = false;
    }

    check (pathSumsOk, "Σ domains[].paths[p].<component> == paths[p].<component>");
    check (snap.domains[static_cast<int> (convo::RetireDomainId::EngineISR)].classified == 1,
           "EngineISR classified == 1");
}

void testCompatibilityAndSizes()   // #9 / #10
{
    std::printf ("[#9/#10] 既存互換とサイズ\n");

    static_assert (convo::isr::RetireLatencyRecorder::kThresholdNs == 50'000'000ull,
                   "50 ms 閾値は変更しない");
    static_assert (static_cast<int> (convo::isr::RetireLatencyRecorder::Path::Count) == 6,
                   "パス数は 6 のまま（キー構造の互換）");
    check (true, "kThresholdNs == 50 ms / Path::Count == 6 (static_assert)");

    check (sizeof (convo::RetireDomainId) == 1, "sizeof(RetireDomainId) == 1");
    check (sizeof (DeletionEntry) == (sizeof (void*) == 8 ? 56u : 0u)
               || sizeof (DeletionEntry) == 64u, "sizeof(DeletionEntry) is 56 or 64 (構成で異なる)");
    check (alignof (DeletionEntry) == 8, "alignof(DeletionEntry) == 8");
    check (std::is_trivially_copyable_v<DeletionEntry>, "DeletionEntry is trivially copyable");
    check (std::is_trivially_destructible_v<DeletionEntry>, "DeletionEntry is trivially destructible");
}

int runAssertProbe()   // #15-c
{
    const bool underDebugger =
#if defined (_WIN32)
        ::IsDebuggerPresent() != FALSE;
#else
        false;
#endif

    std::printf ("[#15-c] 実 ctor の assertion（%s）\n",
                 underDebugger ? "デバッガ接続" : "デバッガ非接続");

    const auto base = convo::EpochDomain::invalidDomainAssignments();
    constructInvalidInSeh();

#if defined (NDEBUG)
    // Release: assertion は無効。ブレークしない。
    check (! g_caught, "Release: no break");
    check (convo::EpochDomain::invalidDomainAssignments() == base + 1, "Release: counter +1");
#else
    if (underDebugger)
    {
        std::printf ("       exception code = 0x%08lx (EXCEPTION_BREAKPOINT = 0x%08x)\n",
                     g_code, (unsigned) EXCEPTION_BREAKPOINT);
        check (g_caught, "Debug + debugger: assertion broke (EXCEPTION_BREAKPOINT caught)");
        check (g_code == EXCEPTION_BREAKPOINT, "exception code is EXCEPTION_BREAKPOINT");
        check (convo::EpochDomain::invalidDomainAssignments() == base + 1,
               "counter was incremented BEFORE the break");
    }
    else
    {
        // 15-c-1: デバッガ非接続ではブレークしない（JUCE の jassertfalse と同じ条件）
        check (! g_caught, "Debug + no debugger: no break");
        check (convo::EpochDomain::invalidDomainAssignments() == base + 1, "counter +1");
    }
#endif

    std::printf ("  failures=%d\n%s\n", g_failures, g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
} // namespace

int main (int argc, char** argv)
{
    // --assert-probe は引数位置に依存させない（デバッガ経由では結果ファイル等が先に来る）
    bool assertProbe = false;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp (argv[i], "--assert-probe") == 0)
            assertProbe = true;
    }

    if (assertProbe)
        return runAssertProbe();

    testValueRange();
    testConstructorContract();
    testRecorderInvariants();
    testCompatibilityAndSizes();

    std::printf ("  failures=%d\n%s\n", g_failures, g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
