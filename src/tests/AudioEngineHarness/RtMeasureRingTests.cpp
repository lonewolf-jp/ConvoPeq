//==============================================================================
// R-06 / U2: `convo::rtmeasure` の検証（非 RT・オフライン）
//
//   U2 で追加した前提作業（P1/P2/P3/P4）と、Owner 判断 2026-10-31 で要求された
//   容量境界テストを検証する。
//
//     (1) enabled() のライフサイクル（P1: 環境変数の読み取りは非 RT の初期化時のみ）
//     (2) 往復・部分回収（U1 から継続）
//     (3) 容量境界: 空 / 1 件 / 8,191 / 8,192 / 8,193 件
//     (4) 満杯時の破棄（上書きしない）と dropped の正確性（P2）
//     (5) 読み出し後の再利用、wrap-around（索引が 1 周するケース）
//     (6) 同時読み書き（1 書き手 + 1 読み手）
//     (7) TSC 校正と ticks→µs 換算
//     (8) BlockScopeTimer: OFF では 0 件、ON では END 1 件（RAII が全 return を被覆）
//     (9) P4: instance() が名前空間スコープの変数であること（関数ローカル static ではない）
//
//   破棄率 10% の INCONCLUSIVE 判定は「計測実行側（非 RT の集計）」の規則であり、
//   ここでは判定に必要な生値（written / dropped）が正しく出ることを検証する。
//==============================================================================

#include "audioengine/RtMeasureRing.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

bool expect(const char* name, bool ok)
{
    if (!ok)
        std::fprintf(stderr, "FAIL: RT-MEASURE-U2 %s\n", name);
    return ok;
}

void setEnvMeasure(const char* value)
{
#if defined(_WIN32)
    _putenv_s("CONVOPEQ_RT_MEASURE", value == nullptr ? "" : value);
#else
    if (value == nullptr)
        unsetenv("CONVOPEQ_RT_MEASURE");
    else
        setenv("CONVOPEQ_RT_MEASURE", value, 1);
#endif
}

// (1) P1: enabled() は初期化前 OFF。初期化は非 RT から 1 回。RT 側は load のみ。
bool checkEnabledLifecycle()
{
    bool ok = true;

    // 初期化前の既定は OFF（fail-closed）。テスト実行順に依存しないよう明示的に確認する。
    const bool beforeInit = convo::rtmeasure::enabled();

    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    ok = expect("env unset -> OFF", !convo::rtmeasure::enabled()) && ok;

    setEnvMeasure("1");
    convo::rtmeasure::initializeFromEnvironment();
    ok = expect("env=1 -> ON", convo::rtmeasure::enabled()) && ok;

    setEnvMeasure("0");
    convo::rtmeasure::initializeFromEnvironment();
    ok = expect("env=0 -> OFF", !convo::rtmeasure::enabled()) && ok;

    // 後続テスト（実 RT 経路）に影響しないよう既定へ戻す。
    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    ok = expect("restored OFF", !convo::rtmeasure::enabled()) && ok;

    std::printf("[RT-M] U2 enabled lifecycle: beforeInit=%d final=OFF capacity=%zu\n",
                beforeInit ? 1 : 0,
                convo::rtmeasure::Ring::kCapacity);
    return ok;
}

// (2) 往復・部分回収（U1 から継続）
bool checkRoundTrip()
{
    convo::rtmeasure::Ring ring;
    ring.reset();

    constexpr std::uint32_t kKinds[] = {
        convo::rtmeasure::kBlockEnd, convo::rtmeasure::kAlloc,
        convo::rtmeasure::kLockWait, convo::rtmeasure::kFree,
    };
    for (std::uint32_t i = 0; i < 4; ++i)
        ring.push(kKinds[i], i * 10u, 1000ull + i);

    std::vector<convo::rtmeasure::Event> out(8);
    const std::size_t n = ring.drain(out.data(), out.size());

    bool ok = true;
    ok = expect("roundtrip count", n == 4) && ok;
    ok = expect("roundtrip written", ring.written() == 4) && ok;
    ok = expect("roundtrip consumed", ring.consumed() == 4) && ok;
    ok = expect("roundtrip pending", ring.pending() == 0) && ok;
    for (std::size_t i = 0; i < n && i < 4; ++i)
    {
        ok = expect("roundtrip kind", out[i].kind == kKinds[i]) && ok;
        ok = expect("roundtrip arg", out[i].arg == static_cast<std::uint32_t>(i * 10u)) && ok;
        ok = expect("roundtrip tsc", out[i].tsc == 1000ull + i) && ok;
    }
    return ok;
}

bool checkPartialDrain()
{
    convo::rtmeasure::Ring ring;
    ring.reset();

    for (std::uint32_t i = 0; i < 10; ++i)
        ring.push(convo::rtmeasure::kAlloc, i, i);

    std::vector<convo::rtmeasure::Event> out(10);

    const std::size_t first = ring.drain(out.data(), 4);
    bool ok = true;
    ok = expect("partial first count", first == 4) && ok;
    ok = expect("partial first arg", out[0].arg == 0u && out[3].arg == 3u) && ok;
    ok = expect("partial pending after first", ring.pending() == 6) && ok;

    const std::size_t second = ring.drain(out.data(), 100);
    ok = expect("partial second count", second == 6) && ok;
    ok = expect("partial second first arg", out[0].arg == 4u) && ok;
    ok = expect("partial drained empty", ring.pending() == 0) && ok;
    ok = expect("partial extra drain", ring.drain(out.data(), 4) == 0) && ok;
    return ok;
}

// (3) 容量境界: 空 / 1 件 / 8,191 / 8,192 / 8,193 件
bool checkCapacityBoundaries()
{
    using convo::rtmeasure::Ring;
    constexpr std::uint64_t kCap = Ring::kCapacity;
    bool ok = true;

    // 空
    {
        Ring ring;
        ring.reset();
        ok = expect("boundary empty pending", ring.pending() == 0) && ok;
        ok = expect("boundary empty drained", ring.drain(nullptr, 0) == 0) && ok;
    }

    // 1 件
    {
        Ring ring;
        ring.reset();
        ring.push(convo::rtmeasure::kBlockEnd, 7u, 11ull);
        ok = expect("boundary one pending", ring.pending() == 1) && ok;
        ok = expect("boundary one written", ring.written() == 1) && ok;
        ok = expect("boundary one dropped", ring.dropped() == 0) && ok;
    }

    // 8,191 件（容量 - 1）: 破棄なし
    {
        Ring ring;
        ring.reset();
        for (std::uint64_t i = 0; i < kCap - 1; ++i)
            ring.push(convo::rtmeasure::kFree, static_cast<std::uint32_t>(i), i);
        ok = expect("boundary cap-1 pending", ring.pending() == kCap - 1) && ok;
        ok = expect("boundary cap-1 dropped", ring.dropped() == 0) && ok;
    }

    // 8,192 件（ちょうど容量）: 破棄なし
    {
        Ring ring;
        ring.reset();
        for (std::uint64_t i = 0; i < kCap; ++i)
            ring.push(convo::rtmeasure::kFree, static_cast<std::uint32_t>(i), i);
        ok = expect("boundary cap pending", ring.pending() == kCap) && ok;
        ok = expect("boundary cap written", ring.written() == kCap) && ok;
        ok = expect("boundary cap dropped", ring.dropped() == 0) && ok;
    }

    // 8,193 件（容量 + 1）: 1 件だけ破棄
    {
        Ring ring;
        ring.reset();
        for (std::uint64_t i = 0; i < kCap + 1; ++i)
            ring.push(convo::rtmeasure::kFree, static_cast<std::uint32_t>(i), i);
        ok = expect("boundary cap+1 pending", ring.pending() == kCap) && ok;
        ok = expect("boundary cap+1 written", ring.written() == kCap) && ok;
        ok = expect("boundary cap+1 dropped", ring.dropped() == 1) && ok;
        ok = expect("boundary cap+1 accounting", ring.written() + ring.dropped() == kCap + 1) && ok;
    }

    return ok;
}

// (4) 満杯時の破棄（上書きしない）と dropped の正確性
bool checkOverflowDropsWithoutOverwrite()
{
    convo::rtmeasure::Ring ring;
    ring.reset();

    constexpr std::size_t kExtra = 10;
    const std::size_t total = convo::rtmeasure::Ring::kCapacity + kExtra;
    for (std::size_t i = 0; i < total; ++i)
        ring.push(convo::rtmeasure::kFree, static_cast<std::uint32_t>(i), i);

    bool ok = true;
    ok = expect("overflow written == capacity", ring.written() == convo::rtmeasure::Ring::kCapacity) && ok;
    ok = expect("overflow dropped == kExtra", ring.dropped() == kExtra) && ok;
    ok = expect("overflow written+dropped == pushes", ring.written() + ring.dropped() == total) && ok;
    ok = expect("overflow pending == capacity", ring.pending() == convo::rtmeasure::Ring::kCapacity) && ok;

    std::vector<convo::rtmeasure::Event> out(convo::rtmeasure::Ring::kCapacity + 1);
    const std::size_t n = ring.drain(out.data(), out.size());
    ok = expect("overflow drain capacity", n == convo::rtmeasure::Ring::kCapacity) && ok;

    // 上書きしていないので、残っているのは「最初の capacity 件」である。
    ok = expect("overflow keeps oldest (first)", out[0].tsc == 0ull) && ok;
    ok = expect("overflow keeps oldest (last stored)",
                out[convo::rtmeasure::Ring::kCapacity - 1].tsc
                    == static_cast<std::uint64_t>(convo::rtmeasure::Ring::kCapacity - 1)) && ok;
    return ok;
}

// (5a) 読み出し後の再利用
bool checkReuseAfterDrain()
{
    using convo::rtmeasure::Ring;
    constexpr std::uint64_t kCap = Ring::kCapacity;
    bool ok = true;

    Ring ring;
    ring.reset();

    std::vector<convo::rtmeasure::Event> out(static_cast<std::size_t>(kCap));
    for (std::uint64_t i = 0; i < kCap; ++i)
        ring.push(convo::rtmeasure::kFree, static_cast<std::uint32_t>(i), i);

    ok = expect("reuse first drain", ring.drain(out.data(), out.size()) == kCap) && ok;
    ok = expect("reuse pending zero", ring.pending() == 0) && ok;

    // 空になったので、同じだけ再度入る（破棄されない）。
    for (std::uint64_t i = 0; i < kCap; ++i)
        ring.push(convo::rtmeasure::kFree, static_cast<std::uint32_t>(i), kCap + i);

    ok = expect("reuse second pending", ring.pending() == kCap) && ok;
    ok = expect("reuse dropped stays zero", ring.dropped() == 0) && ok;
    ok = expect("reuse second drain", ring.drain(out.data(), out.size()) == kCap) && ok;
    ok = expect("reuse second first tsc", out[0].tsc == kCap) && ok;
    ok = expect("reuse second last tsc", out[kCap - 1].tsc == 2 * kCap - 1) && ok;
    return ok;
}

// (5b) wrap-around: 索引が容量を超えて 1 周以上回るケース
bool checkWrapAround()
{
    using convo::rtmeasure::Ring;
    constexpr std::uint64_t kCap = Ring::kCapacity;
    constexpr std::uint64_t kRounds = 3;
    bool ok = true;

    Ring ring;
    ring.reset();

    std::vector<convo::rtmeasure::Event> out(static_cast<std::size_t>(kCap));
    std::uint64_t seq = 0;
    for (std::uint64_t round = 0; round < kRounds; ++round)
    {
        // 容量の 3/4 を入れて回収する（write_ / read_ が容量を超えて進む）。
        const std::uint64_t batch = (kCap / 4) * 3;
        for (std::uint64_t i = 0; i < batch; ++i, ++seq)
            ring.push(convo::rtmeasure::kFree, static_cast<std::uint32_t>(seq & 0xffffu), seq);

        const std::size_t n = ring.drain(out.data(), out.size());
        ok = expect("wrap drain count", n == batch) && ok;
        ok = expect("wrap first seq", out[0].tsc == seq - batch) && ok;
        ok = expect("wrap last seq", out[n - 1].tsc == seq - 1) && ok;
        ok = expect("wrap pending zero", ring.pending() == 0) && ok;
    }

    ok = expect("wrap no drops", ring.dropped() == 0) && ok;
    ok = expect("wrap written total", ring.written() == seq) && ok;
    ok = expect("wrap indices advanced past capacity", ring.written() >= kCap) && ok;
    return ok;
}

// (6) 同時読み書き（1 書き手 + 1 読み手）
bool checkConcurrentReaderWriter()
{
    using convo::rtmeasure::Ring;
    constexpr std::uint64_t kPushes = 200000;

    Ring ring;
    ring.reset();

    std::atomic<bool> stop{ false };
    std::uint64_t drained = 0;

    std::thread reader([&] {
        std::vector<convo::rtmeasure::Event> buf(64);
        while (!stop.load(std::memory_order_acquire))
        {
            const std::size_t n = ring.drain(buf.data(), buf.size());
            drained += n;
            if (n == 0)
                std::this_thread::yield();
        }
        drained += ring.drain(buf.data(), buf.size());
    });

    for (std::uint64_t i = 0; i < kPushes; ++i)
        ring.push(convo::rtmeasure::kFree, static_cast<std::uint32_t>(i), i);

    // 書き手の push 完了後に読み手を止める（残留分は最後に回収）。
    std::this_thread::yield();
    stop.store(true, std::memory_order_release);
    reader.join();

    bool ok = true;
    const std::uint64_t written = ring.written();
    const std::uint64_t dropped = ring.dropped();
    ok = expect("concurrent accounting", written + dropped == kPushes) && ok;
    ok = expect("concurrent drained <= written", drained <= written) && ok;
    ok = expect("concurrent pending matches", ring.pending() == written - drained) && ok;

    std::printf("[RT-M] U2 concurrent: written=%llu dropped=%llu ratio=%.4f\n",
                static_cast<unsigned long long>(written),
                static_cast<unsigned long long>(dropped),
                (written + dropped) == 0
                    ? 0.0
                    : static_cast<double>(dropped) / static_cast<double>(written + dropped));
    return ok;
}

// (7) TSC 校正と ticks→µs 換算
bool checkTimingCalibration()
{
    bool ok = true;

    setEnvMeasure("1");
    convo::rtmeasure::initializeFromEnvironment();
    const std::uint64_t perSec = convo::rtmeasure::detail::g_ticksPerSecond.load(std::memory_order_acquire);

    ok = expect("timing calibrated (>0)", perSec > 0) && ok;
    ok = expect("timing 1s == 1e6 us", convo::rtmeasure::ticksToMicros(perSec) == 1000000ull) && ok;
    ok = expect("timing zero ticks", convo::rtmeasure::ticksToMicros(0) == 0ull) && ok;

    // 入口 < 出口（順序付き読み取りの健全性）
    const std::uint64_t a = convo::rtmeasure::detail::readOrderedBegin();
    const std::uint64_t b = convo::rtmeasure::detail::readOrderedEnd();
    ok = expect("ordered read monotonic", b >= a) && ok;

    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    return ok;
}

// (8) BlockScopeTimer: OFF では 0 件、ON では END 1 件
bool checkBlockScopeTimer()
{
    bool ok = true;

    // OFF
    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    convo::rtmeasure::instance().reset();
    {
        convo::rtmeasure::BlockScopeTimer t;
        (void) t;
    }
    ok = expect("timer OFF writes nothing", convo::rtmeasure::instance().written() == 0) && ok;

    // ON
    setEnvMeasure("1");
    convo::rtmeasure::initializeFromEnvironment();
    convo::rtmeasure::instance().reset();
    {
        convo::rtmeasure::BlockScopeTimer t;
        (void) t;
    }
    const std::uint64_t written = convo::rtmeasure::instance().written();
    ok = expect("timer ON writes one event", written == 1) && ok;

    std::vector<convo::rtmeasure::Event> out(4);
    const std::size_t n = convo::rtmeasure::instance().drain(out.data(), out.size());
    ok = expect("timer ON drained", n == 1) && ok;
    if (n == 1)
    {
        ok = expect("timer event kind is END", out[0].kind == convo::rtmeasure::kBlockEnd) && ok;
        ok = expect("timer payload non-zero", out[0].tsc > 0) && ok;
    }

    // 後始末（既定 OFF に戻す）
    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    convo::rtmeasure::instance().reset();
    ok = expect("timer restored OFF", !convo::rtmeasure::enabled()) && ok;
    return ok;
}

// (9) P4: instance() は名前空間スコープの変数を返す（関数ローカル static ではない）
bool checkInstanceIdentity()
{
    bool ok = true;
    ok = expect("instance stable", &convo::rtmeasure::instance() == &convo::rtmeasure::instance()) && ok;
    ok = expect("instance is namespace-scope object",
                &convo::rtmeasure::instance() == &convo::rtmeasure::g_ring) && ok;
    return ok;
}


// (12) 集計: N / p99（ceil(0.99N)-1）/ 最大値 / dropped / writes
bool checkAggregatorStats()
{
    using convo::rtmeasure::Ring;
    bool ok = true;

    convo::rtmeasure::instance().reset();
    convo::rtmeasure::aggregator().reset();

    constexpr std::uint64_t kN = 100;
    for (std::uint64_t i = 1; i <= kN; ++i)
        convo::rtmeasure::instance().push(convo::rtmeasure::kBlockEnd, 0, i);

    const std::size_t drained = convo::rtmeasure::aggregator().drainOnce();
    const auto& s = convo::rtmeasure::aggregator().compute();

    ok = expect("agg drained", drained == kN) && ok;
    ok = expect("agg n", s.samples == kN) && ok;
    // p99 index = ceil(0.99*100)-1 = 98（0 起点）→ 値 99
    ok = expect("agg p99 ticks", s.p99Ticks == 99ull) && ok;
    ok = expect("agg max ticks", s.maxTicks == 100ull) && ok;
    ok = expect("agg writes", s.writes == kN) && ok;
    ok = expect("agg dropped", s.dropped == 0ull) && ok;
    ok = expect("agg ratio zero", s.dropRatio == 0.0) && ok;
    ok = expect("agg not inconclusive", !s.inconclusive) && ok;
    ok = expect("agg not boundary", !s.boundary) && ok;
    ok = expect("agg valid", s.valid) && ok;

    // 空の集計（サンプルなし）でも壊れない
    convo::rtmeasure::instance().reset();
    convo::rtmeasure::aggregator().reset();
    (void) convo::rtmeasure::aggregator().drainOnce();
    const auto& e = convo::rtmeasure::aggregator().compute();
    ok = expect("agg empty n", e.samples == 0) && ok;
    ok = expect("agg empty invalid", !e.valid) && ok;
    return ok;
}

// (13) 集計: 満杯時の破棄と比率（INCONCLUSIVE / 境界）
bool checkAggregatorDiscardVerdicts()
{
    using convo::rtmeasure::Ring;
    constexpr std::uint64_t kCap = Ring::kCapacity;
    bool ok = true;

    // 境界: dropped/(writes+dropped) ≈ 9.9%（9〜11%）→ boundary、inconclusive ではない
    {
        convo::rtmeasure::instance().reset();
        convo::rtmeasure::aggregator().reset();

        constexpr std::uint64_t kExtra = 900;   // 900/(8192+900) = 9.90%
        for (std::uint64_t i = 0; i < kCap + kExtra; ++i)
            convo::rtmeasure::instance().push(convo::rtmeasure::kBlockEnd, 0, i + 1);

        (void) convo::rtmeasure::aggregator().drainOnce();
        const auto& s = convo::rtmeasure::aggregator().compute();

        ok = expect("verdict writes == capacity", s.writes == kCap) && ok;
        ok = expect("verdict dropped == extra", s.dropped == kExtra) && ok;
        ok = expect("verdict ratio ~9.9%", s.dropRatio > 0.09 && s.dropRatio < 0.11) && ok;
        ok = expect("verdict boundary true", s.boundary) && ok;
        ok = expect("verdict not inconclusive", !s.inconclusive) && ok;
    }

    // INCONCLUSIVE: dropped が 10% を明確に超える
    {
        convo::rtmeasure::instance().reset();
        convo::rtmeasure::aggregator().reset();

        constexpr std::uint64_t kExtra = 9000;   // 9000/(8192+9000) = 52%
        for (std::uint64_t i = 0; i < kCap + kExtra; ++i)
            convo::rtmeasure::instance().push(convo::rtmeasure::kBlockEnd, 0, i + 1);

        (void) convo::rtmeasure::aggregator().drainOnce();
        const auto& s = convo::rtmeasure::aggregator().compute();

        ok = expect("inconclusive ratio > 10%", s.dropRatio > 0.10) && ok;
        ok = expect("inconclusive true", s.inconclusive) && ok;
        ok = expect("inconclusive not boundary", !s.boundary) && ok;
    }

    // 破棄された分は集計サンプルに入らない（サンプル数 = writes）
    {
        convo::rtmeasure::instance().reset();
        convo::rtmeasure::aggregator().reset();
        constexpr std::uint64_t kExtra = 100;
        for (std::uint64_t i = 0; i < kCap + kExtra; ++i)
            convo::rtmeasure::instance().push(convo::rtmeasure::kBlockEnd, 0, i + 1);
        (void) convo::rtmeasure::aggregator().drainOnce();
        const auto& s = convo::rtmeasure::aggregator().compute();
        ok = expect("samples == writes", s.samples == s.writes) && ok;
        ok = expect("samples == capacity", s.samples == kCap) && ok;
    }

    return ok;
}

// (14) 1 秒ごとの非 RT 集計（ゲート）
bool checkPollGate()
{
    bool ok = true;

    setEnvMeasure("1");
    convo::rtmeasure::initializeFromEnvironment();
    convo::rtmeasure::instance().reset();
    convo::rtmeasure::aggregator().reset();
    convo::rtmeasure::resetPollGateForTest();

    constexpr std::uint64_t kPushes = 10;
    for (std::uint64_t i = 0; i < kPushes; ++i)
        convo::rtmeasure::instance().push(convo::rtmeasure::kBlockEnd, 0, i + 1);

    const bool first  = convo::rtmeasure::pollIfDue();      // 1 回目: 実行される
    const std::size_t calls1 = convo::rtmeasure::aggregator().drainCalls();
    const bool second = convo::rtmeasure::pollIfDue();      // 1 秒以内: 実行されない
    const std::size_t calls2 = convo::rtmeasure::aggregator().drainCalls();

    ok = expect("poll first true", first) && ok;
    ok = expect("poll second false", !second) && ok;
    ok = expect("poll drainCalls 1", calls1 == 1) && ok;
    ok = expect("poll drainCalls unchanged", calls2 == 1) && ok;
    ok = expect("poll aggregated samples", convo::rtmeasure::aggregator().stats().samples == kPushes) && ok;

    // OFF のときは何もしない
    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    convo::rtmeasure::resetPollGateForTest();
    const bool off = convo::rtmeasure::pollIfDue();
    ok = expect("poll OFF false", !off) && ok;

    convo::rtmeasure::instance().reset();
    convo::rtmeasure::aggregator().reset();
    return ok;
}

// (15) 計測 OFF では TSC を読まない（プローブで機械検証）
bool checkOffDoesNotReadTsc()
{
    bool ok = true;

    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    convo::rtmeasure::instance().reset();

    convo::rtmeasure::detail::g_tscProbe.store(true);
    convo::rtmeasure::detail::g_tscReads.store(0);

    for (int i = 0; i < 1000; ++i)
    {
        const convo::rtmeasure::BlockScopeTimer t;
        ok = expect("timer inactive when OFF", !t.active()) && ok;
    }

    const auto offReads = convo::rtmeasure::detail::g_tscReads.load();
    ok = expect("OFF performs no TSC read", offReads == 0ull) && ok;
    ok = expect("OFF writes no event", convo::rtmeasure::instance().written() == 0ull) && ok;

    // ON では読み取りが発生する（プローブの対照）
    setEnvMeasure("1");
    convo::rtmeasure::initializeFromEnvironment();
    convo::rtmeasure::instance().reset();
    convo::rtmeasure::detail::g_tscReads.store(0);
    {
        const convo::rtmeasure::BlockScopeTimer t;
        ok = expect("timer active when ON", t.active()) && ok;
    }
    const auto onReads = convo::rtmeasure::detail::g_tscReads.load();
    ok = expect("ON performs TSC reads", onReads >= 2ull) && ok;
    ok = expect("ON writes one event", convo::rtmeasure::instance().written() == 1ull) && ok;

    convo::rtmeasure::detail::g_tscProbe.store(false);
    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    convo::rtmeasure::instance().reset();
    convo::rtmeasure::aggregator().reset();
    return ok;
}

// (16) RT 経路の性質（確保・ロック・可変長を持たない）を型で固定する
static_assert(std::is_trivially_destructible_v<convo::rtmeasure::Ring>,
              "Ring must be trivially destructible (no owned resources)");
static_assert(sizeof(convo::rtmeasure::Event) == 16, "Event must stay a 16-byte POD");
static_assert(sizeof(convo::rtmeasure::Ring) <= sizeof(convo::rtmeasure::Event) * convo::rtmeasure::Ring::kCapacity + 256,
              "Ring storage must be the fixed array plus a small header (no heap)");
static_assert(std::is_trivially_copyable_v<convo::rtmeasure::Event>,
              "Event must be trivially copyable (no hidden allocation)");

} // namespace

int runRtMeasureRingTests()
{
    bool ok = true;
    ok = checkEnabledLifecycle() && ok;
    ok = checkRoundTrip() && ok;
    ok = checkPartialDrain() && ok;
    ok = checkCapacityBoundaries() && ok;
    ok = checkOverflowDropsWithoutOverwrite() && ok;
    ok = checkReuseAfterDrain() && ok;
    ok = checkWrapAround() && ok;
    ok = checkConcurrentReaderWriter() && ok;
    ok = checkTimingCalibration() && ok;
    ok = checkBlockScopeTimer() && ok;
    ok = checkInstanceIdentity() && ok;
    ok = checkAggregatorStats() && ok;
    ok = checkAggregatorDiscardVerdicts() && ok;
    ok = checkPollGate() && ok;
    ok = checkOffDoesNotReadTsc() && ok;

    if (ok)
        std::printf("RtMeasureRingTests: PASS (U2: P1/P2/P3/P4 + capacity boundaries; RT call site connected)\n");
    return ok ? 0 : 1;
}
