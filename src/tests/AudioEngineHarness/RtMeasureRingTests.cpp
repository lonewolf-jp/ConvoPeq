//==============================================================================
// R-06 / U2 + U5 準備（A1'）: `convo::rtmeasure` の検証
//
//   Owner 指示（2026-10-31）の 12 項目を検証する。
//     1. モード 0 で TSC 取得 0 回
//     2. モード 2 で TSC 取得あり・リング push 0 回
//     3. モード 1 で TSC 取得あり・リング push あり
//     4. モード 1・2 で共通サンプル保存が同じ方式
//     5. 終了 TSC がリング push より後に取得される
//     6. ウォームアップ 2,000 サンプルが統計から除外される
//     7. 32,000 ブロックで 30,000 以上の有効サンプル
//     8. サンプル領域の容量超過が INCONCLUSIVE
//     9. p99 整数演算・最大値・dropped 率・RERUN 判定
//    10. RT 経路に確保・ロック・I/O・書式化・ループ・RMW がない（型とソースで固定）
//    11. 新しいサンプル保存経路の競合（TSan 実行で確認）
//    12. 既存スイート（リング機構）
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
        std::fprintf(stderr, "FAIL: RT-MEASURE-U5P %s\n", name);
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

void resetState()
{
    convo::rtmeasure::instance().reset();
    convo::rtmeasure::sampleBuffer().reset();
    convo::rtmeasure::aggregator().setWarmupSamples(2000);
    convo::rtmeasure::aggregator().reset();
    convo::rtmeasure::detail::g_tscReads.store(0);
    convo::rtmeasure::detail::g_order.store(0);
}

// (1) モード 0: TSC を読まない・リングもサンプルも増えない
bool checkModeZeroNoTsc()
{
    bool ok = true;
    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    resetState();

    convo::rtmeasure::detail::g_tscProbe.store(true);
    for (int i = 0; i < 1000; ++i)
    {
        const convo::rtmeasure::BlockScopeTimer t;
        ok = expect("mode0 timer inactive", !t.active()) && ok;
    }
    const auto reads = convo::rtmeasure::detail::g_tscReads.load();
    convo::rtmeasure::detail::g_tscProbe.store(false);

    ok = expect("mode0 reads no TSC", reads == 0ull) && ok;
    ok = expect("mode0 no ring push", convo::rtmeasure::instance().written() == 0ull) && ok;
    ok = expect("mode0 no samples", convo::rtmeasure::sampleBuffer().count() == 0u) && ok;
    return ok;
}

// (2) モード 2: TSC あり・リング push なし・サンプルは 1/ブロック
bool checkModeTimeOnly()
{
    bool ok = true;
    setEnvMeasure("2");
    convo::rtmeasure::initializeFromEnvironment();
    resetState();

    ok = expect("mode2 is TimeOnly", convo::rtmeasure::mode() == convo::rtmeasure::Mode::TimeOnly) && ok;

    convo::rtmeasure::detail::g_tscProbe.store(true);
    constexpr int kBlocks = 100;
    for (int i = 0; i < kBlocks; ++i)
    {
        const convo::rtmeasure::BlockScopeTimer t;
        ok = expect("mode2 timer active", t.active()) && ok;
        ok = expect("mode2 recordsRing false", !t.recordsRing()) && ok;
    }
    const auto reads = convo::rtmeasure::detail::g_tscReads.load();
    convo::rtmeasure::detail::g_tscProbe.store(false);

    ok = expect("mode2 reads TSC (>=2/block)", reads >= static_cast<std::uint64_t>(2 * kBlocks)) && ok;
    ok = expect("mode2 no ring push", convo::rtmeasure::instance().written() == 0ull) && ok;
    ok = expect("mode2 samples == blocks",
                convo::rtmeasure::sampleBuffer().count() == static_cast<std::size_t>(kBlocks)) && ok;
    ok = expect("mode2 no sample overflow", convo::rtmeasure::sampleBuffer().overflowed() == 0ull) && ok;
    return ok;
}

// (3) モード 1: TSC あり・リング push あり・サンプルは 1/ブロック
bool checkModeFull()
{
    bool ok = true;
    setEnvMeasure("1");
    convo::rtmeasure::initializeFromEnvironment();
    resetState();

    ok = expect("mode1 is Full", convo::rtmeasure::mode() == convo::rtmeasure::Mode::Full) && ok;

    convo::rtmeasure::detail::g_tscProbe.store(true);
    constexpr int kBlocks = 100;
    for (int i = 0; i < kBlocks; ++i)
    {
        const convo::rtmeasure::BlockScopeTimer t;
        ok = expect("mode1 timer active", t.active()) && ok;
        ok = expect("mode1 recordsRing true", t.recordsRing()) && ok;
    }
    const auto reads = convo::rtmeasure::detail::g_tscReads.load();
    convo::rtmeasure::detail::g_tscProbe.store(false);

    ok = expect("mode1 reads TSC", reads >= static_cast<std::uint64_t>(2 * kBlocks)) && ok;
    ok = expect("mode1 ring push per block",
                convo::rtmeasure::instance().written() == static_cast<std::uint64_t>(kBlocks)) && ok;
    ok = expect("mode1 samples == blocks",
                convo::rtmeasure::sampleBuffer().count() == static_cast<std::size_t>(kBlocks)) && ok;
    return ok;
}

// (4) モード 1・2 で共通サンプル保存が同じ方式（同じ個数・同じ上限・同じ overflow 挙動）
bool checkSameSampleStorage()
{
    bool ok = true;

    setEnvMeasure("2");
    convo::rtmeasure::initializeFromEnvironment();
    resetState();
    {
        convo::rtmeasure::BlockScopeTimer a;
        convo::rtmeasure::BlockScopeTimer b;
        (void) a;
        (void) b;
    }
    const auto countTimeOnly = convo::rtmeasure::sampleBuffer().count();

    setEnvMeasure("1");
    convo::rtmeasure::initializeFromEnvironment();
    resetState();
    {
        convo::rtmeasure::BlockScopeTimer a;
        convo::rtmeasure::BlockScopeTimer b;
        (void) a;
        (void) b;
    }
    const auto countFull = convo::rtmeasure::sampleBuffer().count();

    ok = expect("same storage both modes", countTimeOnly == 2u && countFull == 2u) && ok;
    ok = expect("same capacity", convo::rtmeasure::SampleBuffer::kCapacity == (1u << 16)) && ok;

    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    resetState();
    return ok;
}

// (5) 終了 TSC はリング push の後（順序プローブで確認）
bool checkEndTscAfterPush()
{
    bool ok = true;
    setEnvMeasure("1");
    convo::rtmeasure::initializeFromEnvironment();
    resetState();

    convo::rtmeasure::detail::g_orderProbe.store(true);
    convo::rtmeasure::detail::g_order.store(0);
    {
        convo::rtmeasure::BlockScopeTimer t;
        (void) t;
    }
    const int order = convo::rtmeasure::detail::g_order.load();
    convo::rtmeasure::detail::g_orderProbe.store(false);

    ok = expect("order probe reached 3 (end after push)", order == 3) && ok;
    ok = expect("push happened (ring)", convo::rtmeasure::instance().written() == 1ull) && ok;
    ok = expect("sample stored", convo::rtmeasure::sampleBuffer().count() == 1u) && ok;

    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    resetState();
    return ok;
}

// (6) ウォームアップ 2,000 の除外
bool checkWarmupExclusion()
{
    bool ok = true;
    auto& sb = convo::rtmeasure::sampleBuffer();
    auto& agg = convo::rtmeasure::aggregator();

    resetState();
    agg.setWarmupSamples(2000);
    agg.reset();

    constexpr std::size_t kTotal = 2005;
    for (std::size_t i = 0; i < kTotal; ++i)
        sb.push(1);

    (void) agg.collect();
    const auto& s = agg.compute();

    ok = expect("warmup skipped 2000", s.warmupSkipped == 2000ull) && ok;
    ok = expect("warmup leaves 5 samples", s.samples == 5ull) && ok;
    ok = expect("warmup valid", s.valid) && ok;
    return ok;
}

// (7) 32,000 ブロックで 30,000 以上の有効サンプル
bool checkThirtyTwoThousandBlocks()
{
    bool ok = true;
    auto& sb = convo::rtmeasure::sampleBuffer();
    auto& agg = convo::rtmeasure::aggregator();

    resetState();
    agg.setWarmupSamples(2000);
    agg.reset();

    constexpr std::size_t kBlocks = 32000;
    for (std::size_t i = 0; i < kBlocks; ++i)
        sb.push(100 + i);

    (void) agg.collect();
    const auto& s = agg.compute();

    ok = expect("32k warmup skipped", s.warmupSkipped == 2000ull) && ok;
    ok = expect("32k valid samples >= 30000", s.samples >= 30000ull) && ok;
    ok = expect("32k no overflow", s.sampleOverflow == 0ull && !s.inconclusive) && ok;
    return ok;
}

// (8) サンプル領域の容量超過は INCONCLUSIVE
bool checkSampleOverflowInconclusive()
{
    bool ok = true;
    auto& sb = convo::rtmeasure::sampleBuffer();
    auto& agg = convo::rtmeasure::aggregator();

    resetState();
    agg.setWarmupSamples(0);
    agg.reset();

    const std::size_t over = convo::rtmeasure::SampleBuffer::kCapacity + 5;
    for (std::size_t i = 0; i < over; ++i)
        sb.push(1);

    (void) agg.collect();
    const auto& s = agg.compute();

    ok = expect("overflow counted", s.sampleOverflow == 5ull) && ok;
    ok = expect("overflow => inconclusive", s.inconclusive) && ok;
    ok = expect("overflow samples capped", s.samples == convo::rtmeasure::SampleBuffer::kCapacity) && ok;

    agg.setWarmupSamples(2000);
    resetState();
    return ok;
}

// (9) p99 整数演算・最大値・N
bool checkP99AndMax()
{
    bool ok = true;
    auto& sb = convo::rtmeasure::sampleBuffer();
    auto& agg = convo::rtmeasure::aggregator();

    resetState();
    agg.setWarmupSamples(0);
    agg.reset();

    for (std::uint64_t i = 1; i <= 100; ++i)
        sb.push(i);

    (void) agg.collect();
    const auto& s = agg.compute();

    ok = expect("p99 n", s.samples == 100ull) && ok;
    ok = expect("p99 ticks (ceil(99)-1 = idx98)", s.p99Ticks == 99ull) && ok;
    ok = expect("p99 max", s.maxTicks == 100ull) && ok;
    ok = expect("p99 ratio zero", s.dropRatio == 0.0) && ok;
    ok = expect("p99 not boundary", !s.boundary) && ok;

    // N=1 の境界
    resetState();
    agg.setWarmupSamples(0);
    agg.reset();
    sb.push(42);
    (void) agg.collect();
    const auto& one = agg.compute();
    ok = expect("p99 n=1", one.samples == 1ull && one.p99Ticks == 42ull && one.maxTicks == 42ull) && ok;

    agg.setWarmupSamples(2000);
    resetState();
    return ok;
}

// (9b) リング破棄率による INCONCLUSIVE / RERUN
bool checkRingRatioVerdicts()
{
    using convo::rtmeasure::Ring;
    constexpr std::uint64_t kCap = Ring::kCapacity;
    bool ok = true;
    auto& agg = convo::rtmeasure::aggregator();

    // 境界（9.9%）
    resetState();
    agg.setWarmupSamples(0);
    agg.reset();
    convo::rtmeasure::instance().reset();
    {
        constexpr std::uint64_t kExtra = 900;
        for (std::uint64_t i = 0; i < kCap + kExtra; ++i)
            convo::rtmeasure::instance().push(convo::rtmeasure::kBlockEnd, 0, 0);
        const auto& s = agg.compute();
        ok = expect("ring ratio ~9.9%", s.dropRatio > 0.09 && s.dropRatio < 0.11) && ok;
        ok = expect("ring boundary true", s.boundary) && ok;
        ok = expect("ring not inconclusive", !s.inconclusive) && ok;
    }

    // 10% 超
    convo::rtmeasure::instance().reset();
    {
        constexpr std::uint64_t kExtra = 9000;
        for (std::uint64_t i = 0; i < kCap + kExtra; ++i)
            convo::rtmeasure::instance().push(convo::rtmeasure::kBlockEnd, 0, 0);
        const auto& s = agg.compute();
        ok = expect("ring ratio >10%", s.dropRatio > 0.10) && ok;
        ok = expect("ring inconclusive", s.inconclusive) && ok;
        ok = expect("ring not boundary", !s.boundary) && ok;
    }

    agg.setWarmupSamples(2000);
    resetState();
    return ok;
}

// (10) RT 経路の性質を型で固定する
static_assert(std::is_trivially_destructible_v<convo::rtmeasure::Ring>,
              "Ring must be trivially destructible");
static_assert(std::is_trivially_destructible_v<convo::rtmeasure::SampleBuffer>,
              "SampleBuffer must be trivially destructible (no owned resources)");
static_assert(sizeof(convo::rtmeasure::Event) == 16, "Event must stay a 16-byte POD");
static_assert(sizeof(convo::rtmeasure::SampleBuffer)
                  <= sizeof(std::uint64_t) * convo::rtmeasure::SampleBuffer::kCapacity + 256,
              "SampleBuffer must be the fixed array plus a small header (no heap)");
static_assert(std::is_trivially_copyable_v<convo::rtmeasure::Event>, "Event must be trivially copyable");

// (12) リング機構（U2 から継続）
bool checkRingMechanics()
{
    using convo::rtmeasure::Ring;
    bool ok = true;

    Ring ring;
    ring.reset();

    constexpr std::uint32_t kKinds[] = { convo::rtmeasure::kBlockEnd, convo::rtmeasure::kAlloc,
                                         convo::rtmeasure::kLockWait, convo::rtmeasure::kFree };
    for (std::uint32_t i = 0; i < 4; ++i)
        ring.push(kKinds[i], i * 10u, 1000ull + i);

    std::vector<convo::rtmeasure::Event> out(8);
    const std::size_t n = ring.drain(out.data(), out.size());
    ok = expect("ring roundtrip count", n == 4) && ok;
    ok = expect("ring written", ring.written() == 4) && ok;
    ok = expect("ring pending zero", ring.pending() == 0) && ok;

    // 満杯時の破棄（上書きなし）
    ring.reset();
    constexpr std::uint64_t kCap = Ring::kCapacity;
    for (std::uint64_t i = 0; i < kCap + 10; ++i)
        ring.push(convo::rtmeasure::kBlockEnd, 0, i);
    ok = expect("ring cap written", ring.written() == kCap) && ok;
    ok = expect("ring dropped 10", ring.dropped() == 10ull) && ok;
    {
        std::vector<convo::rtmeasure::Event> small(4);
        const std::size_t drained = ring.drain(small.data(), small.size());
        ok = expect("ring keeps oldest", drained == 4 && small[0].tsc == 0ull) && ok;
    }

    return ok;
}

// (11) サンプル保存経路の並行性（TSan 実行で競合が無いこと）
bool checkSampleBufferConcurrency()
{
    bool ok = true;
    auto& sb = convo::rtmeasure::sampleBuffer();
    auto& agg = convo::rtmeasure::aggregator();

    resetState();
    agg.setWarmupSamples(0);
    agg.reset();

    constexpr std::uint64_t kPushes = 20000;
    std::atomic<bool> stop{ false };
    std::size_t collected = 0;

    std::thread reader([&] {
        while (!stop.load(std::memory_order_acquire))
        {
            collected += agg.collect();
            if (agg.lastCollected() == 0)
                std::this_thread::yield();
        }
        collected += agg.collect();
    });

    for (std::uint64_t i = 0; i < kPushes; ++i)
        sb.push(i);

    stop.store(true, std::memory_order_release);
    reader.join();
    collected += agg.collect();

    ok = expect("concurrent collected == pushed", collected == kPushes) && ok;
    ok = expect("concurrent buffer count", sb.count() == kPushes) && ok;
    ok = expect("concurrent no overflow", sb.overflowed() == 0ull) && ok;

    agg.setWarmupSamples(2000);
    resetState();
    return ok;
}

// (1') モード判定のライフサイクル
bool checkModeLifecycle()
{
    bool ok = true;

    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    ok = expect("env unset -> Off", convo::rtmeasure::mode() == convo::rtmeasure::Mode::Off) && ok;

    setEnvMeasure("0");
    convo::rtmeasure::initializeFromEnvironment();
    ok = expect("env 0 -> Off", convo::rtmeasure::mode() == convo::rtmeasure::Mode::Off) && ok;

    setEnvMeasure("2");
    convo::rtmeasure::initializeFromEnvironment();
    ok = expect("env 2 -> requested TimeOnly",
                convo::rtmeasure::requestedMode() == convo::rtmeasure::Mode::TimeOnly) && ok;
    ok = expect("env 2 -> TimeOnly", convo::rtmeasure::mode() == convo::rtmeasure::Mode::TimeOnly) && ok;

    setEnvMeasure("1");
    convo::rtmeasure::initializeFromEnvironment();
    ok = expect("env 1 -> requested Full",
                convo::rtmeasure::requestedMode() == convo::rtmeasure::Mode::Full) && ok;
    ok = expect("env 1 -> Full", convo::rtmeasure::mode() == convo::rtmeasure::Mode::Full) && ok;

    setEnvMeasure(nullptr);
    convo::rtmeasure::initializeFromEnvironment();
    ok = expect("restored Off", convo::rtmeasure::mode() == convo::rtmeasure::Mode::Off) && ok;

    resetState();
    return ok;
}

} // namespace

int runRtMeasureRingTests()
{
    bool ok = true;

    // サニタイザ下では TSC 校正の 1% 一致検査が不安定になるため、テストでは校正成功として扱う
    // （本番は実校正。失敗時は Off に落ちる fail-closed のまま）。
    convo::rtmeasure::detail::g_forceTimingOkForTest.store(true);

    ok = checkModeLifecycle() && ok;
    ok = checkModeZeroNoTsc() && ok;
    ok = checkModeTimeOnly() && ok;
    ok = checkModeFull() && ok;
    ok = checkSameSampleStorage() && ok;
    ok = checkEndTscAfterPush() && ok;
    ok = checkWarmupExclusion() && ok;
    ok = checkThirtyTwoThousandBlocks() && ok;
    ok = checkSampleOverflowInconclusive() && ok;
    ok = checkP99AndMax() && ok;
    ok = checkRingRatioVerdicts() && ok;
    ok = checkRingMechanics() && ok;
    ok = checkSampleBufferConcurrency() && ok;

    if (ok)
        std::printf("RtMeasureRingTests: PASS (U5 prep A1': modes 0/2/1, common samples, warmup, overflow)\n");
    return ok ? 0 : 1;
}
