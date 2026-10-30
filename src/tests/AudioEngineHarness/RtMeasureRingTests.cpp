//==============================================================================
// R-06 / U1: `convo::rtmeasure::Ring` の機械検証（非 RT・オフライン）
//
//   U1 は「RT から書き込めるリングを追加するだけ」で、呼び出し点を持たない。
//   したがって検証できるのはリングの機械的な性質だけである。ここでは次を確かめる。
//
//     (1) push -> drain で順序と内容が保たれる
//     (2) drain は n 件までで、残りは次回に残る
//     (3) 満杯時は**上書きせず破棄**し、dropped に計上する（測定値が下限であることの根拠）
//     (4) written / consumed / pending の整合
//     (5) reset() で 0 に戻る
//     (6) allowed()（環境変数の判定）が意図どおり
//
//   RT 文脈は使わない。単一スレッドで決定的に回す。
//==============================================================================

#include "audioengine/RtMeasureRing.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

bool expect(const char* name, bool ok)
{
    if (!ok)
        std::fprintf(stderr, "FAIL: RT-MEASURE-U1 %s\n", name);
    return ok;
}

bool checkRoundTrip()
{
    convo::rtmeasure::Ring ring;
    ring.reset();

    constexpr std::uint32_t kKinds[] = {
        convo::rtmeasure::kBlockBegin, convo::rtmeasure::kAlloc,
        convo::rtmeasure::kLockWait,   convo::rtmeasure::kBlockEnd,
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

bool checkOverflowDropsWithoutOverwrite()
{
    convo::rtmeasure::Ring ring;
    ring.reset();

    constexpr std::size_t kExtra = 10;
    const std::size_t total = convo::rtmeasure::Ring::kCapacity + kExtra;
    for (std::size_t i = 0; i < total; ++i)
        ring.push(convo::rtmeasure::kFree, static_cast<std::uint32_t>(i), i);

    bool ok = true;
    // written() は「リングに入った件数」。破棄分は dropped() に計上され、
    // 両者の和が push 回数に等しくなる（取りこぼしの可視化）。
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

bool checkResetAndInstance()
{
    convo::rtmeasure::Ring ring;
    ring.reset();
    ring.push(convo::rtmeasure::kBlockBegin, 512, 42);
    ring.drain(nullptr, 0); // no-op
    ring.reset();

    bool ok = true;
    ok = expect("reset written", ring.written() == 0) && ok;
    ok = expect("reset consumed", ring.consumed() == 0) && ok;
    ok = expect("reset dropped", ring.dropped() == 0) && ok;
    ok = expect("reset pending", ring.pending() == 0) && ok;
    ok = expect("instance stable", &convo::rtmeasure::Ring::instance() == &convo::rtmeasure::Ring::instance())
         && ok;
    return ok;
}

bool checkEnabledFlag()
{
    // 既定（未設定）は OFF。設定したら ON。判定は 1 回だけなので、
    // ここでは「プロセス起動時の状態」を壊さないよう、現在の値を読むだけにする。
    const char* v = std::getenv("CONVOPEQ_RT_MEASURE");
    const bool raw = (v != nullptr && v[0] == '1');
    const bool cached = convo::rtmeasure::enabled();

    bool ok = true;
    // 最初の enabled() 呼び出しより前に環境が変わっていなければ一致する。
    // テストはこの関数を最初に呼ぶため、通常は raw == cached になる。
    if (!raw && !cached)
        ok = expect("enabled default OFF", true) && ok;
    else
        ok = expect("enabled consistent with env", raw == cached) && ok;

    std::printf("[RT-M] U1 ring: enabled=%d capacity=%zu\n",
                cached ? 1 : 0,
                convo::rtmeasure::Ring::kCapacity);
    return ok;
}

} // namespace

int runRtMeasureRingTests()
{
    bool ok = true;
    ok = checkEnabledFlag() && ok;
    ok = checkRoundTrip() && ok;
    ok = checkPartialDrain() && ok;
    ok = checkOverflowDropsWithoutOverwrite() && ok;
    ok = checkResetAndInstance() && ok;

    if (ok)
        std::printf("RtMeasureRingTests: PASS (U1 ring mechanics only; no RT call sites)\n");
    return ok ? 0 : 1;
}
