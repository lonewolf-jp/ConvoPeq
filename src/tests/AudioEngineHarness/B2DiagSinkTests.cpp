// B2DiagSinkTests.cpp
// ★ B-2 (2026-11-01): 診断 sink の所有契約（DiagSink / DSPCore の attach・detach）と
//   M1〜M7 の測定・記録。合否は B-2 の契約のみで判定し、M1〜M7 は記録のみ（合否に使わない）。
//
// 実行: B2_DIAGSINK_TEST=1 AudioEngineHarness.exe
//   （既定スイートには影響しない opt-in 入口）

#include <JuceHeader.h>

#include "AudioEngine.h"

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <type_traits>

namespace {

int g_failures = 0;

void check(bool condition, const char* what)
{
    if (condition)
    {
        std::printf("[B2DS] ok   %s\n", what);
    }
    else
    {
        std::printf("[B2DS] FAIL %s\n", what);
        ++g_failures;
    }
}

const char* compilerName()
{
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(_MSC_VER)
    return "msvc";
#else
    return "unknown";
#endif
}

} // namespace

int runB2DiagSinkTests()
{
    using Buffer = LockFreeRingBuffer<DiagEvent, DiagRuntimeLimits::BufferCapacity>;

    // ---- M1〜M7（記録のみ。合否には使わない） --------------------------------
    std::printf("[B2DS] M1 sizeof(DSPCore)=%zu\n", sizeof(AudioEngine::DSPCore));
    std::printf("[B2DS] M4 is_standard_layout=%d\n",
                static_cast<int>(std::is_standard_layout_v<AudioEngine::DSPCore>));
    std::printf("[B2DS] M5 atomic<DiagSink*> is_always_lock_free=%d lock_free=%d\n",
                static_cast<int>(std::atomic<DiagSink*>::is_always_lock_free),
                static_cast<int>(std::atomic<DiagSink*>{}.is_lock_free()));
    std::printf("[B2DS] M6 CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=%d\n",
                static_cast<int>(CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS));
    std::printf("[B2DS] M7 compiler=%s msc=%d\n", compilerName(),
#if defined(_MSC_VER)
                static_cast<int>(_MSC_VER)
#else
                0
#endif
    );

    // ---- DiagSink 単体 -------------------------------------------------------
    DiagEvent event {};
    event.category = DiagCategory::EqTime;
    event.eventIndex = 1;

    DiagSink unbound;
    check(!unbound.valid(), "unbound sink is invalid");
    check(!unbound.push(event), "unbound sink push returns false");

    Buffer buffer;
    DiagPerTickCounter tickPushed {};
    DiagPerTickCounter tickDropped {};
    std::atomic<uint64_t> totalPushed { 0 };

    DiagSink sink;
    sink.buffer = &buffer;
    sink.tickPushed = &tickPushed;
    sink.tickDropped = &tickDropped;
    sink.totalPushed = &totalPushed;

    check(sink.valid(), "bound sink is valid");
    check(sink.push(event), "bound sink push succeeds");
    check(tickPushed.value.load() == 1 && totalPushed.load() == 1,
          "pushed counters updated");

    bool sawDrop = false;
    for (std::size_t i = 0; i < DiagRuntimeLimits::BufferCapacity + 8; ++i)
    {
        if (!sink.push(event))
        {
            sawDrop = true;
            break;
        }
    }
    check(sawDrop, "push returns false when the ring is full");
    check(tickDropped.value.load() >= 1, "dropped counter updated");

    // ---- DSPCore の attach / detach -----------------------------------------
    auto core = convo::aligned_make_unique<AudioEngine::DSPCore>();
    check(core->diagSink() == nullptr && core->diagOwnerId() == 0,
          "fresh core is unbound");

    core->attachDiagSink(&sink, 0);
    check(core->diagSink() == nullptr, "ownerId 0 is rejected");

    core->attachDiagSink(nullptr, 7);
    check(core->diagOwnerId() == 0, "null sink is rejected");

    core->attachDiagSink(&sink, 7);
    check(core->diagSink() == &sink && core->diagOwnerId() == 7,
          "first attach is accepted");

    DiagSink otherSink;
    core->attachDiagSink(&otherSink, 8);
    check(core->diagSink() == &sink && core->diagOwnerId() == 7,
          "attach from a different owner is rejected");

    core->attachDiagSink(&sink, 7);
    check(core->diagSink() == &sink, "same owner re-attach is idempotent");

    core->detachDiagSink();
    check(core->diagSink() == nullptr, "detach clears the sink");
    check(core->diagOwnerId() == 7, "detach keeps the owner id");

    // ---- M2 / M3（実インスタンスのアドレス差。offsetof は使わない） ----------
    const char* base = reinterpret_cast<const char*>(core.get());
    const std::size_t offOwner = static_cast<std::size_t>(
        reinterpret_cast<const char*>(&core->diagOwnerId_) - base);
    const std::size_t offSink = static_cast<std::size_t>(
        reinterpret_cast<const char*>(&core->diagSink_) - base);
    std::printf("[B2DS] M2 off_ownerId=%zu\n", offOwner);
    std::printf("[B2DS] M3 off_sink=%zu\n", offSink);
    std::printf("[B2DS] M1+M2/M3 note: offsets are measured via address difference,"
                " not offsetof\n");

    std::printf("[B2DS] RESULT failures=%d\n", g_failures);
    std::fflush(stdout);
    return g_failures == 0 ? 0 : 1;
}
