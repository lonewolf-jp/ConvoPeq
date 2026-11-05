//============================================================================
// RtMeasureOverheadTests.cpp — RT cost of the U5 measurement instrument
//
// Test-only micro-benchmark. Reads CONVOPEQ_RT_MEASURE (0/unset, 2, 1) once,
// then measures the cost of one BlockScopeTimer scope around a trivial body.
// Run the executable three times with different environment values and compare
// the ns/op. No pass/fail threshold is applied: the numbers are evidence for
// the U5 activation decision.
//============================================================================
#include "audioengine/RtMeasureRing.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

int main()
{
    convo::rtmeasure::initializeFromEnvironment();
    const int mode = static_cast<int>(convo::rtmeasure::mode());

    constexpr int kIterations = 200000;
    volatile std::uint64_t sink = 0;

    // Warm up the measurement path itself.
    for (int i = 0; i < 2000; ++i)
    {
        convo::rtmeasure::BlockScopeTimer scope;
        sink = sink + 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; ++i)
    {
        convo::rtmeasure::BlockScopeTimer scope;
        sink = sink + 1;
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double nsPerOp =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(kIterations);
    std::printf("[RTOVERHEAD] mode=%d iterations=%d nsPerOp=%.3f sink=%llu\n",
                mode, kIterations, nsPerOp, static_cast<unsigned long long>(sink));
    return 0;
}
