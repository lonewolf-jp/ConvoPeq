//============================================================================
// CorruptSampleGuardTests.cpp
//
// RES2-02 (O-20): the scalar guard and the AVX2 guard used different limits
// (2^53 vs 1e20). These checks pin the shared limit and require both paths to
// agree on the same samples.
//============================================================================
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>

#include "CorruptSampleGuard.h"

namespace
{
    using convo::numeric_policy::isCorruptSample;

    bool checkLimitConstant()
    {
        if (convo::numeric_policy::kCorruptMagnitudeBits != 0x4340000000000000ULL)
        {
            std::printf("FAIL limit: kCorruptMagnitudeBits is not 2^53\n");
            return false;
        }
        if (convo::numeric_policy::kCorruptMagnitudeLimit != 9007199254740992.0)
        {
            std::printf("FAIL limit: kCorruptMagnitudeLimit is not 2^53\n");
            return false;
        }
        std::printf("[PASS] limit constant is 2^53\n");
        return true;
    }

    struct Case
    {
        double value;
        bool expectCorrupt;
        const char* label;
    };

    std::vector<Case> makeCases()
    {
        constexpr double two53 = 9007199254740992.0;
        return {
            { 0.0, false, "0" },
            { 1.0, false, "1" },
            { -1.0, false, "-1" },
            { two53, false, "2^53" },
            { -two53, false, "-2^53" },
            { std::nextafter(two53, std::numeric_limits<double>::infinity()), true, "2^53+1ulp" },
            { std::nextafter(-two53, -std::numeric_limits<double>::infinity()), true, "-2^53-1ulp" },
            { 1.0e20, true, "1e20" },
            { 1.0e300, true, "1e300" },
            { std::numeric_limits<double>::infinity(), true, "+inf" },
            { -std::numeric_limits<double>::infinity(), true, "-inf" },
            { std::numeric_limits<double>::quiet_NaN(), true, "NaN" },
        };
    }

    bool checkScalarGuard()
    {
        bool ok = true;
        for (const auto& c : makeCases())
        {
            const bool got = isCorruptSample(c.value);
            if (got != c.expectCorrupt)
            {
                std::printf("FAIL scalar: %s -> %d (expected %d)\n", c.label, got ? 1 : 0, c.expectCorrupt ? 1 : 0);
                ok = false;
            }
        }
        if (ok)
            std::printf("[PASS] scalar guard matches the 2^53 boundary\n");
        return ok;
    }

#if defined(__AVX2__)
    bool checkAvx2GuardAgrees()
    {
        const auto cases = makeCases();
        bool ok = true;
        for (size_t base = 0; base < cases.size(); base += 4)
        {
            alignas(32) double lanes[4] = { 0.0, 0.0, 0.0, 0.0 };
            for (size_t lane = 0; lane < 4 && base + lane < cases.size(); ++lane)
                lanes[lane] = cases[base + lane].value;

            const __m256d v = _mm256_load_pd(lanes);
            const int mask = _mm256_movemask_pd(convo::numeric_policy::corruptSampleMask4(v));
            for (size_t lane = 0; lane < 4 && base + lane < cases.size(); ++lane)
            {
                const bool laneCorrupt = (mask & (1 << lane)) != 0;
                const bool scalar = isCorruptSample(cases[base + lane].value);
                if (laneCorrupt != scalar)
                {
                    std::printf("FAIL avx2: %s lane mask=%d scalar=%d\n",
                                cases[base + lane].label, laneCorrupt ? 1 : 0, scalar ? 1 : 0);
                    ok = false;
                }
            }
        }
        if (ok)
            std::printf("[PASS] AVX2 guard agrees with the scalar guard\n");
        return ok;
    }
#else
    bool checkAvx2GuardAgrees()
    {
        std::printf("[SKIP] AVX2 guard: built without __AVX2__\n");
        return true;
    }
#endif
} // namespace

int main()
{
    bool allPassed = true;
    allPassed &= checkLimitConstant();
    allPassed &= checkScalarGuard();
    allPassed &= checkAvx2GuardAgrees();

    if (allPassed)
    {
        std::printf("CorruptSampleGuardTests: PASS\n");
        return 0;
    }
    std::printf("CorruptSampleGuardTests: FAIL\n");
    return 1;
}
