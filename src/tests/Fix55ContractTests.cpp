//==============================================================================
// Fix55ContractTests.cpp
//
// C-ID 登録簿（doc/work113/20261121-1201_詳細検証）のうち、外部依存なしで
// 決定的に検証できる契約を固定する。JUCE / MKL には依存しない。
//
//   C-20  tailL1L2Multiplier は 2 の冪へ丸める（保存値＝実効値）
//   C-29  計測レジストリの初回集計はタグ公開後に ready になる（空タグで ready にならない）
//   C-36  equal-power 近似は 1 を超えない
//   C-37  SPSC リングは容量 0 を許さない（コンパイル時契約）
//==============================================================================

#include <cstdio>
#include <cstring>
#include <string>

#include "TailLayerMultiplier.h"
#include "core/FadeEngine.h"
#include "core/CommandBuffer.h"
#include "core/ReaderMeasureRegistry.h"

namespace
{
int g_failures = 0;

void check (bool condition, const char* what)
{
    if (! condition)
    {
        std::printf ("FAIL: %s\n", what);
        ++g_failures;
    }
    else
    {
        std::printf ("PASS: %s\n", what);
    }
}

// ---- C-20: 丸め結果は必ず [2,16] の 2 の冪で、要求値に最も近い ----
void checkTailMultiplierRounding()
{
    const int inputs[] = { -100, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 1000 };

    bool allPowerOfTwo = true;
    bool allInRange = true;
    for (int v : inputs)
    {
        const int r = convo::sanitizeTailL1L2Multiplier (v);
        if (! convo::isPowerOfTwoValue (r))
            allPowerOfTwo = false;
        if (r < convo::kTailL1L2MultMin || r > convo::kTailL1L2MultMax)
            allInRange = false;
    }

    check (allPowerOfTwo, "C-20 sanitizeTailL1L2Multiplier always returns a power of two");
    check (allInRange, "C-20 sanitizeTailL1L2Multiplier stays inside [kMin,kMax]");
    check (convo::sanitizeTailL1L2Multiplier (3) == 4, "C-20 3 rounds to 4 (equidistant -> larger)");
    check (convo::sanitizeTailL1L2Multiplier (5) == 4, "C-20 5 rounds to 4");
    check (convo::sanitizeTailL1L2Multiplier (7) == 8, "C-20 7 rounds to 8");
    check (convo::sanitizeTailL1L2Multiplier (2) == 2, "C-20 2 is already valid");
}

// ---- C-29: 集計スロットはタグを書いてから used=1 になる ----
void checkAggregateTagPublication()
{
    using Registry = convo::ReaderMeasureRegistry;

    const char* tag = "fix55-c29";
    for (int round = 0; round < 8; ++round)
    {
        const int idx = Registry::claim (tag, 0u, 1u, nullptr, nullptr);
        if (idx < 0)
        {
            check (false, "C-29 claim() returned a record");
            return;
        }
        Registry::close (idx);
    }

    bool foundTagged = false;
    bool readyWithEmptyTag = false;
    for (int i = 0; i < Registry::kTags; ++i)
    {
        if (convo::consumeAtomic (Registry::aggs_[i].used, std::memory_order_acquire) != 1)
            continue;

        const char* aggregateTag = Registry::aggs_[i].tag;
        if (aggregateTag[0] == '\0')
            readyWithEmptyTag = true;
        if (std::strncmp (aggregateTag, tag, sizeof (Registry::aggs_[i].tag)) == 0)
            foundTagged = true;
    }

    check (! readyWithEmptyTag, "C-29 no aggregate is published with an empty tag");
    check (foundTagged, "C-29 the tag is visible once the aggregate is published");
}

// ---- C-36: equal-power 近似は [0,1] ----
void checkEqualPowerRange()
{
    bool inRange = true;
    float previous = -1.0f;
    bool monotonic = true;
    for (int i = 0; i <= 1000; ++i)
    {
        const float x = static_cast<float> (i) / 1000.0f;
        const float y = convo::FadeEngine::equalPowerSinApprox (x);
        if (y < 0.0f || y > 1.0f)
            inRange = false;
        if (y + 1.0e-6f < previous)
            monotonic = false;
        previous = y;
    }

    check (inRange, "C-36 equalPowerSinApprox stays in [0,1]");
    check (monotonic, "C-36 equalPowerSinApprox is monotonic");
    check (convo::FadeEngine::equalPowerSinApprox (1.0f) <= 1.0f,
           "C-36 x=1 does not exceed unity");
    check (convo::FadeEngine::equalPowerSinApprox (0.0f) == 0.0f,
           "C-36 x=0 is zero");
}

// ---- C-37: SPSC リングの基本動作（容量 0 は static_assert で拒否される） ----
void checkSpscRing()
{
    convo::SPSCRingBuffer<int, 4> ring;
    int value = 0;

    check (ring.push (1) && ring.push (2) && ring.push (3) && ring.push (4),
           "C-37 SPSC ring accepts Capacity items");
    check (! ring.push (5), "C-37 SPSC ring rejects the item after Capacity");

    const bool popped = ring.pop (value) && value == 1;
    check (popped, "C-37 SPSC ring preserves FIFO order");
    check (ring.push (5), "C-37 SPSC ring accepts after a pop");
}

} // namespace

int main()
{
    std::printf ("== Fix55ContractTests ==\n");
    checkTailMultiplierRounding();
    checkAggregateTagPublication();
    checkEqualPowerRange();
    checkSpscRing();

    if (g_failures != 0)
    {
        std::printf ("FAILED: %d check(s)\n", g_failures);
        return 1;
    }

    std::printf ("All Fix55 contract checks passed.\n");
    return 0;
}
