// TailLayerMultiplierTests.cpp
// BUG-013: tailL1L2Multiplier の 2 冪制約テスト
//
// 背景:
//   MKLNonUniformConvolver は L2 のパーティション長を
//     l2Part = l1Part * tailL1L2Multiplier
//   で決め、その FFT 長を 2 * l2Part にする。IPP の実数 FFT は 2 の冪長しか
//   扱えないため、倍率が 2 の冪でないと FFT プランが実長より小さく作られ、
//   L2 テールの周波数領域データが壊れる（既定値 8 では発火しない）。
//
// テスト内容:
//   1. isPowerOfTwoValue の判定
//   2. sanitizeTailL1L2Multiplier の丸め規則（近い方・同距離は大きい方）
//   3. 全域で戻り値が 2 の冪かつ [MIN, MAX] に収まる
//   4. 合成後の L2 FFT 長（2 * l1Part * 倍率）が 2 の冪になる
//
// ビルド: カスタム main() + bool testXxx() パターン（ヘッダ単体・外部依存なし）

#include <cstdio>
#include <cstdlib>

#include "TailLayerMultiplier.h"

namespace {

using convo::isPowerOfTwoValue;
using convo::sanitizeTailL1L2Multiplier;
using convo::kTailL1L2MultMin;
using convo::kTailL1L2MultMax;
using convo::kTailL1L2MultDefault;

//==============================================================================
// Test 1: isPowerOfTwoValue
//==============================================================================
[[nodiscard]] bool testIsPowerOfTwoValue()
{
    const int powers[] = { 1, 2, 4, 8, 16, 64, 1024, 4096 };
    for (const int v : powers)
    {
        if (! isPowerOfTwoValue(v))
        {
            std::printf("FAIL testIsPowerOfTwoValue: %d must be a power of two\n", v);
            return false;
        }
    }

    const int nonPowers[] = { -8, -1, 0, 3, 5, 6, 7, 12, 1000, 6144, 12288 };
    for (const int v : nonPowers)
    {
        if (isPowerOfTwoValue(v))
        {
            std::printf("FAIL testIsPowerOfTwoValue: %d must not be a power of two\n", v);
            return false;
        }
    }

    // constexpr 評価できること（コンパイル時契約）
    static_assert(isPowerOfTwoValue(1024), "1024 is a power of two");
    static_assert(! isPowerOfTwoValue(12288), "12288 is not a power of two");

    std::printf("[PASS] testIsPowerOfTwoValue\n");
    return true;
}

//==============================================================================
// Test 2: 丸め規則（近い方へ、同距離は大きい方）
//==============================================================================
[[nodiscard]] bool testSanitizeMapping()
{
    struct Case { int requested; int expected; };
    const Case cases[] = {
        { -100, kTailL1L2MultMin },  // 範囲外（下）→ MIN
        {   -1, kTailL1L2MultMin },
        {    0, kTailL1L2MultMin },
        {    1, kTailL1L2MultMin },
        {    2, 2  },                // 2 の冪はそのまま
        {    3, 4  },                // 同距離 (1,1) → 大きい方
        {    4, 4  },
        {    5, 4  },                // (1,3) → 小さい方
        {    6, 8  },                // 同距離 (2,2) → 大きい方（Air モード旧下限 6 はここで 8）
        {    7, 8  },
        {    8, 8  },                // 既定値は無変更
        {    9, 8  },
        {   10, 8  },
        {   11, 8  },
        {   12, 16 },                // 同距離 (4,4) → 大きい方
        {   13, 16 },
        {   15, 16 },
        {   16, 16 },
        {   17, kTailL1L2MultMax },  // 範囲外（上）→ MAX
        { 1000, kTailL1L2MultMax },
    };

    for (const Case& c : cases)
    {
        const int got = sanitizeTailL1L2Multiplier(c.requested);
        if (got != c.expected)
        {
            std::printf("FAIL testSanitizeMapping: sanitize(%d) = %d, expected %d\n",
                        c.requested, got, c.expected);
            return false;
        }
    }

    std::printf("[PASS] testSanitizeMapping\n");
    return true;
}

//==============================================================================
// Test 3: 全域で 2 の冪かつ範囲内
//==============================================================================
[[nodiscard]] bool testSanitizeInvariants()
{
    for (int requested = -64; requested <= 256; ++requested)
    {
        const int got = sanitizeTailL1L2Multiplier(requested);
        if (! isPowerOfTwoValue(got))
        {
            std::printf("FAIL testSanitizeInvariants: sanitize(%d) = %d is not a power of two\n",
                        requested, got);
            return false;
        }
        if (got < kTailL1L2MultMin || got > kTailL1L2MultMax)
        {
            std::printf("FAIL testSanitizeInvariants: sanitize(%d) = %d is out of range\n",
                        requested, got);
            return false;
        }
        // 冪等（丸め済みの値をもう一度通しても変わらない）
        if (sanitizeTailL1L2Multiplier(got) != got)
        {
            std::printf("FAIL testSanitizeInvariants: sanitize is not idempotent at %d\n", got);
            return false;
        }
    }

    std::printf("[PASS] testSanitizeInvariants\n");
    return true;
}

//==============================================================================
// Test 4: 合成後の L2 FFT 長が 2 の冪になる
//   l1Part は l0Part(2 の冪) * l1Mult(1/2/4) なので 2 の冪。
//   fftSize = 2 * l1Part * mult が 2 の冪であることを確認する。
//==============================================================================
[[nodiscard]] bool testComposedL2FftLengthIsPowerOfTwo()
{
    const int l0Parts[] = { 64, 128, 256, 512, 1024, 2048 };
    const int l1Mults[] = { 1, 2, 4 };

    for (const int l0Part : l0Parts)
    {
        for (const int l1Mult : l1Mults)
        {
            const int l1Part = l0Part * l1Mult;
            for (int requested = kTailL1L2MultMin; requested <= kTailL1L2MultMax; ++requested)
            {
                const int mult = sanitizeTailL1L2Multiplier(requested);
                const int l2Part = l1Part * mult;
                const int fftSize = l2Part * 2;

                if (! isPowerOfTwoValue(l2Part) || ! isPowerOfTwoValue(fftSize))
                {
                    std::printf("FAIL testComposedL2FftLengthIsPowerOfTwo: "
                                "l0Part=%d l1Mult=%d requested=%d -> l2Part=%d fftSize=%d\n",
                                l0Part, l1Mult, requested, l2Part, fftSize);
                    return false;
                }
            }
        }
    }

    // 既定値 8 は従来どおり（回帰確認）
    if (kTailL1L2MultDefault != 8 || sanitizeTailL1L2Multiplier(8) != 8)
    {
        std::printf("FAIL testComposedL2FftLengthIsPowerOfTwo: default must stay 8\n");
        return false;
    }

    std::printf("[PASS] testComposedL2FftLengthIsPowerOfTwo\n");
    return true;
}

} // anonymous namespace

//==============================================================================
// main
//==============================================================================
int main()
{
    bool allPassed = true;

    allPassed &= testIsPowerOfTwoValue();
    allPassed &= testSanitizeMapping();
    allPassed &= testSanitizeInvariants();
    allPassed &= testComposedL2FftLengthIsPowerOfTwo();

    if (allPassed)
    {
        std::printf("\n=== ALL TESTS PASSED ===\n");
        return EXIT_SUCCESS;
    }

    std::printf("\n=== SOME TESTS FAILED ===\n");
    return EXIT_FAILURE;
}
