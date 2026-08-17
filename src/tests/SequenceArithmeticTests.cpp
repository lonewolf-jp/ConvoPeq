//==============================================================================
// SequenceArithmeticTests.cpp — dash2 §1.6.1 (Phase H) modular sequence arithmetic テスト
//
// テスト対象: src/audioengine/SequenceArithmetic.h（convo::isr 名前空間・ヘッダオンリー）
//
// ■ テスト項目（REPAIR_PLAN2-dash2.md §1.6 実装手順 3 の対応）:
//   1. 正常系（non-wrap）: 10→11→12 の isBefore/isAfter/isCompleted 挙動
//   2. out-of-order（将来 sparse completion §1.5 前提）: 11→10 の比較・completed 判定
//   3. duplicate: 10→10（等値は before でも after でもない / completed は true）
//   4. wraparound: UINT64_MAX 近傍（UINT64_MAX-1 → UINT64_MAX → 0 → 1）
//   5. antipode（境界）: dist == 2^63 は before でも after でもない（曖昧）
//   6. seqDistance: wrap を含む modular distance の正しさ
//
// ■ ビルド:
//   CMakeLists.txt に以下を追加（MpscBoundedRingTests と同一パターン）:
//     add_executable(SequenceArithmeticTests
//         src/tests/SequenceArithmeticTests.cpp
//     )
//     add_test(NAME SequenceArithmeticTests COMMAND SequenceArithmeticTests)
//     target_compile_features(SequenceArithmeticTests PRIVATE cxx_std_20)
//     target_compile_options(SequenceArithmeticTests PRIVATE /EHsc /utf-8)
//     target_include_directories(SequenceArithmeticTests PRIVATE
//         ${CMAKE_CURRENT_SOURCE_DIR}
//         ${CMAKE_CURRENT_SOURCE_DIR}/src
//         ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
//         ${CMAKE_CURRENT_SOURCE_DIR}/src/core
//     )
//
//==============================================================================

#include "SequenceArithmetic.h" // テスト対象本体（ヘッダオンリー）

#include <cstdint>
#include <iostream>
#include <limits>

using convo::isr::isAfter;
using convo::isr::isAtOrBefore;
using convo::isr::isBefore;
using convo::isr::isCompleted;
using convo::isr::kSeqHalfModulus;
using convo::isr::seqDistance;

namespace {

int g_failCount = 0;
int g_testCount = 0;

void check(const char* name, bool condition)
{
    ++g_testCount;
    if (condition)
        std::cout << "  PASS: " << name << std::endl;
    else
    {
        std::cout << "  FAIL: " << name << std::endl;
        ++g_failCount;
    }
}

// --- 1. 正常系（non-wrap）: 10 → 11 → 12 ---
void testNormalOrder()
{
    constexpr std::uint64_t a = 10, b = 11, c = 12;
    check("normal: isBefore(10,11)", isBefore(a, b));
    check("normal: isBefore(11,12)", isBefore(b, c));
    check("normal: !isBefore(11,10)", !isBefore(b, a));
    check("normal: isAfter(11,10)", isAfter(b, a));
    check("normal: !isAfter(10,11)", !isAfter(a, b));
    check("normal: isAtOrBefore(11,12)", isAtOrBefore(b, c));
    check("normal: isCompleted(12,12)", isCompleted(c, c));
    check("normal: !isCompleted(13,12)", !isCompleted(std::uint64_t{13}, c));
}

// --- 2. out-of-order（将来 sparse completion 前提）: 11 → 10 ---
//   将来 complete(11) → complete(10) が起きても比較自体は modular で正しく判定される。
//   completed 判定は watermark 次第（10 は watermark 11 に到達済み / 12 は未到達）。
void testOutOfOrder()
{
    check("o-o: !isAfter(10,11) [10 after 11 is stale]", !isAfter(std::uint64_t{10}, std::uint64_t{11}));
    check("o-o: !isBefore(11,10) [11 before 10 is false]", !isBefore(std::uint64_t{11}, std::uint64_t{10}));
    check("o-o: isCompleted(10,11) [watermark 11 reached seq 10]", isCompleted(std::uint64_t{10}, std::uint64_t{11}));
    check("o-o: !isCompleted(12,11) [seq 12 not reached]", !isCompleted(std::uint64_t{12}, std::uint64_t{11}));
}

// --- 3. duplicate: 10 → 10 ---
//   等値は before でも after でもない（INV-X2-4: stale completion は上書きしない）。
//   complete() 側は isAfter が false → watermark 不変（意図通り）。
void testDuplicate()
{
    check("dup: !isBefore(10,10)", !isBefore(std::uint64_t{10}, std::uint64_t{10}));
    check("dup: !isAfter(10,10)", !isAfter(std::uint64_t{10}, std::uint64_t{10}));
    check("dup: isAtOrBefore(10,10)", isAtOrBefore(std::uint64_t{10}, std::uint64_t{10}));
    check("dup: isCompleted(10,10)", isCompleted(std::uint64_t{10}, std::uint64_t{10}));
    // complete(10) 後の complete(10): isAfter(10,10) == false → watermark 安定（不変）
    check("dup: isCompleted(10,10) watermark stable", isCompleted(std::uint64_t{10}, std::uint64_t{10}));
}

// --- 4. wraparound: UINT64_MAX-1 → UINT64_MAX → 0 → 1 ---
//   §1.6.1: UINT64_MAX - 1 → UINT64_MAX → 0 → 1 が単純比較 `a < b` で壊れるケース。
void testWrapAround()
{
    constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
    check("wrap: isBefore(max-1, max)", isBefore(max - 1, max));
    check("wrap: isBefore(max, 0)", isBefore(max, std::uint64_t{0}));
    check("wrap: isBefore(0, 1)", isBefore(std::uint64_t{0}, std::uint64_t{1}));
    check("wrap: isAfter(0, max)", isAfter(std::uint64_t{0}, max));
    check("wrap: isAfter(1, 0)", isAfter(std::uint64_t{1}, std::uint64_t{0}));
    check("wrap: !isBefore(0, max) [backward is not before]", !isBefore(std::uint64_t{0}, max));
    // watermark が wrap した後の completed 判定
    check("wrap: isCompleted(max, 0) [seq max reached by watermark 0]", isCompleted(max, std::uint64_t{0}));
    check("wrap: !isCompleted(1, max) [seq 1 not reached yet]", !isCompleted(std::uint64_t{1}, max));
}

// --- 5. antipode（境界）: dist == 2^63 ---
//   ちょうど半周（2^63）は before でも after でもない（曖昧境界）。
//   2^63-1 は forward half 内（before）・2^63+1 は backward half 内（after）。
void testAntipodeBoundary()
{
    constexpr std::uint64_t a = 0;
    constexpr std::uint64_t half = kSeqHalfModulus; // 2^63
    check("antipode: !isBefore(0, 2^63)", !isBefore(a, half));
    check("antipode: !isAfter(0, 2^63)", !isAfter(a, half));
    check("antipode: isBefore(0, 2^63-1)", isBefore(a, half - 1));
    check("antipode: isAfter(0, 2^63+1)", isAfter(a, half + 1));
    check("antipode: !isBefore(0, 2^63+1)", !isBefore(a, half + 1));
}

// --- 6. seqDistance（modular distance）---
void testSeqDistance()
{
    constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
    check("dist: seqDistance(10,12) == 2", seqDistance(10, 12) == 2);
    check("dist: seqDistance(max,0) == 1 (wrap)", seqDistance(max, 0) == 1);
    check("dist: seqDistance(0,max) == max (backward)", seqDistance(0, max) == max);
    check("dist: seqDistance(a,a) == 0", seqDistance(7, 7) == 0);
}

} // namespace

int main()
{
    std::cout << "SequenceArithmeticTests (dash2 §1.6.1 modular sequence arithmetic)" << std::endl;
    testNormalOrder();
    testOutOfOrder();
    testDuplicate();
    testWrapAround();
    testAntipodeBoundary();
    testSeqDistance();
    std::cout << "Tests: " << g_testCount << ", Failures: " << g_failCount << std::endl;
    if (g_failCount == 0)
        std::cout << "ALL TESTS PASSED" << std::endl;
    else
        std::cout << "SOME TESTS FAILED" << std::endl;
    return (g_failCount == 0) ? 0 : 1;
}
