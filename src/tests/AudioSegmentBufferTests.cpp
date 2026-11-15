// AudioSegmentBufferTests.cpp
// B-1: AudioSegmentBuffer::create() の例外安全（RAII）テスト
//
// 背景:
//   convo::aligned_malloc は失敗時に std::bad_alloc を送出し、nullptr を返さない。
//   旧 create() は
//     * nullptr 判定（到達しないデッドコード）を持ち、
//     * 2 本目の確保で例外が出ると 1 本目（約 29.3 MiB）が漏れ、
//     * new AudioSegmentBuffer() で例外が出ると両方（約 58.6 MiB）が漏れていた。
//   修正後は所有権を常に unique_ptr（AudioSegmentBuffer）が持ち、確保関数が
//   例外を投げた時点で確保済みバッファが RAII で解放される。
//
// テスト内容:
//   1. 成功経路: 生成・書き込み・読み出し（allocator pair の健全性）
//   2. 1 本目の確保で例外 → 何も残らない
//   3. 2 本目の確保で例外 → 1 本目が解放される（B-1 の本命）
//   4. 反復しても PrivateUsage が増えない
//   5. 検出器の感度確認（意図的な 1 本リークを検出できること）
//
// ビルド: カスタム main() + bool testXxx() パターン

#include <JuceHeader.h>
#include "AudioSegmentBuffer.h"

#include <windows.h>
#include <psapi.h>

#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

namespace {

constexpr size_t kBlockBytes =
    static_cast<size_t>(AudioSegmentBuffer::kCapacity) * sizeof(double);

// 1 本のバッファは約 29.3 MiB。リーク検出の閾値はその 1/3 弱に置く。
constexpr SIZE_T kLeakThresholdBytes = 8u * 1024u * 1024u;

int gFailures = 0;

void report(const char* id, bool ok, const char* detail)
{
    std::printf("[%s] %s %s\n", ok ? "PASS" : "FAIL", id, detail);
    if (! ok)
        ++gFailures;
}

SIZE_T privateBytes()
{
    PROCESS_MEMORY_COUNTERS_EX pmc {};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                             sizeof(pmc)) != 0)
        return pmc.PrivateUsage;
    return 0;
}

// n 回目の呼出しで std::bad_alloc を投げる確保関数（1 起点）
struct ThrowingAllocator
{
    int calls = 0;
    int failAt = 0;

    void* operator()(size_t bytes, size_t alignment)
    {
        ++calls;
        if (failAt > 0 && calls == failAt)
            throw std::bad_alloc();
        return convo::aligned_malloc(bytes, alignment);
    }
};

//==============================================================================
// Test 1: 成功経路（生成・書き込み・読み出し・破棄）
//==============================================================================
[[nodiscard]] bool testCreateRoundTrip()
{
    auto buf = AudioSegmentBuffer::create();
    if (! buf)
    {
        std::printf("FAIL testCreateRoundTrip: create() returned null\n");
        return false;
    }

    constexpr int n = 100;
    std::vector<double> left(static_cast<size_t>(n));
    std::vector<double> right(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        left[static_cast<size_t>(i)] = 0.25 * static_cast<double>(i);
        right[static_cast<size_t>(i)] = -0.5 * static_cast<double>(i);
    }

    buf->clear();
    buf->pushBlock(left.data(), right.data(), n);

    if (buf->getNumAvailableSamples() != n)
    {
        std::printf("FAIL testCreateRoundTrip: available=%d expected %d\n",
                    buf->getNumAvailableSamples(), n);
        return false;
    }

    std::vector<double> outLeft(static_cast<size_t>(n), 0.0);
    std::vector<double> outRight(static_cast<size_t>(n), 0.0);
    const int got = buf->copyLatest(outLeft.data(), outRight.data(), n);
    if (got != n)
    {
        std::printf("FAIL testCreateRoundTrip: copyLatest=%d expected %d\n", got, n);
        return false;
    }

    for (int i = 0; i < n; ++i)
    {
        if (outLeft[static_cast<size_t>(i)] != left[static_cast<size_t>(i)]
            || outRight[static_cast<size_t>(i)] != right[static_cast<size_t>(i)])
        {
            std::printf("FAIL testCreateRoundTrip: sample %d mismatch\n", i);
            return false;
        }
    }

    std::printf("[PASS] testCreateRoundTrip (block=%zu bytes x2)\n", kBlockBytes);
    return true;
}

//==============================================================================
// Test 2/3: 確保関数が例外を投げたときの後始末
//==============================================================================
[[nodiscard]] bool testAllocatorFailureReleasesMemory(int failAt, const char* id)
{
    // ヒープの定常状態を作ってから基準値を取る（初回確保の分を除く）
    {
        auto warm = AudioSegmentBuffer::create();
        if (! warm)
        {
            std::printf("FAIL %s: warmup create() returned null\n", id);
            return false;
        }
    }
    const SIZE_T base = privateBytes();

    bool threw = false;
    for (int iteration = 0; iteration < 2; ++iteration)
    {
        ThrowingAllocator alloc;
        alloc.failAt = failAt;
        try
        {
            auto buf = AudioSegmentBuffer::createWithAllocator(alloc);
            // 例外が飛ばないなら注入に失敗している
            std::printf("FAIL %s: allocation failure was not injected (calls=%d)\n",
                        id, alloc.calls);
            return false;
        }
        catch (const std::bad_alloc&)
        {
            threw = true;
        }
    }

    if (! threw)
    {
        std::printf("FAIL %s: std::bad_alloc was not thrown\n", id);
        return false;
    }

    const SIZE_T after = privateBytes();
    const SIZE_T growth = (after > base) ? (after - base) : 0;

    // 2 回反復して 1 本でも漏れていれば 29.3 MiB × 回数 増える
    const bool ok = growth < kLeakThresholdBytes;
    std::printf("[%s] %s failAt=%d growth=%llu bytes (threshold %llu)\n",
                ok ? "PASS" : "FAIL", id,
                failAt,
                static_cast<unsigned long long>(growth),
                static_cast<unsigned long long>(kLeakThresholdBytes));
    if (! ok)
        ++gFailures;
    return ok;
}

//==============================================================================
// Test 4: 検出器の感度確認（意図的な 1 本リークを検出できること）
//   ここが FAIL するなら Test 2/3 の PASS は「測れていないだけ」になる。
//==============================================================================
[[nodiscard]] bool testLeakDetectorSensitivity()
{
    {
        auto warm = AudioSegmentBuffer::create();
        if (! warm)
        {
            std::printf("FAIL testLeakDetectorSensitivity: warmup create() returned null\n");
            return false;
        }
    }
    const SIZE_T base = privateBytes();

    void* leaked = convo::aligned_malloc(kBlockBytes, AudioSegmentBuffer::kAlignment);
    const SIZE_T after = privateBytes();
    const SIZE_T growth = (after > base) ? (after - base) : 0;
    convo::aligned_free(leaked);

    const bool detected = growth >= kLeakThresholdBytes;
    std::printf("[%s] testLeakDetectorSensitivity growth=%llu bytes (threshold %llu)\n",
                detected ? "PASS" : "FAIL",
                static_cast<unsigned long long>(growth),
                static_cast<unsigned long long>(kLeakThresholdBytes));
    if (! detected)
    {
        ++gFailures;
        return false;
    }
    return true;
}

//==============================================================================
// Test 5: 確保関数が nullptr を返したら契約違反として std::bad_alloc
//   createWithAllocator の契約（失敗時は必ず例外）をコード側でも強制している。
//   nullptr を素通しすると「確保済みに見えるが中身が空」のバッファが返る。
//==============================================================================
[[nodiscard]] bool testNullptrAllocatorFailsClosed()
{
    {
        auto warm = AudioSegmentBuffer::create();
        if (! warm)
        {
            std::printf("FAIL testNullptrAllocatorFailsClosed: warmup create() returned null\n");
            return false;
        }
    }
    const SIZE_T base = privateBytes();

    struct NullAllocator
    {
        int calls = 0;
        void* operator()(size_t, size_t)
        {
            ++calls;
            return nullptr;
        }
    };

    NullAllocator alloc;
    bool threw = false;
    try
    {
        auto buf = AudioSegmentBuffer::createWithAllocator(alloc);
        std::printf("FAIL testNullptrAllocatorFailsClosed: nullptr was accepted (calls=%d)\n",
                    alloc.calls);
        return false;
    }
    catch (const std::bad_alloc&)
    {
        threw = true;
    }

    const SIZE_T after = privateBytes();
    const SIZE_T growth = (after > base) ? (after - base) : 0;
    const bool ok = threw && growth < kLeakThresholdBytes;

    std::printf("[%s] testNullptrAllocatorFailsClosed threw=%d calls=%d growth=%llu bytes\n",
                ok ? "PASS" : "FAIL", threw ? 1 : 0, alloc.calls,
                static_cast<unsigned long long>(growth));
    if (! ok)
    {
        ++gFailures;
        return false;
    }
    return true;
}

} // anonymous namespace

//==============================================================================
// main
//==============================================================================
int main()
{
    bool allPassed = true;

    allPassed &= testCreateRoundTrip();
    allPassed &= testLeakDetectorSensitivity();
    allPassed &= testAllocatorFailureReleasesMemory(1, "B-1-first-alloc");
    allPassed &= testAllocatorFailureReleasesMemory(2, "B-1-second-alloc");
    allPassed &= testNullptrAllocatorFailsClosed();

    if (allPassed && gFailures == 0)
    {
        std::printf("\n=== ALL TESTS PASSED ===\n");
        return EXIT_SUCCESS;
    }

    std::printf("\n=== SOME TESTS FAILED (failures=%d) ===\n", gFailures);
    return EXIT_FAILURE;
}
