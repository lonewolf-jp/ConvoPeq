#pragma once

#include <cstdint>
#include <atomic>

// ============================================================================
// Runtime診断ログの一括制御マクロ
//
// CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
//   1: 有効（全診断ログが出力される）
//   0: 無効（診断コードはコンパイルされない）
//
// CMakeLists.txt で -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1 として定義することも可能。
// デフォルトは 0（無効）。音飛び調査時は 1 に変更してリビルドすること。
// ============================================================================

#ifndef CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
#define CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS 0
#endif

// work60: 診断ログサンプリングマスク。CMakeのtarget_compile_definitionsで指定可能。
// ★ [work62] Release:0xFF(1/256), Debug:0x3F(1/64) に引き上げ
#ifndef CONVOPEQ_DIAG_SAMPLE_MASK
#ifdef NDEBUG
#define CONVOPEQ_DIAG_SAMPLE_MASK 0xFF  // Release: 1/256
#else
#define CONVOPEQ_DIAG_SAMPLE_MASK 0x3F  // Debug: 1/64
#endif
#endif

// ============================================================================
// DIAG_MKL_MALLOC / DIAG_MKL_FREE マクロ — 非DIAGビルドでも mkl_malloc に
// 展開するため、外側の #if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS より前に配置。
// 内側の #if で DIAG/非DIAG を切り替える（自己完結）。
//
// ★ work70: CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS ガードの内部にあったため
//   DIAG=0 のビルドで DIAG_MKL_MALLOC が未定義になるバグを修正。
//   本ブロックは独立した #if/#else/#endif を持ち、外側ガードに依存しない。
// ============================================================================
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
  #ifdef _MSC_VER
    #define DIAG_MKL_MALLOC(size, align) convo::diag::diagMklMalloc((size), (align))
    #define DIAG_MKL_FREE(ptr, size) \
        convo::diag::diagMklFree((ptr), (size), __FILE_NAME__, __LINE__, __func__)
  #else
    #define DIAG_MKL_MALLOC(size, align) convo::diag::diagMklMalloc((size), (align))
    #define DIAG_MKL_FREE(ptr, size) \
        convo::diag::diagMklFree((ptr), (size), __FILE__, __LINE__, __func__)
  #endif
#else
  #define DIAG_MKL_MALLOC(size, align) mkl_malloc((size), (align))
  #define DIAG_MKL_FREE(ptr, size)     mkl_free(ptr)
#endif

// ============================================================================
// 共通診断基盤
// ProcessMemoryInfo / getProcessMemoryInfo() / diagSequenceCounter /
// updateAtomicMaximum を一元定義。全ファイルで共有する。
//
// ★ #pragma comment(lib, "psapi.lib") は各 .cpp 側に配置。
// ============================================================================

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS

#include <windows.h>
#include <psapi.h>
#include <cassert>      // assert
#include <algorithm>    // std::max

// ============================================================================
// MklAllocStats — MKL 分配の Single Source of Truth
// allocationMap / mutex なし。呼び出し元が free 時にサイズを渡す。
// ============================================================================
namespace convo::diag {

struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };   // 現在使用量
    std::atomic<uint64_t> peakBytes      { 0 };   // ピーク使用量
    std::atomic<uint64_t> totalAllocBytes{ 0 };   // 累積確保
    std::atomic<uint64_t> totalFreedBytes{ 0 };   // 累積解放
    std::atomic<uint32_t> lostFreeCount  { 0 };   // diagMklFree(size==0) で呼ばれた回数
    std::atomic<uint32_t> zeroAllocSizeCount{ 0 };// addIfAlive で allocSizes==0 を検出した回数
};

inline MklAllocStats& mklStats() noexcept
{
    static MklAllocStats stats{};
    return stats;
}

} // namespace convo::diag

// ---- updateAtomicMaximum : atomic_fetch_max 相当（global scope, 全ファイルで共有） ----
inline void updateAtomicMaximum(std::atomic<uint32_t>& target, uint32_t value) noexcept
{
    uint32_t expected = target.load(std::memory_order_relaxed);
    while (value > expected && !target.compare_exchange_weak(expected, value, std::memory_order_relaxed, std::memory_order_relaxed)) {}
}

inline void updateAtomicMaximum64(std::atomic<uint64_t>& target, uint64_t value) noexcept
{
    uint64_t expected = target.load(std::memory_order_relaxed);
    while (value > expected && !target.compare_exchange_weak(expected, value, std::memory_order_relaxed, std::memory_order_relaxed)) {}
}

namespace convo::diag {

inline void* diagMklMalloc(size_t size, int alignment) noexcept
{
    void* ptr = mkl_malloc(size, alignment);
    if (ptr)
    {
        const uint64_t bytes = static_cast<uint64_t>(size);
        const uint64_t prev = mklStats().allocatedBytes.fetch_add(bytes, std::memory_order_relaxed);
        mklStats().totalAllocBytes.fetch_add(bytes, std::memory_order_relaxed);
        // peak は fetch_add の戻り値を使用（load 省略）
        updateAtomicMaximum64(mklStats().peakBytes, prev + bytes);
    }
    return ptr;
}

inline void diagMklFree(void* ptr, size_t size,
                         const char* file, int line, const char* func) noexcept
{
    if (ptr)
    {
        mkl_free(ptr);
        if (size > 0)
        {
            mklStats().allocatedBytes.fetch_sub(static_cast<uint64_t>(size), std::memory_order_relaxed);
            mklStats().totalFreedBytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
        }
        else
        {
            // size==0 → 解放サイズ不明（lostFreeCount のみ増加）
            // zeroAllocSizeCount は freeTracked 側で管理
            mklStats().lostFreeCount.fetch_add(1, std::memory_order_relaxed);
            // DBG相当は呼び出し元マクロ側で行う
        }
    }
}

// ---- accessors ----
[[nodiscard]] inline uint64_t allocatedBytes() noexcept
    { return mklStats().allocatedBytes.load(std::memory_order_relaxed); }
[[nodiscard]] inline uint64_t peakBytes() noexcept
    { return mklStats().peakBytes.load(std::memory_order_relaxed); }
[[nodiscard]] inline uint64_t totalAllocBytes() noexcept
    { return mklStats().totalAllocBytes.load(std::memory_order_relaxed); }
[[nodiscard]] inline uint64_t totalFreedBytes() noexcept
    { return mklStats().totalFreedBytes.load(std::memory_order_relaxed); }
[[nodiscard]] inline uint32_t lostFreeCount() noexcept
    { return mklStats().lostFreeCount.load(std::memory_order_relaxed); }
[[nodiscard]] inline uint32_t zeroAllocSizeCount() noexcept
    { return mklStats().zeroAllocSizeCount.load(std::memory_order_relaxed); }

inline void resetDiagnostics() noexcept
{
    mklStats().peakBytes.store(mklStats().allocatedBytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
    mklStats().totalAllocBytes.store(0, std::memory_order_relaxed);
    mklStats().totalFreedBytes.store(0, std::memory_order_relaxed);
    mklStats().lostFreeCount.store(0, std::memory_order_relaxed);
    mklStats().zeroAllocSizeCount.store(0, std::memory_order_relaxed);
}

/// ★ OS Private Usage のうち MKL + Retire 以外の全メモリ（OtherPrivate）。
///   以下を含む（非網羅）: JUCE heap, CRT heap, IPP FFT spec/work, Thread stacks,
///   DLL mappings, VirtualAlloc, std::vector/string internal allocations, MKL internal.
inline uint64_t computeOtherPrivate(uint64_t osPrivateMB,
                                    uint64_t mklBytes,
                                    uint64_t retireBytes) noexcept
{
    const int64_t other = static_cast<int64_t>(osPrivateMB) * 1024 * 1024
                        - static_cast<int64_t>(mklBytes)
                        - static_cast<int64_t>(retireBytes);
    return static_cast<uint64_t>(std::max<int64_t>(0, other));
}

} // namespace convo::diag

// ============================================================================
// freeTracked / addIfAlive — ヘルパ関数
// ============================================================================

/// DIAG_MKL_FREE + nullptr 代入を一括化。
/// size>0 → DIAG_MKL_FREE（統計更新あり）
/// size==0 → mkl_free + zeroAllocSizeCount 増加（allocSizes 保存漏れ）
template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p)
    {
        if (size > 0)
        {
            DIAG_MKL_FREE(p, size);
        }
        else
        {
            // size==0 → allocSizes 保存漏れ。zeroAllocSizeCount のみ増加。
            // lostFreeCount は増やさない（diagMklFree(size==0) のみ）。
            convo::diag::mklStats().zeroAllocSizeCount.fetch_add(1, std::memory_order_relaxed);
            mkl_free(p);
        }
        p = nullptr;
    }
}

/// ポインタ生存確認付き加算。ptr==nullptr → 0 を返す。
/// size==0 は allocSizes 保存漏れ（zeroAllocSizeCount 増加）。
/// lostFreeCount は変更しない（diagMklFree の責務）。
inline uint64_t addIfAlive(const double* ptr, size_t allocSize, const char* /*name*/) noexcept
{
    if (ptr)
    {
        if (allocSize == 0)
            convo::diag::mklStats().zeroAllocSizeCount.fetch_add(1, std::memory_order_relaxed);
        return allocSize;
    }
    return 0;
}

// ---- シーケンス番号（全診断ログ共通の単調増加カウンタ） ----
inline std::atomic<uint64_t>& diagSequenceCounter() noexcept
{
    static std::atomic<uint64_t> counter{ 0 };
    return counter;
}

// ---- ProcessMemoryInfo ----
struct ProcessMemoryInfo {
    uint64_t privateUsageMB = 0;
    uint64_t workingSetMB = 0;
    uint64_t pagefileUsageMB = 0;
    uint64_t pageFaultCount = 0;
};

inline ProcessMemoryInfo getProcessMemoryInfo() noexcept
{
    ProcessMemoryInfo info{};
    PROCESS_MEMORY_COUNTERS_EX pmc;
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
        reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)))
    {
        info.privateUsageMB = static_cast<uint64_t>(pmc.PrivateUsage / (1024ULL * 1024ULL));
        info.workingSetMB = static_cast<uint64_t>(pmc.WorkingSetSize / (1024ULL * 1024ULL));
        info.pagefileUsageMB = static_cast<uint64_t>(pmc.PagefileUsage / (1024ULL * 1024ULL));
        info.pageFaultCount = static_cast<uint64_t>(pmc.PageFaultCount);
    }
    return info;
}

#endif // CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
