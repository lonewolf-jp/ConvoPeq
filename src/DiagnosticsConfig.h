#pragma once

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
#define CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS 1
#endif

// ★ RUNTIME_DIAG_LOG マクロ: 単一の diagLog 呼び出しをマクロで囲む
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
#define RUNTIME_DIAG_LOG(x) diagLog(x)
#else
#define RUNTIME_DIAG_LOG(x) do {} while(false)
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

// ---- updateAtomicMaximum : atomic_fetch_max 相当 ----
// compare_exchange_weak ループを隠蔽。memory_order は relaxed で十分。
inline void updateAtomicMaximum(std::atomic<uint32_t>& target, uint32_t value) noexcept
{
    uint32_t expected = target.load(std::memory_order_relaxed);
    while (value > expected && !target.compare_exchange_weak(expected, value,
        std::memory_order_relaxed, std::memory_order_relaxed)) {}
}

#endif // CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
