//==============================================================================
// CpuFeatureCheck.cpp
// ★ [P0-1] AVX2 ランタイム検出
//==============================================================================

#include "CpuFeatureCheck.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#include <cpuid.h>
#endif

namespace convo {

static bool hasAVX2Support() noexcept
{
#if defined(_WIN32)
    // Method 1: IsProcessorFeaturePresent (kernel32.dll, Windows 8.1+)
#ifndef PF_AVX2_INSTRUCTIONS_AVAILABLE
#define PF_AVX2_INSTRUCTIONS_AVAILABLE 10
#endif
    {
        const auto kernel32 = ::GetModuleHandleW(L"kernel32.dll");
        if (kernel32 != nullptr)
        {
            using PFN_IsProcessorFeaturePresent = BOOL (WINAPI*)(DWORD);
            const auto pfn = reinterpret_cast<PFN_IsProcessorFeaturePresent>(
                ::GetProcAddress(kernel32, "IsProcessorFeaturePresent"));
            if (pfn != nullptr)
            {
                const BOOL result = pfn(PF_AVX2_INSTRUCTIONS_AVAILABLE);
                if (result != FALSE)
                    return true;
            }
        }
    }

    // Method 2: __cpuidex による直接チェック
    {
        int cpuInfo[4] = { 0 };
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
        __cpuidex(cpuInfo, 7, 0);
#elif defined(__GNUC__) || defined(__clang__)
        __cpuid_count(7, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#endif
        constexpr int kAVX2Bit = 5;
        constexpr int kFMABit   = 12;
        if ((cpuInfo[1] & (1 << kAVX2Bit)) != 0
            && (cpuInfo[1] & (1 << kFMABit)) != 0)
            return true;
        return false;
    }
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
    // Linux/macOS: __get_cpuid_count
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
    {
        constexpr int kAVX2Bit = 5;
        constexpr int kFMABit  = 12;
        if ((ebx & (1u << kAVX2Bit)) != 0
            && (ebx & (1u << kFMABit)) != 0)
            return true;
    }
    return false;
#else
    // 未知のプラットフォーム: 安全側に倒してチェック通過
    return true;
#endif
}

bool checkAVX2SupportAndWarn() noexcept
{
    if (hasAVX2Support())
        return true;

    // 非対応 CPU: MessageBox でエラー表示
    ::MessageBoxA(nullptr,
        "ConvoPeq には AVX2 および FMA 命令に対応した CPU が必要です。\n"
        "Intel Haswell (2013) 以降、または AMD Excavator (2015) 以降の\n"
        "CPU が必要です。\n\n"
        "この CPU ではアプリケーションがクラッシュする可能性があるため、\n"
        "実行を中断します。",
        "ConvoPeq - CPU 非対応",
        MB_OK | MB_ICONERROR);

    return false;
}

} // namespace convo
