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

    // Method 2: AVX2 実行可否確認（MaxLeaf + OSXSAVE + AVX + FMA + AVX2 + XGETBV）
    // Method 1 は通常成功するため、Method 2 は保険的フォールバック経路。
    {
        // Step 0: CPUID(0) で最大 leaf を確認
        // CPUID leaf 7 は最大 leaf >= 7 の CPU でのみ有効。
        // 現代 CPU では不要に近いが、フォールバック実装として堅牢性を高める。
        int leaf0[4] = { 0 };
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
        __cpuid(leaf0, 0);
#elif defined(__GNUC__) || defined(__clang__)
        __get_cpuid(0, &leaf0[0], &leaf0[1], &leaf0[2], &leaf0[3]);
#endif
        if (static_cast<unsigned>(leaf0[0]) < 7u)
            return false;  // leaf 7 未対応 → AVX2 不可

        // Step 1: leaf 1 で OSXSAVE + AVX + FMA を確認
        int leaf1[4] = { 0 };
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
        __cpuid(leaf1, 1);
#elif defined(__GNUC__) || defined(__clang__)
        __get_cpuid(1, &leaf1[0], &leaf1[1], &leaf1[2], &leaf1[3]);
#endif
        // bit 27: OSXSAVE — XGETBV 発行前に必須（なければ #UD）
        constexpr int kOSXSAVEBit = 27;
        if ((leaf1[2] & (1u << kOSXSAVEBit)) == 0)
            return false;
        // bit 28: AVX — AVX 命令使用に必須（Intel SDM: CPUID.1:ECX[28]）
        constexpr int kAVXBit = 28;
        if ((leaf1[2] & (1u << kAVXBit)) == 0)
            return false;
        // bit 12: FMA3（★ leaf 1 ECX、leaf 7 EBX ではない）
        constexpr int kFMABit = 12;
        if ((leaf1[2] & (1u << kFMABit)) == 0)
            return false;

        // Step 2: leaf 7 で AVX2 を確認
        int leaf7[4] = { 0 };
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
        __cpuidex(leaf7, 7, 0);
#elif defined(__GNUC__) || defined(__clang__)
        __cpuid_count(7, 0, leaf7[0], leaf7[1], leaf7[2], leaf7[3]);
#endif
        constexpr int kAVX2Bit = 5;
        if ((leaf7[1] & (1u << kAVX2Bit)) == 0)
            return false;

        // Step 3: XGETBV で OS の YMM 保存を確認
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
        const unsigned int xcr0 = static_cast<unsigned int>(_xgetbv(0));
#else
        unsigned int xcr0_eax = 0, xcr0_edx = 0;
        __asm__("xgetbv" : "=a"(xcr0_eax), "=d"(xcr0_edx) : "c"(0));
        const unsigned int xcr0 = xcr0_eax;
#endif
        // XCR0[1] = XMM enabled, XCR0[2] = YMM enabled
        if ((xcr0 & 0x6u) != 0x6u)
            return false;

        return true;
    }
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
    // Linux/macOS: CPUID + XGETBV による確認
    {
        // Step 0: CPUID(0) で最大 leaf を確認
        unsigned int maxLeaf = 0;
        unsigned int ignore = 0;
        __get_cpuid(0, &maxLeaf, &ignore, &ignore, &ignore);
        if (maxLeaf < 7u)
            return false;

        // Step 1: leaf 1 で OSXSAVE + AVX + FMA を確認
        unsigned int eax1 = 0, ebx1 = 0, ecx1 = 0, edx1 = 0;
        __get_cpuid(1, &eax1, &ebx1, &ecx1, &edx1);
        // bit 27: OSXSAVE — XGETBV 発行前に必須
        if ((ecx1 & (1u << 27)) == 0)
            return false;
        // bit 28: AVX
        if ((ecx1 & (1u << 28)) == 0)
            return false;
        // bit 12: FMA3（★ leaf 1 ECX、leaf 7 EBX ではない）
        if ((ecx1 & (1u << 12)) == 0)
            return false;

        // Step 2: leaf 7 で AVX2 を確認
        unsigned int eax7 = 0, ebx7 = 0, ecx7 = 0, edx7 = 0;
        __get_cpuid_count(7, 0, &eax7, &ebx7, &ecx7, &edx7);
        if ((ebx7 & (1u << 5)) == 0)  // AVX2 bit
            return false;

        // Step 3: XGETBV で OS の YMM 保存を確認
        unsigned int xcr0_eax = 0, xcr0_edx = 0;
        __asm__("xgetbv" : "=a"(xcr0_eax), "=d"(xcr0_edx) : "c"(0));
        if ((xcr0_eax & 0x6u) != 0x6u)  // XCR0[1]=XMM, [2]=YMM
            return false;

        return true;
    }
#else
    // 未対応コンパイラ: AVX2 を使用できないと判断して安全側に倒す
    return false;
#endif
}

bool checkAVX2SupportAndWarn() noexcept
{
    if (hasAVX2Support())
        return true;

    // 非対応 CPU: MessageBox でエラー表示
    ::MessageBoxW(nullptr,
        L"ConvoPeq には AVX2 および FMA 命令に対応した CPU が必要です。\n"
        L"Intel Haswell (2013) 以降、または AMD Excavator (2015) 以降の\n"
        L"CPU が必要です。\n\n"
        L"この CPU ではアプリケーションがクラッシュする可能性があるため、\n"
        L"実行を中断します。",
        L"ConvoPeq - CPU 非対応",
        MB_OK | MB_ICONERROR);

    return false;
}

} // namespace convo
