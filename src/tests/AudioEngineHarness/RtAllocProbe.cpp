// U3 (R-107 Stage 3): RT 経路の新規確保プローブ（テストバイナリ専用）。
//
// この TU だけがグローバル operator new / delete を置換する。
// 置換関数は確保を行わない（malloc / free を直接使う）ため再帰しない。
#include "RtAllocProbe.h"

#include <cstdio>
#include <cstdlib>
#include <new>

namespace rtalloc
{
void enableFromEnvironment() noexcept
{
    const char* v = std::getenv("CONVOPEQ_RT_ALLOC_PROBE");
    setEnabledForTest((v != nullptr) && (v[0] == '1'));
}

void printReport() noexcept
{
    std::printf("[U3] probe=%s rt_alloc=%llu rt_bytes=%llu rt_max_bytes=%llu\n",
                enabled() ? "on" : "off",
                static_cast<unsigned long long>(count()),
                static_cast<unsigned long long>(bytes()),
                static_cast<unsigned long long>(maxBytes()));
    std::fflush(stdout);
}
} // namespace rtalloc

//==============================================================================
// グローバル operator new / delete の置換（RtAllocProbe.cpp の 1 箇所のみ）。
//   aligned 版は置換しない（aligned 確保は aligned 解放と対でなければならない）。
//==============================================================================
void* operator new(std::size_t n)
{
    void* p = std::malloc(n != 0 ? n : 1);
    if (p == nullptr)
        throw std::bad_alloc();
    rtalloc::noteAlloc(n);
    return p;
}

void* operator new[](std::size_t n)
{
    void* p = std::malloc(n != 0 ? n : 1);
    if (p == nullptr)
        throw std::bad_alloc();
    rtalloc::noteAlloc(n);
    return p;
}

void* operator new(std::size_t n, const std::nothrow_t&) noexcept
{
    void* p = std::malloc(n != 0 ? n : 1);
    if (p != nullptr)
        rtalloc::noteAlloc(n);
    return p;
}

void* operator new[](std::size_t n, const std::nothrow_t&) noexcept
{
    void* p = std::malloc(n != 0 ? n : 1);
    if (p != nullptr)
        rtalloc::noteAlloc(n);
    return p;
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
