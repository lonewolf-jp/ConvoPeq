#pragma once

#include <JuceHeader.h>
#include <vector>
#include <limits>
#include <new>
#include <memory>

#if JUCE_DSP_USE_INTEL_MKL
#include <mkl.h>
#endif

namespace convo {


#if JUCE_DSP_USE_INTEL_MKL

static void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = mkl_malloc(size, (int)alignment);
    if (ptr == nullptr) {
        DBG("Memory allocation failed in aligned_malloc (MKL)");
        throw std::bad_alloc();
    }
    return ptr;
}

static void aligned_free(void* ptr) {
    if (ptr != nullptr) {
        mkl_free(ptr);
    }
}

#else

static void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = nullptr;

#if defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0)
    {
        ptr = nullptr;
    }
#endif

    if (ptr == nullptr) {
        DBG("Memory allocation failed in aligned_malloc (non-MKL)");
        throw std::bad_alloc();
    }
    return ptr;
}

static void aligned_free(void* ptr) {
    if (ptr != nullptr) {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

#endif

//-----------------------------------------------------------------------------
// RAII Helper for aligned memory
//-----------------------------------------------------------------------------
//==============================================================================
// RAII wrapper for convo::aligned_malloc / aligned_free (64-byte align)
// JUCE 8.0.12 + oneMKL 完全対応。reset() で旧メモリ自動解放。
//==============================================================================
template <typename T>
class ScopedAlignedPtr
{
public:
    explicit ScopedAlignedPtr(T* p = nullptr) noexcept : ptr(p) {}
    ~ScopedAlignedPtr() noexcept { reset(nullptr); }

    ScopedAlignedPtr(const ScopedAlignedPtr&) = delete;
    ScopedAlignedPtr& operator=(const ScopedAlignedPtr&) = delete;

    ScopedAlignedPtr(ScopedAlignedPtr&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    ScopedAlignedPtr& operator=(ScopedAlignedPtr&& o) noexcept
    {
        if (this != &o) { reset(o.ptr); o.ptr = nullptr; }
        return *this;
    }

    T* get() const noexcept { return ptr; }
    T* operator->() const noexcept { return ptr; }
    T& operator*() const noexcept { return *ptr; }
    T& operator[](std::ptrdiff_t i) const noexcept { return ptr[i]; }

    T* release() noexcept { T* p = ptr; ptr = nullptr; return p; }
    void reset(T* p = nullptr) noexcept
    {
        if (ptr) aligned_free(ptr);
        ptr = p;
    }
    explicit operator bool() const noexcept { return ptr != nullptr; }

private:
    T* ptr = nullptr;
};

//-----------------------------------------------------------------------------
// STL Allocator for MKL/AVX512 (64-byte alignment)
// std::vector 等で使用するためのカスタムアロケータ
//-----------------------------------------------------------------------------
template <typename T, size_t Alignment = 64>
struct MKLAllocator {
    using value_type = T;

    template <typename U>
    struct rebind {
        using other = MKLAllocator<U, Alignment>;
    };

    MKLAllocator() noexcept {}
    template <typename U> MKLAllocator(const MKLAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

        void* ptr = aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        aligned_free(p);
    }

    bool operator==(const MKLAllocator&) const noexcept { return true; }
    bool operator!=(const MKLAllocator&) const noexcept { return false; }
};

} // namespace convo