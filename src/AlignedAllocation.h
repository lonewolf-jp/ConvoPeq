#pragma once

#include <JuceHeader.h>
#include <vector>
#include <limits>
#include <new>
#include <memory>

#include <mkl.h>

namespace convo {

inline void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = mkl_malloc(size, (int)alignment);
    if (ptr == nullptr) {
        DBG("Memory allocation failed in aligned_malloc (MKL)");
        throw std::bad_alloc();
    }
    return ptr;
}

inline void aligned_free(void* ptr) {
    if (ptr != nullptr) {
        mkl_free(ptr);
    }
}

//-----------------------------------------------------------------------------
// RAII Helper for aligned memory
//-----------------------------------------------------------------------------
//==============================================================================
// RAII wrapper for convo::aligned_malloc / aligned_free (64-byte align)
// JUCE 8.0.12 + oneMKL 完全対応。reset() で旧メモリ自動解放。
//
// Note: This class calls the destructor of a single object (~T()).
//       It is suitable for managing a single non-POD object or an array of POD types.
//       It is NOT suitable for managing an array of non-POD objects.
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
        if (ptr)
        {
            ptr->~T();
            aligned_free(ptr);
        }
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


// =========================================================================
// makeAlignedCopy  – src の内容を 64 バイト境界にコピーした ScopedAlignedPtr を返す
// 非 AudioThread 専用 (LoaderThread 等)。確保失敗時は std::bad_alloc を投げる。
// =========================================================================
inline ScopedAlignedPtr<double> makeAlignedCopy(const double* src, int numSamples)
{
    if (!src || numSamples <= 0)
        return ScopedAlignedPtr<double>(nullptr);

    ScopedAlignedPtr<double> dst(
        static_cast<double*>(aligned_malloc(static_cast<size_t>(numSamples) * sizeof(double), 64)));
    if (dst)
        std::memmove(dst.get(), src, static_cast<size_t>(numSamples) * sizeof(double));
    return dst;
}

} // namespace convo
