#pragma once

#include <vector>
#include <limits>
#include <new>
#include <memory>
#include <type_traits>
#include <utility>

#include <mkl.h>
#include "DiagnosticsConfig.h"

namespace convo {

// ★ v8.3: DIAG_MKL_MALLOC/DIAG_MKL_FREE 経由でメモリ追跡を有効化
//   CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1 時に convo::diag::diagMklMalloc が使用される。
//   =0 時は従来通り生の mkl_malloc が呼ばれる（オーバーヘッドゼロ）。
inline void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = DIAG_MKL_MALLOC(size, (int)alignment);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

// ★ [P1-3] 非スロー版: 失敗時は nullptr を返す
//   命名: _nothrow (例外を投げない契約)
inline void* aligned_malloc_nothrow(size_t size, size_t alignment) noexcept
{
    return DIAG_MKL_MALLOC(size, (int)alignment);
}

inline void aligned_free(void* ptr) noexcept {
    if (ptr != nullptr) {
        // ★ v8.3: 解放側は mkl_free 直接呼び出し（DIAG_MKL_FREE は size 引数が必要）。
        //   allocation tracking は DIAG_MKL_MALLOC 側で行うため、解放トラッキングは
        //   省略する（allocatedBytes は aligned 領域で単調増加傾向を示す）。
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
        if (this != &o) { reset(std::exchange(o.ptr, nullptr)); }
        return *this;
    }

    T* get() const noexcept { return ptr; }
    T* operator->() const noexcept { return ptr; }
    T& operator*() const noexcept { return *ptr; }
    T& operator[](std::ptrdiff_t i) const noexcept { return ptr[i]; }

    T* release() noexcept { T* p = ptr; ptr = nullptr; return p; }
    void reset(T* p = nullptr) noexcept
    {
        static_assert(std::is_trivially_destructible_v<T>,
                      "ScopedAlignedPtr only supports trivially destructible types (POD arrays)");
        if (ptr)
        {
            aligned_free(ptr);
        }
        ptr = p;
    }
    explicit operator bool() const noexcept { return ptr != nullptr; }

private:
    T* ptr = nullptr;
};

//==============================================================================
// 配列用エイリアス（意図を明確化）
//==============================================================================
template <typename T>
using ScopedAlignedArray = ScopedAlignedPtr<T>;

template <typename T>
struct AlignedObjectDeleter
{
    AlignedObjectDeleter() noexcept = default;

    // Converting constructor: enables unique_ptr<T> → unique_ptr<const T> move
    // Required for aligned_unique_ptr<RuntimePublishWorld> → aligned_unique_ptr<const RuntimePublishWorld>
    template <typename U>
    AlignedObjectDeleter(AlignedObjectDeleter<U>&&) noexcept {}

    void operator()(T* ptr) const noexcept
    {
        if (ptr == nullptr)
            return;
        ptr->~T();
        // const_cast: icx requires explicit conversion from const T* to void*
        // Actual deallocation (mkl_free) does not modify the pointer.
        convo::aligned_free(const_cast<std::remove_const_t<T>*>(ptr));
    }
};

template <typename T>
using aligned_unique_ptr = std::unique_ptr<T, AlignedObjectDeleter<T>>;

template <typename T, typename... Args>
inline aligned_unique_ptr<T> aligned_make_unique(Args&&... args)
{
    void* mem = aligned_malloc(sizeof(T), 64);
    try
    {
        return aligned_unique_ptr<T>(new (mem) T(std::forward<Args>(args)...));
    }
    catch (...)
    {
        aligned_free(mem);
        throw;
    }
}

// 配列として使用する場合の推奨ファクトリ関数
template <typename T>
inline ScopedAlignedArray<T> makeAlignedArray(size_t count) {
    static_assert(std::is_trivially_destructible_v<T>,
                  "Aligned array only supports trivially destructible types");
    T* ptr = static_cast<T*>(aligned_malloc(count * sizeof(T), 64));
    if (!ptr) throw std::bad_alloc();
    return ScopedAlignedArray<T>(ptr);
}

// ★ [P1-3] 非スロー版の配列ファクトリ（失敗時は nullptr 内包の ScopedAlignedArray を返す）
template <typename T>
inline ScopedAlignedArray<T> makeAlignedArray_nothrow(size_t count) noexcept {
    static_assert(std::is_trivially_destructible_v<T>,
                  "Aligned array only supports trivially destructible types");
    T* ptr = static_cast<T*>(aligned_malloc_nothrow(count * sizeof(T), 64));
    return ScopedAlignedArray<T>(ptr);
}

// 既存コードとの互換性のため、makeAlignedCopy はそのまま維持
// 新規コードでは makeAlignedArray<double>(size) を推奨

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
