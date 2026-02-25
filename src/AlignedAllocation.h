#pragma once

#include <JuceHeader.h>
#include <vector>
#include <limits>
#include <new>

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

//----------------------------------------------------------
// アラインメントされたバッファ管理用クラス
//----------------------------------------------------------
class AlignedBuffer
{
public:
    AlignedBuffer() : buffer(nullptr), sizeInBytes(0) {}

    // ムーブコンストラクタ
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : buffer(other.buffer), sizeInBytes(other.sizeInBytes)
    {
        other.buffer = nullptr;
        other.sizeInBytes = 0;
    }

    // ムーブ代入演算子
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept
    {
        if (this != &other) {
            freeBuffer();
            buffer = other.buffer;
            sizeInBytes = other.sizeInBytes;
            other.buffer = nullptr;
            other.sizeInBytes = 0;
        }
        return *this;
    }

    ~AlignedBuffer() { freeBuffer(); }

    void allocate(size_t numElements)
    {
        freeBuffer();
        if (numElements > 0)
        {
            sizeInBytes = numElements * sizeof(double);
            buffer = static_cast<double*>(aligned_malloc(sizeInBytes, 64)); // 64-byte alignment
        }
    }

    double* get() const { return buffer; }

    void freeBuffer()
    {
        if (buffer != nullptr) {
            aligned_free(buffer);
            buffer = nullptr;
            sizeInBytes = 0;
        }
    }

private:
    // コピー禁止
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    double* buffer;
    size_t sizeInBytes;
};

//-----------------------------------------------------------------------------
// STL Allocator for MKL/AVX512 (64-byte alignment)
// std::vector 等で使用するためのカスタムアロケータ
//-----------------------------------------------------------------------------
template <typename T, size_t Alignment = 64>
struct MKLAllocator {
    using value_type = T;

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