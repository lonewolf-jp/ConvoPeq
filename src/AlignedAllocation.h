#pragma once

#include <JuceHeader.h>

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
    double* buffer;
    size_t sizeInBytes;
};

} // namespace convo