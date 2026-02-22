#pragma once

#include <JuceHeader.h>

#if JUCE_DSP_USE_INTEL_MKL
#include <mkl.h>
#endif

namespace dsp {


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
    errno = posix_memalign(&ptr, alignment, size);
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

} // namespace dsp