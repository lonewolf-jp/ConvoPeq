#pragma once

#include <mkl_dfti.h>

namespace convo {

/**
    RAII wrapper for MKL DFTI_DESCRIPTOR_HANDLE.

    Automatically calls DftiFreeDescriptor on destruction,
    preventing resource leaks from early returns or exceptions.
*/
struct ScopedDftiDescriptor
{
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;

    ~ScopedDftiDescriptor()
    {
        if (handle != nullptr)
            DftiFreeDescriptor(&handle);
    }

    ScopedDftiDescriptor() = default;

    // Non-copyable (handles ownership)
    ScopedDftiDescriptor(const ScopedDftiDescriptor&) = delete;
    ScopedDftiDescriptor& operator=(const ScopedDftiDescriptor&) = delete;

    // Movable
    ScopedDftiDescriptor(ScopedDftiDescriptor&& other) noexcept
        : handle(other.handle)
    {
        other.handle = nullptr;
    }

    ScopedDftiDescriptor& operator=(ScopedDftiDescriptor&& other) noexcept
    {
        if (handle != nullptr)
            DftiFreeDescriptor(&handle);
        handle = other.handle;
        other.handle = nullptr;
        return *this;
    }

    DFTI_DESCRIPTOR_HANDLE get() const noexcept { return handle; }

    // Returns an out-parameter pointer for DftiCreateDescriptor after releasing current handle.
    DFTI_DESCRIPTOR_HANDLE* put() noexcept
    {
        reset();
        return &handle;
    }

    void reset(DFTI_DESCRIPTOR_HANDLE h = nullptr) noexcept
    {
        if (handle != nullptr)
            DftiFreeDescriptor(&handle);
        handle = h;
    }

    DFTI_DESCRIPTOR_HANDLE release() noexcept
    {
        DFTI_DESCRIPTOR_HANDLE h = handle;
        handle = nullptr;
        return h;
    }
};

} // namespace convo
