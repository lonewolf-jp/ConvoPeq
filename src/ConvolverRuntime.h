#pragma once

#include <cstddef>

#include <JuceHeader.h>
#include <mkl.h>

struct ConvolverRuntime
{
    double* overlapBuffer = nullptr;
    double* inputBuffer = nullptr;
    double* outputBuffer = nullptr;

    int currentFFTSize = 0;
    int currentNumPartitions = 0;

    ~ConvolverRuntime() { clear(); }

    void clear() noexcept
    {
        if (overlapBuffer) { mkl_free(overlapBuffer); overlapBuffer = nullptr; }
        if (inputBuffer) { mkl_free(inputBuffer); inputBuffer = nullptr; }
        if (outputBuffer) { mkl_free(outputBuffer); outputBuffer = nullptr; }
        currentFFTSize = 0;
        currentNumPartitions = 0;
    }

    void reset() noexcept
    {
        if (overlapBuffer && currentFFTSize > 0)
            std::memset(overlapBuffer, 0, static_cast<size_t>(currentFFTSize) * sizeof(double));
        if (inputBuffer && currentFFTSize > 0)
            std::memset(inputBuffer, 0, static_cast<size_t>(currentFFTSize) * sizeof(double));
        if (outputBuffer && currentFFTSize > 0)
            std::memset(outputBuffer, 0, static_cast<size_t>(currentFFTSize) * sizeof(double));
    }

    void reallocate(int fftSize, int numPartitions)
    {
        JUCE_ASSERT_MESSAGE_THREAD;

        if (fftSize <= 0 || numPartitions <= 0)
            return;

        if (fftSize != currentFFTSize || numPartitions != currentNumPartitions)
        {
            clear();

            const size_t bytes = static_cast<size_t>(fftSize) * sizeof(double);
            overlapBuffer = static_cast<double*>(mkl_malloc(bytes, 64));
            inputBuffer = static_cast<double*>(mkl_malloc(bytes, 64));
            outputBuffer = static_cast<double*>(mkl_malloc(bytes, 64));

            jassert(overlapBuffer != nullptr && inputBuffer != nullptr && outputBuffer != nullptr);

            currentFFTSize = fftSize;
            currentNumPartitions = numPartitions;
        }

        reset();
    }

    ConvolverRuntime() = default;
    ConvolverRuntime(const ConvolverRuntime&) = delete;
    ConvolverRuntime& operator=(const ConvolverRuntime&) = delete;
};
