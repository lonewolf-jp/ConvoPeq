#pragma once

#include <cstddef>

#include <JuceHeader.h>

#include "AlignedAllocation.h"

struct ConvolverRuntime
{
    convo::ScopedAlignedPtr<double> overlapBuffer;
    convo::ScopedAlignedPtr<double> inputBuffer;
    convo::ScopedAlignedPtr<double> outputBuffer;

    int currentFFTSize = 0;
    int currentNumPartitions = 0;

    ~ConvolverRuntime() { clear(); }

    void clear() noexcept
    {
        overlapBuffer.reset();
        inputBuffer.reset();
        outputBuffer.reset();
        currentFFTSize = 0;
        currentNumPartitions = 0;
    }

    void reset() noexcept
    {
        if (overlapBuffer && currentFFTSize > 0)
            std::memset(overlapBuffer.get(), 0, static_cast<size_t>(currentFFTSize) * sizeof(double));
        if (inputBuffer && currentFFTSize > 0)
            std::memset(inputBuffer.get(), 0, static_cast<size_t>(currentFFTSize) * sizeof(double));
        if (outputBuffer && currentFFTSize > 0)
            std::memset(outputBuffer.get(), 0, static_cast<size_t>(currentFFTSize) * sizeof(double));
    }

    void reallocate(int fftSize, int numPartitions)
    {
        JUCE_ASSERT_MESSAGE_THREAD;

        if (fftSize <= 0 || numPartitions <= 0)
            return;

        if (fftSize != currentFFTSize || numPartitions != currentNumPartitions)
        {
            clear();

            overlapBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(fftSize));
            inputBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(fftSize));
            outputBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(fftSize));

            jassert(overlapBuffer.get() != nullptr && inputBuffer.get() != nullptr && outputBuffer.get() != nullptr);

            currentFFTSize = fftSize;
            currentNumPartitions = numPartitions;
        }

        reset();
    }

    ConvolverRuntime() = default;
    ConvolverRuntime(const ConvolverRuntime&) = delete;
    ConvolverRuntime& operator=(const ConvolverRuntime&) = delete;
};
