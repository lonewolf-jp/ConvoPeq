#pragma once

#include <JuceHeader.h>
#include <algorithm>

class AudioSegmentBuffer
{
public:
    static constexpr int kMaxSeconds = 5;
    static constexpr int kMaxSampleRate = 768000;
    static constexpr int kCapacity = kMaxSeconds * kMaxSampleRate;

    void clear() noexcept
    {
        writePosition.store(0, std::memory_order_relaxed);
        totalSamples.store(0, std::memory_order_relaxed);
    }

    void pushBlock(const double* left, const double* right, int numSamples) noexcept
    {
        if (left == nullptr || right == nullptr || numSamples <= 0)
            return;

        const int currentWritePos = writePosition.load(std::memory_order_relaxed);
        int first = std::min(numSamples, kCapacity - currentWritePos);
        juce::FloatVectorOperations::copy(leftSamples + currentWritePos, left, first);
        juce::FloatVectorOperations::copy(rightSamples + currentWritePos, right, first);

        if (first < numSamples)
        {
            int second = numSamples - first;
            juce::FloatVectorOperations::copy(leftSamples, left + first, second);
            juce::FloatVectorOperations::copy(rightSamples, right + first, second);
            writePosition.store(second, std::memory_order_release);
        }
        else
        {
            int nextPos = currentWritePos + numSamples;
            if (nextPos >= kCapacity)
                nextPos = 0;
            writePosition.store(nextPos, std::memory_order_release);
        }
        
        const int currentTotal = totalSamples.load(std::memory_order_relaxed);
        totalSamples.store(std::min(kCapacity, currentTotal + numSamples), std::memory_order_release);
    }

    int copyLatest(double* outLeft, double* outRight, int requestedSamples) const noexcept
    {
        if (outLeft == nullptr || outRight == nullptr || requestedSamples <= 0)
            return 0;

        const int currentTotal = totalSamples.load(std::memory_order_acquire);
        const int currentWritePos = writePosition.load(std::memory_order_acquire);

        const int availableSamples = std::min(requestedSamples,
            currentTotal >= kCapacity ? kCapacity : currentTotal);
        const int start = (currentWritePos - availableSamples + kCapacity) % kCapacity;

        for (int i = 0; i < availableSamples; ++i)
        {
            const int sourceIndex = (start + i) % kCapacity;
            outLeft[i] = leftSamples[sourceIndex];
            outRight[i] = rightSamples[sourceIndex];
        }

        return availableSamples;
    }

    int getNumAvailableSamples() const noexcept
    {
        return totalSamples.load(std::memory_order_acquire);
    }

private:
    double leftSamples[kCapacity] = {};
    double rightSamples[kCapacity] = {};
    std::atomic<int> writePosition { 0 };
    std::atomic<int> totalSamples { 0 };
};
