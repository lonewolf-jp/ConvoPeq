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
        writePosition = 0;
        totalSamples = 0;
    }

    void pushBlock(const double* left, const double* right, int numSamples) noexcept
    {
        if (left == nullptr || right == nullptr || numSamples <= 0)
            return;

        int first = std::min(numSamples, kCapacity - writePosition);
        juce::FloatVectorOperations::copy(leftSamples + writePosition, left, first);
        juce::FloatVectorOperations::copy(rightSamples + writePosition, right, first);

        if (first < numSamples)
        {
            int second = numSamples - first;
            juce::FloatVectorOperations::copy(leftSamples, left + first, second);
            juce::FloatVectorOperations::copy(rightSamples, right + first, second);
            writePosition = second;
        }
        else
        {
            writePosition += numSamples;
            if (writePosition >= kCapacity)
                writePosition = 0;
        }
        totalSamples = std::min(kCapacity, totalSamples + numSamples);
    }

    int copyLatest(double* outLeft, double* outRight, int requestedSamples) const noexcept
    {
        if (outLeft == nullptr || outRight == nullptr || requestedSamples <= 0)
            return 0;

        // 満杯状態（totalSamples >= kCapacity）でも常に最新の requestedSamples 分を返す
        // これで buildTrainingSegments() が毎回十分なセグメントを取得できる
        // Audio Thread 安全（ロック・new・I/O・libm なし）
        const int availableSamples = std::min(requestedSamples,
            totalSamples >= kCapacity ? kCapacity : totalSamples);
        const int start = (writePosition - availableSamples + kCapacity) % kCapacity;

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
        return totalSamples;
    }

private:
    double leftSamples[kCapacity] = {};
    double rightSamples[kCapacity] = {};
    int writePosition = 0;
    int totalSamples = 0;
};
