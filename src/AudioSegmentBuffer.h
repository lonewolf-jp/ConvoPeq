#pragma once

#include <JuceHeader.h>
#include <algorithm>

#include "audioengine/AtomicAccess.h"

class AudioSegmentBuffer
{
public:
    static constexpr int kMaxSeconds = 5;
    static constexpr int kMaxSampleRate = 768000;
    static constexpr int kCapacity = kMaxSeconds * kMaxSampleRate;

    void clear() noexcept
    {
        // release: clear 後に pushBlock/copyLatest を実行するスレッドが
        //          acquire で 0 実再を観測できるよう HB を保証。
        convo::publishAtomic(writePosition, 0, std::memory_order_release);
        convo::publishAtomic(totalSamples, 0, std::memory_order_release);
    }

    void pushBlock(const double* left, const double* right, int numSamples) noexcept
    {
        if (left == nullptr || right == nullptr || numSamples <= 0)
            return;

        // acquire: 直前の clear/pushBlock の release と HB し、有効な writePosition を取得。
        const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire);
        int first = std::min(numSamples, kCapacity - currentWritePos);
        juce::FloatVectorOperations::copy(leftSamples + currentWritePos, left, first);
        juce::FloatVectorOperations::copy(rightSamples + currentWritePos, right, first);

        if (first < numSamples)
        {
            int second = numSamples - first;
            juce::FloatVectorOperations::copy(leftSamples, left + first, second);
            juce::FloatVectorOperations::copy(rightSamples, right + first, second);
            // release: 更新後の writePosition を読み取りスレッドに可視化。
            convo::publishAtomic(writePosition, second, std::memory_order_release);
        }
        else
        {
            int nextPos = currentWritePos + numSamples;
            if (nextPos >= kCapacity)
                nextPos = 0;
            // release: 次書き込み位置を読み取りスレッドに可視化。
            convo::publishAtomic(writePosition, nextPos, std::memory_order_release);
        }

        // acquire: clear/pushBlock の release と HB し、有効な totalSamples を取得。
        const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);
        // release: 更新後の totalSamples を読み取りスレッドに可視化。
        convo::publishAtomic(totalSamples, std::min(kCapacity, currentTotal + numSamples), std::memory_order_release);
    }

    int copyLatest(double* outLeft, double* outRight, int requestedSamples) const noexcept
    {
        if (outLeft == nullptr || outRight == nullptr || requestedSamples <= 0)
            return 0;

        // acquire: pushBlock の release と HB し、最新の totalSamples/writePosition を取得。
        const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);
        const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire);

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
        // acquire: pushBlock/clear の release と HB し、最新の totalSamples を取得。
        return convo::consumeAtomic(totalSamples, std::memory_order_acquire);
    }

private:
    double leftSamples[kCapacity] = {};
    double rightSamples[kCapacity] = {};
    std::atomic<int> writePosition { 0 };
    std::atomic<int> totalSamples { 0 };
};
