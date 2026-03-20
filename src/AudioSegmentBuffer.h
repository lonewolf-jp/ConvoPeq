#pragma once

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

        for (int i = 0; i < numSamples; ++i)
        {
            leftSamples[writePosition] = left[i];
            rightSamples[writePosition] = right[i];

            ++writePosition;
            if (writePosition >= kCapacity)
                writePosition = 0;

            // リングバッファが満杯になっても「最新の kCapacity サンプルが利用可能」であることを正しく表現
            // これにより copyLatest() が常に最新データを返せるようになる（最大の原因解消）
            // Audio Thread 内で new / vector::resize / mkl_malloc 禁止の規約を厳守
            totalSamples = std::min(kCapacity, totalSamples + 1);
        }
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
