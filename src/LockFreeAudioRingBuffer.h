#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4324)
#endif

class LockFreeAudioRingBuffer
{
public:
    void prepare(int channels, int size)
    {
        jassert(channels > 0);
        jassert(size > 0);

        storage.setSize(channels, size, false, true, true);
        storage.clear();
        capacity = size;
        numChannels = channels;
        writeIndex.store(0, std::memory_order_release);
        readIndex.store(0, std::memory_order_release);
    }

    void reset() noexcept
    {
        readIndex.store(0, std::memory_order_release);
        writeIndex.store(0, std::memory_order_release);
    }

    int getAvailableSamples() const noexcept
    {
        const auto written = writeIndex.load(std::memory_order_acquire);
        const auto read = readIndex.load(std::memory_order_acquire);
        return static_cast<int>(written - read);
    }

    int getFreeSpace() const noexcept
    {
        return capacity - getAvailableSamples();
    }

    void push(const juce::dsp::AudioBlock<const double>& block) noexcept
    {
        if (capacity <= 0 || numChannels <= 0)
            return;

        const int samplesToWriteRequested = static_cast<int>(block.getNumSamples());
        const int channelsToWrite = juce::jmin(numChannels, static_cast<int>(block.getNumChannels()));
        if (samplesToWriteRequested <= 0 || channelsToWrite <= 0)
            return;

        const auto write = writeIndex.load(std::memory_order_relaxed);
        const auto read = readIndex.load(std::memory_order_acquire);
        const int free = capacity - static_cast<int>(write - read);
        if (free <= 0)
            return;

        const int samplesToWrite = juce::jmin(samplesToWriteRequested, free);
        const int start = static_cast<int>(write % static_cast<uint64_t>(capacity));
        const int firstChunk = juce::jmin(samplesToWrite, capacity - start);
        const int secondChunk = samplesToWrite - firstChunk;

        for (int channel = 0; channel < channelsToWrite; ++channel)
        {
            float* destination = storage.getWritePointer(channel);
            const double* source = block.getChannelPointer(static_cast<size_t>(channel));

            for (int i = 0; i < firstChunk; ++i)
                destination[start + i] = static_cast<float>(source[i]);

            for (int i = 0; i < secondChunk; ++i)
                destination[i] = static_cast<float>(source[firstChunk + i]);
        }

        if (channelsToWrite == 1 && numChannels > 1)
        {
            float* destination = storage.getWritePointer(1);
            const double* source = block.getChannelPointer(0);

            for (int i = 0; i < firstChunk; ++i)
                destination[start + i] = static_cast<float>(source[i]);

            for (int i = 0; i < secondChunk; ++i)
                destination[i] = static_cast<float>(source[firstChunk + i]);
        }

        writeIndex.store(write + static_cast<uint64_t>(samplesToWrite), std::memory_order_release);
    }

    int popMixToMono(float* destination, int requestedSamples) noexcept
    {
        if (destination == nullptr || requestedSamples <= 0 || capacity <= 0 || numChannels <= 0)
            return 0;

        const auto write = writeIndex.load(std::memory_order_acquire);
        const auto read = readIndex.load(std::memory_order_relaxed);
        const int available = static_cast<int>(write - read);
        if (available <= 0)
            return 0;

        const int samplesToRead = juce::jmin(requestedSamples, available);
        const int start = static_cast<int>(read % static_cast<uint64_t>(capacity));
        const int firstChunk = juce::jmin(samplesToRead, capacity - start);
        const int secondChunk = samplesToRead - firstChunk;

        const float* left = storage.getReadPointer(0);
        const float* right = (numChannels > 1) ? storage.getReadPointer(1) : left;

        mixChunk(left + start, right + start, destination, firstChunk);
        if (secondChunk > 0)
            mixChunk(left, right, destination + firstChunk, secondChunk);

        readIndex.store(read + static_cast<uint64_t>(samplesToRead), std::memory_order_release);
        return samplesToRead;
    }

    void skip(int requestedSamples) noexcept
    {
        if (requestedSamples <= 0 || capacity <= 0)
            return;

        const auto write = writeIndex.load(std::memory_order_acquire);
        const auto read = readIndex.load(std::memory_order_relaxed);
        const int available = static_cast<int>(write - read);
        if (available <= 0)
            return;

        const int samplesToSkip = juce::jmin(requestedSamples, available);
        readIndex.store(read + static_cast<uint64_t>(samplesToSkip), std::memory_order_release);
    }

private:
    static void mixChunk(const float* left, const float* right, float* destination, int samples) noexcept
    {
        if (samples <= 0)
            return;

        if (left == right)
        {
            juce::FloatVectorOperations::copy(destination, left, samples);
            return;
        }

        for (int i = 0; i < samples; ++i)
            destination[i] = 0.5f * (left[i] + right[i]);
    }

    juce::AudioBuffer<float> storage;
    int capacity = 0;
    int numChannels = 0;

    alignas(64) std::atomic<uint64_t> writeIndex { 0 };
    alignas(64) std::atomic<uint64_t> readIndex { 0 };
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif
