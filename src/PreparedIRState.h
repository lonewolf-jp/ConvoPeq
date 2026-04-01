#pragma once

#include <cstddef>
#include <cstdint>

#include <JuceHeader.h>
#include <mkl.h>

struct PreparedIRState
{
    double* partitionData = nullptr;
    size_t partitionSizeBytes = 0;
    int numPartitions = 0;
    int numSamples = 0;        // actual IR sample count (before partition padding)
    int fftSize = 0;
    int numChannels = 0;
    double sampleRate = 0.0;
    uint64_t generationId = 0;
    uint64_t cacheKey = 0;
    juce::String originalFileName;
    juce::String originalFilePath;

    PreparedIRState() = default;

    PreparedIRState(PreparedIRState&& other) noexcept
        : partitionData(other.partitionData),
          partitionSizeBytes(other.partitionSizeBytes),
          numPartitions(other.numPartitions),
          numSamples(other.numSamples),
          fftSize(other.fftSize),
          numChannels(other.numChannels),
          sampleRate(other.sampleRate),
          generationId(other.generationId),
          cacheKey(other.cacheKey),
          originalFileName(std::move(other.originalFileName)),
          originalFilePath(std::move(other.originalFilePath))
    {
        other.partitionData = nullptr;
        other.partitionSizeBytes = 0;
    }

    PreparedIRState& operator=(PreparedIRState&& other) noexcept
    {
        if (this != &other)
        {
            if (partitionData)
                mkl_free(partitionData);

            partitionData = other.partitionData;
            partitionSizeBytes = other.partitionSizeBytes;
            numPartitions = other.numPartitions;
            numSamples = other.numSamples;
            fftSize = other.fftSize;
            numChannels = other.numChannels;
            sampleRate = other.sampleRate;
            generationId = other.generationId;
            cacheKey = other.cacheKey;
            originalFileName = std::move(other.originalFileName);
            originalFilePath = std::move(other.originalFilePath);

            other.partitionData = nullptr;
            other.partitionSizeBytes = 0;
        }

        return *this;
    }

    ~PreparedIRState()
    {
        if (partitionData)
            mkl_free(partitionData);
    }

    PreparedIRState(const PreparedIRState&) = delete;
    PreparedIRState& operator=(const PreparedIRState&) = delete;
};
