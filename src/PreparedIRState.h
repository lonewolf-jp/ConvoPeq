#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <juce_core/juce_core.h>

#include <mkl.h>

struct PreparedIRState
{
    double* partitionData = nullptr;
    size_t partitionSizeBytes = 0;
    int numPartitions = 0;
    int fftSize = 0;
    int numChannels = 0;
    double sampleRate = 0.0;
    uint64_t generationId = 0;
    uint64_t cacheKey = 0;
    juce::String originalFileName;
    std::unique_ptr<juce::AudioBuffer<double>> timeDomainIR;
    double scaleFactor = 1.0;
    bool hasScaleFactor = false;

    PreparedIRState() = default;

    PreparedIRState(PreparedIRState&& other) noexcept
        : partitionData(other.partitionData),
          partitionSizeBytes(other.partitionSizeBytes),
          numPartitions(other.numPartitions),
          fftSize(other.fftSize),
          numChannels(other.numChannels),
          sampleRate(other.sampleRate),
          generationId(other.generationId),
          cacheKey(other.cacheKey),
          originalFileName(std::move(other.originalFileName)),
                    timeDomainIR(std::move(other.timeDomainIR)),
                    scaleFactor(other.scaleFactor),
                    hasScaleFactor(other.hasScaleFactor)
    {
        other.partitionData = nullptr;
        other.partitionSizeBytes = 0;
                other.scaleFactor = 1.0;
                other.hasScaleFactor = false;
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
            fftSize = other.fftSize;
            numChannels = other.numChannels;
            sampleRate = other.sampleRate;
            generationId = other.generationId;
            cacheKey = other.cacheKey;
            originalFileName = std::move(other.originalFileName);
            timeDomainIR = std::move(other.timeDomainIR);
            scaleFactor = other.scaleFactor;
            hasScaleFactor = other.hasScaleFactor;

            other.partitionData = nullptr;
            other.partitionSizeBytes = 0;
            other.scaleFactor = 1.0;
            other.hasScaleFactor = false;
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
