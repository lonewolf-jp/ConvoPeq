#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>

#include <JuceHeader.h>

#include "PreparedIRState.h"

struct CacheHeader
{
    uint64_t magic = 0x434F4E564F504551ULL; // "CONVOPEQ"
    uint64_t version = 1;
    uint64_t key = 0;
    uint64_t dataSize = 0;
    uint64_t checksum = 0;
    uint64_t timestamp = 0;
    uint64_t fftSize = 0;
    uint64_t numPartitions = 0;
    uint64_t numChannels = 0;
    double sampleRate = 0.0;
};

class CacheManager
{
public:
    static uint64_t computeKey(const juce::File& file,
                               int fftSize,
                               double sampleRate,
                               int phaseMode,
                               int partitionSize);

    std::unique_ptr<PreparedIRState> load(uint64_t key, uint64_t generationId);
    void save(uint64_t key, const PreparedIRState& state);
    void clear();
    void evictOldest(size_t maxEntries = 10);

private:
    static uint64_t computeFileContentCRC(const juce::File& file);
    static uint64_t computeCRC64(const uint8_t* data, size_t size);
    static uint64_t hashCombine(uint64_t seed, uint64_t value);

    double* copyFromMmapToAligned(juce::MemoryMappedFile& mmap, size_t dataSize);

    juce::File getCacheDirectory() const;
    juce::File getCacheFile(uint64_t key) const;
    bool validateCacheFile(const juce::File& file, uint64_t expectedKey, CacheHeader& headerOut) const;
};
