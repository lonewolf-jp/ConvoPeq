#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <functional>

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
    uint64_t timeDomainChannels = 0;
    uint64_t timeDomainNumSamples = 0;
    uint64_t timeDomainSizeBytes = 0;
};

class CacheManager
{
public:
    using SafeDeleteFn = std::function<bool(uint64_t, int)>;

    static uint64_t computeKey(const juce::File& file,
                               int fftSize,
                               double sampleRate,
                               int phaseMode,
                               int partitionSize);

    std::unique_ptr<PreparedIRState> load(uint64_t key, int fftSize, uint64_t generationId);
    void save(uint64_t key, int fftSize, const PreparedIRState& state);
    void touch(uint64_t key, int fftSize);
    void setSafeDeleteChecker(SafeDeleteFn checker);

    void clear();
    void evictLRU(size_t maxEntries = 10);

private:
    struct CacheEntry
    {
        juce::File file;
        uint64_t originalKey = 0;
        uint64_t lastAccessTime = 0;
        int fftSize = 0;
        std::list<uint64_t>::iterator lruPos;
    };

    static uint64_t computeFileContentCRC(const juce::File& file);
    static uint64_t computeCRC64(const uint8_t* data, size_t size);
    static uint64_t hashCombine(uint64_t seed, uint64_t value);
    static uint64_t makeEntryKey(uint64_t key, int fftSize);

    double* copyFromMmapToAligned(juce::MemoryMappedFile& mmap, size_t dataSize);

    juce::File getCacheDirectory() const;
    juce::File getCacheFile(uint64_t key, int fftSize) const;
    bool validateCacheFile(const juce::File& file, uint64_t expectedKey, int expectedFftSize, CacheHeader& headerOut) const;

    bool isEntrySafeToDelete(uint64_t key, int fftSize) const;

    std::unordered_map<uint64_t, CacheEntry> cacheMap;
    std::list<uint64_t> lruList;
    mutable std::mutex cacheMutex;
    SafeDeleteFn safeDeleteChecker;
};
