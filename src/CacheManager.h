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
#include "convolver/IRLoadAdmission.h"   // ★ B-08: 総量上限はロード予算を再利用する

struct CacheHeader
{
    uint64_t magic = 0x434F4E564F504551ULL; // "CONVOPEQ"
    uint64_t version = 2;
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
    double scaleFactor = 1.0;
    uint64_t hasScaleFactor = 0;
};

class CacheManager
{
public:
    using SafeDeleteFn = std::function<bool(uint64_t, int)>;

    // 既定はユーザーのアプリデータ配下。テストは任意のディレクトリを渡して隔離する。
    explicit CacheManager(juce::File cacheDirectoryOverride = {});

    // ★ B-01 / B-02: キャッシュキーの意味論バージョン。
    //   末尾無音トリムを同期経路にも適用したため、同じファイルから作られる IR の長さが
    //   変わる。version 1 のキー（トリム導入前）とは別キーにして、古い無トリム IR を
    //   再利用しない。version 1 を指定すると従来と同じキーになる（移行・検証用）。
    static constexpr uint64_t kCacheKeySchemaVersion = 2;

    static uint64_t computeKey(const juce::File& file,
                             int fftSize,
                             double sampleRate,
                             int phaseMode,
                             int partitionSize,
                             uint64_t schemaVersion = kCacheKeySchemaVersion);

    std::unique_ptr<PreparedIRState> loadPreparedState(uint64_t key, int fftSize, uint64_t generationId);
    void save(uint64_t key, int fftSize, const PreparedIRState& state);
    void touch(uint64_t key, int fftSize);
    void setSafeDeleteChecker(SafeDeleteFn checker);

    void clear();
    void evictLRU(size_t maxEntries = 10);

    // ★ B-08: ディレクトリ総量の上限。新しい値を発明せず、IR admission のロード予算
    //   （1 GiB）を再利用する。キャッシュは IR データの派生物であり、「同時に保持して
    //   よい IR 量」を超えて残す利点がない。
    static constexpr juce::int64 kMaxCacheDirectoryBytes = convo::irload::kMaxIRLoadBytes;

    // ディレクトリ総量を上限以下へ剪定する（冪等）。最終更新の古い順に削除し、
    // safeDeleteChecker が拒否する使用中エントリは残す。コンストラクタから 1 回呼ぶ。
    void pruneDirectory(juce::int64 maxTotalBytes = kMaxCacheDirectoryBytes);

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

    std::unique_ptr<PreparedIRState> loadPreparedStateFromFile(const juce::File& file,
                                                               uint64_t key,
                                                               int fftSize,
                                                               uint64_t generationId);

    bool isEntrySafeToDelete(uint64_t key, int fftSize);

    std::unordered_map<uint64_t, CacheEntry> cacheMap;
    std::list<uint64_t> lruList;
    std::mutex cacheMutex;
    SafeDeleteFn safeDeleteChecker;
    juce::File cacheDirectoryOverride;
};
