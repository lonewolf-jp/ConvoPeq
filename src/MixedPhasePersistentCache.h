#pragma once

#include <JuceHeader.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace convo {

struct SecondOrderAllpass;

/**
    MixedPhasePersistentCache: 最適化済み MixedPhase IR のディスク永続化キャッシュ。
    CMA-ES/GreedyAdaGrad による位相最適化結果を %APPDATA%/ConvoPeq/MixedPhaseCache/ に保存し、
    次回起動時の再最適化をスキップする。
*/
class MixedPhasePersistentCache
{
public:
    static juce::File getCacheDirectory();

    static juce::File getCacheFile(uint64_t fileHash,
                                   double sampleRate,
                                   int phaseMode,
                                   float freqStartHz, float freqEndHz,
                                   int targetLength);

    static bool load(uint64_t fileHash,
                     double sampleRate,
                     int phaseMode,
                     float freqStartHz, float freqEndHz,
                     int targetLength,
                     juce::AudioBuffer<double>& outIr,
                     std::vector<double>& outRho,
                     std::vector<double>& outTheta);

    static bool save(uint64_t fileHash,
                     double sampleRate,
                     int phaseMode,
                     float freqStartHz, float freqEndHz,
                     int targetLength,
                     const juce::AudioBuffer<double>& ir,
                     const std::vector<double>& rho,
                     const std::vector<double>& theta);

    // Phase 1 (D-1R-2M): touch() public API は削除。
    // LRU timestamp update は load() 内の NonRT metadata update に統合された。
    // (D-02 / D-1R-2L F-P0-06b: load-touch は N → 2N の追記破損だった)

    // ======================================================================
    // TEST-ONLY fault-injection seam (Owner D-1R-2AA section 7).
    //
    // This block is compiled ONLY when CONVOPEQ_PCACHE_TEST_SEAM is defined,
    // and that macro is defined ONLY on the PersistentCacheV3Phase0FailureTests
    // target. No production target (ConvoPeq main, Debug or Release) defines
    // it, therefore a production binary contains no resident flag, no extra
    // branch, no allocation, no lock and no RT-reachable code path. The class
    // also gains no public API in production, because the declarations below
    // are preprocessed away together with the definitions.
    //
    // Rationale (SOURCE-PROVEN): juce::FileOutputStream::flush() returns void
    // in JUCE 8, so a flush failure cannot be observed from its return value.
    // The only way to prove "flush failure is diagnostic-only" is to inject
    // the failure at this seam.
    // ======================================================================
#if defined(CONVOPEQ_PCACHE_TEST_SEAM)
    enum class TimestampWriteFault
    {
        none = 0,       //!< normal path: write(8) then flush() are both issued
        timestampWrite, //!< write() reports failure: the 8 bytes are not written
        flush           //!< write() succeeds; the explicit flush is not issued
    };

    static void armTimestampWriteFaultForTest (TimestampWriteFault mode) noexcept;
    static int  getInjectedFaultAttemptsForTest() noexcept;
#endif

    static void evictLRU(size_t maxCount);

    static void remove(uint64_t fileHash,
                       double sampleRate,
                       int phaseMode,
                       float freqStartHz, float freqEndHz,
                       int targetLength);

    static void clear();

    static size_t getEntryCount();

private:
    static constexpr uint64_t kMagic = 0x4D69786564506800ULL;
    static constexpr uint32_t kVersion = 2;

#pragma pack(push, 1)
    struct DiskHeader {
        uint64_t magic;
        uint32_t version;
        uint32_t headerReserved;
        uint64_t fileHash;
        double sampleRate;
        int32_t phaseMode;
        float freqStartHz;
        float freqEndHz;
        int32_t targetLength;
        uint64_t lastUsedTime;
        int32_t numChannels;
        int32_t numSamples;
        int32_t numAllpassSections;
        int64_t totalFileSize;
    };
#pragma pack(pop)

    static_assert(sizeof(DiskHeader) <= 128, "DiskHeader is too large");

    static uint64_t computeKeyHash(uint64_t fileHash,
                                   double sampleRate,
                                   int phaseMode,
                                   float freqStartHz, float freqEndHz,
                                   int targetLength);
};

} // namespace convo
