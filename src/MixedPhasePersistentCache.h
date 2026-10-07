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
    // Phase 2 (D-1R-2AI): canonical persistent cache version. v1 / v2 are
    // obsolete and are never auto-loaded, migrated, or auto-deleted.
    static constexpr uint32_t kVersion = 3;

    // Phase 2 (Owner D-1R-2AJ section 6): adopted constant. Bound of the
    // artifact's numAllpassSections, enforced independently on the consumer
    // side. The producer's emergent maximum is also 20, but AllpassDesignerConfig
    // accepts any int, so the artifact bound is NOT delegated to the producer.
    static constexpr std::int64_t kMaxAllpassSections = 20;

    // Phase 2: canonical IR-length and channel bounds. These MUST track
    // ConvolverProcessor::MAX_IR_LATENCY (2^21 = 2,097,152) and
    // convo::irload::kMaxIRLoadChannels (8u). They are local aliases so that
    // MixedPhasePersistentCache does not have to include ConvolverProcessor.h
    // (which would drag in the MKL include chain). Phase 0 tests
    // (PersistentCacheV3Phase0SRTests.cpp) already pin the same values against
    // the real declarations, so any drift is caught by the SR sweep.
    static constexpr std::int64_t kMaxIrLatencyBound = 2097152;   // == MAX_IR_LATENCY
    static constexpr std::int64_t kMaxIrChannelsBound = 8;       // == kMaxIRLoadChannels

    // Phase 2: canonical v3 namespace. The v2 directory
    // (%APPDATA%/ConvoPeq/MixedPhaseCache/) is never referenced, so v2 artifacts
    // are never auto-loaded, migrated, auto-deleted, nor LRU candidates.
    static constexpr const char* kCacheNamespaceV3 = "MixedPhaseCacheV3";

    // Phase 2 (Owner D-1R-2AI section 5.1): the LRU / entry-count classification.
    enum class ArtifactState
    {
        currentV3Canonical,   //!< the only LRU candidate and the only counted entry
        stagingOrTemporary,   //!< not *.mph (juce::TemporaryFile staging artifact)
        malformedOrTruncated, //!< DiskHeader cannot be read in full, or size < header
        exactSizeMismatch,    //!< actual file size != checked expected size
        foreign,              //!< magic mismatch
        future,               //!< version > kVersion
        obsolete,             //!< version < kVersion (v1 / v2)
        outsideV3Namespace    //!< not inside the v3 namespace directory
    };

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

    // ----------------------------------------------------------------------
    // Phase 2 canonical validator (Owner D-1R-2AJ P2-I1 / P2-R6).
    //
    // Contract:
    //  - every dimension argument is std::int64_t; the caller widens first and
    //    narrows to the int32_t DiskHeader fields only AFTER this returns true
    //  - no filesystem I/O, no allocation, no lock, no RT policy decision
    //  - the identical function is used by the producer (save) and by the
    //    consumer (load) and by the LRU / entry-count classifier
    //
    // kMaxIrLatency is passed in so that this private helper does not depend on
    // ConvolverProcessor; the value supplied is MAX_IR_LATENCY (2^21).
    // ----------------------------------------------------------------------
    [[nodiscard]] static bool validateCanonicalDims(std::int64_t targetLength,
                                                   std::int64_t numSamples,
                                                   std::int64_t numChannels,
                                                   std::int64_t numAllpassSections,
                                                   std::int64_t kMaxIrLatency,
                                                   std::int64_t maxChannels,
                                                   std::int64_t& expectedFileSizeOut) noexcept;

    // Expected artifact size, derived with checked int64 arithmetic. Never a
    // hard-coded constant (Owner D-1R-2AJ section 6).
    [[nodiscard]] static bool computeExpectedSizeChecked(std::int64_t numChannels,
                                                         std::int64_t numSamples,
                                                         std::int64_t numAllpassSections,
                                                         std::int64_t& expectedSizeOut) noexcept;

    // Classify one directory entry (Owner D-1R-2AJ section 5.1 fixed order).
    [[nodiscard]] static ArtifactState classifyArtifact(const juce::File& file) noexcept;
    [[nodiscard]] static bool isCurrentV3Canonical(ArtifactState state) noexcept
    {
        return state == ArtifactState::currentV3Canonical;
    }
};

} // namespace convo
