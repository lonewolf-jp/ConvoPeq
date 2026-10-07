#include "MixedPhasePersistentCache.h"

#include <JuceHeader.h>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <limits>

// ===========================================================================
// TEST-ONLY fault-injection seam (Owner D-1R-2AA section 7).
//
// Compiled ONLY when CONVOPEQ_PCACHE_TEST_SEAM is defined. That macro is set
// ONLY on the PersistentCacheV3Phase0FailureTests target, so this whole block
// is absent from every production binary (ConvoPeq main, Debug and Release):
// no resident flag, no branch, no allocation, no lock, no RT-reachable code.
//
// The state is a function-local static POD. It is touched only from the
// existing NonRT metadata-update block inside load(); it does not add any new
// reachability from the audio/RT path, and it adds no public API, no
// authority surface and no production constant.
// ===========================================================================
#if defined(CONVOPEQ_PCACHE_TEST_SEAM)
namespace convo {

namespace
{
    struct TestSeamState
    {
        MixedPhasePersistentCache::TimestampWriteFault fault
            = MixedPhasePersistentCache::TimestampWriteFault::none;
        int attempts = 0;
    };

    TestSeamState& testSeamState() noexcept
    {
        static TestSeamState s;
        return s;
    }
}

void MixedPhasePersistentCache::armTimestampWriteFaultForTest (TimestampWriteFault mode) noexcept
{
    testSeamState().fault = mode;
    testSeamState().attempts = 0;
}

int MixedPhasePersistentCache::getInjectedFaultAttemptsForTest() noexcept
{
    return testSeamState().attempts;
}

} // namespace convo
#endif // CONVOPEQ_PCACHE_TEST_SEAM

namespace convo {

// ═══════════════════════════════════════════════════════════════
//  内部ヘルパー
// ═══════════════════════════════════════════════════════════════

static uint64_t hashCombine(uint64_t seed, uint64_t value)
{
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

uint64_t MixedPhasePersistentCache::computeKeyHash(uint64_t fileHash,
                                                    double sampleRate,
                                                    int phaseMode,
                                                    float freqStartHz, float freqEndHz,
                                                    int targetLength)
{
    uint64_t h = fileHash;
    uint64_t srBits = 0;
    std::memcpy(&srBits, &sampleRate, sizeof(double));
    h = hashCombine(h, srBits);
    h = hashCombine(h, static_cast<uint64_t>(phaseMode));

    uint32_t f1Bits = 0;
    uint32_t f2Bits = 0;
    std::memcpy(&f1Bits, &freqStartHz, sizeof(float));
    std::memcpy(&f2Bits, &freqEndHz, sizeof(float));
    h = hashCombine(h, static_cast<uint64_t>(f1Bits));
    h = hashCombine(h, static_cast<uint64_t>(f2Bits));
    h = hashCombine(h, static_cast<uint64_t>(targetLength));
    return h;
}

// ═══════════════════════════════════════════════════════════════
//  パス解決
// ═══════════════════════════════════════════════════════════════

//==============================================================================
// Phase 2 canonical validator (Owner D-1R-2AJ P2-I1).
//
// Implemented in the int64 domain. No filesystem I/O, no allocation, no lock,
// no RT policy decision. Called identically by the producer (save), the
// consumer (load) and the LRU / entry-count classifier, so predicate identity
// is structural (Owner P2-R6 item 9).
//==============================================================================
namespace
{
    // Checked int64 helpers. INT32_MAX is never used as a bound (Owner section 5).
    [[nodiscard]] bool checkedMul(std::int64_t a, std::int64_t b, std::int64_t& out) noexcept
    {
        if (a < 0 || b < 0)
            return false;
        if (a != 0 && b > (std::numeric_limits<std::int64_t>::max)() / a)
            return false;
        out = a * b;
        return true;
    }

    [[nodiscard]] bool checkedAdd(std::int64_t a, std::int64_t b, std::int64_t& out) noexcept
    {
        if (a < 0 || b < 0)
            return false;
        if (a > (std::numeric_limits<std::int64_t>::max)() - b)
            return false;
        out = a + b;
        return true;
    }
} // namespace

bool MixedPhasePersistentCache::computeExpectedSizeChecked(std::int64_t numChannels,
                                                            std::int64_t numSamples,
                                                            std::int64_t numAllpassSections,
                                                            std::int64_t& expectedSizeOut) noexcept
{
    // sizeof(DiskHeader) is 76 under #pragma pack(1); derived, not hard-coded.
    std::int64_t total = static_cast<std::int64_t>(sizeof(DiskHeader));

    std::int64_t perChannel = 0;
    if (! checkedMul(numSamples, static_cast<std::int64_t>(sizeof(double)), perChannel))
        return false;

    std::int64_t payloadAll = 0;
    if (! checkedMul(numChannels, perChannel, payloadAll))
        return false;
    if (! checkedAdd(total, payloadAll, total))
        return false;

    std::int64_t perSection = 0;
    if (! checkedMul(numAllpassSections, static_cast<std::int64_t>(sizeof(double)), perSection))
        return false;

    std::int64_t allpassAll = 0;   // rho + theta
    if (! checkedMul(perSection, 2, allpassAll))
        return false;
    if (! checkedAdd(total, allpassAll, total))
        return false;

    expectedSizeOut = total;
    return true;
}

bool MixedPhasePersistentCache::validateCanonicalDims(std::int64_t targetLength,
                                                       std::int64_t numSamples,
                                                       std::int64_t numChannels,
                                                       std::int64_t numAllpassSections,
                                                       std::int64_t kMaxIrLatency,
                                                       std::int64_t maxChannels,
                                                       std::int64_t& expectedFileSizeOut) noexcept
{
    // Predicate 1: 1 <= targetLength <= MAX_IR_LATENCY
    if (targetLength < 1 || targetLength > kMaxIrLatency)
        return false;
    // Predicate 2: 1 <= numSamples <= MAX_IR_LATENCY
    if (numSamples < 1 || numSamples > kMaxIrLatency)
        return false;
    // Predicate 3: targetLength == numSamples
    if (targetLength != numSamples)
        return false;
    // Predicate 4: 1 <= numChannels <= kMaxIRLoadChannels
    if (numChannels < 1 || numChannels > maxChannels)
        return false;
    // Predicate 5: 0 <= numAllpassSections <= kPersistentCacheMaxAllpassSections
    if (numAllpassSections < 0 || numAllpassSections > kMaxAllpassSections)
        return false;

    // Predicates 6 + 7: checked arithmetic, overflow rejected.
    return computeExpectedSizeChecked(numChannels, numSamples, numAllpassSections,
                                      expectedFileSizeOut);
}

//==============================================================================
// Phase 2 artifact classifier (Owner D-1R-2AJ section 5.1 fixed order).
// Order: v3 namespace -> staging/temporary -> header readable -> magic ->
//        version -> dimensions (predicates 1..7) -> checked expected size vs
//        actual size -> header.totalFileSize == expectedSize.
//==============================================================================
MixedPhasePersistentCache::ArtifactState
MixedPhasePersistentCache::classifyArtifact(const juce::File& file) noexcept
{
    // (1) must live inside the v3 namespace directory
    {
        const auto dir = getCacheDirectory().getFullPathName();
        if (! file.getFullPathName().startsWith (dir))
            return ArtifactState::outsideV3Namespace;
    }

    // (2) staging / temporary: canonical artifacts are *.mph
    if (file.getFileExtension() != ".mph")
        return ArtifactState::stagingOrTemporary;

    const auto actualSize = static_cast<std::int64_t>(file.getSize());
    if (actualSize < static_cast<std::int64_t>(sizeof(DiskHeader)))
        return ArtifactState::malformedOrTruncated;

    juce::FileInputStream stream(file);
    if (! stream.openedOk())
        return ArtifactState::malformedOrTruncated;

    DiskHeader header;
    if (stream.read (&header, sizeof(DiskHeader)) != static_cast<int64> (sizeof(DiskHeader)))
        return ArtifactState::malformedOrTruncated;

    // (4) magic
    if (header.magic != kMagic)
        return ArtifactState::foreign;
    // (5) version
    if (header.version > kVersion)
        return ArtifactState::future;
    if (header.version < kVersion)
        return ArtifactState::obsolete;   // v1 / v2

    // (6) dimensions: int64 widen, then predicates 1..7
    std::int64_t expectedSize = 0;
    if (! validateCanonicalDims (static_cast<std::int64_t> (header.targetLength),
                                static_cast<std::int64_t> (header.numSamples),
                                static_cast<std::int64_t> (header.numChannels),
                                static_cast<std::int64_t> (header.numAllpassSections),
                                kMaxIrLatencyBound,
                                kMaxIrChannelsBound,
                                expectedSize))
        return ArtifactState::malformedOrTruncated;

    // (7) checked expected size == actual file size
    if (actualSize != expectedSize)
        return ArtifactState::exactSizeMismatch;
    // (8) header.totalFileSize == expectedSize
    if (static_cast<std::int64_t> (header.totalFileSize) != expectedSize)
        return ArtifactState::exactSizeMismatch;

    return ArtifactState::currentV3Canonical;
}

juce::File MixedPhasePersistentCache::getCacheDirectory()
{
    // Phase 2 (Owner D-1R-2AJ section 3): the canonical v3 namespace. The v2
    // directory (%APPDATA%/ConvoPeq/MixedPhaseCache/) is never referenced, so
    // v2 artifacts are never auto-loaded, migrated, auto-deleted, nor LRU
    // candidates. Every public path funnels through here (the single sink), so
    // this is the only place that selects the namespace.
    auto dir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                   .getChildFile("ConvoPeq")
                   .getChildFile(kCacheNamespaceV3);
    if (!dir.exists())
     {
         auto result = dir.createDirectory();
         if (!result.wasOk())
             juce::Logger::writeToLog("Warning: Could not create cache directory");
     }
    return dir;
}

juce::File MixedPhasePersistentCache::getCacheFile(uint64_t fileHash,
                                                    double sampleRate,
                                                    int phaseMode,
                                                    float freqStartHz, float freqEndHz,
                                                    int targetLength)
{
    const auto hash = computeKeyHash(fileHash, sampleRate, phaseMode,
                                     freqStartHz, freqEndHz, targetLength);
    const auto filename = juce::String::toHexString(static_cast<int64_t>(hash)) + ".mph";
    return getCacheDirectory().getChildFile(filename);
}

// ═══════════════════════════════════════════════════════════════
//  読み込み
// ═══════════════════════════════════════════════════════════════

bool MixedPhasePersistentCache::load(uint64_t fileHash,
                                     double sampleRate,
                                     int phaseMode,
                                     float freqStartHz, float freqEndHz,
                                     int targetLength,
                                     juce::AudioBuffer<double>& outIr,
                                     std::vector<double>& outRho,
                                     std::vector<double>& outTheta)
{
    const auto file = getCacheFile(fileHash, sampleRate, phaseMode,
                                   freqStartHz, freqEndHz, targetLength);
    if (!file.existsAsFile())
        return false;

    juce::FileInputStream stream(file);
    if (!stream.openedOk())
        return false;

    // ヘッダ読み込み
    DiskHeader header;
    if (stream.read(&header, sizeof(DiskHeader)) != static_cast<int64>(sizeof(DiskHeader)))
        return false;

    // ヘッダ検証
    if (header.magic != kMagic || header.version != kVersion)
        return false;
    if (header.fileHash != fileHash
        || std::abs(header.sampleRate - sampleRate) > 1.0e-6
        || header.phaseMode != static_cast<int32_t>(phaseMode)
        || std::abs(header.freqStartHz - freqStartHz) > 1.0e-6f
        || std::abs(header.freqEndHz - freqEndHz) > 1.0e-6f
        || header.targetLength != static_cast<int32_t>(targetLength))
        return false;

    // 波形データ読み込み
    const std::int64_t targetLength64 = static_cast<std::int64_t>(header.targetLength);
    const std::int64_t numSamples64 = static_cast<std::int64_t>(header.numSamples);
    const std::int64_t numChannels64 = static_cast<std::int64_t>(header.numChannels);
    const std::int64_t numSec64 = static_cast<std::int64_t>(header.numAllpassSections);

    // Phase 2 (Owner D-1R-2AJ P2-I3): canonical predicate 1..7 in the int64
    // domain, run BEFORE any allocation and BEFORE any narrowing.
    std::int64_t expectedSize = 0;
    if (!validateCanonicalDims(targetLength64, numSamples64, numChannels64, numSec64,
                               kMaxIrLatencyBound,
                               static_cast<std::int64_t>(kMaxIrChannelsBound),
                               expectedSize))
        return false;

    // Exact-size validation (predicate 8): header / actual / expected, three-way.
    const std::int64_t actualSize = static_cast<std::int64_t>(file.getSize());
    if (actualSize != expectedSize)
        return false;
    if (static_cast<std::int64_t>(header.totalFileSize) != expectedSize)
        return false;

    // ---- allocation boundary: every rejection above happens before this ----
    const int numChannels = static_cast<int>(numChannels64);
    const int numSamples = static_cast<int>(numSamples64);
    const int numSec = static_cast<int>(numSec64);

    outIr.setSize(numChannels, numSamples);
    for (int ch = 0; ch < numChannels; ++ch)
    {
        double* data = outIr.getWritePointer(ch);
        const auto bytesToRead = static_cast<int64>(static_cast<size_t>(numSamples) * sizeof(double));
        const auto bytesRead = stream.read(data, static_cast<size_t>(numSamples) * sizeof(double));
        if (bytesRead != bytesToRead)
            return false;
    }

    // Allpass セクション読み込み
    outRho.resize(static_cast<size_t>(numSec));
    outTheta.resize(static_cast<size_t>(numSec));

    if (numSec > 0)
    {
        const auto secBytes = static_cast<int64>(static_cast<size_t>(numSec) * sizeof(double));
        if (stream.read(outRho.data(), static_cast<size_t>(numSec) * sizeof(double)) != secBytes)
            return false;
        if (stream.read(outTheta.data(), static_cast<size_t>(numSec) * sizeof(double)) != secBytes)
            return false;
    }

    // LRU用にタイムスタンプ更新（v3 Phase 1: in-place 8-byte write）
    // touch() public API は削除し、load() 内の NonRT metadata update に統合。
    // DiskHeader::lastUsedTime の 8 byte のみを書き換える。
    // setPosition(offsetof(...)) → write(8) → flush() のみを行い、
    // truncate() は使用しない（payload / file size は変更しない）。
    // timestamp write / flush failure は diagnostic のみ（load() == true を維持）。
    {
        const auto touchFile = getCacheFile(fileHash, sampleRate, phaseMode,
                                            freqStartHz, freqEndHz, targetLength);
        if (touchFile.existsAsFile())
        {
            juce::FileOutputStream touchStream(touchFile);
            if (touchStream.openedOk())
            {
                const auto lastUsedTimeOffset = offsetof(DiskHeader, lastUsedTime);
                const auto fileSize = touchFile.getSize();
                if (lastUsedTimeOffset + static_cast<decltype(fileSize)>(sizeof(uint64_t)) <= fileSize)
                {
                    touchStream.setPosition(static_cast<int64>(lastUsedTimeOffset));
                    const uint64_t now = static_cast<uint64_t>(juce::Time::getMillisecondCounter());

                    // Diagnostic-only metadata update. Neither the write result
                    // nor the flush result may change load()'s return value, the
                    // payload bytes, or the file size.
#if defined(CONVOPEQ_PCACHE_TEST_SEAM)
                    const auto fault = testSeamState().fault;
                    if (fault == TimestampWriteFault::timestampWrite)
                    {
                        // Simulated OS-level write failure: the 8 bytes are not
                        // written at all. Nothing else in the file is touched.
                        ++testSeamState().attempts;
                    }
                    else if (fault == TimestampWriteFault::flush)
                    {
                        // write() succeeds, but the explicit flush is reported
                        // as failed and is therefore not issued. The 8 bytes
                        // stay in the stream buffer and are never committed by
                        // this metadata update.
                        if (touchStream.write(&now, sizeof(now)))
                            ++testSeamState().attempts;
                    }
                    else
#endif
                    {
                        (void) touchStream.write(&now, sizeof(now));
                        (void) touchStream.flush();
                    }
                }
            }
        }
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════
//  保存
// ═══════════════════════════════════════════════════════════════

bool MixedPhasePersistentCache::save(uint64_t fileHash,
                                     double sampleRate,
                                     int phaseMode,
                                     float freqStartHz, float freqEndHz,
                                     int targetLength,
                                     const juce::AudioBuffer<double>& ir,
                                     const std::vector<double>& rho,
                                     const std::vector<double>& theta)
{
    const auto file = getCacheFile(fileHash, sampleRate, phaseMode,
                                   freqStartHz, freqEndHz, targetLength);
    const auto dir = file.getParentDirectory();
    if (!dir.exists())
     {
         auto result = dir.createDirectory();
         if (!result.wasOk())
             juce::Logger::writeToLog("Warning: Could not create cache directory");
     }

    juce::TemporaryFile tempFile(file);
    {
        juce::FileOutputStream stream(tempFile.getFile());
        if (!stream.openedOk())
            return false;

        // Phase 2 (Owner D-1R-2AJ P2-I4): widen to the int64 domain, run the
        // canonical validator, and only then narrow into the int32 DiskHeader
        // fields. rho.size() is never cast to int32 before validation.
        const std::int64_t numChannels64 = static_cast<std::int64_t>(ir.getNumChannels());
        const std::int64_t numSamples64 = static_cast<std::int64_t>(ir.getNumSamples());
        const std::int64_t numSec64 = static_cast<std::int64_t>(rho.size());
        const std::int64_t targetLength64 = static_cast<std::int64_t>(targetLength);

        std::int64_t expectedSize = 0;
        if (!validateCanonicalDims(targetLength64, numSamples64, numChannels64, numSec64,
                                   kMaxIrLatencyBound,
                                   static_cast<std::int64_t>(kMaxIrChannelsBound),
                                   expectedSize))
            return false;

        const int numChannels = static_cast<int>(numChannels64);
        const int numSamples = static_cast<int>(numSamples64);
        const int numSec = static_cast<int>(numSec64);

        // ヘッダ書き込み
        DiskHeader header;
        std::memset(&header, 0, sizeof(header));
        header.magic = kMagic;
        header.version = kVersion;
        header.fileHash = fileHash;
        header.sampleRate = sampleRate;
        header.phaseMode = static_cast<int32_t>(phaseMode);
        header.freqStartHz = freqStartHz;
        header.freqEndHz = freqEndHz;
        header.targetLength = static_cast<int32_t>(targetLength);
        header.lastUsedTime = static_cast<uint64_t>(juce::Time::getMillisecondCounter());
        header.numChannels = static_cast<int32_t>(numChannels);
        header.numSamples = static_cast<int32_t>(numSamples);
        header.numAllpassSections = static_cast<int32_t>(numSec);

        if (stream.write(&header, sizeof(DiskHeader)) != true)
            return false;

        // 波形データ書き込み
        for (int ch = 0; ch < numChannels; ++ch)
        {
            const double* data = ir.getReadPointer(ch);
            if (stream.write(data, static_cast<size_t>(numSamples) * sizeof(double)) != true)
                return false;
        }

        // Allpass セクション書き込み
        if (numSec > 0)
        {
            if (stream.write(rho.data(), static_cast<size_t>(numSec) * sizeof(double)) != true)
                return false;
            if (stream.write(theta.data(), static_cast<size_t>(numSec) * sizeof(double)) != true)
                return false;
        }

        // ファイルサイズをヘッダに書き戻し
        header.totalFileSize = stream.getPosition();
        stream.setPosition(0);
        if (stream.write(&header, sizeof(DiskHeader)) != true)
            return false;

        stream.flush();
    }

    return tempFile.overwriteTargetFileWithTemporary();
}

// ═══════════════════════════════════════════════════════════════
//  LRU 管理
// ═══════════════════════════════════════════════════════════════

void MixedPhasePersistentCache::evictLRU(size_t maxCount)
{
    // Phase 2 (Owner D-1R-2AJ section 5): evictLRU(0) is a complete no-op and
    // must not delegate to clear(). 0 is never reinterpreted as "delete all".
    if (maxCount == 0)
        return;

    auto dir = getCacheDirectory();
    juce::Array<juce::File> files;
    dir.findChildFiles(files, juce::File::findFiles, false, "*.mph");

    // Only CurrentV3Canonical artifacts are candidates. Everything else
    // (staging/temporary, malformed, exact-size mismatch, foreign, future,
    // obsolete incl. v1/v2, outside the v3 namespace) is left untouched and is
    // NOT auto-deleted. The candidate count is therefore a subset of files.size().
    std::vector<std::pair<uint64_t, juce::File>> timedFiles;
    timedFiles.reserve(static_cast<size_t>(files.size()));

    for (const auto& f : files)
    {
        if (!isCurrentV3Canonical(classifyArtifact(f)))
            continue;   // leave non-canonical artifacts in place

        juce::FileInputStream stream(f);
        if (!stream.openedOk())
            continue;

        DiskHeader header;
        if (stream.read(&header, sizeof(DiskHeader)) != static_cast<int64>(sizeof(DiskHeader)))
            continue;

        timedFiles.emplace_back(header.lastUsedTime, f);
    }

    if (static_cast<size_t>(timedFiles.size()) <= maxCount)
        return;

    // Evict oldest-lastUsedTime CurrentV3Canonical artifacts until the
    // canonical count fits maxCount.
    std::sort(timedFiles.begin(), timedFiles.end(),
    [](const auto& a, const auto& b) { return a.first < b.first; });

    const int toRemove = static_cast<int>(timedFiles.size()) - static_cast<int>(maxCount);
    for (int i = 0; i < toRemove; ++i)
        timedFiles[static_cast<size_t>(i)].second.deleteFile();
}

void MixedPhasePersistentCache::remove(uint64_t fileHash,
                                       double sampleRate,
                                       int phaseMode,
                                       float freqStartHz, float freqEndHz,
                                       int targetLength)
{
    const auto file = getCacheFile(fileHash, sampleRate, phaseMode,
                                   freqStartHz, freqEndHz, targetLength);
    if (file.existsAsFile())
        file.deleteFile();
}

void MixedPhasePersistentCache::clear()
{
    // Phase 2 (Owner D-1R-2AJ section 5.3): clear() stays a public API but is
    // limited to canonical cache artifacts inside the v3 namespace. The v2
    // directory is never touched, and staging artifacts with another extension
    // are not additionally deleted.
    auto dir = getCacheDirectory();
    if (!dir.exists())
        return;

    juce::Array<juce::File> files;
    dir.findChildFiles(files, juce::File::findFiles, false, "*.mph");
    for (auto& f : files)
    {
        if (isCurrentV3Canonical(classifyArtifact(f)))
            f.deleteFile();
    }
}

size_t MixedPhasePersistentCache::getEntryCount()
{
    // Phase 2 (Owner D-1R-2AJ section 5.2): getEntryCount() is the
    // CurrentV3Canonical semantic count. Staging / malformed / future / obsolete
    // artifacts are NOT counted.
    auto dir = getCacheDirectory();
    if (!dir.exists())
        return 0;

    juce::Array<juce::File> files;
    dir.findChildFiles(files, juce::File::findFiles, false, "*.mph");

    size_t canonical = 0;
    for (const auto& f : files)
        if (isCurrentV3Canonical(classifyArtifact(f)))
            ++canonical;

    return canonical;
}

} // namespace convo
