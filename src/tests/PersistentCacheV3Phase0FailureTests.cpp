// ============================================================================
//  P0-6  -  PersistentCache v3 Phase 0 failure-injection harness
// ----------------------------------------------------------------------------
//  Owner instruction "Owner reply - D-1R-2K Phase 0 GO / Production
//  Implementation NO-GO" section 8, item P0-6, and section 2.3 (Y-3):
//
//    Y-3 (production FaultPoint seam) is NOT approved. The Owner lists the
//    preferred, production-change-free techniques, in order:
//      1. test-owned temporary directory / artifact fixture      <- used here
//      2. filesystem wrapper / dependency injection in the test TU
//      3. JUCE File existing behaviour for permission / truncated /
//         replacement failure fixtures                        <- used here
//      4. minimal compile-time guarded seam ONLY if sequences remain
//         unreproducible                                      <- NOT needed
//
//  This TU therefore adds NO seam and changes NO production source. Failure
//  sequences are produced with:
//      * a private fileHash namespace so only test-owned artifacts are touched
//      * juce::File::setReadOnly()  -> permission failure fixture
//      * juce::FileOutputStream with truncated content -> partial payload fixture
//      * a read-only cache directory -> replacement / commit failure fixture
//      * std::thread for the concurrency sequences
//
//  SAFETY: this test NEVER calls clear() or evictLRU(0) against the live cache,
//  because the current production implementation maps evictLRU(0) to clear()
//  (D-1R-2K K-03 / K-35) and that would delete the user's real cache entries.
//  The current behaviour is asserted by reading the production source text
//  instead (see verifyCurrentProductionContract).
// ============================================================================

#include <JuceHeader.h>

#include "MixedPhasePersistentCache.h"

#include <atomic>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cstring>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what)
{
    ++g_checks;
    if (!ok)
    {
        ++g_failures;
        std::fprintf(stderr, "FAIL  %s\n", what.c_str());
    }
}

void checkEq(long long got, long long want, const std::string& what)
{
    ++g_checks;
    if (got != want)
    {
        ++g_failures;
        std::fprintf(stderr, "FAIL  %s : got=%lld want=%lld\n",
                     what.c_str(), got, want);
    }
}

// ---------------------------------------------------------------------------
// Private identity namespace. Test artifacts are addressed by these hashes only.
// ---------------------------------------------------------------------------
constexpr std::uint64_t kTestHashBase = 0xD1B2C0DE00000001ull;   // Phase 0 marker
std::uint64_t testHash(int index)
{
    return kTestHashBase + static_cast<std::uint64_t>(index) * 0x100000000ull;
}

constexpr double    kTestSr = 48000.0;
constexpr int       kTestPhaseMode = 1;      // PhaseMode::Mixed
constexpr float     kTestF1 = 100.0f;
constexpr float     kTestF2 = 700.0f;
constexpr int       kTestTargetLength = 480; // 48000 * 1.0 s / 100 (small fixture)
constexpr int       kTestChannels = 2;
constexpr int       kTestSections = 4;

/// RAII cleanup: removes only artifacts created by this TU.
struct ArtifactGuard
{
    std::vector<int> indices;

    void track(int i) { indices.push_back(i); }

    ~ArtifactGuard()
    {
        for (int i : indices)
            convo::MixedPhasePersistentCache::remove(testHash(i), kTestSr,
                                                     kTestPhaseMode, kTestF1,
                                                     kTestF2, kTestTargetLength);
    }
};

juce::File fileOf(int index)
{
    return convo::MixedPhasePersistentCache::getCacheFile(
        testHash(index), kTestSr, kTestPhaseMode, kTestF1, kTestF2, kTestTargetLength);
}

juce::AudioBuffer<double> makeIr()
{
    juce::AudioBuffer<double> ir(kTestChannels, kTestTargetLength);
    for (int ch = 0; ch < kTestChannels; ++ch)
        for (int i = 0; i < kTestTargetLength; ++i)
            ir.setSample(ch, i, 0.25 * std::sin(0.01 * static_cast<double>(i))
                                    + 0.001 * static_cast<double>(ch));
    return ir;
}

std::vector<double> makeSections()
{
    return { 0.5, 0.4, 0.3, 0.2 };
}

std::int64_t expectedSizeOf(int channels, int samples, int sections)
{
    return 76 + static_cast<std::int64_t>(channels) * samples * 8
              + static_cast<std::int64_t>(sections) * 8 * 2;
}

// ===========================================================================
//  Current production contract, asserted by reading the production source.
//  This is how we observe evictLRU(0) -> clear() without executing it.
// ===========================================================================
std::string readRepoFile(const char* rel)
{
    namespace fs = std::filesystem;
    const fs::path p(rel);
    for (const auto& prefix : { fs::path("."), fs::path(".."),
                                fs::path("../.."), fs::path("../../..") })
    {
        const fs::path cand = prefix / p;
        std::error_code ec;
        if (fs::exists(cand, ec))
        {
            std::ifstream in(cand, std::ios::in | std::ios::binary);
            std::ostringstream oss;
            oss << in.rdbuf();
            return oss.str();
        }
    }
    return {};
}

bool contains(const std::string& h, const std::string& n)
{
    return h.find(n) != std::string::npos;
}

void verifyCurrentProductionContract()
{
    std::fprintf(stderr, "[P0-6a] current production contract (source-level)\n");

    const std::string src = readRepoFile("src/MixedPhasePersistentCache.cpp");
    check(!src.empty(), "P0-6a production source is readable");

    // Owner section 4.1: evictLRU(0) must become a no-op in v3.
    // Current production routes 0 to clear(); this is the CONTRADICTED
    // condition that Phase 0 records (it is NOT executed here).
    check(contains(src, "if (maxCount == 0)") && contains(src, "clear();"),
          "P0-6a CONFIRMED-BAD: current evictLRU(0) delegates to clear()");

    // Phase 1 (D-1R-2M): touch() は削除され、load() 内の in-place 8-byte
    // timestamp write に置き換えられた。旧 append 破損（N -> 2N）は解消済み。
    check(contains(src, "touchStream.setPosition"),
          "P0-6a Phase1: load() integrates an in-place timestamp write");
    check(!contains(src, "outStream.write(buffer.data(), buffer.size())"),
          "P0-6a Phase1: legacy full-buffer append is gone");
    check(!contains(src, "touch(fileHash,"),
          "P0-6a Phase1: touch() public API removed");

    // D-1R-2K K-26: load() has no upper bound before setSize.
    check(contains(src, "outIr.setSize(numChannels, numSamples);")
              && !contains(src, "numSamples > 2097152")
              && !contains(src, "numSamples > MAX_IR_LATENCY"),
          "P0-6a CONFIRMED-BAD: load() has no numSamples upper bound before setSize");

    // D-1R-2K K-25: load() never compares header.targetLength with header.numSamples.
    check(!contains(src, "header.targetLength != header.numSamples")
              && !contains(src, "header.numSamples != header.targetLength"),
          "P0-6a CONFIRMED-BAD: load() does not verify targetLength == numSamples");

    // D-1R-2K K-20: save() narrows without domain validation.
    check(contains(src, "header.targetLength = static_cast<int32_t>(targetLength);")
              && !contains(src, "isAdmissibleDomain"),
          "P0-6a CONFIRMED-BAD: save() narrows without a domain predicate");

    // v3 single-authority predicate must not exist yet (production is NO-GO).
    check(!contains(src, "isAdmissibleDomain"),
          "P0-6a v3 admissibility predicate is not in production yet (expected)");

    std::fprintf(stderr, "       5 CONTRADICTED production conditions recorded\n");
}

// ===========================================================================
//  FS-1 : header success -> payload success -> timestamp update failure
//  The timestamp update is made to fail with a read-only file attribute.
//  Contract: load() must still return true and the artifact must stay usable.
//
//  This test also VERIFIES Phase 1's corrected timestamp update:
//  a successful load() rewrite must keep the artifact byte-identical in
//  size and must NOT append (no N -> 2N doubling).
// ===========================================================================
void runFS1(ArtifactGuard& guard)
{
    std::fprintf(stderr, "[P0-6b] FS-1 timestamp-update failure\n");

    constexpr int kIdx = 1;
    guard.track(kIdx);

    const auto ir = makeIr();
    const auto sec = makeSections();

    check(convo::MixedPhasePersistentCache::save(testHash(kIdx), kTestSr,
                                                 kTestPhaseMode, kTestF1, kTestF2,
                                                 kTestTargetLength, ir, sec, sec),
          "P0-6b save() succeeds");

    auto f = fileOf(kIdx);
    const auto sizeAfterSave = f.getSize();
    checkEq(static_cast<long long>(sizeAfterSave),
            expectedSizeOf(kTestChannels, kTestTargetLength, kTestSections),
            "P0-6b saved artifact has the expected exact size");

    // ---- FS-1 byte-level baseline (Owner §3.4) ---------------------------
    std::vector<char> bytesBefore;
    {
        std::ifstream in(f.getFullPathName().toStdString(), std::ios::in | std::ios::binary);
        check(in.good(), "P0-6b baseline byte read opens");
        bytesBefore.resize(static_cast<std::size_t>(sizeAfterSave));
        in.read(bytesBefore.data(), static_cast<std::streamsize>(sizeAfterSave));
    }

    // ---- FS-1: timestamp update fails (read-only attribute) --------------
    const bool roOk = f.setReadOnly(true);
    check(roOk, "P0-6b artifact marked read-only (timestamp update will fail)");

    juce::AudioBuffer<double> out(kTestChannels, kTestTargetLength);
    std::vector<double> rho, theta;
    const bool loaded = convo::MixedPhasePersistentCache::load(
        testHash(kIdx), kTestSr, kTestPhaseMode, kTestF1, kTestF2,
        kTestTargetLength, out, rho, theta);

    // Owner section 6: a timestamp failure must NOT turn a valid artifact
    // into load() == false.
    check(loaded, "P0-6b load() returns TRUE despite timestamp update failure");
    checkEq(out.getNumSamples(), kTestTargetLength,
            "P0-6b payload length intact after timestamp failure");
    checkEq(static_cast<long long>(rho.size()), kTestSections,
            "P0-6b allpass section count intact");
    checkEq(static_cast<long long>(f.getSize()), static_cast<long long>(sizeAfterSave),
            "P0-6b failed timestamp update leaves the artifact byte-identical");

    f.setReadOnly(false);

    // ---- current production behaviour when the update SUCCEEDS -----------
    juce::AudioBuffer<double> out2(kTestChannels, kTestTargetLength);
    std::vector<double> rho2, theta2;
    const bool loaded2 = convo::MixedPhasePersistentCache::load(
        testHash(kIdx), kTestSr, kTestPhaseMode, kTestF1, kTestF2,
        kTestTargetLength, out2, rho2, theta2);
    check(loaded2, "P0-6b second load() succeeds");

    const auto sizeAfterTouch = f.getSize();
    std::fprintf(stderr, "       size after save=%lld  after successful touch=%lld\n",
                 static_cast<long long>(sizeAfterSave),
                 static_cast<long long>(sizeAfterTouch));

    // Phase 1 in-place write keeps the artifact at its exact canonical size
    // (no N -> 2N doubling). Owner P1 acceptance criteria:
    //   "truncate() は使用禁止", "file size が touch 前後で完全一致",
    //   "payload byte equality", "timestamp failure/flush failure -> load() == true".
    checkEq(static_cast<long long>(sizeAfterTouch),
            static_cast<long long>(sizeAfterSave),
            "P0-6b Phase1: load() leaves the artifact size unchanged (no N -> 2N)");
    checkEq(static_cast<long long>(sizeAfterTouch),
            static_cast<long long>(expectedSizeOf(kTestChannels, kTestTargetLength, kTestSections)),
            "P0-6b Phase1: artifact size stays at the exact canonical size");

    // ---- FS-1 byte-level payload comparison (Owner §3.4) ------------------
    // bytes [0,48) unchanged, bytes [48,56) may change (timestamp),
    // bytes [56,76) unchanged, bytes [76,end) unchanged.
    std::vector<char> bytesAfter;
    {
        std::ifstream in(f.getFullPathName().toStdString(), std::ios::in | std::ios::binary);
        check(in.good(), "P0-6b post-load byte read opens");
        bytesAfter.resize(static_cast<std::size_t>(sizeAfterTouch));
        in.read(bytesAfter.data(), static_cast<std::streamsize>(sizeAfterTouch));
    }
    checkEq(bytesAfter.size(), bytesBefore.size(),
            "P0-6b payload byte-equality: same file size");

    const std::size_t kTotal = static_cast<std::size_t>(sizeAfterTouch);
    auto cmpRange = [&](std::size_t off, std::size_t len) -> bool
    {
        if (off + len > kTotal) return false;
        return std::memcmp(bytesBefore.data() + off, bytesAfter.data() + off, len) == 0;
    };
    check(cmpRange(0, 48),
          "P0-6b bytes [0,48) unchanged");
    check(cmpRange(56, 76 - 56),
          "P0-6b bytes [56,76) unchanged (header tail)");
    check(cmpRange(76, kTotal - 76),
          "P0-6b payload bytes [76,end) unchanged");

    // The 8 timestamp bytes [48,56) may differ; if size == size,
    // those 8 bytes are the ONLY thing that changed.
    int changedBytes = 0;
    for (std::size_t i = 0; i < kTotal; ++i)
        if (bytesBefore[i] != bytesAfter[i])
            ++changedBytes;
    check(changedBytes <= 8,
          "P0-6b at most the 8 timestamp bytes changed");
}

// ===========================================================================
//  FS-2 : header success -> partial payload -> failure
//  Contract: load() returns false AND the artifact bytes are untouched.
//
//  NOTE on the fixture: juce::FileOutputStream opens with OPEN_ALWAYS plus
//  SetFilePointer(FILE_END), so its position starts at end-of-file and
//  truncate() becomes a no-op. The partial artifact is therefore produced by
//  rewriting only a prefix of the saved bytes, which keeps the 76-byte header
//  intact and removes part of the payload.
// ===========================================================================
void runFS2(ArtifactGuard& guard)
{
    std::fprintf(stderr, "[P0-6c] FS-2 partial payload\n");

    constexpr int kIdx = 2;
    guard.track(kIdx);

    const auto ir = makeIr();
    const auto sec = makeSections();
    check(convo::MixedPhasePersistentCache::save(testHash(kIdx), kTestSr,
                                                 kTestPhaseMode, kTestF1, kTestF2,
                                                 kTestTargetLength, ir, sec, sec),
          "P0-6c save() succeeds");

    auto f = fileOf(kIdx);
    const auto sizeBefore = f.getSize();

    // Read the whole artifact, then write back only a prefix.
    std::vector<char> bytes(static_cast<std::size_t>(sizeBefore));
    {
        std::ifstream in(f.getFullPathName().toStdString(), std::ios::in | std::ios::binary);
        check(in.good(), "P0-6c artifact opened for the partial-payload fixture");
        in.read(bytes.data(), static_cast<std::streamsize>(sizeBefore));
    }

    // Keep the full 76-byte header plus half of the payload.
    const std::size_t keep = 76 + bytes.size() / 2;
    f.deleteFile();
    {
        std::ofstream out(f.getFullPathName().toStdString(),
                          std::ios::out | std::ios::binary | std::ios::trunc);
        check(out.good(), "P0-6c partial artifact opened for writing");
        out.write(bytes.data(), static_cast<std::streamsize>(keep));
    }

    const auto sizeAfterTrunc = f.getSize();
    std::fprintf(stderr, "       size before=%lld  after partial rewrite=%lld\n",
                 static_cast<long long>(sizeBefore),
                 static_cast<long long>(sizeAfterTrunc));
    check(sizeAfterTrunc < sizeBefore && sizeAfterTrunc >= 76,
          "P0-6c artifact truncated to a header-complete, payload-incomplete prefix");

    juce::AudioBuffer<double> out2(kTestChannels, kTestTargetLength);
    std::vector<double> rho, theta;
    const bool loaded = convo::MixedPhasePersistentCache::load(
        testHash(kIdx), kTestSr, kTestPhaseMode, kTestF1, kTestF2,
        kTestTargetLength, out2, rho, theta);

    check(!loaded, "P0-6c load() returns FALSE on a partial payload");
    checkEq(static_cast<long long>(f.getSize()), static_cast<long long>(sizeAfterTrunc),
            "P0-6c partial artifact is left byte-identical by load()");
}

// ===========================================================================
//  FS-3 : canonical replacement failure
//  A read-only cache directory makes the staging commit fail.
//  Contract: save() returns false and the previous artifact stays intact.
// ===========================================================================
void runFS3(ArtifactGuard& guard)
{
    std::fprintf(stderr, "[P0-6d] FS-3 canonical replacement failure\n");

    constexpr int kIdx = 3;
    guard.track(kIdx);

    const auto ir = makeIr();
    const auto sec = makeSections();
    check(convo::MixedPhasePersistentCache::save(testHash(kIdx), kTestSr,
                                                 kTestPhaseMode, kTestF1, kTestF2,
                                                 kTestTargetLength, ir, sec, sec),
          "P0-6d initial save() succeeds");

    auto f = fileOf(kIdx);
    const auto sizeBefore = f.getSize();
    auto dir = convo::MixedPhasePersistentCache::getCacheDirectory();

    // Make the target un-writable: read-only file blocks the replacement.
    f.setReadOnly(true);

    auto ir2 = ir;
    ir2.applyGain(0.5);
    const bool secondSave = convo::MixedPhasePersistentCache::save(
        testHash(kIdx), kTestSr, kTestPhaseMode, kTestF1, kTestF2,
        kTestTargetLength, ir2, sec, sec);

    check(!secondSave, "P0-6d save() returns FALSE when replacement fails");
    checkEq(static_cast<long long>(f.getSize()), static_cast<long long>(sizeBefore),
            "P0-6d previous artifact stays byte-identical after a failed save");

    f.setReadOnly(false);
    (void) dir;
}

// ===========================================================================
//  FS-4 / FS-5 / FS-6 : concurrency
//  Contract: every identity is either fully valid or absent. Never partial.
// ===========================================================================
void runFS4To6(ArtifactGuard& guard)
{
    std::fprintf(stderr, "[P0-6e] FS-4/5/6 concurrency\n");

    constexpr int kBase = 10;
    constexpr int kIdentities = 6;
    constexpr int kIterations = 40;

    for (int i = 0; i < kIdentities; ++i)
        guard.track(kBase + i);

    const auto ir = makeIr();
    const auto sec = makeSections();

    std::atomic<int> loadsOk{0};
    std::atomic<int> partials{0};

    // FS-4: concurrent save -> load
    {
        std::vector<std::thread> ts;
        for (int i = 0; i < kIdentities; ++i)
        {
            ts.emplace_back([&, i] {
                for (int it = 0; it < kIterations; ++it)
                {
                    convo::MixedPhasePersistentCache::save(
                        testHash(kBase + i), kTestSr, kTestPhaseMode, kTestF1,
                        kTestF2, kTestTargetLength, ir, sec, sec);
                    juce::AudioBuffer<double> out(kTestChannels, kTestTargetLength);
                    std::vector<double> rho, theta;
                    if (convo::MixedPhasePersistentCache::load(
                            testHash(kBase + i), kTestSr, kTestPhaseMode, kTestF1,
                            kTestF2, kTestTargetLength, out, rho, theta))
                    {
                        // A successful load must yield a complete payload.
                        if (out.getNumSamples() == kTestTargetLength
                            && static_cast<int>(rho.size()) == kTestSections)
                            ++loadsOk;
                        else
                            ++partials;
                    }
                }
            });
        }
        for (auto& t : ts) t.join();
    }

    check(partials.load() == 0,
          "P0-6e no successful load ever returned a partial payload");
    check(loadsOk.load() > 0,
          "P0-6e concurrent save/load produced successful complete loads");

    std::fprintf(stderr, "       concurrent save->load: ok=%d partial=%d\n",
                 loadsOk.load(), partials.load());

    // ---- After the stress, every artifact must still validate cleanly ----
    int valid = 0;
    for (int i = 0; i < kIdentities; ++i)
    {
        juce::AudioBuffer<double> out(kTestChannels, kTestTargetLength);
        std::vector<double> rho, theta;
        if (convo::MixedPhasePersistentCache::load(
                testHash(kBase + i), kTestSr, kTestPhaseMode, kTestF1, kTestF2,
                kTestTargetLength, out, rho, theta))
            ++valid;
    }
    check(valid == kIdentities,
          "P0-6e all concurrent identities remain loadable after the stress");
    std::fprintf(stderr, "       post-stress valid identities: %d / %d\n",
                 valid, kIdentities);
}

// ===========================================================================
//  Producer invariant: header.targetLength == header.numSamples on disk.
//  Read back the raw bytes to prove the producer actually writes them equal.
// ===========================================================================
void runProducerInvariant(ArtifactGuard& guard)
{
    std::fprintf(stderr, "[P0-6f] producer invariant on disk\n");

    constexpr int kIdx = 4;
    guard.track(kIdx);

    const auto ir = makeIr();          // 2 channels x 480 samples
    const auto sec = makeSections();   // 4 sections
    check(convo::MixedPhasePersistentCache::save(testHash(kIdx), kTestSr,
                                                 kTestPhaseMode, kTestF1, kTestF2,
                                                 kTestTargetLength, ir, sec, sec),
          "P0-6f save() succeeds");

    auto f = fileOf(kIdx);
    const std::string path = f.getFullPathName().toStdString();
    std::ifstream in(path, std::ios::in | std::ios::binary);
    check(in.good(), "P0-6f artifact opened for raw header read");

    unsigned char hdr[76] = {};
    in.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    checkEq(static_cast<long long>(in.gcount()), 76LL,
            "P0-6f 76-byte header read");

    auto rd32 = [&](int off) -> std::int32_t
    {
        std::uint32_t v = 0;
        for (int b = 3; b >= 0; --b)
            v = (v << 8) | hdr[off + b];
        return static_cast<std::int32_t>(v);
    };
    auto rd64 = [&](int off) -> std::int64_t
    {
        std::uint64_t v = 0;
        for (int b = 7; b >= 0; --b)
            v = (v << 8) | hdr[off + b];
        return static_cast<std::int64_t>(v);
    };

    // layout: magic0 version8 headerReserved12 fileHash16 sampleRate24
    //         phaseMode32 freqStart36 freqEnd40 targetLength44
    //         lastUsedTime48 numChannels56 numSamples60 numSections64 total68
    const std::int32_t targetLength = rd32(44);
    const std::int32_t numChannels   = rd32(56);
    const std::int32_t numSamples   = rd32(60);
    const std::int32_t numSections  = rd32(64);
    const std::int64_t totalSize    = rd64(68);

    std::fprintf(stderr, "       header: targetLength=%d numSamples=%d ch=%d sec=%d total=%lld\n",
                 targetLength, numSamples, numChannels, numSections,
                 static_cast<long long>(totalSize));

    checkEq(targetLength, numSamples,
            "P0-6f on-disk header.targetLength == header.numSamples");
    checkEq(targetLength, kTestTargetLength,
            "P0-6f header.targetLength equals the requested trim length");
    checkEq(numChannels, kTestChannels, "P0-6f header.numChannels as written");
    checkEq(numSections, kTestSections, "P0-6f header.numAllpassSections as written");
    checkEq(totalSize, expectedSizeOf(kTestChannels, kTestTargetLength, kTestSections),
            "P0-6f header.totalFileSize equals the checked-arithmetic size");
    checkEq(static_cast<long long>(f.getSize()), totalSize,
            "P0-6f actualFileSize == header.totalFileSize");

    // v3 admissibility predicate applied to the real header
    const bool admissible =
        numChannels >= 1 && numChannels <= 8
        && numSamples  >= 1 && numSamples  <= 2097152
        && targetLength >= 1 && targetLength <= 2097152
        && targetLength == numSamples
        && numSections >= 0 && numSections <= 20;
    check(admissible, "P0-6f the real header satisfies the v3 admissibility predicate");
}

} // namespace

// ===========================================================================
//  FS-7 : independent timestamp-write failure and flush failure injection
//  (Owner D-1R-2AA section 7)
//
//  Contract, verified separately for each of the two faults:
//    (1) load() still returns TRUE          - a valid payload is never turned
//                                              into load() == false
//    (2) payload bytes are unchanged       - byte-level, region [76, size)
//    (3) file size is unchanged
//    (4) the failure is diagnostic-only    - the seam observed the attempt and
//                                              load()'s verdict is identical to
//                                              the unfaulted baseline
//    (5) payload length / section count intact
//
//  The seam is compiled into this target only (CONVOPEQ_PCACHE_TEST_SEAM is
//  defined on PersistentCacheV3Phase0FailureTests alone), so no permanent
//  production fault point exists.
// ===========================================================================
namespace
{
    constexpr std::size_t kHeaderBytes = 76;   //!< header prefix excluded from the payload comparison

    bool readAll (const juce::File& f, std::vector<char>& out)
    {
        const auto n = f.getSize();
        out.assign (static_cast<std::size_t> (n), 0);
        std::ifstream in (f.getFullPathName().toStdString(), std::ios::in | std::ios::binary);
        if (! in.good())
            return false;
        in.read (out.data(), static_cast<std::streamsize> (n));
        return in.good() || in.eof();
    }

    std::size_t countDiffsInHeader (const std::vector<char>& a, const std::vector<char>& b)
    {
        const auto n = std::min (a.size(), b.size());
        const auto lim = std::min<std::size_t> (n, kHeaderBytes);
        std::size_t diffs = 0;
        for (std::size_t i = 0; i < lim; ++i)
            if (a[i] != b[i])
                ++diffs;
        return diffs;
    }

    bool payloadEqual (const std::vector<char>& a, const std::vector<char>& b)
    {
        if (a.size() != b.size())
            return false;
        if (a.size() <= kHeaderBytes)
            return true;
        return std::memcmp (a.data() + kHeaderBytes,
                            b.data() + kHeaderBytes,
                            a.size() - kHeaderBytes) == 0;
    }

    // Runs one fault mode and applies the full five-point contract.
    void runOneFaultMode (ArtifactGuard& guard,
                          int idx,
                          const char* tag,
                          convo::MixedPhasePersistentCache::TimestampWriteFault mode,
                          bool& loadedBaseline)
    {
        guard.track (idx);

        const auto ir = makeIr();
        const auto sec = makeSections();
        check (convo::MixedPhasePersistentCache::save (testHash (idx), kTestSr,
                                                        kTestPhaseMode, kTestF1, kTestF2,
                                                        kTestTargetLength, ir, sec, sec),
               std::string (tag) + " save() succeeds");

        auto f = fileOf (idx);
        const auto sizeAfterSave = f.getSize();
        checkEq (static_cast<long long> (sizeAfterSave),
                 expectedSizeOf (kTestChannels, kTestTargetLength, kTestSections),
                 std::string (tag) + " saved artifact has the expected exact size");

        std::vector<char> before;
        check (readAll (f, before), std::string (tag) + " baseline byte read succeeds");

        // ---- arm exactly one fault ------------------------------------------
        convo::MixedPhasePersistentCache::armTimestampWriteFaultForTest (mode);

        juce::AudioBuffer<double> out (kTestChannels, kTestTargetLength);
        std::vector<double> rho, theta;
        const bool loaded = convo::MixedPhasePersistentCache::load (
            testHash (idx), kTestSr, kTestPhaseMode, kTestF1, kTestF2,
            kTestTargetLength, out, rho, theta);

        const int attempts = convo::MixedPhasePersistentCache::getInjectedFaultAttemptsForTest();
        convo::MixedPhasePersistentCache::armTimestampWriteFaultForTest (
            convo::MixedPhasePersistentCache::TimestampWriteFault::none);

        // ---- (1) valid payload is never turned into load() == false ---------
        check (loaded, std::string (tag) + " (1) load() returns TRUE despite the injected fault");

        // ---- (4) diagnostic-only -------------------------------------------
        check (attempts >= 1,
               std::string (tag) + " (4) the seam observed the injected fault");
        check (loaded == loadedBaseline,
               std::string (tag) + " (4) load() verdict is identical to the unfaulted baseline");

        // ---- (5) payload shape intact ---------------------------------------
        checkEq (out.getNumSamples(), kTestTargetLength,
                 std::string (tag) + " (5) payload length intact");
        checkEq (static_cast<long long> (rho.size()), kTestSections,
                 std::string (tag) + " (5) allpass section count intact");

        // ---- (3) file size unchanged ----------------------------------------
        checkEq (static_cast<long long> (f.getSize()), static_cast<long long> (sizeAfterSave),
                 std::string (tag) + " (3) file size is unchanged");

        // ---- (2) payload byte-identical -------------------------------------
        std::vector<char> after;
        check (readAll (f, after), std::string (tag) + " post-fault byte read succeeds");
        check (payloadEqual (before, after),
               std::string (tag) + " (2) payload bytes [76, size) are byte-identical");
        std::fprintf (stderr, "       %s: header bytes differing = %zu (metadata only)\n",
                      tag, countDiffsInHeader (before, after));
    }
}

void runFS7FlushFaults(ArtifactGuard& guard)
{
    std::fprintf (stderr, "[P0-6e] FS-7 timestamp-write / flush failure injection\n");

    constexpr int kBase = 7;
    guard.track (kBase);

    // Establish the unfaulted baseline verdict first.
    {
        const auto ir = makeIr();
        const auto sec = makeSections();
        check (convo::MixedPhasePersistentCache::save (testHash (kBase), kTestSr,
                                                        kTestPhaseMode, kTestF1, kTestF2,
                                                        kTestTargetLength, ir, sec, sec),
               "P0-6e baseline save() succeeds");
        juce::AudioBuffer<double> out (kTestChannels, kTestTargetLength);
        std::vector<double> rho, theta;
        const bool loaded = convo::MixedPhasePersistentCache::load (
            testHash (kBase), kTestSr, kTestPhaseMode, kTestF1, kTestF2,
            kTestTargetLength, out, rho, theta);
        check (loaded, "P0-6e unfaulted baseline load() returns TRUE");
        checkEq (convo::MixedPhasePersistentCache::getInjectedFaultAttemptsForTest(), 0,
                 "P0-6e seam is inert when no fault is armed");
    }

    bool loadedBaseline = true;

    runOneFaultMode (guard, kBase + 1, "P0-6e TS-write-fault",
                     convo::MixedPhasePersistentCache::TimestampWriteFault::timestampWrite,
                     loadedBaseline);

    runOneFaultMode (guard, kBase + 2, "P0-6e flush-fault",
                     convo::MixedPhasePersistentCache::TimestampWriteFault::flush,
                     loadedBaseline);

    // The two faults must be independently injectable: after both runs the
    // seam is disarmed and a clean load must behave normally again.
    checkEq (convo::MixedPhasePersistentCache::getInjectedFaultAttemptsForTest(), 0,
             "P0-6e disarming resets the seam attempt counter");
}

int main()
{
    // JUCE needs this on Windows console apps that use File / Logger.
    const juce::ScopedJuceInitialiser_GUI juceInit;

    std::fprintf(stderr, "[P0-6] failure-injection harness\n");
    std::fprintf(stderr, "       cache dir = %s\n",
                 convo::MixedPhasePersistentCache::getCacheDirectory()
                     .getFullPathName().toStdString().c_str());

    ArtifactGuard guard;

    try
    {
        verifyCurrentProductionContract();
        runProducerInvariant(guard);
        runFS1(guard);
        runFS2(guard);
        runFS3(guard);
        runFS4To6(guard);
        runFS7FlushFaults(guard);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 2;
    }

    std::fprintf(stderr, "\nPersistentCacheV3Phase0FailureTests: checks=%d failures=%d\n",
                 g_checks, g_failures);
    if (g_failures != 0)
        return 1;
    std::fprintf(stderr, "P0-6: PASS\n");
    return 0;
}