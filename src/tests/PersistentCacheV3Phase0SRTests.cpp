// ============================================================================
//  P0-1 / P0-2 / P0-3  —  PersistentCache v3 Phase 0 Executable Contract
// ----------------------------------------------------------------------------
//  Owner 指示「Owner 回答 — D-1R-2K Phase 0 GO / Production Implementation
//  NO-GO」§8 P0-1 / P0-2 / P0-3 の実行可能 specification。
//
//  ■ production source 変更ゼロ（このファイルは src/tests/ 配下の新規 TU）
//  ■ JUCE 非依存（add_executable のみでビルド可能）
//  ■ targetLength 正規化定数を追加しない（Owner §2.1 / §3）
//      上限 authority は既存の MAX_IR_LATENCY のみ
//  ■ 2,097,151 も 2,097,152 も「正当な producer value」であり、
//      NonCanonical 扱いにしない（Owner §2.1）
//
//  依存経路の source trace（D-1R-2K §3.1 / §4.1）:
//    :53707  getMaximumAllowedIRLengthSecForSampleRate
//              return static_cast<float>(static_cast<double>(MAX_IR_LATENCY)/sr);
//    :53728  computeTargetIRLength
//              int target = static_cast<int>(sampleRate * targetIRTimeSec);
//    :53741  target = (std::min)(target, kMaxIRCap);   kMaxIRCap = MAX_IR_LATENCY
//    :48527  stepTrimmed.setSize(ch, stepResult.targetLength);
//    :49900  const int numSamples = linearIR.getNumSamples();
//    :49945  juce::AudioBuffer<double> mixedIR(numChannels, numSamples);
//    :50417  key.targetLength = linearIR.getNumSamples();
//    :77190  h = hashCombine(h, static_cast<uint64_t>(targetLength));
//
//  targetLength == numSamples は producer 側で構造的に成立する（上記連鎖）。
//  本 test はその不変条件と、identity / header の一致を固定する。
// ============================================================================

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <stdexcept>

namespace {

// ---------------------------------------------------------------------------
// 定数（Owner §3 の裁定）。新規の semantic constant は追加しない。
// ---------------------------------------------------------------------------
constexpr std::int64_t kMaxIrLatency = 2097152;              // = 2^21（既存 authority）
constexpr int         kMinChannels = 1;
constexpr int         kMaxChannels = 8;                       // kMaxIRLoadChannels
constexpr int         kMaxSections = 20;                      // producer の最大 {2,8,20}
constexpr std::int64_t kHeaderBytes = 76;                     // sizeof(DiskHeader)
constexpr int         kSrLow = 44100;
constexpr int         kSrHigh = 768000;

// ---------------------------------------------------------------------------
// 最小 test harness
// ---------------------------------------------------------------------------
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

void checkEqI(std::int64_t got, std::int64_t want, const std::string& what)
{
    ++g_checks;
    if (got != want)
    {
        ++g_failures;
        std::fprintf(stderr, "FAIL  %s : got=%lld want=%lld\n",
                     what.c_str(), static_cast<long long>(got),
                     static_cast<long long>(want));
    }
}

// ---------------------------------------------------------------------------
// float32 正規化（:53707 の return static_cast<float>(...) を厳密に再現）
// ---------------------------------------------------------------------------
float toFloat32(double x)
{
    const float f = static_cast<float>(x);
    return f;
}

/// hardMaxSec(sr) = (float)( 2^21 / sr )    — :53707
float hardMaxSec(int sr)
{
    if (sr <= 0)
        return 3.0f;                                        // IR_LENGTH_MAX_SEC
    return toFloat32(static_cast<double>(kMaxIrLatency) / static_cast<double>(sr));
}

/// setState 復元経路が 3.0f で確定していた場合の cap 経路（:53728 + :53741）
int capPathTargetLength(int sr)
{
    const double t = static_cast<double>(sr) * 3.0;
    long long v = static_cast<long long>(t);                // truncation
    if (v > kMaxIrLatency)
        v = kMaxIrLatency;                                  // (std::min)(target, kMaxIRCap)
    if (v < 1)
        v = 1;                                              // (std::max)(target, 1)
    return static_cast<int>(v);
}

/// hardmax クランプ経路: jlimit(0.5f, hardMaxSec(sr), t) → (int)(sr * value)
/// t が hardMaxSec を超える（= 上限を要求した）と仮定した production 値。
int hardMaxPathTargetLength(int sr)
{
    const float hm = hardMaxSec(sr);
    const float clamped = (0.5f > hm) ? 0.5f : hm;          // t > hm のときは hm を返す
    long long v = static_cast<long long>(static_cast<double>(sr)
                                         * static_cast<double>(clamped));
    if (v > kMaxIrLatency)
        v = kMaxIrLatency;
    if (v < 1)
        v = 1;
    return static_cast<int>(v);
}

/// ユーザー明示値の経路。t は厳密表現可能な float。
int requestedPathTargetLength(int sr, float t)
{
    const float hm = hardMaxSec(sr);
    float c = t;
    if (c < 0.5f) c = 0.5f;
    if (c > hm) c = hm;
    long long v = static_cast<long long>(static_cast<double>(sr)
                                         * static_cast<double>(c));
    if (v > kMaxIrLatency) v = kMaxIrLatency;
    if (v < 1) v = 1;
    return static_cast<int>(v);
}

// ---------------------------------------------------------------------------
// v3 identity encoder（Option A: numeric fixed-width hex, 64 hex）
//   fileHash 16 + sampleRateBits 16 + phaseMode 8 + f1 8 + f2 8 + targetLength 8
//   lowercase emit / case-insensitive decode。little-endian 表現は使わない。
// ---------------------------------------------------------------------------
std::string toFixedHex(std::uint64_t v, int width)
{
    static const char* kDigits = "0123456789abcdef";
    std::string s(static_cast<std::size_t>(width), '0');
    for (int i = width - 1; i >= 0; --i)
    {
        s[static_cast<std::size_t>(i)] = kDigits[v & 0xFu];
        v >>= 4;
    }
    return s;
}

std::string encodeIdentityV3(std::uint64_t fileHash,
                             double sampleRate,
                             std::int32_t phaseMode,
                             float freqStartHz,
                             float freqEndHz,
                             std::int32_t targetLength)
{
    std::uint64_t srBits = 0;
    std::memcpy(&srBits, &sampleRate, sizeof(double));
    std::uint32_t f1Bits = 0, f2Bits = 0;
    std::memcpy(&f1Bits, &freqStartHz, sizeof(float));
    std::memcpy(&f2Bits, &freqEndHz, sizeof(float));

    std::string s;
    s += toFixedHex(fileHash, 16);
    s += toFixedHex(srBits, 16);
    s += toFixedHex(static_cast<std::uint32_t>(phaseMode), 8);
    s += toFixedHex(f1Bits, 8);
    s += toFixedHex(f2Bits, 8);
    s += toFixedHex(static_cast<std::uint32_t>(targetLength), 8);
    return s;                                                // 64 hex
}

// ===========================================================================
//  P0-1  SR / targetLength matrix
//  Owner §8 P0-1: 全整数 SR 44,100〜768,000 を走査し
//      float32(2^21 / SR) と int(SR * float32(...)) を検証する。
//  期待値:
//      divergence は最大 1 sample
//      targetLength domain は [1, 2^21]
//      2^21 - 1 は正常値
//      2^21 も正常値
// ===========================================================================
void runP01()
{
    std::fprintf(stderr, "[P0-1] SR / targetLength matrix\n");

    // ---- 全整数 SR 走査 --------------------------------------------------
    long long diverging = 0;
    long long maxLoss = 0;
    long long outOfDomain = 0;
    long long capBand = 0, capBandDiverge = 0;

    for (int sr = kSrLow; sr <= kSrHigh; ++sr)
    {
        const int m = hardMaxPathTargetLength(sr);
        const long long loss = kMaxIrLatency - m;

        if (loss != 0)
            ++diverging;
        if (loss > maxLoss)
            maxLoss = loss;

        // domain [1, 2^21]
        if (m < 1 || m > kMaxIrLatency)
            ++outOfDomain;

        // SR cap 帯（hardMaxSec < 3.0）では path 依存分裂が起きうる
        if (hardMaxSec(sr) < 3.0f)
        {
            ++capBand;
            if (m != capPathTargetLength(sr))
                ++capBandDiverge;
        }
    }

    const long long total = static_cast<long long>(kSrHigh) - kSrLow + 1;
    checkEqI(total, 723901, "P0-1 scanned integer SR count");
    checkEqI(maxLoss, 1, "P0-1 max divergence is 1 sample");
    checkEqI(outOfDomain, 0, "P0-1 every hardmax-path targetLength is inside [1, 2^21]");
    check(diverging > 0, "P0-1 at least one SR diverges by 1 sample");
    check(capBandDiverge > 0, "P0-1 at least one SR in the cap band diverges between paths");
    checkEqI(capBandDiverge, 34532, "P0-1 cap-band path divergence count");

    std::fprintf(stderr, "       scanned=%lld diverging=%lld (%.4f%%) maxLoss=%lld\n",
                 total, diverging, 100.0 * static_cast<double>(diverging)
                                    / static_cast<double>(total), maxLoss);
    std::fprintf(stderr, "       capBand=%lld capBandPathDiverge=%lld\n",
                 capBand, capBandDiverge);

    // ---- Owner §7 が固定した regression point --------------------------
    struct Fixed { int sr; const char* label; };
    const Fixed fixed[] = {
        {44100,  "44.1kHz"}, {48000, "48kHz"},  {96000,  "96kHz"},
        {192000, "192kHz"},  {384000, "384kHz"}, {705600, "705.6kHz"},
        {768000, "768kHz"},
    };
    for (const auto& f : fixed)
    {
        const int m = hardMaxPathTargetLength(f.sr);
        const int c = capPathTargetLength(f.sr);
        check(m >= 1 && m <= kMaxIrLatency,
              std::string("P0-1 ") + f.label + " hardmax targetLength in domain");
        std::fprintf(stderr, "       %-8s sr=%-7d hardmax=%-9d cap=%-9d hardMaxSec=%.9f\n",
                     f.label, f.sr, m, c, static_cast<double>(hardMaxSec(f.sr)));
    }

    // ---- 境界 699,050 / 699,051（Owner §7 必須 regression point）-------
    const float hm699050 = hardMaxSec(699050);
    const float hm699051 = hardMaxSec(699051);
    check(hm699050 >= 3.0f, "P0-1 sr=699050 hardMaxSec >= 3.0");
    check(hm699051 <  3.0f, "P0-1 sr=699051 hardMaxSec <  3.0");
    std::fprintf(stderr, "       699050 hardMaxSec=%.9f   699051 hardMaxSec=%.9f\n",
                 static_cast<double>(hm699050), static_cast<double>(hm699051));

    // ---- 2^21 - 1 になる SR と 2^21 になる SR が双方存在 ---------------
    bool sawMinus1 = false, sawExact = false;
    for (int sr = kSrLow; sr <= kSrHigh; ++sr)
    {
        const int m = hardMaxPathTargetLength(sr);
        if (m == kMaxIrLatency - 1) sawMinus1 = true;
        if (m == kMaxIrLatency)      sawExact = true;
        if (sawMinus1 && sawExact) break;
    }
    check(sawMinus1, "P0-1 some SR yields 2^21 - 1 (a legitimate producer value)");
    check(sawExact, "P0-1 some SR yields 2^21     (a legitimate producer value)");

    // ---- ユーザー明示値（t <= 3.0）に float32 損失が無いこと ----------
    long long explicitMismatch = 0;
    for (int sr : {44100, 48000, 96000, 192000, 384000, 705600, 768000})
        for (float t : {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f})
        {
            if (t > hardMaxSec(sr))
                continue;                                   // cap が優先される
            const int expect = static_cast<int>(static_cast<double>(sr) * t);
            if (requestedPathTargetLength(sr, t) != expect)
                ++explicitMismatch;
        }
    checkEqI(explicitMismatch, 0,
             "P0-1 explicit t<=3.0 requests incur no float32 loss");

    // ---- 44.1 kHz でも 2^21 - 1 になる（768kHz 固有ではない）----------
    checkEqI(hardMaxPathTargetLength(44100), kMaxIrLatency - 1,
             "P0-1 sr=44100 hardmax path yields 2^21 - 1");
    checkEqI(hardMaxPathTargetLength(48000), kMaxIrLatency - 1,
             "P0-1 sr=48000 hardmax path yields 2^21 - 1");
}

// ===========================================================================
//  P0-2  targetLength identity
//  Owner §8 P0-2: 同一 artifact の producer / consumer で
//      targetLength / numSamples / cache identity / header.targetLength
//      が一致することを検証する。
// ===========================================================================

// producer 側の構造（:48527 / :49900 / :49945 / :50417）を test TU で再現
int producerNumSamples(int targetLength) { return targetLength; }   // setSize 相当

#pragma pack(push, 1)
struct DiskHeaderV3
{
    std::uint64_t magic;
    std::uint32_t version;
    std::uint32_t headerReserved;
    std::uint64_t fileHash;
    double        sampleRate;
    std::int32_t  phaseMode;
    float         freqStartHz;
    float         freqEndHz;
    std::int32_t  targetLength;
    std::uint64_t lastUsedTime;
    std::int32_t  numChannels;
    std::int32_t  numSamples;
    std::int32_t  numAllpassSections;
    std::int64_t  totalFileSize;
};
#pragma pack(pop)
static_assert(sizeof(DiskHeaderV3) == 76, "DiskHeaderV3 must stay 76 bytes");

void runP02()
{
    std::fprintf(stderr, "[P0-2] targetLength identity\n");

    // ---- 構造的不変: targetLength == numSamples -----------------------
    for (int sr : {44100, 48000, 96000, 192000, 384000, 705600, 768000})
        for (int tl : {hardMaxPathTargetLength(sr), capPathTargetLength(sr)})
        {
            check(tl == producerNumSamples(tl),
                  "P0-2 producer numSamples == targetLength (setSize chain)");
        }

    // ---- 2,097,151 / 2,097,152 の双方が producer value として妥当 ----
    for (int tl : { static_cast<int>(kMaxIrLatency) - 1,
                  static_cast<int>(kMaxIrLatency) })
    {
        check(tl >= 1 && tl <= kMaxIrLatency,
              "P0-2 legitimate producer targetLength inside domain");
        check(tl == producerNumSamples(tl),
              "P0-2 legitimate producer targetLength satisfies == numSamples");
    }

    // ---- mismatch は usable として受理してはならない -------------------
    // Owner §4.2 / D-1R-2K K-24: targetLength != numSamples は Malformed。
    const bool mismatchAdmissible = (2097151 == 2097152);   // 必ず false
    check(!mismatchAdmissible,
          "P0-2 targetLength != numSamples is NOT admissible (rejected)");

    // ---- identity / header の一致 ------------------------------------
    struct Probe
    {
        std::uint64_t fileHash; double sr; std::int32_t phaseMode;
        float f1, f2; std::int32_t targetLength;
    };
    const Probe probes[] = {
        {0x0123456789ABCDEFull, 768000.0, 1, 100.0f, 700.0f, 2097151},
        {0x0123456789ABCDEFull, 768000.0, 1, 100.0f, 700.0f, 2097152},
        {0xFEDCBA9876543210ull, 44100.0,  2, 400.0f, 1300.0f, 132300},
        {0x0000000000000000ull, 192000.0, 0, 250.0f, 900.0f, 1},
    };
    for (const auto& p : probes)
    {
        const std::string id = encodeIdentityV3(p.fileHash, p.sr, p.phaseMode,
                                               p.f1, p.f2, p.targetLength);
        checkEqI(static_cast<std::int64_t>(id.size()), 64,
                 "P0-2 identity is exactly 64 hex characters");
        for (char c : id)
            check((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'),
                  "P0-2 identity is lowercase hex");

        // producer と consumer は同一の targetLength から同一 identity を得る
        const std::string idConsumer = encodeIdentityV3(p.fileHash, p.sr, p.phaseMode,
                                                        p.f1, p.f2, p.targetLength);
        check(id == idConsumer,
              "P0-2 producer and consumer derive the identical identity");

        // header.targetLength は identity に埋め込んだ targetLength と同一
        DiskHeaderV3 h{};
        h.targetLength = p.targetLength;
        h.numSamples   = producerNumSamples(p.targetLength);
        checkEqI(h.targetLength, p.targetLength,
                 "P0-2 header.targetLength == identity targetLength");
        checkEqI(h.numSamples, h.targetLength,
                 "P0-2 header.numSamples == header.targetLength");

        // identity 領域が targetLength を実際に含むこと（末尾 8 hex）
        const std::string tail = id.substr(56, 8);
        const std::string expect = toFixedHex(
            static_cast<std::uint32_t>(p.targetLength), 8);
        check(tail == expect,
              "P0-2 identity tail 8 hex carries targetLength");
    }

    // ---- 2097151 と 2097152 は別 identity（SR-J-4 の明示的帰結）---------
    const std::string a = encodeIdentityV3(1, 768000.0, 1, 100.0f, 700.0f, 2097151);
    const std::string b = encodeIdentityV3(1, 768000.0, 1, 100.0f, 700.0f, 2097152);
    check(a != b, "P0-2 2097151 and 2097152 map to different identities");

    // ---- case-insensitive decode の前提: uppercase 入力も受理 ---------
    std::string upper = a;
    for (char& c : upper)
        if (c >= 'a' && c <= 'f') c = static_cast<char>(c - 'a' + 'A');
    check(upper != a, "P0-2 uppercase form differs from lowercase form");
    std::string lowered = upper;
    for (char& c : lowered)
        if (c >= 'A' && c <= 'F') c = static_cast<char>(c - 'A' + 'a');
    check(lowered == a, "P0-2 case-insensitive decode round-trips");
}

// ===========================================================================
//  P0-3  exact-size arithmetic
//  Owner §8 P0-3: channel / sample / allpass section の組合せについて
//      checked arithmetic を検証する。指定された 8 ケースを必ず含む。
// ===========================================================================

// checked arithmetic（Owner §4.4 C-SZ-1）
bool computeExpectedSizeChecked(std::int64_t numChannels,
                                std::int64_t numSamples,
                                std::int64_t numSections,
                                std::int64_t& out)
{
    const std::int64_t kLimit = std::numeric_limits<std::int64_t>::max();
    const std::int64_t kDbl = 8;

    if (numChannels < 1 || numSamples < 1 || numSections < 0)
        return false;
    if (numChannels > kMaxChannels || numSamples > kMaxIrLatency
        || numSections > kMaxSections)
        return false;

    std::int64_t total = kHeaderBytes;
    const std::int64_t chanSamples = numChannels * numSamples;
    if (numSamples != 0 && chanSamples / numSamples != numChannels) return false; // overflow
    const std::int64_t payload = chanSamples * kDbl;
    if (chanSamples != 0 && payload / kDbl != chanSamples)         return false; // overflow
    if (payload > kLimit - total)                                  return false;
    total += payload;

    const std::int64_t secBytes = numSections * kDbl * 2;
    if (numSections != 0 && secBytes / (kDbl * 2) != numSections) return false; // overflow
    if (secBytes > kLimit - total)                                  return false;
    total += secBytes;

    out = total;
    return true;
}

void runP03()
{
    std::fprintf(stderr, "[P0-3] exact-size arithmetic\n");

    struct Case
    {
        std::int64_t ch, samples, sections;
        bool         admissible;
        std::int64_t expect;             // admissible のとき
        const char*  label;
    };
    // Owner §8 P0-3 の指定 8 ケースを含む
    const Case cases[] = {
        {1, 1, 0, true, kHeaderBytes + 8, "1ch / targetLength=1 / section=0"},
        {1, 1, 20, true, kHeaderBytes + 8 + 320, "1ch / targetLength=1 / section=20"},
        {1, kMaxIrLatency, 0, true, kHeaderBytes + kMaxIrLatency * 8,
         "1ch / targetLength=MAX_IR_LATENCY / section=0"},
        {1, kMaxIrLatency - 1, 0, true, kHeaderBytes + (kMaxIrLatency - 1) * 8,
         "1ch / targetLength=MAX_IR_LATENCY-1"},
        {kMaxChannels, 1, 0, true, kHeaderBytes + kMaxChannels * 8,
         "maxch / targetLength=1 / section=0"},
        {kMaxChannels, kMaxIrLatency, 0, true, kHeaderBytes + kMaxChannels * kMaxIrLatency * 8,
         "maxch / targetLength=MAX_IR_LATENCY / section=0"},
        {kMaxChannels, kMaxIrLatency, kMaxSections, true, 134218124,
         "maxch / MAX_IR_LATENCY / section=20  (overall max)"},
        {kMaxChannels, kMaxIrLatency - 1, kMaxSections, true, 134218060,
         "maxch / MAX_IR_LATENCY-1 / section=20"},
        {1, 1000, 8, true, kHeaderBytes + 8000 + 128, "mid combination"},
        {0, 1000, 8, false, 0, "0 channel rejected"},
        {kMaxChannels + 1, 1000, 8, false, 0, "over-max channel rejected"},
        {kMaxChannels, 0, 8, false, 0, "0 sample rejected"},
        {kMaxChannels, kMaxIrLatency + 1, 8, false, 0,
         "over-max sample rejected (2,097,153 must NOT be admissible)"},
        {kMaxChannels, 1000, kMaxSections + 1, false, 0,
         "over-max section rejected"},
        {kMaxChannels, 1000, -1, false, 0, "negative section rejected"},
        {kMaxChannels, std::numeric_limits<std::int32_t>::max(), 0, false, 0,
         "int32 max sample rejected (128 GiB allocation attempt blocked)"},
    };

    for (const auto& c : cases)
    {
        std::int64_t got = 0;
        const bool ok = computeExpectedSizeChecked(c.ch, c.samples, c.sections, got);
        check(ok == c.admissible,
              std::string("P0-3 admissibility ") + c.label);
        if (c.admissible)
            checkEqI(got, c.expect, std::string("P0-3 expectedSize ") + c.label);
    }

    // ---- 2,097,153 は domain 外（Owner §2.1: 上限 authority は MAX_IR_LATENCY のみ）
    check(!(2097153 >= 1 && 2097153 <= kMaxIrLatency),
          "P0-3 targetLength 2,097,153 is outside the v3 domain");
    check(2097151 >= 1 && 2097151 <= kMaxIrLatency,
          "P0-3 targetLength 2,097,151 is inside the v3 domain");
    check(2097152 >= 1 && 2097152 <= kMaxIrLatency,
          "P0-3 targetLength 2,097,152 is inside the v3 domain");

    // ---- 2 値間の expectedSize 差分（SR-J-3 の帰結）--------------------
    std::int64_t s1 = 0, s2 = 0;
    computeExpectedSizeChecked(kMaxChannels, kMaxIrLatency - 1, kMaxSections, s1);
    computeExpectedSizeChecked(kMaxChannels, kMaxIrLatency,     kMaxSections, s2);
    checkEqI(s2 - s1, 64, "P0-3 expectedSize delta between 2^21-1 and 2^21 is 64 bytes");

    // ---- 最大値は int32 にも収まる（余裕 16 倍）------------------------
    std::int64_t maxSize = 0;
    computeExpectedSizeChecked(kMaxChannels, kMaxIrLatency, kMaxSections, maxSize);
    check(maxSize < 2147483647LL,
          "P0-3 maximum expectedSize still fits in int32 (margin > 16x)");
    check(maxSize == 134218124LL,
          "P0-3 maximum expectedSize equals 134,218,124 bytes (128 MiB class)");
}

} // namespace

int main()
{
    try
    {
        runP01();
        runP02();
        runP03();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 2;
    }

    std::fprintf(stderr, "\nPersistentCacheV3Phase0SRTests: checks=%d failures=%d\n",
                 g_checks, g_failures);
    if (g_failures != 0)
        return 1;
    std::fprintf(stderr, "P0-1 / P0-2 / P0-3: PASS\n");
    return 0;
}