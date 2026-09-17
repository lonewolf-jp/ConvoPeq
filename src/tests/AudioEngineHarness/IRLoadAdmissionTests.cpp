// IRLoadAdmissionTests.cpp — WORK102 (big 1-8)
//
// Bounded IR load admission + streaming hash の回帰。
// 契約の正本: doc/work102/big18_failure_contract_arbitration_20260917.md（R2 凍結）
//
// テスト対応:
//   A  1ch accept              G  raw > hardMax だが trim 後 accept
//   B  2ch accept              H  trim 後 FC-FORM-5 reject
//   C  8ch accept              I  streaming hash digest 一致（独立 one-shot と照合）
//   D  9ch deterministic reject  J  allocation failure の graceful path（構造検証・§報告参照）
//   E  1 GiB 境界 predicate      K  cancellation semantics 不変（構造検証・§報告参照）
//   F  INT32_MAX 境界 predicate  L  resample path accept
//
// J / K は決定論的に発火させられないため本 TU では実行せず、ソース差分スコープで確認する
// （IG §7.5）。I の (ii) O(1) メモリも同様にソース検査で確認する。

#include <JuceHeader.h>

#include "ConvolverProcessor.h"
#include "AllpassDesigner.h"
#include "convolver/IRLoadAdmission.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

//----------------------------------------------------------
// helpers
//----------------------------------------------------------

void pumpTestMessages() noexcept
{
#if JUCE_WINDOWS
    MSG msg {};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
#endif
}

// fill: 0 = 単一ピーク（以降は無音）, 1 = 全サンプル非零（末尾無音なし）
juce::File writeTestIr(const juce::String& tag, double sampleRate, int numCh,
                       int totalSamples, int peakPos, int fill)
{
    const juce::File f = juce::File::getSpecialLocation(juce::File::tempDirectory)
                             .getChildFile("convo_work102_" + tag);
    f.deleteFile();

    std::unique_ptr<juce::FileOutputStream> stream(f.createOutputStream());
    if (stream == nullptr)
        return {};

    juce::WavAudioFormat fmt;
    std::unique_ptr<juce::AudioFormatWriter> w(
        fmt.createWriterFor(stream.get(), sampleRate,
                            static_cast<unsigned int>(numCh),
                            16, {}, 0));
    if (w == nullptr)
        return {};

    stream.release(); // writer が所有権を引き継ぐ

    juce::AudioBuffer<float> buf(numCh, totalSamples);
    buf.clear();

    if (fill == 0)
    {
        if (peakPos >= 0 && peakPos < totalSamples)
            for (int ch = 0; ch < numCh; ++ch)
                buf.getWritePointer(ch)[peakPos] = 1.0f;
    }
    else
    {
        // 決定的な非零パターン。16 サンプル周期（44.1k で約 2.76 kHz）なので
        // リサンプル後も十分な振幅が残る。末尾サンプルも非零 = trim が縮めない。
        for (int ch = 0; ch < numCh; ++ch)
        {
            float* p = buf.getWritePointer(ch);
            for (int i = 0; i < totalSamples; ++i)
                p[i] = ((i / 8) % 2 == 0) ? 0.25f : -0.25f;
        }
    }

    int written = 0;
    while (written < totalSamples)
    {
        const int n = std::min(65536, totalSamples - written);
        if (!w->writeFromAudioSampleBuffer(buf, written, n))
            return {};
        written += n;
    }

    w.reset(); // flush
    return f.existsAsFile() ? f : juce::File();
}

// ロード終端（成功 or エラー確定）まで待つ
bool pollTerminal(ConvolverProcessor& conv, int maxIter = 6000)
{
    for (int i = 0; i < maxIter; ++i)
    {
        if (conv.isIRLoaded() || conv.getLastError().isNotEmpty())
            return true;
        pumpTestMessages();
        juce::Thread::sleep(5);
    }
    return false;
}

bool loadAndSettle(ConvolverProcessor& conv, const juce::File& ir, juce::String& err)
{
    conv.loadImpulseResponse(ir, false);
    if (!pollTerminal(conv))
        return false;
    err = conv.getLastError();
    return true;
}

// ★ 片付け契約（work98 §8-13 と同旨）: ConvolverProcessor を使うテストは
//   exit 経路すべてで releaseResources() を呼ぶ。LoaderThread 滞留が後続テスト
//   （testCallerDestroyTerminalDisposition 等）へ非決定性を持ち込むため。
struct ConvReleaser
{
    explicit ConvReleaser(ConvolverProcessor& target) : c(target) {}
    ~ConvReleaser() { c.releaseResources(); }
    ConvReleaser(const ConvReleaser&) = delete;
    ConvReleaser& operator=(const ConvReleaser&) = delete;
    ConvolverProcessor& c;
};

[[nodiscard]] bool containsAll(const juce::String& s, const std::vector<juce::String>& needles)
{
    for (const auto& n : needles)
        if (!s.contains(n))
            return false;
    return true;
}

//----------------------------------------------------------
// 独立 one-shot XXH64（ストリーミング実装の照合用オラクル）
//   seed は production と同じ固定値（変更されたら本テストが落ちる）
//----------------------------------------------------------

constexpr uint64_t kExpectedHashSalt = 0x434f4e564f504551ull; // "CONVOPEQ"

inline uint64_t refRotl64(uint64_t v, int c) noexcept
{
    return (v << c) | (v >> (64 - c));
}

inline uint64_t refReadLE64(const uint8_t* p) noexcept
{
    uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint32_t refReadLE32(const uint8_t* p) noexcept
{
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint64_t refRound(uint64_t acc, uint64_t input) noexcept
{
    constexpr uint64_t kP1 = 11400714785074694791ull;
    constexpr uint64_t kP2 = 14029467366897019727ull;
    acc += input * kP2;
    acc = refRotl64(acc, 31);
    acc *= kP1;
    return acc;
}

inline uint64_t refMergeRound(uint64_t acc, uint64_t val) noexcept
{
    constexpr uint64_t kP1 = 11400714785074694791ull;
    constexpr uint64_t kP4 = 9650029242287828579ull;
    acc ^= refRound(0, val);
    return acc * kP1 + kP4;
}

inline uint64_t refAvalanche(uint64_t h) noexcept
{
    constexpr uint64_t kP2 = 14029467366897019727ull;
    constexpr uint64_t kP3 = 1609587929392839161ull;
    h ^= h >> 33;
    h *= kP2;
    h ^= h >> 29;
    h *= kP3;
    h ^= h >> 32;
    return h;
}

uint64_t refXxh64(const uint8_t* data, size_t len, uint64_t seed) noexcept
{
    constexpr uint64_t kP1 = 11400714785074694791ull;
    constexpr uint64_t kP2 = 14029467366897019727ull;
    constexpr uint64_t kP3 = 1609587929392839161ull;
    constexpr uint64_t kP4 = 9650029242287828579ull;
    constexpr uint64_t kP5 = 2870177450012600261ull;

    const uint8_t* p = data;
    const uint8_t* end = data + len;
    uint64_t h = 0;

    if (len >= 32)
    {
        uint64_t v1 = seed + kP1 + kP2;
        uint64_t v2 = seed + kP2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - kP1;

        const uint8_t* limit = end - 32;
        do
        {
            v1 = refRound(v1, refReadLE64(p)); p += 8;
            v2 = refRound(v2, refReadLE64(p)); p += 8;
            v3 = refRound(v3, refReadLE64(p)); p += 8;
            v4 = refRound(v4, refReadLE64(p)); p += 8;
        } while (p <= limit);

        h = refRotl64(v1, 1) + refRotl64(v2, 7) + refRotl64(v3, 12) + refRotl64(v4, 18);
        h = refMergeRound(h, v1);
        h = refMergeRound(h, v2);
        h = refMergeRound(h, v3);
        h = refMergeRound(h, v4);
    }
    else
    {
        h = seed + kP5;
    }

    h += static_cast<uint64_t>(len);

    while ((p + 8) <= end)
    {
        h ^= refRound(0, refReadLE64(p));
        h = refRotl64(h, 27) * kP1 + kP4;
        p += 8;
    }
    if ((p + 4) <= end)
    {
        h ^= static_cast<uint64_t>(refReadLE32(p)) * kP1;
        h = refRotl64(h, 23) * kP2 + kP3;
        p += 4;
    }
    while (p < end)
    {
        h ^= static_cast<uint64_t>(*p) * kP5;
        h = refRotl64(h, 11) * kP1;
        ++p;
    }

    return refAvalanche(h);
}

//----------------------------------------------------------
// E / F — admission predicate の境界（fixture 不要）
//----------------------------------------------------------

bool checkEFPredicateBoundaries()
{
    using namespace convo::irload;

    // 契約値が R2 の凍結値から動いていないこと
    if (kMaxIRLoadBytes != 1073741824 || kMaxIRLoadChannels != 8u
        || kMaxFileLengthSamples != 2147483647 || kMaxIRResampleOutputSamples != 2097153)
    {
        std::fprintf(stderr, "[WORK102] FAIL E/F: contract constants drifted\n");
        return false;
    }

    // FC-FORM-4 / FC-FORM-2 (channel)
    struct ChCase { unsigned n; bool want; };
    const ChCase chCases[] = {
        {0u, false}, {1u, true}, {2u, true}, {3u, true}, {8u, true}, {9u, false}, {64u, false},
    };
    for (const auto& c : chCases)
    {
        if (admitChannelCount(c.n) != c.want)
        {
            std::fprintf(stderr, "[WORK102] FAIL E/F: admitChannelCount(%u) != %d\n",
                         c.n, c.want ? 1 : 0);
            return false;
        }
    }

    // FC-FORM-3 / FC-FORM-4 (length)
    if (!admitFileLengthNonZero(1) || admitFileLengthNonZero(0) || admitFileLengthNonZero(-1))
    {
        std::fprintf(stderr, "[WORK102] FAIL E/F: fileLength non-zero predicate\n");
        return false;
    }
    if (!admitFileLengthRepresentable(2147483647) || admitFileLengthRepresentable(2147483648LL))
    {
        std::fprintf(stderr, "[WORK102] FAIL E/F: INT32 representation boundary\n");
        return false;
    }

    // FC-FORM-1 (byte): 各 N で limit ちょうど = true / limit+1 = false
    struct ByteCase { unsigned n; int64_t limit; };
    const ByteCase byteCases[] = {
        {1u, 134217728}, // 1073741824 / 8
        {2u, 67108864},  // 1073741824 / 16
        {3u, 44739242},  // 1073741824 / 24
        {4u, 33554432},  // / 32
        {8u, 16777216},  // / 64
    };
    for (const auto& c : byteCases)
    {
        if (!admitByteBudget(c.n, c.limit))
        {
            std::fprintf(stderr, "[WORK102] FAIL E: N=%u L=%lld should be admitted\n",
                         c.n, static_cast<long long>(c.limit));
            return false;
        }
        if (admitByteBudget(c.n, c.limit + 1))
        {
            std::fprintf(stderr, "[WORK102] FAIL E: N=%u L=%lld should be rejected\n",
                         c.n, static_cast<long long>(c.limit + 1));
            return false;
        }
        if (admitByteBudget(c.n, 2147483647LL))
        {
            std::fprintf(stderr, "[WORK102] FAIL E: N=%u L=INT32_MAX must not be admitted\n", c.n);
            return false;
        }
    }

    // 除算形の安全性: N=1 でも INT32_MAX は拒否され、乗算溢れを起こさない
    if (admitByteBudget(1u, 2147483647LL) || admitByteBudget(8u, 2147483647LL))
    {
        std::fprintf(stderr, "[WORK102] FAIL E: INT32_MAX admitted (byte bound must bind first)\n");
        return false;
    }

    // FC-FORM-5 (resample output)
    // 境界: 2097153 samples @48k == 2097153/48000 s（ちょうど）
    if (!admitResampleOutput(2097153, 48000.0, 48000.0))
    {
        std::fprintf(stderr, "[WORK102] FAIL F: FC-FORM-5 boundary (exact limit) rejected\n");
        return false;
    }
    if (admitResampleOutput(2097154, 48000.0, 48000.0))
    {
        std::fprintf(stderr, "[WORK102] FAIL F: FC-FORM-5 boundary (+1) admitted\n");
        return false;
    }
    // vacuous: SR が 0 以下なら resample 非実行
    if (!admitResampleOutput(99999999, 0.0, 48000.0) || !admitResampleOutput(99999999, 48000.0, 0.0))
    {
        std::fprintf(stderr, "[WORK102] FAIL F: FC-FORM-5 vacuous case rejected\n");
        return false;
    }
    // 溢れ回帰: N=1 の最大長 @44.1k → 768k は reject（int 変換より前に弾かれる）
    if (admitResampleOutput(134217728, 44100.0, 768000.0))
    {
        std::fprintf(stderr, "[WORK102] FAIL F: overflow case must be rejected\n");
        return false;
    }
    if (!admitResampleOutput(120422, 44100.0, 768000.0))
    {
        std::fprintf(stderr, "[WORK102] FAIL F: legit 44.1k->768k case rejected\n");
        return false;
    }

    std::printf("[WORK102] checkEFPredicateBoundaries: PASS\n");
    return true;
}

//----------------------------------------------------------
// A / B / C / D — 実ファイルでの channel admission
//----------------------------------------------------------

bool checkChannelAdmissionRuntime()
{
    static juce::ScopedJuceInitialiser_GUI juceInit;

    struct Case { const char* tag; int numCh; bool wantAccept; };
    const Case cases[] = {
        {"a_1ch.wav", 1, true},
        {"b_2ch.wav", 2, true},
        {"c_8ch.wav", 8, true},
        {"d_9ch.wav", 9, false},
    };

    for (const auto& c : cases)
    {
        const juce::File ir = writeTestIr(c.tag, 48000.0, c.numCh, 4800, 100, 0);
        if (!ir.existsAsFile())
        {
            std::fprintf(stderr, "[WORK102] FAIL A-D: fixture write failed for %s\n", c.tag);
            return false;
        }

        ConvolverProcessor conv;
        conv.prepareToPlay(48000.0, 512);
        ConvReleaser convReleaser { conv };

        juce::String err;
        const bool settled = loadAndSettle(conv, ir, err);

        bool ok = false;
        if (!settled)
        {
            std::fprintf(stderr, "[WORK102] FAIL A-D: %s did not settle\n", c.tag);
        }
        else if (c.wantAccept)
        {
            ok = conv.isIRLoaded() && err.isEmpty();
            if (!ok)
                std::fprintf(stderr, "[WORK102] FAIL A-D: %s expected accept, err='%s'\n",
                             c.tag, err.toRawUTF8());
        }
        else
        {
            ok = !conv.isIRLoaded() && containsAll(err, {"too many channels", "9", "8"});
            if (!ok)
                std::fprintf(stderr, "[WORK102] FAIL A-D: %s expected channel reject, err='%s'\n",
                             c.tag, err.toRawUTF8());
        }

        ir.deleteFile();
        if (!ok)
            return false;
    }

    std::printf("[WORK102] checkChannelAdmissionRuntime: PASS (1/2/8ch accept, 9ch reject)\n");
    return true;
}

//----------------------------------------------------------
// G — raw は hardMax 超だが trim 後は許容（FC-FORM-5 が trim 後適用であることの回帰）
//----------------------------------------------------------

bool checkResampleBoundUsesTrimmedLength()
{
    static juce::ScopedJuceInitialiser_GUI juceInit;

    // 48 kHz 処理で hardMaxSec = 2097153/48000 = 43.69 s
    // 50 s のファイルを用意し、内容は先頭のみ（残りは無音）→ trim 後は約 1001 samples。
    // raw に FC-FORM-5 を適用する実装なら拒否される。
    const juce::File ir = writeTestIr("g_trim.wav", 48000.0, 2, 2400000, 1000, 0);
    if (!ir.existsAsFile())
    {
        std::fprintf(stderr, "[WORK102] FAIL G: fixture write failed\n");
        return false;
    }

    ConvolverProcessor conv;
    conv.prepareToPlay(48000.0, 512);
    ConvReleaser convReleaser { conv };

    juce::String err;
    const bool settled = loadAndSettle(conv, ir, err);
    const bool ok = settled && conv.isIRLoaded() && err.isEmpty();
    if (!ok)
        std::fprintf(stderr, "[WORK102] FAIL G: expected accept after trim, err='%s' loaded=%d\n",
                     err.toRawUTF8(), conv.isIRLoaded() ? 1 : 0);

    ir.deleteFile();
    if (ok)
        std::printf("[WORK102] checkResampleBoundUsesTrimmedLength: PASS (raw 50s -> trim -> accept)\n");
    return ok;
}

//----------------------------------------------------------
// H — trim 後も FC-FORM-5 超過なら reject
//----------------------------------------------------------

bool checkResampleBoundRejectsOversized()
{
    static juce::ScopedJuceInitialiser_GUI juceInit;

    // 3 s @44.1kHz を 768 kHz 処理で読む → L_res = 2,304,001 > 2,097,153
    // 末尾に無音が無い fill=1 なので trim は縮めない。
    const juce::File ir = writeTestIr("h_oversize.wav", 44100.0, 2, 132300, 100, 1);
    if (!ir.existsAsFile())
    {
        std::fprintf(stderr, "[WORK102] FAIL H: fixture write failed\n");
        return false;
    }

    ConvolverProcessor conv;
    conv.prepareToPlay(768000.0, 512);
    ConvReleaser convReleaser { conv };

    juce::String err;
    const bool settled = loadAndSettle(conv, ir, err);
    const bool ok = settled && !conv.isIRLoaded()
                 && containsAll(err, {"longer than the DSP can use", "2097153"});
    if (!ok)
        std::fprintf(stderr, "[WORK102] FAIL H: expected FC-FORM-5 reject, err='%s' loaded=%d\n",
                     err.toRawUTF8(), conv.isIRLoaded() ? 1 : 0);

    ir.deleteFile();
    if (ok)
        std::printf("[WORK102] checkResampleBoundRejectsOversized: PASS (3s@44.1k -> 768k reject)\n");
    return ok;
}

//----------------------------------------------------------
// L — resample path の accept
//----------------------------------------------------------

bool checkResamplePathAccepted()
{
    static juce::ScopedJuceInitialiser_GUI juceInit;

    // 1 s @44.1kHz を 48 kHz 処理で読む → L_res = 48,001 <= 2,097,153
    const juce::File ir = writeTestIr("l_resample.wav", 44100.0, 2, 44100, 100, 1);
    if (!ir.existsAsFile())
    {
        std::fprintf(stderr, "[WORK102] FAIL L: fixture write failed\n");
        return false;
    }

    ConvolverProcessor conv;
    conv.prepareToPlay(48000.0, 512);
    ConvReleaser convReleaser { conv };

    juce::String err;
    const bool settled = loadAndSettle(conv, ir, err);
    const bool ok = settled && conv.isIRLoaded() && err.isEmpty();
    if (!ok)
        std::fprintf(stderr, "[WORK102] FAIL L: expected accept, err='%s' loaded=%d\n",
                     err.toRawUTF8(), conv.isIRLoaded() ? 1 : 0);

    ir.deleteFile();
    if (ok)
        std::printf("[WORK102] checkResamplePathAccepted: PASS (44.1k -> 48k resample accept)\n");
    return ok;
}

//----------------------------------------------------------
// I — streaming hash が独立 one-shot と一致すること
//----------------------------------------------------------

bool checkStreamingHashMatchesReference()
{
    // 長さを変えて、stripes / 8-4-1 tail / 4096 read 境界をすべて踏む
    const size_t sizes[] = {0u, 1u, 7u, 31u, 32u, 33u, 63u, 64u, 65u, 4095u, 4096u, 4097u, 100000u};

    for (const size_t n : sizes)
    {
        std::vector<uint8_t> bytes(n);
        for (size_t i = 0; i < n; ++i)
            bytes[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xFFu);

        const juce::File f = juce::File::getSpecialLocation(juce::File::tempDirectory)
                                 .getChildFile("convo_work102_i_" + juce::String(static_cast<int>(n)) + ".bin");
        f.deleteFile();

        if (n > 0)
        {
            if (!f.replaceWithData(bytes.data(), n))
            {
                std::fprintf(stderr, "[WORK102] FAIL I: fixture write failed (n=%zu)\n", n);
                return false;
            }
        }
        else
        {
            if (!f.create().wasOk())
            {
                std::fprintf(stderr, "[WORK102] FAIL I: empty fixture create failed\n");
                return false;
            }
        }

        const uint64_t got = convo::AllpassDesigner::computeIRHash(f);
        const uint64_t want = refXxh64(bytes.data(), n, kExpectedHashSalt);

        f.deleteFile();

        if (got != want)
        {
            std::fprintf(stderr,
                         "[WORK102] FAIL I: digest mismatch n=%zu got=%llu want=%llu\n",
                         n,
                         static_cast<unsigned long long>(got),
                         static_cast<unsigned long long>(want));
            return false;
        }
    }

    std::printf("[WORK102] checkStreamingHashMatchesReference: PASS (13 sizes, one-shot oracle)\n");
    return true;
}

} // namespace

int runIRLoadAdmissionTests()
{
    if (!checkEFPredicateBoundaries())
        return 1;
    if (!checkStreamingHashMatchesReference())
        return 1;
    if (!checkChannelAdmissionRuntime())
        return 1;
    if (!checkResampleBoundUsesTrimmedLength())
        return 1;
    if (!checkResampleBoundRejectsOversized())
        return 1;
    if (!checkResamplePathAccepted())
        return 1;

    std::printf("IRLoadAdmissionTests: PASS (WORK102 A/B/C/D/E/F/G/H/I/L)\n");
    return 0;
}
