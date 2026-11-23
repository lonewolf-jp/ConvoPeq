// CacheManagerHeaderValidationTests.cpp — B-02 回帰
//
// B-02: キャッシュヘッダの時間領域ディスクリプタ（channels×samples×8 と
// timeDomainSizeBytes の一致）を検証せずに復元していたため、細工キャッシュで
// マップ外読み出しに至り得た。本テストは
//   (1) 正常な v2 / v1 キャッシュが従来どおり受理されること（互換性）
//   (2) 不整合なディスクリプタ・部分指定・上限超過・桁あふれが拒否されること
//   (3) 不正エントリが削除され、次回ロードで再構築にフォールバックできること
// を確認する。ASan 実行時は (2) がマップ外読み出しを起こさないことも確認できる。
//
// 実行: CacheManagerHeaderValidationTests

#include <JuceHeader.h>

#include "CacheManager.h"
#include "PreparedIRState.h"
#include "AlignedAllocation.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace
{

int gFailures = 0;

void check (bool ok, const char* id, const juce::String& detail = {})
{
    std::printf ("[%s] %s%s\n", ok ? "PASS" : "FAIL", id,
                 detail.isNotEmpty() ? (" " + detail).toRawUTF8() : "");
    if (! ok)
        ++gFailures;
}

// CacheHeaderV1 は CacheManager.cpp 内の匿名名前空間にあるため、レイアウトを写して
// v1 互換性を検査する（フィールド順・型は本番と同一）。
struct CacheHeaderV1ForTest
{
    uint64_t magic = 0x434F4E564F504551ULL;
    uint64_t version = 1;
    uint64_t key = 0;
    uint64_t dataSize = 0;
    uint64_t checksum = 0;
    uint64_t timestamp = 0;
    uint64_t fftSize = 0;
    uint64_t numPartitions = 0;
    uint64_t numChannels = 0;
    double   sampleRate = 0.0;
    uint64_t timeDomainChannels = 0;
    uint64_t timeDomainNumSamples = 0;
    uint64_t timeDomainSizeBytes = 0;
};

std::unique_ptr<PreparedIRState> makeState (int channels, int samples, int partitionDoubles)
{
    auto state = std::make_unique<PreparedIRState>();

    auto* p = static_cast<double*> (convo::aligned_malloc (sizeof (double) * (size_t) partitionDoubles, 64));
    for (int i = 0; i < partitionDoubles; ++i)
        p[i] = 0.001 * (i + 1);

    state->partitionData = p;
    state->partitionSizeBytes = sizeof (double) * (size_t) partitionDoubles;
    state->numPartitions = 4;
    state->fftSize = 512;
    state->numChannels = channels;
    state->sampleRate = 48000.0;

    if (samples > 0)
    {
        state->timeDomainIR = std::make_unique<juce::AudioBuffer<double>> (channels, samples);
        for (int ch = 0; ch < channels; ++ch)
            for (int i = 0; i < samples; ++i)
                state->timeDomainIR->setSample (ch, i, 0.01 * std::sin (0.01 * (double) i + (double) ch));
    }

    return state;
}

bool writeRaw (const juce::File& file,
               const void* header, size_t headerSize,
               const void* payload, size_t payloadSize,
               const void* timeDomain, size_t timeDomainSize)
{
    file.deleteFile();
    juce::FileOutputStream out (file);
    if (! out.openedOk())
        return false;

    out.write (header, headerSize);
    if (payloadSize > 0)
        out.write (payload, payloadSize);
    if (timeDomainSize > 0 && timeDomain != nullptr)
        out.write (timeDomain, timeDomainSize);

    out.flush();
    return out.getStatus().wasOk();
}

juce::File findCacheFile (const juce::File& dir)
{
    juce::Array<juce::File> files;
    dir.findChildFiles (files, juce::File::findFiles, false, "*.bin");
    return files.isEmpty() ? juce::File() : files.getFirst();
}

} // namespace

int main()
{
    juce::initialiseJuce_GUI();

    const auto tempDir = juce::File::getSpecialLocation (juce::File::tempDirectory)
                            .getChildFile ("convopeq_b02_"
                                           + juce::String::toHexString (juce::Time::currentTimeMillis()));
    tempDir.createDirectory();

    CacheManager cm (tempDir);

    const uint64_t key = 0x1122334455667788ULL;
    const int fft = 512;

    // ---- C-1: 正常な v2 の往復（互換性） ----
    {
        auto state = makeState (2, 1000, 64);
        cm.save (key, fft, *state);

        auto loaded = cm.loadPreparedState (key, fft, 1);
        check (loaded != nullptr, "C-1 valid v2 cache is accepted");
        if (loaded != nullptr)
        {
            check (loaded->timeDomainIR != nullptr
                   && loaded->timeDomainIR->getNumChannels() == 2
                   && loaded->timeDomainIR->getNumSamples() == 1000,
                   "C-1b time domain restored");
            check (loaded->partitionSizeBytes == state->partitionSizeBytes,
                   "C-1c partition payload size");
        }
    }

    const auto cacheFile = findCacheFile (tempDir);
    check (cacheFile.existsAsFile(), "C-0 cache file located");

    juce::MemoryBlock base;
    cacheFile.loadFileAsData (base);

    CacheHeader baseHeader {};
    std::memcpy (&baseHeader, base.getData(), sizeof (CacheHeader));
    const auto* basePayload = static_cast<const uint8_t*> (base.getData()) + sizeof (CacheHeader);

    // 正常な時間領域データ（2ch × 1000）
    const size_t validTdBytes = (size_t) baseHeader.timeDomainSizeBytes;
    check (validTdBytes == 2u * 1000u * sizeof (double), "C-0b baseline td size");

    auto rewrite = [&] (const CacheHeader& h, const void* td, size_t tdSize)
    {
        return writeRaw (cacheFile, &h, sizeof (h), basePayload, (size_t) h.dataSize, td, tdSize);
    };

    std::vector<double> zeroTd (validTdBytes / sizeof (double), 0.0);

    // ---- C-2: tdSizeBytes=8 / samples=1000（B-02 の再現条件） ----
    {
        CacheHeader h = baseHeader;
        h.timeDomainChannels = 1;
        h.timeDomainNumSamples = 1000;
        h.timeDomainSizeBytes = 8;
        rewrite (h, zeroTd.data(), 8);

        auto loaded = cm.loadPreparedState (key, fft, 1);
        check (loaded == nullptr, "C-2 mismatched time-domain size is rejected");
        check (! cacheFile.existsAsFile(), "C-2b invalid entry is removed");
    }

    // ---- C-3: 申告サイズが channels×samples×8 より大きい ----
    {
        CacheHeader h = baseHeader;
        h.timeDomainChannels = 2;
        h.timeDomainNumSamples = 1000;
        h.timeDomainSizeBytes = validTdBytes + 8;
        rewrite (h, zeroTd.data(), validTdBytes); // 実データは元の長さのまま

        check (cm.loadPreparedState (key, fft, 1) == nullptr, "C-3 overstated td size is rejected");
    }

    // ---- C-4: 部分指定（channels だけ 0） ----
    {
        CacheHeader h = baseHeader;
        h.timeDomainChannels = 0;
        h.timeDomainNumSamples = 1000;
        h.timeDomainSizeBytes = 8000;
        rewrite (h, zeroTd.data(), validTdBytes);

        check (cm.loadPreparedState (key, fft, 1) == nullptr, "C-4 partial descriptor is rejected");
    }

    // ---- C-5: channels 上限（9 > 8） ----
    {
        CacheHeader h = baseHeader;
        h.timeDomainChannels = 9;
        h.timeDomainNumSamples = 100;
        h.timeDomainSizeBytes = 9u * 100u * sizeof (double);
        std::vector<double> td ((size_t) (9u * 100u), 0.0);
        rewrite (h, td.data(), td.size() * sizeof (double));

        check (cm.loadPreparedState (key, fft, 1) == nullptr, "C-5 channel cap is enforced");
    }

    // ---- C-6: 桁あふれを狙う channels（UINT64_MAX） ----
    {
        CacheHeader h = baseHeader;
        h.timeDomainChannels = ~0ULL;
        h.timeDomainNumSamples = 1;
        h.timeDomainSizeBytes = 8;
        rewrite (h, zeroTd.data(), 8);

        check (cm.loadPreparedState (key, fft, 1) == nullptr, "C-6 overflow-sized descriptor is rejected");
    }

    // ---- C-7: キー不一致 ----
    {
        CacheHeader h = baseHeader;
        h.key = key ^ 1ULL;
        rewrite (h, zeroTd.data(), validTdBytes);

        check (cm.loadPreparedState (key, fft, 1) == nullptr, "C-7 key mismatch is rejected");
        check (! cacheFile.existsAsFile(), "C-7b mismatched entry is removed");
    }

    // ---- C-8: partition CRC 不一致 ----
    {
        CacheHeader h = baseHeader;
        std::vector<uint8_t> payload ((size_t) h.dataSize, 0);
        std::memcpy (payload.data(), basePayload, (size_t) h.dataSize);
        payload[0] ^= 0xFF;

        writeRaw (cacheFile, &h, sizeof (h), payload.data(), payload.size(),
                  zeroTd.data(), validTdBytes);

        check (cm.loadPreparedState (key, fft, 1) == nullptr, "C-8 payload CRC mismatch is rejected");
    }

    // ---- C-9: v1（時間領域なし）の互換性 ----
    {
        CacheHeaderV1ForTest h {};
        h.key = key;
        h.dataSize = baseHeader.dataSize;
        h.checksum = baseHeader.checksum;
        h.fftSize = (uint64_t) fft;
        h.numPartitions = baseHeader.numPartitions;
        h.numChannels = baseHeader.numChannels;
        h.sampleRate = baseHeader.sampleRate;

        writeRaw (cacheFile, &h, sizeof (h), basePayload, (size_t) h.dataSize, nullptr, 0);

        auto loaded = cm.loadPreparedState (key, fft, 1);
        check (loaded != nullptr, "C-9 v1 cache without time domain is accepted");
        if (loaded != nullptr)
            check (loaded->timeDomainIR == nullptr, "C-9b v1 without td has no time domain");
    }

    // ---- C-9c: v1（正しい時間領域あり）の互換性 ----
    {
        CacheHeaderV1ForTest h {};
        h.key = key;
        h.dataSize = baseHeader.dataSize;
        h.checksum = baseHeader.checksum;
        h.fftSize = (uint64_t) fft;
        h.numPartitions = baseHeader.numPartitions;
        h.numChannels = baseHeader.numChannels;
        h.sampleRate = baseHeader.sampleRate;
        h.timeDomainChannels = 1;
        h.timeDomainNumSamples = 100;
        h.timeDomainSizeBytes = 100u * sizeof (double);

        std::vector<double> td (100, 0.25);
        writeRaw (cacheFile, &h, sizeof (h), basePayload, (size_t) h.dataSize,
                  td.data(), td.size() * sizeof (double));

        auto loaded = cm.loadPreparedState (key, fft, 1);
        check (loaded != nullptr, "C-9c v1 cache with valid time domain is accepted");
        if (loaded != nullptr)
            check (loaded->timeDomainIR != nullptr
                   && loaded->timeDomainIR->getNumSamples() == 100,
                   "C-9d v1 time domain restored");
    }

    // ---- C-10: v2（時間領域なし）の互換性 ----
    {
        CacheHeader h = baseHeader;
        h.timeDomainChannels = 0;
        h.timeDomainNumSamples = 0;
        h.timeDomainSizeBytes = 0;
        rewrite (h, nullptr, 0);

        auto loaded = cm.loadPreparedState (key, fft, 1);
        check (loaded != nullptr, "C-10 v2 cache without time domain is accepted");
        if (loaded != nullptr)
            check (loaded->timeDomainIR == nullptr, "C-10b v2 without td has no time domain");
    }

    // ---- C-11: B-08 ディレクトリ総量の剪定 ----
    {
        auto pruneDir = juce::File::getSpecialLocation (juce::File::tempDirectory)
                            .getChildFile ("convo_b08_prune");
        pruneDir.deleteRecursively();
        pruneDir.createDirectory();

        const auto makeFile = [&pruneDir] (const juce::String& name, int bytes, int ageSeconds)
        {
            auto f = pruneDir.getChildFile (name);
            f.replaceWithText (juce::String::repeatedString ("x", bytes));
            f.setLastModificationTime (juce::Time::getCurrentTime()
                                       - juce::RelativeTime::seconds (static_cast<double> (ageSeconds)));
            return f;
        };

        const auto oldest = makeFile ("0000000000000001_512.bin", 1024, 300);
        const auto middle = makeFile ("0000000000000002_512.bin", 1024, 200);
        const auto newest = makeFile ("0000000000000003_512.bin", 1024, 100);
        const auto decoy  = makeFile ("notes.txt", 4096, 400);
        const auto decoy2 = makeFile ("not-a-cache.bin", 4096, 400);

        {
            CacheManager pruneCm (pruneDir); // 既定上限では何も消えない
            check (oldest.existsAsFile() && middle.existsAsFile() && newest.existsAsFile(),
                   "C-11a constructor keeps entries under the default limit");

            pruneCm.pruneDirectory (1024);

            check (! oldest.existsAsFile() && ! middle.existsAsFile() && newest.existsAsFile(),
                   "C-11b prune removes oldest entries down to the limit");
            check (decoy.existsAsFile() && decoy2.existsAsFile(),
                   "C-11c prune leaves non-cache files untouched");
        }

        pruneDir.deleteRecursively();
    }

    tempDir.deleteRecursively();

    // ---- C-12: キャッシュキーの意味論バージョン（B-01/B-02） ----
    //   末尾無音トリムを同期経路にも適用したため、同じファイルでも IR の長さが変わる。
    //   トリム導入前のキー（v1）と現行キー（v2）が一致しないことを固定する。
    {
        const auto keyFile = tempDir.getChildFile ("c12_key_source.bin");
        keyFile.replaceWithText ("trim-semantics-v2");

        const auto k1 = CacheManager::computeKey (keyFile, fft, 48000.0, 0, 512, 1);
        const auto k2 = CacheManager::computeKey (keyFile, fft, 48000.0, 0, 512, 2);
        const auto kDefault = CacheManager::computeKey (keyFile, fft, 48000.0, 0, 512);

        check (k1 != k2, "C-12a", "schema v1 and v2 produce different keys");
        check (kDefault == k2, "C-12b", "default uses the current schema version");
        check (CacheManager::kCacheKeySchemaVersion >= 2, "C-12c", "trim semantics are versioned");
    }

    std::printf ("[SUMMARY] failures=%d\n", gFailures);
    juce::shutdownJuce_GUI();
    return gFailures == 0 ? 0 : 1;
}
