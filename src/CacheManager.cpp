#include "CacheManager.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>

#include <mkl.h>
#include <mkl_cblas.h>
#include "DiagnosticsConfig.h"
#include "convolver/IRLoadAdmission.h"

namespace
{
constexpr uint64_t kCRC64Poly = 0x42F0E1EBA9EA3693ULL;

struct CacheHeaderV1
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
    double sampleRate = 0.0;
    uint64_t timeDomainChannels = 0;
    uint64_t timeDomainNumSamples = 0;
    uint64_t timeDomainSizeBytes = 0;
};

uint64_t crc64Update(uint64_t crc, const uint8_t* data, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        crc ^= (static_cast<uint64_t>(data[i]) << 56);
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc & 0x8000000000000000ULL) ? ((crc << 1) ^ kCRC64Poly) : (crc << 1);
    }
    return crc;
}

// B-02: 時間領域ディスクリプタの整合検査。
// 上限は IRLoadAdmission の凍結値（8ch / 2^21+1）を再利用し、新しい値を導入しない。
// 乗算はオーバーフローを先に検査してから行う（未検証値の積を作らない）。
bool validateTimeDomainDescriptor(const CacheHeader& h, juce::String& why)
{
    const bool hasChannels = h.timeDomainChannels > 0;
    const bool hasSamples  = h.timeDomainNumSamples > 0;
    const bool hasBytes    = h.timeDomainSizeBytes > 0;

    if (! (hasChannels == hasSamples && hasSamples == hasBytes))
    {
        why = "partial time-domain descriptor";
        return false;
    }

    if (! hasChannels)
        return true; // 時間領域なし（従来のキャッシュ）

    if (h.timeDomainChannels > static_cast<uint64_t>(convo::irload::kMaxIRLoadChannels))
    {
        why = "time-domain channels exceed the IR admission contract (8)";
        return false;
    }

    if (h.timeDomainNumSamples > static_cast<uint64_t>(convo::irload::kMaxIRResampleOutputSamples))
    {
        why = "time-domain samples exceed the IR admission contract (2^21+1)";
        return false;
    }

    if (h.timeDomainNumSamples > std::numeric_limits<size_t>::max() / sizeof(double)
                                                          / h.timeDomainChannels)
    {
        why = "time-domain byte size overflows size_t";
        return false;
    }

    const uint64_t expectedBytes = h.timeDomainChannels
                                 * h.timeDomainNumSamples
                                 * static_cast<uint64_t>(sizeof(double));
    if (h.timeDomainSizeBytes != expectedBytes)
    {
        why = "time-domain size does not match channels*samples*8";
        return false;
    }

    return true;
}
}

uint64_t CacheManager::hashCombine(uint64_t seed, uint64_t value)
{
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

uint64_t CacheManager::makeEntryKey(uint64_t key, int fftSize)
{
    return hashCombine(key, static_cast<uint64_t>(fftSize));
}

uint64_t CacheManager::computeCRC64(const uint8_t* data, size_t size)
{
    return crc64Update(0ULL, data, size);
}

uint64_t CacheManager::computeFileContentCRC(const juce::File& file)
{
    if (!file.existsAsFile())
        return 0;

    std::unique_ptr<juce::FileInputStream> stream(file.createInputStream());
    if (!stream)
        return 0;

    constexpr int kChunkSize = 64 * 1024;
    std::vector<uint8_t> buffer(static_cast<size_t>(kChunkSize));

    uint64_t crc = 0ULL;
    while (!stream->isExhausted())
    {
        const int readBytes = stream->read(buffer.data(), kChunkSize);
        if (readBytes <= 0)
            break;
        crc = crc64Update(crc, buffer.data(), static_cast<size_t>(readBytes));
    }

    return crc;
}

uint64_t CacheManager::computeKey(const juce::File& file,
                                  int fftSize,
                                  double sampleRate,
                                  int phaseMode,
                                  int partitionSize,
                                  uint64_t schemaVersion)
{
    // version 1 はトリム導入前のキー（CRC から開始）。version 2 以降は
    // バージョン値を混ぜて、意味論が変わったキャッシュを再利用しない。
    uint64_t seed = (schemaVersion >= 2)
        ? hashCombine(schemaVersion, computeFileContentCRC(file))
        : computeFileContentCRC(file);
    seed = hashCombine(seed, static_cast<uint64_t>(fftSize));
    seed = hashCombine(seed, static_cast<uint64_t>(phaseMode));
    seed = hashCombine(seed, static_cast<uint64_t>(partitionSize));

    uint64_t srBits = 0;
    static_assert(sizeof(double) == sizeof(uint64_t), "unexpected double size");
    std::memcpy(&srBits, &sampleRate, sizeof(double));
    seed = hashCombine(seed, srBits);

    return seed;
}

CacheManager::CacheManager(juce::File cacheDirOverride)
    : cacheDirectoryOverride(std::move(cacheDirOverride))
{
    // ★ B-08: 跨セッションで増え続ける IRCache を、起動時に一度だけ上限まで剪定する。
    //   セッション中の evictLRU（件数上限）とは別の、総量の歯止め。
    pruneDirectory();
}

juce::File CacheManager::getCacheDirectory() const
{
    auto dir = (cacheDirectoryOverride != juce::File())
                   ? cacheDirectoryOverride
                   : juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                         .getChildFile("ConvoPeq")
                         .getChildFile("IRCache");
    if (!dir.exists())
     {
         auto result = dir.createDirectory();
         if (!result.wasOk())
             juce::Logger::writeToLog("Warning: Could not create cache directory");
     }
    return dir;
}

juce::File CacheManager::getCacheFile(uint64_t key, int fftSize) const
{
    return getCacheDirectory().getChildFile(juce::String::toHexString(static_cast<int64>(key))
                                            + "_"
                                            + juce::String(fftSize)
                                            + ".bin");
}

bool CacheManager::validateCacheFile(const juce::File& file, uint64_t expectedKey, int expectedFftSize, CacheHeader& headerOut) const
{
    if (!file.existsAsFile())
        return false;

    std::unique_ptr<juce::FileInputStream> stream(file.createInputStream());
    if (!stream)
        return false;

    CacheHeaderV1 headerV1{};
    if (stream->read(&headerV1, static_cast<int>(sizeof(CacheHeaderV1))) != static_cast<int>(sizeof(CacheHeaderV1)))
        return false;

    if (headerV1.magic != 0x434F4E564F504551ULL)
        return false;

    if (headerV1.version == 1)
    {
        headerOut.magic = headerV1.magic;
        headerOut.version = headerV1.version;
        headerOut.key = headerV1.key;
        headerOut.dataSize = headerV1.dataSize;
        headerOut.checksum = headerV1.checksum;
        headerOut.timestamp = headerV1.timestamp;
        headerOut.fftSize = headerV1.fftSize;
        headerOut.numPartitions = headerV1.numPartitions;
        headerOut.numChannels = headerV1.numChannels;
        headerOut.sampleRate = headerV1.sampleRate;
        headerOut.timeDomainChannels = headerV1.timeDomainChannels;
        headerOut.timeDomainNumSamples = headerV1.timeDomainNumSamples;
        headerOut.timeDomainSizeBytes = headerV1.timeDomainSizeBytes;
        headerOut.scaleFactor = 1.0;
        headerOut.hasScaleFactor = 0;
    }
    else if (headerV1.version == 2)
    {
        stream->setPosition(0);
        if (stream->read(&headerOut, static_cast<int>(sizeof(CacheHeader))) != static_cast<int>(sizeof(CacheHeader)))
            return false;
    }
    else
    {
        return false;
    }

    if (headerOut.key != expectedKey)
        return false;

    if (headerOut.fftSize != static_cast<uint64_t>(expectedFftSize))
        return false;

    const uint64_t headerSize = (headerOut.version == 1)
                              ? static_cast<uint64_t>(sizeof(CacheHeaderV1))
                              : static_cast<uint64_t>(sizeof(CacheHeader));

    // B-02: 総サイズは符号なしで、減算の前に各項の上限を確かめる（桁あふれを作らない）。
    const juce::int64 fileSizeSigned = file.getSize();
    if (fileSizeSigned < 0)
        return false;

    const uint64_t fileSize = static_cast<uint64_t>(fileSizeSigned);
    if (headerSize > fileSize)
        return false;

    const uint64_t payloadSize = fileSize - headerSize;
    if (headerOut.dataSize > payloadSize)
        return false;
    if (headerOut.timeDomainSizeBytes > payloadSize - headerOut.dataSize)
        return false;
    if (headerOut.dataSize + headerOut.timeDomainSizeBytes != payloadSize)
        return false;

    // int への cast が負にならないことを先に確認する。
    if (headerOut.numPartitions > static_cast<uint64_t>(std::numeric_limits<int>::max())
        || headerOut.numChannels > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return false;

    // B-02: channels×samples×8 と timeDomainSizeBytes の一致（マップ外読み出しの根因）。
    juce::String why;
    if (! validateTimeDomainDescriptor(headerOut, why))
    {
        DBG("CacheManager: rejected cache header: " << why);
        return false;
    }

    return true;
}

double* CacheManager::copyFromMmapToAligned(juce::MemoryMappedFile& mmap, size_t dataSize)
{
    const auto* src = static_cast<const uint8_t*>(mmap.getData());
    if (!src)
        return nullptr;

    double* dst = static_cast<double*>(DIAG_MKL_MALLOC(dataSize, 64));
    if (!dst)
        return nullptr;

    std::memcpy(dst, src, dataSize);

    // Warm up pages to reduce first-touch faults on audio-adjacent paths.
    // ★ work92 C-7 (big 3-1): volatile sink を atomic_signal_fence パターンに置換。
    //   volatile 変数はデータ競合セマンティクスを持たず、最適化抑止も保証されない。
    //   signal fence はコンパイラにメモリアクセス観測を強制する（実行時コスト 0）。
    constexpr size_t kPage = 4096;
    uint8_t* raw = reinterpret_cast<uint8_t*>(dst);
    uint8_t sink = 0;
    for (size_t i = 0; i < dataSize; i += kPage)
    {
        std::atomic_signal_fence(std::memory_order_acquire); // 読み取りの除去を抑止
        sink ^= raw[i];
    }
    (void)sink;
    (void)sink;

    return dst;
}

std::unique_ptr<PreparedIRState> CacheManager::loadPreparedState(uint64_t key, int fftSize, uint64_t generationId)
{
    return loadPreparedStateFromFile(getCacheFile(key, fftSize), key, fftSize, generationId);
}

std::unique_ptr<PreparedIRState> CacheManager::loadPreparedStateFromFile(const juce::File& file,
                                                                        uint64_t key,
                                                                        int fftSize,
                                                                        uint64_t generationId)
{
    CacheHeader header{};
    if (!validateCacheFile(file, key, fftSize, header))
    {
        // B-02: 構造的に不正なエントリは再構築可能なので削除する（毎回の再試行を避ける）。
        // ファイル名はキーから決まるため、キー不一致もこの名前のエントリが不正であることを意味する。
        if (file.existsAsFile())
        {
            juce::Logger::writeToLog("[CACHE] rejected and removed invalid cache entry: " + file.getFileName());
            file.deleteFile();
        }
        return nullptr;
    }

    juce::MemoryMappedFile mmap(file, juce::MemoryMappedFile::readOnly);
    const size_t headerSize = (header.version == 1) ? sizeof(CacheHeaderV1) : sizeof(CacheHeader);
    if (mmap.getSize() <= headerSize)
        return nullptr;

    const auto* mapped = static_cast<const uint8_t*>(mmap.getData());
    if (!mapped)
        return nullptr;

    const uint8_t* dataStart = mapped + headerSize;
    const uint64_t checksum = computeCRC64(dataStart, static_cast<size_t>(header.dataSize));
    if (checksum != header.checksum)
        return nullptr;

    double* copied = static_cast<double*>(DIAG_MKL_MALLOC(static_cast<size_t>(header.dataSize), 64));
    if (!copied)
        return nullptr;

    std::memcpy(copied, dataStart, static_cast<size_t>(header.dataSize));

    // Warm up pages to reduce first-touch faults. (C-7: signal fence パターン — 上記参照)
    constexpr size_t kPage = 4096;
    uint8_t* raw = reinterpret_cast<uint8_t*>(copied);
    uint8_t sink = 0;
    for (size_t i = 0; i < static_cast<size_t>(header.dataSize); i += kPage)
    {
        std::atomic_signal_fence(std::memory_order_acquire); // 読み取りの除去を抑止
        sink ^= raw[i];
    }
    (void)sink;
    (void)sink;

    auto prepared = std::make_unique<PreparedIRState>();
    prepared->partitionData = copied;
    prepared->partitionSizeBytes = static_cast<size_t>(header.dataSize);
    prepared->numPartitions = static_cast<int>(header.numPartitions);
    prepared->fftSize = static_cast<int>(header.fftSize);
    prepared->numChannels = static_cast<int>(header.numChannels);
    prepared->sampleRate = header.sampleRate;
    prepared->generationId = generationId;
    prepared->cacheKey = key;

    // timeDomainIR があれば復元（B-02: 確保の直前にもう一度ディスクリプタを検証する）
    if (header.timeDomainSizeBytes > 0 && header.timeDomainChannels > 0 && header.timeDomainNumSamples > 0)
    {
        juce::String why;
        if (! validateTimeDomainDescriptor(header, why))
        {
            juce::Logger::writeToLog("[CACHE] rejected invalid time-domain descriptor: " + why);
            return nullptr;
        }

        // 符号なしで境界を先に確かめる（減算の桁あふれを作らない）。
        const uint64_t mappedSize = static_cast<uint64_t>(juce::jmax<juce::int64>(0, mmap.getSize()));
        const uint64_t tdOffset = static_cast<uint64_t>(headerSize) + header.dataSize;
        const uint64_t expectedTdBytes = header.timeDomainSizeBytes;
        if (mappedSize < tdOffset || mappedSize - tdOffset < expectedTdBytes)
        {
            juce::Logger::writeToLog("[CACHE] truncated time-domain payload");
            return nullptr;
        }

        const uint8_t* tdStart = mapped + tdOffset;
        const auto tdChannels = static_cast<int>(header.timeDomainChannels);
        const auto tdSamples = static_cast<int>(header.timeDomainNumSamples);

        auto tdBuffer = std::make_unique<juce::AudioBuffer<double>>(tdChannels, tdSamples);
        // ★ work92 C-4 (big 2-4): strict-aliasing 違反の解消。
        //   mmap 先頭は uint8_t バッファであり double としての有効型を持たないため
        //   reinterpret_cast<const double*> の逆参照は UB。バイトオフセット + memcpy
        //   （逆参照なしのコピー）に置換する。数値は bit 単位で同一。
        const uint8_t* tdCursor = tdStart;
        const size_t rowBytes = static_cast<size_t>(tdSamples) * sizeof(double);
        for (int ch = 0; ch < tdChannels; ++ch)
        {
            std::memcpy(tdBuffer->getWritePointer(ch), tdCursor, rowBytes);
            tdCursor += rowBytes;
        }
        prepared->timeDomainIR = std::move(tdBuffer);
    }

    if (header.version < 2)
    {
        DBG("CacheManager: loading old cache version " << header.version << ", recalculating scaleFactor");
        if (prepared->timeDomainIR && prepared->timeDomainIR->getNumSamples() > 0)
        {
            double maxEnergy = 0.0;
            const int ns = prepared->timeDomainIR->getNumSamples();
            const int nc = prepared->timeDomainIR->getNumChannels();
            for (int ch = 0; ch < nc; ++ch)
            {
                const double* data = prepared->timeDomainIR->getReadPointer(ch);
                const double energy = cblas_ddot(ns, data, 1, data, 1);
                if (std::isfinite(energy) && energy > maxEnergy)
                    maxEnergy = energy;
            }

            double sf = 1.0;
            if (maxEnergy > 1.0e-12)
                sf = (1.0 / std::sqrt(maxEnergy)) * 0.5011872336272722;

            for (int ch = 0; ch < nc; ++ch)
                cblas_dscal(ns, sf, prepared->timeDomainIR->getWritePointer(ch), 1);

            if (prepared->partitionData && prepared->partitionSizeBytes > 0)
            {
                const size_t numDoubles = prepared->partitionSizeBytes / sizeof(double);
                cblas_dscal(static_cast<MKL_INT>(numDoubles), sf, prepared->partitionData, 1);
            }

            // 旧フォーマットはここでデータ自体を補正済みのため、以降で再適用しない。
            prepared->scaleFactor = 1.0;
            prepared->hasScaleFactor = false;
            DBG("CacheManager: recalculated scaleFactor = " << sf << " (applied in-place)");
        }
        else
        {
            prepared->scaleFactor = 1.0;
            prepared->hasScaleFactor = false;
        }
    }
    else
    {
        prepared->scaleFactor = header.scaleFactor;
        prepared->hasScaleFactor = (header.hasScaleFactor != 0);
    }

    touch(key, fftSize);

    return prepared;
}

void CacheManager::save(uint64_t key, int fftSize, const PreparedIRState& state)
{
    if (!state.partitionData || state.partitionSizeBytes == 0)
        return;

    const auto file = getCacheFile(key, fftSize);
    const auto temp = file.withFileExtension("tmp");

    CacheHeader header{};
    header.key = key;
    header.dataSize = static_cast<uint64_t>(state.partitionSizeBytes);
    header.checksum = computeCRC64(reinterpret_cast<const uint8_t*>(state.partitionData), state.partitionSizeBytes);
    header.timestamp = static_cast<uint64_t>(juce::Time::getCurrentTime().toMilliseconds());
    header.fftSize = static_cast<uint64_t>(fftSize);
    header.numPartitions = static_cast<uint64_t>(state.numPartitions);
    header.numChannels = static_cast<uint64_t>(state.numChannels);
    header.sampleRate = state.sampleRate;
    header.scaleFactor = state.scaleFactor;
    header.hasScaleFactor = state.hasScaleFactor ? 1ULL : 0ULL;
    header.version = 2;
    if (state.timeDomainIR)
    {
        header.timeDomainChannels = static_cast<uint64_t>(state.timeDomainIR->getNumChannels());
        header.timeDomainNumSamples = static_cast<uint64_t>(state.timeDomainIR->getNumSamples());
        header.timeDomainSizeBytes = static_cast<uint64_t>(state.timeDomainIR->getNumChannels())
                                   * static_cast<uint64_t>(state.timeDomainIR->getNumSamples())
                                   * sizeof(double);
    }
    else
    {
        header.timeDomainChannels = 0;
        header.timeDomainNumSamples = 0;
        header.timeDomainSizeBytes = 0;
    }

    std::unique_ptr<juce::FileOutputStream> out(temp.createOutputStream());
    if (!out)
        return;

    out->write(&header, static_cast<size_t>(sizeof(CacheHeader)));
    out->write(state.partitionData, state.partitionSizeBytes);
    if (state.timeDomainIR && header.timeDomainSizeBytes > 0)
    {
        const int channels = state.timeDomainIR->getNumChannels();
        const int samples = state.timeDomainIR->getNumSamples();
        for (int ch = 0; ch < channels; ++ch)
            out->write(state.timeDomainIR->getReadPointer(ch),
                       static_cast<size_t>(samples) * sizeof(double));
    }
    out->flush();
    if (out->getStatus().failed())
    {
        out.reset();
        temp.deleteFile();
        return;
    }
    out.reset();

    if (!temp.moveFileTo(file))
    {
        temp.deleteFile();
    }
    touch(key, fftSize);
}

void CacheManager::touch(uint64_t key, int fftSize)
{
    const uint64_t entryKey = makeEntryKey(key, fftSize);
    const auto file = getCacheFile(key, fftSize);
    std::lock_guard<std::mutex> lock(cacheMutex);

    auto it = cacheMap.find(entryKey);
    const uint64_t nowMs = static_cast<uint64_t>(juce::Time::getCurrentTime().toMilliseconds());
    if (it != cacheMap.end())
    {
        lruList.erase(it->second.lruPos);
        lruList.push_front(entryKey);
        it->second.lruPos = lruList.begin();
        it->second.lastAccessTime = nowMs;
        it->second.file = file;
        it->second.originalKey = key;
        return;
    }

    lruList.push_front(entryKey);
    CacheEntry entry;
    entry.file = file;
    entry.originalKey = key;
    entry.lastAccessTime = nowMs;
    entry.fftSize = fftSize;
    entry.lruPos = lruList.begin();
    cacheMap.emplace(entryKey, std::move(entry));
}

void CacheManager::setSafeDeleteChecker(SafeDeleteFn checker)
{
    std::lock_guard<std::mutex> lock(cacheMutex);
    safeDeleteChecker = std::move(checker);
}

void CacheManager::pruneDirectory(juce::int64 maxTotalBytes)
{
    if (maxTotalBytes <= 0)
        return;

    const auto dir = getCacheDirectory();

    struct Entry
    {
        juce::File file;
        juce::int64 size = 0;
        juce::Time modified;
        uint64_t key = 0;
        int fftSize = 0;
    };

    std::vector<Entry> entries;
    juce::int64 total = 0;

    // 対象はこのクラスが書く命名（<16 桁 hex>_<fft>.bin）に一致するものだけ。
    // 他のアプリや他バージョンのファイルには触れない。
    for (const auto& f : dir.findChildFiles (juce::File::findFiles, false, "*.bin"))
    {
        const auto base = f.getFileNameWithoutExtension();
        const int sep = base.lastIndexOfChar ('_');

        if (sep <= 0 || sep + 1 >= base.length())
            continue;

        const auto hexPart = base.substring (0, sep);
        const auto fftPart = base.substring (sep + 1);

        if (hexPart.length() != 16 || ! hexPart.containsOnly ("0123456789abcdefABCDEF"))
            continue;

        if (fftPart.isEmpty() || ! fftPart.containsOnly ("0123456789"))
            continue;

        Entry entry;
        entry.file = f;
        entry.size = f.getSize();
        entry.modified = f.getLastModificationTime();
        entry.key = static_cast<uint64_t> (hexPart.getHexValue64());
        entry.fftSize = fftPart.getIntValue();

        total += entry.size;
        entries.push_back (std::move (entry));
    }

    if (total <= maxTotalBytes)
        return;

    std::sort (entries.begin(), entries.end(), [] (const Entry& a, const Entry& b)
    {
        return a.modified < b.modified; // 古い順
    });

    int removed = 0;

    for (const auto& entry : entries)
    {
        if (total <= maxTotalBytes)
            break;

        if (! isEntrySafeToDelete (entry.key, entry.fftSize))
            continue;

        // Windows で mmap 中のファイルは deleteFile が false を返すため、
        // 使用中エントリを巻き込むことはない。
        if (entry.file.deleteFile())
        {
            total -= entry.size;
            ++removed;
        }
    }

    juce::Logger::writeToLog ("[CACHE] prune: removed " + juce::String (removed)
                              + " entries, total now " + juce::String (total)
                              + " bytes (limit " + juce::String (maxTotalBytes) + ")");
}

void CacheManager::clear()
{
    const auto dir = getCacheDirectory();
    if (dir.exists())
        dir.deleteRecursively();
    {
        auto result = dir.createDirectory();
        if (!result.wasOk())
            juce::Logger::writeToLog("Warning: Could not recreate cache directory");
    }

    std::lock_guard<std::mutex> lock(cacheMutex);
    cacheMap.clear();
    lruList.clear();
}

bool CacheManager::isEntrySafeToDelete(uint64_t key, int fftSize)
{
    if (!safeDeleteChecker)
        return true;

    return safeDeleteChecker(key, fftSize);
}

void CacheManager::evictLRU(size_t maxEntries)
{
    std::lock_guard<std::mutex> lock(cacheMutex);

    while (cacheMap.size() > maxEntries && !lruList.empty())
    {
        bool removed = false;

        for (auto rit = lruList.rbegin(); rit != lruList.rend(); ++rit)
        {
            const uint64_t entryKey = *rit;
            auto it = cacheMap.find(entryKey);
            if (it == cacheMap.end())
                continue;

            if (!isEntrySafeToDelete(it->second.originalKey, it->second.fftSize))
                continue;

            it->second.file.deleteFile();
            auto erasePos = std::next(rit).base();
            lruList.erase(erasePos);
            cacheMap.erase(it);
            removed = true;
            break;
        }

        if (!removed)
            break;
    }
}
