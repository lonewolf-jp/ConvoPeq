#include "CacheManager.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include <mkl.h>

namespace
{
constexpr uint64_t kCRC64Poly = 0x42F0E1EBA9EA3693ULL;

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
}

uint64_t CacheManager::hashCombine(uint64_t seed, uint64_t value)
{
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
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
                                  int partitionSize)
{
    uint64_t seed = computeFileContentCRC(file);
    seed = hashCombine(seed, static_cast<uint64_t>(fftSize));
    seed = hashCombine(seed, static_cast<uint64_t>(phaseMode));
    seed = hashCombine(seed, static_cast<uint64_t>(partitionSize));

    uint64_t srBits = 0;
    static_assert(sizeof(double) == sizeof(uint64_t), "unexpected double size");
    std::memcpy(&srBits, &sampleRate, sizeof(double));
    seed = hashCombine(seed, srBits);

    return seed;
}

juce::File CacheManager::getCacheDirectory() const
{
    auto dir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                   .getChildFile("ConvoPeq")
                   .getChildFile("IRCache");
    if (!dir.exists())
        dir.createDirectory();
    return dir;
}

juce::File CacheManager::getCacheFile(uint64_t key) const
{
    return getCacheDirectory().getChildFile(juce::String::toHexString(static_cast<int64>(key)) + ".bin");
}

bool CacheManager::validateCacheFile(const juce::File& file, uint64_t expectedKey, CacheHeader& headerOut) const
{
    if (!file.existsAsFile())
        return false;

    std::unique_ptr<juce::FileInputStream> stream(file.createInputStream());
    if (!stream)
        return false;

    if (stream->read(&headerOut, static_cast<int>(sizeof(CacheHeader))) != static_cast<int>(sizeof(CacheHeader)))
        return false;

    if (headerOut.magic != 0x434F4E564F504551ULL)
        return false;

    if (headerOut.version != 1)
        return false;

    if (headerOut.key != expectedKey)
        return false;

    const int64 expectedTotalSize = static_cast<int64>(sizeof(CacheHeader)) + static_cast<int64>(headerOut.dataSize);
    if (file.getSize() != expectedTotalSize)
        return false;

    return true;
}

double* CacheManager::copyFromMmapToAligned(juce::MemoryMappedFile& mmap, size_t dataSize)
{
    const auto* src = static_cast<const uint8_t*>(mmap.getData());
    if (!src)
        return nullptr;

    double* dst = static_cast<double*>(mkl_malloc(dataSize, 64));
    if (!dst)
        return nullptr;

    std::memcpy(dst, src, dataSize);

    // Warm up pages to reduce first-touch faults on audio-adjacent paths.
    constexpr size_t kPage = 4096;
    volatile uint8_t sink = 0;
    uint8_t* raw = reinterpret_cast<uint8_t*>(dst);
    for (size_t i = 0; i < dataSize; i += kPage)
        sink ^= raw[i];
    (void)sink;

    return dst;
}

std::unique_ptr<PreparedIRState> CacheManager::load(uint64_t key, uint64_t generationId)
{
    CacheHeader header{};
    const auto file = getCacheFile(key);
    if (!validateCacheFile(file, key, header))
        return nullptr;

    juce::MemoryMappedFile mmap(file, juce::MemoryMappedFile::readOnly);
    if (mmap.getSize() <= static_cast<size_t>(sizeof(CacheHeader)))
        return nullptr;

    const auto* mapped = static_cast<const uint8_t*>(mmap.getData());
    if (!mapped)
        return nullptr;

    const uint8_t* dataStart = mapped + sizeof(CacheHeader);
    const uint64_t checksum = computeCRC64(dataStart, static_cast<size_t>(header.dataSize));
    if (checksum != header.checksum)
        return nullptr;

    double* copied = static_cast<double*>(mkl_malloc(static_cast<size_t>(header.dataSize), 64));
    if (!copied)
        return nullptr;

    std::memcpy(copied, dataStart, static_cast<size_t>(header.dataSize));

    // Warm up pages to reduce first-touch faults.
    constexpr size_t kPage = 4096;
    volatile uint8_t sink = 0;
    uint8_t* raw = reinterpret_cast<uint8_t*>(copied);
    for (size_t i = 0; i < static_cast<size_t>(header.dataSize); i += kPage)
        sink ^= raw[i];
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

    return prepared;
}

void CacheManager::save(uint64_t key, const PreparedIRState& state)
{
    if (!state.partitionData || state.partitionSizeBytes == 0)
        return;

    const auto file = getCacheFile(key);
    const auto temp = file.withFileExtension("tmp");

    CacheHeader header{};
    header.key = key;
    header.dataSize = static_cast<uint64_t>(state.partitionSizeBytes);
    header.checksum = computeCRC64(reinterpret_cast<const uint8_t*>(state.partitionData), state.partitionSizeBytes);
    header.timestamp = static_cast<uint64_t>(juce::Time::getCurrentTime().toMilliseconds());
    header.fftSize = static_cast<uint64_t>(state.fftSize);
    header.numPartitions = static_cast<uint64_t>(state.numPartitions);
    header.numChannels = static_cast<uint64_t>(state.numChannels);
    header.sampleRate = state.sampleRate;

    std::unique_ptr<juce::FileOutputStream> out(temp.createOutputStream());
    if (!out)
        return;

    out->write(&header, static_cast<size_t>(sizeof(CacheHeader)));
    out->write(state.partitionData, state.partitionSizeBytes);
    out->flush();

    temp.moveFileTo(file);
}

void CacheManager::clear()
{
    const auto dir = getCacheDirectory();
    if (dir.exists())
        dir.deleteRecursively();
    dir.createDirectory();
}

void CacheManager::evictOldest(size_t maxEntries)
{
    const auto dir = getCacheDirectory();
    if (!dir.exists())
        return;

    juce::Array<juce::File> files;
    dir.findChildFiles(files, juce::File::findFiles, false, "*.bin");

    if (static_cast<size_t>(files.size()) <= maxEntries)
        return;

    std::sort(files.begin(), files.end(), [](const juce::File& a, const juce::File& b)
    {
        return a.getLastModificationTime() < b.getLastModificationTime();
    });

    const size_t removeCount = static_cast<size_t>(files.size()) - maxEntries;
    for (size_t i = 0; i < removeCount; ++i)
        files[static_cast<int>(i)].deleteFile();
}
