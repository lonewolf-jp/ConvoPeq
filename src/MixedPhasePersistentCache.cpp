#include "MixedPhasePersistentCache.h"

#include <JuceHeader.h>
#include <algorithm>
#include <cstring>
#include <vector>

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
                                                    float freqStartHz, float freqEndHz, float tau,
                                                    int targetLength)
{
    uint64_t h = fileHash;
    uint64_t srBits = 0;
    std::memcpy(&srBits, &sampleRate, sizeof(double));
    h = hashCombine(h, srBits);
    h = hashCombine(h, static_cast<uint64_t>(phaseMode));

    uint32_t f1Bits = 0;
    uint32_t f2Bits = 0;
    uint32_t tauBits = 0;
    std::memcpy(&f1Bits, &freqStartHz, sizeof(float));
    std::memcpy(&f2Bits, &freqEndHz, sizeof(float));
    std::memcpy(&tauBits, &tau, sizeof(float));
    h = hashCombine(h, static_cast<uint64_t>(f1Bits));
    h = hashCombine(h, static_cast<uint64_t>(f2Bits));
    h = hashCombine(h, static_cast<uint64_t>(tauBits));
    h = hashCombine(h, static_cast<uint64_t>(targetLength));
    return h;
}

// ═══════════════════════════════════════════════════════════════
//  パス解決
// ═══════════════════════════════════════════════════════════════

juce::File MixedPhasePersistentCache::getCacheDirectory()
{
    auto dir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                   .getChildFile("ConvoPeq")
                   .getChildFile("MixedPhaseCache");
    if (!dir.exists())
        dir.createDirectory();
    return dir;
}

juce::File MixedPhasePersistentCache::getCacheFile(uint64_t fileHash,
                                                    double sampleRate,
                                                    int phaseMode,
                                                    float freqStartHz, float freqEndHz, float tau,
                                                    int targetLength)
{
    const auto hash = computeKeyHash(fileHash, sampleRate, phaseMode,
                                     freqStartHz, freqEndHz, tau, targetLength);
    const auto filename = juce::String::toHexString(static_cast<int64_t>(hash)) + ".mph";
    return getCacheDirectory().getChildFile(filename);
}

// ═══════════════════════════════════════════════════════════════
//  読み込み
// ═══════════════════════════════════════════════════════════════

bool MixedPhasePersistentCache::load(uint64_t fileHash,
                                     double sampleRate,
                                     int phaseMode,
                                     float freqStartHz, float freqEndHz, float tau,
                                     int targetLength,
                                     juce::AudioBuffer<double>& outIr,
                                     std::vector<double>& outRho,
                                     std::vector<double>& outTheta)
{
    const auto file = getCacheFile(fileHash, sampleRate, phaseMode,
                                   freqStartHz, freqEndHz, tau, targetLength);
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
        || std::abs(header.tau - tau) > 1.0e-6f
        || header.targetLength != static_cast<int32_t>(targetLength))
        return false;

    // 波形データ読み込み
    const int numChannels = static_cast<int>(header.numChannels);
    const int numSamples = static_cast<int>(header.numSamples);
    if (numChannels <= 0 || numSamples <= 0)
        return false;

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
    const int numSec = static_cast<int>(header.numAllpassSections);
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

    // LRU用にタイムスタンプ更新（touch相当）
    touch(fileHash, sampleRate, phaseMode, freqStartHz, freqEndHz, tau, targetLength);

    return true;
}

// ═══════════════════════════════════════════════════════════════
//  保存
// ═══════════════════════════════════════════════════════════════

bool MixedPhasePersistentCache::save(uint64_t fileHash,
                                     double sampleRate,
                                     int phaseMode,
                                     float freqStartHz, float freqEndHz, float tau,
                                     int targetLength,
                                     const juce::AudioBuffer<double>& ir,
                                     const std::vector<double>& rho,
                                     const std::vector<double>& theta)
{
    const auto file = getCacheFile(fileHash, sampleRate, phaseMode,
                                   freqStartHz, freqEndHz, tau, targetLength);
    const auto dir = file.getParentDirectory();
    if (!dir.exists())
        dir.createDirectory();

    juce::TemporaryFile tempFile(file);
    {
        juce::FileOutputStream stream(tempFile.getFile());
        if (!stream.openedOk())
            return false;

        const int numChannels = ir.getNumChannels();
        const int numSamples = ir.getNumSamples();
        const int numSec = static_cast<int>(rho.size());

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
        header.tau = tau;
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

void MixedPhasePersistentCache::touch(uint64_t fileHash,
                                      double sampleRate,
                                      int phaseMode,
                                      float freqStartHz, float freqEndHz, float tau,
                                      int targetLength)
{
    const auto file = getCacheFile(fileHash, sampleRate, phaseMode,
                                   freqStartHz, freqEndHz, tau, targetLength);
    if (!file.existsAsFile())
        return;

    // read-modify-write: ファイル全体を読み込み、ヘッダのタイムスタンプのみ更新して書き戻す
    // (FileOutputStream で開くとファイルが切り詰められるため、read-modify-write で行う)
    std::vector<uint8_t> buffer;
    {
        juce::FileInputStream inStream(file);
        if (!inStream.openedOk())
            return;

        const auto fileSize = file.getSize();
        if (fileSize <= 0)
            return;

        buffer.resize(static_cast<size_t>(fileSize));
        if (inStream.read(buffer.data(), static_cast<int>(buffer.size())) != static_cast<int64>(buffer.size()))
            return;
    } // inStream はここで破棄され、ファイルも閉じられる

    // DiskHeader 内 lastUsedTime のバイトオフセット: 52
    static constexpr int kLastUsedTimeOffset = 52;
    if (static_cast<size_t>(kLastUsedTimeOffset) + sizeof(uint64_t) > buffer.size())
        return;

    const uint64_t now = static_cast<uint64_t>(juce::Time::getMillisecondCounter());
    std::memcpy(&buffer[static_cast<size_t>(kLastUsedTimeOffset)], &now, sizeof(now));

    juce::FileOutputStream outStream(file);
    if (!outStream.openedOk())
        return;
    outStream.write(buffer.data(), buffer.size());
}

void MixedPhasePersistentCache::evictLRU(size_t maxCount)
{
    if (maxCount == 0)
    {
        clear();
        return;
    }

    auto dir = getCacheDirectory();
    juce::Array<juce::File> files;
    dir.findChildFiles(files, juce::File::findFiles, false, "*.mph");

    if (static_cast<size_t>(files.size()) <= maxCount)
        return;

    // 各ファイルのヘッダから lastUsedTime を読み、古い順にソート
    // (OSのファイルアクセス時刻は無効化されている可能性があるためヘッダの値を使用する)
    std::vector<std::pair<uint64_t, juce::File>> timedFiles;
    timedFiles.reserve(static_cast<size_t>(files.size()));

    for (const auto& f : files)
    {
        juce::FileInputStream stream(f);
        if (!stream.openedOk())
            continue;

        DiskHeader header;
        if (stream.read(&header, sizeof(DiskHeader)) != static_cast<int64>(sizeof(DiskHeader)))
            continue;
        if (header.magic != kMagic)
            continue;

        timedFiles.emplace_back(header.lastUsedTime, f);
    }

    std::sort(timedFiles.begin(), timedFiles.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    const int toRemove = static_cast<int>(timedFiles.size()) - static_cast<int>(maxCount);
    for (int i = 0; i < toRemove; ++i)
        timedFiles[static_cast<size_t>(i)].second.deleteFile();
}

void MixedPhasePersistentCache::remove(uint64_t fileHash,
                                       double sampleRate,
                                       int phaseMode,
                                       float freqStartHz, float freqEndHz, float tau,
                                       int targetLength)
{
    const auto file = getCacheFile(fileHash, sampleRate, phaseMode,
                                   freqStartHz, freqEndHz, tau, targetLength);
    if (file.existsAsFile())
        file.deleteFile();
}

void MixedPhasePersistentCache::clear()
{
    auto dir = getCacheDirectory();
    if (!dir.exists())
        return;

    juce::Array<juce::File> files;
    dir.findChildFiles(files, juce::File::findFiles, false, "*.mph");
    for (auto& f : files)
        f.deleteFile();
}

size_t MixedPhasePersistentCache::getEntryCount()
{
    auto dir = getCacheDirectory();
    if (!dir.exists())
        return 0;

    juce::Array<juce::File> files;
    dir.findChildFiles(files, juce::File::findFiles, false, "*.mph");
    return static_cast<size_t>(files.size());
}

} // namespace convo
