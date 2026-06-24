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

    static void touch(uint64_t fileHash,
                      double sampleRate,
                      int phaseMode,
                      float freqStartHz, float freqEndHz,
                      int targetLength);

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
