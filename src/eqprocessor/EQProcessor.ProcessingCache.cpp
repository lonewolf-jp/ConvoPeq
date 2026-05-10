//============================================================================
// EQProcessor.ProcessingCache.cpp
//============================================================================
#include "EQProcessor.h"
#include <cstring>

static inline uint32_t floatToCanonicalBits(float f) noexcept
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    return bits & 0x7FFFFFFF;
}

static inline uint64_t hashCombine(uint64_t seed, uint64_t value) noexcept
{
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

uint64_t EQProcessor::computeParamsHash(const convo::EQParameters& params) noexcept
{
    uint64_t hash = 0;

    for (int i = 0; i < 20; ++i)
    {
        const auto& band = params.bands[i];
        hash = hashCombine(hash, floatToCanonicalBits(band.frequency));
        hash = hashCombine(hash, floatToCanonicalBits(band.gain));
        hash = hashCombine(hash, floatToCanonicalBits(band.q));
        hash = hashCombine(hash, static_cast<uint64_t>(band.enabled ? 1 : 0));
        hash = hashCombine(hash, static_cast<uint64_t>(band.type));
        hash = hashCombine(hash, static_cast<uint64_t>(band.channelMode));
    }

    hash = hashCombine(hash, floatToCanonicalBits(params.totalGainDb));
    hash = hashCombine(hash, static_cast<uint64_t>(params.agcEnabled ? 1 : 0));
    hash = hashCombine(hash, floatToCanonicalBits(params.nonlinearSaturation));
    hash = hashCombine(hash, static_cast<uint64_t>(params.filterStructure));

    return hash;
}

EQCoeffCache* EQProcessor::createCoeffCache(
    const convo::EQParameters& eqParams,
    double sampleRate,
    int maxBlockSize,
    uint64_t generation) noexcept
{
    auto* cache = new EQCoeffCache();
    if (!cache) return nullptr;

    cache->paramsHash = computeParamsHash(eqParams);
    cache->sampleRate = sampleRate;
    cache->maxBlockSize = maxBlockSize;
    cache->generation = generation;
    cache->filterStructure = eqParams.filterStructure;

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        const auto& band = eqParams.bands[i];
        cache->bandActive[i] = band.enabled && sampleRate > 0.0;
        cache->channelModes[i] = band.channelMode;

        if (band.enabled && sampleRate > 0.0)
        {
            cache->coeffs[i] = calcSVFCoeffs(
                static_cast<EQBandType>(band.type),
                band.frequency,
                band.gain,
                band.q,
                sampleRate);
        }
        else
        {
            cache->coeffs[i] = EQCoeffsSVF();
        }
    }

    if (cache->filterStructure == 1)
    {
        const int requiredSize = maxBlockSize * MAX_CHANNELS;

        cache->parallelInputBuffer = static_cast<double*>(
            convo::aligned_malloc(requiredSize * sizeof(double), 64));
        cache->parallelWorkBuffer = static_cast<double*>(
            convo::aligned_malloc(requiredSize * sizeof(double), 64));
        cache->parallelAccumBuffer = static_cast<double*>(
            convo::aligned_malloc(requiredSize * sizeof(double), 64));

        if (!cache->parallelInputBuffer || !cache->parallelWorkBuffer || !cache->parallelAccumBuffer)
        {
            if (cache->parallelInputBuffer)
                convo::aligned_free(cache->parallelInputBuffer);
            if (cache->parallelWorkBuffer)
                convo::aligned_free(cache->parallelWorkBuffer);
            if (cache->parallelAccumBuffer)
                convo::aligned_free(cache->parallelAccumBuffer);
            delete cache;
            return nullptr;
        }

        cache->parallelBufferSize = requiredSize;
        std::memset(cache->parallelInputBuffer, 0, requiredSize * sizeof(double));
        std::memset(cache->parallelWorkBuffer, 0, requiredSize * sizeof(double));
        std::memset(cache->parallelAccumBuffer, 0, requiredSize * sizeof(double));
    }

    return cache;
}

EQCoeffCache::~EQCoeffCache()
{
    if (parallelInputBuffer)
    {
        convo::aligned_free(parallelInputBuffer);
        parallelInputBuffer = nullptr;
    }
    if (parallelWorkBuffer)
    {
        convo::aligned_free(parallelWorkBuffer);
        parallelWorkBuffer = nullptr;
    }
    if (parallelAccumBuffer)
    {
        convo::aligned_free(parallelAccumBuffer);
        parallelAccumBuffer = nullptr;
    }
    parallelBufferSize = 0;
}
