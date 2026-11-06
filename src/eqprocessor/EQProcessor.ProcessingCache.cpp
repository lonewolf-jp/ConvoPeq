//============================================================================
// EQProcessor.ProcessingCache.cpp
//============================================================================
#include "EQProcessor.h"
#include <cstring>
#include <new>

uint64_t EQProcessor::computeParamsHash(const convo::EQParameters& params,
                                        double sampleRate,
                                        int maxBlockSize) noexcept
{
    // 符号を保持し、±0 と NaN を正規化したキーから算出する。
    // 係数生成と同じクランプ規約を通すため、キーと係数の意味が一致する。
    return hashCacheKey(makeCacheKey(params, sampleRate, maxBlockSize));
}

EQCoeffCache* EQProcessor::createCoeffCache(
    const convo::EQParameters& eqParams,
    double sampleRate,
    int maxBlockSize,
    uint64_t generation) noexcept
{
    auto* cache = new (std::nothrow) EQCoeffCache();
    if (cache == nullptr) return nullptr;

    cache->key = makeCacheKey(eqParams, sampleRate, maxBlockSize);
    cache->paramsHash = hashCacheKey(cache->key);
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

    return cache;
}
