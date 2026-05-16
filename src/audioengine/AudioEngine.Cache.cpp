#include <JuceHeader.h>
#include "AudioEngine.h"

namespace {
static void retireEQCache(AudioEngine& owner, EQCoeffCache* cache)
{
    if (cache == nullptr)
        return;

    owner.enqueueDeferredDeleteNonRt(cache, [](void* p) { delete static_cast<EQCoeffCache*>(p); });
}
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_CACHE_MANAGER)

bool AudioEngine::EQCacheManager::tryEnqueueDeferredMap(CacheMap* map) noexcept
{
    if (map == nullptr)
        return true;

    owner.enqueueDeferredDeleteNonRt(map, [](void* p) { delete static_cast<CacheMap*>(p); });
    return true;
}

void AudioEngine::EQCacheManager::drainDeferredMapsUnderLock() noexcept
{
    if (enqueueFallbackMaps.empty())
        return;

    auto out = enqueueFallbackMaps.begin();
    for (auto it = enqueueFallbackMaps.begin(); it != enqueueFallbackMaps.end(); ++it)
    {
        if (!tryEnqueueDeferredMap(*it))
            *out++ = *it;
    }

    enqueueFallbackMaps.erase(out, enqueueFallbackMaps.end());
}

void AudioEngine::EQCacheManager::storeNewMap(CacheMap* newMap) noexcept
{
    auto* old = convo::exchangeAtomic(cacheMapPtr, newMap, std::memory_order_acq_rel);
    if (old == nullptr)
        return;

    owner.enqueueDeferredDeleteNonRt(old, [](void* p) { delete static_cast<CacheMap*>(p); });
}

EQCoeffCache* AudioEngine::EQCacheManager::getOrCreate(const convo::EQParameters& params,
                                                       double sampleRate,
                                                       int maxBlockSize,
                                                       uint64_t generation)
{
    const uint64_t hash = EQProcessor::computeParamsHash(params);
    const CacheMap* currentMap = loadMap();
    if (currentMap == nullptr)
        return nullptr;

    auto it = currentMap->map.find(hash);
    if (it != currentMap->map.end())
        return it->second;

    EQCoeffCache* cache = EQProcessor::createCoeffCache(params, sampleRate, maxBlockSize, generation);
    if (cache == nullptr)
        return nullptr;

    std::lock_guard<std::mutex> lock(writeMutex);

    drainDeferredMapsUnderLock();

    // Lock取得中に他スレッドが同じハッシュを追加した可能性を再確認
    currentMap = loadMap();
    if (currentMap == nullptr)
    {
        retireEQCache(owner, cache);
        return nullptr;
    }

    it = currentMap->map.find(hash);
    if (it != currentMap->map.end())
    {
        // 先に追加されたキャッシュを採用し、新規作成分を破棄
        retireEQCache(owner, cache);
        return it->second;
    }

    CacheMap* newMap = nullptr;
    try
    {
        newMap = new CacheMap(*currentMap);
    }
    catch (const std::bad_alloc&)
    {
        retireEQCache(owner, cache);
        return nullptr;
    }

    newMap->map[hash] = cache;
    storeNewMap(newMap);

    return cache;
}

EQCoeffCache* AudioEngine::EQCacheManager::get(uint64_t hash) noexcept
{
    const CacheMap* currentMap = loadMap();
    if (currentMap == nullptr)
        return nullptr;

    const auto it = currentMap->map.find(hash);
    return (it != currentMap->map.end()) ? it->second : nullptr;
}

void AudioEngine::EQCacheManager::releaseCache(EQCoeffCache* cache) noexcept
{
    if (cache != nullptr)
    retireEQCache(owner, cache);
}

AudioEngine::EQCacheManager::~EQCacheManager()
{
    std::lock_guard<std::mutex> lock(writeMutex);

    CacheMap* currentMap = convo::exchangeAtomic(cacheMapPtr, nullptr, std::memory_order_acq_rel);
    std::unique_ptr<CacheMap> owned{currentMap}; // RAII delete (handles null safely)

    for (auto* map : enqueueFallbackMaps)
        std::unique_ptr<CacheMap> ownedMap{map}; // RAII delete

    enqueueFallbackMaps.clear();
}

#endif // CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_CACHE_MANAGER
