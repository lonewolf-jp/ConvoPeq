#include <JuceHeader.h>
#include "AudioEngine.h"


AudioEngine::EQCacheManager::EQCacheManager(AudioEngine& ownerIn) noexcept
    : owner(ownerIn)
{
    convo::publishAtomic(cacheMapPtr, new CacheMap(ownerIn), std::memory_order_release); // release: loadMap acquire と HB
}

[[nodiscard]] bool AudioEngine::EQCacheManager::tryEnqueueDeferredMap(CacheMap* map) noexcept
{
    if (map == nullptr)
        return true;

    return owner.enqueueDeferredDeleteNonRt(map, [](void* p) { delete static_cast<CacheMap*>(p); });
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
    auto* old = convo::exchangeAtomic(cacheMapPtr, newMap, std::memory_order_acq_rel); // acq_rel: acquire で旧 map 取得; release で新 map 公開
    if (old == nullptr)
        return;

    if (!owner.enqueueDeferredDeleteNonRt(old, [](void* p) { delete static_cast<CacheMap*>(p); })) {
        // QueueFull/Shutdown -> caller retains -> keep in enqueueFallbackMaps for later drain
        // caller (getOrCreate) holds writeMutex, so no lock needed
        try {
            enqueueFallbackMaps.push_back(old);
        } catch (...) {
            delete old;
        }
    }
}

EQCoeffCache* AudioEngine::EQCacheManager::getOrCreate(const convo::EQParameters& params,
                                                       double sampleRate,
                                                       int maxBlockSize,
                                                       uint64_t generation)
{
    using convo::isr::DSPHandle;

    // ★ BUG-047: computeParamsHash に sampleRate/maxBlockSize を含めて、
    //            sampleRate 変更時に古い係数キャッシュがヒットしないようにする。
    const uint64_t hash = EQProcessor::computeParamsHash(params, sampleRate, maxBlockSize);
    const CacheMap* currentMap = loadMap();
    if (currentMap == nullptr)
        return nullptr;

    auto it = currentMap->map.find(hash);
    if (it != currentMap->map.end())
    {
        // ★ P0-2: DSPHandle → resolve() でポインタ取得
        const auto resolved = owner.dspHandleRuntime_.resolve(it->second);
        return static_cast<EQCoeffCache*>(resolved.instance);
    }

    // ★ P0-2: キャッシュミス — 新規作成
    EQCoeffCache* cache = EQProcessor::createCoeffCache(params, sampleRate, maxBlockSize, generation);
    if (cache == nullptr)
        return nullptr;

    // ★ P0-2: DSPHandleRuntime に登録
    const DSPHandle handle = owner.dspHandleRuntime_.create(cache);
    if (handle.isNull())
    {
        delete cache;
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(writeMutex);

    drainDeferredMapsUnderLock();

    // Lock取得中に他スレッドが同じハッシュを追加した可能性を再確認
    currentMap = loadMap();
    if (currentMap == nullptr)
    {
        delete cache;
        owner.dspHandleRuntime_.retire(handle);
        return nullptr;
    }

    it = currentMap->map.find(hash);
    if (it != currentMap->map.end())
    {
        // 先に追加されたキャッシュを採用し、新規作成分を破棄
        delete cache;
        owner.dspHandleRuntime_.retire(handle);
        const auto resolved = owner.dspHandleRuntime_.resolve(it->second);
        return static_cast<EQCoeffCache*>(resolved.instance);
    }

    std::unique_ptr<CacheMap> newMap;
    try
    {
        newMap = std::make_unique<CacheMap>(*currentMap);
        newMap->map.emplace(hash, handle);
    }
    catch (const std::bad_alloc&)
    {
        delete cache;
        owner.dspHandleRuntime_.retire(handle);
        return nullptr;
    }
    catch (...)
    {
        delete cache;
        owner.dspHandleRuntime_.retire(handle);
        return nullptr;
    }

    storeNewMap(newMap.release());

    const auto resolved = owner.dspHandleRuntime_.resolve(handle);
    return static_cast<EQCoeffCache*>(resolved.instance);
}

EQCoeffCache* AudioEngine::EQCacheManager::get(uint64_t hash) noexcept
{
    const CacheMap* currentMap = loadMap();
    if (currentMap == nullptr)
        return nullptr;

    const auto it = currentMap->map.find(hash);
    if (it == currentMap->map.end())
        return nullptr;

    // ★ P0-2: DSPHandle → resolve() でポインタ取得
    const auto resolved = owner.dspHandleRuntime_.resolve(it->second);
    return static_cast<EQCoeffCache*>(resolved.instance);
}

[[nodiscard]] bool AudioEngine::EQCacheManager::containsNonRt(uint64_t hash) noexcept
{
    std::lock_guard<std::mutex> lock(writeMutex);
    drainDeferredMapsUnderLock();

    const CacheMap* currentMap = loadMap();
    if (currentMap == nullptr)
        return false;

    return currentMap->map.find(hash) != currentMap->map.end();
}

// ★ D162-2-E (E-1 / INV-D162-6): member teardown 前の全 CacheMap エントリ処分。
//   ~AudioEngine body 内（全 engine member 生存中）から呼ぶこと。shutdownRuntime_ /
//   m_retireRouter / dspHandleRuntime_ 等が全て生存している状態で:
//     1. cacheMapPtr を nullptr に exchange（新規アクセス遮断）
//     2. 全 CacheMap エントリの DSPHandle を dspHandleRuntime_.retire（Retired 遷移）し、
//        resolve で取得した EQCoeffCache* を物理解放
//     3. enqueueFallbackMaps の各 CacheMap も同様に処分
//   呼び出し後は ~CacheMap が engine member 非依存の no-op となる（UAF 排除）。
void AudioEngine::EQCacheManager::drainForShutdown() noexcept
{
    std::lock_guard<std::mutex> lock(writeMutex);

    CacheMap* currentMap = convo::exchangeAtomic(cacheMapPtr, nullptr, std::memory_order_acq_rel);
    if (currentMap != nullptr)
    {
        for (auto& entry : currentMap->map)
        {
            if (entry.second.isNull())
                continue;
            // Retired 遷移（grace period concept は shutdown なので即 reclaim 相当）
            owner.dspHandleRuntime_.retire(entry.second);
            const auto resolved = owner.dspHandleRuntime_.resolve(entry.second);
            if (resolved.instance != nullptr)
                delete static_cast<EQCoeffCache*>(resolved.instance);
        }
        currentMap->map.clear();
        delete currentMap;
    }

    for (auto* map : enqueueFallbackMaps)
    {
        if (map == nullptr)
            continue;
        for (auto& entry : map->map)
        {
            if (entry.second.isNull())
                continue;
            owner.dspHandleRuntime_.retire(entry.second);
            const auto resolved = owner.dspHandleRuntime_.resolve(entry.second);
            if (resolved.instance != nullptr)
                delete static_cast<EQCoeffCache*>(resolved.instance);
        }
        map->map.clear();
        delete map;
    }
    enqueueFallbackMaps.clear();
}

AudioEngine::EQCacheManager::~EQCacheManager()
{
    std::lock_guard<std::mutex> lock(writeMutex);

    CacheMap* currentMap = convo::exchangeAtomic(cacheMapPtr, nullptr, std::memory_order_acq_rel); // acq_rel: acquire で旧 map 取得; release で null 公開
    std::unique_ptr<CacheMap> owned{currentMap}; // RAII delete (handles null safely)

    for (auto* map : enqueueFallbackMaps)
        std::unique_ptr<CacheMap> ownedMap{map}; // RAII delete

    enqueueFallbackMaps.clear();
}
