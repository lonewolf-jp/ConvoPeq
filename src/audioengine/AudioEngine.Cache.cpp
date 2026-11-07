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

    // 係数生成と同じクランプ規約を通したキーで同一性を判定する。
    const EQCoeffKey key = EQProcessor::makeCacheKey(params, sampleRate, maxBlockSize);
    uint64_t hash = EQProcessor::hashCacheKey(key);

    // ロックなしのヒット探索。ハッシュが一致してもキーが違う場合は別ハッシュへ退避する。
    for (int probe = 0; probe < kMaxCollisionProbes; ++probe)
    {
        const CacheMap* currentMap = loadMap();
        if (currentMap == nullptr)
            return nullptr;

        const auto it = currentMap->map.find(hash);
        if (it == currentMap->map.end())
            break;

        const auto resolved = owner.dspHandleRuntime_.resolve(it->second);
        auto* existing = static_cast<EQCoeffCache*>(resolved.instance);
        if (existing != nullptr && EQProcessor::cacheKeyMatches(*existing, key))
            return existing;

        hash = EQProcessor::probeCacheHash(hash, probe);
    }

    // ★ P0-2: キャッシュミス — 新規作成
    EQCoeffCache* cache = EQProcessor::createCoeffCache(params, sampleRate, maxBlockSize, generation);
    if (cache == nullptr)
        return nullptr;

    cache->key = key;
    cache->paramsHash = hash;

    // ★ P0-2: DSPHandleRuntime に登録
    const DSPHandle handle = owner.dspHandleRuntime_.create(cache);
    if (handle.isNull())
    {
        delete cache;
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(writeMutex);

    drainDeferredMapsUnderLock();

    for (int probe = 0; probe < kMaxCollisionProbes; ++probe)
    {
        const CacheMap* currentMap = loadMap();
        if (currentMap == nullptr)
        {
            // ★ F2' (2026-10-25): 削除の前にスロットを解決不能にする。
            owner.invalidateHandleBeforeDestroy(cache);
            delete cache;
            return nullptr;
        }

        const auto it = currentMap->map.find(hash);
        if (it != currentMap->map.end())
        {
            const auto resolved = owner.dspHandleRuntime_.resolve(it->second);
            auto* existing = static_cast<EQCoeffCache*>(resolved.instance);
            if (existing != nullptr && EQProcessor::cacheKeyMatches(*existing, key))
            {
                // 先に追加された同一キャッシュを採用し、新規作成分を破棄
                owner.invalidateHandleBeforeDestroy(cache);
                delete cache;
                return existing;
            }

            hash = EQProcessor::probeCacheHash(hash, probe);
            cache->paramsHash = hash;
            continue;
        }

        std::unique_ptr<CacheMap> newMap;
        try
        {
            newMap = std::make_unique<CacheMap>(*currentMap);
            newMap->map.emplace(hash, handle);
        }
        catch (const std::bad_alloc&)
        {
            // ★ F2' (2026-10-25): 削除の前にスロットを解決不能にする。
            owner.invalidateHandleBeforeDestroy(cache);
            delete cache;
            return nullptr;
        }
        catch (...)
        {
            // ★ F2' (2026-10-25): 削除の前にスロットを解決不能にする。
            owner.invalidateHandleBeforeDestroy(cache);
            delete cache;
            return nullptr;
        }

        storeNewMap(newMap.release());

        const auto resolved = owner.dspHandleRuntime_.resolve(handle);
        return static_cast<EQCoeffCache*>(resolved.instance);
    }

    // 32 個の候補がすべて別キーで埋まる場合は、誤った係数を返さず fail-closed にする。
    owner.invalidateHandleBeforeDestroy(cache);
    delete cache;
    return nullptr;
}

#if defined(CONVOPEQ_ENABLE_TEST_HOOKS) && CONVOPEQ_ENABLE_TEST_HOOKS
bool AudioEngine::EQCacheManager::debugRunCollisionProbe()
{
    convo::EQParameters a;
    a.bands[0].gain = 6.0f;
    convo::EQParameters b = a;
    b.bands[0].gain = -6.0f;

    const auto keyA = EQProcessor::makeCacheKey(a, 48000.0, 512);
    const auto keyB = EQProcessor::makeCacheKey(b, 48000.0, 512);
    if (keyA == keyB)
        return false;

    const uint64_t hashB = EQProcessor::hashCacheKey(keyB);

    EQCoeffCache* cacheA = EQProcessor::createCoeffCache(a, 48000.0, 512, 1u);
    if (cacheA == nullptr)
        return false;
    cacheA->key = keyA;
    cacheA->paramsHash = hashB; // 同一ハッシュへ別キーを置いて衝突を再現する

    const auto handleA = owner.dspHandleRuntime_.create(cacheA);
    if (handleA.isNull())
    {
        delete cacheA;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(writeMutex);
        const CacheMap* currentMap = loadMap();
        if (currentMap == nullptr)
        {
            owner.invalidateHandleBeforeDestroy(cacheA);
            delete cacheA;
            return false;
        }

        std::unique_ptr<CacheMap> newMap;
        try
        {
            newMap = std::make_unique<CacheMap>(*currentMap);
            newMap->map.emplace(hashB, handleA);
        }
        catch (...)
        {
            owner.invalidateHandleBeforeDestroy(cacheA);
            delete cacheA;
            return false;
        }
        storeNewMap(newMap.release());
    }

    EQCoeffCache* cacheB = getOrCreate(b, 48000.0, 512, 2u);
    return cacheB != nullptr
        && cacheB != cacheA
        && EQProcessor::cacheKeyMatches(*cacheB, keyB)
        && cacheB->paramsHash != hashB;
}

bool AudioEngine::EQCacheManager::debugRunDisabledBandKeyProbe()
{
    convo::EQParameters a;
    a.bands[7].enabled = false;
    a.bands[7].frequency = 35.0f;
    a.bands[7].gain = 9.0f;
    a.bands[7].q = 2.5f;

    convo::EQParameters b = a;
    b.bands[7].frequency = 8000.0f;
    b.bands[7].gain = -12.0f;
    b.bands[7].q = 0.5f;

    const auto keyA = EQProcessor::makeCacheKey(a, 48000.0, 512);
    const auto keyB = EQProcessor::makeCacheKey(b, 48000.0, 512);
    if (!(keyA == keyB))
        return false;
    if (EQProcessor::hashCacheKey(keyA) != EQProcessor::hashCacheKey(keyB))
        return false;

    EQCoeffCache* cacheA = getOrCreate(a, 48000.0, 512, 1u);
    if (cacheA == nullptr || !EQProcessor::cacheKeyMatches(*cacheA, keyA))
        return false;

    EQCoeffCache* cacheB = getOrCreate(b, 48000.0, 512, 2u);
    if (cacheB == nullptr || cacheB != cacheA)
        return false;

    convo::EQParameters c = a;
    c.bands[7].enabled = true;
    EQCoeffCache* cacheC = getOrCreate(c, 48000.0, 512, 3u);
    return cacheC != nullptr && cacheC != cacheA;
}

bool AudioEngine::EQCacheManager::debugRunProbeDepthProbe()
{
    // 40 段の衝突連鎖を作り、32 回で打ち切る旧実装では fail-closed になる深さを再現する。
    constexpr int kChainLength = 40;

    convo::EQParameters target;
    target.bands[0].gain = 1.0f;
    const auto targetKey = EQProcessor::makeCacheKey(target, 48000.0, 512);
    const uint64_t targetHash = EQProcessor::hashCacheKey(targetKey);

    uint64_t hash = targetHash;
    for (int i = 0; i < kChainLength; ++i)
    {
        if (i > 0)
            hash = EQProcessor::probeCacheHash(hash, i - 1);

        convo::EQParameters other;
        other.bands[0].gain = 2.0f + static_cast<float>(i);
        EQCoeffCache* cache = EQProcessor::createCoeffCache(
            other, 48000.0, 512, static_cast<uint64_t>(100 + i));
        if (cache == nullptr)
            return false;

        cache->key = EQProcessor::makeCacheKey(other, 48000.0, 512);
        cache->paramsHash = hash;

        const auto handle = owner.dspHandleRuntime_.create(cache);
        if (handle.isNull())
        {
            delete cache;
            return false;
        }

        std::lock_guard<std::mutex> lock(writeMutex);
        const CacheMap* currentMap = loadMap();
        if (currentMap == nullptr)
        {
            owner.invalidateHandleBeforeDestroy(cache);
            delete cache;
            return false;
        }

        std::unique_ptr<CacheMap> newMap;
        try
        {
            newMap = std::make_unique<CacheMap>(*currentMap);
            newMap->map.emplace(hash, handle);
        }
        catch (...)
        {
            owner.invalidateHandleBeforeDestroy(cache);
            delete cache;
            return false;
        }
        storeNewMap(newMap.release());
    }

    EQCoeffCache* resolved = getOrCreate(target, 48000.0, 512, 999u);
    return resolved != nullptr && EQProcessor::cacheKeyMatches(*resolved, targetKey);
}
#endif

// ★ F1' (2026-10-25): RT 経路用。lease を返すので、呼び出し側は
//   「ポインタを使い終わるスコープ全体」で保持すること（規約 R2）。
convo::isr::DSPHandlePin AudioEngine::EQCacheManager::pinByHash(uint64_t hash) noexcept
{
    const CacheMap* currentMap = loadMap();
    if (currentMap == nullptr)
        return {};

    const auto it = currentMap->map.find(hash);
    if (it == currentMap->map.end())
        return {};

    // ★ F1': resolve() ではなく tryPin() を使う（受付閉鎖と pin 数を単一ワードで扱う）。
    return owner.dspHandleRuntime_.tryPin(it->second);
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
//     2. 全 CacheMap エントリを resolve で取得し、★F2' でスロットを無効化してから
//        EQCoeffCache* を物理解放（retire を先にすると v11 候補 1 の resolve が
//        nullptr を返し、キャッシュが解放されない）
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
            // ★ F2' (2026-10-25): 解決 → 無効化 → 削除。
            //   retire を先にすると (v11 候補 1 の resolve は Retired を拒否) instance が
            //   取れず、キャッシュが解放されない。
            const auto resolved = owner.dspHandleRuntime_.resolve(entry.second);
            if (resolved.instance != nullptr)
            {
                owner.invalidateHandleBeforeDestroy(resolved.instance);
                delete static_cast<EQCoeffCache*>(resolved.instance);
            }
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
            // ★ F2' (2026-10-25): 解決 → 無効化 → 削除。
            //   retire を先にすると (v11 候補 1 の resolve は Retired を拒否) instance が
            //   取れず、キャッシュが解放されない。
            const auto resolved = owner.dspHandleRuntime_.resolve(entry.second);
            if (resolved.instance != nullptr)
            {
                owner.invalidateHandleBeforeDestroy(resolved.instance);
                delete static_cast<EQCoeffCache*>(resolved.instance);
            }
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
