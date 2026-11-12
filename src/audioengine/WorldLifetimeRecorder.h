#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "AtomicAccess.h"   // ★ atomic 直接 dot-call を廃止（convo::*Atomic 経由）

namespace convo {
namespace isr {

/**
 * RuntimeWorld lifetime recorder（D1 とは別系統の計測）
 *
 * 起点: commit（owner channel への enqueue 完了）
 * 終点: 旧所有者の物理破棄（deferred delete の deleter 実行）
 *
 * D1（DSP ハンドル／ライフタイム）とは key も判定も共有しない。
 * 固定容量・アトミックのみ（RT 経路では使わない）。
 */
class WorldLifetimeRecorder
{
public:
    enum class Path : std::uint8_t
    {
        Publish = 0,     // 通常公開
        Deferred,        // deferred 再公開
        Bootstrap,       // 初期化時の置換
        Reconfigure,     // 再構成
        Shutdown,        // 終了処理
        Rejected,        // 公開拒否
        Count
    };

    static constexpr std::size_t kPathCount = static_cast<std::size_t> (Path::Count);
    static constexpr std::size_t kSlots = 256;
    static constexpr std::size_t kObservationCapacity = 128;

    static const char* pathName (Path p) noexcept
    {
        switch (p)
        {
            case Path::Publish:     return "publish";
            case Path::Deferred:    return "deferred";
            case Path::Bootstrap:   return "bootstrap";
            case Path::Reconfigure: return "reconfigure";
            case Path::Shutdown:    return "shutdown";
            case Path::Rejected:    return "rejected";
            default:                return "unknown";
        }
    }

    static WorldLifetimeRecorder& instance() noexcept
    {
        static WorldLifetimeRecorder recorder;
        return recorder;
    }

    struct PathStats
    {
        std::uint64_t commits = 0;
        std::uint64_t destroys = 0;
        std::uint64_t maxNs = 0;
        std::uint64_t sumNs = 0;
    };

    struct Observation
    {
        std::uint64_t worldId = 0;
        std::uint64_t lifetimeNs = 0;
        std::uint8_t path = 0xFFu;
    };

    // commit（owner channel への移譲完了）を記録する
    void recordCommit (const void* world, std::uint64_t worldId, Path path) noexcept
    {
        if (world == nullptr)
            return;

        for (std::size_t i = 0; i < kSlots; ++i)
        {
            if (convo::consumeAtomic (slots_[i].state, std::memory_order_acquire) != kStateFree)
                continue;

            std::uint8_t expected = kStateFree;

            if (! convo::compareExchangeAtomic (slots_[i].state, expected, kStateFilling,
                                                std::memory_order_acq_rel,
                                                std::memory_order_relaxed))
            {
                continue;
            }

            slots_[i].world.store (world, std::memory_order_relaxed);
            slots_[i].worldId.store (worldId, std::memory_order_relaxed);
            slots_[i].commitTs.store (nowNs(), std::memory_order_relaxed);
            slots_[i].path.store (static_cast<std::uint8_t> (path), std::memory_order_relaxed);
            convo::publishAtomic (slots_[i].state, kStateReady, std::memory_order_release);

            convo::fetchAddAtomic (pathCommits_[static_cast<std::size_t> (path)], 1u, std::memory_order_acq_rel);
            return;
        }

        convo::fetchAddAtomic (overflow_, 1u, std::memory_order_acq_rel);
    }

    // 物理破棄を記録する
    void recordDestroy (const void* world) noexcept
    {
        if (world == nullptr)
            return;

        for (std::size_t i = 0; i < kSlots; ++i)
        {
            if (convo::consumeAtomic (slots_[i].state, std::memory_order_acquire) != kStateReady)
                continue;

            if (convo::consumeAtomic (slots_[i].world, std::memory_order_acquire) != world)
                continue;

            std::uint8_t expected = kStateReady;

            if (! convo::compareExchangeAtomic (slots_[i].state, expected, kStateFilling,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire))
            {
                continue;
            }

            const std::uint64_t commitTs = slots_[i].commitTs.load (std::memory_order_relaxed);
            const std::uint64_t worldId = slots_[i].worldId.load (std::memory_order_relaxed);
            const auto path = static_cast<Path> (slots_[i].path.load (std::memory_order_relaxed));
            convo::publishAtomic (slots_[i].state, kStateFree, std::memory_order_release);

            const std::uint64_t now = nowNs();
            const std::uint64_t lifetime = (now >= commitTs) ? (now - commitTs) : 0u;
            const std::size_t p = static_cast<std::size_t> (path);

            if (p < kPathCount)
            {
                convo::fetchAddAtomic (pathDestroys_[p], 1u, std::memory_order_acq_rel);
                convo::fetchAddAtomic (pathSumNs_[p], lifetime, std::memory_order_acq_rel);
                updateMax (pathMaxNs_[p], lifetime);
            }

            const std::uint64_t index = convo::fetchAddAtomic (observationCount_, 1u, std::memory_order_acq_rel);
            observations_[index % kObservationCapacity] = Observation{worldId, lifetime,
                                                                      static_cast<std::uint8_t> (path)};
            return;
        }

        convo::fetchAddAtomic (unmatched_, 1u, std::memory_order_acq_rel);
    }

    struct Snapshot
    {
        std::uint64_t commits = 0;
        std::uint64_t destroys = 0;
        std::uint64_t unmatched = 0;
        std::uint64_t overflow = 0;
        PathStats paths[kPathCount];
        std::vector<Observation> observations;
    };

    Snapshot snapshot() const
    {
        Snapshot s;

        for (std::size_t p = 0; p < kPathCount; ++p)
        {
            s.paths[p].commits = convo::consumeAtomic (pathCommits_[p], std::memory_order_acquire);
            s.paths[p].destroys = convo::consumeAtomic (pathDestroys_[p], std::memory_order_acquire);
            s.paths[p].maxNs = convo::consumeAtomic (pathMaxNs_[p], std::memory_order_acquire);
            s.paths[p].sumNs = convo::consumeAtomic (pathSumNs_[p], std::memory_order_acquire);
            s.commits += s.paths[p].commits;
            s.destroys += s.paths[p].destroys;
        }

        s.unmatched = convo::consumeAtomic (unmatched_, std::memory_order_acquire);
        s.overflow = convo::consumeAtomic (overflow_, std::memory_order_acquire);

        const std::uint64_t written = convo::consumeAtomic (observationCount_, std::memory_order_acquire);
        const std::uint64_t limit = written < kObservationCapacity ? written : kObservationCapacity;

        for (std::uint64_t i = 0; i < limit; ++i)
        {
            const auto& o = observations_[i];

            if (o.path == 0xFFu)
                continue;

            s.observations.push_back (o);
        }

        return s;
    }

    void reset() noexcept
    {
        for (std::size_t i = 0; i < kSlots; ++i)
        {
            convo::publishAtomic (slots_[i].state, kStateFree, std::memory_order_release);
            slots_[i].world.store (nullptr, std::memory_order_relaxed);
            slots_[i].worldId.store (0, std::memory_order_relaxed);
            slots_[i].commitTs.store (0, std::memory_order_relaxed);
            slots_[i].path.store (0xFFu, std::memory_order_relaxed);
        }

        for (std::size_t i = 0; i < kObservationCapacity; ++i)
            observations_[i] = Observation{0, 0, 0xFFu};

        for (std::size_t p = 0; p < kPathCount; ++p)
        {
            convo::publishAtomic (pathCommits_[p], 0, std::memory_order_release);
            convo::publishAtomic (pathDestroys_[p], 0, std::memory_order_release);
            convo::publishAtomic (pathMaxNs_[p], 0, std::memory_order_release);
            convo::publishAtomic (pathSumNs_[p], 0, std::memory_order_release);
        }

        convo::publishAtomic (observationCount_, 0, std::memory_order_release);
        convo::publishAtomic (unmatched_, 0, std::memory_order_release);
        convo::publishAtomic (overflow_, 0, std::memory_order_release);
    }

private:
    static constexpr std::uint8_t kStateFree = 0;
    static constexpr std::uint8_t kStateFilling = 1;
    static constexpr std::uint8_t kStateReady = 2;

    struct Slot
    {
        std::atomic<std::uint8_t> state{kStateFree};
        std::atomic<const void*> world{nullptr};
        std::atomic<std::uint64_t> worldId{0};
        std::atomic<std::uint64_t> commitTs{0};
        std::atomic<std::uint8_t> path{0xFFu};
    };

    WorldLifetimeRecorder() { reset(); }

    static std::uint64_t nowNs() noexcept
    {
        using namespace std::chrono;
        return static_cast<std::uint64_t> (
            duration_cast<nanoseconds> (steady_clock::now().time_since_epoch()).count());
    }

    static void updateMax (std::atomic<std::uint64_t>& target, std::uint64_t value) noexcept
    {
        std::uint64_t current = convo::consumeAtomic (target, std::memory_order_acquire);

        while (value > current
               && ! convo::compareExchangeWeakAtomic (target, current, value,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_acquire))
        {
        }
    }

    Slot slots_[kSlots];
    Observation observations_[kObservationCapacity];

    std::atomic<std::uint64_t> pathCommits_[kPathCount];
    std::atomic<std::uint64_t> pathDestroys_[kPathCount];
    std::atomic<std::uint64_t> pathMaxNs_[kPathCount];
    std::atomic<std::uint64_t> pathSumNs_[kPathCount];
    std::atomic<std::uint64_t> observationCount_{0};
    std::atomic<std::uint64_t> unmatched_{0};
    std::atomic<std::uint64_t> overflow_{0};
};

} // namespace isr
} // namespace convo
