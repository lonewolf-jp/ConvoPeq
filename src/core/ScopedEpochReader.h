#pragma once

#include <cstdio>
#include <cstring>
#include <chrono>
#include <atomic>
#include <thread>
#include <functional>

#include <array>
#include <cstdint>

#include "IReaderEpochProvider.h"

namespace convo
{

//==============================================================================
// ScopedEpochReader — RAII reader scope for NON-RT readers of EBR-managed objects.
//
// Why this exists (A-closure, reader safety):
//   A thread that loads a published pointer with acquire and then dereferences it can
//   be preempted between the load and the dereference. If the publisher unlinks the
//   object, retires it at the current epoch and a reclaimer runs while that reader is
//   NOT registered, getMinReaderEpoch() reports "no readers" and the object is freed
//   under the reader -> use-after-free.
//
//   Registering the reader BEFORE the load closes that window: the slot's epoch is
//   <= the epoch at which the object is retired, so `retireEpoch < minReaderEpoch`
//   cannot hold while the reader is inside. After the last dereference the guard
//   exits, and only then may the object be reclaimed.
//
// Contract:
//   * Construct (enter) BEFORE the first load of the published pointer.
//   * Destroy (exit) AFTER the last dereference of it.
//   * Fail-closed: when no slot can be obtained (registration closed after shutdown,
//     or all kMaxReaders slots are taken) active() is false and the caller MUST NOT
//     dereference the published pointer; return a safe default instead.
//
// Slot reuse:
//   A reader slot is registered once per (thread, provider) and then *re-acquired*
//   with IReaderEpochProvider::reacquireReaderSlot(), which is ownership-checked by
//   the provider. That check is what makes the cache safe: if a provider is destroyed
//   and a new one is constructed at the same address, the cached index belongs to the
//   old provider, re-acquire fails, and the slot is registered again (or the guard
//   fails closed when registration is closed). Without the ownership check a stale
//   cache entry would report active()==true with no real epoch protection.
//
// Cost: one re-acquire CAS + one enter/exit pair per scope; no allocation, no lock.
//==============================================================================
class ScopedEpochReader
{
public:
    // ★ 計測 (2026-11-10 Owner 承認): 生成箇所を識別するタグ。
    //   既定は nullptr（従来どおり無タグで登録する）。
    explicit ScopedEpochReader (IReaderEpochProvider& provider,
                                const char* tag = nullptr) noexcept
        : provider_ (&provider),
          slot_ (acquireSlot (provider, tag, &depth_))
    {
        if (slot_ >= 0)
        {
            provider_->enterReader (slot_);
            // ★ 計測 (2026-11-11 Owner 承認): guard インスタンス単位の入場記録
            measureIdx_ = claimMeasure (tag, static_cast<uint32_t> (slot_),
                                        static_cast<uint32_t> (depth_), &provider);
        }
    }

    ~ScopedEpochReader() { reset(); }

    ScopedEpochReader (const ScopedEpochReader&) = delete;
    ScopedEpochReader& operator= (const ScopedEpochReader&) = delete;

    ScopedEpochReader (ScopedEpochReader&& other) noexcept
        : provider_ (other.provider_),
          slot_ (other.slot_),
          depth_ (other.depth_),
          measureIdx_ (other.measureIdx_)
    {
        other.provider_ = nullptr;
        other.slot_ = -1;
        other.measureIdx_ = -1;   // ★ 計測: 二重終了を防ぐ
    }

    ScopedEpochReader& operator= (ScopedEpochReader&&) = delete;

    /** True when a reader slot was obtained and enterReader() ran. */
    [[nodiscard]] bool active() const noexcept { return slot_ >= 0; }

    /** Reader slot index (diagnostics); -1 when inactive. */
    [[nodiscard]] int slotIndex() const noexcept { return slot_; }

private:
    void reset() noexcept
    {
        // ★ 計測: guard の退出を記録する（idx は 1 回だけ消費される）
        if (measureIdx_ >= 0)
        {
            closeMeasure (measureIdx_);
            measureIdx_ = -1;
        }

        if (provider_ != nullptr && slot_ >= 0)
        {
            provider_->exitReader (slot_);
            releaseSlot (*provider_, slot_);
        }

        provider_ = nullptr;
        slot_ = -1;
    }

    struct CacheEntry
    {
        const void* provider = nullptr;
        int slot = -1;
        int depth = 0;   // > 0 while this thread is inside the scope
    };

    // Four entries cover the realistic case (one engine, a few domains). Overflow is
    // not fatal: the thread registers another slot, bounded by kMaxReaders.
    static std::array<CacheEntry, 4>& cache() noexcept
    {
        // NOLINT(thread-local) RT-SAFE: used only by the non-RT reader path (never from the audio
        //   thread); POD, no destructor, one cache per thread, no cross-thread sharing.
        thread_local std::array<CacheEntry, 4> entries {};   // NOLINT(thread-local) RT-SAFE: non-RT only, POD, no destructor, per-thread
        return entries;
    }

    static CacheEntry* find (const IReaderEpochProvider& provider) noexcept
    {
        for (auto& entry : cache())
            if (entry.provider == &provider)
                return &entry;

        return nullptr;
    }

    static CacheEntry* findFree() noexcept
    {
        for (auto& entry : cache())
            if (entry.provider == nullptr)
                return &entry;

        return nullptr;
    }

    static int acquireSlot (IReaderEpochProvider& provider, const char* tag = nullptr,
                            int* outDepth = nullptr) noexcept
    {
        if (auto* entry = find (provider))
        {
            if (entry->depth > 0)          // nested scope on this thread: reuse the slot
            {
                ++entry->depth;
                if (outDepth != nullptr) *outDepth = entry->depth;
                return entry->slot;
            }

            // Ownership-checked re-acquire: succeeds only while this thread still owns
            // the slot in THIS provider. A recycled provider address fails here.
            if (provider.reacquireReaderSlot (entry->slot))
            {
                entry->depth = 1;
                if (outDepth != nullptr) *outDepth = 1;
                return entry->slot;
            }

            entry->provider = nullptr;     // stale entry: drop it and register again
            entry->slot = -1;
            entry->depth = 0;
        }

        const int slot = (tag != nullptr) ? provider.registerReaderThreadWithTag (tag)
                                     : provider.registerReaderThread();

        if (slot < 0)
            return -1;   // fail-closed: registration closed or slots exhausted

        if (auto* entry = findFree())
        {
            entry->provider = &provider;
            entry->slot = slot;
            entry->depth = 1;
            if (outDepth != nullptr) *outDepth = 1;
        }

        return slot;
    }

    static void releaseSlot (const IReaderEpochProvider& provider, int slot) noexcept
    {
        if (auto* entry = find (provider))
            if (entry->slot == slot && entry->depth > 0)
                --entry->depth;
    }

    IReaderEpochProvider* provider_ = nullptr;
    int slot_ = -1;
    int depth_ = 0;         // ★ 計測: 入場時のネスト深さ
    int measureIdx_ = -1;   // ★ 計測: 固定表のレコード番号

public:
    //==========================================================================
    // ★ 計測 (2026-11-11 Owner 限定承認): guard インスタンス単位の入退場記録
    //   固定容量・atomic のみ。動的確保・mutex・RT ログは使わない。
    //   EBR の入場・退出・epoch・所有権・reclaim 判定には一切影響しない。
    //==========================================================================
    static constexpr int kMeasureSlots = 512;   // ★ 64 では 1 実行で飽和した（実測 64 件）

    struct MeasureRecord
    {
        std::atomic<uint8_t>  state{0};          // 0=free 1=active 2=exited
        std::atomic<uint64_t> enterUs{0};
        std::atomic<uint64_t> exitUs{0};
        std::atomic<uint64_t> threadTag{0};
        std::atomic<uint64_t> providerKey{0};
        std::atomic<uint32_t> slot{0xFFFFFFFFu};
        std::atomic<uint32_t> depth{0};
        std::atomic<uint32_t> seq{0};
        char tag[32]{};
    };

    static inline MeasureRecord measure_[kMeasureSlots]{};
    static inline std::atomic<uint32_t> measureSeq_{0};
    static inline std::atomic<uint64_t> measureFailures_{0};   // ★ 計測: 容量不足

    static std::uint64_t nowUsMeas() noexcept
    {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    static int claimMeasure (const char* tag, std::uint32_t slot, std::uint32_t depth,
                             const void* provider) noexcept
    {
        for (int i = 0; i < kMeasureSlots; ++i)
        {
            std::uint8_t expected = 0;
            if (measure_[i].state.compare_exchange_strong (expected, 1,
                                                           std::memory_order_acq_rel,
                                                           std::memory_order_relaxed))
            {
                measure_[i].enterUs.store (nowUsMeas(), std::memory_order_relaxed);
                measure_[i].exitUs.store (0, std::memory_order_relaxed);
                measure_[i].slot.store (slot, std::memory_order_relaxed);
                measure_[i].depth.store (depth, std::memory_order_relaxed);
                measure_[i].providerKey.store (static_cast<std::uint64_t> (
                    reinterpret_cast<std::uintptr_t> (provider)), std::memory_order_relaxed);
                measure_[i].threadTag.store (
                    static_cast<std::uint64_t> (std::hash<std::thread::id>{}(std::this_thread::get_id())),
                    std::memory_order_relaxed);
                measure_[i].seq.store (measureSeq_.fetch_add (1u, std::memory_order_relaxed) + 1u,
                                       std::memory_order_relaxed);
                std::memset (measure_[i].tag, 0, sizeof (measure_[i].tag));
                if (tag != nullptr)
                    std::strncpy (measure_[i].tag, tag, sizeof (measure_[i].tag) - 1);
                else
                    std::strncpy (measure_[i].tag, "untagged", sizeof (measure_[i].tag) - 1);
                return i;
            }
        }
        measureFailures_.fetch_add (1u, std::memory_order_relaxed);
        return -1;
    }

    static void closeMeasure (int idx) noexcept
    {
        if (idx < 0 || idx >= kMeasureSlots)
            return;
        measure_[idx].exitUs.store (nowUsMeas(), std::memory_order_relaxed);
        measure_[idx].state.store (2, std::memory_order_release);
    }

    // NonRT から呼ぶこと。タグ別集計と「終了時に active な guard」を分けて出す。
    static void dumpMeasure (std::FILE* f) noexcept
    {
        if (f == nullptr)
            return;

        const std::uint64_t now = nowUsMeas();
        std::fprintf (f, "{\n  \"schema\": \"guard_instance_measure_v1\",\n");
        std::fprintf (f, "  \"slots\": %d,\n", kMeasureSlots);

        // --- タグ別集計（退出済みのみ） ---
        std::fprintf (f, "  \"exitedByTag\": [");
        bool firstTag = true;
        for (int i = 0; i < kMeasureSlots; ++i)
        {
            const auto& r = measure_[i];
            if (r.state.load (std::memory_order_acquire) != 2)
                continue;
            const std::uint64_t hold = (r.exitUs.load (std::memory_order_relaxed)
                                        - r.enterUs.load (std::memory_order_relaxed));
            std::uint32_t count = 0;
            std::uint64_t maxHold = 0, sumHold = 0;
            for (int j = 0; j < kMeasureSlots; ++j)
            {
                const auto& q = measure_[j];
                if (q.state.load (std::memory_order_acquire) != 2)
                    continue;
                if (std::strncmp (q.tag, r.tag, sizeof (r.tag)) != 0)
                    continue;
                const std::uint64_t h = q.exitUs.load (std::memory_order_relaxed)
                                        - q.enterUs.load (std::memory_order_relaxed);
                ++count; sumHold += h; if (h > maxHold) maxHold = h;
            }
            // 同じタグは 1 回だけ出す
            bool already = false;
            for (int j = 0; j < i; ++j)
            {
                const auto& q = measure_[j];
                if (q.state.load (std::memory_order_acquire) == 2
                    && std::strncmp (q.tag, r.tag, sizeof (r.tag)) == 0)
                {
                    already = true; break;
                }
            }
            if (already)
                continue;
            std::fprintf (f, "%s\n    {\"tag\": \"%s\", \"count\": %u, \"maxHoldUs\": %llu, \"sumHoldUs\": %llu}",
                          firstTag ? "" : ",", r.tag, count,
                          (unsigned long long) maxHold, (unsigned long long) sumHold);
            (void) hold;
            firstTag = false;
        }
        std::fprintf (f, "%s],\n", firstTag ? "" : "\n  ");

        // --- 終了時に active な guard ---
        std::fprintf (f, "  \"activeAtExit\": [");
        bool firstActive = true;
        for (int i = 0; i < kMeasureSlots; ++i)
        {
            const auto& r = measure_[i];
            if (r.state.load (std::memory_order_acquire) != 1)
                continue;
            const std::uint64_t enter = r.enterUs.load (std::memory_order_relaxed);
            std::fprintf (f,
                "%s\n    {\"tag\": \"%s\", \"slot\": %u, \"depth\": %u, \"thread\": %llu, \"provider\": \"0x%llx\", \"enterUs\": %llu, \"holdUs\": %llu, \"seq\": %u}",
                firstActive ? "" : ",", r.tag,
                r.slot.load (std::memory_order_relaxed),
                r.depth.load (std::memory_order_relaxed),
                (unsigned long long) r.threadTag.load (std::memory_order_relaxed),
                (unsigned long long) r.providerKey.load (std::memory_order_relaxed),
                (unsigned long long) enter,
                (unsigned long long) (now - enter),
                r.seq.load (std::memory_order_relaxed));
            firstActive = false;
        }
        std::fprintf (f, "%s],\n", firstActive ? "" : "\n  ");

        // --- ネスト深さの分布（active のみ） ---
        std::fprintf (f, "  \"activeDepthHistogram\": {");
        bool firstDepth = true;
        for (std::uint32_t d = 1; d <= 8; ++d)
        {
            int n = 0;
            for (int i = 0; i < kMeasureSlots; ++i)
                if (measure_[i].state.load (std::memory_order_acquire) == 1
                    && measure_[i].depth.load (std::memory_order_relaxed) == d)
                    ++n;
            if (n == 0)
                continue;
            std::fprintf (f, "%s\"%u\": %d", firstDepth ? "" : ", ", d, n);
            firstDepth = false;
        }
        std::fprintf (f, "}\n}\n");
    }
};

} // namespace convo
