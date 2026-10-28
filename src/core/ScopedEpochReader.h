#pragma once

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
    explicit ScopedEpochReader (IReaderEpochProvider& provider) noexcept
        : provider_ (&provider),
          slot_ (acquireSlot (provider))
    {
        if (slot_ >= 0)
            provider_->enterReader (slot_);
    }

    ~ScopedEpochReader() { reset(); }

    ScopedEpochReader (const ScopedEpochReader&) = delete;
    ScopedEpochReader& operator= (const ScopedEpochReader&) = delete;

    ScopedEpochReader (ScopedEpochReader&& other) noexcept
        : provider_ (other.provider_),
          slot_ (other.slot_)
    {
        other.provider_ = nullptr;
        other.slot_ = -1;
    }

    ScopedEpochReader& operator= (ScopedEpochReader&&) = delete;

    /** True when a reader slot was obtained and enterReader() ran. */
    [[nodiscard]] bool active() const noexcept { return slot_ >= 0; }

    /** Reader slot index (diagnostics); -1 when inactive. */
    [[nodiscard]] int slotIndex() const noexcept { return slot_; }

private:
    void reset() noexcept
    {
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
        thread_local std::array<CacheEntry, 4> entries {};
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

    static int acquireSlot (IReaderEpochProvider& provider) noexcept
    {
        if (auto* entry = find (provider))
        {
            if (entry->depth > 0)          // nested scope on this thread: reuse the slot
            {
                ++entry->depth;
                return entry->slot;
            }

            // Ownership-checked re-acquire: succeeds only while this thread still owns
            // the slot in THIS provider. A recycled provider address fails here.
            if (provider.reacquireReaderSlot (entry->slot))
            {
                entry->depth = 1;
                return entry->slot;
            }

            entry->provider = nullptr;     // stale entry: drop it and register again
            entry->slot = -1;
            entry->depth = 0;
        }

        const int slot = provider.registerReaderThread();

        if (slot < 0)
            return -1;   // fail-closed: registration closed or slots exhausted

        if (auto* entry = findFree())
        {
            entry->provider = &provider;
            entry->slot = slot;
            entry->depth = 1;
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
};

} // namespace convo
