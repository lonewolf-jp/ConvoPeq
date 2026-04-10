//==============================================================================
// ReaderEpoch.cpp
//==============================================================================
#include "ReaderEpoch.h"
#include <cassert>
#include <JuceHeader.h>

namespace convo {

std::atomic<uint64_t> ReaderEpoch::s_readerEpochs[kMaxReaders];
std::atomic<uint64_t> ReaderEpoch::s_globalEpoch{1};
static std::atomic<size_t> s_nextSlot{0};
thread_local size_t t_readerSlot = SIZE_MAX;

size_t ReaderEpoch::getThreadSlot() noexcept
{
    if (t_readerSlot == SIZE_MAX) {
        size_t newSlot = s_nextSlot.fetch_add(1, std::memory_order_relaxed);
        if (newSlot >= kMaxReaders) {
            // スロット枯渇時は共有せず、無効なスロットを返す
#ifdef _DEBUG
            assert(false && "RCU slot exhausted. Increase kMaxReaders.");
#else
            DBG("RCU slot exhausted, new readers will not be protected.");
#endif
            newSlot = kOverflowSlot;
        } else {
            s_readerEpochs[newSlot].store(kIdleEpoch, std::memory_order_relaxed);
        }
        t_readerSlot = newSlot;
    }
    return t_readerSlot;
}

void ReaderEpoch::enter(size_t slot) noexcept
{
    if (slot >= kMaxReaders) return;
    const uint64_t epoch = s_globalEpoch.load(std::memory_order_acquire);
    s_readerEpochs[slot].store(epoch, std::memory_order_release);
}

void ReaderEpoch::exit(size_t slot) noexcept
{
    if (slot >= kMaxReaders) return;
    s_readerEpochs[slot].store(kIdleEpoch, std::memory_order_relaxed);
}

uint64_t ReaderEpoch::getMinActiveEpoch() noexcept
{
    uint64_t minEpoch = s_globalEpoch.load(std::memory_order_acquire);
    for (size_t i = 0; i < kMaxReaders; ++i) {
        uint64_t slotEpoch = s_readerEpochs[i].load(std::memory_order_acquire);
        if (slotEpoch != kIdleEpoch && slotEpoch < minEpoch) {
            minEpoch = slotEpoch;
        }
    }
    return minEpoch;
}

uint64_t ReaderEpoch::advanceGlobalEpoch() noexcept
{
    return s_globalEpoch.fetch_add(1, std::memory_order_acq_rel) + 1;
}

uint64_t ReaderEpoch::getCurrentGlobalEpoch() noexcept
{
    return s_globalEpoch.load(std::memory_order_acquire);
}

} // namespace convo
