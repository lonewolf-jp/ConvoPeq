#include "ISRDSPQuarantine.h"
#include "AtomicAccess.h"

namespace convo::isr {

DSPQuarantineManager::DSPQuarantineManager(std::size_t maxSlots)
    : quarantineFlags_(maxSlots)
{
    for (auto& flag : quarantineFlags_) {
        convo::publishAtomic(flag, false, std::memory_order_relaxed);
    }
}

void DSPQuarantineManager::quarantineHandle(std::uint32_t slot, std::uint32_t) {
    if (slot < quarantineFlags_.size())
        convo::publishAtomic(quarantineFlags_[slot], true, std::memory_order_release);
}

void DSPQuarantineManager::reclaimSlot(std::uint32_t slot) {
    if (slot < quarantineFlags_.size())
        convo::publishAtomic(quarantineFlags_[slot], false, std::memory_order_release);
}

bool DSPQuarantineManager::isQuarantined(std::uint32_t slot) const {
    return slot < quarantineFlags_.size()
        && convo::consumeAtomic(quarantineFlags_[slot], std::memory_order_acquire);
}

} // namespace convo::isr
