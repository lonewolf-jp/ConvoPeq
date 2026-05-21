#pragma once
#include <cstdint>
#include <vector>
#include <atomic>

namespace convo::isr {

class DSPQuarantineManager {
public:
    explicit DSPQuarantineManager(std::size_t maxSlots = 256);
    void quarantineHandle(std::uint32_t slot, std::uint32_t generation);
    void reclaimSlot(std::uint32_t slot);
    bool isQuarantined(std::uint32_t slot) const;
private:
    std::vector<std::atomic<bool>> quarantineFlags_;
};

} // namespace convo::isr
