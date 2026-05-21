#pragma once
#include <cstdint>

namespace convo::isr {

enum class RetireLane {
    RTIntent,
    Coordination,
    Epoch,
    Reclaim,
    Quarantine
};

} // namespace convo::isr
