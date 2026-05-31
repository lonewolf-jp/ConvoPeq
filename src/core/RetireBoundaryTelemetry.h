#pragma once

#include <cstdint>

namespace convo {

struct RetireBoundaryTelemetry
{
    std::uint64_t pendingBacklog = 0;
    std::uint64_t quarantineResidents = 0;
    std::uint64_t totalTransitions = 0;
    bool boundaryActive = false;
};

} // namespace convo
