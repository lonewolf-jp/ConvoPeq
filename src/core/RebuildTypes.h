#pragma once

#include <cstdint>

namespace convo {

enum class RebuildKind : uint32_t
{
    None       = 0,
    Structural = 1 << 0,
    Runtime    = 1 << 2
};

} // namespace convo
