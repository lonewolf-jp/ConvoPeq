#pragma once

#include <cstdint>

namespace convo {

enum class RebuildKind : uint32_t
{
    None       = 0,
    Structural = 1 << 0,
    IRContent  = 1 << 1,
    Runtime    = 1 << 2
};

inline uint32_t toMask(RebuildKind kind) noexcept
{
    return static_cast<uint32_t>(kind);
}

} // namespace convo
