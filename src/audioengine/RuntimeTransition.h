#pragma once

#include <cstdint>

namespace convo {

enum class TransitionPolicy : uint8_t
{
    SmoothOnly = 0,
    HardReset,
    DryAsOld
};

// Runtime publish/crossfade の観測用ステート。
// 所有権は持たず、ポインタは状態可視化のためにのみ保持する。
struct TransitionState
{
    void* current = nullptr;
    void* next = nullptr;
    TransitionPolicy policy = TransitionPolicy::SmoothOnly;
    double fadeTimeSec = 0.0;
    bool active = false;
};

} // namespace convo
