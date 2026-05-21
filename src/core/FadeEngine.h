//==============================================================================
// FadeEngine.h - Fade math utilities (RT-safe, libm-free)
//==============================================================================
#pragma once

namespace convo {

class FadeEngine {
public:
    FadeEngine() = delete;

    // Audio Thread safe equal-power approximation (sin(pi/2*x), libm free)
    static float equalPowerSinApprox(float x) noexcept
    {
        const float t = x * 1.5707963267948966f;
        const float t2 = t * t;
        return t * (1.0f + t2 * (-1.0f / 6.0f + t2 * (1.0f / 120.0f + t2 * (-1.0f / 5040.0f + t2 * (1.0f / 362880.0f)))));
    }
};

} // namespace convo
