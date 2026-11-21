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
        const float sinApprox =
            t * (1.0f + t2 * (-1.0f / 6.0f + t2 * (1.0f / 120.0f + t2 * (-1.0f / 5040.0f + t2 * (1.0f / 362880.0f)))));
        // ★ C-36: the truncated Taylor series overshoots 1 near x == 1 (≈1.0000035).
        //   A gain slightly above unity breaks the equal-power invariant; clamp.
        return sinApprox > 1.0f ? 1.0f : (sinApprox < 0.0f ? 0.0f : sinApprox);
    }
};

} // namespace convo
