#pragma once

#include "AudioEngine.h"

namespace convo::isr {

// CrossfadeAuthority: crossfade 判定のみを行う Authority。
// DSPCore を直接参照せず、RuntimeWorld.dspProjection の投影値のみで判断する。
// Registration (registerCrossfade) は DSPTransition が担当。
class CrossfadeAuthority {
public:
    struct Decision {
        bool needsCrossfade = false;
        bool oldHasIR = false;
        bool newHasIR = false;
        double fadeTimeSec = 0.0;
    };

    explicit CrossfadeAuthority() noexcept = default;

    // evaluate: RuntimeWorld の dspProjection 投影値のみで crossfade 要否を判断。
    // DSPCore 直読は行わない。engine は atomic フェード時間設定の読み取りにのみ使用。
    [[nodiscard]] Decision evaluate(
        const AudioEngine& engine,
        const RuntimePublishWorld& oldWorld,
        const RuntimePublishWorld& newWorld) noexcept;

    // [0-6] Coverage Contract: evaluate() が参照する dspProjection 全フィールド名
    static constexpr std::array<const char*, 3> kEvaluateRelevantFieldNames {{
        "irLoaded",
        "structuralHash",
        "oversamplingFactor"
    }};
};

} // namespace convo::isr
