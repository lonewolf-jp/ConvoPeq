#pragma once

#include "AudioEngine.h"

namespace convo::isr {

// ★ CrossfadePolicy: immutable POD。メソッドや状態を持たない。
//   静的設定（フェード時間・閾値）のみを保持し、実行時状態（HealthState）は含めない。
//   将来 CrossfadeSettings 一括atomic化との統合を考慮し zero-init 必須。
struct CrossfadePolicy {
    double osFadeTimeSec{};
    double irFadeTimeSec{};
    double irLengthFadeTimeSec{};
    double phaseFadeTimeSec{};
    double directHeadFadeTimeSec{};
    double nucFilterFadeTimeSec{};
    double tailFadeTimeSec{};
};

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

    // evaluate: RuntimeWorld の dspProjection 投影値と CrossfadePolicy 静的設定のみで
    // crossfade 要否を判断。AudioEngine には依存しない。
    // ★ HealthState Critical の抑制は Orchestrator または DSPTransition Emergency Override が担当。
    [[nodiscard]] Decision evaluate(
        const RuntimePublishWorld& oldWorld,
        const RuntimePublishWorld& newWorld,
        const CrossfadePolicy& policy) noexcept;

    // [0-6] Coverage Contract: evaluate() が参照する dspProjection 全フィールド名
    static constexpr std::array<const char*, 3> kEvaluateRelevantFieldNames {{
        "irLoaded",
        "structuralHash",
        "oversamplingFactor"
    }};
};

} // namespace convo::isr
