#include "CrossfadeAuthority.h"
#include "AtomicAccess.h"
#include <algorithm>
#include <cstdint>

namespace convo::isr {

CrossfadeAuthority::Decision CrossfadeAuthority::evaluate(
    const RuntimePublishWorld& oldWorld,
    const RuntimePublishWorld& newWorld,
    const CrossfadePolicy& policy) noexcept
{
    Decision ctx;
    // ★ evaluate は純粋に dspProjection 投影値 + Policy 静的設定のみで判断
    //   Critical 時の crossfade 抑制は Orchestrator（makeCrossfadePolicy 後 evaluate 前）または
    //   DSPTransition Emergency Override が担当する。evaluate 自身は HealthState を知らない。

    ctx.oldHasIR = oldWorld.dspProjection.irLoaded;
    ctx.newHasIR = newWorld.dspProjection.irLoaded;
    const bool hasTransition = ctx.oldHasIR || ctx.newHasIR;
    const bool irChanged = (ctx.oldHasIR != ctx.newHasIR);

    if (hasTransition && newWorld.dspProjection.oversamplingFactor != oldWorld.dspProjection.oversamplingFactor) {
        ctx.needsCrossfade = true;
        ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.osFadeTimeSec);
    }
    if (hasTransition) {
        const uint64_t oh = oldWorld.dspProjection.structuralHash;
        const uint64_t nh = newWorld.dspProjection.structuralHash;
        if (oh != nh) {
            ctx.needsCrossfade = true;
            if (irChanged) {
                double clamped = policy.irFadeTimeSec;
                if (clamped < 0.001) clamped = 0.001;
                if (clamped > 0.010) clamped = 0.010;
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, clamped);
            } else {
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.irFadeTimeSec);
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.irLengthFadeTimeSec);
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.phaseFadeTimeSec);
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.directHeadFadeTimeSec);
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.nucFilterFadeTimeSec);
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.tailFadeTimeSec);
            }
        }
    }
    return ctx;
}

} // namespace convo::isr
