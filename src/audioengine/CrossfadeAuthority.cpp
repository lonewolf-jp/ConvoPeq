#include "CrossfadeAuthority.h"
#include "AudioEngine.h"
#include "AtomicAccess.h"

namespace convo::isr {

CrossfadeAuthority::Decision CrossfadeAuthority::evaluate(
    const AudioEngine& engine,
    const RuntimePublishWorld& oldWorld,
    const RuntimePublishWorld& newWorld) noexcept
{
    Decision ctx;

    // Use dspProjection values from RuntimeWorld (DSPCore 直読は行わない)
    ctx.oldHasIR = oldWorld.dspProjection.irLoaded;
    ctx.newHasIR = newWorld.dspProjection.irLoaded;
    const bool hasAudibleConvolverTransition = ctx.oldHasIR || ctx.newHasIR;
    const bool irPresenceChanged = (ctx.oldHasIR != ctx.newHasIR);

    // Oversampling change detection
    if (hasAudibleConvolverTransition
        && newWorld.dspProjection.oversamplingFactor != oldWorld.dspProjection.oversamplingFactor)
    {
        ctx.needsCrossfade = true;
        ctx.fadeTimeSec = std::max(ctx.fadeTimeSec,
            convo::consumeAtomic(engine.m_osFadeTimeSec, std::memory_order_acquire));
    }

    // IR structural change detection
    if (hasAudibleConvolverTransition)
    {
        const uint64_t oldHash = oldWorld.dspProjection.structuralHash;
        const uint64_t newHash = newWorld.dspProjection.structuralHash;
        if (oldHash != newHash)
        {
            ctx.needsCrossfade = true;
            const double baseIrFade = convo::consumeAtomic(
                engine.m_irFadeTimeSec, std::memory_order_acquire);
            if (irPresenceChanged)
            {
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec,
                    std::clamp(baseIrFade, 0.001, 0.010));
            }
            else
            {
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, baseIrFade);
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec,
                    convo::consumeAtomic(engine.m_irLengthFadeTimeSec, std::memory_order_acquire));
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec,
                    convo::consumeAtomic(engine.m_phaseFadeTimeSec, std::memory_order_acquire));
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec,
                    convo::consumeAtomic(engine.m_directHeadFadeTimeSec, std::memory_order_acquire));
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec,
                    convo::consumeAtomic(engine.m_nucFilterFadeTimeSec, std::memory_order_acquire));
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec,
                    convo::consumeAtomic(engine.m_tailFadeTimeSec, std::memory_order_acquire));
            }
        }
    }

    return ctx;
}

} // namespace convo::isr
