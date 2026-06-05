#include "CrossfadeAuthority.h"
#include "AudioEngine.h"
#include "AtomicAccess.h"

namespace convo::isr {

CrossfadeAuthority::Decision CrossfadeAuthority::evaluateAndRegister(
    AudioEngine& engine,
    AudioEngine::DSPCore* oldDSP,
    AudioEngine::DSPCore* newDSP,
    DSPHandle oldHandle,
    DSPHandle newHandle) noexcept
{
    auto decision = computeDecision(engine, oldDSP, newDSP);

    if (decision.needsCrossfade && !oldHandle.isNull() && !newHandle.isNull())
    {
        doRegister(engine, oldHandle, newHandle);
    }

    return decision;
}

CrossfadeAuthority::Decision CrossfadeAuthority::evaluateOnly(
    AudioEngine& engine,
    AudioEngine::DSPCore* oldDSP,
    AudioEngine::DSPCore* newDSP) noexcept
{
    return computeDecision(engine, oldDSP, newDSP);
}

CrossfadeAuthority::Decision CrossfadeAuthority::computeDecision(
    const AudioEngine& engine,
    const AudioEngine::DSPCore* oldDSP,
    const AudioEngine::DSPCore* newDSP) noexcept
{
    Decision ctx;

    if (oldDSP == nullptr || newDSP == nullptr)
        return ctx;

    // IR presence check
    ctx.oldHasIR = oldDSP->convolverRt().isIRLoaded();
    ctx.newHasIR = newDSP->convolverRt().isIRLoaded();
    const bool hasAudibleConvolverTransition = ctx.oldHasIR || ctx.newHasIR;
    const bool irPresenceChanged = (ctx.oldHasIR != ctx.newHasIR);

    // Oversampling change detection
    if (hasAudibleConvolverTransition
        && newDSP->oversamplingFactor != oldDSP->oversamplingFactor)
    {
        ctx.needsCrossfade = true;
        ctx.fadeTimeSec = std::max(ctx.fadeTimeSec,
            convo::consumeAtomic(engine.m_osFadeTimeSec, std::memory_order_acquire));
    }

    // IR structural change detection
    if (hasAudibleConvolverTransition)
    {
        const uint64_t oldHash = oldDSP->convolverRt().getStructuralHash();
        const uint64_t newHash = newDSP->convolverRt().getStructuralHash();
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

void CrossfadeAuthority::doRegister(AudioEngine& engine,
                                     DSPHandle from, DSPHandle to) noexcept
{
    engine.crossfadeAuthorityRuntime_.registerCrossfade(from, to);
}

CrossfadeAuthority::Decision CrossfadeAuthority::evaluateFromWorlds(
    const AudioEngine& engine,
    const RuntimePublishWorld& oldWorld,
    const RuntimePublishWorld& newWorld) noexcept
{
    Decision ctx;

    // IR presence check from world projections
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
