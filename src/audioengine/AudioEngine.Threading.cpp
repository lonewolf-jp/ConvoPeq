#include <JuceHeader.h>
#include "AudioEngine.h"

namespace
{
    constexpr size_t kReclaimBacklogWarnThreshold = 128;
    constexpr int kBaseRetireHighWatermark = 3072;
    constexpr int kBaseRetireLowWatermark = 1024;
    constexpr int kEmergencyReclaimBoostWindowMs = 500;
    constexpr int kEmergencyReclaimBoostMaxCount = 2;
    constexpr int kEmergencyReclaimBoostMinIntervalMs = 10;
    constexpr int kRetirePressureMildPercent = 75;
    constexpr int kRetirePressureMediumPercent = 90;
    constexpr int kRetirePressureSeverePercent = 95;

    [[nodiscard]] double clampScale(double value) noexcept
    {
        return juce::jlimit(0.75, 1.50, value);
    }
}

void AudioEngine::destroyDSPCoreNode(void* p) noexcept
{
    auto* core = static_cast<DSPCore*>(p);
    core->~DSPCore();
    convo::aligned_free(core);
}

[[nodiscard]] uint64_t AudioEngine::publishRcuEpoch() noexcept
{
    return currentRetireEpoch();
}

[[nodiscard]] uint64_t AudioEngine::publishRetireEpoch() noexcept
{
    return m_epochDomain.publish();
}

[[nodiscard]] uint64_t AudioEngine::currentRetireEpoch() const noexcept
{
    return m_epochDomain.current();
}

uint64_t AudioEngine::advanceRetireEpoch() noexcept
{
    return m_epochDomain.advanceEpoch();
}

[[nodiscard]] bool AudioEngine::enqueueRetireEpochBounded(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept
{
    return m_epochDomain.enqueueRetire(ptr, deleter, epoch);
}

[[nodiscard]] uint32_t AudioEngine::activeEpochObserverCount() const noexcept
{
    return m_epochDomain.activeReaderCount();
}

void AudioEngine::enterRcuReader(int readerIndex) noexcept
{
    m_epochDomain.enterReader(readerIndex);
}

void AudioEngine::exitRcuReader(int readerIndex) noexcept
{
    m_epochDomain.exitReader(readerIndex);
}

void AudioEngine::tryReclaimResources() noexcept
{
    convo::fetchAddAtomic(rtAuxMutable_.runtimeReclaimCount, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    m_epochDomain.reclaimRetired();
}

void AudioEngine::drainDeferredRetireQueues(bool allowDuringShutdown) noexcept
{
    if (!allowDuringShutdown && isShutdownInProgress())
        return;

    const double startMs = juce::Time::getMillisecondCounterHiRes();

    auto flushAudioThreadRetireOverflow = [this]() noexcept
    {
        auto* overflow = convo::exchangeAtomic(audioThreadRetireOverflowPtr, nullptr, std::memory_order_acq_rel);
        if (overflow == nullptr)
            return;

        const uint64_t overflowEpoch = convo::exchangeAtomic(rtLocalState_.audioThreadRetireOverflowEpoch, 0, std::memory_order_acq_rel);
        const uint64_t epoch = overflowEpoch != 0 ? overflowEpoch : currentRetireEpoch();
        if (!enqueueRetireEpochBounded(
                overflow,
                &AudioEngine::destroyDSPCoreNode,
                epoch))
        {
            std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
            deferredDeleteFallbackQueue.push_back(DeferredDeleteFallbackEntry {
                overflow,
                &AudioEngine::destroyDSPCoreNode,
                epoch
            });
        }
    };

    auto flushDeferredDeleteFallbackQueue = [this]() noexcept
    {
        std::vector<DeferredDeleteFallbackEntry> pending;
        {
            std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
            if (deferredDeleteFallbackQueue.empty())
                return;
            pending.swap(deferredDeleteFallbackQueue);
        }

        for (auto& entry : pending)
        {
            const uint64_t epoch = entry.epoch != 0 ? entry.epoch : currentRetireEpoch();
            if (!enqueueRetireEpochBounded(entry.ptr, entry.deleter, epoch))
            {
                std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
                deferredDeleteFallbackQueue.push_back(entry);
            }
        }
    };

    flushAudioThreadRetireOverflow();
    flushDeferredDeleteFallbackQueue();
    runtimePublicationBridge_.setReclaimInFlightCount(1);
    m_epochDomain.reclaimRetired();
    m_coordinator.reclaim(m_epochDomain);
    runtimePublicationBridge_.setReclaimInFlightCount(0);

    const auto fallbackDepth = [this]() noexcept -> std::uint64_t
    {
        std::lock_guard<std::mutex> lock(deferredDeleteFallbackMutex);
        return static_cast<std::uint64_t>(deferredDeleteFallbackQueue.size());
    }();

    const std::uint64_t overflowDepth = convo::consumeAtomic(audioThreadRetireOverflowPtr, std::memory_order_acquire) != nullptr ? 1ull : 0ull;
    const std::uint64_t retireDepth = overflowDepth + fallbackDepth;

    convo::publishAtomic(fallbackQueueDepth_, fallbackDepth, std::memory_order_release);
    convo::publishAtomic(retireQueueDepth_, retireDepth, std::memory_order_release);
    runtimePublicationBridge_.setFallbackBacklogCount(fallbackDepth);
    runtimePublicationBridge_.setRetireBacklogCount(retireDepth);
    runtimePublicationBridge_.setDeferredRetireResidencyCount(fallbackDepth);

    const std::uint64_t quarantineResident = retireRuntimeEx_.getQuarantineResidentCount();
    convo::publishAtomic(quarantineResident_, quarantineResident, std::memory_order_release);

    const auto computeBackpressureScales = [this, retireDepth, fallbackDepth, quarantineResident]() noexcept
    {
        const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
        const int osFactorRaw = convo::consumeAtomic(manualOversamplingFactor, std::memory_order_acquire);
        const int osFactor = osFactorRaw > 0 ? osFactorRaw : 1;
        const std::uint64_t rebuildBacklog = convo::consumeAtomic(rebuildBacklog_, std::memory_order_acquire);
        const std::uint64_t publicationBacklog = convo::consumeAtomic(publicationBacklog_, std::memory_order_acquire);

        const double sampleRateScale = clampScale((sr > 0.0) ? (sr / 48000.0) : 1.0);
        const double oversamplingScale = clampScale(static_cast<double>(osFactor));
        const double irComplexityScale = clampScale(1.0 + static_cast<double>(rebuildBacklog + publicationBacklog) * 0.02);
        const double memoryPressureRaw = 1.0
            + static_cast<double>(retireDepth) / static_cast<double>(kBaseRetireHighWatermark)
            + static_cast<double>(fallbackDepth) / static_cast<double>(kBaseRetireLowWatermark)
            + static_cast<double>(quarantineResident) / static_cast<double>(kBaseRetireLowWatermark);
        const double memoryPressureScale = clampScale(memoryPressureRaw);

        const double combinedScale = clampScale((sampleRateScale + oversamplingScale + irComplexityScale + memoryPressureScale) * 0.25);

        return std::pair<int, int> {
            std::max(kBaseRetireHighWatermark, static_cast<int>(std::llround(static_cast<double>(kBaseRetireHighWatermark) * combinedScale))),
            std::max(kBaseRetireLowWatermark, static_cast<int>(std::llround(static_cast<double>(kBaseRetireLowWatermark) * combinedScale)))
        };
    };

    auto [targetHwm, targetLwm] = computeBackpressureScales();
    if (targetLwm >= targetHwm)
        targetLwm = std::max(kBaseRetireLowWatermark, targetHwm - 1);

    int hwm = std::max(kBaseRetireHighWatermark, convo::consumeAtomic(retireHighWatermark_, std::memory_order_acquire));
    int lwm = std::min(hwm - 1, std::max(kBaseRetireLowWatermark, convo::consumeAtomic(retireLowWatermark_, std::memory_order_acquire)));
    const bool wasSaturated = convo::consumeAtomic(retireSaturationActive_, std::memory_order_acquire);

    if (retireDepth >= static_cast<std::uint64_t>(lwm))
    {
        // saturation 方向では単調に強化のみ（緩和禁止）
        hwm = std::max(hwm, targetHwm);
        lwm = std::max(lwm, std::min(targetLwm, hwm - 1));
        convo::publishAtomic(retireHighWatermark_, hwm, std::memory_order_release);
        convo::publishAtomic(retireLowWatermark_, lwm, std::memory_order_release);
    }

    const bool nowSaturated = retireDepth >= static_cast<std::uint64_t>(hwm);
    const std::uint64_t publishCount = convo::consumeAtomic(rtAuxMutable_.runtimePublishCount, std::memory_order_acquire);

    const int retirePressureLevel = evaluateRetirePressureLevelNoRt(retireDepth, hwm);
    applyRetirePressurePolicyNoRt(retirePressureLevel, retireDepth);

    if (!wasSaturated && nowSaturated)
    {
        convo::publishAtomic(retireSaturationActive_, true, std::memory_order_release);
        convo::publishAtomic(retireSaturationRecoveryPending_, false, std::memory_order_release);
        convo::publishAtomic(retireSaturationRecoveryBaselinePublishCount_, publishCount, std::memory_order_release);
        convo::fetchAddAtomic(saturationEnterCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    }
    else if (wasSaturated)
    {
        const bool belowOrEqualLowWatermark = retireDepth <= static_cast<std::uint64_t>(lwm);
        if (belowOrEqualLowWatermark)
        {
            const bool recoveryPending = convo::consumeAtomic(retireSaturationRecoveryPending_, std::memory_order_acquire);
            if (!recoveryPending)
            {
                convo::publishAtomic(retireSaturationRecoveryPending_, true, std::memory_order_release);
                convo::publishAtomic(retireSaturationRecoveryBaselinePublishCount_, publishCount, std::memory_order_release);
            }
            else
            {
                const std::uint64_t baselinePublishCount = convo::consumeAtomic(retireSaturationRecoveryBaselinePublishCount_, std::memory_order_acquire);
                const bool observedAtLeastOnePublicationCycle = publishCount > baselinePublishCount;
                if (observedAtLeastOnePublicationCycle)
                {
                    // stepwise conservative recovery（128刻み）
                    hwm = std::max(kBaseRetireHighWatermark, hwm - 128);
                    lwm = std::max(kBaseRetireLowWatermark, lwm - 128);
                    if (lwm >= hwm)
                        lwm = std::max(kBaseRetireLowWatermark, hwm - 1);
                    convo::publishAtomic(retireHighWatermark_, hwm, std::memory_order_release);
                    convo::publishAtomic(retireLowWatermark_, lwm, std::memory_order_release);

                    convo::publishAtomic(retireSaturationActive_, false, std::memory_order_release);
                    convo::publishAtomic(retireSaturationRecoveryPending_, false, std::memory_order_release);
                    convo::fetchAddAtomic(saturationExitCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
                }
            }
        }
        else
        {
            convo::publishAtomic(retireSaturationRecoveryPending_, false, std::memory_order_release);
            convo::publishAtomic(retireSaturationRecoveryBaselinePublishCount_, publishCount, std::memory_order_release);
        }
    }

    {
        const bool saturatedNow = convo::consumeAtomic(retireSaturationActive_, std::memory_order_acquire);
        const bool protectiveModeActive = convo::consumeAtomic(retireProtectiveModeActive_, std::memory_order_acquire);
        const std::int64_t nowTicks = juce::Time::getHighResolutionTicks();
        const std::int64_t ticksPerSecond = juce::Time::getHighResolutionTicksPerSecond();
        const std::int64_t boostWindowTicks = (ticksPerSecond * kEmergencyReclaimBoostWindowMs) / 1000;
        const std::int64_t minIntervalTicks = (ticksPerSecond * kEmergencyReclaimBoostMinIntervalMs) / 1000;

        if (saturatedNow || protectiveModeActive)
        {
            std::int64_t windowStart = convo::consumeAtomic(emergencyReclaimWindowStartTicks_, std::memory_order_acquire);
            if (windowStart <= 0 || (boostWindowTicks > 0 && (nowTicks - windowStart) > boostWindowTicks))
            {
                convo::publishAtomic(emergencyReclaimWindowStartTicks_, nowTicks, std::memory_order_release);
                convo::publishAtomic(emergencyReclaimBoostCount_, 0, std::memory_order_release);
                windowStart = nowTicks;
            }

            const int boostCount = convo::consumeAtomic(emergencyReclaimBoostCount_, std::memory_order_acquire);
            const std::int64_t lastBoostTicks = convo::consumeAtomic(emergencyReclaimLastBoostTicks_, std::memory_order_acquire);
            const bool withinWindow = (boostWindowTicks <= 0) || ((nowTicks - windowStart) <= boostWindowTicks);
            const bool intervalReady = (lastBoostTicks <= 0) || (minIntervalTicks <= 0) || ((nowTicks - lastBoostTicks) >= minIntervalTicks);

            if (withinWindow && boostCount < kEmergencyReclaimBoostMaxCount && intervalReady)
            {
                runtimePublicationBridge_.setReclaimInFlightCount(1);
                m_epochDomain.reclaimRetired();
                m_coordinator.reclaim(m_epochDomain);
                runtimePublicationBridge_.setReclaimInFlightCount(0);

                convo::publishAtomic(emergencyReclaimLastBoostTicks_, nowTicks, std::memory_order_release);
                convo::fetchAddAtomic(emergencyReclaimBoostCount_, 1, std::memory_order_acq_rel);
            }
        }
        else
        {
            convo::publishAtomic(emergencyReclaimWindowStartTicks_, 0, std::memory_order_release);
            convo::publishAtomic(emergencyReclaimLastBoostTicks_, 0, std::memory_order_release);
            convo::publishAtomic(emergencyReclaimBoostCount_, 0, std::memory_order_release);
        }
    }

    const double elapsedMs = juce::Time::getMillisecondCounterHiRes() - startMs;
    convo::publishAtomic(reclaimLatency_, elapsedMs, std::memory_order_release);

    const uint64_t dropped = convo::consumeAtomic(rtLocalState_.audioThreadRetireEnqueueDropped, std::memory_order_acquire);
    if (dropped >= kReclaimBacklogWarnThreshold)
    {
        juce::Logger::writeToLog("[DIAG] deferred reclaim enqueue drops=" + juce::String(static_cast<juce::int64>(dropped)));
    }
}

int AudioEngine::evaluateRetirePressureLevelNoRt(std::uint64_t retireDepth,
                                                 int highWatermark) const noexcept
{
    const int safeHwm = std::max(1, highWatermark);
    const std::uint64_t ratioPercent = (retireDepth * 100ull) / static_cast<std::uint64_t>(safeHwm);

    if (ratioPercent >= static_cast<std::uint64_t>(kRetirePressureSeverePercent))
        return 3;
    if (ratioPercent >= static_cast<std::uint64_t>(kRetirePressureMediumPercent))
        return 2;
    if (ratioPercent >= static_cast<std::uint64_t>(kRetirePressureMildPercent))
        return 1;
    return 0;
}

void AudioEngine::applyRetirePressurePolicyNoRt(int retirePressureLevel,
                                                std::uint64_t retireDepth) noexcept
{
    const int previousLevel = convo::exchangeAtomic(retirePressureLevel_, retirePressureLevel, std::memory_order_acq_rel);
    const bool previousProtective = convo::consumeAtomic(retireProtectiveModeActive_, std::memory_order_acquire);

    const bool mild = retirePressureLevel >= 1;
    const bool medium = retirePressureLevel >= 2;
    const bool severe = retirePressureLevel >= 3;
    const bool critical = severe && (retireDepth >= static_cast<std::uint64_t>(std::max(1, convo::consumeAtomic(retireHighWatermark_, std::memory_order_acquire))));

    convo::publishAtomic(retirePressureCoalescingActive_, mild, std::memory_order_release);
    convo::publishAtomic(retirePressurePublicationThrottleActive_, medium, std::memory_order_release);
    convo::publishAtomic(retirePressureAdmissionStrict_, severe, std::memory_order_release);
    convo::publishAtomic(retireProtectiveModeActive_, critical, std::memory_order_release);

    if (!previousProtective && critical)
    {
        convo::fetchAddAtomic(retireProtectiveModeEnterCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    }

    if (previousLevel != retirePressureLevel)
    {
        juce::Logger::writeToLog("[DIAG] retire pressure level changed="
            + juce::String(retirePressureLevel)
            + " retireDepth=" + juce::String(static_cast<juce::int64>(retireDepth))
            + " hwm=" + juce::String(convo::consumeAtomic(retireHighWatermark_, std::memory_order_acquire))
            + " policy{coalesce=" + juce::String(mild ? 1 : 0)
            + ", throttle=" + juce::String(medium ? 1 : 0)
            + ", strict=" + juce::String(severe ? 1 : 0)
            + ", protective=" + juce::String(critical ? 1 : 0)
            + "}");
    }
}

bool AudioEngine::shouldRejectRebuildAdmissionForPressure() const noexcept
{
    return convo::consumeAtomic(retirePressureAdmissionStrict_, std::memory_order_acquire);
}

bool AudioEngine::isFullyDrained() noexcept
{
    if (convo::consumeAtomic(commitDrainInProgress, std::memory_order_acquire))
        return false;

    const std::uint64_t publicationBacklog = convo::consumeAtomic(publicationBacklog_, std::memory_order_acquire);
    runtimePublicationBridge_.setPublicationBacklogCount(publicationBacklog);

    const bool hasPendingIntents = hasPendingPublicationIntents();
    runtimePublicationBridge_.setPendingIntentCount(hasPendingIntents ? 1u : 0u);

    const std::uint64_t fallbackDepth = convo::consumeAtomic(fallbackQueueDepth_, std::memory_order_acquire);
    const std::uint64_t retireDepth = convo::consumeAtomic(retireQueueDepth_, std::memory_order_acquire);
    runtimePublicationBridge_.setFallbackBacklogCount(fallbackDepth);
    runtimePublicationBridge_.setRetireBacklogCount(retireDepth);
    runtimePublicationBridge_.setDeferredRetireResidencyCount(fallbackDepth);

    return runtimePublicationBridge_.isFullyDrained();
}

bool AudioEngine::waitForDrain(int timeoutMs, int pollIntervalMs) noexcept
{
    ASSERT_NON_RT_THREAD();

    const int boundedTimeoutMs = juce::jlimit(1, 10000, timeoutMs);
    const int boundedPollIntervalMs = juce::jlimit(1, 5, pollIntervalMs);

    const double startMs = juce::Time::getMillisecondCounterHiRes();
    while (!isFullyDrained())
    {
        drainPublicationLogForShutdown();
        drainDeferredRetireQueues(true);

        const double elapsedMs = juce::Time::getMillisecondCounterHiRes() - startMs;
        if (elapsedMs >= static_cast<double>(boundedTimeoutMs))
            return false;

        juce::Thread::sleep(boundedPollIntervalMs);
    }

    return true;
}

void AudioEngine::processDeferredReleases()
{
    drainDeferredRetireQueues(false);
}
