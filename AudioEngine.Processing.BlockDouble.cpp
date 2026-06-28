#include <JuceHeader.h>
#include "AudioEngine.h"
#include "DiagnosticsConfig.h"
#include "core/RuntimeReaderContext.h"
#include "NoiseShaperLearner.h"
#include "core/RCUReader.h"

namespace
{
    inline double absDiffNoLibm(double a, double b) noexcept
    {
        return absNoLibm(a - b);
    }
}

void AudioEngine::processBlockDouble (juce::AudioBuffer<double>& buffer)
{
    const auto lifecycle = convo::consumeAtomic(lifecycleState, std::memory_order_acquire);
    if (lifecycle != EngineLifecycleState::Prepared)
    {
        buffer.clear();
        return;
    }

    if (isShutdownInProgress())
    {
        shutdownRuntime_.markLateCallback();
        buffer.clear();
        return;
    }

    struct AudioCallbackRuntimeScope final
    {
        AudioEngine& engine;
        convo::isr::LifecycleToken lifecycleToken;
        convo::isr::FirewallToken firewallToken;

        explicit AudioCallbackRuntimeScope(AudioEngine& owner) noexcept
            : engine(owner)
            , lifecycleToken(owner.lifecycleRuntime_.enterAudioCallback())
            , firewallToken(owner.rtCapabilityFirewall_.enter())
        {
            convo::isr::RTAllocatorFirewall::markRTContext(true);
            (void)convo::fetchAddAtomic(engine.rtLocalState_.audioCallbackActiveCount, uint32_t{1}, std::memory_order_acq_rel);
        }

        ~AudioCallbackRuntimeScope() noexcept
        {
            (void)convo::fetchSubAtomic(engine.rtLocalState_.audioCallbackActiveCount, uint32_t{1}, std::memory_order_acq_rel);
            convo::isr::RTAllocatorFirewall::markRTContext(false);
            engine.rtCapabilityFirewall_.leave(firewallToken);
            engine.lifecycleRuntime_.leaveAudioCallback(lifecycleToken);
        }
    } runtimeScope(*this);

    const juce::ScopedNoDenormals noDenormals;
    const convo::numeric_policy::ScopedThreadRole audioThreadScope(convo::numeric_policy::ThreadRole::AudioRealtime);
    ASSERT_AUDIO_THREAD();
    const int numSamples = buffer.getNumSamples();

    struct CallbackTelemetryScope final
    {
        AudioEngine& engine;
        int samples;
        bool enabled;
        std::int64_t startTicks;
        double tickToUs;

        CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn) noexcept
            : engine(owner)
            , samples(numSamplesIn)
            , enabled(owner.isCliProcessingTelemetryEnabled())
            , startTicks(enabled ? juce::Time::getHighResolutionTicks() : 0)
            , tickToUs(enabled
                           ? (1000000.0 / static_cast<double>(juce::Time::getHighResolutionTicksPerSecond()))
                           : 0.0)
        {
        }

        ~CallbackTelemetryScope() noexcept
        {
            if (enabled)
            {
                const std::int64_t endTicks = juce::Time::getHighResolutionTicks();
                const std::int64_t elapsedTicks = endTicks - startTicks;
                const double processTimeUs = (elapsedTicks > 0) ? (static_cast<double>(elapsedTicks) * tickToUs) : 0.0;
                engine.recordAudioCallbackProcessingStats(samples, processTimeUs);
            }
        }
    } callbackTelemetry(*this, numSamples);

    // 事前サニティチェック (getNextAudioBlock と同様)
    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20;
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        buffer.clear();
        return;
    }

    const convo::RuntimeReaderContext audioCtx{ audioThreadRcuReader, convo::ObserveChannel::Audio };
    auto runtimeReadHandle = makeRuntimeReadHandle(audioCtx);
    const auto& runtimeReadHandleRef = runtimeReadHandle;
    const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandleRef);
    if (runtimeWorld == nullptr)
    {
        buffer.clear();
        return;
    }
    const auto authority = AudioCallbackAuthorityView { makeCrossfadePreparedSnapshotFromWorld(*runtimeWorld) };
    DSPCore* dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandleRef);
    if (dsp == nullptr)
    {
        buffer.clear();
        return;
    }

    // AudioThread入口で、現在のDSPが持つ全てのNUCのガードをチェック（デバッグ時のみ）
        #ifdef NUC_DEBUG_GUARDS
        {
        dsp->convolver.debugCheckNucGuards();
        }
    #endif

    // --- ProcessingStateを現行設計で初期化 ---
    const EngineParameterSnapshot parameterSnapshot = captureAudioThreadParameterSnapshot(runtimeWorld);

    DSPCore::ProcessingState procState = buildAudioThreadProcessingState(dsp, parameterSnapshot);

    // DSPCore 固有の上限チェック (getNextAudioBlock と同様)
    if (numSamples > dsp->maxSamplesPerBlock)
    {
        buffer.clear();
        return;
    }

    // ★ ISR準拠: RuntimeWorld 経由でサンプルレートを取得。
    //   RuntimeBuilder は worldOwner->timing.sampleRateHz を
    //   buildRuntimePublishWorld() 時に DSPCore の sampleRate から設定するため、
    //   dsp->sampleRate と runtimeWorld->timing.sampleRateHz は常に一致する。
    const double engineSampleRate = getRuntimeSampleRateHzFromWorld(runtimeReadHandleRef, 0.0);
    if (engineSampleRate <= 0.0
        || absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
    {
        buffer.clear();
        return;
    }

    // --- クロスフェード開始時: スナップショット取得・RT競合ゼロ設計 ---
    DSPCore* fading = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandleRef);
    const auto& preparedCrossfade = authority.preparedCrossfade;
    bool useDryAsOld = preparedCrossfade.useDryAsOld || preparedCrossfade.firstIrDryCrossfadePending;
    if (fading != nullptr && fading == dsp)
    {
        jassertfalse;
        fading = nullptr;
        useDryAsOld = true;
    }
    if (processCrossfadeDelayGateIfPending(fading,
                                           useDryAsOld,
                                           preparedCrossfade,
                                           [&]()
    {
        auto fadingState = procState;
        fadingState.analyzerEnabled = false;
        fadingState.adaptiveCaptureQueue = nullptr;

        fading->processDouble(buffer,
                      analyzerFifo,
                      nullptr,
                      nullptr,
                      fadingState);
    }))
    {
        return;
    }

    armCrossfadeIfPending(fading != nullptr, useDryAsOld, preparedCrossfade);

    const bool canCrossfade = (fading != nullptr || useDryAsOld)
        && crossfadeRuntime_.getGain().isSmoothing()
        && dspCrossfadeDoubleBuffer.getNumChannels() >= 2
        && dspCrossfadeDoubleBuffer.getNumSamples() >= numSamples;

    if (canCrossfade)
    {
        // --- wrap安全・スナップショット設計 ---
        dspCrossfadeDoubleBuffer.clear(0, 0, numSamples);
        dspCrossfadeDoubleBuffer.clear(1, 0, numSamples);

        auto fadingState = procState;
        fadingState.analyzerEnabled = false;
        fadingState.adaptiveCaptureQueue = nullptr;

        if (useDryAsOld)
        {
            const int outChannels = std::min(2, buffer.getNumChannels());
            if (outChannels > 0)
                juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(0, 0), buffer.getReadPointer(0, 0), numSamples);
            if (outChannels > 1)
                juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(1, 0), buffer.getReadPointer(1, 0), numSamples);
        }
        else
        {
            // EBR: managed by RCUReader
            fading->processDoubleToBuffer(buffer, dspCrossfadeDoubleBuffer, analyzerFifo,
                                          nullptr, nullptr, fadingState);
        }
        dsp->processDouble(buffer,
                   analyzerFifo,
                   &inputLevelLinear,
                   &outputLevelLinear,
                   procState);

        // スナップショット（commitNewDSPでセット済み、ここでは読み取り専用）
        const int outChannels = std::min(2, buffer.getNumChannels());
        double* dstL = (outChannels > 0) ? buffer.getWritePointer(0, 0) : nullptr;
        double* dstR = (outChannels > 1) ? buffer.getWritePointer(1, 0) : nullptr;
        const double* oldL = (outChannels > 0) ? dspCrossfadeDoubleBuffer.getReadPointer(0, 0) : nullptr;
        const double* oldR = (outChannels > 1) ? dspCrossfadeDoubleBuffer.getReadPointer(1, 0) : nullptr;

        runLatencyAlignedCrossfadeMixLoop<double>(dstL,
                                                  dstR,
                                                  oldL,
                                                  oldR,
                                                  numSamples,
                                                                  preparedCrossfade.latencyDelayOld,
                                                                  preparedCrossfade.latencyDelayNew,
                                                                  preparedCrossfade.latencyResetPending,
                                                  [](double* outL,
                                                     double* outR,
                                                     int i,
                                                     double gNew,
                                                     double alignedOldL,
                                                     double alignedOldR,
                                                     double alignedNewL,
                                                     double alignedNewR)
                                                  {
                                                      const double gOld = 1.0 - gNew;
                                                      if (outL != nullptr) outL[i] = alignedNewL * gNew + alignedOldL * gOld;
                                                      if (outR != nullptr) outR[i] = alignedNewR * gNew + alignedOldR * gOld;
                                                  });
        if (!useDryAsOld)
        {
            // EBR: managed by RCUReader
        }

        finalizeCrossfadeMixPath(dsp, fading, false);
    }
    else
    {
        dsp->processDouble(buffer,
                           analyzerFifo,
                           &inputLevelLinear,
                           &outputLevelLinear,
                           procState);

        cleanupCrossfadeDirectPath(dsp, fading);
    }

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ callback 開始 tick（フル callback 時間計測用）
    static constexpr auto kNeverStartedTicks = std::numeric_limits<int64_t>::min();
    int64_t cbStartTicks = kNeverStartedTicks;

    // ★ XRUN 検出（callback 時間 + interval 超過）
    {
        const auto t0_start = juce::Time::getHighResolutionTicks();
        cbStartTicks = t0_start;
        const auto ticksPerSec = juce::Time::getHighResolutionTicksPerSecond();
        const double xrunSampleRate = getRuntimeSampleRateHzFromWorld(runtimeReadHandleRef, 0.0);
        const double expectedMs = (xrunSampleRate > 0.0)
            ? static_cast<double>(numSamples) / engineSampleRate * 1000.0
            : 0.0;

        double intervalMs = 0.0;
        const uint64_t lastEnd = convo::consumeAtomic(rtLocalState_.lastCallbackEndTicks, std::memory_order_relaxed);
        if (lastEnd > 0)
        {
            intervalMs = static_cast<double>(t0_start - lastEnd)
                * 1000.0 / static_cast<double>(ticksPerSec);
        }

        const auto t1_end = juce::Time::getHighResolutionTicks();
        const double callbackMs = static_cast<double>(t1_end - t0_start)
            * 1000.0 / static_cast<double>(ticksPerSec);

        constexpr double kFixedMarginMs = 1.0;
        constexpr double kRatioThreshold = 1.2;
        bool xrunDetected = false;

        if (intervalMs > 0.0 && expectedMs > 0.0)
        {
            const double intervalThreshold = std::max(expectedMs * kRatioThreshold, kFixedMarginMs);
            if (intervalMs > intervalThreshold)
                xrunDetected = true;
        }
        if (!xrunDetected && expectedMs > 0.0)
        {
            const double callbackThreshold = std::max(expectedMs * kRatioThreshold, kFixedMarginMs);
            if (callbackMs > callbackThreshold)
                xrunDetected = true;
        }

        if (xrunDetected)
        {
            XRunEvent ev;
            ev.timestampTicks = t1_end;
            ev.callbackMs = callbackMs;
            ev.intervalMs = intervalMs;
            ev.expectedMs = expectedMs;
            ev.generation = static_cast<int>(runtimeWorld->generation);
            ev.retireQueueDepth = convo::consumeAtomic(retireQueueDepth_, std::memory_order_relaxed);
            ev.sequenceNumber = convo::fetchAddAtomic(rtLocalState_.xrunSequenceCounter,
                uint64_t{1}, std::memory_order_acq_rel) + 1u;

            if (!xRunBuffer.push(ev))
            {
                convo::fetchAddAtomic(rtAuxMutable_.xRunDropCount,
                    uint64_t{1}, std::memory_order_relaxed);
            }
        }

        convo::publishAtomic(rtLocalState_.lastCallbackEndTicks, t1_end, std::memory_order_release);
    }

    // ★ ACTIVATE 検出（RuntimeWorld generation 変化）
    {
        const uint64_t currentGen = (runtimeWorld != nullptr)
            ? static_cast<uint64_t>(runtimeWorld->generation)
            : 0;
        const uint64_t prevActivated = convo::consumeAtomic(
            rtLocalState_.lastActivatedGeneration, std::memory_order_relaxed);

        if (currentGen != prevActivated && currentGen > 0)
        {
            convo::publishAtomic(rtLocalState_.lastActivatedGeneration,
                currentGen, std::memory_order_release);

            XRunEvent ev;
            ev.timestampTicks = juce::Time::getHighResolutionTicks();
            ev.generation = static_cast<int>(currentGen);
            xRunBuffer.push(ev);
        }
    }

    // ★ CBSUMMARY 入力更新（RT-safe: atomic relaxed + compare_exchange_weak）
    {
        const auto ticksPerSec = juce::Time::getHighResolutionTicksPerSecond();
        if (cbStartTicks != kNeverStartedTicks)
        {
            const auto nowTicks = juce::Time::getHighResolutionTicks();
            const auto callbackUs = static_cast<uint32_t>(
                static_cast<double>(nowTicks - cbStartTicks)
                * 1000000.0 / static_cast<double>(ticksPerSec));
            updateAtomicMaximum(callbackMaxUs_, callbackUs);

            const uint64_t prevEnd = convo::consumeAtomic(rtLocalState_.lastCallbackEndTicks, std::memory_order_relaxed);
            if (prevEnd > 0)
            {
                const auto intervalUs = static_cast<uint32_t>(
                    static_cast<double>(cbStartTicks - prevEnd)
                    * 1000000.0 / static_cast<double>(ticksPerSec));
                updateAtomicMaximum(intervalMaxUs_, intervalUs);
            }
        }

        callbackCount_.fetch_add(1u, std::memory_order_relaxed);
    }
#endif
}


