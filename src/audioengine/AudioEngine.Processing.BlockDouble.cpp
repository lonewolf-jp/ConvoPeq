#include <JuceHeader.h>
#include "AudioEngine.h"
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
    const convo::EpochDomainReaderGuard epochReaderGuard(m_epochDomain, kAudioEpochReaderIndex);
    ASSERT_AUDIO_THREAD();
    // ★ 追加: RCU ガードで現在の DSP を保護する
    convo::RCUReaderGuard rcuGuard(audioThreadRcuReader);
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

    auto runtimeReadView = readAudioRuntimeView();
    const auto& runtimeReadViewRef = runtimeReadView;
    const auto* runtimeWorld = runtimeReadViewRef.runtimeWorld;
    if (runtimeWorld == nullptr)
    {
        buffer.clear();
        return;
    }
    const auto authority = AudioCallbackAuthorityView { makeCrossfadePreparedSnapshotFromWorld(*runtimeWorld) };
    const auto& runtimePublishView = runtimeReadViewRef.runtimePublish;
    DSPCore* dsp = static_cast<DSPCore*>(runtimePublishView.transition.current);
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

    const double engineSampleRate = runtimePublishView.sampleRateHz;
    if (engineSampleRate <= 0.0
        || absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
    {
        buffer.clear();
        return;
    }

    // --- クロスフェード開始時: スナップショット取得・RT競合ゼロ設計 ---
    DSPCore* fading = runtimePublishView.transition.active
        ? static_cast<DSPCore*>(runtimePublishView.transition.next)
        : nullptr;
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
        && dspCrossfadeGain.isSmoothing()
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
}


