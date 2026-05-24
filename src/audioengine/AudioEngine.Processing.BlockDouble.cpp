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

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_BLOCK_DOUBLE)
void AudioEngine::processBlockDouble (juce::AudioBuffer<double>& buffer)
{
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
            (void)convo::fetchAddAtomic(engine.audioCallbackActiveCount_, uint32_t{1}, std::memory_order_acq_rel);
        }

        ~AudioCallbackRuntimeScope() noexcept
        {
            (void)convo::fetchSubAtomic(engine.audioCallbackActiveCount_, uint32_t{1}, std::memory_order_acq_rel);
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
        std::int64_t started;
        bool enabled;

        CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn) noexcept
            : engine(owner)
            , samples(numSamplesIn)
            , started(0)
            , enabled(owner.isCliProcessingTelemetryEnabled())
        {
            if (enabled)
                started = juce::Time::getHighResolutionTicks();
        }

        ~CallbackTelemetryScope() noexcept
        {
            if (enabled)
                engine.recordAudioCallbackProcessingTimeUs(samples, started);
        }
    } callbackTelemetry(*this, numSamples);

    // 事前サニティチェック (getNextAudioBlock と同様)
    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20;
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        buffer.clear();
        return;
    }

    const auto runtimePublishView = getRuntimePublishView();
    const auto* runtimeGraph = runtimePublishView.graph;
    DSPCore* dsp = (runtimeGraph != nullptr && runtimeGraph->runtimeUuid != 0)
        ? static_cast<DSPCore*>(runtimeGraph->activeNode)
        : nullptr;
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
    const auto observedSnapshot = m_coordinator.observeCurrentRuntime(kAudioEpochReaderIndex);
    const convo::GlobalSnapshot* snap = observedSnapshot.get();
    const EngineParameterSnapshot parameterSnapshot = captureAudioThreadParameterSnapshot(snap);

    DSPCore::ProcessingState procState = buildAudioThreadProcessingState(dsp, parameterSnapshot);

    // DSPCore 固有の上限チェック (getNextAudioBlock と同様)
    if (numSamples > dsp->maxSamplesPerBlock)
    {
        buffer.clear();
        return;
    }

    float snapshotAlpha = 1.0f;
    const convo::GlobalSnapshot* snapshotFrom = nullptr;
    const convo::GlobalSnapshot* snapshotTo = nullptr;
    updateAudioThreadSnapshotFade(numSamples, snapshotAlpha, snapshotFrom, snapshotTo);

    const double engineSampleRate = (runtimeGraph != nullptr) ? runtimeGraph->sampleRate : 0.0;
    if (engineSampleRate <= 0.0
        || absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
    {
        buffer.clear();
        return;
    }

    // --- クロスフェード開始時: スナップショット取得・RT競合ゼロ設計 ---
    DSPCore* fading = resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph);
    bool useDryAsOld = (runtimeGraph != nullptr) ? runtimeGraph->dspCrossfadeUseDryAsOld : false;
    if (fading != nullptr && fading == dsp)
    {
        jassertfalse;
        fading = nullptr;
        useDryAsOld = true;
    }
    const bool hasPendingCrossfade = (runtimeGraph != nullptr) ? runtimeGraph->dspCrossfadePending : false;
    if (processCrossfadeDelayGateIfPending(fading,
                                           useDryAsOld,
                                           hasPendingCrossfade,
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

    armCrossfadeIfPending(fading != nullptr, useDryAsOld, runtimeGraph);

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
                                                                  runtimeGraph,
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
        const bool shouldBlendSnapshots = (snapshotFrom != nullptr)
            && (snapshotTo != nullptr)
            && (absDiffNoLibm(static_cast<double>(snapshotAlpha), 1.0) > 1.0e-6)
            && (dspCrossfadeDoubleBuffer.getNumChannels() >= 2)
            && (dspCrossfadeDoubleBuffer.getNumSamples() >= numSamples);

        if (shouldBlendSnapshots)
        {
            const EngineParameterSnapshot fromParameterSnapshot = captureAudioThreadParameterSnapshot(snapshotFrom);
            const EngineParameterSnapshot toParameterSnapshot = captureAudioThreadParameterSnapshot(snapshotTo);

            auto fromState = buildAudioThreadProcessingState(dsp, fromParameterSnapshot);
            auto toState = buildAudioThreadProcessingState(dsp, toParameterSnapshot);

            fromState.analyzerEnabled = false;
            fromState.adaptiveCaptureQueue = nullptr;

            dsp->processDoubleToBuffer(buffer,
                                       dspCrossfadeDoubleBuffer,
                                       analyzerFifo,
                                       nullptr,
                                       nullptr,
                                       fromState);

            dsp->processDouble(buffer,
                               analyzerFifo,
                               &inputLevelLinear,
                               &outputLevelLinear,
                               toState);

            const double alpha = juce::jlimit(0.0, 1.0, static_cast<double>(snapshotAlpha));
            const double invAlpha = 1.0 - alpha;
            const int outChannels = std::min(2, buffer.getNumChannels());
            for (int ch = 0; ch < outChannels; ++ch)
            {
                double* dst = buffer.getWritePointer(ch, 0);
                const double* srcFrom = dspCrossfadeDoubleBuffer.getReadPointer(ch, 0);
                for (int i = 0; i < numSamples; ++i)
                    dst[i] = dst[i] * alpha + srcFrom[i] * invAlpha;
            }
        }
        else
        {
            dsp->processDouble(buffer,
                               analyzerFifo,
                               &inputLevelLinear,
                               &outputLevelLinear,
                               procState);
        }

        cleanupCrossfadeDirectPath(dsp, fading);
    }
}

#endif

