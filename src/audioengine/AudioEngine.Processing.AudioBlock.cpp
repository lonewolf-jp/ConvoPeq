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

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_AUDIO_BLOCK)

void AudioEngine::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    if (isShutdownInProgress())
    {
        shutdownRuntime_.markLateCallback();
        bufferToFill.clearActiveBufferRegion();
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
    // 入力検証 (Input Validation)
    if (bufferToFill.buffer == nullptr)
    {
        return;
    }

    const int numSamples = bufferToFill.numSamples;
    const int startSample = bufferToFill.startSample;
    auto* buffer = bufferToFill.buffer;

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

    // 事前サニティチェック: 絶対的な上限 (1<<20 ≒ 100万サンプル) で明らかな破損データを弾く。
    // DSPCore の maxSamplesPerBlock は prepareToPlay() でホスト指定値を反映して設定されるため、
    // ここで SAFE_MAX_BLOCK_SIZE (65536) を使うと、131072 等の正当なブロックを誤って拒否する。
    // 【Bug Fix】SAFE_MAX_BLOCK_SIZE による早期リジェクトを廃止し、dsp->maxSamplesPerBlock で
    //            正確なチェックを行う (下記 DSPCore 取得後)。
    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20; // 破損データ検出用上限
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // startSampleの妥当性チェック
    if (startSample < 0 || startSample + numSamples > buffer->getNumSamples())
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // Epoch tracking for lock-free Audio Thread safety
    convo::RCUReaderGuard rcuGuard(audioThreadRcuReader);

    const auto runtimePublishView = getRuntimePublishView();
    const auto* runtimeGraph = runtimePublishView.graph;

    const auto packHandle = [](convo::isr::DSPHandle handle) noexcept -> std::uint64_t
    {
        return (static_cast<std::uint64_t>(handle.generation) << 32)
            | static_cast<std::uint64_t>(handle.slot);
    };

    const auto activeHandle = dspHandleRuntime_.getActiveDSP();
    const auto fadingHandle = dspHandleRuntime_.getFadingDSP();
    const auto resolvedActive = dspHandleRuntime_.resolve(activeHandle);
    const auto resolvedFading = dspHandleRuntime_.resolve(fadingHandle);
    const auto callbackEpoch = convo::fetchAddAtomic(audioCallbackEpochCounter_, uint64_t{1}, std::memory_order_acq_rel) + 1u;
    const auto sampleCursor = convo::fetchAddAtomic(audioSampleCursorCounter_, static_cast<uint64_t>(numSamples), std::memory_order_acq_rel);
    const auto graphRevision = consumeAtomic(runtimeGraphRevision, std::memory_order_acquire);

    const auto rtFrame = convo::isr::makeRTExecutionFrame(
        packHandle(activeHandle),
        packHandle(fadingHandle),
        currentFade_,
        nullptr,
        0,
        sampleCursor,
        callbackEpoch,
        runtimeScope.lifecycleToken.epochId,
        graphRevision,
        &rtTraceRelay_);

    rtTraceRelay_.enqueue({ rtFrame.sampleCursor, 0xA001u, static_cast<std::uint32_t>(numSamples) });

    DSPCore* dsp = resolvedActive.valid
        ? static_cast<DSPCore*>(resolvedActive.instance)
        : ((runtimeGraph != nullptr && runtimeGraph->runtimeUuid != 0)
            ? static_cast<DSPCore*>(runtimeGraph->activeNode)
            : nullptr);
    if (dsp == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    if (dsp != nullptr)
    {
        // DSPCore 固有の上限チェック
        // DSPCore::prepare() でホスト指定の samplesPerBlock を反映した maxSamplesPerBlock が設定される。
        // dsp は RCU で公開済みのため maxSamplesPerBlock は Audio Thread から安全に読み出せる。
        if (numSamples > dsp->maxSamplesPerBlock)
        {
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        // 安全対策: サンプルレート不整合チェック
        // DSPのサンプルレートとエンジンの現在のサンプルレートが一致しない場合、
        // レート変更処理中とみなし、グリッチを防ぐために無音を出力する。
        const double engineSampleRate = (runtimeGraph != nullptr) ? runtimeGraph->sampleRate : 0.0;
        if (engineSampleRate <= 0.0
            || absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
        {
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        // パラメータのロード
        // 【Parameter安全設計】
        // Audio ThreadではAtomic変数の読み取りのみを行い、ロックやメモリ確保を伴う処理は行わない。
        // 構造変更が必要な場合は、別途フラグやUIスレッド経由で再構築を行う。
        // ── Audio Thread 最適化: GlobalSnapshot を優先し、fallback で atomics を読む ──
        const auto observedSnapshot = m_coordinator.observeCurrentRuntime(kAudioEpochReaderIndex);
        const convo::GlobalSnapshot* snap = observedSnapshot.get();
        const EngineParameterSnapshot parameterSnapshot = captureAudioThreadParameterSnapshot(snap);

        DSPCore::ProcessingState procState = buildAudioThreadProcessingState(dsp, parameterSnapshot);

        float snapshotAlpha = 1.0f;
        const convo::GlobalSnapshot* snapshotFrom = nullptr;
        const convo::GlobalSnapshot* snapshotTo = nullptr;
        const bool updateFadeReturned = updateAudioThreadSnapshotFade(numSamples,
                                                                      snapshotAlpha,
                                                                      snapshotFrom,
                                                                      snapshotTo);

        const bool snapshotFading = updateFadeReturned
            && snapshotTo != nullptr;

        if (snapshotFading)
        {
            const int fadeChannels = std::min(dspCrossfadeFloatBuffer.getNumChannels(), buffer->getNumChannels());
            for (int ch = 0; ch < fadeChannels; ++ch)
                dspCrossfadeFloatBuffer.clear(ch, 0, numSamples);

            juce::AudioSourceChannelInfo oldInfo(&dspCrossfadeFloatBuffer, 0, numSamples);
            processWithSnapshot(oldInfo, snapshotFrom, true);
            processWithSnapshot(bufferToFill, snapshotTo, false);

            const float gNew = snapshotAlpha;
            const float gOld = 1.0f - snapshotAlpha;
            const int outChannels = std::min(buffer->getNumChannels(), dspCrossfadeFloatBuffer.getNumChannels());
            float* dstL = (outChannels > 0) ? buffer->getWritePointer(0, startSample) : nullptr;
            float* dstR = (outChannels > 1) ? buffer->getWritePointer(1, startSample) : nullptr;
            const float* oldL = (outChannels > 0) ? dspCrossfadeFloatBuffer.getReadPointer(0, 0) : nullptr;
            const float* oldR = (outChannels > 1) ? dspCrossfadeFloatBuffer.getReadPointer(1, 0) : nullptr;

            for (int i = 0; i < numSamples; ++i)
            {
                if (dstL != nullptr)
                    dstL[i] = dstL[i] * gNew + oldL[i] * gOld;
                if (dstR != nullptr)
                    dstR[i] = dstR[i] * gNew + oldR[i] * gOld;
            }

            return;
        }

        DSPCore* fading = resolvedFading.valid
            ? static_cast<DSPCore*>(resolvedFading.instance)
            : resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph);
        bool useDryAsOld = (runtimeGraph != nullptr) ? runtimeGraph->dspCrossfadeUseDryAsOld : false;
        const bool hasPendingCrossfade = (runtimeGraph != nullptr) ? runtimeGraph->dspCrossfadePending : false;
        if (processCrossfadeDelayGateIfPending(fading,
                                               useDryAsOld,
                                               hasPendingCrossfade,
                                               [&]()
        {
            auto fadingState = procState;
            fadingState.analyzerEnabled = false;
            fadingState.adaptiveCaptureQueue = nullptr;

            fading->process(bufferToFill,
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
            && dspCrossfadeFloatBuffer.getNumChannels() >= 2
            && dspCrossfadeFloatBuffer.getNumSamples() >= numSamples;

        if (canCrossfade)
        {
            juce::AudioSourceChannelInfo fadeInfo(&dspCrossfadeFloatBuffer, 0, numSamples);
            dspCrossfadeFloatBuffer.clear(0, 0, numSamples);
            dspCrossfadeFloatBuffer.clear(1, 0, numSamples);

            auto fadingState = procState;
            fadingState.analyzerEnabled = false;
            fadingState.adaptiveCaptureQueue = nullptr;

            if (useDryAsOld)
            {
                const int outChannels = std::min(2, buffer->getNumChannels());
                if (outChannels > 0)
                    juce::FloatVectorOperations::copy(dspCrossfadeFloatBuffer.getWritePointer(0, 0), buffer->getReadPointer(0, startSample), numSamples);
                if (outChannels > 1)
                    juce::FloatVectorOperations::copy(dspCrossfadeFloatBuffer.getWritePointer(1, 0), buffer->getReadPointer(1, startSample), numSamples);
            }
            else
            {
                // EBR: lifetime managed by RCUReader
                fading->processToBuffer(bufferToFill, dspCrossfadeFloatBuffer, analyzerFifo,
                                       nullptr, nullptr, fadingState);
            }
            dsp->process(bufferToFill,
                         analyzerFifo,
                         &inputLevelLinear,
                         &outputLevelLinear,
                         procState);

            const int outChannels = std::min(2, buffer->getNumChannels());
            float* dstL = (outChannels > 0) ? buffer->getWritePointer(0, startSample) : nullptr;
            float* dstR = (outChannels > 1) ? buffer->getWritePointer(1, startSample) : nullptr;
            const float* oldL = (outChannels > 0) ? dspCrossfadeFloatBuffer.getReadPointer(0, 0) : nullptr;
            const float* oldR = (outChannels > 1) ? dspCrossfadeFloatBuffer.getReadPointer(1, 0) : nullptr;

            runLatencyAlignedCrossfadeMixLoop<float>(dstL,
                                                     dstR,
                                                     oldL,
                                                     oldR,
                                                     numSamples,
                                                     runtimeGraph,
                                                     [this, useDryAsOld](float* outL,
                                                                         float* outR,
                                                                         int i,
                                                                         double gNew,
                                                                         double alignedOldL,
                                                                         double alignedOldR,
                                                                         double alignedNewL,
                                                                         double alignedNewR)
                                                     {
                                                         const double dryScale = useDryAsOld ? dspCrossfadeDryScaleGain.getNextValue() : 1.0;
                                                         const double gOld = 1.0 - gNew;
                                                         const double dryScaledL = alignedOldL * dryScale;
                                                         const double dryScaledR = alignedOldR * dryScale;
                                                         if (outL != nullptr)
                                                             outL[i] = static_cast<float>(alignedNewL * gNew + dryScaledL * gOld);
                                                         if (outR != nullptr)
                                                             outR[i] = static_cast<float>(alignedNewR * gNew + dryScaledR * gOld);
                                                     });

            if (!useDryAsOld)
            {
                // EBR: fading lifetime managed by RCUReaderGuard
            }

            finalizeCrossfadeMixPath(dsp, fading, true);
        }
        else
        {
            // 通常パス（クロスフェードなし）：RCU で dsp の生存が保証されるため addRef/release 不要
            dsp->process(bufferToFill,
                         analyzerFifo,
                         &inputLevelLinear,
                         &outputLevelLinear,
                         procState);
            cleanupCrossfadeDirectPath(dsp, fading);
        }
    }
}

#endif
