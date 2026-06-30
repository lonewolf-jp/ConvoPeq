#include <JuceHeader.h>
#include "AudioEngine.h"
#include "DiagnosticsConfig.h"
#include "core/RuntimeReaderContext.h"
#include "NoiseShaperLearner.h"
#include "core/RCUReader.h"
#include "core/TimeUtils.h"

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
        uint64_t startUs;

        CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn) noexcept
            : engine(owner)
            , samples(numSamplesIn)
            , enabled(owner.isCliProcessingTelemetryEnabled())
            , startUs(convo::getCurrentTimeUs())  // ★ 常時取得（A/B計測用）
        {
        }

        ~CallbackTelemetryScope() noexcept
        {
            const uint64_t endUs = convo::getCurrentTimeUs();
            const uint64_t processTime = (endUs > startUs) ? (endUs - startUs) : 0;
            if (enabled)
            {
                const double processTimeUs = static_cast<double>(processTime);
                engine.recordAudioCallbackProcessingStats(samples, processTimeUs);
            }
        }
    } callbackTelemetry(*this, numSamples);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ work60: DSP_STAGE計測用タイムスタンプ
    uint64_t t1_dspStartUs = 0;
#endif

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

    // ★ A/G/H: コールバック受信診断
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // A: callback受信時刻 + drift計測
    {
        const uint64_t nowUs = convo::getCurrentTimeUs();
        const uint64_t prevEntryUs = rtLocalState_.lastCallbackEntryUs.load(
            std::memory_order_relaxed);
        rtLocalState_.lastCallbackEntryUs.store(nowUs, std::memory_order_relaxed);
        if (prevEntryUs > 0)
        {
            const double sr = getRuntimeSampleRateHzFromWorld(
                runtimeReadHandleRef, 0.0);
            if (sr > 0.0 && numSamples > 0)
            {
                const double expectedUs =
                    static_cast<double>(numSamples) / sr * 1e6;
                const int64_t driftUs = static_cast<int64_t>(
                    static_cast<double>(static_cast<int64_t>(nowUs - prevEntryUs))
                    - expectedUs);
                rtLocalState_.lastCallbackDriftUs.store(
                    driftUs, std::memory_order_relaxed);
            }
        }
    }
    // G: CPU migration記録
    {
        const uint32_t cpu = static_cast<uint32_t>(::GetCurrentProcessorNumber());
        const uint32_t prev = rtLocalState_.lastCallbackProcessor.load(
            std::memory_order_relaxed);
        if (prev != cpu)
        {
            rtLocalState_.lastCallbackProcessor.store(cpu, std::memory_order_relaxed);
            if (prev != UINT32_MAX)
            {
                convo::fetchAddAtomic(rtLocalState_.cpuMigrationCount,
                    uint64_t{1}, std::memory_order_relaxed);
                const uint64_t cbIdx = rtLocalState_.audioCallbackEpochCounter.load(
                    std::memory_order_relaxed);
                const uint64_t pubSeq = (runtimeWorld != nullptr)
                    ? runtimeWorld->metadata.publicationSequence : 0;
                const uint64_t gen = (runtimeWorld != nullptr)
                    ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
                juce::Logger::writeToLog(
                    "[CPU_MIG] callback=" + juce::String(static_cast<int64_t>(cbIdx))
                    + " seq=" + juce::String(static_cast<int64_t>(pubSeq))
                    + " gen=" + juce::String(static_cast<int64_t>(gen))
                    + " cpu=" + juce::String(static_cast<int>(cpu))
                    + " prev=" + juce::String(static_cast<int>(prev)));
            }
        }
    }
    // H: publicationSequence変化検出
    {
        if (runtimeWorld != nullptr)
        {
            const uint64_t seq = runtimeWorld->metadata.publicationSequence;
            const uint64_t prevSeq = rtLocalState_.lastCallbackPublicationSeq.load(
                std::memory_order_relaxed);
            if (seq != prevSeq)
            {
                rtLocalState_.lastCallbackPublicationSeq.store(
                    seq, std::memory_order_relaxed);
                if (prevSeq > 0)
                {
                    const uint64_t cbIdx = rtLocalState_.audioCallbackEpochCounter.load(
                        std::memory_order_relaxed);
                    juce::Logger::writeToLog(
                        "[CB_SEQ] callback=" + juce::String(static_cast<int64_t>(cbIdx))
                        + " gen=" + juce::String(static_cast<int64_t>(runtimeWorld->generation))
                        + " seq=" + juce::String(static_cast<int64_t>(seq))
                        + " prevSeq=" + juce::String(static_cast<int64_t>(prevSeq)));
                }
            }
        }
    }
#endif

    const auto authority = AudioCallbackAuthorityView { makeCrossfadePreparedSnapshotFromWorld(*runtimeWorld) };

    // ★ callback epoch counter（全診断ブロックで使用するcallbackIndexの基盤）
    (void)convo::fetchAddAtomic(rtLocalState_.audioCallbackEpochCounter, uint64_t{1}, std::memory_order_acq_rel);

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

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // work60: dsp->process()直前にcallbackSeq/cpuをDSPCoreとConvolverProcessorへ伝達
    dsp->currentCallbackSeq = convo::consumeAtomic(rtLocalState_.audioCallbackEpochCounter, std::memory_order_relaxed);
    dsp->currentCpu = convo::consumeAtomic(rtLocalState_.lastCallbackProcessor, std::memory_order_relaxed);
    dsp->convolver.currentCallbackSeq.store(dsp->currentCallbackSeq, std::memory_order_relaxed);
    dsp->convolver.currentCpu.store(dsp->currentCpu, std::memory_order_relaxed);
    // work60: DSP_STAGE開始時刻
    t1_dspStartUs = convo::getCurrentTimeUs();
#endif

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
    // ★ work60: DSP_STAGE終了時刻（t2）
    const uint64_t t2_dspEndUs = convo::getCurrentTimeUs();

    // ★ callbackIndex（callback開始直後に1度だけ取得、全診断ブロックで同じ値を使用）
    const auto thisCallbackIndex = convo::consumeAtomic(
        rtLocalState_.audioCallbackEpochCounter, std::memory_order_relaxed);

    // ★ DSP Observe 検出: publicationSequence 変化時のみ記録 + ObserveReason
    static uint64_t s_lastObservedSeq = 0;
    uint64_t observeUs = 0;
    const char* observeReason = "";
    uint64_t dspSeq = 0;
    if (runtimeWorld != nullptr) {
        dspSeq = runtimeWorld->metadata.publicationSequence;
        if (dspSeq > 0 && dspSeq != s_lastObservedSeq) {
            if (dspSeq > s_lastObservedSeq)
                observeReason = "Forward";
            else if (dspSeq < s_lastObservedSeq)
                observeReason = "Rollback";
            else
                observeReason = "Replay";
            s_lastObservedSeq = dspSeq;
            observeUs = convo::getCurrentTimeUs();
        }
    }

    // ★ PublishTimingHistory lookup（DSP observe 時に publishEndUs を取得）
    uint64_t matchedPublishEndUs = 0;
    uint64_t matchedPublishCallbackIdx = 0;
    if (observeUs > 0 && dspSeq > 0) {
        const uint64_t wc = convo::consumeAtomic(
            rtLocalState_.publishTimingWriteCount, std::memory_order_acquire);
        const uint64_t startSlot = (wc > 0) ? (wc - 1) % RTLocalState::kPublishTimingSlots : 0;
        for (size_t i = 0; i < RTLocalState::kPublishTimingSlots; ++i) {
            const uint64_t idx = (startSlot >= i)
                ? (startSlot - i)
                : (RTLocalState::kPublishTimingSlots + startSlot - i);
            const auto& entry = rtLocalState_.publishTimingHistory[idx];
            if (entry.sequence == dspSeq && entry.sequence != 0) {
                matchedPublishEndUs = entry.publishEndUs;
                matchedPublishCallbackIdx = entry.publishCallbackIndex;
                break;
            }
        }
    }

    // ★ callback 開始時刻（フル callback 時間計測用）
    static constexpr auto kNeverStartedUs = std::numeric_limits<uint64_t>::max();
    uint64_t cbStartUs = kNeverStartedUs;
    uint64_t cbPrevEndUs = 0;  // XRUNブロックが上書きする前の前回終了時刻を保存

    // ★ XRUN 検出（callback 時間 + interval 超過）
    {
        const auto t0_start = convo::getCurrentTimeUs();
        cbStartUs = t0_start;
        cbPrevEndUs = convo::consumeAtomic(rtLocalState_.lastCallbackEndTicks, std::memory_order_relaxed);
        const double xrunSampleRate = getRuntimeSampleRateHzFromWorld(runtimeReadHandleRef, 0.0);
        const double expectedMs = (xrunSampleRate > 0.0)
            ? static_cast<double>(numSamples) / xrunSampleRate * 1000.0
            : 0.0;

        double intervalMs = 0.0;
        if (cbPrevEndUs > 0)
        {
            intervalMs = static_cast<double>(t0_start - cbPrevEndUs) / 1000.0;
        }

        const auto t1_end = convo::getCurrentTimeUs();
        const double callbackMs = static_cast<double>(t1_end - t0_start) / 1000.0;

        constexpr double kFixedMarginMs = 3.0;
        constexpr double kRatioThreshold = 1.5;
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
            ev.driftUs = rtLocalState_.lastCallbackDriftUs.load(
                std::memory_order_relaxed);

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
            ev.timestampTicks = convo::getCurrentTimeUs();
            ev.generation = static_cast<int>(currentGen);
            xRunBuffer.push(ev);
        }
    }

    // ★ CBSUMMARY 入力更新（RT-safe: atomic relaxed + compare_exchange_weak）
    {
        if (cbStartUs != kNeverStartedUs)
        {
            const auto nowUs = convo::getCurrentTimeUs();
            const auto callbackUs = static_cast<uint32_t>(nowUs - cbStartUs);
            updateAtomicMaximum(callbackMaxUs_, callbackUs);

            if (cbPrevEndUs > 0)
            {
                const auto intervalUs = static_cast<uint32_t>(cbStartUs - cbPrevEndUs);
                updateAtomicMaximum(intervalMaxUs_, intervalUs);
            }
        }

        callbackCount_.fetch_add(1u, std::memory_order_relaxed);
    }

    // ★ DSP_TIMING 出力（Observe検出時のみ）
    if (observeUs > 0 && dspSeq > 0)
    {
        juce::String log("[DSP_TIMING] seq=");
        log += juce::String(static_cast<juce::int64>(dspSeq));
        log += " gen=" + juce::String(static_cast<juce::int64>(runtimeWorld ? static_cast<uint64_t>(runtimeWorld->generation) : 0));
        log += " worldId=" + juce::String(static_cast<juce::int64>(runtimeWorld ? static_cast<uint64_t>(runtimeWorld->worldId) : 0));
        log += " reason=" + juce::String(observeReason);
        log += " callbackIndex=" + juce::String(static_cast<juce::int64>(thisCallbackIndex));
        if (matchedPublishEndUs > 0)
        {
            const uint64_t observeLatencyUs = observeUs - matchedPublishEndUs;
            log += " observeLatencyUs=" + juce::String(static_cast<juce::int64>(observeLatencyUs));
            log += " pubToObserveUs=" + juce::String(static_cast<juce::int64>(observeLatencyUs));
            if (matchedPublishCallbackIdx > 0 && matchedPublishCallbackIdx < thisCallbackIndex)
            {
                const uint64_t callbacksUntilObserve = thisCallbackIndex - matchedPublishCallbackIdx;
                log += " callbacksUntilObserve=" + juce::String(static_cast<juce::int64>(callbacksUntilObserve));
                log += " publishCallbackIdx=" + juce::String(static_cast<juce::int64>(matchedPublishCallbackIdx));
            }
        }
        DBG(log);
        juce::Logger::writeToLog(log);
    }

    // ★ work60: CALLBACK_STAGE（INPUT/DSP/OUTPUT 3区間 + drift + budget）
    {
        const uint64_t t0 = callbackTelemetry.startUs;
        const uint64_t t3 = convo::getCurrentTimeUs();
        const uint64_t gen = (runtimeWorld != nullptr)
            ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
        const uint32_t cpu = static_cast<uint32_t>(::GetCurrentProcessorNumber());
        const uint64_t expectedUs = rtLocalState_.expectedCallbackIntervalUs;
        const int64_t driftUs = convo::consumeAtomic(
            rtLocalState_.lastCallbackDriftUs, std::memory_order_relaxed);

        if ((thisCallbackIndex & CONVOPEQ_DIAG_SAMPLE_MASK) == 0)
        {
            const uint64_t inputUs = (t1_dspStartUs > t0) ? t1_dspStartUs - t0 : 0;
            const uint64_t dspUs = (t2_dspEndUs > t1_dspStartUs) ? t2_dspEndUs - t1_dspStartUs : 0;
            const uint64_t outputUs = (t3 > t2_dspEndUs) ? t3 - t2_dspEndUs : 0;
            const uint64_t totalUs = (t3 > t0) ? t3 - t0 : 0;
            const uint32_t budgetPermille = (expectedUs > 0)
                ? static_cast<uint32_t>(totalUs * 1000 / expectedUs) : 0;
            juce::String cclog("[CALLBACK_STAGE]");
            cclog += " seq=" + juce::String(static_cast<int64_t>(thisCallbackIndex));
            cclog += " cpu=" + juce::String(static_cast<int>(cpu));
            cclog += " gen=" + juce::String(static_cast<int64_t>(gen));
            cclog += " expected=" + juce::String(static_cast<int64_t>(expectedUs));
            cclog += " drift=" + juce::String(static_cast<int64_t>(driftUs));
            cclog += " input=" + juce::String(static_cast<int64_t>(inputUs));
            cclog += " dsp=" + juce::String(static_cast<int64_t>(dspUs));
            cclog += " output=" + juce::String(static_cast<int64_t>(outputUs));
            cclog += " total=" + juce::String(static_cast<int64_t>(totalUs));
            cclog += " budget=" + juce::String(static_cast<int>(budgetPermille / 10)) + "."
                + juce::String(static_cast<int>(budgetPermille % 10)) + "%";
            DBG(cclog);
            juce::Logger::writeToLog(cclog);
        }
    }

    // ★ B: CallbackTimingHistory リングバッファ書込
    {
        const uint64_t endUs = convo::getCurrentTimeUs();
        const uint64_t processTime = callbackTelemetry.startUs > 0
            ? (endUs > callbackTelemetry.startUs ? endUs - callbackTelemetry.startUs : 0)
            : 0;
        const uint64_t wc = convo::fetchAddAtomic(
            rtLocalState_.callbackTimingWriteCount,
            uint64_t{1}, std::memory_order_relaxed);
        const size_t idx = static_cast<size_t>(
            (wc - 1) % RTLocalState::kCallbackTimingSlots);
        auto& entry = rtLocalState_.callbackTimingHistory[idx];
        entry.callbackIndex = thisCallbackIndex;
        entry.processTimeUs = processTime;
        entry.driftUs = rtLocalState_.lastCallbackDriftUs.load(
            std::memory_order_relaxed);
        entry.cpu = rtLocalState_.lastCallbackProcessor.load(
            std::memory_order_relaxed);
        // expectedIntervalUs を runtimeWorld から計算
        {
            const double sr = getRuntimeSampleRateHzFromWorld(
                runtimeReadHandleRef, 0.0);
            if (sr > 0.0 && numSamples > 0)
            {
                const double expectedUs =
                    static_cast<double>(numSamples) / sr * 1e6;
                const double pct = (static_cast<double>(processTime) / expectedUs) * 100.0;
                entry.budgetPermille = static_cast<uint16_t>(
                    std::min(999.0, std::round(pct * 10.0)));
                entry.expectedIntervalUs = static_cast<uint32_t>(expectedUs);
            }
        }
        entry.sequence.store(thisCallbackIndex, std::memory_order_release);
    }
#endif
}


