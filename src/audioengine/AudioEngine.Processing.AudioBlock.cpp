#include <JuceHeader.h>
#include "AudioEngine.h"
#include "DiagnosticsConfig.h"
#include "NoiseShaperLearner.h"
#include "core/TimeUtils.h"

// MMCSS applyMmcssPriority() は常に呼ばれる（診断有効時のみログ出力）
#include <windows.h>
#include <avrt.h>

namespace
{
    inline double absDiffNoLibm(double a, double b) noexcept
    {
        return absNoLibm(a - b);
    }
}

void AudioEngine::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    const auto lifecycle = convo::consumeAtomic(lifecycleState, std::memory_order_acquire);
    if (lifecycle != EngineLifecycleState::Prepared)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    if (isShutdownInProgress())
    {
        shutdownRuntime_.markLateCallback();
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // ★ [work62] callbackIndex を早期に取得（RuntimeScopeより前）
    //   fetchAddAtomic は pre-increment 値を返すため +1 して post-increment に。
    [[maybe_unused]] const auto thisCallbackIndex = convo::fetchAddAtomic(
        rtLocalState_.audioCallbackEpochCounter, uint64_t{1}, std::memory_order_acq_rel) + 1u;

    // ★ [work62] MMCSS: Audio スレッド初回コールで優先度設定（RT-safe: atomic flag）
    //    常に適用（診断有効時のみログ出力。無効時はスタブ）
    //    mmcssApplied_ は prepareToPlay() でリセットされるため、
    //    デバイス再初期化後も正しく再適用される。
    {
        bool expected = false;
        if (convo::compareExchangeAtomic(mmcssApplied_, expected, true, std::memory_order_acq_rel)) {
            applyMmcssPriority();
        }
    }

    const int numSamples = bufferToFill.numSamples;

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
    // 入力検証 (Input Validation)
    if (bufferToFill.buffer == nullptr)
    {
        return;
    }

    const int startSample = bufferToFill.startSample;
    auto* buffer = bufferToFill.buffer;

    // ★ [work66-P2-4] 共通開始時刻（関数先頭で1回のみ取得。XRUN/t0_start と共有）
    const auto cbStartUs = convo::getCurrentTimeUs();

    struct CallbackTelemetryScope final
    {
        AudioEngine& engine;
        int samples;
        bool enabled;
        uint64_t startUs;

        CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn,
                                uint64_t cbStartUs) noexcept
            : engine(owner)
            , samples(numSamplesIn)
            , enabled(owner.isCliProcessingTelemetryEnabled())
            , startUs(enabled ? cbStartUs : 0)
        {
        }

        ~CallbackTelemetryScope() noexcept
        {
            if (!enabled)
                return;

            const uint64_t endUs = convo::getCurrentTimeUs();
            const uint64_t processTime = (endUs > startUs) ? (endUs - startUs) : 0;

            const double processTimeUs = static_cast<double>(processTime);
            engine.recordAudioCallbackProcessingStats(samples, processTimeUs);
        }
    } callbackTelemetry(*this, numSamples, cbStartUs);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ work60: DSP_STAGE計測用タイムスタンプ。dsp->process()前後で設定。
    uint64_t t1_dspStartUs = 0;
#endif

    // 事前サニティチェック: 絶対的な上限 (1<<20 ≒ 100万サンプル) で明らかな破損データを弾く。
    // DSPCore::prepare() でホスト指定のブロックサイズが maxSamplesPerBlock に反映されるため、
    // ここでは固定の SAFE_MAX_BLOCK_SIZE (65536) を使わず、取得済み DSPCore の値で最終判定する。
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

    // P0-2: 読取入口を単一の callback authority view へ収束（R21 gate: readAudioRuntimeView）
    auto runtimeReadHandle = readAudioRuntimeView();
    const auto& runtimeReadHandleRef = runtimeReadHandle;
    const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandleRef);
    if (runtimeWorld == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // ★ work61: CallbackArrival 最初の記録（RuntimeScope/ISR firewall後、A/G/H/Xrun/Stageより前）
    //   ISR準拠: getRuntimeSampleRateHzFromWorld で sampleRate を取得。
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    {
        const double sr = getRuntimeSampleRateHzFromWorld(runtimeReadHandleRef, 0.0);
        recordCallbackArrival(
            diagBuffer, thisCallbackIndex, numSamples, sr,
            rtLocalState_.cachedThreadId,
            rtAuxMutable_.cbArrivalWritten,
            rtAuxMutable_.cbArrivalDropped,
            rtAuxMutable_.cbArrivalConsecutiveDrops);
    }
#endif

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
                const uint64_t pubSeq = runtimeWorld->metadata.publicationSequence;
                const uint64_t gen = static_cast<uint64_t>(runtimeWorld->generation);
                // work60: Numeric-Only DiagEvent
                DiagEvent event{};
                event.category = DiagCategory::CpuMig;
                event.eventIndex = cbIdx;
                event.data.cpuMig.pubSeq = pubSeq;
                event.data.cpuMig.generation = gen;
                event.data.cpuMig.cpu = cpu;
                event.data.cpuMig.prevCpu = prev;
                if (diagBuffer.push(event))
                {
                    rtAuxMutable_.diagTickPushed.value.fetch_add(1, std::memory_order_relaxed);
                    rtAuxMutable_.diagTotalPushed.fetch_add(1, std::memory_order_relaxed);
                }
                else
                {
                    rtAuxMutable_.diagTickDropped.value.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    // H: publicationSequence変化検出
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
                const uint64_t gen = static_cast<uint64_t>(runtimeWorld->generation);
                // work60: Numeric-Only DiagEvent
                DiagEvent event{};
                event.category = DiagCategory::CallbackSequence;
                event.eventIndex = cbIdx;
                event.data.callbackSequence.generation = gen;
                event.data.callbackSequence.seq = seq;
                event.data.callbackSequence.prevSeq = prevSeq;
                if (diagBuffer.push(event))
                {
                    rtAuxMutable_.diagTickPushed.value.fetch_add(1, std::memory_order_relaxed);
                    rtAuxMutable_.diagTotalPushed.fetch_add(1, std::memory_order_relaxed);
                }
                else
                {
                    rtAuxMutable_.diagTickDropped.value.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
#endif

    const auto authority = AudioCallbackAuthorityView { makeCrossfadePreparedSnapshotFromWorld(*runtimeWorld) };

    const auto callbackEpoch = thisCallbackIndex;  // work61: 早期取得版を流用
    const auto sampleCursor = convo::fetchAddAtomic(rtLocalState_.audioSampleCursorCounter, static_cast<uint64_t>(numSamples), std::memory_order_acq_rel);
    const auto packedActiveHandle = static_cast<std::uint64_t>(
        reinterpret_cast<uintptr_t>(resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandleRef)));

    const auto rtFrame = convo::isr::makeRTExecutionFrame(
        packedActiveHandle,
        0ull,
        currentFade_,
        nullptr,
        0,
        sampleCursor,
        callbackEpoch,
        runtimeScope.lifecycleToken.epochId,
        &rtTraceRelay_);

    rtTraceRelay_.enqueue({ rtFrame.sampleCursor, 0xA001u, static_cast<std::uint32_t>(numSamples) });

    DSPCore* dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandleRef);
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
        // ★ ISR準拠: RuntimeWorld 経由でサンプルレートを取得。
        //   RuntimeBuilder は worldOwner->timing.sampleRateHz を
        //   buildRuntimePublishWorld() 時に DSPCore の sampleRate から設定するため、
        //   dsp->sampleRate と runtimeWorld->timing.sampleRateHz は常に一致する。
        const double engineSampleRate = getRuntimeSampleRateHzFromWorld(runtimeReadHandleRef, 0.0);
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
        // ── Audio Thread authority: RuntimeWorld 由来のスナップショットを使用 ──
        const EngineParameterSnapshot parameterSnapshot = captureAudioThreadParameterSnapshot(runtimeWorld);

        DSPCore::ProcessingState procState = buildAudioThreadProcessingState(dsp, parameterSnapshot);

        DSPCore* fading = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandleRef);
        const auto& preparedCrossfade = authority.preparedCrossfade;
        bool useDryAsOld = preparedCrossfade.useDryAsOld || preparedCrossfade.firstIrDryCrossfadePending;

        if (processCrossfadeDelayGateIfPending(fading,
                                               useDryAsOld,
                                               preparedCrossfade,
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

        armCrossfadeIfPending(fading != nullptr, useDryAsOld, preparedCrossfade);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // work60: dsp->process()直前にcallbackSeq/cpuをDSPCoreとConvolverProcessorへ伝達
    dsp->currentCallbackSeq = convo::consumeAtomic(rtLocalState_.audioCallbackEpochCounter, std::memory_order_relaxed);
    dsp->currentCpu = convo::consumeAtomic(rtLocalState_.lastCallbackProcessor, std::memory_order_relaxed);
    dsp->convolver.currentCallbackSeq.store(dsp->currentCallbackSeq, std::memory_order_relaxed);
    dsp->convolver.currentCpu.store(dsp->currentCpu, std::memory_order_relaxed);
    // work60: DSP_STAGE開始時刻。armCrossfade後、dsp->process()直前。
    t1_dspStartUs = convo::getCurrentTimeUs();
#endif

        const bool canCrossfade = (fading != nullptr || useDryAsOld)
            && crossfadeRuntime_.getGain().isSmoothing()
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
                                                     preparedCrossfade.latencyDelayOld,
                                                     preparedCrossfade.latencyDelayNew,
                                                     preparedCrossfade.latencyResetPending,
                                                     [this, useDryAsOld](float* outL,
                                                                         float* outR,
                                                                         int i,
                                                                         double gNew,
                                                                         double alignedOldL,
                                                                         double alignedOldR,
                                                                         double alignedNewL,
                                                                         double alignedNewR)
                                                     {
                                                         const double dryScale = useDryAsOld ? crossfadeRuntime_.getDryScaleGain().getNextValue() : 1.0;
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

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ work60: DSP_STAGE終了時刻（t2）。dsp->process() 完了直後。
    const uint64_t t2_dspEndUs = convo::getCurrentTimeUs();

    // ★ callbackIndex（callback開始直後に1度だけ取得、全診断ブロックで同じ値を使用。work61: 早期取得版を流用）
    //    thisCallbackIndex は関数冒頭で fetchAddAtomic により取得済み

    // ★ DSP Observe 検出: publicationSequence 変化時のみ記録 + PublicationDirection
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
    //   cbStartUs は関数先頭で取得済みの共通タイムスタンプを流用。
    static constexpr auto kNeverStartedUs = std::numeric_limits<uint64_t>::max();
    uint64_t cbPrevEndUs = 0;  // XRUNブロックが上書きする前の前回終了時刻を保存

    // ★ XRUN 検出（callback 時間 + interval 超過）
    {
        const auto t0_start = cbStartUs;  // ★ P2-4: 共通開始時刻を流用
        cbPrevEndUs = convo::consumeAtomic(rtLocalState_.lastCallbackEndTicks, std::memory_order_relaxed);
        const double engineSampleRate = getRuntimeSampleRateHzFromWorld(runtimeReadHandleRef, 0.0);
        const double expectedMs = (engineSampleRate > 0.0)
            ? static_cast<double>(numSamples) / engineSampleRate * 1000.0
            : 0.0;

        // interval 計測（前回終了時刻からの経過）
        double intervalMs = 0.0;
        if (cbPrevEndUs > 0)
        {
            intervalMs = static_cast<double>(t0_start - cbPrevEndUs) / 1000.0;
        }

        const auto t1_end = convo::getCurrentTimeUs();
        const double callbackMs = static_cast<double>(t1_end - t0_start) / 1000.0;

        // 閾値判定
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
    //    callbackMaxUs_ = callback実行時間の最大値(μs)
    //    intervalMaxUs_ = callback間隔の最大値(μs) ★主指標
    //    callbackCount_ = callback実行回数（Timer側で1秒ごとに collect&reset）
    {
        if (cbStartUs != kNeverStartedUs)
        {
            // callback実行時間（t0_start→t1_end から cbStartUs→t1_end に拡張）
            // cbStartUs は早期 return を通過した時点 = DSP処理後の実測開始点
            const auto nowUs = convo::getCurrentTimeUs();
            // callback実行時間(μs) = nowUs - cbStartUs（既にμs）
            const auto callbackUs = static_cast<uint32_t>(nowUs - cbStartUs);
            updateAtomicMaximum(callbackMaxUs_, callbackUs);

            // interval計測(μs) — cbPrevEndUsからcbStartUsまでの経過時間
            if (cbPrevEndUs > 0)
            {
                const auto intervalUs = static_cast<uint32_t>(cbStartUs - cbPrevEndUs);
                updateAtomicMaximum(intervalMaxUs_, intervalUs);
            }
        }

        // callbackカウント（常時インクリメント）
        callbackCount_.fetch_add(1u, std::memory_order_relaxed);
    }

    // ★ DSP_TIMING 出力（Observe検出時のみ）
    if (observeUs > 0 && dspSeq > 0)
    {
        // work60: Numeric-Only DiagEvent
        const uint64_t gen = (runtimeWorld != nullptr)
            ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
        const uint64_t worldId = (runtimeWorld != nullptr)
            ? static_cast<uint64_t>(runtimeWorld->worldId) : 0;
        DiagEvent event{};
        event.category = DiagCategory::DspTiming;
        event.eventIndex = thisCallbackIndex;
        event.data.dspTiming.dspSeq = static_cast<uint64_t>(dspSeq);
        event.data.dspTiming.generation = gen;
        event.data.dspTiming.worldId = worldId;
        // DspTimingData.direction は uint8_t。observeReason(const char*) を数値にマップ。
        //   0 = None(空文字), 1 = Forward, 2 = Rollback, 3 = Replay
        uint8_t reasonVal = 0;
        if (observeReason[0] != '\0') {
            if (std::strcmp(observeReason, "Forward") == 0) reasonVal = 1;
            else if (std::strcmp(observeReason, "Rollback") == 0) reasonVal = 2;
            else if (std::strcmp(observeReason, "Replay") == 0) reasonVal = 3;
        }
        event.data.dspTiming.direction = static_cast<PublicationDirection>(reasonVal);
        if (matchedPublishEndUs > 0)
        {
            const uint64_t observeLatencyUs = observeUs - matchedPublishEndUs;
            event.data.dspTiming.observeLatencyUs = observeLatencyUs;
            event.data.dspTiming.pubToObserveUs = observeLatencyUs;
            if (matchedPublishCallbackIdx > 0 && matchedPublishCallbackIdx < thisCallbackIndex)
            {
                event.data.dspTiming.callbacksUntilObserve = thisCallbackIndex - matchedPublishCallbackIdx;
                event.data.dspTiming.publishCallbackIdx = matchedPublishCallbackIdx;
            }
        }
        if (diagBuffer.push(event))
        {
            rtAuxMutable_.diagTickPushed.value.fetch_add(1, std::memory_order_relaxed);
            rtAuxMutable_.diagTotalPushed.fetch_add(1, std::memory_order_relaxed);
        }
        else
        {
            rtAuxMutable_.diagTickDropped.value.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // ★ work60: CALLBACK_STAGE（INPUT/DSP/OUTPUT 3区間 + drift + budget）
    {
        const uint64_t t0 = callbackTelemetry.startUs;
        const uint64_t gen = (runtimeWorld != nullptr)
            ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
        const uint64_t expectedUs = rtLocalState_.expectedCallbackIntervalUs;
        const int64_t driftUs = convo::consumeAtomic(
            rtLocalState_.lastCallbackDriftUs, std::memory_order_relaxed);

        if ((thisCallbackIndex & CONVOPEQ_DIAG_SAMPLE_MASK) == 0)
        {
            const uint64_t inputUs = (t1_dspStartUs > t0) ? t1_dspStartUs - t0 : 0;
            const uint64_t t3 = convo::getCurrentTimeUs();
            const uint32_t cpu = static_cast<uint32_t>(::GetCurrentProcessorNumber());
            const uint64_t dspUs = (t2_dspEndUs > t1_dspStartUs) ? t2_dspEndUs - t1_dspStartUs : 0;
            const uint64_t outputUs = (t3 > t2_dspEndUs) ? t3 - t2_dspEndUs : 0;
            const uint64_t totalUs = (t3 > t0) ? t3 - t0 : 0;
            const uint32_t budgetPermille = (expectedUs > 0)
                ? static_cast<uint32_t>(totalUs * 1000 / expectedUs) : 0;
            // work60: Numeric-Only DiagEvent
            DiagEvent event{};
            event.category = DiagCategory::CallbackStage;
            event.eventIndex = thisCallbackIndex;
            event.data.callbackStage.cpu = cpu;
            event.data.callbackStage.generation = gen;
            event.data.callbackStage.expectedUs = expectedUs;
            event.data.callbackStage.driftUs = driftUs;
            event.data.callbackStage.inputUs = inputUs;
            event.data.callbackStage.dspUs = dspUs;
            event.data.callbackStage.outputUs = outputUs;
            event.data.callbackStage.totalUs = totalUs;
            event.data.callbackStage.budgetPermille = static_cast<uint16_t>(budgetPermille);
            if (diagBuffer.push(event))
            {
                rtAuxMutable_.diagTickPushed.value.fetch_add(1, std::memory_order_relaxed);
                rtAuxMutable_.diagTotalPushed.fetch_add(1, std::memory_order_relaxed);
            }
            else
            {
                rtAuxMutable_.diagTickDropped.value.fetch_add(1, std::memory_order_relaxed);
            }
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

    // ★ [work63] シャットダウン要求: Audio Thread 自ら MMCSS を解除（診断ガード外＝常時有効）
    //    Message Thread が mmcssShutdownRequested をセットすると、
    //    次のコールバック終了時にこのスレッド上で AvRevertMmThreadCharacteristics を実行する。
    if (convo::consumeAtomic(mmcssShutdownRequested, std::memory_order_acquire))
    {
        revertMmcssPriorityOnAudioThread();
    }

}

