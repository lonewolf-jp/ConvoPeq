#include <JuceHeader.h>
#include "AudioEngine.h"
#include "RuntimeBuilder.h"

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

static void destroyPublicationIntentNode(void* ptr) noexcept
{
    delete static_cast<AudioEngine::PublicationIntent*>(ptr);
}

template <typename T>
static inline T* sanitizeRawPtr(T* ptr) noexcept
{
    constexpr uintptr_t kInvalidAllOnes = ~static_cast<uintptr_t>(0);
    return (reinterpret_cast<uintptr_t>(ptr) == kInvalidAllOnes) ? nullptr : ptr;
}
}


#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_COMMIT_PREPARE)

void AudioEngine::appendPublicationIntent(DSPCore* newDSP, int generation, int epochReaderIndex) noexcept
{
    if (newDSP == nullptr)
        return;

    const convo::EpochDomainReaderGuard appendGuard(m_epochDomain, epochReaderIndex);

    auto* intent = new PublicationIntent();
    intent->newDSP = newDSP;
    intent->generation = generation;
    // intent は生成直後でまだ他スレッドから不可視のため、next の nullptr 初期化に ordering 不要。
    convo::publishAtomic(intent->next, static_cast<PublicationIntent*>(nullptr), std::memory_order_relaxed);

    PublicationIntent* tail = convo::consumeAtomic(publicationLog.head, std::memory_order_acquire); // acquire: next CAS の release と HB
    if (tail == nullptr)
    {
        retireDSP(newDSP);
        delete intent;
        return;
    }

    for (;;)
    {
        PublicationIntent* next = convo::consumeAtomic(tail->next, std::memory_order_acquire); // acquire: CAS release と HB
        if (next == nullptr)
        {
            if (convo::compareExchangeAtomic(tail->next,
                                             next,
                                             intent,
                                             std::memory_order_release, // release: 後続の acquire load と HB
                                             std::memory_order_acquire)) // acquire: CAS 失敗時のリロード
            {
                PublicationIntent* observedTail = tail;
                // failure 側は head を自分が書き換えない。次ループの acquire load で再取得するため ordering 不要。
                convo::compareExchangeAtomic(publicationLog.head,
                                             observedTail,
                                             intent,
                                             std::memory_order_release, // release: head 更新を公開
                                             std::memory_order_relaxed); // CAS 失敗時は再取得するため relaxed
                break;
            }
        }
        else
        {
            PublicationIntent* observedTail = tail;
            // failure 側は head を自分が書き換えない。次ループの acquire load で再取得するため ordering 不要。
            convo::compareExchangeAtomic(publicationLog.head,
                                         observedTail,
                                         next,
                                         std::memory_order_release, // release: head 更新を公開
                                         std::memory_order_relaxed); // CAS 失敗時は再取得するため relaxed
        }

        tail = convo::consumeAtomic(publicationLog.head, std::memory_order_acquire); // acquire: 更新した head を読み込み
        if (tail == nullptr)
            tail = publicationLogSentinel;
    }
}

void AudioEngine::drainPublicationLogForShutdown() noexcept
{
    PublicationIntent* cursor = convo::consumeAtomic(publicationLog.consumedTail, std::memory_order_acquire); // acquire: executeCommit の publishAtomic release と HB
    if (cursor == nullptr)
        cursor = publicationLogSentinel;

    if (cursor != nullptr)
    {
        for (;;)
        {
            PublicationIntent* const next = convo::consumeAtomic(cursor->next, std::memory_order_acquire); // acquire: appendPublicationIntent の CAS release と HB
            if (next == nullptr)
                break;

            if (next->newDSP != nullptr)
                retireDSP(next->newDSP);

            enqueueDeferredDeleteNonRt(next, destroyPublicationIntentNode);
            convo::publishAtomic(publicationLog.retiredHead, next, std::memory_order_release); // release: 後続の consume acquire と HB
            convo::publishAtomic(publicationLog.consumedTail, next, std::memory_order_release); // release: 次次 consume acquire と HB
            cursor = next;
        }

        convo::publishAtomic(publicationLog.head, cursor, std::memory_order_release); // release: shutdown 後の赴取りを不可視、終了前の統一バリア

        if (cursor != publicationLogSentinel)
            enqueueDeferredDeleteNonRt(cursor, destroyPublicationIntentNode);
    }

    if (publicationLogSentinel != nullptr)
    {
        enqueueDeferredDeleteNonRt(publicationLogSentinel, destroyPublicationIntentNode);
        publicationLogSentinel = nullptr;
    }

    convo::publishAtomic(publicationLog.head, static_cast<PublicationIntent*>(nullptr), std::memory_order_release); // release: shutdown 後の sentinel 彸残を防止
    convo::publishAtomic(publicationLog.consumedTail, static_cast<PublicationIntent*>(nullptr), std::memory_order_release); // release: 後続の acquire を不可視、null 保証
    convo::publishAtomic(publicationLog.retiredHead, static_cast<PublicationIntent*>(nullptr), std::memory_order_release); // release: 後続の consume acquire と HB
}

void AudioEngine::prepareCommit(DSPCore* newDSP, int generation)
{
    if (newDSP == nullptr)
        return;

    if (isShutdownInProgress())
    {
        retireDSP(newDSP);
        return;
    }

    appendPublicationIntent(newDSP, generation, kCommitProducerEpochReaderIndex);

    triggerAsyncUpdate();
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_COMMIT_EXECUTE)
bool AudioEngine::hasPublicationLogPending() noexcept
{
    PublicationIntent* const cursor = convo::consumeAtomic(publicationLog.consumedTail, std::memory_order_acquire); // acquire: executeCommit の publishAtomic release と HB
    return cursor != nullptr && convo::consumeAtomic(cursor->next, std::memory_order_acquire) != nullptr; // acquire: appendPublicationIntent の CAS release と HB
}

bool AudioEngine::hasPendingPublicationIntents() noexcept
{
    return hasPublicationLogPending();
}

void AudioEngine::executeCommit()
{
    if (convo::exchangeAtomic(commitDrainInProgress, true, std::memory_order_acq_rel)) // acq_rel: prior publish をacquire、本体の publish をrelease
        return;

    PublicationIntent* cursor = convo::consumeAtomic(publicationLog.consumedTail, std::memory_order_acquire); // acquire: drainPublicationLogForShutdown の publishAtomic release と HB
    if (cursor == nullptr)
        cursor = publicationLogSentinel;

    if (cursor != nullptr)
    {
        for (;;)
        {
            PublicationIntent* const next = convo::consumeAtomic(cursor->next, std::memory_order_acquire); // acquire: appendPublicationIntent の CAS release と HB
            if (next == nullptr)
                break;

            PublicationIntent* expected = cursor;
            if (!convo::compareExchangeAtomic(publicationLog.consumedTail,
                                              expected,
                                              next,
                                              std::memory_order_acq_rel, // acq_rel: acquire で旧 cursor を読み込み、release で次を公開
                                              std::memory_order_acquire)) // acquire: CAS 失敗時の再読み
            {
                cursor = expected;
                continue;
            }

            if (isShutdownInProgress())
            {
                if (next->newDSP != nullptr)
                    retireDSP(next->newDSP);
            }
            else
            {
                commitNewDSP(next->newDSP, next->generation);
            }

            if (cursor != publicationLogSentinel)
                enqueueDeferredDeleteNonRt(cursor, destroyPublicationIntentNode);

            convo::publishAtomic(publicationLog.retiredHead, cursor, std::memory_order_release); // release: drainPublicationLogForShutdown の consume acquire と HB
            cursor = next;
        }
    }

    const bool hasRemaining = hasPendingPublicationIntents();

    convo::publishAtomic(commitDrainInProgress, false, std::memory_order_release); // release: 次回の hasPublicationLogPending の acquire と HB

    if (hasRemaining && !isShutdownInProgress())
        triggerAsyncUpdate();
}
#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_COMMIT_EXECUTE)


void AudioEngine::commitNewDSP(DSPCore* newDSP, int generation)
{
    struct CrossfadeContext
    {
        bool needsCrossfade = false;
        bool oldHasIR = false;
        bool newHasIR = false;
        double fadeTimeSec = 0.0;
    };

    DSPCore* dspToTrash = nullptr;
    bool scheduleDryAsOldCrossfade = false;
    double dryAsOldFadeTimeSec = 0.0;
    int transitionLatencyDeltaSamples = 0;
    CrossfadeContext crossfadeContext;

    validateDistinctRuntimeSlots("commitNewDSP.entry",
                                 activeDSP,
                                 resolveFadingDSPFromRuntimeWorldOnly(getRuntimePublishView().graph),
                                 nullptr);

    // Lock to ensure the check and commit are atomic with respect to new rebuild requests.
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        // 古いリクエストの結果であれば破棄 (Race condition対策)
        if (generation != consumeAtomic(rebuildGeneration, std::memory_order_acquire)) // acquire: prepareCommit の publishAtomic release と HB
        {
            publishAtomic(lastRejectedGenerationNonRt, static_cast<uint64_t>(generation), std::memory_order_release); // release: UI の consumeAtomic acquire と HB
            retireDSP(newDSP);
            return;
        }

        // 公開不変条件:
        // IR を実際に使う構成では finalized 済みのみ公開する。
        // 一方、IR 未ロード時のパススルーDSPまで弾くと起動直後に無音化するため許可する。
        if (newDSP == nullptr
            || (newDSP->convolverRt().isIRLoaded() && !newDSP->convolverRt().isIRFinalized()))
        {
            DBG("[AudioEngine] commitNewDSP: rejected non-finalized DSP publish");
            publishAtomic(lastRejectedGenerationNonRt, static_cast<uint64_t>(generation), std::memory_order_release); // release: UI の consumeAtomic acquire と HB
            if (newDSP != nullptr)
                retireDSP(newDSP);
            return;
        }

        // 1. 旧 DSP を安全にキャプチャしてから新 DSP を公開する
        dspToTrash = activeDSP;

        const uint64_t newSessionId = convo::fetchAddAtomic(globalCaptureSessionId,
                                    static_cast<uint64_t>(1),
                                    std::memory_order_acq_rel) + 1; // acq_rel: audio thread の capture session 鏃定
        if (newDSP != nullptr)
            newDSP->currentCaptureSessionId = newSessionId;

        // Warmup: FIR 履歴と AGC state を初期化する
        // currentDSP.store より前に実行し、安定した state で Audio thread に提供
        if (newDSP != nullptr)
        {
            convo::RuntimeBuilder builder(*this);
            const convo::BuildError warmupError = builder.executeWarmup(*newDSP);
            if (warmupError != convo::BuildError::None)
            {
                diagLog("[AudioEngine] commitNewDSP: warmup failed, rejecting DSP publish (err=" + juce::String(convo::toString(warmupError)) + ")");
                publishAtomic(lastRejectedGenerationNonRt, static_cast<uint64_t>(generation), std::memory_order_release); // release: UI の consumeAtomic acquire と HB
                retireDSP(newDSP);
                return;
            }
        }

        if (newDSP != nullptr && dspToTrash != nullptr)
        {
            const auto computeCrossfadeContext = [this](const DSPCore* oldDSP, const DSPCore* candidateDSP) noexcept -> CrossfadeContext
            {
                CrossfadeContext ctx;
                if (oldDSP == nullptr || candidateDSP == nullptr)
                    return ctx;

                ctx.oldHasIR = oldDSP->convolverRt().isIRLoaded();
                ctx.newHasIR = candidateDSP->convolverRt().isIRLoaded();
                const bool hasAudibleConvolverTransition = ctx.oldHasIR || ctx.newHasIR;
                const bool irPresenceChanged = (ctx.oldHasIR != ctx.newHasIR);

                if (hasAudibleConvolverTransition
                    && candidateDSP->oversamplingFactor != oldDSP->oversamplingFactor)
                {
                    ctx.needsCrossfade = true;
                    ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_osFadeTimeSec, std::memory_order_acquire)); // acquire: setOversamplingFadeTime publishAtomic release と HB
                }

                if (hasAudibleConvolverTransition)
                {
                    const uint64_t oldHash = oldDSP->convolverRt().getStructuralHash();
                    const uint64_t newHash = candidateDSP->convolverRt().getStructuralHash();
                    if (oldHash != newHash)
                    {
                        ctx.needsCrossfade = true;
                        const double baseIrFade = consumeAtomic(m_irFadeTimeSec, std::memory_order_acquire); // acquire: setIRFadeTime publishAtomic release と HB
                        if (irPresenceChanged)
                        {
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, std::clamp(baseIrFade, 0.001, 0.010));
                        }
                        else
                        {
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, baseIrFade);
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_irLengthFadeTimeSec, std::memory_order_acquire)); // acquire: setIRLengthFadeTime publishAtomic release と HB
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_phaseFadeTimeSec, std::memory_order_acquire)); // acquire: setPhaseFadeTime publishAtomic release と HB
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_directHeadFadeTimeSec, std::memory_order_acquire)); // acquire: setDirectHeadFadeTime publishAtomic release と HB
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_nucFilterFadeTimeSec, std::memory_order_acquire)); // acquire: setNucFilterFadeTime publishAtomic release と HB
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_tailFadeTimeSec, std::memory_order_acquire)); // acquire: setTailFadeTime publishAtomic release と HB
                        }
                    }
                }

                return ctx;
            };

            crossfadeContext = computeCrossfadeContext(dspToTrash, newDSP);

            if (crossfadeContext.needsCrossfade)
            {
                const auto runtimePublishView = getRuntimePublishView();
                const auto* runtimeGraph = runtimePublishView.graph;
                const auto preparedCrossfade = consumeCrossfadePreparedSnapshot();
                const bool hasFadingRuntime = (resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph) != nullptr);
                const bool hasPendingCrossfade = runtimeCrossfadePendingWorldOnly(runtimeGraph)
                    || preparedCrossfade.pending;
                const bool useDryAsOld = runtimeCrossfadeUseDryAsOldWorldOnly(runtimeGraph)
                    || preparedCrossfade.firstIrDryCrossfadePending
                    || preparedCrossfade.useDryAsOld;

                if (hasFadingRuntime || hasPendingCrossfade || useDryAsOld)
                {
                    diagLog("[DIAG] commitNewDSP: deferring commit until active fade settles newUuid="
                        + juce::String(static_cast<juce::int64>(newDSP->runtimeUuid))
                        + " oldUuid=" + juce::String(static_cast<juce::int64>(dspToTrash->runtimeUuid))
                        + " fadeSec=" + juce::String(crossfadeContext.fadeTimeSec, 3));
                    appendPublicationIntent(newDSP, generation, kCommitConsumerEpochReaderIndex);
                    return;
                }
            }
        }

        // 2. 新ランタイム publish を coordinator authority API 経由へ集約する
        publicationCoordinator().adoptAndPublish(newDSP,
                             nullptr,
                             convo::TransitionPolicy::SmoothOnly,
                             0.0,
                             false);

        // 3. EBR：エポックを進める
        m_epochDomain.advanceEpoch();

        validateDistinctRuntimeSlots("commitNewDSP.afterPublish",
                         activeDSP,
                         resolveFadingDSPFromRuntimeWorldOnly(getRuntimePublishView().graph),
                         nullptr);

        // この世代の publish が完了したので outstanding rebuild 窓を閉じる。
        publishAtomic(lastCommittedRebuildGeneration, generation, std::memory_order_release); // release: isRebuildOutstanding の consume acquire と HB

        const bool committedHasIr = newDSP->convolverRt().isIRLoaded();
        const uint64_t committedStructuralHash = committedHasIr
            ? newDSP->convolverRt().getStructuralHash()
            : static_cast<uint64_t>(0);
        publishAtomic(lastCommittedConvolverHasIr_, committedHasIr, std::memory_order_release); // release: UI の consume acquire と HB
        publishAtomic(lastCommittedConvolverStructuralHash_, committedStructuralHash, std::memory_order_release); // release: UI の consume acquire と HB
    }


    // 5. 初回IRロード時（旧DSPなし）: dry を旧信号としてクロスフェード予約
    if (dspToTrash == nullptr
        && newDSP != nullptr
        && newDSP->convolverRt().isIRLoaded()
        && !consumeAtomic(firstIrDryCrossfadeDone, std::memory_order_acquire)) // acquire: armDryAsOldCrossfadeForCurrentDSP publishAtomic release と HB
    {
        // 初回のみ dry -> IR を明示的にフェードし、立ち上がりノイズを抑制する。
        scheduleDryAsOldCrossfade = true;
        dryAsOldFadeTimeSec = std::max(0.001, consumeAtomic(m_irFadeTimeSec, std::memory_order_acquire)); // acquire: setIRFadeTime publishAtomic release と HB

        const bool convBypassedForLatency = consumeAtomic(m_currentConvBypass, std::memory_order_acquire); // acquire: setConvolverBypass publishAtomic release と HB
        const int newLatency = estimateRuntimeLatencyBaseRateSamples(newDSP, convBypassedForLatency);
        int dOld = std::min(newLatency, latencyBufSize - 1); // dry 側を遅延させて整合
        const int dNew = 0;
        publishAtomic(latencyDelayOld, dOld, std::memory_order_release); // release: audio thread の latency read と HB
        publishAtomic(latencyDelayNew, dNew, std::memory_order_release); // release: audio thread の latency read と HB
        publishAtomic(latencyResetPending, true, std::memory_order_release); // release: audio thread の reset poll と HB
        transitionLatencyDeltaSamples = dOld - dNew;

        diagLog("[DIAG] commitNewDSP: dry->IR latency align old="
            + juce::String(static_cast<juce::int64>(dOld))
            + " new=" + juce::String(static_cast<juce::int64>(dNew))
            + " effectiveNew=" + juce::String(static_cast<juce::int64>(newLatency))
            + " convBypassed=" + juce::String(convBypassedForLatency ? 1 : 0));
    }

    diagLog("[DIAG] commitNewDSP: entry gen=" + juce::String(generation)
        + " dspToTrash=" + (dspToTrash != nullptr ? juce::String(dspToTrash->convolverRt().isIRLoaded() ? "IR" : "passthrough") : "null")
        + " oldUuid=" + juce::String(static_cast<juce::int64>(dspToTrash != nullptr ? dspToTrash->runtimeUuid : 0))
        + " irLoaded=" + (newDSP != nullptr ? juce::String((int)newDSP->convolverRt().isIRLoaded()) : "n/a")
        + " newUuid=" + juce::String(static_cast<juce::int64>(newDSP != nullptr ? newDSP->runtimeUuid : 0)));
    // 5. RCU deferred release：旧 DSP を grace period 後に解放する
    if (dspToTrash != nullptr)
    {
        if (newDSP != nullptr)
        {
            if (crossfadeContext.needsCrossfade)
            {
                double fadeTimeSec = crossfadeContext.fadeTimeSec;
                const bool convBypassedForLatency = consumeAtomic(m_currentConvBypass, std::memory_order_acquire); // acquire: setConvolverBypass publishAtomic release と HB
                const int oldLatency = estimateRuntimeLatencyBaseRateSamples(dspToTrash, convBypassedForLatency);
                const int newLatency = estimateRuntimeLatencyBaseRateSamples(newDSP, convBypassedForLatency);
                const int targetLatency = std::max(oldLatency, newLatency);
                int dOld = targetLatency - oldLatency;
                int dNew = targetLatency - newLatency;
                dOld = std::min(dOld, latencyBufSize - 1);
                dNew = std::min(dNew, latencyBufSize - 1);
                publishAtomic(latencyDelayOld, dOld, std::memory_order_release); // release: audio thread の latency read と HB
                publishAtomic(latencyDelayNew, dNew, std::memory_order_release); // release: audio thread の latency read と HB
                // ★ resetはAudioThreadで1回だけ行う
                publishAtomic(latencyResetPending, true, std::memory_order_release); // release: audio thread の reset poll と HB
                transitionLatencyDeltaSamples = dOld - dNew;

                diagLog("[DIAG] commitNewDSP: latency align old="
                    + juce::String(static_cast<juce::int64>(dOld))
                    + " new=" + juce::String(static_cast<juce::int64>(dNew))
                    + " effectiveOld=" + juce::String(static_cast<juce::int64>(oldLatency))
                    + " effectiveNew=" + juce::String(static_cast<juce::int64>(newLatency))
                    + " convBypassed=" + juce::String(convBypassedForLatency ? 1 : 0));

                if (!crossfadeContext.oldHasIR && crossfadeContext.newHasIR)
                    publishAtomic(dspCrossfadeStartDelayBlocks,
                                  std::max(0, consumeAtomic(m_crossfadeStartDelayBlocks, std::memory_order_acquire))); // acquire: setCrossfadeStartDelayBlocks publishAtomic release と HB
                else
                    publishAtomic(dspCrossfadeStartDelayBlocks, 0, std::memory_order_release); // release: audio thread の delay poll と HB

                // デフォルト値（fadeTimeSec==0なら30ms）
                if (fadeTimeSec <= 0.0)
                    fadeTimeSec = 0.030;

                // --- クロスフェードdeduplication・スナップショット ---
                const auto runtimePublishView = getRuntimePublishView();
                const auto* runtimeGraph = runtimePublishView.graph;
                const auto preparedCrossfade = consumeCrossfadePreparedSnapshot();
                const bool hasFadingRuntime = (resolveFadingDSPFromRuntimeWorldOnly(runtimeGraph) != nullptr);
                const bool hasPendingCrossfade = runtimeCrossfadePendingWorldOnly(runtimeGraph)
                    || preparedCrossfade.pending;
                const bool useDryAsOld = runtimeCrossfadeUseDryAsOldWorldOnly(runtimeGraph)
                    || preparedCrossfade.firstIrDryCrossfadePending
                    || preparedCrossfade.useDryAsOld;
                const bool isFadingActive = hasFadingRuntime || hasPendingCrossfade || useDryAsOld;
                publishSmoothTransitionState(activeDSP,
                                             dspToTrash,
                                             fadeTimeSec);
                jassert(!isFadingActive);
                startImmediateSmoothTransition(dspToTrash, fadeTimeSec);
            }
            else
            {
                // クロスフェード不要時は遷移用遅延設定を無効化し、旧DSPを即時解放する。
                publishAtomic(dspCrossfadeStartDelayBlocks, 0, std::memory_order_release); // release: audio thread の delay poll と HB
                retireRuntimeImmediately(dspToTrash);
                publishHardResetForCurrentDSP();
            }
        }
    }

    if (scheduleDryAsOldCrossfade)
    {
        armDryAsOldCrossfadeForCurrentDSP(dryAsOldFadeTimeSec,
                                          uiConvolverProcessor.getCurrentIRScale());

        diagLog("[DIAG] commitNewDSP: first-load dry->IR crossfade armed fadeSec="
            + juce::String(dryAsOldFadeTimeSec, 3)
            + " irName=" + newDSP->convolverRt().getIRName());
    }

    if (newDSP != nullptr)
    {
        diagLog("[DIAG] commitNewDSP: before setMixedPhaseState state="
            + juce::String(newDSP->convolverRt().getMixedPhaseState()));
        uiConvolverProcessor.setMixedPhaseState(newDSP->convolverRt().getMixedPhaseState());
        diagLog("[DIAG] commitNewDSP: after setMixedPhaseState");
    }

    const LearningCommand cmd {
        LearningCommand::Type::DSPReady,
        false,
        consumeAtomic(pendingLearningMode, std::memory_order_acquire), // acquire: setNoiseShaperLearningMode publishAtomic release と HB
        static_cast<uint64_t>(generation)
    };

    diagLog("[DIAG] commitNewDSP: before enqueueLearningCommand");
    if (!enqueueLearningCommand(cmd))
    {
        DBG("[AudioEngine] commitNewDSP: command queue overflow");
        diagLog("[DIAG] commitNewDSP: enqueueLearningCommand overflow");
    }
    else
    {
        diagLog("[DIAG] commitNewDSP: enqueueLearningCommand ok");
    }

    // NOTE: rebuild 完了通知の唯一の発火点。
    // sendChangeMessage() は commitNewDSP() でのみ rebuild 用途で呼ぶ。
    // それ以外の sendChangeMessage() はフェード完了・UIパラメータ変更・
    // 状態復元など rebuild とは独立したイベント用途。
    validateDistinctRuntimeSlots("commitNewDSP.beforeSendChangeMessage",
                                 activeDSP,
                                 resolveFadingDSPFromRuntimeWorldOnly(getRuntimePublishView().graph),
                                 nullptr);
    diagLog("[DIAG] commitNewDSP: queue coalesced change notification");
    queueCoalescedChangeNotification();
}
#endif
