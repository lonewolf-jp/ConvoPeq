#include <JuceHeader.h>
#include "AudioEngine.h"
#include "RuntimeBuilder.h"

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

template <typename T>
static inline T* sanitizeRawPtr(T* ptr) noexcept
{
    constexpr uintptr_t kInvalidAllOnes = ~static_cast<uintptr_t>(0);
    return (reinterpret_cast<uintptr_t>(ptr) == kInvalidAllOnes) ? nullptr : ptr;
}
}


#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_COMMIT_PREPARE)

void AudioEngine::prepareCommit(DSPCore* newDSP, int generation)
{
    if (newDSP == nullptr)
        return;

    if (isShutdownInProgress())
    {
        retireDSP(newDSP);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(deferredCommitMutex);

        if (isShutdownInProgress())
        {
            retireDSP(newDSP);
            return;
        }

        deferredCommitQueue.push(CommitStaging { newDSP, nullptr, generation });
    }

    triggerAsyncUpdate();
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_COMMIT_EXECUTE)
void AudioEngine::executeCommit()
{
    std::queue<CommitStaging> localQueue;

    {
        std::lock_guard<std::mutex> lock(deferredCommitMutex);
        std::swap(localQueue, deferredCommitQueue);
    }

    while (!localQueue.empty())
    {
        auto staging = localQueue.front();
        localQueue.pop();

        if (staging.newDSP == nullptr)
            continue;

        if (isShutdownInProgress())
        {
            retireDSP(staging.newDSP);
            continue;
        }

        commitNewDSP(staging.newDSP, staging.generation);
    }
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
    uint64_t retireEpoch = 0;
    bool scheduleDryAsOldCrossfade = false;
    double dryAsOldFadeTimeSec = 0.0;
    int transitionLatencyDeltaSamples = 0;
    CrossfadeContext crossfadeContext;

    validateDistinctRuntimeSlots("commitNewDSP.entry",
                                 activeDSP,
                                 sanitizeRawPtr(loadFadingOutDSP()),
                                 nullptr);

    // Lock to ensure the check and commit are atomic with respect to new rebuild requests.
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        // 古いリクエストの結果であれば破棄 (Race condition対策)
        if (generation != consumeAtomic(rebuildGeneration, std::memory_order_acquire))
        {
            publishAtomic(lastRejectedGenerationNonRt, static_cast<uint64_t>(generation));
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
            publishAtomic(lastRejectedGenerationNonRt, static_cast<uint64_t>(generation));
            if (newDSP != nullptr)
                retireDSP(newDSP);
            return;
        }

        // 1. 旧 DSP を安全にキャプチャしてから新 DSP を公開する
        dspToTrash = activeDSP;

        const uint64_t newSessionId = globalCaptureSessionId.fetch_add(1, std::memory_order_acq_rel) + 1;
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
                publishAtomic(lastRejectedGenerationNonRt, static_cast<uint64_t>(generation));
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
                    ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_osFadeTimeSec, std::memory_order_acquire));
                }

                if (hasAudibleConvolverTransition)
                {
                    const uint64_t oldHash = oldDSP->convolverRt().getStructuralHash();
                    const uint64_t newHash = candidateDSP->convolverRt().getStructuralHash();
                    if (oldHash != newHash)
                    {
                        ctx.needsCrossfade = true;
                        const double baseIrFade = consumeAtomic(m_irFadeTimeSec, std::memory_order_acquire);
                        if (irPresenceChanged)
                        {
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, std::clamp(baseIrFade, 0.001, 0.010));
                        }
                        else
                        {
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, baseIrFade);
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_irLengthFadeTimeSec, std::memory_order_acquire));
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_phaseFadeTimeSec, std::memory_order_acquire));
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_directHeadFadeTimeSec, std::memory_order_acquire));
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_nucFilterFadeTimeSec, std::memory_order_acquire));
                            ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, consumeAtomic(m_tailFadeTimeSec, std::memory_order_acquire));
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
                const bool hasFadingRuntime = (resolveFadingDSPFromRuntimePublish(runtimeGraph) != nullptr);
                const bool hasPendingCrossfade = runtimeCrossfadePending(runtimeGraph);
                const bool useDryAsOld = runtimeCrossfadeUseDryAsOld(runtimeGraph);

                if (hasFadingRuntime || hasPendingCrossfade || useDryAsOld)
                {
                    diagLog("[DIAG] commitNewDSP: deferring commit until active fade settles newUuid="
                        + juce::String(static_cast<juce::int64>(newDSP->runtimeUuid))
                        + " oldUuid=" + juce::String(static_cast<juce::int64>(dspToTrash->runtimeUuid))
                        + " fadeSec=" + juce::String(crossfadeContext.fadeTimeSec, 3));
                    deferredCommitQueue.push(CommitStaging { newDSP, nullptr, generation });
                    return;
                }
            }
        }

        // 2. 新ランタイム publish と所有権引き渡しを helper 経由で集約する
        publishCurrentDSPAndTakeOwnership(newDSP);

        // 3. EBR：エポックを進める
        convo::EpochManager::instance().advanceEpoch();
        retireEpoch = convo::EpochManager::instance().currentEpoch();
        publishAtomic(g_currentEpoch, retireEpoch);

        publishRuntimeSnapshots(newDSP,
                    nullptr,
                    convo::TransitionPolicy::SmoothOnly,
                    0.0,
                    false);
        publishRuntimeTransitionState(newDSP,
                          nullptr,
                          convo::TransitionPolicy::SmoothOnly,
                          0.0,
                          false);

        validateDistinctRuntimeSlots("commitNewDSP.afterPublish",
                         activeDSP,
                         sanitizeRawPtr(loadFadingOutDSP()),
                         nullptr);

        // この世代の publish が完了したので outstanding rebuild 窓を閉じる。
        publishAtomic(lastCommittedRebuildGeneration, generation);

        const bool committedHasIr = newDSP->convolverRt().isIRLoaded();
        const uint64_t committedStructuralHash = committedHasIr
            ? newDSP->convolverRt().getStructuralHash()
            : static_cast<uint64_t>(0);
        publishAtomic(lastCommittedConvolverHasIr_, committedHasIr, std::memory_order_release);
        publishAtomic(lastCommittedConvolverStructuralHash_, committedStructuralHash, std::memory_order_release);
    }


    // 5. 初回IRロード時（旧DSPなし）: dry を旧信号としてクロスフェード予約
    if (dspToTrash == nullptr
        && newDSP != nullptr
        && newDSP->convolverRt().isIRLoaded()
        && !consumeAtomic(firstIrDryCrossfadeDone))
    {
        // 初回のみ dry -> IR を明示的にフェードし、立ち上がりノイズを抑制する。
        scheduleDryAsOldCrossfade = true;
        dryAsOldFadeTimeSec = std::max(0.001, consumeAtomic(m_irFadeTimeSec, std::memory_order_acquire));

        const bool convBypassedForLatency = consumeAtomic(m_currentConvBypass, std::memory_order_acquire);
        const int newLatency = estimateRuntimeLatencyBaseRateSamples(newDSP, convBypassedForLatency);
        int dOld = std::min(newLatency, latencyBufSize - 1); // dry 側を遅延させて整合
        const int dNew = 0;
        publishAtomic(latencyDelayOld, dOld);
        publishAtomic(latencyDelayNew, dNew);
        publishAtomic(latencyResetPending, true);
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
                const bool convBypassedForLatency = consumeAtomic(m_currentConvBypass, std::memory_order_acquire);
                const int oldLatency = estimateRuntimeLatencyBaseRateSamples(dspToTrash, convBypassedForLatency);
                const int newLatency = estimateRuntimeLatencyBaseRateSamples(newDSP, convBypassedForLatency);
                const int targetLatency = std::max(oldLatency, newLatency);
                int dOld = targetLatency - oldLatency;
                int dNew = targetLatency - newLatency;
                dOld = std::min(dOld, latencyBufSize - 1);
                dNew = std::min(dNew, latencyBufSize - 1);
                publishAtomic(latencyDelayOld, dOld);
                publishAtomic(latencyDelayNew, dNew);
                // ★ resetはAudioThreadで1回だけ行う
                publishAtomic(latencyResetPending, true);
                transitionLatencyDeltaSamples = dOld - dNew;

                diagLog("[DIAG] commitNewDSP: latency align old="
                    + juce::String(static_cast<juce::int64>(dOld))
                    + " new=" + juce::String(static_cast<juce::int64>(dNew))
                    + " effectiveOld=" + juce::String(static_cast<juce::int64>(oldLatency))
                    + " effectiveNew=" + juce::String(static_cast<juce::int64>(newLatency))
                    + " convBypassed=" + juce::String(convBypassedForLatency ? 1 : 0));

                if (!crossfadeContext.oldHasIR && crossfadeContext.newHasIR)
                    publishAtomic(dspCrossfadeStartDelayBlocks,
                                  std::max(0, consumeAtomic(m_crossfadeStartDelayBlocks, std::memory_order_acquire)));
                else
                    publishAtomic(dspCrossfadeStartDelayBlocks, 0);

                // デフォルト値（fadeTimeSec==0なら30ms）
                if (fadeTimeSec <= 0.0)
                    fadeTimeSec = 0.030;

                // --- クロスフェードdeduplication・スナップショット ---
                const auto runtimePublishView = getRuntimePublishView();
                const auto* runtimeGraph = runtimePublishView.graph;
                const auto preparedCrossfade = consumeCrossfadePreparedSnapshot();
                const bool hasFadingRuntime = (resolveFadingDSPFromRuntimePublish(runtimeGraph) != nullptr);
                const bool hasPendingCrossfade = runtimeCrossfadePending(runtimeGraph)
                    || preparedCrossfade.pending;
                const bool useDryAsOld = runtimeCrossfadeUseDryAsOld(runtimeGraph)
                    || preparedCrossfade.firstIrDryCrossfadePending
                    || preparedCrossfade.useDryAsOld;
                const bool isFadingActive = hasFadingRuntime || hasPendingCrossfade || useDryAsOld;
                publishSmoothTransitionState(activeDSP,
                                             dspToTrash,
                                             fadeTimeSec,
                                             transitionLatencyDeltaSamples);
                jassert(!isFadingActive);
                startImmediateSmoothTransition(dspToTrash, fadeTimeSec);
            }
            else
            {
                // クロスフェード不要時は遷移用遅延設定を無効化し、旧DSPを即時解放する。
                publishAtomic(dspCrossfadeStartDelayBlocks, 0);
                retireRuntimeImmediately(dspToTrash);
                publishHardResetForCurrentDSP();
            }
        }
    }

    if (scheduleDryAsOldCrossfade)
    {
        armDryAsOldCrossfadeForCurrentDSP(dryAsOldFadeTimeSec,
                                          transitionLatencyDeltaSamples,
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
        consumeAtomic(pendingLearningMode),
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
                                 sanitizeRawPtr(loadFadingOutDSP()),
                                 nullptr);
    diagLog("[DIAG] commitNewDSP: queue coalesced change notification");
    queueCoalescedChangeNotification();
}
#endif
