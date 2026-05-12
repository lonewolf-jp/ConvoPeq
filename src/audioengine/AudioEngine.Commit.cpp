#include <JuceHeader.h>
#include "AudioEngine.h"
#include "RuntimeBuilder.h"

extern std::atomic<bool> gShuttingDown;

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

    if (shutdownInProgress.load(std::memory_order_acquire) ||
        gShuttingDown.load(std::memory_order_acquire))
    {
        retireDSP(newDSP);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(deferredCommitMutex);

        if (shutdownInProgress.load(std::memory_order_acquire) ||
            gShuttingDown.load(std::memory_order_acquire))
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

        if (shutdownInProgress.load(std::memory_order_acquire) ||
            gShuttingDown.load(std::memory_order_acquire))
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
    DSPCore* dspToTrash = nullptr;
    uint64_t retireEpoch = 0;
    bool scheduleDryAsOldCrossfade = false;
    double dryAsOldFadeTimeSec = 0.0;
    int transitionLatencyDeltaSamples = 0;

    validateDistinctRuntimeSlots("commitNewDSP.entry",
                                 activeDSP,
                                 sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                 nullptr);

    // Lock to ensure the check and commit are atomic with respect to new rebuild requests.
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        // 古いリクエストの結果であれば破棄 (Race condition対策)
        if (generation != rebuildGeneration.load(std::memory_order_relaxed))
        {
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
                retireDSP(newDSP);
                return;
            }
        }

        if (newDSP != nullptr && dspToTrash != nullptr)
        {
            bool needsCrossfade = false;
            double fadeTimeSec = 0.0;

            const bool oldHasIR = dspToTrash->convolverRt().isIRLoaded();
            const bool newHasIR = newDSP->convolverRt().isIRLoaded();
            const bool hasAudibleConvolverTransition = oldHasIR || newHasIR;
            const bool irPresenceChanged = (oldHasIR != newHasIR);

            if (hasAudibleConvolverTransition
                && newDSP->oversamplingFactor != dspToTrash->oversamplingFactor)
            {
                needsCrossfade = true;
                fadeTimeSec = std::max(fadeTimeSec, m_osFadeTimeSec.load(std::memory_order_relaxed));
            }

            if (hasAudibleConvolverTransition)
            {
                const uint64_t oldHash = dspToTrash->convolverRt().getStructuralHash();
                const uint64_t newHash = newDSP->convolverRt().getStructuralHash();
                if (oldHash != newHash)
                {
                    needsCrossfade = true;
                    const double baseIrFade = m_irFadeTimeSec.load(std::memory_order_relaxed);
                    if (irPresenceChanged)
                    {
                        fadeTimeSec = std::max(fadeTimeSec, std::clamp(baseIrFade, 0.001, 0.010));
                    }
                    else
                    {
                        fadeTimeSec = std::max(fadeTimeSec, baseIrFade);
                        fadeTimeSec = std::max(fadeTimeSec, m_irLengthFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_phaseFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_directHeadFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_nucFilterFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_tailFadeTimeSec.load(std::memory_order_relaxed));
                    }
                }
            }

            if (needsCrossfade)
            {
                const auto* runtimeGraph = getRuntimeGraphState();
                const auto* engineRuntime = getEngineRuntimeState();
                const bool hasFadingRuntime = (resolveFadingDSPFromRuntimePublish(runtimeGraph) != nullptr);
                const bool hasPendingCrossfade = dspCrossfadePending.load(std::memory_order_acquire)
                    || runtimeCrossfadePending(engineRuntime, runtimeGraph);
                const bool useDryAsOld = dspCrossfadeUseDryAsOld.load(std::memory_order_acquire)
                    || runtimeCrossfadeUseDryAsOld(engineRuntime, runtimeGraph);

                if (hasFadingRuntime || hasPendingCrossfade || useDryAsOld)
                {
                    diagLog("[DIAG] commitNewDSP: deferring commit until active fade settles newUuid="
                        + juce::String(static_cast<juce::int64>(newDSP->runtimeUuid))
                        + " oldUuid=" + juce::String(static_cast<juce::int64>(dspToTrash->runtimeUuid))
                        + " fadeSec=" + juce::String(fadeTimeSec, 3));
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
        g_currentEpoch.store(retireEpoch, std::memory_order_release);

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
                                     sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                     nullptr);

        // この世代の publish が完了したので outstanding rebuild 窓を閉じる。
        lastCommittedRebuildGeneration.store(generation, std::memory_order_release);
    }


    // 5. 初回IRロード時（旧DSPなし）: dry を旧信号としてクロスフェード予約
    if (dspToTrash == nullptr
        && newDSP != nullptr
        && newDSP->convolverRt().isIRLoaded()
        && !firstIrDryCrossfadeDone.load(std::memory_order_acquire))
    {
        // 初回のみ dry -> IR を明示的にフェードし、立ち上がりノイズを抑制する。
        scheduleDryAsOldCrossfade = true;
        dryAsOldFadeTimeSec = std::max(0.001, m_irFadeTimeSec.load(std::memory_order_relaxed));

        const bool convBypassedForLatency = m_currentConvBypass.load(std::memory_order_relaxed);
        const int newLatency = estimateRuntimeLatencyBaseRateSamples(newDSP, convBypassedForLatency);
        int dOld = std::min(newLatency, latencyBufSize - 1); // dry 側を遅延させて整合
        const int dNew = 0;
        latencyDelayOld.store(dOld, std::memory_order_release);
        latencyDelayNew.store(dNew, std::memory_order_release);
        latencyResetPending.store(true, std::memory_order_release);
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
        // --- クロスフェード判定・遅延整合処理 ---
        bool needsCrossfade = false;
        double fadeTimeSec = 0.0;

        if (newDSP != nullptr && dspToTrash != nullptr)
        {
            const bool oldHasIR = dspToTrash->convolverRt().isIRLoaded();
            const bool newHasIR = newDSP->convolverRt().isIRLoaded();
            const bool hasAudibleConvolverTransition = oldHasIR || newHasIR;
            const bool irPresenceChanged = (oldHasIR != newHasIR);

            // 1. オーバーサンプリング倍率変更
            if (hasAudibleConvolverTransition
                && newDSP->oversamplingFactor != dspToTrash->oversamplingFactor)
            {
                needsCrossfade = true;
                fadeTimeSec = std::max(fadeTimeSec, m_osFadeTimeSec.load(std::memory_order_relaxed));
            }

            // 2. 構造ハッシュ比較によるその他の変更検出
            // 両者とも IR 未ロード（実質 dry/passthrough）の場合、
            // 構造ハッシュ差だけでクロスフェードを発火させる必要はない。
            // 起動直後の設定同期で不要なフェード連打が起きるのを防ぐ。
            if (hasAudibleConvolverTransition)
            {
                const uint64_t oldHash = dspToTrash->convolverRt().getStructuralHash();
                const uint64_t newHash = newDSP->convolverRt().getStructuralHash();
                diagLog("[DIAG] commitNewDSP: hashes oldHash=" + juce::String((int64)oldHash) + " newHash=" + juce::String((int64)newHash) + " needsCF=" + juce::String((int)needsCrossfade));
                if (oldHash != newHash)
                {
                    needsCrossfade = true;
                    const double baseIrFade = m_irFadeTimeSec.load(std::memory_order_relaxed);

                    // IRの有無が切り替わる遷移（passthrough->IR / IR->passthrough）は
                    // 長時間フェードにすると二重処理時間が伸び、再生を圧迫しやすい。
                    // そのため短いフェード時間に制限し、他の長時間系は適用しない。
                    if (irPresenceChanged)
                    {
                        fadeTimeSec = std::max(fadeTimeSec, std::clamp(baseIrFade, 0.001, 0.010));
                    }
                    else
                    {
                        fadeTimeSec = std::max(fadeTimeSec, baseIrFade);
                        fadeTimeSec = std::max(fadeTimeSec, m_irLengthFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_phaseFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_directHeadFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_nucFilterFadeTimeSec.load(std::memory_order_relaxed));
                        fadeTimeSec = std::max(fadeTimeSec, m_tailFadeTimeSec.load(std::memory_order_relaxed));
                    }
                }
            }
            else
            {
                diagLog("[DIAG] commitNewDSP: skip crossfade for passthrough->passthrough");
            }

            // --- レイテンシ差・バッファ初期化はクロスフェード時のみ ---
            if (needsCrossfade)
            {
                const bool convBypassedForLatency = m_currentConvBypass.load(std::memory_order_relaxed);
                const int oldLatency = estimateRuntimeLatencyBaseRateSamples(dspToTrash, convBypassedForLatency);
                const int newLatency = estimateRuntimeLatencyBaseRateSamples(newDSP, convBypassedForLatency);
                const int targetLatency = std::max(oldLatency, newLatency);
                int dOld = targetLatency - oldLatency;
                int dNew = targetLatency - newLatency;
                dOld = std::min(dOld, latencyBufSize - 1);
                dNew = std::min(dNew, latencyBufSize - 1);
                latencyDelayOld.store(dOld, std::memory_order_release);
                latencyDelayNew.store(dNew, std::memory_order_release);
                // ★ resetはAudioThreadで1回だけ行う
                latencyResetPending.store(true, std::memory_order_release);
                transitionLatencyDeltaSamples = dOld - dNew;

                diagLog("[DIAG] commitNewDSP: latency align old="
                    + juce::String(static_cast<juce::int64>(dOld))
                    + " new=" + juce::String(static_cast<juce::int64>(dNew))
                    + " effectiveOld=" + juce::String(static_cast<juce::int64>(oldLatency))
                    + " effectiveNew=" + juce::String(static_cast<juce::int64>(newLatency))
                    + " convBypassed=" + juce::String(convBypassedForLatency ? 1 : 0));

                if (!oldHasIR && newHasIR)
                    dspCrossfadeStartDelayBlocks.store(std::max(0, m_crossfadeStartDelayBlocks.load(std::memory_order_relaxed)), std::memory_order_release);
                else
                    dspCrossfadeStartDelayBlocks.store(0, std::memory_order_release);
            }
            else
            {
                // クロスフェード不要時は絶対に遅延値・バッファを触らない
                dspCrossfadeStartDelayBlocks.store(0, std::memory_order_release);
            }

            // デフォルト値（fadeTimeSec==0なら30ms）
            if (fadeTimeSec <= 0.0)
                fadeTimeSec = 0.030;
        }

        // --- クロスフェードdeduplication・スナップショット ---
        if (needsCrossfade)
        {
            const auto* runtimeGraph = getRuntimeGraphState();
            const auto* engineRuntime = getEngineRuntimeState();
            const bool hasFadingRuntime = (resolveFadingDSPFromRuntimePublish(runtimeGraph) != nullptr);
            const bool atomicPendingCrossfade = dspCrossfadePending.load(std::memory_order_acquire);
            const bool hasPendingCrossfade = atomicPendingCrossfade
                || runtimeCrossfadePending(engineRuntime, runtimeGraph);
            const bool atomicUseDryAsOld = dspCrossfadeUseDryAsOld.load(std::memory_order_acquire);
            const bool useDryAsOld = atomicUseDryAsOld
                || runtimeCrossfadeUseDryAsOld(engineRuntime, runtimeGraph);
            const bool isFadingActive = hasFadingRuntime || hasPendingCrossfade || useDryAsOld;
            publishSmoothTransitionState(activeDSP,
                                         dspToTrash,
                                         fadeTimeSec,
                                         transitionLatencyDeltaSamples);
            jassert(!isFadingActive);
            startImmediateSmoothTransition(dspToTrash, fadeTimeSec);
        }
        else if (dspToTrash)
        {
            // クロスフェード不要時は即時解放
            retireRuntimeImmediately(dspToTrash);
            publishHardResetForCurrentDSP();
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
        pendingLearningMode.load(std::memory_order_acquire),
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
                                 sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire)),
                                 nullptr);
    diagLog("[DIAG] commitNewDSP: before sendChangeMessage");
    sendChangeMessage();
    diagLog("[DIAG] commitNewDSP: after sendChangeMessage");
}
#endif
