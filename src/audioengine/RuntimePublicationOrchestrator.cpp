#include "RuntimePublicationOrchestrator.h"
#include "AudioEngine.h"
#include "RuntimeBuilder.h"
#include "CrossfadeAuthority.h"
#include "FrozenRuntimeWorld.h"
#include <chrono>

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
// 局所 diagLog — 全ファイル統一パターン。
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
#endif

namespace convo::isr {

RuntimePublicationOrchestrator::RuntimePublicationOrchestrator(AudioEngine& engine, uint64_t engineInstanceId) noexcept
    : engine_(engine)
    , stateOwner_(engineInstanceId)  // ★ engineInstanceId 必須 (コンストラクタで設定)
    , telemetryRecorder_()
    , admission_()
    , executor_()
    , transition_(engine)
    , lifetime_(engine)
    , publicationReader(engine.getRetireRouter())
{
    telemetryRecorder_.setStateOwner(&stateOwner_);
    // ★ P1-6: 起動直後の誤検出防止（メンバ初期化子での順序問題を避けるためコンストラクタ本体で初期化）
    convo::publishAtomic(m_lastProgressTimestampUs, getCurrentTimeUs(), std::memory_order_release);
}

PublicationAdmission::Decision RuntimePublicationOrchestrator::trySubmit(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    return trySubmitImpl(req);
}

PublicationAdmission::Decision RuntimePublicationOrchestrator::trySubmitImpl(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    // ---- Phase 1: Admission ----
    // ★ evaluate() は必須。バイパス禁止。
    const convo::RuntimeReaderContext pubCtx{ publicationReader, convo::ObserveChannel::Publication };
    auto decision = admission_.evaluate(req, engine_, pubCtx);
    if (decision != PublicationAdmission::Decision::Accepted)
    {
        // Deferred/Rejected: caller が処理するため、ここでは retire しない
        return decision;
    }

    // ★ v19: StateOwner 記録 (State+Ledgerのみ)
    const auto correlationId = nextCorrelationId();
    stateOwner_.onSubmitted(correlationId.shortValue());

    const auto nowUs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    // ★ v19: TelemetryRecorder 記録 (進捗副産物)
    telemetryRecorder_.recordProgress(correlationId,
        static_cast<uint64_t>(req.generation), 0,
        PublishStage::Submitted, nowUs);

    // ---- Phase 2: Build + Publish (activate 前) ----
    // ★ activate はまだ行わない。まず world を build して publish する。
    // ★ Phase2: DSPHandle → DSPCore* 解決 (Execution Path Handle Normalization)
    auto* newDSPResolved = engine_.resolveDSPHandle(req.newDSP);
    // ★ [PR-4A] S-A2排除: Decision層の DSPCore* 直接参照を Handle 経由に変更
    auto oldHandle = engine_.dspHandleRuntime_.getActiveRuntimeDSPHandle();
    auto* oldDSP = (!oldHandle.isNull())
        ? engine_.resolveDSPHandle(oldHandle)
        : nullptr;
#if defined(JUCE_DEBUG) || defined(CONVO_CI_BUILD)
    if (req.newDSP.isNull()) {
        DBG("[DIAG] trySubmit: newDSP handle is NULL generation=" << req.generation);
    } else if (newDSPResolved == nullptr) {
        DBG("[DIAG] trySubmit: resolveDSPHandle failed slot=" << (int)req.newDSP.slot
            << " gen=" << (int)req.newDSP.generation << " reqGen=" << req.generation);
    }
#endif

    // Step 2a: Build world with default (HardReset) policy first, then
    // evaluate crossfade need, and rebuild with final Specification.
    // ★ work70-v8.3: Specification を先に組み立て、Post-build Mutation を排除。
    //   Builder は Specification を忠実に World に写像するのみ。
    auto worldBuilder = convo::RuntimeBuilder(engine_);
    // Step 2a-1: Create Specification with default HardReset
    convo::RuntimePublishSpecification spec;
    spec.topology.activeDSP = newDSPResolved;
    spec.topology.fadingDSP = oldDSP;
    spec.execution.transitionActive = false;          // HardReset: no transition
    spec.execution.transitionPolicy = static_cast<int>(convo::TransitionPolicy::HardReset);
    spec.execution.fadeTimeSec = 0.0;
    // PublicationSnapshotPart: previousCommittedSequence — Orchestrator が Coordinator から取得
    spec.publicationSnapshot.previousCommittedSequence = engine_.getLastCommittedPublicationSequence();
    // CrossfadeSnapshotPart: crossfadeRuntime の現在状態をスナップショット
    spec.crossfade.startDelayBlocks = engine_.crossfadeRuntime_.getStartDelayBlocks();
    spec.crossfade.dryHoldSamples = engine_.crossfadeRuntime_.getDryHoldSamples();
    spec.crossfade.dryScaleTarget = engine_.crossfadeRuntime_.getDryScaleTarget();
    spec.crossfade.firstIrDryCrossfadePending = engine_.crossfadeRuntime_.isFirstIrDryPending();
    // LatencyPart: engine atomic から収集
    spec.latency.latencyDelayOld = convo::consumeAtomic(engine_.latencyDelayOld, std::memory_order_acquire);
    spec.latency.latencyDelayNew = convo::consumeAtomic(engine_.latencyDelayNew, std::memory_order_acquire);
    // ★ v9.5 P1 phase2: currentRuntimeWorld — Orchestrator が現在の Published World を取得し、
    //   Builder はこれを使って makeEngineRuntimeState() を呼ぶ（Runtime Query 完了済み）
    spec.currentRuntimeWorld = engine_.observePublishedWorld();
    // ProcessingPart: fill from sealedSnapshot (P0 — Builder の暗黙入力を排除)
    {
        const auto& inp = req.sealedSnapshot.buildInput;
        spec.processing.processingOrder = inp.processingOrder;
        spec.processing.eqBypassed = inp.eqBypassed;
        spec.processing.convBypassed = inp.convBypassed;
        spec.processing.softClipEnabled = inp.softClipEnabled;
        spec.processing.saturationAmount = static_cast<float>(inp.saturationAmount);
        spec.processing.inputHeadroomGain = static_cast<float>(inp.inputHeadroomGain);
        spec.processing.outputMakeupGain = static_cast<float>(inp.outputMakeupGain);
        spec.processing.convolverInputTrimGain = static_cast<float>(inp.convolverInputTrimGain);
        spec.processing.autoGainStagingEnabled = inp.autoGainStagingEnabled;
        // ★ Sync RoutingPart for backward compat — ProcessingPart が一次情報源
        spec.routing.processingOrder = inp.processingOrder;
        spec.routing.eqBypassed = inp.eqBypassed;
        spec.routing.convBypassed = inp.convBypassed;
    }

    // ★ v14.0: AnalysisPart — BuildAnalysis からコピー
    // ★ v14.37: verifyBuildBundle で BuildAnalysis + BuildDiagnostics + OversamplingResult + Snapshot の整合性を一括検証
    {
        const auto& ana = req.buildAnalysis;
        const auto& diag = req.buildDiagnostics;
        [[maybe_unused]] const auto& osResult = req.oversamplingResult;
        jassert(convo::verifyBuildBundle(ana, diag, osResult, req.sealedSnapshot));
        jassert(convo::verifyDiagnostics(diag));
        spec.analysis.eqMaxGainDb = ana.eqMaxGainDb;
        spec.analysis.eqMaxQ = ana.eqMaxQ;
        spec.analysis.irFreqPeakGainDb = ana.irFreqPeakGainDb;
        spec.analysis.additionalAttenuationDb = ana.additionalAttenuationDb;
        spec.analysis.analysisVersion = diag.analysisVersion;
    }

    // ★ v9.7 P7-A1: RetirePart — engine atomic から収集（sealedSnapshot には含まれないため）
    spec.retire.retireQueueDepth = convo::consumeAtomic(engine_.retireQueueDepth_, std::memory_order_acquire);
    // ★ v9.7 P7-A2: AdaptivePart — engine atomic から収集
    {
        const int bankIdx = convo::consumeAtomic(engine_.currentAdaptiveCoeffBankIndex, std::memory_order_acquire);
        spec.adaptive.coeffBankIndex = bankIdx;
        if (bankIdx >= 0 && bankIdx < static_cast<int>(kNumAdaptiveCoeffBanks))
        {
            const auto& bank = engine_.getAdaptiveCoeffBankForIndex(bankIdx);
            spec.adaptive.coeffGeneration = convo::consumeAtomic(bank.generation, std::memory_order_acquire);
        }
    }

    // Step 2a-2: Build preliminary world for crossfade evaluation

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    diagLog("[DIAG_AUTH] CoordExit gen=" + juce::String(req.generation)
        + " transitionActive=" + juce::String(static_cast<int>(spec.execution.transitionActive))
        + " currentUuid=" + juce::String(static_cast<juce::int64>(newDSPResolved ? newDSPResolved->runtimeUuid : 0))
        + " nextUuid=" + juce::String(static_cast<juce::int64>(oldDSP ? oldDSP->runtimeUuid : 0))
        + " spec.fadingRuntimeUuid=" + juce::String(static_cast<juce::int64>(spec.topology.fadingDSP ? spec.topology.fadingDSP->runtimeUuid : 0)));
#endif

    auto worldOwner = worldBuilder.buildRuntimePublishWorld(&req.sealedSnapshot, spec);

    if (!worldOwner) {
        // Build failed: retire new DSP, keep old world
        if (!req.newDSP.isNull())
            lifetime_.retire(newDSPResolved);
        stateOwner_.onExecutorFailed(correlationId.shortValue());
        telemetryRecorder_.recordProgress(correlationId,
            static_cast<uint64_t>(req.generation), 0,
            PublishStage::Built, nowUs);
        telemetryRecorder_.recordFailure(FailureStage::Execution,
            FailureReason::PublishFailed, "trySubmit:build",
            correlationId.shortValue(), nowUs);
        return PublicationAdmission::Decision::RejectedNotFinalized;
    }

    stateOwner_.onBuilt(correlationId.shortValue());
    telemetryRecorder_.recordProgress(correlationId,
        static_cast<uint64_t>(req.generation), 0,
        PublishStage::Built, nowUs);

    // Step 2b: Crossfade decision using RuntimeWorld projection values + Policy
    const auto* oldWorld = engine_.observePublishedWorld();
    if (oldWorld == nullptr)
    {
        DBG("[DIAG] trySubmit: oldWorld is null - skipping crossfade evaluation, proceeding directly");
    }

    convo::isr::CrossfadeAuthority::Decision cfDecision {};
    if (oldWorld != nullptr)
    {
        auto policy = engine_.makeCrossfadePolicy();
        CrossfadeAuthority crossfade;
        cfDecision = crossfade.evaluate(*oldWorld, *worldOwner, policy);
    }
    else
    {
        cfDecision.needsCrossfade = false;
        cfDecision.fadeTimeSec = 0.0;
        cfDecision.oldHasIR = false;
        cfDecision.newHasIR = worldOwner->dspProjection.irLoaded;
    }

    // HealthState Critical 時は crossfade を強制抑制
    {
        auto ref = engine_.getHealthStateRef();
        if (ref) {
            auto health = convo::consumeAtomic(*ref, std::memory_order_acquire);
            if (health == convo::ISRHealthState::Critical) {
                cfDecision.needsCrossfade = false;
                cfDecision.fadeTimeSec = 0.0;
            }
        }
    }

    // Step 2c: Update Specification with crossfade decision (NOT the world! — Post-build Mutation 排除)
    if (cfDecision.needsCrossfade && oldDSP != nullptr)
    {
        spec.execution.transitionPolicy = static_cast<int>(convo::TransitionPolicy::SmoothOnly);
        spec.execution.transitionActive = true;
        spec.execution.fadeTimeSec = cfDecision.fadeTimeSec;
        // ★ work70-v8.3: hasFadingRuntime は Topology から導出。ここでは設定不要。
    }

    // ★ work70-v8.3: Rebuild world from finalized Specification (single build)
    if (cfDecision.needsCrossfade && oldDSP != nullptr)
    {
        worldOwner = worldBuilder.buildRuntimePublishWorld(&req.sealedSnapshot, spec);
        if (!worldOwner) {
            if (!req.newDSP.isNull())
                lifetime_.retire(newDSPResolved);
            stateOwner_.onExecutorFailed(correlationId.shortValue());
            telemetryRecorder_.recordProgress(correlationId,
                static_cast<uint64_t>(req.generation), 0,
                PublishStage::Built, nowUs);
            telemetryRecorder_.recordFailure(FailureStage::Execution,
                FailureReason::PublishFailed, "trySubmit:rebuild",
                correlationId.shortValue(), nowUs);
            return PublicationAdmission::Decision::RejectedNotFinalized;
        }
    }

    stateOwner_.onValidated(correlationId.shortValue());
    telemetryRecorder_.recordProgress(correlationId,
        static_cast<uint64_t>(req.generation), 0,
        PublishStage::Validated, nowUs);

    // ★ Phase4: worldOwner → FrozenRuntimeWorld wrap → publish
    // ★ v8.3: Builder は const World を返すが、FrozenRuntimeWorld の releaseState() が
    //   非 const を要求するため const_cast を使用。seal 後は Coordinator 内で immutable。
    auto frozen = convo::aligned_make_unique<convo::FrozenRuntimeWorld>(
        convo::aligned_unique_ptr<RuntimeState>(
            const_cast<RuntimeState*>(worldOwner.release())));
    // ★ B4: oldHandle = current active DSP handle を渡し、Rebuild (#7) の retire 意図を伝搬する。
    // ★ CoordinatorLoop 上の deferred resubmit（waitForReceipt=false）では fire-and-forget
    //   publish を使用: receipt は同スレッドの processIntent でしか配送されないため、
    //   同期 wait は自己待ち（最大250msストール）になる。enqueue 済み + 所有権移譲済みなので
    //   次 tick で executePublish が commit する。
    auto result = executor_.publish(engine_, std::move(frozen), req.newDSP, oldHandle);
    if (result != PublishResult::Success) {
        juce::Logger::writeToLog("[DIAG] trySubmit: executor_.publish FAILED gen="
            + juce::String(req.generation)
            + " result=" + juce::String(static_cast<int>(result)));
        // publish 失敗: activate/crossfade/retire は一切行わない
        // ★ work70 Phase2: commitRuntimePublication の ScopeExit が Handle を
        //   rollback 済み（Reclaimed）。したがって retireDSPHandleForRuntime は
        //   false を返すため lifetime_.retire() は無効。
        //   代わりに destroyRolledBackDSP() で未公開 DSPCore を直接破棄する。
        if (newDSPResolved != nullptr)
            lifetime_.destroyRolledBackDSP(newDSPResolved);
        // ★ v19: StateOwner + TelemetryRecorder 記録
        stateOwner_.onExecutorFailed(correlationId.shortValue());
        telemetryRecorder_.recordFailure(FailureStage::Execution,
            FailureReason::PublishFailed, "trySubmit:publish",
            correlationId.shortValue(), nowUs);
        telemetryRecorder_.recordProgress(correlationId,
            static_cast<uint64_t>(req.generation), 0,
            PublishStage::Published, nowUs);
        return PublicationAdmission::Decision::RejectedShutdown;
    }

    juce::Logger::writeToLog("[DIAG] trySubmit: executor_.publish SUCCEEDED gen="
        + juce::String(req.generation));
    // ★ v19: StateOwner + TelemetryRecorder: Published 記録
    stateOwner_.onPublished(correlationId.shortValue());
    telemetryRecorder_.recordProgress(correlationId,
        static_cast<uint64_t>(req.generation), 0,
        PublishStage::Published, nowUs);

    // ★ B4-a4: publish 成功後の activate/crossfade/retire と epoch advance は
    //   ISR PublishExecutor::executePublish の Execution tail（onPublishCompleted →
    //   advanceRetireEpoch → onPublishCommitted）に一本化された。
    //   executor_.publish（async facade）は完了通知（notifyPublishReceipt）を受けてから
    //   return するため、ここで再実行すると二重実行になる。削除:
    //     transition_.onPublishCompleted(newDSPResolved, oldDSP, cfDecision, lifetime_);
    //     engine_.advanceRetireEpoch();

    return PublicationAdmission::Decision::Accepted;
}

void RuntimePublicationOrchestrator::onPublishCommitted(PublicationSequenceId seqId) noexcept {
    // ★ (a) Completion layer — ISR post-commit notification (not via IntentHandlerContext).
    //   Invoked from the ISR PublishExecutor once authority.commit() succeeds; records the
    //   committed sequence + progress timestamp so the P1-6 stall observer tracks ISR commits.
    //   Audio-thread trySubmit keeps its own inline completion path.
    convo::publishAtomic(m_lastObservedSequence, seqId, std::memory_order_release);
    convo::publishAtomic(m_lastProgressTimestampUs, getCurrentTimeUs(), std::memory_order_release);
    // ★ B3/C2: per-receipt completion — Producer はこの seqId で自分の publish 完了を待てるようになる。
    engine_.notifyPublishReceipt(seqId);
}

void RuntimePublicationOrchestrator::submitPublishRequest(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    auto decision = trySubmitImpl(req);
    const auto nowUs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    switch (decision) {
        case PublicationAdmission::Decision::Accepted:
            return;
        case PublicationAdmission::Decision::DeferredFadingActive:
            enqueueDeferred(req);
            return;
        case PublicationAdmission::Decision::RejectedStaleGeneration:
            stateOwner_.onRejected(0);
            telemetryRecorder_.recordFailure(FailureStage::Admission,
                FailureReason::StaleGeneration, "submitPublishRequest:stale",
                0, nowUs);
            return;
        default:
            return;
        case PublicationAdmission::Decision::RejectedNotFinalized:
            stateOwner_.onRejected(0);
            telemetryRecorder_.recordFailure(FailureStage::Admission,
                FailureReason::ValidationFailed, "submitPublishRequest:notFinalized",
                0, nowUs);
            return;
        case PublicationAdmission::Decision::RejectedPressure:
            stateOwner_.onRejected(0);
            telemetryRecorder_.recordFailure(FailureStage::Admission,
                FailureReason::QueuePressure, "submitPublishRequest:pressure",
                0, nowUs);
            return;
        case PublicationAdmission::Decision::RejectedShutdown:
            stateOwner_.onRejected(0);
            telemetryRecorder_.recordFailure(FailureStage::Shutdown,
                FailureReason::ShutdownRejected, "submitPublishRequest:shutdown",
                0, nowUs);
            return;
    }
}

// enqueueDeferred — global sequence スナップショットを記録
void RuntimePublicationOrchestrator::enqueueDeferred(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    // 上書きカウント
    if (hasDeferred_.load(std::memory_order_acquire))
        convo::fetchAddAtomic(deferredOverwriteCount_, uint64_t{1},
            std::memory_order_release);

    const auto now = convo::getCurrentTimeUs();

    // 上書き時は滞留時間を maxDeferredAgeMs に反映
    if (deferredSlot_.has_value()) {
        const uint64_t ageMs = (now - deferredSlot_->enqueueTimestampUs) / 1000;
        uint64_t currentMax = convo::consumeAtomic(maxDeferredAgeMs_,
            std::memory_order_acquire);
        while (ageMs > currentMax) {
            if (convo::compareExchangeAtomic(maxDeferredAgeMs_, currentMax,
                    ageMs, std::memory_order_acq_rel,
                    std::memory_order_acquire))
                break;
        }
    }

    deferredSlot_ = DeferredPublishSlot{
        .request = req,
        .guard = DeferredGuard{
            .generation = static_cast<uint64_t>(req.generation),
            .sequence = engine_.getLastCommittedPublicationSequence()
        },
        .lastDiscardReason = DiscardReason::None,
        .enqueueTimestampUs = now
    };
hasDeferred_.store(true, std::memory_order_release);

    // ★ v19: DeferredHealth 記録
    DeferredHealth dh;
    dh.deferredCount = 1;
    dh.oldestDeferredAgeMs = 0;  // 新規enqueue
    dh.overwriteCount = convo::consumeAtomic(deferredOverwriteCount_, std::memory_order_acquire);
    dh.lastDiscardReason = DiscardReason::None;
    telemetryRecorder_.recordDeferredHealth(dh);
}

// ★ C-2.2: shutdown 時に deferred publish を強制消去
void RuntimePublicationOrchestrator::clearDeferredForShutdown() noexcept
{
    const auto nowUs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    if (hasDeferred_.load(std::memory_order_acquire)) {
        if (deferredSlot_.has_value())
            deferredSlot_->lastDiscardReason = DiscardReason::ShutdownDiscard;
        deferredSlot_.reset();
        hasDeferred_.store(false, std::memory_order_release);
    }

    // ★ v19: DeferredHealth 記録
    DeferredHealth dh;
    dh.deferredCount = 0;
    dh.overwriteCount = convo::consumeAtomic(deferredOverwriteCount_, std::memory_order_acquire);
    dh.lastDiscardReason = DiscardReason::ShutdownDiscard;
    dh.lastDiscardTimestampUs = nowUs;
    telemetryRecorder_.recordDeferredHealth(dh);
}

// ★ A-2.5: DrainAudit 用 — deferred publish 最長滞留時間
uint64_t RuntimePublicationOrchestrator::getMaxDeferredAgeMs() const noexcept
{
    return convo::consumeAtomic(maxDeferredAgeMs_, std::memory_order_acquire);
}

// ★ C-2.1: 監査用 — deferred overwrite 回数
std::uint64_t RuntimePublicationOrchestrator::deferredOverwriteCount() const noexcept
{
    return convo::consumeAtomic(deferredOverwriteCount_, std::memory_order_acquire);
}

// ── CorrelationId 採番 ──
CorrelationId RuntimePublicationOrchestrator::nextCorrelationId() noexcept
{
    const auto cid = telemetryRecorder_.nextCorrelationId(stateOwner_.state().engineInstanceId);
    stateOwner_.setLastCorrelationId(cid);
    return cid;
}

// ── 健全性スナップショット ──
void RuntimePublicationOrchestrator::publishHealthSnapshot(uint64_t externalReclaimedCount) noexcept
{
    const auto& state = stateOwner_.state();
    const auto nowUs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    OrchestratorHealthSnapshot snapshot;
    snapshot.submittedCount = state.progress.submittedCount;
    snapshot.publishedCount = state.progress.publishedCount;
    snapshot.retiredCount = state.progress.retiredCount;
    snapshot.reclaimedCount = externalReclaimedCount;  // ★ C-3: EpochDomain から受け取る
    snapshot.executorQueueDepth = state.progress.executorQueueDepth;
    snapshot.lastProgressTimestampUs = state.progress.lastProgressTimestampUs;
    snapshot.stuckStage = state.progress.detectStuckStage();
    snapshot.timestampUs = nowUs;

    telemetryRecorder_.recordHealth(snapshot);
}

} // namespace convo::isr
