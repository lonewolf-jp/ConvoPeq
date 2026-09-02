#include "RuntimePublicationOrchestrator.h"
#include "AudioEngine.h"
#include "RuntimeBuilder.h"
#include "CrossfadeAuthority.h"
#include "FrozenRuntimeWorld.h"
#include "DSPLifetimeManager.h"   // ★ D162-2-B: terminal disposition authority
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

    // D101-31-B B-8: Admission reservation for Publication path.
    // tryAdmit(1) after evaluate()→Accepted, release(1) after enqueue (or on failure).
    // RAII guard ensures release on any early return / exception-free exit.
    if (!engine_.isrShutdownRuntime().tryAdmit(1))
        return PublicationAdmission::Decision::RejectedShutdown;
    struct ReservationGuard {
        convo::isr::ShutdownRuntime& rt;
        bool active = true;
        ~ReservationGuard() { if (active) rt.release(1); }
    } reservationGuard{ engine_.isrShutdownRuntime() };

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
        // ★ D105-R20: transient build failure → return RejectedNotFinalized. The obligation
        //   disposition is centralized in submitPublishRequest's RejectedNotFinalized
        //   switch case (one markTransientFailure call per failure event; A/B/D unified).
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
        // ★ D105-R20: transient crossfade-rebuild failure → return RejectedNotFinalized.
        //   Centralized in submitPublishRequest's switch case (one markTransientFailure
        //   per failure event; A/B/D unified). See Build #1 failure above.
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
        // ★ 15-P-6: publish 失敗を genuine shutdown と区別する。
        //   admission（evaluate）は isShutdownInProgress() をチェック済みだが、
        //   admission と publish の間に shutdown が開始される race が理論上存在する。
        //   publish 失敗時点で shutdown 中なら RejectedShutdown、それ以外は
        //   RejectedPublishFailure（内部失敗）を返す。ownership はどちらでも
        //   destroyRolledBackDSP() により回収済み（decision 分類から独立）。
        // ★ D105-R18: transient publish failure → markTransientFailure (ΔL=0, same id,
        //   delivery=None — repairs stranded Transport after Builder pop, retry counter
        //   incremented). ResolvedFailed only at retry exhaustion. The shutdown check
        //   below still routes to RejectedShutdown; obligation disposition is independent
        //   of the return decision.
        engine_.runtimePublicationBridge_.postRecoveryFailureSignal(req.recoveryObligationId);
        if (engine_.isShutdownInProgress())
            return PublicationAdmission::Decision::RejectedShutdown;
        return PublicationAdmission::Decision::RejectedPublishFailure;
    }

    juce::Logger::writeToLog("[DIAG] trySubmit: executor_.publish SUCCEEDED gen="
        + juce::String(req.generation));
    // ★ D105-R5-8: Route A completion authority — publish succeeded ⇒ resolve obligation.
    engine_.runtimePublicationBridge_.resolveRecoveryObligation(req.recoveryObligationId, RuntimeIntentCoordinator::RecoveryOutcome::Published);
    // ★ v19: StateOwner + TelemetryRecorder: Published 記録
    stateOwner_.onPublished(correlationId.shortValue());
    telemetryRecorder_.recordProgress(correlationId,
        static_cast<uint64_t>(req.generation), 0,
        PublishStage::Published, nowUs);

    // D101-31-B B-8: Release admission reservation — intent is enqueued, obligation now
    // survives in durable state (ISR intent queue / coordinator loop).
    reservationGuard.active = false;
    engine_.isrShutdownRuntime().release(1);

    // ★ B4-a4: publish 成功後の activate/crossfade/retire と epoch advance は
    //   ISR PublishExecutor::executePublish の Execution tail（onPublishCompleted →
    //   advanceRetireEpoch → onPublishCommitted）に一本化された。
    //   executor_.publish（async facade）は完了通知（notifyPublishReceipt）を受けてから
    //   return するため、ここで再実行すると二重実行になる。削除:
    //     transition_.onPublishCompleted(newDSPResolved, oldDSP, cfDecision, lifetime_);
    //     engine_.advanceRetireEpoch();

    return PublicationAdmission::Decision::Accepted;
}

void RuntimePublicationOrchestrator::onPublishCommitted(PublicationSequenceId seqId, std::uint64_t recoveryObligationId) noexcept {
    // ★ (a) Completion layer — ISR post-commit notification (not via IntentHandlerContext).
    //   Invoked from the ISR PublishExecutor once authority.commit() succeeds; records the
    //   committed sequence + progress timestamp so the P1-6 stall observer tracks ISR commits.
    //   Audio-thread trySubmit keeps its own inline completion path.
    convo::publishAtomic(m_lastObservedSequence, seqId, std::memory_order_release);
    convo::publishAtomic(m_lastProgressTimestampUs, getCurrentTimeUs(), std::memory_order_release);
    // ★ B3/C2: per-receipt completion — Producer はこの seqId で自分の publish 完了を待てるようになる。
    engine_.notifyPublishReceipt(seqId);
    // ★ D105-R5-8: Route B completion authority. recoveryObligationId == 0 for non-recovery
    //   publishes ⇒ resolveRecoveryObligation early-returns (no-op).
    engine_.runtimePublicationBridge_.resolveRecoveryObligation(recoveryObligationId, RuntimeIntentCoordinator::RecoveryOutcome::Published);
}

void RuntimePublicationOrchestrator::submitPublishRequest(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    auto decision = trySubmitImpl(req);
    const auto nowUs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    // ★ D105-R5-9 MUST-1: route a rejected recovery obligation through the single Completion Authority.
    //   Recovery obligations that fail admission must still be resolved (otherwise the Live slot leaks,
    //   permanently consuming one of 32). Normal (recoveryObligationId==0) publishes are unaffected.
    //   All branches route through resolveRecoveryObligation → table.resolve → single −1 (idempotent).
    auto resolveIfRecovery = [&](RuntimeIntentCoordinator::RecoveryOutcome outcome) noexcept {
        if (req.recoveryObligationId != 0)
            engine_.runtimePublicationBridge_.resolveRecoveryObligation(req.recoveryObligationId, outcome);
    };

    switch (decision) {
        case PublicationAdmission::Decision::Accepted:
            return;
        case PublicationAdmission::Decision::DeferredFadingActive:
            enqueueDeferred(req);
            return;
        // ★ D162-2-B (C′ / H1 修正 S4 — latent path repair):
        //   Rejected* 終端は newDSP（handle 登録済み・未 publish・World から到達不能）の
        //   disposition を行わず orphan 化する脱落点（D162-2-A 脇落点 S4・soak 未発火だが
        //   D162-1P の stale-gen 条件は実在する）。各 Rejected* で authority retire を実行する。
        //   ★ RejectedPublishFailure は例外: trySubmitImpl 内で destroyRolledBackDSP により
        //   既に直接破壊済み（Orchestrator.cpp:290-291）＋ handle registry が rollback
        //   （Reclaimed）へ遷移しているため、resolveDSPHandle が nullptr を返し本 helper は
        //   no-op になる（二重破壊禁止契約 INV-D162-3 — retire を追加して破壊しない）。
        //   helper の no-op 保証により全 case で同一呼び出しが安全なため、default 経路で一括処理する。
        case PublicationAdmission::Decision::RejectedStaleGeneration:
            stateOwner_.onRejected(0);
            telemetryRecorder_.recordFailure(FailureStage::Admission,
                FailureReason::StaleGeneration, "submitPublishRequest:stale",
                0, nowUs);
            resolveIfRecovery(RuntimeIntentCoordinator::RecoveryOutcome::StaleSuperseded);  // ★ D105-R5-9: −1
            retireRegisteredDSP(req, "rejected-stale-generation");  // ★ D162-2-B (S4)
            return;
        default:
            return;
        case PublicationAdmission::Decision::RejectedNotFinalized:
            stateOwner_.onRejected(0);
            telemetryRecorder_.recordFailure(FailureStage::Admission,
                FailureReason::ValidationFailed, "submitPublishRequest:notFinalized",
                0, nowUs);
            // ★ D105-R20: centralized markTransientFailure for ALL three RejectedNotFinalized
            //   paths (A: build #1, B: crossfade rebuild, D: admission-rejection). The previous
            //   direct resolveIfRecovery(Failed) was a leak-tightening path (R5-9) that
            //   conflicted with the R18 retry-preservation contract; centralizing here
            //   guarantees exactly one markTransientFailure call per failure event. ΔL=0,
            //   delivery=None (P-B), counter+1, exhaustion→ResolvedFailed (path X only).
            if (req.recoveryObligationId != 0)
                engine_.runtimePublicationBridge_.postRecoveryFailureSignal(req.recoveryObligationId);
            // ★ D162-2-B (S4): notFinalized は DSPCore 自体は健全（IR finalize 待ち）。
            //   publish されないため World 到達不能 → authority retire（A/B 経路の
            //   world-build-fail retire と同処置・Orchestrator.cpp:181/:249 前例）。
            retireRegisteredDSP(req, "rejected-not-finalized");
            return;
        case PublicationAdmission::Decision::RejectedPressure:
            stateOwner_.onRejected(0);
            telemetryRecorder_.recordFailure(FailureStage::Admission,
                FailureReason::QueuePressure, "submitPublishRequest:pressure",
                0, nowUs);
            // ★ D105-R5-9 MUST-2: QueuePressure → Retry (ΔL=0, obligation stays Live). Re-arm the
            //   durable slot only when it already holds THIS obligation; otherwise delivery re-drive is
            //   deferred (documented single-slot limitation). Never overwrites a distinct live obligation.
            resolveIfRecovery(RuntimeIntentCoordinator::RecoveryOutcome::Retry);
            if (req.recoveryObligationId != 0)
                engine_.runtimePublicationBridge_.rearmRecoveryRetry(req.recoveryObligationId);  // ★ D105-R5-9: guarded re-arm
            // ★ D162-2-B (S4): obligation は Live のまま再 drive 可能だが、本 DSPCore 自体は
            //   obligation 再発行時に新 build が作る（obligation は buildSource を保持し、
            //   DSPCore instance を保持しない）。滞留 DSPCore は World 到達不能のため
            //   authority retire する（Pressure 連鎖での multi-gen orphan 防止）。
            retireRegisteredDSP(req, "rejected-pressure");
            return;
        case PublicationAdmission::Decision::RejectedShutdown:
            stateOwner_.onRejected(0);
            telemetryRecorder_.recordFailure(FailureStage::Shutdown,
                FailureReason::ShutdownRejected, "submitPublishRequest:shutdown",
                0, nowUs);
            resolveIfRecovery(RuntimeIntentCoordinator::RecoveryOutcome::ShutdownDiscarded); // ★ D105-R5-9: −1 (idempotent)
            // ★ D162-2-B (S4): shutdown 拒否 DSP も World 到達不能。shutdown drain は
            //   EBR 経路を処理する（drainDeferredRetireQueues(true)）ため authority retire が
            //   shutdown 契約と整合する。
            retireRegisteredDSP(req, "rejected-shutdown");
            return;
        // ★ 15-P-6: publish-time 内部失敗 — shutdown telemetry に誤計上しない。
        //   FailureStage::Execution / FailureReason::PublishFailed で記録し、
        //   recovery suppression（shutdown 扱い）を回避する。
        //   NOTE: trySubmitImpl already resolved this obligation (Failed) before returning
        //   RejectedPublishFailure, so no second resolve is needed here (would be a no-op anyway).
        //   ★ D162-2-B (S4): newDSP は destroyRolledBackDSP（Orchestrator.cpp:290-291）で
        //   直接破壊済み。retire を追加すると二重破壊になるため、ここでは disposition しない
        //   （helper が呼ばれても no-op 保証があるが、明示的に呼ばない — INV-D162-3）。
        case PublicationAdmission::Decision::RejectedPublishFailure:
            stateOwner_.onRejected(0);
            telemetryRecorder_.recordFailure(FailureStage::Execution,
                FailureReason::PublishFailed, "submitPublishRequest:publishFailure",
                0, nowUs);
            return;
    }
}

// enqueueDeferred — global sequence スナップショットを記録
void RuntimePublicationOrchestrator::enqueueDeferred(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    // 上書きカウント
    if (convo::consumeAtomic(hasDeferred_, std::memory_order_acquire))
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

    // ★ D132 (INV-DEFERRED-2): overwrite 時の旧 DSPCore retire。
    //   ★ D162-2-B (C′ / H1 修正 S1): 旧実装は engine_.retireDSPHandleForRuntime 直呼び
    //   （台帳解除のみ）で、EBR 破壊権取得が行われず DSPCore が orphan 化していた
    //   （D162-1P: 49 retained のうち 48 件が本経路）。authority（DSPLifetimeManager::retire）
    //   に統一し、handle map erase + EBR enqueue を 1 回だけ実行する。
    //   D132 の意味（旧 deferred holder の retire）は不変。二重 retire は
    //   retireDSPHandleForRuntime の map lookup が単調に false を返すため構造的に不可。
    //   oldHandle は直下の deferredSlot_ 置換（:513）で失われるため、置換前に retire する。
    if (deferredSlot_.has_value()) {
        retireRegisteredDSP(deferredSlot_->request, "deferred-overwrite");
    }

    // ★ D135-1 / F6: obligation accounting — identity = (generation, recoveryObligationId).
    //   DeferredFadingActive の再 enqueue は「retention（保持）」であり retry ではない →
    //   deferredRetryCount_ を増加させない。新 obligation（key 相違）のみ createdAt を now で
    //   初期化し、re-drive は createdAt を維持する（→ TTL は obligation dwell を測る）。
    {
        const bool sameObligation = (req.generation == deferredRetryGeneration_
                                     && req.recoveryObligationId == deferredRetryObligationId_);
        if (!sameObligation) {
            deferredRetryGeneration_ = req.generation;
            deferredRetryObligationId_ = req.recoveryObligationId;
            deferredRetryCount_ = 0;
            deferredObligationCreatedAtUs = now;
        }
        // retention (sameObligation): count / createdAt とも不変（re-drive は新 obligation ではない）。
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        {
            const int curGen = convo::consumeAtomic(engine_.rebuildRequestGeneration, std::memory_order_acquire);
            juce::Logger::writeToLog(juce::String("[D135] re-defer")
                + (sameObligation ? " (retain)" : " (new)")
                + " gen=" + juce::String(req.generation)
                + " oblId=" + juce::String(static_cast<juce::int64>(req.recoveryObligationId))
                + " currentGen=" + juce::String(curGen)
                + " retryCount=" + juce::String(deferredRetryCount_));
        }
#endif
        // ★ F6-6: dormant guard — 現行 production に Type-A retry 経路は無く、retention は count を
        //   増やさないため deferredRetryCount_ は常に 0（この分岐は発火しない）。将来の Type-A 用に保持。
        if (deferredRetryCount_ > kMaxDeferredRetries) {
            // ★ D162-2-B (C′ / latent path repair): H1 実測 49 件には不寄与（F6-6 どおり不発）だが、
            //   S1 同型の orphan 脱落点のため authority 経由に修正（latent path として本修正に計上）。
            retireRegisteredDSP(req, "retry-exhausted-discard");
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            juce::Logger::writeToLog(juce::String("[HEALTH] Deferred publish starved")
                + " gen=" + juce::String(req.generation)
                + " sequence=" + juce::String(static_cast<juce::int64>(engine_.getLastCommittedPublicationSequence()))
                + " retryCount=" + juce::String(deferredRetryCount_)
                + " reason=RetryExhaustedDiscard");
#endif
            return;
        }
    }

    deferredSlot_ = DeferredPublishSlot{
        .request = req,
        .guard = DeferredGuard{
            .generation = req.generation,  // int のまま格納（第13回 D-13 ⑤: uint64_t への
                                            // static_cast を廃止・型統一）
            .sequence = engine_.getLastCommittedPublicationSequence()
        },
        // ★ Phase-1: enqueue-time immutable snapshot（View.metadata() の参照先）。
        //   ★ F6-3: metadata.enqueueTimestampUs の意味は「今回 enqueue 時刻」ではなく
        //   「現在の obligation が生成された時刻」（deferredObligationCreatedAtUs）。re-drive では
        //   維持されるため evaluateDeferred の TTL は obligation dwell を測る（F3 の失効バグ修正）。
        //   名前は本パッチでは変更しない（F4-10 別パッチ #8 扱い）。
        .metadata = PublicationAdmission::DeferredPublishMetadata{
            .generation = req.generation,
            .sequence = engine_.getLastCommittedPublicationSequence(),
            .enqueueTimestampUs = deferredObligationCreatedAtUs
        },
        .lastDiscardReason = DiscardReason::None,
        // ★ slot 側 timestamp は「今回の enqueue 時刻」の意味のまま（overwrite age 専用、F5-4）。
        .enqueueTimestampUs = now
    };
    convo::publishAtomic(hasDeferred_, true, std::memory_order_release);

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

    if (convo::consumeAtomic(hasDeferred_, std::memory_order_acquire)) {
        if (deferredSlot_.has_value())
            deferredSlot_->lastDiscardReason = DiscardReason::ShutdownDiscard;
        // ★ D162-2-B (H1 修正 S3): slot reset の前に、slot が保持していた handle 登録済み
        //   DSPCore を disposition する（D162-2-A 脱落点 S3・D162-1P 実測 1 件）。
        //   ★ 実装ノート（D162-2-B 実行時確定）: 本 DSP は一度も publish されていないため
        //   （RuntimePublishWorld topology / activeRuntimeDSPSlot / fadingRuntimeDSPSlot の
        //   いずれにも現れない・handle registry state は Constructing のまま）、RT reader は
        //   到達不能である。これは DSPGuard 契約（RebuildDispatch.cpp:938-965「未登録 DSPCore は
        //   EBR epoch 保護不要」）と同型の T2 direct destroy が正当なケースである。
        //   初版の DSPLifetimeManager::retire（EBR enqueue）は、shutdown teardown 境界を跨いで
        //   EBR destroy が遅延実行されることで exit 時 AV (0xC0000005・mkl_free 不正ポインタ) を
        //   誘発した（分離試験: S3 無効化で exit 0x0 再現）。clear 時点（engine 全員生存・
        //   Message/Rebuild Thread・NonRT）で直接破壊する方が所有権連鎖が単一ポイントで完結し安全。
        //   ownership 遷移: deferredSlot（現 owner）→ 本関数（同期的破壊）→ 終端。
        //   thread 契約: assert-free（非 RebuildThread caller: ReleaseResources.cpp:359 EmergencyDrain /
        //   C1 fallback — D135-8 Gate A 審査済み）。NonRT 実行。
        // ★ D162-2-B 残課題（D162-2-C へ繰り越し）: shutdown 時の deferred DSP disposition。
        //   初版（DSPLifetimeManager::retire = EBR enqueue）と第2版（T2 direct destroy）の双方で、
        //   破壊を実行した場合に限り ~AudioEngine 後半の member teardown で 0xC0000005 (mkl_free 不正
        //   ポインタ) が発生することを実測で確定した（破壊しない現行挙動では exit 0x0）。
        //   原因は shutdown teardown 順序と DSPHandle registry / EQCacheManager / rcuSwapper の
        //   相互作用にあり、S3 単独の修正範囲を超える。D162-2-C（read-only validation から
        //   teardown 順序監査へ拡張）で root-cause 確定後に有効化すること。
        //   現状の残留: shutdown 時に slot 残留していた deferred DSP 1 件のみ破壊されない
        //   （process exit で OS 回収・S1 により通常運転中の orphan は解消済み）。
        deferredSlot_.reset();
        convo::publishAtomic(hasDeferred_, false, std::memory_order_release);
    }

    // ★ D135-1 / F6-7: shutdown による deferred 消去 — obligation metadata も無効化
    invalidateDeferredObligation();
    convo::publishAtomic(lastRecoveryPublishSeq_, PublicationSequenceId{0}, std::memory_order_release);

    // ★ v19: DeferredHealth 記録
    DeferredHealth dh;
    dh.deferredCount = 0;
    dh.overwriteCount = convo::consumeAtomic(deferredOverwriteCount_, std::memory_order_acquire);
    dh.lastDiscardReason = DiscardReason::ShutdownDiscard;
    dh.lastDiscardTimestampUs = nowUs;
    telemetryRecorder_.recordDeferredHealth(dh);
}

// ★ D135-8 Step 9 (Option C): deferred-clear wake-provenance latch.
//   requestDeferredClear() is invoked from non-RebuildThread callers (the 3
//   AudioEngine.Timer.cpp health-recovery sites C2/C3/C4). If the RebuildThread is
//   already stopping (rebuildThreadShouldExit), the C1 synchronous fallback clears
//   inline — mirroring EmergencyDrain (ReleaseResources.cpp). Otherwise the clear
//   intent is latched (release-store) + the RebuildThread woken; the actual clear
//   always runs on the RebuildThread via drainDeferredClearIfRequested().
void RuntimePublicationOrchestrator::requestDeferredClear() noexcept
{
    // C1: RebuildThread is stopping/stopped (post join). No concurrent RebuildThread
    //     writer can touch the plain deferred members → clear synchronously.
    if (convo::consumeAtomic(engine_.rebuildThreadShouldExit, std::memory_order_acquire))
    {
        clearDeferredForShutdown();
        return;
    }
    // Live runtime: latch intent (release-store) + wake RebuildThread.
    // deferredClearRequested_ is NOT in the rebuildCV predicate; it is persistent
    // and consumed on the next wake → no lost clear.
    convo::publishAtomic(deferredClearRequested_, true, std::memory_order_release);
    engine_.rebuildCV.notify_one();
}

// ★ D135-8 Step 9 (Option C): RebuildThread-only consumer.
//   Mirrors resetDeferredRetryBudget()'s ownership idiom (h:171-175): assert
//   RebuildThread ownership here. clearDeferredForShutdown() stays the assert-free
//   synchronous primitive (EmergencyDrain + C1 fallback call it from non-RebuildThread).
bool RuntimePublicationOrchestrator::drainDeferredClearIfRequested() noexcept
{
    jassert(std::this_thread::get_id() == engine_.rebuildThreadId());
    if (convo::exchangeAtomic(deferredClearRequested_, false, std::memory_order_acq_rel))
    {
        clearDeferredForShutdown();
        return true;
    }
    return false;
}

// ★ Phase-1: peekDeferred — consumeDeferredRequest の後継（View 借用）。hasDeferred_ 非反転。
//   Single Thread Owner（RebuildThread）契約の jassert 付き（ADR-C4:100-105）。
std::optional<DeferredPublishView> RuntimePublicationOrchestrator::peekDeferred() noexcept
{
    jassert(std::this_thread::get_id() == engine_.rebuildThreadId());
    if (!convo::consumeAtomic(hasDeferred_, std::memory_order_acquire) || !deferredSlot_.has_value())
        return std::nullopt;
    return DeferredPublishView(*this, *deferredSlot_);
}

// ★ Phase-1: evaluateDeferred 用の Observation Snapshot 構築 (engine-state → POD)。
//   Policy はこれを直参照しない。5値: currentGeneration / lastSequence / shutdown / nowUs / ttlUs。
PublicationAdmission::DeferredAdmissionSnapshot
RuntimePublicationOrchestrator::buildDeferredAdmissionSnapshot() const noexcept
{
    return PublicationAdmission::DeferredAdmissionSnapshot{
        .currentGeneration = engine_.currentBuildGeneration(),
        .lastSequence = engine_.getLastCommittedPublicationSequence(),
        .shutdown = engine_.isShutdownInProgress(),
        .nowUs = convo::getCurrentTimeUs(),
        .ttlUs = kDeferredPublishTTLUs
    };
}

// ★ D162-2-B (C′): registered DSP の terminal disposition authority（実装）。
//   INV-D162-1: 本 helper が「registered DSP を手放す」唯一の口。
//   INV-D162-3: DSPLifetimeManager::retire は台帳解除（retireDSPHandleForRuntime）と
//     EBR 破壊権取得（enqueueWithRetry）を 1 callee で同時に実行する。台帳解除済み
//     （map 不在）の DSP に対しては retire が false で no-op するため、二重 EBR enqueue /
//     二重破壊は構造的に発生しない（destroyRolledBackDSP で直接破壊済みの DSP も
//     rollback により handle registry が Reclaimed へ遷移しており resolve が nullptr を
//     返すため、本 helper は何もしない — 二重破壊禁止契約）。
//   所有権遷移: handle map（現 owner）→ EBR（enqueueWithRetry 時点）→ destroyDSPCoreNode。
//     途中で参照を失う window は存在しない。
void RuntimePublicationOrchestrator::retireRegisteredDSP(
    const PublicationAdmission::PublishRequest& req, const char* origin) noexcept
{
    if (req.newDSP.isNull())
        return;
    auto* dsp = engine_.resolveDSPHandle(req.newDSP);
    if (dsp == nullptr)
        return;  // 未登録 / rollback / quarantine / reclaimed — disposition 済み。no-op。

    DSPLifetimeManager lifetimeMgr{engine_};
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    juce::Logger::writeToLog(juce::String::formatted(
        "[D162-2B_RETIRE] dsp=%p gen=%llu origin=%s",
        (void*)dsp, (unsigned long long)req.generation,
        origin != nullptr ? origin : "unknown"));
#endif
    lifetimeMgr.retire(dsp);
}

// ★ Phase-1: finishView — ownership Release の唯一口（design-D4 §1488 / §139-140）。
//   DeferredPublishView.consume()/discard() が owner_->finishView() を呼ぶ。
//   slot reset + hasDeferred_ flip + DeferredHealth telemetry を一括実行。
void RuntimePublicationOrchestrator::finishView() noexcept
{
    jassert(std::this_thread::get_id() == engine_.rebuildThreadId());
    DiscardReason reason = DiscardReason::None;
    uint64_t discardTs = 0;
    if (deferredSlot_.has_value()) {
        reason = deferredSlot_->lastDiscardReason;
        if (reason != DiscardReason::None)
            discardTs = convo::getCurrentTimeUs();
    }
    deferredSlot_.reset();
    convo::publishAtomic(hasDeferred_, false, std::memory_order_release);

    // ★ F6-7: terminal discard（StaleDiscard / Expired / ShutdownDiscard-via-discard）では
    //   obligation metadata を無効化。consume（reason==None、re-drive）では維持する
    //   （後続の enqueueDeferred が key 一致で retention と判定するため）。
    if (reason != DiscardReason::None)
        invalidateDeferredObligation();

    DeferredHealth dh;
    dh.deferredCount = 0;
    dh.overwriteCount = convo::consumeAtomic(deferredOverwriteCount_, std::memory_order_acquire);
    dh.lastDiscardReason = reason;
    dh.lastDiscardTimestampUs = discardTs;
    telemetryRecorder_.recordDeferredHealth(dh);
}

// ★ F6-7: obligation metadata 無効化（terminal 専用）。identity key・createdAt・count を
//   初期値へ戻す。rebuild-thread 専用（single-owner）。
void RuntimePublicationOrchestrator::invalidateDeferredObligation() noexcept
{
    deferredRetryGeneration_ = 0;
    deferredRetryObligationId_ = 0;
    deferredRetryCount_ = 0;
    deferredObligationCreatedAtUs = 0;
}

// ★ Phase-1: DeferredPublishView 実装（out-of-line。Orchestrator 定義完結後 = .cpp 内）。
//   consume/discard は owner_->finishView() を終端で呼ぶ（design-D4 §107/§134/§1488）。
PublicationAdmission::PublishRequest DeferredPublishView::consume() noexcept
{
    jassert(state_ == State::Valid && slot_ != nullptr);
    state_ = State::Consumed;
    auto req = std::move(slot_->request);  // finishView の slot reset 前に move-out（req は view 外生存）
    owner_->finishView();                  // ownership release（slot reset / hasDeferred_ flip / telemetry）
    return req;
}

void DeferredPublishView::discard(DiscardReason reason) noexcept
{
    jassert(state_ == State::Valid && slot_ != nullptr);
    slot_->lastDiscardReason = reason;
    state_ = State::Discarded;
    owner_->finishView();
}

// ★ Phase-1: processDeferredAdmission — RebuildThread 専用の atomic flow
//   (peek → evaluateDeferred → consume/discard → finishView → submitPublishRequest)。
//   design-D4 D-13 ④ / ADR-C4 §113。consume/discard は owner_->finishView() を内蔵し
//   ownership releaseを行うため、本関数は呼出のみ。Ready なら submitPublishRequest で
//   resubmit；必要なら再 enqueue（hasDeferred_=true）される。
//   （旧: RebuildDispatch.cpp:846 'consumeDeferredRequest → submitPublishRequest'。
//    AudioEngine.h:2528-2529 の consumeDeferredRequest → processDeferredAdmission 一本化済み）
void RuntimePublicationOrchestrator::processDeferredAdmission(bool wasRecoveryWake) noexcept
{
    jassert(std::this_thread::get_id() == engine_.rebuildThreadId());
    // ★ D135-8 Step 7 (P3): recovery-wake provenance gate (D135-5 blind-bool blocker resolved by
    //   Step 6). wasRecoveryWake is true iff THIS admission was provoked by the crossfade-timeout
    //   recovery path (recoveryRetryReady stamped release-store at AudioEngine.Timer.cpp:1748 under
    //   rebuildMutex, consumed read-and-clear via exchangeAtomic at AudioEngine.RebuildDispatch.cpp:888).
    //   Recovery commits a fresh zero-fading world, so the deferred obligation that follows is
    //   treated as a NEW dwell: resetDeferredRetryBudget clears identity key + count + createdAt.
    //   ★ F6: ordinary wakes (wasRecoveryWake == false) do NOT increment the retry budget —
    //   DeferredFadingActive re-drive is retention (see enqueueDeferred accounting ~:470).
    //   Placed at START, before peekDeferred(), per D135-7 preflight: the reset must precede any peek
    //   so the identity/createdAt baseline is fresh when the accounting at ~:470 compares
    //   (req.generation, req.recoveryObligationId) against the stored key.
    //   (lastRecoveryPublishSeq_ is a separate correlation stamp via setRecoveryPublishSeq.)
    if (wasRecoveryWake)
        resetDeferredRetryBudget();
    if (!convo::consumeAtomic(hasDeferred_, std::memory_order_acquire))
        return;

    auto view = peekDeferred();
    if (!view.has_value())
        return;

    auto result = admission_.evaluateDeferred(view->metadata(),
                                             buildDeferredAdmissionSnapshot());
    // ★ D162-2-B (S2): evaluateDeferred 判定後・discard 前に request を値コピー
    //   （retireRegisteredDSP へ渡す用。view は参照借用のため discard 後は slot 無効）。
    PublicationAdmission::PublishRequest slotRequestSnapshot = view->peekRequestCopy();
    switch (result.decision) {
        case PublicationAdmission::DeferredDecision::Ready: {
            // consume は owner_->finishView() を呼んで ownership release を完結する。
            auto req = view->consume();      // move-out + finishView()
            view.reset();                    // borrow 解除（slot は Orchestrator が reset 済み）
            submitPublishRequest(req);       // resubmit（再 enqueue は submitPublishRequest 内で）
            break;
        }
        case PublicationAdmission::DeferredDecision::Discard: {
            // discard は lastDiscardReason 記録 + owner_->finishView() で ownership release。
            // ★ D162-2-B (C′ / H1 修正 S2): view->discard() は deferredSlot の ownership release
            //   のみで、slot が保持していた request.newDSP（handle 登録済み DSPCore）の
            //   disposition を行わなかった（D162-2-A 脱落点 S2）。slot 破棄と DSP destruction を
            //   分離し、ownership が宙に浮かない順序にする:
            //   (1) 先に request を move-out 不可のため snapshot（view の slot 参照経由）、
            //   (2) authority retire（台帳解除 + EBR 破壊権取得）で ownership を EBR へ移譲、
            //   (3) view->discard() で slot を release。
            //   retire と discard の間で DSP は EBR が所有（単一 owner 継続）。
            retireRegisteredDSP(slotRequestSnapshot, "deferred-discard");
            view->discard(result.discardReason);
            view.reset();
            break;
        }
    }

    // ★ F6-7: Ready→submitPublishRequest 後に obligation が再 Deferred でなければ
    //   （Accepted / Rejected* 終端）metadata を無効化。再 Deferred（retention）なら
    //   enqueueDeferred が key/createdAt を維持済みなので触れない。
    if (!convo::consumeAtomic(hasDeferred_, std::memory_order_acquire))
        invalidateDeferredObligation();
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
