#include <JuceHeader.h>
#include "AudioEngine.h"
#include "RuntimePublicationValidator.h"
#include "RuntimePublicationOrchestrator.h"

namespace {
void diagLog(const juce::String& message)
{
#if defined(JUCE_DEBUG) || defined(CONVO_CI_BUILD)
    DBG(message);
    juce::Logger::writeToLog(message);
#else
    juce::ignoreUnused(message);
#endif
}

[[nodiscard]] inline bool validateSemanticCompleteness(const RuntimePublishWorld& world) noexcept
{
    if (world.schemaVersion != convo::isr::kRuntimeSemanticSchemaVersion)
        return false;

    if (world.metadata.schemaVersion != world.schemaVersion)
        return false;

    if (world.metadata.publicationSequence != world.publication.sequenceId)
        return false;

    if (!RuntimeState::validateDescriptorSet()
        || !convo::isr::PublicationSemantic::validateDescriptorSet())
        return false;

    if (world.generation == 0
        || world.generationSemantic.runtimeGeneration == 0
        || world.publication.sequenceId == 0
        || world.publication.epoch == 0
        || world.publication.mappedRuntimeGeneration == 0)
        return false;

    if (world.generationSemantic.runtimeGeneration != world.generation
        || world.publication.mappedRuntimeGeneration != world.generation)
        return false;

    if (world.projectionFreshness.projectionGeneration != world.publication.mappedRuntimeGeneration)
        return false;

    if (world.projectionFreshness.projectionGeneration != world.generation
        || world.projectionFreshness.projectionRevision != world.generation)
        return false;

    return true;
}

[[nodiscard]] inline bool validateRuntimeGraphAuthorityContract(const RuntimePublishWorld& world) noexcept
{
    // 3.2.9: RuntimeGraph の Authoritative フィールドは RuntimeWorld の
    // Semantic 構造体に移管されたため、graph との一致検証は不要。
    // RuntimeGraph は Projection + Diagnostic のみを保持する。
    if (!convo::RuntimeGraph::validateDescriptorSet())
        return false;

    if (!convo::RuntimeGraph::validateDecisionCoverageContract())
        return false;

    const bool hasGraphActiveNode = (world.graph.activeNode != nullptr);
    const bool hasGraphFadingNode = (world.graph.fadingNode != nullptr);
    const bool hasTopologyUuid = (world.topology.runtimeUuid != 0);

    if (hasGraphActiveNode != hasTopologyUuid)
    {
        juce::Logger::writeToLog("[AUTH_CONTRACT] FAIL activeNode=" + juce::String(static_cast<int64_t>(reinterpret_cast<intptr_t>(world.graph.activeNode)))
            + " topologyUuid=" + juce::String(static_cast<juce::int64>(world.topology.runtimeUuid))
            + " hasGraphActiveNode=" + juce::String(static_cast<int>(hasGraphActiveNode))
            + " hasTopologyUuid=" + juce::String(static_cast<int>(hasTopologyUuid)));
        return false;
    }

    if (hasGraphFadingNode != world.topology.hasFadingRuntime)
    {
        juce::Logger::writeToLog("[AUTH_CONTRACT] FAIL fadingNode=" + juce::String(static_cast<int64_t>(reinterpret_cast<intptr_t>(world.graph.fadingNode)))
            + " hasFading=" + juce::String(static_cast<int>(world.topology.hasFadingRuntime)));
        return false;
    }

    if (world.execution.transitionActive != world.topology.hasFadingRuntime)
    {
        juce::Logger::writeToLog("[AUTH_CONTRACT] FAIL transitionActive=" + juce::String(static_cast<int>(world.execution.transitionActive))
            + " hasFading=" + juce::String(static_cast<int>(world.topology.hasFadingRuntime)));
        return false;
    }

    return true;
}

[[nodiscard]] inline bool hasEquivalentTransitionSemantic(const RuntimePublishWorld& world) noexcept
{
    return world.execution.transitionActive == world.topology.hasFadingRuntime;
}

inline void forceSemanticTransactionState(std::atomic<std::uint8_t>& state,
                                          convo::isr::SemanticTransactionState next) noexcept
{
    convo::publishAtomic(state,
                         static_cast<std::uint8_t>(next),
                         std::memory_order_release);
}

[[nodiscard]] inline bool transitionSemanticTransactionState(std::atomic<std::uint8_t>& state,
                                                             convo::isr::SemanticTransactionState next) noexcept
{
    auto observedRaw = convo::consumeAtomic(state, std::memory_order_acquire);
    for (;;)
    {
        const auto from = static_cast<convo::isr::SemanticTransactionState>(observedRaw);
        if (!convo::isr::isValidSemanticTransactionTransition(from, next))
            return false;

        const auto desiredRaw = static_cast<std::uint8_t>(next);
        if (convo::compareExchangeAtomic(state,
                                         observedRaw,
                                         desiredRaw,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
            return true;
    }
}
}  // namespace

[[nodiscard]] bool AudioEngine::runPublicationPrecheckNonRt(const RuntimePublishWorld& world) noexcept
{
    // Delegate pure validation to RuntimePublicationValidator (Sprint-4 P3-A)
    static const iso::audio_engine::RuntimePublicationValidator validator;

    const auto validationResult = validator.validatePublication(world);
    if (!validationResult.isValid) {
        diagLog(juce::String("[DIAG] runPublicationPrecheckNonRt: validator reject reason=\"")
            + juce::String(validationResult.errorMessage)
            + " generation=" + juce::String(static_cast<juce::int64>(world.generation))
            + " seq=" + juce::String(static_cast<juce::int64>(world.publication.sequenceId))
            + " runtimeUuid=" + juce::String(static_cast<juce::int64>(world.topology.runtimeUuid)));
        return false;
    }

    forceSemanticTransactionState(semanticTransactionState_, convo::isr::SemanticTransactionState::Building);

    const auto rejectWithEvidence = [this, &world](const char* reason) noexcept {
        diagLog("[DIAG] runPublicationPrecheckNonRt: reject reason="
            + juce::String(reason != nullptr ? reason : "unknown")
            + " generation=" + juce::String(static_cast<juce::int64>(world.generation))
            + " seq=" + juce::String(static_cast<juce::int64>(world.publication.sequenceId))
            + " runtimeUuid=" + juce::String(static_cast<juce::int64>(world.topology.runtimeUuid)));
        if (!transitionSemanticTransactionState(semanticTransactionState_, convo::isr::SemanticTransactionState::Rejected))
        {
            forceSemanticTransactionState(semanticTransactionState_, convo::isr::SemanticTransactionState::Rejected);
        }
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        debugRuntime_.validateOwnershipClosure();
        emitEvidenceTickNonRt(true);
        return false;
    };

    // Stage 1: semantic completeness.
    if (!validateSemanticCompleteness(world))
        return rejectWithEvidence("semantic_completeness");

    // Stage 1.5: RuntimeGraph authority contract (fail-closed).
    if (!validateRuntimeGraphAuthorityContract(world))
        return rejectWithEvidence("runtime_graph_authority_contract");

    if (!transitionSemanticTransactionState(semanticTransactionState_, convo::isr::SemanticTransactionState::Validated))
        return rejectWithEvidence("semantic_state_validated_transition");

    // Stage 2: semantic validity.
    if (!convo::isr::isValidRoutingSemantic(world.routing)
        || !convo::isr::isValidExecutionSemantic(world.execution))
        return rejectWithEvidence("routing_or_execution_semantic_invalid");

    if (world.publication.previousSequenceId >= world.publication.sequenceId)
        return rejectWithEvidence("publication_sequence_non_monotonic");

    // Stage 3: runtime admission — shutdown check (consolidated from acceptsRuntimePublication)
    if (isShutdownInProgress())
        return rejectWithEvidence("publish_shutdown_in_progress");

    if (!hasEquivalentTransitionSemantic(world))
        return rejectWithEvidence("transition_semantic_mismatch");

    const auto lastCommittedGeneration = convo::consumeAtomic(lastCommittedRuntimeGeneration_, std::memory_order_acquire);
    const auto lastCommittedSequence = convo::consumeAtomic(lastCommittedPublicationSequence_, std::memory_order_acquire);
    if (lastCommittedGeneration != 0 && world.generation <= lastCommittedGeneration)
    {
        convo::publishAtomic(lastDroppedGeneration_, world.generation, std::memory_order_release);
        return rejectWithEvidence("generation_not_monotonic");
    }

    if (lastCommittedSequence != 0 && world.publication.sequenceId <= lastCommittedSequence)
    {
        convo::publishAtomic(lastDroppedGeneration_, world.generation, std::memory_order_release);
        return rejectWithEvidence("sequence_not_monotonic");
    }

    if (world.topology.hasFadingRuntime)
    {
        if (world.topology.fadingRuntimeUuid == 0
            || world.topology.fadingRuntimeUuid == world.topology.runtimeUuid)
            return rejectWithEvidence("invalid_fading_topology_identity");
    }
    else if (world.topology.fadingRuntimeUuid != 0)
    {
        return rejectWithEvidence("unexpected_fading_uuid_without_flag");
    }

    const bool hasTransitionNext = world.topology.hasFadingRuntime;
    if (hasTransitionNext)
    {
        if (world.execution.crossfadeStartDelayBlocks < 0
            || world.execution.crossfadeDryHoldSamples < 0)
            return rejectWithEvidence("invalid_crossfade_delay_values");
    }

    if (world.overlap.fadeTimeSec < 0.0
        || world.overlap.dryScaleTarget < 0.0)
        return rejectWithEvidence("invalid_overlap_values");

    if (!hasTransitionNext
        && world.overlap.firstIrDryCrossfadePending)
        return rejectWithEvidence("dry_crossfade_pending_without_transition");

    if (!world.isFrozen())
    {
        return rejectWithEvidence("world_not_frozen");
    }

    if (!world.isSealedRecursively())
    {
        return rejectWithEvidence("world_not_sealed_recursively");
    }

    const bool hasActive = (world.topology.runtimeUuid != 0);
    const bool hasFading = world.topology.hasFadingRuntime;

    if (!hasActive && !hasFading && !hasTransitionNext)
        return true;

    convo::isr::PayloadClosureDescriptor closure{};
    closure.closureId = static_cast<uint32_t>((world.generation != 0)
        ? world.generation
        : 1u);

    std::uint32_t nextNodeId = 1;
    std::uint32_t activeNodeId = 0;
    std::uint32_t fadingNodeId = 0;
    std::uint32_t transitionNodeId = 0;

    const auto makeClosureNode = [](std::uint32_t nodeId, convo::isr::PayloadTier tier) {
        convo::isr::ClosureNodeRef ref{};
        ref.nodeId = nodeId;
        ref.payloadTier = static_cast<std::uint32_t>(tier);
        ref.kind = 1u;        // DSP node
        ref.ownership = 2u;   // Engine-owned shared runtime object
        ref.mutability = 1u;  // immutable payload
        ref.lifetime = 2u;    // runtime publication lifetime
        ref.hbDomain = 1u;    // publication HB domain
        ref.authority = 1u;   // NonRT publication authority
        ref.allocator = 1u;   // engine allocator domain
        return ref;
    };

    if (hasActive) {
        activeNodeId = nextNodeId++;
        closure.nodes.push_back(makeClosureNode(activeNodeId, convo::isr::PayloadTier::InlineImmutable));
    }

    if (hasFading && world.topology.fadingRuntimeUuid != world.topology.runtimeUuid) {
        fadingNodeId = nextNodeId++;
        closure.nodes.push_back(makeClosureNode(fadingNodeId, convo::isr::PayloadTier::ImmutableShared));
    }

    if (hasTransitionNext) {
        transitionNodeId = nextNodeId++;
        closure.nodes.push_back(makeClosureNode(transitionNodeId, convo::isr::PayloadTier::ImmutableShared));
    }

    if (activeNodeId != 0 && transitionNodeId != 0) {
        closure.edges.push_back(activeNodeId);
        closure.edges.push_back(transitionNodeId);
    }

    if (activeNodeId != 0 && fadingNodeId != 0) {
        closure.edges.push_back(activeNodeId);
        closure.edges.push_back(fadingNodeId);
    }

    convo::isr::TieredPayloadDescriptor descriptor{};
    descriptor.tier = hasTransitionNext
        ? convo::isr::PayloadTier::ImmutableShared
        : convo::isr::PayloadTier::InlineImmutable;
    descriptor.requiresRT = false;
    descriptor.hasExternalResource = false;
    descriptor.pinnedLifetime = true;

    const bool closureValid = closureGraphWalker_.validateGraph(closure);
    const bool precheckValid = precheckRuntimePublication(closure, descriptor);
    if (!closureValid || !precheckValid) {
        return rejectWithEvidence("closure_or_precheck_invalid");
    }

    if (!transitionSemanticTransactionState(semanticTransactionState_, convo::isr::SemanticTransactionState::Committed))
        return rejectWithEvidence("semantic_state_committed_transition");

    return true;
}

// buildRuntimePublishWorld() implementation removed from Bridge (#5/#7 Sprint-2)
// Build authority belongs to RuntimeBuilder, not Bridge
// Callers should use RuntimeBuilder directly to build RuntimePublishWorld

void AudioEngine::onRuntimePublishedNonRt(const RuntimePublishWorld& world) noexcept
{
    // ★ P3-B: World 発行を監査記録
    worldLifecycleAudit_.onWorldPublished(
        world.worldId,
        world.publication.epoch,
        convo::isr::CorrelationId{engineInstanceId_, world.publication.sequenceId});

    if (!transitionSemanticTransactionState(semanticTransactionState_, convo::isr::SemanticTransactionState::Published))
    {
        forceSemanticTransactionState(semanticTransactionState_, convo::isr::SemanticTransactionState::Published);
    }

    const auto updateMinMetric = [](std::atomic<std::uint64_t>& dst, std::uint64_t value) noexcept
    {
        auto observed = convo::consumeAtomic(dst, std::memory_order_acquire);
        while ((observed == 0 || value < observed)
               && !convo::compareExchangeAtomic(dst,
                                                observed,
                                                value,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire))
        {
        }
    };

    const auto updateMaxMetric = [](std::atomic<std::uint64_t>& dst, std::uint64_t value) noexcept
    {
        auto observed = convo::consumeAtomic(dst, std::memory_order_acquire);
        while (value > observed
               && !convo::compareExchangeAtomic(dst,
                                                observed,
                                                value,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire))
        {
        }
    };

    debugRuntime_.recordShadowCompareObservation(world.publication.sequenceId, world.semanticHash);

    const bool observeRollbackRequested = convo::exchangeAtomic(observeMonotonicRollbackRequested_, false, std::memory_order_acq_rel);
    const bool shadowEscalated = debugRuntime_.escalationCount() > 0;
    const bool monotonicViolated = debugRuntime_.monotonicViolationCount() > 0;
    if (observeRollbackRequested
        || shadowEscalated
        || monotonicViolated
        || world.publication.sequenceId <= world.publication.previousSequenceId)
    {
        retireRuntimeEx_.requestRollback();
    }

    debugRuntime_.recordHBEdge(100u,
                               200u,
                               static_cast<std::uint64_t>(world.generation),
                               static_cast<std::uint64_t>(world.runtimeVersion),
                               static_cast<int>(std::memory_order_release));

    runtimePublicationBridge_.commit(convo::isr::PublishAuthority::Granted,
                                     convo::isr::RuntimeBoundary::NonRTWorld,
                                     &world,
                                     world.generation,
                                     world.publication.sequenceId,
                                     world.publication.epoch,
                                     world.publication.mappedRuntimeGeneration);
    convo::publishAtomic(lastCommittedRuntimeGeneration_, world.generation, std::memory_order_release);
    convo::publishAtomic(lastCommittedPublicationSequence_, world.publication.sequenceId, std::memory_order_release);
    convo::fetchAddAtomic(publishedWorldCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    updateMinMetric(oldestPublishedGeneration_, world.generation);
    updateMaxMetric(youngestPublishedGeneration_, world.generation);

#if defined(JUCE_DEBUG) || defined(CONVO_CI_BUILD)
    debugRuntime_.emitCIArtifacts();
#endif
    debugRuntime_.emitHBTrace();
    emitEvidenceTickNonRt(false);
}

void AudioEngine::onRuntimeRetiredNonRt(const RuntimePublishWorld* world) noexcept
{
    ASSERT_NON_RT_THREAD();

    if (world == nullptr)
        return;

    // ★ P3-B: World 退役を監査記録
    worldLifecycleAudit_.onWorldRetired(world->worldId, world->publication.epoch);

    debugRuntime_.recordHBEdge(200u,
                               300u,
                               static_cast<std::uint64_t>(world->runtimeVersion),
                               static_cast<std::uint64_t>(world->generation),
                               static_cast<int>(std::memory_order_acq_rel));

    if (shutdownRuntime_.isShutdownInProgress())
        shutdownRuntime_.markPostStopEnqueue();

    runtimePublicationBridge_.retire(convo::isr::RetireAuthority::Granted,
                                     convo::isr::RuntimeBoundary::NonRTWorld,
                                     world);

    const std::uint32_t slot = static_cast<std::uint32_t>(world->generation % 256u);
    // ★ B-1.6: truncation 削除（uint64_t 化により情報損失ゼロ）
    std::uint64_t generation = world->generation;
    if (generation == 0u)
        generation = 1u;

    convo::isr::RetireIntent intent{};
    intent.dspSlot = slot;
    intent.generation = generation;
    intent.retireEpoch = static_cast<std::uint64_t>(world->generation);
    intent.isValid = true;

    retireRuntime_.emitRetireIntentRT(intent);
    runtimePublicationBridge_.setPendingIntentCount(retireRuntime_.pendingIntentCount());
    runtimePublicationBridge_.setRetireBacklogCount(retireRuntime_.pendingIntentCount());
    const auto pendingIntents = retireRuntime_.dequeuePendingRetireIntents();
    convo::publishAtomic(pendingRetireGenerationCount_, static_cast<std::uint64_t>(pendingIntents.size()), std::memory_order_release);

    const auto updateMinMetric = [](std::atomic<std::uint64_t>& dst, std::uint64_t value) noexcept
    {
        auto observed = convo::consumeAtomic(dst, std::memory_order_acquire);
        while ((observed == 0 || value < observed)
               && !convo::compareExchangeAtomic(dst,
                                                observed,
                                                value,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire))
        {
        }
    };

    const auto updateMaxMetric = [](std::atomic<std::uint64_t>& dst, std::uint64_t value) noexcept
    {
        auto observed = convo::consumeAtomic(dst, std::memory_order_acquire);
        while (value > observed
               && !convo::compareExchangeAtomic(dst,
                                                observed,
                                                value,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire))
        {
        }
    };

    std::uint64_t pendingMinGeneration = 0;
    std::uint64_t pendingMaxGeneration = 0;
    for (const auto& pending : pendingIntents)
    {
        if (!pending.isValid)
            continue;

        const auto generationValue = static_cast<std::uint64_t>(pending.generation);
        if (pendingMinGeneration == 0 || generationValue < pendingMinGeneration)
            pendingMinGeneration = generationValue;
        if (generationValue > pendingMaxGeneration)
            pendingMaxGeneration = generationValue;
    }

    if (pendingMinGeneration != 0)
    {
        updateMinMetric(oldestPendingGeneration_, pendingMinGeneration);
        updateMinMetric(oldestRetirePendingGeneration_, pendingMinGeneration);
    }
    if (pendingMaxGeneration != 0)
        updateMaxMetric(newestPendingGeneration_, pendingMaxGeneration);

    const double nowMs = juce::Time::getMillisecondCounterHiRes();
    if (!pendingIntents.empty())
    {
        auto firstSeen = convo::consumeAtomic(oldestPendingFirstSeenMs_, std::memory_order_acquire);
        if (firstSeen <= 0.0)
        {
            convo::publishAtomic(oldestPendingFirstSeenMs_, nowMs, std::memory_order_release);
            firstSeen = nowMs;
        }
        convo::publishAtomic(oldestPendingAge_, std::max(0.0, nowMs - firstSeen), std::memory_order_release);
    }
    else
    {
        convo::publishAtomic(oldestPendingFirstSeenMs_, 0.0, std::memory_order_release);
        convo::publishAtomic(oldestPendingAge_, 0.0, std::memory_order_release);
    }

    const double oldestPendingAgeMs = convo::consumeAtomic(oldestPendingAge_, std::memory_order_acquire);
    const std::uint64_t maxRetireDeferralEpochs = convo::consumeAtomic(maxRetireDeferralEpochs_, std::memory_order_acquire);
    const double maxRetireWallClockMs = convo::consumeAtomic(maxRetireWallClockMs_, std::memory_order_acquire);
    const bool transitionSemanticMismatch = !hasEquivalentTransitionSemantic(*world);
    if (transitionSemanticMismatch)
    {
        diagLog("[ISR][Leak-04-Guard] transition semantic mismatch on retire path: execution.transitionActive="
            + juce::String(world->execution.transitionActive ? 1 : 0)
            + " topology.hasFadingRuntime="
            + juce::String(world->topology.hasFadingRuntime ? 1 : 0));
        retireRuntimeEx_.requestRollback();
    }

    const bool hasAnyPendingTransition = world->topology.hasFadingRuntime || !pendingIntents.empty();

    for (const auto& pending : pendingIntents)
    {
        if (!pending.isValid)
            continue;

        const auto pendingGeneration = static_cast<std::uint64_t>(pending.generation);
        const auto maxObservedGeneration = convo::consumeAtomic(youngestObservedGeneration_, std::memory_order_acquire);
        const auto callbackActiveCount = convo::consumeAtomic(rtLocalState_.audioCallbackActiveCount, std::memory_order_acquire);
        const bool graceCompleted = retireRuntimeEx_.isGracePeriodCompleted(pendingGeneration,
                                             maxObservedGeneration,
                                             callbackActiveCount);
        const bool pendingIntentOwned = pending.isValid;
        const auto* currentPublished = RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore);
        const bool authoritativeOwnershipReleased = (currentPublished != world);
        const std::uint64_t retireDeferralEpochs = (maxObservedGeneration > pendingGeneration)
            ? (maxObservedGeneration - pendingGeneration)
            : 0u;
        const bool exceededDeferralThresholds = retireRuntimeEx_.hasExceededDeferralThresholds(retireDeferralEpochs,
                                                                                                oldestPendingAgeMs,
                                                                                                maxRetireDeferralEpochs,
                                                                                                maxRetireWallClockMs);

        const auto pendingSlot = static_cast<std::uint32_t>(pending.dspSlot & 0xFFu);
        retireRuntime_.acknowledgeRetireCoordination(pending);
        retireRuntimeEx_.emitIntent(pendingSlot, pending.generation);
        retireRuntimeEx_.enqueueRetire(pendingSlot);
        retireRuntimeEx_.settleEpoch(pendingSlot);
        if (exceededDeferralThresholds)
        {
            convo::fetchAddAtomic(retireEscalationCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
            quarantineSlot(pendingSlot, generation, convo::isr::QuarantineReason::RetireDeferralTimeout);

            const bool noReader = graceCompleted;
            const bool noExecutorReference = authoritativeOwnershipReleased;
            const bool noPendingTransition = !hasAnyPendingTransition;
            if (retireRuntimeEx_.canReclaimAfterEscalation(noReader,
                                                           noExecutorReference,
                                                           noPendingTransition))
            {
                retireRuntimeEx_.reclaim(pendingSlot);
            }
        }
        else if (retireRuntimeEx_.canTransitionRetirePendingToFree(graceCompleted,
                                                                    pendingIntentOwned,
                                                                    authoritativeOwnershipReleased))
        {
            retireRuntimeEx_.reclaim(pendingSlot);
        }
        else
        {
            quarantineSlot(pendingSlot, generation, convo::isr::QuarantineReason::RetireDeferralTimeout);
        }
    }

    convo::fetchAddAtomic(retiredWorldCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    updateMinMetric(oldestRetiredGeneration_, world->generation);

    runtimePublicationBridge_.setPendingIntentCount(retireRuntime_.pendingIntentCount());
    runtimePublicationBridge_.setRetireBacklogCount(retireRuntime_.pendingIntentCount());
    emitEvidenceTickNonRt(false);
}

void AudioEngine::emitEvidenceTickNonRt(bool force) noexcept
{
    const std::int64_t nowTicks = juce::Time::getHighResolutionTicks();
    const std::int64_t minIntervalTicks = juce::Time::secondsToHighResolutionTicks(1.0);
    const std::int64_t lastTicks = convo::consumeAtomic(rtAuxMutable_.lastEvidenceEmitHighResTicks, std::memory_order_acquire);

    if (!force && lastTicks != 0 && (nowTicks - lastTicks) < minIntervalTicks)
        return;

    convo::publishAtomic(rtAuxMutable_.lastEvidenceEmitHighResTicks, nowTicks, std::memory_order_release);

    const auto evidenceRoot = std::filesystem::current_path() / "evidence";
    retireRuntimeEx_.emitRetireTimeline(evidenceRoot / "retire_timeline.json");
    evidenceExporter_.exportEvidence();
    worldLifecycleAudit_.tryDumpPeriodic();
}

// [PR-3/3A] Orchestrator 経路のみ。Deferred は submitPublishRequest が自動 enqueue する。
// Phase2: DSPHandle 事前登録 (Execution Path Handle Normalization)
void AudioEngine::enqueuePublicationIntentForRuntimeCommit(DSPCore* newDSP,
                                                           int generation,
                                                           const convo::RuntimeBuildSnapshot& sealedSnapshot)
{
    if (newDSP == nullptr)
        return;

    // Phase2: commit 時に DSPHandle を事前登録する
    auto handle = registerDSPHandleForRuntime(newDSP);

    convo::isr::PublicationAdmission::PublishRequest req;
    req.newDSP = handle;
    req.generation = generation;
    req.sealedSnapshot = sealedSnapshot;

    runtimeOrchestrator_->submitPublishRequest(req);

    // DSP commit 完了時に DSPReady を常に enqueue する。
    // processLearningCommands の DSPReady ハンドラが
    // learningRuntimeState に応じて適切に処理する。
    // (WaitingForDSP の場合は学習再開、Idle/Running の場合は何もしない)
    const LearningCommand dspReadyCmd {
        LearningCommand::Type::DSPReady,
        false,
        convo::consumeAtomic(pendingLearningMode, std::memory_order_acquire),
        pendingIRGeneration
    };
    enqueueLearningCommand(dspReadyCmd);
}
