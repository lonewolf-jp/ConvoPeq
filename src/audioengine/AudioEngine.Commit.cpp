#include <JuceHeader.h>
#include "AudioEngine.h"
#include "RuntimeBuilder.h"
#include "RuntimePublicationValidator.h"

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

void destroyPublicationIntentNode(void* ptr) noexcept
{
    delete static_cast<AudioEngine::PublicationIntent*>(ptr);
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
    if (!convo::RuntimeGraph::validateDescriptorSet())
        return false;

    if (!convo::RuntimeGraph::validateDecisionCoverageContract())
        return false;

    if (world.routing.eqBypassed != world.graph.eqBypassed)
        return false;

    if (world.routing.convBypassed != world.graph.convBypassed)
        return false;

    const bool hasGraphActiveNode = (world.graph.activeNode != nullptr)
        || (world.graph.runtimeUuid != 0)
        || (world.graph.transitionCurrentRuntimeUuid != 0);
    const bool hasGraphFadingNode = (world.graph.fadingNode != nullptr)
        || (world.graph.fadingRuntimeUuid != 0)
        || (world.graph.transitionNextRuntimeUuid != 0);

    if (hasGraphActiveNode != (world.topology.runtimeUuid != 0))
        return false;

    if (hasGraphFadingNode != world.topology.hasFadingRuntime)
        return false;

    if (world.topology.runtimeUuid != world.graph.runtimeUuid)
        return false;

    if (world.topology.fadingRuntimeUuid != world.graph.fadingRuntimeUuid)
        return false;

    if (world.graph.transitionCurrentRuntimeUuid != 0
        && world.graph.transitionCurrentRuntimeUuid != world.graph.runtimeUuid)
        return false;

    if (world.graph.transitionNextRuntimeUuid != 0
        && world.graph.transitionNextRuntimeUuid != world.graph.fadingRuntimeUuid)
        return false;

    if (world.execution.transitionActive != world.topology.hasFadingRuntime)
        return false;

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

[[nodiscard]] bool AudioEngine::acceptsRuntimePublication() const noexcept
{
    const auto currentLifecycle = convo::consumeAtomic(lifecycleState, std::memory_order_acquire);
    const auto currentPhase = convo::consumeAtomic(shutdownPhase, std::memory_order_acquire);

    if (currentLifecycle != EngineLifecycleState::Prepared)
        return false;

    if (currentPhase != ShutdownPhase::Running)
        return false;

    if (shutdownRuntime_.isShutdownInProgress())
        return false;

    return !isShutdownInProgress();
}

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

    // Stage 3: runtime admission.
    if (!acceptsRuntimePublication())
        return rejectWithEvidence("accepts_runtime_publication_false");

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
    std::uint32_t generation = static_cast<std::uint32_t>(world->generation & 0xFFFFFFFFu);
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
            retireRuntimeEx_.quarantine(pendingSlot);

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
            retireRuntimeEx_.quarantine(pendingSlot);
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
}

void AudioEngine::appendPublicationIntentForCommitSlot(DSPCore* newDSP,
                                                       int targetWorldId,
                                                       CommitReaderSlot readerSlot,
                                                       const convo::RuntimeBuildSnapshot& sealedSnapshot) noexcept
{
    if (newDSP == nullptr)
        return;

    if (!acceptsRuntimePublication())
    {
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        retireDSP(newDSP);
        return;
    }

    const int epochReaderIndex = toCommitReaderIndex(readerSlot);

    const convo::EpochDomainReaderGuard appendGuard(m_epochDomain, epochReaderIndex);

    const bool publicationThrottleActive = convo::consumeAtomic(retirePressurePublicationThrottleActive_, std::memory_order_acquire);
    if (publicationThrottleActive && hasPendingPublicationIntents())
    {
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        retireDSP(newDSP);
        return;
    }

    const auto targetWorldIdU64 = static_cast<std::uint64_t>(std::max(0, targetWorldId));
    const auto lastEnqueuedTargetWorldId = convo::consumeAtomic(lastEnqueuedPublicationTargetWorldId_, std::memory_order_acquire);
    if (targetWorldIdU64 <= lastEnqueuedTargetWorldId)
    {
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        convo::publishAtomic(lastDroppedGeneration_, targetWorldIdU64, std::memory_order_release);
        retireDSP(newDSP);
        return;
    }

    auto* intent = new PublicationIntent();
    intent->newDSP = newDSP;
    intent->targetWorldId = targetWorldIdU64;
    intent->requestId = convo::fetchAddAtomic(publicationIntentRequestIdCounter_,
                                              static_cast<std::uint64_t>(1),
                                              std::memory_order_acq_rel) + 1;
    intent->enqueueTimeTicks = juce::Time::getHighResolutionTicks();
    intent->runtimeBuildSnapshot = sealedSnapshot;
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

    const auto backlog = hasPendingPublicationIntents() ? 1ull : 0ull;
    convo::publishAtomic(lastEnqueuedPublicationTargetWorldId_, targetWorldIdU64, std::memory_order_release);
    convo::publishAtomic(publicationBacklog_, backlog, std::memory_order_release);
    runtimePublicationBridge_.setPublicationBacklogCount(backlog);
    runtimePublicationBridge_.setPendingIntentCount(backlog);
}

void AudioEngine::appendPublicationIntentForCommitProducer(DSPCore* newDSP,
                                                           int targetWorldId,
                                                           const convo::RuntimeBuildSnapshot& sealedSnapshot) noexcept
{
    appendPublicationIntentForCommitSlot(newDSP, targetWorldId, CommitReaderSlot::Producer, sealedSnapshot);
}

void AudioEngine::appendPublicationIntentForCommitConsumer(DSPCore* newDSP,
                                                           int targetWorldId,
                                                           const convo::RuntimeBuildSnapshot& sealedSnapshot) noexcept
{
    appendPublicationIntentForCommitSlot(newDSP, targetWorldId, CommitReaderSlot::Consumer, sealedSnapshot);
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
    convo::publishAtomic(lastEnqueuedPublicationTargetWorldId_, static_cast<std::uint64_t>(0), std::memory_order_release);
    convo::publishAtomic(publicationBacklog_, 0ull, std::memory_order_release);
    runtimePublicationBridge_.setPublicationBacklogCount(0u);
    runtimePublicationBridge_.setPendingIntentCount(0u);
}

void AudioEngine::enqueuePublicationIntentForRuntimeCommit(DSPCore* newDSP,
                                                           int generation,
                                                           const convo::RuntimeBuildSnapshot& sealedSnapshot)
{
    if (newDSP == nullptr)
        return;

    if (!acceptsRuntimePublication())
    {
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        retireDSP(newDSP);
        return;
    }

    appendPublicationIntentForCommitProducer(newDSP, generation, sealedSnapshot);

    triggerAsyncUpdate();
}

[[nodiscard]] bool AudioEngine::hasPublicationLogPending() noexcept
{
    PublicationIntent* const cursor = convo::consumeAtomic(publicationLog.consumedTail, std::memory_order_acquire); // acquire: executeCommit の publishAtomic release と HB
    return cursor != nullptr && convo::consumeAtomic(cursor->next, std::memory_order_acquire) != nullptr; // acquire: appendPublicationIntent の CAS release と HB
}

[[nodiscard]] bool AudioEngine::hasPendingPublicationIntents() noexcept
{
    return hasPublicationLogPending();
}

void AudioEngine::drainPublicationIntentsForRuntimeCommit()
{
    if (!acceptsRuntimePublication())
    {
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        return;
    }

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
                applyRuntimeCommitFromIntent(next->newDSP,
                                             static_cast<int>(next->targetWorldId),
                                             next->runtimeBuildSnapshot);
            }

            if (cursor != publicationLogSentinel)
                enqueueDeferredDeleteNonRt(cursor, destroyPublicationIntentNode);

            convo::publishAtomic(publicationLog.retiredHead, cursor, std::memory_order_release); // release: drainPublicationLogForShutdown の consume acquire と HB
            cursor = next;
        }
    }

    const bool hasRemaining = hasPendingPublicationIntents();

    convo::publishAtomic(publicationBacklog_, hasRemaining ? 1ull : 0ull, std::memory_order_release);
    runtimePublicationBridge_.setPublicationBacklogCount(hasRemaining ? 1u : 0u);
    runtimePublicationBridge_.setPendingIntentCount(hasRemaining ? 1u : 0u);

    convo::publishAtomic(commitDrainInProgress, false, std::memory_order_release); // release: 次回の hasPublicationLogPending の acquire と HB

    if (hasRemaining && !isShutdownInProgress())
        triggerAsyncUpdate();
}

void AudioEngine::applyRuntimeCommitFromIntent(DSPCore* newDSP,
                                               int generation,
                                               const convo::RuntimeBuildSnapshot& sealedSnapshot)
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

    const auto replaceFadingRuntimeDSPAndRetirePrevious = [this](DSPCore* dsp) noexcept
    {
        DSPCore* atomicCurrent = getActiveRuntimeDSP();
        const auto runtimeReadHandle = readControlRuntimeHandle();
        validateDistinctRuntimeSlots("replaceFadingRuntimeDSPAndRetirePrevious.before",
                                     atomicCurrent,
                         resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                                     nullptr);

        auto* const prevRaw = exchangeFadingRuntimeDSP(dsp);
        if (auto* prev = (reinterpret_cast<uintptr_t>(prevRaw) == (~static_cast<uintptr_t>(0))) ? nullptr : prevRaw)
        {
            if (prev == dsp)
            {
                logUnexpectedRuntimeTransition("replaceFadingRuntimeDSPAndRetirePrevious", prev, dsp);
                jassert(prev != dsp);
                return;
            }

            retireDSP(prev);
        }

        validateDistinctRuntimeSlots("replaceFadingRuntimeDSPAndRetirePrevious.after",
                                     atomicCurrent,
                                     resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                                     nullptr);
        logRuntimeTransitionEvent("replaceFadingRuntimeDSPAndRetirePrevious", dsp);
    };

    const auto publishSmoothTransitionState = [this, &sealedSnapshot](DSPCore* nextDSP,
                                                                      DSPCore* previousDSP,
                                                                      double fadeTimeSec) noexcept
    {
        if (nextDSP == nullptr || nextDSP == previousDSP)
        {
            logUnexpectedRuntimeTransition("publishSmoothTransitionState", nextDSP, previousDSP);
            jassert(nextDSP != nullptr && nextDSP != previousDSP);
        }

        // Migrated to publishWorld() with pre-built RuntimePublishWorld (Sprint-2 P1-A)
        auto coordinator = makeRuntimePublicationCoordinator();
        auto worldBuilder = convo::RuntimeBuilder(*this);
        auto worldOwner = worldBuilder.buildRuntimePublishWorld(nextDSP,
                                                                 previousDSP,
                                                                 convo::TransitionPolicy::SmoothOnly,
                                                                 fadeTimeSec,
                                                                 true,
                                                                 &sealedSnapshot);
        coordinator.publishWorld(std::move(worldOwner));
        logRuntimeTransitionEvent("publishSmoothTransitionState", nextDSP, previousDSP);
    };

    const auto startImmediateSmoothTransition = [this, &replaceFadingRuntimeDSPAndRetirePrevious, &sealedSnapshot](DSPCore* previousDSP,
                                                                                                                   double fadeTimeSec) noexcept
    {
        DSPCore* atomicCurrent = getActiveRuntimeDSP();
        const auto runtimeReadHandle = readControlRuntimeHandle();
        if (previousDSP == nullptr || previousDSP == atomicCurrent)
        {
            logUnexpectedRuntimeTransition("startImmediateSmoothTransition", atomicCurrent, previousDSP);
            jassert(previousDSP != nullptr && previousDSP != atomicCurrent);
        }

        const double rampSampleRate = std::max(1.0,
            (atomicCurrent != nullptr) ? atomicCurrent->sampleRate : consumeAtomic(currentSampleRate, std::memory_order_acquire));
        dspCrossfadeGain.reset(rampSampleRate, std::max(0.001, fadeTimeSec));
        dspCrossfadeGain.setCurrentAndTargetValue(0.0);

        replaceFadingRuntimeDSPAndRetirePrevious(previousDSP);
        publishAtomic(dspCrossfadeUseDryAsOld, false, std::memory_order_release);
        publishAtomic(firstIrDryCrossfadePending, false, std::memory_order_release);
        publishAtomic(queuedFadeTimeSec, fadeTimeSec, std::memory_order_release);
        publishAtomic(dspCrossfadePending, true, std::memory_order_release);
        setIRChangeFlag();
        
        // Migrated to publishWorld() with pre-built RuntimePublishWorld (Sprint-2 P1-A)
        {
            auto coordinator = makeRuntimePublicationCoordinator();
            auto worldBuilder = convo::RuntimeBuilder(*this);
            auto worldOwner = worldBuilder.buildRuntimePublishWorld(atomicCurrent,
                                                                     previousDSP,
                                                                     convo::TransitionPolicy::SmoothOnly,
                                                                     fadeTimeSec,
                                                                     true,
                                                                     &sealedSnapshot);
            coordinator.publishWorld(std::move(worldOwner));
        }
        
        validateDistinctRuntimeSlots("startImmediateSmoothTransition",
                                     atomicCurrent,
                                     resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                                     nullptr);
        logRuntimeTransitionEvent("startImmediateSmoothTransition", atomicCurrent, previousDSP);
    };

    const auto retireRuntimeImmediately = [this](DSPCore* dsp) noexcept
    {
        if (dsp == nullptr)
            return;

        const auto runtimeReadHandle = readControlRuntimeHandle();
        auto* publishedCurrent = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        DSPCore* atomicCurrent = getActiveRuntimeDSP();
        if (dsp == atomicCurrent || dsp == publishedCurrent)
        {
            logUnexpectedRuntimeTransition("retireRuntimeImmediately", atomicCurrent, dsp);
            jassert(dsp != atomicCurrent && dsp != publishedCurrent);
            return;
        }

        logRuntimeTransitionEvent("retireRuntimeImmediately", dsp);
        retireDSP(dsp);
    };

    const auto publishHardResetForCurrentDSP = [this, &sealedSnapshot]() noexcept
    {
        DSPCore* atomicCurrent = getActiveRuntimeDSP();
        const auto runtimeReadHandle = readControlRuntimeHandle();
        if (atomicCurrent == nullptr)
        {
            logUnexpectedRuntimeTransition("publishHardResetForCurrentDSP", nullptr, nullptr);
            jassert(atomicCurrent != nullptr);
        }

        publishAtomic(dspCrossfadePending, false, std::memory_order_release);
        publishAtomic(dspCrossfadeUseDryAsOld, false, std::memory_order_release);
        publishAtomic(firstIrDryCrossfadePending, false, std::memory_order_release);
        publishAtomic(dspCrossfadeStartDelayBlocks, 0, std::memory_order_release);
        publishAtomic(dspCrossfadeDryHoldSamples, 0, std::memory_order_release);
        
        // Migrated to publishWorld() with pre-built RuntimePublishWorld (Sprint-2 P1-A)
        {
            auto coordinator = makeRuntimePublicationCoordinator();
            auto worldBuilder = convo::RuntimeBuilder(*this);
            auto worldOwner = worldBuilder.buildRuntimePublishWorld(atomicCurrent,
                                                                     nullptr,
                                                                     convo::TransitionPolicy::HardReset,
                                                                     0.0,
                                                                     false,
                                                                     &sealedSnapshot);
            coordinator.publishWorld(std::move(worldOwner));
        }
        validateDistinctRuntimeSlots("publishHardResetForCurrentDSP",
                                     atomicCurrent,
                                     resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                                     nullptr);
        logRuntimeTransitionEvent("publishHardResetForCurrentDSP", atomicCurrent);
    };

    const auto armDryAsOldCrossfadeForCurrentDSP = [this, &sealedSnapshot](double fadeTimeSec,
                                                                           double targetIrScale) noexcept
    {
        DSPCore* atomicCurrent = getActiveRuntimeDSP();
        const auto runtimeReadHandle = readControlRuntimeHandle();
        if (atomicCurrent == nullptr)
        {
            logUnexpectedRuntimeTransition("armDryAsOldCrossfadeForCurrentDSP", nullptr, nullptr);
            jassert(atomicCurrent != nullptr);
        }

        const double rampSampleRate = std::max(1.0,
            (atomicCurrent != nullptr) ? atomicCurrent->sampleRate : consumeAtomic(currentSampleRate, std::memory_order_acquire));
        dspCrossfadeGain.reset(rampSampleRate, std::max(0.001, fadeTimeSec));
        dspCrossfadeGain.setCurrentAndTargetValue(0.0);

        publishAtomic(queuedFadeTimeSec, fadeTimeSec, std::memory_order_release);
        publishAtomic(dspCrossfadeDryHoldSamples,
                      std::max(1, consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire)));
        dspCrossfadeDryScaleGain.reset(std::max(1.0, consumeAtomic(currentSampleRate, std::memory_order_acquire)), 0.060);
        dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
        publishAtomic(dspCrossfadeDryScaleTarget, targetIrScale, std::memory_order_release);
        publishAtomic(dspCrossfadeUseDryAsOld, true, std::memory_order_release);
        publishAtomic(firstIrDryCrossfadePending, true, std::memory_order_release);
        publishAtomic(dspCrossfadePending, true, std::memory_order_release);
        publishAtomic(firstIrDryCrossfadeDone, true, std::memory_order_release);
        setIRChangeFlag();
        
        // Migrated to publishWorld() with pre-built RuntimePublishWorld (Sprint-2 P1-A)
        {
            auto coordinator = makeRuntimePublicationCoordinator();
            auto worldBuilder = convo::RuntimeBuilder(*this);
            auto worldOwner = worldBuilder.buildRuntimePublishWorld(atomicCurrent,
                                                                     nullptr,
                                                                     convo::TransitionPolicy::DryAsOld,
                                                                     fadeTimeSec,
                                                                     true,
                                                                     &sealedSnapshot);
            coordinator.publishWorld(std::move(worldOwner));
        }
        validateDistinctRuntimeSlots("armDryAsOldCrossfadeForCurrentDSP",
                                     atomicCurrent,
                                     resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                                     nullptr);
        logRuntimeTransitionEvent("armDryAsOldCrossfadeForCurrentDSP", atomicCurrent);
    };

    const auto runtimeReadHandleAtEntry = readControlRuntimeHandle();
    validateDistinctRuntimeSlots("commitNewDSP.entry",
                                 getActiveRuntimeDSP(),
                                 resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandleAtEntry),
                                 nullptr);

    // Lock to ensure the check and commit are atomic with respect to new rebuild requests.
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        // 古いリクエストの結果であれば破棄 (Race condition対策)
        if (generation != consumeAtomic(rebuildGeneration, std::memory_order_acquire)) // acquire: prepareCommit の publishAtomic release と HB
        {
            publishAtomic(rtAuxMutable_.lastRejectedGenerationNonRt, static_cast<uint64_t>(generation), std::memory_order_release); // release: UI の consumeAtomic acquire と HB
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
            publishAtomic(rtAuxMutable_.lastRejectedGenerationNonRt, static_cast<uint64_t>(generation), std::memory_order_release); // release: UI の consumeAtomic acquire と HB
            if (newDSP != nullptr)
                retireDSP(newDSP);
            return;
        }

        // 1. 旧 DSP を安全にキャプチャしてから新 DSP を公開する
        dspToTrash = getActiveRuntimeDSP();

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
                publishAtomic(rtAuxMutable_.lastRejectedGenerationNonRt, static_cast<uint64_t>(generation), std::memory_order_release); // release: UI の consumeAtomic acquire と HB
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
                const auto runtimeReadHandle = readControlRuntimeHandle();
                const bool hasFadingRuntime = hasFadingRuntimeInWorld(runtimeReadHandle);
                const bool hasPendingCrossfade = hasPendingCrossfadeInWorld(runtimeReadHandle);
                const bool useDryAsOld = shouldUseDryAsOldInWorld(runtimeReadHandle);

                if (hasFadingRuntime || hasPendingCrossfade || useDryAsOld)
                {
                    diagLog("[DIAG] commitNewDSP: deferring commit until active fade settles newUuid="
                        + juce::String(static_cast<juce::int64>(newDSP->runtimeUuid))
                        + " oldUuid=" + juce::String(static_cast<juce::int64>(dspToTrash->runtimeUuid))
                        + " fadeSec=" + juce::String(crossfadeContext.fadeTimeSec, 3));
                    appendPublicationIntentForCommitConsumer(newDSP, generation, sealedSnapshot);
                    return;
                }
            }
        }

        // 2. 新ランタイム publish を 2 段直列で明示する
        setActiveRuntimeDSP(newDSP);

        const auto previousHandle = dspHandleRuntime_.getActiveRuntimeDSPHandle();
        const auto newHandle = registerDSPHandleForRuntime(newDSP);
        if (crossfadeContext.needsCrossfade
            && !previousHandle.isNull()
            && !newHandle.isNull())
        {
            const auto crossfadeId = dspHandleRuntime_.beginCrossfade(previousHandle, newHandle);
            crossfadeAuthorityRuntime_.registerCrossfade(previousHandle, newHandle);
            publishAtomic(activeCrossfadeId_, crossfadeId, std::memory_order_release);
        }
        else
        {
            if (!previousHandle.isNull())
            {
                dspHandleRuntime_.retire(previousHandle);
                dspHandleRuntime_.reclaim(previousHandle);
            }

            if (!newHandle.isNull())
                dspHandleRuntime_.activate(newHandle);

            publishAtomic(activeCrossfadeId_, static_cast<convo::isr::CrossfadeId>(0u), std::memory_order_release);
        }
        
        // Migrated to publishWorld() with pre-built RuntimePublishWorld (Sprint-2 P1-A)
        {
            auto coordinator = makeRuntimePublicationCoordinator();
            auto worldBuilder = convo::RuntimeBuilder(*this);
            auto worldOwner = worldBuilder.buildRuntimePublishWorld(newDSP,
                                                                     nullptr,
                                                                     convo::TransitionPolicy::SmoothOnly,
                                                                     0.0,
                                                                     false,
                                                                     &sealedSnapshot);
            coordinator.publishWorld(std::move(worldOwner));
        }

        // 3. EBR：エポックを進める
        advanceRetireEpoch();

        const auto runtimeReadHandleAfterPublish = readControlRuntimeHandle();
        validateDistinctRuntimeSlots("commitNewDSP.afterPublish",
                 getActiveRuntimeDSP(),
             resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandleAfterPublish),
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
        publishLatencyDelayAtomics(dOld, dNew);
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
                publishLatencyDelayAtomics(dOld, dNew);
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
                const auto runtimeReadHandle = readControlRuntimeHandle();
                const bool hasFadingRuntime = hasFadingRuntimeInWorld(runtimeReadHandle);
                const bool hasPendingCrossfade = hasPendingCrossfadeInWorld(runtimeReadHandle);
                const bool useDryAsOld = shouldUseDryAsOldInWorld(runtimeReadHandle);
                const bool isFadingActive = hasFadingRuntime || hasPendingCrossfade || useDryAsOld;
                publishSmoothTransitionState(getActiveRuntimeDSP(),
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
    // sendChangeMessage() は runtime commit apply 経路でのみ rebuild 用途で呼ぶ。
    // それ以外の sendChangeMessage() はフェード完了・UIパラメータ変更・
    // 状態復元など rebuild とは独立したイベント用途。
    const auto runtimeReadHandleBeforeNotify = readControlRuntimeHandle();
    validateDistinctRuntimeSlots("commitNewDSP.beforeSendChangeMessage",
                                 getActiveRuntimeDSP(),
                                 resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandleBeforeNotify),
                                 nullptr);
    diagLog("[DIAG] commitNewDSP: queue coalesced change notification");
    if (!exchangeAtomic(pendingChangeNotification, true))
        triggerAsyncUpdate();
}

void AudioEngine::publishRuntimeStateNonRt(DSPCore* current,
                                           DSPCore* next,
                                           convo::TransitionPolicy policy,
                                           double fadeTimeSec,
                                           bool active,
                                           const convo::RuntimeBuildSnapshot* sealedSnapshot) noexcept
{
    auto coordinator = makeRuntimePublicationCoordinator();
    auto worldBuilder = convo::RuntimeBuilder(*this);
    auto worldOwner = worldBuilder.buildRuntimePublishWorld(current,
                                                             next,
                                                             policy,
                                                             fadeTimeSec,
                                                             active,
                                                             sealedSnapshot);
    coordinator.publishWorld(std::move(worldOwner));
}
