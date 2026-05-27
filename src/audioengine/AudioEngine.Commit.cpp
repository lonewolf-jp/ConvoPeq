#include <JuceHeader.h>
#include "AudioEngine.h"
#include "RuntimeBuilder.h"

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
}  // namespace

std::atomic<std::uint64_t> AudioEngine::runtimeVersionCounterStorage_ { 1 };

std::atomic<std::uint64_t>& AudioEngine::runtimeVersionCounter() noexcept
{
    return runtimeVersionCounterStorage_;
}

[[nodiscard]] std::uint64_t AudioEngine::reserveNextRuntimeVersion() noexcept
{
    // acq_rel: runtime version counter を increment し、複数 world lifecycle across に unique ID を割り当て。
    return convo::fetchAddAtomic(runtimeVersionCounter(),
                                 static_cast<std::uint64_t>(1),
                                 std::memory_order_acq_rel);
}

[[nodiscard]] bool AudioEngine::runPublicationPrecheckNonRt(const RuntimePublishWorld& world) noexcept
{
    if (!world.isSealedRecursively())
    {
        debugRuntime_.validateOwnershipClosure();
        emitEvidenceTickNonRt(true);
        return false;
    }

    const bool hasActive = (world.graph.activeNode != nullptr);
    const bool hasFading = (world.graph.fadingNode != nullptr);
    const bool hasTransitionNext = (world.engine.transition.next != nullptr);

    if (!hasActive && !hasFading && !hasTransitionNext)
        return true;

    convo::isr::PayloadClosureDescriptor closure{};
    closure.closureId = static_cast<uint32_t>((world.runtimeVersion != 0)
        ? world.runtimeVersion
        : (world.generation != 0 ? world.generation : 1));

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

    if (hasFading && world.graph.fadingNode != world.graph.activeNode) {
        fadingNodeId = nextNodeId++;
        closure.nodes.push_back(makeClosureNode(fadingNodeId, convo::isr::PayloadTier::ImmutableShared));
    }

    if (hasTransitionNext
        && world.engine.transition.next != world.graph.activeNode
        && world.engine.transition.next != world.graph.fadingNode) {
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
    descriptor.tier = world.engine.transition.active
        ? convo::isr::PayloadTier::ImmutableShared
        : convo::isr::PayloadTier::InlineImmutable;
    descriptor.requiresRT = false;
    descriptor.hasExternalResource = false;
    descriptor.pinnedLifetime = true;

    const bool closureValid = closureGraphWalker_.validateGraph(closure);
    const bool precheckValid = precheckRuntimePublication(closure, descriptor);
    if (!closureValid || !precheckValid) {
        debugRuntime_.validateOwnershipClosure();
        emitEvidenceTickNonRt(true);
        return false;
    }

    return true;
}

void AudioEngine::onRuntimePublishedNonRt(const RuntimePublishWorld& world) noexcept
{
    debugRuntime_.recordHBEdge(100u,
                               200u,
                               static_cast<std::uint64_t>(world.generation),
                               static_cast<std::uint64_t>(world.runtimeVersion),
                               static_cast<int>(std::memory_order_release));

    commitRuntimePublication(world);

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

    retireRuntimePublication(world);

    const std::uint32_t slot = static_cast<std::uint32_t>(world->generation % 256u);
    std::uint32_t generation = static_cast<std::uint32_t>(world->runtimeVersion & 0xFFFFFFFFu);
    if (generation == 0u)
        generation = 1u;

    convo::isr::RetireIntent intent{};
    intent.dspSlot = slot;
    intent.generation = generation;
    intent.retireEpoch = static_cast<std::uint64_t>(world->generation);
    intent.isValid = true;

    retireRuntime_.emitRetireIntentRT(intent);
    const auto pendingIntents = retireRuntime_.dequeuePendingRetireIntents();
    for (const auto& pending : pendingIntents)
    {
        if (!pending.isValid)
            continue;

        const auto pendingSlot = static_cast<std::uint32_t>(pending.dspSlot & 0xFFu);
        retireRuntime_.acknowledgeRetireCoordination(pending);
        retireRuntimeEx_.emitIntent(pendingSlot, pending.generation);
        retireRuntimeEx_.enqueueRetire(pendingSlot);
        retireRuntimeEx_.settleEpoch(pendingSlot);
        retireRuntimeEx_.reclaim(pendingSlot);
    }
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

void AudioEngine::appendPublicationIntentForCommitSlot(DSPCore* newDSP, int generation, CommitReaderSlot readerSlot) noexcept
{
    if (newDSP == nullptr)
        return;

    const int epochReaderIndex = toCommitReaderIndex(readerSlot);

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

void AudioEngine::appendPublicationIntentForCommitProducer(DSPCore* newDSP, int generation) noexcept
{
    appendPublicationIntentForCommitSlot(newDSP, generation, CommitReaderSlot::Producer);
}

void AudioEngine::appendPublicationIntentForCommitConsumer(DSPCore* newDSP, int generation) noexcept
{
    appendPublicationIntentForCommitSlot(newDSP, generation, CommitReaderSlot::Consumer);
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

    appendPublicationIntentForCommitProducer(newDSP, generation);

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

    const auto replaceFadingRuntimeDSPAndRetirePrevious = [this](DSPCore* dsp) noexcept
    {
        DSPCore* atomicCurrent = getActiveRuntimeDSP();
        const auto runtimeReadView = readControlRuntimeView();
        const auto* runtimeGraph = getRuntimeGraph(runtimeReadView);
        validateDistinctRuntimeSlots("replaceFadingRuntimeDSPAndRetirePrevious.before",
                                     atomicCurrent,
                         resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeGraph),
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
                                     resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeGraph),
                                     nullptr);
        logRuntimeTransitionEvent("replaceFadingRuntimeDSPAndRetirePrevious", dsp);
    };

    const auto publishSmoothTransitionState = [this](DSPCore* nextDSP,
                                                     DSPCore* previousDSP,
                                                     double fadeTimeSec) noexcept
    {
        if (nextDSP == nullptr || nextDSP == previousDSP)
        {
            logUnexpectedRuntimeTransition("publishSmoothTransitionState", nextDSP, previousDSP);
            jassert(nextDSP != nullptr && nextDSP != previousDSP);
        }

        makeRuntimePublicationCoordinator()
            .publishState(nextDSP,
                          previousDSP,
                          convo::TransitionPolicy::SmoothOnly,
                          fadeTimeSec,
                          true);
        logRuntimeTransitionEvent("publishSmoothTransitionState", nextDSP, previousDSP);
    };

    const auto startImmediateSmoothTransition = [this, &replaceFadingRuntimeDSPAndRetirePrevious](DSPCore* previousDSP,
                                                                                                double fadeTimeSec) noexcept
    {
        DSPCore* atomicCurrent = getActiveRuntimeDSP();
        const auto runtimeReadView = readControlRuntimeView();
        const auto* runtimeGraph = getRuntimeGraph(runtimeReadView);
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
        makeRuntimePublicationCoordinator()
            .publishState(atomicCurrent,
                          previousDSP,
                          convo::TransitionPolicy::SmoothOnly,
                          fadeTimeSec,
                          true);
        validateDistinctRuntimeSlots("startImmediateSmoothTransition",
                                     atomicCurrent,
                                     resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeGraph),
                                     nullptr);
        logRuntimeTransitionEvent("startImmediateSmoothTransition", atomicCurrent, previousDSP);
    };

    const auto retireRuntimeImmediately = [this](DSPCore* dsp) noexcept
    {
        if (dsp == nullptr)
            return;

        const auto runtimeReadView = readControlRuntimeView();
        const auto* runtimeGraph = getRuntimeGraph(runtimeReadView);
        auto* publishedCurrent = (runtimeGraph != nullptr)
            ? static_cast<DSPCore*>(runtimeGraph->activeNode)
            : nullptr;
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

    const auto publishHardResetForCurrentDSP = [this]() noexcept
    {
        DSPCore* atomicCurrent = getActiveRuntimeDSP();
        const auto runtimeReadView = readControlRuntimeView();
        const auto* runtimeGraph = getRuntimeGraph(runtimeReadView);
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
        makeRuntimePublicationCoordinator()
            .publishState(atomicCurrent,
                          nullptr,
                          convo::TransitionPolicy::HardReset,
                          0.0,
                          false);
        validateDistinctRuntimeSlots("publishHardResetForCurrentDSP",
                                     atomicCurrent,
                                     resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeGraph),
                                     nullptr);
        logRuntimeTransitionEvent("publishHardResetForCurrentDSP", atomicCurrent);
    };

    const auto armDryAsOldCrossfadeForCurrentDSP = [this](double fadeTimeSec,
                                                          double targetIrScale) noexcept
    {
        DSPCore* atomicCurrent = getActiveRuntimeDSP();
        const auto runtimeReadView = readControlRuntimeView();
        const auto* runtimeGraph = getRuntimeGraph(runtimeReadView);
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
        makeRuntimePublicationCoordinator()
            .publishState(atomicCurrent,
                          nullptr,
                          convo::TransitionPolicy::DryAsOld,
                          fadeTimeSec,
                          true);
        validateDistinctRuntimeSlots("armDryAsOldCrossfadeForCurrentDSP",
                                     atomicCurrent,
                                     resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeGraph),
                                     nullptr);
        logRuntimeTransitionEvent("armDryAsOldCrossfadeForCurrentDSP", atomicCurrent);
    };

    const auto runtimeReadViewAtEntry = readControlRuntimeView();
    validateDistinctRuntimeSlots("commitNewDSP.entry",
                                 getActiveRuntimeDSP(),
                                 resolveFadingRuntimeDSPFromRuntimeWorldOnly(getRuntimeGraph(runtimeReadViewAtEntry)),
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
                const auto runtimeReadView = readControlRuntimeView();
                const auto* runtimeGraph = getRuntimeGraph(runtimeReadView);
                const auto preparedCrossfade = consumeCrossfadePreparedSnapshot();
                const bool hasFadingRuntime = (resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeGraph) != nullptr);
                const bool hasPendingCrossfade = ((runtimeGraph != nullptr) ? runtimeGraph->dspCrossfadePending : false)
                    || preparedCrossfade.pending;
                const bool useDryAsOld = ((runtimeGraph != nullptr) ? runtimeGraph->dspCrossfadeUseDryAsOld : false)
                    || preparedCrossfade.firstIrDryCrossfadePending
                    || preparedCrossfade.useDryAsOld;

                if (hasFadingRuntime || hasPendingCrossfade || useDryAsOld)
                {
                    diagLog("[DIAG] commitNewDSP: deferring commit until active fade settles newUuid="
                        + juce::String(static_cast<juce::int64>(newDSP->runtimeUuid))
                        + " oldUuid=" + juce::String(static_cast<juce::int64>(dspToTrash->runtimeUuid))
                        + " fadeSec=" + juce::String(crossfadeContext.fadeTimeSec, 3));
                    appendPublicationIntentForCommitConsumer(newDSP, generation);
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

        makeRuntimePublicationCoordinator()
            .publishState(newDSP,
                          nullptr,
                          convo::TransitionPolicy::SmoothOnly,
                          0.0,
                          false);

        // 3. EBR：エポックを進める
        advanceRetireEpoch();

        const auto runtimeReadViewAfterPublish = readControlRuntimeView();
        validateDistinctRuntimeSlots("commitNewDSP.afterPublish",
                 getActiveRuntimeDSP(),
             resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadViewAfterPublish.graph),
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
                const auto runtimeReadView = readControlRuntimeView();
                const auto* runtimeGraph = getRuntimeGraph(runtimeReadView);
                const auto preparedCrossfade = consumeCrossfadePreparedSnapshot();
                const bool hasFadingRuntime = (resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeGraph) != nullptr);
                const bool hasPendingCrossfade = ((runtimeGraph != nullptr) ? runtimeGraph->dspCrossfadePending : false)
                    || preparedCrossfade.pending;
                const bool useDryAsOld = ((runtimeGraph != nullptr) ? runtimeGraph->dspCrossfadeUseDryAsOld : false)
                    || preparedCrossfade.firstIrDryCrossfadePending
                    || preparedCrossfade.useDryAsOld;
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
    // sendChangeMessage() は commitNewDSP() でのみ rebuild 用途で呼ぶ。
    // それ以外の sendChangeMessage() はフェード完了・UIパラメータ変更・
    // 状態復元など rebuild とは独立したイベント用途。
    const auto runtimeReadViewBeforeNotify = readControlRuntimeView();
    validateDistinctRuntimeSlots("commitNewDSP.beforeSendChangeMessage",
                                 getActiveRuntimeDSP(),
                                 resolveFadingRuntimeDSPFromRuntimeWorldOnly(getRuntimeGraph(runtimeReadViewBeforeNotify)),
                                 nullptr);
    diagLog("[DIAG] commitNewDSP: queue coalesced change notification");
    if (!exchangeAtomic(pendingChangeNotification, true))
        triggerAsyncUpdate();
}
