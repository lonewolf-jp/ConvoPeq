#include "RuntimeBuilder.h"

#include <bit>
#include <cstdint>

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
// 局所 diagLog — 全ファイル統一パターン。
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
#endif





namespace convo {

namespace {

// ★ 現在は未使用（セマンティックハッシュの将来拡張用）。
[[maybe_unused]] [[nodiscard]] std::uint64_t hashBuildInput(const BuildInput& buildInput) noexcept
{
    std::uint64_t hash = 1469598103934665603ull;
    const auto mix = [&hash](std::uint64_t value) noexcept {
        hash ^= value;
        hash *= 1099511628211ull;
    };

    mix(static_cast<std::uint64_t>(buildInput.sampleRate == 0.0 ? 0 : std::bit_cast<std::uint64_t>(buildInput.sampleRate)));
    mix(static_cast<std::uint64_t>(buildInput.blockSize));
    mix(static_cast<std::uint64_t>(buildInput.ditherBitDepth));
    mix(static_cast<std::uint64_t>(buildInput.oversamplingFactor));
    mix(static_cast<std::uint64_t>(buildInput.oversamplingType));
    mix(static_cast<std::uint64_t>(buildInput.noiseShaperType));
    mix(static_cast<std::uint64_t>(buildInput.processingOrder));
    mix(static_cast<std::uint64_t>(buildInput.eqBypassed));
    mix(static_cast<std::uint64_t>(buildInput.convBypassed));
    mix(static_cast<std::uint64_t>(buildInput.softClipEnabled));
    mix(std::bit_cast<std::uint64_t>(buildInput.saturationAmount));
    mix(std::bit_cast<std::uint64_t>(buildInput.inputHeadroomGain));
    mix(std::bit_cast<std::uint64_t>(buildInput.outputMakeupGain));
    mix(std::bit_cast<std::uint64_t>(buildInput.convolverInputTrimGain));
    return hash;
}

} // namespace

const char* toString(BuildError error) noexcept
{
    switch (error)
    {
        case BuildError::None:
            return "None";
        case BuildError::InvalidInput:
            return "InvalidInput";
        case BuildError::ResourceUnavailable:
            return "ResourceUnavailable";
        case BuildError::MKLFailure:
            return "MKLFailure";
        case BuildError::ConvolverFailure:
            return "ConvolverFailure";
        case BuildError::PrepareFailure:
            return "PrepareFailure";
        case BuildError::WarmupFailed:
            return "WarmupFailed";
        case BuildError::InternalError:
            return "InternalError";
    }

    return "Unknown";
}

convo::aligned_unique_ptr<RuntimePublishWorld>
RuntimeBuilder::createBootstrapWorld() noexcept
{
    const auto publicationIdentity = engine.reserveRuntimePublicationIdentity();
    const auto bootstrapGeneration = publicationIdentity.generation;
    const auto bootstrapWorldId = publicationIdentity.worldId;
    const auto bootstrapPublicationSequence = publicationIdentity.publicationSequence;

    auto worldOwner = RuntimePublishWorld::createForBuilder(RuntimePublishWorld::BuilderToken {});
    worldOwner->assertMutable();

    // Identity
    worldOwner->worldId = bootstrapWorldId;
    worldOwner->generation = bootstrapGeneration;
    worldOwner->runtimeVersion = bootstrapGeneration;
    worldOwner->transitionId = 0;

    // Schema
    worldOwner->schemaVersion = convo::isr::kRuntimeSemanticSchemaVersion;
    worldOwner->metadata.schemaVersion = worldOwner->schemaVersion;
    worldOwner->metadata.publicationSequence = bootstrapPublicationSequence;

    // GenerationSemantic
    worldOwner->generationSemantic.runtimeGeneration = bootstrapGeneration;
    worldOwner->generationSemantic.activationEpoch = bootstrapGeneration;

    // Topology (all zero/default: no active runtime, no fading runtime)
    worldOwner->topology.runtimeUuid = 0;
    worldOwner->topology.fadingRuntimeUuid = 0;
    // ★ v8.3: hasFadingRuntime 削除（Bootstrap World では fadingRuntimeUuid=0 で代替）

    // Routing (default processing order 0, no bypass)
    worldOwner->routing.processingOrder = 0;
    worldOwner->routing.eqBypassed = false;
    worldOwner->routing.convBypassed = false;

    // Execution (idle, no transition)
    worldOwner->execution.transitionActive = false;
    worldOwner->execution.transitionPolicy = 0;
    worldOwner->execution.latencyCompensationSamples = 0;
    worldOwner->execution.crossfadeStartDelayBlocks = 0;
    worldOwner->execution.crossfadeDryHoldSamples = 0;

    // Publication
    worldOwner->publication.sequenceId = bootstrapPublicationSequence;
    worldOwner->publication.epoch = static_cast<convo::isr::PublicationEpoch>(bootstrapGeneration);
    worldOwner->publication.mappedRuntimeGeneration = bootstrapGeneration;
    worldOwner->publication.previousSequenceId = 0;

    // Overlap (no crossfade)
    worldOwner->overlap.useDryAsOld = false;
    worldOwner->overlap.firstIrDryCrossfadePending = false;
    worldOwner->overlap.dryScaleTarget = 1.0;
    worldOwner->overlap.fadeTimeSec = 0.0;

    // Retire (zero backlog)
    worldOwner->retire.retireEpoch = bootstrapGeneration;
    worldOwner->retire.retireBacklog = 0;
    worldOwner->retire.deferredResidency = 0;

    // Timing
    worldOwner->timing.sampleRateHz = 48000.0;
    worldOwner->timing.queuedFadeTimeSec = 0.0;

    // Latency (zero)
    worldOwner->latency.latencyDelayOld = 0;
    worldOwner->latency.latencyDelayNew = 0;
    worldOwner->latency.latencyDeltaSamples = 0;

    // Resource (defaults)
    worldOwner->resource.oversamplingFactor = 1;
    worldOwner->resource.ditherBitDepth = 0;
    worldOwner->resource.noiseShaperType = 0;

    // Affinity
    worldOwner->affinity.rebuildWorkerRunning = false;

    // Automation (defaults)
    worldOwner->automation.eqBypassed = false;
    worldOwner->automation.convBypassed = false;
    worldOwner->automation.softClipEnabled = false;
    worldOwner->automation.saturationAmount = 0.0;
    worldOwner->automation.inputHeadroomGain = 1.0;
    worldOwner->automation.outputMakeupGain = 1.0;
    worldOwner->automation.convolverInputTrimGain = 1.0;

    // Coefficient (no adaptive coefficients yet)
    worldOwner->coefficient.adaptiveCoeffBankIndex = -1;
    worldOwner->coefficient.adaptiveCoeffGeneration = 0;
    worldOwner->coefficient.eqCoeffHash = 0;

    // ProjectionFreshness
    worldOwner->projectionFreshness.projectionGeneration = bootstrapGeneration;
    worldOwner->projectionFreshness.projectionRevision = bootstrapGeneration;
    worldOwner->projectionFreshness.maxStalenessWindows = 1u;

    worldOwner->freeze();
    return worldOwner;
}

convo::aligned_unique_ptr<const RuntimePublishWorld>
RuntimeBuilder::buildRuntimePublishWorld(
    const convo::RuntimeBuildSnapshot* sealedSnapshot,
    const RuntimePublishSpecification& spec) noexcept
{
    const auto publicationIdentity = engine.reserveRuntimePublicationIdentity();
    const auto nextGraphGeneration = publicationIdentity.generation;
    const auto nextWorldId = publicationIdentity.worldId;
    const auto nextPublicationSequence = publicationIdentity.publicationSequence;

    // ★ v8.3: Specification から current/next 解決（INV-12: Builder は Spec 以外 consult しない）
    auto* current = spec.topology.activeDSP;
    auto* next = spec.topology.fadingDSP;
    const convo::TransitionPolicy policy = static_cast<convo::TransitionPolicy>(spec.execution.transitionPolicy);
    const double fadeTimeSec = spec.execution.fadeTimeSec;
    const bool active = spec.execution.transitionActive;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    {
        diagLog("[DIAG_AUTH] BuilderEntry gen=" + juce::String(static_cast<juce::int64>(nextGraphGeneration))
            + " transitionActive=" + juce::String(static_cast<int>(active))
            + " currentUuid=" + juce::String(static_cast<juce::int64>(current ? current->runtimeUuid : 0))
            + " nextUuid=" + juce::String(static_cast<juce::int64>(next ? next->runtimeUuid : 0)));
    }
#endif

    // ★ v9.5 P1 phase2: computeRuntimePublishComputation() を排除。
    //   Runtime Query 部分は Orchestrator が事前に実行済み（currentRuntimeWorld, previousCommittedSequence）。
    //   Builder は Pure Calculation（makeEngineRuntimeState / makeRuntimeGraphState）のみを実行。
    const auto previousCommittedSequence = spec.publicationSnapshot.previousCommittedSequence;
    auto engineState = engine.makeEngineRuntimeState(
        const_cast<AudioEngine::DSPCore*>(current),
        const_cast<AudioEngine::DSPCore*>(next),
        policy, fadeTimeSec, active,
        spec.currentRuntimeWorld);
#pragma warning(push)
#pragma warning(disable : 4996)
    engineState.revision = nextGraphGeneration;
#pragma warning(pop)
    auto graphState = engine.makeRuntimeGraphState(engineState);

    auto worldOwner = RuntimePublishWorld::createForBuilder(RuntimePublishWorld::BuilderToken {});
    worldOwner->assertMutable();
    worldOwner->worldId = nextWorldId;
    worldOwner->generation = nextGraphGeneration;
    worldOwner->engine = engineState;
    worldOwner->graph = graphState;
    worldOwner->runtimeVersion = nextGraphGeneration;
    worldOwner->transitionId = nextGraphGeneration + (active ? 0x1000000000000000ULL : 0);

    worldOwner->schemaVersion = convo::isr::kRuntimeSemanticSchemaVersion;
    worldOwner->metadata.schemaVersion = worldOwner->schemaVersion;
    worldOwner->metadata.publicationSequence = nextPublicationSequence;

    worldOwner->generationSemantic.runtimeGeneration = nextGraphGeneration;
    worldOwner->generationSemantic.activationEpoch = nextGraphGeneration;

    // Topology: Specification の topology 部を写像
    {
        worldOwner->topology.runtimeUuid = (current != nullptr) ? current->runtimeUuid : 0;
        worldOwner->topology.fadingRuntimeUuid = (active && next != nullptr) ? next->runtimeUuid : 0;
        // ★ v8.3: hasFadingRuntime 削除 — graphState.fadingNode != nullptr から導出

        const double sr = (current != nullptr) ? current->sampleRate : 48000.0;
        worldOwner->timing.sampleRateHz = sr;
        worldOwner->timing.queuedFadeTimeSec = fadeTimeSec;

        worldOwner->resource.oversamplingFactor = (current != nullptr)
            ? static_cast<int>(current->oversamplingFactor) : 1;
        worldOwner->resource.ditherBitDepth = (current != nullptr)
            ? current->ditherBitDepth : 0;
        worldOwner->resource.noiseShaperType = (current != nullptr)
            ? static_cast<int>(current->noiseShaperType) : 0;
    }

    // [PR-4] DSP semantic projection (current DSPCore → RuntimeWorld for crossfade/admission)
    // [PR-2] Changed from DSPCore direct read to sealedSnapshot values
    {
        if (sealedSnapshot != nullptr) {
            worldOwner->dspProjection.irLoaded = sealedSnapshot->irLoaded;
            worldOwner->dspProjection.irFinalized = sealedSnapshot->irFinalized;
            worldOwner->dspProjection.structuralHash = sealedSnapshot->structuralHash;
            worldOwner->dspProjection.oversamplingFactor = sealedSnapshot->oversamplingFactor;
            worldOwner->dspProjection.sampleRate = sealedSnapshot->sampleRate;
            worldOwner->dspProjection.baseLatencySamples = sealedSnapshot->baseLatencySamples;
        } else if (current != nullptr) {
            worldOwner->dspProjection.irLoaded = current->convolverRt().isIRLoaded();
            worldOwner->dspProjection.irFinalized = current->convolverRt().isIRFinalized();
            worldOwner->dspProjection.structuralHash = current->convolverRt().getStructuralHash();
            worldOwner->dspProjection.oversamplingFactor = static_cast<int>(current->oversamplingFactor);
            worldOwner->dspProjection.sampleRate = current->sampleRate;
            worldOwner->dspProjection.baseLatencySamples = engine.estimateRuntimeLatencyBaseRateSamples(current, false);
        }
    }

    // sealedSnapshot Authority 経路と fallback 経路の分離
    const bool useSealedSnapshot = (sealedSnapshot != nullptr);
    {
        // ★ v9.5 P2: CrossfadeSnapshotPart/LatencyPart から読み取り（Orchestrator が収集済み）
        const int latencyDelayOld = spec.latency.latencyDelayOld;
        const int latencyDelayNew = spec.latency.latencyDelayNew;
        const int startDelayBlocks = spec.crossfade.startDelayBlocks;
        const int dryHoldSamples = spec.crossfade.dryHoldSamples;
        const double dryScaleTarget = spec.crossfade.dryScaleTarget;
        const bool firstIrDry = spec.crossfade.firstIrDryCrossfadePending;
        const std::uint64_t retireBacklog = spec.retire.retireQueueDepth;
        const bool rebuildWorkerRunning = false;

        // Execution/Routing: Specification を忠実に写像（INV-12）
        worldOwner->execution.transitionActive = spec.execution.transitionActive;
        worldOwner->execution.transitionPolicy = spec.execution.transitionPolicy;
        worldOwner->routing.processingOrder = spec.routing.processingOrder;
        worldOwner->routing.eqBypassed = spec.routing.eqBypassed;
        worldOwner->routing.convBypassed = spec.routing.convBypassed;

        // ★ v8.3: Overlap/latency は engine の現在値から設定（CrossfadeRuntime と連携）
        worldOwner->overlap.useDryAsOld = (policy == convo::TransitionPolicy::DryAsOld);
        worldOwner->overlap.fadeTimeSec = fadeTimeSec;
        worldOwner->overlap.firstIrDryCrossfadePending = firstIrDry;
        worldOwner->overlap.dryScaleTarget = dryScaleTarget;

        worldOwner->execution.crossfadeStartDelayBlocks = startDelayBlocks;
        worldOwner->execution.crossfadeDryHoldSamples = dryHoldSamples;

        worldOwner->latency.latencyDelayOld = latencyDelayOld;
        worldOwner->latency.latencyDelayNew = latencyDelayNew;

        worldOwner->retire.retireBacklog = retireBacklog;
        worldOwner->affinity.rebuildWorkerRunning = rebuildWorkerRunning;

        // ★ v9.4 P0: ProcessingPart から読み取り（Orchestrator が sealedSnapshot/atomic から収集済み）
        //   これにより Builder が engine atomic を直接読むパスが排除された。
        worldOwner->automation.eqBypassed = spec.processing.eqBypassed;
        worldOwner->automation.convBypassed = spec.processing.convBypassed;
        worldOwner->automation.softClipEnabled = spec.processing.softClipEnabled;
        worldOwner->automation.saturationAmount = spec.processing.saturationAmount;
        worldOwner->automation.inputHeadroomGain = spec.processing.inputHeadroomGain;
        worldOwner->automation.outputMakeupGain = spec.processing.outputMakeupGain;
        worldOwner->automation.convolverInputTrimGain = spec.processing.convolverInputTrimGain;
        // ★ Resource/Timing は current DSPCore から取得（Specification の将来拡張対象）
        if (useSealedSnapshot)
        {
            const auto& sealedBuildInput = sealedSnapshot->buildInput;
            worldOwner->resource.ditherBitDepth = sealedBuildInput.ditherBitDepth;
            worldOwner->resource.noiseShaperType = sealedBuildInput.noiseShaperType;
            worldOwner->timing.sampleRateHz = sealedBuildInput.sampleRate;
        }
        else
        {
            worldOwner->resource.oversamplingFactor = (current != nullptr)
                ? static_cast<int>(current->oversamplingFactor) : 1;
            worldOwner->resource.ditherBitDepth = (current != nullptr)
                ? current->ditherBitDepth : 0;
            worldOwner->resource.noiseShaperType = (current != nullptr)
                ? static_cast<int>(current->noiseShaperType) : 0;
            worldOwner->timing.sampleRateHz = (current != nullptr) ? current->sampleRate : 48000.0;
        }
    }

    worldOwner->publication.sequenceId = nextPublicationSequence;
    worldOwner->publication.epoch = static_cast<convo::isr::PublicationEpoch>(nextGraphGeneration);
    worldOwner->publication.mappedRuntimeGeneration = nextGraphGeneration;
    worldOwner->publication.previousSequenceId = previousCommittedSequence;

    worldOwner->retire.retireEpoch = nextGraphGeneration;
    worldOwner->retire.deferredResidency = 0;
    worldOwner->latency.latencyDeltaSamples = 0;

    // Coefficient fields
    const int bankIndex = spec.adaptive.coeffBankIndex;
    worldOwner->coefficient.adaptiveCoeffBankIndex = bankIndex;
    worldOwner->coefficient.adaptiveCoeffGeneration = static_cast<uint32_t>(spec.adaptive.coeffGeneration);
    worldOwner->coefficient.eqCoeffHash = 0;

    worldOwner->projectionFreshness.projectionGeneration = nextGraphGeneration;
    worldOwner->projectionFreshness.projectionRevision = nextGraphGeneration;
    worldOwner->projectionFreshness.maxStalenessWindows = 1u;

    // Semantic hashes (保留 — 現状維持)
    worldOwner->semanticHash.generationSemanticHash = worldOwner->generationSemantic.runtimeGeneration
        ^ (worldOwner->generationSemantic.activationEpoch << 1);
    worldOwner->semanticHash.topologyHash = worldOwner->topology.runtimeUuid
        ^ (worldOwner->topology.fadingRuntimeUuid << 1)
        ^ ((worldOwner->topology.fadingRuntimeUuid != 0) ? 0x9E3779B97F4A7C15ull : 0ull);
    worldOwner->semanticHash.executionHash = static_cast<std::uint64_t>(worldOwner->execution.transitionPolicy + 0x9E3779B9)
        ^ (worldOwner->execution.transitionActive ? 0x517CC1B727220A95ull : 0ull);
    worldOwner->semanticHash.routingHash = static_cast<std::uint64_t>(worldOwner->routing.processingOrder + 0x85EBCA6Bu)
        ^ (worldOwner->routing.eqBypassed ? 0x27D4EB2Full : 0ull)
        ^ (worldOwner->routing.convBypassed ? 0x165667B1ull : 0ull);

    // freeze は caller (coordinator.publishWorld) が行う

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    diagLog("[DIAG_AUTH] BuilderExit gen=" + juce::String(static_cast<juce::int64>(worldOwner->generation))
        + " graph.fadingNode=" + juce::String(static_cast<juce::int64>(reinterpret_cast<intptr_t>(worldOwner->graph.fadingNode)))
        + " fadingRuntimeUuid=" + juce::String(static_cast<juce::int64>(worldOwner->topology.fadingRuntimeUuid))
        + " transitionActive=" + juce::String(static_cast<int>(worldOwner->execution.transitionActive)));
#endif

    return worldOwner;
}

BuildResult RuntimeBuilder::build(const BuildInput& in,
                                  const ConvolverProcessor::BuildSnapshot& convolverBuildSnapshot) noexcept
{
    BuildResult result {};

    if (in.sampleRate <= 0.0 || in.blockSize <= 0)
    {
        result.error = BuildError::InvalidInput;
        return result;
    }

    convo::aligned_unique_ptr<AudioEngine::DSPCore> runtime;

    try
    {
        runtime = convo::aligned_make_unique<AudioEngine::DSPCore>();
        runtime->convolverRt().setVisualizationEnabled(false);
        runtime->convolverRt().applyBuildSnapshot(convolverBuildSnapshot);
        // Transfer actual IR data (applyBuildSnapshot only copies metadata, not the AudioBuffer)
        runtime->convolverRt().transferIRStateFrom(engine.getConvolverProcessor());
        runtime->prepare(in.sampleRate,
                         in.blockSize,
                         in.ditherBitDepth,
                         in.oversamplingFactor,
                         static_cast<AudioEngine::OversamplingType>(in.oversamplingType),
                         static_cast<AudioEngine::NoiseShaperType>(in.noiseShaperType),
                         &engine);
        result.runtime = runtime.release();
        result.prepared = true;
        return result;
    }
    catch (const std::bad_alloc&)
    {
        result.error = BuildError::ResourceUnavailable;
        return result;
    }
    catch (...)
    {
        result.error = BuildError::InternalError;
        return result;
    }
}

BuildError RuntimeBuilder::validateWarmup(const AudioEngine::DSPCore& runtime) const noexcept
{
    juce::ignoreUnused(engine);

    if (runtime.convolverRt().isIRLoaded() && !runtime.convolverRt().isIRFinalized())
        return BuildError::WarmupFailed;

    return BuildError::None;
}

} // namespace convo
