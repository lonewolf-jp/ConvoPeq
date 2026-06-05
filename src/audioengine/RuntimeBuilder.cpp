#include "RuntimeBuilder.h"

#include <bit>
#include <cstdint>

namespace convo {

namespace {

[[nodiscard]] std::uint64_t hashBuildInput(const BuildInput& buildInput) noexcept
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
    worldOwner->topology.hasFadingRuntime = false;

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

    return worldOwner;
}

convo::aligned_unique_ptr<RuntimePublishWorld>
RuntimeBuilder::buildRuntimePublishWorld(AudioEngine::DSPCore* current,
                                         AudioEngine::DSPCore* next,
                                         convo::TransitionPolicy policy,
                                         double fadeTimeSec,
                                         bool active,
                                         const convo::RuntimeBuildSnapshot* sealedSnapshot) noexcept
{
    const auto publicationIdentity = engine.reserveRuntimePublicationIdentity();
    const auto nextGraphGeneration = publicationIdentity.generation;
    const auto nextWorldId = publicationIdentity.worldId;
    const auto nextPublicationSequence = publicationIdentity.publicationSequence;

    auto publishComputation = engine.computeRuntimePublishComputation(current,
                                                                      next,
                                                                      policy,
                                                                      fadeTimeSec,
                                                                      active,
                                                                      nextGraphGeneration);
    auto engineState = std::move(publishComputation.engineState);
    auto graphState = std::move(publishComputation.graphState);
    const auto previousCommittedSequence = publishComputation.previousCommittedSequence;

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

    // 3.2.9: RuntimeGraph から Authoritative フィールド削除済み。
    // topology/routing/resource/automation 等の Semantic 構造体から値を参照。
    // [C4996 fix] engineState (EngineRuntime) は deprecated のため、
    // 関数パラメータ current/next および DSPCore 直アクセスに置換。
    {
        const bool hasFading = (graphState.fadingNode != nullptr);

        worldOwner->topology.runtimeUuid = (current != nullptr) ? current->runtimeUuid : 0;
        worldOwner->topology.fadingRuntimeUuid = (next != nullptr) ? next->runtimeUuid : 0;
        worldOwner->topology.hasFadingRuntime = hasFading;

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
    {
        if (current != nullptr) {
            worldOwner->dspProjection.irLoaded = current->convolverRt().isIRLoaded();
            worldOwner->dspProjection.irFinalized = current->convolverRt().isIRFinalized();
            worldOwner->dspProjection.structuralHash = current->convolverRt().getStructuralHash();
            worldOwner->dspProjection.oversamplingFactor = static_cast<int>(current->oversamplingFactor);
            worldOwner->dspProjection.sampleRate = current->sampleRate;
            worldOwner->dspProjection.baseLatencySamples = engine.estimateRuntimeLatencyBaseRateSamples(current, false);
        }
    }

    // [100%] sealedSnapshot Authority (選択肢A):
    // sealedSnapshot が存在する場合、その BuildInput 値が唯一の Authority。
    // atomic からの読み取りは sealedSnapshot 不在時（bootstrap/fallback）のみ行う。
    // ★ sealedSnapshot 存在時、重複フィールドの atomic 読み取りを完全にスキップする。
    const bool useSealedSnapshot = (sealedSnapshot != nullptr);
    // [C4996 fix] Read value fields from AudioEngine atomics directly (was engineState.X)
    {
        // These values are obtained from AudioEngine's atomic members (non-deprecated access)
        const auto processingOrder = useSealedSnapshot ? static_cast<convo::ProcessingOrder>(sealedSnapshot->buildInput.processingOrder)
            : convo::consumeAtomic(engine.currentProcessingOrder, std::memory_order_acquire);
        const bool eqBypassed = useSealedSnapshot ? sealedSnapshot->buildInput.eqBypassed
            : convo::consumeAtomic(engine.eqBypassActive, std::memory_order_acquire);
        const bool convBypassed = useSealedSnapshot ? sealedSnapshot->buildInput.convBypassed
            : convo::consumeAtomic(engine.convBypassActive, std::memory_order_acquire);
        const bool softClipEnabled = useSealedSnapshot ? sealedSnapshot->buildInput.softClipEnabled
            : convo::consumeAtomic(engine.softClipEnabled, std::memory_order_acquire);
        const float saturationAmount = useSealedSnapshot ? static_cast<float>(sealedSnapshot->buildInput.saturationAmount)
            : static_cast<float>(convo::consumeAtomic(engine.saturationAmount, std::memory_order_acquire));
        const float inputHeadroomGain = useSealedSnapshot ? static_cast<float>(sealedSnapshot->buildInput.inputHeadroomGain)
            : static_cast<float>(convo::consumeAtomic(engine.inputHeadroomGain, std::memory_order_acquire));
        const float outputMakeupGain = useSealedSnapshot ? static_cast<float>(sealedSnapshot->buildInput.outputMakeupGain)
            : static_cast<float>(convo::consumeAtomic(engine.outputMakeupGain, std::memory_order_acquire));
        const float convolverInputTrimGain = useSealedSnapshot ? static_cast<float>(sealedSnapshot->buildInput.convolverInputTrimGain)
            : static_cast<float>(convo::consumeAtomic(engine.convolverInputTrimGain, std::memory_order_acquire));
        // Non-sealedSnapshot fields: always read from atomics
        const int latencyDelayOld = convo::consumeAtomic(engine.latencyDelayOld, std::memory_order_acquire);
        const int latencyDelayNew = static_cast<int>(convo::consumeAtomic(engine.latencyDelayNew, std::memory_order_acquire));
        const int startDelayBlocks = convo::consumeAtomic(engine.dspCrossfadeStartDelayBlocks, std::memory_order_acquire);
        const int dryHoldSamples = convo::consumeAtomic(engine.dspCrossfadeDryHoldSamples, std::memory_order_acquire);
        const double dryScaleTarget = convo::consumeAtomic(engine.dspCrossfadeDryScaleTarget, std::memory_order_acquire);
        const bool firstIrDry = convo::consumeAtomic(engine.firstIrDryCrossfadePending, std::memory_order_acquire);
        const std::uint64_t retireBacklog = convo::consumeAtomic(engine.retireQueueDepth_, std::memory_order_acquire);
        const bool rebuildWorkerRunning = false; // not available as atomic; default false

        worldOwner->routing.processingOrder = static_cast<int>(processingOrder);
        worldOwner->routing.eqBypassed = eqBypassed;
        worldOwner->routing.convBypassed = convBypassed;

        worldOwner->execution.transitionActive = active;
        worldOwner->execution.transitionPolicy = static_cast<int>(policy);
        worldOwner->execution.crossfadeStartDelayBlocks = startDelayBlocks;
        worldOwner->execution.crossfadeDryHoldSamples = dryHoldSamples;

        worldOwner->overlap.useDryAsOld = active;
        worldOwner->overlap.fadeTimeSec = fadeTimeSec;
        worldOwner->overlap.firstIrDryCrossfadePending = firstIrDry;
        worldOwner->overlap.dryScaleTarget = dryScaleTarget;

        worldOwner->automation.eqBypassed = eqBypassed;
        worldOwner->automation.convBypassed = convBypassed;
        worldOwner->automation.softClipEnabled = softClipEnabled;
        worldOwner->automation.saturationAmount = saturationAmount;
        worldOwner->automation.inputHeadroomGain = inputHeadroomGain;
        worldOwner->automation.outputMakeupGain = outputMakeupGain;
        worldOwner->automation.convolverInputTrimGain = convolverInputTrimGain;

        worldOwner->latency.latencyDelayOld = latencyDelayOld;
        worldOwner->latency.latencyDelayNew = latencyDelayNew;

        worldOwner->retire.retireBacklog = retireBacklog;

        worldOwner->affinity.rebuildWorkerRunning = rebuildWorkerRunning;

        // [100%] sealedSnapshot が存在する場合、resource/timing フィールドも atomic ではなく snapshot から設定
        if (useSealedSnapshot)
        {
            worldOwner->resource.oversamplingFactor = sealedSnapshot->buildInput.oversamplingFactor;
            worldOwner->resource.ditherBitDepth = sealedSnapshot->buildInput.ditherBitDepth;
            worldOwner->resource.noiseShaperType = sealedSnapshot->buildInput.noiseShaperType;
            worldOwner->timing.sampleRateHz = sealedSnapshot->buildInput.sampleRate;
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

    // activationEpoch is derived from generationSemantic.activationEpoch (#17 Sprint-1)
    // Do not set timing.activationEpoch directly

    worldOwner->latency.latencyDeltaSamples = 0;

    // SchedulingSemantic fields are derived from ExecutionSemantic (#16 Sprint-1)
    // Do not set scheduling.* directly - they mirror execution.* for backward compatibility
    // TODO: Remove SchedulingSemantic entirely after all consumers migrate to execution.*

    worldOwner->coefficient.adaptiveCoeffBankIndex = 0;
    worldOwner->coefficient.adaptiveCoeffGeneration = 0;
    worldOwner->coefficient.eqCoeffHash = 0;

    worldOwner->projectionFreshness.projectionGeneration = nextGraphGeneration;
    worldOwner->projectionFreshness.projectionRevision = nextGraphGeneration;
    worldOwner->projectionFreshness.maxStalenessWindows = 1u;

    // [100%] sealedSnapshot override block は削除: 選択肢A により原子ブロック内で完結。
    // 重複フィールド設定は上記の useSealedSnapshot 分岐で atomic 読み取りをスキップしているため、
    // 別途 sealedSnapshot ブロックでの再設定は不要。

    worldOwner->semanticHash.generationSemanticHash = worldOwner->generationSemantic.runtimeGeneration
        ^ (worldOwner->generationSemantic.activationEpoch << 1);
    worldOwner->semanticHash.topologyHash = worldOwner->topology.runtimeUuid
        ^ (worldOwner->topology.fadingRuntimeUuid << 1)
        ^ (worldOwner->topology.hasFadingRuntime ? 0x9E3779B97F4A7C15ull : 0ull);
    worldOwner->semanticHash.executionHash = static_cast<std::uint64_t>(worldOwner->execution.transitionPolicy + 0x9E3779B9)
        ^ (worldOwner->execution.transitionActive ? 0x517CC1B727220A95ull : 0ull)
        ^ (static_cast<std::uint64_t>(worldOwner->execution.latencyCompensationSamples + 0x80000000ull) << 1)
        ^ (static_cast<std::uint64_t>(worldOwner->execution.crossfadeStartDelayBlocks + 0x80000000ull) << 2)
        ^ (static_cast<std::uint64_t>(worldOwner->execution.crossfadeDryHoldSamples + 0x80000000ull) << 3);
    worldOwner->semanticHash.routingHash = static_cast<std::uint64_t>(worldOwner->routing.processingOrder + 0x85EBCA6Bu)
        ^ (worldOwner->routing.eqBypassed ? 0x27D4EB2Full : 0ull)
        ^ (worldOwner->routing.convBypassed ? 0x165667B1ull : 0ull);
    worldOwner->semanticHash.payloadHash = static_cast<std::uint64_t>(worldOwner->resource.oversamplingFactor + 0x9E37)
        ^ (static_cast<std::uint64_t>(worldOwner->resource.ditherBitDepth + 0x100) << 8)
        ^ (static_cast<std::uint64_t>(worldOwner->resource.noiseShaperType + 0x10) << 16)
        ^ worldOwner->coefficient.eqCoeffHash
        ^ (static_cast<std::uint64_t>(worldOwner->coefficient.adaptiveCoeffBankIndex + 0x100) << 24)
        ^ (static_cast<std::uint64_t>(worldOwner->coefficient.adaptiveCoeffGeneration) << 32)
        ^ std::bit_cast<std::uint64_t>(worldOwner->automation.saturationAmount)
        ^ std::bit_cast<std::uint64_t>(worldOwner->automation.inputHeadroomGain)
        ^ std::bit_cast<std::uint64_t>(worldOwner->automation.outputMakeupGain)
        ^ std::bit_cast<std::uint64_t>(worldOwner->automation.convolverInputTrimGain)
        ^ (worldOwner->automation.softClipEnabled ? 0xD1B54A32D192ED03ull : 0ull);
    if (sealedSnapshot != nullptr)
    {
        worldOwner->semanticHash.payloadHash = hashBuildInput(sealedSnapshot->buildInput);
    }
    worldOwner->semanticHash.publicationSemanticHash = worldOwner->publication.sequenceId
        ^ (worldOwner->publication.epoch << 1)
        ^ (worldOwner->publication.mappedRuntimeGeneration << 2)
        ^ (worldOwner->publication.previousSequenceId << 3);
    worldOwner->semanticHash.overlapSemanticHash = (worldOwner->overlap.useDryAsOld ? 0xA24BAED4963EE407ull : 0ull)
        ^ (worldOwner->overlap.firstIrDryCrossfadePending ? 0x9FB21C651E98DF25ull : 0ull)
        ^ (worldOwner->execution.transitionActive ? 0xC2B2AE3D27D4EB4Full : 0ull);
    worldOwner->semanticHash.retireSemanticHash = worldOwner->retire.retireEpoch
        ^ (worldOwner->retire.retireBacklog << 1)
        ^ (worldOwner->retire.deferredResidency << 2);

    // Added for full inventory coverage (#18 Sprint-3)
    auto bitCast = [](double val) noexcept -> std::uint64_t {
        std::uint64_t res = 0;
        std::memcpy(&res, &val, sizeof(val));
        return res;
    };
    worldOwner->semanticHash.timingHash = bitCast(worldOwner->timing.sampleRateHz)
        ^ bitCast(worldOwner->timing.queuedFadeTimeSec);
    worldOwner->semanticHash.latencyHash = static_cast<std::uint64_t>(worldOwner->latency.latencyDelayOld + 0x80000000u)
        ^ (static_cast<std::uint64_t>(worldOwner->latency.latencyDelayNew + 0x80000000u) << 1)
        ^ (static_cast<std::uint64_t>(worldOwner->latency.latencyDeltaSamples + 0x80000000u) << 2);
    worldOwner->semanticHash.resourceHash = static_cast<std::uint64_t>(worldOwner->resource.oversamplingFactor)
        ^ (static_cast<std::uint64_t>(worldOwner->resource.ditherBitDepth + 0x100) << 8)
        ^ (static_cast<std::uint64_t>(worldOwner->resource.noiseShaperType + 0x10) << 16);
    worldOwner->semanticHash.automationHash = (worldOwner->automation.eqBypassed ? 0x9E3779B97F4A7C15ull : 0ull)
        ^ (worldOwner->automation.convBypassed ? 0x517CC1B727220A95ull : 0ull)
        ^ (worldOwner->automation.softClipEnabled ? 0xC2B2AE3D27D4EB4Full : 0ull)
        ^ bitCast(worldOwner->automation.saturationAmount)
        ^ bitCast(worldOwner->automation.inputHeadroomGain)
        ^ bitCast(worldOwner->automation.outputMakeupGain)
        ^ bitCast(worldOwner->automation.convolverInputTrimGain);
    worldOwner->semanticHash.coefficientHash = static_cast<std::uint64_t>(worldOwner->coefficient.adaptiveCoeffBankIndex + 0x100)
        ^ (static_cast<std::uint64_t>(worldOwner->coefficient.adaptiveCoeffGeneration) << 8)
        ^ (worldOwner->coefficient.eqCoeffHash << 16);

    worldOwner->freeze();
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
