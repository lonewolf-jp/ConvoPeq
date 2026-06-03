#include "RuntimeBuilder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <immintrin.h>

#include <mkl_vml.h>

namespace convo {

namespace {

void diagLogRuntimeBuilder(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

int getWarmupBlocksOverrideFromEnv() noexcept
{
    // static mutable 禁止 (rule 4.3.1, 4.3.3): 環境変数を都度読む
    int parsed = -1;
    if (const char* env = std::getenv("CONVOPEQ_WARMUP_BLOCKS"))
    {
        char* end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end != env)
            parsed = std::clamp(static_cast<int>(v), 0, 16);
    }
    return parsed;
}

// static mutable 禁止 (rule 4.3.1): WarmupBlockStats/statsByBlocks は削除。
// 各ビルドの elapsed のみをログする。
juce::String formatWarmupElapsedSummary(int blocks, std::uint64_t elapsedUs)
{
    return "[DIAG] warmup summary blocks="
        + juce::String(blocks)
        + " elapsedMs=" + juce::String(static_cast<double>(elapsedUs) / 1000.0, 3);
}

enum class WarmupPhase : int
{
    Init = 0,
    ZeroState,
    DenormalGuardEnable,
    WarmSignal,
    Ready
};

const char* warmupPhaseToString(WarmupPhase phase) noexcept
{
    switch (phase)
    {
        case WarmupPhase::Init: return "INIT";
        case WarmupPhase::ZeroState: return "ZERO_STATE";
        case WarmupPhase::DenormalGuardEnable: return "DENORMAL_GUARD_ENABLE";
        case WarmupPhase::WarmSignal: return "WARM_SIGNAL";
        case WarmupPhase::Ready: return "READY";
    }
    return "UNKNOWN";
}

juce::String formatWarmupPhaseTimingSummary(const std::array<std::uint64_t, 5>& phaseElapsedUs)
{
    const std::uint64_t totalUs = phaseElapsedUs[0]
        + phaseElapsedUs[1]
        + phaseElapsedUs[2]
        + phaseElapsedUs[3]
        + phaseElapsedUs[4];

    return "[DIAG] warmup phase-ms init=" + juce::String(static_cast<double>(phaseElapsedUs[0]) / 1000.0, 3)
        + " zero=" + juce::String(static_cast<double>(phaseElapsedUs[1]) / 1000.0, 3)
        + " denorm=" + juce::String(static_cast<double>(phaseElapsedUs[2]) / 1000.0, 3)
        + " signal=" + juce::String(static_cast<double>(phaseElapsedUs[3]) / 1000.0, 3)
        + " ready=" + juce::String(static_cast<double>(phaseElapsedUs[4]) / 1000.0, 3)
        + " total=" + juce::String(static_cast<double>(totalUs) / 1000.0, 3);
}

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

class ScopedWarmupDenormalGuard
{
public:
    ScopedWarmupDenormalGuard() noexcept
    {
        oldMxcsr = _mm_getcsr();
        oldVmlMode = vmlGetMode();

        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    }

    ~ScopedWarmupDenormalGuard() noexcept
    {
        _mm_setcsr(oldMxcsr);
        vmlSetMode(oldVmlMode);
    }

    ScopedWarmupDenormalGuard(const ScopedWarmupDenormalGuard&) = delete;
    ScopedWarmupDenormalGuard& operator=(const ScopedWarmupDenormalGuard&) = delete;

private:
    unsigned int oldMxcsr { 0 };
    int oldVmlMode { 0 };
};

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

    worldOwner->topology.runtimeUuid = graphState.runtimeUuid;
    worldOwner->topology.fadingRuntimeUuid = graphState.fadingRuntimeUuid;
    worldOwner->topology.hasFadingRuntime = (graphState.fadingNode != nullptr);

    worldOwner->routing.processingOrder = engineState.processingOrder;
    worldOwner->routing.eqBypassed = engineState.eqBypassed;
    worldOwner->routing.convBypassed = engineState.convBypassed;

    worldOwner->execution.transitionActive = engineState.transition.active;
    worldOwner->execution.transitionPolicy = static_cast<int>(engineState.transition.policy);
    worldOwner->execution.latencyCompensationSamples = engineState.transition.latencyDeltaSamples;
    worldOwner->execution.crossfadeStartDelayBlocks = engineState.dspCrossfadeStartDelayBlocks;
    worldOwner->execution.crossfadeDryHoldSamples = engineState.dspCrossfadeDryHoldSamples;

    worldOwner->publication.sequenceId = nextPublicationSequence;
    worldOwner->publication.epoch = static_cast<convo::isr::PublicationEpoch>(nextGraphGeneration);
    worldOwner->publication.mappedRuntimeGeneration = nextGraphGeneration;
    worldOwner->publication.previousSequenceId = previousCommittedSequence;

    worldOwner->overlap.useDryAsOld = engineState.dspCrossfadeUseDryAsOld;
    worldOwner->overlap.firstIrDryCrossfadePending = engineState.firstIrDryCrossfadePending;
    worldOwner->overlap.dryScaleTarget = engineState.dryScaleTarget;
    worldOwner->overlap.fadeTimeSec = engineState.queuedFadeTimeSec;

    worldOwner->retire.retireEpoch = nextGraphGeneration;
    worldOwner->retire.retireBacklog = engineState.retireBacklog;
    worldOwner->retire.deferredResidency = engineState.deferredResidency;

    worldOwner->timing.sampleRateHz = graphState.sampleRate;
    worldOwner->timing.queuedFadeTimeSec = engineState.queuedFadeTimeSec;
    // activationEpoch is derived from generationSemantic.activationEpoch (#17 Sprint-1)
    // Do not set timing.activationEpoch directly

    worldOwner->latency.latencyDelayOld = engineState.latencyDelayOld;
    worldOwner->latency.latencyDelayNew = engineState.latencyDelayNew;
    worldOwner->latency.latencyDeltaSamples = engineState.transition.latencyDeltaSamples;

    // SchedulingSemantic fields are derived from ExecutionSemantic (#16 Sprint-1)
    // Do not set scheduling.* directly - they mirror execution.* for backward compatibility
    // TODO: Remove SchedulingSemantic entirely after all consumers migrate to execution.*

    worldOwner->resource.oversamplingFactor = graphState.oversamplingFactor;
    worldOwner->resource.ditherBitDepth = graphState.ditherBitDepth;
    worldOwner->resource.noiseShaperType = graphState.noiseShaperType;

    worldOwner->affinity.rebuildWorkerRunning = engineState.rebuildWorkerRunning;

    worldOwner->automation.eqBypassed = engineState.eqBypassed;
    worldOwner->automation.convBypassed = engineState.convBypassed;
    worldOwner->automation.softClipEnabled = engineState.softClipEnabled;
    worldOwner->automation.saturationAmount = engineState.saturationAmount;
    worldOwner->automation.inputHeadroomGain = engineState.inputHeadroomGain;
    worldOwner->automation.outputMakeupGain = engineState.outputMakeupGain;
    worldOwner->automation.convolverInputTrimGain = engineState.convolverInputTrimGain;

    worldOwner->coefficient.adaptiveCoeffBankIndex = engineState.adaptiveCoeffBankIndex;
    worldOwner->coefficient.adaptiveCoeffGeneration = engineState.adaptiveCoeffGeneration;
    worldOwner->coefficient.eqCoeffHash = engineState.eqCoeffHash;

    worldOwner->projectionFreshness.projectionGeneration = nextGraphGeneration;
    worldOwner->projectionFreshness.projectionRevision = graphState.generation;
    worldOwner->projectionFreshness.maxStalenessWindows = 1u;

    if (sealedSnapshot != nullptr)
    {
        jassert(sealedSnapshot->sealed);

        const auto& sealedBuildInput = sealedSnapshot->buildInput;

        worldOwner->engine.processingOrder = sealedBuildInput.processingOrder;
        worldOwner->engine.eqBypassed = sealedBuildInput.eqBypassed;
        worldOwner->engine.convBypassed = sealedBuildInput.convBypassed;
        worldOwner->engine.softClipEnabled = sealedBuildInput.softClipEnabled;
        worldOwner->engine.saturationAmount = sealedBuildInput.saturationAmount;
        worldOwner->engine.inputHeadroomGain = sealedBuildInput.inputHeadroomGain;
        worldOwner->engine.outputMakeupGain = sealedBuildInput.outputMakeupGain;
        worldOwner->engine.convolverInputTrimGain = sealedBuildInput.convolverInputTrimGain;

        worldOwner->graph.sampleRate = sealedBuildInput.sampleRate;
        worldOwner->graph.ditherBitDepth = sealedBuildInput.ditherBitDepth;
        worldOwner->graph.noiseShaperType = sealedBuildInput.noiseShaperType;
        worldOwner->graph.oversamplingFactor = sealedBuildInput.oversamplingFactor;
        worldOwner->graph.eqBypassed = sealedBuildInput.eqBypassed;
        worldOwner->graph.convBypassed = sealedBuildInput.convBypassed;
        worldOwner->graph.softClipEnabled = sealedBuildInput.softClipEnabled;
        worldOwner->graph.saturationAmount = sealedBuildInput.saturationAmount;
        worldOwner->graph.inputHeadroomGain = sealedBuildInput.inputHeadroomGain;
        worldOwner->graph.outputMakeupGain = sealedBuildInput.outputMakeupGain;
        worldOwner->graph.convolverInputTrimGain = sealedBuildInput.convolverInputTrimGain;

        worldOwner->routing.processingOrder = sealedBuildInput.processingOrder;
        worldOwner->routing.eqBypassed = sealedBuildInput.eqBypassed;
        worldOwner->routing.convBypassed = sealedBuildInput.convBypassed;

        worldOwner->timing.sampleRateHz = sealedBuildInput.sampleRate;

        worldOwner->resource.oversamplingFactor = sealedBuildInput.oversamplingFactor;
        worldOwner->resource.ditherBitDepth = sealedBuildInput.ditherBitDepth;
        worldOwner->resource.noiseShaperType = sealedBuildInput.noiseShaperType;

        worldOwner->automation.eqBypassed = sealedBuildInput.eqBypassed;
        worldOwner->automation.convBypassed = sealedBuildInput.convBypassed;
        worldOwner->automation.softClipEnabled = sealedBuildInput.softClipEnabled;
        worldOwner->automation.saturationAmount = sealedBuildInput.saturationAmount;
        worldOwner->automation.inputHeadroomGain = sealedBuildInput.inputHeadroomGain;
        worldOwner->automation.outputMakeupGain = sealedBuildInput.outputMakeupGain;
        worldOwner->automation.convolverInputTrimGain = sealedBuildInput.convolverInputTrimGain;
    }

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
        ^ (worldOwner->engine.transition.active ? 0xC2B2AE3D27D4EB4Full : 0ull);
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

int RuntimeBuilder::getRequiredWarmupBlocks(const AudioEngine::DSPCore& runtime) const noexcept
{
    // FIR 履歴初期化に必要なブロック数を決定
    // - IR 未ロード時: 0 ブロック（パススルーなので初期化不要）
    // - IR ロード時: レイテンシ量に応じて 1..4 ブロック
    if (!runtime.convolverRt().isIRLoaded())
        return 0;

    const int blockSize = std::max(1, runtime.convolverRt().getCurrentBufferSize());
    const int totalLatency = std::max(0, runtime.convolverRt().getTotalLatencySamples());
    const int blocksForLatency = (totalLatency + blockSize - 1) / blockSize;
    const int recommended = std::clamp(blocksForLatency + 1, 1, 4);

    const int overrideBlocks = getWarmupBlocksOverrideFromEnv();
    const int selected = (overrideBlocks >= 0) ? overrideBlocks : recommended;

    if (overrideBlocks >= 0)
    {
        diagLogRuntimeBuilder("[DIAG] warmup config: override blocks="
            + juce::String(overrideBlocks)
            + " recommended=" + juce::String(recommended)
            + " latencySamples=" + juce::String(totalLatency)
            + " blockSize=" + juce::String(blockSize));
    }
    else
    {
        diagLogRuntimeBuilder("[DIAG] warmup config: auto blocks="
            + juce::String(selected)
            + " latencySamples=" + juce::String(totalLatency)
            + " blockSize=" + juce::String(blockSize));
    }

    return selected;
}

BuildError RuntimeBuilder::executeWarmup(AudioEngine::DSPCore& runtime) noexcept
{
    const auto start = std::chrono::steady_clock::now();
    auto phaseStart = start;
    WarmupPhase phase = WarmupPhase::Init;
    std::array<std::uint64_t, 5> phaseElapsedUs { 0, 0, 0, 0, 0 };

    const auto accumulatePhaseUntil = [&](const std::chrono::steady_clock::time_point now) noexcept
    {
        const auto deltaUsSigned = std::chrono::duration_cast<std::chrono::microseconds>(now - phaseStart).count();
        const std::uint64_t deltaUs = static_cast<std::uint64_t>(std::max<std::int64_t>(deltaUsSigned, 0));
        const int phaseIndex = std::clamp(static_cast<int>(phase), 0, static_cast<int>(phaseElapsedUs.size()) - 1);
        phaseElapsedUs[static_cast<size_t>(phaseIndex)] += deltaUs;
        phaseStart = now;
    };

    const auto switchPhase = [&](WarmupPhase nextPhase) noexcept
    {
        const auto now = std::chrono::steady_clock::now();
        accumulatePhaseUntil(now);
        phase = nextPhase;
        diagLogRuntimeBuilder("[DIAG] warmup phase=" + juce::String(warmupPhaseToString(phase)));
    };

    diagLogRuntimeBuilder("[DIAG] warmup phase=" + juce::String(warmupPhaseToString(phase)));

    if (validateWarmup(runtime) != BuildError::None)
    {
        accumulatePhaseUntil(std::chrono::steady_clock::now());
        diagLogRuntimeBuilder("[DIAG] warmup result: failed early-validation irLoaded="
            + juce::String(static_cast<int>(runtime.convolverRt().isIRLoaded()))
            + " irFinalized=" + juce::String(static_cast<int>(runtime.convolverRt().isIRFinalized())));
        diagLogRuntimeBuilder("[DIAG] warmup failed phase=" + juce::String(warmupPhaseToString(phase)));
        diagLogRuntimeBuilder(formatWarmupPhaseTimingSummary(phaseElapsedUs));
        return BuildError::WarmupFailed;
    }

    const int warmupBlocks = getRequiredWarmupBlocks(runtime);
    const int blockSize = std::max(1, runtime.convolverRt().getCurrentBufferSize());
    const int totalLatency = std::max(0, runtime.convolverRt().getTotalLatencySamples());

    if (warmupBlocks <= 0)
    {
        accumulatePhaseUntil(std::chrono::steady_clock::now());
        const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        const std::uint64_t elapsedUsSafe = static_cast<std::uint64_t>(std::max<std::int64_t>(elapsedUs, 0));
        diagLogRuntimeBuilder("[DIAG] warmup result: skipped blocks=0"
            " elapsedMs=" + juce::String(static_cast<double>(elapsedUs) / 1000.0, 3)
            + " latencySamples=" + juce::String(totalLatency)
            + " blockSize=" + juce::String(blockSize));
        diagLogRuntimeBuilder(formatWarmupPhaseTimingSummary(phaseElapsedUs));
        diagLogRuntimeBuilder(formatWarmupElapsedSummary(0, elapsedUsSafe));
        return BuildError::None;
    }

    juce::AudioBuffer<double> warmupBuffer(2, blockSize);
    juce::dsp::AudioBlock<double> warmupBlock(warmupBuffer);

    switchPhase(WarmupPhase::ZeroState);
    warmupBuffer.clear();

    switchPhase(WarmupPhase::DenormalGuardEnable);
    ScopedWarmupDenormalGuard denormalGuard;

    switchPhase(WarmupPhase::WarmSignal);

    // 無音ブロックを流して Convolver の内部履歴を安定化する。
    for (int i = 0; i < warmupBlocks; ++i)
    {
        warmupBuffer.clear();
        runtime.convolverRt().process(warmupBlock);

        if (runtime.convolverRt().isIRLoaded() && !runtime.convolverRt().isIRFinalized())
        {
            accumulatePhaseUntil(std::chrono::steady_clock::now());
            const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count();
            diagLogRuntimeBuilder("[DIAG] warmup result: failed mid-run blocksDone="
                + juce::String(i + 1)
                + " / " + juce::String(warmupBlocks)
                + " elapsedMs=" + juce::String(static_cast<double>(elapsedUs) / 1000.0, 3)
                + " latencySamples=" + juce::String(totalLatency)
                + " blockSize=" + juce::String(blockSize));
            diagLogRuntimeBuilder("[DIAG] warmup failed phase=" + juce::String(warmupPhaseToString(phase)));
            diagLogRuntimeBuilder(formatWarmupPhaseTimingSummary(phaseElapsedUs));
            return BuildError::WarmupFailed;
        }
    }

    switchPhase(WarmupPhase::Ready);
    accumulatePhaseUntil(std::chrono::steady_clock::now());

    const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    const std::uint64_t elapsedUsSafe = static_cast<std::uint64_t>(std::max<std::int64_t>(elapsedUs, 0));

    // static mutable 禁止 (rule 4.3.1): 累積カウンタ削除。現在のビルドの elapsed のみ記録。
    diagLogRuntimeBuilder("[DIAG] warmup result: success blocks="
        + juce::String(warmupBlocks)
        + " elapsedMs=" + juce::String(static_cast<double>(elapsedUs) / 1000.0, 3)
        + " latencySamples=" + juce::String(totalLatency)
        + " blockSize=" + juce::String(blockSize));
    diagLogRuntimeBuilder(formatWarmupPhaseTimingSummary(phaseElapsedUs));
    diagLogRuntimeBuilder(formatWarmupElapsedSummary(warmupBlocks, elapsedUsSafe));

    return BuildError::None;
}

} // namespace convo
