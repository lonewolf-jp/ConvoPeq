#include "RuntimeBuilder.h"

#include <algorithm>
#include <array>
#include <atomic>
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
                                         bool active) noexcept
{
    const auto nextGraphGeneration = engine.reserveNextRuntimeGraphGeneration();
    const auto nextWorldId = engine.runtimeWorldIdGenerator_.next();
    const auto nextPublicationSequence = convo::fetchAddAtomic(engine.publicationSequenceCounter_,
                                    static_cast<convo::isr::PublicationSequenceId>(1),
                                    std::memory_order_acq_rel) + 1;

    auto engineState = engine.makeEngineRuntimeState(current, next, policy, fadeTimeSec, active);
    engineState.revision = nextGraphGeneration;
    auto graphState = engine.makeRuntimeGraphState(engineState);
    graphState.generation = nextGraphGeneration;

    const auto* previousWorld = AudioEngine::RuntimePublicationCoordinator::observeWorldHandle(engine.runtimeStore);

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

    worldOwner->routing.processingOrder = static_cast<int>(engine.getProcessingOrder());
    worldOwner->routing.eqBypassed = graphState.eqBypassed;
    worldOwner->routing.convBypassed = graphState.convBypassed;

    worldOwner->execution.transitionActive = engineState.transition.active;
    worldOwner->execution.transitionPolicy = static_cast<int>(engineState.transition.policy);
    worldOwner->execution.latencyCompensationSamples = engineState.transition.latencyDeltaSamples;
    worldOwner->execution.crossfadeStartDelayBlocks = engineState.dspCrossfadeStartDelayBlocks;
    worldOwner->execution.crossfadeDryHoldSamples = engineState.dspCrossfadeDryHoldSamples;

    worldOwner->publication.sequenceId = nextPublicationSequence;
    worldOwner->publication.epoch = static_cast<convo::isr::PublicationEpoch>(nextGraphGeneration);
    worldOwner->publication.mappedRuntimeGeneration = nextGraphGeneration;
    worldOwner->publication.previousSequenceId = previousWorld != nullptr
        ? previousWorld->publication.sequenceId
        : static_cast<convo::isr::PublicationSequenceId>(0);

    worldOwner->overlap.useDryAsOld = engineState.dspCrossfadeUseDryAsOld;
    worldOwner->overlap.firstIrDryCrossfadePending = engineState.firstIrDryCrossfadePending;
    worldOwner->overlap.dryScaleTarget = engineState.dryScaleTarget;
    worldOwner->overlap.fadeTimeSec = engineState.queuedFadeTimeSec;

    worldOwner->retire.retireEpoch = nextGraphGeneration;
    worldOwner->retire.retireBacklog = engine.consumeAtomic(engine.retireQueueDepth_, std::memory_order_acquire);
    worldOwner->retire.deferredResidency = engine.consumeAtomic(engine.fallbackQueueDepth_, std::memory_order_acquire);

    worldOwner->timing.sampleRateHz = graphState.sampleRate;
    worldOwner->timing.queuedFadeTimeSec = engineState.queuedFadeTimeSec;
    worldOwner->timing.activationEpoch = nextGraphGeneration;

    worldOwner->latency.latencyDelayOld = engineState.latencyDelayOld;
    worldOwner->latency.latencyDelayNew = engineState.latencyDelayNew;
    worldOwner->latency.latencyDeltaSamples = engineState.transition.latencyDeltaSamples;

    worldOwner->scheduling.transitionActive = engineState.transition.active;
    worldOwner->scheduling.crossfadeStartDelayBlocks = engineState.dspCrossfadeStartDelayBlocks;
    worldOwner->scheduling.crossfadeDryHoldSamples = engineState.dspCrossfadeDryHoldSamples;

    worldOwner->resource.oversamplingFactor = graphState.oversamplingFactor;
    worldOwner->resource.ditherBitDepth = graphState.ditherBitDepth;
    worldOwner->resource.noiseShaperType = graphState.noiseShaperType;

    worldOwner->affinity.rebuildWorkerRunning = engine.consumeAtomic(engine.rebuildThreadIsRunning, std::memory_order_acquire);

    worldOwner->automation.eqBypassed = graphState.eqBypassed;
    worldOwner->automation.convBypassed = graphState.convBypassed;
    worldOwner->automation.softClipEnabled = graphState.softClipEnabled;

    worldOwner->coefficient.adaptiveCoeffBankIndex = graphState.adaptiveCoeffBankIndex;
    worldOwner->coefficient.adaptiveCoeffGeneration = graphState.adaptiveCoeffGeneration;
    worldOwner->coefficient.eqCoeffHash = 0;
    if (const auto* eqState = engine.uiEqEditor.getEQStateSnapshot())
        worldOwner->coefficient.eqCoeffHash = EQProcessor::computeParamsHash(eqState->toEQParameters());

    worldOwner->projectionFreshness.projectionGeneration = nextGraphGeneration;
    worldOwner->projectionFreshness.projectionRevision = graphState.generation;
    worldOwner->projectionFreshness.maxStalenessWindows = 1u;

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
        ^ worldOwner->coefficient.eqCoeffHash;
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

    worldOwner->freeze();
    return worldOwner;
}

BuildResult RuntimeBuilder::build(const BuildInput& in) noexcept
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
        engine.applyCurrentConvolverSnapshotToRuntime(*runtime);
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
