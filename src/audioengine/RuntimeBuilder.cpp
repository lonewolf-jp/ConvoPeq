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

static void diagLogRuntimeBuilder(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

static int getWarmupBlocksOverrideFromEnv() noexcept
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
static juce::String formatWarmupElapsedSummary(int blocks, std::uint64_t elapsedUs)
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

static const char* warmupPhaseToString(WarmupPhase phase) noexcept
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

static juce::String formatWarmupPhaseTimingSummary(const std::array<std::uint64_t, 5>& phaseElapsedUs)
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

bool tryBuildInputFromCommand(const EngineCommand& cmd, BuildInput& out) noexcept
{
    switch (cmd.type)
    {
        case CommandType::UpdateParameters:
        case CommandType::ChangeSampleRate:
        case CommandType::ChangeOversampling:
            break;

        default:
            return false;
    }

    if (cmd.sampleRate <= 0.0 || cmd.blockSize <= 0)
        return false;

    out.sampleRate = cmd.sampleRate;
    out.blockSize = cmd.blockSize;
    out.ditherBitDepth = cmd.intValue;
    out.oversamplingFactor = cmd.oversamplingFactor;
    out.oversamplingType = cmd.oversamplingType;
    out.noiseShaperType = cmd.noiseShaperType;
    return true;
}

BuildResult RuntimeBuilder::build(const BuildInput& in) noexcept
{
    return build(in, engine.getConvolverProcessor().captureBuildSnapshot());
}

BuildResult RuntimeBuilder::build(const BuildInput& in, const ConvolverProcessor::BuildSnapshot& snapshot) noexcept
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
        runtime->convolverRt().applyBuildSnapshot(snapshot);
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

BuildResult RuntimeBuilder::build(const EngineCommand& cmd) noexcept
{
    return build(cmd, engine.getConvolverProcessor().captureBuildSnapshot());
}

BuildResult RuntimeBuilder::build(const EngineCommand& cmd, const ConvolverProcessor::BuildSnapshot& snapshot) noexcept
{
    BuildInput input {};
    if (!tryBuildInputFromCommand(cmd, input))
    {
        BuildResult result {};
        result.error = BuildError::InvalidInput;
        return result;
    }

    return build(input, snapshot);
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
