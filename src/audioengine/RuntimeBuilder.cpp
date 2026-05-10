#include "RuntimeBuilder.h"

namespace convo {

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
    BuildResult result {};

    if (in.sampleRate <= 0.0 || in.blockSize <= 0)
    {
        result.error = BuildError::InvalidInput;
        return result;
    }

    auto* runtime = new (std::nothrow) AudioEngine::DSPCore();
    if (runtime == nullptr)
    {
        result.error = BuildError::ResourceUnavailable;
        return result;
    }

    runtime->convolver.setVisualizationEnabled(false);
    runtime->convolver.syncStateFrom(engine.getConvolverProcessor());

    try
    {
        runtime->prepare(in.sampleRate,
                         in.blockSize,
                         in.ditherBitDepth,
                         in.oversamplingFactor,
                         static_cast<AudioEngine::OversamplingType>(in.oversamplingType),
                         static_cast<AudioEngine::NoiseShaperType>(in.noiseShaperType),
                         &engine);
        result.runtime = runtime;
        result.prepared = true;
        return result;
    }
    catch (...)
    {
        delete runtime;
        result.error = BuildError::InternalError;
        return result;
    }
}

BuildResult RuntimeBuilder::build(const EngineCommand& cmd) noexcept
{
    BuildInput input {};
    if (!tryBuildInputFromCommand(cmd, input))
    {
        BuildResult result {};
        result.error = BuildError::InvalidInput;
        return result;
    }

    return build(input);
}

BuildError RuntimeBuilder::validateWarmup(const AudioEngine::DSPCore& runtime) const noexcept
{
    juce::ignoreUnused(engine);

    if (runtime.convolver.isIRLoaded() && !runtime.convolver.isIRFinalized())
        return BuildError::WarmupFailed;

    return BuildError::None;
}

} // namespace convo
