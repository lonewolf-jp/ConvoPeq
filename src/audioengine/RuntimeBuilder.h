#pragma once

#include "AudioEngine.h"
#include "RuntimeBuildTypes.h"
#include "RuntimeCommand.h"

namespace convo {

enum class BuildError {
    None,
    InvalidInput,
    ResourceUnavailable,
    WarmupFailed,
    InternalError
};

struct BuildResult {
    AudioEngine::DSPCore* runtime = nullptr;
    BuildError error = BuildError::None;
    bool prepared = false;
};

const char* toString(BuildError error) noexcept;
bool tryBuildInputFromCommand(const EngineCommand& cmd, BuildInput& out) noexcept;

class RuntimeBuilder {
public:
    explicit RuntimeBuilder(AudioEngine& owner) noexcept : engine(owner) {}

    BuildResult build(const BuildInput& in) noexcept;
    BuildResult build(const EngineCommand& cmd) noexcept;
    BuildError validateWarmup(const AudioEngine::DSPCore& runtime) const noexcept;

private:
    AudioEngine& engine;
};

} // namespace convo
