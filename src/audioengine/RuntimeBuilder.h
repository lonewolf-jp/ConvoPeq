#pragma once

#include "AudioEngine.h"
#include "RuntimeBuildTypes.h"

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

class RuntimeBuilder {
public:
    explicit RuntimeBuilder(AudioEngine& owner) noexcept : engine(owner) {}

    [[nodiscard]] convo::aligned_unique_ptr<RuntimePublishWorld>
    buildRuntimePublishWorld(AudioEngine::DSPCore* current,
                             AudioEngine::DSPCore* next,
                             convo::TransitionPolicy policy,
                             double fadeTimeSec,
                             bool active,
                             const convo::RuntimeBuildSnapshot* sealedSnapshot = nullptr) noexcept;

    BuildResult build(const BuildInput& in,
                      const ConvolverProcessor::BuildSnapshot& convolverBuildSnapshot) noexcept;
    BuildError validateWarmup(const AudioEngine::DSPCore& runtime) const noexcept;

    // Warmup: FIR 履歴と AGC state 初期化
    int getRequiredWarmupBlocks(const AudioEngine::DSPCore& runtime) const noexcept;
    BuildError executeWarmup(AudioEngine::DSPCore& runtime) noexcept;

private:
    AudioEngine& engine;
};

} // namespace convo
