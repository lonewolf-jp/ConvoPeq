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

    // Bootstrap World: 初期化時に初回 publish する最小限の RuntimePublishWorld を生成する。
    // 全てのデフォルト値で初期化され、AudioEngine::initialize() 実行直後に
    // RuntimePublicationCoordinator::publishWorld() で公開される。
    // これにより publishRuntimeStateNonRt が初回コール時に world==nullptr の
    // fallback を必要としなくなる。
    [[nodiscard]] convo::aligned_unique_ptr<RuntimePublishWorld>
    createBootstrapWorld() noexcept;

    BuildResult build(const BuildInput& in,
                      const ConvolverProcessor::BuildSnapshot& convolverBuildSnapshot) noexcept;
    // Warmup validation (still used by RebuildDispatch)
    BuildError validateWarmup(const AudioEngine::DSPCore& runtime) const noexcept;

private:
    AudioEngine& engine;
};

} // namespace convo
