#pragma once

#include "AudioEngine.h"
#include "RuntimeBuildTypes.h"

namespace convo {

enum class BuildError {
    None,
    InvalidInput,
    ResourceUnavailable,
    MKLFailure,          // ★ C-2: MKL 初期化・FFT 計画失敗
    ConvolverFailure,    // ★ C-2: Convolver Build 失敗
    PrepareFailure,      // ★ C-2: DSPCore::prepare() 失敗
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

    // ★ S-2: HealthState 参照設定
    void setHealthStateRef(const std::atomic<ISRHealthState>* ref) noexcept {
        m_healthStateRef = ref;
    }

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
    // ★ S-2: HealthState 参照
    const std::atomic<ISRHealthState>* m_healthStateRef = nullptr;
};

} // namespace convo
