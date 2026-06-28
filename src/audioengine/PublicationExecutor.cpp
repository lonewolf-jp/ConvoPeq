#include "PublicationExecutor.h"
#include "AudioEngine.h"
#include "RuntimeBuilder.h"

namespace convo::isr {

PublishResult PublicationExecutor::publish(
    AudioEngine& engine,
    convo::aligned_unique_ptr<convo::FrozenRuntimeWorld> frozen) noexcept
{
    if (!frozen)
        return PublishResult::PublishFailed;

    // ★ Phase4: FrozenRuntimeWorld から RuntimeState* を抽出して Coordinator に渡す
    //   releaseState() で所有権を放棄 → Coordinator の retire 経路が unseal + aligned_free を担当
    auto* rawState = frozen->releaseState();
    if (rawState == nullptr)
        return PublishResult::PublishFailed;

    // aligned_unique_ptr<RuntimeState> でラップ（AlignedObjectDeleter が aligned_free を実行）
    auto stateOwner = convo::aligned_unique_ptr<RuntimeState>(rawState);

    auto coordinator = engine.makeRuntimePublicationCoordinator();
    const auto outcome = coordinator.publishWorld(std::move(stateOwner));

    switch (outcome) {
        case PublishStageResult::Success:
            return PublishResult::Success;
        case PublishStageResult::Rejected:
            return PublishResult::ValidationFailed;
        case PublishStageResult::Failed:
            return PublishResult::PublishFailed;
    }

    // fallback (全ての enum 値に対応済みだが、コンパイラ警告対策)
    return PublishResult::PublishFailed;
}

} // namespace convo::isr
