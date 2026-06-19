#include "PublicationExecutor.h"
#include "AudioEngine.h"
#include "RuntimeBuilder.h"

namespace convo::isr {

PublishResult PublicationExecutor::publish(
    AudioEngine& engine,
    convo::aligned_unique_ptr<RuntimePublishWorld> worldOwner) noexcept
{
    if (!worldOwner)
        return PublishResult::PublishFailed;

    // Phase 1+2: Delegate to coordinator's publishWorld (validate + publishAndSwap)
    // ★ P0-2: publishWorld が PublishStageResult を返すようになったため、
    //   その結果を PublishResult にマッピングする。
    auto coordinator = engine.makeRuntimePublicationCoordinator();
    const auto outcome = coordinator.publishWorld(std::move(worldOwner));

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
