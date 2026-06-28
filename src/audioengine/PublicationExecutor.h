#pragma once

#include "AudioEngine.h"
#include "AlignedAllocation.h"
#include "FrozenRuntimeWorld.h"

namespace convo::isr {

// PublishResult: PublicationExecutor::publish() の結果
enum class PublishResult {
    Success,
    ValidationFailed,
    PublishFailed,
    BridgeFailed
};

// PublicationExecutor: validate → publishAndSwap → retire old を実行する。
// Coordinator から呼ばれる。
// ★ activate は行わない (DSPTransition が担当)
// ★ publish 失敗時は activate/crossfade/retire を一切行わない
class PublicationExecutor {
public:
    PublicationExecutor() noexcept = default;

    // publish: world を publishAndSwap する（AudioEngine の store/bridge を使用）。
    // ★ Phase4: FrozenRuntimeWorld を受け取り、内部の RuntimeState* を抽出して
    //   Coordinator の publishWorld に渡す（Builder→Runtime 二段階モデル）
    [[nodiscard]] PublishResult publish(
        AudioEngine& engine,
        convo::aligned_unique_ptr<convo::FrozenRuntimeWorld> frozen) noexcept;

    void advanceEpoch() noexcept {}
};

} // namespace convo::isr
