#include "FrozenRuntimeWorld.h"
#include "AudioEngine.h"  // RuntimeState 完全定義

namespace convo {

FrozenRuntimeWorld::FrozenRuntimeWorld(aligned_unique_ptr<RuntimeState> state) noexcept
    : state_(std::move(state))
{
    assert(state_ && "FrozenRuntimeWorld: state must not be null");
    // ★ freeze は必須ではない（buildRuntimePublishWorld では coordinator.publishWorld
    //   が sealRecursively を呼ぶため）。createBootstrapWorld では自前で freeze する。
}

FrozenRuntimeWorld::~FrozenRuntimeWorld()
{
    if (state_)
    {
        // ★ unseal は所有権を持つ場合のみ実行
        //   releaseState() で所有権を移譲した後は state_==nullptr なので実行しない
        state_->unseal();
    }
}

void FrozenRuntimeWorld::sealRecursively() noexcept
{
    // RuntimeState は既に freeze() 済みだが、Coordinator の publish パスが
    // sealRecursively() を呼ぶため、無害なデリゲートを提供する。
    state_->sealRecursively();
}

} // namespace convo
