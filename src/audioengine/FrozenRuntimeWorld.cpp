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

// ★ 2026-11-08: release は所有権の移譲のみを行う（seal は publish transaction 側の責務）。
RuntimeState* FrozenRuntimeWorld::releaseState() noexcept
{
    // ★ 2026-11-08: seal を前提にしない。実経路では world は release 時点では未 seal で、
    //   publish transaction の中で sealRecursively() → 検証（AudioEngine.Commit.cpp の
    //   isFrozen / isSealedRecursively）→ commit の順に進む。ここで seal を要求すると
    //   正当な publish を拒否してしまう（D12-1-C で実測）。
    //   不変性の強制点は commit 側であり、本メソッドは所有権の移譲だけを担う。
    if (!state_)
        return nullptr;
    return state_.release();
}

} // namespace convo
