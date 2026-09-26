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
    // ★ work70 P1-a: Orchestrator が事前登録した DSPHandle を existingHandle として受け取り、
    //   commitRuntimePublication（register→rollback トランザクション）を実行する。
    // ★ B4: oldHandle = Rebuild (#7) の old DSP retire 意図（current active DSP handle）。
    //   trySubmit が解決した oldHandle を渡し、idle publish とは異なり retire する意図を表現する。
    // ★ P3-5-R27: origin plumbing — trySubmitImpl の req.recoveryObligationId を
    //   intent payload まで搬送する（0 = Main／!=0 = Recovery・既存意味のまま）。
    //   新 struct field なし。default 0 のため既存 caller（idle／bootstrap 等）は不変。
    [[nodiscard]] PublishResult publish(
        AudioEngine& engine,
        convo::aligned_unique_ptr<convo::FrozenRuntimeWorld> frozen,
        convo::isr::DSPHandle existingHandle,
        convo::isr::DSPHandle oldHandle,
        std::uint64_t recoveryObligationId = 0) noexcept;

    void advanceEpoch() noexcept {}

private:
    // publish / publishFireAndForget の共通実装。
    // ★ P3-5-R27: recoveryObligationId を commit／fire-and-forget 両経路へ中継する。
    [[nodiscard]] PublishResult publishImpl(
        AudioEngine& engine,
        convo::aligned_unique_ptr<convo::FrozenRuntimeWorld> frozen,
        convo::isr::DSPHandle existingHandle,
        convo::isr::DSPHandle oldHandle,
        bool waitForReceipt,
        std::uint64_t recoveryObligationId = 0) noexcept;
};

} // namespace convo::isr
