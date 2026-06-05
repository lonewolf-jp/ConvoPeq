#include "RuntimePublicationOrchestrator.h"
#include "AudioEngine.h"
#include "RuntimeBuilder.h"
#include "CrossfadeAuthority.h"

namespace convo::isr {

RuntimePublicationOrchestrator::RuntimePublicationOrchestrator(AudioEngine& engine) noexcept
    : engine_(engine)
    , admission_()
    , executor_()
    , transition_(engine)
    , lifetime_(engine)
{
}

PublicationAdmission::Decision RuntimePublicationOrchestrator::trySubmit(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    // ---- Phase 1: Admission ----
    // ★ evaluate() は必須。バイパス禁止。
    auto decision = admission_.evaluate(req, engine_);
    if (decision != PublicationAdmission::Decision::Accepted)
    {
        // Deferred/Rejected: caller が処理するため、ここでは retire しない
        return decision;
    }

    // ---- Phase 2: Build + Publish (activate 前) ----
    // ★ activate はまだ行わない。まず world を build して publish する。
    auto* oldDSP = engine_.getActiveRuntimeDSP();

    // Crossfade decision using CrossfadeAuthority
    CrossfadeAuthority crossfade;
    auto cfDecision = crossfade.evaluateOnly(engine_, oldDSP,
        static_cast<AudioEngine::DSPCore*>(req.newDSP));

    // Build world using RuntimeBuilder
    auto worldBuilder = convo::RuntimeBuilder(engine_);
    auto worldOwner = worldBuilder.buildRuntimePublishWorld(
        static_cast<AudioEngine::DSPCore*>(req.newDSP), oldDSP,
        cfDecision.needsCrossfade ? convo::TransitionPolicy::SmoothOnly : convo::TransitionPolicy::HardReset,
        cfDecision.fadeTimeSec, cfDecision.needsCrossfade,
        &req.sealedSnapshot);

    if (!worldOwner) {
        // Build failed: retire new DSP, keep old world
        if (req.newDSP != nullptr)
            lifetime_.retire(static_cast<AudioEngine::DSPCore*>(req.newDSP));
        return PublicationAdmission::Decision::RejectedNotFinalized;
    }

    // ★ PublicationExecutor::publish() は PublishResult を返す。
    // 失敗時は activate/crossfade/retire を行わない。
    auto result = executor_.publish(engine_, std::move(worldOwner));
    if (result != PublishResult::Success) {
        // publish 失敗: activate/crossfade/retire は一切行わない
        if (req.newDSP != nullptr)
            lifetime_.retire(static_cast<AudioEngine::DSPCore*>(req.newDSP));
        return PublicationAdmission::Decision::RejectedShutdown;
    }

    // ---- Phase 3: Publish 成功確認後に DSP Lifetime 操作 ----
    // ★ activate は publish 成功後にのみ実行する。
    //    (publish 失敗時は activeDSP を書き換えず、不整合を防止)
    transition_.onPublishCompleted(static_cast<AudioEngine::DSPCore*>(req.newDSP), oldDSP, cfDecision, lifetime_);

    // ---- Phase 4: Epoch advance ----
    // advanceRetireEpoch は publish 後に epoch を進める。
    // AudioEngine::advanceRetireEpoch() は retire queue の drain を行う。
    engine_.advanceRetireEpoch();

    return PublicationAdmission::Decision::Accepted;
}

void RuntimePublicationOrchestrator::submitPublishRequest(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    auto decision = trySubmit(req);
    switch (decision) {
        case PublicationAdmission::Decision::Accepted:
            return;
        case PublicationAdmission::Decision::DeferredFadingActive:
            admission_.enqueueDeferred(req);
            return;
        case PublicationAdmission::Decision::RejectedStaleGeneration:
        case PublicationAdmission::Decision::RejectedNotFinalized:
        case PublicationAdmission::Decision::RejectedPressure:
        case PublicationAdmission::Decision::RejectedShutdown:
            // trySubmit が既に retireDSP 済み
            return;
    }
}

void RuntimePublicationOrchestrator::notifyTransitionComplete(
    AudioEngine::DSPCore* currentAfterFade) noexcept
{
    if (currentAfterFade == nullptr)
        return;

    transition_.onTransitionComplete(currentAfterFade);
}

} // namespace convo::isr
