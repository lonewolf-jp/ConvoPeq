#include "RuntimePublicationOrchestrator.h"
#include "AudioEngine.h"
#include "RuntimeBuilder.h"
#include "CrossfadeAuthority.h"
#include <chrono>

namespace convo::isr {

RuntimePublicationOrchestrator::RuntimePublicationOrchestrator(AudioEngine& engine) noexcept
    : engine_(engine)
    , admission_()
    , executor_()
    , transition_(engine)
    , lifetime_(engine)
    , publicationReader(engine.getRetireRouter())
{
}

PublicationAdmission::Decision RuntimePublicationOrchestrator::trySubmit(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    // ---- Phase 1: Admission ----
    // ★ evaluate() は必須。バイパス禁止。
    const convo::RuntimeReaderContext pubCtx{ publicationReader, convo::ObserveChannel::Publication };
    auto decision = admission_.evaluate(req, engine_, pubCtx);
    if (decision != PublicationAdmission::Decision::Accepted)
    {
        // Deferred/Rejected: caller が処理するため、ここでは retire しない
        return decision;
    }

    // ---- Phase 2: Build + Publish (activate 前) ----
    // ★ activate はまだ行わない。まず world を build して publish する。
    // ★ Phase2: DSPHandle → DSPCore* 解決 (Execution Path Handle Normalization)
    auto* newDSPResolved = engine_.resolveDSPHandle(req.newDSP);
    // ★ [PR-4A] S-A2排除: Decision層の DSPCore* 直接参照を Handle 経由に変更
    auto oldHandle = engine_.dspHandleRuntime_.getActiveRuntimeDSPHandle();
    auto* oldDSP = (!oldHandle.isNull())
        ? engine_.resolveDSPHandle(oldHandle)
        : nullptr;
#if defined(JUCE_DEBUG) || defined(CONVO_CI_BUILD)
    if (req.newDSP.isNull()) {
        DBG("[DIAG] trySubmit: newDSP handle is NULL generation=" << req.generation);
    } else if (newDSPResolved == nullptr) {
        DBG("[DIAG] trySubmit: resolveDSPHandle failed slot=" << (int)req.newDSP.slot
            << " gen=" << (int)req.newDSP.generation << " reqGen=" << req.generation);
    }
#endif

    // Step 2a: Build world with default (HardReset) policy first
    // (evaluate 前に world が必要だが、RuntimePublishWorld は非デフォルト構築可能のため)
    auto worldBuilder = convo::RuntimeBuilder(engine_);
    auto worldOwner = worldBuilder.buildRuntimePublishWorld(
        newDSPResolved, oldDSP,
        convo::TransitionPolicy::HardReset, 0.0, false,
        &req.sealedSnapshot);

    if (!worldOwner) {
        // Build failed: retire new DSP, keep old world
        if (!req.newDSP.isNull())
            lifetime_.retire(newDSPResolved);
        return PublicationAdmission::Decision::RejectedNotFinalized;
    }

    // Step 2b: Crossfade decision using RuntimeWorld projection values
    // (DSPCore 直読は行わない)
    const auto* oldWorld = engine_.runtimeStore.observe();
    CrossfadeAuthority crossfade;
    auto cfDecision = crossfade.evaluate(engine_, *oldWorld, *worldOwner);

    // Step 2c: Update world with crossfade decision if needed
    // ★ oldDSP が nullptr の場合はクロスフェード不能 — 判定を無効化する
    if (cfDecision.needsCrossfade && oldDSP != nullptr)
    {
        worldOwner->assertMutable();
        worldOwner->execution.transitionPolicy = static_cast<int>(convo::TransitionPolicy::SmoothOnly);
        worldOwner->execution.transitionActive = true;
        worldOwner->topology.hasFadingRuntime = true;
        worldOwner->overlap.fadeTimeSec = cfDecision.fadeTimeSec;
    }

    auto result = executor_.publish(engine_, std::move(worldOwner));
    if (result != PublishResult::Success) {
        // publish 失敗: activate/crossfade/retire は一切行わない
        if (!req.newDSP.isNull())
            lifetime_.retire(newDSPResolved);
        return PublicationAdmission::Decision::RejectedShutdown;
    }

    // ---- Phase 3: Publish 成功確認後に DSP Lifetime 操作 ----
    // ★ activate は publish 成功後にのみ実行する。
    //    (publish 失敗時は activeDSP を書き換えず、不整合を防止)
    transition_.onPublishCompleted(newDSPResolved, oldDSP, cfDecision, lifetime_);

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
            enqueueDeferred(req);
            return;
        case PublicationAdmission::Decision::RejectedStaleGeneration:
        case PublicationAdmission::Decision::RejectedNotFinalized:
        case PublicationAdmission::Decision::RejectedPressure:
        case PublicationAdmission::Decision::RejectedShutdown:
            // trySubmit が既に retireDSP 済み
            return;
    }
}

// ★ C-2.2: enqueueDeferred — global sequence スナップショットを記録
void RuntimePublicationOrchestrator::enqueueDeferred(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    // 上書きカウント
    if (hasDeferred_)
        convo::fetchAddAtomic(deferredOverwriteCount_, uint64_t{1},
            std::memory_order_release);

    const auto now = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());

    // 上書き時は滞留時間を maxDeferredAgeMs に反映
    if (deferredSlot_.has_value()) {
        const uint64_t ageMs = (now - deferredSlot_->enqueueTimestampUs) / 1000;
        uint64_t currentMax = convo::consumeAtomic(maxDeferredAgeMs_,
            std::memory_order_acquire);
        while (ageMs > currentMax) {
            if (convo::compareExchangeAtomic(maxDeferredAgeMs_, currentMax,
                    ageMs, std::memory_order_acq_rel,
                    std::memory_order_acquire))
                break;
        }
    }

    deferredSlot_ = DeferredPublishSlot{
        .request = req,
        .guard = DeferredGuard{
            .generation = static_cast<uint64_t>(req.generation),
            .sequence = engine_.getLastCommittedPublicationSequence()
        },
        .lastDiscardReason = DiscardReason::None,
        .enqueueTimestampUs = now
    };
    hasDeferred_ = true;
}

// ★ C-2.3: notifyTransitionComplete — stale discard 実装
void RuntimePublicationOrchestrator::notifyTransitionComplete(
    AudioEngine::DSPCore* currentAfterFade) noexcept
{
    if (currentAfterFade == nullptr)
        return;

    transition_.onTransitionComplete(currentAfterFade);

    // ★ A-2.2: shutdown 中は deferred 再投入をキャンセル（残留タスク防止）
    if (engine_.isShutdownInProgress()) {
        if (hasDeferred_) {
            if (deferredSlot_.has_value())
                deferredSlot_->lastDiscardReason = DiscardReason::ShutdownDiscard;
            deferredSlot_.reset();
            hasDeferred_ = false;
        }
        return;
    }

    // ★ C-2.3: stale discard（二重検査: generation + publication sequence）
    if (hasDeferred_ && deferredSlot_.has_value())
    {
        auto& deferred = *deferredSlot_;

        // 1. generation 検査
        const int currentGen = convo::consumeAtomic(
            engine_.rebuildRequestGeneration, std::memory_order_acquire);
        if (deferred.guard.generation != 0ull
            && deferred.guard.generation != static_cast<uint64_t>(currentGen)) {
            deferred.lastDiscardReason = DiscardReason::StaleDiscard;
            deferredSlot_.reset();
            hasDeferred_ = false;
            return;
        }

        // 2. publication sequence 検査
        const auto currentPubSeq = engine_.getLastCommittedPublicationSequence();
        if (deferred.guard.sequence < currentPubSeq) {
            deferred.lastDiscardReason = DiscardReason::StaleDiscard;
            deferredSlot_.reset();
            hasDeferred_ = false;
            return;
        }

        // 有効な deferred → submit
        auto req = deferred.request;
        deferredSlot_.reset();
        hasDeferred_ = false;
        submitPublishRequest(req);
    }
}

// ★ C-2.2: shutdown 時に deferred publish を強制消去
void RuntimePublicationOrchestrator::clearDeferredForShutdown() noexcept
{
    if (hasDeferred_) {
        if (deferredSlot_.has_value())
            deferredSlot_->lastDiscardReason = DiscardReason::ShutdownDiscard;
        deferredSlot_.reset();
        hasDeferred_ = false;
    }
}

// ★ A-2.5: DrainAudit 用 — deferred publish 最長滞留時間
uint64_t RuntimePublicationOrchestrator::getMaxDeferredAgeMs() const noexcept
{
    return convo::consumeAtomic(maxDeferredAgeMs_, std::memory_order_acquire);
}

// ★ C-2.1: 監査用 — deferred overwrite 回数
std::uint64_t RuntimePublicationOrchestrator::deferredOverwriteCount() const noexcept
{
    return convo::consumeAtomic(deferredOverwriteCount_, std::memory_order_acquire);
}

} // namespace convo::isr
