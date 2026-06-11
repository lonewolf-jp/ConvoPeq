#pragma once

#include "AudioEngine.h"
#include "CrossfadeAuthority.h"
#include "DSPLifetimeManager.h"
#include "RuntimeBuilder.h"

namespace convo::isr {

/*
 * Crossfade Registration Authority
 *
 * Crossfade registration is owned exclusively by DSPTransition.
 *
 * Responsibility boundary:
 *   Decision  → CrossfadeAuthority  (WHETHER a crossfade is required)
 *   Execution → DSPTransition       (Transition lifecycle execution)
 *   Registration → DSPTransition    (Registration at transition execution start)
 *
 * Rationale:
 *   - CrossfadeAuthority evaluates dspProjection values and decides
 *     whether a crossfade is needed. It must NOT own registration.
 *   - DSPTransition executes the actual transition lifecycle (activate,
 *     crossfade start, retire). Registration must occur at the point
 *     where transition execution begins.
 *   - Merging Decision and Registration into a single class would
 *     create an Authority violation (Decision → Execution coupling).
 *
 * Do NOT register crossfades from:
 *   - CrossfadeAuthority          (would merge Decision + Execution)
 *   - RuntimeBuilder              (build-time, no lifecycle context)
 *   - RuntimePublicationOrchestrator (coordination only, no execution)
 *   - AudioEngine commit path     (publish admission, no transition)
 *
 * Any new registration site requires architecture review.
 * CI gate: grep "registerCrossfade(" → DSPTransition only → pass
 */

// DSPTransition: publish 成功後に DSP Lifetime 操作を実行する。
// Coordinator::submitPublishRequest() から呼ばれる。
// ★ activate は publish 成功後にのみ実行する。
//   (publish 失敗時は activeDSP を書き換えず、不整合を防止)
class DSPTransition {
public:
    explicit DSPTransition(AudioEngine& engine) noexcept : engine_(engine) {}

    // onPublishCompleted: publish 成功後に DSP activate/crossfade/retire を実行
    // ★ publish 成功 = この関数が呼ばれていること
    // ★ activate は publish 成功後にのみ実行
    void onPublishCompleted(AudioEngine::DSPCore* newDSP,
                            AudioEngine::DSPCore* oldDSP,
                            const CrossfadeAuthority::Decision& decision,
                            DSPLifetimeManager& lifetime) noexcept
    {
        // 1. activate (publish 成功後にのみ実行)
        lifetime.activate(newDSP);

        // 2. Crossfade または Retire
        if (decision.needsCrossfade && oldDSP != nullptr) {
            auto oldHandle = engine_.dspHandleRuntime_.getActiveRuntimeDSPHandle();
            auto newHandle = engine_.registerDSPHandleForRuntime(newDSP);

            if (!oldHandle.isNull() && !newHandle.isNull()) {
                // CrossfadeAuthority に registration を委譲
                const auto xfadeId = engine_.crossfadeAuthorityRuntime_.registerCrossfade(oldHandle, newHandle);
                (void)xfadeId;  // CrossfadeId は将来 AudioThread 完了検出時の通知に使用
                engine_.publishAtomic(engine_.activeCrossfadeId_,
                                     static_cast<convo::isr::CrossfadeId>(0u),
                                     std::memory_order_release);
            }

            // exchangeFadingRuntimeDSP + retire
            auto* prevRaw = engine_.exchangeFadingRuntimeDSP(oldDSP);
            if (auto* prev = (reinterpret_cast<uintptr_t>(prevRaw) == (~static_cast<uintptr_t>(0)))
                             ? nullptr : prevRaw)
            {
                if (prev != oldDSP) {
                    // 異なる DSP が fading に入っていた場合も retire
                    lifetime.retire(prev);
                }
            }

            // crossfade atomic 設定 (CrossfadeRuntime 委譲)
            const double rampSampleRate = std::max(1.0,
                (newDSP != nullptr) ? newDSP->sampleRate
                    : convo::consumeAtomic(engine_.currentSampleRate, std::memory_order_acquire));
            engine_.crossfadeRuntime_.start(decision.fadeTimeSec, rampSampleRate);
            engine_.setIRChangeFlag();
        } else if (oldDSP != nullptr) {
            // Crossfade 不要: 即時 retire
            engine_.crossfadeRuntime_.complete();
            lifetime.retire(oldDSP);
        }
    }

    // notifyTransitionComplete: クロスフェード完了時の処理
    // Timer から呼ばれる (代替: Coordinator::notifyTransitionComplete)
    void onTransitionComplete(AudioEngine::DSPCore* currentAfterFade) noexcept
    {
        if (currentAfterFade == nullptr)
            return;

        auto* doneRaw = engine_.exchangeFadingRuntimeDSP(nullptr);
        if (auto* done = (reinterpret_cast<uintptr_t>(doneRaw) == (~static_cast<uintptr_t>(0)))
                         ? nullptr : doneRaw)
        {
            // retire old fading DSP
            DSPLifetimeManager lifetime(engine_);
            lifetime.retire(done);
        }

        engine_.crossfadeRuntime_.setDryHoldSamples(0);
        engine_.refreshCrossfadePreparedSnapshotFromAtomics();

        // publish idling world (Coordinator 経由)
        auto coordinator = engine_.makeRuntimePublicationCoordinator();
        auto worldBuilder = convo::RuntimeBuilder(engine_);
        auto worldOwner = worldBuilder.buildRuntimePublishWorld(currentAfterFade,
                                                                 nullptr,
                                                                 convo::TransitionPolicy::HardReset,
                                                                 0.0,
                                                                 false);
        if (worldOwner) {
            coordinator.publishWorld(std::move(worldOwner));
        }
    }

private:
    AudioEngine& engine_;
};

} // namespace convo::isr
