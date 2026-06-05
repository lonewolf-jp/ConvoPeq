#pragma once

#include "AudioEngine.h"
#include "CrossfadeAuthority.h"
#include "DSPLifetimeManager.h"
#include "RuntimeBuilder.h"

namespace convo::isr {

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
                engine_.crossfadeAuthorityRuntime_.registerCrossfade(oldHandle, newHandle);
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

            // crossfade atomic 設定
            const double rampSampleRate = std::max(1.0,
                (newDSP != nullptr) ? newDSP->sampleRate
                    : convo::consumeAtomic(engine_.currentSampleRate, std::memory_order_acquire));
            engine_.dspCrossfadeGain.reset(rampSampleRate, std::max(0.001, decision.fadeTimeSec));
            engine_.dspCrossfadeGain.setCurrentAndTargetValue(0.0);
            convo::publishAtomic(engine_.dspCrossfadeUseDryAsOld, false, std::memory_order_release);
            convo::publishAtomic(engine_.firstIrDryCrossfadePending, false, std::memory_order_release);
            convo::publishAtomic(engine_.queuedFadeTimeSec, decision.fadeTimeSec, std::memory_order_release);
            convo::publishAtomic(engine_.dspCrossfadePending, true, std::memory_order_release);
            engine_.setIRChangeFlag();
        } else if (oldDSP != nullptr) {
            // Crossfade 不要: 即時 retire
            convo::publishAtomic(engine_.dspCrossfadePending, false, std::memory_order_release);
            convo::publishAtomic(engine_.dspCrossfadeUseDryAsOld, false, std::memory_order_release);
            convo::publishAtomic(engine_.firstIrDryCrossfadePending, false, std::memory_order_release);
            convo::publishAtomic(engine_.dspCrossfadeStartDelayBlocks, 0, std::memory_order_release);
            convo::publishAtomic(engine_.dspCrossfadeDryHoldSamples, 0, std::memory_order_release);
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
            DSPLifetimeManager lifetime(const_cast<AudioEngine&>(engine_));
            lifetime.retire(done);
        }

        convo::publishAtomic(engine_.dspCrossfadeDryHoldSamples, 0, std::memory_order_release);
        engine_.refreshCrossfadePreparedSnapshotFromAtomics();

        // publish idling world (Coordinator 経由)
        auto coordinator = const_cast<AudioEngine&>(engine_).makeRuntimePublicationCoordinator();
        auto worldBuilder = convo::RuntimeBuilder(const_cast<AudioEngine&>(engine_));
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
