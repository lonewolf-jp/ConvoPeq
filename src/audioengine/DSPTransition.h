#pragma once

#include "AudioEngine.h"
#include "CrossfadeAuthority.h"
#include "DSPLifetimeManager.h"

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
                            convo::isr::DSPHandle oldHandle,
                            const CrossfadeAuthority::Decision& decision,
                            DSPLifetimeManager& lifetime) noexcept
    {
        // ★ P2.5-1: Emergency Override — TOCTOU 対策（Admission 通過後 Critical 検知の最終安全網）
        {
            auto ref = engine_.getHealthStateRef();
            if (ref) {
                auto health = convo::consumeAtomic(*ref, std::memory_order_acquire);
                if (health == convo::ISRHealthState::Critical) {
                    lifetime.activate(newDSP);
                    if (oldDSP != nullptr) {
                        // ★ Temporary: exchangeFadingRuntimeDSP (A-6 fix).
                        //   Will be removed after B-1 CAS-only claimFadingRuntimeDSP().
                        auto* prevRaw = engine_.exchangeFadingRuntimeDSP(oldDSP);
                        engine_.crossfadeRuntime_.complete();
                        lifetime.retire(oldDSP);
                        if (prevRaw != oldDSP) {
                            if (prevRaw != nullptr)
                                lifetime.retire(prevRaw);
                        }
                        // ★ enqueueHealthEvent で非同期投入（層の逆流＋同期実行防止）
                        const uint64_t abortCount = engine_.crossfadeRuntime_.incrementEmergencyAbortCount();
                        engine_.enqueueHealthEvent(convo::HealthEvent{convo::getCurrentTimeUs(),
                            convo::HealthEvent::Severity::Warning,
                            EVENT_CROSSFADE_ABORTED_EMERGENCY,
                            abortCount, 0});
                    }
                    return;  // 通常のクロスフェード処理をスキップ
                }
            }
        }

        // 1. activate (publish 成功後にのみ実行)
        lifetime.activate(newDSP);

        // 2. Crossfade または Retire
        if (decision.needsCrossfade && oldDSP != nullptr) {
            // ★ BUG-054: oldHandle は呼び出し側（enqueue 時に resolve 済みの真の old DSP handle）を
            //   使用する。getActiveRuntimeDSPHandle() は commitRuntimePublication の activate 後に
            //   呼ばれるため NEW DSP の handle を返し、old==new の同一 crossfade を登録していた。
            auto newHandle = engine_.registerDSPHandleForRuntime(newDSP);

            if (!oldHandle.isNull() && !newHandle.isNull()) {
                // CrossfadeAuthorityRuntime が CrossfadeId を発行（唯一権威）
                const auto xfadeId = engine_.crossfadeAuthorityRuntime_.registerCrossfade(oldHandle, newHandle);
                // DSPHandleRuntime: Authority 発行の ID で状態遷移
                engine_.dspHandleRuntime_.beginCrossfade(oldHandle, newHandle, xfadeId);
            }

            // ★ B-1: CAS-only fading slot claim.
            //   exchangeFadingRuntimeDSP の代わりに claimFadingRuntimeDSP を使用。
            //   CAS 成功時点で slot = oldDSP となり、exchange() は不要。
            //   prevRaw を取得する必要がない（CAS が nullptr→oldDSP の直接遷移を保証）。
            const bool claimed = engine_.claimFadingRuntimeDSP(oldDSP);
            if (!claimed) {
                // CAS 失敗 → 別の遷移が既にスロットを占有。oldDSP を直接 retire。
                lifetime.retire(oldDSP);
            } else {
                // ★ HW-1: Publication Metadata を保存（Timer retire パスで epoch 伝搬に使用）
                const auto epoch = engine_.currentPublicationEpoch();
                // ★ P0-2b: DSPHandle のみを保存（DSPCore* は削除）
                // ★ BUG-054: 内側変数を fadingHandle に改名（oldHandle は引数と衝突しない）
                const auto fadingHandle = engine_.dspHandleRuntime_.getFadingRuntimeDSPHandle();
                engine_.storeReceipt(fadingHandle, epoch);
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

    // onTransitionComplete: クロスフェード完了時の処理
    // ★ 注: Coordinator::notifyTransitionComplete は削除済み (ADR-C4)。
    //   本関数も現在呼び出し元ゼロの legacy 実装。
    //   publish ブロックは publishIdleWorldOnly() に置換済みで、実際のクロスフェード
    //   完了経路は Timer が publishIdleWorldOnly() を直接呼ぶ。
    //   残存する fading スロットクリア / crossfade snapshot 更新は Layer 2/3 統合時に
    //   整理予定 (設計知見は ADR-C4 に移管済み)。
    void onTransitionComplete(AudioEngine::DSPCore* currentAfterFade) noexcept
    {
        if (currentAfterFade == nullptr)
            return;

        // ★ B-1: CAS-based fading slot clear
        AudioEngine::DSPCore* current = convo::consumeAtomic(engine_.fadingRuntimeDSPSlot, std::memory_order_acquire);
        if (current != nullptr
            && convo::compareExchangeAtomic(engine_.fadingRuntimeDSPSlot, current,
                                         static_cast<AudioEngine::DSPCore*>(nullptr),
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
        {
            DSPLifetimeManager lifetime(engine_);
            // ★ ISR: Observer — Intent Queue push のみ
            //   Self-contained Intent: getFadingRuntimeDSPHandle から DSPHandle を取得
            const auto fadingHandle = engine_.dspHandleRuntime_.getFadingRuntimeDSPHandle();
            if (!fadingHandle.isNull())
                engine_.runtimePublicationBridge_.submitObserve(fadingHandle, engine_.currentPublicationEpoch());
        }

        engine_.crossfadeRuntime_.setDryHoldSamples(0);
        engine_.refreshCrossfadePreparedSnapshotFromAtomics();

        // publish idling world (publishIdleWorldOnly 経由)
        (void)engine_.publishIdleWorldOnly(currentAfterFade,
            convo::TransitionPolicy::HardReset);
    }

private:
    AudioEngine& engine_;
};

} // namespace convo::isr
