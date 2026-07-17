#pragma once

#include "AudioEngine.h"
#include "ISRRetireRouter.h"

// DSPLifetimeManager: DSP の activation / crossfade / retire を一元管理する。
// Publication 完了後に NonRT で非同期的に呼ばれる。
//
// ★Phase-B: ISRRetireRouter 経由で EpochDomain に直接 enqueueRetire する。
//   retireDSP()（削除済み R-2）のラッパではなく、Router → EpochDomain へ直接委譲する。
class DSPLifetimeManager {
public:
    explicit DSPLifetimeManager(AudioEngine& engine) noexcept
        : engine_(engine), router_(engine.m_retireRouter.get()) {}

    explicit DSPLifetimeManager(AudioEngine& engine, convo::isr::ISRRetireRouter* router) noexcept
        : engine_(engine), router_(router) {}

    // Authority: DSPLifetimeManager (Lifecycle Authority)
    // ★ work70-FIX: activate — DSPCore* + DSPHandle 両方を活性化する。
    //   Authority: DSPLifetimeManager (DSPCore* lifecycle) + DSPHandleRuntime (Handle lifecycle)
    //   が、activeRuntimeDSPHandle_ の更新は commitRuntimePublication() が唯一のAuthority。
    //   ここでは DSPCore* の setActiveRuntimeDSP のみ行い、Handle の activate は行わない。
    //   [設計決定]: activeRuntimeDSPHandle_ は commitRuntimePublication() 内の publish 成功後にのみ更新。
    void activate(AudioEngine::DSPCore* dsp) noexcept
    {
        if (dsp == nullptr) return;
        engine_.setActiveRuntimeDSP(dsp);
    }

    // Authority: CrossfadeAuthorityRuntime（id は Authority から注入）
    void beginCrossfade(convo::isr::DSPHandle from, convo::isr::DSPHandle to, convo::isr::CrossfadeId id) noexcept
    {
        engine_.dspHandleRuntime_.beginCrossfade(from, to, id);
    }

    // Authority: DSPLifetimeManager (Lifecycle Authority)
    // Retire pipeline: DSPLifetimeManager → ISRRetireRouter → EpochDomain
    // [Bug2 Phase1] enqueueWithRetry に委譲（リトライロジックは Router に集約）
    void retire(AudioEngine::DSPCore* dsp) noexcept
    {
        if (dsp == nullptr) return;
        // 1. Release DSP handle (must happen before enqueue)
        if (!engine_.retireDSPHandleForRuntime(dsp))
            return;

        // 2. Route through ISRRetireRouter（enqueueWithRetry が tryReclaim + 再試行を内包）
        const uint64_t epoch = router_->currentEpoch();
        router_->enqueueWithRetry(static_cast<void*>(dsp),
                                   &AudioEngine::destroyDSPCoreNode,
                                   epoch,
                                   DeletionEntryType::Generic);

        convo::fetchAddAtomic(engine_.rtAuxMutable_.runtimeRetireCount,
                              static_cast<std::uint64_t>(1),
                              std::memory_order_acq_rel);

        // ★ work70 P1-c: 最新の retire 対象世代を記録（MEM_SNAP の retiringGeneration 用）
        const uint64_t committedGen = convo::consumeAtomic(
            engine_.lastCommittedRuntimeGeneration_, std::memory_order_acquire);
        convo::publishAtomic(currentRetiringGeneration_, committedGen, std::memory_order_release);
    }

    void retireDeferred() noexcept
    {
        // deferred queue drain: handled by AudioEngine threading
    }

    AudioEngine::DSPCore* getActive() const noexcept { return engine_.getActiveRuntimeDSP(); }

    // ★ work70 Phase2: destroyRolledBackDSP — EBR を経由しない特殊破棄ルート。
    //   「Publication Authority から返却された未公開オブジェクト（Never Published Object）」
    //   のみを対象とし、EBR epoch 保護は不要（publish されたことのない DSPCore は
    //   Audio Thread から到達不能なため）。
    //   事前条件: Handle は既に rollback 済み（Reclaimed）。
    //   post-condition: DSPCore のメモリが解放される。
    void destroyRolledBackDSP(AudioEngine::DSPCore* dsp) noexcept
    {
        if (dsp == nullptr) return;
        engine_.destroyDSPCoreNode(dsp);
    }

    // ★ work70 P1-c: MEM_SNAP の retiringGeneration 用（DSPLifetimeManager が唯一の Authority）
    [[nodiscard]] uint64_t retiringGeneration() const noexcept {
        return convo::consumeAtomic(currentRetiringGeneration_, std::memory_order_acquire);
    }

private:
    AudioEngine& engine_;
    convo::isr::ISRRetireRouter* router_;
    std::atomic<uint64_t> currentRetiringGeneration_{0};  // ★ work70 P1-c: retire 対象世代
};
