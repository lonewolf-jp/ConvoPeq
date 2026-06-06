#pragma once

#include <atomic>
#include <algorithm>
#include "AtomicAccess.h"
#include "DspNumericPolicy.h"

// CrossfadeRuntime: Crossfade の実行状態を一元管理する Runtime Component。
// AudioEngine から分離され、DSPTransition から参照される。
// ★ Authority/Execution 分離: 判断 (WHETHER) は CrossfadeAuthority、
//   状態管理 (WHAT/STATE) は CrossfadeRuntime が担当。
//
// ★ CrossfadeRuntime は AudioEngine.h を include しない。
//   循環依存回避のため、必要な型 (LinearRamp) は DspNumericPolicy.h から取り込む。
//
// ★ CrossfadeRuntime は CrossfadeId を生成/管理しない。
//   Crossfade ID の権威は CrossfadeAuthorityRuntime::registerCrossfade() が保持。
//   CrossfadeRuntime は Execution Cache として振る舞う。

namespace convo::isr {

class CrossfadeRuntime {
public:
    CrossfadeRuntime() noexcept = default;

    // start: DSPTransition::onPublishCompleted() から呼ばれる
    void start(double fadeTimeSec, double sampleRate) noexcept
    {
        gain_.reset(sampleRate, std::max(0.001, fadeTimeSec));
        gain_.setCurrentAndTargetValue(0.0);
        convo::publishAtomic(pending_, true, std::memory_order_release);
        convo::publishAtomic(queuedFadeTimeSec_, fadeTimeSec, std::memory_order_release);
        convo::publishAtomic(useDryAsOld_, false, std::memory_order_release);
        convo::publishAtomic(firstIrDryPending_, false, std::memory_order_release);
        convo::publishAtomic(startDelayBlocks_, 0, std::memory_order_release);
        convo::publishAtomic(dryHoldSamples_, 0, std::memory_order_release);
        // activeCrossfadeId_ は触らない — CrossfadeAuthorityRuntime の権威
    }

    // complete: クロスフェード完了時 (timer から)
    void complete() noexcept
    {
        convo::publishAtomic(pending_, false, std::memory_order_release);
        convo::publishAtomic(queuedFadeTimeSec_, 0.030, std::memory_order_release);
    }

    // reset: shutdown/releaseResources 時
    void reset() noexcept
    {
        convo::publishAtomic(pending_, false, std::memory_order_release);
        convo::publishAtomic(useDryAsOld_, false, std::memory_order_release);
        convo::publishAtomic(firstIrDryPending_, false, std::memory_order_release);
        convo::publishAtomic(firstIrDryDone_, false, std::memory_order_release);
        convo::publishAtomic(startDelayBlocks_, 0, std::memory_order_release);
        convo::publishAtomic(dryHoldSamples_, 0, std::memory_order_release);
        convo::publishAtomic(queuedFadeTimeSec_, 0.030, std::memory_order_release);
        convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release);
        gain_.setCurrentAndTargetValue(1.0);
        dryScaleGain_.setCurrentAndTargetValue(1.0);
    }

    // === RT-safe read-only access ===
    [[nodiscard]] bool isPending() const noexcept
        { return convo::consumeAtomic(pending_, std::memory_order_acquire); }
    [[nodiscard]] bool useDryAsOld() const noexcept
        { return convo::consumeAtomic(useDryAsOld_, std::memory_order_acquire); }
    [[nodiscard]] bool isFirstIrDryPending() const noexcept
        { return convo::consumeAtomic(firstIrDryPending_, std::memory_order_acquire); }
    [[nodiscard]] bool isFirstIrDryDone() const noexcept
        { return convo::consumeAtomic(firstIrDryDone_, std::memory_order_acquire); }
    [[nodiscard]] int getStartDelayBlocks() const noexcept
        { return convo::consumeAtomic(startDelayBlocks_, std::memory_order_acquire); }
    [[nodiscard]] int getDryHoldSamples() const noexcept
        { return convo::consumeAtomic(dryHoldSamples_, std::memory_order_acquire); }
    [[nodiscard]] double getQueuedFadeTimeSec() const noexcept
        { return convo::consumeAtomic(queuedFadeTimeSec_, std::memory_order_acquire); }
    [[nodiscard]] double getDryScaleTarget() const noexcept
        { return convo::consumeAtomic(dryScaleTarget_, std::memory_order_acquire); }
    [[nodiscard]] convo::LinearRamp& getGain() noexcept { return gain_; }
    [[nodiscard]] const convo::LinearRamp& getGain() const noexcept { return gain_; }
    [[nodiscard]] convo::LinearRamp& getDryScaleGain() noexcept { return dryScaleGain_; }
    [[nodiscard]] const convo::LinearRamp& getDryScaleGain() const noexcept { return dryScaleGain_; }

    // === NonRT publish setters ===
    void setUseDryAsOld(bool v) noexcept
        { convo::publishAtomic(useDryAsOld_, v, std::memory_order_release); }
    void setFirstIrDryPending(bool v) noexcept
        { convo::publishAtomic(firstIrDryPending_, v, std::memory_order_release); }
    void setFirstIrDryDone(bool v) noexcept
        { convo::publishAtomic(firstIrDryDone_, v, std::memory_order_release); }
    void setStartDelayBlocks(int v) noexcept
        { convo::publishAtomic(startDelayBlocks_, v, std::memory_order_release); }
    void setDryHoldSamples(int v) noexcept
        { convo::publishAtomic(dryHoldSamples_, v, std::memory_order_release); }
    void setDryScaleTarget(double v) noexcept
        { convo::publishAtomic(dryScaleTarget_, v, std::memory_order_release); }

private:
    std::atomic<bool> pending_{ false };
    std::atomic<bool> useDryAsOld_{ false };
    std::atomic<bool> firstIrDryPending_{ false };
    std::atomic<bool> firstIrDryDone_{ false };
    std::atomic<int> startDelayBlocks_{ 0 };
    std::atomic<int> dryHoldSamples_{ 0 };
    std::atomic<double> queuedFadeTimeSec_{ 0.030 };
    std::atomic<double> dryScaleTarget_{ 1.0 };
    convo::LinearRamp gain_;
    convo::LinearRamp dryScaleGain_;
    // activeCrossfadeId_ は CrossfadeRuntime に持たせない
    // CrossfadeAuthorityRuntime が唯一権威
};

} // namespace convo::isr
