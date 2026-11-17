#pragma once
//==============================================================================
// DrainCapability.h — WORK113 Queue/Gate/Drain 再設計（v4.13 §3/§4）
//
// shutdown の最終 drain を実行する権限。自己申告の bool ではなく、shutdown
// authority（convo::isr::ShutdownRuntime）が検証済みの quiescence proof から
// 発行する capability である。
//
//   - 発行は ShutdownRuntime のみ（private ctor + friend）
//   - move-only（コピー禁止）
//   - single-use（consume() の CAS で 1 回だけ成功）
//   - 発行時点の epoch を保持し、drain 側が fence 後の epoch と照合する
//
// 発行条件は ShutdownRuntime::tryMakeDrainCapability() が確認する:
//   proof.valid()（Q0〜Q7）、finalizeRetire 呼び出し 1 回、
//   requests == DEnqueues + untransferred（所有権が残っていない）
//==============================================================================

#include <atomic>
#include <cstdint>
#include <utility>

namespace convo
{

namespace isr { class ShutdownRuntime; }

class DrainCapability
{
public:
    DrainCapability (const DrainCapability&) = delete;
    DrainCapability& operator= (const DrainCapability&) = delete;

    DrainCapability (DrainCapability&& other) noexcept
        : valid_ (other.valid_)
        , producersJoined_ (other.producersJoined_)
        , readersQuiescent_ (other.readersQuiescent_)
        , postStopEnqueueZero_ (other.postStopEnqueueZero_)
        , epochSettled_ (other.epochSettled_)
        , epochAtIssue_ (other.epochAtIssue_)
        , finalizeCalls_ (other.finalizeCalls_)
        , finalizeRequests_ (other.finalizeRequests_)
    {
        other.valid_ = false;
        other.consumed_.store (true, std::memory_order_release);
    }

    DrainCapability& operator= (DrainCapability&&) = delete;

    [[nodiscard]] bool consume() noexcept
    {
        if (! valid_)
            return false;
        bool expected = false;
        return consumed_.compare_exchange_strong (expected, true,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire);
    }

    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] bool producersJoined() const noexcept { return producersJoined_; }
    [[nodiscard]] bool readersQuiescent() const noexcept { return readersQuiescent_; }
    [[nodiscard]] bool postStopEnqueueZero() const noexcept { return postStopEnqueueZero_; }
    [[nodiscard]] bool epochSettled() const noexcept { return epochSettled_; }
    [[nodiscard]] std::uint64_t epochAtIssue() const noexcept { return epochAtIssue_; }
    [[nodiscard]] std::uint64_t finalizeCalls() const noexcept { return finalizeCalls_; }
    [[nodiscard]] std::uint64_t finalizeRequests() const noexcept { return finalizeRequests_; }

private:
    friend class convo::isr::ShutdownRuntime;

    DrainCapability (bool valid,
                     bool producersJoined,
                     bool readersQuiescent,
                     bool postStopEnqueueZero,
                     bool epochSettled,
                     std::uint64_t epochAtIssue,
                     std::uint64_t finalizeCalls,
                     std::uint64_t finalizeRequests) noexcept
        : valid_ (valid)
        , producersJoined_ (producersJoined)
        , readersQuiescent_ (readersQuiescent)
        , postStopEnqueueZero_ (postStopEnqueueZero)
        , epochSettled_ (epochSettled)
        , epochAtIssue_ (epochAtIssue)
        , finalizeCalls_ (finalizeCalls)
        , finalizeRequests_ (finalizeRequests)
    {
    }

    bool valid_ { false };
    bool producersJoined_ { false };
    bool readersQuiescent_ { false };
    bool postStopEnqueueZero_ { false };
    bool epochSettled_ { false };
    std::uint64_t epochAtIssue_ { 0 };
    std::uint64_t finalizeCalls_ { 0 };
    std::uint64_t finalizeRequests_ { 0 };
    std::atomic<bool> consumed_ { false };
};

} // namespace convo
