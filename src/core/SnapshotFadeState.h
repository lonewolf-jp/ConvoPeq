#pragma once

#include <atomic>
#include <cstdint>

#include "audioengine/AtomicAccess.h"

namespace convo {

enum class FadeState : uint8_t {
    Idle,
    FadingIn
};

class SnapshotFadeState final
{
public:
    SnapshotFadeState() noexcept
    {
        initialize();
    }

    void initialize() noexcept
    {
        // release ×4: 初期化完了後に他スレッドが acquire で fade 状態を安全に観測できるよう HB を形成。
        convo::publishAtomic(alpha_, 1.0, std::memory_order_release);
        convo::publishAtomic(state_, FadeState::Idle, std::memory_order_release);
        convo::publishAtomic(totalSamples_, 0, std::memory_order_release);
        convo::publishAtomic(remainingSamples_, 0, std::memory_order_release);
    }

    void start(int fadeSamples) noexcept
    {
        // release ×4: startFade 後の advance/update/tryComplete/isFading acquire 側へ開始状態を公開。
        convo::publishAtomic(totalSamples_, fadeSamples, std::memory_order_release);
        convo::publishAtomic(remainingSamples_, fadeSamples, std::memory_order_release);
        convo::publishAtomic(alpha_, 0.0, std::memory_order_release);
        convo::publishAtomic(state_, FadeState::FadingIn, std::memory_order_release);
    }

    void advance(int numSamples) noexcept
    {
        if (state() != FadeState::FadingIn)
            return;

        const int remaining = remainingCount();
        if (remaining <= 0)
            return;

        const int newRemaining = remaining - numSamples;
        if (newRemaining <= 0)
        {
            // release: tryComplete の acquire と HB し残量ゼロを公開。
            convo::publishAtomic(remainingSamples_, 0, std::memory_order_release);
            return;
        }

        // release: tryComplete/advance 次回の acquire と HB し最新残量を公開。
        convo::publishAtomic(remainingSamples_, newRemaining, std::memory_order_release);
        const int total = totalCount();
        if (total > 0)
        {
            const double nextAlpha = 1.0 - static_cast<double>(newRemaining) / static_cast<double>(total);
            // release: updateFade の acquire と HB し最新アルファを公開。
            convo::publishAtomic(alpha_, nextAlpha, std::memory_order_release);
        }
    }

    bool tryComplete() noexcept
    {
        if (state() != FadeState::FadingIn)
            return false;

        if (remainingCount() > 0)
            return false;

        FadeState expected = FadeState::FadingIn;
        return convo::compareExchangeAtomic(state_,
                                            expected,
                                            FadeState::Idle,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire);
    }

    void resetToIdle() noexcept
    {
        // release ×4: advance/update/tryComplete/isFading acquire 側へ idle 状態を公開。
        convo::publishAtomic(state_, FadeState::Idle, std::memory_order_release);
        convo::publishAtomic(alpha_, 1.0, std::memory_order_release);
        convo::publishAtomic(totalSamples_, 0, std::memory_order_release);
        convo::publishAtomic(remainingSamples_, 0, std::memory_order_release);
    }

    [[nodiscard]] FadeState state() const noexcept
    {
        // acquire: start/reset/complete の release と HB して fadeState を観測。
        return convo::consumeAtomic(state_, std::memory_order_acquire);
    }

    [[nodiscard]] bool isFading() const noexcept
    {
        return state() != FadeState::Idle;
    }

    [[nodiscard]] double alpha() const noexcept
    {
        // acquire: advance/reset/complete の alpha release と HB。
        return convo::consumeAtomic(alpha_, std::memory_order_acquire);
    }

    [[nodiscard]] int totalCount() const noexcept
    {
        // acquire: start/reset の totalSamples release と HB。
        return convo::consumeAtomic(totalSamples_, std::memory_order_acquire);
    }

    [[nodiscard]] int remainingCount() const noexcept
    {
        // acquire: start/advance/reset/complete の remainingSamples release と HB。
        return convo::consumeAtomic(remainingSamples_, std::memory_order_acquire);
    }

private:
    std::atomic<double> alpha_ { 1.0 };
    std::atomic<FadeState> state_ { FadeState::Idle };
    std::atomic<int> totalSamples_ { 0 };
    std::atomic<int> remainingSamples_ { 0 };
};

} // namespace convo
