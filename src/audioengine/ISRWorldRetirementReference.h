#pragma once

#include "AtomicAccess.h"
#include "ISRWorldRetirementTelemetry.h"   // ★ T1 (D100.4): releaseObserved 転送（sampler の outstanding 推定を正しくする）
#include <atomic>
#include <cstdint>

namespace convo {
namespace isr {

// ★ T1 (D94/D95/D96/D97/D98): reference observer — measurement only・retirement authority ではない。
//   no ownership・no reclaim decision・no lifetime decision・no admission decision・no R/R_cap・no ReservationExhausted。
//   D94 設計原則: reference instrumentation は T1 telemetry の契約に混ぜない（production diagnostic と分離）。
//
//   目的: T_w（retirement event reference maximum）を高頻度（event-driven）に観測し、100ms sampler の O_w
//   （sampled windowMax）と同一 windowId で比較する（E_w = T_w - O_w）。
//
//   event-driven（D95 固定点 4）: acquire/release event 自体を観測点として running max を更新する。
//   release は terminal deleter 成功後のみ（D95 固定点 3・D87 exactly-once と同じ terminalization boundary）。
//
//   window 境界（D96 実装ゲート ④）: Start/End は既存 sampler boundary（T1 の sampler が linearize）。
//   observer の running は sampler の state 遷移に同期して切り替わる（冪等）。
//
//   onRelease() は例外を投げない・所有権を変更しない・reclaim を再試行しない（D97・retirement control-flow に波及しない）。
class WorldRetirementReferenceObserver {
public:
    // ★ acquire event（publish 成功・onRuntimePublishedNonRt）: referenceAcquireCount_++ → running max 更新（event-driven）。
    //   LP = publish 成功（CoordinatorLoop Non-RT・atomic のみ・RT-safe）。
    void onAcquire() noexcept
    {
        convo::fetchAddAtomic(referenceAcquireCount_, std::uint64_t{1}, std::memory_order_acq_rel);
        updateRunningMax();
    }

    // ★ release event（type==World の terminal deleter 成功後・4 箇所）: referenceReleaseCount_++ → running max 更新。
    //   例外・所有権変更・reclaim 再試行なし（retirement control-flow に波及しない・D97）。
    //   ★ T1 (R3 authority fix / D100 separate): reference max 用のみ。telemetry.releaseObserved へは転送しない（separate observation）。
    //     T1 release measurement の authoritative source は worldReclaimCount → sampler → addReleaseObserved(delta) のみ。
    void onRelease() noexcept
    {
        convo::fetchAddAtomic(referenceReleaseCount_, std::uint64_t{1}, std::memory_order_acq_rel);
        updateRunningMax();
    }

    // ★ T1 (D100.4): sampler 側の release 観測カウンタ（releaseObserved）を更新する telemetry を設定。
    //   non-owning・AudioEngine が初期化時に配線（reference observer は measurement only・D94）。
    void setTelemetry(WorldRetirementTelemetry* telemetry) noexcept
    {
        telemetry_ = telemetry;
    }

    // ★ Start（sampler の linearization point で通知・冪等）: baseline = 現在の outstanding（0 にリセットしない・
    //   D95 固定点 1・6）・referenceMax を baseline に初期化・running 開始。
    void onMeasurementStart() noexcept
    {
        std::uint8_t expected = 0;
        if (convo::compareExchangeAtomic(running_, expected, std::uint8_t{1},
                                         std::memory_order_acq_rel, std::memory_order_acquire))
        {
            // 初回のみ baseline を設定（window ごとに 1 回・冪等）
            const auto baseline = referenceOutstanding();
            convo::publishAtomic(referenceMax_, baseline, std::memory_order_release);
        }
    }

    // ★ End（sampler の linearization point で通知）: running 停止（referenceMax を T_w として確定・
    //   End tick までに発生した terminal release を含む・D95 固定点 5・7）。
    void onMeasurementEnd() noexcept
    {
        convo::publishAtomic(running_, std::uint8_t{0}, std::memory_order_release);
    }

    [[nodiscard]] std::uint64_t referenceAcquireCount() const noexcept
    {
        return convo::consumeAtomic(referenceAcquireCount_, std::memory_order_acquire);
    }

    [[nodiscard]] std::uint64_t referenceReleaseCount() const noexcept
    {
        return convo::consumeAtomic(referenceReleaseCount_, std::memory_order_acquire);
    }

    // signedWide: acquire - release（D82 の signedWide 算術・unsigned wraparound 回避）。
    [[nodiscard]] std::int64_t referenceOutstanding() const noexcept
    {
        return static_cast<std::int64_t>(referenceAcquireCount())
             - static_cast<std::int64_t>(referenceReleaseCount());
    }

    // ★ referenceMax（window 内 running max・event-driven）: T_w。
    [[nodiscard]] std::int64_t referenceMax() const noexcept
    {
        return convo::consumeAtomic(referenceMax_, std::memory_order_acquire);
    }

    [[nodiscard]] bool isRunning() const noexcept
    {
        return convo::consumeAtomic(running_, std::memory_order_acquire) != 0;
    }

private:
    // ★ acquire/release event の双方で呼ぶ（D95 固定点 4・event-driven 更新）。
    //   window 外（running == 0）のイベントは T_w に含めない（Start 前の履歴を持ち込まない・D95 固定点 2）。
    void updateRunningMax() noexcept
    {
        if (convo::consumeAtomic(running_, std::memory_order_acquire) == 0)
            return;
        const auto outstanding = referenceOutstanding();
        auto current = convo::consumeAtomic(referenceMax_, std::memory_order_acquire);
        while (outstanding > current
               && !convo::compareExchangeAtomic(referenceMax_, current, outstanding,
                                                std::memory_order_acq_rel, std::memory_order_acquire))
        {
        }
    }

    std::atomic<std::uint64_t> referenceAcquireCount_{0};
    std::atomic<std::uint64_t> referenceReleaseCount_{0};
    std::atomic<std::int64_t> referenceMax_{0};   // window 内 running max（T_w）
    std::atomic<std::uint8_t> running_{0};        // 1 = window 内（sampler boundary に同期）
    WorldRetirementTelemetry* telemetry_{nullptr}; // ★ T1 (D100.4): releaseObserved 転送先（non-owning）
};

} // namespace isr
} // namespace convo
