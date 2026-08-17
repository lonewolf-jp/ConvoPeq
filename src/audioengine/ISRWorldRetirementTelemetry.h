#pragma once

#include "AtomicAccess.h"
#include <atomic>
#include <cstdint>

namespace convo {
namespace isr {

// ★ T1 (Phase I): observation window tag（D76.2）— sampler が観測ウィンドウの状態を表す。
enum class ObservationWindowTag : uint8_t {
    Normal = 0,     // 通常観測
    Stall,          // retire 滞留（T_stall = 5s 超過等・D82 関連）
    Shutdown,       // shutdown 中（最終 export 用）
    Catastrophic    // 異常（quarantine overflow 等・診断のみ）
};

// ★ T1: window tag の表示名（export 用・診断のみ）。
inline const char* windowTagName(int tag) noexcept
{
    switch (static_cast<ObservationWindowTag>(tag)) {
        case ObservationWindowTag::Normal:      return "normal";
        case ObservationWindowTag::Stall:       return "stall";
        case ObservationWindowTag::Shutdown:    return "shutdown";
        case ObservationWindowTag::Catastrophic: return "catastrophic";
    }
    return "unknown";
}

// ★ T1 (D91): measurement window state — sampler が唯一の transition owner（D91 基準 3）。
//   Idle → StartRequested → Running → EndRequested → Closed → Idle（単調遷移・D91.1）。
enum class MeasurementState : uint8_t {
    Idle = 0,
    StartRequested,
    Running,
    EndRequested,
    Closed
};

// ★ T1 (D91): Closed window の immutable snapshot（export race 対策・D91 監視項目 4）。
//   trivially copyable（全スカラー）→ std::atomic<MeasurementSnapshot> で publish 可能。
struct MeasurementSnapshot {
    std::uint64_t windowId = 0;
    std::uint64_t startAcquire = 0;
    std::uint64_t startRelease = 0;
    std::uint64_t endAcquire = 0;
    std::uint64_t endRelease = 0;
    std::int64_t finalEstimate = 0;
    std::int64_t windowMax = 0;                // bounded sampled maximum（D91 基準 8）
    std::uint64_t windowStartTimestampUs = 0;
    std::uint64_t windowEndTimestampUs = 0;
    std::uint64_t sampleCount = 0;
    std::uint64_t maxSamplingGapUs = 0;
    std::uint64_t missedTickCount = 0;
    std::uint64_t counterWrapped = 0;          // 診断のみ（D91 基準 9・trigger にしない）
    std::uint64_t valid = 0;
};
static_assert(std::is_trivially_copyable_v<MeasurementSnapshot>,
    "MeasurementSnapshot must be trivially copyable for atomic publish");

// ★ T1 (Phase I): World retirement observation counters — observational state ONLY.
//   NOT a reservation authority（held_count / held_set / free_stack / token / R gate は T2・D76/D86）。
//   D76.4 不変条件: 「T1 telemetry state is observational state and is not a reservation authority」。
//
//   責務分離（D83.2 / D86）:
//     publish / retirement terminal path → acquireObserved / releaseObserved（atomic observation counters）
//     Non-RT sampler → A/R loads → signedWide(A) - signedWide(R) → observedOutstandingEstimate
//                     → observedOutstandingMax → window-tagged export
//
//   acquireObserved: retirement obligation 生成（publish 成功・CoordinatorLoop Non-RT）で +1（D76.3）。
//   releaseObserved: sampler が storage 側の worldReclaimCount（type==World の terminal deleter 実行数・D86）
//                    の累積差分を反映（D83.2 責務分離・sampler は Non-RT）。
//   observedOutstandingEstimate: D82 の signedWide(A) - signedWide(R)（unsigned subtraction の
//                                wraparound を回避・int64 に cast 後減算）。
//   observedOutstandingMax: Non-RT sampler 側でのみ更新（D83/D86・acquire/release 側では更新しない）。
class WorldRetirementTelemetry {
public:
    // ★ acquire observation: retirement obligation 生成（publish 成功）で +1。
    //   LP = publish 成功（onRuntimePublishedNonRt・CoordinatorLoop Non-RT・atomic のみ・RT-safe）。
    void onAcquireObserved() noexcept
    {
        convo::fetchAddAtomic(acquireObserved_, std::uint64_t{1}, std::memory_order_acq_rel);
    }

    // ★ release observation 反映: sampler（Non-RT）が worldReclaimCount の累積差分を反映する。
    //   実体の更新（type==World の terminal deleter 実行後・D86.1 の順序）は storage 側の
    //   worldReclaimCount_ が担う。本メソッドは sampler が差分を移すための入口。
    void onReleaseObserved() noexcept
    {
        convo::fetchAddAtomic(releaseObserved_, std::uint64_t{1}, std::memory_order_acq_rel);
    }

    // ★ sampler が storage 側の worldReclaimCount の累積差分（delta）を一度に反映（Non-RT・効率化）。
    void addReleaseObserved(std::uint64_t count) noexcept
    {
        if (count == 0)
            return;
        convo::fetchAddAtomic(releaseObserved_, count, std::memory_order_acq_rel);
    }

    [[nodiscard]] std::uint64_t acquireObserved() const noexcept
    {
        return convo::consumeAtomic(acquireObserved_, std::memory_order_acquire);
    }

    [[nodiscard]] std::uint64_t releaseObserved() const noexcept
    {
        return convo::consumeAtomic(releaseObserved_, std::memory_order_acquire);
    }

    // ★ D82: O = signedWide(A) - signedWide(R)（unsigned subtraction の wraparound を回避）。
    //   A / R は uint64_t（monotonic・測定期間中 wraparound しない前提・D82.2）。
    [[nodiscard]] std::int64_t observedOutstandingEstimate() const noexcept
    {
        return static_cast<std::int64_t>(acquireObserved())
             - static_cast<std::int64_t>(releaseObserved());
    }

    // ★ 負値検出は診断のみ（D77.2 / D86 非交渉条件 6）: observedOutstanding < 0 が発生しても
    //   処理停止・rollback・補正を行わない（lifetime correctness に影響しない）。
    [[nodiscard]] bool isNegative() const noexcept
    {
        return observedOutstandingEstimate() < 0;
    }

    // ★ observedOutstandingMax: Non-RT sampler 側でのみ更新（D83.2 / D86 非交渉条件 8）。
    //   acquire/release 側では更新しない（sampler 責務・RT 側状態管理と分離）。
    void updateObservedOutstandingMax(std::int64_t value) noexcept
    {
        auto current = convo::consumeAtomic(observedOutstandingMax_, std::memory_order_acquire);
        while (value > current
               && !convo::compareExchangeAtomic(observedOutstandingMax_, current, value,
                                                std::memory_order_acq_rel, std::memory_order_acquire))
        {
        }
    }

    [[nodiscard]] std::int64_t observedOutstandingMax() const noexcept
    {
        return convo::consumeAtomic(observedOutstandingMax_, std::memory_order_acquire);
    }

    // ★ window tag（D76.2）: sampler が観測ウィンドウの状態を設定（Non-RT）。
    void setWindowTag(ObservationWindowTag tag) noexcept
    {
        convo::publishAtomic(windowTag_, static_cast<std::uint8_t>(tag), std::memory_order_release);
    }

    [[nodiscard]] ObservationWindowTag windowTag() const noexcept
    {
        return static_cast<ObservationWindowTag>(
            convo::consumeAtomic(windowTag_, std::memory_order_acquire));
    }

    // ── ★ T1 (D91): window-reset measurement API（Non-RT 限定・D91 基準 1・2）──

    // Start request: CAS Idle → StartRequested（Idle 以外は無視・D91.1 上書き・重複要求契約）。
    //   StartRequested 中の追加 Start は既存 measurement window を変更しない。
    void requestMeasurementStart() noexcept
    {
        std::uint8_t expected = static_cast<std::uint8_t>(MeasurementState::Idle);
        convo::compareExchangeAtomic(measurementState_, expected,
                                     static_cast<std::uint8_t>(MeasurementState::StartRequested),
                                     std::memory_order_acq_rel, std::memory_order_acquire);
    }

    // End request: CAS Running → EndRequested（Running 以外は無視・D91.1 上書き・重複要求契約）。
    //   EndRequested / Closed 中の追加 End は既存 measurement window を変更しない。
    void requestMeasurementEnd() noexcept
    {
        std::uint8_t expected = static_cast<std::uint8_t>(MeasurementState::Running);
        convo::compareExchangeAtomic(measurementState_, expected,
                                     static_cast<std::uint8_t>(MeasurementState::EndRequested),
                                     std::memory_order_acq_rel, std::memory_order_acquire);
    }

    // ★ sampler が各 tick の最後に呼ぶ（timerCallback・唯一の transition owner・D91 基準 3）。
    //   request を観測し、Start/End transition を実行する（sampler が linearization point・D91.1）。
    void samplerTick(std::uint64_t nowTimestampUs) noexcept
    {
        const auto state = static_cast<MeasurementState>(
            convo::consumeAtomic(measurementState_, std::memory_order_acquire));
        switch (state) {
            case MeasurementState::StartRequested: beginWindow(nowTimestampUs); break;
            case MeasurementState::Running:        sampleWindow(nowTimestampUs); break;
            case MeasurementState::EndRequested:   closeWindow(nowTimestampUs); break;
            case MeasurementState::Idle:
            case MeasurementState::Closed:
            default: break;
        }
    }

    // Closed snapshot の読み取り（export 用・window state を変更しない・D91 基準 10）。
    [[nodiscard]] MeasurementSnapshot lastClosedSnapshot() const noexcept
    {
        return convo::consumeAtomic(snapshot_, std::memory_order_acquire);
    }

    [[nodiscard]] MeasurementState measurementState() const noexcept
    {
        return static_cast<MeasurementState>(
            convo::consumeAtomic(measurementState_, std::memory_order_acquire));
    }

private:
    // ★ sampler のみ（MessageThread）が呼ぶ・window transition の linearization point（D91.1）。
    void beginWindow(std::uint64_t nowTimestampUs) noexcept
    {
        // A0/R0 snapshot・windowMax 初期値 = 最初の estimate（D91 監視項目 1）・windowStart・windowId++。
        const auto a0 = acquireObserved();
        const auto r0 = releaseObserved();
        const std::int64_t firstEstimate = static_cast<std::int64_t>(a0) - static_cast<std::int64_t>(r0);
        const std::uint64_t newWindowId = convo::consumeAtomic(windowId_, std::memory_order_acquire) + 1;
        convo::publishAtomic(windowId_, newWindowId, std::memory_order_release);
        convo::publishAtomic(startAcquire_, a0, std::memory_order_release);
        convo::publishAtomic(startRelease_, r0, std::memory_order_release);
        convo::publishAtomic(windowMax_, firstEstimate, std::memory_order_release);   // 監視項目 1
        convo::publishAtomic(windowStartTimestampUs_, nowTimestampUs, std::memory_order_release);
        convo::publishAtomic(sampleCount_, std::uint64_t{1}, std::memory_order_release);
        convo::publishAtomic(lastSampleTimestampUs_, nowTimestampUs, std::memory_order_release);
        convo::publishAtomic(maxSamplingGapUs_, std::uint64_t{0}, std::memory_order_release);
        convo::publishAtomic(missedTickCount_, std::uint64_t{0}, std::memory_order_release);
        convo::publishAtomic(measurementState_, static_cast<std::uint8_t>(MeasurementState::Running),
                             std::memory_order_release);
    }

    // ★ sampler のみ・Running 中の各 tick で estimate を計算し windowMax を更新（D91 基準 6）。
    void sampleWindow(std::uint64_t nowTimestampUs) noexcept
    {
        const auto a = acquireObserved();
        const auto r = releaseObserved();
        const std::int64_t estimate = static_cast<std::int64_t>(a) - static_cast<std::int64_t>(r);
        updateWindowMax(estimate);
        // sampling gap / missed tick 統計（D89.2 measurement protocol）
        const auto prevTs = convo::consumeAtomic(lastSampleTimestampUs_, std::memory_order_acquire);
        const std::uint64_t gapUs = (nowTimestampUs > prevTs)
            ? (nowTimestampUs - prevTs) : std::uint64_t{0};
        if (gapUs > (kExpectedTickIntervalUs * 2))
            convo::fetchAddAtomic(missedTickCount_, std::uint64_t{1}, std::memory_order_acq_rel);
        updateMaxSamplingGap(gapUs);
        convo::publishAtomic(lastSampleTimestampUs_, nowTimestampUs, std::memory_order_release);
        convo::fetchAddAtomic(sampleCount_, std::uint64_t{1}, std::memory_order_acq_rel);
    }

    // ★ sampler のみ・End transition で A1/R1 → estimate → windowMax 更新 → Closed（D91 監視項目 2 の順序）。
    void closeWindow(std::uint64_t nowTimestampUs) noexcept
    {
        const auto a1 = acquireObserved();
        const auto r1 = releaseObserved();
        const std::int64_t finalEstimate = static_cast<std::int64_t>(a1) - static_cast<std::int64_t>(r1);
        updateWindowMax(finalEstimate);   // 監視項目 2: 最後の観測値も windowMax に含める
        const auto a0 = convo::consumeAtomic(startAcquire_, std::memory_order_acquire);
        const auto r0 = convo::consumeAtomic(startRelease_, std::memory_order_acquire);
        const std::uint64_t wrapped = ((a1 < a0) || (r1 < r0)) ? 1 : 0;   // 診断のみ（D91 基準 9）
        convo::publishAtomic(counterWrapped_, wrapped, std::memory_order_release);
        convo::publishAtomic(endAcquire_, a1, std::memory_order_release);
        convo::publishAtomic(endRelease_, r1, std::memory_order_release);
        convo::publishAtomic(windowEndTimestampUs_, nowTimestampUs, std::memory_order_release);
        // Closed snapshot を immutable publish（D91 監視項目 4・export race 対策）
        MeasurementSnapshot snap{};
        snap.windowId = convo::consumeAtomic(windowId_, std::memory_order_acquire);
        snap.startAcquire = a0;
        snap.startRelease = r0;
        snap.endAcquire = a1;
        snap.endRelease = r1;
        snap.finalEstimate = finalEstimate;
        snap.windowMax = convo::consumeAtomic(windowMax_, std::memory_order_acquire);
        snap.windowStartTimestampUs = convo::consumeAtomic(windowStartTimestampUs_, std::memory_order_acquire);
        snap.windowEndTimestampUs = nowTimestampUs;
        snap.sampleCount = convo::consumeAtomic(sampleCount_, std::memory_order_acquire);
        snap.maxSamplingGapUs = convo::consumeAtomic(maxSamplingGapUs_, std::memory_order_acquire);
        snap.missedTickCount = convo::consumeAtomic(missedTickCount_, std::memory_order_acquire);
        snap.counterWrapped = wrapped;
        snap.valid = 1;
        convo::publishAtomic(snapshot_, snap, std::memory_order_release);
        // 監視項目 3: Closed → Idle（明示的・同一 tick で次の Start request が失われない）
        //   この間に Start request が発行されていたら measurementState_ は StartRequested のまま →
        //   EndRequested → Idle の CAS は失敗 → 次の tick で beginWindow が実行される（request は失われない）。
        std::uint8_t expected = static_cast<std::uint8_t>(MeasurementState::EndRequested);
        convo::compareExchangeAtomic(measurementState_, expected,
                                     static_cast<std::uint8_t>(MeasurementState::Idle),
                                     std::memory_order_acq_rel, std::memory_order_acquire);
    }

    void updateWindowMax(std::int64_t value) noexcept
    {
        auto current = convo::consumeAtomic(windowMax_, std::memory_order_acquire);
        while (value > current
               && !convo::compareExchangeAtomic(windowMax_, current, value,
                                                std::memory_order_acq_rel, std::memory_order_acquire))
        {
        }
    }

    void updateMaxSamplingGap(std::uint64_t gapUs) noexcept
    {
        auto current = convo::consumeAtomic(maxSamplingGapUs_, std::memory_order_acquire);
        while (gapUs > current
               && !convo::compareExchangeAtomic(maxSamplingGapUs_, current, gapUs,
                                                std::memory_order_acq_rel, std::memory_order_acquire))
        {
        }
    }

    std::atomic<std::uint64_t> acquireObserved_{0};
    std::atomic<std::uint64_t> releaseObserved_{0};
    std::atomic<std::int64_t> observedOutstandingMax_{0};          // ★ sampler のみ更新（D83.2）
    std::atomic<std::uint8_t> windowTag_{static_cast<std::uint8_t>(ObservationWindowTag::Normal)};

    // ★ T1 (D90.3 / D91): window-reset measurement state（すべて atomic・bounded measurement）。
    static constexpr std::uint64_t kExpectedTickIntervalUs = 100'000;   // 100ms（timerPeriodMs_）
    std::atomic<std::uint8_t> measurementState_{static_cast<std::uint8_t>(MeasurementState::Idle)};
    std::atomic<std::uint64_t> windowId_{0};
    std::atomic<std::uint64_t> windowStartTimestampUs_{0};
    std::atomic<std::uint64_t> windowEndTimestampUs_{0};
    std::atomic<std::uint64_t> startAcquire_{0};
    std::atomic<std::uint64_t> startRelease_{0};
    std::atomic<std::uint64_t> endAcquire_{0};
    std::atomic<std::uint64_t> endRelease_{0};
    std::atomic<std::int64_t> windowMax_{0};                          // bounded・sampler のみ（D91 基準 8）
    std::atomic<std::uint64_t> sampleCount_{0};
    std::atomic<std::uint64_t> maxSamplingGapUs_{0};
    std::atomic<std::uint64_t> missedTickCount_{0};
    std::atomic<std::uint64_t> lastSampleTimestampUs_{0};
    std::atomic<std::uint64_t> counterWrapped_{0};                    // 診断のみ（D91 基準 9）
    std::atomic<MeasurementSnapshot> snapshot_{};                     // Closed result の immutable publish
};

} // namespace isr
} // namespace convo
