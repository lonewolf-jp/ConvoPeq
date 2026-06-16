#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <filesystem>
#include "AtomicAccess.h"  // ★ A-2/A-3: convo::consumeAtomic / publishAtomic
#include "RuntimeDrainAudit.h"  // ★ P2-B: ShutdownBlockingReason

namespace convo {

// ★ P1-B: ISR Health State
enum class ISRHealthState : uint8_t;

namespace isr {

/**
 * ISR 10層 Architecture Layer 8: Shutdown FSM
 * coordinated shutdown sequence と barrier transition
 */

/**
 * Shutdown phase
 */
enum class ShutdownPhase : uint8_t
{
    Running,
    AudioStopped,
    ObserverDrained,
    RetireClosed,
    EpochSettled,
    ReclaimComplete,
    // ★ C-2: EmergencyDrain — Optional/CompileFlag による最終手段
    //   デフォルトではスキップ（既存の graceful drain で十分）
    //   #ifdef CONVOPEQ_EMERGENCY_DRAIN で有効化
    EmergencyDrain,   // ★ C-2
    VerifyDrained,    // ★ P3: 最終監査フェーズ
    TimedOut,
    Failed,
    ShutdownComplete
};

/**
 * ★ P2-B/Practical-3: Shutdown 完了阻害要因
 */
enum class ShutdownBlockingReason : uint8_t
{
    None = 0,
    PendingPublication,
    PendingRetire,
    ActiveCrossfade,
    DeferredPublish,
    QuarantineResident,
    RouterPendingRetire,
    ReaderActive,
    Unknown
};

// ★ A-2: ShutdownBlockingReason 別統計
//    各メンバを個別 std::atomic<uint64_t> にする (32バイト構造体の丸ごと atomic は不可)
//    sizeof(BlockingReasonStats) = 32 > 16 (x64 HW atomic limit: CMPXCHG16B)
//    std::atomic<BlockingReasonStats> は MSVC STL で内部ミューテックスに fallback する
// ★ alignas(64): 配列として連続配置された際の False Sharing を防止
#pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
struct alignas(64) BlockingReasonStats {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> maxDurationUs{0};
    std::atomic<uint64_t> firstSeenUs{0};
};
#pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

// ★ A-2: enum から導出することで enum 変更時の追従漏れを防止
static constexpr size_t kBlockingReasonCount =
    static_cast<size_t>(ShutdownBlockingReason::Unknown) + 1;

// ★ A-3: BlockingReasonEvent を 64bit にパック (8bit reason + 56bit timestampUs)
//    std::atomic<uint64_t> として扱うことで Tearing を完全防止
using PackedBlockingEvent = std::atomic<uint64_t>;

inline uint64_t packEvent(ShutdownBlockingReason reason, uint64_t timestampUs) noexcept {
    return (timestampUs << 8) | static_cast<uint64_t>(reason);
}

// ★ A-3: 独立 TinyRingBuffer (TelemetryRecorder 非依存)
//   要素を std::atomic<uint64_t> にパックすることで Tearing を完全防止。
//   push は fetch_add でインデックス確保後、atomic store。
//   forEach は acquire load で書き込み完了後のデータのみを安全に読む。
template<size_t N>
class TinyRingBuffer {
    static_assert(N > 0 && N <= 256, "TinyRingBuffer size must be 1..256");
public:
    void push(ShutdownBlockingReason reason, uint64_t timestampUs) noexcept {
        // 1. 現在の書き込み位置を取得 (単一Writer前提、relaxedで安全)
        const auto currentIdx = convo::consumeAtomic(writePos_, std::memory_order_relaxed);
        // 2. データを先行して書き込む (Readerはまだこのインデックスを知らない)
        data_[currentIdx % N].store(packEvent(reason, timestampUs), std::memory_order_relaxed);
        // 3. release store: インデックスを更新し、データの書き込み完了を公開
        //    ★ fetch_add は不可: インデックスがデータより先に公開されるため
        convo::publishAtomic(writePos_, currentIdx + 1, std::memory_order_release);
    }
    [[nodiscard]] size_t size() const noexcept {
        const auto wp = convo::consumeAtomic(writePos_, std::memory_order_acquire);
        return wp < N ? wp : N;
    }
    // ★ Seqlock 方式の安全な読み出し
    template<typename F>
    void forEach(F&& callback) const noexcept {
        uint64_t wpBefore, wpAfter;
        size_t currentSize, startIdx;
        std::array<uint64_t, N> snapshot;
        do {
            wpBefore = convo::consumeAtomic(writePos_, std::memory_order_acquire);
            currentSize = (wpBefore < N) ? static_cast<size_t>(wpBefore) : N;
            startIdx = (wpBefore < N) ? 0 : static_cast<size_t>((wpBefore - N) % N);
            for (size_t i = 0; i < currentSize; ++i) {
                snapshot[i] = convo::consumeAtomic(data_[(startIdx + i) % N], std::memory_order_relaxed);
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            wpAfter = convo::consumeAtomic(writePos_, std::memory_order_relaxed);
        } while (wpBefore != wpAfter);
        for (size_t i = 0; i < currentSize; ++i) {
            const auto packed = snapshot[i];
            const auto reason = static_cast<ShutdownBlockingReason>(packed & 0xFF);
            const auto ts = packed >> 8;
            callback(reason, ts);
        }
    }
private:
    std::array<PackedBlockingEvent, N> data_{};
    std::atomic<uint64_t> writePos_{0};
};

// [work37 Phase 3.1] ShutdownResult — シャットダウン結果を構造化
struct ShutdownResult {
    bool completed{false};
    ShutdownPhase finalPhase{ShutdownPhase::ShutdownComplete};
    ISRHealthState healthState{static_cast<ISRHealthState>(0)};
    ShutdownBlockingReason blockingReason{ShutdownBlockingReason::None};
    uint64_t durationMs{0};
    uint32_t transitionViolations{0};
    uint32_t lateCallbackCount{0};
    uint32_t postStopEnqueueCount{0};
};

/**
 * Shutdown runtime FSM
 */
// ★ A-2: reasonToString — 独立関数として抽出
[[nodiscard]] const char* reasonToString(convo::isr::ShutdownBlockingReason reason) noexcept;

class ShutdownRuntime
{
public:
    ShutdownRuntime();
    ~ShutdownRuntime();

    // Initiate shutdown sequence
    void initiateShutdown();

    // Check current shutdown phase
    ShutdownPhase getPhase() const noexcept;

    // ★ P1-1: enum 順序非依存の terminal 判定
    static bool isTerminalPhase(ShutdownPhase p) noexcept {
        return p == ShutdownPhase::ShutdownComplete
            || p == ShutdownPhase::TimedOut
            || p == ShutdownPhase::Failed;
    }

    // ★ P1-1: TimedOut/Failed 上書き前の最終フェーズを取得（障害解析用）
    ShutdownPhase getLastNonTerminalPhase() const noexcept;

    // NonRT: advance shutdown phase
    void advancePhase() noexcept;
    bool transitionTo(ShutdownPhase target) noexcept;

    // RT: check if shutdown in progress
    bool isShutdownInProgress() const noexcept;

    // ★ P1-1: タイムアウト・異常終了を記録（transitionTo をバイパスして直接 store）
    void markTimedOut(ShutdownBlockingReason reason = ShutdownBlockingReason::Unknown) noexcept;
    void markFailed(ShutdownBlockingReason reason = ShutdownBlockingReason::Unknown) noexcept;

    // ★ P2-B: 完了阻害要因を取得（障害解析用）
    ShutdownBlockingReason getBlockingReason() const noexcept;

    // [work37 Phase 3.2] シャットダウン結果を収集する
    [[nodiscard]] ShutdownResult collectResult(ISRHealthState healthState,
                                                uint64_t startTimestampMs) const noexcept;

    // Emit final shutdown trace (work37: healthState を JSON に追加)
    void emitShutdownTrace(ISRHealthState healthState = static_cast<ISRHealthState>(0)) const;

    // Update bounded teardown counters (SH-1..SH-4)
    void setBoundedTeardownCounters(uint32_t callbackCount,
                                    uint32_t activeCrossfade,
                                    uint32_t pendingRetire,
                                    uint32_t observerCount) noexcept;

    // SH-5/SH-6: detect callbacks/enqueue after stop transition
    void markLateCallback() noexcept;
    void markPostStopEnqueue() noexcept;

private:
    // ★ A-2: シャットダウン開始時刻
    uint64_t shutdownStartUs_{0};

    // ★ A-2: ShutdownBlockingReason 別統計配列
    std::array<BlockingReasonStats, kBlockingReasonCount> blockingReasonStats_;

    // ★ A-3: Blocking Reason 時系列履歴リングバッファ (64エントリ)
    TinyRingBuffer<64> blockingReasonHistory_;

    std::atomic<ShutdownPhase> phase_{ShutdownPhase::Running};
    // ★ P1-1: TimedOut/Failed 上書き前の最終フェーズ（障害解析用）
    std::atomic<ShutdownPhase> lastNonTerminalPhase_{ShutdownPhase::Running};
    std::atomic<uint32_t> transitionViolations_{0};
    std::atomic<uint32_t> sh1CallbackCount_{0};
    std::atomic<uint32_t> sh2ActiveCrossfade_{0};
    std::atomic<uint32_t> sh3PendingRetire_{0};
    std::atomic<uint32_t> sh4ObserverCount_{0};
    std::atomic<uint32_t> sh5LateCallbackCount_{0};
    std::atomic<uint32_t> sh6PostStopEnqueueCount_{0};
    // ★ P2-B: Shutdown 完了阻害要因（markTimedOut/Failed 時に保存）
    std::atomic<ShutdownBlockingReason> blockingReason_{ShutdownBlockingReason::None};
};

}  // namespace isr
}  // namespace convo
