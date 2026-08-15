#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>  // ★ dash2 §2.2 (Phase A1): tryMakeQuiescenceProof の返り値
#include "AtomicAccess.h"  // ★ A-2/A-3: convo::consumeAtomic / publishAtomic
#include "RuntimeDrainAudit.h"  // ★ P2-B: ShutdownBlockingReason
#include "ISRLifetimeProof.h"   // ★ dash2 §2.2 (Phase A1): ShutdownQuiescenceProof / ReclaimPermit / ShutdownRuntimeIdentity

namespace convo {

// ★ P1-B: ISR Health State
enum class ISRHealthState : uint8_t;

namespace isr {

// ★ dash2 §2.2 (Phase A2 — Step 14): ReclaimAuthority の前方宣言。
//   ShutdownRuntime コンストラクタ（constructor 固定注入）/ Proof 生成時の bindShutdownIdentity に使用。
//   完全定義は ISRRuntimePublicationCoordinator.h（friend 宣言は Coordinator 側）。
class RuntimeIntentCoordinator;

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
    ActiveBuilder,   // ★ SHUTDOWN-7: Builder が Build Session 進行中（Shutdown 完了条件）
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
 * ★ dash2 §2.5 (H.11.4): AdmissionState 4-state FSM
 *   - Open:    新規 enqueue 許可
 *   - Closing: closeAdmission() 呼出し済み / producer join 中
 *   - Closed:  producer join 完了 / 新規 enqueue 拒否（不可逆）
 *   - Faulted: underflow 等の異常
 *   遷移: Open→Closing→Closed（✅）、Closed→Open（❌ 禁止 — resurrection 防止 INV-LIFE-9）、
 *         任意→Faulted（✅ underflow/overflow）。
 *   ［実装は Phase B3。production の 4 admission 経路 gate は本 FSM と
 *     CoordinatorState::ShuttingDown / isShutdownInProgress() を併用して統一する］
 */
enum class AdmissionState : uint8_t
{
    Open = 0,
    Closing,
    Closed,
    Faulted
};

/**
 * Shutdown runtime FSM
 */
// ★ A-2: reasonToString — 独立関数として抽出
[[nodiscard]] const char* reasonToString(convo::isr::ShutdownBlockingReason reason) noexcept;

class ShutdownRuntime
{
public:
    // ★ dash2 §2.2 (Phase A2 — Step 14 / Authority Singularization 完了):
    //   ReclaimAuthority（RuntimeIntentCoordinator）を constructor で固定注入する。
    //   ［AudioEngine は composition root として constructor initializer で依存を渡すのみ。
    //     setReclaimAuthority 等の public mutator は廃止 — association は immutable］
    explicit ShutdownRuntime(class RuntimeIntentCoordinator& reclaimAuthority) noexcept;
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

    // ── ★ dash2 §2.2 (Phase A2 — Step 14 / Authority Singularization) ──
    //   shutdown identity の binding authority は ShutdownRuntime（本クラス）のみ。
    //   AudioEngine は identity を bind しない（transport only）。
    //   ReclaimAuthority（RuntimeIntentCoordinator）は friend なので bindShutdownIdentity を呼べる。
    //   ［コメントと実装の責務境界を一致: setShutdownIdentity は AudioEngine から廃止し、
    //     ShutdownRuntime → ReclaimAuthority の単一 lifetime authority 経路に収束］

    // ★ dash2 §2.2 (Step 14 — Authority Singularization 完了): setReclaimAuthority は廃止。
    //   ReclaimAuthority の wiring は public mutator ではなく constructor 固定注入で行う
    //   （上記コンストラクタ参照）— 外部 caller による wiring 権限は構造的に存在しない。

    // ── ★ dash2 §2.2 (Phase A1 — H.11.17.5 15-Step, type only) ──
    //   ShutdownQuiescenceProof / ReclaimPermit は「型のみ・production 未接続」。
    //   production reclaim への接続は Phase A2（15-Step 7-15 / A2-G01〜G23 PASS）で行う。
    //   本メソッド群は ShutdownRuntime のみが生成する（INV-LIFE-3/4 / AC-X3-11）。

    // ── ★ dash2 §2.2 (Phase A2 — G10/G13/G19/G20): Quiescence 観測値 ──
    //   tryMakeQuiescenceProof が authority から取得・検証する観測値の束（H.11.11.3 Q0〜Q7）。
    //   ShutdownRuntime は EpochDomain を直接参照できないため、AudioEngine（EpochDomain 所有者）が
    //   proof 生成時に本構造体へ実測値を詰めて渡す（authority singularization — 観測は AudioEngine、
    //   検証・Proof 生成は ShutdownRuntime の責務分離）。
    struct QuiescenceObservation {
        // Q0: OutstandingAdmissionReservations == 0（第九者必須修正2 — close vs enqueue race を閉じる）
        bool admissionReservationsZero{false};
        // Q2: AllProducersJoined（requestShutdown 確定 + producer join 完了）
        bool allProducersJoined{false};
        // Q3: ReaderRegistrationClosed（EpochDomain::readerRegistrationClosed()）
        bool readerRegistrationClosed{false};
        // Q4: ActiveReaders == 0（ISRRetireRouter::activeReaderCount() == 0）
        bool activeReadersZero{false};
        // Q5: EpochSettled（EpochQuiescenceEvidence — H.11.13.3）
        bool epochSettled{false};
        // Q6: postStopEnqueueCount == 0（ShutdownRuntime 内部 sh6PostStopEnqueueCount_）
        bool postStopEnqueueZero{false};
        // Q7: NoResurrection（AdmissionState == Closed 固定）
        bool noResurrection{false};
        // ★ G19: epochGeneration（EpochDomain::epochGeneration()）
        uint64_t epochGeneration{0};
        // ★ G20: readerRegistrationGeneration（EpochDomain::readerRegistrationGeneration()）
        uint64_t readerRegistrationGeneration{0};
    };

    // Step 4 (拡張): Proof 生成 API（Q0〜Q7 全条件を observation から検証）。
    //   ⚠️ 簡易生成（if (isFullyDrained()) return Proof{};）は禁止（A2-G05）。
    //   全 Q 条件成立時のみ valid な Proof を返す（identity は shutdownGeneration + epoch/readerReg
    //   generation を束縛 — G17〜G20）。production 未接続（Phase A2 Step 9-15 で接続）。
    [[nodiscard]] std::optional<ShutdownQuiescenceProof>
    tryMakeQuiescenceProof(const QuiescenceObservation& observation) noexcept;

    // Step 3/5: ReclaimPermit 生成（Proof.identity と同一 identity で発行 — INV-LIFE-5/6）。
    //   Proof が valid でない場合は nullopt（簡易生成防止）。type only では未使用。
    [[nodiscard]] std::optional<ReclaimPermit> tryMakeReclaimPermit(const ShutdownQuiescenceProof&) noexcept;

    // ── ★ dash2 §2.5 (Phase B3 — H.11.4): AdmissionState FSM ──
    //   4 経路（Publication/Recovery/Build/Retire）の admission gate を本 FSM で統一する。
    //   - closeAdmission(): Open→Closing 遷移（requestShutdown 時に呼ばれる前提）
    //   - joinProducers():  Closing→Closed 遷移（producer join 完了後に呼ぶ — 不可逆）
    //   - isAdmissionOpen(): Open 判定（enqueue gate 用）
    //   - admissionState(): 現在状態（診断用）
    //   Closed→Open は存在しない（INV-LIFE-9 no-resurrection）。
    void closeAdmission() noexcept;
    void joinProducers() noexcept;
    [[nodiscard]] bool isAdmissionOpen() const noexcept;
    [[nodiscard]] AdmissionState admissionState() const noexcept;

    // ★ dash2 §2.2 (Phase A2 — Step 14 / Race B / T10): 現在の shutdown transaction generation。
    //   closeAdmission()（shutdown 開始）で確定し、その shutdown 中は固定。
    //   ReclaimPermit.identity().generation との照合（stale Permit 拒否 — AC-5）に使用。
    //   ［generation は Proof 生成ごとではなく shutdown transaction ごとに進む — 同一 shutdown 内の
    //     複数 reclaim で identity が安定する（H.11.11.9.3 Step 11 linearization と整合）］
    [[nodiscard]] uint64_t currentShutdownGeneration() const noexcept;

    // ★ dash2 §2.2 (Phase A2 — Step 14 / AUTH-09/13): Runtime インスタンス一意 ID。
    //   ShutdownRuntimeIdentity::engineInstanceId に束縛し、cross-runtime confusion
    //   （Runtime A/B の generation=1 が同一に見える問題）を防ぐ。
    [[nodiscard]] uint64_t engineInstanceId() const noexcept { return engineInstanceId_; }

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

    // ── ★ dash2 §2.2 (Phase A1 — type only) ──
    //   シャットダウン回数（単調増加）。ShutdownRuntimeIdentity の generation 源。
    //   tryMakeQuiescenceProof / tryMakeReclaimPermit の実装（Phase A2）で使用。
    std::atomic<uint64_t> shutdownGeneration_{0};

    // ── ★ dash2 §2.2 (Phase A2 — AUTH-09/13): Runtime インスタンス一意 ID。
    //   コンストラクタで g_shutdownRuntimeInstanceCounter から割り当てる。
    //   ShutdownRuntimeIdentity::engineInstanceId に束縛（cross-runtime confusion 防止）。
    uint64_t engineInstanceId_{0};

    // ── ★ dash2 §2.2 (Phase A2 — Step 14 / Authority Singularization 完了) ──
    //   ReclaimAuthority への reference member（constructor 固定注入 — 型レベルで null 不可・
    //   書き換え API 非存在・optional wiring / runtime reconfiguration 不在）。
    //   ShutdownRuntime が Proof 生成成功時に bindShutdownIdentity を呼ぶ（friend 経由）。
    //   ［AudioEngine は composition root として constructor initializer で依存を渡すのみ］
    class RuntimeIntentCoordinator& reclaimAuthority_;

    // ── ★ dash2 §2.5 (Phase B3 — H.11.4): AdmissionState FSM ──
    //   Open→Closing→Closed の不可逆遷移。Closed→Open 禁止（INV-LIFE-9）。
    std::atomic<AdmissionState> admissionState_{AdmissionState::Open};
};

}  // namespace isr
}  // namespace convo
