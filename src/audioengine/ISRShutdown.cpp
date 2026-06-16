#include "ISRShutdown.h"
#include "AtomicAccess.h"
#include "RuntimeDrainAudit.h"  // ★ P2-B: getPrimaryBlockingReason
#include "RuntimeHealthMonitor.h"  // ★ work37: ISRHealthState 完全型
#include "core/TimeUtils.h"  // ★ A-2: getCurrentTimeUs

#include <filesystem>
#include <fstream>
#include <thread>  // ★ A-2: rename リトライ用 sleep_for

namespace convo {
namespace isr {

ShutdownRuntime::ShutdownRuntime() = default;
ShutdownRuntime::~ShutdownRuntime() = default;

// ★ A-2: reasonToString 実装
const char* reasonToString(ShutdownBlockingReason reason) noexcept {
    switch (reason) {
        case ShutdownBlockingReason::None: return "None";
        case ShutdownBlockingReason::PendingPublication: return "PendingPublication";
        case ShutdownBlockingReason::PendingRetire: return "PendingRetire";
        case ShutdownBlockingReason::ActiveCrossfade: return "ActiveCrossfade";
        case ShutdownBlockingReason::DeferredPublish: return "DeferredPublish";
        case ShutdownBlockingReason::QuarantineResident: return "QuarantineResident";
        case ShutdownBlockingReason::RouterPendingRetire: return "RouterPendingRetire";
        case ShutdownBlockingReason::ReaderActive: return "ReaderActive";
        case ShutdownBlockingReason::Unknown: return "Unknown";
    }
    return "Unknown";
}

void ShutdownRuntime::initiateShutdown()
{
    // ★ A-2: シャットダウン開始時刻を記録
    shutdownStartUs_ = convo::getCurrentTimeUs();
    transitionTo(ShutdownPhase::AudioStopped);
}

ShutdownPhase ShutdownRuntime::getPhase() const noexcept
{
    return convo::consumeAtomic(phase_, std::memory_order_acquire);
}

ShutdownPhase ShutdownRuntime::getLastNonTerminalPhase() const noexcept
{
    return convo::consumeAtomic(lastNonTerminalPhase_, std::memory_order_acquire);
}

ShutdownBlockingReason ShutdownRuntime::getBlockingReason() const noexcept
{
    return convo::consumeAtomic(blockingReason_, std::memory_order_acquire);
}

void ShutdownRuntime::markTimedOut(ShutdownBlockingReason reason) noexcept
{
    const uint64_t nowUs = convo::getCurrentTimeUs();

    // ★ A-3: 時系列履歴に追加
    blockingReasonHistory_.push(reason, nowUs);

    // ★ A-2: 統計更新
    // 配列外参照防止: enum 値をサニタイズ
    size_t idx = static_cast<size_t>(reason);
    if (idx >= kBlockingReasonCount) {
        idx = static_cast<size_t>(ShutdownBlockingReason::Unknown);
    }
    auto& stats = blockingReasonStats_[idx];
    stats.count.fetch_add(1, std::memory_order_acq_rel);

    // firstSeenUs: CAS で初回のみ設定
    uint64_t expected = 0;
    stats.firstSeenUs.compare_exchange_strong(expected, nowUs,
        std::memory_order_acq_rel, std::memory_order_acquire);

    // duration: shutdown 開始からの経過時間
    const uint64_t elapsed = (nowUs > shutdownStartUs_)
        ? (nowUs - shutdownStartUs_) : 0;

    // maxDurationUs: fetch_max (CAS loop)
    uint64_t currentMax = stats.maxDurationUs.load(std::memory_order_acquire);
    while (elapsed > currentMax) {
        if (stats.maxDurationUs.compare_exchange_weak(currentMax, elapsed,
                std::memory_order_acq_rel, std::memory_order_acquire))
            break;
    }

    // ★ P2-B: 阻害要因を保存
    convo::publishAtomic(blockingReason_, reason, std::memory_order_release);
    // ★ P1-1: 現在の phase を保存してから上書き
    convo::publishAtomic(lastNonTerminalPhase_,
                         convo::consumeAtomic(phase_, std::memory_order_acquire),
                         std::memory_order_release);
    convo::publishAtomic(phase_, ShutdownPhase::TimedOut, std::memory_order_release);
}

void ShutdownRuntime::markFailed(ShutdownBlockingReason reason) noexcept
{
    // ★ P2-B: 阻害要因を保存
    convo::publishAtomic(blockingReason_, reason, std::memory_order_release);
    convo::publishAtomic(lastNonTerminalPhase_,
                         convo::consumeAtomic(phase_, std::memory_order_acquire),
                         std::memory_order_release);
    convo::publishAtomic(phase_, ShutdownPhase::Failed, std::memory_order_release);
}

void ShutdownRuntime::advancePhase() noexcept
{
    const ShutdownPhase current = convo::consumeAtomic(phase_, std::memory_order_acquire);

    // ★ P1-1: terminal 状態からは advance しない
    if (isTerminalPhase(current))
        return;

    ShutdownPhase next = current;
    switch (current) {
        case ShutdownPhase::Running:
            next = ShutdownPhase::AudioStopped;
            break;
        case ShutdownPhase::AudioStopped:
            next = ShutdownPhase::ObserverDrained;
            break;
        case ShutdownPhase::ObserverDrained:
            next = ShutdownPhase::RetireClosed;
            break;
        case ShutdownPhase::RetireClosed:
            next = ShutdownPhase::EpochSettled;
            break;
        case ShutdownPhase::EpochSettled:
            next = ShutdownPhase::ReclaimComplete;
            break;
        case ShutdownPhase::ReclaimComplete:
            // ★ C-2: CONVOPEQ_EMERGENCY_DRAIN 有効時のみ EmergencyDrain を経由
            next = ShutdownPhase::VerifyDrained;
            break;
        case ShutdownPhase::EmergencyDrain:        // ★ C-2
            next = ShutdownPhase::VerifyDrained;
            break;
        case ShutdownPhase::VerifyDrained:
            next = ShutdownPhase::ShutdownComplete;
            break;
        case ShutdownPhase::TimedOut:
        case ShutdownPhase::Failed:
        case ShutdownPhase::ShutdownComplete:
        default:
            return;
    }

    (void)transitionTo(next);
}

bool ShutdownRuntime::transitionTo(ShutdownPhase target) noexcept
{
    const auto current = convo::consumeAtomic(phase_, std::memory_order_acquire);
    const auto c = static_cast<int>(current);
    const auto t = static_cast<int>(target);

    // ★ P1-1: TimedOut(6)/Failed(7) を ShutdownComplete(8) の前に挿入したため、
    //   ReclaimComplete(5)→ShutdownComplete(8) のような terminal 状態をスキップする
    //   遷移を許可する。terminal 状態のみをスキップする遷移は許容。
    bool allowed = (t == c || t == c + 1);
    if (!allowed && t > c + 1) {
        // terminal 状態のみをスキップしているか確認
        allowed = true;
        for (int i = c + 1; i < t; ++i) {
            if (!isTerminalPhase(static_cast<ShutdownPhase>(i))) {
                allowed = false;
                break;
            }
        }
    }

    if (!allowed) {
        (void)convo::fetchAddAtomic(transitionViolations_, uint32_t{1}, std::memory_order_acq_rel);
        return false;
    }

    convo::publishAtomic(phase_, target, std::memory_order_release);
    return true;
}

bool ShutdownRuntime::isShutdownInProgress() const noexcept
{
    const ShutdownPhase current = convo::consumeAtomic(phase_, std::memory_order_acquire);
    return current != ShutdownPhase::Running && !isTerminalPhase(current);
}

// [work37 Phase 3.2] collectResult — シャットダウン結果を収集
ShutdownResult ShutdownRuntime::collectResult(
    ISRHealthState healthState, uint64_t startTimestampMs) const noexcept
{
    const auto nowMs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    ShutdownResult result;
    result.completed = (convo::consumeAtomic(phase_, std::memory_order_acquire)
                        == ShutdownPhase::ShutdownComplete);
    result.finalPhase = convo::consumeAtomic(phase_, std::memory_order_acquire);
    result.healthState = healthState;
    result.blockingReason = convo::consumeAtomic(blockingReason_, std::memory_order_acquire);
    result.durationMs = (nowMs > startTimestampMs) ? (nowMs - startTimestampMs) : 0;
    result.transitionViolations = convo::consumeAtomic(transitionViolations_,
                                                       std::memory_order_acquire);
    result.lateCallbackCount = convo::consumeAtomic(sh5LateCallbackCount_,
                                                    std::memory_order_acquire);
    result.postStopEnqueueCount = convo::consumeAtomic(sh6PostStopEnqueueCount_,
                                                       std::memory_order_acquire);
    return result;
}

// [work37 Phase 3.3] healthState を JSON に追加
void ShutdownRuntime::emitShutdownTrace(ISRHealthState healthState) const
{
    // ★ ★ A-2: アトミックファイル置換: .tmp に書き込み後 rename
    const auto outputPath = std::filesystem::current_path() / "evidence" / "shutdown_trace.json";
    const auto tmpPath = std::filesystem::current_path() / "evidence" / "shutdown_trace.json.tmp";
    std::error_code ec;
    std::filesystem::create_directories(outputPath.parent_path(), ec);
    if (ec) return;

    std::ofstream file(tmpPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        // ★ フォールバック: 一意化したファイル名で %TEMP% に書き込み
        static std::atomic<uint32_t> s_fallbackCounter{0};
        const auto timestamp = convo::getCurrentTimeUs();
        const auto count = s_fallbackCounter.fetch_add(1, std::memory_order_relaxed);
        const auto fallbackName = std::string("shutdown_trace_fallback_")
            + std::to_string(timestamp) + "_" + std::to_string(count) + ".json";
        std::error_code ec2;
        const auto tempDir = std::filesystem::temp_directory_path(ec2);
        if (ec2) return;
        const auto fallbackPath = tempDir / fallbackName;
        file.open(fallbackPath, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) return;
    }

    const auto phase = convo::consumeAtomic(phase_, std::memory_order_acquire);
    const auto violations = convo::consumeAtomic(transitionViolations_, std::memory_order_acquire);
    const auto sh1 = convo::consumeAtomic(sh1CallbackCount_, std::memory_order_acquire);
    const auto sh2 = convo::consumeAtomic(sh2ActiveCrossfade_, std::memory_order_acquire);
    const auto sh3 = convo::consumeAtomic(sh3PendingRetire_, std::memory_order_acquire);
    const auto sh4 = convo::consumeAtomic(sh4ObserverCount_, std::memory_order_acquire);
    const auto sh5 = convo::consumeAtomic(sh5LateCallbackCount_, std::memory_order_acquire);
    const auto sh6 = convo::consumeAtomic(sh6PostStopEnqueueCount_, std::memory_order_acquire);
    const auto reason = convo::consumeAtomic(blockingReason_, std::memory_order_acquire);  // ★ P2-B

    const char* phaseName = "Running";
    switch (phase) {
    case ShutdownPhase::Running: phaseName = "Running"; break;
    case ShutdownPhase::AudioStopped: phaseName = "AudioStopped"; break;
    case ShutdownPhase::ObserverDrained: phaseName = "ObserverDrained"; break;
    case ShutdownPhase::RetireClosed: phaseName = "RetireClosed"; break;
    case ShutdownPhase::EpochSettled: phaseName = "EpochSettled"; break;
    case ShutdownPhase::ReclaimComplete: phaseName = "ReclaimComplete"; break;
    case ShutdownPhase::EmergencyDrain: phaseName = "EmergencyDrain"; break;  // ★ C-2
    case ShutdownPhase::VerifyDrained: phaseName = "VerifyDrained"; break;
    case ShutdownPhase::TimedOut: phaseName = "TimedOut"; break;
    case ShutdownPhase::Failed: phaseName = "Failed"; break;
    case ShutdownPhase::ShutdownComplete: phaseName = "ShutdownComplete"; break;
    }

    const char* reasonName = "None";
    switch (reason) {
    case ShutdownBlockingReason::None: reasonName = "None"; break;
    case ShutdownBlockingReason::PendingPublication: reasonName = "PendingPublication"; break;
    case ShutdownBlockingReason::PendingRetire: reasonName = "PendingRetire"; break;
    case ShutdownBlockingReason::ActiveCrossfade: reasonName = "ActiveCrossfade"; break;
    case ShutdownBlockingReason::DeferredPublish: reasonName = "DeferredPublish"; break;
    case ShutdownBlockingReason::QuarantineResident: reasonName = "QuarantineResident"; break;
    case ShutdownBlockingReason::RouterPendingRetire: reasonName = "RouterPendingRetire"; break;
    case ShutdownBlockingReason::ReaderActive: reasonName = "ReaderActive"; break;
    case ShutdownBlockingReason::Unknown: reasonName = "Unknown"; break;
    }

    const bool boundedComplete = (sh1 == 0u && sh2 == 0u && sh3 == 0u && sh4 == 0u && sh5 == 0u && sh6 == 0u);

    // [work37 Phase 3.3] healthState を JSON に追加
    const char* healthStateName = "Unknown";
    switch (healthState) {
        case static_cast<ISRHealthState>(0): healthStateName = "Healthy"; break;
        case static_cast<ISRHealthState>(1): healthStateName = "Degraded"; break;
        case static_cast<ISRHealthState>(2): healthStateName = "Critical"; break;
        default: break;
    }

    file << "{\n";
    file << "  \"schema\": \"shutdown_trace_v4\",\n";
    file << "  \"phase\": " << static_cast<int>(phase) << ",\n";
    file << "  \"phaseName\": \"" << phaseName << "\",\n";
    file << "  \"healthState\": " << static_cast<int>(healthState) << ",\n";
    file << "  \"healthStateName\": \"" << healthStateName << "\",\n";
    file << "  \"blockingReason\": \"" << reasonName << "\",\n";  // ★ P2-B
    file << "  \"blockingReasonCode\": " << static_cast<int>(reason) << ",\n";
    file << "  \"transitionViolations\": " << violations << ",\n";
    file << "  \"sh1_callbackCount\": " << sh1 << ",\n";
    file << "  \"sh2_activeCrossfade\": " << sh2 << ",\n";
    file << "  \"sh3_pendingRetire\": " << sh3 << ",\n";
    file << "  \"sh4_observerCount\": " << sh4 << ",\n";
    file << "  \"sh5_lateCallbackCount\": " << sh5 << ",\n";
    file << "  \"sh6_postStopEnqueueCount\": " << sh6 << ",\n";

    // ★ A-2: BlockingReasonStats JSON出力
    file << "  \"blockingReasonStats\": [\n";
    for (size_t i = 0; i < kBlockingReasonCount; ++i) {
        const auto& stats = blockingReasonStats_[i];
        const auto count = stats.count.load(std::memory_order_acquire);
        const auto maxDur = stats.maxDurationUs.load(std::memory_order_acquire);
        const auto firstSeen = stats.firstSeenUs.load(std::memory_order_acquire);
        if (i > 0) file << ",\n";
        file << "    {\n";
        file << "      \"reason\": \"" << convo::isr::reasonToString(static_cast<ShutdownBlockingReason>(i)) << "\",\n";
        file << "      \"count\": " << count << ",\n";
        file << "      \"maxDurationUs\": " << maxDur << ",\n";
        file << "      \"firstSeenUs\": " << firstSeen << "\n";
        file << "    }";
    }
    file << "\n  ],\n";

    file << "  \"verified\": " << ((violations == 0 && boundedComplete) ? "true" : "false") << "\n";
    file << "}\n";

    file.close();
    // ★ ★ 書き込みエラー検出: ディスクフルや権限エラーは close 後にも fail になる
    if (file.fail()) return;

    // ★ rename リトライ: 最大3回、100ms 間隔（Windows ファイルロック対策）
    constexpr int kMaxRenameRetries = 3;
    constexpr auto kRenameRetryInterval = std::chrono::milliseconds(100);
    for (int retry = 0; retry < kMaxRenameRetries; ++retry) {
        std::filesystem::rename(tmpPath, outputPath, ec);
        if (!ec) break;  // 成功
        if (retry < kMaxRenameRetries - 1) {
            std::this_thread::sleep_for(kRenameRetryInterval);
        }
    }
    // ★ 全リトライ失敗時は別名で保存
    if (ec) {
        static std::atomic<uint32_t> s_renameFallbackCounter{0};
        const auto altPath = std::filesystem::current_path() / "evidence"
            / ("shutdown_trace_" + std::to_string(
                s_renameFallbackCounter.fetch_add(1, std::memory_order_relaxed)) + ".json");
        std::filesystem::rename(tmpPath, altPath, ec);
    }
    // 前回の .tmp が残存していれば削除
    std::filesystem::remove(tmpPath, ec);
}

void ShutdownRuntime::setBoundedTeardownCounters(uint32_t callbackCount,
                                                 uint32_t activeCrossfade,
                                                 uint32_t pendingRetire,
                                                 uint32_t observerCount) noexcept
{
    convo::publishAtomic(sh1CallbackCount_, callbackCount, std::memory_order_release);
    convo::publishAtomic(sh2ActiveCrossfade_, activeCrossfade, std::memory_order_release);
    convo::publishAtomic(sh3PendingRetire_, pendingRetire, std::memory_order_release);
    convo::publishAtomic(sh4ObserverCount_, observerCount, std::memory_order_release);
}

void ShutdownRuntime::markLateCallback() noexcept
{
    (void)convo::fetchAddAtomic(sh5LateCallbackCount_, uint32_t{1}, std::memory_order_acq_rel);
}

void ShutdownRuntime::markPostStopEnqueue() noexcept
{
    (void)convo::fetchAddAtomic(sh6PostStopEnqueueCount_, uint32_t{1}, std::memory_order_acq_rel);
}

}  // namespace isr
}  // namespace convo
