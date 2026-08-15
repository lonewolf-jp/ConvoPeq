#include "ISRShutdown.h"
#include "AtomicAccess.h"
#include "RuntimeDrainAudit.h"  // ★ P2-B: getPrimaryBlockingReason
#include "RuntimeHealthMonitor.h"  // ★ work37: ISRHealthState 完全型
#include "core/TimeUtils.h"  // ★ A-2: getCurrentTimeUs
#include "ISRRuntimePublicationCoordinator.h"  // ★ dash2 §2.2 (Step 14): RuntimeIntentCoordinator 完全型（bindShutdownIdentity 呼び出し用）

#include <filesystem>
#include <fstream>
#include <thread>  // ★ A-2: rename リトライ用 sleep_for

namespace convo {
namespace isr {

// ★ dash2 §2.2 (Phase A2 — AUTH-09/13): グローバルなインスタンス ID カウンタ。
//   ShutdownRuntime ごとに一意の engineInstanceId を割り当て、cross-runtime confusion を防ぐ。
namespace {
std::atomic<uint64_t> g_shutdownRuntimeInstanceCounter{0};
}

// ── ★ dash2 §2.2 (Phase A2 — Step 14 / Authority Singularization 完了) ──
//   ReclaimAuthority（RuntimeIntentCoordinator）を constructor で固定注入する。
//   ［AudioEngine は composition root として constructor initializer で依存を渡すのみ。
//     setReclaimAuthority 等の public mutator は廃止 — association は immutable（reference member）］
ShutdownRuntime::ShutdownRuntime(RuntimeIntentCoordinator& reclaimAuthority) noexcept
    : engineInstanceId_(g_shutdownRuntimeInstanceCounter.fetch_add(1, std::memory_order_relaxed) + 1)
    , reclaimAuthority_(reclaimAuthority)
{
}
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
        case ShutdownBlockingReason::ActiveBuilder: return "ActiveBuilder";
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
    convo::fetchAddAtomic(stats.count, uint64_t{1}, std::memory_order_acq_rel);

    // firstSeenUs: CAS で初回のみ設定
    uint64_t expected = 0;
    convo::compareExchangeAtomic(stats.firstSeenUs, expected, nowUs,
        std::memory_order_acq_rel, std::memory_order_acquire);

    // duration: shutdown 開始からの経過時間
    const uint64_t elapsed = (nowUs > shutdownStartUs_)
        ? (nowUs - shutdownStartUs_) : 0;

    // maxDurationUs: fetch_max (CAS loop)
    uint64_t currentMax = convo::consumeAtomic(stats.maxDurationUs, std::memory_order_acquire);
    while (elapsed > currentMax) {
        if (convo::compareExchangeAtomic(stats.maxDurationUs, currentMax, elapsed,
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
    case ShutdownBlockingReason::ActiveBuilder: reasonName = "ActiveBuilder"; break;  // ★ SHUTDOWN-7
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
        const auto count = convo::consumeAtomic(stats.count, std::memory_order_acquire);
        const auto maxDur = convo::consumeAtomic(stats.maxDurationUs, std::memory_order_acquire);
        const auto firstSeen = convo::consumeAtomic(stats.firstSeenUs, std::memory_order_acquire);
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

// ── ★ dash2 §2.2 (Phase A2 — G10/G13/G19/G20, H.11.11.3 Q0〜Q7) ──
//   Proof 生成 API。全 Q 条件を observation（AudioEngine が authority から収集）と
//   内部 FSM（AdmissionState / postStopEnqueue）から検証する。
//   ⚠️ 簡易生成（if (isFullyDrained()) return ...）は A2-G05 により禁止。
//   ［不変条件: INV-LIFE-3/4 — Proof / Permit は ShutdownRuntime のみ生成可能］
//   ［G19/G20: epochGeneration / readerRegistrationGeneration を identity に束縛 —
//     EpochDomain からの実供給（EpochDomain.h publishEpoch/closeReaderRegistration）を
//     AudioEngine が observation に詰めて渡す。型フィールドの存在だけでなく実データフローを確立］
std::optional<ShutdownQuiescenceProof>
ShutdownRuntime::tryMakeQuiescenceProof(const QuiescenceObservation& observation) noexcept
{
    // Q0〜Q7 を個別に観測（Q1/Q6 は ShutdownRuntime 内部、その他は observation）
    const bool q0 = observation.admissionReservationsZero;
    const bool q1 = (admissionState() == AdmissionState::Closed);
    const bool q2 = observation.allProducersJoined;
    const bool q3 = observation.readerRegistrationClosed;
    const bool q4 = observation.activeReadersZero;
    const bool q5 = observation.epochSettled;
    const bool q6 = observation.postStopEnqueueZero;
    const bool q7 = observation.noResurrection;

    // 全 Q 成立時のみ Proof 生成（quiescence 証明 — new obligation なし）
    if (!(q0 && q1 && q2 && q3 && q4 && q5 && q6 && q7))
        return std::nullopt;

    // ★ G17〜G20: shutdownGeneration + epoch/readerReg generation を identity に束縛。
    //   Shutdown N の Permit を Shutdown N+1 で使えないようにする（T10 / Permit ABA）。
    //   ★ dash2 §2.2 (Step 14 — Race B 修正): generation は closeAdmission()（shutdown 開始）で
    //   確定し、ここでは現在値をそのまま使う（fetch_add しない）。Proof 生成ごとに generation が
    //   進むと同一 shutdown 内の複数 reclaim で identity が不安定になり、正規の Permit まで
    //   stale 扱いになるため（H.11.11.9.3 Step 11 linearization と整合）。
    //   ★ dash2 §2.2 (AUTH-09/13): engineInstanceId も束縛（cross-runtime confusion 防止）。
    ShutdownRuntimeIdentity id;
    id.engineInstanceId = engineInstanceId_;
    id.generation = convo::consumeAtomic(shutdownGeneration_, std::memory_order_acquire);
    id.epochGeneration = observation.epochGeneration;
    id.readerRegistrationGeneration = observation.readerRegistrationGeneration;

    // ★ dash2 §2.2 (Step 14 — Authority Singularization 完了): binding authority は本クラス（ShutdownRuntime）。
    //   AudioEngine は bind に関与しない。ReclaimAuthority は friend なので bindShutdownIdentity を呼ぶ。
    //   ［reclaimAuthority_ は reference member（constructor 固定注入 — 型レベルで null 不可・
    //     optional wiring / runtime reconfiguration 不在）。Unbound → Bound(N) → 固定
    //     （再 bind は Coordinator 側で無視）］
    reclaimAuthority_.bindShutdownIdentity(id);

    ShutdownQuiescenceProof proof(id);
    proof.valid_ = true;
    proof.qAdmissionReservationsZero_ = q0;
    proof.qAdmissionClosed_ = q1;
    proof.qAllProducersJoined_ = q2;
    proof.qReaderRegClosed_ = q3;
    proof.qActiveReadersZero_ = q4;
    proof.qEpochSettled_ = q5;
    proof.qPostStopEnqueueZero_ = q6;
    proof.qNoResurrection_ = q7;
    return proof;
}

std::optional<ReclaimPermit> ShutdownRuntime::tryMakeReclaimPermit(
    const ShutdownQuiescenceProof& proof) noexcept
{
    // ★ dash2 §2.2 (G17〜G21): Proof.identity と同一 identity で Permit を発行（INV-LIFE-5/6）。
    //   - Proof が valid でない場合は nullopt（簡易生成防止）
    //   - Permit の consume() CAS が stale reject（G21）/ single-use（INV-LIFE-7 / T9）を担う
    //   - identity 束縛により Shutdown N の Permit が Shutdown N+1 で使えない（G18〜G20 / T10）
    if (!proof.valid())
        return std::nullopt;

    ReclaimPermit permit(proof.identity());
    return permit;
}

// ── ★ dash2 §2.5 (Phase B3 — H.11.4): AdmissionState FSM 実装 ──
//   Open→Closing→Closed の不可逆遷移（INV-LIFE-9: Closed→Open 禁止）。
void ShutdownRuntime::closeAdmission() noexcept
{
    AdmissionState expected = AdmissionState::Open;
    // Open のときのみ Closing へ（CAS で不可逆遷移を原子的に）
    if (convo::compareExchangeAtomic(admissionState_, expected, AdmissionState::Closing,
                                     std::memory_order_acq_rel, std::memory_order_acquire))
    {
        // ★ dash2 §2.2 (Step 14 — Race B / T10): shutdown transaction 開始を確定（generation 前進）。
        //   Proof 生成はこの確定済み generation を使うため、同一 shutdown 内の複数 reclaim で
        //   identity が安定し、Shutdown N+1 の begin で stale になる（H.11.11.9.3 Step 11）。
        //   ［Open→Closing 遷移成功時のみ increment — 二重開始防止（INV-LIFE-6）］
        (void)convo::fetchAddAtomic(shutdownGeneration_, static_cast<uint64_t>(1),
                                    std::memory_order_acq_rel);
    }
}

uint64_t ShutdownRuntime::currentShutdownGeneration() const noexcept
{
    return convo::consumeAtomic(shutdownGeneration_, std::memory_order_acquire);
}

void ShutdownRuntime::joinProducers() noexcept
{
    AdmissionState expected = AdmissionState::Closing;
    // Closing のときのみ Closed へ（producer join 完了を通知 — 不可逆）
    convo::compareExchangeAtomic(admissionState_, expected, AdmissionState::Closed,
                                 std::memory_order_acq_rel, std::memory_order_acquire);
}

bool ShutdownRuntime::isAdmissionOpen() const noexcept
{
    // Open のみ許可。Closing / Closed / Faulted では enqueue 拒否。
    return convo::consumeAtomic(admissionState_, std::memory_order_acquire) == AdmissionState::Open;
}

AdmissionState ShutdownRuntime::admissionState() const noexcept
{
    return convo::consumeAtomic(admissionState_, std::memory_order_acquire);
}

}  // namespace isr
}  // namespace convo
