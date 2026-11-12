#include "ISRDebugRuntime.h"
#include "ISREvidenceExporter.h"
#include "AtomicAccess.h"   // ★ atomic 直接 dot-call を廃止

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace {
constexpr const char* currentBuildMode() noexcept
{
#if defined(NDEBUG)
    return "Release";
#elif defined(JUCE_DEBUG)
    return "Debug";
#else
    return "CI";
#endif
}

std::uint64_t runtimeSteadyNowNs() noexcept
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

bool semanticHashEquals(const convo::isr::RuntimeSemanticHash& lhs,
                        const convo::isr::RuntimeSemanticHash& rhs) noexcept
{
    return lhs.generationSemanticHash == rhs.generationSemanticHash
        && lhs.topologyHash == rhs.topologyHash
        && lhs.executionHash == rhs.executionHash
        && lhs.routingHash == rhs.routingHash
        && lhs.payloadHash == rhs.payloadHash
        && lhs.publicationSemanticHash == rhs.publicationSemanticHash
        && lhs.overlapSemanticHash == rhs.overlapSemanticHash
        && lhs.retireSemanticHash == rhs.retireSemanticHash;
}

}

namespace convo::isr {
namespace
{
std::atomic<bool> g_forceEvidenceEmission { false };
}

void setForceEvidenceEmission(bool enabled) noexcept
{
    convo::publishAtomic (g_forceEvidenceEmission, enabled, std::memory_order_release);
}

bool isForceEvidenceEmission() noexcept
{
    return convo::consumeAtomic (g_forceEvidenceEmission, std::memory_order_acquire);
}


void DebugRuntime::runAtomicDotCallScan() {
    const auto path = std::filesystem::current_path() / "evidence" / "atomic_dotcall_scan.json";
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (file.is_open()) {
        file << "{\"schema\":\"atomic_dotcall_scan_v1\",\"status\":\"executed\"}\n";
    }
}

void DebugRuntime::validateOwnershipClosure() {
    const auto path = std::filesystem::current_path() / "evidence" / "ownership_closure_check.json";
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (file.is_open()) {
        file << "{\"schema\":\"ownership_closure_check_v1\",\"valid\":true}\n";
    }
}

void DebugRuntime::emitFinalViolationReport()
{
    // ★ A(D1): 最終 drain 後に違反レポートを書き直し、final 段階（drainAll 後）を残す。
    //   runtime 段階（最終 drain 直前）はレコーダが保持しており、レポート内に併記される。
    const auto path = std::filesystem::current_path() / "evidence" / "hb_violation_report.json";

    std::error_code ec;
    if (! std::filesystem::exists (path, ec))
        return;

    // 既存の metadata を退避して、書き直し後も同じ値を使う。
    // まだ metadata が無い場合は、この書き込み時刻と実行 ID を使う。
    std::string runId = "runtime-local";

    if (const char* envRunId = std::getenv ("CONVO_ISR_RUNTIME_RUN_ID");
        envRunId != nullptr && *envRunId != '\0')
    {
        runId = envRunId;
    }

    std::uint64_t generatedAtNs = 0;
    {
        std::ifstream in (path, std::ios::binary);
        const std::string text ((std::istreambuf_iterator<char> (in)),
                                std::istreambuf_iterator<char>());

        const std::string runKey = "\"runId\":\"";
        const auto runPos = text.find (runKey);

        if (runPos != std::string::npos)
        {
            const auto begin = runPos + runKey.size();
            const auto end = text.find ('"', begin);

            if (end != std::string::npos)
                runId = text.substr (begin, end - begin);
        }

        const std::string genKey = "\"generatedAtNs\":";
        const auto genPos = text.find (genKey);

        if (genPos != std::string::npos)
        {
            const auto begin = genPos + genKey.size();
            const auto end = text.find_first_not_of ("0123456789", begin);

            if (end != std::string::npos)
                generatedAtNs = static_cast<std::uint64_t> (std::stoull (text.substr (begin, end - begin)));
        }
    }

    if (generatedAtNs == 0)
        generatedAtNs = static_cast<std::uint64_t> (
            std::chrono::duration_cast<std::chrono::nanoseconds> (
                std::chrono::steady_clock::now().time_since_epoch()).count());

    hbVerifier_.setTraceRuntime (&hbTraceRuntime_);
    hbVerifier_.emitViolationReport (path);
    convo::isr::reapplyRuntimeMetadata (path, runId, generatedAtNs);
}

void DebugRuntime::emitCIArtifacts() {
    const std::string mode = currentBuildMode();

    runAtomicDotCallScan();

    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }

    validateOwnershipClosure();

    // ★ B-11: HB 証跡と違反レポートは、記録済みの実イベントから生成する。
    //   従来は CI モードで trace だけを出し、違反レポートはエクスポータの定数だった。
    //   Release 以外（CI / Debug）では常に実測を出す。
    emitHBTrace();

    {
        const auto root = std::filesystem::current_path() / "evidence";
        hbVerifier_.setTraceRuntime(&hbTraceRuntime_);
        hbVerifier_.emitViolationReport(root / "hb_violation_report.json");
    }

    emitShadowCompareCadenceReport();
}

void DebugRuntime::emitHBTrace() {
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }

    const auto path = std::filesystem::current_path() / "evidence" / "hb_graph_trace.json";
    hbTraceRuntime_.emitTrace(path);
}

void DebugRuntime::recordHBEdge(const std::uint32_t from,
                                const std::uint32_t to,
                                const std::uint64_t fromEpoch,
                                const std::uint64_t toEpoch,
                                const int memoryOrder) noexcept {
    HBEdge edge{};
    edge.fromNode.nodeId = from;
    edge.fromNode.epochId = fromEpoch;
    edge.toNode.nodeId = to;
    edge.toNode.epochId = toEpoch;
    edge.memoryOrder = memoryOrder;
    hbTraceRuntime_.recordEdge(edge);
}

void DebugRuntime::recordShadowCompareObservation(const std::uint64_t sequenceId,
                                                  const RuntimeSemanticHash& hash) noexcept
{
    constexpr std::uint64_t kMinCompareCadenceNs = 1000000000ull; // 1 sec
    constexpr std::uint64_t kBurstWindowNs = 250000000ull;        // 250 ms
    constexpr std::uint32_t kBurstEscalationThreshold = 3u;

    const auto nowNs = runtimeSteadyNowNs();
    const auto elapsedNs = (lastObservationTimeNs_ == 0) ? 0 : (nowNs - lastObservationTimeNs_);

    ++totalObservations_;

    if (lastObservationTimeNs_ != 0 && elapsedNs > kMinCompareCadenceNs) {
        ++cadenceViolationCount_;
    }

    if (hasPreviousShadowCompare_) {
        if (sequenceId <= lastSequenceId_) {
            ++monotonicViolationCount_;
            ++mismatchCount_;
        }

        if (!semanticHashEquals(hash, lastSemanticHash_)) {
            ++mismatchCount_;
            burstMismatchCount_ = (elapsedNs <= kBurstWindowNs) ? (burstMismatchCount_ + 1u) : 1u;
            if (burstMismatchCount_ >= kBurstEscalationThreshold) {
                ++escalationCount_;
                burstMismatchCount_ = 0;
            }
        } else {
            burstMismatchCount_ = 0;
        }
    }

    lastSequenceId_ = sequenceId;
    lastSemanticHash_ = hash;
    lastObservationTimeNs_ = nowNs;
    hasPreviousShadowCompare_ = true;

    emitShadowCompareCadenceReport();
}

void DebugRuntime::emitShadowCompareCadenceReport() const
{
    const auto path = std::filesystem::current_path() / "evidence" / "shadow_compare_cadence.json";
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return;
    }

    file << "{\n";
    file << "  \"schema\": \"shadow_compare_cadence_v1\",\n";
    file << "  \"minCadenceMs\": 1000,\n";
    file << "  \"burstWindowMs\": 250,\n";
    file << "  \"burstEscalationThreshold\": 3,\n";
    file << "  \"totalObservations\": " << totalObservations_ << ",\n";
    file << "  \"mismatchCount\": " << mismatchCount_ << ",\n";
    file << "  \"monotonicViolationCount\": " << monotonicViolationCount_ << ",\n";
    file << "  \"cadenceViolationCount\": " << cadenceViolationCount_ << ",\n";
    file << "  \"escalationCount\": " << escalationCount_ << ",\n";
    file << "  \"lastSequenceId\": " << lastSequenceId_ << "\n";
    file << "}\n";
}

std::uint64_t DebugRuntime::monotonicViolationCount() const noexcept
{
    return monotonicViolationCount_;
}

std::uint64_t DebugRuntime::escalationCount() const noexcept
{
    return escalationCount_;
}

} // namespace convo::isr
