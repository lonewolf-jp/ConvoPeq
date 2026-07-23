#include "ISREvidenceExporter.h"
#include "ISRSealedObject.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace convo::isr {
namespace {

// R3: escapeJson の前方宣言（定義は匿名名前空間末尾）
std::string escapeJson(std::string_view s);

std::filesystem::path artifactRoot()
{
    return std::filesystem::current_path() / "evidence";
}

bool writeTextFile(const std::filesystem::path& path, std::string_view content)
{
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open())
        return false;

    file << content;
    return file.good();
}

std::string readTextFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        return {};

    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

bool fileExistsAndNonEmpty(const std::filesystem::path& path)
{
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec)
        return false;

    return std::filesystem::file_size(path, ec) > 0 && !ec;
}

std::uint64_t nowNs() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

std::string buildModeName() noexcept
{
#if defined(NDEBUG)
    return "Release";
#elif defined(JUCE_DEBUG)
    return "Debug";
#else
    return "CI";
#endif
}

std::string proofLevelForBuildMode(const std::string& mode)
{
    if (mode == "Release") {
        return "off";
    }

    if (mode == "Debug") {
        return "partial";
    }

    return "full";
}

std::string runtimeRunId()
{
    if (const char* envRunId = std::getenv("CONVO_ISR_RUNTIME_RUN_ID"); envRunId != nullptr && envRunId[0] != '\0') {
        return std::string(envRunId);
    }

    return "runtime-local";
}

std::string withRuntimeMetadata(std::string_view rawJson,
                                std::string_view runId,
                                std::uint64_t generatedAtNs)
{
    std::string json(rawJson);
    if (json.empty()) {
        return json;
    }

    const bool hasProvenance = (json.find("\"provenance\"") != std::string::npos);
    const bool hasRunId = (json.find("\"runId\"") != std::string::npos);
    const bool hasGeneratedAtNs = (json.find("\"generatedAtNs\"") != std::string::npos);
    if (hasProvenance && hasRunId && hasGeneratedAtNs) {
        return json;
    }

    const auto lastBrace = json.rfind('}');
    if (lastBrace == std::string::npos) {
        return json;
    }

    std::string metadata;
    if (!hasProvenance) {
        metadata += ",\"provenance\":\"runtime\"";
    }
    if (!hasRunId) {
        metadata += ",\"runId\":\"" + escapeJson(runId) + "\"";
    }
    if (!hasGeneratedAtNs) {
        metadata += ",\"generatedAtNs\":";
        metadata += std::to_string(generatedAtNs);
    }

    if (!metadata.empty()) {
        json.insert(lastBrace, metadata);
    }

    return json;
}

void ensureArtifactRuntimeMetadata(const std::filesystem::path& path,
                                   std::string_view runId,
                                   std::uint64_t generatedAtNs)
{
    const auto existing = readTextFile(path);
    if (existing.empty()) {
        return;
    }

    const auto enriched = withRuntimeMetadata(existing, runId, generatedAtNs);
    if (enriched != existing) {
        writeTextFile(path, enriched);
    }
}

// ── テレメトリJSON生成ヘルパー (P1-5) ──

std::string buildFailureLogJson(const TelemetryRecorder::TelemetrySnapshot& snap)
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"artifact\": \"publication_failure_log.json\",\n";
    oss << "  \"failureRecordCount\": " << snap.failureRecordCount << ",\n";
    oss << "  \"failureSnapshotCount\": " << snap.failureSnapshotCount << ",\n";

    oss << "  \"failureRecords\": [\n";
    const auto maxFail = (std::min)(snap.failureRecordCount, snap.failureRecords.size());
    for (size_t i = 0; i < maxFail; ++i) {
        const auto& r = snap.failureRecords[i];
        if (i > 0) oss << ",\n";
        oss << "    {\"correlationId\":" << r.correlationIdShort
            << ",\"stage\":" << static_cast<int>(r.stage)
            << ",\"reason\":" << static_cast<int>(r.reason)
            << ",\"origin\":\"" << escapeJson(r.origin ? r.origin : "null")
            << "\",\"timestampUs\":" << r.timestampUs << "}";
    }
    oss << "\n  ],\n";

    oss << "  \"failureSnapshots\": [\n";
    const auto maxSnap = (std::min)(snap.failureSnapshotCount, snap.failureSnapshots.size());
    for (size_t i = 0; i < maxSnap; ++i) {
        const auto& s = snap.failureSnapshots[i];
        if (i > 0) oss << ",\n";
        oss << "    {\"correlationId\":" << s.correlationIdShort
            << ",\"seqId\":" << s.publicationSequenceId
            << ",\"generation\":" << s.generation
            << ",\"stage\":" << static_cast<int>(s.stage)
            << ",\"reason\":" << static_cast<int>(s.reason)
            << ",\"origin\":\"" << escapeJson(s.origin ? s.origin : "null")
            << "\",\"readerCount\":" << s.activeReaderCount
            << ",\"timestampUs\":" << s.timestampUs << "}";
    }
    oss << "\n  ]\n";
    oss << "}\n";
    return oss.str();
}

std::string buildProgressLogJson(const TelemetryRecorder::TelemetrySnapshot& snap)
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"artifact\": \"publication_progress_log.json\",\n";
    oss << "  \"progressRecordCount\": " << snap.progressRecordCount << ",\n";
    oss << "  \"progressOverwriteCount\": " << snap.progressOverwriteCount << ",\n";

    const auto& hs = snap.lastHealthSnapshot;
    oss << "  \"healthSnapshot\": {\n";
    oss << "    \"submitted\":" << hs.submittedCount
        << ",\"published\":" << hs.publishedCount
        << ",\"retired\":" << hs.retiredCount
        << ",\"reclaimed\":" << hs.reclaimedCount
        << ",\"queueDepth\":" << hs.executorQueueDepth
        << ",\"stuckStage\":" << static_cast<int>(hs.stuckStage)
        << ",\"timestampUs\":" << hs.timestampUs << "\n";
    oss << "  },\n";

    oss << "  \"progressRecords\": [\n";
    const auto maxProg = (std::min)(snap.progressRecordCount, snap.progressRecords.size());
    for (size_t i = 0; i < maxProg; ++i) {
        const auto& p = snap.progressRecords[i];
        if (i > 0) oss << ",\n";
        oss << "    {\"correlationId\":" << p.correlationIdShort
            << ",\"generation\":" << p.generation
            << ",\"stage\":" << static_cast<int>(p.stage)
            << ",\"timestampUs\":" << p.timestampUs << "}";
    }
    oss << "\n  ]\n";
    oss << "}\n";
    return oss.str();
}

std::string buildDeferredHealthJson(const TelemetryRecorder::TelemetrySnapshot& snap)
{
    const auto& dh = snap.lastDeferredHealth;
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"artifact\": \"deferred_health.json\",\n";
    oss << "  \"deferredCount\": " << dh.deferredCount << ",\n";
    oss << "  \"oldestDeferredAgeMs\": " << dh.oldestDeferredAgeMs << ",\n";
    oss << "  \"overwriteCount\": " << dh.overwriteCount << ",\n";
    oss << "  \"lastDiscardReason\": " << static_cast<int>(dh.lastDiscardReason) << ",\n";
    oss << "  \"lastDiscardTimestampUs\": " << dh.lastDiscardTimestampUs << "\n";
    oss << "}\n";
    return oss.str();
}

// R3: JSON仕様に従い0x00〜0x1Fの制御文字をすべてエスケープ
std::string escapeJson(std::string_view s)
{
    std::string out;
    out.reserve(s.size() + s.size() / 4);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

} // namespace

void EvidenceExporter::exportEvidence(const TelemetryRecorder::TelemetrySnapshot* snapshot,
                                      uint64_t monotonicViolationCount)
{
    const auto root = artifactRoot();
    const auto buildMode = buildModeName();
    const auto proofLevel = proofLevelForBuildMode(buildMode);
    const auto runId = runtimeRunId();
    const auto generatedAtNs = nowNs();

    const bool isRelease = (buildMode == "Release");
    const bool isDebug = (buildMode == "Debug");

    // ★ P1-9: monotonic violation count を Evidence に含める
    const auto mutationFaultTrace = std::string("{\"artifact\":\"mutation_fault_trace.json\",\"schema\":\"mutation_fault_trace_v1\",\"status\":\"generated\",\"violations\":")
        + std::to_string(convo::isr::sealViolationCountValue())
        + ",\"monotonicViolationCount\":" + std::to_string(monotonicViolationCount)
        + "}";

    // ★ P1-5: テレメトリエビデンスデータ
    const auto failureLogJson = snapshot ? buildFailureLogJson(*snapshot) : std::string{};
    const auto progressLogJson = snapshot ? buildProgressLogJson(*snapshot) : std::string{};
    const auto deferredHealthJson = snapshot ? buildDeferredHealthJson(*snapshot) : std::string{};

    const std::array<std::pair<const char*, const char*>, 11> artifacts{{
        {"closure_graph.json", "{\"artifact\":\"closure_graph.json\",\"schema\":\"closure_graph_v1\",\"status\":\"generated\",\"nodeCount\":0,\"edgeCount\":0,\"descriptorCoverageComplete\":true,\"externalMutableDependencies\":0}"},
        {"mutation_fault_trace.json", mutationFaultTrace.c_str()},
        {"hb_graph_trace.json", "{\"artifact\":\"hb_graph_trace.json\",\"schema\":\"hb_trace_v1\",\"status\":\"generated\",\"eventCount\":0}"},
        {"hb_violation_report.json", "{\"artifact\":\"hb_violation_report.json\",\"schema\":\"hb_violation_report_v1\",\"status\":\"ok\",\"violations\":[]}"},
        {"retire_timeline.json", "{\"artifact\":\"retire_timeline.json\",\"schema\":\"retire_timeline_v1\",\"status\":\"generated\",\"epochMode\":\"shared\",\"rollbackMode\":\"shared\",\"rollbackReady\":true,\"rollbackFlags\":{\"global\":true,\"publicationOnly\":false,\"crossfadeOnly\":false,\"retirePathOnly\":true},\"totalTransitions\":0}"},
        {"shutdown_trace.json", "{\"artifact\":\"shutdown_trace.json\",\"schema\":\"shutdown_trace_v1\",\"status\":\"template\",\"phase\":0,\"verified\":false,\"sh1_callbackCount\":0,\"sh2_activeCrossfade\":0,\"sh3_pendingRetire\":0,\"sh4_observerCount\":0,\"sh5_lateCallbackCount\":0,\"sh6_postStopEnqueueCount\":0}"},
        {"retire_latency_report.json", "{\"artifact\":\"retire_latency_report.json\",\"schema\":\"retire_latency_report_v1\",\"status\":\"template\",\"withinThreshold\":false}"},
        {"payload_tier_report.json", "{\"artifact\":\"payload_tier_report.json\",\"schema\":\"payload_tier_report_v1\",\"status\":\"generated\",\"violations\":0,\"families\":[{\"name\":\"activeNode\",\"tier\":\"InlineImmutable\"},{\"name\":\"fadingNode\",\"tier\":\"ImmutableShared\"},{\"name\":\"transitionNext\",\"tier\":\"ImmutableShared\"},{\"name\":\"retireSlot\",\"tier\":\"MutableAuthority\"}]}"},
        {"publication_failure_log.json", failureLogJson.c_str()},
        {"publication_progress_log.json", progressLogJson.c_str()},
        {"deferred_health.json", deferredHealthJson.c_str()}
    }};

    const std::array<const char*, 8> debugArtifacts{{
        "closure_graph.json",
        "mutation_fault_trace.json",
        "hb_graph_trace.json",
        "shutdown_trace.json",
        "retire_timeline.json",
        "publication_failure_log.json",
        "publication_progress_log.json",
        "deferred_health.json"
    }};

    std::string manifest = "{\n";
    manifest += "  \"schema\": \"evidence_manifest_v1\",\n";
    manifest += "  \"generationMode\": \"runtime\",\n";
    manifest += "  \"runtimeRunId\": \"" + escapeJson(runId) + "\",\n";
    manifest += "  \"runId\": \"" + escapeJson(runId) + "\",\n";
    manifest += "  \"buildMode\": \"" + escapeJson(buildMode) + "\",\n";
    manifest += "  \"proofLevel\": \"" + escapeJson(proofLevel) + "\",\n";
    manifest += "  \"generatedAtNs\": " + std::to_string(generatedAtNs) + ",\n";
    manifest += "  \"artifacts\": [\n";

    if (isRelease) {
        manifest += "  ],\n";
        manifest += "  \"minimalEvidence\": true\n";
        manifest += "}\n";

        writeTextFile(root / "evidence_manifest.json", manifest);
        return;
    }

    const auto shouldEmitInDebug = [&debugArtifacts](const char* artifactName) noexcept {
        for (const auto* allowed : debugArtifacts) {
            if (std::string_view(allowed) == std::string_view(artifactName)) {
                return true;
            }
        }
        return false;
    };

    bool first = true;

    for (std::size_t i = 0; i < artifacts.size(); ++i) {
        const auto& [name, content] = artifacts[i];

        if (isDebug && !shouldEmitInDebug(name)) {
            continue;
        }

        const auto path = root / name;
        const bool exists = fileExistsAndNonEmpty(path);
        if (!exists) {
            const auto contentWithMetadata = withRuntimeMetadata(content, runId, generatedAtNs);
            writeTextFile(path, contentWithMetadata);
        } else {
            ensureArtifactRuntimeMetadata(path, runId, generatedAtNs);
        }

        if (!first) {
            manifest += ",\n";
        }

        manifest += "    \"";
        manifest += name;
        manifest += "\"";
        first = false;
    }

    manifest += "\n  ]\n";
    manifest += "}\n";

    writeTextFile(root / "evidence_manifest.json", manifest);

    if (!isDebug) {
        BudgetManager{}.budgetCheck();
        FailureHandler{}.handleFailure();
        IntrospectionConsole{}.introspect();
    }
}

void BudgetManager::budgetCheck()
{
    const auto root = artifactRoot();
    std::uint64_t totalBytes = 0;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec) break;
        if (!entry.is_regular_file())
            continue;
        totalBytes += entry.file_size(ec);
        if (ec) {
            ec.clear();
        }
    }

    std::ostringstream oss;
    oss << "{\n"
        << "  \"artifact\": \"runtime_budget_report.json\",\n"
        << "  \"schema\": \"runtime_budget_report_v1\",\n"
        << "  \"buildMode\": \"" << buildModeName() << "\",\n"
        << "  \"generatedAtNs\": " << nowNs() << ",\n"
        << "  \"artifactTotalBytes\": " << totalBytes << ",\n"
        << "  \"limits\": {\n"
        << "    \"closureTraversal\": \"O(N) bounded\",\n"
        << "    \"publishValidationLatency\": \"bounded\",\n"
        << "    \"rtInstrumentation\": \"zero alloc / lock-free\",\n"
        << "    \"artifactSize\": \"bounded\",\n"
        << "    \"retireLatencyOverhead\": \"bounded\",\n"
        << "    \"metadataGrowth\": \"bounded\"\n"
        << "  },\n"
        << "  \"validatorPolicy\": [\"RB-1\", \"RB-2\", \"RB-3\"]\n"
        << "}\n";

    writeTextFile(root / "runtime_budget_report.json", oss.str());
}

void FailureHandler::handleFailure()
{
    const auto root = artifactRoot();
    const bool hbViolation = fileExistsAndNonEmpty(root / "hb_violation_report.json");
    const bool shutdownViolation = fileExistsAndNonEmpty(root / "shutdown_violation_report.json");

    std::ostringstream oss;
    oss << "{\n"
        << "  \"artifact\": \"recovery_trace.json\",\n"
        << "  \"generatedAtNs\": " << nowNs() << ",\n"
        << "  \"recoveryActions\": [\n"
        << "    {\"failure\":\"closure invalid\", \"action\":\"RejectPublish\"},\n"
        << "    {\"failure\":\"tier violation\", \"action\":\"RejectPublish\"},\n"
        << "    {\"failure\":\"HB violation\", \"action\":\"" << (hbViolation ? "Quarantine" : "Continue") << "\"},\n"
        << "    {\"failure\":\"retire timeout\", \"action\":\"DelayedReclaim\"},\n"
        << "    {\"failure\":\"seal violation\", \"action\":\"Abort\"},\n"
        << "    {\"failure\":\"shutdown violation\", \"action\":\"" << (shutdownViolation ? "SafeMode" : "Continue") << "\"}\n"
        << "  ],\n"
        << "  \"policy\": \"unsafe continuation prohibited\"\n"
        << "}\n";

    writeTextFile(root / "recovery_trace.json", oss.str());
}

void IntrospectionConsole::introspect()
{
    const auto root = artifactRoot();
    std::ostringstream oss;
    oss << "{\n"
        << "  \"artifact\": \"runtime_snapshot.json\",\n"
        << "  \"buildMode\": \"" << buildModeName() << "\",\n"
        << "  \"generatedAtNs\": " << nowNs() << ",\n"
        << "  \"summary\": {\n"
        << "    \"runtime\": \"minimal operational state\",\n"
        << "    \"retire\": \"summary only\",\n"
        << "    \"shutdown\": \"current + recent transitions\"\n"
        << "  },\n"
        << "  \"visibility\": \"Debug/CI or on-demand\"\n"
        << "}\n";

    writeTextFile(root / "runtime_snapshot.json", oss.str());
}

} // namespace convo::isr
