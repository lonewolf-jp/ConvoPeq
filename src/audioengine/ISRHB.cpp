#include "ISRHB.h"
#include "AtomicAccess.h"
#include "ISRRetireLatency.h"
#include "WorldLifetimeRecorder.h"   // ★ A(D1): 退役遅延の集計
#include "core/SnapshotCoordinator.h"
#include "core/ReaderMeasureRegistry.h"   // ★ 計測 (2026-11-11)
#include <cstdio>   // ★ C2: RT 契約違反カウンタの出力

#include <chrono>
#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace convo {
namespace isr {

void HBTraceRuntime::recordEdge(const HBEdge& edge) noexcept
{
    const auto nowNs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());

    HBTraceEvent event{
        .timestamp = nowNs,
        .edge = edge,
        .isRelease = edge.memoryOrder == static_cast<int>(std::memory_order_release)
                  || edge.memoryOrder == static_cast<int>(std::memory_order_acq_rel),
        .isAcquire = edge.memoryOrder == static_cast<int>(std::memory_order_acquire)
                  || edge.memoryOrder == static_cast<int>(std::memory_order_acq_rel)
    };

    {
        std::lock_guard<std::mutex> lock(traceMutex_);
        traceEvents_.push_back(event);
    }

    (void)convo::fetchAddAtomic(eventCount_, uint64_t{1}, std::memory_order_relaxed);
}

void HBTraceRuntime::emitTrace(const std::filesystem::path& outputPath)
{
    std::error_code ec;
    std::filesystem::create_directories(outputPath.parent_path(), ec);

    std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return;
    }

    std::vector<HBTraceEvent> snapshot;
    {
        std::lock_guard<std::mutex> lock(traceMutex_);
        snapshot = traceEvents_;
    }

    file << "{\n";
    file << "  \"schema\": \"hb_trace_v1\",\n";
    file << "  \"eventCount\": "
         << convo::consumeAtomic(eventCount_, std::memory_order_acquire)
         << ",\n";
    file << "  \"events\": [\n";

    for (std::size_t i = 0; i < snapshot.size(); ++i) {
        const auto& e = snapshot[i];
        file << "    {\"ts\": " << e.timestamp
             << ", \"from\": " << e.edge.fromNode.nodeId
             << ", \"to\": " << e.edge.toNode.nodeId
             << ", \"fromEpoch\": " << e.edge.fromNode.epochId
             << ", \"toEpoch\": " << e.edge.toNode.epochId
             << ", \"mo\": " << e.edge.memoryOrder
             << ", \"release\": " << (e.isRelease ? "true" : "false")
             << ", \"acquire\": " << (e.isAcquire ? "true" : "false")
             << "}";
        if (i + 1u < snapshot.size()) {
            file << ",";
        }
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";
}

bool HBTraceRuntime::validateMonotonicTimestamps() noexcept
{
    std::lock_guard<std::mutex> lock(traceMutex_);
    if (traceEvents_.empty()) {
        return true;
    }

    uint64_t prev = traceEvents_.front().timestamp;
    for (std::size_t i = 1; i < traceEvents_.size(); ++i) {
        if (traceEvents_[i].timestamp < prev) {
            return false;
        }
        prev = traceEvents_[i].timestamp;
    }

    return true;
}

std::vector<HBTraceEvent> HBTraceRuntime::snapshotEvents()
{
    std::lock_guard<std::mutex> lock(traceMutex_);
    return traceEvents_;
}

bool HBRuntimeCore::verifyHBGuarantee(const HBNode& from, const HBNode& to) const noexcept
{
    if (from.epochId > to.epochId) {
        return false;
    }

    if (from.nodeId == to.nodeId) {
        return true;
    }

    for (const auto& edge : edges_) {
        if (edge.fromNode.nodeId == from.nodeId && edge.toNode.nodeId == to.nodeId
            && edge.fromNode.epochId <= edge.toNode.epochId) {
            return true;
        }
    }

    return false;
}

void HBRuntimeCore::registerEdge(const HBEdge& edge)
{
    edges_.push_back(edge);
}

bool HBVerifierRuntime::validateHBGraph() const noexcept
{
    if (!hasTraceRuntimeBound()) {
        return true;
    }

    return traceRuntime_->validateMonotonicTimestamps() && checkEpochOrder();
}

void HBVerifierRuntime::emitViolationReport(const std::filesystem::path& outputPath) const
{
    // ★ W-2 (2026-11-10): レポート生成時点（final 段）の累積値を保持する。
    //   postTeardown との差分を取るためだけに使い、判定には使わない。
    convo::isr::RetireLatencyRecorder::instance().captureFinalStage();

    // ★ 計測 (2026-11-11): final 段の reader レジストリ（active 一覧を含む）
    {
        std::error_code ecR;
        const auto p = std::filesystem::current_path() / "evidence" / "reader_registry_final.json";
        std::filesystem::create_directories(p.parent_path(), ecR);
        if (std::FILE* rf = std::fopen(p.string().c_str(), "wb"))
        {
            convo::ReaderMeasureRegistry::dump(rf, "final");
            std::fclose(rf);
        }
    }

    std::error_code ec;
    std::filesystem::create_directories(outputPath.parent_path(), ec);

    std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return;
    }

    const bool valid = validateHBGraph();
    const auto scenarios = runScenarioSuite();

    // ★ B-11: status は「グラフの妥当性」だけでなく「シナリオの合否」も反映する。
    //   以前は scenario が fail でも status=ok になり、status だけを見る検証器が
    //   失敗を見逃せた（実測で retire_delay=fail / status=ok を確認）。
    bool anyScenarioFailed = false;
    for (const auto& scenario : scenarios) {
        if (!scenario.passed) {
            anyScenarioFailed = true;
            break;
        }
    }
    const bool ok = valid && !anyScenarioFailed;

    file << "{\n";
    file << "  \"schema\": \"hb_violation_report_v1\",\n";
    file << "  \"status\": \"" << (ok ? "ok" : "violation") << "\",\n";
    file << "  \"violations\": [";
    bool wroteViolation = false;
    if (!valid) {
        file << "\"non-monotonic timestamp detected\"";
        wroteViolation = true;
    }

    for (const auto& scenario : scenarios) {
        if (!scenario.passed) {
            if (wroteViolation) {
                file << ",";
            }
            file << "\"scenario_failed:" << scenario.name << "\"";
            wroteViolation = true;
        }
    }

    file << "],\n";
    file << "  \"scenarioResults\": [\n";
    for (std::size_t i = 0; i < scenarios.size(); ++i) {
        const auto& scenario = scenarios[i];
        file << "    {\"name\": \"" << scenario.name << "\", \"result\": \""
             << (scenario.passed ? "pass" : "fail") << "\"";

        if (scenario.reason != nullptr)
            file << ", \"reason\": \"" << scenario.reason << "\"";

        if (scenario.required > 0)
            file << ", \"observations\": " << scenario.observations
                 << ", \"required\": " << scenario.required
                 << ", \"maxLatencyNs\": " << scenario.maxLatencyNs;

        file << "}";
        if (i + 1u < scenarios.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "  ],\n";

    // ★ A(D1): 退役遅延の集計（全体・経路別）と生の観測
    file << "  \"retireLatencyD1\": {\n";
    file << "    \"thresholdNs\": " << convo::isr::RetireLatencyRecorder::kThresholdNs << ",\n";
    file << "    \"required\": 59,\n";
    // ★ C2: RT 契約違反（RT からの resetFadeStateAndRetireTarget 呼出し）の回数
    file << "    \"snapshotRtViolations\": " << convo::SnapshotCoordinator::rtViolationCount() << ",\n";
    {
        const auto timingRuntime = convo::isr::RetireLatencyRecorder::instance().runtimeStage();
        const auto timingSnapshot = convo::isr::RetireLatencyRecorder::instance().snapshot();

        file << "    \"timing\": {\n";
        file << "      \"stage\": \"" << (timingRuntime.valid ? "runtime" : "final") << "\",\n";
        file << "      \"ebrCount\": " << (timingRuntime.valid ? timingRuntime.ebrCount : timingSnapshot.ebrCount) << ",\n";
        file << "      \"ebrSumNs\": " << (timingRuntime.valid ? timingRuntime.ebrSumNs : timingSnapshot.ebrSumNs) << ",\n";
        file << "      \"ebrMaxNs\": " << (timingRuntime.valid ? timingRuntime.ebrMaxNs : timingSnapshot.ebrMaxNs) << ",\n";
        file << "      \"execCount\": " << (timingRuntime.valid ? timingRuntime.execCount : timingSnapshot.execCount) << ",\n";
        file << "      \"execSumNs\": " << (timingRuntime.valid ? timingRuntime.execSumNs : timingSnapshot.execSumNs) << ",\n";
        file << "      \"execMaxNs\": " << (timingRuntime.valid ? timingRuntime.execMaxNs : timingSnapshot.execMaxNs) << ",\n";
        file << "      \"unobservedSafe\": " << (timingRuntime.valid ? timingRuntime.unobservedSafe : timingSnapshot.unobservedSafe) << ",\n";
        file << "      \"rtRequests\": " << timingSnapshot.rtRequests << "\n";
        file << "    },\n";
    }

    const auto& latencyRecorder = convo::isr::RetireLatencyRecorder::instance();
    const auto latencySnapshot = latencyRecorder.snapshot();
    const auto latencyRuntime = latencyRecorder.runtimeStage();

    convo::isr::RetireLatencyRecorder::StageStats latencyFinal;
    latencyFinal.valid = true;
    latencyFinal.requests = latencySnapshot.requests;
    latencyFinal.completions = latencySnapshot.completions;
    latencyFinal.unmatchedCompletions = latencySnapshot.unmatchedCompletions;
    latencyFinal.overflow = latencySnapshot.overflow;
    latencyFinal.idExhausted = latencySnapshot.idExhausted;
    latencyFinal.postShutdownRequests = latencySnapshot.postShutdownRequests;
    latencyFinal.recordMisses = latencySnapshot.recordMisses;   // ★ C5
    latencyFinal.pending = (latencySnapshot.requests > latencySnapshot.completions)
                             ? (latencySnapshot.requests - latencySnapshot.completions) : 0;

    for (std::size_t p = 0; p < convo::isr::RetireLatencyRecorder::kPathCount; ++p)
        latencyFinal.paths[p] = latencySnapshot.paths[p];

    const auto writeLatencyStage = [&file] (
        const char* label,
        const convo::isr::RetireLatencyRecorder::StageStats& stage,
        bool last) noexcept
    {
        file << "    \"" << label << "\": {\n";
        file << "      \"captured\": " << (stage.valid ? "true" : "false") << ",\n";
        file << "      \"requests\": " << stage.requests << ",\n";
        file << "      \"completions\": " << stage.completions << ",\n";
        file << "      \"pending\": " << stage.pending << ",\n";
        file << "      \"unmatchedCompletions\": " << stage.unmatchedCompletions << ",\n";
        file << "      \"overflow\": " << stage.overflow << ",\n";
        file << "      \"idExhausted\": " << stage.idExhausted << ",\n";
        file << "      \"postShutdownRequests\": " << stage.postShutdownRequests << ",\n";
        file << "      \"postShutdownCompletions\": " << stage.postShutdownCompletions << ",\n";
    file << "      \"recordMisses\": " << stage.recordMisses << ",\n";
        file << "      \"paths\": [";

        bool firstPathInStage = true;

        for (std::size_t p = 0; p < convo::isr::RetireLatencyRecorder::kPathCount; ++p)
        {
            const auto& ps = stage.paths[p];

            if (ps.count == 0)
                continue;

            file << (firstPathInStage ? "\n" : ",\n");
            file << "        {\"name\": \""
                 << convo::isr::RetireLatencyRecorder::pathName(
                        static_cast<convo::isr::RetireLatencyRecorder::Path>(p))
                 << "\", \"count\": " << ps.count
                 << ", \"overThreshold\": " << ps.overThreshold
                 << ", \"maxNs\": " << ps.maxNs << "}";
            firstPathInStage = false;
        }

        file << (firstPathInStage ? "]\n" : "\n      ]\n");
        file << "    }" << (last ? "\n" : ",\n");
    };

    writeLatencyStage ("runtime", latencyRuntime, false);
    writeLatencyStage ("final", latencyFinal, false);

    file << "    \"observations\": [";

    bool firstLatencyObs = true;

    for (const auto& o : latencySnapshot.observations)
    {
        file << (firstLatencyObs ? "\n" : ",\n");
        file << "      {\"requestId\": " << o.requestId
             << ", \"epoch\": " << o.epoch
             << ", \"path\": \""
             << convo::isr::RetireLatencyRecorder::pathName(
                    static_cast<convo::isr::RetireLatencyRecorder::Path>(o.path))
             << "\", \"latencyNs\": " << o.latencyNs
             << ", \"enqueueTsNs\": " << o.enqueueTsNs
             << ", \"safeTsNs\": " << o.safeTsNs
             << ", \"minReaderEpoch\": " << o.minReaderEpoch
             << ", \"minReaderSlot\": " << o.minReaderSlot << "}";
        firstLatencyObs = false;
    }

    file << (firstLatencyObs ? "],\n" : "\n    ],\n");
    file << "    \"pending\": [";

    bool firstLatencyPending = true;

    for (const auto& o : latencySnapshot.pending)
    {
        file << (firstLatencyPending ? "\n" : ",\n");
        file << "      {\"requestId\": " << o.requestId << ", \"epoch\": " << o.epoch << "}";
        firstLatencyPending = false;
    }

    file << (firstLatencyPending ? "]\n" : "\n    ]\n");
    file << "  },\n";

    // ★ 段階③ (2026-11-11, Owner 承認): 3 分類診断の追加セクション。
    //   既存の retireLatencyD1（判定に使う）は一切変更しない。ここは診断のみ。
    {
        const auto clsSnapshot = convo::isr::RetireLatencyRecorder::instance().snapshot();
        const auto& tot = clsSnapshot.classTotals;

        file << "  \"retireLatencyClassified\": {\n";
        file << "    \"note\": \"diagnostic only; the verdict uses retireLatencyD1.runtime and is unchanged\",\n";
        file << "    \"thresholdNs\": " << convo::isr::RetireLatencyRecorder::kThresholdNs << ",\n";
        file << "    \"required\": 59,\n";
        file << "    \"paths\": [";

        bool firstClassPath = true;

        for (std::size_t p = 0; p < convo::isr::RetireLatencyRecorder::kPathCount; ++p)
        {
            const auto& cp = clsSnapshot.classifiedPaths[p];

            if (cp.driverWait.count == 0 && cp.ebrWait.count == 0 && cp.execWait.count == 0
                && cp.unclassifiedCount == 0 && cp.excludedCount == 0)
                continue;

            const auto writeComponent = [&file] (const char* name,
                                                 const convo::isr::RetireLatencyRecorder::ClassComponent& c)
            {
                file << "\"" << name << "\": {\"count\": " << c.count
                     << ", \"sumNs\": " << c.sumNs
                     << ", \"maxNs\": " << c.maxNs
                     << ", \"over50ms\": " << c.overThreshold << "}";
            };

            file << (firstClassPath ? "\n" : ",\n");
            file << "      {\"name\": \""
                 << convo::isr::RetireLatencyRecorder::pathName(
                        static_cast<convo::isr::RetireLatencyRecorder::Path>(p))
                 << "\", ";
            writeComponent ("driverWait", cp.driverWait);
            file << ", ";
            writeComponent ("ebrWait", cp.ebrWait);
            file << ", ";
            writeComponent ("execWait", cp.execWait);
            file << ", \"unclassified\": {\"count\": " << cp.unclassifiedCount
                 << ", \"sumNs\": " << cp.unclassifiedSumNs << "}"
                 << ", \"excluded\": {\"count\": " << cp.excludedCount << "}";
            file << "}";
            firstClassPath = false;
        }

        file << (firstClassPath ? "],\n" : "\n    ],\n");
        file << "    \"totals\": {\"requests\": " << tot.requests
             << ", \"completions\": " << tot.completions
             << ", \"classified\": " << tot.classified
             << ", \"unclassified\": " << tot.unclassified
             << ", \"excluded\": " << tot.excluded
             << ", \"consistencyErrors\": " << tot.consistencyErrors
             << ", \"pending\": " << tot.pending << "},\n";
        // ★ domain tag (v2.9 §5.1): ドメイン別の 4 行を常に出力する（0 件でも省略しない）。
        //   paths / totals は既存の合算のままで、キーも値も変えない。
        {
            const auto& domains = clsSnapshot.domains;

            file << "    \"domains\": [";

            for (int d = 0; d < convo::kRetireDomainCount; ++d)
            {
                const auto& ds = domains[static_cast<std::size_t>(d)];
                const auto id = static_cast<convo::RetireDomainId>(d);

                file << (d == 0 ? "\n" : ",\n");
                file << "      {\"id\": \"" << convo::retireDomainName(id) << "\""
                     << ", \"count\": " << ds.count
                     << ", \"classified\": " << ds.classified
                     << ", \"unclassified\": " << ds.unclassified
                     << ", \"excluded\": " << ds.excluded
                     << ", \"consistencyErrors\": " << ds.consistencyErrors
                     << ", \"sumNs\": " << ds.sumNs
                     << ", \"maxNs\": " << ds.maxNs
                     << ", \"over50ms\": " << ds.overThreshold;

                if (id == convo::RetireDomainId::Unknown)
                {
                    file << ", \"unknownReasons\": {\"providerUnknown\": " << clsSnapshot.providerUnknown
                         << ", \"providerAbsent\": " << clsSnapshot.providerAbsent << "}";
                }
                else if (id == convo::RetireDomainId::Unstamped)
                {
                    file << ", \"unknownReasons\": {\"stampMissing\": " << clsSnapshot.stampMissing << "}";
                }

                file << ", \"paths\": [";

                bool firstDomainPath = true;

                for (std::size_t p = 0; p < convo::isr::RetireLatencyRecorder::kPathCount; ++p)
                {
                    const auto& cp = ds.paths[p];

                    if (cp.driverWait.count == 0 && cp.ebrWait.count == 0 && cp.execWait.count == 0
                        && cp.unclassifiedCount == 0 && cp.excludedCount == 0)
                        continue;

                    const auto writeComponent = [&file] (const char* name,
                                                         const convo::isr::RetireLatencyRecorder::ClassComponent& c)
                    {
                        file << "\"" << name << "\": {\"count\": " << c.count
                             << ", \"sumNs\": " << c.sumNs
                             << ", \"maxNs\": " << c.maxNs
                             << ", \"over50ms\": " << c.overThreshold << "}";
                    };

                    file << (firstDomainPath ? "\n" : ",\n");
                    file << "        {\"name\": \""
                         << convo::isr::RetireLatencyRecorder::pathName(
                                static_cast<convo::isr::RetireLatencyRecorder::Path>(p))
                         << "\", ";
                    writeComponent ("driverWait", cp.driverWait);
                    file << ", ";
                    writeComponent ("ebrWait", cp.ebrWait);
                    file << ", ";
                    writeComponent ("execWait", cp.execWait);
                    file << ", \"unclassified\": {\"count\": " << cp.unclassifiedCount
                         << ", \"sumNs\": " << cp.unclassifiedSumNs << "}"
                         << ", \"excluded\": {\"count\": " << cp.excludedCount << "}";
                    file << "}";
                    firstDomainPath = false;
                }

                file << (firstDomainPath ? "]" : "\n      ]");
                file << "}";
            }

            file << "\n    ],\n";

            std::uint64_t sumCount = 0, sumClassified = 0, sumUnclassified = 0, sumExcluded = 0,
                           sumConsistency = 0, sumDriver = 0, sumEbr = 0, sumExec = 0,
                           sumUnclassifiedPath = 0, sumExcludedPath = 0;

            for (int d = 0; d < convo::kRetireDomainCount; ++d)
            {
                const auto& ds = domains[static_cast<std::size_t>(d)];
                sumCount += ds.count;
                sumClassified += ds.classified;
                sumUnclassified += ds.unclassified;
                sumExcluded += ds.excluded;
                sumConsistency += ds.consistencyErrors;

                for (std::size_t p = 0; p < convo::isr::RetireLatencyRecorder::kPathCount; ++p)
                {
                    sumDriver += ds.paths[p].driverWait.count;
                    sumEbr += ds.paths[p].ebrWait.count;
                    sumExec += ds.paths[p].execWait.count;
                    sumUnclassifiedPath += ds.paths[p].unclassifiedCount;
                    sumExcludedPath += ds.paths[p].excludedCount;
                }
            }

            std::uint64_t totalsDriver = 0, totalsEbr = 0, totalsExec = 0,
                           totalsUnclassifiedPath = 0, totalsExcludedPath = 0;

            for (std::size_t p = 0; p < convo::isr::RetireLatencyRecorder::kPathCount; ++p)
            {
                totalsDriver += clsSnapshot.classifiedPaths[p].driverWait.count;
                totalsEbr += clsSnapshot.classifiedPaths[p].ebrWait.count;
                totalsExec += clsSnapshot.classifiedPaths[p].execWait.count;
                totalsUnclassifiedPath += clsSnapshot.classifiedPaths[p].unclassifiedCount;
                totalsExcludedPath += clsSnapshot.classifiedPaths[p].excludedCount;
            }

            const bool domainSumsConsistent =
                sumCount == tot.completions
                && sumClassified == tot.classified
                && sumUnclassified == tot.unclassified
                && sumExcluded == tot.excluded
                && sumConsistency == tot.consistencyErrors
                && sumDriver == totalsDriver
                && sumEbr == totalsEbr
                && sumExec == totalsExec
                && sumUnclassifiedPath == totalsUnclassifiedPath
                && sumExcludedPath == totalsExcludedPath;

            file << "    \"stampMissing\": " << clsSnapshot.stampMissing << ",\n";
            file << "    \"providerUnknown\": " << clsSnapshot.providerUnknown << ",\n";
            file << "    \"providerAbsent\": " << clsSnapshot.providerAbsent << ",\n";
            file << "    \"domainSumsConsistent\": " << (domainSumsConsistent ? "true" : "false") << ",\n";
        }

        const bool countsConsistent = (tot.classified + tot.unclassified + tot.excluded) == tot.completions;
        file << "    \"countsConsistent\": " << (countsConsistent ? "true" : "false") << "\n";
        file << "  },\n";
    }

    // ★ Item2: RuntimeWorld の寿命（D1 は別系統・別判定）
    {
        const auto worldSnapshot = convo::isr::WorldLifetimeRecorder::instance().snapshot();

        file << "  \"worldLifetime\": {\n";
        file << "    \"commits\": " << worldSnapshot.commits << ",\n";
        file << "    \"destroys\": " << worldSnapshot.destroys << ",\n";
        file << "    \"unmatched\": " << worldSnapshot.unmatched << ",\n";
        file << "    \"overflow\": " << worldSnapshot.overflow << ",\n";
        file << "    \"paths\": [";

        bool firstWorldPath = true;

        for (std::size_t p = 0; p < convo::isr::WorldLifetimeRecorder::kPathCount; ++p)
        {
            const auto& ps = worldSnapshot.paths[p];

            if (ps.commits == 0 && ps.destroys == 0)
                continue;

            file << (firstWorldPath ? "\n" : ",\n");
            file << "      {\"name\": \""
                 << convo::isr::WorldLifetimeRecorder::pathName(
                        static_cast<convo::isr::WorldLifetimeRecorder::Path>(p))
                 << "\", \"commits\": " << ps.commits
                 << ", \"destroys\": " << ps.destroys
                 << ", \"sumNs\": " << ps.sumNs
                 << ", \"maxNs\": " << ps.maxNs << "}";
            firstWorldPath = false;
        }

        file << (firstWorldPath ? "],\n" : "\n    ],\n");
        file << "    \"observations\": [";

        bool firstWorldObs = true;

        for (const auto& o : worldSnapshot.observations)
        {
            file << (firstWorldObs ? "\n" : ",\n");
            file << "      {\"worldId\": " << o.worldId
                 << ", \"path\": \""
                 << convo::isr::WorldLifetimeRecorder::pathName(
                        static_cast<convo::isr::WorldLifetimeRecorder::Path>(o.path))
                 << "\", \"lifetimeNs\": " << o.lifetimeNs << "}";
            firstWorldObs = false;
        }

        file << (firstWorldObs ? "]\n" : "\n    ]\n");
        file << "  }\n";
    }

    file << "}\n";
}

bool HBVerifierRuntime::simulateReorderScenario(HBReorderScenario scenario) const noexcept
{
    if (!hasTraceRuntimeBound() || !hasAnyEvent()) {
        return false;
    }

    const bool valid = validateHBGraph();
    if (!valid) {
        return false;
    }

    switch (scenario) {
    case HBReorderScenario::ForcedReorder:
        return checkEpochOrder();
    case HBReorderScenario::EpochLag:
        return checkBoundedEpochLag(4u);
    case HBReorderScenario::RetireDelay:
        return evaluateRetireDelay().passed;
    case HBReorderScenario::ObserveRace:
        return checkObserveRaceFree();
    default:
        return false;
    }
}

std::array<HBScenarioResult, 4> HBVerifierRuntime::runScenarioSuite() const noexcept
{
    return {{
        {"forced_reorder", simulateReorderScenario(HBReorderScenario::ForcedReorder)},
        {"epoch_lag", simulateReorderScenario(HBReorderScenario::EpochLag)},
        evaluateRetireDelay(),
        {"observe_race", simulateReorderScenario(HBReorderScenario::ObserveRace)}
    }};
}

void HBVerifierRuntime::setTraceRuntime(HBTraceRuntime* traceRuntime) noexcept
{
    traceRuntime_ = traceRuntime;
}

bool HBVerifierRuntime::hasTraceRuntimeBound() const noexcept
{
    return traceRuntime_ != nullptr;
}

bool HBVerifierRuntime::hasAnyEvent() const noexcept
{
    if (!hasTraceRuntimeBound()) {
        return false;
    }

    const auto events = traceRuntime_->snapshotEvents();
    return !events.empty();
}

bool HBVerifierRuntime::checkEpochOrder() const noexcept
{
    if (!hasTraceRuntimeBound()) {
        return false;
    }

    const auto events = traceRuntime_->snapshotEvents();
    for (const auto& e : events) {
        if (e.edge.toNode.epochId < e.edge.fromNode.epochId) {
            return false;
        }
    }

    return true;
}

bool HBVerifierRuntime::checkBoundedEpochLag(std::uint64_t maxLag) const noexcept
{
    if (!hasTraceRuntimeBound()) {
        return false;
    }

    const auto events = traceRuntime_->snapshotEvents();
    for (const auto& e : events) {
        if (e.edge.toNode.epochId >= e.edge.fromNode.epochId) {
            if ((e.edge.toNode.epochId - e.edge.fromNode.epochId) > maxLag) {
                return false;
            }
        }
    }

    return true;
}

HBScenarioResult HBVerifierRuntime::evaluateRetireDelay() const noexcept
{
    // N_req = ceil(ln(0.05)/ln(0.95)) = 59（95% 信頼で超過率 5% 未満、超過 0 件の場合）
    constexpr std::uint64_t kNReq = 59;

    const auto& recorder = convo::isr::RetireLatencyRecorder::instance();

    // 判定は「最終 drain 直前（runtime）」の状態で行う。drainAll の解放は通常経路に混ぜない。
    auto stage = recorder.runtimeStage();

    if (! stage.valid)
    {
        const auto snapshot = recorder.snapshot();
        stage.valid = true;
        stage.requests = snapshot.requests;
        stage.completions = snapshot.completions;
        stage.unmatchedCompletions = snapshot.unmatchedCompletions;
        stage.overflow = snapshot.overflow;
        stage.idExhausted = snapshot.idExhausted;
        stage.postShutdownRequests = snapshot.postShutdownRequests;
        stage.pending = (snapshot.requests > snapshot.completions)
                          ? (snapshot.requests - snapshot.completions) : 0;

        for (std::size_t p = 0; p < convo::isr::RetireLatencyRecorder::kPathCount; ++p)
            stage.paths[p] = snapshot.paths[p];
    }

    using Path = convo::isr::RetireLatencyRecorder::Path;

    // 通常経路: sync / drain / queue / quarantine（queue は EBR 待ちを含むが D1 の一部）
    constexpr std::size_t kNormalPaths[] = {
        static_cast<std::size_t> (Path::Sync),
        static_cast<std::size_t> (Path::Drain),
        static_cast<std::size_t> (Path::Queue),
        static_cast<std::size_t> (Path::Quarantine)
    };

    std::uint64_t normalCount = 0;
    std::uint64_t normalMaxNs = 0;

    for (const auto index : kNormalPaths)
    {
        normalCount += stage.paths[index].count;

        if (stage.paths[index].maxNs > normalMaxNs)
            normalMaxNs = stage.paths[index].maxNs;
    }

    HBScenarioResult result;
    result.name = "retire_delay";
    result.required = kNReq;
    result.observations = normalCount;
    result.maxLatencyNs = normalMaxNs;

    const auto insufficient = [&result] (const char* why)
    {
        result.passed = false;
        result.insufficient = true;
        result.reason = why;
        return result;
    };

    if (stage.overflow > 0 || stage.idExhausted > 0)
        return insufficient ("observation_overflow");

    if (stage.pending > 0)
        return insufficient ("incomplete_requests");

    if (normalCount < kNReq)
        return insufficient ("insufficient_observations");

    if (normalMaxNs > convo::isr::RetireLatencyRecorder::kThresholdNs)
    {
        result.passed = false;
        result.reason = "latency_exceeds_threshold";
        return result;
    }

    result.passed = true;
    return result;
}

bool HBVerifierRuntime::checkObserveRaceFree() const noexcept
{
    if (!hasTraceRuntimeBound()) {
        return false;
    }

    const auto events = traceRuntime_->snapshotEvents();
    std::unordered_map<std::uint64_t, std::uint32_t> sinkByEpoch;

    for (const auto& e : events) {
        const std::uint64_t key = (static_cast<std::uint64_t>(e.edge.toNode.nodeId) << 32u)
            | static_cast<std::uint64_t>(e.edge.toNode.epochId & 0xFFFFFFFFu);
        const std::uint32_t from = e.edge.fromNode.nodeId;

        const auto iter = sinkByEpoch.find(key);
        if (iter == sinkByEpoch.end()) {
            sinkByEpoch.emplace(key, from);
            continue;
        }

        if (iter->second != from) {
            return false;
        }
    }

    return true;
}

}  // namespace isr
}  // namespace convo
