#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include "TelemetryRecorder.h"

namespace convo::isr {

class EvidenceExporter {
public:
    // exportEvidence: テレメトリデータを含むエビデンスを出力する。
    // forceFullEvidence: Release ビルドでも全 artifact を書き出す（実測証跡モード用）。
    //   既定 false。CI の run 経路だけが true を渡す。
    void exportEvidence(const TelemetryRecorder::TelemetrySnapshot* snapshot = nullptr,
                        uint64_t monotonicViolationCount = 0,
                        bool forceFullEvidence = false);
};

class BudgetManager {
public:
    void budgetCheck();
};

class FailureHandler {
public:
    void handleFailure();
};

class IntrospectionConsole {
public:
    void introspect();
};

// ★ A(D1): 既存 artifact に runtime metadata（provenance/runId/generatedAtNs）を付与する。
void reapplyRuntimeMetadata(const std::filesystem::path& path,
                                   std::string_view runId,
                                   std::uint64_t generatedAtNs);

} // namespace convo::isr
