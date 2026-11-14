#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include "TelemetryRecorder.h"
#include "LiveRuntimeEvidence.h"

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

// ★ Live Runtime 証拠: 実行中のプロセス（AudioEngine）が観測した値を JSON に書く。
//   本関数は「受け取った値を書くだけ」で、Runtime の実行・値の生成・補完は行わない。
//   evidence/live_runtime_evidence.json を tmp + rename で書く。
//   戻り値: 書き込みに成功したら true。runId 空・時刻不整合・I/O 失敗は false（fail-closed）。
bool writeLiveRuntimeEvidence(const LiveRuntimeObservation& observation,
                              std::string* errorOut = nullptr);

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
