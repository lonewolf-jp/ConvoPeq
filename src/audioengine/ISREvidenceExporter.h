#pragma once
#include <string>
#include "TelemetryRecorder.h"

namespace convo::isr {

class EvidenceExporter {
public:
    // exportEvidence: テレメトリデータを含むエビデンスを出力する。
    void exportEvidence(const TelemetryRecorder::TelemetrySnapshot* snapshot = nullptr,
                        uint64_t monotonicViolationCount = 0);
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

} // namespace convo::isr
