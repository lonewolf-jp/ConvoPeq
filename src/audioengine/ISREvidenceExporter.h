#pragma once
#include <string>

namespace convo::isr {

class EvidenceExporter {
public:
    void exportEvidence();
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
