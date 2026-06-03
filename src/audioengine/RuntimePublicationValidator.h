#pragma once

#include "ISRRuntimeSemanticSchema.h"
#include <memory>
#include <string>

struct RuntimeState;

namespace iso::audio_engine {
using RuntimePublishWorld = ::RuntimeState;

struct RuntimeValidationResult {
    bool isValid = true;
    std::string errorMessage;
};

/**
 * RuntimePublicationValidator
 * 
 * 責務: Runtime publication の事前検証 (precheck) を実行する。
 * 
 * Design Principle:
 * - Pure validation logic only (no side effects)
 * - No dependency on AudioEngine
 * - Stateless (can be shared across threads)
 * 
 * This class extracts the pure validation logic from
 * AudioEngine::runPublicationPrecheckNonRt() to achieve
 * separation of concerns.
 */
class RuntimePublicationValidator {
public:
    RuntimePublicationValidator() = default;
    ~RuntimePublicationValidator() = default;

    /**
     * Validate publication before execution.
     * 
     * @param world The RuntimePublishWorld to validate
     * @return RuntimeValidationResult with success/failure and error message
     */
    RuntimeValidationResult validatePublication(
        const RuntimePublishWorld& world) const;

    /**
     * Validate semantic consistency.
     * 
     * @param world The RuntimePublishWorld to validate
     * @return true if semantics are consistent
     */
    bool validateSemanticConsistency(
        const RuntimePublishWorld& world) const;

    /**
     * Validate topology constraints.
     * 
     * @param world The RuntimePublishWorld to validate
     * @return true if topology is valid
     */
    bool validateTopology(const RuntimePublishWorld& world) const;

    /**
     * Validate resource availability.
     * 
     * @param world The RuntimePublishWorld to validate
     * @return true if resources are available
     */
    bool validateResources(const RuntimePublishWorld& world) const;

private:
    // Helper methods
    bool checkExecutionSemanticValidity(
        const convo::isr::ExecutionSemantic& exec) const;
    
    bool checkActivationEpochConsistency(
        const convo::isr::GenerationSemantic& gen,
        const convo::isr::TimingSemantic& timing) const;
    
    bool checkNoConflictingTransitions(
        const RuntimePublishWorld& world) const;
};

} // namespace iso::audio_engine
