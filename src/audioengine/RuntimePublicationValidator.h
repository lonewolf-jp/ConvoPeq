#pragma once

#include "ISRRuntimeSemanticSchema.h"
#include <memory>
#include <string>

struct RuntimeState;

namespace iso::audio_engine {
using RuntimePublishWorld = ::RuntimeState;

// ★ Validator の公開APIとして ValidationFailureReason を定義
enum class ValidationFailureReason : uint8_t {
    None,
    InvalidTopology,
    InvalidResources,
    InvalidTransition,
    SemanticInconsistency
};

struct RuntimeValidationResult {
    bool isValid = true;
    std::string errorMessage;
    ValidationFailureReason failureReason{ValidationFailureReason::None};
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
 *
 * ★ Phase-4: Builder/Validator 責務定義
 *   | レイヤ         | 責務                                   | 根拠                                           |
 *   |----------------|----------------------------------------|------------------------------------------------|
 *   | RuntimeBuilder | semantic 値の正しい設定                | Builder は各フィールドに適切な値を設定することが責務 |
 *   | Validator      | 不変条件の最終確認                      | Builder の設定漏れ・バグを検出する安全網           |
 *   | Orchestrator   | 運用ポリシーの適用                      | HealthState 等の実行時状態に基づく上書き判断       |
 *
 *   原則: Builder の通過が Validator 通過を保証するわけではない。
 *   Validator は Builder とは独立した Permissionless Check として動作する。
 *   両者が一致していることは望ましいが、Validator は Builder の実装詳細を知らず、
 *   純粋に世界の状態だけを検証する。
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
