#include "RuntimePublicationValidator.h"
#include "ISRRuntimeSemanticSchema.h"
#include "AudioEngine.h"
#include <string>

namespace iso::audio_engine {

RuntimeValidationResult RuntimePublicationValidator::validatePublication(
    const RuntimePublishWorld& world) const
{
    RuntimeValidationResult result;

    // 1. Semantic consistency check
    if (!validateSemanticConsistency(world)) {
        result.isValid = false;
        result.errorMessage = "Semantic consistency check failed";
        result.failureReason = ValidationFailureReason::SemanticInconsistency;
        return result;
    }

    // 2. Topology validation
    if (!validateTopology(world)) {
        result.isValid = false;
        result.errorMessage = "Topology validation failed";
        result.failureReason = ValidationFailureReason::InvalidTopology;
        return result;
    }

    // 3. Resource availability check
    if (!validateResources(world)) {
        result.isValid = false;
        result.errorMessage = "Resource availability check failed";
        result.failureReason = ValidationFailureReason::InvalidResources;
        return result;
    }

    // 4. Check for conflicting transitions
    if (!checkNoConflictingTransitions(world)) {
        result.isValid = false;
        result.errorMessage = "Conflicting transitions detected";
        result.failureReason = ValidationFailureReason::InvalidTransition;
        return result;
    }

    return result;
}

bool RuntimePublicationValidator::validateSemanticConsistency(
    const RuntimePublishWorld& world) const
{
    const auto& gen = world.generationSemantic;
    const auto& timing = world.timing;
    const auto& exec = world.execution;

    // Check activation epoch consistency
    if (!checkActivationEpochConsistency(gen, timing)) {
        return false;
    }

    // Check execution semantic validity
    if (!checkExecutionSemanticValidity(exec)) {
        return false;
    }

    // ★ P4-1: Publication sequence — generation > 0 なら sequenceId が 0 でないこと
    //   Bootstrap world (generation=0) でも sequenceId は通常 1 以上だが、
    //   generation を Bootstrap 判別の唯一の基準とする
    if (world.generation > 0 && world.publication.sequenceId == 0)
        return false;

    return true;
}

bool RuntimePublicationValidator::validateTopology(
    const RuntimePublishWorld& world) const
{
    const auto& topology = world.topology;
    const auto& execution = world.execution;

    // ★ v8.3: Validator 三カテゴリ（Topology / Execution / Identity）

    // === Topology Invariant ===
    // hasFadingRuntime は保持せず、graph.fadingNode != nullptr が唯一の Authority。
    // runtimeUuid==0 は Bootstrap/Shutdown として許容。
    if (topology.runtimeUuid == 0) {
        if (execution.transitionActive) return false;
        if (topology.fadingRuntimeUuid != 0) return false;
    }

    // === Execution Invariant ===
    // transitionActive は Topology とは独立。fadeTimeSec=0 のケースでは
    // fadingRuntimeUuid != 0 でも transitionActive=false であり得る。
    if (execution.transitionPolicy < 0 || execution.transitionPolicy > 2)
        return false;

    // ★ P4-1: RoutingSemantic — processingOrder は 0 または 1 のみ許容
    if (world.routing.processingOrder < 0 || world.routing.processingOrder > 1)
        return false;

    // === Identity Invariant ===
    // fadingRuntimeUuid（保持）は導出可能だが、graph寿命超えたstable identifierとして保持。
    // 自己同一性チェック: 同一ノードの二重登録検出
    if (topology.fadingRuntimeUuid != 0 && topology.fadingRuntimeUuid == topology.runtimeUuid)
        return false;

    // ★ P4-1: GenerationSemantic — generation > 0 なら runtimeGeneration > 0
    if (world.generation > 0 && world.generationSemantic.runtimeGeneration == 0)
        return false;

    return true;
}

bool RuntimePublicationValidator::validateResources(
    const RuntimePublishWorld& world) const
{
    const auto& resource = world.resource;

    // Oversampling: 2のべき乗かつ1〜16
    const int os = resource.oversamplingFactor;
    if (os < 1 || os > 16 || (os & (os - 1)) != 0)
        return false;

    // Dither: 0, 16, 24, 32 のみ許容（kAdaptiveBitDepthValues との整合性）
    const int dd = resource.ditherBitDepth;
    if (dd != 0 && dd != 16 && dd != 24 && dd != 32)
        return false;

    // NoiseShaper: 0, 1, 2, 3 のみ許容（Fixed15Tap の追加）
    const int ns = resource.noiseShaperType;
    if (ns < 0 || ns > 3)
        return false;

    return true;
}

bool RuntimePublicationValidator::checkExecutionSemanticValidity(
    const convo::isr::ExecutionSemantic& exec) const
{
    // Validate execution semantic fields
    // - transitionActive should be consistent with crossfade parameters
    // - crossfadeStartDelayBlocks should be non-negative
    // - crossfadeDryHoldSamples should be within acceptable range

    if (exec.crossfadeStartDelayBlocks < 0) {
        return false;
    }

    if (exec.crossfadeDryHoldSamples < 0) {
        return false;
    }

    return true;
}

bool RuntimePublicationValidator::checkActivationEpochConsistency(
    const convo::isr::GenerationSemantic& gen,
    const convo::isr::TimingSemantic& timing) const
{
    // Since TimingSemantic.activationEpoch is now a derived field,
    // we don't need to check consistency here.
    // The authority is GenerationSemantic.activationEpoch only.

    // However, we can add sanity checks if needed
    // For example, activationEpoch should be monotonically increasing

    return true;
}

bool RuntimePublicationValidator::checkNoConflictingTransitions(
    const RuntimePublishWorld& world) const
{
    const auto& exec = world.execution;
    const auto& overlap = world.overlap;

    const bool active = exec.transitionActive;
    const double fade = overlap.fadeTimeSec;

    if (!active) {
        // ★ transitionActive=false でも fadeTimeSec が残るケースを許容
        //   フェード完了直後の Idle World publish 時など、fadeTimeSec が保持されたまま
        //   遷移する将来実装が入り得る。そのため fade > 0.0 は reject しない。
        //   ただし負の fade は常に異常値として reject。
        if (fade < 0.0) return false;
        // useDryAsOld=true かつ !active は意味論的に矛盾
        if (overlap.useDryAsOld) return false;
        return true;
    }

    const auto policy = static_cast<convo::TransitionPolicy>(exec.transitionPolicy);

    switch (policy) {
        case convo::TransitionPolicy::SmoothOnly:
            if (fade < 0.0) return false;  // 負値のみ拒否（0.0はフォールバック機構に委ねる）
            break;
        case convo::TransitionPolicy::DryAsOld:
            if (fade < 0.0) return false;
            if (!overlap.useDryAsOld) return false;
            break;
        case convo::TransitionPolicy::HardReset:
            if (fade < 0.0) return false;  // 負値も拒否
            if (fade > 0.0) return false;  // 正値も拒否（HardReset は fade=0.0 のみ許容）
            if (overlap.useDryAsOld) return false;
            break;
        default:
            return false;  // 未知の policy 値は拒否
    }

    return true;
}

} // namespace iso::audio_engine
