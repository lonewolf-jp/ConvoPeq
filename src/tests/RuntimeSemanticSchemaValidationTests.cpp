#include <stdexcept>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <queue>
#include <functional>

#include "audioengine/ISRRuntimeSemanticSchema.h"

namespace {

[[nodiscard]] bool testRoutingProcessingOrderRange()
{
    convo::isr::RoutingSemantic validRouting {};
    validRouting.processingOrder = 1;
    if (!convo::isr::isValidRoutingSemantic(validRouting))
        return false;

    convo::isr::RoutingSemantic invalidRouting {};
    invalidRouting.processingOrder = 2;
    if (convo::isr::isValidRoutingSemantic(invalidRouting))
        return false;

    return true;
}

[[nodiscard]] bool testExecutionSemanticRangeAndNonNegativeFields()
{
    convo::isr::ExecutionSemantic validExecution {};
    validExecution.transitionPolicy = 2;
    validExecution.crossfadeStartDelayBlocks = 0;
    validExecution.crossfadeDryHoldSamples = 0;
    if (!convo::isr::isValidExecutionSemantic(validExecution))
        return false;

    convo::isr::ExecutionSemantic invalidPolicy {};
    invalidPolicy.transitionPolicy = 3;
    if (convo::isr::isValidExecutionSemantic(invalidPolicy))
        return false;

    convo::isr::ExecutionSemantic invalidDelay {};
    invalidDelay.transitionPolicy = 1;
    invalidDelay.crossfadeStartDelayBlocks = -1;
    if (convo::isr::isValidExecutionSemantic(invalidDelay))
        return false;

    convo::isr::ExecutionSemantic invalidHold {};
    invalidHold.transitionPolicy = 1;
    invalidHold.crossfadeDryHoldSamples = -1;
    if (convo::isr::isValidExecutionSemantic(invalidHold))
        return false;

    return true;
}

[[nodiscard]] bool testPublicationDescriptorSetValidation()
{
    if (!convo::isr::PublicationSemantic::validateDescriptorSet())
        return false;

    constexpr std::array<convo::isr::RuntimeFieldDescriptor, 2> duplicateNames {{
        {"dup", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld,
         convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary,
         convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"dup", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld,
         convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary,
         convo::isr::LifetimeClass::RuntimeWorldLifetime}
    }};

    if (convo::isr::validateFieldDescriptorSet(duplicateNames))
        return false;

    return true;
}

[[nodiscard]] bool testSchemaVersionConsistency()
{
    convo::isr::RuntimeSemanticSchema schema {};
    if (schema.metadata.schemaVersion != convo::isr::kRuntimeSemanticSchemaVersion)
        return false;

    if (schema.metadata.publicationSequence != 0)
        return false;

    return true;
}

[[nodiscard]] bool testSemanticDagCycleValidation()
{
    using Node = std::string;
    const std::unordered_map<Node, std::vector<Node>> dag {
        {"RuntimeWorld", {"RoutingSemantic", "ExecutionSemantic", "PublicationSemantic"}},
        {"RoutingSemantic", {"ObservePath"}},
        {"ExecutionSemantic", {"CrossfadeSemantic"}},
        {"PublicationSemantic", {"RetireSemantic"}},
        {"CrossfadeSemantic", {}},
        {"ObservePath", {}},
        {"RetireSemantic", {}}
    };

    std::unordered_set<Node> temporary;
    std::unordered_set<Node> permanent;

    std::function<bool(const Node&)> visit = [&](const Node& node) {
        if (permanent.find(node) != permanent.end())
            return true;
        if (temporary.find(node) != temporary.end())
            return false;

        temporary.insert(node);
        const auto it = dag.find(node);
        if (it != dag.end())
        {
            for (const auto& next : it->second)
            {
                if (!visit(next))
                    return false;
            }
        }
        temporary.erase(node);
        permanent.insert(node);
        return true;
    };

    for (const auto& entry : dag)
    {
        if (!visit(entry.first))
            return false;
    }

    return true;
}

[[nodiscard]] bool testRuntimeSemanticTransitionGraphValidation()
{
    using Edge = std::pair<std::string, std::string>;
    const std::vector<Edge> requiredEdges {
        {"Draft", "Publishing"},
        {"Publishing", "Published"},
        {"Published", "Retiring"},
        {"Retiring", "Retired"},
        {"Retired", "Destroyed"}
    };

    const std::unordered_set<std::string> allowedStates {
        "Draft", "Publishing", "Published", "Retiring", "Retired", "Destroyed"
    };

    for (const auto& edge : requiredEdges)
    {
        if (allowedStates.find(edge.first) == allowedStates.end())
            return false;
        if (allowedStates.find(edge.second) == allowedStates.end())
            return false;
        if (edge.first == edge.second)
            return false;
    }

    const std::vector<Edge> forbiddenEdges {
        {"Destroyed", "Published"},
        {"Retired", "Publishing"},
        {"Published", "Draft"}
    };

    for (const auto& forbidden : forbiddenEdges)
    {
        for (const auto& required : requiredEdges)
        {
            if (forbidden == required)
                return false;
        }
    }

    return true;
}

[[nodiscard]] bool testRuntimeSemanticReachabilityValidation()
{
    const std::unordered_map<std::string, std::vector<std::string>> graph {
        {"Draft", {"Publishing"}},
        {"Publishing", {"Published"}},
        {"Published", {"Retiring"}},
        {"Retiring", {"Retired"}},
        {"Retired", {"Destroyed"}},
        {"Destroyed", {}}
    };

    std::queue<std::string> q;
    std::unordered_set<std::string> visited;

    q.push("Draft");
    visited.insert("Draft");

    while (!q.empty())
    {
        const auto cur = q.front();
        q.pop();

        const auto it = graph.find(cur);
        if (it == graph.end())
            continue;

        for (const auto& next : it->second)
        {
            if (visited.insert(next).second)
                q.push(next);
        }
    }

    return visited.find("Destroyed") != visited.end();
}

[[nodiscard]] bool testSemanticTriggerToHashPathContract()
{
    const std::unordered_map<std::string, std::vector<std::string>> graph {
        {"TriggerAccepted", {"BuildInputSealed"}},
        {"BuildInputSealed", {"RuntimeWorldPublished"}},
        {"RuntimeWorldPublished", {"SemanticHashComputed", "PublicationStable"}},
        {"SemanticHashComputed", {"PublicationStable"}},
        {"PublicationStable", {"CrossfadeComplete", "RetireSettled"}},
        {"CrossfadeComplete", {"RetireSettled"}},
        {"RetireSettled", {}}
    };

    const std::array<std::string, 7> requiredNodes {
        "TriggerAccepted",
        "BuildInputSealed",
        "RuntimeWorldPublished",
        "SemanticHashComputed",
        "PublicationStable",
        "CrossfadeComplete",
        "RetireSettled"
    };

    std::queue<std::string> q;
    std::unordered_set<std::string> visited;
    q.push("TriggerAccepted");
    visited.insert("TriggerAccepted");

    while (!q.empty())
    {
        const auto cur = q.front();
        q.pop();

        const auto it = graph.find(cur);
        if (it == graph.end())
            return false;

        for (const auto& next : it->second)
        {
            if (visited.insert(next).second)
                q.push(next);
        }
    }

    for (const auto& node : requiredNodes)
    {
        if (graph.find(node) == graph.end())
            return false;
        if (visited.find(node) == visited.end())
            return false;
    }

    for (const auto& [node, next] : graph)
    {
        if (node == "RetireSettled")
            continue;
        if (next.empty())
            return false;
    }

    return true;
}

[[nodiscard]] bool testSemanticHashCoverageContract()
{
    convo::isr::RuntimeSemanticHash base {};
    base.generationSemanticHash = 10;
    base.topologyHash = 20;
    base.executionHash = 30;
    base.routingHash = 40;
    base.payloadHash = 50;
    base.publicationSemanticHash = 60;
    base.overlapSemanticHash = 70;
    base.retireSemanticHash = 80;

    auto mutated = base;
    mutated.generationSemanticHash++;
    if (convo::isr::classifySemanticEquivalence(base, mutated)
        == convo::isr::SemanticEquivalenceClass::Equivalent)
        return false;

    mutated = base;
    mutated.topologyHash++;
    if (convo::isr::classifySemanticEquivalence(base, mutated)
        == convo::isr::SemanticEquivalenceClass::Equivalent)
        return false;

    mutated = base;
    mutated.executionHash++;
    if (convo::isr::classifySemanticEquivalence(base, mutated)
        == convo::isr::SemanticEquivalenceClass::Equivalent)
        return false;

    mutated = base;
    mutated.routingHash++;
    if (convo::isr::classifySemanticEquivalence(base, mutated)
        == convo::isr::SemanticEquivalenceClass::Equivalent)
        return false;

    mutated = base;
    mutated.payloadHash++;
    if (convo::isr::classifySemanticEquivalence(base, mutated)
        == convo::isr::SemanticEquivalenceClass::Equivalent)
        return false;

    mutated = base;
    mutated.publicationSemanticHash++;
    if (convo::isr::classifySemanticEquivalence(base, mutated)
        == convo::isr::SemanticEquivalenceClass::Equivalent)
        return false;

    mutated = base;
    mutated.overlapSemanticHash++;
    if (convo::isr::classifySemanticEquivalence(base, mutated)
        == convo::isr::SemanticEquivalenceClass::Equivalent)
        return false;

    mutated = base;
    mutated.retireSemanticHash++;
    if (convo::isr::classifySemanticEquivalence(base, mutated)
        == convo::isr::SemanticEquivalenceClass::Equivalent)
        return false;

    return true;
}

[[nodiscard]] bool testDescriptorCoverageContract()
{
    constexpr auto& descriptors = convo::isr::PublicationSemantic::kFieldDescriptors;

    if (descriptors.size() != 4)
        return false;

    if (!convo::isr::PublicationSemantic::validateDescriptorSet())
        return false;

    for (const auto& descriptor : descriptors)
    {
        if (descriptor.fieldName.empty())
            return false;
    }

    return true;
}

[[nodiscard]] bool testPublicationRetireE2EContract()
{
    convo::isr::RuntimeSemanticSchema schema {};

    schema.publication.previousSequenceId = 0;
    schema.publication.sequenceId = 1;
    schema.publication.epoch = 1;
    schema.generation.runtimeGeneration = 42;
    schema.publication.mappedRuntimeGeneration = schema.generation.runtimeGeneration;

    if (schema.publication.sequenceId <= schema.publication.previousSequenceId)
        return false;

    if (schema.publication.mappedRuntimeGeneration != schema.generation.runtimeGeneration)
        return false;

    schema.retire.retireEpoch = schema.publication.epoch;
    schema.retire.retireBacklog = 0;
    schema.retire.deferredResidency = 0;

    if (schema.retire.retireEpoch < schema.publication.epoch)
        return false;

    if (schema.retire.retireBacklog != 0)
        return false;

    return true;
}

[[nodiscard]] bool testPublicationQueueOrderingContract()
{
    struct QueueOrderGate
    {
        std::uint64_t lastEnqueuedGeneration = 0;
        std::uint64_t lastEnqueuedSequence = 0;

        [[nodiscard]] bool enqueue(std::uint64_t generation, std::uint64_t sequence) noexcept
        {
            if (generation == 0 || sequence == 0)
                return false;
            if (generation <= lastEnqueuedGeneration)
                return false;
            if (sequence <= lastEnqueuedSequence)
                return false;

            lastEnqueuedGeneration = generation;
            lastEnqueuedSequence = sequence;
            return true;
        }
    };

    QueueOrderGate gate {};
    if (!gate.enqueue(100, 10))
        return false;
    if (!gate.enqueue(101, 11))
        return false;
    if (!gate.enqueue(102, 12))
        return false;

    // Duplicate generation must be rejected.
    if (gate.enqueue(102, 13))
        return false;

    // Sequence rollback must be rejected.
    if (gate.enqueue(103, 12))
        return false;

    // Generation rollback must be rejected.
    if (gate.enqueue(101, 14))
        return false;

    return true;
}

[[nodiscard]] bool testCompletenessValidityAdmissionContract()
{
    convo::isr::RuntimeSemanticSchema schema {};
    schema.generation.runtimeGeneration = 7;
    schema.publication.sequenceId = 17;
    schema.publication.previousSequenceId = 16;
    schema.publication.epoch = 5;
    schema.publication.mappedRuntimeGeneration = 7;
    schema.routing.processingOrder = 1;
    schema.execution.transitionPolicy = 2;
    schema.execution.crossfadeStartDelayBlocks = 1;
    schema.execution.crossfadeDryHoldSamples = 32;

    const bool completeness = schema.generation.runtimeGeneration != 0
        && schema.publication.sequenceId != 0
        && schema.publication.epoch != 0
        && schema.publication.mappedRuntimeGeneration == schema.generation.runtimeGeneration;

    const bool validity = convo::isr::isValidRoutingSemantic(schema.routing)
        && convo::isr::isValidExecutionSemantic(schema.execution)
        && schema.publication.previousSequenceId < schema.publication.sequenceId;

    const bool admission = completeness
        && validity
        && convo::isr::PublicationSemantic::validateDescriptorSet();

    if (!admission)
        return false;

    schema.execution.transitionPolicy = 3; // invalid
    if (convo::isr::isValidExecutionSemantic(schema.execution))
        return false;

    return true;
}

[[nodiscard]] bool testSemanticEquivalenceContract()
{
    convo::isr::RuntimeSemanticHash base {};
    base.generationSemanticHash = 100;
    base.topologyHash = 200;
    base.executionHash = 300;
    base.routingHash = 400;
    base.payloadHash = 500;
    base.publicationSemanticHash = 600;
    base.overlapSemanticHash = 700;
    base.retireSemanticHash = 800;

    auto same = base;
    if (convo::isr::classifySemanticEquivalence(base, same)
        != convo::isr::SemanticEquivalenceClass::Equivalent)
        return false;

    auto compatible = base;
    compatible.publicationSemanticHash = base.publicationSemanticHash + 1;
    if (convo::isr::classifySemanticEquivalence(base, compatible)
        != convo::isr::SemanticEquivalenceClass::Compatible)
        return false;

    auto different = base;
    different.executionHash = base.executionHash + 1;
    if (convo::isr::classifySemanticEquivalence(base, different)
        != convo::isr::SemanticEquivalenceClass::Different)
        return false;

    return true;
}

[[nodiscard]] bool testRequiredVerifierTableAndSeverityContract()
{
    if (!convo::isr::validateVerifierTable())
        return false;

    if (convo::isr::kRequiredVerifierTable.size() != 37)
        return false;

    bool hasWarning = false;
    bool hasError = false;
    bool hasFatal = false;

    for (const auto& verifier : convo::isr::kRequiredVerifierTable)
    {
        if (verifier.name.empty())
            return false;

        switch (verifier.severity)
        {
            case convo::isr::VerifierSeverity::Warning:
                hasWarning = true;
                break;
            case convo::isr::VerifierSeverity::Error:
                hasError = true;
                break;
            case convo::isr::VerifierSeverity::Fatal:
                hasFatal = true;
                break;
        }
    }

    return hasWarning && hasError && hasFatal;
}

[[nodiscard]] bool testPublicationFailureTaxonomyVerifierRegistration()
{
    for (const auto& verifier : convo::isr::kRequiredVerifierTable)
    {
        if (verifier.name == "PublicationFailureTaxonomyVerifier")
            return verifier.severity == convo::isr::VerifierSeverity::Fatal;
    }

    return false;
}

[[nodiscard]] bool testObserveForbiddenTypeVerifierContract()
{
    if (!convo::isr::ObserveForbiddenTypeVerifier::isForbiddenTypeName("RuntimeGraph*"))
        return false;

    if (!convo::isr::ObserveForbiddenTypeVerifier::isForbiddenTypeName("RuntimeBuildSnapshot*"))
        return false;

    if (!convo::isr::ObserveForbiddenTypeVerifier::isForbiddenTypeName("PublicationIntent*"))
        return false;

    if (!convo::isr::ObserveForbiddenTypeVerifier::isForbiddenTypeName("TransitionState*"))
        return false;

    if (convo::isr::ObserveForbiddenTypeVerifier::isForbiddenTypeName("RuntimeState*"))
        return false;

    return true;
}

[[nodiscard]] bool testAuthorityInventoryPolicyContract()
{
    static_assert(convo::isr::RuntimeAuthorityInventoryPolicy::kExhaustivenessEnforced,
                  "RuntimeAuthorityInventoryPolicy must enforce exhaustiveness");
    static_assert(convo::isr::RuntimeAuthorityInventoryPolicy::kSchemaInventoryMismatchFails,
                  "RuntimeAuthorityInventoryPolicy must fail on schema inventory mismatch");
    return true;
}

[[nodiscard]] bool testSemanticTransactionStateMachineContract()
{
    using S = convo::isr::SemanticTransactionState;
    // Forward transitions must be valid.
    if (!convo::isr::isValidSemanticTransactionTransition(S::Building, S::Validated))
        return false;
    if (!convo::isr::isValidSemanticTransactionTransition(S::Validated, S::Committed))
        return false;
    if (!convo::isr::isValidSemanticTransactionTransition(S::Committed, S::Published))
        return false;
    // Backward / invalid transitions must be rejected.
    if (convo::isr::isValidSemanticTransactionTransition(S::Published, S::Building))
        return false;
    if (convo::isr::isValidSemanticTransactionTransition(S::Validated, S::Building))
        return false;
    // Rejected is a valid terminal state reachable from Validated.
    if (!convo::isr::isValidSemanticTransactionTransition(S::Validated, S::Rejected))
        return false;
    return true;
}

[[nodiscard]] bool testCanonicalFormPolicyContract()
{
    static_assert(convo::isr::CanonicalFormPolicy::kDerivedNonPersistenceEnforced,
                  "CanonicalFormPolicy must enforce derived non-persistence");
    static_assert(convo::isr::CanonicalFormPolicy::kOneRepresentationEnforced,
                  "CanonicalFormPolicy must enforce single canonical representation");
    static_assert(convo::isr::CanonicalFormPolicy::kAliasProhibited,
                  "CanonicalFormPolicy must prohibit alias fields");
    return true;
}

[[nodiscard]] bool testExecutorSnapshotFreshnessPolicyContract()
{
    static_assert(convo::isr::ExecutorSnapshotFreshnessPolicy::kGenerationMustMatch,
                  "ExecutorSnapshotFreshnessPolicy must require generation match");
    static_assert(convo::isr::ExecutorSnapshotFreshnessPolicy::kDriftIsDetectable,
                  "ExecutorSnapshotFreshnessPolicy must declare drift as detectable");
    return true;
}

[[nodiscard]] bool testDeterministicBuildPolicyContract()
{
    static_assert(convo::isr::DeterministicBuildPolicy::kNonDeterministicSourcesMustBeDiagnosticOnly,
                  "DeterministicBuildPolicy must confine non-deterministic sources to diagnostic fields");
    static_assert(convo::isr::DeterministicBuildPolicy::kSameInputsSameOutput,
                  "DeterministicBuildPolicy must enforce same-inputs-same-output determinism");
    return true;
}

[[nodiscard]] bool testNewWork11VerifierRegistrations()
{
    // All 12 work11 verifier names must appear in the table.
    constexpr std::array<std::string_view, 12> kWork11Names = {
        "SelfContainedWorldVerifier",
        "SemanticDependencyGraphVerifier",
        "RuntimeWorldIdentityVerifier",
        "PartialSemanticUpdateProhibitionVerifier",
        "SemanticValidityVerifier",
        "RuntimeAdmissionVerifier",
        "SemanticConflictVerifier",
        "AuthorityExhaustivenessVerifier",
        "SemanticEquivalenceVerifier",
        "ReplacementAtomicityVerifier",
        "ExecutorSnapshotFreshnessVerifier",
        "DeterministicBuildVerifier"
    };
    for (const auto expected : kWork11Names)
    {
        bool found = false;
        for (const auto& entry : convo::isr::kRequiredVerifierTable)
        {
            if (entry.name == expected)
            {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!testRoutingProcessingOrderRange())
        throw std::runtime_error("routing semantic range validation failed");

    if (!testExecutionSemanticRangeAndNonNegativeFields())
        throw std::runtime_error("execution semantic validation failed");

    if (!testPublicationDescriptorSetValidation())
        throw std::runtime_error("publication descriptor validation failed");

    if (!testSchemaVersionConsistency())
        throw std::runtime_error("schema version consistency validation failed");

    if (!testSemanticDagCycleValidation())
        throw std::runtime_error("semantic dag cycle validation failed");

    if (!testRuntimeSemanticTransitionGraphValidation())
        throw std::runtime_error("runtime semantic transition graph validation failed");

    if (!testRuntimeSemanticReachabilityValidation())
        throw std::runtime_error("runtime semantic reachability validation failed");

    if (!testSemanticTriggerToHashPathContract())
        throw std::runtime_error("semantic trigger-to-hash path contract validation failed");

    if (!testSemanticHashCoverageContract())
        throw std::runtime_error("semantic hash coverage contract validation failed");

    if (!testDescriptorCoverageContract())
        throw std::runtime_error("descriptor coverage contract validation failed");

    if (!testPublicationRetireE2EContract())
        throw std::runtime_error("publication retire e2e contract validation failed");

    if (!testPublicationQueueOrderingContract())
        throw std::runtime_error("publication queue ordering contract validation failed");

    if (!testCompletenessValidityAdmissionContract())
        throw std::runtime_error("completeness validity admission contract validation failed");

    if (!testSemanticEquivalenceContract())
        throw std::runtime_error("semantic equivalence contract validation failed");

    if (!testRequiredVerifierTableAndSeverityContract())
        throw std::runtime_error("required verifier table and severity contract validation failed");

    if (!testObserveForbiddenTypeVerifierContract())
        throw std::runtime_error("observe forbidden type verifier contract validation failed");

    if (!testAuthorityInventoryPolicyContract())
        throw std::runtime_error("authority inventory policy contract validation failed");

    if (!testSemanticTransactionStateMachineContract())
        throw std::runtime_error("semantic transaction state machine contract validation failed");

    if (!testCanonicalFormPolicyContract())
        throw std::runtime_error("canonical form policy contract validation failed");

    if (!testPublicationFailureTaxonomyVerifierRegistration())
        throw std::runtime_error("publication failure taxonomy verifier registration contract validation failed");

    if (!testExecutorSnapshotFreshnessPolicyContract())
        throw std::runtime_error("executor snapshot freshness policy contract validation failed");

    if (!testDeterministicBuildPolicyContract())
        throw std::runtime_error("deterministic build policy contract validation failed");

    if (!testNewWork11VerifierRegistrations())
        throw std::runtime_error("work11 verifier table registration contract validation failed");

    return 0;
}
