#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace convo::isr {

inline constexpr std::uint32_t kRuntimeSemanticSchemaVersion = 7u;

enum class SemanticCategory : std::uint8_t
{
    Authority = 0,
    Derived,
    Diagnostic,
    Telemetry,
    Cache
};

enum class OwnershipClass : std::uint8_t
{
    RuntimePublicationCoordinator = 0,
    RuntimeWorld,
    RuntimeGraph,
    PublicationSemantic,
    ObserveLocal,
    DiagnosticOnly
};

enum class MutabilityClass : std::uint8_t
{
    ImmutableAfterPublish = 0,
    MutablePrePublish,
    ObserveLocalMutable,
    DiagnosticMutable
};

enum class VisibilityClass : std::uint8_t
{
    PublicationBoundary = 0,
    ObserveBoundary,
    DiagnosticBoundary
};

enum class LifetimeClass : std::uint8_t
{
    RuntimeWorldLifetime = 0,
    ProcessLifetime,
    ObserveLocalLifetime,
    DiagnosticLifetime
};

struct RuntimeFieldDescriptor
{
    std::string_view fieldName {};
    SemanticCategory semanticCategory = SemanticCategory::Derived;
    OwnershipClass ownership = OwnershipClass::DiagnosticOnly;
    MutabilityClass mutability = MutabilityClass::DiagnosticMutable;
    VisibilityClass visibility = VisibilityClass::DiagnosticBoundary;
    LifetimeClass lifetime = LifetimeClass::DiagnosticLifetime;
};

enum class RuntimeAuthorityClass : std::uint8_t
{
    Authoritative = 0,
    Derived,
    Diagnostic,
    ExecutorLocal
};

struct RuntimeAuthorityInventoryEntry
{
    std::string_view fieldName {};
    RuntimeAuthorityClass authorityClass = RuntimeAuthorityClass::Diagnostic;
};

template <std::size_t N>
[[nodiscard]] inline constexpr bool validateFieldDescriptorSet(const std::array<RuntimeFieldDescriptor, N>& descriptors) noexcept
{
    if constexpr (N == 0)
        return false;

    for (std::size_t i = 0; i < N; ++i)
    {
        if (descriptors[i].fieldName.empty())
            return false;

        for (std::size_t j = i + 1; j < N; ++j)
        {
            if (descriptors[i].fieldName == descriptors[j].fieldName)
                return false;
        }
    }

    return true;
}

template <std::size_t N>
[[nodiscard]] inline constexpr bool validateAuthorityInventorySet(const std::array<RuntimeAuthorityInventoryEntry, N>& inventory) noexcept
{
    if constexpr (N == 0)
        return false;

    for (std::size_t i = 0; i < N; ++i)
    {
        if (inventory[i].fieldName.empty())
            return false;

        for (std::size_t j = i + 1; j < N; ++j)
        {
            if (inventory[i].fieldName == inventory[j].fieldName)
                return false;
        }
    }

    return true;
}

template <std::size_t InventoryN, std::size_t DescriptorN>
[[nodiscard]] inline constexpr bool validateAuthorityInventoryAgainstDescriptors(
    const std::array<RuntimeAuthorityInventoryEntry, InventoryN>& inventory,
    const std::array<RuntimeFieldDescriptor, DescriptorN>& descriptors) noexcept
{
    if (!validateAuthorityInventorySet(inventory) || !validateFieldDescriptorSet(descriptors))
        return false;

    for (const auto& descriptor : descriptors)
    {
        bool found = false;
        for (const auto& entry : inventory)
        {
            if (entry.fieldName == descriptor.fieldName)
            {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }

    for (const auto& entry : inventory)
    {
        bool found = false;
        for (const auto& descriptor : descriptors)
        {
            if (descriptor.fieldName == entry.fieldName)
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

template <std::size_t N>
[[nodiscard]] inline constexpr bool validateReadAuthorityInventorySet(
    const std::array<RuntimeAuthorityInventoryEntry, N>& inventory) noexcept
{
    return validateAuthorityInventorySet(inventory);
}

template <std::size_t InventoryN, std::size_t DescriptorN>
[[nodiscard]] inline constexpr bool validateReadAuthorityInventoryAgainstDescriptors(
    const std::array<RuntimeAuthorityInventoryEntry, InventoryN>& readInventory,
    const std::array<RuntimeFieldDescriptor, DescriptorN>& descriptors) noexcept
{
    if (!validateReadAuthorityInventorySet(readInventory) || !validateFieldDescriptorSet(descriptors))
        return false;

    for (const auto& entry : readInventory)
    {
        bool found = false;
        for (const auto& descriptor : descriptors)
        {
            if (entry.fieldName == descriptor.fieldName)
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

using PublicationSequenceId = std::uint64_t;
using PublicationEpoch = std::uint64_t;

struct GenerationSemantic
{
    std::uint64_t runtimeGeneration = 0;
    std::uint64_t activationEpoch = 0;
};

struct TopologySemantic
{
    std::uint64_t runtimeUuid = 0;
    std::uint64_t fadingRuntimeUuid = 0;
    bool hasFadingRuntime = false;
};

struct RoutingSemantic
{
    int processingOrder = 0;
    bool eqBypassed = false;
    bool convBypassed = false;
};

struct ExecutionSemantic
{
    bool transitionActive = false;
    int transitionPolicy = 0;
    int latencyCompensationSamples = 0;
    int crossfadeStartDelayBlocks = 0;
    int crossfadeDryHoldSamples = 0;
};

[[nodiscard]] inline constexpr bool isValidRoutingSemantic(const RoutingSemantic& routing) noexcept
{
    constexpr int kMinProcessingOrder = 0;
    constexpr int kMaxProcessingOrder = 1;
    return routing.processingOrder >= kMinProcessingOrder
        && routing.processingOrder <= kMaxProcessingOrder;
}

[[nodiscard]] inline constexpr bool isValidExecutionSemantic(const ExecutionSemantic& execution) noexcept
{
    constexpr int kMinTransitionPolicy = 0;
    constexpr int kMaxTransitionPolicy = 2;
    if (execution.transitionPolicy < kMinTransitionPolicy
        || execution.transitionPolicy > kMaxTransitionPolicy)
        return false;

    if (execution.crossfadeStartDelayBlocks < 0
        || execution.crossfadeDryHoldSamples < 0)
        return false;

    return true;
}

struct PublicationSemantic
{
    PublicationSequenceId sequenceId = 0;
    PublicationEpoch epoch = 0;
    std::uint64_t mappedRuntimeGeneration = 0;
    std::uint64_t previousSequenceId = 0;

    static constexpr std::array<RuntimeFieldDescriptor, 4> kFieldDescriptors {{
        {"sequenceId", SemanticCategory::Authority, OwnershipClass::PublicationSemantic, MutabilityClass::MutablePrePublish, VisibilityClass::PublicationBoundary, LifetimeClass::RuntimeWorldLifetime},
        {"epoch", SemanticCategory::Authority, OwnershipClass::PublicationSemantic, MutabilityClass::MutablePrePublish, VisibilityClass::PublicationBoundary, LifetimeClass::RuntimeWorldLifetime},
        {"mappedRuntimeGeneration", SemanticCategory::Derived, OwnershipClass::PublicationSemantic, MutabilityClass::MutablePrePublish, VisibilityClass::PublicationBoundary, LifetimeClass::RuntimeWorldLifetime},
        {"previousSequenceId", SemanticCategory::Authority, OwnershipClass::PublicationSemantic, MutabilityClass::MutablePrePublish, VisibilityClass::PublicationBoundary, LifetimeClass::RuntimeWorldLifetime}
    }};

    [[nodiscard]] static constexpr bool validateDescriptorSet() noexcept
    {
        return validateFieldDescriptorSet(kFieldDescriptors);
    }
};

struct RuntimeMetadata
{
    std::uint32_t schemaVersion = kRuntimeSemanticSchemaVersion;
    PublicationSequenceId publicationSequence = 0;
};

struct OverlapSemantic
{
    bool useDryAsOld = false;
    bool firstIrDryCrossfadePending = false;
    double dryScaleTarget = 1.0;
    double fadeTimeSec = 0.0;
};

struct RetireSemantic
{
    std::uint64_t retireEpoch = 0;
    std::uint64_t retireBacklog = 0;
    std::uint64_t deferredResidency = 0;
};

struct TimingSemantic
{
    double sampleRateHz = 0.0;
    double queuedFadeTimeSec = 0.0;
    std::uint64_t activationEpoch = 0;
};

struct LatencySemantic
{
    int latencyDelayOld = 0;
    int latencyDelayNew = 0;
    int latencyDeltaSamples = 0;
};

struct SchedulingSemantic
{
    bool transitionActive = false;
    int crossfadeStartDelayBlocks = 0;
    int crossfadeDryHoldSamples = 0;
};

struct ResourceSemantic
{
    int oversamplingFactor = 1;
    int ditherBitDepth = 0;
    int noiseShaperType = 0;
};

struct AffinitySemantic
{
    bool rebuildWorkerRunning = false;
};

struct AutomationSemantic
{
    bool eqBypassed = false;
    bool convBypassed = false;
    bool softClipEnabled = false;
    double saturationAmount = 0.0;
    double inputHeadroomGain = 1.0;
    double outputMakeupGain = 1.0;
    double convolverInputTrimGain = 1.0;
};

struct CoefficientSemantic
{
    int adaptiveCoeffBankIndex = -1;
    std::uint32_t adaptiveCoeffGeneration = 0;
    std::uint64_t eqCoeffHash = 0;
};

struct RuntimeSemanticSchema
{
    RuntimeMetadata metadata {};
    GenerationSemantic generation {};
    TopologySemantic topology {};
    RoutingSemantic routing {};
    ExecutionSemantic execution {};
    PublicationSemantic publication {};
    OverlapSemantic overlap {};
    RetireSemantic retire {};
    TimingSemantic timing {};
    LatencySemantic latency {};
    SchedulingSemantic scheduling {};
    ResourceSemantic resource {};
    AffinitySemantic affinity {};
    AutomationSemantic automation {};
    CoefficientSemantic coefficient {};
};

struct ProjectionFreshness
{
    std::uint64_t projectionGeneration = 0;
    std::uint64_t projectionRevision = 0;
    std::uint32_t maxStalenessWindows = 1;
};

struct RuntimeSemanticHash
{
    std::uint64_t generationSemanticHash = 0;
    std::uint64_t topologyHash = 0;
    std::uint64_t executionHash = 0;
    std::uint64_t routingHash = 0;
    std::uint64_t payloadHash = 0;
    std::uint64_t publicationSemanticHash = 0;
    std::uint64_t overlapSemanticHash = 0;
    std::uint64_t retireSemanticHash = 0;
};

enum class SemanticEquivalenceClass : std::uint8_t
{
    Equivalent = 0,
    Compatible,
    Different
};

[[nodiscard]] inline constexpr SemanticEquivalenceClass classifySemanticEquivalence(
    const RuntimeSemanticHash& lhs,
    const RuntimeSemanticHash& rhs) noexcept
{
    if (lhs.generationSemanticHash == rhs.generationSemanticHash
        && lhs.topologyHash == rhs.topologyHash
        && lhs.executionHash == rhs.executionHash
        && lhs.routingHash == rhs.routingHash
        && lhs.payloadHash == rhs.payloadHash
        && lhs.publicationSemanticHash == rhs.publicationSemanticHash
        && lhs.overlapSemanticHash == rhs.overlapSemanticHash
        && lhs.retireSemanticHash == rhs.retireSemanticHash)
        return SemanticEquivalenceClass::Equivalent;

    if (lhs.generationSemanticHash == rhs.generationSemanticHash
        && lhs.topologyHash == rhs.topologyHash
        && lhs.executionHash == rhs.executionHash
        && lhs.routingHash == rhs.routingHash
        && lhs.payloadHash == rhs.payloadHash)
        return SemanticEquivalenceClass::Compatible;

    return SemanticEquivalenceClass::Different;
}

enum class VerifierSeverity : std::uint8_t
{
    Warning = 0,
    Error,
    Fatal
};

struct VerifierDescriptor
{
    std::string_view name {};
    VerifierSeverity severity = VerifierSeverity::Error;
};

inline constexpr std::array<VerifierDescriptor, 37> kRequiredVerifierTable {{
    {"SchemaCompletenessVerifier", VerifierSeverity::Fatal},
    {"PartialPublicationVerifier", VerifierSeverity::Fatal},
    {"PublicationSingleSourceVerifier", VerifierSeverity::Fatal},
    {"ObservePathVerifier", VerifierSeverity::Fatal},
    {"OverlapAuthorityVerifier", VerifierSeverity::Fatal},
    {"RetirePressureVerifier", VerifierSeverity::Error},
    {"RetireStarvationVerifier", VerifierSeverity::Error},
    {"SemanticHashDriftVerifier", VerifierSeverity::Error},
    {"ShadowCompareContractVerifier", VerifierSeverity::Error},
    {"ImmutableWorldVerifier", VerifierSeverity::Fatal},
    {"GenerationMonotonicityVerifier", VerifierSeverity::Fatal},
    {"VisibilityMonotonicityVerifier", VerifierSeverity::Fatal},
    {"PublicationOrderingVerifier", VerifierSeverity::Fatal},
    {"BuildIsolationVerifier", VerifierSeverity::Fatal},
    {"RetireSafetyVerifier", VerifierSeverity::Fatal},
    {"ObserveForbiddenTypeVerifier", VerifierSeverity::Fatal},
    {"PublicationFailureTaxonomyVerifier", VerifierSeverity::Fatal},
    {"MemoryOrderingContractVerifier", VerifierSeverity::Fatal},
    {"OwnershipTransferContractVerifier", VerifierSeverity::Fatal},
    {"ABAHazardVerifier", VerifierSeverity::Error},
    {"RTBoundaryVerifier", VerifierSeverity::Fatal},
    {"HiddenAuthorityVerifier", VerifierSeverity::Fatal},
    {"SemanticAliasVerifier", VerifierSeverity::Warning},
    {"MultiWriterProhibitionVerifier", VerifierSeverity::Fatal},
    {"RetireEscalationVerifier", VerifierSeverity::Fatal},
    // §5.4.1 work11: 12 verifiers adopted in 2026-05-31 revision
    {"SelfContainedWorldVerifier", VerifierSeverity::Fatal},
    {"SemanticDependencyGraphVerifier", VerifierSeverity::Fatal},
    {"RuntimeWorldIdentityVerifier", VerifierSeverity::Fatal},
    {"PartialSemanticUpdateProhibitionVerifier", VerifierSeverity::Fatal},
    {"SemanticValidityVerifier", VerifierSeverity::Fatal},
    {"RuntimeAdmissionVerifier", VerifierSeverity::Fatal},
    {"SemanticConflictVerifier", VerifierSeverity::Fatal},
    {"AuthorityExhaustivenessVerifier", VerifierSeverity::Fatal},
    {"SemanticEquivalenceVerifier", VerifierSeverity::Error},
    {"ReplacementAtomicityVerifier", VerifierSeverity::Fatal},
    {"ExecutorSnapshotFreshnessVerifier", VerifierSeverity::Error},
    {"DeterministicBuildVerifier", VerifierSeverity::Error}
}};

[[nodiscard]] inline constexpr bool validateVerifierTable() noexcept
{
    for (std::size_t i = 0; i < kRequiredVerifierTable.size(); ++i)
    {
        if (kRequiredVerifierTable[i].name.empty())
            return false;

        for (std::size_t j = i + 1; j < kRequiredVerifierTable.size(); ++j)
        {
            if (kRequiredVerifierTable[i].name == kRequiredVerifierTable[j].name)
                return false;
        }
    }

    return true;
}

static_assert(kRequiredVerifierTable.size() == 37,
              "Verifier registry contract requires exactly 37 verifier entries.");
static_assert(validateVerifierTable(),
              "Verifier registry contract violation: duplicate or empty verifier names are forbidden.");

struct ObserveForbiddenTypeVerifier
{
    static constexpr std::array<std::string_view, 4> kForbiddenTypeNames {{
        "RuntimeGraph*",
        "RuntimeBuildSnapshot*",
        "PublicationIntent*",
        "TransitionState*"
    }};

    [[nodiscard]] static constexpr bool isForbiddenTypeName(std::string_view typeName) noexcept
    {
        for (const auto forbidden : kForbiddenTypeNames)
        {
            if (forbidden == typeName)
                return true;
        }

        return false;
    }
};

// §3.1 RuntimeAuthorityInventory: exhaustiveness enforcement policy.
// §3.1.1 Schema-inventory consistency: any unclassified authority field is a build-fail.
struct RuntimeAuthorityInventoryPolicy
{
    static constexpr bool kExhaustivenessEnforced = true;
    static constexpr bool kSchemaInventoryMismatchFails = true;
};

// §3.19.5 Semantic Transaction state machine.
// Permitted transitions:
//   Building -> Validated -> Committed -> Published
//   Building | Validated | Committed  -> Rejected (terminal)
// Published and Rejected are terminal states.
enum class SemanticTransactionState : std::uint8_t
{
    Building = 0,
    Validated,
    Committed,
    Published,
    Rejected
};

[[nodiscard]] inline constexpr bool isValidSemanticTransactionTransition(
    SemanticTransactionState from, SemanticTransactionState to) noexcept
{
    switch (from)
    {
        case SemanticTransactionState::Building:
            return to == SemanticTransactionState::Validated
                || to == SemanticTransactionState::Rejected;
        case SemanticTransactionState::Validated:
            return to == SemanticTransactionState::Committed
                || to == SemanticTransactionState::Rejected;
        case SemanticTransactionState::Committed:
            return to == SemanticTransactionState::Published
                || to == SemanticTransactionState::Rejected;
        case SemanticTransactionState::Published:
            return false; // terminal: immutable published world
        case SemanticTransactionState::Rejected:
            return false; // terminal: rejected transaction discarded
        default:
            return false;
    }
}

// §3.19.1 Canonical Semantic Form policy.
// §3.19.2 Derived Semantic Non-Persistence: derived fields must not persist as authority.
struct CanonicalFormPolicy
{
    static constexpr bool kDerivedNonPersistenceEnforced = true;
    static constexpr bool kOneRepresentationEnforced = true;
    static constexpr bool kAliasProhibited = true;
};

// §3.17.2 Executor Snapshot Freshness Contract.
// executorSnapshotGeneration == worldGeneration is the validity criterion.
struct ExecutorSnapshotFreshnessPolicy
{
    static constexpr bool kGenerationMustMatch = true;
    static constexpr bool kDriftIsDetectable = true;
};

// §3.17.3 Construction Determinism Contract.
// Same semantic inputs must produce the same canonical world.
// Non-deterministic sources must be isolated to Diagnostic fields.
struct DeterministicBuildPolicy
{
    static constexpr bool kNonDeterministicSourcesMustBeDiagnosticOnly = true;
    static constexpr bool kSameInputsSameOutput = true;
};

} // namespace convo::isr
