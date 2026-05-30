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

struct PublicationSemantic
{
    PublicationSequenceId sequenceId = 0;
    PublicationEpoch epoch = 0;
    std::uint64_t mappedRuntimeGeneration = 0;
    std::uint64_t previousSequenceId = 0;

    static constexpr std::array<RuntimeFieldDescriptor, 4> kFieldDescriptors {{
        {"sequenceId", SemanticCategory::Authority, OwnershipClass::PublicationSemantic, MutabilityClass::MutablePrePublish, VisibilityClass::PublicationBoundary, LifetimeClass::RuntimeWorldLifetime},
        {"epoch", SemanticCategory::Authority, OwnershipClass::PublicationSemantic, MutabilityClass::MutablePrePublish, VisibilityClass::PublicationBoundary, LifetimeClass::RuntimeWorldLifetime},
        {"mappedRuntimeGeneration", SemanticCategory::Authority, OwnershipClass::PublicationSemantic, MutabilityClass::MutablePrePublish, VisibilityClass::PublicationBoundary, LifetimeClass::RuntimeWorldLifetime},
        {"previousSequenceId", SemanticCategory::Authority, OwnershipClass::PublicationSemantic, MutabilityClass::MutablePrePublish, VisibilityClass::PublicationBoundary, LifetimeClass::RuntimeWorldLifetime}
    }};

    [[nodiscard]] static constexpr bool validateDescriptorSet() noexcept
    {
        return validateFieldDescriptorSet(kFieldDescriptors);
    }
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
};

struct CoefficientSemantic
{
    int adaptiveCoeffBankIndex = -1;
    std::uint32_t adaptiveCoeffGeneration = 0;
    std::uint64_t eqCoeffHash = 0;
};

struct RuntimeSemanticSchema
{
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

} // namespace convo::isr
