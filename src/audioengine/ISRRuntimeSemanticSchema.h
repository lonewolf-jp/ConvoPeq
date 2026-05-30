#pragma once

#include <cstdint>

namespace convo::isr {

inline constexpr std::uint32_t kRuntimeSemanticSchemaVersion = 7u;

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
