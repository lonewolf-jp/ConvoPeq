#pragma once

#include <array>
#include <cstdint>
#include "ISRAuthorityClass.h"
#include "ISRRuntimeSemanticSchema.h"

namespace convo {

// IMMUTABLE_RUNTIME: publish 後に変更しない Phase-1 骨格。
struct RuntimeGraph
{
    // IMMUTABLE_RUNTIME: runtime identity / publish generation
    // AuthorityClass::Authoritative
    std::uint64_t runtimeUuid = 0;
    std::uint64_t fadingRuntimeUuid = 0;
    std::uint64_t transitionCurrentRuntimeUuid = 0;
    std::uint64_t transitionNextRuntimeUuid = 0;
    std::uint64_t generation = 0;

    // IMMUTABLE_RUNTIME: graph node pointers (visibility only, no ownership)
    // AuthorityClass::Derived
    void* activeNode = nullptr;
    void* fadingNode = nullptr;

    // IMMUTABLE_RUNTIME: core processing metadata
    // AuthorityClass::Authoritative
    double sampleRate = 0.0;
    int ditherBitDepth = 0;
    int noiseShaperType = 0;
    int oversamplingFactor = 1;

    // IMMUTABLE_RUNTIME: processing mode flags
    bool eqBypassed = false;
    bool convBypassed = false;
    bool softClipEnabled = false;
    // IMMUTABLE_RUNTIME: gain / saturation parameters
    double saturationAmount = 0.0;
    double inputHeadroomGain = 1.0;
    double outputMakeupGain = 1.0;
    double convolverInputTrimGain = 1.0;

    // IMMUTABLE_RUNTIME: adaptive capture snapshot metadata
    int adaptiveCoeffBankIndex = -1;
    std::uint32_t adaptiveCoeffGeneration = 0;
    std::uint64_t captureSessionId = 0;

    // IMMUTABLE_RUNTIME: EQ AGC coefficient table snapshot view
    const double* eqAgcAttackCoeffTable = nullptr;
    const double* eqAgcReleaseCoeffTable = nullptr;
    const double* eqAgcSmoothCoeffTable = nullptr;
    int eqAgcCoeffTableCapacity = 0;

    static constexpr std::array<convo::isr::RuntimeFieldDescriptor, 7> kFieldDescriptors {{
        {"runtimeUuid", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"fadingRuntimeUuid", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"generation", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"activeNode", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"fadingNode", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"sampleRate", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"oversamplingFactor", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime}
    }};

    [[nodiscard]] static constexpr bool validateDescriptorSet() noexcept
    {
        return convo::isr::validateFieldDescriptorSet(kFieldDescriptors);
    }
};

} // namespace convo
