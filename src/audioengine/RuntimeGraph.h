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

    static constexpr std::array<convo::isr::RuntimeFieldDescriptor, 25> kFieldDescriptors {{
        {"runtimeUuid", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"fadingRuntimeUuid", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"transitionCurrentRuntimeUuid", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"transitionNextRuntimeUuid", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"generation", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"activeNode", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"fadingNode", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"sampleRate", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"ditherBitDepth", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"noiseShaperType", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"oversamplingFactor", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"eqBypassed", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"convBypassed", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"softClipEnabled", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"saturationAmount", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"inputHeadroomGain", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"outputMakeupGain", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"convolverInputTrimGain", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"adaptiveCoeffBankIndex", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"adaptiveCoeffGeneration", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"captureSessionId", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"eqAgcAttackCoeffTable", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"eqAgcReleaseCoeffTable", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"eqAgcSmoothCoeffTable", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"eqAgcCoeffTableCapacity", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime}
    }};

    static constexpr std::array<convo::isr::RuntimeAuthorityInventoryEntry, 25> kAuthorityInventory {{
        {"runtimeUuid", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"fadingRuntimeUuid", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"transitionCurrentRuntimeUuid", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"transitionNextRuntimeUuid", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"generation", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"activeNode", convo::isr::RuntimeAuthorityClass::Derived},
        {"fadingNode", convo::isr::RuntimeAuthorityClass::Derived},
        {"sampleRate", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"ditherBitDepth", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"noiseShaperType", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"oversamplingFactor", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"eqBypassed", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"convBypassed", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"softClipEnabled", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"saturationAmount", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"inputHeadroomGain", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"outputMakeupGain", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"convolverInputTrimGain", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"adaptiveCoeffBankIndex", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"adaptiveCoeffGeneration", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"captureSessionId", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"eqAgcAttackCoeffTable", convo::isr::RuntimeAuthorityClass::Derived},
        {"eqAgcReleaseCoeffTable", convo::isr::RuntimeAuthorityClass::Derived},
        {"eqAgcSmoothCoeffTable", convo::isr::RuntimeAuthorityClass::Derived},
        {"eqAgcCoeffTableCapacity", convo::isr::RuntimeAuthorityClass::Derived}
    }};

    static constexpr std::array<std::string_view, 8> kDecisionRelevantFieldNames {{
        "eqBypassed",
        "convBypassed",
        "activeNode",
        "fadingNode",
        "runtimeUuid",
        "fadingRuntimeUuid",
        "transitionCurrentRuntimeUuid",
        "transitionNextRuntimeUuid"
    }};

    [[nodiscard]] static constexpr bool hasFieldDescriptor(std::string_view fieldName) noexcept
    {
        for (const auto& descriptor : kFieldDescriptors)
        {
            if (descriptor.fieldName == fieldName)
                return true;
        }
        return false;
    }

    [[nodiscard]] static constexpr bool hasAuthorityInventoryField(std::string_view fieldName) noexcept
    {
        for (const auto& entry : kAuthorityInventory)
        {
            if (entry.fieldName == fieldName)
                return true;
        }
        return false;
    }

    [[nodiscard]] static constexpr bool validateDecisionCoverageContract() noexcept
    {
        for (const auto fieldName : kDecisionRelevantFieldNames)
        {
            if (!hasFieldDescriptor(fieldName) || !hasAuthorityInventoryField(fieldName))
                return false;
        }
        return true;
    }

    [[nodiscard]] static constexpr bool validateDescriptorSet() noexcept
    {
        return convo::isr::validateFieldDescriptorSet(kFieldDescriptors)
            && convo::isr::validateAuthorityInventorySet(kAuthorityInventory)
            && convo::isr::validateAuthorityInventoryAgainstDescriptors(kAuthorityInventory, kFieldDescriptors)
            && validateDecisionCoverageContract();
    }
};

} // namespace convo
