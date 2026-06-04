#pragma once

#include <array>
#include <cstdint>
#include "ISRRuntimeSemanticSchema.h"

namespace convo {

// IMMUTABLE_RUNTIME: publish 後に変更しない Phase-1 骨格。
// 3.2.9: Projection + Diagnostic のみ。全 Authoritative フィールドは
// RuntimeWorld の対応する Semantic 構造体（topology/routing/automation/resource/coefficient）
// に移管済み。
struct RuntimeGraph
{
    // IMMUTABLE_RUNTIME: graph node pointers (visibility only, no ownership)
    // AuthorityClass::Derived (Projection)
    void* activeNode = nullptr;
    void* fadingNode = nullptr;

    // IMMUTABLE_RUNTIME: adaptive capture snapshot metadata
    // AuthorityClass::Diagnostic
    std::uint64_t captureSessionId = 0;

    // IMMUTABLE_RUNTIME: EQ AGC coefficient table snapshot view
    // AuthorityClass::Diagnostic
    const double* eqAgcAttackCoeffTable = nullptr;
    const double* eqAgcReleaseCoeffTable = nullptr;
    const double* eqAgcSmoothCoeffTable = nullptr;
    int eqAgcCoeffTableCapacity = 0;

    static constexpr std::array<convo::isr::RuntimeFieldDescriptor, 7> kFieldDescriptors {{
        {"activeNode", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"fadingNode", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"captureSessionId", convo::isr::SemanticCategory::Diagnostic, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::DiagnosticMutable, convo::isr::VisibilityClass::DiagnosticBoundary, convo::isr::LifetimeClass::DiagnosticLifetime},
        {"eqAgcAttackCoeffTable", convo::isr::SemanticCategory::Diagnostic, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::DiagnosticMutable, convo::isr::VisibilityClass::DiagnosticBoundary, convo::isr::LifetimeClass::DiagnosticLifetime},
        {"eqAgcReleaseCoeffTable", convo::isr::SemanticCategory::Diagnostic, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::DiagnosticMutable, convo::isr::VisibilityClass::DiagnosticBoundary, convo::isr::LifetimeClass::DiagnosticLifetime},
        {"eqAgcSmoothCoeffTable", convo::isr::SemanticCategory::Diagnostic, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::DiagnosticMutable, convo::isr::VisibilityClass::DiagnosticBoundary, convo::isr::LifetimeClass::DiagnosticLifetime},
        {"eqAgcCoeffTableCapacity", convo::isr::SemanticCategory::Diagnostic, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::DiagnosticMutable, convo::isr::VisibilityClass::DiagnosticBoundary, convo::isr::LifetimeClass::DiagnosticLifetime}
    }};

    static constexpr std::array<convo::isr::RuntimeAuthorityInventoryEntry, 7> kAuthorityInventory {{
        {"activeNode", convo::isr::RuntimeAuthorityClass::Derived},
        {"fadingNode", convo::isr::RuntimeAuthorityClass::Derived},
        {"captureSessionId", convo::isr::RuntimeAuthorityClass::Diagnostic},
        {"eqAgcAttackCoeffTable", convo::isr::RuntimeAuthorityClass::Diagnostic},
        {"eqAgcReleaseCoeffTable", convo::isr::RuntimeAuthorityClass::Diagnostic},
        {"eqAgcSmoothCoeffTable", convo::isr::RuntimeAuthorityClass::Diagnostic},
        {"eqAgcCoeffTableCapacity", convo::isr::RuntimeAuthorityClass::Diagnostic}
    }};

    static constexpr std::array<std::string_view, 2> kDecisionRelevantFieldNames {{
        "activeNode",
        "fadingNode"
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
