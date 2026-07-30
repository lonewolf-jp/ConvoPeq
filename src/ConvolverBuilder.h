#pragma once

// ConvolverBuilder.h  ── ISR Plan Builder (NonRT Authority)
//
// ISR Design: Builder is the sole authority for Plan creation/destruction.
// FFTExecutionContext receives a const Plan& and must NOT own or extend
// the Plan's lifetime (EC-1, EC-2, PLAN-LT-1〜7).
//
// Builder → Plan → FFTExecutionContext → Layer (Data Only)

#include "FFTBackend.h"
#include "FFTExecutionContext.h"

#include <memory>
#include <vector>

namespace convo
{

//==============================================================================
// ConvolverBuilder  ── Non-RT Plan factory
//
// PLAN-LT-1:  Only Builder creates/destroys Plans.
// PLAN-LT-2:  Builder creates Plan via createPlan() and injects it into
//             FFTExecutionContext.
// PLAN-LT-5:  Builder calls destroyPlan() after all ExecutionContexts
//             have released their references.
// PLAN-LT-7:  Builder must ensure all ExecutionContext instances release
//             the Plan before calling destroyPlan().
//==============================================================================
class ConvolverBuilder
{
public:
    using Plan = ProductionFft::Plan;

    ConvolverBuilder() = default;
    ~ConvolverBuilder() = default;

    // Non-copyable, movable
    ConvolverBuilder(const ConvolverBuilder&) = delete;
    ConvolverBuilder& operator=(const ConvolverBuilder&) = delete;
    ConvolverBuilder(ConvolverBuilder&&) = default;
    ConvolverBuilder& operator=(ConvolverBuilder&&) = default;

    // ---- Plan lifecycle (NonRT only) ----

    /// Create a new FFT Plan for the given FFT size.
    /// Returns valid Plan on success, invalid Plan (isValid()==false) on failure.
    [[nodiscard]] Plan createPlan(int fftSize)
    {
        return ProductionFft::createPlan(fftSize);
    }

    /// Destroy a Plan and release all associated resources.
    /// Must only be called after all ExecutionContexts holding a reference
    /// to this Plan have been destroyed (PLAN-LT-5).
    void destroyPlan(Plan& plan) noexcept
    {
        ProductionFft::destroyPlan(plan);
    }

    /// Create an FFTExecutionContext bound to the given Plan.
    /// The Plan must outlive the returned ExecutionContext (PLAN-LT-3).
    [[nodiscard]] FFTExecutionContext createExecutionContext(const Plan& plan) noexcept
    {
        return FFTExecutionContext(plan);
    }

    // ---- Convenience: create multiple layer plans ----

    /// Create plans for all layers of a non-uniform convolver.
    /// Returns vector of plans corresponding to layer FFT sizes.
    [[nodiscard]] std::vector<Plan> createLayerPlans(
        const int* fftSizes, int numLayers)
    {
        std::vector<Plan> plans;
        plans.reserve(static_cast<size_t>(numLayers));
        for (int i = 0; i < numLayers; ++i)
        {
            plans.push_back(createPlan(fftSizes[i]));
        }
        return plans;
    }

    /// Destroy all plans in a vector.
    void destroyAllPlans(std::vector<Plan>& plans) noexcept
    {
        for (auto& plan : plans)
        {
            if (plan.isValid())
                destroyPlan(plan);
        }
        plans.clear();
    }
};

} // namespace convo
