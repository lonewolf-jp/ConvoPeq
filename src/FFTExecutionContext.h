#pragma once

// FFTExecutionContext.h  ── ISR Execution Context for FFT operations
//
// ISR Design: Layer = Data Only, FFTExecutionContext = Execution Only.
// Builder owns the Plan; ExecutionContext holds a const pointer.
//
// EC-1:  ExecutionContext must NOT own the Plan.
// EC-2:  ExecutionContext must NOT extend Plan lifetime.
// EC-3:  ExecutionContext must NOT mutate Plan after Publish.
// EC-4:  ExecutionContext must NOT allocate/free/exception/mutex during RT.
// EC-5:  ExecutionContext must be Thread-safe (single RT caller).
//
// PLAN-LT-8:  ExecutionContext does NOT own the Plan. The pointer points to
//             a Plan managed by Builder (or MKLNonUniformConvolver internally).
//             Plan lifetime must exceed ExecutionContext lifetime.
// PLAN-LT-9:  forward() / inverse() must NOT be called when plan_ == nullptr.
//             Debug: assert. Release: return FftStatus::NotInitialized (fail-closed).
// PLAN-LT-10: setPlan() must only be called by Builder (NonRT) during the
//             Build phase. Must NOT be called after Publish or from RT thread.
//
// The pointer (rather than reference) design enables Builder-phase
// re-initialization:  Builder::createPlan() → ExecutionContext::setPlan()
// → Publish.  Post-Publish reassignment is forbidden (PLAN-LT-10).

#include "FFTBackend.h"

#include <cassert>

namespace convo
{

//==============================================================================
// FFTExecutionContext  ── stateless FFT executor
//
// PLAN-LT-8: Non-owning pointer to Plan managed by Builder or
//            MKLNonUniformConvolver internally.
// PLAN-LT-9: nullptr guard in forward/inverse (fail-closed in Release).
// PLAN-LT-10: setPlan() is NonRT Builder-only.
//==============================================================================
class FFTExecutionContext
{
public:
    using Plan = ProductionFft::Plan;

    /// Default constructor — plan_ == nullptr (must call setPlan() before RT).
    FFTExecutionContext() noexcept = default;

    /// Construct with a Plan reference — Plan must outlive this context.
    explicit FFTExecutionContext(const Plan& plan) noexcept
        : plan_(&plan) {}

    /// Set/replace the Plan (NonRT Builder phase only, PLAN-LT-10).
    /// assert fires if plan_ is already set (prevents post-Publish reassign).
    void setPlan(const Plan& plan) noexcept
    {
        assert(plan_ == nullptr);  // Build phase only — no post-Publish reassign
        plan_ = &plan;
    }

    // Non-copyable, movable (move preserves pointer).
    FFTExecutionContext(const FFTExecutionContext&) = delete;
    FFTExecutionContext& operator=(const FFTExecutionContext&) = delete;
    FFTExecutionContext(FFTExecutionContext&&) = default;
    FFTExecutionContext& operator=(FFTExecutionContext&&) = default;

    // ---- RT-safe operations (const, noexcept) ----

    /// Process forward FFT for a layer: real input → CCS output.
    /// PLAN-LT-9: Returns FftStatus::NotInitialized if plan_ is null.
    [[nodiscard]] FftStatus processLayerFwd(const double* fftTimeBuf,
                                              double* currentFDLSlot) const noexcept;

    /// Process inverse FFT for a layer: CCS accum → real output.
    /// PLAN-LT-9: Returns FftStatus::NotInitialized if plan_ is null.
    [[nodiscard]] FftStatus processLayerInv(const double* accumBuf,
                                              double* fftOutBuf) const noexcept;

    /// Low-level forward FFT with explicit buffers (for warmup / testing).
    /// PLAN-LT-9: Returns FftStatus::NotInitialized if plan_ is null.
    [[nodiscard]] FftStatus forwardRealToCCS(const double* input,
                                              double* outputCCS) const noexcept
    {
        if (plan_ == nullptr)
            return FftStatus::NotInitialized;
        return fft_.forwardRealToCCS(*plan_, input, outputCCS);
    }

    /// Low-level inverse FFT with explicit buffers (for warmup / testing).
    /// PLAN-LT-9: Returns FftStatus::NotInitialized if plan_ is null.
    [[nodiscard]] FftStatus inverseCCSToR(const double* inputCCS,
                                           double* output) const noexcept
    {
        if (plan_ == nullptr)
            return FftStatus::NotInitialized;
        return fft_.inverseCCSToR(*plan_, inputCCS, output);
    }

    /// Access the underlying Plan (for Builder verification).
    [[nodiscard]] const Plan& getPlan() const noexcept
    {
        assert(plan_ != nullptr);
        return *plan_;
    }

    /// Check if the held Plan is valid.
    [[nodiscard]] bool isPlanValid() const noexcept { return plan_ && plan_->isValid(); }

    /// Check if a Plan has been assigned.
    [[nodiscard]] bool hasPlan() const noexcept { return plan_ != nullptr; }

private:
    const Plan* plan_ = nullptr;  // Non-owning pointer (PLAN-LT-8)
    ProductionFft fft_;            // Stateless (FFT-PROD-15)
};

} // namespace convo
