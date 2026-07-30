#pragma once

// FFTBackend.h  ── FFT Abstraction Layer (C++20 Concept)
//
// Provides:
//   - FftStatus / FftStage           type-safe enumerations
//   - FftBackendConcept               C++20 concept for FFT backends
//   - ProductionFft (Plan + RT call)  Intel IPP production backend
//   - TestFft                         injectable test backend
//
// ISR Design: Plan lifecycle is owned by Builder (NonRT),
//             RT instance calls are const & noexcept.
//
// FFT-PROD-15: ProductionFft must not hold any persistent state
//              beyond Plan (no mutable cache, no diagnostic state,
//              no internal state mutation during RT).

#include <cstdint>
#include <concepts>
#include <type_traits>

#include <ipp.h>       // IppsFFTSpec_R_64f, Ipp8u, ippsFFTFwd_RToCCS_64f, etc.

namespace convo
{

//==============================================================================
// FftStatus  ── type-safe FFT operation result
//==============================================================================
enum class FftStatus : int
{
    Ok = 0,
    InvalidArgument,
    AllocationFailure,
    BackendError,
    NotInitialized
};

/// Convert IppStatus to FftStatus (noexcept, constexpr-compatible).
constexpr FftStatus toFftStatus(IppStatus status) noexcept
{
    if (status == ippStsNoErr)          return FftStatus::Ok;
    if (status == ippStsNullPtrErr ||
        status == ippStsSizeErr  ||
        status == ippStsBadArgErr)      return FftStatus::InvalidArgument;
    if (status == ippStsMemAllocErr)    return FftStatus::AllocationFailure;
    return FftStatus::BackendError;
}

//==============================================================================
// FftStage  ── identifies which FFT call site produced an error
//==============================================================================
enum class FftStage : int
{
    Unknown              = 0,
    IrForward            = 1,
    IrInverse            = 2,
    RuntimeForwardProcess = 3,
    RuntimeInverseProcess = 4,
    RuntimeForwardAdd    = 5,
    RuntimeInverseAdd    = 6,
    TailInverse          = 7,
    Diagnostic           = 99
};

/// Safely clamp a legacy int stage to FftStage (ERRATA-V2023-2).
constexpr FftStage toFftStage(int legacyStage) noexcept
{
    if (legacyStage >= static_cast<int>(FftStage::IrForward)
        && legacyStage <= static_cast<int>(FftStage::TailInverse))
        return static_cast<FftStage>(legacyStage);
    if (legacyStage == static_cast<int>(FftStage::Diagnostic))
        return FftStage::Diagnostic;
    return FftStage::Diagnostic;   // ← unknown → safe clamp
}

//==============================================================================
// FftBackendConcept  ── static polymorphic FFT backend requirement
//
// Separates Plan (created/destroyed by Builder, NonRT) from
// const RT call (forward/invoke, noexcept).
//==============================================================================
template <class B>
concept FftBackendConcept =
    requires
    {
        typename B::Plan;
    }
    && requires(const B& b, typename B::Plan& plan, const double* in, double* out)
    {
        { B::createPlan(0) }                -> std::same_as<typename B::Plan>;
        { B::destroyPlan(plan) }            -> std::same_as<void>;
        { plan.isValid() }                  -> std::same_as<bool>;

        { b.forwardRealToCCS(plan, in, out) } noexcept -> std::same_as<FftStatus>;
        { b.inverseCCSToR(plan, in, out) }   noexcept -> std::same_as<FftStatus>;
    };

//==============================================================================
// ProductionFft  ── Intel IPP production FFT backend
//
// FFT-PROD-1:  Holds IppsFFTSpec_R_64f* (non-owning).
// FFT-PROD-2:  Plan create/destroy is NonRT only.
// FFT-PROD-3:  forward/inverse are RT-callable.
// FFT-PROD-4:  forward/inverse are noexcept.
// FFT-PROD-11: Plan::workBuffer is 64-byte aligned.
// FFT-PROD-15: No mutable cache / diagnostic state.
//==============================================================================
class ProductionFft
{
public:
    // ---- Plan (owned by Builder, NonRT) ----
    struct Plan
    {
        IppsFFTSpec_R_64f* spec       = nullptr;
        Ipp8u*             workBuffer = nullptr;  // 64-byte aligned (FFT-PROD-11)
        int                fftSize    = 0;
        int                complexSize = 0;

        [[nodiscard]] bool isValid() const noexcept { return spec != nullptr; }
    };

    static Plan createPlan(int fftSize);
    static void destroyPlan(Plan& plan) noexcept;

    // ---- RT-safe call (const, noexcept) ----
    [[nodiscard]] FftStatus forwardRealToCCS(const Plan& plan,
                                              const double* input,
                                              double* outputCCS) const noexcept;

    [[nodiscard]] FftStatus inverseCCSToR(const Plan& plan,
                                           const double* inputCCS,
                                           double* output) const noexcept;

    // Default constructor — no state (FFT-PROD-15)
    ProductionFft() = default;
    ~ProductionFft() = default;
    ProductionFft(const ProductionFft&) = delete;
    ProductionFft& operator=(const ProductionFft&) = delete;
    ProductionFft(ProductionFft&&) = default;
    ProductionFft& operator=(ProductionFft&&) = default;
};

static_assert(FftBackendConcept<ProductionFft>,
              "ProductionFft must satisfy FftBackendConcept");

//==============================================================================
// TestFft  ── injectable test backend with error injection
//
// Enables fail-closed testing of clearFFTOutputOnError() without
// relying on actual IPP failures.
//==============================================================================
class TestFft
{
public:
    // TestFft has no real Plan — dummy for concept compliance
    struct Plan
    {
        [[nodiscard]] bool isValid() const noexcept { return true; }
    };

    static Plan createPlan(int /*fftSize*/) { return Plan{}; }
    static void destroyPlan(Plan& /*plan*/) noexcept {}

    [[nodiscard]] FftStatus forwardRealToCCS(const Plan& /*plan*/,
                                              const double* /*input*/,
                                              double* /*outputCCS*/) const noexcept
    {
        return injectError_ ? FftStatus::BackendError : FftStatus::Ok;
    }

    [[nodiscard]] FftStatus inverseCCSToR(const Plan& /*plan*/,
                                           const double* /*inputCCS*/,
                                           double* /*output*/) const noexcept
    {
        return injectError_ ? FftStatus::BackendError : FftStatus::Ok;
    }

    void setInjectError(bool inject) noexcept { injectError_ = inject; }

private:
    bool injectError_ = false;
};

static_assert(FftBackendConcept<TestFft>,
              "TestFft must satisfy FftBackendConcept");

//==============================================================================
// Debug helpers (FFT-PROD-14)
//==============================================================================
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
#include <cassert>
#include <cstdint>

static void assertWorkBufferAlignment(const ProductionFft::Plan& plan) noexcept
{
    if (plan.workBuffer != nullptr)
    {
        const auto addr = reinterpret_cast<std::uintptr_t>(plan.workBuffer);
        assert((addr % 64) == 0);  // 64-byte alignment required
    }
}
#endif

} // namespace convo
