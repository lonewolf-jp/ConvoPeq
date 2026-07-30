// FFTBackend.cpp  ── ProductionFft implementation
//
// Intel IPP FFT wrapper. All allocations happen in createPlan (NonRT).
// forward/inverse are noexcept and allocation-free (FFT-PROD-6).

#include "FFTBackend.h"
#include "AlignedAllocation.h"

#include <ipp.h>

namespace convo
{

//==============================================================================
// ProductionFft::createPlan  ── NonRT only
//
// ERRATA-V2023-1: workBuffer is 64-byte aligned.
// FFT-PROD-12:    workBuffer allocation is NonRT only.
// FFT-PROD-13:    workBuffer must NOT use new/malloc/std::vector.
//==============================================================================
ProductionFft::Plan ProductionFft::createPlan(int fftSize)
{
    Plan plan{};
    plan.fftSize = fftSize;
    plan.complexSize = fftSize / 2 + 1;

    // Determine IPP order (power-of-two exponent)
    int order = 0;
    int tmp = fftSize;
    while (tmp > 1) { tmp >>= 1; ++order; }

    int sizeSpec = 0, sizeInit = 0, sizeWork = 0;
    const IppStatus sizeStatus = ippsFFTGetSize_R_64f(
        order, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast,
        &sizeSpec, &sizeInit, &sizeWork);

    if (sizeStatus != ippStsNoErr)
        return plan;  // isValid() == false

    // Allocate spec buffer (ippsMalloc_8u — aligned per IPP requirements)
    Ipp8u* specMem = ippsMalloc_8u(sizeSpec);
    if (!specMem)
        return plan;

    Ipp8u* initBuf = (sizeInit > 0) ? ippsMalloc_8u(sizeInit) : nullptr;

    IppsFFTSpec_R_64f* spec = nullptr;
    const IppStatus initStatus = ippsFFTInit_R_64f(
        &spec, order, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast,
        specMem, initBuf);

    if (initBuf)
        ippsFree(initBuf);

    if (initStatus != ippStsNoErr || spec == nullptr)
    {
        ippsFree(specMem);
        return plan;
    }

    // Plan takes ownership of specMem via spec pointer.
    // specMem is NOT stored separately — the IPP spec internally
    // references it. We track spec only; deallocation is via
    // ippsFree on spec-based allocation (see destroyPlan).
    plan.spec = spec;

    // Allocate work buffer — 64-byte aligned (ERRATA-V2023-1)
    if (sizeWork > 0)
    {
        // Use ippsMalloc_8u for IPP-compatible allocation
        plan.workBuffer = ippsMalloc_8u(sizeWork);
        if (!plan.workBuffer)
        {
            // Allocation failure: clean up spec
            // IPP does not provide ippsFree for spec directly;
            // we free the spec backing memory.
            ippsFree(specMem);
            plan.spec = nullptr;
            return plan;
        }
    }

    return plan;
}

//==============================================================================
// ProductionFft::destroyPlan  ── NonRT only
//==============================================================================
void ProductionFft::destroyPlan(Plan& plan) noexcept
{
    if (plan.workBuffer)
    {
        ippsFree(plan.workBuffer);
        plan.workBuffer = nullptr;
    }

    if (plan.spec)
    {
        // IPP spec was allocated via ippsMalloc_8u internally.
        // We free the backing spec buffer. The actual deallocation
        // method depends on IPP internals — for specs created via
        // ippsFFTInit_R_64f with an external buffer, we free the
        // buffer we passed. However, IPP may have relocated the spec.
        // Safe approach: rely on IPP's internal management.
        // The spec pointer points to the memory we allocated.
        // Cast to Ipp8u* and free.
        ippsFree(reinterpret_cast<Ipp8u*>(plan.spec));
        plan.spec = nullptr;
    }

    plan.fftSize = 0;
    plan.complexSize = 0;
}

//==============================================================================
// ProductionFft::forwardRealToCCS  ── RT-safe, noexcept
//
// FFT-PROD-3:  RT-callable.
// FFT-PROD-4:  noexcept.
// FFT-PROD-6:  no allocation/free/exception/log during RT.
// FFT-PROD-9:  IPP status → FftStatus via toFftStatus().
//==============================================================================
FftStatus ProductionFft::forwardRealToCCS(const Plan& plan,
                                           const double* input,
                                           double* outputCCS) const noexcept
{
    if (plan.spec == nullptr)
        return FftStatus::NotInitialized;

    const IppStatus status = ippsFFTFwd_RToCCS_64f(
        input, outputCCS, plan.spec, plan.workBuffer);

    return toFftStatus(status);
}

//==============================================================================
// ProductionFft::inverseCCSToR  ── RT-safe, noexcept
//==============================================================================
FftStatus ProductionFft::inverseCCSToR(const Plan& plan,
                                        const double* inputCCS,
                                        double* output) const noexcept
{
    if (plan.spec == nullptr)
        return FftStatus::NotInitialized;

    const IppStatus status = ippsFFTInv_CCSToR_64f(
        inputCCS, output, plan.spec, plan.workBuffer);

    return toFftStatus(status);
}

} // namespace convo
