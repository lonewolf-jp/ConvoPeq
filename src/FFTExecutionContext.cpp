// FFTExecutionContext.cpp  ── ExecutionContext implementation (Layer-independent stubs)

#include "FFTExecutionContext.h"

namespace convo
{

FftStatus FFTExecutionContext::processLayerFwd(
    const double* fftTimeBuf, double* currentFDLSlot) const noexcept
{
    if (plan_ == nullptr)
        return FftStatus::NotInitialized;
    return fft_.forwardRealToCCS(*plan_, fftTimeBuf, currentFDLSlot);
}

FftStatus FFTExecutionContext::processLayerInv(
    const double* accumBuf, double* fftOutBuf) const noexcept
{
    if (plan_ == nullptr)
        return FftStatus::NotInitialized;
    return fft_.inverseCCSToR(*plan_, accumBuf, fftOutBuf);
}

} // namespace convo
