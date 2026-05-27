#include "ConvolverState.h"

namespace convo {

std::atomic<uint64_t> ConvolverState::stateIdCounterStorage_ { 0 };

std::atomic<uint64_t>& ConvolverState::stateIdCounter() noexcept
{
    return stateIdCounterStorage_;
}

uint64_t ConvolverState::generateNewStateId() noexcept
{
    return convo::fetchAddAtomic(stateIdCounter(),
                                 static_cast<uint64_t>(1),
                                 std::memory_order_acq_rel) + 1; // acq_rel: acquire で前回の fetchAdd と HB し単調増加を確認; release で stateId 比較スレッドの acquire と HB
}

} // namespace convo
