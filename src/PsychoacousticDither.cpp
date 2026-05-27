#include "PsychoacousticDither.h"

namespace convo {

std::atomic<uint64_t> PsychoacousticDither::instanceSeedCounterStorage_ { 0 };

std::atomic<uint64_t>& PsychoacousticDither::instanceSeedCounter() noexcept
{
    return instanceSeedCounterStorage_;
}

} // namespace convo
