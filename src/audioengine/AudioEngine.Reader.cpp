#include <JuceHeader.h>
#include "AudioEngine.h"
#include "ISRRetireRouter.h"

//==============================================================================
// [P0-15] Reader PR: RCU reader operations
//         (enterReader / exitReader / activeReaderCount)
//         Part of AudioEngine.Threading.cpp 3-way split.
//==============================================================================

void AudioEngine::enterRcuReader(int readerIndex) noexcept
{
    m_retireRouter->enterReader(readerIndex);
}

void AudioEngine::exitRcuReader(int readerIndex) noexcept
{
    m_retireRouter->exitReader(readerIndex);
}

[[nodiscard]] uint32_t AudioEngine::activeEpochObserverCount() const noexcept
{
    return m_retireRouter->activeReaderCount();
}
