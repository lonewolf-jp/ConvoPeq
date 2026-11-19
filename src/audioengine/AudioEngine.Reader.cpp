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
    // ★ WORK113 R1: 診断用の件数なので depth のみ（activity は exit 中を二重計上する）。
    return m_retireRouter->activeDepthReaderCount();
}
