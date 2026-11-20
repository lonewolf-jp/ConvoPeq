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
    // ★ WORK113 R1: この値は shutdown trace の SH4（observerCount）に入り、
    //   verified の判定に使われる。exit 実行中の reader は depth 0 だが
    //   まだ epoch を手放していないので、活動数（exit 中を含む）を返す。
    //   診断の件数が必要な場合は activeDepthReaderCount() を直接使うこと。
    return m_retireRouter->activeReaderCount();
}
