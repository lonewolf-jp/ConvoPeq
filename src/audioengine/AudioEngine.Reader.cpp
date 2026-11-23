#include <JuceHeader.h>
#include "AudioEngine.h"
#include "ISRRetireRouter.h"
#include "core/ReaderMeasureRegistry.h"   // ★ 計測 (2026-11-11)

//==============================================================================
// [P0-15] Reader PR: RCU reader operations
//         (enterReader / exitReader / activeReaderCount)
//         Part of AudioEngine.Threading.cpp 3-way split.
//==============================================================================

namespace {
// ★ 計測 (2026-11-11 Owner 限定承認): 音声 reader の最外層入退場のみを 1 区画として記録。
//   RT 側は atomic のみ（確保・mutex・出力なし）。ネストは深さで数え、二重計上しない。
thread_local std::uint32_t tlAudioReaderDepth = 0;
thread_local int tlAudioReaderMeasure = -1;
}

void AudioEngine::enterRcuReader(int readerIndex) noexcept
{
    m_retireRouter->enterReader(readerIndex);

    if (tlAudioReaderDepth++ == 0)
    {
        tlAudioReaderMeasure = convo::ReaderMeasureRegistry::claim (
            "AudioReader", static_cast<std::uint32_t>(readerIndex), 1u, this, m_retireRouter.get());
    }
}

void AudioEngine::exitRcuReader(int readerIndex) noexcept
{
    m_retireRouter->exitReader(readerIndex);

    if (tlAudioReaderDepth > 0 && --tlAudioReaderDepth == 0)
    {
        if (tlAudioReaderMeasure >= 0)
        {
            convo::ReaderMeasureRegistry::close (tlAudioReaderMeasure);
            tlAudioReaderMeasure = -1;
        }
    }
}

[[nodiscard]] uint32_t AudioEngine::activeEpochObserverCount() const noexcept
{
    return m_retireRouter->activeReaderCount();
}
