#include <JuceHeader.h>
#include "AudioEngine.h"

//==============================================================================
// [P0-15] Publication PR: Epoch publication operations
//         (publish / current / advanceEpoch) → Router::publishEpoch()
//         Part of AudioEngine.Threading.cpp 3-way split.
//==============================================================================

[[nodiscard]] uint64_t AudioEngine::snapshotRcuEpoch() noexcept
{
    return currentRetireEpoch();
}

[[nodiscard]] uint64_t AudioEngine::markRetireEpoch() noexcept
{
    return m_retireRouter->publishEpoch();
}

[[nodiscard]] uint64_t AudioEngine::currentRetireEpoch() const noexcept
{
    return m_retireRouter->currentEpoch();
}

uint64_t AudioEngine::advanceRetireEpoch() noexcept
{
    return m_retireRouter->publishEpoch();
}
