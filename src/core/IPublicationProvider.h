#pragma once

#include <cstdint>

//==============================================================================
// IPublicationProvider.h — Epoch publication abstract interface.
//
// [work21 Phase-D] Separated from IRetireProvider so that the three EBR
// responsibilities (Reader, Publication, Retire) each have their own
// interface. Consumers that only need epoch advancement depend on this
// minimal interface instead of the combined IRetireProvider.
//
// This interface LIVES in src/core/ and is IMPLEMENTED by ISRRetireRouter
// (src/audioengine/), avoiding a core→audioengine dependency direction.
//==============================================================================

namespace convo {

class IPublicationProvider
{
public:
    virtual ~IPublicationProvider() = default;

    /** Advance the epoch (publish a new generation). Returns new epoch value. */
    virtual uint64_t publishEpoch() noexcept = 0;
};

} // namespace convo
