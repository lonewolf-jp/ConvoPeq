#pragma once

#include "IReaderEpochProvider.h"
#include "IRetireProvider.h"
#include "IPublicationProvider.h"

//==============================================================================
// IEpochProvider.h — Combined EBR (Epoch-Based Reclamation) abstract interface.
//
// [work21 Phase-D] Inherits from IReaderEpochProvider (reader mgmt + epoch
// queries, 7 methods), IPublicationProvider (publishEpoch, 1 method), and
// IRetireProvider (retire operations, 2 methods).
//==============================================================================

namespace convo {

class IEpochProvider : public IReaderEpochProvider,
                       public IPublicationProvider,
                       public IRetireProvider
{
public:
    ~IEpochProvider() override = default;
};

} // namespace convo
