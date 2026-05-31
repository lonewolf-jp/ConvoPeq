#pragma once

#include "PreparedIRState.h"
#include "SafeStateSwapper.h"

using ConvolverIRPayload = PreparedIRState;

namespace convo
{
    using RuntimeStateSwapper = SafeStateSwapper;
}
