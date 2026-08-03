#pragma once

// A3 / ADR-D3: The canonical RuntimeWorldAuthority (ISR Authority Surface) is defined in
//   RuntimeWorldAuthority.h with the full Step 5-3 D3 surface — i.e. PendingPublishRegistry
//   plus the registry() accessor used by the ISR PublishExecutor at commit time.
//   This header forwards to that definition so every ISR Authority consumer — AudioEngine
//   (owner of the authority) and RuntimePublishExecutor (resolver of the gap registry at
//   commit) — observes a single, D3-complete RuntimeWorldAuthority, eliminating the pre-D3
//   stub divergence where the authority lacked the gap registry.
#include "RuntimeWorldAuthority.h"
