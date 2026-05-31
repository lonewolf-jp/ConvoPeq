#pragma once

#include <cstdint>

namespace convo::isr {

// Runtime state classification used by ISR governance.
// - Authoritative: semantic source of truth for runtime behavior.
// - Derived: deterministic projection from authoritative state.
// - Diagnostic: telemetry/debug only; must not drive runtime decisions.
// - ExecutorLocal: local execution detail; must not be published as semantic source.
// - Transitional: migration residue managed by the external manifest and expiry rules.
enum class AuthorityClass : std::uint8_t {
    Authoritative = 0,
    Derived,
    Diagnostic,
    ExecutorLocal
};

// Retire enqueue outcome classification for pressure-governed teardown paths.
// - Success: enqueued through the bounded retire queue.
// - QueuePressure: accepted via fallback path under pressure.
// - QueueFull: accepted at/over high fallback depth (action required by coordinator).
// - Shutdown: enqueue request rejected because shutdown phase disallows new work.
enum class RetireEnqueueResult : std::uint8_t {
    Success = 0,
    QueuePressure,
    QueueFull,
    Shutdown
};

} // namespace convo::isr
