#pragma once

namespace MKLRealTime {

// MUST be called before any MKL DFTI usage.
// Safe to call multiple times (call_once ensures single execution).
void setup() noexcept;

} // namespace MKLRealTime
