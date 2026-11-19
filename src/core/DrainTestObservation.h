#pragma once

// \u2605 WORK113 DQ-V2 v2 (2026-11-19 Owner \u627f\u8a8d): \u30c6\u30b9\u30c8\u5c02\u7528\u306e\u7834\u68c4\u89b3\u6e2c\u3002
//
//   CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY \u3092\u5b9a\u7fa9\u3059\u308b\u30c6\u30b9\u30c8\u30bf\u30fc\u30b2\u30c3\u30c8\u3060\u3051\u304c include \u3059\u308b\u3002
//   \u88fd\u54c1\u30d3\u30eb\u30c9\u3067\u306f include \u81ea\u4f53\u304c #if \u306e\u5185\u5074\u306b\u3042\u308b\u305f\u3081\u3001\u3053\u306e\u30d8\u30c3\u30c0\u306f\u8aad\u307e\u308c\u306a\u3044\u3002
//   \u89b3\u6e2c\u5bfe\u8c61\uff08DSPCore \u306e\u30dd\u30a4\u30f3\u30bf\uff09\u306b\u5bfe\u3059\u308b\u518d\u8a66\u884c\u306e\u547c\u3073\u51fa\u3057\u3068\u7d50\u679c\u3092\u6570\u3048\u308b\u3002
//   \u30af\u30e9\u30b9\u306e\u30e1\u30f3\u30d0\u306b\u306f\u8ffd\u52a0\u3057\u306a\u3044\uff08\u30ec\u30a4\u30a2\u30a6\u30c8\u3092\u5909\u3048\u306a\u3044\uff09\u3002

#include <atomic>
#include <cstdint>

#include "audioengine/AtomicAccess.h"

namespace convo::testobs
{

struct RetryableDestroyLog
{
    std::atomic<const void*> target { nullptr };
    std::atomic<std::uint32_t> attempts { 0 };          // \u5165\u53e3\u306e\u547c\u3073\u51fa\u3057\u56de\u6570
    std::atomic<std::uint32_t> deferredAttempts { 0 };  // Deferred \u3067\u623b\u3063\u305f\u56de\u6570
    std::atomic<std::uint32_t> destroyedAttempts { 0 }; // Destroyed \u3067\u623b\u3063\u305f\u56de\u6570
    std::atomic<bool> released { false };               // \u5bfe\u8c61\u30dd\u30a4\u30f3\u30bf\u304c aligned_free \u3092\u901a\u904e\u3057\u305f
};

inline RetryableDestroyLog& log() noexcept
{
    static RetryableDestroyLog l;
    return l;
}

// \u89b3\u6e2c\u5bfe\u8c61\u3092\u8a2d\u5b9a\u3057\u3001\u8a08\u6570\u3092 0 \u306b\u623b\u3059\uff08\u521d\u671f\u306e Deferred \u3092\u6570\u3048\u306a\u3044\u305f\u3081\uff09\u3002
inline void setTarget (const void* p) noexcept
{
    auto& l = log();
    convo::publishAtomic (l.target, p, std::memory_order_release);
    convo::publishAtomic (l.attempts, std::uint32_t{0}, std::memory_order_relaxed);
    convo::publishAtomic (l.deferredAttempts, std::uint32_t{0}, std::memory_order_relaxed);
    convo::publishAtomic (l.destroyedAttempts, std::uint32_t{0}, std::memory_order_relaxed);
    convo::publishAtomic (l.released, false, std::memory_order_relaxed);
}

inline bool matches (const void* p) noexcept
{
    const void* t = convo::consumeAtomic (log().target, std::memory_order_acquire);
    return t != nullptr && t == p;
}

inline void noteEntry (const void* p) noexcept
{
    if (matches (p))
        (void) convo::fetchAddAtomic (log().attempts, std::uint32_t{1}, std::memory_order_acq_rel);
}

inline void noteDeferred (const void* p) noexcept
{
    if (matches (p))
        (void) convo::fetchAddAtomic (log().deferredAttempts, std::uint32_t{1},
                                      std::memory_order_acq_rel);
}

inline void noteDestroyed (const void* p) noexcept
{
    if (matches (p))
        (void) convo::fetchAddAtomic (log().destroyedAttempts, std::uint32_t{1},
                                      std::memory_order_acq_rel);
}

inline void noteReleased (const void* p) noexcept
{
    if (matches (p))
        convo::publishAtomic (log().released, true, std::memory_order_release);
}

} // namespace convo::testobs
