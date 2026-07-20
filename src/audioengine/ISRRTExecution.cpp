#include "ISRRTExecution.h"
#include "AtomicAccess.h"
#include <cassert>
#include <cstring>
#include <thread>

namespace convo {
namespace isr {

namespace {
std::atomic<bool> s_sharedRtContextFlag{ false };
}

std::atomic<bool>& detail::sharedRtContextFlag() noexcept
{
    return s_sharedRtContextFlag;
}

// ============================================================================
// RTTraceRelay Implementation
// ============================================================================

RTTraceRelay::RTTraceRelay()
{
    convo::publishAtomic(buffer_, new RTTraceEvent[RELAY_BUFFER_SIZE], std::memory_order_release);
}

RTTraceRelay::~RTTraceRelay()
{
    auto* buf = convo::consumeAtomic(buffer_, std::memory_order_acquire);
    if (buf) {
        delete[] buf;
    }
}

void RTTraceRelay::enqueue(const RTTraceEvent& event) noexcept
{
    auto* buf = convo::consumeAtomic(buffer_, std::memory_order_acquire);
    if (!buf) return;

    uint64_t writeIdx = convo::consumeAtomic(writeIndex_, std::memory_order_relaxed);
    uint64_t nextIdx = (writeIdx + 1) % RELAY_BUFFER_SIZE;

    uint64_t readIdx = convo::consumeAtomic(readIndex_, std::memory_order_acquire);
    if (nextIdx == readIdx) {
        return;
    }

    buf[writeIdx] = event;
    convo::publishAtomic(writeIndex_, nextIdx, std::memory_order_release);
}

void RTTraceRelay::drain()
{
    auto* buf = convo::consumeAtomic(buffer_, std::memory_order_acquire);
    if (!buf) {
        return;
    }

    uint64_t readIdx = convo::consumeAtomic(readIndex_, std::memory_order_acquire);
    const uint64_t writeIdx = convo::consumeAtomic(writeIndex_, std::memory_order_acquire);

    while (readIdx != writeIdx) {
        (void)buf[readIdx];
        readIdx = (readIdx + 1u) % RELAY_BUFFER_SIZE;
    }

    convo::publishAtomic(readIndex_, readIdx, std::memory_order_release);
}

size_t RTTraceRelay::getCurrentDrainCount() const noexcept
{
    uint64_t writeIdx = convo::consumeAtomic(writeIndex_, std::memory_order_acquire);
    uint64_t readIdx = convo::consumeAtomic(readIndex_, std::memory_order_acquire);

    if (writeIdx >= readIdx) {
        return static_cast<size_t>(writeIdx - readIdx);
    } else {
        return static_cast<size_t>(RELAY_BUFFER_SIZE - (readIdx - writeIdx));
    }
}

// ============================================================================
// RTCapabilityFirewall Implementation
// ============================================================================

RTCapabilityFirewall::RTCapabilityFirewall() = default;
RTCapabilityFirewall::~RTCapabilityFirewall() = default;

FirewallToken RTCapabilityFirewall::enter() noexcept
{
    FirewallToken token{
        .threadId = std::this_thread::get_id(),
        .epochId = 0,
        .isValid = true,
    };

    // ★ [work66-P2-3] sharedRtContextFlag は単なる状態フラグであり、
    //   他データとの同期(HB)を必要としないため relaxed が許される。
    //   前提: isRTContext() 自体は実装済みだが、現時点のコードベースで呼出箇所は存在しない。
    //   将来 CONVO_USE_IS_RT_CONTEXT を定義して isRTContext() を使用する場合、
    //   writer の memory_order を release に戻す必要がある（reader 側が acquire のため）。
#if defined(NDEBUG) && !defined(CONVO_CI_BUILD)
 #if defined(CONVO_USE_IS_RT_CONTEXT)
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_release);
 #else
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_relaxed);
 #endif
#else
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_release);
#endif
    return token;
}

void RTCapabilityFirewall::leave(const FirewallToken& token) noexcept
{
    assert(token.isValid);
    assert(token.threadId == std::this_thread::get_id());

#if defined(NDEBUG) && !defined(CONVO_CI_BUILD)
 #if defined(CONVO_USE_IS_RT_CONTEXT)
    convo::publishAtomic(detail::sharedRtContextFlag(), false, std::memory_order_release);
 #else
    convo::publishAtomic(detail::sharedRtContextFlag(), false, std::memory_order_relaxed);
 #endif
#else
    convo::publishAtomic(detail::sharedRtContextFlag(), false, std::memory_order_release);
#endif
}

void RTCapabilityFirewall::auditPublishAttempt(const char* callSite) noexcept
{
#if JUCE_DEBUG || CONVO_CI_BUILD
    if (convo::consumeAtomic(detail::sharedRtContextFlag(), std::memory_order_acquire)) {
        assert(false && "publishAtomic called from RT context!");
    }
#endif
    (void)callSite;
}

// ============================================================================
// RTAllocatorFirewall Implementation
// ============================================================================

void RTAllocatorFirewall::onAllocAttempt(size_t size, const char* callSite) noexcept
{
#if JUCE_DEBUG || CONVO_CI_BUILD
    if (convo::consumeAtomic(detail::sharedRtContextFlag(), std::memory_order_acquire)) {
        assert(false && "heap allocation attempted from RT context!");
    }
#endif
    (void)size;
    (void)callSite;
}

void RTAllocatorFirewall::markRTContext(bool entering) noexcept
{
#if defined(NDEBUG) && !defined(CONVO_CI_BUILD)
    convo::publishAtomic(detail::sharedRtContextFlag(), entering, std::memory_order_relaxed);
#else
    convo::publishAtomic(detail::sharedRtContextFlag(), entering, std::memory_order_release);
#endif
}

bool RTAllocatorFirewall::isRTContext() noexcept
{
    return convo::consumeAtomic(detail::sharedRtContextFlag(), std::memory_order_acquire);
}

}  // namespace isr
}  // namespace convo
