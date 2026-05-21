#include "ISRRTExecution.h"
#include "AtomicAccess.h"
#include <cassert>
#include <cstring>
#include <thread>

namespace convo {
namespace isr {

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

thread_local bool RTCapabilityFirewall::isRTContextFlag_ = false;

RTCapabilityFirewall::RTCapabilityFirewall() = default;
RTCapabilityFirewall::~RTCapabilityFirewall() = default;

FirewallToken RTCapabilityFirewall::enter() noexcept
{
    FirewallToken token{
        .threadId = std::this_thread::get_id(),
        .epochId = 0,
        .isValid = true
    };

    isRTContextFlag_ = true;
    return token;
}

void RTCapabilityFirewall::leave(const FirewallToken& token) noexcept
{
    // Verify epoch consistency and clear context
    assert(token.isValid);
    assert(token.threadId == std::this_thread::get_id());

    isRTContextFlag_ = false;
}

void RTCapabilityFirewall::auditPublishAttempt(const char* callSite) noexcept
{
#if JUCE_DEBUG || CONVO_CI_BUILD
    if (isRTContextFlag_) {
        assert(false && "publishAtomic called from RT context!");
    }
#endif
    (void)callSite;
}

// ============================================================================
// RTAllocatorFirewall Implementation
// ============================================================================

thread_local bool RTAllocatorFirewall::isRTContextFlag_ = false;

void RTAllocatorFirewall::onAllocAttempt(size_t size, const char* callSite) noexcept
{
#if JUCE_DEBUG || CONVO_CI_BUILD
    if (isRTContextFlag_) {
        assert(false && "heap allocation attempted from RT context!");
    }
#endif
    (void)size;
    (void)callSite;
}

void RTAllocatorFirewall::markRTContext(bool entering) noexcept
{
    isRTContextFlag_ = entering;
}

bool RTAllocatorFirewall::isRTContext() noexcept
{
    return isRTContextFlag_;
}

}  // namespace isr
}  // namespace convo
