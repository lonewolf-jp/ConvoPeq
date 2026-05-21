#pragma once

#include <cassert>
#include <atomic>
#include <cstdint>
#include "AtomicAccess.h"

namespace convo {
namespace isr {

/**
 * ISR 10層 Architecture Layer 5: Sealed Object
 * mutable state の mutation detection と enforcement
 */

/**
 * mutation tracking state
 */
enum class SealState
{
    Unsealed,    // Normal mutation allowed
    Sealed,      // Mutation prohibited (publish phase)
    Sealed_Recursive  // Sealed + recursive children sealed
};

/**
 * sealed object base（CRTP pattern）
 */
template<typename Derived>
class SealedObject
{
public:
    SealedObject() : sealState_(SealState::Unsealed) {}
    virtual ~SealedObject() = default;

    // Seal this object (enable readonly mode)
    void seal() noexcept
    {
        convo::publishAtomic(sealState_, SealState::Sealed, std::memory_order_release);
    }

    // Seal recursively (seal all child objects)
    void sealRecursively() noexcept
    {
        convo::publishAtomic(sealState_, SealState::Sealed_Recursive, std::memory_order_release);
    }

    // Check if sealed
    bool isSealed() const noexcept
    {
        auto state = convo::consumeAtomic(sealState_, std::memory_order_acquire);
        return state != SealState::Unsealed;
    }

    // Check if recursively sealed
    bool isSealedRecursively() const noexcept
    {
        return convo::consumeAtomic(sealState_, std::memory_order_acquire) == SealState::Sealed_Recursive;
    }

    // Unseal (for reclaim phase)
    void unseal() noexcept
    {
        convo::publishAtomic(sealState_, SealState::Unsealed, std::memory_order_release);
    }

    // Assert mutation is allowed
    void assertMutable() const noexcept
    {
        assert(convo::consumeAtomic(sealState_, std::memory_order_acquire) == SealState::Unsealed);
    }

protected:
    // Check seal state (for derived class assertions)
    bool isMutable() const noexcept
    {
        return convo::consumeAtomic(sealState_, std::memory_order_acquire) == SealState::Unsealed;
    }

private:
    std::atomic<SealState> sealState_;
};

}  // namespace isr
}  // namespace convo
