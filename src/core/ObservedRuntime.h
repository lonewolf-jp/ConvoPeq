#pragma once

#include <thread>
#include <type_traits>

#include "EpochDomain.h"
#include "GlobalSnapshot.h"

namespace convo {

struct ObservedRuntime
{
    explicit ObservedRuntime(EpochDomain& domain, int readerIndex) noexcept
        : guard(domain, readerIndex), ownerThreadId(std::this_thread::get_id())
    {
    }

    ObservedRuntime(const ObservedRuntime&) = delete;
    ObservedRuntime& operator=(const ObservedRuntime&) = delete;
    ObservedRuntime(ObservedRuntime&&) noexcept = default;
    ObservedRuntime& operator=(ObservedRuntime&&) noexcept = default;

    const GlobalSnapshot* get() const noexcept
    {
        if (ownerThreadId != std::this_thread::get_id())
            return nullptr;
        return ptr;
    }

    explicit operator bool() const noexcept
    {
        return ownerThreadId == std::this_thread::get_id() && ptr != nullptr;
    }

    EpochDomainReaderGuard guard;
    const GlobalSnapshot* ptr = nullptr;
    std::thread::id ownerThreadId;
};

static_assert(!std::is_copy_constructible_v<ObservedRuntime>);
static_assert(std::is_move_constructible_v<ObservedRuntime>);

} // namespace convo
