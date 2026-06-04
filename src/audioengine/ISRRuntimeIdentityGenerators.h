#pragma once

#include <atomic>
#include <cstdint>

#include "AtomicAccess.h"

namespace convo::isr {

class RuntimeWorldIdGenerator
{
public:
    // DIAGNOSTIC ONLY: RuntimeWorldIdGenerator produces identifiers for
    // trace/correlation/diagnostic purposes. Must NOT be used for:
    // - Authority decisions (branch, condition, ordering)
    // - Publication ordering
    // - Retire ordering
    // - Hash keys in semantic structures
    [[nodiscard]] std::uint64_t next() noexcept
    {
        return convo::fetchAddAtomic(counter_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel) + 1u;
    }

private:
    std::atomic<std::uint64_t> counter_ { 0 };
};

class RuntimeGenerationGenerator
{
public:
    [[nodiscard]] std::uint64_t next() noexcept
    {
        return convo::fetchAddAtomic(counter_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel) + 1u;
    }

private:
    std::atomic<std::uint64_t> counter_ { 0 };
};

} // namespace convo::isr
