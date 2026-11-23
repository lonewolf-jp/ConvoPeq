#include <atomic>
#include <cstdint>

namespace convo
{
    std::atomic<std::uint32_t> g_nextRange { 0 };
}

extern "C" std::uint32_t probe_thread_local_range() noexcept
{
    thread_local std::uint32_t assigned = 0xFFFFFFFFu;

    if (assigned == 0xFFFFFFFFu)
        assigned = convo::g_nextRange.fetch_add (1u, std::memory_order_relaxed) % 8u;

    return assigned;
}
