// D154 runtime lock-free probe (diagnostic helper — NOT production/test source).
// Replicates the EXACT RecoveryLifecycleWord layout of D152-R1 §1 and reports
// sizeof/alignof + std::atomic<W>::is_lock_free() + CAS semantics (success and
// failure-with-comparand-update) under the same MSVC x64 toolchain used by the build.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <type_traits>

struct alignas(16) RecoveryLifecycleWord {
    std::uint64_t obligationId = 0;
    std::uint8_t  state        = 0;
    std::uint8_t  pending      = 0;
    std::uint8_t  adjudicated  = 0;
    std::uint8_t  delivery     = 0;
    std::uint8_t  pad[4]       = {};
};
static_assert(sizeof(RecoveryLifecycleWord) == 16);
static_assert(alignof(RecoveryLifecycleWord) == 16);
static_assert(std::is_trivially_copyable_v<RecoveryLifecycleWord>);
static_assert(std::is_standard_layout_v<RecoveryLifecycleWord>);
static_assert(alignof(std::atomic<RecoveryLifecycleWord>) >= 16);

int main()
{
    std::atomic<RecoveryLifecycleWord> a{};
    const bool lf = a.is_lock_free();
    std::printf("sizeof=%zu alignof=%zu atomic_alignof=%zu is_lock_free=%d\n",
        sizeof(RecoveryLifecycleWord), alignof(RecoveryLifecycleWord),
        alignof(std::atomic<RecoveryLifecycleWord>), lf ? 1 : 0);

    RecoveryLifecycleWord exp{};
    RecoveryLifecycleWord des{};
    des.obligationId = 7;
    des.state = 1;
    const bool ok1 = a.compare_exchange_strong(exp, des, std::memory_order_acq_rel);

    RecoveryLifecycleWord bad{};
    bad.obligationId = 99;
    const bool ok2 = a.compare_exchange_strong(bad, des, std::memory_order_acq_rel);
    std::printf("cas_success=%d cas_fail_updates_comparand=%d (observed id=%llu)\n",
        ok1 ? 1 : 0, (!ok2 && bad.obligationId == 7) ? 1 : 0,
        (unsigned long long)bad.obligationId);

    return (lf && ok1 && !ok2 && bad.obligationId == 7) ? 0 : 1;
}
