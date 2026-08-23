// TerminalTelemetryContractTests.cpp
// Phase 9-B Step 5-I: Terminal Telemetry Instrumentation Contract Tests
//
// Tests T-5.1 through T-5.6: Verify the 4 telemetry members added to
// TerminalReclaimAuthority and the AtomicAccess wrapper convention fixes.

#include <cstdint>
#include <stdexcept>

#include "audioengine/ISRRetireRouter.h"
#include "audioengine/AtomicAccess.h"  // atomic-dot-call policy: convo::consumeAtomic / publishAtomic

// ── T-5.1: Initial state — all telemetry counters are zero
[[nodiscard]] bool testTerminalTelemetryInitialState()
{
    convo::isr::TerminalReclaimAuthority auth;

    if (auth.terminalPeakResident() != 0)
        return false;
    if (auth.terminalStoreCount() != 0)
        return false;
    if (auth.terminalDrainAllCount() != 0)
        return false;
    if (auth.terminalDrainEntryCount() != 0)
        return false;
    if (auth.residentCountAtomic() != 0)
        return false;
    if (auth.reclaimCount() != 0)
        return false;

    return true;
}

// ── T-5.2: 3 stores — storeCount == 3, peakResident == 3
[[nodiscard]] bool testTerminalTelemetryAfterThreeStores()
{
    convo::isr::TerminalReclaimAuthority auth;

    for (int i = 0; i < 3; ++i) {
        auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x3000 + i * 0x10));
        auto* deleter = +[](void*) noexcept {};
        if (!auth.store(ptr, deleter, 100,  // NOLINT(atomic-dot-call)
                        DeletionEntryType::Generic, "test"))
            return false;
    }

    if (auth.terminalStoreCount() != 3)
        return false;
    if (auth.terminalPeakResident() != 3)
        return false;
    if (auth.residentCountAtomic() != 3)
        return false;
    if (auth.reclaimCount() != 0)
        return false;

    return true;
}

// ── T-5.3: drain after store — resident drops to 0, reclaimCount for World
[[nodiscard]] bool testTerminalTelemetryAfterDrain()
{
    convo::isr::TerminalReclaimAuthority auth;

    auto* deleter = +[](void* p) noexcept { (void)p; };

    if (!auth.store(reinterpret_cast<void*>(static_cast<uintptr_t>(0x3000)),  // NOLINT(atomic-dot-call)
                    deleter, 100, DeletionEntryType::Generic, "g1"))
        return false;
    if (!auth.store(reinterpret_cast<void*>(static_cast<uintptr_t>(0x3010)),  // NOLINT(atomic-dot-call)
                    deleter, 100, DeletionEntryType::Generic, "g2"))
        return false;
    if (!auth.store(reinterpret_cast<void*>(static_cast<uintptr_t>(0x3020)),  // NOLINT(atomic-dot-call)
                    deleter, 100, DeletionEntryType::World, "w1"))
        return false;

    if (auth.residentCountAtomic() != 3)
        return false;
    if (auth.reclaimCount() != 0)
        return false;
    if (auth.terminalStoreCount() != 3)
        return false;

    uint64_t minReader = 200;
    auto isOlder = [](uint64_t a, uint64_t b) noexcept {
        return static_cast<int64_t>(a - b) < 0;
    };
    auth.drain(minReader, isOlder);

    if (auth.residentCountAtomic() != 0)
        return false;
    if (auth.reclaimCount() != 1)
        return false;
    if (auth.terminalDrainAllCount() != 0)
        return false;
    if (auth.terminalDrainEntryCount() != 0)
        return false;
    if (auth.terminalStoreCount() != 3)
        return false;

    return true;
}

// ── T-5.4: peak monotonicity — peak never decreases
[[nodiscard]] bool testTerminalTelemetryPeakMonotonicity()
{
    convo::isr::TerminalReclaimAuthority auth;
    auto* deleter = +[](void*) noexcept {};

    uint32_t prevPeak = 0;

    for (int i = 0; i < 5; ++i) {
        auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x4000 + i * 0x10));
        if (!auth.store(ptr, deleter, 100, DeletionEntryType::Generic, "test"))  // NOLINT(atomic-dot-call)
            return false;

        const uint32_t peak = auth.terminalPeakResident();
        if (peak < prevPeak)
            return false;
        prevPeak = peak;

        if (peak != static_cast<uint32_t>(i + 1))
            return false;
    }

    if (auth.terminalPeakResident() != 5)
        return false;
    if (auth.residentCountAtomic() != 5)
        return false;

    uint64_t minReader = 200;
    auto isOlder = [](uint64_t a, uint64_t b) noexcept {
        return static_cast<int64_t>(a - b) < 0;
    };
    auth.drain(minReader, isOlder);

    if (auth.residentCountAtomic() != 0)
        return false;
    if (auth.terminalPeakResident() != 5)
        return false;

    return true;
}

// ── T-5.5: drainAll — drainAllCount and drainEntryCount
[[nodiscard]] bool testTerminalTelemetryDrainAll()
{
    convo::isr::TerminalReclaimAuthority auth;
    auto* deleter = +[](void* p) noexcept { (void)p; };

    for (int i = 0; i < 4; ++i) {
        auto* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x5000 + i * 0x10));
        if (!auth.store(ptr, deleter, 100, DeletionEntryType::Generic, "test"))  // NOLINT(atomic-dot-call)
            return false;
    }

    if (auth.terminalStoreCount() != 4)
        return false;
    if (auth.residentCountAtomic() != 4)
        return false;
    if (auth.terminalDrainAllCount() != 0)
        return false;
    if (auth.terminalDrainEntryCount() != 0)
        return false;

    auth.drainAll();

    if (auth.terminalDrainAllCount() != 1)
        return false;
    if (auth.terminalDrainEntryCount() != 4)
        return false;
    if (auth.residentCountAtomic() != 0)
        return false;
    if (auth.terminalStoreCount() != 4)
        return false;
    if (auth.terminalPeakResident() != 4)
        return false;

    auth.drainAll();
    if (auth.terminalDrainAllCount() != 2)
        return false;
    if (auth.terminalDrainEntryCount() != 4)
        return false;

    return true;
}

// ── T-5.6: Generic/World separation
[[nodiscard]] bool testTerminalTelemetryGenericWorldSeparation()
{
    convo::isr::TerminalReclaimAuthority auth;

    static std::atomic<int> s_deleteCount{0};
    auto* countingDeleter = +[](void* p) noexcept {
        (void)p;
        convo::fetchAddAtomic(s_deleteCount, 1);  // NOLINT(atomic-dot-call)
    };

    if (!auth.store(reinterpret_cast<void*>(static_cast<uintptr_t>(0x6000)),  // NOLINT(atomic-dot-call)
                    countingDeleter, 100, DeletionEntryType::Generic, "g1"))
        return false;
    if (!auth.store(reinterpret_cast<void*>(static_cast<uintptr_t>(0x6010)),  // NOLINT(atomic-dot-call)
                    countingDeleter, 100, DeletionEntryType::Generic, "g2"))
        return false;
    if (!auth.store(reinterpret_cast<void*>(static_cast<uintptr_t>(0x6020)),  // NOLINT(atomic-dot-call)
                    countingDeleter, 100, DeletionEntryType::World, "w1"))
        return false;
    if (!auth.store(reinterpret_cast<void*>(static_cast<uintptr_t>(0x6030)),  // NOLINT(atomic-dot-call)
                    countingDeleter, 100, DeletionEntryType::World, "w2"))
        return false;

    if (auth.terminalStoreCount() != 4)
        return false;
    if (auth.residentCountAtomic() != 4)
        return false;
    if (auth.reclaimCount() != 0)
        return false;

    convo::publishAtomic(s_deleteCount, 0);  // NOLINT(atomic-dot-call)

    auth.drainAll();

    if (auth.terminalDrainAllCount() != 1)
        return false;
    if (auth.terminalDrainEntryCount() != 4)
        return false;
    if (auth.residentCountAtomic() != 0)
        return false;
    if (convo::consumeAtomic(s_deleteCount) != 4)  // NOLINT(atomic-dot-call)
        return false;
    if (auth.reclaimCount() != 2)
        return false;

    return true;
}

int main()
{
    if (!testTerminalTelemetryInitialState())
        throw std::runtime_error("T-5.1: terminal telemetry initial state failed");

    if (!testTerminalTelemetryAfterThreeStores())
        throw std::runtime_error("T-5.2: terminal telemetry after 3 stores failed");

    if (!testTerminalTelemetryAfterDrain())
        throw std::runtime_error("T-5.3: terminal telemetry after drain failed");

    if (!testTerminalTelemetryPeakMonotonicity())
        throw std::runtime_error("T-5.4: terminal telemetry peak monotonicity failed");

    if (!testTerminalTelemetryDrainAll())
        throw std::runtime_error("T-5.5: terminal telemetry drainAll failed");

    if (!testTerminalTelemetryGenericWorldSeparation())
        throw std::runtime_error("T-5.6: terminal telemetry Generic/World separation failed");

    return 0;
}
