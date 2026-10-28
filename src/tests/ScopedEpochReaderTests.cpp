// =============================================================================
// ScopedEpochReaderTests.cpp — A-closure (reader safety) regression test.
//
// Defect class (proven by inspection, not by racing):
//   A non-RT reader that loads a published pointer with acquire and dereferences it
//   without holding an EBR reader slot can be preempted between load and dereference.
//   The publisher then unlinks the object, retires it at the current epoch, and a
//   reclaimer observes "no readers" (getMinReaderEpoch() == currentEpoch) and frees
//   the object under the reader -> use-after-free.
//
// Fix under test:
//   convo::ScopedEpochReader registers a reader slot BEFORE the load and keeps it
//   until after the last dereference, so `retireEpoch < minReaderEpoch` cannot hold
//   while the reader is inside. When no slot can be obtained it is fail-closed
//   (active() == false) and callers must not dereference.
//
// Determinism:
//   Single thread, no races. The oracle is "was the deleter called?" only; no
//   production internals and no debug assertions are used as the oracle.
//   T1/T2 include controls proving the object IS reclaimed once the reader exits,
//   so a PASS cannot come from "reclaim never happens".
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "core/EpochDomain.h"
#include "core/ScopedEpochReader.h"

using convo::EpochDomain;
using convo::ScopedEpochReader;

namespace
{

struct Counter
{
    int frees = 0;
    int alive = 0;
};

Counter* g_counter = nullptr;

void countingDeleter (void* p) noexcept
{
    if (g_counter != nullptr)
    {
        ++g_counter->frees;
        --g_counter->alive;
    }

    std::free (p);
}

void* allocate()
{
    void* p = std::malloc (64);

    if (p == nullptr)
        throw std::runtime_error ("allocation failed");

    if (g_counter != nullptr)
        ++g_counter->alive;

    return p;
}

// ── T1: reader slot blocks reclamation across the read window ───────────────
bool testT1_readerSlotBlocksReclaim()
{
    EpochDomain domain;
    Counter counter;
    g_counter = &counter;

    void* object = allocate();
    const std::uint64_t retireEpoch = domain.currentEpoch();

    {
        ScopedEpochReader reader (domain);

        if (! reader.active())
            return false;

        // Retire at the epoch the reader is in, then advance and try to reclaim.
        if (! domain.enqueueRetire (object, countingDeleter, retireEpoch))
            return false;

        domain.publishEpoch();
        domain.tryReclaim();

        if (counter.frees != 0)          // freed while the reader is inside
            return false;
    }

    // Reader gone: the same reclaim must now free it (control).
    domain.publishEpoch();
    domain.tryReclaim();
    g_counter = nullptr;

    return counter.frees == 1 && counter.alive == 0;
}

// ── T2: nested readers keep the barrier until the outermost exit ────────────
bool testT2_nestedReadersKeepBarrier()
{
    EpochDomain domain;
    Counter counter;
    g_counter = &counter;

    void* object = allocate();
    const std::uint64_t retireEpoch = domain.currentEpoch();

    {
        ScopedEpochReader outer (domain);

        if (! outer.active())
            return false;

        {
            ScopedEpochReader inner (domain);

            if (! inner.active())
                return false;

            if (! domain.enqueueRetire (object, countingDeleter, retireEpoch))
                return false;

            domain.publishEpoch();
            domain.tryReclaim();

            if (counter.frees != 0)
                return false;
        }

        // Inner exited, outer still holds the slot: still must not free.
        domain.publishEpoch();
        domain.tryReclaim();

        if (counter.frees != 0)
            return false;
    }

    domain.publishEpoch();
    domain.tryReclaim();
    g_counter = nullptr;

    return counter.frees == 1;
}

// ── T3: no reader -> reclaimed on the first reclaim after publish (control) ─
bool testT3_noReaderReclaimsImmediately()
{
    EpochDomain domain;
    Counter counter;
    g_counter = &counter;

    void* object = allocate();
    const std::uint64_t retireEpoch = domain.currentEpoch();

    if (! domain.enqueueRetire (object, countingDeleter, retireEpoch))
        return false;

    domain.publishEpoch();
    domain.tryReclaim();
    g_counter = nullptr;

    return counter.frees == 1;
}

// ── T4: fail-closed after reader registration is closed ─────────────────────
bool testT4_failClosedAfterRegistrationClosed()
{
    EpochDomain domain;
    domain.closeReaderRegistration();

    ScopedEpochReader reader (domain);

    // No slot -> caller must not dereference (fail-closed contract).
    return ! reader.active();
}

} // namespace

int main()
{
    int failures = 0;

    try
    {
        if (! testT1_readerSlotBlocksReclaim())
        {
            std::fprintf (stderr, "FAIL: T1 reader slot must block reclaim across the read window\n");
            ++failures;
        }
        else
        {
            std::printf ("ScopedEpochReaderTests: T1 PASS (slot blocks reclaim; frees after exit)\n");
        }

        if (! testT2_nestedReadersKeepBarrier())
        {
            std::fprintf (stderr, "FAIL: T2 nested readers must keep the barrier until the outer exit\n");
            ++failures;
        }
        else
        {
            std::printf ("ScopedEpochReaderTests: T2 PASS (nesting keeps barrier)\n");
        }

        if (! testT3_noReaderReclaimsImmediately())
        {
            std::fprintf (stderr, "FAIL: T3 control - without a reader the object must be reclaimed\n");
            ++failures;
        }
        else
        {
            std::printf ("ScopedEpochReaderTests: T3 PASS (control: reclaim happens without a reader)\n");
        }

        if (! testT4_failClosedAfterRegistrationClosed())
        {
            std::fprintf (stderr, "FAIL: T4 closed registration must yield an inactive (fail-closed) guard\n");
            ++failures;
        }
        else
        {
            std::printf ("ScopedEpochReaderTests: T4 PASS (fail-closed after registration close)\n");
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf (stderr, "ScopedEpochReaderTests: exception: %s\n", e.what());
        return 1;
    }

    if (failures != 0)
    {
        std::fprintf (stderr, "ScopedEpochReaderTests: %d test(s) FAILED\n", failures);
        return 1;
    }

    std::printf ("ScopedEpochReaderTests: PASS (T1/T2/T3/T4)\n");
    return 0;
}
