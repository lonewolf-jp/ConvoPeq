// V11ReaderReacquireTests.cpp — V-11 P0 regression: reader protection must be
// re-establishable after reader registration is closed, without admitting a new reader.
//
// Bug: EpochDomain::reserveReaderThread() refuses once registrationClosed_ is set, even
// for the thread that already owns the slot. RCUReader::enter() then fails,
// rootEnterSucceeded() is false, no slot contributes to getMinReaderEpoch(), and an
// object retired at the previous epoch is reclaimed while the callback reads it (UAF).
//
// Fix under test: IReaderEpochProvider::reacquireReaderSlot() — ownership-checked
// re-acquisition that works after close and refuses foreign threads / new readers.
//
// Oracle: the real reclaim pair (RetireQuarantineStore::quarantine/drain with
// EpochDomain::isOlder) is driven directly, so "was it freed" is an observation of the
// production deleter being invoked, not an inference.
//
// No new CTest target: these run as sub-tests of the AudioEngineHarness exe.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>

#include "core/EpochDomain.h"
#include "core/RCUReader.h"
#include "audioengine/RetireQuarantineStore.h"

namespace {

constexpr uint64_t kV11ValidMagic = 0xC0FFEE11ULL;
constexpr uint64_t kV11FreedMagic = 0xDEADDEADULL;

int   g_v11DeleterCalls = 0;
void* g_v11LastDeleted  = nullptr;

void v11TestDeleter(void* p) noexcept
{
    ++g_v11DeleterCalls;
    g_v11LastDeleted = p;
}

void v11ResetCounters() noexcept
{
    g_v11DeleterCalls = 0;
    g_v11LastDeleted  = nullptr;
}

int v11FindActiveSlot(convo::EpochDomain& domain) noexcept
{
    for (int i = 0; i < domain.readerCapacity(); ++i)
    {
        if (domain.getReaderSlotDetail(i).depth > 0)
            return i;
    }
    return -1;
}

// v11_1: baseline re-entry while registration is open.
bool checkV11BaselineReentry()
{
    convo::EpochDomain domain;
    convo::RCUReader reader(domain);

    bool first = false, second = false;
    { convo::RCUReaderGuard g(reader); first = g.rootEnterSucceeded(); }
    { convo::RCUReaderGuard g(reader); second = g.rootEnterSucceeded(); }

    if (!first || !second)
    {
        std::printf("  v11_1: open reentry enter#1=%d enter#2=%d\n", first ? 1 : 0, second ? 1 : 0);
        return false;
    }
    return true;
}

// v11_2/3/4: after close, protection is re-established and the retired object survives
// until the guard is dropped.
bool checkV11ProtectionAfterClose()
{
    v11ResetCounters();
    convo::EpochDomain domain;
    convo::isr::RetireQuarantineStore store;
    convo::RCUReader reader(domain);

    { convo::RCUReaderGuard g(reader);
      if (!g.rootEnterSucceeded())
      {
          std::printf("  v11_2: initial enter failed\n");
          return false;
      } }

    domain.closeReaderRegistration();

    static int object = 0;
    void* obj = &object;

    bool enteredAfterClose = false;
    {
        convo::RCUReaderGuard g(reader);
        enteredAfterClose = g.rootEnterSucceeded();
        if (!enteredAfterClose)
        {
            std::printf("  v11_2: enter() refused after close (protection not re-established)\n");
            return false;
        }

        const uint64_t retireEpoch = domain.currentEpoch();
        store.quarantine(obj, v11TestDeleter, retireEpoch,
                         DeletionEntryType::Generic, "v11-harness");
        domain.publishEpoch();                       // new publication advances the epoch

        store.drain(domain.getMinReaderEpoch(), convo::EpochDomain::isOlder);
        if (g_v11DeleterCalls != 0)
        {
            std::printf("  v11_3: object freed while the guard was held (calls=%d)\n",
                        g_v11DeleterCalls);
            return false;
        }
    }

    store.drain(domain.getMinReaderEpoch(), convo::EpochDomain::isOlder);
    if (g_v11DeleterCalls != 1 || g_v11LastDeleted != obj)
    {
        std::printf("  v11_4: object not freed after the guard dropped (calls=%d)\n",
                    g_v11DeleterCalls);
        return false;
    }
    return true;
}

// v11_5: a thread that never registered must not obtain a slot after close.
bool checkV11NewThreadStillRefused()
{
    convo::EpochDomain domain;
    domain.closeReaderRegistration();

    std::atomic<int> outcome{99};
    std::thread t([&]
    {
        convo::RCUReader reader(domain);
        convo::RCUReaderGuard g(reader);
        outcome.store(g.rootEnterSucceeded() ? 1 : 0, std::memory_order_release);
    });
    t.join();

    const int got = outcome.load(std::memory_order_acquire);
    if (got != 0)
    {
        std::printf("  v11_5: new thread obtained a slot after close (enter=%d)\n", got);
        return false;
    }
    return true;
}

// v11_6: ownership — a foreign thread cannot re-acquire the owner's slot.
bool checkV11ForeignReacquireRefused()
{
    convo::EpochDomain domain;
    convo::RCUReader reader(domain);

    int slot = -1;
    { convo::RCUReaderGuard g(reader);
      if (!g.rootEnterSucceeded())
          return false;
      slot = v11FindActiveSlot(domain); }

    if (slot < 0)
    {
        std::printf("  v11_6: could not identify the held slot\n");
        return false;
    }

    domain.closeReaderRegistration();

    std::atomic<int> foreign{99};
    std::thread t([&] { foreign.store(domain.reacquireReaderSlot(slot) ? 1 : 0,
                                       std::memory_order_release); });
    t.join();

    const int got = foreign.load(std::memory_order_acquire);
    if (got != 0)
    {
        std::printf("  v11_6: foreign thread re-acquired slot %d\n", slot);
        return false;
    }

    if (!domain.reacquireReaderSlot(slot))
    {
        std::printf("  v11_6: owner could not re-acquire slot %d\n", slot);
        return false;
    }
    return true;
}

struct V11ConcResult
{
    long long reads      = 0;
    long long detections = 0;
    long long enterFail  = 0;
    long long reclaims   = 0;
};

struct V11Payload
{
    uint64_t magic = kV11ValidMagic;
    uint64_t seq   = 0;
};

void v11PayloadDeleter(void* p) noexcept
{
    if (p != nullptr)
        static_cast<V11Payload*>(p)->magic = kV11FreedMagic;
}

// registeredBeforeClose == false models the pre-fix / unprotected reader (it can never
// obtain a slot, so nothing pins the epoch); true exercises the re-acquire path.
V11ConcResult v11RunConcurrency(bool registeredBeforeClose, long long iterations)
{
    convo::EpochDomain domain;
    convo::isr::RetireQuarantineStore store;
    std::atomic<V11Payload*> world{nullptr};

    std::atomic<bool>      start{false};
    std::atomic<bool>      readerReady{false};
    std::atomic<long long> reads{0};
    std::atomic<long long> detections{0};
    std::atomic<long long> enterFail{0};
    std::atomic<long long> reclaims{0};

    if (!registeredBeforeClose)
        domain.closeReaderRegistration();

    std::thread reader([&]
    {
        convo::RCUReader rcu(domain);

        if (registeredBeforeClose)
        {
            { convo::RCUReaderGuard g(rcu);
              if (!g.rootEnterSucceeded())
                  enterFail.fetch_add(1, std::memory_order_relaxed); }
        }

        readerReady.store(true, std::memory_order_release);
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();

        for (long long i = 0; i < iterations; ++i)
        {
            convo::RCUReaderGuard g(rcu);
            if (!g.rootEnterSucceeded())
                enterFail.fetch_add(1, std::memory_order_relaxed);

            V11Payload* p = world.load(std::memory_order_acquire);
            if (p != nullptr)
            {
                reads.fetch_add(1, std::memory_order_relaxed);
                const uint64_t m1 = p->magic;
                std::this_thread::yield();     // widen the read window
                const uint64_t m2 = p->magic;
                if (m1 != kV11ValidMagic || m2 != kV11ValidMagic)
                    detections.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    while (!readerReady.load(std::memory_order_acquire))
        std::this_thread::yield();

    if (registeredBeforeClose)
        domain.closeReaderRegistration();

    world.store(new V11Payload{kV11ValidMagic, 0}, std::memory_order_release);
    start.store(true, std::memory_order_release);

    for (long long i = 0; i < iterations; ++i)
    {
        V11Payload* np = new V11Payload{kV11ValidMagic, static_cast<uint64_t>(i) + 1};
        V11Payload* old = world.exchange(np, std::memory_order_acq_rel);

        const uint64_t retireEpoch = domain.currentEpoch();
        if (old != nullptr)
            store.quarantine(old, v11PayloadDeleter, retireEpoch,
                             DeletionEntryType::Generic, "v11-harness-conc");

        domain.publishEpoch();

        const std::size_t residentBefore = store.residentCount();
        store.drain(domain.getMinReaderEpoch(), convo::EpochDomain::isOlder);
        const std::size_t residentAfter = store.residentCount();
        if (residentAfter < residentBefore)
            reclaims.fetch_add(static_cast<long long>(residentBefore - residentAfter),
                               std::memory_order_relaxed);
    }

    reader.join();

    V11ConcResult r;
    r.reads      = reads.load();
    r.detections = detections.load();
    r.enterFail  = enterFail.load();
    r.reclaims   = reclaims.load();
    return r;
}

// v11_7/v11_8: under real thread contention the unprotected reader is caught reading a
// reclaimed object, and the re-acquiring reader never is.
bool checkV11Concurrency()
{
    constexpr long long kIters = 50000;

    const V11ConcResult unprotected = v11RunConcurrency(false, kIters);
    if (unprotected.detections <= 0)
    {
        std::printf("  v11_7: detector did not fire on the unprotected path (reads=%lld)\n",
                    unprotected.reads);
        return false;
    }

    const V11ConcResult protectedRun = v11RunConcurrency(true, kIters);
    if (protectedRun.detections != 0 || protectedRun.enterFail != 0 || protectedRun.reclaims <= 0)
    {
        std::printf("  v11_8: protected path detections=%lld enterFail=%lld reclaims=%lld\n",
                    protectedRun.detections, protectedRun.enterFail, protectedRun.reclaims);
        return false;
    }

    std::printf("  v11_7/v11_8: unprotected detections=%lld, protected detections=0 (reads=%lld)\n",
                unprotected.detections, protectedRun.reads);
    return true;
}

// v11_9: quarantine is preserved across a re-acquire (invariant #3).
bool checkV11QuarantinePreserved()
{
    v11ResetCounters();
    convo::EpochDomain domain;
    convo::isr::RetireQuarantineStore store;
    convo::RCUReader reader(domain);

    int slot = -1;
    { convo::RCUReaderGuard g(reader);
      if (!g.rootEnterSucceeded())
          return false;
      slot = v11FindActiveSlot(domain); }

    if (slot < 0)
    {
        std::printf("  v11_9: could not identify the held slot\n");
        return false;
    }

    // Idle now: quarantine the slot, then close registration.
    const bool quarantined = domain.quarantineReader(slot);
    const int before = domain.quarantinedReaderCount();
    domain.closeReaderRegistration();

    static int object = 0;
    void* obj = &object;
    bool protectedWhileActive = false;

    {
        convo::RCUReaderGuard g(reader);
        if (!g.rootEnterSucceeded())
        {
            std::printf("  v11_9: enter refused for a quarantined-idle slot\n");
            return false;
        }

        const uint64_t retireEpoch = domain.currentEpoch();
        store.quarantine(obj, v11TestDeleter, retireEpoch,
                         DeletionEntryType::Generic, "v11-harness-q");
        domain.publishEpoch();
        store.drain(domain.getMinReaderEpoch(), convo::EpochDomain::isOlder);
        protectedWhileActive = (g_v11DeleterCalls == 0);

        if (domain.quarantinedReaderCount() != before)
        {
            std::printf("  v11_9: quarantine flag cleared by re-acquire (%d -> %d)\n",
                        before, domain.quarantinedReaderCount());
            return false;
        }
    }

    store.drain(domain.getMinReaderEpoch(), convo::EpochDomain::isOlder);
    const bool freed = (g_v11DeleterCalls == 1) && (g_v11LastDeleted == obj);
    const bool flagKept = domain.quarantinedReaderCount() == before;

    if (!quarantined || !protectedWhileActive || !freed || !flagKept)
    {
        std::printf("  v11_9: quarantined=%d protectedWhileActive=%d freed=%d flagKept=%d\n",
                    quarantined ? 1 : 0, protectedWhileActive ? 1 : 0, freed ? 1 : 0,
                    flagKept ? 1 : 0);
        return false;
    }
    return true;
}

// v11_10: the re-acquire is a single-shot reservation. A second call without an enter in
// between must be refused, which is what keeps "reserved but not entered" from being
// mistaken for protection. RCUReader always follows a successful re-acquire with
// enterReader() in the same call (acquireThreadSlot -> enterReader).
bool checkV11ReacquireIsSingleShot()
{
    convo::EpochDomain domain;
    convo::RCUReader reader(domain);

    int slot = -1;
    { convo::RCUReaderGuard g(reader);
      if (!g.rootEnterSucceeded())
          return false;
      slot = v11FindActiveSlot(domain); }

    if (slot < 0)
        return false;

    domain.closeReaderRegistration();

    const bool first  = domain.reacquireReaderSlot(slot);
    const bool second = domain.reacquireReaderSlot(slot);

    if (!first || second)
    {
        std::printf("  v11_10: first=%d second=%d (expected 1/0)\n",
                    first ? 1 : 0, second ? 1 : 0);
        return false;
    }
    return true;
}


// v11_11 (RA-6): quarantine requested while the slot is busy is a *pending* flag. It must
// not change the barrier accounting mid-block, must be promoted when the guard drops, and
// the resulting quarantined-idle slot must still be re-acquirable with protection.
bool checkV11QuarantinePendingTransition()
{
    v11ResetCounters();
    convo::EpochDomain domain;
    convo::isr::RetireQuarantineStore store;
    convo::RCUReader reader(domain);

    int slot = -1;
    bool requested = false;
    bool notCountedWhilePending = false;
    {
        convo::RCUReaderGuard g(reader);
        if (!g.rootEnterSucceeded())
            return false;
        slot = v11FindActiveSlot(domain);
        if (slot < 0)
            return false;

        // Contract: with depth > 0 the quarantine is deferred, so the call reports
        // false and the pending flag is the acknowledgement (EpochDomain.h:327-328,379).
        requested = !domain.quarantineReader(slot);
        notCountedWhilePending = (domain.quarantinedReaderCount() == 0);
    }                                                        // exit -> promote

    const bool promoted = (domain.quarantinedReaderCount() == 1);

    static int object = 0;
    void* obj = &object;
    bool protectedWhileActive = false;
    {
        convo::RCUReaderGuard g(reader);
        if (!g.rootEnterSucceeded())
        {
            std::printf("  v11_11: re-entry refused after pending quarantine\n");
            return false;
        }
        store.quarantine(obj, v11TestDeleter, domain.currentEpoch(),
                         DeletionEntryType::Generic, "v11-11");
        domain.publishEpoch();
        store.drain(domain.getMinReaderEpoch(), convo::EpochDomain::isOlder);
        protectedWhileActive = (g_v11DeleterCalls == 0);
    }

    store.drain(domain.getMinReaderEpoch(), convo::EpochDomain::isOlder);
    const bool freed = (g_v11DeleterCalls == 1) && (g_v11LastDeleted == obj);

    std::printf("  v11_11: deferred=%d notCountedWhilePending=%d promoted=%d "
                "protected=%d freed=%d\n",
                requested ? 1 : 0, notCountedWhilePending ? 1 : 0, promoted ? 1 : 0,
                protectedWhileActive ? 1 : 0, freed ? 1 : 0);

    if (!requested || !notCountedWhilePending || !promoted || !protectedWhileActive || !freed)
        return false;
    return true;
}

// v11_12 (RA-7): with every slot held, a further reader must be refused, and releasing one
// must let a reader re-enter - the refusal must not poison the slots that are already held.
bool checkV11CapacityRefusalRecovers()
{
    convo::EpochDomain domain;

    std::array<std::unique_ptr<convo::RCUReader>, convo::EpochDomain::kMaxReaders> readers;
    std::array<std::unique_ptr<convo::RCUReaderGuard>, convo::EpochDomain::kMaxReaders> guards;

    int held = 0;
    for (int i = 0; i < convo::EpochDomain::kMaxReaders; ++i)
    {
        readers[i] = std::make_unique<convo::RCUReader>(domain);
        guards[i] = std::make_unique<convo::RCUReaderGuard>(*readers[i]);
        if (guards[i]->rootEnterSucceeded())
            ++held;
    }

    convo::RCUReader extra(domain);
    convo::RCUReaderGuard extraGuard(extra);
    const bool overflowRefused = !extraGuard.rootEnterSucceeded();

    guards[0].reset();                       // release one slot
    convo::RCUReaderGuard reenter(*readers[0]);
    const bool recovered = reenter.rootEnterSucceeded();

    for (auto& g : guards)
        g.reset();

    std::printf("  v11_12: held=%d/%d overflowRefused=%d recovered=%d\n",
                held, convo::EpochDomain::kMaxReaders,
                overflowRefused ? 1 : 0, recovered ? 1 : 0);

    if (held != convo::EpochDomain::kMaxReaders || !overflowRefused || !recovered)
        return false;
    return true;
}

} // namespace

int runV11ReaderReacquireTests()
{
    bool ok = true;

    if (!checkV11BaselineReentry())
    {
        std::fprintf(stderr, "FAIL: checkV11BaselineReentry\n");
        ok = false;
    }
    if (!checkV11ProtectionAfterClose())
    {
        std::fprintf(stderr, "FAIL: checkV11ProtectionAfterClose\n");
        ok = false;
    }
    if (!checkV11NewThreadStillRefused())
    {
        std::fprintf(stderr, "FAIL: checkV11NewThreadStillRefused\n");
        ok = false;
    }
    if (!checkV11ForeignReacquireRefused())
    {
        std::fprintf(stderr, "FAIL: checkV11ForeignReacquireRefused\n");
        ok = false;
    }
    if (!checkV11Concurrency())
    {
        std::fprintf(stderr, "FAIL: checkV11Concurrency\n");
        ok = false;
    }
    if (!checkV11QuarantinePreserved())
    {
        std::fprintf(stderr, "FAIL: checkV11QuarantinePreserved\n");
        ok = false;
    }
    if (!checkV11ReacquireIsSingleShot())
    {
        std::fprintf(stderr, "FAIL: checkV11ReacquireIsSingleShot\n");
        ok = false;
    }
    if (!checkV11QuarantinePendingTransition())
    {
        std::fprintf(stderr, "FAIL: checkV11QuarantinePendingTransition\n");
        ok = false;
    }
    if (!checkV11CapacityRefusalRecovers())
    {
        std::fprintf(stderr, "FAIL: checkV11CapacityRefusalRecovers\n");
        ok = false;
    }

    return ok ? 0 : 1;
}
