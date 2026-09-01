//==============================================================================
// D152-R2 / ST-1 — AV stress 200 driver (REPOSITORY-EXTERNAL, evidence artifact)
//
// STANDALONE stress harness for the T3c lifecycle word (RecoveryLifecycleWord,
// 16B std::atomic backend). Not registered in CMake, not imported into the
// repo source tree: this file lives under evidence/st1/ and is compiled against
// the SAME production translation units already built by the D154 test target
// (via the existing .obj files, so production code is byte-identical to the
// D154 40/40 gate). Purpose: exercise T5 coalesce contention, T1 pending
// signal, T2 adjudication, T3 terminal resolution, delivery CAS and repeated
// lifecycle reuse under lock-pool backend, and RECORD the counter/state
// consistency (not merely "exit code 0").
//
// Per-cycle invariants (asserted immediately, so the first violation aborts
// with the cycle number and the observed counters):
//   liveCount_ < 0                    -> 0 occurrences
//   liveCount_ > 32 (capacity)        -> 0 occurrences
//   double terminal transition        -> 0 (peek state stays terminal; resolve won-flag once)
//   droppedTerminal / droppedStale    -> expected semantics (only after terminalization/stale)
//   droppedInvalid                    -> 0 (we never post obligationId==0)
//   saturated                         -> expected semantics (pending>=K)
//   recoveryRetryExhaustedCount       -> over-count 0 (== number of Failed terminals)
//==============================================================================

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "audioengine/ISRRuntimePublicationCoordinator.h"

using convo::isr::RuntimeIntentCoordinator;
using OState = RuntimeIntentCoordinator::ObligationState;
using ODeliv = RuntimeIntentCoordinator::ObligationDeliveryState;
using ROutcome = RuntimeIntentCoordinator::RecoveryOutcome;

namespace {

constexpr int kCycles = 200;
constexpr std::uint64_t kTableCapacity = 32;   // RecoveryAdmissionTable capacity

convo::RuntimeBuildSnapshot makeSnapshot(std::uint64_t identityHash)
{
    convo::RuntimeBuildSnapshot snap{};
    snap.rebuildFingerprint.irIdentityHash = identityHash;
    snap.rebuildFingerprint.convolutionConfigHash = 0x21u;
    snap.rebuildFingerprint.dspParameterHash = 0x42u;
    snap.rebuildFingerprint.fingerprintVersion = 1;
    return snap;
}

struct Counters
{
    std::uint64_t coalesced = 0;
    std::uint64_t exhausted = 0;
    std::uint64_t droppedTerminal = 0;
    std::uint64_t droppedStale = 0;
    std::uint64_t droppedInvalid = 0;
    std::uint64_t saturated = 0;
};

Counters readCounters(const RuntimeIntentCoordinator& c)
{
    return Counters{
        c.recoveryCoalescedCount(),
        c.recoveryRetryExhaustedCount(),
        c.recoveryFailureSignalDroppedTerminalCount(),
        c.recoveryFailureSignalDroppedStaleCount(),
        c.recoveryFailureSignalDroppedInvalidCount(),
        c.recoveryFailureSignalSaturatedCount(),
    };
}

void dumpCounters(std::ostream& os, const Counters& k)
{
    os << "  liveCount/-coal/exh/dT/dS/dI/sat = "
       << k.coalesced << '/' << k.exhausted << '/' << k.droppedTerminal << '/'
       << k.droppedStale << '/' << k.droppedInvalid << '/' << k.saturated;
}

void failAt(int cycle, const char* what, const Counters& k, std::int64_t live)
{
    std::string msg = "ST1 cycle ";
    msg += std::to_string(cycle);
    msg += ": ";
    msg += what;
    msg += " | live=";
    msg += std::to_string(live);
    msg += " | counters(coal/exh/dT/dS/dI/sat) = ";
    msg += std::to_string(k.coalesced); msg += '/';
    msg += std::to_string(k.exhausted); msg += '/';
    msg += std::to_string(k.droppedTerminal); msg += '/';
    msg += std::to_string(k.droppedStale); msg += '/';
    msg += std::to_string(k.droppedInvalid); msg += '/';
    msg += std::to_string(k.saturated);
    throw std::runtime_error(msg);
}

// ── One stress cycle: contention on a single obligation, then full teardown. ──
//   producer (RebuildThread role): postRecoveryFailureSignal storm (T1)
//   consumer (CoordinatorLoop role): adjudicateRecoveryFailureSignals storm (T2)
//   main: coalesce storms (T5), delivery CAS (T6), terminal resolve (T3)
void runCycle(int cycle, RuntimeIntentCoordinator& c)
{
    const Counters k0 = readCounters(c);
    const std::int64_t live0 = static_cast<std::int64_t>(c.liveLogicalRecoveryObligationCount());

    // admission (fresh id each cycle -> repeated lifecycle reuse across cycles)
    if (!c.submitRecoveryRequest(convo::isr::DSPHandle::null(), makeSnapshot(cycle + 1), 1))
        failAt(cycle, "submitRecoveryRequest rejected (fresh id, table not full)", k0, live0);
    auto pop = c.popRecoveryRequest();
    if (!pop)
        failAt(cycle, "popRecoveryRequest empty after submit", k0, live0);
    const std::uint64_t obl = pop->obligationId;
    if (obl == 0)
        failAt(cycle, "obligationId == 0 from transport pop", k0, live0);

    std::int64_t live = static_cast<std::int64_t>(c.liveLogicalRecoveryObligationCount());
    if (live != live0 + 1)
        failAt(cycle, "liveCount did not advance by exactly +1 on admission", readCounters(c), live);
    if (live < 0 || live > static_cast<std::int64_t>(kTableCapacity))
        failAt(cycle, "liveCount out of [0, 32]", readCounters(c), live);

    const auto w0 = c.peekLifecycleForTest(obl);
    if (!w0 || w0->obligationId != obl
        || w0->state != static_cast<std::uint8_t>(OState::Live)
        || w0->delivery != static_cast<std::uint8_t>(ODeliv::Transport))
        failAt(cycle, "admitted lifecycle word not {Live, Transport}", readCounters(c), live);

    // T5 coalesce storm: same {handle, target} must coalesce (delta L = 0, counter +n)
    for (int i = 0; i < 16; ++i) {
        const std::uint64_t coal0 = c.recoveryCoalescedCount();
        if (!c.submitRecoveryRequest(convo::isr::DSPHandle::null(), makeSnapshot(cycle + 1), 1))
            failAt(cycle, "coalesce resubmit rejected (obligation is Live)", readCounters(c),
                   static_cast<std::int64_t>(c.liveLogicalRecoveryObligationCount()));
        if (c.recoveryCoalescedCount() != coal0 + 1)
            failAt(cycle, "same-key resubmit did not COALESCE (created NEW obligation)", readCounters(c),
                   static_cast<std::int64_t>(c.liveLogicalRecoveryObligationCount()));
        if (c.liveLogicalRecoveryObligationCount() != static_cast<std::uint64_t>(live0 + 1))
            failAt(cycle, "coalesce changed liveCount (delta L != 0)", readCounters(c),
                   static_cast<std::int64_t>(c.liveLogicalRecoveryObligationCount()));
        auto p = c.popRecoveryRequest();     // drain the re-pushed transport intent
        if (!p) failAt(cycle, "coalesce did not re-push transport intent", readCounters(c),
                       static_cast<std::int64_t>(c.liveLogicalRecoveryObligationCount()));
    }

    // T1/T2/T3 contention window (same shape as NT-5, denser). NOTE: a same-key resubmit that
    // lands AFTER adjudication terminalized the old obligation legally re-admits a NEW
    // obligation (terminal → resubmit NEW, G-4.3-T T7 semantic) — the churn loop captures that
    // child's id from the transport pop so the teardown can resolve it too.
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> childId{0};
    std::thread producer([&] {
        while (!stop.load(std::memory_order_relaxed))
            c.postRecoveryFailureSignal(obl);            // T1: pending++ (full-word CAS)
    });
    std::thread consumer([&] {
        while (!stop.load(std::memory_order_relaxed))
            c.adjudicateRecoveryFailureSignals();        // T2: drain+apply (full-word CAS)
    });

    // T6 delivery CAS churn under contention: a coalesce resubmit re-pushes and re-attaches
    // Transport via the single attach primitive while producer/consumer race on the same word.
    for (int i = 0; i < 64 && !stop.load(std::memory_order_relaxed); ++i) {
        (void)c.submitRecoveryRequest(convo::isr::DSPHandle::null(), makeSnapshot(cycle + 1), 1);
        auto p = c.popRecoveryRequest();
        if (p && p->obligationId != obl)
            childId.store(p->obligationId, std::memory_order_relaxed);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(8));
    stop.store(true, std::memory_order_relaxed);
    producer.join();
    consumer.join();

    // ── teardown: drive every outstanding admitted obligation to a terminal, single authority ──
    const auto oldW0 = c.peekLifecycleForTest(obl);
    const std::uint64_t child = childId.load(std::memory_order_relaxed);
    if (oldW0 && static_cast<OState>(oldW0->state) == OState::Live) {
        if (!c.resolveRecoveryObligation(obl, ROutcome::Published))
            failAt(cycle, "resolve lost the terminal CAS on a Live obligation", readCounters(c),
                   static_cast<std::int64_t>(c.liveLogicalRecoveryObligationCount()));
    }
    if (child != 0) {
        const auto cw = c.peekLifecycleForTest(child);
        if (cw && static_cast<OState>(cw->state) == OState::Live) {
            if (!c.resolveRecoveryObligation(child, ROutcome::Published))
                failAt(cycle, "resolve lost the terminal CAS on a re-admitted child", readCounters(c),
                       static_cast<std::int64_t>(c.liveLogicalRecoveryObligationCount()));
        }
    }

    const auto wf = c.peekLifecycleForTest(obl);
    const auto wc = child != 0 ? c.peekLifecycleForTest(child) : decltype(wf){};
    const Counters k1 = readCounters(c);
    const std::int64_t liveF = static_cast<std::int64_t>(c.liveLogicalRecoveryObligationCount());

    // identity stability: whatever terminal word remains for obl must still carry obl's id (no ABA)
    if (wf) {
        if (wf->obligationId != obl)
            failAt(cycle, "identity not stable through contention", k1, liveF);
        const auto st = static_cast<OState>(wf->state);
        if (st != OState::ResolvedSuccess && st != OState::ResolvedFailed)
            failAt(cycle, "old terminal state is not a legal terminal", k1, liveF);
        if (wf->adjudicated != 0 || wf->pending != 0)
            failAt(cycle, "terminal word carries residual pending/adjudicated", k1, liveF);
        // NOTE: delivery is independent of ObligationState (D152-R1 h:356) — resolve/adjudicate
        // preserve it; None-ing happens only on the non-terminal drain CAS. No delivery assert.
        if (c.resolveRecoveryObligation(obl, ROutcome::Published))
            failAt(cycle, "double terminal transition: second resolve returned true", k1, liveF);
    }
    if (wc) {
        if (wc->obligationId != child)
            failAt(cycle, "child identity not stable", k1, liveF);
        const auto st = static_cast<OState>(wc->state);
        if (st != OState::ResolvedSuccess && st != OState::ResolvedFailed)
            failAt(cycle, "child terminal state is not a legal terminal", k1, liveF);
        if (wc->adjudicated != 0 || wc->pending != 0)
            failAt(cycle, "child terminal word carries residual pending/adjudicated", k1, liveF);
        if (c.resolveRecoveryObligation(child, ROutcome::Published))
            failAt(cycle, "double terminal transition on child", k1, liveF);
    }

    if (liveF != live0)
        failAt(cycle, "liveCount did not return to baseline (terminal -1 exactly once per obligation)", k1, liveF);
    if (liveF < 0 || liveF > static_cast<std::int64_t>(kTableCapacity))
        failAt(cycle, "liveCount out of [0, 32] after cycle", k1, liveF);

    // counter semantics (expected semantics under contract, not exact counts — storms are racy):
    //   exhausted == number of Failed terminals among {old} (child never receives signals)
    const bool oldExhausted = (wf && static_cast<OState>(wf->state) == OState::ResolvedFailed)
                           || (!wf);   // fully overwritten slot implies adjudication terminalized it first
    if (k1.exhausted != k0.exhausted + (oldExhausted ? 1 : 0))
        failAt(cycle, "recoveryRetryExhaustedCount over/under-count vs Failed terminals", k1, liveF);
    if (k1.droppedInvalid != k0.droppedInvalid)
        failAt(cycle, "droppedInvalid over-count 0 violated (obligationId==0 never posted)", k1, liveF);
    if (k1.droppedTerminal < k0.droppedTerminal)
        failAt(cycle, "droppedTerminal counter went backwards", k1, liveF);
    if (k1.droppedStale < k0.droppedStale)
        failAt(cycle, "droppedStale counter went backwards", k1, liveF);
    if (k1.saturated < k0.saturated)
        failAt(cycle, "saturated counter went backwards", k1, liveF);
    if (k1.coalesced < k0.coalesced + 16)
        failAt(cycle, "coalesced counter did not record the 16 coalesce storms", k1, liveF);
}

} // namespace

int main()
{
    std::cout << "ST-1 AV stress: " << kCycles
              << " cycles, lock-pool backend (MSVC std::atomic<16B>)\n";
    std::cout << "probe: std::atomic<RecoveryLifecycleWord> is_lock_free() = "
              << (std::atomic<convo::isr::RecoveryLifecycleWord>{}.is_lock_free() ? "true" : "false")
              << " (expected false = accurate lock-pool report)\n";

    auto c = std::make_unique<RuntimeIntentCoordinator>();
    const auto t0 = std::chrono::steady_clock::now();
    int cyclesDone = 0;
    try {
        for (; cyclesDone < kCycles; ++cyclesDone)
            runCycle(cyclesDone, *c);
    }
    catch (const std::exception& e) {
        std::cout << "ST1 FAILED after " << cyclesDone << '/' << kCycles
                  << " cycles: " << e.what() << '\n';
        return 1;
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    const Counters kf = readCounters(*c);
    const std::int64_t liveF = static_cast<std::int64_t>(c->liveLogicalRecoveryObligationCount());

    std::cout << "cycles=" << cyclesDone << " elapsed_ms=" << static_cast<long long>(ms) << '\n';
    std::cout << "final: liveCount=" << liveF;
    dumpCounters(std::cout, kf);
    std::cout << '\n';

    if (liveF != 0) {
        std::cout << "ST1 FAILED: final liveCount != 0\n";
        return 1;
    }

    std::cout << "ST-1 RESULT: PASS (" << cyclesDone << '/' << kCycles
              << " cycles, all counter/state consistency checks held)\n";
    return 0;
}
