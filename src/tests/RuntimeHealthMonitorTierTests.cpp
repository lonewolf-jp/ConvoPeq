//==============================================================================
// RuntimeHealthMonitorTierTests.cpp — D101-9 Step 5-VI-D
//
// HealthMonitor Threshold Contract Tests (5-VI-B design / 5-VI-C implementation).
//
// ■ Test strategy (Step 5-VI-D §D-14 priority 3: minimal test-only seam):
//   The tier state machine (evaluateRetireChainTiers) is private by design; the sole
//   raw-read site is takeSnapshot() and the sole delta site is evaluateRetireChainTiers()
//   (Gate G1/G2 of 5-VI-B). ISRRetireRouter's terminal accessors are non-virtual, so a
//   fake router cannot intercept them. A one-line `friend struct
//   RuntimeHealthMonitorTierTestAccess;` grants the test direct entry to feed synthetic
//   TrendSnapshot pairs through the REAL delta machine — no production behavior change.
//
// ■ Covered contract groups:
//   D1 bootstrap/snapshot/delta, D2 Tier2/Tier3, D3 Tier4/Tier5/episode latch,
//   D4 correlation/reset/evidence-value semantics.
//
// ■ Explicitly NOT tested here (deferred per Step 5-VI-D §D-15): N>1 exit,
//   terminalPeakResident thresholds, K_terminal(4092/8192), S=2, T_stall_design=30s,
//   quarantine from Terminal events, computeTrend integration, ISRHealthState::Critical
//   wiring, EVENT_TERMINAL_EPISODE_CLEARED(1018).
//
//==============================================================================

#include "audioengine/RuntimeHealthMonitor.h"
#include "audioengine/ISRRetire.h"                              // isr::LifetimeState 完全型（link stub）
#include "audioengine/ISRRuntimePublicationCoordinator.h"       // isr::RuntimeIntentCoordinator 完全型
#include "audioengine/RuntimePublicationOrchestrator.h"         // isr::RuntimePublicationOrchestrator 完全型
#include "core/TimeUtils.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

//==============================================================================
// 簡易 TestRunner (MpscBoundedRingTests と同一パターン)
//==============================================================================
int g_testCount = 0;
int g_failCount = 0;

void checkTrue(const char* name, bool condition)
{
    if (condition) {
        std::cout << "  PASS: " << name << std::endl;
        ++g_testCount;
    } else {
        std::cout << "  FAIL: " << name << " -- condition was false" << std::endl;
        ++g_testCount;
        ++g_failCount;
    }
}

void checkEq(const char* name, long long actual, long long expected)
{
    if (actual == expected) {
        std::cout << "  PASS: " << name << std::endl;
        ++g_testCount;
    } else {
        std::cout << "  FAIL: " << name << " -- actual=" << actual
                  << " expected=" << expected << std::endl;
        ++g_testCount;
        ++g_failCount;
    }
}
} // namespace (test runner helpers)

void checkFalse(const char* name, bool condition)
{
    if (!condition) {
        std::cout << "  PASS: " << name << std::endl;
        ++g_testCount;
    } else {
        std::cout << "  FAIL: " << name << " -- condition was true" << std::endl;
        ++g_testCount;
        ++g_failCount;
    }
}

//==============================================================================
// Event recorder
//==============================================================================
struct RecordedEvent {
    uint32_t code = 0;
    int severity = 0;               // HealthEvent::Severity
    uint64_t value = 0;
    int32_t readerIndex = -1;
    uint64_t readerEpoch = 0;
    uint64_t residencyTimeUs = 0;
};

class EventRecorder
{
public:
    void attach(convo::RuntimeHealthMonitor& m)
    {
        m.setEventCallback([this](const convo::HealthEvent& ev) {
            RecordedEvent r;
            r.code = ev.eventCode;
            r.severity = static_cast<int>(ev.severity);
            r.value = ev.value;
            r.readerIndex = ev.readerIndex;
            r.readerEpoch = ev.readerEpoch;
            r.residencyTimeUs = ev.residencyTimeUs;
            events.push_back(r);
        });
    }
    size_t count(uint32_t code) const
    {
        size_t n = 0;
        for (const auto& e : events) if (e.code == code) ++n;
        return n;
    }
    bool has(uint32_t code) const { return count(code) > 0; }
    const RecordedEvent* last(uint32_t code) const
    {
        const RecordedEvent* found = nullptr;
        for (const auto& e : events) if (e.code == code) found = &e;
        return found;
    }
    void clear() { events.clear(); }
    std::vector<RecordedEvent> events;
};

//==============================================================================
// Test-only access seam (friend declared in RuntimeHealthMonitor.h)
//==============================================================================
namespace convo {
struct RuntimeHealthMonitorTierTestAccess {
    using Snap = TrendSnapshot;

    // Feed one tick through the REAL delta machine, then roll the previous sample.
    static void feedTick(RuntimeHealthMonitor& m, const Snap& now)
    {
        m.evaluateRetireChainTiers(now, m.m_prevTickSnapshot_);
        m.m_prevTickSnapshot_ = now;
        m.m_prevTickSnapshotValid_ = true;
    }
    static void primePrev(RuntimeHealthMonitor& m, const Snap& prev)
    {
        m.m_prevTickSnapshot_ = prev;
        m.m_prevTickSnapshotValid_ = true;
    }
    static void invalidatePrev(RuntimeHealthMonitor& m) { m.m_prevTickSnapshotValid_ = false; }
    static bool admissionLatched(const RuntimeHealthMonitor& m) { return m.m_terminalAdmissionLatched_; }
    static bool growthLatched(const RuntimeHealthMonitor& m) { return m.m_terminalGrowthSustainedLatched_; }
    static unsigned growthTicks(const RuntimeHealthMonitor& m) { return m.m_terminalGrowthTicks_; }
    static void setStuckDiagnosis(RuntimeHealthMonitor& m,
                                  int32_t idx, std::uint64_t epoch, std::uint64_t residencyUs)
    {
        m.m_lastStuckDiagnosis_.isStuck = true;
        m.m_lastStuckDiagnosis_.readerIndex = idx;
        m.m_lastStuckDiagnosis_.readerEpoch = epoch;
        m.m_lastStuckDiagnosis_.residencyTimeUs = residencyUs;
    }
    static void clearStuckDiagnosis(RuntimeHealthMonitor& m)
    {
        m.m_lastStuckDiagnosis_ = RuntimeHealthMonitor::CachedStuckDiagnosis{};
    }
    static void setLastEvidenceUs(RuntimeHealthMonitor& m, std::uint64_t us)
    {
        m.m_lastTerminalEvidenceUs_ = us;
    }
};
} // namespace convo

using convo::RuntimeHealthMonitor;
using convo::RuntimeHealthMonitorTierTestAccess;
using Snap = convo::TrendSnapshot;

//==============================================================================
// Link stubs (Step 5-VI-D §D-14): the tests drive evaluateRetireChainTiers(),
// emitTerminalChainEvent(), reset() and the ctor directly — tick() is NEVER called.
// The out-of-line definitions below are referenced by unrelated monitor functions
// (tick / checkPublicationStall / checkRetireStall) compiled from
// RuntimeHealthMonitor.cpp; they are stubbed here so the test TU links standalone
// without dragging in the engine graph. Production code is untouched.
//==============================================================================
namespace convo {

RuntimePolicyEngine::RuntimePolicyEngine() noexcept = default;

bool RecoveryBudget::isExhausted(std::uint64_t) const noexcept { return false; }
bool RecoveryBudget::isStormDetected(RecoveryAction, std::uint64_t) const noexcept { return false; }
void RecoveryBudget::record(RecoveryAction, std::uint64_t) noexcept {}
void RecoveryBudget::recordCycleCompletion(std::uint64_t) noexcept {}
void RecoveryBudget::recordHeavyReach(std::uint64_t) noexcept {}
void RecoveryBudget::reset() noexcept {}

PolicyDecision RuntimePolicyEngine::evaluateAggregate(MonitorState, MonitorState, MonitorState,
                                                      MonitorState, MonitorState, MonitorState) noexcept
{
    return PolicyDecision{};
}
bool RuntimePolicyEngine::canExecute(RecoveryAction) const noexcept { return false; }
void RuntimePolicyEngine::markExecuted(RecoveryAction) noexcept {}
void RuntimePolicyEngine::reset() noexcept {}
void RuntimePolicyEngine::markForVerification(RecoveryAction, const TrendSnapshot&) noexcept {}
VerificationEntry& RuntimePolicyEngine::getEntry(RecoveryAction) noexcept
{
    static VerificationEntry e;
    return e;
}
const VerificationEntry& RuntimePolicyEngine::getEntry(RecoveryAction) const noexcept
{
    static VerificationEntry e;
    return e;
}
void RuntimePolicyEngine::resetVerification() noexcept {}
bool RuntimePolicyEngine::hasPendingVerification() const noexcept { return false; }
void RuntimePolicyEngine::markExecutedCritical(RecoveryAction) noexcept {}
struct RecoveryBudget& RuntimePolicyEngine::getBudget() noexcept
{
    static RecoveryBudget b;
    return b;
}
const struct RecoveryBudget& RuntimePolicyEngine::getBudget() const noexcept
{
    static RecoveryBudget b;
    return b;
}

std::uint64_t isr::LifetimeState::pendingIntentCount() const noexcept { return 0; }
std::uint64_t isr::RuntimeIntentCoordinator::getPublicationBacklogCount() const noexcept { return 0; }
std::uint64_t isr::RuntimePublicationOrchestrator::getMaxDeferredAgeMs() const noexcept { return 0; }

} // namespace convo

constexpr uint32_t kEvQEngaged  = convo::EVENT_EMERGENCY_Q_ENGAGED;          // 1014
constexpr uint32_t kEvOverflow  = convo::EVENT_QUARANTINE_OVERFLOW_DETECTED; // 1015
constexpr uint32_t kEvAdmission = convo::EVENT_TERMINAL_ADMISSION;           // 1016
constexpr uint32_t kEvGrowth    = convo::EVENT_TERMINAL_GROWTH_SUSTAINED;    // 1017

Snap makeSnap(std::uint64_t store, std::uint64_t resident,
              std::uint64_t eqResident, std::uint64_t overflow,
              std::uint64_t minEpoch = 0,
              std::uint32_t readers = 0, std::uint64_t pending = 0)
{
    Snap s;
    s.pendingRetire = pending;
    s.activeReaderCount = readers;
    s.terminalStoreCount = store;
    s.terminalReclaimResidentCount = resident;
    s.emergencyQuarantineResidentCount = eqResident;
    s.quarantineOverflowCount = overflow;
    s.minReaderEpoch = minEpoch;
    return s;
}

//==============================================================================
// D-2: Bootstrap — Tier 2 evaluated, Tiers 3-5 skipped on first tick
//==============================================================================
bool testBootstrapSkip()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    // All deltas WOULD be positive if a prev sample existed — none does.
    RuntimeHealthMonitorTierTestAccess::feedTick(
        m, makeSnap(/*store*/5, /*resident*/2, /*eq*/1, /*overflow*/3));

    checkEq("bootstrap: exactly 1 event emitted", static_cast<long long>(rec.events.size()), 1);
    checkTrue("bootstrap: 1014 fired (absolute gauge)", rec.has(kEvQEngaged));
    checkFalse("bootstrap: no 1015", rec.has(kEvOverflow));
    checkFalse("bootstrap: no 1016", rec.has(kEvAdmission));
    checkFalse("bootstrap: no 1017", rec.has(kEvGrowth));
    checkFalse("bootstrap: admission not latched", RuntimeHealthMonitorTierTestAccess::admissionLatched(m));
    return true;
}

//==============================================================================
// D-3: Tier 2 engage / hold / clear / re-engage
//==============================================================================
bool testTier2EngageClearReengage()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);

    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(0,0,0,0));
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(0,0,1,0));   // E 0→1
    checkEq("tier2: engage fires once", static_cast<long long>(rec.count(kEvQEngaged)), 1);
    checkEq("tier2: severity is Warning",
            rec.last(kEvQEngaged) ? static_cast<long long>(rec.last(kEvQEngaged)->severity) : -1,
            static_cast<long long>(convo::HealthEvent::Severity::Warning));

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(0,0,1,0));   // E 1→1
    checkEq("tier2: hold does not re-fire", static_cast<long long>(rec.count(kEvQEngaged)), 1);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(0,0,0,0));   // E 1→0
    checkEq("tier2: clear is silent", static_cast<long long>(rec.count(kEvQEngaged)), 1);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(0,0,1,0));   // E 0→1 again
    checkEq("tier2: re-engage fires again", static_cast<long long>(rec.count(kEvQEngaged)), 2);
    return true;
}

//==============================================================================
// D-4: Tier 3 — Q+E AGGREGATE cumulative overflow delta
//==============================================================================
bool testTier3OverflowDelta()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(0,0,0,10));

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(0,0,0,11));  // 10→11
    checkEq("tier3: first burst fires once", static_cast<long long>(rec.count(kEvOverflow)), 1);
    checkEq("tier3: severity is Warning",
            rec.last(kEvOverflow) ? static_cast<long long>(rec.last(kEvOverflow)->severity) : -1,
            static_cast<long long>(convo::HealthEvent::Severity::Warning));
    checkEq("tier3: value carries cumulative counter",
            rec.last(kEvOverflow) ? static_cast<long long>(rec.last(kEvOverflow)->value) : -1, 11);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(0,0,0,12));  // 11→12 (still Warning state)
    checkEq("tier3: consecutive increase does not re-fire while Warning held",
            static_cast<long long>(rec.count(kEvOverflow)), 1);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(0,0,0,12));  // 12→12
    checkEq("tier3: flat counter emits nothing", static_cast<long long>(rec.count(kEvOverflow)), 1);
    return true;
}

//==============================================================================
// D-5: Tier 4 single admission + episode latch (no re-fire while Δstore>0 continues)
//==============================================================================
bool testTier4SingleAdmissionAndLatch()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,0,0,0));

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(101,1,0,0)); // store 100→101
    checkEq("tier4: admission fires exactly once", static_cast<long long>(rec.count(kEvAdmission)), 1);
    checkTrue("tier4: admission latched", RuntimeHealthMonitorTierTestAccess::admissionLatched(m));
    checkEq("tier4: severity is Error",
            rec.last(kEvAdmission) ? static_cast<long long>(rec.last(kEvAdmission)->severity) : -1,
            static_cast<long long>(convo::HealthEvent::Severity::Error));
    checkEq("tier4: value = cumulative store count",
            rec.last(kEvAdmission) ? static_cast<long long>(rec.last(kEvAdmission)->value) : -1, 101);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(102,2,0,0)); // store 101→102
    checkEq("tier4: continued Δstore>0 does not re-fire (episode entry, not per-tick)",
            static_cast<long long>(rec.count(kEvAdmission)), 1);
    return true;
}

//==============================================================================
// D-6: Tier 4 periodic evidence at 10 s boundary (no double-fire on transition tick)
//==============================================================================
bool testTier4PeriodicEvidence()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,0,0,0));

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(101,1,0,0)); // transition tick
    checkEq("periodic: transition fires once", static_cast<long long>(rec.count(kEvAdmission)), 1);

    // <10s: no evidence
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(102,1,0,0));
    checkEq("periodic: below interval emits nothing",
            static_cast<long long>(rec.count(kEvAdmission)), 1);

    // ≥10s since last evidence → periodic evidence re-emission
    const uint64_t nowUs = convo::getCurrentTimeUs();
    RuntimeHealthMonitorTierTestAccess::setLastEvidenceUs(m, nowUs - 11'000'000ULL);
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(103,1,0,0));
    checkEq("periodic: past interval re-emits evidence",
            static_cast<long long>(rec.count(kEvAdmission)), 2);

    // immediately after evidence refresh → suppressed again (double-fire suppression)
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(104,1,0,0));
    checkEq("periodic: refreshed timer suppresses immediate re-fire",
            static_cast<long long>(rec.count(kEvAdmission)), 2);
    return true;
}

//==============================================================================
// D-7 Case 1: Tier 5 positive growth ×2 (no event on first tick)
//==============================================================================
bool testTier5PositiveX2()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,0,0,0));

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(100,1,0,0)); // res 0→1
    checkEq("tier5 case1: ticks=1 after first growth",
            static_cast<long long>(RuntimeHealthMonitorTierTestAccess::growthTicks(m)), 1);
    checkEq("tier5 case1: no 1017 on first growth", static_cast<long long>(rec.count(kEvGrowth)), 0);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(100,2,0,0)); // res 1→2
    checkEq("tier5 case1: ticks=2 after second growth",
            static_cast<long long>(RuntimeHealthMonitorTierTestAccess::growthTicks(m)), 2);
    checkEq("tier5 case1: 1017 latches on second consecutive growth",
            static_cast<long long>(rec.count(kEvGrowth)), 1);
    checkTrue("tier5 case1: growth latched", RuntimeHealthMonitorTierTestAccess::growthLatched(m));

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(100,3,0,0)); // res 2→3
    checkEq("tier5 case1: no re-fire while latched", static_cast<long long>(rec.count(kEvGrowth)), 1);
    return true;
}

//==============================================================================
// D-7 Case 2: plateau resets the growth counter (Δ==0 contract)
//==============================================================================
bool testTier5PlateauReset()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,1,0,0));

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(100,2,0,0)); // 1→2 : ticks=1
    checkEq("tier5 case2: ticks=1", static_cast<long long>(RuntimeHealthMonitorTierTestAccess::growthTicks(m)), 1);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(100,2,0,0)); // 2→2 : reset
    checkEq("tier5 case2: plateau resets ticks to 0",
            static_cast<long long>(RuntimeHealthMonitorTierTestAccess::growthTicks(m)), 0);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(100,3,0,0)); // 2→3 : ticks=1 only
    checkEq("tier5 case2: post-plateau growth is first (not second) consecutive",
            static_cast<long long>(RuntimeHealthMonitorTierTestAccess::growthTicks(m)), 1);
    checkEq("tier5 case2: 1017 never fired", static_cast<long long>(rec.count(kEvGrowth)), 0);
    return true;
}

//==============================================================================
// D-7 Case 3: drain resets the growth counter (Δ<0 contract, healthy recovery)
//==============================================================================
bool testTier5DrainReset()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,2,0,0));

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(100,3,0,0)); // +1 : ticks=1
    checkEq("tier5 case3: ticks=1", static_cast<long long>(RuntimeHealthMonitorTierTestAccess::growthTicks(m)), 1);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(100,2,0,0)); // -1 : drain, reset
    checkEq("tier5 case3: drain resets ticks to 0",
            static_cast<long long>(RuntimeHealthMonitorTierTestAccess::growthTicks(m)), 0);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(100,3,0,0)); // +1 : first again
    checkEq("tier5 case3: post-drain growth is first consecutive",
            static_cast<long long>(RuntimeHealthMonitorTierTestAccess::growthTicks(m)), 1);
    checkEq("tier5 case3: 1017 never fired", static_cast<long long>(rec.count(kEvGrowth)), 0);
    return true;
}

//==============================================================================
// D-8: Tier 4 / Tier 5 independence
//==============================================================================
bool testIndependenceAdmissionWithoutGrowth()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,0,0,0));
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(101,0,0,0)); // store+1, res flat
    checkTrue("indep A: 1016 fired", rec.has(kEvAdmission));
    checkFalse("indep A: 1017 not fired", rec.has(kEvGrowth));
    checkEq("indep A: growthTicks==0",
            static_cast<long long>(RuntimeHealthMonitorTierTestAccess::growthTicks(m)), 0);
    return true;
}

bool testIndependenceGrowthWithoutAdmission()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,0,0,0));
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(100,1,0,0)); // res 0→1, store flat
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(100,2,0,0)); // res 1→2
    checkFalse("indep B: 1016 not fired (no Δstore)", rec.has(kEvAdmission));
    checkTrue("indep B: 1017 fired (sustained growth)", rec.has(kEvGrowth));
    return true;
}

//==============================================================================
// D-9: Episode exit (N=1) and re-arm
//==============================================================================
bool testEpisodeExitAndRearm()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,0,0,0));

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(101,1,0,0)); // episode starts
    checkTrue("exit: episode latched", RuntimeHealthMonitorTierTestAccess::admissionLatched(m));

    // Drain complete: resident back to 0 AND admissions stopped.
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(101,0,0,0));
    checkFalse("exit: latch cleared on resident==0 ∧ dStore==0",
               RuntimeHealthMonitorTierTestAccess::admissionLatched(m));
    checkFalse("exit: growth latch also cleared",
               RuntimeHealthMonitorTierTestAccess::growthLatched(m));
    checkEq("exit: growthTicks cleared",
            static_cast<long long>(RuntimeHealthMonitorTierTestAccess::growthTicks(m)), 0);

    // New episode: another admission re-fires 1016.
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(102,1,0,0));
    checkEq("re-arm: new episode fires 1016 again",
            static_cast<long long>(rec.count(kEvAdmission)), 2);
    return true;
}

//==============================================================================
// D-11: Reader correlation — correlated / suspected / uncorrelated
//==============================================================================
bool testCorrelationCorrelated()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    RuntimeHealthMonitorTierTestAccess::setStuckDiagnosis(m, /*idx*/7, /*epoch*/555, /*residency*/123456);
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,0,0,0));
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(101,1,0,0));

    const auto* ev = rec.last(kEvAdmission);
    checkTrue("corr: 1016 fired", ev != nullptr);
    checkEq("corr: readerIndex propagated", ev ? static_cast<long long>(ev->readerIndex) : -99, 7);
    checkEq("corr: readerEpoch propagated", ev ? static_cast<long long>(ev->readerEpoch) : -99, 555);
    checkEq("corr: residencyTimeUs propagated",
            ev ? static_cast<long long>(ev->residencyTimeUs) : -99, 123456);
    return true;
}

bool testCorrelationSuspected()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    // No stuck diagnosis (cache empty). readers>0 with minReaderEpoch stagnation across
    // the pair — event still fires, but reader fields remain UNSET (no invented verdict).
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,0,0,0,/*minEpoch*/42,/*readers*/1));
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(101,1,0,0,/*minEpoch*/42,/*readers*/1));

    const auto* ev = rec.last(kEvAdmission);
    checkTrue("suspected: 1016 fired", ev != nullptr);
    checkEq("suspected: readerIndex left unset",
            ev ? static_cast<long long>(ev->readerIndex) : 0, -1);
    checkEq("suspected: readerEpoch left unset",
            ev ? static_cast<long long>(ev->readerEpoch) : 0, 0);
    return true;
}

bool testCorrelationUncorrelated()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    RuntimeHealthMonitorTierTestAccess::clearStuckDiagnosis(m);
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,0,0,0,/*minEpoch*/10,/*readers*/0));
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(101,1,0,0,/*minEpoch*/77,/*readers*/0));

    const auto* ev = rec.last(kEvAdmission);
    checkTrue("uncorr: 1016 fired (cause open)", ev != nullptr);
    checkEq("uncorr: reader fields unset",
            ev ? static_cast<long long>(ev->readerIndex) : 0, -1);
    return true;
}

//==============================================================================
// D-10: Reset hygiene — no cross-episode carryover
//==============================================================================
bool testResetHygiene()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);

    // Build up full state: engaged warning, latches, ticks, prev snapshot, stale timer.
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(100,0,0,0));
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(101,1,0,0));   // admission latched
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(101,2,0,0));   // growth ticks=1
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(101,3,1,5));   // E engaged + overflow warning
    checkTrue("reset: precondition — admission latched",
              RuntimeHealthMonitorTierTestAccess::admissionLatched(m));

    m.reset();

    checkFalse("reset: admission latch cleared",
               RuntimeHealthMonitorTierTestAccess::admissionLatched(m));
    checkFalse("reset: growth latch cleared",
               RuntimeHealthMonitorTierTestAccess::growthLatched(m));
    checkEq("reset: growthTicks cleared",
            static_cast<long long>(RuntimeHealthMonitorTierTestAccess::growthTicks(m)), 0);

    // Post-reset tick behaves as BOOTSTRAP even though values continue from pre-reset:
    // deltas vs the wiped prev snapshot must NOT produce Tier 3-5 events.
    rec.clear();
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(102,4,1,6));
    checkFalse("reset: no 1015 from carried-over delta", rec.has(kEvOverflow));
    checkFalse("reset: no 1016 from carried-over delta", rec.has(kEvAdmission));
    checkFalse("reset: no 1017 from carried-over delta", rec.has(kEvGrowth));
    checkEq("reset: periodic-evidence timer cleared (no surprise evidence)",
            static_cast<long long>(rec.count(kEvAdmission)), 0);

    // And a fresh episode can start normally afterwards.
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(103,5,1,6));
    checkTrue("reset: fresh episode fires 1016 normally", rec.has(kEvAdmission));
    return true;
}

//==============================================================================
// D-12: Event payload value semantics (per-code primary metric)
//==============================================================================
bool testPayloadValueSemantics()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);

    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(0,0,0,40));
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(0,0,3,41));   // E engage + overflow
    checkEq("payload: 1014 value = E resident",
            rec.last(kEvQEngaged) ? static_cast<long long>(rec.last(kEvQEngaged)->value) : -1, 3);
    checkEq("payload: 1015 value = cumulative overflow",
            rec.last(kEvOverflow) ? static_cast<long long>(rec.last(kEvOverflow)->value) : -1, 41);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(500,1,3,41)); // admission
    checkEq("payload: 1016 value = cumulative store",
            rec.last(kEvAdmission) ? static_cast<long long>(rec.last(kEvAdmission)->value) : -1, 500);

    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(500,2,3,41)); // growth ×1
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(500,9,3,41)); // growth ×2 → latch
    // 1017 latches on the SECOND consecutive growth tick, where resident was still 2;
    // the later jump to 9 arrives after the latch and does not re-fire.
    checkEq("payload: 1017 value = current resident at latch",
            rec.last(kEvGrowth) ? static_cast<long long>(rec.last(kEvGrowth)->value) : -1, 2);
    return true;
}

//==============================================================================
// Wraparound safety: signed delta near u64 magnitudes stays monotonic-positive
//==============================================================================
bool testSignedDeltaLargeValues()
{
    RuntimeHealthMonitor m;
    EventRecorder rec;
    rec.attach(m);
    constexpr std::uint64_t kBig = (std::uint64_t{1} << 62); // ~4.6e18, far from sign bit
    RuntimeHealthMonitorTierTestAccess::primePrev(m, makeSnap(kBig, 0, 0, 0));
    RuntimeHealthMonitorTierTestAccess::feedTick(m, makeSnap(kBig + 5, 1, 0, 0));
    checkTrue("wraparound: large-magnitude Δstore>0 still admits", rec.has(kEvAdmission));
    return true;
}

int main()
{
    std::cout << "=== RuntimeHealthMonitorTierTests (D101-9 Step 5-VI-D) ===" << std::endl;

    testBootstrapSkip();
    testTier2EngageClearReengage();
    testTier3OverflowDelta();
    testTier4SingleAdmissionAndLatch();
    testTier4PeriodicEvidence();
    testTier5PositiveX2();
    testTier5PlateauReset();
    testTier5DrainReset();
    testIndependenceAdmissionWithoutGrowth();
    testIndependenceGrowthWithoutAdmission();
    testEpisodeExitAndRearm();
    testCorrelationCorrelated();
    testCorrelationSuspected();
    testCorrelationUncorrelated();
    testResetHygiene();
    testPayloadValueSemantics();
    testSignedDeltaLargeValues();

    std::cout << "=== " << g_testCount << " checks, " << g_failCount << " failures ===" << std::endl;
    return g_failCount == 0 ? 0 : 1;
}
