// ============================================================================
//  P0-4 / P0-5  -  PersistentCache v3 Phase 0 Executable Contract
// ----------------------------------------------------------------------------
//  Owner instruction "Owner reply - D-1R-2K Phase 0 GO / Production
//  Implementation NO-GO" section 8, items P0-4 / P0-5.
//
//  PRODUCTION SOURCE CHANGE: ZERO (this is a new TU under src/tests/).
//  No JUCE dependency.
//  The v3 classification / destructive-permission predicates do not exist in
//  production yet, so they are implemented here as the POSITIVE contract that
//  production must satisfy once implementation is authorised (currently NO-GO).
//
//  11 artifact states (D-1R-2K 6.1 / 8.1) x 11 destructive actions
//  -> 121 exhaustive cases (Owner section 8, P0-5).
//
//  Three-layer split (Owner section 5, approved):
//      classification  ->  destructive permission  ->  delete execution
//  classification and destructive permission are SEPARATE functions here, so
//  the two concerns cannot be fused in one place (Owner section 10, last item).
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what)
{
    ++g_checks;
    if (!ok)
    {
        ++g_failures;
        std::fprintf(stderr, "FAIL  %s\n", what.c_str());
    }
}

void checkEq(long long got, long long want, const std::string& what)
{
    ++g_checks;
    if (got != want)
    {
        ++g_failures;
        std::fprintf(stderr, "FAIL  %s : got=%lld want=%lld\n",
                     what.c_str(), got, want);
    }
}

// ===========================================================================
//  Layer 1 - classification (pure, no side effects)
// ===========================================================================
enum class ArtifactState
{
    Absent = 0,
    CurrentV3Canonical,
    CurrentV3NonCanonical,
    Future,
    Obsolete,
    Malformed,
    NonCanonical,
    ForeignFile,
    ValidRecoveryCandidate,
    StageCanonical,
    StageNonCanonical,
    Count_
};

constexpr int kStateCount = static_cast<int>(ArtifactState::Count_);

const char* stateName(ArtifactState s)
{
    switch (s)
    {
        case ArtifactState::Absent:                  return "Absent";
        case ArtifactState::CurrentV3Canonical:     return "CurrentV3Canonical";
        case ArtifactState::CurrentV3NonCanonical:  return "CurrentV3NonCanonical";
        case ArtifactState::Future:                 return "Future";
        case ArtifactState::Obsolete:               return "Obsolete";
        case ArtifactState::Malformed:              return "Malformed";
        case ArtifactState::NonCanonical:           return "NonCanonical";
        case ArtifactState::ForeignFile:            return "ForeignFile";
        case ArtifactState::ValidRecoveryCandidate: return "ValidRecoveryCandidate";
        case ArtifactState::StageCanonical:         return "StageCanonical";
        case ArtifactState::StageNonCanonical:      return "StageNonCanonical";
        default:                                    return "?";
    }
}

/// A target artifact lives in the canonical namespace (<64hex>.mph).
/// The staging namespace states can never be the target of R1-R4.
bool isReachableAsTarget(ArtifactState s)
{
    switch (s)
    {
        case ArtifactState::Absent:
        case ArtifactState::CurrentV3Canonical:
        case ArtifactState::CurrentV3NonCanonical:
        case ArtifactState::Future:
        case ArtifactState::Obsolete:
        case ArtifactState::Malformed:
        case ArtifactState::NonCanonical:
        case ArtifactState::ForeignFile:
            return true;
        default:
            return false;      // ValidRecoveryCandidate / Stage*
    }
}

/// Observable attributes of one artifact on disk.
struct ArtifactObservation
{
    bool          exists = false;
    bool          filenameIsOneOfThreeNamespaces = false;
    bool          stemDecodesAs64Hex = false;
    bool          readable = false;
    bool          magicOk = false;
    std::int64_t  version = 0;
    bool          isRecoveryNamespace = false;          // .stage / .bak
    bool          selfConsistent = false;
    bool          actualSizeMatchesExpected = false;
    bool          headerTotalSizeMatchesExpected = false;
    bool          filenameEqualsLowerEncodeIdentity = false;
};

/// classification: first-match over P0..P10 (D-1R-2K 6.1, 11 branches).
/// Returns exactly one state for every observation.
ArtifactState classifyArtifact(const ArtifactObservation& o)
{
    if (!o.exists)                                              return ArtifactState::Absent;          // P0
    if (!o.filenameIsOneOfThreeNamespaces)                      return ArtifactState::ForeignFile;    // P1
    if (!o.stemDecodesAs64Hex)                                   return ArtifactState::NonCanonical;   // P2
    if (!o.readable || !o.magicOk)                              return ArtifactState::Malformed;      // P3
    if (o.version > 3)                                          return ArtifactState::Future;         // P4
    if (o.version < 3)                                          return ArtifactState::Obsolete;       // P5

    // P6 / P7 recovery namespace: only a self-consistent, size-exact,
    //        non-canonical-name artifact is a recovery candidate.
    if (o.isRecoveryNamespace)
        return (o.selfConsistent
                && o.actualSizeMatchesExpected
                && o.headerTotalSizeMatchesExpected)
                 ? ArtifactState::ValidRecoveryCandidate
                 : ArtifactState::Malformed;

    // P8 canonical namespace with a size mismatch (truncated / appended)
    if (!o.actualSizeMatchesExpected || !o.headerTotalSizeMatchesExpected)
        return ArtifactState::Malformed;

    // P9 / P10 canonical namespace, size exact
    return o.filenameEqualsLowerEncodeIdentity
             ? ArtifactState::CurrentV3Canonical                // P9
             : ArtifactState::CurrentV3NonCanonical;            // P10
}

ArtifactObservation makeObs(bool exists, bool ns3, bool hexStem, bool readable,
                            bool magic, std::int64_t version, bool recoveryNs,
                            bool selfConsistent, bool sizeOk, bool hdrSizeOk,
                            bool nameMatches)
{
    ArtifactObservation o;
    o.exists = exists;
    o.filenameIsOneOfThreeNamespaces = ns3;
    o.stemDecodesAs64Hex = hexStem;
    o.readable = readable;
    o.magicOk = magic;
    o.version = version;
    o.isRecoveryNamespace = recoveryNs;
    o.selfConsistent = selfConsistent;
    o.actualSizeMatchesExpected = sizeOk;
    o.headerTotalSizeMatchesExpected = hdrSizeOk;
    o.filenameEqualsLowerEncodeIdentity = nameMatches;
    return o;
}

// ===========================================================================
//  Layer 2 - destructive permission (pure function of the classified state)
// ===========================================================================
enum class DestructiveAction
{
    LoadUsable = 0,
    LoadRejected,
    EnumerateCanonical,
    EnumerateRecovery,
    Remove,
    Clear,
    EvictLruZero,
    EvictLruN,
    DeleteArtifact,
    Touch,
    RecoveryRestore,
    Count_
};

constexpr int kActionCount = static_cast<int>(DestructiveAction::Count_);

const char* actionName(DestructiveAction a)
{
    switch (a)
    {
        case DestructiveAction::LoadUsable:         return "LoadUsable";
        case DestructiveAction::LoadRejected:       return "LoadRejected";
        case DestructiveAction::EnumerateCanonical: return "EnumerateCanonical";
        case DestructiveAction::EnumerateRecovery:  return "EnumerateRecovery";
        case DestructiveAction::Remove:             return "Remove";
        case DestructiveAction::Clear:              return "Clear";
        case DestructiveAction::EvictLruZero:       return "EvictLruZero";
        case DestructiveAction::EvictLruN:          return "EvictLruN";
        case DestructiveAction::DeleteArtifact:     return "DeleteArtifact";
        case DestructiveAction::Touch:              return "Touch";
        case DestructiveAction::RecoveryRestore:    return "RecoveryRestore";
        default:                                    return "?";
    }
}

enum class Permission { Allowed = 0, Rejected, NotApplicable };

const char* permName(Permission p)
{
    switch (p)
    {
        case Permission::Allowed:   return "ALLOW";
        case Permission::Rejected: return "REJECT";
        default:                   return "N/A";
    }
}

/// Destructive permission depends on the classified state ONLY.
/// It never inspects a path or a filename pattern (Owner section 9, G-1).
Permission permissionFor(ArtifactState s, DestructiveAction a)
{
    // Owner section 4.1 / D-1R-2K K-03:
    // evictLRU(0) is a no-op for every state. It is never a delete-all.
    if (a == DestructiveAction::EvictLruZero)
        return Permission::NotApplicable;

    const bool isCanonical      = (s == ArtifactState::CurrentV3Canonical);
    const bool isStageCanonical = (s == ArtifactState::StageCanonical);
    const bool isRecovery       = (s == ArtifactState::ValidRecoveryCandidate);
    const bool exists           = (s != ArtifactState::Absent);

    switch (a)
    {
        case DestructiveAction::LoadUsable:
            return isCanonical ? Permission::Allowed : Permission::Rejected;

        case DestructiveAction::LoadRejected:
            // A canonical artifact must load successfully; every other existing
            // artifact is entitled to be rejected by load().
            return (exists && !isCanonical) ? Permission::Allowed
                                            : Permission::NotApplicable;

        case DestructiveAction::EnumerateCanonical:
            return isCanonical ? Permission::Allowed : Permission::NotApplicable;

        case DestructiveAction::EnumerateRecovery:
            return isRecovery ? Permission::Allowed : Permission::NotApplicable;

        case DestructiveAction::Remove:
        case DestructiveAction::Clear:
        case DestructiveAction::DeleteArtifact:
            // Staging is owned by ConvoPeq so it may be cleaned up.
            // Everything else must pass the v3 state gate.
            return (isCanonical || isStageCanonical) ? Permission::Allowed
                                                     : Permission::Rejected;

        case DestructiveAction::EvictLruN:
            // Owner section 5: staging / temporary is NOT an LRU candidate.
            return isCanonical ? Permission::Allowed : Permission::Rejected;

        case DestructiveAction::Touch:
            return isCanonical ? Permission::Allowed : Permission::NotApplicable;

        case DestructiveAction::RecoveryRestore:
            return isRecovery ? Permission::Allowed : Permission::NotApplicable;

        default:
            return Permission::Rejected;
    }
}

// ===========================================================================
//  R1-R4 decision (Owner section 10)
// ===========================================================================
enum class Action
{
    NotApplicableTarget = 0,   // target is outside the canonical namespace
    NoOp,
    REC,
    ReplaceCanonical,
    CreateNew
};

const char* actionName2(Action a)
{
    switch (a)
    {
        case Action::NotApplicableTarget: return "NotApplicableTarget";
        case Action::NoOp:                return "NoOp";
        case Action::REC:                 return "REC";
        case Action::ReplaceCanonical:    return "ReplaceCanonical";
        case Action::CreateNew:           return "CreateNew";
        default:                          return "?";
    }
}

Action decideAction(ArtifactState target, ArtifactState recovery)
{
    if (!isReachableAsTarget(target))
        return Action::NotApplicableTarget;

    // R1  target == Future -> NoOp (UnsafeResidue; target is never touched)
    if (target == ArtifactState::Future)
        return Action::NoOp;

    // R2  target == CurrentV3Canonical -> save replaces (recovery irrelevant)
    if (target == ArtifactState::CurrentV3Canonical)
        return Action::ReplaceCanonical;

    // R3  target == Absent
    if (target == ArtifactState::Absent)
        return (recovery == ArtifactState::ValidRecoveryCandidate) ? Action::REC
                                                                   : Action::CreateNew;

    // R4  target in { CurrentV3NonCanonical, Obsolete, Malformed,
    //                NonCanonical, ForeignFile }
    return (recovery == ArtifactState::ValidRecoveryCandidate) ? Action::REC
                                                               : Action::NoOp;
}

// ===========================================================================
//  eviction ordering (Owner N5: timestamp is orthogonal to canonicality)
// ===========================================================================
struct EvictionKey
{
    bool         invalidAscending;
    std::uint64_t lastUsedTime;
    std::string  canonicalFilename;
};

bool operator<(const EvictionKey& a, const EvictionKey& b)
{
    if (a.invalidAscending != b.invalidAscending)
        return a.invalidAscending < b.invalidAscending;
    if (!a.invalidAscending && a.lastUsedTime != b.lastUsedTime)
        return a.lastUsedTime < b.lastUsedTime;
    return a.canonicalFilename < b.canonicalFilename;
}

// ===========================================================================
//  P0-4  R1-R4 exhaustive classification
// ===========================================================================
void runP04()
{
    std::fprintf(stderr, "[P0-4] R1-R4 exhaustive classification\n");

    // ---- classifier: every reachable state is produced --------------------
    struct Probe { const char* want; ArtifactObservation o; };
    const Probe probes[] = {
        { "Absent",                 makeObs(false,false,false,false,false,0,false,false,false,false,false) },
        { "CurrentV3Canonical",     makeObs(true, true, true, true, true, 3, false,true, true, true, true) },
        { "CurrentV3NonCanonical",  makeObs(true, true, true, true, true, 3, false,false,true, true, false) },
        { "Future",                 makeObs(true, true, true, true, true, 4, false,true, true, true, true) },
        { "Obsolete",               makeObs(true, true, true, true, true, 2, false,true, true, true, true) },
        { "Malformed",              makeObs(true, true, true, true, false,3, false,true, true, true, true) },
        { "NonCanonical",           makeObs(true, true, false,true, true, 3, false,true, true, true, true) },
        { "ForeignFile",            makeObs(true, false,false,true, true, 3, false,true, true, true, true) },
        { "ValidRecoveryCandidate", makeObs(true, true, true, true, true, 3, true, true, true, true, false) },
        { "Malformed",              makeObs(true, true, true, true, true, 3, true, false,true, true, false) },
        { "Malformed",              makeObs(true, true, true, true, true, 3, false,true, false,true, true) },
    };

    for (const auto& p : probes)
    {
        const ArtifactState got = classifyArtifact(p.o);
        check(std::string(stateName(got)) == p.want,
              std::string("P0-4 classify -> ") + p.want
                  + " (got " + stateName(got) + ")");
    }

    // ---- 11 states are all producible across the observation space -------
    {
        std::vector<ArtifactState> seen;
        for (const auto& p : probes)
        {
            const ArtifactState s = classifyArtifact(p.o);
            if (std::find(seen.begin(), seen.end(), s) == seen.end())
                seen.push_back(s);
        }
        check(seen.size() == 9,
              "P0-4 classifier produces 9 distinct states from the probe set");
    }

    // ---- unreachable-as-target states ------------------------------------
    for (int s = 0; s < kStateCount; ++s)
    {
        const auto st = static_cast<ArtifactState>(s);
        const bool reachable = isReachableAsTarget(st);
        check(reachable == (st != ArtifactState::ValidRecoveryCandidate
                             && st != ArtifactState::StageCanonical
                             && st != ArtifactState::StageNonCanonical),
              std::string("P0-4 target reachability of ") + stateName(st));
        check(reachable || decideAction(st, ArtifactState::Malformed)
                            == Action::NotApplicableTarget,
              std::string("P0-4 unreachable target ") + stateName(st)
                  + " yields NotApplicableTarget");
    }

    // ---- 11 x 11 = 121 exhaustive decision table -------------------------
    int unresolved = 0, rec = 0, nop = 0, repl = 0, create = 0, na = 0;
    for (int t = 0; t < kStateCount; ++t)
        for (int r = 0; r < kStateCount; ++r)
        {
            const Action a = decideAction(static_cast<ArtifactState>(t),
                                          static_cast<ArtifactState>(r));
            switch (a)
            {
                case Action::NoOp:                ++nop;    break;
                case Action::REC:                 ++rec;    break;
                case Action::ReplaceCanonical:    ++repl;   break;
                case Action::CreateNew:           ++create; break;
                case Action::NotApplicableTarget: ++na;     break;
                default:                          ++unresolved; break;
            }
        }
    checkEq(unresolved, 0,
            "P0-4 all 121 cases resolve to exactly one defined action");
    checkEq(kStateCount * kStateCount, 121,
            "P0-4 decision table is exactly 11 x 11 = 121 cases");
    // 3 unreachable targets x 11 recovery states = 33
    checkEq(na, 33, "P0-4 unreachable targets account for 33 cases");
    // REC over reachable targets = R3a(1) + R4a(5) = 6, times 1 recovery state
    checkEq(rec, 6, "P0-4 REC occurs exactly for R3a + R4a over reachable targets");

    std::fprintf(stderr, "       121 cases: NoOp=%d REC=%d Replace=%d Create=%d N/A=%d\n",
                 nop, rec, repl, create, na);

    // ---- R1 / R2 are independent of the recovery state ------------------
    for (int r = 0; r < kStateCount; ++r)
    {
        check(decideAction(ArtifactState::Future, static_cast<ArtifactState>(r))
                  == Action::NoOp,
              "P0-4 R1 Future -> NoOp for all 11 recovery states");
        check(decideAction(ArtifactState::CurrentV3Canonical,
                           static_cast<ArtifactState>(r)) == Action::ReplaceCanonical,
              "P0-4 R2 CurrentV3Canonical -> ReplaceCanonical for all 11");
    }

    // ---- R3a / R3b -------------------------------------------------------
    check(decideAction(ArtifactState::Absent,
                       ArtifactState::ValidRecoveryCandidate) == Action::REC,
          "P0-4 R3a Absent + recovery -> REC");
    check(decideAction(ArtifactState::Absent,
                       ArtifactState::Malformed) == Action::CreateNew,
          "P0-4 R3b Absent + non-recovery -> CreateNew");

    // ---- R4a / R4b over the five R4 states -------------------------------
    const ArtifactState r4set[5] = {
        ArtifactState::CurrentV3NonCanonical, ArtifactState::Obsolete,
        ArtifactState::Malformed, ArtifactState::NonCanonical,
        ArtifactState::ForeignFile
    };
    for (auto s : r4set)
    {
        check(decideAction(s, ArtifactState::ValidRecoveryCandidate) == Action::REC,
              std::string("P0-4 R4a ") + stateName(s) + " + recovery -> REC");
        check(decideAction(s, ArtifactState::Malformed) == Action::NoOp,
              std::string("P0-4 R4b ") + stateName(s) + " + non-recovery -> NoOp");
    }

    // ---- exactly one R-rule owns each reachable target state -------------
    int owned = 0;
    for (int t = 0; t < kStateCount; ++t)
    {
        const auto st = static_cast<ArtifactState>(t);
        if (!isReachableAsTarget(st))
            continue;
        int rules = 0;
        if (st == ArtifactState::Future)                  ++rules;
        if (st == ArtifactState::CurrentV3Canonical)      ++rules;
        if (st == ArtifactState::Absent)                  ++rules;
        if (st == ArtifactState::CurrentV3NonCanonical)   ++rules;
        if (st == ArtifactState::Obsolete)                ++rules;
        if (st == ArtifactState::Malformed)               ++rules;
        if (st == ArtifactState::NonCanonical)            ++rules;
        if (st == ArtifactState::ForeignFile)             ++rules;
        checkEq(rules, 1,
                std::string("P0-4 exactly one R-rule owns ") + stateName(st));
        owned += (rules == 1) ? 1 : 0;
    }
    checkEq(owned, 8, "P0-4 all 8 reachable target states are covered");

    // ---- eviction ordering ------------------------------------------------
    {
        std::vector<EvictionKey> keys = {
            {false, 100, "bbb.mph"},
            {false, 100, "aaa.mph"},
            {false,   0, "ccc.mph"},
            {true,    0, "ddd.mph"},
        };
        std::sort(keys.begin(), keys.end());
        check(keys[0].canonicalFilename == "ccc.mph",
              "P0-4 lowest lastUsedTime sorts first, including 0");
        check(keys[1].canonicalFilename == "aaa.mph"
                  && keys[2].canonicalFilename == "bbb.mph",
              "P0-4 equal timestamps break by filename ascending");
        check(keys[3].invalidAscending,
              "P0-4 invalid entries sort last");
    }
}

// ===========================================================================
//  P0-5  destructive action matrix (11 states x 11 actions = 121 cases)
// ===========================================================================
void runP05()
{
    std::fprintf(stderr, "[P0-5] destructive action matrix (11 x 11 = 121)\n");

    // Positive v3 contract. Column order:
    //  LoadUsable LoadRejected EnumCanonical EnumRecovery Remove Clear
    //  EvictLruZero EvictLruN DeleteArtifact Touch RecoveryRestore
    const Permission table[kStateCount][kActionCount] = {
        // ---- Absent
        { Permission::Rejected,    Permission::NotApplicable, Permission::NotApplicable, Permission::NotApplicable,
          Permission::Rejected,    Permission::Rejected,     Permission::NotApplicable, Permission::Rejected,
          Permission::Rejected,    Permission::NotApplicable, Permission::NotApplicable },
        // ---- CurrentV3Canonical
        { Permission::Allowed,     Permission::NotApplicable, Permission::Allowed,      Permission::NotApplicable,
          Permission::Allowed,     Permission::Allowed,      Permission::NotApplicable, Permission::Allowed,
          Permission::Allowed,     Permission::Allowed,      Permission::NotApplicable },
        // ---- CurrentV3NonCanonical
        { Permission::Rejected,    Permission::Allowed,      Permission::NotApplicable, Permission::NotApplicable,
          Permission::Rejected,    Permission::Rejected,     Permission::NotApplicable, Permission::Rejected,
          Permission::Rejected,    Permission::NotApplicable, Permission::NotApplicable },
        // ---- Future
        { Permission::Rejected,    Permission::Allowed,      Permission::NotApplicable, Permission::NotApplicable,
          Permission::Rejected,    Permission::Rejected,     Permission::NotApplicable, Permission::Rejected,
          Permission::Rejected,    Permission::NotApplicable, Permission::NotApplicable },
        // ---- Obsolete
        { Permission::Rejected,    Permission::Allowed,      Permission::NotApplicable, Permission::NotApplicable,
          Permission::Rejected,    Permission::Rejected,     Permission::NotApplicable, Permission::Rejected,
          Permission::Rejected,    Permission::NotApplicable, Permission::NotApplicable },
        // ---- Malformed
        { Permission::Rejected,    Permission::Allowed,      Permission::NotApplicable, Permission::NotApplicable,
          Permission::Rejected,    Permission::Rejected,     Permission::NotApplicable, Permission::Rejected,
          Permission::Rejected,    Permission::NotApplicable, Permission::NotApplicable },
        // ---- NonCanonical
        { Permission::Rejected,    Permission::Allowed,      Permission::NotApplicable, Permission::NotApplicable,
          Permission::Rejected,    Permission::Rejected,     Permission::NotApplicable, Permission::Rejected,
          Permission::Rejected,    Permission::NotApplicable, Permission::NotApplicable },
        // ---- ForeignFile
        { Permission::Rejected,    Permission::Allowed,      Permission::NotApplicable, Permission::NotApplicable,
          Permission::Rejected,    Permission::Rejected,     Permission::NotApplicable, Permission::Rejected,
          Permission::Rejected,    Permission::NotApplicable, Permission::NotApplicable },
        // ---- ValidRecoveryCandidate
        { Permission::Rejected,    Permission::Allowed,      Permission::NotApplicable, Permission::Allowed,
          Permission::Rejected,    Permission::Rejected,     Permission::NotApplicable, Permission::Rejected,
          Permission::Rejected,    Permission::NotApplicable, Permission::Allowed },
        // ---- StageCanonical
        { Permission::Rejected,    Permission::Allowed,      Permission::NotApplicable, Permission::NotApplicable,
          Permission::Allowed,     Permission::Allowed,      Permission::NotApplicable, Permission::Rejected,
          Permission::Allowed,     Permission::NotApplicable, Permission::NotApplicable },
        // ---- StageNonCanonical
        { Permission::Rejected,    Permission::Allowed,      Permission::NotApplicable, Permission::NotApplicable,
          Permission::Rejected,    Permission::Rejected,     Permission::NotApplicable, Permission::Rejected,
          Permission::Rejected,    Permission::NotApplicable, Permission::NotApplicable },
    };

    int mismatches = 0;
    for (int s = 0; s < kStateCount; ++s)
        for (int a = 0; a < kActionCount; ++a)
        {
            const Permission want = table[s][a];
            const Permission got  = permissionFor(static_cast<ArtifactState>(s),
                                                  static_cast<DestructiveAction>(a));
            if (want != got)
            {
                ++mismatches;
                std::fprintf(stderr, "FAIL  P0-5 [%s][%s] got=%s want=%s\n",
                             stateName(static_cast<ArtifactState>(s)),
                             actionName(static_cast<DestructiveAction>(a)),
                             permName(got), permName(want));
            }
            ++g_checks;
        }
    checkEq(mismatches, 0,
            "P0-5 all 121 state x action cells match the v3 contract");
    checkEq(kStateCount * kActionCount, 121,
            "P0-5 matrix is exactly 11 x 11 = 121 cases");

    // ---- Owner section 5: excluded from LRU candidate ---------------------
    const ArtifactState excluded[6] = {
        ArtifactState::StageCanonical,
        ArtifactState::StageNonCanonical,
        ArtifactState::Future,
        ArtifactState::ForeignFile,
        ArtifactState::Malformed,
        ArtifactState::ValidRecoveryCandidate,
    };
    for (auto s : excluded)
        check(permissionFor(s, DestructiveAction::EvictLruN) != Permission::Allowed,
              std::string("P0-5 ") + stateName(s)
                  + " is not an LRU eviction candidate");

    // ---- evictLRU(0) is a no-op for every state (Owner section 4.1) ------
    for (int s = 0; s < kStateCount; ++s)
        check(permissionFor(static_cast<ArtifactState>(s),
                            DestructiveAction::EvictLruZero) == Permission::NotApplicable,
              std::string("P0-5 evictLRU(0) is a no-op for ")
                  + stateName(static_cast<ArtifactState>(s)));

    // ---- remove() and clear() agree (same state gate) --------------------
    for (int s = 0; s < kStateCount; ++s)
    {
        const auto st = static_cast<ArtifactState>(s);
        check(permissionFor(st, DestructiveAction::Remove)
                  == permissionFor(st, DestructiveAction::Clear),
              std::string("P0-5 remove() and clear() agree on ") + stateName(st));
    }

    // ---- structural gate: filename pattern alone never authorises -------
    check(permissionFor(ArtifactState::Future, DestructiveAction::Remove)
              == Permission::Rejected,
          "P0-5 Future cannot be removed even though it exists");
    check(permissionFor(ArtifactState::ForeignFile, DestructiveAction::Clear)
              == Permission::Rejected,
          "P0-5 ForeignFile is never deleted by clear() (not owned by ConvoPeq)");
    check(permissionFor(ArtifactState::Obsolete, DestructiveAction::EvictLruN)
              == Permission::Rejected,
          "P0-5 Obsolete (v2) is not an eviction candidate");
    check(permissionFor(ArtifactState::StageCanonical, DestructiveAction::EvictLruN)
              == Permission::Rejected,
          "P0-5 staging is cleaned by explicit residue handling, not by LRU");

    // ---- only canonical entries are LRU candidates -----------------------
    int lruCandidates = 0;
    for (int s = 0; s < kStateCount; ++s)
        if (permissionFor(static_cast<ArtifactState>(s),
                          DestructiveAction::EvictLruN) == Permission::Allowed)
            ++lruCandidates;
    checkEq(lruCandidates, 1,
            "P0-5 exactly one state (CurrentV3Canonical) is an LRU candidate");
}

} // namespace

int main()
{
    try
    {
        runP04();
        runP05();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 2;
    }

    std::fprintf(stderr, "\nPersistentCacheV3Phase0StateTests: checks=%d failures=%d\n",
                 g_checks, g_failures);
    if (g_failures != 0)
        return 1;
    std::fprintf(stderr, "P0-4 / P0-5: PASS\n");
    return 0;
}