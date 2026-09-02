// BuildErrorClassificationTests.cpp — D101-13 Phase D-3
// Contract test for classifyBuildError() → BuildOutcome (8-value default policy)
// Uses standalone main() with no external framework (existing repo convention).
// Tested: Test A (exact matrix), Test B (table coverage), Test C (classifier/table consistency),
//         Test E (defensive out-of-range fallback). No BuildContext, no scheduler, no retry wiring.

#include <iostream>
#include <string>
#include <stdexcept>

#include "audioengine/BuildErrorPolicy.h"

namespace {
int g_pass = 0;
int g_fail = 0;

void check(bool cond, const char* /*unused*/ = "")
{
    // For internal use; caller prints context on failure
    (void)cond;
}

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << (msg) << " @ " << __LINE__ << "\n"; \
            ++g_fail; \
        } else { \
            ++g_pass; \
        } \
    } while (0)

// Helper: stringify enums via underlying int for diagnostics
const char* toStringFC(convo::FailureClassification v)
{
    switch (v) {
        case convo::FailureClassification::Permanent: return "Permanent";
        case convo::FailureClassification::Transient: return "Transient";
        case convo::FailureClassification::Infrastructure: return "Infrastructure";
        case convo::FailureClassification::Fatal: return "Fatal";
    }
    return "UnknownFC";
}

const char* toStringRD(convo::RetryDisposition v)
{
    switch (v) {
        case convo::RetryDisposition::NoRetry: return "NoRetry";
        case convo::RetryDisposition::RetryBackoff: return "RetryBackoff";
        case convo::RetryDisposition::RetryImmediate: return "RetryImmediate";
    }
    return "UnknownRD";
}

struct Expected {
    convo::BuildError error;
    convo::FailureClassification classification;
    convo::RetryDisposition retry;
};

// D-2 ratified 8-value default policy (matches kBuildErrorDefaultTable)
constexpr Expected kExpected[8] = {
    { convo::BuildError::None,              convo::FailureClassification::Permanent,      convo::RetryDisposition::NoRetry },
    { convo::BuildError::InvalidInput,      convo::FailureClassification::Permanent,      convo::RetryDisposition::NoRetry },
    { convo::BuildError::ResourceUnavailable, convo::FailureClassification::Transient,   convo::RetryDisposition::RetryBackoff },
    { convo::BuildError::MKLFailure,        convo::FailureClassification::Fatal,          convo::RetryDisposition::NoRetry },
    { convo::BuildError::ConvolverFailure,  convo::FailureClassification::Infrastructure, convo::RetryDisposition::RetryBackoff },
    { convo::BuildError::PrepareFailure,    convo::FailureClassification::Infrastructure, convo::RetryDisposition::RetryBackoff },
    { convo::BuildError::WarmupFailed,      convo::FailureClassification::Transient,      convo::RetryDisposition::RetryImmediate },
    { convo::BuildError::InternalError,     convo::FailureClassification::Fatal,          convo::RetryDisposition::NoRetry },
};

// ── Test A — exact policy matrix (3 fields per value) ──
[[nodiscard]] bool runTestA()
{
    bool ok = true;
    for (auto& exp : kExpected) {
        const auto out = convo::classifyBuildError(exp.error);
        const int idx = static_cast<int>(exp.error);
        bool pass = (out.error == exp.error)
                 && (out.classification == exp.classification)
                 && (out.retry == exp.retry);
        if (!pass) {
            std::cerr << "[FAIL] TestA idx=" << idx
                      << " want(" << toStringFC(exp.classification) << "," << toStringRD(exp.retry) << ")"
                      << " got(" << toStringFC(out.classification) << "," << toStringRD(out.retry) << ")\n";
            ++g_fail;
            ok = false;
        } else {
            ++g_pass;
        }
        // Also verify returned error echoes input
        if (out.error != exp.error) {
            std::cerr << "[FAIL] TestA error echo mismatch idx=" << idx << "\n";
            ++g_fail;
            ok = false;
        } else {
            ++g_pass;
        }
    }
    if (ok) std::cerr << "[PASS] TestA exact policy matrix (8x3 fields)\n";
    return ok;
}

// ── Test B — table coverage: kBuildErrorDefaultTable index/error alignment ──
[[nodiscard]] bool runTestB()
{
    bool ok = true;
    constexpr size_t kSize = sizeof(convo::kBuildErrorDefaultTable) / sizeof(convo::BuildOutcome);
    CHECK(kSize == 8u, "kBuildErrorDefaultTable size == 8");
    // Also verify static_assert equivalence at runtime: table covers InternalError+1
    CHECK(kSize == static_cast<size_t>(convo::BuildError::InternalError) + 1u,
          "table size == InternalError+1");
    for (size_t i = 0; i < kSize; ++i) {
        const auto& row = convo::kBuildErrorDefaultTable[i];
        const auto want = static_cast<convo::BuildError>(i);
        if (row.error != want) {
            std::cerr << "[FAIL] TestB index=" << i << " table.error != BuildError(i)\n";
            ++g_fail;
            ok = false;
        } else {
            ++g_pass;
        }
        // Also verify row matches kExpected
        const auto& exp = kExpected[i];
        bool match = (row.classification == exp.classification) && (row.retry == exp.retry);
        if (!match) {
            std::cerr << "[FAIL] TestB index=" << i << " table row != expected\n";
            ++g_fail;
            ok = false;
        } else {
            ++g_pass;
        }
    }
    if (ok) std::cerr << "[PASS] TestB table coverage (index/error alignment)\n";
    return ok;
}

// ── Test C — classifier/table consistency ──
[[nodiscard]] bool runTestC()
{
    bool ok = true;
    for (size_t i = 0; i < 8u; ++i) {
        const auto err = static_cast<convo::BuildError>(i);
        const auto via = convo::classifyBuildError(err);
        const auto& tbl = convo::kBuildErrorDefaultTable[i];
        bool eq = (via.error == tbl.error)
               && (via.classification == tbl.classification)
               && (via.retry == tbl.retry);
        if (!eq) {
            std::cerr << "[FAIL] TestC idx=" << i << " classify != table[i]\n";
            ++g_fail;
            ok = false;
        } else {
            ++g_pass;
        }
    }
    if (ok) std::cerr << "[PASS] TestC classifier/table consistency (8)\n";
    return ok;
}

// ── Test D — kBuildErrorNames coverage (sanity, not policy but table integrity) ──
[[nodiscard]] bool runTestD()
{
    bool ok = true;
    constexpr size_t kN = sizeof(convo::kBuildErrorNames) / sizeof(const char*);
    CHECK(kN == 8u, "kBuildErrorNames size == 8");
    for (size_t i = 0; i < kN; ++i) {
        const auto err = static_cast<convo::BuildError>(i);
        const char* s = convo::classifyBuildErrorToString(err);
        const char* t = convo::kBuildErrorNames[i];
        bool same = (s == t) || (s && t && std::string(s) == std::string(t));
        if (!same) {
            std::cerr << "[FAIL] TestD name mismatch idx=" << i << " via=" << (s ? s : "null") << " table=" << (t ? t : "null") << "\n";
            ++g_fail;
            ok = false;
        } else {
            ++g_pass;
        }
        if (!s || !*s) {
            std::cerr << "[FAIL] TestD empty name idx=" << i << "\n";
            ++g_fail;
            ok = false;
        } else {
            ++g_pass;
        }
    }
    if (ok) std::cerr << "[PASS] TestD names coverage\n";
    return ok;
}

// ── Test E — defensive out-of-range fallback ──
[[nodiscard]] bool runTestE()
{
    bool ok = true;
    auto testOne = [&](convo::BuildError bad) {
        const auto out = convo::classifyBuildError(bad);
        bool pass = (out.error == convo::BuildError::InternalError)
                 && (out.classification == convo::FailureClassification::Fatal)
                 && (out.retry == convo::RetryDisposition::NoRetry);
        if (!pass) {
            std::cerr << "[FAIL] TestE bad=" << static_cast<int>(bad) << "\n";
            ++g_fail;
            ok = false;
        } else {
            ++g_pass;
        }
        const char* s = convo::classifyBuildErrorToString(bad);
        if (!s || std::string(s) != "Unknown") {
            std::cerr << "[FAIL] TestE toString bad=" << static_cast<int>(bad) << " want Unknown got " << (s ? s : "null") << "\n";
            ++g_fail;
            ok = false;
        } else {
            ++g_pass;
        }
    };
    testOne(static_cast<convo::BuildError>(255));
    testOne(static_cast<convo::BuildError>(8));
    testOne(static_cast<convo::BuildError>(100));
    if (ok) std::cerr << "[PASS] TestE defensive fallback (out-of-range → InternalError/Fatal/NoRetry + Unknown)\n";
    return ok;
}

// =============================================================================
// ★ CR-α (ND-07 contract): Site 3 warmup retry — bounded retry + backoff policy
//   T-CRα-1: retryBackoffDelayMs table（0/10/20/40/80/80 + saturation）
//   T-CRα-2: normative default {10,80,2} + kMaxWarmupConsecutiveRetries = 3
//   T-CRα-3: warmupRetryDecision — context / obsolete / max（純関数・Scheduler 不使用）
//   T-CRα-4: disposition → delay mapping（Immediate=0 / Backoff=policy / NoRetry=none）
//   T-CRα-5/6: source inspection（exhaustion で再発行なし・diagLog 項目）は CR-α-2/5 の
//              read-only verification で実施（本ファイルは純関数のみ）
// =============================================================================
[[nodiscard]] bool runTestF()
{
    bool ok = true;
    using convo::BuildError;
    using convo::RetryBackoffPolicy;
    using convo::RetryDisposition;
    using convo::WarmupRetryAction;
    using Action = WarmupRetryAction::Value;

    // --- T-CRα-2: normative default {10,80,2} + max = 3（recovery K=4 と別ドメイン・別値） ---
    {
        constexpr RetryBackoffPolicy p = convo::kDefaultWarmupRetryBackoff;
        CHECK(p.initialDelayMs == 10, "T-CRα-2: default initialDelayMs must be 10");
        CHECK(p.maxDelayMs == 80, "T-CRα-2: default maxDelayMs must be 80");
        CHECK(p.multiplier == 2, "T-CRα-2: default multiplier must be 2");
        CHECK(convo::kMaxWarmupConsecutiveRetries == 3,
              "T-CRα-2: Site 3 warmup retry bound must be 3 (deliberately distinct from recovery K=4)");
    }

    // --- T-CRα-1: retryBackoffDelayMs table + saturation ---
    {
        constexpr RetryBackoffPolicy p = convo::kDefaultWarmupRetryBackoff;
        CHECK(convo::retryBackoffDelayMs(p, 0) == 0, "T-CRα-1: attempt 0 → 0");
        CHECK(convo::retryBackoffDelayMs(p, 1) == 10, "T-CRα-1: attempt 1 → 10");
        CHECK(convo::retryBackoffDelayMs(p, 2) == 20, "T-CRα-1: attempt 2 → 20");
        CHECK(convo::retryBackoffDelayMs(p, 3) == 40, "T-CRα-1: attempt 3 → 40");
        CHECK(convo::retryBackoffDelayMs(p, 4) == 80, "T-CRα-1: attempt 4 → 80 (saturation)");
        CHECK(convo::retryBackoffDelayMs(p, 5) == 80, "T-CRα-1: attempt 5 → 80 (saturated)");
        CHECK(convo::retryBackoffDelayMs(p, 100) == 80, "T-CRα-1: attempt 100 → 80 (no overflow)");
        // saturation 構造: initial > max の policy でも cap される
        constexpr RetryBackoffPolicy inverted { 100, 50, 2 };
        CHECK(convo::retryBackoffDelayMs(inverted, 1) == 50, "T-CRα-1: initial > max is capped");
    }

    // --- T-CRα-3: decision — context / obsolete / max ---
    {
        constexpr RetryBackoffPolicy p = convo::kDefaultWarmupRetryBackoff;
        // context 非該当（IR 未 loading）→ NoRetry（WarmupFailed は無条件 retry ではない）
        CHECK(convo::warmupRetryDecision(1, 3, false, false, RetryDisposition::RetryImmediate, p).action
                  == Action::NoRetry, "T-CRα-3: !contextRetryable → NoRetry");
        // obsolete → NoRetry（schedule しない）
        CHECK(convo::warmupRetryDecision(1, 3, true, true, RetryDisposition::RetryImmediate, p).action
                  == Action::NoRetry, "T-CRα-3: obsolete → NoRetry");
        // NoRetry disposition → NoRetry
        CHECK(convo::warmupRetryDecision(1, 3, true, false, RetryDisposition::NoRetry, p).action
                  == Action::NoRetry, "T-CRα-4: NoRetry disposition → NoRetry");
        // attempt 4（4 回目の failure）→ Exhausted（max=3 超過）
        CHECK(convo::warmupRetryDecision(4, 3, true, false, RetryDisposition::RetryImmediate, p).action
                  == Action::Exhausted, "T-CRα-3: attempt 4 > max → Exhausted");
        CHECK(convo::warmupRetryDecision(5, 3, true, false, RetryDisposition::RetryBackoff, p).action
                  == Action::Exhausted, "T-CRα-3: attempt 5 → Exhausted");
    }

    // --- T-CRα-4: disposition → delay mapping ---
    {
        constexpr RetryBackoffPolicy p = convo::kDefaultWarmupRetryBackoff;
        // RetryImmediate → delay 0（latency-sensitive — bounded のみ適用）
        const auto imm1 = convo::warmupRetryDecision(1, 3, true, false, RetryDisposition::RetryImmediate, p);
        CHECK(imm1.action == Action::Schedule && imm1.delayMs == 0,
              "T-CRα-4: RetryImmediate attempt 1 → Schedule delay 0");
        const auto imm3 = convo::warmupRetryDecision(3, 3, true, false, RetryDisposition::RetryImmediate, p);
        CHECK(imm3.action == Action::Schedule && imm3.delayMs == 0,
              "T-CRα-4: RetryImmediate attempt 3 → Schedule delay 0");
        // RetryBackoff → policy 表（attempt 1 → 10）
        const auto bk1 = convo::warmupRetryDecision(1, 3, true, false, RetryDisposition::RetryBackoff, p);
        CHECK(bk1.action == Action::Schedule && bk1.delayMs == 10,
              "T-CRα-4: RetryBackoff attempt 1 → Schedule delay 10");
        const auto bk2 = convo::warmupRetryDecision(2, 3, true, false, RetryDisposition::RetryBackoff, p);
        CHECK(bk2.action == Action::Schedule && bk2.delayMs == 20,
              "T-CRα-4: RetryBackoff attempt 2 → Schedule delay 20");
    }

    if (ok) std::cerr << "[PASS] TestF warmup retry policy (delay table + saturation + decision + mapping)\n";
    return ok;
}

} // namespace

int main()
{
    bool a = runTestA();
    bool b = runTestB();
    bool c = runTestC();
    bool d = runTestD();
    bool e = runTestE();
    bool f = runTestF();

    std::cerr << "[BuildErrorClassification] checks=" << g_pass << " fails=" << g_fail
              << ((a&&b&&c&&d&&e&&f) ? " PASS" : " FAIL") << "\n";
    std::cout << g_pass << " checks, " << g_fail << " failures\n";
    return (a && b && c && d && e && f && g_fail == 0) ? 0 : 1;
}
