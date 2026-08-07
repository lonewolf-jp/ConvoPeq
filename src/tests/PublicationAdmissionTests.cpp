//==============================================================================
// PublicationAdmissionTests.cpp
//
// Admission Policy Unit Test (Phase-2, post Phase-1 atomic refactor).
//
// ■ テスト対象: PublicationAdmission::evaluateDeferred()  (src/audioengine/PublicationAdmission.cpp)
// ■ 性質: Admission = Decisionのみ / Engine直参照なし / DeferredAdmissionSnapshotの5値のみ
//   → ISR Pure-Policy: 入力を入れれば出力が決まる（副作用なし、スレッド不問）。
//
// ■ 判定順序 (design-D4 A-4 / D-9 / ADR-C4:41,64):
//     Shutdown → TTL → Generation → Sequence
//   Decision = Ready | Discard の2値。破棄理由は DiscardReason で区別する
//   （StaleDiscard: TTL/Generation/Sequence, ShutdownDiscard: shutdown）。
//
// ■ 注意 (design-V1.1): TTL超過は現在 Expired ではなく StaleDiscard を返す
//   （実装: PublicationAdmission.cpp:76「★work37: Expired を別enum化可能」）。
//   本テストは実装を仕様として固定する（将来 Expired 化する場合は本テストも更新）。
//
// ■ ビルド (CMakeLists.txt へ追加):
//     add_executable(PublicationAdmissionTests src/tests/PublicationAdmissionTests.cpp)
//     target_include_directories(PublicationAdmissionTests PRIVATE
//         ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
//         ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
//     target_compile_features(PublicationAdmissionTests PRIVATE cxx_std_20)
//     add_test(NAME PublicationAdmissionTests COMMAND PublicationAdmissionTests)
//   (admission_.evaluateDeferred は .cpp で定義されるため、リンクには
//    PublicationAdmission.cpp（JUCE/r8brain ツールチェーン）が必要。)
//
//==============================================================================

#include "PublicationAdmission.h"   // PublicationAdmission, DeferredAdmissionResult, DeferredDecision, DiscardReason, DeferredPublishMetadata, DeferredAdmissionSnapshot

#include <cstdint>
#include <iostream>

namespace {

using Decision     = convo::isr::PublicationAdmission::DeferredDecision;
using Reason       = convo::isr::DiscardReason;
using Metadata     = convo::isr::PublicationAdmission::DeferredPublishMetadata;
using Snapshot     = convo::isr::PublicationAdmission::DeferredAdmissionSnapshot;
using Result       = convo::isr::PublicationAdmission::DeferredAdmissionResult;
using SeqId        = convo::isr::PublicationSequenceId;

// ---- helpers ----
inline Metadata md(int generation, SeqId sequence, uint64_t enqueueUs) {
    return Metadata{ .generation = generation, .sequence = sequence, .enqueueTimestampUs = enqueueUs };
}
inline Snapshot snap(int currentGeneration, SeqId lastSequence, bool shutdown, uint64_t nowUs, uint64_t ttlUs) {
    return Snapshot{ .currentGeneration = currentGeneration, .lastSequence = lastSequence,
                     .shutdown = shutdown, .nowUs = nowUs, .ttlUs = ttlUs };
}

#define CHECK_EQ(actual, expected, label) \
    if (!((actual) == (expected))) { \
        std::cerr << "FAIL: " << (label) << " (line " << __LINE__ << ")\n"; return false; }

// 1) Shutdown — 終了中は一切 publish しない（TTL/Generation/Sequence より優先）。
[[nodiscard]] bool caseShutdown() {
    const auto r = convo::isr::PublicationAdmission{}.evaluateDeferred(
        md(1, SeqId{5}, 1000),
        snap(1, SeqId{5}, /*shutdown*/true, 999'999'999u, 30'000'000u));
    CHECK_EQ(r.decision,        Decision::Discard,   "shutdown: decision");
    CHECK_EQ(r.discardReason,   Reason::ShutdownDiscard, "shutdown: reason");
    return true;
}

// 2) Shutdown + TTL-overdue が同時 → Shutdown が優先される（順序保証）。
[[nodiscard]] bool caseShutdownOverTtl() {
    const auto r = convo::isr::PublicationAdmission{}.evaluateDeferred(
        md(1, SeqId{5}, 1'000u),                // enqueueUs 1ms ago
        snap(1, SeqId{5}, true, 31'000'000u, 30'000'000u));  // age 30s > ttl 30s だが shutdown 優先
    CHECK_EQ(r.discardReason, Reason::ShutdownDiscard, "shutdown>TTL: reason (shutdown priority)");
    return true;
}

// 3) TTL over (stale) — ageUs > ttlUs → Discard/StaleDiscard。
[[nodiscard]] bool caseTtlExpired() {
    const auto r = convo::isr::PublicationAdmission{}.evaluateDeferred(
        md(1, SeqId{5}, 1'000u),
        snap(1, SeqId{5}, false, 31'000'000u, 30'000'000u));  // age 30s > ttl
    CHECK_EQ(r.decision,      Decision::Discard,    "ttl: decision");
    CHECK_EQ(r.discardReason, Reason::StaleDiscard, "ttl: reason");
    return true;
}

// 4) Generation mismatch — rebuild 時代が違う → Discard/StaleDiscard。
[[nodiscard]] bool caseStaleGeneration() {
    const auto r = convo::isr::PublicationAdmission{}.evaluateDeferred(
        md(7, SeqId{5}, 1'000u),                       // enqueued in gen 7
        snap(9, SeqId{5}, false, 2'000u, 30'000'000u)); // current gen 9
    CHECK_EQ(r.discardReason, Reason::StaleDiscard, "generation: reason");
    return true;
}

// 5) Sequence stale — lastSequence より古い → Discard/StaleDiscard (history no-rewind)。
[[nodiscard]] bool caseStaleSequence() {
    const auto r = convo::isr::PublicationAdmission{}.evaluateDeferred(
        md(1, SeqId{3}, 1'000u),                       // enqueued at seq 3
        snap(1, SeqId{8}, false, 2'000u, 30'000'000u)); // last committed seq 8
    CHECK_EQ(r.discardReason, Reason::StaleDiscard, "sequence: reason");
    return true;
}

// 6) 全条件正常 → Ready/None。
[[nodiscard]] bool caseAllNormal() {
    const auto r = convo::isr::PublicationAdmission{}.evaluateDeferred(
        md(1, SeqId{5}, 1'000u),
        snap(1, SeqId{5}, false, 1'500u, 30'000'000u));  // gen ok, seq not older, age 0.5s < ttl
    CHECK_EQ(r.decision,      Decision::Ready,    "normal: decision");
    CHECK_EQ(r.discardReason, Reason::None,     "normal: reason");
    return true;
}

// 7) boundary: ageUs == ttlUs（not > ttlUs）→ Ready。
[[nodiscard]] bool caseTtlBoundaryExact() {
    const auto r = convo::isr::PublicationAdmission{}.evaluateDeferred(
        md(1, SeqId{5}, 1'000u),
        snap(1, SeqId{5}, false, 30'000'001u, 30'000'000u)); // age exactly == ttl
    CHECK_EQ(r.decision, Decision::Ready, "ttl boundary: ==ttl → Ready");
    return true;
}

// 8) boundary: sequence == lastSequence → Ready（history no-rewind は < なので == は許容）。
[[nodiscard]] bool caseSequenceBoundaryEqual() {
    const auto r = convo::isr::PublicationAdmission{}.evaluateDeferred(
        md(1, SeqId{8}, 1'000u),
        snap(1, SeqId{8}, false, 1'500u, 30'000'000u));
    CHECK_EQ(r.decision, Decision::Ready, "seq boundary: ==lastSequence → Ready");
    return true;
}

// 9) Shutdown は TTL/Generation/Sequence の全条件でも優先される（順序保証の総合検証）。
//    shutdown=true かつ gen mismatch かつ seq stale かつ ttl over の場合でも
//    → Discard / ShutdownDiscard（他の理由は問わない）。
[[nodiscard]] bool caseShutdownOverridesAllStale() {
    const auto r = convo::isr::PublicationAdmission{}.evaluateDeferred(
        md(7, SeqId{3}, 1'000u),                          // gen mismatch + seq stale
        snap(9, SeqId{8}, true, 31'000'000u, 30'000'000u)); // shutdown + ttl over
    CHECK_EQ(r.decision,      Decision::Discard,       "shutdown-over-all: decision");
    CHECK_EQ(r.discardReason,  Reason::ShutdownDiscard, "shutdown-over-all: reason (shutdown priority over TTL/Gen/Seq)");
    return true;
}

} // namespace

int main() {
    int failures = 0;
    auto run = [&](const char* name, bool (*fn)()) {
        if (fn()) { std::cout << "PASS " << name << "\n"; }
        else { std::cerr << "FAIL " << name << "\n"; ++failures; }
    };
    run("shutdown",                caseShutdown);
    run("shutdown_over_ttl",       caseShutdownOverTtl);
    run("ttl_expired",             caseTtlExpired);
    run("stale_generation",        caseStaleGeneration);
    run("stale_sequence",          caseStaleSequence);
    run("all_normal",              caseAllNormal);
    run("ttl_boundary_exact",      caseTtlBoundaryExact);
    run("sequence_boundary_equal", caseSequenceBoundaryEqual);
    run("shutdown_overrides_all_stale", caseShutdownOverridesAllStale);

    std::cout << (failures ? "FAILED" : "ALL PASSED") << " (" << (9 - failures) << "/9)\n";
    return failures ? 1 : 0;
}
