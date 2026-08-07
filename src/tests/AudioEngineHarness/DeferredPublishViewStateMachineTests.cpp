// DeferredPublishViewStateMachineTests.cpp
// design-D4 不変条件 8（状態遷移表・design-D4:101-127）の自動検証。
// AudioEngineHarness 上の Integration Test（第13回レビュー反映・design-D4 修正）。
//
// ★ 位置づけ（standalone ではなく AudioEngineHarness 統合）:
//   DeferredPublishView は RuntimePublicationOrchestrator&（→ AudioEngine&）を要求する
//   Authority チェーン（View → finishView → Orchestrator → AudioEngine）の一部であり、
//   単独オブジェクトとしてテストできない。また peek/consume/discard/finishView には
//   jassert(thread == rebuildThreadId()) のスレッドガードがある。したがって:
//     - RebuildThread 専有の遷移（peek→Valid / consume→Consumed+Released /
//       discard→Discarded+Released）は実経路（requestRebuild → CoordinatorLoop →
//       RebuildThread.processDeferredAdmission）を駆動する
//       DeferredFlowIntegrationTests.cpp が実行時検証する（本ファイルは重複しない）。
//     - 本ファイルは View の型レベル契約と、テストスレッドから到達可能な状態機械
//       （default ctor → MovedFrom / move 系 / metadata const& / fail-fast 行の
//       コード検査マッピング）を集中検証する。
//   ★ Testing Principle（ADR-C4 追記・第13回反映）:
//     RebuildThread ownership を強制するオブジェクトは AudioEngineHarness 上で検証する。
//     Standalone main() 単体テスト（add_executable/add_test）は純粋 Policy
//     （PublicationAdmissionTests = evaluateDeferred）のみに予約する。
//
// ■ 状態遷移表（design-D4:103-119）の検証マッピング:
//   | 行 | 遷移 | 検証方法 |
//   |----|------|---------|
//   | Released→Valid | peekDeferred | DeferredFlow（実経路・hasDeferred 反転観測） |
//   | Valid→Consumed+Released | consume | DeferredFlow（実経路・slot drain 観測） |
//   | Valid→Discarded+Released | discard | DeferredFlow（実経路・slot drain 観測） |
//   | Valid→Valid（metadata） | metadata const& | 本ファイル（直接構築・参照同一性） |
//   | Valid→MovedFrom（move ctor） | move ctor | 本ファイル（static_assert + 実行時） |
//   | Valid→（move 代入） | move assign | 本ファイル（static_assert + 実行時） |
//   | Valid→dtor | fail-fast | コード検査（dtor の jassertfalse・本ヘッダで明示） |
//   | Consumed/Discarded/MovedFrom→API | 契約違反 | コード検査（metadata/consume/discard の jassert） |
//   | MovedFrom→dtor | OK | 本ファイル（default/move 後の MovedFrom 破棄が通ること） |
//
// ■ 制約の明文化（design-D4 §111 / ADR-C4:115）:
//   Valid 状態の View は fail-fast デストラクタ（jassert(state_==Valid)）のため、
//   テストスレッドからは破棄できない（consume/discard は RebuildThread 専有）。
//   よって「Valid 行の観測」は heap 上にリークして行う（意図的・tiny 単一オブジェクト・
//   プロセス終了時に OS 回収）。これは fail-fast 対象のテストで標準的な手法であり、
//   リーク検出器（ASan/LSan）対象の通常ビルドでは JUCE jassert が Debug 限定のため
//   リークもプロセス生存期間に限られる。

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <type_traits>

#include "AudioEngineHarness.h"
#include "DeferredPublicationTestAccess.h"
#include "audioengine/RuntimePublicationOrchestrator.h"
#include "audioengine/RuntimePublicationState.h"

namespace {

using convo::isr::DeferredPublishView;
using convo::isr::DeferredPublishSlot;
using convo::isr::DiscardReason;
using ViewState = DeferredPublishView::State;

#define CHECK(cond, label) \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", (label), __LINE__); \
        return false; \
    }

// ── 型レベル契約: copy 禁止 / move-only（遷移表「Valid → move ctor / move 代入」の
//    静的根拠。Consumed/Discarded/MovedFrom 後の copy は型として不可能）。 ──
static_assert(!std::is_copy_constructible_v<DeferredPublishView>,
              "DeferredPublishView must be move-only (design-D4 §89 / ADR-C4)");
static_assert(!std::is_copy_assignable_v<DeferredPublishView>,
              "DeferredPublishView must be move-only (design-D4 §89 / ADR-C4)");
static_assert(std::is_move_constructible_v<DeferredPublishView>,
              "DeferredPublishView must be move-constructible");
static_assert(std::is_move_assignable_v<DeferredPublishView>,
              "DeferredPublishView must be move-assignable");

// ── 1. default ctor → MovedFrom（遷移表「MovedFrom」行・破棄は OK） ──
bool testDefaultViewIsMovedFrom()
{
    DeferredPublishView v;
    CHECK(v.state() == ViewState::MovedFrom, "default ctor: state == MovedFrom");
    CHECK(!v.isValid(), "default ctor: isValid() == false");
    // MovedFrom は slot_ == nullptr のため破棄は fail-fast を踏まない（表「MovedFrom→dtor OK」）。
    return true;
}

// ── 2. MovedFrom ↔ move 系: 自己代入ガード / move ctor / move 代入 ──
bool testMovedFromMoveSemantics()
{
    // 自己代入は this != &other で no-op（Orchestrator.h:66）。
    DeferredPublishView a;
    a = std::move(a);
    CHECK(a.state() == ViewState::MovedFrom, "self move-assign: no-op (still MovedFrom)");
    CHECK(!a.isValid(), "self move-assign: still invalid");

    // move ctor: source は MovedFrom のまま、dest も MovedFrom（slot なし転送）。
    DeferredPublishView b;
    DeferredPublishView c(std::move(b));
    CHECK(c.state() == ViewState::MovedFrom, "move ctor (default→default): dest MovedFrom");
    CHECK(!c.isValid(), "move ctor (default→default): dest invalid");
    CHECK(b.state() == ViewState::MovedFrom, "move ctor: source remains MovedFrom");

    // move 代入: 同様に MovedFrom 状態を転送。
    DeferredPublishView d;
    DeferredPublishView e;
    e = std::move(d);
    CHECK(e.state() == ViewState::MovedFrom, "move assign (default→default): dest MovedFrom");
    CHECK(!e.isValid(), "move assign (default→default): dest invalid");
    return true;
}

// ── 3. Valid 行の観測（metadata const& 同一性 / state / isValid） ──
//    Valid View は fail-fast デストラクタのためテストスレッドから破棄不可 →
//    heap リークで観測（制約の明文化参照）。ローカル slot を参照先に使うのは、
//    View の state_ / metadata() アクセサの検証が目的であり、consume/discard/
//    finishView（RebuildThread 専有）は呼ばないため安全。
bool testValidViewObservation()
{
    // 実 AudioEngine + 実 Orchestrator を要求（Authority チェーンの実在証明）。
    // 本テストは View の state_ / metadata() アクセサのみを観測し、consume/discard/
    // finishView（RebuildThread 専有）は呼ばないため、スレッド起動（start）は不要。
    AudioEngineHarness h;
    auto& orch = DeferredPublicationTestAccess::orchestrator(h.engine());
    (void)orch;

    // heap リーク（意図的）: Valid 状態の View は test thread から破棄できない。
    auto* slot = new DeferredPublishSlot();
    slot->metadata.generation = 7;
    slot->metadata.sequence = convo::isr::PublicationSequenceId{42};
    slot->metadata.enqueueTimestampUs = 1234;

    auto* view = new DeferredPublishView(orch, *slot);
    CHECK(view->state() == ViewState::Valid, "(owner,slot) ctor: state == Valid");
    CHECK(view->isValid(), "(owner,slot) ctor: isValid() == true");

    // metadata() は slot の immutable snapshot への const 参照（design-D4 §89-96）。
    CHECK(&view->metadata() == &slot->metadata, "metadata(): const& identity to slot.metadata");
    CHECK(view->metadata().generation == 7, "metadata(): generation preserved");
    CHECK(view->metadata().sequence == convo::isr::PublicationSequenceId{42}, "metadata(): sequence preserved");
    CHECK(view->metadata().enqueueTimestampUs == 1234, "metadata(): enqueueTimestampUs preserved");

    // move ctor: source → MovedFrom / dest → Valid（遷移表「Valid→move ctor」行）。
    auto* moved = new DeferredPublishView(std::move(*view));
    CHECK(view->state() == ViewState::MovedFrom, "move ctor: source -> MovedFrom");
    CHECK(!view->isValid(), "move ctor: source !isValid()");
    CHECK(moved->state() == ViewState::Valid, "move ctor: dest -> Valid");
    CHECK(moved->isValid(), "move ctor: dest isValid()");
    CHECK(&moved->metadata() == &slot->metadata, "move ctor: dest metadata identity preserved");

    // move 代入（dest が MovedFrom のとき）: dest → Valid / source → MovedFrom。
    //   ※ dest が Valid のままの move 代入は契約違反（Orchestrator.h:68 jassertfalse）=
    //   コード検査で担保（本テストは実行しない）。
    DeferredPublishView sink;   // default → MovedFrom
    sink = std::move(*moved);
    CHECK(moved->state() == ViewState::MovedFrom, "move assign: source -> MovedFrom");
    CHECK(sink.state() == ViewState::Valid, "move assign: dest -> Valid");
    CHECK(sink.isValid(), "move assign: dest isValid()");
    CHECK(&sink.metadata() == &slot->metadata, "move assign: dest metadata identity preserved");

    // ※ sink / moved / view / slot は意図的にリーク（制約の明文化参照）。
    return true;
}

} // namespace

// main 側（PublishPipelineIntegrationTests.cpp）から呼ばれるエントリ
int runDeferredPublishViewStateMachineTests()
{
    if (!testDefaultViewIsMovedFrom())
    {
        std::fprintf(stderr, "FAIL: testDefaultViewIsMovedFrom\n");
        return 1;
    }
    if (!testMovedFromMoveSemantics())
    {
        std::fprintf(stderr, "FAIL: testMovedFromMoveSemantics\n");
        return 1;
    }
    if (!testValidViewObservation())
    {
        std::fprintf(stderr, "FAIL: testValidViewObservation\n");
        return 1;
    }
    std::printf("DeferredPublishViewStateMachineTests: PASS\n");
    return 0;
}
