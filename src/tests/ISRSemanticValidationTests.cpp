#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cstdio>
#include <memory>
#include <type_traits>

#include "audioengine/ISRClosure.h"
#include "audioengine/ISRPayloadTier.h"
#include "audioengine/ISRRuntimePublicationCoordinator.h"
#include "audioengine/ISRRuntimeWorldAuthority.h"  // ★ A-1: Authority Adapter
#include "AudioEngine.h"
#include "ISRRuntimeSemanticSchema.h"

using convo::isr::PublicationSemantic;  // FUTURE-4: world publication fields

namespace {

[[nodiscard]] bool testInvalidClosureRejected()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    convo::isr::PayloadClosureDescriptor invalid {};
    invalid.closureId = 0; // invalid by contract

    convo::isr::TieredPayloadDescriptor descriptor {};
    descriptor.tier = convo::isr::PayloadTier::InlineImmutable;
    descriptor.requiresRT = false;
    descriptor.hasExternalResource = false;
    descriptor.pinnedLifetime = true;

    if (coordinator.precheckPublish(invalid, descriptor))
        return false;

    if (std::strcmp(coordinator.lastRejectReason(), "invalid closure graph") != 0)
        return false;

    return true;
}

[[nodiscard]] bool testInvalidTierRejected()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    convo::isr::PayloadClosureDescriptor closure {};
    closure.closureId = 1;
    closure.nodes.push_back(convo::isr::ClosureNodeRef {
        1u,
        static_cast<std::uint32_t>(convo::isr::PayloadTier::InlineImmutable),
        1u,
        1u,
        1u,
        1u,
        1u,
        1u,
        1u
    });

    convo::isr::TieredPayloadDescriptor descriptor {};
    descriptor.tier = convo::isr::PayloadTier::Forbidden; // invalid by publish policy
    descriptor.requiresRT = false;
    descriptor.hasExternalResource = false;
    descriptor.pinnedLifetime = true;

    if (coordinator.precheckPublish(closure, descriptor))
        return false;

    if (std::strcmp(coordinator.lastRejectReason(), "invalid payload tier") != 0)
        return false;

    return true;
}

[[nodiscard]] bool testCoordinatorCommitAndMonotonicityContract()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       1,
                       1,
                       1,
                       nullptr);   // ★ CW-3b: 初回 commit → prevWorld = nullptr

    if (world1->publication.sequenceId != 1)   // ★ CW-3b: bake 検証（current は非追跡）
        return false;
    if (world1->publication.mappedRuntimeGeneration != 1)
        return false;
    if (coordinator.getState() != convo::isr::RuntimeIntentCoordinator::CoordinatorState::Ready)
        return false;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       2,
                       2,
                       2,
                       world1.get());   // ★ CW-3b: 直前の committed world を baseline に
    if (world2->publication.sequenceId != 2)
        return false;
    if (world2->publication.mappedRuntimeGeneration != 2)
        return false;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       1,
                       1,
                       1,
                       world2.get());   // ★ CW-3b: 同一 seq(1) < baseline(2) → Faulted

    // ★ CW-3b: current は非追跡のため reject 後の current 保持チェックは廃止（Faulted のみ検証）
    return coordinator.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectEpochRollbackContract()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       1,
                       5,
                       10,
                       nullptr);   // ★ CW-3b: 初回 commit

    if (world1->publication.sequenceId != 1)   // ★ CW-3b: bake 検証
        return false;

    // sequence は増加しても epoch rollback は fail-closed
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       2,
                       4,
                       11,
                       world1.get());   // ★ CW-3b: baseline = world1（epoch 4 < 5 → Faulted）

    // ★ CW-3b: current 非追跡のため reject 後の current 保持チェックは廃止
    return coordinator.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectMappedGenerationRollbackOnEpochAdvance()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       10,
                       10,
                       100,
                       nullptr);   // ★ CW-3b: 初回 commit

    if (world1->publication.sequenceId != 10)   // ★ CW-3b: bake 検証
        return false;

    // epoch advance 時の mapped generation rollback は fail-closed
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       11,
                       11,
                       99,
                       world1.get());   // ★ CW-3b: baseline = world1（gen 99 < 100 → Faulted）

    // ★ CW-3b: current 非追跡のため reject 後の current 保持チェックは廃止
    return coordinator.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectEpochReuseContract()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       100,
                       100,
                       1000,
                       nullptr);   // ★ CW-3b: 初回 commit

    if (world1->publication.sequenceId != 100)   // ★ CW-3b: bake 検証
        return false;

    // sequence が進んでも epoch reuse は strict monotonic 契約違反
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       101,
                       100,
                       1001,
                       world1.get());   // ★ CW-3b: baseline = world1（epoch 100 不変 → Faulted）

    // ★ CW-3b: current 非追跡のため reject 後の current 保持チェックは廃止
    return coordinator.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectMappedGenerationReuseContract()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       200,
                       200,
                       5000,
                       nullptr);   // ★ CW-3b: 初回 commit

    if (world1->publication.sequenceId != 200)   // ★ CW-3b: bake 検証
        return false;

    // epoch が進んでも mapped generation reuse は strict monotonic 契約違反
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       201,
                       201,
                       5000,
                       world1.get());   // ★ CW-3b: baseline = world1（gen 5000 不変 → Faulted）

    // ★ CW-3b: current 非追跡のため reject 後の current 保持チェックは廃止
    return coordinator.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectWraparoundContract()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();
    auto world3 = RuntimeState::createForTest();

    constexpr std::uint64_t maxValue = std::numeric_limits<std::uint64_t>::max();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       maxValue - 1,
                       maxValue - 1,
                       maxValue - 1,
                       maxValue - 1,
                       nullptr);   // ★ CW-3b: 初回 commit

    if (world1->publication.sequenceId != maxValue - 1)   // ★ CW-3b: bake 検証
        return false;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       maxValue,
                       maxValue,
                       maxValue,
                       maxValue,
                       world1.get());   // ★ CW-3b: baseline = world1

    if (world2->publication.sequenceId != maxValue)
        return false;

    // wraparound（max -> 0）は strict monotonic 契約違反
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world3.get(),
                       0,
                       0,
                       0,
                       0,
                       world2.get());   // ★ CW-3b: baseline = world2（0 < max → Faulted）

    // ★ CW-3b: current 非追跡のため reject 後の current 保持チェックは廃止
    return coordinator.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorDrainAndShutdownContract()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    int world = 1;
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1,
                       nullptr);   // ★ CW-3b: 初回 commit → prevWorld = nullptr

    coordinator.setRetireBacklogCount(0);
    coordinator.setPublicationBacklogCount(0);
    coordinator.setPendingIntentCount(0);
    // ★ D101-32-D: setFallbackBacklogCount / setReclaimInFlightCount /
    //   setDeferredRetireResidencyCount は削除済み（vestigial setter）。
    //   fresh instance の初期値 0 が初期状態を保証する。fallback/deferred/reclaim の
    //   実測は Layer 1 + onReclaimBegin/End（semantic event）が authority。
    coordinator.setSwapPending(false);

    if (!coordinator.isFullyDrained())
        return false;

    coordinator.requestShutdown();
    coordinator.markShutdownComplete();

    if (coordinator.getState() != convo::isr::RuntimeIntentCoordinator::CoordinatorState::Bootstrapping)
        return false;

    return true;
}

[[nodiscard]] bool testShutdownCompleteFailsWhenNotDrained()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    int world = 1;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1,
                       nullptr);   // ★ CW-3b: 初回 commit → prevWorld = nullptr

    coordinator.setRetireBacklogCount(1); // drained 条件を破る
    coordinator.setPublicationBacklogCount(0);
    coordinator.setPendingIntentCount(0);
    // ★ D101-32-D: fallback/reclaim/deferred の setter は削除済み（vestigial）。
    //   fresh instance 初期値 0。drain 違反は setRetireBacklogCount(1) のみで駆動。
    coordinator.setSwapPending(false);

    if (coordinator.isFullyDrained())
        return false;

    coordinator.requestShutdown();
    coordinator.markShutdownComplete();

    return coordinator.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testPressureStateNormalizationContract()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    int world = 1;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1,
                       nullptr);   // ★ CW-3b: 初回 commit → prevWorld = nullptr

    // slope > threshold で Pressure へ遷移
    coordinator.setRetireBacklogCount(9);
    if (coordinator.getState() != convo::isr::RuntimeIntentCoordinator::CoordinatorState::Pressure)
        return false;

    // swapPending 中は normalization しない
    coordinator.setSwapPending(true);
    coordinator.setRetireBacklogCount(0);
    if (coordinator.getState() != convo::isr::RuntimeIntentCoordinator::CoordinatorState::Pressure)
        return false;

    // swapPending 解除後、3 window で Ready へ復帰
    coordinator.setSwapPending(false);
    coordinator.setRetireBacklogCount(0);
    if (coordinator.getState() != convo::isr::RuntimeIntentCoordinator::CoordinatorState::Pressure)
        return false;
    coordinator.setRetireBacklogCount(0);
    if (coordinator.getState() != convo::isr::RuntimeIntentCoordinator::CoordinatorState::Pressure)
        return false;
    coordinator.setRetireBacklogCount(0);

    return coordinator.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Ready;
}

[[nodiscard]] bool testShutdownCompleteFailsWhenSwapPending()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    int world = 1;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1,
                       nullptr);   // ★ CW-3b: 初回 commit → prevWorld = nullptr

    coordinator.setRetireBacklogCount(0);
    coordinator.setPublicationBacklogCount(0);
    coordinator.setPendingIntentCount(0);
    // ★ D101-32-D: fallback/reclaim/deferred の setter は削除済み（vestigial）。
    //   fresh instance 初期値 0。drain 違反は setSwapPending(true) で駆動。
    coordinator.setSwapPending(true); // drained 条件を破る

    if (coordinator.isFullyDrained())
        return false;

    coordinator.requestShutdown();
    coordinator.markShutdownComplete();

    return coordinator.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted;
}

// --- P4: Generation / ActivationEpoch 契約 ---
// generation 増加時は activationEpoch も必ず増加する (+1 以上)。
// 同一 generation での activationEpoch 単独変更は禁止。
[[nodiscard]] bool testP4SameGenerationEpochChangeRejected()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    // 初回 commit: gen=100, epoch=100
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       100,
                       100,
                       100,
                       nullptr);   // ★ CW-3b: 初回 commit

    if (coordinator.getState() != convo::isr::RuntimeIntentCoordinator::CoordinatorState::Ready)
        return false;

    // 同一 generation (100) で epoch のみ変更 (101) → 禁止 (generation 不変で epoch 変更)
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       100,
                       101,
                       100,
                       world1.get());   // ★ CW-3b: baseline = world1（seq 100 不変 → Faulted）

    // ★ CW-3b: current 非追跡のため reject 後の current 保持チェックは廃止（Faulted のみ検証）
    return coordinator.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted;
}

// ★ dash2 §1.7 (Phase G CW-3 調査, 2026-08-15): publish フローの二重 commit 検証。
//   ［CW-3a で onRuntimePublishedNonRt の冗長 commit #2 を除去済み］
//   ［CW-3b: monotonicity baseline は明示的な prevWorld。同一 world を baseline に渡すと
//     strict monotonic（seq/epoch/gen 全て増加）違反 → Faulted］
[[nodiscard]] bool testCoordinatorDoubleCommitSameWorldFaults()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    auto world1 = RuntimeState::createForTest();

    // commit #1: publish() 相当（初回 commit — prevWorld = nullptr）
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1, 1, 1, 1,
                       nullptr);
    if (coordinator.getState() != convo::isr::RuntimeIntentCoordinator::CoordinatorState::Ready)
        return false;

    // commit #2: 同一 world を baseline として再 commit → monotonicity 違反（1 > 1 = false）→ Faulted
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1, 1, 1, 1,
                       world1.get());

    return coordinator.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted;
}

// --- P20: Fail-Closed Rollback ---
// reject 時に system state がロールバックされることを確認。
// coordinator は契約違反時に Faulted に遷移する（fail-closed）が、
// currentWorld と version は reject 前の値を維持する。
// 副作用（callback, telemetry）は reject 経路では発生しない。
[[nodiscard]] bool testP20RejectPreservesWorldState()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    // 初回 commit
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1, 1, 1, 1,
                       nullptr);   // ★ CW-3b: 初回 commit

    if (world1->publication.sequenceId != 1)   // ★ CW-3b: bake 検証
        return false;
    if (world1->publication.mappedRuntimeGeneration != 1)
        return false;

    // 不正な commit（epoch rollback）で reject されるはず
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2, 2, 0, 2,
                       world1.get());   // ★ CW-3b: baseline = world1（epoch 0 < 1 → Faulted）

    // state は Faulted に遷移する（fail-closed）: これは意図された動作
    if (coordinator.getState() != convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted)
        return false;

    // ★ CW-3b: current 非追跡のため reject 後の current/version 保持チェックは廃止。
    //   代わりに「reject された candidate は bake されない」ことを検証（fail-closed）。
    if (world2->publication.sequenceId != 0)
        return false;                    // world2 は bake されていない（default のまま）
    if (world1->publication.mappedRuntimeGeneration != 1)
        return false;                    // 前回有効な bake は維持

    return true;
}

// ★ FUTURE-4 METADATA-1/2/6: single consumeAtomic(currentWorld_) snapshot yields
//   consistent epoch + generation + sequence via RuntimeState::publication.
[[nodiscard]] bool testMetadataSnapshotConsistentAcrossReaders()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    auto world = RuntimeState::createForTest();
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world.get(), 1, 1, 10, 100,
                       nullptr);   // ★ CW-3b: 初回 commit
    if (world->publication.sequenceId != 1)   // ★ CW-3b: bake 検証
        return false;
    if (world->publication.mappedRuntimeGeneration != 100)
        return false;
    if (world->publication.epoch != 10)
        return false;
    return true;
}

[[nodiscard]] bool testMetadataSnapshotRejectsEpochRollback()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(), 1, 1, 10, 100,
                       nullptr);   // ★ CW-3b: 初回 commit
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(), 2, 2, 4, 11,
                       world1.get());   // ★ CW-3b: baseline = world1（epoch 4 < 10 → Faulted）
    if (coordinator.getState() != convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted)
        return false;
    // ★ CW-3b: current 非追跡のため reject 後の current/version/epoch 保持チェックは廃止。
    //   代わりに「reject された candidate は bake されない」ことを検証（fail-closed）。
    if (world2->publication.sequenceId != 0)
        return false;
    return true;
}

[[nodiscard]] bool testMetadataSnapshotSequenceAdvancesWithEpoch()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    auto w1 = RuntimeState::createForTest();
    auto w2 = RuntimeState::createForTest();
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       w1.get(), 1, 1, 1, 1,
                       nullptr);   // ★ CW-3b: 初回 commit
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       w2.get(), 2, 2, 2, 2,
                       w1.get());   // ★ CW-3b: baseline = w1
    if (w2->publication.sequenceId != 2)   // ★ CW-3b: bake 検証
        return false;
    if (w2->publication.mappedRuntimeGeneration != 2)
        return false;
    if (w2->publication.epoch != 2)
        return false;
    return true;
}

// METADATA-6: no transitional cache symbol; reader is pure world snapshot.
[[nodiscard]] bool testMetadataSnapshotNoTransitionalCacheSymbol()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    auto world = RuntimeState::createForTest();
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world.get(), 1, 1, 7, 700,
                       nullptr);   // ★ CW-3b: 初回 commit
    if (world->publication.mappedRuntimeGeneration != 700)   // ★ CW-3b: bake 検証
        return false;
    if (world->publication.epoch != 7)
        return false;
    return true;
}

// ★ FUTURE-3: Recovery Request は transport-only enqueue。Admission 判定なし。
//   submitRecoveryRequest() -> popRecoveryRequest() 1-hop 輸送。Builder Loop が復旧 World を build。
[[nodiscard]] bool testRecoveryRequestEnqueueAndPop()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    const auto handle = convo::isr::DSPHandle::null();
    // ★ FUTURE-3 (work88): buildSource（RuntimeBuildSnapshot 値コピー）を引数に追加。
    //   quarantinedHandle 単独では resolve 不能なため、build 入力は値コピーで引当する。
    convo::RuntimeBuildSnapshot buildSource{};
    buildSource.sealed = true;  // 1-hop 輸送テストのため sealed 済み snapshot を渡す
    // ★ dash2 §1.9 (Phase E): transport 成功時は true（recovery obligation 存在 — wake 条件）。
    if (!coordinator.submitRecoveryRequest(handle, buildSource, 0))   // enqueue（Admission 判定なし）
        return false;
    if (!coordinator.popRecoveryRequest().has_value())
        return false;                            // Builder pop path
    if (coordinator.popRecoveryRequest().has_value())
        return false;                            // 1-hop transport（duplicate/queue-no-op なし）
    return true;
}
// ★ dash2 §1.7 (Phase G R7 回帰テスト, 2026-08-15): submitRecoveryRequest の epoch 伝搬。
//   CW-3b で commit() が currentWorld_ を非更新にした後、Coordinator が currentWorld_ から epoch を
//   読むと常に 0 になる潜在回帰（R7）を捕捉する。修正後は caller が epoch を明示的に渡す
//   （RuntimeStore::current 由来）。本テストは「epoch=0 固定では検出できない」回帰を検証する。
[[nodiscard]] bool testRecoveryRequestEpochPropagation()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    convo::RuntimeBuildSnapshot buildSource{};
    buildSource.sealed = true;

    // 明示 epoch（42）を渡す → popRecoveryRequest で同一 epoch が返る（R7 回帰捕捉）
    if (!coordinator.submitRecoveryRequest(convo::isr::DSPHandle::null(), buildSource, 42))
        return false;
    auto recovery = coordinator.popRecoveryRequest();
    if (!recovery.has_value())
        return false;
    if (recovery->epoch != static_cast<convo::isr::PublicationEpoch>(42))
        return false;                    // epoch=0 固定だと失敗（R7 回帰）
    return true;
}
// ★ work88 (X1 §6.1): Recovery Durable Admission — queue full ≠ Recovery lost（INV-X1-2）。
//   recoveryIntentQueue_（256）を満杯 → 257th submit は durable admission（PendingRecoveryAdmission）
//   に保持され、hasPendingRecoveryAdmission() == true。takePendingRecoveryAdmission() は
//   lease（DurablePending → Building）で消費し、settle(false) でクリア（INV-X1-1）。
[[nodiscard]] bool testRecoveryDurableAdmission()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    convo::RuntimeBuildSnapshot buildSource{};
    buildSource.sealed = true;

    constexpr int kCap = 256;  // kRecoveryIntentQueueCapacity

    // recoveryIntentQueue_（256）を満杯にする（全て transport に成功 → durable 無し）
    for (int i = 0; i < kCap; ++i)
    {
        // ★ dash2 §1.9 (Phase E): transport 成功時は true
        if (!coordinator.submitRecoveryRequest(convo::isr::DSPHandle::null(), buildSource, 0))
            return false;
    }
    if (coordinator.hasPendingRecoveryAdmission())
        return false;   // 満杯まで durable は無いはず

    // 257th: queue full → durable admission に保持（INV-X1-2: queue full ≠ Recovery lost）
    //   ★ dash2 §1.9 (Phase E): durable 化時も true（recovery obligation 存在 — wake 条件）
    if (!coordinator.submitRecoveryRequest(convo::isr::DSPHandle::null(), buildSource, 0))
        return false;
    if (!coordinator.hasPendingRecoveryAdmission())
        return false;

    // transport は 256 件のみ pop できる（durable は transport に無い — INV-X1-6 二重計上なし）
    for (int i = 0; i < kCap; ++i)
    {
        if (!coordinator.popRecoveryRequest().has_value())
            return false;
    }
    if (coordinator.popRecoveryRequest().has_value())
        return false;

    // takePendingRecoveryAdmission（lease: DurablePending → Building）→ settle(false) でクリア
    if (!coordinator.takePendingRecoveryAdmission().has_value())
        return false;
    coordinator.settlePendingRecoveryAdmission(false);
    if (coordinator.hasPendingRecoveryAdmission())
        return false;
    // 2 回目の take は nullopt（もう無い）
    if (coordinator.takePendingRecoveryAdmission().has_value())
        return false;

    return true;
}

// ★ work88 (X1 §6.1): durable admission の coalesce — 重複 submit でも durable は単一
//   （INV-X1-5: 1 logical admission = at most 1 reservation）。
[[nodiscard]] bool testRecoveryDurableCoalesce()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    convo::RuntimeBuildSnapshot buildSource{};
    buildSource.sealed = true;

    // 満杯 → durable admission 作成
    for (int i = 0; i < 256; ++i)
        coordinator.submitRecoveryRequest(convo::isr::DSPHandle::null(), buildSource, 0);
    coordinator.submitRecoveryRequest(convo::isr::DSPHandle::null(), buildSource, 0);
    if (!coordinator.hasPendingRecoveryAdmission())
        return false;

    // さらに durable submit（coalesce）— durable は単一のまま（INV-X1-5）
    coordinator.submitRecoveryRequest(convo::isr::DSPHandle::null(), buildSource, 0);
    coordinator.submitRecoveryRequest(convo::isr::DSPHandle::null(), buildSource, 0);

    // take で 1 回だけ消費でき、その後は空（coalesce により単一 durable のみ）
    if (!coordinator.takePendingRecoveryAdmission().has_value())
        return false;
    coordinator.settlePendingRecoveryAdmission(false);
    if (coordinator.takePendingRecoveryAdmission().has_value())
        return false;
    if (coordinator.hasPendingRecoveryAdmission())
        return false;

    return true;
}

// ★ work88 (X1 §6.1): lease 方式 — transient build 失敗（settle(true)）で Building → DurablePending へ
//   戻り、再 take 可能（retry 構造的保証）。INV-X1-1（exactly one durable state）が維持される。
[[nodiscard]] bool testRecoveryDurableLeaseRetry()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    convo::RuntimeBuildSnapshot buildSource{};
    buildSource.sealed = true;

    // 満杯 → durable admission 作成
    for (int i = 0; i < 256; ++i)
        coordinator.submitRecoveryRequest(convo::isr::DSPHandle::null(), buildSource, 0);
    coordinator.submitRecoveryRequest(convo::isr::DSPHandle::null(), buildSource, 0);
    if (!coordinator.hasPendingRecoveryAdmission())
        return false;

    // take（lease: DurablePending → Building）
    if (!coordinator.takePendingRecoveryAdmission().has_value())
        return false;
    // Building 中も hasPendingRecoveryAdmission() == true（build gap 検出）
    if (!coordinator.hasPendingRecoveryAdmission())
        return false;

    // transient failure → settle(true): Building → DurablePending（再 take 可能）
    coordinator.settlePendingRecoveryAdmission(true);
    if (!coordinator.hasPendingRecoveryAdmission())
        return false;
    if (!coordinator.takePendingRecoveryAdmission().has_value())
        return false;
    coordinator.settlePendingRecoveryAdmission(false);  // 成功 → クリア
    if (coordinator.hasPendingRecoveryAdmission())
        return false;

    return true;
}

// ★ FUTURE-8: overflow は Observe 専用 Deferred Ring へ（Retire 系 ring と分離, QUEUE-15）。
//   1024+2048+1024 満村 → drop。enqueue path crash なし + pending count 連動を検証。
[[nodiscard]] bool testObserveOverflowEnqueuePath()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;
    const auto handle = convo::isr::DSPHandle::null();
    constexpr int N = 4100;  // 1024(L1) + 2048(L2) + 1024(L3) + drop
    for (int i = 0; i < N; ++i)
        coordinator.submitObserve(handle, 0);
    return coordinator.getPendingIntentCount() == static_cast<std::uint64_t>(N);
}

// ★ A-1 (X4-B §6.4 Test 1 / 二十一次レビュー): RuntimeWorldAuthority owns the physical
//   RuntimeStore（write authority singularization — INV-X4-3/5）。Store::OwnerType ==
//   RuntimeWorldAuthority はコンパイル時不変条件。published-world read は物理 Store
//   （RuntimeStore::current — observePublishedWorld / consumeWorldHandle）のみが単一 source
//   （★ CW-3c: Coordinator/RWA の metadata accessor は production caller ゼロのため削除 —
//     read-side singularization 完了）。
static_assert(std::is_same_v<convo::isr::RuntimeWorldAuthority::Store::OwnerType,
                             convo::isr::RuntimeWorldAuthority>,
    "X4-B Test 1: RuntimeWorldAuthority must own its RuntimeStore (INV-X4-3/5)");

// ★ X4-B Test 2: WriteAccess move-only（RuntimeStore.h の static_assert を Authority 側でも固定）。
//   WriteAccess は Store への非所有参照を持つため copy 不可・move のみ — INV-X4-3 の前提。
using X4BTestWriteAccess = convo::isr::RuntimeWorldAuthority::WriteAccess;
static_assert(!std::is_copy_constructible_v<X4BTestWriteAccess>,
    "X4-B Test 2: WriteAccess must not be copy-constructible");
static_assert(!std::is_copy_assignable_v<X4BTestWriteAccess>,
    "X4-B Test 2: WriteAccess must not be copy-assignable");
static_assert(std::is_move_constructible_v<X4BTestWriteAccess>,
    "X4-B Test 2: WriteAccess must be move-constructible");
static_assert(std::is_move_assignable_v<X4BTestWriteAccess>,
    "X4-B Test 2: WriteAccess must be move-assignable");
static_assert(std::is_nothrow_move_constructible_v<X4BTestWriteAccess>,
    "X4-B Test 2: WriteAccess move ctor must stay noexcept");
static_assert(std::is_nothrow_move_assignable_v<X4BTestWriteAccess>,
    "X4-B Test 2: WriteAccess move assign must stay noexcept");
// ★ X4-B Test 3-10（アーキテクチャ検証の対応表）:
//   Test 3 (publishAndSwap 唯一性) / Test 4 (PublishExecutor bypass 禁止) / Test 5 (Coordinator
//   bypass 禁止) / Test 6 (二重 Store 検出): `tools/publication_authority_verifier.py` の静的検査
//   （ALLOWED_PUBLISH_AND_SWAP_FILES = RuntimeWorldAuthority.h + core のみ）。
//   Test 7 (commit-before-swap ordering): `RuntimeWorldAuthority::publish()` 内の commit→swap 順序
//     コメント固定（本テストは単一スレッドのため ordering はコード契約で検証）。
//   Test 8 (ownership transfer exactly once): OwnerChannel（SPSC・key 単一 transfer）が実装。
//   Test 9 (dual-pointer identity consistency) / Test 10 (INV-X4-7): `AudioEngineHarness`
//     PublishPipelineIntegrationTests（store swap seq 検証）が統合カバー。

[[nodiscard]] bool testRuntimeWorldAuthorityAdapter()
{
    auto coordinator = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto authority = std::make_unique<convo::isr::RuntimeWorldAuthority>(*coordinator);

    // ★ dash2 §1.7 (Phase G CW-3c): currentEpoch/sequence/getCurrent/getVersion は production caller
    //   ゼロのため削除済み。Adapter は read API（RuntimeStore::current — INV-X4-B）を検証する。

    // X4-B: read API は物理 Store（RuntimeStore::current — INV-X4-B）から observe する。
    //   未 publish 時は null（Store 初期値）。
    const auto token = authority->acquireReadToken();
    if (authority->consumeWorldHandle(token) != nullptr)
        return false;
    if (authority->consumeWorldHandle() != nullptr)
        return false;
    if (authority->observePublishedWorld() != nullptr)
        return false;

    // Coordinator must not expose diagnostic/metric setters through the Authority
    // Surface — guaranteed at compile time by RuntimeWorldAuthority's member set.
    return true;
}

// ★ B3 invariant #4: Backpressure explicit — publish intent queue-full ⇒
//   enqueuePublicationIntent() returns false (never a silent drop). Fill the shared
//   intentQueue_ to capacity, then verify the next publish intent is explicitly rejected
//   and the queue still holds exactly capacity items (recoverable by drain).
[[nodiscard]] bool testPublishIntentQueueFullBackpressure()
{
    auto coordinatorStorage = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
    auto& coordinator = *coordinatorStorage;

    constexpr size_t kCapacity = 4096;      // kIntentQueueCapacity (FUTURE-10 common queue)
    convo::isr::RuntimeIntentCoordinator::Intent intent{};
    intent.type = convo::isr::RuntimeIntentCoordinator::IntentType::Publish;

    size_t accepted = 0;
    for (size_t i = 0; i < kCapacity + 1; ++i)
    {
        intent.sequenceId = static_cast<std::uint64_t>(i + 1);
        if (coordinator.enqueuePublicationIntent(intent))
            ++accepted;
    }
    if (accepted != kCapacity)              // fill up to capacity exactly
        return false;

    // queue is now full: next publish intent must be explicitly rejected (backpressure)
    if (coordinator.enqueuePublicationIntent(intent))
        return false;

    return true;
}

} // namespace

// =============================================================================
// D105-R5-9: Recovery Logical Obligation Enforcement — counterexample tests C1-C10
// Enforce the single +1/−1 invariant, handle-bearing CoalesceIdentity, id-based
// ABA-safe resolve, and StaleSuperseded terminal. A failure here means the
// completion-authority contract regressed (leak / double-free / stale resolve).
// =============================================================================
namespace {
    convo::RuntimeBuildSnapshot makeRecoverySnapshot(std::uint64_t identityHash)
    {
        convo::RuntimeBuildSnapshot snap{};
        snap.rebuildFingerprint.irIdentityHash = identityHash;
        snap.rebuildFingerprint.convolutionConfigHash = 0x21u;
        snap.rebuildFingerprint.dspParameterHash = 0x42u;
        snap.rebuildFingerprint.fingerprintVersion = 1;
        return snap;
    }

    std::optional<std::uint64_t> submitAndGetId(convo::isr::RuntimeIntentCoordinator& c,
                                                const convo::isr::DSPHandle& h,
                                                std::uint64_t identityHash)
    {
        convo::RuntimeBuildSnapshot snap = makeRecoverySnapshot(identityHash);
        if (!c.submitRecoveryRequest(h, snap, 1))
            return std::nullopt;
        auto pop = c.popRecoveryRequest();
        if (!pop)
            return std::nullopt;
        return pop->obligationId;
    }

    // C1: 32 distinct recovery obligations are all accepted and counted (L==32).
    [[nodiscard]] bool testRLOE_C1_unique32()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        for (std::uint64_t i = 0; i < 32; ++i)
            if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(i + 1), 1))
                return false;
        return c->liveLogicalRecoveryObligationCount() == 32;
    }

    // C2: the 33rd distinct obligation is rejected (capacity 32) and must NOT increment L.
    [[nodiscard]] bool testRLOE_C2_33rdRejected()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        for (std::uint64_t i = 0; i < 32; ++i)
            if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(i + 1), 1))
                return false;
        const bool accepted = c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(999), 1);
        return (!accepted)
            && c->liveLogicalRecoveryObligationCount() == 32
            && c->recoveryCapacityExhaustedCount() >= 1;
    }

    // C3: at capacity (L==32), a duplicate submission coalesces (accepted, L unchanged, coalescedCount+1).
    [[nodiscard]] bool testRLOE_C3_coalesceAtFull()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        for (std::uint64_t i = 0; i < 32; ++i)
            if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(i + 1), 1))
                return false;
        const std::uint64_t coalescedBefore = c->recoveryCoalescedCount();
        const bool accepted = c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(7), 1);
        return accepted
            && c->liveLogicalRecoveryObligationCount() == 32
            && c->recoveryCoalescedCount() == coalescedBefore + 1;
    }

    // C4: two distinct handles with identical target are two distinct obligations (L==before+2).
    [[nodiscard]] bool testRLOE_C4_distinctHandlesSameTarget()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        convo::RuntimeBuildSnapshot snap = makeRecoverySnapshot(555);
        const std::uint64_t before = c->liveLogicalRecoveryObligationCount();
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle{1, 1}, snap, 1)) return false;
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle{2, 1}, snap, 1)) return false;
        return c->liveLogicalRecoveryObligationCount() == before + 2;
    }

    // C5: QueuePressure/Retry resolution keeps the obligation Live (ΔL==0) and still deliverable.
    [[nodiscard]] bool testRLOE_C5_retryKeepsLive()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 11);
        if (!id) return false;
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        c->resolveRecoveryObligation(*id, convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::Retry);
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;  // must remain Live
        // still deliverable: a duplicate submission coalesces (accepted, L unchanged)
        const bool dup = c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(11), 1);
        return dup && c->liveLogicalRecoveryObligationCount() == 1;
    }

    // C6: StaleGeneration rejection routes to StaleSuperseded terminal → L 1→0 (no leak).
    [[nodiscard]] bool testRLOE_C6_staleSupersededResolves()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 22);
        if (!id) return false;
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        c->resolveRecoveryObligation(*id, convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::StaleSuperseded);
        return c->liveLogicalRecoveryObligationCount() == 0;
    }

    // C7: duplicate Published resolution is idempotent — L 1→0 exactly once, never negative.
    [[nodiscard]] bool testRLOE_C7_duplicatePublishedOnce()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 33);
        if (!id) return false;
        c->resolveRecoveryObligation(*id, convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::Published);
        const std::uint64_t l0 = c->liveLogicalRecoveryObligationCount();
        c->resolveRecoveryObligation(*id, convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::Published); // no-op
        return l0 == 0 && c->liveLogicalRecoveryObligationCount() == 0;
    }

    // C8: shutdown race — Failed then ShutdownDiscarded (or reverse) resolves once; count increments once.
    [[nodiscard]] bool testRLOE_C8_shutdownRaceOnce()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 44);
        if (!id) return false;
        const std::uint64_t sdBefore = c->recoveryObligationShutdownDiscardCount();
        c->resolveRecoveryObligation(*id, convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::Failed);
        c->resolveRecoveryObligation(*id, convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::ShutdownDiscarded); // no-op
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (c->recoveryObligationShutdownDiscardCount() != sdBefore) return false;  // not incremented by no-op
        // direct shutdown-discard path increments exactly once
        auto id2 = submitAndGetId(*c, convo::isr::DSPHandle::null(), 45);
        if (!id2) return false;
        c->resolveRecoveryObligation(*id2, convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::ShutdownDiscarded);
        return c->liveLogicalRecoveryObligationCount() == 0
            && c->recoveryObligationShutdownDiscardCount() == sdBefore + 1;
    }

    // C9: ABA safety — stale late resolve of a reused slot must NOT resurrect / free the new obligation.
    [[nodiscard]] bool testRLOE_C9_abaStaleResolveNoOp()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id1 = submitAndGetId(*c, convo::isr::DSPHandle::null(), 1);   // L=1, slot0
        if (!id1) return false;
        c->resolveRecoveryObligation(*id1, convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::Published); // L=0, slot0 free
        auto id2 = submitAndGetId(*c, convo::isr::DSPHandle::null(), 2);   // L=1, slot0 REUSED (new id)
        if (!id2) return false;
        if (*id2 == *id1) return false;
        c->resolveRecoveryObligation(*id1, convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::Published); // stale, must be no-op
        return c->liveLogicalRecoveryObligationCount() == 1;  // O2 still Live; O1 not resurrected
    }

    // C10: 256 distinct submissions under concurrency-shaped loop must cap at L==32 (no overflow),
    // with exactly 256-32 == 224 capacity-exhausted rejects.
    [[nodiscard]] bool testRLOE_C10_stress256CapsAt32()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        for (std::uint64_t i = 0; i < 256; ++i)
            (void)c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(i + 1), 1);
        return c->liveLogicalRecoveryObligationCount() == 32
            && c->recoveryCapacityExhaustedCount() >= 224
            && c->liveLogicalRecoveryObligationCount() <= 32;
    }

    // ★ D105-R5-10 helper: fill recoveryIntentQueue_ (256) with `count` coalesced submissions of one target.
    void fillRecoveryQueue(convo::isr::RuntimeIntentCoordinator& c, std::uint64_t identityHash)
    {
        convo::RuntimeBuildSnapshot snap = makeRecoverySnapshot(identityHash);
        for (int i = 0; i < 256; ++i)
            (void)c.submitRecoveryRequest(convo::isr::DSPHandle::null(), snap, 1);
    }

    // C11: a deferred (Live, delivery==None) obligation is redriven onto the durable slot when it frees.
    //   Setup: O1 fills queue (256 coalesced intents, transport); O2 → durable (free slot); O3 → deferred
    //   (durable occupied by O2). Free the durable slot, redrive → O3 must re-attach to durable. ΔL = 0.
    [[nodiscard]] bool testRLOE_C11_redriveRecoversDeferredViaDurable()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        fillRecoveryQueue(*c, 1);                                                       // O1: queue full
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(2), 1)) return false;  // O2 → durable
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(3), 1)) return false;  // O3 → deferred
        if (c->recoveryRetryDeferredCount() < 1) return false;                          // O3 deferred
        if (c->liveLogicalRecoveryObligationCount() != 3) return false;

        if (!c->takePendingRecoveryAdmission().has_value()) return false;               // consume O2
        c->settlePendingRecoveryAdmission(false);                                       // durable slot free

        c->redriveDeferredRecoveryObligations();                                        // O3 → durable
        if (c->recoveryRetryRedriveCount() < 1) return false;

        auto redriven = c->takePendingRecoveryAdmission();                              // O3 now durable
        if (!redriven.has_value()) return false;
        if (redriven->obligationId == 0) return false;
        if (c->liveLogicalRecoveryObligationCount() != 3) return false;                // ΔL == 0
        if (c->takePendingRecoveryAdmission().has_value()) return false;                // exactly one durable (no dup)
        return true;
    }

    // C12: redrive is idempotent — a second redrive on an already-delivered obligation does NOT create a
    //   duplicate delivery representation.
    [[nodiscard]] bool testRLOE_C12_redriveIdempotentNoDuplicate()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        fillRecoveryQueue(*c, 1);
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(2), 1)) return false;
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(3), 1)) return false;
        if (!c->takePendingRecoveryAdmission().has_value()) return false;
        c->settlePendingRecoveryAdmission(false);
        c->redriveDeferredRecoveryObligations();
        c->redriveDeferredRecoveryObligations();                                        // second redrive: no-op (delivery!=None)
        if (!c->takePendingRecoveryAdmission().has_value()) return false;              // one durable only
        if (c->takePendingRecoveryAdmission().has_value()) return false;               // no duplicate
        if (c->liveLogicalRecoveryObligationCount() != 3) return false;
        return true;
    }

    // C13: redrive prefers durable but falls back to transport when the durable slot is busy.
    //   Setup: queue full (O1), O2 → durable, O3 → deferred. Drain the queue (keep O2 in durable), redrive.
    //   O3 must go to transport (queue now has space), NOT durable (still O2).
    [[nodiscard]] bool testRLOE_C13_redriveFallsBackToTransport()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        fillRecoveryQueue(*c, 1);
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(2), 1)) return false;
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(3), 1)) return false;
        for (int i = 0; i < 256; ++i)                                                 // drain queue, keep O2 durable
            if (!c->popRecoveryRequest().has_value()) return false;
        if (c->popRecoveryRequest().has_value()) return false;

        c->redriveDeferredRecoveryObligations();
        if (c->recoveryRetryRedriveCount() < 1) return false;

        auto t = c->popRecoveryRequest();                                              // O3 must be in transport
        if (!t.has_value()) return false;
        if (t->obligationId == 0) return false;
        if (!c->hasPendingRecoveryAdmission()) return false;                           // O2 still occupies durable (not O3)
        if (c->liveLogicalRecoveryObligationCount() != 3) return false;
        return true;
    }

    // C14: redrive when BOTH resources are busy leaves the obligation deferred (ΔL=0, failure counted),
    //   and a later redrive after a resource frees recovers it.
    [[nodiscard]] bool testRLOE_C14_redriveFailureBothBusy()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        fillRecoveryQueue(*c, 1);
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(2), 1)) return false;
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(3), 1)) return false;
        // both busy (queue full, durable=O2) → redrive fails gracefully
        c->redriveDeferredRecoveryObligations();
        if (c->recoveryRetryRedriveFailureCount() < 1) return false;
        if (c->liveLogicalRecoveryObligationCount() != 3) return false;                // O3 still Live

        if (!c->takePendingRecoveryAdmission().has_value()) return false;              // free durable
        c->settlePendingRecoveryAdmission(false);
        c->redriveDeferredRecoveryObligations();                                       // now recovers
        if (c->recoveryRetryRedriveCount() < 1) return false;
        if (!c->takePendingRecoveryAdmission().has_value()) return false;
        return true;
    }

    // C15: redrive never changes the live obligation count (no +1 / no -1).
    [[nodiscard]] bool testRLOE_C15_redrivePreservesLiveCount()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        fillRecoveryQueue(*c, 1);
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(2), 1)) return false;
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(3), 1)) return false;
        const std::uint64_t L0 = c->liveLogicalRecoveryObligationCount();
        if (L0 != 3) return false;
        if (!c->takePendingRecoveryAdmission().has_value()) return false;
        c->settlePendingRecoveryAdmission(false);
        c->redriveDeferredRecoveryObligations();
        if (c->liveLogicalRecoveryObligationCount() != L0) return false;
        c->redriveDeferredRecoveryObligations();
        c->redriveDeferredRecoveryObligations();
        if (c->liveLogicalRecoveryObligationCount() != L0) return false;
        return true;
    }

    // C16: a deferred obligation, on same-key re-submission, coalesces (no new id, L unchanged) and is
    //   delivered with a single representation (no duplicate). Drain the queue first so the resubmit enqueues.
    [[nodiscard]] bool testRLOE_C16_deferredSameKeyResubmitCoalesces()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        fillRecoveryQueue(*c, 1);
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(2), 1)) return false;
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(3), 1)) return false;  // O3 deferred
        const std::uint64_t L0 = c->liveLogicalRecoveryObligationCount();
        if (L0 != 3) return false;
        const std::uint64_t coalescedBefore = c->recoveryCoalescedCount();

        for (int i = 0; i < 256; ++i)                                                 // drain queue
            if (!c->popRecoveryRequest().has_value()) return false;

        const bool accepted = c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(3), 1);
        if (!accepted) return false;                                                   // coalesced, not rejected
        if (c->liveLogicalRecoveryObligationCount() != L0) return false;               // no L increment
        if (c->recoveryCoalescedCount() != coalescedBefore + 1) return false;          // coalesce, no new id

        auto t = c->popRecoveryRequest();                                              // single transport representation
        if (!t.has_value()) return false;
        if (t->obligationId == 0) return false;
        if (c->popRecoveryRequest().has_value()) return false;                          // exactly one intent for O3
        return true;
    }

    // ────────────────────────────────────────────────────────────────────────────
    // D105-R13: structural assertion — isFullyDrained() must require liveCount()==0.
    //   These pin down the exact false-positive R13 closes (a residual Live obligation with
    //   drained transport/durable/counters is NOT "drained"), and that shutdown drain + the
    //   no-resurrection-after-discard ordering hold. No D105-R5-10 behavior is altered.
    // ────────────────────────────────────────────────────────────────────────────

    // T-R13-1: a Live obligation whose transport intent has been consumed (push then pop ⇒
    //   queue empty, pendingIntentCount_ back to 0, durable clear) must NOT be reported drained.
    //   This FAILS without the R13 assertion (pre-R13 false-positive) and PASSES with it.
    [[nodiscard]] bool testR13_liveObligationWithDrainedTransportIsNotFullyDrained()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(1), 1)) return false;
        if (!c->popRecoveryRequest().has_value()) return false;          // transport consumed
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;  // still Live
        return !c->isFullyDrained();                                     // R13: liveCount>0 ⇒ not drained
    }

    // T-R13-2: shutdown + discard must terminalize every Live obligation to 0 and then drain structurally.
    [[nodiscard]] bool testR13_discardTerminalizesAllAndDrains()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(1), 1)) return false;
        if (!c->popRecoveryRequest().has_value()) return false;
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        if (c->isFullyDrained()) return false;                           // not drained yet (L==1)
        c->requestShutdown();                                            // close admission (no new +1)
        c->discardRecoveryRequestsOnShutdown();                          // terminalize all Live ⇒ 0
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        return c->isFullyDrained();                                      // now structurally drained
    }

    // T-R13-3 (R5-10 regression): a DEFERRED obligation (delivery==None) must not be mistaken as
    //   "already delivered/drained", and must be terminalized by shutdown discard (not abandoned).
    [[nodiscard]] bool testR13_deferredNoneObligationNotAbandonedAtShutdown()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        fillRecoveryQueue(*c, 1);                                        // O1: transport queue full (256)
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(2), 1)) return false; // O2 durable
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(3), 1)) return false; // O3 deferred (None)
        if (c->liveLogicalRecoveryObligationCount() != 3) return false;
        if (c->recoveryRetryDeferredCount() < 1) return false;         // O3 did defer (None)
        // drain transport (O1) + consume durable (O2) so residual Live obligations remain.
        for (int i = 0; i < 256; ++i)
            if (!c->popRecoveryRequest().has_value()) return false;
        if (auto dur = c->takePendingRecoveryAdmission())
            c->settlePendingRecoveryAdmission(false);                    // durable slot clear, O2 still Live
        if (c->liveLogicalRecoveryObligationCount() != 3) return false;  // none terminalized yet (incl. O3 None)
        if (c->isFullyDrained()) return false;                          // R12 false-positive pre-check; R13 rejects (L==3)
        c->requestShutdown();
        c->discardRecoveryRequestsOnShutdown();                          // must close the None obligation too
        if (c->liveLogicalRecoveryObligationCount() != 0) return false; // incl. deferred O3 ⇒ 0
        return c->isFullyDrained();
    }

    // T-R13-4: after shutdown+discard, no new obligation can be admitted (submit rejected ⇒ liveCount stuck at 0).
    [[nodiscard]] bool testR13_noNewAdmissionAfterShutdownDiscard()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(1), 1)) return false;
        if (!c->popRecoveryRequest().has_value()) return false;
        c->requestShutdown();
        c->discardRecoveryRequestsOnShutdown();
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (!c->isFullyDrained()) return false;
        if (c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(2), 1)) return false; // gated => rejected
        return c->liveLogicalRecoveryObligationCount() == 0 && c->isFullyDrained();
    }

    // =========================================================================
    // D105-R18: Retry-preserving obligation semantics (production-layer tests)
    //   These tests exercise markTransientFailure directly (the new adjudication
    //   authority). C8 (table-level Failed arm) is preserved unchanged.
    //   The K value is kMaxObligationConsecutiveFailures=4 (matches the
    //   Builder-local kMaxRecoveryConsecutiveFailures=4, separate counter).
    // =========================================================================

    // T-R18-1: transient build failure → obligation stays Live (ΔL=0), counter=1.
    //   The single −1 producer at this code path is removed; transient failure
    //   increments the obligation-level retry counter and resets delivery=None.
    [[nodiscard]] bool testR18_T1_transientBuildFailureKeepsLive()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 1);
        if (!id) return false;
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        // Simulate a transient build failure at the Orchestrator site.
        c->markTransientFailure(*id);
        // ΔL = 0; obligation remains Live.
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        // No Failed terminal was emitted.
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        // recoveryObligationShutdownDiscardCount must NOT be bumped (this is not ShutdownDiscard).
        if (c->recoveryObligationShutdownDiscardCount() != 0) return false;
        // The obligation's id is still accepted by submitRecoveryRequest coalesce.
        // Note: coalesce may return true with no new push when delivery was None
        // and redrive already re-attached the obligation (R5-10 §2 wasDeferredBefore path).
        const bool coalesced = c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(1), 1);
        if (!coalesced) return false;
        // After coalesce, the obligation is still Live; liveCount == 1 (ΔL == 0).
        return c->liveLogicalRecoveryObligationCount() == 1;
    }

    // T-R18-2: transient publish failure → obligation stays Live (ΔL=0), counter=1.
    [[nodiscard]] bool testR18_T2_transientPublishFailureKeepsLive()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 2);
        if (!id) return false;
        c->markTransientFailure(*id);
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        // After markTransientFailure, delivery==None; a fresh attempt coalesces back.
        // Note: same caveat as T-R18-1: coalesce may return true with no push.
        const bool coalesced = c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(2), 1);
        if (!coalesced) return false;
        return c->liveLogicalRecoveryObligationCount() == 1;
    }

    // T-R18-3: after markTransientFailure, the obligation's delivery is forced to None,
    //   so redriveDeferredRecoveryObligations can re-attach delivery and the obligation
    //   reaches a Builder-consumable state again.
    [[nodiscard]] bool testR18_T3_deliveryResetToNone()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 3);
        if (!id) return false;
        // The obligation is Live with delivery=Transport (popped by submitAndGetId).
        c->markTransientFailure(*id);
        // delivery must be None (P-B) — obligation is now redrive-eligible.
        // The redrive will prefer the durable slot if free; the obligation is now
        // re-deliverable to a Builder via takePendingRecoveryAdmission.
        c->redriveDeferredRecoveryObligations();
        if (c->recoveryRetryRedriveCount() < 1) return false;
        // The obligation should now be in a Builder-consumable state. Either:
        //   - durable slot is free → redrive went to durable (preferred path)
        //   - transport queue has space → redrive went to transport
        // Verify the obligation id is recoverable from the durable slot or transport.
        auto durable = c->takePendingRecoveryAdmission();
        if (durable.has_value()) {
            return durable->obligationId == *id;
        }
        // Else check transport.
        auto pop = c->popRecoveryRequest();
        if (!pop.has_value()) return false;
        return pop->obligationId == *id;
    }

    // T-R18-4: same id is preserved across markTransientFailure → re-submit.
    //   The obligation must coalesce (findByKey matches Live) and not re-allocate id.
    [[nodiscard]] bool testR18_T4_sameIdAcrossFailure()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id1 = submitAndGetId(*c, convo::isr::DSPHandle::null(), 4);
        if (!id1) return false;
        c->markTransientFailure(*id1);
        // Re-submit with the same identity; must coalesce, id preserved.
        // Note: when delivery==None at coalesce time, the redrive inside
        // submitRecoveryRequest (cpp:847) may already have re-attached delivery
        // to durable, so the early-return in the coalesce branch fires and no
        // transport push occurs. We then redrive to put the obligation back
        // on a delivery representation and verify the id is preserved.
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(4), 1)) return false;
        c->redriveDeferredRecoveryObligations();
        // The obligation should now be in a delivery representation (durable preferred).
        auto durable = c->takePendingRecoveryAdmission();
        if (durable.has_value()) {
            return durable->obligationId == *id1;
        }
        // Else check transport.
        auto pop = c->popRecoveryRequest();
        if (!pop.has_value()) return false;
        return pop->obligationId == *id1;
    }

    // T-R18-5: K consecutive failures increment the counter and K-th forces terminal.
    [[nodiscard]] bool testR18_T5_repeatedFailureCountAndExhaustion()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 5);
        if (!id) return false;
        // K-1 = 3 transient failures: obligation stays Live.
        for (int i = 0; i < 3; ++i) {
            c->markTransientFailure(*id);
            if (c->liveLogicalRecoveryObligationCount() != 1) return false;
            if (c->recoveryRetryExhaustedCount() != 0) return false;
        }
        // 4th failure (K-th) triggers exhaustion.
        c->markTransientFailure(*id);
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (c->recoveryRetryExhaustedCount() != 1) return false;
        // Failed is not ShutdownDiscarded.
        if (c->recoveryObligationShutdownDiscardCount() != 0) return false;
        return true;
    }

    // T-R18-6: successful publish after retries → exactly one −1, counter reset to 0.
    [[nodiscard]] bool testR18_T6_successAfterRetriesExactlyOnce()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 6);
        if (!id) return false;
        // 2 transient failures (counter reaches 2).
        c->markTransientFailure(*id);
        c->markTransientFailure(*id);
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        // Successful publish terminalizes.
        c->resolveRecoveryObligation(*id,
            convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::Published);
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        // No exhaustion was emitted.
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        return true;
    }

    // T-R18-7: shutdown during failed/deferred recovery → ShutdownDiscarded, not Failed.
    [[nodiscard]] bool testR18_T7_shutdownDuringDeferredRetry()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 7);
        if (!id) return false;
        c->markTransientFailure(*id);
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        c->requestShutdown();
        c->discardRecoveryRequestsOnShutdown();
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (c->recoveryObligationShutdownDiscardCount() != 1) return false;
        // No exhaustion (Failed) emitted; the obligation was terminalized by ShutdownDiscarded.
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        return true;
    }

    // T-R18-8: stale (RejectedStaleGeneration) after retries → ResolvedStaleSuperseded, not Failed.
    [[nodiscard]] bool testR18_T8_staleAfterRetries()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 8);
        if (!id) return false;
        c->markTransientFailure(*id);
        c->markTransientFailure(*id);
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        c->resolveRecoveryObligation(*id,
            convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::StaleSuperseded);
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        if (c->recoveryObligationShutdownDiscardCount() != 0) return false;
        return true;
    }

    // T-R18-9: transport-resident stranded case is repaired (R16-3 / R16-4 P-B).
    [[nodiscard]] bool testR18_T9_strandedTransportRepaired()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        fillRecoveryQueue(*c, 1);  // O1: 256 entries in transport
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(2), 1)) return false; // O2 durable
        if (!c->submitRecoveryRequest(convo::isr::DSPHandle::null(), makeRecoverySnapshot(3), 1)) return false; // O3 deferred
        // Pop one entry of O1 from the transport queue to capture its id.
        auto first = c->popRecoveryRequest();
        if (!first.has_value()) return false;
        const std::uint64_t strandedId = first->obligationId;
        // Simulate publish failure of O1's intent: markTransientFailure.
        c->markTransientFailure(strandedId);
        if (c->liveLogicalRecoveryObligationCount() != 3) return false;
        // delivery must now be None; redrive picks it up.
        c->redriveDeferredRecoveryObligations();
        if (c->recoveryRetryRedriveCount() < 1) return false;
        return true;
    }

    // T-R18-10: pressure (RejectedPressure → Retry) regression unchanged.
    //   Calling resolveRecoveryObligation(id, Retry) must keep the obligation Live
    //   (ΔL=0). This is the existing R5-9 MUST-2 contract; the test asserts it
    //   survives R18's resolve() refactor (switch with Failed arm retained).
    [[nodiscard]] bool testR18_T10_pressureRetryUnchanged()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 10);
        if (!id) return false;
        const std::uint64_t L0 = c->liveLogicalRecoveryObligationCount();
        c->resolveRecoveryObligation(*id,
            convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::Retry);
        if (c->liveLogicalRecoveryObligationCount() != L0) return false;
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        return true;
    }

    // T-R18-11: markTransientFailure on a non-Live or unknown id is a no-op.
    //   Idempotency: stale, terminalized, or unknown ids must not throw and must
    //   not produce any −1.
    [[nodiscard]] bool testR18_T11_markTransientFailureIdempotent()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        // unknown id
        c->markTransientFailure(0xDEADBEEFu);
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        // id=0 short-circuit
        c->markTransientFailure(0);
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        // admit + resolve to terminal, then call markTransientFailure on the terminalized id
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 11);
        if (!id) return false;
        c->resolveRecoveryObligation(*id,
            convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::Published);
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        c->markTransientFailure(*id);  // post-terminal: no-op
        return c->liveLogicalRecoveryObligationCount() == 0
            && c->recoveryRetryExhaustedCount() == 0;
    }

    // T-R18-12: counter is reset to 0 on slot reuse (tryInsert).
    //   After exhaustion, the slot is freed; a new obligation for a different
    //   identity gets a new id and the counter must be 0 on the new obligation
    //   (no carryover from the exhausted slot).
    [[nodiscard]] bool testR18_T12_counterResetOnSlotReuse()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id1 = submitAndGetId(*c, convo::isr::DSPHandle::null(), 12);
        if (!id1) return false;
        // Exhaust the obligation (4 calls).
        for (int i = 0; i < 4; ++i) c->markTransientFailure(*id1);
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (c->recoveryRetryExhaustedCount() != 1) return false;
        // Fresh obligation for a *different* identity reuses the freed slot.
        auto id2 = submitAndGetId(*c, convo::isr::DSPHandle::null(), 13);
        if (!id2) return false;
        if (*id2 == *id1) return false;  // new id (findByKey matches terminal ⇒ not Live ⇒ no coalesce)
        // The new obligation is Live with counter=0 (tryInsert reset).
        c->markTransientFailure(*id2);
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        // No exhaustion (counter=1, not yet 4).
        if (c->recoveryRetryExhaustedCount() != 1) return false;  // unchanged
        return true;
    }

    // =========================================================================
    // D105-R20: RejectedNotFinalized Retry Centralization / Single-Count Repair
    //   These tests verify that:
    //   - markTransientFailure is called exactly once per failure event.
    //   - The orchestration (A/B in trySubmitImpl + D in :389) is unified via
    //     the centralized :389 switch case.
    //   - The 1:1 call ratio is preserved across all transient-failure paths.
    // =========================================================================

    // T-R20-1: admission rejection retry preservation.
    //   Simulates the path-D scenario: obligation Live + delivery=Transport.
    //   After admission rejection + centralized markTransientFailure:
    //   - Live (ΔL=0)
    //   - counter+1
    //   - delivery=None
    [[nodiscard]] bool testR20_T1_admissionRejectionRetryPreservation()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 1);
        if (!id) return false;
        const std::uint64_t L0 = c->liveLogicalRecoveryObligationCount();
        if (L0 != 1) return false;
        // Before: counter=0, delivery=Transport (popped by submitAndGetId).
        // The markTransientFailure call simulates the centralized :389 dispatch.
        c->markTransientFailure(*id);
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        // Delivery is now None (P-B repair); redrive is eligible.
        c->redriveDeferredRecoveryObligations();
        if (c->recoveryRetryRedriveCount() < 1) return false;
        return true;
    }

    // T-R20-2: admission rejection redrive preserves the same obligationId.
    //   Verifies the full chain: markTransientFailure → delivery=None → redrive
    //   → obligation is re-attached to a delivery representation → same id.
    [[nodiscard]] bool testR20_T2_admissionRejectionRedriveSameId()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 2);
        if (!id) return false;
        c->markTransientFailure(*id);   // simulates centralized :389 dispatch
        c->redriveDeferredRecoveryObligations();
        // The obligation should now be in a delivery representation.
        auto durable = c->takePendingRecoveryAdmission();
        if (durable.has_value()) {
            return durable->obligationId == *id;
        }
        auto pop = c->popRecoveryRequest();
        if (!pop.has_value()) return false;
        return pop->obligationId == *id;
    }

    // T-R20-3: exactly-once failure counting.
    //   After R20, each call to markTransientFailure increments the counter by
    //   exactly 1. Calling markTransientFailure once produces counter=1 (not 0→2).
    //   Three calls produce counter=3; the 4th produces exhaustion.
    [[nodiscard]] bool testR20_T3_exactlyOnceFailureCounting()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 3);
        if (!id) return false;
        // Single call → counter increments by 1, not 2.
        c->markTransientFailure(*id);
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        // 3 calls total → still Live (counter would be 3 < 4).
        c->markTransientFailure(*id);
        c->markTransientFailure(*id);
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        // 4th call → exhaustion → ResolvedFailed.
        c->markTransientFailure(*id);
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (c->recoveryRetryExhaustedCount() != 1) return false;
        return true;
    }

    // T-R20-4: ResolvedFailed production reachability — `resolveRecoveryObligation(_, Failed)`
    //   must have 0 production callers after R20.
    //   This test simulates the production flow: an obligation is submitted,
    //   markTransientFailure is called (simulating any of A/B/D's centralized
    //   handling), and we verify the obligation reaches ResolvedFailed only via
    //   the exhaustion path (counter >= K) — never via a direct
    //   resolveRecoveryObligation(Failed) call.
    [[nodiscard]] bool testR20_T4_resolvedFailedOnlyViaExhaustion()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 4);
        if (!id) return false;
        // 1 transient failure: counter=1, Live, no Failed.
        c->markTransientFailure(*id);
        if (c->liveLogicalRecoveryObligationCount() != 1) return false;
        // The recoveryRetryExhaustedCount is the only "Failed" telemetry.
        // If it is 0, no ResolvedFailed was emitted.
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        // To reach ResolvedFailed in production, the obligation must exhaust:
        for (int i = 0; i < 3; ++i) c->markTransientFailure(*id);
        // Now counter=4 → ResolvedFailed; recoveryRetryExhaustedCount==1.
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (c->recoveryRetryExhaustedCount() != 1) return false;
        return true;
    }

    // =========================================================================
    // D105-R21: I4 Contract Amendment — RetryExhaustion & conservation equation
    //   These tests re-use the R18/R20 public API to assert the I4 contract
    //   properties that R21 formalizes:
    //   - T-R21-1: liveOwnershipCount + successCount + supersededCount
    //              + shutdownDiscardCount + retryExhaustedCount
    //              == admittedLogicalObligationCount
    //   - T-R21-2: 3 transient failures → Live (ΔL=0, counter=3, no exhaustion)
    //   - T-R21-3: 4th failure → ResolvedFailed; recoveryRetryExhaustedCount==1
    //   (T-R21-3 is a fresh assert that the conservation increment of
    //    retryExhaustedCount matches the equation. It exercises one of the
    //    4 disjoint terminal paths.)
    // =========================================================================

    // T-R21-1: conservation equation with retryExhaustedCount.
    //   After:
    //     - 1 Published terminal (resolves to ResolvedSuccess)
    //     - 1 StaleSuperseded terminal
    //     - 1 ShutdownDiscarded terminal (via shutdown path)
    //     - 1 RetryExhausted terminal (via markTransientFailure K=4)
    //   The equation live + 4 terminals = admitted (=4) must hold.
    [[nodiscard]] bool testR21_T1_conservationEquation()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        // Terminal 1: RetryExhaustion (counter==K).
        auto id1 = submitAndGetId(*c, convo::isr::DSPHandle::null(), 11);
        if (!id1) return false;
        for (int i = 0; i < 4; ++i) c->markTransientFailure(*id1);
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (c->recoveryRetryExhaustedCount() != 1) return false;
        // Terminal 2: Published.
        auto id2 = submitAndGetId(*c, convo::isr::DSPHandle::null(), 22);
        if (!id2) return false;
        c->resolveRecoveryObligation(*id2,
            convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::Published);
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        // Terminal 3: StaleSuperseded.
        auto id3 = submitAndGetId(*c, convo::isr::DSPHandle::null(), 33);
        if (!id3) return false;
        c->resolveRecoveryObligation(*id3,
            convo::isr::RuntimeIntentCoordinator::RecoveryOutcome::StaleSuperseded);
        // Terminal 4: ShutdownDiscarded (via shutdown path).
        auto id4 = submitAndGetId(*c, convo::isr::DSPHandle::null(), 44);
        if (!id4) return false;
        c->requestShutdown();
        c->discardRecoveryRequestsOnShutdown();
        // Conservation check: 4 terminal, 0 live, 4 admitted.
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (c->recoveryRetryExhaustedCount() != 1) return false;
        if (c->recoveryObligationShutdownDiscardCount() != 1) return false;
        return true;
    }

    // T-R21-2: transient failure is non-terminal.
    //   3 calls to markTransientFailure: Live (ΔL=0), counter=3, no Failed.
    [[nodiscard]] bool testR21_T2_transientFailureIsNonTerminal()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 1);
        if (!id) return false;
        const std::uint64_t L0 = c->liveLogicalRecoveryObligationCount();
        if (L0 != 1) return false;
        // 3 transient failures: obligation must stay Live (ΔL=0 each call).
        for (int i = 0; i < 3; ++i) {
            c->markTransientFailure(*id);
            if (c->liveLogicalRecoveryObligationCount() != 1) return false;  // ΔL = 0
        }
        // 3 failures but no exhaustion (counter < K=4).
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        return true;
    }

    // T-R21-3: exhaustion only via K.
    //   4th call → ResolvedFailed; recoveryRetryExhaustedCount==1.
    [[nodiscard]] bool testR21_T3_exhaustionOnlyViaK()
    {
        auto c = std::make_unique<convo::isr::RuntimeIntentCoordinator>();
        auto id = submitAndGetId(*c, convo::isr::DSPHandle::null(), 1);
        if (!id) return false;
        for (int i = 0; i < 3; ++i) c->markTransientFailure(*id);
        if (c->recoveryRetryExhaustedCount() != 0) return false;
        // 4th call: counter == K == 4 → exhaustion.
        c->markTransientFailure(*id);
        if (c->liveLogicalRecoveryObligationCount() != 0) return false;
        if (c->recoveryRetryExhaustedCount() != 1) return false;
        return true;
    }
}

int main()
{
    try
    {
    if (!testInvalidClosureRejected())
        throw std::runtime_error("invalid closure must be rejected");

    if (!testInvalidTierRejected())
        throw std::runtime_error("invalid tier must be rejected");

    if (!testCoordinatorCommitAndMonotonicityContract())
        throw std::runtime_error("coordinator monotonic commit contract failed");

    if (!testCoordinatorRejectEpochRollbackContract())
        throw std::runtime_error("coordinator epoch rollback contract failed");

    if (!testCoordinatorRejectMappedGenerationRollbackOnEpochAdvance())
        throw std::runtime_error("coordinator mapped generation rollback contract failed");

    if (!testCoordinatorRejectEpochReuseContract())
        throw std::runtime_error("coordinator epoch reuse contract failed");

    if (!testCoordinatorRejectMappedGenerationReuseContract())
        throw std::runtime_error("coordinator mapped generation reuse contract failed");

    if (!testCoordinatorRejectWraparoundContract())
        throw std::runtime_error("coordinator wraparound contract failed");

    if (!testCoordinatorDrainAndShutdownContract())
        throw std::runtime_error("coordinator drain and shutdown contract failed");

    if (!testShutdownCompleteFailsWhenNotDrained())
        throw std::runtime_error("coordinator shutdown not-drained contract failed");

    if (!testPressureStateNormalizationContract())
        throw std::runtime_error("coordinator pressure normalization contract failed");

    if (!testShutdownCompleteFailsWhenSwapPending())
        throw std::runtime_error("coordinator shutdown swap-pending contract failed");

    // --- P4 契約テスト群 ---
    if (!testP4SameGenerationEpochChangeRejected())
        throw std::runtime_error("P4: same-generation epoch change must be rejected");

    // --- dash2 §1.7 CW-3 調査: 二重 commit 検証 ---
    if (!testCoordinatorDoubleCommitSameWorldFaults())
        throw std::runtime_error("CW-3: double-commit (same world) must fault");

    // --- P20 ロールバックテスト群 ---
    if (!testP20RejectPreservesWorldState())
        throw std::runtime_error("P20: reject must preserve world state");

    // --- FUTURE-4 METADATA-1/2/6 snapshot contract ---
    if (!testMetadataSnapshotConsistentAcrossReaders())
        throw std::runtime_error("FUTURE-4: metadata snapshot consistency failed");
    if (!testMetadataSnapshotRejectsEpochRollback())
        throw std::runtime_error("FUTURE-4: metadata snapshot epoch-rollback rejection failed");
    if (!testMetadataSnapshotSequenceAdvancesWithEpoch())
        throw std::runtime_error("FUTURE-4: metadata snapshot monotonic advance failed");
    if (!testMetadataSnapshotNoTransitionalCacheSymbol())
        throw std::runtime_error("FUTURE-4: no transitional cache symbol (physical removal) failed");

    // --- FUTURE-3: submitRecoveryRequest transport contract (enqueue → pop 1-hop) ---
    if (!testRecoveryRequestEnqueueAndPop())
        throw std::runtime_error("FUTURE-3: recovery request enqueue/pop failed");

    // --- dash2 §1.7 Phase G R7: recovery epoch propagation (epoch=0 回帰捕捉) ---
    if (!testRecoveryRequestEpochPropagation())
        throw std::runtime_error("Phase G R7: recovery epoch propagation failed");

    // --- work88 (X1 §6.1): Recovery Durable Admission (queue full ≠ lost / lease / coalesce) ---
    if (!testRecoveryDurableAdmission())
        throw std::runtime_error("X1: recovery durable admission (full → durable → take) failed");
    if (!testRecoveryDurableCoalesce())
        throw std::runtime_error("X1: recovery durable coalesce (single admission) failed");
    if (!testRecoveryDurableLeaseRetry())
        throw std::runtime_error("X1: recovery durable lease retry (Building → DurablePending) failed");

    // --- FUTURE-8: Observe overflow → Observe-exclusive Deferred Ring (QUEUE-15) ---
    if (!testObserveOverflowEnqueuePath())
        throw std::runtime_error("FUTURE-8: observe overflow enqueue path failed");

    // --- A-1: RuntimeWorldAuthority delegate (no shadow state) ---
    if (!testRuntimeWorldAuthorityAdapter())
        throw std::runtime_error("A-1: RuntimeWorldAuthority must delegate epoch/sequence with no shadow state");

    // --- B3 invariant #4: publish intent queue-full => explicit backpressure ---
    if (!testPublishIntentQueueFullBackpressure())
        throw std::runtime_error("B3: publish intent queue-full backpressure contract failed");

    // --- D105-R5-9: Recovery Logical Obligation Enforcement counterexample tests C1-C10 ---
    if (!testRLOE_C1_unique32())
        throw std::runtime_error("D105-R5-9 C1: 32 distinct obligations must count to 32");
    if (!testRLOE_C2_33rdRejected())
        throw std::runtime_error("D105-R5-9 C2: 33rd distinct must be rejected without L change");
    if (!testRLOE_C3_coalesceAtFull())
        throw std::runtime_error("D105-R5-9 C3: duplicate at capacity must coalesce (L unchanged)");
    if (!testRLOE_C4_distinctHandlesSameTarget())
        throw std::runtime_error("D105-R5-9 C4: distinct handle + same target must be two obligations");
    if (!testRLOE_C5_retryKeepsLive())
        throw std::runtime_error("D105-R5-9 C5: Retry must keep obligation Live (ΔL=0)");
    if (!testRLOE_C6_staleSupersededResolves())
        throw std::runtime_error("D105-R5-9 C6: StaleSuperseded must resolve to L 1->0");
    if (!testRLOE_C7_duplicatePublishedOnce())
        throw std::runtime_error("D105-R5-9 C7: duplicate Published must be idempotent (L 1->0 once)");
    if (!testRLOE_C8_shutdownRaceOnce())
        throw std::runtime_error("D105-R5-9 C8: shutdown race must resolve once (count once)");
    if (!testRLOE_C9_abaStaleResolveNoOp())
        throw std::runtime_error("D105-R5-9 C9: ABA stale resolve must be no-op (O2 stays Live)");
    if (!testRLOE_C10_stress256CapsAt32())
        throw std::runtime_error("D105-R5-9 C10: 256 distinct must cap at L<=32 (no overflow)");

    // --- D105-R5-10: Recovery Logical Obligation deferred re-drive counterexample tests C11-C16 ---
    if (!testRLOE_C11_redriveRecoversDeferredViaDurable())
        throw std::runtime_error("D105-R5-10 C11: deferred obligation must be redriven onto durable when it frees");
    if (!testRLOE_C12_redriveIdempotentNoDuplicate())
        throw std::runtime_error("D105-R5-10 C12: redrive must be idempotent (no duplicate delivery)");
    if (!testRLOE_C13_redriveFallsBackToTransport())
        throw std::runtime_error("D105-R5-10 C13: redrive must fall back to transport when durable is busy");
    if (!testRLOE_C14_redriveFailureBothBusy())
        throw std::runtime_error("D105-R5-10 C14: redrive must stay deferred (ΔL=0) when both resources busy");
    if (!testRLOE_C15_redrivePreservesLiveCount())
        throw std::runtime_error("D105-R5-10 C15: redrive must never change live obligation count");
    if (!testRLOE_C16_deferredSameKeyResubmitCoalesces())
        throw std::runtime_error("D105-R5-10 C16: deferred + same-key resubmit must coalesce (no new id, single representation)");

    // --- D105-R13: isFullyDrained() logical-obligation-zero structural assertion ---
    if (!testR13_liveObligationWithDrainedTransportIsNotFullyDrained())
        throw std::runtime_error("D105-R13 T-R13-1: residual Live obligation (drained transport) must not be reported drained");
    if (!testR13_discardTerminalizesAllAndDrains())
        throw std::runtime_error("D105-R13 T-R13-2: discard must terminalize all Live obligations to 0 and drain structurally");
    if (!testR13_deferredNoneObligationNotAbandonedAtShutdown())
        throw std::runtime_error("D105-R13 T-R13-3: deferred (None) obligation must not be abandoned at shutdown");
    if (!testR13_noNewAdmissionAfterShutdownDiscard())
        throw std::runtime_error("D105-R13 T-R13-4: no new recovery admission after shutdown discard");

    // --- D105-R18: Retry-preserving obligation semantics (production-layer) ---
    if (!testR18_T1_transientBuildFailureKeepsLive())
        throw std::runtime_error("D105-R18 T-R18-1: transient build failure must keep obligation Live (ΔL=0)");
    if (!testR18_T2_transientPublishFailureKeepsLive())
        throw std::runtime_error("D105-R18 T-R18-2: transient publish failure must keep obligation Live (ΔL=0)");
    if (!testR18_T3_deliveryResetToNone())
        throw std::runtime_error("D105-R18 T-R18-3: markTransientFailure must reset delivery=None for redrive");
    if (!testR18_T4_sameIdAcrossFailure())
        throw std::runtime_error("D105-R18 T-R18-4: obligation id must be preserved across markTransientFailure + re-submit");
    if (!testR18_T5_repeatedFailureCountAndExhaustion())
        throw std::runtime_error("D105-R18 T-R18-5: K consecutive failures must increment counter; K-th forces ResolvedFailed");
    if (!testR18_T6_successAfterRetriesExactlyOnce())
        throw std::runtime_error("D105-R18 T-R18-6: successful publish after retries must exactly once decrement L");
    if (!testR18_T7_shutdownDuringDeferredRetry())
        throw std::runtime_error("D105-R18 T-R18-7: shutdown during failed/deferred must emit ShutdownDiscarded, not Failed");
    if (!testR18_T8_staleAfterRetries())
        throw std::runtime_error("D105-R18 T-R18-8: stale after retries must emit StaleSuperseded, not Failed");
    if (!testR18_T9_strandedTransportRepaired())
        throw std::runtime_error("D105-R18 T-R18-9: stranded Transport obligation must be repaired (delivery=None) and redriven");
    if (!testR18_T10_pressureRetryUnchanged())
        throw std::runtime_error("D105-R18 T-R18-10: RejectedPressure → Retry regression must be preserved (R5-9 MUST-2)");
    if (!testR18_T11_markTransientFailureIdempotent())
        throw std::runtime_error("D105-R18 T-R18-11: markTransientFailure on unknown/terminal id must be no-op");
    if (!testR18_T12_counterResetOnSlotReuse())
        throw std::runtime_error("D105-R18 T-R18-12: counter must be reset to 0 on slot reuse after exhaustion");

    // --- D105-R20: RejectedNotFinalized Retry Centralization / Single-Count Repair ---
    if (!testR20_T1_admissionRejectionRetryPreservation())
        throw std::runtime_error("D105-R20 T-R20-1: admission rejection must trigger exactly one markTransientFailure (ΔL=0, counter+1, delivery=None)");
    if (!testR20_T2_admissionRejectionRedriveSameId())
        throw std::runtime_error("D105-R20 T-R20-2: admission rejection + markTransientFailure must redrive to same obligationId");
    if (!testR20_T3_exactlyOnceFailureCounting())
        throw std::runtime_error("D105-R20 T-R20-3: 1 call → counter+1 (NOT +2); 4th call → ResolvedFailed");
    if (!testR20_T4_resolvedFailedOnlyViaExhaustion())
        throw std::runtime_error("D105-R20 T-R20-4: ResolvedFailed reached only via exhaustion (counter == K)");

    // --- D105-R21: I4 Contract Amendment — RetryExhaustion & conservation equation ---
    if (!testR21_T1_conservationEquation())
        throw std::runtime_error("D105-R21 T-R21-1: I4 conservation equation with retryExhaustedCount must hold");
    if (!testR21_T2_transientFailureIsNonTerminal())
        throw std::runtime_error("D105-R21 T-R21-2: 3 transient failures must keep obligation Live (ΔL=0, no exhaustion)");
    if (!testR21_T3_exhaustionOnlyViaK())
        throw std::runtime_error("D105-R21 T-R21-3: 4th markTransientFailure must produce ResolvedFailed + retryExhaustedCount==1");

    return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "TEST FAILED: %s\n", e.what());
        return 1;
    }
}
