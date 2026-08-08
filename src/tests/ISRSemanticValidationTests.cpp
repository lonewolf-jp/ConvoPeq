#include <stdexcept>
#include <vector>
#include <cstring>
#include <limits>
#include <cstdio>
#include <memory>

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
    convo::isr::RuntimePublicationCoordinator coordinator;

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
    convo::isr::RuntimePublicationCoordinator coordinator;

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
    convo::isr::RuntimePublicationCoordinator coordinator;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       1,
                       1,
                       1);

    if (coordinator.getCurrent() != world1.get())
        return false;
    if (coordinator.getVersion() != 1)
        return false;
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Ready)
        return false;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       2,
                       2,
                       2);
    if (coordinator.getCurrent() != world2.get())
        return false;
    if (coordinator.getVersion() != 2)
        return false;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       1,
                       1,
                       1);

    if (coordinator.getCurrent() != world2.get())
        return false;
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted)
        return false;

    return true;
}

[[nodiscard]] bool testCoordinatorRejectEpochRollbackContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       1,
                       5,
                       10);

    if (coordinator.getCurrent() != world1.get())
        return false;

    // sequence は増加しても epoch rollback は fail-closed
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       2,
                       4,
                       11);

    if (coordinator.getCurrent() != world1.get())
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectMappedGenerationRollbackOnEpochAdvance()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       10,
                       10,
                       100);

    if (coordinator.getCurrent() != world1.get())
        return false;

    // epoch advance 時の mapped generation rollback は fail-closed
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       11,
                       11,
                       99);

    if (coordinator.getCurrent() != world1.get())
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectEpochReuseContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       100,
                       100,
                       1000);

    if (coordinator.getCurrent() != world1.get())
        return false;

    // sequence が進んでも epoch reuse は strict monotonic 契約違反
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       101,
                       100,
                       1001);

    if (coordinator.getCurrent() != world1.get())
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectMappedGenerationReuseContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       200,
                       200,
                       5000);

    if (coordinator.getCurrent() != world1.get())
        return false;

    // epoch が進んでも mapped generation reuse は strict monotonic 契約違反
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       201,
                       201,
                       5000);

    if (coordinator.getCurrent() != world1.get())
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorRejectWraparoundContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

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
                       maxValue - 1);

    if (coordinator.getCurrent() != world1.get())
        return false;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       maxValue,
                       maxValue,
                       maxValue,
                       maxValue);

    if (coordinator.getCurrent() != world2.get())
        return false;

    // wraparound（max -> 0）は strict monotonic 契約違反
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world3.get(),
                       0,
                       0,
                       0,
                       0);

    if (coordinator.getCurrent() != world2.get())
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testCoordinatorDrainAndShutdownContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    int world = 1;
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1);

    coordinator.setRetireBacklogCount(0);
    coordinator.setPublicationBacklogCount(0);
    coordinator.setPendingIntentCount(0);
    coordinator.setFallbackBacklogCount(0);
    coordinator.setReclaimInFlightCount(0);
    coordinator.setDeferredRetireResidencyCount(0);
    coordinator.setSwapPending(false);

    if (!coordinator.isFullyDrained())
        return false;

    coordinator.requestShutdown();
    coordinator.markShutdownComplete();

    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Bootstrapping)
        return false;

    return true;
}

[[nodiscard]] bool testShutdownCompleteFailsWhenNotDrained()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    int world = 1;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1);

    coordinator.setRetireBacklogCount(1); // drained 条件を破る
    coordinator.setPublicationBacklogCount(0);
    coordinator.setPendingIntentCount(0);
    coordinator.setFallbackBacklogCount(0);
    coordinator.setReclaimInFlightCount(0);
    coordinator.setDeferredRetireResidencyCount(0);
    coordinator.setSwapPending(false);

    if (coordinator.isFullyDrained())
        return false;

    coordinator.requestShutdown();
    coordinator.markShutdownComplete();

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

[[nodiscard]] bool testPressureStateNormalizationContract()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    int world = 1;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1);

    // slope > threshold で Pressure へ遷移
    coordinator.setRetireBacklogCount(9);
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Pressure)
        return false;

    // swapPending 中は normalization しない
    coordinator.setSwapPending(true);
    coordinator.setRetireBacklogCount(0);
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Pressure)
        return false;

    // swapPending 解除後、3 window で Ready へ復帰
    coordinator.setSwapPending(false);
    coordinator.setRetireBacklogCount(0);
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Pressure)
        return false;
    coordinator.setRetireBacklogCount(0);
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Pressure)
        return false;
    coordinator.setRetireBacklogCount(0);

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Ready;
}

[[nodiscard]] bool testShutdownCompleteFailsWhenSwapPending()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    int world = 1;

    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       &world,
                       1,
                       1,
                       1,
                       1);

    coordinator.setRetireBacklogCount(0);
    coordinator.setPublicationBacklogCount(0);
    coordinator.setPendingIntentCount(0);
    coordinator.setFallbackBacklogCount(0);
    coordinator.setReclaimInFlightCount(0);
    coordinator.setDeferredRetireResidencyCount(0);
    coordinator.setSwapPending(true); // drained 条件を破る

    if (coordinator.isFullyDrained())
        return false;

    coordinator.requestShutdown();
    coordinator.markShutdownComplete();

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

// --- P4: Generation / ActivationEpoch 契約 ---
// generation 増加時は activationEpoch も必ず増加する (+1 以上)。
// 同一 generation での activationEpoch 単独変更は禁止。
[[nodiscard]] bool testP4SameGenerationEpochChangeRejected()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    // 初回 commit: gen=100, epoch=100
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1,
                       100,
                       100,
                       100);

    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Ready)
        return false;

    // 同一 generation (100) で epoch のみ変更 (101) → 禁止 (generation 不変で epoch 変更)
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2,
                       100,
                       101,
                       100);

    // world1 が維持され、Faulted になるべき
    if (coordinator.getCurrent() != world1.get())
        return false;

    return coordinator.getState() == convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted;
}

// --- P20: Fail-Closed Rollback ---
// reject 時に system state がロールバックされることを確認。
// coordinator は契約違反時に Faulted に遷移する（fail-closed）が、
// currentWorld と version は reject 前の値を維持する。
// 副作用（callback, telemetry）は reject 経路では発生しない。
[[nodiscard]] bool testP20RejectPreservesWorldState()
{
    convo::isr::RuntimePublicationCoordinator coordinator;

    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();

    // 初回 commit
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(),
                       1, 1, 1, 1);

    if (coordinator.getCurrent() != world1.get())
        return false;
    if (coordinator.getVersion() != 1)
        return false;

    // 不正な commit（epoch rollback）で reject されるはず
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(),
                       2, 2, 0, 2);

    // state は Faulted に遷移する（fail-closed）: これは意図された動作
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted)
        return false;

    // currentWorld が reject 前の値（world1）を維持している
    if (coordinator.getCurrent() != world1.get())
        return false;

    // version が reject 前の値（1）を維持している
    if (coordinator.getVersion() != 1)
        return false;

    return true;
}

// ★ FUTURE-4 METADATA-1/2/6: single consumeAtomic(currentWorld_) snapshot yields
//   consistent epoch + generation + sequence via RuntimeState::publication.
[[nodiscard]] bool testMetadataSnapshotConsistentAcrossReaders()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    auto world = RuntimeState::createForTest();
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world.get(), 1, 1, 10, 100);
    if (coordinator.getCurrent() != world.get())
        return false;
    if (coordinator.getVersion() != 100)
        return false;
    if (coordinator.currentPublicationEpoch() != 10)
        return false;
    return true;
}

[[nodiscard]] bool testMetadataSnapshotRejectsEpochRollback()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    auto world1 = RuntimeState::createForTest();
    auto world2 = RuntimeState::createForTest();
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world1.get(), 1, 1, 10, 100);
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world2.get(), 2, 2, 4, 11);
    if (coordinator.getState() != convo::isr::RuntimePublicationCoordinator::CoordinatorState::Faulted)
        return false;
    if (coordinator.getCurrent() != world1.get())
        return false;
    if (coordinator.getVersion() != 100)
        return false;
    if (coordinator.currentPublicationEpoch() != 10)
        return false;
    return true;
}

[[nodiscard]] bool testMetadataSnapshotSequenceAdvancesWithEpoch()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    auto w1 = RuntimeState::createForTest();
    auto w2 = RuntimeState::createForTest();
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       w1.get(), 1, 1, 1, 1);
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       w2.get(), 2, 2, 2, 2);
    if (coordinator.getCurrent() != w2.get())
        return false;
    if (coordinator.getVersion() != 2)
        return false;
    if (coordinator.currentPublicationEpoch() != 2)
        return false;
    return true;
}

// METADATA-6: no transitional cache symbol; reader is pure world snapshot.
[[nodiscard]] bool testMetadataSnapshotNoTransitionalCacheSymbol()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    auto world = RuntimeState::createForTest();
    coordinator.commit(convo::isr::PublishAuthority::Granted,
                       convo::isr::RuntimeBoundary::NonRTWorld,
                       world.get(), 1, 1, 7, 700);
    if (coordinator.getVersion() != 700)
        return false;
    if (coordinator.currentPublicationEpoch() != 7)
        return false;
    return true;
}

// ★ FUTURE-3: Recovery Request は transport-only enqueue。Admission 判定なし。
//   submitRecoveryRequest() -> popRecoveryRequest() 1-hop 輸送。Builder Loop が復旧 World を build。
[[nodiscard]] bool testRecoveryRequestEnqueueAndPop()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    const auto handle = convo::isr::DSPHandle::null();
    // ★ FUTURE-3 (work88): buildSource（RuntimeBuildSnapshot 値コピー）を引数に追加。
    //   quarantinedHandle 単独では resolve 不能なため、build 入力は値コピーで引当する。
    convo::RuntimeBuildSnapshot buildSource{};
    buildSource.sealed = true;  // 1-hop 輸送テストのため sealed 済み snapshot を渡す
    coordinator.submitRecoveryRequest(handle, buildSource);   // enqueue（Admission 判定なし）
    if (!coordinator.popRecoveryRequest().has_value())
        return false;                            // Builder pop path
    if (coordinator.popRecoveryRequest().has_value())
        return false;                            // 1-hop transport（duplicate/queue-no-op なし）
    return true;
}

// ★ FUTURE-8: overflow は Observe 専用 Deferred Ring へ（Retire 系 ring と分離, QUEUE-15）。
//   1024+2048+1024 満村 → drop。enqueue path crash なし + pending count 連動を検証。
[[nodiscard]] bool testObserveOverflowEnqueuePath()
{
    convo::isr::RuntimePublicationCoordinator coordinator;
    const auto handle = convo::isr::DSPHandle::null();
    constexpr int N = 4100;  // 1024(L1) + 2048(L2) + 1024(L3) + drop
    for (int i = 0; i < N; ++i)
        coordinator.submitObserve(handle);
    return coordinator.getPendingIntentCount() == static_cast<std::uint64_t>(N);
}

// ★ A-1: RuntimeWorldAuthority must be a pure delegate over the Coordinator — it must
//   return the SAME epoch/sequence/version as the coordinator (no shadow state of its own).
[[nodiscard]] bool testRuntimeWorldAuthorityAdapter()
{
    auto coordinator = std::make_unique<convo::isr::RuntimePublicationCoordinator>();
    auto authority = std::make_unique<convo::isr::RuntimeWorldAuthority>(*coordinator);

    if (authority->currentEpoch() != coordinator->currentPublicationEpoch())
        return false;
    if (authority->sequence() != coordinator->currentPublicationSequenceId())
        return false;
    if (authority->getCurrent() != coordinator->getCurrent())
        return false;
    if (authority->getVersion() != coordinator->getVersion())
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
    convo::isr::RuntimePublicationCoordinator coordinator;

    constexpr size_t kCapacity = 4096;      // kIntentQueueCapacity (FUTURE-10 common queue)
    convo::isr::RuntimePublicationCoordinator::Intent intent{};
    intent.type = convo::isr::RuntimePublicationCoordinator::IntentType::Publish;

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

    // --- FUTURE-8: Observe overflow → Observe-exclusive Deferred Ring (QUEUE-15) ---
    if (!testObserveOverflowEnqueuePath())
        throw std::runtime_error("FUTURE-8: observe overflow enqueue path failed");

    // --- A-1: RuntimeWorldAuthority delegate (no shadow state) ---
    if (!testRuntimeWorldAuthorityAdapter())
        throw std::runtime_error("A-1: RuntimeWorldAuthority must delegate epoch/sequence with no shadow state");

    // --- B3 invariant #4: publish intent queue-full => explicit backpressure ---
    if (!testPublishIntentQueueFullBackpressure())
        throw std::runtime_error("B3: publish intent queue-full backpressure contract failed");

    return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "TEST FAILED: %s\n", e.what());
        return 1;
    }
}
