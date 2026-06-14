//==============================================================================
// SnapshotCoordinator.cpp - Phase 4
//==============================================================================

#include "SnapshotCoordinator.h"

#include "audioengine/AtomicAccess.h"
#include "SnapshotFactory.h"

namespace convo {

void SnapshotCoordinator::startFade(GlobalSnapshot* target, int fadeSamples) noexcept
{
	if (target == nullptr || fadeSamples <= 0)
	{
		switchImmediate(target);
		return;
	}

	// 初回適用で current が未初期化の場合は、
	// null 起点フェードを避けて即時反映する。
	if (m_slots.loadCurrent(std::memory_order_acquire) == nullptr)
	{
		switchImmediate(target);
		return;
	}

	constexpr auto snapshotDeleter = [](void* ptr) noexcept
	{
		SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(ptr));
	};

	GlobalSnapshot* oldTarget = m_slots.exchangeTarget(target, std::memory_order_acq_rel);
	if (oldTarget) {
		const uint64_t retireEpoch = m_epochProvider->currentEpoch();
		// [work37 Phase 1.2] enqueueWithRetry を使用（startFade は NonRT Timer からのみ）
		enqueueWithRetry(*m_epochProvider, oldTarget, snapshotDeleter, retireEpoch);
	}

	m_fade.start(fadeSamples);
}

void SnapshotCoordinator::advanceFade(int numSamples) noexcept
{
	m_fade.advance(numSamples);
}

bool SnapshotCoordinator::tryCompleteFade() noexcept
{
	if (!m_fade.tryComplete())
		return false;

	completeFade();
	return true;
}

void SnapshotCoordinator::resetFadeStateAndRetireTarget() noexcept
{
	constexpr auto snapshotDeleter = [](void* ptr) noexcept
	{
		SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(ptr));
	};

	GlobalSnapshot* target = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
	if (target)
	{
		const uint64_t retireEpoch = m_epochProvider->publishEpoch();
		m_epochProvider->enqueueRetire(target, snapshotDeleter, retireEpoch);
	}

	m_fade.resetToIdle();
}

void SnapshotCoordinator::completeFade() noexcept
{
	GlobalSnapshot* target = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
	if (!target)
		return;

	constexpr auto snapshotDeleter = [](void* ptr) noexcept
	{
		SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(ptr));
	};

	const uint64_t retireEpoch = m_epochProvider->publishEpoch();
	GlobalSnapshot* old = m_slots.exchangeCurrent(target, std::memory_order_acq_rel);
	if (old)
		// [work37 Phase 1.2] enqueueWithRetry を使用（completeFade は NonRT）
		enqueueWithRetry(*m_epochProvider, old, snapshotDeleter, retireEpoch);

	m_fade.resetToIdle();
}

} // namespace convo
