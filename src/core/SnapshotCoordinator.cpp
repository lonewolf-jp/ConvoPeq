//==============================================================================
// SnapshotCoordinator.cpp - Phase 4
//==============================================================================

#include "SnapshotCoordinator.h"

#include "audioengine/AtomicAccess.h"

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
	if (convo::consumeAtomic(m_current, std::memory_order_acquire) == nullptr)
	{
		switchImmediate(target);
		return;
	}

	GlobalSnapshot* oldTarget = convo::exchangeAtomic(m_target, target, std::memory_order_acq_rel);
	if (oldTarget) {
		// Audio Thread が参照中の可能性があるため、即時 delete せず RCU 遅延解放
		const uint64_t retireEpoch = m_epochCore.current();
		m_deletionQueue.enqueue(
			oldTarget,
			[](void* p) { SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(p)); },
			retireEpoch,
			DeletionEntryType::Generic
		);
	}

	convo::publishAtomic(m_fadeTotalSamples, fadeSamples, std::memory_order_release);
	convo::publishAtomic(m_fadeRemainingSamples, fadeSamples, std::memory_order_release);
	convo::publishAtomic(m_fadeAlpha, 0.0, std::memory_order_release);
	convo::publishAtomic(m_fadeCompleted, false, std::memory_order_release);
	convo::publishAtomic(m_fadeState, FadeState::FadingIn, std::memory_order_release);
}

void SnapshotCoordinator::advanceFade(int numSamples) noexcept
{
	if (convo::consumeAtomic(m_fadeState, std::memory_order_acquire) != FadeState::FadingIn)
		return;

	const int remaining = convo::consumeAtomic(m_fadeRemainingSamples, std::memory_order_acquire);
	if (remaining <= 0)
		return;

	const int newRemaining = remaining - numSamples;
	if (newRemaining <= 0)
	{
		convo::publishAtomic(m_fadeRemainingSamples, 0, std::memory_order_release);
		requestFadeCompletion();
		return;
	}

	convo::publishAtomic(m_fadeRemainingSamples, newRemaining, std::memory_order_release);
	const int total = convo::consumeAtomic(m_fadeTotalSamples, std::memory_order_acquire);
	if (total > 0)
	{
		const double alpha = 1.0 - static_cast<double>(newRemaining) / static_cast<double>(total);
		convo::publishAtomic(m_fadeAlpha, alpha, std::memory_order_release);
	}
}

void SnapshotCoordinator::requestFadeCompletion() noexcept
{
	convo::publishAtomic(m_fadeCompleted, true, std::memory_order_release);
}

bool SnapshotCoordinator::tryCompleteFade() noexcept
{
	if (!convo::exchangeAtomic(m_fadeCompleted, false, std::memory_order_acq_rel))
		return false;

	if (convo::consumeAtomic(m_fadeState, std::memory_order_acquire) != FadeState::FadingIn ||
		convo::consumeAtomic(m_fadeRemainingSamples, std::memory_order_acquire) > 0)
		return false;

	completeFade();
	return true;
}

void SnapshotCoordinator::abortFade() noexcept
{
	GlobalSnapshot* target = convo::exchangeAtomic(m_target, nullptr, std::memory_order_acq_rel);
	if (target)
		SnapshotFactory::destroy(target);

	convo::publishAtomic(m_fadeState, FadeState::Idle, std::memory_order_release);
	convo::publishAtomic(m_fadeAlpha, 1.0, std::memory_order_release);
	convo::publishAtomic(m_fadeRemainingSamples, 0, std::memory_order_release);
	convo::publishAtomic(m_fadeCompleted, false, std::memory_order_release);
}

void SnapshotCoordinator::completeFade() noexcept
{
	GlobalSnapshot* target = convo::exchangeAtomic(m_target, nullptr, std::memory_order_acq_rel);
	if (!target)
		return;

	const uint64_t retireEpoch = m_epochCore.publish();
	GlobalSnapshot* old = convo::exchangeAtomic(m_current, target, std::memory_order_acq_rel);
	if (old)
	{
		m_deletionQueue.enqueue(
			old,
			[](void* p) { SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(p)); },
			retireEpoch,
			DeletionEntryType::Generic);
	}

	convo::publishAtomic(m_fadeState, FadeState::Idle, std::memory_order_release);
	convo::publishAtomic(m_fadeAlpha, 1.0, std::memory_order_release);
	convo::publishAtomic(m_fadeRemainingSamples, 0, std::memory_order_release);
	convo::publishAtomic(m_fadeCompleted, false, std::memory_order_release);
}

} // namespace convo
