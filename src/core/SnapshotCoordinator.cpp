//==============================================================================
// SnapshotCoordinator.cpp - Phase 4
//==============================================================================

#include "SnapshotCoordinator.h"

namespace convo {

void SnapshotCoordinator::startFade(const GlobalSnapshot* target, int fadeSamples) noexcept
{
	if (target == nullptr || fadeSamples <= 0)
	{
		switchImmediate(target);
		return;
	}

	// 初回適用で current が未初期化の場合は、
	// null 起点フェードを避けて即時反映する。
	if (m_current.load(std::memory_order_acquire) == nullptr)
	{
		switchImmediate(target);
		return;
	}

	const GlobalSnapshot* oldTarget = m_target.exchange(target, std::memory_order_acq_rel);
	if (oldTarget) {
		// Audio Thread が参照中の可能性があるため、即時 delete せず RCU 遅延解放
		const uint64_t retireEpoch = SnapshotEpoch::get();
		m_deletionQueue.enqueue(
			const_cast<GlobalSnapshot*>(oldTarget),
			[](void* p) { SnapshotFactory::destroy(static_cast<const GlobalSnapshot*>(p)); },
			retireEpoch,
			DeletionEntryType::Generic
		);
	}

	m_fadeTotalSamples.store(fadeSamples, std::memory_order_release);
	m_fadeRemainingSamples.store(fadeSamples, std::memory_order_release);
	m_fadeAlpha.store(0.0, std::memory_order_release);
	m_fadeCompleted.store(false, std::memory_order_release);
	m_fadeState.store(FadeState::FadingIn, std::memory_order_release);
}

void SnapshotCoordinator::advanceFade(int numSamples) noexcept
{
	if (m_fadeState.load(std::memory_order_acquire) != FadeState::FadingIn)
		return;

	const int remaining = m_fadeRemainingSamples.load(std::memory_order_acquire);
	if (remaining <= 0)
		return;

	const int newRemaining = remaining - numSamples;
	if (newRemaining <= 0)
	{
		m_fadeRemainingSamples.store(0, std::memory_order_release);
		requestFadeCompletion();
		return;
	}

	m_fadeRemainingSamples.store(newRemaining, std::memory_order_release);
	const int total = m_fadeTotalSamples.load(std::memory_order_acquire);
	if (total > 0)
	{
		const double alpha = 1.0 - static_cast<double>(newRemaining) / static_cast<double>(total);
		m_fadeAlpha.store(alpha, std::memory_order_release);
	}
}

void SnapshotCoordinator::requestFadeCompletion() noexcept
{
	m_fadeCompleted.store(true, std::memory_order_release);
}

bool SnapshotCoordinator::tryCompleteFade() noexcept
{
	if (!m_fadeCompleted.exchange(false, std::memory_order_acq_rel))
		return false;

	if (m_fadeState.load(std::memory_order_acquire) != FadeState::FadingIn ||
		m_fadeRemainingSamples.load(std::memory_order_acquire) > 0)
		return false;

	completeFade();
	return true;
}

void SnapshotCoordinator::abortFade() noexcept
{
	const GlobalSnapshot* target = m_target.exchange(nullptr, std::memory_order_acq_rel);
	if (target)
		SnapshotFactory::destroy(target);

	m_fadeState.store(FadeState::Idle, std::memory_order_release);
	m_fadeAlpha.store(1.0, std::memory_order_release);
	m_fadeRemainingSamples.store(0, std::memory_order_release);
	m_fadeCompleted.store(false, std::memory_order_release);
}

void SnapshotCoordinator::completeFade() noexcept
{
	const GlobalSnapshot* target = m_target.exchange(nullptr, std::memory_order_acq_rel);
	if (!target)
		return;

	const uint64_t retireEpoch = SnapshotEpoch::advance();
	const GlobalSnapshot* old = m_current.exchange(target, std::memory_order_acq_rel);
	if (old)
	{
		m_deletionQueue.enqueue(
			const_cast<GlobalSnapshot*>(old),
			[](void* p) { SnapshotFactory::destroy(static_cast<const GlobalSnapshot*>(p)); },
			retireEpoch,
			DeletionEntryType::Generic);
	}

	m_fadeState.store(FadeState::Idle, std::memory_order_release);
	m_fadeAlpha.store(1.0, std::memory_order_release);
	m_fadeRemainingSamples.store(0, std::memory_order_release);
	m_fadeCompleted.store(false, std::memory_order_release);
}

} // namespace convo
