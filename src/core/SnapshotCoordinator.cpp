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
	if (m_slots.loadCurrent(std::memory_order_acquire) == nullptr) // acquire: switchImmediate/completeFade の release と HB し最新 current を観測
	{
		switchImmediate(target);
		return;
	}

	GlobalSnapshot* oldTarget = m_slots.exchangeTarget(target, std::memory_order_acq_rel); // acq_rel: acquire で旧 target の書き込みと HB; release で completeFade の acquire と HB し新 target を公開
	if (oldTarget) {
		// Audio Thread が参照中の可能性があるため、即時 delete せず RCU 遅延解放
		const uint64_t retireEpoch = m_epochDomain->current();
		m_retire.retire(oldTarget, retireEpoch);
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
	GlobalSnapshot* target = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel); // acq_rel: acquire で startFade の release と HB し旧 target 取得; release で次回 startFade の acquire と HB (null 公開)
	if (target)
	{
		const uint64_t retireEpoch = m_epochDomain->publish();
		m_retire.retire(target, retireEpoch);
	}

	m_fade.resetToIdle();
}

void SnapshotCoordinator::completeFade() noexcept
{
	GlobalSnapshot* target = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel); // acq_rel: acquire で startFade の release と HB し target 取得; release で次回 startFade の acquire と HB (null 公開)
	if (!target)
		return;

	const uint64_t retireEpoch = m_epochDomain->publish();
	GlobalSnapshot* old = m_slots.exchangeCurrent(target, std::memory_order_acq_rel); // acq_rel: acquire で旧 current への全書き込みと HB; release で observeCurrent/updateFade の acquire と HB し新 current を公開
	m_retire.retire(old, retireEpoch);

	m_fade.resetToIdle();
}

} // namespace convo
