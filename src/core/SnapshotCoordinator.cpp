//==============================================================================
// SnapshotCoordinator.cpp - Phase 4
//==============================================================================

#include "SnapshotCoordinator.h"

#include <cassert>  // ★ BUG-015/027 (work88): store full 検出用

#include "audioengine/AtomicAccess.h"
#include "audioengine/ISRAuthorityClass.h"
#include "audioengine/ISRRetireRouter.h"  // ★ BUG-015/027 (work88): quarantineRetire 経由の退避移送
#include "SnapshotFactory.h"

namespace convo {

// ★ BUG-015/027 (work88): enqueueWithRetry 失敗時の退避移送（Category A — SnapshotCoordinator）。
//   五次レビュー §5: SnapshotCoordinator は退避ストアを直接保持しない。
//   Router API（quarantineRetire）経由で Router 内部の RetireQuarantineStore へ移送する。
//   directDelete は禁止（RT 参照中の UAF 排除）。store full 時も deleter を実行しない
//   （capacity exhaustion は health escalation で先行検知 — ここでは jassert で異常検出）。
void SnapshotCoordinator::quarantineRetireSink(void* ptr, void (*deleter)(void*),
                                               uint64_t epoch, const char* reason) noexcept
{
    if (m_retireSink == nullptr || ptr == nullptr || deleter == nullptr)
        return;
    const bool stored = m_retireSink->quarantineRetire(
        ptr, deleter, epoch, DeletionEntryType::Generic, reason);
    if (!stored)
        assert(false && "RetireQuarantineStore capacity exhaustion - EBR 破綻の可能性");
}

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
		const auto result = enqueueWithRetry(*m_epochProvider, oldTarget, snapshotDeleter, retireEpoch);
		if (!result) {
			// ★ BUG-015/027 (work88): 退避ストアへ移送（directDelete しない）
			quarantineRetireSink(oldTarget, snapshotDeleter, retireEpoch, "startFade:queueFull");
		}
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
	{
		// [work37 Phase 1.2] enqueueWithRetry を使用（completeFade は NonRT）
		const auto result = enqueueWithRetry(*m_epochProvider, old, snapshotDeleter, retireEpoch);
		if (!result) {
			// ★ BUG-015/027 (work88): 退避ストアへ移送（directDelete しない）
			quarantineRetireSink(old, snapshotDeleter, retireEpoch, "completeFade:queueFull");
		}
	}

	m_fade.resetToIdle();
}

} // namespace convo
