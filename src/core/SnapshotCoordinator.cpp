//==============================================================================
// SnapshotCoordinator.cpp - Phase 4
//==============================================================================

#include "SnapshotCoordinator.h"
#include "../audioengine/ISRRetireLatency.h"   // ★ C2: 要求記録（RT 安全版は使わない）
#include "../DspNumericPolicy.h"              // ★ C2: isAudioThread() による Non-RT 契約の検査

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
// ★ STG-11-D2 (R-A): Q-full 時に E → T へ昇格し、retire ownership を終端まで保持する。
//   Q のみへの直接移送では、Q 満杯時に Release で silent loss になった。
//   E（EmergencyQuarantineStore）/ T（TerminalReclaimAuthority）は既存 public API を
//   そのまま使用する。ISRRetireRouter.* の意味変更なし、新規 authority なし。
//   全段は同一 engine domain / 同一 router で評価される（epoch provenance 維持）。
//   同一 ptr を複数 container に格納しない（各段は ownership transfer 成立でのみ return）。
void SnapshotCoordinator::quarantineRetireSink(void* ptr, void (*deleter)(void*),
                                               uint64_t epoch, const char* reason) noexcept
{
    // ★ 設計 v2: この関数は startFade / completeFade の失敗時からのみ呼ばれる。
    //   両経路は呼出し前に recordRequest 済みのため、ここでは記録を追加しない。
    //   将来この関数を直接呼ぶ経路を追加する場合は、呼出元で先に記録すること。
    if (m_retireSink == nullptr || ptr == nullptr || deleter == nullptr)
        return;
    // Stage 1: Q（RetireQuarantineStore）。格納成功で ownership 移転成立。
    if (m_retireSink->quarantineRetire(
            ptr, deleter, epoch, DeletionEntryType::Generic, reason))
        return;  // Q owns ptr
    // Stage 2: E（EmergencyQuarantineStore）。Q-full 時のみ到達。
    if (m_retireSink->emergencyQuarantine(
            ptr, deleter, epoch, DeletionEntryType::Generic, reason, 0, 0))
        return;  // E owns ptr
    // Stage 3: T（TerminalReclaimAuthority）。growable のため常に受領し、
    //   ownership は必ず移転する（戻り値は常に true が契約）。
    //   epoch safe かつ Non-RT（全呼び出し元 NonRT）なら即時破棄、
    //   epoch unsafe なら保持（drain() が epoch safe 時に解放）。
    //   ★ assert による検出に依存しない（Release でも終端解決する）。
    (void) m_retireSink->terminalReclaim(
        ptr, deleter, epoch, DeletionEntryType::Generic, reason);
    // T owns ptr
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
bool snapshotRequestCreated = false;
			const auto snapshotRequestId = convo::isr::RetireLatencyRecorder::instance().recordRequest (
				oldTarget, retireEpoch, m_epochProvider->getMinReaderEpoch(), 0xFFFFFFFFu, &snapshotRequestCreated);
			(void) snapshotRequestId;
			(void) snapshotRequestCreated;
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
	// ★ C2 (2026-11-10, Owner 承認): 本 API は Non-RT 専用。
	//   RT から呼ばれた場合は契約違反として検出し、状態を一切変更せずに返す。
	//   target / fade 状態 / 所有権 / 要求記録のいずれも触らない（後続の Non-RT 呼出しが
	//   同じ前提で通常どおり退役させる）。
	if (convo::numeric_policy::isAudioThread())
	{
		// ★ C2: 違反は全構成で記録する（assertion より前に加算する）。
		convo::fetchAddAtomic (SnapshotCoordinator::s_rtViolationCount, std::uint64_t{1}, std::memory_order_release);
		jassert (false && "SnapshotCoordinator::resetFadeStateAndRetireTarget is Non-RT only");
		return;
	}

	constexpr auto snapshotDeleter = [](void* ptr) noexcept
	{
		SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(ptr));
	};

	GlobalSnapshot* target = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
	if (target)
	{
		const uint64_t retireEpoch = m_epochProvider->publishEpoch();

		// ★ C2: 移送 1 回につき要求を 1 回だけ記録する。失敗しても取り消さない。
		//   格納できなかった場合は quarantineRetireSink が Q→E→T で必ず受領し、
		//   完了フックが同じ object で対応する。
		(void) convo::isr::RetireLatencyRecorder::instance().recordRequest (
			target, retireEpoch, m_epochProvider->getMinReaderEpoch(), 0xFFFFFFFFu);

		const bool resetStored = m_epochProvider->enqueueRetire(target, snapshotDeleter, retireEpoch);

		if (! resetStored)
			quarantineRetireSink(target, snapshotDeleter, retireEpoch, "resetFadeStateAndRetireTarget:queueFull");
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
bool snapshotRequestCreated = false;
		const auto snapshotRequestId = convo::isr::RetireLatencyRecorder::instance().recordRequest (
			old, retireEpoch, m_epochProvider->getMinReaderEpoch(), 0xFFFFFFFFu, &snapshotRequestCreated);
		(void) snapshotRequestId;
		(void) snapshotRequestCreated;
		const auto result = enqueueWithRetry(*m_epochProvider, old, snapshotDeleter, retireEpoch);
		if (!result) {
			// ★ BUG-015/027 (work88): 退避ストアへ移送（directDelete しない）
			quarantineRetireSink(old, snapshotDeleter, retireEpoch, "completeFade:queueFull");
		}
	}

	m_fade.resetToIdle();
}

} // namespace convo
