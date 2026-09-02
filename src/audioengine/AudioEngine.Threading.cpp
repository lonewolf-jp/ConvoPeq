#include <JuceHeader.h>
#include <algorithm>
#include "AudioEngine.h"
#include "DiagnosticsConfig.h"             // ★ D162-1R-B: footprint 破壊側ログ
#include "ISRDSPQuarantine.h"
#include "RuntimeDrainAudit.h"
#include "RuntimePublicationOrchestrator.h"
#include "ISRRetireOverflowRing.h"         // ★ Phase1: RetireOverflowRing 完全型
#include "DSPLifetimeManager.h"            // ★ FUTURE-9: runCoordinatorPhase
#include "ISRCoordinatorLoop.h"            // ★ FUTURE-9: coordinatorLoop_ complete type

//==============================================================================
// [P0-15] AudioEngine.Threading.cpp — 3PR分割済み.
// 共通定数は AudioEngine.Retire.cpp に移動.
// 本ファイルには非3PR関数のみ残す.
//==============================================================================

void AudioEngine::destroyDSPCoreNode(void* p) noexcept
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ D117 root-cause audit: observation-only trace (no semantic change).
    juce::Logger::writeToLog(juce::String::formatted("[D117_DESTROY] dsp=%p", p));
    // ★ D162-1R-B: 破壊前の実測 footprint を DSPCore 保存値から出力
    //   （AC-R5: construct/retained/destroy を pointer identity で突合）。
    if (p != nullptr)
    {
        auto* coreDiag = static_cast<DSPCore*>(p);
        const auto& fp = coreDiag->diagFootprint;
        juce::Logger::writeToLog(juce::String::formatted(
            "[DSP_DESTROY_FOOTPRINT] dsp=%p gen=%llu trackedFootprint=%zu "
            "(convolver=%zu irData=%zu nuc=%zu ipp=%zu latency=%zu eq=%zu other=%zu)",
            p,
            (unsigned long long)coreDiag->diagGeneration.load(std::memory_order_relaxed),
            fp.total(), fp.convolver, fp.irData, fp.nuc, fp.ipp,
            fp.latency, fp.eq, fp.other));
    }
#endif
    auto* core = static_cast<DSPCore*>(p);
    core->~DSPCore();
    convo::aligned_free(core);
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ D162-1R-B: 破壊完了確認（保存 footprint 全カテゴリは DSPCore と運命を共にするため
    //   帰属対象 0 = released）。実メモリ還収は OS/allocator 依存（H4 判定は MEM_SNAP 側）。
    juce::Logger::writeToLog(juce::String::formatted(
        "[DSP_FOOTPRINT_RELEASED] dsp=%p remaining=0", p));
#endif
}

bool AudioEngine::shouldRejectRebuildAdmissionForPressure() const noexcept
{
    // 既存: retire queue pressure チェック
    if (convo::consumeAtomic(retirePressureAdmissionStrict_, std::memory_order_acquire))
        return true;

    // ★ S-2: HealthState Critical の場合も Rebuild を拒否
    if (m_healthMonitor.getHealthState() == convo::ISRHealthState::Critical)
        return true;

    return false;
}

// ★ A-1.6: 3系統の隔離を1トランザクションとして実行（1 truth + 2 projections）
bool AudioEngine::quarantineSlot(uint32_t slot, uint64_t generation,
                                  convo::isr::QuarantineReason reason) noexcept
{
    ASSERT_NON_RT_THREAD();

    // Step 1: Truth store 更新（唯一の隔離判定）
    const bool applied = dspQuarantineManager_.quarantineHandle(slot, generation, reason);

    // Step 2: Truth 確認（既に隔離済みの場合は何もしない）
    if (!applied)
        return false;

    // ★ B18: Step 2b — Projection 更新前に resolve() して retire に委譲する
    //    resolve() は registry_ (state==Active) を読む必要があるため、
    //    quarantineSlot() (state→Quarantined) より前に実行必須。
    const convo::isr::DSPHandle handle{slot, generation};
    const auto resolved = dspHandleRuntime_.resolve(handle);
    DSPCore* dsp = static_cast<DSPCore*>(resolved.instance);
    if (dsp != nullptr) {
        // ★ D162-2-B (C′): quarantine も DSPCore の到達不能脱落点だった（D162-2-A §1 #14）。
        //   意味論確定（orphan — 意図的 resident ownership ではない）:
        //   (a) resolve は Quarantined state を拒否するため、quarantine 後は誰も DSPCore に
        //       到達できない（RebuildDispatch RECOVERY-6 comment / ISRDSPHandle.cpp resolve）。
        //   (b) DSPQuarantineManager は metadata（slot/generation/reason の audit log）のみを
        //       保持し DSPCore* を保持しない（ISRDSPQuarantine.cpp:17-47）。
        //   (c) Recovery は buildSource snapshot から再 build し、quarantined DSPCore を
        //       参照しない（RECOVERY-SEMANTIC-001 / ProcessIntent.cpp:126-131）。
        //   (d) quarantine 解放（Commit.cpp destroyForShutdown / destroyQuarantineSlot）は
        //       slot 状態遷移のみで DSPCore を破壊しない。
        //   よって台帳解除のみの retireDSPHandleForRuntime 直呼び（旧実装・コメントの
        //   「EpochDomain 経由の deferred delete」は実体が無かった）から authority
        //   （DSPLifetimeManager::retire = 台帳解除 + EBR 破壊権取得）に統一する。
        //   EBR は epoch gate 付きのため、quarantine 直前まで RT 参照があった場合も
        //   reader grace 経過後にのみ破壊される（INV-D162-5 準拠）。
        DSPLifetimeManager lifetimeMgr(*this);
        lifetimeMgr.retire(dsp);  // EpochDomain 経由の deferred delete（EBR enqueue 含む）
    }

    // Step 3: Projection 更新（truth を反映）
    dspHandleRuntime_.quarantineSlot(slot);
    worldAuthority_.lifetime().quarantine(slot);

    return true;
}

// ★ A-2.5: collectDrainAudit — shutdown 完了条件の監査構造体を収集
convo::isr::RuntimeDrainAudit AudioEngine::collectDrainAudit() noexcept
{
    // ★ detectStuckReaders は1回だけ呼び出し、2つのフィールドで再利用（二重呼出の改善①）
    const auto readerStuckInfo = m_retireRouter
        ? m_retireRouter->detectStuckReaders(10)
        : convo::StuckReaderInfo{};

    return convo::isr::RuntimeDrainAudit{
        .pendingPublication = runtimePublicationBridge_.getPublicationBacklogCount(),
        .pendingRetire = worldAuthority_.lifetime().pendingIntentCount(),
        .activeCrossfadeCount = crossfadeRuntime_.isPending() ? 1u : 0u,
        .routerPendingRetire = static_cast<uint64_t>(m_retireRouter->pendingRetireCount())
            + convo::consumeAtomic(fallbackQueueDepth_, std::memory_order_acquire),  // ★ P1-9: ring+fallback 合計
        .maxDeferredAgeMs = runtimeOrchestrator_
            ? runtimeOrchestrator_->getMaxDeferredAgeMs() : 0u,
        .deferredPublish = (runtimeOrchestrator_
            && runtimeOrchestrator_->hasDeferredRequest()) ? 1u : 0u,
        .quarantineResident = dspQuarantineManager_.residentCount(),
        .oldestPendingAgeMs = static_cast<uint64_t>(
            std::max(0.0, convo::consumeAtomic(oldestPendingAge_, std::memory_order_acquire))),
        .maxQuarantineAgeSec = dspQuarantineManager_.getMaxEntryAgeSec(),
        // ★ C-1: WorldLifecycleAudit から World カウンタ取得
        .activeWorldCount = worldLifecycleAudit_.activeWorldCount(),
        .publishedCount = worldLifecycleAudit_.publishedCount(),
        .retiredCount = worldLifecycleAudit_.retiredCount(),
        // ★ A-2/A-3: Reader 状態収集（detectStuckReaders は1回のみ）
        .activeReaderCount = m_retireRouter ? m_retireRouter->activeReaderCount() : 0u,
        .stuckReaderCount = readerStuckInfo.isStuck ? 1u : 0u,
        .maxReaderResidencyUs = readerStuckInfo.residencyTimeUs,
        // ★ B-2: HealthState 診断情報
        .healthState = m_healthMonitor.getHealthState(),
        // ★ A-2: EBR Queue Visibility 統計
        .reclaimAttemptCount = m_retireRouter
            ? m_retireRouter->reclaimAttemptCount() : 0,
        .reclaimSuccessCount = m_retireRouter
            ? m_retireRouter->reclaimSuccessCount() : 0,
        .overflowCount = m_retireRouter
            ? m_retireRouter->overflowCount() : 0,
        // ★ Phase2: OverflowRing 滞留数
        .overflowRingResident = worldAuthority_.lifetime().getOverflowRing()
            ? worldAuthority_.lifetime().getOverflowRing()->residentCount() : 0
    };
}

bool AudioEngine::isFullyDrained() noexcept
{
    const bool hasDeferredCommit = (runtimeOrchestrator_ != nullptr && runtimeOrchestrator_->hasDeferredRequest());
    // ★ work88 (P2-1 §1.1.5): setPendingIntentCount / setPublicationBacklogCount の絶対値上書きは廃止。
    //   pendingIntentCount_ は reservation ベース（push 成功 fetchAdd / pop 成功 fetchSub）で
    //   正確に維持され、ここでの hasDeferredCommit 混入は RetireIntent を誤って pending に
    //   計上していた（混入の温床）。publicationBacklogCount_ も同様に Coordinator 内部で
    //   実測維持される。
    //   戻り値の !hasDeferredCommit は維持（deferred commit が残っている間は drain 完了としない）。

    // ★ dash2 §1.4 (B0-4): external setter（setFallbackBacklogCount / setRetireBacklogCount /
    //   setDeferredRetireResidencyCount）による Coordinator への絶対値上書きを廃止。
    //   retire バックログは router の実測値（pendingRetireCount）を Layer 1 で直接判定する。
    //   RetireIntent 滞留（lifetime().pendingIntentCount() — Commit が emit する RetireIntent）も
    //   実測で直接判定する（旧 setRetireBacklogCount スナップショットが担っていた契約を Layer 1 で維持）。
    //   fallback / deferred retire は Layer 2（Coordinator）の queue emptiness + 内部
    //   semantic カウンタが担当（dash2 §1.4 設計方針 — isFullyDrained は実測値を直接判定）。
    const std::uint64_t retireDepth = (m_retireRouter != nullptr)
        ? static_cast<std::uint64_t>(m_retireRouter->pendingRetireCount()) : 0u;
    const std::uint64_t lifetimeRetireIntentPending = worldAuthority_.lifetime().pendingIntentCount();

    // ★ work88 (X6 §6.6): quarantineResidentCount_ の aggregate 上書き（ringResident + dspQuarantine）
    //   は廃止（INV-X6-4 — 混在禁止）。実在 quarantine DSP 数は DSPQuarantineManager::residentCount()
    //   が唯一の source of truth。overflow ring（retire 系）・DSPQuarantine・RetireQuarantineStore の
    //   3 semantic を個別に直接判定する（いずれも shutdown では waitForDrain 前に drain 済み）。
    const auto ringResident = worldAuthority_.lifetime().getOverflowRing()
        ? worldAuthority_.lifetime().getOverflowRing()->residentCount() : size_t{0};
    const auto dspQuarantineResident = dspQuarantineManager_.residentCount();
    const auto retireQuarantineResident = (m_retireRouter != nullptr)
        ? static_cast<std::uint64_t>(m_retireRouter->quarantineResidentCount()) : 0u;

    // ★ 15-P-5: TerminalReclaimAuthority 滞留も直接判定する（P-4 追加の最終退避層）。
    //   quarantineResidentCount() は Q + EmergencyQ のみで Terminal を含まないため、
    //   ここを追加しないと「Terminal に World が残っているのに isFullyDrained()==true」と
    //   誤判定し、waitForDrain が premature に成功を返す（shutdown 中の World リーク経路）。
    //   Terminal は epoch-gated drain（drainTerminalReclaim）と強制 drain（drainAll）の
    //   両方で空になるため、この判定は shutdown 完了の正しい必要条件である。
    const auto terminalReclaimResident = (m_retireRouter != nullptr)
        ? static_cast<std::uint64_t>(m_retireRouter->terminalReclaimResidentCount()) : 0u;

    // ★ work88 (X3 §6.3 / INV-X3-5): pendingReclaimHandles_ が reclaim pending の source of truth。
    //   reclaimInFlightCount_（+1/0 リセットの近似 counter）だけでは複数 pending reclaim を正確に
    //   数えられない（A pending + B pending で A 成功 → count=0 なのに B が残る状態を作り得る）。
    //   したがって pendingReclaimHandles_.empty() を追加する（二十六次レビュー必須修正2 / INV-X3-5）。
    //   評価は waitForDrain 時点（reclaim producer join 済み）のため、観測直後の push はない（AC-X3-14）。
    bool pendingReclaimEmpty = false;
    {
        std::lock_guard<std::mutex> lock(pendingReclaimHandlesMutex_);
        pendingReclaimEmpty = pendingReclaimHandles_.empty();
    }

    return !hasDeferredCommit
        && pendingReclaimEmpty
        && retireDepth == 0
        && lifetimeRetireIntentPending == 0
        && ringResident == 0
        && dspQuarantineResident == 0
        && retireQuarantineResident == 0
        && terminalReclaimResident == 0
        && runtimePublicationBridge_.isFullyDrained();
}

bool AudioEngine::waitForDrain(int timeoutMs, int pollIntervalMs) noexcept
{
    ASSERT_NON_RT_THREAD();
    // ★ P1-4: waitForDrain は AudioStopped 以降でのみ呼ばれる。
    //   新しい ShutdownPhase が追加された場合はここに追加すること。
    [[maybe_unused]] const auto phase = shutdownRuntime_.getPhase();
    jassert(phase == convo::isr::ShutdownPhase::AudioStopped
         || phase == convo::isr::ShutdownPhase::ObserverDrained
         || phase == convo::isr::ShutdownPhase::RetireClosed
         || phase == convo::isr::ShutdownPhase::EpochSettled
         || phase == convo::isr::ShutdownPhase::ReclaimComplete
         || phase == convo::isr::ShutdownPhase::EmergencyDrain     // ★ C-2
         || phase == convo::isr::ShutdownPhase::TimedOut
         || phase == convo::isr::ShutdownPhase::Failed
         || phase == convo::isr::ShutdownPhase::ShutdownComplete);

    const int boundedTimeoutMs = juce::jlimit(1, 10000, timeoutMs);
    const int boundedPollIntervalMs = juce::jlimit(1, 5, pollIntervalMs);

    const double startMs = juce::Time::getMillisecondCounterHiRes();
    while (!isFullyDrained())
    {
        drainDeferredRetireQueues(true);

        const double elapsedMs = juce::Time::getMillisecondCounterHiRes() - startMs;
        if (elapsedMs >= static_cast<double>(boundedTimeoutMs))
            return false;

        juce::Thread::sleep(boundedPollIntervalMs);
    }

    return true;
}

void AudioEngine::processDeferredReleases()
{
    drainDeferredRetireQueues(false);
}

// ★ E-1.9-B: Event-driven drain wake with fallback timeout.
//   Delegates to ISRRetireRouter's CV wait. The predicate is the E-1.9-A atomic
//   counters (pendingRetireCount / residentCountAtomic) — Semantic Single Source,
//   no drainSignaled_ state. Non-RT only (CoordinatorLoop context).
void AudioEngine::waitForDrainSignalOrTimeout(int timeoutMs) noexcept
{
    if (m_retireRouter != nullptr)
        m_retireRouter->waitForDrainSignalOrTimeout(timeoutMs);
    else
        juce::Thread::sleep(timeoutMs);  // fallback if router not initialized
}

//==============================================================================
// ★ FUTURE-9: Dedicated Coordinator Worker — Scheduling Authority lifecycle.
//   The periodic Coordinator cadence (processIntent / overflow drain / deferred
//   resubmit) is relocated here from AudioEngine::timerCallback so the
//   MessageThread Timer becomes observe-only (submitObserve). The worker is a
//   plain juce::Thread (NonRT); the ISR invariants below are unchanged:
//     • runtimePublicationBridge_.processIntent / drainOverflowRing operate on
//       lock-free queues + atomic counters (safe off-MessageThread).
//     • Deferred resubmit is MessageManager-free: runtimeOrchestrator_
    //       submitPublishRequest / processDeferredAdmission (atomic hasDeferred_).
//==============================================================================
void AudioEngine::startCoordinatorLoop() noexcept
{
    jassert(coordinatorLoop_ == nullptr);
    coordinatorLoop_ = std::make_unique<convo::isr::CoordinatorLoop>(*this);
    coordinatorLoop_->startLoop();
}

void AudioEngine::shutdownCoordinatorLoop() noexcept
{
    if (coordinatorLoop_)
    {
        coordinatorLoop_->stopLoop();
        coordinatorLoop_.reset();
    }
}

void AudioEngine::runCoordinatorPhase() noexcept
{
    // ★ ISR: Coordinator — processIntent (relocated from AudioEngine::timerCallback)
    {
        DSPLifetimeManager lifetimeMgr(*this);
        runtimePublicationBridge_.processIntent(*this, lifetimeMgr);
    }

    // ★ D152 T2 (D152-R1): adjudicate recovery failure signals — CoordinatorLoop-only consumer.
    //   Runs AFTER processIntent (obligations admitted/terminalized this tick are visible) and
    //   BEFORE redrive, so a failure→None transition becomes redrive-eligible in the SAME tick
    //   (D148 §1-D placement). Adjudication commits only via full-word lifecycle CAS; exhausted
    //   telemetry and liveCount −1 are CAS-winner-gated. CL-only placement is a determinism
    //   convention — no thread-id assert (D152 §8; TEST-ONLY wrapper runs elsewhere).
    runtimePublicationBridge_.adjudicateRecoveryFailureSignals();

    // ★ D105-R5-10: re-drive any deferred Live recovery obligations each Coordinator tick (option C).
    //   Runs on the CoordinatorLoop (producer) thread — SPSC-safe. Complements the opportunistic trigger
    //   in submitRecoveryRequest: ensures deferred obligations are redispatched once a delivery resource
    //   frees, even when no new recovery submissions occur.
    runtimePublicationBridge_.redriveDeferredRecoveryObligations();

    // ★ G-4.4-P2 (D137): redrive attach → Builder wake. If any deferred obligation re-acquired a
    //   delivery representation (None → Transport/Durable) — including the opportunistic redrive run
    //   inside processIntent's submitRecoveryRequest this same phase — raise the EXISTING
    //   recoveryPending predicate + notify, exactly the submitRecoveryIntent wake protocol
    //   (AudioEngine.h). Event-driven: zero wakes when nothing attached (no per-tick notify —
    //   preserves F6-5). The latch is CoordinatorLoop-only (same thread set/consume — no new
    //   cross-thread ordering; recoveryPending remains rebuildMutex-guarded).
    if (runtimePublicationBridge_.consumeRedriveWake())
    {
        {
            std::lock_guard<std::mutex> lock(rebuildMutex);
            recoveryPending = true;
        }
        rebuildCV.notify_all();
    }

    // [PR-3] Deferred publish resubmit — Coordinator は Decision/Routing のみに徹する。
    //   ★ ISR Builder/Coordinator 分離: Coordinator は world build / publish を実行しない。
    //     deferred がある場合、publishRetryReady フラグを立てて RebuildThread を起床させるだけ。
    //     RebuildThread が processDeferredAdmission() を実行する（peek → evaluate → consume/discard → finishView → submitPublishRequest）
    //     （Builder 責務は RebuildThread に一元化）。
    //   ★ ビジーループ防止: predicate に hasDeferredRequest() を直接入れず、フラグ駆動にする。
    //     Deferred が継続しても RebuildThread は休眠し、Coordinator が次 1ms tick で再通知するまで
    //     再試行しない。これにより crossfade 終了まで CPU スピンすることはない。
    // ★ F6-5: watchdog 化 — 現行は毎 tick publishRetryReady を立てていた（= 実質 1ms polling）。
    //   F6 では fade-complete wake（P3, Timer.cpp）が正常経路となり、本 poll は lost-wake 自己修復
    //   専用として kDeferredWakeWatchdogTicks ごとにのみ発火する。publishRetryReady の意味・
    //   rebuildMutex→flag→unlock→notify の順序・hasDeferred_ を predicate にしない方針は不変。
    if (!isShutdownInProgress()
        && runtimeOrchestrator_ != nullptr
        && runtimeOrchestrator_->hasDeferredRequest())
    {
        if (++coordinatorDeferredWatchdogTicks_ >= kDeferredWakeWatchdogTicks)
        {
            coordinatorDeferredWatchdogTicks_ = 0;
            {
                std::lock_guard<std::mutex> lock(rebuildMutex);
                publishRetryReady = true;
            }
            rebuildCV.notify_one();
        }
    }
    else
    {
        coordinatorDeferredWatchdogTicks_ = 0;
    }

    // ★ Phase1: OverflowRing drain (relocated from timerCallback).
    {
        if (worldAuthority_.lifetime().getOverflowRing())
        {
            const auto drainResult = runtimePublicationBridge_.drainOverflowRing(
                *worldAuthority_.lifetime().getOverflowRing(), worldAuthority_.lifetime(), false);
            if (drainResult.reinjectedCount > 0)
                m_retireRouter->tryReclaim();
        }
    }

    // ★ E-1.9-B Phase2: Deferred retire drain (Q/E/T).
    //   Event-driven: CoordinatorLoop wakes via drainCv_ when Q/E/T receives entries
    //   (signalDrainWakeup() from enqueueWithRetry). The 1ms timeout fallback in
    //   waitForDrainSignalOrTimeout ensures periodic polling even without signals.
    //   E-1.9-A empty-guard inside drainDeferredRetireQueues(false) prevents
    //   wasted work on spurious wakes. Non-shutdown only (allowDuringShutdown=false).
    //   ★ B-I5: Inserted at END of runCoordinatorPhase, AFTER all existing phases —
    //   preserves existing phase ordering (processIntent → deferred resubmit → overflow drain).
    //   Q/E/T drain is the final step, ensuring epoch advances from overflow drain
    //   are visible before draining retirement stores.
    drainDeferredRetireQueues(false);
}
