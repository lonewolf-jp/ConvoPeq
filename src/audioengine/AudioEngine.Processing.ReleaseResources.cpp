#include <JuceHeader.h>
#include "AudioEngine.h"
#include "core/RuntimeReaderContext.h"
#include "DSPLifetimeManager.h"
#include "RuntimeBuilder.h"
#include "NoiseShaperLearner.h"
#include "RuntimePublicationOrchestrator.h"  // ★ work37: clearDeferredForShutdown 完全型必要
#include "ISRRetireOverflowRing.h"           // ★ Phase2: RetireOverflowEntry 完全型

namespace {

#include <debugapi.h>

static juce::String captureCallStack()
{
    void* stack[32];
    WORD frames = CaptureStackBackTrace(1, 32, stack, nullptr);
    juce::String result;
    for (WORD i = 0; i < frames; ++i)
    {
        if (i > 0) result += "\n";
        result += juce::String::toHexString(reinterpret_cast<uintptr_t>(stack[i]));
    }
    return result;
}

void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

void AudioEngine::releaseResources()
{
    ASSERT_NON_RT_THREAD();
    diagLog("[DIAG] releaseResources: enter");

    auto previousState = convo::consumeAtomic(lifecycleState, std::memory_order_acquire);
    for (;;)
    {
        if (previousState == EngineLifecycleState::Destroyed)
        {
            diagLog("[DIAG] releaseResources: ignored in Destroyed state");
            return;
        }

        if (previousState == EngineLifecycleState::Unprepared)
        {
            auto cs = captureCallStack();
            diagLog("[DIAG] releaseResources: duplicate release ignored (already Unprepared)\n"
                    "Callstack:\n" + cs);
            return;
        }

        if (previousState == EngineLifecycleState::Releasing)
        {
            diagLog("[DIAG] releaseResources: already Releasing");
            return;
        }

        if (convo::compareExchangeAtomic(lifecycleState,
                         previousState,
                         EngineLifecycleState::Releasing,
                         std::memory_order_acq_rel,
                         std::memory_order_acquire))
            break;
    }

    // P0-A0: LifecycleIsolationRuntime integration - enter release phase
    auto lifecycleToken = lifecycleRuntime_.enterRelease();

    setShutdownPhase(ShutdownPhase::StopAcceptingWork, "releaseResources");
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::AudioStopped);
    runtimePublicationBridge_.requestShutdown();

    // ★ [work70 v9.11] MMCSS シャットダウン: フラグ経由で Audio Thread に委譲。
    //    Message Thread はフラグのみセットし、実際の AvRevert は次回コールバックで実行される。
    //    NativeRT モード（useMmcssPriority=false）の復元は finalizeMmcssShutdown() で行う。
    {
        const auto mmcssPolicy = getCurrentMmcssPolicy();
        if (mmcssPolicy == MmcssPolicy::SelfManagedProAudio
            || mmcssPolicy == MmcssPolicy::SelfManagedPlayback)
        {
            convo::publishAtomic(mmcssShutdownRequested, true, std::memory_order_release);
        }
        // JuceManaged / None → JUCE manages shutdown. NativeRT は finalizeMmcssShutdown で復元。
    }
    finalizeMmcssShutdown();

    // 非MT起点の pending rebuild 要求と AsyncUpdater キューをシャットダウン直後に廃棄する。
    // stopRebuildThread より先に実行して handleAsyncUpdate が後から rebuild を発火しないようにする。
    clearRebuildReason(RebuildReason::StructuralFromNonMT);
    clearRebuildReason(RebuildReason::DeferredStructural);
    clearRebuildReason(RebuildReason::DeferredFinalizeAware);
    convo::publishAtomic(deferredFinalizeFirstSeenTicks_, 0, std::memory_order_release);
    cancelPendingUpdate();
    crossfadeRuntime_.reset();
    convo::publishAtomic(latencyResetPending, false, std::memory_order_release);
    convo::publishAtomic(lastIssuedConvolverStructuralHash_, 0, std::memory_order_release);
    convo::publishAtomic(lastCommittedConvolverStructuralHash_, 0, std::memory_order_release);
    convo::publishAtomic(lastCommittedConvolverHasIr_, false, std::memory_order_release);
    convo::publishAtomic(currentSampleRate, 0.0, std::memory_order_release);

    convo::publishAtomic(inputLevelLinear, 0.0f, std::memory_order_release);
    convo::publishAtomic(outputLevelLinear, 0.0f, std::memory_order_release);

    if (noiseShaperLearner)
    {
        juce::Logger::writeToLog("[AudioEngine] releaseResources: stopping learner");
        noiseShaperLearner->stopLearning();
    }

    resetLearningControlState();
    setShutdownPhase(ShutdownPhase::StopAudio, "releaseResources");

    DSPCore* activeToRelease = nullptr;
    DSPCore* fadingToRelease = nullptr;
    DSPCore* pendingNewToRelease = nullptr;
    DSPCore* pendingCurrentToRelease = nullptr;

    // ★ [PR-A2] DSPLifetimeManager 経由で retire (lifetime は lock 外でも参照可能にする)
    DSPLifetimeManager lifetimeForShutdown(*this);

    {
        std::lock_guard<std::mutex> lk(rebuildMutex);
        const convo::RuntimeReaderContext messageCtx{ messageThreadRcuReader, convo::ObserveChannel::Message };
        const auto runtimeReadHandle = makeRuntimeReadHandle(messageCtx);
        validateDistinctRuntimeSlots("releaseResources.beforeClear",
                 getActiveRuntimeDSP(),
                 resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                         nullptr);

        convo::fetchAddAtomic(rebuildRequestGeneration, 1, std::memory_order_acq_rel);
        {
            // ★ BUG-051: sentinel (uintptr_t)-1 は書き込まれない（常に nullptr or 有効 ptr）。
            //   死んだ再解釈チェックを除去。
            activeToRelease = getActiveRuntimeDSP();
        }
        setActiveRuntimeDSP(nullptr);

        {
            // ★ B-1: CAS-based fading slot clear
            DSPCore* current = convo::consumeAtomic(fadingRuntimeDSPSlot, std::memory_order_acquire);
            if (current != nullptr
                && convo::compareExchangeAtomic(fadingRuntimeDSPSlot, current,
                                                 static_cast<DSPCore*>(nullptr),
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire))
                fadingToRelease = current;
            else
                fadingToRelease = nullptr;
        }
        crossfadeRuntime_.reset();
        refreshCrossfadePreparedSnapshotFromAtomics();

        if (hasPendingTask)
        {
            // ★ BUG-051: sentinel (uintptr_t)-1 は書き込まれない（常に nullptr or 有効 ptr）。
            pendingCurrentToRelease = pendingTask.currentDSP;
            pendingTask.currentDSP = nullptr;
            hasPendingTask = false;
            publishRetryReady = false;
        }

        // Migrated to publishWorld() with pre-built RuntimePublishWorld (Sprint-2 P1-A)
        {
            auto worldBuilder = convo::RuntimeBuilder(*this);
            auto worldOwner = worldBuilder.buildRuntimePublishWorld(nullptr,
                                                                     nullptr,
                                                                     convo::TransitionPolicy::HardReset,
                                                                     0.0,
                                                                     false);
            // ★ B4: idle publish (#4) — 登録なし・oldHandle は null 固定
            const auto pubResult = commitRuntimePublication(std::move(worldOwner),
                                     RegistrationContext::none(),
                                     convo::isr::DSPHandle::null());
            juce::ignoreUnused(pubResult);
        }

        validateDistinctRuntimeSlots("releaseResources.afterClear",
                 getActiveRuntimeDSP(),
                 resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                         nullptr);
    }

    diagLog("[DIAG] releaseResources: before stopRebuildThread");
    setShutdownPhase(ShutdownPhase::StopWorkers, "releaseResources");
    shutdownCoordinatorLoop();  // ★ FUTURE-9: join Coordinator Worker before drains
    stopRebuildThread();

    // ★★★ Phase 9-A: Q1/Q7 Admission Closure (D101-8 Step 8R code gap fix) ★★★
    // Producers are joined (Q2), so no new work can be generated.
    // closeAdmission(): Open→Closing (satisfies Q7 NoResurrection via !isAdmissionOpen()).
    // joinProducers(): Closing→Closed (satisfies Q1 AdmissionClosed — requires state==Closing).
    //   NOTE: Thread joins above do NOT update admissionState_ — both calls are required.
    //   closeAdmission() advances shutdownGeneration_ (identity binding for ReclaimPermit).
    shutdownRuntime_.closeAdmission();
    shutdownRuntime_.joinProducers();

    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::ObserverDrained);
    diagLog("[DIAG] releaseResources: after stopRebuildThread");

    setShutdownPhase(ShutdownPhase::ForceEpochAdvance, "releaseResources");
    advanceRetireEpoch();
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::RetireClosed);
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::EpochSettled);

    setShutdownPhase(ShutdownPhase::DrainRetire, "releaseResources");

    // ★ work88 (X3 §6.3 / INV-X3-4 / INV-ISR-04): CloseReaderRegistration — graceful drain の前に
    //   reader 新規登録を永久に封じる。graceful drain は activeReaderCount()==0 を待つが、
    //   reader 登録を封じないと「0 に達した後に新しい reader が登録される」可能性がある。
    //   先に readerRegistrationClosed を確立すれば、graceful drain 完了時点で再登録が構造的に不可能。
    //   （登録済み slot の enter/exit は継続可能 — 既存 Reader の epoch 安全性は維持）
    m_epochDomain.closeReaderRegistration();

    // ★ Phase5: Shutdown 時、全保留Intentを Critical に昇格（優先度ベースの早期回収）
    worldAuthority_.lifetime().escalateAllRetires(convo::isr::RetirePriority::Critical);

    // ★ Practical-7: Graceful Drain Phase（最大5秒間のポーリング待機 + OverflowRing 再注入）
    {
        constexpr int kGracefulDrainMaxMs = 5000;
        constexpr int kGracefulDrainPollMs = 10;
        constexpr uint32_t kMaxReinjectPerCycle = 128;  // ★ Phase2: 1ループ当たりの再注入上限
        int waitedMs = 0;
        while (waitedMs < kGracefulDrainMaxMs)
        {
            if (m_retireRouter->pendingRetireCount() == 0
                && m_retireRouter->activeReaderCount() == 0)
                break;

            // ★ Phase2: OverflowRing 再注入（DSPQuarantine エントリを最後まで再注入機会あり）
            {
                uint32_t reinjectBudget = kMaxReinjectPerCycle;
                convo::isr::RetireOverflowEntry entry;
                while (reinjectBudget > 0 && worldAuthority_.lifetime().getOverflowRing()
                       && worldAuthority_.lifetime().getOverflowRing()->pop(entry))
                {
                    worldAuthority_.lifetime().emitRetireIntent(entry.intent);
                    --reinjectBudget;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(kGracefulDrainPollMs));
            waitedMs += kGracefulDrainPollMs;
            m_retireRouter->publishEpoch();
            m_retireRouter->tryReclaim();

            // ★ work88 (X6 §6.6): coordinator の quarantineResidentCount_ aggregate 上書きは廃止
            //   （INV-X6-4 — ring と DSP を混ぜない）。drain 判定は AudioEngine::isFullyDrained が
            //   DSPQuarantineManager / overflowRing / RetireQuarantineStore を直接判定する（X6）。
        }

        // ★ Phase2 5.5: Timeout到達 → 最終Drain（1回限定）
        if (waitedMs >= kGracefulDrainMaxMs)
        {
            diagLog("[AUDIT] releaseResources: graceful drain timeout after "
                + juce::String(kGracefulDrainMaxMs)
                + "ms, pendingRetireCount="
                + juce::String(static_cast<int>(m_retireRouter->pendingRetireCount()))
                + " -- performing final drain");

            // a. ForceEpochAdvance
            m_retireRouter->publishEpoch();

            // b. OverflowRing 全件Drain（unlimited）
            if (worldAuthority_.lifetime().getOverflowRing())
            {
                convo::isr::RetireOverflowEntry entry;
                while (worldAuthority_.lifetime().getOverflowRing()->pop(entry))
                {
                    worldAuthority_.lifetime().emitRetireIntent(entry.intent);
                }
            }

            // c. 最終Reclaim
            m_retireRouter->tryReclaim();

            // d. 最終DeferredDrain
            drainDeferredRetireQueues(false);

            if (m_retireRouter->pendingRetireCount() == 0
                && m_retireRouter->activeReaderCount() == 0)
            {
                diagLog("[AUDIT] releaseResources: final drain succeeded");
            }
            else
            {
                diagLog("[AUDIT] releaseResources: final drain incomplete -- pendingRetire="
                    + juce::String(static_cast<int>(m_retireRouter->pendingRetireCount()))
                    + " activeReaders="
                    + juce::String(static_cast<int>(m_retireRouter->activeReaderCount())));
            }
        }
        else
        {
            // ★ Phase2: タイムアウト前に完了した場合も coordinator カウントを最終更新
            // ★ dash2 §1.4 (B0-6): setQuarantineResidentCount（ringResident → quarantine カウンタへの
            //   domain mixing）を撤去。overflow ring resident（retire 系）を quarantine カウンタに
            //   混ぜるのは INV-X6-4 違反。実在 quarantine DSP 数は Layer 1（AudioEngine::isFullyDrained）
            //   が DSPQuarantineManager::residentCount() を直接判定する（dash2 §1.4 設計方針）。
        }
    }

    // [P1 Phase1-B] drainPublicationLogForShutdown removed

    if (activeToRelease)
        lifetimeForShutdown.retire(activeToRelease);
    if (fadingToRelease)
        lifetimeForShutdown.retire(fadingToRelease);
    if (pendingNewToRelease)
        lifetimeForShutdown.retire(pendingNewToRelease);
    if (pendingCurrentToRelease)
        lifetimeForShutdown.retire(pendingCurrentToRelease);

    // shutdown/release シーケンスでは明示的に deferred retire queue をドレインする。
    // 通常タイマー経路は Releasing 中に early-return するため、ここで最終回収を保証する。
    drainDeferredRetireQueues(true);
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::ReclaimComplete);

    // ★ C-2: EmergencyDrain -- Optional 最終手段（デフォルトはスキップ）
    //   常に EmergencyDrain フェーズを経由（ReclaimComplete+1=EmergencyDrain のため単一遷移）
    //   [work37 Phase 8.2] コンパイル時マクロから実行時判定に変更。
    //   PolicyEngine が requestEmergencyDrain() を設定した場合のみ有効な処理を実行。
    //   Reader slot の epoch/depth 強制書き換えは一切禁止。
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::EmergencyDrain);
    if (m_healthMonitor.isEmergencyDrainRequested())
    {
        diagLog("[DIAG] releaseResources: EmergencyDrain phase enter (runtime)");

        [[maybe_unused]] constexpr int kEmergencyDrainMaxMs = 500;
        const auto emergencyStartMs = juce::Time::getMillisecondCounterHiRes();

        // Deferred publish クリア
        if (runtimeOrchestrator_)
            runtimeOrchestrator_->clearDeferredForShutdown();
        diagLog("[DIAG] releaseResources: EmergencyDrain -- cleared deferred publish");

        // 安全な tryReclaim（drainAll 禁止）
        {
            const auto preReclaimPending = m_retireRouter->pendingRetireCount();
            m_epochDomain.tryReclaim();
            const auto postReclaimPending = m_retireRouter->pendingRetireCount();
            diagLog("[DIAG] releaseResources: EmergencyDrain -- tryReclaim done (pending "
                + juce::String(static_cast<int>(preReclaimPending)) + " → "
                + juce::String(static_cast<int>(postReclaimPending)) + ")");
        }

        // Crossfade timeout recovery の強制実行
        if (crossfadeRuntime_.isPending())
        {
            diagLog("[DIAG] releaseResources: EmergencyDrain -- forcing crossfade recovery");
            crossfadeRuntime_.reset();
        }

        const auto emergencyElapsedMs = juce::Time::getMillisecondCounterHiRes() - emergencyStartMs;
        diagLog("[DIAG] releaseResources: EmergencyDrain phase completed in "
            + juce::String(emergencyElapsedMs, 1) + "ms");
        emitEvidenceTickNonRt(true);
    }
    else
    {
        // DiagnosticMode: evidence 出力のみ（EmergencyDrain 未要求時）
        const auto audit = collectDrainAudit();
        if (!audit.isAllZero() || audit.stuckReaderCount > 0)
        {
            diagLog("[DIAG] releaseResources: EmergencyDrain (diagnostic only) -- "
                "pendingPub=" + juce::String(static_cast<int64>(audit.pendingPublication)) +
                " pendingRetire=" + juce::String(static_cast<int64>(audit.pendingRetire)) +
                " stuckReaders=" + juce::String(static_cast<int64>(audit.stuckReaderCount)));
        }
    }

    // ★★★ PR2: Quarantine 全スロット強制解放（シャットダウン専用）
    //    この時点で GracefulDrain が activeReaderCount==0 を確認済み
    {
        // ★ Phase 3: EpochDomain の Reader quarantine を全解除
        m_retireRouter->unquarantineAllReaders();

        // ★ BUG-015/027 (work88): RetireQuarantineStore の全強制解放（Audio Thread 停止後 — drainAllUnsafe 契約）
        //   retire enqueue 失敗で退避されたエントリを shutdown 時に確定解放する。
        const auto quarantinedRetireResident = m_retireRouter->quarantineResidentCount();
        if (quarantinedRetireResident > 0) {
            diagLog("[DIAG] releaseResources: retireQuarantineStore resident="
                    + juce::String(static_cast<int64>(quarantinedRetireResident))
                    + " -- performing shutdown drain");
            m_retireRouter->drainAllQuarantineStore();
        }

        const auto residentBefore = dspQuarantineManager_.residentCount();
        if (residentBefore > 0) {
            diagLog("[DIAG] releaseResources: quarantinedSlots="
                    + juce::String(static_cast<int>(residentBefore))
                    + " -- performing shutdown cleanup");

            for (uint32_t slot = 0; slot < convo::isr::DSPHandleRuntime::MAX_DSP_SLOTS; ++slot) {
                // 系統②: フラグ確認＋解放（非アクティブなら false → スキップ）
                if (dspQuarantineManager_.destroyForShutdown(slot)) {
                    // 系統①: DSPHandleRegistry の Quarantined→Reclaimed 遷移
                    //   destroyForShutdown が quarantine フラグ確認を済ませているため安全
                    dspHandleRuntime_.destroyQuarantineSlot(slot, 0);
                    // 系統③: レーン解放 + quarantineResidentCount--
                    worldAuthority_.lifetime().reclaim(slot);
                }
            }

            // バッチ compaction（ループ内個別 compaction より効率的）
            dspQuarantineManager_.compactAuditLog();

            const auto residentAfter = dspQuarantineManager_.residentCount();
            diagLog("[DIAG] releaseResources: quarantine cleanup done "
                    + juce::String(static_cast<int>(residentBefore))
                    + " -> " + juce::String(static_cast<int>(residentAfter)));
        }
    }

    // ★ P3: VerifyDrained — 最終監査フェーズ
    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::VerifyDrained);
    diagLog("[DIAG] releaseResources: VerifyDrained -- collecting drain audit");

    const auto activeHandle = dspHandleRuntime_.getActiveRuntimeDSPHandle();
    const auto fadingHandle = dspHandleRuntime_.getFadingRuntimeDSPHandle();
    // ★ work88 (X3 §6.3 / R4 Phase 4): shutdownReclaim bypass を廃止し、Reclaim Authority
    //   （ShutdownQuiescent モード）に一本化。retire → reclaim の順序は維持（retire は冪等 —
    //   AC-X3-16 二重 retire なし）。
    // ★ dash2 §2.2 (Phase A2 — Step 11): caller-side shutdown 判断（readerRegistrationClosed()）
    //   を撤去し、tryShutdownQuiescentReclaim（ShutdownRuntime が Proof → Permit → reclaim）に
    //   委譲（AC-2: caller-side shutdown 判断 0 件）。retire は事前実行（冪等 — 既存契約）。
    if (!activeHandle.isNull())
    {
        dspHandleRuntime_.retire(activeHandle);
        // ★ C4834 対応: reclaim の戻り値（Proof/Permit 認可結果）を確認。
        //   releaseResources は DrainRetire フェーズで quiescence 成立済みのため true が期待。
        const bool reclaimed = tryShutdownQuiescentReclaim(activeHandle);
        jassert(reclaimed);
        juce::ignoreUnused(reclaimed);
    }
    if (!fadingHandle.isNull() && fadingHandle != activeHandle)
    {
        dspHandleRuntime_.retire(fadingHandle);
        const bool reclaimed = tryShutdownQuiescentReclaim(fadingHandle);
        jassert(reclaimed);
        juce::ignoreUnused(reclaimed);
    }

    diagLog("[DIAG] releaseResources: before ui processor release");
    diagLog("[DIAG] releaseResources: before uiConvolverProcessor.releaseResources");
    uiConvolverProcessor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiConvolverProcessor.releaseResources");

    diagLog("[DIAG] releaseResources: before uiEqEditor.releaseResources");
    uiEqEditor.releaseResources();
    diagLog("[DIAG] releaseResources: after uiEqEditor.releaseResources");

    diagLog("[DIAG] releaseResources: after ui processor release");

    diagLog("[DIAG] releaseResources: skip deferred reclaim (reconfigure phase)");

    // ★ work88 (X4-B §6.4 / X4-B-7): shutdown clear を RuntimeWorldAuthority 経由に一本化
    //   （一時生成 makeRuntimePublishAuthority() は X4-B-8 で廃止）。clear は publish() と統合せず
    //   null 公開（world クリア）として別 API を authority 配下に置く（二十次レビュー §15）。
    //   戻り値の oldWorld は従来 bridge.retireRuntimePublishWorldNonRt(world, true) で retire していた。
    //   shutdown 中は enqueueDeferredDeleteNonRt が Shutdown を返すため実質 no-op だが契約を維持する。
    worldAuthority_.requestShutdownClearNonRt();
    auto* clearedWorld = worldAuthority_.clearPublishedRuntimeSnapshotsNonRt();
    if (clearedWorld != nullptr)
    {
        RuntimePublicationBridge clearBridge{ *this, runtimePublicationValidator_ };
        clearBridge.retirePublishedRuntimeWorldNonRt(clearedWorld, true);
    }

    // ★ 15-P-5: clear 後の最終強制 drain（quiescence 確立時のみ）。
    //   clearPublishedRuntimeSnapshotsNonRt が返した oldWorld は enqueueDeferredDeleteNonRt →
    //   shutdownReclaim → terminalReclaim 経由で、epoch safe なら即時破棄、epoch unsafe
    //   （stuck reader が残る場合）なら TerminalReclaimAuthority に保持される。
    //   drainAllQuarantineStore()（PR2）はこの clear より前に実行済みのため、ここで保持された
    //   World はそこでは対象外。quiescence（activeReaderCount==0）を確認してから Q + E + Terminal
    //   を強制 drain し、Terminal 残留 World のリークを防ぐ。stuck reader が残る場合は
    //   UAF 回避のため強制 drain をスキップし、epoch-gated drain（waitForDrain 内の
    //   drainTerminalReclaim）に委ねる。
    if (m_retireRouter->activeReaderCount() == 0)
        m_retireRouter->drainAllQuarantineStore();

    // ★ work88 (SHUTDOWN-7 五次レビュー): SHUTDOWN-ORDER 契約の防御的検証。
    //   順序不変条件: requestShutdown(:75) → shutdownCoordinatorLoop(:189, join) →
    //   stopRebuildThread(:190, join で Builder 完全終了) → drain wait(:430)。
    //   drain wait 時点で Builder（rebuildThreadIsRunning）は必ず false のはず。
    //   jassert で契約破れ（順序変更・追加経路）を即時検出する。
    jassert(!convo::consumeAtomic(rebuildThreadIsRunning, std::memory_order_acquire));

    const bool drainedWithinBudget = waitForDrain(2000, 2);
    const bool timedOut = !drainedWithinBudget;

    // ★ 15-P-4-5-FIX: Drain residual RetireIntents (slot-state System 1) after graceful drain.
    //   OverflowRing may still hold entries that drainOverflowRing() couldn't process because
    //   dequeuePendingRetireIntents() is ONLY called in RT commit path (now stopped).
    //   This is idempotent and safe — reclaim() is a no-op on already-Reclaimed slots.
    drainPendingRetireIntentsForShutdown();

    if (timedOut) {
        // ★ A-3: VerifyDrained で Reader 異常を検出 → markTimedOut に ReaderActive を伝達
        auto audit = collectDrainAudit();
        auto reason = convo::isr::ShutdownBlockingReason::Unknown;
        if (audit.stuckReaderCount > 0)
            reason = convo::isr::ShutdownBlockingReason::ReaderActive;
        else if (convo::consumeAtomic(rebuildThreadIsRunning, std::memory_order_acquire))
            reason = convo::isr::ShutdownBlockingReason::ActiveBuilder;   // ★ SHUTDOWN-7: Builder が Build Session 進行中
        shutdownRuntime_.markTimedOut(reason);
    }

    // ★ 改善③: World Consistency 診断は VerifyDrained では常に実行（タイムアウト有無に依存しない）
    {
        const auto audit = collectDrainAudit();
        const auto cs = audit.verifyWorldConsistency();
        if (cs != convo::isr::RuntimeDrainAudit::ConsistencyState::Consistent) {
            diagLog("[AUDIT] VerifyDrained: world consistency="
                + juce::String(static_cast<int>(cs))
                + " published=" + juce::String(static_cast<juce::int64>(audit.publishedCount))
                + " retired=" + juce::String(static_cast<juce::int64>(audit.retiredCount))
                + " active=" + juce::String(static_cast<juce::int64>(audit.activeWorldCount)));
            // ★ B-2: HealthState を診断情報として出力
            diagLog("[AUDIT] VerifyDrained: healthState="
                + juce::String(static_cast<int>(audit.healthState))
                + " activeReaders=" + juce::String(static_cast<juce::int64>(audit.activeReaderCount))
                + " stuckReaders=" + juce::String(static_cast<juce::int64>(audit.stuckReaderCount)));
            emitEvidenceTickNonRt(true);
        }
    }

    if (!drainedWithinBudget || !isFullyDrained())
    {
        if (timedOut)
            diagLog("[DIAG] releaseResources: drain timeout reached, performing safe tryReclaim (drainAll skipped)");

        // [P1 Phase1-B] drainPublicationLogForShutdown removed
        drainDeferredRetireQueues(true);
        m_epochDomain.tryReclaim();  // ★ P1-2: drainAll 禁止 → 安全な tryReclaim
    }

    m_coordinator.finalizeShutdown(timedOut);  // ★ P1-2: 二段構えの正常系

    // ★ 15-P-CROSS-IMPLEMENTATION-1: GAP-CROSS-1 fix — terminal drain of residual OwnerChannel owners.
    //   finalizeShutdown 直後: producer/consumer は既に停止済み (shutdownCoordinatorLoop join,
    //   stopRebuildThread join)。quiescence は確認済み (advanceRetireEpoch, drainAllQuarantineStore).
    //   drainAllNonRt は consume->publish(nullptr,release) single-transfer — owner==nullptr で
    //   empty slot を検出するため re-drain は no-op。
    //   ★ Phase I-T1-D101-1F: OwnerChannel residual = pre-publication transport residue.
    //   publishAndSwap LP を通過していないため W ∉ PublishedDomain — DeletionEntryType::Generic
    //   （destruction only・no World retirement observation）として既存 DeferredDeletionQueue →
    //   reclaim → deleter chain へ移譲する。
    //   enqueueRetire は Success|QueuePressure のみ返す (Shutdown は dead code) ため、
    //   callback は既存 authority chain へ ownership を確実に移転する。
    const auto drainedResidual = worldAuthority_.ownerChannel().drainAllNonRt(
        [this](const RuntimeState* raw) noexcept {
            // const RuntimeState* -> void*: existing World deleter
            // (retirePublishedRuntimeWorldNonRt, AudioEngine.h:3525) が
            // static_cast<RuntimePublishWorld*>(p) で const_cast を行う。
            enqueueDeferredDeleteNonRtWithResult(
                const_cast<RuntimeState*>(raw),
                [](void* p) noexcept {
                    auto* ptr = static_cast<RuntimePublishWorld*>(p);
                    ptr->unseal();
                    ptr->~RuntimePublishWorld();
                    convo::aligned_free(ptr);
                },
                DeletionEntryType::Generic);
        });
    if (drainedResidual > 0)
        diagLog("[AUDIT] drainAllNonRt residual: reclaimed "
            + juce::String(static_cast<juce::int64>(drainedResidual))
            + " residual OwnerChannel owners -> terminal retire chain");

    // ★ A-2.7: ReleaseResources の DrainAudit 統合
    const auto currentShutdownPhase = shutdownRuntime_.getPhase();
    const bool traceSafe = (currentShutdownPhase >= convo::isr::ShutdownPhase::EpochSettled);
    const auto audit = collectDrainAudit();
    if (!drainedWithinBudget || !audit.isAllZero()) {
        diagLog("[ISR][Shutdown] Drain incomplete: "
                "pendingPub=" + juce::String(static_cast<int64>(audit.pendingPublication)) +
                " pendingRetire=" + juce::String(static_cast<int64>(audit.pendingRetire)) +
                " crossfade=" + juce::String(static_cast<int64>(audit.activeCrossfadeCount)) +
                " routerPendingRetire=" + juce::String(static_cast<int64>(audit.routerPendingRetire)) +
                " maxDeferredAgeMs=" + juce::String(static_cast<int64>(audit.maxDeferredAgeMs)) +
                " deferred=" + juce::String(static_cast<int64>(audit.deferredPublish)) +
                " quarantine=" + juce::String(static_cast<int64>(audit.quarantineResident)) +
                " oldestAgeMs=" + juce::String(static_cast<int64>(audit.oldestPendingAgeMs)) +
                " (observation only)");
        if (traceSafe) {
            const auto evidenceRoot = std::filesystem::current_path() / "evidence";
            worldAuthority_.lifetime().emitRetireTrace(evidenceRoot / "retire_trace_shutdown_last.json");
        }
    }
    if (audit.quarantineResident > 0) {
        diagLog("[ISR][Shutdown] Drain complete but quarantine residents remain: "
                + juce::String(static_cast<int64>(audit.quarantineResident)));
    }

    runtimePublicationBridge_.markShutdownComplete();

    const auto pendingRetireCount = [&]() noexcept -> uint32_t
    {
        return m_retireRouter->pendingRetireCount();
    }();

    const auto activeCrossfadeCount = crossfadeRuntime_.isPending() ? 1u : 0u;
    shutdownRuntime_.setBoundedTeardownCounters(
        convo::consumeAtomic(rtLocalState_.audioCallbackActiveCount, std::memory_order_acquire),
        activeCrossfadeCount,
        pendingRetireCount,
        activeEpochObserverCount());

    debugRuntime_.recordHBEdge(300u,
                               400u,
                               static_cast<std::uint64_t>(pendingRetireCount),
                               static_cast<std::uint64_t>(activeCrossfadeCount),
                               static_cast<int>(std::memory_order_acq_rel));

    shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::ShutdownComplete);
    shutdownRuntime_.emitShutdownTrace();

    emitEvidenceTickNonRt(true);

    convo::publishAtomic(lifecycleState, EngineLifecycleState::Unprepared, std::memory_order_release);
    diagLog("[DIAG] releaseResources: ABOUT_TO_EXIT_SCOPE");

    // P0-A0: LifecycleIsolationRuntime integration - leave release phase
    lifecycleRuntime_.leaveRelease(lifecycleToken);
}

// ★ 15-P-4-5-FIX: EmergencyDrain-safe drain of RetireIntent (slot-state) system.
//   Drains OverflowRing → emitRetireIntent → MPSC queue → processIntent → reclaim().
//   This is the *slot-state* drain (System 1), separate from DeferredDeletionQueue (System 2/pointer-lifetime).
//   Idempotent: reclaim() is a state-transition no-op on already-Reclaimed slots; recomputeIntentPending()
//   handles zero-count correctly. Must be called with no active RT audio thread (post-stopRebuildThread).
void AudioEngine::drainPendingRetireIntentsForShutdown() noexcept
{
    try {
        auto& lifetime = worldAuthority_.lifetime();

        // Step 1: Drain OverflowRing (SPSC lock-free) — pop all entries, emitRetireIntent each
        {
            auto* overflowRing = lifetime.getOverflowRing();
            if (overflowRing != nullptr)
            {
                convo::isr::RetireOverflowEntry entry;
                while (overflowRing->pop(entry))
                {
                    lifetime.emitRetireIntent(entry.intent);
                }
                // Safety: clear in case partial pop occurred
                overflowRing->clear();
            }
        }

        // Step 2: Drain LifetimeState MPSC queue + fallback queue (mutex-protected)
        //   dequeueOne() pops from lock-free slots_[256] MPSC queue
        //   dequeueFallback() pops mutex-protected fallbackQueue_
        {
            convo::isr::RetireIntent intent;
            constexpr int kMaxIntentDrain = 65536;  // safety bound — should never hit in practice
            int drained = 0;

            while (drained < kMaxIntentDrain)
            {
                // Try MPSC queue first (lock-free, noexcept)
                if (lifetime.dequeueOne(intent))
                {
                    lifetime.reclaim(intent.dspSlot);
                    ++drained;
                    continue;
                }
                // Try fallback queue (mutex-protected, noexcept)
                if (lifetime.dequeueFallback(intent))
                {
                    lifetime.reclaim(intent.dspSlot);
                    ++drained;
                    continue;
                }
                // Both queues empty — done
                break;
            }
        }

        // Step 3: If any intents were re-injected into OverflowRing (e.g. by reclaim → enqueueRetire),
        //   drain again (bounded 3 iterations — overflow ring should not refill during shutdown
        //   since no RT thread is pushing).
        for (int iter = 0; iter < 3; ++iter)
        {
            auto* overflowRing2 = lifetime.getOverflowRing();
            if (overflowRing2 == nullptr) break;

            convo::isr::RetireOverflowEntry entry2;
            bool refilled = false;
            while (overflowRing2->pop(entry2))
            {
                lifetime.emitRetireIntent(entry2.intent);
                refilled = true;
            }
            if (refilled)
            {
                overflowRing2->clear();
                // Drain the re-injected intents
                convo::isr::RetireIntent intent2;
                while (lifetime.dequeueOne(intent2) || lifetime.dequeueFallback(intent2))
                {
                    lifetime.reclaim(intent2.dspSlot);
                }
            }
            else
            {
                break;  // No refill — fully drained
            }
        }
    }
    catch (...) {
        // noexcept context — swallow any unexpected exception
        // (reclaim/dequeueOne/dequeueFallback are effectively noexcept: atomics + mutex only)
    }
}
