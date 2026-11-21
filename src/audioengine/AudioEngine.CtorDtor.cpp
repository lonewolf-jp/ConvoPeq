#include <JuceHeader.h>
#include "AtomicAccess.h"
#include "AudioEngine.h"
#include "RetryScheduler.h" // ★ D101-24 Step 1: complete type for unique_ptr<RetryScheduler> destruction
#include "core/RuntimeReaderContext.h"
#include "RuntimePublicationOrchestrator.h"
#include "NoiseShaperLearner.h"
#include "ISRRetireRouter.h"
#include "DSPLifetimeManager.h"
#include "ISRRetireLatency.h"
#include "WorldLifetimeRecorder.h"   // ★ A(D1): 退役遅延の測定

namespace {
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

// ★ 静的メンバ定義: 全局一意 Engine インスタンスID カウンタ
std::atomic<uint64_t> AudioEngine::s_nextEngineInstanceId_{0};

// ★ 2026-10-30 (別変更単位): 生存エンジン数の診断カウンタ。診断のみで、振る舞いは変えない。
std::atomic<int> AudioEngine::s_engineInstanceCount_{0};

// ★ WORK113: reader 基盤をプロセス寿命へ移した累計（shutdown の fail-closed 経路）。
std::atomic<int> AudioEngine::s_abandonedReaderSubstrates_{0};
std::atomic<int> AudioEngine::s_abandonedSubstrateOverflow_{0};
std::atomic<int> AudioEngine::s_shutdownPublisherIdleFailures_{0};
std::atomic<int> AudioEngine::s_shutdownReaderAdmissionIdleFailures_{0};
std::atomic<int> AudioEngine::s_shutdownEngineReaderBusy_{0};
std::atomic<const char*> AudioEngine::s_lastShutdownHoldReason_{nullptr};

namespace
{
// 警告の重複抑制用（最初の 1 回だけ出す）。
std::atomic<bool> s_engineMultiInstanceWarned{false};
}
// ★ F2' (2026-10-25): destroyDSPCoreNode（static）から解放直前のハンドル無効化を行うための
//   「現在のエンジン」。製品はプロセス内 1 エンジン、テストは逐次生成。
std::atomic<AudioEngine*> AudioEngine::s_handleInvalidationOwner{nullptr};

// ★ R-51 (2026-10-30): 破棄専用索引の実体。
std::mutex AudioEngine::s_destroyIndexMutex;
std::unordered_map<void*, AudioEngine::DestroyIndexEntry> AudioEngine::s_destroyIndex;

AudioEngine::AudioEngine()
    : uiEqEditor(*this)
    , eqCacheManager(*this)
#pragma warning(push)
#pragma warning(disable : 4996) // [[deprecated]] — transitional, SnapshotCoordinator EpochDomain (P1-7)
    , m_coordinator(m_epochDomain)
#pragma warning(pop)
    , m_workerThread(m_commandBuffer, m_generationManager, &affinityManager)
    , worldAuthority_(runtimePublicationBridge_)
    , shutdownRuntime_(runtimePublicationBridge_)  // ★ dash2 §2.2 (Step 14 — Authority Singularization 完了): constructor 固定注入（setReclaimAuthority 廃止 — immutable association）
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ WORK113 変更1: 識別レジストリのログ出力先を 1 回だけ設定する（以後は読み取り専用）。
    convo::diag::registry().setLogSink ([] (const char* line)
    {
        juce::Logger::writeToLog (juce::String::fromUTF8 (line));
    });

    // ★ WORK113 Gate: RT から NonRT API を呼ぶ誤用を Debug で検出するための probe。
    m_epochDomain.setRtThreadProbe ([]() noexcept
    {
        return convo::numeric_policy::isAudioThread();
    });
#endif

    // ★ A(D1): 測定器はエンジン 1 個につき 1 回リセットする（前インスタンスの観測を持ち越さない）
    convo::isr::RetireLatencyRecorder::instance().reset();
    convo::isr::WorldLifetimeRecorder::instance().reset();   // ★ Item2: World 寿命の記録

    // ★ engineInstanceId 初期化 (全局一意)
    engineInstanceId_ = s_nextEngineInstanceId_.fetch_add(1, std::memory_order_relaxed) + 1; // NOLINT(atomic-dot-call): relaxed counter

    // [work21] ISRRetireRouter初期化
    // ★ T1 (D100.4): reference observer に telemetry を配線（onRelease → releaseObserved 転送・
    //   sampler の outstanding 推定 acquireObserved - releaseObserved を正しくする）。
    worldRetirementReference_.setTelemetry(&worldRetirementTelemetry_);
    m_retireRouter = std::make_unique<convo::isr::ISRRetireRouter>(m_epochDomain, &worldRetirementReference_);
    // ★ BUG-015/027 (work88): SnapshotCoordinator の退避移送先を Router に接続
    //   （Category A — Router API 経由で RetireQuarantineStore へ移送。直接保持はしない）
    m_coordinator.setRetireSink(m_retireRouter.get());
    // [PR-1.5] RuntimePublicationOrchestrator 初期化 (engineInstanceId を注入)
    runtimeOrchestrator_ = std::make_unique<convo::isr::RuntimePublicationOrchestrator>(*this, engineInstanceId_);

    // [P1 Phase1-B] PublicationIntent/PublicationLog initialization removed
    uiConvolverProcessor.setRcuProvider(*this);
    uiConvolverProcessor.setRetireCoordinator(&runtimePublicationBridge_);
    // Route EQ retirement through coordinator
    uiEqEditor.setRetireCoordinator(&runtimePublicationBridge_);

    // ★ P1-8: RuntimeHealthMonitor 初期化
    m_healthMonitor.setRetireRouter(m_retireRouter.get());
    m_healthMonitor.setOrchestrator(runtimeOrchestrator_.get());
    m_healthMonitor.setRetireHighWatermarkRef(&retireHighWatermark_);
    m_healthMonitor.setCrossfadeRuntime(&crossfadeRuntime_);
    m_healthMonitor.setCrossfadeEventDropRef(crossfadeRuntime_.getCrossfadeEventDropCountRef());
    // ★ Work38: Retire Reclaim Latency 監視用参照（型安全）
    //   reclaimLatency_ は AudioEngine の atomic<double> — オーバーロード解決により double* 版が呼ばれる
    m_healthMonitor.setMaxRetireAgeRef(&reclaimLatency_);
    // ★ Practical-4: Reader Slot 使用率監視用参照
    //   activeReaderCount は ISRRetireRouter 経由で取得（HealthMonitor が直接読む）
    m_healthMonitor.setOverflowCountRef(m_retireRouter->getOverflowCountRef());
    m_healthMonitor.setEventCallback(
        [this](const convo::HealthEvent& ev) { onHealthEvent(ev); });

    // [work37 Phase 4.1] PolicyEngine Action Callback
    m_healthMonitor.setActionCallback(
        [this](convo::RecoveryAction action) { executeRecoveryAction(action); });

    // [work39 Phase 6] RestoreStep2 Callback — publishIdleWorldOnly(HardReset)
    m_healthMonitor.setRestoreStep2Callback([this]() {
        if (convo::consumeAtomic(m_lastHardResetGeneration_, std::memory_order_acquire)
            != convo::consumeAtomic(m_restoreGeneration_, std::memory_order_acquire)) {
            const convo::RuntimeReaderContext ctx{
                messageThreadRcuReader, convo::ObserveChannel::Message };
            const auto handle = makeRuntimeReadHandle(ctx);
            auto* dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(handle);
            if (dsp != nullptr) {
                (void)publishIdleWorldOnly(dsp, convo::TransitionPolicy::HardReset);
                convo::publishAtomic(m_lastHardResetGeneration_,
                    convo::consumeAtomic(m_restoreGeneration_, std::memory_order_acquire),
                    std::memory_order_release);
                convo::publishAtomic(m_restorePhase_, convo::RestorePhase::IdleWorldPublished,
                    std::memory_order_release);
            }
        }
    });

    // ★ P1-B: Admission に HealthState 参照を設定
    runtimeOrchestrator_->setAdmissionHealthStateRef(m_healthMonitor.getHealthStateRef());

    // ★ B14: Vyukov MPSC Retire Queue 初期化
    worldAuthority_.lifetime().initQueue();

    // ★ D101-24 Step 2: production DispatchFn wiring (AudioEngine owns → DispatchFn → submitRebuildIntent boundary)
    //   RetryScheduler owns time/ordering only; BuildError/RetryDisposition stays caller-side.
    retryScheduler_ = std::make_unique<RetryScheduler>([this](const RetryScheduleRequest& req) noexcept {
        this->submitRebuildIntent(req.kind, req.reason, req.rebuildClass, req.collapsePolicy);
    });

    // ★ F2' (2026-10-25): destroyDSPCoreNode は static（EBR の関数ポインタ）なので this を
    //   持てない。解放直前のハンドル無効化のために「今生きているエンジン」を 1 つ公開する。
    convo::publishAtomic(s_handleInvalidationOwner, this, std::memory_order_release);

    // ★ 2026-10-30 (診断): 正常に成立した実例をカウントする。
    //   複数インスタンスは許容する（禁止・例外化・assert はしない）。非 RT 経路で 1 回だけ警告する。
    {
        const int liveCount = convo::fetchAddAtomic(s_engineInstanceCount_, 1,
                                                     std::memory_order_acq_rel) + 1;
        if (liveCount > 1)
        {
            bool expected = false;
            if (convo::compareExchangeAtomic(s_engineMultiInstanceWarned, expected, true,
                                             std::memory_order_acq_rel, std::memory_order_acquire))
            {
                juce::Logger::writeToLog(juce::String::formatted(
                    "[ENGINE_MULTI_INSTANCE] %d AudioEngine instances are alive in this process",
                    liveCount));
            }
        }
    }
}

AudioEngine::~AudioEngine()
{
    // ★ F2' (2026-10-25): 所有者の解除は「DSPCore を解放し終えた後」でなければならない。
    //   dtor の後半（drainDeferredRetireQueues / drainAllQuarantineStore）も
    //   destroyDSPCoreNode を呼ぶため、冒頭で解除すると teardown 経由の解放だけ HL-1 が
    //   効かず、F2' 導入前の状態に戻ってしまう（レビュー指摘 A）。
    //   解除は本体末尾（§末尾の compare_exchange）で行う。
    diagLog("[DIAG] ~AudioEngine: enter");
    // Shutdown sequence (list.md 12.1.2):
    // 1) stop callbacks/workers, 2) detach published runtime pointers,
    // 3) retire captured runtimes, 4) force epoch advance, 5) deterministic drain/reclaim.
    // 以後の順序を固定して、終了時の reclaim レースを防止する。
    setShutdownPhase(ShutdownPhase::StopAcceptingWork, "~AudioEngine");
    convo::publishAtomic(lifecycleState, EngineLifecycleState::Releasing, std::memory_order_release); // release: isShuttingDown の acquire と HB
    cancelPendingUpdate();
    runtimePublicationBridge_.requestShutdown();

    // 終了順序を固定化して、終了時フリーズを防ぐ。
    setShutdownPhase(ShutdownPhase::StopAudio, "~AudioEngine");
    stopTimer();

    setShutdownPhase(ShutdownPhase::StopWorkers, "~AudioEngine");
    // ★ D101-24 Step 4: RetryScheduler を先に停止 (stop accepting → clear/discard → prevent callback → join)
    //   rebuildThread 停止前に scheduler の dispatch を止めないと shutdown 後に submitRebuildIntent が走る。
    //   member destruction order に依存せず明示 shutdown (idempotent)。
    if (retryScheduler_)
        retryScheduler_->shutdown();
    // releaseResources が未実行の異常系でも worker 終了を保証する。
    shutdownCoordinatorLoop();  // ★ FUTURE-9: join Coordinator Worker (defensive)
    stopRebuildThread();

    // まず rebuild thread 側へ終了を通知し、pending task を破棄して
    // 終了時に重い再構築へ入る経路を閉じる。
    // pending task を破棄して進行中 rebuild を obsolete にし、thread を停止する。
    // ★ D162-2-E (E-2): activeToRelease / fadingToRelease は slot 切離しの観測用のみとなり
    //   （E-2 により handle-based retire に移管）、pointer-value retirement には使用しない。
    DSPCore* activeToRelease = nullptr;
    DSPCore* fadingToRelease = nullptr;
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);
        const convo::RuntimeReaderContext messageCtx{ messageThreadRcuReader, convo::ObserveChannel::Message };
        const auto runtimeReadHandle = makeRuntimeReadHandle(messageCtx);
        validateDistinctRuntimeSlots("~AudioEngine.beforeClear",
                 getActiveRuntimeDSP(),
                         resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                         nullptr);

        convo::fetchAddAtomic(rebuildRequestGeneration, 1, std::memory_order_acq_rel); // acq_rel: rebuild observer の acquire と HB

        // active runtime slot / fading runtime slot はここでスロットを切り離すだけにして、
        // 実体の解放は retireDSPHandleForRuntime() → deferred delete / epoch drain に寄せる。
        {
            // ★ BUG-051: sentinel (uintptr_t)-1 は書き込まれない（常に nullptr or 有効 ptr）。
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

        validateDistinctRuntimeSlots("~AudioEngine.afterClear",
                 getActiveRuntimeDSP(),
                 resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle),
                         nullptr);

        // pendingTask.currentDSP は worker 側の未コミット生成物なので、
        // ここで回収して以後の commit 経路に残さない。
        if (hasPendingTask)
        {
            if (pendingTask.currentDSP)
            {
                DSPLifetimeManager lifetimeMgr(*this);
                lifetimeMgr.retire(pendingTask.currentDSP);
                pendingTask.currentDSP = nullptr;
            }

            hasPendingTask = false;
            publishRetryReady = false;
        }
    }

    // [P1 Phase1-B] drainPublicationLogForShutdown removed

    // ★ D162-2-E (E-2 / INV-D162-7): activeRuntimeDSPSlot / fadingRuntimeDSPSlot は
    //   placeholder 専用レガシースロット（h:2267 comment 参照 — runtime world 公開後は null）であり、
    //   bootstrap placeholder 破壊後は dangling ポインタを保持し続ける。
    //   この値を retireDSPHandleForRuntime（map.find by pointer value）に渡すと、
    //   address reuse 時に生存 DSP の map entry を誤 lookup して二重破壊を引き起こし得る
    //   （D162-2-C §4.3 / D0-5-A）。
    //   修正: pointer-value retirement を廃止し、handle registry の authority
    //   （getActiveRuntimeDSPHandle / getFadingRuntimeDSPHandle — generation 検証付き）のみを
    //   retirement identity として使用する。slot 値自体は topology convenience state であり、
    //   ownership authority に昇格させない。
    //   なお releaseResources が正常実行済みの場合、これらの slot は既に null のため
    //   retiredByHandle は no-op（二重 retire 不可・DSPHandleRuntime::retire は冪等）。
    //   activeToRelease / fadingToRelease は handle-based retire で全てカバーされるため
    //   pointer-value retirement は廃止（変数は観測用として維持・juce::ignoreUnused で抑制）。
    {
        DSPLifetimeManager lifetimeMgr(*this);
        const auto activeHandleAtDtor = dspHandleRuntime_.getActiveRuntimeDSPHandle();
        if (!activeHandleAtDtor.isNull())
            lifetimeMgr.retireByHandle(activeHandleAtDtor);
        const auto fadingHandleAtDtor = dspHandleRuntime_.getFadingRuntimeDSPHandle();
        if (!fadingHandleAtDtor.isNull() && fadingHandleAtDtor != activeHandleAtDtor)
            lifetimeMgr.retireByHandle(fadingHandleAtDtor);
        juce::ignoreUnused(activeToRelease);
        juce::ignoreUnused(fadingToRelease);
    }

    uiConvolverProcessor.removeChangeListener(this);
    uiEqEditor.removeChangeListener(this);

    // Note: stopRebuildThread は releaseResources() で呼ばれる。
    // dtor が releaseResources 経由で呼ばれる場合、stopRebuildThread は既に完了している。
    // dtor が直接呼ばれる場合（例：ホストが releaseResources を呼ばない異常系）、
    // rebuildThreadShouldExit が既に true なので thread ループは速やかに終了する。

    // Snapshot worker を停止。
    shutdownWorkerThread();

    setShutdownPhase(ShutdownPhase::ForceEpochAdvance, "~AudioEngine");
    m_retireRouter->publishEpoch();

    // ★ Practical-7: Graceful Drain Phase — pendingRetireCount が 0 になるまでポーリング待機
    //   最大 5 秒間のみ待機し、タイムアウト時は強制 drain にフォールバック。
    setShutdownPhase(ShutdownPhase::DrainRetire, "~AudioEngine");
    // ★ work88 (X3 §6.3 / INV-X3-4 / INV-ISR-04): CloseReaderRegistration（系統2 — ~AudioEngine）。
    //   releaseResources 未実行の異常系 shutdown でも、graceful drain 前に reader 新規登録を封じる
    //   （0 に達した後の再登録を構造的に排除）。登録済み slot の enter/exit は継続可能。
    m_epochDomain.closeReaderRegistration();
    {
        const int kGracefulDrainMaxMs =
            convo::consumeAtomic(shutdownWaitTimeoutMs_, std::memory_order_acquire);
        constexpr int kGracefulDrainPollMs = 10;
        int waitedMs = 0;
        while (waitedMs < kGracefulDrainMaxMs)
        {
            if (m_retireRouter->pendingRetireCount() == 0
                && m_retireRouter->activeReaderCount() == 0)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(kGracefulDrainPollMs));
            waitedMs += kGracefulDrainPollMs;
            // tick: reclaim を進めて pendingRetire の消化を促進
            m_retireRouter->publishEpoch();
            m_retireRouter->tryReclaim();
        }
        if (waitedMs >= kGracefulDrainMaxMs)
        {
            diagLog("[AUDIT] Graceful drain timeout after " + juce::String(kGracefulDrainMaxMs)
                + "ms, pendingRetireCount="
                + juce::String(static_cast<int>(m_retireRouter->pendingRetireCount()))
                + " — forcing drain");
        }
    }

    // ★ WORK113: terminal shutdown の停止状態を確定する（fence / gate は以後解除しない）。
    m_epochDomain.engageShutdownReaderFence();
    const int shutdownWaitMs =
        convo::consumeAtomic(shutdownWaitTimeoutMs_, std::memory_order_acquire);
    // ★ publisher idle の待ちは admission と分ける。テストは admission だけを意図的に
    //   短くし、publisher はスケジューリングのぶらつきでタイムアウトしない予算にできる。
    const int publisherWaitOverride =
        convo::consumeAtomic(shutdownPublisherWaitTimeoutMs_, std::memory_order_acquire);
    const int publisherWaitMs = publisherWaitOverride > 0 ? publisherWaitOverride : shutdownWaitMs;
    const bool terminalPublishersIdle =
        m_epochDomain.closePublishersAndWaitIdle(std::chrono::milliseconds(publisherWaitMs));
    if (! terminalPublishersIdle)
    {
        (void) convo::fetchAddAtomic(s_shutdownPublisherIdleFailures_, 1, std::memory_order_acq_rel);
        diagLog("[FAULT] epoch_publishers_not_idle timeouts="
                + juce::String(static_cast<int>(m_epochDomain.epochIdleTimeouts())));
    }

    // ★ WORK113: reader admission（予約〜enter）の完了を待つ。fence と admission gate は
    //   閉じているため新規の入場は開始されず、待機後は「入場中の reader が存在しない」
    //   ことが確定する（proof と checked drain の前提）。
    const bool terminalReaderAdmissionsIdle =
        m_epochDomain.waitForReaderAdmissions(std::chrono::milliseconds(shutdownWaitMs));
    if (! terminalReaderAdmissionsIdle)
    {
        (void) convo::fetchAddAtomic(s_shutdownReaderAdmissionIdleFailures_, 1, std::memory_order_acq_rel);
        diagLog("[FAULT] reader_admissions_not_idle active="
                + juce::String(static_cast<int>(m_epochDomain.readerAdmissionsActive()))
                + " pendingSlots="
                + juce::String(static_cast<int>(m_epochDomain.admissionPendingSlots())));
    }

    // ★ WORK113 (2026-11-18 Owner 指示): 破棄・回収の前提は次の 3 条件である。
    //   (1) publisher が idle: false は PublishScope 内にいることを意味し、そのデストラクタは
    //       gate_.exit() を呼ぶ。domain を破棄すると破棄済み gate へのアクセスになる。
    //   (2) reader admission が idle: 予約〜enter の途中の reader が残っていない。
    //   (3) active reader が 0: enterReader 完了後、保護区間内に留まる reader が残っていない。
    //       (2) だけでは depth>0 の reader を見落とす。
    //   (4) engine 所有 reader が保護区間外: audioThreadRcuReader / messageThreadRcuReader は
    //       AudioEngine のメンバーであり、使用中に engine を破棄すると reader 自身のメンバーが
    //       寿命切れになる。ホスト契約（audio callback の停止・join 後に破棄する）を実行時に観測する。
    //       契約の根拠: 製品は ~MainWindow で removeAudioCallback → closeAudioDevice の後に
    //       member の audioEngine を破棄する。harness は stop() で audio thread を join してから
    //       engine を破棄する。
    //   取得順序: fence は既に閉じている（engageShutdownReaderFence）。admission gate も閉じて
    //   いるため新規 admission は開始できず、(2) の成立後は新規入場も起きない。したがって (3) は
    //   単調にしか減らない値を (2) の後に読むことになり、以後の増加はない。
    //   安全性を証明できない対象は強制解放せず保持する（リークを選び、UAF を選ばない）。
    const bool activeReadersIdle =
        (m_retireRouter != nullptr && m_retireRouter->activeReaderCount() == 0);
    const bool engineReadersIdle =
        ! audioThreadRcuReader.isInside() && ! messageThreadRcuReader.isInside();
    if (! engineReadersIdle)
    {
        (void) convo::fetchAddAtomic(s_shutdownEngineReaderBusy_, 1, std::memory_order_acq_rel);
        diagLog("[FAULT] engine_reader_busy audioInside="
                + juce::String(static_cast<int>(audioThreadRcuReader.isInside() ? 1 : 0))
                + " messageInside="
                + juce::String(static_cast<int>(messageThreadRcuReader.isInside() ? 1 : 0)));
    }

    const bool shutdownSubstrateQuiescent =
        terminalPublishersIdle && terminalReaderAdmissionsIdle
        && activeReadersIdle && engineReadersIdle;

    // 実際に成立しなかった条件を保持理由として記録する（診断の整合）。
    // 文字列は DIAG ON のときだけ持つ（OFF ではログも evidence も出さない）。
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const char* const shutdownHoldReason =
        ! terminalPublishersIdle      ? "epoch_publishers_not_idle"
        : ! terminalReaderAdmissionsIdle ? "reader_admissions_not_idle"
        : ! activeReadersIdle         ? "active_readers_not_idle"
        : "engine_readers_not_idle";

    convo::publishAtomic(s_lastShutdownHoldReason_,
                         shutdownSubstrateQuiescent ? static_cast<const char*>(nullptr)
                                                    : shutdownHoldReason,
                         std::memory_order_release);
#else
    const char* const shutdownHoldReason = nullptr;
#endif

    // 最終 retire は 1 回だけ（releaseResources で実行済みなら 0 回）。
    //   ここは retire 要求の移送のみで、ポインタの解放は行わない。
    (void) m_coordinator.finalizeRetireOnce();

    // ★ work88 (X4-B §6.4 / X4-B-7): shutdown clear を RuntimeWorldAuthority 経由に一本化
    //   （一時生成 makeRuntimePublishAuthority() は X4-B-8 で廃止）。詳細は ReleaseResources 側に同様。
    worldAuthority_.requestShutdownClearNonRt();
    auto* clearedWorld = worldAuthority_.clearPublishedRuntimeSnapshotsNonRt();
    if (clearedWorld != nullptr)
    {
        RuntimePublicationBridge clearBridge{ *this, runtimePublicationValidator_ };
        clearBridge.retirePublishedRuntimeWorldNonRt(clearedWorld, true);
    }

    // 解放を伴う deferred queue の回収は publisher と reader の静止時だけ実行する。
    if (shutdownSubstrateQuiescent)
    {
        drainDeferredRetireQueues(true);
    }
    else
    {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        diagLog("[WORK113] shutdown hold: deferred retire queues retained (publisher or reader not idle)");
#endif
    }

    // ★ 15-P-4-5-FIX: Drain residual RetireIntents (slot-state System 1) before pointer-lifetime
    //   drain (drainAll). OverflowRing may hold entries not yet processed because the RT commit
    //   path (the only caller of dequeuePendingRetireIntents()) has stopped.
    // ★ WORK113 (2026-11-18 Owner 指示): この関数は retire 要求を移送するだけで物理解放は
    //   しないが、EpochControl::reclaim() によって slot の lifecycle lane を
    //   ReclaimEligible → Reclaimed へ進める。静止を確認できない状態で slot の再利用可能性を
    //   動かさないため、fail-closed 経路では実行しない（残存要求は保持する）。
    if (shutdownSubstrateQuiescent)
    {
        drainPendingRetireIntentsForShutdown();
    }
    else
    {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        diagLog("[WORK113] shutdown hold: residual RetireIntents retained (slot lanes untouched)");
#endif
    }

    // ★ 15-P-5: 完全 drain（D + Q + E + Terminal）。m_epochDomain.drainAll() は D のみのため、
    //   TerminalReclaimAuthority に保持された World が漏れる。静止を確認できたときだけ
    //   m_retireRouter->drainAll()（D + Q/E/T の強制解放）を実行する。
    // ★ WORK113 (2026-11-18 Owner 指示): 4 条件のいずれかが欠ける場合は force drain を行わない。
    //   active reader が残る場合、Q/E/T の drainAllQuarantineStore（epoch 非依存）が参照中の
    //   World / DSP を解放し得る。旧 stuck-reader fallback（D + Q/E/T の強制解放）は撤去済みで、
    //   ここでは所有権を保持する fail-closed 経路だけを通る。
    if (! shutdownSubstrateQuiescent)
    {
        holdRetireQueuesForShutdownNonRt(shutdownHoldReason,
                                         terminalPublishersIdle,
                                         terminalReaderAdmissionsIdle,
                                         activeReadersIdle,
                                         engineReadersIdle);
    }
    else
    {
        m_retireRouter->drainAll();
    }

    // ★ A(D1): 最終 drain 後の状態を違反レポートへ反映する（runtime 段階は保持される）
    debugRuntime_.emitFinalViolationReport();

    // ★ F1'-b (2026-10-25) 監査対応（欠陥 C）: 延期リストの最終 drain。
    //   EBR store を全部 drain した後でも、pin 残留で延期された DSPCore は
    //   エンジン再試行リストに残っている。ここで 1 度試行し、
    //   回収できたものを解放する。回収できないものは **強制解放しない**（UAF を選ばない）。
    {
        const auto deferredBefore = deferredDestroyResidentCount();
        if (deferredBefore > 0)
        {
            if (shutdownSubstrateQuiescent)
            {
                (void)retryDeferredDestroys();
            }
            else
            {
                // ★ WORK113: 静止を確認できないため、pin が解放済みに見えても回収しない。
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
                diagLog("[WORK113] shutdown hold: deferred DSPCore retry skipped (retained)");
#endif
            }
            const auto deferredAfter = deferredDestroyResidentCount();
            juce::Logger::writeToLog(juce::String::formatted(
                "[F1_DEFERRED_SHUTDOWN] before=%zu after=%zu%s",
                deferredBefore, deferredAfter,
                deferredAfter > 0 ? " (uncollected: pins still held — leaked, not UAF)" : ""));
            if (deferredAfter > 0)
            {
                // 未回収を明示的に残す（~AudioEngine はこの後 member teardown に入る）。
                // 強制解放すると UAF になるため、リークを選ぶ。
                diagLog("[F1_DEFERRED_SHUTDOWN] uncollected deferred DSPCore retained");
            }
        }
    }

    // ★ B-2 (2026-11-01): 回収できずに残した DSPCore（leak 選択分）から診断 sink を外す。
    //   RT は StopAudio で join 済み、reader は sink を使わない（activeReaderCount は
    //   RCU 読取区間の数であって診断生産者ではない）ため、ここでの detach は安全。
    //   detach しないと、破棄済み AudioEngine の diagBuffer を指したまま leak コアが残る。
    {
        std::lock_guard<std::mutex> lock(deferredDestroyMutex_);
        for (DSPCore* core : deferredDestroy_)
        {
            if (core != nullptr)
                core->detachDiagSink();
        }
    }
    // ★ D162-2-E (E-3 / INV-D162-8): 全 EBR store drain 後の pending 残留確認。
    //   D8 の drainAll / drainAllQuarantineStore が完了した時点で DSPCore 破壊は EBR closure を
    //   全て消化しているはず。残留があれば member teardown に EBR entry を持ち越すことになり
    //   INV-D162-8 違反（観測のみ・shutdown order は変更しない）。
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    {
        const auto residualPending = m_retireRouter->pendingRetireCount();
        if (residualPending != 0)
            diagLog("[D162-2E] INV-D162-8: EBR residual pending=" + juce::String((int)residualPending)
                + " after drainAll — DSPCore destruction may carry into member teardown");
        if (shutdownSubstrateQuiescent)
            jassert(residualPending == 0); // Debug のみ assert（Release では diagLog のみ）
        // 静止を確認できない場合は残留が想定内（fail-closed で保持している）。
    }
#endif
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    {
        // ★ WORK113 変更1: 帰属の最終スナップショットを evidence/ に書く（失敗しても shutdown は続行）。
        const auto totals = convo::diag::registry().totals();
        const auto metaCounters = (m_retireRouter != nullptr)
            ? m_retireRouter->metaCounters()
            : convo::isr::ISRRetireRouter::MetaCounters {};

        juce::String json;
        json << "{\n  \"schema\": \"memory_attribution_v1\",\n";
        json << "  \"totals\": {\"liveCores\": " << static_cast<int>(totals.liveCores)
             << ", \"liveOnlyCores\": " << static_cast<int>(totals.liveOnlyCores)
             << ", \"deferredCores\": " << static_cast<int>(totals.deferredCores)
             << ", \"destroyedCores\": " << static_cast<int>(totals.destroyedCores)
             << ", \"measuredCoreBytes\": " << static_cast<juce::int64>(totals.measuredCoreBytes)
             << ", \"liveConvolvers\": " << static_cast<int>(totals.liveConvolvers)
             << ", \"measuredConvolverBytes\": " << static_cast<juce::int64>(totals.measuredConvolverBytes)
             << ", \"createdTotal\": " << static_cast<int>(totals.createdTotal)
             << ", \"destroyedTotal\": " << static_cast<int>(totals.destroyedTotal)
             << ", \"unregisteredCreated\": " << static_cast<int>(totals.unregisteredCreated)
             << ", \"unregisteredDestroyed\": " << static_cast<int>(totals.unregisteredDestroyed)
             << ", \"overflow\": " << static_cast<int>(totals.overflowCount)
             << ", \"uuidZero\": " << static_cast<int>(totals.uuidZeroCount)
             << ", \"consistent\": " << (totals.consistent() ? "true" : "false") << "},\n";
        json << "  \"meta\": {\"sized\": " << static_cast<juce::int64>(metaCounters.sized)
             << ", \"stored\": " << static_cast<juce::int64>(metaCounters.stored)
             << ", \"lostQuarantine\": " << static_cast<juce::int64>(metaCounters.lostQuarantine)
             << ", \"lostEmergency\": " << static_cast<juce::int64>(metaCounters.lostEmergency)
             << ", \"lostTerminal\": " << static_cast<juce::int64>(metaCounters.lostTerminal)
             << ", \"unsupportedProvider\": " << static_cast<juce::int64>(metaCounters.unsupportedProvider)
             << "},\n";
        json << "  " << juce::String(convo::diag::registry().recordsJson().c_str()) << "\n}\n";

        auto evidenceFile = juce::File::getCurrentWorkingDirectory()
                                .getChildFile("evidence/memory_attribution.json");
        evidenceFile.getParentDirectory().createDirectory();
        evidenceFile.replaceWithText(json);
    }
#endif
    runtimePublicationBridge_.markShutdownComplete();

    // ★ 15-P-5: Post-shutdown Faulted state diagnostic
    //   markShutdownComplete 後、Coordinator が Faulted 状態である場合は shutdown 異常を記録。
    if (runtimePublicationBridge_.getState() == convo::isr::RuntimeIntentCoordinator::CoordinatorState::Faulted)
        diagLog("[FAULT] ~AudioEngine: coordinator in Faulted state after markShutdownComplete — "
                "residual intents may remain in System 1 queues");

    // ...既存の解放処理...
    if (latencyBufOldL) { convo::aligned_free(latencyBufOldL); latencyBufOldL = nullptr; }
    if (latencyBufOldR) { convo::aligned_free(latencyBufOldR); latencyBufOldR = nullptr; }
    if (latencyBufNewL) { convo::aligned_free(latencyBufNewL); latencyBufNewL = nullptr; }
    if (latencyBufNewR) { convo::aligned_free(latencyBufNewR); latencyBufNewR = nullptr; }
    latencyBufSize = 0;
    // ★ D162-2-E (E-1 / INV-D162-6): member teardown 開始前に EQ cache を全件処分。
    //   この時点では shutdownRuntime_ / m_retireRouter / dspHandleRuntime_ 等が全て生存しているため、
    //   dspHandleRuntime_.retire + resolve + delete が安全に実行できる。
    //   呼び出し後は ~CacheMap が engine member 非依存の no-op となる（UAF 排除）。
    if (shutdownSubstrateQuiescent)
    {
        eqCacheManager.drainForShutdown();
    }
    else
    {
        // ★ WORK113: 静止を確認できないため、EQ cache の強制破棄は行わない。
        //   （~EQCacheManager は CacheMap のみを破棄し、EQCoeffCache の実体は解放しない）
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        diagLog("[WORK113] shutdown hold: EQ cache drain skipped (retained)");
#endif
    }

    // ★ WORK113 Queue/Gate/Drain: 全 retire source の処理後、最後に checked drain（終端）。
    //   fence / gate は閉じたまま（解除しない）。idle 不成立なら drain しない。
    if (shutdownSubstrateQuiescent)
    {
        const bool checkedDrainCompleted = finalizeAndDrainRetireQueueNonRt();
        juce::ignoreUnused(checkedDrainCompleted);
    }
    else
    {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        diagLog(juce::String("[DRAIN_DIAG] checkedDrain skipped=") + shutdownHoldReason);
#endif
    }

    // ★ WORK113 (2026-11-18 Owner 指示): 静止を確認できない場合、参照され得る基盤
    //   （EpochDomain / ISRRetireRouter / observer / telemetry）をプロセス寿命へ移してから
    //   engine を破棄する。停止中の reader が再開しても、publisher が破棄後に gate scope を
    //   終了しても、参照先は生存している。
    if (! shutdownSubstrateQuiescent)
    {
        abandonReaderSubstrateForShutdownNonRt();
    }

    setShutdownPhase(ShutdownPhase::Destroy, "~AudioEngine");
    convo::publishAtomic(lifecycleState, EngineLifecycleState::Destroyed, std::memory_order_release); // release: isShuttingDown の acquire と HB

    // ★ F2' (2026-10-25): ここで所有者を降りる。これ以降は DSPCore を解放しない
    //   （最後の解放は上の drain 群で完了している）。dtor 冒頭で降りると teardown 経由の
    //   解放が HL-1 の外に出るため、必ずこの位置（レビュー指摘 A）。
    // ★ R-51: 破棄専用索引からこのエンジンの項目を落とす（drain 完了後）。
    eraseDestroyIndexForOwner(this);

    {
        AudioEngine* expected = this;
        // R-46: 第 3 引数は T desired（U&& ではない）ため nullptr から T を推論できない。
        //       同ファイル内の他の呼び出しと同じく型を明示する（意味の変更なし）。
        convo::compareExchangeAtomic(s_handleInvalidationOwner, expected,
                                     static_cast<AudioEngine*>(nullptr),
                                     std::memory_order_acq_rel, std::memory_order_acquire);
    }

    diagLog("[DIAG] ~AudioEngine: shutdown sequence complete exit");

    // ★ 2026-10-30 (診断): コンストラクタのインクリメントと対称に、デストラクタの最後で減算する。
    //   ここまでで drain・索引解除・所有者解除が完了しているため、「生存中」として数える対象は
    //   実際に使用可能なインスタンスと一致する。
    (void)convo::fetchSubAtomic(s_engineInstanceCount_, 1, std::memory_order_acq_rel);
}

// [work37 Phase 9.16/9.44] 正常 publish 完了時 — RollbackToLastHealthyWorld + LearnerRollback
void AudioEngine::notifyHealthyPublication(uint64_t worldId) noexcept
{
    convo::publishAtomic(lastHealthyWorldId_, worldId, std::memory_order_release);
    convo::publishAtomic(lastHealthyPublicationTimestampUs_, convo::getCurrentTimeUs(),
                         std::memory_order_release);
    // [work37 Phase 9.44] Learner 正常状態を定期保存
    if (noiseShaperLearner && noiseShaperLearner->isRunning()) {
        convo::NoiseShaperLearnerState current;
        noiseShaperLearner->getState(current);
        lastKnownGoodNoiseShaper_.state = current;
        lastKnownGoodNoiseShaper_.timestampUs = convo::getCurrentTimeUs();
        lastKnownGoodNoiseShaper_.publicationSequence =
            convo::consumeAtomic(publicationSequenceCounter_, std::memory_order_acquire);
        lastKnownGoodNoiseShaper_.isValid = true;
    }
}
