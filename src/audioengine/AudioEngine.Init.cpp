#include <JuceHeader.h>
#include "AudioEngine.h"
#include "RuntimeBuilder.h"

namespace {
[[maybe_unused]] void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

void AudioEngine::initialize()
{
    crossfadeRuntime_.reset();
    publishLatencyDelayAtomics(0, 0);
    convo::publishAtomic(latencyResetPending, false, std::memory_order_release);
    resetLatencyDelayRtState();
    crossfadeRuntime_.getGain().reset(48000.0, 0.03);
    crossfadeRuntime_.getGain().setCurrentAndTargetValue(1.0);
    crossfadeRuntime_.getDryScaleGain().reset(48000.0, 0.060);
    crossfadeRuntime_.getDryScaleGain().setCurrentAndTargetValue(1.0);
    refreshCrossfadePreparedSnapshotFromAtomics();

    // ==================================================================
    // 段階 1：RCU 基盤の初期化
    // ==================================================================
    // B22: 旧 SPSC キュー (queueWrite/queueRead/overflowList) は廃止。
    //      deferred reclaim は EpochDomain 配下で初期化済み。
    // readerEpochs と globalEpoch は静的初期化で 0

    // Start worker thread
    rebuildThread = std::thread(&AudioEngine::rebuildThreadLoop, this);

    // 初期DSP構築 (デフォルト設定)
    // 安全対策: バッファサイズを余裕を持って確保 (kInitialPrepareMaxBlock)
    // これにより、デバイス初期化前やバッファサイズ変更時の不整合による音切れ/無音を防ぐ。
    // ★ v8.3: SAFE_MAX_BLOCK_SIZE(65536) ではなく kInitialPrepareMaxBlock(4096) を使用。
    //   prepareToPlay 到達後は実ブロックサイズに更新される。
    //   prepareToPlay 前に rebuild が走った場合の初回確保量を削減する。
    constexpr int kInitialPrepareMaxBlock = 4096;
    convo::publishAtomic(maxSamplesPerBlock, kInitialPrepareMaxBlock, std::memory_order_release); // release: process の acquire と HB
    convo::publishAtomic(currentSampleRate, 48000.0, std::memory_order_release); // release: process/loader の acquire と HB

    // Bootstrap World: publish BEFORE submitting rebuild intent, so that
    // the rebuild worker always finds a non-null runtimeWorld when building.
    // ★ B4-a3: Bootstrap は同期例外。initialize() の Bootstrap publish は CoordinatorLoop
    //   起動（startCoordinatorLoop）より前で、非同期 enqueue + 完了通知待ちは待機対象が
    //   存在せずデッドロックするため、worldAuthority_.publish() で同期 publish する
    //   （X4-B 後は RuntimeWorldAuthority が sole physical publish gateway — INV-X4-2）。
    {
        convo::RuntimeBuilder bootstrapBuilder(*this);
        auto bootstrapWorld = bootstrapBuilder.createBootstrapWorld();
        // ★ work88 (X4-B §6.4 / X4-B-6): Bootstrap publish を RuntimeWorldAuthority 経由に一本化
        //   （sole physical publish gateway — INV-X4-2）。一時生成 makeRuntimePublishAuthority() は
        //   X4-B-8 で廃止。createBootstrapWorld は自前 freeze 済みのため seal は冪等。
        //   validate / didPublish / willRetire / retire は Bridge（従来 publishWorld 内部）で実行。
        const_cast<RuntimePublishWorld*>(bootstrapWorld.get())->sealRecursively();
        RuntimePublicationBridge bootstrapBridge{ *this, runtimePublicationValidator_ };
        if (!bootstrapBridge.validatePublicationNonRt(*bootstrapWorld))
        {
            // ★ work88 (§4.5 / R5): bootstrap world が Rejected の場合の Debug 早期診断。
            //   旧 publishWorld は PublishStageResult::Rejected/Failed を返した（dash §4.5）。
            //   X4-B 後は validate 失敗（Rejected 相当）でこの分岐に入る — Debug のみ検出。
            jassertfalse;
            auto* rejectedWorld = const_cast<RuntimePublishWorld*>(bootstrapWorld.release());
            bootstrapBridge.retireRejectedRuntimeWorldNonRt(rejectedWorld);
        }
        else
        {
            const auto* bootstrapWorldPtr = bootstrapWorld.get();
            bool committed = false;
            auto* oldWorld = worldAuthority_.publish(std::move(bootstrapWorld),
                convo::isr::RuntimeWorldAuthority::PublishMetadata{
                    convo::isr::RuntimeBoundary::NonRTWorld,
                    bootstrapWorldPtr->generation,
                    bootstrapWorldPtr->publication.sequenceId,
                    bootstrapWorldPtr->publication.epoch,
                    bootstrapWorldPtr->publication.mappedRuntimeGeneration },
                &committed);
            // ★ work88（監査軽微指摘2）: publish() 成功時のみ didPublish/willRetire/retire。
            //   Bootstrap は validate 済みのため通常失敗しないが、失敗時は world が publish() 内で
            //   破棄されるため *bootstrapWorldPtr を deref しない（dangling deref 防止）。
            if (committed)
            {
                bootstrapBridge.didPublishRuntimeNonRt(*bootstrapWorldPtr);
                bootstrapBridge.willRetireRuntimeNonRt(oldWorld);
                bootstrapBridge.retirePublishedRuntimeWorldNonRt(oldWorld, false);
            }
        }
    }

    // Now submit rebuild intent — the worker will find a valid Bootstrap World.
    submitRebuildIntent(convo::RebuildKind::Structural,
                        RebuildTelemetryReason::RequestRebuildKindEntry,
                        RebuildTelemetryClass::Structural,
                        RebuildTelemetryPolicy::Replaceable);

    m_fadeFloatBuffer.setSize(2, SAFE_MAX_BLOCK_SIZE, false, false, true);
    m_fadeDoubleBuffer.setSize(2, SAFE_MAX_BLOCK_SIZE, false, false, true);

    // オーディオデバイスがまだ開始していない段階でも、IRロード側には実用的な既定値を渡す。

    // ★ work60: モジュール別 DiagEvent リングバッファポインタを初期化
    //   DSPCoreFloat/DSPCoreDouble の logEqTime → eqDiagBuffer
    //   ConvolverProcessor.Runtime の convDiagBuffer
    setEqDiagBuffer(diagBuffer, rtAuxMutable_.diagTickPushed,
                    rtAuxMutable_.diagTickDropped, rtAuxMutable_.diagTotalPushed);
    setConvDiagBuffer(diagBuffer, rtAuxMutable_.diagTickPushed,
                      rtAuxMutable_.diagTickDropped, rtAuxMutable_.diagTotalPushed);
    // SAFE_MAX_BLOCK_SIZE をそのまま使うと不要に巨大な一時NUCを組んでメモリ使用量が跳ねるため、
    // ローダー用の暫定値は一般的な 48kHz / 512samples に固定する。
    uiConvolverProcessor.prepareToPlay(48000.0, 512);

    uiConvolverProcessor.addChangeListener(this);
    uiEqEditor.addChangeListener(this);

    // タイマー開始 (100ms間隔)
    // - DSP再構築リクエストのポーリング (Audio Threadからの依頼を処理)
    // - ガベージコレクション
    startTimer(100);
    timerPeriodMs_ = 100;
    startCoordinatorLoop();  // ★ FUTURE-9: Dedicated Coordinator Worker starts ISR cadence

    // ★ [work64] ThreadAffinityManager 初期化（動的計算）
    {
        ThreadAffinityMasks affinityMasks{};
        auto topo = ThreadAffinityManager::detectCoreTopology();

        if (topo.physicalCoreCount == 0) {
            // ★ v16: API 失敗 → アフィニティ無効
            hasHeterogeneousCores_ = false;
            diagLog("[AFFINITY] GetLogicalProcessorInformationEx failed: Affinity disabled.");
        } else if (topo.hasHeterogeneousArchitecture) {
            // P/E混在 → MMCSS Deadline QoS に委任
            hasHeterogeneousCores_ = true;
            diagLog("[AFFINITY] P/E heterogeneous cores (N="
                    + juce::String(topo.physicalCoreCount)
                    + "). Affinity disabled — MMCSS Deadline QoS active.");
        } else {
            // 対称コア → 末尾1物理コアをAudio専用に
            affinityMasks = ThreadAffinityManager::computeSymmetricMasks(topo);
            hasHeterogeneousCores_ = false;
            diagLog("[AFFINITY] Symmetric cores (N="
                    + juce::String(topo.physicalCoreCount)
                    + "). Audio pinned to last physical core.");
        }

        affinityManager.initialize(affinityMasks);

        // ★ v14/v21: 起動時診断ログ — nonAudioMask は affinityMasks の実フィールドから計算
        //   （P/E環境では全マスクがゼロで正しく表示される）
        {
            DWORD_PTR nonAudioMask = 0;
            nonAudioMask |= affinityMasks.worker;
            nonAudioMask |= affinityMasks.learnerMain;
            nonAudioMask |= affinityMasks.learnerEvalBase;
            nonAudioMask |= affinityMasks.heavyBackground;
            nonAudioMask |= affinityMasks.lightBackground;
            nonAudioMask |= affinityMasks.ui;

            diagLog("[AFFINITY] coreTopology: physical=" + juce::String(topo.physicalCoreCount)
                + " logical=" + juce::String(::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS))
                + " heterogeneous=" + juce::String(hasHeterogeneousCores_ ? "true" : "false"));
            diagLog("[AFFINITY] audioMask=0x" + juce::String::toHexString(static_cast<uint64_t>(affinityMasks.audioRealtime))
                + " nonAudio=0x" + juce::String::toHexString(static_cast<uint64_t>(nonAudioMask))
                + " worker=0x" + juce::String::toHexString(static_cast<uint64_t>(affinityMasks.worker))
                + " learner=0x" + juce::String::toHexString(static_cast<uint64_t>(affinityMasks.learnerMain))
                + " heavyBG=0x" + juce::String::toHexString(static_cast<uint64_t>(affinityMasks.heavyBackground))
                + " lightBG=0x" + juce::String::toHexString(static_cast<uint64_t>(affinityMasks.lightBackground))
                + " ui=0x" + juce::String::toHexString(static_cast<uint64_t>(affinityMasks.ui)));
        }
    }

    // ★ [work64] 順序入替（v7）: initialize() の後で WorkerThread を起動
    initWorkerThread();
}

void AudioEngine::initWorkerThread()
{
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    m_workerThread.start();
    affinityManager.applyCurrentThreadPolicy(ThreadType::Worker);
}

void AudioEngine::shutdownWorkerThread()
{
    m_workerThread.stop();
}

void AudioEngine::debugAssertNotAudioThread() const
{
    // Control path 共通チェック。
    // Message Thread / Worker Thread は許可し、Audio Thread のみ禁止する。
    jassert(!convo::numeric_policy::isAudioThread());
}

void AudioEngine::debugAssertAudioThread() const
{
    // Audio Thread 専用チェック。
    jassert(convo::numeric_policy::isAudioThread());
}
