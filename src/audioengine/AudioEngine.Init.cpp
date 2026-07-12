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
    {
        convo::RuntimeBuilder bootstrapBuilder(*this);
        bootstrapBuilder.setHealthStateRef(getHealthStateRef());
        auto bootstrapWorld = bootstrapBuilder.createBootstrapWorld();
        auto coordinator = makeRuntimePublicationCoordinator();
        const auto result = commitRuntimePublication(coordinator, std::move(bootstrapWorld),
                                 RegistrationContext::none());
        juce::ignoreUnused(result);
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
