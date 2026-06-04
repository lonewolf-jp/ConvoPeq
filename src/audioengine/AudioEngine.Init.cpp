#include <JuceHeader.h>
#include "AudioEngine.h"
#include "RuntimeBuilder.h"

namespace {
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

void AudioEngine::initialize()
{
    convo::publishAtomic(dspCrossfadePending, false, std::memory_order_release); // release: process の acquire と HB
    convo::publishAtomic(dspCrossfadeStartDelayBlocks, 0, std::memory_order_release); // release: process の acquire と HB
    convo::publishAtomic(firstIrDryCrossfadePending, false, std::memory_order_release); // release: process の acquire と HB
    convo::publishAtomic(firstIrDryCrossfadeDone, false, std::memory_order_release); // release: process の acquire と HB
    convo::publishAtomic(dspCrossfadeUseDryAsOld, false, std::memory_order_release); // release: process の acquire と HB
    convo::publishAtomic(dspCrossfadeDryHoldSamples, 0, std::memory_order_release); // release: process の acquire と HB
    publishLatencyDelayAtomics(0, 0);
    convo::publishAtomic(latencyResetPending, false, std::memory_order_release); // release: process の acquire と HB
    convo::publishAtomic(queuedFadeTimeSec, 0.03, std::memory_order_release); // release: process の acquire と HB
    resetLatencyDelayRtState();

    dspCrossfadeGain.reset(48000.0, 0.03);
    dspCrossfadeGain.setCurrentAndTargetValue(1.0);
    dspCrossfadeDryScaleGain.reset(48000.0, 0.060);  // Initialize with 60ms ramp time
    dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
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
    // 安全対策: バッファサイズを余裕を持って確保 (SAFE_MAX_BLOCK_SIZE)
    // これにより、デバイス初期化前やバッファサイズ変更時の不整合による音切れ/無音を防ぐ
    convo::publishAtomic(maxSamplesPerBlock, SAFE_MAX_BLOCK_SIZE, std::memory_order_release); // release: process の acquire と HB
    convo::publishAtomic(currentSampleRate, 48000.0, std::memory_order_release); // release: process/loader の acquire と HB

    // Bootstrap World: publish BEFORE submitting rebuild intent, so that
    // the rebuild worker always finds a non-null runtimeWorld when building.
    {
        convo::RuntimeBuilder bootstrapBuilder(*this);
        auto bootstrapWorld = bootstrapBuilder.createBootstrapWorld();
        auto coordinator = makeRuntimePublicationCoordinator();
        coordinator.publishWorld(std::move(bootstrapWorld));
    }

    // Now submit rebuild intent — the worker will find a valid Bootstrap World.
    submitRebuildIntent(convo::RebuildKind::Structural,
                        RebuildTelemetryReason::RequestRebuildKindEntry,
                        RebuildTelemetryClass::Structural,
                        RebuildTelemetryPolicy::Replaceable);

    m_fadeFloatBuffer.setSize(2, SAFE_MAX_BLOCK_SIZE, false, false, true);
    m_fadeDoubleBuffer.setSize(2, SAFE_MAX_BLOCK_SIZE, false, false, true);

    // オーディオデバイスがまだ開始していない段階でも、IRロード側には実用的な既定値を渡す。
    // SAFE_MAX_BLOCK_SIZE をそのまま使うと不要に巨大な一時NUCを組んでメモリ使用量が跳ねるため、
    // ローダー用の暫定値は一般的な 48kHz / 512samples に固定する。
    uiConvolverProcessor.prepareToPlay(48000.0, 512);

    uiConvolverProcessor.addChangeListener(this);
    uiEqEditor.addChangeListener(this);

    // タイマー開始 (100ms間隔)
    // - DSP再構築リクエストのポーリング (Audio Threadからの依頼を処理)
    // - ガベージコレクション
    startTimer(100);

    initWorkerThread();
}

void AudioEngine::initWorkerThread()
{
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    m_workerThread.start();
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
