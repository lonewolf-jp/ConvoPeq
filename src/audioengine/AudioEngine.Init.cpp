#include <JuceHeader.h>
#include "AudioEngine.h"

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
    requestRebuild(48000.0, SAFE_MAX_BLOCK_SIZE);
    convo::publishAtomic(maxSamplesPerBlock, SAFE_MAX_BLOCK_SIZE, std::memory_order_release); // release: process の acquire と HB
    convo::publishAtomic(currentSampleRate, 48000.0, std::memory_order_release); // release: process/loader の acquire と HB

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

    m_workerThread.setSnapshotCreator(&AudioEngine::onSnapshotRequired, this);
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

bool AudioEngine::enqueueSnapshotCommand() noexcept
{
    constexpr const char* kPhase5TagReduce = "phase5_reduce_target";
    constexpr const char* kPhase5TagKeep = "phase5_keep_target";

    const uint64_t intentId = nextRebuildTelemetryIntentId();
    emitRebuildTelemetry(RebuildTelemetryEvent::Requested,
                         intentId,
                         RebuildTelemetryReason::EnqueueSnapshotCommand,
                         RebuildTelemetryDecision::Accepted,
                         0,
                         0,
                         RebuildTelemetryClass::Snapshot,
                         RebuildTelemetryPolicy::NA);

    auto makeDebounceKey = [](uint64_t seed, uint64_t value) noexcept -> uint64_t
    {
        // 64-bit mix (no libm, no allocation)
        constexpr uint64_t kMul = 0x9E3779B185EBCA87ull;
        seed ^= value + kMul + (seed << 6) + (seed >> 2);
        return seed;
    };

    const auto* mm = juce::MessageManager::getInstanceWithoutCreating();
    if (mm != nullptr && mm->isThisTheMessageThread())
    {
        uint64_t eqHash = 0;
        if (const auto* eqState = uiEqEditor.getEQStateSnapshot())
            eqHash = EQProcessor::computeParamsHash(eqState->toEQParameters());

        uint64_t key = 0xD6E8FEB86659FD93ull;
        key = makeDebounceKey(key, eqHash);
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_pendingIRChange, std::memory_order_acquire))); // acquire: setIRChangeFlag の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_pendingNSChange, std::memory_order_acquire))); // acquire: setNSChangeFlag の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_pendingAGCChange, std::memory_order_acquire))); // acquire: setAGCChangeFlag の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentEqBypass, std::memory_order_acquire))); // acquire: setEQBypass の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentConvBypass, std::memory_order_acquire))); // acquire: setConvolverBypass の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentProcessingOrder, std::memory_order_acquire))); // acquire: setProcessingOrder の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentSoftClipEnabled, std::memory_order_acquire))); // acquire: setSoftClipEnabled の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentOversamplingFactor, std::memory_order_acquire))); // acquire: setOversamplingFactor の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentOversamplingType, std::memory_order_acquire))); // acquire: setOversamplingType の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentDitherBitDepth, std::memory_order_acquire))); // acquire: setDitherBitDepth の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentNoiseShaperType, std::memory_order_acquire))); // acquire: setNoiseShaperType の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentInputHeadroomDb, std::memory_order_acquire))); // acquire: setInputHeadroomDb の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentOutputMakeupDb, std::memory_order_acquire))); // acquire: setOutputMakeupDb の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentConvInputTrimDb, std::memory_order_acquire))); // acquire: setConvolverInputTrimDb の publishAtomic release と HB
        key = makeDebounceKey(key, static_cast<uint64_t>(convo::consumeAtomic(m_currentSaturationAmount, std::memory_order_acquire))); // acquire: setSaturationAmount の publishAtomic release と HB

        const bool hasLastKey = convo::consumeAtomic(rtAuxMutable_.hasLastEnqueuedSnapshotDebounceKey, std::memory_order_acquire); // acquire: publishAtomic release と HB
        const uint64_t lastKey = convo::consumeAtomic(rtAuxMutable_.lastEnqueuedSnapshotDebounceKey, std::memory_order_acquire); // acquire: publishAtomic release と HB
        if (hasLastKey && lastKey == key)
        {
            diagLog("[VERIFY] enqueue snapshot debounced: identical snapshot intent");
            emitRebuildTelemetry(RebuildTelemetryEvent::Merged,
                                 intentId,
                                 RebuildTelemetryReason::SnapshotIntentDebounced,
                                 RebuildTelemetryDecision::Merged,
                                 0,
                                 0,
                                 RebuildTelemetryClass::Snapshot,
                                 RebuildTelemetryPolicy::Replaceable,
                                 kPhase5TagReduce);
            return true;
        }

        const uint64_t generation = m_generationManager.bumpGeneration();
        const convo::ParameterCommand cmd(convo::ParameterCommand::Type::ParameterChanged, generation);
        if (!m_commandBuffer.push(cmd))
        {
            DBG("AudioEngine: CommandBuffer full, dropping parameter change command");
            emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                                 intentId,
                                 RebuildTelemetryReason::SnapshotCommandBufferFull,
                                 RebuildTelemetryDecision::Dropped,
                                 0,
                                 0,
                                 RebuildTelemetryClass::Snapshot,
                                 RebuildTelemetryPolicy::NA,
                                 kPhase5TagKeep);
            return false;
        }

        convo::publishAtomic(rtAuxMutable_.lastEnqueuedSnapshotDebounceKey, key, std::memory_order_release); // release: consume 次回 acquire と HB
        convo::publishAtomic(rtAuxMutable_.hasLastEnqueuedSnapshotDebounceKey, true, std::memory_order_release); // release: consume 次回 acquire と HB
        emitRebuildTelemetry(RebuildTelemetryEvent::Dispatched,
                     intentId,
                     RebuildTelemetryReason::SnapshotCommandQueued,
                     RebuildTelemetryDecision::Dispatched,
                     0,
                     0,
                     RebuildTelemetryClass::Snapshot,
                     RebuildTelemetryPolicy::NA);
        return true;
    }

    const uint64_t generation = m_generationManager.bumpGeneration();
    const convo::ParameterCommand cmd(convo::ParameterCommand::Type::ParameterChanged, generation);
    if (!m_commandBuffer.push(cmd))
    {
        DBG("AudioEngine: CommandBuffer full, dropping parameter change command");
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                             intentId,
                             RebuildTelemetryReason::SnapshotCommandBufferFullNonMt,
                             RebuildTelemetryDecision::Dropped,
                             0,
                             0,
                             RebuildTelemetryClass::Snapshot,
                             RebuildTelemetryPolicy::NA,
                             kPhase5TagKeep);
        return false;
    }
    emitRebuildTelemetry(RebuildTelemetryEvent::Dispatched,
                         intentId,
                         RebuildTelemetryReason::SnapshotCommandQueuedNonMt,
                         RebuildTelemetryDecision::Dispatched,
                         0,
                         0,
                         RebuildTelemetryClass::Snapshot,
                         RebuildTelemetryPolicy::NA);
    return true;
}

void AudioEngine::onSnapshotRequired(void* userData, uint64_t generation)
{
    auto* self = static_cast<AudioEngine*>(userData);
    if (self == nullptr)
        return;

    if (self->isShutdownInProgress())
        return;

    self->createSnapshotFromCurrentState(generation);
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
