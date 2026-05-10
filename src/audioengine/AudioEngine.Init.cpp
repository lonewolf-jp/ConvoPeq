#include <JuceHeader.h>
#include "AudioEngine.h"

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_INIT_LIFECYCLE)

void AudioEngine::initialize()
{
    firstIrDryCrossfadePending.store(false, std::memory_order_release);
    firstIrDryCrossfadeDone.store(false, std::memory_order_release);
    dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
    dspCrossfadeDryHoldSamples.store(0, std::memory_order_release);
    dspCrossfadeDryScaleGain.reset(48000.0, 0.060);  // Initialize with 60ms ramp time
    dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);

    // ==================================================================
    // 段階 1：RCU 基盤の初期化
    // ==================================================================
    // B22: 旧 SPSC キュー (queueWrite/queueRead/overflowList) は廃止。
    //      g_deletionQueue は静的初期化される。
    // readerEpochs と globalEpoch は静的初期化で 0

    // Start worker thread
    rebuildThread = std::thread(&AudioEngine::rebuildThreadLoop, this);

    // 初期DSP構築 (デフォルト設定)
    // 安全対策: バッファサイズを余裕を持って確保 (SAFE_MAX_BLOCK_SIZE)
    // これにより、デバイス初期化前やバッファサイズ変更時の不整合による音切れ/無音を防ぐ
    requestRebuild(48000.0, SAFE_MAX_BLOCK_SIZE);
    maxSamplesPerBlock.store(SAFE_MAX_BLOCK_SIZE);
    currentSampleRate.store(48000.0);

    m_fadeFloatBuffer.setSize(2, SAFE_MAX_BLOCK_SIZE, false, false, true);
    m_fadeDoubleBuffer.setSize(2, SAFE_MAX_BLOCK_SIZE, false, false, true);

    // オーディオデバイスがまだ開始していない段階でも、IRロード側には実用的な既定値を渡す。
    // SAFE_MAX_BLOCK_SIZE をそのまま使うと不要に巨大な一時NUCを組んでメモリ使用量が跳ねるため、
    // ローダー用の暫定値は一般的な 48kHz / 512samples に固定する。
    uiConvolverProcessor.prepareToPlay(48000.0, 512);

    uiConvolverProcessor.addChangeListener(this);
    uiEqEditor.addChangeListener(this);
    uiConvolverProcessor.addListener(this);

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
        key = makeDebounceKey(key, static_cast<uint64_t>(m_pendingIRChange.load(std::memory_order_acquire)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_pendingNSChange.load(std::memory_order_acquire)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_pendingAGCChange.load(std::memory_order_acquire)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentEqBypass.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentConvBypass.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentProcessingOrder.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentSoftClipEnabled.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentOversamplingFactor.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentOversamplingType.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentDitherBitDepth.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentNoiseShaperType.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentInputHeadroomDb.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentOutputMakeupDb.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentConvInputTrimDb.load(std::memory_order_relaxed)));
        key = makeDebounceKey(key, static_cast<uint64_t>(m_currentSaturationAmount.load(std::memory_order_relaxed)));

        const bool hasLastKey = hasLastEnqueuedSnapshotDebounceKey_.load(std::memory_order_acquire);
        const uint64_t lastKey = lastEnqueuedSnapshotDebounceKey_.load(std::memory_order_acquire);
        if (hasLastKey && lastKey == key)
        {
            diagLog("[VERIFY] enqueue snapshot debounced: identical snapshot intent");
            return true;
        }

        const uint64_t generation = m_generationManager.bumpGeneration();
        const convo::ParameterCommand cmd(convo::ParameterCommand::Type::ParameterChanged, generation);
        if (!m_commandBuffer.push(cmd))
        {
            DBG("AudioEngine: CommandBuffer full, dropping parameter change command");
            return false;
        }

        lastEnqueuedSnapshotDebounceKey_.store(key, std::memory_order_release);
        hasLastEnqueuedSnapshotDebounceKey_.store(true, std::memory_order_release);
        return true;
    }

    const uint64_t generation = m_generationManager.bumpGeneration();
    const convo::ParameterCommand cmd(convo::ParameterCommand::Type::ParameterChanged, generation);
    if (!m_commandBuffer.push(cmd))
    {
        DBG("AudioEngine: CommandBuffer full, dropping parameter change command");
        return false;
    }
    return true;
}

void AudioEngine::onSnapshotRequired(void* userData, uint64_t generation)
{
    auto* self = static_cast<AudioEngine*>(userData);
    if (self == nullptr)
        return;

    if (self->shutdownInProgress.load(std::memory_order_acquire))
        return;

    self->createSnapshotFromCurrentState(generation);
}

void AudioEngine::debugAssertNotAudioThread() const
{
    // Worker Thread 専用チェック。
    // 現状は簡易的に Message Thread でないことを確認する。
    // （Worker Thread は Message Thread ではないため、このチェックで十分）
    jassert(!juce::MessageManager::getInstance()->isThisTheMessageThread());
}

bool AudioEngine::waitForAudioBlockBoundary(uint64_t observedCounter, uint32_t timeoutMs) const noexcept
{
    const uint32_t startMs = juce::Time::getMillisecondCounter();
    while (!rebuildThreadShouldExit.load(std::memory_order_acquire))
    {
        if (m_audioBlockCounter.load(std::memory_order_acquire) != observedCounter)
            return true;

        if ((juce::Time::getMillisecondCounter() - startMs) >= timeoutMs)
            return false;

        juce::Thread::sleep(1);
    }

    return false;
}

#endif // CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_INIT_LIFECYCLE
