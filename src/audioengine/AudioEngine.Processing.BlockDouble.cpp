#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"
#include "core/RCUReader.h"

namespace
{
    inline double absDiffNoLibm(double a, double b) noexcept
    {
        return absNoLibm(a - b);
    }
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_BLOCK_DOUBLE)
void AudioEngine::processBlockDouble (juce::AudioBuffer<double>& buffer)
{
    constexpr int kAudioEpochReaderIndex = 0;
    const juce::ScopedNoDenormals noDenormals;
    const convo::numeric_policy::ScopedThreadRole audioThreadScope(convo::numeric_policy::ThreadRole::AudioRealtime);
    const convo::EpochCoreReaderGuard epochReaderGuard(m_epochCore, kAudioEpochReaderIndex);
    ASSERT_AUDIO_THREAD();
    ++m_audioBlockCounterRtLocal; // RT-local counter (non-atomic, RT thread only)

    // ★ 追加: RCU ガードで現在の DSP を保護する
    convo::RCUReaderGuard rcuGuard(audioThreadRcuReader);
    const int numSamples = buffer.getNumSamples();
    // 事前サニティチェック (getNextAudioBlock と同様)
    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20;
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        applySafeSilentFallback(buffer);
        return;
    }

    const auto runtimePublishView = getRuntimePublishView();
    const auto* runtimeGraph = runtimePublishView.graph;
    DSPCore* dsp = resolveCurrentDSPFromRuntimePublish(runtimeGraph);
    if (dsp == nullptr)
    {
        applySafeSilentFallback(buffer);
        return;
    }

    // AudioThread入口で、現在のDSPが持つ全てのNUCのガードをチェック（デバッグ時のみ）
        #ifdef NUC_DEBUG_GUARDS
        {
        dsp->convolver.debugCheckNucGuards();
        }
    #endif

    // --- ProcessingStateを現行設計で初期化 ---
    const convo::GlobalSnapshot* snap = m_coordinator.getCurrent();
    const EngineParameterSnapshot parameterSnapshot = captureAudioThreadParameterSnapshot(snap);

    DSPCore::ProcessingState procState = buildAudioThreadProcessingState(dsp, parameterSnapshot);

    // DSPCore 固有の上限チェック (getNextAudioBlock と同様)
    if (numSamples > dsp->maxSamplesPerBlock)
    {
        applySafeSilentFallback(buffer);
        return;
    }

    float snapshotAlpha = 1.0f;
    const convo::GlobalSnapshot* snapshotFrom = nullptr;
    const convo::GlobalSnapshot* snapshotTo = nullptr;
    updateAudioThreadSnapshotFade(numSamples, snapshotAlpha, snapshotFrom, snapshotTo);
    (void) snapshotAlpha;

    const double engineSampleRate = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
    if (absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
    {
        applySafeSilentFallback(buffer);
        return;
    }

    // --- クロスフェード開始時: スナップショット取得・RT競合ゼロ設計 ---
    DSPCore* fading = resolveFadingDSPFromRuntimePublish(runtimeGraph);
    bool useDryAsOld = runtimeCrossfadeUseDryAsOld(runtimeGraph);
    const bool hasPendingCrossfade = runtimeCrossfadePending(runtimeGraph);
    if (processCrossfadeDelayGateIfPending(fading,
                                           useDryAsOld,
                                           hasPendingCrossfade,
                                           [&]()
    {
        auto fadingState = makeCrossfadeAuxState(procState);

        std::atomic<float> fadingInputMeter { 0.0f };
        std::atomic<float> fadingOutputMeter { 0.0f };
        fading->processDouble(buffer,
                      analyzerFifo,
                      inputLevelLinear,
                      outputLevelLinear,
                      fadingState);
    }))
    {
        return;
    }

    armCrossfadeIfPending(fading != nullptr, useDryAsOld, runtimeGraph);

    const bool canCrossfade = (fading != nullptr || useDryAsOld)
        && dspCrossfadeGain.isSmoothing()
        && dspCrossfadeDoubleBuffer.getNumChannels() >= 2
        && dspCrossfadeDoubleBuffer.getNumSamples() >= numSamples;

    if (canCrossfade)
    {
        // --- wrap安全・スナップショット設計 ---
        dspCrossfadeDoubleBuffer.clear(0, 0, numSamples);
        dspCrossfadeDoubleBuffer.clear(1, 0, numSamples);

        auto fadingState = makeCrossfadeAuxState(procState);

        std::atomic<float> fadingInputMeter { 0.0f };
        std::atomic<float> fadingOutputMeter { 0.0f };
        if (useDryAsOld)
        {
            const int outChannels = std::min(2, buffer.getNumChannels());
            if (outChannels > 0)
                juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(0, 0), buffer.getReadPointer(0, 0), numSamples);
            if (outChannels > 1)
                juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(1, 0), buffer.getReadPointer(1, 0), numSamples);
        }
        else
        {
            // EBR: managed by RCUReader
            fading->processDoubleToBuffer(buffer, dspCrossfadeDoubleBuffer, analyzerFifo,
                                          fadingInputMeter, fadingOutputMeter, fadingState);
        }
        dsp->processDouble(buffer,
                   analyzerFifo,
                   inputLevelLinear,
                   outputLevelLinear,
                   procState);

        // スナップショット（commitNewDSPでセット済み、ここでは読み取り専用）
        const int outChannels = std::min(2, buffer.getNumChannels());
        double* dstL = (outChannels > 0) ? buffer.getWritePointer(0, 0) : nullptr;
        double* dstR = (outChannels > 1) ? buffer.getWritePointer(1, 0) : nullptr;
        const double* oldL = (outChannels > 0) ? dspCrossfadeDoubleBuffer.getReadPointer(0, 0) : nullptr;
        const double* oldR = (outChannels > 1) ? dspCrossfadeDoubleBuffer.getReadPointer(1, 0) : nullptr;

        runLatencyAlignedCrossfadeMixLoop<double>(dstL,
                                                  dstR,
                                                  oldL,
                                                  oldR,
                                                  numSamples,
                                                                  runtimeGraph,
                                                  [](double* outL,
                                                     double* outR,
                                                     int i,
                                                     double gNew,
                                                     double alignedOldL,
                                                     double alignedOldR,
                                                     double alignedNewL,
                                                     double alignedNewR)
                                                  {
                                                      const double gOld = 1.0 - gNew;
                                                      if (outL != nullptr) outL[i] = alignedNewL * gNew + alignedOldL * gOld;
                                                      if (outR != nullptr) outR[i] = alignedNewR * gNew + alignedOldR * gOld;
                                                  });
        if (!useDryAsOld)
        {
            // EBR: managed by RCUReader
        }

        finalizeCrossfadeMixPath(false);
    }
    else
    {
        dsp->processDouble(buffer,
                           analyzerFifo,
                           inputLevelLinear,
                           outputLevelLinear,
                           procState);
        cleanupCrossfadeDirectPath(fading);
    }
}

#endif

