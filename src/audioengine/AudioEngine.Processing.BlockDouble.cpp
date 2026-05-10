#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"
#include "core/RCUReader.h"

static thread_local convo::RCUReader tls_rcuReader;

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
    const juce::ScopedNoDenormals noDenormals;
    m_audioBlockCounter.fetch_add(1, std::memory_order_release);

    // ★ 追加: RCU ガードで現在の DSP を保護する
    convo::RCUReaderGuard rcuGuard(tls_rcuReader);
    const int numSamples = buffer.getNumSamples();
    // 事前サニティチェック (getNextAudioBlock と同様)
    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20;
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        buffer.clear();
        return;
    }

    DSPCore* dsp = currentDSP.load(std::memory_order_acquire);
    if (dsp == nullptr)
    {
        buffer.clear();
        return;
    }

    // AudioThread入口で、現在のDSPが持つ全てのNUCのガードをチェック（デバッグ時のみ）
        #ifdef NUC_DEBUG_GUARDS
        {
        dsp->convolver.debugCheckNucGuards();
        }
    #endif

    // --- ProcessingStateを現行設計で初期化 ---
    const bool eqBypassed = eqBypassActive.load(std::memory_order_relaxed);
    const bool convBypassed = convBypassActive.load(std::memory_order_relaxed);
    const ProcessingOrder order = currentProcessingOrder.load(std::memory_order_relaxed);
    const bool softClip = softClipEnabled.load(std::memory_order_relaxed);
    const float satAmount = saturationAmount.load(std::memory_order_relaxed);
    const double headroomGain = inputHeadroomGain.load(std::memory_order_relaxed);
    const double makeupGain = outputMakeupGain.load(std::memory_order_relaxed);
    const double convInputTrimGain = convolverInputTrimGain.load(std::memory_order_relaxed);
    const bool adaptiveCaptureEnabled = noiseShaperLearner && noiseShaperLearner->isRunning();

    DSPCore::ProcessingState procState = buildAudioThreadProcessingState(dsp,
                                                                         eqBypassed,
                                                                         convBypassed,
                                                                         order,
                                                                         softClip,
                                                                         satAmount,
                                                                         headroomGain,
                                                                         makeupGain,
                                                                         convInputTrimGain,
                                                                         adaptiveCaptureEnabled);

    // DSPCore 固有の上限チェック (getNextAudioBlock と同様)
    if (numSamples > dsp->maxSamplesPerBlock)
    {
        buffer.clear();
        return;
    }

    float snapshotAlpha = 1.0f;
    const convo::GlobalSnapshot* snapshotFrom = nullptr;
    const convo::GlobalSnapshot* snapshotTo = nullptr;
    updateAudioThreadSnapshotFade(numSamples, snapshotAlpha, snapshotFrom, snapshotTo);
    (void) snapshotAlpha;

    const double engineSampleRate = currentSampleRate.load(std::memory_order_relaxed);
    if (absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
    {
        inputLevelLinear.store(0.0f);
        outputLevelLinear.store(0.0f);
        buffer.clear();
        return;
    }

    // --- クロスフェード開始時: スナップショット取得・RT競合ゼロ設計 ---
    DSPCore* fading = sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire));
    bool useDryAsOld = dspCrossfadeUseDryAsOld.load(std::memory_order_acquire);
    if (processCrossfadeDelayGateIfPending(fading,
                                           useDryAsOld,
                                           [&]()
    {
        auto fadingState = makeCrossfadeAuxState(procState);

        std::atomic<float> fadingInputMeter { 0.0f };
        std::atomic<float> fadingOutputMeter { 0.0f };
        fading->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, fadingState);
    }))
    {
        return;
    }

    armCrossfadeIfPending(dsp, fading != nullptr, useDryAsOld);

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
        dsp->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

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
        dsp->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);
        cleanupCrossfadeDirectPath(fading);
    }
}

#endif

