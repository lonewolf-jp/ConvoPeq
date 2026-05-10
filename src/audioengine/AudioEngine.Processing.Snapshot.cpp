#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_SNAPSHOT)

void AudioEngine::processWithSnapshot(const juce::AudioSourceChannelInfo& bufferToFill,
                                      const convo::GlobalSnapshot* snap,
                                      bool isFadingTarget)
{
    if (snap == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    DSPCore* dsp = currentDSP.load(std::memory_order_acquire);
    if (dsp == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    const uint64_t hash = snap->eqCoeffHash;
    if (hash != debugLastAppliedEqHash.load(std::memory_order_relaxed))
    {
        debugLastAppliedEqHash.store(hash, std::memory_order_relaxed);
        debugAppliedEqHashVersion.fetch_add(1u, std::memory_order_relaxed);
    }

    const bool eqBypassed = snap->eqBypass;
    const bool convBypassed = snap->convBypass;
    const ProcessingOrder order = snap->processingOrder;
    const bool softClip = snap->softClipEnabled;
    const float satAmt = snap->saturationAmount;
    const double headroomGain = snap->inputHeadroomGain;
    const double makeupGain = snap->outputMakeupGain;
    const double convInputTrimGain = snap->convInputTrimGain;

    if (!isFadingTarget)
    {
        eqBypassActive.store(eqBypassed, std::memory_order_relaxed);
        convBypassActive.store(convBypassed, std::memory_order_relaxed);
    }

    const AnalyzerSource analyzerSource = currentAnalyzerSource.load(std::memory_order_relaxed);
    const bool analyzerEnabledNow = analyzerEnabled.load(std::memory_order_relaxed);
    const convo::HCMode hcMode = convHCFilterMode.load(std::memory_order_relaxed);
    const convo::LCMode lcMode = convLCFilterMode.load(std::memory_order_relaxed);
    const convo::HCMode lpfMode = eqLPFFilterMode.load(std::memory_order_relaxed);
    const int adaptiveCoeffBankIndex = currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
    const auto& adaptiveCoeffBank = getAdaptiveCoeffBankForIndex(adaptiveCoeffBankIndex);
    const bool adaptiveCaptureEnabled = noiseShaperLearner && noiseShaperLearner->isRunning();

    const uint32_t genSnapshot = adaptiveCoeffBank.generation.load(std::memory_order_acquire);
    const CoeffSet* safeAdaptiveSet = AudioEngine::getActiveCoeffSet(adaptiveCoeffBank);
    const uint32_t adaptiveGenAfter = genSnapshot;

    DSPCore::ProcessingState procState {
        .eqBypassed               = eqBypassed,
        .convBypassed             = convBypassed,
        .order                    = order,
        .analyzerSource           = analyzerSource,
        .analyzerEnabled          = isFadingTarget ? false : analyzerEnabledNow,
        .softClipEnabled          = softClip,
        .saturationAmount         = satAmt,
        .inputHeadroomGain        = headroomGain,
        .outputMakeupGain         = makeupGain,
        .convolverInputTrimGain   = convInputTrimGain,
        .convHCMode               = hcMode,
        .convLCMode               = lcMode,
        .eqLPFMode                = lpfMode,
        .adaptiveCoeffBankIndex   = adaptiveCoeffBankIndex,
        .adaptiveCoeffSet         = safeAdaptiveSet,
        .adaptiveCoeffGeneration  = adaptiveGenAfter,
        .adaptiveCaptureSampleRateHz = static_cast<int>(dsp->sampleRate + 0.5),
        .adaptiveCaptureBitDepth  = dsp->ditherBitDepth,
        .captureSessionId         = dsp->currentCaptureSessionId,
        .adaptiveCaptureQueue     = adaptiveCaptureEnabled ? &audioCaptureQueue : nullptr
    };

    std::atomic<float> fadingInputMeter { 0.0f };
    std::atomic<float> fadingOutputMeter { 0.0f };
    auto& inMeter = isFadingTarget ? fadingInputMeter : inputLevelLinear;
    auto& outMeter = isFadingTarget ? fadingOutputMeter : outputLevelLinear;
    dsp->process(bufferToFill, analyzerFifo, inMeter, outMeter, procState);
}

#endif
