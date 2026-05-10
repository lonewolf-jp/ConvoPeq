#include <JuceHeader.h>
#include "AudioEngine.h"

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_UI_EVENTS)

void AudioEngine::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    if (source == &uiEqEditor)
    {
        // UI (SpectrumAnalyzerComponent など) が EQ 編集を即時反映できるよう通知する。
        // 実 DSP 反映は従来どおり requestRebuild() 経由で行う。
        sendChangeMessage();
        requestRebuild(convo::RebuildKind::Structural);
    }
}

void AudioEngine::convolverParamsChanged(ConvolverProcessor* processor)
{
    if (processor == &uiConvolverProcessor)
    {
        const bool suppressIntermediateMixedPhasePublish =
            uiConvolverProcessor.isProgressiveUpgradeEnabled()
            && uiConvolverProcessor.getPhaseMode() == ConvolverProcessor::PhaseMode::Mixed
            && uiConvolverProcessor.getActiveCacheFFTSize() > 0
            && uiConvolverProcessor.getActiveCacheFFTSize() < uiConvolverProcessor.getTargetUpgradeFFTSize();

        if (suppressIntermediateMixedPhasePublish)
        {
            diagLog("[DIAG] convolverParamsChanged: SUPPRESSED intermediate progressive mixed-phase publish fft="
                    + juce::String(uiConvolverProcessor.getActiveCacheFFTSize())
                    + " targetFFT=" + juce::String(uiConvolverProcessor.getTargetUpgradeFFTSize())
                    + " irName=" + uiConvolverProcessor.getIRName());
            return;
        }

        bool needsStructuralRebuild = false;
        double srForRebuild = 0.0;
        uint64_t uiStructuralHash = 0;
        bool uiHasIrForRebuild = false;
        bool dspHasIrForRebuild = false;

        {
            std::lock_guard<std::mutex> lk(rebuildMutex);
            diagLog("[DIAG] convolverParamsChanged: enter");
            if (activeDSP)
            {
                activeDSP->convolver.syncParametersFrom(uiConvolverProcessor);

                const bool uiHasIr = uiConvolverProcessor.isIRLoaded();
                const bool dspHasIr = activeDSP->convolver.isIRLoaded();
                uiHasIrForRebuild = uiHasIr;
                dspHasIrForRebuild = dspHasIr;

                if (uiHasIr)
                    uiStructuralHash = uiConvolverProcessor.getStructuralHash();

                needsStructuralRebuild = (uiHasIr != dspHasIr);

                if (!needsStructuralRebuild && uiHasIr)
                {
                    needsStructuralRebuild =
                        activeDSP->convolver.getIRName() != uiConvolverProcessor.getIRName()
                     || activeDSP->convolver.getIRLength() != uiConvolverProcessor.getIRLength()
                     || activeDSP->convolver.getPhaseMode() != uiConvolverProcessor.getPhaseMode()
                     || activeDSP->convolver.getExperimentalDirectHeadEnabled() != uiConvolverProcessor.getExperimentalDirectHeadEnabled()
                     || std::abs(activeDSP->convolver.getTargetIRLength() - uiConvolverProcessor.getTargetIRLength()) > 0.001f;
                }
            }
            else
            {
                needsStructuralRebuild = uiConvolverProcessor.isIRLoaded();
                uiHasIrForRebuild = needsStructuralRebuild;
                dspHasIrForRebuild = false;
                if (needsStructuralRebuild)
                    uiStructuralHash = uiConvolverProcessor.getStructuralHash();
            }

            if (needsStructuralRebuild)
                srForRebuild = currentSampleRate.load(std::memory_order_acquire);
        }

        // 同一構造ハッシュで再通知が来ても、重い Structural rebuild を再発火させない。
        // これにより IR 読み込み後の rebuild 連鎖（CPU スパイク）を抑止する。
        if (needsStructuralRebuild && uiStructuralHash != 0)
        {
            const uint64_t prevHash = lastIssuedConvolverStructuralHash_.load(std::memory_order_acquire);
            if (prevHash == uiStructuralHash)
            {
                diagLog("[DIAG] convolverParamsChanged: BLOCKED by hash dedup hash="
                    + juce::String::toHexString((int64_t) uiStructuralHash));
                needsStructuralRebuild = false;
            }
            else
                lastIssuedConvolverStructuralHash_.store(uiStructuralHash, std::memory_order_release);
                        diagLog("[DIAG] convolverParamsChanged: requestRebuild Structural hash="
                            + juce::String::toHexString((int64_t) uiStructuralHash)
                            + " irName=" + uiConvolverProcessor.getIRName());
        }

        if (needsStructuralRebuild && uiHasIrForRebuild && !dspHasIrForRebuild)
        {
            const int64_t nowTicks = juce::Time::getHighResolutionTicks();
            const int64_t appliedTicks = uiConvolverProcessor.getLastPreparedIRApplyTicks();
            const int64_t minDeltaTicks = juce::Time::getHighResolutionTicksPerSecond() / 5; // 200ms

            if (appliedTicks > 0 && (nowTicks - appliedTicks) < minDeltaTicks)
            {
                deferredStructuralRebuildPending_.store(true, std::memory_order_release);
                deferredStructuralRebuildDueTicks_.store(appliedTicks + minDeltaTicks, std::memory_order_release);
                pendingStructuralRebuildFromNonMT_.store(false, std::memory_order_release);
                needsStructuralRebuild = false;

                diagLog("[DIAG] convolverParamsChanged: DEFERRED Structural rebuild after prepared IR apply and cleared pending Structural bit");
            }
        }

        if (needsStructuralRebuild)
        {
            requestRebuild(convo::RebuildKind::Structural);
        }

        if (needsStructuralRebuild && srForRebuild > 0.0)
        {
            ++pendingIRGeneration;
            setIRChangeFlag();

            const LearningCommand cmd {
                LearningCommand::Type::IRChanged,
                false,
                pendingLearningMode.load(std::memory_order_acquire),
                pendingIRGeneration
            };

            if (!enqueueLearningCommand(cmd))
            {
                DBG("[AudioEngine] convolverParamsChanged: command queue overflow");
            }
        }
    }
}

#endif // CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_UI_EVENTS
