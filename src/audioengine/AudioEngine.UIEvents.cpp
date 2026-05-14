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

            const bool uiHasIr = uiConvolverProcessor.isIRLoaded();
            const bool committedHasIr = convo::consumeAtomic(lastCommittedConvolverHasIr_, std::memory_order_acquire);
            uiHasIrForRebuild = uiHasIr;
            dspHasIrForRebuild = committedHasIr;

            if (uiHasIr)
                uiStructuralHash = uiConvolverProcessor.getStructuralHash();

            needsStructuralRebuild = (uiHasIr != committedHasIr);

            if (!needsStructuralRebuild && uiHasIr)
            {
                const uint64_t committedStructuralHash = convo::consumeAtomic(lastCommittedConvolverStructuralHash_, std::memory_order_acquire);
                needsStructuralRebuild = (uiStructuralHash != committedStructuralHash);
            }

            if (needsStructuralRebuild)
                srForRebuild = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
        }

        // 同一構造ハッシュで再通知が来ても、重い Structural rebuild を再発火させない。
        // これにより IR 読み込み後の rebuild 連鎖（CPU スパイク）を抑止する。
        if (needsStructuralRebuild && uiStructuralHash != 0)
        {
            const uint64_t prevHash = convo::consumeAtomic(lastIssuedConvolverStructuralHash_, std::memory_order_acquire);
            if (prevHash == uiStructuralHash)
            {
                diagLog("[DIAG] convolverParamsChanged: BLOCKED by hash dedup hash="
                    + juce::String::toHexString((int64_t) uiStructuralHash));
                needsStructuralRebuild = false;
            }
            else
                convo::publishAtomic(lastIssuedConvolverStructuralHash_, uiStructuralHash, std::memory_order_release);
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
                setRebuildReason(RebuildReason::DeferredStructural);
                convo::publishAtomic(deferredStructuralRebuildDueTicks_, appliedTicks + minDeltaTicks, std::memory_order_release);
                clearRebuildReason(RebuildReason::StructuralFromNonMT);
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
                convo::consumeAtomic(pendingLearningMode, std::memory_order_acquire),
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
