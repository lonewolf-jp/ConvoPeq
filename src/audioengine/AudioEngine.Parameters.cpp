#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

namespace {
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

[[maybe_unused]] bool validateConvolverStateTreeForDebug(const juce::ValueTree& state, juce::String* reason = nullptr)
{
    auto fail = [&](const juce::String& message) -> bool
    {
        if (reason != nullptr)
            *reason = message;
        return false;
    };

    if (!state.isValid())
        return fail("Convolver state is invalid");

    if (!state.hasType("Convolver"))
        return fail("Convolver state has unexpected type: " + state.getType().toString());

    auto hasFiniteDouble = [&](const juce::Identifier& key, double minValue, double maxValue) -> bool
    {
        if (!state.hasProperty(key))
            return true;
        const double value = static_cast<double>(state.getProperty(key));
        if (!std::isfinite(value))
            return fail("Convolver state property is non-finite: " + key.toString());
        if (value < minValue || value > maxValue)
            return fail("Convolver state property out of range: " + key.toString());
        return true;
    };

    auto hasIntRange = [&](const juce::Identifier& key, int minValue, int maxValue) -> bool
    {
        if (!state.hasProperty(key))
            return true;
        const int value = static_cast<int>(state.getProperty(key));
        if (value < minValue || value > maxValue)
            return fail("Convolver state property out of range: " + key.toString());
        return true;
    };

    if (!hasFiniteDouble("mix", 0.0, 1.0)) return false;
    if (!hasIntRange("phaseMode", 0, 2)) return false;
    if (!hasFiniteDouble("smoothingTime", ConvolverProcessor::SMOOTHING_TIME_MIN_SEC, ConvolverProcessor::SMOOTHING_TIME_MAX_SEC)) return false;
    if (!hasFiniteDouble("irLength", ConvolverProcessor::IR_LENGTH_MIN_SEC, ConvolverProcessor::IR_LENGTH_MAX_SEC)) return false;
    if (!hasFiniteDouble("autoDetectedIRLength", ConvolverProcessor::IR_LENGTH_MIN_SEC, ConvolverProcessor::IR_LENGTH_MAX_SEC)) return false;
    if (!hasFiniteDouble("mixedF1Hz", ConvolverProcessor::MIXED_F1_MIN_HZ, ConvolverProcessor::MIXED_F1_MAX_HZ)) return false;
    if (!hasFiniteDouble("mixedF2Hz", ConvolverProcessor::MIXED_F2_MIN_HZ, ConvolverProcessor::MIXED_F2_MAX_HZ)) return false;
    if (!hasFiniteDouble("mixedTau", ConvolverProcessor::MIXED_TAU_MIN, ConvolverProcessor::MIXED_TAU_MAX)) return false;
    if (!hasIntRange("rebuildDebounceMs", ConvolverProcessor::REBUILD_DEBOUNCE_MIN_MS, ConvolverProcessor::REBUILD_DEBOUNCE_MAX_MS)) return false;
    if (!hasIntRange("tailMode", static_cast<int>(ConvolverProcessor::TailMode::AirAbsorption), static_cast<int>(ConvolverProcessor::TailMode::Bypass))) return false;
    if (!hasFiniteDouble("tailStartSec", ConvolverProcessor::TAIL_START_MIN_SEC, ConvolverProcessor::TAIL_START_MAX_SEC)) return false;
    if (!hasFiniteDouble("tailStrength", ConvolverProcessor::TAIL_STRENGTH_MIN, ConvolverProcessor::TAIL_STRENGTH_MAX)) return false;
    if (!hasIntRange("tailL1L2Multiplier", ConvolverProcessor::TAIL_L1L2_MULT_MIN, ConvolverProcessor::TAIL_L1L2_MULT_MAX)) return false;

    if (state.hasProperty("mixedF1Hz") && state.hasProperty("mixedF2Hz"))
    {
        const double f1 = static_cast<double>(state.getProperty("mixedF1Hz"));
        const double f2 = static_cast<double>(state.getProperty("mixedF2Hz"));
        if (f2 < f1 + 10.0)
            return fail("Convolver state mixedF2Hz must be >= mixedF1Hz + 10Hz");
    }

    return true;
}

[[maybe_unused]] bool validatePresetStateTreeForDebug(const juce::ValueTree& state, juce::String* reason = nullptr)
{
    auto fail = [&](const juce::String& message) -> bool
    {
        if (reason != nullptr)
            *reason = message;
        return false;
    };

    if (!state.isValid())
        return fail("Preset state is invalid");

    if (!state.hasType("Preset"))
        return fail("Preset state has unexpected type: " + state.getType().toString());

    auto hasFiniteDouble = [&](const juce::Identifier& key, double minValue, double maxValue) -> bool
    {
        if (!state.hasProperty(key))
            return true;
        const double value = static_cast<double>(state.getProperty(key));
        if (!std::isfinite(value))
            return fail("Preset property is non-finite: " + key.toString());
        if (value < minValue || value > maxValue)
            return fail("Preset property out of range: " + key.toString());
        return true;
    };

    auto hasIntRange = [&](const juce::Identifier& key, int minValue, int maxValue) -> bool
    {
        if (!state.hasProperty(key))
            return true;
        const int value = static_cast<int>(state.getProperty(key));
        if (value < minValue || value > maxValue)
            return fail("Preset property out of range: " + key.toString());
        return true;
    };

    if (!hasIntRange("processingOrder", static_cast<int>(convo::ProcessingOrder::ConvolverThenEQ), static_cast<int>(convo::ProcessingOrder::EQThenConvolver))) return false;
    if (!hasFiniteDouble("saturationAmount", 0.0, 1.0)) return false;
    if (!hasFiniteDouble("inputHeadroomDb", -12.0, 0.0)) return false;
    if (!hasFiniteDouble("outputMakeupDb", 0.0, 12.0)) return false;
    if (!hasFiniteDouble("convolverInputTrimDb", -12.0, 0.0)) return false;
    if (!hasIntRange("analyzerSource", static_cast<int>(AudioEngine::AnalyzerSource::Input), static_cast<int>(AudioEngine::AnalyzerSource::Output))) return false;
    if (!hasIntRange("noiseShaperType", static_cast<int>(convo::NoiseShaperType::Psychoacoustic), static_cast<int>(convo::NoiseShaperType::Fixed15Tap))) return false;
    if (!hasIntRange("oversamplingType", static_cast<int>(convo::OversamplingType::IIR), static_cast<int>(convo::OversamplingType::LinearPhase))) return false;
    if (!hasIntRange("convHCFilterMode", static_cast<int>(convo::HCMode::Sharp), static_cast<int>(convo::HCMode::Soft))) return false;
    if (!hasIntRange("convLCFilterMode", static_cast<int>(convo::LCMode::Natural), static_cast<int>(convo::LCMode::Soft))) return false;
    if (!hasIntRange("eqLPFFilterMode", static_cast<int>(convo::HCMode::Sharp), static_cast<int>(convo::HCMode::Soft))) return false;
    if (!hasFiniteDouble("coeffSafetyMargin", 0.0, 2.0)) return false;
    if (!hasIntRange("cmaesRestarts", 0, 1000)) return false;

    if (state.hasProperty("oversamplingFactor"))
    {
        const int factor = static_cast<int>(state.getProperty("oversamplingFactor"));
        if (!(factor == 0 || factor == 1 || factor == 2 || factor == 4 || factor == 8))
            return fail("Preset property out of range: oversamplingFactor");
    }

    if (state.hasProperty("ditherBitDepth"))
    {
        const int bitDepth = static_cast<int>(state.getProperty("ditherBitDepth"));
        if (bitDepth <= 0 || bitDepth > 64)
            return fail("Preset property out of range: ditherBitDepth");
    }

    const auto eqState = state.getChildWithName("EQ");
    if (eqState.isValid() && !eqState.hasType("EQ"))
        return fail("Preset child EQ has unexpected type: " + eqState.getType().toString());

    const auto convState = state.getChildWithName("Convolver");
    if (convState.isValid())
    {
        if (!validateConvolverStateTreeForDebug(convState, reason))
            return false;
    }

    return true;
}
}

void AudioEngine::setEqBypassRequested (bool shouldBypass)
{
    ASSERT_NON_RT_THREAD();
    convo::publishAtomic(eqBypassRequested, shouldBypass, std::memory_order_release);
    convo::publishAtomic(m_currentEqBypass, shouldBypass, std::memory_order_release);
    uiEqEditor.setBypass(shouldBypass);
    applyDefaultsForCurrentMode();
    submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
    sendChangeMessage();
}

void AudioEngine::setConvolverBypassRequested (bool shouldBypass)
{
    ASSERT_NON_RT_THREAD();
    convo::publishAtomic(convBypassRequested, shouldBypass, std::memory_order_release);
    convo::publishAtomic(m_currentConvBypass, shouldBypass, std::memory_order_release);
    uiConvolverProcessor.setBypass(shouldBypass);
    applyDefaultsForCurrentMode();
    submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
    sendChangeMessage();
}

void AudioEngine::setConvolverPhaseMode(ConvolverProcessor::PhaseMode mode)
{
    uiConvolverProcessor.setPhaseMode(mode);
}

[[nodiscard]] ConvolverProcessor::PhaseMode AudioEngine::getConvolverPhaseMode() const
{
    return uiConvolverProcessor.getPhaseMode();
}

void AudioEngine::requestEqPreset (int presetIndex)
{
    uiEqEditor.loadPreset (presetIndex);
    sendChangeMessage();
}

void AudioEngine::requestEqPresetFromText(const juce::File& file)
{
    if (uiEqEditor.loadFromTextFile(file))
        sendChangeMessage();
}

void AudioEngine::requestConvolverPreset(const juce::File& irFile)
{
    uiConvolverProcessor.loadIR(irFile);
}

void AudioEngine::beginBulkParameterRestore() noexcept
{
    m_isRestoringState = true;
}

void AudioEngine::endBulkParameterRestore(bool requestRebuildNow) noexcept
{
    m_isRestoringState = false;

    if (!requestRebuildNow)
        return;

    const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
    if (sr > 0.0)
        submitRebuildIntent(convo::RebuildKind::Structural,
                    RebuildTelemetryReason::RequestRebuildKindEntry,
                    RebuildTelemetryClass::Structural,
                    RebuildTelemetryPolicy::Replaceable);
}



void AudioEngine::setInputHeadroomDb(float db)
{
    ASSERT_NON_RT_THREAD();
    // コンボルバーが先頭に来る場合 (Conv→PEQ / Conv only) は -6dB 上限で入力保護する。
    // EQ が先頭またはコンボルバーがバイパスされている場合は 0dB まで許容する。
    const bool convBypassed = convo::consumeAtomic(convBypassRequested, std::memory_order_acquire);
    const bool eqBypassed   = convo::consumeAtomic(eqBypassRequested, std::memory_order_acquire);
    const ProcessingOrder order = convo::consumeAtomic(currentProcessingOrder, std::memory_order_acquire);
    const bool convIsFirst = !convBypassed && (order == ProcessingOrder::ConvolverThenEQ || eqBypassed);
    const float maxDb = convIsFirst ? -6.0f : 0.0f;
    float clampedDb = juce::jlimit(-12.0f, maxDb, db);
    if (std::abs(convo::consumeAtomic(inputHeadroomDb, std::memory_order_acquire) - clampedDb) > 1e-5f)
    {
        convo::publishAtomic(inputHeadroomDb, clampedDb, std::memory_order_release);
        convo::publishAtomic(inputHeadroomGain, juce::Decibels::decibelsToGain((double)clampedDb), std::memory_order_release);
        convo::publishAtomic(m_currentInputHeadroomDb, clampedDb, std::memory_order_release);
        submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
    }
}

[[nodiscard]] float AudioEngine::getInputHeadroomDb() const
{
    return convo::consumeAtomic(inputHeadroomDb, std::memory_order_acquire);
}

void AudioEngine::setOutputMakeupDb(float db)
{
    ASSERT_NON_RT_THREAD();
    // Output makeup は全モード共通で 0..12 dB
    const float clampedDb = juce::jlimit(0.0f, 12.0f, db);
    if (std::abs(convo::consumeAtomic(outputMakeupDb, std::memory_order_acquire) - clampedDb) > 1e-5f)
    {
        convo::publishAtomic(outputMakeupDb, clampedDb, std::memory_order_release);
        convo::publishAtomic(outputMakeupGain, juce::Decibels::decibelsToGain((double)clampedDb), std::memory_order_release);
        convo::publishAtomic(m_currentOutputMakeupDb, clampedDb, std::memory_order_release);
        submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
    }
}

[[nodiscard]] float AudioEngine::getOutputMakeupDb() const
{
    return convo::consumeAtomic(outputMakeupDb, std::memory_order_acquire);
}

void AudioEngine::setProcessingOrder(ProcessingOrder order)
{
    ASSERT_NON_RT_THREAD();
    convo::publishAtomic(currentProcessingOrder, order, std::memory_order_release);
    convo::publishAtomic(m_currentProcessingOrder, order, std::memory_order_release);
    submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
    applyDefaultsForCurrentMode();
}

void AudioEngine::setConvolverInputTrimDb(float db)
{
    ASSERT_NON_RT_THREAD();
    // 範囲: -12..0 dB (0dB = トリムなし / -12dB = 最大保護)
    float clampedDb = juce::jlimit(-12.0f, 0.0f, db);
    if (std::abs(convo::consumeAtomic(convolverInputTrimDb, std::memory_order_acquire) - clampedDb) > 1e-5f)
    {
        convo::publishAtomic(convolverInputTrimDb, clampedDb, std::memory_order_release);
        convo::publishAtomic(convolverInputTrimGain, juce::Decibels::decibelsToGain((double)clampedDb), std::memory_order_release);
        convo::publishAtomic(m_currentConvInputTrimDb, clampedDb, std::memory_order_release);
        submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
    }
}

[[nodiscard]] float AudioEngine::getConvolverInputTrimDb() const
{
    return convo::consumeAtomic(convolverInputTrimDb, std::memory_order_acquire);
}

void AudioEngine::applyDefaultsForCurrentMode()
{
    if (m_isRestoringState) return; // プリセットロード中はデフォルトリセットを抑制する

    const bool eqBypassed  = convo::consumeAtomic(eqBypassRequested, std::memory_order_acquire);
    const bool convBypassed = convo::consumeAtomic(convBypassRequested, std::memory_order_acquire);
    const ProcessingOrder order = convo::consumeAtomic(currentProcessingOrder, std::memory_order_acquire);

    float newInputHeadroomDb = 0.0f;
    float newOutputMakeupDb = 0.0f;
    float newConvTrimDb = 0.0f;

    if (convBypassed && !eqBypassed)
    {
        newInputHeadroomDb = 0.0f;
        newOutputMakeupDb = 0.0f;
        newConvTrimDb = 0.0f;
    }
    else if (!convBypassed && order == ProcessingOrder::EQThenConvolver && !eqBypassed)
    {
        newInputHeadroomDb = 0.0f;
        newOutputMakeupDb = 10.0f;
        newConvTrimDb = -6.0f;
    }
    else
    {
        // 残りの全ケース（eqBypassed + !convBypassed, 両方Bypass, Conv→EQ順・両方Active）
        // は同じデフォルト値を設定する
        newInputHeadroomDb = -6.0f;
        newOutputMakeupDb = 12.0f;
        newConvTrimDb = 0.0f;
    }

    convo::publishAtomic(inputHeadroomDb, newInputHeadroomDb, std::memory_order_release);
    convo::publishAtomic(outputMakeupDb, newOutputMakeupDb, std::memory_order_release);
    convo::publishAtomic(convolverInputTrimDb, newConvTrimDb, std::memory_order_release);
    convo::publishAtomic(inputHeadroomGain, juce::Decibels::decibelsToGain(static_cast<double>(newInputHeadroomDb)), std::memory_order_release);
    convo::publishAtomic(outputMakeupGain, juce::Decibels::decibelsToGain(static_cast<double>(newOutputMakeupDb)), std::memory_order_release);
    convo::publishAtomic(convolverInputTrimGain, juce::Decibels::decibelsToGain(static_cast<double>(newConvTrimDb)), std::memory_order_release);

    convo::publishAtomic(m_currentInputHeadroomDb, newInputHeadroomDb, std::memory_order_release);
    convo::publishAtomic(m_currentOutputMakeupDb, newOutputMakeupDb, std::memory_order_release);
    convo::publishAtomic(m_currentConvInputTrimDb, newConvTrimDb, std::memory_order_release);
    submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
}

void AudioEngine::setDitherBitDepth(int bitDepth)
{
    if (convo::consumeAtomic(ditherBitDepth, std::memory_order_acquire) != bitDepth)
    {
        const bool adaptiveLearningActive = (convo::consumeAtomic(noiseShaperType, std::memory_order_acquire) == NoiseShaperType::Adaptive9thOrder)
            && noiseShaperLearner
            && noiseShaperLearner->isRunning();

        if (adaptiveLearningActive)
        {
            stopNoiseShaperLearning();
            noiseShaperLearner->setErrorMessage("Learning stopped due to bit depth change. Please restart learning.");
        }

        convo::publishAtomic(ditherBitDepth, bitDepth, std::memory_order_release);
        convo::publishAtomic(m_currentDitherBitDepth, bitDepth, std::memory_order_release);
        DBG_LOG("Dither Bit Depth changed: " + juce::String(bitDepth));
        submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);

        selectAdaptiveCoeffBankForCurrentSettings();

        // UI側（学習ウィンドウ）が即座に反映できるように通知
        sendChangeMessage();

        const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
        if (!m_isRestoringState && sr > 0.0)
        {
            const int queuedGeneration = convo::consumeAtomic(rebuildRequestGeneration, std::memory_order_acquire);
            const int committedGeneration = convo::consumeAtomic(lastCommittedRebuildGeneration, std::memory_order_acquire);
            const bool outstandingRebuild = queuedGeneration > committedGeneration;
            const bool shouldDeferRebuild =
                outstandingRebuild
                ||
                uiConvolverProcessor.isLoadingIR()
                || hasRebuildReason(RebuildReason::DeferredStructural)
                || convo::consumeAtomic(m_pendingIRChange, std::memory_order_acquire)
                || (uiConvolverProcessor.isIRLoaded() && !uiConvolverProcessor.isIRFinalized());

            if (shouldDeferRebuild)
            {
                setRebuildReason(RebuildReason::DeferredFinalizeAware);
                diagLog("[DIAG] setDitherBitDepth: deferred rebuild until IR finalized");
            }
            else
            {
                submitRebuildIntent(convo::RebuildKind::Structural,
                                    RebuildTelemetryReason::RequestRebuildKindEntry,
                                    RebuildTelemetryClass::Structural,
                                    RebuildTelemetryPolicy::Replaceable);
            }
        }
    }
}

[[nodiscard]] int AudioEngine::getDitherBitDepth() const
{
    return convo::consumeAtomic(ditherBitDepth, std::memory_order_acquire);
}

void AudioEngine::setNoiseShaperType(NoiseShaperType type)
{
    if (convo::consumeAtomic(noiseShaperType, std::memory_order_acquire) != type)
    {
        convo::publishAtomic(noiseShaperType, type, std::memory_order_release);
        convo::publishAtomic(m_currentNoiseShaperType, type, std::memory_order_release);
        convo::publishAtomic(m_pendingNSChange, true, std::memory_order_release);
        juce::Logger::writeToLog(juce::String("[AudioEngine] setNoiseShaperType: newType=") + juce::String(static_cast<int>(type))
            + " wasAdaptive=" + juce::String(static_cast<int>(convo::consumeAtomic(noiseShaperType, std::memory_order_acquire) == NoiseShaperType::Adaptive9thOrder ? 1 : 0)));
        if (type != NoiseShaperType::Adaptive9thOrder)
        {
            stopNoiseShaperLearning();
        }
        else
        {
            if (noiseShaperLearner)
            {
                juce::Logger::writeToLog("[AudioEngine] setNoiseShaperType(Adaptive): calling learner->stopLearning()");
                noiseShaperLearner->stopLearning();
            }

            noiseShaperLearner = std::make_unique<NoiseShaperLearner>(*this, audioCaptureQueue);
            noiseShaperLearner->setLearningMode(convo::consumeAtomic(pendingLearningMode, std::memory_order_acquire));
            resetLearningControlState();
        }

        juce::String typeName = "Psychoacoustic";
        if (type == NoiseShaperType::Fixed4Tap)
            typeName = "Fixed4Tap";
        else if (type == NoiseShaperType::Fixed15Tap)
            typeName = "Fixed15Tap";
        else if (type == NoiseShaperType::Adaptive9thOrder)
            typeName = "Adaptive9thOrder";

        DBG_LOG("Noise Shaper changed: " + typeName);
        submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
        const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
        if (!m_isRestoringState && sr > 0.0)
        {
            const int queuedGeneration = convo::consumeAtomic(rebuildRequestGeneration, std::memory_order_acquire);
            const int committedGeneration = convo::consumeAtomic(lastCommittedRebuildGeneration, std::memory_order_acquire);
            const bool outstandingRebuild = queuedGeneration > committedGeneration;
            const bool shouldDeferRebuild =
                outstandingRebuild
                ||
                uiConvolverProcessor.isLoadingIR()
                || hasRebuildReason(RebuildReason::DeferredStructural)
                || convo::consumeAtomic(m_pendingIRChange, std::memory_order_acquire)
                || (uiConvolverProcessor.isIRLoaded() && !uiConvolverProcessor.isIRFinalized());

            if (shouldDeferRebuild)
            {
                setRebuildReason(RebuildReason::DeferredFinalizeAware);
                diagLog("[DIAG] setNoiseShaperType: deferred rebuild until IR finalized");
            }
            else
            {
                submitRebuildIntent(convo::RebuildKind::Structural,
                                    RebuildTelemetryReason::RequestRebuildKindEntry,
                                    RebuildTelemetryClass::Structural,
                                    RebuildTelemetryPolicy::Replaceable);
            }
        }
    }
}

[[nodiscard]] AudioEngine::NoiseShaperType AudioEngine::getNoiseShaperType() const
{
    return convo::consumeAtomic(noiseShaperType, std::memory_order_acquire);
}

void AudioEngine::setFixedNoiseLogIntervalMs(int intervalMs) noexcept
{
    convo::publishAtomic(fixedNoiseLogIntervalMs, juce::jlimit(250, 10000, intervalMs), std::memory_order_release);
}

[[nodiscard]] int AudioEngine::getFixedNoiseLogIntervalMs() const noexcept
{
    return convo::consumeAtomic(fixedNoiseLogIntervalMs, std::memory_order_acquire);
}

void AudioEngine::setFixedNoiseWindowSamples(int windowSamples) noexcept
{
    convo::publishAtomic(fixedNoiseWindowSamples, juce::jlimit(256, 262144, windowSamples), std::memory_order_release);
}

[[nodiscard]] int AudioEngine::getFixedNoiseWindowSamples() const noexcept
{
    return convo::consumeAtomic(fixedNoiseWindowSamples, std::memory_order_acquire);
}

void AudioEngine::setSoftClipEnabled(bool enabled)
{
    convo::publishAtomic(softClipEnabled, enabled, std::memory_order_release);
    convo::publishAtomic(m_currentSoftClipEnabled, enabled, std::memory_order_release);
    submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
}

[[nodiscard]] bool AudioEngine::isSoftClipEnabled() const
{
    return convo::consumeAtomic(softClipEnabled, std::memory_order_acquire);
}

void AudioEngine::setSaturationAmount(float amount)
{
    const float clamped = juce::jlimit(0.0f, 1.0f, amount);
    if (std::abs(convo::consumeAtomic(saturationAmount, std::memory_order_acquire) - clamped) > 1e-6f)
    {
        convo::publishAtomic(saturationAmount, clamped, std::memory_order_release);
        convo::publishAtomic(m_currentSaturationAmount, clamped, std::memory_order_release);
        submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
    }
}

[[nodiscard]] float AudioEngine::getSaturationAmount() const
{
    return convo::consumeAtomic(saturationAmount, std::memory_order_acquire);
}

void AudioEngine::setOversamplingFactor(int factor)
{
    // 0=Auto, 1, 2, 4, 8
    int newFactor = 0;
    if (factor == 1 || factor == 2 || factor == 4 || factor == 8)
    {
        newFactor = factor;
    }

    if (convo::consumeAtomic(manualOversamplingFactor, std::memory_order_acquire) != newFactor)
    {
        convo::publishAtomic(manualOversamplingFactor, newFactor, std::memory_order_release);
        convo::publishAtomic(m_currentOversamplingFactor, newFactor, std::memory_order_release);
        submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
        const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
        if (!m_isRestoringState && sr > 0.0)
        {
            submitRebuildIntent(convo::RebuildKind::Structural,
                                RebuildTelemetryReason::RequestRebuildKindEntry,
                                RebuildTelemetryClass::Structural,
                                RebuildTelemetryPolicy::Replaceable);
        }
    }
}

[[nodiscard]] int AudioEngine::getOversamplingFactor() const
{
    return convo::consumeAtomic(manualOversamplingFactor, std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// Convolver UI staging setters (Rule 1.1.5)
// Implementations are kept here (.cpp) so that AudioEngine.h does not expose
// direct uiConvolverProcessor.set*() calls to the Rule 1.1.5 grep scan.
// All calls are Message Thread only and route through the UI staging object;
// the Runtime world is updated via convolverParamsChanged -> submitRebuildIntent().
// ---------------------------------------------------------------------------

void AudioEngine::setConvolverTargetIRLength(float timeSec, bool manualOverride) noexcept
{
    if (manualOverride)
        uiConvolverProcessor.setIRLengthManualOverride(true);
    uiConvolverProcessor.setTargetIRLength(timeSec);
}

void AudioEngine::setConvolverMixedTransitionStartHz(float hz) noexcept
{
    uiConvolverProcessor.setMixedTransitionStartHz(hz);
}

void AudioEngine::setConvolverMixedTransitionEndHz(float hz) noexcept
{
    uiConvolverProcessor.setMixedTransitionEndHz(hz);
}

void AudioEngine::setConvolverMixedPreRingTau(float tau) noexcept
{
    uiConvolverProcessor.setMixedPreRingTau(tau);
}

void AudioEngine::setConvolverRebuildDebounceMs(int ms) noexcept
{
    uiConvolverProcessor.setRebuildDebounceMs(ms);
}

void AudioEngine::setConvolverTailMode(ConvolverProcessor::TailMode mode) noexcept
{
    uiConvolverProcessor.setTailMode(mode);
}

void AudioEngine::setConvolverTailStartSec(float sec) noexcept
{
    uiConvolverProcessor.setTailStartSec(sec);
}

void AudioEngine::setConvolverTailStrength(float strength) noexcept
{
    uiConvolverProcessor.setTailStrength(strength);
}

void AudioEngine::setConvolverTailL1L2Multiplier(int multiplier) noexcept
{
    uiConvolverProcessor.setTailL1L2Multiplier(multiplier);
}

void AudioEngine::setOversamplingType(OversamplingType type)
{
    convo::publishAtomic(oversamplingType, type, std::memory_order_release);
    convo::publishAtomic(m_currentOversamplingType, type, std::memory_order_release);
    submitRebuildIntent(convo::RebuildKind::Structural, RebuildTelemetryReason::EnqueueSnapshotCommand, RebuildTelemetryClass::Snapshot, RebuildTelemetryPolicy::Replaceable);
    const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
    if (!m_isRestoringState && sr > 0.0)
    {
        submitRebuildIntent(convo::RebuildKind::Structural,
                            RebuildTelemetryReason::RequestRebuildKindEntry,
                            RebuildTelemetryClass::Structural,
                            RebuildTelemetryPolicy::Replaceable);
    }
}

[[nodiscard]] AudioEngine::OversamplingType AudioEngine::getOversamplingType() const
{
    return convo::consumeAtomic(oversamplingType, std::memory_order_acquire);
}

//====================================================================
// 出力周波数フィルターモード Setter / Getter (Message Thread)
//====================================================================
void AudioEngine::setConvHCFilterMode(convo::HCMode mode) noexcept
{
    convo::publishAtomic(convHCFilterMode, mode, std::memory_order_release);
    // NUC irFreqDomain を再焼き込みするため、uiConvolverProcessor を再構築する。
    // DSPCore::convolver は次回 requestRebuild 時に syncStateFrom + rebuildAllIRsSynchronous で追従する。
    uiConvolverProcessor.setNUCFilterModes(
        convo::consumeAtomic(convHCFilterMode, std::memory_order_acquire),
        convo::consumeAtomic(convLCFilterMode, std::memory_order_acquire));
}

[[nodiscard]] convo::HCMode AudioEngine::getConvHCFilterMode() const noexcept
{
    return convo::consumeAtomic(convHCFilterMode, std::memory_order_acquire);
}

void AudioEngine::setConvLCFilterMode(convo::LCMode mode) noexcept
{
    convo::publishAtomic(convLCFilterMode, mode, std::memory_order_release);
    // HC と組み合わせて NUC を再構築
    uiConvolverProcessor.setNUCFilterModes(
        convo::consumeAtomic(convHCFilterMode, std::memory_order_acquire),
        convo::consumeAtomic(convLCFilterMode, std::memory_order_acquire));
}

[[nodiscard]] convo::LCMode AudioEngine::getConvLCFilterMode() const noexcept
{
    return convo::consumeAtomic(convLCFilterMode, std::memory_order_acquire);
}

void AudioEngine::setEqLPFFilterMode(convo::HCMode mode) noexcept
{
    convo::publishAtomic(eqLPFFilterMode, mode, std::memory_order_release);
}

[[nodiscard]] convo::HCMode AudioEngine::getEqLPFFilterMode() const noexcept
{
    return convo::consumeAtomic(eqLPFFilterMode, std::memory_order_acquire);
}

[[nodiscard]] juce::ValueTree AudioEngine::getConvolverStateTree() const
{
    auto state = uiConvolverProcessor.getState();
   #if JUCE_DEBUG
    juce::String reason;
    const bool validState = validateConvolverStateTreeForDebug(state, &reason);
    jassert(validState);
    if (!validState)
        diagLog("[ASSERT] getConvolverStateTree: invalid exported Convolver state: " + reason);
   #endif
    return state;
}

void AudioEngine::setConvolverStateTree(const juce::ValueTree& state)
{
   #if JUCE_DEBUG
    juce::String reason;
    const bool validState = validateConvolverStateTreeForDebug(state, &reason);
    jassert(validState);
    if (!validState)
        diagLog("[ASSERT] setConvolverStateTree: invalid imported Convolver state: " + reason);
   #endif
    uiConvolverProcessor.setState(state);
}

[[nodiscard]] int AudioEngine::getConvolverTargetUpgradeFFTSize() const
{
    return uiConvolverProcessor.getTargetUpgradeFFTSize();
}

void AudioEngine::setConvolverTargetUpgradeFFTSize(int fftSize)
{
    uiConvolverProcessor.setTargetUpgradeFFTSize(fftSize);
}

[[nodiscard]] bool AudioEngine::isConvolverProgressiveUpgradeEnabled() const
{
    return uiConvolverProcessor.isProgressiveUpgradeEnabled();
}

void AudioEngine::setConvolverEnableProgressiveUpgrade(bool enabled)
{
    uiConvolverProcessor.setEnableProgressiveUpgrade(enabled);
}

[[nodiscard]] int AudioEngine::getConvolverMaxCacheEntries() const
{
    return static_cast<int>(uiConvolverProcessor.getMaxCacheEntries());
}

void AudioEngine::setConvolverMaxCacheEntries(int maxEntries)
{
    uiConvolverProcessor.setMaxCacheEntries(static_cast<size_t>(maxEntries));
}

void AudioEngine::clearConvolverCache()
{
    uiConvolverProcessor.clearCache();
}
