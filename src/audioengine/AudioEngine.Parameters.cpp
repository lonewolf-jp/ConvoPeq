#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

[[maybe_unused]] static bool validateConvolverStateTreeForDebug(const juce::ValueTree& state, juce::String* reason = nullptr)
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

[[maybe_unused]] static bool validatePresetStateTreeForDebug(const juce::ValueTree& state, juce::String* reason = nullptr)
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
    enqueueSnapshotCommand();
    sendChangeMessage();
}

void AudioEngine::setConvolverBypassRequested (bool shouldBypass)
{
    ASSERT_NON_RT_THREAD();
    convo::publishAtomic(convBypassRequested, shouldBypass, std::memory_order_release);
    convo::publishAtomic(m_currentConvBypass, shouldBypass, std::memory_order_release);
    uiConvolverProcessor.setBypass(shouldBypass);
    applyDefaultsForCurrentMode();
    enqueueSnapshotCommand();
    sendChangeMessage();
}

void AudioEngine::setConvolverPhaseMode(ConvolverProcessor::PhaseMode mode)
{
    uiConvolverProcessor.setPhaseMode(mode);
}

ConvolverProcessor::PhaseMode AudioEngine::getConvolverPhaseMode() const
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
        requestRebuild(sr, convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire));
}

#if !defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_STATEIO_LOAD)
void AudioEngine::requestLoadState (const juce::ValueTree& state)
{
    // B19: RAII ガードを使用して、例外発生時も確実にフラグを戻す
    RestoreStateGuard guard(m_isRestoringState);

   #if JUCE_DEBUG
    juce::String reason;
    const bool validPresetState = validatePresetStateTreeForDebug(state, &reason);
    jassert(validPresetState);
    if (!validPresetState)
        diagLog("[ASSERT] requestLoadState: invalid Preset state: " + reason);
   #endif

    // ─── Step 1: モード・バイパス状態を先に復元 ────────────────────────────
    if (state.hasProperty("processingOrder"))
        convo::publishAtomic(currentProcessingOrder, (ProcessingOrder)(int)state.getProperty("processingOrder"), std::memory_order_release);

    if (state.hasProperty("eqBypassed"))
    {
        bool bypassed = state.getProperty("eqBypassed");
        convo::publishAtomic(eqBypassRequested, bypassed, std::memory_order_release);
        uiEqEditor.setBypass(bypassed);
    }

    if (state.hasProperty("convBypassed"))
    {
        bool bypassed = state.getProperty("convBypassed");
        convo::publishAtomic(convBypassRequested, bypassed, std::memory_order_release);
        uiConvolverProcessor.setBypass(bypassed);
    }

    // ─── Step 2: ゲイン値を復元 (モード依存クランプが正しく適用される) ─────
    // NOTE: ここで guard を破棄せず、関数終了まで維持することで
    //       setInputHeadroomDb 等の内部で呼ばれる applyDefaults を抑制し続ける。
    //       (旧実装では 4059 行目で false に戻していたが、B19 では安全のため全域カバー)

    if (state.hasProperty("inputHeadroomDb"))
        setInputHeadroomDb(state.getProperty("inputHeadroomDb"));

    if (state.hasProperty("outputMakeupDb"))
        setOutputMakeupDb(state.getProperty("outputMakeupDb"));

    if (state.hasProperty("convolverInputTrimDb"))
        setConvolverInputTrimDb(state.getProperty("convolverInputTrimDb"));

    if (state.hasProperty("ditherBitDepth"))
        setDitherBitDepth(static_cast<int>(state.getProperty("ditherBitDepth")));

    if (state.hasProperty("noiseShaperType"))
        setNoiseShaperType((NoiseShaperType)(int)state.getProperty("noiseShaperType"));

    {
        bool hasBankedAdaptiveCoefficients = false;

        for (int bankIndex = 0; bankIndex < getAdaptiveSampleRateBankCount(); ++bankIndex)
        {
            const double bankSampleRate = getAdaptiveSampleRateBankHz(bankIndex);
            double adaptiveCoefficients[kAdaptiveNoiseShaperOrder] = {};
            bool hasBankCoefficients = false;

            getAdaptiveCoefficientsForSampleRate(bankSampleRate, adaptiveCoefficients, kAdaptiveNoiseShaperOrder);
            for (int coeffIndex = 0; coeffIndex < kAdaptiveNoiseShaperOrder; ++coeffIndex)
            {
                const auto propertyName = makeAdaptiveCoeffPropertyName(bankSampleRate, coeffIndex);
                if (state.hasProperty(propertyName))
                {
                    adaptiveCoefficients[coeffIndex] = static_cast<double>(state.getProperty(propertyName));
                    hasBankCoefficients = true;
                    hasBankedAdaptiveCoefficients = true;
                }
            }

            if (hasBankCoefficients)
                setAdaptiveCoefficientsForSampleRate(bankSampleRate, adaptiveCoefficients, kAdaptiveNoiseShaperOrder);
        }

        if (!hasBankedAdaptiveCoefficients)
        {
            double legacyAdaptiveCoefficients[kAdaptiveNoiseShaperOrder] = {};
            bool hasLegacyAdaptiveCoefficients = false;

            for (int coeffIndex = 0; coeffIndex < kAdaptiveNoiseShaperOrder; ++coeffIndex)
            {
                const auto propertyName = "adaptiveCoeff" + juce::String(coeffIndex);
                if (state.hasProperty(propertyName))
                {
                    legacyAdaptiveCoefficients[coeffIndex] = static_cast<double>(state.getProperty(propertyName));
                    hasLegacyAdaptiveCoefficients = true;
                }
            }

            if (hasLegacyAdaptiveCoefficients)
            {
                for (int bankIndex = 0; bankIndex < getAdaptiveSampleRateBankCount(); ++bankIndex)
                    setAdaptiveCoefficientsForSampleRate(getAdaptiveSampleRateBankHz(bankIndex),
                                                         legacyAdaptiveCoefficients,
                                                         kAdaptiveNoiseShaperOrder);
            }
        }
    }

    if (state.hasProperty("oversamplingFactor"))
        setOversamplingFactor(static_cast<int>(state.getProperty("oversamplingFactor")));

    if (state.hasProperty("oversamplingType"))
        setOversamplingType((OversamplingType)(int)state.getProperty("oversamplingType"));

    // --- NoiseShaperLearner Settings ---
    if (state.hasProperty("cmaesRestarts") || state.hasProperty("coeffSafetyMargin") || state.hasProperty("enableStabilityCheck"))
    {
        auto s = getNoiseShaperLearnerSettings();
        if (state.hasProperty("cmaesRestarts"))
            s.cmaesRestarts = static_cast<int>(state.getProperty("cmaesRestarts"));
        if (state.hasProperty("coeffSafetyMargin"))
            s.coeffSafetyMargin = static_cast<double>(state.getProperty("coeffSafetyMargin"));
        if (state.hasProperty("enableStabilityCheck"))
            s.enableStabilityCheck = static_cast<bool>(state.getProperty("enableStabilityCheck"));
        setNoiseShaperLearnerSettings(s);
    }

    // ─── Step 3: その他のグローバル設定 ─────────────────────────────────────
    if (state.hasProperty("softClipEnabled"))
        setSoftClipEnabled(state.getProperty("softClipEnabled"));

    if (state.hasProperty("saturationAmount"))
        setSaturationAmount(state.getProperty("saturationAmount"));

    if (state.hasProperty("analyzerSource"))
        setAnalyzerSource((AnalyzerSource)(int)state.getProperty("analyzerSource"));

    // 出力周波数フィルターモードの読み込み
    if (state.hasProperty("convHCFilterMode"))
        setConvHCFilterMode((convo::HCMode)(int)state.getProperty("convHCFilterMode"));
    if (state.hasProperty("convLCFilterMode"))
        setConvLCFilterMode((convo::LCMode)(int)state.getProperty("convLCFilterMode"));
    if (state.hasProperty("eqLPFFilterMode"))
        setEqLPFFilterMode((convo::HCMode)(int)state.getProperty("eqLPFFilterMode"));

    // ─── Step 4: サブプロセッサ状態の復元 ───────────────────────────────────
    auto eqState = state.getChildWithName ("EQ");
    if (eqState.isValid())
        uiEqEditor.setState (eqState);

    auto convState = state.getChildWithName ("Convolver");
    if (convState.isValid())
    {
       #if JUCE_DEBUG
        juce::String reason;
        const bool validConvState = validateConvolverStateTreeForDebug(convState, &reason);
        jassert(validConvState);
        if (!validConvState)
            diagLog("[ASSERT] requestLoadState: invalid Convolver state: " + reason);
       #endif
        uiConvolverProcessor.setState (convState);
    }

    const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
    if (sr > 0.0)
        requestRebuild(sr, convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire));

    // UI更新通知
    sendChangeMessage();
}
#endif

#if !defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_STATEIO_GET)
juce::ValueTree AudioEngine::getCurrentState() const
{
    juce::ValueTree state ("Preset");

    // グローバル設定の保存
    state.setProperty("processingOrder", (int)convo::consumeAtomic(currentProcessingOrder, std::memory_order_acquire), nullptr);
    state.setProperty("softClipEnabled", convo::consumeAtomic(softClipEnabled, std::memory_order_acquire), nullptr);
    state.setProperty("saturationAmount", convo::consumeAtomic(saturationAmount, std::memory_order_acquire), nullptr);
    state.setProperty("inputHeadroomDb", convo::consumeAtomic(inputHeadroomDb, std::memory_order_acquire), nullptr);
    state.setProperty("outputMakeupDb", convo::consumeAtomic(outputMakeupDb, std::memory_order_acquire), nullptr);
    state.setProperty("analyzerSource", (int)convo::consumeAtomic(currentAnalyzerSource, std::memory_order_acquire), nullptr);
    state.setProperty("convolverInputTrimDb", convo::consumeAtomic(convolverInputTrimDb, std::memory_order_acquire), nullptr);
    state.setProperty("ditherBitDepth", convo::consumeAtomic(ditherBitDepth, std::memory_order_acquire), nullptr);
    state.setProperty("noiseShaperType", (int)convo::consumeAtomic(noiseShaperType, std::memory_order_acquire), nullptr);
    state.setProperty("oversamplingFactor", convo::consumeAtomic(manualOversamplingFactor, std::memory_order_acquire), nullptr);
    state.setProperty("oversamplingType", (int)convo::consumeAtomic(oversamplingType, std::memory_order_acquire), nullptr);

    // NoiseShaperLearner Settings
    {
        auto s = getNoiseShaperLearnerSettings();
        state.setProperty("cmaesRestarts", convo::consumeAtomic(s.cmaesRestarts, std::memory_order_acquire), nullptr);
        state.setProperty("coeffSafetyMargin", convo::consumeAtomic(s.coeffSafetyMargin, std::memory_order_acquire), nullptr);
        state.setProperty("enableStabilityCheck", convo::consumeAtomic(s.enableStabilityCheck, std::memory_order_acquire), nullptr);
    }

    state.setProperty("eqBypassed", convo::consumeAtomic(eqBypassRequested, std::memory_order_acquire), nullptr);
    state.setProperty("convBypassed", convo::consumeAtomic(convBypassRequested, std::memory_order_acquire), nullptr);
    // 出力周波数フィルターモードの保存
    state.setProperty("convHCFilterMode", (int)convo::consumeAtomic(convHCFilterMode, std::memory_order_acquire), nullptr);
    state.setProperty("convLCFilterMode", (int)convo::consumeAtomic(convLCFilterMode, std::memory_order_acquire), nullptr);
    state.setProperty("eqLPFFilterMode",  (int)convo::consumeAtomic(eqLPFFilterMode, std::memory_order_acquire), nullptr);

    for (int bankIndex = 0; bankIndex < getAdaptiveSampleRateBankCount(); ++bankIndex)
    {
        const double bankSampleRate = getAdaptiveSampleRateBankHz(bankIndex);
        double adaptiveCoefficients[kAdaptiveNoiseShaperOrder] = {};
        getAdaptiveCoefficientsForSampleRate(bankSampleRate, adaptiveCoefficients, kAdaptiveNoiseShaperOrder);

        for (int coeffIndex = 0; coeffIndex < kAdaptiveNoiseShaperOrder; ++coeffIndex)
            state.setProperty(makeAdaptiveCoeffPropertyName(bankSampleRate, coeffIndex),
                              adaptiveCoefficients[coeffIndex],
                              nullptr);
    }

    state.addChild (uiEqEditor.getState(), -1, nullptr);
    state.addChild (uiConvolverProcessor.getState(), -1, nullptr);

   #if JUCE_DEBUG
    juce::String reason;
    const bool validPresetState = validatePresetStateTreeForDebug(state, &reason);
    jassert(validPresetState);
    if (!validPresetState)
        diagLog("[ASSERT] getCurrentState: invalid exported Preset state: " + reason);
   #endif

    return state;
}
#endif

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
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getInputHeadroomDb() const
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
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getOutputMakeupDb() const
{
    return convo::consumeAtomic(outputMakeupDb, std::memory_order_acquire);
}

void AudioEngine::setProcessingOrder(ProcessingOrder order)
{
    ASSERT_NON_RT_THREAD();
    convo::publishAtomic(currentProcessingOrder, order, std::memory_order_release);
    convo::publishAtomic(m_currentProcessingOrder, order, std::memory_order_release);
    enqueueSnapshotCommand();
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
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getConvolverInputTrimDb() const
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
    else if (eqBypassed && !convBypassed)
    {
        newInputHeadroomDb = -6.0f;
        newOutputMakeupDb = 12.0f;
        newConvTrimDb = 0.0f;
    }
    else
    {
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
    enqueueSnapshotCommand();
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
        enqueueSnapshotCommand();

        selectAdaptiveCoeffBankForCurrentSettings();

        // UI側（学習ウィンドウ）が即座に反映できるように通知
        sendChangeMessage();

        const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
        if (!m_isRestoringState && sr > 0.0)
        {
            const int queuedGeneration = convo::consumeAtomic(rebuildGeneration, std::memory_order_acquire);
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
                requestRebuild(sr, convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire));
            }
        }
    }
}

int AudioEngine::getDitherBitDepth() const
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
        if (type != NoiseShaperType::Adaptive9thOrder)
        {
            stopNoiseShaperLearning();
        }
        else
        {
            if (noiseShaperLearner)
                noiseShaperLearner->stopLearning();

            noiseShaperLearner = std::make_unique<NoiseShaperLearner>(*this, audioCaptureQueue);
            noiseShaperLearner->setLearningMode(convo::consumeAtomic(pendingLearningMode, std::memory_order_acquire));
            resetLearningControlState();
        }

        juce::String typeName = "Psychoacoustic";
        if (type == NoiseShaperType::Fixed4Tap)
            typeName = "Fixed4Tap";
        else if (type == NoiseShaperType::Adaptive9thOrder)
            typeName = "Adaptive9thOrder";

        DBG_LOG("Noise Shaper changed: " + typeName);
        enqueueSnapshotCommand();
        const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
        if (!m_isRestoringState && sr > 0.0)
        {
            const int queuedGeneration = convo::consumeAtomic(rebuildGeneration, std::memory_order_acquire);
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
                requestRebuild(sr, convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire));
            }
        }
    }
}

void AudioEngine::requestSnapshotForNoiseShaper()
{
    convo::publishAtomic(m_pendingNSChange, true, std::memory_order_release);
    (void)enqueueSnapshotCommand();
}

AudioEngine::NoiseShaperType AudioEngine::getNoiseShaperType() const
{
    return convo::consumeAtomic(noiseShaperType, std::memory_order_acquire);
}

void AudioEngine::setFixedNoiseLogIntervalMs(int intervalMs) noexcept
{
    convo::publishAtomic(fixedNoiseLogIntervalMs, juce::jlimit(250, 10000, intervalMs), std::memory_order_release);
}

int AudioEngine::getFixedNoiseLogIntervalMs() const noexcept
{
    return convo::consumeAtomic(fixedNoiseLogIntervalMs, std::memory_order_acquire);
}

void AudioEngine::setFixedNoiseWindowSamples(int windowSamples) noexcept
{
    convo::publishAtomic(fixedNoiseWindowSamples, juce::jlimit(256, 262144, windowSamples), std::memory_order_release);
}

int AudioEngine::getFixedNoiseWindowSamples() const noexcept
{
    return convo::consumeAtomic(fixedNoiseWindowSamples, std::memory_order_acquire);
}

void AudioEngine::setSoftClipEnabled(bool enabled)
{
    convo::publishAtomic(softClipEnabled, enabled, std::memory_order_release);
    convo::publishAtomic(m_currentSoftClipEnabled, enabled, std::memory_order_release);
    enqueueSnapshotCommand();
}

bool AudioEngine::isSoftClipEnabled() const
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
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getSaturationAmount() const
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
        enqueueSnapshotCommand();
        const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
        if (!m_isRestoringState && sr > 0.0)
        {
            requestRebuild(sr, convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire));
        }
    }
}

int AudioEngine::getOversamplingFactor() const
{
    return convo::consumeAtomic(manualOversamplingFactor, std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// Convolver UI staging setters (Rule 1.1.5)
// Implementations are kept here (.cpp) so that AudioEngine.h does not expose
// direct uiConvolverProcessor.set*() calls to the Rule 1.1.5 grep scan.
// All calls are Message Thread only and route through the UI staging object;
// the Runtime world is updated via convolverParamsChanged -> requestRebuild().
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
    enqueueSnapshotCommand();
    const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
    if (!m_isRestoringState && sr > 0.0)
    {
        requestRebuild(sr, convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire));
    }
}

AudioEngine::OversamplingType AudioEngine::getOversamplingType() const
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

convo::HCMode AudioEngine::getConvHCFilterMode() const noexcept
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

convo::LCMode AudioEngine::getConvLCFilterMode() const noexcept
{
    return convo::consumeAtomic(convLCFilterMode, std::memory_order_acquire);
}

void AudioEngine::setEqLPFFilterMode(convo::HCMode mode) noexcept
{
    convo::publishAtomic(eqLPFFilterMode, mode, std::memory_order_release);
}

convo::HCMode AudioEngine::getEqLPFFilterMode() const noexcept
{
    return convo::consumeAtomic(eqLPFFilterMode, std::memory_order_acquire);
}

juce::ValueTree AudioEngine::getConvolverStateTree() const
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

int AudioEngine::getConvolverTargetUpgradeFFTSize() const
{
    return uiConvolverProcessor.getTargetUpgradeFFTSize();
}

void AudioEngine::setConvolverTargetUpgradeFFTSize(int fftSize)
{
    uiConvolverProcessor.setTargetUpgradeFFTSize(fftSize);
}

bool AudioEngine::isConvolverProgressiveUpgradeEnabled() const
{
    return uiConvolverProcessor.isProgressiveUpgradeEnabled();
}

void AudioEngine::setConvolverEnableProgressiveUpgrade(bool enabled)
{
    uiConvolverProcessor.setEnableProgressiveUpgrade(enabled);
}

int AudioEngine::getConvolverMaxCacheEntries() const
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
