#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PARAMETERS)

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

    // ─── Step 1: モード・バイパス状態を先に復元 ────────────────────────────
    if (state.hasProperty("processingOrder"))
        convo::publishAtomic(currentProcessingOrder, (ProcessingOrder)(int)state.getProperty("processingOrder"));

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
        uiConvolverProcessor.setState (convState);

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
    state.setProperty("processingOrder", (int)convo::consumeAtomic(currentProcessingOrder), nullptr);
    state.setProperty("softClipEnabled", convo::consumeAtomic(softClipEnabled), nullptr);
    state.setProperty("saturationAmount", convo::consumeAtomic(saturationAmount), nullptr);
    state.setProperty("inputHeadroomDb", convo::consumeAtomic(inputHeadroomDb), nullptr);
    state.setProperty("outputMakeupDb", convo::consumeAtomic(outputMakeupDb), nullptr);
    state.setProperty("analyzerSource", (int)convo::consumeAtomic(currentAnalyzerSource), nullptr);
    state.setProperty("convolverInputTrimDb", convo::consumeAtomic(convolverInputTrimDb), nullptr);
    state.setProperty("ditherBitDepth", convo::consumeAtomic(ditherBitDepth), nullptr);
    state.setProperty("noiseShaperType", (int)convo::consumeAtomic(noiseShaperType), nullptr);
    state.setProperty("oversamplingFactor", convo::consumeAtomic(manualOversamplingFactor), nullptr);
    state.setProperty("oversamplingType", (int)convo::consumeAtomic(oversamplingType), nullptr);

    // NoiseShaperLearner Settings
    {
        auto s = getNoiseShaperLearnerSettings();
        state.setProperty("cmaesRestarts", convo::consumeAtomic(s.cmaesRestarts), nullptr);
        state.setProperty("coeffSafetyMargin", convo::consumeAtomic(s.coeffSafetyMargin), nullptr);
        state.setProperty("enableStabilityCheck", convo::consumeAtomic(s.enableStabilityCheck), nullptr);
    }

    state.setProperty("eqBypassed", convo::consumeAtomic(eqBypassRequested), nullptr);
    state.setProperty("convBypassed", convo::consumeAtomic(convBypassRequested), nullptr);
    // 出力周波数フィルターモードの保存
    state.setProperty("convHCFilterMode", (int)convo::consumeAtomic(convHCFilterMode), nullptr);
    state.setProperty("convLCFilterMode", (int)convo::consumeAtomic(convLCFilterMode), nullptr);
    state.setProperty("eqLPFFilterMode",  (int)convo::consumeAtomic(eqLPFFilterMode), nullptr);

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
    if (std::abs(convo::consumeAtomic(inputHeadroomDb) - clampedDb) > 1e-5f)
    {
        convo::publishAtomic(inputHeadroomDb, clampedDb);
        convo::publishAtomic(inputHeadroomGain, juce::Decibels::decibelsToGain((double)clampedDb));
        convo::publishAtomic(m_currentInputHeadroomDb, clampedDb, std::memory_order_release);
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getInputHeadroomDb() const
{
    return convo::consumeAtomic(inputHeadroomDb);
}

void AudioEngine::setOutputMakeupDb(float db)
{
    ASSERT_NON_RT_THREAD();
    // Output makeup は全モード共通で 0..12 dB
    const float clampedDb = juce::jlimit(0.0f, 12.0f, db);
    if (std::abs(convo::consumeAtomic(outputMakeupDb) - clampedDb) > 1e-5f)
    {
        convo::publishAtomic(outputMakeupDb, clampedDb);
        convo::publishAtomic(outputMakeupGain, juce::Decibels::decibelsToGain((double)clampedDb));
        convo::publishAtomic(m_currentOutputMakeupDb, clampedDb, std::memory_order_release);
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getOutputMakeupDb() const
{
    return convo::consumeAtomic(outputMakeupDb);
}

void AudioEngine::setProcessingOrder(ProcessingOrder order)
{
    ASSERT_NON_RT_THREAD();
    convo::publishAtomic(currentProcessingOrder, order);
    convo::publishAtomic(m_currentProcessingOrder, order, std::memory_order_release);
    enqueueSnapshotCommand();
    applyDefaultsForCurrentMode();
}

void AudioEngine::setConvolverInputTrimDb(float db)
{
    ASSERT_NON_RT_THREAD();
    // 範囲: -12..0 dB (0dB = トリムなし / -12dB = 最大保護)
    float clampedDb = juce::jlimit(-12.0f, 0.0f, db);
    if (std::abs(convo::consumeAtomic(convolverInputTrimDb) - clampedDb) > 1e-5f)
    {
        convo::publishAtomic(convolverInputTrimDb, clampedDb);
        convo::publishAtomic(convolverInputTrimGain, juce::Decibels::decibelsToGain((double)clampedDb));
        convo::publishAtomic(m_currentConvInputTrimDb, clampedDb, std::memory_order_release);
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getConvolverInputTrimDb() const
{
    return convo::consumeAtomic(convolverInputTrimDb);
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
    if (convo::consumeAtomic(ditherBitDepth) != bitDepth)
    {
        const bool adaptiveLearningActive = (convo::consumeAtomic(noiseShaperType, std::memory_order_acquire) == NoiseShaperType::Adaptive9thOrder)
            && noiseShaperLearner
            && noiseShaperLearner->isRunning();

        if (adaptiveLearningActive)
        {
            stopNoiseShaperLearning();
            noiseShaperLearner->setErrorMessage("Learning stopped due to bit depth change. Please restart learning.");
        }

        convo::publishAtomic(ditherBitDepth, bitDepth);
        convo::publishAtomic(m_currentDitherBitDepth, bitDepth, std::memory_order_release);
        DBG_LOG("Dither Bit Depth changed: " + juce::String(bitDepth));
        enqueueSnapshotCommand();

        selectAdaptiveCoeffBankForCurrentSettings();

        // UI側（学習ウィンドウ）が即座に反映できるように通知
        sendChangeMessage();

        const double sr = convo::consumeAtomic(currentSampleRate);
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
                requestRebuild(sr, convo::consumeAtomic(maxSamplesPerBlock));
            }
        }
    }
}

int AudioEngine::getDitherBitDepth() const
{
    return convo::consumeAtomic(ditherBitDepth);
}

void AudioEngine::setNoiseShaperType(NoiseShaperType type)
{
    if (convo::consumeAtomic(noiseShaperType) != type)
    {
        convo::publishAtomic(noiseShaperType, type);
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
        const double sr = convo::consumeAtomic(currentSampleRate);
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
                requestRebuild(sr, convo::consumeAtomic(maxSamplesPerBlock));
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
    return convo::consumeAtomic(noiseShaperType);
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
    convo::publishAtomic(saturationAmount, clamped, std::memory_order_release);
    convo::publishAtomic(m_currentSaturationAmount, clamped, std::memory_order_release);
    enqueueSnapshotCommand();
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

    if (convo::consumeAtomic(manualOversamplingFactor) != newFactor)
    {
        convo::publishAtomic(manualOversamplingFactor, newFactor);
        convo::publishAtomic(m_currentOversamplingFactor, newFactor, std::memory_order_release);
        enqueueSnapshotCommand();
        const double sr = convo::consumeAtomic(currentSampleRate);
        if (!m_isRestoringState && sr > 0.0)
        {
            requestRebuild(sr, convo::consumeAtomic(maxSamplesPerBlock));
        }
    }
}

int AudioEngine::getOversamplingFactor() const
{
    return convo::consumeAtomic(manualOversamplingFactor);
}

void AudioEngine::setOversamplingType(OversamplingType type)
{
    convo::publishAtomic(oversamplingType, type);
    convo::publishAtomic(m_currentOversamplingType, type, std::memory_order_release);
    enqueueSnapshotCommand();
    const double sr = convo::consumeAtomic(currentSampleRate);
    if (!m_isRestoringState && sr > 0.0)
    {
        requestRebuild(sr, convo::consumeAtomic(maxSamplesPerBlock));
    }
}

AudioEngine::OversamplingType AudioEngine::getOversamplingType() const
{
    return convo::consumeAtomic(oversamplingType);
}

//──────────────────────────────────────────────────────────────────────────
// 出力周波数フィルターモード Setter / Getter (Message Thread)
//──────────────────────────────────────────────────────────────────────────
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

#endif // defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PARAMETERS)
