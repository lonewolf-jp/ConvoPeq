#include <JuceHeader.h>
#include "AudioEngine.h"

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
    eqBypassRequested.store (shouldBypass, std::memory_order_release);
    m_currentEqBypass.store(shouldBypass, std::memory_order_release);
    uiEqEditor.setBypass(shouldBypass);
    applyDefaultsForCurrentMode();
    enqueueSnapshotCommand();
    sendChangeMessage();
}

void AudioEngine::setConvolverBypassRequested (bool shouldBypass)
{
    convBypassRequested.store (shouldBypass, std::memory_order_release);
    m_currentConvBypass.store(shouldBypass, std::memory_order_release);
    uiConvolverProcessor.setBypass(shouldBypass);
    applyDefaultsForCurrentMode();
    enqueueSnapshotCommand();
    sendChangeMessage();
}

void AudioEngine::setConvolverUseMinPhase(bool useMinPhase)
{
    setConvolverPhaseMode(useMinPhase ? ConvolverProcessor::PhaseMode::Minimum
                                      : ConvolverProcessor::PhaseMode::AsIs);
}

bool AudioEngine::getConvolverUseMinPhase() const
{
    return getConvolverPhaseMode() == ConvolverProcessor::PhaseMode::Minimum;
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

    const double sr = currentSampleRate.load(std::memory_order_acquire);
    if (sr > 0.0)
        requestRebuild(sr, maxSamplesPerBlock.load(std::memory_order_acquire));
}

#if !defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_STATEIO_LOAD)
void AudioEngine::requestLoadState (const juce::ValueTree& state)
{
    // B19: RAII ガードを使用して、例外発生時も確実にフラグを戻す
    RestoreStateGuard guard(m_isRestoringState);

    // ─── Step 1: モード・バイパス状態を先に復元 ────────────────────────────
    if (state.hasProperty("processingOrder"))
        currentProcessingOrder.store((ProcessingOrder)(int)state.getProperty("processingOrder"));

    if (state.hasProperty("eqBypassed"))
    {
        bool bypassed = state.getProperty("eqBypassed");
        eqBypassRequested.store(bypassed, std::memory_order_release);
        uiEqEditor.setBypass(bypassed);
    }

    if (state.hasProperty("convBypassed"))
    {
        bool bypassed = state.getProperty("convBypassed");
        convBypassRequested.store(bypassed, std::memory_order_release);
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

    const double sr = currentSampleRate.load(std::memory_order_acquire);
    if (sr > 0.0)
        requestRebuild(sr, maxSamplesPerBlock.load(std::memory_order_acquire));

    // UI更新通知
    sendChangeMessage();
}
#endif

#if !defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_STATEIO_GET)
juce::ValueTree AudioEngine::getCurrentState() const
{
    juce::ValueTree state ("Preset");

    // グローバル設定の保存
    state.setProperty("processingOrder", (int)currentProcessingOrder.load(), nullptr);
    state.setProperty("softClipEnabled", softClipEnabled.load(), nullptr);
    state.setProperty("saturationAmount", saturationAmount.load(), nullptr);
    state.setProperty("inputHeadroomDb", inputHeadroomDb.load(), nullptr);
    state.setProperty("outputMakeupDb", outputMakeupDb.load(), nullptr);
    state.setProperty("analyzerSource", (int)currentAnalyzerSource.load(), nullptr);
    state.setProperty("convolverInputTrimDb", convolverInputTrimDb.load(), nullptr);
    state.setProperty("ditherBitDepth", ditherBitDepth.load(), nullptr);
    state.setProperty("noiseShaperType", (int)noiseShaperType.load(), nullptr);
    state.setProperty("oversamplingFactor", manualOversamplingFactor.load(), nullptr);
    state.setProperty("oversamplingType", (int)oversamplingType.load(), nullptr);

    // NoiseShaperLearner Settings
    {
        auto s = getNoiseShaperLearnerSettings();
        state.setProperty("cmaesRestarts", s.cmaesRestarts.load(), nullptr);
        state.setProperty("coeffSafetyMargin", s.coeffSafetyMargin.load(), nullptr);
        state.setProperty("enableStabilityCheck", s.enableStabilityCheck.load(), nullptr);
    }

    state.setProperty("eqBypassed", eqBypassRequested.load(), nullptr);
    state.setProperty("convBypassed", convBypassRequested.load(), nullptr);
    // 出力周波数フィルターモードの保存
    state.setProperty("convHCFilterMode", (int)convHCFilterMode.load(), nullptr);
    state.setProperty("convLCFilterMode", (int)convLCFilterMode.load(), nullptr);
    state.setProperty("eqLPFFilterMode",  (int)eqLPFFilterMode.load(), nullptr);

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
    // コンボルバーが先頭に来る場合 (Conv→PEQ / Conv only) は -6dB 上限で入力保護する。
    // EQ が先頭またはコンボルバーがバイパスされている場合は 0dB まで許容する。
    const bool convBypassed = convBypassRequested.load(std::memory_order_relaxed);
    const bool eqBypassed   = eqBypassRequested.load(std::memory_order_relaxed);
    const ProcessingOrder order = currentProcessingOrder.load(std::memory_order_relaxed);
    const bool convIsFirst = !convBypassed && (order == ProcessingOrder::ConvolverThenEQ || eqBypassed);
    const float maxDb = convIsFirst ? -6.0f : 0.0f;
    float clampedDb = juce::jlimit(-12.0f, maxDb, db);
    if (std::abs(inputHeadroomDb.load() - clampedDb) > 1e-5f)
    {
        inputHeadroomDb.store(clampedDb);
        inputHeadroomGain.store(juce::Decibels::decibelsToGain((double)clampedDb));
        m_currentInputHeadroomDb.store(clampedDb, std::memory_order_relaxed);
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getInputHeadroomDb() const
{
    return inputHeadroomDb.load();
}

void AudioEngine::setOutputMakeupDb(float db)
{
    // Output makeup は全モード共通で 0..12 dB
    const float clampedDb = juce::jlimit(0.0f, 12.0f, db);
    if (std::abs(outputMakeupDb.load() - clampedDb) > 1e-5f)
    {
        outputMakeupDb.store(clampedDb);
        outputMakeupGain.store(juce::Decibels::decibelsToGain((double)clampedDb));
        m_currentOutputMakeupDb.store(clampedDb, std::memory_order_relaxed);
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getOutputMakeupDb() const
{
    return outputMakeupDb.load();
}

void AudioEngine::setProcessingOrder(ProcessingOrder order)
{
    currentProcessingOrder.store(order);
    m_currentProcessingOrder.store(order, std::memory_order_relaxed);
    enqueueSnapshotCommand();
    applyDefaultsForCurrentMode();
}

void AudioEngine::setConvolverInputTrimDb(float db)
{
    // 範囲: -12..0 dB (0dB = トリムなし / -12dB = 最大保護)
    float clampedDb = juce::jlimit(-12.0f, 0.0f, db);
    if (std::abs(convolverInputTrimDb.load() - clampedDb) > 1e-5f)
    {
        convolverInputTrimDb.store(clampedDb);
        convolverInputTrimGain.store(juce::Decibels::decibelsToGain((double)clampedDb));
        m_currentConvInputTrimDb.store(clampedDb, std::memory_order_relaxed);
        enqueueSnapshotCommand();
    }
}

float AudioEngine::getConvolverInputTrimDb() const
{
    return convolverInputTrimDb.load();
}

void AudioEngine::applyDefaultsForCurrentMode()
{
    if (m_isRestoringState) return; // プリセットロード中はデフォルトリセットを抑制する

    const bool eqBypassed  = eqBypassRequested.load(std::memory_order_relaxed);
    const bool convBypassed = convBypassRequested.load(std::memory_order_relaxed);
    const ProcessingOrder order = currentProcessingOrder.load(std::memory_order_relaxed);

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

    inputHeadroomDb.store(newInputHeadroomDb, std::memory_order_relaxed);
    outputMakeupDb.store(newOutputMakeupDb, std::memory_order_relaxed);
    convolverInputTrimDb.store(newConvTrimDb, std::memory_order_relaxed);
    inputHeadroomGain.store(juce::Decibels::decibelsToGain(static_cast<double>(newInputHeadroomDb)), std::memory_order_relaxed);
    outputMakeupGain.store(juce::Decibels::decibelsToGain(static_cast<double>(newOutputMakeupDb)), std::memory_order_relaxed);
    convolverInputTrimGain.store(juce::Decibels::decibelsToGain(static_cast<double>(newConvTrimDb)), std::memory_order_relaxed);

    m_currentInputHeadroomDb.store(newInputHeadroomDb, std::memory_order_relaxed);
    m_currentOutputMakeupDb.store(newOutputMakeupDb, std::memory_order_relaxed);
    m_currentConvInputTrimDb.store(newConvTrimDb, std::memory_order_relaxed);
    enqueueSnapshotCommand();
}

void AudioEngine::setDitherBitDepth(int bitDepth)
{
    if (ditherBitDepth.load() != bitDepth)
    {
        const bool adaptiveLearningActive = (noiseShaperType.load(std::memory_order_relaxed) == NoiseShaperType::Adaptive9thOrder)
            && noiseShaperLearner
            && noiseShaperLearner->isRunning();

        if (adaptiveLearningActive)
        {
            stopNoiseShaperLearning();
            noiseShaperLearner->setErrorMessage("Learning stopped due to bit depth change. Please restart learning.");
        }

        ditherBitDepth.store(bitDepth);
        m_currentDitherBitDepth.store(bitDepth, std::memory_order_relaxed);
        DBG_LOG("Dither Bit Depth changed: " + juce::String(bitDepth));
        enqueueSnapshotCommand();

        selectAdaptiveCoeffBankForCurrentSettings();

        // UI側（学習ウィンドウ）が即座に反映できるように通知
        sendChangeMessage();

        const double sr = currentSampleRate.load();
        if (!m_isRestoringState && sr > 0.0)
        {
            const int queuedGeneration = rebuildGeneration.load(std::memory_order_acquire);
            const int committedGeneration = lastCommittedRebuildGeneration.load(std::memory_order_acquire);
            const bool outstandingRebuild = queuedGeneration > committedGeneration;
            const bool shouldDeferRebuild =
                outstandingRebuild
                ||
                uiConvolverProcessor.isLoadingIR()
                || deferredStructuralRebuildPending_.load(std::memory_order_acquire)
                || m_pendingIRChange.load(std::memory_order_acquire)
                || (uiConvolverProcessor.isIRLoaded() && !uiConvolverProcessor.isIRFinalized());

            if (shouldDeferRebuild)
            {
                deferredFinalizeAwareRebuildPending_.store(true, std::memory_order_release);
                diagLog("[DIAG] setDitherBitDepth: deferred rebuild until IR finalized");
            }
            else
            {
                requestRebuild(sr, maxSamplesPerBlock.load());
            }
        }
    }
}

int AudioEngine::getDitherBitDepth() const
{
    return ditherBitDepth.load();
}

void AudioEngine::setNoiseShaperType(NoiseShaperType type)
{
    if (noiseShaperType.load() != type)
    {
        noiseShaperType.store(type);
        m_currentNoiseShaperType.store(type, std::memory_order_relaxed);
        m_pendingNSChange.store(true, std::memory_order_release);
        if (type != NoiseShaperType::Adaptive9thOrder)
        {
            stopNoiseShaperLearning();
        }
        else
        {
            if (noiseShaperLearner)
                noiseShaperLearner->stopLearning();

            noiseShaperLearner = std::make_unique<NoiseShaperLearner>(*this, audioCaptureQueue);
            noiseShaperLearner->setLearningMode(pendingLearningMode.load(std::memory_order_acquire));
            resetLearningControlState();
        }

        juce::String typeName = "Psychoacoustic";
        if (type == NoiseShaperType::Fixed4Tap)
            typeName = "Fixed4Tap";
        else if (type == NoiseShaperType::Adaptive9thOrder)
            typeName = "Adaptive9thOrder";

        DBG_LOG("Noise Shaper changed: " + typeName);
        enqueueSnapshotCommand();
        const double sr = currentSampleRate.load();
        if (!m_isRestoringState && sr > 0.0)
        {
            const int queuedGeneration = rebuildGeneration.load(std::memory_order_acquire);
            const int committedGeneration = lastCommittedRebuildGeneration.load(std::memory_order_acquire);
            const bool outstandingRebuild = queuedGeneration > committedGeneration;
            const bool shouldDeferRebuild =
                outstandingRebuild
                ||
                uiConvolverProcessor.isLoadingIR()
                || deferredStructuralRebuildPending_.load(std::memory_order_acquire)
                || m_pendingIRChange.load(std::memory_order_acquire)
                || (uiConvolverProcessor.isIRLoaded() && !uiConvolverProcessor.isIRFinalized());

            if (shouldDeferRebuild)
            {
                deferredFinalizeAwareRebuildPending_.store(true, std::memory_order_release);
                diagLog("[DIAG] setNoiseShaperType: deferred rebuild until IR finalized");
            }
            else
            {
                requestRebuild(sr, maxSamplesPerBlock.load());
            }
        }
    }
}

void AudioEngine::requestSnapshotForNoiseShaper()
{
    m_pendingNSChange.store(true, std::memory_order_release);
    (void)enqueueSnapshotCommand();
}

void AudioEngine::commitAGCChange()
{
    m_pendingAGCChange.store(true, std::memory_order_release);
    (void)enqueueSnapshotCommand();
}

AudioEngine::NoiseShaperType AudioEngine::getNoiseShaperType() const
{
    return noiseShaperType.load();
}

void AudioEngine::setFixedNoiseLogIntervalMs(int intervalMs) noexcept
{
    fixedNoiseLogIntervalMs.store(juce::jlimit(250, 10000, intervalMs), std::memory_order_relaxed);
}

int AudioEngine::getFixedNoiseLogIntervalMs() const noexcept
{
    return fixedNoiseLogIntervalMs.load(std::memory_order_relaxed);
}

void AudioEngine::setFixedNoiseWindowSamples(int windowSamples) noexcept
{
    fixedNoiseWindowSamples.store(juce::jlimit(256, 262144, windowSamples), std::memory_order_relaxed);
}

int AudioEngine::getFixedNoiseWindowSamples() const noexcept
{
    return fixedNoiseWindowSamples.load(std::memory_order_relaxed);
}

void AudioEngine::setSoftClipEnabled(bool enabled)
{
    softClipEnabled.store(enabled, std::memory_order_relaxed);
    m_currentSoftClipEnabled.store(enabled, std::memory_order_relaxed);
    enqueueSnapshotCommand();
}

bool AudioEngine::isSoftClipEnabled() const
{
    return softClipEnabled.load(std::memory_order_relaxed);
}

void AudioEngine::setSaturationAmount(float amount)
{
    const float clamped = juce::jlimit(0.0f, 1.0f, amount);
    saturationAmount.store(clamped, std::memory_order_relaxed);
    m_currentSaturationAmount.store(clamped, std::memory_order_relaxed);
    enqueueSnapshotCommand();
}

float AudioEngine::getSaturationAmount() const
{
    return saturationAmount.load(std::memory_order_relaxed);
}

void AudioEngine::setOversamplingFactor(int factor)
{
    // 0=Auto, 1, 2, 4, 8
    int newFactor = 0;
    if (factor == 1 || factor == 2 || factor == 4 || factor == 8)
    {
        newFactor = factor;
    }

    if (manualOversamplingFactor.load() != newFactor)
    {
        manualOversamplingFactor.store(newFactor);
        m_currentOversamplingFactor.store(newFactor, std::memory_order_relaxed);
        enqueueSnapshotCommand();
        const double sr = currentSampleRate.load();
        if (!m_isRestoringState && sr > 0.0)
        {
            requestRebuild(sr, maxSamplesPerBlock.load());
        }
    }
}

int AudioEngine::getOversamplingFactor() const
{
    return manualOversamplingFactor.load();
}

void AudioEngine::setOversamplingType(OversamplingType type)
{
    oversamplingType.store(type);
    m_currentOversamplingType.store(type, std::memory_order_relaxed);
    enqueueSnapshotCommand();
    const double sr = currentSampleRate.load();
    if (!m_isRestoringState && sr > 0.0)
    {
        requestRebuild(sr, maxSamplesPerBlock.load());
    }
}

AudioEngine::OversamplingType AudioEngine::getOversamplingType() const
{
    return oversamplingType.load();
}

//──────────────────────────────────────────────────────────────────────────
// 出力周波数フィルターモード Setter / Getter (Message Thread)
//──────────────────────────────────────────────────────────────────────────
void AudioEngine::setConvHCFilterMode(convo::HCMode mode) noexcept
{
    convHCFilterMode.store(mode, std::memory_order_relaxed);
    // NUC irFreqDomain を再焼き込みするため、uiConvolverProcessor を再構築する。
    // DSPCore::convolver は次回 requestRebuild 時に syncStateFrom + rebuildAllIRsSynchronous で追従する。
    uiConvolverProcessor.setNUCFilterModes(
        convHCFilterMode.load(std::memory_order_relaxed),
        convLCFilterMode.load(std::memory_order_relaxed));
}

convo::HCMode AudioEngine::getConvHCFilterMode() const noexcept
{
    return convHCFilterMode.load(std::memory_order_relaxed);
}

void AudioEngine::setConvLCFilterMode(convo::LCMode mode) noexcept
{
    convLCFilterMode.store(mode, std::memory_order_relaxed);
    // HC と組み合わせて NUC を再構築
    uiConvolverProcessor.setNUCFilterModes(
        convHCFilterMode.load(std::memory_order_relaxed),
        convLCFilterMode.load(std::memory_order_relaxed));
}

convo::LCMode AudioEngine::getConvLCFilterMode() const noexcept
{
    return convLCFilterMode.load(std::memory_order_relaxed);
}

void AudioEngine::setEqLPFFilterMode(convo::HCMode mode) noexcept
{
    eqLPFFilterMode.store(mode, std::memory_order_relaxed);
}

convo::HCMode AudioEngine::getEqLPFFilterMode() const noexcept
{
    return eqLPFFilterMode.load(std::memory_order_relaxed);
}

#endif // defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PARAMETERS)
