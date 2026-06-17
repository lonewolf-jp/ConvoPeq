#include <JuceHeader.h>
#include "AudioEngine.h"

namespace
{
inline juce::String makeAdaptiveCoeffPropertyName(double sampleRate, int coeffIndex)
{
    return "adaptiveCoeff_" + juce::String(static_cast<int>(sampleRate + 0.5)) + "_" + juce::String(coeffIndex);
}

class RestoreStateGuard
{
public:
    explicit RestoreStateGuard(bool& flag) noexcept : m_flag(flag)
    {
        m_flag = true;
    }

    ~RestoreStateGuard() noexcept
    {
        m_flag = false;
    }

    RestoreStateGuard(const RestoreStateGuard&) = delete;
    RestoreStateGuard& operator=(const RestoreStateGuard&) = delete;

private:
    bool& m_flag;
};
}

void AudioEngine::requestLoadState (const juce::ValueTree& state)
{
    // B19: RAII ガードを使用して、例外発生時も確実にフラグを戻す
    RestoreStateGuard guard(m_isRestoringState);

    // ─── Step 1: モード・バイパス状態を先に復元 ────────────────────────────
    if (state.hasProperty("processingOrder"))
    {
        const auto order = (ProcessingOrder)(int)state.getProperty("processingOrder");
        convo::publishAtomic(currentProcessingOrder, order, std::memory_order_release);
        convo::publishAtomic(m_currentProcessingOrder, order, std::memory_order_release);
    }

    if (state.hasProperty("eqBypassed"))
    {
        bool bypassed = state.getProperty("eqBypassed");
        convo::publishAtomic(eqBypassRequested, bypassed, std::memory_order_release);
        convo::publishAtomic(m_currentEqBypass, bypassed, std::memory_order_release);
        uiEqEditor.setBypass(bypassed);
    }

    if (state.hasProperty("convBypassed"))
    {
        bool bypassed = state.getProperty("convBypassed");
        convo::publishAtomic(convBypassRequested, bypassed, std::memory_order_release);
        convo::publishAtomic(m_currentConvBypass, bypassed, std::memory_order_release);
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
        [[maybe_unused]] bool hasBankedAdaptiveCoefficients = false;

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
        submitRebuildIntent(convo::RebuildKind::Structural,
                            RebuildTelemetryReason::RequestRebuildKindEntry,
                            RebuildTelemetryClass::Structural,
                            RebuildTelemetryPolicy::Replaceable);

    // UI更新通知
    sendChangeMessage();
}

[[nodiscard]] juce::ValueTree AudioEngine::getCurrentState() const
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
    return state;
}
