//============================================================================
#pragma once

#include <JuceHeader.h>
#include <atomic>

#include "AlignedAllocation.h"

class CustomInputOversampler
{
public:
    enum class Preset
    {
        IIRLike,
        LinearPhase
    };

    static constexpr bool isLinearPhaseFIR = true;
    static constexpr bool isSymmetricUpDown = true;

    CustomInputOversampler() = default;
    ~CustomInputOversampler();

    CustomInputOversampler(const CustomInputOversampler&) = delete;
    CustomInputOversampler& operator=(const CustomInputOversampler&) = delete;

    void prepare(int maxInputBlockSize, int ratio, Preset preset);
    void reset() noexcept;
    void release() noexcept;

    bool isActive() const noexcept { return upsampleRatio > 1; }
    int getRatio() const noexcept { return upsampleRatio; }
    int getMaxUpsampledBlockSize() const noexcept { return maxUpsampledBlockSize; }

    juce::dsp::AudioBlock<double> processUp(const juce::dsp::AudioBlock<double>& inputBlock, int numChannels) noexcept;
    void processDown(const juce::dsp::AudioBlock<double>& upsampledBlock,
                     juce::dsp::AudioBlock<double>& outputBlock,
                     int numChannels) noexcept;

    // 異常フラグを取得してリセットする (Audio Thread 安全)
    bool consumeCorruptionFlag() noexcept
    {
        return corruptionDetected.exchange(false, std::memory_order_acq_rel);
    }

    // 全ステージ履歴をゼロクリアして異常フラグを解除する
    void clearAllStages() noexcept;

private:
    struct Stage
    {
        int taps = 0;
        int centerTap = 0;
        int centerParity = 0;
        int convParity = 0;
        int convCount = 0;
        int centerDelayInput = 0;
        int historyUpKeep = 0;
        int historyDownKeep = 0;
        int maxInputSamples = 0;
        int maxOutputSamples = 0;

        double centerCoeff = 0.5;
        convo::ScopedAlignedPtr<double> convCoeffs;
        convo::ScopedAlignedPtr<double> convCoeffsReversed;
        convo::ScopedAlignedPtr<double> upHistory[2];
        convo::ScopedAlignedPtr<double> downHistory[2];
        int upHistorySize = 0;
        int downHistorySize = 0;
    };

    void clearStage(Stage& stage) noexcept;
    void prepareStage(Stage& stage, int taps, double attenuationDb, int stageInputMax);

    static double besselI0(double x) noexcept;
    static double dotProductAvx2(const double* x, const double* coeffs, int n) noexcept;

    static int sanitizeRatio(int ratio) noexcept;
    static int tapsForStage(int stageIndex, Preset preset) noexcept;
    static double attenuationForStage(int stageIndex, Preset preset) noexcept;

    void interpolateStage(const Stage& stage,
                          const double* input,
                          int inputSamples,
                          double* output,
                          int channel) noexcept;

    void decimateStage(const Stage& stage,
                       const double* input,
                       int inputSamples,
                       double* output,
                       int channel) noexcept;

    int upsampleRatio = 1;
    Preset activePreset = Preset::IIRLike;
    int numStages = 0;
    int maxInputBlockSize = 0;
    int maxUpsampledBlockSize = 0;

    Stage stages[3];

    convo::ScopedAlignedPtr<double> workA[2];
    convo::ScopedAlignedPtr<double> workB[2];
    int workCapacity = 0;

    double* blockChannels[2] = { nullptr, nullptr };
    std::atomic<bool> corruptionDetected { false };
};
