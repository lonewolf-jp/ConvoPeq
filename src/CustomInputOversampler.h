//============================================================================
#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>

#include "AlignedAllocation.h"

#include "audioengine/AtomicAccess.h"

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
    static constexpr int kMaxChannels = 2;

    CustomInputOversampler() = default;
    ~CustomInputOversampler();

    CustomInputOversampler(const CustomInputOversampler&) = delete;
    CustomInputOversampler& operator=(const CustomInputOversampler&) = delete;

    void prepare(int maxInputBlockSize, int ratio, Preset preset);
    void reset() noexcept;
    void release() noexcept;

    juce::dsp::AudioBlock<double> processUp(juce::dsp::AudioBlock<double>& inputBlock, int numChannels) noexcept;
    void processDown(const juce::dsp::AudioBlock<double>& upsampledBlock,
                     juce::dsp::AudioBlock<double>& outputBlock,
                     int numChannels) noexcept;

    // 異常フラグを取得してリセットする (Audio Thread 安全)
    bool consumeCorruptionFlag() noexcept
    {
        return convo::exchangeAtomic(corruptionDetected, false, std::memory_order_acq_rel);
    }

    std::uint64_t getCorruptionEventCount() const noexcept
    {
        return convo::consumeAtomic(corruptionEventCount, std::memory_order_acquire);
    }

    std::uint64_t getCorruptionAutoClearCount() const noexcept
    {
        return convo::consumeAtomic(corruptionAutoClearCount, std::memory_order_acquire);
    }

    bool isHardFallbackActive() const noexcept
    {
        return convo::consumeAtomic(hardFallbackActive, std::memory_order_acquire);
    }

    void resetCorruptionTelemetry() noexcept
    {
        convo::publishAtomic(corruptionEventCount, static_cast<std::uint64_t>(0), std::memory_order_release);
        convo::publishAtomic(corruptionAutoClearCount, static_cast<std::uint64_t>(0), std::memory_order_release);
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
    static double dotProductDecimateAvx2(const double* history, const double* coeffs, int convCount) noexcept;

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
    void markCorruptionDetected() noexcept;

    int upsampleRatio = 1;
    Preset activePreset = Preset::IIRLike;
    int numStages = 0;
    int maxInputBlockSize = 0;
    int maxUpsampledBlockSize = 0;

    Stage stages[3];

    convo::ScopedAlignedPtr<double> workA[2];
    convo::ScopedAlignedPtr<double> workB[2];
    int workCapacity = 0;

    std::array<convo::NonOwningPtr<double>, kMaxChannels> blockChannels {};
    // RT-SAFE: blockChannelView is a member (not thread_local) so the returned AudioBlock's
    // channel pointer array remains valid after processUp() returns.
    double* blockChannelView[kMaxChannels] = {};
    std::atomic<bool> corruptionDetected { false };
    std::atomic<std::uint64_t> corruptionEventCount { 0 };
    std::atomic<std::uint64_t> corruptionAutoClearCount { 0 };
    std::atomic<std::uint32_t> consecutiveCorruptionAutoClearCount { 0 };
    std::atomic<bool> hardFallbackActive { false };

    static constexpr std::uint32_t kHardFallbackAutoClearThreshold = 4;
};
