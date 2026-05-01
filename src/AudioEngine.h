
//============================================================================
#pragma once
//============================================================================

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>
#include <vector>
#include <memory>
#include <juce_dsp/juce_dsp.h>

#include "ConvolverProcessor.h"
#include "EQProcessor.h"
#include "EQEditProcessor.h"
#include "NoiseShaperLearner.h"
#include "LockFreeRingBuffer.h"
#include "core/EngineView.h"
#include "core/RebuildTypes.h"
#include "core/ThreadAffinityManager.h"

namespace convo {
    struct EngineState;
}

struct AudioBlock {
    int numSamples = 0;
    uint64_t sessionId = 0;
    int sampleRateHz = 0;
    int adaptiveCoeffBankIndex = 0;
    double L[256] = {};
    double R[256] = {};
};

static constexpr int kAdaptiveNoiseShaperOrder = 9;
static constexpr int kAdaptiveNoiseShaperSampleRateBankCount = 10;
static constexpr int kAdaptiveBitDepthCount = 3;
static constexpr int kAdaptiveBitDepthValues[kAdaptiveBitDepthCount] = {16, 24, 32};
static constexpr int kLearningModeCount = 6;

class AudioEngine : public juce::AudioSource,
                    public juce::ChangeBroadcaster,
                    private juce::ChangeListener,
                    private ConvolverProcessor::Listener,
                    private juce::Timer,
                    private juce::AsyncUpdater
{
public:
    AudioEngine();
    ~AudioEngine() override;

    using ProcessingOrder = convo::ProcessingOrder;
    using OversamplingType = convo::OversamplingType;
    using NoiseShaperType = convo::NoiseShaperType;

    // AudioSource overrides
    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override;
    void releaseResources() override;
    void getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill) override;
    void processBlockDouble (juce::AudioBuffer<double>& buffer);

    struct LatencyBreakdown {
        int totalLatencyBaseRateSamples = 0;
    };
    LatencyBreakdown getCurrentLatencyBreakdown() const;

    // Snapshot Management
    void requestRebuild(convo::RebuildKind kind) noexcept;
    void requestRebuild(double sampleRate, int maxBlockSize);
    void publishEngineState(convo::EngineState&& newState, float fadeTimeSec);
    
    int getTotalLatencySamples() const { return 0; } // Implement properly if needed
    double getSampleRate() const { return currentSampleRate.load(std::memory_order_acquire); }
    int getDitherBitDepth() const { return 24; } // Default for now, was a member before

    // Learning
    struct LearningCommand {
        enum class Type { Start, Stop, IRChanged, DSPReady };
        Type type;
        NoiseShaperLearner::LearningMode mode;
        bool resume;
        int irGeneration;
    };
    bool enqueueLearningCommand(const LearningCommand& cmd) noexcept;
    bool dequeueLearningCommand(LearningCommand& cmd) noexcept;

    struct LearnerDispatchAction {
        enum class Type { Start, Stop };
        Type type;
        bool resume;
        NoiseShaperLearner::LearningMode mode;
    };
    bool enqueueLearnerDispatch(const LearnerDispatchAction& action) noexcept;
    bool dequeueLearnerDispatch(LearnerDispatchAction& action) noexcept;

    enum class LearningRuntimeState { Idle, Running, WaitingForDSP };

    // Learning Bridges
    void setAdaptiveNoiseShaperState(int bankIndex, const NoiseShaperLearner::State& state);
    bool getAdaptiveNoiseShaperState(int bankIndex, NoiseShaperLearner::State& state) const;
    void getAdaptiveCoefficientsForSampleRateAndBitDepth(double sr, int bd, double* coeffs, int order) const;
    const ThreadAffinityManager& getAffinityManager() const { return affinityManager; }
    void requestSnapshotForNoiseShaper() {} // Placeholder
    void publishCoeffs(const double* coeffs);
    void requestAdaptiveAutosave() {} // Placeholder

    const convo::EngineView& getActiveView() const { return m_views[m_activeIndex.load(std::memory_order_acquire)]; }

private:
    void handleAsyncUpdate() override;
    void timerCallback() override;
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void convolverParamsChanged(ConvolverProcessor* processor) override;

    void advanceFade(float step);
    void processWithState(juce::AudioBuffer<float>& output, const convo::EngineState& state, int startSample, int numSamples, float gain);
    void executeCommit();
    void processRebuildRequestsInternal();
    bool enqueueSnapshotCommand();

    // -- Sub-processors --
    ConvolverProcessor uiConvolverProcessor;
    EQEditProcessor uiEqEditor { *this };
    std::unique_ptr<NoiseShaperLearner> noiseShaperLearner;
    LockFreeRingBuffer<AudioBlock, 4096> captureQueue;
    ThreadAffinityManager affinityManager;

    // -- Snapshot Double Buffer --
    static constexpr int kNumViews = 2;
    convo::EngineView m_views[kNumViews];
    std::atomic<int> m_activeIndex {0};

    // -- Buffers --
    juce::AudioBuffer<float> m_fadeFloatBuffer;
    juce::AudioBuffer<double> m_fadeDoubleBuffer;
    juce::AudioBuffer<float> m_tmpA;
    juce::AudioBuffer<float> m_tmpB;

    // -- Learning State --
    std::atomic<LearningRuntimeState> learningRuntimeState { LearningRuntimeState::Idle };
    std::atomic<NoiseShaperLearner::LearningMode> requestedLearningMode { NoiseShaperLearner::LearningMode::None };
    std::atomic<bool> requestedLearningResume { false };
    std::atomic<int> requestedLearningGeneration { 0 };
    std::atomic<int> currentIRGeneration { 0 };
    std::atomic<int> pendingIRGeneration { 0 };
    std::atomic<NoiseShaperLearner::LearningMode> pendingLearningMode { NoiseShaperLearner::LearningMode::None };
    std::atomic<int> currentAdaptiveCoeffBankIndex { 0 };

    // -- Learning Queues --
    static constexpr uint32_t learningCommandBufferMask = 127;
    LearningCommand learningCommandBuffer[128];
    std::atomic<uint32_t> learningCommandWrite {0}, learningCommandRead {0};

    static constexpr uint32_t learnerDispatchBufferMask = 127;
    LearnerDispatchAction learnerDispatchBuffer[128];
    std::atomic<uint32_t> learnerDispatchWrite {0}, learnerDispatchRead {0};
    std::atomic<bool> learnerDispatchOverflow { false };
    std::atomic<LearnerDispatchAction> lastFailedAction;

    // -- Rebuild / Deferred Tasks --
    std::atomic<uint64_t> lastIRContentRebuildTicks_ { 0 };
    std::atomic<uint32_t> pendingRebuildMask_ { 0 };
    std::atomic<bool> deferredStructuralRebuildPending_ { false };
    std::atomic<int64_t> deferredStructuralRebuildDueTicks_ { 0 };
    std::atomic<bool> deferredFinalizeAwareRebuildPending_ { false };
    std::atomic<int> rebuildGeneration { 0 };
    std::atomic<int> lastCommittedRebuildGeneration { 0 };
    std::atomic<double> currentSampleRate { 0.0 };
    std::atomic<int> maxSamplesPerBlock { 0 };
    std::array<NoiseShaperLearner::State, kAdaptiveNoiseShaperSampleRateBankCount> savedStates {};
    bool m_isRestoringState { false };

    // -- Fading / Legacy Compat (to be cleaned up in cpp if possible, but keep members for now) --
    std::atomic<bool> shutdownInProgress { false };
    std::atomic<bool> m_pendingIRChange { false };
    std::atomic<uint64_t> m_audioBlockCounter { 0 };
    std::atomic<bool> dspCrossfadePending { false };
    std::atomic<bool> fadeQueued { false };
    std::atomic<float> queuedNextFadeTimeSec { 0.0f };
    std::atomic<float> queuedFadeTimeSec { 0.0f };

    // -- Epoch Management for Retired Engine Reclamation --
    std::atomic<uint64_t> m_audioEpoch { 0 };

    // -- Diagnostics --
    std::atomic<int> fixedNoiseWindowSamples { 4096 };
    std::atomic<int> fixedNoiseLogIntervalMs { 1000 };
    uint32 fixedNoiseLastLogMs = 0;

    // -- Analyzer --
    enum class AnalyzerSource { Input, Output };
    std::atomic<AnalyzerSource> m_analyzerSource { AnalyzerSource::Output };

    JUCE_DECLARE_WEAK_REFERENCEABLE(AudioEngine)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioEngine)
};
