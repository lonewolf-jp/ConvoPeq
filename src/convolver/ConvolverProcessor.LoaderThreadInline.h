#pragma once

class LoaderThread : public juce::Thread
{
public:
    LoaderThread(ConvolverProcessor& p, const juce::File& f, double sr, int bs, ConvolverProcessor::PhaseMode phase,
                 float mixedF1, float mixedF2,
                 const ConvolverProcessor::BuildSnapshot& buildSnapshotIn);
    LoaderThread(ConvolverProcessor& p, const juce::AudioBuffer<double>& src, double srcSR, double sr, int bs, ConvolverProcessor::PhaseMode phase,
                 float mixedF1, float mixedF2, double scale,
                 const ConvolverProcessor::BuildSnapshot& buildSnapshotIn);
    ~LoaderThread() override;

    std::function<bool()> externalCancellationCheck;

    struct LoadResult
    {
        juce::AudioBuffer<double> loadedIR;
        double loadedSR = 0.0;
        int targetLength = 0;
        juce::AudioBuffer<double> displayIR;
        StereoConvolver* newConv = nullptr;
        bool success = false;
        bool finalizeQueued = false;
        double scaleFactor = 1.0;
        juce::String errorMessage;
    };

    void run() override;
    LoadResult performLoad(juce::Thread* thread);

    int estimatePeakLatencySamples(const juce::AudioBuffer<double>& trimmed, int targetLength) const;
    bool buildConvolverFromTrimmed(LoadResult& result,
                                   const juce::AudioBuffer<double>& trimmed,
                                   double sr,
                                   int bs,
                                   juce::Thread* thread);

    bool initializeConvolverSynchronously(LoadResult& result,
                                          convo::ScopedAlignedPtr<double> irL,
                                          convo::ScopedAlignedPtr<double> irR,
                                          double sr,
                                          int irPeakLatency,
                                          int internalBlockSize,
                                          int callBlockSize);

    bool queueFinalizeOnMessageThread(LoadResult& result,
                                      convo::ScopedAlignedPtr<double> irL,
                                      convo::ScopedAlignedPtr<double> irR,
                                      double sr,
                                      int irPeakLatency,
                                      int internalBlockSize,
                                      int callBlockSize);

    void runSynchronously();

    enum class StepState { LoadIR, Trim, Transform, Build, Done, Error };

    StepState stepState { StepState::LoadIR };
    LoadResult stepResult;
    juce::AudioBuffer<double> stepTrimmed;
    uint64_t stepFileHash { 0 };
    juce::Thread* stepCurrentThread = nullptr;

    bool isDone() const noexcept { return stepState == StepState::Done; }
    bool hasError() const noexcept { return stepState == StepState::Error; }
    LoadResult& getStepResult() noexcept { return stepResult; }
    bool stepOnce();

private:
    bool doLoadIRStep();
    bool doTrimStep();
    bool doTransformStep();
    bool doBuildStep();

    ConvolverProcessor& owner;
    juce::WeakReference<ConvolverProcessor> weakOwner;
    juce::File file;
    juce::AudioBuffer<double> sourceIR;
    double sourceSampleRate = 0.0;
    double sampleRate;
    int blockSize;
    ConvolverProcessor::PhaseMode phaseMode;
    float mixedTransitionStartHz;
    float mixedTransitionEndHz;
    ConvolverProcessor::BuildSnapshot buildSnapshot;
    bool isRebuild;
    double scaleFactor = 1.0;
};
