#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "audioengine/AudioEngine.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "AlignedAllocation.h"
#include <mkl.h>

#include "audioengine/AtomicAccess.h"

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOADER_THREAD)

ConvolverProcessor::LoaderThread::LoaderThread(ConvolverProcessor& p, const juce::File& f, double sr, int bs, ConvolverProcessor::PhaseMode phase,
                                 float mixedF1, float mixedF2,
                                 const ConvolverProcessor::BuildSnapshot& buildSnapshotIn)
    : Thread("IRLoader"), owner(p), weakOwner(&p), file(f), sampleRate(sr), blockSize(bs), phaseMode(phase),
    mixedTransitionStartHz(mixedF1), mixedTransitionEndHz(mixedF2),
    buildSnapshot(buildSnapshotIn), isRebuild(false)
{}

ConvolverProcessor::LoaderThread::LoaderThread(ConvolverProcessor& p, const juce::AudioBuffer<double>& src, double srcSR, double sr, int bs, ConvolverProcessor::PhaseMode phase,
                                 float mixedF1, float mixedF2, double scale,
                                 const ConvolverProcessor::BuildSnapshot& buildSnapshotIn)
    : Thread("IRRebuilder"), owner(p), weakOwner(&p), sourceIR(src), sourceSampleRate(srcSR), sampleRate(sr), blockSize(bs), phaseMode(phase),
    mixedTransitionStartHz(mixedF1), mixedTransitionEndHz(mixedF2),
    buildSnapshot(buildSnapshotIn), isRebuild(true), scaleFactor(scale)
{}

ConvolverProcessor::LoaderThread::~LoaderThread()
{
    stopThread(500);

    auto* conv = std::exchange(stepResult.newConv, nullptr);
    owner.retireStereoConvolver(conv, 0);
}

void ConvolverProcessor::LoaderThread::run()
{
    if (auto* provider = owner.getRcuProvider(); provider != nullptr)
        provider->getAffinityManager().applyCurrentThreadPolicy(ThreadType::HeavyBackground);

    juce::ScopedNoDenormals noDenormals;

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

    struct FlagResetter {
        ConvolverProcessor& p;
        juce::WeakReference<ConvolverProcessor> weakP;
        const juce::Thread& t;
        bool success = false;
        ~FlagResetter() {
            if (!success && !t.threadShouldExit()) {
                auto wp = weakP;
                const bool queued = juce::MessageManager::callAsync([wp] {
                    if (auto* o = wp.get()) {
                        convo::publishAtomic(o->isLoading, false, std::memory_order_release); // release: timer/UI の isLoading acquire と HB
                        convo::publishAtomic(o->isRebuilding, false, std::memory_order_release); // release: timer/load 経路 acquire と HB
                    }
                });

                if (!queued)
                {
                    if (auto* o = wp.get())
                    {
                        convo::publishAtomic(o->isLoading, false, std::memory_order_release); // release: timer/UI の isLoading acquire と HB（callAsync 失敗時）
                        convo::publishAtomic(o->isRebuilding, false, std::memory_order_release); // release: timer/load 経路 acquire と HB（callAsync 失敗時）
                    }
                }
            }
        }
    } resetter { owner, weakOwner, *this };

    LoadResult result = performLoad(this);

    resetter.success = (result.success || result.finalizeQueued);

    owner.retireStereoConvolver(std::exchange(result.newConv, nullptr), 0);

    if (!result.success && result.errorMessage.isNotEmpty() && !threadShouldExit())
    {
        auto wp = weakOwner;
        const juce::String error = result.errorMessage;
        const bool queued = juce::MessageManager::callAsync([wp, error]()
        {
            if (auto* o = wp.get())
                o->handleLoadError(error);
        });

        if (!queued)
            juce::Logger::writeToLog("LoaderThread: callAsync failed in error path; dropping UI error dispatch");
    }
}

ConvolverProcessor::LoaderThread::LoadResult ConvolverProcessor::LoaderThread::performLoad(juce::Thread* thread)
{
    std::function<bool()> savedCheck = externalCancellationCheck;
    if (thread != nullptr)
    {
        externalCancellationCheck = [thread, saved = savedCheck]() -> bool {
            if (thread->threadShouldExit()) return true;
            return saved && saved();
        };
    }

    stepCurrentThread = thread;
    stepState = StepState::LoadIR;
    stepResult = LoadResult{};
    stepTrimmed.setSize(0, 0);
    stepFileHash = 0;

    try
    {
        while (true)
        {
            const bool terminal = stepOnce();
            if (terminal) break;
        }
    }
    catch (const std::bad_alloc&)
    {
        stepResult.errorMessage = "IR too large (Out of Memory)";
        juce::Logger::writeToLog("LoaderThread: " + stepResult.errorMessage);
    }
    catch (const std::exception& e)
    {
        stepResult.errorMessage = "Error loading IR: " + juce::String(e.what());
        juce::Logger::writeToLog("LoaderThread: " + stepResult.errorMessage);
    }
    catch (...)
    {
        stepResult.errorMessage = "Unknown error loading IR";
        juce::Logger::writeToLog("LoaderThread: " + stepResult.errorMessage);
    }

    externalCancellationCheck = std::move(savedCheck);
    stepCurrentThread = nullptr;
    return std::move(stepResult);
}

int ConvolverProcessor::LoaderThread::estimatePeakLatencySamples(const juce::AudioBuffer<double>& trimmed, int targetLength) const
{
    int irPeakLatency = 0;
    if (trimmed.getNumChannels() > 0)
    {
        constexpr double ENERGY_THRESHOLD = 0.999;
        double maxCentroid = 0.0;

        const int length = targetLength;
        std::vector<double> energyBuffer(static_cast<size_t>(length));
        double* energy = energyBuffer.data();
        for (int ch = 0; ch < trimmed.getNumChannels(); ++ch)
        {
            const double* data = trimmed.getReadPointer(ch);

            double totalEnergy = 0.0;
            for (int i = 0; i < length; ++i)
            {
                const double e = data[i] * data[i];
                energy[i] = e;
                totalEnergy += e;
            }
            if (totalEnergy < 1e-12)
                continue;

            double cumulative = 0.0;
            int cutoff = length - 1;
            for (int i = 0; i < length; ++i)
            {
                cumulative += energy[i];
                if (cumulative >= totalEnergy * ENERGY_THRESHOLD)
                {
                    cutoff = i;
                    break;
                }
            }

            double sumE = 0.0;
            double sumW = 0.0;
            for (int i = 0; i <= cutoff; ++i)
            {
                const double e = energy[i];
                sumE += e;
                sumW += static_cast<double>(i) * e;
            }

            const double centroid = (sumE > 0.0) ? (sumW / sumE) : 0.0;
            if (centroid > maxCentroid)
                maxCentroid = centroid;
        }

        irPeakLatency = static_cast<int>(std::floor(maxCentroid + 0.5));
        irPeakLatency = juce::jlimit(0, targetLength - 1, irPeakLatency);
    }

    return irPeakLatency;
}

bool ConvolverProcessor::LoaderThread::buildConvolverFromTrimmed(LoadResult& result,
                                                                  const juce::AudioBuffer<double>& trimmed,
                                                                  double sr,
                                                                  int bs,
                                                                  juce::Thread* thread)
{
    if (trimmed.getNumChannels() == 0)
        return false;

    const int irPeakLatency = estimatePeakLatencySamples(trimmed, result.targetLength);

    auto irL = convo::makeAlignedArray<double>(static_cast<size_t>(result.targetLength));
    auto irR = convo::makeAlignedArray<double>(static_cast<size_t>(result.targetLength));

    const double* srcL = trimmed.getReadPointer(0);
    const double* srcR = (trimmed.getNumChannels() > 1) ? trimmed.getReadPointer(1) : srcL;
    std::memcpy(irL.get(), srcL, result.targetLength * sizeof(double));
    std::memcpy(irR.get(), srcR, result.targetLength * sizeof(double));

    const int internalBlockSize = juce::nextPowerOfTwo(bs);

    if (owner.isVisualizationEnabled())
    {
        result.displayIR = trimmed;
        result.displayIR.applyGain(result.scaleFactor);
    }

    if (thread == nullptr)
        return initializeConvolverSynchronously(result,
                                                std::move(irL),
                                                std::move(irR),
                                                sr,
                                                irPeakLatency,
                                                internalBlockSize,
                                                bs);

    return queueFinalizeOnMessageThread(result,
                                        std::move(irL),
                                        std::move(irR),
                                        sr,
                                        irPeakLatency,
                                        internalBlockSize,
                                        bs);
}

bool ConvolverProcessor::LoaderThread::initializeConvolverSynchronously(LoadResult& result,
                                                                         convo::ScopedAlignedPtr<double> irL,
                                                                         convo::ScopedAlignedPtr<double> irR,
                                                                         double sr,
                                                                         int irPeakLatency,
                                                                         int internalBlockSize,
                                                                         int callBlockSize)
{
    auto newConv = convo::aligned_make_unique<StereoConvolver>();

    convo::FilterSpec spec;
    spec.sampleRate = sr;
    {
        spec.hcMode = static_cast<convo::HCMode>(buildSnapshot.nucHCMode);
        spec.lcMode = static_cast<convo::LCMode>(buildSnapshot.nucLCMode);
        spec.tailMode = juce::jlimit(static_cast<int>(TailMode::AirAbsorption),
                                     static_cast<int>(TailMode::Bypass),
                                     buildSnapshot.tailMode);
        spec.tailEnabled = (spec.tailMode != static_cast<int>(TailMode::Bypass));
        spec.tailStartSeconds = static_cast<double>(buildSnapshot.tailStartSec);
        spec.tailStrength = static_cast<double>(buildSnapshot.tailStrength);
        spec.tailL1L2Multiplier = buildSnapshot.tailL1L2Multiplier;
    }

    if (newConv->init(irL.release(), irR.release(), result.targetLength, sr, irPeakLatency,
                             internalBlockSize, callBlockSize, result.scaleFactor,
                             owner.getExperimentalDirectHeadEnabled(),
                             &spec, &owner))
    {
        result.newConv = newConv.release();
        result.success = true;
        return true;
    }

    result.success = false;
    result.errorMessage = "Failed to initialize NUC engine (Memory allocation or MKL setup failed).";
    return false;
}

bool ConvolverProcessor::LoaderThread::queueFinalizeOnMessageThread(LoadResult& result,
                                                                     convo::ScopedAlignedPtr<double> irL,
                                                                     convo::ScopedAlignedPtr<double> irR,
                                                                     double sr,
                                                                     int irPeakLatency,
                                                                     int internalBlockSize,
                                                                     int callBlockSize)
{
    auto loadedIRRaw = new juce::AudioBuffer<double>(std::move(result.loadedIR));
    auto displayIRRaw = new juce::AudioBuffer<double>(std::move(result.displayIR));
    auto irLRaw = irL.release();
    auto irRRaw = irR.release();

    // ★ H-02: ラムダ内で unique_ptr / ScopedAlignedPtr にラップしているため、
    // weakOwner.get() が nullptr を返してもリソースは解放される。
    // 唯一の未解放経路は JUCE シャットダウン時の MessageManager キュー破棄だが、
    // これは正常シャットダウンで許容範囲の動作である。
    const bool queued = juce::MessageManager::callAsync([weakOwner = this->weakOwner,
                                     irLRaw,
                                     irRRaw,
                                     loadedIRRaw,
                                     displayIRRaw,
                                     length = result.targetLength,
                                     sr,
                                     peak = irPeakLatency,
                                     known = internalBlockSize,
                                     callQ = callBlockSize,
                                     isReb = isRebuild,
                                     file = file,
                                     buildSnapshot = this->buildSnapshot,
                                     scale = result.scaleFactor]()
    {
        convo::ScopedAlignedPtr<double> irLHolder(irLRaw);
        convo::ScopedAlignedPtr<double> irRHolder(irRRaw);
        std::unique_ptr<juce::AudioBuffer<double>> loadedIRHolder(loadedIRRaw);
        std::unique_ptr<juce::AudioBuffer<double>> displayIRHolder(displayIRRaw);

        if (auto* ownerPtr = weakOwner.get())
        {
            ownerPtr->finalizeNUCEngineOnMessageThread(std::move(irLHolder),
                                                       std::move(irRHolder),
                                                       length, sr, peak, known, callQ, isReb, file,
                                                       buildSnapshot,
                                                       scale, std::move(loadedIRHolder), std::move(displayIRHolder));
        }
    });

    if (!queued)
    {
        convo::aligned_free(irLRaw);
        convo::aligned_free(irRRaw);
        std::unique_ptr<juce::AudioBuffer<double>>{loadedIRRaw};  // RAII delete
        std::unique_ptr<juce::AudioBuffer<double>>{displayIRRaw}; // RAII delete

        juce::Logger::writeToLog("LoaderThread: callAsync failed, aborting IR load");
        result.errorMessage = "Internal message queue full, cannot complete IR load";

        owner.retireStereoConvolver(std::exchange(result.newConv, nullptr), 0);

        juce::MessageManager::callAsync([weakOwner = this->weakOwner, errorMsg = result.errorMessage]()
        {
            if (auto* ownerPtr = weakOwner.get())
                ownerPtr->handleLoadError(errorMsg);
        });

        return false;
    }

    result.finalizeQueued = true;
    return true;
}

void ConvolverProcessor::LoaderThread::runSynchronously()
{
    juce::ScopedNoDenormals noDenormals;
    LoadResult result = performLoad(nullptr);

    if (result.success)
    {
        auto* conv = std::exchange(result.newConv, nullptr);
        stepResult.newConv = nullptr;
        auto loadedIR = std::make_unique<juce::AudioBuffer<double>>(std::move(result.loadedIR));
        auto displayIR = std::make_unique<juce::AudioBuffer<double>>(std::move(result.displayIR));
        owner.applyNewState(conv, std::move(loadedIR), result.loadedSR, result.targetLength, isRebuild, file,
                            result.scaleFactor, std::move(displayIR), /*async=*/false);
    }
    else
    {
        if (result.newConv)
            owner.retireStereoConvolver(std::exchange(result.newConv, nullptr), 0);
    }
}

bool ConvolverProcessor::LoaderThread::stepOnce()
{
    switch (stepState)
    {
        case StepState::LoadIR:
            if (!doLoadIRStep()) { stepState = StepState::Error; return true; }
            stepState = StepState::Trim;
            return false;

        case StepState::Trim:
            if (!doTrimStep()) { stepState = StepState::Error; return true; }
            stepState = StepState::Transform;
            return false;

        case StepState::Transform:
            if (!doTransformStep()) { stepState = StepState::Error; return true; }
            stepState = StepState::Build;
            return false;

        case StepState::Build:
            if (!doBuildStep()) { stepState = StepState::Error; return true; }
            stepState = StepState::Done;
            return true;

        case StepState::Done:
        case StepState::Error:
            return true;
    }
    return true;
}

bool ConvolverProcessor::LoaderThread::doLoadIRStep()
{
    stepFileHash = 0;
    if (!isRebuild && file.existsAsFile())
        stepFileHash = convo::AllpassDesigner::computeIRHash(file);

    if (isRebuild)
    {
        stepResult.loadedIR = std::move(sourceIR);
        stepResult.loadedSR = sourceSampleRate;
        stepResult.scaleFactor = this->scaleFactor;
    }
    else
    {
        if (!file.existsAsFile())
        {
            stepResult.errorMessage = "IR file not found: " + file.getFullPathName();
            return false;
        }

        juce::AudioFormatManager formatManager;
        formatManager.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
        if (!reader)
        {
            stepResult.errorMessage = "Unsupported audio format or corrupted file: " + file.getFileName();
            return false;
        }

        const int64 fileLength = reader->lengthInSamples;
        const int numChannels = static_cast<int>(reader->numChannels);
        static constexpr int64 MAX_FILE_LENGTH = 2147483647;

        if (fileLength > MAX_FILE_LENGTH)
        {
            stepResult.errorMessage = "IR file is too large (exceeds 2GB samples limit).";
            return false;
        }
        if (numChannels <= 0)
        {
            stepResult.errorMessage = "Invalid channel count in IR file.";
            return false;
        }

        juce::AudioBuffer<float> tempFloatBuffer(numChannels, static_cast<int>(fileLength));
        if (!reader->read(&tempFloatBuffer, 0, static_cast<int>(fileLength), 0, true, true))
        {
            stepResult.errorMessage = "Failed to read audio data from file.";
            return false;
        }

        auto tempAligned = convo::makeAlignedArray<double>(static_cast<size_t>(fileLength));
        if (!tempAligned)
        {
            stepResult.errorMessage = "Failed to allocate temporary buffer for IR loading.";
            return false;
        }

        stepResult.loadedIR.setSize(numChannels, static_cast<int>(fileLength));
        for (int ch = 0; ch < numChannels; ++ch)
        {
            const float* src = tempFloatBuffer.getReadPointer(ch);
            convo::input_transform::convertFloatToDoubleHighQuality(
                src, tempAligned.get(), static_cast<int>(fileLength));
            stepResult.loadedIR.copyFrom(ch, 0, tempAligned.get(), static_cast<int>(fileLength));
        }
        stepResult.loadedSR = reader->sampleRate;
    }

    return (stepResult.loadedIR.getNumSamples() > 0 && stepResult.loadedIR.getNumChannels() > 0);
}

bool ConvolverProcessor::LoaderThread::doTrimStep()
{
    auto shouldStop = [this]() -> bool {
        return externalCancellationCheck && externalCancellationCheck();
    };

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    if (stepResult.loadedIR.getNumSamples() > 0)
    {
        const int numSamples = stepResult.loadedIR.getNumSamples();
        const int numChannels = stepResult.loadedIR.getNumChannels();
        const double threshold = 1.0e-15;

        int newLength = 0;
        if (numChannels > 0)
        {
            const double* ch0 = stepResult.loadedIR.getReadPointer(0);
            const double* ch1 = (numChannels > 1) ? stepResult.loadedIR.getReadPointer(1) : nullptr;

            const __m256d vThreshold = _mm256_set1_pd(threshold);
            const __m256d vSignMask = _mm256_set1_pd(-0.0);

            int i = numSamples;
            bool found = false;
            for (; i >= 4; i -= 4)
            {
                __m256d v0 = _mm256_loadu_pd(ch0 + i - 4);
                __m256d abs0 = _mm256_andnot_pd(vSignMask, v0);
                __m256d mask = _mm256_cmp_pd(abs0, vThreshold, _CMP_GT_OQ);
                if (ch1)
                {
                    __m256d v1 = _mm256_loadu_pd(ch1 + i - 4);
                    __m256d abs1 = _mm256_andnot_pd(vSignMask, v1);
                    mask = _mm256_or_pd(mask, _mm256_cmp_pd(abs1, vThreshold, _CMP_GT_OQ));
                }
                if (_mm256_testz_pd(mask, mask) == 0)
                {
                    for (int j = i - 1; j >= i - 4; --j)
                    {
                        if (std::abs(ch0[j]) > threshold || (ch1 && std::abs(ch1[j]) > threshold))
                        { newLength = j + 1; found = true; break; }
                    }
                    if (found) break;
                }
            }
            if (!found)
            {
                for (int j = i - 1; j >= 0; --j)
                {
                    if (std::abs(ch0[j]) > threshold || (ch1 && std::abs(ch1[j]) > threshold))
                    { newLength = j + 1; break; }
                }
            }
        }
        if (newLength < numSamples)
        {
            stepResult.loadedIR.setSize(numChannels, std::max(1, newLength), true);
            ConvolverProcessorInternal::shrinkToFit(stepResult.loadedIR);
        }
    }

    if (stepResult.loadedSR > 0.0 && sampleRate > 0.0 &&
        std::abs(stepResult.loadedSR - sampleRate) > 1e-6)
    {
        const uint64_t myGen = owner.convolverStateGeneration.getCurrentGeneration();
        const r8b::EDSPFilterPhaseResponse r8bPhase =
            (owner.getResamplingPhaseMode() == ResamplingPhaseMode::Linear)
                ? r8b::fprLinearPhase : r8b::fprMinPhase;

        auto resampleOut = ConvolverProcessorInternal::resampleIR(
            stepResult.loadedIR, stepResult.loadedSR, sampleRate, r8bPhase,
            [&]() -> bool {
                return shouldStop() ||
                       !owner.convolverStateGeneration.isCurrentGeneration(myGen);
            });

        if (!owner.convolverStateGeneration.isCurrentGeneration(myGen))
            return false;

        switch (resampleOut.result)
        {
            case ConvolverProcessorInternal::ResampleResult::Success:
                stepResult.loadedIR = std::move(resampleOut.buffer);
                stepResult.loadedSR = sampleRate;
                break;
            case ConvolverProcessorInternal::ResampleResult::Cancelled:
                return false;
            case ConvolverProcessorInternal::ResampleResult::SilentIR:
                stepResult.errorMessage = "IR is silent (all samples near zero).";
                return false;
            case ConvolverProcessorInternal::ResampleResult::Error:
            default:
                stepResult.errorMessage = "Resampling failed (unknown error).";
                return false;
        }
    }

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    if (stepResult.loadedSR > 0.0 && stepResult.loadedIR.getNumSamples() > 0)
    {
        for (int ch = 0; ch < stepResult.loadedIR.getNumChannels(); ++ch)
        {
            convo::UltraHighRateDCBlocker dcBlocker;
            dcBlocker.init(stepResult.loadedSR, 1.0);
            double* data = stepResult.loadedIR.getWritePointer(ch);
            dcBlocker.process(data, stepResult.loadedIR.getNumSamples());
        }
    }

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    if (stepResult.loadedIR.getNumSamples() > 0)
    {
        const int numSamples = stepResult.loadedIR.getNumSamples();
        for (int ch = 0; ch < stepResult.loadedIR.getNumChannels(); ++ch)
        {
            if (!ConvolverProcessorInternal::applyAsymmetricTukey(stepResult.loadedIR.getWritePointer(ch), numSamples))
            {
                stepResult.errorMessage = "Failed to allocate Tukey window buffer (Out of Memory).";
                return false;
            }
        }
    }

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    stepResult.targetLength = owner.computeTargetIRLength(stepResult.loadedSR, stepResult.loadedIR.getNumSamples());
    stepTrimmed.setSize(stepResult.loadedIR.getNumChannels(), stepResult.targetLength);
    stepTrimmed.clear();

    const int copySamples = std::min(stepResult.targetLength, stepResult.loadedIR.getNumSamples());
    constexpr int minFadeSamples = 256;
    constexpr double fadeRatio = 0.02;
    const int maxFadeSamples = juce::jmax(minFadeSamples, static_cast<int>(std::round(sampleRate * 0.080)));
    int fadeSamples = static_cast<int>(std::round(static_cast<double>(copySamples) * fadeRatio));
    fadeSamples = juce::jlimit(minFadeSamples, maxFadeSamples, fadeSamples);
    fadeSamples = juce::jmax(0, juce::jmin(fadeSamples, copySamples - 1));

    for (int ch = 0; ch < stepResult.loadedIR.getNumChannels(); ++ch)
    {
        stepTrimmed.copyFrom(ch, 0, stepResult.loadedIR, ch, 0, copySamples);
        if (fadeSamples > 0)
            stepTrimmed.applyGainRamp(ch, copySamples - fadeSamples, fadeSamples, 1.0, 0.0);
    }

    return true;
}

bool ConvolverProcessor::LoaderThread::doTransformStep()
{
    auto shouldStop = [this]() -> bool {
        return externalCancellationCheck && externalCancellationCheck();
    };

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    auto validateBuffer = [](const juce::AudioBuffer<double>& buf) -> bool
    {
        if (buf.getNumSamples() <= 0 || buf.getNumChannels() <= 0) return false;
        double maxAbs = 0.0;
        for (int ch = 0; ch < buf.getNumChannels(); ++ch)
        {
            const double* ptr = buf.getReadPointer(ch);
            for (int i = 0; i < buf.getNumSamples(); ++i)
            {
                if (!std::isfinite(ptr[i])) return false;
                maxAbs = std::max(maxAbs, std::abs(ptr[i]));
            }
        }
        return maxAbs > 1.0e-12;
    };

    if (phaseMode == ConvolverProcessor::PhaseMode::Minimum ||
        phaseMode == ConvolverProcessor::PhaseMode::Mixed)
    {
        bool wasCancelled = false;
        auto minPhaseIR = ConvolverProcessorInternal::convertToMinimumPhase(stepTrimmed, shouldStop, &wasCancelled);
        if (wasCancelled) return false;

        if (validateBuffer(minPhaseIR))
        {
            if (phaseMode == ConvolverProcessor::PhaseMode::Minimum)
            {
                stepTrimmed = std::move(minPhaseIR);
            }
            else
            {
                bool mixedCancelled = false;
                auto progressCb = [this](float p) { owner.setLoadingProgress(p); };
                auto mixedIR = convertToMixedPhase(&owner, stepFileHash, stepTrimmed, minPhaseIR,
                                                   sampleRate,
                                                   static_cast<double>(mixedTransitionStartHz),
                                                   static_cast<double>(mixedTransitionEndHz),
                                                   32.0,  // tau (dummy, unused in DSP)
                                                   shouldStop, &mixedCancelled, progressCb);
                if (mixedCancelled) return false;
                if (validateBuffer(mixedIR))
                    stepTrimmed = std::move(mixedIR);
            }
        }
    }

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    {
        const IRState* currentState = owner.acquireIRState();
        auto currentIr = (currentState != nullptr) ? currentState->ir : nullptr;
        const double currentScale = convo::consumeAtomic(owner.currentIRScale, std::memory_order_acquire); // acquire: applyNewState の publishAtomic release と HB
        const auto scaleInfo = IRConverter::computeScaleFactor(stepTrimmed, currentIr, currentScale);
        owner.releaseIRState(currentState);

        stepResult.scaleFactor = scaleInfo.hasScaleFactor ? scaleInfo.scaleFactor : 1.0;
    }

    return true;
}

bool ConvolverProcessor::LoaderThread::doBuildStep()
{
    return buildConvolverFromTrimmed(stepResult, stepTrimmed, sampleRate, blockSize, stepCurrentThread);
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOADER_THREAD
