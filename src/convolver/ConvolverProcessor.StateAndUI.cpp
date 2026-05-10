#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "audioengine/AudioEngine.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "core/EpochManager.h"
#include "AlignedAllocation.h"

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_STATE_UI)

using namespace ConvolverProcessorInternal;

// ────────────────────────────────────────────────────────────────
// State Management and UI Updates
// ────────────────────────────────────────────────────────────────

static void applySmoothing(const float* magnitudes, float* smoothed, int numBins)
{
    if (magnitudes == nullptr || smoothed == nullptr || numBins <= 0) return;

    smoothed[0] = magnitudes[0];
    const float bandwidth = 1.0f / 6.0f;
    const float factor = std::pow(2.0f, bandwidth * 0.5f);

    for (int i = 1; i < numBins; ++i)
    {
        float sum = 0.0f;
        int count = 0;

        int startBin = static_cast<int>(static_cast<float>(i) / factor);
        int endBin = static_cast<int>(static_cast<float>(i) * factor);

        startBin = (std::max)(1, startBin);
        endBin = (std::min)(numBins - 1, endBin);

        for (int j = startBin; j <= endBin; ++j)
        {
            sum += magnitudes[j];
            count++;
        }

        if (count > 0)
            smoothed[i] = sum / static_cast<float>(count);
        else
            smoothed[i] = magnitudes[i];
    }
}

juce::ValueTree ConvolverProcessor::getState() const
{
    juce::ValueTree v ("Convolver");
    v.setProperty ("mix", mixTarget.load(), nullptr);
    v.setProperty ("bypassed", bypassed.load(), nullptr);
    v.setProperty ("phaseMode", static_cast<int>(getPhaseMode()), nullptr);
    v.setProperty ("useMinPhase", getUseMinPhase(), nullptr);
    v.setProperty ("smoothingTime", smoothingTimeSec.load(), nullptr);
    v.setProperty ("irLength", targetIRLengthSec.load(), nullptr);
    v.setProperty ("autoDetectedIRLength", autoDetectedIRLengthSec.load(std::memory_order_acquire), nullptr);
    v.setProperty ("irLengthManualOverride", irLengthManualOverride.load(std::memory_order_acquire), nullptr);
    v.setProperty ("mixedF1Hz", mixedTransitionStartHz.load(std::memory_order_acquire), nullptr);
    v.setProperty ("mixedF2Hz", mixedTransitionEndHz.load(std::memory_order_acquire), nullptr);
    v.setProperty ("mixedTau", mixedPreRingTau.load(std::memory_order_acquire), nullptr);
    v.setProperty ("rebuildDebounceMs", rebuildDebounceMs.load(std::memory_order_acquire), nullptr);
    v.setProperty ("experimentalDirectHeadEnabled", experimentalDirectHeadEnabled.load(std::memory_order_acquire), nullptr);
    v.setProperty ("tailProcessingMode", tailProcessingMode.load(std::memory_order_acquire), nullptr);
    v.setProperty ("tailRolloffStartHz", tailRolloffStartHz.load(std::memory_order_acquire), nullptr);
    v.setProperty ("tailRolloffStrength", tailRolloffStrength.load(std::memory_order_acquire), nullptr);
    v.setProperty ("partitionTailStrength", partitionTailStrength.load(std::memory_order_acquire), nullptr);
    v.setProperty ("targetUpgradeFFTSize", getTargetUpgradeFFTSize(), nullptr);
    v.setProperty ("enableProgressiveUpgrade", isProgressiveUpgradeEnabled(), nullptr);
    v.setProperty ("maxCacheEntries", static_cast<int>(getMaxCacheEntries()), nullptr);
    {
        const juce::ScopedLock sl(irFileLock);
        v.setProperty ("irPath", currentIrFile.getFullPathName(), nullptr);
    }
    return v;
}

void ConvolverProcessor::setState(const juce::ValueTree& v)
{
    if (v.hasProperty ("mix")) setMix (v.getProperty ("mix"));
    if (v.hasProperty ("bypassed")) setBypass (v.getProperty ("bypassed"));
    if (v.hasProperty ("phaseMode"))
    {
        const int modeRaw = static_cast<int>(v.getProperty("phaseMode"));
        const int modeClamped = juce::jlimit(static_cast<int>(PhaseMode::AsIs), static_cast<int>(PhaseMode::Minimum), modeRaw);
        setPhaseMode(static_cast<PhaseMode>(modeClamped));
    }
    else if (v.hasProperty ("useMinPhase"))
    {
        setUseMinPhase (v.getProperty ("useMinPhase"));
    }
    if (v.hasProperty ("smoothingTime")) setSmoothingTime (v.getProperty ("smoothingTime"));

    const bool hasSavedAutoLength = v.hasProperty ("autoDetectedIRLength");
    const bool hasSavedManualOverride = v.hasProperty ("irLengthManualOverride");

    if (hasSavedManualOverride)
    {
        const bool isManual = static_cast<bool>(v.getProperty ("irLengthManualOverride"));

        if (isManual)
        {
            if (hasSavedAutoLength)
            {
                const float autoLength = static_cast<float>(v.getProperty ("autoDetectedIRLength"));
                const float clampedAutoLength = juce::jlimit(IR_LENGTH_MIN_SEC,
                                                             getMaximumAllowedIRLengthSec(currentSampleRate.load(std::memory_order_acquire)),
                                                             autoLength);
                autoDetectedIRLengthSec.store(clampedAutoLength, std::memory_order_release);
            }

            if (v.hasProperty ("irLength"))
                setTargetIRLength (v.getProperty ("irLength"));

            setIRLengthManualOverride (true);
        }
        else
        {
            if (hasSavedAutoLength)
                applyAutoDetectedIRLength (v.getProperty ("autoDetectedIRLength"));
            else if (v.hasProperty ("irLength"))
                applyAutoDetectedIRLength (v.getProperty ("irLength"));

            setIRLengthManualOverride (false);
        }
    }
    else if (v.hasProperty ("irLength"))
    {
        setTargetIRLength (v.getProperty ("irLength"));
    }

    if (v.hasProperty ("mixedF1Hz")) setMixedTransitionStartHz (v.getProperty ("mixedF1Hz"));
    if (v.hasProperty ("mixedF2Hz")) setMixedTransitionEndHz (v.getProperty ("mixedF2Hz"));
    if (v.hasProperty ("mixedTau")) setMixedPreRingTau (v.getProperty ("mixedTau"));
    if (v.hasProperty ("rebuildDebounceMs")) setRebuildDebounceMs (static_cast<int>(v.getProperty("rebuildDebounceMs")));
    if (v.hasProperty ("experimentalDirectHeadEnabled")) setExperimentalDirectHeadEnabled (v.getProperty ("experimentalDirectHeadEnabled"));
    if (v.hasProperty ("targetUpgradeFFTSize")) setTargetUpgradeFFTSize (static_cast<int>(v.getProperty("targetUpgradeFFTSize")));
    if (v.hasProperty ("enableProgressiveUpgrade")) setEnableProgressiveUpgrade (static_cast<bool>(v.getProperty("enableProgressiveUpgrade")));
    if (v.hasProperty ("maxCacheEntries")) setMaxCacheEntries (static_cast<size_t>(static_cast<int>(v.getProperty("maxCacheEntries"))));

    const bool hasTailMode = v.hasProperty("tailProcessingMode");
    const bool hasTailStart = v.hasProperty("tailRolloffStartHz");
    const bool hasTailStrength = v.hasProperty("tailRolloffStrength");
    const bool hasPartitionTailStrength = v.hasProperty("partitionTailStrength");
    const bool hasAnyTailKey = hasTailMode || hasTailStart || hasTailStrength || hasPartitionTailStrength;

    if (hasAnyTailKey)
    {
        const int resolvedMode = hasTailMode
            ? juce::jlimit(0, 1, static_cast<int>(v.getProperty("tailProcessingMode")))
            : 0;

        if (hasTailMode)
            setTailProcessingMode(static_cast<int>(v.getProperty("tailProcessingMode")));
        else
            setTailProcessingMode(0);

        if (hasTailStart)
            setTailRolloffStartHz(static_cast<float>(v.getProperty("tailRolloffStartHz")));
        else
            setTailRolloffStartHz(resolvedMode == 0 ? TAIL_AIR_ROLLOFF_START_DEFAULT_HZ
                                                    : TAIL_LAYER_ROLLOFF_START_DEFAULT_HZ);

        if (hasTailStrength)
            setTailRolloffStrength(static_cast<float>(v.getProperty("tailRolloffStrength")));
        else
            setTailRolloffStrength(resolvedMode == 0 ? TAIL_AIR_ROLLOFF_STRENGTH_DEFAULT
                                                     : TAIL_LAYER_ROLLOFF_STRENGTH_DEFAULT);

        if (hasPartitionTailStrength)
            setPartitionTailStrength(static_cast<float>(v.getProperty("partitionTailStrength")));
        else
            setPartitionTailStrength(TAIL_PARTITION_STRENGTH_DEFAULT);
    }
    else
    {
        setTailProcessingMode(0);
        setTailRolloffStartHz(TAIL_ROLLOFF_START_DEFAULT_HZ);
        setTailRolloffStrength(0.0f);
        setPartitionTailStrength(TAIL_PARTITION_STRENGTH_DEFAULT);
    }

    if (v.hasProperty ("irPath"))
    {
        juce::File fileToLoad;
        juce::String path = v.getProperty ("irPath").toString();
        if (path.isNotEmpty())
        {
            juce::File f (path);
            if (f.existsAsFile())
            {
                const juce::ScopedLock sl(irFileLock);
                if (f != currentIrFile)
                    fileToLoad = f;
            }
            else
            {
                lastError = "IR not found: " + f.getFileName();
                postCoalescedChangeNotification();
            }
        }

        if (fileToLoad.existsAsFile())
            loadIR(fileToLoad);
    }
}

void ConvolverProcessor::syncStateFrom(const ConvolverProcessor& other)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    mixTarget.store(other.mixTarget.load(), std::memory_order_release);
    bypassed.store(other.bypassed.load(), std::memory_order_release);
    phaseMode.store(other.phaseMode.load(std::memory_order_acquire), std::memory_order_release);
    smoothingTimeSec.store(other.smoothingTimeSec.load(), std::memory_order_release);
    targetIRLengthSec.store(other.targetIRLengthSec.load(), std::memory_order_release);
    mixedTransitionStartHz.store(other.mixedTransitionStartHz.load(std::memory_order_acquire), std::memory_order_release);
    mixedTransitionEndHz.store(other.mixedTransitionEndHz.load(std::memory_order_acquire), std::memory_order_release);
    mixedPreRingTau.store(other.mixedPreRingTau.load(std::memory_order_acquire), std::memory_order_release);
    rebuildDebounceMs.store(other.rebuildDebounceMs.load(std::memory_order_acquire), std::memory_order_release);
    experimentalDirectHeadEnabled.store(other.experimentalDirectHeadEnabled.load(std::memory_order_acquire), std::memory_order_release);
    tailProcessingMode.store(other.tailProcessingMode.load(std::memory_order_acquire), std::memory_order_release);
    tailRolloffStartHz.store(other.tailRolloffStartHz.load(std::memory_order_acquire), std::memory_order_release);
    tailRolloffStrength.store(other.tailRolloffStrength.load(std::memory_order_acquire), std::memory_order_release);
    partitionTailStrength.store(other.partitionTailStrength.load(std::memory_order_acquire), std::memory_order_release);
    targetUpgradeFFTSize.store(other.targetUpgradeFFTSize.load(std::memory_order_acquire), std::memory_order_release);
    enableProgressiveUpgrade.store(other.enableProgressiveUpgrade.load(std::memory_order_acquire), std::memory_order_release);
    maxCacheEntries.store(other.maxCacheEntries.load(std::memory_order_acquire), std::memory_order_release);

    if (const IRState* otherState = other.acquireIRState())
    {
        if (otherState->irOwner)
            updateIRState(otherState->irOwner, otherState->sampleRate);
        releaseIRState(otherState);
    }
    {
        const juce::ScopedLock sl(irFileLock);
        currentIrFile = other.currentIrFile;
    }
    irName = other.irName;
    irLength.store(other.irLength.load(std::memory_order_acquire), std::memory_order_release);
    currentIRScale.store(other.currentIRScale.load(std::memory_order_acquire), std::memory_order_release);

    nucHCMode.store(other.nucHCMode.load(std::memory_order_acquire), std::memory_order_release);
    nucLCMode.store(other.nucLCMode.load(std::memory_order_acquire), std::memory_order_release);

    const uint64_t retireEpoch = rcuProvider ? rcuProvider->publishRcuEpoch() : 1;
    auto* oldConv = m_activeEngine.exchange(nullptr, std::memory_order_acq_rel);
    if (oldConv)
        retireStereoConvolver(oldConv, retireEpoch);
}

void ConvolverProcessor::syncParametersFrom(const ConvolverProcessor& other)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    mixTarget.store(other.mixTarget.load(), std::memory_order_release);
    bypassed.store(other.bypassed.load(), std::memory_order_release);
    smoothingTimeSec.store(other.smoothingTimeSec.load(), std::memory_order_release);
    rebuildDebounceMs.store(other.rebuildDebounceMs.load(std::memory_order_acquire), std::memory_order_release);

    if (std::abs(currentSampleRate.load() - other.currentSampleRate.load()) < 1e-6)
    {
        struct GlobalGuard {
            const ConvolverProcessor& cp;
            GlobalGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterGlobalReader(2); }
            ~GlobalGuard() { cp.exitGlobalReader(2); }
        } guard(other);

        auto* otherConv = other.m_activeEngine.load(std::memory_order_acquire);
        auto* expectedConv = m_activeEngine.load(std::memory_order_acquire);

        if (otherConv != expectedConv && otherConv != nullptr)
        {
            shareConvolutionEngineFrom(other);
        }
    }
}

void ConvolverProcessor::shareConvolutionEngineFrom(const ConvolverProcessor& other)
{
    struct GlobalGuard {
        const ConvolverProcessor& cp;
        GlobalGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterGlobalReader(2); }
        ~GlobalGuard() { cp.exitGlobalReader(2); }
    } guard(other);

    auto* otherConv = other.m_activeEngine.load(std::memory_order_acquire);
    if (otherConv == nullptr)
        return;

    auto* clonedConv = otherConv->clone();
    if (clonedConv == nullptr)
        return;

    const uint64_t retireEpoch = rcuProvider ? rcuProvider->publishRcuEpoch() : 1;
    auto* oldConv = m_activeEngine.exchange(clonedConv, std::memory_order_acq_rel);
    if (oldConv)
        retireStereoConvolver(oldConv, retireEpoch);

    auto* otherSnap = other.cachedLatency.load(std::memory_order_acquire);
    auto* newSnap = otherSnap ? new LatencySnapshot(*otherSnap) : new LatencySnapshot();
    auto* oldSnap = cachedLatency.exchange(newSnap, std::memory_order_acq_rel);
    delete oldSnap;

    irLength.store(other.irLength.load(std::memory_order_acquire), std::memory_order_release);
    uiAlgorithmLatencySamples.store(other.uiAlgorithmLatencySamples.load(std::memory_order_acquire), std::memory_order_release);
    uiIrPeakLatencySamples.store(other.uiIrPeakLatencySamples.load(std::memory_order_acquire), std::memory_order_release);
    uiTotalLatencySamples.store(other.uiTotalLatencySamples.load(std::memory_order_acquire), std::memory_order_release);
    uiDirectHeadActive.store(other.uiDirectHeadActive.load(std::memory_order_acquire), std::memory_order_release);
    requestHostDisplayUpdate();
}

ConvolverProcessor::IRLoadPreview ConvolverProcessor::analyzeImpulseResponseFile(const juce::File& irFile, double processingSampleRate)
{
    IRLoadPreview preview;
    preview.recommendedMaxSec = IR_LENGTH_MAX_SEC;
    preview.hardMaxSec = getMaximumAllowedIRLengthSecForSampleRate(processingSampleRate);

    juce::AudioBuffer<double> loadedIR;
    double loadedSampleRate = 0.0;
    if (!loadImpulseResponsePreviewFile(irFile, loadedIR, loadedSampleRate, preview.errorMessage))
        return preview;

    const auto neverCancel = []() { return false; };

    if (loadedIR.getNumSamples() > 0)
    {
        const int numSamples = loadedIR.getNumSamples();
        const int numChannels = loadedIR.getNumChannels();
        const double threshold = 1.0e-15;
        int newLength = 0;

        if (numChannels > 0)
        {
            const double* ch0Ptr = loadedIR.getReadPointer(0);
            const double* ch1Ptr = (numChannels > 1) ? loadedIR.getReadPointer(1) : nullptr;

            for (int j = numSamples - 1; j >= 0; --j)
            {
                if (std::abs(ch0Ptr[j]) > threshold || (ch1Ptr && std::abs(ch1Ptr[j]) > threshold))
                {
                    newLength = j + 1;
                    break;
                }
            }
        }

        if (newLength < numSamples)
        {
            loadedIR.setSize(numChannels, juce::jmax(1, newLength), true);
            shrinkToFit(loadedIR);
        }
    }

    if (loadedSampleRate > 0.0 && processingSampleRate > 0.0 && std::abs(loadedSampleRate - processingSampleRate) > 1e-6)
    {
        auto resampleOut = resampleIR(loadedIR, loadedSampleRate, processingSampleRate,
                                      r8b::fprLinearPhase, neverCancel);
        if (resampleOut.result != ResampleResult::Success ||
            resampleOut.buffer.getNumSamples() == 0)
        {
            preview.errorMessage = "Resampling failed (unknown error).";
            return preview;
        }

        loadedIR = std::move(resampleOut.buffer);
        loadedSampleRate = processingSampleRate;
    }

    if (loadedSampleRate > 0.0 && loadedIR.getNumSamples() > 0)
    {
        for (int ch = 0; ch < loadedIR.getNumChannels(); ++ch)
        {
            convo::UltraHighRateDCBlocker dcBlocker;
            dcBlocker.init(loadedSampleRate, 1.0);
            dcBlocker.process(loadedIR.getWritePointer(ch), loadedIR.getNumSamples());
        }
    }

    if (loadedIR.getNumSamples() > 0)
    {
        const int numSamples = loadedIR.getNumSamples();
        for (int ch = 0; ch < loadedIR.getNumChannels(); ++ch)
        {
            if (!applyAsymmetricTukey(loadedIR.getWritePointer(ch), numSamples))
            {
                preview.errorMessage = "Failed to allocate Tukey window buffer (Out of Memory).";
                return preview;
            }
        }
    }

    const int detectedSamples = estimateEffectiveIRLengthSamples(loadedIR, loadedSampleRate);
    preview.autoDetectedLengthSamples = detectedSamples;
    preview.autoDetectedLengthSec = (loadedSampleRate > 0.0)
                                  ? static_cast<float>(static_cast<double>(detectedSamples) / loadedSampleRate)
                                  : IR_LENGTH_DEFAULT_SEC;
    preview.exceedsRecommended = preview.autoDetectedLengthSec > preview.recommendedMaxSec;
    preview.exceedsHardLimit = preview.autoDetectedLengthSec > preview.hardMaxSec;
    preview.success = true;
    return preview;
}

std::vector<float> ConvolverProcessor::getIRWaveform() const
{
    const juce::ScopedLock sl(visualizationDataLock);
    return irWaveform;
}

std::vector<float> ConvolverProcessor::getIRMagnitudeSpectrum() const
{
    const juce::ScopedLock sl(visualizationDataLock);
    return irMagnitudeSpectrum;
}

double ConvolverProcessor::getIRSpectrumSampleRate() const
{
    const juce::ScopedLock sl(visualizationDataLock);
    return irSpectrumSampleRate;
}

void ConvolverProcessor::createWaveformSnapshot(const juce::AudioBuffer<double>& irBuffer)
{
    const juce::ScopedLock sl(visualizationDataLock);

    irWaveform.assign(WAVEFORM_POINTS, 0.0f);

    const int numSamples = irBuffer.getNumSamples();
    const int numChannels = irBuffer.getNumChannels();

    if (numSamples <= 0 || numChannels <= 0)
        return;

    const int samplesPerPoint = (std::max)(1, numSamples / WAVEFORM_POINTS);

    float maxAbs = 0.0f;

    for (int i = 0; i < WAVEFORM_POINTS; ++i)
    {
        float peak = 0.0f;
        int startSample = i * samplesPerPoint;
        int endSample = (std::min)(numSamples, startSample + samplesPerPoint);

        for (int ch = 0; ch < numChannels; ++ch)
            for (int j = startSample; j < endSample; ++j)
                peak = (std::max)(peak, static_cast<float>(std::abs(irBuffer.getReadPointer(ch)[j])));

        irWaveform[i] = peak;
        maxAbs = (std::max)(maxAbs, peak);
    }

    if (maxAbs > 0.0f)
        for (float& val : irWaveform) val /= maxAbs;
}

void ConvolverProcessor::createFrequencyResponseSnapshot(const juce::AudioBuffer<double>& irBuffer, double sampleRate)
{
    const juce::ScopedLock sl(visualizationDataLock);

    irSpectrumSampleRate = sampleRate;
    irMagnitudeSpectrum.clear();

    const int numSamples = irBuffer.getNumSamples();
    if (numSamples <= 0 || irBuffer.getNumChannels() < 1) return;

    int fftSize = juce::nextPowerOfTwo(numSamples);
    const int maxFFTSize = 65536;
    if (fftSize > maxFFTSize) fftSize = maxFFTSize;
    if (fftSize < 512) fftSize = 512;

    if (cachedFFTBufferCapacity < fftSize * 2)
    {
        cachedFFTBuffer.reset(static_cast<float*>(convo::aligned_malloc(fftSize * 2 * sizeof(float), 64)));
        cachedFFTBufferCapacity = fftSize * 2;
    }

    juce::FloatVectorOperations::clear(cachedFFTBuffer.get(), fftSize * 2);

    const double* src = irBuffer.getReadPointer(0);
    const int copyLen = (std::min)(numSamples, fftSize);
    float* dst = cachedFFTBuffer.get();

    if (fftHandle && fftHandleSize != fftSize)
    {
        DftiFreeDescriptor(&fftHandle);
        fftHandle = nullptr;
        fftHandleSize = 0;
    }

    if (!fftHandle)
    {
        if (DftiCreateDescriptor(&fftHandle, DFTI_SINGLE, DFTI_COMPLEX, 1, fftSize) != DFTI_NO_ERROR) return;
        if (DftiSetValue(fftHandle, DFTI_PLACEMENT, DFTI_INPLACE) != DFTI_NO_ERROR) { DftiFreeDescriptor(&fftHandle); fftHandle = nullptr; return; }
        if (DftiCommitDescriptor(fftHandle) != DFTI_NO_ERROR) { DftiFreeDescriptor(&fftHandle); fftHandle = nullptr; return; }
        fftHandleSize = fftSize;
    }

    for (int i = 0; i < copyLen; ++i) {
        dst[2 * i] = static_cast<float>(src[i]);
        dst[2 * i + 1] = 0.0f;
    }
    for (int i = copyLen; i < fftSize; ++i) {
        dst[2 * i] = 0.0f;
        dst[2 * i + 1] = 0.0f;
    }

    if (DftiComputeForward(fftHandle, dst) != DFTI_NO_ERROR) return;

    const int numBins = fftSize / 2 + 1;
    float* magBuf = dst + fftSize + 16;
    vcAbs(numBins, reinterpret_cast<const MKL_Complex8*>(dst), magBuf);
    std::memcpy(dst, magBuf, numBins * sizeof(float));

    if (cachedMagnitudeBufferCapacity < numBins)
    {
        cachedLinearMagsBuffer.reset(static_cast<float*>(convo::aligned_malloc(static_cast<size_t>(numBins) * sizeof(float), 64)));
        cachedSmoothedMagsBuffer.reset(static_cast<float*>(convo::aligned_malloc(static_cast<size_t>(numBins) * sizeof(float), 64)));
        if (!cachedLinearMagsBuffer || !cachedSmoothedMagsBuffer)
        {
            cachedMagnitudeBufferCapacity = 0;
            return;
        }
        cachedMagnitudeBufferCapacity = numBins;
    }

    std::memcpy(cachedLinearMagsBuffer.get(), cachedFFTBuffer.get(), static_cast<size_t>(numBins) * sizeof(float));
    applySmoothing(cachedLinearMagsBuffer.get(), cachedSmoothedMagsBuffer.get(), numBins);

    irMagnitudeSpectrum.resize(numBins);

    for (int i = 0; i < numBins; ++i)
    {
        float mag = cachedSmoothedMagsBuffer[i];
        irMagnitudeSpectrum[i] = (mag > 1e-9f) ? juce::Decibels::gainToDecibels(mag) : -100.0f;
    }
}

ConvolverProcessor::LatencyBreakdown ConvolverProcessor::getLatencyBreakdown() const
{
    struct GlobalGuard {
        const ConvolverProcessor& cp;
        GlobalGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterGlobalReader(2); }
        ~GlobalGuard() { cp.exitGlobalReader(2); }
    } guard(*this);

    LatencyBreakdown breakdown;
    if (auto* conv = m_activeEngine.load(std::memory_order_acquire))
    {
        const bool directHeadActive = conv->storedDirectHeadEnabled;
        breakdown.directHeadActive = directHeadActive;
        breakdown.algorithmLatencySamples = directHeadActive ? 0 : juce::jmax(0, conv->latency);
        breakdown.irPeakLatencySamples = juce::jmax(0, conv->irLatency);
        breakdown.totalLatencySamples = juce::jmax(0,
            breakdown.algorithmLatencySamples + breakdown.irPeakLatencySamples);

        if (breakdown.algorithmLatencySamples == 0 &&
            breakdown.irPeakLatencySamples == 0 &&
            breakdown.totalLatencySamples == 0)
        {
            const int snapTotal = uiTotalLatencySamples.load(std::memory_order_acquire);
            if (snapTotal > 0)
            {
                breakdown.algorithmLatencySamples = uiAlgorithmLatencySamples.load(std::memory_order_acquire);
                breakdown.irPeakLatencySamples = uiIrPeakLatencySamples.load(std::memory_order_acquire);
                breakdown.totalLatencySamples = snapTotal;
                breakdown.directHeadActive = uiDirectHeadActive.load(std::memory_order_acquire);
            }
        }
    }

    if (breakdown.algorithmLatencySamples == 0 &&
        breakdown.irPeakLatencySamples == 0 &&
        breakdown.totalLatencySamples == 0)
    {
        const int snapTotal = uiTotalLatencySamples.load(std::memory_order_acquire);
        if (snapTotal > 0)
        {
            breakdown.algorithmLatencySamples = uiAlgorithmLatencySamples.load(std::memory_order_acquire);
            breakdown.irPeakLatencySamples = uiIrPeakLatencySamples.load(std::memory_order_acquire);
            breakdown.totalLatencySamples = snapTotal;
            breakdown.directHeadActive = uiDirectHeadActive.load(std::memory_order_acquire);
        }
    }

    return breakdown;
}

int ConvolverProcessor::getLatencySamples() const
{
    auto snap = cachedLatency.load(std::memory_order_acquire);
    return snap ? snap->totalLatencySamples : 0;
}

int ConvolverProcessor::getTotalLatencySamples() const
{
    auto snap = cachedLatency.load(std::memory_order_acquire);
    return snap ? snap->totalLatencySamples : 0;
}

void ConvolverProcessor::updateLatencyCache() noexcept
{
    LatencySnapshot snap;
    const auto breakdown = getLatencyBreakdown();
    snap.algorithmLatencySamples = breakdown.algorithmLatencySamples;
    snap.irPeakLatencySamples = breakdown.irPeakLatencySamples;
    snap.totalLatencySamples = breakdown.totalLatencySamples;
    snap.hasParallelDryPath = breakdown.directHeadActive;

    auto* newSnap = new LatencySnapshot(snap);
    auto* oldSnap = cachedLatency.exchange(newSnap, std::memory_order_acq_rel);
    delete oldSnap;
}

void ConvolverProcessor::requestHostDisplayUpdate()
{
    auto snap = cachedLatency.load(std::memory_order_acquire);
    const int total = snap ? snap->totalLatencySamples : 0;
    if (total == lastReportedLatency)
        return;

    if (latencyChangePending.exchange(true, std::memory_order_acq_rel))
        return;

    const bool queued = juce::MessageManager::callAsync([weakThis = juce::WeakReference<ConvolverProcessor>(this)]
    {
        if (auto* self = weakThis.get())
        {
            auto snap2 = self->cachedLatency.load(std::memory_order_acquire);
            const int latest = snap2 ? snap2->totalLatencySamples : 0;
            if (latest != self->lastReportedLatency)
            {
                self->lastReportedLatency = latest;
                self->postCoalescedChangeNotification();
            }
            self->latencyChangePending.store(false, std::memory_order_release);
        }
    });

    if (!queued)
        latencyChangePending.store(false, std::memory_order_release);
}

void ConvolverProcessor::setPhaseMode(PhaseMode mode)
{
    const int newMode = static_cast<int>(mode);
    const int oldMode = phaseMode.exchange(newMode, std::memory_order_acq_rel);
    if (oldMode != newMode)
    {
        listeners.call(&Listener::convolverParamsChanged, this);
        requestDebouncedRebuild();
    }
}

void ConvolverProcessor::setUseMinPhase(bool shouldUseMinPhase)
{
    setPhaseMode(shouldUseMinPhase ? PhaseMode::Minimum : PhaseMode::AsIs);
}

void ConvolverProcessor::setNUCFilterModes(convo::HCMode hcMode, convo::LCMode lcMode)
{
    const int newHC = static_cast<int>(hcMode);
    const int newLC = static_cast<int>(lcMode);

    const bool changed = (nucHCMode.exchange(newHC) != newHC) ||
                         (nucLCMode.exchange(newLC) != newLC);

    if (changed)
        requestDebouncedRebuild();
}

void ConvolverProcessor::setTailProcessingMode(int mode)
{
    const int clamped = juce::jlimit(0, 1, mode);
    const int prev = tailProcessingMode.exchange(clamped, std::memory_order_acq_rel);
    if (prev != clamped)
    {
        listeners.call(&Listener::convolverParamsChanged, this);
        requestDebouncedRebuild();
    }
}

void ConvolverProcessor::setTailRolloffStartHz(float hz)
{
    const float clamped = juce::jlimit(TAIL_ROLLOFF_START_MIN_HZ, TAIL_ROLLOFF_START_MAX_HZ, hz);
    const float prev = tailRolloffStartHz.exchange(clamped, std::memory_order_acq_rel);
    if (std::abs(prev - clamped) > 1.0e-5f)
    {
        listeners.call(&Listener::convolverParamsChanged, this);
        requestDebouncedRebuild();
    }
}

void ConvolverProcessor::setTailRolloffStrength(float strength)
{
    const float clamped = juce::jlimit(TAIL_ROLLOFF_STRENGTH_MIN, TAIL_ROLLOFF_STRENGTH_MAX, strength);
    const float prev = tailRolloffStrength.exchange(clamped, std::memory_order_acq_rel);
    if (std::abs(prev - clamped) > 1.0e-5f)
    {
        listeners.call(&Listener::convolverParamsChanged, this);
        requestDebouncedRebuild();
    }
}

void ConvolverProcessor::setPartitionTailStrength(float strength)
{
    const float clamped = juce::jlimit(TAIL_PARTITION_STRENGTH_MIN, TAIL_PARTITION_STRENGTH_MAX, strength);
    const float prev = partitionTailStrength.exchange(clamped, std::memory_order_acq_rel);
    if (std::abs(prev - clamped) > 1.0e-5f)
    {
        listeners.call(&Listener::convolverParamsChanged, this);
        requestDebouncedRebuild();
    }
}

uint64_t ConvolverProcessor::getStructuralHash() const noexcept
{
    uint64_t hash = 0x9e3779b97f4a7c15ULL;

    auto hashCombine = [&hash](uint64_t value) {
        hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    };

    hashCombine(activeCacheKey.load(std::memory_order_acquire));
    hashCombine(static_cast<uint64_t>(irLength.load(std::memory_order_acquire)));
    hashCombine(static_cast<uint64_t>(phaseMode.load(std::memory_order_acquire)));

    auto floatBits = [](float f) -> uint32_t {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        return bits;
    };
    hashCombine(floatBits(mixedTransitionStartHz.load(std::memory_order_acquire)));
    hashCombine(floatBits(mixedTransitionEndHz.load(std::memory_order_acquire)));
    hashCombine(floatBits(mixedPreRingTau.load(std::memory_order_acquire)));

    hashCombine(experimentalDirectHeadEnabled.load(std::memory_order_acquire) ? 1ULL : 0ULL);
    hashCombine(static_cast<uint64_t>(nucHCMode.load(std::memory_order_acquire)));
    hashCombine(static_cast<uint64_t>(nucLCMode.load(std::memory_order_acquire)));
    hashCombine(static_cast<uint64_t>(getTailProcessingMode()));
    hashCombine(floatBits(getTailRolloffStartHz()));
    hashCombine(floatBits(getTailRolloffStrength()));
    hashCombine(floatBits(getPartitionTailStrength()));

    return hash;
}

uint64_t ConvolverProcessor::getActiveCacheKey() const noexcept
{
    return activeCacheKey.load(std::memory_order_acquire);
}

int ConvolverProcessor::getActiveCacheFFTSize() const noexcept
{
    return activeCacheFFTSize.load(std::memory_order_acquire);
}

int ConvolverProcessor::getNUCHCMode() const noexcept
{
    return nucHCMode.load(std::memory_order_acquire);
}

int ConvolverProcessor::getNUCLCMode() const noexcept
{
    return nucLCMode.load(std::memory_order_acquire);
}

void ConvolverProcessor::setResamplingPhaseMode(ResamplingPhaseMode mode)
{
    currentResamplingPhaseMode.store(mode, std::memory_order_release);
    requestDebouncedRebuild();
}

ConvolverProcessor::ResamplingPhaseMode ConvolverProcessor::getResamplingPhaseMode() const
{
    return currentResamplingPhaseMode.load(std::memory_order_acquire);
}

float ConvolverProcessor::getMaximumAllowedIRLengthSecForSampleRate(double sampleRate)
{
    if (sampleRate <= 0.0)
        return IR_LENGTH_MAX_SEC;

    return static_cast<float>(static_cast<double>(MAX_IR_LATENCY) / sampleRate);
}

float ConvolverProcessor::getMaximumAllowedIRLengthSec(double sampleRate) const
{
    const double sr = (sampleRate > 0.0)
                    ? sampleRate
                    : currentSampleRate.load(std::memory_order_acquire);

    return getMaximumAllowedIRLengthSecForSampleRate(sr);
}

int ConvolverProcessor::computeTargetIRLength(double sampleRate, int originalLength) const
{
    juce::ignoreUnused(originalLength);
    const double targetIRTimeSec = targetIRLengthSec.load();
    static constexpr int kMaxIRCap = MAX_IR_LATENCY;

    int target = static_cast<int>(sampleRate * targetIRTimeSec);

    target = (std::min)(target, kMaxIRCap);
    target = (std::max)(target, 1);

    return target;
}

void ConvolverProcessor::debugCheckAtomicLockFree() const
{
   #if JUCE_DEBUG
    if (!std::atomic<LatencySnapshot>{}.is_lock_free())
    {
        DBG("[ConvolverProcessor] std::atomic<LatencySnapshot> is not lock-free on this build; continuing with implementation-provided atomic semantics.");
    }
   #endif
}

void ConvolverProcessor::forceCleanup()
{
    if (rebuildJob)
    {
        rebuildJob->reset();
        rebuildJob.reset();
    }

    std::deque<std::unique_ptr<LoaderThread>> loadersToDelete;
    loadersToDelete.swap(loaderTrashBin);
    if (activeLoader)
        loadersToDelete.push_back(std::move(activeLoader));

    for (auto& loader : loadersToDelete)
    {
        if (loader)
            loader->stopThread(500);
    }
    loadersToDelete.clear();
}

void ConvolverProcessor::updateConvolverState(convo::ConvolverState* newState)
{
    JUCE_ASSERT_MESSAGE_THREAD;
    jassert(newState != nullptr);
    if (!newState) return;

    jassert(!writerActive.exchange(true, std::memory_order_acquire));

    if (!convolverStateGeneration.isCurrentGeneration(newState->generationId))
    {
        juce::Logger::writeToLog("ConvolverProcessor::updateConvolverState: stale generation, discarding state (gen="
            + juce::String((int)newState->generationId) + ")");
        delete newState;
        writerActive.store(false, std::memory_order_release);
        return;
    }

    rcuSwapper.swap(newState);
    convolverState.store(const_cast<convo::ConvolverState*>(newState), std::memory_order_release);
    convo::EpochManager::instance().advanceEpoch();

    writerActive.store(false, std::memory_order_release);
}

void ConvolverProcessor::updateConvolverState(std::unique_ptr<convo::ConvolverState> newState)
{
    updateConvolverState(newState.release());
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_STATE_UI
