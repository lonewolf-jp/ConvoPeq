#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include <mkl.h>

#include "audioengine/AtomicAccess.h"

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_MIXED_PHASE)

using namespace ConvolverProcessorInternal;

// ────────────────────────────────────────────────────────────────
// Mixed Phase Conversion
// ────────────────────────────────────────────────────────────────

juce::AudioBuffer<double> ConvolverProcessor::convertToMixedPhase(ConvolverProcessor* owner,
                                                               uint64_t fileHash,
                                                               const juce::AudioBuffer<double>& linearIR,
                                                               const juce::AudioBuffer<double>& minimumIR,
                                                               double sampleRate,
                                                               double transitionLoHz,
                                                               double transitionHiHz,
                                                               double tau,
                                                               const std::function<bool()>& shouldExit,
                                                               bool* wasCancelled,
                                                               std::function<void(float)> progressCallback)
{
    const auto setMixedPhaseState = [owner](int state)
    {
        if (owner != nullptr)
            convo::publishAtomic(owner->mixedPhaseState, state, std::memory_order_release);
    };

    auto result = convertToMixedPhaseAllpass(owner, fileHash, linearIR, minimumIR, sampleRate,
                                             transitionLoHz, transitionHiHz,
                                             tau, shouldExit, wasCancelled, progressCallback);

    if (result.getNumSamples() > 0)
    {
        setMixedPhaseState(2);
        return result;
    }

    if (result.getNumSamples() == 0 && (wasCancelled == nullptr || !*wasCancelled))
    {
        juce::Logger::writeToLog("Allpass design failed, falling back to Phase 1.");
        auto fallbackResult = convertToMixedPhaseFallback(linearIR, minimumIR, sampleRate,
                                           transitionLoHz, transitionHiHz,
                                           tau, shouldExit, wasCancelled);
        if (fallbackResult.getNumSamples() > 0)
        {
            setMixedPhaseState(2);
            juce::Logger::writeToLog("[MixedPhase] State -> Completed (fallback)");
            if (progressCallback) progressCallback(1.0f);
        }
        else
        {
            setMixedPhaseState(0);
        }
        return fallbackResult;
    }

    setMixedPhaseState(0);
    return result;
}

juce::AudioBuffer<double> ConvolverProcessor::convertToMixedPhaseAllpass(ConvolverProcessor* owner,
                                                               uint64_t fileHash,
                                                               const juce::AudioBuffer<double>& linearIR,
                                                               const juce::AudioBuffer<double>& minimumIR,
                                                               double sampleRate,
                                                               double transitionLoHz,
                                                               double transitionHiHz,
                                                               double tau,
                                                               const std::function<bool()>& shouldExit,
                                                               bool* wasCancelled,
                                                               std::function<void(float)> progressCallback)
{
    const auto setMixedPhaseState = [owner](int state)
    {
        if (owner != nullptr)
            convo::publishAtomic(owner->mixedPhaseState, state, std::memory_order_release);
    };

    if (wasCancelled) *wasCancelled = false;

    if (owner && fileHash != 0) {
        ConvolverProcessor::IRCacheKey key;
        key.fileHash = fileHash;
        key.sampleRate = sampleRate;
        key.phaseMode = ConvolverProcessor::PhaseMode::Mixed;
        key.f1 = static_cast<float>(transitionLoHz);
        key.f2 = static_cast<float>(transitionHiHz);
        key.tau = static_cast<float>(tau);
        key.targetLength = linearIR.getNumSamples();

        const juce::ScopedLock sl(owner->cacheMutex);
        auto it = owner->irCache.find(key);
        if (it != owner->irCache.end()) {
            it->second.lastUsedTime = juce::Time::getMillisecondCounter();
            if (it->second.ir) {
                juce::Logger::writeToLog("convertToMixedPhaseAllpass: Cache Hit!");
                setMixedPhaseState(2);
                juce::Logger::writeToLog("[MixedPhase] State -> Completed (cache hit)");
                if (progressCallback) progressCallback(1.0f);
                return *(it->second.ir);
            }
        }
    }

#if defined(__AVX2__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

    const int numSamples = linearIR.getNumSamples();
    const int numChannels = linearIR.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0)
        return {};

    if (minimumIR.getNumSamples() != numSamples || minimumIR.getNumChannels() != numChannels || sampleRate <= 0.0)
        return {};

    if (transitionHiHz <= transitionLoHz)
        return {};

    setMixedPhaseState(1);
    juce::Logger::writeToLog("[MixedPhase] State -> Optimizing");

    const int fftSize = juce::nextPowerOfTwo(numSamples * 4);
    static constexpr int MAX_MIXED_FFT_SIZE = 8388608;
    if (fftSize > MAX_MIXED_FFT_SIZE)
    {
        juce::Logger::writeToLog("convertToMixedPhaseAllpass: fftSize (" + juce::String(fftSize) + ") exceeds limit.");
        return {};
    }

    convo::ScopedDftiDescriptor dfti;
    const MKL_LONG len = static_cast<MKL_LONG>(fftSize);
    if (DftiCreateDescriptor(dfti.put(), DFTI_DOUBLE, DFTI_COMPLEX, 1, len) != DFTI_NO_ERROR)
        return {};
    if (DftiSetValue(dfti.handle, DFTI_PLACEMENT, DFTI_INPLACE) != DFTI_NO_ERROR)
        return {};
    if (DftiSetValue(dfti.handle, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(fftSize)) != DFTI_NO_ERROR)
        return {};
    if (DftiCommitDescriptor(dfti.handle) != DFTI_NO_ERROR)
        return {};

    const int half = fftSize / 2;
    const int complexSize = half + 1;

    auto linearSpec = convo::makeAlignedArray<MKL_Complex16>(static_cast<size_t>(fftSize));
    auto minimumSpec = convo::makeAlignedArray<MKL_Complex16>(static_cast<size_t>(fftSize));
    auto targetPhase = convo::makeAlignedArray<double>(static_cast<size_t>(complexSize));

    if (!linearSpec || !minimumSpec || !targetPhase)
        return {};

    const double invSpan = 1.0 / (transitionHiHz - transitionLoHz);
    juce::AudioBuffer<double> mixedIR(numChannels, numSamples);

    bool reuseMixedDesignAcrossChannels = false;
    if (numChannels > 1)
    {
        auto channelsAlmostEqual = [numSamples](const juce::AudioBuffer<double>& buffer,
                                                int refChannel,
                                                int targetChannel,
                                                double tolerance) -> bool
        {
            const double* ref = buffer.getReadPointer(refChannel);
            const double* dst = buffer.getReadPointer(targetChannel);
            for (int i = 0; i < numSamples; ++i)
            {
                if (std::abs(ref[i] - dst[i]) > tolerance)
                    return false;
            }
            return true;
        };

        reuseMixedDesignAcrossChannels = true;
        for (int ch = 1; ch < numChannels; ++ch)
        {
            if (!channelsAlmostEqual(linearIR, 0, ch, 1.0e-12)
                || !channelsAlmostEqual(minimumIR, 0, ch, 1.0e-12))
            {
                reuseMixedDesignAcrossChannels = false;
                break;
            }
        }

        if (reuseMixedDesignAcrossChannels)
            juce::Logger::writeToLog("MixedPhase: channel optimization enabled (reuse ch0 result for identical channels)");
    }

    try
    {
        for (int ch = 0; ch < numChannels; ++ch)
        {
            if (reuseMixedDesignAcrossChannels && ch > 0)
            {
                mixedIR.copyFrom(ch, 0, mixedIR, 0, 0, numSamples);
                continue;
            }

            if (checkCancellation(shouldExit, wasCancelled))
                return {};

            const double* srcLinear = linearIR.getReadPointer(ch);
            const double* srcMinimum = minimumIR.getReadPointer(ch);

            int peakDelay = 0;
            double maxPeakVal = 0.0;
            for (int i = 0; i < numSamples; ++i)
            {
                const double val = std::abs(srcLinear[i]);
                if (val > maxPeakVal)
                {
                    maxPeakVal = val;
                    peakDelay = i;
                }
            }

            juce::Logger::writeToLog("Linear IR peak delay: "
                                     + juce::String(peakDelay)
                                     + " samples");

            std::memset(linearSpec.get(), 0, static_cast<size_t>(fftSize) * sizeof(MKL_Complex16));
            std::memset(minimumSpec.get(), 0, static_cast<size_t>(fftSize) * sizeof(MKL_Complex16));

            for (int i = 0; i < numSamples; ++i)
            {
                linearSpec.get()[i].real = srcLinear[i];
                minimumSpec.get()[i].real = srcMinimum[i];
            }

            if (DftiComputeForward(dfti.handle, linearSpec.get()) != DFTI_NO_ERROR) return {};
            if (DftiComputeForward(dfti.handle, minimumSpec.get()) != DFTI_NO_ERROR) return {};

            std::vector<double> phiMinUnwrapped(static_cast<size_t>(complexSize));
            phiMinUnwrapped[0] = std::atan2(minimumSpec.get()[0].imag, minimumSpec.get()[0].real);
            for (int k = 1; k < complexSize; ++k)
            {
                const double raw = std::atan2(minimumSpec.get()[k].imag, minimumSpec.get()[k].real);
                const double delta = raw - phiMinUnwrapped[static_cast<size_t>(k - 1)];
                if (delta > juce::MathConstants<double>::pi)
                    phiMinUnwrapped[static_cast<size_t>(k)] = phiMinUnwrapped[static_cast<size_t>(k - 1)]
                                                           + delta - 2.0 * juce::MathConstants<double>::pi;
                else
                    phiMinUnwrapped[static_cast<size_t>(k)] = phiMinUnwrapped[static_cast<size_t>(k - 1)] + delta;
            }

            std::vector<bool> phaseValid(static_cast<size_t>(complexSize), true);
            for (int k = 0; k < complexSize; ++k)
            {
                const double freq = (static_cast<double>(k) * sampleRate) / static_cast<double>(fftSize);

                double wLinear = 1.0;
                if (freq >= transitionHiHz)
                    wLinear = 0.0;
                else if (freq > transitionLoHz)
                {
                    const double x = (freq - transitionLoHz) * invSpan;
                    wLinear = 0.5 * (1.0 + std::cos(juce::MathConstants<double>::pi * x));
                }
                const double wMinimum = 1.0 - wLinear;

                const double omega = 2.0 * juce::MathConstants<double>::pi * k / fftSize;
                const double phi_lin = -omega * peakDelay;
                const double phi_min = phiMinUnwrapped[static_cast<size_t>(k)];

                const double phiTarget = wLinear * phi_lin + wMinimum * phi_min;
                const double magnitude = std::hypot(linearSpec.get()[k].real, linearSpec.get()[k].imag);

                if (magnitude < 1.0e-10)
                {
                    phaseValid[static_cast<size_t>(k)] = false;
                    targetPhase.get()[k] = (k > 0) ? targetPhase.get()[k - 1] : 0.0;
                }
                else
                {
                    targetPhase.get()[k] = phiTarget;
                }
            }

            const double dOmega = 2.0 * juce::MathConstants<double>::pi / fftSize;
            constexpr double maxAllowedGD = 120.0;
            const double maxPhaseSlope = maxAllowedGD * dOmega;

            for (int k = 1; k < complexSize; ++k)
            {
                if (!std::isfinite(targetPhase.get()[k]))
                {
                    targetPhase.get()[k] = targetPhase.get()[k - 1];
                    phaseValid[static_cast<size_t>(k)] = false;
                    continue;
                }

                const double delta = targetPhase.get()[k] - targetPhase.get()[k - 1];
                if (std::abs(delta) > maxPhaseSlope)
                {
                    targetPhase.get()[k] = targetPhase.get()[k - 1];
                    phaseValid[static_cast<size_t>(k)] = false;
                }
            }

            std::vector<double, convo::MKLAllocator<double>> targetGroupDelay(complexSize, 0.0);

            for (int k = 0; k < complexSize; ++k)
            {
                double dPhi = 0.0;
                if (k == 0)
                    dPhi = (targetPhase.get()[1] - targetPhase.get()[0]) / dOmega;
                else if (k == complexSize - 1)
                    dPhi = (targetPhase.get()[k] - targetPhase.get()[k - 1]) / dOmega;
                else
                    dPhi = (targetPhase.get()[k + 1] - targetPhase.get()[k - 1]) / (2.0 * dOmega);

                targetGroupDelay[k] = -dPhi;
                targetGroupDelay[k] -= static_cast<double>(peakDelay);
            }

            {
                constexpr int smoothWindow = 5;
                std::vector<double, convo::MKLAllocator<double>> movingAvg(complexSize, 0.0);
                for (int k = 0; k < complexSize; ++k)
                {
                    const int start = std::max(0, k - smoothWindow);
                    const int end = std::min(complexSize - 1, k + smoothWindow);

                    double sum = 0.0;
                    for (int j = start; j <= end; ++j)
                        sum += targetGroupDelay[static_cast<size_t>(j)];

                    movingAvg[static_cast<size_t>(k)] = sum / static_cast<double>(end - start + 1);
                }
                targetGroupDelay.swap(movingAvg);
            }

            {
                const double minGD = *std::min_element(targetGroupDelay.begin(), targetGroupDelay.end());
                const double maxGD = *std::max_element(targetGroupDelay.begin(), targetGroupDelay.end());
                juce::Logger::writeToLog("Target group delay range: "
                                         + juce::String(minGD)
                                         + " to "
                                         + juce::String(maxGD)
                                         + " samples");
            }

            {
                const double minGD = *std::min_element(targetGroupDelay.begin(), targetGroupDelay.end());
                if (minGD < 0.0)
                {
                    const double offset = -minGD + 5.0;
                    for (auto& gd : targetGroupDelay)
                        gd += offset;
                }
            }

            if (!targetGroupDelay.empty())
            {
                std::vector<double, convo::MKLAllocator<double>> smoothed(targetGroupDelay.size(), 0.0);
                constexpr double alpha = 0.45;
                smoothed[0] = targetGroupDelay[0];
                for (size_t i = 1; i < targetGroupDelay.size(); ++i)
                    smoothed[i] = alpha * targetGroupDelay[i] + (1.0 - alpha) * smoothed[i - 1];

                targetGroupDelay.swap(smoothed);
            }

            for (int k = 1; k < complexSize; ++k)
            {
                if (!std::isfinite(targetGroupDelay[static_cast<size_t>(k)])
                    || std::abs(targetGroupDelay[static_cast<size_t>(k)]) > maxAllowedGD * 2.0)
                {
                    targetGroupDelay[static_cast<size_t>(k)] = targetGroupDelay[static_cast<size_t>(k - 1)];
                }
            }

            {
                for (auto& gd : targetGroupDelay)
                    gd = std::clamp(gd, 0.0, maxAllowedGD);
            }

            {
                double minGDPostFix = std::numeric_limits<double>::max();
                double maxGDPostFix = std::numeric_limits<double>::lowest();
                for (const auto gd : targetGroupDelay)
                {
                    if (gd < minGDPostFix) minGDPostFix = gd;
                    if (gd > maxGDPostFix) maxGDPostFix = gd;
                }
                juce::Logger::writeToLog("GD range (post-fix): "
                                         + juce::String(minGDPostFix)
                                         + " to "
                                         + juce::String(maxGDPostFix)
                                         + " samples");
            }

            {
                const double minGDc = *std::min_element(targetGroupDelay.begin(), targetGroupDelay.end());
                const double maxGDc = *std::max_element(targetGroupDelay.begin(), targetGroupDelay.end());
                juce::Logger::writeToLog("Target GD after clamping: "
                                         + juce::String(minGDc)
                                         + " to "
                                         + juce::String(maxGDc)
                                         + " samples");
            }

            const bool liveReconfigure = (owner != nullptr) && convo::consumeAtomic(owner->isPrepared, std::memory_order_acquire);
            const bool highRateLive = liveReconfigure && sampleRate >= 96000.0;

            const int optimFreqPoints = liveReconfigure ? (highRateLive ? 12 : 64) : 256;

            std::vector<double> targetGroupDelayStd(targetGroupDelay.begin(), targetGroupDelay.end());

            std::vector<double> optim_freq_hz(optimFreqPoints);
            std::vector<double> optim_target_gd(optimFreqPoints);
            {
                const double logMin = std::log(20.0);
                const double logMax = std::log(sampleRate / 2.0);
                for (int i = 0; i < optimFreqPoints; ++i)
                {
                    const double f = std::exp(logMin + (logMax - logMin) * i / (optimFreqPoints - 1));
                    optim_freq_hz[i] = f;
                    const double kReal = f * static_cast<double>(fftSize) / sampleRate;
                    const int k0 = std::clamp(static_cast<int>(kReal), 0, complexSize - 1);
                    const int k1 = std::min(k0 + 1, complexSize - 1);
                    const double t = kReal - std::floor(kReal);
                    optim_target_gd[i] = (1.0 - t) * targetGroupDelayStd[k0] + t * targetGroupDelayStd[k1];
                }
            }

            convo::AllpassDesigner::Config designer_config;
            designer_config.numSections = liveReconfigure ? (highRateLive ? 2 : 8) : 20;
            designer_config.method = convo::OptimizationMethod::CMAES;
            designer_config.freqPoints = optimFreqPoints;
            designer_config.minFreqHz = 20.0;
            designer_config.maxFreqHz = sampleRate / 2.0;
            designer_config.cmaesMaxGenerations = liveReconfigure ? (highRateLive ? 3 : 12) : 160;
            designer_config.cmaesPopulationSize = liveReconfigure ? (highRateLive ? 6 : 12) : 64;
            designer_config.cmaesInitialSigma = 1.0;
            designer_config.cmaesParams.sigmaMin = 0.002;
            designer_config.cmaesParams.sigmaMax = 2.0;
#if defined(JUCE_DEBUG)
            designer_config.cmaesSeed = 0xDEADBEEFCAFEBABEULL;
#endif
            designer_config.progressCallback = progressCallback;

            const bool preferGreedyForLive = liveReconfigure && highRateLive;
            if (preferGreedyForLive)
            {
                designer_config.method = convo::OptimizationMethod::GreedyAdaGrad;
                designer_config.maxIterations = 4;
                designer_config.learningRate = 0.006;
            }

            std::vector<convo::SecondOrderAllpass> allpass_sections;
            convo::AllpassDesigner designer;

            if (progressCallback) progressCallback(0.1f);
            bool designSuccess = false;
            if (designer_config.method == convo::OptimizationMethod::GreedyAdaGrad)
            {
                juce::Logger::writeToLog("MixedPhase: starting GreedyAdaGrad with "
                                         + juce::String(designer_config.freqPoints)
                                         + " freq points, maxIter="
                                         + juce::String(designer_config.maxIterations));

                designSuccess = designer.design(sampleRate, optim_freq_hz, optim_target_gd,
                                                designer_config, allpass_sections, shouldExit);
                juce::Logger::writeToLog("MixedPhase: design result = " + juce::String(designSuccess ? 0 : 1));
            }
            else
            {
                juce::Logger::writeToLog("MixedPhase: starting design with "
                                         + juce::String(designer_config.freqPoints)
                                         + " freq points, maxGen="
                                         + juce::String(designer_config.cmaesMaxGenerations));

                const auto designResult = designer.designWithCMAES(sampleRate, optim_freq_hz, optim_target_gd,
                                            designer_config, allpass_sections, shouldExit);
                juce::Logger::writeToLog("MixedPhase: design result = " + juce::String(static_cast<int>(designResult)));

                designSuccess = (designResult == convo::DesignResult::Success);

                if (!designSuccess && !(shouldExit && shouldExit()))
                {
                    designer_config.method = convo::OptimizationMethod::GreedyAdaGrad;
                    designer_config.maxIterations = 50;
                    designer_config.learningRate = 0.01;
                    designSuccess = designer.design(sampleRate, optim_freq_hz, optim_target_gd,
                                                    designer_config, allpass_sections, shouldExit);
                }
            }

            if (progressCallback) progressCallback(0.9f);

            if (!designSuccess)
            {
                setMixedPhaseState(0);
                juce::Logger::writeToLog("[MixedPhase] State -> WaitingIR (design failed)");
                if (progressCallback) progressCallback(1.0f);
                return {};
            }

            std::vector<double> freq_hz(complexSize);
            for (int k = 0; k < complexSize; ++k)
                freq_hz[k] = (static_cast<double>(k) * sampleRate) / static_cast<double>(fftSize);

            auto allpass_response = convo::AllpassDesigner::computeResponse(allpass_sections, sampleRate, freq_hz);

            for (int k = 0; k < fftSize; ++k)
            {
                const int mirroredBin = (k <= half) ? k : (fftSize - k);
                std::complex<double> ap = allpass_response[mirroredBin];
                if (k > half) ap = std::conj(ap);

                std::complex<double> h_linear(linearSpec.get()[k].real, linearSpec.get()[k].imag);
                std::complex<double> h_mixed = h_linear * ap;
                linearSpec.get()[k].real = h_mixed.real();
                linearSpec.get()[k].imag = h_mixed.imag();
            }

            if (DftiComputeBackward(dfti.handle, linearSpec.get()) != DFTI_NO_ERROR)
                return {};

            double* mixedTime = mixedIR.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
            {
                const double value = linearSpec.get()[i].real;
                mixedTime[i] = (std::abs(value) < 1.0e-18) ? 0.0 : value;
            }
        }

        for (int ch = 0; ch < numChannels; ++ch)
        {
            double rmsLinear = 0.0;
            double rmsMixed = 0.0;
            const double* srcL = linearIR.getReadPointer(ch);
            const double* srcM = mixedIR.getReadPointer(ch);
            for (int i = 0; i < numSamples; ++i)
            {
                rmsLinear += srcL[i] * srcL[i];
                rmsMixed += srcM[i] * srcM[i];
            }
            rmsLinear = std::sqrt(rmsLinear / numSamples);
            rmsMixed = std::sqrt(rmsMixed / numSamples);
            if (rmsMixed > 1e-12 && rmsLinear > 1e-12)
            {
                const double gain = rmsLinear / rmsMixed;
                double* dst = mixedIR.getWritePointer(ch);
                for (int i = 0; i < numSamples; ++i)
                    dst[i] *= gain;
            }
        }

        for (int ch = 0; ch < numChannels; ++ch)
        {
            const double* p = mixedIR.getReadPointer(ch);
            for (int i = 0; i < numSamples; ++i)
            {
                if (!std::isfinite(p[i]))
                {
                    juce::Logger::writeToLog("convertToMixedPhaseAllpass: Safety guard triggered (NaN/Inf detected), returning empty.");
                    return {};
                }
            }
        }

        double peak = 0.0;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            const double* p = mixedIR.getReadPointer(ch);
            for (int i = 0; i < numSamples; ++i)
                peak = std::max(peak, std::abs(p[i]));
        }

#if defined(JUCE_DEBUG)
        constexpr double PEAK_LIMIT = 8.0;
        constexpr double CREST_LIMIT = 200.0;
#else
        constexpr double PEAK_LIMIT = 4.0;
        constexpr double CREST_LIMIT = 50.0;
#endif
        if (peak > PEAK_LIMIT)
        {
            juce::Logger::writeToLog("convertToMixedPhaseAllpass: Excessive peak after RMS normalization (peak=" + juce::String(peak) + "), falling back to Phase 1.");
            return {};
        }

        {
            double sumSq = 0.0;
            for (int ch = 0; ch < numChannels; ++ch)
            {
                const double* p = mixedIR.getReadPointer(ch);
                for (int i = 0; i < numSamples; ++i)
                    sumSq += p[i] * p[i];
            }
            const double rms = std::sqrt(sumSq / static_cast<double>(numChannels * numSamples));
            if (rms < 1.0e-6)
            {
                juce::Logger::writeToLog("convertToMixedPhaseAllpass: RMS too low (" + juce::String(rms) + "), falling back.");
                return {};
            }

            const double crest = peak / rms;
            if (crest > CREST_LIMIT && rms < 1.0e-4)
            {
                juce::Logger::writeToLog("convertToMixedPhaseAllpass: Crest factor too high (" + juce::String(crest)
                                         + ") with low RMS (" + juce::String(rms) + "), falling back.");
                return {};
            }
        }

        if (peak > 0.99)
        {
            const double gain = 0.98 / peak;
            mixedIR.applyGain(gain);
        }

        if (owner && fileHash != 0) {
            ConvolverProcessor::IRCacheKey key;
            key.fileHash = fileHash;
            key.sampleRate = sampleRate;
            key.phaseMode = ConvolverProcessor::PhaseMode::Mixed;
            key.f1 = static_cast<float>(transitionLoHz);
            key.f2 = static_cast<float>(transitionHiHz);
            key.tau = static_cast<float>(tau);
            key.targetLength = linearIR.getNumSamples();

            const juce::ScopedLock sl(owner->cacheMutex);
            ConvolverProcessor::CacheEntry entry;
            entry.ir = std::make_unique<juce::AudioBuffer<double>>(mixedIR);
            entry.lastUsedTime = juce::Time::getMillisecondCounter();
            owner->irCache[key] = std::move(entry);
            owner->evictOldestCacheEntry();
        }

        if (progressCallback) progressCallback(1.0f);
        setMixedPhaseState(2);
        juce::Logger::writeToLog("[MixedPhase] State -> Completed");
        return mixedIR;
    }
    catch (...)
    {
        setMixedPhaseState(0);
        juce::Logger::writeToLog("[MixedPhase] State -> WaitingIR (exception)");
        throw;
    }
}

juce::AudioBuffer<double> ConvolverProcessor::convertToMixedPhaseFallback(const juce::AudioBuffer<double>& linearIR,
                                                             const juce::AudioBuffer<double>& minimumIR,
                                                             double sampleRate,
                                                             double transitionLoHz,
                                                             double transitionHiHz,
                                                             double tau,
                                                             const std::function<bool()>& shouldExit,
                                                             bool* wasCancelled)
{
    if (wasCancelled) *wasCancelled = false;
    (void)tau;

#if defined(__AVX2__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

    const int numSamples = linearIR.getNumSamples();
    const int numChannels = linearIR.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0)
        return {};

    if (minimumIR.getNumSamples() != numSamples || minimumIR.getNumChannels() != numChannels || sampleRate <= 0.0)
        return {};

    if (transitionHiHz <= transitionLoHz)
        return {};

    const int fftSize = juce::nextPowerOfTwo(numSamples);
    static constexpr int MAX_MIXED_FFT_SIZE = 8388608;
    if (fftSize > MAX_MIXED_FFT_SIZE)
    {
        juce::Logger::writeToLog("convertToMixedPhase: fftSize (" + juce::String(fftSize) + ") exceeds limit.");
        return {};
    }

    juce::AudioBuffer<double> mixedIR(numChannels, numSamples);

    convo::ScopedDftiDescriptor dfti;
    const MKL_LONG len = static_cast<MKL_LONG>(fftSize);
    if (DftiCreateDescriptor(dfti.put(), DFTI_DOUBLE, DFTI_COMPLEX, 1, len) != DFTI_NO_ERROR)
        return {};
    if (DftiSetValue(dfti.handle, DFTI_PLACEMENT, DFTI_INPLACE) != DFTI_NO_ERROR)
        return {};
    if (DftiSetValue(dfti.handle, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(fftSize)) != DFTI_NO_ERROR)
        return {};
    if (DftiCommitDescriptor(dfti.handle) != DFTI_NO_ERROR)
        return {};

    const int half = fftSize / 2;
    const int complexSize = half + 1;

    auto linearSpec = convo::makeAlignedArray<MKL_Complex16>(static_cast<size_t>(fftSize));
    auto minimumSpec = convo::makeAlignedArray<MKL_Complex16>(static_cast<size_t>(fftSize));
    auto deltaPhi = convo::makeAlignedArray<double>(static_cast<size_t>(complexSize));

    if (!linearSpec || !minimumSpec || !deltaPhi)
        return {};

    const double invSpan = 1.0 / (transitionHiHz - transitionLoHz);

    for (int ch = 0; ch < numChannels; ++ch)
    {
        if (checkCancellation(shouldExit, wasCancelled))
            return {};

        const double* srcLinear = linearIR.getReadPointer(ch);
        const double* srcMinimum = minimumIR.getReadPointer(ch);

        int peakDelay = 0;
        double maxVal = 0.0;
        for (int i = 0; i < numSamples; ++i)
        {
            double val = std::abs(srcLinear[i]);
            if (val > maxVal)
            {
                maxVal = val;
                peakDelay = i;
            }
        }

        std::memset(linearSpec.get(), 0, static_cast<size_t>(fftSize) * sizeof(MKL_Complex16));
        std::memset(minimumSpec.get(), 0, static_cast<size_t>(fftSize) * sizeof(MKL_Complex16));

        for (int i = 0; i < numSamples; ++i)
        {
            linearSpec.get()[i].real = srcLinear[i];
            minimumSpec.get()[i].real = srcMinimum[i];
        }

        if (DftiComputeForward(dfti.handle, linearSpec.get()) != DFTI_NO_ERROR) return {};
        if (DftiComputeForward(dfti.handle, minimumSpec.get()) != DFTI_NO_ERROR) return {};

        for (int k = 0; k < complexSize; ++k)
        {
            const double freq = (static_cast<double>(k) * sampleRate) / static_cast<double>(fftSize);

            double wLinear = 1.0;
            if (freq >= transitionHiHz)
                wLinear = 0.0;
            else if (freq > transitionLoHz)
            {
                const double x = (freq - transitionLoHz) * invSpan;
                wLinear = 0.5 * (1.0 + std::cos(juce::MathConstants<double>::pi * x));
            }
            const double wMinimum = 1.0 - wLinear;

            const double omega = 2.0 * juce::MathConstants<double>::pi * k / fftSize;
            const double phi_lin = -omega * peakDelay;

            const double phi_min = std::atan2(minimumSpec.get()[k].imag, minimumSpec.get()[k].real);

            const double phi_target = wLinear * phi_lin + wMinimum * phi_min;
            deltaPhi.get()[k] = phi_target - phi_lin;
        }

        unwrapPhaseRadians(deltaPhi.get(), complexSize);

        for (int k = 0; k < fftSize; ++k)
        {
            const int mirroredBin = (k <= half) ? k : (fftSize - k);
            const double dPhi = (k <= half) ? deltaPhi.get()[k] : -deltaPhi.get()[mirroredBin];

            const double re = linearSpec.get()[k].real;
            const double im = linearSpec.get()[k].imag;

            const double cosD = std::cos(dPhi);
            const double sinD = std::sin(dPhi);

            linearSpec.get()[k].real = re * cosD - im * sinD;
            linearSpec.get()[k].imag = re * sinD + im * cosD;
        }

        if (DftiComputeBackward(dfti.handle, linearSpec.get()) != DFTI_NO_ERROR)
            return {};

        double* mixedTime = mixedIR.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i)
        {
            const double value = linearSpec.get()[i].real;
            mixedTime[i] = (std::abs(value) < 1.0e-18) ? 0.0 : value;
        }
    }

    return mixedIR;
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_MIXED_PHASE
