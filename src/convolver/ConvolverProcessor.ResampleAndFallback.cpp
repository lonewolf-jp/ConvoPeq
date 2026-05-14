#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "CDSPResampler.h"
#include "AlignedAllocation.h"
#include <mkl.h>
#include <mkl_vml.h>

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RESAMPLE)

// ────────────────────────────────────────────────────────────────
// Resampling and Helper Functions
// ────────────────────────────────────────────────────────────────

namespace ConvolverProcessorInternal
{

ResampleOutput resampleIR(const juce::AudioBuffer<double>& inputIR,
                          double inputSR,
                          double targetSR,
                          r8b::EDSPFilterPhaseResponse phaseMode,
                          const std::function<bool()>& shouldExit)
{
    if (inputSR <= 0.0 || targetSR <= 0.0 || std::abs(inputSR - targetSR) <= 1e-6)
        return { inputIR, ResampleResult::Success };

    const int numCh = inputIR.getNumChannels();
    const int inLen = inputIR.getNumSamples();
    if (numCh <= 0 || inLen <= 0)
        return {{}, ResampleResult::Error};

    double peak = 0.0;
    for (int ch = 0; ch < numCh; ++ch)
    {
        const double* data = inputIR.getReadPointer(ch);
        for (int i = 0; i < inLen; ++i)
            peak = std::max(peak, std::abs(data[i]));
    }
    if (peak < 1e-12)
        return {{}, ResampleResult::SilentIR};

    constexpr double transBand = 2.0;
    constexpr double stopBandAtten = 140.0;

    std::vector<int> lengths(static_cast<size_t>(numCh));
    std::vector<std::vector<double>> chData(static_cast<size_t>(numCh));
    int maxLen = 0;

    for (int ch = 0; ch < numCh; ++ch)
    {
        if (shouldExit && shouldExit())
            return {{}, ResampleResult::Cancelled};

        r8b::CDSPResampler resampler(inputSR, targetSR, inLen,
                                     transBand, stopBandAtten, phaseMode);

        const int maxOut = resampler.getMaxOutLen(inLen);
        if (maxOut <= 0)
            return {{}, ResampleResult::Error};

        auto& buf = chData[static_cast<size_t>(ch)];
        buf.resize(static_cast<size_t>(maxOut), 0.0);

        resampler.oneshot(inputIR.getReadPointer(ch), inLen, buf.data(), maxOut);

        const int effectiveLen = [&buf, maxOut]() {
            constexpr double TAIL_DB = -160.0;
            double peakValue = 0.0;
            for (int i = 0; i < maxOut; ++i)
                peakValue = std::max(peakValue, std::abs(buf[i]));
            if (peakValue < 1e-12)
                return 1;

            const double threshold = peakValue * std::pow(10.0, TAIL_DB / 20.0);
            int consecutive = 0;
            constexpr int MIN_CONSECUTIVE = 8;
            for (int i = maxOut - 1; i >= 0; --i) {
                if (std::abs(buf[i]) > threshold) {
                    consecutive++;
                    if (consecutive >= MIN_CONSECUTIVE)
                        return i + MIN_CONSECUTIVE;
                } else {
                    consecutive = 0;
                }
            }
            return 1;
        }();

        lengths[static_cast<size_t>(ch)] = effectiveLen;
        maxLen = std::max(maxLen, effectiveLen);
    }

    juce::AudioBuffer<double> result(numCh, maxLen);
    result.clear();

    for (int ch = 0; ch < numCh; ++ch)
    {
        const int copy = std::min(lengths[static_cast<size_t>(ch)], maxLen);
        std::memcpy(result.getWritePointer(ch),
                    chData[static_cast<size_t>(ch)].data(),
                    static_cast<size_t>(copy) * sizeof(double));
    }

    if (maxLen < result.getNumSamples())
        result.setSize(result.getNumChannels(), maxLen, true, true, true);

    return { std::move(result), ResampleResult::Success };
}

static double calculate_post_alpha(int n_taps)
{
    if (n_taps <= 0) return 0.05;
    double log2n = std::log2(static_cast<double>(n_taps));
    double alpha = 0.05 + 0.033 * (log2n - 10.0);
    return std::max(0.05, std::min(0.25, alpha));
}

bool applyAsymmetricTukey(double* data, int numSamples)
{
    if (!data || numSamples <= 0) return true;

    auto* start = data;
    auto* end = data + numSamples;
    auto it = std::max_element(start, end, [](double a, double b){
        return std::abs(a) < std::abs(b);
    });
    int peakIndex = static_cast<int>(std::distance(start, it));

    const double alpha_pre = 0.05;
    const double alpha_post = calculate_post_alpha(numSamples);
    const double pi = juce::MathConstants<double>::pi;

    auto window_vals = convo::makeAlignedArray<double>(static_cast<size_t>(numSamples));
    if (!window_vals) return false;

    std::fill_n(window_vals.get(), numSamples, 1.0);

    if (peakIndex > 0)
    {
        const int pre_taper_len = static_cast<int>(std::floor(peakIndex * alpha_pre));
        if (pre_taper_len > 0)
        {
            auto cos_args = convo::makeAlignedArray<double>(static_cast<size_t>(pre_taper_len));
            if (!cos_args) return false;

            const double scale = pi / (peakIndex * alpha_pre);
            const double offset = -pi;
            for (int i = 0; i < pre_taper_len; ++i)
                cos_args.get()[i] = scale * static_cast<double>(i) + offset;

            vdCos(pre_taper_len, cos_args.get(), window_vals.get());

            for (int i = 0; i < pre_taper_len; ++i)
                window_vals.get()[i] = 0.5 * (1.0 + window_vals.get()[i]);
        }
    }

    const double dist_to_end = static_cast<double>(numSamples - 1 - peakIndex);
    if (dist_to_end > 1.0e-9)
    {
        const int post_taper_start_idx = peakIndex + static_cast<int>(std::ceil(dist_to_end * (1.0 - alpha_post)));
        const int post_taper_len = numSamples - post_taper_start_idx;
        if (post_taper_len > 0)
        {
            auto cos_args = convo::makeAlignedArray<double>(static_cast<size_t>(post_taper_len));
            if (!cos_args) return false;
            double* post_window_vals = window_vals.get() + post_taper_start_idx;
            auto post_cos_vals = convo::makeAlignedArray<double>(static_cast<size_t>(post_taper_len));
            if (!post_cos_vals) return false;

            const double scale = (pi / alpha_post) / dist_to_end;
            const double offset = (pi / alpha_post) * (((double)post_taper_start_idx - (double)peakIndex) / dist_to_end - (1.0 - alpha_post));
            for (int i = 0; i < post_taper_len; ++i)
                cos_args.get()[i] = scale * static_cast<double>(i) + offset;

            vdCos(post_taper_len, cos_args.get(), post_cos_vals.get());

            for (int i = 0; i < post_taper_len; ++i)
                post_window_vals[i] = 0.5 * (1.0 + post_cos_vals.get()[i]);
        }
    }

    if ((reinterpret_cast<uintptr_t>(data) & 63u) == 0)
    {
        vdMul(numSamples, data, window_vals.get(), data);
    }
    else
    {
        auto aligned_data = convo::makeAlignedArray<double>(static_cast<size_t>(numSamples));
        if (!aligned_data) return false;
        std::memmove(aligned_data.get(), data, static_cast<size_t>(numSamples) * sizeof(double));
        vdMul(numSamples, aligned_data.get(), window_vals.get(), aligned_data.get());
        std::memmove(data, aligned_data.get(), static_cast<size_t>(numSamples) * sizeof(double));
    }
    return true;
}

int estimateEffectiveIRLengthSamples(const juce::AudioBuffer<double>& irBuffer, double sampleRate)
{
    const int numSamples = irBuffer.getNumSamples();
    const int numChannels = irBuffer.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0 || sampleRate <= 0.0)
        return 0;

    auto envelope = convo::makeAlignedArray<double>(static_cast<size_t>(numSamples));
    if (!envelope)
        return 0;
    std::fill_n(envelope.get(), numSamples, 0.0);
    double peak = 0.0;
    int peakIndex = 0;

    for (int i = 0; i < numSamples; ++i)
    {
        double sampleMax = 0.0;
        for (int ch = 0; ch < numChannels; ++ch)
            sampleMax = (std::max)(sampleMax, std::abs(irBuffer.getSample(ch, i)));

        envelope.get()[i] = sampleMax;
        if (sampleMax > peak)
        {
            peak = sampleMax;
            peakIndex = i;
        }
    }

    if (peak <= 1.0e-12)
        return juce::jmax(1, juce::jmin(numSamples, static_cast<int>(std::round(sampleRate * ConvolverProcessor::IR_LENGTH_MIN_SEC))));

    const int rmsWindow = juce::jmax(1, static_cast<int>(std::round(sampleRate * 0.010)));
    const int sustainSamples = juce::jmax(rmsWindow, static_cast<int>(std::round(sampleRate * 0.050)));
    const int minimumKeepSamples = juce::jmax(0, static_cast<int>(std::round(sampleRate * 0.200)));
    const int scanStart = juce::jmin(numSamples, peakIndex + minimumKeepSamples);
    const int scanLimit = juce::jmax(scanStart, numSamples - rmsWindow);
    const int scanStep = juce::jmax(1, rmsWindow / 8);
    const double thresholdAmp = peak * std::pow(10.0, -50.0 / 20.0);

    auto prefix = convo::makeAlignedArray<double>(static_cast<size_t>(numSamples) + 1u);
    if (!prefix)
        return juce::jmax(1, juce::jmin(numSamples, static_cast<int>(std::round(sampleRate * ConvolverProcessor::IR_LENGTH_MIN_SEC))));
    std::fill_n(prefix.get(), static_cast<size_t>(numSamples) + 1u, 0.0);
    for (int i = 0; i < numSamples; ++i)
        prefix.get()[static_cast<size_t>(i) + 1u] = prefix.get()[static_cast<size_t>(i)] + envelope.get()[i] * envelope.get()[i];

    int belowStart = -1;
    for (int i = scanStart; i <= scanLimit; i += scanStep)
    {
        const int windowEnd = juce::jmin(numSamples, i + rmsWindow);
        const double meanSquare = (prefix.get()[static_cast<size_t>(windowEnd)] - prefix.get()[static_cast<size_t>(i)])
                                / static_cast<double>(windowEnd - i);
        const double rms = std::sqrt((std::max)(0.0, meanSquare));

        if (rms <= thresholdAmp)
        {
            if (belowStart < 0)
                belowStart = i;

            if ((i - belowStart) >= sustainSamples)
                return juce::jlimit(1, numSamples, juce::jmax(peakIndex + minimumKeepSamples, belowStart + rmsWindow));
        }
        else
        {
            belowStart = -1;
        }
    }

    return numSamples;
}

bool loadImpulseResponsePreviewFile(const juce::File& file,
                                    juce::AudioBuffer<double>& loadedIR,
                                    double& loadedSampleRate,
                                    juce::String& errorMessage)
{
    if (!file.existsAsFile())
    {
        errorMessage = "IR file not found: " + file.getFullPathName();
        return false;
    }

    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (!reader)
    {
        errorMessage = "Unsupported audio format or corrupted file: " + file.getFileName();
        return false;
    }

    const int64 fileLength = reader->lengthInSamples;
    const int numChannels = static_cast<int>(reader->numChannels);
    static constexpr int64 maxFileLength = 2147483647;

    if (fileLength > maxFileLength)
    {
        errorMessage = "IR file is too large (exceeds 2GB samples limit).";
        return false;
    }

    if (numChannels <= 0)
    {
        errorMessage = "Invalid channel count in IR file.";
        return false;
    }

    juce::AudioBuffer<float> tempFloatBuffer(numChannels, static_cast<int>(fileLength));
    if (!reader->read(&tempFloatBuffer, 0, static_cast<int>(fileLength), 0, true, true))
    {
        errorMessage = "Failed to read audio data from file.";
        return false;
    }

    auto tempAlignedBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(fileLength));
    if (!tempAlignedBuffer)
    {
        errorMessage = "Failed to allocate temporary buffer for IR loading.";
        return false;
    }

    loadedIR.setSize(numChannels, static_cast<int>(fileLength));
    for (int ch = 0; ch < numChannels; ++ch)
    {
        const float* src = tempFloatBuffer.getReadPointer(ch);
        convo::input_transform::convertFloatToDoubleHighQuality(src, tempAlignedBuffer.get(), static_cast<int>(fileLength));
        loadedIR.copyFrom(ch, 0, tempAlignedBuffer.get(), static_cast<int>(fileLength));
    }

    loadedSampleRate = reader->sampleRate;
    return true;
}

juce::AudioBuffer<double> convertToMinimumPhase(const juce::AudioBuffer<double>& linearIR,
                                                const std::function<bool()>& shouldExit,
                                                bool* wasCancelled)
{
    if (wasCancelled) *wasCancelled = false;

    const int numSamples = linearIR.getNumSamples();
    if (numSamples <= 0 || linearIR.getNumChannels() < 1) return {};
    const int fftSize = juce::nextPowerOfTwo(numSamples * 4);

    static constexpr int MAX_MINPHASE_FFT_SIZE = 8388608;
    if (fftSize > MAX_MINPHASE_FFT_SIZE)
    {
        juce::Logger::writeToLog("convertToMinimumPhase: fftSize (" + juce::String(fftSize) + ") exceeds limit. Skipping min-phase conversion to prevent excessive memory usage.");
        return {};
    }

    juce::AudioBuffer<double> minPhaseIR(linearIR.getNumChannels(), numSamples);

    convo::ScopedDftiDescriptor dfti;

    const MKL_LONG len = static_cast<MKL_LONG>(fftSize);
    if (DftiCreateDescriptor(dfti.put(), DFTI_DOUBLE, DFTI_COMPLEX, 1, len) != DFTI_NO_ERROR)
    {
        juce::Logger::writeToLog("convertToMinimumPhase: DftiCreateDescriptor failed.");
        return {};
    }

    if (DftiSetValue(dfti.handle, DFTI_PLACEMENT, DFTI_INPLACE) != DFTI_NO_ERROR)
    {
        juce::Logger::writeToLog("convertToMinimumPhase: DftiSetValue(DFTI_PLACEMENT) failed.");
        return {};
    }

    if (DftiSetValue(dfti.handle, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(fftSize)) != DFTI_NO_ERROR)
    {
        juce::Logger::writeToLog("convertToMinimumPhase: DftiSetValue(DFTI_BACKWARD_SCALE) failed.");
        return {};
    }

    if (DftiCommitDescriptor(dfti.handle) != DFTI_NO_ERROR)
    {
        juce::Logger::writeToLog("convertToMinimumPhase: DftiCommitDescriptor failed.");
        return {};
    }

    auto spectrum = convo::makeAlignedArray<MKL_Complex16>(static_cast<size_t>(fftSize));
    if (!spectrum)
        return {};

    for (int ch = 0; ch < linearIR.getNumChannels(); ++ch)
    {
        if (checkCancellation(shouldExit, wasCancelled))
            return {};

        const double* src = linearIR.getReadPointer(ch);
        for (int i = 0; i < fftSize; ++i)
        {
            spectrum.get()[i].real = (i < numSamples) ? src[i] : 0.0;
            spectrum.get()[i].imag = 0.0;
        }

        if (DftiComputeForward(dfti.handle, spectrum.get()) != DFTI_NO_ERROR) {
            juce::Logger::writeToLog("convertToMinimumPhase: DftiComputeForward (1) failed.");
            return {};
        }

        {
            auto mag = convo::makeAlignedArray<double>(static_cast<size_t>(fftSize));

            vzAbs(fftSize, spectrum.get(), mag.get());

            for (int i = 0; i < fftSize; ++i)
                mag[i] = std::max(mag[i], 1.0e-300);

            vdLn(fftSize, mag.get(), mag.get());

            for (int i = 0; i < fftSize; ++i)
                { spectrum.get()[i].real = mag[i]; spectrum.get()[i].imag = 0.0; }
        }

        if (DftiComputeBackward(dfti.handle, spectrum.get()) != DFTI_NO_ERROR) {
            juce::Logger::writeToLog("convertToMinimumPhase: DftiComputeBackward (1) failed.");
            return {};
        }

        const int half = fftSize / 2;
        spectrum.get()[0].imag = 0.0;
        for (int i = 1; i < half; ++i)
        {
            spectrum.get()[i].real *= 2.0;
            spectrum.get()[i].imag = 0.0;
        }
        spectrum.get()[half].imag = 0.0;
        for (int i = half + 1; i < fftSize; ++i)
        {
            spectrum.get()[i].real = 0.0;
            spectrum.get()[i].imag = 0.0;
        }

        if (DftiComputeForward(dfti.handle, spectrum.get()) != DFTI_NO_ERROR) {
            juce::Logger::writeToLog("convertToMinimumPhase: DftiComputeForward (2) failed.");
            return {};
        }

        {
            for (int i = 0; i < fftSize; ++i)
            {
                spectrum.get()[i].real = juce::jlimit(-50.0, 50.0, spectrum.get()[i].real);
                spectrum.get()[i].imag = juce::jlimit(-50.0, 50.0, spectrum.get()[i].imag);
            }

            vzExp(fftSize, spectrum.get(), spectrum.get());

            for (int i = 0; i < fftSize; ++i)
                if (!std::isfinite(spectrum.get()[i].real) || !std::isfinite(spectrum.get()[i].imag)) return {};
        }

        if (DftiComputeBackward(dfti.handle, spectrum.get()) != DFTI_NO_ERROR) {
            juce::Logger::writeToLog("convertToMinimumPhase: DftiComputeBackward (2) failed.");
            return {};
        }

        double* dst = minPhaseIR.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i)
        {
            double v = spectrum.get()[i].real;
            if (!std::isfinite(v))
                return {};
            if (std::abs(v) < 1.0e-18)
                v = 0.0;
            dst[i] = v;
        }
    }

    return minPhaseIR;
}

} // namespace ConvolverProcessorInternal

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RESAMPLE
