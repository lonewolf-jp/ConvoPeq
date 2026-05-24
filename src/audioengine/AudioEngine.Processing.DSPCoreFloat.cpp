#include <JuceHeader.h>
#include <immintrin.h>
#include <optional>
#include "AudioEngine.h"

namespace
{
inline bool isFiniteNoLibm(double x) noexcept
{
    union { double d; uint64_t u; } v { x };
    return ((v.u >> 52) & 0x7FFu) != 0x7FFu;
}

inline bool isFiniteAndAbsBelowNoLibm(double x, double threshold) noexcept
{
    return isFiniteNoLibm(x) && (absNoLibm(x) < threshold);
}

inline double absDiffNoLibm(double a, double b) noexcept
{
    return absNoLibm(a - b);
}

inline void applyGainRamp(float* __restrict data, int numSamples,
                          float startGain, float gainStep) noexcept
{
    int i = 0;
    const int vEnd = numSamples & ~7;
    __m256 gain = _mm256_setr_ps(startGain,
                                 startGain + gainStep,
                                 startGain + gainStep * 2.0f,
                                 startGain + gainStep * 3.0f,
                                 startGain + gainStep * 4.0f,
                                 startGain + gainStep * 5.0f,
                                 startGain + gainStep * 6.0f,
                                 startGain + gainStep * 7.0f);
    const __m256 step = _mm256_set1_ps(gainStep * 8.0f);

    for (; i < vEnd; i += 8)
    {
        __m256 x = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_mul_ps(x, gain));
        gain = _mm256_add_ps(gain, step);
    }

    float g = startGain + gainStep * static_cast<float>(i);
    for (; i < numSamples; ++i, g += gainStep)
        data[i] *= g;
}

inline void scaleBlockFallback(double* data, int numSamples, double gain) noexcept
{
    int i = 0;
    const int vEnd = numSamples / 4 * 4;
    const __m256d vGain = _mm256_set1_pd(gain);
    for (; i < vEnd; i += 4)
    {
        __m256d vData = _mm256_loadu_pd(data + i);
        _mm256_storeu_pd(data + i, _mm256_mul_pd(vData, vGain));
    }
    for (; i < numSamples; ++i)
        data[i] *= gain;
}

static inline double fastTanh(double x) noexcept
{
    constexpr double numA = 10395.0;
    constexpr double numB = 1260.0;
    constexpr double numC = 21.0;
    constexpr double denA = 10395.0;
    constexpr double denB = 4725.0;
    constexpr double denC = 210.0;
    constexpr double clipThreshold = 4.5;

    if (x >= clipThreshold) return 1.0;
    if (x <= -clipThreshold) return -1.0;
    const double x2 = x * x;

    const double num = x * (numA + x2 * (numB + x2 * numC));
    const double den = denA + x2 * (denB + x2 * (denC + x2));
    return num / den;
}

static inline double musicalSoftClipScalar(double x, double threshold, double knee, double asymmetry) noexcept
{
    const double abs_x = absNoLibm(x);
    const double clip_start = threshold - knee;

    if (knee < 1.0e-9)
        return (x > threshold) ? threshold : ((x < -threshold) ? -threshold : x);

    if (abs_x < clip_start)
        return x;

    const double sign = (x > 0.0) ? 1.0 : -1.0;

    double knee_shape = 1.0;
    if (abs_x < threshold + knee)
    {
        const double t = (abs_x - clip_start) / (2.0 * knee);
        knee_shape = t * t * (3.0 - 2.0 * t);
    }

    const double linear = abs_x;
    const double clipped = threshold + knee * fastTanh((abs_x - threshold) / knee);

    const double asymmetric_gain = 1.0 - asymmetry * (1.0 - sign) * 0.5 * knee_shape;
    return sign * (linear * (1.0 - knee_shape) + clipped * knee_shape) * asymmetric_gain;
}

static void softClipBlockAVX2(double* __restrict data, int numSamples,
                              double threshold, double knee,
                              double asymmetry, double& prevSample) noexcept
{
    int i = 0;
    for (; i < numSamples; ++i)
    {
        double x = data[i];

        if (!isFiniteAndAbsBelowNoLibm(x, 1.0e300))
            x = 0.0;

        x = 0.5 * (x + prevSample);
        prevSample = x;
        data[i] = musicalSoftClipScalar(x, threshold, knee, asymmetry);
    }
}
}

void AudioEngine::DSPCore::process(const juce::AudioSourceChannelInfo& bufferToFill,
                                   LockFreeAudioRingBuffer& analyzerFifo,
                                   std::atomic<float>* inputLevelLinear,
                                   std::atomic<float>* outputLevelLinear,
                                   const ProcessingState& state)
{
    const int numSamples = bufferToFill.numSamples;

    if (numSamples > maxSamplesPerBlock)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    if (oversamplingFactor > 1)
    {
        const int expectedUpSize = numSamples * static_cast<int>(oversamplingFactor);
        if (expectedUpSize > maxInternalBlockSize)
        {
            bufferToFill.clearActiveBufferRegion();
            return;
        }
    }

    const bool inputTap = state.analyzerEnabled && (state.analyzerSource == AnalyzerSource::Input);
    const float rawInputLinear = processInput(bufferToFill, numSamples, state.inputHeadroomGain,
                                              inputTap, analyzerFifo);
    if (inputLevelLinear != nullptr)
        convo::publishAtomic(*inputLevelLinear, rawInputLinear, std::memory_order_release);

    double* channels[2] = { alignedL.get(), alignedR.get() };
    juce::dsp::AudioBlock<double> processBlock(channels, 2, numSamples);

    juce::dsp::AudioBlock<double> originalBlock = processBlock;

    if (oversamplingFactor > 1)
    {
        processBlock = oversampling.processUp(originalBlock, static_cast<int>(originalBlock.getNumChannels()));

        if (processBlock.getNumSamples() == 0 || processBlock.getNumSamples() > static_cast<size_t>(maxInternalBlockSize))
        {
            jassertfalse;
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        const int numOSSamples = (int)processBlock.getNumSamples();
        auto& dc = dcBlockers();
        if (processBlock.getNumChannels() > 0)
            dc.oversampledL.process(processBlock.getChannelPointer(0), numOSSamples);
        if (processBlock.getNumChannels() > 1)
            dc.oversampledR.process(processBlock.getChannelPointer(1), numOSSamples);
    }

    int numProcSamples = (int)processBlock.getNumSamples();
    int numProcChannels = (int)processBlock.getNumChannels();

    const convo::EQParameters* eqParamsToUse = state.eqParams;
    const EQCoeffCache* eqCacheToUse = state.eqCache;

    eqRt().setBypassFromRT(state.eqBypassed); // RT-local shadow に書き込み（publishAtomic の RT 使用禁止のため）

    if (state.order == ProcessingOrder::ConvolverThenEQ)
    {
        if (!state.convBypassed)
            convolverRt().process(processBlock);

        if (!state.eqBypassed)
        {
            if (eqParamsToUse != nullptr)
                eqRt().process(processBlock, *eqParamsToUse, eqCacheToUse);
            else
                eqRt().process(processBlock);
        }
        else
        {
            eqRt().process(processBlock);
        }
    }
    else
    {
        if (!state.eqBypassed)
        {
            if (eqParamsToUse != nullptr)
                eqRt().process(processBlock, *eqParamsToUse, eqCacheToUse);
            else
                eqRt().process(processBlock);
        }
        else
        {
            eqRt().process(processBlock);
        }

        if (!state.convBypassed)
        {
            if (absDiffNoLibm(state.convolverInputTrimGain, 1.0) > 1.0e-12)
            {
                for (size_t ch = 0; ch < processBlock.getNumChannels(); ++ch)
                {
                    double* ptr = processBlock.getChannelPointer(ch);
                    scaleBlockFallback(ptr, (int)processBlock.getNumSamples(), state.convolverInputTrimGain);
                }
            }
            convolverRt().process(processBlock);
        }
    }

    {
        const bool convActive = !state.convBypassed;
        const bool eqActive   = !state.eqBypassed;
        if (convActive || eqActive)
        {
            const bool convIsLast = convActive &&
                (!eqActive || state.order == ProcessingOrder::EQThenConvolver);
            if (!convIsLast)
            {
                outputFilter.process(processBlock, false,
                                     state.convHCMode, state.convLCMode, state.eqLPFMode);
            }
        }
    }

    for (size_t ch = 0; ch < processBlock.getNumChannels(); ++ch)
    {
        double* ptr = processBlock.getChannelPointer(ch);
        scaleBlockFallback(ptr, (int)processBlock.getNumSamples(), state.outputMakeupGain);
    }

    if (state.softClipEnabled)
    {
        auto& history = histories();
        const double sat = static_cast<double>(state.saturationAmount);
        const double clipThreshold = 0.95 - 0.45 * sat;
        const double clipKnee = 0.05 + 0.35 * sat;
        const double clipAsymmetry = 0.10 * sat;

        for (int ch = 0; ch < numProcChannels; ++ch)
        {
            double* data = processBlock.getChannelPointer(ch);
            softClipBlockAVX2(data, numProcSamples, clipThreshold, clipKnee, clipAsymmetry,
                              history.softClipPrevSample[ch < 2 ? ch : 1]);
        }
    }

    if (oversamplingFactor > 1)
    {
        oversampling.processDown(processBlock, originalBlock, static_cast<int>(originalBlock.getNumChannels()));
        processBlock = originalBlock;
    }

    if (state.analyzerEnabled && state.analyzerSource == AnalyzerSource::Output)
        pushToFifo(processBlock, analyzerFifo);

    const float outputLinear = measureLevel(originalBlock);
    if (outputLevelLinear != nullptr)
        convo::publishAtomic(*outputLevelLinear, outputLinear, std::memory_order_release);

    processOutput(bufferToFill, numSamples, state);

    auto& ramp = ramps();
    int fadeLeft = ramp.fadeInSamplesLeft;
    if (fadeLeft > 0)
    {
        const int rampThisBlock = std::min(numSamples, fadeLeft);
        const float gainStep = 1.0f / static_cast<float>(FADE_IN_SAMPLES);
        const float startGain = static_cast<float>(FADE_IN_SAMPLES - fadeLeft) * gainStep;
        auto* buffer = bufferToFill.buffer;
        const int startSample = bufferToFill.startSample;
        const int numChannels = buffer->getNumChannels();

        for (int ch = 0; ch < numChannels; ++ch)
            applyGainRamp(buffer->getWritePointer(ch, startSample), rampThisBlock, startGain, gainStep);

        ramp.fadeInSamplesLeft = fadeLeft - rampThisBlock;
    }
}
