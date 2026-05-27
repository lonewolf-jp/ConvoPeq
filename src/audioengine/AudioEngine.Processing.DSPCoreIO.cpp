#include <JuceHeader.h>
#include <immintrin.h>
#include "AudioEngine.h"
#include "InputBitDepthTransform.h"

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

inline void sanitizeFiniteChunk(double* data, int count) noexcept
{
    if (data == nullptr || count <= 0)
        return;

    for (int i = 0; i < count; ++i)
    {
        if (!isFiniteAndAbsBelowNoLibm(data[i], 1.0e300))
            data[i] = 0.0;
    }
}

inline double fastTanh(double x) noexcept
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

inline double musicalSoftClipScalar(double x, double threshold, double knee, double asymmetry) noexcept
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

inline void pushAdaptiveCaptureBlocks(LockFreeRingBuffer<AudioBlock, 4096>* captureQueue,
                                      const double* left,
                                      const double* right,
                                      int numSamples,
                                      int sampleRateHz,
                                      int bitDepth,
                                      int adaptiveCoeffBankIndex,
                                      uint64_t captureSessionId) noexcept
{
    if (captureQueue == nullptr || left == nullptr || numSamples <= 0)
        return;

    static constexpr int kBlockSize = 256;
    for (int offset = 0; offset < numSamples; offset += kBlockSize)
    {
        const int currentBlockSize = std::min(kBlockSize, numSamples - offset);
        const double* srcL = left + offset;
        const double* srcR = (right != nullptr) ? (right + offset) : srcL;

        if (!captureQueue->pushWithWriter([&](AudioBlock& block) noexcept
        {
            block.numSamples = currentBlockSize;
            block.sampleRateHz = sampleRateHz;
            block.bitDepth = bitDepth;
            block.adaptiveCoeffBankIndex = adaptiveCoeffBankIndex;
            block.sessionId = captureSessionId;

            const int simdCount = currentBlockSize & ~3;
            int i = 0;

            for (; i < simdCount; i += 4)
            {
                __m256d v = _mm256_loadu_pd(srcL + i);
                _mm256_storeu_pd(block.L + i, v);
            }
            for (; i < currentBlockSize; ++i)
                block.L[i] = srcL[i];

            i = 0;
            for (; i < simdCount; i += 4)
            {
                __m256d v = _mm256_loadu_pd(srcR + i);
                _mm256_storeu_pd(block.R + i, v);
            }
            for (; i < currentBlockSize; ++i)
                block.R[i] = srcR[i];
        }))
        {
            // Audio Thread では side-channel atomic 書き込みを行わない。
            // キュー満杯時は静かにドロップする（従来挙動維持）。
        }
    }
}
}

float AudioEngine::DSPCore::measureLevel (const juce::dsp::AudioBlock<const double>& block) const noexcept
{
    double maxLevel = 0.0;
    const int numChannels = (int)block.getNumChannels();
    const int numSamples = (int)block.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto range = juce::FloatVectorOperations::findMinAndMax(block.getChannelPointer(ch), numSamples);
        const double level = std::max(absNoLibm(range.getStart()), absNoLibm(range.getEnd()));
        if (level > maxLevel) maxLevel = level;
    }

    return static_cast<float>(maxLevel);
}

void AudioEngine::DSPCore::pushToFifo(const juce::dsp::AudioBlock<const double>& block,
                                      LockFreeAudioRingBuffer& analyzerFifo) const noexcept
{
    analyzerFifo.push(block);
}

float AudioEngine::DSPCore::processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples,
                                          double headroomGain,
                                          bool analyzerInputTap,
                                          LockFreeAudioRingBuffer& analyzerFifo) noexcept
{
    auto* buffer = bufferToFill.buffer;
    const int startSample = bufferToFill.startSample;
    const int effectiveInputChannels = std::min(buffer->getNumChannels(), 2);

    for (int ch = 0; ch < effectiveInputChannels; ++ch)
    {
        const float* src = buffer->getReadPointer(ch, startSample);
        double* dst = (ch == 0) ? alignedL.get() : alignedR.get();
        convo::input_transform::convertFloatToDoubleHighQuality(src, dst, numSamples, 1.0);
    }

    const bool expandMono = (effectiveInputChannels == 1);
    const int clearStartCh = expandMono ? 2 : effectiveInputChannels;

    for (int ch = clearStartCh; ch < 2; ++ch)
    {
        double* dst = (ch == 0) ? alignedL.get() : alignedR.get();
        juce::FloatVectorOperations::clear(dst, numSamples);
    }

    if (expandMono)
        juce::FloatVectorOperations::copy(alignedR.get(), alignedL.get(), numSamples);

    sanitizeFiniteChunk(alignedL.get(), numSamples);
    sanitizeFiniteChunk(alignedR.get(), numSamples);

    double* channels[2] = { alignedL.get(), alignedR.get() };
    juce::dsp::AudioBlock<double> block(channels, 2, numSamples);
    const float inputLevel = measureLevel(block);

    if (analyzerInputTap)
        pushToFifo(block, analyzerFifo);

    if (absDiffNoLibm(headroomGain, 1.0) > 1e-9)
    {
        for (int ch = 0; ch < 2; ++ch)
            scaleBlockFallback(block.getChannelPointer(ch), numSamples, headroomGain);
    }

    double* lPtr = alignedL.get();
    double* rPtr = alignedR.get();
    auto& dc = dcBlockers();
    dc.inputL.process(lPtr, numSamples);
    dc.inputR.process(rPtr, numSamples);

    return inputLevel;
}

float AudioEngine::DSPCore::processInputDouble(const juce::AudioBuffer<double>& buffer, int numSamples,
                                               double headroomGain,
                                               bool analyzerInputTap,
                                               LockFreeAudioRingBuffer& analyzerFifo) noexcept
{
    const int effectiveInputChannels = std::min(buffer.getNumChannels(), 2);

    for (int ch = 0; ch < effectiveInputChannels; ++ch)
    {
        const double* src = buffer.getReadPointer(ch);
        double* dst = (ch == 0) ? alignedL.get() : alignedR.get();
        convo::input_transform::convertDoubleToDoubleHighQuality(src, dst, numSamples, 1.0);
    }

    const bool expandMono = (effectiveInputChannels == 1);
    const int clearStartCh = expandMono ? 2 : effectiveInputChannels;

    for (int ch = clearStartCh; ch < 2; ++ch)
    {
        double* dst = (ch == 0) ? alignedL.get() : alignedR.get();
        juce::FloatVectorOperations::clear(dst, numSamples);
    }

    if (expandMono)
        std::memcpy(alignedR.get(), alignedL.get(), numSamples * sizeof(double));

    sanitizeFiniteChunk(alignedL.get(), numSamples);
    sanitizeFiniteChunk(alignedR.get(), numSamples);

    double* channels[2] = { alignedL.get(), alignedR.get() };
    juce::dsp::AudioBlock<double> block(channels, 2, numSamples);
    const float inputLevel = measureLevel(block);

    if (analyzerInputTap)
        pushToFifo(block, analyzerFifo);

    if (absDiffNoLibm(headroomGain, 1.0) > 1e-9)
    {
        for (int ch = 0; ch < 2; ++ch)
            scaleBlockFallback(block.getChannelPointer(ch), numSamples, headroomGain);
    }

    double* lPtr = alignedL.get();
    double* rPtr = alignedR.get();
    auto& dc = dcBlockers();
    dc.inputL.process(lPtr, numSamples);
    dc.inputR.process(rPtr, numSamples);

    return inputLevel;
}

void AudioEngine::DSPCore::applyFixedLatencyDelay(double* dataL, double* dataR, int numSamples) noexcept
{
    auto& history = histories();
    if (history.fixedLatencySamples <= 0 || history.fixedLatencyBufferSize <= 0 || dataL == nullptr)
        return;

    const int delay = std::min(history.fixedLatencySamples, history.fixedLatencyBufferSize - 1);
    int writePos = history.fixedLatencyWritePos;
    const int bufferSize = history.fixedLatencyBufferSize;
    double* delayL = history.fixedLatencyBufferL.get();
    double* delayR = history.fixedLatencyBufferR.get();

    for (int i = 0; i < numSamples; ++i)
    {
        delayL[writePos] = dataL[i];
        if (dataR != nullptr)
            delayR[writePos] = dataR[i];

        int readPos = writePos - delay;
        while (readPos < 0)
            readPos += bufferSize;

        dataL[i] = delayL[readPos];
        if (dataR != nullptr)
            dataR[i] = delayR[readPos];

        ++writePos;
        if (writePos >= bufferSize)
            writePos = 0;
    }

    history.fixedLatencyWritePos = writePos;
}

double AudioEngine::DSPCore::musicalSoftClip(double x, double threshold, double knee, double asymmetry) noexcept
{
    return musicalSoftClipScalar(x, threshold, knee, asymmetry);
}

void AudioEngine::DSPCore::processOutput(const juce::AudioSourceChannelInfo& bufferToFill,
                                         int numSamples,
                                         const ProcessingState& state) noexcept
{
    auto* buffer = bufferToFill.buffer;
    const int startSample = bufferToFill.startSample;
    constexpr double kOutputHeadroom = 0.8912509381337456;

    const bool applyDither = (ditherBitDepth > 0);
    const int numChannels = std::min(2, buffer->getNumChannels());

    if (numSamples <= 0 || numChannels <= 0)
    {
        if (numSamples > 0)
            bufferToFill.clearActiveBufferRegion();
        return;
    }

    double* dataL = (numChannels > 0) ? alignedL.get() : nullptr;
    double* dataR = (numChannels > 1) ? alignedR.get() : nullptr;
    float* dstL = (numChannels > 0) ? buffer->getWritePointer(0, startSample) : nullptr;
    float* dstR = (numChannels > 1) ? buffer->getWritePointer(1, startSample) : nullptr;

    if (dataL == nullptr || dstL == nullptr)
    {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    auto& dc = dcBlockers();
    dc.outputL.process(dataL, numSamples);
    if (dataR) dc.outputR.process(dataR, numSamples);

    {
        const __m256d vInf = _mm256_set1_pd(1.0e300);
        int i = 0;
        const int vEnd = numSamples / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vL = _mm256_loadu_pd(dataL + i);
            __m256d nanMaskL = _mm256_cmp_pd(vL, vL, _CMP_ORD_Q);
            __m256d infMaskL = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vL), vInf, _CMP_LT_OQ);
            __m256d maskL = _mm256_and_pd(nanMaskL, infMaskL);
            vL = _mm256_and_pd(vL, maskL);
            _mm256_storeu_pd(dataL + i, vL);

            if (dataR)
            {
                __m256d vR = _mm256_loadu_pd(dataR + i);
                __m256d nanMaskR = _mm256_cmp_pd(vR, vR, _CMP_ORD_Q);
                __m256d infMaskR = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vR), vInf, _CMP_LT_OQ);
                __m256d maskR = _mm256_and_pd(nanMaskR, infMaskR);
                vR = _mm256_and_pd(vR, maskR);
                _mm256_storeu_pd(dataR + i, vR);
            }
        }
        for (; i < numSamples; ++i)
        {
            double v = dataL[i];
            if (!isFiniteAndAbsBelowNoLibm(v, 1.0e300)) v = 0.0;
            dataL[i] = v;
            if (dataR)
            {
                v = dataR[i];
                if (!isFiniteAndAbsBelowNoLibm(v, 1.0e300)) v = 0.0;
                dataR[i] = v;
            }
        }
    }

    pushAdaptiveCaptureBlocks(state.adaptiveCaptureQueue,
                              dataL,
                              dataR,
                              numSamples,
                              state.adaptiveCaptureSampleRateHz,
                              state.adaptiveCaptureBitDepth,
                              state.adaptiveCoeffBankIndex,
                              state.captureSessionId);

    if (noiseShaperType == NoiseShaperType::Adaptive9thOrder
        && state.adaptiveCoeffSet != nullptr
        && (activeAdaptiveCoeffBankIndex != state.adaptiveCoeffBankIndex
            || activeAdaptiveCoeffGeneration != state.adaptiveCoeffGeneration))
    {
        adaptiveNoiseShaper.applyMatchedCoefficients(state.adaptiveCoeffSet->k, kAdaptiveNoiseShaperOrder);
        activeAdaptiveCoeffBankIndex = state.adaptiveCoeffBankIndex;
        activeAdaptiveCoeffGeneration = state.adaptiveCoeffGeneration;
    }

    if (applyDither)
    {
        if (noiseShaperType == NoiseShaperType::Fixed4Tap)
            fixedNoiseShaper.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
        else if (noiseShaperType == NoiseShaperType::Fixed15Tap)
            fixed15TapNoiseShaper.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
        else if (noiseShaperType == NoiseShaperType::Adaptive9thOrder)
            adaptiveNoiseShaper.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
        else
            dither.processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom);
    }
    else
    {
        for (int i = 0; i < numSamples; ++i) dataL[i] *= kOutputHeadroom;
        if (dataR)
            for (int i = 0; i < numSamples; ++i) dataR[i] *= kOutputHeadroom;
    }

    {
        const __m256d vInf = _mm256_set1_pd(1.0e300);
        int i = 0;
        const int vEnd = numSamples / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vL = _mm256_loadu_pd(dataL + i);
            __m256d nanMaskL = _mm256_cmp_pd(vL, vL, _CMP_ORD_Q);
            __m256d infMaskL = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vL), vInf, _CMP_LT_OQ);
            __m256d maskL = _mm256_and_pd(nanMaskL, infMaskL);
            vL = _mm256_and_pd(vL, maskL);
            _mm256_storeu_pd(dataL + i, vL);

            if (dataR)
            {
                __m256d vR = _mm256_loadu_pd(dataR + i);
                __m256d nanMaskR = _mm256_cmp_pd(vR, vR, _CMP_ORD_Q);
                __m256d infMaskR = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vR), vInf, _CMP_LT_OQ);
                __m256d maskR = _mm256_and_pd(nanMaskR, infMaskR);
                vR = _mm256_and_pd(vR, maskR);
                _mm256_storeu_pd(dataR + i, vR);
            }
        }
        for (; i < numSamples; ++i)
        {
            double v = dataL[i];
            if (!isFiniteAndAbsBelowNoLibm(v, 1.0e300)) v = 0.0;
            dataL[i] = v;
            if (dataR)
            {
                v = dataR[i];
                if (!isFiniteAndAbsBelowNoLibm(v, 1.0e300)) v = 0.0;
                dataR[i] = v;
            }
        }
    }

    applyFixedLatencyDelay(dataL, dataR, numSamples);

    for (int i = 0; i < numSamples; ++i)
        dstL[i] = static_cast<float>(juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataL[i]));

    if (dstR)
        for (int i = 0; i < numSamples; ++i)
            dstR[i] = static_cast<float>(juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataR[i]));

    for (int ch = numChannels; ch < buffer->getNumChannels(); ++ch)
        buffer->clear(ch, startSample, numSamples);
}
