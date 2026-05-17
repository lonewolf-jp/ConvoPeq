#include <JuceHeader.h>
#include <immintrin.h>
#include "AudioEngine.h"

namespace
{
namespace TanhApprox
{
    constexpr double NUM_A = 10395.0;
    constexpr double NUM_B = 1260.0;
    constexpr double NUM_C = 21.0;
    constexpr double DEN_A = 10395.0;
    constexpr double DEN_B = 4725.0;
    constexpr double DEN_C = 210.0;
    constexpr double CLIP_THRESHOLD = 4.5;
}

inline bool isFiniteNoLibm(double x) noexcept
{
    union { double d; uint64_t u; } v { x };
    return ((v.u >> 52) & 0x7FFu) != 0x7FFu;
}

inline bool isFiniteAndAbsBelowNoLibm(double x, double threshold) noexcept
{
    return isFiniteNoLibm(x) && (absNoLibm(x) < threshold);
}

inline void applyGainRamp(double* __restrict data, int numSamples,
                          double startGain, double increment) noexcept
{
    __m256d vGain = _mm256_set_pd(startGain + 3.0 * increment,
                                   startGain + 2.0 * increment,
                                   startGain + increment,
                                   startGain);
    const __m256d vInc4 = _mm256_set1_pd(4.0 * increment);

    int i = 0;
    const int vEnd = numSamples / 4 * 4;
    for (; i < vEnd; i += 4)
    {
        __m256d vData = _mm256_loadu_pd(data + i);
        _mm256_storeu_pd(data + i, _mm256_mul_pd(vData, vGain));
        vGain = _mm256_add_pd(vGain, vInc4);
    }

    double gain = startGain + static_cast<double>(i) * increment;
    for (; i < numSamples; ++i) { data[i] *= gain; gain += increment; }
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
    using namespace TanhApprox;

    if (x >= CLIP_THRESHOLD) return 1.0;
    if (x <= -CLIP_THRESHOLD) return -1.0;
    const double x2 = x * x;

    const double num = x * (NUM_A + x2 * (NUM_B + x2 * NUM_C));
    const double den = DEN_A + x2 * (DEN_B + x2 * (DEN_C + x2));
    return num / den;
}

static inline double musicalSoftClipScalar(double x, double threshold, double knee, double asymmetry) noexcept
{
    const double abs_x = absNoLibm(x);
    const double clip_start = threshold - knee;

    if (knee < 1.0e-9) return (x > threshold) ? threshold : ((x < -threshold) ? -threshold : x);

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
                               double threshold, double knee, double asymmetry,
                               double& prevSampleInOut) noexcept
{
    const double clip_start = threshold - knee;
    jassert(knee > 1.0e-9);

    const __m256d vClipStart   = _mm256_set1_pd(clip_start);
    const __m256d vThreshold   = _mm256_set1_pd(threshold);
    const __m256d vKnee        = _mm256_set1_pd(knee);
    const __m256d vAsym        = _mm256_set1_pd(asymmetry);

    const __m256d vRecipKnee   = _mm256_set1_pd(1.0 / knee);
    const __m256d vRecipKnee2  = _mm256_set1_pd(1.0 / (2.0 * knee));

    const __m256d vOne         = _mm256_set1_pd(1.0);
    const __m256d vMinusOne    = _mm256_set1_pd(-1.0);
    const __m256d vTwo         = _mm256_set1_pd(2.0);
    const __m256d vThree       = _mm256_set1_pd(3.0);
    const __m256d vNegThree    = _mm256_set1_pd(-3.0);
    const __m256d vHalf        = _mm256_set1_pd(0.5);

    const __m256d vNumA        = _mm256_set1_pd(TanhApprox::NUM_A);
    const __m256d vNumB        = _mm256_set1_pd(TanhApprox::NUM_B);
    const __m256d vNumC        = _mm256_set1_pd(TanhApprox::NUM_C);
    const __m256d vDenB        = _mm256_set1_pd(TanhApprox::DEN_B);
    const __m256d vDenC        = _mm256_set1_pd(TanhApprox::DEN_C);
    const __m256d vZero        = _mm256_setzero_pd();
    const __m256d vSignMask    = _mm256_set1_pd(-0.0);

    double prevScalar = prevSampleInOut;

    int i = 0;
    const int vEnd = numSamples / 4 * 4;
    for (; i < vEnd; i += 4)
    {
            __m256d x    = _mm256_loadu_pd(data + i);

        {
            const __m128d xLow       = _mm256_castpd256_pd128(x);
            const __m128d xHigh      = _mm256_extractf128_pd(x, 1);
            const __m128d prevLow128 = _mm_unpacklo_pd(_mm_set_sd(prevScalar), xLow);
            const __m128d prevHigh128= _mm_shuffle_pd(xLow, xHigh, 0x1);
            const __m256d prevVec    = _mm256_set_m128d(prevHigh128, prevLow128);

            const __m256d midVec     = _mm256_mul_pd(_mm256_add_pd(prevVec, x), vHalf);
            const __m256d absMidVec  = _mm256_andnot_pd(vSignMask, midVec);

            const __m256d vTiny      = _mm256_set1_pd(1e-15);
            const __m256d needMidClip= _mm256_cmp_pd(absMidVec, vThreshold, _CMP_GT_OQ);
            const __m256d safeAbsMid = _mm256_max_pd(absMidVec, vTiny);
            const __m256d midGainRaw = _mm256_div_pd(vThreshold, safeAbsMid);
            const __m256d midGain    = _mm256_blendv_pd(vOne, midGainRaw, needMidClip);
            x = _mm256_mul_pd(x, midGain);
        }

        __m256d absX = _mm256_andnot_pd(vSignMask, x);

        __m256d needClip = _mm256_cmp_pd(absX, vClipStart, _CMP_GT_OQ);

        __m256d maskSignPos = _mm256_cmp_pd(x, vZero, _CMP_GT_OQ);
        __m256d sign = _mm256_blendv_pd(vMinusOne, vOne, maskSignPos);

        __m256d t = _mm256_mul_pd(_mm256_sub_pd(absX, vClipStart), vRecipKnee2);
        t = _mm256_min_pd(_mm256_max_pd(t, vZero), vOne);
        __m256d t2 = _mm256_mul_pd(t, t);
        __m256d ks = _mm256_mul_pd(t2, _mm256_fnmadd_pd(vTwo, t, vThree));

        __m256d arg = _mm256_mul_pd(_mm256_sub_pd(absX, vThreshold), vRecipKnee);
        __m256d satHi    = _mm256_cmp_pd(arg, vThree,    _CMP_GE_OQ);
        __m256d satLo    = _mm256_cmp_pd(arg, vNegThree, _CMP_LE_OQ);
        __m256d arg2     = _mm256_mul_pd(arg, arg);

        __m256d num      = _mm256_mul_pd(arg,
                            _mm256_fmadd_pd(arg2,
                                _mm256_fmadd_pd(arg2, vNumC, vNumB),
                            vNumA));
        __m256d den      = _mm256_fmadd_pd(arg2,
                            _mm256_fmadd_pd(arg2,
                                _mm256_fmadd_pd(arg2, vDenC, vDenB),
                            vDenC),
                           vNumA);
        __m256d tanhVal  = _mm256_div_pd(num, den);
        tanhVal = _mm256_blendv_pd(tanhVal, vOne,      satHi);
        tanhVal = _mm256_blendv_pd(tanhVal, vMinusOne, satLo);

        __m256d clipped = _mm256_fmadd_pd(vKnee, tanhVal, vThreshold);

        __m256d linear  = absX;
        __m256d mixed   = _mm256_fmadd_pd(_mm256_sub_pd(clipped, linear), ks, linear);

        __m256d factor = _mm256_mul_pd(vAsym, _mm256_sub_pd(vOne, sign));
        factor = _mm256_mul_pd(factor, vHalf);
        factor = _mm256_mul_pd(factor, ks);
        __m256d asymmetric_gain = _mm256_sub_pd(vOne, factor);

        __m256d result = _mm256_mul_pd(sign, _mm256_mul_pd(mixed, asymmetric_gain));

        result = _mm256_blendv_pd(x, result, needClip);
            _mm256_storeu_pd(data + i, result);

        prevScalar = data[i + 3];
    }

    for (; i < numSamples; ++i)
    {
        const double mid    = (prevScalar + data[i]) * 0.5;
        const double absMid = absNoLibm(mid);
        double x = data[i];
        if (absMid > threshold)
            x *= threshold / absMid;

        if (absNoLibm(x) > clip_start)
            x = musicalSoftClipScalar(x, threshold, knee, asymmetry);

        data[i] = x;
        prevScalar = x;
    }

    prevSampleInOut = prevScalar;
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

    static std::atomic<uint64_t> dropCount { 0 };

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
            dropCount.fetch_add(1, std::memory_order_acq_rel); // RT-RESTRICTED: drop diagnostic counter only, no blocking
        }
    }
}
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_TO_BUFFER)

void AudioEngine::DSPCore::processDoubleToBuffer(const juce::AudioBuffer<double>& source,
                                                 juce::AudioBuffer<double>& destination,
                                                 LockFreeAudioRingBuffer& analyzerFifo,
                                                 std::atomic<float>& inputLevelLinear,
                                                 std::atomic<float>& outputLevelLinear,
                                                 const ProcessingState& state)
{
    const int numSamples = source.getNumSamples();
    const int numChannels = std::min(2, source.getNumChannels());

    if (numSamples <= 0 || destination.getNumSamples() < numSamples)
    {
        destination.clear();
        return;
    }

    for (int ch = 0; ch < numChannels; ++ch)
    {
        const double* src = source.getReadPointer(ch, 0);
        double* dst = destination.getWritePointer(ch, 0);
        juce::FloatVectorOperations::copy(dst, src, numSamples);
    }
    for (int ch = numChannels; ch < destination.getNumChannels(); ++ch)
        destination.clear(ch, 0, numSamples);

    processDouble(destination, analyzerFifo, inputLevelLinear, outputLevelLinear, state);
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_DSP_DOUBLE)

void AudioEngine::DSPCore::processDouble(juce::AudioBuffer<double>& buffer,
                                         LockFreeAudioRingBuffer& analyzerFifo,
                                         std::atomic<float>& inputLevelLinear,
                                         std::atomic<float>& outputLevelLinear,
                                         const ProcessingState& state)
{
    const int numSamples = buffer.getNumSamples();

    if (numSamples > maxSamplesPerBlock)
    {
        buffer.clear();
        return;
    }

    if (oversamplingFactor > 1)
    {
        const int expectedUpSize = numSamples * static_cast<int>(oversamplingFactor);
        if (expectedUpSize > maxInternalBlockSize)
        {
            buffer.clear();
            return;
        }
    }

    const bool inputTapD = state.analyzerEnabled && (state.analyzerSource == AnalyzerSource::Input);
    const float rawInputLinearD = processInputDouble(buffer, numSamples, state.inputHeadroomGain,
                                                     inputTapD, analyzerFifo);
    convo::publishAtomic(inputLevelLinear, rawInputLinearD, std::memory_order_release);

    const bool requestedFullBypass = state.eqBypassed && state.convBypassed;
    auto& ramp = ramps();
    if (requestedFullBypass != ramp.bypassedDouble)
    {
        ramp.bypassFadeGainDouble.setTargetValue(requestedFullBypass ? 0.0 : 1.0);
        ramp.bypassedDouble = requestedFullBypass;
    }

    double* channels[2] = { alignedL.get(), alignedR.get() };
    juce::dsp::AudioBlock<double> processBlock(channels, 2, numSamples);
    juce::dsp::AudioBlock<double> originalBlock = processBlock;

    if (dryBypassBufferDoubleL && dryBypassBufferDoubleR && dryBypassCapacityDouble >= numSamples)
    {
        juce::FloatVectorOperations::copy(dryBypassBufferDoubleL.get(), alignedL.get(), numSamples);
        juce::FloatVectorOperations::copy(dryBypassBufferDoubleR.get(), alignedR.get(), numSamples);
    }

    if (oversamplingFactor > 1)
    {
        processBlock = oversampling.processUp(originalBlock, static_cast<int>(originalBlock.getNumChannels()));

        if (processBlock.getNumSamples() == 0 || processBlock.getNumSamples() > static_cast<size_t>(maxInternalBlockSize))
        {
            jassertfalse;
            buffer.clear();
            return;
        }

        const int numOSSamples = static_cast<int>(processBlock.getNumSamples());
        auto& dc = dcBlockers();
        if (processBlock.getNumChannels() > 0)
            dc.oversampledL.process(processBlock.getChannelPointer(0), numOSSamples);
        if (processBlock.getNumChannels() > 1)
            dc.oversampledR.process(processBlock.getChannelPointer(1), numOSSamples);
    }

    const int numProcSamples = static_cast<int>(processBlock.getNumSamples());
    const int numProcChannels = static_cast<int>(processBlock.getNumChannels());

    // RCU lifetime guarantee:
    // - processDouble() 呼び出し元は Audio Thread 側で RCUReaderGuard を保持する。
    // - snap はその guard 期間中 read-only で有効。
    // - eqCacheToUse は snap->eqCoeffHash で引いたキャッシュを参照し、
    //   reclaim は epoch 遅延解放で行われるため同期間は安全に参照可能。
    const convo::GlobalSnapshot* snap = ownerEngine ? ownerEngine->m_coordinator.getCurrent() : nullptr;
    const bool useSnapshotEq = (snap != nullptr);
    const convo::EQParameters* eqParamsToUse = nullptr;
    const EQCoeffCache* eqCacheToUse = nullptr;
    if (useSnapshotEq && ownerEngine != nullptr)
    {
        const uint64_t hash = snap->eqCoeffHash;
        eqParamsToUse = &snap->eqParams;
        eqCacheToUse = ownerEngine->eqCacheManager.get(hash);
        if (hash != convo::consumeAtomic(ownerEngine->debugLastAppliedEqHash, std::memory_order_acquire))
        {
            ownerEngine->debugAppliedEqHashVersion.fetch_add(1u, std::memory_order_acq_rel); // RT-RESTRICTED: debug version counter only, no blocking
        }
    }
    else if (ownerEngine != nullptr)
    {
        uint64_t fallbackHash = 0;
        const convo::EQParameters& fallbackParams = ownerEngine->getLatestEqParamsFallback(fallbackHash);
        eqParamsToUse = &fallbackParams;
        eqCacheToUse = ownerEngine->eqCacheManager.get(fallbackHash);
        if (fallbackHash != 0 && fallbackHash != convo::consumeAtomic(ownerEngine->debugLastAppliedEqHash, std::memory_order_acquire))
        {
            ownerEngine->debugAppliedEqHashVersion.fetch_add(1u, std::memory_order_acq_rel); // RT-RESTRICTED: debug version counter only, no blocking
        }
    }

    eqRt().setBypassFromRT(state.eqBypassed); // RT-local shadow に書き込み（publishAtomic の RT 使用禁止のため）

    if (state.order == ProcessingOrder::ConvolverThenEQ)
    {
        if (!state.convBypassed)
            convolverRt().process(processBlock);
        if (!state.eqBypassed)
        {
            if (eqParamsToUse != nullptr)
            {
                eqRt().process(processBlock, *eqParamsToUse, eqCacheToUse);
            }
            else
            {
                eqRt().process(processBlock);
            }
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
            {
                eqRt().process(processBlock, *eqParamsToUse, eqCacheToUse);
            }
            else
            {
                eqRt().process(processBlock);
            }
        }
        else
        {
            eqRt().process(processBlock);
        }
        if (!state.convBypassed)
        {
            if (state.convolverInputTrimGain != 1.0)
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
        const double clipKnee      = 0.05 + 0.35 * sat;
        const double clipAsymmetry = 0.10 * sat;

        for (int ch = 0; ch < numProcChannels; ++ch)
        {
            double* data = processBlock.getChannelPointer(ch);
            softClipBlockAVX2(data, numProcSamples, clipThreshold, clipKnee, clipAsymmetry,
                               history.softClipPrevSample[ch < 2 ? ch : 1]);
        }
    }

    const bool bypassBlendRequested = ramp.bypassFadeGainDouble.isSmoothing() || requestedFullBypass;
    if (oversamplingFactor == 1
        && dryBypassBufferDoubleL
        && dryBypassBufferDoubleR
        && dryBypassCapacityDouble >= numSamples
        && bypassBlendRequested)
    {
        double* wetL = (numProcChannels > 0) ? processBlock.getChannelPointer(0) : nullptr;
        double* wetR = (numProcChannels > 1) ? processBlock.getChannelPointer(1) : nullptr;
        const double* dryL = dryBypassBufferDoubleL.get();
        const double* dryR = dryBypassBufferDoubleR.get();
        for (int i = 0; i < numProcSamples; ++i)
        {
            const double gWet = ramp.bypassFadeGainDouble.getNextValue();
            const double gDry = 1.0 - gWet;
            if (wetL != nullptr)
                wetL[i] = wetL[i] * gWet + dryL[i] * gDry;
            if (wetR != nullptr)
                wetR[i] = wetR[i] * gWet + dryR[i] * gDry;
        }
    }

    if (oversamplingFactor > 1)
    {
        oversampling.processDown(processBlock, originalBlock, static_cast<int>(originalBlock.getNumChannels()));
        processBlock = originalBlock;

        if (bypassBlendRequested)
        {
            double* wetL = processBlock.getNumChannels() > 0 ? processBlock.getChannelPointer(0) : nullptr;
            double* wetR = processBlock.getNumChannels() > 1 ? processBlock.getChannelPointer(1) : nullptr;
            for (int i = 0; i < numSamples; ++i)
            {
                const double gWet = ramp.bypassFadeGainDouble.getNextValue();
                if (wetL != nullptr)
                    wetL[i] *= gWet;
                if (wetR != nullptr)
                    wetR[i] *= gWet;
            }
        }
    }

    if (state.analyzerEnabled && state.analyzerSource == AnalyzerSource::Output)
        pushToFifo(processBlock, analyzerFifo);

    const float outputLinear = measureLevel(originalBlock);
    convo::publishAtomic(outputLevelLinear, outputLinear, std::memory_order_release);

    processOutputDouble(buffer, numSamples, state);

    int fadeLeft = ramp.fadeInSamplesLeft;
    if (fadeLeft > 0)
    {
        const int rampThisBlock = std::min(numSamples, fadeLeft);
        const double gainStep = 1.0 / static_cast<double>(FADE_IN_SAMPLES);
        const double startGain = static_cast<double>(FADE_IN_SAMPLES - fadeLeft) * gainStep;
        const int numChannels = buffer.getNumChannels();

        for (int ch = 0; ch < numChannels; ++ch)
            applyGainRamp(buffer.getWritePointer(ch), rampThisBlock, startGain, gainStep);

        ramp.fadeInSamplesLeft = fadeLeft - rampThisBlock;
    }
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_DSP_IO)

void AudioEngine::DSPCore::processOutputDouble(juce::AudioBuffer<double>& buffer,
                                               int numSamples,
                                               const ProcessingState& state) noexcept
{
    constexpr double kOutputHeadroom = 0.8912509381337456;
    const bool applyDither = (ditherBitDepth > 0);
    const int numChannels = std::min(2, buffer.getNumChannels());
    double* dataL = (numChannels > 0) ? alignedL.get() : nullptr;
    double* dataR = (numChannels > 1) ? alignedR.get() : nullptr;

    auto& dc = dcBlockers();
    dc.outputL.processStereo(dataL, dataR, numSamples, dc.outputR);

    {
        const __m256d vInf = _mm256_set1_pd(1.0e300);
        int i = 0;
        const int vEnd = numSamples / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vL = _mm256_loadu_pd(dataL + i);
            __m256d nanMaskL = _mm256_cmp_pd(vL, vL, _CMP_ORD_Q);
            __m256d infMaskL = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vL), vInf, _CMP_LT_OQ);
            _mm256_storeu_pd(dataL + i, _mm256_and_pd(vL, _mm256_and_pd(nanMaskL, infMaskL)));

            if (dataR != nullptr)
            {
                __m256d vR = _mm256_loadu_pd(dataR + i);
                __m256d nanMaskR = _mm256_cmp_pd(vR, vR, _CMP_ORD_Q);
                __m256d infMaskR = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vR), vInf, _CMP_LT_OQ);
                _mm256_storeu_pd(dataR + i, _mm256_and_pd(vR, _mm256_and_pd(nanMaskR, infMaskR)));
            }
        }

        for (; i < numSamples; ++i)
        {
            if (!isFiniteAndAbsBelowNoLibm(dataL[i], 1.0e300))
                dataL[i] = 0.0;

            if (dataR != nullptr && !isFiniteAndAbsBelowNoLibm(dataR[i], 1.0e300))
                dataR[i] = 0.0;
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
        for (int i = 0; i < numSamples; ++i)
        {
            dataL[i] *= kOutputHeadroom;
            if (dataR != nullptr)
                dataR[i] *= kOutputHeadroom;
        }
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
            _mm256_storeu_pd(dataL + i, _mm256_and_pd(vL, _mm256_and_pd(nanMaskL, infMaskL)));

            if (dataR != nullptr)
            {
                __m256d vR = _mm256_loadu_pd(dataR + i);
                __m256d nanMaskR = _mm256_cmp_pd(vR, vR, _CMP_ORD_Q);
                __m256d infMaskR = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), vR), vInf, _CMP_LT_OQ);
                _mm256_storeu_pd(dataR + i, _mm256_and_pd(vR, _mm256_and_pd(nanMaskR, infMaskR)));
            }
        }

        for (; i < numSamples; ++i)
        {
            if (!isFiniteAndAbsBelowNoLibm(dataL[i], 1.0e300))
                dataL[i] = 0.0;

            if (dataR != nullptr && !isFiniteAndAbsBelowNoLibm(dataR[i], 1.0e300))
                dataR[i] = 0.0;
        }
    }

    {
        const __m256d vLimit = _mm256_set1_pd(kOutputHeadroom);
        const __m256d vNegLimit = _mm256_set1_pd(-kOutputHeadroom);
        int i = 0;
        const int vEnd = (numSamples / 4) * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d vL = _mm256_loadu_pd(dataL + i);
            vL = _mm256_min_pd(_mm256_max_pd(vL, vNegLimit), vLimit);
            _mm256_storeu_pd(dataL + i, vL);

            if (dataR != nullptr)
            {
                __m256d vR = _mm256_loadu_pd(dataR + i);
                vR = _mm256_min_pd(_mm256_max_pd(vR, vNegLimit), vLimit);
                _mm256_storeu_pd(dataR + i, vR);
            }
        }

        for (; i < numSamples; ++i)
        {
            dataL[i] = juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataL[i]);
            if (dataR != nullptr)
                dataR[i] = juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataR[i]);
        }
    }

    applyFixedLatencyDelay(dataL, dataR, numSamples);

    juce::FloatVectorOperations::copy(buffer.getWritePointer(0, 0), dataL, numSamples);
    if (numChannels > 1 && dataR != nullptr)
        juce::FloatVectorOperations::copy(buffer.getWritePointer(1, 0), dataR, numSamples);

    for (int channel = numChannels; channel < buffer.getNumChannels(); ++channel)
        buffer.clear(channel, 0, numSamples);
}

#endif
