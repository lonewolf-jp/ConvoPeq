#include <JuceHeader.h>
#include <immintrin.h>
#include "AudioEngine.h"
#include "DiagnosticsConfig.h"
#include "core/TimeUtils.h"

#include <cstdint>
#include <atomic>

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
namespace {
[[maybe_unused]] void diagLog(const juce::String& message)
{
    DBG(message); // NOLINT(rt-logger)
    juce::Logger::writeToLog(message); // NOLINT(rt-logger)
}
}

// work60: Numeric-Only DiagEvent — String構築を排除（変数は extern / DSPCoreFloat.cpp で一元管理）
// ★ [work65] eqDiagBuffer は AudioEngine.h の extern 宣言 + DSPCoreFloat.cpp の実体を参照

void logEqTime(uint64_t eqStartUs, int numSamples, int /*numChannels*/,
               const convo::EQParameters* eqParams,
               convo::ProcessingOrder order,
               double sampleRate,
               uint64_t callbackSeq, uint32_t cpu)
{
    const uint64_t eqElapsedUs = convo::getCurrentTimeUs() - eqStartUs;
    if (sampleRate <= 0.0 || numSamples <= 0 || eqDiagBuffer == nullptr) return;
    if ((callbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK) != 0) return;

    int activeBands = 0;
    if (eqParams != nullptr) {
        for (int i = 0; i < 20; ++i)
            if (eqParams->bands[i].enabled
                && std::abs(eqParams->bands[i].gain) > 0.01f)
                ++activeBands;
    }
    const double expectedUs = static_cast<double>(numSamples) / sampleRate * 1e6;
    const uint32_t budgetPermille = (expectedUs > 0.0)
        ? static_cast<uint32_t>((static_cast<double>(eqElapsedUs) / expectedUs) * 1000.0)
        : 0;

    DiagEvent event{};
    event.category = DiagCategory::EqTime;
    event.eventIndex = callbackSeq;
    event.data.eqTime.cpu = cpu;
    event.data.eqTime.us = eqElapsedUs;
    event.data.eqTime.activeBands = static_cast<uint8_t>(activeBands);
    event.data.eqTime.order = static_cast<uint8_t>(order);
    event.data.eqTime.budgetPercent = budgetPermille;
    if (eqDiagBuffer->push(event))
    {
        eqTickPushed->value.fetch_add(1, std::memory_order_relaxed);
        eqTotalPushed->fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        eqTickDropped->value.fetch_add(1, std::memory_order_relaxed);
    }
}
#endif

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

inline double fastTanh(double x) noexcept
{
    using namespace TanhApprox;

    if (x >= CLIP_THRESHOLD) return 1.0;
    if (x <= -CLIP_THRESHOLD) return -1.0;
    const double x2 = x * x;

    const double num = x * (NUM_A + x2 * (NUM_B + x2 * NUM_C));
    const double den = DEN_A + x2 * (DEN_B + x2 * (DEN_C + x2));
    return num / den;
}

inline double musicalSoftClipScalar(double x, double threshold, double knee, double asymmetry) noexcept
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

void softClipBlockAVX2(double* __restrict data, int numSamples,
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
    const __m256d vHalf        = _mm256_set1_pd(0.5);

    const __m256d vNumA        = _mm256_set1_pd(TanhApprox::NUM_A);
    const __m256d vNumB        = _mm256_set1_pd(TanhApprox::NUM_B);
    const __m256d vNumC        = _mm256_set1_pd(TanhApprox::NUM_C);
    const __m256d vDenA        = _mm256_set1_pd(TanhApprox::DEN_A);
    const __m256d vDenB        = _mm256_set1_pd(TanhApprox::DEN_B);
    const __m256d vDenC        = _mm256_set1_pd(TanhApprox::DEN_C);
    const __m256d vClipThreshold = _mm256_set1_pd(TanhApprox::CLIP_THRESHOLD);
    const __m256d vZero        = _mm256_setzero_pd();
    const __m256d vSignMask    = _mm256_set1_pd(-0.0);

    double prevScalar = prevSampleInOut;

    int i = 0;
    const int vEnd = numSamples / 4 * 4;
    for (; i < vEnd; i += 4)
    {
            __m256d x    = _mm256_loadu_pd(data + i);

        // [P3] midVec事前平均化ブロックを完全削除
        // このブロックは threshold レベルでのハードリミッティングを引き起こし、
        // SoftClip本来の滑らかなKnee特性を損なっていた。
        // x はそのまま後続のSoftClip（fastTanh近似）へ流れる。
        // 削除によりAVX2パスとスカラーフォールバックパスの動作が一致する。

        __m256d absX = _mm256_andnot_pd(vSignMask, x);

        __m256d needClip = _mm256_cmp_pd(absX, vClipStart, _CMP_GT_OQ);

        __m256d maskSignPos = _mm256_cmp_pd(x, vZero, _CMP_GT_OQ);
        __m256d sign = _mm256_blendv_pd(vMinusOne, vOne, maskSignPos);

        __m256d t = _mm256_mul_pd(_mm256_sub_pd(absX, vClipStart), vRecipKnee2);
        t = _mm256_min_pd(_mm256_max_pd(t, vZero), vOne);
        __m256d t2 = _mm256_mul_pd(t, t);
        __m256d ks = _mm256_mul_pd(t2, _mm256_fnmadd_pd(vTwo, t, vThree));

        __m256d arg = _mm256_mul_pd(_mm256_sub_pd(absX, vThreshold), vRecipKnee);
        __m256d satHi    = _mm256_cmp_pd(arg, vClipThreshold, _CMP_GE_OQ);
        __m256d satLo    = _mm256_cmp_pd(arg, _mm256_sub_pd(vZero, vClipThreshold), _CMP_LE_OQ);
        __m256d arg2     = _mm256_mul_pd(arg, arg);

        __m256d num      = _mm256_mul_pd(arg,
                            _mm256_fmadd_pd(arg2,
                                _mm256_fmadd_pd(arg2, vNumC, vNumB),
                            vNumA));
        __m256d denInner = _mm256_fmadd_pd(arg2, vOne, vDenC);
        __m256d denMid   = _mm256_fmadd_pd(arg2, denInner, vDenB);
        __m256d den      = _mm256_fmadd_pd(arg2, denMid, vDenA);
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
        const double nextPrev = data[i + 3]; // [BUG-04] store前に元の入力値を退避
            _mm256_storeu_pd(data + i, result);

        prevScalar = nextPrev;
    }

    for (; i < numSamples; ++i)
    {
        const double inputVal = data[i]; // 元の入力を退避
        double x = inputVal;
        if (absNoLibm(x) > clip_start)
            x = musicalSoftClipScalar(x, threshold, knee, asymmetry);

        data[i] = x;
        prevScalar = inputVal; // 修正: 処理前の生入力値を保存
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

void AudioEngine::DSPCore::processDoubleToBuffer(const juce::AudioBuffer<double>& source,
                                                 juce::AudioBuffer<double>& destination,
                                                 LockFreeAudioRingBuffer& analyzerFifo,
                                                 std::atomic<float>* inputLevelLinear,
                                                 std::atomic<float>* outputLevelLinear,
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

void AudioEngine::DSPCore::processDouble(juce::AudioBuffer<double>& buffer,
                                         LockFreeAudioRingBuffer& analyzerFifo,
                                         std::atomic<float>* inputLevelLinear,
                                         std::atomic<float>* outputLevelLinear,
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
    if (inputLevelLinear != nullptr)
        convo::publishAtomic(*inputLevelLinear, rawInputLinearD, std::memory_order_release);

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

    // [DIAG] テストトーン注入は削除（work52調査終了）

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

    const convo::EQParameters* eqParamsToUse = state.eqParams;
    const EQCoeffCache* eqCacheToUse = state.eqCache;

    eqRt().setBypassFromRT(state.eqBypassed); // RT-local shadow に書き込み（publishAtomic の RT 使用禁止のため）

    if (state.order == ProcessingOrder::ConvolverThenEQ)
    {
        if (!state.convBypassed)
        {
            convolverRt().process(processBlock);
            // ★ [work65] work52キャプチャコード削除
        }
        if (!state.eqBypassed)
        {
            if (eqParamsToUse != nullptr)
            {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
                const uint64_t eqStartUs = convo::getCurrentTimeUs();
#endif
                eqRt().process(processBlock, *eqParamsToUse, eqCacheToUse);
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
                logEqTime(eqStartUs, numProcSamples, numProcChannels, eqParamsToUse, state.order, sampleRate, currentCallbackSeq, currentCpu);
#endif
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
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
                const uint64_t eqStartUs = convo::getCurrentTimeUs();
#endif
                eqRt().process(processBlock, *eqParamsToUse, eqCacheToUse);
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
                logEqTime(eqStartUs, numProcSamples, numProcChannels, eqParamsToUse, state.order, sampleRate, currentCallbackSeq, currentCpu);
#endif
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
            // ★ [work65] work52キャプチャコード削除
        }
    }

    {
        const bool convActive = !state.convBypassed;
        const bool eqActive   = !state.eqBypassed;
        if (convActive || eqActive)
        {
            const bool convIsLast = convActive &&
                (!eqActive || state.order == ProcessingOrder::EQThenConvolver);
            outputFilter.process(processBlock, convIsLast,
                                 state.convHCMode, state.convLCMode, state.eqLPFMode);
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

        if (oversamplingFactor > 1)
        {
            // 既存: アップサンプル領域でSoftClip（エイリアシング保護済み）
            for (int ch = 0; ch < numProcChannels; ++ch)
            {
                double* data = processBlock.getChannelPointer(ch);
                softClipBlockAVX2(data, numProcSamples, clipThreshold, clipKnee, clipAsymmetry,
                                   history.softClipPrevSample[ch < 2 ? ch : 1]);
            }
        }
        else
        {
            // 局所2倍OS: processUp → SoftClip → processDown
            const int nChOS = static_cast<int>(originalBlock.getNumChannels());
            auto osBlock = softClipOS.processUp(originalBlock, nChOS);
            const int osSamples = static_cast<int>(osBlock.getNumSamples());
            for (int ch = 0; ch < nChOS; ++ch)
            {
                double* osData = osBlock.getChannelPointer(ch);
                softClipBlockAVX2(osData, osSamples, clipThreshold, clipKnee, clipAsymmetry,
                                   history.softClipPrevSample[ch < 2 ? ch : 1]);
            }
            softClipOS.processDown(osBlock, originalBlock, nChOS);
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
            const double* dryL = dryBypassBufferDoubleL ? dryBypassBufferDoubleL.get() : nullptr;
            const double* dryR = dryBypassBufferDoubleR ? dryBypassBufferDoubleR.get() : nullptr;
            const bool canUseDry = (dryL != nullptr)
                && (dryR != nullptr)
                && (dryBypassCapacityDouble >= numSamples);
            for (int i = 0; i < numSamples; ++i)
            {
                const double gWet = ramp.bypassFadeGainDouble.getNextValue();
                const double gDry = 1.0 - gWet;
                if (wetL != nullptr)
                    wetL[i] = canUseDry ? (wetL[i] * gWet + dryL[i] * gDry) : (wetL[i] * gWet);
                if (wetR != nullptr)
                    wetR[i] = canUseDry ? (wetR[i] * gWet + dryR[i] * gDry) : (wetR[i] * gWet);
            }
        }
    }

    if (state.analyzerEnabled && state.analyzerSource == AnalyzerSource::Output)
        pushToFifo(processBlock, analyzerFifo);

    const float outputLinear = measureLevel(originalBlock);
    if (outputLevelLinear != nullptr)
        convo::publishAtomic(*outputLevelLinear, outputLinear, std::memory_order_release);

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

void AudioEngine::DSPCore::processOutputDouble(juce::AudioBuffer<double>& buffer,
                                               int numSamples,
                                               const ProcessingState& state) noexcept
{
    constexpr double kOutputHeadroom = 0.8912509381337456;
    const bool applyDither = (ditherBitDepth > 0);
    const int numChannels = std::min(2, buffer.getNumChannels());

    if (numSamples <= 0 || numChannels <= 0)
    {
        if (numSamples > 0)
            buffer.clear();
        return;
    }

    double* dataL = (numChannels > 0) ? alignedL.get() : nullptr;
    double* dataR = (numChannels > 1) ? alignedR.get() : nullptr;

    if (dataL == nullptr)
    {
        buffer.clear();
        return;
    }

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

    // TruePeak検出（BS.1770-4/5準拠）
    truePeakDetector.processBlock(dataL, dataR, numSamples);

    // LUFSブロック平均電力（BS.1770-4/5 + EBU R128）
    loudnessMeter.processBlock(dataL, dataR, numSamples);

    if (noiseShaperType == NoiseShaperType::Adaptive9thOrder
        && state.adaptiveCoeffSet != nullptr
        && (activeAdaptiveCoeffBankIndex != state.adaptiveCoeffBankIndex
            || activeAdaptiveCoeffGeneration != state.adaptiveCoeffGeneration))
    {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansStartUs = convo::getCurrentTimeUs();
#endif
        adaptiveBankSwitchCount.fetch_add(1, std::memory_order_relaxed);
        adaptiveNoiseShaper.applyMatchedCoefficients(state.adaptiveCoeffSet->k, kAdaptiveNoiseShaperOrder);
        activeAdaptiveCoeffBankIndex = state.adaptiveCoeffBankIndex;
        activeAdaptiveCoeffGeneration = state.adaptiveCoeffGeneration;
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansElapsedUs = convo::getCurrentTimeUs() - ansStartUs;
        // ★ [work65] RT-safe: writeToLog → LockFreeRingBuffer (DiagEvent)
        if ((currentCallbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK) == 0 && eqDiagBuffer != nullptr)
        {
            DiagEvent event{};
            event.category = DiagCategory::AnsSwitchTime;
            event.eventIndex = currentCallbackSeq;
            event.data.ansSwitchTime.elapsedUs = ansElapsedUs;
            [[maybe_unused]] const bool pushed = eqDiagBuffer->push(event);
            juce::ignoreUnused(pushed); // drop on full is acceptable
        }
#endif
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
