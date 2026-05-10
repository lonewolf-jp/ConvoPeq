#include <immintrin.h>
#include <JuceHeader.h>
#include "AudioEngine.h"

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_HELPERS_NUMERIC)

namespace
{
    static constexpr std::array<double, convo::FixedNoiseShaper::ORDER> kFixedNoiseShaperTunedCoeffs
    {
        0.46, 0.28, 0.17, 0.09
    };

    static constexpr std::array<double, convo::Fixed15TapNoiseShaper::ORDER> kFixed15TapNoiseShaperTunedCoeffs
    {
        2.033, -2.165, 1.959, -1.590, 1.221, -0.886, 0.604, -0.389, 0.235, -0.132, 0.068, -0.031, 0.012, -0.004, 0.001, 0.0
    };

    static constexpr std::array<double, kAdaptiveNoiseShaperOrder> kDefaultAdaptiveNoiseShaperCoeffs
    {
        0.82, -0.68, 0.55, -0.43, 0.33, -0.25, 0.18, -0.12, 0.07
    };

    static constexpr std::array<double, kAdaptiveNoiseShaperSampleRateBankCount> kAdaptiveSupportedSampleRatesHz
    {
        44100.0, 48000.0, 88200.0, 96000.0, 176400.0,
        192000.0, 352800.0, 384000.0, 705600.0, 768000.0
    };

    inline double absNoLibm(double x) noexcept
    {
        union { double d; uint64_t u; } v { x };
        v.u &= 0x7FFFFFFFFFFFFFFFULL;
        return v.d;
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

    inline double absDiffNoLibm(double a, double b) noexcept
    {
        return absNoLibm(a - b);
    }

    inline int clampAdaptiveBankIndex(int bankIndex) noexcept
    {
        if (bankIndex < 0)
            return 0;

        if (bankIndex >= kAdaptiveNoiseShaperSampleRateBankCount)
            return kAdaptiveNoiseShaperSampleRateBankCount - 1;

        return bankIndex;
    }

    inline juce::String makeAdaptiveCoeffPropertyName(double sampleRate, int coeffIndex)
    {
        return "adaptiveCoeff_" + juce::String(static_cast<int>(sampleRate + 0.5)) + "_" + juce::String(coeffIndex);
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
                dropCount.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    static_assert(CustomInputOversampler::isLinearPhaseFIR
                  && CustomInputOversampler::isSymmetricUpDown,
                  "Oversampling latency formula assumes symmetric linear-phase FIR with identical up/down taps");

    inline double estimateOversamplingLatencySamplesImpl(int oversamplingFactor,
                                                         AudioEngine::OversamplingType oversamplingType,
                                                         double baseSampleRate) noexcept
    {
        if (oversamplingFactor <= 1 || baseSampleRate <= 0.0)
            return 0.0;

        const int numStages = (oversamplingFactor == 8) ? 3 : ((oversamplingFactor == 4) ? 2 : ((oversamplingFactor == 2) ? 1 : 0));
        if (numStages <= 0)
            return 0.0;

        const int* taps = nullptr;
        static constexpr int iirLikeTaps[3] = { 511, 127, 31 };
        static constexpr int linearPhaseTaps[3] = { 1023, 255, 63 };
        taps = (oversamplingType == AudioEngine::OversamplingType::LinearPhase) ? linearPhaseTaps : iirLikeTaps;

        double totalLatencyBaseSamples = 0.0;
        for (int stage = 0; stage < numStages; ++stage)
        {
            const double stageRate = baseSampleRate * static_cast<double>(1 << (stage + 1));
            const double groupDelaySamplesAtStageRate = static_cast<double>(taps[stage] - 1);
            const double delayBaseSamples = groupDelaySamplesAtStageRate * (baseSampleRate / stageRate);
            totalLatencyBaseSamples += delayBaseSamples;
        }

        return totalLatencyBaseSamples;
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

    inline void applyGainRamp(float* __restrict data, int numSamples,
                              float startGain, float increment) noexcept
    {
        __m256 vGain = _mm256_set_ps(startGain + 7.0f * increment,
                                     startGain + 6.0f * increment,
                                     startGain + 5.0f * increment,
                                     startGain + 4.0f * increment,
                                     startGain + 3.0f * increment,
                                     startGain + 2.0f * increment,
                                     startGain + increment,
                                     startGain);
        const __m256 vInc8 = _mm256_set1_ps(8.0f * increment);

        int i = 0;
        const int vEnd = numSamples / 8 * 8;
        for (; i < vEnd; i += 8)
        {
            __m256 vData = _mm256_loadu_ps(data + i);
            _mm256_storeu_ps(data + i, _mm256_mul_ps(vData, vGain));
            vGain = _mm256_add_ps(vGain, vInc8);
        }

        float gain = startGain + static_cast<float>(i) * increment;
        for (; i < numSamples; ++i) { data[i] *= gain; gain += increment; }
    }

    inline bool isAligned64(const void* ptr) noexcept
    {
        return (reinterpret_cast<std::uintptr_t>(ptr) & static_cast<std::uintptr_t>(63)) == 0;
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

    // Padé近似による高速tanh (std::exp回避)
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

    [[maybe_unused]] static void softClipBlockAVX2(double* __restrict data, int numSamples,
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

        const __m256d vNumA        = _mm256_set1_pd(10395.0);
        const __m256d vNumB        = _mm256_set1_pd(1260.0);
        const __m256d vNumC        = _mm256_set1_pd(21.0);
        const __m256d vDenB        = _mm256_set1_pd(4725.0);
        const __m256d vDenC        = _mm256_set1_pd(210.0);
        const __m256d vZero        = _mm256_setzero_pd();
        const __m256d vSignMask    = _mm256_set1_pd(-0.0);

        double prevScalar = prevSampleInOut;

        int i = 0;
        const int vEnd = numSamples / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            __m256d x = _mm256_loadu_pd(data + i);

            {
                const __m128d xLow = _mm256_castpd256_pd128(x);
                const __m128d xHigh = _mm256_extractf128_pd(x, 1);
                const __m128d prevLow128 = _mm_unpacklo_pd(_mm_set_sd(prevScalar), xLow);
                const __m128d prevHigh128 = _mm_shuffle_pd(xLow, xHigh, 0x1);
                const __m256d prevVec = _mm256_set_m128d(prevHigh128, prevLow128);

                const __m256d midVec = _mm256_mul_pd(_mm256_add_pd(prevVec, x), vHalf);
                const __m256d absMidVec = _mm256_andnot_pd(vSignMask, midVec);

                const __m256d vTiny = _mm256_set1_pd(1e-15);
                const __m256d needMidClip = _mm256_cmp_pd(absMidVec, vThreshold, _CMP_GT_OQ);
                const __m256d safeAbsMid = _mm256_max_pd(absMidVec, vTiny);
                const __m256d midGainRaw = _mm256_div_pd(vThreshold, safeAbsMid);
                const __m256d midGain = _mm256_blendv_pd(vOne, midGainRaw, needMidClip);
                x = _mm256_mul_pd(x, midGain);
            }

            __m256d absX = _mm256_andnot_pd(vSignMask, x);
            __m256d needClip = _mm256_cmp_pd(absX, vClipStart, _CMP_GT_OQ);

            __m256d maskSignPos = _mm256_cmp_pd(x, vZero, _CMP_GT_OQ);
            __m256d sign = _mm256_blendv_pd(vMinusOne, vOne, maskSignPos);

            __m256d arg = _mm256_mul_pd(_mm256_sub_pd(absX, vThreshold), vRecipKnee);
            __m256d satHi = _mm256_cmp_pd(arg, vThree, _CMP_GE_OQ);
            __m256d satLo = _mm256_cmp_pd(arg, vNegThree, _CMP_LE_OQ);
            __m256d arg2 = _mm256_mul_pd(arg, arg);

            __m256d num = _mm256_mul_pd(arg,
                                _mm256_fmadd_pd(arg2,
                                    _mm256_fmadd_pd(arg2, vNumC, vNumB),
                                vNumA));
            __m256d den = _mm256_fmadd_pd(arg2,
                                _mm256_fmadd_pd(arg2,
                                    _mm256_fmadd_pd(arg2, vDenC, vDenB),
                                vDenC),
                               vNumA);
            __m256d tanhVal = _mm256_div_pd(num, den);
            tanhVal = _mm256_blendv_pd(tanhVal, vOne, satHi);
            tanhVal = _mm256_blendv_pd(tanhVal, vMinusOne, satLo);

            __m256d clipped = _mm256_fmadd_pd(vKnee, tanhVal, vThreshold);
            __m256d linear = absX;
            __m256d mixed = _mm256_fmadd_pd(_mm256_sub_pd(clipped, linear), needClip, linear);

            __m256d factor = _mm256_mul_pd(vAsym, _mm256_sub_pd(vOne, sign));
            factor = _mm256_mul_pd(factor, vHalf);
            factor = _mm256_mul_pd(factor, needClip);
            __m256d asymmetric_gain = _mm256_sub_pd(vOne, factor);

            __m256d result = _mm256_mul_pd(sign, _mm256_mul_pd(mixed, asymmetric_gain));
            result = _mm256_blendv_pd(x, result, needClip);
            _mm256_storeu_pd(data + i, result);

            prevScalar = data[i + 3];
        }

        for (; i < numSamples; ++i)
        {
            const double mid = (prevScalar + data[i]) * 0.5;
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
}

#endif // defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_HELPERS_NUMERIC)
