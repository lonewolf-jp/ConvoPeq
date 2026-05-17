#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "audioengine/AudioEngine.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "core/RCUReader.h"

#include "audioengine/AtomicAccess.h"

std::atomic<int> g_totalLatencyClampCount { 0 };

namespace
{
    // Audio thread path avoids libm calls for deterministic realtime behavior.
    static inline double equalPowerSin(double x) noexcept
    {
        const double t = x * (juce::MathConstants<double>::pi * 0.5);
        const double t2 = t * t;
        return t * (1.0 + t2 * (-1.0 / 6.0 + t2 * (1.0 / 120.0 + t2 * (-1.0 / 5040.0 + t2 * (1.0 / 362880.0)))));
    }
}

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RUNTIME)

// ────────────────────────────────────────────────────────────────
// Runtime (process block execution, latency calculation, DSP updates)
// ────────────────────────────────────────────────────────────────

void ConvolverProcessor::refreshLatency()
{
    // Index 3 for background/loader thread safety
    struct GlobalGuard {
        const ConvolverProcessor& cp;
        GlobalGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterGlobalReader(3); }
        ~GlobalGuard() { cp.exitGlobalReader(3); }
    } guard(*this);

    // IR未確定中（位相最適化/再構築中）は中間レイテンシ反映を完全停止する。
    if (!convo::consumeAtomic(irFinalized, std::memory_order_acquire))
        return;

    auto* conv = loadActiveEngine(std::memory_order_acquire);
    double totalLatency = 0.0;
    if (conv)
    {
        const int algorithmLatency = conv->storedDirectHeadEnabled ? 0 : juce::jmax(0, conv->latency);
        const int irPeakLatency = juce::jmax(0, conv->irLatency);
        convo::publishAtomic(uiAlgorithmLatencySamples, algorithmLatency, std::memory_order_release);
        convo::publishAtomic(uiIrPeakLatencySamples, irPeakLatency, std::memory_order_release);
        convo::publishAtomic(uiTotalLatencySamples, juce::jmin(juce::jmax(0, algorithmLatency + irPeakLatency), MAX_TOTAL_DELAY), std::memory_order_release);
        convo::publishAtomic(uiDirectHeadActive, conv->storedDirectHeadEnabled, std::memory_order_release);
        totalLatency = static_cast<double>(juce::jmin(juce::jmax(0, algorithmLatency + irPeakLatency), MAX_TOTAL_DELAY));
    }

    updateLatencyCache();
    requestHostDisplayUpdate();

    // Audio Thread に遅延値の更新を委譲（即時リセット用）
    convo::publishAtomic(pendingLatencyValue, totalLatency, std::memory_order_release);
    // [F-01 fix] 世代カウンターをインクリメント (NonRT → RT 通知、RTでのatomic write禁止対策)
    latencyResetPendingGen.fetch_add(1, std::memory_order_acq_rel);
    latencyChangeRequestedGen.fetch_add(1, std::memory_order_acq_rel);
}

void ConvolverProcessor::processBypassWithLatencyCompensation(juce::dsp::AudioBlock<double>& block,
                                                              const StereoConvolver& conv) noexcept
{
    const int procChannels = (std::min)(static_cast<int>(block.getNumChannels()), 2);
    const int numSamples = static_cast<int>(block.getNumSamples());
    if (procChannels == 0 || numSamples <= 0)
        return;

    double* delayBuf[2] = { delayBuffer[0].get(), delayBuffer[1].get() };
    int activeDelayCapacity = delayBufferCapacity;

    if (delayBuf[0] == nullptr || delayBuf[1] == nullptr || activeDelayCapacity < DELAY_BUFFER_SIZE)
        return;

    const int algorithmLatency = conv.storedDirectHeadEnabled ? 0 : juce::jmax(0, conv.latency);
    const int irPeakLatency = juce::jmax(0, conv.irLatency);
    int delaySamples = juce::jmax(0, algorithmLatency + irPeakLatency);
    delaySamples = juce::jmin(delaySamples, MAX_TOTAL_DELAY);
    delaySamples = juce::jlimit(0, DELAY_BUFFER_SIZE - 1, delaySamples);

    const int writePos = delayWritePos;

    // 1) 入力をリングバッファへ保存
    for (int ch = 0; ch < procChannels; ++ch)
    {
        const double* src = block.getChannelPointer(static_cast<size_t>(ch));
        double* buf = delayBuf[ch];

        const int samplesFirst = std::min(numSamples, DELAY_BUFFER_SIZE - writePos);
        const int samplesSecond = numSamples - samplesFirst;

        std::memcpy(buf + writePos, src, static_cast<size_t>(samplesFirst) * sizeof(double));
        if (samplesSecond > 0)
            std::memcpy(buf, src + samplesFirst, static_cast<size_t>(samplesSecond) * sizeof(double));
    }

    // 2) 遅延した信号を出力へ戻す
    int readPos = (writePos - delaySamples) & DELAY_BUFFER_MASK;
    if (readPos < 0)
        readPos += DELAY_BUFFER_SIZE;

    for (int ch = 0; ch < procChannels; ++ch)
    {
        const double* srcBuf = delayBuf[ch];
        double* dstBuf = block.getChannelPointer(static_cast<size_t>(ch));

        const int samplesFirst = std::min(numSamples, DELAY_BUFFER_SIZE - readPos);
        const int samplesSecond = numSamples - samplesFirst;

        juce::FloatVectorOperations::copy(dstBuf, srcBuf + readPos, samplesFirst);
        if (samplesSecond > 0)
            juce::FloatVectorOperations::copy(dstBuf + samplesFirst, srcBuf, samplesSecond);
    }

    const int nextWritePos = (writePos + numSamples) & DELAY_BUFFER_MASK;
    delayWritePos = nextWritePos;
}

void ConvolverProcessor::enterGlobalReader(int readerIndex) const noexcept
{
    if (auto* provider = getRcuProvider(); provider != nullptr)
        provider->enterRcuReader(readerIndex);
}

void ConvolverProcessor::exitGlobalReader(int readerIndex) const noexcept
{
    if (auto* provider = getRcuProvider(); provider != nullptr)
        provider->exitRcuReader(readerIndex);
}

//--------------------------------------------------------------
// process (Audio Thread)
// リアルタイム制約 (Real-time Constraints)
//    - メモリ確保なし (No Malloc)
//    - ロックなし (No Lock)
//    - ファイルI/Oなし (No I/O)
//    - 待機なし (No Wait)
//    - RCU (Read-Copy-Update) パターンを使用
//--------------------------------------------------------------
void ConvolverProcessor::process(juce::dsp::AudioBlock<double>& block)
{
    convo::RCUReaderGuard guard(runtimeRcuReader);

    auto& activeCrossfadeGain = crossfadeGain;
    auto& activeMixSmoother = mixSmoother;
    auto& activeLatencySmoother = latencySmoother;
    auto& activeDelayWritePos = delayWritePos;
    auto& activeOldDelay = oldDelay;

    double* delayBuf[2] = { delayBuffer[0].get(), delayBuffer[1].get() };
    double* dryBuf[2] = { dryBufferStorage[0].get(), dryBufferStorage[1].get() };
    double* oldDryBuf[2] = { oldDryBufferStorage[0].get(), oldDryBufferStorage[1].get() };
    double* wetBuf[2] = { wetBufferStorage[0].get(), wetBufferStorage[1].get() };
    double* smoothingBuf[2] = { smoothingBufferStorage[0].get(), smoothingBufferStorage[1].get() };
    int activeDryCapacity = dryBufferCapacity;
    int activeWetCapacity = wetBufferCapacity;
    int activeSmoothingCapacity = smoothingBufferCapacity;

    if (!convo::consumeAtomic(isPrepared, std::memory_order_acquire))
    {
        // [F-01 fix] RT atomic write 禁止: 世代カウンター比較のみ (非atomicローカル変数で追跡)
        {
            const uint64_t curGen = convo::consumeAtomic(firstProcessCallGen, std::memory_order_acquire);
            if (curGen != m_firstProcessCallGenSeen)
            {
                m_firstProcessCallGenSeen = curGen;
                block.clear();
            }
        }
        return;
    }

    static constexpr double kLatencyRetargetThresholdSamples = 2.0;

    juce::ScopedNoDenormals noDenormals;

    auto* conv = loadActiveEngine(std::memory_order_acquire);

    if (!conv)
        return;

    const auto runtimeSnapshot = captureRuntimeProcessSnapshot();

    if (runtimeSnapshot.bypassed)
    {
        processBypassWithLatencyCompensation(block, *conv);
        return;
    }

    // レイテンシー補正の更新
    {
        const int rawAlgorithmLatency = conv->storedDirectHeadEnabled ? 0 : juce::jmax(0, conv->latency);
        const int rawIrPeakLatency = juce::jmax(0, conv->irLatency);

        const int algorithmLatency = juce::jmin(rawAlgorithmLatency, MAX_BLOCK_SIZE);
        const int irPeakLatency = juce::jmin(rawIrPeakLatency, MAX_IR_LATENCY);

        const int64_t calculatedLatency64 = static_cast<int64_t>(algorithmLatency)
                                          + static_cast<int64_t>(irPeakLatency);
        int totalLatency = static_cast<int>(std::min<int64_t>(calculatedLatency64, MAX_TOTAL_DELAY));

        if (rawAlgorithmLatency != algorithmLatency || rawIrPeakLatency != irPeakLatency)
            g_totalLatencyClampCount.fetch_add(1, std::memory_order_acq_rel);

        if (absNoLibm(activeLatencySmoother.getTargetValue() - static_cast<double>(totalLatency)) >= kLatencyRetargetThresholdSamples)
        {
            if (!activeCrossfadeGain.isSmoothing())
            {
                activeOldDelay = activeLatencySmoother.getCurrentValue();
                activeCrossfadeGain.setCurrentAndTargetValue(0.0);
                activeCrossfadeGain.setTargetValue(1.0);
                activeLatencySmoother.setTargetValue(static_cast<double>(totalLatency));
            }
        }
    }

    const int procChannels = (std::min)((int)block.getNumChannels(), 2);
    const int numSamples = (int)block.getNumSamples();

    if (numSamples <= 0 || procChannels == 0 || numSamples > activeDryCapacity)
        return;

    jassert(activeWetCapacity >= numSamples);
    if (numSamples > activeWetCapacity)
        return;

    // [F-01 fix]
    {
        const uint64_t curGen = convo::consumeAtomic(latencyResetPendingGen, std::memory_order_acquire);
        if (curGen != m_latencyResetPendingGenSeen)
        {
            m_latencyResetPendingGenSeen = curGen;
            const double val = convo::consumeAtomic(pendingLatencyValue, std::memory_order_acquire);
            activeLatencySmoother.setCurrentAndTargetValue(val);
        }
    }

    {
        const uint64_t curGen = convo::consumeAtomic(latencyChangeRequestedGen, std::memory_order_acquire);
        if (curGen != m_latencyChangeRequestedGenSeen)
        {
            m_latencyChangeRequestedGenSeen = curGen;
            const double newTarget = convo::consumeAtomic(pendingLatencyValue, std::memory_order_acquire);
            if (absNoLibm(activeLatencySmoother.getTargetValue() - newTarget) >= 2.0)
            {
                if (!activeCrossfadeGain.isSmoothing())
                {
                    activeOldDelay = activeLatencySmoother.getCurrentValue();
                    activeLatencySmoother.setTargetValue(newTarget);
                    activeCrossfadeGain.setCurrentAndTargetValue(0.0);
                    activeCrossfadeGain.setTargetValue(1.0);
                }
            }
        }
    }

    {
        const uint64_t curGen = convo::consumeAtomic(mixSmootherResetPendingGen, std::memory_order_acquire);
        if (curGen != m_mixSmootherResetPendingGenSeen)
        {
            m_mixSmootherResetPendingGenSeen = curGen;
            activeMixSmoother.setCurrentAndTargetValue(static_cast<double>(runtimeSnapshot.mixTarget));
        }
    }

    {
        const uint64_t curGen = convo::consumeAtomic(smoothingTimeChangePendingGen, std::memory_order_acquire);
        if (curGen != m_smoothingTimeChangePendingGenSeen)
        {
            m_smoothingTimeChangePendingGenSeen = curGen;
            const float newTime = runtimeSnapshot.smoothingTimeSec;
            const double sampleRate = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
            if (sampleRate > 0.0)
            {
                const double currentVal = activeMixSmoother.getCurrentValue();
                const double targetVal = activeMixSmoother.getTargetValue();
                activeMixSmoother.reset(sampleRate, static_cast<double>(newTime));
                activeMixSmoother.setCurrentAndTargetValue(currentVal);
                activeMixSmoother.setTargetValue(targetVal);
            }
        }
    }

    const double targetMixValue = static_cast<double>(runtimeSnapshot.mixTarget);
    if (absNoLibm(activeMixSmoother.getTargetValue() - targetMixValue) > 1.0e-5)
    {
        activeMixSmoother.setTargetValue(targetMixValue);
    }

    const bool isSmoothing = activeMixSmoother.isSmoothing();
    const bool needsConvolution = isSmoothing || targetMixValue > 0.001;
    const bool needsDrySignal   = isSmoothing || targetMixValue < 0.999;

    // Dry信号生成
    {
        int wPos = activeDelayWritePos;
        for (int ch = 0; ch < procChannels; ++ch)
        {
            const double* src = block.getChannelPointer(ch);
            double* buf = delayBuf[ch];

            int samplesFirst = std::min(numSamples, DELAY_BUFFER_SIZE - wPos);
            int samplesSecond = numSamples - samplesFirst;

            std::memcpy(buf + wPos, src, samplesFirst * sizeof(double));
            if (samplesSecond > 0)
                std::memcpy(buf, src + samplesFirst, samplesSecond * sizeof(double));
        }

        if (activeCrossfadeGain.isSmoothing())
        {
            const double newDelay = activeLatencySmoother.getTargetValue();
            double* delayFadeRamp = wetBuf[0];
            int activeDelayCrossfadeSamples = 0;

            for (; activeDelayCrossfadeSamples < numSamples; ++activeDelayCrossfadeSamples)
            {
                delayFadeRamp[activeDelayCrossfadeSamples] = activeCrossfadeGain.getNextValue();
                if (!activeCrossfadeGain.isSmoothing())
                {
                    ++activeDelayCrossfadeSamples;
                    break;
                }
            }
            for (int i = activeDelayCrossfadeSamples; i < numSamples; ++i)
                delayFadeRamp[i] = 1.0;

            auto readInterpolated = [&](double delay, double* dst, int ch, int samplesToRead)
            {
                if (samplesToRead <= 0) return;

                const double* srcBuf = delayBuf[ch];
                double rPos = static_cast<double>(activeDelayWritePos) - delay;
                rPos -= std::floor(rPos / DELAY_BUFFER_SIZE) * DELAY_BUFFER_SIZE;

                const int iRead = static_cast<int>(rPos);
                const double frac = rPos - iRead;

                if (absNoLibm(frac) < 1.0e-6)
                {
                    int rPosInt = iRead;
                    int samplesFirst = std::min(samplesToRead, DELAY_BUFFER_SIZE - rPosInt);
                    juce::FloatVectorOperations::copy(dst, srcBuf + rPosInt, samplesFirst);
                    if (samplesToRead > samplesFirst)
                        juce::FloatVectorOperations::copy(dst + samplesFirst, srcBuf, samplesToRead - samplesFirst);
                    return;
                }
                else if (absNoLibm(frac - 1.0) < 1.0e-6)
                {
                    int rPosInt = (iRead + 1) & DELAY_BUFFER_MASK;
                    int samplesFirst = std::min(samplesToRead, DELAY_BUFFER_SIZE - rPosInt);
                    juce::FloatVectorOperations::copy(dst, srcBuf + rPosInt, samplesFirst);
                    if (samplesToRead > samplesFirst)
                        juce::FloatVectorOperations::copy(dst + samplesFirst, srcBuf, samplesToRead - samplesFirst);
                    return;
                }

                const double t = frac;
                const double t2 = t * t;
                const double t3 = t2 * t;
                const double w0 = -0.5 * t3 + t2 - 0.5 * t;
                const double w1 =  1.5 * t3 - 2.5 * t2 + 1.0;
                const double w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
                const double w3 =  0.5 * t3 - 0.5 * t2;

                int i = 0;
                if (iRead >= 1 && iRead + samplesToRead + 2 < DELAY_BUFFER_SIZE)
                {
                    const double* s = srcBuf + iRead;
#if defined(__AVX2__)
                    const __m256d vw0 = _mm256_set1_pd(w0);
                    const __m256d vw1 = _mm256_set1_pd(w1);
                    const __m256d vw2 = _mm256_set1_pd(w2);
                    const __m256d vw3 = _mm256_set1_pd(w3);

                    for (; i <= samplesToRead - 4; i += 4)
                    {
                        __m256d p0 = _mm256_loadu_pd(s + i - 1);
                        __m256d p1 = _mm256_loadu_pd(s + i);
                        __m256d p2 = _mm256_loadu_pd(s + i + 1);
                        __m256d p3 = _mm256_loadu_pd(s + i + 2);
                        __m256d sum = _mm256_mul_pd(p0, vw0);
                        sum = _mm256_fmadd_pd(p1, vw1, sum);
                        sum = _mm256_fmadd_pd(p2, vw2, sum);
                        sum = _mm256_fmadd_pd(p3, vw3, sum);
                        _mm256_storeu_pd(dst + i, sum);
                    }
#endif
                    for (; i < samplesToRead; ++i)
                        dst[i] = w0 * s[i - 1] + w1 * s[i] + w2 * s[i + 1] + w3 * s[i + 2];
                }
                else
                {
                    for (; i < samplesToRead; ++i)
                    {
                        int idx = iRead + i;
                        double p0 = srcBuf[(idx - 1) & DELAY_BUFFER_MASK];
                        double p1 = srcBuf[(idx    ) & DELAY_BUFFER_MASK];
                        double p2 = srcBuf[(idx + 1) & DELAY_BUFFER_MASK];
                        double p3 = srcBuf[(idx + 2) & DELAY_BUFFER_MASK];
                        dst[i] = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;
                    }
                }
            };

            if (activeDelayCrossfadeSamples > 0)
            {
                for (int ch = 0; ch < procChannels; ++ch)
                    readInterpolated(activeOldDelay, oldDryBuf[ch], ch, activeDelayCrossfadeSamples);
            }

            for (int ch = 0; ch < procChannels; ++ch)
                readInterpolated(newDelay, dryBuf[ch], ch, numSamples);

            if (activeDelayCrossfadeSamples > 0)
            {
                for (int ch = 0; ch < procChannels; ++ch)
                {
                    double* newSamples = dryBuf[ch];
                    const double* oldSamples = oldDryBuf[ch];
                    const double* fadeInRamp = delayFadeRamp;
#if defined(__AVX2__)
                    int i = 0;
                    const int vEnd = activeDelayCrossfadeSamples / 4 * 4;
                    const __m256d vOne = _mm256_set1_pd(1.0);
                    for (; i < vEnd; i += 4)
                    {
                        const __m256d vFade = _mm256_loadu_pd(fadeInRamp + i);
                        const __m256d vNew = _mm256_loadu_pd(newSamples + i);
                        const __m256d vOld = _mm256_loadu_pd(oldSamples + i);
                        const __m256d vOut = _mm256_add_pd(_mm256_mul_pd(vNew, vFade),
                                                           _mm256_mul_pd(vOld, _mm256_sub_pd(vOne, vFade)));
                        _mm256_storeu_pd(newSamples + i, vOut);
                    }
                    for (; i < activeDelayCrossfadeSamples; ++i)
                        newSamples[i] = newSamples[i] * fadeInRamp[i] + oldSamples[i] * (1.0 - fadeInRamp[i]);
#else
                    for (int i = 0; i < activeDelayCrossfadeSamples; ++i)
                        newSamples[i] = newSamples[i] * fadeInRamp[i] + oldSamples[i] * (1.0 - fadeInRamp[i]);
#endif
                }
            }

            if (!activeCrossfadeGain.isSmoothing())
            {
                activeLatencySmoother.setCurrentAndTargetValue(activeLatencySmoother.getTargetValue());
                activeOldDelay = activeLatencySmoother.getCurrentValue();
            }
        }
        else
        {
            int delayInt = static_cast<int>(activeLatencySmoother.getCurrentValue() + 0.5);
            int rPos = (activeDelayWritePos - delayInt) & DELAY_BUFFER_MASK;
            if (rPos < 0) rPos += DELAY_BUFFER_SIZE;

            for (int ch = 0; ch < procChannels; ++ch)
            {
                double* srcBuf = delayBuf[ch];
                double* dstBuf = dryBuf[ch];

                int samplesFirst = std::min(numSamples, DELAY_BUFFER_SIZE - rPos);
                int samplesSecond = numSamples - samplesFirst;

                juce::FloatVectorOperations::copy(dstBuf, srcBuf + rPos, samplesFirst);
                if (samplesSecond > 0)
                    juce::FloatVectorOperations::copy(dstBuf + samplesFirst, srcBuf, samplesSecond);
            }
        }

        activeDelayWritePos = (activeDelayWritePos + numSamples) & DELAY_BUFFER_MASK;
    }

    // Wet信号生成 & Mix
    const double headroom = CONVOLUTION_HEADROOM_GAIN;

    const double* wetGains = nullptr;
    const double* dryGains = nullptr;

    if (isSmoothing)
    {
        if (activeSmoothingCapacity < numSamples)
            return;

        double* wg = smoothingBuf[0];
        double* dg = smoothingBuf[1];

        for (int i = 0; i < numSamples; ++i)
        {
            const double mix = activeMixSmoother.getNextValue();
            wg[i] = equalPowerSin(mix)         * headroom;
            dg[i] = equalPowerSin(1.0 - mix);
        }
        wetGains = wg;
        dryGains = dg;
    }

    const int quantizedCallSamples = juce::jmax(1, conv->callQuantumSamples);
    const int prewarmedMaxSamples = juce::jmax(1, conv->prewarmedMaxSamples);
    const int guardedCallSamples = juce::jmin(quantizedCallSamples, prewarmedMaxSamples);
    const int callLen = guardedCallSamples;

    auto mixSmoothingSmall = [](double* dst, const double* wet, const double* dry,
                                const double* wetGain, const double* dryGain, int n) noexcept
    {
#if defined(__AVX2__)
        int i = 0;
        const int vEnd = n / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            const __m256d vWet = _mm256_loadu_pd(wet + i);
            const __m256d vDry = _mm256_loadu_pd(dry + i);
            const __m256d vWG = _mm256_loadu_pd(wetGain + i);
            const __m256d vDG = _mm256_loadu_pd(dryGain + i);
            const __m256d vOut = _mm256_add_pd(_mm256_mul_pd(vWet, vWG), _mm256_mul_pd(vDry, vDG));
            _mm256_storeu_pd(dst + i, vOut);
        }
        for (; i < n; ++i)
            dst[i] = wet[i] * wetGain[i] + dry[i] * dryGain[i];
#else
        for (int i = 0; i < n; ++i)
            dst[i] = wet[i] * wetGain[i] + dry[i] * dryGain[i];
#endif
    };

    auto mixSteadySmall = [](double* dst, const double* wet, const double* dry,
                             double wetG, double dryG, int n) noexcept
    {
#if defined(__AVX2__)
        int i = 0;
        const int vEnd = n / 4 * 4;
        const __m256d vWG = _mm256_set1_pd(wetG);
        const __m256d vDG = _mm256_set1_pd(dryG);
        for (; i < vEnd; i += 4)
        {
            const __m256d vWet = _mm256_loadu_pd(wet + i);
            const __m256d vDry = _mm256_loadu_pd(dry + i);
            const __m256d vOut = _mm256_add_pd(_mm256_mul_pd(vWet, vWG), _mm256_mul_pd(vDry, vDG));
            _mm256_storeu_pd(dst + i, vOut);
        }
        for (; i < n; ++i)
            dst[i] = wet[i] * wetG + dry[i] * dryG;
#else
        for (int i = 0; i < n; ++i)
            dst[i] = wet[i] * wetG + dry[i] * dryG;
#endif
    };

    auto scaleDrySmall = [](double* dst, const double* dry, const double* gain, int n) noexcept
    {
#if defined(__AVX2__)
        int i = 0;
        const int vEnd = n / 4 * 4;
        for (; i < vEnd; i += 4)
        {
            const __m256d vDry = _mm256_loadu_pd(dry + i);
            const __m256d vGain = _mm256_loadu_pd(gain + i);
            _mm256_storeu_pd(dst + i, _mm256_mul_pd(vDry, vGain));
        }
        for (; i < n; ++i)
            dst[i] = dry[i] * gain[i];
#else
        for (int i = 0; i < n; ++i)
            dst[i] = dry[i] * gain[i];
#endif
    };

    for (int ch = 0; ch < procChannels; ++ch)
    {
        const double wetG = needsConvolution ? (equalPowerSin(targetMixValue) * headroom) : 0.0;
        const double dryG = needsDrySignal   ?  equalPowerSin(1.0 - targetMixValue)         : 0.0;
        const double* inputBase = block.getChannelPointer(ch);
        double* wetBase = wetBuf[ch];
        const double* dryBase = dryBuf[ch];
        double* dstBase = block.getChannelPointer(ch);

        int processed = 0;
        while (processed < numSamples)
        {
            const int chunkSamples = juce::jmin(callLen, numSamples - processed);

            const double* input = inputBase + processed;
            double* wetOut = wetBase + processed;

            conv->process(ch, input, wetOut, chunkSamples);

            const double* wetSignal = wetOut;
            int validWetSamples = chunkSamples;

            double* dst = dstBase + processed;
            const double* dry = dryBase + processed;

            if (isSmoothing)
            {
                if (validWetSamples > 0)
                {
                    const double* wetGainPtr = wetGains + processed;
                    const double* dryGainPtr = dryGains + processed;
                    mixSmoothingSmall(dst, wetSignal, dry, wetGainPtr, dryGainPtr, validWetSamples);
                }

                if (chunkSamples > validWetSamples)
                {
                    const int remainder = chunkSamples - validWetSamples;
                    const double* remDry = dry + validWetSamples;
                    const double* remGain = dryGains + processed + validWetSamples;
                    double* remDst = dst + validWetSamples;
                    scaleDrySmall(remDst, remDry, remGain, remainder);
                }
            }
            else
            {
                if (validWetSamples > 0)
                {
                    mixSteadySmall(dst, wetSignal, dry, wetG, dryG, validWetSamples);
                }

                if (chunkSamples > validWetSamples)
                {
                    const int remainder = chunkSamples - validWetSamples;
                    const double* remDry = dry + validWetSamples;
                    double* remDst = dst + validWetSamples;
                    mixSteadySmall(remDst, remDry, remDry, 0.0, dryG, remainder);
                }
            }

            processed += chunkSamples;
        }
    }

}

//--------------------------------------------------------------
// Parameter getters/setters (Message Thread safe)
//--------------------------------------------------------------

void ConvolverProcessor::setMix(float mixAmount)
{
    float newVal = juce::jlimit(0.0f, 1.0f, mixAmount);
    float prev;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        prev = pendingOverride.mix;
        if (std::abs(prev - newVal) > 1.0e-5f)
            pendingOverride.mix = newVal;
    }
    if (std::abs(prev - newVal) > 1.0e-5f)
    {
        // H3: shadow atomic 廃止。pendingOverride.mix が唯一の Source of Truth。
        publishRuntimeProcessSnapshot();
        postCoalescedChangeNotification();
    }
}

float ConvolverProcessor::getMix() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return pendingOverride.mix;
}

void ConvolverProcessor::setBypass(bool shouldBypass)
{
    bool prev;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        prev = pendingOverride.bypassed;
        if (prev != shouldBypass)
            pendingOverride.bypassed = shouldBypass;
    }
    if (prev != shouldBypass)
    {
        // H3: shadow atomic 廃止。pendingOverride.bypassed が唯一の Source of Truth。
        publishRuntimeProcessSnapshot();
        postCoalescedChangeNotification();
    }
}

void ConvolverProcessor::setTargetIRLength(float timeSec)
{
    const float maxAllowedSec = getMaximumAllowedIRLengthSec(convo::consumeAtomic(currentSampleRate, std::memory_order_acquire));
    float clampedTime = juce::jlimit(IR_LENGTH_MIN_SEC, maxAllowedSec, timeSec);
    float prev;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        prev = pendingOverride.targetIRLengthSec;
        pendingOverride.targetIRLengthSec = clampedTime;
    }
    if (std::abs(prev - clampedTime) > 1e-5f)
    {
        postCoalescedChangeNotification();
    }
}

void ConvolverProcessor::applyAutoDetectedIRLength(float timeSec)
{
    const float maxAllowedSec = getMaximumAllowedIRLengthSec(convo::consumeAtomic(currentSampleRate, std::memory_order_acquire));
    const float clampedTime = juce::jlimit(IR_LENGTH_MIN_SEC, maxAllowedSec, timeSec);

    float prevTarget;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        pendingOverride.autoDetectedIRLengthSec = clampedTime;
        pendingOverride.irLengthManualOverride = false;
        prevTarget = pendingOverride.targetIRLengthSec;
    }

    if (std::abs(prevTarget - clampedTime) > 1.0e-5f)
    {
        {
            const juce::ScopedLock lock(pendingOverrideLock);
            pendingOverride.targetIRLengthSec = clampedTime;
        }
        postCoalescedChangeNotification();
    }
}

void ConvolverProcessor::setIRLengthManualOverride(bool isManual)
{
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        pendingOverride.irLengthManualOverride = isManual;
    }
}

void ConvolverProcessor::setSmoothingTime(float timeSec)
{
    float clampedTime = juce::jlimit(SMOOTHING_TIME_MIN_SEC, SMOOTHING_TIME_MAX_SEC, timeSec);
    float prev;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        prev = pendingOverride.smoothingTimeSec;
        if (std::abs(prev - clampedTime) > 1.0e-5f)
            pendingOverride.smoothingTimeSec = clampedTime;
    }
    if (std::abs(prev - clampedTime) > 1.0e-5f)
    {
        // H3: shadow atomic 廃止。pendingOverride.smoothingTimeSec が唯一の Source of Truth。
        publishRuntimeProcessSnapshot();
        // [F-01 fix] 世代カウンターをインクリメント (NonRT → RT 通知)
        smoothingTimeChangePendingGen.fetch_add(1, std::memory_order_acq_rel);

        postCoalescedChangeNotification();
    }
}

float ConvolverProcessor::getTargetIRLength() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return pendingOverride.targetIRLengthSec;
}

float ConvolverProcessor::getAutoDetectedIRLength() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return pendingOverride.autoDetectedIRLengthSec;
}

bool ConvolverProcessor::hasManualIRLengthOverride() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return pendingOverride.irLengthManualOverride;
}

float ConvolverProcessor::getSmoothingTime() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return pendingOverride.smoothingTimeSec;
}

// setMixRT / setSmoothingTimeRT: H3 修正により廃止。
// shadow atomic を除去したため実装を削除。
// 必要な場合は setMix() / setSmoothingTime()（Message Thread）を使用すること。

void ConvolverProcessor::setMixedTransitionStartHz(float hz)
{
    const float clamped = juce::jlimit(MIXED_F1_MIN_HZ, MIXED_F1_MAX_HZ, hz);
    float currentEnd;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        currentEnd = pendingOverride.mixedTransitionEndHz;
    }
    if (currentEnd < clamped + 10.0f)
        currentEnd = juce::jlimit(MIXED_F2_MIN_HZ, MIXED_F2_MAX_HZ, clamped + 10.0f);

    float prevStart, prevEnd;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        prevStart = pendingOverride.mixedTransitionStartHz;
        prevEnd = pendingOverride.mixedTransitionEndHz;
        pendingOverride.mixedTransitionStartHz = clamped;
        pendingOverride.mixedTransitionEndHz = currentEnd;
    }

    if (std::abs(prevStart - clamped) > 1.0e-5f || std::abs(prevEnd - currentEnd) > 1.0e-5f)
    {
        // H4 fix: UI notification のみ。rebuild トリガーは UI layer から snapshot publication 経由で行うこと。
        postCoalescedChangeNotification();
    }
}

float ConvolverProcessor::getMixedTransitionStartHz() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return pendingOverride.mixedTransitionStartHz;
}

void ConvolverProcessor::setMixedTransitionEndHz(float hz)
{
    float currentStart;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        currentStart = pendingOverride.mixedTransitionStartHz;
    }
    const float minEnd = (std::max)(MIXED_F2_MIN_HZ, currentStart + 10.0f);
    const float clamped = juce::jlimit(minEnd, MIXED_F2_MAX_HZ, hz);

    float prev;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        prev = pendingOverride.mixedTransitionEndHz;
        pendingOverride.mixedTransitionEndHz = clamped;
    }
    if (std::abs(prev - clamped) > 1.0e-5f)
    {
        // H4 fix: UI notification のみ。rebuild トリガーは UI layer から snapshot publication 経由で行うこと。
        postCoalescedChangeNotification();
    }
}

float ConvolverProcessor::getMixedTransitionEndHz() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return pendingOverride.mixedTransitionEndHz;
}

void ConvolverProcessor::setMixedPreRingTau(float tau)
{
    const float clamped = juce::jlimit(MIXED_TAU_MIN, MIXED_TAU_MAX, tau);
    float prev;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        prev = pendingOverride.mixedPreRingTau;
        pendingOverride.mixedPreRingTau = clamped;
    }
    if (std::abs(prev - clamped) > 1.0e-5f)
    {
        // H4 fix: UI notification のみ。rebuild トリガーは UI layer から snapshot publication 経由で行うこと。
        postCoalescedChangeNotification();
    }
}

float ConvolverProcessor::getMixedPreRingTau() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return pendingOverride.mixedPreRingTau;
}

void ConvolverProcessor::setExperimentalDirectHeadEnabled(bool enabled)
{
    bool prev;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        prev = pendingOverride.experimentalDirectHeadEnabled;
        pendingOverride.experimentalDirectHeadEnabled = enabled;
    }
    if (prev != enabled)
    {
        // H4 fix: UI notification のみ。rebuild トリガーは UI layer から snapshot publication 経由で行うこと。
        postCoalescedChangeNotification();
    }
}

void ConvolverProcessor::setRebuildDebounceMs(int ms)
{
    const int clampedMs = juce::jlimit(REBUILD_DEBOUNCE_MIN_MS, REBUILD_DEBOUNCE_MAX_MS, ms);
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        pendingOverride.rebuildDebounceMs = clampedMs;
    }
}

int ConvolverProcessor::getRebuildDebounceMs() const
{
    const juce::ScopedLock lock(pendingOverrideLock);
    return pendingOverride.rebuildDebounceMs;
}

void ConvolverProcessor::StereoConvolver::reset()
{
    if (nucConvolvers[0]) nucConvolvers[0]->Reset();
    if (nucConvolvers[1]) nucConvolvers[1]->Reset();
}

void ConvolverProcessor::StereoConvolver::process(int channel, const double* in, double* out, int numSamples)
{
#ifdef NUC_DEBUG_GUARDS
    if (nucConvolvers[channel])
        nucConvolvers[channel]->checkGuards();
#endif
    if (channel < 0 || channel >= 2 || !nucConvolvers[channel])
    {
        std::memset(out, 0, numSamples * sizeof(double));
        return;
    }

    nucConvolvers[channel]->Add(in, numSamples);
    const int got = nucConvolvers[channel]->Get(out, numSamples);
    if (got < numSamples)
        std::memset(out + got, 0, (numSamples - got) * sizeof(double));
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RUNTIME
