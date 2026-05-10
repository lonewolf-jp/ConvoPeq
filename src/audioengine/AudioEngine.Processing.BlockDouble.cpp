#include <JuceHeader.h>
#include "AudioEngine.h"
#include "core/RCUReader.h"

static thread_local convo::RCUReader tls_rcuReader;

static void retireDSP(AudioEngine::DSPCore* dsp)
{
    if (dsp) convo::retireObject(dsp, [](void* p) { delete static_cast<AudioEngine::DSPCore*>(p); });
}

namespace
{
    inline double absDiffNoLibm(double a, double b) noexcept
    {
        return absNoLibm(a - b);
    }
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_BLOCK_DOUBLE)
void AudioEngine::processBlockDouble (juce::AudioBuffer<double>& buffer)
{
    const juce::ScopedNoDenormals noDenormals;
    m_audioBlockCounter.fetch_add(1, std::memory_order_release);

    // ★ 追加: RCU ガードで現在の DSP を保護する
    convo::RCUReaderGuard rcuGuard(tls_rcuReader);
    const int numSamples = buffer.getNumSamples();
    // 事前サニティチェック (getNextAudioBlock と同様)
    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20;
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        buffer.clear();
        return;
    }

    DSPCore* dsp = currentDSP.load(std::memory_order_acquire);
    if (dsp == nullptr)
    {
        buffer.clear();
        return;
    }

    // AudioThread入口で、現在のDSPが持つ全てのNUCのガードをチェック（デバッグ時のみ）
        #ifdef NUC_DEBUG_GUARDS
        {
        dsp->convolver.debugCheckNucGuards();
        }
    #endif

    // --- ProcessingStateを現行設計で初期化 ---
    const bool eqBypassed = eqBypassActive.load(std::memory_order_relaxed);
    const bool convBypassed = convBypassActive.load(std::memory_order_relaxed);
    const ProcessingOrder order = currentProcessingOrder.load(std::memory_order_relaxed);
    const bool analyzerEnabledNow = analyzerEnabled.load(std::memory_order_relaxed);
    const AnalyzerSource analyzerSourceNow = currentAnalyzerSource.load(std::memory_order_relaxed);
    const convo::HCMode hcMode = convHCFilterMode.load(std::memory_order_relaxed);
    const convo::LCMode lcMode = convLCFilterMode.load(std::memory_order_relaxed);
    const convo::HCMode lpfMode = eqLPFFilterMode.load(std::memory_order_relaxed);
    const int adaptiveCoeffBankIndex = currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
    const auto& adaptiveCoeffBank = getAdaptiveCoeffBankForIndex(adaptiveCoeffBankIndex);
    const uint32_t genSnapshot = adaptiveCoeffBank.generation.load(std::memory_order_acquire);
    const CoeffSet* safeAdaptiveSet = AudioEngine::getActiveCoeffSet(adaptiveCoeffBank);
    const uint32_t adaptiveGenAfter = genSnapshot;
    const bool adaptiveCaptureEnabled = noiseShaperLearner && noiseShaperLearner->isRunning();

    DSPCore::ProcessingState procState {
        eqBypassed,
        convBypassed,
        order,
        analyzerSourceNow,
        analyzerEnabledNow,
        softClipEnabled.load(std::memory_order_relaxed),
        saturationAmount.load(std::memory_order_relaxed),
        inputHeadroomGain.load(std::memory_order_relaxed),
        outputMakeupGain.load(std::memory_order_relaxed),
        convolverInputTrimGain.load(std::memory_order_relaxed),
        hcMode,
        lcMode,
        lpfMode,
        adaptiveCoeffBankIndex,
        safeAdaptiveSet,
        adaptiveGenAfter,
        static_cast<int>(dsp->sampleRate + 0.5),
        dsp->ditherBitDepth,
        dsp->currentCaptureSessionId,
        adaptiveCaptureEnabled ? &audioCaptureQueue : nullptr
    };

    // DSPCore 固有の上限チェック (getNextAudioBlock と同様)
    if (numSamples > dsp->maxSamplesPerBlock)
    {
        buffer.clear();
        return;
    }

    // EQ スナップショットフェードを進める (getNextAudioBlock と同等)
    if (m_coordinator.isFading())
        m_coordinator.advanceFade(numSamples);

        debugLastCoordinatorIsFading.store(m_coordinator.isFading() ? 1 : 0, std::memory_order_relaxed);

        float snapshotAlpha = 1.0f;
        const convo::GlobalSnapshot* snapshotFrom = nullptr;
        const convo::GlobalSnapshot* snapshotTo = nullptr;
        const bool updateFadeReturned = m_coordinator.updateFade(snapshotAlpha, snapshotFrom, snapshotTo);
        debugLastUpdateFadeReturned.store(updateFadeReturned ? 1 : 0, std::memory_order_relaxed);
        debugLastSnapshotFromNull.store(snapshotFrom == nullptr ? 1 : 0, std::memory_order_relaxed);
        debugLastSnapshotToNull.store(snapshotTo == nullptr ? 1 : 0, std::memory_order_relaxed);
        (void) snapshotAlpha;

    const double engineSampleRate = currentSampleRate.load(std::memory_order_relaxed);
    if (absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
    {
        inputLevelLinear.store(0.0f);
        // --- クロスフェード・遅延整合処理（現行設計に準拠） ---
        DSPCore* fading = sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire));
        bool useDryAsOld = dspCrossfadeUseDryAsOld.load(std::memory_order_acquire);
        int pendingFadeDelayBlocks = dspCrossfadeStartDelayBlocks.load(std::memory_order_acquire);
        if (fading != nullptr
            && !useDryAsOld
            && dspCrossfadePending.load(std::memory_order_acquire)
            && pendingFadeDelayBlocks > 0)
        {
            dspCrossfadeStartDelayBlocks.store(pendingFadeDelayBlocks - 1, std::memory_order_release);

            auto fadingState = procState;
            fadingState.analyzerEnabled = false;
            fadingState.adaptiveCaptureQueue = nullptr;

            std::atomic<float> fadingInputMeter { 0.0f };
            std::atomic<float> fadingOutputMeter { 0.0f };
            fading->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, fadingState);
            return;
        }

        if ((fading != nullptr || firstIrDryCrossfadePending.load(std::memory_order_acquire))
            && dspCrossfadePending.exchange(false, std::memory_order_acq_rel))
        {
            const double fadeSec = std::max(0.001, queuedFadeTimeSec.load(std::memory_order_acquire));
            dspCrossfadeGain.reset(std::max(1.0, dsp->sampleRate), fadeSec);
            dspCrossfadeGain.setCurrentAndTargetValue(0.0);
            dspCrossfadeGain.setTargetValue(1.0);

            // レイテンシ整合値を Audio Thread スナップショットへ反映する。
            latencyDelayOld_RT = latencyDelayOld.load(std::memory_order_acquire);
            latencyDelayNew_RT = latencyDelayNew.load(std::memory_order_acquire);

            if (firstIrDryCrossfadePending.exchange(false, std::memory_order_acq_rel))
            {
                dspCrossfadeUseDryAsOld.store(true, std::memory_order_release);
                useDryAsOld = true;
            }
        }

        const bool canCrossfade = (fading != nullptr || useDryAsOld)
            && dspCrossfadeGain.isSmoothing()
            && dspCrossfadeDoubleBuffer.getNumChannels() >= 2
            && dspCrossfadeDoubleBuffer.getNumSamples() >= numSamples;

        if (canCrossfade)
        {
            // 旧DSPの出力をバッファに生成
            dspCrossfadeDoubleBuffer.clear(0, 0, numSamples);
            dspCrossfadeDoubleBuffer.clear(1, 0, numSamples);

            auto fadingState = procState;
            fadingState.analyzerEnabled = false;
            fadingState.adaptiveCaptureQueue = nullptr;

            std::atomic<float> fadingInputMeter { 0.0f };
            std::atomic<float> fadingOutputMeter { 0.0f };
            if (useDryAsOld)
            {
                const int outChannels = std::min(2, buffer.getNumChannels());
                if (outChannels > 0)
                    juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(0, 0), buffer.getReadPointer(0, 0), numSamples);
                if (outChannels > 1)
                    juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(1, 0), buffer.getReadPointer(1, 0), numSamples);
            }
            else
            {
                // EBR: lifetime managed by RCUReader
                fading->processDoubleToBuffer(buffer, dspCrossfadeDoubleBuffer, analyzerFifo,
                                              fadingInputMeter, fadingOutputMeter, fadingState);
            }
            dsp->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

            const int outChannels = std::min(2, buffer.getNumChannels());
            double* dstL = (outChannels > 0) ? buffer.getWritePointer(0, 0) : nullptr;
            double* dstR = (outChannels > 1) ? buffer.getWritePointer(1, 0) : nullptr;
            const double* oldL = (outChannels > 0) ? dspCrossfadeDoubleBuffer.getReadPointer(0, 0) : nullptr;
            const double* oldR = (outChannels > 1) ? dspCrossfadeDoubleBuffer.getReadPointer(1, 0) : nullptr;

            // 遅延整合バッファを使ったクロスフェード
            const int bufferSize = latencyBufSize;
            const int delayOld = latencyDelayOld_RT;
            const int delayNew = latencyDelayNew_RT;
            double gNew = dspCrossfadeGain.getCurrentValue();
            const double gTarget = dspCrossfadeGain.getTargetValue();
            const double dg = (gTarget - gNew) / numSamples;
            // resetPendingはAudioThreadで1回だけ処理
            if (latencyResetPending.exchange(false, std::memory_order_acq_rel)) {
                if (latencyBufOldL) std::memset(latencyBufOldL, 0, sizeof(double) * bufferSize);
                if (latencyBufOldR) std::memset(latencyBufOldR, 0, sizeof(double) * bufferSize);
                if (latencyBufNewL) std::memset(latencyBufNewL, 0, sizeof(double) * bufferSize);
                if (latencyBufNewR) std::memset(latencyBufNewR, 0, sizeof(double) * bufferSize);
                latencyWritePos = 0;
            }
            for (int i = 0; i < numSamples; ++i)
            {
                latencyBufOldL[latencyWritePos] = (oldL != nullptr) ? oldL[i] : 0.0;
                latencyBufOldR[latencyWritePos] = (oldR != nullptr) ? oldR[i] : 0.0;
                latencyBufNewL[latencyWritePos] = (dstL != nullptr) ? dstL[i] : 0.0;
                latencyBufNewR[latencyWritePos] = (dstR != nullptr) ? dstR[i] : 0.0;

                int readOld = latencyWritePos - delayOld;
                int readNew = latencyWritePos - delayNew;
                // 完全wrap
                while (readOld < 0) readOld += bufferSize;
                while (readOld >= bufferSize) readOld -= bufferSize;
                while (readNew < 0) readNew += bufferSize;
                while (readNew >= bufferSize) readNew -= bufferSize;

                const double alignedOldL = latencyBufOldL[readOld];
                const double alignedOldR = latencyBufOldR[readOld];
                const double alignedNewL = latencyBufNewL[readNew];
                const double alignedNewR = latencyBufNewR[readNew];

                gNew += dg;
                const double gOld = 1.0 - gNew;

                if (dstL) dstL[i] = alignedNewL * gNew + alignedOldL * gOld;
                if (dstR) dstR[i] = alignedNewR * gNew + alignedOldR * gOld;

                latencyWritePos++;
                if (latencyWritePos >= bufferSize)
                    latencyWritePos = 0;
            }
            if (!useDryAsOld)
            {
                // EBR: managed by RCUReader
            }

            if (!dspCrossfadeGain.isSmoothing())
            {
                if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                    retireDSP(done);
                dspCrossfadeGain.setCurrentAndTargetValue(1.0);
                dspCrossfadeDryScaleGain.setCurrentAndTargetValue(1.0);
                dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
            }
            return;
        }

        // --- 通常パス（クロスフェードなし） ---
        dsp->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

        if (fading != nullptr && !dspCrossfadeGain.isSmoothing())
        {
            if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                retireDSP(done);
        }
        if (!dspCrossfadeGain.isSmoothing())
            dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
        return;
    }

    // --- クロスフェード開始時: スナップショット取得・RT競合ゼロ設計 ---
    DSPCore* fading = sanitizeRawPtr(fadingOutDSP.load(std::memory_order_acquire));
    bool useDryAsOld = dspCrossfadeUseDryAsOld.load(std::memory_order_acquire);
    int pendingFadeDelayBlocks = dspCrossfadeStartDelayBlocks.load(std::memory_order_acquire);
    if (fading != nullptr
        && !useDryAsOld
        && dspCrossfadePending.load(std::memory_order_acquire)
        && pendingFadeDelayBlocks > 0)
    {
        dspCrossfadeStartDelayBlocks.store(pendingFadeDelayBlocks - 1, std::memory_order_release);

        auto fadingState = procState;
        fadingState.analyzerEnabled = false;
        fadingState.adaptiveCaptureQueue = nullptr;

        std::atomic<float> fadingInputMeter { 0.0f };
        std::atomic<float> fadingOutputMeter { 0.0f };
        fading->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, fadingState);
        return;
    }

    if ((fading != nullptr || firstIrDryCrossfadePending.load(std::memory_order_acquire))
        && dspCrossfadePending.exchange(false, std::memory_order_acq_rel))
    {
        // queuedFadeTimeSecはcommitNewDSPでセット済み、ここでスナップショット
        const double fadeSec = std::max(0.001, queuedFadeTimeSec.load(std::memory_order_acquire));
        // latencyDelayOld/New, latencyWritePos, latencyBuf*もcommitNewDSPでセット済み
        // AudioThread側は読み取り専用、atomic不要
        dspCrossfadeGain.reset(std::max(1.0, dsp->sampleRate), fadeSec);
        dspCrossfadeGain.setCurrentAndTargetValue(0.0);
        dspCrossfadeGain.setTargetValue(1.0);

        // レイテンシ整合値を Audio Thread スナップショットへ反映する。
        latencyDelayOld_RT = latencyDelayOld.load(std::memory_order_acquire);
        latencyDelayNew_RT = latencyDelayNew.load(std::memory_order_acquire);

        if (firstIrDryCrossfadePending.exchange(false, std::memory_order_acq_rel))
        {
            dspCrossfadeUseDryAsOld.store(true, std::memory_order_release);
            useDryAsOld = true;
        }
    }

    const bool canCrossfade = (fading != nullptr || useDryAsOld)
        && dspCrossfadeGain.isSmoothing()
        && dspCrossfadeDoubleBuffer.getNumChannels() >= 2
        && dspCrossfadeDoubleBuffer.getNumSamples() >= numSamples;

    if (canCrossfade)
    {
        // --- wrap安全・スナップショット設計 ---
        dspCrossfadeDoubleBuffer.clear(0, 0, numSamples);
        dspCrossfadeDoubleBuffer.clear(1, 0, numSamples);

        auto fadingState = procState;
        fadingState.analyzerEnabled = false;
        fadingState.adaptiveCaptureQueue = nullptr;

        std::atomic<float> fadingInputMeter { 0.0f };
        std::atomic<float> fadingOutputMeter { 0.0f };
        if (useDryAsOld)
        {
            const int outChannels = std::min(2, buffer.getNumChannels());
            if (outChannels > 0)
                juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(0, 0), buffer.getReadPointer(0, 0), numSamples);
            if (outChannels > 1)
                juce::FloatVectorOperations::copy(dspCrossfadeDoubleBuffer.getWritePointer(1, 0), buffer.getReadPointer(1, 0), numSamples);
        }
        else
        {
            // EBR: managed by RCUReader
            fading->processDoubleToBuffer(buffer, dspCrossfadeDoubleBuffer, analyzerFifo,
                                          fadingInputMeter, fadingOutputMeter, fadingState);
        }
        dsp->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

        // スナップショット（commitNewDSPでセット済み、ここでは読み取り専用）
        const int outChannels = std::min(2, buffer.getNumChannels());
        double* dstL = (outChannels > 0) ? buffer.getWritePointer(0, 0) : nullptr;
        double* dstR = (outChannels > 1) ? buffer.getWritePointer(1, 0) : nullptr;
        const double* oldL = (outChannels > 0) ? dspCrossfadeDoubleBuffer.getReadPointer(0, 0) : nullptr;
        const double* oldR = (outChannels > 1) ? dspCrossfadeDoubleBuffer.getReadPointer(1, 0) : nullptr;

        const int bufferSize = latencyBufSize;
        int writePos = latencyWritePos;
        // ===== 遅延整合スナップショット取得（AudioThread）=====
        // ===== RT snapshot値を使用 =====
        const int delayOld = latencyDelayOld_RT;
        const int delayNew = latencyDelayNew_RT;
        // ===== resetPending処理（AudioThreadのみ）=====
        if (latencyResetPending.exchange(false, std::memory_order_acq_rel)) {
            if (latencyBufOldL) std::memset(latencyBufOldL, 0, sizeof(double) * bufferSize);
            if (latencyBufOldR) std::memset(latencyBufOldR, 0, sizeof(double) * bufferSize);
            if (latencyBufNewL) std::memset(latencyBufNewL, 0, sizeof(double) * bufferSize);
            if (latencyBufNewR) std::memset(latencyBufNewR, 0, sizeof(double) * bufferSize);
            writePos = 0;
        }
        for (int i = 0; i < numSamples; ++i) {
            latencyBufOldL[writePos] = (oldL != nullptr) ? oldL[i] : 0.0;
            latencyBufOldR[writePos] = (oldR != nullptr) ? oldR[i] : 0.0;
            latencyBufNewL[writePos] = (dstL != nullptr) ? dstL[i] : 0.0;
            latencyBufNewR[writePos] = (dstR != nullptr) ? dstR[i] : 0.0;

            int readOld = writePos - delayOld;
            int readNew = writePos - delayNew;
            // 完全wrap
            while (readOld < 0) readOld += bufferSize;
            while (readOld >= bufferSize) readOld -= bufferSize;
            while (readNew < 0) readNew += bufferSize;
            while (readNew >= bufferSize) readNew -= bufferSize;

            const double alignedOldL = latencyBufOldL[readOld];
            const double alignedOldR = latencyBufOldR[readOld];
            const double alignedNewL = latencyBufNewL[readNew];
            const double alignedNewR = latencyBufNewR[readNew];

            const double gNew = dspCrossfadeGain.getNextValue();
            const double gOld = 1.0 - gNew;

            if (dstL) dstL[i] = alignedNewL * gNew + alignedOldL * gOld;
            if (dstR) dstR[i] = alignedNewR * gNew + alignedOldR * gOld;

            writePos = (writePos + 1);
            if (writePos == bufferSize)
                writePos = 0;
        }
        latencyWritePos = writePos;
        if (!useDryAsOld)
        {
            // EBR: managed by RCUReader
        }

        if (!dspCrossfadeGain.isSmoothing())
        {
            if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                retireDSP(done);
            dspCrossfadeGain.setCurrentAndTargetValue(1.0);
            dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
        }
    }
    else
    {
        dsp->processDouble(buffer, analyzerFifo, inputLevelLinear, outputLevelLinear, procState);

        if (fading != nullptr && !dspCrossfadeGain.isSmoothing())
        {
            if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, std::memory_order_acq_rel)))
                retireDSP(done);
        }
        if (!dspCrossfadeGain.isSmoothing())
            dspCrossfadeUseDryAsOld.store(false, std::memory_order_release);
    }
}

#endif

