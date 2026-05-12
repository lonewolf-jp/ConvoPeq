#include <JuceHeader.h>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"
#include "core/RCUReader.h"

namespace
{
    inline double absDiffNoLibm(double a, double b) noexcept
    {
        return absNoLibm(a - b);
    }
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_AUDIO_BLOCK)

void AudioEngine::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
    constexpr int kAudioEpochReaderIndex = 0;
    const juce::ScopedNoDenormals noDenormals;
    const convo::numeric_policy::ScopedThreadRole audioThreadScope(convo::numeric_policy::ThreadRole::AudioRealtime);
    const convo::EpochCoreReaderGuard epochReaderGuard(m_epochCore, kAudioEpochReaderIndex);
    ASSERT_AUDIO_THREAD();
    m_audioBlockCounter.fetch_add(1, std::memory_order_release);

    // 入力検証 (Input Validation)
    if (bufferToFill.buffer == nullptr)
        return;

    const int numSamples = bufferToFill.numSamples;
    const int startSample = bufferToFill.startSample;
    auto* buffer = bufferToFill.buffer;

    // 事前サニティチェック: 絶対的な上限 (1<<20 ≒ 100万サンプル) で明らかな破損データを弾く。
    // DSPCore の maxSamplesPerBlock は prepareToPlay() でホスト指定値を反映して設定されるため、
    // ここで SAFE_MAX_BLOCK_SIZE (65536) を使うと、131072 等の正当なブロックを誤って拒否する。
    // 【Bug Fix】SAFE_MAX_BLOCK_SIZE による早期リジェクトを廃止し、dsp->maxSamplesPerBlock で
    //            正確なチェックを行う (下記 DSPCore 取得後)。
    constexpr int ABSOLUTE_MAX_BLOCK_SIZE = 1 << 20; // 破損データ検出用上限
    if (numSamples <= 0 || numSamples > ABSOLUTE_MAX_BLOCK_SIZE)
    {
        applySafeSilentFallback(bufferToFill);
        return;
    }

    // startSampleの妥当性チェック
    if (startSample < 0 || startSample + numSamples > buffer->getNumSamples())
    {
        applySafeSilentFallback(bufferToFill);
        return;
    }

    // Epoch tracking for lock-free Audio Thread safety
    convo::RCUReaderGuard rcuGuard(audioThreadRcuReader);

    const auto* runtimeGraph = getRuntimeGraphState();
    DSPCore* dsp = resolveCurrentDSPFromRuntimePublish(runtimeGraph);
    if (dsp == nullptr)
    {
        applySafeSilentFallback(bufferToFill);
        return;
    }

    if (dsp != nullptr)
    {
        // DSPCore 固有の上限チェック
        // DSPCore::prepare() でホスト指定の samplesPerBlock を反映した maxSamplesPerBlock が設定される。
        // dsp は RCU で公開済みのため maxSamplesPerBlock は Audio Thread から安全に読み出せる。
        if (numSamples > dsp->maxSamplesPerBlock)
        {
            applySafeSilentFallback(bufferToFill);
            return;
        }

        // 安全対策: サンプルレート不整合チェック
        // DSPのサンプルレートとエンジンの現在のサンプルレートが一致しない場合、
        // レート変更処理中とみなし、グリッチを防ぐために無音を出力する。
        const double engineSampleRate = currentSampleRate.load(std::memory_order_relaxed);
        if (absDiffNoLibm(dsp->sampleRate, engineSampleRate) > 1e-6)
        {
            // 不整合時はレベルメーターもリセットして誤表示を防ぐ
            inputLevelLinear.store(0.0f);
            outputLevelLinear.store(0.0f);
            applySafeSilentFallback(bufferToFill);
            return;
        }

        // パラメータのロード
        // 【Parameter安全設計】
        // Audio ThreadではAtomic変数の読み取りのみを行い、ロックやメモリ確保を伴う処理は行わない。
        // 構造変更が必要な場合は、別途フラグやUIスレッド経由で再構築を行う。
        // ── Audio Thread 最適化: GlobalSnapshot を優先し、fallback で atomics を読む ──
        const convo::GlobalSnapshot* snap = m_coordinator.getCurrent();
        const EngineParameterSnapshot parameterSnapshot = captureAudioThreadParameterSnapshot(snap);

        // UI表示用: 比較なしで直接ストア（ロード→比較→ストアより高速）
        eqBypassActive.store(parameterSnapshot.eqBypassed, std::memory_order_relaxed);
        convBypassActive.store(parameterSnapshot.convBypassed, std::memory_order_relaxed);
        DSPCore::ProcessingState procState = buildAudioThreadProcessingState(dsp, parameterSnapshot);

        float snapshotAlpha = 1.0f;
        const convo::GlobalSnapshot* snapshotFrom = nullptr;
        const convo::GlobalSnapshot* snapshotTo = nullptr;
        const bool updateFadeReturned = updateAudioThreadSnapshotFade(numSamples,
                                                                      snapshotAlpha,
                                                                      snapshotFrom,
                                                                      snapshotTo);

        const bool snapshotFading = updateFadeReturned
            && snapshotTo != nullptr;

        if (snapshotFading)
        {
            const int fadeChannels = std::min(dspCrossfadeFloatBuffer.getNumChannels(), buffer->getNumChannels());
            for (int ch = 0; ch < fadeChannels; ++ch)
                dspCrossfadeFloatBuffer.clear(ch, 0, numSamples);

            juce::AudioSourceChannelInfo oldInfo(&dspCrossfadeFloatBuffer, 0, numSamples);
            processWithSnapshot(oldInfo, snapshotFrom, true);
            processWithSnapshot(bufferToFill, snapshotTo, false);

            const float gNew = snapshotAlpha;
            const float gOld = 1.0f - snapshotAlpha;
            const int outChannels = std::min(buffer->getNumChannels(), dspCrossfadeFloatBuffer.getNumChannels());
            float* dstL = (outChannels > 0) ? buffer->getWritePointer(0, startSample) : nullptr;
            float* dstR = (outChannels > 1) ? buffer->getWritePointer(1, startSample) : nullptr;
            const float* oldL = (outChannels > 0) ? dspCrossfadeFloatBuffer.getReadPointer(0, 0) : nullptr;
            const float* oldR = (outChannels > 1) ? dspCrossfadeFloatBuffer.getReadPointer(1, 0) : nullptr;

            for (int i = 0; i < numSamples; ++i)
            {
                if (dstL != nullptr)
                    dstL[i] = dstL[i] * gNew + oldL[i] * gOld;
                if (dstR != nullptr)
                    dstR[i] = dstR[i] * gNew + oldR[i] * gOld;
            }

            return;
        }

        DSPCore* fading = resolveFadingDSPFromRuntimePublish(runtimeGraph);
        const auto* engineRuntime = getEngineRuntimeState();
        const bool atomicUseDryAsOld = dspCrossfadeUseDryAsOld.load(std::memory_order_acquire);
        bool useDryAsOld = atomicUseDryAsOld
            || runtimeCrossfadeUseDryAsOld(engineRuntime, runtimeGraph);
        const bool atomicPendingCrossfade = dspCrossfadePending.load(std::memory_order_acquire);
        const bool hasPendingCrossfade = atomicPendingCrossfade
            || runtimeCrossfadePending(engineRuntime, runtimeGraph);
        const int pendingFadeDelayBlocks = dspCrossfadeStartDelayBlocks.load(std::memory_order_acquire);
        if (processCrossfadeDelayGateIfPending(fading,
                                               useDryAsOld,
                                               hasPendingCrossfade,
                                               pendingFadeDelayBlocks,
                                               [&]()
        {
            auto fadingState = makeCrossfadeAuxState(procState);
            syncEqAgcTableViewFromRuntimeGraph(dspExecutionStateFading, runtimeGraph);

            std::atomic<float> fadingInputMeter { 0.0f };
            std::atomic<float> fadingOutputMeter { 0.0f };
            fading->processV2(bufferToFill,
                              analyzerFifo,
                              inputLevelLinear,
                              outputLevelLinear,
                              runtimeGraph,
                              dspExecutionStateFading,
                              fadingState);
        }))
        {
            return;
        }

        armCrossfadeIfPending(dsp, fading != nullptr, useDryAsOld, runtimeGraph);

        const bool canCrossfade = (fading != nullptr || useDryAsOld)
            && dspCrossfadeGain.isSmoothing()
            && dspCrossfadeFloatBuffer.getNumChannels() >= 2
            && dspCrossfadeFloatBuffer.getNumSamples() >= numSamples;

        if (canCrossfade)
        {
            juce::AudioSourceChannelInfo fadeInfo(&dspCrossfadeFloatBuffer, 0, numSamples);
            dspCrossfadeFloatBuffer.clear(0, 0, numSamples);
            dspCrossfadeFloatBuffer.clear(1, 0, numSamples);

            auto fadingState = makeCrossfadeAuxState(procState);

            std::atomic<float> fadingInputMeter { 0.0f };
            std::atomic<float> fadingOutputMeter { 0.0f };
            if (useDryAsOld)
            {
                const int outChannels = std::min(2, buffer->getNumChannels());
                if (outChannels > 0)
                    juce::FloatVectorOperations::copy(dspCrossfadeFloatBuffer.getWritePointer(0, 0), buffer->getReadPointer(0, startSample), numSamples);
                if (outChannels > 1)
                    juce::FloatVectorOperations::copy(dspCrossfadeFloatBuffer.getWritePointer(1, 0), buffer->getReadPointer(1, startSample), numSamples);
            }
            else
            {
                // EBR: lifetime managed by RCUReader
                syncEqAgcTableViewFromRuntimeGraph(dspExecutionStateFading, runtimeGraph);
                fading->processToBuffer(bufferToFill, dspCrossfadeFloatBuffer, analyzerFifo,
                                       fadingInputMeter, fadingOutputMeter, fadingState);
            }
            syncEqAgcTableViewFromRuntimeGraph(dspExecutionStateCurrent, runtimeGraph);
            dsp->processV2(bufferToFill,
                           analyzerFifo,
                           inputLevelLinear,
                           outputLevelLinear,
                           runtimeGraph,
                           dspExecutionStateCurrent,
                           procState);

            const int outChannels = std::min(2, buffer->getNumChannels());
            float* dstL = (outChannels > 0) ? buffer->getWritePointer(0, startSample) : nullptr;
            float* dstR = (outChannels > 1) ? buffer->getWritePointer(1, startSample) : nullptr;
            const float* oldL = (outChannels > 0) ? dspCrossfadeFloatBuffer.getReadPointer(0, 0) : nullptr;
            const float* oldR = (outChannels > 1) ? dspCrossfadeFloatBuffer.getReadPointer(1, 0) : nullptr;

            runLatencyAlignedCrossfadeMixLoop<float>(dstL,
                                                     dstR,
                                                     oldL,
                                                     oldR,
                                                     numSamples,
                                                     runtimeGraph,
                                                     [this, useDryAsOld](float* outL,
                                                                         float* outR,
                                                                         int i,
                                                                         double gNew,
                                                                         double alignedOldL,
                                                                         double alignedOldR,
                                                                         double alignedNewL,
                                                                         double alignedNewR)
                                                     {
                                                         const double dryScale = useDryAsOld ? dspCrossfadeDryScaleGain.getNextValue() : 1.0;
                                                         const double gOld = 1.0 - gNew;
                                                         const double dryScaledL = alignedOldL * dryScale;
                                                         const double dryScaledR = alignedOldR * dryScale;
                                                         if (outL != nullptr)
                                                             outL[i] = static_cast<float>(alignedNewL * gNew + dryScaledL * gOld);
                                                         if (outR != nullptr)
                                                             outR[i] = static_cast<float>(alignedNewR * gNew + dryScaledR * gOld);
                                                     });

            if (!useDryAsOld)
            {
                // EBR: fading lifetime managed by RCUReaderGuard
            }

            finalizeCrossfadeMixPath(true);
        }
        else
        {
            // 通常パス（クロスフェードなし）：RCU で dsp の生存が保証されるため addRef/release 不要
            dsp->processV2(bufferToFill,
                           analyzerFifo,
                           inputLevelLinear,
                           outputLevelLinear,
                           runtimeGraph,
                           dspExecutionStateCurrent,
                           procState);
            cleanupCrossfadeDirectPath(fading);
        }
    }

}

#endif
