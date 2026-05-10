#include <JuceHeader.h>
#include "AudioEngine.h"

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
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_DSP_PREPARE)

AudioEngine::DSPCore::DSPCore() = default;

void AudioEngine::DSPCore::prepare(double newSampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType, NoiseShaperType selectedNoiseShaperType, AudioEngine* owner)
{
    this->sampleRate = newSampleRate;
    this->noiseShaperType = selectedNoiseShaperType;
    this->ownerEngine = owner;
    convolver.setRcuProvider(ownerEngine);

    int targetFactor = 1;
    if (manualOversamplingFactor > 0)
    {
        targetFactor = manualOversamplingFactor;
    }
    else
    {
        // 自動設定 (デフォルト)
        if (newSampleRate >= 705600)
            targetFactor = 1;
        else if (newSampleRate >= 352800)
            targetFactor =  2;
        else if (newSampleRate >= 176400)
            targetFactor =  4;
        else if (newSampleRate >= 88200)
            targetFactor = 8;
         else
             targetFactor = 8;
    }

    // 制限: サンプルレートに応じた最大倍率を適用
    int maxFactor = 1;
    if (newSampleRate <= 96000.0)       maxFactor = 8;
    else if (newSampleRate <= 192000.0) maxFactor = 4;
    else if (newSampleRate <= 384000.0) maxFactor = 2;

    targetFactor = std::min(targetFactor, maxFactor);

    size_t factorLog2 = 0;
    if (targetFactor >= 8)      factorLog2 = 3;
    else if (targetFactor >= 4) factorLog2 = 2;
    else if (targetFactor >= 2) factorLog2 = 1;
    else                        factorLog2 = 0;

    oversamplingFactor = (size_t)1 << factorLog2;

    // ==================================================================
    // 【Issue 3 完全修正】内部最大バッファサイズの計算（推奨A）
    // 固定で SAFE_MAX_BLOCK_SIZE × 8 を確保
    // 理由:
    //   ・OS=8x時のupBlockサイズを完全にカバー
    //   ・RCU再構築（IRロード・プリセット切替・OS変更）ごとにresizeしない
    //   ・MKLAllocator + 64byteアライメントの最適化が最大限活きる
    //   ・将来16x OS対応もこの定数1箇所変更だけで済む
    // ==================================================================
    constexpr int MAX_OS_FACTOR = 8;
    // [FIX] Ensure we cover the requested block size even if it exceeds SAFE_MAX_BLOCK_SIZE
    const int inputMaxBlock     = std::max(SAFE_MAX_BLOCK_SIZE, samplesPerBlock);
    const int internalMaxBlock  = inputMaxBlock * MAX_OS_FACTOR;

    maxSamplesPerBlock   = inputMaxBlock;
    maxInternalBlockSize = internalMaxBlock;

// === 【パッチ3】raw aligned_malloc確保（message threadのみ・64byte保証）===
    const int newRequired = internalMaxBlock;
    if (newRequired > alignedCapacity || !alignedL || !alignedR)
    {
        // Exception-safe allocation using local ScopedAlignedPtr
        auto newL = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(newRequired) * sizeof(double), 64)));
        auto newR = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(newRequired) * sizeof(double), 64)));

        // 明示的ゼロクリア（Denormal/NaN防止）
        juce::FloatVectorOperations::clear(newL.get(), newRequired);
        juce::FloatVectorOperations::clear(newR.get(), newRequired);

        // Commit (noexcept move)
        alignedL = std::move(newL);
        alignedR = std::move(newR);
        alignedCapacity = newRequired;
    }

    if (newRequired > dryBypassCapacityDouble || !dryBypassBufferDoubleL || !dryBypassBufferDoubleR)
    {
        auto newDryL = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(newRequired) * sizeof(double), 64)));
        auto newDryR = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(newRequired) * sizeof(double), 64)));
        juce::FloatVectorOperations::clear(newDryL.get(), newRequired);
        juce::FloatVectorOperations::clear(newDryR.get(), newRequired);
        dryBypassBufferDoubleL = std::move(newDryL);
        dryBypassBufferDoubleR = std::move(newDryR);
        dryBypassCapacityDouble = newRequired;
    }

    bypassFadeGainDouble.reset(newSampleRate, 0.005);
    bypassFadeGainDouble.setCurrentAndTargetValue(1.0);
    bypassedDouble = false;

    const auto osPreset = (oversamplingType == OversamplingType::LinearPhase)
                        ? CustomInputOversampler::Preset::LinearPhase
                        : CustomInputOversampler::Preset::IIRLike;
    oversampling.prepare(inputMaxBlock, static_cast<int>(oversamplingFactor), osPreset);

    const double processingRate = newSampleRate * static_cast<double>(oversamplingFactor);
    const int processingBlockSize = samplesPerBlock * static_cast<int>(oversamplingFactor);

    // プロセッサの準備
    // Convolverには実際のブロックサイズを渡す (パーティションサイズ決定やLoaderThreadで使用)
    convolver.prepareToPlay(processingRate, processingBlockSize);

    // EQも内部最大サイズで準備（より安全）
    eq.prepareToPlay(processingRate, internalMaxBlock);

    // 出力段(processOutput)で実行されるため、オーバーサンプリング前のレートとサイズを使用する
    // 【最適化】UltraHighRateDCBlocker の init() は sampleRate + cutoffHz を受け取る
    dcBlockerL.init(newSampleRate, 3.0);
    dcBlockerR.init(newSampleRate, 3.0);

    // 入力段用DCBlockerの準備
    inputDCBlockerL.init(newSampleRate, 3.0);
    inputDCBlockerR.init(newSampleRate, 3.0);

    // オーバーサンプリング後のDC除去用 (1Hzカットオフ)
    osDCBlockerL.init(processingRate, 1.0);
    osDCBlockerR.init(processingRate, 1.0);

    // ノイズシェーパーの準備 (出力段で行うため元のサンプルレート)
    if (selectedNoiseShaperType == NoiseShaperType::Psychoacoustic)
        dither.prepare(newSampleRate, bitDepth);
    else if (selectedNoiseShaperType == NoiseShaperType::Fixed4Tap)
    {
        fixedNoiseShaper.setCoefficients(kFixedNoiseShaperTunedCoeffs);
        fixedNoiseShaper.prepare(newSampleRate, bitDepth);
    }
    else if (selectedNoiseShaperType == NoiseShaperType::Fixed15Tap)
    {
        fixed15TapNoiseShaper.setCoefficients(kFixed15TapNoiseShaperTunedCoeffs);
        fixed15TapNoiseShaper.prepare(newSampleRate, bitDepth);
    }
    else
    {
        adaptiveNoiseShaper.prepare(bitDepth);
        adaptiveNoiseShaper.setCoefficients(kDefaultAdaptiveNoiseShaperCoeffs.data(), kAdaptiveNoiseShaperOrder);
        activeAdaptiveCoeffGeneration = 0;
        activeAdaptiveCoeffBankIndex = -1;
    }
    this->ditherBitDepth = bitDepth; // DSPCoreのメンバーに保存

    // 出力周波数フィルターの係数を事前計算 (processingRate: OS後のレート)
    // filter.txt: ハイカット/ローカット(①) / ローパス/ハイパス(②) の全モード分を一括生成
    outputFilter.prepare(processingRate);

    // 【Issue 5】Fade-inカウンタをリセット
    fadeInSamplesLeft.store(0, std::memory_order_relaxed);

    // 初期状態は固定レイテンシなし
    setFixedLatencySamples(0);
}

void AudioEngine::DSPCore::setFixedLatencySamples(int samples)
{
    const int clamped = std::max(0, samples);
    fixedLatencySamples = clamped;
    fixedLatencyWritePos = 0;

    const int requiredSize = clamped + std::max(1, maxInternalBlockSize) + 2;
    if (requiredSize > fixedLatencyBufferSize || !fixedLatencyBufferL || !fixedLatencyBufferR)
    {
        auto newDelayL = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(requiredSize) * sizeof(double), 64)));
        auto newDelayR = convo::ScopedAlignedPtr<double>(static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(requiredSize) * sizeof(double), 64)));

        juce::FloatVectorOperations::clear(newDelayL.get(), requiredSize);
        juce::FloatVectorOperations::clear(newDelayR.get(), requiredSize);

        fixedLatencyBufferL = std::move(newDelayL);
        fixedLatencyBufferR = std::move(newDelayR);
        fixedLatencyBufferSize = requiredSize;
    }
    else if (fixedLatencyBufferSize > 0)
    {
        juce::FloatVectorOperations::clear(fixedLatencyBufferL.get(), fixedLatencyBufferSize);
        juce::FloatVectorOperations::clear(fixedLatencyBufferR.get(), fixedLatencyBufferSize);
    }
}

void AudioEngine::DSPCore::reset()
{
    convolver.reset();
    eq.reset();
    dcBlockerL.reset();
    dcBlockerR.reset();
    inputDCBlockerL.reset();
    inputDCBlockerR.reset();
    osDCBlockerL.reset();
    osDCBlockerR.reset();
    dither.reset();
    fixedNoiseShaper.reset();
    adaptiveNoiseShaper.reset();
    oversampling.reset();
    outputFilter.reset();
    activeAdaptiveCoeffGeneration = 0;
    activeAdaptiveCoeffBankIndex = -1;

    // 【パッチ3】rawバッファクリア（alignedCapacity使用）
    if (alignedL && alignedCapacity > 0)
        juce::FloatVectorOperations::clear(alignedL.get(), alignedCapacity);
    if (alignedR && alignedCapacity > 0)
        juce::FloatVectorOperations::clear(alignedR.get(), alignedCapacity);
    if (dryBypassBufferDoubleL && dryBypassCapacityDouble > 0)
        juce::FloatVectorOperations::clear(dryBypassBufferDoubleL.get(), dryBypassCapacityDouble);
    if (dryBypassBufferDoubleR && dryBypassCapacityDouble > 0)
        juce::FloatVectorOperations::clear(dryBypassBufferDoubleR.get(), dryBypassCapacityDouble);

    bypassFadeGainDouble.setCurrentAndTargetValue(1.0);
    bypassedDouble = false;

    fixedLatencyWritePos = 0;
    if (fixedLatencyBufferL && fixedLatencyBufferSize > 0)
        juce::FloatVectorOperations::clear(fixedLatencyBufferL.get(), fixedLatencyBufferSize);
    if (fixedLatencyBufferR && fixedLatencyBufferSize > 0)
        juce::FloatVectorOperations::clear(fixedLatencyBufferR.get(), fixedLatencyBufferSize);

    // インターサンプルピーク用ブロック間状態をリセット
    softClipPrevSample[0] = 0.0;
    softClipPrevSample[1] = 0.0;
}

#endif
