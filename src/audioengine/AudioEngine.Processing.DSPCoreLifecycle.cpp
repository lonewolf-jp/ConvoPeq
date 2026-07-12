#include <JuceHeader.h>
#include "AudioEngine.h"
#include "DiagnosticsConfig.h"

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
namespace {
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
} // namespace
#endif

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

    // ★ 2026-06-23: P7(Ne10型/現行)向けに再最適化
    // 元値: 0.82, -0.68, 0.55, -0.43, 0.33, -0.25, 0.18, -0.12, 0.07 (Pattern A向け)
    // 新値: CMA-ES学習済み係数(192kHz/32bit mode4/5)の平均
    // 検証: 全ビット深度(16/24/32)でDC driftなし、NTF一致確認済み
    static constexpr std::array<double, kAdaptiveNoiseShaperOrder> kDefaultAdaptiveNoiseShaperCoeffs
    {
        -0.003796, -0.006752, 0.008418, -0.010546, 0.004716, -0.007624, -0.020750, -0.002049, -0.003632
    };
}

std::atomic<std::uint64_t> AudioEngine::DSPCore::runtimeUuidCounterStorage_{ 1 };

std::atomic<std::uint64_t>& AudioEngine::DSPCore::runtimeUuidCounter() noexcept
{
    return runtimeUuidCounterStorage_;
}

// ★ work70: DSPCore::liveCount 静的定義
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
std::atomic<uint32_t> AudioEngine::DSPCore::liveCount { 0 };
#endif

[[nodiscard]] std::uint64_t AudioEngine::DSPCore::reserveNextRuntimeUuid() noexcept
{
    return convo::fetchAddAtomic(runtimeUuidCounter(),
                                 static_cast<std::uint64_t>(1),
                                 std::memory_order_acq_rel);
}

AudioEngine::DSPCore::DSPCore()
    : dcBlockerState(new DCBlockerRuntimeState())
    , convolverState(new ConvolverRuntimeState())
    , eqState(new EQRuntimeState())
    , rampState(new RampRuntimeState())
    , historyState(new HistoryRuntimeState())
    , runtimeUuid(reserveNextRuntimeUuid())
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    liveCount.fetch_add(1, std::memory_order_relaxed);
#endif
    convolverState->bind(convolver);
    eqState->bind(eq);
}

void AudioEngine::DSPCore::prepare(double newSampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType, NoiseShaperType selectedNoiseShaperType, AudioEngine* owner)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const double prepareStartMs = juce::Time::getMillisecondCounterHiRes();
    static const auto elapsedSince = [](double fromMs) -> double {
        return juce::Time::getMillisecondCounterHiRes() - fromMs;
    };
    diagLog("[DSPCORE_PREPARE] enter");
#else
    juce::Logger::writeToLog("[DSPCORE_PREPARE] enter");
#endif

    // Route EQ and Convolver retirement through AudioEngine's coordinator
    if (owner != nullptr)
    {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        double t0 = juce::Time::getMillisecondCounterHiRes();
        diagLog("[DSPCORE_PREPARE] calling setRetireCoordinator");
        eq.setRetireCoordinator(&owner->runtimePublicationBridge_);
        convolver.setRetireCoordinator(&owner->runtimePublicationBridge_);
        diagLog("[DSPCORE_PREPARE] setRetireCoordinator done: " + juce::String(elapsedSince(t0), 2) + "ms");
#else
        juce::Logger::writeToLog("[DSPCORE_PREPARE] setRetireCoordinator");
        eq.setRetireCoordinator(&owner->runtimePublicationBridge_);
        convolver.setRetireCoordinator(&owner->runtimePublicationBridge_);
        juce::Logger::writeToLog("[DSPCORE_PREPARE] setRetireCoordinator done");
#endif
    }

    // Exception-safety context:
    // - prepare() は Message/Worker 側の non-RT 経路で実行される。
    // - list.md の RT 例外禁止(2.3)・RT allocation禁止(2.1)の対象外。
    // - ここでの確保は RAII で commit 前に完了させ、Audio Thread へは完成状態のみ publish する。
    this->sampleRate = newSampleRate;
    this->noiseShaperType = selectedNoiseShaperType;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    diagLog("[DSPCORE_PREPARE] sampleRate set: sr=" + juce::String(newSampleRate) + " spb=" + juce::String(samplesPerBlock));
#else
    juce::Logger::writeToLog("[DSPCORE_PREPARE] sampleRate set: sr=" + juce::String(newSampleRate) + " spb=" + juce::String(samplesPerBlock));
#endif

    int targetFactor = 1;
    if (manualOversamplingFactor > 0)
        targetFactor = manualOversamplingFactor;
    else
    {
        if (newSampleRate >= 705600)      targetFactor = 1;
        else if (newSampleRate >= 352800) targetFactor = 2;
        else if (newSampleRate >= 176400) targetFactor = 4;
        else if (newSampleRate >= 88200)  targetFactor = 8;
        else                              targetFactor = 8;
    }

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
    activeOversamplingType = oversamplingType;

    constexpr int MAX_OS_FACTOR = 8;
    // ★ v8.3: PrepareBlockSizingPolicy を使用（SAFE_MAX_BLOCK_SIZE floor を廃止）
    const int inputMaxBlock     = AudioEngine::PrepareBlockSizingPolicy::apply(samplesPerBlock);
    const int internalMaxBlock  = inputMaxBlock * MAX_OS_FACTOR;
    maxSamplesPerBlock   = inputMaxBlock;
    maxInternalBlockSize = internalMaxBlock;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    diagLog("[DSPCORE_PREPARE] blockCalc: inputMaxBlock=" + juce::String(inputMaxBlock) + " internalMaxBlock=" + juce::String(internalMaxBlock));
    const int newRequired = internalMaxBlock;
    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] allocating aligned buffers: required=" + juce::String(internalMaxBlock));
#else
    juce::Logger::writeToLog("[DSPCORE_PREPARE] blockCalc: inputMaxBlock=" + juce::String(inputMaxBlock) + " internalMaxBlock=" + juce::String(internalMaxBlock));
    const int newRequired = internalMaxBlock;
    juce::Logger::writeToLog("[DSPCORE_PREPARE] allocating aligned buffers: required=" + juce::String(internalMaxBlock));
#endif

    if (newRequired > alignedCapacity || !alignedL || !alignedR)
    {
        auto newL = convo::makeAlignedArray<double>(static_cast<size_t>(newRequired));
        auto newR = convo::makeAlignedArray<double>(static_cast<size_t>(newRequired));
        juce::FloatVectorOperations::clear(newL.get(), newRequired);
        juce::FloatVectorOperations::clear(newR.get(), newRequired);
        alignedL = std::move(newL);
        alignedR = std::move(newR);
        alignedCapacity = newRequired;
    }

    if (newRequired > dryBypassCapacityDouble || !dryBypassBufferDoubleL || !dryBypassBufferDoubleR)
    {
        auto newDryL = convo::makeAlignedArray<double>(static_cast<size_t>(newRequired));
        auto newDryR = convo::makeAlignedArray<double>(static_cast<size_t>(newRequired));
        juce::FloatVectorOperations::clear(newDryL.get(), newRequired);
        juce::FloatVectorOperations::clear(newDryR.get(), newRequired);
        dryBypassBufferDoubleL = std::move(newDryL);
        dryBypassBufferDoubleR = std::move(newDryR);
        dryBypassCapacityDouble = newRequired;
    }

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    diagLog("[DSPCORE_PREPARE] aligned buffers done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    auto& ramp = ramps();

    // 11 timing pairs
    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling ramp.prepare");
    ramp.prepare(newSampleRate);
    diagLog("[DSPCORE_PREPARE] ramp.prepare done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling oversampling.prepare");
    oversampling.prepare(inputMaxBlock, static_cast<int>(oversamplingFactor),
        (oversamplingType == OversamplingType::LinearPhase) ? CustomInputOversampler::Preset::LinearPhase : CustomInputOversampler::Preset::IIRLike);
    diagLog("[DSPCORE_PREPARE] oversampling.prepare done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling softClipOS.prepareSingleStage");
    softClipOS.prepareSingleStage(31, 90.0, internalMaxBlock);
    diagLog("[DSPCORE_PREPARE] softClipOS.prepareSingleStage done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    const double processingRate = newSampleRate * static_cast<double>(oversamplingFactor);
    const int processingBlockSize = samplesPerBlock * static_cast<int>(oversamplingFactor);
    diagLog("[DSPCORE_PREPARE] processingRate=" + juce::String(processingRate) + " processingBlockSize=" + juce::String(processingBlockSize));

    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling convolverState->prepare");
    convolverState->prepare(owner, processingRate, processingBlockSize);
    diagLog("[DSPCORE_PREPARE] convolverState->prepare done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling eqState->prepare");
    eqState->prepare(processingRate, internalMaxBlock);
    diagLog("[DSPCORE_PREPARE] eqState->prepare done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling dcBlockers().init");
    dcBlockers().init(newSampleRate, processingRate);
    diagLog("[DSPCORE_PREPARE] dcBlockers().init done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling noise shaper prepare: type=" + juce::String(static_cast<int>(selectedNoiseShaperType)));
    if (selectedNoiseShaperType == NoiseShaperType::Psychoacoustic) dither.prepare(newSampleRate, bitDepth);
    else if (selectedNoiseShaperType == NoiseShaperType::Fixed4Tap) { fixedNoiseShaper.setCoefficients(kFixedNoiseShaperTunedCoeffs); fixedNoiseShaper.prepare(newSampleRate, bitDepth); }
    else if (selectedNoiseShaperType == NoiseShaperType::Fixed15Tap) { fixed15TapNoiseShaper.setCoefficients(kFixed15TapNoiseShaperTunedCoeffs); fixed15TapNoiseShaper.prepare(newSampleRate, bitDepth); }
    else { adaptiveNoiseShaper.prepare(bitDepth); adaptiveNoiseShaper.setCoefficients(kDefaultAdaptiveNoiseShaperCoeffs.data(), kAdaptiveNoiseShaperOrder); activeAdaptiveCoeffGeneration = 0; activeAdaptiveCoeffBankIndex = -1; }
    this->ditherBitDepth = bitDepth;
    diagLog("[DSPCORE_PREPARE] noise shaper prepare done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling outputFilter.prepare");
    outputFilter.prepare(processingRate);
    diagLog("[DSPCORE_PREPARE] outputFilter.prepare done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling truePeakDetector.prepare");
    truePeakDetector.prepare(newSampleRate, maxInternalBlockSize);
    diagLog("[DSPCORE_PREPARE] truePeakDetector.prepare done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling loudnessMeter.prepare");
    loudnessMeter.prepare(newSampleRate, maxInternalBlockSize);
    diagLog("[DSPCORE_PREPARE] loudnessMeter.prepare done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling peakLimiter.prepare");
    peakLimiter.prepare(newSampleRate, 100.0); // Release 100ms
    diagLog("[DSPCORE_PREPARE] peakLimiter.prepare done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    { double t0 = juce::Time::getMillisecondCounterHiRes(); diagLog("[DSPCORE_PREPARE] calling setFixedLatencySamples");
    setFixedLatencySamples(0);
    diagLog("[DSPCORE_PREPARE] setFixedLatencySamples done: " + juce::String(elapsedSince(t0), 2) + "ms"); }

    diagLog("[DSPCORE_PREPARE] total: " + juce::String(elapsedSince(prepareStartMs), 2) + "ms");
#else
    auto& ramp = ramps();
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling ramp.prepare");
    ramp.prepare(newSampleRate);
    juce::Logger::writeToLog("[DSPCORE_PREPARE] ramp.prepare done");
    const auto osPreset = (oversamplingType == OversamplingType::LinearPhase) ? CustomInputOversampler::Preset::LinearPhase : CustomInputOversampler::Preset::IIRLike;
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling oversampling.prepare");
    oversampling.prepare(inputMaxBlock, static_cast<int>(oversamplingFactor), osPreset);
    juce::Logger::writeToLog("[DSPCORE_PREPARE] oversampling.prepare done");
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling softClipOS.prepareSingleStage");
    softClipOS.prepareSingleStage(31, 90.0, internalMaxBlock);
    juce::Logger::writeToLog("[DSPCORE_PREPARE] softClipOS.prepareSingleStage done");
    const double processingRate = newSampleRate * static_cast<double>(oversamplingFactor);
    const int processingBlockSize = samplesPerBlock * static_cast<int>(oversamplingFactor);
    juce::Logger::writeToLog("[DSPCORE_PREPARE] processingRate=" + juce::String(processingRate) + " processingBlockSize=" + juce::String(processingBlockSize));
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling convolverState->prepare");
    convolverState->prepare(owner, processingRate, processingBlockSize);
    juce::Logger::writeToLog("[DSPCORE_PREPARE] convolverState->prepare done");
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling eqState->prepare");
    eqState->prepare(processingRate, internalMaxBlock);
    juce::Logger::writeToLog("[DSPCORE_PREPARE] eqState->prepare done");
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling dcBlockers().init");
    dcBlockers().init(newSampleRate, processingRate);
    juce::Logger::writeToLog("[DSPCORE_PREPARE] dcBlockers().init done");
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling noise shaper prepare: type=" + juce::String(static_cast<int>(selectedNoiseShaperType)));
    if (selectedNoiseShaperType == NoiseShaperType::Psychoacoustic) dither.prepare(newSampleRate, bitDepth);
    else if (selectedNoiseShaperType == NoiseShaperType::Fixed4Tap) { fixedNoiseShaper.setCoefficients(kFixedNoiseShaperTunedCoeffs); fixedNoiseShaper.prepare(newSampleRate, bitDepth); }
    else if (selectedNoiseShaperType == NoiseShaperType::Fixed15Tap) { fixed15TapNoiseShaper.setCoefficients(kFixed15TapNoiseShaperTunedCoeffs); fixed15TapNoiseShaper.prepare(newSampleRate, bitDepth); }
    else { adaptiveNoiseShaper.prepare(bitDepth); adaptiveNoiseShaper.setCoefficients(kDefaultAdaptiveNoiseShaperCoeffs.data(), kAdaptiveNoiseShaperOrder); activeAdaptiveCoeffGeneration = 0; activeAdaptiveCoeffBankIndex = -1; }
    this->ditherBitDepth = bitDepth;
    juce::Logger::writeToLog("[DSPCORE_PREPARE] noise shaper prepare done");
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling outputFilter.prepare");
    outputFilter.prepare(processingRate);
    juce::Logger::writeToLog("[DSPCORE_PREPARE] outputFilter.prepare done");
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling truePeakDetector.prepare");
    truePeakDetector.prepare(newSampleRate, maxInternalBlockSize);
    juce::Logger::writeToLog("[DSPCORE_PREPARE] truePeakDetector.prepare done");
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling loudnessMeter.prepare");
    loudnessMeter.prepare(newSampleRate, maxInternalBlockSize);
    juce::Logger::writeToLog("[DSPCORE_PREPARE] loudnessMeter.prepare done");
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling peakLimiter.prepare");
    peakLimiter.prepare(newSampleRate, 100.0); // Release 100ms
    juce::Logger::writeToLog("[DSPCORE_PREPARE] peakLimiter.prepare done");
    juce::Logger::writeToLog("[DSPCORE_PREPARE] calling setFixedLatencySamples");
    setFixedLatencySamples(0);
    juce::Logger::writeToLog("[DSPCORE_PREPARE] setFixedLatencySamples done");
#endif
}

void AudioEngine::DSPCore::setFixedLatencySamples(int samples)
{
    histories().configureFixedLatencySamples(samples, maxInternalBlockSize);
}

// ★ v8.3: TrackedMemoryStatistics 収集 — NonRT 専用
AudioEngine::DSPCore::TrackedMemoryStatistics
AudioEngine::DSPCore::collectTrackedMemoryStatistics() const noexcept
{
    ASSERT_NON_RT_THREAD();

    TrackedMemoryStatistics stats {};

    // Oversampling work buffers (stages[3] × upHistory/downHistory × 2ch)
    // Estimate: maxInternalBlockSize × upsampleRatio × sizeof(double) × stages
    //   (保守的に maxInputBlock × 8 × 8B × 2ch × 3stages = large, but practical)
    const size_t osFactor = oversamplingFactor;
    stats.oversampling = static_cast<size_t>(maxSamplesPerBlock) * osFactor * sizeof(double) * 2;

    // SoftClip OS (single stage)
    stats.softClip = static_cast<size_t>(maxInternalBlockSize) * sizeof(double) * 2;

    // EQ scratch/dry/parallel buffers (estimated from internalBlockSize × 2ch)
    stats.eqProcessor = static_cast<size_t>(maxInternalBlockSize) * sizeof(double) * 4;

    // alignedL/R + dryBypassDoubleL/R
    stats.alignedBuffers = static_cast<size_t>(alignedCapacity) * sizeof(double) * 2
                         + static_cast<size_t>(dryBypassCapacityDouble) * sizeof(double) * 2;

    // fixedLatency × 2 (old + new) × 2ch
    stats.latencyBuffers = static_cast<size_t>(histories().fixedLatencyBufferSize) * sizeof(double) * 4;

    // TruePeakDetector internal (estimated: internalBlockSize × 2ch × filter order)
    stats.truePeakDetector = static_cast<size_t>(maxInternalBlockSize) * sizeof(double) * 4;

    // Convolver internal (no IR = minimal — IR itself is not tracked here)
    stats.convolver = 0;  // ConvolverProcessor manages its own memory separately

    // Crossfade buffers (owned by AudioEngine, not DSPCore — report 0 here)
    stats.crossfade = 0;

    // DCBlocker/LoudnessMeter/PeakLimiter/NoiseShaper (fixed-size state, ~few KB each)
    stats.misc = 4096 * 4;  // Conservative estimate for all fixed-size DSP states

    // otherTracked: captured by summing prepare() allocated buffers not in above
    stats.otherTracked = 0;

    return stats;
}

void AudioEngine::DSPCore::reset()
{
    convolverState->resetForRuntime();
    eqState->resetForRuntime();
    dcBlockers().reset();
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

    ramps().resetForRuntime();
    histories().resetForRuntime();
}
