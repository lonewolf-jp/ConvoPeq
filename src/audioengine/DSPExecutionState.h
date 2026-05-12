#pragma once

#include <array>
#include <cstdint>
#include "DspNumericPolicy.h"
#include "core/RCUReader.h"

namespace convo {

// DSP_THREAD_STATE: Convolver の実行時 mutable 状態。
// audio thread 専有、crossfade 期間中も thread-confined。
struct ConvolverDSPState
{
    // IMMUTABLE_RUNTIME_CONSTANT
    static constexpr int kMaxBlockSize = 32768;

    // ── Bypass delay compensation ──
    // DELAY_BUFFER_SIZE は IRs の最大遅延によって決まる (通常数千サンプル)
    double* bypassDelayBuf[2] = { nullptr, nullptr };
    int bypassDelayCapacity = 0;
    int delayWritePos = 0;

    // ── Processing buffers (pointers to ConvolverProcessor storage) ──
    // prepareToPlay で ConvolverProcessor により割り当てられ、
    // bindExecutionState で参照される。
    double* dryBuf[2] = { nullptr, nullptr };
    double* oldDryBuf[2] = { nullptr, nullptr };
    double* wetBuf[2] = { nullptr, nullptr };
    double* smoothingBuf[2] = { nullptr, nullptr };
    int dryCapacity = 0;
    int oldDryCapacity = 0;
    int wetCapacity = 0;
    int smoothingCapacity = 0;

    // ── Crossfade / Mix ramps ──
    convo::LinearRamp crossfadeGain { 1.0 };
    convo::LinearRamp mixSmoother { 1.0 };
    convo::LinearRamp latencySmoother { 0.0 };
    double oldDelay = 0.0;
    bool initialized = false;

    // ── Latency alignment ──
    // latency crossfade 中のバッファ＆ writePos
    struct LatencyAlign {
        double* bufs[4] = { nullptr, nullptr, nullptr, nullptr };  // old_L, old_R, new_L, new_R
        int capacity = 0;
        int writePos = 0;
        int delaySamples[2] = { 0, 0 };  // [old, new]
    } latencyAlign;

    // ── RCU reader for Convolver state snapshot ──
    convo::RCUReader convRcuReader;
};

// DSP_THREAD_STATE: EQ の実行時 mutable 状態。
struct EQDSPState
{
    static constexpr int kNumBands = 20;
    static constexpr int kMaxChannels = 2;

    // [channel][band][z1/z2]
    std::array<std::array<std::array<double, 2>, kNumBands>, kMaxChannels> filterState{};

    double agcCurrentGain = 1.0;
    double agcEnvInput = 0.0;
    double agcEnvOutput = 0.0;
    double cachedInputRms = 0.0;
    const double* agcAttackCoeffTable = nullptr;
    const double* agcReleaseCoeffTable = nullptr;
    const double* agcSmoothCoeffTable = nullptr;
    int agcCoeffTableCapacity = 0;

    convo::RCUReader rcuReader;

    double* dryBypassBuffer = nullptr;
    int dryBypassCapacity = 0;

    double* parallelInputBuffer = nullptr;
    double* parallelWorkBuffer = nullptr;
    double* parallelAccumBuffer = nullptr;
    int parallelBufferCapacity = 0;

    double* structureOldOutBuffer = nullptr;
    double* structureNewOutBuffer = nullptr;
    int structureXfadeBufferCapacity = 0;

    bool bypassed = false;
    int activeStructure = 0;
    convo::LinearRamp smoothTotalGain { 1.0 };
    convo::LinearRamp bypassFadeGain { 1.0 };
    bool rampsInitialized = false;
};

// DSP_THREAD_STATE: DC blocker runtime state
struct DCBlockerDSPState
{
    struct ChannelState {
        double m_state = 0.0;  // DC追跡一次遅延値
    };
    std::array<ChannelState, 2> channels;  // [L, R]
};

// DSP_THREAD_STATE: Ramp state (bypass fade-in)
struct RampDSPState
{
    int fadeInSamplesLeft = 0;
    double bypassFadeGain = 1.0;
    bool bypassed = false;
};

// DSP_THREAD_STATE: Fixed latency buffer and history
struct HistoryDSPState
{
    double* fixedLatBufL = nullptr;
    double* fixedLatBufR = nullptr;
    int bufCapacity = 0;
    int writePos = 0;
    double softClipPrev = 0.0;
};

// DSP_THREAD_STATE: Output filter biquad state (4 cascaded filters)
struct OutputFilterDSPState
{
    struct BiquadState {
        double w1 = 0.0;
        double w2 = 0.0;
    };
    // hc, lc, hpf, lpf
    std::array<BiquadState, 4> states;
};

// DSP_THREAD_STATE: Oversampling filter state
struct OversamplingDSPState
{
    static constexpr int kMaxStages = 8;

    struct StageState {
        double* upHistory = nullptr;
        int upHistoryCapacity = 0;
        double* downHistory = nullptr;
        int downHistoryCapacity = 0;
        double centerDelayInput = 0.0;
    };

    std::array<StageState, kMaxStages> stages;
    double* workBufA = nullptr;
    double* workBufB = nullptr;
    int workBufCapacity = 0;
    std::atomic<bool> corruptionDetected { false };
};

// DSP_THREAD_STATE: Fixed noise shaper state
struct FixedNoiseShaperDSPState
{
    static constexpr int kOrder = 10;
    static constexpr int kMaxChannels = 2;

    // errors[ch][tap]
    std::array<std::array<double, kOrder>, kMaxChannels> errors{};
    int writePos = 0;

    // RNG state for noise generation
    struct Xoshiro256State {
        std::uint64_t s[4] = { 0 };
    } rngState;

    std::atomic<bool> needsReset { false };
};

// DSP_THREAD_STATE: 15-tap fixed noise shaper state (same structure as FixedNoiseShaper)
struct Fixed15TapNoiseShaperDSPState
{
    static constexpr int kOrder = 15;
    static constexpr int kMaxChannels = 2;

    std::array<std::array<double, kOrder>, kMaxChannels> errors{};
    int writePos = 0;

    struct Xoshiro256State {
        std::uint64_t s[4] = { 0 };
    } rngState;

    std::atomic<bool> needsReset { false };
};

// DSP_THREAD_STATE: Adaptive (lattice) noise shaper state
struct AdaptiveNoiseShaperDSPState
{
    static constexpr int kMaxOrder = 64;
    static constexpr int kMaxChannels = 2;

    struct LatticeState {
        std::array<double, kMaxOrder> states{};
    };
    std::array<LatticeState, kMaxChannels> channels;

    struct Xoshiro256State {
        std::uint64_t s[4] = { 0 };
    } rngState;

    // Working coefficients (updated from RuntimeGraph)
    double* coeffsCopy = nullptr;
    int coeffsCapacity = 0;
};

// DSP_THREAD_STATE: Psychoacoustic dither state (after RNG worker separation)
struct DitherDSPState
{
    // RNG ring reference (SPSC shared with RNG worker)
    // Note: actual ring is managed by AudioEngine RNG worker
    void* rngRingRef = nullptr;
    int rngReadPos = 0;

    // State buffer for shaper
    double* shaperStateBuffer = nullptr;
    int shaperStateCapacity = 0;
};

// DSP_THREAD_STATE: Scratch and utility buffers
struct ScratchDSPState
{
    double* alignedL = nullptr;
    double* alignedR = nullptr;
    int alignedCapacity = 0;
};

// DSP_THREAD_STATE: Crossfade state
struct CrossfadeDSPState
{
    convo::LinearRamp gainRamp { 1.0 };
    convo::LinearRamp dryScaleRamp { 1.0 };

    double* mixBufL = nullptr;
    double* mixBufR = nullptr;
    int mixBufCapacity = 0;
};

// DSP_THREAD_STATE: Audio Thread でのみ更新する Phase-1 骨格。
struct DSPExecutionState
{
    void* currentNode = nullptr;
    void* nextNode = nullptr;
    std::uint64_t observedGeneration = 0;
    bool inCrossfade = false;

    ConvolverDSPState conv;  // IMMUTABLE_RUNTIME の graph ref を audio thread 側で実行
    EQDSPState eq;           // IMMUTABLE_RUNTIME の coeffs/AGC tables を audio thread 側で参照
    DCBlockerDSPState dcBlocker;
    RampDSPState ramp;
    HistoryDSPState history;
    OutputFilterDSPState outputFilter;
    OversamplingDSPState oversampling;
    FixedNoiseShaperDSPState fixedNsState;
    Fixed15TapNoiseShaperDSPState fixed15NsState;
    AdaptiveNoiseShaperDSPState adaptiveNsState;
    DitherDSPState ditherState;
    ScratchDSPState scratch;
    CrossfadeDSPState crossfade;
};

} // namespace convo
