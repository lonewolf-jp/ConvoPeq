//============================================================================
#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>

#include "AlignedAllocation.h"
#include "audioengine/AtomicAccess.h"
#include "LockFreeRingBuffer.h"

//============================================================================
/**
    LoudnessMeter ── ITU-R BS.1770-4/5 + EBU R128 準拠ラウドネスメーター

    K-weighting フィルタ処理 → ブロック平均電力 → RingBuffer publish
    集計（Momentary/Short-term/Integrated）は専用ワーカースレッド。

    Audio Thread: processBlock() のみ呼び出し（lock-free, 非メモリ確保）
*/
class LoudnessMeter
{
public:
    static constexpr int kMaxChannels = 2;
    static constexpr double kChannelWeightStereo[2] = { 1.0, 1.0 };

    LoudnessMeter() = default;
    ~LoudnessMeter() = default;

    LoudnessMeter(const LoudnessMeter&) = delete;
    LoudnessMeter& operator=(const LoudnessMeter&) = delete;

    /** prepare: (Message Thread) */
    void prepare(double sampleRate, int maxBlockSize);

    /** Audio Thread: ブロックの平均電力を計算しRingBufferにpublish */
    void processBlock(const double* dataL, const double* dataR, int numSamples) noexcept;

    void reset() noexcept;

    /** サンプルレートに応じてK-weightingフィルタ係数を再計算（prepare内部で自動呼出） */
    void updateCoefficients(double sampleRate);

    //--- RingBuffer (Audio Thread publish, Worker Thread consume) ---
    struct BlockPower {
        double meanSquare = 0.0; // チャンネル重み適用済み M/S
        double peakLinear = 0.0;
        uint64_t blockIndex = 0;
    };

    LockFreeRingBuffer<BlockPower, 4096>& getRingBuffer() noexcept { return ringBufferStorage->ringBuffer; }

private:
    // K-weighting filter (2-stage biquad, BS.1770-4 Table 1)
    // Stage 1: Pre-filter (High-shelf)
    // Stage 2: RLB filter (High-pass)
    struct KWeightingState {
        double x1[2] = {}; // 入力遅延
        double x2[2] = {};
        double y1[2] = {};
        double y2[2] = {};
    };

    KWeightingState preFilterState[2];  // [channel]
    KWeightingState rlbFilterState[2];

    // ★ [work74 FIX-02] サンプルレート依存係数（updateCoefficients で設定）
    //   48kHz固定値 kPreBiquad / kRlbBiquad に代わり、インスタンスごとに保持する。
    double preFilterCoeffs[5] = { 0 };
    double rlbFilterCoeffs[5] = { 0 };

    double sampleRate = 0.0;
    uint64_t blockCounter = 0;
    int preparedBlockSize = 0;

    // Audio Thread安全のため事前確保されたワークバッファ
    convo::ScopedAlignedPtr<double> filterWorkBuffer;
    int filterWorkCapacity = 0;

    // リングバッファはDSPCoreサイズ削減のため動的確保（ScopedAlignedPtr内にRingBufferStorageを配置）
    struct RingBufferStorage {
        alignas(64) LockFreeRingBuffer<BlockPower, 4096> ringBuffer;
    };
    convo::ScopedAlignedPtr<RingBufferStorage> ringBufferStorage;

    inline double processKWeightingStage(const double coeffs[5], KWeightingState& state, double x) noexcept
    {
        // Direct Form I: a0=1 normalized
        // coeffs = {b0, b1, b2, a1, a2}
        const double y = coeffs[0] * x + coeffs[1] * state.x1[0] + coeffs[2] * state.x2[0]
                       - coeffs[3] * state.y1[0] - coeffs[4] * state.y2[0];
        state.x2[0] = state.x1[0];
        state.x1[0] = x;
        state.y2[0] = state.y1[0];
        state.y1[0] = y;
        return y;
    }
};

    //--- K-weighting coefficients (48kHz, ITU-R BS.1770-4 Table 1) ---
    //   48kHz固定値として定義するが、updateCoefficients() で任意のサンプルレートに対応する。
    //   ★ [work74 FIX-02] これらの定数は48kHzデフォルト値として保持し、
    //     インスタンスメンバ preFilterCoeffs / rlbFilterCoeffs への初期値として使用する。
    //
    //   初期値設定: LoudnessMeter コンストラクタで static constexpr からコピーする。
    //   （static constexpr 配列をそのままにすることで後方互換性を維持）
// Stage 1: Pre-filter (High-shelf)
static constexpr double kPreBiquad[5] = {
    1.535124859586970, -2.691696189406380, 1.198392810852850,
    -1.690659293182410, 0.732480774215850
};

// Stage 2: RLB filter (High-pass)
static constexpr double kRlbBiquad[5] = {
    1.0, -2.0, 1.0,
    -1.990047454833980, 0.990072250366210
};
