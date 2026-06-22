//============================================================================
#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>

#include "AlignedAllocation.h"
#include "audioengine/AtomicAccess.h"

//============================================================================
/**
    TruePeakDetector ── ITU-R BS.1770-4/5 準拠 True Peak 検出器

    4倍オーバーサンプリングによるインターサンプルピーク検出。
    計測専用（ゲイン演算なし）。Audio Thread 安全。
*/
class TruePeakDetector
{
public:
    static constexpr bool isLinearPhaseFIR = true;
    static constexpr int kOversamplingRatio = 4;
    static constexpr int kMaxChannels = 2;
    // 確定tap数: 63 （ITU-R BS.1770-3 Example 48tapを上回る。Hansen 2012文献確定）
    static constexpr int kDefaultTaps = 63;
    static constexpr double kDefaultAttenuationDb = 100.0;

    TruePeakDetector() = default;
    ~TruePeakDetector();

    TruePeakDetector(const TruePeakDetector&) = delete;
    TruePeakDetector& operator=(const TruePeakDetector&) = delete;

    /** 4倍オーバーサンプラを準備 (Message Thread) */
    void prepare(double sampleRate, int maxBlockSize, int taps = kDefaultTaps);

    /** Audio Thread: ブロックのTruePeakを検出 */
    double processBlock(const double* dataL, const double* dataR, int numSamples) noexcept;

    void reset() noexcept;

private:
    convo::ScopedAlignedPtr<double> upsampleBuffer;
    int bufferCapacity = 0;
    int upsampledCapacity = 0;
    double peakHold = 0.0;
    std::atomic<double> currentSampleRate{ 0.0 };

    // 内部4倍オーバーサンプラ
    struct Stage {
        int taps = 0;
        int centerTap = 0;
        int centerParity = 0;
        int convParity = 0;
        int convCount = 0;
        int centerDelayInput = 0;
        int historyUpKeep = 0;
        int historyDownKeep = 0;
        int maxInputSamples = 0;
        int maxOutputSamples = 0;
        double centerCoeff = 0.5;
        convo::ScopedAlignedPtr<double> convCoeffs;
        convo::ScopedAlignedPtr<double> convCoeffsReversed;
        convo::ScopedAlignedPtr<double> upHistory[2];
        int upHistorySize = 0;
    };

    Stage stages[2]; // 4x = 2 stages (2x + 2x)

    void prepareStage(Stage& stage, int taps, double attenuationDb, int stageInputMax);
    static double besselI0(double x) noexcept;
    static double dotProductAvx2(const double* x, const double* coeffs, int n) noexcept;

    void interpolateStage(const Stage& stage,
                          const double* input, int inputSamples,
                          double* output, int channel) noexcept;
};
