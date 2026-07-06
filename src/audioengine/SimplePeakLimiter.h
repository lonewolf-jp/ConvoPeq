//==============================================================================
// SimplePeakLimiter.h
// ★ [P1-1] Release-only Simple Peak Limiter
//
// 設計: Phase 1 — LookAhead なし。Attack 0ms、Release 50-200ms 適応。
//       Soft knee 1.0dB で自然なクリップ特性。
// 位置付け: 既存 Hard Clamp（jlimit, Safety Net）の前段で動作する。
// ISR: 状態は release envelope (double) のみ。LookAhead FIFO 不要。
//==============================================================================
#pragma once

#include <JuceHeader.h>

class SimplePeakLimiter
{
public:
    // prepare: サンプルレートと Release 時定数 (ms) を設定
    void prepare(double sampleRate, double releaseMs) noexcept
    {
        const double releaseSec = releaseMs * 0.001;
        releaseCoeff = (releaseSec > 0.0 && sampleRate > 0.0)
            ? std::exp(-1.0 / (sampleRate * releaseSec))
            : 0.0;
    }

    // reset: envelope を 1.0 (no gain reduction) にリセット
    void reset() noexcept
    {
        envelope = 1.0;
    }

    // processBlock: Stereo ブロックに対してピークリミッティングを適用
    // thresholdLinear: リミッター閾値（linear 倍率, e.g. 0.891 = -1.0dB）
    // kneeLinear: ソフトニー幅（linear 倍率, e.g. 0.122 = 約1dB）
    void processBlock(double* dataL, double* dataR, int numSamples,
                      double thresholdLinear, double kneeLinear) noexcept
    {
        if (dataL == nullptr || numSamples <= 0)
            return;

        const double clipStart = thresholdLinear - kneeLinear * 0.5;
        const bool hasR = (dataR != nullptr);

        for (int i = 0; i < numSamples; ++i)
        {
            // 両チャンネルの最大絶対値を検出
            const double absL = std::abs(dataL[i]);
            const double absR = hasR ? std::abs(dataR[i]) : absL;
            const double peak = juce::jmax(absL, absR);

            // 必要なゲインリダクションを計算 (soft knee)
            double desiredGain = 1.0;
            if (peak > clipStart)
            {
                if (peak <= thresholdLinear)
                {
                    // Knee 領域: 3次スプライン補間
                    const double t = (peak - clipStart) / kneeLinear;
                    const double kneeShape = t * t * (3.0 - 2.0 * t);
                    desiredGain = 1.0 - (1.0 - thresholdLinear / peak) * kneeShape;
                }
                else
                {
                    // リミッティング領域: threshold / peak
                    desiredGain = thresholdLinear / peak;
                }
            }

            // Envelope 追跡: Attack 即時, Release 時定数
            if (desiredGain < envelope)
                envelope = desiredGain;
            else
                envelope = 1.0 + (envelope - 1.0) * releaseCoeff;

            // Apply gain reduction
            dataL[i] *= envelope;
            if (hasR)
                dataR[i] *= envelope;
        }
    }

    double getCurrentEnvelope() const noexcept { return envelope; }

private:
    double releaseCoeff = 0.0;
    double envelope = 1.0;
};
