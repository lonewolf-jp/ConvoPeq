//==============================================================================
// SimplePeakLimiter.h
// ★ [P1-1] Release-only Simple Peak Limiter
//
// 設計: Phase 1 — LookAhead なし。Attack 0ms、Release 50-200ms 適応。
//       Soft knee 1.0dB で自然なクリップ特性。
//       ニーは thresholdLinear を中心に左右へ kneeDb/2 dB（対数振幅で対称）。
// 位置付け: 既存 Hard Clamp（jlimit, Safety Net）の前段で動作する。
// ISR: 状態は release envelope (double) のみ。LookAhead FIFO 不要。
//
// ★ #238: 旧実装はニーを [threshold - knee/2, threshold] に置いていた。
//   この区間は threshold / peak >= 1 なのでゲイン削減が起きず、終端も
//   smoothstep の t=0.5 で止まっていた。現在は境界条件から導いた 3 次
//   エルミートで、開始点（ゲイン 1、傾き 0）から終了点（T/peak とその傾き）
//   まで C1 で接続する。導出と数値表は
//   doc/work113/20261122-*/report.md を参照。
//==============================================================================
#pragma once

#include <JuceHeader.h>
#include <cmath>

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
    // thresholdLinear: リミッター閾値（linear 倍率。例: 0.841395 = -1.5 dBFS）
    // kneeDb: ソフトニーの全幅（dB）。thresholdLinear を中心に左右へ kneeDb/2 dB。
    //         0 以下・NaN はハードニー（thresholdLinear で折れる特性）として扱う。
    //         幅の定義は振幅の dB なので、線形振幅では上下の幅が等しくない。
    void processBlock(double* dataL, double* dataR, int numSamples,
                      double thresholdLinear, double kneeDb) noexcept
    {
        if (dataL == nullptr || numSamples <= 0)
            return;

        const bool hasR = (dataR != nullptr);

        // 不正な閾値ではリミッティングしない（ゲイン 1 のまま envelope を解放）。
        const bool limitEnabled = (thresholdLinear > 0.0) && std::isfinite(thresholdLinear);
        const double knee = (kneeDb > 0.0 && std::isfinite(kneeDb)) ? kneeDb : 0.0;
        bool softKnee = limitEnabled && knee > 0.0;

        // ニー境界と終端点のゲイン・傾き。指数は 1 ブロックに 2 回だけ評価する。
        // 終端点は threshold * 10^(knee/40)（= -1.0 dBFS at the call sites）。
        double kneeStart = thresholdLinear;
        double kneeEnd = thresholdLinear;
        double kneeWidth = 0.0;
        double endGain = 1.0;
        double endSlope = 0.0;

        if (softKnee)
        {
            kneeStart = thresholdLinear * std::pow(10.0, -knee / 40.0);
            kneeEnd = thresholdLinear * std::pow(10.0, knee / 40.0);
            kneeWidth = kneeEnd - kneeStart;
            endGain = thresholdLinear / kneeEnd;          // hard-knee gain at kneeEnd
            endSlope = -thresholdLinear / (kneeEnd * kneeEnd);

            // 極端な kneeDb で境界が非有限になったらハードニーへ落とす。
            if (!(kneeStart > 0.0) || !std::isfinite(kneeEnd) || !(kneeEnd > kneeStart))
                softKnee = false;
        }

        for (int i = 0; i < numSamples; ++i)
        {
            // 両チャンネルの最大絶対値を検出
            const double absL = std::abs(dataL[i]);
            const double absR = hasR ? std::abs(dataR[i]) : absL;
            const double peak = juce::jmax(absL, absR);

            // 0 除算を防止（無音はゲイン 1 のまま）
            const double safePeak = juce::jmax(peak, 1.0e-12);

            double desiredGain = 1.0;
            if (limitEnabled && safePeak > (softKnee ? kneeStart : thresholdLinear))
            {
                if (!softKnee || safePeak >= kneeEnd)
                {
                    // リミッティング領域: threshold / peak（ハードニーの特性）
                    desiredGain = thresholdLinear / safePeak;
                }
                else
                {
                    // ニー領域: 両端の値と傾きを満たす 3 次エルミート
                    //   g(kneeStart) = 1,        g'(kneeStart) = 0
                    //   g(kneeEnd)   = T/kneeEnd, g'(kneeEnd)   = -T/kneeEnd^2
                    const double x = (safePeak - kneeStart) / kneeWidth;
                    const double x2 = x * x;
                    const double x3 = x2 * x;
                    const double h00 = 2.0 * x3 - 3.0 * x2 + 1.0;
                    const double h01 = -2.0 * x3 + 3.0 * x2;
                    const double h11 = x3 - x2;
                    desiredGain = h00 + h01 * endGain + h11 * kneeWidth * endSlope;
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
