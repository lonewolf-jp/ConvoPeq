#pragma once

#include <JuceHeader.h>

//==============================================================================
/**
    IRAnalyzer — IR (Impulse Response) 解析専用コンポーネント。

    IRConverter から FFT 解析を分離し、以下の責務を担当する:
    - 周波数応答ピーク推定（FFT + Tukey窓 + ガウス補間）
    - 将来拡張: group delay / phase ripple / crest factor / minimum phase 判定

    本設計では振幅推定（最大振幅）のみを行い、PSD推定は行わない。
    Harris(1978) の窓関数分類に基づき、コヒーレントゲインのみ補正する。
*/
namespace IRAnalyzer
{
    // ★ kMaxAnalysisWindow = 65536 上限。
    //   IR長がこれを超える場合、最初の kMaxAnalysisWindow sample のみ解析対象。
    //   将来 192kHz/384kHz 対応時は Policy 化を検討。
    inline constexpr int kMaxAnalysisWindow = 65536;

    // Tukey α=0.5（両端 25% コサインテーパー、中央 50% フラット）
    inline constexpr double kTukeyAlpha = 0.5;

    //==============================================================================
    /**
        IRの周波数応答ピークをFFT解析で推定。
        Tukey窓（α=0.5）適用後の複素スペクトル振幅の最大値を返す。

        @param ir  入力 IR（任意長、任意チャンネル数）
        @return 線形振幅値（倍率）。IRが無効な場合は 1.0

        備考:
        - FFT サイズは nextPowerOfTwo(min(ir長, kMaxAnalysisWindow))
        - MKL DFTI 前方変換（DFTI_BACKWARD_SCALE = 1/N, 前方無スケール）
        - コヒーレントゲイン補正（windowMean で除算）
        - 3点ガウス補間で FFT bin 間ピーク誤差を軽減
    */
    [[nodiscard]] double estimateMaxFrequencyResponseGain(
        const juce::AudioBuffer<double>& ir) noexcept;

} // namespace IRAnalyzer
