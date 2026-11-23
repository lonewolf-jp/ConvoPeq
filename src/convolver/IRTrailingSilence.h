#pragma once

// ★ B-01 / B-02: 末尾無音トリムの共通規則。
//
//   LoaderThread（非同期・本適用）と IRConverter（同期・プレビュー/キャッシュ）で
//   同じ規則を使う。規則は従来の LoaderThread 実装と同じ:
//     - 閾値より大きい絶対値を持つ最後のサンプル位置 + 1 を長さとする
//     - 1 チャネル目（index 0）と 2 チャネル目（存在すれば）の大きい方を採る
//     - 該当が無ければ 1 サンプルを残す
//
//   JUCE に依存しない（単体テストで境界値を直接叩ける）。

#include <cmath>

namespace convo::irtrim
{

template <typename Sample>
int computeTrimmedLength (const Sample* channel0,
                          const Sample* channel1,
                          int numSamples,
                          double threshold = 1.0e-15) noexcept
{
    if (channel0 == nullptr || numSamples <= 0)
        return 0;

    for (int i = numSamples - 1; i >= 0; --i)
    {
        const double a0 = std::abs (static_cast<double> (channel0[i]));
        if (a0 > threshold)
            return i + 1;

        if (channel1 != nullptr)
        {
            const double a1 = std::abs (static_cast<double> (channel1[i]));
            if (a1 > threshold)
                return i + 1;
        }
    }

    // 全サンプルが閾値以下でも 1 サンプルは残す（0 長にしない）。
    return 1;
}

} // namespace convo::irtrim
