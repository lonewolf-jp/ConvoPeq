#pragma once

// [WORK113-17] §15 transition 計測ユーティリティ（test-only・計算のみ / I/O なし）。
//
// BassBuzzMeasurement の probe capture（outCapL: capture 開始からの連続波形）に対し、
// flip 時刻前後の以下を算出する（表示は呼び出し側で行う）:
//   - max sample-to-sample jump（pre 定常窓 vs transition 窓）
//   - 1ms RMS エンベロープの transition 窓内最大値
//   - DC offset（pre / post 定常窓）
//   - pre/post 定常窓の振幅（ステップ量）
//   - non-finite 数
//
// rebuild 系 flip（EQ bypass / EQ param）の着地は非同期（debounce + rebuild）のため
// transition 窓を flip +0.9s まで確保し、pre 窓は flip 0.30-0.08s 前、
// post 窓は flip +0.95s 以降を定常とみなす。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace convo_buzz_transition {

struct TransitionResult
{
    double flipAtSec = 0.0;
    double jumpPre = 0.0;
    double jumpTransition = 0.0;
    double rmsPre = 0.0;
    double transientRmsMax = 0.0;
    double dcPre = 0.0;
    double dcPost = 0.0;
    double amplitudePre = 0.0;
    double amplitudePost = 0.0;
    long long nonFiniteCount = 0;
    int needsCheck = 0;
};

inline TransitionResult computeTransitionMetrics(const std::vector<float>& samples,
                                                 double sampleRate,
                                                 double flipAtSec,
                                                 double captureSec)
{
    TransitionResult result;
    result.flipAtSec = flipAtSec;

    const long long total = static_cast<long long>(samples.size());
    if (total <= 1)
        return result;

    auto clampIndex = [&](double timeSec) -> long long {
        const long long v = static_cast<long long>(timeSec * sampleRate);
        if (v < 0) return 0;
        if (v > total - 1) return total - 1;
        return v;
    };
    auto scanJump = [&](long long begin, long long end) -> double {
        double best = 0.0;
        const long long stop = (end < total) ? end : total;
        for (long long i = (begin > 0 ? begin : 1); i < stop; ++i)
        {
            const double d = std::fabs(static_cast<double>(samples[static_cast<size_t>(i)])
                                     - static_cast<double>(samples[static_cast<size_t>(i - 1)]));
            if (d > best) best = d;
        }
        return best;
    };
    auto scanRms = [&](long long begin, long long end) -> double {
        double acc = 0.0;
        long long count = 0;
        const long long stop = (end < total) ? end : total;
        for (long long i = (begin > 0 ? begin : 0); i < stop; ++i)
        {
            const double v = static_cast<double>(samples[static_cast<size_t>(i)]);
            acc += v * v;
            ++count;
        }
        return (count > 0) ? std::sqrt(acc / static_cast<double>(count)) : 0.0;
    };
    auto scanDc = [&](long long begin, long long end) -> double {
        double acc = 0.0;
        long long count = 0;
        const long long stop = (end < total) ? end : total;
        for (long long i = (begin > 0 ? begin : 0); i < stop; ++i)
        {
            acc += static_cast<double>(samples[static_cast<size_t>(i)]);
            ++count;
        }
        return (count > 0) ? acc / static_cast<double>(count) : 0.0;
    };
    auto scanPeak = [&](long long begin, long long end) -> double {
        double best = 0.0;
        const long long stop = (end < total) ? end : total;
        for (long long i = (begin > 0 ? begin : 0); i < stop; ++i)
        {
            const double a = std::fabs(static_cast<double>(samples[static_cast<size_t>(i)]));
            if (a > best) best = a;
        }
        return best;
    };

    const long long flipIndex = clampIndex(flipAtSec);
    const long long preBegin = clampIndex(flipAtSec - 0.30);
    const long long preEnd = clampIndex(flipAtSec - 0.08);
    const long long transitionEnd = clampIndex(flipAtSec + 0.90);
    const long long postBegin = clampIndex(flipAtSec + 0.95);

    const long long envelopeWidth = static_cast<long long>(sampleRate * 0.001);
    const long long envelopeStop = (transitionEnd < total) ? transitionEnd : total;
    double transientRmsMax = 0.0;
    for (long long i = flipIndex; i + envelopeWidth < envelopeStop; i += envelopeWidth)
    {
        const double r = scanRms(i, i + envelopeWidth);
        if (r > transientRmsMax) transientRmsMax = r;
    }

    long long nonFiniteCount = 0;
    for (const float v : samples)
    {
        if (!std::isfinite(static_cast<double>(v))) ++nonFiniteCount;
    }

    result.jumpPre = scanJump(preBegin, preEnd);
    result.jumpTransition = scanJump(flipIndex, transitionEnd);
    result.rmsPre = scanRms(preBegin, preEnd);
    result.dcPre = scanDc(preBegin, preEnd);
    result.dcPost = scanDc(postBegin, total);
    result.amplitudePre = scanPeak(preBegin, preEnd);
    result.amplitudePost = scanPeak(postBegin, total);
    result.transientRmsMax = transientRmsMax;
    result.nonFiniteCount = nonFiniteCount;
    result.needsCheck = (result.jumpTransition > 0.05
                         || std::fabs(result.dcPost - result.dcPre) > 1.0e-3
                         || result.nonFiniteCount > 0) ? 1 : 0;
    return result;
}

// [WORK113-17] TransitionResult を 1 行のログ文字列に整形する（I/O は呼び出し側）。
//   小値（DC 等）を保持するため DC は scientific、他は fixed 6 桁。
inline std::string formatTransitionLine(const TransitionResult& tr, const std::string& tag)
{
    std::ostringstream os;
    os << "[TRANSITION] " << tag
       << " flipT=" << std::fixed << std::setprecision(2) << tr.flipAtSec
       << " jumpPre=" << std::setprecision(6) << tr.jumpPre
       << " jumpTr=" << tr.jumpTransition
       << " rmsPre=" << tr.rmsPre
       << " rmsTrMax=" << tr.transientRmsMax
       << " dcPre=" << std::setprecision(3) << std::scientific << tr.dcPre
       << " dcPost=" << tr.dcPost
       << " ampPre=" << std::setprecision(6) << std::fixed << tr.amplitudePre
       << " ampPost=" << tr.amplitudePost
       << " nonFinite=" << tr.nonFiniteCount
       << " check=" << tr.needsCheck;
    return os.str();
}

} // namespace convo_buzz_transition
