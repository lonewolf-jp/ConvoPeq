//============================================================================
// PolyphaseGainCandidateRef.h — Phase 0 Shadow Reference (test-only, v2 full)
// work113 / remediation v3.1 §2.9（R15-3 + R16-1 + R17-3 + R17-4）/ R8-1
//
// production src/ 変更 0。Phase 0 characterization 用の Shadow 独立 reference model。
//   ★ R15-3: production を直接呼ばない（同一数学的契約を別コードで独立実装）
//   ★ R16-1: prepareStage の 10 手順を演算順序ごと独立実装
//   ★ R17-3: 同一 test executable 内で production（CustomInputOversampler.cpp）と
//            同一 compile option で compile され、scalar↔scalar / SIMD↔SIMD の
//            対応経路で bitwise 比較に供する
//   ★ R17-4: R17-4 (A)〜(E) 検証の受皿（StageModel 全フィールド公開）
//
// Candidate E:  centerValue *= 2.0（両 polyphase 位相への対称適用）— candidate hypothesis。
//   shadow cand は予測にすぎない（R12-8）: 最終判定は Phase 1 の flag ON ビルド実測。
//
// 履歴アルゴリズムは production CustomInputOversampler.cpp の
// interpolateStage(:492-568) / decimateStage(:570-723) / reset(:452-467) /
// clearAllStages(:469-484) / processUp(:725-783) / processDown(:785-872) と
// 同一の観測契約（値・演算順序・denorm/isBadSample 位置）を独立コードで実装する。
//============================================================================
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__AVX2__) || defined(__FMA__)
#include <immintrin.h>
#endif

namespace convo::polyphase_ref
{

// compile-time と同義の shadow 切替（0/1 以外は ill-formed）。
// production の CONVOPEQ_CORRECT_POLYPHASE_GAIN とは独立に、test 側で切替可能にする。
#ifndef CONVOPEQ_POLYPHASE_REF_CANDIDATE
#define CONVOPEQ_POLYPHASE_REF_CANDIDATE 0
#endif
#if CONVOPEQ_POLYPHASE_REF_CANDIDATE != 0 && CONVOPEQ_POLYPHASE_REF_CANDIDATE != 1
#error "CONVOPEQ_POLYPHASE_REF_CANDIDATE must be 0 or 1"
#endif

// ── 定数パス契約（R16-1） ────────────────────────────────────────────────
// juce::MathConstants<double>::pi = static_cast<double>(3.141592653589793238L)
// と同一式。test 側で juce 定数との bitwise 一致を assert する。
inline constexpr double kPiRef = static_cast<double>(3.141592653589793238L);

// production DspNumericPolicy.h:132-138 kDenormThresholdAudioState = 1.0e-20 と同値。
inline constexpr double kDenormThresholdRef = 1.0e-20;

// ── production 独立実装群（値・演算順序は production と同一・コードは別物） ──

inline double fastAbsRef(double x) noexcept
{
    std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    bits &= 0x7FFFFFFFFFFFFFFFULL;
    return std::bit_cast<double>(bits);
}

// scalar isBadSample 契約（production CIO.cpp:24-33 と同一述語・独立実装）:
//   NaN/Inf（指数部全ビット）または |x| > 2^53（0x4340000000000000）
inline bool isBadSampleRef(double x) noexcept
{
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
    const std::uint64_t exp = bits & 0x7FF0000000000000ULL;
    if (exp == 0x7FF0000000000000ULL)
        return true;
    constexpr std::uint64_t limit = 0x4340000000000000ULL;
    return (bits & 0x7FFFFFFFFFFFFFFFULL) > limit;
}

// besselI0 — production CIO.cpp:144-157（private）と同一級数和の独立実装。
inline double besselI0Ref(double x) noexcept
{
    double sum = 1.0;
    double term = 1.0;
    const double xx = x * x;
    for (int n = 1; n < 100; ++n)
    {
        term *= xx / (4.0 * static_cast<double>(n) * static_cast<double>(n));
        sum += term;
        if (term < sum * 1.0e-18)
            break;
    }
    return sum;
}

// Kaiser β（production CIO.cpp:301-304 と同一 3 分岐）。
inline double kaiserBetaRef(double attenuationDb) noexcept
{
    return (attenuationDb > 50.0)
         ? (0.1102 * (attenuationDb - 8.7))
         : ((attenuationDb >= 21.0)
              ? (0.5842 * std::pow(attenuationDb - 21.0, 0.4)
                 + 0.07886 * (attenuationDb - 21.0))
              : 0.0);
}

inline constexpr int kMaxChannelsRef = 2;

//============================================================================
// StageModel — production Stage（CIO.h:74-94）の全フィールドを独立保持。
// R17-4 (A) coefficient 検証の比較対象。bitwise 比較は値のみに依存するため
// std::vector を使用する（アライメントは FP 結果に影響しない）。
//============================================================================
struct StageModel
{
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
    std::vector<double> convCoeffs;
    std::vector<double> convCoeffsReversed;
    std::vector<double> upHistory[kMaxChannelsRef];
    std::vector<double> downHistory[kMaxChannelsRef];
    int upHistorySize = 0;
    int downHistorySize = 0;

    // shadow 側の診断カウンタ（production Stage には存在しない・corruption 位置同期の観測用）。
    // R17-4 (A) の比較対象には含めない（production 側フィールドではないため）。
    std::uint64_t corruptionEvents = 0;
};

//============================================================================
// prepareStageModel — production prepareStage（:287-390）の 10 手順を
// 演算順序ごと独立実装（R16-1）。戻り値 false はallocation失敗のみ。
//============================================================================
inline bool prepareStageModel(StageModel& stage, int taps, double attenuationDb, int stageInputMax)
{
    stage = StageModel{};

    // (1) taps を奇数に（juce::jmax(3, taps|1) と同一評価形）
    stage.taps = (3 > (taps | 1)) ? 3 : (taps | 1);
    // (2) center index
    stage.centerTap = (stage.taps - 1) / 2;
    // (3) parity
    stage.centerParity = stage.centerTap & 1;
    stage.convParity = 1 - stage.centerParity;
    stage.maxInputSamples = stageInputMax;
    stage.maxOutputSamples = stageInputMax * 2;

    // (4) Kaiser β（3 分岐） + (5) i0Beta
    const double beta = kaiserBetaRef(attenuationDb);
    const double i0Beta = besselI0Ref(beta);
    const int M = stage.centerTap;

    // (6) sinc × Kaiser window（juce::jmax(0.0, y) = (0.0 < y ? y : 0.0) と同一評価形）
    std::vector<double> raw(static_cast<std::size_t>(stage.taps), 0.0);
    for (int n = 0; n < stage.taps; ++n)
    {
        const double t = static_cast<double>(n - M);
        const double sinc = (n == M)
                          ? 0.5
                          : (std::sin(kPiRef * 0.5 * t) / (kPiRef * t));
        const double frac = static_cast<double>(n - M) / static_cast<double>(M);
        const double yWindow = 1.0 - frac * frac;
        const double win = (0.0 < yWindow) ? yWindow : 0.0;
        const double window = besselI0Ref(beta * std::sqrt(win)) / i0Beta;
        raw[static_cast<std::size_t>(n)] = sinc * window;
    }

    // (7) halfband zeroing（centerParity 一致の非 center を 0）
    for (int n = 0; n < stage.taps; ++n)
    {
        if (n != stage.centerTap && ((n & 1) == stage.centerParity))
            raw[static_cast<std::size_t>(n)] = 0.0;
    }

    // (8) sum → 1.0 正規化（hoisted inv = 1.0/sum・逐次加算）
    double sum = 0.0;
    for (int i = 0; i < stage.taps; ++i)
        sum += raw[static_cast<std::size_t>(i)];
    if (fastAbsRef(sum) > 1.0e-20)
    {
        const double inv = 1.0 / sum;
        for (int i = 0; i < stage.taps; ++i)
            raw[static_cast<std::size_t>(i)] *= inv;
    }

    // (9) center = 0.5 固定 + 非center → 0.5 正規化 + 再固定
    raw[static_cast<std::size_t>(stage.centerTap)] = 0.5;
    double nonCenterSum = 0.0;
    for (int i = 0; i < stage.taps; ++i)
    {
        if (i != stage.centerTap)
            nonCenterSum += raw[static_cast<std::size_t>(i)];
    }
    if (fastAbsRef(nonCenterSum) > 1.0e-20)
    {
        const double scale = 0.5 / nonCenterSum;
        for (int i = 0; i < stage.taps; ++i)
        {
            if (i != stage.centerTap)
                raw[static_cast<std::size_t>(i)] *= scale;
        }
    }
    raw[static_cast<std::size_t>(stage.centerTap)] = 0.5;

    // (10) polyphase extraction + history 契約
    stage.convCount = (stage.taps - stage.convParity + 1) / 2;
    stage.convCoeffs.assign(static_cast<std::size_t>(stage.convCount), 0.0);
    stage.convCoeffsReversed.assign(static_cast<std::size_t>(stage.convCount), 0.0);
    for (int r = 0; r < stage.convCount; ++r)
    {
        const int k = stage.convParity + (r << 1);
        stage.convCoeffs[static_cast<std::size_t>(r)]
            = (k < stage.taps) ? raw[static_cast<std::size_t>(k)] : 0.0;
        stage.convCoeffsReversed[static_cast<std::size_t>(stage.convCount - 1 - r)]
            = stage.convCoeffs[static_cast<std::size_t>(r)];
    }

    stage.centerCoeff = raw[static_cast<std::size_t>(stage.centerTap)];
    stage.centerDelayInput = (stage.centerTap - stage.centerParity) / 2;
    stage.historyUpKeep = (stage.convCount - 1 > stage.centerDelayInput)
                        ? (stage.convCount - 1) : stage.centerDelayInput;
    // loadStride2 が ptr[-6] までアクセスするため +6 マージン（production :370-372 と同一）
    stage.historyDownKeep = (stage.centerTap > (stage.convParity + ((stage.convCount - 1) << 1) + 6))
                          ? stage.centerTap
                          : (stage.convParity + ((stage.convCount - 1) << 1) + 6);

    stage.upHistorySize = stage.historyUpKeep + stage.maxInputSamples + 16;
    stage.downHistorySize = stage.historyDownKeep + stage.maxOutputSamples + 16;

    for (int ch = 0; ch < kMaxChannelsRef; ++ch)
    {
        stage.upHistory[ch].assign(static_cast<std::size_t>(stage.upHistorySize), 0.0);
        stage.downHistory[ch].assign(static_cast<std::size_t>(stage.downHistorySize), 0.0);
    }
    return true;
}

#if defined(__AVX2__)
//============================================================================
// dotProductAvx2Ref — production CIO.cpp:159-216 と同一帰還構造の独立実装
//   （4 accumulator × 16 要素 main loop + prefetch guard(i+64<n) + 4 要素剰余 +
//     tree reduction + horizontal add + scalar 剰余 + isBadSample/denorm 内部清掃）
//   production と同一 compile option の同一 target 内で bitwise 一致する。
//============================================================================
inline double dotProductAvx2Ref(const double* __restrict x,
                                const double* __restrict coeffs,
                                int n) noexcept
{
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    int i = 0;
    for (; i <= n - 16; i += 16)
    {
        if (i + 64 < n)
        {
            _mm_prefetch(reinterpret_cast<const char*>(x + i + 64), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(coeffs + i + 64), _MM_HINT_T0);
        }
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i),      _mm256_loadu_pd(coeffs + i),      acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 4),  _mm256_loadu_pd(coeffs + i + 4),  acc1);
        acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 8),  _mm256_loadu_pd(coeffs + i + 8),  acc2);
        acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 12), _mm256_loadu_pd(coeffs + i + 12), acc3);
    }
    for (; i <= n - 4; i += 4)
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_loadu_pd(coeffs + i), acc0);

    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);

    __m128d vLo = _mm256_castpd256_pd128(acc0);
    __m128d vHi = _mm256_extractf128_pd(acc0, 1);
    __m128d vSum = _mm_add_pd(vLo, vHi);
    vSum = _mm_hadd_pd(vSum, vSum);
    double sum = _mm_cvtsd_f64(vSum);

#if defined(__AVX2__)
    _mm256_zeroupper();
#endif

    for (; i < n; ++i)
        sum += x[i] * coeffs[i];

    if (isBadSampleRef(sum))
        sum = 0.0;
    else if (fastAbsRef(sum) < kDenormThresholdRef)
        sum = 0.0;

    return sum;
}
#endif // __AVX2__

#if defined(__AVX2__) && defined(__FMA__)
//============================================================================
// loadStride2Ref — production CIO.cpp:54-67 と同一構造の独立実装。
//   { ptr[0], ptr[-2], ptr[-4], ptr[-6] } を返す stride-2 ロード。
//============================================================================
inline __m256d loadStride2Ref(const double* ptr) noexcept
{
    __m128d v0 = _mm_loadu_pd(ptr - 6);
    __m128d v1 = _mm_loadu_pd(ptr - 4);
    __m128d v2 = _mm_loadu_pd(ptr - 2);
    __m128d v3 = _mm_loadu_pd(ptr);
    __m128d vLow = _mm_unpacklo_pd(v3, v2);
    __m128d vHigh = _mm_unpacklo_pd(v1, v0);
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(vLow), vHigh, 1);
}

//============================================================================
// dotProductDecimateAvx2Ref — production CIO.cpp:219-285 と同一帰還構造の
// 独立実装（8-way unroll + tree reduction + scalar 剰余・内部清掃なし）。
//============================================================================
inline double dotProductDecimateAvx2Ref(const double* __restrict history,
                                        const double* __restrict coeffs,
                                        int convCount) noexcept
{
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();
    __m256d acc4 = _mm256_setzero_pd();
    __m256d acc5 = _mm256_setzero_pd();
    __m256d acc6 = _mm256_setzero_pd();
    __m256d acc7 = _mm256_setzero_pd();

    int r = 0;
    const int unrollEnd = (convCount / 32) * 32;
    for (; r < unrollEnd; r += 32)
    {
        acc0 = _mm256_fmadd_pd(loadStride2Ref(history - (r << 1)),       _mm256_loadu_pd(coeffs + r),       acc0);
        acc1 = _mm256_fmadd_pd(loadStride2Ref(history - ((r +  4) << 1)), _mm256_loadu_pd(coeffs + r +  4), acc1);
        acc2 = _mm256_fmadd_pd(loadStride2Ref(history - ((r +  8) << 1)), _mm256_loadu_pd(coeffs + r +  8), acc2);
        acc3 = _mm256_fmadd_pd(loadStride2Ref(history - ((r + 12) << 1)), _mm256_loadu_pd(coeffs + r + 12), acc3);
        acc4 = _mm256_fmadd_pd(loadStride2Ref(history - ((r + 16) << 1)), _mm256_loadu_pd(coeffs + r + 16), acc4);
        acc5 = _mm256_fmadd_pd(loadStride2Ref(history - ((r + 20) << 1)), _mm256_loadu_pd(coeffs + r + 20), acc5);
        acc6 = _mm256_fmadd_pd(loadStride2Ref(history - ((r + 24) << 1)), _mm256_loadu_pd(coeffs + r + 24), acc6);
        acc7 = _mm256_fmadd_pd(loadStride2Ref(history - ((r + 28) << 1)), _mm256_loadu_pd(coeffs + r + 28), acc7);
    }

    const int simdEnd = (convCount / 4) * 4;
    for (; r < simdEnd; r += 4)
    {
        __m256d vS = loadStride2Ref(history - (r << 1));
        __m256d vC = _mm256_loadu_pd(coeffs + r);
        acc0 = _mm256_fmadd_pd(vS, vC, acc0);
    }

    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc4 = _mm256_add_pd(acc4, acc5);
    acc6 = _mm256_add_pd(acc6, acc7);
    acc0 = _mm256_add_pd(acc0, acc2);
    acc4 = _mm256_add_pd(acc4, acc6);
    acc0 = _mm256_add_pd(acc0, acc4);

    __m128d vLo = _mm256_castpd256_pd128(acc0);
    __m128d vHi = _mm256_extractf128_pd(acc0, 1);
    __m128d vSum = _mm_add_pd(vLo, vHi);
    vSum = _mm_hadd_pd(vSum, vSum);
    double result = _mm_cvtsd_f64(vSum);

    for (; r < convCount; ++r)
        result += coeffs[r] * history[-(r << 1)];

    return result;
}
#endif // __AVX2__ && __FMA__

//============================================================================
// interpolateStageRef — production CIO.cpp:492-568 と同一観測契約の独立実装。
// candidate 時: convValue *= 2.0（既存）に加えて centerValue *= 2.0（案 E）。
//============================================================================
inline void interpolateStageRef(StageModel& stage,
                                const double* input,
                                int inputSamples,
                                double* output,
                                int channel,
                                bool candidate) noexcept
{
    double* history = stage.upHistory[channel].data();
    if (history == nullptr || input == nullptr || output == nullptr)
        return;

    const int keep = stage.historyUpKeep;
    const int capacity = stage.upHistorySize;
    for (int i = 0; i < inputSamples; ++i)
        history[keep + i] = input[i];

    for (int n = 0; n < inputSamples; ++n)
    {
        const int idx = keep + n;
        if (idx < (stage.convCount - 1) || idx >= capacity)
        {
            stage.corruptionEvents += 1;
            output[n * 2 + 0] = 0.0;
            output[n * 2 + 1] = 0.0;
            continue;
        }

        const double* xWindow = history + idx - (stage.convCount - 1);
        double convValue = 0.0;
        bool bad = false;

#if defined(__AVX2__)
        if (stage.convCount >= 4)
        {
            convValue = dotProductAvx2Ref(xWindow, stage.convCoeffsReversed.data(), stage.convCount);
            if (isBadSampleRef(convValue))
                bad = true;
        }
        else
#endif
        {
            for (int r = 0; r < stage.convCount; ++r)
            {
                const double x = xWindow[r];
                if (isBadSampleRef(x))
                {
                    bad = true;
                    break;
                }
                convValue += stage.convCoeffsReversed[r] * x;
            }
        }

        double centerValue = 0.0;
        if (idx >= stage.centerDelayInput)
            centerValue = stage.centerCoeff * history[idx - stage.centerDelayInput];
        else
            bad = true;

        if (bad || isBadSampleRef(centerValue))
        {
            stage.corruptionEvents += 1;
            output[n * 2 + 0] = 0.0;
            output[n * 2 + 1] = 0.0;
            continue;
        }

        convValue *= 2.0;
        if (candidate)
            centerValue *= 2.0;   // ★ 案 E（candidate hypothesis・runtime 切替）
        if (fastAbsRef(convValue) < kDenormThresholdRef) convValue = 0.0;
        if (fastAbsRef(centerValue) < kDenormThresholdRef) centerValue = 0.0;

        const int outBase = n << 1;
        output[outBase + stage.convParity] = convValue;
        output[outBase + stage.centerParity] = centerValue;
    }

    std::memmove(history, history + inputSamples, static_cast<std::size_t>(keep) * sizeof(double));
}

//============================================================================
// decimateStageRef — production CIO.cpp:570-723 と同一観測契約の独立実装。
//============================================================================
inline void decimateStageRef(StageModel& stage,
                             const double* input,
                             int inputSamples,
                             double* output,
                             int channel) noexcept
{
    double* history = stage.downHistory[channel].data();
    if (history == nullptr || input == nullptr || output == nullptr)
        return;

    const int keep = stage.historyDownKeep;
    const int capacity = stage.downHistorySize;

    // silence fast path（production :584-613 と同一）
    bool inputSilent = true;
    for (int i = 0; i < inputSamples; ++i)
    {
        if (fastAbsRef(input[i]) > kDenormThresholdRef)
        {
            inputSilent = false;
            break;
        }
    }

    if (inputSilent)
    {
        bool historySilent = true;
        for (int i = 0; i < keep; ++i)
        {
            if (fastAbsRef(history[i]) > kDenormThresholdRef)
            {
                historySilent = false;
                break;
            }
        }

        if (historySilent)
        {
            const int outSamples = inputSamples >> 1;
            for (int i = 0; i < outSamples; ++i)
                output[i] = 0.0;
            for (int i = 0; i < keep; ++i)
                history[i] = 0.0;
            return;
        }
    }

    for (int i = 0; i < inputSamples; ++i)
        history[keep + i] = input[i];

    const int outSamples = inputSamples >> 1;
    const double* coeffs = stage.convCoeffs.data();

    if (outSamples <= 0)
        return;

    // 境界 guard（production :626-649 と同一評価・kLoadStride2Offset = 6 を含む）
    const int baseMax = keep + ((outSamples - 1) << 1);
    const bool centerTapOk = (keep >= stage.centerTap) && (baseMax < capacity);
    constexpr int kLoadStride2Offset = 6;
    const int globalMinConvIdx = keep - stage.convParity - ((stage.convCount - 1) << 1);
    const int globalMaxConvIdx = baseMax - stage.convParity;
    const int avxMinConvIdx = globalMinConvIdx - kLoadStride2Offset;
    const bool convTapOk = (avxMinConvIdx >= 0) && (globalMaxConvIdx < capacity);

    if (!centerTapOk || !convTapOk || stage.convCount <= 0)
    {
        for (int n = 0; n < outSamples; ++n)
            output[n] = 0.0;
        stage.corruptionEvents += 1;
        return;
    }

    for (int n = 0; n < outSamples; ++n)
    {
        const int base = keep + (n << 1);

        const double centerSample = history[base - stage.centerTap];
        double acc = stage.centerCoeff * centerSample;
        if (isBadSampleRef(acc))
        {
            output[n] = 0.0;
            stage.corruptionEvents += 1;
            continue;
        }

#if defined(__AVX2__) && defined(__FMA__)
        if (stage.convCount >= 8)
        {
            acc += dotProductDecimateAvx2Ref(history + (base - stage.convParity), coeffs, stage.convCount);
        }
        else if (stage.convCount >= 4)
        {
            // convCount 4〜7: production :678-699 と同一の簡易 AVX2 経路
            __m256d vAcc = _mm256_setzero_pd();
            int r = 0;
            const int simdEnd = (stage.convCount / 4) * 4;
            for (; r < simdEnd; r += 4)
            {
                __m256d vS = loadStride2Ref(history + (base - stage.convParity) - (r << 1));
                __m256d vC = _mm256_loadu_pd(coeffs + r);
                vAcc = _mm256_fmadd_pd(vS, vC, vAcc);
            }
            __m128d vLo = _mm256_castpd256_pd128(vAcc);
            __m128d vHi = _mm256_extractf128_pd(vAcc, 1);
            __m128d vSum = _mm_add_pd(vLo, vHi);
            vSum = _mm_hadd_pd(vSum, vSum);
            acc += _mm_cvtsd_f64(vSum);
            for (; r < stage.convCount; ++r)
                acc += coeffs[r] * history[base - stage.convParity - (r << 1)];
        }
        else
#endif
        {
            for (int r = 0; r < stage.convCount; ++r)
                acc += coeffs[r] * history[base - stage.convParity - (r << 1)];
        }

        if (isBadSampleRef(acc))
        {
            output[n] = 0.0;
            stage.corruptionEvents += 1;
        }
        else
        {
            if (fastAbsRef(acc) < kDenormThresholdRef) acc = 0.0;
            output[n] = acc;
        }
    }

    std::memmove(history, history + inputSamples, static_cast<std::size_t>(keep) * sizeof(double));
}

//============================================================================
// PolyphaseGainShadow — production processUp/processDown（:725-872）と同一
// 観測契約の独立 multi-stage 実装。candidate は runtime 切替（macro 既定）。
//============================================================================
class PolyphaseGainShadow
{
public:
    enum class Preset { IIRLike, LinearPhase };

    explicit PolyphaseGainShadow(bool candidateMode
                                 = (CONVOPEQ_POLYPHASE_REF_CANDIDATE != 0)) noexcept
        : candidateMode_(candidateMode) {}

    // production prepare(:415-450) と同一契約（内部 tap/attenuation 表は独立定義）。
    bool prepare(int maxInputBlockSize, int ratio, Preset preset) noexcept
    {
        release();

        maxInputBlockSize_ = (1 > maxInputBlockSize) ? 1 : maxInputBlockSize;
        upsampleRatio_ = sanitizeRatioRef(ratio);
        activePreset_ = preset;
        numStages_ = (upsampleRatio_ == 8) ? 3 : ((upsampleRatio_ == 4) ? 2 : ((upsampleRatio_ == 2) ? 1 : 0));
        if (numStages_ < 0) numStages_ = 0;
        if (numStages_ > 3) numStages_ = 3;   // stages_[3] の静的境界に対する fail-closed guard
        maxUpsampledBlockSize_ = maxInputBlockSize_ * upsampleRatio_;

        int stageInputMax = maxInputBlockSize_;
        for (int i = 0; i < numStages_; ++i)
        {
            StageModel& st = stages_[static_cast<std::size_t>(i)];
            if (!prepareStageModel(st, tapsForStageRef(i, preset), attenuationForStageRef(i, preset), stageInputMax))
            {
                release();
                return false;
            }
            stageInputMax *= 2;
        }

        workCapacity_ = (1 > maxUpsampledBlockSize_) ? 1 : maxUpsampledBlockSize_;
        for (int ch = 0; ch < kMaxChannelsRef; ++ch)
        {
            workA_[ch].assign(static_cast<std::size_t>(workCapacity_), 0.0);
            workB_[ch].assign(static_cast<std::size_t>(workCapacity_), 0.0);
        }
        prepared_ = numStages_ > 0;
        return prepared_;
    }

    // production prepareSingleStage(:392-413) と同一契約。
    bool prepareSingleStageModel(int taps, double attenDb, int stageInputMax) noexcept
    {
        release();
        upsampleRatio_ = 2;
        numStages_ = 1;
        maxInputBlockSize_ = stageInputMax;
        maxUpsampledBlockSize_ = stageInputMax * 2;
        if (!prepareStageModel(stages_[0], taps, attenDb, stageInputMax))
        {
            release();
            return false;
        }
        workCapacity_ = maxUpsampledBlockSize_;
        for (int ch = 0; ch < kMaxChannelsRef; ++ch)
        {
            workA_[ch].assign(static_cast<std::size_t>(workCapacity_), 0.0);
            workB_[ch].assign(static_cast<std::size_t>(workCapacity_), 0.0);
        }
        prepared_ = true;
        return true;
    }

    // production reset(:452-467) 契約 = 履歴全 clear + atomic 3 フラグ相当。
    void reset() noexcept
    {
        for (int i = 0; i < numStages_; ++i)
        {
            StageModel& st = stages_[static_cast<std::size_t>(i)];
            for (int ch = 0; ch < kMaxChannelsRef; ++ch)
            {
                std::fill(st.upHistory[ch].begin(), st.upHistory[ch].end(), 0.0);
                std::fill(st.downHistory[ch].begin(), st.downHistory[ch].end(), 0.0);
            }
        }
        corruptionDetected_ = false;
        consecutiveCorruptionAutoClearCount_ = 0;
        hardFallbackActive_ = false;
    }

    // production clearAllStages(:469-484) 契約 = 履歴 clear + atomic 1 のみ。
    void clearAllStages() noexcept
    {
        for (int i = 0; i < numStages_; ++i)
        {
            StageModel& st = stages_[static_cast<std::size_t>(i)];
            for (int ch = 0; ch < kMaxChannelsRef; ++ch)
            {
                std::fill(st.upHistory[ch].begin(), st.upHistory[ch].end(), 0.0);
                std::fill(st.downHistory[ch].begin(), st.downHistory[ch].end(), 0.0);
            }
        }
        corruptionDetected_ = false;
    }

    void release() noexcept
    {
        for (auto& st : stages_)
            st = StageModel{};
        for (int ch = 0; ch < kMaxChannelsRef; ++ch)
        {
            workA_[ch].clear();
            workB_[ch].clear();
        }
        workCapacity_ = 0;
        upsampleRatio_ = 1;
        numStages_ = 0;
        maxInputBlockSize_ = 0;
        maxUpsampledBlockSize_ = 0;
        prepared_ = false;
        corruptionDetected_ = false;
        consecutiveCorruptionAutoClearCount_ = 0;
    }

    [[nodiscard]] bool isPrepared() const noexcept { return prepared_; }
    [[nodiscard]] int stageCount() const noexcept { return numStages_; }
    [[nodiscard]] int ratio() const noexcept { return upsampleRatio_; }
    [[nodiscard]] std::uint64_t corruptionEventCount() const noexcept { return corruptionEventCount_; }

    // ── processUp（production :725-783 と同一観測契約） ────────────────────
    // in/out は 2 チャネル固定（kMaxChannelsRef）。戻り値 = 出力サンプル数（guard 発動時 0）。
    int processUp(const double* const* in, int numSamples, double* const* out) noexcept
    {
        if (in == nullptr || out == nullptr || numSamples <= 0)
            return 0;
        // hardFallback passthrough（production :728-738 と同一契約）
        if (hardFallbackActive_)
        {
            for (int ch = 0; ch < kMaxChannelsRef; ++ch)
                std::memcpy(out[ch], in[ch], static_cast<std::size_t>(numSamples) * sizeof(double));
            return numSamples;
        }
        if (numSamples > maxInputBlockSize_)
            return 0;   // production :744-748 safety guard と同一（空を返す）
        if (upsampleRatio_ <= 1 || numStages_ == 0)
        {
            for (int ch = 0; ch < kMaxChannelsRef; ++ch)
                std::memcpy(out[ch], in[ch], static_cast<std::size_t>(numSamples) * sizeof(double));
            return numSamples;
        }

        const double* currIn[kMaxChannelsRef] = { in[0], in[1] };
        int currSamples = numSamples;

        for (int stageIndex = 0; stageIndex < numStages_; ++stageIndex)
        {
            std::vector<double>* stageOut[kMaxChannelsRef]
                = { ((stageIndex & 1) == 0) ? &workA_[0] : &workB_[0],
                    ((stageIndex & 1) == 0) ? &workA_[1] : &workB_[1] };
            for (int ch = 0; ch < kMaxChannelsRef; ++ch)
                interpolateStageRef(stages_[static_cast<std::size_t>(stageIndex)],
                                    currIn[ch], currSamples,
                                    stageOut[ch]->data(), ch, candidateMode_);
            currIn[0] = stageOut[0]->data();
            currIn[1] = stageOut[1]->data();
            currSamples <<= 1;
        }

        for (int ch = 0; ch < kMaxChannelsRef; ++ch)
            std::memcpy(out[ch], currIn[ch], static_cast<std::size_t>(currSamples) * sizeof(double));
        return currSamples;
    }

    // ── processDown（production :785-872 と同一観測契約） ─────────────────
    int processDown(const double* const* in, int numSamples, double* const* out) noexcept
    {
        if (in == nullptr || out == nullptr || numSamples <= 0)
            return 0;

        // hardFallback passthrough（production :789-806 と同一契約）
        if (hardFallbackActive_)
        {
            for (int ch = 0; ch < kMaxChannelsRef; ++ch)
                std::memcpy(out[ch], in[ch], static_cast<std::size_t>(numSamples) * sizeof(double));
            return numSamples;
        }

        // corruption auto-clear（production :808-819 と同一契約・値には影響しない clean 経路）
        if (corruptionDetected_)
        {
            corruptionDetected_ = false;
            corruptionAutoClearCount_ += 1;
            consecutiveCorruptionAutoClearCount_ += 1;
            if (consecutiveCorruptionAutoClearCount_ >= kHardFallbackAutoClearThresholdRef)
                hardFallbackActive_ = true;
            clearAllStages();
            for (int ch = 0; ch < kMaxChannelsRef; ++ch)
                std::fill(out[ch], out[ch] + numSamples, 0.0);
            return numSamples;
        }
        consecutiveCorruptionAutoClearCount_ = 0;

        if (numSamples > maxUpsampledBlockSize_)
        {
            corruptionDetected_ = true;
            corruptionEventCount_ += 1;
            for (int ch = 0; ch < kMaxChannelsRef; ++ch)
                std::fill(out[ch], out[ch] + numSamples, 0.0);
            return numSamples;
        }

        if (upsampleRatio_ <= 1 || numStages_ == 0)
        {
            for (int ch = 0; ch < kMaxChannelsRef; ++ch)
                std::memcpy(out[ch], in[ch], static_cast<std::size_t>(numSamples) * sizeof(double));
            return numSamples;
        }

        const double* currIn[kMaxChannelsRef] = { in[0], in[1] };
        int currSamples = numSamples;

        for (int stageIndex = numStages_ - 1; stageIndex >= 0; --stageIndex)
        {
            if (stageIndex < 0 || stageIndex > 2)
            {
                // stages_[3] の静的境界に対する fail-closed guard（prepare の clamp と二重化）
                for (int ch = 0; ch < kMaxChannelsRef; ++ch)
                    std::fill(out[ch], out[ch] + numSamples, 0.0);
                corruptionDetected_ = true;
                return numSamples;
            }
            std::vector<double>* stageOut[kMaxChannelsRef]
                = { (((numStages_ - 1 - stageIndex) & 1) == 0) ? &workA_[0] : &workB_[0],
                    (((numStages_ - 1 - stageIndex) & 1) == 0) ? &workA_[1] : &workB_[1] };
            for (int ch = 0; ch < kMaxChannelsRef; ++ch)
                decimateStageRef(stages_[static_cast<std::size_t>(stageIndex)],
                                 currIn[ch], currSamples,
                                 stageOut[ch]->data(), ch);
            currIn[0] = stageOut[0]->data();
            currIn[1] = stageOut[1]->data();
            currSamples >>= 1;
        }

        for (int ch = 0; ch < kMaxChannelsRef; ++ch)
            std::memcpy(out[ch], currIn[ch], static_cast<std::size_t>(currSamples) * sizeof(double));
        return currSamples;
    }

    // ── DC 予測（R8-1 / v2.2 §2.1 継承）: base 0.75^N / cand 1.0 ───────────
    [[nodiscard]] double centerPhaseGain() const noexcept { return candidateMode_ ? 2.0 : 1.0; }

    [[nodiscard]] double expectedDcRoundTrip() const noexcept
    {
        double v = 1.0;
        if (!candidateMode_)
            for (int i = 0; i < numStages_; ++i)
                v *= 0.75;
        return v;
    }

private:
    static int sanitizeRatioRef(int ratio) noexcept
    {
        if (ratio >= 8) return 8;
        if (ratio >= 4) return 4;
        if (ratio >= 2) return 2;
        return 1;
    }

    // production tapsForStage(:84-94) / attenuationForStage(:96-106) と同一表の独立定義。
    static int tapsForStageRef(int stageIndex, Preset preset) noexcept
    {
        if (preset == Preset::LinearPhase)
        {
            static constexpr int taps[3] = { 1023, 255, 63 };
            return taps[(stageIndex < 0) ? 0 : ((stageIndex > 2) ? 2 : stageIndex)];
        }
        static constexpr int taps[3] = { 511, 127, 31 };
        return taps[(stageIndex < 0) ? 0 : ((stageIndex > 2) ? 2 : (stageIndex))];
    }

    static double attenuationForStageRef(int stageIndex, Preset preset) noexcept
    {
        if (preset == Preset::LinearPhase)
        {
            static constexpr double attenuation[3] = { 160.0, 140.0, 120.0 };
            return attenuation[(stageIndex < 0) ? 0 : ((stageIndex > 2) ? 2 : stageIndex)];
        }
        static constexpr double attenuation[3] = { 140.0, 110.0, 90.0 };
        return attenuation[(stageIndex < 0) ? 0 : ((stageIndex > 2) ? 2 : stageIndex)];
    }

    static constexpr std::uint32_t kHardFallbackAutoClearThresholdRef = 4;

    bool candidateMode_ = false;
    bool prepared_ = false;
    int upsampleRatio_ = 1;
    int numStages_ = 0;
    int maxInputBlockSize_ = 0;
    int maxUpsampledBlockSize_ = 0;
    int workCapacity_ = 0;
    Preset activePreset_ = Preset::IIRLike;

    StageModel stages_[3];
    std::vector<double> workA_[kMaxChannelsRef];
    std::vector<double> workB_[kMaxChannelsRef];

    std::uint64_t corruptionEventCount_ = 0;
    std::uint64_t corruptionAutoClearCount_ = 0;
    std::uint32_t consecutiveCorruptionAutoClearCount_ = 0;
    bool corruptionDetected_ = false;
    bool hardFallbackActive_ = false;
};

} // namespace convo::polyphase_ref
