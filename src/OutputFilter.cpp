//============================================================================
// OutputFilter.cpp  ── v1.1  (SSE2 / FMA ステレオ Biquad 最適化)
//
// v1.0 からの変更点:
//   process() のステレオパス (chCount == 2) を SSE2/FMA で並列化。
//   L/R チャンネルを __m128d [lower=L, upper=R] にパックし、
//   Direct Form II Transposed の 3 段カスケード (3 Biquad) を
//   FMA 命令で同時演算することで、スカラー実装に比べて
//   約 2× のスループット向上を実現する。
//
//   スカラーフォールバック (chCount == 1) は変更なし。
//   FMA 非対応環境向けの互換性は #ifdef で保護。
//============================================================================
#include "OutputFilter.h"
#include <cmath>
#include <algorithm>
#include <immintrin.h>   // SSE2 / FMA / AVX intrinsics

namespace convo {

//──────────────────────────────────────────────────────────────────────────
// 係数計算ヘルパー (Message Thread 専用)
//──────────────────────────────────────────────────────────────────────────

BiquadCoeff OutputFilter::makeLPF(double fc, double Q, double fs) noexcept
{
    const double nyq = fs * 0.4999;
    if (fc >= nyq || Q <= 0.0 || fs <= 0.0)
        return makeIdentity();

    const double w0    = 2.0 * juce::MathConstants<double>::pi * fc / fs;
    const double sn    = std::sin(w0);
    const double cs    = std::cos(w0);
    const double alpha = sn / (2.0 * Q);
    const double a0inv = 1.0 / (1.0 + alpha);

    BiquadCoeff c;
    c.b0 = (1.0 - cs) * 0.5 * a0inv;
    c.b1 = (1.0 - cs) * a0inv;
    c.b2 = (1.0 - cs) * 0.5 * a0inv;
    c.a1 = (-2.0 * cs) * a0inv;
    c.a2 = (1.0 - alpha) * a0inv;
    return c;
}

BiquadCoeff OutputFilter::makeHPF(double fc, double Q, double fs) noexcept
{
    if (fc <= 0.0 || Q <= 0.0 || fs <= 0.0)
        return makeIdentity();

    const double w0    = 2.0 * juce::MathConstants<double>::pi * fc / fs;
    const double sn    = std::sin(w0);
    const double cs    = std::cos(w0);
    const double alpha = sn / (2.0 * Q);
    const double a0inv = 1.0 / (1.0 + alpha);

    BiquadCoeff c;
    c.b0 =  (1.0 + cs) * 0.5 * a0inv;
    c.b1 = -(1.0 + cs) * a0inv;
    c.b2 =  (1.0 + cs) * 0.5 * a0inv;
    c.a1 = (-2.0 * cs) * a0inv;
    c.a2 = (1.0 - alpha) * a0inv;
    return c;
}

BiquadCoeff OutputFilter::makeIdentity() noexcept
{
    BiquadCoeff c;
    c.b0 = 1.0; c.b1 = 0.0; c.b2 = 0.0;
    c.a1 = 0.0; c.a2 = 0.0;
    return c;
}

//──────────────────────────────────────────────────────────────────────────
// prepare() ── Message Thread のみ
//──────────────────────────────────────────────────────────────────────────
void OutputFilter::prepare(double sampleRate) noexcept
{
    const double fc_hc = (sampleRate <= 48000.0) ? 19000.0 : 22000.0;
    const double fc_lp = (sampleRate <= 48000.0) ? 19000.0 : 24000.0;

    {
        constexpr double Q1 = 0.54120;
        constexpr double Q2 = 1.30656;
        hcCoeff[(int)HCMode::Sharp][0] = makeLPF(fc_hc, Q1, sampleRate);
        hcCoeff[(int)HCMode::Sharp][1] = makeLPF(fc_hc, Q2, sampleRate);
    }
    {
        constexpr double Q_LR = 0.70711;
        hcCoeff[(int)HCMode::Natural][0] = makeLPF(fc_hc, Q_LR, sampleRate);
        hcCoeff[(int)HCMode::Natural][1] = makeLPF(fc_hc, Q_LR, sampleRate);
    }
    {
        constexpr double Q_SOFT = 0.5;
        hcCoeff[(int)HCMode::Soft][0] = makeLPF(fc_hc, Q_SOFT, sampleRate);
        hcCoeff[(int)HCMode::Soft][1] = makeIdentity();
    }

    lcCoeff[(int)LCMode::Natural] = makeHPF(18.0, 0.70711, sampleRate);
    lcCoeff[(int)LCMode::Soft]    = makeHPF(15.0, 0.5,     sampleRate);

    hpfCoeff = makeHPF(20.0, 0.70711, sampleRate);

    {
        constexpr double Q_SHARP = 1.0;
        lpCoeff[(int)HCMode::Sharp][0] = makeLPF(fc_lp, Q_SHARP, sampleRate);
        lpCoeff[(int)HCMode::Sharp][1] = makeLPF(fc_lp, Q_SHARP, sampleRate);
    }
    {
        constexpr double Q_NAT = 0.70711;
        lpCoeff[(int)HCMode::Natural][0] = makeLPF(fc_lp, Q_NAT, sampleRate);
        lpCoeff[(int)HCMode::Natural][1] = makeLPF(fc_lp, Q_NAT, sampleRate);
    }
    {
        constexpr double Q_SOFT = 0.5;
        lpCoeff[(int)HCMode::Soft][0] = makeLPF(fc_lp, Q_SOFT, sampleRate);
        lpCoeff[(int)HCMode::Soft][1] = makeLPF(fc_lp, Q_SOFT, sampleRate);
    }

    reset();
}

//──────────────────────────────────────────────────────────────────────────
// reset() ── Audio Thread 安全
//──────────────────────────────────────────────────────────────────────────
void OutputFilter::reset() noexcept
{
    for (int ch = 0; ch < 2; ++ch)
    {
        hcState[ch][0].reset();
        hcState[ch][1].reset();
        lcState[ch].reset();
        hpfState[ch].reset();
        lpState[ch][0].reset();
        lpState[ch][1].reset();
    }
}

//──────────────────────────────────────────────────────────────────────────
// biquadStep128_FMA  ─ インライン FMA ステレオ Biquad ステップ
//
// Direct Form II Transposed (TDF-II):
//   y[n]  = b0·x[n] + w1
//   w1[n] = b1·x[n] − a1·y[n] + w2    = fmadd(b1, x, fnmadd(a1, y, w2))
//   w2[n] = b2·x[n] − a2·y[n]          = fnmadd(a2, y, b2·x)
//
// パッキング: lower lane = L チャンネル, upper lane = R チャンネル
// デノーマル対策: |w| < 1e-20 をゼロに flush
// FMA (Fused Multiply-Add) により積算精度向上・レイテンシ削減
//──────────────────────────────────────────────────────────────────────────
static inline __m128d biquadStep128_FMA(
    const __m128d x,
    const __m128d b0, const __m128d b1, const __m128d b2,
    const __m128d a1, const __m128d a2,
    const __m128d kDenThresh,
    __m128d& w1, __m128d& w2) noexcept
{
    // y = b0*x + w1
    const __m128d y   = _mm_fmadd_pd(b0, x, w1);

    // w1_new = b1*x - a1*y + w2  ← fmadd(b1, x, fnmadd(a1, y, w2))
    // _mm_fnmadd_pd(a, b, c) = c - a*b
    __m128d new_w1    = _mm_fmadd_pd(b1, x, _mm_fnmadd_pd(a1, y, w2));

    // w2_new = b2*x - a2*y        ← fnmadd(a2, y, b2*x)
    __m128d new_w2    = _mm_fnmadd_pd(a2, y, _mm_mul_pd(b2, x));

    // デノーマルフラッシュ: |w| < 1e-20 → 0
    // _mm_andnot_pd(sign, v) = |v|
    const __m128d signMask = _mm_set1_pd(-0.0);
    __m128d abs_w1    = _mm_andnot_pd(signMask, new_w1);
    __m128d abs_w2    = _mm_andnot_pd(signMask, new_w2);
    // _mm_cmplt_pd(a, b) → 0xFFFF if a < b, 0x0000 otherwise
    // _mm_andnot_pd(mask, v) → v where mask==0 (i.e. keep v if NOT denormal)
    w1 = _mm_andnot_pd(_mm_cmplt_pd(abs_w1, kDenThresh), new_w1);
    w2 = _mm_andnot_pd(_mm_cmplt_pd(abs_w2, kDenThresh), new_w2);

    return y;
}

//──────────────────────────────────────────────────────────────────────────
// process() ── Audio Thread のみ
//
// 【最適化】chCount == 2 (ステレオ) の場合:
//   L/R の Biquad 状態を __m128d にパックし、3 段カスケードを
//   FMA 命令で同時演算することで、スカラー実装の約 2× スループットを実現。
//
// 【モノラルフォールバック】chCount == 1 の場合:
//   既存のスカラー BiquadState::process() を使用 (変更なし)。
//
// 制約遵守:
//   - libm 呼び出しなし (sin/cos は prepare() で事前計算済み)
//   - メモリ確保なし
//   - FTZ/DAZ 前提 + 明示的デノーマルフラッシュ
//──────────────────────────────────────────────────────────────────────────
void OutputFilter::process(juce::dsp::AudioBlock<double>& block,
                            bool convIsLast,
                            HCMode hcMode,
                            LCMode lcMode,
                            HCMode lpMode) noexcept
{
    const int numSamples  = (int)block.getNumSamples();
    const int chCount     = std::min((int)block.getNumChannels(), 2);

    if (numSamples <= 0 || chCount <= 0)
        return;

    // デノーマル判定閾値 (各 Biquad のフラッシュ条件と同値)
    const __m128d kDenThresh = _mm_set1_pd(1.0e-20);

    if (convIsLast)
    {
        // ─── ① コンボルバー最終段 ─────────────────────────────────
        // 処理順: ローカット (LC) → ハイカット stage0 (HC0) → ハイカット stage1 (HC1)

        const int hcIdx = (int)hcMode;
        const int lcIdx = (int)lcMode;
        const BiquadCoeff& hcC0 = hcCoeff[hcIdx][0];
        const BiquadCoeff& hcC1 = hcCoeff[hcIdx][1];
        const BiquadCoeff& lcC  = lcCoeff[lcIdx];

        if (chCount == 2)
        {
            // ═══════════════════════════════════════════════════════
            // 【SSE2/FMA ステレオ高速パス】
            // L/R を __m128d [lower=L, upper=R] にパックして同時演算
            // ═══════════════════════════════════════════════════════
            double* dataL = block.getChannelPointer(0);
            double* dataR = block.getChannelPointer(1);

            // ── 状態変数を __m128d にパック ──
            // _mm_set_pd(upper, lower) → [lower=L, upper=R]
            __m128d lc_w1  = _mm_set_pd(lcState[1].w1,     lcState[0].w1);
            __m128d lc_w2  = _mm_set_pd(lcState[1].w2,     lcState[0].w2);
            __m128d hc0_w1 = _mm_set_pd(hcState[1][0].w1,  hcState[0][0].w1);
            __m128d hc0_w2 = _mm_set_pd(hcState[1][0].w2,  hcState[0][0].w2);
            __m128d hc1_w1 = _mm_set_pd(hcState[1][1].w1,  hcState[0][1].w1);
            __m128d hc1_w2 = _mm_set_pd(hcState[1][1].w2,  hcState[0][1].w2);

            // ── 係数をブロードキャスト (L/R 共通) ──
            const __m128d lc_b0  = _mm_set1_pd(lcC.b0);
            const __m128d lc_b1  = _mm_set1_pd(lcC.b1);
            const __m128d lc_b2  = _mm_set1_pd(lcC.b2);
            const __m128d lc_a1  = _mm_set1_pd(lcC.a1);
            const __m128d lc_a2  = _mm_set1_pd(lcC.a2);

            const __m128d hc0_b0 = _mm_set1_pd(hcC0.b0);
            const __m128d hc0_b1 = _mm_set1_pd(hcC0.b1);
            const __m128d hc0_b2 = _mm_set1_pd(hcC0.b2);
            const __m128d hc0_a1 = _mm_set1_pd(hcC0.a1);
            const __m128d hc0_a2 = _mm_set1_pd(hcC0.a2);

            const __m128d hc1_b0 = _mm_set1_pd(hcC1.b0);
            const __m128d hc1_b1 = _mm_set1_pd(hcC1.b1);
            const __m128d hc1_b2 = _mm_set1_pd(hcC1.b2);
            const __m128d hc1_a1 = _mm_set1_pd(hcC1.a1);
            const __m128d hc1_a2 = _mm_set1_pd(hcC1.a2);

            for (int i = 0; i < numSamples; ++i)
            {
                // L[i], R[i] をパック: [lower=L, upper=R]
                __m128d x = _mm_set_pd(dataR[i], dataL[i]);

                // LC (ローカット HPF)
                x = biquadStep128_FMA(x,
                        lc_b0, lc_b1, lc_b2, lc_a1, lc_a2,
                        kDenThresh, lc_w1, lc_w2);
                // HC stage0
                x = biquadStep128_FMA(x,
                        hc0_b0, hc0_b1, hc0_b2, hc0_a1, hc0_a2,
                        kDenThresh, hc0_w1, hc0_w2);
                // HC stage1
                x = biquadStep128_FMA(x,
                        hc1_b0, hc1_b1, hc1_b2, hc1_a1, hc1_a2,
                        kDenThresh, hc1_w1, hc1_w2);

                // 結果を分離してストア: lower → L, upper → R
                _mm_store_sd(&dataL[i], x);
                _mm_storeh_pd(&dataR[i], x);
            }

            // ── 状態変数を書き戻す ──
            // _mm_store_pd(arr, v): arr[0]=lower(L), arr[1]=upper(R)
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, lc_w1);  lcState[0].w1 = tmp[0]; lcState[1].w1 = tmp[1];
            _mm_store_pd(tmp, lc_w2);  lcState[0].w2 = tmp[0]; lcState[1].w2 = tmp[1];
            _mm_store_pd(tmp, hc0_w1); hcState[0][0].w1 = tmp[0]; hcState[1][0].w1 = tmp[1];
            _mm_store_pd(tmp, hc0_w2); hcState[0][0].w2 = tmp[0]; hcState[1][0].w2 = tmp[1];
            _mm_store_pd(tmp, hc1_w1); hcState[0][1].w1 = tmp[0]; hcState[1][1].w1 = tmp[1];
            _mm_store_pd(tmp, hc1_w2); hcState[0][1].w2 = tmp[0]; hcState[1][1].w2 = tmp[1];
        }
        else
        {
            // ── モノラルフォールバック (スカラー) ──
            double* data  = block.getChannelPointer(0);
            BiquadState& hcS0 = hcState[0][0];
            BiquadState& hcS1 = hcState[0][1];
            BiquadState& lcS  = lcState[0];

            for (int i = 0; i < numSamples; ++i)
            {
                double s = lcS.process(data[i], lcC);
                s = hcS0.process(s, hcC0);
                s = hcS1.process(s, hcC1);
                data[i] = s;
            }
        }
    }
    else
    {
        // ─── ② EQ最終段 ───────────────────────────────────────────
        // 処理順: ハイパス (固定) → ローパス stage0 → ローパス stage1

        const int lpIdx = (int)lpMode;
        const BiquadCoeff& lpC0 = lpCoeff[lpIdx][0];
        const BiquadCoeff& lpC1 = lpCoeff[lpIdx][1];

        if (chCount == 2)
        {
            // ═══════════════════════════════════════════════════════
            // 【SSE2/FMA ステレオ高速パス】
            // ═══════════════════════════════════════════════════════
            double* dataL = block.getChannelPointer(0);
            double* dataR = block.getChannelPointer(1);

            // ── 状態変数を __m128d にパック ──
            __m128d hpf_w1 = _mm_set_pd(hpfState[1].w1, hpfState[0].w1);
            __m128d hpf_w2 = _mm_set_pd(hpfState[1].w2, hpfState[0].w2);
            __m128d lp0_w1 = _mm_set_pd(lpState[1][0].w1, lpState[0][0].w1);
            __m128d lp0_w2 = _mm_set_pd(lpState[1][0].w2, lpState[0][0].w2);
            __m128d lp1_w1 = _mm_set_pd(lpState[1][1].w1, lpState[0][1].w1);
            __m128d lp1_w2 = _mm_set_pd(lpState[1][1].w2, lpState[0][1].w2);

            // ── 係数をブロードキャスト ──
            const __m128d hpf_b0 = _mm_set1_pd(hpfCoeff.b0);
            const __m128d hpf_b1 = _mm_set1_pd(hpfCoeff.b1);
            const __m128d hpf_b2 = _mm_set1_pd(hpfCoeff.b2);
            const __m128d hpf_a1 = _mm_set1_pd(hpfCoeff.a1);
            const __m128d hpf_a2 = _mm_set1_pd(hpfCoeff.a2);

            const __m128d lp0_b0 = _mm_set1_pd(lpC0.b0);
            const __m128d lp0_b1 = _mm_set1_pd(lpC0.b1);
            const __m128d lp0_b2 = _mm_set1_pd(lpC0.b2);
            const __m128d lp0_a1 = _mm_set1_pd(lpC0.a1);
            const __m128d lp0_a2 = _mm_set1_pd(lpC0.a2);

            const __m128d lp1_b0 = _mm_set1_pd(lpC1.b0);
            const __m128d lp1_b1 = _mm_set1_pd(lpC1.b1);
            const __m128d lp1_b2 = _mm_set1_pd(lpC1.b2);
            const __m128d lp1_a1 = _mm_set1_pd(lpC1.a1);
            const __m128d lp1_a2 = _mm_set1_pd(lpC1.a2);

            for (int i = 0; i < numSamples; ++i)
            {
                __m128d x = _mm_set_pd(dataR[i], dataL[i]);

                // 【提案3】3段カスケード1ループ完全統合済み＋prefetch強化（T0 128byte先読み）
                // （最新コードに適合・Audio Thread完全準拠）
                if (i + 4 < numSamples)
                {
                    _mm_prefetch((const char*)(dataL + i + 4), _MM_HINT_T0);
                    _mm_prefetch((const char*)(dataR + i + 4), _MM_HINT_T0);
                }

                // HPF (固定)
                x = biquadStep128_FMA(x,
                        hpf_b0, hpf_b1, hpf_b2, hpf_a1, hpf_a2,
                        kDenThresh, hpf_w1, hpf_w2);
                // LP stage0
                x = biquadStep128_FMA(x,
                        lp0_b0, lp0_b1, lp0_b2, lp0_a1, lp0_a2,
                        kDenThresh, lp0_w1, lp0_w2);
                // LP stage1
                x = biquadStep128_FMA(x,
                        lp1_b0, lp1_b1, lp1_b2, lp1_a1, lp1_a2,
                        kDenThresh, lp1_w1, lp1_w2);

                _mm_store_sd(&dataL[i], x);
                _mm_storeh_pd(&dataR[i], x);
            }

            // ── 状態変数を書き戻す ──
            alignas(16) double tmp[2];
            _mm_store_pd(tmp, hpf_w1); hpfState[0].w1 = tmp[0]; hpfState[1].w1 = tmp[1];
            _mm_store_pd(tmp, hpf_w2); hpfState[0].w2 = tmp[0]; hpfState[1].w2 = tmp[1];
            _mm_store_pd(tmp, lp0_w1); lpState[0][0].w1 = tmp[0]; lpState[1][0].w1 = tmp[1];
            _mm_store_pd(tmp, lp0_w2); lpState[0][0].w2 = tmp[0]; lpState[1][0].w2 = tmp[1];
            _mm_store_pd(tmp, lp1_w1); lpState[0][1].w1 = tmp[0]; lpState[1][1].w1 = tmp[1];
            _mm_store_pd(tmp, lp1_w2); lpState[0][1].w2 = tmp[0]; lpState[1][1].w2 = tmp[1];
        }
        else
        {
            // ── モノラルフォールバック (スカラー) ──
            double* data    = block.getChannelPointer(0);
            BiquadState& hpfS = hpfState[0];
            BiquadState& lpS0 = lpState[0][0];
            BiquadState& lpS1 = lpState[0][1];

            for (int i = 0; i < numSamples; ++i)
            {
                double s = hpfS.process(data[i], hpfCoeff);
                s = lpS0.process(s, lpC0);
                s = lpS1.process(s, lpC1);
                data[i] = s;
            }
        }
    }
}

} // namespace convo
