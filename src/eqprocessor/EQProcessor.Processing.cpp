//============================================================================
// EQProcessor.Processing.cpp
//============================================================================
#include "EQProcessor.h"
#include "DspNumericPolicy.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "core/RCUReader.h"
#include "dsp/math/FastTanhApprox.h"

#include "audioengine/AtomicAccess.h"

#if defined(__AVX2__) || defined(__FMA__)
 #include <immintrin.h>
#endif

namespace
{
    inline double calculateRMS(const double* data, int numSamples) noexcept
{
    if (data == nullptr || numSamples <= 0)
        return 0.0;

    double sumSq = 0.0;
#if defined(__AVX2__)
    int i = 0;
    const int vEnd = numSamples / 4 * 4;
    __m256d vSumSq = _mm256_setzero_pd();
    for (; i < vEnd; i += 4)
    {
        __m256d vData = _mm256_loadu_pd(data + i);
        vSumSq = _mm256_fmadd_pd(vData, vData, vSumSq);
    }
    alignas(32) double temp[4];
    _mm256_store_pd(temp, vSumSq);
    sumSq = temp[0] + temp[1] + temp[2] + temp[3];
    for (; i < numSamples; ++i)
        sumSq += data[i] * data[i];
#else
    for (int i = 0; i < numSamples; ++i)
        sumSq += data[i] * data[i];
#endif

    __m128d n = _mm_set_sd(static_cast<double>(numSamples));
    __m128d vSumSqSd = _mm_set_sd(sumSq);
    __m128d vRms = _mm_sqrt_sd(_mm_setzero_pd(), _mm_div_sd(vSumSqSd, n));
    double rms;
    _mm_store_sd(&rms, vRms);
    return rms;
}

    inline double absNoLibm(double value) noexcept
    {
        // ISR: std::bit_cast の中間変数形式（union UB 排除）
        auto bits = std::bit_cast<uint64_t>(value);
        bits &= 0x7FFFFFFFFFFFFFFFULL;
        return std::bit_cast<double>(bits);
    }

    inline bool isFiniteNoLibm(double value) noexcept
    {
        const __m128d v = _mm_set1_pd(value);
        const __m128d diff = _mm_sub_pd(v, v);
        const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());
        return _mm_movemask_pd(finiteMask) == 0x3;
    }

    inline bool isFiniteAndAbsInRangeMask(double value, double minAbsInclusive, double maxAbsExclusive) noexcept
    {
        const __m128d v = _mm_set1_pd(value);
        const __m128d diff = _mm_sub_pd(v, v);
        const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());

        const __m128d signMask = _mm_set1_pd(-0.0);
        const __m128d absV = _mm_andnot_pd(signMask, v);
        const __m128d minV = _mm_set1_pd(minAbsInclusive);
        const __m128d maxV = _mm_set1_pd(maxAbsExclusive);
        const __m128d geMinMask = _mm_cmpge_pd(absV, minV);
        const __m128d ltMaxMask = _mm_cmplt_pd(absV, maxV);

        const __m128d validMask = _mm_and_pd(finiteMask, _mm_and_pd(geMinMask, ltMaxMask));
        return _mm_movemask_pd(validMask) == 0x3;
    }

    // ── SSE2 ベクトル NaN/Inf 範囲チェック ──
    // isFiniteAndAbsInRangeMask の __m128d ベクトル版。
    // NaN/Inf または範囲外の要素は 0.0 に変換し、有効な要素はそのまま返す。
    inline __m128d sanitizeFiniteInRangeV(
        __m128d value, __m128d minAbsInclusive, __m128d maxAbsExclusive) noexcept
    {
        const __m128d diff = _mm_sub_pd(value, value);
        const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());
        const __m128d signMask = _mm_set1_pd(-0.0);
        const __m128d absV = _mm_andnot_pd(signMask, value);
        const __m128d geMinMask = _mm_cmpge_pd(absV, minAbsInclusive);
        const __m128d ltMaxMask = _mm_cmplt_pd(absV, maxAbsExclusive);
        const __m128d validMask = _mm_and_pd(finiteMask, _mm_and_pd(geMinMask, ltMaxMask));
        return _mm_and_pd(value, validMask);
    }

    // fastTanh（出力用）— ★ Bug3: 共通 Utility convo::dsp::fastTanh に委譲
    //   係数は現行の 27/9 を維持（DefaultFastTanhPolicy）。
    //   Padé 近似の変更（5次/6次）は別チケットで実施。
    inline double fastTanhScalarOutput(double x) noexcept
    {
        return convo::dsp::fastTanh<>(x);
    }

    inline __m128d fastTanhV128Output(__m128d x) noexcept
    {
        return convo::dsp::fastTanhV128<>(x);
    }

    inline double equalPowerSin(double x) noexcept
    {
        const double t = x * (juce::MathConstants<double>::pi * 0.5);
        const double t2 = t * t;
        return t * (1.0 + t2 * (-1.0 / 6.0 + t2 * (1.0 / 120.0 + t2 * (-1.0 / 5040.0 + t2 * (1.0 / 362880.0)))));
    }

//--------------------------------------------------------------
// 単一チャンネル・単一バンドのフィルタ処理 (TPT SVF)
// Topology-Preserving Transform State Variable Filter
// 参照: Vadim Zavalishin "The Art of VA Filter Design"
//--------------------------------------------------------------
    inline void processBand (double* data, int numSamples,
                             const EQCoeffsSVF& c,
                             double* state,
                             double saturation)
    {
        double ic1eq = state[0];
        double ic2eq = state[1];

        const double a1 = c.a1;
        const double a2 = c.a2;
        const double a3 = c.a3;
        const double m0 = c.m0;
        const double m1 = c.m1;
        const double m2 = c.m2;

        [[maybe_unused]] constexpr double DENORMAL_THRESHOLD = convo::numeric_policy::kDenormThresholdAudioState;

        for (int n = 0; n < numSamples; ++n)
        {
            const double v0 = data[n];
            const double v3 = v0 - ic2eq;
            const double v1 = a1 * ic1eq + a2 * v3;
            const double v2 = ic2eq + a2 * ic1eq + a3 * v3;

            ic1eq = 2.0 * v1 - ic1eq;
            ic2eq = 2.0 * v2 - ic2eq;

            double output = m0 * v0 + m1 * v1 + m2 * v2;

            if (saturation > 0.0)
            {
                const double oneMinusSat = 1.0 - saturation;
                output = output * oneMinusSat + fastTanhScalarOutput(output) * saturation;
            }

            // NaN/Infチェックとクランプを追加 (processBandStereoと一貫性を保つ)
            if (!isFiniteAndAbsInRangeMask(output, 0.0, 1.0e15))
                output = 0.0;

            // 出力もクランプして発散を防ぐ
            data[n] = std::clamp(output, -100.0, 100.0);

            // 状態変数が Inf/NaN に発散した場合は即座にリセットして次サンプルへの伝播を遮断する。
            // FTZ/DAZ 有効下でも Inf は flush されないため、この防衛は省略できない。
            // 条件式 !(abs < 閾値) は NaN に対しても true を返すため NaN/Inf を一括で捕捉する。
            // ループ内で状態変数のみをチェックする (出力データへのクランプより低コスト)。
            if (!isFiniteAndAbsInRangeMask(ic1eq, 0.0, 1.0e15)) ic1eq = 0.0;
            if (!isFiniteAndAbsInRangeMask(ic2eq, 0.0, 1.0e15)) ic2eq = 0.0;
        }

        // Denormal対策 & NaNチェック
        // Note: ScopedNoDenormals (DAZ/FTZ) が有効な場合でも、完全な0にならないと
        // 極小値が循環し続ける可能性があるため、明示的にフラッシュして計算負荷を抑える。
        ic1eq = killDenormal(ic1eq);
        ic2eq = killDenormal(ic2eq);

        state[0] = ic1eq;
        state[1] = ic2eq;
    }

    // ── 追加: Stereo 2ch 同時処理 (SSE2 / AVX2 FMA) ──
    // L, R が完全に独立した IIR 状態を持つため、128-bit レジスタに
    // [L_value, R_value] をパックして同時演算し、メモリ帯域を節約する。
    inline void processBandStereo(double* __restrict dataL,
                                   double* __restrict dataR,
                                   int numSamples,
                                   const EQCoeffsSVF& c,
                                   double* __restrict stateL,
                                   double* __restrict stateR,
                                   double saturation) noexcept
    {
        // フィルタ状態を __m128d にパック: lower=L, upper=R
        __m128d ic1eq = _mm_set_pd(stateR[0], stateL[0]);
        __m128d ic2eq = _mm_set_pd(stateR[1], stateL[1]);

        const __m128d a1  = _mm_set1_pd(c.a1);
        const __m128d a2  = _mm_set1_pd(c.a2);
        const __m128d a3  = _mm_set1_pd(c.a3);
        const __m128d m0  = _mm_set1_pd(c.m0);
        const __m128d m1  = _mm_set1_pd(c.m1);
        const __m128d m2  = _mm_set1_pd(c.m2);
        const __m128d two = _mm_set1_pd(2.0);
        const __m128d cHigh = _mm_set1_pd(100.0);
        const __m128d cLow  = _mm_set1_pd(-100.0);

        [[maybe_unused]] constexpr double DENORMAL_THRESHOLD = convo::numeric_policy::kDenormThresholdAudioState;

        for (int n = 0; n < numSamples; ++n)
        {
            // 【提案2強化版】AVX2相当のprefetch + 128byte先読み（L1キャッシュ最適化）
            // 最新コードのprocessBandStereoに適合した形で追加（__m128dベースを維持しつつ帯域向上）
            if (n + 8 < numSamples)
            {
                _mm_prefetch(reinterpret_cast<const char*>(dataL + n + 8), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(dataR + n + 8), _MM_HINT_T0);
            }

            // L[n] と R[n] を同時ロード
            const __m128d v0 = _mm_set_pd(dataR[n], dataL[n]);

            const __m128d v3 = _mm_sub_pd(v0, ic2eq);
            // FMA: a1*ic1eq + a2*v3
            const __m128d v1 = _mm_fmadd_pd(a1, ic1eq, _mm_mul_pd(a2, v3));
            // FMA: ic2eq + a2*ic1eq + a3*v3
            const __m128d v2 = _mm_fmadd_pd(a2, ic1eq,
                                _mm_fmadd_pd(a3, v3, ic2eq));

            ic1eq = _mm_fmsub_pd(two, v1, ic1eq);  // 2*v1 - ic1eq
            ic2eq = _mm_fmsub_pd(two, v2, ic2eq);  // 2*v2 - ic2eq

            // FMA: m0*v0 + m1*v1 + m2*v2
            __m128d output = _mm_fmadd_pd(m0, v0,
                              _mm_fmadd_pd(m1, v1,
                               _mm_mul_pd(m2, v2)));

            if (saturation > 0.0)
            {
                const __m128d vSat = _mm_set1_pd(saturation);
                const __m128d vOneMinusSat = _mm_set1_pd(1.0 - saturation);
                output = _mm_add_pd(_mm_mul_pd(output, vOneMinusSat),
                                    _mm_mul_pd(fastTanhV128Output(output), vSat));
            }

            // NaN/Infチェック + 範囲クランプ (processBand と一貫性を保つ)
            {
                const __m128d vMinZero = _mm_setzero_pd();
                const __m128d vMaxRange = _mm_set1_pd(1.0e15);
                output = sanitizeFiniteInRangeV(output, vMinZero, vMaxRange);
                // ★ state 変数 NaN/Inf ガード (processBand と同等)
                ic1eq = sanitizeFiniteInRangeV(ic1eq, vMinZero, vMaxRange);
                ic2eq = sanitizeFiniteInRangeV(ic2eq, vMinZero, vMaxRange);
            }

            // クランプ (-100, +100) で発散防止
            output = _mm_min_pd(_mm_max_pd(output, cLow), cHigh);

            // L: lower element, R: upper element
            _mm_store_sd(&dataL[n], output);
            _mm_storeh_pd(&dataR[n], output);
        }

        // Denormal フラッシュ & NaN チェック (状態変数のみ) - SIMD最適化
        ic1eq = killDenormalV(ic1eq);
        ic2eq = killDenormalV(ic2eq);

        // 状態を書き戻す (L/Rチャンネルに分離)
        _mm_storeu_pd(stateL, _mm_unpacklo_pd(ic1eq, ic2eq)); // [ic1eq_L, ic2eq_L]
        _mm_storeu_pd(stateR, _mm_unpackhi_pd(ic1eq, ic2eq)); // [ic1eq_R, ic2eq_R]
    }

    // ── 追加: AVX2 Gain Ramp ──
    inline void applyGainRamp_AVX2(double* __restrict data, int numSamples,
                                     double startGain, double increment) noexcept
    {
        // 各レーンの初期ゲイン: [g0, g0+inc, g0+2*inc, g0+3*inc]
        __m256d vGain = _mm256_set_pd(startGain + 3.0 * increment,
                                       startGain + 2.0 * increment,
                                       startGain + increment,
                                       startGain);
        const __m256d vInc4 = _mm256_set1_pd(4.0 * increment);
        const __m256d vInc16 = _mm256_set1_pd(16.0 * increment);

        int i = 0;
        const int vEnd16 = numSamples / 16 * 16;
        const int vEnd4 = numSamples / 4 * 4;
        for (; i < vEnd16; i += 16)
        {
            _mm_prefetch(reinterpret_cast<const char*>(data + i + 64), _MM_HINT_T0);

            // 1
            __m256d vData0 = _mm256_loadu_pd(data + i);
            __m256d vOut0  = _mm256_mul_pd(vData0, vGain);
            _mm256_storeu_pd(data + i, vOut0);
            __m256d vGain1 = _mm256_add_pd(vGain, vInc4);

            // 2
            __m256d vData1 = _mm256_loadu_pd(data + i + 4);
            __m256d vOut1  = _mm256_mul_pd(vData1, vGain1);
            _mm256_storeu_pd(data + i + 4, vOut1);
            __m256d vGain2 = _mm256_add_pd(vGain1, vInc4);

            // 3
            __m256d vData2 = _mm256_loadu_pd(data + i + 8);
            __m256d vOut2  = _mm256_mul_pd(vData2, vGain2);
            _mm256_storeu_pd(data + i + 8, vOut2);
            __m256d vGain3 = _mm256_add_pd(vGain2, vInc4);

            // 4
            __m256d vData3 = _mm256_loadu_pd(data + i + 12);
            __m256d vOut3  = _mm256_mul_pd(vData3, vGain3);
            _mm256_storeu_pd(data + i + 12, vOut3);

            vGain = _mm256_add_pd(vGain, vInc16);
        }
        // Remaining
        for (; i < vEnd4; i += 4)
        {
            __m256d vData = _mm256_loadu_pd(data + i);
            __m256d vOut  = _mm256_mul_pd(vData, vGain);
            _mm256_storeu_pd(data + i, vOut);
            vGain = _mm256_add_pd(vGain, vInc4);
        }
        // スカラー残余
        double gain = startGain + static_cast<double>(i) * increment;
        for (; i < numSamples; ++i) { data[i] *= gain; gain += increment; }
    }
}

//--------------------------------------------------------------
// AGCゲイン計算 (Private)
//--------------------------------------------------------------
double EQProcessor::calculateAGCGain(double inputEnv, double outputEnv) const noexcept
{
    constexpr double MIN_ENV = 1e-6;
    if (outputEnv < MIN_ENV) return 1.0;

    const double ratio = inputEnv / outputEnv;

    // ヒステリシス帯（±0.5dB相当）の導入
    // 0.5dB ≒ 1.059 の比率（20 * log10(1.059) ≈ 0.5dB）
    constexpr double DEAD_ZONE_RATIO = 1.059;
    if (ratio > 1.0 / DEAD_ZONE_RATIO && ratio < DEAD_ZONE_RATIO)
        return 1.0;  // 微小変動は無視

    // ゲイン制限（線形領域で直接適用）
    return juce::jlimit(static_cast<double>(AGC_MIN_GAIN), static_cast<double>(AGC_MAX_GAIN), ratio);
}

//--------------------------------------------------------------
// AGC処理 (Private)
// ★ [P2-3] 注意: AGC はブロックレート（コールバック単位）で RMS エンベロープを更新する。
//   アタック/リリースの実効時間分解能はブロックサイズに依存する。
//   ブロックサイズが大きい（1024 等）場合 + 速いアタック設定では、
//   意図通りの追従速度にならない可能性がある。
//--------------------------------------------------------------
void EQProcessor::processAGC(juce::dsp::AudioBlock<double>& block)
{
    const int numChannels = std::min((int)block.getNumChannels(), MAX_CHANNELS);
    const int numSamples = (int)block.getNumSamples();

    const double attackCoeff = convo::consumeAtomic(agcAttackCoeff, std::memory_order_acquire);   // acquire: prepareToPlay の publishAtomic release と HB
    const double releaseCoeff = convo::consumeAtomic(agcReleaseCoeff, std::memory_order_acquire); // acquire: prepareToPlay の publishAtomic release と HB
    const double smoothCoeff = convo::consumeAtomic(agcSmoothCoeff, std::memory_order_acquire);   // acquire: prepareToPlay の publishAtomic release と HB

    double blockAttackCoeff;
    double blockReleaseCoeff;
    double blockSmoothCoeff;

    const double* activeAttackCoeffTable = agcAttackCoeffTable.get();
    const double* activeReleaseCoeffTable = agcReleaseCoeffTable.get();
    const double* activeSmoothCoeffTable = agcSmoothCoeffTable.get();
    const int activeCoeffTableCapacity = agcCoeffTableCapacity;

    if (numSamples >= 0
        && numSamples < activeCoeffTableCapacity
        && activeAttackCoeffTable
        && activeReleaseCoeffTable
        && activeSmoothCoeffTable)
    {
        blockAttackCoeff = activeAttackCoeffTable[numSamples];
        blockReleaseCoeff = activeReleaseCoeffTable[numSamples];
        blockSmoothCoeff = activeSmoothCoeffTable[numSamples];
    }
    else
    {
        const double attackEpsilon = 1.0 - attackCoeff;
        const double releaseEpsilon = 1.0 - releaseCoeff;
        const double smoothEpsilon = 1.0 - smoothCoeff;
        blockAttackCoeff = std::min(1.0, static_cast<double>(numSamples) * attackEpsilon);
        blockReleaseCoeff = std::min(1.0, static_cast<double>(numSamples) * releaseEpsilon);
        blockSmoothCoeff = std::min(1.0, static_cast<double>(numSamples) * smoothEpsilon);
    }

    double inputRMS = cachedInputRMS;
    double outputRMS = 0.0;
    for (int ch = 0; ch < numChannels; ++ch)
    {
        const double* data = block.getChannelPointer(ch);
        const double rms = calculateRMS(data, numSamples);
        if (rms > outputRMS)
            outputRMS = rms;
    }

    static constexpr double MAX_ENV_VALUE = 1000.0;
    if (!isFiniteNoLibm(inputRMS) || inputRMS > MAX_ENV_VALUE)   inputRMS = MAX_ENV_VALUE;
    if (!isFiniteNoLibm(outputRMS) || outputRMS > MAX_ENV_VALUE) outputRMS = MAX_ENV_VALUE;

    double envIn = rtAgcEnvInputShadow;
    double envOut = rtAgcEnvOutputShadow;
    double currentGain = rtAgcCurrentGainShadow;
    if (!isFiniteNoLibm(envIn))  envIn = 0.0;
    if (!isFiniteNoLibm(envOut)) envOut = 0.0;
    if (!isFiniteNoLibm(currentGain)) currentGain = 1.0;

    const double inAlpha = (inputRMS > envIn) ? blockAttackCoeff : blockReleaseCoeff;
    const double outAlpha = (outputRMS > envOut) ? blockAttackCoeff : blockReleaseCoeff;
    envIn = envIn * (1.0 - inAlpha) + inputRMS * inAlpha;
    envOut = envOut * (1.0 - outAlpha) + outputRMS * outAlpha;

    constexpr double DENORM_THRESH = convo::numeric_policy::kDenormThresholdAudioState;
    if (envIn < DENORM_THRESH) envIn = 0.0;
    if (envOut < DENORM_THRESH) envOut = 0.0;

    const double targetGain = calculateAGCGain(envIn, envOut);
    const double nextGain = currentGain * (1.0 - blockSmoothCoeff) + targetGain * blockSmoothCoeff;

    rtAgcEnvInputShadow = envIn;
    rtAgcEnvOutputShadow = envOut;
    rtAgcCurrentGainShadow = nextGain;

    const double gainIncrement = (nextGain - currentGain) / static_cast<double>(numSamples);
    for (int ch = 0; ch < numChannels; ++ch)
        applyGainRamp_AVX2(block.getChannelPointer(ch), numSamples, currentGain, gainIncrement);
}

//--------------------------------------------------------------
// サイレンス検出 (Private)
//--------------------------------------------------------------
bool EQProcessor::isBufferSilent(const juce::AudioBuffer<double>& buffer, int numSamples) const noexcept
{
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        if (buffer.getMagnitude(ch, 0, numSamples) > 1.0e-8)
            return false;
    }
    return true;
}

bool EQProcessor::isAudioBlockSilent(const juce::dsp::AudioBlock<double>& block, int numChannels, int numSamples) const noexcept
{
    constexpr double silenceThreshold = 1.0e-8;

    for (int ch = 0; ch < numChannels; ++ch)
    {
        const double* channelData = block.getChannelPointer(ch);
        for (int i = 0; i < numSamples; ++i)
        {
            if (absNoLibm(channelData[i]) > silenceThreshold)
                return false;
        }
    }

    return true;
}

//--------------------------------------------------------------
// process (Audio Thread)
// リアルタイム制約 (Real-time Constraints)
//    - メモリ確保なし (No Malloc)
//    - ロックなし (No Lock)
//    - ファイルI/Oなし (No I/O)
//    - 待機なし (No Wait): 処理落ちの原因となるため (Audio Threadでの待機は厳禁)
//    - RCU (Read-Copy-Update) パターンにより、ロックフリーで安全に係数を更新
//--------------------------------------------------------------
void EQProcessor::process(juce::dsp::AudioBlock<double>& block)
{
    convo::RCUReaderGuard guard(rcuReader);
    const auto* stateSnapshot = loadCurrentState(std::memory_order_acquire); // acquire: exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    // Audio Thread 入口で MXCSR の FTZ/DAZ を関数スコープで保証する。
    // 呼び出し元設定に依存せず、EQ 単体でもデノーマル起因の負荷増大を防ぐ。
    juce::ScopedNoDenormals noDenormals;

    // m_rtBypassShadow は AudioEngine::DSPCore::processDoubleToBuffer/
    // processFloatToBuffer 内で state.eqBypassed (RuntimeSnapshot由来) から
    // setBypassFromRT() 経由で毎ブロック設定される。
    // 初期値 false (= 非バイパス) は初回 process() 呼び出し前に DSPCore が
    // 設定するまで有効であり、その間に process() が呼ばれることはない（H-01）。
    const bool requestedBypass = m_rtBypassShadow; // RT-local shadow（atomic write 禁止のため setBypassFromRT 経由で設定）
    auto* activeBypassRamp = &bypassFadeGain;
    bool effectiveBypass = rtBypassedShadow;

    const double targetBypassFade = requestedBypass ? 0.0 : 1.0;
    const double currentBypassTarget = activeBypassRamp->getTargetValue();
    if (absNoLibm(currentBypassTarget - targetBypassFade) > 1.0e-12)
    {
        if (!requestedBypass && effectiveBypass)
        {
            rtDeferredBandResetMask |= 0xFFFFFFFFu;
            rtBypassedShadow = false;
        }
        activeBypassRamp->setTargetValue(targetBypassFade);
        effectiveBypass = rtBypassedShadow;
    }

    const bool bypassTransitionActive = activeBypassRamp->isSmoothing();

    // フェードアウト完了時に bypassed を true にする
    if (requestedBypass && !effectiveBypass && !bypassTransitionActive)
    {
        rtBypassedShadow = true;
        effectiveBypass = true;
    }

    if (requestedBypass && effectiveBypass && !bypassTransitionActive)
        return;

    auto& activeFilterState = filterState;

    double* activeDryBypassBuffer = dryBypassBuffer.get();
    const int activeDryBypassCapacity = dryBypassCapacity;

    double* activeParallelInputBuffer = parallelInputBuffer.get();
    double* activeParallelWorkBuffer = parallelWorkBuffer.get();
    double* activeParallelAccumBuffer = parallelAccumBuffer.get();
    const int activeParallelBufferCapacity = parallelBufferCapacity;

    double* activeStructureOldOutBuffer = structureOldOutBuffer.get();
    double* activeStructureNewOutBuffer = structureNewOutBuffer.get();
    const int activeStructureXfadeBufferCapacity = structureXfadeBufferCapacity;

    const int numSamples = (int)block.getNumSamples();

    // ==================================================================
    // 【Issue 4 追加安全ガード】オーバーラン即検出（Audio Thread安全版）
    // ==================================================================
    // パッチ2: jassert削除 + 安全クリア（ガイドライン厳守）
    if (numSamples <= 0)
        return;

    if (static_cast<size_t>(numSamples) > static_cast<size_t>(maxInternalBlockSize))
    {
        // バッファオーバーラン時は安全にゼロクリアして早期リターン
        for (int ch = 0; ch < (int)block.getNumChannels(); ++ch)
            juce::FloatVectorOperations::clear(block.getChannelPointer(ch), numSamples);
        return;
    }

    const int numChannels = std::min((int)block.getNumChannels(), MAX_CHANNELS);
    if (numChannels <= 0)
        return;
    const double saturation = (stateSnapshot != nullptr)
        ? static_cast<double>(stateSnapshot->nonlinearSaturation)
        : 0.2;
    const bool canSafelyResetState = requestedBypass
                                     || effectiveBypass
                                     || bypassTransitionActive
                                     || isAudioBlockSilent(block, numChannels, numSamples);

    double* dryCopyBase = nullptr;
    if (bypassTransitionActive)
    {
        const int requiredDrySamples = numSamples * numChannels;
        if (activeDryBypassBuffer != nullptr && requiredDrySamples <= activeDryBypassCapacity)
        {
            dryCopyBase = activeDryBypassBuffer;
            for (int ch = 0; ch < numChannels; ++ch)
                std::memcpy(dryCopyBase + (ch * numSamples),
                            block.getChannelPointer(ch),
                            sizeof(double) * static_cast<size_t>(numSamples));
        }
    }

    // ── State Reset Handling ──
    // パラメータ変更に伴う状態リセット要求を処理
    const std::uint64_t agcResetSerialNow = convo::consumeAtomic(agcResetSerial, std::memory_order_acquire); // acquire: requestAgcReset/prepareToPlay/reset の release/acq_rel と HB
    if (agcResetSerialNow != rtSeenAgcResetSerial)
    {
        rtSeenAgcResetSerial = agcResetSerialNow;
        rtAgcCurrentGainShadow = 1.0;
        rtAgcEnvInputShadow = 0.0;
        rtAgcEnvOutputShadow = 0.0;
    }

    const std::uint64_t bandResetPackedNow = convo::consumeAtomic(bandResetPacked, std::memory_order_acquire); // acquire: requestBandReset/prepareToPlay/reset の release/acq_rel と HB
    const std::uint64_t bandResetSerialNow = static_cast<std::uint64_t>(bandResetSerialFromPacked(bandResetPackedNow));
    if (bandResetSerialNow != rtSeenBandResetSerial)
    {
        rtSeenBandResetSerial = bandResetSerialNow;
        rtDeferredBandResetMask |= bandResetMaskFromPacked(bandResetPackedNow);
    }

    uint32_t mask = rtDeferredBandResetMask;
    rtDeferredBandResetMask = 0;
    if (mask != 0)
    {
        if (!canSafelyResetState)
        {
            rtDeferredBandResetMask |= mask;
        }
        // 最適化: 全バンドリセットの場合は memset で一括クリア
        else if (mask == 0xFFFFFFFF)
        {
            std::memset(activeFilterState.data(), 0, sizeof(activeFilterState));
        }
        else
        {
            for (int i = 0; i < NUM_BANDS; ++i)
            {
                if (mask & (1u << i))
                    for (int ch = 0; ch < kFilterChannels; ++ch)
                        std::memset(activeFilterState[ch][i].data(), 0, sizeof(double) * 2);
            }
        }
    }

    const bool isAgcEnabled = (stateSnapshot != nullptr)
        ? stateSnapshot->agcEnabled
        : false;
    // ✅ フィルタ処理前に入力レベルをキャッシュ (AGCが有効な場合のみ)
    if (isAgcEnabled)
    {
        double& cachedInputRMSRef = cachedInputRMS;
        cachedInputRMSRef = 0.0;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            const double* data = block.getChannelPointer(ch);
            const double rms = calculateRMS(data, numSamples);

            if (rms > cachedInputRMSRef)
                cachedInputRMSRef = rms;
        }
    }

    // ── 最適化: アクティブなバンドノードを事前にスタックへロード ──
    // チャンネルごとのループ内で atomic load を繰り返すと負荷が高いため、
    // 処理開始時に一度だけロードする。
    // Note: 寿命管理は EBR (Epoch-Based Reclamation) により保証されるため、
    // ここでは Raw Pointer を安全に使用できる（Audio Thread は現在の Epoch を保護中）。
    struct ActiveBandNode {
        const BandNode* node;
        int index;
    };
    std::array<ActiveBandNode, NUM_BANDS> activeBands;
    int numActiveBands = 0;

    // Note: bandNodeBits[] stores non-owning handles. We assume resolved nodes are valid during this block.
    // because deletion is deferred by EBR (reader epoch mechanism).
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        auto* node = loadBandNode(i, std::memory_order_acquire); // acquire: exchangeBandNode/publishBandNode の release/acq_rel と HB
        if (node && node->active)
        {
            activeBands[numActiveBands] = { node, i };
            numActiveBands++;
        }
    }

    using FilterStateStorage = decltype(activeFilterState);

    double* msWork = msWorkBuffer.get();

    const auto processSerial = [&](double* dataL,
                                   double* dataR,
                                   FilterStateStorage& states)
    {
        const bool canProcessMonoMidSide = (msWork != nullptr);
        for (int i = 0; i < numActiveBands; ++i)
        {
            const auto& band = activeBands[i];
            const EQChannelMode mode = band.node->mode;

            if (mode == EQChannelMode::Stereo && numChannels >= 2)
            {
                processBandStereo(dataL, dataR, numSamples,
                                  band.node->coeffs,
                                  states[0][band.index].data(),
                                  states[1][band.index].data(),
                                  saturation);
            }
            else if (mode == EQChannelMode::Mid)
            {
                if (!canProcessMonoMidSide) continue;
                if (numChannels < 2)
                {
                    // Mono→Mid: dataLそのまま処理、R=M
                    processBand(dataL, numSamples, band.node->coeffs,
                                states[2][band.index].data(), saturation);
                    juce::FloatVectorOperations::copy(dataR, dataL, numSamples);
                    continue;
                }
                // MとSをエンコード
                juce::FloatVectorOperations::copy(msWork, dataL, numSamples);
                juce::FloatVectorOperations::add(msWork, dataR, numSamples);
                juce::FloatVectorOperations::multiply(msWork, 0.5, numSamples);
                juce::FloatVectorOperations::copy(msWork + numSamples, dataL, numSamples);
                juce::FloatVectorOperations::subtract(msWork + numSamples, dataR, numSamples);
                juce::FloatVectorOperations::multiply(msWork + numSamples, 0.5, numSamples);
                // Mid成分のみ処理
                processBand(msWork, numSamples, band.node->coeffs,
                            states[2][band.index].data(), saturation);
                // デコード: L=M+S, R=M-S
                for (int n = 0; n < numSamples; ++n) {
                    dataL[n] = msWork[n] + msWork[numSamples + n];
                    dataR[n] = msWork[n] - msWork[numSamples + n];
                }
            }
            else if (mode == EQChannelMode::Side)
            {
                if (numChannels < 2)
                {
                    // Mono→Side: Side=0出力
                    juce::FloatVectorOperations::clear(dataL, numSamples);
                    continue;
                }
                // MとSをエンコード
                juce::FloatVectorOperations::copy(msWork, dataL, numSamples);
                juce::FloatVectorOperations::add(msWork, dataR, numSamples);
                juce::FloatVectorOperations::multiply(msWork, 0.5, numSamples);
                juce::FloatVectorOperations::copy(msWork + numSamples, dataL, numSamples);
                juce::FloatVectorOperations::subtract(msWork + numSamples, dataR, numSamples);
                juce::FloatVectorOperations::multiply(msWork + numSamples, 0.5, numSamples);
                // Side成分のみ処理
                processBand(msWork + numSamples, numSamples, band.node->coeffs,
                            states[3][band.index].data(), saturation);
                // デコード: L=M+S, R=M-S
                for (int n = 0; n < numSamples; ++n) {
                    dataL[n] = msWork[n] + msWork[numSamples + n];
                    dataR[n] = msWork[n] - msWork[numSamples + n];
                }
            }
            else
            {
                if ((mode == EQChannelMode::Stereo || mode == EQChannelMode::Left) && numChannels > 0)
                    processBand(dataL, numSamples, band.node->coeffs, states[0][band.index].data(), saturation);
                if ((mode == EQChannelMode::Stereo || mode == EQChannelMode::Right) && numChannels > 1)
                    processBand(dataR, numSamples, band.node->coeffs, states[1][band.index].data(), saturation);
            }
        }
    };

    const auto processParallel = [&](const double* srcL,
                                     const double* srcR,
                                     double* dstL,
                                     double* dstR,
                                     FilterStateStorage& states)
    {
        if (!(activeParallelWorkBuffer && activeParallelAccumBuffer))
        {
            std::memcpy(dstL, srcL, sizeof(double) * static_cast<size_t>(numSamples));
            if (numChannels > 1)
                std::memcpy(dstR, srcR, sizeof(double) * static_cast<size_t>(numSamples));
            return;
        }

        double* workL = activeParallelWorkBuffer;
        double* workR = activeParallelWorkBuffer + numSamples;
        double* accumL = activeParallelAccumBuffer;
        double* accumR = activeParallelAccumBuffer + numSamples;
        juce::FloatVectorOperations::clear(accumL, numSamples);
        if (numChannels > 1)
            juce::FloatVectorOperations::clear(accumR, numSamples);

        for (int i = 0; i < numActiveBands; ++i)
        {
            const auto& band = activeBands[i];
            const EQChannelMode mode = band.node->mode;

            if (mode == EQChannelMode::Stereo && numChannels >= 2)
            {
                juce::FloatVectorOperations::copy(workL, srcL, numSamples);
                juce::FloatVectorOperations::copy(workR, srcR, numSamples);
                processBandStereo(workL, workR, numSamples,
                                  band.node->coeffs,
                                  states[0][band.index].data(),
                                  states[1][band.index].data(),
                                  saturation);
                juce::FloatVectorOperations::add(accumL, workL, numSamples);
                juce::FloatVectorOperations::subtract(accumL, srcL, numSamples);
                juce::FloatVectorOperations::add(accumR, workR, numSamples);
                juce::FloatVectorOperations::subtract(accumR, srcR, numSamples);
            }
            else if (mode == EQChannelMode::Mid || mode == EQChannelMode::Side)
            {
                if (numChannels < 2)
                {
                    if (mode == EQChannelMode::Mid)
                    {
                        // Mono→Mid: srcLそのまま、accum加算
                        juce::FloatVectorOperations::copy(workL, srcL, numSamples);
                        processBand(workL, numSamples, band.node->coeffs,
                                    states[2][band.index].data(), saturation);
                        juce::FloatVectorOperations::add(accumL, workL, numSamples);
                        juce::FloatVectorOperations::subtract(accumL, srcL, numSamples);
                    }
                    // Mono→Side: 何もしない（Side=0）
                    continue;
                }
                // ① MとSをエンコード → workM, workS (msWork[0..n]=M, [n..2n]=S)
                juce::FloatVectorOperations::copy(msWork, srcL, numSamples);
                juce::FloatVectorOperations::add(msWork, srcR, numSamples);
                juce::FloatVectorOperations::multiply(msWork, 0.5, numSamples);
                juce::FloatVectorOperations::copy(msWork + numSamples, srcL, numSamples);
                juce::FloatVectorOperations::subtract(msWork + numSamples, srcR, numSamples);
                juce::FloatVectorOperations::multiply(msWork + numSamples, 0.5, numSamples);

                // ② 処理: Mid→msWork[0..n], Side→msWork[n..2n]
                auto* targetState = (mode == EQChannelMode::Mid)
                    ? states[2][band.index].data()
                    : states[3][band.index].data();
                auto* targetBuf = (mode == EQChannelMode::Mid)
                    ? msWork
                    : (msWork + numSamples);
                processBand(targetBuf, numSamples, band.node->coeffs, targetState, saturation);

                // ③ デコード → workL, workRへ
                for (int n = 0; n < numSamples; ++n)
                {
                    workL[n] = msWork[n] + msWork[numSamples + n];
                    workR[n] = msWork[n] - msWork[numSamples + n];
                }
                // ④ 差分加算
                for (int n = 0; n < numSamples; ++n)
                {
                    accumL[n] += workL[n] - srcL[n];
                    accumR[n] += workR[n] - srcR[n];
                }
            }
            else
            {
                if ((mode == EQChannelMode::Stereo || mode == EQChannelMode::Left) && numChannels > 0)
                {
                    juce::FloatVectorOperations::copy(workL, srcL, numSamples);
                    processBand(workL, numSamples, band.node->coeffs, states[0][band.index].data(), saturation);
                    juce::FloatVectorOperations::add(accumL, workL, numSamples);
                    juce::FloatVectorOperations::subtract(accumL, srcL, numSamples);
                }

                if ((mode == EQChannelMode::Stereo || mode == EQChannelMode::Right) && numChannels > 1)
                {
                    juce::FloatVectorOperations::copy(workR, srcR, numSamples);
                    processBand(workR, numSamples, band.node->coeffs, states[1][band.index].data(), saturation);
                    juce::FloatVectorOperations::add(accumR, workR, numSamples);
                    juce::FloatVectorOperations::subtract(accumR, srcR, numSamples);
                }
            }
        }

        juce::FloatVectorOperations::copy(dstL, srcL, numSamples);
        juce::FloatVectorOperations::add(dstL, accumL, numSamples);
        if (numChannels > 1)
        {
            juce::FloatVectorOperations::copy(dstR, srcR, numSamples);
            juce::FloatVectorOperations::add(dstR, accumR, numSamples);
        }
    };

    auto activeMode = rtActiveStructureShadow;
    auto requestedMode = (stateSnapshot != nullptr)
        ? static_cast<FilterStructure>(stateSnapshot->filterStructure)
        : activeMode;
    const bool canUseParallelBuffers = (activeParallelInputBuffer != nullptr)
                                       && activeParallelBufferCapacity >= (numSamples * numChannels);
    const bool canUseStructureXfade = canUseParallelBuffers
                                      && activeStructureOldOutBuffer != nullptr
                                      && activeStructureNewOutBuffer != nullptr
                                      && activeStructureXfadeBufferCapacity >= (numSamples * numChannels);

    auto* blockL = block.getChannelPointer(0);
    double* blockR = (numChannels > 1) ? block.getChannelPointer(1) : nullptr;

    if (requestedMode != activeMode && canUseStructureXfade)
    {
        double* srcL = activeParallelInputBuffer;
        double* srcR = (numChannels > 1) ? (activeParallelInputBuffer + numSamples) : nullptr;
        std::memcpy(srcL, blockL, sizeof(double) * static_cast<size_t>(numSamples));
        if (numChannels > 1)
            std::memcpy(srcR, blockR, sizeof(double) * static_cast<size_t>(numSamples));

        double* oldL = activeStructureOldOutBuffer;
        double* oldR = (numChannels > 1) ? (activeStructureOldOutBuffer + numSamples) : nullptr;
        double* newL = activeStructureNewOutBuffer;
        double* newR = (numChannels > 1) ? (activeStructureNewOutBuffer + numSamples) : nullptr;
        std::memcpy(oldL, srcL, sizeof(double) * static_cast<size_t>(numSamples));
        std::memcpy(newL, srcL, sizeof(double) * static_cast<size_t>(numSamples));
        if (numChannels > 1)
        {
            std::memcpy(oldR, srcR, sizeof(double) * static_cast<size_t>(numSamples));
            std::memcpy(newR, srcR, sizeof(double) * static_cast<size_t>(numSamples));
        }

        auto oldStateSnapshot = activeFilterState;
        if (activeMode == FilterStructure::Serial)
            processSerial(oldL, oldR, oldStateSnapshot);
        else
            processParallel(srcL, srcR, oldL, oldR, oldStateSnapshot);

        if (requestedMode == FilterStructure::Serial)
            processSerial(newL, newR, activeFilterState);
        else
            processParallel(srcL, srcR, newL, newR, activeFilterState);

        const double step = 1.0 / static_cast<double>(numSamples);
        for (int n = 0; n < numSamples; ++n)
        {
            const double t = (n + 1.0) * step;
            const double wNew = equalPowerSin(t);
            const double wOld = equalPowerSin(1.0 - t);
            blockL[n] = oldL[n] * wOld + newL[n] * wNew;
            if (numChannels > 1)
                blockR[n] = oldR[n] * wOld + newR[n] * wNew;
        }

        rtActiveStructureShadow = requestedMode;
    }
    else
    {
        if (requestedMode != activeMode)
        {
            activeMode = requestedMode;
            rtActiveStructureShadow = activeMode;
        }

        if (activeMode == FilterStructure::Serial || !canUseParallelBuffers)
        {
            processSerial(blockL, blockR, activeFilterState);
        }
        else
        {
            double* srcL = activeParallelInputBuffer;
            double* srcR = (numChannels > 1) ? (activeParallelInputBuffer + numSamples) : nullptr;
            std::memcpy(srcL, blockL, sizeof(double) * static_cast<size_t>(numSamples));
            if (numChannels > 1)
                std::memcpy(srcR, blockR, sizeof(double) * static_cast<size_t>(numSamples));
            processParallel(srcL, srcR, blockL, blockR, activeFilterState);
        }
    }

    // トータルゲイン / AGC 適用
    if (isAgcEnabled)
    {
        processAGC(block);
    }
    else
    {
        // 【Fix Bug #7】Audio Thread内でのlibm呼び出し禁止。
        // juce::Decibels::decibelsToGain() は内部で std::pow() (libm) を呼ぶため
        // Audio Thread 内での使用は規約違反である。
        // 対策: ゲイン値はMessage Thread側でdBからlinearに変換し、
        // totalGainTarget (atomic<double>) で事前に渡す。
        // Audio Threadでは atomic load のみ行う。
        const double targetGain = convo::consumeAtomic(totalGainTarget, std::memory_order_acquire); // acquire: storeTotalGainDb の publishAtomic release と HB

        auto* activeGainRamp = &smoothTotalGain;
        const double gainTarget = activeGainRamp->getTargetValue();
        if (absNoLibm(gainTarget - targetGain) > 1e-6)
            activeGainRamp->setTargetValue(targetGain);

        const double startGain = activeGainRamp->getCurrentValue();
        activeGainRamp->skip(numSamples);
        const double endGain = activeGainRamp->getCurrentValue();

        // AGC 無効時のゲインランプ
        const double increment = (endGain - startGain) / static_cast<double>(numSamples);
        for (int ch = 0; ch < numChannels; ++ch)
            applyGainRamp_AVX2(block.getChannelPointer(ch), numSamples, startGain, increment);
    }

    if (bypassTransitionActive)
    {
        const bool canBlendDry = (dryCopyBase != nullptr);
        for (int sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex)
        {
            const double wetGainState = (activeBypassRamp != nullptr)
                ? activeBypassRamp->getNextValue()
                : bypassFadeGain.getNextValue();
            const double dryGain = 1.0 - wetGainState;

            if (canBlendDry)
            {
                for (int ch = 0; ch < numChannels; ++ch)
                {
                    double* wetPtr = block.getChannelPointer(ch);
                    const double dryValue = dryCopyBase[ch * numSamples + sampleIndex];
                    wetPtr[sampleIndex] = wetPtr[sampleIndex] * wetGainState + dryValue * dryGain;
                }
            }
            else
            {
                // dry copy が確保できない場合でも、遷移進行は止めない（wet-only 減衰）。
                for (int ch = 0; ch < numChannels; ++ch)
                {
                    double* wetPtr = block.getChannelPointer(ch);
                    wetPtr[sampleIndex] = wetPtr[sampleIndex] * wetGainState;
                }
            }
        }

        const bool smoothingAfterBlend = activeBypassRamp->isSmoothing();
        if (!smoothingAfterBlend)
        {
            if (requestedBypass)
                rtBypassedShadow = true;
            else
                rtBypassedShadow = false;
        }
    }
}

void EQProcessor::process(juce::dsp::AudioBlock<double>& block,
                          const convo::EQParameters& eqParams,
                          const EQCoeffCache* coeffCache)
{
    const bool effectiveBypassed = rtBypassedShadow;
    const bool bypassSmoothing = bypassFadeGain.isSmoothing();

    // 既存バイパス遷移ロジックを維持するため、遷移中は既存パスへフォールバック
    if (coeffCache == nullptr
        || m_rtBypassShadow // RT-local shadow（atomic write 禁止のため setBypassFromRT 経由で設定）
        || effectiveBypassed
        || bypassSmoothing)
    {
        process(block);
        return;
    }

    // Mid/Sideモード検出時は基本process()へフォールバック（M/S処理は基本パスのみ対応）
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        if (coeffCache->bandActive[i] && coeffCache->channelModes[i] >= 3)
        {
            process(block);
            return;
        }
    }

    juce::ScopedNoDenormals noDenormals;

    const int numSamples = static_cast<int>(block.getNumSamples());
    if (numSamples <= 0)
        return;

    if (static_cast<size_t>(numSamples) > static_cast<size_t>(maxInternalBlockSize))
    {
        for (int ch = 0; ch < static_cast<int>(block.getNumChannels()); ++ch)
            juce::FloatVectorOperations::clear(block.getChannelPointer(ch), numSamples);
        return;
    }

    const int numChannels = std::min(static_cast<int>(block.getNumChannels()), MAX_CHANNELS);
    if (numChannels <= 0)
        return;

    auto& activeFilterState = filterState;

    double* activeParallelInputBuffer = parallelInputBuffer.get();
    double* activeParallelWorkBuffer = parallelWorkBuffer.get();
    double* activeParallelAccumBuffer = parallelAccumBuffer.get();
    const int activeParallelBufferCapacity = parallelBufferCapacity;

    const std::uint64_t agcResetSerialNow = convo::consumeAtomic(agcResetSerial, std::memory_order_acquire); // acquire: requestAgcReset/prepareToPlay/reset の release/acq_rel と HB
    if (agcResetSerialNow != rtSeenAgcResetSerial)
    {
        rtSeenAgcResetSerial = agcResetSerialNow;
        rtAgcCurrentGainShadow = 1.0;
        rtAgcEnvInputShadow = 0.0;
        rtAgcEnvOutputShadow = 0.0;
    }

    const std::uint64_t bandResetPackedNow = convo::consumeAtomic(bandResetPacked, std::memory_order_acquire); // acquire: requestBandReset/prepareToPlay/reset の release/acq_rel と HB
    const std::uint64_t bandResetSerialNow = static_cast<std::uint64_t>(bandResetSerialFromPacked(bandResetPackedNow));
    if (bandResetSerialNow != rtSeenBandResetSerial)
    {
        rtSeenBandResetSerial = bandResetSerialNow;
        rtDeferredBandResetMask |= bandResetMaskFromPacked(bandResetPackedNow);
    }

    uint32_t mask = rtDeferredBandResetMask;
    rtDeferredBandResetMask = 0;
    if (mask != 0)
    {
        if (isAudioBlockSilent(block, numChannels, numSamples))
        {
            if (mask == 0xFFFFFFFFu)
            {
                std::memset(activeFilterState.data(), 0, sizeof(activeFilterState));
            }
            else
            {
                for (int i = 0; i < NUM_BANDS; ++i)
                {
                    if (mask & (1u << i))
                    {
                        for (int ch = 0; ch < kFilterChannels; ++ch)
                            std::memset(activeFilterState[ch][i].data(), 0, sizeof(double) * 2);
                    }
                }
            }
        }
        else
        {
            rtDeferredBandResetMask |= mask;
        }
    }

    const double saturation = static_cast<double>(eqParams.nonlinearSaturation);

    if (eqParams.agcEnabled)
    {
        double& cachedInputRMSRef = cachedInputRMS;
        cachedInputRMSRef = 0.0;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            const double* data = block.getChannelPointer(ch);
            const double rms = calculateRMS(data, numSamples);
            if (rms > cachedInputRMSRef)
                cachedInputRMSRef = rms;
        }
    }

    double* blockL = block.getChannelPointer(0);
    double* blockR = (numChannels > 1) ? block.getChannelPointer(1) : nullptr;

    if (coeffCache->filterStructure == static_cast<int>(FilterStructure::Parallel))
    {
        if (!(activeParallelInputBuffer && activeParallelWorkBuffer && activeParallelAccumBuffer)
            || activeParallelBufferCapacity < (numSamples * numChannels))
        {
            // Parallelバッファが無い/不足時は安全にSerial処理へフォールバック
            for (int i = 0; i < NUM_BANDS; ++i)
            {
                if (!coeffCache->bandActive[i])
                    continue;

                const EQCoeffsSVF& c = coeffCache->coeffs[i];
                const EQChannelMode mode = static_cast<EQChannelMode>(coeffCache->channelModes[i]);

                if (mode == EQChannelMode::Stereo && numChannels >= 2)
                {
                    processBandStereo(blockL, blockR, numSamples, c,
                                      activeFilterState[0][i].data(),
                                      activeFilterState[1][i].data(),
                                      saturation);
                }
                else
                {
                    if ((mode == EQChannelMode::Stereo || mode == EQChannelMode::Left) && numChannels > 0)
                        processBand(blockL, numSamples, c, activeFilterState[0][i].data(), saturation);
                    if ((mode == EQChannelMode::Stereo || mode == EQChannelMode::Right) && numChannels > 1)
                        processBand(blockR, numSamples, c, activeFilterState[1][i].data(), saturation);
                }
            }
        }
        else
        {
            double* srcL = activeParallelInputBuffer;
            double* srcR = (numChannels > 1) ? (activeParallelInputBuffer + numSamples) : nullptr;
            double* workL = activeParallelWorkBuffer;
            double* workR = (numChannels > 1) ? (activeParallelWorkBuffer + numSamples) : nullptr;
            double* accumL = activeParallelAccumBuffer;
            double* accumR = (numChannels > 1) ? (activeParallelAccumBuffer + numSamples) : nullptr;

            juce::FloatVectorOperations::copy(srcL, blockL, numSamples);
            if (numChannels > 1)
                juce::FloatVectorOperations::copy(srcR, blockR, numSamples);

            juce::FloatVectorOperations::clear(accumL, numSamples);
            if (numChannels > 1)
                juce::FloatVectorOperations::clear(accumR, numSamples);

            for (int i = 0; i < NUM_BANDS; ++i)
            {
                if (!coeffCache->bandActive[i])
                    continue;

                const EQCoeffsSVF& c = coeffCache->coeffs[i];
                const EQChannelMode mode = static_cast<EQChannelMode>(coeffCache->channelModes[i]);

                if (mode == EQChannelMode::Stereo && numChannels >= 2)
                {
                    juce::FloatVectorOperations::copy(workL, srcL, numSamples);
                    juce::FloatVectorOperations::copy(workR, srcR, numSamples);
                    processBandStereo(workL, workR, numSamples, c,
                                      activeFilterState[0][i].data(),
                                      activeFilterState[1][i].data(),
                                      saturation);
                    juce::FloatVectorOperations::add(accumL, workL, numSamples);
                    juce::FloatVectorOperations::subtract(accumL, srcL, numSamples);
                    juce::FloatVectorOperations::add(accumR, workR, numSamples);
                    juce::FloatVectorOperations::subtract(accumR, srcR, numSamples);
                }
                else
                {
                    if ((mode == EQChannelMode::Stereo || mode == EQChannelMode::Left) && numChannels > 0)
                    {
                        juce::FloatVectorOperations::copy(workL, srcL, numSamples);
                        processBand(workL, numSamples, c, activeFilterState[0][i].data(), saturation);
                        juce::FloatVectorOperations::add(accumL, workL, numSamples);
                        juce::FloatVectorOperations::subtract(accumL, srcL, numSamples);
                    }

                    if ((mode == EQChannelMode::Stereo || mode == EQChannelMode::Right) && numChannels > 1)
                    {
                        juce::FloatVectorOperations::copy(workR, srcR, numSamples);
                        processBand(workR, numSamples, c, activeFilterState[1][i].data(), saturation);
                        juce::FloatVectorOperations::add(accumR, workR, numSamples);
                        juce::FloatVectorOperations::subtract(accumR, srcR, numSamples);
                    }
                }
            }

            juce::FloatVectorOperations::copy(blockL, srcL, numSamples);
            juce::FloatVectorOperations::add(blockL, accumL, numSamples);
            if (numChannels > 1)
            {
                juce::FloatVectorOperations::copy(blockR, srcR, numSamples);
                juce::FloatVectorOperations::add(blockR, accumR, numSamples);
            }
        }
    }
    else
    {
        for (int i = 0; i < NUM_BANDS; ++i)
        {
            if (!coeffCache->bandActive[i])
                continue;

            const EQCoeffsSVF& c = coeffCache->coeffs[i];
            const EQChannelMode mode = static_cast<EQChannelMode>(coeffCache->channelModes[i]);

            if (mode == EQChannelMode::Stereo && numChannels >= 2)
            {
                processBandStereo(blockL, blockR, numSamples, c,
                                  activeFilterState[0][i].data(),
                                  activeFilterState[1][i].data(),
                                  saturation);
            }
            else
            {
                if ((mode == EQChannelMode::Stereo || mode == EQChannelMode::Left) && numChannels > 0)
                    processBand(blockL, numSamples, c, activeFilterState[0][i].data(), saturation);
                if ((mode == EQChannelMode::Stereo || mode == EQChannelMode::Right) && numChannels > 1)
                    processBand(blockR, numSamples, c, activeFilterState[1][i].data(), saturation);
            }
        }
    }

    if (eqParams.agcEnabled)
    {
        processAGC(block);
    }
    else
    {
        const double targetGain = convo::consumeAtomic(totalGainTarget, std::memory_order_acquire); // acquire: storeTotalGainDb の publishAtomic release と HB
        auto* activeGainRamp = &smoothTotalGain;
        const double gainTarget = activeGainRamp->getTargetValue();
        if (absNoLibm(gainTarget - targetGain) > 1e-6)
            activeGainRamp->setTargetValue(targetGain);

        const double startGain = activeGainRamp->getCurrentValue();
        activeGainRamp->skip(numSamples);
        const double endGain = activeGainRamp->getCurrentValue();
        const double increment = (endGain - startGain) / static_cast<double>(numSamples);

        for (int ch = 0; ch < numChannels; ++ch)
            applyGainRamp_AVX2(block.getChannelPointer(ch), numSamples, startGain, increment);
    }
}


//--------------------------------------------------------------
// BandNode作成 (Message Thread)
//--------------------------------------------------------------
