//============================================================================
// EQProcessor.Coefficients.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// 係数計算: SVF/Biquad設計, BandNode生成・更新
// Topology-Preserving Transform (TPT) State Variable Filter実装
// 参照: Vadim Zavalishin "The Art of VA Filter Design"
//============================================================================
#include "EQProcessor.h"
#include <cmath>
#include <complex>
#include <algorithm>
#include "core/EpochManager.h"

#include "audioengine/AtomicAccess.h"

//============================================================================
// Destruction handler for EBR (Epoch-Based Reclamation)
// L5: epoch-only 削除。RefCountedDeferred の二重ライフタイムモデルを廃止。
//============================================================================
static void deleteBandNodePtr_coeff(void* p) { delete static_cast<EQProcessor::BandNode*>(p); }

static void retireBandNode(EQProcessor::BandNode* node)
{
    if (node) {
        const uint64_t epoch = convo::EpochManager::instance().currentEpoch();
        g_deletionQueue.enqueue(node, deleteBandNodePtr_coeff, epoch);
    }
}

//============================================================================
// BandNode生成 (Message Thread)
//============================================================================
EQProcessor::BandNode* EQProcessor::createBandNode(int band, const EQState& state) const
{
    auto node = new BandNode();
    // EBR: retirement managed by retireBandNode
    const auto& params = state.bands[band];
    const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);

    node->active = params.enabled;
    node->mode = state.bandChannelModes[band];

    // prepareToPlay 前は sampleRate が未確定のため、係数計算を保留して
    if (sr > 0.0)
    {
        node->coeffs = calcSVFCoeffs(state.bandTypes[band], params.frequency, params.gain, params.q, sr);
    }
    else
    {
        node->coeffs = EQCoeffsSVF();
        node->active = false;
    }

    // 最適化: ゲインが0dB付近ならスキップ
    if (node->active) {
        EQBandType type = state.bandTypes[band];
        if ((type != EQBandType::LowPass && type != EQBandType::HighPass) && std::abs(params.gain) < 0.01f)
            node->active = false;
    }

    return node;
}

//--------------------------------------------------------------
// BandNode更新 (Message Thread)
// 内部係数クロスフェードを使わず、最新係数に直接差し替える
//--------------------------------------------------------------
void EQProcessor::updateBandNode(int band)
{
    if (band < 0 || band >= NUM_BANDS) return;

    auto state = loadCurrentState(std::memory_order_acquire);
    if (state == nullptr) return;
    auto newNode = createBandNode(band, *state);
    BandNode* oldNode = exchangeBandNode(band, newNode, std::memory_order_acq_rel);

    activeBandNodes[band] = newNode;
    // L5 fix: retire old node BEFORE advanceEpoch so epoch N is captured (not N+1).
    if (oldNode)
    {
        retireBandNode(oldNode);
    }
    convo::EpochManager::instance().advanceEpoch();
}

//============================================================================
// パラメータ検証とクランプ (Helper)
//============================================================================
void EQProcessor::validateAndClampParameters(float& freq, float& gainDb, float& q, double sr) noexcept
{
    // 周波数をナイキスト周波数以下にクランプ
    const float nyquist = static_cast<float>(sr * 0.5);
    const float maxFreq = std::min(DSP_MAX_FREQ, nyquist * DSP_MAX_FREQ_NYQUIST_RATIO);
    freq = juce::jlimit(DSP_MIN_FREQ, maxFreq, freq);

    // Qを安全な範囲にクランプ
    q = juce::jlimit(DSP_MIN_Q, DSP_MAX_Q, q);

    // ゲインを実用範囲にクランプ
    gainDb = juce::jlimit(DSP_MIN_GAIN_DB, DSP_MAX_GAIN_DB, gainDb);
}

//============================================================================
// SVF係数計算 (Audio Thread用)
//============================================================================
EQCoeffsSVF EQProcessor::calcSVFCoeffs(EQBandType type, float freq, float gainDb, float q, double sr) noexcept
{
    // パラメータ検証 (Parameter Validation)
    // 不正な値から保護し、安全な範囲にクランプ
    if (sr <= 0.0)
    {
        jassertfalse;
        // 不正なサンプルレートでは計算不能なため、バイパス係数を返す
        EQCoeffsSVF c;
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }
    validateAndClampParameters(freq, gainDb, q, sr);

    const double f = static_cast<double>(freq);
    const double g = static_cast<double>(gainDb);
    const double Q = static_cast<double>(q);
    const double s = sr;

    switch (type)
    {
        case EQBandType::LowShelf:  return calcLowShelfSVF(f, g, Q, s);
        case EQBandType::Peaking:   return calcPeakingSVF(f, g, Q, s);
        case EQBandType::HighShelf: return calcHighShelfSVF(f, g, Q, s);
        case EQBandType::LowPass:   return calcLowPassSVF(f, Q, s);
        case EQBandType::HighPass:  return calcHighPassSVF(f, Q, s);
    }
    return {}; // unreachable
}

//============================================================================
// Biquad係数計算 (UI Thread用)
//============================================================================
EQCoeffsBiquad EQProcessor::calcBiquadCoeffs(EQBandType type, float freq, float gainDb, float q, double sr) noexcept
{
    // パラメータ検証 (Parameter Validation)
    // [FIX] 上限チェックを削除: 以前の sr > 384000.0 は誤り。
    // オーバーサンプリング使用時 (例: 96kHz × 8 = 768kHz) にsr=48000へフォールバックし、
    // zCache (処理srベース) とbiquad係数 (48kHzベース) が不整合になり、
    // 表示曲線のピーク周波数が最大16倍ずれる不具合を引き起こしていた。
    // (例: 100Hz設定 → 1600Hz表示、ユーザー報告: 25-100Hzブーストが800Hz付近に表示)
    // Biquad計算式はsr > 0.0であれば任意のサンプルレートで数学的に正確に動作する。
    if (sr <= 0.0 || !std::isfinite(sr))
    {
        jassertfalse;
        sr = 48000.0;
    }

    validateAndClampParameters(freq, gainDb, q, sr);

    const double f = static_cast<double>(freq);
    const double g = static_cast<double>(gainDb);
    const double Q = static_cast<double>(q);
    const double s = sr;

    switch (type)
    {
        case EQBandType::LowShelf:  return calcLowShelfBiquad(f, g, Q, s);
        case EQBandType::Peaking:   return calcPeakingBiquad(f, g, Q, s);
        case EQBandType::HighShelf: return calcHighShelfBiquad(f, g, Q, s);
        case EQBandType::LowPass:   return calcLowPassBiquad(f, Q, s);
        case EQBandType::HighPass:  return calcHighPassBiquad(f, Q, s);
    }
    return {};
}

EQCoeffsBiquad EQProcessor::calcLowShelfBiquad(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsBiquad c;
    const double A     = std::pow(10.0, gainDb / 40.0);
    const double w0    = 2.0 * juce::MathConstants<double>::pi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);
    const double sqrtA = std::sqrt(A);
    const double twoSqrtAAlpha = 2.0 * sqrtA * alpha;

    if (!std::isfinite(alpha) || !std::isfinite(twoSqrtAAlpha))
    {
        c.b0 = 1.0; c.b1 = 0.0; c.b2 = 0.0;
        c.a0 = 1.0; c.a1 = 0.0; c.a2 = 0.0;
        return c;
    }

    const double a0 = (A + 1.0) + (A - 1.0) * cosw0 + twoSqrtAAlpha;
    if (std::abs(a0) < 1.0e-15)
    {
        c.b0 = 1.0; c.b1 = 0.0; c.b2 = 0.0;
        c.a0 = 1.0; c.a1 = 0.0; c.a2 = 0.0;
        return c;
    }

    c.b0 =       A * ((A + 1.0) - (A - 1.0) * cosw0 + twoSqrtAAlpha);
    c.b1 =  2.0 * A * ((A - 1.0) - (A + 1.0) * cosw0);
    c.b2 =  A * ((A + 1.0) - (A - 1.0) * cosw0 - twoSqrtAAlpha);
    c.a0 =           a0;
    c.a1 = -2.0     * ((A - 1.0) + (A + 1.0) * cosw0               );
    c.a2 =           ((A + 1.0) + (A - 1.0) * cosw0 - twoSqrtAAlpha);
    return c;
}

EQCoeffsBiquad EQProcessor::calcPeakingBiquad(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsBiquad c;
    const double A     = std::pow(10.0, gainDb / 40.0);
    const double w0    = 2.0 * juce::MathConstants<double>::pi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);

    if (!std::isfinite(alpha))
    {
        c.b0 = 1.0; c.b1 = 0.0; c.b2 = 0.0;
        c.a0 = 1.0; c.a1 = 0.0; c.a2 = 0.0;
        return c;
    }

    const double a0 = 1.0 + alpha / A;
    if (std::abs(a0) < 1.0e-15)
    {
        c.b0 = 1.0; c.b1 = 0.0; c.b2 = 0.0;
        c.a0 = 1.0; c.a1 = 0.0; c.a2 = 0.0;
        return c;
    }

    c.b0 =  1.0 + alpha * A;
    c.b1 = -2.0 * cosw0;
    c.b2 =  1.0 - alpha * A;
    c.a0 =  a0;
    c.a1 = -2.0 * cosw0;
    c.a2 =  1.0 - alpha / A;
    return c;
}

EQCoeffsBiquad EQProcessor::calcHighShelfBiquad(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsBiquad c;
    const double A     = std::pow(10.0, gainDb / 40.0);
    const double w0    = 2.0 * juce::MathConstants<double>::pi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);
    const double sqrtA = std::sqrt(A);
    const double twoSqrtAAlpha = 2.0 * sqrtA * alpha;

    if (!std::isfinite(alpha) || !std::isfinite(twoSqrtAAlpha))
    {
        c.b0 = 1.0; c.b1 = 0.0; c.b2 = 0.0;
        c.a0 = 1.0; c.a1 = 0.0; c.a2 = 0.0;
        return c;
    }

    const double a0 = (A + 1.0) - (A - 1.0) * cosw0 + twoSqrtAAlpha;
    if (std::abs(a0) < 1.0e-15)
    {
        c.b0 = 1.0; c.b1 = 0.0; c.b2 = 0.0;
        c.a0 = 1.0; c.a1 = 0.0; c.a2 = 0.0;
        return c;
    }

    c.b0 =       A * ((A + 1.0) + (A - 1.0) * cosw0 + twoSqrtAAlpha);
    c.b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0               );
    c.b2 =       A * ((A + 1.0) + (A - 1.0) * cosw0 - twoSqrtAAlpha);
    c.a0 =           a0;
    c.a1 =  2.0     * ((A - 1.0) - (A + 1.0) * cosw0               );
    c.a2 =           ((A + 1.0) - (A - 1.0) * cosw0 - twoSqrtAAlpha);
    return c;
}

EQCoeffsBiquad EQProcessor::calcLowPassBiquad(double freq, double q, double sr) noexcept
{
    EQCoeffsBiquad c;
    const double w0    = 2.0 * juce::MathConstants<double>::pi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);

    if (!std::isfinite(alpha))
    {
        c.b0 = 1.0; c.b1 = 0.0; c.b2 = 0.0;
        c.a0 = 1.0; c.a1 = 0.0; c.a2 = 0.0;
        return c;
    }

    const double a0 = 1.0 + alpha;
    if (std::abs(a0) < 1.0e-15)
    {
        c.b0 = 1.0; c.b1 = 0.0; c.b2 = 0.0;
        c.a0 = 1.0; c.a1 = 0.0; c.a2 = 0.0;
        return c;
    }

    c.b0 =  (1.0 - cosw0) / 2.0;
    c.b1 =   1.0 - cosw0;
    c.b2 =  (1.0 - cosw0) / 2.0;
    c.a0 =   a0;
    c.a1 =  -2.0 * cosw0;
    c.a2 =   1.0 - alpha;
    return c;
}

EQCoeffsBiquad EQProcessor::calcHighPassBiquad(double freq, double q, double sr) noexcept
{
    EQCoeffsBiquad c;
    const double w0    = 2.0 * juce::MathConstants<double>::pi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);

    if (!std::isfinite(alpha))
    {
        c.b0 = 1.0; c.b1 = 0.0; c.b2 = 0.0;
        c.a0 = 1.0; c.a1 = 0.0; c.a2 = 0.0;
        return c;
    }

    const double a0 = 1.0 + alpha;
    if (std::abs(a0) < 1.0e-15)
    {
        c.b0 = 1.0; c.b1 = 0.0; c.b2 = 0.0;
        c.a0 = 1.0; c.a1 = 0.0; c.a2 = 0.0;
        return c;
    }

    c.b0 =  (1.0 + cosw0) / 2.0;
    c.b1 = -(1.0 + cosw0);
    c.b2 =  (1.0 + cosw0) / 2.0;
    c.a0 =   a0;
    c.a1 =  -2.0 * cosw0;
    c.a2 =   1.0 - alpha;
    return c;
}

//============================================================================
// 周波数応答（マグニチュードの二乗）計算
// sqrtを避けるため、連鎖的な計算やdB変換前の最適化に使用
//============================================================================
float EQProcessor::getMagnitudeSquared(const EQCoeffsBiquad& c, float freq, float sampleRate) noexcept
{
    const double w = 2.0 * juce::MathConstants<double>::pi * static_cast<double>(freq) / static_cast<double>(sampleRate);
    std::complex<double> z(std::cos(w), std::sin(w));
    return getMagnitudeSquared(c, z);
}

float EQProcessor::getMagnitudeSquared(const EQCoeffsBiquad& c, const std::complex<double>& z) noexcept
{
    std::complex<double> z2 = z * z;
    std::complex<double> num = c.b0 * z2 + c.b1 * z + c.b2;
    std::complex<double> den = c.a0 * z2 + c.a1 * z + c.a2;

    double denNorm = std::norm(den); // normはマグニチュードの二乗を返す
    if (denNorm < 1e-18) return 0.0f;

    return static_cast<float>(std::norm(num) / denNorm);
}

//============================================================================
// SVF係数から等価 Biquad 係数を計算 (UI表示用)
//============================================================================
EQCoeffsBiquad EQProcessor::svfToDisplayBiquad(const EQCoeffsSVF& svf) noexcept
{
    EQCoeffsBiquad bq;
    const double a1 = svf.a1, a2 = svf.a2, a3 = svf.a3;
    const double m0 = svf.m0, m1 = svf.m1, m2 = svf.m2;

    if (a1 < 1e-15) { bq.b0 = 1.0; bq.a0 = 1.0; return bq; }

    const double g2  = a3 / a1;
    const double g   = a2 / a1;
    const double gk  = (1.0 - a1 - a3) / a1;

    bq.a0 =  1.0 + gk + g2;
    bq.a1 = -2.0 + 2.0 * g2;
    bq.a2 =  1.0 - gk + g2;

    bq.b0 = m0 * (1.0 + gk + g2) + m1 * g + m2 * g2;
    bq.b1 = -2.0 * m0 + 2.0 * (m0 + m2) * g2;
    bq.b2 = m0 * (1.0 - gk + g2) - m1 * g + m2 * g2;

    return bq;
}

//============================================================================
// SVF係数実装
// 参照: "The Art of VA Filter Design" by Vadim Zavalishin
//============================================================================
EQCoeffsSVF EQProcessor::calcLowShelfSVF(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double A = std::pow(10.0, gainDb / 40.0);
    const double g = std::tan(juce::MathConstants<double>::pi * freq / sr) / std::sqrt(A);
    const double k = 1.0 / q;

    // NaN/Infチェック: tan()が発散した場合など
    if (!std::isfinite(g) || !std::isfinite(k))
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }

    // 除算ゼロ保護 (Division by Zero Protection)
    const double denominator = 1.0 + g * (g + k);
    if (std::abs(denominator) < 1.0e-15)
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0;
        c.a2 = 0.0;
        c.a3 = 0.0;
        c.m0 = 1.0;
        c.m1 = 0.0;
        c.m2 = 0.0;
        return c;
    }

    c.a1 = 1.0 / denominator;
    c.a2 = g * c.a1;
    c.a3 = g * c.a2;
    c.m0 = 1.0;
    c.m1 = k * (A - 1.0);
    c.m2 = A * A - 1.0;
    return c;
}

EQCoeffsSVF EQProcessor::calcPeakingSVF(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double A = std::pow(10.0, gainDb / 40.0);
    const double g = std::tan(juce::MathConstants<double>::pi * freq / sr);
    const double k = 1.0 / (q * A);

    // NaN/Infチェック
    if (!std::isfinite(g) || !std::isfinite(k))
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }

    // 除算ゼロ保護
    const double denominator = 1.0 + g * (g + k);
    if (std::abs(denominator) < 1.0e-15)
    {
        c.a1 = 1.0;
        c.a2 = 0.0;
        c.a3 = 0.0;
        c.m0 = 1.0;
        c.m1 = 0.0;
        c.m2 = 0.0;
        return c;
    }

    c.a1 = 1.0 / denominator;
    c.a2 = g * c.a1;
    c.a3 = g * c.a2;
    c.m0 = 1.0;
    c.m1 = (A - 1.0 / A) / q;
    c.m2 = 0.0;
    return c;
}

EQCoeffsSVF EQProcessor::calcHighShelfSVF(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double A = std::pow(10.0, gainDb / 40.0);
    const double g = std::tan(juce::MathConstants<double>::pi * freq / sr) * std::sqrt(A);
    const double k = 1.0 / q;

    // NaN/Infチェック
    if (!std::isfinite(g) || !std::isfinite(k))
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }

    // 除算ゼロ保護
    const double denominator = 1.0 + g * (g + k);
    if (std::abs(denominator) < 1.0e-15)
    {
        c.a1 = 1.0;
        c.a2 = 0.0;
        c.a3 = 0.0;
        c.m0 = 1.0;
        c.m1 = 0.0;
        c.m2 = 0.0;
        return c;
    }

    c.a1 = 1.0 / denominator;
    c.a2 = g * c.a1;
    c.a3 = g * c.a2;
    c.m0 = A * A;
    c.m1 = k * (1.0 - A) * A;
    c.m2 = 1.0 - A * A;
    return c;
}

EQCoeffsSVF EQProcessor::calcLowPassSVF(double freq, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double g = std::tan(juce::MathConstants<double>::pi * freq / sr);
    const double k = 1.0 / q;

    // NaN/Infチェック
    if (!std::isfinite(g) || !std::isfinite(k))
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }

    // 除算ゼロ保護
    const double denominator = 1.0 + g * (g + k);
    if (std::abs(denominator) < 1.0e-15)
    {
        c.a1 = 1.0;
        c.a2 = 0.0;
        c.a3 = 0.0;
        c.m0 = 1.0; // バイパス
        c.m1 = 0.0;
        c.m2 = 0.0;
        return c;
    }

    c.a1 = 1.0 / denominator;
    c.a2 = g * c.a1;
    c.a3 = g * c.a2;
    c.m0 = 0.0;
    c.m1 = 0.0;
    c.m2 = 1.0;
    return c;
}

EQCoeffsSVF EQProcessor::calcHighPassSVF(double freq, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double g = std::tan(juce::MathConstants<double>::pi * freq / sr);
    const double k = 1.0 / q;

    // NaN/Infチェック
    if (!std::isfinite(g) || !std::isfinite(k))
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }

    // 除算ゼロ保護
    const double denominator = 1.0 + g * (g + k);
    if (std::abs(denominator) < 1.0e-15)
    {
        c.a1 = 1.0;
        c.a2 = 0.0;
        c.a3 = 0.0;
        c.m0 = 1.0;
        c.m1 = 0.0;
        c.m2 = 0.0;
        return c;
    }

    c.a1 = 1.0 / denominator;
    c.a2 = g * c.a1;
    c.a3 = g * c.a2;
    c.m0 = 1.0;
    c.m1 = -k;
    c.m2 = -1.0;
    return c;
}
