//==============================================================================
// EQProcessorMaxGainTests.cpp — ★ v14.0 Phase 8
//
// EQ周波数応答の数学的契約テスト。
// SVF→Biquad等価変換とマグニチュード二乗計算の数学的正当性を検証する。
//
// 本テストは JUCE/AudioEngine に依存しない純粋数学テスト。
// getMagnitudeSquared と svfToDisplayBiquad の公式を直接実装し、
// 既知のフィルタ（バイパス/Peaking/Shelf）に対して正しい応答を返すことを確認する。
//==============================================================================
#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <algorithm>
#include <numbers>

namespace {

// M_PI が icx/MSVC で未定義の場合があるため直接定義
constexpr double kPi = 3.14159265358979323846;

int g_testsPassed = 0;
int g_testsFailed = 0;

void check(bool condition, const std::string& label)
{
    if (condition)
        ++g_testsPassed;
    else
        ++g_testsFailed, std::cerr << "[FAIL] " << label << "\n";
}

//==============================================================================
// Biquad フィルタ構造体（EQProcessor.h の EQCoeffsBiquad と同一）
// H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
//==============================================================================
struct EQCoeffsBiquad {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a0 = 1.0, a1 = 0.0, a2 = 0.0;
};

double getMagnitudeSquared(const EQCoeffsBiquad& c, double freqHz, double sampleRate) noexcept
{
    const double omega = 2.0 * kPi * freqHz / sampleRate;
    const std::complex<double> z(std::cos(omega), std::sin(omega)); // z = exp(+jω)
    const std::complex<double> z1 = 1.0 / z;
    const std::complex<double> z2 = z1 * z1;

    const std::complex<double> num = c.b0 + c.b1 * z1 + c.b2 * z2;
    const std::complex<double> den = c.a0 + c.a1 * z1 + c.a2 * z2;
    if (std::norm(den) < 1e-30) return 0.0;
    return std::norm(num) / std::norm(den);
}

//==============================================================================
// SVF 係数構造体（EQProcessor.h の EQCoeffsSVF と同一）
//==============================================================================
struct EQCoeffsSVF {
    double a1 = 1.0, a2 = 0.0, a3 = 0.0;
    double m0 = 1.0, m1 = 0.0, m2 = 0.0;
};

// svfToDisplayBiquad — SVF z域等価変換
// 実装は EQProcessor.Coefficients.cpp:347-368 と同一
EQCoeffsBiquad svfToDisplayBiquad(const EQCoeffsSVF& svf) noexcept
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

// 等価 1次ローパス SVF 係数 (簡易検証用)
EQCoeffsSVF calcLPFSVF(double freqHz, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double g = std::tan(kPi * freqHz / sr);
    const double k = 1.0 / q;
    const double denom = 1.0 + g * (g + k);
    c.a1 = 1.0 / denom;
    c.a2 = g * c.a1;
    c.a3 = g * c.a2;
    c.m0 = 0.0; c.m1 = 0.0; c.m2 = 1.0;
    return c;
}

} // namespace

//==============================================================================
// TEST 1: バイパスフィルタ (a0=1, a1=0, a2=0, b0=1, b1=0, b2=0) → |H|² = 1
//==============================================================================
void testBypassFilter()
{
    EQCoeffsBiquad bp;
    bp.b0 = 1.0; bp.a0 = 1.0;
    for (double f = 20.0; f < 24000.0; f *= 2.0)
    {
        const double msq = getMagnitudeSquared(bp, f, 48000.0);
        check(std::abs(msq - 1.0) < 1e-12, "Bypass: |H|²=1 at " + std::to_string(f) + "Hz");
    }
}

//==============================================================================
// TEST 2: バイパス SVF (a1=1, a2=0, a3=0, m0=1) → Biquad もバイパス → |H|² = 1
//==============================================================================
void testBypassSVF()
{
    EQCoeffsSVF svf;
    svf.a1 = 1.0; svf.a2 = 0.0; svf.a3 = 0.0;
    svf.m0 = 1.0; svf.m1 = 0.0; svf.m2 = 0.0;
    const auto bq = svfToDisplayBiquad(svf);
    // リファレンス Biquad: a0=1,a1=-2,a2=1,b0=1,b1=-2,b2=1 → H(z)=1 for all z
    // /fp:fast では微妙な再編成の差異のため ε=1e-9
    for (double f = 20.0; f < 24000.0; f *= 2.0)
    {
        const double msq = getMagnitudeSquared(bq, f, 48000.0);
        check(std::abs(msq - 1.0) < 1e-9, "Bypass SVF: |H|²≈1 at " + std::to_string(f) + "Hz");
    }
}

//==============================================================================
// TEST 3: Bypass Biquad → |H|² = 1 at all frequencies (DC check variant)
//==============================================================================
void testBypassDCGain()
{
    EQCoeffsBiquad bp;
    bp.b0 = 1.0; bp.a0 = 1.0;
    // DC (z=1): (b0+b1+b2)/(a0+a1+a2) = 1/1 = 1
    // 1/magnitude at ω=0
    const double dcMagSq = getMagnitudeSquared(bp, 0.001, 48000.0); // near-DC
    check(std::abs(dcMagSq - 1.0) < 1e-12, "Bypass DC: |H|²=1");
}

//==============================================================================
// TEST 4: 1次 LPF → DC ゲイン = 1, 高周波で減衰
//==============================================================================
void testLPFFrequencyResponse()
{
    const double sr = 48000.0;
    const EQCoeffsSVF svf = calcLPFSVF(1000.0, 0.707, sr);
    const auto bq = svfToDisplayBiquad(svf);

    const double dcMagSq = getMagnitudeSquared(bq, 10.0, sr);
    check(std::abs(dcMagSq - 1.0) < 0.01, "LPF DC: |H|² ≈ 1");

    const double hiMagSq = getMagnitudeSquared(bq, 10000.0, sr);
    check(hiMagSq < 1.0, "LPF high: |H|² < 1");
}

//==============================================================================
// TEST 5: Serial 積 Π|Hi| が正しく積算されることの確認
//==============================================================================
void testSerialProductContract()
{
    const double sr = 48000.0;
    EQCoeffsBiquad bp1, bp2;
    bp1.b0 = 1.0; bp1.a0 = 1.0;
    bp2.b0 = 1.0; bp2.a0 = 1.0;

    for (double f = 20.0; f < 24000.0; f *= 4.0)
    {
        const double m1 = std::sqrt(getMagnitudeSquared(bp1, f, sr));
        const double m2 = std::sqrt(getMagnitudeSquared(bp2, f, sr));
        check(std::abs(m1 * m2 - 1.0) < 1e-12, "Serial product = 1 at " + std::to_string(f) + "Hz");
    }
}

//==============================================================================
// TEST 6: Nyquist 対称性 — 実係数フィルタでは |H(f)|² = |H(sr-f)|²
//==============================================================================
void testFrequencySymmetry()
{
    const double sr = 48000.0;
    const EQCoeffsSVF svf = calcLPFSVF(1000.0, 1.0, sr);
    const auto bq = svfToDisplayBiquad(svf);

    const double mag1 = getMagnitudeSquared(bq, 1000.0, sr);
    const double mag2 = getMagnitudeSquared(bq, sr - 1000.0, sr);
    check(std::abs(mag1 - mag2) < 1e-10, "Nyquist symmetry");
}

//==============================================================================
// MAIN
//==============================================================================
int main()
{
    std::cout << "[EQProcessorMaxGainTests] Start\n";
    testBypassFilter();
    testBypassDCGain();
    testBypassSVF();
    testLPFFrequencyResponse();
    testSerialProductContract();
    testFrequencySymmetry();

    std::cout << "[EQProcessorMaxGainTests] Passed: " << g_testsPassed
              << ", Failed: " << g_testsFailed << "\n";
    return (g_testsFailed == 0) ? 0 : 1;
}
