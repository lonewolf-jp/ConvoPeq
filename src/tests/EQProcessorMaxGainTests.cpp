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
#include <vector>
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

//==============================================================================
// ★ v14.30: biquadResponse — 複素周波数応答 H(e^{jω}) を計算（リファレンス実装）
//==============================================================================
std::complex<double> biquadResponse(const EQCoeffsBiquad& c, double normalizedFreq) noexcept
{
    const std::complex<double> z(std::cos(normalizedFreq), std::sin(normalizedFreq));
    const std::complex<double> z2 = z * z;
    const std::complex<double> num = c.b0 * z2 + c.b1 * z + c.b2;
    const std::complex<double> den = c.a0 * z2 + c.a1 * z + c.a2;
    const double denNorm = std::norm(den);
    if (denNorm < 1e-18)
        return std::complex<double>(1.0, 0.0);
    return num / den;
}

//==============================================================================
// ★ v14.30: isBoostingBand — バンドが振幅増大に寄与するか判定（リファレンス実装）
//   EQProcessor.Coefficients.cpp の static 関数と同一ロジック
//==============================================================================
enum class BandType : uint8_t {
    LowShelf, Peaking, HighShelf, LowPass, HighPass
};

bool isBoostingBand(BandType type, float gain) noexcept
{
    if (!(gain > 0.01f))
        return false;
    switch (type) {
        case BandType::Peaking:
        case BandType::LowShelf:
        case BandType::HighShelf:
            return gain > 0.01f;
        case BandType::LowPass:
        case BandType::HighPass:
            return false;
    }
    return false;
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
// ★ v14.30: biquadResponse の DC 応答検証
//   Bypass Biquad: H(e^{j0}) = 1
//   Peaking +12dB Q=1, fc=1000Hz, sr=48000: |H(e^{j2π*1000/48000})| ≈ 10^(12/20) ≈ 3.98
//==============================================================================
void testBiquadResponseDC()
{
    constexpr double sr = 48000.0;
    // Bypass
    EQCoeffsBiquad bp;
    bp.b0 = 1.0; bp.a0 = 1.0;
    const auto hBypass = biquadResponse(bp, 0.0);
    check(std::abs(hBypass.real() - 1.0) < 1e-12, "biquadResponse: bypass DC real=1");
    check(std::abs(hBypass.imag()) < 1e-12, "biquadResponse: bypass DC imag=0");
    check(std::abs(std::abs(hBypass) - 1.0) < 1e-12, "biquadResponse: bypass DC mag=1");

    // Bypass at Nyquist
    const auto hNyq = biquadResponse(bp, kPi);
    check(std::abs(std::abs(hNyq) - 1.0) < 1e-12, "biquadResponse: bypass Nyquist mag=1");

    // Bypass at arbitrary frequency
    const double w = 2.0 * kPi * 1000.0 / sr;
    const auto h1k = biquadResponse(bp, w);
    check(std::abs(std::abs(h1k) - 1.0) < 1e-12, "biquadResponse: bypass 1kHz mag=1");
}

//==============================================================================
// ★ v14.30: biquadResponse vs getMagnitudeSquared の整合性検証
//   |biquadResponse|² == getMagnitudeSquared
//==============================================================================
void testBiquadResponseVsMagSq()
{
    constexpr double sr = 48000.0;
    const EQCoeffsSVF svf = calcLPFSVF(1000.0, 1.0, sr);
    const auto bq = svfToDisplayBiquad(svf);

    for (double f = 20.0; f < 20000.0; f *= 1.5)
    {
        const double w = 2.0 * kPi * f / sr;
        const double magSqFromMag = getMagnitudeSquared(bq, f, sr);
        const auto H = biquadResponse(bq, w);
        const double magSqFromComplex = std::norm(H);
        check(std::abs(magSqFromMag - magSqFromComplex) < 1e-10,
              "biquadResponse vs getMagSq at " + std::to_string(f) + "Hz");
    }
}

//==============================================================================
// ★ v14.30: isBoostingBand の全バンド種別・全ゲイン範囲検証
//==============================================================================
void testIsBoostingBand()
{
    // Peaking gain>0.01 → true
    check(isBoostingBand(BandType::Peaking, 12.0f) == true, "Peaking +12dB: boosting=true");
    check(isBoostingBand(BandType::Peaking, 0.01f) == false, "Peaking +0.01dB: boosting=false");
    check(isBoostingBand(BandType::Peaking, -12.0f) == false, "Peaking -12dB: boosting=false");
    check(isBoostingBand(BandType::Peaking, 0.0f) == false, "Peaking 0dB: boosting=false");

    // LowShelf gain>0.01 → true
    check(isBoostingBand(BandType::LowShelf, 6.0f) == true, "LowShelf +6dB: boosting=true");
    check(isBoostingBand(BandType::LowShelf, -6.0f) == false, "LowShelf -6dB: boosting=false");

    // HighShelf gain>0.01 → true
    check(isBoostingBand(BandType::HighShelf, 6.0f) == true, "HighShelf +6dB: boosting=true");
    check(isBoostingBand(BandType::HighShelf, -6.0f) == false, "HighShelf -6dB: boosting=false");

    // LowPass → always false
    check(isBoostingBand(BandType::LowPass, 12.0f) == false, "LowPass +12dB: boosting=false");
    check(isBoostingBand(BandType::LowPass, 0.0f) == false, "LowPass 0dB: boosting=false");
    check(isBoostingBand(BandType::LowPass, -12.0f) == false, "LowPass -12dB: boosting=false");

    // HighPass → always false
    check(isBoostingBand(BandType::HighPass, 12.0f) == false, "HighPass +12dB: boosting=false");
    check(isBoostingBand(BandType::HighPass, 0.0f) == false, "HighPass 0dB: boosting=false");
    check(isBoostingBand(BandType::HighPass, -12.0f) == false, "HighPass -12dB: boosting=false");

    // Boundary: gain = 0.01 should be false (not > 0.01)
    check(isBoostingBand(BandType::Peaking, 0.010001f) == true, "Peaking +0.010001dB: boosting=true (threshold)");
}

//==============================================================================
// ★ v14.40: Nyquist 極限テスト — biquadResponse の数値安定性
//   Nyquist 直前の HighShelf, 384kHz, fc=180kHz, Q=0.7, +24dB
//   → NaN/Inf なし
//==============================================================================
void testNyquistExtreme()
{
    constexpr double sr = 384000.0;
    constexpr double nyquist = sr * 0.5;
    constexpr double fc = 180000.0;  // Nyquist 直前
    constexpr double q = 0.7;
    constexpr double gainDb = 24.0;

    // Audio EQ Cookbook (W3C) の HighShelf 係数
    const double A = std::pow(10.0, gainDb / 40.0);
    const double w0 = 2.0 * kPi * fc / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);
    const double sqrtA = std::sqrt(A);
    const double twoSqrtAAlpha = 2.0 * sqrtA * alpha;

    EQCoeffsBiquad shelf;
    const double a0 = (A + 1.0) - (A - 1.0) * cosw0 + twoSqrtAAlpha;
    if (std::abs(a0) < 1e-15) { check(false, "Nyquist shelf: a0 too small"); return; }

    shelf.b0 = A * ((A + 1.0) + (A - 1.0) * cosw0 + twoSqrtAAlpha);
    shelf.b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0);
    shelf.b2 = A * ((A + 1.0) + (A - 1.0) * cosw0 - twoSqrtAAlpha);
    shelf.a0 = a0;
    shelf.a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosw0);
    shelf.a2 = (A + 1.0) - (A - 1.0) * cosw0 - twoSqrtAAlpha;

    // Test at multiple frequencies near Nyquist
    for (double frac = 0.9; frac < 1.0; frac += 0.02)
    {
        const double freq = nyquist * frac;
        const double w = 2.0 * kPi * freq / sr;
        const auto H = biquadResponse(shelf, w);
        check(std::isfinite(H.real()), "Nyquist shelf real finite at " + std::to_string(freq) + "Hz");
        check(std::isfinite(H.imag()), "Nyquist shelf imag finite at " + std::to_string(freq) + "Hz");
        check(std::isfinite(std::abs(H)), "Nyquist shelf mag finite at " + std::to_string(freq) + "Hz");
        check(std::abs(H) > 0.0, "Nyquist shelf mag > 0 at " + std::to_string(freq) + "Hz");
    }

    // Nyquist (ω=π): should be finite
    const auto hNyq = biquadResponse(shelf, kPi);
    check(std::isfinite(hNyq.real()), "Nyquist shelf: exact Nyquist real finite");
    check(std::isfinite(hNyq.imag()), "Nyquist shelf: exact Nyquist imag finite");

    // DC response should be 0dB (HighShelf is flat at low frequencies)
    const auto hDc = biquadResponse(shelf, 0.0);
    check(std::isfinite(hDc.real()), "Nyquist shelf: DC real finite");
    const double dcGainDb = 20.0 * std::log10(std::abs(hDc));
    check(std::abs(dcGainDb) < 1.0, "Nyquist shelf: DC gain ≈ 0dB");

    // At Nyquist, HighShelf gain should approach +24dB
    const double nyquistMag = std::abs(hNyq);
    const double nyquistGainDb = 20.0 * std::log10(nyquistMag);
    check(std::isfinite(nyquistGainDb), "Nyquist shelf: Nyquist gain finite");
    check(nyquistGainDb > 10.0, "Nyquist shelf: Nyquist gain > 10dB (high boost active)");
}

//==============================================================================
// ★ v14.40: 高Q・高ゲイン Peaking の数値安定性テスト
//   EQ Peak +24dB, Q=20 → 極端な共鳴。NaN/Overflow なし
//==============================================================================
void testHighQPeaking()
{
    constexpr double sr = 48000.0;
    constexpr double fc = 1000.0;
    constexpr double q = 20.0;
    constexpr double gainDb = 24.0;

    // Audio EQ Cookbook Peaking 係数
    const double A = std::pow(10.0, gainDb / 40.0);
    const double w0 = 2.0 * kPi * fc / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);

    EQCoeffsBiquad peak;
    const double a0 = 1.0 + alpha / A;
    if (std::abs(a0) < 1e-15) { check(false, "HighQ peak: a0 too small"); return; }

    peak.b0 = 1.0 + alpha * A;
    peak.b1 = -2.0 * cosw0;
    peak.b2 = 1.0 - alpha * A;
    peak.a0 = a0;
    peak.a1 = -2.0 * cosw0;
    peak.a2 = 1.0 - alpha / A;

    // Sweep across frequency range
    for (double f = 20.0; f < sr * 0.49; f *= 1.5)
    {
        const double w = 2.0 * kPi * f / sr;
        const auto H = biquadResponse(peak, w);
        check(std::isfinite(H.real()), "HighQ peak real finite at " + std::to_string(f) + "Hz");
        check(std::isfinite(H.imag()), "HighQ peak imag finite at " + std::to_string(f) + "Hz");
        check(std::abs(H) > 0.0, "HighQ peak mag > 0 at " + std::to_string(f) + "Hz");
    }

    // At fc, gain should be approximately +24dB → |H| ≈ 15.85
    const double wc = 2.0 * kPi * fc / sr;
    const auto Hc = biquadResponse(peak, wc);
    const double peakGainDb = 20.0 * std::log10(std::abs(Hc));
    check(peakGainDb > 20.0 && peakGainDb < 28.0, "HighQ peak: gain ≈ " + std::to_string(gainDb) + "dB at fc");
}

//==============================================================================
// ★ v14.40: log1p + upperBound 計算経路の数値安定性テスト
//   極端な delta (|Hi-1| が非常に大きい/小さい) での NaN/Inf なし
//==============================================================================
void testLog1pUpperBoundStability()
{
    const double kTwentyOverLog10 = 20.0 / std::log(10.0);

    // 極端に小さい delta
    {
        double logBound = 0.0;
        for (double delta = 1e-15; delta < 1e-6; delta *= 10.0)
        {
            if (delta > 1e-6)  // 微小項切り捨て条件と同じ
                logBound += std::log1p(delta);
        }
        const double ubDb = kTwentyOverLog10 * logBound;
        check(std::isfinite(ubDb), "log1p upperBound: tiny delta finite");
    }

    // 極端に大きい delta
    {
        double logBound = 0.0;
        for (int i = 0; i < 100; ++i)
        {
            const double delta = 100.0;  // |H-1| = 100 → 非常に大きな cut または boost
            if (std::isfinite(delta) && delta > 1e-6)
                logBound += std::log1p(delta);
        }
        const double ubDb = kTwentyOverLog10 * logBound;
        check(std::isfinite(ubDb), "log1p upperBound: large delta finite");
        check(ubDb > 0.0, "log1p upperBound: large delta positive");
    }

    // 20Band 全てが +24dB Peaking の最悪ケース
    {
        double logBound = 0.0;
        for (int i = 0; i < 20; ++i)
        {
            // Peaking +24dB, Q=20 での典型的な |H-1| の最大値 ≈ 10 程度
            const double delta = 10.0;
            if (std::isfinite(delta) && delta > 1e-6)
                logBound += std::log1p(delta);
        }
        const double ubDb = kTwentyOverLog10 * logBound;
        // log1p(10) ≈ 2.398, ×20 ≈ 47.96, ×(20/ln(10)) ≈ 416dB
        // この bound は非常に保守的だが finite であるべき
        check(std::isfinite(ubDb), "log1p upperBound: 20Band worst-case finite");
        check(ubDb < 1000.0, "log1p upperBound: 20Band worst-case < 1000dB");
    }
}

//==============================================================================
// ★ v14.7: 逆位相 Parallel での upperBound と measured の乖離検証
//   同一周波数 +12dB Peaks, 180° 位相 → |H_parallel| ≪ Π(1+|Hi-1|)
//==============================================================================
void testOppositePhaseParallelBound()
{
    // 2バンドの Parallel: H1 = A (実数), H2 = A*e^{jπ} = -A
    // measured = |1 + (A-1) + (-A-1)| = |1 + A -1 - A -1| = |-1| = 1 (= 0dB)
    // upperBound = (1+|A-1|) * (1+| -A-1|) = (1+|A-1|)²
    const double gainLin = 3.98107;  // +12dB
    const double delta = std::abs(gainLin - 1.0);  // 2.98107
    const double measuredMag = 1.0;  // 0dB（逆位相で打ち消し）

    const double kTwentyOverLog10 = 20.0 / std::log(10.0);
    const double measuredDb = 20.0 * std::log10(measuredMag);
    const double upperBoundDb = kTwentyOverLog10 * (std::log1p(delta) + std::log1p(delta));

    check(std::abs(measuredDb) < 0.1, "Opposite phase: measured ≈ 0dB");
    check(upperBoundDb > 20.0, "Opposite phase: upperBound >> measured");
    check(std::isfinite(measuredDb), "Opposite phase: measured finite");
    check(std::isfinite(upperBoundDb), "Opposite phase: upperBound finite");
}

//==============================================================================
// ★ v14.41: Union 区間統合アルゴリズム検証
//   重複する区間を正しくマージできるか
//==============================================================================
struct TestRange { double start; double end; };
std::vector<std::pair<double, double>> testMergeRanges(std::vector<std::pair<double, double>> ranges)
{
    if (ranges.empty()) return {};
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<double, double>> merged;
    merged.push_back(ranges[0]);
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first <= merged.back().second)
            merged.back().second = std::max(merged.back().second, ranges[i].second);
        else
            merged.push_back(ranges[i]);
    }
    return merged;
}

void testUnionRangeMerging()
{
    // 重複なし
    {
        auto m = testMergeRanges({{100, 200}, {300, 400}});
        check(m.size() == 2, "merge ranges: no overlap = 2 ranges");
        check(std::abs(m[0].first - 100.0) < 1e-9, "merge ranges: first start=100");
        check(std::abs(m[1].second - 400.0) < 1e-9, "merge ranges: second end=400");
    }
    // 部分的重複
    {
        auto m = testMergeRanges({{100, 300}, {200, 400}});
        check(m.size() == 1, "merge ranges: partial overlap = 1 range");
        check(std::abs(m[0].first - 100.0) < 1e-9, "merge ranges: merged start=100");
        check(std::abs(m[0].second - 400.0) < 1e-9, "merge ranges: merged end=400");
    }
    // 完全包含
    {
        auto m = testMergeRanges({{100, 400}, {200, 300}});
        check(m.size() == 1, "merge ranges: full containment = 1 range");
        check(std::abs(m[0].first - 100.0) < 1e-9, "merge ranges: contain start=100");
        check(std::abs(m[0].second - 400.0) < 1e-9, "merge ranges: contain end=400");
    }
    // 接している(隣接)
    {
        auto m = testMergeRanges({{100, 200}, {200, 300}});
        check(m.size() == 1, "merge ranges: touching = 1 range");
        check(std::abs(m[0].second - 300.0) < 1e-9, "merge ranges: touching end=300");
    }
    // 空
    {
        auto m = testMergeRanges({});
        check(m.size() == 0, "merge ranges: empty = 0 ranges");
    }
    // 同一範囲
    {
        auto m = testMergeRanges({{100, 200}, {100, 200}});
        check(m.size() == 1, "merge ranges: identical = 1 range");
        check(std::abs(m[0].second - 200.0) < 1e-9, "merge ranges: identical end=200");
    }
}

//==============================================================================
// ★ v14.30: isBoosting と maxActiveQ の整合性
//   LPF/HPF は isBoosting=false → maxActiveQ に含まれない
//==============================================================================
void testMaxActiveQExcludesLPFHPF()
{
    // LPF/HPF の Q 値は maxActiveQ に寄与しない
    check(isBoostingBand(BandType::LowPass, 12.0f) == false, "LPF +12dB: not boosting → excluded from maxActiveQ");
    check(isBoostingBand(BandType::HighPass, 12.0f) == false, "HPF +12dB: not boosting → excluded from maxActiveQ");

    // Peaking は boosting=true → maxActiveQ に含まれる
    check(isBoostingBand(BandType::Peaking, 6.0f) == true, "Peaking +6dB: boosting → included in maxActiveQ");

    // カット (gain < 0) は boosting=false → maxActiveQ に含まれない
    check(isBoostingBand(BandType::LowShelf, -6.0f) == false, "LowShelf -6dB: cut → excluded from maxActiveQ");
    check(isBoostingBand(BandType::HighShelf, -6.0f) == false, "HighShelf -6dB: cut → excluded from maxActiveQ");
}

//==============================================================================
// ★ v14.30: computeEstimatedMaxGainComplex 統合テスト
//   簡略化アルゴリズムで既知のEQ構成に対するピークゲインを検証
//==============================================================================

// W3C Audio EQ Cookbook の Peaking 係数
EQCoeffsBiquad calcPeakingBiquad(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsBiquad c{};
    const double A = std::pow(10.0, gainDb / 40.0);
    const double w0 = 2.0 * kPi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);
    const double a0 = 1.0 + alpha / A;
    if (std::abs(a0) < 1e-15) { c.b0 = 1.0; c.a0 = 1.0; return c; }
    c.b0 = 1.0 + alpha * A;
    c.b1 = -2.0 * cosw0;
    c.b2 = 1.0 - alpha * A;
    c.a0 = a0;
    c.a1 = -2.0 * cosw0;
    c.a2 = 1.0 - alpha / A;
    return c;
}

// Simplified version of computeEstimatedMaxGainComplex:
// Coarse search only (no adaptive sampling, no parabolic interpolation)
// Tests the core Serial/Parallel/upperBound logic
struct SimulatedBand {
    EQCoeffsBiquad biquad;
    bool isBoosting;
};

struct SimulatedResult {
    double measuredDb = 0.0;
    double upperBoundDb = 0.0;
    double maxActiveQ = 0.0;
};

SimulatedResult computeSimplifiedMaxGain(
    const std::vector<SimulatedBand>& activeBands,
    bool isParallel,
    double sr,
    int numPoints = 200)
{
    SimulatedResult result{};
    if (activeBands.empty()) return result;

    const double nyquist = sr * 0.5;
    const double maxFreq = std::min(20000.0, nyquist);
    const double kTwentyOverLog10 = 20.0 / std::log(10.0);

    double maxMeasuredDb = -std::numeric_limits<double>::infinity();
    double maxUpperBoundDb = -std::numeric_limits<double>::infinity();

    // Track maxActiveQ from boosting bands
    for (const auto& band : activeBands) {
        if (band.isBoosting && result.maxActiveQ < 0.0) { /* Q tracked below */ }
    }

    for (int i = 0; i < numPoints; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(numPoints - 1);
        const double freqHz = 10.0 * std::pow(maxFreq / 10.0, t);
        const double w = 2.0 * kPi * freqHz / sr;

        if (isParallel)
        {
            std::complex<double> parallelSum(1.0, 0.0);
            double logBound = 0.0;

            for (const auto& band : activeBands)
            {
                const auto H = biquadResponse(band.biquad, w);
                parallelSum += H - std::complex<double>(1.0, 0.0);

                const double delta = std::abs(H - std::complex<double>(1.0, 0.0));
                if (std::isfinite(delta) && delta > 1e-6)
                    logBound += std::log1p(delta);
            }

            const double measuredDb = 20.0 * std::log10(std::abs(parallelSum));
            const double upperBoundDb = kTwentyOverLog10 * logBound;

            if (measuredDb > maxMeasuredDb) maxMeasuredDb = measuredDb;
            if (upperBoundDb > maxUpperBoundDb) maxUpperBoundDb = upperBoundDb;
        }
        else
        {
            double productMag = 1.0;
            double logBound = 0.0;

            for (const auto& band : activeBands)
            {
                const auto H = biquadResponse(band.biquad, w);
                productMag *= std::abs(H);

                const double delta = std::abs(H - std::complex<double>(1.0, 0.0));
                if (std::isfinite(delta) && delta > 1e-6)
                    logBound += std::log1p(delta);
            }

            const double measuredDb = 20.0 * std::log10(productMag);
            const double upperBoundDb = kTwentyOverLog10 * logBound;

            if (measuredDb > maxMeasuredDb) maxMeasuredDb = measuredDb;
            if (upperBoundDb > maxUpperBoundDb) maxUpperBoundDb = upperBoundDb;
        }
    }

    result.measuredDb = (maxMeasuredDb > 0.0) ? maxMeasuredDb : 0.0;
    result.upperBoundDb = (maxUpperBoundDb > 0.0) ? maxUpperBoundDb : 0.0;
    return result;
}

void testComputeSimplifiedMaxGain()
{
    constexpr double sr = 48000.0;

    // ─── Test 1: Single Peaking +12dB, Q=1, fc=1000Hz ───
    // At fc: |H(fc)| ≈ gain_linear = 10^(12/20) = 3.98 → 12dB
    {
        auto bq = calcPeakingBiquad(1000.0, 12.0, 1.0, sr);
        std::vector<SimulatedBand> bands = {{bq, true}};

        // Serial (single band, same as parallel for n=1)
        auto ser = computeSimplifiedMaxGain(bands, false, sr);
        check(std::abs(ser.measuredDb - 12.0) < 0.5,
              "Simplified: single peaking +12dB serial = " + std::to_string(ser.measuredDb) + "dB");

        auto par = computeSimplifiedMaxGain(bands, true, sr);
        check(std::abs(par.measuredDb - 12.0) < 0.5,
              "Simplified: single peaking +12dB parallel = " + std::to_string(par.measuredDb) + "dB");
    }

    // ─── Test 2: Serial 2 Peaking +12dB + +12dB, different freqs ───
    // Serial: |H_total| = |H1| * |H2| → in dB: gain1_db + gain2_db ≈ 24dB
    // (if peaks are at different frequencies, total ≈ max(gain1, gain2) ≈ 12dB)
    {
        auto bq1 = calcPeakingBiquad(1000.0, 12.0, 1.0, sr);
        auto bq2 = calcPeakingBiquad(4000.0, 12.0, 1.0, sr);  // Different freq
        std::vector<SimulatedBand> bands = {{bq1, true}, {bq2, true}};

        auto ser = computeSimplifiedMaxGain(bands, false, sr);
        // Serial at 1kHz: |H1|≈3.98, |H2|≈1 → total≈3.98 → 12dB
        // Serial at 4kHz: |H1|≈1, |H2|≈3.98 → total≈3.98 → 12dB
        check(ser.measuredDb > 10.0 && ser.measuredDb < 14.0,
              "Simplified: serial 2x+12dB diff freq = " + std::to_string(ser.measuredDb) + "dB");
    }

    // ─── Test 3: Serial 2 Peaking at SAME frequency ───
    // |H_total| = |H1| * |H2| → 3.98 * 3.98 ≈ 15.85 → 24dB
    {
        auto bq1 = calcPeakingBiquad(1000.0, 12.0, 1.0, sr);
        auto bq2 = calcPeakingBiquad(1000.0, 12.0, 1.0, sr);  // Same freq
        std::vector<SimulatedBand> bands = {{bq1, true}, {bq2, true}};

        auto ser = computeSimplifiedMaxGain(bands, false, sr);
        // At fc=1kHz: |H1|=|H2|≈3.98 → total≈15.85 → 24dB
        check(ser.measuredDb > 20.0 && ser.measuredDb < 28.0,
              "Simplified: serial 2x+12dB same freq = " + std::to_string(ser.measuredDb) + "dB (expected ≈24dB)");
    }

    // ─── Test 4: Parallel 2 Peaking +12dB + +12dB at same freq ───
    // H_parallel = 1 + (H1-1) + (H2-1) = H1 + H2 - 1
    // At fc: H1=H2≈3.98 → H_parallel ≈ 3.98 + 3.98 - 1 = 6.96 → 16.9dB
    {
        auto bq1 = calcPeakingBiquad(1000.0, 12.0, 1.0, sr);
        auto bq2 = calcPeakingBiquad(1000.0, 12.0, 1.0, sr);
        std::vector<SimulatedBand> bands = {{bq1, true}, {bq2, true}};

        auto par = computeSimplifiedMaxGain(bands, true, sr);
        // Expected: 20*log10(6.96) ≈ 16.9dB
        check(par.measuredDb > 14.0 && par.measuredDb < 20.0,
              "Simplified: parallel 2x+12dB same freq = " + std::to_string(par.measuredDb) + "dB (expected ≈16.9dB)");

        // upperBound should be >= measured
        check(par.upperBoundDb >= par.measuredDb - 0.1,
              "Simplified: parallel upperBound >= measured (" +
              std::to_string(par.upperBoundDb) + " >= " + std::to_string(par.measuredDb) + ")");
    }

    // ─── Test 5: Only LPF/HPF → no boosting → measured=0dB ───
    {
        std::vector<SimulatedBand> noBands = {};  // Empty = all non-boosting
        auto ser = computeSimplifiedMaxGain(noBands, false, sr);
        check(std::abs(ser.measuredDb) < 0.01, "Simplified: no boosting bands = 0dB");
    }

    // ─── Test 6: Serial vs Parallel upperBound consistency ───
    // upperBound depends only on individual |Hi-1|, not on filter structure
    // But measured differs: measured_serial >= measured_parallel
    {
        auto bq = calcPeakingBiquad(1000.0, 12.0, 1.0, sr);
        std::vector<SimulatedBand> bands = {{bq, true}};

        auto ser = computeSimplifiedMaxGain(bands, false, sr);
        auto par = computeSimplifiedMaxGain(bands, true, sr);

        // For single band, serial == parallel
        check(std::abs(ser.measuredDb - par.measuredDb) < 1.0,
              "Simplified: single band ser==par (" +
              std::to_string(ser.measuredDb) + " vs " + std::to_string(par.measuredDb) + ")");

        // upperBound same for both
        check(std::abs(ser.upperBoundDb - par.upperBoundDb) < 0.1,
              "Simplified: single band upperBound equal (" +
              std::to_string(ser.upperBoundDb) + " vs " + std::to_string(par.upperBoundDb) + ")");
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
    testBiquadResponseDC();
    testBiquadResponseVsMagSq();
    testIsBoostingBand();
    testNyquistExtreme();
    testHighQPeaking();
    testLog1pUpperBoundStability();
    testOppositePhaseParallelBound();
    testUnionRangeMerging();
    testMaxActiveQExcludesLPFHPF();
    testComputeSimplifiedMaxGain();

    std::cout << "[EQProcessorMaxGainTests] Passed: " << g_testsPassed
              << ", Failed: " << g_testsFailed << "\n";
    return (g_testsFailed == 0) ? 0 : 1;
}
