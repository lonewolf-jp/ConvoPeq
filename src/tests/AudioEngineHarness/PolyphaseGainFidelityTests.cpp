//============================================================================
// PolyphaseGainFidelityTests.cpp — Phase 0 REF-FIDELITY 基盤（test-only・40 番目 target）
// work113 / remediation v3.1 §2.7 0-1b + §2.9（R15-3 + R16-1 + R17-1〜R17-4）
//
// production src/ 変更 0。production CustomInputOversampler.cpp を同一 target・同一
// compile option（R17-3: 同一 BUILD-ID / scalar↔scalar / SIMD↔corresponding SIMD）で
// compile し、PolyphaseGainCandidateRef.h の Shadow 独立 reference model と
// bitwise 比較する。
//
// 出力: [BUILD] 6 要素ブロック（R15-1 + R16-3）→ 各 test PASS/FAIL 行 → 総合判定。
//       FAIL は fail-closed（exit 1）。
//
// ★ 位置付け（v3.1 R17-7・外部監査 §2 を反映）:
//   本 exe の REF-FIDELITY は「Shadow fidelity gate」であり production 正しさの gate では
//   ない（R12-8）。P0-I の事前数学検証（tmp/v31_p0i_attribution_check.py）は
//   「P0-I 実装済み PASS ではなく契約の事前数学検証 PASS」であり、0-5 の SoftClip 実測は
//   G-0/C0 承認後の Phase 0 characterization で実施する。
//============================================================================
#include <JuceHeader.h>
#include "CustomInputOversampler.h"
#include "DspNumericPolicy.h"
#include "PolyphaseGainCandidateRef.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr int kBlockSize = 8192;     // 全検証共通の 1 ブロック入力数
constexpr double kDcTol = 1.0e-9;    // DC 期待値の float 許容（bitwise 比較とは別系統）

// ── deterministic PRNG（seed 固定・libm 不使用 → ビルド構成非依存） ─────────
struct DeterministicPRNG
{
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;
    explicit DeterministicPRNG(std::uint64_t seed) noexcept
        : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    DeterministicPRNG() = default;
    double next() noexcept
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
    }
};

std::size_t g_pass = 0;
std::size_t g_fail = 0;

void report(const char* name, bool ok, const std::string& detail = {})
{
    if (ok) { ++g_pass; std::printf("  [PASS] %s %s\n", name, detail.c_str()); }
    else    { ++g_fail; std::printf("  [FAIL] %s %s\n", name, detail.c_str()); }
    std::fflush(stdout);
}

// bitwise 比較（±0・NaN・全ビット厳密）
bool bitwiseEqual(const double* a, const double* b, std::size_t n) noexcept
{
    for (std::size_t i = 0; i < n; ++i)
        if (std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(b[i]))
            return false;
    return true;
}

std::string firstDiff(const double* a, const double* b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        if (std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(b[i]))
        {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "@[%zu] %.17g vs %.17g", i, a[i], b[i]);
            return buf;
        }
    return {};
}

std::string fmt(const char* format, ...)
{
    char buf[256];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);
    return buf;
}

// ── [BUILD] 6 要素ブロック（R15-1 + R16-3） ───────────────────────────────
// exe 側は (d) production_flag / (e) shadow_candidate / (f) build configuration を確定。
// (a)(b)(c) は runner が build_identity_gate.py --emit-build-id で log 先頭に連結する。
void emitBuildIdBlock()
{
    std::printf("[BUILD]\n");
    std::printf("  snapshot_identity : (runner: build_identity_gate.py --emit-build-id)\n");
    std::printf("  git_head          : (runner 補完)\n");
    std::printf("  working_tree      : (runner 補完)\n");
#ifdef CONVOPEQ_CORRECT_POLYPHASE_GAIN
    std::printf("  production_flag   : 1 (UNEXPECTED — Phase 0 契約違反: R15-2)\n");
#else
    std::printf("  production_flag   : undefined/off (macro absent in this TU / Phase 0 R15-2)\n");
#endif
    std::printf("  shadow_candidate  : %d (CONVOPEQ_POLYPHASE_REF_CANDIDATE)\n",
                CONVOPEQ_POLYPHASE_REF_CANDIDATE);
#if defined(__AVX2__) && defined(__FMA__)
    std::printf("  build_config      : MSVC %d / AVX2+FMA / C++20\n", _MSC_VER);
#elif defined(__AVX2__)
    std::printf("  build_config      : MSVC %d / AVX2 (no __FMA__ macro) / C++20\n", _MSC_VER);
#else
    std::printf("  build_config      : MSVC %d / scalar / C++20\n", _MSC_VER);
#endif
    std::printf("[BUILD-END]\n");
    std::fflush(stdout);
}

using Shadow = convo::polyphase_ref::PolyphaseGainShadow;

// ── round-trip 1 ブロック runners（production / shadow 共通形式） ─────────
//   戻り値: dn サンプル数（= n）/ guard 発動時 -1
// ── round-trip 1 ブロック runners（production / shadow 共通形式） ─────────
//   戻り値: dn サンプル数（= n）/ guard 発動時 -1
int runProdRoundTrip(CustomInputOversampler& prod,
                     double* const* in, int numSamples,
                     double* const* upOut, double* const* dnOut,
                     int expectedRatio)
{
    juce::dsp::AudioBlock<double> inBlock(in, 2, static_cast<std::size_t>(numSamples));
    auto upBlock = prod.processUp(inBlock, 2);
    const std::size_t gotUp = upBlock.getNumSamples();
    for (int ch = 0; ch < 2; ++ch)
        std::memcpy(upOut[ch], upBlock.getChannelPointer(ch), gotUp * sizeof(double));
    if (gotUp != static_cast<std::size_t>(numSamples) * expectedRatio)
        return -1;

    juce::dsp::AudioBlock<double> upSrcBlock(upOut, 2, gotUp);
    juce::dsp::AudioBlock<double> outBlock(dnOut, 2, static_cast<std::size_t>(numSamples));
    prod.processDown(upSrcBlock, outBlock, 2);
    return numSamples;
}

int runShadowRoundTrip(Shadow& shadow,
                       const double* const* in, int numSamples,
                       double* const* upOut, double* const* dnOut,
                       int expectedRatio)
{
    const int gotUp = shadow.processUp(in, numSamples, upOut);
    if (gotUp != numSamples * expectedRatio)
        return -1;
    const double* upIn[2] = { upOut[0], upOut[1] };
    return shadow.processDown(upIn, numSamples * expectedRatio, dnOut);
}

// prod と shadow の round-trip を bitwise 比較する
std::pair<bool, std::string> compareRoundTrip(CustomInputOversampler& prod, Shadow& shadow,
                                              double* const* in, int numSamples,
                                              double* const* prodUp, double* const* prodDn,
                                              double* const* shadowUp, double* const* shadowDn)
{
    const int ratio = shadow.ratio();
    if (runProdRoundTrip(prod, in, numSamples, prodUp, prodDn, ratio) != numSamples)
        return { false, "prod round-trip failed" };
    const int gotShadow = runShadowRoundTrip(shadow, in, numSamples, shadowUp, shadowDn, ratio);
    if (gotShadow != numSamples)
        return { false, fmt("shadow round-trip samples=%d", gotShadow) };

    const std::size_t nUp = static_cast<std::size_t>(numSamples) * ratio;
    const std::size_t nDn = static_cast<std::size_t>(numSamples);
    for (int ch = 0; ch < 2; ++ch)
    {
        if (!bitwiseEqual(prodUp[ch], shadowUp[ch], nUp))
            return { false, fmt("up ch%d %s", ch, firstDiff(prodUp[ch], shadowUp[ch], nUp).c_str()) };
        if (!bitwiseEqual(prodDn[ch], shadowDn[ch], nDn))
            return { false, fmt("dn ch%d %s", ch, firstDiff(prodDn[ch], shadowDn[ch], nDn).c_str()) };
    }
    return { true, {} };
}

double steadyMean(const double* x, std::size_t n) noexcept
{
    double s = 0.0;
    for (std::size_t i = n / 2; i < n; ++i)
        s += x[i];
    return s / static_cast<double>(n - n / 2);
}

// ── 共通設定表 ────────────────────────────────────────────────────────────
struct Config
{
    int ratio;
    CustomInputOversampler::Preset preset;
    const char* name;
};
constexpr Config kConfigs[] =
{
    { 2, CustomInputOversampler::Preset::IIRLike,     "IIR3 r=2" },
    { 4, CustomInputOversampler::Preset::IIRLike,     "IIR3 r=4" },
    { 8, CustomInputOversampler::Preset::IIRLike,     "IIR3 r=8" },
    { 2, CustomInputOversampler::Preset::LinearPhase, "LP3  r=2" },
    { 4, CustomInputOversampler::Preset::LinearPhase, "LP3  r=4" },
    { 8, CustomInputOversampler::Preset::LinearPhase, "LP3  r=8" },
};

Shadow::Preset toShadowPreset(CustomInputOversampler::Preset p) noexcept
{
    return (p == CustomInputOversampler::Preset::LinearPhase)
                ? Shadow::Preset::LinearPhase : Shadow::Preset::IIRLike;
}

// ── R17-4 (A) coefficient invariant × 7 design（shadow 内部構造 assert） ────
// 注: production Stage は private のため、production 側係数の一致は本体 REF-FIDELITY
//     （本体試験）の bitwise 出力比較で検証する（より強い証拠）。本検証は Shadow の
//     10 手順契約（R16-1）の構造不変式を閉じた形で確定する。
void testCoefficientInvariants()
{
    struct Design { int taps; double atten; };
    static constexpr Design designs[] =
    {
        { 511, 140.0 }, { 127, 110.0 }, { 31, 90.0 },
        { 1023, 160.0 }, { 255, 140.0 }, { 63, 120.0 },
        { 3, 90.0 },   // convCount=2 → scalar↔scalar 経路（R17-3 の scalar 契約）
    };

    bool allOk = true;
    std::string detail;
    for (const auto& d : designs)
    {
        convo::polyphase_ref::StageModel st;
        if (!convo::polyphase_ref::prepareStageModel(st, d.taps, d.atten, kBlockSize))
        {
            report("R17-4(A) coefficient invariant", false, "prepareStageModel failed");
            return;
        }

        const int eTaps = (3 > (d.taps | 1)) ? 3 : (d.taps | 1);
        const int eCenter = (eTaps - 1) / 2;
        const int eCPar = eCenter & 1;
        const int eVPar = 1 - eCPar;
        const int eCount = (eTaps - eVPar + 1) / 2;
        const int eCdi = (eCenter - eCPar) / 2;
        const int eUpKeep = (eCount - 1 > eCdi) ? (eCount - 1) : eCdi;
        const int eDnKeep = (eCenter > (eVPar + ((eCount - 1) << 1) + 6))
                          ? eCenter : (eVPar + ((eCount - 1) << 1) + 6);

        bool ok = (st.taps == eTaps) && (st.centerTap == eCenter)
               && (st.centerParity == eCPar) && (st.convParity == eVPar)
               && (st.convCount == eCount) && (st.centerDelayInput == eCdi)
               && (st.historyUpKeep == eUpKeep) && (st.historyDownKeep == eDnKeep)
               && (st.upHistorySize == eUpKeep + kBlockSize + 16)
               && (st.downHistorySize == eDnKeep + 2 * kBlockSize + 16)
               && (st.convCoeffs.size() == static_cast<std::size_t>(eCount))
               && (st.convCoeffsReversed.size() == static_cast<std::size_t>(eCount));

        double firSum = st.centerCoeff;
        double convSum = 0.0;
        for (std::size_t r = 0; r < st.convCoeffs.size(); ++r)
        {
            convSum += st.convCoeffs[r];
            if (std::bit_cast<std::uint64_t>(st.convCoeffsReversed[r])
                != std::bit_cast<std::uint64_t>(st.convCoeffs[st.convCoeffs.size() - 1 - r]))
                ok = false;
        }
        firSum += convSum;

        ok = ok
          && (std::bit_cast<std::uint64_t>(st.centerCoeff) == std::bit_cast<std::uint64_t>(0.5))
          && (std::fabs(firSum - 1.0) <= 1.0e-12)
          && (std::fabs(convSum - 0.5) <= 1.0e-12);

        if (!ok)
        {
            allOk = false;
            detail = fmt("design %d/%.0f firSum=%.17g convSum=%.17g", d.taps, d.atten, firSum, convSum);
            break;
        }
    }
    report("R17-4(A) coefficient invariant ×7 design (FIRsum=1.0 / center=0.5 / convSum=0.5 / 構造式)",
           allOk, detail);
}

// ═════════════════════════════════════════════════════════════════════════════
// Phase 0 characterization（P0-A〜P0-I・R17-1 動作点固定 / R12-3 責務分離）
//   C1 PASS ≠ B-1 PASS ≠ 案E採用 — 本 battery は Phase 0 の gate 実測であり、
//   production 変更（centerValue *= 2.0 等）は引き続き HOLD。
//   production 正しさの gate は Shadow ではなく P0 gate 自体（R12-8 の性格注記は維持）。
// ═════════════════════════════════════════════════════════════════════════════

// ── 単点 DFT（Goertzel 型直接評価・決定論的加算順序） ──
//   freq は cycles/sample。|Σ x[k]·e^{−j2π·freq·k}| を返す。
double dftMag(const double* x, std::size_t n, double freq)
{
    const double w = 2.0 * juce::MathConstants<double>::pi * freq;
    double re = 0.0;
    double im = 0.0;
    for (std::size_t k = 0; k < n; ++k)
    {
        const double ang = w * static_cast<double>(k);
        re += x[k] * std::cos(ang);
        im -= x[k] * std::sin(ang);
    }
    return std::sqrt(re * re + im * im);
}

// impulse response → 指定周波数の |H|（dB）
double dtftDb(const std::vector<double>& h, double freq)
{
    const double mag = dftMag(h.data(), h.size(), freq);
    return 20.0 * std::log10(mag + 1.0e-300);
}

// ── P0-I: production musicalSoftClipScalar の点wise 参照実装（R12-4 harness 責務） ──
//   DSPCoreDouble.cpp:107-131 と同一アルゴリズムの harness 転写（scalar 経路・
//   SoftClipPadePolicy::compute と同一 10395 Padé・FastTanhApprox.h:104-106 と同一クランプ）。
double fastTanhPade10395Ref(double x) noexcept
{
    if (x >= 4.5) return 1.0;
    if (x <= -4.5) return -1.0;
    const double x2 = x * x;
    return x * (10395.0 + x2 * (1260.0 + x2 * 21.0))
         / (10395.0 + x2 * (4725.0 + x2 * (210.0 + x2)));
}

struct SoftClipParams { double threshold; double knee; double asymmetry; };
constexpr SoftClipParams kP0iOp{ 0.50, 0.40, 0.10 };   // production マッピング sat=1.0（DSPCoreDouble.cpp:487-489）

double musicalSoftClipRef(double x, const SoftClipParams& p) noexcept
{
    const double abs_x = std::fabs(x);
    const double clip_start = p.threshold - p.knee;

    if (p.knee < 1.0e-9)
        return (x > p.threshold) ? p.threshold : ((x < -p.threshold) ? -p.threshold : x);
    if (abs_x < clip_start)
        return x;

    const double sign = (x > 0.0) ? 1.0 : -1.0;
    double knee_shape = 1.0;
    if (abs_x < p.threshold + p.knee)
    {
        const double t = (abs_x - clip_start) / (2.0 * p.knee);
        knee_shape = t * t * (3.0 - 2.0 * t);
    }
    const double linear = abs_x;
    const double clipped = p.threshold + p.knee * fastTanhPade10395Ref((abs_x - p.threshold) / p.knee);
    const double asymmetric_gain = 1.0 - p.asymmetry * (1.0 - sign) * 0.5 * knee_shape;
    return sign * (linear * (1.0 - knee_shape) + clipped * knee_shape) * asymmetric_gain;
}

// float 語転写（P0-G pair b 用・production float 経路と同一アルゴリズム構造）
float musicalSoftClipRefF(float xf, float thf, float knf, float asf) noexcept
{
    const float abs_x = std::fabs(xf);
    const float clip_start = thf - knf;
    if (knf < 1.0e-9f)
        return (xf > thf) ? thf : ((xf < -thf) ? -thf : xf);
    if (abs_x < clip_start)
        return xf;
    const float sign = (xf > 0.0f) ? 1.0f : -1.0f;
    float knee_shape = 1.0f;
    if (abs_x < thf + knf)
    {
        const float t = (abs_x - clip_start) / (2.0f * knf);
        knee_shape = t * t * (3.0f - 2.0f * t);
    }
    const float linear = abs_x;
    const float argf = (abs_x - thf) / knf;
    float tanhf;
    if (argf >= 4.5f) tanhf = 1.0f;
    else if (argf <= -4.5f) tanhf = -1.0f;
    else { const float x2 = argf * argf;
           tanhf = argf * (10395.0f + x2 * (1260.0f + x2 * 21.0f))
                 / (10395.0f + x2 * (4725.0f + x2 * (210.0f + x2))); }
    const float clipped = thf + knf * tanhf;
    const float asymmetric_gain = 1.0f - asf * (1.0f - sign) * 0.5f * knee_shape;
    return sign * (linear * (1.0f - knee_shape) + clipped * knee_shape) * asymmetric_gain;
}

// ── shadow 係数 → rawCoeffs 再構築（P0-D per-design 測定用） ──
std::vector<double> rawCoeffsFromStage(const convo::polyphase_ref::StageModel& st)
{
    std::vector<double> raw(static_cast<std::size_t>(st.taps), 0.0);
    raw[static_cast<std::size_t>(st.centerTap)] = st.centerCoeff;
    for (int r = 0; r < st.convCount; ++r)
        raw[static_cast<std::size_t>(st.convParity + 2 * r)] = st.convCoeffs[static_cast<std::size_t>(r)];
    return raw;
}

void runPhase0Tests()
{
    std::printf("  --- Phase 0 characterization（P0-A〜P0-I / P0-G・R17-1 動作点 / R12-3 責務分離） ---\n");

    std::vector<double> inA[2];
    std::vector<double> bUp[2], bDn[2], cUp[2], cDn[2];
    for (int ch = 0; ch < 2; ++ch)
    {
        inA[ch].assign(static_cast<std::size_t>(kBlockSize), 0.0);
        bUp[ch].assign(static_cast<std::size_t>(8 * kBlockSize), 0.0);
        bDn[ch].assign(static_cast<std::size_t>(kBlockSize), 0.0);
        cUp[ch].assign(static_cast<std::size_t>(8 * kBlockSize), 0.0);
        cDn[ch].assign(static_cast<std::size_t>(kBlockSize), 0.0);
    }
    double* inP[2] = { inA[0].data(), inA[1].data() };
    double* bUpP[2] = { bUp[0].data(), bUp[1].data() };
    double* bDnP[2] = { bDn[0].data(), bDn[1].data() };
    double* cUpP[2] = { cUp[0].data(), cUp[1].data() };
    double* cDnP[2] = { cDn[0].data(), cDn[1].data() };
    const int w = kBlockSize / 2;
    constexpr std::size_t kProbeWindow = 2048;
    const double kPi = juce::MathConstants<double>::pi;

    // ═══ T12: P0-F latency gate ×4 経路（impulse argmax ∈ [floor(D), floor(D)+1]） ═══
    {
        struct LatConfig { const char* name; bool single; int tapsSingle; int ratio; double D; CustomInputOversampler::Preset preset; };
        static constexpr LatConfig lats[] =
        {
            { "S1(31/90) N=1",     true,  31, 2, 15.0,   CustomInputOversampler::Preset::IIRLike },
            { "511/140 single",    true, 511, 2, 255.0,  CustomInputOversampler::Preset::IIRLike },
            { "IIR3(511/127/31)",  false,  0, 8, 290.25, CustomInputOversampler::Preset::IIRLike },
            { "LP3(1023/255/63)",  false,  0, 8, 582.25, CustomInputOversampler::Preset::LinearPhase },
        };
        int passCnt = 0;
        std::string detail;
        for (const auto& lc : lats)
        {
            CustomInputOversampler prod;
            Shadow base{ false };
            Shadow cand{ true };
            if (lc.single)
            {
                prod.prepareSingleStage(lc.tapsSingle, 90.0, kBlockSize);
                base.prepareSingleStageModel(lc.tapsSingle, 90.0, kBlockSize);
                cand.prepareSingleStageModel(lc.tapsSingle, 90.0, kBlockSize);
            }
            else
            {
                prod.prepare(kBlockSize, lc.ratio, lc.preset);
                base.prepare(kBlockSize, lc.ratio, toShadowPreset(lc.preset));
                cand.prepare(kBlockSize, lc.ratio, toShadowPreset(lc.preset));
            }
            prod.reset(); base.reset(); cand.reset();

            for (int ch = 0; ch < 2; ++ch)
                std::fill(inA[ch].begin(), inA[ch].end(), 0.0);
            inA[0][w] = 1.0;
            inA[1][w] = 1.0;

            const auto rb = compareRoundTrip(prod, base, inP, kBlockSize, bUpP, bDnP, cUpP, cDnP);
            cand.reset();
            runShadowRoundTrip(cand, inP, kBlockSize, cUpP, cDnP, lc.ratio);

            int pkB = -1, pkC = -1;
            double vB = 0.0, vC = 0.0;
            for (int i = w; i < kBlockSize; ++i)
            {
                if (std::fabs(bDn[0][i]) > vB) { vB = std::fabs(bDn[0][i]); pkB = i; }
                if (std::fabs(cDn[0][i]) > vC) { vC = std::fabs(cDn[0][i]); pkC = i; }
            }
            const int offB = (pkB >= 0) ? pkB - w : -1;
            const int offC = (pkC >= 0) ? pkC - w : -1;
            const int lo = static_cast<int>(std::floor(lc.D));
            const bool ok = rb.first && offB >= lo && offB <= lo + 1
                         && offC >= lo && offC <= lo + 1;   // base==cand も gate
            if (!ok) detail = fmt("%s offB=%d offC=%d D=%.2f bitwise=%d",
                                  lc.name, offB, offC, lc.D, rb.first ? 1 : 0);
            else ++passCnt;
        }
        report("P0-F latency gate ×4 経路 (argmax ∈ [floor(D), floor(D)+1]・base==cand・bitwise)",
               passCnt == 4, detail);
    }

    // ═══ T13: 0-1 G-BL baseline DC / 0-2 P0-A candidate DC ×6 config（±1e-6 gate） ═══
    for (const auto& c : kConfigs)
    {
        CustomInputOversampler prod;
        Shadow base{ false };
        Shadow cand{ true };
        prod.prepare(kBlockSize, c.ratio, c.preset);
        base.prepare(kBlockSize, c.ratio, toShadowPreset(c.preset));
        cand.prepare(kBlockSize, c.ratio, toShadowPreset(c.preset));
        prod.reset(); base.reset(); cand.reset();
        for (int ch = 0; ch < 2; ++ch)
            std::fill(inA[ch].begin(), inA[ch].end(), 1.0);
        const auto r = compareRoundTrip(prod, base, inP, kBlockSize, bUpP, bDnP, cUpP, cDnP);
        cand.reset();
        runShadowRoundTrip(cand, inP, kBlockSize, cUpP, cDnP, c.ratio);
        const int stages = (c.ratio == 8) ? 3 : ((c.ratio == 4) ? 2 : 1);
        double expect = 1.0;
        for (int i = 0; i < stages; ++i) expect *= 0.75;
        const double dcBase = steadyMean(bDn[0].data(), bDn[0].size());
        const double dcCand = steadyMean(cDn[0].data(), cDn[0].size());
        const bool ok = r.first
                     && std::fabs(dcBase - expect) <= 1.0e-6
                     && std::fabs(dcCand - 1.0) <= 1.0e-6;
        report(fmt("0-1 G-BL + 0-2 P0-A DC %s (±1e-6)", c.name).c_str(), ok,
               fmt("dcBase=%.12f expect=%.12f dcCand=%.12f", dcBase, expect, dcCand));
    }

    // ═══ T14: P0-B unity (cand) + P0-C ripple + P0-C' differential ×4 経路 ═══
    {
        struct RouteConfig { const char* name; bool single; int tapsSingle; int ratio; double N; CustomInputOversampler::Preset preset; double devLimitF; };
        static constexpr RouteConfig routes[] =
        {
            { "S1(31/90) N=1",  true,  31, 2, 1, CustomInputOversampler::Preset::IIRLike,      0.30 },
            { "511/140 single", true, 511, 2, 1, CustomInputOversampler::Preset::IIRLike,      0.30 },
            { "IIR3 N=3",       false,   0, 8, 3, CustomInputOversampler::Preset::IIRLike,      0.45 },
            { "LP3 N=3",        false,   0, 8, 3, CustomInputOversampler::Preset::LinearPhase,  0.45 },
        };
        for (const auto& rc : routes)
        {
            CustomInputOversampler prod;
            Shadow cand{ true };
            if (rc.single)
            {
                prod.prepareSingleStage(rc.tapsSingle, 90.0, kBlockSize);
                cand.prepareSingleStageModel(rc.tapsSingle, 90.0, kBlockSize);
            }
            else
            {
                prod.prepare(kBlockSize, rc.ratio, rc.preset);
                cand.prepare(kBlockSize, rc.ratio, toShadowPreset(rc.preset));
            }
            prod.reset(); cand.reset();

            for (int ch = 0; ch < 2; ++ch)
                std::fill(inA[ch].begin(), inA[ch].end(), 0.0);
            inA[0][w] = 1.0;
            inA[1][w] = 1.0;

            runProdRoundTrip(prod, inP, kBlockSize, bUpP, bDnP, rc.ratio);
            runShadowRoundTrip(cand, inP, kBlockSize, cUpP, cDnP, rc.ratio);

            std::vector<double> hB(bDn[0].begin() + w, bDn[0].begin() + w + kProbeWindow);
            std::vector<double> hC(cDn[0].begin() + w, cDn[0].begin() + w + kProbeWindow);

            const double u50 = dtftDb(hC, 50.0 / 192000.0);
            const double u1k = dtftDb(hC, 1000.0 / 192000.0);
            double rMax = -1.0e30, rMin = 1.0e30;
            for (double f = 0.005; f <= 0.30 + 1.0e-12; f += 0.001)
            {
                const double db = dtftDb(hC, f);
                rMax = std::max(rMax, db);
                rMin = std::min(rMin, db);
            }
            const double ripple = rMax - rMin;

            const double devTerm = 20.0 * rc.N * std::log10(4.0 / 3.0);
            static constexpr double devPts[] = { 50.0, 1000.0, 10000.0, 48000.0, 86400.0, 94080.0 };
            double maxDev = 0.0;
            for (double fq : devPts)
            {
                const double fn = fq / 192000.0;
                if (fn > rc.devLimitF + 1.0e-12) continue;
                const double dev = dtftDb(hC, fn) - dtftDb(hB, fn) - devTerm;
                maxDev = std::max(maxDev, std::fabs(dev));
            }
            const bool ok = std::fabs(u50) <= 0.1 && std::fabs(u1k) <= 0.1
                         && ripple <= 0.05
                         && maxDev <= 0.05;
            report(fmt("P0-B/C/C' %s (unity±0.1 / ripple≤0.05 / diff≤0.05)", rc.name).c_str(),
                   ok, fmt("u50=%.4f u1k=%.4f ripple=%.4f maxDev=%.4f", u50, u1k, ripple, maxDev));
        }
    }

    // ═══ T15: P0-D per-design stopband floor ×6 design ═══
    {
        struct Dsgn { int taps; double atten; };
        static constexpr Dsgn ds[] = { {511,140},{127,110},{31,90},{1023,160},{255,140},{63,120} };
        for (const auto& d : ds)
        {
            convo::polyphase_ref::StageModel st;
            convo::polyphase_ref::prepareStageModel(st, d.taps, d.atten, kBlockSize);
            const auto raw = rawCoeffsFromStage(st);
            const double target = -(d.atten - 3.0);
            const double floorDb = -(d.atten - 10.0);

            double tEnd = 0.0;
            bool found = false;
            for (double f = 0.0005; f <= 0.5 + 1.0e-12; f += 0.0005)
            {
                if (dtftDb(raw, f) <= target) { tEnd = f; found = true; break; }
            }
            bool ok = found;
            double stopMax = -1.0e30, stopMin = 1.0e30;
            double edge = 0.0;
            if (found)
            {
                double lo = tEnd - 0.0005, hi = tEnd;
                for (int it = 0; it < 40 && (hi - lo) > 1.0e-6; ++it)
                {
                    const double mid = 0.5 * (lo + hi);
                    if (dtftDb(raw, mid) <= target) hi = mid;
                    else lo = mid;
                }
                tEnd = hi;
                for (double f = tEnd; f <= 0.5 + 1.0e-12; f += 0.0005)
                {
                    const double db = dtftDb(raw, f);
                    stopMax = std::max(stopMax, db);
                    stopMin = std::min(stopMin, db);
                }
                ok = stopMax <= floorDb;
            }
            // −0.1 dB edge（record）
            bool dropped = false;
            for (double f = 0.0005; f <= 0.4999 + 1.0e-12 && !dropped; f += 0.0001)
            {
                if (dtftDb(raw, f) < -0.1) dropped = true;
                else edge = f;
            }
            report(fmt("P0-D stopband floor %d/%.0f (≤%.0f dB @[t_end,0.5])",
                       d.taps, d.atten, floorDb).c_str(), ok,
                   fmt("t_end=%.5f stopMax=%.1f stopMin=%.1f edge=%.4f",
                       tEnd, stopMax, stopMin, edge));
        }
    }

    // ═══ T16: P0-E E-1 base D1 + E-1c cand D1（単段 ×3 design × f̂）+ E-2 D2 ═══
    {
        static constexpr double fhats[] = { 0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45 };
        bool e1Ok = true, e1cOk = true;
        double minMarginC = 1.0e30;
        std::string detail;
        for (const int dTaps : { 31, 127, 511 })
        {
            const double dAtten = (dTaps == 31) ? 90.0 : ((dTaps == 127) ? 110.0 : 140.0);
            CustomInputOversampler prod;
            Shadow cand{ true };
            prod.prepareSingleStage(dTaps, dAtten, kBlockSize);
            cand.prepareSingleStageModel(dTaps, dAtten, kBlockSize);

            for (double fh : fhats)
            {
                for (int ch = 0; ch < 2; ++ch)
                    for (int i = 0; i < kBlockSize; ++i)
                        inA[ch][i] = std::sin(2.0 * kPi * fh * static_cast<double>(i));
                prod.reset(); cand.reset();

                {
                    juce::dsp::AudioBlock<double> inBlock(inP, 2, static_cast<std::size_t>(kBlockSize));
                    auto upB = prod.processUp(inBlock, 2);
                    const std::size_t nUp = upB.getNumSamples();
                    for (int ch = 0; ch < 2; ++ch)
                        std::memcpy(bUp[ch].data(), upB.getChannelPointer(ch), nUp * sizeof(double));
                }
                cand.reset();
                const int nUpC = cand.processUp(inP, kBlockSize, cUpP);
                if (nUpC != 2 * kBlockSize) { e1cOk = false; detail = "cand up failed"; break; }

                // D1（up レート・expected 周波数主報告・freq: tone = f̂/2, image = (1−f̂)/2）
                const double tB = dftMag(bUp[0].data(), 2 * kBlockSize, fh / 2.0);
                const double iB = dftMag(bUp[0].data(), 2 * kBlockSize, (1.0 - fh) / 2.0);
                const double d1B = 20.0 * std::log10(iB / (tB + 1.0e-300));
                const double tC = dftMag(cUp[0].data(), 2 * kBlockSize, fh / 2.0);
                const double iC = dftMag(cUp[0].data(), 2 * kBlockSize, (1.0 - fh) / 2.0);
                const double d1C = 20.0 * std::log10(iC / (tC + 1.0e-300));

                if (fh <= 0.30 + 1.0e-12
                    && std::fabs(d1B - (-9.542425)) > 0.01)
                {
                    e1Ok = false;
                    detail = fmt("E-1 %d/%.0f f̂=%.2f d1B=%.6f", dTaps, dAtten, fh, d1B);
                }
                if (fh <= 0.35 + 1.0e-12)
                {
                    if (d1C > -9.442)
                    {
                        e1cOk = false;
                        detail = fmt("E-1c %d/%.0f f̂=%.2f d1C=%.3f", dTaps, dAtten, fh, d1C);
                    }
                    minMarginC = std::min(minMarginC, -9.442 - d1C);
                }
            }
        }
        report("P0-E E-1 base D1 −9.542425 ±0.01 @f̂≤0.30 (prod 31/127/511 ×4 f̂)", e1Ok, detail);
        report("P0-E E-1c cand D1 ≤ −9.442 @f̂≤0.35 (shadow 31/127/511 ×5 f̂)", e1cOk,
               fmt("min margin=%.1f dB", minMarginC));
        std::printf("  [record] E-1b cand D1 は rect / Hann+trim8 併記（手法感度帯・gate 不使用）\n");

        // E-2: D2 単段 round-trip tone（base==cand ≤0.01 dB）
        {
            bool d2Ok = true;
            std::string d2Detail;
            double d2Rec[3] = { 0.0, 0.0, 0.0 };
            static constexpr double d2Fhats[] = { 0.05, 0.10, 0.20 };
            for (int si = 0; si < 3; ++si)
            {
                const double fh = d2Fhats[si];
                CustomInputOversampler prod;
                Shadow cand{ true };
                prod.prepareSingleStage(31, 90.0, kBlockSize);
                cand.prepareSingleStageModel(31, 90.0, kBlockSize);
                for (int ch = 0; ch < 2; ++ch)
                    for (int i = 0; i < kBlockSize; ++i)
                        inA[ch][i] = std::sin(2.0 * kPi * fh * static_cast<double>(i));
                prod.reset(); cand.reset();

                runProdRoundTrip(prod, inP, kBlockSize, bUpP, bDnP, 2);
                runShadowRoundTrip(cand, inP, kBlockSize, cUpP, cDnP, 2);

                const int trim = 31 + 256;
                const int m = kBlockSize - trim;
                std::vector<double> wb(static_cast<std::size_t>(m)), wc(static_cast<std::size_t>(m));
                for (int n = 0; n < m; ++n)
                {
                    const double hann = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(n)
                                                             / static_cast<double>(m - 1)));
                    wb[n] = bDn[0][trim + n] * hann;
                    wc[n] = cDn[0][trim + n] * hann;
                }
                // D2（output rate・f̂ vs 0.5−f̂・Hann+trim）
                const double d2B = 20.0 * std::log10((dftMag(wb.data(), wb.size(), 0.5 - fh) + 1.0e-300)
                                                   / (dftMag(wb.data(), wb.size(), fh) + 1.0e-300));
                const double d2C = 20.0 * std::log10((dftMag(wc.data(), wc.size(), 0.5 - fh) + 1.0e-300)
                                                   / (dftMag(wc.data(), wc.size(), fh) + 1.0e-300));
                d2Rec[si] = d2B;
                if (std::fabs(d2B - d2C) > 0.01)
                {
                    d2Ok = false;
                    d2Detail = fmt("f̂=%.2f d2B=%.3f d2C=%.3f", fh, d2B, d2C);
                }
            }
            report("P0-E E-2 D2 base==cand ≤0.01 dB @f̂ {0.05,0.10,0.20} (S1)", d2Ok, d2Detail);
            std::printf("  [record] D2 base = %.2f / %.2f / %.2f dB (f̂ 0.05/0.10/0.20)\n",
                        d2Rec[0], d2Rec[1], d2Rec[2]);
        }
    }

    // ═══ T17: P0-I SoftClip local OS per-phase attribution（R17-1 完全契約） ═══
    {
        CustomInputOversampler prod;
        Shadow base{ false };
        Shadow cand{ true };
        prod.prepareSingleStage(31, 90.0, kBlockSize);
        base.prepareSingleStageModel(31, 90.0, kBlockSize);
        cand.prepareSingleStageModel(31, 90.0, kBlockSize);
        convo::polyphase_ref::StageModel s1;
        convo::polyphase_ref::prepareStageModel(s1, 31, 90.0, kBlockSize);
        const int cPar = s1.centerParity;   // =1（S1 31/90）
        const int vPar = s1.convParity;     // =0

        // I-a: OS 単独 round-trip DC（SoftClip 無効・A=0.5）±1e-6
        {
            for (int ch = 0; ch < 2; ++ch)
                std::fill(inA[ch].begin(), inA[ch].end(), 0.5);
            prod.reset(); base.reset(); cand.reset();
            const auto r = compareRoundTrip(prod, base, inP, kBlockSize, bUpP, bDnP, cUpP, cDnP);
            cand.reset();
            runShadowRoundTrip(cand, inP, kBlockSize, cUpP, cDnP, 2);
            const double dcB = steadyMean(bDn[0].data(), bDn[0].size());
            const double dcC = steadyMean(cDn[0].data(), cDn[0].size());
            // A=0.5 入力に対する利得比で gate（base 0.75 / cand 1.0）
            const bool ok = r.first
                         && std::fabs(dcB / 0.5 - 0.75) <= 1.0e-6
                         && std::fabs(dcC / 0.5 - 1.0) <= 1.0e-6;
            report("P0-I I-a OS-only DC (SoftClip 無効・A=0.5): base 0.75 / cand 1.0 ±1e-6", ok,
                   fmt("dcB=%.12f dcC=%.12f", dcB, dcC));
        }

        // 刺激集合（R17-1 動作点固定）
        struct Stim { const char* name; int mode; double fhat; double amp; };
        std::vector<Stim> stims;
        for (double a : { 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0 }) stims.push_back({ "DC", 0, 0.0, a });
        for (double a : { 0.5, 1.0, 2.0, 4.0, 8.0 }) stims.push_back({ "sine f̂=0.2", 1, 0.20, a });
        for (double a : { 1.0, 2.0, 4.0, 8.0 }) stims.push_back({ "PRNG(seed20260921)", 2, 0.0, a });

        bool safetyOk = true, c1Ok = true, c2Ok = true, c3Ok = true, c4Ok = true;
        std::size_t evBTotal = 0, evCTotal = 0;
        double maxAbsY = 0.0;
        std::size_t clampCnt = 0;      // |y| ≥ 1 − 1e-12（この動作点では θ+κ=0.9 → 0 件が正当）
        std::size_t tanhClampCnt = 0;  // |arg| ≥ 4.5（E3・有意な clamp 記録）
        std::string detail;

        for (std::size_t si = 0; si < stims.size() && detail.empty(); ++si)
        {
            const auto& stm = stims[si];
            for (int ch = 0; ch < 2; ++ch)
            {
                for (int i = 0; i < kBlockSize; ++i)
                {
                    if (stm.mode == 0) inA[ch][i] = stm.amp;
                    else if (stm.mode == 1) inA[ch][i] = stm.amp * std::sin(2.0 * kPi * stm.fhat * static_cast<double>(i));
                    else inA[ch][i] = 0.0;
                }
            }
            if (stm.mode == 2)
            {
                DeterministicPRNG rng(20260921ULL);
                for (int ch = 0; ch < 2; ++ch)
                    for (int i = 0; i < kBlockSize; ++i)
                        inA[ch][i] = stm.amp * rng.next();
            }

            prod.reset(); cand.reset();
            {
                juce::dsp::AudioBlock<double> inBlock(inP, 2, static_cast<std::size_t>(kBlockSize));
                auto upB = prod.processUp(inBlock, 2);
                const std::size_t nUp = upB.getNumSamples();
                for (int ch = 0; ch < 2; ++ch)
                    std::memcpy(bUp[ch].data(), upB.getChannelPointer(ch), nUp * sizeof(double));
            }
            cand.reset();
            const int nUpC = cand.processUp(inP, kBlockSize, cUpP);
            if (nUpC != 2 * kBlockSize) { safetyOk = false; detail = "cand up failed"; break; }

            std::vector<double> yb(2 * kBlockSize), yc(2 * kBlockSize);
            for (int i = 0; i < 2 * kBlockSize; ++i)
            {
                yb[i] = musicalSoftClipRef(bUp[0][i], kP0iOp);
                yc[i] = musicalSoftClipRef(cUp[0][i], kP0iOp);
            }

            // per-phase 走査（S1: conv=偶数位 / center=奇数位）
            for (int i = 0; i < kBlockSize && detail.empty(); ++i)
            {
                const int pc = 2 * i + cPar;
                const int pv = 2 * i + vPar;

                // I-0: conv 位相 bitwise 一致 / center 位相 bitwise 2×（denorm 帯例外）
                if (std::bit_cast<std::uint64_t>(cUp[0][pv]) != std::bit_cast<std::uint64_t>(bUp[0][pv]))
                { c1Ok = false; detail = fmt("I-0 conv mismatch @%d", pv); break; }

                const double xbc = bUp[0][pc];
                const double xcc = cUp[0][pc];
                if (std::fabs(xbc) >= 2.0e-20
                    && std::bit_cast<std::uint64_t>(xcc) != std::bit_cast<std::uint64_t>(2.0 * xbc))
                { c1Ok = false; detail = fmt("I-0 center not 2x @%d", pc); break; }

                const double e1B = std::fabs(xbc) >= (kP0iOp.threshold - kP0iOp.knee);
                const double e1C = std::fabs(xcc) >= (kP0iOp.threshold - kP0iOp.knee);
                // (c2): base ⊆ candidate（単調性）
                if (e1B && !e1C) { c2Ok = false; detail = fmt("c2 subset violated @%d", pc); break; }
                // (c3): candidate のみの event は envelope |x_b| ≥ θ_evt/2
                if (e1C && !e1B && std::fabs(xbc) < 0.5 * (kP0iOp.threshold - kP0iOp.knee))
                { c3Ok = false; detail = fmt("c3 envelope violated @%d (|xb|=%.3g)", pc, xbc); break; }
                // (c4): candidate center 出力 = F(2·x_b)（点wise 参照と bitwise 一致）
                if (std::bit_cast<std::uint64_t>(yc[pc])
                    != std::bit_cast<std::uint64_t>(musicalSoftClipRef(2.0 * xbc, kP0iOp)))
                { c4Ok = false; detail = fmt("c4 pointwise mismatch @%d", pc); break; }

                // (c1) conv 位相出力 bitwise 一致
                if (std::bit_cast<std::uint64_t>(yc[pv]) != std::bit_cast<std::uint64_t>(yb[pv]))
                { c1Ok = false; detail = fmt("c1 conv output mismatch @%d", pv); break; }

                // event 件数・指標（E1・入力側）
                if (e1B) ++evBTotal;
                if (e1C) ++evCTotal;
                // clamp 記録
                if (std::fabs(yb[i]) >= 1.0 - 1.0e-12) ++clampCnt;
                if (std::fabs(yc[i]) >= 1.0 - 1.0e-12) ++clampCnt;
                const double argB = (std::fabs(bUp[0][pc]) - kP0iOp.threshold) / kP0iOp.knee;
                if (std::fabs(argB) >= 4.5) ++tanhClampCnt;
                maxAbsY = std::max(maxAbsY, std::max(std::fabs(yb[i]), std::fabs(yc[i])));

                // I-b: NaN/Inf 0・|y| ≤ 1.0
                if (!std::isfinite(yb[i]) || !std::isfinite(yc[i]))
                { safetyOk = false; detail = fmt("NaN/Inf @%d", i); break; }
                if (std::fabs(yb[i]) > 1.0 || std::fabs(yc[i]) > 1.0)
                { safetyOk = false; detail = fmt("|y|>1.0 @%d", i); break; }
            }
            if (!detail.empty()) break;   // 1 件でも gate 違反があれば刺激ループを抜ける
        }

        const bool ok = safetyOk && c1Ok && c2Ok && c3Ok && c4Ok;
        report("P0-I I-b/c safety + per-phase attribution (15 stimuli)", ok, detail);
        std::printf("  [record] events: base=%zu cand=%zu ΔS=%lld R_s=%.3f | max|y|=%.6f clamp(|y|≥1−1e-12)=%zu tanhClamp(|arg|≥4.5)=%zu\n",
                    evBTotal, evCTotal, (long long)((std::ptrdiff_t)evCTotal - (std::ptrdiff_t)evBTotal),
                    (evBTotal > 0) ? static_cast<double>(evCTotal) / static_cast<double>(evBTotal) : 1.0,
                    maxAbsY, clampCnt, tanhClampCnt);
    }

    // ═══ T18: P0-G float/double equivalence ×2 pairs ═══
    {
        // pair b (SoftClip): F(x) と F(float→double 量子化 x) の差
        // pair a (OS round-trip): prod RT(x) vs prod RT(float 量子化 x)
        bool ok = true;
        double maxAbsB = 0.0, maxAbsA = 0.0, rmsB = 0.0, rmsA = 0.0, maxRel = 0.0;
        std::string detail;

        // pair b（SoftClip）
        {
            DeterministicPRNG rng(20260921ULL);
            for (int i = 0; i < 2 * kBlockSize; ++i)
            {
                const double x = rng.next();
                const double xf = static_cast<double>(static_cast<float>(x));
                const double yd = musicalSoftClipRef(x, kP0iOp);
                const double yq = musicalSoftClipRef(xf, kP0iOp);
                const double e = std::fabs(yd - yq);
                maxAbsB = std::max(maxAbsB, e);
                rmsB += (yd - yq) * (yd - yq);
                if (std::fabs(yd) > 1.0e-6 && e > 0.0)
                    maxRel = std::max(maxRel, e / std::fabs(yd));
            }
            rmsB = std::sqrt(rmsB / static_cast<double>(2 * kBlockSize));
            if (!(maxAbsB <= 5.0e-7 && rmsB <= 5.0e-8)) ok = false;
        }

        // pair a（OS round-trip・float 量子化入力）
        {
            CustomInputOversampler prod;
            prod.prepareSingleStage(31, 90.0, kBlockSize);
            DeterministicPRNG rng(20260921ULL);
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < kBlockSize; ++i)
                    inA[ch][i] = rng.next();
            prod.reset();
            runProdRoundTrip(prod, inP, kBlockSize, bUpP, bDnP, 2);
            std::vector<double> ref(kBlockSize);
            for (int i = 0; i < kBlockSize; ++i) ref[i] = bDn[0][i];

            // float 量子化入力で再実行
            prod.reset();
            DeterministicPRNG rng2(20260921ULL);
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < kBlockSize; ++i)
                    inA[ch][i] = static_cast<double>(static_cast<float>(rng2.next()));
            prod.reset();
            runProdRoundTrip(prod, inP, kBlockSize, bUpP, bDnP, 2);

            double sumSq = 0.0;
            for (int i = 0; i < kBlockSize; ++i)
            {
                const double e = std::fabs(ref[i] - bDn[0][i]);
                maxAbsA = std::max(maxAbsA, e);
                sumSq += e * e;
                if (std::fabs(ref[i]) > 1.0e-6 && e > 0.0)
                    maxRel = std::max(maxRel, e / std::fabs(ref[i]));
            }
            rmsA = std::sqrt(sumSq / static_cast<double>(kBlockSize));
            ok = ok && maxAbsA <= 5.0e-7 && rmsA <= 5.0e-8;
        }

        ok = ok && maxAbsB <= 5.0e-7 && rmsB <= 5.0e-8;
        report("P0-G float/double equivalence ×2 pairs (maxAbsErr≤5e-7 / RMS≤5e-8 / 相対誤差 record)", ok,
               fmt("RT maxAbs=%.2e rms=%.2e | SC maxAbs=%.2e rms=%.2e | maxRel=%.2e",
                   maxAbsA, rmsA, maxAbsB, rmsB, maxRel));
    }
}


int runAllTests()
{
    emitBuildIdBlock();

    // ── 定数パス契約（R16-1） ──
    {
        const bool piOk = std::bit_cast<std::uint64_t>(convo::polyphase_ref::kPiRef)
                       == std::bit_cast<std::uint64_t>(juce::MathConstants<double>::pi);
        report("R16-1 kPiRef == juce::MathConstants<double>::pi (bitwise)", piOk);
    }
    {
        const bool tOk = std::bit_cast<std::uint64_t>(convo::polyphase_ref::kDenormThresholdRef)
                      == std::bit_cast<std::uint64_t>(convo::numeric_policy::kDenormThresholdAudioState);
        report("R16-1 kDenormThresholdRef == numeric_policy::kDenormThresholdAudioState (bitwise)", tOk);
    }

    // ── R16-3 candidate マクロ 2 系統一致性 assert ──
    {
        Shadow defaultShadow{};
        Shadow explicitBase{ false };
        Shadow explicitCand{ true };
        defaultShadow.prepare(kBlockSize, 8, Shadow::Preset::IIRLike);
        explicitBase.prepare(kBlockSize, 8, Shadow::Preset::IIRLike);
        explicitCand.prepare(kBlockSize, 8, Shadow::Preset::IIRLike);

        double baseDc = 1.0;
        for (int i = 0; i < 3; ++i) baseDc *= 0.75;

        const bool ok =
            std::bit_cast<std::uint64_t>(defaultShadow.centerPhaseGain())
         == std::bit_cast<std::uint64_t>(explicitBase.centerPhaseGain())
         && std::bit_cast<std::uint64_t>(defaultShadow.expectedDcRoundTrip())
         == std::bit_cast<std::uint64_t>(explicitBase.expectedDcRoundTrip())
         && std::bit_cast<std::uint64_t>(defaultShadow.expectedDcRoundTrip())
         == std::bit_cast<std::uint64_t>(baseDc)
         && std::bit_cast<std::uint64_t>(explicitCand.expectedDcRoundTrip())
         == std::bit_cast<std::uint64_t>(1.0)
         && std::bit_cast<std::uint64_t>(explicitCand.centerPhaseGain())
         == std::bit_cast<std::uint64_t>(2.0);
        report("R16-3 candidate macro 2 系統一致性 (default macro ↔ runtime {false,true})", ok);
    }

    // ── R17-4 (A) coefficient invariant ──
    testCoefficientInvariants();

    // ── 共通バッファ（up は最大 8 倍レートを収容: kBlockSize·8・R17-4 (E) chunk 検証を含む） ──
    std::vector<double> inBuf[2];
    std::vector<double> prodUp[2], prodDn[2];
    std::vector<double> shadowUp[2], shadowDn[2];
    for (int ch = 0; ch < 2; ++ch)
    {
        inBuf[ch].assign(static_cast<std::size_t>(kBlockSize), 0.0);
        prodUp[ch].assign(static_cast<std::size_t>(8 * kBlockSize), 0.0);
        prodDn[ch].assign(static_cast<std::size_t>(kBlockSize), 0.0);
        shadowUp[ch].assign(static_cast<std::size_t>(8 * kBlockSize), 0.0);
        shadowDn[ch].assign(static_cast<std::size_t>(kBlockSize), 0.0);
    }
    double* inPtrs[2] = { inBuf[0].data(), inBuf[1].data() };
    double* prodUpP[2] = { prodUp[0].data(), prodUp[1].data() };
    double* prodDnP[2] = { prodDn[0].data(), prodDn[1].data() };
    double* shadowUpP[2] = { shadowUp[0].data(), shadowUp[1].data() };
    double* shadowDnP[2] = { shadowDn[0].data(), shadowDn[1].data() };

    // ── R17-4 (B) zero-input ──
    for (const auto& c : kConfigs)
    {
        CustomInputOversampler prod;
        Shadow shadow{ false };
        prod.prepare(kBlockSize, c.ratio, c.preset);
        shadow.prepare(kBlockSize, c.ratio, toShadowPreset(c.preset));
        prod.reset(); shadow.reset();
        for (int ch = 0; ch < 2; ++ch)
            std::fill(inBuf[ch].begin(), inBuf[ch].end(), 0.0);

        const auto r = compareRoundTrip(prod, shadow, inPtrs, kBlockSize,
                                        prodUpP, prodDnP, shadowUpP, shadowDnP);
        bool ok = r.first;
        for (int ch = 0; ch < 2 && ok; ++ch)
            for (std::size_t i = 0; i < prodDn[ch].size(); ++i)
                if (std::bit_cast<std::uint64_t>(prodDn[ch][i])
                    != std::bit_cast<std::uint64_t>(0.0))
                { ok = false; break; }
        report(fmt("R17-4(B) zero-input %s", c.name).c_str(), ok, r.second);
    }

    // ── R17-4 (C) constant-input DC ──
    for (const auto& c : kConfigs)
    {
        CustomInputOversampler prod;
        Shadow base{ false };
        Shadow cand{ true };
        prod.prepare(kBlockSize, c.ratio, c.preset);
        base.prepare(kBlockSize, c.ratio, toShadowPreset(c.preset));
        cand.prepare(kBlockSize, c.ratio, toShadowPreset(c.preset));
        prod.reset(); base.reset(); cand.reset();

        for (int ch = 0; ch < 2; ++ch)
            std::fill(inBuf[ch].begin(), inBuf[ch].end(), 1.0);

        const auto r = compareRoundTrip(prod, base, inPtrs, kBlockSize,
                                        prodUpP, prodDnP, shadowUpP, shadowDnP);
        const int stages = (c.ratio == 8) ? 3 : ((c.ratio == 4) ? 2 : 1);
        double expect = 1.0;
        for (int i = 0; i < stages; ++i) expect *= 0.75;
        const double dcBase = steadyMean(prodDn[0].data(), prodDn[0].size());

        // candidate は予測系（R12-8）: production を触らず shadow cand のみ再実行
        runShadowRoundTrip(cand, inPtrs, kBlockSize, shadowUpP, shadowDnP, c.ratio);
        const double dcCand = steadyMean(shadowDn[0].data(), shadowDn[0].size());

        const bool ok = r.first
                     && std::fabs(dcBase - expect) <= kDcTol
                     && std::fabs(dcCand - 1.0) <= kDcTol;
        report(fmt("R17-4(C) constant-input DC %s", c.name).c_str(), ok,
               fmt("dcBase=%.12f expect=%.12f dcCand=%.12f", dcBase, expect, dcCand));
    }

    // ── R17-4 (D) impulse（warm-start 規約・R17-7 で数値確定） ──
    {
        // 単段 S1 (31/90): h_rt = 2·(conv⋆conv) + 0.25·δ[15]
        CustomInputOversampler prod;
        Shadow base{ false };
        Shadow cand{ true };
        prod.prepareSingleStage(31, 90.0, kBlockSize);
        base.prepareSingleStageModel(31, 90.0, kBlockSize);
        cand.prepareSingleStageModel(31, 90.0, kBlockSize);
        prod.reset(); base.reset(); cand.reset();

        for (int ch = 0; ch < 2; ++ch)
            std::fill(inBuf[ch].begin(), inBuf[ch].end(), 0.0);
        inBuf[0][kBlockSize / 2] = 1.0;
        inBuf[1][kBlockSize / 2] = 1.0;

        const auto r = compareRoundTrip(prod, base, inPtrs, kBlockSize,
                                        prodUpP, prodDnP, shadowUpP, shadowDnP);
        const int w = kBlockSize / 2;
        double sumBase = 0.0;
        int peakIdx = -1;
        double peakVal = 0.0;
        for (int i = w; i < kBlockSize; ++i)
        {
            sumBase += prodDn[0][i];
            if (std::fabs(prodDn[0][i]) > peakVal)
            {
                peakVal = std::fabs(prodDn[0][i]);
                peakIdx = i;
            }
        }

        // candidate（shadow cand のみ再実行）
        cand.reset();
        runShadowRoundTrip(cand, inPtrs, kBlockSize, shadowUpP, shadowDnP, 2);
        double sumCand = 0.0;
        for (int i = w; i < kBlockSize; ++i)
            sumCand += shadowDn[0][i];

        // R5-9 構造恒等式の閉形式: h_rt = 2·(conv⋆conv) + 0.25·δ[centerTap]
        //   centerTap = 15 の位置には conv⋆conv の寄与も載るため、期待値は
        //   shadow の独立係数（StageModel）から構築する（0.25 は δ 項のみの値）。
        convo::polyphase_ref::StageModel s1;
        convo::polyphase_ref::prepareStageModel(s1, 31, 90.0, kBlockSize);
        double cc15 = 0.0;
        for (std::size_t r = 0; r < s1.convCoeffs.size(); ++r)
        {
            const std::ptrdiff_t k = static_cast<std::ptrdiff_t>(s1.centerTap) - static_cast<std::ptrdiff_t>(r);
            if (k >= 0 && k < static_cast<std::ptrdiff_t>(s1.convCoeffs.size()))
                cc15 += s1.convCoeffs[r] * s1.convCoeffs[static_cast<std::size_t>(k)];
        }
        const double expectPeak = 2.0 * cc15 + 0.25;

        const bool ok = r.first
                     && peakIdx == w + 15
                     && std::fabs(prodDn[0][w + 15] - expectPeak) <= kDcTol
                     && std::fabs(sumBase - 0.75) <= kDcTol
                     && std::fabs(sumCand - 1.0) <= kDcTol;
        report("R17-4(D) impulse warm-start S1(31/90): h_rt=2·(conv⋆conv)+0.25δ[15] / Σ≈0.75 / cand Σ≈1.0 / bitwise",
               ok, fmt("peak@%d expectPeak=%.12f sumBase=%.12f sumCand=%.12f | %s",
                       peakIdx - w, expectPeak, sumBase, sumCand, r.second.c_str()));
    }
    {
        // IIR3 r=8: Σh_rt = 0.75³ / peak ∈ [floor(290.25), floor+1]
        CustomInputOversampler prod;
        Shadow base{ false };
        prod.prepare(kBlockSize, 8, CustomInputOversampler::Preset::IIRLike);
        base.prepare(kBlockSize, 8, Shadow::Preset::IIRLike);
        prod.reset(); base.reset();

        for (int ch = 0; ch < 2; ++ch)
            std::fill(inBuf[ch].begin(), inBuf[ch].end(), 0.0);
        inBuf[0][kBlockSize / 2] = 1.0;
        inBuf[1][kBlockSize / 2] = 1.0;

        const auto r = compareRoundTrip(prod, base, inPtrs, kBlockSize,
                                        prodUpP, prodDnP, shadowUpP, shadowDnP);
        const int w = kBlockSize / 2;
        double sum = 0.0;
        int peakIdx = w;
        double peakVal = 0.0;
        for (int i = w; i < kBlockSize; ++i)
        {
            sum += prodDn[0][i];
            if (std::fabs(prodDn[0][i]) > peakVal)
            {
                peakVal = std::fabs(prodDn[0][i]);
                peakIdx = i;
            }
        }
        const bool ok = r.first
                     && std::fabs(sum - 0.421875) <= kDcTol
                     && peakIdx >= w + 290 && peakIdx <= w + 291;
        report("R17-4(D) impulse warm-start IIR3 r=8: Σ≈0.75³ / peak∈[290,291] / bitwise",
               ok, fmt("peak offset=%d sum=%.12f", peakIdx - w, sum));
    }

    // ── R17-4 (E) random block partition invariance（production + shadow） ──
    for (const auto& c : kConfigs)
    {
        CustomInputOversampler prod;
        Shadow shadow{ false };
        prod.prepare(kBlockSize, c.ratio, c.preset);
        shadow.prepare(kBlockSize, c.ratio, toShadowPreset(c.preset));

        // 1-shot baseline（production + shadow・同一 PRNG ストリーム）
        DeterministicPRNG rng;
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < kBlockSize; ++i)
                inBuf[ch][i] = rng.next();
        prod.reset(); shadow.reset();
        const auto base = compareRoundTrip(prod, shadow, inPtrs, kBlockSize,
                                           prodUpP, prodDnP, shadowUpP, shadowDnP);
        if (!base.first)
        {
            report(fmt("R17-4(E) partition %s (baseline)", c.name).c_str(), false, base.second);
            continue;
        }
        std::vector<double> baseUp[2], baseDn[2];
        for (int ch = 0; ch < 2; ++ch)
        {
            baseUp[ch].assign(prodUp[ch].begin(), prodUp[ch].end());
            baseDn[ch].assign(prodDn[ch].begin(), prodDn[ch].end());
        }
        std::vector<double> streamIn[2];
        for (int ch = 0; ch < 2; ++ch)
            streamIn[ch].assign(inBuf[ch].begin(), inBuf[ch].end());

        // partition 集合（R17-4 (E)）: 細分割 + mixed
        std::vector<std::vector<int>> partitions;
        {
            static constexpr int parts[] = { 1, 2, 3, 5, 7, 11, 15, 31, 63, 127, 256, 512, 1024 };
            std::vector<int> fine;
            int total = 0;
            for (int s : parts)
            {
                if (total + s <= kBlockSize)
                {
                    fine.push_back(s);
                    total += s;
                }
            }
            if (total < kBlockSize)
                fine.push_back(kBlockSize - total);
            partitions.push_back(fine);

            std::vector<int> mixed{ kBlockSize / 2, kBlockSize / 4, kBlockSize / 8, kBlockSize / 8 };
            partitions.push_back(mixed);
        }

        bool allOk = true;
        std::string detail;
        for (const auto& p : partitions)
        {
            int sum = 0;
            for (int s : p) sum += s;
            if (sum != kBlockSize || sum <= 0)
            {
                allOk = false;
                detail = "partition sum mismatch";
                break;
            }

            prod.reset(); shadow.reset();
            std::vector<double> upAccum[2], dnAccum[2];
            for (int ch = 0; ch < 2; ++ch)
            {
                upAccum[ch].assign(static_cast<std::size_t>(c.ratio * kBlockSize), 0.0);
                dnAccum[ch].assign(static_cast<std::size_t>(kBlockSize), 0.0);
            }

            int offset = 0;
            bool partOk = true;
            for (int s : p)
            {
                for (int ch = 0; ch < 2; ++ch)
                    std::memcpy(inBuf[ch].data(), streamIn[ch].data() + offset,
                                static_cast<std::size_t>(s) * sizeof(double));
                const auto r = compareRoundTrip(prod, shadow, inPtrs, s,
                                                prodUpP, prodDnP, shadowUpP, shadowDnP);
                if (!r.first)
                {
                    partOk = false;
                    detail = fmt("size=%d %s", s, r.second.c_str());
                    break;
                }
                for (int ch = 0; ch < 2; ++ch)
                {
                    std::memcpy(upAccum[ch].data() + c.ratio * offset, prodUp[ch].data(),
                                static_cast<std::size_t>(c.ratio * s) * sizeof(double));
                    std::memcpy(dnAccum[ch].data() + offset, prodDn[ch].data(),
                                static_cast<std::size_t>(s) * sizeof(double));
                }
                offset += s;
            }

            if (partOk)
            {
                for (int ch = 0; ch < 2 && partOk; ++ch)
                {
                    if (!bitwiseEqual(upAccum[ch].data(), baseUp[ch].data(),
                                      static_cast<std::size_t>(c.ratio * kBlockSize))
                     || !bitwiseEqual(dnAccum[ch].data(), baseDn[ch].data(),
                                      static_cast<std::size_t>(kBlockSize)))
                    {
                        partOk = false;
                        detail = "accum vs one-shot mismatch";
                    }
                }
            }
            if (!partOk)
            {
                allOk = false;
                break;
            }
        }
        report(fmt("R17-4(E) random block partition invariance %s (%d partitions)",
                   c.name, static_cast<int>(partitions.size())).c_str(), allOk, detail);
    }

    // ── REF-FIDELITY 本体: 3 PRNG blocks × 2 preset × ratio {2,4,8} + reset contract ──
    for (const auto& c : kConfigs)
    {
        CustomInputOversampler prod;
        Shadow shadow{ false };
        prod.prepare(kBlockSize, c.ratio, c.preset);
        shadow.prepare(kBlockSize, c.ratio, toShadowPreset(c.preset));

        DeterministicPRNG rng;
        bool allOk = true;
        std::string detail;
        for (int block = 0; block < 3; ++block)
        {
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < kBlockSize; ++i)
                    inBuf[ch][i] = rng.next();
            const auto r = compareRoundTrip(prod, shadow, inPtrs, kBlockSize,
                                            prodUpP, prodDnP, shadowUpP, shadowDnP);
            if (!r.first)
            {
                allOk = false;
                detail = fmt("block%d %s", block, r.second.c_str());
                break;
            }
        }
        if (allOk)
        {
            // reset contract（atomic 3 相当）後の同一刺激で再び bitwise 一致
            prod.reset(); shadow.reset();
            const auto r = compareRoundTrip(prod, shadow, inPtrs, kBlockSize,
                                            prodUpP, prodDnP, shadowUpP, shadowDnP);
            if (!r.first)
            {
                allOk = false;
                detail = "after reset: " + r.second;
            }
        }
        report(fmt("REF-FIDELITY main %s (3 PRNG blocks up+dn bitwise + reset contract)",
                   c.name).c_str(), allOk, detail);
    }

    // ── scalar↔scalar 経路: 3-tap synthetic stage（convCount=2 → 両側 scalar 経路） ──
    {
        CustomInputOversampler prod;
        Shadow shadow{ false };
        prod.prepareSingleStage(3, 90.0, kBlockSize);
        shadow.prepareSingleStageModel(3, 90.0, kBlockSize);
        prod.reset(); shadow.reset();

        DeterministicPRNG rng;
        bool allOk = true;
        std::string detail;
        for (int block = 0; block < 3; ++block)
        {
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < kBlockSize; ++i)
                    inBuf[ch][i] = rng.next();
            const auto r = compareRoundTrip(prod, shadow, inPtrs, kBlockSize,
                                            prodUpP, prodDnP, shadowUpP, shadowDnP);
            if (!r.first)
            {
                allOk = false;
                detail = r.second;
                break;
            }
        }
        report("scalar↔scalar REF-FIDELITY (3-tap synthetic・convCount=2 両側 scalar 経路)",
               allOk, detail);
    }

    std::printf("  [INFO] P0-I (SoftClip local OS) measurement は Phase 0 0-5 で実施（G-0/C0 承認後・R17-1 契約）\n");
    std::printf("  [INFO] production_flag = undefined/off のまま（HOLD: centerValue *= 2.0 / CMake flag / default ON / calibration）\n");

    // ── Phase 0 characterization（P0-A〜P0-I / P0-G・R17-1 動作点 / R12-3 責務分離） ──
    std::printf("  ──── Phase 0 characterization（G-0/T-1/T-2/T-3/C0 承認済み 2026-09-22） ────\n");
    runPhase0Tests();
    std::printf("  [INFO] production_flag = undefined/off のまま（HOLD 継続）・P0 gate 判定は本 battery の PASS/FAIL 行\n");

    return (g_fail == 0) ? 0 : 1;
}

} // namespace

int main()
{
    std::printf("=== PolyphaseGainFidelityTests (work113 C1 / Phase 0 REF-FIDELITY 基盤) ===\n");
    const int rc = runAllTests();
    std::printf("=== summary: PASS=%zu FAIL=%zu ===\n", g_pass, g_fail);
    return rc;
}
