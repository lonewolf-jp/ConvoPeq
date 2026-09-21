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
