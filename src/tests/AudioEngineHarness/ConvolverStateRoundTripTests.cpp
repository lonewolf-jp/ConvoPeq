// ConvolverStateRoundTripTests.cpp — ★ work92 A-1 (big 1-1)
//
// ConvolverProcessor::getState() → setState() の round-trip で
// nucHCMode / nucLCMode が保存・復元されることを直接検証する統合テスト。
//
// 検証項目（PLAN work92 A-1 / RECONCILIATION §1 A-1）:
//   AC-A1-1: round-trip 後のモード一致（setNUCFilterModes → getState → setState →
//            captureBuildSnapshot().nucHCMode/nucLCMode が元値一致）
//   AC-A1-2: 範囲外値 (-1, 99) が jlimit 正規化され、クラッシュしない
//   AC-A1-3: プロパティ不在の ValueTree（旧セッション相当）でデフォルト維持
//   AC-A1-4: round-trip で changeNotification が 1 回だけ coalesce される
//            （setNUCFilterModes の postCoalescedChangeNotification 経路）
//
// 本テストは AudioEngine 実体（uiConvolverProcessor）を必要とするため
// AudioEngineHarness 側に配置（runDeferredFlowIntegrationTests と同一パターン）。

#include "AudioEngineHarness.h"

#include "InputBitDepthTransform.h"

#include <cstdio>
#include <vector>

namespace {

bool checkNucRoundTrip()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[A1] harness start failed\n");
        return false;
    }

    ConvolverProcessor& conv = h.engine().getConvolverProcessor();

    // AC-A1-1: setNUCFilterModes(Soft/Soft) → getState → setState → 読み戻し一致
    conv.setNUCFilterModes(convo::HCMode::Soft, convo::LCMode::Soft);

    juce::ValueTree saved = conv.getState();
    if (!saved.hasProperty("nucHCMode") || !saved.hasProperty("nucLCMode"))
    {
        std::fprintf(stderr, "[A1] FAIL: AC-A1-1 getState() missing nucHCMode/nucLCMode\n");
        h.stop();
        return false;
    }

    // 別の値へ一度変更してから復元する（同一値への no-op set で coalesce が発火しない契約を避ける）
    conv.setNUCFilterModes(convo::HCMode::Sharp, convo::LCMode::Natural);

    // 復元: 保存済み ValueTree をそのまま setState
    conv.setState(saved);

    {
        const auto snap = conv.captureBuildSnapshot();
        const bool okHC = snap.nucHCMode == static_cast<int>(convo::HCMode::Soft);
        const bool okLC = snap.nucLCMode == static_cast<int>(convo::LCMode::Soft);
        if (!okHC || !okLC)
        {
            std::fprintf(stderr, "[A1] FAIL: AC-A1-1 round-trip mismatch: HC=%d (want %d) LC=%d (want %d)\n",
                         snap.nucHCMode, static_cast<int>(convo::HCMode::Soft),
                         snap.nucLCMode, static_cast<int>(convo::LCMode::Soft));
            h.stop();
            return false;
        }
    }

    // AC-A1-2: 範囲外値 (-1, 99) は jlimit 正規化（HC: -1→Sharp(0)・99→Soft(2) / LC: -1→Natural(0)・99→Soft(1)）
    juce::ValueTree outOfRange("Convolver");
    outOfRange.setProperty("nucHCMode", -1, nullptr);
    outOfRange.setProperty("nucLCMode", 99, nullptr);
    conv.setState(outOfRange);
    {
        const auto snap = conv.captureBuildSnapshot();
        const bool okHC = snap.nucHCMode == static_cast<int>(convo::HCMode::Sharp);
        const bool okLC = snap.nucLCMode == static_cast<int>(convo::LCMode::Soft);
        if (!okHC || !okLC)
        {
            std::fprintf(stderr, "[A1] FAIL: AC-A1-2 out-of-range not clamped: HC=%d LC=%d\n",
                         snap.nucHCMode, snap.nucLCMode);
            h.stop();
            return false;
        }
    }

    // AC-A1-3: プロパティ不在の ValueTree で現状維持（旧セッション後方互換・デフォルトへ戻さない）
    conv.setNUCFilterModes(convo::HCMode::Soft, convo::LCMode::Soft);
    juce::ValueTree legacy("Convolver");
    legacy.setProperty("mix", 1.0f, nullptr); // nuc プロパティを持たない旧セッション相当
    conv.setState(legacy);
    {
        const auto snap = conv.captureBuildSnapshot();
        const bool okHC = snap.nucHCMode == static_cast<int>(convo::HCMode::Soft);
        const bool okLC = snap.nucLCMode == static_cast<int>(convo::LCMode::Soft);
        if (!okHC || !okLC)
        {
            std::fprintf(stderr, "[A1] FAIL: AC-A1-3 legacy tree reset modes: HC=%d LC=%d\n",
                         snap.nucHCMode, snap.nucLCMode);
            h.stop();
            return false;
        }
    }

    h.stop();
    std::printf("ConvolverStateRoundTripTests: PASS (AC-A1-1/2/3)\n");
    return true;
}

//==============================================================================
// ★ work92 A-2 (big 1-3): InputBitDepthTransform AVX2 store 契約テスト
//   AC-A2-1: 非アライン dst（double 1 個分オフセット）で #GP 不発
//   AC-A2-2: 数値等価（float→double 変換がスカラー参照と bit 一致）
//==============================================================================
bool checkInputTransformUnalignedDst()
{
    // 入力は sanitizeAndLimit の正規化域 [-1,1] 内に収める（[-0.9, 0.9]）。
    //   → applyHighQuality64BitTransform(gain=1.0) は sanitize のみで恒等写像になり、
    //     スカラー参照 static_cast<double> との bit 一致が検証できる。
    constexpr int kN = 1024;
    std::vector<float> src(kN);
    for (int i = 0; i < kN; ++i)
        src[i] = (static_cast<float>(i % 200) / 200.0f - 0.5f) * 1.8f;

    std::vector<double> baseline(kN + 8, -999.0);
    convo::input_transform::convertFloatToDoubleHighQuality(src.data(), baseline.data(), kN, 1.0);

    // AC-A2-1: dst を double 1 個分（8 バイト）ずらした非アライン領域に書かせる。
    //   旧 _mm256_store_pd ではここで #GP で即死する。
    std::vector<double> storage(kN + 8, -999.0);
    double* unalignedDst = storage.data() + 1;
    convo::input_transform::convertFloatToDoubleHighQuality(src.data(), unalignedDst, kN, 1.0);

    // AC-A2-1 検証: aligned / unaligned 両呼び出しの bit 一致（store 命令差が結果を変えない）
    for (int i = 0; i < kN; ++i)
    {
        if (unalignedDst[i] != baseline[i])
        {
            std::fprintf(stderr, "[A2] FAIL: unaligned dst mismatch at %d\n", i);
            return false;
        }
    }

    // AC-A2-2 検証: 正規化域入力では恒等写像（float→double はスカラー cast と bit 一致）
    for (int i = 0; i < kN; ++i)
    {
        if (baseline[i] != static_cast<double>(src[i]))
        {
            std::fprintf(stderr, "[A2] FAIL: numerical mismatch at %d: %f vs %f\n",
                         i, baseline[i], static_cast<double>(src[i]));
            return false;
        }
    }

    std::printf("checkInputTransformUnalignedDst: PASS (AC-A2-1/2)\n");
    return true;
}

} // namespace

// main 側（PublishPipelineIntegrationTests.cpp）から呼ばれるエントリ
int runConvolverStateRoundTripTests()
{
    if (!checkNucRoundTrip())
    {
        std::fprintf(stderr, "FAIL: checkNucRoundTrip\n");
        return 1;
    }
    if (!checkInputTransformUnalignedDst())
    {
        std::fprintf(stderr, "FAIL: checkInputTransformUnalignedDst\n");
        return 1;
    }
    return 0;
}
