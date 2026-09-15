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

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

// ============================================================================
// ★ H-02 (T1-T4): canonical helper measureIrPeakLatencySamples 回帰テスト
//
// 契約（H-02 Implementation Gate / Contract Freeze 2026-09-13 §2-§4・§7）:
//   C1: irPeakLatency = argmax |sample[ch][i]|，N=min(getNumSamples(), targetLength)，
//       C=getNumChannels() 実測，走査 ch-major・i 昇順・strict > ⇒ tie-break lowest index
//   C2: targetLength<=0 / 全 zero / 非有限全除外 ⇒ 0。NaN/±Inf は比較から明示 skip
//       （+Inf の index が選ばれることを禁止。従来 LoaderThread 重心経路の
//        NaN→static_cast<int>(floor(NaN+.5)) UB を消去）
//   C3: helper 単一・決定的純関数（同一 buffer・同一 targetLength ⇒ 同一返り値）
//   T1-T4 は helper 直接測定（private static → CONVOPEQ_UNIT_TESTS 限定 friend シーム経由）。
//   新 CTest 登録なし（AudioEngineHarness 既存 exe のサブテストとして run 側から呼出）。
// ============================================================================

// ConvolverProcessor.h の #if defined(CONVOPEQ_UNIT_TESTS) friend 宣言に対応する本体定義。
// Production ビルドでは friend 宣言ごと存在せず、本 TU もテストビルド専用。
struct IRPeakLatencyTestAccess
{
    static int measure(const juce::AudioBuffer<double>& ir, int targetLength) noexcept
    {
        return ConvolverProcessor::measureIrPeakLatencySamples(ir, targetLength);
    }
};

// ★ M-04: ConvolverProcessor.h の friend 宣言（global 名前空間で解決される）に対応する本体定義。
//   oversized telemetry counter 読み取りと reporter（timerCallback）駆動専用。
struct M04OversizedTestAccess
{
    static int count() noexcept
    {
        return convo::consumeAtomic(ConvolverProcessor::oversizedBlockCounter(), std::memory_order_relaxed);
    }
    static void pumpReport(ConvolverProcessor& cp) { cp.timerCallback(); }
};

// ★ M-02: ConvolverProcessor.h の friend 宣言（global 名前空間で解決される）に対応する本体定義。
//   wet scrub telemetry counter 読み取りと reporter（timerCallback）駆動専用。
//   （production 駆動休眠は doc/work98 §1.6 の既知事項。test pump が唯一の駆動元）
struct M02NonFiniteTestAccess
{
    static int count() noexcept
    {
        return convo::consumeAtomic(ConvolverProcessor::nonFiniteBlockCounter(), std::memory_order_relaxed);
    }
    static void pumpReport(ConvolverProcessor& cp) { cp.timerCallback(); }
};

// ★ SR-03 T-SR03: AudioEngine.h の #if defined(CONVOPEQ_UNIT_TESTS) friend 宣言に対応する本体定義。
//   latencyDelay publish 単一関口（SR03-C1）の clamp / telemetry / RT 消費前提条件を直接検証する。
class LatencyDelayWiringTestAccess
{
public:
    static void publish(AudioEngine& e, int oldDelay, int newDelay) noexcept
    {
        e.publishLatencyDelayAtomics(oldDelay, newDelay);
    }
    static int readOld(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.latencyDelayOld, std::memory_order_acquire);
    }
    static int readNew(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.latencyDelayNew, std::memory_order_acquire);
    }
    static uint64_t clampCount(AudioEngine& e) noexcept
    {
        return convo::consumeAtomic(e.latencyDelayClampCount_, std::memory_order_acquire);
    }
    static int ringSize(AudioEngine& e) noexcept
    {
        return e.latencyBufSize;
    }
    // T-SR03-6 決定論性: 起動時 crossfade がリングへ残した状態を既知のクリア状態へ戻す
    //（audio thread 停止域専用。prepareToPlay と同一の初期化意味論）。
    static void clearRing(AudioEngine& e) noexcept
    {
        if (e.latencyBufSize <= 0)
            return;
        const size_t bytes = sizeof(double) * static_cast<size_t>(e.latencyBufSize);
        if (e.latencyBufOldL) std::memset(e.latencyBufOldL, 0, bytes);
        if (e.latencyBufOldR) std::memset(e.latencyBufOldR, 0, bytes);
        if (e.latencyBufNewL) std::memset(e.latencyBufNewL, 0, bytes);
        if (e.latencyBufNewR) std::memset(e.latencyBufNewR, 0, bytes);
        e.latencyWritePos = 0;
    }
    // RT アライナ（runLatencyAlignedCrossfadeMixLoop）を bounded delay で直接実行する seam。
    // probe(i, gain, alignedOldL, alignedNewL) をサンプル毎に呼ぶ。audio thread 停止域で実行すること。
    template <typename ProbeFn>
    static void runAligner(AudioEngine& e, double* dstL, double* dstR,
                           const double* oldL, const double* oldR, int numSamples,
                           int delayOld, int delayNew, ProbeFn probe)
    {
        e.runLatencyAlignedCrossfadeMixLoop<double>(
            dstL, dstR, oldL, oldR, numSamples, delayOld, delayNew, false,
            [&probe](double* outL, double* outR, int i, double g,
                     double aoL, double, double anL, double anR) {
                probe(i, g, aoL, anL);
                if (outL != nullptr) *outL = anL;
                if (outR != nullptr) *outR = anR;
            });
    }
};

namespace {

bool expectIrPeak(const juce::AudioBuffer<double>& ir, int targetLength,
                  int expected, const char* label)
{
    const int actual = IRPeakLatencyTestAccess::measure(ir, targetLength);
    if (actual != expected)
    {
        std::fprintf(stderr, "[H02] FAIL: %s expected=%d actual=%d\n", label, expected, actual);
        return false;
    }
    return true;
}

// T1: 48k 指数減衰 τ=100ms・sample0 ピーク ⇒ irPeakLatency == 0
//     （旧 energy-centroid 経路では ~数千サンプルの重心を返し失敗していた。
//       max-abs argmax では 0 = 直接音到達申告として正しい）
bool checkH02T1ExpPeakSample0()
{
    constexpr double sr = 48000.0;
    constexpr double tau = 0.1; // 100 ms
    constexpr int n = 48000;    // 1 s
    juce::AudioBuffer<double> ir(1, n);
    double* d = ir.getWritePointer(0);
    for (int i = 0; i < n; ++i)
        d[i] = std::exp(-static_cast<double>(i) / (tau * sr)); // 最大値は i=0 (1.0)

    if (!expectIrPeak(ir, n, 0, "T1 exp tau=100ms peak@0"))
        return false;

    std::printf("checkH02T1ExpPeakSample0: PASS (T1)\n");
    return true;
}

// T2: L peak@100 (0.8) / R peak@300 (0.9) ⇒ 300（全チャネル argmax・channel 実測）
bool checkH02T2MultiChannel()
{
    juce::AudioBuffer<double> ir(2, 512);
    ir.clear();
    ir.getWritePointer(0)[100] = 0.8;
    ir.getWritePointer(1)[300] = -0.9; // |−0.9| > |0.8| ⇒ R 側が勝つ
    if (!expectIrPeak(ir, 512, 300, "T2 L@100/R@300"))
        return false;

    // mono (C=1) 通常経路（同データ L のみ）
    juce::AudioBuffer<double> mono(1, 512);
    mono.clear();
    mono.getWritePointer(0)[100] = 0.8;
    if (!expectIrPeak(mono, 512, 100, "T2b mono"))
        return false;

    std::printf("checkH02T2MultiChannel: PASS (T2)\n");
    return true;
}

// T3: s10=+1, s20=-1（|value| 同値）⇒ 10（strict > tie-break lowest index）
bool checkH02T3TieBreakLowestIndex()
{
    juce::AudioBuffer<double> ir(1, 64);
    ir.clear();
    ir.getWritePointer(0)[10] = 1.0;
    ir.getWritePointer(0)[20] = -1.0;
    if (!expectIrPeak(ir, 64, 10, "T3 tie-break +1@10/-1@20"))
        return false;

    // ch-major 確認: ch0@30 と ch1@15 が同値最大 ⇒ 先に到達した ch0@30 が勝つ
    //（契約の走査順 ch-major・i 昇順の定義そのまま）
    juce::AudioBuffer<double> dual(2, 64);
    dual.clear();
    dual.getWritePointer(0)[30] = 1.0;
    dual.getWritePointer(1)[15] = 1.0;
    if (!expectIrPeak(dual, 64, 30, "T3b ch-major first-arrival"))
        return false;

    std::printf("checkH02T3TieBreakLowestIndex: PASS (T3)\n");
    return true;
}

// T4: helper 決定的純関数性（同一 buffer・同一 targetLength 同値）＋非有限 skip
//     （+Inf index 不被選・全 NaN→0・全 Inf→0）＋境界（zero/targetLength/channel）
bool checkH02T4NonFiniteAndBoundaries()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    // (a) 決定性: 同一入力 2 回測定で同値（両呼出点同一性の根拠 = 純関数同一 code path）
    juce::AudioBuffer<double> ir(2, 128);
    ir.clear();
    ir.getWritePointer(0)[64] = 0.7;
    ir.getWritePointer(1)[65] = 0.6;
    const int first = IRPeakLatencyTestAccess::measure(ir, 128);
    const int second = IRPeakLatencyTestAccess::measure(ir, 128);
    if (first != 64 || second != first)
    {
        std::fprintf(stderr, "[H02] FAIL: T4a determinism first=%d second=%d (want 64)\n", first, second);
        return false;
    }

    // (b) finite peak + NaN elsewhere ⇒ peak index（NaN 不選出）
    juce::AudioBuffer<double> withNaN(1, 128);
    withNaN.clear();
    withNaN.getWritePointer(0)[50] = 0.5;
    withNaN.getWritePointer(0)[0] = nan;
    withNaN.getWritePointer(0)[127] = nan;
    if (!expectIrPeak(withNaN, 128, 50, "T4b finite+NaN"))
        return false;

    // (c) finite peak + +Inf ⇒ +Inf の index が選ばれないこと（契約 H02-C2 明示禁止）
    juce::AudioBuffer<double> withInf(1, 128);
    withInf.clear();
    withInf.getWritePointer(0)[50] = 0.5;
    withInf.getWritePointer(0)[7] = inf;
    withInf.getWritePointer(0)[9] = -inf;
    if (!expectIrPeak(withInf, 128, 50, "T4c finite+Inf"))
        return false;

    // (d) 全 NaN ⇒ 0 / 全 ±Inf ⇒ 0
    juce::AudioBuffer<double> allNaN(1, 32);
    for (int i = 0; i < 32; ++i)
        allNaN.getWritePointer(0)[i] = nan;
    if (!expectIrPeak(allNaN, 32, 0, "T4d all NaN"))
        return false;

    juce::AudioBuffer<double> allInf(1, 32);
    for (int i = 0; i < 32; ++i)
        allInf.getWritePointer(0)[i] = (i % 2 == 0) ? inf : -inf;
    if (!expectIrPeak(allInf, 32, 0, "T4e all Inf"))
        return false;

    // (f) 全サンプル zero ⇒ 0（bestIndex 初期値が勝つ）
    juce::AudioBuffer<double> zero(2, 32);
    zero.clear();
    if (!expectIrPeak(zero, 32, 0, "T4f all zero"))
        return false;

    // (g) targetLength <= 0 ⇒ 0（C-5 ガード保持）
    if (!expectIrPeak(ir, 0, 0, "T4g targetLength=0"))
        return false;
    if (!expectIrPeak(ir, -5, 0, "T4h targetLength<0"))
        return false;

    // (h) N = min(samples, targetLength) clamp: targetLength 超のピークは対象外
    juce::AudioBuffer<double> beyond(1, 100);
    beyond.clear();
    beyond.getWritePointer(0)[80] = 1.0;
    if (!expectIrPeak(beyond, 50, 0, "T4i peak beyond targetLength"))
        return false;

    // (j) 空バッファ: samples==0 / channels==0 ⇒ 0
    juce::AudioBuffer<double> noSamples(1, 0);
    if (!expectIrPeak(noSamples, 16, 0, "T4j numSamples=0"))
        return false;
    juce::AudioBuffer<double> noChannels(0, 16);
    if (!expectIrPeak(noChannels, 16, 0, "T4k numChannels=0"))
        return false;

    std::printf("checkH02T4NonFiniteAndBoundaries: PASS (T4)\n");
    return true;
}

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

// ============================================================================
// ★ SR-03 (SG-3): latencyDelay publish 単一関口 clamp / telemetry / RT bounded 消費の回帰
//   契約: SR03-C1（cap = latencyBufSize>0 ? size-1 : 0 / published = jlimit(0,cap,raw) /
//         超過値ごとに counter +1 / latencyBufSize==0 は (0,0) 固定）
//         SR03-C2（RT runLatencyAlignedCrossfadeMixLoop / wrapIdx 不動・bounded 値のみ消費）
// ============================================================================
bool checkSR03LatencyDelayClamp()
{
    AudioEngineHarness h;
    AudioEngine& e = h.engine();

    // T-SR03-1: 未 prepared（latencyBufSize==0）→ (0,0) 固定・counter 不発
    if (LatencyDelayWiringTestAccess::ringSize(e) != 0)
    {
        std::fprintf(stderr, "[SR03] FAIL: T1 initial ring size=%d (want 0)\n",
                     LatencyDelayWiringTestAccess::ringSize(e));
        return false;
    }
    {
        const uint64_t base = LatencyDelayWiringTestAccess::clampCount(e);
        LatencyDelayWiringTestAccess::publish(e, 0, 0);
        const int ro = LatencyDelayWiringTestAccess::readOld(e);
        const int rn = LatencyDelayWiringTestAccess::readNew(e);
        const uint64_t c = LatencyDelayWiringTestAccess::clampCount(e);
        if (ro != 0 || rn != 0 || c != base)
        {
            std::fprintf(stderr, "[SR03] FAIL: T1 unprepared publish(0,0) old=%d new=%d count+%llu\n",
                         ro, rn, static_cast<unsigned long long>(c - base));
            return false;
        }
    }

    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[SR03] harness start failed\n");
        return false;
    }

    const int size = LatencyDelayWiringTestAccess::ringSize(e);
    if (size <= 1)
    {
        std::fprintf(stderr, "[SR03] FAIL: prepared ring size=%d\n", size);
        h.stop();
        return false;
    }
    const int cap = size - 1;

    // T-SR03-2: raw == capacity-1（境界内最大）→ 無変換・counter 不発
    {
        const uint64_t base = LatencyDelayWiringTestAccess::clampCount(e);
        LatencyDelayWiringTestAccess::publish(e, cap, 0);
        const int ro = LatencyDelayWiringTestAccess::readOld(e);
        const uint64_t c = LatencyDelayWiringTestAccess::clampCount(e);
        if (ro != cap || c != base)
        {
            std::fprintf(stderr, "[SR03] FAIL: T2 cap=%d -> old=%d count+%llu\n",
                         cap, ro, static_cast<unsigned long long>(c - base));
            h.stop();
            return false;
        }
    }

    // T-SR03-3: raw == capacity → clamp + counter +2（old/new 独立超過）
    {
        const uint64_t base = LatencyDelayWiringTestAccess::clampCount(e);
        LatencyDelayWiringTestAccess::publish(e, size, size);
        const int ro = LatencyDelayWiringTestAccess::readOld(e);
        const int rn = LatencyDelayWiringTestAccess::readNew(e);
        const uint64_t c = LatencyDelayWiringTestAccess::clampCount(e);
        if (ro != cap || rn != cap || (c - base) != 2)
        {
            std::fprintf(stderr, "[SR03] FAIL: T3 size=%d -> old=%d new=%d count+%llu (want +2)\n",
                         size, ro, rn, static_cast<unsigned long long>(c - base));
            h.stop();
            return false;
        }
    }

    // T-SR03-4: raw >> capacity → clamp・bounded 不変式 0 <= atom <= cap
    {
        const uint64_t base = LatencyDelayWiringTestAccess::clampCount(e);
        LatencyDelayWiringTestAccess::publish(e, 2 * size, 2621440); // MAX_TOTAL_DELAY 相当
        const int ro = LatencyDelayWiringTestAccess::readOld(e);
        const int rn = LatencyDelayWiringTestAccess::readNew(e);
        const uint64_t c = LatencyDelayWiringTestAccess::clampCount(e);
        if (ro != cap || rn != cap || (c - base) != 2)
        {
            std::fprintf(stderr, "[SR03] FAIL: T4 -> old=%d new=%d count+%llu\n",
                         ro, rn, static_cast<unsigned long long>(c - base));
            h.stop();
            return false;
        }
    }

    // T-SR03-5: 構造的到達最大総和 2,621,439 (= MAX_BLOCK_SIZE + MAX_IR_LATENCY-1)
    {
        const uint64_t base = LatencyDelayWiringTestAccess::clampCount(e);
        LatencyDelayWiringTestAccess::publish(e, 2621439, 0);
        const int ro = LatencyDelayWiringTestAccess::readOld(e);
        const uint64_t c = LatencyDelayWiringTestAccess::clampCount(e);
        if (ro != cap || (c - base) != 1)
        {
            std::fprintf(stderr, "[SR03] FAIL: T5 2621439 -> old=%d count+%llu\n",
                         ro, static_cast<unsigned long long>(c - base));
            h.stop();
            return false;
        }
    }

    // T-SR03-6: RT 消費の bounded 前提 — aligner を delayOld=cap / delayNew=0 でラップ 2 周実行。
    //   readOld = (writePos - cap) % size = (writePos + 1) % size の関係を用いると:
    //   lap1: i<size-1 は slot 未書込=clearRing 済 0 → alignedOld==0、
    //         i==size-1 は slot0（同 lap i=0 で書込済）→ alignedOld==7.0
    //   lap2: 全 slot が lap1 の oldL=7.0 → alignedOld==7.0
    //   delayNew=0 → alignedNew は書き込み直後値 = 入力 passthrough（両 lap）
    LatencyDelayWiringTestAccess::publish(e, 0, 0);
    h.stopAudioOnly();
    LatencyDelayWiringTestAccess::clearRing(e);
    {
        const int n = size;
        std::vector<double> dstL(static_cast<size_t>(n)), dstR(static_cast<size_t>(n), 0.0);
        std::vector<double> oldL(static_cast<size_t>(n), 7.0), oldR(static_cast<size_t>(n), 7.0);
        bool ok = true;
        for (int lap = 1; lap <= 2 && ok; ++lap)
        {
            for (int i = 0; i < n; ++i)
                dstL[static_cast<size_t>(i)] = 1000.0 + static_cast<double>(i);
            bool allOldOk = true, allNewPass = true;
            LatencyDelayWiringTestAccess::runAligner(
                e, dstL.data(), dstR.data(), oldL.data(), oldR.data(), n, cap, 0,
                [&](int i, double, double aoL, double anL) {
                    if (lap == 1)
                    {
                        const double want = (i == n - 1) ? 7.0 : 0.0;
                        if (std::fabs(aoL - want) > 1e-12) allOldOk = false;
                    }
                    else
                    {
                        if (std::fabs(aoL - 7.0) > 1e-12) allOldOk = false;
                    }
                    if (std::fabs(anL - (1000.0 + static_cast<double>(i))) > 1e-12) allNewPass = false;
                });
            if (!allNewPass)
            {
                std::fprintf(stderr, "[SR03] FAIL: T6 lap%d new passthrough violated\n", lap);
                ok = false;
            }
            if (!allOldOk)
            {
                std::fprintf(stderr, "[SR03] FAIL: T6 lap%d old ring semantics mismatch (wrapIdx read)\n", lap);
                ok = false;
            }
        }
        if (!ok)
        {
            h.stop();
            return false;
        }
    }
    LatencyDelayWiringTestAccess::publish(e, 0, 0);
    h.stop();

    std::printf("checkSR03LatencyDelayClamp: PASS (T-SR03-1..6)\n");
    return true;
}

// ============================================================================
// ★ H-01 (T-H01-1..6): internal dry alignment = irPeak のみ／host PDC 不変 の回帰
//
// 契約（H-01 Implementation Gate / Contract Freeze 2026-09-14）:
//   wet 実到着位置を基準に、dry 直音到着位置との差が 0 samples であること（核心判定）。
//   flag 語義: kDryDelayUsesAlgorithmLatency true=legacy(algo+peak) / false=H-01(peakのみ)。
//   host PDC / breakdown（algo+peak）は flag に関係なく不変（方案 C）。
// 実装: ConvolverProcessor を単体で駆動（メモリ WAV → loadImpulseResponse → poll → impulse 計測）。
// ============================================================================

namespace {

// 指定 peakPos に δ を持つ 48k/2ch/100ms WAV を temp に書き出す。
juce::File writeH01TempIr(const juce::String& tag, int peakPos)
{
    const juce::File f = juce::File::getSpecialLocation(juce::File::tempDirectory)
                             .getChildFile(tag);
    f.deleteFile();
    std::unique_ptr<juce::OutputStream> stream(f.createOutputStream());
    if (stream == nullptr)
        return {};
    juce::WavAudioFormat fmt;
    std::unique_ptr<juce::AudioFormatWriter> w(fmt.createWriterFor(stream,
        juce::AudioFormatWriterOptions{}
            .withSampleRate(48000.0)
            .withNumChannels(2)
            .withBitsPerSample(16)));
    if (w == nullptr)
        return {};
    juce::AudioBuffer<float> buf(2, 4800);
    buf.clear();
    buf.getWritePointer(0)[peakPos] = 1.0f;
    buf.getWritePointer(1)[peakPos] = 1.0f;
    if (!w->writeFromAudioSampleBuffer(buf, 0, 4800))
        return {};
    w.reset(); // flush before stream destruction
    return f;
}

// ★ 座標系: measureH01Arrival は **グローバル sample index** を返す（impulse は
//   kH01ImpulseBase に打たれる）。到着期待値 = kH01ImpulseBase + peakPos。
// ★ warmup は NUC B13-GATE の lead 制約（lead ≤ o_L - B、P=4096 実測 o_L=5760・B=512
//   → lead≤5248）を満たす 10 block に抑える（mix/latency smoother ramp は 960/4800 sample
//   で 10 block=5120 内で収束）。
constexpr int kH01Block = 512;
constexpr int kH01WarmupBlocks = 10;
constexpr int kH01ImpulseBase = kH01WarmupBlocks * kH01Block;

// mix を設定後、impulse(ch0/ch1 @block0/sample0)を process で流し、
// 最大 |L|+|R| 到着(global sample index)を返す。warmup で mix/smoothing ramp を吸収。
int measureH01Arrival(ConvolverProcessor& conv, float mix, int captureBlocks)
{
    conv.setMix(mix);
    constexpr int kBlock = kH01Block;
    constexpr int kWarmup = kH01WarmupBlocks;
    juce::AudioBuffer<double> ab(2, kBlock);
    juce::dsp::AudioBlock<double> blk(ab);
    for (int i = 0; i < kWarmup; ++i)
    {
        ab.clear();
        conv.process(blk);
    }
    int bestIdx = -1;
    double bestVal = 0.0;
    for (int b = 0; b < captureBlocks; ++b)
    {
        ab.clear();
        if (b == 0)
        {
            ab.getWritePointer(0)[0] = 1.0;
            ab.getWritePointer(1)[0] = 1.0;
        }
        conv.process(blk);
        for (int i = 0; i < ab.getNumSamples(); ++i)
        {
            const double v = std::fabs(ab.getWritePointer(0)[i]) + std::fabs(ab.getWritePointer(1)[i]);
            if (v > bestVal)
            {
                bestVal = v;
                bestIdx = (b + kWarmup) * kBlock + i;
            }
        }
    }
    return bestIdx;
}

// ★ JUCE 制約: loadImpulseResponse の finalize は queueFinalizeOnMessageThread（MessageManager）経由。
//   console test には MainApplication の message loop が無いため、test 本体と同じ main thread で
//   MM を初期化し、poll ループ内で Win32 メッセージを自 pump する（コンソール JUCE の定石）。
//   （runDispatchLoopUntil は JUCE_MODAL_LOOPS_PERMITTED=0 で不使用可、専用 pump thread は
//    Windows の JUCE message thread 制約と衝突するため本方式を採用）
namespace {

void pumpH01Messages() noexcept
{
#if JUCE_WINDOWS
    MSG msg {};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
#endif
}

} // namespace

// load 完了を irPeakLatencySamples 経由で poll（最大 ~10s、messages pump 込み）。失敗時は進捗を診断 print。
bool pollH01Peak(ConvolverProcessor& conv, int expectedPeak)
{
    for (int i = 0; i < 2000; ++i)
    {
        const auto bd = conv.getLatencyBreakdown();
        if (bd.irPeakLatencySamples == expectedPeak && bd.totalLatencySamples >= expectedPeak)
            return true;
        pumpH01Messages();
        juce::Thread::sleep(5);
    }
    std::fprintf(stderr, "[H01] diag: loading=%d finalized=%d irLen=%d algo=%d peak=%d err='%s'\n",
                 conv.isLoadingIR() ? 1 : 0, conv.isIRFinalized() ? 1 : 0,
                 conv.getIRLength(),
                 conv.getLatencyBreakdown().algorithmLatencySamples,
                 conv.getLatencyBreakdown().irPeakLatencySamples,
                 conv.getLastError().toRawUTF8());
    pumpH01Messages();
    return false;
}

} // namespace

bool checkH01DryWetAlignment()
{
    constexpr int kPeakA = 100;
    constexpr int kPeakB = 200;

    // ★ 診断: JUCE Logger の既定出力は ODS でキャプチャ不能なため、この check 中は
    //   stderr に反映する（LoaderThread / finalize / init 失敗理由の確定用）。
    struct H01Logger : juce::Logger {
        void logMessage(const juce::String& m) override
        {
            std::fprintf(stderr, "[H01log] %s\n", m.toRawUTF8());
            std::fflush(stderr);
        }
    };
    static H01Logger h01Log;
    juce::Logger* prevLogger = juce::Logger::getCurrentLogger();
    juce::Logger::setCurrentLogger(&h01Log);

    ConvolverProcessor conv;
    conv.prepareToPlay(48000.0, 512);

    // ★ JUCE 制約: finalize の MessageManager ディスパッチには main thread の MM が必要
    //   （poll 側の pumpH01Messages で Win32 メッセージを消費する）。static = プロセス lifetime・冪等。
    static juce::ScopedJuceInitialiser_GUI h01JuceInit;
    (void)h01JuceInit;

    const juce::File irA = writeH01TempIr("h01_a.wav", kPeakA);
    if (!irA.existsAsFile())
    {
        std::fprintf(stderr, "[H01] FAIL: temp IR A write failed\n");
        return false;
    }
    conv.loadImpulseResponse(irA, false);
    if (!pollH01Peak(conv, kPeakA))
    {
        const auto bd0 = conv.getLatencyBreakdown();
        std::fprintf(stderr, "[H01] FAIL: IR A load timeout (irPeak=%d total=%d)\n",
                     bd0.irPeakLatencySamples, bd0.totalLatencySamples);
        irA.deleteFile();
        return false;
    }

    // T-H01-4（先に PDC 申告値）: breakdown = algo + peak（報告基準は flag 無関係で不変）。
    const auto bd = conv.getLatencyBreakdown();
    if (bd.irPeakLatencySamples != kPeakA
        || bd.algorithmLatencySamples <= 0
        || bd.totalLatencySamples != bd.algorithmLatencySamples + kPeakA)
    {
        std::fprintf(stderr, "[H01] FAIL: T-H01-4 PDC breakdown: algo=%d peak=%d total=%d\n",
                     bd.algorithmLatencySamples, bd.irPeakLatencySamples, bd.totalLatencySamples);
        irA.deleteFile();
        return false;
    }
    const int algorithmLatency = bd.algorithmLatencySamples;

    // T-H01-1/2/3/5/6 共通の計測ウィンドウ（peak + ring tail 余裕）
    const int kBlocks = (kPeakB + 2048) / 512 + 6;

    const int wetA = measureH01Arrival(conv, 1.0f, kBlocks);   // mix=1.0 → wet のみ
    const int dryA = measureH01Arrival(conv, 0.0f, kBlocks);   // mix=0.0 → dry のみ
    const int mix50A = measureH01Arrival(conv, 0.5f, kBlocks); // mix=0.5 → 構成波 peak

    bool ok = true;
    // T-H01-2: wet 実到着 == impulse global + peak（基準の成立・global 座標系）
    if (wetA != kH01ImpulseBase + kPeakA)
    {
        std::fprintf(stderr, "[H01] FAIL: T-H01-2 wet arrival=%d (want %d)\n",
                     wetA, kH01ImpulseBase + kPeakA);
        ok = false;
    }
    // T-H01-1 核心: wet 実測到着基準で dry との差 0（legacy なら algo=algorithmLatency の差が出る）
    if (dryA != wetA || mix50A != wetA)
    {
        std::fprintf(stderr, "[H01] FAIL: T-H01-1 wet=%d dry=%d mix50=%d (want wet==dry==mix50, algo=%d)\n",
                     wetA, dryA, mix50A, algorithmLatency);
        ok = false;
    }

    // T-H01-3: bypass（dry リングのみ・main と同一 alignment 基準）は dry/wet と同位置
    conv.setMix(1.0f);
    conv.setBypass(true);
    const int bypA = measureH01Arrival(conv, 1.0f, kBlocks);
    conv.setBypass(false);
    if (bypA != wetA)
    {
        std::fprintf(stderr, "[H01] FAIL: T-H01-3 bypass arrival=%d (want %d)\n", bypA, wetA);
        ok = false;
    }

    // T-H01-5: IR 差替（peak 100→200、Δ=100≥retarget 閾値 2.0 → crossfade 発火後安定）
    const juce::File irB = writeH01TempIr("h01_b.wav", kPeakB);
    if (!irB.existsAsFile())
    {
        std::fprintf(stderr, "[H01] FAIL: temp IR B write failed\n");
        irA.deleteFile();
        return false;
    }
    conv.loadImpulseResponse(irB, false);
    if (!pollH01Peak(conv, kPeakB))
    {
        std::fprintf(stderr, "[H01] FAIL: IR B load timeout\n");
        irA.deleteFile(); irB.deleteFile();
        return false;
    }
    const auto bdB = conv.getLatencyBreakdown();
    if (bdB.totalLatencySamples != bdB.algorithmLatencySamples + kPeakB)
    {
        std::fprintf(stderr, "[H01] FAIL: T-H01-5 PDC after retarget: algo=%d total=%d (want %d)\n",
                     bdB.algorithmLatencySamples, bdB.totalLatencySamples,
                     bdB.algorithmLatencySamples + kPeakB);
        ok = false;
    }
    const int wetB = measureH01Arrival(conv, 1.0f, kBlocks);
    const int dryB = measureH01Arrival(conv, 0.0f, kBlocks);
    if (wetB != kH01ImpulseBase + kPeakB || dryB != wetB)
    {
        std::fprintf(stderr, "[H01] FAIL: T-H01-5 post-retarget wet=%d dry=%d (want %d/%d)\n",
                     wetB, dryB, kH01ImpulseBase + kPeakB, kH01ImpulseBase + kPeakB);
        ok = false;
    }

    // T-H01-6: directHead ON（algo==0 → 変換恒等）で同一アサーション（回帰）
    {
        ConvolverProcessor convDh;
        convDh.prepareToPlay(48000.0, 512);
        convDh.setExperimentalDirectHeadEnabled(true);
        const juce::File irD = writeH01TempIr("h01_d.wav", kPeakA);
        bool dhReady = false;
        if (irD.existsAsFile())
        {
            convDh.loadImpulseResponse(irD, false);
            dhReady = pollH01Peak(convDh, kPeakA);
            if (!dhReady)
                std::fprintf(stderr, "[H01] FAIL: directHead IR load timeout\n");
            irD.deleteFile();
        }
        else
        {
            std::fprintf(stderr, "[H01] FAIL: directHead IR write failed\n");
        }
        if (dhReady)
        {
            const auto bdD = convDh.getLatencyBreakdown();
            const int wetD = measureH01Arrival(convDh, 1.0f, kBlocks);
            const int dryD = measureH01Arrival(convDh, 0.0f, kBlocks);
            if (bdD.algorithmLatencySamples != 0 || wetD != kH01ImpulseBase + kPeakA || dryD != wetD)
            {
                std::fprintf(stderr, "[H01] FAIL: T-H01-6 directHead algo=%d wet=%d dry=%d\n",
                             bdD.algorithmLatencySamples, wetD, dryD);
                ok = false;
            }
        }
        else
        {
            ok = false;
        }
    }

    irA.deleteFile();
    irB.deleteFile();
    juce::Logger::setCurrentLogger(prevLogger);

    if (!ok)
        return false;
    // ★ 注記: 本 check の駆動中、NUC 側の [B13-GATE] L1 observability log が lead 前提
    //   （engine callback 定常スケジュール想定の gate）で NG 通知し得る（standalone burst 駆動固有）。
    //   実測の出力配置は正確（wet=impulse+peak が全ケース一致）であり、本 log は
    //   合否条件に含めない（MT-NUPC の oPE 観測と同扱い。product エンジン駆動経路では未発火確認済）。
    std::printf("checkH01DryWetAlignment: PASS (T-H01-1..6: algo=%d, wet=%d dry=%d mix50=%d bypass=%d, retarget wet=%d, directHead ok)\n",
                algorithmLatency, wetA, dryA, mix50A, bypA, wetB);
    return true;
}

// ============================================================================
// ★ SR-01(B) (T-SR01-1..8): hardMaxSec(sr) = MAX_IR_LATENCY / sr 契約凍結の回帰
//   根拠: doc/work95/sr01b_implementation_gate_contract_freeze_20260914.md
//   C-1 copySnapshotToPendingUnlocked の SR 依存 clamp / C-2 UI canonical helper 統一
//   / C-3 computeTargetIRLength cap 発動ログ。容量定数（cap 2^21 / DELAY 2^22）は
//   ピン留めアサーションで不変契約化する（変更＝STOP 条件）。
//   凍結値: @768k setter/snapshot 経路 targetLength = 2,097,151
//             = int(sr × (double)float(MAX_IR_LATENCY/sr))   [float 正規化込み]
//           @768k cap 発動経路（未 prepare 復元 3.0 → load）= 2,097,152 = MAX_IR_LATENCY
// ============================================================================

namespace {

// 実装と同一の丸み連鎖（double 除算 → float 正規化 → double 戻し × sr → int トランケート）。
// ※ float 除算を直接使うと double 丸め経路と一致しない可能性があるため本形を契約とする。
constexpr float kSR01HardMax768kF = static_cast<float>(2097152.0 / 768000.0);
constexpr int kSR01Target768kSetter = static_cast<int>(768000.0 * static_cast<double>(kSR01HardMax768kF));
static_assert(kSR01Target768kSetter == 2097151, "SR-01 frozen value drift (768k setter path)");

// C-3 ログ捕捉: "[SR-01]" を含む行だけをロック付きで蓄積し、診断のため stderr へも写す。
struct SR01Logger : juce::Logger
{
    juce::CriticalSection cs;
    juce::String captured;

    void logMessage(const juce::String& m) override
    {
        {
            const juce::ScopedLock sl(cs);
            if (m.contains("SR-01"))
                captured += m + "\n";
        }
        std::fprintf(stderr, "[SR01log] %s\n", m.toRawUTF8());
        std::fflush(stderr);
    }
    [[nodiscard]] bool has(const juce::String& token)
    {
        const juce::ScopedLock sl(cs);
        return captured.contains(token);
    }
    void clear()
    {
        const juce::ScopedLock sl(cs);
        captured.clear();
    }
};

SR01Logger g_sr01Logger;

struct SR01LoggerScope
{
    juce::Logger* prev;
    SR01LoggerScope() : prev(juce::Logger::getCurrentLogger())
    {
        juce::Logger::setCurrentLogger(&g_sr01Logger);
    }
    ~SR01LoggerScope() { juce::Logger::setCurrentLogger(prev); }
};

// numCh ch × totalSamples @ sampleRate の WAV を temp に書き出す。peakPos<0 は全面ゼロ。
// 768k×3s = 2,304,000 samples（float buf で ~18MB）をチャンク分割で書く。
juce::File writeSR01TempIr(const juce::String& tag, double sampleRate, int numCh, int totalSamples, int peakPos)
{
    const juce::File f = juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile(tag);
    f.deleteFile();
    std::unique_ptr<juce::OutputStream> stream(f.createOutputStream());
    if (stream == nullptr)
        return {};
    juce::WavAudioFormat fmt;
    std::unique_ptr<juce::AudioFormatWriter> w(fmt.createWriterFor(stream,
        juce::AudioFormatWriterOptions{}
            .withSampleRate(sampleRate)
            .withNumChannels(numCh)
            .withBitsPerSample(16)));
    if (w == nullptr)
        return {};
    juce::AudioBuffer<float> buf(numCh, totalSamples);
    buf.clear();
    if (peakPos >= 0)
    {
        for (int ch = 0; ch < numCh; ++ch)
            buf.getWritePointer(ch)[peakPos] = 1.0f;
    }
    int written = 0;
    while (written < totalSamples)
    {
        const int n = std::min(65536, totalSamples - written);
        if (!w->writeFromAudioSampleBuffer(buf, written, n))
            return {};
        written += n;
    }
    w.reset(); // flush before stream destruction
    return f;
}

// irPeak == expectPeak かつ irLength == expectIRLen を同一 finalize 経由で poll（最大 ~60s）。
bool pollSR01Load(ConvolverProcessor& conv, int expectPeak, int expectIRLen, int maxIter = 12000)
{
    for (int i = 0; i < maxIter; ++i)
    {
        const auto bd = conv.getLatencyBreakdown();
        if (bd.irPeakLatencySamples == expectPeak && conv.getIRLength() == expectIRLen)
            return true;
        pumpH01Messages();
        juce::Thread::sleep(5);
    }
    const auto bd = conv.getLatencyBreakdown();
    std::fprintf(stderr, "[SR01] poll timeout: peak=%d/%d irLen=%d/%d err='%s'\n",
                 bd.irPeakLatencySamples, expectPeak, conv.getIRLength(), expectIRLen,
                 conv.getLastError().toRawUTF8());
    return false;
}

} // namespace

// T-SR01-1: hardMaxSec 式テーブル一致 + 容量不変契約のピン留め
bool checkSR01HardMaxFormula()
{
    static constexpr int kFrozenMaxIrLatency = 2097152;    // 2^21 — 変更は STOP 条件（容量不変契約）
    static constexpr int kFrozenDelayBuffer  = 4194304;    // 2^22 — 同上
    if (ConvolverProcessor::MAX_IR_LATENCY != kFrozenMaxIrLatency
        || ConvolverProcessor::DELAY_BUFFER_SIZE != kFrozenDelayBuffer
        || ConvolverProcessor::IR_LENGTH_MAX_SEC != 3.0f
        || ConvolverProcessor::IR_LENGTH_MIN_SEC != 0.5f)
    {
        std::fprintf(stderr, "[SR01] FAIL: capacity constants drifted (MAX_IR=%d DELAY=%d)\n",
                     ConvolverProcessor::MAX_IR_LATENCY, ConvolverProcessor::DELAY_BUFFER_SIZE);
        return false;
    }
    const double srs[] = {44100.0, 48000.0, 88200.0, 96000.0, 176400.0, 192000.0,
                          352800.0, 384000.0, 705600.0, 768000.0};
    for (const double sr : srs)
    {
        const float want = static_cast<float>(static_cast<double>(kFrozenMaxIrLatency) / sr);
        const float got = ConvolverProcessor::getMaximumAllowedIRLengthSecForSampleRate(sr);
        if (got != want)
        {
            std::fprintf(stderr, "[SR01] FAIL: hardMax(%f)=%f want %f\n", sr, got, want);
            return false;
        }
        // 非発動域（sr <= 699,050 Hz = 2^21/3.0）は hardMax >= 3.0 → 既存挙動不変の保証
        if (sr <= 699050.0 && got < 3.0f)
        {
            std::fprintf(stderr, "[SR01] FAIL: hardMax(%f)=%f < 3.0 in non-binding region\n", sr, got);
            return false;
        }
    }
    if (ConvolverProcessor::getMaximumAllowedIRLengthSecForSampleRate(768000.0) != kSR01HardMax768kF)
    {
        std::fprintf(stderr, "[SR01] FAIL: 768k hardMax != frozen %f\n", kSR01HardMax768kF);
        return false;
    }
    if (ConvolverProcessor::getMaximumAllowedIRLengthSecForSampleRate(0.0) != 3.0f
        || ConvolverProcessor::getMaximumAllowedIRLengthSecForSampleRate(-48000.0) != 3.0f)
    {
        std::fprintf(stderr, "[SR01] FAIL: sr<=0 fallback != 3.0\n");
        return false;
    }
    std::printf("checkSR01HardMaxFormula: PASS (44.1k..768k + constants pinned)\n");
    return true;
}

// T-SR01-2: setter 系 clamp（@768k → hardMax / @48k → 3.0 維持 / 下限 0.5）
bool checkSR01SetterClamp()
{
    static juce::ScopedJuceInitialiser_GUI sr01JuceInit;
    ConvolverProcessor conv;
    conv.prepareToPlay(768000.0, 512);
    const float hardMax768 = conv.getMaximumAllowedIRLengthSec();
    if (hardMax768 != kSR01HardMax768kF)
    {
        std::fprintf(stderr, "[SR01] FAIL: getMaximumAllowedIRLengthSec(768k)=%f want %f\n",
                     hardMax768, kSR01HardMax768kF);
        return false;
    }
    conv.setTargetIRLength(3.0f);
    if (conv.getTargetIRLength() != hardMax768)
    {
        std::fprintf(stderr, "[SR01] FAIL: setTargetIRLength(3.0)@768k -> %f want %f\n",
                     conv.getTargetIRLength(), hardMax768);
        return false;
    }
    conv.applyAutoDetectedIRLength(10.0f);
    if (conv.getTargetIRLength() != hardMax768)
    {
        std::fprintf(stderr, "[SR01] FAIL: applyAutoDetected(10.0)@768k -> %f\n", conv.getTargetIRLength());
        return false;
    }
    conv.prepareToPlay(48000.0, 512);
    conv.setTargetIRLength(3.0f);
    if (conv.getTargetIRLength() != 3.0f)
    {
        std::fprintf(stderr, "[SR01] FAIL: setTargetIRLength(3.0)@48k -> %f\n", conv.getTargetIRLength());
        return false;
    }
    conv.setTargetIRLength(0.3f);
    if (conv.getTargetIRLength() != ConvolverProcessor::IR_LENGTH_MIN_SEC)
    {
        std::fprintf(stderr, "[SR01] FAIL: lower clamp -> %f\n", conv.getTargetIRLength());
        return false;
    }
    std::printf("checkSR01SetterClamp: PASS (768k hardMax / 48k unchanged / lower 0.5)\n");
    return true;
}

// T-SR01-3: ★C-1 回帰 — BuildSnapshot 適用経路の clamp が SR 依存 hardMax に統一されたこと
bool checkSR01SnapshotClamp()
{
    ConvolverProcessor conv;
    conv.prepareToPlay(48000.0, 512);
    conv.setTargetIRLength(3.0f);
    conv.applyAutoDetectedIRLength(3.0f);
    const auto snap = conv.captureBuildSnapshot();
    if (snap.targetIRLengthSec != 3.0f)
    {
        std::fprintf(stderr, "[SR01] FAIL: snapshot target not 3.0 @48k\n");
        return false;
    }
    // 48k で確定した 3.0 を 768k 環境へ適用 → C-1 後は hardMax(2.7306667f)。修正前は 3.0 が素通しだった。
    conv.prepareToPlay(768000.0, 512);
    conv.applyBuildSnapshot(snap);
    const float hardMax768 = conv.getMaximumAllowedIRLengthSec();
    if (conv.getTargetIRLength() != hardMax768
        || conv.getAutoDetectedIRLength() != hardMax768)
    {
        std::fprintf(stderr, "[SR01] FAIL: C-1 snapshot@768k target=%f auto=%f want %f\n",
                     conv.getTargetIRLength(), conv.getAutoDetectedIRLength(), hardMax768);
        return false;
    }
    // 48k へ戻して再適用 → 3.0 保存（非発動域の完全回帰）
    conv.prepareToPlay(48000.0, 512);
    conv.applyBuildSnapshot(snap);
    if (conv.getTargetIRLength() != 3.0f || conv.getAutoDetectedIRLength() != 3.0f)
    {
        std::fprintf(stderr, "[SR01] FAIL: C-1 snapshot@48k target=%f\n", conv.getTargetIRLength());
        return false;
    }
    std::printf("checkSR01SnapshotClamp: PASS (C-1: 768k clamp / 48k unchanged)\n");
    return true;
}

// T-SR01-4a: 768k で 3s IR ロード（setter→hardMax 経路）— targetLength 2,097,151・cap ログ不发火
bool checkSR01Load768k3s()
{
    SR01LoggerScope scope;
    g_sr01Logger.clear();
    ConvolverProcessor conv;
    conv.prepareToPlay(768000.0, 512);
    conv.setTargetIRLength(3.0f);
    const juce::File ir = writeSR01TempIr("sr01_4a.wav", 768000.0, 2, 2304000, 100);
    if (!ir.existsAsFile())
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-4a temp IR write failed\n");
        return false;
    }
    conv.loadImpulseResponse(ir, false);
    const bool ok = pollSR01Load(conv, 100, kSR01Target768kSetter);
    // breakdown 契約（H-02/SR-03 不変）: total == algo + peak。cap 未満経路なので C-3 発火しない。
    const auto bd = conv.getLatencyBreakdown();
    const bool contract = ok
        && bd.totalLatencySamples == bd.algorithmLatencySamples + 100
        && !g_sr01Logger.has("SR-01");
    ir.deleteFile();
    if (!contract)
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-4a ok=%d algo=%d total=%d capLog=%d\n",
                     ok ? 1 : 0, bd.algorithmLatencySamples, bd.totalLatencySamples,
                     g_sr01Logger.has("SR-01") ? 1 : 0);
        return false;
    }
    std::printf("checkSR01Load768k3s: PASS (4a: irLen=%d, no cap log, breakdown intact)\n",
                kSR01Target768kSetter);
    return true;
}

// T-SR01-4b: 未 prepare 復元(3.0)→ prepare(768k) → ロードで cap 発動 — 2,097,152・C-3 ログ発火
bool checkSR01CapFiredLog()
{
    SR01LoggerScope scope;
    g_sr01Logger.clear();
    ConvolverProcessor conv;
    juce::ValueTree st("SR01B");
    st.setProperty("irLength", 3.0, nullptr);
    st.setProperty("irLengthManualOverride", true, nullptr);
    conv.setState(st);            // currentSampleRate==0 → フォールバック 3.0 が許容（契約 §6）
    conv.prepareToPlay(768000.0, 512); // prepare は pending を再同期しない（契約 §1.3 注記）
    if (conv.getTargetIRLength() != 3.0f)
    {
        std::fprintf(stderr, "[SR01] FAIL: 4b pre-load pending=%f want 3.0\n", conv.getTargetIRLength());
        return false;
    }
    const juce::File ir = writeSR01TempIr("sr01_4b.wav", 768000.0, 2, 2304000, 100);
    if (!ir.existsAsFile())
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-4b temp IR write failed\n");
        return false;
    }
    conv.loadImpulseResponse(ir, false);
    const bool ok = pollSR01Load(conv, 100, 2097152);   // min(raw 2,304,000, cap) = 2^21 ちょうど
    const bool logged = g_sr01Logger.has("MAX_IR_LATENCY");
    ir.deleteFile();
    if (!ok || !logged)
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-4b ok=%d capLog=%d (want both 1)\n",
                     ok ? 1 : 0, logged ? 1 : 0);
        return false;
    }
    std::printf("checkSR01CapFiredLog: PASS (4b: irLen=2097152, [SR-01] trim log fired)\n");
    return true;
}

// T-SR01-5: 44.1k/48k 完全回帰 — 3s = 132,300 / 144,000 samples、cap ログなし
bool checkSR01LowRateRegression()
{
    SR01LoggerScope scope;
    struct Case { double sr; int samples; };
    const Case cases[] = {{44100.0, 132300}, {48000.0, 144000}};
    for (const Case& c : cases)
    {
        g_sr01Logger.clear();
        ConvolverProcessor conv;
        conv.prepareToPlay(c.sr, 512);
        conv.setTargetIRLength(3.0f);
        const juce::String tag = juce::String("sr01_5_") + juce::String(static_cast<juce::int64>(c.sr)) + ".wav";
        const juce::File ir = writeSR01TempIr(tag, c.sr, 2, c.samples, 100);
        if (!ir.existsAsFile())
        {
            std::fprintf(stderr, "[SR01] FAIL: T-SR01-5 IR write %s\n", tag.toRawUTF8());
            return false;
        }
        conv.loadImpulseResponse(ir, false);
        const bool ok = pollSR01Load(conv, 100, c.samples) && !g_sr01Logger.has("SR-01");
        ir.deleteFile();
        if (!ok)
        {
            std::fprintf(stderr, "[SR01] FAIL: T-SR01-5 sr=%f irLen=%d want=%d capLog=%d\n",
                         c.sr, conv.getIRLength(), c.samples, g_sr01Logger.has("SR-01") ? 1 : 0);
            return false;
        }
    }
    std::printf("checkSR01LowRateRegression: PASS (44.1k=132300 / 48k=144000, no cap log)\n");
    return true;
}

// T-SR01-6: 永続状態の SR 横断互換 — 48k保存3.0→768k復元=hardMax / 768k保存2.73→48k=2.73 のまま
bool checkSR01StateCompatibility()
{
    ConvolverProcessor src;
    src.prepareToPlay(48000.0, 512);
    src.setTargetIRLength(3.0f);
    src.setIRLengthManualOverride(true);
    const auto tree48 = src.getState();
    {
        const double saved = static_cast<double>(tree48.getProperty("irLength", -1.0));
        if (saved < ConvolverProcessor::IR_LENGTH_MIN_SEC - 1e-9
            || saved > ConvolverProcessor::IR_LENGTH_MAX_SEC + 1e-9)
        {
            std::fprintf(stderr, "[SR01] FAIL: T-SR01-6 getState irLength=%f outside [0.5,3.0]\n", saved);
            return false;
        }
    }
    ConvolverProcessor dst;
    dst.prepareToPlay(768000.0, 512);
    dst.setState(tree48);
    if (dst.getTargetIRLength() != kSR01HardMax768kF)
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-6 restore@768k -> %f want %f\n",
                     dst.getTargetIRLength(), kSR01HardMax768kF);
        return false;
    }
    const auto tree768 = dst.getState();      // irLength = 2.7306667
    ConvolverProcessor back;
    back.prepareToPlay(48000.0, 512);
    back.setState(tree768);
    if (back.getTargetIRLength() != kSR01HardMax768kF)
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-6 re-restore@48k %f (want 2.7306667, no lift)\n",
                     back.getTargetIRLength());
        return false;
    }
    std::printf("checkSR01StateCompatibility: PASS (persist window / hardMax on restore / no lift)\n");
    return true;
}

// T-SR01-7: ロード失敗（SilentIR・別 SR の resample 経路）で旧 engine 保持・publish 単一性。
//   NUC malloc 失敗そのものは注入点を持たない（凍結契約の「注入可なら」条件による。
//   本ケースは load 破棄経路全体の回帰として旧 engine 保持と finalized 維持を検証する）。
bool checkSR01LoadFailureKeepsEngine()
{
    SR01LoggerScope scope;
    ConvolverProcessor conv;
    conv.prepareToPlay(48000.0, 512);
    const juce::File ok = writeSR01TempIr("sr01_7a.wav", 48000.0, 2, 4800, 100);
    if (!ok.existsAsFile() || !conv.loadImpulseResponse(ok, false))
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-7 baseline load rejected\n");
        return false;
    }
    // 基準ロード確定を待って irLength を記録（peak==100 と同一 commit で publish 済み）
    {
        bool ready = false;
        for (int i = 0; i < 4000; ++i)
        {
            const auto bd = conv.getLatencyBreakdown();
            if (bd.irPeakLatencySamples == 100 && conv.getIRLength() > 0)
            {
                ready = true;
                break;
            }
            pumpH01Messages();
            juce::Thread::sleep(5);
        }
        if (!ready)
        {
            std::fprintf(stderr, "[SR01] FAIL: T-SR01-7 baseline never finalized\n");
            ok.deleteFile();
            return false;
        }
    }
    const int lenBefore = conv.getIRLength();
    ok.deleteFile();
    const juce::File silent = writeSR01TempIr("sr01_7b.wav", 44100.0, 2, 4410, -1); // 別SR → resample 経路
    if (!silent.existsAsFile())
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-7 silent IR write failed\n");
        return false;
    }
    conv.loadImpulseResponse(silent, false);
    bool failed = false;
    for (int i = 0; i < 4000; ++i)
    {
        // silence-trim → resample 経路の順序依存で "silent" / "Resampling failed" のいずれかで終わる。
        // 本契約の核心は「失敗時に旧 engine が保持されること」（publish 単一性）であり、
        // 文言のどちらでもよい。ローディング完了＋非空 error を破棄判定の必要条件とする。
        const auto err = conv.getLastError();
        if (!conv.isLoadingIR() && err.isNotEmpty()
            && (err.containsIgnoreCase("silent") || err.containsIgnoreCase("resampl")))
        {
            failed = true;
            break;
        }
        pumpH01Messages();
        juce::Thread::sleep(5);
    }
    const bool kept = failed && conv.getIRLength() == lenBefore && conv.isIRFinalized();
    silent.deleteFile();
    if (!kept)
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-7 failed=%d irLen %d->%d finalized=%d\n",
                     failed ? 1 : 0, lenBefore, conv.getIRLength(), conv.isIRFinalized() ? 1 : 0);
        return false;
    }
    std::printf("checkSR01LoadFailureKeepsEngine: PASS (SilentIR rejected, old engine intact)\n");
    return true;
}

// T-SR01-8: mono IR → ch0==ch1 複製契約（wet 到着位置の L/R 一致・H-01 基準との非抵触確認）
bool checkSR01MonoDuplication()
{
    ConvolverProcessor conv;
    conv.prepareToPlay(48000.0, 512);
    conv.setTargetIRLength(1.0f);
    const juce::File mono = writeSR01TempIr("sr01_8.wav", 48000.0, 1, 4800, 100);
    if (!mono.existsAsFile())
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-8 mono IR write failed\n");
        return false;
    }
    conv.loadImpulseResponse(mono, false);
    const bool loaded = pollSR01Load(conv, 100, 48000);   // int(48000 × 1.0s)
    mono.deleteFile();
    if (!loaded)
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-8 mono load not finalized\n");
        return false;
    }
    conv.setMix(1.0f);
    juce::AudioBuffer<double> ab(2, kH01Block);
    juce::dsp::AudioBlock<double> blk(ab);
    for (int i = 0; i < kH01WarmupBlocks; ++i)
    {
        ab.clear();
        conv.process(blk);
    }
    int idxL = -1, idxR = -1;
    double maxL = 0.0, maxR = 0.0;
    constexpr int kSR01CaptureBlocks = 16;
    for (int b = 0; b < kSR01CaptureBlocks; ++b)
    {
        ab.clear();
        if (b == 0)
        {
            ab.getWritePointer(0)[0] = 1.0;
            ab.getWritePointer(1)[0] = 1.0;
        }
        conv.process(blk);
        for (int i = 0; i < ab.getNumSamples(); ++i)
        {
            const double l = std::fabs(ab.getReadPointer(0)[i]);
            const double r = std::fabs(ab.getReadPointer(1)[i]);
            if (l > maxL) { maxL = l; idxL = (b + kH01WarmupBlocks) * kH01Block + i; }
            if (r > maxR) { maxR = r; idxR = (b + kH01WarmupBlocks) * kH01Block + i; }
        }
    }
    const int want = kH01ImpulseBase + 100;
    if (idxL != want || idxR != want || maxL <= 0.0 || maxR <= 0.0)
    {
        std::fprintf(stderr, "[SR01] FAIL: T-SR01-8 mono L=%d R=%d want=%d\n", idxL, idxR, want);
        return false;
    }
    std::printf("checkSR01MonoDuplication: PASS (L/R argmax == %d == impulse+peak)\n", want);
    return true;
}

// ============================================================================
// ★ M-04 (T-M04-1..5): oversized block deterministic containment（entry gate）
//   根拠: doc/work97/m04_pre_audit_failure_contract_freeze_20260915.md
//   契約: numSamples > MAX_BLOCK_SIZE の入力は ring/retarget/smoother/crossfade/NUC を
//   一切進行させず決定的無音 + telemetry。bypass 含む全経路共通（cond-3）。
//   重大度訂正: Medium-High(stale) → Low-Medium(latent hardening / OOB-prevention)。
// ============================================================================

// private telemetry 読み取りと reporter 駆動専用の test シームは
// ファイル先頭 global スコープの M04OversizedTestAccess（friend 探索名前空間一致のため）


bool checkM04OversizedContainment()
{
    static juce::ScopedJuceInitialiser_GUI m04Init;
    struct M04Logger : juce::Logger {
        juce::String captured;
        void logMessage(const juce::String& m) override
        {
            if (m.contains("M-04 oversized block containment"))
                captured += m + "\n";
        }
    };
    M04Logger m04Log;
    juce::Logger* prevLogger = juce::Logger::getCurrentLogger();

    auto makeInput = [](int blockIdx) {
        juce::AudioBuffer<double> ab(2, 512);
        ab.clear();
        const int p = 37 + blockIdx * 97;            // 決定的パターン
        // ★ ch0 側 %512 化（work98 §8-9: 無 modulo は buf 512 超えで OOB 書込 = heap corruption。
        //   ch1 は元から %512。T-M04 の意味（決定的パターン注入）は不変）
        ab.getWritePointer(0)[p % 512] = 1.0;
        ab.getWritePointer(1)[(p + 13) % 512] = 0.5;
        return ab;
    };

    ConvolverProcessor control;
    ConvolverProcessor gated;
    control.prepareToPlay(48000.0, 512);
    gated.prepareToPlay(48000.0, 512);
    control.setTargetIRLength(1.0f);
    gated.setTargetIRLength(1.0f);

    const juce::File ir = writeSR01TempIr("m04_ir.wav", 48000.0, 2, 4800, 100);
    if (!ir.existsAsFile())
    {
        std::fprintf(stderr, "[M-04] FAIL: IR write failed\n");
        return false;
    }
    control.loadImpulseResponse(ir, false);
    gated.loadImpulseResponse(ir, false);
    const bool bothLoaded = pollSR01Load(control, 100, 48000) && pollSR01Load(gated, 100, 48000);
    ir.deleteFile();
    if (!bothLoaded)
    {
        std::fprintf(stderr, "[M-04] FAIL: IR load not finalized\n");
        return false;
    }

    control.setMix(0.5f);
    gated.setMix(0.5f);   // 同一 ramp 状態から開始

    // --- T-M04-1: oversized (524,289) → 決定的無音 + counter+1（ramp 中） ---
    const int cnt0 = M04OversizedTestAccess::count();
    bool ok = true;
    {
        juce::AudioBuffer<double> big(2, ConvolverProcessor::MAX_BLOCK_SIZE + 1);
        big.clear();
        for (int ch = 0; ch < 2; ++ch)
            juce::FloatVectorOperations::fill(big.getWritePointer(ch), 0.25, big.getNumSamples());
        juce::dsp::AudioBlock<double> bigBlock(big);
        gated.process(bigBlock);
        double maxAbs = 0.0;
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < big.getNumSamples(); i += 4096)
                maxAbs = std::max(maxAbs, std::fabs(big.getReadPointer(ch)[i]));
        if (maxAbs != 0.0)
        {
            std::fprintf(stderr, "[M-04] FAIL: T-M04-1 oversized block not silent\n");
            ok = false;
        }
        if (M04OversizedTestAccess::count() != cnt0 + 1)
        {
            std::fprintf(stderr, "[M-04] FAIL: T-M04-1 counter delta != 1\n");
            ok = false;
        }
    }

    // --- T-M04-2: 無状態性 — 対照系と正常 block 出力がビット一致 ---
    juce::AudioBuffer<double> outC[8], outT[8];
    for (int j = 0; j < 8; ++j)
    {
        juce::AudioBuffer<double> ab = makeInput(j);
        {
            juce::dsp::AudioBlock<double> blk(ab);
            control.process(blk);
        }
        outC[j] = ab;

        // gated: 4 ブロック目に追加で oversized を挟む（状態を進めないことを検証）
        if (j == 4)
        {
            juce::AudioBuffer<double> big2(2, ConvolverProcessor::MAX_BLOCK_SIZE + 2);
            big2.clear();
            juce::dsp::AudioBlock<double> big2Block(big2);
            gated.process(big2Block);
        }
        juce::AudioBuffer<double> ab2 = makeInput(j);
        {
            juce::dsp::AudioBlock<double> blk(ab2);
            gated.process(blk);
        }
        outT[j] = ab2;
    }
    for (int j = 0; j < 8; ++j)
    {
        for (int ch = 0; ch < 2; ++ch)
        {
            if (std::memcmp(outC[j].getReadPointer(ch), outT[j].getReadPointer(ch),
                            sizeof(double) * outC[j].getNumSamples()) != 0)
            {
                std::fprintf(stderr, "[M-04] FAIL: T-M04-2 block %d ch %d diverged\n", j, ch);
                ok = false;
            }
        }
    }

    // --- T-M04-3: bypass でも同一 entry contract（silence + telemetry） ---
    {
        gated.setBypass(true);
        const int cntB = M04OversizedTestAccess::count();
        juce::AudioBuffer<double> big3(2, ConvolverProcessor::MAX_BLOCK_SIZE + 1);
        for (int ch = 0; ch < 2; ++ch)
            juce::FloatVectorOperations::fill(big3.getWritePointer(ch), 0.75, big3.getNumSamples());
        juce::dsp::AudioBlock<double> big3Block(big3);
        gated.process(big3Block);
        double maxAbs = 0.0;
        for (int ch = 0; ch < 2; ++ch)
            maxAbs = std::max(maxAbs, std::fabs(big3.getReadPointer(ch)[big3.getNumSamples() / 2]));
        if (maxAbs != 0.0 || M04OversizedTestAccess::count() != cntB + 1)
        {
            std::fprintf(stderr, "[M-04] FAIL: T-M04-3 bypass oversized not contained\n");
            ok = false;
        }
        gated.setBypass(false);
    }

    // --- T-M04-5: NonRT reporter（timerCallback 経由・ヒステリシス確認） ---
    {
        juce::Logger::setCurrentLogger(&m04Log);
        M04OversizedTestAccess::pumpReport(gated);          // delta あり → ログ 1 件
        const int logs1 = m04Log.captured.contains("M-04") ? 1 : 0;
        M04OversizedTestAccess::pumpReport(gated);          // delta なし → 追加ログなし（ヒステリシス）
        juce::Logger::setCurrentLogger(prevLogger);
        const int nlFirst = m04Log.captured.indexOf("\n");
        const int nlLast  = m04Log.captured.lastIndexOf("\n");
        if (logs1 == 0 || nlFirst != nlLast)   // ちょうど 1 行でなければならない
        {
            std::fprintf(stderr, "[M-04] FAIL: T-M04-5 reporter log/hysteresis: '%s'\n",
                         m04Log.captured.toRawUTF8());
            ok = false;
        }
    }

    // --- T-M04-4: 境界証明 numSamples == MAX_BLOCK_SIZE は gate 不発（`>` semantics） ---
    {
        const int cntE = M04OversizedTestAccess::count();
        juce::AudioBuffer<double> exact(2, ConvolverProcessor::MAX_BLOCK_SIZE);
        exact.clear();
        exact.getWritePointer(0)[1000] = 1.0;
        exact.getWritePointer(1)[1000] = 1.0;
        gated.setMix(1.0f);
        {
            juce::dsp::AudioBlock<double> blk(exact);
            gated.process(blk);   // 524,288 ちょうど: 以後の capacity guard も通過（==は有効域）
        }
        double energy = 0.0;
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < exact.getNumSamples(); i += 256)
                energy += std::fabs(exact.getReadPointer(ch)[i]);
        if (energy <= 0.0)
        {
            std::fprintf(stderr, "[M-04] FAIL: T-M04-4 boundary block produced no output (gate mis-fire at ==?)\n");
            ok = false;
        }
        if (M04OversizedTestAccess::count() != cntE)
        {
            std::fprintf(stderr, "[M-04] FAIL: T-M04-4 boundary block counted by gate\n");
            ok = false;
        }
    }

    if (!ok)
        return false;
    std::printf("checkM04OversizedContainment: PASS (T-M04-1..5: silence+telemetry / bit-identical recovery / bypass gate / reporter / == boundary not gated)\n");
    return true;
}

// ============================================================================
// ★ M-02 (T-M02-1..5): wet scrub telemetry（C-1 のみ・D-1=(a) telemetry-only）
//   根拠: doc/work98/m02_pre_audit_failure_contract_freeze_20260915.md
//   単位契約（§4 C-1 凍結）: counter 増分 = scrub 発火 chunk 数（+1/発火 chunk）。
//     sanitizeFiniteChunk の返却「置換サンプル数」は検出判定専用であり counter 単位ではない。
//   recovery: 受動的自己失効。実測失効窓 = burst 終了から約 100 blocks（doc/work98 §8-12。
//     本 commit の drain 上限 512 blocks ≫ 窓。失効後 16 blocks 連続一致で安定を判定する）。
//     Reset/re-init/pending は対象外（D-1=(a) 承認・C-2(c) 却下）。
//   トリガー設計:
//   - 定常 mix=1.0（dryG=0）→ 最終出力 = scrub 適用 wet。burst 中 per-block finiteness は
//     wet スクラブの保証として主張（dry リング汚染は §2-(a) スコープ外・主張しない）。
//   - 注入は quiet_NaN のみ（±Inf は Debug L0 killDenormalV を通過し持続が非決定化・§8-11）。
//   - getWritePointer 計算 index は必ず %512（OOB=heap corruption の教訓・§8-9）。
// ============================================================================

bool checkM02NonFiniteTelemetry()
{
    static juce::ScopedJuceInitialiser_GUI m02Init;
    struct M02Logger : juce::Logger {
        juce::String captured;
        void logMessage(const juce::String& m) override
        {
            if (m.contains("M-02 non-finite wet scrub"))
                captured += m + "\n";
        }
    };
    M02Logger m02Log;
    juce::Logger* prevLogger = juce::Logger::getCurrentLogger();

    auto makeClean = [](int blockIdx) {
        juce::AudioBuffer<double> ab(2, 512);
        ab.clear();
        const int p = 11 + blockIdx * 61;
        ab.getWritePointer(0)[p % 512] = 0.25;          // ★ %512 必須（§8-9）
        ab.getWritePointer(1)[(p + 17) % 512] = -0.125;
        return ab;
    };
    auto injectNaN = [](juce::AudioBuffer<double>& ab) {
        ab.getWritePointer(0)[3] = std::numeric_limits<double>::quiet_NaN();
        ab.getWritePointer(1)[7] = std::numeric_limits<double>::quiet_NaN();
    };
    auto allFinite = [](const juce::AudioBuffer<double>& ab) {
        for (int ch = 0; ch < ab.getNumChannels(); ++ch)
            for (int i = 0; i < ab.getNumSamples(); ++i)
                if (!std::isfinite(ab.getReadPointer(ch)[i])) return false;
        return true;
    };
    auto blocksEqual = [](const juce::AudioBuffer<double>& x, const juce::AudioBuffer<double>& y) {
        for (int ch = 0; ch < x.getNumChannels(); ++ch)
            if (std::memcmp(x.getReadPointer(ch), y.getReadPointer(ch),
                            sizeof(double) * static_cast<size_t>(x.getNumSamples())) != 0)
                return false;
        return true;
    };
    auto processBlock = [](ConvolverProcessor& cp, juce::AudioBuffer<double>& ab) {
        juce::dsp::AudioBlock<double> blk(ab);
        cp.process(blk);
    };

    ConvolverProcessor control;
    ConvolverProcessor gated;
    control.prepareToPlay(48000.0, 512);
    gated.prepareToPlay(48000.0, 512);
    control.setTargetIRLength(1.0f);
    gated.setTargetIRLength(1.0f);

    const juce::File ir = writeSR01TempIr("m02_ir.wav", 48000.0, 2, 4800, 100);
    if (!ir.existsAsFile())
    {
        std::fprintf(stderr, "[M-02] FAIL: IR write failed\n");
        control.releaseResources();
        gated.releaseResources();
        return false;
    }
    control.loadImpulseResponse(ir, false);
    gated.loadImpulseResponse(ir, false);
    const bool bothLoaded = pollSR01Load(control, 100, 48000) && pollSR01Load(gated, 100, 48000);
    ir.deleteFile();
    if (!bothLoaded)
    {
        std::fprintf(stderr, "[M-02] FAIL: IR load not finalized\n");
        control.releaseResources();
        gated.releaseResources();
        return false;
    }
    control.setMix(1.0f);
    gated.setMix(1.0f);   // 定常時 dryG=0 → 最終出力 = scrub 適用 wet

    bool ok = true;

    // --- T-M02-3: 検証済ロード鎖の無発火 + warmup 40 blocks（mix ramp 収束・両系一致基線）---
    const int cntPre = M02NonFiniteTestAccess::count();
    int equalRunWarm = 0;
    for (int j = 0; j < 40; ++j)
    {
        juce::AudioBuffer<double> a = makeClean(j);
        juce::AudioBuffer<double> b = makeClean(j);
        processBlock(control, a);
        processBlock(gated, b);
        if (j >= 32)
        {
            if (!allFinite(b))
            {
                std::fprintf(stderr, "[M-02] FAIL: T-M02-3 warmup block %d not finite\n", j);
                ok = false;
            }
            equalRunWarm = blocksEqual(a, b) ? equalRunWarm + 1 : 0;
        }
    }
    if (M02NonFiniteTestAccess::count() != cntPre)
    {
        std::fprintf(stderr, "[M-02] FAIL: T-M02-3 load/warmup counter delta != 0\n");
        ok = false;
    }
    if (equalRunWarm < 4)
    {
        std::fprintf(stderr, "[M-02] FAIL: T-M02-3 baseline control!=gated before burst (equalRun=%d)\n", equalRunWarm);
        ok = false;
    }

    // --- T-M02-1: NaN burst 8 blocks（gated のみ、clean 入力との差分注入）---
    //   burst 中の dst 最終ブロックの finiteness は主張しない: 注入 NaN は dry リング
    //   （delayBuffer 生入力写像）へも供給され、mixSteadySmall の `dry[i]*dryG` は
    //   dryG=0.0 でも IEEE 上 NaN を生成する（0*NaN=NaN・doc/work98 §8-14/§2-(a)）。
    //   wet スクラブ（C-1）の有限性証明は drain ループ（入力=dry 洁净）で担保する。
    for (int j = 0; j < 8; ++j)
    {
        juce::AudioBuffer<double> a = makeClean(40 + j);
        juce::AudioBuffer<double> b = makeClean(40 + j);
        injectNaN(b);
        processBlock(control, a);
        processBlock(gated, b);
    }

    // --- T-M02-2: 有界受動失効 recovery — clean 入力で対照系と連続 16 blocks ビット一致まで drain ---
    //   失効窓実測 ≒ 100 blocks（doc/work98 §8-12）。cap 512 で非回復（=永久ミュート）を検出する。
    //   各 drain block の allFinite = wetOut が非有限でも dst 有限（= scrub 動作の直接証明、
    //   発火継続区間 48..147 を含む）。
    int stableRun = 0;
    int drained = 0;
    for (; drained < 512 && stableRun < 16; ++drained)
    {
        const int idx = 48 + drained;
        juce::AudioBuffer<double> a = makeClean(idx);
        juce::AudioBuffer<double> b = makeClean(idx);
        processBlock(control, a);
        processBlock(gated, b);
        if (!allFinite(b))
        {
            std::fprintf(stderr, "[M-02] FAIL: T-M02-2 drain block %d not finite\n", idx);
            ok = false;
        }
        stableRun = blocksEqual(a, b) ? stableRun + 1 : 0;
    }
    if (stableRun < 16)
    {
        std::fprintf(stderr, "[M-02] FAIL: T-M02-2 recovery not stable within cap (drained=%d stableRun=%d)\n",
                     drained, stableRun);
        ok = false;
    }
    const int cnt1 = M02NonFiniteTestAccess::count();
    if (cnt1 <= cntPre)
    {
        std::fprintf(stderr, "[M-02] FAIL: T-M02-1 counter not fired (pre=%d post=%d)\n", cntPre, cnt1);
        ok = false;
    }

    // --- T-M02-5: 復旧後の正常 block は scrub 無発火（誤発火防止） ---
    {
        const int cntR = M02NonFiniteTestAccess::count();
        for (int j = 0; j < 8; ++j)
        {
            juce::AudioBuffer<double> b = makeClean(600 + j);
            processBlock(gated, b);
        }
        if (M02NonFiniteTestAccess::count() != cntR)
        {
            std::fprintf(stderr, "[M-02] FAIL: T-M02-5 clean blocks after recovery fired scrub\n");
            ok = false;
        }
    }

    // --- T-M02-4: NonRT reporter（timerCallback・pump ×2 でちょうど 1 行 = ヒステリシス） ---
    {
        juce::Logger::setCurrentLogger(&m02Log);
        M02NonFiniteTestAccess::pumpReport(gated);   // delta あり → 1 行
        M02NonFiniteTestAccess::pumpReport(gated);   // delta なし → 追加なし
        juce::Logger::setCurrentLogger(prevLogger);
        const int nlFirst = m02Log.captured.indexOf("\n");
        const int nlLast  = m02Log.captured.lastIndexOf("\n");
        if (m02Log.captured.isEmpty() || nlFirst != nlLast)
        {
            std::fprintf(stderr, "[M-02] FAIL: T-M02-4 reporter log/hysteresis: '%s'\n",
                         m02Log.captured.toRawUTF8());
            ok = false;
        }
    }

    // 後片付け: LoaderThread/retire を停止（テスト間干渉・非決定性排除）
    control.releaseResources();
    gated.releaseResources();

    if (!ok)
        return false;
    std::printf("checkM02NonFiniteTelemetry: PASS (T-M02-1..5: load-clean baseline / burst->fired / drain finiteness+bounded aging recovery / reporter hysteresis / no false fire)\n");
    return true;
}

// ============================================================================
// ★ M-01 (T-M01-1/2): L1/L2 pre-IFFT denormal hygiene（D-M01=β scalar killDenormal）
//   根拠: doc/work99/m01_pre_audit_failure_contract_freeze_20260915.md
//   契約の二層検証:
//   (a) primitive 層（T-M01-1a・観察可能・vacuous でない）:
//       guard 本体と同一の #if 条件で、Release では恒等性（=命令増 0 の構造保証 T-M01-4）、
//       Debug では subnormal→0・normal/NaN/Inf/±0 保持（=M-02 責務分離 G-M01-4）。
//   (b) DSP 層（T-M01-1b・契約回帰ロック）: L1 活性幾何へ subnormal 専用入力を注入し、
//       全出力サンプルの不変式「x == 0.0 || |x| >= DBL_MIN」を両ビルドで主張。
//       ※ 正直な位置づけ（work99 §8-1）: RT 実行時は ScopedNoDenormals（process() :288）と
//       MainApplication の per-thread FTZ/DAZ 設定により、guard 追加前から HW が subnormal
//       演算結果を flush しているため、この不変式は pre-guard でも成立する（=バグ再現でなく
//       契約の固定。guard は software 層の defense-in-depth）。finiteness への弱体化は禁止
//       （ユーザー指示）— 厳密な 0/DBL_MIN 判定を維持する。
//   T-M01-2（正常信号 long-run）は (b) の clean 区間 + warmup で同一不変式を検証。
//   T-M01-3（M-02 非退行）は本 binary の checkM02NonFiniteTelemetry（両構成 ctest）で強制、
//   T-M01-4（Release 等価）は (a) Release 分岐 + semantic-hash verifier 群（PUSH GATE）で担保。
// ============================================================================

bool checkM01DenormalHygiene()
{
    static juce::ScopedJuceInitialiser_GUI m01Init;
    bool ok = true;

    // --- (a) primitive 契約（guard 本体と同一のビルド条件） ---
    {
        constexpr double kSub  = 1.0e-310;   // 真の IEEE subnormal
        constexpr double kNorm = 1.0e-300;   // normal（subnormal 境界 DBL_MIN ≒2.2e-308 より上）
        const double kNaN = std::numeric_limits<double>::quiet_NaN();
        const double kInf = std::numeric_limits<double>::infinity();
        // リテラルが実際に subnormal であることの前提検証（ツールチェーン依存排除）
        if (!(kSub != 0.0 && std::fabs(kSub) < DBL_MIN))
        {
            std::fprintf(stderr, "[M-01] FAIL: preflight subnormal literal invalid\n");
            return false;
        }
#if !defined(JUCE_DEBUG) && !defined(_DEBUG) && !defined(CONVOPEQ_DEBUG_DENORMALS)
        // Release: guard はコンパイル時 no-op（T-M01-4 構造保証）
        if (killDenormal(kSub) != kSub) { std::fprintf(stderr, "[M-01] FAIL: T-M01-1a release not identity\n"); ok = false; }
        if (killDenormal(kNorm) != kNorm) { ok = false; std::fprintf(stderr, "[M-01] FAIL: T-M01-1a release normal altered\n"); }
        if (!std::isnan(killDenormal(kNaN))) { ok = false; std::fprintf(stderr, "[M-01] FAIL: T-M01-1a release NaN touched\n"); }
        if (killDenormal(kInf) != kInf) { ok = false; std::fprintf(stderr, "[M-01] FAIL: T-M01-1a release Inf touched\n"); }
#else
        // Debug: 厳密 subnormal のみ 0 化。normal/NaN/Inf/±0 は保持（責務分離）
        if (killDenormal(kSub) != 0.0) { std::fprintf(stderr, "[M-01] FAIL: T-M01-1a debug subnormal not flushed\n"); ok = false; }
        if (killDenormal(-kSub) != 0.0) { std::fprintf(stderr, "[M-01] FAIL: T-M01-1a debug neg subnormal not flushed\n"); ok = false; }
        if (killDenormal(kNorm) != kNorm) { std::fprintf(stderr, "[M-01] FAIL: T-M01-1a debug normal altered\n"); ok = false; }
        if (!std::isnan(killDenormal(kNaN))) { std::fprintf(stderr, "[M-01] FAIL: T-M01-1a debug NaN not preserved\n"); ok = false; }
        if (killDenormal(kInf) != kInf) { std::fprintf(stderr, "[M-01] FAIL: T-M01-1a debug Inf not preserved\n"); ok = false; }
        if (killDenormal(0.0) != 0.0 || std::signbit(killDenormal(-0.0)) == false)
        { std::fprintf(stderr, "[M-01] FAIL: T-M01-1a debug signed zero altered\n"); ok = false; }
#endif
    }

    // --- (b) DSP 契約回帰ロック: L1 活性幾何（4800/512/48k → l1Len=704）+ subnormal 注入 ---
    auto makePattern = [](int blockIdx) {
        juce::AudioBuffer<double> ab(2, 512);
        ab.clear();
        const int p = 11 + blockIdx * 61;
        ab.getWritePointer(0)[p % 512] = 0.25;
        ab.getWritePointer(1)[(p + 17) % 512] = -0.125;
        return ab;
    };
    // 契約判定（厳密）: 全サンプルが有限 かつ「0 または |x| >= DBL_MIN」
    auto contractHolds = [](const juce::AudioBuffer<double>& ab) {
        for (int ch = 0; ch < ab.getNumChannels(); ++ch)
            for (int i = 0; i < ab.getNumSamples(); ++i)
            {
                const double x = ab.getReadPointer(ch)[i];
                if (!std::isfinite(x)) return false;
                if (x != 0.0 && std::fabs(x) < DBL_MIN) return false;   // subnormal 出現 = 契約違反
            }
        return true;
    };
    auto energy = [](const juce::AudioBuffer<double>& ab) {
        double e = 0.0;
        for (int ch = 0; ch < ab.getNumChannels(); ++ch)
            for (int i = 0; i < ab.getNumSamples(); ++i)
                e += std::fabs(ab.getReadPointer(ch)[i]);
        return e;
    };

    ConvolverProcessor cp;
    cp.prepareToPlay(48000.0, 512);
    cp.setTargetIRLength(1.0f);
    const juce::File ir = writeSR01TempIr("m01_ir.wav", 48000.0, 2, 4800, 100);
    if (!ir.existsAsFile())
    {
        std::fprintf(stderr, "[M-01] FAIL: IR write failed\n");
        cp.releaseResources();
        return false;
    }
    cp.loadImpulseResponse(ir, false);
    const bool loaded = pollSR01Load(cp, 100, 48000);
    ir.deleteFile();
    if (!loaded)
    {
        std::fprintf(stderr, "[M-01] FAIL: IR load not finalized\n");
        cp.releaseResources();
        return false;
    }
    cp.setMix(1.0f);

    auto processBlock = [&cp](juce::AudioBuffer<double>& ab) {
        juce::dsp::AudioBlock<double> blk(ab);
        cp.process(blk);
    };

    // warmup（ramp 収束）+ T-M01-2 前半: 正常信号で不変式・非零応答確認
    double warmEnergy = 0.0;
    for (int j = 0; j < 16; ++j)
    {
        juce::AudioBuffer<double> a = makePattern(j);
        processBlock(a);
        if (!contractHolds(a))
        {
            std::fprintf(stderr, "[M-01] FAIL: T-M01-2 warmup block %d subnormal in output\n", j);
            ok = false;
        }
        warmEnergy += energy(a);
    }
    if (warmEnergy <= 0.0)   // 空証防止: engine が実際に応答していること
    {
        std::fprintf(stderr, "[M-01] FAIL: warmup produced no output (vacuous)\n");
        ok = false;
    }

    // T-M01-1b: subnormal 専用 impulse を 8 blocks 注入（L1 part ちょうど）→ 40 blocks drain
    for (int j = 0; j < 8; ++j)
    {
        juce::AudioBuffer<double> a = makePattern(16 + j);
        a.getWritePointer(0)[4] = 1.0e-310;
        a.getWritePointer(1)[60] = -1.0e-311;
        processBlock(a);
        if (!contractHolds(a))
        {
            std::fprintf(stderr, "[M-01] FAIL: T-M01-1b inject block %d contract violated\n", j);
            ok = false;
        }
    }
    for (int j = 0; j < 40; ++j)
    {
        juce::AudioBuffer<double> a = makePattern(24 + j);
        processBlock(a);
        if (!contractHolds(a))
        {
            std::fprintf(stderr, "[M-01] FAIL: T-M01-1b drain block %d contract violated\n", 24 + j);
            ok = false;
        }
    }

    // T-M01-2 後半: 長尺 clean 運転で不変式継続（guard が正常信号の意味論を変えない回帰側）
    for (int j = 0; j < 32; ++j)
    {
        juce::AudioBuffer<double> a = makePattern(64 + j);
        processBlock(a);
        if (!contractHolds(a))
        {
            std::fprintf(stderr, "[M-01] FAIL: T-M01-2 clean block %d contract violated\n", 64 + j);
            ok = false;
        }
    }

    cp.releaseResources();

    if (!ok)
        return false;
    std::printf("checkM01DenormalHygiene: PASS (T-M01-1: primitive contract + subnormal-injection invariant lock / T-M01-2: clean long-run invariant)\n");
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
    // ★ H-02 T1-T4（helper 直接・新 CTest 登録なし）
    if (!checkH02T1ExpPeakSample0())
    {
        std::fprintf(stderr, "FAIL: checkH02T1ExpPeakSample0\n");
        return 1;
    }
    if (!checkH02T2MultiChannel())
    {
        std::fprintf(stderr, "FAIL: checkH02T2MultiChannel\n");
        return 1;
    }
    if (!checkH02T3TieBreakLowestIndex())
    {
        std::fprintf(stderr, "FAIL: checkH02T3TieBreakLowestIndex\n");
        return 1;
    }
    if (!checkH02T4NonFiniteAndBoundaries())
    {
        std::fprintf(stderr, "FAIL: checkH02T4NonFiniteAndBoundaries\n");
        return 1;
    }
    // ★ SR-03 T-SR03-1〜6（publish 単一関口 clamp / telemetry / RT bounded 消費）
    if (!checkSR03LatencyDelayClamp())
    {
        std::fprintf(stderr, "FAIL: checkSR03LatencyDelayClamp\n");
        return 1;
    }
    // ★ H-01 T-H01-1..6（internal dry alignment = irPeak／host PDC 不変）
    if (!checkH01DryWetAlignment())
    {
        std::fprintf(stderr, "FAIL: checkH01DryWetAlignment\n");
        return 1;
    }
    // ★ SR-01(B) T-SR01-1..8（SR 依存 hardMax・C-1 snapshot clamp・C-3 trim ログ・互換回帰）
    if (!checkSR01HardMaxFormula())
    {
        std::fprintf(stderr, "FAIL: checkSR01HardMaxFormula\n");
        return 1;
    }
    if (!checkSR01SetterClamp())
    {
        std::fprintf(stderr, "FAIL: checkSR01SetterClamp\n");
        return 1;
    }
    if (!checkSR01SnapshotClamp())
    {
        std::fprintf(stderr, "FAIL: checkSR01SnapshotClamp\n");
        return 1;
    }
    if (!checkSR01Load768k3s())
    {
        std::fprintf(stderr, "FAIL: checkSR01Load768k3s\n");
        return 1;
    }
    if (!checkSR01CapFiredLog())
    {
        std::fprintf(stderr, "FAIL: checkSR01CapFiredLog\n");
        return 1;
    }
    if (!checkSR01LowRateRegression())
    {
        std::fprintf(stderr, "FAIL: checkSR01LowRateRegression\n");
        return 1;
    }
    if (!checkSR01StateCompatibility())
    {
        std::fprintf(stderr, "FAIL: checkSR01StateCompatibility\n");
        return 1;
    }
    if (!checkSR01LoadFailureKeepsEngine())
    {
        std::fprintf(stderr, "FAIL: checkSR01LoadFailureKeepsEngine\n");
        return 1;
    }
    if (!checkSR01MonoDuplication())
    {
        std::fprintf(stderr, "FAIL: checkSR01MonoDuplication\n");
        return 1;
    }
    // ★ M-04 T-M04-1..5（oversized deterministic containment / 無状態性 / reporter / 境界 >）
    if (!checkM04OversizedContainment())
    {
        std::fprintf(stderr, "FAIL: checkM04OversizedContainment\n");
        return 1;
    }
    // ★ M-02 T-M02-1..5（wet scrub telemetry / 単位契約 / reporter ヒステリシス / 受動失効 recovery）
    if (!checkM02NonFiniteTelemetry())
    {
        std::fprintf(stderr, "FAIL: checkM02NonFiniteTelemetry\n");
        return 1;
    }
    // ★ M-01 T-M01-1/2（pre-IFFT denormal hygiene・T-M01-3 は本 M-02 チェックの両構成 PASS で強制、
    //   T-M01-4 は Release 分岐 + semantic verifier 群で担保）
    if (!checkM01DenormalHygiene())
    {
        std::fprintf(stderr, "FAIL: checkM01DenormalHygiene\n");
        return 1;
    }
    return 0;
}
