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

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
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
    return 0;
}
