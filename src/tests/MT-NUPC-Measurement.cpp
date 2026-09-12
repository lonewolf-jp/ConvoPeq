// MT-NUPC-Measurement.cpp
// Gardner Null Test v2.9 (Step 3) — NUPC レイヤー間遅延アライメント測定
//
// 仕様: doc/work57/null_test_procedure_v2.md (v2.9)
//   M1: layerGain 反映 reference との Null Test（波形の正しさ）
//   M2: 三時刻観測（content_time / t_write_callback / t_output_callback）
//       → outputPlacementError（時間軸の正しさ・**主判定**）
//   M3: インパルス応答の L1/L2 成分位置同定（独立補助証拠）
//
// 判定帯（§6）:
//   outputPlacementError == 0      → B13 正当（**唯一の合格条件**）
//   1 ≤ |error| ≤ 64               → 要精査（合格ではない）
//   |error| ≥ 128                  → B13 補償不成立
//
// 実装上の規約:
//   - outputDelaySamples は **assert しない**（測定対象。startup log のみ）
//   - Get() には必ず有効な出力バッファを渡す（Get(nullptr) は L1/L2 読み出しをスキップ）
//   - M2 は impulse 位置 n0 ごとに独立 run（M1 差分のクリーン化・M3 ピーク分離・CSV n0 帰属のため。
//     oPE 本体は write/read スケジュールのみで決まり重ね合わせに非依存）
//   - exit code は構造的健全性（SetImpulse 成功・event coverage=1.0・CSV 書き込み成功）のみを反映。
//     oPE 値（B13 正当/不成立）は Step 4 の解析対象であり exit code に影響させない。
//   - 出力は std::cout / std::ofstream（型安全ストリーム）に統一
//
// ビルド: CMake ターゲット MTNUPCMeasurement（CONVOPEQ_ENABLE_ISR_TESTS）
// 依存: MKL, JUCE（JuceHeader 経由）, NUPCTestAccess.h（friend 読み取りのみ）

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <filesystem>

#include <mkl_dfti.h>

#include "MKLNonUniformConvolver.h"
#include "NUPCTestAccess.h"      // friend 読み取りのみ（同一ディレクトリ）
#include "audioengine/AtomicAccess.h"
#include "DspNumericPolicy.h"

namespace {

constexpr int    kBlockSize      = 64;
constexpr double kSampleRate     = 48000.0;
constexpr int    kL0MaxParts     = 32;   // MKLNonUniformConvolver.h
constexpr int    kL1MaxParts     = 64;
constexpr int    kTailL1L2Mult   = 8;    // filterSpec=nullptr / FilterSpec デフォルト
constexpr int    kMaxBlocksTrack = 512;  // run ごとに event を追跡するブロック数上限

// ── 層構成の理論値（48 kHz / blockSize=64 / tailMode=1） ──────────────────
//   l0Part = nextPow2(max(64,64)) = 64、l1Part = 512、l2Part = 4096
//   l0LenByTailStart = llround(tailStartSec × 48000) ≥ 5760（tailMode=1 の 0.12 クランプ）
//   → l0LenTarget = jlimit(64, 2048, ≥5760) = **2048 常時クランプ**（手順書 §1）
struct LayerCfgTheo
{
    int offset[3] = { 0, 0, 0 };
    int len[3]    = { 0, 0, 0 };
    int partSize[3] = { 0, 0, 0 };
    int numLayers = 0;
};

int nextPow2 (int n) noexcept
{
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

LayerCfgTheo computeLayerCfg (int irLen, bool tailEnabled)
{
    LayerCfgTheo cfg;
    const int l0Part  = nextPow2 (std::max (kBlockSize, 64));   // 64
    const int l1Part  = l0Part * kTailL1L2Mult;                 // 512
    const int l2Part  = l1Part * kTailL1L2Mult;                 // 4096
    const int l0MaxLen = kL0MaxParts * l0Part;                  // 2048
    const int l0Len    = std::min (irLen, l0MaxLen);            // tailMode=1 クランプで l0LenTarget==l0MaxLen
    const int l1Len    = tailEnabled ? std::max (0, std::min (irLen - l0Len, kL1MaxParts * l1Part)) : 0;
    const int l2Len    = tailEnabled ? std::max (0, irLen - l0Len - l1Len) : 0;

    cfg.partSize[0] = l0Part; cfg.partSize[1] = l1Part; cfg.partSize[2] = l2Part;
    cfg.len[0] = l0Len;  cfg.len[1] = l1Len;  cfg.len[2] = l2Len;
    cfg.offset[0] = 0;   cfg.offset[1] = l0Len; cfg.offset[2] = l0Len + l1Len;
    cfg.numLayers = (l0Len > 0 ? 1 : 0) + (l1Len > 0 ? 1 : 0) + (l2Len > 0 ? 1 : 0);
    return cfg;
}

// ── 判定帯（手順書 §6） ──────────────────────────────────────────────────
std::string bandOf (long long ope)
{
    if (ope == 0) return "PASS(B13-ALIGNED)";
    const long long a = ope < 0 ? -ope : ope;
    if (a <= 64)  return "INVESTIGATE(1..64)";
    return "FAIL(B13-NOT-ALIGNED)";
}

// ── テスト IR（決定的。ダイレクト + 顕著な早期反射 + 減衰ノイズ） ────────
void buildTestIR (int irLen, std::vector<double>& ir)
{
    ir.assign ((size_t) irLen, 0.0);
    std::uint64_t seed = 0x9E3779B97F4A7C15ull;
    auto rnd = [&seed]() {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        return (double) (seed >> 11) / 9007199254740992.0;
    };
    for (int i = 0; i < irLen; ++i)
        ir[(size_t) i] = std::exp (-4.0 * (double) i / (double) std::max (irLen, 1)) * (rnd() * 2.0 - 1.0) * 0.3;
    if (irLen > 0)   ir[0]   = 1.0;    // ダイレクト音
    if (irLen > 100) ir[100] += 0.5;   // 顕著な早期反射（M3 ピーク同定用）
    if (irLen > 300) ir[300] += 0.25;
}

// ── M1: layerGain 反映 reference（インパルス応答の解析的構成） ───────────
// 入力が単位インパルス x[n0]=1 のため、time-domain 畳み込み
//   y[n] = Σ_li gain[li] · Σ_m ir[off[li]+m] · x[n − n0 − off[li] − m]
// は y[n0 + off[li] + m] += gain[li]·ir[off[li]+m] の直接配置と数学的に等価。
// （ゲインは layerTailGain accessor の**実測値**を使用 — 設計値のハードコード禁止）
void buildReference (const double* ir, const LayerCfgTheo& cfg, int numLayers,
                     const double* gain, int n0, int totalLen, std::vector<double>& yRef)
{
    yRef.assign ((size_t) totalLen, 0.0);
    for (int li = 0; li < numLayers; ++li)
    {
        for (int m = 0; m < cfg.len[li]; ++m)
        {
            const std::size_t idx = (std::size_t) (n0 + cfg.offset[li] + m);
            if (idx < yRef.size())
                yRef[idx] += gain[li] * ir[(size_t) (cfg.offset[li] + m)];
        }
    }
}

// ── M1: 帯域別誤差スペクトル（MKL DFTI・real FFT） ────────────────────────
// bands: [0,200) / [200,1000) / [1000,10000) / [10000,24000] Hz
void computeBandSpectrum (const std::vector<double>& diff, const std::vector<double>& ref, double outDb[4])
{
    const std::size_t srcLen = diff.size();
    std::size_t n = 1;
    while (n < srcLen) n <<= 1;

    // ★ INPLACE real FFT (CCS) の出力は N+2 doubles（re0 と re_N/2 が単独 + (N/2-1) 複素ペア）。
    //   N サイズのバッファに書くと 2 doubles のヒープ破壊になるため N+2 を確保する。
    std::vector<double> d (n + 2, 0.0), r (n + 2, 0.0);
    std::copy (diff.begin(), diff.end(), d.begin());
    std::copy (ref.begin(),  ref.end(),  r.begin());

    DFTI_DESCRIPTOR_HANDLE hd = nullptr, hr = nullptr;
    MKL_LONG st = DftiCreateDescriptor (&hd, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG) n);
    if (st == 0) st = DftiSetValue (hd, DFTI_PLACEMENT, DFTI_INPLACE);
    if (st == 0) st = DftiCommitDescriptor (hd);
    if (st == 0) st = DftiCreateDescriptor (&hr, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG) n);
    if (st == 0) st = DftiSetValue (hr, DFTI_PLACEMENT, DFTI_INPLACE);
    if (st == 0) st = DftiCommitDescriptor (hr);
    if (st != 0)
    {
        DftiFreeDescriptor (&hd);
        DftiFreeDescriptor (&hr);
        for (int b = 0; b < 4; ++b) outDb[b] = -999.0;
        return;
    }
    DftiComputeForward (hd, d.data());
    DftiComputeForward (hr, r.data());
    DftiFreeDescriptor (&hd);
    DftiFreeDescriptor (&hr);

    const double bandEdge[5] = { 0.0, 200.0, 1000.0, 10000.0, 24000.0 };
    double eDiff[4] = { 0, 0, 0, 0 }, eRef[4] = { 0, 0, 0, 0 };
    const std::size_t half = n / 2;
    for (std::size_t k = 0; k <= half; ++k)
    {
        double re, im, rre, rim;
        if (k == 0)          { re = d[0]; im = 0.0; rre = r[0]; rim = 0.0; }
        else if (k == half)  { re = d[1]; im = 0.0; rre = r[1]; rim = 0.0; }
        else                 { re = d[2*k]; im = d[2*k+1]; rre = r[2*k]; rim = r[2*k+1]; }
        const double f = (double) k * kSampleRate / (double) n;
        int b = 3;
        for (int bi = 3; bi >= 0; --bi) if (f >= bandEdge[bi]) { b = bi; break; }
        eDiff[b] += re*re + im*im;
        eRef[b]  += rre*rre + rim*rim;
    }
    for (int b = 0; b < 4; ++b)
    {
        const double ratio = (eRef[b] > 1e-300) ? eDiff[b] / eRef[b] : (eDiff[b] > 0.0 ? 1e12 : 0.0);
        outDb[b] = (ratio > 0.0) ? 10.0 * std::log10 (ratio) : -999.0;
    }
}

struct M1Result
{
    double rmsDb = 0.0;          // 20log10(RMS(diff)/RMS(ref))
    double peakDb = 0.0;         // 20log10(max|diff|/RMS(ref))
    double bandDb[4] = { 0, 0, 0, 0 };
    double segRmsDb[3] = { 0, 0, 0 };  // セグメント別（L0 区間 / L1 区間 / テール区間）
};

// ── M3: セグメント相関による位置同定 ─────────────────────────────────────
// k* = argmax_{k ∈ [searchFrom, searchTo)} Σ_m seg[m] · nuc[k+m]
int correlationPeak (const std::vector<double>& nuc, const std::vector<double>& seg,
                     int searchFrom, int searchTo)
{
    int bestK = searchFrom;
    double bestVal = -1e300;
    const int segLen = (int) seg.size();
    const int lim = std::min (searchTo, (int) nuc.size() - segLen);
    for (int k = std::max (0, searchFrom); k < lim; ++k)
    {
        double acc = 0.0;
        for (int m = 0; m < segLen; ++m)
            acc += seg[(size_t) m] * nuc[(size_t) (k + m)];
        if (acc > bestVal) { bestVal = acc; bestK = k; }
    }
    return bestK;
}

// ── M2: 1 run（1 impulse / 1 Reset からのコールバック系列） ──────────────
struct M2RunResult
{
    bool setupOk = false;
    int  numLayers = 0;
    int  partSize[3] = { 0, 0, 0 };
    int  numPartsIR[3] = { 0, 0, 0 };
    int  ppc[3] = { 0, 0, 0 };
    int  outputDelaySamples[3] = { 0, 0, 0 };
    int  delayLineCapacity[3] = { 0, 0, 0 };
    double layerGain[3] = { 1.0, 0.0, 0.0 };
    int  totalCallbacks = 0;
    // write/read-anchor events（layer 別、j 昇順）
    int  writeEventCb[3][kMaxBlocksTrack] = {};
    int  writeEventCount[3] = { 0, 0, 0 };
    int  readAnchorCb[3][kMaxBlocksTrack] = {};
    int  readAnchorCount[3] = { 0, 0, 0 };
    long long opeConst[3] = { 0, 0, 0 };       // 定常 oPE（全 j 一致時の値）— **Primary gate**
    bool opeConstant[3] = { false, false, false };
    long long effConst[3] = { 0, 0, 0 };       // 定常 effectiveDelay（t − R_after）— 補助診断（H1: layer 別）
    bool effConstant[3] = { false, false, false };
    std::string firstReadMode[3];
    std::uint32_t i2ViolationCount = 0;        // I2 違反カウンタ（deterministic safety guard）実測値
    bool coverageOk[3] = { false, false, false };
    // ★ P0-1: Phase 0/1 分離集計（I6 契約 — coverage は Phase 1 のみで判定）
    int  phase0Callbacks[3] = { 0, 0, 0 };      // t < o_L の callback 数
    int  phase0Writes[3]    = { 0, 0, 0 };      // Phase 0 中の write event 数（未 read で正常）
    int  phase0Reads[3]     = { 0, 0, 0 };      // Phase 0 中の read 実行数（Policy R では 0 期待）
    int  phase1Callbacks[3] = { 0, 0, 0 };      // t ≥ o_L の callback 数
    int  phase1StartT[3]    = { 0, 0, 0 };      // Phase 1 開始時刻（= o_L）
    int  phase1ExpectedAnchors[3] = { 0, 0, 0 };// logical position ごとの expected read-anchor
    int  phase1ObservedAnchors[3] = { 0, 0, 0 };// 実測 read-anchor（unique）
    int  phase1MissingAnchors[3]  = { 0, 0, 0 };
    int  phase1DuplicateAnchors[3]= { 0, 0, 0 };
    double coveragePhase1[3] = { 0.0, 0.0, 0.0 };
    // ★ P1-x-I: R2-b WRITE/READ-CADENCE-SPLIT + R3-b′ margin 計測
    //   観測側の純粋関数のみで導出（production 変更・accessor 追加なし）
    int  lead[3] = { 0, 0, 0 };                  // lead = B×(bpp + distCbs − 2) — gate 式の観測側ミラー
    long long marginMin[3] = { 0, 0, 0 };        // min(delayWriteCursor_after − (readStart + B)) over executed reads
    long long marginMax[3] = { 0, 0, 0 };
    int  ringSplitCount[3] = { 0, 0, 0 };        // (readStart % cap) + B > cap — I5 下で構造的に 0 期待
    int  straddleCount[3]  = { 0, 0, 0 };        // (readStart % ps) + B > ps — I5 下で構造的に 0 期待
    bool telemetryModelMatch[3] = { false, false, false };   // B軸: 本番 delayReadCursor == readStart + B（連続性含む）
    bool readStartIndependent[3] = { false, false, false };  // C軸: readStart == t0 − o_L（wAfter 非依存の独立再計算と一致）
    bool clockContinuous[3] = { false, false, false };       // R4: t0 == cb×B 全行・evs 数 == totalCallbacks
    bool csvOk = false;
    std::string csvPath;
};

// Policy R（本番実装と同一の式 — Repair Design Rev 4 §2.2 / I1〜I6）で観測値を導出
// I4 Get Clock: t0 = この Get ブロックの先頭時刻。F1: Phase 0 判定（t0 < o_L）は減算前。
struct CursorObs
{
    std::uint64_t t0 = 0;
    std::uint64_t readStart = 0;   // 論理リードヘッド（F2: t0 − o_L）
    bool phase0 = false;           // t0 < o_L（L 寄与なし — reference 整合）
    bool readExecuted = false;
    std::string mode = "PHASE0";   // PHASE0 / POLICY-R / UNAVAILABLE
};

CursorObs observeCursor (std::uint64_t wAfter, std::uint64_t t0, int outputDelay, int blockSize)
{
    CursorObs o;
    o.t0 = t0;
    o.phase0 = (t0 < (std::uint64_t) outputDelay);
    if (o.phase0)
    {
        o.mode = "PHASE0";
        return o;    // Phase 0: 加算なし（reference と整合 — 欠落ではない）
    }
    o.readStart = t0 - (std::uint64_t) outputDelay;   // I4: 減算は Phase 0 判定後（underflow なし）
    o.readExecuted = o.readStart + (std::uint64_t) blockSize <= wAfter;
    o.mode = o.readExecuted ? "POLICY-R" : "UNAVAILABLE";
    return o;
}

bool runM2 (int irLen, bool tailEnabled, int n0, const double* ir,
            const LayerCfgTheo& cfg, const char* caseName, int runId,
            M2RunResult& out)
{
    convo::MKLNonUniformConvolver conv;
    bool ok;
    if (tailEnabled)
        ok = conv.SetImpulse (ir, irLen, kBlockSize, 1.0, /*enableDirectHead*/ false, /*filterSpec*/ nullptr);
    else
    {
        convo::FilterSpec spec {};    // デフォルト（tailMode=1, tailStartSeconds=0.085）+ tail bypass
        spec.tailEnabled = false;
        ok = conv.SetImpulse (ir, irLen, kBlockSize, 1.0, false, &spec);
    }
    if (!ok) return false;

    out.setupOk = true;
    out.numLayers = convo::NUPCTestAccess::numActiveLayers (conv);
    for (int li = 0; li < out.numLayers; ++li)
    {
        out.partSize[li]           = convo::NUPCTestAccess::layerPartSize (conv, li);
        out.numPartsIR[li]         = convo::NUPCTestAccess::layerNumPartsIR (conv, li);
        out.ppc[li]                = convo::NUPCTestAccess::layerPartsPerCallback (conv, li);
        out.outputDelaySamples[li] = convo::NUPCTestAccess::layerOutputDelaySamples (conv, li);
        out.delayLineCapacity[li]  = convo::NUPCTestAccess::layerDelayLineCapacity (conv, li);
        out.layerGain[li]          = convo::NUPCTestAccess::layerTailGain (conv, li);
    }
    out.i2ViolationCount = convo::NUPCTestAccess::delayI2ViolationCount (conv);

    // ── CSV（std::ofstream・型安全書き込み） ──
    std::filesystem::create_directories ("nupc_v29_csv");
    out.csvPath = std::string ("nupc_v29_csv/M2_") + caseName + "_run" + std::to_string (runId) + ".csv";
    std::ofstream csv (out.csvPath);
    out.csvOk = csv.good();
    if (csv)
        csv << "run_id,impulse_pos,t,layer,layerGain,t0,"
               "delayWriteCursor_before,delayWriteCursor_after,"
               "delayReadCursor_before,delayReadCursor_after,"
               "readStart,readExecuted,readMode,phase0,"
               "i2ViolationCount,outputDelaySamples,partSize,numPartsIR,partsPerCallback\n";

    const int totalLen = ((n0 + irLen * 2 + 8192 + kBlockSize - 1) / kBlockSize) * kBlockSize;
    const int totalCallbacks = totalLen / kBlockSize;
    out.totalCallbacks = totalCallbacks;

    std::vector<double> inBlock ((size_t) kBlockSize, 0.0);
    std::vector<double> outBlock ((size_t) kBlockSize, 0.0);

    // run 中の生イベント記録（解析は run 終了後）— Policy R: read は t0 ベース
    struct Ev { int cb; std::uint64_t t0, wBefore, wAfter, rBefore, rAfter; };
    std::vector<Ev> evs[3];

    for (int c = 0; c < totalCallbacks; ++c)
    {
        const int t = c * kBlockSize;
        for (int i = 0; i < kBlockSize; ++i)
            inBlock[(size_t) i] = (t + i == n0) ? 1.0 : 0.0;

        // ── Add 前後で delayWriteCursor を読む（write event 検出） ──
        std::uint64_t wBefore[3] = { 0, 0, 0 }, wAfter[3] = { 0, 0, 0 };
        for (int li = 1; li < out.numLayers; ++li)
            wBefore[li] = convo::NUPCTestAccess::layerDelayWriteCursor (conv, li);
        conv.Add (inBlock.data(), kBlockSize);
        for (int li = 1; li < out.numLayers; ++li)
            wAfter[li] = convo::NUPCTestAccess::layerDelayWriteCursor (conv, li);

        // ── Get（有効出力バッファ必須。本番と同じ Add→Get 順序） ──
        // I4 Get Clock: t0 = この Get ブロックの先頭時刻（Get 呼び出し時点の clock）
        const std::uint64_t t0 = convo::NUPCTestAccess::outputSamplesProcessed (conv);
        std::uint64_t rBefore[3] = { 0, 0, 0 }, rAfter[3] = { 0, 0, 0 };
        for (int li = 1; li < out.numLayers; ++li)
            rBefore[li] = convo::NUPCTestAccess::layerDelayReadCursor (conv, li);
        conv.Get (outBlock.data(), kBlockSize);
        for (int li = 1; li < out.numLayers; ++li)
            rAfter[li] = convo::NUPCTestAccess::layerDelayReadCursor (conv, li);

        // ── 本番実装と同一の式（Policy R）で観測値を再構成 ──
        // evs は全 callback を記録する（Phase 0 を含む）。write event は Phase 0 期間にも発生
        // （t_write(0) = lead < o_L）するため、read 実行の有無でフィルタすると write 欠落になる。
        // read-anchor / I2 判定は CursorObs（phase0 / readExecuted）で分離。
        for (int li = 1; li < out.numLayers; ++li)
        {
            const CursorObs o = observeCursor (wAfter[li], t0, out.outputDelaySamples[li], kBlockSize);
            evs[li].push_back ({ c, t0, wBefore[li], wAfter[li], rBefore[li], rAfter[li] });
            if (csv)
            {
                csv << runId << ',' << n0 << ',' << t << ',' << li << ',' << out.layerGain[li] << ',' << t0 << ','
                    << wBefore[li] << ',' << wAfter[li] << ','
                    << rBefore[li] << ',' << rAfter[li] << ','
                    << o.readStart << ',' << (o.readExecuted ? 1 : 0) << ',' << o.mode << ',' << (o.phase0 ? 1 : 0) << ','
                    << out.i2ViolationCount << ',' << out.outputDelaySamples[li] << ',' << out.partSize[li] << ','
                    << out.numPartsIR[li] << ',' << out.ppc[li] << '\n';
            }
        }
    }

    // ── イベント抽出と oPE / effectiveDelay の導出 ──
    for (int li = 1; li < out.numLayers; ++li)
    {
        const int ps  = out.partSize[li];
        const int oL  = out.outputDelaySamples[li];
        const int irOffset = cfg.offset[li];   // L1: l0Len、L2: l0Len+l1Len
        const int numBlocks = (int) (evs[li].empty() ? 0 : evs[li].back().wAfter / (std::uint64_t) ps);

        // ── P0-1: Phase 0/1 分離集計（I6 契約 — coverage は Phase 1 のみで判定） ──
        // Phase 境界 callback（I5: o_L % B == 0 → 境界は整数 callback）
        const int bCb = oL / kBlockSize;       // t = o_L となる callback（Phase 1 開始）
        out.phase0Callbacks[li] = std::min (totalCallbacks, bCb);
        out.phase1Callbacks[li] = std::max (0, totalCallbacks - bCb);
        out.phase1StartT[li]    = oL;

        // observed anchor 集計（j ごとの観測 callback — duplicate 検出対応）
        //   read-anchor: readExecuted ∧ readStart == j×ps（Policy R: readStart = t0 − o_L）
        //   Phase 0（t < o_L）は加算なし（F1 契約）→ read-anchor は Phase 1 でのみ発生
        std::map<int, std::vector<int>> anchorCbs;   // j → callback list（重複検出対応）
        int p0w = 0, p0r = 0, p1w = 0;
        int wj = 0;
        for (const auto& e : evs[li])
        {
            const CursorObs o = observeCursor (e.wAfter, e.t0, oL, kBlockSize);
            // write event（W_before == j×ps ∧ W_after == (j+1)×ps、j 昇順照合）
            if (e.wBefore == (std::uint64_t) (wj * ps) && e.wAfter == (std::uint64_t) ((wj + 1) * ps))
            {
                if (e.cb < bCb) ++p0w; else ++p1w;
                ++wj;
            }
            // read-anchor（readStart == j×ps）
            if (o.readExecuted)
            {
                const std::uint64_t psU = (std::uint64_t) ps;
                if (o.readStart % psU == 0)
                {
                    const int j = (int) (o.readStart / psU);
                    anchorCbs[j].push_back (e.cb);
                    if (o.phase0) ++p0r;   // Phase 0 中の read 実行は I1 違反候補
                }
            }
        }
        out.phase0Writes[li] = p0w;
        out.phase0Reads[li]  = p0r;

        // write event 抽出（writeEventCb — 既存の逐次照合を維持。coverageOk 判定からは除外）
        int j = 0;
        for (const auto& e : evs[li])
        {
            if (e.wBefore == (std::uint64_t) (j * ps) && e.wAfter == (std::uint64_t) ((j + 1) * ps))
            {
                if (j < kMaxBlocksTrack) out.writeEventCb[li][out.writeEventCount[li]++] = e.cb;
                ++j;
            }
        }

        // expected: logical position j のうち、content_time(j) = jP + o_L が
        // run 時間内（最終 callback の末尾 t = totalCallbacks × B）に到達するもののみ。
        // run 時間外の j は expected に含めない（missing 3 = run 長不足分の誤集計を修正）。
        const long long totalStream = (long long) totalCallbacks * kBlockSize;
        int expected = 0;
        for (int jx = 0; jx < numBlocks; ++jx)
            if ((long long) jx * ps + oL <= totalStream) ++expected;
        out.phase1ExpectedAnchors[li] = expected;
        out.phase1ObservedAnchors[li] = (int) anchorCbs.size();
        int missing = 0, dup = 0;
        for (int jx = 0; jx < numBlocks; ++jx)
        {
            const auto it = anchorCbs.find (jx);
            if (it == anchorCbs.end())
            {
                // run 時間内の content のみ missing として数える
                if ((long long) jx * ps + oL <= totalStream) ++missing;
            }
            else if (it->second.size() > 1) dup += (int) it->second.size() - 1;
        }
        out.phase1MissingAnchors[li] = missing;
        out.phase1DuplicateAnchors[li] = dup;
        out.coveragePhase1[li] = (out.phase1ExpectedAnchors[li] > 0)
            ? (double) out.phase1ObservedAnchors[li] / (double) out.phase1ExpectedAnchors[li] : 0.0;

        // oPE 計算（anchorCbs ベース — 全 anchor で同一値か確認）
        long long opeVal = 0; bool opeSeen = false; bool opeAllSame = true;
        for (const auto& [jx, cbs] : anchorCbs)
        {
            const long long contentTime = (long long) jx * ps + irOffset;
            for (int cb : cbs)
            {
                const long long ope = (long long) cb * kBlockSize - contentTime;
                if (! opeSeen) { opeVal = ope; opeSeen = true; }
                else if (ope != opeVal) opeAllSame = false;
            }
        }
        out.opeConstant[li] = opeAllSame && opeSeen;
        out.opeConst[li] = opeVal;

        // effectiveDelay（t − R_after）— 補助診断（Policy R では o_L − B 定常）
        //   evs 全体（read 実行 callback）から計算 — anchorCbs ループでは rAfter が取れないため分離
        // ★ P1-x-I: 同一ループで margin 分布・telemetry 突合・split カウント・clock 連続性を実施
        long long effVal = 0; bool effSeen = false; bool effAllSame = true;
        long long mMin = 0, mMax = 0; bool mSeen = false;
        bool telemetryOk = true, independenceOk = true, clockOk = true, telemetrySeen = false;
        int ringSplit = 0, straddle = 0;
        const std::uint64_t capU = (std::uint64_t) out.delayLineCapacity[li];
        const std::uint64_t psU  = (std::uint64_t) ps;
        std::uint64_t prevRAfter = 0; bool prevSeen = false;
        for (const auto& e : evs[li])
        {
            // R4: clock 連続性 — 各 Get の t0 が cb×B と一致（連続する output stream clock）
            if (e.t0 != (std::uint64_t) e.cb * (std::uint64_t) kBlockSize) clockOk = false;
            const CursorObs o = observeCursor (e.wAfter, e.t0, oL, kBlockSize);
            if (o.readExecuted)
            {
                const long long eff = (long long) e.cb * kBlockSize - (long long) e.rAfter;
                if (! effSeen) { effVal = eff; effSeen = true; }
                else if (eff != effVal) effAllSame = false;

                // B軸: 本番 telemetry（delayReadCursor）と Policy R モデルの突合
                //   rAfter == readStart + B（cpp:1832 delayReadCursor = readStart + numSamples）
                //   + 連続性: rBefore == 直前の rAfter（Get 間で telemetry cursor が途切れない）
                if (e.rAfter != o.readStart + (std::uint64_t) kBlockSize) telemetryOk = false;
                if (prevSeen && e.rBefore != prevRAfter) telemetryOk = false;
                prevRAfter = e.rAfter; prevSeen = true;
                telemetrySeen = true;

                // C軸: readStart の独立性 — wAfter を使わない独立再計算（t0 − o_L）と一致
                //   = read 位置が write cursor の変動に依存しないことの実測
                const std::uint64_t readStartModel = e.t0 - (std::uint64_t) oL;   // Phase 1 のみ到達
                if (o.readStart != readStartModel) independenceOk = false;

                // R3-b′: margin = write head − (readStart + B) — 期待: margin_min == o_L − lead
                const long long margin = (long long) e.wAfter
                                       - (long long) (o.readStart + (std::uint64_t) kBlockSize);
                if (! mSeen) { mMin = margin; mMax = margin; mSeen = true; }
                else { mMin = std::min (mMin, margin); mMax = std::max (mMax, margin); }

                // D軸: split 2 種（I5: P%B==0 ∧ o_L%B==0 下では構造的に 0 期待）
                if ((o.readStart % capU) + (std::uint64_t) kBlockSize > capU) ++ringSplit;
                if ((o.readStart % psU) + (std::uint64_t) kBlockSize > psU) ++straddle;
            }
        }
        out.effConstant[li] = effAllSame && effSeen;
        out.effConst[li] = effVal;
        // ★ P1-x-I: lead 再計算（gate 式ミラー — 観測側の純粋関数・cpp:990-993 の実測値から導出）
        {
            const int bpp     = (ps + kBlockSize - 1) / kBlockSize;
            const int distCbs = (out.numPartsIR[li] + out.ppc[li] - 1) / out.ppc[li];
            out.lead[li] = kBlockSize * (bpp + distCbs - 2);
        }
        out.marginMin[li] = mSeen ? mMin : 0;
        out.marginMax[li] = mSeen ? mMax : 0;
        out.ringSplitCount[li] = ringSplit;
        out.straddleCount[li]  = straddle;
        out.telemetryModelMatch[li] = telemetrySeen && telemetryOk;
        out.readStartIndependent[li] = telemetrySeen && independenceOk;
        out.clockContinuous[li] = clockOk && ((int) evs[li].size() == out.totalCallbacks);

        // coverageOk（Phase 1 基準 — P0-1 で変更）:
        //   Phase 1 expected 全 position が observed（missing 0）かつ duplicate 0。
        //   Phase 0 中の write（未 read）は正常扱い（I6 契約）— coverageOk に含めない。
        //   numBlocks と expected の差（run 時間外の論理位置）も異常ではない。
        out.coverageOk[li] = (out.phase1ExpectedAnchors[li] > 0
                              && out.phase1MissingAnchors[li] == 0
                              && out.phase1DuplicateAnchors[li] == 0);
        // 初回 read の mode
        if (! evs[li].empty())
            out.firstReadMode[li] = observeCursor (evs[li].front().wAfter, evs[li].front().t0,
                                                   out.outputDelaySamples[li], kBlockSize).mode;
    }
    out.i2ViolationCount = convo::NUPCTestAccess::delayI2ViolationCount (conv);
    return true;
}

// ── M2 定常 oPE の判定サマリー（1 layer 分） ─────────────────────────────
// 修復後（Policy R）: Primary gate は oPE == 0。effDelay は補助診断（Policy R では o_L − B 定常）。
void printM2Line (const char* caseName, int runId, int li, const M2RunResult& r, int expectedOpe, int expectedEff)
{
    std::cout << "  M2 " << std::left << std::setw (4) << caseName << " run" << runId
              << " L" << li
              << ": oPE=" << r.opeConst[li]
              << " (expected " << expectedOpe << ", " << ((r.opeConst[li] == expectedOpe) ? "MODEL-MATCH" : "MODEL-DIFF") << ")"
              << " band=" << bandOf (r.opeConst[li])
              << " | effDelay=" << r.effConst[li] << " (expected " << expectedEff << ", " << (r.effConstant[li] ? "const" : "varies") << ")"
              << " | readMode first=" << r.firstReadMode[li]
              << " | i2Violations=" << r.i2ViolationCount
              << " | coverage=" << (r.coverageOk[li] ? "OK" : "MISSING")
              << "\n";
    // ★ P0-1: Phase 0/1 分離集計（I6 契約 — coverage は Phase 1 のみで判定）
    std::cout << std::fixed << std::setprecision (6)
              << "  COV L" << li << ": Phase0(cb=" << r.phase0Callbacks[li]
              << " writes=" << r.phase0Writes[li] << " reads=" << r.phase0Reads[li] << ")"
              << " Phase1(start=" << r.phase1StartT[li] << " cb=" << r.phase1Callbacks[li] << ")"
              << " expected=" << r.phase1ExpectedAnchors[li]
              << " observed=" << r.phase1ObservedAnchors[li]
              << " missing=" << r.phase1MissingAnchors[li]
              << " dup=" << r.phase1DuplicateAnchors[li]
              << " coveragePhase1=" << r.coveragePhase1[li] << "\n";
}

// ── P1-x 5 軸判定（run レベル・R2-b WRITE/READ-CADENCE-SPLIT） ─────────────
// A output-placement / B stream-read-anchor / C write-read-independence /
// D block-boundary-independence / E phase-0 — M2 一本化を排除し個別 assert する。
struct P1xAxes { bool a = false, b = false, c = false, d = false, e = false; };

P1xAxes computeP1xAxes (const M2RunResult& r, int expectedOpeL1, int expectedOpeL2)
{
    P1xAxes x { true, true, true, true, true };
    for (int li = 1; li < r.numLayers; ++li)
    {
        // A: output placement — oPE 定常かつ expected（0）と一致
        if (! (r.opeConstant[li] && r.opeConst[li] == (li == 1 ? expectedOpeL1 : expectedOpeL2)))
            x.a = false;
        // B: stream read anchor — 本番 telemetry が Policy R モデルと一致
        //    （rAfter == readStart + B・rBefore/rAfter 連続・readStart = t0 − o_L）
        if (! r.telemetryModelMatch[li]) x.b = false;
        // C: write/read independence — readStart が t0 のみで決定（write cursor 変動に非依存）
        //    + Get 間の clock 連続性（R4）
        if (! (r.readStartIndependent[li] && r.clockContinuous[li])) x.c = false;
        // D: block-boundary independence — ring split / partition straddle とも 0
        //    （I5 下の構造的不発 — 0 以外は Policy R モデル破綻）
        if (! (r.ringSplitCount[li] == 0 && r.straddleCount[li] == 0)) x.d = false;
        // E: Phase 0 invariant — Phase 0 中 read 0 + I2 violation 0
        if (! (r.phase0Reads[li] == 0 && r.i2ViolationCount == 0)) x.e = false;
    }
    return x;
}

// sweep サマリ用の 1 行（case × layer ごとの lead/margin 計測結果）
struct P1xSweepRow
{
    const char* caseName = nullptr;
    int irLen = 0;
    int li = 0;
    int numPartsIR = 0;
    int ppc = 0;
    int lead = 0;
    long long marginMin = 0;
    long long marginMax = 0;
    long long theoryMarginMin = 0;   // o_L − lead
    long long ope = 0;
    std::uint32_t i2 = 0;
    long long expectedEff = 0;
    long long actualEff = 0;
    bool effConstant = false;
};

} // anonymous namespace

// ══════════════════════════════════════════════════════════════════════════
// main — T1〜T8 実行
// ══════════════════════════════════════════════════════════════════════════
int main()
{
    juce::initialiseJuce_GUI();

    int structuralFailures = 0;
    std::cout << "=== Gardner Null Test v2.9 (Step 3) ===\n";
    std::cout << "sampleRate=" << kSampleRate << " blockSize=" << kBlockSize << " (48kHz/64 専用仕様)\n\n";

    // ── テスト IR ──
    constexpr int maxIrLen = 46000;   // ★ P1-x-I: R2bL2 (irLen=45000) 用に拡大
    std::vector<double> ir;
    buildTestIR (maxIrLen, ir);

    // ── case 定義（期待値は Policy R 修復後: oPE == 0 / effDelay == o_L − B） ──
    // （旧 policy 実測値 −1152〜−1600 / −30720 は Step 3 報告書 v1.2 に記録済み）
    struct Case
    {
        const char* name = nullptr;
        int irLen = 0;
        bool tailEnabled = false;
        std::vector<int> n0s {};
        int expectedOpeL1 = 0;
        int expectedEffL1 = 0;
        int expectedOpeL2 = 0;
        int expectedEffL2 = 0;
    };
    const Case cases[] = {
        { "T1",  2000,  true,  { 0 },                            0, 0,    0,     0 },
        { "T2",  2048,  true,  { 0 },                            0, 0,    0,     0 },
        { "T3",  2049,  true,  { 0, 4096 },                      0, 1984, 0,     0 },
        { "T4",  5000,  true,  { 0, 4096, 8192, 16384 },         0, 1984, 0,     0 },
        { "T5",  8000,  true,  { 0, 4096, 8192, 16384 },         0, 1984, 0,     0 },
        { "T6", 12000,  true,  { 0, 4096, 8192, 16384 },         0, 1984, 0,     0 },
        { "T7", 40000,  true,  { 0, 4096, 8192, 16384 },         0, 1984, 0,     34752 },
        { "T8",  5000,  false, { 0 },                            0, 0,    0,     0 },
        // ★ P1-x-I R2-b WRITE/READ-CADENCE-SPLIT sweep 追加ケース
        //   R2b  : irLen=20000 → l1Len=17952 (nIR=36, ppc=5, distCbs=8, lead=896)・l2Len=0 → L1 のみ
        //   R2bL2: irLen=45000 → l1Len=32768(cap) (nIR=64, ppc=8, lead=896) + l2Len=10184
        //          (nIR=3, ppc=1, distCbs=3, lead=64×(64+3−2)=4160) — L2 lead 変動の実証
        { "R2b",  20000, true, { 0, 4096 },                      0, 1984, 0,     0 },
        { "R2bL2",45000, true, { 0, 4096 },                      0, 1984, 0,     34752 },
    };

    // ★ P1-x-I: R2-b sweep 収集 + M1 結果収集（サマリで併記）
    std::vector<P1xSweepRow> p1xSweep;
    std::map<std::string, std::string> p1xM1Verdict;   // caseName → PASS/FAIL/EXCLUDED

    std::cout << "=== M2: 三時刻観測（主判定）===\n";
    for (const auto& cs : cases)
    {
        const LayerCfgTheo cfg = computeLayerCfg (cs.irLen, cs.tailEnabled);
        int runId = 0;
        for (int n0 : cs.n0s)
        {
            M2RunResult r;
            if (! runM2 (cs.irLen, cs.tailEnabled, n0, ir.data(), cfg, cs.name, runId, r))
            {
                std::cout << "  M2 " << cs.name << " run" << runId << ": SetImpulse FAILED\n";
                ++structuralFailures;
                ++runId;
                continue;
            }

            // ── スタートアップゲート（expected/actual 分離。outputDelaySamples は assert しない） ──
            std::cout << "  M2 " << cs.name << " run" << runId << " (n0=" << n0 << "): layers=" << r.numLayers << " |";
            for (int li = 0; li < r.numLayers; ++li)
            {
                const int expLen = cfg.len[li];
                const int expParts = (expLen == 0) ? 0 : (expLen + r.partSize[li] - 1) / r.partSize[li];
                std::cout << " L" << li << ": partSize=" << r.partSize[li] << "(exp " << cfg.partSize[li] << ")"
                          << " numPartsIR=" << r.numPartsIR[li] << "(exp " << expParts << ")"
                          << " ppc=" << r.ppc[li]
                          << " outputDelay=" << r.outputDelaySamples[li]
                          << "(cap " << r.delayLineCapacity[li] << ")"
                          << " gain=" << std::fixed << std::setprecision (6) << r.layerGain[li] << " |";
            }
            std::cout << " csv=" << (r.csvOk ? "OK" : "FAIL") << "\n";

            // topology 照合（partSize / numPartsIR は構造値 → 不一致は構造的異常）
            for (int li = 0; li < r.numLayers; ++li)
            {
                const int expLen = cfg.len[li];
                const int expParts = (expLen == 0) ? 0 : (expLen + r.partSize[li] - 1) / r.partSize[li];
                if (r.partSize[li] != cfg.partSize[li] || r.numPartsIR[li] != expParts)
                {
                    std::cout << "    [STRUCTURAL-FAIL] L" << li << " topology mismatch (partSize "
                              << r.partSize[li] << " vs " << cfg.partSize[li] << ", numPartsIR "
                              << r.numPartsIR[li] << " vs " << expParts << ")\n";
                    ++structuralFailures;
                }
                if (li >= 1 && ! r.coverageOk[li])
                {
                    // P0-1: coverage は Phase 1 基準（missing/dup = 0 が合格）。
                    //   Phase 0 中の write（未 read）は正常 — write>read の総数比は判定に使用しない。
                    std::cout << "    [STRUCTURAL-FAIL] L" << li
                              << " event coverage (Phase 1) missing=" << r.phase1MissingAnchors[li]
                              << " dup=" << r.phase1DuplicateAnchors[li]
                              << " (writes=" << r.writeEventCount[li]
                              << ", Phase1 anchors=" << r.phase1ObservedAnchors[li] << ")\n";
                    ++structuralFailures;
                }
            }

            // ── M2 判定行（期待値は case 別 — 手順書 §2.3） ──
            if (r.numLayers >= 2)
                printM2Line (cs.name, runId, 1, r, cs.expectedOpeL1, cs.expectedEffL1);
            if (r.numLayers >= 3)
                printM2Line (cs.name, runId, 2, r, cs.expectedOpeL2, cs.expectedEffL2);

            // ★ P1-x-I: 5 軸判定（R2-b WRITE/READ-CADENCE-SPLIT — M2 一本化の排除）
            const P1xAxes ax = computeP1xAxes (r, cs.expectedOpeL1, cs.expectedOpeL2);
            for (int li = 1; li < r.numLayers; ++li)
            {
                const bool m2ok = r.opeConstant[li]
                               && r.opeConst[li] == (li == 1 ? cs.expectedOpeL1 : cs.expectedOpeL2);
                std::cout << "  P1-x axes " << cs.name << " run" << runId << " L" << li
                          << ": A output-placement=" << (ax.a ? "PASS" : "FAIL")
                          << " B stream-read-anchor=" << (ax.b ? "PASS" : "FAIL")
                          << " C write/read-independence=" << (ax.c ? "PASS" : "FAIL")
                          << " D block-boundary-independence=" << (ax.d ? "PASS" : "FAIL")
                          << " E phase-0=" << (ax.e ? "PASS" : "FAIL")
                          << " | M2=" << (m2ok ? "PASS" : "FAIL")
                          << " I2=" << (r.i2ViolationCount == 0 ? "PASS" : "FAIL")
                          << "\n";
                p1xSweep.push_back ({ cs.name, cs.irLen, li,
                                      r.numPartsIR[li], r.ppc[li], r.lead[li],
                                      r.marginMin[li], r.marginMax[li],
                                      (long long) r.outputDelaySamples[li] - r.lead[li],
                                      r.opeConst[li], r.i2ViolationCount,
                                      (li == 1 ? cs.expectedEffL1 : cs.expectedEffL2),
                                      r.effConst[li], r.effConstant[li] });
            }
            ++runId;
        }
    }

    // ── M1: layerGain 反映 Null Test（主要 case を全長計算） ──
    std::cout << "\n=== M1: layerGain 反映 Null Test（align=0）===\n";
    for (const auto& cs : cases)
    {
        const LayerCfgTheo cfg = computeLayerCfg (cs.irLen, cs.tailEnabled);
        const int n0 = cs.n0s.empty() ? 0 : cs.n0s[0];
        const int totalLen = ((n0 + cs.irLen * 2 + 8192 + kBlockSize - 1) / kBlockSize) * kBlockSize;

        convo::MKLNonUniformConvolver conv;
        bool ok;
        if (cs.tailEnabled)
            ok = conv.SetImpulse (ir.data(), cs.irLen, kBlockSize, 1.0, false, nullptr);
        else
        {
            convo::FilterSpec spec {}; spec.tailEnabled = false;
            ok = conv.SetImpulse (ir.data(), cs.irLen, kBlockSize, 1.0, false, &spec);
        }
        if (! ok)
        {
            std::cout << "  M1 " << cs.name << ": SetImpulse FAILED\n";
            ++structuralFailures;
            continue;
        }

        const int numLayers = convo::NUPCTestAccess::numActiveLayers (conv);
        double gain[3] = { 1.0, 0.0, 0.0 };
        for (int li = 0; li < numLayers; ++li)
            gain[li] = convo::NUPCTestAccess::layerTailGain (conv, li);

        std::vector<double> inBlock ((size_t) kBlockSize, 0.0), outBlock ((size_t) kBlockSize, 0.0);
        std::vector<double> nuc ((size_t) totalLen, 0.0);
        for (int c = 0; c < totalLen / kBlockSize; ++c)
        {
            for (int i = 0; i < kBlockSize; ++i)
                inBlock[(size_t) i] = (c * kBlockSize + i == n0) ? 1.0 : 0.0;
            conv.Add (inBlock.data(), kBlockSize);
            conv.Get (outBlock.data(), kBlockSize);   // 有効出力バッファ必須
            std::copy (outBlock.begin(), outBlock.end(), nuc.begin() + (std::size_t) (c * kBlockSize));
        }

        // reference（layerGain 実測値を反映 — 設計値ハードコード禁止）
        std::vector<double> yRef;
        buildReference (ir.data(), cfg, numLayers, gain, n0, totalLen, yRef);

        std::vector<double> diff ((size_t) totalLen, 0.0);
        double sumRefSq = 0.0, sumDiffSq = 0.0, maxAbs = 0.0;
        for (int n = 0; n < totalLen; ++n)
        {
            const double d = nuc[(size_t) n] - yRef[(size_t) n];
            diff[(size_t) n] = d;
            sumRefSq  += yRef[(size_t) n] * yRef[(size_t) n];
            sumDiffSq += d * d;
            if (std::fabs (d) > maxAbs) maxAbs = std::fabs (d);
        }
        const double rmsRef = std::sqrt (sumRefSq / (double) totalLen);
        const double rmsDiff = std::sqrt (sumDiffSq / (double) totalLen);
        M1Result m1;
        m1.rmsDb  = (rmsDiff > 0.0 && rmsRef > 0.0) ? 20.0 * std::log10 (rmsDiff / rmsRef) : -999.0;
        m1.peakDb = (maxAbs > 0.0 && rmsRef > 0.0) ? 20.0 * std::log10 (maxAbs / rmsRef) : -999.0;
        computeBandSpectrum (diff, yRef, m1.bandDb);

        // セグメント別 RMS（位置局在: L0 区間 / L1 区間 / テール区間）
        const int segStart[3] = { n0, n0 + cfg.offset[1], n0 + cfg.offset[2] };
        const int segLen[3]   = { cfg.len[0], cfg.len[1], cfg.len[2] };
        for (int s = 0; s < 3; ++s)
        {
            double sd = 0.0, sr = 0.0;
            for (int n = segStart[s]; n < segStart[s] + segLen[s] && n < totalLen; ++n)
            {
                sd += diff[(size_t) n] * diff[(size_t) n];
                sr += yRef[(size_t) n] * yRef[(size_t) n];
            }
            const double rmsSd = std::sqrt (sd / (double) std::max (segLen[s], 1));
            const double rmsSr = std::sqrt (sr / (double) std::max (segLen[s], 1));
            m1.segRmsDb[s] = (rmsSd > 0.0 && rmsSr > 0.0) ? 20.0 * std::log10 (rmsSd / rmsSr) : -999.0;
        }

        std::cout << std::fixed << std::setprecision (2)
                  << "  M1 " << cs.name << ": RMS=" << std::setw (8) << m1.rmsDb
                  << " dB (gate: " << (cs.tailEnabled ? (m1.rmsDb < -90.0 ? "PASS" : "FAIL")
                                                       : "EXCLUDED (filterSpec mismatch)") << ")"
                  << " Peak=" << m1.peakDb
                  << " dB | band[dB] " << m1.bandDb[0] << "/" << m1.bandDb[1] << "/" << m1.bandDb[2] << "/" << m1.bandDb[3]
                  << " | seg RMS[dB] L0:" << m1.segRmsDb[0] << " L1:" << m1.segRmsDb[1] << " L2:" << m1.segRmsDb[2]
                  << std::setprecision (6) << " | gain=" << gain[1] << "/" << gain[2] << "\n";
        p1xM1Verdict[cs.name] = cs.tailEnabled ? (m1.rmsDb < -90.0 ? "PASS" : "FAIL")
                                               : "EXCLUDED (filterSpec mismatch)";
    }

    // ── M3: インパルス応答の L1/L2 成分位置同定（補助証拠） ──
    std::cout << "\n=== M3: 位置同定（補助証拠・k*(B) − (n0+IR_offset) ≒ oPE 期待値）===\n";
    for (const auto& cs : cases)
    {
        if (cs.irLen <= 2048) continue;   // L1 なし（T1/T2）
        const LayerCfgTheo cfg = computeLayerCfg (cs.irLen, cs.tailEnabled);
        if (cfg.len[1] <= 0) continue;    // T8 は tail bypass
        const int n0 = cs.n0s.empty() ? 0 : cs.n0s[0];

        convo::MKLNonUniformConvolver conv;
        conv.SetImpulse (ir.data(), cs.irLen, kBlockSize, 1.0, false, nullptr);
        const int numLayers = convo::NUPCTestAccess::numActiveLayers (conv);
        const int totalLen = ((n0 + cs.irLen * 2 + 8192 + kBlockSize - 1) / kBlockSize) * kBlockSize;
        std::vector<double> inBlock ((size_t) kBlockSize, 0.0), outBlock ((size_t) kBlockSize, 0.0);
        std::vector<double> nuc ((size_t) totalLen, 0.0);
        for (int c = 0; c < totalLen / kBlockSize; ++c)
        {
            for (int i = 0; i < kBlockSize; ++i)
                inBlock[(size_t) i] = (c * kBlockSize + i == n0) ? 1.0 : 0.0;
            conv.Add (inBlock.data(), kBlockSize);
            conv.Get (outBlock.data(), kBlockSize);
            std::copy (outBlock.begin(), outBlock.end(), nuc.begin() + (std::size_t) (c * kBlockSize));
        }

        // L1 セグメント（gain 反映）で相関
        std::vector<double> seg ((size_t) cfg.len[1]);
        const double g1 = convo::NUPCTestAccess::layerTailGain (conv, numLayers >= 2 ? 1 : 0);
        for (int m = 0; m < cfg.len[1]; ++m) seg[(size_t) m] = g1 * ir[(size_t) (cfg.offset[1] + m)];
        const int k1 = correlationPeak (nuc, seg, n0 + cfg.offset[1] - 2048, n0 + cfg.offset[1] + 16384);
        std::cout << "  M3 " << cs.name << " L1: k*(B)=" << k1 << ", n0+IR_offset=" << (n0 + cfg.offset[1])
                  << ", diff=" << (k1 - (n0 + cfg.offset[1])) << " (expected oPE=" << cs.expectedOpeL1 << ")\n";

        // L2 セグメント（存在時）
        if (numLayers >= 3 && cfg.len[2] > 0)
        {
            std::vector<double> seg2 ((size_t) cfg.len[2]);
            const double g2 = convo::NUPCTestAccess::layerTailGain (conv, 2);
            for (int m = 0; m < cfg.len[2]; ++m) seg2[(size_t) m] = g2 * ir[(size_t) (cfg.offset[2] + m)];
            const int k2 = correlationPeak (nuc, seg2, n0 + cfg.offset[2] - 32768, n0 + cfg.offset[2] + 32768);
            std::cout << "  M3 " << cs.name << " L2: k*(B)=" << k2 << ", n0+IR_offset=" << (n0 + cfg.offset[2])
                      << ", diff=" << (k2 - (n0 + cfg.offset[2])) << " (expected oPE=" << cs.expectedOpeL2 << ")\n";
        }
    }

    // ══ P1-x-I: R2-b WRITE/READ-CADENCE-SPLIT sweep サマリ + R3-b′ 飽和証明 ══
    std::cout << "\n=== P1-x R2-b WRITE/READ-CADENCE-SPLIT sweep summary ===\n";
    std::cout << "  (write: partition-completion burst [P samples] / read: output callback [B samples]\n"
              << "   — cadence intentionally different; NOT a ring-buffer two-segment split)\n";
    std::cout << "  [STRUCTURAL] ring-wrap split & partition straddle: unreachable under I5 (P%B==0 and o_L%B==0)\n";
    bool p1xAllOk = true;
    for (const auto& s : p1xSweep)
    {
        const bool theoryMatch = (s.marginMin == s.theoryMarginMin);
        const bool marginOk    = (s.marginMin >= (long long) kBlockSize);   // I2 gate 合格の runtime 意味
        const bool effOk       = s.effConstant && s.actualEff == s.expectedEff;
        if (! marginOk || ! effOk || s.ope != 0 || s.i2 != 0) p1xAllOk = false;
        std::cout << "  " << std::left << std::setw (6) << s.caseName
                  << " irLen=" << std::setw (6) << s.irLen
                  << " L" << s.li
                  << " nIR=" << std::setw (3) << s.numPartsIR
                  << " ppc=" << s.ppc
                  << " lead=" << std::setw (5) << s.lead
                  << " margin=[" << s.marginMin << ".." << s.marginMax << "]"
                  << " theory(oL-lead)=" << s.theoryMarginMin
                  << " " << (theoryMatch ? "THEORY-MATCH" : "THEORY-DIFF")
                  << " | eff=" << s.actualEff << "(exp " << s.expectedEff << (effOk ? ",MATCH)" : ",DIFF)")
                  << " oPE=" << s.ope
                  << " i2=" << s.i2
                  << " M1=" << (p1xM1Verdict.count (s.caseName) ? p1xM1Verdict[s.caseName] : "n/a")
                  << (marginOk ? "" : " [MARGIN-FAIL]") << "\n";
    }
    std::cout << "  [STRUCTURAL] L1 lead_max = 896 (numPartsIR <= kL1MaxParts=64 -> ppc<=8 -> distCbs<=8) — saturation by cap\n";
    std::cout << "  [R3-b'] I2 boundary (margin==B) unreachable under current ppc design — scheduler-change-dependent\n";
    std::cout << "  P1-x sweep verdict: " << (p1xAllOk ? "PASS" : "FAIL") << "\n";

    std::cout << "\n=== EXIT: structural failures = " << structuralFailures << " ===\n";
    std::cout << "B13 判定（outputPlacementError）は上記 M2 行の値と判定帯（==0 / 1..64 / >=128）で Step 4 解析してください。\n";
    std::cout << "（exit code は構造的健全性のみを反映 — oPE 値は測定結果であり Step 4 の解析対象）\n";

    juce::shutdownJuce_GUI();
    return (structuralFailures == 0) ? 0 : 1;
}
