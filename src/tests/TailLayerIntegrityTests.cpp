//============================================================================
// TailLayerIntegrityTests.cpp — Part 8 F/H/I の Production 回帰テスト
//
// 目的:
//   * F: Bypass（tailMode=2）で IR 全体が厳密に再構成されること
//   * H: 既定 tailMode=1 のテール保持がブロック長に依存しないこと
//   * I: 長い IR でテール層の先頭パーティションが欠落しないこと
//
// 測定条件（実装準備仕様 §3.3）:
//   48 kHz、IR 0.5 / 1.0 / 3.0 秒、block 64〜32768、インパルス入力、
//   誤差はピーク整合後の max|out - ir|（倍精度）。
//   CPU は Add+Get の壁時計時間 ÷ 処理音声秒数（5 回の中央値）。
//   メモリは SetImpulse 前後の PrivateUsage 差分（1 インスタンス）。
//
// 出力: [PASS]/[FAIL] 行（受入 ID ごと）と測定値。FAIL は exit 1（fail-closed）。
//============================================================================
#include <JuceHeader.h>
#include "MKLNonUniformConvolver.h"
#include "tests/NUPCTestAccess.h"

#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
constexpr double kSampleRate = 48000.0;
constexpr double kPi         = 3.14159265358979323846;

// 受入許容（実装準備仕様 §3.2 / §3.3）
constexpr double kReconTolerance   = 1.0e-10; // F-1
constexpr double kTailEnergyTolDb  = 0.1;     // F-2 / H-1
constexpr double kWindowRatio      = 0.5;     // F-3
constexpr double kCpuLimitMsPerSec = 10.0;    // §3.3
constexpr double kMemLimitMB       = 32.0;    // §3.3

int gFailures = 0;

void report (const char* id, bool ok, const std::string& detail)
{
    std::printf ("[%s] %s %s\n", ok ? "PASS" : "FAIL", id, detail.c_str());
    if (! ok) ++gFailures;
}

std::string fmt (const char* format, ...)
{
    char buf[512];
    va_list args;
    va_start (args, format);
    std::vsnprintf (buf, sizeof (buf), format, args);
    va_end (args);
    return std::string (buf);
}

SIZE_T privateBytes()
{
    PROCESS_MEMORY_COUNTERS_EX pmc {};
    if (GetProcessMemoryInfo (GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*> (&pmc),
                              sizeof (pmc)) != 0)
        return pmc.PrivateUsage;
    return 0;
}

std::vector<double> makeIr (int irLen)
{
    std::vector<double> ir ((size_t) irLen, 0.0);
    for (int i = 0; i < irLen; ++i)
    {
        const double t = i / kSampleRate;
        ir[(size_t) i] = 0.5 * std::exp (-t / 0.15) * std::sin (2.0 * kPi * 997.0 * t);
    }
    ir[(size_t) (irLen - 1)] += 0.25;   // 末尾マーカー
    return ir;
}

double energyDb (const std::vector<double>& v, int from, int to)
{
    double e = 0.0;
    for (int i = from; i < to && i < (int) v.size(); ++i)
        e += v[(size_t) i] * v[(size_t) i];
    return e > 0.0 ? 10.0 * std::log10 (e) : -999.0;
}

double maxAbs (const std::vector<double>& v, int from, int to)
{
    double m = 0.0;
    for (int i = from; i < to && i < (int) v.size(); ++i)
        m = std::max (m, std::fabs (v[(size_t) i]));
    return m;
}

int argMaxAbs (const std::vector<double>& v)
{
    int best = 0;
    double m = -1.0;
    for (int i = 0; i < (int) v.size(); ++i)
    {
        const double a = std::fabs (v[(size_t) i]);
        if (a > m) { m = a; best = i; }
    }
    return best;
}

struct RunResult
{
    bool ok = false;
    int layers = 0;
    int partSize[3] { 0, 0, 0 };
    int numPartsIR[3] { 0, 0, 0 };
    int heldSamples = 0;
    int lag = 0;
    int firstErr = -1;
    double maxErr = 0.0;
    double tailEnergyDeltaDb = 0.0;
    double eFixedDb = 0.0;
    double minWindowRatio = 0.0;
    double cpuMsPerAudioSec = 0.0;
    double memDeltaMB = 0.0;
};

RunResult runCase (const std::vector<double>& ir, int block, bool tailEnabled)
{
    using convo::MKLNonUniformConvolver;

    RunResult r;
    const SIZE_T memBefore = privateBytes();

    MKLNonUniformConvolver conv;
    convo::FilterSpec spec {};
    spec.sampleRate = kSampleRate;
    spec.tailEnabled = tailEnabled;
    spec.tailMode = tailEnabled ? 1 : 2;

    if (! conv.SetImpulse (ir.data(), (int) ir.size(), block, 1.0, false, &spec))
        return r;

    const SIZE_T memAfter = privateBytes();
    r.ok = true;
    r.layers = convo::NUPCTestAccess::numActiveLayers (conv);
    for (int li = 0; li < r.layers && li < 3; ++li)
    {
        r.partSize[li]   = convo::NUPCTestAccess::layerPartSize (conv, li);
        r.numPartsIR[li] = convo::NUPCTestAccess::layerNumPartsIR (conv, li);
    }
    r.heldSamples = r.partSize[0] * r.numPartsIR[0];

    const int irLen = (int) ir.size();
    const int totalLen = ((irLen + 8 * block + block - 1) / block) * block;
    std::vector<double> in ((size_t) block, 0.0);
    std::vector<double> out ((size_t) totalLen, 0.0);

    const auto runOnce = [&]() -> double
    {
        for (int t = 0; t < totalLen; t += block)
        {
            for (int i = 0; i < block; ++i)
                in[(size_t) i] = ((t + i) == 0) ? 1.0 : 0.0;
            conv.Add (in.data(), block);
            conv.Get (out.data() + t, block);
        }
        return 0.0;
    };

    // 1 回目: 指標を取る（新しいインスタンス・残留なし）
    const auto t0 = std::chrono::steady_clock::now();
    runOnce();
    const auto t1 = std::chrono::steady_clock::now();
    const double firstMs = std::chrono::duration<double> (t1 - t0).count() * 1000.0;

    // 指標は 1 回目の出力から取る（2 回目以降は状態を持ち越すため時間計測にのみ使う）
    r.lag = argMaxAbs (out) - argMaxAbs (ir);

    double maxErr = 0.0;
    for (int i = 0; i < irLen; ++i)
    {
        const int j = i + r.lag;
        const double o = (j >= 0 && j < totalLen) ? out[(size_t) j] : 0.0;
        const double e = std::fabs (o - ir[(size_t) i]);
        if (e > maxErr) maxErr = e;
        if (e > 1.0e-6 && r.firstErr < 0) r.firstErr = i;
    }
    r.maxErr = maxErr;

    // 末尾 100 ms の保持（F-2）
    const int tailStart = std::max (0, irLen - (int) (0.1 * kSampleRate));
    r.eFixedDb = energyDb (out, r.lag + irLen / 2, r.lag + irLen);
    r.tailEnergyDeltaDb = energyDb (out, r.lag + tailStart, r.lag + irLen)
                        - energyDb (ir, tailStart, irLen);

    // 無音窓がないこと（F-3）
    double minRatio = 1.0e9;
    const int window = 8192;
    for (int w = 0; w < irLen; w += window)
    {
        const int b = std::min (irLen, w + window);
        const double o = maxAbs (out, r.lag + w, r.lag + b);
        const double i = maxAbs (ir, w, b);
        if (i > 1.0e-12) minRatio = std::min (minRatio, o / i);
    }
    r.minWindowRatio = (minRatio > 1.0e8) ? 0.0 : minRatio;

    std::vector<double> timings { firstMs };
    for (int rep = 0; rep < 4; ++rep)
    {
        const auto a = std::chrono::steady_clock::now();
        runOnce();
        const auto b = std::chrono::steady_clock::now();
        timings.push_back (std::chrono::duration<double> (b - a).count() * 1000.0);
    }
    std::sort (timings.begin(), timings.end());
    const double audioSec = (double) totalLen / kSampleRate;
    r.cpuMsPerAudioSec = timings[2] / audioSec;   // 5 回の中央値
    r.memDeltaMB = (double) (memAfter > memBefore ? memAfter - memBefore : 0) / (1024.0 * 1024.0);
    return r;
}

void checkF (const std::vector<double>& ir, int block)
{
    const auto r = runCase (ir, block, false);
    const int irLen = (int) ir.size();
    if (! r.ok)
    {
        report ("F-1", false, fmt ("block=%d ir=%d SetImpulse failed", block, irLen));
        return;
    }

    const int expectPart0 = juce::nextPowerOfTwo (std::max (block, 64));
    bool geomOk = r.layers >= 1 && r.layers <= 3 && r.partSize[0] == expectPart0;
    for (int li = 0; li < r.layers; ++li)
        geomOk = geomOk && r.partSize[li] > 0 && r.numPartsIR[li] >= 1;
    report ("F-4", geomOk, fmt ("block=%d ir=%d layers=%d l0=%d/%d",
                                block, irLen, r.layers, r.partSize[0], r.numPartsIR[0]));

    report ("F-1", r.maxErr <= kReconTolerance,
            fmt ("block=%d ir=%d maxErr=%.3e firstErr=%d", block, irLen, r.maxErr, r.firstErr));
    report ("F-2", std::fabs (r.tailEnergyDeltaDb) <= kTailEnergyTolDb,
            fmt ("block=%d ir=%d tailDelta=%.3f dB", block, irLen, r.tailEnergyDeltaDb));
    report ("F-3", r.minWindowRatio >= kWindowRatio,
            fmt ("block=%d ir=%d minWindowRatio=%.3f", block, irLen, r.minWindowRatio));
    report ("RES-F", r.cpuMsPerAudioSec <= kCpuLimitMsPerSec && r.memDeltaMB <= kMemLimitMB,
            fmt ("block=%d ir=%d cpu=%.3f ms/s mem=%.2f MB", block, irLen,
                 r.cpuMsPerAudioSec, r.memDeltaMB));
}

void checkH (int block, double baselineDb, double tolDb, const char* id)
{
    const auto ir = makeIr (48000);
    const auto r = runCase (ir, block, true);
    if (! r.ok)
    {
        report (id, false, fmt ("block=%d SetImpulse failed", block));
        return;
    }
    const bool ok = std::fabs (r.eFixedDb - baselineDb) <= tolDb;
    report (id, ok, fmt ("block=%d eFixed=%.3f dB baseline=%.3f dB tol=%.2f",
                         block, r.eFixedDb, baselineDb, tolDb));
}

void checkI (const std::vector<double>& ir, int block)
{
    const auto off = runCase (ir, block, false);
    report ("I-1", off.ok && off.firstErr < 0,
            fmt ("block=%d ir=%d firstErr=%d maxErr=%.3e",
                 block, (int) ir.size(), off.firstErr, off.maxErr));

    const auto on = runCase (ir, block, true);
    const bool geomOk = on.ok
                        && (on.layers == 1
                            || (on.partSize[1] <= 4 * on.partSize[0]
                                && on.partSize[1] <= on.heldSamples));
    report ("I-2", geomOk, fmt ("block=%d l0=%d/%d l1=%d/%d",
                                block, on.partSize[0], on.numPartsIR[0],
                                on.partSize[1], on.numPartsIR[1]));
}

void checkH4 (const std::vector<double>& ir, int block)
{
    const auto r = runCase (ir, block, true);
    report ("H-4", r.ok && r.eFixedDb > -60.0,
            fmt ("block=%d ir=%d eFixed=%.2f dB", block, (int) ir.size(), r.eFixedDb));
}

void checkGeometryAtMaxBlock()
{
    using convo::MKLNonUniformConvolver;
    std::vector<double> ir = makeIr (48000);
    MKLNonUniformConvolver conv;
    convo::FilterSpec spec {};
    spec.sampleRate = kSampleRate;
    spec.tailEnabled = true;
    spec.tailMode = 1;
    const bool ok = conv.SetImpulse (ir.data(), (int) ir.size(), 524288, 1.0, false, &spec);
    if (! ok)
    {
        report ("F-5", false, "block=524288 SetImpulse failed");
        return;
    }
    const int layers = convo::NUPCTestAccess::numActiveLayers (conv);
    const int p0 = convo::NUPCTestAccess::layerPartSize (conv, 0);
    const int n0 = convo::NUPCTestAccess::layerNumPartsIR (conv, 0);
    const bool geomOk = layers >= 1 && layers <= 3 && p0 == 524288 && n0 >= 1 && n0 <= 256;
    report ("F-5", geomOk, fmt ("block=524288 layers=%d l0=%d/%d", layers, p0, n0));
}
} // namespace

int main()
{
    juce::initialiseJuce_GUI();

    struct Case { int block; int irLen; };
    const Case cases[] = {
        { 64,   24000 }, { 64,   48000 }, { 64,   144000 },
        { 128,  24000 }, { 128,  48000 }, { 128,  144000 },
        { 256,  24000 }, { 256,  48000 }, { 256,  144000 },
        { 512,  24000 }, { 512,  48000 }, { 512, 144000 },
        { 1024, 24000 }, { 1024, 48000 }, { 1024, 144000 },
        { 2048, 24000 }, { 2048, 48000 }, { 2048, 144000 },
        { 4096, 24000 }, { 4096, 48000 }, { 4096, 144000 },
        { 8192, 24000 }, { 8192, 48000 }, { 8192, 144000 },
        { 16384, 24000 }, { 16384, 48000 }, { 16384, 144000 },
        { 32768, 24000 }, { 32768, 48000 }, { 32768, 144000 },
    };

    for (const auto& c : cases)
    {
        const auto ir = makeIr (c.irLen);
        checkF (ir, c.block);
    }
    checkGeometryAtMaxBlock();

    // ── I: block 512 の 3.0 秒 IR（先頭パーティション欠落の回帰） ──
    {
        const auto ir = makeIr (144000);
        checkI (ir, 512);
    }

    // ── H-4: 3.0 秒 IR のテール保持（無音区間がないこと） ──
    for (int b : { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 })
        checkH4 (makeIr (144000), b);

    // ── H: 既定 tailMode=1 のテール保持（IR 1.0 秒・固定区間 0.5〜1.0 秒） ──
    {
        const auto ir = makeIr (48000);
        const auto base = runCase (ir, 256, true);
        if (base.ok)
        {
            std::printf ("[record] H baseline block=256 eFixed=%.3f dB\n", base.eFixedDb);
            for (int b : { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 })
                checkH (b, base.eFixedDb, (b <= 128) ? 2.5 : 0.1, (b <= 128) ? "H-2" : "H-1");
        }
        else
        {
            report ("H-1", false, "baseline block=256 SetImpulse failed");
        }
    }

    std::printf ("[SUMMARY] failures=%d\n", gFailures);
    return (gFailures == 0) ? 0 : 1;
}



