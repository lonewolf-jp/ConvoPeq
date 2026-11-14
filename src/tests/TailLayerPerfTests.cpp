//============================================================================
// TailLayerPerfTests.cpp — RES-F（CPU 性能予算）の Release 専用テスト
//
// 目的:
//   * 仕様 §3.3「測定条件・集計方法（Owner 条件 4）」に測定方法を一致させる
//     （doc/work113/20261109-1500_実装準備仕様/report.md）
//   * 基準構成は MSVC Release x64。Debug では CPU を評価しない（明示 SKIP）
//
// 仕様 §3.3 の測定定義:
//   CPU = `Add` + `Get` の総壁時計時間 ÷ 処理した音声秒数
//         5 回の中央値（初回のウォームアップは除外）
//   基準構成 = MSVC Release x64、1 スレッド、同一マシン、
//              他プロセスの負荷を排除した状態
//   許容値   = CPU ≤ 10 ms/音声 1 秒、メモリ増分 ≤ 32 MB
//
// 本テストの追加条件（測定区間の短時間化を防ぐ）:
//   * 1 サンプルあたり最低 5 音声秒を測定する
//   * 外乱検出（壁時計 / プロセス CPU 時間 > 1.25）で再測定し、3 回とも外乱なら FAIL
//   * プロセス CPU 時間は外乱判定と記録にのみ使い、合否判定には使わない
//
// 出力: [PASS]/[FAIL] RES-F 行、[DIAG] 行（5 サンプル・中央値・音声秒・比・試行回数）。
//       FAIL は exit 1（fail-closed）。Debug 構成では [SKIP] を出して 0 で終了する。
//============================================================================
#include <JuceHeader.h>
#include "MKLNonUniformConvolver.h"

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

// 受入許容（実装準備仕様 §3.3）。値は変更しない。
constexpr double kCpuLimitMsPerSec = 10.0;
constexpr double kMemLimitMB       = 32.0;

// 測定条件（本テストの追加条件）
constexpr int    kSamples               = 5;     // 仕様: 5 回の中央値
constexpr double kMinAudioSecPerSample  = 5.0;   // 1 サンプルあたり最低音声秒
constexpr int    kMaxAttempts           = 5;     // 外乱時の再測定回数
constexpr double kDisturbWallCpuRatio   = 1.25;  // 壁時計 / CPU 時間の上限
constexpr double kDisturbSpread         = 1.25;  // 5 サンプルの max/min の上限

int gFailures = 0;

void report (const char* id, bool ok, const std::string& detail)
{
    std::printf ("[%s] %s %s\n", ok ? "PASS" : "FAIL", id, detail.c_str());
    if (! ok) ++gFailures;
}

std::string fmt (const char* format, ...)
{
    char buf[1024];
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

// プロセス CPU 時間（kernel + user）。外乱判定と記録にのみ使う。
double processCpuSeconds()
{
    FILETIME created {}, exited {}, kernel {}, user {};
    if (GetProcessTimes (GetCurrentProcess(), &created, &exited, &kernel, &user) == 0)
        return -1.0;

    const auto toSeconds = [] (const FILETIME& ft) -> double
    {
        const unsigned long long v =
            ((unsigned long long) ft.dwHighDateTime << 32) | (unsigned long long) ft.dwLowDateTime;
        return (double) v / 1.0e7;
    };
    return toSeconds (kernel) + toSeconds (user);
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

struct PerfResult
{
    bool   ok = false;
    double cpuMsPerAudioSec = 0.0;   // 5 回の中央値
    double memDeltaMB = 0.0;
    double samples[kSamples] {};
    double audioPerSample = 0.0;
    double wallCpuRatio = 0.0;
    double spread = 1.0;
    int    attempts = 0;
    bool   disturbed = false;
};

PerfResult runCase (const std::vector<double>& ir, int block)
{
    using convo::MKLNonUniformConvolver;

    PerfResult r;
    const SIZE_T memBefore = privateBytes();

    MKLNonUniformConvolver conv;
    convo::FilterSpec spec {};
    spec.sampleRate = kSampleRate;
    spec.tailEnabled = true;
    spec.tailMode = 1;

    if (! conv.SetImpulse (ir.data(), (int) ir.size(), block, 1.0, false, &spec))
        return r;

    const SIZE_T memAfter = privateBytes();
    r.ok = true;
    r.memDeltaMB = (double) (memAfter > memBefore ? memAfter - memBefore : 0) / (1024.0 * 1024.0);

    const int irLen = (int) ir.size();
    const int totalLen = ((irLen + 8 * block + block - 1) / block) * block;
    std::vector<double> in ((size_t) block, 0.0);
    std::vector<double> out ((size_t) totalLen, 0.0);

    const auto runOnce = [&]()
    {
        for (int t = 0; t < totalLen; t += block)
        {
            for (int i = 0; i < block; ++i)
                in[(size_t) i] = ((t + i) == 0) ? 1.0 : 0.0;
            conv.Add (in.data(), block);
            conv.Get (out.data() + t, block);
        }
    };

    const double audioPerRun = (double) totalLen / kSampleRate;
    const int repsPerSample =
        std::max (1, (int) std::ceil (kMinAudioSecPerSample / audioPerRun));
    r.audioPerSample = audioPerRun * repsPerSample;

    for (int attempt = 1; attempt <= kMaxAttempts; ++attempt)
    {
        r.attempts = attempt;

        // 仕様 §3.3: 初回はウォームアップとして除外する（中央値に含めない）
        runOnce();

        double samples[kSamples] {};
        const double cpu0 = processCpuSeconds();
        const auto wall0 = std::chrono::steady_clock::now();

        for (int s = 0; s < kSamples; ++s)
        {
            const auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < repsPerSample; ++k)
                runOnce();
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double> (t1 - t0).count() * 1000.0;
            samples[s] = ms / r.audioPerSample;   // ms / 音声 1 秒
        }

        const auto wall1 = std::chrono::steady_clock::now();
        const double cpu1 = processCpuSeconds();

        double sorted[kSamples] {};
        std::copy (samples, samples + kSamples, sorted);
        std::sort (sorted, sorted + kSamples);
        r.cpuMsPerAudioSec = sorted[kSamples / 2];   // 5 回の中央値（添字 2）
        std::copy (samples, samples + kSamples, r.samples);

        // 壁時計 / プロセス CPU 時間の比は試行全体（ウォームアップ + 5 サンプル）で取る。
        // サンプル単位では Windows の CPU 時間がタイマー刻み（既定 15.6 ms）で量子化され、
        // 20〜30 ms の窓では比が 1.5 や 0.8 に飛ぶため使えない。
        const double wallMs = std::chrono::duration<double> (wall1 - wall0).count() * 1000.0;
        const double cpuMs = (cpu0 > 0.0 && cpu1 > 0.0) ? (cpu1 - cpu0) * 1000.0 : -1.0;
        r.wallCpuRatio = (cpuMs > 0.0) ? wallMs / cpuMs : 0.0;

        r.spread = (sorted[0] > 0.0) ? (sorted[kSamples - 1] / sorted[0]) : 1.0e9;

        // 外乱の判定は 2 系統。
        //   (1) 壁時計 / CPU 時間の比: スケジューリング待ち（待たされた）を検出する。
        //   (2) 5 サンプルの max/min: 競合で実行速度そのものが変動する場合を検出する。
        //       負荷時は壁時計と CPU 時間が同じだけ増えるため、(1) だけでは検出できない。
        r.disturbed = (cpuMs <= 0.0)
                   || (r.wallCpuRatio > kDisturbWallCpuRatio)
                   || (r.spread > kDisturbSpread);

        if (! r.disturbed)
            break;

        std::printf ("[RETRY] block=%d ir=%d attempt=%d disturbed (wall/cpu=%.3f spread=%.3f)\n",
                     block, irLen, attempt, r.wallCpuRatio, r.spread);
    }

    return r;
}

void runPerfCase (const std::vector<double>& ir, int block)
{
    const auto r = runCase (ir, block);
    const int irLen = (int) ir.size();

    if (! r.ok)
    {
        report ("RES-F", false, fmt ("block=%d ir=%d SetImpulse failed", block, irLen));
        return;
    }

    const bool cpuOk = ! r.disturbed && r.cpuMsPerAudioSec <= kCpuLimitMsPerSec;
    const bool memOk = r.memDeltaMB <= kMemLimitMB;
    report ("RES-F", cpuOk && memOk,
            fmt ("block=%d ir=%d cpu=%.3f ms/s mem=%.2f MB", block, irLen,
                 r.cpuMsPerAudioSec, r.memDeltaMB));

    std::printf ("[DIAG] block=%d ir=%d samples=[%.3f %.3f %.3f %.3f %.3f] median=%.3f "
                 "audio/sample=%.2fs wall/cpu=%.3f spread=%.3f attempts=%d%s\n",
                 block, irLen, r.samples[0], r.samples[1], r.samples[2], r.samples[3],
                 r.samples[4], r.cpuMsPerAudioSec, r.audioPerSample, r.wallCpuRatio,
                 r.spread, r.attempts, r.disturbed ? " DISTURBED" : "");

    if (r.disturbed)
        std::printf ("[INFO] block=%d ir=%d: %d 回とも外乱（wall/cpu=%.3f > %.2f）。"
                     "他プロセスの負荷を排除した状態で再実行すること\n",
                     block, irLen, r.attempts, r.wallCpuRatio, kDisturbWallCpuRatio);
}
} // namespace

int main()
{
#if ! defined (NDEBUG)
    // 仕様 §3.3 の基準構成は MSVC Release x64。Debug では CPU を評価しない。
    // PASS と表示して評価を隠さないため、明示的に SKIP を出す。
    std::printf ("[SKIP] RES-F (Debug build: CPU budget is evaluated in Release by "
                 "TailLayerPerfTests)\n");
    return 0;
#else
    juce::initialiseJuce_GUI();

    std::printf ("[INFO] RES-F performance test (spec §3.3: Add+Get wall clock / audio seconds, "
                 "median of %d samples after 1 warm-up, >= %.1f audio sec per sample)\n",
                 kSamples, kMinAudioSecPerSample);

    struct Case { int block; int irLen; };
    const Case cases[] = {
        { 64,   24000 }, { 64,   48000 }, { 64,   144000 },
        { 128,  24000 }, { 128,  48000 }, { 128,  144000 },
        { 256,  24000 }, { 256,  48000 }, { 256,  144000 },
        { 512,  24000 }, { 512,  48000 }, { 512,  144000 },
        { 1024, 24000 }, { 1024, 48000 }, { 1024, 144000 },
        { 2048, 24000 }, { 2048, 48000 }, { 2048, 144000 },
        { 4096, 24000 }, { 4096, 48000 }, { 4096, 144000 },
        { 8192, 24000 }, { 8192, 48000 }, { 8192, 144000 },
        { 16384, 24000 }, { 16384, 48000 }, { 16384, 144000 },
        { 32768, 24000 }, { 32768, 48000 }, { 32768, 144000 },
    };

    for (const auto& c : cases)
        runPerfCase (makeIr (c.irLen), c.block);

    std::printf ("[SUMMARY] failures=%d\n", gFailures);
    return (gFailures == 0) ? 0 : 1;
#endif
}
