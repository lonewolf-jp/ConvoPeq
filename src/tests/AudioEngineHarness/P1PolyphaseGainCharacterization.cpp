// P1PolyphaseGainCharacterization.cpp
// work113 P1-1（R12-8）: production chain における CONVOPEQ_CORRECT_POLYPHASE_GAIN
//   OFF / ON の characterization（measurement-only・production source 無変更）。
//
// 実行: AudioEngineHarness.exe --p1-char
//   P1-0-E で固定した測定 matrix（Level / Latency / SoftClip / Limiter）を
//   実 AudioEngine（AudioEngineHarness）上で実施する。
//   出力行は "[P1CHAR] ..." 固定 prefix（OFF/ON ログ比較を機械的に行うため）。

#include "AudioEngineHarness.h"
#include "audioengine/AudioEngine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(CONVOPEQ_CORRECT_POLYPHASE_GAIN) && (CONVOPEQ_CORRECT_POLYPHASE_GAIN)
    #define P1CHAR_FLAG_MACRO 1
#else
    #define P1CHAR_FLAG_MACRO 0
#endif

namespace {

constexpr double kSr         = 48000.0;
constexpr int    kBlock      = 512;
constexpr double kPi         = 3.14159265358979323846;
constexpr int    kWarmBlocks = 4;
constexpr int    kCapBlocks  = 12;
constexpr int    kAnalysisN  = 4096;          // 解析窓（capture の末尾）

// limiter / hard clamp 契約値（production: DSPCoreDouble.cpp:715-749）
constexpr double kLimThreshold = 0.8413951287507587;
constexpr double kHardClamp    = 0.8912509381337456;

// ── settle ヘルパ（BassBuzzMeasurement と同一方式・test-only） ──
void pumpMessages() noexcept
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

void sleepPump(int ms)
{
    const auto start = std::chrono::steady_clock::now();
    const auto budget = std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() - start < budget)
    {
        pumpMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

bool waitBacklogZero(AudioEngine& e, int timeoutMs)
{
    const auto start = std::chrono::steady_clock::now();
    const auto budget = std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() - start < budget)
    {
        if (e.getPublicationBacklogCount() == 0) return true;
        pumpMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return e.getPublicationBacklogCount() == 0;
}

bool waitWorldPublished(AudioEngine& e, long long before, int timeoutMs)
{
    const auto start = std::chrono::steady_clock::now();
    const auto budget = std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() - start < budget)
    {
        const long long cur = static_cast<long long>(e.getLastCommittedPublicationSequence());
        if (cur != before && e.getPublicationBacklogCount() == 0)
        {
            sleepPump(250);
            const long long cur2 = static_cast<long long>(e.getLastCommittedPublicationSequence());
            if (cur2 == cur && e.getPublicationBacklogCount() == 0) return true;
        }
        pumpMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

// ── 信号生成 + 回収（tap 内の確保・ロックは測定用途のみ） ──
enum class Signal { Sine, Impulse };

struct Capture
{
    std::mutex m;
    Signal signal = Signal::Sine;
    double freq = 1000.0;
    double amp  = 0.1;
    double phase = 0.0;
    int    blockIndex = 0;
    int    outBlocks = 0;          // 決定論的 capture 長（run 間の sample 比較を可能にする）
    bool   feed = true;
    bool   captureInput = false;
    std::vector<float> inL, outL;
};

void configureCapture(Capture& c, Signal sig, double freq, double amp, bool captureInput)
{
    std::lock_guard<std::mutex> lk(c.m);
    c.signal = sig;
    c.freq = freq;
    c.amp = amp;
    c.phase = 0.0;
    c.blockIndex = 0;
    c.outBlocks = 0;
    c.feed = true;
    c.captureInput = captureInput;
    c.inL.clear(); c.outL.clear();
    const std::size_t reserveN = static_cast<std::size_t>(kBlock) * (kWarmBlocks + kCapBlocks);
    c.inL.reserve(reserveN); c.outL.reserve(reserveN);
}

HarnessTapFn makeTap(Capture& c)
{
    return [&c](juce::AudioBuffer<float>& buffer, bool isInput)
    {
        const int n = buffer.getNumSamples();
        if (isInput)
        {
            float* L = buffer.getWritePointer(0);
            float* R = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : nullptr;
            std::lock_guard<std::mutex> lk(c.m);
            const int blk = c.blockIndex++;
            for (int i = 0; i < n; ++i)
            {
                double v = 0.0;
                if (c.feed)
                {
                    if (c.signal == Signal::Sine)
                    {
                        v = c.amp * std::sin(c.phase);
                        c.phase += 2.0 * kPi * c.freq / kSr;
                        if (c.phase >= 2.0 * kPi) c.phase -= 2.0 * kPi;
                    }
                    else if (blk == 0 && i == 0)
                    {
                        v = c.amp;   // impulse（1 サンプル）
                    }
                }
                L[i] = static_cast<float>(v);
                if (R != nullptr) R[i] = static_cast<float>(v);
                if (c.captureInput) c.inL.push_back(static_cast<float>(v));
            }
        }
        else
        {
            const float* L = buffer.getReadPointer(0);
            std::lock_guard<std::mutex> lk(c.m);
            if (c.outBlocks >= kCapBlocks) return;   // 余剰ブロックは捨てる（長さを固定）
            ++c.outBlocks;
            for (int i = 0; i < n; ++i) c.outL.push_back(L[i]);
        }
    };
}

// ── 解析（窓は capture 末尾 kAnalysisN） ──
double dftMag(const std::vector<float>& x, std::size_t from, int n, double freq)
{
    double re = 0.0, im = 0.0, wsum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double w = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i)
                                               / static_cast<double>(n - 1)));
        const double ph = 2.0 * kPi * freq * static_cast<double>(i) / kSr;
        const double v = static_cast<double>(x[from + static_cast<std::size_t>(i)]);
        re += v * w * std::cos(ph);
        im -= v * w * std::sin(ph);
        wsum += w;
    }
    return 2.0 * std::sqrt(re * re + im * im) / std::max(1.0e-30, wsum);
}

double maxAbsWindow(const std::vector<float>& x, std::size_t from, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::fabs(static_cast<double>(x[from + static_cast<std::size_t>(i)])));
    return m;
}

int countAtOrAboveWindow(const std::vector<float>& x, std::size_t from, int n, double level)
{
    int cnt = 0;
    for (int i = 0; i < n; ++i)
        if (std::fabs(static_cast<double>(x[from + static_cast<std::size_t>(i)])) >= level - 1.0e-12)
            ++cnt;
    return cnt;
}

// 2 つの capture の sample 差イベント数（|a-b| > 0）と max|a-b|
void diffStats(const std::vector<float>& a, const std::vector<float>& b,
               std::size_t from, int n, int& diffCount, double& maxDiff)
{
    diffCount = 0; maxDiff = 0.0;
    if (a.size() != b.size()) return;
    for (int i = 0; i < n; ++i)
    {
        const std::size_t idx = from + static_cast<std::size_t>(i);
        const double d = std::fabs(static_cast<double>(a[idx]) - static_cast<double>(b[idx]));
        if (d > 0.0) ++diffCount;
        maxDiff = std::max(maxDiff, d);
    }
}

// 出力ヘルパ（format string を使わない・1 行ごとに flush）
void emitLine(const std::string& s)
{
    std::cout << s << '\n';
    std::cout.flush();
}

std::string kv(const char* key, double value, int prec)
{
    std::ostringstream o;
    o << ' ' << key << '=' << std::fixed << std::setprecision(prec) << value;
    return o.str();
}

std::string kvi(const char* key, long long value)
{
    std::ostringstream o;
    o << ' ' << key << '=' << value;
    return o.str();
}

// flag 間比較用の sample 印刷（固定本数・1 行 16 個）
void printSamples(const char* caseId, const std::vector<float>& y, std::size_t from, int n)
{
    std::ostringstream head;
    head << "[P1CHAR] samples case=" << caseId << " n=" << n;
    emitLine(head.str());
    for (int i = 0; i < n; i += 16)
    {
        std::ostringstream row;
        row << "[P1CHAR] v";
        const int m = std::min(16, n - i);
        for (int k = 0; k < m; ++k)
            row << ' ' << std::setprecision(7)
                << static_cast<double>(y[from + static_cast<std::size_t>(i + k)]);
        emitLine(row.str());
    }
}

// ── 1 ケースの測定 ──
struct CaseOut
{
    double dftDb = 0.0;        // 出力（post-limiter）の DFT レベル
    double gainDb = 0.0;       // dftDb − ampDb（chain 伝達利得）
    double outMax = 0.0;
    double limiterInPeakDb = 0.0;   // getOutputLevel(): pre-limiter peak（peak 定義）
    int    limitingEngaged = 0;     // |y| ≥ θ（post-limiter・limiting 作用の proxy）
    int    hardClamp = 0;           // |y| ≥ hard clamp 値
    int    argmax = -1;
    int    reportedLatency = -1;
    bool   ok = false;
};

void configureChain(AudioEngine& e, int osFactor, convo::OversamplingType type,
                    bool softClip, float sat)
{
    e.beginBulkParameterRestore();
    // wet 経路（conv bypass + EQ active・identity）— P1-0-E の測定前提
    e.setEqBypassRequested(false);
    e.setConvolverBypassRequested(true);
    e.setAutoGainStagingEnabled(false);
    e.setInputHeadroomDb(0.0f);
    e.setOutputMakeupDb(0.0f);
    e.setConvolverInputTrimDb(0.0f);
    e.setSoftClipEnabled(softClip);
    e.setSaturationAmount(sat);
    e.setOversamplingFactor(osFactor);
    e.setOversamplingType(type);
    e.setDitherBitDepth(32);
    // identity EQ（全 band 無効・total 0dB・AGC off・saturation 0）
    for (int i = 0; i < 20; ++i)
    {
        e.setEQBandEnabled(i, false);
        e.setEQBandGain(i, 0.0f);
    }
    e.setEQTotalGain(0.0f);
    e.setEQAGCEnabled(false);
    e.setEQNonlinearSaturation(0.0f);
    e.endBulkParameterRestore(true);
}

bool runCase(AudioEngineHarness& h, Capture& cap, Signal sig, double freq, double ampDb,
             int osFactor, convo::OversamplingType type, bool softClip, float sat, CaseOut& out)
{
    AudioEngine& e = h.engine();
    const long long seqBefore = static_cast<long long>(e.getLastCommittedPublicationSequence());
    configureChain(e, osFactor, type, softClip, sat);
    if (!waitBacklogZero(e, 30000))
        emitLine(std::string("[P1CHAR] WARN backlog not zero os=") + std::to_string(osFactor)
                 + " sc=" + (softClip ? "1" : "0"));
    waitWorldPublished(e, seqBefore, 30000);
    sleepPump(800);

    const double amp = std::pow(10.0, ampDb / 20.0);
    configureCapture(cap, sig, freq, amp, /*captureInput*/ false);
    h.setTap(makeTap(cap));

    const long long b0 = h.blocksProcessed();
    const long long want = static_cast<long long>(kWarmBlocks + kCapBlocks);
    const auto start = std::chrono::steady_clock::now();
    while (h.blocksProcessed() - b0 < want)
    {
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(20)) break;
        pumpMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    sleepPump(30);
    h.clearTap();

    const std::size_t total = cap.outL.size();
    if (total < static_cast<std::size_t>(kAnalysisN)) return false;

    const std::size_t from = total - static_cast<std::size_t>(kAnalysisN);
    out.outMax = maxAbsWindow(cap.outL, from, kAnalysisN);
    out.hardClamp = countAtOrAboveWindow(cap.outL, from, kAnalysisN, kHardClamp);
    out.limitingEngaged = countAtOrAboveWindow(cap.outL, from, kAnalysisN, kLimThreshold);
    out.limiterInPeakDb = static_cast<double>(e.getOutputLevel());
    if (sig == Signal::Impulse)
    {
        int best = 0;
        double bestV = -1.0;
        for (std::size_t i = 0; i < total; ++i)
        {
            const double v = std::fabs(static_cast<double>(cap.outL[i]));
            if (v > bestV) { bestV = v; best = static_cast<int>(i); }
        }
        out.argmax = best;
        out.reportedLatency = e.getTotalLatencySamples();
    }
    else
    {
        const double mag = dftMag(cap.outL, from, kAnalysisN, freq);
        out.dftDb = 20.0 * std::log10(mag + 1.0e-300);
        out.gainDb = out.dftDb - ampDb;
    }
    out.ok = true;
    return true;
}

} // namespace

int runP1PolyphaseGainCharacterization(int argc, char* argv[])
{
    (void)argc; (void)argv;

    emitLine(std::string("[P1CHAR] flag_macro=") + std::to_string((int)P1CHAR_FLAG_MACRO)
             + " (0=OFF build / 1=ON build)");

    AudioEngineHarness h;
    if (!h.start(kSr, kBlock))
    {
        emitLine("[P1CHAR] FAIL harness start");
        return 1;
    }
    Capture cap;
    int cases = 0, failures = 0;

    // ── (A) Level: 3 factors × 3 freqs × 2 levels（+ factor8/1kHz/0dBFS record）──
    {
        const double freqs[] = { 50.0, 1000.0, 10000.0 };
        const double amps[]  = { -20.0, -6.0 };
        for (int osF : { 2, 4, 8 })
        {
            const int nStages = (osF == 2) ? 1 : ((osF == 4) ? 2 : 3);
            for (double fq : freqs)
            {
                for (double adb : amps)
                {
                    CaseOut o {};
                    if (!runCase(h, cap, Signal::Sine, fq, adb, osF, convo::OversamplingType::IIR,
                                 false, 0.0f, o))
                    { ++failures; continue; }
                    ++cases;
                    emitLine(std::string("[P1CHAR] kind=level") + kvi("os", osF)
                             + kvi("n", nStages) + " type=IIR sc=0"
                             + kv("freq", fq, 1) + kv("ampDb", adb, 1)
                             + kv("dftDb", o.dftDb, 4) + kv("gainDb", o.gainDb, 4)
                             + kv("outMax", o.outMax, 6) + kv("limiterInPeakDb", o.limiterInPeakDb, 2)
                             + kvi("limitingEngaged", o.limitingEngaged)
                             + kvi("hardClamp", o.hardClamp));
                }
            }
        }
        CaseOut o {};
        if (runCase(h, cap, Signal::Sine, 1000.0, 0.0, 8, convo::OversamplingType::IIR,
                    false, 0.0f, o))
        {
            ++cases;
            emitLine(std::string("[P1CHAR] kind=level os=8 n=3 type=IIR sc=0 freq=1000.0 ampDb=0.0")
                     + kv("dftDb", o.dftDb, 4) + kv("gainDb", o.gainDb, 4)
                     + kv("outMax", o.outMax, 6) + kv("limiterInPeakDb", o.limiterInPeakDb, 2)
                     + kvi("limitingEngaged", o.limitingEngaged) + kvi("hardClamp", o.hardClamp)
                     + " (record: limiter 接触域)");
        }
        else ++failures;
    }

    // ── (B) Latency: reported + impulse argmax ×4 経路 ──
    {
        struct Route { int osF; convo::OversamplingType type; const char* name; };
        const Route routes[] = {
            { 2, convo::OversamplingType::IIR,         "os2-IIR-N1" },
            { 4, convo::OversamplingType::IIR,         "os4-IIR-N2" },
            { 8, convo::OversamplingType::IIR,         "os8-IIR-N3" },
            { 8, convo::OversamplingType::LinearPhase, "os8-LP-N3"  },
        };
        for (const Route& r : routes)
        {
            CaseOut o {};
            if (!runCase(h, cap, Signal::Impulse, 0.0, -12.0, r.osF, r.type, false, 0.0f, o))
            { ++failures; continue; }
            ++cases;
            emitLine(std::string("[P1CHAR] kind=latency route=") + r.name
                     + kvi("reported", o.reportedLatency) + kvi("argmax", o.argmax)
                     + kv("outMax", o.outMax, 6));
        }
    }

    // ── (C) SoftClip / Limiter: 2 factors × 2 levels × sc{0,1}（sat=1.0）──
    {
        for (int osF : { 2, 8 })
        {
            const int nStages = (osF == 2) ? 1 : 3;
            for (double adb : { -6.0, 0.0 })
            {
                std::vector<float> yScOff, yScOn;
                CaseOut oOff {}, oOn {};
                const bool okOff = runCase(h, cap, Signal::Sine, 1000.0, adb, osF,
                                           convo::OversamplingType::IIR, false, 1.0f, oOff);
                if (okOff) yScOff = cap.outL;
                const bool okOn = runCase(h, cap, Signal::Sine, 1000.0, adb, osF,
                                          convo::OversamplingType::IIR, true, 1.0f, oOn);
                if (okOn) yScOn = cap.outL;
                if (!okOff || !okOn) { ++failures; continue; }
                ++cases;

                int engDiff = 0; double engMax = 0.0;
                if (yScOff.size() == yScOn.size() && !yScOff.empty())
                {
                    const std::size_t from = yScOff.size() - static_cast<std::size_t>(kAnalysisN);
                    diffStats(yScOn, yScOff, from, kAnalysisN, engDiff, engMax);
                }
                emitLine(std::string("[P1CHAR] kind=softclip") + kvi("os", osF) + kvi("n", nStages)
                         + kv("ampDb", adb, 1) + " sat=1.00"
                         + kv("outMax_sc0", oOff.outMax, 6) + kv("outMax_sc1", oOn.outMax, 6)
                         + kv("limiterInPeakDb_sc1", oOn.limiterInPeakDb, 2)
                         + kvi("limitingEngaged_sc0", oOff.limitingEngaged)
                         + kvi("limitingEngaged_sc1", oOn.limitingEngaged)
                         + kvi("hardClamp_sc0", oOff.hardClamp)
                         + kvi("hardClamp_sc1", oOn.hardClamp)
                         + kvi("clipEngagement", engDiff) + kv("clipEngMax", engMax, 6)
                         + kvi("capN_sc0", static_cast<long long>(yScOff.size()))
                         + kvi("capN_sc1", static_cast<long long>(yScOn.size())));
            }
        }
    }

    // ── (D) flag 間 sample 差解析用の固定 dump（os8 / ampDb=-6 / 1024 サンプル）──
    {
        CaseOut o {};
        if (runCase(h, cap, Signal::Sine, 1000.0, -6.0, 8, convo::OversamplingType::IIR,
                    true, 1.0f, o) && cap.outL.size() >= 1024u)
        {
            printSamples("os8_am6_sc1", cap.outL, cap.outL.size() - 1024u, 1024);
            ++cases;
        }
        else ++failures;
    }

    h.stop();
    emitLine(std::string("[P1CHAR] summary") + kvi("flag_macro", (int)P1CHAR_FLAG_MACRO)
             + kvi("cases", cases) + kvi("failures", failures));
    return failures == 0 ? 0 : 1;
}
