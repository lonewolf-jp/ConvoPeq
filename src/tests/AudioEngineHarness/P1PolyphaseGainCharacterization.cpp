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
#include <cstring>
#include <fstream>
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
// P1-5: heavy matrix（EQ/IR/staging/sat/limiter sweep）の実行可否。
//   false のときは preset replay（G）と listening materials（M）のみを実行する
//   （1 config-pair あたり約 37 s の実測レートでは全行列が 1 build あたり 90 分超になるため）。
constexpr bool   kP15FullMatrix = true;
// P3-5: g0/os1 限定 vehicle（test-only）。
//   true のとき section I の g0/os1 のみを実行し、他 section（A-H,J-M）を抑止する。
//   変更禁止: kP15FullMatrix・production・settle/sleep の既存値。
constexpr bool   kP15G0OS1Only = true;
// P3-5: bounded post-settle（P3-4 §5）。
//   crossfade/ramp（既定0.030 s≒2.8 blocks）＋arm遅延に対する時間的余裕。
//   ramp完了の観測・証明ではない。既存 sleepPump(800) は変更しない。
constexpr int    kP35PostSettleMs = 500;

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

// P3-5-R1: publish-timeout diagnostic 用の前方宣言・flag（test-only・既存WARN機構のみ使用）。
void emitLine(const std::string& s);
static bool gP35PubDiag = false;
static long long gP35PubDiagNextMarkMs = 0;

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
        if (gP35PubDiag)
        {
            const long long elapsedMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count();
            if (elapsedMs >= gP35PubDiagNextMarkMs)
            {
                emitLine(std::string("[P1CHAR] WARN pubdiag t=") + std::to_string(elapsedMs)
                         + " before=" + std::to_string(before)
                         + " seq=" + std::to_string(cur)
                         + " backlog=" + std::to_string((long long)e.getPublicationBacklogCount()));
                gP35PubDiagNextMarkMs += 5000;
            }
        }
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
enum class Signal { Sine, Impulse, Program };

struct Capture
{
    std::mutex m;
    Signal signal = Signal::Sine;
    double freq = 1000.0;
    double amp  = 0.1;
    double phase = 0.0;
    int    blockIndex = 0;
    int    outBlocks = 0;          // 決定論的 capture 長（run 間の sample 比較を可能にする）
    int    capBlocks = kCapBlocks; // capture するブロック数（Program 素材では延長）
    unsigned long long progSeed = 0x2545F4914F6CDD1DULL;
    bool   feed = true;
    bool   captureInput = false;
    std::vector<float> inL, outL;
};

void configureCapture(Capture& c, Signal sig, double freq, double amp, bool captureInput,
                      int capBlocks = kCapBlocks)
{
    std::lock_guard<std::mutex> lk(c.m);
    c.signal = sig;
    c.freq = freq;
    c.amp = amp;
    c.phase = 0.0;
    c.blockIndex = 0;
    c.outBlocks = 0;
    c.capBlocks = capBlocks;
    c.progSeed = 0x2545F4914F6CDD1DULL;
    c.feed = true;
    c.captureInput = captureInput;
    c.inL.clear(); c.outL.clear();
    const std::size_t reserveN = static_cast<std::size_t>(kBlock) * (kWarmBlocks + capBlocks);
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
                    else if (c.signal == Signal::Program)
                    {
                        const long long idx = static_cast<long long>(blk) * n + i;
                        const double t = static_cast<double>(idx) / kSr;
                        const double g = c.amp * 2.0;   // ampDb=-6 で nominal（正弦 0.5 / click 0.9 / noise 0.25）
                        if (t < 0.6) v = g * 0.5 * std::sin(2.0 * kPi * 1000.0 * t);
                        else if (t < 0.6005) v = g * 0.9 * ((idx & 1) ? -1.0 : 1.0);
                        else if (t < 1.6)
                        {
                            c.progSeed = c.progSeed * 6364136223846793005ULL + 1442695040888963407ULL;
                            v = g * 0.25 * ((static_cast<double>((c.progSeed >> 11) & 0xFFFFFFULL) / 8388608.0) - 1.0);
                        }
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
            if (c.outBlocks >= c.capBlocks) return;   // 余剰ブロックは捨てる（長さを固定）
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

// P3-5-R10: active snapshot の token 化（既存 kv/kvi 形式に合わせる。新 logger 機構なし）。
// requested/active の解釈・比較は監査側で行い、vehicle 内では行わない。
std::string snapTokens(const char* p, const AudioEngine::RuntimeActiveSnapshot& s)
{
    std::ostringstream o;
    o << ' ' << p << "_gen=" << (long long)s.generation
      << ' ' << p << "_wid=" << (long long)s.worldId
      << ' ' << p << "_seq=" << (long long)s.publicationSequence
      << ' ' << p << "_ord=" << s.processingOrder
      << ' ' << p << "_eqb=" << (s.eqBypassed ? 1 : 0)
      << ' ' << p << "_cvb=" << (s.convBypassed ? 1 : 0)
      << ' ' << p << "_sc=" << (s.softClipEnabled ? 1 : 0);
    o << std::fixed << std::setprecision(4);
    o << ' ' << p << "_sat=" << (double)s.saturationAmount
      << ' ' << p << "_hr=" << (double)s.inputHeadroomGain
      << ' ' << p << "_mu=" << (double)s.outputMakeupGain
      << ' ' << p << "_tr=" << (double)s.convolverInputTrimGain;
    o << ' ' << p << "_os=" << s.oversamplingFactor
      << ' ' << p << "_irl=" << (s.irLoaded ? 1 : 0)
      << ' ' << p << "_irf=" << (s.irFinalized ? 1 : 0);
    o << ' ' << p << "_hash=" << (long long)s.structuralHash;
    o << ' ' << p << "_fade=" << (double)s.fadeTimeSec;
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

// ── AB/ABX 用 WAV 出力（固定リテラルパス・16-bit PCM stereo）──
bool writeWav16(const char* path, const std::vector<float>& mono, int sampleRate)
{
    if (path == nullptr || mono.empty()) return false;
    const unsigned int frames = static_cast<unsigned int>(mono.size());
    const unsigned int dataBytes = frames * 4u;   // stereo 16-bit
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) return false;
    auto u32 = [&f](unsigned int v)
    { const char b[4] = { char(v & 0xFF), char((v >> 8) & 0xFF), char((v >> 16) & 0xFF), char((v >> 24) & 0xFF) }; f.write(b, 4); };
    auto u16 = [&f](unsigned short v)
    { const char b[2] = { char(v & 0xFF), char((v >> 8) & 0xFF) }; f.write(b, 2); };
    f.write("RIFF", 4); u32(36u + dataBytes); f.write("WAVE", 4);
    f.write("fmt ", 4); u32(16u); u16(1); u16(2); u32(static_cast<unsigned int>(sampleRate));
    u32(static_cast<unsigned int>(sampleRate) * 4u); u16(4); u16(16);
    f.write("data", 4); u32(dataBytes);
    for (unsigned int i = 0; i < frames; ++i)
    {
        const double clamped = std::max(-1.0, std::min(1.0, static_cast<double>(mono[i])));
        const short s = static_cast<short>(std::lround(clamped * 32767.0));
        u16(static_cast<unsigned short>(s));
        u16(static_cast<unsigned short>(s));
    }
    f.close();
    return true;
}

// 聴感評価素材の出力先（固定リテラルのみ・可変パス組み立てなし）
const char* listenPath(const char* cat, bool on)
{
    if (on)
    {
        if (std::strcmp(cat, "dry") == 0)  return "tmp/p15_listen_dry_on.wav";
        if (std::strcmp(cat, "eqid") == 0) return "tmp/p15_listen_eqid_on.wav";
        if (std::strcmp(cat, "eq6") == 0)  return "tmp/p15_listen_eq6_on.wav";
        if (std::strcmp(cat, "ir") == 0)   return "tmp/p15_listen_ir_on.wav";
        if (std::strcmp(cat, "sc") == 0)   return "tmp/p15_listen_sc_on.wav";
        if (std::strcmp(cat, "lim") == 0)  return "tmp/p15_listen_lim_on.wav";
    }
    else
    {
        if (std::strcmp(cat, "dry") == 0)  return "tmp/p15_listen_dry_off.wav";
        if (std::strcmp(cat, "eqid") == 0) return "tmp/p15_listen_eqid_off.wav";
        if (std::strcmp(cat, "eq6") == 0)  return "tmp/p15_listen_eq6_off.wav";
        if (std::strcmp(cat, "ir") == 0)   return "tmp/p15_listen_ir_off.wav";
        if (std::strcmp(cat, "sc") == 0)   return "tmp/p15_listen_sc_off.wav";
        if (std::strcmp(cat, "lim") == 0)  return "tmp/p15_listen_lim_off.wav";
    }
    return nullptr;
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
    // P3-5-R10: T6/T9 active snapshot（値コピーのみ。所有権なし）。
    AudioEngine::RuntimeActiveSnapshot snapBefore {};
    AudioEngine::RuntimeActiveSnapshot snapAfter {};
};

void configureChain(AudioEngine& e, int osFactor, convo::OversamplingType type,
                    bool softClip, float sat, float headroomDb,
                    float makeupDb = 0.0f, float trimDb = 0.0f,
                    float eqBoostDb = 0.0f, bool convBypass = true, bool eqBypass = false)
{
    e.beginBulkParameterRestore();
    // wet 経路（EQ active・conv は convBypass 指定に従う）— P1-0-E の測定前提
    e.setEqBypassRequested(eqBypass);
    e.setConvolverBypassRequested(convBypass);
    e.setAutoGainStagingEnabled(false);
    e.setInputHeadroomDb(headroomDb);
    e.setOutputMakeupDb(makeupDb);
    e.setConvolverInputTrimDb(trimDb);
    e.setSoftClipEnabled(softClip);
    e.setSaturationAmount(sat);
    e.setOversamplingFactor(osFactor);
    e.setOversamplingType(type);
    e.setDitherBitDepth(32);
    // EQ: eqBoostDb > 0 のとき band 8（1 kHz）のみ boost、それ以外は identity
    for (int i = 0; i < 20; ++i)
    {
        e.setEQBandEnabled(i, false);
        e.setEQBandGain(i, 0.0f);
    }
    if (eqBoostDb > 0.0f)
    {
        e.setEQBandFrequency(8, 1000.0f);
        e.setEQBandQ(8, 0.707f);
        e.setEQBandType(8, static_cast<EQBandType>(1));
        e.setEQBandGain(8, eqBoostDb);
        e.setEQBandEnabled(8, true);
    }
    e.setEQTotalGain(0.0f);
    e.setEQAGCEnabled(false);
    e.setEQNonlinearSaturation(0.0f);
    e.endBulkParameterRestore(true);
}

// test IR（固定リテラルパス）のロードと finalize 待ち（BassBuzzMeasurement と同一方式）
bool ensureTestIr(AudioEngineHarness& h, const juce::File& irFile)
{
    AudioEngine& e = h.engine();
    e.getConvolverProcessor().loadImpulseResponse(irFile, false);
    for (int i = 0; i < 3000; ++i)
    {
        if (e.getConvolverProcessor().isIRFinalized()) return true;
        pumpMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return e.getConvolverProcessor().isIRFinalized();
}

bool runCase(AudioEngineHarness& h, Capture& cap, Signal sig, double freq, double ampDb,
             int osFactor, convo::OversamplingType type, bool softClip, float sat, CaseOut& out,
             float headroomDb = 0.0f, float makeupDb = 0.0f, float trimDb = 0.0f,
             float eqBoostDb = 0.0f, bool convBypass = true, int capBlocks = kCapBlocks,
             bool eqBypass = false, bool requirePublish = false)
{
    AudioEngine& e = h.engine();
    // P3-5-R17: queue/take/publish delta observation（既存 getter＋WARN 行のみ）。
    // 新規 logger/prefix family なし。絶対値を記録し delta は監査側で算出する。
    // P3-5-R19: buildResult を追加（B2 到達観測。validation／commit 含まず）。
    // P3-5-R22: commit-enqueue を追加（main-site 到達観測。recovery 含まず）。
    // P3-5-R23: lastDroppedGeneration を追加（既存値の観測のみ。新規drop計装なし）。
    // P3-5-R27: coordinatorTake を追加（Main-origin pop 観測。Recovery 含まず・R26-A path A）。
    // build-result／commit／health／pressure の新規計装は追加しない（R17 §5遵守）。
    auto emitQDelta = [&](const char* tag)
    {
        const auto d = e.getRebuildDispatchDiagnostics();
        const auto lc = e.getRuntimeLifecycleDiagnostics();
        emitLine(std::string("[P1CHAR] WARN qdelta tag=") + tag
                 + kvi("req", (long long)d.requestCount)
                 + kvi("que", (long long)d.queuedCount)
                 + kvi("dup", (long long)d.blockedPendingDuplicateCount)
                 + kvi("take", (long long)e.getRebuildTakeCount())
                 + kvi("bld", (long long)e.getRebuildBuildResultCount())
                 + kvi("cmt", (long long)e.getRebuildCommitEnqueueCount())
                 + kvi("coord", (long long)e.getCoordinatorTakeCount())
                 + kvi("drp", (long long)lc.lastDroppedGeneration)
                 + kvi("seq", (long long)e.getLastCommittedPublicationSequence())
                 + kvi("blo", (long long)e.getPublicationBacklogCount()));
    };
    emitQDelta("pre");
    const long long seqBefore = static_cast<long long>(e.getLastCommittedPublicationSequence());
    configureChain(e, osFactor, type, softClip, sat, headroomDb, makeupDb, trimDb,
                   eqBoostDb, convBypass, eqBypass);
    if (!waitBacklogZero(e, 30000))
        emitLine(std::string("[P1CHAR] WARN backlog not zero os=") + std::to_string(osFactor)
                 + " sc=" + (softClip ? "1" : "0"));
    if (requirePublish)
    {
        // P3-5: 第1ケース strict-honor。publish未確認状態では capture しない。
        // 成功条件は sequence変化＋backlog==0 であり、今回configureChainの
        // publishを意味しない（P3-4 §4-2）。false時は runPair 側で failure 扱い。
        // P3-5-R1: 待機中の seq/backlog を既存WARN行で観測する（test-only diagnostic）。
        gP35PubDiag = true;
        gP35PubDiagNextMarkMs = 0;
        const bool okPublish = waitWorldPublished(e, seqBefore, 30000);
        gP35PubDiag = false;
        emitQDelta("post");
        if (!okPublish)
        {
            emitLine(std::string("[P1CHAR] WARN publish not confirmed os=") + std::to_string(osFactor)
                     + " sc=" + (softClip ? "1" : "0"));
            return false;
        }
    }
    else if (kP15G0OS1Only)
    {
        // P3-5: 第2ケース以降は同一parameterのため merge-no-dispatch を許容する。
        // sequence-unchanged＋backlog==0 の確認のうち backlog のみ既存WARN機構で記録する。
        // 「publish成功」とは扱わない（P3-4 §4-2・P3-5 §3B）。
        if (e.getPublicationBacklogCount() != 0)
            emitLine(std::string("[P1CHAR] WARN backlog not zero (inherited-world check) os=")
                     + std::to_string(osFactor) + " sc=" + (softClip ? "1" : "0"));
        emitQDelta("post");
    }
    else
    {
        waitWorldPublished(e, seqBefore, 30000);
    }
    sleepPump(800);
    if (kP15G0OS1Only)
    {
        // P3-5: bounded post-settle（P3-4 §5）。
        // crossfade/rampに対する時間的余裕であり、ramp完了の観測・証明ではない。
        sleepPump(kP35PostSettleMs);
    }

    // P3-5-R10 T6: capture直前の active snapshot（瞬間read・保持しない）。
    out.snapBefore = e.getActiveRuntimeSnapshot();

    const double amp = std::pow(10.0, ampDb / 20.0);
    configureCapture(cap, sig, freq, amp, /*captureInput*/ false, capBlocks);
    h.setTap(makeTap(cap));

    const long long b0 = h.blocksProcessed();
    const long long want = static_cast<long long>(kWarmBlocks + capBlocks);
    const auto start = std::chrono::steady_clock::now();
    while (h.blocksProcessed() - b0 < want)
    {
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(20)) break;
        pumpMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    sleepPump(30);
    h.clearTap();

    // P3-5-R10 T9: capture直後の active snapshot（瞬間read・保持しない）。
    out.snapAfter = e.getActiveRuntimeSnapshot();

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
    // P3-5: g0/os1限定時は抑止。
    if (!kP15G0OS1Only)
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
    // P3-5: g0/os1限定時は抑止。
    if (!kP15G0OS1Only)
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
    // P3-5: g0/os1限定時は抑止。
    if (!kP15G0OS1Only)
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
    // P3-5: g0/os1限定時は抑止。
    if (!kP15G0OS1Only)
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

    // ── (E) P1-3 pre-adoption matrix: os{1,2,4,8} × ampDb{-20,-6,0} × sc{0,1}（1 kHz）──
    // P3-5: g0/os1限定時は抑止。
    if (!kP15G0OS1Only)
    {
        for (int osF : { 1, 2, 4, 8 })
        {
            const int nStages = (osF == 1) ? 0 : ((osF == 2) ? 1 : ((osF == 4) ? 2 : 3));
            for (double adb : { -20.0, -6.0, 0.0 })
            {
                std::vector<float> yOff, yOn;
                CaseOut oOff {}, oOn {};
                const bool okOff = runCase(h, cap, Signal::Sine, 1000.0, adb, osF,
                                           convo::OversamplingType::IIR, false, 1.0f, oOff);
                if (okOff) yOff = cap.outL;
                const bool okOn = runCase(h, cap, Signal::Sine, 1000.0, adb, osF,
                                          convo::OversamplingType::IIR, true, 1.0f, oOn);
                if (okOn) yOn = cap.outL;
                if (!okOff || !okOn) { ++failures; continue; }
                ++cases;
                int engDiff = 0; double engMax = 0.0;
                if (yOff.size() == yOn.size() && !yOff.empty())
                {
                    const std::size_t from = yOff.size() - static_cast<std::size_t>(kAnalysisN);
                    diffStats(yOn, yOff, from, kAnalysisN, engDiff, engMax);
                }
                emitLine(std::string("[P1CHAR] p13") + kvi("os", osF) + kvi("n", nStages)
                         + kv("ampDb", adb, 1)
                         + kv("gainDb_sc0", oOff.gainDb, 4) + kv("gainDb_sc1", oOn.gainDb, 4)
                         + kv("limiterInPeakDb_sc0", oOff.limiterInPeakDb, 2)
                         + kv("limiterInPeakDb_sc1", oOn.limiterInPeakDb, 2)
                         + kvi("limitingEngaged_sc0", oOff.limitingEngaged)
                         + kvi("limitingEngaged_sc1", oOn.limitingEngaged)
                         + kvi("hardClamp_sc0", oOff.hardClamp)
                         + kvi("hardClamp_sc1", oOn.hardClamp)
                         + kvi("clipEngagement", engDiff) + kv("clipEngMax", engMax, 6));
            }
        }
    }

    // ── (F) P1-3 staging 非0: headroom -6 dB（staging = -6 dB）・os8・-6 dBFS・1 kHz・sc{0,1} ──
    // P3-5: g0/os1限定時は抑止。
    if (!kP15G0OS1Only)
    {
        for (int sc = 0; sc <= 1; ++sc)
        {
            CaseOut o {};
            if (!runCase(h, cap, Signal::Sine, 1000.0, -6.0, 8, convo::OversamplingType::IIR,
                         sc == 1, 1.0f, o, /*headroomDb*/ -6.0f))
            { ++failures; continue; }
            ++cases;
            emitLine(std::string("[P1CHAR] p13staging os=8 n=3 headroomDb=-6.0")
                     + kvi("sc", sc) + " ampDb=-6.0"
                     + kv("gainDb", o.gainDb, 4) + kv("limiterInPeakDb", o.limiterInPeakDb, 2)
                     + kvi("limitingEngaged", o.limitingEngaged) + kvi("hardClamp", o.hardClamp));
        }
    }

    // ═══ P1-5: HOLD 解除用 characterization（test-only・production 変更なし） ═══

    struct PairCfg
    {
        double ampDb = -6.0;
        int os = 8;
        float headroom = 0.0f, makeup = 0.0f, trim = 0.0f, eqBoost = 0.0f;
        bool convBypass = true;
        float satOff = 1.0f, satOn = 1.0f;
        bool twoState = true;
        const char* irPath = nullptr;
    };
    auto runPair = [&](const char* kind, const std::string& id, const PairCfg& p,
                       bool firstInVehicle = false)
    {
        CaseOut oOff {};
        if (p.irPath != nullptr && !ensureTestIr(h, juce::File(juce::String(p.irPath))))
        { ++failures; return; }
        const bool a = runCase(h, cap, Signal::Sine, 1000.0, p.ampDb, p.os,
                               convo::OversamplingType::IIR, false, p.satOff, oOff,
                               p.headroom, p.makeup, p.trim, p.eqBoost, p.convBypass,
                               kCapBlocks, false, firstInVehicle);
        const int nStages = (p.os == 1) ? 0 : ((p.os == 2) ? 1 : ((p.os == 4) ? 2 : 3));
        if (!a) { ++failures; return; }
        if (!p.twoState)
        {
            ++cases;
            emitLine(std::string("[P1CHAR] ") + kind + " id=" + id + kvi("os", p.os)
                     + kvi("n", nStages) + kv("ampDb", p.ampDb, 1)
                     + kv("gainDb_sc0", oOff.gainDb, 4)
                     + kvi("limitingEngaged_sc0", oOff.limitingEngaged)
                     + kvi("hardClamp_sc0", oOff.hardClamp));
            return;
        }
        std::vector<float> yOff = cap.outL;
        if (p.irPath != nullptr && !ensureTestIr(h, juce::File(juce::String(p.irPath))))
        { ++failures; return; }
        CaseOut oOn {};
        const bool b = runCase(h, cap, Signal::Sine, 1000.0, p.ampDb, p.os,
                               convo::OversamplingType::IIR, true, p.satOn, oOn,
                               p.headroom, p.makeup, p.trim, p.eqBoost, p.convBypass);
        if (!b) { ++failures; return; }
        std::vector<float> yOn = cap.outL;
        ++cases;
        int engDiff = 0; double engMax = 0.0;
        if (yOff.size() == yOn.size() && !yOff.empty())
        {
            const std::size_t from = yOff.size() - static_cast<std::size_t>(kAnalysisN);
            diffStats(yOn, yOff, from, kAnalysisN, engDiff, engMax);
        }
        emitLine(std::string("[P1CHAR] ") + kind + " id=" + id
                 + kvi("os", p.os) + kvi("n", nStages) + kv("ampDb", p.ampDb, 1)
                 + kv("gainDb_sc0", oOff.gainDb, 4) + kv("gainDb_sc1", oOn.gainDb, 4)
                 + kvi("limitingEngaged_sc0", oOff.limitingEngaged)
                 + kvi("limitingEngaged_sc1", oOn.limitingEngaged)
                 + kvi("hardClamp_sc0", oOff.hardClamp) + kvi("hardClamp_sc1", oOn.hardClamp)
                 + kvi("clipEngagement", engDiff) + kv("clipEngMax", engMax, 6)
                 + snapTokens("ab0", oOff.snapBefore) + snapTokens("aa0", oOff.snapAfter)
                 + snapTokens("ab1", oOn.snapBefore) + snapTokens("aa1", oOn.snapAfter));
    };

    // ── (G) Preset replay（代表 7 プリセット × sc{0,1}・sat 0.1・1 kHz・-6 dBFS）──
    // P3-5: g0/os1限定時は抑止。
    if (!kP15G0OS1Only)
    {
        struct Preset { const char* id; float headroom; float makeup; float trim; int os; };
        const Preset presets[] = {
            { "P1", -6.0f, 12.0f, 0.0f, 0 }, { "P2", 0.0f, 0.0f, 0.0f, 0 },
            { "P3", -6.0f, 6.0f, 0.0f, 0 },  { "P4", -6.0f, 12.0f, 6.0f, 0 },
            { "P5", -6.0f, 12.0f, 0.0f, 2 }, { "P6", -6.0f, 12.0f, 0.0f, 4 },
            { "P7", -6.0f, 12.0f, 0.0f, 8 },
        };
        for (const Preset& p : presets)
        {
            PairCfg c;
            c.ampDb = -6.0; c.os = p.os; c.headroom = p.headroom; c.makeup = p.makeup;
            c.trim = p.trim; c.satOff = 0.1f; c.satOn = 0.1f;
            runPair("p15preset", p.id, c);
        }
    }

    // ── (H) EQ boost + OS（1 kHz band +{3,6,9} dB × os{1,4,8} × amp{-20,0}・sat 1.0）──
    // P3-5: g0/os1限定時は抑止。
    if (kP15FullMatrix && !kP15G0OS1Only)
    {
        for (int osF : { 1, 4, 8 })
            for (float boost : { 3.0f, 6.0f, 9.0f })
                for (double adb : { -20.0, 0.0 })
                {
                    PairCfg c;
                    c.ampDb = adb; c.os = osF; c.eqBoost = boost; c.satOff = 1.0f; c.satOn = 1.0f;
                    runPair("p15eq", "b" + std::to_string((int)boost) + "_am" + std::to_string((int)adb), c);
                }
    }

    // ── (I) IR boost + OS（test IR g0/g3/g6 × os{1,4,8} × amp{-20,-6}・sc{0,1}）──
    // P3-5: g0/os1限定時は g0×os1 のみ（topology: runPair・ensureTestIr×2・sc0/sc1維持）。
    if (kP15FullMatrix)
    {
        struct IrCase { const char* path; const char* label; };
        const IrCase irs[] = {
            { "tmp/p15_ir_g0.wav", "g0" },
        };
        bool firstPair = true;   // P3-5: vehicle内最初の runPair のみ strict-honor（P3-4 §4-3）
        for (const IrCase& ic : irs)
            for (int osF : { 1 })
                for (double adb : { -20.0, -6.0 })
                {
                    PairCfg c;
                    c.ampDb = adb; c.os = osF; c.convBypass = false; c.irPath = ic.path;
                    c.satOff = 1.0f; c.satOn = 1.0f;
                    runPair("p15ir", std::string(ic.label) + "_os" + std::to_string(osF)
                             + "_am" + std::to_string((int)adb), c, firstPair);
                    firstPair = false;
                }
    }

    // ── (J) Staging × nonlinear（staging{-12,-6,0,+6} × os{2,4,8} × amp{-20,-6,0}・代表点）──
    // P3-5: g0/os1限定時は抑止。
    if (kP15FullMatrix && !kP15G0OS1Only)
    {
        struct StCase { float headroom; float makeup; const char* label; };
        const StCase stages[] = {
            { -12.0f, 0.0f, "st-12" }, { -6.0f, 0.0f, "st-6" },
            { 0.0f, 0.0f, "st0" },     { 0.0f, 6.0f, "st+6" },
        };
        for (const StCase& st : stages)
        {
            for (int osF : { 2, 4, 8 })
                for (double adb : { -20.0, -6.0, 0.0 })
                {
                    PairCfg c;
                    c.ampDb = adb; c.os = osF; c.headroom = st.headroom; c.makeup = st.makeup;
                    c.satOff = 1.0f; c.satOn = 1.0f;
                    runPair("p15staging", std::string(st.label) + "_os" + std::to_string(osF)
                             + "_am" + std::to_string((int)adb), c);
                }
        }
    }

    // ── (K) Saturation sweep（sat{0,0.1,0.25,0.5,0.75,1.0} × os{1,4,8} × amp{-6,0}（+os8/-20））──
    // P3-5: g0/os1限定時は抑止。
    if (kP15FullMatrix && !kP15G0OS1Only)
    {
        const float sats[] = { 0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 1.0f };
        for (float st : sats)
        {
            for (int osF : { 1, 4, 8 })
            {
                for (double adb : { -6.0, 0.0 })
                {
                    PairCfg c;
                    c.ampDb = adb; c.os = osF; c.satOff = st; c.satOn = st;
                    runPair("p15sat", "s" + std::to_string((int)(st * 100.0f)) + "_os"
                             + std::to_string(osF) + "_am" + std::to_string((int)adb), c);
                }
                if (osF == 8)
                {
                    PairCfg c;
                    c.ampDb = -20.0; c.os = osF; c.satOff = st; c.satOn = st;
                    runPair("p15sat", "s" + std::to_string((int)(st * 100.0f))
                             + "_os8_am-20", c);
                }
            }
        }
    }

    // ── (K2) P1-5-HR Step 3: reduced nonlinear 12 条件（sat{0.1,1.0} × os{1,8} × amp{-20,-6,0}・neutral staging）──
    // P3-5: g0/os1限定時は抑止（crash域 p15hrnl を vehicle から除外）。
    if (!kP15G0OS1Only)
    {
        auto rmsWin = [](const std::vector<float>& v, std::size_t from, std::size_t n)
        {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) { const double x = static_cast<double>(v[from + i]); s += x * x; }
            return std::sqrt(s / static_cast<double>(n));
        };
        for (float sat : { 0.1f, 1.0f })
        {
            for (int osF : { 1, 8 })
            {
                for (double adb : { -20.0, -6.0, 0.0 })
                {
                    const int nStages = (osF == 1) ? 0 : ((osF == 2) ? 1 : ((osF == 4) ? 2 : 3));
                    CaseOut o0 {};
                    if (!runCase(h, cap, Signal::Sine, 1000.0, adb, osF,
                                 convo::OversamplingType::IIR, false, sat, o0,
                                 0.0f, 0.0f, 0.0f, 0.0f, true))
                    { ++failures; continue; }
                    const std::vector<float> y0 = cap.outL;
                    CaseOut o1 {};
                    if (!runCase(h, cap, Signal::Sine, 1000.0, adb, osF,
                                 convo::OversamplingType::IIR, true, sat, o1,
                                 0.0f, 0.0f, 0.0f, 0.0f, true))
                    { ++failures; continue; }
                    const std::vector<float> y1 = cap.outL;
                    int engDiff = 0; double engMax = 0.0;
                    double rms0 = 0.0, rms1 = 0.0;
                    if (y0.size() == y1.size() && y0.size() >= static_cast<std::size_t>(kAnalysisN))
                    {
                        const std::size_t from = y0.size() - static_cast<std::size_t>(kAnalysisN);
                        diffStats(y1, y0, from, kAnalysisN, engDiff, engMax);
                        rms0 = rmsWin(y0, from, static_cast<std::size_t>(kAnalysisN));
                        rms1 = rmsWin(y1, from, static_cast<std::size_t>(kAnalysisN));
                    }
                    ++cases;
                    emitLine(std::string("[P1CHAR] p15hrnl")
                             + kv("sat", (double)sat, 2) + kvi("os", osF) + kvi("n", nStages)
                             + kv("ampDb", adb, 1)
                             + kv("gainDb_sc0", o0.gainDb, 4) + kv("gainDb_sc1", o1.gainDb, 4)
                             + kv("outMax_sc0", o0.outMax, 6) + kv("outMax_sc1", o1.outMax, 6)
                             + kv("rms_sc0", rms0, 6) + kv("rms_sc1", rms1, 6)
                             + kv("limiterInPeakDb_sc0", o0.limiterInPeakDb, 2)
                             + kv("limiterInPeakDb_sc1", o1.limiterInPeakDb, 2)
                             + kvi("limitingEngaged_sc0", o0.limitingEngaged)
                             + kvi("limitingEngaged_sc1", o1.limitingEngaged)
                             + kvi("hardClamp_sc0", o0.hardClamp) + kvi("hardClamp_sc1", o1.hardClamp)
                             + kvi("clipEngagement", engDiff) + kv("clipEngMax", engMax, 6));
                }
            }
        }
    }

    // ── (L) Limiter engagement sweep（amp -20..0 step 2 × os{1,4,8}・sc=0）──
    // P3-5: g0/os1限定時は抑止。
    if (kP15FullMatrix && !kP15G0OS1Only)
    {
        for (int osF : { 1, 4, 8 })
            for (int adb = -20; adb <= 0; adb += 2)
            {
                PairCfg c;
                c.ampDb = (double)adb; c.os = osF; c.twoState = false; c.satOff = 0.0f;
                runPair("p15lim", "am" + std::to_string(adb) + "_os" + std::to_string(osF), c);
            }
    }

    // ── (M) 聴感評価素材（AB/ABX 用 WAV・Program 信号 1.7 s・sat 0.1）──
    // P3-5: g0/os1限定時は抑止。
    if (!kP15G0OS1Only)
    {
        struct ListenCase
        { const char* cat; double ampDb; bool eqBypass; float eqBoost; bool convBypass;
          const char* irPath; bool softClip; };
        const ListenCase lcases[] = {
            { "dry",  -6.0, true,  0.0f, true,  nullptr,             false },
            { "eqid", -6.0, false, 0.0f, true,  nullptr,             false },
            { "eq6",  -6.0, false, 6.0f, true,  nullptr,             false },
            { "ir",   -6.0, false, 0.0f, false, "tmp/p15_ir_g0.wav", false },
            { "sc",   -6.0, false, 0.0f, true,  nullptr,             true  },
            { "lim",   0.0, false, 0.0f, true,  nullptr,             false },
        };
        for (const ListenCase& lc : lcases)
        {
            for (int on = 0; on <= 1; ++on)
            {
                if (lc.irPath != nullptr && !ensureTestIr(h, juce::File(juce::String(lc.irPath))))
                { ++failures; continue; }
                CaseOut o {};
                if (!runCase(h, cap, Signal::Program, 1000.0, lc.ampDb, 8, convo::OversamplingType::IIR,
                             lc.softClip || on == 1, 0.1f, o, 0.0f, 0.0f, 0.0f, lc.eqBoost,
                             lc.convBypass, /*capBlocks*/ 160, lc.eqBypass))
                { ++failures; continue; }
                ++cases;
                const char* path = listenPath(lc.cat, on == 1);
                const bool wrote = writeWav16(path, cap.outL, static_cast<int>(kSr));
                emitLine(std::string("[P1CHAR] p15listen cat=") + lc.cat + kvi("on", on)
                         + kv("ampDb", lc.ampDb, 1) + kvi("frames", (long long)cap.outL.size())
                         + " file=" + (path ? path : "-") + kvi("ok", wrote ? 1 : 0));
            }
        }
    }

    h.stop();
    emitLine(std::string("[P1CHAR] summary") + kvi("flag_macro", (int)P1CHAR_FLAG_MACRO)
             + kvi("cases", cases) + kvi("failures", failures));
    return failures == 0 ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// P3-5-R30: Recovery-origin vehicle（test-only・D1 = submitRecoveryIntent 直呼び）
//   Full AudioEngine 上で Recovery-origin publish を決定論的に 1 件以上発生させる。
//   production 変更 0・新 counter/getter 0・既存観測（cmt/coord/seq/drp 他）のみ。
//   D105 winner identity は判定しない（R28/R29 の UNOBSERVABLE 維持）。
//   実行: AudioEngineHarness.exe --p1-recovery-origin
//   注意: submitRecoveryIntent の producer は本来 CoordinatorLoop だが、本 vehicle では
//         非 quarantine のため recoveryIntentQueue_ の producer は本スレッドのみ
//         （SPSC 単一 producer 維持）。詳細は R30 文書 §12 に記録。
// ─────────────────────────────────────────────────────────────────────────────
struct P1RecSnap
{
    long long req = 0, que = 0, dup = 0, take = 0, bld = 0;
    long long cmt = 0, coord = 0, drp = 0, seq = 0, blo = 0;
};

P1RecSnap p1RecReadSnap(AudioEngine& e)
{
    const auto d  = e.getRebuildDispatchDiagnostics();
    const auto lc = e.getRuntimeLifecycleDiagnostics();
    P1RecSnap s;
    s.req   = (long long)d.requestCount;
    s.que   = (long long)d.queuedCount;
    s.dup   = (long long)d.blockedPendingDuplicateCount;
    s.take  = (long long)e.getRebuildTakeCount();
    s.bld   = (long long)e.getRebuildBuildResultCount();
    s.cmt   = (long long)e.getRebuildCommitEnqueueCount();
    s.coord = (long long)e.getCoordinatorTakeCount();
    s.drp   = (long long)lc.lastDroppedGeneration;
    s.seq   = (long long)e.getLastCommittedPublicationSequence();
    s.blo   = (long long)e.getPublicationBacklogCount();
    return s;
}

std::string p1RecSnapFields(const P1RecSnap& s)
{
    return kvi("req", s.req) + kvi("que", s.que) + kvi("dup", s.dup)
         + kvi("take", s.take) + kvi("bld", s.bld) + kvi("cmt", s.cmt)
         + kvi("coord", s.coord) + kvi("drp", s.drp) + kvi("seq", s.seq)
         + kvi("blo", s.blo);
}

// drain 判定の既存観測（collectDrainAudit・public・R30 の STOP-E 診断用）
std::string p1RecDrainFields(AudioEngine& e)
{
    const auto a = e.collectDrainAudit();
    return kvi("fullyDrained", e.isFullyDrained() ? 1 : 0)
         + kvi("pendPub", (long long)a.pendingPublication)
         + kvi("pendRetire", (long long)a.pendingRetire)
         + kvi("xfade", (long long)a.activeCrossfadeCount)
         + kvi("routerPending", (long long)a.routerPendingRetire)
         + kvi("deferred", (long long)a.deferredPublish)
         + kvi("quarRes", (long long)a.quarantineResident)
         + kvi("activeWorlds", (long long)a.activeWorldCount)
         + kvi("published", (long long)a.publishedCount)
         + kvi("retired", (long long)a.retiredCount)
         + kvi("activeReaders", (long long)a.activeReaderCount)
         + kvi("stuckReaders", (long long)a.stuckReaderCount)
         + kvi("overflowRes", (long long)a.overflowRingResident);
}

int runP1RecoveryOrigin(int argc, char* argv[])
{
    (void)argc; (void)argv;
    constexpr int kEpisodes = 3;
    constexpr int kWaitMs   = 20000;

    emitLine("[P1REC] start r30_d1_recovery_origin");

    AudioEngineHarness h;
    if (!h.start(kSr, kBlock))
    {
        emitLine("[P1REC] PRECONDITION FAILURE harness start");
        return 1;
    }
    AudioEngine& e = h.engine();

    // Precondition 1: authoritative published Runtime の成立
    bool ready = false;
    for (int i = 0; i < 3000 && !ready; ++i)
    {
        const auto* w = e.observePublishedWorld();
        ready = (w != nullptr && w->engine.current != nullptr
                 && e.hasAuthoritativePublishedRuntime());
        if (!ready) { pumpMessages(); std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
    }
    if (!ready)
    {
        emitLine("[P1REC] PRECONDITION FAILURE no authoritative published runtime");
        h.stop();
        return 2;
    }

    // Precondition 2: active DSP handle（Recovery build の gate は非 null handle のみ要求）
    const auto* w0 = e.observePublishedWorld();
    auto* activeDSP = static_cast<AudioEngine::DSPCore*>(w0->engine.current);
    const auto handle = e.registerDSPHandleForRuntime(activeDSP);
    if (handle.isNull())
    {
        emitLine("[P1REC] PRECONDITION FAILURE null handle");
        h.stop();
        return 2;
    }
    emitLine(std::string("[P1REC] precondition ok")
             + kvi("slot", (long long)handle.slot)
             + kvi("gen", (long long)handle.generation));

    // ★ P3-5-R32: baseline settle（起動直後の bootstrap/rebuild publish を control から除外する）。
    //   drain 観測は行わない（R31: waitForDrain は shutdown 契約内のみ）。
    sleepPump(1000);

    int okEpisodes = 0;

    // Control（trigger なし）: 同程度の待機で seq が自発的に進まないことを確認し、
    //   episode の seq 前進が trigger 起因であることを window 粒度で裏付ける。
    {
        const long long seqC0 = (long long)e.getLastCommittedPublicationSequence();
        const auto start  = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(5000))
        { pumpMessages(); std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
        const long long seqC1 = (long long)e.getLastCommittedPublicationSequence();
        emitLine(std::string("[P1REC] control_notrigger") + kvi("waitMs", 5000)
                 + kvi("dSeq", seqC1 - seqC0));
    }
    for (int ep = 1; ep <= kEpisodes; ++ep)
    {
        const P1RecSnap pre = p1RecReadSnap(e);

        auto snapshot = e.getCurrentBuildSnapshotForRecovery();
        snapshot.sealed = true;
        const long long seqBefore = pre.seq;

        // D1 trigger（1 episode につき 1 回・逐次。戻り値のみで publish 成功と判定しない）
        e.submitRecoveryIntent(handle, snapshot);

        // publish completion 待ち（Recovery-origin publish の seq advancement）
        bool seqAdvanced = false;
        {
            const auto start  = std::chrono::steady_clock::now();
            const auto budget = std::chrono::milliseconds(kWaitMs);
            while (std::chrono::steady_clock::now() - start < budget)
            {
                if ((long long)e.getLastCommittedPublicationSequence() > seqBefore)
                { seqAdvanced = true; break; }
                pumpMessages();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        // ★ P3-5-R32: episode 内 waitForDrain/drain_pre/drain_post は撤去（Running 中は
        //   shutdown 契約外 — R31 STOP-1）。terminalization は h.stop() 後の T3 で観測する。

        const P1RecSnap post = p1RecReadSnap(e);
        const long long dSeq   = post.seq   - pre.seq;
        const long long dCoord = post.coord - pre.coord;
        const long long dCmt   = post.cmt   - pre.cmt;
        const long long dDrp   = post.drp   - pre.drp;
        const long long dTake  = post.take  - pre.take;
        const long long dBld   = post.bld   - pre.bld;

        emitLine(std::string("[P1REC] episode")
                 + kvi("ep", ep)
                 + kvi("dSeq", dSeq) + kvi("dCoord", dCoord) + kvi("dCmt", dCmt)
                 + kvi("dDrp", dDrp) + kvi("dTake", dTake) + kvi("dBld", dBld)
                 + kvi("seqAdvanced", seqAdvanced ? 1 : 0));
        emitLine(std::string("[P1REC]   pre ") + p1RecSnapFields(pre));
        emitLine(std::string("[P1REC]   post") + p1RecSnapFields(post));

        // 期待形: seq 前進・coord 不変（terminalization 判定は h.stop() 後の T3 で行う）
        if (seqAdvanced && dCoord == 0 && dCmt == 0 && dTake == 0 && dBld == 0)
        {
            ++okEpisodes;
        }
        else
        {
            emitLine(std::string("[P1REC] WARN episode not ok")
                     + kvi("ep", ep)
                     + kvi("seqAdvanced", seqAdvanced ? 1 : 0)
                     + kvi("dCoord", dCoord) + kvi("dCmt", dCmt)
                     + kvi("dTake", dTake) + kvi("dBld", dBld));
        }
    }

    // ★ P3-5-R32: episode 完了後、Recovery publish が in-flight でないことを短い settle で確認する
    //   （各 episode は seq advancement を待っているため原則 in-flight なし）。
    sleepPump(500);

    // terminal observation（R31 の契約: waitForDrain は shutdown 内部でのみ使用される）。
    //   h.stop() = stopAudioOnly → requestTerminalRelease → releaseResources(terminal pass)
    //   → AudioStopped → … → VerifyDrained → waitForDrain → finalizeShutdown → ShutdownComplete
    h.stop();

    // ★ h.stop() の後だけ terminal state を既存 public API で観測する（Running 中は観測しない）。
    const auto termPhase = e.isrShutdownRuntime().getPhase();
    const auto termResult = e.isrShutdownRuntime().collectResult(
        static_cast<convo::ISRHealthState>(0), 0);
    const auto termAdmission = e.isrShutdownRuntime().admissionState();
    const bool phaseComplete = (termPhase == convo::isr::ShutdownPhase::ShutdownComplete);
    const bool phaseTimeout  = (termPhase == convo::isr::ShutdownPhase::TimedOut);
    const bool phaseFailed   = (termPhase == convo::isr::ShutdownPhase::Failed);

    emitLine(std::string("[P1REC] terminal")
             + kvi("phase", (long long)static_cast<int>(termPhase))
             + kvi("phaseComplete", phaseComplete ? 1 : 0)
             + kvi("phaseTimeout", phaseTimeout ? 1 : 0)
             + kvi("phaseFailed", phaseFailed ? 1 : 0)
             + kvi("completed", termResult.completed ? 1 : 0)
             + kvi("blockingReason", (long long)static_cast<int>(termResult.blockingReason))
             + kvi("violations", (long long)termResult.transitionViolations)
             + kvi("admissionClosed",
                   (termAdmission == convo::isr::AdmissionState::Closed) ? 1 : 0)
             + kvi("lateCallbacks", (long long)termResult.lateCallbackCount)
             + kvi("postStopEnqueue", (long long)termResult.postStopEnqueueCount));

    // drain audit は shutdown 後の diagnostic evidence（補助・主判定には使わない）。
    emitLine(std::string("[P1REC]   terminal_drain_audit ") + p1RecDrainFields(e));

    const bool terminalOk = phaseComplete && termResult.completed
                         && (termResult.transitionViolations == 0);
    emitLine(std::string("[P1REC] summary")
             + kvi("episodes", kEpisodes) + kvi("okEpisodes", okEpisodes)
             + kvi("terminalOk", terminalOk ? 1 : 0));
    return (okEpisodes == kEpisodes && terminalOk) ? 0 : 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// P3-5-FPM-Impl-1: M0/M1/M2 measurement vehicle（test-only）。
//   FPM-PREP-1 §3-§9 の測定プロトコルを test vehicle 化する。
//   production 変更 0・新 getter/counter 0・既存 public API のみ。
//   既存 R30 helper（p1RecReadSnap／p1RecDrainFields／sleepPump／waitWorldPublished／
//   waitBacklogZero／kvi／emitLine）を流用。terminal success authority は T3b 5点のみ。
//   実行: AudioEngineHarness.exe --fpm-m0 | --fpm-m1 | --fpm-m2
// ─────────────────────────────────────────────────────────────────────────────
namespace {

// T3b terminal capture（同一 shutdown episode の 5点＋diagnostic）。
//   取得子は既存 public API のみ。collectDrainAudit は diagnostic（authority ではない）。
struct FpmT3bCapture
{
    long long phase = 0;
    bool phaseComplete = false;
    bool completed = false;
    long long blockingReason = 0;
    long long violations = 0;
    bool fullyDrained = false;
    bool t3b = false;
    // diagnostic（authority ではない）
    long long lateCallbacks = 0;
    long long postStopEnqueue = 0;
    // drain audit 快照（diagnostic）
    long long pendPub = 0, pendRetire = 0, xfade = 0, routerPending = 0;
    long long deferred = 0, quarRes = 0;
    long long activeWorlds = 0, published = 0, retired = 0;
    long long activeReaders = 0, stuckReaders = 0, overflowRes = 0;
};

FpmT3bCapture fpmCaptureT3b(AudioEngine& e)
{
    FpmT3bCapture c;
    const auto phase = e.isrShutdownRuntime().getPhase();
    const auto result = e.isrShutdownRuntime().collectResult(
        static_cast<convo::ISRHealthState>(0), 0);
    const bool drained = e.isFullyDrained();
    const auto audit = e.collectDrainAudit();
    c.phase = (long long)static_cast<int>(phase);
    c.phaseComplete = (phase == convo::isr::ShutdownPhase::ShutdownComplete);
    c.completed = result.completed;
    c.blockingReason = (long long)static_cast<int>(result.blockingReason);
    c.violations = (long long)result.transitionViolations;
    c.fullyDrained = drained;
    // T3b authority：5点の conjunction のみ。completed 単独・phase 単独・
    // R30/R32 型（phase&&completed&&violations）は success 判定に使わない。
    c.t3b = c.phaseComplete && result.completed
        && (result.blockingReason == convo::isr::ShutdownBlockingReason::None)
        && (result.transitionViolations == 0) && drained;
    c.lateCallbacks = (long long)result.lateCallbackCount;
    c.postStopEnqueue = (long long)result.postStopEnqueueCount;
    c.pendPub = (long long)audit.pendingPublication;
    c.pendRetire = (long long)audit.pendingRetire;
    c.xfade = (long long)audit.activeCrossfadeCount;
    c.routerPending = (long long)audit.routerPendingRetire;
    c.deferred = (long long)audit.deferredPublish;
    c.quarRes = (long long)audit.quarantineResident;
    c.activeWorlds = (long long)audit.activeWorldCount;
    c.published = (long long)audit.publishedCount;
    c.retired = (long long)audit.retiredCount;
    c.activeReaders = (long long)audit.activeReaderCount;
    c.stuckReaders = (long long)audit.stuckReaderCount;
    c.overflowRes = (long long)audit.overflowRingResident;
    return c;
}

void fpmEmitT3b(const char* tag, const FpmT3bCapture& c)
{
    emitLine(std::string("[FPM] ") + tag + std::string(" terminal")
             + kvi("phase", c.phase)
             + kvi("phaseComplete", c.phaseComplete ? 1 : 0)
             + kvi("completed", c.completed ? 1 : 0)
             + kvi("blockingReason", c.blockingReason)
             + kvi("violations", c.violations)
             + kvi("fullyDrained", c.fullyDrained ? 1 : 0)
             + kvi("t3b", c.t3b ? 1 : 0)
             + kvi("lateCallbacks", c.lateCallbacks)
             + kvi("postStopEnqueue", c.postStopEnqueue));
    emitLine(std::string("[FPM] ") + tag + std::string(" terminal_drain_audit")
             + kvi("pendPub", c.pendPub)
             + kvi("pendRetire", c.pendRetire)
             + kvi("xfade", c.xfade)
             + kvi("routerPending", c.routerPending)
             + kvi("deferred", c.deferred)
             + kvi("quarRes", c.quarRes)
             + kvi("activeWorlds", c.activeWorlds)
             + kvi("published", c.published)
             + kvi("retired", c.retired)
             + kvi("activeReaders", c.activeReaders)
             + kvi("stuckReaders", c.stuckReaders)
             + kvi("overflowRes", c.overflowRes));
}

// run 分類（FPM-PREP-1 §9）。
//   VALID：T3b==true。INVALID-TERMINAL：T3a==true && T3b==false。
const char* fpmClassify(bool t3b, bool phaseComplete)
{
    if (t3b) return "VALID";
    if (phaseComplete) return "INVALID-TERMINAL";
    return "ABORTED";
}

// startup settle：authoritative runtime＋seq 前進＋backlog 0（R30 Precondition 流用）。
bool fpmStartupSettle(AudioEngine& e, const char* tag)
{
    bool ready = false;
    for (int i = 0; i < 3000 && !ready; ++i)
    {
        const auto* w = e.observePublishedWorld();
        ready = (w != nullptr && w->engine.current != nullptr
                 && e.hasAuthoritativePublishedRuntime());
        if (!ready) { pumpMessages(); std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
    }
    if (!ready)
    {
        emitLine(std::string("[FPM] ") + tag + std::string(" ABORTED no authoritative runtime"));
        return false;
    }
    const long long seq0 = (long long)e.getLastCommittedPublicationSequence();
    if (!waitWorldPublished(e, seq0, 20000))
    {
        emitLine(std::string("[FPM] ") + tag + std::string(" ABORTED startup publish not settled"));
        return false;
    }
    if (!waitBacklogZero(e, 10000))
    {
        emitLine(std::string("[FPM] ") + tag + std::string(" ABORTED backlog not zero"));
        return false;
    }
    sleepPump(500);
    return true;
}

// single recovery episode（R30 episode-driving の最小流用・kEpisodes=1）。
//   E2 winner なし。Running 中 waitForDrain なし。
bool fpmSingleRecoveryEpisode(AudioEngine& e, const char* tag)
{
    const auto* w0 = e.observePublishedWorld();
    if (w0 == nullptr || w0->engine.current == nullptr)
    {
        emitLine(std::string("[FPM] ") + tag + std::string(" ABORTED no published world"));
        return false;
    }
    auto* activeDSP = static_cast<AudioEngine::DSPCore*>(w0->engine.current);
    const auto handle = e.registerDSPHandleForRuntime(activeDSP);
    if (handle.isNull())
    {
        emitLine(std::string("[FPM] ") + tag + std::string(" ABORTED null handle"));
        return false;
    }
    const P1RecSnap pre = p1RecReadSnap(e);
    auto snapshot = e.getCurrentBuildSnapshotForRecovery();
    snapshot.sealed = true;
    const long long seqBefore = pre.seq;
    e.submitRecoveryIntent(handle, snapshot);
    bool seqAdvanced = false;
    {
        const auto start = std::chrono::steady_clock::now();
        const auto budget = std::chrono::milliseconds(20000);
        while (std::chrono::steady_clock::now() - start < budget)
        {
            if ((long long)e.getLastCommittedPublicationSequence() > seqBefore)
            { seqAdvanced = true; break; }
            pumpMessages();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    const P1RecSnap post = p1RecReadSnap(e);
    const long long dSeq   = post.seq   - pre.seq;
    const long long dCoord = post.coord - pre.coord;
    const long long dCmt   = post.cmt   - pre.cmt;
    const long long dTake  = post.take  - pre.take;
    const long long dBld   = post.bld   - pre.bld;
    emitLine(std::string("[FPM] ") + tag + std::string(" episode")
             + kvi("dSeq", dSeq) + kvi("dCoord", dCoord) + kvi("dCmt", dCmt)
             + kvi("dTake", dTake) + kvi("dBld", dBld)
             + kvi("seqAdvanced", seqAdvanced ? 1 : 0));
    emitLine(std::string("[FPM] ") + tag + std::string("   pre ") + p1RecSnapFields(pre));
    emitLine(std::string("[FPM] ") + tag + std::string("   post") + p1RecSnapFields(post));
    return seqAdvanced && dCoord == 0 && dCmt == 0 && dTake == 0 && dBld == 0;
}

} // namespace

// M0 Control：Recovery なし・publication 最小・crossfade 最小・retire 最小。
//   start → startup settle → stop → T3b capture → evidence emit。
//   M0 が INVALID-TERMINAL でも捏造せず分類のみ（FPM-PREP-1 §5）。
int runFpmM0()
{
    constexpr const char* kTag = "m0";
    emitLine("[FPM] start fpm_m0_control");
    AudioEngineHarness h;
    if (!h.start(kSr, kBlock))
    {
        emitLine("[FPM] m0 ABORTED harness start");
        return 2;
    }
    AudioEngine& e = h.engine();
    if (!fpmStartupSettle(e, kTag))
    {
        h.stop();
        return 2;
    }
    h.stop();
    const auto cap = fpmCaptureT3b(e);
    fpmEmitT3b(kTag, cap);
    const char* cls = fpmClassify(cap.t3b, cap.phaseComplete);
    emitLine(std::string("[FPM] m0 summary") + kvi("t3b", cap.t3b ? 1 : 0)
             + std::string(" class=") + cls);
    // exit code は vehicle sequence 完遂を示す（T3b 成否は class フィールドで判定・捏造しない）。
    return 0;
}

// M1 Single Recovery：episode=1（R30 流用）。E2 winner なし。Running 中 waitForDrain なし。
int runFpmM1()
{
    constexpr const char* kTag = "m1";
    emitLine("[FPM] start fpm_m1_single_recovery");
    AudioEngineHarness h;
    if (!h.start(kSr, kBlock))
    {
        emitLine("[FPM] m1 ABORTED harness start");
        return 2;
    }
    AudioEngine& e = h.engine();
    if (!fpmStartupSettle(e, kTag))
    {
        h.stop();
        return 2;
    }
    sleepPump(1000);
    const bool epOk = fpmSingleRecoveryEpisode(e, kTag);
    emitLine(std::string("[FPM] m1 episode_ok") + kvi("ok", epOk ? 1 : 0));
    sleepPump(500);
    h.stop();
    const auto cap = fpmCaptureT3b(e);
    fpmEmitT3b(kTag, cap);
    const char* cls = fpmClassify(cap.t3b, cap.phaseComplete);
    emitLine(std::string("[FPM] m1 summary") + kvi("episodeOk", epOk ? 1 : 0)
             + kvi("t3b", cap.t3b ? 1 : 0) + std::string(" class=") + cls);
    return 0;
}

// M2 Full Pipeline：publish→crossfade settle→recovery→publish→settle→shutdown→T3b。
//   音質・buzz・limiter・NUC は扱わない（pipeline integrity のみ）。
int runFpmM2()
{
    constexpr const char* kTag = "m2";
    emitLine("[FPM] start fpm_m2_full_pipeline");
    AudioEngineHarness h;
    if (!h.start(kSr, kBlock))
    {
        emitLine("[FPM] m2 ABORTED harness start");
        return 2;
    }
    AudioEngine& e = h.engine();
    if (!fpmStartupSettle(e, kTag))
    {
        h.stop();
        return 2;
    }
    sleepPump(1000);
    // publish operation：requestRebuild(Structural)（D167-5 と同一の公開入口・
    // lifecycleState を触らない。prepareToPlay は audio 停止後のみ呼ぶ harness 契約のため
    // audio 実行中の M2 では使用しない）。
    const long long seq0 = (long long)e.getLastCommittedPublicationSequence();
    e.requestRebuild(convo::RebuildKind::Structural);
    if (!waitWorldPublished(e, seq0, 20000))
    {
        emitLine("[FPM] m2 ABORTED publish-1 not settled");
        h.stop();
        return 2;
    }
    // crossfade settle：activeCrossfadeCount==0（diagnostic 条件・authority ではない）。
    {
        bool settled = false;
        for (int i = 0; i < 1000 && !settled; ++i)
        {
            settled = (e.collectDrainAudit().activeCrossfadeCount == 0);
            if (!settled) { pumpMessages(); std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
        }
        emitLine(std::string("[FPM] m2 xfade_settle") + kvi("settled", settled ? 1 : 0));
    }
    const bool epOk = fpmSingleRecoveryEpisode(e, kTag);
    emitLine(std::string("[FPM] m2 episode_ok") + kvi("ok", epOk ? 1 : 0));
    // 2nd publish：再度 Structural rebuild を要求（D167-5 と同一の公開入口）。
    const long long seq1 = (long long)e.getLastCommittedPublicationSequence();
    e.requestRebuild(convo::RebuildKind::Structural);
    if (!waitWorldPublished(e, seq1, 20000))
    {
        emitLine("[FPM] m2 ABORTED publish-2 not settled");
        h.stop();
        return 2;
    }
    sleepPump(500);
    h.stop();
    const auto cap = fpmCaptureT3b(e);
    fpmEmitT3b(kTag, cap);
    const char* cls = fpmClassify(cap.t3b, cap.phaseComplete);
    emitLine(std::string("[FPM] m2 summary") + kvi("episodeOk", epOk ? 1 : 0)
             + kvi("t3b", cap.t3b ? 1 : 0) + std::string(" class=") + cls);
    return 0;
}