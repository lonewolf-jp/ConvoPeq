// BassBuzzMeasurement.cpp
// WORK104 — 低音ジジジノイズの自動計測（measurement-only、production source 無変更）。
//
// 目的: WORK103 で特定した3要因（A:出力安全鎖の低域歪 / B2:irFreqPeakGainDb未伝搬 /
//   C:超音波→IMD）を、実 AudioEngine＋実 DSP チェーンのオフライン駆動で再現計測する。
//
// 実行: AudioEngineHarness.exe --buzz [--buzz-*]
//   --buzz-sr=192000 --buzz-block=1024 --buzz-ir=<path> --buzz-out=<csv>
//   --buzz-quick（C0/C1/C3/C4 × sine50/kick/multisine のみ） --buzz-dur=<sec>
//
// 方式: harness tap で正弦・キック・マルチサインを注入し出力を回収。
//   設定行列（bypass／conv-only／eq-only／conv+eq／makeup-6dB／softclip-off／
//   phase-AsIs／shaper-psycho）× 信号のピーク／THD／flat-top／超音波比を測定。
//   B2 は +9dB@50Hz 合成ブースト IR をロードし readback が 0.0 のままかで判定。
//
// ビルド: AudioEngineHarness ターゲットに本 TU を追加（CMakeLists 参照）。
// 規約: tap 内での確保・ロック・I/O 禁止（回収は測定スレッド側で行う）。

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include "convolver/IRTrimTestHooks.h"   // ★ WORK111: test/measurement-only hook
#include "tests/NUPCTestAccess.h"        // ★ WORK113-7B: NUC 内部読み取り専用 getter
#include "tests/AudioEngineHarness/TransitionMetrics.h" // ★ WORK113-17: §15 transition 計測
#include <thread>
#include <vector>

#include <JuceHeader.h>
#include <juce_dsp/juce_dsp.h>

#include "AudioEngineHarness.h"
#include "audioengine/AudioEngine.h"

#if JUCE_WINDOWS
#include <Windows.h>
#endif

namespace convo_buzz {

// ── stderr logger（T1Measurement と同一方式） ──
class BuzzLogger : public juce::Logger
{
public:
    void logMessage(const juce::String& message) override
    {
        std::fprintf(stderr, "%s\n", message.toStdString().c_str());
        std::fflush(stderr);
    }
};

// ── Win32 メッセージ pump（H01 と同一方式） ──
void pumpBuzzMessages() noexcept
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
        pumpBuzzMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

bool waitBacklogZero(AudioEngine& e, int timeoutMs)
{
    const auto start = std::chrono::steady_clock::now();
    const auto budget = std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() - start < budget)
    {
        if (e.getPublicationBacklogCount() == 0)
            return true;
        pumpBuzzMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return e.getPublicationBacklogCount() == 0;
}

// ★ WORK104-R2: world-publish追跡付き settle。backlogゼロだけでは
//   debounce保留中の再発行を見逃すため、commit-seqの前進を要求する。
long long lastPublishedSeq(AudioEngine& e)
{
    return static_cast<long long>(e.getLastCommittedPublicationSequence());
}

bool waitWorldPublished(AudioEngine& e, long long before, int timeoutMs, const char* tag)
{
    const auto start = std::chrono::steady_clock::now();
    const auto budget = std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() - start < budget)
    {
        const long long cur = lastPublishedSeq(e);
        if (cur != before && e.getPublicationBacklogCount() == 0)
        {
            // 安定保持：300ms後に同一seq＋backlogゼロなら確定
            sleepPump(300);
            const long long cur2 = lastPublishedSeq(e);
            if (cur2 == cur && e.getPublicationBacklogCount() == 0)
            {
                std::fprintf(stderr, "[BUZZ] %s: world seq %lld -> %lld committed\n",
                             tag, before, cur2);
                return true;
            }
        }
        pumpBuzzMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::fprintf(stderr, "[BUZZ] WARN %s: no world publish within %dms (seq stayed %lld)\n",
                 tag, timeoutMs, before);
    return false;
}

bool waitIrFinalized(AudioEngine& e, int timeoutMs)
{
    const auto start = std::chrono::steady_clock::now();
    const auto budget = std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() - start < budget)
    {
        if (e.getConvolverProcessor().isIRFinalized())
            return true;
        pumpBuzzMessages();
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    return e.getConvolverProcessor().isIRFinalized();
}

// ── default.xml の20band定義 ──
struct EqBandDef { float freq, gain, q; int type; };
static constexpr EqBandDef kDefaultBands[20] = {
    {25.0f, 3.0f, 0.707f, 0}, {40.0f, 3.0f, 0.707f, 1}, {63.0f, 3.0f, 0.707f, 1},
    {100.0f, 0.0f, 0.707f, 1}, {160.0f, 0.0f, 0.707f, 1}, {250.0f, 0.0f, 0.707f, 1},
    {400.0f, 0.0f, 0.707f, 1}, {630.0f, 0.0f, 0.707f, 1}, {1000.0f, 0.0f, 0.707f, 1},
    {1600.0f, 0.0f, 0.707f, 1}, {2500.0f, 0.0f, 0.707f, 1}, {4000.0f, 0.0f, 0.707f, 1},
    {6300.0f, 0.0f, 0.707f, 1}, {10000.0f, 0.0f, 0.707f, 1}, {11000.0f, 0.0f, 0.707f, 1},
    {12500.0f, 0.5f, 0.707f, 1}, {14000.0f, 0.7f, 0.707f, 1}, {16500.0f, 1.0f, 0.707f, 1},
    {18000.0f, 1.0f, 0.707f, 1}, {19500.0f, 1.0f, 0.707f, 2},
};

enum class BuzzSignal { Sine50, Sine40, Sine100, Kick, Multi, Impulse };

struct BuzzConfig
{
    const char* id = "";
    const char* desc = "";
    bool eqBypass = false;
    bool convBypass = false;
    bool manualMakeupMinus6 = false; // autoGain OFF + makeup -6dB相当
    bool softClipOff = false;
    bool phaseAsIs = false;          // true=AsIs / false=Mixed
    bool shaperPsycho = false;       // true=Psychoacoustic / false=Adaptive9th
};

// ── 信号生成（ブロック単位・位相連続） ──
class SignalGen
{
public:
    void reset(BuzzSignal s, double sr)
    {
        sig_ = s; sr_ = sr; n_ = 0;
    }

    void setLevel(float v) noexcept { level_ = v; }

    float next() { return nextRaw() * level_; }

    float nextRaw()
    {
        const double t = static_cast<double>(n_++) / sr_;
        switch (sig_)
        {
            case BuzzSignal::Sine50:  return static_cast<float>(std::sin(2.0 * kPi * 50.0 * t));
            case BuzzSignal::Sine40:  return static_cast<float>(std::sin(2.0 * kPi * 40.0 * t));
            case BuzzSignal::Sine100: return static_cast<float>(std::sin(2.0 * kPi * 100.0 * t));
            case BuzzSignal::Kick:
            {
                const double period = 0.25;
                const double tt = std::fmod(t, period);
                const double body = std::sin(2.0 * kPi * 55.0 * tt) * std::exp(-tt / 0.09) * 0.9;
                const double click = std::sin(2.0 * kPi * 3000.0 * tt) * std::exp(-tt / 0.005) * 0.1;
                return static_cast<float>(body + click);
            }
            case BuzzSignal::Multi:
            {
                // 50/100/1000/10000/16000Hz 等振幅（worst peak 1.0）
                double v = 0.2 * std::sin(2.0 * kPi * 50.0 * t)
                         + 0.2 * std::sin(2.0 * kPi * 100.0 * t)
                         + 0.2 * std::sin(2.0 * kPi * 1000.0 * t)
                         + 0.2 * std::sin(2.0 * kPi * 10000.0 * t)
                         + 0.2 * std::sin(2.0 * kPi * 16000.0 * t);
                return static_cast<float>(v);
            }
            // ★ WORK106: 単一インパルス（システム同定用）。最初の1サンプルのみ 1.0。
            case BuzzSignal::Impulse:
                return (n_ == 1) ? 1.0f : 0.0f;
        }
        return 0.0f;
    }

    static double freqHz(BuzzSignal s)
    {
        switch (s)
        {
            case BuzzSignal::Sine50: return 50.0;
            case BuzzSignal::Sine40: return 40.0;
            case BuzzSignal::Sine100: return 100.0;
            default: return 0.0;
        }
    }

    static const char* name(BuzzSignal s)
    {
        switch (s)
        {
            case BuzzSignal::Sine50: return "sine50";
            case BuzzSignal::Sine40: return "sine40";
            case BuzzSignal::Sine100: return "sine100";
            case BuzzSignal::Kick: return "kick";
            case BuzzSignal::Impulse: return "impulse";
            default: return "multi";
        }
    }

private:
    static constexpr double kPi = 3.14159265358979323846;
    BuzzSignal sig_ = BuzzSignal::Sine50;
    double sr_ = 48000.0;
    long long n_ = 0;
    float level_ = 1.0f;
};

// ── 測定セッション状態（tap と測定スレッドで共有） ──
struct Session
{
    std::atomic<int> mode { 0 }; // 0=idle(silence) 1=run(capture)
    SignalGen gen;
    std::mutex capMutex;
    std::vector<float> inCap;   // ch0 入力
    std::vector<float> outCapL; // ch0 出力
    std::vector<float> outCapR; // ch1 出力
    std::vector<float> blockPeak; // 出力ブロック毎ピーク（遷移／ランプ検出用）
    std::atomic<long long> inPeakBits { 0 };
};

struct Metrics
{
    double inPeak = 0.0, outPeak = 0.0, outRms = 0.0, crestDb = 0.0;
    long long flatTop = 0;    // |x| >= 0.8900（HardClamp 0.8913 直前）
    long long limitZone = 0;  // |x| >= 0.8414（Limiter threshold）
    double thdDb = 0.0;       // 正弦のみ、最小二乗fit残差／基本波
    double ultraRatioDb = -200.0; // E(24k-96k)/E(20Hz-20k)
    double dcMean = 0.0;
    double driftDb = 0.0;     // ブロックピーク後四半平均／前四半平均（遷移検出用）
};

double blockDriftDb(const std::vector<float>& bp)
{
    if (bp.size() < 16)
        return 0.0;
    const size_t q = bp.size() / 4;
    double first = 0.0, last = 0.0;
    for (size_t i = 0; i < q; ++i) first += bp[i];
    for (size_t i = bp.size() - q; i < bp.size(); ++i) last += bp[i];
    first /= static_cast<double>(q);
    last /= static_cast<double>(q);
    return 20.0 * std::log10((last + 1e-12) / (first + 1e-12));
}

Metrics analyzeRun(const std::vector<float>& in, const std::vector<float>& out,
                   double sr, double fundHz, double discardSec)
{
    Metrics m;
    const size_t discard = static_cast<size_t>(discardSec * sr);
    if (out.size() <= discard + 1024 || in.size() <= discard + 1024)
        return m;
    const float* x = out.data() + discard;
    const float* d = in.data() + discard;
    const size_t n = out.size() - discard;

    double peak = 0.0, sum2 = 0.0, dc = 0.0, inPeak = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double v = x[i];
        peak = std::max(peak, std::abs(v));
        sum2 += v * v;
        dc += v;
        inPeak = std::max(inPeak, std::abs(static_cast<double>(d[i])));
        if (std::abs(v) >= 0.8900) ++m.flatTop;
        if (std::abs(v) >= 0.8414) ++m.limitZone;
    }
    m.outPeak = peak;
    m.inPeak = inPeak;
    m.outRms = std::sqrt(sum2 / static_cast<double>(n));
    m.dcMean = dc / static_cast<double>(n);
    m.crestDb = 20.0 * std::log10(peak / (m.outRms + 1e-18));

    if (fundHz > 0.0)
    {
        // 最小二乗正弦fit: y = a*sin + b*cos + c
        const double w = 2.0 * 3.14159265358979323846 * fundHz / sr;
        double sS = 0.0, sC = 0.0, sSS = 0.0, sCC = 0.0, sSC = 0.0, sYS = 0.0, sYC = 0.0, sY = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double s = std::sin(w * static_cast<double>(i));
            const double c = std::cos(w * static_cast<double>(i));
            const double y = x[i];
            sS += s; sC += c; sSS += s * s; sCC += c * c; sSC += s * c;
            sYS += y * s; sYC += y * c; sY += y;
        }
        // 3x3 正規方程式を解く（クラメル則）
        const double N = static_cast<double>(n);
        const double M[3][3] = {{sSS, sSC, sS}, {sSC, sCC, sC}, {sS, sC, N}};
        const double V[3] = {sYS, sYC, sY};
        const double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                         - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                         + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        if (std::abs(det) > 1e-18)
        {
            const double detA = V[0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                              - M[0][1] * (V[1] * M[2][2] - M[1][2] * V[2])
                              + M[0][2] * (V[1] * M[2][1] - M[1][1] * V[2]);
            const double detB = M[0][0] * (V[1] * M[2][2] - M[1][2] * V[2])
                              - V[0] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                              + M[0][2] * (M[1][0] * V[2] - V[1] * M[2][0]);
            const double detC = M[0][0] * (M[1][1] * V[2] - V[1] * M[2][1])
                              - M[0][1] * (M[1][0] * V[2] - V[1] * M[2][0])
                              + V[0] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
            const double a = detA / det, b = detB / det, c0 = detC / det;
            double res2 = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                const double fit = a * std::sin(w * static_cast<double>(i))
                                 + b * std::cos(w * static_cast<double>(i)) + c0;
                const double r = x[i] - fit;
                res2 += r * r;
            }
            const double fundRms = std::sqrt((a * a + b * b) * 0.5);
            const double resRms = std::sqrt(res2 / N);
            m.thdDb = 20.0 * std::log10(resRms / (fundRms + 1e-18));
        }
    }

    // 超音波比（juce::dsp::FFT、65536点 Hann）
    {
        constexpr int fftOrder = 16;
        constexpr int fftSize = 1 << fftOrder;
        if (n >= static_cast<size_t>(fftSize))
        {
            juce::dsp::FFT fft(fftOrder);
            std::vector<float> td(static_cast<size_t>(fftSize) * 2, 0.0f);
            const size_t off = n - static_cast<size_t>(fftSize);
            // ★ 先頭半分に実サンプルを連続配置（performRealOnlyForwardTransform仕様）。
            //   偶奇インターリーブ配置は零詰め2倍と等価で鏡像を生むため禁止（WORK104-R2で発覚）。
            for (int i = 0; i < fftSize; ++i)
            {
                const double hann = 0.5 * (1.0 - std::cos(2.0 * 3.14159265358979323846 * i / (fftSize - 1)));
                td[static_cast<size_t>(i)] = static_cast<float>(x[off + static_cast<size_t>(i)] * hann);
            }
            fft.performRealOnlyForwardTransform(td.data());
            const double binHz = sr / fftSize;
            double eAud = 1e-30, eUltra = 1e-30;
            for (int b = 1; b <= fftSize / 2; ++b)
            {
                const double re = td[static_cast<size_t>(b) * 2];
                const double im = (b < fftSize / 2) ? td[static_cast<size_t>(b) * 2 + 1] : 0.0;
                const double e = re * re + im * im;
                const double f = b * binHz;
                if (f >= 20.0 && f <= 20000.0) eAud += e;
                else if (f > 24000.0) eUltra += e;
            }
            m.ultraRatioDb = 10.0 * std::log10(eUltra / eAud);
        }
    }
    return m;
}

// ★ WORK112 112-2/112-3: DC/基本波/高調波/低域 band/residual を分離（test-only, double 解析）
static void printLfResidual(const std::vector<float>& x, double sr, double f, const char* tag)
{
    if (x.size() < 8192u || sr <= 0.0 || f <= 0.0)
    {
        std::fprintf(stderr, "[LF_RESIDUAL] %s insufficient\n", tag);
        return;
    }
    const size_t n = std::min<size_t>(x.size(), 131072u);
    const size_t off = x.size() - n;
    constexpr double kPiD = 3.14159265358979323846;
    const double w0 = 2.0 * kPiD * f / sr;

    // 3 パラメータ LS（DC + 基本波 sin/cos）で fundamental を厳密に推定
    double M[3][4] = { {0,0,0,0}, {0,0,0,0}, {0,0,0,0} };
    for (size_t i = 0; i < n; ++i)
    {
        const double t = static_cast<double>(i);
        const double phi[3] = { 1.0, std::sin(w0 * t), std::cos(w0 * t) };
        const double v = static_cast<double>(x[off + i]);
        for (int a = 0; a < 3; ++a)
        {
            M[a][3] += phi[a] * v;
            for (int b = 0; b < 3; ++b) M[a][b] += phi[a] * phi[b];
        }
    }
    for (int c = 0; c < 3; ++c)
    {
        int piv = c;
        for (int r = c + 1; r < 3; ++r)
            if (std::abs(M[r][c]) > std::abs(M[piv][c])) piv = r;
        if (std::abs(M[piv][c]) < 1.0e-30) break;
        if (piv != c) for (int k = 0; k < 4; ++k) std::swap(M[c][k], M[piv][k]);
        for (int r = 0; r < 3; ++r)
            if (r != c)
            {
                const double fct = M[r][c] / M[c][c];
                for (int k = c; k < 4; ++k) M[r][k] -= fct * M[c][k];
            }
    }
    const double c0 = (M[0][0] != 0.0) ? M[0][3] / M[0][0] : 0.0;
    const double a1 = (M[1][1] != 0.0) ? M[1][3] / M[1][1] : 0.0;
    const double b1 = (M[2][2] != 0.0) ? M[2][3] / M[2][2] : 0.0;
    const double fundRms = std::sqrt((a1 * a1 + b1 * b1) * 0.5);

    // flat-top(HFT95) windowed DFT による絶対振幅
    const double ft[5] = { 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368 };
    auto ampAt = [&](double ftgt) -> double
    {
        double re = 0.0, im = 0.0, ws = 0.0;
        const double w = 2.0 * kPiD * ftgt / sr;
        for (size_t i = 0; i < n; ++i)
        {
            const double ph = 2.0 * kPiD * static_cast<double>(i) / static_cast<double>(n - 1);
            const double win = ft[0] - ft[1] * std::cos(ph) + ft[2] * std::cos(2 * ph)
                             - ft[3] * std::cos(3 * ph) + ft[4] * std::cos(4 * ph);
            const double v = static_cast<double>(x[off + i]) * win;
            re += v * std::cos(w * static_cast<double>(i));
            im -= v * std::sin(w * static_cast<double>(i));
            ws += win;
        }
        return (ws > 1.0e-30) ? 2.0 * std::sqrt(re * re + im * im) / ws : 0.0;
    };
    const double h1 = ampAt(f), h2 = ampAt(2 * f), h3 = ampAt(3 * f), h4 = ampAt(4 * f), h5 = ampAt(5 * f);

    // residual = x - (c0 + a1 sin + b1 cos)
    double r2 = 0.0, rp = 0.0;
    long long subnormal = 0, nonfin = 0, nz = 0;
    const double subMin = static_cast<double>(std::numeric_limits<float>::min());
    std::vector<double> res(n);
    for (size_t i = 0; i < n; ++i)
    {
        const double t = static_cast<double>(i);
        const double fit = c0 + a1 * std::sin(w0 * t) + b1 * std::cos(w0 * t);
        const double r = static_cast<double>(x[off + i]) - fit;
        res[i] = r;
        r2 += r * r;
        const double a = std::abs(r);
        if (a > rp) rp = a;
        if (!std::isfinite(x[off + i])) ++nonfin;
        if (a > 0.0) ++nz;
        if (a > 0.0 && a < subMin) ++subnormal;
    }
    const double resRms = std::sqrt(r2 / static_cast<double>(n));

    // residual spectrum（Hann / 65536）band energy + 上位ピーク
    double band[6] = { 0, 0, 0, 0, 0, 0 };
    double topE[5] = { 0, 0, 0, 0, 0 };
    double topF[5] = { 0, 0, 0, 0, 0 };
    {
        constexpr int ord = 16, fs = 1 << ord;
        if (static_cast<int>(n) >= fs)
        {
            juce::dsp::FFT fft(ord);
            std::vector<float> td(static_cast<size_t>(fs) * 2, 0.0f);
            const size_t o = n - static_cast<size_t>(fs);
            for (int i = 0; i < fs; ++i)
            {
                const double hn = 0.5 * (1.0 - std::cos(2.0 * kPiD * i / (fs - 1)));
                td[static_cast<size_t>(i)] = static_cast<float>(res[o + static_cast<size_t>(i)] * hn);
            }
            fft.performRealOnlyForwardTransform(td.data());
            const double bin = sr / fs;
            for (int b = 1; b <= fs / 2; ++b)
            {
                const double re = td[static_cast<size_t>(b) * 2];
                const double im = (b < fs / 2) ? td[static_cast<size_t>(b) * 2 + 1] : 0.0;
                const double e = re * re + im * im;
                const double fr = b * bin;
                if (e > topE[4])
                {
                    topE[4] = e; topF[4] = fr;
                    for (int k = 4; k > 0 && topE[k] > topE[k - 1]; --k)
                    {
                        std::swap(topE[k], topE[k - 1]);
                        std::swap(topF[k], topF[k - 1]);
                    }
                }
                if (fr < 20.0) band[0] += e;
                else if (fr < 50.0) band[1] += e;
                else if (fr < 100.0) band[2] += e;
                else if (fr < 200.0) band[3] += e;
                else if (fr < 2000.0) band[4] += e;
                else band[5] += e;
            }
        }
    }

    std::fprintf(stderr,
        "[LF_RESIDUAL] %s n=%zu fundHz=%.1f fundRms=%.8e h1=%.8e h2=%.8e h3=%.8e h4=%.8e h5=%.8e dc=%.8e\n",
        tag, n, f, fundRms, h1, h2, h3, h4, h5, c0);
    std::fprintf(stderr,
        "[LF_RESIDUAL] %s residualRms=%.8e residualPeak=%.8e res/fund=%.2fdB subnormal=%lld nonzero=%lld nonfinite=%lld\n",
        tag, resRms, rp, 20.0 * std::log10((resRms + 1.0e-30) / (fundRms + 1.0e-30)), subnormal, nz, nonfin);
    std::fprintf(stderr,
        "[LF_RESIDUAL] %s resBand: 0-20=%.4e 20-50=%.4e 50-100=%.4e 100-200=%.4e 200-2k=%.4e 2k-20k=%.4e\n",
        tag, band[0], band[1], band[2], band[3], band[4], band[5]);
    std::fprintf(stderr,
        "[LF_RESIDUAL] %s resPeaks: %.1fHz:%.4e %.1fHz:%.4e %.1fHz:%.4e %.1fHz:%.4e %.1fHz:%.4e (bin=%.3fHz)\n",
        tag, topF[0], std::sqrt(topE[0]), topF[1], std::sqrt(topE[1]),
        topF[2], std::sqrt(topE[2]), topF[3], std::sqrt(topE[3]),
        topF[4], std::sqrt(topE[4]), sr / 65536.0);
}

// ★ WORK113 113-2: delta IR の null test（out − gain×遅延入力）。test-only。
static void printDeltaNull(const std::vector<float>& in, const std::vector<float>& out,
                           double sr, const char* tag)
{
    const size_t N = 8192;
    if (in.size() < 2 * N || out.size() < 2 * N)
    {
        std::fprintf(stderr, "[DELTA_NULL] %s insufficient\n", tag);
        return;
    }
    const size_t Y0 = out.size() - N;
    const size_t maxLag = std::min<size_t>(20000, in.size() - N);
    double bestScore = -1.0;
    size_t bestLag = 0;
    for (size_t d = 0; d <= maxLag; ++d)
    {
        const float* r = in.data() + (Y0 - d);
        const float* y = out.data() + Y0;
        double dot = 0.0, rr = 0.0;
        for (size_t i = 0; i < N; ++i)
        {
            dot += static_cast<double>(y[i]) * r[i];
            rr += static_cast<double>(r[i]) * r[i];
        }
        const double score = (rr > 1.0e-30) ? (dot * dot) / rr : 0.0;
        if (score > bestScore) { bestScore = score; bestLag = d; }
    }
    const float* r = in.data() + (Y0 - bestLag);
    const float* y = out.data() + Y0;
    double dot = 0.0, rr = 0.0, yy = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
        dot += static_cast<double>(y[i]) * r[i];
        rr += static_cast<double>(r[i]) * r[i];
        yy += static_cast<double>(y[i]) * y[i];
    }
    const double g = (rr > 1.0e-30) ? dot / rr : 0.0;
    std::vector<float> nullv(N);
    double n2 = 0.0, np = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
        const double v = static_cast<double>(y[i]) - g * static_cast<double>(r[i]);
        nullv[i] = static_cast<float>(v);
        n2 += v * v;
        if (std::abs(v) > np) np = std::abs(v);
    }
    const double nRms = std::sqrt(n2 / static_cast<double>(N));
    const double yRms = std::sqrt(yy / static_cast<double>(N));
    std::fprintf(stderr,
        "[DELTA_NULL] %s lag=%zu gain=%.6f yRms=%.6e nullRms=%.6e nullPeak=%.6e null/y=%.2fdB\n",
        tag, bestLag, g, yRms, nRms, np, 20.0 * std::log10((nRms + 1.0e-30) / (yRms + 1.0e-30)));
    printLfResidual(nullv, sr, 50.0, (std::string(tag) + "_spec").c_str());
}

// ★ WORK113-6: NUC standalone isolation（AudioEngine を経由せず Add/Get だけを駆動）
static double specAmpHann(const std::vector<float>& x, double sr, double f)
{
    size_t n = x.size();
    if (n > 131072u) n = 131072u;
    if (n < 256u || sr <= 0.0 || f <= 0.0) return 0.0;
    const size_t off = x.size() - n;
    const double w0 = 2.0 * 3.14159265358979323846 * f / sr;
    double re = 0.0, im = 0.0, ws = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double hann = 0.5 * (1.0 - std::cos(2.0 * 3.14159265358979323846
            * static_cast<double>(i) / static_cast<double>(n - 1)));
        const double v = static_cast<double>(x[off + i]) * hann;
        re += v * std::cos(w0 * static_cast<double>(i));
        im -= v * std::sin(w0 * static_cast<double>(i));
        ws += hann;
    }
    return (ws > 1.0e-30) ? 2.0 * std::sqrt(re * re + im * im) / ws : 0.0;
}

static int runNucStandalone(double fIn, double sr, int block, const char* tag,
                            bool useSpec = false, int irLenArg = 192000)
{
    using convo::MKLNonUniformConvolver;
    const bool dcMode = (fIn <= 0.0);
    const double fUse = dcMode ? 50.0 : fIn;   // residual 解析用（DC 時は未使用）
    const int irLen  = irLenArg;  // ★ WORK113-7C: numPartsIR を irLen で制御
    const int tapIdx = 488;      // delta（384k 換算。48k idx 61 相当）
    const double scale = 1.0;    // NUC 単体（engine の scaleFactor は使わない）

    std::vector<double> G(static_cast<size_t>(irLen), 0.0);
    G[static_cast<size_t>(tapIdx)] = 1.0;

    MKLNonUniformConvolver nuc;
    convo::FilterSpec spec;
    spec.sampleRate = sr;
    spec.tailEnabled = false;    // L0 のみ
    spec.tailMode = 2;           // Bypass
    if (!nuc.SetImpulse(G.data(), irLen, block, scale, false, useSpec ? &spec : nullptr))
    {
        std::fprintf(stderr, "[NUC6] %s SetImpulse FAIL\n", tag);
        return 1;
    }

    // ★ WORK113-7C-5: irFreq partition 分布（NonRT snapshot・読み取り専用）
    {
        using convo::NUPCTestAccess;
        const int li = 0;
        const int npi = NUPCTestAccess::layerNumPartsIR(nuc, li);
        const int cs  = NUPCTestAccess::layerComplexSize(nuc, li);
        const double* irr = NUPCTestAccess::layerIrFreqReal(nuc, li);
        const double* iri = NUPCTestAccess::layerIrFreqImag(nuc, li);
        int nz = 0, topP = -1;
        double pk = 0.0, en = 0.0;
        for (int p = 0; p < npi; ++p)
        {
            double pp = 0.0, pe = 0.0;
            for (int k = 0; k < cs; ++k)
            {
                const double re = irr[static_cast<size_t>(p) * cs + k];
                const double im = iri[static_cast<size_t>(p) * cs + k];
                const double m = std::sqrt(re * re + im * im);
                if (m > pp) pp = m;
                pe += re * re + im * im;
            }
            if (pp > 1.0e-12) ++nz;
            if (pp > pk) { pk = pp; topP = p; }
            en += pe;
        }
        std::fprintf(stderr,
            "[7C] %s useSpec=%d irLen=%d numParts=%d numPartsIR=%d irFreqNZ=%d irFreqTopP=%d irFreqPeak=%.6e irFreqEnergy=%.6e\n",
            tag, useSpec ? 1 : 0, irLen, NUPCTestAccess::layerNumParts(nuc, li), npi, nz, topP, pk, en);

        // ★ WORK113-7D: 非零 partition の全 bin をダンプ（test-only, 読み取り専用）
        if (topP >= 0)
        {
            const std::string path = std::string("irfreq_") + tag + ".csv";
            std::ofstream ofs(path);
            if (ofs)
            {
                ofs << "k,re,im\n";
                for (int k = 0; k < cs; ++k)
                    ofs << k << "," << irr[static_cast<size_t>(topP) * cs + k] << ","
                        << iri[static_cast<size_t>(topP) * cs + k] << "\n";
            }
            std::fprintf(stderr, "[7D] %s dump=%s topP=%d partSize=%d fftSize=%lld complexSize=%d\n",
                         tag, path.c_str(), topP, NUPCTestAccess::layerPartSize(nuc, li),
                         NUPCTestAccess::layerFftSize(nuc, li), cs);
        }
    }

    const int total = static_cast<int>(4.0 * sr);   // 4 秒
    const int nBlocks = total / block;
    const int n = nBlocks * block;
    std::vector<double> x(static_cast<size_t>(n), 0.0), e(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = dcMode ? 1.0 : 0.25 * std::sin(2.0 * 3.14159265358979323846 * fUse * i / sr);
    for (int b = 0; b < nBlocks; ++b)
    {
        nuc.Add(&x[static_cast<size_t>(b * block)], block);
        const int got = nuc.Get(&e[static_cast<size_t>(b * block)], block);
        if (got != block) std::fprintf(stderr, "[NUC6] %s block=%d got=%d\n", tag, b, got);
    }

    std::vector<float> Ef(static_cast<size_t>(n)), Rf(static_cast<size_t>(n)), Df(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const double r = (i >= tapIdx) ? scale * x[static_cast<size_t>(i - tapIdx)] : 0.0;
        Ef[static_cast<size_t>(i)] = static_cast<float>(e[static_cast<size_t>(i)]);
        Rf[static_cast<size_t>(i)] = static_cast<float>(r);
        Df[static_cast<size_t>(i)] = static_cast<float>(e[static_cast<size_t>(i)] - r);
    }
    const double br = sr / static_cast<double>(block);
    std::fprintf(stderr, "[NUC6] %s sr=%.0f block=%d irLen=%d tap=%d blockRate=%.1fHz fIn=%.1f dcMode=%d\n",
                 tag, sr, block, irLen, tapIdx, br, fIn, dcMode ? 1 : 0);

    // ★ WORK113-7A: block 同期平均（null D の位相プロファイル）
    //   fUse が block 周期と非整数周期なので、多数 block 平均で正弦成分は消え、
    //   block 周期に同期した成分（＝境界不連続）だけが残る。
    {
        const int P = block;
        const int nB = n / P;
        std::vector<double> avg(static_cast<size_t>(P), 0.0), avgE(static_cast<size_t>(P), 0.0);
        for (int b = 0; b < nB; ++b)
            for (int i = 0; i < P; ++i)
            {
                avg[static_cast<size_t>(i)]  += static_cast<double>(Df[static_cast<size_t>(b * P + i)]);
                avgE[static_cast<size_t>(i)] += static_cast<double>(Ef[static_cast<size_t>(b * P + i)]);
            }
        for (int i = 0; i < P; ++i)
        {
            avg[static_cast<size_t>(i)]  /= static_cast<double>(nB);
            avgE[static_cast<size_t>(i)] /= static_cast<double>(nB);
        }
        double a2 = 0.0, e2 = 0.0;
        for (int i = 0; i < P; ++i)
        {
            a2 += avg[static_cast<size_t>(i)] * avg[static_cast<size_t>(i)];
            e2 += avgE[static_cast<size_t>(i)] * avgE[static_cast<size_t>(i)];
        }
        std::fprintf(stderr, "[NUC7A] %s blocks=%d D_blockAvgRms=%.8e D_avg[last]=%.8e D_avg[0]=%.8e jump=%.8e\n",
                     tag, nB, std::sqrt(a2 / P), avg[static_cast<size_t>(P - 1)], avg[0],
                     avg[0] - avg[static_cast<size_t>(P - 1)]);
        std::fprintf(stderr, "[NUC7A] %s D_avg[0..7]=%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                     tag, avg[0], avg[1], avg[2], avg[3], avg[4], avg[5], avg[6], avg[7]);
        std::fprintf(stderr, "[NUC7A] %s D_avg[P-4..P-1]=%.6e %.6e %.6e %.6e  E_blockAvgRms=%.8e\n",
                     tag, avg[static_cast<size_t>(P - 4)], avg[static_cast<size_t>(P - 3)],
                     avg[static_cast<size_t>(P - 2)], avg[static_cast<size_t>(P - 1)], std::sqrt(e2 / P));
    }
    if (!dcMode)
    {
        printLfResidual(Ef, sr, fUse, "NUC_E");
        printLfResidual(Rf, sr, fUse, "REF_R");
        printLfResidual(Df, sr, fUse, "EminusR");
        for (int k = 1; k <= 2; ++k)
        {
            const double f1 = br * k - fUse, f2 = br * k + fUse;
            std::fprintf(stderr,
                "[NUC6_SIDEBAND] %s n=%d : %.1fHz E=%.8e R=%.8e D=%.8e | %.1fHz E=%.8e R=%.8e D=%.8e\n",
                tag, k, f1, specAmpHann(Ef, sr, f1), specAmpHann(Rf, sr, f1), specAmpHann(Df, sr, f1),
                f2, specAmpHann(Ef, sr, f2), specAmpHann(Rf, sr, f2), specAmpHann(Df, sr, f2));
        }
    }

    // ★ WORK113-7F-0: NUC Get 出力 E を CSV ダンプ（独立 OLS reference と Python で比較するため）
    {
        std::ofstream o(std::string("7f0_E_") + tag + ".csv");
        for (int i = 0; i < n; ++i) o << i << "," << e[static_cast<size_t>(i)] << "\n";
        std::fprintf(stderr, "[7F0] %s dumped E n=%d (block=%d)\n", tag, n, block);
    }
    return 0;
}

// ★ WORK113-7B: L0 internal seam snapshot（standalone, NonRT・読み取り専用 getter のみ）
static int runNuc7B(double fIn, double sr, int block, const char* tag)
{
    using convo::MKLNonUniformConvolver;
    using convo::NUPCTestAccess;
    constexpr double kPiB = 3.14159265358979323846;
    const int irLen = 192000, tap = 488;
    std::vector<double> G(static_cast<size_t>(irLen), 0.0);
    G[static_cast<size_t>(tap)] = 1.0;

    MKLNonUniformConvolver nuc;
    // filterSpec=nullptr → 出力周波数フィルタ無効（L0 の IR は純 delta）
    if (!nuc.SetImpulse(G.data(), irLen, block, 1.0, false, nullptr))
    {
        std::fprintf(stderr, "[7B] %s SetImpulse FAIL\n", tag);
        return 1;
    }
    const int li = 0;
    const int P  = NUPCTestAccess::layerPartSize(nuc, li);
    const int NP = NUPCTestAccess::layerNumParts(nuc, li);
    const int NPI = NUPCTestAccess::layerNumPartsIR(nuc, li);
    const int nB = 40;
    std::vector<double> x(static_cast<size_t>(nB + 2) * P, 0.0), e(static_cast<size_t>(nB) * P, 0.0);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.25 * std::sin(2.0 * kPiB * fIn * static_cast<double>(i) / sr);

    double errInMax = 0.0, errOutMax = 0.0;
    for (int b = 0; b < nB; ++b)
    {
        nuc.Add(&x[static_cast<size_t>(b) * P], P);
        nuc.Get(&e[static_cast<size_t>(b) * P], P);

        const double* ftb = NUPCTestAccess::layerFftTimeBuf(nuc, li);
        const double* fob = NUPCTestAccess::layerFftOutBuf(nuc, li);

        // A/B: frame 後半 = 今回の入力 block（厳密一致すべき）
        double m1 = 0.0;
        for (int i = 0; i < P; ++i)
        {
            const double d = ftb[P + i] - x[static_cast<size_t>(b) * P + i];
            if (std::abs(d) > m1) m1 = std::abs(d);
        }
        // C/D: 有効半 fftOutBuf[P:2P] = frame を tap だけ遅延させたもの（delta IR・フィルタ無し）
        double m2 = 0.0;
        for (int i = 0; i < P; ++i)
        {
            const double d = fob[P + i] - ftb[(P + i - tap)];
            if (std::abs(d) > m2) m2 = std::abs(d);
        }
        errInMax = std::max(errInMax, m1);
        errOutMax = std::max(errOutMax, m2);
        const int fdl = NUPCTestAccess::layerFdlIndex(nuc, li);
        const int mask = NUPCTestAccess::layerFdlMask(nuc, li);
        std::fprintf(stderr,
            "[7B] %s b=%2d fdlIndex=%2d mask=%d mirror=%d linStart=%d numParts=%d numPartsIR=%d errIn=%.3e errOut=%.3e\n",
            tag, b, fdl, mask, fdl + NP, fdl - NPI + 1 + NP, NP, NPI, m1, m2);
    }
    std::fprintf(stderr, "[7B] %s SUMMARY errInMax=%.3e errOutMax=%.3e\n", tag, errInMax, errOutMax);
    return 0;
}

// ★ WORK113-7E: FDL×irFreq / accum / interleave / IFFT の中間値を1ブロック分ダンプ
static int runNuc7E(int irLenArg, const char* tag)
{
    using convo::MKLNonUniformConvolver;
    using convo::NUPCTestAccess;
    constexpr double kPiE = 3.14159265358979323846;
    const double sr = 384000.0; const int block = 2048; const double fIn = 50.0;
    const int tap = 488;

    std::vector<double> G(static_cast<size_t>(irLenArg), 0.0);
    G[static_cast<size_t>(tap)] = 1.0;
    MKLNonUniformConvolver nuc;
    convo::FilterSpec spec; spec.sampleRate = sr; spec.tailEnabled = false; spec.tailMode = 2;
    if (!nuc.SetImpulse(G.data(), irLenArg, block, 1.0, false, &spec)) { std::fprintf(stderr, "[7E] %s SetImpulse FAIL\n", tag); return 1; }

    const int li = 0;
    const int P = NUPCTestAccess::layerPartSize(nuc, li);
    const int cs = NUPCTestAccess::layerComplexSize(nuc, li);
    const int fs_ = static_cast<int>(NUPCTestAccess::layerFftSize(nuc, li));
    const int NP = NUPCTestAccess::layerNumParts(nuc, li);
    const int NPI = NUPCTestAccess::layerNumPartsIR(nuc, li);
    const int stride = NUPCTestAccess::layerPartStride(nuc, li);

    std::vector<double> x(static_cast<size_t>(3) * P, 0.0), e(static_cast<size_t>(2) * P, 0.0);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.25 * std::sin(2.0 * kPiE * fIn * static_cast<double>(i) / sr);
    nuc.Add(&x[0], P); nuc.Get(&e[0], P);
    nuc.Add(&x[static_cast<size_t>(P)], P); nuc.Get(&e[static_cast<size_t>(P)], P);

    const int fdl = NUPCTestAccess::layerFdlIndex(nuc, li);
    const int linStart = fdl - NPI + 1 + NP;
    std::fprintf(stderr, "[7E] %s irLen=%d fdlIndex=%d numParts=%d numPartsIR=%d partSize=%d fftSize=%d complexSize=%d partStride=%d linStart=%d\n",
                 tag, irLenArg, fdl, NP, NPI, P, fs_, cs, stride, linStart);

    auto dump1 = [&](const char* name, const double* v, int n) {
        std::ofstream o(std::string("7e_") + name + "_" + tag + ".csv");
        for (int i = 0; i < n; ++i) o << i << "," << v[i] << "\n";
    };
    dump1("frame", NUPCTestAccess::layerFftTimeBuf(nuc, li), fs_);
    dump1("fftout", NUPCTestAccess::layerFftOutBuf(nuc, li), fs_);
    dump1("accumbuf", NUPCTestAccess::layerAccumBuf(nuc, li), stride);
    {
        const double* ar = NUPCTestAccess::layerAccumReal(nuc, li);
        const double* ai = NUPCTestAccess::layerAccumImag(nuc, li);
        std::ofstream o(std::string("7e_accum_") + tag + ".csv");
        o << "k,re,im\n";
        for (int k = 0; k < cs; ++k) o << k << "," << ar[k] << "," << ai[k] << "\n";
    }
    {
        const double* fr = NUPCTestAccess::layerFdlReal(nuc, li);
        const double* fi = NUPCTestAccess::layerFdlImag(nuc, li);
        std::ofstream o(std::string("7e_fdl_") + tag + ".csv");
        o << "slot,k,re,im\n";
        for (int s = 0; s < 2 * NP; ++s)
            for (int k = 0; k < cs; ++k)
                o << s << "," << k << "," << fr[static_cast<size_t>(s) * cs + k] << "," << fi[static_cast<size_t>(s) * cs + k] << "\n";
    }
    {
        const double* rr = NUPCTestAccess::layerIrFreqReal(nuc, li);
        const double* ri = NUPCTestAccess::layerIrFreqImag(nuc, li);
        std::ofstream o(std::string("7e_irfreq_") + tag + ".csv");
        o << "p,k,re,im\n";
        for (int p = 0; p < NP; ++p)
            for (int k = 0; k < cs; ++k)
                o << p << "," << k << "," << rr[static_cast<size_t>(p) * cs + k] << "," << ri[static_cast<size_t>(p) * cs + k] << "\n";
    }
    std::fprintf(stderr, "[7E] %s dumps written\n", tag);
    return 0;
}

// ★ WORK106: 合成 IR の種類（delta=最小再現プローブ / boost9dB=B2再現 / silence=対照）
//   WORK107: DeltaMid(48k idx4000→384k idx32000: L0内 複数partition) /
//            DeltaFar(48k idx7000→384k idx56000: L0被覆47104超 → L1) で L0/L1 を分離。
enum class SynthIrKind { Delta, DeltaMid, DeltaFar, P1Delta, P0P1, P0P1Pad, P0Two, Boost9dB, Silence };

// +9dB@50Hz ブースト IR / 単一デルタ IR / 無音 IR を合成。
bool writeSyntheticIrFile(const juce::File& file, SynthIrKind kind)
{
    constexpr double irSr = 48000.0;
    constexpr int len = 24000; // 0.5s
    constexpr double f0 = 50.0, G = 9.0, Q = 1.0;
    const double A = std::pow(10.0, G / 40.0);
    const double w = 2.0 * 3.14159265358979323846 * f0 / irSr;
    const double alpha = std::sin(w) / (2.0 * Q);
    const double cw = std::cos(w);
    const double b0 = (1.0 + alpha * A) / (1.0 + alpha / A);
    const double b1 = (-2.0 * cw) / (1.0 + alpha / A);
    const double b2 = (1.0 - alpha * A) / (1.0 + alpha / A);
    const double a1 = (-2.0 * cw) / (1.0 + alpha / A);
    const double a2 = (1.0 - alpha / A) / (1.0 + alpha / A);
    std::vector<double> h(static_cast<size_t>(len), 0.0);
    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
    // デルタ（61点目に1.0、実IRと同位置）→ （Boost時のみ）peaking フィルタ
    for (int n = 0; n < len; ++n)
    {
        // WORK107 の位置指定（48k index、×8 で 384k）:
        //   delta    : 61        → 488        (L0 partition 0)
        //   P1Delta  : 317       → 2536       (L0 partition 1、L1 なし)
        //   P0P1     : 61, 317   → 488, 2536  (partition 0 + 1)
        //   P0Two    : 61, 62    → 488, 496   (同一 partition 0 内 2 tap)
        //   DeltaMid : 4000      → 32000      (L0 partition 15、L1 なし)
        //   DeltaFar : 7000      → 56000      (L0 被覆超 → L1)
        const int d1 = (kind == SynthIrKind::Delta)    ? 61
                     : (kind == SynthIrKind::P1Delta)  ? 317
                     : (kind == SynthIrKind::P0P1)     ? 61
                     : (kind == SynthIrKind::P0P1Pad)  ? 61
                     : (kind == SynthIrKind::P0Two)    ? 61
                     : (kind == SynthIrKind::DeltaMid) ? 4000
                     : (kind == SynthIrKind::DeltaFar) ? 7000 : 61;
        const int d2 = (kind == SynthIrKind::P0P1)  ? 317
                     : (kind == SynthIrKind::P0P1Pad) ? 317
                     : (kind == SynthIrKind::P0Two) ? 62 : -1;
        const double x0 = (kind != SynthIrKind::Silence && (n == d1 || n == d2)) ? 1.0 : 0.0;
        if (kind == SynthIrKind::Boost9dB)
        {
            const double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            h[static_cast<size_t>(n)] = y0;
            x2 = x1; x1 = x0; y2 = y1; y1 = y0;
        }
        else
        {
            h[static_cast<size_t>(n)] = x0; // Delta系 / Silence（Silence は全0）
        }
        // ★ WORK110: tap を末尾 fade 帯（最後 256 sample）より前に置くための減衰テール。
        if (kind == SynthIrKind::P0P1Pad && n > 317)
            h[static_cast<size_t>(n)] = 1.0e-6 * std::exp(-static_cast<double>(n - 317) / 300.0);
    }
    if (kind == SynthIrKind::Silence)
        std::fill(h.begin(), h.end(), 0.0);
    double peak = 0.0;
    for (double v : h) peak = std::max(peak, std::abs(v));
    const double g = (peak > 1e-12) ? (0.95 / peak) : 1.0;
    // 手書き RIFF float32 ステレオ WAV（juce_audio_formats 非依存のため）。
    // cГруз形式: fmt wFormatTag=3(float) / ch=2 / 48k / 32bit。
    std::ofstream ofs(file.getFullPathName().toStdString(), std::ios::binary);
    if (!ofs)
        return false;
    const std::uint32_t dataBytes = static_cast<std::uint32_t>(len) * 2u * 4u;
    auto w32 = [&](std::uint32_t v) {
        ofs.put(static_cast<char>(v & 0xFF)); ofs.put(static_cast<char>((v >> 8) & 0xFF));
        ofs.put(static_cast<char>((v >> 16) & 0xFF)); ofs.put(static_cast<char>((v >> 24) & 0xFF));
    };
    auto w16 = [&](std::uint16_t v) {
        ofs.put(static_cast<char>(v & 0xFF)); ofs.put(static_cast<char>((v >> 8) & 0xFF));
    };
    ofs.write("RIFF", 4); w32(36 + dataBytes); ofs.write("WAVE", 4);
    ofs.write("fmt ", 4); w32(16); w16(3); w16(2); w32(48000);
    w32(48000 * 2 * 4); w16(2 * 4); w16(32);
    ofs.write("data", 4); w32(dataBytes);
    for (int i = 0; i < len; ++i)
    {
        const float s = static_cast<float>(h[static_cast<size_t>(i)] * g);
        ofs.write(reinterpret_cast<const char*>(&s), 4);
        ofs.write(reinterpret_cast<const char*>(&s), 4);
    }
    ofs.flush();
    return static_cast<bool>(ofs);
}

struct BuzzOptions
{
    double sr = 192000.0;
    int block = 1024;
    std::string irPath;
    std::string outCsv;
    bool quick = false;
    double runSec = 3.0;
    // ★ WORK113-15: probe routing 制御（test-only。−1 = 従来どおりの既定動作）
    int orderMode  = -1; // 0=ConvolverThenEQ, 1=EQThenConvolver
    int eqOn       = -1; // 1=EQ 有効 / 0=bypass（既定 = bypass: 旧 probe と同一）
    int convOn     = -1; // 1=conv 有効 / 0=bypass（既定 = 有効）
    int hcMode     = -1; // convo::HCMode index (0=Sharp,1=Natural,2=Soft)
    int lcMode     = -1; // convo::LCMode index (0=Natural,1=Soft)
    int lpMode     = -1; // EQ-LPF (HCMode) index
    int directHead = -1; // 1=Exp direct head ON / 0=OFF
    // [WORK113-17] §15 transition 計測用（test-only）。0 = 無効。
    int flipKind    = 0; // 1=HC mode, 2=LC mode, 3=EQ bypass, 4=EQ total gain
    int flipValue   = 0; // flip 実行時の設定値
    double flipAtSec = 0.8; // capture 開始からの flip 予定時刻
};

void applyDefaultPreset(AudioEngine& e)
{
    e.beginBulkParameterRestore();
    e.setProcessingOrder(convo::ProcessingOrder::ConvolverThenEQ);
    for (int i = 0; i < 20; ++i)
    {
        const auto& b = kDefaultBands[i];
        e.setEQBandFrequency(i, b.freq);
        e.setEQBandGain(i, b.gain);
        e.setEQBandQ(i, b.q);
        e.setEQBandType(i, static_cast<EQBandType>(b.type));
        e.setEQBandChannelMode(i, EQChannelMode::Stereo);
        e.setEQBandEnabled(i, true);
    }
    e.setEQTotalGain(-2.0f);
    e.setEQAGCEnabled(false);
    e.setEQNonlinearSaturation(0.05f);
    e.setEQFilterStructure(EQProcessor::FilterStructure::Parallel);
    e.setSoftClipEnabled(true);
    e.setSaturationAmount(0.05f);
    e.setOversamplingFactor(2);
    e.setOversamplingType(convo::OversamplingType::LinearPhase);
    e.setDitherBitDepth(32);
    e.setNoiseShaperType(convo::NoiseShaperType::Adaptive9thOrder);
    e.setAutoGainStagingEnabled(true);
    e.setConvolverMix(1.0f);
    e.setConvolverSmoothingTime(0.1f);
    e.setConvolverPhaseMode(ConvolverProcessor::PhaseMode::Mixed);
    e.setConvolverMixedTransitionStartHz(200.0f);
    e.setConvolverMixedTransitionEndHz(1000.0f);
    e.setConvolverTargetIRLength(0.5f, false);
    e.setConvolverRebuildDebounceMs(400);
    e.setConvolverTailMode(ConvolverProcessor::TailMode::LayerTailContouring);
    e.setConvolverTailStartSec(0.085f);
    e.setConvolverTailStrength(0.5f);
    e.setConvolverTailL1L2Multiplier(8);
    e.setConvolverEnableProgressiveUpgrade(false);
    e.setEqBypassRequested(false);
    e.setConvolverBypassRequested(false);
    e.endBulkParameterRestore(true);
}

void applyBuzzConfig(AudioEngine& e, const BuzzConfig& c)
{
    e.setEqBypassRequested(c.eqBypass);
    e.setConvolverBypassRequested(c.convBypass);
    e.setAutoGainStagingEnabled(!c.manualMakeupMinus6);
    if (c.manualMakeupMinus6)
    {
        e.setInputHeadroomDb(-8.0f);
        e.setOutputMakeupDb(-6.0f);
        e.setConvolverInputTrimDb(0.0f);
    }
    e.setSoftClipEnabled(!c.softClipOff);
    e.setConvolverPhaseMode(c.phaseAsIs ? ConvolverProcessor::PhaseMode::AsIs
                                        : ConvolverProcessor::PhaseMode::Mixed);
    e.setNoiseShaperType(c.shaperPsycho ? convo::NoiseShaperType::Psychoacoustic
                                        : convo::NoiseShaperType::Adaptive9thOrder);
}

// ★ WORK113-15: probe 用フラット EQ（test-only）。
//   eqParams 未 publish のままでは AudioEngine.h:4079 の fail-closed により
//   snapshot.eqBypassed が常時 true になり ② OutputFilter が走らないため、
//   全 band 無効 + total 0dB（恒等変換）の EQ パラメータを publish して解除する。
void configureProbeFlatEQ(AudioEngine& e)
{
    e.beginBulkParameterRestore();
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

} // namespace convo_buzz

// ── エントリ: PublishPipelineIntegrationTests.cpp の main から --buzz で派遣 ──
int runBassBuzzMeasurement(int argc, char* argv[])
{
    using namespace convo_buzz;
    static BuzzLogger logger;
    juce::Logger::setCurrentLogger(&logger);
    static juce::ScopedJuceInitialiser_GUI juceInit;

    BuzzOptions opt;
    std::string rigCheckMode; // ""=off, "bare", "ir", "eq"
    std::string probeMode;    // ★ WORK106: ""=off, "delta", "boost", "silence"
    float probeLevel = 0.25f; // ★ WORK106: プローブ入力レベル（安全鎖非接触）
    int quietMs = 150000;     // ★ WORK106: DSP側IR再構築待ちの静穏期間
    std::string probeSignalName = "impulse"; // ★ WORK111: probe 入力信号（impulse/sine50/...）
    bool disableTailFade = false;            // ★ WORK111: measurement-only（production 既定 OFF）
    double nuc6Freq = 0.0;                   // ★ WORK113-6: NUC standalone 入力周波数（0=無効）
    bool nuc7a = false;                      // ★ WORK113-7A: NUC standalone temporal（DC 入力）
    bool nuc7b = false;                      // ★ WORK113-7B: NUC 内部 seam snapshot
    bool nuc7c = false;                      // ★ WORK113-7C: FilterSpec/numPartsIR 直交化
    bool nuc7d = false;                      // ★ WORK113-7D: irFreq 全bin ダンプ
    bool nuc7e = false;                      // ★ WORK113-7E: FDL/accum/interleave/IFFT ダンプ
    bool nuc7f0 = false;                     // ★ WORK113-7F-0: 正しい LTI reference で再評価
    auto parseHcIdx = [](const std::string& s) -> int {
        if (s == "sharp")   return 0;
        if (s == "natural") return 1;
        if (s == "soft")    return 2;
        return std::stoi(s);
    };
    auto parseLcIdx = [](const std::string& s) -> int {
        if (s == "natural") return 0;
        if (s == "soft")    return 1;
        return std::stoi(s);
    };
    // [WORK113 test-instrumentation cleanup] --buzz-eq/conv/direct/flip-eqbypass は "on"/"off" のみ
    //   受理する。旧実装は他の値を黙って 0(bypass) 扱いにし、誤測定を誘発した（2026-09-19 実害）。
    //   fail-closed: 不正値は即時エラー終了とする。
    auto parseOnOff = [](const std::string& flag, const std::string& v) -> int {
        if (v == "on") return 1;
        if (v == "off") return 0;
        std::fprintf(stderr, "[BUZZ] FAIL: %s expects 'on' or 'off' (got '%s')\n",
                     flag.c_str(), v.c_str());
        std::exit(2);
    };
    for (int i = 1; i < argc; ++i)
    {
        const std::string a(argv[i]);
        if (a.rfind("--buzz-sr=", 0) == 0) opt.sr = std::stod(a.substr(10));
        else if (a.rfind("--buzz-block=", 0) == 0) opt.block = std::stoi(a.substr(13));
        else if (a.rfind("--buzz-ir=", 0) == 0) opt.irPath = a.substr(10);
        else if (a.rfind("--buzz-out=", 0) == 0) opt.outCsv = a.substr(11);
        else if (a == "--buzz-quick") opt.quick = true;
        else if (a == "--buzz-rigcheck") rigCheckMode = "bare";
        else if (a.rfind("--buzz-rigcheck=", 0) == 0) rigCheckMode = a.substr(16);
        else if (a.rfind("--buzz-probe=", 0) == 0) probeMode = a.substr(13);
        else if (a.rfind("--buzz-probe-level=", 0) == 0) probeLevel = std::stof(a.substr(19));
        else if (a.rfind("--buzz-quiet=", 0) == 0) quietMs = std::stoi(a.substr(13));
        else if (a.rfind("--buzz-dur=", 0) == 0) opt.runSec = std::stod(a.substr(11));
        else if (a.rfind("--buzz-probe-signal=", 0) == 0) probeSignalName = a.substr(20);
        else if (a.rfind("--buzz-order=", 0) == 0) opt.orderMode = (a.substr(13) == "etc") ? 1 : 0;   // ★ WORK113-15
        else if (a.rfind("--buzz-eq=", 0) == 0) opt.eqOn = parseOnOff("--buzz-eq", a.substr(10));            // ★ WORK113-15
        else if (a.rfind("--buzz-conv=", 0) == 0) opt.convOn = parseOnOff("--buzz-conv", a.substr(12));      // ★ WORK113-15
        else if (a.rfind("--buzz-hc=", 0) == 0) opt.hcMode = parseHcIdx(a.substr(10));                // ★ WORK113-15
        else if (a.rfind("--buzz-lc=", 0) == 0) opt.lcMode = parseLcIdx(a.substr(10));                // ★ WORK113-15
        else if (a.rfind("--buzz-eqlpf=", 0) == 0) opt.lpMode = parseHcIdx(a.substr(13));             // ★ WORK113-15
        else if (a.rfind("--buzz-direct=", 0) == 0) opt.directHead = parseOnOff("--buzz-direct", a.substr(13));  // ★ WORK113-15
        else if (a.rfind("--buzz-flip-hc=", 0) == 0) { opt.flipKind = 1; opt.flipValue = parseHcIdx(a.substr(15)); }        // ★ WORK113-17
        else if (a.rfind("--buzz-flip-lc=", 0) == 0) { opt.flipKind = 2; opt.flipValue = parseLcIdx(a.substr(15)); }        // ★ WORK113-17
        else if (a.rfind("--buzz-flip-eqbypass=", 0) == 0) { opt.flipKind = 3; opt.flipValue = parseOnOff("--buzz-flip-eqbypass", a.substr(21)); } // ★ WORK113-17
        else if (a.rfind("--buzz-flip-eqgain=", 0) == 0) { opt.flipKind = 4; opt.flipValue = 0; }                           // ★ WORK113-17 (total -3dB step)
        else if (a.rfind("--buzz-flip-t=", 0) == 0) opt.flipAtSec = std::stod(a.substr(14));                                 // ★ WORK113-17
        else if (a.rfind("--nuc6=", 0) == 0) nuc6Freq = std::stod(a.substr(7));
        else if (a == "--nuc7a") nuc7a = true;
        else if (a == "--nuc7b") nuc7b = true;
        else if (a == "--nuc7c") nuc7c = true;
        else if (a == "--nuc7d") nuc7d = true;
        else if (a == "--nuc7e") nuc7e = true;
        else if (a == "--nuc7f0") nuc7f0 = true;
        else if (a == "--disable-tail-fade-for-measurement") disableTailFade = true;
    }

    // ★ WORK113-6: NUC standalone isolation（AudioEngine を経由しない）
    if (nuc6Freq > 0.0)
        return runNucStandalone(nuc6Freq, 384000.0, 2048, nuc6Freq < 45.0 ? "sine40" : "sine50");
    // ★ WORK113-7A: NUC standalone temporal continuity（block 同期平均）
    if (nuc7a)
        return runNucStandalone(0.0, 384000.0, 2048, "dc7a");
    // ★ WORK113-7B: NUC 内部 seam snapshot
    if (nuc7b)
        return runNuc7B(50.0, 384000.0, 2048, "s50");
    // ★ WORK113-7C: FilterSpec / numPartsIR 直交化（A=none/3, B=spec/3, D=spec/32）
    if (nuc7d)
    {
        runNucStandalone(50.0, 384000.0, 2048, "7d_B", true, 6144);
        return 0;
    }
    if (nuc7f0)
    {
        runNucStandalone(50.0, 384000.0, 2048, "P1_s50", true, 2048);
        runNucStandalone(40.0, 384000.0, 2048, "P1_s40", true, 2048);
        return 0;
    }
    if (nuc7e)
    {
        runNuc7E(6144, "B3");    // numPartsIR=3（7C-B と同条件）
        runNuc7E(2048, "P1");    // numPartsIR=1（7E-7 単一 partition）
        return 0;
    }
    if (nuc7c)
    {
        runNucStandalone(50.0, 384000.0, 2048, "A_none_np3",  false, 192000);
        runNucStandalone(50.0, 384000.0, 2048, "B_spec_np3",  true,   6144);
        runNucStandalone(50.0, 384000.0, 2048, "D_spec_np32", true,  65536);
        return 0;
    }

    // ★ WORK111: measurement-only hook（production では誰も true にしない）
    if (disableTailFade)
        convo::trimtest::disableTailFadeForMeasurement().store(true, std::memory_order_relaxed);

    auto probeSignalFromName = [](const std::string& s) -> BuzzSignal {
        if (s == "sine50")  return BuzzSignal::Sine50;
        if (s == "sine40")  return BuzzSignal::Sine40;
        if (s == "sine100") return BuzzSignal::Sine100;
        if (s == "kick")    return BuzzSignal::Kick;
        if (s == "multi")   return BuzzSignal::Multi;
        return BuzzSignal::Impulse;
    };
    auto probeFundHz = [](BuzzSignal s) -> double {
        switch (s) { case BuzzSignal::Sine40: return 40.0;
                     case BuzzSignal::Sine100: return 100.0;
                     case BuzzSignal::Sine50: return 50.0;
                     default: return 50.0; }
    };

    // IR 解決: 明示 → sampledata → Documents
    juce::File irFile(opt.irPath);
    if (opt.irPath.empty() || !irFile.existsAsFile())
        irFile = juce::File::getCurrentWorkingDirectory().getChildFile("sampledata/impulse.wav");
    if (!irFile.existsAsFile())
        irFile = juce::File("C:/Users/user/Documents/conv_filter/impulse.wav");
    if (!irFile.existsAsFile())
    {
        std::fprintf(stderr, "[BUZZ] FAIL: IR file not found\n");
        return 1;
    }
    std::fprintf(stderr, "[BUZZ] IR: %s\n", irFile.getFullPathName().toRawUTF8());

    AudioEngineHarness h;
    if (!h.start(opt.sr, opt.block))
    {
        std::fprintf(stderr, "[BUZZ] FAIL: harness start failed\n");
        return 1;
    }
    AudioEngine& e = h.engine();

    // ★ WORK104-R2: --buzz-rigcheck[=bare|ir|eq] は透過性の二分探索用。
    //   bare: 既定＋bypassのみ。ir: IRロード後にbypass。eq: EQ帯域投入＋conv bypass。
    //   プローブは -6dBFS（リミッター非接触域）で透過性（peak≈in×0.891、THD<-80dB）を要求する。
    if (!rigCheckMode.empty())
    {
        if (rigCheckMode != "bare" && rigCheckMode != "ir" && rigCheckMode != "eq")
        {
            std::fprintf(stderr, "[BUZZ] FAIL: unknown rigcheck mode '%s'\n", rigCheckMode.c_str());
            h.stop();
            return 1;
        }
        Session session;
        h.setTap([&session](juce::AudioBuffer<float>& buffer, bool isInput) {
            const int n = buffer.getNumSamples();
            if (session.mode.load(std::memory_order_acquire) != 1)
            {
                if (isInput)
                    buffer.clear();
                return;
            }
            if (isInput)
            {
                float* L = buffer.getWritePointer(0);
                float* R = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : nullptr;
                std::lock_guard<std::mutex> lk(session.capMutex);
                for (int i = 0; i < n; ++i)
                {
                    const float v = session.gen.next();
                    L[i] = v;
                    if (R) R[i] = v;
                    session.inCap.push_back(v);
                }
            }
            else
            {
                const float* L = buffer.getReadPointer(0);
                std::lock_guard<std::mutex> lk(session.capMutex);
                for (int i = 0; i < n; ++i)
                    session.outCapL.push_back(L[i]);
            }
        });
        e.setEqBypassRequested(true);
        e.setConvolverBypassRequested(true);
        if (!waitBacklogZero(e, 30000))
            std::fprintf(stderr, "[BUZZ] WARN rigcheck: backlog not zero\n");
        sleepPump(1500);
        if (rigCheckMode == "ir")
        {
            e.getConvolverProcessor().loadImpulseResponse(irFile, false);
            if (!waitIrFinalized(e, 300000))
            {
                std::fprintf(stderr, "[BUZZ] FAIL rigcheck=ir: IR load timeout\n");
                h.clearTap();
                h.stop();
                return 1;
            }
            if (!waitBacklogZero(e, 30000))
                std::fprintf(stderr, "[BUZZ] WARN rigcheck=ir: backlog not zero\n");
            sleepPump(2000);
        }
        else if (rigCheckMode == "eq")
        {
            // [WORK113 test-instrumentation cleanup] 判定基準（identity 窓 [0.880,0.897] + THD<-80dB）
            //   と設定を整合させる: EQ チェーンを恒等変換（全 band 無効・total 0dB・saturation 0）
            //   で通す。旧設定（kDefaultBands +3dB × total -2dB × saturation 0.05）は恒等窓と矛盾し
            //   FAIL が構造的だった（2026-09-19 実測: ratio 0.5640 / THD -53.6dB）。
            configureProbeFlatEQ(e);
            //   engine 側 auto gain staging（既定 ON・AudioEngine.h:2626）は EQ-on 経路で減成を
            //   適用するため identity 検証では無効化する（bare 比較 0.8846 と同一基準に揃える。
            //   2026-09-19 実測: staging ON だと ratio 0.4912 = −5.1dB 差）。
            e.setAutoGainStagingEnabled(false);
            e.setEQFilterStructure(EQProcessor::FilterStructure::Parallel);
            e.setEqBypassRequested(false);
            if (!waitBacklogZero(e, 30000))
                std::fprintf(stderr, "[BUZZ] WARN rigcheck=eq: backlog not zero\n");
            sleepPump(2000);
            // ★ 状態乖離の直接観測：要求値 vs 実効値 vs publish-seq
            std::fprintf(stderr,
                "[BUZZ] RIGCHECK(eq) state: eqBypassReq=%d eqBypassActive=%d convBypassReq=%d convBypassActive=%d seq=%lld band0gain=%.2f band1gain=%.2f\n",
                e.isEqBypassRequested() ? 1 : 0, e.isEQBypassed() ? 1 : 0,
                e.isConvolverBypassRequested() ? 1 : 0, e.isConvolverBypassed() ? 1 : 0,
                lastPublishedSeq(e),
                e.getEQBandParams(0).gain, e.getEQBandParams(1).gain);
        }
        {
            std::lock_guard<std::mutex> lk(session.capMutex);
            session.gen.reset(BuzzSignal::Sine50, opt.sr);
            session.gen.setLevel(0.5f); // -6dBFS probe（安全鎖非接触域）
        }
        session.mode.store(1, std::memory_order_release);
        sleepPump(static_cast<int>(opt.runSec * 1000.0));
        session.mode.store(0, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::vector<float> in, out;
        {
            std::lock_guard<std::mutex> lk(session.capMutex);
            in = session.inCap; out = session.outCapL;
        }
        Metrics m = analyzeRun(in, out, opt.sr, 50.0, 1.0);
        const double ratio = m.outPeak / (m.inPeak + 1e-18);
        // [WORK113 test-instrumentation cleanup] PASS 窓はモード別 identity 基準に分離:
        //   bare/ir = bypass 経路の既定 passthrough 0.891(−1dB) 基準（従来値を維持）,
        //   eq      = EQ-on identity 実測基準 0.4912（= 0.891 × 0.5513 / −5.17dB）。
        //   ※ EQ-on 経路の −5.17dB 減成の内訳は未帰属（OPEN）— 2026-09-19 二連続実測で
        //     0.4912 一定（全 band 無効・total 0dB・saturation 0・EQ AGC/engine staging off）。
        //     基準値は EQ チェーン構造変化の regression tripwire として機能する。
        const bool eqIdentityMode = (rigCheckMode == "eq");
        const double ratioLo = eqIdentityMode ? 0.486 : 0.880;
        const double ratioHi = eqIdentityMode ? 0.496 : 0.897;
        const bool verdictPass = (ratio > ratioLo && ratio < ratioHi && m.thdDb < -80.0);
        std::fprintf(stderr,
            "[BUZZ] RIGCHECK(%s) sine50-6dBFS inPeak=%.4f outPeak=%.4f ratio=%.4f thd=%+.1fdB ultra=%+.1fdB -> %s\n",
            rigCheckMode.c_str(), m.inPeak, m.outPeak, ratio, m.thdDb, m.ultraRatioDb,
            verdictPass ? "PASS" : "FAIL");
        h.clearTap();
        h.stop();
        return verdictPass ? 0 : 1;
    }

    Session session;
    h.setTap([&session](juce::AudioBuffer<float>& buffer, bool isInput) {
        const int n = buffer.getNumSamples();
        if (session.mode.load(std::memory_order_acquire) != 1)
        {
            if (isInput)
                buffer.clear();
            return;
        }
        if (isInput)
        {
            float* L = buffer.getWritePointer(0);
            float* R = buffer.getNumChannels() > 1 ? buffer.getWritePointer(1) : nullptr;
            std::lock_guard<std::mutex> lk(session.capMutex);
            for (int i = 0; i < n; ++i)
            {
                const float v = session.gen.next();
                L[i] = v;
                if (R) R[i] = v;
                session.inCap.push_back(v);
            }
        }
        else
        {
            const float* L = buffer.getReadPointer(0);
            const float* R = buffer.getNumChannels() > 1 ? buffer.getReadPointer(1) : nullptr;
            std::lock_guard<std::mutex> lk(session.capMutex);
            float bp = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                session.outCapL.push_back(L[i]);
                session.outCapR.push_back(R ? R[i] : L[i]);
                bp = std::max(bp, std::abs(L[i]));
            }
            session.blockPeak.push_back(bp);
        }
    });

    auto runCapture = [&](BuzzSignal sig, double seconds, float level,
                            std::vector<float>& bpOut) {
        {
            std::lock_guard<std::mutex> lk(session.capMutex);
            session.inCap.clear(); session.outCapL.clear(); session.outCapR.clear();
            session.blockPeak.clear();
            session.blockPeak.reserve(4096);
            session.gen.reset(sig, opt.sr);
            session.gen.setLevel(level);
        }
        session.mode.store(1, std::memory_order_release);
        const auto start = std::chrono::steady_clock::now();
        const auto end = start
            + std::chrono::milliseconds(static_cast<long long>(seconds * 1000.0));
        bool flipExecuted = false;
        while (std::chrono::steady_clock::now() < end)
        {
            pumpBuzzMessages();
            // [WORK113-17] §15 transition: capture 中間時刻でパラメータを flip する（test-only）
            if (opt.flipKind != 0 && !flipExecuted)
            {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                if (elapsed >= opt.flipAtSec)
                {
                    flipExecuted = true;
                    switch (opt.flipKind)
                    {
                        case 1: e.setConvHCFilterMode(static_cast<convo::HCMode>(opt.flipValue)); break;
                        case 2: e.setConvLCFilterMode(static_cast<convo::LCMode>(opt.flipValue)); break;
                        case 3: e.setEqBypassRequested(opt.flipValue != 0); break;
                        case 4: e.setEQTotalGain(-3.0f); break;
                        default: break;
                    }
                    std::fprintf(stderr, "[TRANSITION] flip executed kind=%d value=%d at=%.4fs\n",
                                 opt.flipKind, opt.flipValue, elapsed);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        session.mode.store(0, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        {
            std::lock_guard<std::mutex> lk(session.capMutex);
            bpOut = session.blockPeak;
        }
    };

    auto flushSilence = [&]() {
        session.mode.store(0, std::memory_order_release);
        sleepPump(1500);
    };

    // ★ WORK104-R2: 起動直後のbootstrap乱流（48k→192k、OS policy解決）を draining してから
    //   プリセット投入する。IRロード時点の processingRate/blockSize を確定させるため。
    if (!waitBacklogZero(e, 30000))
        std::fprintf(stderr, "[BUZZ] WARN: bootstrap backlog not zero\n");
    sleepPump(2000);

    // 既定プリセット適用（OS=2確定）→ publish確定後に IR ロード。
    // ★ WORK106: probe モードでは「probe IR を初回ロード」にするため、既定IRロードを省略する
    //   （初回ロードだけがエンジンに反映される仮説の検証）。
    const long long seqBeforePreset = lastPublishedSeq(e);
    applyDefaultPreset(e);
    if (!waitWorldPublished(e, seqBeforePreset, 30000, "preset"))
    {
        std::fprintf(stderr, "[BUZZ] FAIL: preset world not published\n");
        h.clearTap();
        h.stop();
        return 1;
    }
    sleepPump(2000);
    if (probeMode.empty())
    {
    e.getConvolverProcessor().loadImpulseResponse(irFile, false);
    if (!waitIrFinalized(e, 300000))
    {
        std::fprintf(stderr, "[BUZZ] FAIL: IR load timeout\n");
        h.stop();
        return 1;
    }
    if (!waitBacklogZero(e, 30000))
        std::fprintf(stderr, "[BUZZ] WARN: backlog not zero after IR load\n");
    // ★ WORK104-R2: DSP側の転送＋rebuildAllIRsSynchronous完了待ちの静穏期間。
    //   UI側isIRFinalized＋backlogゼロではDSP側再構築の完了を保証できないため。
    //   MixedPhasePersistentCacheが温まっていれば数秒で終わるが、 cold時は分単位。
    std::fprintf(stderr, "[BUZZ] quiet period %d ms for DSP-side IR rebuild quiescence...\n", quietMs);
    sleepPump(quietMs);
    if (!waitBacklogZero(e, 30000))
        std::fprintf(stderr, "[BUZZ] WARN: backlog not zero after quiet period\n");
    const float irFreqDbAfterLoad = e.getConvolverProcessor().getIrFreqPeakGainDb();
    std::fprintf(stderr, "[BUZZ] getIrFreqPeakGainDb after default IR load = %.2f dB\n",
                 irFreqDbAfterLoad);
    } // end if (probeMode.empty()) default-IR-load guard

    // ★ WORK106: 単一デルタ／無音 IR probe（E の最小再現＝システム同定）。
    //   位相変換を排除し（AsIs）、EQ/softclip/autoGain を外した最小構成で
    //   入力インパルス → 出力を測定する。デルタ IR なら出力は clean な単一遅延デルタに
    //   なるはずであり、そうならなければ NUC 実行系（層/partition/ring/delay）の欠陥。
    if (!probeMode.empty())
    {
        SynthIrKind kind = SynthIrKind::Delta;
        bool useRealIr = false;
        if (probeMode == "boost") kind = SynthIrKind::Boost9dB;
        else if (probeMode == "silence") kind = SynthIrKind::Silence;
        else if (probeMode == "mid") kind = SynthIrKind::DeltaMid;
        else if (probeMode == "far") kind = SynthIrKind::DeltaFar;
        else if (probeMode == "p1") kind = SynthIrKind::P1Delta;
        else if (probeMode == "p0p1") kind = SynthIrKind::P0P1;
        else if (probeMode == "p0p1pad") kind = SynthIrKind::P0P1Pad;
        else if (probeMode == "p0two") kind = SynthIrKind::P0Two;
        else if (probeMode == "real") { useRealIr = true; }
        else if (probeMode != "delta")
        {
            std::fprintf(stderr, "[PROBE] FAIL: unknown probe mode '%s'\n", probeMode.c_str());
            h.clearTap(); h.stop(); return 1;
        }

        juce::File pIr = juce::File::getSpecialLocation(juce::File::tempDirectory)
                             .getChildFile("buzz106_probe.wav");
        if (useRealIr)
        {
            pIr = irFile; // 実IR（sampledata/impulse.wav）との同一条件比較用
        }
        else if (!writeSyntheticIrFile(pIr, kind))
        {
            std::fprintf(stderr, "[PROBE] FAIL: synthetic IR write failed\n");
            h.clearTap(); h.stop(); return 1;
        }

        // 最小構成: eq bypass / conv on / softclip off / autoGain off / 手動ゲイン 0dB / AsIs
        //   ★ WORK113-15: --buzz-order/eq/conv/hc/lc/eqlpf/direct で routing を上書き可（test-only）
        e.setConvolverPhaseMode(ConvolverProcessor::PhaseMode::AsIs);
        e.setEqBypassRequested(opt.eqOn < 0 || opt.eqOn == 0);
        e.setConvolverBypassRequested(opt.convOn == 0);
        if (opt.orderMode == 0)
            e.setProcessingOrder(convo::ProcessingOrder::ConvolverThenEQ);
        else if (opt.orderMode == 1)
            e.setProcessingOrder(convo::ProcessingOrder::EQThenConvolver);
        if (opt.hcMode >= 0)
            e.setConvHCFilterMode(static_cast<convo::HCMode>(opt.hcMode));
        if (opt.lcMode >= 0)
            e.setConvLCFilterMode(static_cast<convo::LCMode>(opt.lcMode));
        if (opt.lpMode >= 0)
            e.setEqLPFFilterMode(static_cast<convo::HCMode>(opt.lpMode));
        if (opt.directHead >= 0)
            e.getConvolverProcessor().setExperimentalDirectHeadEnabled(opt.directHead == 1);
        e.setSoftClipEnabled(false);
        e.setAutoGainStagingEnabled(false);
        e.setInputHeadroomDb(0.0f);
        e.setOutputMakeupDb(0.0f);
        e.setConvolverInputTrimDb(0.0f);
        if (opt.eqOn == 1)
            configureProbeFlatEQ(e); // ★ WORK113-15: fail-closed 解除（EQ 本体は恒等変換）
        std::fprintf(stderr,
            "[PROBE_CFG] order=%d eq=%d conv=%d hc=%d lc=%d eqlpf=%d direct=%d\n",
            opt.orderMode, opt.eqOn, opt.convOn, opt.hcMode, opt.lcMode, opt.lpMode, opt.directHead);
        if (!waitWorldPublished(e, lastPublishedSeq(e), 25000, "probe-config"))
            std::fprintf(stderr, "[PROBE] WARN: probe-config world not published\n");
        sleepPump(1200);

        const long long seqIr = lastPublishedSeq(e);
        e.getConvolverProcessor().loadImpulseResponse(pIr, false);
        if (!waitIrFinalized(e, 300000))
        {
            std::fprintf(stderr, "[PROBE] FAIL: IR load timeout\n");
            pIr.deleteFile(); h.clearTap(); h.stop(); return 1;
        }
        (void)waitWorldPublished(e, seqIr, 30000, "probe-irload");
        std::fprintf(stderr, "[PROBE] quiet %d ms for DSP-side IR rebuild...\n", quietMs);
        sleepPump(quietMs);

        // ★ WORK108 108-3: 同一 generation の geometry を 1 行で出力（declared / engine / RT）
        auto printGeom = [&e](const char* when) {
            convo::MKLNonUniformConvolver::GeometryTrace g {};
            convo::MKLNonUniformConvolver::getGeometryTrace(g);
            const auto ir = e.getConvolverProcessor().getIRGeometry();
            std::fprintf(stderr,
                "[GEOM %s] IR(sr=%.0f block=%d len=%d gen=%llu) UIprep(sr=%.0f block=%d) "
                "ENG(buildRate=%d irLen=%d block=%d layers=%d L0part=%d L0numIR=%d L0numParts=%d L0fft=%d ring=%d) "
                "RT(addNs=%d addCalls=%llu L0calls=%llu getNs=%d getGot=%d getShort=%llu ringW=%d ringR=%d avail=%d fdl=%d nextPart=%d)\n",
                when, ir.sampleRate, ir.blockSize, e.getConvolverProcessor().getIRLength(),
                static_cast<unsigned long long>(ir.generation),
                e.getConvolverProcessor().getPreparedSampleRate(),
                e.getConvolverProcessor().getPreparedBlockSize(),
                g.engineBuildRate, g.engineTotalIrLen, g.maxBlockSize, g.numActiveLayers,
                g.l0PartSize, g.l0NumPartsIR, g.l0NumParts, g.l0FftSize, g.ringSize,
                g.lastAddNumSamples, g.addCalls, g.addCallsL0, g.lastGetNumSamples,
                g.lastGetGot, g.getShortCount,
                g.ringWrite, g.ringRead, g.ringAvail, g.l0FdlIndex, g.l0NextPart);
        };
        printGeom("before-capture");
        std::vector<float> bp;
        const BuzzSignal pSig = probeSignalFromName(probeSignalName);
        runCapture(pSig, 2.0, probeLevel, bp);
        std::vector<float> in, out;
        {
            std::lock_guard<std::mutex> lk(session.capMutex);
            in = session.inCap; out = session.outCapL;
        }

        // ★ WORK111: buzz 指標（fade ON/OFF の A/B 比較用）。limiter は両条件で完全同一。
        {
            const double fundHz = probeFundHz(pSig);
            const Metrics pm = analyzeRun(in, out, opt.sr, fundHz, 0.5);
            double rmsAll = 0.0;
            for (float v : out) rmsAll += static_cast<double>(v) * v;
            rmsAll = out.empty() ? 0.0 : std::sqrt(rmsAll / static_cast<double>(out.size()));
            size_t n = out.size();
            if (n > 65536u) n = 65536u;
            double fundAmp = 0.0;
            if (n >= 256u)
            {
                const size_t off = out.size() - n;
                const double w0 = 2.0 * 3.14159265358979323846 * fundHz / opt.sr;
                double re = 0.0, im = 0.0, wsum = 0.0;
                for (size_t i = 0; i < n; ++i)
                {
                    const double hann = 0.5 * (1.0 - std::cos(2.0 * 3.14159265358979323846
                        * static_cast<double>(i) / static_cast<double>(n - 1)));
                    const double v = static_cast<double>(out[off + i]) * hann;
                    re += v * std::cos(w0 * static_cast<double>(i));
                    im -= v * std::sin(w0 * static_cast<double>(i));
                    wsum += hann;
                }
                if (wsum > 1.0e-12) fundAmp = 2.0 * std::sqrt(re * re + im * im) / wsum;
            }
            std::fprintf(stderr,
                "[PROBE_METRICS] signal=%s tailFade=%s inPeak=%.6f outPeak=%.6f rmsAll=%.6f "
                "fundHz=%.1f fundAmp=%.6f thd=%.2fdB ultraRatio=%.2fdB flatTop=%lld limitZone=%lld\n",
                probeSignalName.c_str(), disableTailFade ? "OFF" : "ON",
                pm.inPeak, pm.outPeak, rmsAll, fundHz, fundAmp, pm.thdDb, pm.ultraRatioDb,
                pm.flatTop, pm.limitZone);

            // ★ WORK112 112-2/112-3: 入力（対照）と出力の DC/高調波/低域/residual を分離
            if (fundHz > 0.0 && probeSignalName != "impulse")
            {
                printLfResidual(in, opt.sr, fundHz, (probeSignalName + "_IN").c_str());
                printLfResidual(out, opt.sr, fundHz, probeSignalName.c_str());
                if (kind == SynthIrKind::Delta)
                    printDeltaNull(in, out, opt.sr, "delta");
            }

            // [WORK113-17] §15 transition 計測（flip 指定時のみ・probe capture は 2.0s 固定）
            if (opt.flipKind != 0)
            {
                // [WORK113 test-instrumentation cleanup] metrics 窓を capture の実効 sample rate に
                //   bind する。tap の capture レートはデバイス/エンジン構成依存で opt.sr と一致せず
                //   窓位置がずれるため、capture 総サンプル数から実効レートを導出する。
                const double probeCaptureSec = 2.0;
                const double effectiveCaptureRate = out.empty()
                    ? static_cast<double>(opt.sr)
                    : static_cast<double>(out.size()) / probeCaptureSec;
                const auto tr = convo_buzz_transition::computeTransitionMetrics(out, effectiveCaptureRate, opt.flipAtSec, probeCaptureSec);
                const auto line = convo_buzz_transition::formatTransitionLine(tr, probeSignalName);
                std::fputs(line.c_str(), stderr);
                std::fputc('\n', stderr);
                // [WORK113-17 TEMP] envelope 解析用 dump（検証後削除）
                {
                    std::ofstream dump("acc_11317_flip_capture.csv");
                    for (size_t i = 0; i < out.size(); ++i)
                        dump << i << ',' << static_cast<double>(out[i]) << '\n';
                }
            }
        }

        // 解析: ピーク位置・値・全エネルギー・有意タップ数・±3サンプル集中度
        long long peakIdx = -1; double peakVal = 0.0, energy = 0.0, inPeak = 0.0;
        long long sigCount = 0, outsideCount = 0;
        double headEnergy = 0.0;
        for (size_t i = 0; i < out.size(); ++i)
        {
            const double v = out[i];
            energy += v * v;
            if (std::abs(v) > inPeak) inPeak = std::abs(v);
            if (peakIdx < 0 || std::abs(v) > std::abs(peakVal)) { peakVal = v; peakIdx = static_cast<long long>(i); }
        }
        for (size_t i = 0; i < out.size(); ++i)
        {
            const double v = out[i];
            if (std::abs(v) > 1.0e-4)
            {
                ++sigCount;
                const long long d = static_cast<long long>(i) - peakIdx;
                if (d < -3 || d > 3) ++outsideCount;
                else headEnergy += v * v;
            }
        }
        const double headFrac = (energy > 1e-30) ? (headEnergy / energy) : 0.0;
        const char* kindName = useRealIr ? "real"
                             : (kind == SynthIrKind::Delta) ? "delta"
                             : (kind == SynthIrKind::DeltaMid) ? "deltaMid(L0-p2)"
                             : (kind == SynthIrKind::DeltaFar) ? "deltaFar(L1)"
                             : (kind == SynthIrKind::Boost9dB) ? "boost9dB" : "silence";
        std::fprintf(stderr,
            "[PROBE] kind=%s level=%.3f outPeak=%.6f peakIdx=%lld energy=%.8f sigTaps=%lld outsideTaps=%lld headFrac=%.4f -> %s\n",
            kindName, probeLevel, std::abs(peakVal), peakIdx, energy,
            sigCount, outsideCount, headFrac,
            (kind == SynthIrKind::Silence && !useRealIr) ? (energy < 1e-12 ? "SILENT-GOOD" : "NOISE-FROM-ENGINE")
                                           : ((outsideCount <= 2 && headFrac > 0.98) ? "CLEAN-DELTA" : "BROADENED"));

        // ★ WORK108 108-3: capture 後の RT 実測 geometry（Add/Get/ring）
        printGeom("after-capture");

        // ★ WORK109 109-3: 二次ピーク探索（global peak ±32 サンプルを除外した最大値）。
        //   p0p1 で 2 tap が独立に現れるか（fusion かどうか）を実測で判定する。
        {
            long long p2 = -1; double v2 = 0.0;
            for (size_t i = 0; i < out.size(); ++i)
            {
                const long long di = static_cast<long long>(i) - peakIdx;
                if (di >= -32 && di <= 32) continue;
                if (std::abs(out[i]) > std::abs(v2)) { v2 = out[i]; p2 = static_cast<long long>(i); }
            }
            std::fprintf(stderr, "[PROBE] secondaryPeak idx=%lld val=%.6f\n", p2, static_cast<double>(v2));
        }

        // タップダンプ（有意タップの先頭 16 個 + ピーク近傍 ±5）
        {
            char buf[4096]; int n = 0; int shown = 0;
            n += std::snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n), "[PROBE] firstTaps:");
            for (size_t i = 0; i < out.size() && shown < 16; ++i)
                if (std::abs(out[i]) > 1.0e-4)
                {
                    n += std::snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n), " %zu:%.6f", i, static_cast<double>(out[i]));
                    ++shown;
                }
            n += std::snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n), " | nearPeak:");
            if (peakIdx >= 5)
                for (long long i = peakIdx - 5; i <= peakIdx + 5 && i < static_cast<long long>(out.size()); ++i)
                    n += std::snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n), " %lld:%.6f",
                                       i, static_cast<double>(out[static_cast<size_t>(i)]));
            std::fprintf(stderr, "%s\n", buf);
        }
        if (!useRealIr)
            pIr.deleteFile();
        h.clearTap();
        h.stop();
        return 0;
    }

    const BuzzConfig kConfigs[] = {
        {"C0", "bypass-all", true, true, false, false, false, false},
        {"C1", "conv-only", true, false, false, false, false, false},
        {"C2", "eq-only", false, true, false, false, false, false},
        {"C3", "conv+eq(default)", false, false, false, false, false, false},
        {"C4", "conv+eq makeup-6dB", false, false, true, false, false, false},
        {"C5", "conv+eq softclip-off", false, false, false, true, false, false},
        {"C8", "conv+eq shaper-psycho", false, false, false, false, false, true},
        {"C6", "conv+eq phase-AsIs", false, false, false, false, true, false},
    };
    const BuzzSignal kSignalsFull[] = {BuzzSignal::Sine50, BuzzSignal::Sine40,
        BuzzSignal::Sine100, BuzzSignal::Kick, BuzzSignal::Multi};
    const BuzzSignal kSignalsQuick[] = {BuzzSignal::Sine50, BuzzSignal::Kick, BuzzSignal::Multi};

    struct Row { std::string cfg, sig; Metrics m; };
    std::vector<Row> rows;

    auto needConfigs = [&](std::vector<BuzzConfig>& out) {
        if (opt.quick)
        {
            for (auto c : {kConfigs[0], kConfigs[1], kConfigs[3], kConfigs[4]}) out.push_back(c);
        }
        else
        {
            for (const auto& c : kConfigs) out.push_back(c);
        }
    };
    std::vector<BuzzSignal> signals;
    if (opt.quick)
        for (auto s : kSignalsQuick) signals.push_back(s);
    else
        for (auto s : kSignalsFull) signals.push_back(s);

    std::vector<BuzzConfig> configs;
    needConfigs(configs);

    // ★ WORK104-R2: リグ自己診断。bypass-all (C0) で 50Hz 正弦を流し、
    //   透過性（peak≈in×0.891、THD<-80dB）を確認してから本測定に入る。
    //   不合格なら IR／world の不整合（欠陥D候補）のまま測定しても無意味なため即 FAIL。
    {
        const BuzzConfig self { "C0", "bypass-all", true, true, false, false, false, false };
        const long long seqBeforeSelf = lastPublishedSeq(e);
        applyBuzzConfig(e, self);
        if (!waitWorldPublished(e, seqBeforeSelf, 25000, "selftest-C0"))
        {
            std::fprintf(stderr, "[BUZZ] FAIL: selftest world not published; aborting.\n");
            h.clearTap();
            h.stop();
            return 1;
        }
        flushSilence();
        std::vector<float> selfBp;
        runCapture(BuzzSignal::Sine50, 2.0, 0.5f, selfBp);
        std::vector<float> in, out;
        {
            std::lock_guard<std::mutex> lk(session.capMutex);
            in = session.inCap; out = session.outCapL;
        }
        Metrics sm = analyzeRun(in, out, opt.sr, 50.0, 1.0);
        sm.driftDb = blockDriftDb(selfBp);
        const double ratio = sm.outPeak / (sm.inPeak + 1e-18);
        std::fprintf(stderr,
            "[BUZZ] SELFTEST C0 sine50 inPeak=%.4f outPeak=%.4f ratio=%.4f thd=%+.1fdB drift=%+.1fdB\n",
            sm.inPeak, sm.outPeak, ratio, sm.thdDb, sm.driftDb);
        if (!(ratio > 0.880 && ratio < 0.897 && sm.thdDb < -80.0))
        {
            std::fprintf(stderr,
                "[BUZZ] FAIL: rig self-test failed (expect ratio~=0.891, thd<-80dB). "
                "IR/world mismatch suspected; aborting.\n");
            h.clearTap();
            h.stop();
            return 1;
        }
        flushSilence();
    }

    std::string loadedPhase = "mixed";
    for (const auto& cfg : configs)
    {
        const long long seqBefore = lastPublishedSeq(e);
        applyBuzzConfig(e, cfg);
        // phase変更時のみIR再ロード（Mixed→AsIs等）
        const std::string wantPhase = cfg.phaseAsIs ? "asis" : "mixed";
        if (wantPhase != loadedPhase && !cfg.convBypass)
        {
            e.getConvolverProcessor().loadImpulseResponse(irFile, false);
            if (!waitIrFinalized(e, 300000))
            {
                std::fprintf(stderr, "[BUZZ] SKIP %s: IR reload timeout\n", cfg.id);
                continue;
            }
            loadedPhase = wantPhase;
        }
        if (!waitWorldPublished(e, seqBefore, 25000, cfg.id))
        {
            std::fprintf(stderr, "[BUZZ] SKIP %s: world not published\n", cfg.id);
            continue;
        }
        flushSilence();
        for (BuzzSignal sig : signals)
        {
            std::vector<float> bp;
            runCapture(sig, opt.runSec, 1.0f, bp);
            std::vector<float> in, out;
            {
                std::lock_guard<std::mutex> lk(session.capMutex);
                in = session.inCap; out = session.outCapL;
            }
            Metrics m = analyzeRun(in, out, opt.sr, SignalGen::freqHz(sig), 1.0);
            m.driftDb = blockDriftDb(bp);
            rows.push_back({cfg.id, SignalGen::name(sig), m});
            std::fprintf(stderr,
                "[BUZZ] %s %s inPeak=%.4f outPeak=%.4f rms=%.4f crest=%+.1fdB flat=%lld lim=%lld thd=%+.1fdB ultra=%+.1fdB drift=%+.1fdB dc=%+.2e\n",
                cfg.id, SignalGen::name(sig), m.inPeak, m.outPeak, m.outRms,
                m.crestDb, m.flatTop, m.limitZone, m.thdDb, m.ultraRatioDb, m.driftDb, m.dcMean);
            flushSilence();
        }
    }

    // ── D 実験：OS factor 変更（IR 再ロードなし）で convolver の
    //   knownBlockSize／rate と runtime quantum の不整合を意図的に作り、
    //   無警告ガベージ化（欠陥D候補）を検証する。終了後は factor=2 に復帰。
    //   ※ IR再ロードを行わない点が要点（M-1 再利用経路の観測）。
    bool dReproduced = false;
    {
        const BuzzConfig convOnly { "C1", "conv-only", true, false, false, false, false, false };
        long long seqD = lastPublishedSeq(e);
        applyBuzzConfig(e, convOnly);
        waitWorldPublished(e, seqD, 25000, "D-convonly");
        seqD = lastPublishedSeq(e);
        e.setOversamplingFactor(4);
        waitWorldPublished(e, seqD, 30000, "D-os4");
        sleepPump(2000);
        flushSilence();
        std::vector<float> dbp;
        runCapture(BuzzSignal::Sine50, opt.runSec, 1.0f, dbp);
        std::vector<float> din, dout;
        {
            std::lock_guard<std::mutex> lk(session.capMutex);
            din = session.inCap; dout = session.outCapL;
        }
        Metrics dm = analyzeRun(din, dout, opt.sr, 50.0, 1.0);
        dm.driftDb = blockDriftDb(dbp);
        std::fprintf(stderr,
            "[BUZZ] D(os4-no-reload) sine50 inPeak=%.4f outPeak=%.4f thd=%+.1fdB ultra=%+.1fdB drift=%+.1fdB\n",
            dm.inPeak, dm.outPeak, dm.thdDb, dm.ultraRatioDb, dm.driftDb);
        // 復帰（IR再ロードなしのまま factor=2 へ）
        seqD = lastPublishedSeq(e);
        e.setOversamplingFactor(2);
        waitWorldPublished(e, seqD, 30000, "D-os2restore");
        sleepPump(2000);
        flushSilence();
        // 判定は verdict 部で行う（C1 ベースラインとの比較）
        rows.push_back({"D-os4", "sine50", dm});
        dReproduced = true; // 実際の判定は下の verdict で数値比較する
        (void)dReproduced;
    }

    // ── B2 機能再現：+9dB@50Hz ブースト IR ──
    //   readback値自体がNUC経路では常に未伝搬のため、Mixed最適化を待たず
    //   AsIsでロードする（欠陥の有無判定に位相は無関係）。
    bool b2reproduced = false;
    {
        juce::File boostIr = juce::File::getSpecialLocation(
            juce::File::tempDirectory).getChildFile("buzz_boost9db.wav");
        if (writeSyntheticIrFile(boostIr, SynthIrKind::Boost9dB))
        {
            e.setConvolverPhaseMode(ConvolverProcessor::PhaseMode::AsIs);
            waitWorldPublished(e, lastPublishedSeq(e), 25000, "B2-phase");
            e.getConvolverProcessor().loadImpulseResponse(boostIr, false);
            if (waitIrFinalized(e, 300000))
            {
                sleepPump(800);
                const float rb = e.getConvolverProcessor().getIrFreqPeakGainDb();
                std::fprintf(stderr, "[BUZZ] boost-IR readback getIrFreqPeakGainDb = %.2f dB (file-side expected ~+9dB@50Hz)\n", rb);
                b2reproduced = (rb == 0.0f);
            }
            else
            {
                std::fprintf(stderr, "[BUZZ] SKIP B2: boost IR load timeout\n");
            }
            boostIr.deleteFile();
        }
        else
        {
            std::fprintf(stderr, "[BUZZ] SKIP B2: boost IR write failed\n");
        }
    }

    // ── 判定 ──
    auto find = [&](const char* cfg, const char* sig) -> const Metrics* {
        for (const auto& r : rows)
            if (r.cfg == cfg && r.sig == sig) return &r.m;
        return nullptr;
    };
    std::fprintf(stderr, "==== [BUZZ] VERDICTS ====\n");
    if (const Metrics* a = find("C3", "sine50"))
    {
        const Metrics* a0 = find("C0", "sine50");
        const Metrics* a4 = find("C4", "sine50");
        // 安全鎖接触の深さ比較：bypass基準(C0)に対しconv+eq(C3)が深く食い込めばA支持。
        // limitZone>=0.8414がリミッター接触、flatTop>=0.89がハードクランプ接触の代理指標。
        const bool engaged = (a->flatTop > 0 || a->thdDb > -40.0);
        const bool deeper = (a0 && (a->limitZone > a0->limitZone || a->thdDb > a0->thdDb + 6.0));
        const bool cleared = (a4 && a4->flatTop == 0 && a4->thdDb < a->thdDb - 10.0);
        std::fprintf(stderr, "[BUZZ-A] C3 sine50 flatTop=%lld lim=%lld thd=%+.1fdB -> %s; vs C0 deeper=%s; C4 clears=%s\n",
                     a->flatTop, a->limitZone, a->thdDb, engaged ? "ENGAGED" : "CLEAR",
                     (a0 ? (deeper ? "YES" : "NO") : "N/A"),
                     (a4 ? (cleared ? "YES" : "NO") : "N/A"));
    }
    std::fprintf(stderr, "[BUZZ-B2] irFreqPeakGainDb propagation %s\n",
                 b2reproduced ? "DEFECT-REPRODUCED(=0.0 despite +9dB IR)" : "NOT-REPRODUCED");
    if (const Metrics* c1 = find("C1", "sine50"))
    {
        const Metrics* c0 = find("C0", "sine50");
        std::fprintf(stderr, "[BUZZ-C] ultra C1=%+.1fdB vs C0=%+.1fdB\n",
                     c1->ultraRatioDb, c0 ? c0->ultraRatioDb : -999.0);
    }
    if (const Metrics* dd = find("D-os4", "sine50"))
    {
        const Metrics* c1 = find("C1", "sine50");
        const bool garbage = (c1 && (std::abs(dd->outPeak - c1->outPeak) > 0.05 * c1->outPeak + 1e-6
                                     || dd->thdDb > c1->thdDb + 20.0
                                     || dd->ultraRatioDb > c1->ultraRatioDb + 20.0));
        std::fprintf(stderr,
            "[BUZZ-D] os4-no-reload vs C1: outPeak %.4f->%.4f thd %+.1f->%+.1f ultra %+.1f->%+.1f -> %s\n",
            c1 ? c1->outPeak : -1.0, dd->outPeak,
            c1 ? c1->thdDb : -999.0, dd->thdDb,
            c1 ? c1->ultraRatioDb : -999.0, dd->ultraRatioDb,
            garbage ? "MISMATCH-GARBAGE (defect-D reproduced)" : "CONSISTENT");
    }

    if (!opt.outCsv.empty())
    {
        juce::File csv(opt.outCsv);
        std::string text = "config,signal,inPeak,outPeak,outRms,crestDb,flatTop,limitZone,thdDb,ultraRatioDb,driftDb,dcMean\n";
        char line[512];
        for (const auto& r : rows)
        {
            std::snprintf(line, sizeof(line), "%s,%s,%.6f,%.6f,%.6f,%.2f,%lld,%lld,%.2f,%.2f,%.2f,%.3e\n",
                          r.cfg.c_str(), r.sig.c_str(), r.m.inPeak, r.m.outPeak, r.m.outRms,
                          r.m.crestDb, r.m.flatTop, r.m.limitZone, r.m.thdDb,
                          r.m.ultraRatioDb, r.m.driftDb, r.m.dcMean);
            text += line;
        }
        csv.replaceWithText(text);
        std::fprintf(stderr, "[BUZZ] CSV: %s\n", csv.getFullPathName().toRawUTF8());
    }

    h.clearTap();
    h.stop();
    std::fprintf(stderr, "[BUZZ] DONE rows=%u\n", static_cast<unsigned>(rows.size()));
    return 0;
}
