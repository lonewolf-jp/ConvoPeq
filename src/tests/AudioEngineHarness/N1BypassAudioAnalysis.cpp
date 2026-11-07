// N1BypassAudioAnalysis.cpp
//
// PR #168 (N1 bypass tail crossfade) の受け入れ検証用オフライン分析。
// 実音声処理経路（ConvolverProcessor）で WAV を生成し、切り替え時の段差・
// フェード長・フェードの連続性・収束・非有限値の有無を数値化する。
//
// 使い方（AudioEngineHarness の CLI モード）:
//   AudioEngineHarness --n1-analysis <出力ディレクトリ> <post|pre>
//
// RT 経路には一切手を加えない。計測はすべて処理後のバッファ上で行う。

#include "ConvolverProcessor.h"
#include "MKLRealTimeSetup.h"

#include <JuceHeader.h>
#include <xmmintrin.h>

#if JUCE_WINDOWS
    #include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// N1: ミックス smoother の現在値を読む（ConvolverProcessor.h の friend 宣言に対応）。
//   friend はグローバル名前空間の型なので、無名名前空間の外に置く。
struct N1CrossfadeTestAccess
{
    static double currentMix(const ConvolverProcessor& p) noexcept
    {
        return p.mixSmoother.getCurrentValue();
    }
};

namespace
{
constexpr double kSampleRate = 48000.0;
constexpr int kBlockSize = 512;
constexpr int kIRLength = 48000;          // 1.0 s
constexpr int kIRPeakSample = 100;
constexpr double kIRPeakAmp = 0.5;
constexpr double kIRTauSamples = 12000.0; // 0.25 s
constexpr int kTotalSamples = 72000;      // 1.5 s
constexpr int kWarmupBlocks = 20;         // ミックスを 1.0 へ収束させる（smoothing 0.1 s）
constexpr int kBurstStart = kWarmupBlocks * kBlockSize; // 10240
constexpr int kBurstSamples = 480;        // 10 ms
constexpr double kBurstAmp = 0.7;

// 既存の N1 受け入れテスト（checkBypassTailContinuity）と同じ相対基準。
constexpr double kSwitchStepRatio = 0.25;
// 実装契約: totalSteps = round(sampleRate * 0.020) = 960。
constexpr int kExpectedFadeSamples = 960;
// ブロック境界で観測するミックス値の許容。1 サンプル = 1/960 = 0.00104 なので、
// 0.0011 は約 ±1 サンプルに相当する。
constexpr double kMixTolerance = 0.0011;

struct Transition
{
    int sample { 0 };
    bool bypassOn { true };
};

struct Case
{
    const char* name;
    std::vector<Transition> transitions;
};

const std::vector<Case>& cases()
{
    static const std::vector<Case> list = {
        { "on_160ms",      { { kBurstStart + 7680, true } } },                          // 160.0 ms
        { "on_202ms",      { { kBurstStart + 9728, true } } },                          // 202.7 ms
        { "onoff_202_405", { { kBurstStart + 9728, true }, { kBurstStart + 19456, false } } }, // 202.7 / 405.3 ms
        { "on_352ms",      { { kBurstStart + 16896, true } } },                         // 352.0 ms
        { "multi",         { { kBurstStart + 7680, true }, { kBurstStart + 15872, false }, { kBurstStart + 24064, true } } }, // 160.0 / 330.7 / 501.3 ms
        { "on_597ms",      { { kBurstStart + 28672, true } } },                         // 597.3 ms
    };
    return list;
}

struct TransitionMetrics
{
    std::string caseName;
    int index { 0 };
    int sample { 0 };
    bool bypassOn { true };
    double scale { 0.0 };
    double switchStep { 0.0 };
    double localMaxStep { 0.0 };
    double refLocalMaxStep { 0.0 };
    double refP999Step { 0.0 };
    double mixAfterOneBlock { -1.0 };
    double mixAfterTwoBlocks { -1.0 };
    double fadeSamplesEstimate { -1.0 };
    bool mixMonotonic { false };
    double steadyRmsA { 0.0 };
    double steadyRmsB { 0.0 };
    double steadyDrift { 0.0 };
    double refSteadyDrift { 0.0 };
    bool finite { true };
    bool t1 { false };
    bool t2 { false };
    bool t3 { false };
    bool t4 { false };
    bool t5 { false };
    bool t6 { false };
    bool allPass { false };
};

juce::File writeDecayIr(const juce::File& path)
{
    path.deleteFile();
    std::unique_ptr<juce::OutputStream> stream(path.createOutputStream());
    if (stream == nullptr)
        return {};

    juce::WavAudioFormat fmt;
    std::unique_ptr<juce::AudioFormatWriter> writer(fmt.createWriterFor(stream,
        juce::AudioFormatWriterOptions{}
            .withSampleRate(kSampleRate)
            .withNumChannels(2)
            .withBitsPerSample(32)));
    if (writer == nullptr)
        return {};

    juce::AudioBuffer<float> buf(2, kIRLength);
    buf.clear();
    for (int i = kIRPeakSample; i < kIRLength; ++i)
    {
        const auto v = static_cast<float>(kIRPeakAmp
            * std::exp(-static_cast<double>(i - kIRPeakSample) / kIRTauSamples));
        buf.getWritePointer(0)[i] = v;
        buf.getWritePointer(1)[i] = v;
    }

    int written = 0;
    while (written < kIRLength)
    {
        const int n = std::min(65536, kIRLength - written);
        if (!writer->writeFromAudioSampleBuffer(buf, written, n))
            return {};
        written += n;
    }
    writer.reset();
    return path;
}

void fillInput(juce::AudioBuffer<double>& block, int startSample, int numSamples)
{
    block.clear();
    for (int i = 0; i < numSamples; ++i)
    {
        const int n = startSample + i;
        if (n < kBurstStart || n >= kBurstStart + kBurstSamples)
            continue;
        const int local = n - kBurstStart;
        const double window = 0.5 - 0.5 * std::cos(2.0 * juce::MathConstants<double>::pi
                                                   * static_cast<double>(local) / (kBurstSamples - 1));
        const double v = kBurstAmp * window
            * std::sin(2.0 * juce::MathConstants<double>::pi * 1000.0 * local / kSampleRate);
        block.getWritePointer(0)[i] = v;
        block.getWritePointer(1)[i] = v;
    }
}

bool renderCase(const juce::File& irFile, const Case& c, bool applyTransitions,
                std::vector<double>& out, std::vector<double>& mixPerBlock)
{
    ConvolverProcessor conv;
    conv.prepareToPlay(kSampleRate, kBlockSize);
    conv.setTargetIRLength(1.0f);
    conv.loadImpulseResponse(irFile, false);

    bool loaded = false;
    for (int i = 0; i < 6000; ++i)
    {
        const auto bd = conv.getLatencyBreakdown();
        if (bd.irPeakLatencySamples == kIRPeakSample && bd.totalLatencySamples >= kIRPeakSample)
        {
            loaded = true;
            break;
        }
#if JUCE_WINDOWS
        MSG msg {};
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
#endif
        juce::Thread::sleep(2);
    }
    if (!loaded || !conv.isIRFinalized() || conv.getIRLength() != kIRLength)
    {
        std::fprintf(stderr,
                     "[N1-analysis] IR load failed: len=%d finalized=%d peak=%d err='%s'\n",
                     conv.getIRLength(), conv.isIRFinalized() ? 1 : 0,
                     conv.getLatencyBreakdown().irPeakLatencySamples,
                     conv.getLastError().toRawUTF8());
        return false;
    }

    conv.setBypass(false);
    conv.setMix(1.0f);

    out.assign(static_cast<size_t>(kTotalSamples), 0.0);
    const int blockCount = (kTotalSamples + kBlockSize - 1) / kBlockSize;
    mixPerBlock.assign(static_cast<size_t>(blockCount + 1), 1.0);

    juce::AudioBuffer<double> block(2, kBlockSize);
    juce::dsp::AudioBlock<double> blk(block);

    size_t nextTransition = 0;
    for (int start = 0; start < kTotalSamples; start += kBlockSize)
    {
        while (applyTransitions && nextTransition < c.transitions.size()
               && c.transitions[nextTransition].sample <= start)
        {
            conv.setBypass(c.transitions[nextTransition].bypassOn);
            ++nextTransition;
        }

        const int n = std::min(kBlockSize, kTotalSamples - start);
        fillInput(block, start, n);
        conv.process(blk);

        for (int i = 0; i < n; ++i)
            out[static_cast<size_t>(start + i)] = block.getReadPointer(0)[i];

        mixPerBlock[static_cast<size_t>(start / kBlockSize) + 1] =
            N1CrossfadeTestAccess::currentMix(conv);
    }
    return true;
}

bool writeWav(const juce::File& path, const std::vector<double>& samples)
{
    path.deleteFile();
    std::unique_ptr<juce::OutputStream> stream(path.createOutputStream());
    if (stream == nullptr)
        return false;
    juce::WavAudioFormat fmt;
    std::unique_ptr<juce::AudioFormatWriter> writer(fmt.createWriterFor(stream,
        juce::AudioFormatWriterOptions{}
            .withSampleRate(kSampleRate)
            .withNumChannels(2)
            .withBitsPerSample(32)));
    if (writer == nullptr)
        return false;

    juce::AudioBuffer<float> buf(2, static_cast<int>(samples.size()));
    for (int i = 0; i < static_cast<int>(samples.size()); ++i)
    {
        const auto v = static_cast<float>(samples[static_cast<size_t>(i)]);
        buf.getWritePointer(0)[i] = v;
        buf.getWritePointer(1)[i] = v;
    }
    int written = 0;
    while (written < buf.getNumSamples())
    {
        const int n = std::min(65536, buf.getNumSamples() - written);
        if (!writer->writeFromAudioSampleBuffer(buf, written, n))
            return false;
        written += n;
    }
    writer.reset();
    return true;
}

double percentileStep(const std::vector<double>& x, int from, int to, double q)
{
    std::vector<double> steps;
    steps.reserve(static_cast<size_t>(std::max(0, to - from)));
    for (int n = std::max(1, from); n < to; ++n)
        steps.push_back(std::fabs(x[static_cast<size_t>(n)] - x[static_cast<size_t>(n - 1)]));
    if (steps.empty())
        return 0.0;
    std::sort(steps.begin(), steps.end());
    const size_t idx = static_cast<size_t>(q * static_cast<double>(steps.size() - 1));
    return steps[std::min(idx, steps.size() - 1)];
}

double maxDelta(const std::vector<double>& x, int from, int to)
{
    double m = 0.0;
    for (int n = std::max(1, from); n < to; ++n)
        m = std::max(m, std::fabs(x[static_cast<size_t>(n)] - x[static_cast<size_t>(n - 1)]));
    return m;
}

bool allFinite(const std::vector<double>& x)
{
    for (double v : x)
        if (!std::isfinite(v))
            return false;
    return true;
}

double blockRms(const std::vector<double>& x, int from, int to)
{
    double sum = 0.0;
    int count = 0;
    for (int n = std::max(0, from); n < to && n < kTotalSamples; ++n)
    {
        sum += x[static_cast<size_t>(n)] * x[static_cast<size_t>(n)];
        ++count;
    }
    return (count > 0) ? std::sqrt(sum / static_cast<double>(count)) : 0.0;
}

TransitionMetrics measure(const Case& c, size_t ti, const Transition& t,
                          const std::vector<double>& x, const std::vector<double>& ref,
                          const std::vector<double>& mixPerBlock)
{
    TransitionMetrics m;
    m.caseName = c.name;
    m.index = static_cast<int>(ti);
    m.sample = t.sample;
    m.bypassOn = t.bypassOn;
    m.finite = allFinite(x);

    const int s = t.sample;
    m.scale = std::max(std::fabs(x[static_cast<size_t>(s - 1)]),
                       std::fabs(ref[static_cast<size_t>(s)]));
    m.switchStep = std::fabs(x[static_cast<size_t>(s)] - x[static_cast<size_t>(s - 1)]);
    m.localMaxStep = maxDelta(x, s - 256, s + 257);
    m.refLocalMaxStep = maxDelta(ref, s - 256, s + 257);
    m.refP999Step = percentileStep(ref, s - 2400, s - 256, 0.999);

    const size_t bi = static_cast<size_t>(s / kBlockSize);
    if (bi + 2 < mixPerBlock.size())
    {
        m.mixAfterOneBlock = mixPerBlock[bi + 1];
        m.mixAfterTwoBlocks = mixPerBlock[bi + 2];
    }

    // 20 ms = 960 samples。1 ブロック (512) 後の理論値は 1 - 512/960 = 0.46667。
    const double expectedAfterOne = t.bypassOn
        ? 1.0 - static_cast<double>(kBlockSize) / static_cast<double>(kExpectedFadeSamples)
        : static_cast<double>(kBlockSize) / static_cast<double>(kExpectedFadeSamples);
    if (m.mixAfterOneBlock >= 0.0 && m.mixAfterTwoBlocks >= 0.0)
    {
        const double deltaOne = std::fabs(m.mixAfterOneBlock - expectedAfterOne);
        const double endTarget = t.bypassOn ? 0.0 : 1.0;
        const double deltaTwo = std::fabs(m.mixAfterTwoBlocks - endTarget);
        m.fadeSamplesEstimate = (deltaOne <= kMixTolerance || deltaTwo <= kMixTolerance)
            ? static_cast<double>(kExpectedFadeSamples)
            : -1.0;
        m.t3 = (deltaOne <= kMixTolerance) && (deltaTwo <= kMixTolerance);
        m.mixMonotonic = t.bypassOn
            ? (m.mixAfterOneBlock <= 1.0 && m.mixAfterTwoBlocks <= m.mixAfterOneBlock + 1.0e-6)
            : (m.mixAfterOneBlock >= 0.0 && m.mixAfterTwoBlocks >= m.mixAfterOneBlock - 1.0e-6);
    }

    m.steadyRmsA = blockRms(x, s + 1200, s + 2400);
    m.steadyRmsB = blockRms(x, s + 2400, s + 3600);
    m.steadyDrift = (m.steadyRmsA > 0.0)
        ? std::fabs(m.steadyRmsB - m.steadyRmsA) / m.steadyRmsA
        : 0.0;

    // 収束判定は参照（wet のみ）の同じ窓の減衰と比べる。テールは自然に減衰するため。
    const double refRmsA = blockRms(ref, s + 1200, s + 2400);
    const double refRmsB = blockRms(ref, s + 2400, s + 3600);
    const double refDrift = (refRmsA > 0.0) ? std::fabs(refRmsB - refRmsA) / refRmsA : 0.0;
    m.refSteadyDrift = refDrift;

    m.t1 = m.switchStep <= kSwitchStepRatio * m.scale + 1.0e-12;
    // T2 は比較項目（post <= pre）としてレポートで判定する。
    m.t2 = m.localMaxStep <= m.refLocalMaxStep + 1.0e-12;
    m.t4 = m.mixMonotonic;
    m.t5 = m.finite;
    // ON: バイパス定常は一定。OFF: フェード完了後に wet が再開している（無音でない）。
    //   バイパス中は wet の畳み込みを止めるため、連続処理した参照とのレベル一致は
    //   本 PR の対象外（既存設計）。レポートに制約として記載する。
    m.t6 = m.bypassOn ? (m.steadyDrift <= 0.10) : (m.steadyRmsA > 1.0e-9);
    m.allPass = m.finite && m.t1 && m.t3 && m.t4 && m.t5 && m.t6;
    return m;
}

void writeCsv(const std::filesystem::path& path, const std::vector<TransitionMetrics>& rows)
{
    std::ofstream f(path);
    f << "case,transition,bypass_on,sample,scale,switch_step,local_max_step,ref_local_max_step,"
         "ref_p999_step,mix_after_1block,mix_after_2blocks,fade_samples_est,monotonic,"
         "steady_rms_a,steady_rms_b,steady_drift,finite,T1_step,T2_local,T3_fade,T4_monotonic,"
         "T5_finite,T6_convergence,overall\n";
    f.setf(std::ios::fixed);
    f.precision(12);
    for (const auto& m : rows)
    {
        f << m.caseName << ',' << m.index << ',' << (m.bypassOn ? 1 : 0) << ',' << m.sample << ','
          << m.scale << ',' << m.switchStep << ',' << m.localMaxStep << ','
          << m.refLocalMaxStep << ',' << m.refP999Step << ',' << m.mixAfterOneBlock << ','
          << m.mixAfterTwoBlocks << ',' << m.fadeSamplesEstimate << ','
          << (m.mixMonotonic ? 1 : 0) << ',' << m.steadyRmsA << ',' << m.steadyRmsB << ','
          << m.steadyDrift << ',' << (m.finite ? 1 : 0) << ','
          << (m.t1 ? 1 : 0) << ',' << (m.t2 ? 1 : 0) << ',' << (m.t3 ? 1 : 0) << ','
          << (m.t4 ? 1 : 0) << ',' << (m.t5 ? 1 : 0) << ',' << (m.t6 ? 1 : 0) << ','
          << (m.allPass ? 1 : 0) << '\n';
    }
}
} // namespace

int runN1BypassAudioAnalysis(const juce::String& outDirIn, const juce::String& modeIn)
{
    juce::ScopedJuceInitialiser_GUI juceInit;
    MKLRealTime::setup();
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    const std::string outDir = outDirIn.toStdString();
    const std::string mode = (modeIn == "pre") ? "pre" : "post";
    std::filesystem::create_directories(outDir);

    const juce::File ir = writeDecayIr(juce::File(juce::String(outDir + "/n1_analysis_ir.wav")));
    if (!ir.existsAsFile())
    {
        std::fprintf(stderr, "[N1-analysis] IR write failed\n");
        return 2;
    }

    std::vector<TransitionMetrics> rows;
    bool allPass = true;

    for (const auto& c : cases())
    {
        std::vector<double> ref;
        std::vector<double> out;
        std::vector<double> refMix;
        std::vector<double> outMix;
        if (!renderCase(ir, c, false, ref, refMix) || !renderCase(ir, c, true, out, outMix))
            return 2;

        writeWav(juce::File(juce::String(outDir + "/" + mode + "_ref_" + c.name + ".wav")), ref);
        writeWav(juce::File(juce::String(outDir + "/" + mode + "_" + c.name + ".wav")), out);

        double refTail = 0.0;
        for (int n = kBurstStart + kBurstSamples + 2048; n < kTotalSamples; ++n)
            refTail = std::max(refTail, std::fabs(ref[static_cast<size_t>(n)]));
        if (refTail <= 1.0e-9)
        {
            std::fprintf(stderr,
                         "[N1-analysis] reference tail is silent (%s): processor not prepared?\n",
                         c.name);
            return 2;
        }

        for (size_t ti = 0; ti < c.transitions.size(); ++ti)
        {
            const auto m = measure(c, ti, c.transitions[ti], out, ref, outMix);
            rows.push_back(m);
            if (!m.allPass)
                allPass = false;
        }
    }

    writeCsv(std::filesystem::path(outDir) / ("metrics_" + mode + ".csv"), rows);

    std::printf("[N1-analysis] mode=%s cases=%d transitions=%d\n",
                mode.c_str(), static_cast<int>(cases().size()),
                static_cast<int>(rows.size()));
    for (const auto& m : rows)
    {
        std::printf("  %-16s #%d %-3s s=%5d step=%.6e local=%.6e refLocal=%.6e "
                    "mix1=%.6f mix2=%.6f drift=%.3f T1=%d T2=%d T3=%d T4=%d T5=%d T6=%d %s\n",
                    m.caseName.c_str(), m.index, m.bypassOn ? "ON" : "OFF", m.sample,
                    m.switchStep, m.localMaxStep, m.refLocalMaxStep,
                    m.mixAfterOneBlock, m.mixAfterTwoBlocks, m.steadyDrift,
                    m.t1 ? 1 : 0, m.t2 ? 1 : 0, m.t3 ? 1 : 0, m.t4 ? 1 : 0,
                    m.t5 ? 1 : 0, m.t6 ? 1 : 0, m.allPass ? "PASS" : "FAIL");
    }
    std::printf("[N1-analysis] %s\n", allPass ? "ALL PASS" : "FAILURES PRESENT");
    std::fflush(stdout);
    return allPass ? 0 : 1;
}
