//============================================================================
// PolyphaseGainMetrics.cpp — numeric metrics for the polyphase gain convention
//
// Test-only. Prints one [METRIC] line per configuration so the same binary
// built with CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF and ON can be compared
// without listening tests. No pass/fail threshold is applied here: the
// metrics are evidence for the adoption decision.
//
// Measured per configuration (IIR3 r2/r4/r8, LP3 r2/r8):
//   DC gain, step overshoot and settling, broadband RMS/peak ratio,
//   frequency response at 0.05/0.10/0.20/0.40 of the base rate,
//   impulse latency (argmax), CPU time per 512-sample block.
//============================================================================
#include <JuceHeader.h>
#include "CustomInputOversampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
constexpr int kBlock = 512;
constexpr int kChannels = 2;

struct Config
{
    const char* name;
    int ratio;
    CustomInputOversampler::Preset preset;
};

const Config kConfigs[] = {
    { "IIR3_r2", 2, CustomInputOversampler::Preset::IIRLike },
    { "IIR3_r4", 4, CustomInputOversampler::Preset::IIRLike },
    { "IIR3_r8", 8, CustomInputOversampler::Preset::IIRLike },
    { "LP3_r2",  2, CustomInputOversampler::Preset::LinearPhase },
    { "LP3_r8",  8, CustomInputOversampler::Preset::LinearPhase },
};

constexpr int kMaxRatio = 8;
constexpr int kUpMax = kBlock * kMaxRatio;

struct Runner
{
    CustomInputOversampler os;
    double inL[kBlock] = {};
    double inR[kBlock] = {};
    double upL[kUpMax] = {};
    double upR[kUpMax] = {};
    double dnL[kBlock] = {};
    double dnR[kBlock] = {};

    void init(const Config& c)
    {
        os.prepare(kBlock, c.ratio, c.preset);
        os.reset();
    }

    void reset()
    {
        os.reset();
        std::fill(std::begin(inL), std::end(inL), 0.0);
        std::fill(std::begin(inR), std::end(inR), 0.0);
    }

    void roundTrip()
    {
        double* inPtrs[kChannels] = { inL, inR };
        juce::dsp::AudioBlock<double> inBlock(inPtrs, kChannels, static_cast<std::size_t>(kBlock));
        auto upBlock = os.processUp(inBlock, kChannels);
        const std::size_t upN = upBlock.getNumSamples();
        std::memcpy(upL, upBlock.getChannelPointer(0), upN * sizeof(double));
        std::memcpy(upR, upBlock.getChannelPointer(1), upN * sizeof(double));

        double* upPtrs[kChannels] = { upL, upR };
        double* dnPtrs[kChannels] = { dnL, dnR };
        juce::dsp::AudioBlock<double> upSrc(upPtrs, kChannels, upN);
        juce::dsp::AudioBlock<double> outBlock(dnPtrs, kChannels, static_cast<std::size_t>(kBlock));
        os.processDown(upSrc, outBlock, kChannels);
    }
};

double meanOf(const double* p, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += p[i];
    return n > 0 ? s / static_cast<double>(n) : 0.0;
}

double rmsOf(const double* a, const double* b, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * a[i] + b[i] * b[i];
    return n > 0 ? std::sqrt(s / static_cast<double>(2 * n)) : 0.0;
}

double peakOf(const double* a, const double* b, int n)
{
    double p = 0.0;
    for (int i = 0; i < n; ++i) p = std::max(p, std::max(std::fabs(a[i]), std::fabs(b[i])));
    return p;
}

int argmaxOffset(const double* p, int n)
{
    int best = -1;
    double v = 0.0;
    for (int i = 0; i < n; ++i)
    {
        if (std::fabs(p[i]) > v)
        {
            v = std::fabs(p[i]);
            best = i;
        }
    }
    return best;
}

double dcGain(const Config& c)
{
    Runner r;
    r.init(c);
    double last = 0.0;
    for (int block = 0; block < 12; ++block)
    {
        for (int i = 0; i < kBlock; ++i)
        {
            r.inL[i] = 0.5;
            r.inR[i] = 0.5;
        }
        r.roundTrip();
        if (block == 11) last = 0.5 * (meanOf(r.dnL + kBlock - 128, 128) + meanOf(r.dnR + kBlock - 128, 128));
    }
    return last / 0.5;
}

void stepMetrics(const Config& c, double& overshoot, int& settleSamples)
{
    // Pass 1: measure the final value and the peak of the settled block.
    double finalValue = 0.0;
    double peak = 0.0;
    {
        Runner r;
        r.init(c);
        for (int block = 0; block < 8; ++block)
        {
            const bool high = block >= 4;
            for (int i = 0; i < kBlock; ++i)
            {
                r.inL[i] = high ? 1.0 : 0.0;
                r.inR[i] = high ? 1.0 : 0.0;
            }
            r.roundTrip();
            if (block == 7)
            {
                finalValue = 0.5 * (meanOf(r.dnL + kBlock - 64, 64) + meanOf(r.dnR + kBlock - 64, 64));
                peak = peakOf(r.dnL, r.dnR, kBlock);
            }
        }
    }
    overshoot = finalValue != 0.0 ? peak / std::fabs(finalValue) - 1.0 : 0.0;

    // Pass 2: first sample within 1% of the final value (relative to the step).
    settleSamples = -1;
    Runner r2;
    r2.init(c);
    for (int block = 0; block < 8 && settleSamples < 0; ++block)
    {
        const bool high = block >= 4;
        for (int i = 0; i < kBlock; ++i)
        {
            r2.inL[i] = high ? 1.0 : 0.0;
            r2.inR[i] = high ? 1.0 : 0.0;
        }
        r2.roundTrip();
        if (!high) continue;
        const double tol = 0.01 * std::fabs(finalValue);
        for (int i = 0; i < kBlock; ++i)
        {
            if (std::fabs(r2.dnL[i] - finalValue) <= tol && std::fabs(r2.dnR[i] - finalValue) <= tol)
            {
                settleSamples = (block - 4) * kBlock + i;
                break;
            }
        }
    }
}

double latencyOf(const Config& c)
{
    Runner r;
    r.init(c);
    r.inL[10] = 1.0;
    r.inR[10] = 1.0;
    r.roundTrip();
    const int a = argmaxOffset(r.dnL, kBlock);
    const int b = argmaxOffset(r.dnR, kBlock);
    return static_cast<double>(std::max(a, b) - 10);
}

struct Broadband
{
    double rmsRatio = 0.0;
    double peakRatio = 0.0;
};

Broadband broadbandMetrics(const Config& c)
{
    Runner r;
    r.init(c);
    std::uint64_t s = 0x123456789ABCDEFULL;
    auto next = [&s]() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<double>((s >> 11) & 0x1FFFFFFFFFFFFFULL) / static_cast<double>(0x1FFFFFFFFFFFFFULL) - 0.5;
    };
    double inRms = 0.0, outRms = 0.0, inPeak = 0.0, outPeak = 0.0;
    for (int block = 0; block < 16; ++block)
    {
        for (int i = 0; i < kBlock; ++i)
        {
            r.inL[i] = next();
            r.inR[i] = next();
        }
        r.roundTrip();
        if (block >= 12)
        {
            inRms += rmsOf(r.inL, r.inR, kBlock);
            outRms += rmsOf(r.dnL, r.dnR, kBlock);
            inPeak = std::max(inPeak, peakOf(r.inL, r.inR, kBlock));
            outPeak = std::max(outPeak, peakOf(r.dnL, r.dnR, kBlock));
        }
    }
    Broadband b;
    b.rmsRatio = inRms > 0.0 ? outRms / inRms : 0.0;
    b.peakRatio = inPeak > 0.0 ? outPeak / inPeak : 0.0;
    return b;
}

double responseAt(const Config& c, double f)
{
    Runner r;
    r.init(c);
    const double pi = 3.14159265358979323846;
    double inRms = 0.0, outRms = 0.0;
    int n = 0;
    for (int block = 0; block < 12; ++block)
    {
        for (int i = 0; i < kBlock; ++i)
        {
            const double phase = 2.0 * pi * f * static_cast<double>(n + i);
            r.inL[i] = 0.5 * std::sin(phase);
            r.inR[i] = 0.5 * std::sin(phase);
        }
        r.roundTrip();
        if (block >= 8)
        {
            inRms += rmsOf(r.inL, r.inR, kBlock);
            outRms += rmsOf(r.dnL, r.dnR, kBlock);
        }
        n += kBlock;
    }
    return inRms > 0.0 ? outRms / inRms : 0.0;
}

void cpuMetrics(const Config& c, double& avgUs, double& maxUs)
{
    Runner r;
    r.init(c);
    const double pi = 3.14159265358979323846;
    int n = 0;
    for (int block = 0; block < 8; ++block)  // warmup
    {
        for (int i = 0; i < kBlock; ++i) { r.inL[i] = 0.1 * std::sin(0.01 * (n + i)); r.inR[i] = r.inL[i]; }
        r.roundTrip();
        n += kBlock;
    }
    double total = 0.0;
    maxUs = 0.0;
    constexpr int kBlocks = 200;
    for (int block = 0; block < kBlocks; ++block)
    {
        for (int i = 0; i < kBlock; ++i) { r.inL[i] = 0.1 * std::sin(0.01 * (n + i)); r.inR[i] = r.inL[i]; }
        const auto t0 = std::chrono::steady_clock::now();
        r.roundTrip();
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        total += us;
        maxUs = std::max(maxUs, us);
        n += kBlock;
    }
    avgUs = total / static_cast<double>(kBlocks);
}
} // namespace

int main()
{
#if defined(CONVOPEQ_CORRECT_POLYPHASE_GAIN) && (CONVOPEQ_CORRECT_POLYPHASE_GAIN)
    std::printf("[METRIC-BUILD] flag=ON\n");
#else
    std::printf("[METRIC-BUILD] flag=OFF\n");
#endif
    for (const auto& c : kConfigs)
    {
        double overshoot = 0.0, avgUs = 0.0, maxUs = 0.0;
        int settle = -1;
        const double dc = dcGain(c);
        stepMetrics(c, overshoot, settle);
        const Broadband b = broadbandMetrics(c);
        const double latency = latencyOf(c);
        cpuMetrics(c, avgUs, maxUs);
        std::printf("[METRIC] cfg=%s dc=%.9f overshoot=%.6f settle=%d rmsRatio=%.9f peakRatio=%.9f "
                    "g005=%.9f g010=%.9f g020=%.9f g040=%.9f latency=%.1f usAvg=%.3f usMax=%.3f\n",
                    c.name, dc, overshoot, settle, b.rmsRatio, b.peakRatio,
                    responseAt(c, 0.05), responseAt(c, 0.10), responseAt(c, 0.20), responseAt(c, 0.40),
                    latency, avgUs, maxUs);
    }
    return 0;
}
