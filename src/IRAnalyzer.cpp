#include "IRAnalyzer.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <memory>

namespace IRAnalyzer {

//==============================================================================
// SimpleRealFFT — 自己完結型 radix-2 DIT Cooley-Tukey 実数→CCS FFT
//==============================================================================
static void simpleRealFFT(double* data, int N) noexcept
{
    std::vector<std::complex<double>> buf(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        buf[static_cast<size_t>(i)] = std::complex<double>(data[i], 0.0);

    // ビット逆順並べ替え
    for (int i = 1, j = 0; i < N; ++i)
    {
        int bit = N >> 1;
        while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) std::swap(buf[static_cast<size_t>(i)], buf[static_cast<size_t>(j)]);
    }

    // Cooley-Tukey 反復 FFT (DIT)
    const double pi = juce::MathConstants<double>::pi;
    for (int len = 2; len <= N; len <<= 1)
    {
        const double angle = -2.0 * pi / static_cast<double>(len);
        const std::complex<double> w(std::cos(angle), std::sin(angle));
        for (int i = 0; i < N; i += len)
        {
            std::complex<double> twiddle(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j)
            {
                const size_t i1 = static_cast<size_t>(i + j);
                const size_t i2 = static_cast<size_t>(i + j + len / 2);
                const auto t = twiddle * buf[i2];
                const auto u = buf[i1];
                buf[i1] = u + t;
                buf[i2] = u - t;
                twiddle *= w;
            }
        }
    }

    // 複素結果を CCS 形式にパック
    const int halfN = N / 2;
    data[0] = buf[0].real();
    data[1] = buf[static_cast<size_t>(halfN)].real();
    for (int k = 1; k < halfN; ++k)
    {
        const size_t idx = static_cast<size_t>(2 * k);
        data[idx]     = buf[static_cast<size_t>(k)].real();
        data[idx + 1] = buf[static_cast<size_t>(k)].imag();
    }
}

//==============================================================================
double estimateMaxFrequencyResponseGain(
    const juce::AudioBuffer<double>& ir) noexcept
{
    const int numSamples = ir.getNumSamples();
    const int numChannels = ir.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0)
        return 1.0;

    const int copyLen = std::min(numSamples, kMaxAnalysisWindow);
    const int fftSize = juce::nextPowerOfTwo(copyLen);
    if (fftSize < 2)
        return 1.0;

    // Tukey 窓生成 (α=0.5)
    const double pi = juce::MathConstants<double>::pi;
    const double taperLen = kTukeyAlpha * static_cast<double>(fftSize - 1) * 0.5;
    auto tukeyWindow = std::make_unique<double[]>(static_cast<size_t>(fftSize));
    for (int i = 0; i < fftSize; ++i)
    {
        const double t = static_cast<double>(i);
        if (t < taperLen)
        {
            const double cosArg = (2.0 * pi * t) / (kTukeyAlpha * static_cast<double>(fftSize - 1));
            tukeyWindow[i] = 0.5 * (1.0 + std::cos(cosArg - pi));
        }
        else if (t > static_cast<double>(fftSize - 1) - taperLen)
        {
            const double cosArg = (2.0 * pi * (t - (static_cast<double>(fftSize - 1) - taperLen)))
                                  / (kTukeyAlpha * static_cast<double>(fftSize - 1));
            tukeyWindow[i] = 0.5 * (1.0 + std::cos(cosArg));
        }
        else { tukeyWindow[i] = 1.0; }
    }

    double windowSum = 0.0;
    for (int i = 0; i < copyLen; ++i) windowSum += tukeyWindow[i];
    const double windowMean = windowSum / static_cast<double>(copyLen);
    if (windowMean < 1e-18) return 1.0;

    double maxMagnitude = 0.0;

    // 自己完結型 FFT (MKL/IPP 非依存)
    for (int ch = 0; ch < numChannels; ++ch)
    {
        const double* src = ir.getReadPointer(ch);
        std::vector<double> out(static_cast<size_t>(fftSize) + 2, 0.0);
        for (int i = 0; i < copyLen; ++i)
            out[static_cast<size_t>(i)] = src[i] * tukeyWindow[i];

        simpleRealFFT(out.data(), fftSize);

        const int numBins = fftSize / 2;
        for (int bin = 0; bin <= numBins; ++bin)
        {
            double re = 0.0, im = 0.0;
            if (bin == 0)        { re = out[0]; im = 0.0; }
            else if (bin == numBins) { re = out[1]; im = 0.0; }
            else                 { re = out[2 * bin]; im = out[2 * bin + 1]; }
            const double mag = std::sqrt(re * re + im * im);
            maxMagnitude = std::max(maxMagnitude, mag);
        }

        // 3点ガウス補間
        {
            auto mags = std::make_unique<double[]>(static_cast<size_t>(numBins + 1));
            mags[0] = std::abs(out[0]);
            for (int b = 1; b < numBins; ++b)
            {
                const int idx = 2 * b;
                mags[b] = std::sqrt(out[idx] * out[idx] + out[idx + 1] * out[idx + 1]);
            }
            mags[numBins] = std::abs(out[1]);
            for (int b = 1; b < numBins - 1; ++b)
            {
                const double ym1 = mags[b - 1], y0 = mags[b], yp1 = mags[b + 1];
                if (y0 > ym1 && y0 > yp1 && y0 > 1e-18 && ym1 > 1e-18 && yp1 > 1e-18)
                {
                    const double logYm1 = std::log(ym1), logY0 = std::log(y0), logYp1 = std::log(yp1);
                    const double denom = logYm1 - 2.0 * logY0 + logYp1;
                    if (std::abs(denom) > 1e-18)
                    {
                        const double delta = 0.5 * (logYm1 - logYp1) / denom;
                        maxMagnitude = std::max(maxMagnitude, y0 * std::exp(-delta * (logY0 - logYm1)));
                    }
                }
            }
        }
    }

    // コヒーレントゲイン補正
    maxMagnitude /= windowMean;
    return (maxMagnitude > 1e-18) ? maxMagnitude : 1.0;
}

} // namespace IRAnalyzer
