#pragma once

//============================================================================
// ConvolverProcessor.Internal.h
// 内部ヘルパー関数、定数、構造体の集約
//============================================================================

#include <JuceHeader.h>
#include "CDSPResampler.h"
#include <mkl.h>
#include <mkl_vml.h>

namespace ConvolverProcessorInternal
{

    // ────────────────────────────────────────────────────────────────
    // スレッドキャンセル確認
    // ────────────────────────────────────────────────────────────────
    inline bool checkCancellation(const std::function<bool()>& shouldExit, bool* wasCancelled = nullptr) noexcept
    {
        if (shouldExit && shouldExit())
        {
            if (wasCancelled)
                *wasCancelled = true;
            return true;
        }
        return false;
    }

    // ────────────────────────────────────────────────────────────────
    // 位相差アンラップ
    // ────────────────────────────────────────────────────────────────
    inline void unwrapPhaseRadians(double* phase, int size, double tol = juce::MathConstants<double>::pi)
    {
        if (size < 2) return;
        double correction = 0.0;
        for (int i = 1; i < size; ++i)
        {
            double delta = phase[i] - phase[i - 1];
            if (delta > tol)
                correction -= 2.0 * juce::MathConstants<double>::pi;
            else if (delta < -tol)
                correction += 2.0 * juce::MathConstants<double>::pi;
            phase[i] += correction;
        }
    }

    // ────────────────────────────────────────────────────────────────
    // メモリ削減：AudioBuffer 容量最適化
    // ────────────────────────────────────────────────────────────────
    inline void shrinkToFit(juce::AudioBuffer<double>& buffer)
    {
        if (buffer.getNumSamples() == 0 || buffer.getNumChannels() == 0)
            return;

        juce::AudioBuffer<double> newBuffer(buffer.getNumChannels(), buffer.getNumSamples());
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            newBuffer.copyFrom(ch, 0, buffer, ch, 0, buffer.getNumSamples());

        buffer = std::move(newBuffer);
    }

    // ────────────────────────────────────────────────────────────────
    // リサンプリング結果の状態
    // ────────────────────────────────────────────────────────────────
    enum class ResampleResult { Success, SilentIR, Cancelled, Error };
    struct ResampleOutput {
        juce::AudioBuffer<double> buffer;
        ResampleResult result;
    };

    // Shared helpers used across split translation units.
    ResampleOutput resampleIR(const juce::AudioBuffer<double>& inputIR,
                              double inputSR,
                              double targetSR,
                              r8b::EDSPFilterPhaseResponse phaseMode,
                              const std::function<bool()>& shouldExit);

    bool applyAsymmetricTukey(double* data, int numSamples);

    int estimateEffectiveIRLengthSamples(const juce::AudioBuffer<double>& irBuffer,
                                         double sampleRate);

    bool loadImpulseResponsePreviewFile(const juce::File& file,
                                        juce::AudioBuffer<double>& loadedIR,
                                        double& loadedSampleRate,
                                        juce::String& errorMessage);

    juce::AudioBuffer<double> convertToMinimumPhase(const juce::AudioBuffer<double>& linearIR,
                                                    const std::function<bool()>& shouldExit,
                                                    bool* wasCancelled);

    // ────────────────────────────────────────────────────────────────
    // 2 の累乗へ切り上げ
    // ────────────────────────────────────────────────────────────────
    inline int nextPow2(int x)
    {
        if (x <= 0) return 1;
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return x + 1;
    }

    // ────────────────────────────────────────────────────────────────
    // コンボリューション サイジング計算
    // ────────────────────────────────────────────────────────────────
    struct ConvolverSizing
    {
        int firstPartition;
        int maxFFTSize;
    };

    inline ConvolverSizing computeMasteringSizing(int internalBlockSize, int irLength)
    {
        ConvolverSizing s{};
        int fp = nextPow2(internalBlockSize * 4);
        fp = std::clamp(fp, 4096, 16384);
        s.firstPartition = fp;

        int mfsBase = irLength / 4;
        constexpr int kMFSUpper = 131072;
        mfsBase = std::clamp(mfsBase, s.firstPartition, kMFSUpper);
        s.maxFFTSize = nextPow2(mfsBase);

        if (s.maxFFTSize < s.firstPartition)
            s.maxFFTSize = s.firstPartition;
        if (s.maxFFTSize < internalBlockSize)
            s.maxFFTSize = nextPow2(internalBlockSize);

        return s;
    }

} // namespace ConvolverProcessorInternal
