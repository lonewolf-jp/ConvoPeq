#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include "MKLNonUniformConvolver.h"
#include "AlignedAllocation.h"

namespace convo {

    /**
     * @struct StereoConvolver
     * @brief 2ch分の MKLNonUniformConvolver を管理する構造体。
     * 
     * 参照カウントによるライフサイクル管理を行い、Audio Thread から安全に参照できる。
     */
    struct StereoConvolver
    {
        StereoConvolver() = default;
        ~StereoConvolver()
        {
            for (auto& nuc : nucConvolvers)
                nuc.reset();
            
            if (irData[0]) { convo::aligned_free(irData[0]); irData[0] = nullptr; }
            if (irData[1]) { convo::aligned_free(irData[1]); irData[1] = nullptr; }
        }

        // 参照カウント (RCU用)
        void addRef() noexcept { refCount.fetch_add(1, std::memory_order_relaxed); }
        void release() noexcept
        {
            if (refCount.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                this->~StereoConvolver();
                convo::aligned_free(const_cast<StereoConvolver*>(this));
            }
        }

        std::atomic<int> refCount{ 0 };

        // NUC エンジン (L/R)
        std::array<convo::ScopedAlignedPtr<convo::MKLNonUniformConvolver>, 2> nucConvolvers;

        // IR データ (MKLアライメント済み)
        double* irData[2] = { nullptr, nullptr };
        int irDataLength = 0;
        int irLatency = 0;
        int latency = 0;

        // 設定値のキャッシュ
        double storedSampleRate = 0.0;
        int storedMaxFFTSize = 0;
        int storedKnownBlockSize = 0;
        int storedFirstPartition = 0;
        int callQuantumSamples = 0;
        int prewarmedMaxSamples = 0;
        double storedScale = 1.0;
        bool storedDirectHeadEnabled = false;

        bool init(double* irL, double* irR, int length, double sr, int peakDelay,
                  int maxFFTSize, int knownBlockSize, int firstPartition, int preferredCallSize, double scale,
                  bool enableDirectHead,
                  const convo::FilterSpec* filterSpec = nullptr)
        {
            // Ownership transfer
            irData[0] = irL;
            irData[1] = irR;
            irDataLength = length;
            this->irLatency = peakDelay;
            callQuantumSamples = juce::jmax(1, preferredCallSize);
            prewarmedMaxSamples = callQuantumSamples;
            storedSampleRate = sr;
            storedMaxFFTSize = maxFFTSize;
            storedKnownBlockSize = knownBlockSize;
            storedFirstPartition = firstPartition;
            storedScale = scale;
            storedDirectHeadEnabled = enableDirectHead;

            try
            {
                void* rn0 = convo::aligned_malloc(sizeof(convo::MKLNonUniformConvolver), 64);
                new (rn0) convo::MKLNonUniformConvolver();
                nucConvolvers[0].reset(static_cast<convo::MKLNonUniformConvolver*>(rn0));

                void* rn1 = convo::aligned_malloc(sizeof(convo::MKLNonUniformConvolver), 64);
                new (rn1) convo::MKLNonUniformConvolver();
                nucConvolvers[1].reset(static_cast<convo::MKLNonUniformConvolver*>(rn1));

                if (nucConvolvers[0]->SetImpulse(irData[0], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec) &&
                    nucConvolvers[1]->SetImpulse(irData[1], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec))
                {
                    latency = nucConvolvers[0]->getLatency();
                    return true;
                }
            }
            catch (const std::bad_alloc&) {}

            nucConvolvers[0].reset();
            nucConvolvers[1].reset();
            return false;
        }

        void reset()
        {
            for (auto& nuc : nucConvolvers)
                if (nuc) nuc->Reset();
        }

        void process(int channel, const double* in, double* out, int numSamples)
        {
            if (channel < 2 && nucConvolvers[channel])
                nucConvolvers[channel]->Process(in, out, numSamples);
        }

        bool areNUCDescriptorsCommitted() const noexcept
        {
            for (const auto& conv : nucConvolvers)
            {
                if (!conv || !conv->areFftDescriptorsCommitted())
                    return false;
            }
            return true;
        }
    };

} // namespace convo
