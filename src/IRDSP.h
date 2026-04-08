#pragma once

#include <JuceHeader.h>
#include <functional>
#include "CDSPResampler.h"

struct ResampleConfig {
    double transBand = 2.0;                // 遷移帯域幅（1.0=鋭い, 3.0=速い）
    double stopBandAtten = 140.0;          // 減衰量（dB）
    r8b::EDSPFilterPhaseResponse phase = r8b::fprLinearPhase;
    int chunkSizeBase = 2048;              // チャンクサイズ（1024〜8192推奨）
};

namespace IRDSP {
    // 高品質リサンプリング（r8brain使用）
    juce::AudioBuffer<double> resampleIR(
        const juce::AudioBuffer<double>& inputIR,
        double inputSR,
        double targetSR,
        const std::function<bool()>& shouldExit = nullptr,
        const ResampleConfig& cfg = {});
}
