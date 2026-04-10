//==============================================================================
// SnapshotParams.h
// スナップショット生成パラメータ受け渡し用の値構造体
// v13.0 Phase 2 改訂版 – enum class を保持
//==============================================================================
#pragma once

#include <array>
#include <cstdint>
#include "../ConvolverState.h"
#include "EQParameters.h"
#include "Types.h"

namespace convo {

struct SnapshotParams {
    const ConvolverState* convState = nullptr;
    EQParameters eqParams{};
    std::array<double, 9> nsCoeffs{};

    double inputHeadroomGain = 0.0;
    double outputMakeupGain = 0.0;
    double convInputTrimGain = 1.0;

    bool convBypass = false;
    bool eqBypass = false;
    ProcessingOrder processingOrder = ProcessingOrder::ConvolverThenEQ;

    bool softClipEnabled = false;
    float saturationAmount = 0.0f;

    OversamplingType oversamplingType = OversamplingType::IIR;
    int oversamplingFactor = 1;

    int ditherBitDepth = 24;
    NoiseShaperType noiseShaperType = NoiseShaperType::Psychoacoustic;

    uint64_t generation = 0;

    SnapshotParams() = default;
    SnapshotParams(const SnapshotParams&) = default;
    SnapshotParams& operator=(const SnapshotParams&) = default;
};

} // namespace convo
