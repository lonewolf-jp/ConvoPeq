//==============================================================================
// SnapshotAssembler.h
// パラメータから SnapshotParams を組み立てる純粋ビルダ（メモリ確保禁止）
// v13.0 Phase 2 改訂版 – enum class 対応
//==============================================================================
#pragma once

#include "SnapshotParams.h"
#include "EQParameters.h"
#include "Types.h"

namespace convo {

class SnapshotAssembler {
public:
    static SnapshotParams assemble(
        const ConvolverState* conv,
        const EQParameters& eq,
        const std::array<double, 9>& nsCoeffs,
        double inputHeadroomGain,
        double outputMakeupGain,
        double convInputTrimGain,
        bool convBypass,
        bool eqBypass,
        bool softClipEnabled,
        float saturationAmount,
        ProcessingOrder processingOrder,
        OversamplingType oversamplingType,
        int oversamplingFactor,
        int ditherBitDepth,
        NoiseShaperType noiseShaperType,
        uint64_t generation,
        double sampleRate,          // v2.3 フェーズ1 追加
        int maxBlockSize,           // v2.3 フェーズ1 追加
        uint64_t eqCoeffHash        // v2.3 フェーズ1 追加
    ) noexcept;

private:
    SnapshotAssembler() = delete;
};

} // namespace convo
