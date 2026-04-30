#pragma once

#include <atomic>
#include <cstring>
#include <cstdint>
#include "core/Types.h"

namespace convo {

struct EngineState {
    // バイナリ互換性を保証するため、サイズ変更は絶対に禁止する
    static constexpr size_t kDSPCoreSize   = 16384;
    static constexpr size_t kEQStateSize   =  4096;
    static constexpr size_t kSnapshotSize  =  2048;

    alignas(64) uint8_t dspBlob[kDSPCoreSize];
    alignas(64) uint8_t eqBlob [kEQStateSize];
    alignas(64) uint8_t snapBlob[kSnapshotSize];

    uint64_t generation = 0;
    uint64_t captureSessionId = 0;
    double sampleRate = 0.0;
    int bitDepth = 24;
    NoiseShaperType noiseShaperType = NoiseShaperType::None;
    bool isValid = false;

    void copyFrom(const EngineState& src) noexcept {
        std::memcpy(dspBlob, src.dspBlob, sizeof(dspBlob));
        std::memcpy(eqBlob,  src.eqBlob,  sizeof(eqBlob));
        std::memcpy(snapBlob, src.snapBlob, sizeof(snapBlob));
        generation = src.generation;
        captureSessionId = src.captureSessionId;
        sampleRate = src.sampleRate;
        bitDepth = src.bitDepth;
        noiseShaperType = src.noiseShaperType;
        isValid    = src.isValid;
    }

    EngineState() = default;
    EngineState(const EngineState&) = delete;
    EngineState& operator=(const EngineState&) = delete;
    EngineState(EngineState&&) = delete;
    EngineState& operator=(EngineState&&) = delete;
};

struct alignas(64) EngineView {
    EngineState current;
    EngineState previous;
    float alpha = 1.0f;
    bool previousValid = false;

    EngineView() = default;
    EngineView(const EngineView&) = delete;
    EngineView& operator=(const EngineView&) = delete;
    EngineView(EngineView&&) = delete;
    EngineView& operator=(EngineView&&) = delete;
};

} // namespace convo
