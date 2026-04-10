//==============================================================================
// Types.h
// DSP コアで使用される列挙型の一元定義（レイヤー分離のための基盤）
// v13.0 Phase 2
//==============================================================================
#pragma once

namespace convo {

// 処理順序（EQ と Convolver の直列接続順）
enum class ProcessingOrder {
    ConvolverThenEQ = 0,
    EQThenConvolver = 1
};

// オーバーサンプリングフィルタタイプ
enum class OversamplingType {
    IIR = 0,
    LinearPhase = 1
};

// ノイズシェーパータイプ
enum class NoiseShaperType {
    Psychoacoustic = 0,
    Fixed4Tap = 1,
    Adaptive9thOrder = 2,
    Fixed15Tap = 3
};

} // namespace convo
