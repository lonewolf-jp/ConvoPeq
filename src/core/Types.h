//==============================================================================
// Types.h
// DSP コアで使用される列挙型の一元定義（レイヤー分離のための基盤）
// v17.15 - ActiveSnapshot 構造体追加
//==============================================================================
#pragma once

#include <memory>
#include <cstdint>

namespace convo {

// Forward declarations
struct GlobalSnapshot;

/**
 * ActiveSnapshot - RCU 保護されたアクティブなスナップショット
 * 
 * 設計原則:
 *   1. 公開後不変原則：構築後一切の変更を許さない（const unique_ptr で保証）
 *   2. 所有権ツリー完全閉鎖：ActiveSnapshot → GlobalSnapshot → ConvolverState を
 *      unique_ptr で所有し、親子別個の retire は絶対に行わない
 *   3. 1 リタイア 1 ツリー原則：retire は常に ActiveSnapshot 単位で行う
 */
struct ActiveSnapshot {
    std::unique_ptr<const GlobalSnapshot> current;
    std::unique_ptr<const GlobalSnapshot> previous;
    float fadeAlpha;
    uint64_t generation;

    ActiveSnapshot(std::unique_ptr<const GlobalSnapshot> cur,
                   std::unique_ptr<const GlobalSnapshot> prev,
                   float alpha, 
                   uint64_t gen)
        : current(std::move(cur))
        , previous(std::move(prev))
        , fadeAlpha(alpha)
        , generation(gen)
    {}
    
    ~ActiveSnapshot() = default;
    
    // コピー禁止、ムーブ許可（所有権移動はムーブでのみ）
    ActiveSnapshot(const ActiveSnapshot&) = delete;
    ActiveSnapshot& operator=(const ActiveSnapshot&) = delete;
    ActiveSnapshot(ActiveSnapshot&&) = default;
    ActiveSnapshot& operator=(ActiveSnapshot&&) = default;
};

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
