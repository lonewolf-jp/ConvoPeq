#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

namespace convo {

// ============================================================================
// EngineState: 完全に不透明なバイナリスナップショット
// ============================================================================
// AudioEngine はこの構造体の内部を決して解釈してはならない。
// 責務は DSP コンポーネント側が serializeTo/deserializeFrom で読み書きする。
// ============================================================================

struct EngineState 
{
    // サイズはコンパイル時に固定。絶対に変更しない。
    static constexpr size_t kDSPSize   = 16384;
    static constexpr size_t kEQSize    =  4096;
    static constexpr size_t kSnapSize  =  2048;

    alignas(64) uint8_t dsp[kDSPSize];
    alignas(64) uint8_t eq[kEQSize];
    alignas(64) uint8_t snap[kSnapSize];

    bool isValid = false;

    // 固定長コピー（O(1) bounded）
    void copyFrom(const EngineState& src) noexcept 
    {
        std::memcpy(dsp, src.dsp, sizeof(dsp));
        std::memcpy(eq,  src.eq,  sizeof(eq));
        std::memcpy(snap, src.snap, sizeof(snap));
        isValid = src.isValid;
    }

    EngineState() 
    { 
        std::memset(dsp, 0, sizeof(dsp));
        std::memset(eq, 0, sizeof(eq));
        std::memset(snap, 0, sizeof(snap));
        isValid = false;
    }

    // コピー/ムーブコンストラクタと代入演算子は削除 (copyFrom のみ使用)
    EngineState(const EngineState&) = delete;
    EngineState& operator=(const EngineState&) = delete;
    EngineState(EngineState&&) = delete;
    EngineState& operator=(EngineState&&) = delete;
    
    ~EngineState() = default;
};

// バイナリ互換性の絶対保証
static_assert(sizeof(EngineState) ==
              EngineState::kDSPSize +
              EngineState::kEQSize +
              EngineState::kSnapSize +
              sizeof(bool),
              "EngineState size mismatch – binary compatibility broken");

// ============================================================================
// EngineView: 公開されるスナップショット
// ============================================================================
// ダブルバッファ配列内に value として保持。alignas でキャッシュライン分離。
// ============================================================================

struct alignas(64) EngineView 
{
    EngineState current;
    EngineState previous;
    float alpha = 1.0f;
    bool previousValid = false;

    EngineView() = default;

    // コピー/ムーブ禁止 (スロット直接操作のみ)
    EngineView(const EngineView&) = delete;
    EngineView& operator=(const EngineView&) = delete;
    EngineView(EngineView&&) = delete;
    EngineView& operator=(EngineView&&) = delete;
    
    ~EngineView() = default;
};

} // namespace convo
