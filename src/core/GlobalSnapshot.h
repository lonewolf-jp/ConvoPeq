//==============================================================================
// GlobalSnapshot.h
// 不変な DSP パラメータコンテナ（値型・コピー禁止）
// v13.0 Phase 2 改訂版 – SnapshotParams から一括構築
//==============================================================================
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include "../ConvolverState.h"
#include "EQParameters.h"
#include "SnapshotParams.h"
#include "Types.h"

namespace convo {

struct GlobalSnapshot {
    // ---------- 観測専用ポインタ（寿命は SafeStateSwapper 管理） ----------
    const ConvolverState* convState = nullptr;
    uint64_t convStateId = 0;

    // ---------- 値型パラメータ（完全自己完結） ----------
    EQParameters eqParams{};
    std::array<double, 9> nsCoeffs{};
    uint64_t contentHash = 0;           // パラメータの一意性ハッシュ（高速否定用）

    // ---------- ゲイン関連 ----------
    double inputHeadroomGain = 0.0;
    double outputMakeupGain = 0.0;
    double convInputTrimGain = 1.0;

    // ---------- バイパスと処理順序 ----------
    bool convBypass = false;
    bool eqBypass = false;
    ProcessingOrder processingOrder = ProcessingOrder::ConvolverThenEQ;

    // ---------- ソフトクリップ ----------
    bool softClipEnabled = false;
    float saturationAmount = 0.0f;

    // ---------- オーバーサンプリング ----------
    OversamplingType oversamplingType = OversamplingType::IIR;
    int oversamplingFactor = 1;

    // ---------- ノイズシェーパー ----------
    int ditherBitDepth = 24;
    NoiseShaperType noiseShaperType = NoiseShaperType::Psychoacoustic;

    // ---------- 世代管理 ----------
    uint64_t generation = 0;

    // v2.3 フェーズ 1 追加フィールド
    uint64_t eqCoeffHash = 0;           // EQパラメータハッシュ値（キャッシュ識別子）

#ifdef _DEBUG
    mutable std::atomic<bool> alive{true};
    uint64_t snapshotId = 0;
#endif

    // SnapshotParams から全フィールドを初期化
    explicit GlobalSnapshot(const SnapshotParams& params) noexcept;

    // コピー・ムーブ禁止
    GlobalSnapshot(const GlobalSnapshot&) = delete;
    GlobalSnapshot& operator=(const GlobalSnapshot&) = delete;
    GlobalSnapshot(GlobalSnapshot&&) = delete;
    GlobalSnapshot& operator=(GlobalSnapshot&&) = delete;

    // デストラクタは public（unique_ptr で所有されるためデフォルト実装）
    ~GlobalSnapshot() = default;
};

} // namespace convo
