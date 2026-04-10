//==============================================================================
// EQParameters.h
// 20バンドEQのパラメータを値型で保持する構造体（外部寿命依存を排除）
// v13.0 設計ロック準拠
//==============================================================================
#pragma once

#include <array>

namespace convo {

// 1バンドのパラメータ
struct EQBandParams {
    float frequency = 1000.0f;
    float gain = 0.0f;
    float q = 0.707f;
    bool enabled = true;
    int type = 1;        // 0:LowShelf, 1:Peaking, 2:HighShelf, 3:LowPass, 4:HighPass
    int channelMode = 0; // 0:Stereo, 1:Left, 2:Right
};

// 20バンド EQ の全パラメータ（値型、コピー可能）
struct EQParameters {
    std::array<EQBandParams, 20> bands{};
    float totalGainDb = 0.0f;
    bool agcEnabled = false;               // AGC 有効フラグ（スナップショット保持対象）
    float nonlinearSaturation = 0.2f;      // SVF 非線形飽和度（0.0〜1.0）
    int filterStructure = 0;               // 0:Serial, 1:Parallel（スナップショット保持対象）

    // デフォルト値で初期化
    EQParameters() {
        const float defaultFreqs[20] = {
            20.0f, 32.0f, 50.0f, 80.0f, 125.0f,
            200.0f, 315.0f, 500.0f, 800.0f, 1250.0f,
            2000.0f, 3150.0f, 5000.0f, 8000.0f, 12500.0f,
            16000.0f, 19000.0f, 20000.0f, 22000.0f, 24000.0f
        };
        for (int i = 0; i < 20; ++i) {
            bands[i].frequency = defaultFreqs[i];
            bands[i].gain = 0.0f;
            bands[i].q = 0.707f;
            bands[i].enabled = true;
            bands[i].type = 1;
            bands[i].channelMode = 0;
        }
    }
};

} // namespace convo
