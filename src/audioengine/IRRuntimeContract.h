#pragma once
// IRRuntimeContract.h — WORK105: IR／ランタイム形状契約（NonRT検証・純粋関数）
//
// 背景（WORK104 実行時確定）:
//   UI convolver はホストレートで IR を build するが（PrepareToPlay.cpp）、
//   DSP world は processing rate（host × OS）で畳み込む。IRState がホスト
//   レートのまま DSP world へ transfer されると、パーティション畳み込みの
//   前提（rate 一致・block quantum 一致）が崩れ、無警告でガベージを出力する
//   （C1: 0.0474／THD-16.5dB／超音波+82dB、D実験で OS 変更が性状を変える）。
//
// 本契約は Build（NonRT）時点で照合し、不一致なら publish を拒否する。
// RT 側へ検証を持ち込まない（RT は Read→Execute→Output に限定）。
// 既存の RT oversize gate（M-04）は defense-in-depth として維持。
//
// 依存: なし（JUCE/MKL 非依存・単体テスト容易）。noexcept・確保なし。

#include <cmath>
#include <cstdint>

namespace convo {

enum class IRRuntimeContractViolation : std::uint8_t {
    None = 0,
    RateMismatch = 1,   // IR rate != world processing rate → publish 拒否
    BlockMismatch = 2,  // IR block quantum != world processing quantum → publish 拒否
    UnknownSourceGeometry = 3, // IR 側形状不明（旧経路）→ 許可＋呼び出し側で loud log
};

struct IRRuntimeContractResult {
    IRRuntimeContractViolation violation = IRRuntimeContractViolation::None;
    [[nodiscard]] bool ok() const noexcept
    {
        return violation == IRRuntimeContractViolation::None
            || violation == IRRuntimeContractViolation::UnknownSourceGeometry;
    }
    [[nodiscard]] bool refused() const noexcept { return !ok(); }
};

// irRate: IRState.sampleRate（IR build 時の processing rate。<=0 は不明）
// irBlock: IRState.blockSize（IR build 時の processing quantum。<=0 は不明）
// worldRate/worldBlock: DSP prepare 直後の convolver 形状（必ず既知のはず）。
// 判定は純粋・決定的。呼び出し側（RuntimeBuilder::build）は refused 時に
// BuildError::IRRateMismatch / IRBlockMismatch で publish を拒否すること。
[[nodiscard]] inline IRRuntimeContractResult checkIRRuntimeContract(
    double irRate, int irBlock, double worldRate, int worldBlock) noexcept
{
    if (!(irRate > 0.0) || irBlock <= 0)
        return { IRRuntimeContractViolation::UnknownSourceGeometry };
    if (!(worldRate > 0.0) || worldBlock <= 0)
        return { IRRuntimeContractViolation::UnknownSourceGeometry };
    const double denom = irRate > worldRate ? irRate : worldRate;
    const double relDiff = (irRate > worldRate ? (irRate - worldRate) : (worldRate - irRate)) / denom;
    if (relDiff > 1.0e-9)
        return { IRRuntimeContractViolation::RateMismatch };
    if (irBlock != worldBlock)
        return { IRRuntimeContractViolation::BlockMismatch };
    return { IRRuntimeContractViolation::None };
}

[[nodiscard]] inline const char* toString(IRRuntimeContractViolation v) noexcept
{
    switch (v)
    {
        case IRRuntimeContractViolation::None: return "None";
        case IRRuntimeContractViolation::RateMismatch: return "RateMismatch";
        case IRRuntimeContractViolation::BlockMismatch: return "BlockMismatch";
        case IRRuntimeContractViolation::UnknownSourceGeometry: return "UnknownSourceGeometry";
    }
    return "Unknown";
}

} // namespace convo
