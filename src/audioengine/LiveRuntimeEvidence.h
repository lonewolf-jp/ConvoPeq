//============================================================================
// LiveRuntimeEvidence.h — Live Runtime 証拠の観測値
//
// 本構造体は「実行中のプロセスが観測した値」だけを持つ。値の生成・補完はしない。
// 出力は EvidenceExporter（ISREvidenceExporter.h）が行い、Exporter は Runtime 実行の
// 責務を持たない（実行主体 = AudioEngine / 出力主体 = Exporter の分離）。
//
// 用途: `--verify-runtime-evidence-live` の証拠生成。
//   既存の `--verify-runtime-evidence`（Exporter 単体呼び出し）とは別物である。
//============================================================================
#pragma once

#include <cstdint>
#include <string>

namespace convo::isr
{

struct LiveRuntimeObservation
{
    // 実行識別（環境変数 CONVO_ISR_RUNTIME_RUN_ID 由来。空なら失敗扱い）
    std::string runId;
    int processId = 0;

    // プロセス内の単調時計（steady_clock）。窓判定はファイル mtime で行う。
    std::uint64_t startedAtNs = 0;
    std::uint64_t emittedAtNs = 0;

    // Runtime（AudioEngine）の観測値
    bool engineInitialized = false;          // オーディオデバイスが開かれ prepare 済み
    std::uint64_t audioCallbackCount = 0;    // 音声コールバック回数（単調増加・RT が加算）
    double sampleRateHz = 0.0;
    int blockSamples = 0;
    bool irLoaded = false;
    int irLength = 0;
    std::string deviceType;
    std::string deviceName;
};

} // namespace convo::isr
