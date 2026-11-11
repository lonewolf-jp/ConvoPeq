#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "ISRRuntimeSemanticSchema.h"
#include "ISRHB.h"

namespace convo::isr {

// ★ B-11: 実測証跡モード。Release ビルドでも HB 証跡と違反レポートを発行させる。
//   既定は false（本番の Release では従来どおり発行しない）。
void setForceEvidenceEmission(bool enabled) noexcept;
bool isForceEvidenceEmission() noexcept;

class DebugRuntime {
public:
    void runAtomicDotCallScan();
    void validateOwnershipClosure();
    void emitCIArtifacts();
    void emitFinalViolationReport();   // ★ A(D1): 最終 drain 後の状態で違反レポートを更新
    void emitHBTrace();
    void recordHBEdge(std::uint32_t from,
                      std::uint32_t to,
                      std::uint64_t fromEpoch,
                      std::uint64_t toEpoch,
                      int memoryOrder) noexcept;
    void recordShadowCompareObservation(std::uint64_t sequenceId,
                                        const RuntimeSemanticHash& hash) noexcept;
    void emitShadowCompareCadenceReport() const;
    [[nodiscard]] std::uint64_t monotonicViolationCount() const noexcept;
    [[nodiscard]] std::uint64_t escalationCount() const noexcept;

private:
    HBTraceRuntime hbTraceRuntime_;
    // ★ B-11: HB 違反レポート（status/violations/scenarioResults）の実測生成器。
    //   従来は未接続で、エクスポータが status:"ok" の定数を書いていた。
    mutable HBVerifierRuntime hbVerifier_;
    bool hasPreviousShadowCompare_{false};
    std::uint64_t lastSequenceId_{0};
    RuntimeSemanticHash lastSemanticHash_{};
    std::uint64_t lastObservationTimeNs_{0};
    std::uint64_t totalObservations_{0};
    std::uint64_t mismatchCount_{0};
    std::uint64_t monotonicViolationCount_{0};
    std::uint64_t cadenceViolationCount_{0};
    std::uint64_t escalationCount_{0};
    std::uint32_t burstMismatchCount_{0};
};

} // namespace convo::isr
