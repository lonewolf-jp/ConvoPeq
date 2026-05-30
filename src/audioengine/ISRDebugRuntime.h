#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "ISRRuntimeSemanticSchema.h"
#include "ISRHB.h"

namespace convo::isr {

class DebugRuntime {
public:
    void runAtomicDotCallScan();
    void validateOwnershipClosure();
    void emitCIArtifacts();
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
