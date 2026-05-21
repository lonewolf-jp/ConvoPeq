#pragma once
#include <string>
#include <vector>
#include <cstdint>
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

private:
    HBTraceRuntime hbTraceRuntime_;
};

} // namespace convo::isr
