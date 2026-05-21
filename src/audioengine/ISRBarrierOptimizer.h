#pragma once
#include <cstdint>

namespace convo::isr {

enum class BuildMode {
    Release,
    Debug,
    CI
};

class BarrierOptimizer {
public:
    void setBuildMode(BuildMode mode);
    void optimizeBarriers();
private:
    BuildMode mode_ = BuildMode::Release;
};

} // namespace convo::isr
