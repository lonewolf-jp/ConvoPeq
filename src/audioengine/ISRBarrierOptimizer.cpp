#include "ISRBarrierOptimizer.h"

namespace convo::isr {

void BarrierOptimizer::setBuildMode(BuildMode mode) {
    mode_ = mode;
}

void BarrierOptimizer::optimizeBarriers() {
    switch (mode_) {
    case BuildMode::Release:
        break;
    case BuildMode::Debug:
        break;
    case BuildMode::CI:
        break;
    }
}

} // namespace convo::isr
