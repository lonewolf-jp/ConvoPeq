#pragma once
#include <vector>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include "ISRClosure.h"

namespace convo::isr {

class ClosureGraphWalker {
public:
    bool validateGraph(const PayloadClosureDescriptor& closure, std::string_view validationError = {});
    void emitClosureArtifact(const PayloadClosureDescriptor& closure,
                             bool valid,
                             std::string_view validationError = {},
                             const std::filesystem::path& outputPath = std::filesystem::path{}) const;
};

} // namespace convo::isr
