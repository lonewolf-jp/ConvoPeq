#include "ISRClosureGraphWalker.h"
#include <chrono>
#include <filesystem>
#include <fstream>

namespace convo::isr {

namespace {
const char* buildModeName() noexcept
{
#if defined(NDEBUG)
    return "Release";
#elif defined(JUCE_DEBUG)
    return "Debug";
#else
    return "CI";
#endif
}
}

bool ClosureGraphWalker::validateGraph(const PayloadClosureDescriptor& closure, std::string_view validationError) {
    ClosureValidator validator;
    const bool valid = validator.validateClosureGraph(closure);
    emitClosureArtifact(closure,
                        valid,
                        validationError.empty() ? (valid ? "" : "closure graph validation failed") : validationError,
                        {});
    return valid;
}

void ClosureGraphWalker::emitClosureArtifact(const PayloadClosureDescriptor& closure,
                                             bool valid,
                                             std::string_view validationError,
                                             const std::filesystem::path& outputPath) const {
    const auto path = outputPath.empty()
        ? (std::filesystem::current_path() / "evidence" / "closure_graph.json")
        : outputPath;

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return;
    }

    const auto nowNs = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());

    bool descriptorCoverageComplete = true;
    for (const auto& node : closure.nodes) {
        const bool nodeCovered = node.kind != 0u
            && node.ownership != 0u
            && node.mutability != 0u
            && node.lifetime != 0u
            && node.hbDomain != 0u
            && node.authority != 0u
            && node.allocator != 0u;
        if (!nodeCovered) {
            descriptorCoverageComplete = false;
            break;
        }
    }

    file << "{\n";
    file << "  \"schema\": \"closure_graph_v1\",\n";
    file << "  \"buildMode\": \"" << buildModeName() << "\",\n";
    file << "  \"timestamp_ns\": " << nowNs << ",\n";
    file << "  \"status\": \"" << (valid ? "valid" : "invalid") << "\",\n";
    file << "  \"closureId\": " << closure.closureId << ",\n";
    file << "  \"nodeCount\": " << closure.nodes.size() << ",\n";
    file << "  \"edgeCount\": " << (closure.edges.size() / 2u) << ",\n";
    file << "  \"descriptorCoverageComplete\": " << (descriptorCoverageComplete ? "true" : "false") << ",\n";
    file << "  \"externalMutableDependencies\": " << closure.externalMutableDependencies << ",\n";
    file << "  \"validationErrors\": [";
    if (!valid) {
        file << "\"" << std::string(validationError) << "\"";
    }
    file << "]\n";
    file << "}\n";
}

} // namespace convo::isr
