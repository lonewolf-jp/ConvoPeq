#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

[[nodiscard]] std::string resolveRepoRelativePath(const char* relativePath)
{
    namespace fs = std::filesystem;
    const fs::path p(relativePath);

    for (const auto& prefix : { fs::path("."), fs::path(".."), fs::path("../.."), fs::path("../../..") })
    {
        const fs::path candidate = prefix / p;
        if (fs::exists(candidate))
            return candidate.string();
    }

    throw std::runtime_error(std::string("failed to resolve repository-relative path: ") + relativePath);
}

[[nodiscard]] std::string readAllText(const char* path)
{
    const std::string resolved = resolveRepoRelativePath(path);
    std::ifstream in(resolved, std::ios::in | std::ios::binary);
    if (!in)
        throw std::runtime_error(std::string("failed to open file: ") + path);

    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

[[nodiscard]] bool contains(const std::string& haystack, const std::string& needle)
{
    return haystack.find(needle) != std::string::npos;
}

[[nodiscard]] bool testRebuildAdmissionRegression()
{
    const auto rebuildDispatch = readAllText("src/audioengine/AudioEngine.RebuildDispatch.cpp");

    const auto requireContains = [](const std::string& haystack, const std::string& needle, const char* label) -> bool
    {
        if (contains(haystack, needle))
            return true;

        std::cerr << "[RebuildAdmissionRegression] missing " << label << ": " << needle << '\n';
        return false;
    };

    if (!requireContains(rebuildDispatch,
                         "const bool rebuildOutstanding = queuedGenerationSnapshot > committedGenerationSnapshot;",
                         "rebuild outstanding guard"))
        return false;

    if (!requireContains(rebuildDispatch,
                         "if (!rebuildOutstanding)",
                         "pending intent reset condition when outstanding queue is empty"))
        return false;

    if (!requireContains(rebuildDispatch,
                         "rebuildAdmissionPendingIntent_.valid = false;",
                         "pending intent reset when outstanding queue is empty"))
        return false;

    if (!requireContains(rebuildDispatch,
                         "sameAsPendingWouldMerge = rebuildOutstanding",
                         "merge gate tied to outstanding queue"))
        return false;

    if (!requireContains(rebuildDispatch,
                         "if (sameAsPendingWouldMerge",
                         "latest-wins merge branch condition"))
        return false;

    if (!requireContains(rebuildDispatch,
                         "&& collapsePolicy == RebuildTelemetryPolicy::Replaceable",
                         "latest-wins merge branch policy gate"))
        return false;

    if (!requireContains(rebuildDispatch,
                         "RebuildTelemetryReason::SameAsPendingWouldMerge",
                         "same-as-pending telemetry reason"))
        return false;

    return true;
}

} // namespace

int main()
{
    if (!testRebuildAdmissionRegression())
        throw std::runtime_error("Rebuild admission regression contract failed");

    return 0;
}
