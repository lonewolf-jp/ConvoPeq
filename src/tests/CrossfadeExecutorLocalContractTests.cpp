#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>

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

[[nodiscard]] bool testCrossfadeExecutorLocalContract()
{
    const auto commit = readAllText("src/audioengine/AudioEngine.Commit.cpp");
    const auto timer = readAllText("src/audioengine/AudioEngine.Timer.cpp");
    const auto header = readAllText("src/audioengine/AudioEngine.h");

    // PR-06: CrossfadePreparedSnapshot must not be semantic source for branch/rebuild/publish/retire/execution decisions.
    if (contains(commit, "consumeCrossfadePreparedSnapshot("))
        return false;
    if (contains(commit, "preparedCrossfade"))
        return false;

    if (contains(timer, "consumeCrossfadePreparedSnapshot("))
        return false;
    if (contains(timer, "preparedCrossfade"))
        return false;

    // makeEngineRuntimeState must source semantic values from RuntimeWorld/atomics, not prepared snapshot cache.
    if (!contains(header, "inline convo::EngineRuntime makeEngineRuntimeState"))
        return false;
    if (contains(header, "refreshCrossfadePreparedSnapshotFromAtomics();\n        const auto prepared = consumeCrossfadePreparedSnapshot();"))
        return false;

    return true;
}

} // namespace

int main()
{
    if (!testCrossfadeExecutorLocalContract())
        throw std::runtime_error("crossfade executor-local contract failed");

    return 0;
}
