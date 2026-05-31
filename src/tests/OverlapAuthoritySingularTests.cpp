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

[[nodiscard]] bool testOverlapAuthoritySingularContract()
{
    const auto header = readAllText("src/audioengine/AudioEngine.h");
    const auto audioBlock = readAllText("src/audioengine/AudioEngine.Processing.AudioBlock.cpp");
    const auto blockDouble = readAllText("src/audioengine/AudioEngine.Processing.BlockDouble.cpp");

    if (!contains(header, "makeCrossfadePreparedSnapshotFromWorld"))
        return false;

    if (!contains(audioBlock, "authority.preparedCrossfade"))
        return false;
    if (!contains(blockDouble, "authority.preparedCrossfade"))
        return false;

    if (contains(audioBlock, "crossfadePreparedSnapshot"))
        return false;
    if (contains(blockDouble, "crossfadePreparedSnapshot"))
        return false;

    return true;
}

} // namespace

int main()
{
    if (!testOverlapAuthoritySingularContract())
        throw std::runtime_error("overlap authority singular contract failed");

    return 0;
}
