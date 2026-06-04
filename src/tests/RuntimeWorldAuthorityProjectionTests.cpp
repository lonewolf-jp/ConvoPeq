#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <array>

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

[[nodiscard]] bool hasCodeExtension(const std::filesystem::path& path)
{
    const auto ext = path.extension().string();
    return ext == ".h"
        || ext == ".hpp"
        || ext == ".cpp"
        || ext == ".cxx"
        || ext == ".cc";
}

[[nodiscard]] bool isOpaqueContractScanTarget(const std::filesystem::path& relativePath)
{
    const auto normalized = relativePath.generic_string();
    if (normalized.rfind("src/tests/", 0) == 0)
        return false;
    if (normalized == "src/audioengine/AudioEngine.h")
        return false;
    return hasCodeExtension(relativePath);
}

[[nodiscard]] bool hasForbiddenRuntimeReadHandleFieldAccess(const std::string& text)
{
    static constexpr std::array<const char*, 8> forbiddenPatterns {
        "runtimeReadHandle.runtimeWorld",
        "runtimeReadHandleRef.runtimeWorld",
        "runtimeReadHandle.runtimePublish",
        "runtimeReadHandleRef.runtimePublish",
        "runtimeReadHandle.observedSnapshot",
        "runtimeReadHandleRef.observedSnapshot",
        "readView.runtimeWorld",
        "readView.observedSnapshot"
    };

    for (const auto* pattern : forbiddenPatterns)
    {
        if (contains(text, pattern))
            return true;
    }
    return false;
}

[[nodiscard]] bool testRuntimeReadHandleOpaqueContract()
{
    namespace fs = std::filesystem;
    const fs::path srcRoot = resolveRepoRelativePath("src");

    for (const auto& entry : fs::recursive_directory_iterator(srcRoot))
    {
        if (!entry.is_regular_file())
            continue;

        const fs::path relative = fs::relative(entry.path(), srcRoot.parent_path());
        if (!isOpaqueContractScanTarget(relative))
            continue;

        const auto text = readAllText(relative.generic_string().c_str());
        if (hasForbiddenRuntimeReadHandleFieldAccess(text))
            return false;
    }

    return true;
}

[[nodiscard]] bool testRuntimeWorldAuthorityProjectionContract()
{
    const auto header = readAllText("src/audioengine/AudioEngine.h");
    const auto builder = readAllText("src/audioengine/RuntimeBuilder.cpp");
    const auto schema = readAllText("src/audioengine/ISRRuntimeSemanticSchema.h");
    const auto learner = readAllText("src/NoiseShaperLearner.cpp");
    const auto timer = readAllText("src/audioengine/AudioEngine.Timer.cpp");
    const auto spectrum = readAllText("src/SpectrumAnalyzerComponent.cpp");
    const auto prepareToPlay = readAllText("src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp");
    const auto snapshot = readAllText("src/audioengine/AudioEngine.Processing.Snapshot.cpp");
    const auto audioBlock = readAllText("src/audioengine/AudioEngine.Processing.AudioBlock.cpp");
    const auto blockDouble = readAllText("src/audioengine/AudioEngine.Processing.BlockDouble.cpp");
    const auto commit = readAllText("src/audioengine/AudioEngine.Commit.cpp");

    if (!contains(schema, "double saturationAmount = 0.0;"))
        return false;
    if (!contains(schema, "double inputHeadroomGain = 1.0;"))
        return false;
    if (!contains(schema, "double outputMakeupGain = 1.0;"))
        return false;
    if (!contains(schema, "double convolverInputTrimGain = 1.0;"))
        return false;

    if (!contains(builder, "worldOwner->automation.saturationAmount = engineState.saturationAmount;"))
        return false;
    if (!contains(builder, "worldOwner->automation.inputHeadroomGain = engineState.inputHeadroomGain;"))
        return false;
    if (!contains(builder, "worldOwner->automation.outputMakeupGain = engineState.outputMakeupGain;"))
        return false;
    if (!contains(builder, "worldOwner->automation.convolverInputTrimGain = engineState.convolverInputTrimGain;"))
        return false;
    if (!contains(builder, "worldOwner->coefficient.adaptiveCoeffBankIndex = engineState.adaptiveCoeffBankIndex;"))
        return false;
    if (!contains(builder, "worldOwner->coefficient.adaptiveCoeffGeneration = engineState.adaptiveCoeffGeneration;"))
        return false;
    if (contains(header, "graph.oversamplingFactor = std::max(1, consumeAtomic(manualOversamplingFactor, std::memory_order_acquire));"))
        return false;
    if (contains(header, "graph.ditherBitDepth = consumeAtomic(ditherBitDepth, std::memory_order_acquire);"))
        return false;
    if (contains(header, "graph.noiseShaperType = static_cast<int>(consumeAtomic(noiseShaperType, std::memory_order_acquire));"))
        return false;
    if (contains(header, "graph.sampleRate = consumeAtomic(currentSampleRate, std::memory_order_acquire);"))
        return false;

    if (!contains(header, "snapshot.saturationAmount = static_cast<float>(world->automation.saturationAmount);"))
        return false;
    if (!contains(header, "snapshot.inputHeadroomGain = world->automation.inputHeadroomGain;"))
        return false;
    if (!contains(header, "snapshot.outputMakeupGain = world->automation.outputMakeupGain;"))
        return false;
    if (!contains(header, "snapshot.convolverInputTrimGain = world->automation.convolverInputTrimGain;"))
        return false;
    if (!contains(header, "snapshot.adaptiveCoeffBankIndex = world->coefficient.adaptiveCoeffBankIndex;"))
        return false;
    if (!contains(header, "snapshot.adaptiveCoeffGeneration = world->coefficient.adaptiveCoeffGeneration;"))
        return false;
    if (!contains(header, "static constexpr std::array<convo::isr::RuntimeAuthorityInventoryEntry, 9> kRuntimeReadAuthorityInventory"))
        return false;
    if (!contains(header, "{\"automation\", convo::isr::RuntimeAuthorityClass::Derived}"))
        return false;
    if (!contains(header, "{\"coefficient\", convo::isr::RuntimeAuthorityClass::Derived}"))
        return false;
    if (contains(header, "convo::TransitionState transition {}"))
        return false;
    if (contains(header, "struct RuntimePublishView"))
        return false;
    if (contains(header, "world != nullptr ? world->engine.transition : convo::TransitionState{}"))
        return false;

    if (contains(header, "world->graph.saturationAmount"))
        return false;
    if (contains(header, "world->graph.inputHeadroomGain"))
        return false;
    if (contains(header, "world->graph.outputMakeupGain"))
        return false;
    if (contains(header, "world->graph.convolverInputTrimGain"))
        return false;

    if (!contains(header, "static inline const RuntimePublishWorld* getRuntimeWorldFromReadHandle(const RuntimeReadHandle& runtimeReadHandle) noexcept"))
        return false;
    if (!contains(header, "const RuntimePublishWorld* runtimeWorldPtr() const noexcept"))
        return false;
    if (!contains(header, "const convo::GlobalSnapshot* observedSnapshotPtr() const noexcept"))
        return false;
    if (!contains(header, "friend class AudioEngine;"))
        return false;
    if (contains(header, "convo::ObservedRuntime observedSnapshot;"))
        return false;
    if (contains(header, "const RuntimePublishWorld* runtimeWorld = nullptr;"))
        return false;
    if (contains(header, "return runtimeReadHandle.runtimeWorld;"))
        return false;
    if (contains(header, "return runtimeReadHandle.observedSnapshot.get();"))
        return false;
    if (!contains(header, "static inline double getRuntimeSampleRateHzFromWorld(const RuntimeReadHandle& runtimeReadHandle"))
        return false;
    if (!contains(header, "return static_cast<DSPCore*>(runtimeWorld->engine.fading);"))
        return false;
    if (!contains(header, "? static_cast<DSPCore*>(runtimeWorld->engine.current)"))
        return false;
    if (!contains(header, "// Bootstrap World guarantees non-null (#3.2.5)"))
        return false;
    if (!contains(header, "&& (consumeAtomic(lastCommittedRuntimeGeneration_, std::memory_order_acquire) == 0);"))
        return false;
    if (!contains(header, ".allowRetireFallback = false"))
        return false;
    if (!contains(header, ".allowAdaptiveBankIndexFallback = allowInitialAtomicFallback"))
        return false;
    if (contains(header, ".allowAdaptiveGenerationFallback"))
        return false;
    if (!contains(header, "runtime.adaptiveCoeffBankIndex = (runtimeWorld != nullptr)"))
        return false;
    if (!contains(header, "runtime.adaptiveCoeffGeneration = (runtimeWorld != nullptr)"))
        return false;
    if (!contains(header, ": 0u;"))
        return false;
    if (!contains(header, "runtime.retireBacklog = (runtimeWorld != nullptr)"))
        return false;
    if (!contains(header, "? runtimeWorld->retire.retireBacklog"))
        return false;
    if (!contains(header, "runtime.deferredResidency = (runtimeWorld != nullptr)"))
        return false;
    if (!contains(header, "? runtimeWorld->retire.deferredResidency"))
        return false;
    if (contains(header, "runtime.retireBacklog = consumeAtomic(retireQueueDepth_, std::memory_order_acquire);"))
        return false;
    if (contains(header, "runtime.deferredResidency = consumeAtomic(fallbackQueueDepth_, std::memory_order_acquire);"))
        return false;
    if (!contains(header, "jassert(runtimeWorld != nullptr"))
        return false;
    if (!contains(header, "|| fallbackPolicy.allowTransitionFallback"))
        return false;
    if (!contains(header, "if (runtimeWorld == nullptr"))
        return false;
    if (!contains(header, "&& !(fallbackPolicy.allowTransitionFallback"))
        return false;
    if (!contains(header, "convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);"))
        return false;
    if (!contains(header, "convo::publishAtomic(observeMonotonicRollbackRequested_, true, std::memory_order_release);"))
        return false;
    if (contains(learner, "runtimeReadHandle.runtimePublish.transition.current"))
        return false;
    if (!contains(learner, "resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle)"))
        return false;
    if (contains(timer, "runtimePublishView.transition"))
        return false;
    if (!contains(timer, "runtimeWorld->execution.transitionPolicy"))
        return false;
    if (!contains(timer, "runtimeWorld->overlap.fadeTimeSec"))
        return false;
    if (!contains(timer, "runtimeWorld->execution.latencyCompensationSamples"))
        return false;
    if (!contains(spectrum, "const auto runtimeReadHandle = engine.readControlRuntimeHandle();"))
        return false;
    if (!contains(spectrum, "const auto* snap = AudioEngine::getRuntimeSnapshotFromReadHandle(runtimeReadHandle);"))
        return false;
    if (contains(spectrum, "runtimeReadHandle.runtimeWorld"))
        return false;
    if (contains(prepareToPlay, "runtimeReadHandle.runtimeWorld"))
        return false;
    if (contains(snapshot, "runtimeReadHandle.runtimeWorld"))
        return false;
    if (contains(audioBlock, "runtimeReadHandle.runtimeWorld") || contains(audioBlock, "runtimeReadHandleRef.runtimeWorld"))
        return false;
    if (contains(audioBlock, "runtimeReadHandle.runtimePublish") || contains(audioBlock, "runtimeReadHandleRef.runtimePublish"))
        return false;
    if (contains(blockDouble, "runtimeReadHandle.runtimeWorld") || contains(blockDouble, "runtimeReadHandleRef.runtimeWorld"))
        return false;
    if (contains(blockDouble, "runtimeReadHandle.runtimePublish") || contains(blockDouble, "runtimeReadHandleRef.runtimePublish"))
        return false;
    if (contains(snapshot, "runtimeReadHandle.runtimePublish") || contains(snapshot, "runtimeReadHandleRef.runtimePublish"))
        return false;
    if (contains(commit, "runtimeReadHandle.runtimeWorld"))
        return false;
    if (!contains(prepareToPlay, "getTransitionPolicyFromRuntimeWorld(runtimeReadHandle"))
        return false;
    if (!contains(commit, "hasFadingRuntimeInWorld(runtimeReadHandle)"))
        return false;
    if (!contains(audioBlock, "getRuntimeSampleRateHzFromWorld(runtimeReadHandleRef"))
        return false;
    if (!contains(blockDouble, "getRuntimeSampleRateHzFromWorld(runtimeReadHandleRef"))
        return false;
    if (!contains(snapshot, "getRuntimeSampleRateHzFromWorld(runtimeReadHandle"))
        return false;

    return true;
}

} // namespace

int main()
{
    if (!testRuntimeWorldAuthorityProjectionContract())
        throw std::runtime_error("runtime world authority projection contract failed");

    if (!testRuntimeReadHandleOpaqueContract())
        throw std::runtime_error("runtime read handle opaque contract failed");

    return 0;
}
