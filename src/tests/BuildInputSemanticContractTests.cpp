#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <iostream>

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

[[nodiscard]] bool testBuildInputSemanticContract()
{
    const auto requireContains = [](const std::string& haystack, const std::string& needle, const char* label) -> bool
    {
        if (contains(haystack, needle))
            return true;

        std::cerr << "[BuildInputSemanticContract] missing " << label << ": " << needle << '\n';
        return false;
    };

    const auto buildTypes = readAllText("src/audioengine/RuntimeBuildTypes.h");
    const auto rebuildDispatch = readAllText("src/audioengine/AudioEngine.RebuildDispatch.cpp");
    const auto commit = readAllText("src/audioengine/AudioEngine.Commit.cpp");
    const auto audioHeader = readAllText("src/audioengine/AudioEngine.h");
    const auto runtimeBuilderHeader = readAllText("src/audioengine/RuntimeBuilder.h");
    const auto runtimeBuilderCpp = readAllText("src/audioengine/RuntimeBuilder.cpp");
    const auto admissionHeader = readAllText("src/audioengine/PublicationAdmission.h");
    const auto snapshot = readAllText("src/audioengine/AudioEngine.Snapshot.cpp");
    const auto init = readAllText("src/audioengine/AudioEngine.Init.cpp");

    for (const auto& requiredField : {
             std::string("int processingOrder = 0;"),
             std::string("bool eqBypassed = false;"),
             std::string("bool convBypassed = false;"),
             std::string("bool softClipEnabled = false;"),
             std::string("double saturationAmount = 0.0;"),
             std::string("double inputHeadroomGain = 1.0;"),
             std::string("double outputMakeupGain = 1.0;"),
             std::string("double convolverInputTrimGain = 1.0;") })
    {
        if (!contains(buildTypes, requiredField)) {
            std::cerr << "[FAIL] Phase1 missing: " << requiredField << '\n';
            return false;
        }
    }

    for (const auto& compatibilityField : {
             std::string("&& snapshot.buildInput.processingOrder == other.buildInput.processingOrder"),
             std::string("&& snapshot.buildInput.eqBypassed == other.buildInput.eqBypassed"),
             std::string("&& snapshot.buildInput.convBypassed == other.buildInput.convBypassed"),
             std::string("&& snapshot.buildInput.softClipEnabled == other.buildInput.softClipEnabled"),
             std::string("&& snapshot.buildInput.saturationAmount == other.buildInput.saturationAmount"),
             std::string("&& snapshot.buildInput.inputHeadroomGain == other.buildInput.inputHeadroomGain"),
             std::string("&& snapshot.buildInput.outputMakeupGain == other.buildInput.outputMakeupGain"),
             std::string("&& snapshot.buildInput.convolverInputTrimGain == other.buildInput.convolverInputTrimGain") })
    {
        if (!contains(buildTypes, compatibilityField)) {
            std::cerr << "[FAIL] Phase2 missing: " << compatibilityField << '\n';
            return false;
        }
    }

    for (const auto& requiredAssignment : {
             std::string("task.buildInput.processingOrder = static_cast<int>(paramSnapshot.processingOrder);"),
             std::string("task.buildInput.eqBypassed = paramSnapshot.eqBypassed;"),
             std::string("task.buildInput.convBypassed = paramSnapshot.convBypassed;"),
             std::string("task.buildInput.softClipEnabled = paramSnapshot.softClipEnabled;"),
             std::string("task.buildInput.saturationAmount = paramSnapshot.saturationAmount;"),
             std::string("task.buildInput.inputHeadroomGain = paramSnapshot.inputHeadroomGain;"),
             std::string("task.buildInput.outputMakeupGain = paramSnapshot.outputMakeupGain;"),
             std::string("task.buildInput.convolverInputTrimGain = paramSnapshot.convolverInputTrimGain;") })
    {
        if (!contains(rebuildDispatch, requiredAssignment)) {
            std::cerr << "[FAIL] Phase3 missing: " << requiredAssignment << '\n';
            return false;
        }
    }

    for (const auto& requiredSnapshotPlumbing : {
             std::string("enqueuePublicationIntentForRuntimeCommit(dspToCommit, task.generation, task.runtimeBuildSnapshot);") })
    {
        const bool found = contains(commit, requiredSnapshotPlumbing)
            || contains(rebuildDispatch, requiredSnapshotPlumbing);
        if (!found)
        {
            std::cerr << "[BuildInputSemanticContract] missing commit/rebuild dispatch: "
                      << requiredSnapshotPlumbing << '\n';
            return false;
        }
    }

    if (!requireContains(audioHeader, "void enqueuePublicationIntentForRuntimeCommit(DSPCore* newDSP, int generation, const convo::RuntimeBuildSnapshot& sealedSnapshot);", "audio header enqueuePublicationIntentForRuntimeCommit"))
        return false;
    // [P1 Phase1-B] appendPublicationIntentForCommitProducer/Consumer removed
    if (!requireContains(runtimeBuilderHeader, "const convo::RuntimeBuildSnapshot* sealedSnapshot = nullptr", "runtime builder header sealed snapshot"))
        return false;
    if (!requireContains(admissionHeader, "RuntimeBuildSnapshot sealedSnapshot;", "admission header sealed snapshot"))
        return false;
    if (!requireContains(runtimeBuilderCpp, "if (sealedSnapshot != nullptr)", "runtime builder cpp sealed branch"))
        return false;
    // ★ work56 Phase1: sealedSnapshot ブロック内でローカルエイリアス使用
    if (!requireContains(runtimeBuilderCpp, "const auto& sealedBuildInput = sealedSnapshot->buildInput;", "runtime builder cpp sealed build input alias"))
        return false;
    // routing - from spec.routing (ProcessingPart 経由、Orchestrator が設定)
    // ★ P0: sealedBuildInput からの直接読み取り → ProcessingPart 経由に移行
    if (!requireContains(runtimeBuilderCpp, "worldOwner->routing.processingOrder = spec.routing.processingOrder;", "runtime builder cpp spec routing processingOrder"))
        return false;
    if (!requireContains(runtimeBuilderCpp, "worldOwner->routing.eqBypassed = spec.routing.eqBypassed;", "runtime builder cpp spec routing eqBypassed"))
        return false;
    if (!requireContains(runtimeBuilderCpp, "worldOwner->routing.convBypassed = spec.routing.convBypassed;", "runtime builder cpp spec routing convBypassed"))
        return false;
    // automation - from spec.processing (ProcessingPart 経由)
    // ★ P0: sealedBuildInput → spec.processing に移行（Specification Completeness 原則）
    if (!requireContains(runtimeBuilderCpp, "worldOwner->automation.softClipEnabled = spec.processing.softClipEnabled;", "runtime builder cpp spec automation soft clip"))
        return false;
    if (!requireContains(runtimeBuilderCpp, "worldOwner->automation.saturationAmount = spec.processing.saturationAmount;", "runtime builder cpp spec automation saturation"))
        return false;
    if (!requireContains(runtimeBuilderCpp, "worldOwner->automation.inputHeadroomGain = spec.processing.inputHeadroomGain;", "runtime builder cpp spec automation input gain"))
        return false;
    if (!requireContains(runtimeBuilderCpp, "worldOwner->automation.outputMakeupGain = spec.processing.outputMakeupGain;", "runtime builder cpp spec automation output gain"))
        return false;
    if (!requireContains(runtimeBuilderCpp, "worldOwner->automation.convolverInputTrimGain = spec.processing.convolverInputTrimGain;", "runtime builder cpp spec automation convolver trim"))
        return false;
    // resource/timing - from sealedBuildInput (sealedSnapshot path, oversamplingFactor 除く)
    if (!requireContains(runtimeBuilderCpp, "worldOwner->resource.ditherBitDepth = sealedBuildInput.ditherBitDepth;", "runtime builder cpp sealed resource dither"))
        return false;
    if (!requireContains(runtimeBuilderCpp, "worldOwner->resource.noiseShaperType = sealedBuildInput.noiseShaperType;", "runtime builder cpp sealed resource noise shaper"))
        return false;
    if (!requireContains(runtimeBuilderCpp, "worldOwner->timing.sampleRateHz = sealedBuildInput.sampleRate;", "runtime builder cpp sealed timing sample rate"))
        return false;
    // ★ P0: semanticHash.payloadHash 計算は hashBuildInput の未使用化により削除済み
    //   （セマンティックハッシュは semanticHash の各フィールドで個別に計算）
    //   代替: セマンティックハッシュの完全性は executionHash/routingHash で担保
    if (!requireContains(rebuildDispatch,
                         "runtimeBuilder.build(task.runtimeBuildSnapshot.buildInput,",
                         "rebuild dispatch sealed convolver snapshot call"))
        return false;
    if (!requireContains(rebuildDispatch,
                         "task.convolverBuildSnapshot);",
                         "rebuild dispatch sealed convolver snapshot arg"))
        return false;
    if (!requireContains(runtimeBuilderHeader,
                         "BuildResult build(const BuildInput& in,",
                         "runtime builder header sealed convolver snapshot signature"))
        return false;
    if (!requireContains(runtimeBuilderHeader,
                         "const ConvolverProcessor::BuildSnapshot& convolverBuildSnapshot) noexcept;",
                         "runtime builder header sealed convolver snapshot arg"))
        return false;
    if (!requireContains(runtimeBuilderCpp,
                         "runtime->convolverRt().applyBuildSnapshot(convolverBuildSnapshot);",
                         "runtime builder cpp applies sealed convolver snapshot"))
        return false;
    if (contains(runtimeBuilderCpp, "engine.applyCurrentConvolverSnapshotToRuntime(*runtime);"))
        return false;
    if (contains(audioHeader, "applyCurrentConvolverSnapshotToRuntime"))
        return false;

    for (const auto& requiredCapture : {
             std::string("snapshot.processingOrder = convo::consumeAtomic(engine.currentProcessingOrder, std::memory_order_acquire);"),
             std::string("snapshot.eqBypassed = convo::consumeAtomic(engine.eqBypassRequested, std::memory_order_acquire);"),
             std::string("snapshot.convBypassed = convo::consumeAtomic(engine.convBypassRequested, std::memory_order_acquire);"),
             std::string("snapshot.softClipEnabled = convo::consumeAtomic(engine.softClipEnabled, std::memory_order_acquire);"),
             std::string("snapshot.inputHeadroomGain = convo::consumeAtomic(engine.inputHeadroomGain, std::memory_order_acquire);"),
             std::string("snapshot.outputMakeupGain = convo::consumeAtomic(engine.outputMakeupGain, std::memory_order_acquire);"),
             std::string("snapshot.convolverInputTrimGain = convo::consumeAtomic(engine.convolverInputTrimGain, std::memory_order_acquire);") })
    {
        if (!contains(rebuildDispatch, requiredCapture))
            return false;
    }

    for (const auto& requiredHash : {
             std::string("mixHash(static_cast<std::uint64_t>(snapshot.buildInput.processingOrder));"),
             std::string("mixHash(static_cast<std::uint64_t>(snapshot.buildInput.eqBypassed));"),
             std::string("mixHash(static_cast<std::uint64_t>(snapshot.buildInput.convBypassed));"),
             std::string("mixHash(static_cast<std::uint64_t>(snapshot.buildInput.softClipEnabled));"),
             std::string("mixHash(std::bit_cast<std::uint64_t>(snapshot.buildInput.saturationAmount));"),
             std::string("mixHash(std::bit_cast<std::uint64_t>(snapshot.buildInput.inputHeadroomGain));"),
             std::string("mixHash(std::bit_cast<std::uint64_t>(snapshot.buildInput.outputMakeupGain));"),
             std::string("mixHash(std::bit_cast<std::uint64_t>(snapshot.buildInput.convolverInputTrimGain));") })
    {
        if (!contains(rebuildDispatch, requiredHash))
            return false;
    }

    for (const auto& requiredSnapshotSource : {
             std::string("const double inputHeadroomGainValue = convo::consumeAtomic(inputHeadroomGain, std::memory_order_acquire);"),
             std::string("const double outputMakeupGainValue = convo::consumeAtomic(outputMakeupGain, std::memory_order_acquire);"),
             std::string("const double convInputTrimGainValue = convo::consumeAtomic(convolverInputTrimGain, std::memory_order_acquire);"),
             std::string("const bool convBypass = convo::consumeAtomic(convBypassRequested, std::memory_order_acquire);"),
             std::string("const bool eqBypass = convo::consumeAtomic(eqBypassRequested, std::memory_order_acquire);"),
             std::string("const bool softClip = convo::consumeAtomic(softClipEnabled, std::memory_order_acquire);"),
             std::string("const float satAmount = convo::consumeAtomic(saturationAmount, std::memory_order_acquire);"),
             std::string("const convo::ProcessingOrder order = convo::consumeAtomic(currentProcessingOrder, std::memory_order_acquire);") })
    {
        if (!contains(snapshot, requiredSnapshotSource))
            return false;
    }

    for (const auto& forbiddenShadow : {
             std::string("m_currentInputHeadroomDb"),
             std::string("m_currentOutputMakeupDb"),
             std::string("m_currentConvInputTrimDb"),
             std::string("m_currentEqBypass"),
             std::string("m_currentConvBypass"),
             std::string("m_currentSoftClipEnabled"),
             std::string("m_currentSaturationAmount"),
             std::string("m_currentProcessingOrder") })
    {
        if (contains(snapshot, forbiddenShadow))
            return false;
    }

    for (const auto& requiredDebounceSource : {
             std::string("consumeAtomic(eqBypassRequested, std::memory_order_acquire)"),
             std::string("consumeAtomic(convBypassRequested, std::memory_order_acquire)"),
             std::string("consumeAtomic(currentProcessingOrder, std::memory_order_acquire)"),
             std::string("consumeAtomic(softClipEnabled, std::memory_order_acquire)"),
             std::string("consumeAtomic(inputHeadroomGain, std::memory_order_acquire)"),
             std::string("consumeAtomic(outputMakeupGain, std::memory_order_acquire)"),
             std::string("consumeAtomic(convolverInputTrimGain, std::memory_order_acquire)"),
             std::string("consumeAtomic(saturationAmount, std::memory_order_acquire)") })
    {
        if (!contains(audioHeader, requiredDebounceSource))
            return false;
    }

    for (const auto& forbiddenInitShadow : {
             std::string("m_currentEqBypass"),
             std::string("m_currentConvBypass"),
             std::string("m_currentProcessingOrder"),
             std::string("m_currentSoftClipEnabled"),
             std::string("m_currentInputHeadroomGain"),
             std::string("m_currentOutputMakeupGain"),
             std::string("m_currentConvInputTrimGain"),
             std::string("m_currentSaturationAmount") })
    {
        if (contains(init, forbiddenInitShadow))
            return false;
    }

    return true;
}

} // namespace

int main()
{
    std::cerr << "[START] test begin" << std::endl;
    const bool result = testBuildInputSemanticContract();
    std::cerr << "[END] result=" << (result ? "true" : "false") << std::endl;
    if (!result)
    {
        std::cerr << "build input semantic contract failed" << std::endl;
        return 1;
    }
    return 0;
}
