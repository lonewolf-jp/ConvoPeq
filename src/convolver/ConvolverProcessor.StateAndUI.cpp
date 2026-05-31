#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "audioengine/AudioEngine.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "core/EpochDomain.h"
#include "AlignedAllocation.h"
#include "DftiHandle.h"

#include "audioengine/AtomicAccess.h"

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_STATE_UI)

using namespace ConvolverProcessorInternal;

namespace {

inline void hashCombineUInt64(std::uint64_t& seed, std::uint64_t value) noexcept
{
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

inline std::uint32_t floatBits(float value) noexcept
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::uint64_t computeBuildSnapshotFingerprint(const ConvolverProcessor::BuildSnapshot& snapshot) noexcept
{
    // NOTE:
    //   ここで算出する fingerprint は getStructuralHash() とは用途が異なる。
    //   BuildSnapshot の同一性確認/診断向けに、同期対象メタデータも含めて広くハッシュする。
    std::uint64_t hash = 0x9e3779b97f4a7c15ULL;

    // ---- コンボルバー同期対象パラメータ ----
    hashCombineUInt64(hash, static_cast<std::uint64_t>(floatBits(snapshot.mix)));
    hashCombineUInt64(hash, snapshot.bypassed ? 1ULL : 0ULL);
    hashCombineUInt64(hash, static_cast<std::uint64_t>(snapshot.phaseMode));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(snapshot.resamplingPhaseMode));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(floatBits(snapshot.smoothingTimeSec)));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(floatBits(snapshot.targetIRLengthSec)));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(floatBits(snapshot.mixedTransitionStartHz)));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(floatBits(snapshot.mixedTransitionEndHz)));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(floatBits(snapshot.mixedPreRingTau)));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(snapshot.rebuildDebounceMs));
    hashCombineUInt64(hash, snapshot.experimentalDirectHeadEnabled ? 1ULL : 0ULL);
    hashCombineUInt64(hash, static_cast<std::uint64_t>(snapshot.tailMode));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(floatBits(snapshot.tailStartSec)));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(floatBits(snapshot.tailStrength)));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(snapshot.tailL1L2Multiplier));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(snapshot.targetUpgradeFFTSize));
    hashCombineUInt64(hash, snapshot.enableProgressiveUpgrade ? 1ULL : 0ULL);
    hashCombineUInt64(hash, static_cast<std::uint64_t>(snapshot.maxCacheEntries));

    // ---- NUC フィルターモード ----
    hashCombineUInt64(hash, static_cast<std::uint64_t>(snapshot.nucHCMode));
    hashCombineUInt64(hash, static_cast<std::uint64_t>(snapshot.nucLCMode));

    // ---- メタデータ（snapshot 同一性確認用）----
    hashCombineUInt64(hash, static_cast<std::uint64_t>(snapshot.irLength));

    std::uint64_t scaleBits = 0;
    static_assert(sizeof(scaleBits) == sizeof(snapshot.currentIRScale));
    std::memcpy(&scaleBits, &snapshot.currentIRScale, sizeof(scaleBits));
    hashCombineUInt64(hash, scaleBits);

    const auto pathUtf8 = snapshot.irFile.getFullPathName().toRawUTF8();
    for (const char* ch = pathUtf8; ch != nullptr && *ch != '\0'; ++ch)
        hashCombineUInt64(hash, static_cast<std::uint64_t>(static_cast<unsigned char>(*ch)));

    const auto nameUtf8 = snapshot.irName.toRawUTF8();
    for (const char* ch = nameUtf8; ch != nullptr && *ch != '\0'; ++ch)
        hashCombineUInt64(hash, static_cast<std::uint64_t>(static_cast<unsigned char>(*ch)));

    return hash;
}

} // namespace

// ────────────────────────────────────────────────────────────────
// State Management and UI Updates
// ────────────────────────────────────────────────────────────────

namespace {
void applySmoothing(const float* magnitudes, float* smoothed, int numBins)
{
    if (magnitudes == nullptr || smoothed == nullptr || numBins <= 0) return;

    smoothed[0] = magnitudes[0];
    const float bandwidth = 1.0f / 6.0f;
    const float factor = std::pow(2.0f, bandwidth * 0.5f);

    for (int i = 1; i < numBins; ++i)
    {
        float sum = 0.0f;
        int count = 0;

        int startBin = static_cast<int>(static_cast<float>(i) / factor);
        int endBin = static_cast<int>(static_cast<float>(i) * factor);

        startBin = (std::max)(1, startBin);
        endBin = (std::min)(numBins - 1, endBin);

        for (int j = startBin; j <= endBin; ++j)
        {
            sum += magnitudes[j];
            count++;
        }

        if (count > 0)
            smoothed[i] = sum / static_cast<float>(count);
        else
            smoothed[i] = magnitudes[i];
    }
}
}

void ConvolverProcessor::copyPendingToSnapshotUnlocked(BuildSnapshot& snapshot) const noexcept
{
    // [更新ガイド]
    // - ここでの代入順は BuildSnapshot の定義順に揃える。
    // - 項目追加時は copySnapshotToPendingUnlocked() も同時更新する。
    // - fingerprint / structural hash の対象可否を同時に見直す。
    snapshot.mix = pendingOverride.mix;
    snapshot.bypassed = pendingOverride.bypassed;
    snapshot.phaseMode = pendingOverride.phaseMode;
    snapshot.resamplingPhaseMode = pendingOverride.resamplingPhaseMode;
    snapshot.smoothingTimeSec = pendingOverride.smoothingTimeSec;
    snapshot.targetIRLengthSec = pendingOverride.targetIRLengthSec;
    snapshot.autoDetectedIRLengthSec = pendingOverride.autoDetectedIRLengthSec;
    snapshot.irLengthManualOverride = pendingOverride.irLengthManualOverride;
    snapshot.mixedTransitionStartHz = pendingOverride.mixedTransitionStartHz;
    snapshot.mixedTransitionEndHz = pendingOverride.mixedTransitionEndHz;
    snapshot.mixedPreRingTau = pendingOverride.mixedPreRingTau;
    snapshot.rebuildDebounceMs = pendingOverride.rebuildDebounceMs;
    snapshot.experimentalDirectHeadEnabled = pendingOverride.experimentalDirectHeadEnabled;
    snapshot.tailMode = pendingOverride.tailMode;
    snapshot.tailStartSec = pendingOverride.tailStartSec;
    snapshot.tailStrength = pendingOverride.tailStrength;
    snapshot.tailL1L2Multiplier = pendingOverride.tailL1L2Multiplier;
    snapshot.targetUpgradeFFTSize = pendingOverride.targetUpgradeFFTSize;
    snapshot.enableProgressiveUpgrade = pendingOverride.enableProgressiveUpgrade;
    snapshot.maxCacheEntries = pendingOverride.maxCacheEntries;
    snapshot.nucHCMode = pendingOverride.nucHCMode;
    snapshot.nucLCMode = pendingOverride.nucLCMode;
}

void ConvolverProcessor::copySnapshotToPendingUnlocked(const BuildSnapshot& snapshot) noexcept
{
    // [更新ガイド]
    // - copyPendingToSnapshotUnlocked() と対称性を維持する。
    // - 片方向のみ更新すると snapshot 同期の欠落を招くため注意。
    pendingOverride.mix = juce::jlimit(MIX_MIN, MIX_MAX, snapshot.mix);
    pendingOverride.bypassed = snapshot.bypassed;
    pendingOverride.phaseMode = juce::jlimit(static_cast<int>(PhaseMode::AsIs),
                                             static_cast<int>(PhaseMode::Minimum),
                                             snapshot.phaseMode);
    pendingOverride.resamplingPhaseMode = juce::jlimit(static_cast<int>(ResamplingPhaseMode::Linear),
                                                       static_cast<int>(ResamplingPhaseMode::Minimum),
                                                       snapshot.resamplingPhaseMode);
    pendingOverride.smoothingTimeSec = juce::jlimit(SMOOTHING_TIME_MIN_SEC,
                                                    SMOOTHING_TIME_MAX_SEC,
                                                    snapshot.smoothingTimeSec);
    pendingOverride.targetIRLengthSec = juce::jlimit(IR_LENGTH_MIN_SEC,
                                                     IR_LENGTH_MAX_SEC,
                                                     snapshot.targetIRLengthSec);
    pendingOverride.autoDetectedIRLengthSec = juce::jlimit(IR_LENGTH_MIN_SEC,
                                                           IR_LENGTH_MAX_SEC,
                                                           snapshot.autoDetectedIRLengthSec);
    pendingOverride.irLengthManualOverride = snapshot.irLengthManualOverride;
    pendingOverride.mixedTransitionStartHz = juce::jlimit(MIXED_F1_MIN_HZ,
                                                          MIXED_F1_MAX_HZ,
                                                          snapshot.mixedTransitionStartHz);
    pendingOverride.mixedTransitionEndHz = juce::jlimit((std::max)(MIXED_F2_MIN_HZ, pendingOverride.mixedTransitionStartHz + 10.0f),
                                                        MIXED_F2_MAX_HZ,
                                                        snapshot.mixedTransitionEndHz);
    pendingOverride.mixedPreRingTau = juce::jlimit(MIXED_TAU_MIN,
                                                   MIXED_TAU_MAX,
                                                   snapshot.mixedPreRingTau);
    pendingOverride.rebuildDebounceMs = juce::jlimit(REBUILD_DEBOUNCE_MIN_MS,
                                                     REBUILD_DEBOUNCE_MAX_MS,
                                                     snapshot.rebuildDebounceMs);
    pendingOverride.experimentalDirectHeadEnabled = snapshot.experimentalDirectHeadEnabled;
    pendingOverride.tailMode = juce::jlimit(static_cast<int>(TailMode::AirAbsorption),
                                            static_cast<int>(TailMode::Bypass),
                                            snapshot.tailMode);
    pendingOverride.tailStartSec = juce::jlimit(TAIL_START_MIN_SEC,
                                                TAIL_START_MAX_SEC,
                                                snapshot.tailStartSec);
    pendingOverride.tailStrength = juce::jlimit(TAIL_STRENGTH_MIN,
                                                TAIL_STRENGTH_MAX,
                                                snapshot.tailStrength);
    pendingOverride.tailL1L2Multiplier = juce::jlimit(TAIL_L1L2_MULT_MIN,
                                                      TAIL_L1L2_MULT_MAX,
                                                      snapshot.tailL1L2Multiplier);
    pendingOverride.targetUpgradeFFTSize = juce::jlimit(0, 4096, snapshot.targetUpgradeFFTSize);
    pendingOverride.enableProgressiveUpgrade = snapshot.enableProgressiveUpgrade;
    pendingOverride.maxCacheEntries = juce::jlimit(0, 64, snapshot.maxCacheEntries);
    pendingOverride.nucHCMode = juce::jlimit(static_cast<int>(convo::HCMode::Sharp),
                                             static_cast<int>(convo::HCMode::Soft),
                                             snapshot.nucHCMode);
    pendingOverride.nucLCMode = juce::jlimit(static_cast<int>(convo::LCMode::Natural),
                                             static_cast<int>(convo::LCMode::Soft),
                                             snapshot.nucLCMode);
}

[[nodiscard]] juce::ValueTree ConvolverProcessor::getState() const
{
    juce::ValueTree v ("Convolver");
    v.setProperty ("phaseMode", static_cast<int>(getPhaseMode()), nullptr);
    v.setProperty ("useMinPhase", getUseMinPhase(), nullptr);
    float mix, smoothingTime, irLen, autoIRL, mixF1, mixF2, mixTau;
    int tailMode, tailMult;
    float tailStart, tailStrength;
    bool bypassedState, irManual;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        mix     = pendingOverride.mix;
        bypassedState = pendingOverride.bypassed;
        smoothingTime = pendingOverride.smoothingTimeSec;
        irLen   = pendingOverride.targetIRLengthSec;
        autoIRL = pendingOverride.autoDetectedIRLengthSec;
        irManual = pendingOverride.irLengthManualOverride;
        mixF1   = pendingOverride.mixedTransitionStartHz;
        mixF2   = pendingOverride.mixedTransitionEndHz;
        mixTau  = pendingOverride.mixedPreRingTau;
        tailMode = pendingOverride.tailMode;
        tailStart = pendingOverride.tailStartSec;
        tailStrength = pendingOverride.tailStrength;
        tailMult = pendingOverride.tailL1L2Multiplier;
    }
    v.setProperty ("mix", mix, nullptr);
    v.setProperty ("bypassed", bypassedState, nullptr);
    v.setProperty ("smoothingTime", smoothingTime, nullptr);
    v.setProperty ("irLength", irLen, nullptr);
    v.setProperty ("autoDetectedIRLength", autoIRL, nullptr);
    v.setProperty ("irLengthManualOverride", irManual, nullptr);
    v.setProperty ("mixedF1Hz", mixF1, nullptr);
    v.setProperty ("mixedF2Hz", mixF2, nullptr);
    v.setProperty ("mixedTau", mixTau, nullptr);
    v.setProperty ("rebuildDebounceMs", getRebuildDebounceMs(), nullptr);
    v.setProperty ("experimentalDirectHeadEnabled", getExperimentalDirectHeadEnabled(), nullptr);
    v.setProperty ("tailMode", tailMode, nullptr);
    v.setProperty ("tailStartSec", tailStart, nullptr);
    v.setProperty ("tailStrength", tailStrength, nullptr);
    v.setProperty ("tailL1L2Multiplier", tailMult, nullptr);
    v.setProperty ("targetUpgradeFFTSize", getTargetUpgradeFFTSize(), nullptr);
    v.setProperty ("enableProgressiveUpgrade", isProgressiveUpgradeEnabled(), nullptr);
    v.setProperty ("maxCacheEntries", static_cast<int>(getMaxCacheEntries()), nullptr);
    {
        const juce::ScopedLock sl(irFileLock);
        v.setProperty ("irPath", currentIrFile.getFullPathName(), nullptr);
    }
    return v;
}

[[nodiscard]] ConvolverProcessor::BuildSnapshot ConvolverProcessor::captureBuildSnapshot() const
{
    BuildSnapshot snapshot;

    // pending override から値を読む（H3 フェーズ 2c 統一化）
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        copyPendingToSnapshotUnlocked(snapshot);
    }

    snapshot.irName = irName;
    snapshot.irLength = convo::consumeAtomic(irLength, std::memory_order_acquire); // acquire: applyBuildSnapshot の publishAtomic release と HB
    snapshot.currentIRScale = convo::consumeAtomic(currentIRScale, std::memory_order_acquire); // acquire: applyBuildSnapshot の publishAtomic release と HB
    {
        const juce::ScopedLock sl(irFileLock);
        snapshot.irFile = currentIrFile;
    }
    snapshot.fingerprint = computeBuildSnapshotFingerprint(snapshot);
    return snapshot;
}

void ConvolverProcessor::applyBuildSnapshot(const BuildSnapshot& snapshot)
{
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        copySnapshotToPendingUnlocked(snapshot);
    }

    {
        const juce::ScopedLock sl(irFileLock);
        currentIrFile = snapshot.irFile;
    }
    irName = snapshot.irName;
    convo::publishAtomic(irLength, snapshot.irLength, std::memory_order_release); // release: captureBuildSnapshot の acquire と HB
    convo::publishAtomic(currentIRScale, snapshot.currentIRScale, std::memory_order_release); // release: captureBuildSnapshot の acquire と HB

    publishRuntimeProcessSnapshot();
}

void ConvolverProcessor::setState(const juce::ValueTree& v)
{
    if (v.hasProperty ("mix")) setMix (v.getProperty ("mix"));
    if (v.hasProperty ("bypassed")) setBypass (v.getProperty ("bypassed"));
    if (v.hasProperty ("phaseMode"))
    {
        const int modeRaw = static_cast<int>(v.getProperty("phaseMode"));
        const int modeClamped = juce::jlimit(static_cast<int>(PhaseMode::AsIs), static_cast<int>(PhaseMode::Minimum), modeRaw);
        setPhaseMode(static_cast<PhaseMode>(modeClamped));
    }
    else if (v.hasProperty ("useMinPhase"))
    {
        setUseMinPhase (v.getProperty ("useMinPhase"));
    }
    if (v.hasProperty ("smoothingTime")) setSmoothingTime (v.getProperty ("smoothingTime"));

    const bool hasSavedAutoLength = v.hasProperty ("autoDetectedIRLength");
    const bool hasSavedManualOverride = v.hasProperty ("irLengthManualOverride");

    if (hasSavedManualOverride)
    {
        const bool isManual = static_cast<bool>(v.getProperty ("irLengthManualOverride"));

        if (isManual)
        {
            if (hasSavedAutoLength)
            {
                const float autoLength = static_cast<float>(v.getProperty ("autoDetectedIRLength"));
                const float clampedAutoLength = juce::jlimit(IR_LENGTH_MIN_SEC,
                                                             getMaximumAllowedIRLengthSec(convo::consumeAtomic(currentSampleRate, std::memory_order_acquire)), // acquire: prepareToPlay/applyNewState の publishAtomic release と HB
                                                             autoLength);
                {
                    const juce::ScopedLock lock(pendingOverrideLock);
                    pendingOverride.autoDetectedIRLengthSec = clampedAutoLength;
                }
            }

            if (v.hasProperty ("irLength"))
                setTargetIRLength (v.getProperty ("irLength"));

            setIRLengthManualOverride (true);
        }
        else
        {
            if (hasSavedAutoLength)
                applyAutoDetectedIRLength (v.getProperty ("autoDetectedIRLength"));
            else if (v.hasProperty ("irLength"))
                applyAutoDetectedIRLength (v.getProperty ("irLength"));

            setIRLengthManualOverride (false);
        }
    }
    else if (v.hasProperty ("irLength"))
    {
        setTargetIRLength (v.getProperty ("irLength"));
    }

    if (v.hasProperty ("mixedF1Hz")) setMixedTransitionStartHz (v.getProperty ("mixedF1Hz"));
    if (v.hasProperty ("mixedF2Hz")) setMixedTransitionEndHz (v.getProperty ("mixedF2Hz"));
    if (v.hasProperty ("mixedTau")) setMixedPreRingTau (v.getProperty ("mixedTau"));
    if (v.hasProperty ("rebuildDebounceMs")) setRebuildDebounceMs (static_cast<int>(v.getProperty("rebuildDebounceMs")));
    if (v.hasProperty ("experimentalDirectHeadEnabled")) setExperimentalDirectHeadEnabled (v.getProperty ("experimentalDirectHeadEnabled"));

    if (v.hasProperty ("tailMode"))
        setTailMode(static_cast<TailMode>(juce::jlimit(static_cast<int>(TailMode::AirAbsorption),
                                   static_cast<int>(TailMode::Bypass),
                                                       static_cast<int>(v.getProperty("tailMode")))));

    if (v.hasProperty ("tailStartSec")) setTailStartSec (static_cast<float>(v.getProperty("tailStartSec")));
    if (v.hasProperty ("tailStrength")) setTailStrength (static_cast<float>(v.getProperty("tailStrength")));
    if (v.hasProperty ("tailL1L2Multiplier")) setTailL1L2Multiplier (static_cast<int>(v.getProperty("tailL1L2Multiplier")));

    if (v.hasProperty ("targetUpgradeFFTSize")) setTargetUpgradeFFTSize (static_cast<int>(v.getProperty("targetUpgradeFFTSize")));
    if (v.hasProperty ("enableProgressiveUpgrade")) setEnableProgressiveUpgrade (static_cast<bool>(v.getProperty("enableProgressiveUpgrade")));
    if (v.hasProperty ("maxCacheEntries")) setMaxCacheEntries (static_cast<size_t>(static_cast<int>(v.getProperty("maxCacheEntries"))));

    if (v.hasProperty ("irPath"))
    {
        juce::File fileToLoad;
        juce::String path = v.getProperty ("irPath").toString();
        if (path.isNotEmpty())
        {
            juce::File f (path);
            if (f.existsAsFile())
            {
                const juce::ScopedLock sl(irFileLock);
                if (f != currentIrFile)
                    fileToLoad = f;
            }
            else
            {
                lastError = "IR not found: " + f.getFileName();
                postCoalescedChangeNotification();
            }
        }

        if (fileToLoad.existsAsFile())
            loadIR(fileToLoad);
    }
}

void ConvolverProcessor::syncStateFrom(const ConvolverProcessor& other)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    const BuildSnapshot snapshot = other.captureBuildSnapshot();
    applyBuildSnapshot(snapshot);

    if (const IRState* otherState = other.acquireIRState())
    {
        if (otherState->irOwner)
            updateIRState(otherState->irOwner, otherState->sampleRate);
        releaseIRState(otherState);
    }
    {
        const juce::ScopedLock sl(irFileLock);
        currentIrFile = other.currentIrFile;
    }
    irName = other.irName;
    convo::publishAtomic(irLength, convo::consumeAtomic(other.irLength, std::memory_order_acquire), std::memory_order_release); // acquire: other.captureBuildSnapshot の publishAtomic release と HB; release: captureBuildSnapshot の acquire と HB
    convo::publishAtomic(currentIRScale, convo::consumeAtomic(other.currentIRScale, std::memory_order_acquire), std::memory_order_release); // acquire: other.applyBuildSnapshot の publishAtomic release と HB; release: captureBuildSnapshot の acquire と HB

    const uint64_t retireEpoch = (getRcuProvider() != nullptr) ? getRcuProvider()->snapshotRcuEpoch() : 1;
    auto* oldConv = exchangeActiveEngine(nullptr, std::memory_order_acq_rel); // acq_rel: acquire で旧 engine 取得; release で null 公開
    if (oldConv)
        retireStereoConvolver(oldConv, retireEpoch);
}

void ConvolverProcessor::shareConvolutionEngineFrom(const ConvolverProcessor& other)
{
    struct GlobalGuard {
        const ConvolverProcessor& cp;
        GlobalGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterGlobalReader(2); }
        ~GlobalGuard() { cp.exitGlobalReader(2); }
    } guard(other);

    auto* otherConv = other.loadActiveEngine(std::memory_order_acquire); // acquire: other.exchangeActiveEngine acq_rel と HB
    if (otherConv == nullptr)
        return;

    auto* clonedConv = otherConv->clone();
    if (clonedConv == nullptr)
        return;

    const uint64_t retireEpoch = (getRcuProvider() != nullptr) ? getRcuProvider()->snapshotRcuEpoch() : 1;
    auto* oldConv = exchangeActiveEngine(clonedConv, std::memory_order_acq_rel); // acq_rel: acquire で旧 engine 取得; release で新 engine 公開
    if (oldConv)
        retireStereoConvolver(oldConv, retireEpoch);

    auto* otherSnap = convo::consumeAtomic(other.cachedLatency, std::memory_order_acquire); // acquire: other.updateLatencyCache の exchangeAtomic acq_rel と HB
    auto* newSnap = otherSnap ? new LatencySnapshot(*otherSnap) : new LatencySnapshot();
    auto* oldSnap = convo::exchangeAtomic(cachedLatency, newSnap, std::memory_order_acq_rel); // acq_rel: acquire で旧 snapshot 取得; release で新 snapshot 公開
    std::unique_ptr<LatencySnapshot> owned{oldSnap}; // RAII delete

    convo::publishAtomic(irLength, convo::consumeAtomic(other.irLength, std::memory_order_acquire), std::memory_order_release); // acquire: other.applyBuildSnapshot の publishAtomic release と HB; release: captureBuildSnapshot の acquire と HB
    convo::publishAtomic(uiAlgorithmLatencySamples, convo::consumeAtomic(other.uiAlgorithmLatencySamples, std::memory_order_acquire), std::memory_order_release); // acquire: other.refreshLatency の publishAtomic release と HB; release: UI の acquire と HB
    convo::publishAtomic(uiIrPeakLatencySamples, convo::consumeAtomic(other.uiIrPeakLatencySamples, std::memory_order_acquire), std::memory_order_release); // acquire: other.refreshLatency の publishAtomic release と HB; release: UI の acquire と HB
    convo::publishAtomic(uiTotalLatencySamples, convo::consumeAtomic(other.uiTotalLatencySamples, std::memory_order_acquire), std::memory_order_release); // acquire: other.refreshLatency の publishAtomic release と HB; release: UI の acquire と HB
    convo::publishAtomic(uiDirectHeadActive, convo::consumeAtomic(other.uiDirectHeadActive, std::memory_order_acquire), std::memory_order_release); // acquire: other.refreshLatency の publishAtomic release と HB; release: UI の acquire と HB
    requestHostDisplayUpdate();
}

[[nodiscard]] ConvolverProcessor::IRLoadPreview ConvolverProcessor::analyzeImpulseResponseFile(const juce::File& irFile, double processingSampleRate)
{
    IRLoadPreview preview;
    preview.recommendedMaxSec = IR_LENGTH_MAX_SEC;
    preview.hardMaxSec = getMaximumAllowedIRLengthSecForSampleRate(processingSampleRate);

    juce::AudioBuffer<double> loadedIR;
    double loadedSampleRate = 0.0;
    if (!loadImpulseResponsePreviewFile(irFile, loadedIR, loadedSampleRate, preview.errorMessage))
        return preview;

    const auto neverCancel = []() { return false; };

    if (loadedIR.getNumSamples() > 0)
    {
        const int numSamples = loadedIR.getNumSamples();
        const int numChannels = loadedIR.getNumChannels();
        const double threshold = 1.0e-15;
        int newLength = 0;

        if (numChannels > 0)
        {
            const double* ch0Ptr = loadedIR.getReadPointer(0);
            const double* ch1Ptr = (numChannels > 1) ? loadedIR.getReadPointer(1) : nullptr;

            for (int j = numSamples - 1; j >= 0; --j)
            {
                if (std::abs(ch0Ptr[j]) > threshold || (ch1Ptr && std::abs(ch1Ptr[j]) > threshold))
                {
                    newLength = j + 1;
                    break;
                }
            }
        }

        if (newLength < numSamples)
        {
            loadedIR.setSize(numChannels, juce::jmax(1, newLength), true);
            shrinkToFit(loadedIR);
        }
    }

    if (loadedSampleRate > 0.0 && processingSampleRate > 0.0 && std::abs(loadedSampleRate - processingSampleRate) > 1e-6)
    {
        auto resampleOut = resampleIR(loadedIR, loadedSampleRate, processingSampleRate,
                                      r8b::fprLinearPhase, neverCancel);
        if (resampleOut.result != ResampleResult::Success ||
            resampleOut.buffer.getNumSamples() == 0)
        {
            preview.errorMessage = "Resampling failed (unknown error).";
            return preview;
        }

        loadedIR = std::move(resampleOut.buffer);
        loadedSampleRate = processingSampleRate;
    }

    if (loadedSampleRate > 0.0 && loadedIR.getNumSamples() > 0)
    {
        for (int ch = 0; ch < loadedIR.getNumChannels(); ++ch)
        {
            convo::UltraHighRateDCBlocker dcBlocker;
            dcBlocker.init(loadedSampleRate, 1.0);
            dcBlocker.process(loadedIR.getWritePointer(ch), loadedIR.getNumSamples());
        }
    }

    if (loadedIR.getNumSamples() > 0)
    {
        const int numSamples = loadedIR.getNumSamples();
        for (int ch = 0; ch < loadedIR.getNumChannels(); ++ch)
        {
            if (!applyAsymmetricTukey(loadedIR.getWritePointer(ch), numSamples))
            {
                preview.errorMessage = "Failed to allocate Tukey window buffer (Out of Memory).";
                return preview;
            }
        }
    }

    const int detectedSamples = estimateEffectiveIRLengthSamples(loadedIR, loadedSampleRate);
    preview.autoDetectedLengthSamples = detectedSamples;
    preview.autoDetectedLengthSec = (loadedSampleRate > 0.0)
                                  ? static_cast<float>(static_cast<double>(detectedSamples) / loadedSampleRate)
                                  : IR_LENGTH_DEFAULT_SEC;
    preview.exceedsRecommended = preview.autoDetectedLengthSec > preview.recommendedMaxSec;
    preview.exceedsHardLimit = preview.autoDetectedLengthSec > preview.hardMaxSec;
    preview.success = true;
    return preview;
}

[[nodiscard]] std::vector<float> ConvolverProcessor::getIRWaveform()
{
    const juce::ScopedLock sl(visualizationDataLock);
    return irWaveform;
}

[[nodiscard]] std::vector<float> ConvolverProcessor::getIRMagnitudeSpectrum()
{
    const juce::ScopedLock sl(visualizationDataLock);
    return irMagnitudeSpectrum;
}

[[nodiscard]] double ConvolverProcessor::getIRSpectrumSampleRate()
{
    const juce::ScopedLock sl(visualizationDataLock);
    return irSpectrumSampleRate;
}

void ConvolverProcessor::createWaveformSnapshot(const juce::AudioBuffer<double>& irBuffer)
{
    const juce::ScopedLock sl(visualizationDataLock);

    irWaveform.assign(WAVEFORM_POINTS, 0.0f);

    const int numSamples = irBuffer.getNumSamples();
    const int numChannels = irBuffer.getNumChannels();

    if (numSamples <= 0 || numChannels <= 0)
        return;

    const int samplesPerPoint = (std::max)(1, numSamples / WAVEFORM_POINTS);

    float maxAbs = 0.0f;

    for (int i = 0; i < WAVEFORM_POINTS; ++i)
    {
        float peak = 0.0f;
        int startSample = i * samplesPerPoint;
        int endSample = (std::min)(numSamples, startSample + samplesPerPoint);

        for (int ch = 0; ch < numChannels; ++ch)
            for (int j = startSample; j < endSample; ++j)
                peak = (std::max)(peak, static_cast<float>(std::abs(irBuffer.getReadPointer(ch)[j])));

        irWaveform[i] = peak;
        maxAbs = (std::max)(maxAbs, peak);
    }

    if (maxAbs > 0.0f)
        for (float& val : irWaveform) val /= maxAbs;
}

void ConvolverProcessor::createFrequencyResponseSnapshot(const juce::AudioBuffer<double>& irBuffer, double sampleRate)
{
    const juce::ScopedLock sl(visualizationDataLock);

    irSpectrumSampleRate = sampleRate;
    irMagnitudeSpectrum.clear();

    const int numSamples = irBuffer.getNumSamples();
    if (numSamples <= 0 || irBuffer.getNumChannels() < 1) return;

    int fftSize = juce::nextPowerOfTwo(numSamples);
    const int maxFFTSize = 65536;
    if (fftSize > maxFFTSize) fftSize = maxFFTSize;
    if (fftSize < 512) fftSize = 512;

    if (cachedFFTBufferCapacity < fftSize * 2)
    {
        cachedFFTBuffer = convo::makeAlignedArray<float>(static_cast<size_t>(fftSize) * 2u);
        cachedFFTBufferCapacity = fftSize * 2;
    }

    juce::FloatVectorOperations::clear(cachedFFTBuffer.get(), fftSize * 2);

    const double* src = irBuffer.getReadPointer(0);
    const int copyLen = (std::min)(numSamples, fftSize);
    float* dst = cachedFFTBuffer.get();

    if (fftHandle.get() != nullptr && fftHandleSize != fftSize)
    {
        fftHandle.reset();
        fftHandleSize = 0;
    }

    if (fftHandle.get() == nullptr)
    {
        convo::ScopedDftiDescriptor localDfti;
        if (DftiCreateDescriptor(localDfti.put(), DFTI_SINGLE, DFTI_COMPLEX, 1, fftSize) != DFTI_NO_ERROR) return;
        if (DftiSetValue(localDfti.handle, DFTI_PLACEMENT, DFTI_INPLACE) != DFTI_NO_ERROR) return;
        if (DftiCommitDescriptor(localDfti.handle) != DFTI_NO_ERROR) return;
        fftHandle.reset(localDfti.release());
        fftHandleSize = fftSize;
    }

    for (int i = 0; i < copyLen; ++i) {
        dst[2 * i] = static_cast<float>(src[i]);
        dst[2 * i + 1] = 0.0f;
    }
    for (int i = copyLen; i < fftSize; ++i) {
        dst[2 * i] = 0.0f;
        dst[2 * i + 1] = 0.0f;
    }

    if (DftiComputeForward(fftHandle.get(), dst) != DFTI_NO_ERROR) return;

    const int numBins = fftSize / 2 + 1;
    float* magBuf = dst + fftSize + 16;
    vcAbs(numBins, reinterpret_cast<const MKL_Complex8*>(dst), magBuf);
    std::memcpy(dst, magBuf, numBins * sizeof(float));

    if (cachedMagnitudeBufferCapacity < numBins)
    {
        cachedLinearMagsBuffer = convo::makeAlignedArray<float>(static_cast<size_t>(numBins));
        cachedSmoothedMagsBuffer = convo::makeAlignedArray<float>(static_cast<size_t>(numBins));
        if (!cachedLinearMagsBuffer || !cachedSmoothedMagsBuffer)
        {
            cachedMagnitudeBufferCapacity = 0;
            return;
        }
        cachedMagnitudeBufferCapacity = numBins;
    }

    std::memcpy(cachedLinearMagsBuffer.get(), cachedFFTBuffer.get(), static_cast<size_t>(numBins) * sizeof(float));
    applySmoothing(cachedLinearMagsBuffer.get(), cachedSmoothedMagsBuffer.get(), numBins);

    irMagnitudeSpectrum.resize(numBins);

    for (int i = 0; i < numBins; ++i)
    {
        float mag = cachedSmoothedMagsBuffer[i];
        irMagnitudeSpectrum[i] = (mag > 1e-9f) ? juce::Decibels::gainToDecibels(mag) : -100.0f;
    }
}

[[nodiscard]] ConvolverProcessor::LatencyBreakdown ConvolverProcessor::getLatencyBreakdown() const
{
    struct GlobalGuard {
        const ConvolverProcessor& cp;
        GlobalGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterGlobalReader(2); }
        ~GlobalGuard() { cp.exitGlobalReader(2); }
    } guard(*this);

    LatencyBreakdown breakdown;
    if (auto* conv = loadActiveEngine(std::memory_order_acquire)) // acquire: exchangeActiveEngine acq_rel と HB
    {
        const bool directHeadActive = conv->storedDirectHeadEnabled;
        breakdown.directHeadActive = directHeadActive;
        breakdown.algorithmLatencySamples = directHeadActive ? 0 : juce::jmax(0, conv->latency);
        breakdown.irPeakLatencySamples = juce::jmax(0, conv->irLatency);
        breakdown.totalLatencySamples = juce::jmax(0,
            breakdown.algorithmLatencySamples + breakdown.irPeakLatencySamples);

        if (breakdown.algorithmLatencySamples == 0 &&
            breakdown.irPeakLatencySamples == 0 &&
            breakdown.totalLatencySamples == 0)
        {
            const int snapTotal = convo::consumeAtomic(uiTotalLatencySamples, std::memory_order_acquire); // acquire: refreshLatency の publishAtomic release と HB
            if (snapTotal > 0)
            {
                breakdown.algorithmLatencySamples = convo::consumeAtomic(uiAlgorithmLatencySamples, std::memory_order_acquire); // acquire: refreshLatency の publishAtomic release と HB
                breakdown.irPeakLatencySamples = convo::consumeAtomic(uiIrPeakLatencySamples, std::memory_order_acquire); // acquire: refreshLatency の publishAtomic release と HB
                breakdown.totalLatencySamples = snapTotal;
                breakdown.directHeadActive = convo::consumeAtomic(uiDirectHeadActive, std::memory_order_acquire); // acquire: refreshLatency の publishAtomic release と HB
            }
        }
    }

    if (breakdown.algorithmLatencySamples == 0 &&
        breakdown.irPeakLatencySamples == 0 &&
        breakdown.totalLatencySamples == 0)
    {
        const int snapTotal = convo::consumeAtomic(uiTotalLatencySamples, std::memory_order_acquire); // acquire: refreshLatency の publishAtomic release と HB
        if (snapTotal > 0)
        {
            breakdown.algorithmLatencySamples = convo::consumeAtomic(uiAlgorithmLatencySamples, std::memory_order_acquire); // acquire: refreshLatency の publishAtomic release と HB
            breakdown.irPeakLatencySamples = convo::consumeAtomic(uiIrPeakLatencySamples, std::memory_order_acquire); // acquire: refreshLatency の publishAtomic release と HB
            breakdown.totalLatencySamples = snapTotal;
            breakdown.directHeadActive = convo::consumeAtomic(uiDirectHeadActive, std::memory_order_acquire); // acquire: refreshLatency の publishAtomic release と HB
        }
    }

    return breakdown;
}

[[nodiscard]] int ConvolverProcessor::getLatencySamples() const
{
    auto snap = convo::consumeAtomic(cachedLatency, std::memory_order_acquire); // acquire: updateLatencyCache の exchangeAtomic acq_rel と HB
    return snap ? snap->totalLatencySamples : 0;
}

[[nodiscard]] int ConvolverProcessor::getTotalLatencySamples() const
{
    auto snap = convo::consumeAtomic(cachedLatency, std::memory_order_acquire); // acquire: updateLatencyCache の exchangeAtomic acq_rel と HB
    return snap ? snap->totalLatencySamples : 0;
}

void ConvolverProcessor::updateLatencyCache() noexcept
{
    LatencySnapshot snap;
    const auto breakdown = getLatencyBreakdown();
    snap.algorithmLatencySamples = breakdown.algorithmLatencySamples;
    snap.irPeakLatencySamples = breakdown.irPeakLatencySamples;
    snap.totalLatencySamples = breakdown.totalLatencySamples;
    snap.hasParallelDryPath = breakdown.directHeadActive;

    auto* newSnap = new LatencySnapshot(snap);
    auto* oldSnap = convo::exchangeAtomic(cachedLatency, newSnap, std::memory_order_acq_rel); // acq_rel: acquire で旧 snapshot 取得; release で新 snapshot 公開
    std::unique_ptr<LatencySnapshot> ownedOld{oldSnap}; // RAII delete
}

void ConvolverProcessor::requestHostDisplayUpdate()
{
    auto snap = convo::consumeAtomic(cachedLatency, std::memory_order_acquire); // acquire: updateLatencyCache の exchangeAtomic acq_rel と HB
    const int total = snap ? snap->totalLatencySamples : 0;
    if (total == lastReportedLatency)
        return;

    if (convo::exchangeAtomic(latencyChangePending, true, std::memory_order_acq_rel)) // acq_rel: acquire で先決値監測; release で pending=true 公開
        return;

    const bool queued = juce::MessageManager::callAsync([weakThis = juce::WeakReference<ConvolverProcessor>(this)]
    {
        if (auto* self = weakThis.get())
        {
            auto snap2 = convo::consumeAtomic(self->cachedLatency, std::memory_order_acquire); // acquire: updateLatencyCache の exchangeAtomic acq_rel と HB
            const int latest = snap2 ? snap2->totalLatencySamples : 0;
            if (latest != self->lastReportedLatency)
            {
                self->lastReportedLatency = latest;
                self->postCoalescedChangeNotification();
            }
            convo::publishAtomic(self->latencyChangePending, false, std::memory_order_release); // release: pending=false 公開（callAsync 成功時）
        }
    });

    if (!queued)
        convo::publishAtomic(latencyChangePending, false, std::memory_order_release); // release: pending=false 公開（callAsync 失敗時）
}

void ConvolverProcessor::setPhaseMode(PhaseMode mode)
{
    const int newMode = static_cast<int>(mode);
    int oldMode;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        oldMode = pendingOverride.phaseMode;
    }
    if (oldMode != newMode)
    {
        // Pending override に値を記録（snapshot ベースのデータフロー）
        {
            const juce::ScopedLock lock(pendingOverrideLock);
            pendingOverride.phaseMode = newMode;
        }
        postCoalescedChangeNotification();
    }
}

[[nodiscard]] ConvolverProcessor::PhaseMode ConvolverProcessor::getPhaseMode() const
{
    const BuildSnapshot snapshot = captureBuildSnapshot();
    const int mode = juce::jlimit(static_cast<int>(PhaseMode::AsIs),
                                  static_cast<int>(PhaseMode::Minimum),
                                  snapshot.phaseMode);
    return static_cast<PhaseMode>(mode);
}

void ConvolverProcessor::setUseMinPhase(bool shouldUseMinPhase)
{
    setPhaseMode(shouldUseMinPhase ? PhaseMode::Minimum : PhaseMode::AsIs);
}

void ConvolverProcessor::setNUCFilterModes(convo::HCMode hcMode, convo::LCMode lcMode)
{
    const int newHC = static_cast<int>(hcMode);
    const int newLC = static_cast<int>(lcMode);

    int oldHC, oldLC;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        oldHC = pendingOverride.nucHCMode;
        oldLC = pendingOverride.nucLCMode;
    }

    const bool changed = (oldHC != newHC) || (oldLC != newLC);

    if (changed)
    {
        {
            const juce::ScopedLock lock(pendingOverrideLock);
            pendingOverride.nucHCMode = newHC;
            pendingOverride.nucLCMode = newLC;
        }
        postCoalescedChangeNotification();
    }
}

[[nodiscard]] uint64_t ConvolverProcessor::getStructuralHash() const noexcept
{
    uint64_t hash = 0x9e3779b97f4a7c15ULL;
    const BuildSnapshot snapshot = captureBuildSnapshot();

    auto hashCombine = [&hash](uint64_t value) {
        hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    };

    hashCombine(convo::consumeAtomic(activeCacheKey, std::memory_order_acquire)); // acquire: applyComputedIR/applyNewState の publishAtomic release と HB
    hashCombine(static_cast<uint64_t>(convo::consumeAtomic(irLength, std::memory_order_acquire))); // acquire: applyBuildSnapshot/applyComputedIR の publishAtomic release と HB
    hashCombine(static_cast<uint64_t>(juce::jlimit(static_cast<int>(PhaseMode::AsIs),
                                                   static_cast<int>(PhaseMode::Minimum),
                                                   snapshot.phaseMode)));

    auto floatBits = [](float f) -> uint32_t {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        return bits;
    };
    hashCombine(floatBits(snapshot.mixedTransitionStartHz));
    hashCombine(floatBits(snapshot.mixedTransitionEndHz));
    hashCombine(floatBits(snapshot.mixedPreRingTau));

    hashCombine(snapshot.experimentalDirectHeadEnabled ? 1ULL : 0ULL);
    hashCombine(static_cast<uint64_t>(snapshot.nucHCMode));
    hashCombine(static_cast<uint64_t>(snapshot.nucLCMode));
    hashCombine(static_cast<uint64_t>(snapshot.tailMode));
    hashCombine(floatBits(snapshot.tailStartSec));
    hashCombine(floatBits(snapshot.tailStrength));
    hashCombine(static_cast<uint64_t>(snapshot.tailL1L2Multiplier));

    return hash;
}

[[nodiscard]] uint64_t ConvolverProcessor::getActiveCacheKey() const noexcept
{
    return convo::consumeAtomic(activeCacheKey, std::memory_order_acquire); // acquire: applyComputedIR/applyNewState の publishAtomic release と HB
}

[[nodiscard]] int ConvolverProcessor::getActiveCacheFFTSize() const noexcept
{
    return convo::consumeAtomic(activeCacheFFTSize, std::memory_order_acquire); // acquire: applyComputedIR/applyNewState の publishAtomic release と HB
}

[[nodiscard]] int ConvolverProcessor::getNUCHCMode() const noexcept
{
    const BuildSnapshot snapshot = captureBuildSnapshot();
    return snapshot.nucHCMode;
}

[[nodiscard]] int ConvolverProcessor::getNUCLCMode() const noexcept
{
    const BuildSnapshot snapshot = captureBuildSnapshot();
    return snapshot.nucLCMode;
}

void ConvolverProcessor::setResamplingPhaseMode(ResamplingPhaseMode mode)
{
    bool changed = false;
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        const int newMode = static_cast<int>(mode);
        changed = (pendingOverride.resamplingPhaseMode != newMode);
        pendingOverride.resamplingPhaseMode = newMode;
    }
    if (changed)
        postCoalescedChangeNotification();
}

[[nodiscard]] ConvolverProcessor::ResamplingPhaseMode ConvolverProcessor::getResamplingPhaseMode() const
{
    const BuildSnapshot snapshot = captureBuildSnapshot();
    const int mode = juce::jlimit(static_cast<int>(ResamplingPhaseMode::Linear),
                                  static_cast<int>(ResamplingPhaseMode::Minimum),
                                  snapshot.resamplingPhaseMode);
    return static_cast<ResamplingPhaseMode>(mode);
}

[[nodiscard]] bool ConvolverProcessor::getExperimentalDirectHeadEnabled() const
{
    const BuildSnapshot snapshot = captureBuildSnapshot();
    return snapshot.experimentalDirectHeadEnabled;
}

[[nodiscard]] float ConvolverProcessor::getMaximumAllowedIRLengthSecForSampleRate(double sampleRate)
{
    if (sampleRate <= 0.0)
        return IR_LENGTH_MAX_SEC;

    return static_cast<float>(static_cast<double>(MAX_IR_LATENCY) / sampleRate);
}

[[nodiscard]] float ConvolverProcessor::getMaximumAllowedIRLengthSec(double sampleRate) const
{
    const double sr = (sampleRate > 0.0)
                    ? sampleRate
                    : convo::consumeAtomic(currentSampleRate, std::memory_order_acquire); // acquire: prepareToPlay/applyNewState の publishAtomic release と HB

    return getMaximumAllowedIRLengthSecForSampleRate(sr);
}

int ConvolverProcessor::computeTargetIRLength(double sampleRate, int originalLength) const
{
    juce::ignoreUnused(originalLength);
    const double targetIRTimeSec = [this]() -> double {
        const juce::ScopedLock lock(pendingOverrideLock);
        return static_cast<double>(pendingOverride.targetIRLengthSec);
    }();
    static constexpr int kMaxIRCap = MAX_IR_LATENCY;

    int target = static_cast<int>(sampleRate * targetIRTimeSec);

    target = (std::min)(target, kMaxIRCap);
    target = (std::max)(target, 1);

    return target;
}

void ConvolverProcessor::debugCheckAtomicLockFree() const
{
   #if JUCE_DEBUG
    if (!std::atomic<LatencySnapshot>{}.is_lock_free())
    {
        DBG("[ConvolverProcessor] std::atomic<LatencySnapshot> is not lock-free on this build; continuing with implementation-provided atomic semantics.");
    }
   #endif
}

void ConvolverProcessor::forceCleanup()
{
    if (rebuildJob)
    {
        rebuildJob->reset();
        rebuildJob.reset();
    }

    std::deque<std::unique_ptr<LoaderThread>> loadersToDelete;
    loadersToDelete.swap(loaderTrashBin);
    if (activeLoader)
        loadersToDelete.push_back(std::move(activeLoader));

    for (auto& loader : loadersToDelete)
    {
        if (loader)
            loader->stopThread(500);
    }
    loadersToDelete.clear();
}

void ConvolverProcessor::updateConvolverState(convo::ConvolverState* newState)
{
    JUCE_ASSERT_MESSAGE_THREAD;
    jassert(newState != nullptr);
    if (!newState) return;

    jassert(!convo::exchangeAtomic(writerActive, true, std::memory_order_acquire)); // acquire: 先行 交換会技 = falseを測定

    if (!convolverStateGeneration.isCurrentGeneration(newState->generationId))
    {
        juce::Logger::writeToLog("ConvolverProcessor::updateConvolverState: stale generation, discarding state (gen="
            + juce::String((int)newState->generationId) + ")");
        std::unique_ptr<convo::ConvolverState> owned{newState}; // RAII delete (stale discard)
        convo::publishAtomic(writerActive, false, std::memory_order_release); // release: writerActive=false 公開
        return;
    }

    rcuSwapper.swap(newState);
    convo::publishAtomic(convolverState, newState, std::memory_order_release); // release: convolverState 公開
    m_epochDomain.advanceEpoch();

    convo::publishAtomic(writerActive, false, std::memory_order_release); // release: writerActive=false 公開
}

void ConvolverProcessor::updateConvolverState(std::unique_ptr<convo::ConvolverState> newState)
{
    updateConvolverState(newState.release());
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_STATE_UI
