#include <JuceHeader.h>
#include <bit>
#include "AudioEngine.h"
#include "NoiseShaperLearner.h"
#include "RuntimeBuilder.h"
#include "DSPLifetimeManager.h"

namespace {
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

struct BuildParameterSnapshot
{
    int ditherDepth = 0;
    int oversamplingFactor = 0;
    AudioEngine::OversamplingType oversamplingType = AudioEngine::OversamplingType::IIR;
    AudioEngine::NoiseShaperType noiseShaperType = AudioEngine::NoiseShaperType::Psychoacoustic;
    convo::ProcessingOrder processingOrder = convo::ProcessingOrder::ConvolverThenEQ;
    bool eqBypassed = false;
    bool convBypassed = false;
    bool softClipEnabled = false;
    double saturationAmount = 0.0;
    double inputHeadroomGain = 1.0;
    double outputMakeupGain = 1.0;
    double convolverInputTrimGain = 1.0;
};

BuildParameterSnapshot captureBuildParameterSnapshot(const AudioEngine& engine) noexcept
{
    BuildParameterSnapshot snapshot {};
    snapshot.ditherDepth = convo::consumeAtomic(engine.ditherBitDepth, std::memory_order_acquire);
    snapshot.oversamplingFactor = convo::consumeAtomic(engine.manualOversamplingFactor, std::memory_order_acquire);
    snapshot.oversamplingType = convo::consumeAtomic(engine.oversamplingType, std::memory_order_acquire);
    snapshot.noiseShaperType = convo::consumeAtomic(engine.noiseShaperType, std::memory_order_acquire);
    snapshot.processingOrder = convo::consumeAtomic(engine.currentProcessingOrder, std::memory_order_acquire);
    snapshot.eqBypassed = convo::consumeAtomic(engine.eqBypassRequested, std::memory_order_acquire);
    snapshot.convBypassed = convo::consumeAtomic(engine.convBypassRequested, std::memory_order_acquire);
    snapshot.softClipEnabled = convo::consumeAtomic(engine.softClipEnabled, std::memory_order_acquire);
    snapshot.saturationAmount = static_cast<double>(convo::consumeAtomic(engine.saturationAmount, std::memory_order_acquire));
    snapshot.inputHeadroomGain = convo::consumeAtomic(engine.inputHeadroomGain, std::memory_order_acquire);
    snapshot.outputMakeupGain = convo::consumeAtomic(engine.outputMakeupGain, std::memory_order_acquire);
    snapshot.convolverInputTrimGain = convo::consumeAtomic(engine.convolverInputTrimGain, std::memory_order_acquire);
    return snapshot;
}

bool equalsBuildParameterSnapshot(const BuildParameterSnapshot& lhs,
                                  const BuildParameterSnapshot& rhs) noexcept
{
    return lhs.ditherDepth == rhs.ditherDepth
        && lhs.oversamplingFactor == rhs.oversamplingFactor
        && lhs.oversamplingType == rhs.oversamplingType
        && lhs.noiseShaperType == rhs.noiseShaperType
        && lhs.processingOrder == rhs.processingOrder
        && lhs.eqBypassed == rhs.eqBypassed
        && lhs.convBypassed == rhs.convBypassed
        && lhs.softClipEnabled == rhs.softClipEnabled
        && lhs.saturationAmount == rhs.saturationAmount
        && lhs.inputHeadroomGain == rhs.inputHeadroomGain
        && lhs.outputMakeupGain == rhs.outputMakeupGain
        && lhs.convolverInputTrimGain == rhs.convolverInputTrimGain;
}

bool shouldRetryWarmupFailure(const AudioEngine::DSPCore& dsp) noexcept
{
    return dsp.convolverRt().isLoadingIR();
}

convo::RuntimeBuildSnapshot captureRuntimeBuildSnapshot(const convo::BuildInput& buildInput,
                                                        const ConvolverProcessor::BuildSnapshot& convolverSnapshot,
                                                        int generation,
                                                        std::uint64_t structuralHash,
                                                        bool irLoaded,
                                                        bool irFinalized) noexcept
{
    convo::RuntimeBuildSnapshot snapshot {};
    snapshot.generation = generation;
    snapshot.buildInput = buildInput;
    snapshot.convolverFingerprint = convolverSnapshot.fingerprint;
    snapshot.rebuildFingerprint.fingerprintVersion = convo::RuntimeBuildFingerprint{}.fingerprintVersion;
    snapshot.rebuildFingerprint.irIdentityHash = structuralHash;
    snapshot.rebuildFingerprint.convolutionConfigHash = convolverSnapshot.fingerprint;
    snapshot.rebuildFingerprint.sampleRate = buildInput.sampleRate;
    snapshot.rebuildFingerprint.blockSize = buildInput.blockSize;
    // [PR-2] DSP semantic projection snapshot values
    snapshot.irLoaded = irLoaded;
    snapshot.irFinalized = irFinalized;
    snapshot.structuralHash = structuralHash;
    snapshot.oversamplingFactor = buildInput.oversamplingFactor;
    snapshot.sampleRate = buildInput.sampleRate;
    return snapshot;
}

convo::RuntimeBuildSnapshot finalizeRuntimeBuildSnapshot(convo::RuntimeBuildSnapshot snapshot) noexcept
{
    // finalize: semantic input のみで正規化し、非決定的入力を取り込まない。
    snapshot.buildInput.sampleRate = std::max(0.0, snapshot.buildInput.sampleRate);
    snapshot.buildInput.blockSize = std::max(0, snapshot.buildInput.blockSize);
    snapshot.buildInput.oversamplingFactor = std::max(0, snapshot.buildInput.oversamplingFactor);
    snapshot.rebuildFingerprint.sampleRate = snapshot.buildInput.sampleRate;
    snapshot.rebuildFingerprint.blockSize = snapshot.buildInput.blockSize;

    std::uint64_t paramHash = 1469598103934665603ull;
    const auto mixHash = [&paramHash](std::uint64_t value) noexcept {
        paramHash ^= value;
        paramHash *= 1099511628211ull;
    };

    mixHash(static_cast<std::uint64_t>(snapshot.buildInput.ditherBitDepth));
    mixHash(static_cast<std::uint64_t>(snapshot.buildInput.oversamplingFactor));
    mixHash(static_cast<std::uint64_t>(snapshot.buildInput.oversamplingType));
    mixHash(static_cast<std::uint64_t>(snapshot.buildInput.noiseShaperType));
    mixHash(static_cast<std::uint64_t>(snapshot.buildInput.processingOrder));
    mixHash(static_cast<std::uint64_t>(snapshot.buildInput.eqBypassed));
    mixHash(static_cast<std::uint64_t>(snapshot.buildInput.convBypassed));
    mixHash(static_cast<std::uint64_t>(snapshot.buildInput.softClipEnabled));
    mixHash(static_cast<std::uint64_t>(snapshot.buildInput.blockSize));
    mixHash(std::bit_cast<std::uint64_t>(snapshot.buildInput.saturationAmount));
    mixHash(std::bit_cast<std::uint64_t>(snapshot.buildInput.inputHeadroomGain));
    mixHash(std::bit_cast<std::uint64_t>(snapshot.buildInput.outputMakeupGain));
    mixHash(std::bit_cast<std::uint64_t>(snapshot.buildInput.convolverInputTrimGain));
    mixHash(static_cast<std::uint64_t>(snapshot.convolverFingerprint));
    snapshot.rebuildFingerprint.dspParameterHash = paramHash;

    return snapshot;
}

convo::RuntimeBuildSnapshot sealRuntimeBuildSnapshot(convo::RuntimeBuildSnapshot snapshot) noexcept
{
    snapshot.sealed = true;
    return snapshot;
}
}

void AudioEngine::submitRebuildIntent(convo::RebuildKind kind,
                                      RebuildTelemetryReason reason,
                                      RebuildTelemetryClass rebuildClass,
                                      RebuildTelemetryPolicy collapsePolicy) noexcept
{
    constexpr const char* kPhase5TagReduce = "phase5_reduce_target";
    constexpr const char* kPhase5TagKeep = "phase5_keep_target";

    const int64_t nowTicks = juce::Time::getHighResolutionTicks();
    const std::uint32_t fingerprintVersion = convo::RuntimeBuildFingerprint {}.fingerprintVersion;
    uint64_t structuralHash = 0;
    uint64_t fingerprint = 0;
    bool isMessageThread = false;
    if (auto* mm = juce::MessageManager::getInstanceWithoutCreating(); mm != nullptr)
        isMessageThread = mm->isThisTheMessageThread();

    if (kind == convo::RebuildKind::Structural && isMessageThread)
    {
        if (uiConvolverProcessor.isIRLoaded())
            structuralHash = uiConvolverProcessor.getStructuralHash();
        fingerprint = uiConvolverProcessor.captureBuildSnapshot().fingerprint;
    }

    const double srSnapshot = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
    const int bsSnapshot = convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire);
    const bool deferCategory = (kind == convo::RebuildKind::Structural) && !(srSnapshot > 0.0 && bsSnapshot > 0);
    const int queuedGenerationSnapshot = convo::consumeAtomic(rebuildRequestGeneration, std::memory_order_acquire);
    const int committedGenerationSnapshot = convo::consumeAtomic(lastCommittedRebuildGeneration, std::memory_order_acquire);
    const bool rebuildOutstanding = queuedGenerationSnapshot > committedGenerationSnapshot;

    bool sameAsPendingWouldMerge = false;
    bool shouldApplyLatestWinsMerge = false;
    int64_t latestWinsWindowTicks = 0;
    int latestWinsWindowMs = 0;

    if (collapsePolicy == RebuildTelemetryPolicy::Replaceable)
    {
        // UI burst 吸収専用。MustExecute では抑止しない。
        latestWinsWindowMs = (kind == convo::RebuildKind::Structural)
            ? std::max(1, uiConvolverProcessor.getRebuildDebounceMs())
            : 50;
        latestWinsWindowTicks = (juce::Time::getHighResolutionTicksPerSecond() * latestWinsWindowMs) / 1000;
    }

    {
        std::lock_guard<std::mutex> lock(rebuildAdmissionIntentMutex_);
        if (!rebuildOutstanding)
            rebuildAdmissionPendingIntent_.valid = false;

        const auto& pending = rebuildAdmissionPendingIntent_;
        sameAsPendingWouldMerge = rebuildOutstanding
            && pending.valid
            && pending.kind == kind
            && pending.rebuildClass == rebuildClass
            && pending.collapsePolicy == collapsePolicy
            && pending.fingerprintVersion == fingerprintVersion
            && pending.structuralHash == structuralHash
            && pending.fingerprint == fingerprint
            && pending.deferCategory == deferCategory;

        if (sameAsPendingWouldMerge
            && collapsePolicy == RebuildTelemetryPolicy::Replaceable
            && pending.lastIntentTicks > 0
            && latestWinsWindowTicks > 0)
        {
            const int64_t elapsed = nowTicks - pending.lastIntentTicks;
            shouldApplyLatestWinsMerge = (elapsed >= 0 && elapsed <= latestWinsWindowTicks);
        }

        rebuildAdmissionPendingIntent_.valid = true;
        rebuildAdmissionPendingIntent_.kind = kind;
        rebuildAdmissionPendingIntent_.rebuildClass = rebuildClass;
        rebuildAdmissionPendingIntent_.collapsePolicy = collapsePolicy;
        rebuildAdmissionPendingIntent_.fingerprintVersion = fingerprintVersion;
        rebuildAdmissionPendingIntent_.structuralHash = structuralHash;
        rebuildAdmissionPendingIntent_.fingerprint = fingerprint;
        rebuildAdmissionPendingIntent_.deferCategory = deferCategory;
        rebuildAdmissionPendingIntent_.lastIntentTicks = nowTicks;
    }

    const uint64_t intentId = nextRebuildTelemetryIntentId();
    emitRebuildTelemetry(RebuildTelemetryEvent::Requested,
                         intentId,
                         reason,
                         RebuildTelemetryDecision::Accepted,
                         structuralHash,
                         fingerprint,
                         rebuildClass,
                         collapsePolicy);

    if (isShutdownInProgress())
    {
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                             intentId,
                             RebuildTelemetryReason::ShutdownInProgress,
                             RebuildTelemetryDecision::Suppressed,
                             structuralHash,
                             fingerprint,
                             rebuildClass,
                             collapsePolicy,
                             kPhase5TagKeep);
        return;
    }

    if (shouldRejectRebuildAdmissionForPressure())
    {
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                             intentId,
                             RebuildTelemetryReason::RetirePressureSevere,
                             RebuildTelemetryDecision::Suppressed,
                             structuralHash,
                             fingerprint,
                             rebuildClass,
                             collapsePolicy,
                             kPhase5TagKeep);
        return;
    }

    if (sameAsPendingWouldMerge && shouldApplyLatestWinsMerge)
    {
        emitRebuildTelemetry(RebuildTelemetryEvent::Merged,
                             intentId,
                             RebuildTelemetryReason::SameAsPendingWouldMerge,
                             RebuildTelemetryDecision::Merged,
                             structuralHash,
                             fingerprint,
                             rebuildClass,
                             collapsePolicy,
                             kPhase5TagReduce,
                             static_cast<double>(latestWinsWindowMs));
        convo::fetchAddAtomic(rebuildCollapseCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        return;
    }

    if (isShutdownInProgress())
    {
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                     intentId,
                     RebuildTelemetryReason::ShutdownInProgress,
                     RebuildTelemetryDecision::Suppressed,
                     0,
                     0,
                     RebuildTelemetryClass::Structural,
                     collapsePolicy,
                     kPhase5TagKeep);
        return;
    }

    if (kind == convo::RebuildKind::None || kind == convo::RebuildKind::Runtime)
    {
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                     intentId,
                     RebuildTelemetryReason::KindFiltered,
                     RebuildTelemetryDecision::Suppressed,
                     0,
                     0,
                     rebuildClass,
                     collapsePolicy,
                     kPhase5TagKeep);
        return;
    }

    // Message Thread かつ実行コンテキスト有効時は直接 rebuild 実行へ進む
    if (kind == convo::RebuildKind::Structural)
    {
        if (isMessageThread)
        {
            clearRebuildReason(RebuildReason::StructuralFromNonMT);

            const double sr = srSnapshot;
            const int bs = bsSnapshot;
            if (sr > 0.0 && bs > 0)
            {
                emitRebuildTelemetry(RebuildTelemetryEvent::Dispatched,
                                     intentId,
                                     RebuildTelemetryReason::DelegateRequestRebuildSrBs,
                                     RebuildTelemetryDecision::Dispatched,
                                     structuralHash,
                                     fingerprint,
                                     RebuildTelemetryClass::Structural,
                                     collapsePolicy);
                requestRebuild(sr, bs, collapsePolicy == RebuildTelemetryPolicy::MustExecute);
                return;
            }

            setRebuildReason(RebuildReason::DeferredFinalizeAware);
            emitRebuildTelemetry(RebuildTelemetryEvent::Deferred,
                                 intentId,
                                 RebuildTelemetryReason::MissingSrBs,
                                 RebuildTelemetryDecision::Deferred,
                                 0,
                                 0,
                                 RebuildTelemetryClass::FinalizeAware,
                                 collapsePolicy);
            return;
        }
    }

    // 非MTパス: bool フラグをセットして MT へ通知
    const bool wasNewlyPending = setRebuildReason(RebuildReason::StructuralFromNonMT);
    if (wasNewlyPending)
    {
        emitRebuildTelemetry(RebuildTelemetryEvent::Dispatched,
                             intentId,
                             RebuildTelemetryReason::NonMtTriggerAsync,
                             RebuildTelemetryDecision::Dispatched,
                             0,
                             0,
                             RebuildTelemetryClass::Structural,
                             collapsePolicy);
        triggerAsyncUpdate();
    }
    else
    {
        emitRebuildTelemetry(RebuildTelemetryEvent::Merged,
                             intentId,
                             RebuildTelemetryReason::NonMtAlreadyPending,
                             RebuildTelemetryDecision::Merged,
                             0,
                             0,
                             RebuildTelemetryClass::Structural,
                             collapsePolicy);
    }
}

void AudioEngine::handleAsyncUpdate()
{
    if (isShutdownInProgress())
        return;

    // [PR-3] Old pending commit path removed. Orchestrator handles deferred commits.

    // 非MT起点の Structural rebuild 要求を消費して実行する
    if (clearRebuildReason(RebuildReason::StructuralFromNonMT))
    {
        RebuildTelemetryPolicy collapsePolicy = RebuildTelemetryPolicy::NA;
        {
            std::lock_guard<std::mutex> lock(rebuildAdmissionIntentMutex_);
            const auto& pending = rebuildAdmissionPendingIntent_;
            if (pending.valid && pending.kind == convo::RebuildKind::Structural)
                collapsePolicy = pending.collapsePolicy;
        }

        const uint64_t intentId = nextRebuildTelemetryIntentId();
        emitRebuildTelemetry(RebuildTelemetryEvent::Requested,
                     intentId,
                     RebuildTelemetryReason::AsyncBridgeConsume,
                     RebuildTelemetryDecision::Accepted,
                     0,
                     0,
                     RebuildTelemetryClass::Structural,
                     collapsePolicy);

        const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
        const int bs = convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire);
        if (sr > 0.0 && bs > 0)
        {
            emitRebuildTelemetry(RebuildTelemetryEvent::Dispatched,
                                 intentId,
                                 RebuildTelemetryReason::AsyncBridgeDelegateSrBs,
                                 RebuildTelemetryDecision::Dispatched,
                                 0,
                                 0,
                                 RebuildTelemetryClass::Structural,
                                 collapsePolicy);
            requestRebuild(sr, bs, collapsePolicy == RebuildTelemetryPolicy::MustExecute);
        }
        else
        {
            setRebuildReason(RebuildReason::DeferredFinalizeAware);
            emitRebuildTelemetry(RebuildTelemetryEvent::Deferred,
                                 intentId,
                                 RebuildTelemetryReason::AsyncBridgeMissingSrBs,
                                 RebuildTelemetryDecision::Deferred,
                                 0,
                                 0,
                                 RebuildTelemetryClass::FinalizeAware,
                                 collapsePolicy);
        }
    }
}

void AudioEngine::requestRebuild(convo::RebuildKind kind) noexcept
{
    if (isShutdownInProgress())
    {
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        return;
    }

    if (shouldRejectRebuildAdmissionForPressure())
    {
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        return;
    }

    submitRebuildIntent(kind,
                        RebuildTelemetryReason::RequestRebuildKindEntry,
                        RebuildTelemetryClass::Structural,
                        RebuildTelemetryPolicy::Replaceable);
}


//--------------------------------------------------------------
// requestRebuild - DSPグラフの再構築 (Message Thread)
//--------------------------------------------------------------
void AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock, bool forceMustExecute)
{
    constexpr const char* kPhase5TagKeep = "phase5_keep_target";

    // forceMustExecute=true は、重複抑止より実行優先を選ぶ明示的な回避経路。
    const auto collapsePolicy = forceMustExecute
        ? RebuildTelemetryPolicy::MustExecute
        : RebuildTelemetryPolicy::Replaceable;

    const uint64_t intentId = nextRebuildTelemetryIntentId();
    const double requestStartMs = juce::Time::getMillisecondCounterHiRes();
    emitRebuildTelemetry(RebuildTelemetryEvent::Requested,
                         intentId,
                         RebuildTelemetryReason::RequestRebuildSrBs,
                         RebuildTelemetryDecision::Accepted,
                         0,
                         0,
                         RebuildTelemetryClass::Structural,
                         collapsePolicy);

    if (isShutdownInProgress())
    {
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                             intentId,
                             RebuildTelemetryReason::ShutdownInProgress,
                             RebuildTelemetryDecision::Suppressed,
                             0,
                             0,
                             RebuildTelemetryClass::Structural,
                             collapsePolicy,
                             kPhase5TagKeep);
        return;
    }

    if (shouldRejectRebuildAdmissionForPressure())
    {
        convo::fetchAddAtomic(publicationRejectCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                             intentId,
                             RebuildTelemetryReason::RetirePressureSevere,
                             RebuildTelemetryDecision::Suppressed,
                             0,
                             0,
                             RebuildTelemetryClass::Structural,
                             collapsePolicy,
                             kPhase5TagKeep);
        return;
    }

    convo::fetchAddAtomic(rtAuxMutable_.debugRebuildDispatchRequestCount, 1, std::memory_order_acq_rel);

    // UIコンポーネントへのアクセスを行うため、必ずMessage Threadで実行すること
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    // シャットダウン中のリクエストは早期に切る
    if (isShutdownInProgress())
    {
        // Phase5: 維持対象（グローバル安全ガード）
        diagLog("[DIAG][PHASE5-KEEP] requestRebuild(sr,bs): ignored during shutdown");
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                     intentId,
                     RebuildTelemetryReason::ShutdownInProgress,
                     RebuildTelemetryDecision::Suppressed,
                     0,
                     0,
                     RebuildTelemetryClass::Structural,
                     collapsePolicy,
                     kPhase5TagKeep);
        return;
    }

    // Phase5: 縮退対象（deferred window suppress）は削除済み。
    // DeferredStructural が立っていても、ここでは suppress せず通常の queue 判定へ進める。

    const int publishedFftSize = uiConvolverProcessor.getActiveCacheFFTSize();
    const int targetFftSize = uiConvolverProcessor.getTargetUpgradeFFTSize();
    const bool suppressIntermediateMixedPhasePublish =
        uiConvolverProcessor.isProgressiveUpgradeEnabled()
        && uiConvolverProcessor.getPhaseMode() == ConvolverProcessor::PhaseMode::Mixed
        && publishedFftSize > 0
        && publishedFftSize < targetFftSize;

    if (suppressIntermediateMixedPhasePublish)
    {
        // Phase5: 維持対象（中間状態 publish 防止の安全ガード）
        diagLog("[DIAG][PHASE5-KEEP] requestRebuild(sr,bs): SUPPRESSED intermediate progressive mixed-phase publish fft="
                + juce::String(publishedFftSize)
                + " targetFFT=" + juce::String(targetFftSize)
                + " SR=" + juce::String(sampleRate, 2));
        emitRebuildTelemetry(RebuildTelemetryEvent::Suppressed,
                     intentId,
                     RebuildTelemetryReason::MixedPhaseIntermediate,
                     RebuildTelemetryDecision::Suppressed,
                     0,
                     0,
                     RebuildTelemetryClass::Structural,
                 collapsePolicy,
                 kPhase5TagKeep);
        return;
    }

    // NOTE: 意図的に learner を停止しない。rebuild 中も learner は稼働を継続し、
    // DSP commit 後に captureSessionSignature() で自然なセッション再設定を行う。
    // rebuild 完了時に DSPReady が enqueue され、processLearningCommands が
    // learningRuntimeState に応じて適切に再開を処理する。
    // ここで stopLearning() を呼ぶと learningRuntimeState が Running のまま
    // ワーカーだけが停止し、DSPReady が WaitingForDSP 以外では再開しないため、
    // 学習が永久に停止する不具合の原因となる。

    // rebuild 開始時のパラメータを凍結し、
    // task 作成・重複判定・runtime command で同一 snapshot を使う。
    const BuildParameterSnapshot paramSnapshot = captureBuildParameterSnapshot(*this);
    int generation = 0;

    RebuildTask task;
    task.currentDSP = nullptr;
    task.buildInput.sampleRate = sampleRate;
    task.buildInput.blockSize = samplesPerBlock;
    task.buildInput.ditherBitDepth = paramSnapshot.ditherDepth;
    task.buildInput.oversamplingFactor = paramSnapshot.oversamplingFactor;
    task.buildInput.oversamplingType = static_cast<int>(paramSnapshot.oversamplingType);
    task.buildInput.noiseShaperType = static_cast<int>(paramSnapshot.noiseShaperType);
    task.buildInput.processingOrder = static_cast<int>(paramSnapshot.processingOrder);
    task.buildInput.eqBypassed = paramSnapshot.eqBypassed;
    task.buildInput.convBypassed = paramSnapshot.convBypassed;
    task.buildInput.softClipEnabled = paramSnapshot.softClipEnabled;
    task.buildInput.saturationAmount = paramSnapshot.saturationAmount;
    task.buildInput.inputHeadroomGain = paramSnapshot.inputHeadroomGain;
    task.buildInput.outputMakeupGain = paramSnapshot.outputMakeupGain;
    task.buildInput.convolverInputTrimGain = paramSnapshot.convolverInputTrimGain;
    task.convolverBuildSnapshot = uiConvolverProcessor.captureBuildSnapshot();
    const uint64_t structuralHash = uiConvolverProcessor.isIRLoaded() ? uiConvolverProcessor.getStructuralHash() : 0;

    DSPCore* currentToRelease = nullptr;
    bool queued = false;
    bool blockedAsDuplicate = false;
    const bool allowDuplicateSuppression = !forceMustExecute;
    {
        std::lock_guard<std::mutex> lock(rebuildMutex);

        if (hasPendingTask)
        {
            if (allowDuplicateSuppression)
            {
                BuildParameterSnapshot pendingSnapshot {};
                pendingSnapshot.ditherDepth = pendingTask.buildInput.ditherBitDepth;
                pendingSnapshot.oversamplingFactor = pendingTask.buildInput.oversamplingFactor;
                pendingSnapshot.oversamplingType = static_cast<OversamplingType>(pendingTask.buildInput.oversamplingType);
                pendingSnapshot.noiseShaperType = static_cast<NoiseShaperType>(pendingTask.buildInput.noiseShaperType);
                pendingSnapshot.processingOrder = static_cast<ProcessingOrder>(pendingTask.buildInput.processingOrder);
                pendingSnapshot.eqBypassed = pendingTask.buildInput.eqBypassed;
                pendingSnapshot.convBypassed = pendingTask.buildInput.convBypassed;
                pendingSnapshot.softClipEnabled = pendingTask.buildInput.softClipEnabled;
                pendingSnapshot.saturationAmount = pendingTask.buildInput.saturationAmount;
                pendingSnapshot.inputHeadroomGain = pendingTask.buildInput.inputHeadroomGain;
                pendingSnapshot.outputMakeupGain = pendingTask.buildInput.outputMakeupGain;
                pendingSnapshot.convolverInputTrimGain = pendingTask.buildInput.convolverInputTrimGain;

                const bool sameAsPending =
                    std::abs(pendingTask.buildInput.sampleRate - sampleRate) <= 1.0e-6
                    && pendingTask.buildInput.blockSize == samplesPerBlock
                    && equalsBuildParameterSnapshot(pendingSnapshot, paramSnapshot)
                    && pendingTask.convolverBuildSnapshot.fingerprint == task.convolverBuildSnapshot.fingerprint
                    && convo::isRuntimeBuildSnapshotSealedAndCompatible(pendingTask.runtimeBuildSnapshot,
                                                                        task.runtimeBuildSnapshot);

                if (sameAsPending)
                {
                    blockedAsDuplicate = true;
                }
                else
                {
                    currentToRelease = pendingTask.currentDSP;
                }
            }
            else
            {
                currentToRelease = pendingTask.currentDSP;
            }
        }

        if (!blockedAsDuplicate)
        {
            generation = ++rebuildRequestGeneration;
            task.generation = generation;
            task.runtimeBuildSnapshot = sealRuntimeBuildSnapshot(finalizeRuntimeBuildSnapshot(
                captureRuntimeBuildSnapshot(task.buildInput,
                                            task.convolverBuildSnapshot,
                                            generation,
                                            structuralHash,
                                            uiConvolverProcessor.isIRLoaded(),
                                            uiConvolverProcessor.isIRFinalized())));
            pendingTask = task;
            hasPendingTask = true;
            convo::publishAtomic(rebuildBacklog_, static_cast<std::uint64_t>(1), std::memory_order_release);
            lastQueuedTaskSignature = task;
            rtAuxMutable_.lastQueuedTaskTicks = juce::Time::getHighResolutionTicks();
            queued = true;
        }
    }

    if (queued)
    {
        convo::fetchAddAtomic(rtAuxMutable_.debugRebuildDispatchQueuedCount, 1, std::memory_order_acq_rel);
        rebuildCV.notify_all();
        diagLog("[DIAG] requestRebuild(sr,bs): task queued generation=" + juce::String(generation)
            + " SR=" + juce::String(sampleRate, 2));
        const double latencyMs = juce::Time::getMillisecondCounterHiRes() - requestStartMs;
        emitRebuildTelemetry(RebuildTelemetryEvent::Dispatched,
                     intentId,
                     RebuildTelemetryReason::TaskQueued,
                     RebuildTelemetryDecision::Dispatched,
                             structuralHash,
                             task.convolverBuildSnapshot.fingerprint,
                     RebuildTelemetryClass::Structural,
                 collapsePolicy,
                     "N/A",
                     latencyMs);
    }
    else
    {
        // Phase5: PendingDuplicate は維持対象（pending queue の過負荷抑止ガード）。
        // 直近縮退（RecentDuplicate / DeferredStructuralWindow）後も、
        // 同一 pending の無限置換を防ぐため merge 扱いを維持する。
        convo::fetchAddAtomic(rtAuxMutable_.debugRebuildDispatchBlockedPendingDuplicateCount, 1, std::memory_order_acq_rel);
        diagLog("[DIAG][PHASE5-KEEP] requestRebuild(sr,bs): BLOCKED duplicate pending task SR="
            + juce::String(sampleRate, 2));
        emitRebuildTelemetry(RebuildTelemetryEvent::Merged,
                             intentId,
                             RebuildTelemetryReason::PendingDuplicate,
                             RebuildTelemetryDecision::Merged,
                             structuralHash,
                             task.convolverBuildSnapshot.fingerprint,
                             RebuildTelemetryClass::Structural,
                             collapsePolicy,
                             kPhase5TagKeep);
                    convo::fetchAddAtomic(rebuildCollapseCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    }

    // Destroy orphaned DSP objects outside the lock.
    if (currentToRelease)
    {
        DSPLifetimeManager lifetimeMgr(*this);
        lifetimeMgr.retire(currentToRelease);
    }

    juce::ignoreUnused(queued);
}



void AudioEngine::stopRebuildThread()
{
    setShutdownPhase(ShutdownPhase::StopWorkers, "stopRebuildThread");

    // exit フラグを立てる（predicate が次に評価された時に break する）
    convo::publishAtomic(rebuildThreadShouldExit, true, std::memory_order_release);

    // 待機中のスレッドを確実に起こす
    rebuildCV.notify_all();

    if (rebuildThread.joinable())
        rebuildThread.join();

}



void AudioEngine::rebuildThreadLoop()
{
    affinityManager.applyCurrentThreadPolicy(ThreadType::HeavyBackground);

    // Set denormal handling modes for this thread. This is crucial for performance
    // in MKL VML and AVX/SSE operations, which can be significantly slowed down
    // by subnormal numbers. This setting is thread-local.
    vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    convo::publishAtomic(rebuildThreadIsRunning, true, std::memory_order_release);

    while (true)
    {
        try
        {
            RebuildTask task;
            {
                std::unique_lock<std::mutex> lock(rebuildMutex);
                rebuildCV.wait(lock, [this] { return hasPendingTask || convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire); });

                if (convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire)) break;
                if (isShutdownInProgress())
                {
                    hasPendingTask = false;
                    pendingTask.currentDSP = nullptr;
                    break;
                }

                // Copy task and clear pendingTask pointers to transfer ownership
                task = pendingTask;
                pendingTask.currentDSP = nullptr;

                hasPendingTask = false;
                convo::publishAtomic(rebuildBacklog_, static_cast<std::uint64_t>(0), std::memory_order_release);
            }

            struct DSPGuard
            {
                AudioEngine* owner;
                DSPCore* ptr;
                ~DSPGuard()
                {
                    if (owner != nullptr && ptr != nullptr)
                    {
                        DSPLifetimeManager lifetimeMgr(*owner);
                        lifetimeMgr.retire(ptr);
                    }
                }
            } dspGuard { this, nullptr };

            convo::RuntimeBuilder runtimeBuilder(*this);
            // ★ S-2: HealthState 参照を RuntimeBuilder に設定
            runtimeBuilder.setHealthStateRef(getHealthStateRef());

            // Helper to check obsolescence
            const auto isObsolete = [&] {
                return isRebuildObsolete(task.generation) || convo::consumeAtomic(rebuildThreadShouldExit, std::memory_order_acquire);
            };

            if (isObsolete())
                continue;

            if (!task.runtimeBuildSnapshot.sealed)
                continue;

            // 1. Prepare (メモリ確保)
            const double buildStartMs = juce::Time::getMillisecondCounterHiRes();
            convo::BuildResult buildResult = runtimeBuilder.build(task.runtimeBuildSnapshot.buildInput,
                                                                  task.convolverBuildSnapshot);
            const double buildElapsedMs = juce::Time::getMillisecondCounterHiRes() - buildStartMs;

            if (buildResult.runtime == nullptr)
            {
                diagLog("[DIAG] rebuildThreadLoop: RuntimeBuilder build failed generation="
                        + juce::String(task.generation)
                        + " error=" + juce::String(convo::toString(buildResult.error))
                        + " source=task-snapshot");
                continue;
            }

            dspGuard.ptr = buildResult.runtime;
            auto* newDSP = buildResult.runtime;

            if (isObsolete())
                continue;

            // 2. Rebuild IR if needed (Heavy operation)
            double rebuildIrElapsedMs = 0.0;
            if (newDSP->convolverRt().getIRLength() > 0)
            {
                if (isObsolete())
                    continue;
                const double rebuildIrStartMs = juce::Time::getMillisecondCounterHiRes();
                newDSP->convolverRt().rebuildAllIRsSynchronous(isObsolete);
                rebuildIrElapsedMs = juce::Time::getMillisecondCounterHiRes() - rebuildIrStartMs;
            }
            diagLog("[DIAG] rebuildThreadLoop: generation=" + juce::String(task.generation)
                + " build=" + juce::String(buildElapsedMs, 1) + "ms"
                + " rebuildIR=" + juce::String(rebuildIrElapsedMs, 1) + "ms");

            const auto warmupError = runtimeBuilder.validateWarmup(*newDSP);
            if (warmupError != convo::BuildError::None)
            {
                const bool retryable = shouldRetryWarmupFailure(*newDSP);
                diagLog("[DIAG] rebuildThreadLoop: RuntimeBuilder warmup failed generation="
                    + juce::String(task.generation)
                    + " error=" + juce::String(convo::toString(warmupError))
                    + " retryable=" + juce::String(static_cast<int>(retryable))
                    + " irLoaded=" + juce::String(static_cast<int>(newDSP->convolverRt().isIRLoaded()))
                    + " irFinalized=" + juce::String(static_cast<int>(newDSP->convolverRt().isIRFinalized()))
                    + " irLoading=" + juce::String(static_cast<int>(newDSP->convolverRt().isLoadingIR())));

                if (retryable)
                    submitRebuildIntent(convo::RebuildKind::Structural,
                                        RebuildTelemetryReason::RebuildThreadWarmupRetry,
                                        RebuildTelemetryClass::Structural,
                                        RebuildTelemetryPolicy::Replaceable);

                continue;
            }

            if (isObsolete())
                continue;

            // 4. Refresh Latency (Prevent pitch slide during fade-in)
            newDSP->convolverRt().refreshLatency();

            // 5. Fade In
            newDSP->ramps().fadeInSamplesLeft = DSPCore::FADE_IN_SAMPLES;

            // Log convolver pipeline status before commit
            {
                const auto& buildInput = task.runtimeBuildSnapshot.buildInput;
                juce::Logger::writeToLog("[CONV_STATUS] rebuildThreadLoop: generation="
                    + juce::String(task.generation)
                    + " irLoaded=" + juce::String(static_cast<int>(newDSP->convolverRt().isIRLoaded()))
                    + " irLen=" + juce::String(newDSP->convolverRt().getIRLength())
                    + " convBypass=" + juce::String(static_cast<int>(buildInput.convBypassed))
                    + " sr=" + juce::String(buildInput.sampleRate, 1)
                    + " osFactor=" + juce::String(buildInput.oversamplingFactor)
                    + " processingRate=" + juce::String(newDSP->sampleRate * static_cast<double>(newDSP->oversamplingFactor), 1));
            }

            // ★ Phase2: DSP構築完了後の RuntimeBuildSnapshot 投影値更新
            // PR-2 設計: snapshot 投影値は DSPCore の実値を持つべき。
            // buildInput.oversamplingFactor (=0 for Auto) を DSP 解決値で上書きする。
            // これにより dspProjection.oversamplingFactor が正しい値を持つ。
            //
            // NOTE: 現在は oversamplingFactor のみ修正。他の PR-2 投影フィールド
            // (irLoaded/irFinalized/structuralHash/sampleRate/baseLatencySamples) は
            // 以下の理由により意図的に deferred:
            // - irLoaded/irFinalized/structuralHash: UI ConvolverProcessor 由来だが
            //   CrossfadeAuthority の実用上問題ない
            // - sampleRate: buildInput 値が実質的に DSP 値と一致
            // - baseLatencySamples: 生産コードで未消費
            task.runtimeBuildSnapshot.oversamplingFactor = static_cast<int>(newDSP->oversamplingFactor);

            // 6. Commit on Message Thread
            // Release ownership from guard, pass to commitNewDSP
            DSPCore* dspToCommit = dspGuard.ptr;
            dspGuard.ptr = nullptr;
            enqueuePublicationIntentForRuntimeCommit(dspToCommit, task.generation, task.runtimeBuildSnapshot);
        }
        catch (const std::exception& e)
        {
            DBG("AudioEngine::rebuildThreadLoop exception: " << e.what());
            juce::ignoreUnused(e);
        }
        catch (...)
        {
            DBG("AudioEngine::rebuildThreadLoop unknown exception");
        }
    }

    convo::publishAtomic(rebuildThreadIsRunning, false, std::memory_order_release);
}
