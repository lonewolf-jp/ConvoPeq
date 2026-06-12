
//============================================================================
#pragma once
//============================================================================

#include <cstdint>

static constexpr int kAdaptiveNoiseShaperOrder = 9;
static constexpr int kAdaptiveNoiseShaperSampleRateBankCount = 10;

// BitDepth も一緒に管理するための拡張（16/24/32 の3段階）
static constexpr int kAdaptiveBitDepthCount = 3;
inline constexpr int kAdaptiveBitDepthValues[kAdaptiveBitDepthCount] = {16, 24, 32};
static constexpr int kLearningModeCount = 6;

// ストリーミング信号キャプチャ用 AudioBlock（2ch, 256サンプル）
struct AudioBlock {
    double L[256];
    double R[256];
    int numSamples = 0;
    int sampleRateHz = 0;
    int bitDepth = 0;
    int adaptiveCoeffBankIndex = 0;
    std::uint64_t sessionId = 0;
};

// RT/Worker間ダブルバッファ係数連携（RCU）
struct CoeffSet {
    static constexpr int kDim = kAdaptiveNoiseShaperOrder;
    double k[kDim] = {};
};

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <type_traits>
#include <vector>
#include <utility>
#include <juce_dsp/juce_dsp.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <immintrin.h>

#include "AlignedAllocation.h"
#include "CustomInputOversampler.h"
#include "ConvolverProcessor.h"
#include "EQProcessor.h"
#include "core/RCUReader.h"
#include "core/RuntimeReaderContext.h"
#include "EQEditProcessor.h"
#include "PsychoacousticDither.h"
#include "FixedNoiseShaper.h"
#include "Fixed15TapNoiseShaper.h"
#include "LatticeNoiseShaper.h"
#include "OutputFilter.h"
#include "DspNumericPolicy.h"
#include "UltraHighRateDCBlocker.h"
#include "LockFreeRingBuffer.h"
#include "LockFreeAudioRingBuffer.h"
#include "NoiseShaperLearnerTypes.h"
#include "GenerationManager.h"
#include "RuntimeBuildTypes.h"
#include "RuntimeTransition.h"
#include "AtomicAccess.h"
#include "CrossfadeRuntime.h"
#include "core/Types.h"
#include "core/SnapshotCoordinator.h"
#include "core/EpochDomain.h"
#include "core/RuntimePublicationCoordinator.h"
#include "core/CommandBuffer.h"
#include "core/ThreadAffinityManager.h"
#include "core/WorkerThread.h"
#include "core/RebuildTypes.h"
#include "ISRLifecycle.h"
#include "ISRRTExecution.h"
#include "ISRDSPHandle.h"
#include "ISRClosure.h"
#include "ISRAuthorityClass.h"
// RuntimePublicationOrchestrator は前方宣言 + unique_ptr で管理 (循環依存回避)
namespace convo::isr { class RuntimePublicationOrchestrator; }
namespace convo::isr { class ISRRetireRouter; }
#include "ISRRuntimeSemanticSchema.h"
#include "ISRRuntimeIdentityGenerators.h"
#include "ISRPayloadTier.h"
#include "ISRRetire.h"
#include "ISRShutdown.h"
#include "ISRRuntimePublicationCoordinator.h"
#include "ISRDSPQuarantine.h"
#include "ISRClosureGraphWalker.h"
#include "ISRDebugRuntime.h"
#include "ISRRetireRuntimeEx.h"
#include "RuntimeDrainAudit.h"
// ISRRetireRouter forward-declared below (reduce include chain for C1060)
#include "ISRBarrierOptimizer.h"
#include "ISREvidenceExporter.h"
#include "RuntimeHealthMonitor.h"
#include "RuntimePublicationValidator.h"
#include "WorldLifecycleAudit.h"

class NoiseShaperLearner;
class AudioEngine;
namespace convo { class RuntimeBuilder; }

// デバッグビルド時のみログを出力するマクロ
#if defined(JUCE_DEBUG) && !defined(NDEBUG)
    #define DBG_LOG(msg) juce::Logger::writeToLog(msg)
#else
    #define DBG_LOG(msg) ((void)0)
#endif

inline double absNoLibm(double x) noexcept
{
    // ISR: std::bit_cast の中間変数形式（union UB 排除）
    auto bits = std::bit_cast<std::uint64_t>(x);
    bits &= 0x7FFFFFFFFFFFFFFFULL;
    return std::bit_cast<double>(bits);
}

struct RuntimeState : convo::isr::SealedObject<RuntimeState>
{
#pragma warning(push)
#pragma warning(disable : 4996) // [[deprecated]] EngineRuntime — transitional, verifier-enforced
    struct BuilderToken
    {
    private:
        friend class AudioEngine;
        friend class convo::RuntimeBuilder;
        constexpr BuilderToken() noexcept = default;
    };

    RuntimeState() = delete;
    explicit RuntimeState(BuilderToken) noexcept
    {
    }

    RuntimeState(const RuntimeState&) = delete;
    RuntimeState& operator=(const RuntimeState&) = delete;
    RuntimeState(RuntimeState&&) = delete;
    RuntimeState& operator=(RuntimeState&&) = delete;

    [[nodiscard]] static convo::aligned_unique_ptr<RuntimeState> createForBuilder(BuilderToken token) noexcept
    {
        return convo::aligned_make_unique<RuntimeState>(token);
    }

    // AuthorityClass::Diagnostic (trace/correlation only, must not drive runtime branching)
    std::uint64_t worldId = 0;
    // AuthorityClass::Derived
    convo::EngineRuntime engine {};
    // AuthorityClass::Derived
    convo::RuntimeGraph graph {};
    // AuthorityClass::Authoritative
    std::uint64_t generation = 0;
    // AuthorityClass::Diagnostic (must not drive runtime branching)
    std::uint64_t runtimeVersion = 0;  // Diagnostic mirror (authoritative source is generation)
    // AuthorityClass::Diagnostic (trace/correlation only)
    std::uint64_t transitionId = 0;    // Unique identifier per crossfade

    // AuthorityClass::Authoritative (schema governance)
    std::uint32_t schemaVersion = convo::isr::kRuntimeSemanticSchemaVersion;
    // AuthorityClass::Authoritative (schema governance)
    convo::isr::RuntimeMetadata metadata {};

    // AuthorityClass::Derived (canonical source is `generation`)
    convo::isr::GenerationSemantic generationSemantic {};
    // AuthorityClass::Authoritative
    convo::isr::TopologySemantic topology {};
    // AuthorityClass::Authoritative
    convo::isr::RoutingSemantic routing {};
    // AuthorityClass::Authoritative
    convo::isr::ExecutionSemantic execution {};
    // AuthorityClass::Authoritative
    convo::isr::PublicationSemantic publication {};
    // AuthorityClass::Authoritative
    convo::isr::OverlapSemantic overlap {};
    // AuthorityClass::Authoritative
    convo::isr::RetireSemantic retire {};

    // AuthorityClass::Authoritative
    convo::isr::TimingSemantic timing {};
    // AuthorityClass::Authoritative
    convo::isr::LatencySemantic latency {};
    // SchedulingSemantic is deprecated (#16 Sprint-1) - fields are derived from ExecutionSemantic
    // Retained for backward compatibility during migration period
    // convo::isr::SchedulingSemantic scheduling {};  // Deprecated: use execution.* instead

    // AuthorityClass::Derived
    convo::isr::ResourceSemantic resource {};
    // AuthorityClass::Diagnostic
    convo::isr::AffinitySemantic affinity {};
    // AuthorityClass::Derived
    convo::isr::AutomationSemantic automation {};
    // AuthorityClass::Derived
    convo::isr::CoefficientSemantic coefficient {};

    // AuthorityClass::Derived (DSPCore → RuntimeWorld projection for crossfade/admission decisions)
    struct DSPSemanticProjection {
        bool irLoaded = false;
        bool irFinalized = false;
        uint64_t structuralHash = 0;
        int oversamplingFactor = 1;
        double sampleRate = 48000.0;
        int baseLatencySamples = 0;
    } dspProjection;

    // AuthorityClass::Diagnostic (projection freshness metadata)
    convo::isr::ProjectionFreshness projectionFreshness {};
    // AuthorityClass::Diagnostic (hash is fingerprint only; non-authoritative)
    convo::isr::RuntimeSemanticHash semanticHash {};

    static constexpr std::array<convo::isr::RuntimeFieldDescriptor, 21> kFieldDescriptors {{
        {"worldId", convo::isr::SemanticCategory::Diagnostic, convo::isr::OwnershipClass::DiagnosticOnly, convo::isr::MutabilityClass::DiagnosticMutable, convo::isr::VisibilityClass::DiagnosticBoundary, convo::isr::LifetimeClass::DiagnosticLifetime},
        {"generation", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"generationSemantic", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"topology", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"routing", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"execution", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"publication", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"overlap", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"metadata", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"retire", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"timing", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"latency", convo::isr::SemanticCategory::Authority, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::PublicationBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        // "scheduling" deprecated (#16 Sprint-1) - fields are derived from execution.*
        {"graph", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeGraph, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::ObserveBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"engine", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::ObserveBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"resource", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::ObserveBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"affinity", convo::isr::SemanticCategory::Diagnostic, convo::isr::OwnershipClass::DiagnosticOnly, convo::isr::MutabilityClass::DiagnosticMutable, convo::isr::VisibilityClass::DiagnosticBoundary, convo::isr::LifetimeClass::DiagnosticLifetime},
        {"automation", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::ObserveBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"coefficient", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::ObserveBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"dspProjection", convo::isr::SemanticCategory::Derived, convo::isr::OwnershipClass::RuntimeWorld, convo::isr::MutabilityClass::MutablePrePublish, convo::isr::VisibilityClass::ObserveBoundary, convo::isr::LifetimeClass::RuntimeWorldLifetime},
        {"projectionFreshness", convo::isr::SemanticCategory::Diagnostic, convo::isr::OwnershipClass::DiagnosticOnly, convo::isr::MutabilityClass::DiagnosticMutable, convo::isr::VisibilityClass::DiagnosticBoundary, convo::isr::LifetimeClass::DiagnosticLifetime},
        {"semanticHash", convo::isr::SemanticCategory::Diagnostic, convo::isr::OwnershipClass::DiagnosticOnly, convo::isr::MutabilityClass::DiagnosticMutable, convo::isr::VisibilityClass::DiagnosticBoundary, convo::isr::LifetimeClass::DiagnosticLifetime}
    }};

    static constexpr std::array<convo::isr::RuntimeAuthorityInventoryEntry, 21> kRuntimeAuthorityInventory {{
        {"worldId", convo::isr::RuntimeAuthorityClass::Diagnostic},
        {"generation", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"generationSemantic", convo::isr::RuntimeAuthorityClass::Derived},
        {"topology", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"routing", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"execution", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"publication", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"overlap", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"metadata", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"retire", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"timing", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"latency", convo::isr::RuntimeAuthorityClass::Authoritative},
        // "scheduling" deprecated (#16 Sprint-1) - fields are derived from execution.*
        {"graph", convo::isr::RuntimeAuthorityClass::Derived},
        {"engine", convo::isr::RuntimeAuthorityClass::Derived},
        {"resource", convo::isr::RuntimeAuthorityClass::Derived},
        {"affinity", convo::isr::RuntimeAuthorityClass::Diagnostic},
        {"automation", convo::isr::RuntimeAuthorityClass::Derived},
        {"coefficient", convo::isr::RuntimeAuthorityClass::Derived},
        {"dspProjection", convo::isr::RuntimeAuthorityClass::Derived},
        {"projectionFreshness", convo::isr::RuntimeAuthorityClass::Diagnostic},
        {"semanticHash", convo::isr::RuntimeAuthorityClass::Diagnostic}
    }};

    static constexpr const auto& kAuthorityInventory = kRuntimeAuthorityInventory;

    static constexpr std::array<convo::isr::RuntimeAuthorityInventoryEntry, 10> kRuntimeReadAuthorityInventory {{
        {"topology", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"routing", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"execution", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"publication", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"overlap", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"latency", convo::isr::RuntimeAuthorityClass::Authoritative},
        {"engine", convo::isr::RuntimeAuthorityClass::Derived},
        {"automation", convo::isr::RuntimeAuthorityClass::Derived},
        {"coefficient", convo::isr::RuntimeAuthorityClass::Derived},
        {"dspProjection", convo::isr::RuntimeAuthorityClass::Derived}
    }};

    [[nodiscard]] static constexpr bool validateDescriptorSet() noexcept
    {
        return convo::isr::validateFieldDescriptorSet(kFieldDescriptors)
            && convo::isr::validateAuthorityInventorySet(kRuntimeAuthorityInventory)
            && convo::isr::validateAuthorityInventoryAgainstDescriptors(kRuntimeAuthorityInventory, kFieldDescriptors)
            && convo::isr::validateReadAuthorityInventorySet(kRuntimeReadAuthorityInventory)
            && convo::isr::validateReadAuthorityInventoryAgainstDescriptors(kRuntimeReadAuthorityInventory, kFieldDescriptors)
            && convo::isr::PublicationSemantic::validateDescriptorSet();
    }
};

using RuntimePublishWorld = RuntimeState;
static_assert(!std::is_default_constructible_v<RuntimePublishWorld>,
              "RuntimePublishWorld must not be default-constructible outside builder path");

// ★ P0-2/3: Forward declarations for friend classes
namespace convo::isr {
    class PublicationExecutor;
    class DSPTransition;
}

// AudioEngine.h  ── v0.2 (JUCE 8.0.12対応)
//
// オーディオエンジン - AudioSource実装
//
// ■ JUCE AudioSource仕様:
//   - getNextAudioBlock(...) : オーディオ処理コールバック
//   - prepareToPlay(...) : 再生準備（バッファ確保など）
//   - releaseResources() : リソース解放（再生停止時）
//
// ■ スレッド安全性とリアルタイム制約:
//   - getNextAudioBlock: Audio Threadで実行されます。
//     - リアルタイム制約があります（ブロック不可、ロック不可、メモリ割り当て不可、IR再ロード不可）。
//   - prepareToPlay / releaseResources: Audio Thread の開始前/終了後に Message Thread から呼ばれます。
//   - パラメータ設定: Message Thread から呼ばれます。std::atomic を使用して Audio Thread と安全に同期します (RCUパターン)。
//   - readFromFifo: Message Thread (Timer) から呼ばれます。FIFOバッファからデータを取得します。
//============================================================================


class AudioEngine : public juce::AudioSource,
                                     public juce::ChangeBroadcaster,
                                     private juce::ChangeListener,
                                     private ConvolverProcessor::Listener,
                                     private juce::Timer,
                                     private juce::AsyncUpdater
{
public:
    friend class convo::RuntimeBuilder;
    using SampleType = double; // 内部DSP精度 (JUCE推奨)

    using ProcessingOrder = convo::ProcessingOrder;
    using OversamplingType = convo::OversamplingType;   // ここに追加
    using NoiseShaperType = convo::NoiseShaperType;     // ここに追加

    enum class AnalyzerSource
    {
        Input,
        Output
    };

    struct EngineParameterSnapshot
    {
        bool eqBypassed = false;
        bool convBypassed = false;
        ProcessingOrder order = ProcessingOrder::ConvolverThenEQ;
        AnalyzerSource analyzerSource = AnalyzerSource::Output;
        bool analyzerEnabled = false;
        bool softClipEnabled = false;
        float saturationAmount = 0.0f;
        double inputHeadroomGain = 1.0;
        double outputMakeupGain = 1.0;
        double convolverInputTrimGain = 1.0;
        convo::HCMode convHCMode = convo::HCMode::Natural;
        convo::LCMode convLCMode = convo::LCMode::Natural;
        convo::HCMode eqLPFMode = convo::HCMode::Natural;
        int adaptiveCoeffBankIndex = 0;
        const CoeffSet* adaptiveCoeffSet = nullptr;
        uint32_t adaptiveCoeffGeneration = 0;
        bool adaptiveCaptureEnabled = false;
        const convo::EQParameters* snapshotEqParams = nullptr;
        uint64_t snapshotEqCoeffHash = 0;
    };

    //----------------------------------------------------------
     // DSPコア (Audio Threadで実行される処理のコンテナ)
    //----------------------------------------------------------
    struct DSPCore
    {
        struct DCBlockerRuntimeState
        {
            convo::UltraHighRateDCBlocker outputL, outputR;
            convo::UltraHighRateDCBlocker inputL, inputR;
            convo::UltraHighRateDCBlocker oversampledL, oversampledR;

            void init(double sampleRate, double processingRate) noexcept
            {
                outputL.init(sampleRate, 3.0);
                outputR.init(sampleRate, 3.0);
                inputL.init(sampleRate, 3.0);
                inputR.init(sampleRate, 3.0);
                oversampledL.init(processingRate, 1.0);
                oversampledR.init(processingRate, 1.0);
            }

            void reset() noexcept
            {
                outputL.reset();
                outputR.reset();
                inputL.reset();
                inputR.reset();
                oversampledL.reset();
                oversampledR.reset();
            }
        };

        struct ConvolverRuntimeState
        {
            ConvolverProcessor* processor = nullptr;

            void bind(ConvolverProcessor& value) noexcept
            {
                processor = &value;
            }

            ConvolverProcessor& ref() noexcept
            {
                jassert(processor != nullptr);
                return *processor;
            }

            const ConvolverProcessor& ref() const noexcept
            {
                jassert(processor != nullptr);
                return *processor;
            }

            void prepare(AudioEngine* ownerEngine, double processingRate, int processingBlockSize) noexcept
            {
                auto& proc = ref();
                jassert(ownerEngine != nullptr);
                proc.setRcuProvider(*ownerEngine);
                proc.prepareToPlay(processingRate, processingBlockSize);
            }

            void resetForRuntime() noexcept
            {
                ref().reset();
            }

            void cleanupForRuntime() noexcept
            {
                ref().cleanup();
            }
        };

        struct EQRuntimeState
        {
            EQProcessor* processor = nullptr;

            void bind(EQProcessor& value) noexcept
            {
                processor = &value;
            }

            EQProcessor& ref() noexcept
            {
                jassert(processor != nullptr);
                return *processor;
            }

            const EQProcessor& ref() const noexcept
            {
                jassert(processor != nullptr);
                return *processor;
            }

            void prepare(double processingRate, int internalMaxBlock) noexcept
            {
                ref().prepareToPlay(processingRate, internalMaxBlock);
            }

            void resetForRuntime() noexcept
            {
                ref().reset();
            }

            void cleanupForRuntime() noexcept
            {
                ref().cleanup();
            }
        };

        struct RampRuntimeState
        {
            int fadeInSamplesLeft = 0;
            convo::LinearRamp bypassFadeGainDouble;
            bool bypassedDouble = false;

            void prepare(double sampleRate) noexcept
            {
                bypassFadeGainDouble.reset(sampleRate, 0.005);
                bypassFadeGainDouble.setCurrentAndTargetValue(1.0);
                bypassedDouble = false;
                fadeInSamplesLeft = 0;
            }

            void resetForRuntime() noexcept
            {
                bypassFadeGainDouble.setCurrentAndTargetValue(1.0);
                bypassedDouble = false;
                fadeInSamplesLeft = 0;
            }
        };

        struct HistoryRuntimeState
        {
            convo::ScopedAlignedPtr<double> fixedLatencyBufferL;
            convo::ScopedAlignedPtr<double> fixedLatencyBufferR;
            int fixedLatencyBufferSize = 0;
            int fixedLatencyWritePos = 0;
            int fixedLatencySamples = 0;
            double softClipPrevSample[2] = {0.0, 0.0};

            void clearSoftClipHistory() noexcept
            {
                softClipPrevSample[0] = 0.0;
                softClipPrevSample[1] = 0.0;
            }

            void configureFixedLatencySamples(int samples, int maxInternalBlockSize) noexcept
            {
                const int clamped = std::max(0, samples);
                fixedLatencySamples = clamped;
                fixedLatencyWritePos = 0;

                const int requiredSize = clamped + std::max(1, maxInternalBlockSize) + 2;
                if (requiredSize > fixedLatencyBufferSize || !fixedLatencyBufferL || !fixedLatencyBufferR)
                {
                    auto newDelayL = convo::makeAlignedArray<double>(static_cast<size_t>(requiredSize));
                    auto newDelayR = convo::makeAlignedArray<double>(static_cast<size_t>(requiredSize));

                    juce::FloatVectorOperations::clear(newDelayL.get(), requiredSize);
                    juce::FloatVectorOperations::clear(newDelayR.get(), requiredSize);

                    fixedLatencyBufferL = std::move(newDelayL);
                    fixedLatencyBufferR = std::move(newDelayR);
                    fixedLatencyBufferSize = requiredSize;
                }
                else if (fixedLatencyBufferSize > 0)
                {
                    juce::FloatVectorOperations::clear(fixedLatencyBufferL.get(), fixedLatencyBufferSize);
                    juce::FloatVectorOperations::clear(fixedLatencyBufferR.get(), fixedLatencyBufferSize);
                }
            }

            void resetForRuntime() noexcept
            {
                fixedLatencyWritePos = 0;
                if (fixedLatencyBufferL && fixedLatencyBufferSize > 0)
                    juce::FloatVectorOperations::clear(fixedLatencyBufferL.get(), fixedLatencyBufferSize);
                if (fixedLatencyBufferR && fixedLatencyBufferSize > 0)
                    juce::FloatVectorOperations::clear(fixedLatencyBufferR.get(), fixedLatencyBufferSize);
                clearSoftClipHistory();
            }
        };

        struct ProcessingState
        {
            bool eqBypassed;
            bool convBypassed;
            ProcessingOrder order;
            AnalyzerSource analyzerSource;
            bool analyzerEnabled;
            bool softClipEnabled;
            float saturationAmount;
            double inputHeadroomGain;
            double outputMakeupGain;
            double convolverInputTrimGain; // EQThenConvolver 時のコンボルバー入力トリム
            // 出力周波数フィルターモード
            convo::HCMode convHCMode;  // ① ハイカットモード
            convo::LCMode convLCMode;  // ① ローカットモード
            convo::HCMode eqLPFMode;   // ② EQローパスモード
            int adaptiveCoeffBankIndex;
            const CoeffSet* adaptiveCoeffSet;
            uint32_t adaptiveCoeffGeneration;
            int adaptiveCaptureSampleRateHz;
            int adaptiveCaptureBitDepth;
            uint64_t captureSessionId;
            LockFreeRingBuffer<AudioBlock, 4096>* adaptiveCaptureQueue;
            const convo::EQParameters* eqParams;
            const EQCoeffCache* eqCache;
            uint64_t eqCoeffHash;
        };

        DSPCore();
        DSPCore(const DSPCore&) = delete;
        DSPCore& operator=(const DSPCore&) = delete;

    ~DSPCore()
    {
        // Explicitly clean up convolver resources to ensure no WDL memory is leaked,
        // especially for instances that are destroyed from the trash bin.
        convolver.forceCleanup();
    }

    void prepare(double sampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType, NoiseShaperType selectedNoiseShaperType, AudioEngine* owner);
    void setFixedLatencySamples(int samples);
    void reset();
        void process(const juce::AudioSourceChannelInfo& bufferToFill, LockFreeAudioRingBuffer& analyzerFifo,
             std::atomic<float>* inputLevelLinear,
             std::atomic<float>* outputLevelLinear, const ProcessingState& state);
    void processToBuffer(const juce::AudioSourceChannelInfo& source,
                         juce::AudioBuffer<float>& destination,
                 LockFreeAudioRingBuffer& analyzerFifo,
                 std::atomic<float>* inputLevelLinear,
                 std::atomic<float>* outputLevelLinear,
                         const ProcessingState& state);
    void processDouble(juce::AudioBuffer<double>& buffer,
                   LockFreeAudioRingBuffer& analyzerFifo,
                   std::atomic<float>* inputLevelLinear,
                   std::atomic<float>* outputLevelLinear,
                       const ProcessingState& state);
    void processDoubleToBuffer(const juce::AudioBuffer<double>& source,
                               juce::AudioBuffer<double>& destination,
                       LockFreeAudioRingBuffer& analyzerFifo,
                       std::atomic<float>* inputLevelLinear,
                       std::atomic<float>* outputLevelLinear,
                               const ProcessingState& state);
        ConvolverProcessor convolver;
        EQProcessor eq;
        // DC blocker state is detached from DSP graph members.
        std::unique_ptr<DCBlockerRuntimeState> dcBlockerState;
        std::unique_ptr<ConvolverRuntimeState> convolverState;
        std::unique_ptr<EQRuntimeState> eqState;
        std::unique_ptr<RampRuntimeState> rampState;
        std::unique_ptr<HistoryRuntimeState> historyState;

        DCBlockerRuntimeState& dcBlockers() noexcept
        {
            jassert(dcBlockerState != nullptr);
            return *dcBlockerState;
        }

        const DCBlockerRuntimeState& dcBlockers() const noexcept
        {
            jassert(dcBlockerState != nullptr);
            return *dcBlockerState;
        }

        ConvolverProcessor& convolverRt() noexcept
        {
            jassert(convolverState != nullptr);
            return convolverState->ref();
        }

        const ConvolverProcessor& convolverRt() const noexcept
        {
            jassert(convolverState != nullptr);
            return convolverState->ref();
        }

        EQProcessor& eqRt() noexcept
        {
            jassert(eqState != nullptr);
            return eqState->ref();
        }

        const EQProcessor& eqRt() const noexcept
        {
            jassert(eqState != nullptr);
            return eqState->ref();
        }

        RampRuntimeState& ramps() noexcept
        {
            jassert(rampState != nullptr);
            return *rampState;
        }

        const RampRuntimeState& ramps() const noexcept
        {
            jassert(rampState != nullptr);
            return *rampState;
        }

        HistoryRuntimeState& histories() noexcept
        {
            jassert(historyState != nullptr);
            return *historyState;
        }

        const HistoryRuntimeState& histories() const noexcept
        {
            jassert(historyState != nullptr);
            return *historyState;
        }

        ::convo::PsychoacousticDither dither;
        ::convo::FixedNoiseShaper fixedNoiseShaper;
        ::convo::Fixed15TapNoiseShaper fixed15TapNoiseShaper;
        LatticeNoiseShaper adaptiveNoiseShaper;
        // 出力周波数フィルター (① ハイカット/ローカット / ② ローパス/ハイパス)
        convo::OutputFilter outputFilter;

        CustomInputOversampler oversampling;
        size_t oversamplingFactor = 1;
        OversamplingType activeOversamplingType = OversamplingType::IIR;
        int ditherBitDepth = 0; // DSPCore内でディザリング判定に使用
        NoiseShaperType noiseShaperType = NoiseShaperType::Psychoacoustic;
        uint32_t activeAdaptiveCoeffGeneration = 0;
        int activeAdaptiveCoeffBankIndex = -1;
        uint64_t currentCaptureSessionId = 0;
        std::uint64_t runtimeUuid = 0;
        double sampleRate = 0.0;

    // 【パッチ3】MKL用rawアライメントバッファ（vector完全排除・ガイドライン厳守）
        convo::ScopedAlignedPtr<double> alignedL;
        convo::ScopedAlignedPtr<double> alignedR;
        int alignedCapacity = 0;                  // 現在確保済み容量（再確保判定用）

        int maxSamplesPerBlock = 0;               // ホスト指定の入力ブロック上限（DSPCore::prepare() で設定）

        // ─────────────────────────────────────────────────────────────
        // 【Issue 3 修正】内部処理用最大バッファサイズ
        // 理由: Oversampling有効時（最大8x）、processSamplesUp() 後の
        //      ブロックサイズが SAFE_MAX_BLOCK_SIZE × 8 まで拡大しうるため。
        //      固定で ×8 確保することで RCU 再構築時の resize を回避する。
        // ─────────────────────────────────────────────────────────────
        int maxInternalBlockSize = 0;             // OS考慮後の最大サイズ（SAFE_MAX_BLOCK_SIZE × 8）
        static constexpr int FADE_IN_SAMPLES = 2048; // 42ms @ 48kHz
        // B2: processDouble 用のバイパスフェード状態
        convo::ScopedAlignedPtr<double> dryBypassBufferDoubleL;
        convo::ScopedAlignedPtr<double> dryBypassBufferDoubleR;
        int dryBypassCapacityDouble = 0;

        // Helpers
        float measureLevel (const juce::dsp::AudioBlock<const double>& block) const noexcept;
        void applyFixedLatencyDelay(double* dataL, double* dataR, int numSamples) noexcept;
        void pushToFifo(const juce::dsp::AudioBlock<const double>& block,
                        LockFreeAudioRingBuffer& analyzerFifo) const noexcept;
        // analyzerInputTap=true の場合、ヘッドルームゲイン適用前の raw 入力を
        // analyzerFifo にプッシュする。
        // これにより、インプットスペアナ/レベルメーターがヘッドルーム非適用の
        // "入力されたデータそのもの" を表示できる。
        float processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples,
                           double headroomGain,
                           bool analyzerInputTap,
                       LockFreeAudioRingBuffer& analyzerFifo) noexcept;
        void processOutput(const juce::AudioSourceChannelInfo& bufferToFill,
                           int numSamples,
                           const ProcessingState& state) noexcept;
        float processInputDouble(const juce::AudioBuffer<double>& buffer, int numSamples,
                                 double headroomGain,
                                 bool analyzerInputTap,
                           LockFreeAudioRingBuffer& analyzerFifo) noexcept;
        void processOutputDouble(juce::AudioBuffer<double>& buffer,
                                 int numSamples,
                                 const ProcessingState& state) noexcept;
    private:
        static std::atomic<std::uint64_t> runtimeUuidCounterStorage_;
        static std::atomic<std::uint64_t>& runtimeUuidCounter() noexcept;
        [[nodiscard]] static std::uint64_t reserveNextRuntimeUuid() noexcept;
        static double musicalSoftClip(double x, double threshold, double knee, double asymmetry) noexcept;
    };

    class Listener
    {
    public:
        virtual ~Listener() = default;
        virtual void eqSettingsChanged() = 0;
    };

    // FIFO設定
    static constexpr int FIFO_SIZE = 1048576;  // Lock-free FIFO サイズ (2^20, SAFE_MAX_BLOCK_SIZE * 8x OS をカバー)

    // EQ応答曲線計算用の定数
    static constexpr int   NUM_DISPLAY_BARS = 128;
    static constexpr float EQ_GAIN_EPSILON = 0.01f;          // ゲインがこれ以下なら無視
    static constexpr float EQ_UNITY_GAIN_EPSILON = 1.0e-5f;  // 1.0との比較用

    // ── 安全性制限 ──
    static constexpr double SAFE_MIN_SAMPLE_RATE = 8000.0;
    static constexpr double SAFE_MAX_SAMPLE_RATE = 768000.0;
    static constexpr int    SAFE_MAX_BLOCK_SIZE  = 65536; // 8x Oversampling対応のため拡張

    //----------------------------------------------------------
    // コンストラクタ
    //----------------------------------------------------------
    AudioEngine();
    ~AudioEngine() override;
    void initialize();

    //----------------------------------------------------------
    // AudioSource インターフェース
    //----------------------------------------------------------
    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override;
    void releaseResources() override;
    void getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill) override;

    // ==================================================================
    // EBR (Epoch-Based Reclamation) 基盤
    // ==================================================================
    void tryReclaimResources() noexcept;

    // RCU Reader Support
    [[nodiscard]] uint64_t snapshotRcuEpoch() noexcept;
    void enterRcuReader(int readerIndex) noexcept;
    void exitRcuReader(int readerIndex) noexcept;
    [[nodiscard]] uint64_t markRetireEpoch() noexcept;
    [[nodiscard]] uint64_t currentRetireEpoch() const noexcept;
    uint64_t advanceRetireEpoch() noexcept;
    [[nodiscard]] bool enqueueRetireEpochBounded(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept;
    [[nodiscard]] uint32_t activeEpochObserverCount() const noexcept;

    void processBlockDouble (juce::AudioBuffer<double>& buffer);
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void convolverParamsChanged(ConvolverProcessor* processor) override;
    void timerCallback() override;

    //----------------------------------------------------------
    // 外部インターフェース (Message Thread)
    //----------------------------------------------------------
    ConvolverProcessor& getConvolverProcessor() { return uiConvolverProcessor; }
    [[nodiscard]] const ConvolverProcessor& getConvolverProcessor() const { return uiConvolverProcessor; }
    EQEditProcessor& getEQProcessor() { return uiEqEditor; }
    ThreadAffinityManager& getAffinityManager() noexcept { return affinityManager; }
    [[nodiscard]] const ThreadAffinityManager& getAffinityManager() const noexcept { return affinityManager; }

    // ========================================================
    // EQ Parameter Wrappers (Thread-safe delegation to uiEqEditor)
    // ========================================================
    void setEQBandGain(int band, float gainDb) { uiEqEditor.setBandGain(band, gainDb); }
    void setEQBandFrequency(int band, float freq) { uiEqEditor.setBandFrequency(band, freq); }
    void setEQBandQ(int band, float q) { uiEqEditor.setBandQ(band, q); }
    void setEQBandEnabled(int band, bool enabled) { uiEqEditor.setBandEnabled(band, enabled); }
    void setEQBandType(int band, EQBandType type) { uiEqEditor.setBandType(band, type); }
    void setEQBandChannelMode(int band, EQChannelMode mode) { uiEqEditor.setBandChannelMode(band, mode); }
    void setEQTotalGain(float gainDb) { uiEqEditor.setTotalGain(gainDb); }
    void setEQAGCEnabled(bool enabled) { uiEqEditor.setAGCEnabled(enabled); }
    void setEQNonlinearSaturation(float value) noexcept { uiEqEditor.setNonlinearSaturation(value); }
    void setEQFilterStructure(EQProcessor::FilterStructure mode) noexcept { uiEqEditor.setFilterStructure(mode); }

    [[nodiscard]] EQBandParams getEQBandParams(int band) const { return uiEqEditor.getBandParams(band); }
    [[nodiscard]] EQBandType getEQBandType(int band) const { return uiEqEditor.getBandType(band); }
    [[nodiscard]] EQChannelMode getEQBandChannelMode(int band) const { return uiEqEditor.getBandChannelMode(band); }
    [[nodiscard]] float getEQTotalGain() const { return uiEqEditor.getTotalGain(); }
    [[nodiscard]] bool isEQAGCEnabled() const { return uiEqEditor.getAGCEnabled(); }
    [[nodiscard]] float getEQNonlinearSaturation() const noexcept { return uiEqEditor.getNonlinearSaturation(); }
    [[nodiscard]] EQProcessor::FilterStructure getEQFilterStructure() const noexcept { return uiEqEditor.getFilterStructure(); }
    [[nodiscard]] const void* getEQStateSnapshot() const { return uiEqEditor.getEQStateSnapshot(); }
    void resetEQToDefaults() { uiEqEditor.reset(); }

    // acquire: prepareToPlay/releaseResources の release と HB し、有効なサンプルレートを取得。
    [[nodiscard]] double getSampleRate() const { return consumeAtomic(currentSampleRate, std::memory_order_acquire); }
    [[nodiscard]] double getProcessingSampleRate() const;
    struct LatencyBreakdown
    {
        int oversamplingLatencyBaseRateSamples = 0;
        int convolverAlgorithmLatencyBaseRateSamples = 0;
        int convolverIRPeakLatencyBaseRateSamples = 0;
        int convolverTotalLatencyBaseRateSamples = 0;
        int totalLatencyBaseRateSamples = 0;
    };

    [[nodiscard]] LatencyBreakdown getCurrentLatencyBreakdown() const;
    [[nodiscard]] int getCurrentLatencySamples() const;
    [[nodiscard]] int getTotalLatencySamples() const;  // PDC 用エイリアス (getCurrentLatencySamples と同値)

    // 【Fix Bug #8】gainToDecibels (std::log10 / libm) を Audio Thread から排除。
    // Audio Thread は linear gain を inputLevelLinear / outputLevelLinear に格納し、
    // getter (UI Thread) で dB 変換する。
    // acquire: Audio Thread の release publishAtomic (inputLevelLinear/outputLevelLinear) と HB
    //          し、最新のレベル値を UI Thread から安全に取得する。
    [[nodiscard]] float getInputLevel() const
    {
        const float linear = consumeAtomic(inputLevelLinear, std::memory_order_acquire);
        return (linear > LEVEL_METER_MIN_MAG)
               ? juce::Decibels::gainToDecibels(linear)
               : LEVEL_METER_MIN_DB;
    }
    [[nodiscard]] float getOutputLevel() const
    {
        const float linear = consumeAtomic(outputLevelLinear, std::memory_order_acquire);
        return (linear > LEVEL_METER_MIN_MAG)
               ? juce::Decibels::gainToDecibels(linear)
               : LEVEL_METER_MIN_DB;
    }




    [[nodiscard]] int getFifoNumReady() const { return analyzerFifo.getAvailableSamples(); }
    void readFromFifo(float* dest, int numSamples);
    void skipFifo(int numSamples);

    void calcEQResponseCurve(float* outMagnitudesL, float* outMagnitudesR, const std::complex<double>* zArray, int numPoints, double sampleRate);

    // パラメータ設定 (Thread-safe)
    void setEqBypassRequested (bool shouldBypass);
    void setConvolverBypassRequested (bool shouldBypass);
    // 以下 4 getter は acquire: 対応 setter の release publishAtomic/compareExchange と HB し、
    // Message Thread から bypass 状態の最新値を安全に観測する。
    [[nodiscard]] bool isEqBypassRequested() const noexcept { return consumeAtomic(eqBypassRequested, std::memory_order_acquire); }
    [[nodiscard]] bool isConvolverBypassRequested() const noexcept { return consumeAtomic(convBypassRequested, std::memory_order_acquire); }
    [[nodiscard]] bool isEQBypassed() const noexcept { return consumeAtomic(eqBypassActive, std::memory_order_acquire); }
    [[nodiscard]] bool isConvolverBypassed() const noexcept { return consumeAtomic(convBypassActive, std::memory_order_acquire); }

    void setConvolverPhaseMode(ConvolverProcessor::PhaseMode mode);
    [[nodiscard]] ConvolverProcessor::PhaseMode getConvolverPhaseMode() const;

    void requestEqPreset (int presetIndex);
    void requestEqPresetFromText(const juce::File& file);
    void requestConvolverPreset (const juce::File& irFile);

    void requestLoadState (const juce::ValueTree& state);
    [[nodiscard]] juce::ValueTree getCurrentState() const;
    void beginBulkParameterRestore() noexcept;
    void endBulkParameterRestore(bool requestRebuildNow = true) noexcept;

    void setProcessingOrder(ProcessingOrder order);
    // acquire: setProcessingOrder の release と HB し、最新の ProcessingOrder を観測。
    [[nodiscard]] ProcessingOrder getProcessingOrder() const { return consumeAtomic(currentProcessingOrder, std::memory_order_acquire); }

    // release: UI スレッドからの設定を Audio Thread が acquire で観測できるよう公開。
    // acquire: 対応 setter の release と HB し、最新値を取得。
    void setAnalyzerSource(AnalyzerSource source) { publishAtomic(currentAnalyzerSource, source, std::memory_order_release); }
    [[nodiscard]] AnalyzerSource getAnalyzerSource() const { return consumeAtomic(currentAnalyzerSource, std::memory_order_acquire); }
    void setAnalyzerEnabled(bool enabled) noexcept { publishAtomic(analyzerEnabled, enabled, std::memory_order_release); }
    [[nodiscard]] bool isAnalyzerEnabled() const noexcept { return consumeAtomic(analyzerEnabled, std::memory_order_acquire); }

    [[nodiscard]] inline const convo::RuntimeGraph* getRuntimeGraph() const noexcept
    {
        const auto readToken = RuntimePublicationCoordinator::acquireReadToken(runtimeStore);
        const auto* world = RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore, readToken);
        return (world != nullptr) ? &world->graph : nullptr;
    }

    void setInputHeadroomDb(float db);
    [[nodiscard]] float getInputHeadroomDb() const;

    void setOutputMakeupDb(float db);
    [[nodiscard]] float getOutputMakeupDb() const;

    void setConvolverInputTrimDb(float db);
    [[nodiscard]] float getConvolverInputTrimDb() const;

    // Audio Thread command queue 経路を廃止し、
    // Message Thread 上の UI staging -> snapshot/rebuild 経路に統一する。
    void setConvolverMix(float value) noexcept
    {
        ASSERT_NON_RT_THREAD();
        uiConvolverProcessor.setMix(value);
        submitRebuildIntent(convo::RebuildKind::Structural,
                            RebuildTelemetryReason::EnqueueSnapshotCommand,
                            RebuildTelemetryClass::Snapshot,
                            RebuildTelemetryPolicy::Replaceable);
    }

    void setConvolverSmoothingTime(float timeSec) noexcept
    {
        ASSERT_NON_RT_THREAD();
        uiConvolverProcessor.setSmoothingTime(timeSec);
        submitRebuildIntent(convo::RebuildKind::Structural,
                            RebuildTelemetryReason::EnqueueSnapshotCommand,
                            RebuildTelemetryClass::Snapshot,
                            RebuildTelemetryPolicy::Replaceable);
    }

    void setConvolverTargetIRLength(float timeSec, bool manualOverride = false) noexcept;
    void setConvolverMixedTransitionStartHz(float hz) noexcept;
    void setConvolverMixedTransitionEndHz(float hz) noexcept;
    void setConvolverMixedPreRingTau(float tau) noexcept;
    void setConvolverRebuildDebounceMs(int ms) noexcept;
    void setConvolverTailMode(ConvolverProcessor::TailMode mode) noexcept;
    void setConvolverTailStartSec(float sec) noexcept;
    void setConvolverTailStrength(float strength) noexcept;
    void setConvolverTailL1L2Multiplier(int multiplier) noexcept;

    // Convolver State Tree & Cache Settings
    [[nodiscard]] juce::ValueTree getConvolverStateTree() const;
    void setConvolverStateTree(const juce::ValueTree& state);
    [[nodiscard]] int getConvolverTargetUpgradeFFTSize() const;
    void setConvolverTargetUpgradeFFTSize(int fftSize);
    [[nodiscard]] bool isConvolverProgressiveUpgradeEnabled() const;
    void setConvolverEnableProgressiveUpgrade(bool enabled);
    [[nodiscard]] int getConvolverMaxCacheEntries() const;
    void setConvolverMaxCacheEntries(int maxEntries);
    void clearConvolverCache();

    void setDitherBitDepth(int bitDepth);
    [[nodiscard]] int getDitherBitDepth() const;

    void setNoiseShaperType(NoiseShaperType type);
    [[nodiscard]] NoiseShaperType getNoiseShaperType() const;
    void requestRebuild(convo::RebuildKind kind) noexcept;
    void requestStructuredRebuildIntent(convo::RebuildKind kind) noexcept
    {
        submitRebuildIntent(kind,
                            RebuildTelemetryReason::RequestRebuildKindEntry,
                            RebuildTelemetryClass::Structural,
                            RebuildTelemetryPolicy::Replaceable);
    }
    void setFixedNoiseLogIntervalMs(int intervalMs) noexcept;
    [[nodiscard]] int getFixedNoiseLogIntervalMs() const noexcept;
    void setFixedNoiseWindowSamples(int windowSamples) noexcept;
    [[nodiscard]] int getFixedNoiseWindowSamples() const noexcept;

    void setIRFadeSamples(int samples) noexcept
    {
        const int clamped = juce::jlimit(0, 96000, samples);
        // release: m_irFadeSamples の更新を Audio Thread が acquire で観測できるよう公開。
        publishAtomic(m_irFadeSamples, clamped, std::memory_order_release);

        // acquire: prepareToPlay release と HB し、有効な currentSampleRate を取得。
        const double sr = consumeAtomic(currentSampleRate, std::memory_order_acquire);
        if (sr > 0.0)
        {
            const double fadeSec = (clamped > 0)
                ? (static_cast<double>(clamped) / sr)
                : 0.001;
            // release: m_irFadeTimeSec の更新を Audio Thread が acquire で観測できるよう公開。
            publishAtomic(m_irFadeTimeSec, fadeSec, std::memory_order_release);
        }
    }
    // release: 各 setXxx の release と HB する getter は Audio Thread が acquire で観測。
    void setEQFadeSamples(int samples) noexcept { publishAtomic(m_eqFadeSamples, samples, std::memory_order_release); }
    [[nodiscard]] int getIRFadeSamples() const noexcept { return consumeAtomic(m_irFadeSamples, std::memory_order_acquire); }
    [[nodiscard]] int getEQFadeSamples() const noexcept { return consumeAtomic(m_eqFadeSamples, std::memory_order_acquire); }
    [[nodiscard]] bool isFading() const noexcept { return m_coordinator.isFading(); }
    // release: IR 変更フラグを Audio Thread の acquire で観測できるよう公開。
    void setIRChangeFlag() noexcept { publishAtomic(m_pendingIRChange, true, std::memory_order_release); }

    // acquire: Audio Thread の release (debugLastCreatedEqHash publish) と HB し、診断ハッシュを取得。
    [[nodiscard]] uint64_t getLastCreatedEqHashForDebug() const noexcept { return consumeAtomic(rtAuxMutable_.debugLastCreatedEqHash, std::memory_order_acquire); }

    void setSoftClipEnabled(bool enabled);
    [[nodiscard]] bool isSoftClipEnabled() const;

    void setSaturationAmount(float amount);
    [[nodiscard]] float getSaturationAmount() const;

    [[nodiscard]] bool isShutdownInProgress() const noexcept
    {
        // ★ A-2.1: OR 判定を永久維持（EngineLifecycleState + ShutdownPhase の二重保護）
        //   EngineLifecycleState と ShutdownPhase は異なるライフサイクル視点であり、
        //   移行期間に乖離が発生し得る。OR 判定により「どちらかが shutdown 状態なら
        //   shutdown とみなす」安全側の判定を恒久的に維持する。
        //   ShutdownRuntime のみへの完全委譲は行わない。
        const auto state = consumeAtomic(lifecycleState, std::memory_order_acquire);
        const bool lifecycleShutdown = (state == EngineLifecycleState::Releasing
                                     || state == EngineLifecycleState::Destroyed);
        return lifecycleShutdown || shutdownRuntime_.isShutdownInProgress();
    }

    // [[deprecated("Use PublicationAdmission::evaluate() instead")]]
    // [[nodiscard]] bool acceptsRuntimePublication() const noexcept;
    [[nodiscard]] bool isFullyDrained() noexcept;
    [[nodiscard]] bool waitForDrain(int timeoutMs = 2000, int pollIntervalMs = 2) noexcept;

    // ★ A-2.5: シャットダウン完了条件の監査構造体を収集
    [[nodiscard]] convo::isr::RuntimeDrainAudit collectDrainAudit() noexcept;

    // ★ A-1.6: 3系統の隔離を1トランザクションとして実行
    bool quarantineSlot(uint32_t slot, uint64_t generation,
                        convo::isr::QuarantineReason reason) noexcept;

    void drainDeferredRetireQueues(bool allowDuringShutdown) noexcept;

    void setOversamplingFactor(int factor);
    [[nodiscard]] int getOversamplingFactor() const;

    void setOversamplingType(OversamplingType type);
    [[nodiscard]] OversamplingType getOversamplingType() const;

    // ────────────────────────────────────────────────────────────────
    // 出力周波数フィルター設定 (Thread-safe)
    //
    // convHCMode / convLCMode: ① コンボルバー最終段の場合に使用
    // eqLPFMode              : ② EQ最終段の場合に使用
    // ────────────────────────────────────────────────────────────────
    void setConvHCFilterMode(convo::HCMode mode) noexcept;
    [[nodiscard]] convo::HCMode getConvHCFilterMode() const noexcept;

    void setConvLCFilterMode(convo::LCMode mode) noexcept;
    [[nodiscard]] convo::LCMode getConvLCFilterMode() const noexcept;

    void setEqLPFFilterMode(convo::HCMode mode) noexcept;
    [[nodiscard]] convo::HCMode getEqLPFFilterMode() const noexcept;

    // --- Adaptiveノイズシェイパー学習サポート ---
    void startNoiseShaperLearning(convo::NoiseShaperLearningMode mode, bool resume = false);
    void stopNoiseShaperLearning();
    void setNoiseShaperLearningMode(convo::NoiseShaperLearningMode mode);
    // acquire: setNoiseShaperLearningMode の release と HB し、最新の LearningMode を取得。
    [[nodiscard]] convo::NoiseShaperLearningMode getNoiseShaperLearningMode() const { return consumeAtomic(pendingLearningMode, std::memory_order_acquire); }
    [[nodiscard]] bool isNoiseShaperLearning() const;
    [[nodiscard]] const convo::NoiseShaperLearnerProgress& getNoiseShaperLearningProgress() const;
    [[nodiscard]] int copyNoiseShaperLearningHistory(double* outScores, int maxPoints) const noexcept;
    // 学習ワーカーが記録したエラーメッセージを返す（UI 表示用）。エラーなしは nullptr。
    [[nodiscard]] const char* getNoiseShaperLearningError() const noexcept;
    [[nodiscard]] static int getAdaptiveSampleRateBankCount() noexcept;

    struct RuntimeLifecycleDiagnostics
    {
        std::uint64_t publishCount = 0;
        std::uint64_t retireCount = 0;
        std::uint64_t reclaimCount = 0;
        std::uint64_t lastCommittedGeneration = 0;
        std::uint64_t lastCommittedPublicationSequence = 0;
        std::uint64_t lastDroppedGeneration = 0;
    };

    struct RuntimeBackpressureTelemetry
    {
        std::uint64_t retireQueueDepth = 0;
        std::uint64_t fallbackQueueDepth = 0;
        std::uint64_t quarantineResident = 0;
        std::uint64_t publicationBacklog = 0; // Phase1-B: kept for ABI compat (always 0)
        std::uint64_t rebuildBacklog = 0;
        std::uint64_t saturationEnterCount = 0;
        std::uint64_t saturationExitCount = 0;
        std::uint64_t publicationRejectCount = 0;
        std::uint64_t rebuildCollapseCount = 0;
        int retirePressureLevel = 0;
        std::uint64_t retireEscalationCount = 0;
        std::uint64_t retireProtectiveModeEnterCount = 0;
        std::uint64_t maxRetireDeferralEpochs = 0;
        double maxRetireWallClockMs = 0.0;
        double reclaimLatency = 0.0;
    };

    enum class ResidencyAuthority : uint8_t
    {
        PublicationCoordinator = 0,
        DeferredDeleteFallback,
        EpochRetire,
        ShutdownDrain
    };

    [[nodiscard]] RuntimeLifecycleDiagnostics getRuntimeLifecycleDiagnostics() const noexcept
    {
        return {
            consumeAtomic(rtAuxMutable_.runtimePublishCount),
            consumeAtomic(rtAuxMutable_.runtimeRetireCount),
            consumeAtomic(rtAuxMutable_.runtimeReclaimCount),
            consumeAtomic(lastCommittedRuntimeGeneration_),
            consumeAtomic(lastCommittedPublicationSequence_),
            consumeAtomic(lastDroppedGeneration_)
        };
    }

    [[nodiscard]] convo::isr::PublicationSequenceId getLastCommittedPublicationSequence() const noexcept
    {
        return consumeAtomic(lastCommittedPublicationSequence_, std::memory_order_acquire);
    }

    [[nodiscard]] RuntimeBackpressureTelemetry getRuntimeBackpressureTelemetry() const noexcept
    {
        return {
            consumeAtomic(retireQueueDepth_, std::memory_order_acquire),
            consumeAtomic(fallbackQueueDepth_, std::memory_order_acquire),
            consumeAtomic(quarantineResident_, std::memory_order_acquire),
            static_cast<std::uint64_t>(0), // publicationBacklog_ removed in Phase1-B
            consumeAtomic(rebuildBacklog_, std::memory_order_acquire),
            consumeAtomic(saturationEnterCount_, std::memory_order_acquire),
            consumeAtomic(saturationExitCount_, std::memory_order_acquire),
            consumeAtomic(publicationRejectCount_, std::memory_order_acquire),
            consumeAtomic(rebuildCollapseCount_, std::memory_order_acquire),
            consumeAtomic(retirePressureLevel_, std::memory_order_acquire),
            consumeAtomic(retireEscalationCount_, std::memory_order_acquire),
            consumeAtomic(retireProtectiveModeEnterCount_, std::memory_order_acquire),
            consumeAtomic(maxRetireDeferralEpochs_, std::memory_order_acquire),
            consumeAtomic(maxRetireWallClockMs_, std::memory_order_acquire),
            consumeAtomic(reclaimLatency_, std::memory_order_acquire)
        };
    }

    struct CliProcessingTelemetrySnapshot
    {
        bool enabled = false;
        std::uint64_t callbackCount = 0;
        double lastProcessTimeUs = 0.0;
        double avgProcessTimeUs = 0.0;
        double maxProcessTimeUs = 0.0;
        int lastBlockSamples = 0;
        double sampleRateHz = 0.0;
    };

    struct RTLocalState
    {
        std::atomic<uint64_t> audioCallbackEpochCounter { 0 };
        std::atomic<uint64_t> audioSampleCursorCounter { 0 };
        std::atomic<uint32_t> audioCallbackActiveCount { 0 };
        std::atomic<uint64_t> audioThreadRetireEnqueueDropped { 0 };
    };

    struct RTAuxMutable
    {
        uint64_t debugLastReportedCreatedEqHash { std::numeric_limits<uint64_t>::max() };
        int debugLastReportedDspReady { -1 };
        int debugLastReportedTransitionActive { -1 };
        int debugLastReportedTransitionPolicy { -1 };
        uint64_t debugLastReportedTransitionCurrentPtr { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedTransitionNextPtr { std::numeric_limits<uint64_t>::max() };
        double debugLastReportedTransitionFadeSec { -1.0 };
        int debugLastReportedTransitionLatencyDeltaSamples { std::numeric_limits<int>::max() };
        uint64_t debugLastReportedRuntimeSnapshotRevision { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimePublishCurrentUuid { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimePublishFadingUuid { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimePublishTransitionCurrentUuid { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimePublishTransitionNextUuid { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimePublishCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimeRetireCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRuntimeReclaimCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildRequestCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildQueuedCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildBlockedPendingDuplicateCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildBlockedRecentDuplicateCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildRuntimeQueueFullCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildDrainedCommandCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildMatchedRuntimeCommandCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedRebuildTaskSnapshotFallbackCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedConvolverRebuildRequestCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedConvolverRebuildDeferredAfterLoadCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedConvolverRebuildScheduledCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedConvolverRebuildTriggeredCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedEqCacheSnapshotCreateMissCount { std::numeric_limits<uint64_t>::max() };
        uint64_t debugLastReportedEqCacheRuntimeLookupMissCount { std::numeric_limits<uint64_t>::max() };
        int debugLastReportedShutdownPhase { -1 };
        bool debugEqCacheLookupMissLatched { false };
        uint64_t debugEqCacheLookupMissLatchedHash { 0 };
        uint32 fixedNoiseLastLogMs { 0 };
        int64_t lastQueuedTaskTicks { 0 };
        std::atomic<std::uint64_t> rebuildTelemetryNextIntentId { 1 };
        std::atomic<std::int64_t> lastEvidenceEmitHighResTicks { 0 };
        std::atomic<std::uint64_t> runtimePublishCount { 0 };
        std::atomic<std::uint64_t> runtimeRetireCount { 0 };
        std::atomic<std::uint64_t> runtimeReclaimCount { 0 };
        std::atomic<uint64_t> lastRejectedGenerationNonRt { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchRequestCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchQueuedCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchBlockedPendingDuplicateCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchBlockedRecentDuplicateCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchRuntimeQueueFullCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchDrainedCommandCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchMatchedRuntimeCommandCount { 0 };
        std::atomic<std::uint64_t> debugRebuildDispatchTaskSnapshotFallbackCount { 0 };
        std::atomic<std::uint64_t> eqCacheSnapshotCreateMissCountNonRt { 0 };
        std::atomic<std::uint64_t> eqCacheRuntimeLookupMissCountNonRt { 0 };
        std::atomic<uint64_t> debugLastCreatedEqHash { 0 };
        std::atomic<uint64_t> lastEnqueuedSnapshotDebounceKey { 0 };
        std::atomic<bool> hasLastEnqueuedSnapshotDebounceKey { false };
        std::atomic<bool> cliProcessingTelemetryEnabled { false };
        std::atomic<std::uint64_t> cliTelemetryCallbackCount { 0 };
        std::atomic<double> cliTelemetryAccumulatedUs { 0.0 };
        std::atomic<double> cliTelemetryMaxUs { 0.0 };
        std::atomic<double> cliTelemetryLastUs { 0.0 };
        std::atomic<int> cliTelemetryLastBlockSamples { 0 };
    };

    void setCliProcessingTelemetryEnabled(bool enabled) noexcept
    {
        publishAtomic(rtAuxMutable_.cliProcessingTelemetryEnabled, enabled, std::memory_order_release);
        if (!enabled)
        {
            exchangeAtomic(rtAuxMutable_.cliTelemetryCallbackCount, static_cast<std::uint64_t>(0), std::memory_order_acq_rel);
            exchangeAtomic(rtAuxMutable_.cliTelemetryAccumulatedUs, 0.0, std::memory_order_acq_rel);
            exchangeAtomic(rtAuxMutable_.cliTelemetryMaxUs, 0.0, std::memory_order_acq_rel);
            exchangeAtomic(rtAuxMutable_.cliTelemetryLastUs, 0.0, std::memory_order_acq_rel);
            exchangeAtomic(rtAuxMutable_.cliTelemetryLastBlockSamples, 0, std::memory_order_acq_rel);
        }
    }

    [[nodiscard]] bool isCliProcessingTelemetryEnabled() const noexcept
    {
        return consumeAtomic(rtAuxMutable_.cliProcessingTelemetryEnabled, std::memory_order_acquire);
    }

    [[nodiscard]] CliProcessingTelemetrySnapshot consumeCliProcessingTelemetrySnapshot() noexcept
    {
        CliProcessingTelemetrySnapshot snapshot {};
        snapshot.enabled = isCliProcessingTelemetryEnabled();
        snapshot.callbackCount = exchangeAtomic(rtAuxMutable_.cliTelemetryCallbackCount, static_cast<std::uint64_t>(0), std::memory_order_acq_rel);
        const double accumulatedUs = exchangeAtomic(rtAuxMutable_.cliTelemetryAccumulatedUs, 0.0, std::memory_order_acq_rel);
        snapshot.maxProcessTimeUs = exchangeAtomic(rtAuxMutable_.cliTelemetryMaxUs, 0.0, std::memory_order_acq_rel);
        snapshot.lastProcessTimeUs = consumeAtomic(rtAuxMutable_.cliTelemetryLastUs, std::memory_order_acquire);
        snapshot.lastBlockSamples = consumeAtomic(rtAuxMutable_.cliTelemetryLastBlockSamples, std::memory_order_acquire);
        snapshot.sampleRateHz = consumeAtomic(currentSampleRate, std::memory_order_acquire);
        if (snapshot.callbackCount > 0)
            snapshot.avgProcessTimeUs = accumulatedUs / static_cast<double>(snapshot.callbackCount);
        return snapshot;
    }

    struct RebuildDispatchDiagnostics
    {
        std::uint64_t requestCount = 0;
        std::uint64_t queuedCount = 0;
        std::uint64_t blockedPendingDuplicateCount = 0;
        std::uint64_t blockedRecentDuplicateCount = 0;
        std::uint64_t runtimeQueueFullCount = 0;
        std::uint64_t drainedCommandCount = 0;
        std::uint64_t matchedRuntimeCommandCount = 0;
        std::uint64_t taskSnapshotFallbackCount = 0;
    };
    struct EqCacheMissDiagnostics
    {
        std::uint64_t snapshotCreateMissCount = 0;
        std::uint64_t runtimeLookupMissCount = 0;
    };

    [[nodiscard]] RebuildDispatchDiagnostics getRebuildDispatchDiagnostics() const noexcept
    {
        return {
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchRequestCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchQueuedCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchBlockedPendingDuplicateCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchBlockedRecentDuplicateCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchRuntimeQueueFullCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchDrainedCommandCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchMatchedRuntimeCommandCount),
            consumeAtomic(rtAuxMutable_.debugRebuildDispatchTaskSnapshotFallbackCount)
        };
    }
    [[nodiscard]] EqCacheMissDiagnostics getEqCacheMissDiagnostics() const noexcept
    {
        return {
            consumeAtomic(rtAuxMutable_.eqCacheSnapshotCreateMissCountNonRt, std::memory_order_acquire),
            consumeAtomic(rtAuxMutable_.eqCacheRuntimeLookupMissCountNonRt, std::memory_order_acquire)
        };
    }

    EqCacheMissDiagnostics consumeEqCacheMissDiagnostics() noexcept
    {
        EqCacheMissDiagnostics snapshot {};
        snapshot.snapshotCreateMissCount = exchangeAtomic(rtAuxMutable_.eqCacheSnapshotCreateMissCountNonRt,
                                                          static_cast<std::uint64_t>(0),
                                                          std::memory_order_acq_rel);
        snapshot.runtimeLookupMissCount = exchangeAtomic(rtAuxMutable_.eqCacheRuntimeLookupMissCountNonRt,
                                                         static_cast<std::uint64_t>(0),
                                                         std::memory_order_acq_rel);
        return snapshot;
    }

    // --- NoiseShaperLearner Settings ---
    [[nodiscard]] convo::NoiseShaperLearnerSettings getNoiseShaperLearnerSettings() const;
    void setNoiseShaperLearnerSettings(const convo::NoiseShaperLearnerSettings& settings);

    [[nodiscard]] static double getAdaptiveSampleRateBankHz(int bankIndex) noexcept;
    void getCurrentAdaptiveCoefficients(double* outCoeffs, int maxCoefficients) const noexcept;
    void getAdaptiveCoefficientsForSampleRate(double sampleRate, double* outCoeffs, int maxCoefficients) const noexcept;
    void setAdaptiveCoefficientsForSampleRate(double sampleRate, const double* coeffs, int numCoefficients);
    void getAdaptiveCoefficientsForSampleRateAndBitDepth(double sampleRate, int bitDepth, double* outCoeffs, int maxCoefficients) const noexcept;
    void setAdaptiveCoefficientsForSampleRateAndBitDepth(double sampleRate, int bitDepth, const double* coeffs, int numCoefficients);
    void setAdaptiveAutosaveCallback(std::function<void()> callback);
    void requestAdaptiveAutosave();
    // NoiseShaperLearner から学習済み係数を受け取るコールバック (Worker Thread)
    void storeLearnedCoeffs(const double* coeffs);

    // --- Adaptive ノイズシェイパー係数インデックス計算（UI スレッドからアクセス可能） ---
    [[nodiscard]] static int getAdaptiveCoeffBankIndex(double sampleRate, int bitDepth, convo::NoiseShaperLearningMode mode) noexcept;

    [[nodiscard]] bool getAdaptiveNoiseShaperState(int bankIndex, convo::NoiseShaperLearnerState& outState) const noexcept;
    void setAdaptiveNoiseShaperState(int bankIndex, const convo::NoiseShaperLearnerState& inState) noexcept;

private:
    void recordAudioCallbackProcessingStats(int numSamples, double processTimeUs) noexcept
    {
        if (!consumeAtomic(rtAuxMutable_.cliProcessingTelemetryEnabled, std::memory_order_relaxed))
            return;

        convo::fetchAddAtomic(rtAuxMutable_.cliTelemetryCallbackCount, static_cast<std::uint64_t>(1), std::memory_order_relaxed);
        publishAtomic(rtAuxMutable_.cliTelemetryLastBlockSamples, numSamples, std::memory_order_relaxed);
        publishAtomic(rtAuxMutable_.cliTelemetryLastUs, processTimeUs, std::memory_order_relaxed);

        double expectedAccum = consumeAtomic(rtAuxMutable_.cliTelemetryAccumulatedUs, std::memory_order_relaxed);
        while (!convo::compareExchangeAtomic(rtAuxMutable_.cliTelemetryAccumulatedUs,
                             expectedAccum,
                             expectedAccum + processTimeUs,
                             std::memory_order_relaxed,
                             std::memory_order_relaxed))
        {
        }

        double expectedMax = consumeAtomic(rtAuxMutable_.cliTelemetryMaxUs, std::memory_order_relaxed);
         while (processTimeUs > expectedMax
             && !convo::compareExchangeAtomic(rtAuxMutable_.cliTelemetryMaxUs,
                                  expectedMax,
                                  processTimeUs,
                                  std::memory_order_relaxed,
                                  std::memory_order_relaxed))
        {
        }
    }

    //==================================================
    // クロスフェード用レイテンシ整合バッファ（最大2秒@48kHz, double精度, L/R独立）
    // 64byte aligned double* buffers (RT外確保)
    // ===== 遅延整合バッファ・スレッド安全 =====
    double* latencyBufOldL = nullptr;
    double* latencyBufOldR = nullptr;
    double* latencyBufNewL = nullptr;
    double* latencyBufNewR = nullptr;
    int latencyBufSize = 0;
    // AudioThread専用（atomic不要）
    int latencyWritePos = 0;
    // 遅延値はatomicで管理（MessageThread→AudioThread）
    std::atomic<int> latencyDelayOld { 0 };
    std::atomic<int> latencyDelayNew { 0 };
    int dspCrossfadeStartDelayBlocks_RT = 0;
    bool dspCrossfadeArmed_RT = false;
    // バッファリセット要求（MessageThread→AudioThread）
    std::atomic<bool> latencyResetPending { false };
    static constexpr int kMaxLatencySamples = 1536000; // 最大2秒@768kHz対応
    static constexpr int MAX_LATENCY_ALIGN_SAMPLES = 96000 * 2; // 2秒@48kHz
    class EQCacheManager
    {
    public:
        explicit EQCacheManager(AudioEngine& ownerIn) noexcept;
        EQCoeffCache* getOrCreate(const convo::EQParameters& params,
                                  double sampleRate,
                                  int maxBlockSize,
                                  uint64_t generation);
        EQCoeffCache* get(uint64_t hash) noexcept;
        [[nodiscard]] bool containsNonRt(uint64_t hash) noexcept;
        void releaseCache(EQCoeffCache* cache) noexcept;
        ~EQCacheManager();

    private:
        struct CacheMap
        {
            explicit CacheMap(AudioEngine& ownerIn) noexcept
                : owner(&ownerIn)
            {
            }

            CacheMap(const CacheMap& other)
                : owner(other.owner)
            {
                for (const auto& entry : other.map)
                {
                    if (entry.second != nullptr)
                    {
                        // EBR: Using RefCountedDeferred for cache objects as they are shared
                        entry.second->addRef();
                    }

                    map.emplace(entry.first, entry.second);
                }
            }

            ~CacheMap()
            {
                jassert(owner != nullptr);
                for (auto& entry : map)
                {
                    if (entry.second != nullptr)
                        entry.second->release(*owner->m_retireRouter);
                }
            }

            AudioEngine* owner = nullptr;
            std::unordered_map<uint64_t, EQCoeffCache*> map;
        };

        const CacheMap* loadMap() noexcept
        {
            return AudioEngine::consumeAtomicPtr(cacheMapPtr);
        }

        void storeNewMap(CacheMap* newMap) noexcept;
        void drainDeferredMapsUnderLock() noexcept;
        [[nodiscard]] bool tryEnqueueDeferredMap(CacheMap* map) noexcept;

        AudioEngine& owner;
        std::mutex writeMutex;
        std::atomic<CacheMap*> cacheMapPtr { nullptr };
        std::vector<CacheMap*> enqueueFallbackMaps;
    };

    static double estimateOversamplingLatencySamples(int oversamplingFactor,
                                                     OversamplingType oversamplingType,
                                                     double baseSampleRate) noexcept;

public:
    //----------------------------------------------------------
    // 処理チェーンコンポーネント
    //----------------------------------------------------------
     // UI/State管理用のインスタンス (Audio Threadでは使用しない)
    ConvolverProcessor  uiConvolverProcessor;
    EQEditProcessor uiEqEditor;

    LockFreeAudioRingBuffer analyzerFifo;

    //----------------------------------------------------------
    // 状態管理
    //----------------------------------------------------------
    // active runtime DSP slot: 現行 DSP の非所有スロット。
    //            実際の解放は retireDSP() → deferred delete / retire queue で行う。
    convo::NonOwningPtr<DSPCore> activeRuntimeDSPSlot { nullptr };
    // fading runtime DSP slot: フェード中 DSP の非所有スロット。
    //               寿命は publish/retire の順序に従い、active runtime slot と独立して非所有で管理する。
    convo::NonOwningPtr<DSPCore> fadingRuntimeDSPSlot { nullptr };

    inline DSPCore* exchangeFadingRuntimeDSP(DSPCore* value) noexcept
    {


        DSPCore* previous = fadingRuntimeDSPSlot.get();
        fadingRuntimeDSPSlot.operator=(value);
        return previous;
    }

    [[nodiscard]] inline DSPCore* getActiveRuntimeDSP() const noexcept
    {
        return activeRuntimeDSPSlot.get();
    }

    inline void setActiveRuntimeDSP(DSPCore* value) noexcept
    {
        activeRuntimeDSPSlot = value;
    }

    [[nodiscard]] inline bool hasActiveRuntimeDSP() const noexcept
    {
        return getActiveRuntimeDSP() != nullptr;
    }

    inline DSPCore* releaseActiveRuntimeDSP() noexcept
    {
        DSPCore* activeRaw = getActiveRuntimeDSP();
        setActiveRuntimeDSP(nullptr);
        return activeRaw;
    }

    struct RuntimeReadHandle
    {
    public:
        RuntimeReadHandle(const RuntimeReadHandle&) = delete;
        RuntimeReadHandle& operator=(const RuntimeReadHandle&) = delete;
        RuntimeReadHandle(RuntimeReadHandle&&) noexcept = default;
        RuntimeReadHandle& operator=(RuntimeReadHandle&&) noexcept = default;

        [[nodiscard]] const RuntimePublishWorld* runtimeWorldPtr() const noexcept
        {
            return runtimeWorld_;
        }

        [[nodiscard]] const convo::GlobalSnapshot* observedSnapshotPtr() const noexcept
        {
            return observedSnapshot_.get();
        }

    private:
        friend class AudioEngine;

        RuntimeReadHandle(convo::ObservedRuntime&& observedSnapshotIn,
                          const RuntimePublishWorld* runtimeWorldIn) noexcept
            : observedSnapshot_(std::move(observedSnapshotIn))
            , runtimeWorld_(runtimeWorldIn)
        {
        }

        convo::ObservedRuntime observedSnapshot_;
        const RuntimePublishWorld* runtimeWorld_ = nullptr;
    };

    struct CrossfadePreparedSnapshot
    {
        bool pending = false;
        bool useDryAsOld = false;
        bool firstIrDryCrossfadePending = false;
        double fadeTimeSec = 0.0;
        int latencyDelayOld = 0;
        int latencyDelayNew = 0;
        int startDelayBlocks = 0;
        int dryHoldSamples = 0;
        bool latencyResetPending = false;
        double dryScaleTarget = 1.0;       // dry-as-old crossfade の IR スケール目標値
    };

    struct AudioCallbackAuthorityView
    {
        explicit AudioCallbackAuthorityView(CrossfadePreparedSnapshot preparedCrossfadeIn) noexcept
            : preparedCrossfade(preparedCrossfadeIn)
        {
        }

        CrossfadePreparedSnapshot preparedCrossfade {};
    };

    class RuntimePublicationBridge;

    convo::isr::RuntimeWorldIdGenerator runtimeWorldIdGenerator_ {};
    convo::isr::RuntimeGenerationGenerator runtimeGenerationGenerator_ {};
    std::atomic<convo::isr::PublicationSequenceId> publicationSequenceCounter_ { 0 };
    std::atomic<std::uint64_t> lastCommittedRuntimeGeneration_ { 0 };
    std::atomic<convo::isr::PublicationSequenceId> lastCommittedPublicationSequence_ { 0 };
    std::atomic<std::uint64_t> lastDroppedGeneration_ { 0 };
    std::atomic<std::uint64_t> publishedWorldCount_ { 0 };
    std::atomic<std::uint64_t> retiredWorldCount_ { 0 };
    std::atomic<std::uint64_t> oldestPublishedGeneration_ { 0 };
    std::atomic<std::uint64_t> youngestPublishedGeneration_ { 0 };
    std::atomic<std::uint64_t> oldestObservedGeneration_ { 0 };
    std::atomic<std::uint64_t> youngestObservedGeneration_ { 0 };
    std::atomic<std::uint64_t> oldestRetiredGeneration_ { 0 };
    std::atomic<std::uint64_t> oldestPendingGeneration_ { 0 };
    std::atomic<std::uint64_t> newestPendingGeneration_ { 0 };
    std::atomic<std::uint64_t> oldestRetirePendingGeneration_ { 0 };
    std::atomic<std::uint64_t> pendingRetireGenerationCount_ { 0 };
    std::atomic<double> oldestPendingAge_ { 0.0 };
    std::atomic<double> oldestPendingFirstSeenMs_ { 0.0 };
    std::array<CrossfadePreparedSnapshot, 2> crossfadePreparedSnapshots_ {};
    std::atomic<int> crossfadePreparedSnapshotIndex_ { 0 };

        // モード別フェード時間（秒）
        std::atomic<double> m_irFadeTimeSec { 0.080 };
        std::atomic<double> m_irLengthFadeTimeSec { 0.050 };
        std::atomic<double> m_phaseFadeTimeSec { 0.060 };
        std::atomic<double> m_directHeadFadeTimeSec { 0.010 };
        std::atomic<double> m_nucFilterFadeTimeSec { 0.030 };
        std::atomic<double> m_tailFadeTimeSec { 0.030 };
        std::atomic<double> m_osFadeTimeSec { 0.030 };
        std::atomic<int> m_crossfadeStartDelayBlocks { 1 };

    convo::isr::CrossfadeRuntime crossfadeRuntime_;
    juce::AudioBuffer<float> dspCrossfadeFloatBuffer;
    juce::AudioBuffer<double> dspCrossfadeDoubleBuffer;

    std::atomic<double> currentSampleRate{48000.0};
    // 【Fix Bug #8】linear gain を格納 (dB変換はgetInputLevel/getOutputLevelで行う)
    std::atomic<float> inputLevelLinear{0.0f};
    std::atomic<float> outputLevelLinear{0.0f};
    std::atomic<int>   maxSamplesPerBlock{4096};

    std::atomic<bool> eqBypassRequested { false };
    std::atomic<bool> convBypassRequested { false };
    std::atomic<bool> eqBypassActive   { false };
    std::atomic<bool> convBypassActive { false };

    enum class RebuildReason : uint32_t
    {
        None = 0u,
        StructuralFromNonMT = 1u << 0,
        DeferredStructural = 1u << 1,
        DeferredFinalizeAware = 1u << 2
    };

    std::atomic<uint32_t> rebuildReasonFlags_ { 0u };
    std::atomic<int64_t> deferredStructuralRebuildDueTicks_ { 0 };
    std::atomic<int64_t> deferredFinalizeFirstSeenTicks_ { 0 };
    std::atomic<bool> pendingChangeNotification { false };
    // 同一IR構造に対する Structural rebuild の多重発火を抑止する。
    // 値は「直近で rebuild を要求した UI 側 Convolver 構造ハッシュ」。
    std::atomic<uint64_t> lastIssuedConvolverStructuralHash_{ 0 };
    // active runtime slot 実体を直接読まずに判定するための、commit済み Convolver 構造スナップショット。
    std::atomic<uint64_t> lastCommittedConvolverStructuralHash_{ 0 };
    std::atomic<bool> lastCommittedConvolverHasIr_{ false };
    EQCacheManager eqCacheManager;
    std::atomic<ProcessingOrder> currentProcessingOrder{ProcessingOrder::ConvolverThenEQ};
    std::atomic<AnalyzerSource> currentAnalyzerSource { AnalyzerSource::Output };
    std::atomic<bool> analyzerEnabled { false };
    std::atomic<int> ditherBitDepth { 0 }; // 0 = 未初期化 (DeviceSettingsで最大値に設定される)
    std::atomic<NoiseShaperType> noiseShaperType { NoiseShaperType::Psychoacoustic };

    // 【False Sharing 防止】頻繁な UI 更新変数を独立キャッシュラインへ配置
    struct LearningCommand
    {
        enum class Type : uint8_t { Start, Stop, IRChanged, DSPReady };

        Type type = Type::Stop;
        bool resume = false;
        convo::NoiseShaperLearningMode mode = convo::NoiseShaperLearningMode::Short;
        uint64_t irGeneration = 0;
    };

    struct LearnerDispatchAction
    {
        enum class Type : uint8_t { Start, Stop };

        Type type = Type::Stop;
        bool resume = false;
        convo::NoiseShaperLearningMode mode = convo::NoiseShaperLearningMode::Short;
    };

    enum class LearningRuntimeState : uint8_t { Idle, WaitingForDSP, Running };

    static constexpr uint32_t learningCommandBufferSize = 128;
    static constexpr uint32_t learningCommandBufferMask = learningCommandBufferSize - 1;
    static constexpr uint32_t learnerDispatchBufferSize = 32;
    static constexpr uint32_t learnerDispatchBufferMask = learnerDispatchBufferSize - 1;

    #pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    #pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<convo::NoiseShaperLearningMode> pendingLearningMode { convo::NoiseShaperLearningMode::Short };
    alignas(64) std::atomic<bool> adaptiveCaptureActiveRt { false };
    alignas(64) std::atomic<uint64_t> globalCaptureSessionId { 1 };
    #pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

    LearningCommand learningCommandBuffer[learningCommandBufferSize] {};
    LearnerDispatchAction learnerDispatchBuffer[learnerDispatchBufferSize] {};
    std::atomic<bool> learnerDispatchOverflow { false };
    std::atomic<LearnerDispatchAction> lastFailedAction {};

    #pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    #pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<uint32_t> learningCommandWrite { 0 };  // Message/UI thread only
    alignas(64) std::atomic<uint32_t> learningCommandRead { 0 };   // Audio thread only
    alignas(64) std::atomic<uint32_t> learnerDispatchWrite { 0 };  // Audio thread only
    alignas(64) std::atomic<uint32_t> learnerDispatchRead { 0 };   // Message thread only
    #pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

    LearningRuntimeState learningRuntimeState = LearningRuntimeState::Idle;
    convo::NoiseShaperLearningMode requestedLearningMode = convo::NoiseShaperLearningMode::Short;
    bool requestedLearningResume = false;
    uint64_t requestedLearningGeneration = 0;
    uint64_t currentIRGeneration = 0; // Audio thread only
    uint64_t pendingIRGeneration = 0; // Message/UI thread only

    std::atomic<int> fixedNoiseLogIntervalMs { 2000 };
    std::atomic<int> fixedNoiseWindowSamples { 8192 };
    std::atomic<bool> softClipEnabled { true };

    #pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    #pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<float> saturationAmount { 0.1f };
    #pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

    std::atomic<int> manualOversamplingFactor { 0 }; // 0=Auto, 1=1x, 2=2x, 4=4x, 8=8x
    std::atomic<OversamplingType> oversamplingType { OversamplingType::IIR };

    #pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    #pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<float> inputHeadroomDb { -6.0f };
    alignas(64) std::atomic<double> inputHeadroomGain { 0.5011872336272722 }; // -6dB
    alignas(64) std::atomic<float> outputMakeupDb { 12.0f };
    alignas(64) std::atomic<double> outputMakeupGain { 3.981071705534972 }; // +12dB
    #pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

    std::atomic<int> rebuildRequestGeneration { 0 }; // 非同期リビルドの競合防止用
    std::atomic<int> lastCommittedRebuildGeneration { 0 }; // commit 完了済み世代

    #pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    #pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<float> convolverInputTrimDb { 0.0f };
    alignas(64) std::atomic<double> convolverInputTrimGain { 1.0 }; // 0 dB
    #pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

    bool m_isRestoringState { false }; // requestLoadState 中はデフォルトリセットを抑制 (Message Thread のみ)
    // 出力周波数フィルターモード (Thread-safe)
    std::atomic<convo::HCMode> convHCFilterMode { convo::HCMode::Natural }; // ① ハイカット
    std::atomic<convo::LCMode> convLCFilterMode { convo::LCMode::Natural }; // ① ローカット
    std::atomic<convo::HCMode> eqLPFFilterMode  { convo::HCMode::Natural }; // ② EQローパス

    // dB変換時の下限値
    static constexpr float LEVEL_METER_MIN_DB  = -120.0f;
    static constexpr float LEVEL_METER_MIN_MAG = 1e-6f;

    // EQ応答曲線計算用ワークバッファ (Message Thread/UI Threadで再利用)
    std::array<float, NUM_DISPLAY_BARS> eqTotalMagSqLBuffer;
    std::array<float, NUM_DISPLAY_BARS> eqTotalMagSqRBuffer;
    std::array<float, NUM_DISPLAY_BARS> eqBandMagSqBuffer;

    //----------------------------------------------------------
    // プライベートヘルパー (Message Thread のみ)
    //----------------------------------------------------------
    void applyDefaultsForCurrentMode();

    //----------------------------------------------------------
    // ヘルパー関数
    //----------------------------------------------------------
    // Note: This function performs memory allocation (including MKL) and other blocking operations
    // such as IR resampling. It MUST only be called from the message thread.
    // prepareToPlay() routes to the message thread before invoking this helper.
    void requestRebuild(double sampleRate, int samplesPerBlock, bool forceMustExecute = false);
    // [P1 Phase1-B] PublicationIntent/PublicationLog 完全削除。
    // 直接 commitNewDSP を呼び出す単一スロットの pending commit を使用。
    void enqueuePublicationIntentForRuntimeCommit(DSPCore* newDSP, int generation, const convo::RuntimeBuildSnapshot& sealedSnapshot);
    // acquire: requestRebuild の rebuildRequestGeneration 更新 release と HB し、
    //          リビルド世代が古いか否かを各スレッドから安全に判定。
    [[nodiscard]] bool isRebuildObsolete(int generation) const { return generation != consumeAtomic(rebuildRequestGeneration, std::memory_order_acquire); }
    bool enqueueLearningCommand(const LearningCommand& cmd) noexcept;
    [[nodiscard]] bool dequeueLearningCommand(LearningCommand& cmd) noexcept;
    bool enqueueLearnerDispatch(const LearnerDispatchAction& action) noexcept;
    [[nodiscard]] bool dequeueLearnerDispatch(LearnerDispatchAction& action) noexcept;
    void processLearningCommands() noexcept;
    void processDeferredLearningActions();
    void resetLearningControlState() noexcept;
    void processWithSnapshot(const juce::AudioSourceChannelInfo& bufferToFill,
                             const convo::GlobalSnapshot* snap,
                             bool isFadingTarget);
    [[nodiscard]] int evaluateRetirePressureLevelNoRt(std::uint64_t retireDepth,
                                                      int highWatermark) const noexcept;
    void applyRetirePressurePolicyNoRt(int retirePressureLevel,
                                       std::uint64_t retireDepth) noexcept;
    [[nodiscard]] bool shouldRejectRebuildAdmissionForPressure() const noexcept;
    void handleAsyncUpdate() override;

    void createSnapshotFromCurrentState(uint64_t generation);
    void initWorkerThread();
    void shutdownWorkerThread();

    enum class EngineLifecycleState : int
    {
        Unprepared = 0,
        Preparing,
        Prepared,
        Releasing,
        Destroyed
    };

    enum class ShutdownPhase : int
    {
        Running = 0,
        StopAcceptingWork,
        StopAudio,
        StopWorkers,
        ForceEpochAdvance,
        DrainRetire,
        Destroy
    };

    static const char* shutdownPhaseToString(ShutdownPhase phase) noexcept
    {
        switch (phase)
        {
            case ShutdownPhase::Running: return "RUNNING";
            case ShutdownPhase::StopAcceptingWork: return "STOP_ACCEPTING_WORK";
            case ShutdownPhase::StopAudio: return "STOP_AUDIO";
            case ShutdownPhase::StopWorkers: return "STOP_WORKERS";
            case ShutdownPhase::ForceEpochAdvance: return "FORCE_EPOCH_ADVANCE";
            case ShutdownPhase::DrainRetire: return "DRAIN_RETIRE";
            case ShutdownPhase::Destroy: return "DESTROY";
        }
        return "UNKNOWN";
    }

    void setShutdownPhase(ShutdownPhase nextPhase, const char* origin) noexcept
    {
        // acq_rel: 取得側 acquire で前フェーズの操作を観測し、
        //          解放側 release で新フェーズを全スレッドに公開。
        const ShutdownPhase previous = exchangeAtomic(shutdownPhase, nextPhase, std::memory_order_acq_rel);
        if (previous == nextPhase)
            return;

        const juce::String log = "[DIAG] shutdown phase: "
            + juce::String(shutdownPhaseToString(previous))
            + " -> "
            + juce::String(shutdownPhaseToString(nextPhase))
            + " at "
            + juce::String(origin != nullptr ? origin : "unknown");
        DBG(log);
        juce::Logger::writeToLog(log);
    }

    // Worker thread for rebuilds
    void rebuildThreadLoop();
    void stopRebuildThread();
    std::thread rebuildThread;
    std::mutex rebuildMutex;
    std::condition_variable rebuildCV;
    std::atomic<bool> rebuildThreadShouldExit { false };
    std::atomic<bool> rebuildThreadIsRunning { false };
    std::atomic<ShutdownPhase> shutdownPhase { ShutdownPhase::Running };
    std::atomic<EngineLifecycleState> lifecycleState { EngineLifecycleState::Unprepared };
    bool hasPendingTask = false;

    struct RebuildTask {
        DSPCore* currentDSP = nullptr;
        convo::BuildInput buildInput {};
        ConvolverProcessor::BuildSnapshot convolverBuildSnapshot {};
        convo::RuntimeBuildSnapshot runtimeBuildSnapshot {};
        int generation = 0;
    };
    RebuildTask pendingTask;
    RebuildTask lastQueuedTaskSignature;

    // --- Adaptiveノイズシェイパー学習用メンバー ---
    struct AdaptiveCoeffBankSlot
    {
        double sampleRateHz = 0.0;
        CoeffSet coeffSetA {};
        CoeffSet coeffSetB {};
        std::atomic<int> activeIndex { 0 };   // 0 = A, 1 = B
        std::atomic<uint32_t> generation { 1u };
        convo::NoiseShaperLearnerState state {};
        std::mutex stateMutex;
        std::atomic<bool> writeLock { false };  // CAS用書き込みロック
    };

    LockFreeRingBuffer<AudioBlock, 4096> audioCaptureQueue;
    std::unique_ptr<NoiseShaperLearner> noiseShaperLearner;
    ThreadAffinityManager affinityManager;
    std::array<AdaptiveCoeffBankSlot, kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount> adaptiveCoeffBanks {};
    std::atomic<int> currentAdaptiveCoeffBankIndex { 1 };
    std::mutex adaptiveAutosaveCallbackMutex;
    std::function<void()> adaptiveAutosaveCallback;

    // [PR-3] Old pending commit slot removed. Orchestrator handles deferred commits.

    static int resolveAdaptiveCoeffBankIndex(double sampleRate) noexcept;
    static int getAdaptiveBitDepthIndex(int bitDepth) noexcept;
    AdaptiveCoeffBankSlot& getAdaptiveCoeffBankForIndex(int bankIndex) noexcept;
    [[nodiscard]] const AdaptiveCoeffBankSlot& getAdaptiveCoeffBankForIndex(int bankIndex) const noexcept;
    void selectAdaptiveCoeffBankForCurrentSettings() noexcept;
    void storeLearnedCoeffsToBank(int bankIndex, const double* coeffs);

    static constexpr uint32_t rebuildReasonMask(RebuildReason reason) noexcept
    {
        return static_cast<uint32_t>(reason);
    }

    [[nodiscard]] bool hasRebuildReason(RebuildReason reason) const noexcept
    {
        // acquire: setRebuildReason の acq_rel release-side と HB し、最新の flags を観測。
        const uint32_t flags = convo::consumeAtomic(rebuildReasonFlags_, std::memory_order_acquire);
        return (flags & rebuildReasonMask(reason)) != 0u;
    }

    bool setRebuildReason(RebuildReason reason) noexcept
    {
        // acq_rel: 取得側 acquire で直前の clearRebuildReason を観測し、
        //          解放側 release で新 flag を hasRebuildReason の acquire に公開。
        const uint32_t oldFlags = convo::fetchOrAtomic(rebuildReasonFlags_, rebuildReasonMask(reason), std::memory_order_acq_rel);
        return (oldFlags & rebuildReasonMask(reason)) == 0u;
    }

    bool clearRebuildReason(RebuildReason reason) noexcept
    {
        // acq_rel: 取得側 acquire で直前の setRebuildReason を観測し、
        //          解放側 release でクリア後の状態を hasRebuildReason acquire に公開。
        const uint32_t oldFlags = convo::fetchAndAtomic(rebuildReasonFlags_, static_cast<uint32_t>(~rebuildReasonMask(reason)), std::memory_order_acq_rel);
        return (oldFlags & rebuildReasonMask(reason)) != 0u;
    }

    [[nodiscard]] uint64_t nextRebuildTelemetryIntentId() noexcept
    {
        return convo::fetchAddAtomic(rtAuxMutable_.rebuildTelemetryNextIntentId, static_cast<uint64_t>(1), std::memory_order_acq_rel);
    }

    enum class RebuildTelemetryEvent : uint8_t
    {
        Requested,
        Merged,
        Suppressed,
        Deferred,
        ForcedDispatch,
        Dispatched
    };

    enum class RebuildTelemetryReason : uint8_t
    {
        ConvolverParamsChanged,
        MixedPhaseIntermediate,
        HashDedup,
        PreparedIRApplyWindow,
        SnapshotEnqueueFailed,
        SnapshotEnqueued,
        RequestRebuildKindEntry,
        UiEqEditorChangeListener,
        PrepareToPlayNonMt,
        RebuildThreadWarmupRetry,
        ShutdownInProgress,
        KindFiltered,
        DelegateRequestRebuildSrBs,
        MissingSrBs,
        NonMtTriggerAsync,
        NonMtAlreadyPending,
        AsyncBridgeConsume,
        AsyncBridgeDelegateSrBs,
        AsyncBridgeMissingSrBs,
        RequestRebuildSrBs,
        DeferredStructuralWindow,
        TaskQueued,
        RecentDuplicate,
        PendingDuplicate,
        DeferredStructuralDue,
        DeferredStructuralRebuildRequested,
        DeferredFinalizeReady,
        DeferredFinalizeRebuildRequested,
        EnqueueSnapshotCommand,
        SnapshotIntentDebounced,
        SnapshotCommandBufferFull,
        SnapshotCommandQueued,
        SnapshotCommandBufferFullNonMt,
        SnapshotCommandQueuedNonMt,
        RetirePressureSevere,
        SameAsPendingWouldMerge
    };

    enum class RebuildTelemetryClass : uint8_t
    {
        NA,
        Structural,
        FinalizeAware,
        Snapshot
    };

    enum class RebuildTelemetryPolicy : uint8_t
    {
        NA,
        Replaceable,
        MustExecute
    };
    enum class RebuildTelemetryDecision : uint8_t
    {
        Accepted,
        Suppressed,
        Deferred,
        Dispatched,
        Merged,
        Dropped,
        Released
    };

    static const char* toTelemetryEventString(RebuildTelemetryEvent value) noexcept
    {
        switch (value)
        {
            case RebuildTelemetryEvent::Requested: return "REBUILD_REQUESTED";
            case RebuildTelemetryEvent::Merged: return "REBUILD_MERGED";
            case RebuildTelemetryEvent::Suppressed: return "REBUILD_SUPPRESSED";
            case RebuildTelemetryEvent::Deferred: return "REBUILD_DEFERRED";
            case RebuildTelemetryEvent::ForcedDispatch: return "REBUILD_FORCED_DISPATCH";
            case RebuildTelemetryEvent::Dispatched: return "REBUILD_DISPATCHED";
        }
        return "REBUILD_UNKNOWN";
    }

    static const char* toTelemetryReasonString(RebuildTelemetryReason value) noexcept
    {
        switch (value)
        {
            case RebuildTelemetryReason::ConvolverParamsChanged: return "convolver_params_changed";
            case RebuildTelemetryReason::MixedPhaseIntermediate: return "mixed_phase_intermediate";
            case RebuildTelemetryReason::HashDedup: return "hash_dedup";
            case RebuildTelemetryReason::PreparedIRApplyWindow: return "prepared_ir_apply_window";
            case RebuildTelemetryReason::SnapshotEnqueueFailed: return "snapshot_enqueue_failed";
            case RebuildTelemetryReason::SnapshotEnqueued: return "snapshot_enqueued";
            case RebuildTelemetryReason::RequestRebuildKindEntry: return "requestRebuild_kind_entry";
            case RebuildTelemetryReason::UiEqEditorChangeListener: return "ui_eq_editor_change_listener";
            case RebuildTelemetryReason::PrepareToPlayNonMt: return "prepare_to_play_non_mt";
            case RebuildTelemetryReason::RebuildThreadWarmupRetry: return "rebuild_thread_warmup_retry";
            case RebuildTelemetryReason::ShutdownInProgress: return "shutdown_in_progress";
            case RebuildTelemetryReason::KindFiltered: return "kind_filtered";
            case RebuildTelemetryReason::DelegateRequestRebuildSrBs: return "delegate_requestRebuild_sr_bs";
            case RebuildTelemetryReason::MissingSrBs: return "missing_sr_bs";
            case RebuildTelemetryReason::NonMtTriggerAsync: return "non_mt_trigger_async";
            case RebuildTelemetryReason::NonMtAlreadyPending: return "non_mt_already_pending";
            case RebuildTelemetryReason::AsyncBridgeConsume: return "async_bridge_consume";
            case RebuildTelemetryReason::AsyncBridgeDelegateSrBs: return "async_bridge_delegate_sr_bs";
            case RebuildTelemetryReason::AsyncBridgeMissingSrBs: return "async_bridge_missing_sr_bs";
            case RebuildTelemetryReason::RequestRebuildSrBs: return "requestRebuild_sr_bs";
            case RebuildTelemetryReason::DeferredStructuralWindow: return "deferred_structural_window";
            case RebuildTelemetryReason::TaskQueued: return "task_queued";
            case RebuildTelemetryReason::RecentDuplicate: return "recent_duplicate";
            case RebuildTelemetryReason::PendingDuplicate: return "pending_duplicate";
            case RebuildTelemetryReason::DeferredStructuralDue: return "deferred_structural_due";
            case RebuildTelemetryReason::DeferredStructuralRebuildRequested: return "deferred_structural_rebuild_requested";
            case RebuildTelemetryReason::DeferredFinalizeReady: return "deferred_finalize_ready";
            case RebuildTelemetryReason::DeferredFinalizeRebuildRequested: return "deferred_finalize_rebuild_requested";
            case RebuildTelemetryReason::EnqueueSnapshotCommand: return "enqueue_snapshot_command";
            case RebuildTelemetryReason::SnapshotIntentDebounced: return "snapshot_intent_debounced";
            case RebuildTelemetryReason::SnapshotCommandBufferFull: return "snapshot_command_buffer_full";
            case RebuildTelemetryReason::SnapshotCommandQueued: return "snapshot_command_queued";
            case RebuildTelemetryReason::SnapshotCommandBufferFullNonMt: return "snapshot_command_buffer_full_non_mt";
            case RebuildTelemetryReason::SnapshotCommandQueuedNonMt: return "snapshot_command_queued_non_mt";
            case RebuildTelemetryReason::RetirePressureSevere: return "retire_pressure_severe";
            case RebuildTelemetryReason::SameAsPendingWouldMerge: return "same_as_pending_would_merge";
        }
        return "unknown_reason";
    }

    static const char* toTelemetryClassString(RebuildTelemetryClass value) noexcept
    {
        switch (value)
        {
            case RebuildTelemetryClass::NA: return "N/A";
            case RebuildTelemetryClass::Structural: return "Structural";
            case RebuildTelemetryClass::FinalizeAware: return "FinalizeAware";
            case RebuildTelemetryClass::Snapshot: return "Snapshot";
        }
        return "N/A";
    }

    static const char* toTelemetryPolicyString(RebuildTelemetryPolicy value) noexcept
    {
        switch (value)
        {
            case RebuildTelemetryPolicy::NA: return "N/A";
            case RebuildTelemetryPolicy::Replaceable: return "Replaceable";
            case RebuildTelemetryPolicy::MustExecute: return "MustExecute";
        }
        return "N/A";
    }

    static const char* toTelemetryDecisionString(RebuildTelemetryDecision value) noexcept
    {
        switch (value)
        {
            case RebuildTelemetryDecision::Accepted: return "accepted";
            case RebuildTelemetryDecision::Suppressed: return "suppressed";
            case RebuildTelemetryDecision::Deferred: return "deferred";
            case RebuildTelemetryDecision::Dispatched: return "dispatched";
            case RebuildTelemetryDecision::Merged: return "merged";
            case RebuildTelemetryDecision::Dropped: return "dropped";
            case RebuildTelemetryDecision::Released: return "released";
        }
        return "unknown";
    }

    void emitRebuildTelemetry(RebuildTelemetryEvent eventName,
                              uint64_t intentId,
                              RebuildTelemetryReason reason,
                              RebuildTelemetryDecision decision,
                              uint64_t hash = 0,
                              uint64_t fingerprint = 0,
                              RebuildTelemetryClass rebuildClass = RebuildTelemetryClass::NA,
                              RebuildTelemetryPolicy collapsePolicy = RebuildTelemetryPolicy::NA,
                              const char* finalizeState = "N/A",
                              double latencyMs = -1.0) const noexcept
    {
        juce::String log = "[REBUILD_TELEMETRY] event=";
        log += juce::String(toTelemetryEventString(eventName));
        log += " intentId=" + juce::String(static_cast<juce::int64>(intentId));
        log += " reason=" + juce::String(toTelemetryReasonString(reason));
        log += " class=" + juce::String(toTelemetryClassString(rebuildClass));
        log += " policy=" + juce::String(toTelemetryPolicyString(collapsePolicy));
        log += " hash=0x" + juce::String::toHexString(static_cast<juce::int64>(hash));
        log += " fingerprint=0x" + juce::String::toHexString(static_cast<juce::int64>(fingerprint));
        log += " finalizeState=" + juce::String(finalizeState != nullptr ? finalizeState : "N/A");
        log += " decision=" + juce::String(toTelemetryDecisionString(decision));
        if (latencyMs >= 0.0)
            log += " latencyMs=" + juce::String(latencyMs, 3);

        DBG(log);
        juce::Logger::writeToLog(log);
    }

    void submitRebuildIntent(convo::RebuildKind kind,
                             RebuildTelemetryReason reason,
                             RebuildTelemetryClass rebuildClass = RebuildTelemetryClass::NA,
                             RebuildTelemetryPolicy collapsePolicy = RebuildTelemetryPolicy::NA) noexcept;

    struct RebuildAdmissionIntentState
    {
        bool valid = false;
        convo::RebuildKind kind = convo::RebuildKind::None;
        RebuildTelemetryClass rebuildClass = RebuildTelemetryClass::NA;
        RebuildTelemetryPolicy collapsePolicy = RebuildTelemetryPolicy::NA;
        std::uint32_t fingerprintVersion = 1u;
        uint64_t structuralHash = 0;
        uint64_t fingerprint = 0;
        bool deferCategory = false;
        int64_t lastIntentTicks = 0;
    };



    void debugAssertNotAudioThread() const;
    void debugAssertAudioThread() const;

    inline convo::EngineRuntime makeEngineRuntimeState(DSPCore* current,
                                                        DSPCore* next,
                                                        convo::TransitionPolicy policy,
                                                        double fadeTimeSec,
                                                        bool active,
                                                        const RuntimePublishWorld* runtimeWorld) noexcept
    {
        // IR-7 (Chapter 8): State transition precondition validation.
        // NOTE: EngineRuntime is deprecated. Projection values should be sourced
        // from RuntimeSemanticSchema fields (world->routing, world->automation, etc.)
        // instead of this struct. See 3.2.7 for migration plan.
        jassert(current != nullptr); // Current DSP must always be valid at publish time.
        jassert(fadeTimeSec >= 0.0);
        jassert(runtimeWorld != nullptr); // Bootstrap World ensures non-null at first publish (#3.2.5)

        // Safety guard: when runtimeWorld is null (e.g. after releaseResources cleared the
        // published world but before prepareToPlay publishes a new one), produce a minimal
        // EngineRuntime from atomics so caller never dereferences a null world.
        if (runtimeWorld == nullptr)
        {
            convo::EngineRuntime fallback {};
            fallback.current = current;
            fallback.currentRuntimeUuid = (current != nullptr) ? current->runtimeUuid : 0;
            fallback.transition.current = current;
            fallback.transition.next = next;
            fallback.transitionCurrentRuntimeUuid = (current != nullptr) ? current->runtimeUuid : 0;
            fallback.transitionNextRuntimeUuid = (next != nullptr) ? next->runtimeUuid : 0;
            fallback.transition.policy = policy;
            fallback.transition.fadeTimeSec = fadeTimeSec;
            fallback.transition.active = active;
            fallback.latencyDelayOld = consumeAtomic(latencyDelayOld, std::memory_order_acquire);
            fallback.latencyDelayNew = consumeAtomic(latencyDelayNew, std::memory_order_acquire);
            fallback.latencyResetPending = consumeAtomic(latencyResetPending, std::memory_order_acquire);
            fallback.dspCrossfadePending = crossfadeRuntime_.isPending();
            fallback.dspCrossfadeUseDryAsOld = crossfadeRuntime_.useDryAsOld();
            fallback.firstIrDryCrossfadePending = crossfadeRuntime_.isFirstIrDryPending();
            fallback.processingOrder = static_cast<int>(consumeAtomic(currentProcessingOrder, std::memory_order_acquire));
            fallback.eqBypassed = consumeAtomic(eqBypassActive, std::memory_order_acquire);
            fallback.convBypassed = consumeAtomic(convBypassActive, std::memory_order_acquire);
            fallback.softClipEnabled = consumeAtomic(softClipEnabled, std::memory_order_acquire);
            fallback.saturationAmount = consumeAtomic(saturationAmount, std::memory_order_acquire);
            fallback.inputHeadroomGain = consumeAtomic(inputHeadroomGain, std::memory_order_acquire);
            fallback.outputMakeupGain = consumeAtomic(outputMakeupGain, std::memory_order_acquire);
            fallback.convolverInputTrimGain = consumeAtomic(convolverInputTrimGain, std::memory_order_acquire);
            fallback.retireBacklog = 0;
            fallback.deferredResidency = 0;
            fallback.rebuildWorkerRunning = false;
            fallback.adaptiveCoeffBankIndex = -1;
            fallback.adaptiveCoeffGeneration = 0;
            fallback.eqCoeffHash = 0;
            fallback.queuedFadeTimeSec = crossfadeRuntime_.getQueuedFadeTimeSec();
            fallback.dspCrossfadeStartDelayBlocks = crossfadeRuntime_.getStartDelayBlocks();
            fallback.dspCrossfadeDryHoldSamples = crossfadeRuntime_.getDryHoldSamples();
            fallback.dryScaleTarget = crossfadeRuntime_.getDryScaleTarget();
            return fallback;
        }

        const bool crossfadePending = runtimeWorld->engine.dspCrossfadePending;
        const bool crossfadeUseDryAsOld = runtimeWorld->overlap.useDryAsOld;
        const bool firstLoadDryPending = runtimeWorld->overlap.firstIrDryCrossfadePending;
        const double queuedFadeTime = runtimeWorld->overlap.fadeTimeSec;
        const int delayOld = runtimeWorld->latency.latencyDelayOld;
        const int delayNew = runtimeWorld->latency.latencyDelayNew;
        const int startDelayBlocks = runtimeWorld->execution.crossfadeStartDelayBlocks;
        const int dryHoldSamples = runtimeWorld->execution.crossfadeDryHoldSamples;
        const bool resetLatencyPending = runtimeWorld->engine.latencyResetPending;
        const double dryScaleTarget = runtimeWorld->overlap.dryScaleTarget;

        // IR-7: Crossfade parameter sanity checks (invariant enforcement).
        jassert(queuedFadeTime >= 0.0);
        jassert(delayOld >= 0 && delayNew >= 0);
        jassert(startDelayBlocks >= 0);
        jassert(dryHoldSamples >= 0);

        convo::EngineRuntime runtime {};
        const auto getRuntimeUuid = [](DSPCore* dsp) noexcept -> std::uint64_t
        {
            return dsp != nullptr ? dsp->runtimeUuid : 0;
        };

        const bool transitionActive = active && next != nullptr;
        DSPCore* fading = transitionActive ? next : nullptr;

        runtime.current = current;
        runtime.currentRuntimeUuid = getRuntimeUuid(current);
        runtime.transition.current = current;
        runtime.transition.next = next;
        runtime.transitionCurrentRuntimeUuid = getRuntimeUuid(current);
        runtime.transitionNextRuntimeUuid = getRuntimeUuid(next);
        runtime.latencyDelayOld = delayOld;
        runtime.latencyDelayNew = delayNew;
        runtime.transition.policy = policy;
        runtime.transition.fadeTimeSec = fadeTimeSec;
        runtime.transition.latencyDeltaSamples = runtime.latencyDelayOld - runtime.latencyDelayNew;
        runtime.transition.active = transitionActive;
        runtime.fading = fading;
        runtime.fadingRuntimeUuid = getRuntimeUuid(fading);
        runtime.latencyResetPending = resetLatencyPending;
        runtime.dspCrossfadePending = crossfadePending;
        runtime.dspCrossfadeUseDryAsOld = crossfadeUseDryAsOld;
        runtime.firstIrDryCrossfadePending = firstLoadDryPending;
        runtime.processingOrder = runtimeWorld->routing.processingOrder;
        runtime.eqBypassed = runtimeWorld->routing.eqBypassed;
        runtime.convBypassed = runtimeWorld->routing.convBypassed;
        runtime.softClipEnabled = runtimeWorld->automation.softClipEnabled;
        runtime.saturationAmount = runtimeWorld->automation.saturationAmount;
        runtime.inputHeadroomGain = runtimeWorld->automation.inputHeadroomGain;
        runtime.outputMakeupGain = runtimeWorld->automation.outputMakeupGain;
        runtime.convolverInputTrimGain = runtimeWorld->automation.convolverInputTrimGain;
        runtime.retireBacklog = runtimeWorld->retire.retireBacklog;
        runtime.deferredResidency = runtimeWorld->retire.deferredResidency;
        runtime.rebuildWorkerRunning = runtimeWorld->affinity.rebuildWorkerRunning;
        runtime.adaptiveCoeffBankIndex = runtimeWorld->coefficient.adaptiveCoeffBankIndex;
        runtime.adaptiveCoeffGeneration = runtimeWorld->coefficient.adaptiveCoeffGeneration;
        runtime.eqCoeffHash = runtimeWorld->coefficient.eqCoeffHash;
        runtime.queuedFadeTimeSec = queuedFadeTime;
        runtime.dspCrossfadeStartDelayBlocks = startDelayBlocks;
        runtime.dspCrossfadeDryHoldSamples = dryHoldSamples;
        runtime.dryScaleTarget = dryScaleTarget;
        return runtime;
    }

    [[nodiscard]] inline CrossfadePreparedSnapshot consumeCrossfadePreparedSnapshot() const noexcept
    {
        // acquire: publishCrossfadePreparedSnapshot の release と HB し、
        //          最新のクロスフェードスナップショットスロットを取得。
        const int slot = static_cast<int>(convo::consumeAtomic(crossfadePreparedSnapshotIndex_, std::memory_order_acquire) & 1u);
        return crossfadePreparedSnapshots_[slot];
    }

    inline void publishCrossfadePreparedSnapshot(const CrossfadePreparedSnapshot& snapshot) noexcept
    {
        // acquire: 現スロットの読み出しは直前の公開を観測。
        const int currentSlot = convo::consumeAtomic(crossfadePreparedSnapshotIndex_, std::memory_order_acquire) & 1;
        const int nextSlot = currentSlot ^ 1;
        crossfadePreparedSnapshots_[nextSlot] = snapshot;
        // release: 新スロット書き込み完了を consumeCrossfadePreparedSnapshot の acquire に公開。
        convo::publishAtomic(crossfadePreparedSnapshotIndex_, nextSlot, std::memory_order_release);
    }

    inline void refreshCrossfadePreparedSnapshotFromAtomics() noexcept
    {
        // ★ CrossfadeRuntime 経由で crossfade 状態を読み取る
        //   latency 系フィールドは AudioEngine 直下の atomic から読み取る
        const CrossfadePreparedSnapshot snapshot {
            .pending = crossfadeRuntime_.isPending(),
            .useDryAsOld = crossfadeRuntime_.useDryAsOld(),
            .firstIrDryCrossfadePending = crossfadeRuntime_.isFirstIrDryPending(),
            .fadeTimeSec = crossfadeRuntime_.getQueuedFadeTimeSec(),
            .latencyDelayOld = consumeAtomic(latencyDelayOld),
            .latencyDelayNew = consumeAtomic(latencyDelayNew),
            .startDelayBlocks = crossfadeRuntime_.getStartDelayBlocks(),
            .dryHoldSamples = crossfadeRuntime_.getDryHoldSamples(),
            .latencyResetPending = consumeAtomic(latencyResetPending),
            .dryScaleTarget = crossfadeRuntime_.getDryScaleTarget()
        };

        publishCrossfadePreparedSnapshot(snapshot);
    }

    inline void publishLatencyDelayAtomics(int oldDelay,
                                           int newDelay) noexcept
    {
        convo::publishAtomic(latencyDelayOld, oldDelay, std::memory_order_release); // release: audio thread の latency read と HB
        convo::publishAtomic(latencyDelayNew, newDelay, std::memory_order_release); // release: audio thread の latency read と HB
    }

    inline void resetLatencyDelayRtState() noexcept
    {
    }

    // getRetireRouter: RCUReader 生成用の IReaderEpochProvider を返す。
    // 戻り値は抽象インターフェース（IReaderEpochProvider）であり、具象型 ISRRetireRouter には依存しない。
    // これにより将来 RetireRouter の差し替え時に影響範囲を Reader 生成箇所のみに限定できる。
    // 注意: このメソッドが返すポインタの寿命は AudioEngine 全体の寿命と一致する。
    // AudioEngine より長生きする RCUReader を生成してはならない。
    [[nodiscard]] convo::IReaderEpochProvider& getRetireRouter() noexcept
    {
        jassert(m_retireRouter != nullptr);
        return *m_retireRouter;
    }

    [[nodiscard]] inline RuntimeReadHandle makeRuntimeReadHandle(
        const convo::RuntimeReaderContext& ctx) noexcept
    {
        switch (ctx.channel)
        {
        case convo::ObserveChannel::Audio:
            debugAssertAudioThread();
            break;
        case convo::ObserveChannel::Message:
        case convo::ObserveChannel::Publication:
            debugAssertNotAudioThread();
            break;
        default:
            // Worker: アサーションなし（Worker スレッド識別は現状未実装のため skip）
            break;
        }

        const auto readToken = RuntimePublicationCoordinator::acquireReadToken(runtimeStore);
        const auto* world = RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore, readToken);
        if (world != nullptr)
        {
            const int slot = juce::jlimit(0, convo::kObserveChannelCount - 1,
                                          static_cast<int>(ctx.channel));

            const auto currentGeneration = world->generation;
            const auto currentSequence = world->publication.sequenceId;
            const auto previousGeneration = consumeAtomic(observeLastSeenGeneration_[slot], std::memory_order_acquire);
            const auto previousSequence = consumeAtomic(observeLastSeenSequenceId_[slot], std::memory_order_acquire);

            const bool generationBackward = (previousGeneration != 0 && currentGeneration < previousGeneration);
            const bool sequenceBackward = (previousSequence != 0 && currentSequence < previousSequence);

            if (generationBackward || sequenceBackward)
            {
                fetchAddAtomic(observeMonotonicViolationCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
                publishAtomic(observeMonotonicRollbackRequested_, true, std::memory_order_release);
            }

            if (currentGeneration > previousGeneration)
                publishAtomic(observeLastSeenGeneration_[slot], currentGeneration, std::memory_order_release);
            if (currentSequence > previousSequence)
                publishAtomic(observeLastSeenSequenceId_[slot], currentSequence, std::memory_order_release);

            auto updateMinMetric = [](std::atomic<std::uint64_t>& dst, std::uint64_t value) noexcept
            {
                auto observed = convo::consumeAtomic(dst, std::memory_order_acquire);
                while ((observed == 0 || value < observed)
                       && !convo::compareExchangeAtomic(dst,
                                                        observed,
                                                        value,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_acquire))
                {
                }
            };

            auto updateMaxMetric = [](std::atomic<std::uint64_t>& dst, std::uint64_t value) noexcept
            {
                auto observed = convo::consumeAtomic(dst, std::memory_order_acquire);
                while (value > observed
                       && !convo::compareExchangeAtomic(dst,
                                                        observed,
                                                        value,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_acquire))
                {
                }
            };

            updateMinMetric(oldestObservedGeneration_, currentGeneration);
            updateMaxMetric(youngestObservedGeneration_, currentGeneration);
        }

        auto observed = m_coordinator.observeCurrentRuntime(ctx.reader);

        return RuntimeReadHandle {
            std::move(observed),
            world
        };
    }

    [[nodiscard]] static inline CrossfadePreparedSnapshot makeCrossfadePreparedSnapshotFromWorld(const RuntimePublishWorld& world) noexcept
    {
        CrossfadePreparedSnapshot snapshot {};
        snapshot.pending = world.engine.dspCrossfadePending;
        snapshot.useDryAsOld = world.overlap.useDryAsOld;
        snapshot.firstIrDryCrossfadePending = world.overlap.firstIrDryCrossfadePending;
        snapshot.fadeTimeSec = world.overlap.fadeTimeSec;
        snapshot.latencyDelayOld = world.latency.latencyDelayOld;
        snapshot.latencyDelayNew = world.latency.latencyDelayNew;
        snapshot.startDelayBlocks = world.execution.crossfadeStartDelayBlocks;
        snapshot.dryHoldSamples = world.execution.crossfadeDryHoldSamples;
        snapshot.latencyResetPending = world.engine.latencyResetPending;
        snapshot.dryScaleTarget = world.overlap.dryScaleTarget;
        return snapshot;
    }

    [[nodiscard]] static inline const RuntimePublishWorld* getRuntimeWorldFromReadHandle(const RuntimeReadHandle& runtimeReadHandle) noexcept
    {
        return runtimeReadHandle.runtimeWorldPtr();
    }

    [[nodiscard]] static inline convo::TransitionPolicy getTransitionPolicyFromRuntimeWorld(const RuntimeReadHandle& runtimeReadHandle,
                                                                                             convo::TransitionPolicy fallback = convo::TransitionPolicy::SmoothOnly) noexcept
    {
        const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
        return (runtimeWorld != nullptr)
            ? static_cast<convo::TransitionPolicy>(runtimeWorld->execution.transitionPolicy)
            : fallback;
    }

    [[nodiscard]] static inline double getOverlapFadeTimeFromRuntimeWorld(const RuntimeReadHandle& runtimeReadHandle,
                                                                           double fallback = 0.0) noexcept
    {
        const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
        return (runtimeWorld != nullptr)
            ? runtimeWorld->overlap.fadeTimeSec
            : fallback;
    }

    [[nodiscard]] static inline bool hasFadingRuntimeInWorld(const RuntimeReadHandle& runtimeReadHandle) noexcept
    {
        const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
        return (runtimeWorld != nullptr) && runtimeWorld->topology.hasFadingRuntime;
    }

    [[nodiscard]] static inline bool hasPendingCrossfadeInWorld(const RuntimeReadHandle& runtimeReadHandle) noexcept
    {
        const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
        return (runtimeWorld != nullptr) && runtimeWorld->engine.dspCrossfadePending;
    }

    [[nodiscard]] static inline bool shouldUseDryAsOldInWorld(const RuntimeReadHandle& runtimeReadHandle) noexcept
    {
        const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
        return (runtimeWorld != nullptr)
            && (runtimeWorld->overlap.firstIrDryCrossfadePending
                || runtimeWorld->overlap.useDryAsOld);
    }

    [[nodiscard]] static inline double getRuntimeSampleRateHzFromWorld(const RuntimeReadHandle& runtimeReadHandle,
                                                                        double fallback = 0.0) noexcept
    {
        const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
        return (runtimeWorld != nullptr)
            ? runtimeWorld->timing.sampleRateHz
            : fallback;
    }

    [[nodiscard]] static inline const convo::RuntimeGraph* getRuntimeGraph(const RuntimeReadHandle& runtimeReadHandle) noexcept
    {
        const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
        return (runtimeWorld != nullptr) ? &runtimeWorld->graph : nullptr;
    }

    [[nodiscard]] static inline const convo::GlobalSnapshot* getRuntimeSnapshotFromReadHandle(const RuntimeReadHandle& runtimeReadHandle) noexcept
    {
        return runtimeReadHandle.observedSnapshotPtr();
    }

    [[nodiscard]] inline DSPCore* resolveFadingRuntimeDSPFromRuntimeWorldOnly(const RuntimeReadHandle& runtimeReadHandle) const noexcept
    {
        const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
        if (runtimeWorld == nullptr
            || !runtimeWorld->topology.hasFadingRuntime)
            return nullptr;

        return static_cast<DSPCore*>(runtimeWorld->engine.fading);
    }

    [[nodiscard]] inline DSPCore* resolveActiveRuntimeDSPFromRuntimeWorldOnly(const RuntimeReadHandle& runtimeReadHandle) const noexcept
    {
        const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
        return (runtimeWorld != nullptr)
            ? static_cast<DSPCore*>(runtimeWorld->engine.current)
            : nullptr;
    }

    inline convo::RuntimeGraph makeRuntimeGraphState(const convo::EngineRuntime& state) noexcept
    {
        // 3.2.9: RuntimeGraph は Projection + Diagnostic のみに縮退。
        // Authoritative フィールド（runtimeUuid, eqBypassed 等）は
        // RuntimeWorld の対応する Semantic 構造体から参照する。
        convo::RuntimeGraph graph {};
        graph.activeNode = state.current;
        graph.fadingNode = state.fading;

        auto* current = static_cast<DSPCore*>(state.current);
        if (current != nullptr)
        {
            auto& eq = current->eqRt();
            graph.eqAgcAttackCoeffTable = eq.getAgcAttackCoeffTable();
            graph.eqAgcReleaseCoeffTable = eq.getAgcReleaseCoeffTable();
            graph.eqAgcSmoothCoeffTable = eq.getAgcSmoothCoeffTable();
            graph.eqAgcCoeffTableCapacity = eq.getAgcCoeffTableCapacity();
        }

        return graph;
    }

    template <typename T>
    static inline void publishAtomicPtr(std::atomic<T*>& dst, T* value) noexcept
    {
        // release: pointer を consumeAtomicPtr acquire に公開。
        convo::publishAtomic(dst, value, std::memory_order_release);
    }

    template <typename T>
    [[nodiscard]] static inline T* consumeAtomicPtr(const std::atomic<T*>& src) noexcept
    {
        // acquire: publishAtomicPtr release と HB し、最新の pointer を取得。
        return convo::consumeAtomic(src, std::memory_order_acquire);
    }

    template <typename T, typename U,
              typename = std::enable_if_t<std::is_convertible_v<U, T*>>>
    static inline T* exchangeAtomicPtr(std::atomic<T*>& dst, U&& value) noexcept
    {
        // acq_rel: 旧 pointer を acquire で取得し、新 pointer を release で公開。
        return convo::exchangeAtomic(dst, static_cast<T*>(std::forward<U>(value)), std::memory_order_acq_rel);
    }

    template <typename T>
    static inline void publishAtomic(std::atomic<T>& dst,
                                     T value,
                                     std::memory_order order = std::memory_order_release) noexcept
    {
        // order (release/relaxed): callerの consumeAtomic acquire と HB し、値を公開。
        convo::publishAtomic(dst, value, order);
    }

    template <typename T>
    static inline T consumeAtomic(const std::atomic<T>& src,
                                  std::memory_order order = std::memory_order_acquire) noexcept
    {
        // order (acquire/relaxed): callerの publishAtomic release と HB し、値を取得。
        return convo::consumeAtomic(src, order);
    }

    template <typename T>
    static inline T exchangeAtomic(std::atomic<T>& dst,
                                   T value,
                                   std::memory_order order = std::memory_order_acq_rel) noexcept
    {
        // order (acq_rel): 旧値を acquire で取得し、新値を release で公開。
        return convo::exchangeAtomic(dst, value, order);
    }

    template <typename T, typename U,
              typename = std::enable_if_t<std::is_integral_v<T> && std::is_convertible_v<U, T>>>
    static inline T fetchAddAtomic(std::atomic<T>& dst,
                                   U value,
                                   std::memory_order order = std::memory_order_acq_rel) noexcept
    {
        // order (acq_rel): 旧値を acquire で辻取り、新値を release で公開。世代カウンター advance、etc.に使用。
        return convo::fetchAddAtomic(dst, value, order);
    }

    [[nodiscard]] inline std::uint64_t reserveNextRuntimeGraphGeneration() noexcept
    {
        // acq_rel: publish count を increment し、runtime world 公開イベント数を追跡。
        const auto nextGraphGeneration = runtimeGenerationGenerator_.next();
        convo::fetchAddAtomic(rtAuxMutable_.runtimePublishCount,
                              static_cast<std::uint64_t>(1),
                              std::memory_order_acq_rel);
        return nextGraphGeneration;
    }

    struct RuntimePublishComputation
    {
        convo::EngineRuntime engineState {};
        convo::RuntimeGraph graphState {};
        convo::isr::PublicationSequenceId previousCommittedSequence = 0;
    };

    [[nodiscard]] inline RuntimePublishComputation computeRuntimePublishComputation(DSPCore* current,
                                                                                     DSPCore* next,
                                                                                     convo::TransitionPolicy policy,
                                                                                     double fadeTimeSec,
                                                                                     bool active,
                                                                                     std::uint64_t generation) noexcept
    {
        RuntimePublishComputation computation {};
        const auto readToken = RuntimePublicationCoordinator::acquireReadToken(runtimeStore);
        const auto* runtimeWorld = RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore, readToken);

        jassert(runtimeWorld != nullptr); // Bootstrap World guarantees non-null (#3.2.5)

        computation.engineState = makeEngineRuntimeState(current,
                                                         next,
                                                         policy,
                                                         fadeTimeSec,
                                                         active,
                                                         runtimeWorld);
        computation.engineState.revision = generation;
        computation.graphState = makeRuntimeGraphState(computation.engineState);
        computation.previousCommittedSequence = getLastCommittedPublicationSequence();
        return computation;
    }

    struct RuntimePublicationIdentity
    {
        std::uint64_t generation = 0;
        std::uint64_t worldId = 0;
        convo::isr::PublicationSequenceId publicationSequence = 0;
    };

    [[nodiscard]] inline RuntimePublicationIdentity reserveRuntimePublicationIdentity() noexcept
    {
        RuntimePublicationIdentity identity {};
        identity.generation = reserveNextRuntimeGraphGeneration();
        identity.worldId = runtimeWorldIdGenerator_.next();
        identity.publicationSequence = convo::fetchAddAtomic(publicationSequenceCounter_,
                                                             static_cast<convo::isr::PublicationSequenceId>(1),
                                                             std::memory_order_acq_rel) + 1;
        return identity;
    }

    //=== RuntimePublicationCoordinator NonRT helper API ===//
    // AudioEngine 内部の publish/retire helper（NonRT 専用）。

    [[nodiscard]] bool runPublicationPrecheckNonRt(const RuntimePublishWorld& world) noexcept;
    void onRuntimePublishedNonRt(const RuntimePublishWorld& world) noexcept;
    void onRuntimeRetiredNonRt(const RuntimePublishWorld* world) noexcept;
    void emitEvidenceTickNonRt(bool force) noexcept;
    void onHealthEvent(const convo::HealthEvent& event) noexcept;  // ★ P1-8: HealthMonitor コールバック

    // ★ P1-6/8: Publication backlog の公開（RuntimeHealthMonitor → Orchestrator → AudioEngine → bridge）
    [[nodiscard]] uint64_t getPublicationBacklogCount() const noexcept {
        return runtimePublicationBridge_.getPublicationBacklogCount();
    }
    // ★ P1-6/8: Retire pending intent の公開
    [[nodiscard]] uint64_t getRetirePendingIntentCount() const noexcept {
        return retireRuntime_.pendingIntentCount();
    }

    //=== End RuntimePublicationCoordinator NonRT helper API ===//

    class RuntimePublicationBridge final
    {
    public:
        explicit RuntimePublicationBridge(AudioEngine& engine, iso::audio_engine::RuntimePublicationValidator& validator) noexcept
            : engine_(&engine), validator_(&validator)
        {
        }

        // buildRuntimePublishWorld() removed from Bridge (#5/#7 Sprint-2)
        // Build authority belongs to RuntimeBuilder, not Bridge
        // Bridge responsibility: validate / didPublish / willRetire only

        [[nodiscard]] bool validatePublicationNonRt(const RuntimePublishWorld& world) noexcept
        {
            // Delegation to independent validator (#21 Sprint-4)
            const auto result = validator_->validatePublication(world);
            if (!result.isValid) {
                return false;
            }
            // Additional engine-specific checks (if any) can be added here
            return engine_->runPublicationPrecheckNonRt(world);
        }

        void didPublishRuntimeNonRt(const RuntimePublishWorld& world) noexcept
        {
            engine_->onRuntimePublishedNonRt(world);
        }

        void willRetireRuntimeNonRt(const RuntimePublishWorld* world) noexcept
        {
            if (world == nullptr)
                return;

            if (engine_->isShutdownInProgress())
                return;

            engine_->onRuntimeRetiredNonRt(world);
        }

        void retireRuntimePublishWorldNonRt(RuntimePublishWorld* world, bool resetRevision) noexcept
        {
            if (world == nullptr)
                return;

            engine_->enqueueDeferredDeleteNonRt(world, [](void* p)
            {
                auto* ptr = static_cast<RuntimePublishWorld*>(p);
                ptr->~RuntimePublishWorld();
                convo::aligned_free(ptr);
            });
            (void)resetRevision;
        }

    private:
        AudioEngine* engine_ = nullptr;
        iso::audio_engine::RuntimePublicationValidator* validator_ = nullptr;
    };

    using RuntimePublicationCoordinator = convo::RuntimePublicationCoordinator<RuntimePublishWorld,
                                                                               DSPCore*,
                                                                               RuntimePublicationBridge>;

    static_assert(!std::is_copy_constructible_v<RuntimePublicationCoordinator>,
                  "RuntimePublicationCoordinator must remain move-only");
    static_assert(!std::is_copy_assignable_v<RuntimePublicationCoordinator>,
                  "RuntimePublicationCoordinator must remain move-only");
    static_assert(std::is_move_constructible_v<RuntimePublicationCoordinator>,
                  "RuntimePublicationCoordinator must remain move-constructible");

    using RuntimePublishStore = RuntimePublicationCoordinator::Store;

    RuntimePublishStore runtimeStore;
    iso::audio_engine::RuntimePublicationValidator runtimePublicationValidator_;

    // ★ P0-4: runtimeStore.observe() の getter ラッパー
    [[nodiscard]] const RuntimePublishWorld* observePublishedWorld() const noexcept {
        return RuntimePublicationCoordinator::consumePublishedWorld(runtimeStore);
    }

    // RuntimePublicationOrchestrator: 前方宣言 + unique_ptr (循環依存回避)
    // 実体は AudioEngine.cpp のコンストラクタで初期化
    std::unique_ptr<convo::isr::RuntimePublicationOrchestrator> runtimeOrchestrator_;

private:
    // ★ P0-2/3: Coordinator生成は friend 宣言されたクラスに限定
    friend class convo::isr::RuntimePublicationOrchestrator;
    friend class convo::isr::PublicationExecutor;
    friend class convo::isr::DSPTransition;

    [[nodiscard]] inline RuntimePublicationCoordinator makeRuntimePublicationCoordinator() noexcept
    {
        using RuntimePublicationCoordinatorFactory = RuntimePublicationCoordinator;
        return RuntimePublicationCoordinatorFactory::create(
            RuntimePublicationBridge { *this, runtimePublicationValidator_ }, runtimeStore);
    }


    inline void publishWorld(convo::aligned_unique_ptr<RuntimePublishWorld> worldOwner) noexcept
    {
        auto coordinator = makeRuntimePublicationCoordinator();
        coordinator.publishWorld(std::move(worldOwner));
    }

public:
    [[nodiscard]] inline bool precheckRuntimePublication(const convo::isr::PayloadClosureDescriptor& closure,
                                                         const convo::isr::TieredPayloadDescriptor& descriptor) noexcept
    {
        auto& runtimePublicationCoordinator = runtimePublicationBridge_;
        return runtimePublicationCoordinator.precheckPublish(closure, descriptor);
    }

    inline void logUnexpectedRuntimeTransition(const char* origin,
                                               DSPCore* current,
                                               DSPCore* candidate) const noexcept
    {
        const auto currentUuid = (current != nullptr) ? current->runtimeUuid : 0;
        const auto candidateUuid = (candidate != nullptr) ? candidate->runtimeUuid : 0;
        const juce::String message = "[DIAG] runtime transition anomaly origin="
            + juce::String(origin != nullptr ? origin : "unknown")
            + " currentUuid=" + juce::String(static_cast<juce::int64>(currentUuid))
            + " candidateUuid=" + juce::String(static_cast<juce::int64>(candidateUuid));
        DBG(message);
        juce::Logger::writeToLog(message);
    }

    inline void logRuntimeTransitionEvent(const char* origin,
                                          DSPCore* primary,
                                          DSPCore* secondary = nullptr) const noexcept
    {
        const auto getUuid = [](DSPCore* dsp) noexcept -> std::uint64_t
        {
            return (dsp != nullptr) ? dsp->runtimeUuid : 0;
        };

        DSPCore* current = getActiveRuntimeDSP();
        const auto readToken = RuntimePublicationCoordinator::acquireReadToken(runtimeStore);
        const auto* publishedWorld = RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore, readToken);
        const auto& transition = (publishedWorld != nullptr) ? publishedWorld->engine.transition : convo::TransitionState{};
        const bool transitionActive = (publishedWorld != nullptr) && publishedWorld->topology.hasFadingRuntime;
        auto* fading = (transitionActive && transition.next != nullptr)
            ? static_cast<DSPCore*>(transition.next)
            : nullptr;
        const auto revision = (publishedWorld != nullptr) ? publishedWorld->generation : static_cast<std::uint64_t>(0);
        const auto publishedCurrentUuid = (transition.current != nullptr)
            ? static_cast<DSPCore*>(transition.current)->runtimeUuid
            : 0;
        const auto publishedFadingUuid = (transitionActive && transition.next != nullptr)
            ? static_cast<DSPCore*>(transition.next)->runtimeUuid
            : 0;
        const bool publishedWorldIsNull = (publishedWorld == nullptr);
        const bool transitionCurrentIsNull = (transition.current == nullptr);

        const juce::String message = "[DIAG] runtime transition event origin="
            + juce::String(origin != nullptr ? origin : "unknown")
            + " primaryUuid=" + juce::String(static_cast<juce::int64>(getUuid(primary)))
            + " secondaryUuid=" + juce::String(static_cast<juce::int64>(getUuid(secondary)))
            + " currentUuid=" + juce::String(static_cast<juce::int64>(getUuid(current)))
            + " fadingUuid=" + juce::String(static_cast<juce::int64>(getUuid(fading)))
            + " publishRev=" + juce::String(static_cast<juce::int64>(revision))
            + " publishCurrentUuid=" + juce::String(static_cast<juce::int64>(publishedCurrentUuid))
            + " publishFadingUuid=" + juce::String(static_cast<juce::int64>(publishedFadingUuid))
            + " worldNull=" + juce::String(static_cast<int>(publishedWorldIsNull))
            + " transCurNull=" + juce::String(static_cast<int>(transitionCurrentIsNull));
        DBG(message);
        juce::Logger::writeToLog(message);
    }

    inline bool validateDistinctRuntimeSlots(const char* origin,
                                             DSPCore* active,
                                             DSPCore* fading,
                                             DSPCore* queued) const noexcept
    {
        const bool activeEqualsFading = (active != nullptr && active == fading);
        const bool activeEqualsQueued = (active != nullptr && active == queued);
        const bool fadingEqualsQueued = (fading != nullptr && fading == queued);

        if (!activeEqualsFading && !activeEqualsQueued && !fadingEqualsQueued)
            return true;

        const auto getUuid = [](DSPCore* dsp) noexcept -> std::uint64_t
        {
            return (dsp != nullptr) ? dsp->runtimeUuid : 0;
        };

        const juce::String message = "[DIAG] runtime slot overlap origin="
            + juce::String(origin != nullptr ? origin : "unknown")
            + " activeUuid=" + juce::String(static_cast<juce::int64>(getUuid(active)))
            + " fadingUuid=" + juce::String(static_cast<juce::int64>(getUuid(fading)))
            + " queuedUuid=" + juce::String(static_cast<juce::int64>(getUuid(queued)));
        DBG(message);
        juce::Logger::writeToLog(message);
        jassert(!activeEqualsFading && !activeEqualsQueued && !fadingEqualsQueued);
        return false;
    }

    // Audio Thread でも使えるように、ログ出力を行わない軽量版。
    inline bool validateDistinctRuntimeSlotsRT(DSPCore* active,
                                               DSPCore* fading,
                                               DSPCore* queued) const noexcept
    {
        const bool activeEqualsFading = (active != nullptr && active == fading);
        const bool activeEqualsQueued = (active != nullptr && active == queued);
        const bool fadingEqualsQueued = (fading != nullptr && fading == queued);

        jassert(!activeEqualsFading && !activeEqualsQueued && !fadingEqualsQueued);
        return !activeEqualsFading && !activeEqualsQueued && !fadingEqualsQueued;
    }

    inline EngineParameterSnapshot captureAudioThreadParameterSnapshot(const convo::GlobalSnapshot* snap,
                                                                       bool isFadingTarget = false) const noexcept
    {
        EngineParameterSnapshot snapshot {};
        // acquire: setXxx の release を観測しパラメータ最新値を取得。
        snapshot.eqBypassed = (snap != nullptr) ? snap->eqBypass : consumeAtomic(eqBypassRequested, std::memory_order_acquire);
        snapshot.convBypassed = (snap != nullptr) ? snap->convBypass : consumeAtomic(convBypassRequested, std::memory_order_acquire);
        snapshot.order = (snap != nullptr) ? snap->processingOrder : consumeAtomic(currentProcessingOrder, std::memory_order_acquire);
        snapshot.softClipEnabled = (snap != nullptr) ? snap->softClipEnabled : consumeAtomic(softClipEnabled, std::memory_order_acquire);
        snapshot.saturationAmount = (snap != nullptr) ? snap->saturationAmount : consumeAtomic(saturationAmount, std::memory_order_acquire);
        snapshot.inputHeadroomGain = (snap != nullptr) ? snap->inputHeadroomGain : consumeAtomic(inputHeadroomGain, std::memory_order_acquire);
        snapshot.outputMakeupGain = (snap != nullptr) ? snap->outputMakeupGain : consumeAtomic(outputMakeupGain, std::memory_order_acquire);
        snapshot.convolverInputTrimGain = (snap != nullptr) ? snap->convInputTrimGain : consumeAtomic(convolverInputTrimGain, std::memory_order_acquire);
        snapshot.analyzerSource = consumeAtomic(currentAnalyzerSource, std::memory_order_acquire);
        snapshot.analyzerEnabled = isFadingTarget ? false : consumeAtomic(analyzerEnabled, std::memory_order_acquire);
        snapshot.convHCMode = consumeAtomic(convHCFilterMode, std::memory_order_acquire);
        snapshot.convLCMode = consumeAtomic(convLCFilterMode, std::memory_order_acquire);
        snapshot.eqLPFMode = consumeAtomic(eqLPFFilterMode, std::memory_order_acquire);
        snapshot.adaptiveCoeffBankIndex = consumeAtomic(currentAdaptiveCoeffBankIndex, std::memory_order_acquire);
        snapshot.adaptiveCoeffGeneration = 0u;
        const auto& adaptiveCoeffBank = getAdaptiveCoeffBankForIndex(snapshot.adaptiveCoeffBankIndex);
        snapshot.adaptiveCoeffSet = getActiveCoeffSet(adaptiveCoeffBank);
        snapshot.adaptiveCaptureEnabled = consumeAtomic(adaptiveCaptureActiveRt, std::memory_order_acquire);
        if (snap != nullptr)
        {
            snapshot.snapshotEqParams = &snap->eqParams;
            snapshot.snapshotEqCoeffHash = snap->eqCoeffHash;
        }
        return snapshot;
    }

    inline EngineParameterSnapshot captureAudioThreadParameterSnapshot(const RuntimePublishWorld* world,
                                                                       bool isFadingTarget = false) const noexcept
    {
        EngineParameterSnapshot snapshot {};

        if (world != nullptr)
        {
            const auto processingOrder = world->routing.processingOrder;
            snapshot.eqBypassed = world->routing.eqBypassed;
            snapshot.convBypassed = world->routing.convBypassed;
            snapshot.order = (processingOrder == static_cast<int>(ProcessingOrder::EQThenConvolver))
                ? ProcessingOrder::EQThenConvolver
                : ProcessingOrder::ConvolverThenEQ;
            snapshot.softClipEnabled = world->automation.softClipEnabled;
            snapshot.saturationAmount = static_cast<float>(world->automation.saturationAmount);
            snapshot.inputHeadroomGain = world->automation.inputHeadroomGain;
            snapshot.outputMakeupGain = world->automation.outputMakeupGain;
            snapshot.convolverInputTrimGain = world->automation.convolverInputTrimGain;
            snapshot.adaptiveCoeffBankIndex = world->coefficient.adaptiveCoeffBankIndex;
            snapshot.adaptiveCoeffGeneration = world->coefficient.adaptiveCoeffGeneration;
            const auto& adaptiveCoeffBank = getAdaptiveCoeffBankForIndex(snapshot.adaptiveCoeffBankIndex);
            snapshot.adaptiveCoeffSet = getActiveCoeffSet(adaptiveCoeffBank);
            snapshot.snapshotEqCoeffHash = world->coefficient.eqCoeffHash;
        }
        else
        {
            snapshot.eqBypassed = consumeAtomic(eqBypassRequested, std::memory_order_acquire);
            snapshot.convBypassed = consumeAtomic(convBypassRequested, std::memory_order_acquire);
            snapshot.order = consumeAtomic(currentProcessingOrder, std::memory_order_acquire);
            snapshot.softClipEnabled = consumeAtomic(softClipEnabled, std::memory_order_acquire);
            snapshot.saturationAmount = consumeAtomic(saturationAmount, std::memory_order_acquire);
            snapshot.inputHeadroomGain = consumeAtomic(inputHeadroomGain, std::memory_order_acquire);
            snapshot.outputMakeupGain = consumeAtomic(outputMakeupGain, std::memory_order_acquire);
            snapshot.convolverInputTrimGain = consumeAtomic(convolverInputTrimGain, std::memory_order_acquire);
        }

        snapshot.analyzerSource = consumeAtomic(currentAnalyzerSource, std::memory_order_acquire);
        snapshot.analyzerEnabled = isFadingTarget ? false : consumeAtomic(analyzerEnabled, std::memory_order_acquire);
        snapshot.convHCMode = consumeAtomic(convHCFilterMode, std::memory_order_acquire);
        snapshot.convLCMode = consumeAtomic(convLCFilterMode, std::memory_order_acquire);
        snapshot.eqLPFMode = consumeAtomic(eqLPFFilterMode, std::memory_order_acquire);
        snapshot.adaptiveCoeffBankIndex = consumeAtomic(currentAdaptiveCoeffBankIndex, std::memory_order_acquire);
        snapshot.adaptiveCoeffGeneration = 0u;
        const auto& adaptiveCoeffBank = getAdaptiveCoeffBankForIndex(snapshot.adaptiveCoeffBankIndex);
        snapshot.adaptiveCoeffSet = getActiveCoeffSet(adaptiveCoeffBank);
        snapshot.adaptiveCaptureEnabled = consumeAtomic(adaptiveCaptureActiveRt, std::memory_order_acquire);
        return snapshot;
    }

    inline DSPCore::ProcessingState buildAudioThreadProcessingState(DSPCore* dsp,
                                                                     const EngineParameterSnapshot& snapshot) noexcept
    {
        const convo::EQParameters* eqParams = snapshot.snapshotEqParams;
        uint64_t eqCoeffHash = snapshot.snapshotEqCoeffHash;

        const EQCoeffCache* eqCache = (eqParams != nullptr && eqCoeffHash != 0)
            ? eqCacheManager.get(eqCoeffHash)
            : nullptr;
        // ISR厳密化: hash から cache を解決できない場合は EQ を fail-close で bypass する。
        // 非snapshot系フォールバックへ降りるより、公開済み不変状態を維持することを優先する。
        const bool eqBypassedFailClosed = snapshot.eqBypassed || (eqParams == nullptr) || (eqCache == nullptr);

        return DSPCore::ProcessingState {
            .eqBypassed = eqBypassedFailClosed,
            .convBypassed = snapshot.convBypassed,
            .order = snapshot.order,
            .analyzerSource = snapshot.analyzerSource,
            .analyzerEnabled = snapshot.analyzerEnabled,
            .softClipEnabled = snapshot.softClipEnabled,
            .saturationAmount = snapshot.saturationAmount,
            .inputHeadroomGain = snapshot.inputHeadroomGain,
            .outputMakeupGain = snapshot.outputMakeupGain,
            .convolverInputTrimGain = snapshot.convolverInputTrimGain,
            .convHCMode = snapshot.convHCMode,
            .convLCMode = snapshot.convLCMode,
            .eqLPFMode = snapshot.eqLPFMode,
            .adaptiveCoeffBankIndex = snapshot.adaptiveCoeffBankIndex,
            .adaptiveCoeffSet = snapshot.adaptiveCoeffSet,
            .adaptiveCoeffGeneration = snapshot.adaptiveCoeffGeneration,
            .adaptiveCaptureSampleRateHz = static_cast<int>(dsp->sampleRate + 0.5),
            .adaptiveCaptureBitDepth = dsp->ditherBitDepth,
            .captureSessionId = dsp->currentCaptureSessionId,
            .adaptiveCaptureQueue = snapshot.adaptiveCaptureEnabled ? &audioCaptureQueue : nullptr,
            .eqParams = eqParams,
            .eqCache = eqCache,
            .eqCoeffHash = eqCoeffHash
        };
    }

    inline bool updateAudioThreadSnapshotFade(int numSamples,
                                              float& snapshotAlpha,
                                              const convo::GlobalSnapshot*& snapshotFrom,
                                              const convo::GlobalSnapshot*& snapshotTo) noexcept
    {
        juce::ignoreUnused(numSamples);
        snapshotAlpha = 1.0f;
        snapshotFrom = nullptr;
        snapshotTo = nullptr;
        return false;
    }

    inline void armCrossfadeIfPending(bool hasFading,
                                      bool& useDryAsOld,
                                      const CrossfadePreparedSnapshot& prepared) noexcept
    {
        const bool hasPendingCrossfade = prepared.pending;

        if (!hasPendingCrossfade)
        {
            dspCrossfadeArmed_RT = false;
            dspCrossfadeStartDelayBlocks_RT = 0;
            return;
        }

        const bool firstLoadDryPending = prepared.firstIrDryCrossfadePending;

        if ((hasFading || firstLoadDryPending)
            && !dspCrossfadeArmed_RT)
        {
            dspCrossfadeArmed_RT = true;
            dspCrossfadeStartDelayBlocks_RT = prepared.startDelayBlocks;

            // C1-2: activate のみ Audio Thread で実行 (reset/setCurrentAndTargetValue は Message Thread 側)
            crossfadeRuntime_.getGain().setTargetValue(1.0);

            if (firstLoadDryPending)
            {
                useDryAsOld = true;
                crossfadeRuntime_.getDryScaleGain().setTargetValue(prepared.dryScaleTarget);
            }
        }
    }

    template <typename ProcessFn>
    inline bool processCrossfadeDelayGateIfPending(DSPCore* fading,
                                                   bool useDryAsOld,
                                                   const CrossfadePreparedSnapshot& prepared,
                                                   ProcessFn processFn) noexcept
    {
        if (fading != nullptr
            && !useDryAsOld
            && prepared.pending
            && dspCrossfadeStartDelayBlocks_RT > 0)
        {
            --dspCrossfadeStartDelayBlocks_RT;
            processFn();
            return true;
        }

        return false;
    }

    inline void finalizeCrossfadeMixPath(DSPCore* current,
                                         DSPCore* fading,
                                         bool resetDryScaleGain) noexcept
    {
        if (!crossfadeRuntime_.getGain().isSmoothing())
        {
            validateDistinctRuntimeSlotsRT(current, fading, nullptr);

            if (resetDryScaleGain)
            {
                crossfadeRuntime_.getDryScaleGain().current = 1.0;
                crossfadeRuntime_.getDryScaleGain().target = 1.0;
                crossfadeRuntime_.getDryScaleGain().step = 0.0;
                crossfadeRuntime_.getDryScaleGain().remaining = 0;
            }
        }
    }

    inline void cleanupCrossfadeDirectPath(DSPCore* current,
                                           DSPCore* fading) noexcept
    {
        if (fading != nullptr && !crossfadeRuntime_.getGain().isSmoothing())
        {
            validateDistinctRuntimeSlotsRT(current, fading, nullptr);
        }
    }

    inline void resetLatencyBuffersIfPending(int bufferSize,
                                             int& writePos,
                                             bool runtimeResetPending) noexcept
    {
        if (runtimeResetPending)
        {
            if (latencyBufOldL) std::memset(latencyBufOldL, 0, sizeof(double) * bufferSize);
            if (latencyBufOldR) std::memset(latencyBufOldR, 0, sizeof(double) * bufferSize);
            if (latencyBufNewL) std::memset(latencyBufNewL, 0, sizeof(double) * bufferSize);
            if (latencyBufNewR) std::memset(latencyBufNewR, 0, sizeof(double) * bufferSize);
            writePos = 0;
        }
    }

    inline int estimateRuntimeLatencyBaseRateSamples(const DSPCore* dsp,
                                                     bool convolverBypassed) const noexcept
    {
        if (dsp == nullptr)
            return 0;

        const double cachedSampleRate = consumeAtomic(currentSampleRate, std::memory_order_acquire);
        const double baseSampleRate = cachedSampleRate > 0.0
            ? cachedSampleRate
            : dsp->sampleRate;

        const int osFactor = std::max(1, static_cast<int>(dsp->oversamplingFactor));
        const auto toBaseRateSamples = [osFactor](int processingRateSamples) -> int
        {
            return juce::jmax(0,
                static_cast<int>(std::lround(static_cast<double>(processingRateSamples)
                    / static_cast<double>(osFactor))));
        };

        int totalLatency = juce::jmax(0,
            static_cast<int>(std::lround(estimateOversamplingLatencySamples(
                osFactor,
                dsp->activeOversamplingType,
                baseSampleRate))));

        if (!convolverBypassed)
        {
            const auto convBreakdown = dsp->convolver.getLatencyBreakdown();
            totalLatency += toBaseRateSamples(convBreakdown.totalLatencySamples);
        }

        return juce::jmax(0, totalLatency);
    }

    template <typename SampleType, typename MixFn>
    inline void runLatencyAlignedCrossfadeMixLoop(SampleType* dstL,
                                                  SampleType* dstR,
                                                  const SampleType* oldL,
                                                  const SampleType* oldR,
                                                  int numSamples,
                                                  int delayOld,
                                                  int delayNew,
                                                  bool runtimeResetPending,
                                                  MixFn mixFn) noexcept
    {
        const int bufferSize = latencyBufSize;
        int writePos = latencyWritePos;

        resetLatencyBuffersIfPending(bufferSize, writePos, runtimeResetPending);

        for (int i = 0; i < numSamples; ++i)
        {
            latencyBufOldL[writePos] = (oldL != nullptr) ? static_cast<double>(oldL[i]) : 0.0;
            latencyBufOldR[writePos] = (oldR != nullptr) ? static_cast<double>(oldR[i]) : 0.0;
            latencyBufNewL[writePos] = (dstL != nullptr) ? static_cast<double>(dstL[i]) : 0.0;
            latencyBufNewR[writePos] = (dstR != nullptr) ? static_cast<double>(dstR[i]) : 0.0;

            auto wrapIdx = [](int idx, int sz) { while (idx < 0) idx += sz; while (idx >= sz) idx -= sz; return idx; };
            const int readOld = wrapIdx(writePos - delayOld, bufferSize);
            const int readNew = wrapIdx(writePos - delayNew, bufferSize);

            const double alignedOldL = latencyBufOldL[readOld];
            const double alignedOldR = latencyBufOldR[readOld];
            const double alignedNewL = latencyBufNewL[readNew];
            const double alignedNewR = latencyBufNewR[readNew];

            const double gNew = crossfadeRuntime_.getGain().getNextValue();
            mixFn(dstL, dstR, i, gNew, alignedOldL, alignedOldR, alignedNewL, alignedNewR);

            ++writePos;
            if (writePos >= bufferSize)
                writePos = 0;
        }

        latencyWritePos = writePos;
    }

    friend class NoiseShaperLearner;
    friend class EQEditProcessor;
    // RuntimePublicationCoordinator は AudioEngine のネストクラスであるため
    // C++11 以降は自動的にプライベートメンバへアクセス可能。friend 宣言は不要。

//==============================================================================
// インラインヘルパー関数（Adaptive 係数アクセス）
//==============================================================================

// Audio Thread 用：現在アクティブな係数セットを取得（ロックフリー）
[[nodiscard]] static inline const CoeffSet* getActiveCoeffSet(const AdaptiveCoeffBankSlot& slot) noexcept
{
    // acquire: CoeffSetWriteLockGuard::commit の publishAtomic release と HB し、アクティブインデックスを確定取得。
    const int activeIdx = consumeAtomic(slot.activeIndex, std::memory_order_acquire);
    // Memory barrier ensures coeffSetA/coeffSetB contents are fully visible
    std::atomic_thread_fence(std::memory_order_acquire);
    return (activeIdx == 0)
           ? &slot.coeffSetA
           : &slot.coeffSetB;
}

// 書き込み側用：非アクティブバッファの予約（CAS）
static inline bool reserveInactiveCoeffSet(AdaptiveCoeffBankSlot& slot) noexcept
{
    bool expected = false;
    // acquire: writeLock acquire CAS で直前のロック解放を観測し、CAS成功側で排他制御を確立。
    return convo::compareExchangeAtomic(slot.writeLock,
                                        expected,
                                        true,
                                        std::memory_order_acquire,
                                        std::memory_order_acquire);
}

// 書き込み側用：予約した非アクティブセットへのポインタ取得
[[nodiscard]] static inline CoeffSet* getReservedInactiveCoeffSet(AdaptiveCoeffBankSlot& slot) noexcept
{
    // acquire: reserveInactiveCoeffSet CAS success の acquire と HB し、ロック下での確定インデックスを取得。
    int active = consumeAtomic(slot.activeIndex, std::memory_order_acquire);
    return (active == 0) ? &slot.coeffSetB : &slot.coeffSetA;
}

inline bool enqueueDeferredDeleteNonRt(void* ptr, void (*deleter)(void*)) noexcept
{
    const auto result = enqueueDeferredDeleteNonRtWithResult(ptr, deleter);
    return result != convo::isr::RetireEnqueueResult::Shutdown;
}

inline convo::isr::RetireEnqueueResult enqueueDeferredDeleteNonRtWithResult(void* ptr, void (*deleter)(void*)) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return convo::isr::RetireEnqueueResult::Success;

    if (isShutdownInProgress())
        return convo::isr::RetireEnqueueResult::Shutdown;

    const uint64_t epoch = markRetireEpoch();

    // [P0-5] 単一回試行 + drop. Router経由.
    if (m_retireRouter->enqueueRetire(ptr, deleter, epoch, DeletionEntryType::Generic) == convo::isr::RetireEnqueueResult::Success)
    {
        runtimePublicationBridge_.setRetireBacklogCount(
            static_cast<std::uint64_t>(m_retireRouter->pendingRetireCount()));
        return convo::isr::RetireEnqueueResult::Success;
    }

    // [P0-5] enqueue failure -> best-effort drain + telemetry.
    drainDeferredRetireQueues(false);
    const std::uint64_t retireDepth = static_cast<std::uint64_t>(m_retireRouter->pendingRetireCount());
    convo::publishAtomic(retireQueueDepth_, retireDepth, std::memory_order_release);
    runtimePublicationBridge_.setRetireBacklogCount(retireDepth);
    return convo::isr::RetireEnqueueResult::QueuePressure;
}

inline void retireDSP(DSPCore* dsp) noexcept
{
    if (dsp == nullptr)
        return;

    // 退役の唯一の入口。
    // ここでは「公開済みハンドルの解放」と「実体の deferred delete 予約」をまとめて行い、
    // active runtime slot / fading runtime slot など複数の非所有スロットからの回収責務を集約する。
    if (!retireDSPHandleForRuntime(dsp))
        return;

    convo::fetchAddAtomic(rtAuxMutable_.runtimeRetireCount,
                         static_cast<std::uint64_t>(1),
                         std::memory_order_acq_rel);
    switch (enqueueDeferredDeleteNonRtWithResult(dsp, &AudioEngine::destroyDSPCoreNode))
    {
        case convo::isr::RetireEnqueueResult::Success:
        case convo::isr::RetireEnqueueResult::QueuePressure:
            return;
        case convo::isr::RetireEnqueueResult::QueueFull:
            convo::fetchAddAtomic(rtAuxMutable_.debugRebuildDispatchRuntimeQueueFullCount,
                                  static_cast<std::uint64_t>(1),
                                  std::memory_order_acq_rel);
            return;
        case convo::isr::RetireEnqueueResult::Shutdown:
            return;
    }
}

inline convo::isr::DSPHandle registerDSPHandleForRuntime(DSPCore* dsp) noexcept
{
    if (dsp == nullptr)
        return convo::isr::DSPHandle::null();

    std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_);

    auto it = runtimeDSPHandleMap_.find(dsp);
    if (it != runtimeDSPHandleMap_.end())
        return it->second;

    const auto handle = dspHandleRuntime_.create(dsp);
    runtimeDSPHandleMap_.emplace(dsp, handle);
    return handle;
}

// resolveDSPHandle: DSPHandle → DSPCore* 解決 (Phase2: Execution Path Handle Normalization)
// DSPHandleRuntime::resolve() は ResolvedDSP を返すため、後方互換のために DSPCore* に単純化する。
// 解決失敗時は nullptr を返す（caller が適切に処理する）。
inline DSPCore* resolveDSPHandle(convo::isr::DSPHandle handle) noexcept
{
    if (handle.isNull())
        return nullptr;

    const auto resolved = dspHandleRuntime_.resolve(handle);
#if defined(JUCE_DEBUG) || defined(CONVO_CI_BUILD)
    if (!resolved.valid) {
        DBG("[DIAG] resolveDSPHandle: invalid handle slot=" << (int)handle.slot
            << " gen=" << (int)handle.generation << " isStale=" << (resolved.isStale ? 1 : 0));
    }
#endif
    if (!resolved.valid || resolved.isStale)
        return nullptr;

    return static_cast<DSPCore*>(resolved.instance);
}

inline bool retireDSPHandleForRuntime(DSPCore* dsp) noexcept
{
    if (dsp == nullptr)
        return false;

    std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_);
    const auto it = runtimeDSPHandleMap_.find(dsp);
    if (it == runtimeDSPHandleMap_.end())
        return false;

    const auto handle = it->second;
    runtimeDSPHandleMap_.erase(it);
    if (!handle.isNull())
    {
        dspHandleRuntime_.retire(handle);
        dspHandleRuntime_.reclaim(handle);
    }

    return true;
}

//==============================================================================
// RAII ガードクラス（例外安全性確保＋commit() 最適化）
//==============================================================================
class CoeffSetWriteLockGuard
{
public:
    explicit CoeffSetWriteLockGuard(AdaptiveCoeffBankSlot& s) noexcept
        : slot(s), acquired(false), committed(false) {}

    ~CoeffSetWriteLockGuard() noexcept
    {
        // commit() が呼ばれていない場合のみ、ロックを解放
        if (acquired && !committed)
            publishAtomic(slot.writeLock, false, std::memory_order_release);
    }

    bool acquire() noexcept
    {
        bool expected = false;
        // acquire: writeLock acquire CAS で直前のロック解放を観測し、CAS成功側で排他制御を確立。
        acquired = convo::compareExchangeAtomic(slot.writeLock,
                            expected,
                            true,
                            std::memory_order_acquire,
                            std::memory_order_acquire);
        return acquired;
    }

    // commit() を呼ぶことで、デストラクタでのロック解放をスキップ
    void commit() noexcept
    {
        if (!acquired || committed)
            return;

        // acquire: acquire() の CAS success と HB し、ロック下での確定インデックスを取得。
        int oldActive = consumeAtomic(slot.activeIndex, std::memory_order_acquire);
        // release: 新インデックスをgetActiveCoeffSet acquire に公開し、バッファスイッチを原子化。
        publishAtomic(slot.activeIndex, 1 - oldActive, std::memory_order_release);
        // acq_rel: 世代を原子的に increment し、getActiveCoeffSet の generation 更新を release で公開。
        convo::fetchAddAtomic(slot.generation,
                             1u,
                             std::memory_order_acq_rel);
        // release: ロック解放を次の acquire() CAS failure に公開し、排他制御を終了。
        publishAtomic(slot.writeLock, false, std::memory_order_release);
        committed = true;
    }

    [[nodiscard]] bool isAcquired() const noexcept { return acquired; }
    [[nodiscard]] bool isCommitted() const noexcept { return committed; }

    CoeffSetWriteLockGuard(const CoeffSetWriteLockGuard&) = delete;
    CoeffSetWriteLockGuard& operator=(const CoeffSetWriteLockGuard&) = delete;

    private:
        AdaptiveCoeffBankSlot& slot;
        bool acquired;
        bool committed;
    };

    // リリースキューに溜まったエントリを解放可能なものから処理する
    void processDeferredReleases();
    static void destroyDSPCoreNode(void* p) noexcept;

    // スナップショット基盤（Phase 2）
    // ==================================================================
    convo::EpochDomain m_epochDomain;
    // [work21] ISRRetireRouter — 唯一のretire/publication API入口(unique_ptr, forward-declared)
    std::unique_ptr<convo::isr::ISRRetireRouter> m_retireRouter;
    // DSP_THREAD_STATE: AudioEngine process系で使うaudio-thread専用RCU reader。
    convo::RCUReader audioThreadRcuReader { m_epochDomain };
    // Message Thread + JUCE Timer 専用 RCU reader。
    convo::RCUReader messageThreadRcuReader { m_epochDomain };
    RTLocalState rtLocalState_ {};
    RTAuxMutable rtAuxMutable_ {};
    convo::SnapshotCoordinator m_coordinator;
    GenerationManager m_generationManager;

    // ==================================================================
    // Phase 3: コマンドバッファ + ワーカースレッド
    // ==================================================================
    convo::CommandBuffer m_commandBuffer;
    convo::WorkerThread m_workerThread;
    std::mutex rebuildAdmissionIntentMutex_;
    RebuildAdmissionIntentState rebuildAdmissionPendingIntent_ {};

    std::atomic<float> m_currentInputHeadroomDb { -6.0f };
    std::atomic<float> m_currentOutputMakeupDb { 12.0f };
    std::atomic<float> m_currentConvInputTrimDb { 0.0f };
    std::atomic<bool> m_currentEqBypass { false };
    std::atomic<bool> m_currentConvBypass { false };
    std::atomic<convo::ProcessingOrder> m_currentProcessingOrder { convo::ProcessingOrder::ConvolverThenEQ };
    std::atomic<bool> m_currentSoftClipEnabled { true };
    std::atomic<float> m_currentSaturationAmount { 0.1f };

    std::atomic<std::uint64_t> retireQueueDepth_ { 0 };
    std::atomic<std::uint64_t> maxPendingRetireObserved_ { 0 };  // ★ Practical-1
    std::atomic<std::uint64_t> fallbackQueueDepth_ { 0 };
    std::atomic<std::uint64_t> quarantineResident_ { 0 };
    // publicationBacklog_ removed in Phase1-B; kept as always-0 legacy slot
    std::atomic<std::uint64_t> rebuildBacklog_ { 0 };
    std::atomic<std::uint64_t> saturationEnterCount_ { 0 };
    std::atomic<std::uint64_t> saturationExitCount_ { 0 };
    std::atomic<std::uint64_t> publicationRejectCount_ { 0 };
    std::atomic<std::uint8_t> semanticTransactionState_ {
        static_cast<std::uint8_t>(convo::isr::SemanticTransactionState::Building)
    };
    std::atomic<std::uint64_t> rebuildCollapseCount_ { 0 };
    std::atomic<int> retirePressureLevel_ { 0 };
    std::atomic<bool> retirePressureCoalescingActive_ { false };
    std::atomic<bool> retirePressurePublicationThrottleActive_ { false };
    std::atomic<bool> retirePressureAdmissionStrict_ { false };
    std::atomic<bool> retireProtectiveModeActive_ { false };
    std::atomic<std::uint64_t> prevDroppedSnapshot_ { 0 };
    std::atomic<std::uint64_t> retireEscalationCount_ { 0 };
    std::atomic<std::uint64_t> retireProtectiveModeEnterCount_ { 0 };
    std::atomic<std::uint64_t> maxRetireDeferralEpochs_ { 256 };
    std::atomic<double> maxRetireWallClockMs_ { 5000.0 };
    std::atomic<double> reclaimLatency_ { 0.0 };
    std::atomic<int> retireHighWatermark_ { 3072 };
    std::atomic<int> retireLowWatermark_ { 1024 };
    std::atomic<bool> retireSaturationActive_ { false };
    std::atomic<bool> retireSaturationRecoveryPending_ { false };
    std::atomic<std::uint64_t> retireSaturationRecoveryBaselinePublishCount_ { 0 };
    std::atomic<std::int64_t> emergencyReclaimWindowStartTicks_ { 0 };
    std::atomic<std::int64_t> emergencyReclaimLastBoostTicks_ { 0 };
    std::atomic<int> emergencyReclaimBoostCount_ { 0 };
    std::atomic<int> m_currentOversamplingFactor { 0 };
    std::atomic<convo::OversamplingType> m_currentOversamplingType { convo::OversamplingType::IIR };
    std::atomic<int> m_currentDitherBitDepth { 24 };
    std::atomic<convo::NoiseShaperType> m_currentNoiseShaperType { convo::NoiseShaperType::Psychoacoustic };

    static constexpr int DEFAULT_IR_FADE_SAMPLES = 2048;
    static constexpr int DEFAULT_EQ_FADE_SAMPLES = 256;
    static constexpr int DEFAULT_NS_FADE_SAMPLES = 1024;
    static constexpr int DEFAULT_AGC_FADE_SAMPLES = 128;
    std::atomic<int> m_irFadeSamples{ DEFAULT_IR_FADE_SAMPLES };
    std::atomic<int> m_eqFadeSamples{ DEFAULT_EQ_FADE_SAMPLES };
    std::atomic<int> m_nsFadeSamples{ DEFAULT_NS_FADE_SAMPLES };
    std::atomic<int> m_agcFadeSamples{ DEFAULT_AGC_FADE_SAMPLES };
    std::atomic<bool> m_pendingIRChange{ false };
    std::atomic<bool> m_pendingNSChange{ false };
    std::atomic<bool> m_pendingAGCChange{ false };
    // Audio Thread -> Message Thread 反映確認用 (RT安全: atomic store/fetch_add のみ)
    // Non-RT diagnostics: most recently rejected rebuild generation in commitNewDSP.
    juce::AudioBuffer<float> m_fadeFloatBuffer;
    juce::AudioBuffer<double> m_fadeDoubleBuffer;

    // ==================================================================
    // ISR Phase 0: Lifecycle Isolation Runtime（P0-A0）
    // ==================================================================

    convo::isr::LifecycleIsolationRuntime lifecycleRuntime_;
    convo::isr::LifecycleBarrierRuntime lifecycleBarrierRuntime_{ lifecycleRuntime_ };

    // ===================== ISR Phase 1-9: Core Runtimes =====================
    convo::isr::RuntimePublicationCoordinator runtimePublicationBridge_;
    convo::isr::DSPQuarantineManager dspQuarantineManager_;
    convo::isr::ClosureGraphWalker closureGraphWalker_;
    convo::isr::DebugRuntime debugRuntime_;
    convo::isr::RetireRuntime retireRuntime_;
    convo::isr::RetireRuntimeEx retireRuntimeEx_;
    convo::isr::ShutdownRuntime shutdownRuntime_;
    convo::isr::BarrierOptimizer barrierOptimizer_;
    convo::isr::EvidenceExporter evidenceExporter_;
    convo::isr::WorldLifecycleAudit worldLifecycleAudit_;
    convo::isr::BudgetManager budgetManager_;
    convo::isr::FailureHandler failureHandler_;
    convo::isr::IntrospectionConsole introspectionConsole_;
    convo::RuntimeHealthMonitor m_healthMonitor;  // ★ P1-8: Pull型監視エンジン

    std::array<std::atomic<std::uint64_t>, convo::kObserveChannelCount> observeLastSeenGeneration_ {};
    std::array<std::atomic<std::uint64_t>, convo::kObserveChannelCount> observeLastSeenSequenceId_ {};
    std::atomic<std::uint64_t> observeMonotonicViolationCount_ { 0 };
    std::atomic<bool> observeMonotonicRollbackRequested_ { false };

    // ★ Engine インスタンス識別子 (全局一意。再生成後もユニーク)
    static std::atomic<uint64_t> s_nextEngineInstanceId_;
    uint64_t engineInstanceId_{0};

    // ==================================================================

    JUCE_DECLARE_WEAK_REFERENCEABLE(AudioEngine)
        // ==================================================================
        // ISR Phase 1: RT Execution Frame Separation（P0-A1）
        // ==================================================================
        convo::isr::RTCapabilityFirewall rtCapabilityFirewall_;
        convo::isr::RTAllocatorFirewall rtAllocatorFirewall_;
        convo::isr::RTTraceRelay rtTraceRelay_;

        // Current fade accumulator (RCU-managed, updated during crossfade)
        convo::isr::FadeAccumulator currentFade_{ 0.0, 0.0, false };

        // ==================================================================
        // ISR Phase 2: DSPHandle Runtime（P0-A2）
        // ==================================================================
        convo::isr::DSPHandleRuntime dspHandleRuntime_;
        convo::isr::CrossfadeAuthorityRuntime crossfadeAuthorityRuntime_;
        std::mutex runtimeDSPHandleMapMutex_;
        std::unordered_map<DSPCore*, convo::isr::DSPHandle> runtimeDSPHandleMap_;
        std::atomic<convo::isr::CrossfadeId> activeCrossfadeId_{ 0u };

        // ==================================================================

};

#pragma warning(pop) // EngineRuntime deprecation — transitional

inline bool AudioEngine::enqueueLearningCommand(const LearningCommand& cmd) noexcept
{
    // acquire: processLearningCommands publishAtomic release と HB し、learningCommandRead/Write 最新値を取得。
    const uint32_t currentWrite = consumeAtomic(learningCommandWrite, std::memory_order_acquire);
    const uint32_t currentRead = consumeAtomic(learningCommandRead, std::memory_order_acquire);
    const uint32_t next = (currentWrite + 1u) & learningCommandBufferMask;
    if (next == currentRead)
    {
        jassertfalse;
        return false;
    }

    learningCommandBuffer[currentWrite] = cmd;
    // release: バッファ重の書込みを dequeueLearningCommand acquire fence に公開。
    std::atomic_thread_fence(std::memory_order_release);
    // release: キュー埋まりを dequeueLearningCommand consumeAtomic acquire に公開。
    publishAtomic(learningCommandWrite, next, std::memory_order_release);
    return true;
}

[[nodiscard]] inline bool AudioEngine::dequeueLearningCommand(LearningCommand& cmd) noexcept
{
    // acquire: enqueueLearningCommand publishAtomic release と HB し、learningCommandRead/Write 最新値を取得。
    const uint32_t currentRead = consumeAtomic(learningCommandRead, std::memory_order_acquire);
    const uint32_t currentWrite = consumeAtomic(learningCommandWrite, std::memory_order_acquire);
    if (currentRead == currentWrite)
        return false;

    std::atomic_thread_fence(std::memory_order_acquire);
    // acquire: fence でバッファ重の設定を後続のインストラクション下鉅に収所。
    cmd = learningCommandBuffer[currentRead];
    // release: キュー出しを enqueueLearningCommand consumeAtomic acquire に公開。
    publishAtomic(learningCommandRead, (currentRead + 1u) & learningCommandBufferMask, std::memory_order_release);
    return true;
}

inline bool AudioEngine::enqueueLearnerDispatch(const LearnerDispatchAction& action) noexcept
{
    // acquire: processDeferredLearningActions publishAtomic release と HB し、learnerDispatchRead/Write 最新値を取得。
    const uint32_t currentWrite = consumeAtomic(learnerDispatchWrite, std::memory_order_acquire);
    const uint32_t currentRead = consumeAtomic(learnerDispatchRead, std::memory_order_acquire);
    const uint32_t next = (currentWrite + 1u) & learnerDispatchBufferMask;
    if (next == currentRead)
    {
        // release: overflow フラグを processDeferredLearningActions consume acquire に公開。
        publishAtomic(lastFailedAction, action, std::memory_order_release);
        publishAtomic(learnerDispatchOverflow, true, std::memory_order_release);
        return false;
    }

    learnerDispatchBuffer[currentWrite] = action;
    // release: バッファ重の書込みを dequeueLearnerDispatch acquire fence に公開。
    std::atomic_thread_fence(std::memory_order_release);
    // release: キュー埋まりを dequeueLearnerDispatch consumeAtomic acquire に公開。
    publishAtomic(learnerDispatchWrite, next, std::memory_order_release);
    // release: overflow クリアを processDeferredLearningActions consume acquire に公開。
    publishAtomic(learnerDispatchOverflow, false, std::memory_order_release);
    return true;
}

[[nodiscard]] inline bool AudioEngine::dequeueLearnerDispatch(LearnerDispatchAction& action) noexcept
{
    // acquire: enqueueLearnerDispatch publishAtomic release と HB し、learnerDispatchRead/Write 最新値を取得。
    const uint32_t currentRead = consumeAtomic(learnerDispatchRead, std::memory_order_acquire);
    const uint32_t currentWrite = consumeAtomic(learnerDispatchWrite, std::memory_order_acquire);
    if (currentRead == currentWrite)
        return false;

    std::atomic_thread_fence(std::memory_order_acquire);
    // acquire: fence でバッファ重の設定を後続のインストラクション下鉅に収所。
    action = learnerDispatchBuffer[currentRead];
    // release: キュー出しを enqueueLearnerDispatch consumeAtomic acquire に公開。
    publishAtomic(learnerDispatchRead, (currentRead + 1u) & learnerDispatchBufferMask, std::memory_order_release);
    return true;
}


