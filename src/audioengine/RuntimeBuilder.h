#pragma once

#include "AudioEngine.h"
#include "RuntimeBuildTypes.h"

namespace convo {

// ★ v8.3: RuntimePublishSpecification — Orchestrator が生成する Specification。
//   Builder の内部型ではなく、独立した契約型。
//   事実上 RuntimePublishWorld の DTO（Data Transfer Object）であり、
//   Builder はこれを忠実に World に写像する Deterministic Construction を行う。
//   version フィールドにより Builder/Orchestrator 間の互換性を管理。
//   将来のパラメータ増加に備え TopologyPart / ExecutionPart / RoutingPart の
//   三部構成を維持し、Latency / Resources / Diagnostics 等の追加は各部の
//   拡張または新 Part 追加で対応。
struct RuntimePublishSpecification {
    uint32_t version = 1;  // Specification バージョン（Builder/Orchestrator 間互換性管理）

    struct TopologyPart {
        const AudioEngine::DSPCore* activeDSP = nullptr;  // Active ノード（nullptr=該当なし）
        const AudioEngine::DSPCore* fadingDSP = nullptr;  // Fading ノード（nullptr=クロスフェードなし）
    } topology;

    struct ExecutionPart {
        bool transitionActive = false;       // トランジション有効（Orchestrator が決定）
        int transitionPolicy = 0;            // convo::TransitionPolicy（0=HardReset, 1=SmoothOnly, 2=DryAsOld）
        double fadeTimeSec = 0.0;            // フェード時間（秒）
    } execution;

    struct RoutingPart {
        int processingOrder = 0;             // 0=EQ→Conv, 1=Conv→EQ
        bool eqBypassed = false;
        bool convBypassed = false;
    } routing;

    // ★ v9.4 P0: ProcessingPart — Builder が直接読んでいた engine atomic 8値を格納。
    //   Orchestrator が sealedSnapshot または engine atomic から収集して設定する。
    //   processingOrder/eqBypassed/convBypassed は RoutingPart と重複するが、
    //   ProcessingPart が一次情報源であり RoutingPart は後方互換性のため維持。
    struct ProcessingPart {
        int processingOrder = 0;             // 0=EQ→Conv, 1=Conv→EQ
        bool eqBypassed = false;
        bool convBypassed = false;
        bool softClipEnabled = false;
        float saturationAmount = 0.0f;
        float inputHeadroomGain = 1.0f;
        float outputMakeupGain = 1.0f;
        float convolverInputTrimGain = 1.0f;
    } processing;

    // ★ v9.5 P1: PublicationSnapshotPart — Publication 履歴情報のスナップショット。
    //   Orchestrator が Coordinator から取得して格納する。Builder はこの値のみを参照し、
    //   Coordinator に直接問い合わせない（INV-12/13）。
    struct PublicationSnapshotPart {
        convo::isr::PublicationSequenceId previousCommittedSequence = 0;
    } publicationSnapshot;

    // ★ v9.5 P2: CrossfadeSnapshotPart — Crossfade Runtime のスナップショット。
    //   Orchestrator が engine.crossfadeRuntime_ から収集して格納する。
    //   Builder はこの値のみを参照し、CrossfadeRuntime に直接問い合わせない。
    struct CrossfadeSnapshotPart {
        int startDelayBlocks = 0;
        int dryHoldSamples = 0;
        double dryScaleTarget = 1.0;
        bool firstIrDryCrossfadePending = false;
    } crossfade;

    // ★ v9.5 P2: LatencyPart — レイテンシ状態のスナップショット。
    //   Orchestrator が engine atomic から収集して格納する。
    struct LatencyPart {
        int latencyDelayOld = 0;
        int latencyDelayNew = 0;
    } latency;

    // ★ v9.5 P1 phase2: CurrentRuntimeWorld — Orchestrator が observePublishedWorld() から取得した
    //   現在の RuntimePublishWorld ポインタ。Builder はこれを makeEngineRuntimeState() の引数として
    //   使用する。Runtime Query は Orchestrator が事前に実行し、結果のスナップショットを渡す。
    const RuntimePublishWorld* currentRuntimeWorld = nullptr;

    // ★ v9.7 P7-A1: RetirePart — Retire Queue 深度のスナップショット。
    //   Orchestrator が engine atomic から収集して格納する。
    //   Builder はこの値のみを参照し、engine.retireQueueDepth_ に直接アクセスしない。
    struct RetirePart {
        std::uint64_t retireQueueDepth = 0;
    } retire;

    // ★ v9.7 P7-A2: AdaptivePart — Adaptive 係数バンクインデックスと世代番号のスナップショット。
    //   Orchestrator が engine.currentAdaptiveCoeffBankIndex および bank.generation から収集して格納する。
    //   Builder はこの値のみを参照し、engine の atomic や CoeffBank に直接アクセスしない。
    struct AdaptivePart {
        int coeffBankIndex = -1;
        std::uint64_t coeffGeneration = 0;
    } adaptive;
};

enum class BuildError {
    None,
    InvalidInput,
    ResourceUnavailable,
    MKLFailure,          // ★ C-2: MKL 初期化・FFT 計画失敗
    ConvolverFailure,    // ★ C-2: Convolver Build 失敗
    PrepareFailure,      // ★ C-2: DSPCore::prepare() 失敗
    WarmupFailed,
    InternalError
};

struct BuildResult {
    AudioEngine::DSPCore* runtime = nullptr;
    BuildError error = BuildError::None;
    bool prepared = false;
};

const char* toString(BuildError error) noexcept;

class RuntimeBuilder {
public:
    explicit RuntimeBuilder(AudioEngine& owner) noexcept : engine(owner) {}

    // ★ v8.3: const RuntimePublishWorld を返す — INV-11 のコンパイル時保証
    //   buildRuntimePublishWorld() 完了後は World を変更してはならない。
    //   内部で Coordinator が sealRecursively() を呼ぶ必要があるため、
    //   RuntimePublicationCoordinator が const_unique_ptr を受け入れる。
    [[nodiscard]] convo::aligned_unique_ptr<const RuntimePublishWorld>
    buildRuntimePublishWorld(
        const convo::RuntimeBuildSnapshot* sealedSnapshot,
        const RuntimePublishSpecification& spec = RuntimePublishSpecification{}) noexcept;

    // Old signature (bootstrap backward compat — non-const for Coordinator compatibility)
    [[nodiscard]] convo::aligned_unique_ptr<RuntimePublishWorld>
    buildRuntimePublishWorld(AudioEngine::DSPCore* current,
                             AudioEngine::DSPCore* next,
                             convo::TransitionPolicy policy,
                             double fadeTimeSec,
                             bool active,
                             const convo::RuntimeBuildSnapshot* sealedSnapshot = nullptr) noexcept
    {
        RuntimePublishSpecification spec;
        spec.topology.activeDSP = current;
        spec.topology.fadingDSP = next;
        spec.execution.transitionActive = active;
        spec.execution.transitionPolicy = static_cast<int>(policy);
        spec.execution.fadeTimeSec = fadeTimeSec;
        // ★ v9.5 fallback: old sig callers には sealedSnapshot がないため、engine atomic から
        //   ProcessingPart/CrossfadeSnapshotPart/LatencyPart/PublicationSnapshotPart を設定する。
        //   （新しい Orchestrator 経由の呼び出し元は sealedSnapshot から Orchestrator が設定済み）
        spec.processing.processingOrder = static_cast<int>(convo::consumeAtomic(engine.currentProcessingOrder, std::memory_order_relaxed));
        spec.processing.eqBypassed = convo::consumeAtomic(engine.eqBypassActive, std::memory_order_relaxed);
        spec.processing.convBypassed = convo::consumeAtomic(engine.convBypassActive, std::memory_order_relaxed);
        spec.processing.softClipEnabled = convo::consumeAtomic(engine.softClipEnabled, std::memory_order_relaxed);
        spec.processing.saturationAmount = static_cast<float>(convo::consumeAtomic(engine.saturationAmount, std::memory_order_relaxed));
        spec.processing.inputHeadroomGain = static_cast<float>(convo::consumeAtomic(engine.inputHeadroomGain, std::memory_order_relaxed));
        spec.processing.outputMakeupGain = static_cast<float>(convo::consumeAtomic(engine.outputMakeupGain, std::memory_order_relaxed));
        spec.processing.convolverInputTrimGain = static_cast<float>(convo::consumeAtomic(engine.convolverInputTrimGain, std::memory_order_relaxed));
        // ★ Sync RoutingPart for backward compat（ProcessingPart が一次情報源）
        spec.routing.processingOrder = spec.processing.processingOrder;
        spec.routing.eqBypassed = spec.processing.eqBypassed;
        spec.routing.convBypassed = spec.processing.convBypassed;
        spec.publicationSnapshot.previousCommittedSequence = engine.getLastCommittedPublicationSequence();
        spec.crossfade.startDelayBlocks = engine.crossfadeRuntime_.getStartDelayBlocks();
        spec.crossfade.dryHoldSamples = engine.crossfadeRuntime_.getDryHoldSamples();
        spec.crossfade.dryScaleTarget = engine.crossfadeRuntime_.getDryScaleTarget();
        spec.crossfade.firstIrDryCrossfadePending = engine.crossfadeRuntime_.isFirstIrDryPending();
        spec.latency.latencyDelayOld = convo::consumeAtomic(engine.latencyDelayOld, std::memory_order_relaxed);
        spec.latency.latencyDelayNew = static_cast<int>(convo::consumeAtomic(engine.latencyDelayNew, std::memory_order_relaxed));
        // ★ v9.7 P7-A1: RetirePart — old sig fallback: engine atomic から収集
        spec.retire.retireQueueDepth = convo::consumeAtomic(engine.retireQueueDepth_, std::memory_order_relaxed);
        // ★ v9.7 P7-A2: AdaptivePart — old sig fallback: engine atomic から収集
        {
            const int bankIdx = convo::consumeAtomic(engine.currentAdaptiveCoeffBankIndex, std::memory_order_relaxed);
            spec.adaptive.coeffBankIndex = bankIdx;
            if (bankIdx >= 0 && bankIdx < static_cast<int>(kNumAdaptiveCoeffBanks))
            {
                const auto& bank = engine.getAdaptiveCoeffBankForIndex(bankIdx);
                spec.adaptive.coeffGeneration = convo::consumeAtomic(bank.generation, std::memory_order_relaxed);
            }
        }
        spec.currentRuntimeWorld = engine.observePublishedWorld();
        // const 版に委譲 → unique_ptr<const T> から unique_ptr<T> へは
        // ムーブ不可のため、内部実装呼び出し
        auto constWorld = buildRuntimePublishWorld(sealedSnapshot, spec);
        // const_cast: FrozenRuntimeWorld/Coordinator 経路では seal 後に immutable 化される
        return convo::aligned_unique_ptr<RuntimePublishWorld>(
            const_cast<RuntimePublishWorld*>(constWorld.release()));
    }

    // Bootstrap World: 初期化時に初回 publish する最小限の RuntimePublishWorld を生成する。
    // 全てのデフォルト値で初期化され、AudioEngine::initialize() 実行直後に
    // RuntimePublicationCoordinator::publishWorld() で公開される。
    // これにより publishRuntimeStateNonRt が初回コール時に world==nullptr の
    // fallback を必要としなくなる。
    [[nodiscard]] convo::aligned_unique_ptr<RuntimePublishWorld>
    createBootstrapWorld() noexcept;

    BuildResult build(const BuildInput& in,
                      const ConvolverProcessor::BuildSnapshot& convolverBuildSnapshot) noexcept;
    // Warmup validation (still used by RebuildDispatch)
    BuildError validateWarmup(const AudioEngine::DSPCore& runtime) const noexcept;

private:
    AudioEngine& engine;
};

} // namespace convo
