#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <thread>  // ★ Phase-1: std::this_thread::get_id() Single Thread Owner スレッドガード
#include "AtomicAccess.h"
#include "RuntimePublicationState.h"
#include "TelemetryRecorder.h"
#include "PublicationAdmission.h"
#include "PublicationExecutor.h"
#include "DSPTransition.h"
#include "DSPLifetimeManager.h"
#include "ISRRuntimeSemanticSchema.h"
#include "core/RCUReader.h"
#include "core/TimeUtils.h"

class AudioEngine;
class RuntimePublicationOrchestrator;

namespace convo::isr {

// ★ C-2.1: DeferredGuard — stale discard 用のガード情報
struct DeferredGuard {
    int generation;        // req.generation と同型（int）。uint64_t だった旧型は ISR A-4
                          // 比較 (m.generation != ctx.currentGeneration) で -Wsign-compare
                          // を誘発したため統一（第13回レビュー反映・D-13 ⑤）。
    PublicationSequenceId sequence;
};

// ★ C-2.1: DeferredPublishSlot — sequence 番号付きの deferred publish slot
struct DeferredPublishSlot {
    PublicationAdmission::PublishRequest request;
    DeferredGuard guard;
    PublicationAdmission::DeferredPublishMetadata metadata{};  // ★ Phase-1: enqueue-time immutable snapshot (View.metadata() の参照先)
    DiscardReason lastDiscardReason{DiscardReason::None};
    uint64_t enqueueTimestampUs{0};
};
// ★ Phase-1: DeferredPublishView — move-only Protocol View over a DeferredPublishSlot。
//   Single Thread Owner（RebuildThread）契約下でのみ peek/evaluate/consume/discard が成立。
//   slot 寿命は Orchestrator（deferredSlot_）が保証するため、View は借用ポインタのみ保持。
//   ★ 責務分離 (ADR-C4 §Consequences / design-D4 §134 / §1488):
//     Admission は Decision（+discardReason）のみ返す・Store 触らない。
//     Store mutation は View の consume()/discard() が行うが、ownership Release
//     （deferredSlot_.reset() / hasDeferred_ flip / telemetry）は View が
//     owner_->finishView() へ委譲する（Owner が Owner を管理する = Authority Singularization）。
//     state_ 遷移は View が保持；所有権解除は Orchestrator::finishView() が行う。
//   ★ finishView() は Orchestrator の公開 API。View は owner_ バックポインタで到達する
//     （design-D4 §107/§134/§1488）。consume/discard/dtor は終端で owner_->finishView() を呼ぶ。
//   ★ 実装は RuntimePublicationOrchestrator.cpp で out-of-line 定義（owner_->finishView() の呼出のため、
//     Orchestrator クラス定義完結後 = .cpp 内）。
class DeferredPublishView {
public:
    enum class State : uint8_t { Valid, Consumed, Discarded, MovedFrom };

    DeferredPublishView() noexcept = default;
    DeferredPublishView(RuntimePublicationOrchestrator& owner, DeferredPublishSlot& slot) noexcept
        : owner_(&owner), slot_(&slot), state_(State::Valid) {}

    DeferredPublishView(const DeferredPublishView&) = delete;
    DeferredPublishView& operator=(const DeferredPublishView&) = delete;
    DeferredPublishView(DeferredPublishView&& o) noexcept
        : owner_(o.owner_), slot_(o.slot_), state_(o.state_) {
        o.owner_ = nullptr; o.slot_ = nullptr; o.state_ = State::MovedFrom;
    }
    DeferredPublishView& operator=(DeferredPublishView&& o) noexcept {
        if (this != &o) {
            if (slot_ != nullptr && state_ == State::Valid)
                jassertfalse;  // peek→evaluate→consume/discard 未完了のまま上書き = バグ
            owner_ = o.owner_; slot_ = o.slot_; state_ = o.state_;
            o.owner_ = nullptr; o.slot_ = nullptr; o.state_ = State::MovedFrom;
        }
        return *this;
    }

    ~DeferredPublishView() {
        // ADR-C4 §107: 暗黙 discard しない。Valid のまま破棄 = peek-only 欠陥 → fail-fast。
        // （design-D4 §111 の「re-peek」方針は本 ADR で fail-fast へ更新済み）。
        if (slot_ != nullptr && state_ == State::Valid)
            jassertfalse;
    }

    [[nodiscard]] State state() const noexcept { return state_; }
    [[nodiscard]] bool isValid() const noexcept { return state_ == State::Valid; }

    // Admission へ渡す immutable metadata（slot.metadata の const 参照）。
    // const& 安全は: (a) View が slot 寿命を保証 (Single Thread Owner) +
    // (b) consume/discard 後の呼び出し禁止 (state_ ガード) + (c) slot 非変更契約
    //   （ADR-C4 90-95「将来スレッドモデルが変わる場合は値返し」）。
    [[nodiscard]] const PublicationAdmission::DeferredPublishMetadata& metadata() const noexcept {
        jassert(state_ == State::Valid);
        return slot_->metadata;
    }

    // consume: request を move-out し state_ を State::Consumed へ遷移。
    //   終端で owner_->finishView() を呼び、ownership release（slot reset / hasDeferred_ flip /
    //   telemetry）を Orchestrator が一括実行する（design-D4 §106/§134/§426-427）。
    //   move-out は finishView の slot reset 前に行う（req は view 外で生存）。
    [[nodiscard]] PublicationAdmission::PublishRequest consume() noexcept;

    // discard: Admission が決定した DiscardReason を記録し state_ を State::Discarded へ遷移。
    //   終端で owner_->finishView() を呼び ownership release を行う（ADR principle: Admission = reason 決定のみ）。
    void discard(DiscardReason reason) noexcept;

    // ★ D162-2-B (S2): slot 保持中の request の値コピー（Single Thread Owner 契約内の読み取り）。
    //   discard/consume 前に DSP disposition 用の handle を snapshot するために使用する。
    //   Valid 状態でのみ呼ぶこと（それ以外は default request を返す）。
    [[nodiscard]] PublicationAdmission::PublishRequest peekRequestCopy() const noexcept {
        if (state_ != State::Valid || slot_ == nullptr)
            return PublicationAdmission::PublishRequest{};
        return slot_->request;
    }

private:
    RuntimePublicationOrchestrator* owner_{nullptr};
    DeferredPublishSlot* slot_{nullptr};
    State state_{State::MovedFrom};
};

// RuntimePublicationOrchestrator: AudioEngine レベルの publish オーケストレーション。
// Coordinator::submitPublishRequest() の実装を提供する。
// AudioEngine に注入され、Admission → Executor → DSPTransition の順で委譲する。
//
// ★ activate (DSP スロット書き換え) は publish 成功後に行う。
// ★ submitPublishRequest → evaluate → Accepted → execute の順を厳守。
//
// ★ v19: StateOwner + TelemetryRecorder の両方を保持。
//   Orchestrator が stateOwner.onXxx() + telemetryRecorder.recordXxx() を呼ぶ。
class RuntimePublicationOrchestrator {
public:
    explicit RuntimePublicationOrchestrator(AudioEngine& engine, uint64_t engineInstanceId) noexcept;

    // [work37 Phase 6] Deferred Publish TTL — 30秒超過で破棄
    static constexpr uint64_t kDeferredPublishTTLUs = 30'000'000;  // 30秒

    // trySubmit: publish 要求を試行する。
    // Admission → Accepted の場合のみ Executor → DSPTransition まで実行。
    // Deferred/Rejected の場合は caller が適切に処理するよう決定値を返す。
    // Returns: Admission::Decision (Accepted: 全処理完了 / Deferred: 保留 / Rejected*: 却下)
    [[nodiscard]] PublicationAdmission::Decision trySubmit(const PublicationAdmission::PublishRequest& req) noexcept;

    // trySubmitImpl: trySubmit / submitPublishRequest の共通実装。
    //   常に同期 publish。"submitPublishRequest() = Admission → Accepted なら同期 Publish" で
    //   API 意味論を一本化する。fire-and-forget (waitForReceipt=false) 経路は Coordinator が
    //   Builder を兼ねることによる自己待ちを招いたため削除。deferred resubmit は
    //   Coordinator が RebuildThread へハンドオフし、RebuildThread が同期 submitPublishRequest を
    //   実行する（receipt は CoordinatorLoop の processIntent が配送するため自己待ちにならない）。
    [[nodiscard]] PublicationAdmission::Decision trySubmitImpl(
        const PublicationAdmission::PublishRequest& req) noexcept;

    // ★ (a) Completion layer — ISR post-commit notifier.
    //   Single seam for "publish committed"; the ISR PublishExecutor routes here
    //   (NOT via IntentHandlerContext), keeping intent handlers HANDLER-1 (pure).
    //   Audio-thread trySubmit retains its inline completion (unchanged).
    void onPublishCommitted(PublicationSequenceId seqId, std::uint64_t recoveryObligationId) noexcept;

    // ★ A3 Step 5-3: access to the stateless publish-completion facade (ADR-D2), owned by
    //   the orchestrator (audio-thread world-publish path). Bound into IntentHandlerContext
    //   by processIntent so the ISR publish execution tail can drive activate/crossfade/retire.
    [[nodiscard]] DSPTransition& transition() noexcept { return transition_; }

    // submitPublishRequest: publish 要求を処理する (deferred は自動 enqueue)。
    //   常に同期。deferred resubmit は RebuildThread が本 API を呼ぶことで実行される。
    void submitPublishRequest(const PublicationAdmission::PublishRequest& req) noexcept;

    // hasDeferredRequest: 保留中 publish 要求確認 (DrainAudit / RuntimeHealth / Stall 用)。
    [[nodiscard]] bool hasDeferredRequest() const noexcept { return convo::consumeAtomic(hasDeferred_, std::memory_order_acquire); }
    // ★ D135-1: test 用 — retry 世代/カウント取得 (hasDeferred と併せて観測)
    [[nodiscard]] int getRetryGeneration() const noexcept { return deferredRetryGeneration_; }
    [[nodiscard]] uint8_t getRetryCount() const noexcept { return deferredRetryCount_; }

    // ★ D135-1: crossfade-timeout recovery 発行 world の seq を記録/取得。
    void setRecoveryPublishSeq(PublicationSequenceId seq) noexcept {
        convo::publishAtomic(lastRecoveryPublishSeq_, seq, std::memory_order_release);
    }
    [[nodiscard]] PublicationSequenceId recoveryPublishSeq() const noexcept {
        return convo::consumeAtomic(lastRecoveryPublishSeq_, std::memory_order_acquire);
    }
    // ★ D135-1 / F6: recovery が zero-fading world を commit した時点で obligation metadata をリセット。
    //   identity key・count・createdAt を初期化し、recovery 後の再駆動を新 dwell として扱う。
    //   （F6: retention は count を増やさないため count は実質常に 0。recovery provenance は維持。）
    void resetDeferredRetryBudget() noexcept {
        jassert(std::this_thread::get_id() == engine_.rebuildThreadId());
        deferredRetryGeneration_ = 0;
        deferredRetryObligationId_ = 0;
        deferredRetryCount_ = 0;
        deferredObligationCreatedAtUs = 0;
    }

    // ★ Phase-1: peekDeferred — consumeDeferredRequest の後継 (View 経由のみ)（design-D4 D-13 ①）。
    //   hasDeferred_ は反転しない（peek only）。実際の ownership release は
    //   DeferredPublishView.consume()/discard() 内の owner_->finishView() が担う。
    //   Single Thread Owner（RebuildThread）契約の jassert 付き (ADR-C4:100-105)。
    [[nodiscard]] std::optional<DeferredPublishView> peekDeferred() noexcept;

    // ★ Phase-1: finishView — ownership Release の唯一口（design-D4 §1488 / §139-140）。
    //   DeferredPublishView.consume()/discard() が owner_->finishView() を呼ぶ。
    //   slot reset + hasDeferred_ flip + DeferredHealth telemetry を一括実行。
    void finishView() noexcept;

    // ★ Phase-1: processDeferredAdmission — RebuildThread 専用の atomic flow
    //   (design-D4 D-13 ④ / ADR-C4:113)。peek → evaluateDeferred → consume/discard →
    //   releaseSlot → (Ready なら submitPublishRequest；再 Deferred は submitPublishRequest
    //   が自動 enqueue)。AudioEngine.h:2528-2529 の
    //   'consumeDeferredRequest → submitPublishRequest' を一本化。
    // ★ D135-8 Step 7 (P3): wake-provenance discriminator wired end-to-end (P1 blocked → resolved).
    void processDeferredAdmission(bool wasRecoveryWake) noexcept;

    // ★ C-2.2: shutdown 時に deferred publish を強制消去
    void clearDeferredForShutdown() noexcept;

    // ★ D135-8 Step 9 (Option C): deferred-clear wake-provenance latch.
    //   requestDeferredClear() is the sole Writer (non-RebuildThread, Timer.cpp);
    //   drainDeferredClearIfRequested() is the sole Reader (RebuildThread only).
    //   deferredClearRequested_ is PREDICATE-INELIGIBLE: persistent latch consumed
    //   on every wake, so a lost notify_one cannot starve the clear.
    void requestDeferredClear() noexcept;
    bool drainDeferredClearIfRequested() noexcept;

    // ★ A-2.5: DrainAudit 用 — deferred publish 最長滞留時間
    [[nodiscard]] uint64_t getMaxDeferredAgeMs() const noexcept;

    // ★ C-2.1: 監査用 — deferred overwrite 回数
    [[nodiscard]] std::uint64_t deferredOverwriteCount() const noexcept;

    // ── StateOwner アクセサ ──
    [[nodiscard]] RuntimePublicationStateOwner& stateOwner() noexcept { return stateOwner_; }
    [[nodiscard]] const RuntimePublicationStateOwner& stateOwner() const noexcept { return stateOwner_; }

    // ★ C-1: World 退役を Pipeline Ledger に通知
    void notifyWorldRetired(uint64_t worldId) noexcept {
        stateOwner_.onRetired(worldId);
    }

    // ── TelemetryRecorder アクセサ ──
    [[nodiscard]] TelemetryRecorder& telemetryRecorder() noexcept { return telemetryRecorder_; }
    [[nodiscard]] const TelemetryRecorder& telemetryRecorder() const noexcept { return telemetryRecorder_; }

    // ★ P1-B: Admission に HealthState 参照を設定
    void setAdmissionHealthStateRef(const std::atomic<ISRHealthState>* ref) noexcept {
        admission_.setHealthStateRef(ref);
    }

    // ── 健全性スナップショット ──
    void publishHealthSnapshot(uint64_t externalReclaimedCount) noexcept;

    // ── CorrelationId 採番 ──
    [[nodiscard]] CorrelationId nextCorrelationId() noexcept;

    // ★ P1-6: 出版停滞監視 — 進捗観測の更新（非const、timerCallback から呼ぶ）
    void updateProgressObservation() noexcept {
        PublicationSequenceId current = engine_.getLastCommittedPublicationSequence();
        PublicationSequenceId last = m_lastObservedSequence.load(std::memory_order_relaxed); // NOLINT(atomic-dot-call): relaxed counter
        if (current > last) {
            m_lastObservedSequence.store(current, std::memory_order_relaxed); // NOLINT(atomic-dot-call): relaxed counter
            m_lastProgressTimestampUs.store(getCurrentTimeUs(), std::memory_order_relaxed); // NOLINT(atomic-dot-call): relaxed timestamp
        }
    }

    // ★ P1-6: 出版停滞監視 — 停滞検出（const、read-only）
    [[nodiscard]] bool isPublicationStalled() const noexcept {
        uint64_t elapsed = getCurrentTimeUs()
            - convo::consumeAtomic(m_lastProgressTimestampUs, std::memory_order_acquire);
        return elapsed >= kPublicationStallThresholdUs;
    }

    // ★ P1-6: prepareToPlay での再初期化用
    void resetProgressObservation() noexcept {
        convo::publishAtomic(m_lastProgressTimestampUs, getCurrentTimeUs(), std::memory_order_release);
    }

    // ★ P1-6: RuntimeHealthMonitor からのアクセス用
    [[nodiscard]] uint64_t getPendingIntentCount() const noexcept {
        return engine_.getRetirePendingIntentCount();
    }

    // ★ P1-6: PublicationBacklog の公開（RuntimeHealthMonitor → Orchestrator → AudioEngine → bridge）
    [[nodiscard]] uint64_t getPublicationBacklogCount() const noexcept {
        return engine_.getPublicationBacklogCount();
    }

    // ★ D162-2-B (C′): registered DSP の terminal disposition authority。
    //   INV-D162-1/3: enqueuePublicationIntentForRuntimeCommit で handle 登録済みの DSPCore を
    //   手放す全経路（deferred overwrite/discard/shutdown clear/admission reject/dormant
    //   retry-exhausted）は、この helper 経由で DSPLifetimeManager::retire
    //   （= retireDSPHandleForRuntime による台帳解除 + enqueueWithRetry による EBR 破壊権取得）
    //   を 1 回だけ実行する。retireDSPHandleForRuntime を単独で terminal disposition として
    //   使用しない（台帳解除のみで DSPCore が orphan 化する — D162-2-A §3）。
    //   例外安全: retire は noexcept、null-safe（未登録 DSP → false → no-op、
    //   DSPGuard/destroyRolledBackDSP 契約の direct-destroy 対象には触れない）。
    //   呼出スレッド: RebuildThread（deferred 系）／Message Thread（shutdown clear）— 全て NonRT。
    void retireRegisteredDSP(const PublicationAdmission::PublishRequest& req,
                             const char* origin) noexcept;

private:
    // ★ P1-6: 出版停滞監視用フィールド（30秒以上 sequence が進まない場合に stall 検出）
    static constexpr uint64_t kPublicationStallThresholdUs = 30'000'000;
    std::atomic<PublicationSequenceId> m_lastObservedSequence {0};
    std::atomic<uint64_t> m_lastProgressTimestampUs {0};
    // ★ C-2.1: std::optional<PublishRequest> → DeferredPublishSlot
    std::optional<DeferredPublishSlot> deferredSlot_;
    std::atomic<bool> hasDeferred_{false};

    // ★ C-2.1: 監査カウンタ
    std::atomic<uint64_t> deferredOverwriteCount_{0};
    std::atomic<uint64_t> maxDeferredAgeMs_{0};

    // ★ D135-1 / F6: Deferred obligation accounting — rebuild-thread single-owner。
    //   obligation identity = (generation, recoveryObligationId)。generation 単独ではない
    //   （recovery は同一 generation を再利用して別 payload を発行する — RebuildDispatch:1035-1036）。
    //   enqueueDeferred は deferredSlot_ を構造体ごと置換するため counter/createdAt は Orchestrator
    //   メンバに保持する（slot 内ではない — consume→finishView で slot は消えるため、re-drive 時に
    //   旧 slot から timestamp は読めない）。rebuild-thread 専用（同一スレッド）で rebuildMutex 不要。
    //   ★ F6 契約: DeferredFadingActive の再駆動は「retention（保持）」であり retry ではない。
    //   同一 (G,O) の re-drive では deferredRetryCount_ を増加させない → RetryExhaustedDiscard は
    //   retention に対して発生しない。kMaxDeferredRetries は将来の真の Type-A retry 経路用の
    //   dormant guard として保持（削除しない）。判定は現行実装 `>` を正とする（F5-9/F6-6）。
    int deferredRetryGeneration_{0};                 // 現在の obligation identity: generation
    std::uint64_t deferredRetryObligationId_{0};     // 現在の obligation identity: recoveryObligationId（0=通常）
    uint8_t deferredRetryCount_{0};                  // Type-A retry 回数（retention では不増加 → 現行 0 固定）
    std::uint64_t deferredObligationCreatedAtUs{0};  // obligation 生成時刻（TTL source-of-truth、re-drive で維持）
    static constexpr uint8_t kMaxDeferredRetries = 2;   // F6: retention に適用しない（dormant guard）
    // ★ D135-1: crossfade-timeout recovery が発行した idle world の seq。
    std::atomic<PublicationSequenceId> lastRecoveryPublishSeq_{0};
    // ★ D135-8 Step 9 (Option C): clear-provenance latch. std::atomic because
    //   requestDeferredClear() (Timer.cpp, non-RebuildThread) writes and
    //   drainDeferredClearIfRequested() (RebuildThread) reads. Predicate-ineligible;
    //   persistent, drained on every wake (no lost clear).
    std::atomic<bool> deferredClearRequested_{false};

    void enqueueDeferred(const PublicationAdmission::PublishRequest& req) noexcept;

    // ★ F6-7: obligation metadata（identity key + createdAt + count）を無効化する。
    //   terminal（Accepted / StaleDiscard / Expired / ShutdownDiscard）で呼ぶ。
    //   re-drive（retention）では呼ばない（enqueueDeferred が key 一致で維持するため）。
    void invalidateDeferredObligation() noexcept;

    // ★ Phase-1 private helpers (Single Thread Owner / RebuildThread 前提)
    [[nodiscard]] PublicationAdmission::DeferredAdmissionSnapshot buildDeferredAdmissionSnapshot() const noexcept;

    AudioEngine& engine_;

    // ★ v19: StateOwner + TelemetryRecorder (分離)
    RuntimePublicationStateOwner stateOwner_;
    TelemetryRecorder telemetryRecorder_;

    PublicationAdmission admission_;
    PublicationExecutor executor_;
    DSPTransition transition_;
    DSPLifetimeManager lifetime_;
    convo::RCUReader publicationReader;
};

} // namespace convo::isr
