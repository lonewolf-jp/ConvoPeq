#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// ISRLifetimeProof.h — Shutdown lifetime proof / permit types (dash2 §2.2 / H.11)
//
// ★ REPAIR_PLAN2-dash2.md 実装: Phase A1（H.11.17.5 15-Step 1-6, type only）
//   Step 1: ShutdownRuntimeIdentity
//   Step 2: ShutdownQuiescenceProof（Q0〜Q7 — H.11.11.3）
//   Step 3: ReclaimPermit（move-only / single-use / identity-bound — H.11.11.9.5）
//   Step 6: ReclaimIdentity（DSPHandle + retireSequence — H.11.11.9.5）
//
// ★ 現在は「型のみ・production 未接続」。production reclaim の接続は
//   Phase A2（15-Step 7-15 / A2-G01〜G23 PASS 後）で行う（H.11.17.5）。
//   ReclaimPermit の生成は ShutdownRuntime のみ（friend / AC-X3-11 / INV-LIFE-4）。
//
// ★ 不変条件（H.11.11.7 INV-LIFE）:
//   INV-LIFE-3: ShutdownQuiescenceProof は ShutdownRuntime のみ生成可能
//   INV-LIFE-4: ReclaimPermit は ShutdownRuntime の shutdown transaction からのみ生成可能
//   INV-LIFE-5: Proof.identity == Permit.identity
//   INV-LIFE-6: Permit.identity == current shutdown transaction identity
//   INV-LIFE-7: Permit は一度しか consume できない
//   INV-LIFE-8: Proof 生成後、lifetime obligation を生成する API はすべて reject
//   INV-LIFE-9: Closed → Open は存在しない（no-resurrection）
// ═══════════════════════════════════════════════════════════════════════════

#include <atomic>
#include <cstdint>
#include <optional>
#include <utility>

#include "ISRDSPHandle.h"   // ReclaimIdentity::handle（DSPHandle）

namespace convo {
namespace isr {

// ShutdownRuntime の前方宣言（friend 宣言用。完全定義は ISRShutdown.h）
class ShutdownRuntime;

// ── Step 1: ShutdownRuntimeIdentity ─────────────────────────────────────────
//   shutdown transaction を識別する immutable identity。
//   - engineInstanceId: Runtime インスタンスを一意に識別（cross-runtime confusion 防止 — AUTH-09/13）。
//     Runtime A / Shutdown N と Runtime B / Shutdown N が同一 generation（例: 1）でも、
//     engineInstanceId で区別する（Permit の cross-runtime injection を構造的に防止）。
//   - generation: Shutdown N を一意に識別（Permit ABA 防止 — T10 / Race B）
//   - epochGeneration / readerRegistrationGeneration: Epoch / reader registration の
//     generation を束縛し、Shutdown N の Permit を Shutdown N+1 で使えないようにする
//     （第三者的レビュー反映 3 — permit identity を ShutdownRuntime に束縛）。
struct ShutdownRuntimeIdentity {
    std::uint64_t engineInstanceId{0};                 // Runtime インスタンス一意 ID（AUTH-09/13）
    std::uint64_t generation{0};                       // シャットダウン回数（単調増加）
    std::uint64_t epochGeneration{0};                  // EpochDomain の generation
    std::uint64_t readerRegistrationGeneration{0};     // Reader registration の generation

    [[nodiscard]] bool operator==(const ShutdownRuntimeIdentity& o) const noexcept = default;
    [[nodiscard]] bool isNull() const noexcept { return engineInstanceId == 0 && generation == 0; }
};

// ── Step 6: ReclaimIdentity ─────────────────────────────────────────────────
//   reclaim obligation の identity（H.11.11.9.5）。
//   pendingReclaimHandles_（std::vector<DSPHandle>）を本 identity set へ昇格する際の要素。
//   retireSequence は deterministic ordering（INV-FIFO-1 secondary — memory safety には使わない）。
//   ★ NonRT ReclaimAuthority 専用（第七者 #12 — Audio Thread からは呼ばない / AC-ISR-1）。
struct ReclaimIdentity {
    DSPHandle   handle;            // reclaim 対象 DSPHandle
    std::uint64_t retireSequence{0}; // deterministic ordering（INV-FIFO-1）

    [[nodiscard]] bool operator==(const ReclaimIdentity& o) const noexcept = default;
};

// ── Step 2: ShutdownQuiescenceProof ─────────────────────────────────────────
//   「これ以降、新しい lifetime obligation が発生しない」ことの証明（Q0〜Q7, H.11.11.3）。
//   ⚠️ 循環排除（第五者レビュー）: pendingReclaimIdentities.empty() と
//   LifetimeAccounting.isDrained() は Proof 条件に含めない（ShutdownCompletionProof 側 / C1/C2）。
//   Q0〜Q7 は「quiescence（新 obligation なし）」の証明であり、completion（全消滅）ではない。
//   ［immutable — 生成は ShutdownRuntime::tryMakeQuiescenceProof() のみ（INV-LIFE-3）］
class ShutdownQuiescenceProof {
public:
    ShutdownQuiescenceProof(const ShutdownQuiescenceProof&) = delete;
    ShutdownQuiescenceProof& operator=(const ShutdownQuiescenceProof&) = delete;
    ShutdownQuiescenceProof(ShutdownQuiescenceProof&&) noexcept = default;
    ShutdownQuiescenceProof& operator=(ShutdownQuiescenceProof&&) noexcept = default;

    // Q0〜Q7 全条件が成立した場合のみ生成される（valid() == true）。
    [[nodiscard]] bool valid() const noexcept { return valid_; }

    [[nodiscard]] const ShutdownRuntimeIdentity& identity() const noexcept { return identity_; }

    // Q 条件別の観測結果（監査・診断用。Proof の成否は valid_ で判定）
    [[nodiscard]] bool admissionReservationsZero() const noexcept { return qAdmissionReservationsZero_; }
    [[nodiscard]] bool admissionClosed() const noexcept { return qAdmissionClosed_; }
    [[nodiscard]] bool allProducersJoined() const noexcept { return qAllProducersJoined_; }
    [[nodiscard]] bool readerRegistrationClosed() const noexcept { return qReaderRegClosed_; }
    [[nodiscard]] bool activeReadersZero() const noexcept { return qActiveReadersZero_; }
    [[nodiscard]] bool epochSettled() const noexcept { return qEpochSettled_; }
    [[nodiscard]] bool postStopEnqueueZero() const noexcept { return qPostStopEnqueueZero_; }
    [[nodiscard]] bool noResurrection() const noexcept { return qNoResurrection_; }

private:
    // 生成は ShutdownRuntime のみ（INV-LIFE-3 / AC-X3-11）。簡易生成（if (isFullyDrained()) return ...）
    // を防止（A2-G05 — 全条件を authority から取得して検証するのは tryMakeQuiescenceProof 側の責務）。
    friend class ShutdownRuntime;

    explicit ShutdownQuiescenceProof(ShutdownRuntimeIdentity id) noexcept
        : identity_(id) {}

    ShutdownRuntimeIdentity identity_;

    bool valid_{false};
    bool qAdmissionReservationsZero_{false};   // Q0
    bool qAdmissionClosed_{false};             // Q1
    bool qAllProducersJoined_{false};          // Q2
    bool qReaderRegClosed_{false};             // Q3
    bool qActiveReadersZero_{false};           // Q4
    bool qEpochSettled_{false};                // Q5
    bool qPostStopEnqueueZero_{false};         // Q6
    bool qNoResurrection_{false};              // Q7
};

// ── Step 3: ReclaimPermit ───────────────────────────────────────────────────
//   shutdown transaction の reclaim phase への参加権（H.11.11.9.5）。
//   - move-only（コピー禁止）: 二重 reclaim（reclaim(permit); reclaim(permit);）を構造的防止
//   - single-use: consume() で Consumed state へ原子遷移（INV-LIFE-7 / T9 concurrent double reclaim）
//   - identity-bound: Proof.identity と一致する shutdown transaction でのみ有効（INV-LIFE-5/6 / T10 ABA）
//   - 万能 token ではない: Permit 自体が delete DSPCore を許可するのではなく、
//     ShutdownQuiescent ReclaimAuthority の起動権（第九者 #8 / H.11.11.9.5）。
class ReclaimPermit {
public:
    ReclaimPermit(const ReclaimPermit&) = delete;
    ReclaimPermit& operator=(const ReclaimPermit&) = delete;
    ReclaimPermit(ReclaimPermit&& other) noexcept
        : identity_(other.identity_), state_(other.state_.load(std::memory_order_relaxed))
    {
        // move 元は無効化（同一 Permit の二重使用防止）
        other.state_.store(State::Consumed, std::memory_order_relaxed);
    }
    ReclaimPermit& operator=(ReclaimPermit&& other) noexcept
    {
        if (this != &other) {
            identity_ = other.identity_;
            state_.store(other.state_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            other.state_.store(State::Consumed, std::memory_order_relaxed);
        }
        return *this;
    }

    [[nodiscard]] const ShutdownRuntimeIdentity& identity() const noexcept { return identity_; }

    // single-use consume（linearization point — 第七者 B / INV-LIFE-7）。
    // すでに Consumed の場合は false（二重 reclaim 構造的防止 — T9）。
    [[nodiscard]] bool consume() noexcept
    {
        State expected = State::Issued;
        return state_.compare_exchange_strong(expected, State::Consumed,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire);
    }

    [[nodiscard]] bool isConsumed() const noexcept
    {
        return state_.load(std::memory_order_acquire) == State::Consumed;
    }

private:
    friend class ShutdownRuntime;

    enum class State : uint8_t { Issued = 0, Consumed };

    explicit ReclaimPermit(ShutdownRuntimeIdentity id) noexcept : identity_(id) {}

    ShutdownRuntimeIdentity identity_;
    std::atomic<State> state_{State::Issued};
};

// ── ShutdownRuntime 拡張用 forward declaration ──────────────────────────────
//   tryMakeQuiescenceProof / tryMakeReclaimPermit は ShutdownRuntime のメソッドとして
//   ISRShutdown.h / ISRShutdown.cpp に追加する（H.11.11.3 / H.11.11.9.3）。

} // namespace isr
} // namespace convo
