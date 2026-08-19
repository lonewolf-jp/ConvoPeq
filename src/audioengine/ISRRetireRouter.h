#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cassert>
#include <functional>
#include <vector>

// ISR P1-19: 公開APIに EpochDomain 型を露出しない。
// コンストラクタは IEpochProvider& を受け取り、内部でダウンキャストする。
#include "../DeferredDeletionQueue.h" // DeletionEntryType
#include "core/IEpochProvider.h"
#include "core/IRetireRouter.h"
#include "ISRAuthorityClass.h"
#include "ISRDSPHandle.h"
#include "RetireQuarantineStore.h"   // ★ BUG-015/027 (work88): retire enqueue 失敗時の退避ストア

namespace convo {
namespace isr {

// [work21 P1-3] Retire lifecycle states
enum class RetireState : uint8_t
{
    Created = 0,
    Active,
    PendingRetire,
    Retiring,
    Reclaimed
};

// [work21] Forward declarations for Policy lanes
class DSPRetirePolicy;
class SnapshotRetirePolicy;
class DeferredRetirePolicy;
class WorldRetirementReferenceObserver;   // ★ T1 (D98): reference observer（non-owning 参照のみ・前方宣言）

/**
 * TerminalReclaimAuthority — Final ownership authority when all bounded stores
 * are exhausted (DeferredDeletionQueue + RetireQuarantineStore + EmergencyQ).
 *
 * Ownership contract:
 *   - enqueueWithRetry() transfers ptr to this authority ONLY when all stores + EmergencyQ are full.
 *   - If epoch is safe (isOlder(world_epoch, minReaderEpoch) == false), deleter
 *     executes immediately (synchronous destruction in Non-RT context).
 *   - If epoch is unsafe, the entry is retained in the internal pending list.
 *   - drain() attempts to reclaim epoch-safe entries (called from tryReclaim).
 *   - drainAll() force-releases all entries (called during shutdown, audio thread stopped).
 *
 * ★ P-4 (15-P-4): The pending list is GROWABLE (std::vector). This guarantees the
 *   authority ALWAYS accepts an entry — there is NO "store full" failure path.
 *   Rationale: all callers are Non-RT (verified in 15-P-4-0), so heap allocation
 *   is acceptable. This eliminates the EBR-failure leak: "全 store full / epoch
 *   unsafe" can never leave ptr unowned, because this authority always takes
 *   ownership and waits for the epoch to become safe (drain()).
 *
 * This class is NOT used for shutdown-only paths. Shutdown uses
 * ShutdownReclaimAuthority (see AudioEngine.h drainAll / shutdown path).
 *
 * ISR P1-19 conformance: epoch provider is injected, no EpochDomain exposure.
 */
class TerminalReclaimAuthority {
public:
    struct Entry {
        void* ptr = nullptr;
        void (*deleter)(void*) = nullptr;
        uint64_t epoch = 0;
        DeletionEntryType type = DeletionEntryType::Generic;
        const char* reason = nullptr;
    };

    // Store an entry. Called when all bounded stores + EmergencyQ are full.
    // ★ P-4: ALWAYS returns true (growable store) — ownership always transfers.
    bool store(void* ptr, void (*deleter)(void*), uint64_t epoch,
               DeletionEntryType type, const char* reason) noexcept;

    // Drain epoch-safe entries. Executes deleter for entries where isOlder(entry.epoch, minReaderEpoch) == true.
    // isOlderFn: EpochDomain::isOlder(a, b) == static_cast<int64_t>(a - b) < 0
    void drain(uint64_t minReaderEpoch,
               const std::function<bool(uint64_t, uint64_t)>& isOlderFn) noexcept;

    // Drain ALL entries unconditionally (shutdown only — audio thread must be stopped).
    void drainAll() noexcept;

    // Try to reclaim epoch-safe entries and, if any are safe, destroy immediately.
    // Returns true if at least one entry was reclaimed.
    bool tryReclaim(uint64_t minReaderEpoch,
                    const std::function<bool(uint64_t, uint64_t)>& isOlderFn) noexcept;

    [[nodiscard]] std::size_t residentCount() const noexcept;
    // ★ E-1.9-A: ロックフリー滞留カウンタ読み取り（empty-drain suppression 用）
    [[nodiscard]] uint32_t residentCountAtomic() const noexcept
    {
        return convo::consumeAtomic(residentAtomic_, std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t reclaimCount() const noexcept {
        return convo::consumeAtomic(reclaimCount_, std::memory_order_acquire);
    }

    // ★ P-4: Record a World reclaim (synchronous destruction path in ISRRetireRouter).
    //   Increments reclaimCount_ and notifies the reference observer (non-owning).
    void recordWorldReclaim() noexcept {
        ++reclaimCount_;
        if (referenceObserver_ != nullptr)
            referenceObserver_->onRelease();
    }

    void setReferenceObserver(WorldRetirementReferenceObserver* observer) noexcept {
        referenceObserver_ = observer;
    }

private:
    // ★ P-4: Growable store (std::vector) — Non-RT only, heap allocation acceptable.
    //   Guarantees store() ALWAYS succeeds → no EBR-failure leak path.
    std::vector<Entry> entries_;
    mutable std::mutex mtx_;  // Non-RT only — std::mutex acceptable
    std::atomic<std::uint64_t> reclaimCount_{0};
    WorldRetirementReferenceObserver* referenceObserver_ = nullptr;  // non-owning
    // ★ E-1.9-A: ロックフリー滞留カウンタ（Phase E §1.9-A empty-drain suppression）
    std::atomic<uint32_t> residentAtomic_{0};
};

/**
 * ISRRetireRouter — Thin stateless dispatcher for retire operations.
 *
 * [work21 P0-1] Design constraints:
 *   Allowed: route / enqueue / observer factory
 *   Forbidden: state / policy / decision (delegated to Policy lanes)
 *
 * This class is the SINGLE public entry point for all retire operations.
 * It wraps EpochDomain internally so that callers do not need direct
 * EpochDomain reference. Policy lanes (DSPRetirePolicy, SnapshotRetirePolicy,
 * DeferredRetirePolicy) handle actual execution logic.
 *
 * Phase-C target: All EpochDomain direct call sites migrate to this API.
 *
 * ISR P1-19 conformance: EpochDomain 完全型は .cpp のみでインクルード。
 *   .h では前方宣言のみで十分（コンストラクタの参照パラメータとポインタメンバ）。
 */
class ISRRetireRouter : public convo::IEpochProvider,
                        public convo::IRetireRouter
{
public:
    explicit ISRRetireRouter(convo::IEpochProvider& provider,
                             convo::isr::WorldRetirementReferenceObserver* referenceObserver = nullptr) noexcept;

    ISRRetireRouter(const ISRRetireRouter&) = delete;
    ISRRetireRouter& operator=(const ISRRetireRouter&) = delete;
    ISRRetireRouter(ISRRetireRouter&&) = delete;
    ISRRetireRouter& operator=(ISRRetireRouter&&) = delete;

    // ── Epoch API (Router経由でEpochDomainを間接参照、実装は .cpp) ──

    uint64_t snapshotEpoch() const noexcept;
    uint64_t publishEpoch() noexcept override;
    uint32_t activeReaderCount() const noexcept override;
    int readerCapacity() const noexcept override;
    uint64_t currentEpoch() const noexcept override;
    uint64_t getMinReaderEpoch() const noexcept override;
    int registerReaderThread() noexcept override;
    bool reserveReaderThread(int readerIndex) noexcept override;
    void enterReader(int readerIndex) noexcept override;
    void exitReader(int readerIndex) noexcept override;
    uint64_t minReaderEpoch() const noexcept;

    // ★ B-1: Reader Slot 詳細取得 (delegates to provider)
    [[nodiscard]] ReaderSlotDetail getReaderSlotDetail(int readerIndex) const noexcept override;

    // ★ Practical-1: Reader Stuck 診断 (delegates to provider)
    [[nodiscard]] StuckReaderInfo detectStuckReaders(uint64_t stuckThreshold) const noexcept override;

    // ★ A-2: EBR Queue Visibility 統計 (delegates to provider)
    [[nodiscard]] uint64_t reclaimAttemptCount() const noexcept override;
    [[nodiscard]] uint64_t reclaimSuccessCount() const noexcept override;

    // ── Retire API (実装は .cpp) ──

    RetireEnqueueResult enqueueRetire(void* ptr,
                                      void (*deleter)(void*),
                                      uint64_t epoch,
                                      DeletionEntryType type) noexcept;
    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept override;

    // ★ Bug2 Phase1: リトライロジックを内包した enqueue（Authority 集約）
    //   内部で tryReclaim + 再試行を行い、最終失敗時に QueuePressure を RuntimeHealthMonitor へ通知する。
    RetireEnqueueResult enqueueWithRetry(void* ptr,
                                          void (*deleter)(void*),
                                          uint64_t epoch,
                                          DeletionEntryType type) noexcept;

    // ★ R-1: IRetireRouter インターフェース実装
    //   retireRT: 単発 enqueue、リトライなし（RT-safe）。bool 戻り値で成否を伝える。
    [[nodiscard]] bool retireRT(void* ptr, void (*deleter)(void*)) noexcept override;
    //   retire: リトライ込み（NonRT）。QueuePressure 通知は Router 内部で完結。
    void retire(void* ptr, void (*deleter)(void*)) noexcept override;

    void tryReclaim() noexcept override;
    uint32_t pendingRetireCount() const noexcept override;
    void drainAll() noexcept override;

    // ★ BUG-015/027 (work88): Router API — 退避ストアへの移送（directDelete しない）。
    //   Retire authority は 1 個のまま（本 Router 配下に Queue と QuarantineStore を単一配置）。
    //   SnapshotCoordinator / DSPLifetimeManager はこの API 経由でのみ退避ストアに移送し、
    //   ストアを直接保持しない（五次レビュー §5 — Authority Singularization）。
    //   戻り値 false = store full（呼出し元は deleter を実行してはならない。
    //   health escalation で容量枯渇を先行検知する）。
    bool quarantineRetire(void* ptr, void (*deleter)(void*), uint64_t epoch,
                          DeletionEntryType type, const char* reason,
                          uint64_t publicationSequenceId = 0, uint64_t generation = 0) noexcept;

    // ★ BUG-015/027: 退避ストア滞留件数（backpressure テレメトリ / high watermark 監視用）
    [[nodiscard]] std::size_t quarantineResidentCount() const noexcept;
    // ★ BUG-015/027: store full で quarantine が拒否された回数（EBR 破綻診断用）
    [[nodiscard]] std::uint64_t quarantineOverflowCount() const noexcept;
    // ★ T1 (D86): type==World の terminal deleter 実行数（world 物理破棄数・primary + quarantine 合算）。
    //   release observation（案 B）の一次情報源。sampler（Non-RT）が読み取る（D83.2 責務分離）。
    [[nodiscard]] std::uint64_t worldReclaimCount() const noexcept;
    // ★ BUG-015/027: shutdown 時 — Audio Thread 停止後のみ全強制解放
    void drainAllQuarantineStore() noexcept;

    // ★ Practical-3: Overflow レート監視用カウンター
    [[nodiscard]] uint64_t overflowCount() const noexcept {
        return convo::consumeAtomic(m_overflowCount_, std::memory_order_acquire);
    }
    [[nodiscard]] const std::atomic<uint64_t>* getOverflowCountRef() const noexcept {
        return &m_overflowCount_;
    }

    // ★ Practical-4: Forced reclaim 時刻（enqueueRetire QueuePressure 時の即時 tryReclaim 用）
    void setLastForcedReclaimTimeUs(uint64_t t) noexcept {
        convo::publishAtomic(m_lastForcedReclaimTimeUs_, t, std::memory_order_release);
    }
    [[nodiscard]] uint64_t lastForcedReclaimTimeUs() const noexcept {
        return convo::consumeAtomic(m_lastForcedReclaimTimeUs_, std::memory_order_acquire);
    }

    // ★ Phase 3: Reader Quarantine 委譲（IEpochProvider → EpochDomain）
    [[nodiscard]] bool quarantineReader(int readerIndex) noexcept override
    {
        return provider_->quarantineReader(readerIndex);
    }

    void unquarantineAllReaders() noexcept override
    {
        provider_->unquarantineAllReaders();
    }

    [[nodiscard]] int quarantinedReaderCount() const noexcept override
    {
        return provider_->quarantinedReaderCount();
    }

    // ★ work70: 退役キュー滞留バイト数（診断用概算）
    //   Diagnostic estimate only.
    //   Returns the sum of object sizes for which a non-zero objectBytes
    //   was provided at enqueue time.
    //   Does NOT include allocator overhead (malloc bookkeeping, alignment padding).
    //   Does NOT represent process heap usage of the retire queue.
    [[nodiscard]] uint64_t pendingRetireBytes() const noexcept override
    {
        return convo::consumeAtomic(m_pendingRetireBytes_, std::memory_order_acquire);
    }

    /// trackedPendingEntries と pendingRetireCount の比率（0.0〜1.0）。
    /// objectBytes > 0 のエントリの割合。
    [[nodiscard]] double trackedRatio() const noexcept
    {
        const uint32_t tracked = convo::consumeAtomic(
            m_trackedPendingEntries_, std::memory_order_acquire);
        const uint32_t total = pendingRetireCount();
        if (total == 0) return 0.0;
        const uint32_t clamped = std::min(tracked, total);
        return static_cast<double>(clamped) / static_cast<double>(total);
    }

    /// trackedPendingEntries の raw 値。objectBytes > 0 のエントリ数。
    [[nodiscard]] uint32_t trackedPendingEntries() const noexcept
    {
        return convo::consumeAtomic(m_trackedPendingEntries_, std::memory_order_acquire);
    }

    // ★ P-4: EmergencyQuarantineStore API — D+Q full 時の第3退避層
    //   戻り値 false = store full（TerminalReclaimAuthority への移送を示す）
    bool emergencyQuarantine(void* ptr, void (*deleter)(void*), uint64_t epoch,
                             DeletionEntryType type, const char* reason,
                             uint64_t publicationSequenceId = 0, uint64_t generation = 0) noexcept;

    // ★ P-4: EmergencyQuarantineStore 滞留件数
    [[nodiscard]] std::size_t emergencyQuarantineResidentCount() const noexcept;

    // ★ P-4: TerminalReclaimAuthority API — 最終退避層
    //   epoch safe かつ Non-RT なら場面で deleter 実行、unsafe または RT なら保留。
    //   ★ P-4: 戻り値は常に true（growable store により ownership は必ず移転）。
    bool terminalReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                         DeletionEntryType type, const char* reason) noexcept;

    // ★ P-4: TerminalReclaimAuthority 滞留件数
    [[nodiscard]] std::size_t terminalReclaimResidentCount() const noexcept;

    // ★ E-1.9-A: Q + EmergencyQ + TerminalReclaimAuthority のロックフリー滞留合計
    //   empty-drain suppression 用の atomic カウンタ。RT パスから安全に呼び出し可能。
    [[nodiscard]] uint32_t residentCountAtomic() const noexcept
    {
        return m_retireQuarantine.residentCountAtomic()
             + m_emergencyQuarantine.residentCountAtomic()
             + m_terminalReclaim.residentCountAtomic();
    }

    // ★ E-1.9-B: Event-driven drain wake primitive.
    //   CoordinatorLoop (Non-RT) blocks on drainCv_ with predicate:
    //     pendingRetireCount() != 0 || residentCountAtomic() != 0
    //   Producers (enqueueWithRetry, Non-RT only) call signalDrainWakeup()
    //   after placing an entry in Q/E/T. The predicate reads the E-1.9-A atomic
    //   counters — no separate drainSignaled_ state is introduced (Semantic
    //   Single Source: resident count is the sole authority for "has pending").
    //   ★ B-R3: signalDrainWakeup() acquires drainCvMtx_ before notify_one to
    //   participate in the CV synchronization protocol (prevents lost-wake).
    //   Non-RT only: no RT thread ever touches drainCv_ or drainCvMtx_.
    void signalDrainWakeup() noexcept;

    // ★ E-1.9-B: CoordinatorLoop calls this to block until drain predicate is
    //   true or timeoutMs elapses. Preserves the 1ms polling fallback.
    //   Non-RT only (called from CoordinatorLoop::run).
    void waitForDrainSignalOrTimeout(int timeoutMs) noexcept;

    // ★ P-4: TerminalReclaimAuthority の drain（epoch-gated, 定期呼び出し）
    void drainTerminalReclaim() noexcept;

    // ★ P-4: ShutdownReclaimAuthority — shutdown 専用の ownership transfer
    //   shutdown 中に enqueueDeferredDeleteNonRtWithResult が early-return する際、
    //   ptr を捨てずに TerminalReclaimAuthority へ移送する（epoch-gated destruction）。
    //   shutdown 中は Audio Thread 停止済みのため epoch は安全 → 即時破棄される。
    //   戻り値 true = ownership transfer 成立（caller は ptr を触ってはならない）。
    //   ★ P-4: 戻り値 false は存在しない（growable store により必ず移転）。
    bool shutdownReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                         DeletionEntryType type) noexcept;

private:
    // ★ BUG-015/027: tryReclaim 直後に退避ストアを drain（epoch 安全到達分のみ deleter 実行）
    void drainQuarantineStore() noexcept;
    // ★ P-4: EmergencyQ + TerminalReclaimAuthority の epoch-gated drain
    void drainEmergencyAndTerminal() noexcept;

    // ★ P-4: EpochDomain::isOlder の static 版（TerminalReclaim 内から呼び出し）
    //   isOlder(a, b) = static_cast<int64_t>(a - b) < 0 (wraparound-safe)
    static bool isOlder(uint64_t a, uint64_t b) noexcept;

    convo::IEpochProvider* provider_ = nullptr;
    convo::isr::WorldRetirementReferenceObserver* referenceObserver_ = nullptr;   // ★ T1 (D98): non-owning（measurement only・D97）
    std::atomic<uint64_t> m_overflowCount_{0};
    std::atomic<uint64_t> m_lastForcedReclaimTimeUs_{0};

    // ★ BUG-015/027: retire enqueue 失敗時の退避ストア（Router Policy lane 配下に単一配置）
    RetireQuarantineStore m_retireQuarantine;
    // ★ P-4: EmergencyQuarantineStore — D+Q full 時の第3退避層（同一タイプ・別インスタンス）
    RetireQuarantineStore m_emergencyQuarantine;
    // ★ P-4: TerminalReclaimAuthority — D+Q+E 全滿時の最終退避層（epoch-gated synchronous destruction）
    TerminalReclaimAuthority m_terminalReclaim;
    // ★ work70: 診断用カウンタ
    std::atomic<uint64_t> m_pendingRetireBytes_{0};
    std::atomic<uint32_t> m_trackedPendingEntries_{0};

    // ★ E-1.9-B: drain wake primitive (Non-RT only)
    //   drainCv_ / drainCvMtx_ are used by CoordinatorLoop to block on the
    //   E-1.9-A atomic predicates (pendingRetireCount / residentCountAtomic).
    //   signalDrainWakeup() is called from enqueueWithRetry (Non-RT producers of
    //   Q/E/T entries). No RT thread ever touches these — verified by B-R2-2.
    std::condition_variable drainCv_;
    std::mutex drainCvMtx_;

    // ★ B-R3/R5: Test-only access for the lost-wake regression test.
    //   The test class accesses drainCv_ / drainCvMtx_ through this friend to
    //   deterministically force the "consumer holds lock, producer notifies"
    //   interleaving. The primitives are NOT exposed as public API — production
    //   code must go through signalDrainWakeup() / waitForDrainSignalOrTimeout().
    friend class RetireGraceSemanticsTestAccess;
};

} // namespace isr
} // namespace convo
