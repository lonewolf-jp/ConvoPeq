//============================================================================
// ISRRetireRouter.cpp — EpochDomain ラッパーの実装
//
// ISR P1-19: EpochDomain の完全型はこの .cpp に閉じ込め、
// .h では前方宣言のみとする。これにより公開APIへの EpochDomain 露出を排除する。
//============================================================================

#include "ISRRetireRouter.h"
#include "core/TimeUtils.h"     // ★ Practical-4: getCurrentTimeUs
#include "../DspNumericPolicy.h" // ★ P-4: isAudioThread() — RT 防御（synchronous destruction 禁止）

namespace convo {
namespace isr {

//============================================================================
// TerminalReclaimAuthority — implementation
//============================================================================
// Final ownership authority when all bounded stores (D+Q+E) are exhausted.
// Ownership contract: ptr transferred here ⟹ caller retains NO ownership.
// If epoch-safe: deleter executes immediately (synchronous, Non-RT).
// If epoch-unsafe: entry retained; drain() reclaims when epoch becomes safe.
//
// ★ P-4 (15-P-4): entries_ is GROWABLE (std::vector). store() ALWAYS succeeds,
//   so there is NO "store full" failure path. This guarantees the ownership
//   invariant: enqueueWithRetry() never returns with ptr unowned.

bool TerminalReclaimAuthority::store(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                     DeletionEntryType type, const char* reason) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return true;  // no-op は成功扱い

    std::lock_guard<std::mutex> lock(mtx_);
    entries_.push_back(Entry{ptr, deleter, epoch, type, reason});
    residentAtomic_.fetch_add(1, std::memory_order_release);
    return true;  // ★ P-4: growable store — ALWAYS accepts
}

void TerminalReclaimAuthority::drain(uint64_t minReaderEpoch,
                                     const std::function<bool(uint64_t, uint64_t)>& isOlderFn) noexcept
{
    // Extract epoch-safe entries under lock, execute deleter outside lock (reentrancy-safe)
    // ★ P-4: EBR 安全条件は RetireQuarantineStore::drain と同一 —
    //   isOlder(entry.epoch, minReaderEpoch) == true（entry.epoch < minReaderEpoch）→ safe。
    std::vector<Entry> pending;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        std::size_t w = 0;
        for (std::size_t r = 0; r < entries_.size(); ++r) {
            auto& e = entries_[r];
            if (e.ptr != nullptr && e.deleter != nullptr
                && isOlderFn(e.epoch, minReaderEpoch))  // epoch < minReaderEpoch → safe
            {
                pending.push_back(e);
                e = Entry{};
            } else {
                if (w != r)
                    entries_[w] = e;
                ++w;
            }
        }
        entries_.resize(w);
    }
    // ★ E-1.9-A: 解放されたエントリ数だけロックフリーカウンタを decrement
    residentAtomic_.fetch_sub(static_cast<uint32_t>(pending.size()), std::memory_order_release);
    for (auto& e : pending) {
        e.deleter(e.ptr);
        if (e.type == DeletionEntryType::World)
        {
            ++reclaimCount_;
            if (referenceObserver_ != nullptr)
                referenceObserver_->onRelease();
        }
    }
}

void TerminalReclaimAuthority::drainAll() noexcept
{
    std::vector<Entry> pending;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        pending.swap(entries_);  // take all entries under lock
        // ★ E-1.9-A: ロックフリーカウンタをリセット（shutdown drain）
        residentAtomic_.store(0, std::memory_order_release);
    }
    for (auto& e : pending) {
        if (e.ptr != nullptr && e.deleter != nullptr) {
            e.deleter(e.ptr);
            if (e.type == DeletionEntryType::World)
            {
                ++reclaimCount_;
                if (referenceObserver_ != nullptr)
                    referenceObserver_->onRelease();
            }
        }
    }
}

bool TerminalReclaimAuthority::tryReclaim(uint64_t minReaderEpoch,
                                        const std::function<bool(uint64_t, uint64_t)>& isOlderFn) noexcept
{
    // Drain epoch-safe entries; return true if at least one was reclaimed
    std::size_t beforeCount = 0;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        beforeCount = entries_.size();
    }
    if (beforeCount == 0)
        return false;

    drain(minReaderEpoch, isOlderFn);

    std::size_t afterCount = 0;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        afterCount = entries_.size();
    }
    return (beforeCount > afterCount);
}

std::size_t TerminalReclaimAuthority::residentCount() const noexcept
{
    std::lock_guard<std::mutex> lock(mtx_);
    return entries_.size();
}

ISRRetireRouter::ISRRetireRouter(IEpochProvider& provider,
                                 convo::isr::WorldRetirementReferenceObserver* referenceObserver) noexcept
    : provider_(&provider)
    , referenceObserver_(referenceObserver)
{
    // ★ T1 (D98): reference observer を storage（RetireQuarantineStore・EpochDomain）へ伝搬（non-owning・一方向依存）。
    m_retireQuarantine.setReferenceObserver(referenceObserver);
    // ★ P-4: EmergencyQ + TerminalReclaimAuthority にも observer を伝搬
    m_emergencyQuarantine.setReferenceObserver(referenceObserver);
    m_terminalReclaim.setReferenceObserver(referenceObserver);
    provider_->setReferenceObserver(referenceObserver);
}

uint64_t ISRRetireRouter::snapshotEpoch() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->currentEpoch();
}

uint64_t ISRRetireRouter::publishEpoch() noexcept
{
    assert(provider_ != nullptr);
    return provider_->publishEpoch();
}

uint32_t ISRRetireRouter::activeReaderCount() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->activeReaderCount();
}

uint64_t ISRRetireRouter::currentEpoch() const noexcept
{
    return snapshotEpoch();
}

uint64_t ISRRetireRouter::getMinReaderEpoch() const noexcept
{
    return minReaderEpoch();
}

int ISRRetireRouter::registerReaderThread() noexcept
{
    assert(provider_ != nullptr);
    return provider_->registerReaderThread();
}

bool ISRRetireRouter::reserveReaderThread(int readerIndex) noexcept
{
    assert(provider_ != nullptr);
    return provider_->reserveReaderThread(readerIndex);
}

void ISRRetireRouter::enterReader(int readerIndex) noexcept
{
    assert(provider_ != nullptr);
    provider_->enterReader(readerIndex);
}

void ISRRetireRouter::exitReader(int readerIndex) noexcept
{
    assert(provider_ != nullptr);
    provider_->exitReader(readerIndex);
}

convo::ReaderSlotDetail ISRRetireRouter::getReaderSlotDetail(int readerIndex) const noexcept
{
    assert(provider_ != nullptr);
    return provider_->getReaderSlotDetail(readerIndex);
}

uint64_t ISRRetireRouter::minReaderEpoch() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->getMinReaderEpoch();
}

int ISRRetireRouter::readerCapacity() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->readerCapacity();
}

StuckReaderInfo ISRRetireRouter::detectStuckReaders(uint64_t stuckThreshold) const noexcept
{
    // ★ Practical-1: IEpochProvider の virtual detectStuckReaders 経由で委譲
    //   dynamic_cast 不要。ISR P1-19 / P0-A 完全準拠。
    assert(provider_ != nullptr);
    return provider_->detectStuckReaders(stuckThreshold);
}

RetireEnqueueResult ISRRetireRouter::enqueueRetire(void* ptr,
                                                    void (*deleter)(void*),
                                                    uint64_t epoch,
                                                    DeletionEntryType type) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return RetireEnqueueResult::Success;

    // Route through IEpochProvider interface（★ T1: telemetry type tag を伝搬・D86）。
    if (provider_->enqueueRetireTyped(ptr, deleter, epoch, type))
    {
        // ★ work70: サイズ追跡は enqueue 時に objectBytes が設定されている場合のみ。
        //   現在の呼び出し元は objectBytes=0 のため trackedRatio=0% となる。
        //   将来、特定の呼び出し元でサイズ設定する場合に対応。
        return RetireEnqueueResult::Success;
    }

    // ★ Practical-4: QueueFull → 同期的 tryReclaim を１度だけ試行（レート制限付き）
    const uint64_t nowUs = convo::getCurrentTimeUs();
    const uint64_t lastReclaim = convo::consumeAtomic(m_lastForcedReclaimTimeUs_, std::memory_order_acquire);
    if (nowUs - lastReclaim > 500'000) // 500ms cooldown
    {
        convo::publishAtomic(m_lastForcedReclaimTimeUs_, nowUs, std::memory_order_release);
        provider_->tryReclaim();

        // 再試行: reclaim 後に空きができたか確認（★ T1: type を伝搬・D86）
        if (provider_->enqueueRetireTyped(ptr, deleter, epoch, type))
            return RetireEnqueueResult::Success;
    }

    // ★ Practical-3: Overflow カウンター増加（Rate監視用）
    convo::fetchAddAtomic(m_overflowCount_, uint64_t{1}, std::memory_order_release);
    return RetireEnqueueResult::QueuePressure;
}

bool ISRRetireRouter::enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept
{
    return enqueueRetire(ptr, deleter, epoch, DeletionEntryType::Generic)
        == RetireEnqueueResult::Success;
}

// ★ R-1: IRetireRouter::retireRT — 単発 enqueue、リトライなし（RT-safe）
bool ISRRetireRouter::retireRT(void* ptr, void (*deleter)(void*)) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return true;
    return provider_->enqueueRetire(ptr, deleter, provider_->currentEpoch());
}

// ★ R-1: IRetireRouter::retire — リトライ込み（NonRT）
void ISRRetireRouter::retire(void* ptr, void (*deleter)(void*)) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return;
    const auto result = enqueueWithRetry(ptr, deleter, provider_->currentEpoch(), DeletionEntryType::Generic);
    if (result != RetireEnqueueResult::Success) {
        // ★ Future: RuntimeHealthMonitor へ通知
    }
}

// ★ Bug2 Phase1: リトライロジックを Router に集約。呼び出し元はリトライループ不要。
RetireEnqueueResult ISRRetireRouter::enqueueWithRetry(void* ptr,
                                                        void (*deleter)(void*),
                                                        uint64_t epoch,
                                                        DeletionEntryType type) noexcept
{
    // ★ B-I3: RT boundary — enqueueWithRetry can reach Q/E/T (mutex + allocation),
    //   therefore it MUST only be called from Non-RT context. All RT callers use
    //   retireRT() → enqueueRetire() (D queue only, lock-free).
    //   NOTE: This assert is a guard rail, not a proof of RT safety — production
    //   caller enumeration (B-R2-2) is the authoritative verification.
    jassert(!convo::numeric_policy::isAudioThread());

    // ★ P-4: Ownership chain: D → Q → EmergencyQ → TerminalReclaimAuthority
    //   Ownership invariant: ptr を手放す前に、必ず次の authority に ownership が移る。
    //   assert(false) → return という経路は残さない（Release で L > 0 が発生する）。
    //   ★ P-4: TerminalReclaimAuthority は growable store のため常に ownership を受領する。
    //     したがって「全 store full / epoch unsafe」でも ptr が宙に浮くことはない。

    // --- Stage 1: DeferredDeletionQueue (D) ---
    auto result = enqueueRetire(ptr, deleter, epoch, type);
    if (result == RetireEnqueueResult::Success)
        return result;  // D owns ptr ✅

    // --- Stage 2: Retry cycle (tryReclaim → re-enqueue) ---
    constexpr int kMaxRetry = 2;
    for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
        provider_->tryReclaim();
        drainEmergencyAndTerminal();  // drain E + Terminal (epoch-gated)
        result = enqueueRetire(ptr, deleter, epoch, type);
        if (result == RetireEnqueueResult::Success)
            return result;  // D owns ptr ✅
        if (result != RetireEnqueueResult::QueuePressure)
            break;  // Shutdown etc. — exit loop
    }

    // --- Stage 3: RetireQuarantineStore (Q) ---
    if (result == RetireEnqueueResult::QueuePressure || result == RetireEnqueueResult::QueueFull)
    {
        // ★ P-4: D full + retry exhausted → Q へ移送
        const bool stored = m_retireQuarantine.quarantine(
            ptr, deleter, epoch, type, "enqueueWithRetry:QueuePressure",
            /*publicationSequenceId=*/0, /*generation=*/0);
        if (stored)
            result = RetireEnqueueResult::QueuePressure;  // Q owns ptr ✅
        else
        {
            // ★ P-4: Q full → Stage 4: EmergencyQuarantineStore (E)
            //   ★ drainAllUnsafe は呼ばない（Audio Thread 稼働中の UAF リスク + 無意味：
            //     ptr はまだ Q に無いため Q を空にしても空きは増えない）。
            const bool estored = m_emergencyQuarantine.quarantine(
                ptr, deleter, epoch, type, "enqueueWithRetry:EmergencyQuarantine",
                /*publicationSequenceId=*/0, /*generation=*/0);
            if (estored)
                result = RetireEnqueueResult::QueuePressure;  // E owns ptr ✅
            else
            {
                // ★ P-4: E full → Stage 5: TerminalReclaimAuthority
                //   D+Q+E 全滿 → TerminalReclaimAuthority へ移送
                //   epoch safe かつ Non-RT なら即座に deleter 実行（synchronous destruction）
                //   epoch unsafe なら保持（drain() が epoch safe になった時に解放）
                //   ★ P-4: growable store により常に true → ownership は必ず移転。
                //     assert(false) 経路は存在しない（EBR 破綻による L > 0 は構造的に排除）。
                const bool tstored = terminalReclaim(ptr, deleter, epoch, type,
                                                     "enqueueWithRetry:TerminalReclaim");
                (void)tstored;  // ★ P-4: 常に true（growable store）
                result = RetireEnqueueResult::TerminalReclaim;  // Terminal owns ptr ✅
            }
        }

        // ★ E-1.9-B-R2-1: Single signal point — Q/E/T received an entry.
        //   The entry is now resident in Q, E, or T (residentAtomic_ has been
        //   incremented under the store mutex). Signal the CoordinatorLoop
        //   to wake from CV wait. signalDrainWakeup() acquires drainCvMtx_
        //   before notify_one (B-R3 fix) — this serializes the notify with the
        //   consumer's wait transition, eliminating the lost-wake window.
        signalDrainWakeup();
        return result;
    }

    // ★ P-4: Shutdown — caller retains ownership (enqueueDeferredDeleteNonRtWithResult handles)
    return result;
}

void ISRRetireRouter::tryReclaim() noexcept
{
    assert(provider_ != nullptr);
    provider_->tryReclaim();
    // ★ BUG-015/027 (work88): tryReclaim 直後に退避ストアを drain（epoch 安全到達分のみ deleter 実行）
    drainQuarantineStore();
    // ★ P-4: EmergencyQ + TerminalReclaimAuthority の epoch-gated drain
    drainEmergencyAndTerminal();
}

// ★ BUG-015/027 (work88): 退避ストアを drain。
//   epoch 比較は EpochDomain::isOlder と同一セマンティクス（wraparound 安全）を
//   インライン実装（ISR P1-19: EpochDomain 完全型を .h に露出しない）。
void ISRRetireRouter::drainQuarantineStore() noexcept
{
    const uint64_t minReader = minReaderEpoch();
    m_retireQuarantine.drain(minReader, [](uint64_t a, uint64_t b) noexcept {
        return static_cast<int64_t>(a - b) < 0;  // == EpochDomain::isOlder(a, b)
    });
}

// ★ BUG-015/027 (work88): Router API — 退避ストアへの移送（directDelete しない）。
//   store full 時は deleter を実行せず false を返す（UAF 構造的排除）。
//   capacity exhaustion は health escalation（AudioEngine 側が quarantineResidentCount /
//   quarantineOverflowCount を監視）で先行検知する。
bool ISRRetireRouter::quarantineRetire(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                       DeletionEntryType type, const char* reason,
                                       uint64_t publicationSequenceId, uint64_t generation) noexcept
{
    return m_retireQuarantine.quarantine(ptr, deleter, epoch, type, reason,
                                         publicationSequenceId, generation);
}

// ★ T1 (D86): type==World の terminal deleter 実行数（world 物理破棄数・primary + quarantine 合算）。
//   release observation（案 B）の一次情報源。sampler（Non-RT）が読み取る（D83.2 責務分離）。
uint64_t ISRRetireRouter::worldReclaimCount() const noexcept
{
    assert(provider_ != nullptr);
    // ★ P-4: EmergencyQ + TerminalReclaimAuthority の world 破棄数も合算
    return provider_->worldReclaimCount()
        + m_retireQuarantine.worldReclaimCount()
        + m_emergencyQuarantine.worldReclaimCount()
        + m_terminalReclaim.reclaimCount();
}

std::size_t ISRRetireRouter::quarantineResidentCount() const noexcept
{
    // ★ P-4: Q + EmergencyQ の合算（backpressure テレメトリ）
    return m_retireQuarantine.residentCount() + m_emergencyQuarantine.residentCount();
}

std::uint64_t ISRRetireRouter::quarantineOverflowCount() const noexcept
{
    // ★ P-4: Q + EmergencyQ の合算（EBR 破綻診断）
    return m_retireQuarantine.overflowCount() + m_emergencyQuarantine.overflowCount();
}

// ★ BUG-015/027: shutdown 時 — Audio Thread 停止後のみ全強制解放（drainAllUnsafe と同契約）
void ISRRetireRouter::drainAllQuarantineStore() noexcept
{
    m_retireQuarantine.drainAllUnsafe();
    // ★ P-4: EmergencyQ も全強制解放（Audio Thread 停止後）
    m_emergencyQuarantine.drainAllUnsafe();
    // ★ P-4: TerminalReclaimAuthority も全強制解放（Audio Thread 停止後）
    m_terminalReclaim.drainAll();
}

// ★ E-1.9-B: Signal the CoordinatorLoop that Q/E/T may have new entries.
//   Non-RT only: all Q/E/T producers are Non-RT (verified B-R2-2 / R3-5).
//
//   ★ B-R3 FIX: notify_one MUST be issued while holding drainCvMtx_.
//   Without the mutex, the following interleaving loses the wake:
//     Consumer: lock(drainCvMtx_), predicate check → false
//     Producer: residentAtomic_++ (predicate true), notify_one() → NO waiter yet → LOST
//     Consumer: wait_for → unlock + block → sleeps until timeout (1ms latency regression)
//   Acquiring drainCvMtx_ in the signal path serializes the notify with the
//   consumer's wait transition:
//     Case 1 (producer first): producer locks, notifies (no waiter), unlocks;
//       consumer locks, predicate check → TRUE → skips wait entirely.
//     Case 2 (consumer first): consumer locks, predicate false, enters wait
//       (atomically releases lock); producer acquires lock, notifies → wakes
//       consumer; consumer rechecks predicate → TRUE → proceeds immediately.
//   The resident counter itself stays atomic (NOT protected by drainCvMtx_) —
//   only the notify participates in the CV synchronization protocol.
void ISRRetireRouter::signalDrainWakeup() noexcept
{
    std::lock_guard<std::mutex> lock(drainCvMtx_);
    drainCv_.notify_one();
}

// ★ E-1.9-B: Wait for drain predicate to become true or timeout.
//   Predicate: pendingRetireCount() != 0 || residentCountAtomic() != 0
//   These are the SAME E-1.9-A atomic counters used by the empty-drain
//   suppression gate — Semantic Single Source (no drainSignaled_ state).
//   timeoutMs fallback preserves the 1ms polling cadence of CoordinatorLoop.
//   Non-RT only (called from CoordinatorLoop::run).
void ISRRetireRouter::waitForDrainSignalOrTimeout(int timeoutMs) noexcept
{
    std::unique_lock<std::mutex> lock(drainCvMtx_);
    drainCv_.wait_for(lock, std::chrono::milliseconds(timeoutMs < 0 ? 0 : timeoutMs),
        [&] {
            return pendingRetireCount() != 0
                || residentCountAtomic() != 0;
        });
    // Predicate true (drain needed) or timeout expired — caller proceeds to runCoordinatorPhase.
}

// ★ P-4: EmergencyQuarantineStore API — D+Q full 時の第3退避層
bool ISRRetireRouter::emergencyQuarantine(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                          DeletionEntryType type, const char* reason,
                                          uint64_t publicationSequenceId, uint64_t generation) noexcept
{
    return m_emergencyQuarantine.quarantine(ptr, deleter, epoch, type, reason,
                                            publicationSequenceId, generation);
}

// ★ P-4: EmergencyQuarantineStore 滞留件数
std::size_t ISRRetireRouter::emergencyQuarantineResidentCount() const noexcept
{
    return m_emergencyQuarantine.residentCount();
}

// ★ P-4: TerminalReclaimAuthority API — 最終退避層
bool ISRRetireRouter::terminalReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                      DeletionEntryType type, const char* reason) noexcept
{
    // ★ P-4: EBR 安全条件は DeferredDeletionQueue::reclaim / RetireQuarantineStore::drain と同一 —
    //   isOlder(epoch, minReaderEpoch) == true（epoch < minReaderEpoch）→ 全 Reader が epoch を
    //   通過済み → synchronous destruction 安全。
    // epoch safe かつ Non-RT なら即座に deleter 実行（synchronous destruction）
    // epoch unsafe または RT スレッドなら保持（drain() が epoch safe になった時に解放）
    const uint64_t minReader = minReaderEpoch();
    const bool epochSafe = ISRRetireRouter::isOlder(epoch, minReader);  // epoch < minReaderEpoch → safe
    const bool isRt = convo::numeric_policy::isAudioThread();  // ★ P-4: RT 防御

    if (epochSafe && !isRt)
    {
        // Synchronous destruction — Non-RT context guaranteed by caller
        deleter(ptr);
        if (type == DeletionEntryType::World)
            m_terminalReclaim.recordWorldReclaim();
        return true;  // destroyed immediately, no storage needed
    }

    // epoch unsafe OR RT caller → store for later drain
    // ★ P-4: growable store — ALWAYS accepts (ownership always transfers)
    return m_terminalReclaim.store(ptr, deleter, epoch, type, reason);
}

// ★ P-4: TerminalReclaimAuthority 滞留件数
std::size_t ISRRetireRouter::terminalReclaimResidentCount() const noexcept
{
    return m_terminalReclaim.residentCount();
}

// ★ P-4: TerminalReclaimAuthority の epoch-gated drain
void ISRRetireRouter::drainTerminalReclaim() noexcept
{
    const uint64_t minReader = minReaderEpoch();
    m_terminalReclaim.drain(minReader, [](uint64_t a, uint64_t b) noexcept {
        return static_cast<int64_t>(a - b) < 0;  // == EpochDomain::isOlder(a, b)
    });
}

// ★ P-4: EmergencyQ + TerminalReclaimAuthority の epoch-gated drain
void ISRRetireRouter::drainEmergencyAndTerminal() noexcept
{
    // drain EmergencyQuarantineStore (epoch-gated)
    {
        const uint64_t minReader = minReaderEpoch();
        m_emergencyQuarantine.drain(minReader, [](uint64_t a, uint64_t b) noexcept {
            return static_cast<int64_t>(a - b) < 0;  // == EpochDomain::isOlder(a, b)
        });
    }
    // drain TerminalReclaimAuthority (epoch-gated)
    drainTerminalReclaim();
}

// ★ P-4: isOlder の static 版（TerminalReclaim 内から呼び出し）
bool ISRRetireRouter::isOlder(uint64_t a, uint64_t b) noexcept
{
    return static_cast<int64_t>(a - b) < 0;
}

// ★ P-4: ShutdownReclaimAuthority — shutdown 専用の ownership transfer
//   shutdown 中に enqueueDeferredDeleteNonRtWithResult が early-return する際、
//   ptr を捨てずに TerminalReclaimAuthority へ移送する（epoch-gated destruction）。
//   shutdown 中は Audio Thread 停止済みのため epoch は安全 → 即時破棄される。
bool ISRRetireRouter::shutdownReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                      DeletionEntryType type) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return true;  // no-op は成功扱い

    // TerminalReclaimAuthority へ移送（epoch-gated destruction）
    //   epoch safe → 即時破棄（shutdown 中は Audio Thread 停止済みのため通常こちら）
    //   epoch unsafe → 保持（drainAll() が shutdown 時に全強制解放）
    return terminalReclaim(ptr, deleter, epoch, type, "shutdownReclaim");
}

uint32_t ISRRetireRouter::pendingRetireCount() const noexcept
{
    // ★ P0-A: IRetireProvider 経由で委譲（dynamic_cast 不要）
    assert(provider_ != nullptr);
    return provider_->pendingRetireCount();
}

void ISRRetireRouter::drainAll() noexcept
{
    // ★ P0-A: IRetireProvider 経由で委譲（dynamic_cast 不要）
    assert(provider_ != nullptr);
    provider_->drainAll();
    // ★ BUG-015/027 + P-4: shutdown 時は退避ストアも全強制解放（Audio Thread 停止後）
    //   drainAllQuarantineStore() が Q + EmergencyQ + TerminalReclaimAuthority を全て解放する。
    drainAllQuarantineStore();
}

// ★ A-2: EBR Queue Visibility 統計委譲
uint64_t ISRRetireRouter::reclaimAttemptCount() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->reclaimAttemptCount();
}

uint64_t ISRRetireRouter::reclaimSuccessCount() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->reclaimSuccessCount();
}

} // namespace isr
} // namespace convo
