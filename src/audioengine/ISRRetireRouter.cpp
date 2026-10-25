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
    entries_.push_back(Entry{ptr, deleter, nullptr, epoch, type, reason, 0});
    // ★ Step 5-I / AtomicAccess convention: use convo:: wrapper (not raw fetch_add)
    convo::fetchAddAtomic(residentAtomic_, uint32_t{1}, std::memory_order_release);
    // ★ Step 5-I: telemetry — cumulative store count (Generic + World)
    convo::fetchAddAtomic(terminalStoreCount_, std::uint64_t{1}, std::memory_order_release);
    // ★ Step 5-I: telemetry — peak resident update (CAS loop, uses residentAtomic_ as authoritative source)
    {
        const uint32_t current = convo::consumeAtomic(residentAtomic_, std::memory_order_acquire);
        uint32_t expected = convo::consumeAtomic(terminalPeakResident_, std::memory_order_acquire);
        while (current > expected) {
            if (convo::compareExchangeAtomic(terminalPeakResident_, expected, current,
                    std::memory_order_acq_rel, std::memory_order_acquire))
                break;
        }
    }
    return true;  // ★ P-4: growable store → ALWAYS accepts
}

// F1'-b (2026-10-25): retryable 破棄の格納。所有権は常に移転する。
bool TerminalReclaimAuthority::storeRetryable(void* ptr, RetryableDeleter retryable,
                                              uint64_t epoch, DeletionEntryType type,
                                              const char* reason) noexcept
{
    if (ptr == nullptr || retryable == nullptr)
        return true;

    std::lock_guard<std::mutex> lock(mtx_);
    entries_.push_back(Entry{ptr, nullptr, retryable, epoch, type, reason, 0});
    convo::fetchAddAtomic(residentAtomic_, uint32_t{1}, std::memory_order_release);
    convo::fetchAddAtomic(terminalStoreCount_, std::uint64_t{1}, std::memory_order_release);
    {
        const uint32_t current = convo::consumeAtomic(residentAtomic_, std::memory_order_acquire);
        uint32_t expected = convo::consumeAtomic(terminalPeakResident_, std::memory_order_acquire);
        while (current > expected) {
            if (convo::compareExchangeAtomic(terminalPeakResident_, expected, current,
                    std::memory_order_acq_rel, std::memory_order_acquire))
                break;
        }
    }
    return true;
}

// F1'-b: 延期されたまま残っているエントリ数（shutdown 後の未回収観測用）。
//   注意: 「retryable で格納されたが、まだ 1 度も延期されていない」エントリは
//   未回収ではない（次回 drain で破棄される予定）。ここでは
//   「延期を経験したエントリ」= 破棄が進んでいないものを数える。
std::size_t TerminalReclaimAuthority::deferredResidentCount() const noexcept
{
    std::lock_guard<std::mutex> lock(mtx_);
    std::size_t n = 0;
    for (const auto& e : entries_)
    {
        if (e.ptr != nullptr && e.retryable != nullptr && e.deferredAttempts > 0)
            ++n;
    }
    return n;
}

// F1'-b: retryable として保持されているエントリ総数（延期前も含む）。
//   「所有権を保持しているか」の観測はこちらを使う。
std::size_t TerminalReclaimAuthority::retryableResidentCount() const noexcept
{
    std::lock_guard<std::mutex> lock(mtx_);
    std::size_t n = 0;
    for (const auto& e : entries_)
    {
        if (e.ptr != nullptr && e.retryable != nullptr)
            ++n;
    }
    return n;
}

void TerminalReclaimAuthority::drain(uint64_t minReaderEpoch,
                                     const std::function<bool(uint64_t, uint64_t)>& isOlderFn) noexcept
{
    // ★ F1'-b (2026-10-25): 所有権の扱いを「消費」から「結果を見て消費」へ変更。
    //   旧実装は epoch-safe なエントリを pending に移した時点で entries_ から消していたため、
    //   deleter が「解放しなかった」と判断しても所有権が失われていた
    //   （再試行不能・residentAtomic_ の誤減少・shutdown 終端条件の誤成立）。
    //
    //   新規則:
    //     - retryable を持つエントリ: entries_ に残したまま呼び出す。
    //         Destroyed → 消費（所有権を放棄）
    //         Deferred  → entries_ に残す（所有権を保持し、次回 drain で再試行）
    //     - retryable を持たない従来エントリ: これまでどおり取り出して deleter を実行する。
    std::vector<Entry> legacyPending;
    std::size_t destroyedRetryable = 0;
    std::size_t transferredRetryable = 0;   // E: Deferred -> the callee owns it

    // --- 1) epoch-safe な retryable 候補を集める（ロック内・呼び出しはしない）---
    std::vector<std::size_t> candidates;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        std::size_t w = 0;
        for (std::size_t r = 0; r < entries_.size(); ++r) {
            auto& e = entries_[r];
            const bool hasTarget = (e.ptr != nullptr)
                && (e.deleter != nullptr || e.retryable != nullptr);
            const bool safe = hasTarget && isOlderFn(e.epoch, minReaderEpoch);
            if (!safe) {
                if (w != r) entries_[w] = e;
                ++w;
                continue;
            }
            if (e.retryable != nullptr) {
                if (w != r) entries_[w] = e;   // 残す（結果を見てから決める）
                candidates.push_back(w);
                ++w;
            } else {
                legacyPending.push_back(e);    // 従来経路: 消費
                e = Entry{};
            }
        }
        entries_.resize(w);
    }

    // --- 2) retryable をロック外で実行し、結果で消費/保持を決める ---
    for (std::size_t idx : candidates) {
        Entry snapshot{};
        bool present = false;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (idx < entries_.size() && entries_[idx].ptr != nullptr
                && entries_[idx].retryable != nullptr) {
                snapshot = entries_[idx];
                present = true;
            }
        }
        if (!present)
            continue;

        const DestroyOutcome outcome = snapshot.retryable(snapshot.ptr);

        std::lock_guard<std::mutex> lock(mtx_);
        if (idx < entries_.size() && entries_[idx].ptr == snapshot.ptr) {
            if (outcome == DestroyOutcome::Destroyed) {
                entries_[idx] = Entry{};
                ++destroyedRetryable;
            } else {
                // ★ 所有権を保持: エントリは残し、試行回数だけ増やす。
                // E (2026-10-25): Deferred = the callee retained ownership (the engine
                // retryable registers the pointer in its own retry list). Dropping the
                // entry here is what prevents the same pointer from being owned twice.
                entries_[idx] = Entry{};
                ++transferredRetryable;
                convo::fetchAddAtomic(ownershipTransferredCount_, std::uint64_t{1},
                                      std::memory_order_acq_rel);
                convo::fetchAddAtomic(deferredAttemptCount_, std::uint64_t{1},
                                      std::memory_order_acq_rel);
            }
        }
    }

    // --- 3) 消費したスロットを詰める ---
    if (destroyedRetryable > 0 || transferredRetryable > 0) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::size_t w = 0;
        for (std::size_t r = 0; r < entries_.size(); ++r) {
            if (entries_[r].ptr == nullptr)
                continue;
            if (w != r) entries_[w] = entries_[r];
            ++w;
        }
        entries_.resize(w);
    }

    // --- 4) residentAtomic_ は「未回収のエントリ数」。Deferred は減らさない（RT-3）---
    const std::size_t consumed = legacyPending.size() + destroyedRetryable
        + transferredRetryable;
    if (consumed > 0)
        convo::fetchSubAtomic(residentAtomic_, static_cast<uint32_t>(consumed),
                              std::memory_order_release);

    // --- 5) 従来 deleter の実行（ロック外・reentrancy 安全）---
    for (auto& e : legacyPending) {
        e.deleter(e.ptr);
        if (e.type == DeletionEntryType::World)
        {
            convo::fetchAddAtomic(reclaimCount_, std::uint64_t{1}, std::memory_order_acq_rel);
            if (referenceObserver_ != nullptr)
                referenceObserver_->onRelease();
        }
    }
}

void TerminalReclaimAuthority::drainAll() noexcept
{
    // ★ Step 5-I: telemetry — increment drainAll invocation count
    convo::fetchAddAtomic(terminalDrainAllCount_, std::uint64_t{1}, std::memory_order_acq_rel);

    // ★ F1'-b (2026-10-25): drainAll でも Deferred を強制解放しない（RT-4）。
    //   shutdown 中に pin 保持者がいる可能性を排除できないため、UAF ではなく
    //   「未回収として残す」方を選ぶ。残数は deferredResidentCount() で観測できる。
    std::vector<Entry> pending;      // 従来 deleter（消費してよい）
    std::vector<Entry> kept;         // retryable で Deferred だったもの（所有権保持）
    {
        std::lock_guard<std::mutex> lock(mtx_);
        pending.swap(entries_);      // take all entries under lock
        convo::publishAtomic(residentAtomic_, uint32_t{0}, std::memory_order_release);
    }

    for (auto& e : pending) {
        if (e.ptr == nullptr)
            continue;

        if (e.retryable != nullptr)
        {
            // 結果を見て決める。Deferred なら所有権を保持して残す。
            const DestroyOutcome outcome = e.retryable(e.ptr);
            if (outcome == DestroyOutcome::Destroyed)
            {
                convo::fetchAddAtomic(terminalDrainEntryCount_, std::uint64_t{1},
                                      std::memory_order_release);
                if (e.type == DeletionEntryType::World)
                {
                    convo::fetchAddAtomic(reclaimCount_, std::uint64_t{1},
                                          std::memory_order_acq_rel);
                    if (referenceObserver_ != nullptr)
                        referenceObserver_->onRelease();
                }
            }
            else
            {
                // E (2026-10-25): Deferred = the callee retained ownership (the engine
                // retry list). drainAll must NOT keep the entry either, otherwise the same
                // pointer stays owned by two authorities across shutdown.
                convo::fetchAddAtomic(ownershipTransferredCount_, std::uint64_t{1},
                                      std::memory_order_acq_rel);
                convo::fetchAddAtomic(terminalDrainEntryCount_, std::uint64_t{1},
                                      std::memory_order_release);
            }
            continue;
        }

        if (e.deleter != nullptr) {
            e.deleter(e.ptr);
            // ★ Step 5-I: telemetry — count actually drained entries
            convo::fetchAddAtomic(terminalDrainEntryCount_, std::uint64_t{1}, std::memory_order_release);
            if (e.type == DeletionEntryType::World)
            {
                // ★ Step 5-I / AtomicAccess convention: use convo:: wrapper (not raw ++)
                convo::fetchAddAtomic(reclaimCount_, std::uint64_t{1}, std::memory_order_acq_rel);
                if (referenceObserver_ != nullptr)
                    referenceObserver_->onRelease();
            }
        }
    }

    // ★ F1'-b: Deferred だったエントリを所有権ごと戻す（未回収として保持）。
    if (!kept.empty())
    {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& e : kept)
            entries_.push_back(e);
        convo::publishAtomic(residentAtomic_,
                             static_cast<uint32_t>(entries_.size()),
                             std::memory_order_release);
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

// V-11: delegate ownership-checked re-acquisition (see IReaderEpochProvider).
//   Fail-closed when no provider is installed, matching the other delegations.
bool ISRRetireRouter::reacquireReaderSlot(int readerIndex) noexcept
{
    assert(provider_ != nullptr);
    if (provider_ == nullptr)
        return false;
    return provider_->reacquireReaderSlot(readerIndex);
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
    return m_terminalReclaim.store(ptr, deleter, epoch, type, reason);  // NOLINT(atomic-dot-call)
}

// ★ P-4: TerminalReclaimAuthority 滞留件数
std::size_t ISRRetireRouter::terminalReclaimResidentCount() const noexcept
{
    return m_terminalReclaim.residentCount();
}

// ★ F1'-b (2026-10-25): 再試行可能な破棄の受け付け。
//   epoch safe かつ Non-RT なら即時実行し、Destroyed なら完了。
//   Deferred なら所有権ごと TerminalReclaimAuthority に移す（以後 drain で再試行）。
bool ISRRetireRouter::terminalReclaimRetryable(void* ptr, RetryableDeleter retryable,
                                               uint64_t epoch, DeletionEntryType type,
                                               const char* reason) noexcept
{
    if (ptr == nullptr || retryable == nullptr)
        return true;

    const uint64_t minReader = minReaderEpoch();
    const bool epochSafe = ISRRetireRouter::isOlder(epoch, minReader);
    const bool isRt = convo::numeric_policy::isAudioThread();

    // ★ F1'-b (2026-10-25) 監査対応（欠陥 B）: 所有権の一元化。
    //   `retryable` が Deferred を返した場合、その実装（destroyDSPCoreNodeRetryable）は
    //   **自分でエンジンの再試行リストに登録する**。ここでさらに
    //   storeRetryable() すると同じポインタが 2 箇所に登録され、
    //   片方が解放した後も他方が残る（= 二重破棄 / UAF）。
    //
    //   したがって「Deferred は deleter の所有者に委ねる」を契約とし、
    //   TerminalReclaimAuthority への格納は epoch が安全でない場合
    //   （= deleter をまだ呼べない場合）に限る。
    if (epochSafe && !isRt)
    {
        if (retryable(ptr) == DestroyOutcome::Destroyed)
        {
            if (type == DeletionEntryType::World)
                m_terminalReclaim.recordWorldReclaim();
        }
        // Deferred の場合、所有権は retryable の実装（エンジン再試行リスト）が保持する。
        // ここで再格納しない（二重所有の禁止）。
        return true;
    }

    // epoch unsafe / RT: deleter を呼べないので、Terminal に預けて後続 drain に委ねる。
    //   （この経路では retryable はまだ 1 度も呼ばれていないため、所有権は Terminal が持つ。）
    return m_terminalReclaim.storeRetryable(ptr, retryable, epoch, type, reason);
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
