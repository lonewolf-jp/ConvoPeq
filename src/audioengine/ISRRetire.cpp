#include "ISRRetire.h"
#include <immintrin.h>  // _mm_pause
#include "ISRRetireOverflowRing.h"
#include "AtomicAccess.h"

#include <algorithm>
#include <chrono>

namespace convo {
namespace isr {

void RetireRuntime::initQueue() noexcept
{
    for (size_t i = 0; i < RETIRE_INTENT_QUEUE_SIZE; ++i) {
        convo::publishAtomic(slots_[i].sequence,
                             static_cast<uint64_t>(i),
                             std::memory_order_release);
    }
    convo::publishAtomic(enqueueTicket_, uint64_t{0}, std::memory_order_relaxed);
    convo::publishAtomic(dequeuePos_, uint64_t{0}, std::memory_order_relaxed);
}

void RetireRuntime::emitRetireIntent(const RetireIntent& intent) noexcept
{
    // ★ Step 1: MPSC Queue に slot を予約 (Vyukov protocol)
    const uint64_t ticket = convo::fetchAddAtomic(enqueueTicket_, 1, std::memory_order_acq_rel);
    const size_t idx = ticket % RETIRE_INTENT_QUEUE_SIZE;

    RetireIntent localIntent = intent;

    // ★ Step 2: bounded spin — Consumer が slot を解放するまで待機
    static constexpr int kMaxProducerSpin = 64;
    for (int spin = 0;; ++spin) {
        uint64_t slotSeq = convo::consumeAtomic(
            slots_[idx].sequence, std::memory_order_acquire);
        if (slotSeq == ticket) break;  // slot 獲得

        if (spin >= kMaxProducerSpin) {
            // ★ bounded spin 失敗 → tombstone + fallback
            slots_[idx].payload = RetireIntent{};
            slots_[idx].payload.dspSlot = UINT32_MAX;  // tombstone 識別子
            convo::publishAtomic(slots_[idx].sequence, ticket + 1, std::memory_order_release);

            std::lock_guard<std::mutex> lock(fallbackMutex_);
            if (fallbackCount_ < FALLBACK_QUEUE_CAPACITY) {
                const size_t tail = (fallbackHead_ + fallbackCount_) % FALLBACK_QUEUE_CAPACITY;
                fallbackQueue_[tail] = localIntent;
                ++fallbackCount_;
                convo::publishAtomic(fallbackQueuePeak_, fallbackCount_.load(std::memory_order_relaxed), std::memory_order_release);
                (void)convo::fetchAddAtomic(overflowCount_, uint64_t{1}, std::memory_order_acq_rel);
            } else {
                // ★ Fallback も満杯 → OverflowRing へ退避試行
                bool dropped = true;
                if (overflowRing_ != nullptr) {
                    RetireOverflowEntry entry{localIntent, static_cast<uint64_t>(
                        std::chrono::steady_clock::now().time_since_epoch().count()), 0};
                    if (overflowRing_->tryPush(entry)) {
                        convo::fetchAddAtomic(quarantineRescuedCount_, uint64_t{1}, std::memory_order_release);
                        dropped = false;
                    }
                }
                if (dropped) {
                    (void)convo::fetchAddAtomic(fallbackOverflowCount_, uint64_t{1}, std::memory_order_acq_rel);
                    (void)convo::fetchAddAtomic(overflowCount_, uint64_t{1}, std::memory_order_acq_rel);
                    (void)convo::fetchAddAtomic(droppedIntentCount_, uint64_t{1}, std::memory_order_acq_rel);
                }
            }

            // ★ C-1: overflowStartTimestamp_ を初回のみ設定（CAS）
            uint64_t expected = 0;
            convo::compareExchangeAtomic(overflowStartTimestamp_, expected,
                static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()),
                std::memory_order_release);
            (void)convo::fetchAddAtomic(overflowWindowCounter_, uint64_t{1}, std::memory_order_release);
            convo::publishAtomic(lastOverflowTicks_,
                static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()),
                std::memory_order_release);
            return;
        }
        _mm_pause();
    }

    // ★ C-1: success（キュー空きあり）: overflow が継続中ならタイムスタンプをリセット
    uint64_t prevStart = convo::exchangeAtomic(overflowStartTimestamp_, uint64_t{0},
        std::memory_order_release);
    (void)prevStart;

    // ★ Step 3: payload 書き込み + release
    slots_[idx].payload = localIntent;
    convo::publishAtomic(slots_[idx].sequence, ticket + 1, std::memory_order_release);
}

void RetireRuntime::emitRetireIntentRT(const RetireIntent& intent) noexcept
{
    // ★ Finding 9: 「RT」は RealTime thread safety を意味しない。
    //   実装は emitRetireIntent() を素通しし、輻輳時に std::mutex をロックする。
    //   現時点では呼び出し元は全て非 RT スレッドであることを確認済み。
    //   将来 Audio Thread から呼び出す場合は、mutex を使わない別実装を用意すること。
    //   将来リネーム予定: emitRetireIntentFromNonRT（バージョンアップ時に実施）
    //   注: jassert(!isAudioThread()) は ISRRetire.cpp では JUCE ヘッダ未インクルードのため使用不可
    emitRetireIntent(intent);
}

bool RetireRuntime::dequeueOne(RetireIntent& out) noexcept
{
    for (;;) {
        const uint64_t pos = convo::consumeAtomic(dequeuePos_, std::memory_order_relaxed);
        const size_t idx = static_cast<size_t>(pos % RETIRE_INTENT_QUEUE_SIZE);
        const uint64_t expectedSeq = pos + 1;
        const uint64_t slotSeq = convo::consumeAtomic(
            slots_[idx].sequence, std::memory_order_acquire);
        if (slotSeq != expectedSeq) return false;  // 未 ready

        out = slots_[idx].payload;
        // ★ tombstone check: dspSlot == UINT32_MAX で fallback 経由の無効 intent を識別
        if (out.dspSlot == UINT32_MAX) {
            convo::publishAtomic(slots_[idx].sequence,
                pos + RETIRE_INTENT_QUEUE_SIZE,
                std::memory_order_release);
            convo::publishAtomic(dequeuePos_, pos + 1, std::memory_order_relaxed);
            continue;  // 次へ (ループ)
        }
        // ★ slot 解放: sequence = dequeuePos + SIZE → 次 cycle Producer が再利用可能
        convo::publishAtomic(slots_[idx].sequence,
            pos + RETIRE_INTENT_QUEUE_SIZE,
            std::memory_order_release);
        convo::publishAtomic(dequeuePos_, pos + 1, std::memory_order_relaxed);
        return true;
    }
}

bool RetireRuntime::dequeueFallback(RetireIntent& out) noexcept
{
    std::lock_guard<std::mutex> lock(fallbackMutex_);
    const size_t count = convo::consumeAtomic(fallbackCount_, std::memory_order_relaxed);
    if (count == 0) return false;
    out = fallbackQueue_[fallbackHead_];
    fallbackHead_ = (fallbackHead_ + 1) % FALLBACK_QUEUE_CAPACITY;
    convo::publishAtomic(fallbackCount_, count - 1, std::memory_order_relaxed);
    return true;
}

std::vector<RetireIntent> RetireRuntime::dequeuePendingRetireIntents() noexcept
{
    std::vector<RetireIntent> result;
    result.reserve(128);

    // 1. Drain Vyukov MPSC queue + Fallback (fair scheduling: main 8 : fallback 1)
    {
        constexpr size_t kMainToFallbackRatio = 8;
        RetireIntent raw;
        while (true) {
            bool progressed = false;
            for (size_t i = 0; i < kMainToFallbackRatio; ++i) {
                if (!dequeueOne(raw)) break;
                result.push_back(raw);
                progressed = true;
            }
            if (dequeueFallback(raw)) {
                result.push_back(raw);
                progressed = true;
            }
            if (!progressed) break;
        }
    }

    // ★ Phase 5: 複合ソートキー (priority, retireEpoch, generation, dspSlot)
    std::stable_sort(result.begin(), result.end(), [](const RetireIntent& lhs, const RetireIntent& rhs) noexcept {
        if (lhs.priority != rhs.priority)
            return lhs.priority > rhs.priority;  // priority降順（Critical最先頭）
        if (lhs.retireEpoch != rhs.retireEpoch)
            return lhs.retireEpoch < rhs.retireEpoch;
        if (lhs.generation != rhs.generation)
            return lhs.generation < rhs.generation;
        return lhs.dspSlot < rhs.dspSlot;
    });

    return result;
}

std::uint64_t RetireRuntime::pendingIntentCount() const noexcept
{
    const uint64_t enqueued = convo::consumeAtomic(enqueueTicket_, std::memory_order_acquire);
    const uint64_t consumed = convo::consumeAtomic(dequeuePos_, std::memory_order_acquire);
    const uint64_t mainPending = (enqueued > consumed) ? (enqueued - consumed) : 0;
    const uint64_t fbPending = convo::consumeAtomic(fallbackCount_, std::memory_order_relaxed);
    return mainPending + fbPending;
}

std::uint64_t RetireRuntime::overflowCount() const noexcept
{
    return convo::consumeAtomic(overflowCount_, std::memory_order_acquire);
}

std::uint64_t RetireRuntime::droppedIntentCount() const noexcept
{
    return convo::consumeAtomic(droppedIntentCount_, std::memory_order_acquire);
}

void RetireRuntime::acknowledgeRetireCoordination(const RetireIntent& intent)
{
    const auto idx = static_cast<std::size_t>(intent.dspSlot % RETIRE_INTENT_QUEUE_SIZE);
    convo::publishAtomic(acknowledgeGeneration_[idx], intent.generation, std::memory_order_release);
    (void)convo::fetchAddAtomic(acknowledgedCount_, uint64_t{1}, std::memory_order_acq_rel);
}

// ★ C-1: overflow 継続時間追跡 getter
std::uint64_t RetireRuntime::overflowStartTimestamp() const noexcept
{
    return convo::consumeAtomic(overflowStartTimestamp_, std::memory_order_acquire);
}

std::uint64_t RetireRuntime::lastOverflowTicks() const noexcept
{
    return convo::consumeAtomic(lastOverflowTicks_, std::memory_order_acquire);
}

std::uint64_t RetireRuntime::overflowWindowCounter() const noexcept
{
    return convo::consumeAtomic(overflowWindowCounter_, std::memory_order_acquire);
}

std::uint64_t RetireRuntime::lastOverflowWindowCount() const noexcept
{
    return convo::consumeAtomic(lastOverflowWindowCount_, std::memory_order_acquire);
}

// ★ P1: Fallback queue metrics
std::size_t RetireRuntime::fallbackOccupancy() const noexcept
{
    return convo::consumeAtomic(fallbackCount_, std::memory_order_acquire);
}

std::size_t RetireRuntime::fallbackHighWatermark() const noexcept
{
    return convo::consumeAtomic(fallbackQueuePeak_, std::memory_order_acquire);
}

std::uint64_t RetireRuntime::fallbackOverflowCount() const noexcept
{
    return convo::consumeAtomic(fallbackOverflowCount_, std::memory_order_acquire);
}

// ★ Phase5: 全保留中Intentの優先度を底上げ（Shutdown時の Critical 一括昇格用）
//   Shutdown/AudioStopped 後は Audio Thread が動作していないため、
//   MPSC queue + fallback queue の走査は安全。
//   ※ isValid/priority は std::atomic ではないが、Shutdown中は単一スレッドアクセス。
void RetireRuntime::escalateAllRetires(RetirePriority minPriority) noexcept
{
    // Vyukov MPSC slots: 全スロットを走査
    for (size_t i = 0; i < RETIRE_INTENT_QUEUE_SIZE; ++i)
    {
        auto& intent = slots_[i].payload;
        // ★ dspSlot != UINT32_MAX で有効な intent を識別 (isValid 廃止)
        if (intent.dspSlot != UINT32_MAX
            && static_cast<uint8_t>(intent.priority) < static_cast<uint8_t>(minPriority))
        {
            intent.priority = minPriority;
        }
    }

    // Fallback queue: 全エントリを走査
    {
        std::lock_guard<std::mutex> lock(fallbackMutex_);
        for (size_t i = 0; i < fallbackCount_; ++i)
        {
            const size_t idx = (fallbackHead_ + i) % FALLBACK_QUEUE_CAPACITY;
            auto& intent = fallbackQueue_[idx];
            if (static_cast<uint8_t>(intent.priority) < static_cast<uint8_t>(minPriority))
            {
                intent.priority = minPriority;
            }
        }
    }
}

}  // namespace isr
}  // namespace convo
