#pragma once

#include <array>
#include <cassert>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>

#include "../DeferredDeletionQueue.h"
#include "IEpochProvider.h"
#include "audioengine/AtomicAccess.h"
#include "ThreadHash.h"

namespace convo {

class EpochDomain : public IEpochProvider
{
public:
    static constexpr int kMaxReaders = 64;
    static constexpr uint64_t kInactiveEpoch = std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t kReservedEpoch = std::numeric_limits<uint64_t>::max() - 1;

    EpochDomain() : globalEpoch(1)
    {
        for (auto& slot : readers)
        {
            // release: コンストラクタ内で単一スレッドから初期化するが、
            //          完了後に他スレッドがオブジェクトを取得する際に acquire で可視性を保証するため release。
            convo::publishAtomic(slot.epoch, kInactiveEpoch, std::memory_order_release);
            convo::publishAtomic(slot.depth, static_cast<uint32_t>(0), std::memory_order_release);
            // ★ Phase 3: quarantineFlags ゼロ初期化
            convo::publishAtomic(slot.quarantineFlags, static_cast<uint8_t>(0), std::memory_order_release);
        }
    }

    int registerReaderThread() noexcept override
    {
        return registerReaderThread("unnamed");
    }

    // ★ C-3: タグ名付き Reader 登録
    int registerReaderThread(const char* tag) noexcept
    {
        for (int i = 0; i < kMaxReaders; ++i)
        {
            uint64_t expected = kInactiveEpoch;
            // acq_rel/acquire: 成功側 release で slot 取得を他スレッドに公開し、
            //                  failure 側 acquire で競合の write を観測してループを継続。
            if (convo::compareExchangeAtomic(readers[static_cast<size_t>(i)].epoch,
                                             expected,
                                             kReservedEpoch,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
            {
                // release: depth ゼロ化を slot 取得後に他スレッドが観測できるよう publish。
                convo::publishAtomic(readers[static_cast<size_t>(i)].depth,
                                     static_cast<uint32_t>(0),
                                     std::memory_order_release);
                // ★ C-3: 所有者タグ設定（CAS 成功後は単一スレッドのみがアクセス可能）
                if (tag != nullptr) {
                    std::strncpy(readers[static_cast<size_t>(i)].ownerTag, tag,
                                 sizeof(readers[static_cast<size_t>(i)].ownerTag) - 1);
                    readers[static_cast<size_t>(i)].ownerTag[
                        sizeof(readers[static_cast<size_t>(i)].ownerTag) - 1] = '\0';
                }
                convo::publishAtomic(readers[static_cast<size_t>(i)].ownerThreadId,
                                     convo::cachedThreadHash(),
                                     std::memory_order_release);
                return i;
            }
        }

        return -1;
    }

    bool reserveReaderThread(int readerIndex) noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return false;

        uint64_t expected = kInactiveEpoch;
        // acq_rel/acquire: registerReaderThread と同じ HB 保証が必要。
        //   成功側 release で予約を公開し、failure 側 acquire で競合書き込みを観測。
        const bool reserved = convo::compareExchangeAtomic(
            readers[static_cast<size_t>(readerIndex)].epoch,
            expected,
            kReservedEpoch,
            std::memory_order_acq_rel,
            std::memory_order_acquire);

        if (reserved)
        {
            // release: depth ゼロ化を他スレッドが観測できるよう予約成功後に publish。
            convo::publishAtomic(readers[static_cast<size_t>(readerIndex)].depth,
                                 static_cast<uint32_t>(0),
                                 std::memory_order_release);
        }

        return reserved;
    }

    [[deprecated("Use RCUReader::enter() instead. See refactoring_plan.md P1-18.")]]
    void enterReader(int readerIndex) noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return;

        auto& slot = readers[static_cast<size_t>(readerIndex)];
        // acq_rel: 取得側 acquire で直前の exitReader release を観測し、
        //          放出側 release で後続の epoch load が depth > 0 可視後に行われることを保証。
        const uint32_t previousDepth = convo::fetchAddAtomic(slot.depth,
                                                              static_cast<uint32_t>(1),
                                                              std::memory_order_acq_rel);
        if (previousDepth > 0)
            return;

        // ★ Practical-8: 初回 enter 時に滞留開始時刻を記録
        const uint64_t nowUs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        convo::publishAtomic(slot.residencyStartTimestampUs, nowUs, std::memory_order_release);

        const uint64_t epoch = currentEpoch();
        // release: epoch を publish することで reclaimers が slot.epoch の safe-below 判定に使用可能となる。
        convo::publishAtomic(slot.epoch, epoch, std::memory_order_release);
    }

    [[deprecated("Use RCUReader::exit() instead. See refactoring_plan.md P1-18.")]]
    void exitReader(int readerIndex) noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return;

        auto& slot = readers[static_cast<size_t>(readerIndex)];
        // acq_rel: 取得側 acquire で enterReader 以降の読み取りが完了していることを観測し、
        //          放出側 release でその読み取りが slot.epoch の inactive 化より先に完了することを保証。
        const uint32_t previousDepth = convo::fetchSubAtomic(slot.depth,
                                                              static_cast<uint32_t>(1),
                                                              std::memory_order_acq_rel);
        if (previousDepth == 0)
        {
            convo::publishAtomic(slot.depth, static_cast<uint32_t>(0), std::memory_order_release);
            return;
        }

        if (previousDepth > 1)
            return;

        // ★ Practical-8: 最終 exit 時に滞留時刻をクリア
        convo::publishAtomic(slot.residencyStartTimestampUs, uint64_t{0}, std::memory_order_release);

        // release: epoch を kInactiveEpoch に戻し、reclaimers がこのスロットを safe-below 判定から除外可能にする。
        convo::publishAtomic(slot.epoch, kInactiveEpoch, std::memory_order_release);

        // ★ Phase 3: pendingQuarantine が設定されていた場合、今 quarantine を確定する。
        //   この Reader は exit 後も quarantined フラグにより getMinReaderEpoch から除外される。
        const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
        if ((flags & ReaderSlot::kPendingQuarantineFlag) != 0)
        {
            // release: quarantined フラグを設定し、pending をクリア。
            //   Coordinator が acquire でこの書き込みを観測する。
            convo::publishAtomic(slot.quarantineFlags,
                                 static_cast<uint8_t>(ReaderSlot::kQuarantinedFlag),
                                 std::memory_order_release);
        }
    }

    uint64_t currentEpoch() const noexcept override
    {
        // acquire: advanceEpoch の acq_rel release-side と HB し、最新 epoch を観測する。
        return convo::consumeAtomic(globalEpoch, std::memory_order_acquire);
    }

    // [work21] IEpochProvider::publishEpoch — inline advance to avoid deprecated call
    uint64_t publishEpoch() noexcept override
    {
        return convo::fetchAddAtomic(globalEpoch,
                                     static_cast<uint64_t>(1),
                                     std::memory_order_acq_rel);
    }

    uint64_t current() const noexcept
    {
        return currentEpoch();
    }

    uint64_t publish() noexcept
    {
        return publishEpoch();
    }

    uint64_t getMinReaderEpoch() const noexcept override
    {
        uint64_t minEpoch = currentEpoch();

        for (const auto& slot : readers)
        {
            // ★ Phase 3: quarantined Reader は safe-epoch 計算から除外
            //   kQuarantinedFlag 設定時は depth==0 が不変条件:
            //     - 即座 quarantine: depth==0 でのみ設定
            //     - 遅延 quarantine: exitReader で depth:1→0 後に昇格
            //   したがって depth の再チェックは不要だが、防衛的アサートで担保する。
            const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
            if ((flags & ReaderSlot::kQuarantinedFlag) != 0)
            {
                assert(convo::consumeAtomic(slot.depth, std::memory_order_acquire) == 0
                    && "quarantined reader must have depth==0");
                continue;
            }

            // acquire: enterReader release の depth 書き込みと HB し、depth 読み取り後に epoch を読む。
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
            if (depth == 0)
                continue;

            // acquire: enterReader の epoch publish release と HB し、安全に epoch 値を取得。
            const uint64_t epoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
            if (epoch == kInactiveEpoch || epoch == kReservedEpoch)
                continue;

            if (isOlder(epoch, minEpoch))
                minEpoch = epoch;
        }

        return minEpoch;
    }

    uint32_t activeReaderCount() const noexcept override
    {
        uint32_t count = 0;

        for (const auto& slot : readers)
        {
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
            if (depth != 0)
                ++count;
        }

        return count;
    }

    int readerCapacity() const noexcept override
    {
        return kMaxReaders;
    }

    // ── ★ Phase 3: Reader Quarantine API ──

    // ★ stuck Reader を quarantined にマーク（killしない）
    //   depth==0: 即座quarantine → true
    //   depth>0: pendingQuarantine設定 → exitReader時にquarantine → false (deferred)
    [[nodiscard]] bool quarantineReader(int readerIndex) noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return false;

        auto& slot = readers[static_cast<size_t>(readerIndex)];
        // acquire: enterReader/exitReader の深度変更を観測
        const uint32_t d = convo::consumeAtomic(slot.depth, std::memory_order_acquire);

        if (d == 0)
        {
            // 即座 quarantine: 深度0 → 他スレッドが参照していない
            // release: Coordinator が設定した quarantined フラグを
            //   getMinReaderEpoch が acquire で観測可能にする。
            convo::publishAtomic(slot.quarantineFlags,
                                 static_cast<uint8_t>(ReaderSlot::kQuarantinedFlag),
                                 std::memory_order_release);
            return true;
        }

        // depth > 0: 遅延隔離 — exitReader で depth==0 になった時点で quarantine 確定
        // release: pending フラグを exitReader が acquire で観測可能にする。
        convo::publishAtomic(slot.quarantineFlags,
                             static_cast<uint8_t>(ReaderSlot::kPendingQuarantineFlag),
                             std::memory_order_release);
        return false;
    }

    // ★ Shutdown専用: 全quarantined Readerを解放（destroyForShutdown と同じパターン）
    void unquarantineAllReaders() noexcept override
    {
        for (auto& slot : readers)
        {
            convo::publishAtomic(slot.quarantineFlags,
                                 static_cast<uint8_t>(0),
                                 std::memory_order_release);
        }
    }

    // ★ quarantined Reader数の取得（kQuarantinedFlagでカウント）
    [[nodiscard]] int quarantinedReaderCount() const noexcept override
    {
        int count = 0;
        for (const auto& slot : readers)
        {
            const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
            if ((flags & ReaderSlot::kQuarantinedFlag) != 0)
                ++count;
        }
        return count;
    }

    // ★ Phase 3: Debug 検証 — quarantined slot の不変条件をチェック
    //   Release ビルドでは除去される（assert は NDEBUG で消滅）
    void verifyReaderInvariants() const noexcept
    {
        for (int i = 0; i < kMaxReaders; ++i)
        {
            const auto& slot = readers[static_cast<size_t>(i)];
            const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
            const uint64_t epoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
            (void)epoch;  // suppress icx -Wunused-variable (used only in assert)

            const bool isQuarantined = (flags & ReaderSlot::kQuarantinedFlag) != 0;
            const bool isPending = (flags & ReaderSlot::kPendingQuarantineFlag) != 0;

            // ★ quarantined Reader は epoch==kInactiveEpoch を期待（exitReader が epoch を inactive にした後で quarantine に遷移する）
            //   ※ 厳密には quarantined フラグ設定前に epoch が inactive になる保証はない（exitReader の acq_rel 順序で暗黙保証）。
            //     ここでは depth==0 かつ quarantined の場合は epoch も inactive であることを確認。
            if (isQuarantined && depth == 0)
            {
                assert(epoch == kInactiveEpoch || epoch == kReservedEpoch);
            }

            // ★ pendingQuarantine 設定中は depth > 0 を期待（depth==0 なら即座に quarantined に遷移するため pending は残らない）
            if (isPending)
            {
                assert(depth > 0 && "pendingQuarantine set but depth==0: should have been promoted to quarantined");
            }

            // ★ quarantined + pending は同時に成立しない
            assert(!(isQuarantined && isPending));
        }
    }

    // [work21] IEpochProvider::tryReclaim — inline reclaim to avoid deprecated call
    void tryReclaim() noexcept override
    {
        // ★ ★ A-2: 統計カウンタ (Local Aggregation によりキャッシュ競合低減)
        constexpr uint32_t kCounterAggregationInterval = 1024;
        const uint32_t localCount = reclaimLocalCounter_.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((localCount % kCounterAggregationInterval) == 0) {
            reclaimAttemptCount_.fetch_add(kCounterAggregationInterval, std::memory_order_relaxed);
        }
        const auto n = deferredDeletionQueue.reclaim(getMinReaderEpoch());
        reclaimSuccessCount_.fetch_add(n, std::memory_order_relaxed);
    }

    // ★ P0-A/P2-A: IRetireProvider インターフェース実装（public 必須）
    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept override
    {
        return deferredDeletionQueue.enqueue(ptr, deleter, epoch);
    }

    void drainAll() noexcept override
    {
        deferredDeletionQueue.drainAllUnsafe();
    }

    [[nodiscard]] uint32_t pendingRetireCount() const noexcept override
    {
        return deferredDeletionQueue.sizeApprox();
    }

    static bool isOlder(uint64_t a, uint64_t b) noexcept
    {
        return static_cast<int64_t>(a - b) < 0;
    }

    // ★ B-1: Reader Slot 詳細取得（アクティブ Reader の epoch/depth/residency を返す）
    [[nodiscard]] ReaderSlotDetail getReaderSlotDetail(int readerIndex) const noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return ReaderSlotDetail{};

        const auto& slot = readers[static_cast<size_t>(readerIndex)];
        const uint64_t epoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
        const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
        const uint64_t startUs = convo::consumeAtomic(slot.residencyStartTimestampUs, std::memory_order_acquire);
        const auto nowUs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        const uint64_t residencyUs = (startUs != 0 && depth > 0) ? (nowUs - startUs) : 0;

        return ReaderSlotDetail{epoch, depth, residencyUs, (depth > 0)};
    }

    // [work37 Phase 2.1] 複合判定: epoch差 AND residency 時間条件
    //   条件1: epoch差 > threshold AND residency > 1秒 → Stuck
    //   条件2: residency > 30秒 (epoch差不問) → Chronic Stuck
    //   条件3: depth > 0 AND residency > 10秒 AND pendingRetire > 0 → Warning Stuck
    [[nodiscard]] StuckReaderInfo detectStuckReaders(uint64_t stuckThreshold) const noexcept override {
        StuckReaderInfo info;
        info.currentEpoch = convo::consumeAtomic(globalEpoch, std::memory_order_acquire);
        info.minReaderEpoch = getMinReaderEpoch();
        info.pendingRetireCount = deferredDeletionQueue.sizeApprox();

        constexpr uint64_t kResidencyStuckUs = 1'000'000;      // 1秒 — epoch差とのAND条件
        constexpr uint64_t kChronicResidencyUs = 30'000'000;   // 30秒 — epoch差不問
        constexpr uint64_t kWarningResidencyUs = 10'000'000;   // 10秒 — Warning用

        const auto nowUs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());

        for (int i = 0; i < kMaxReaders; ++i) {
            const auto& slot = readers[i];
            const uint64_t readerEpoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
            if (readerEpoch == kInactiveEpoch)
                continue;

            const uint64_t ec = slot.enterCount.load(std::memory_order_relaxed);
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);

            // ★ P4.5: residencyTime を実時間ベースで計算（epoch差ではなくsteady_clock）
            const uint64_t startUs = convo::consumeAtomic(slot.residencyStartTimestampUs, std::memory_order_acquire);
            const uint64_t residencyUs = (startUs != 0 && depth > 0) ? (nowUs - startUs) : 0;

            // [work37] 条件2: residency > 30秒 (epoch差不問) → Chronic Stuck
            if (depth > 0 && residencyUs > kChronicResidencyUs && info.pendingRetireCount > 0) {
                info.readerIndex = i;
                info.readerEpoch = readerEpoch;
                info.enterCount = ec;
                info.isStuck = true;
                info.isChronic = true;
                info.residencyTimeUs = residencyUs;
                // [work37 9.42] ReaderSlot の所有者情報をコピー
                std::strncpy(info.ownerTag, slot.ownerTag, sizeof(info.ownerTag) - 1);
                info.ownerTag[sizeof(info.ownerTag) - 1] = '\0';
                info.ownerThreadId = convo::consumeAtomic(slot.ownerThreadId, std::memory_order_acquire);
                break;
            }

            // [work37] 条件3: depth > 0 AND residency > 10秒 AND pendingRetire > 0 → Warning Stuck
            if (depth > 0 && residencyUs > kWarningResidencyUs && info.pendingRetireCount > 0) {
                info.readerIndex = i;
                info.readerEpoch = readerEpoch;
                info.enterCount = ec;
                info.isStuck = true;
                info.residencyTimeUs = residencyUs;
                info.isChronic = false;
                // [work37 9.42] ReaderSlot の所有者情報をコピー
                std::strncpy(info.ownerTag, slot.ownerTag, sizeof(info.ownerTag) - 1);
                info.ownerTag[sizeof(info.ownerTag) - 1] = '\0';
                info.ownerThreadId = convo::consumeAtomic(slot.ownerThreadId, std::memory_order_acquire);
                break;
            }

            // 複合判定: epoch差 AND residency
            if (depth > 0 && readerEpoch < info.currentEpoch) {
                const uint64_t epochGap = info.currentEpoch - readerEpoch;
                // [work37] 条件1: epoch差 > threshold AND residency > 1秒
                if (epochGap > stuckThreshold && residencyUs > kResidencyStuckUs) {
                    info.readerIndex = i;
                    info.readerEpoch = readerEpoch;
                    info.enterCount = ec;
                    info.isStuck = true;
                    info.residencyTimeUs = residencyUs;
                    // [work37 9.42] ReaderSlot の所有者情報をコピー
                    std::strncpy(info.ownerTag, slot.ownerTag, sizeof(info.ownerTag) - 1);
                    info.ownerTag[sizeof(info.ownerTag) - 1] = '\0';
                    info.ownerThreadId = convo::consumeAtomic(slot.ownerThreadId, std::memory_order_acquire);
                    break;
                }
            }
        }
        return info;
    }

    // ★ P2-A: 以下の deprecated API は移行完了により private 化。
    //   外部からの新規使用を禁止し、publishEpoch() / tryReclaim() を推奨。
private:
    [[deprecated("Use publishEpoch() instead.")]]
    uint64_t advanceEpoch() noexcept
    {
        return convo::fetchAddAtomic(globalEpoch,
                                     static_cast<uint64_t>(1),
                                     std::memory_order_acq_rel);
    }

    [[deprecated("Use tryReclaim() instead.")]]
    void reclaimRetired() noexcept
    {
        deferredDeletionQueue.reclaim(getMinReaderEpoch());
    }

    [[deprecated("Use coordinator.enqueueRetire() instead.")]]
    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type) noexcept
    {
        return deferredDeletionQueue.enqueue(ptr, deleter, epoch, type);
    }

    struct ReaderSlot
    {
        std::atomic<uint64_t> epoch { kInactiveEpoch };
        std::atomic<uint32_t> depth { 0 };
        std::atomic<uint64_t> enterCount { 0 };  // ★ P3-1: enter 回数のみカウント（軽量）
        std::atomic<uint64_t> residencyStartTimestampUs { 0 }; // ★ P4.5: steady_clock ベースの滞留開始時刻
        // ★ C-3: Reader 所有者情報
        std::atomic<uint64_t> ownerThreadId { 0 };       // std::thread::id のハッシュ値
        char ownerTag[32] {};  // "AudioThread", "TimerThread" 等（CAS 排他下で設定、stale read 許容）

        // ★ Phase 3: Reader Quarantine フラグ
        //   bit 0 (kQuarantinedFlag):   隔離済 — getMinReaderEpoch から除外
        //   bit 1 (kPendingQuarantineFlag):  exitReader で depth==0 になったら隔離
        static constexpr uint8_t kQuarantinedFlag       = 0x01;
        static constexpr uint8_t kPendingQuarantineFlag  = 0x02;
        std::atomic<uint8_t> quarantineFlags{0};
    };

    std::atomic<uint64_t> globalEpoch;
    std::array<ReaderSlot, kMaxReaders> readers;
    DeferredDeletionQueue deferredDeletionQueue;

    // ★ A-2: EBR Queue Visibility 統計カウンタ
    std::atomic<uint64_t> reclaimAttemptCount_{0};
    std::atomic<uint64_t> reclaimSuccessCount_{0};
    // ★ A-2: Local Aggregation 用カウンタ (per-core cache line)
#pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<uint32_t> reclaimLocalCounter_{0};
#pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

public:
    // ★ A-2: 公開アクセサ
    [[nodiscard]] uint64_t reclaimAttemptCount() const noexcept override {
        // ★ 未集計分を加算 (relaxed で十分: 診断目的のため正確性は要求されない)
        const auto local = reclaimLocalCounter_.load(std::memory_order_relaxed);
        const auto committed = convo::consumeAtomic(reclaimAttemptCount_, std::memory_order_acquire);
        return committed + (local % 1024);
    }
    [[nodiscard]] uint64_t reclaimSuccessCount() const noexcept override {
        return convo::consumeAtomic(reclaimSuccessCount_, std::memory_order_acquire);
    }
};

} // namespace convo
