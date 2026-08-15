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
    // ★ work88 (X3 §6.3 / INV-X3-4): reader registration permanently closed — shutdown 後は
    //   新規登録を拒否する（registrationClosed_）。登録済み slot の enter/exit は継続可能
    //   （既存 Reader の epoch 安全性は維持 — 十八次別視点14）。
    int registerReaderThread(const char* tag) noexcept
    {
        if (convo::consumeAtomic(registrationClosed_, std::memory_order_acquire))
            return -1;   // ★ X3: reader registration permanently closed（INV-X3-4）
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
        // ★ work88 (X3 §6.3 / INV-X3-4): reserve 経由の登録も封じる（registrationClosed_ ガード）
        if (convo::consumeAtomic(registrationClosed_, std::memory_order_acquire))
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
        // ★ BUG-050: epoch を depth++ より前に store して HB ギャップを除去する。
        //   getMinReaderEpoch が depth>0 を観測した時点（depth acquire）で、
        //   [B'] epoch store（depth++ より前）が release→acquire チェーンで必ず可視化される。
        const uint64_t epoch = currentEpoch();
        convo::publishAtomic(slot.epoch, epoch, std::memory_order_release);

        // release: 後続の epoch load が depth > 0 可視後に行われることを保証。
        const uint32_t previousDepth = convo::fetchAddAtomic(slot.depth,
                                                              static_cast<uint32_t>(1),
                                                              std::memory_order_acq_rel);
        if (previousDepth > 0)
            return;  // ネスト: epoch は active Reader を反映済み（>= 外側 epoch）で安全。再設定不要。

        // ★ Practical-8: 初回 enter 時に滞留開始時刻を記録
        const uint64_t nowUs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        convo::publishAtomic(slot.residencyStartTimestampUs, nowUs, std::memory_order_release);
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
        // ★ BUG-049: CAS で 0x02(pending)→0x01(quarantined) へ原子的に昇格。
        //   plain store だと別 Coordinator が pending を再設定した際に競合するため。
        const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
        if ((flags & ReaderSlot::kPendingQuarantineFlag) != 0)
        {
            uint8_t expected = static_cast<uint8_t>(ReaderSlot::kPendingQuarantineFlag);
            convo::compareExchangeAtomic(slot.quarantineFlags,
                                         expected,
                                         static_cast<uint8_t>(ReaderSlot::kQuarantinedFlag),
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire);
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
        // ★ dash2 §2.2 (G19): epoch 前進と同期して epochGeneration_ をインクリメント。
        //   ShutdownRuntimeIdentity::epochGeneration の実供給源（T10 / Permit ABA 防止）。
        (void)convo::fetchAddAtomic(epochGeneration_, static_cast<uint64_t>(1),
                                    std::memory_order_acq_rel);
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
    // ★ BUG-049: 即座隔離と遅延隔離の並行 store 競合を CAS で排除。
    //   従来は plain store で 0x02(pending) と 0x01(quarantined) が競合し、
    //   depth==0 なのに pending が生存 → verifyReaderInvariants assert 発火。
    //   CAS を使うことで flags は 0x00→0x02(pending のみ) / 0x00→0x01 / 0x02→0x01(/昇格) の
    //   原子的遷移のみ許し、0x03(quarantined|pending) を生成しない。
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
            // ★ 競合を許容しないため CAS を使用。
            //   ① pending(0x02) が別 Coordinator によって設定されていた → quarantined(0x01) へ昇格
            uint8_t expected = static_cast<uint8_t>(ReaderSlot::kPendingQuarantineFlag);
            if (convo::compareExchangeAtomic(slot.quarantineFlags,
                                             expected,
                                             static_cast<uint8_t>(ReaderSlot::kQuarantinedFlag),
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
                return true;
            //   ② 未設定(0x00) からの即座 quarantine
            expected = static_cast<uint8_t>(0);
            if (convo::compareExchangeAtomic(slot.quarantineFlags,
                                             expected,
                                             static_cast<uint8_t>(ReaderSlot::kQuarantinedFlag),
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
                return true;
            //   ③ 既に quarantined/その他 → 隔離は達成済み
            return false;
        }

        // depth > 0: 遅延隔離 — exitReader で depth==0 になった時点で quarantine 確定
        // ★ 既に quarantined なら遅延設定は不要。
        const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
        if ((flags & ReaderSlot::kQuarantinedFlag) != 0)
            return false;
        // 未設定(0x00) からのみ pending(0x02) を CAS で設定（0x03 生成を防止）。
        uint8_t expectPending = static_cast<uint8_t>(0);
        convo::compareExchangeAtomic(slot.quarantineFlags,
                                     expectPending,
                                     static_cast<uint8_t>(ReaderSlot::kPendingQuarantineFlag),
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire);
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

        // ★ BUG-048: 単一ループの break-on-first を、深刻度の高い順の 3 パス評価に変更。
        //   従来は readerIndex 昇順で「最初に見つかった」Reader を報告し、
        //   より深刻な Chronic が index の大きい Reader に居ると見逃され得た。
        //   パス順序: Pass3 Chronic → Pass2 Warning → Pass1 EpochGap。
        auto buildReaderInfo = [&](int i, int severity) -> bool {
            if (i < 0 || i >= kMaxReaders)
                return false;
            const auto& slot = readers[static_cast<size_t>(i)];
            const uint64_t readerEpoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
            if (readerEpoch == kInactiveEpoch)
                return false;

            const uint64_t ec = slot.enterCount.load(std::memory_order_relaxed);
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);

            // ★ P4.5: residencyTime を実時間ベースで計算（epoch差ではなくsteady_clock）
            const uint64_t startUs = convo::consumeAtomic(slot.residencyStartTimestampUs, std::memory_order_acquire);
            const uint64_t residencyUs = (startUs != 0 && depth > 0) ? (nowUs - startUs) : 0;

            bool matched = false;
            switch (severity) {
                case 3:  // [work37] 条件2: residency > 30秒 (epoch差不問) → Chronic Stuck
                    matched = (depth > 0 && residencyUs > kChronicResidencyUs && info.pendingRetireCount > 0);
                    if (matched)
                        info.isChronic = true;
                    break;
                case 2:  // [work37] 条件3: residency > 10秒 AND pendingRetire > 0 → Warning Stuck
                    matched = (depth > 0 && residencyUs > kWarningResidencyUs && info.pendingRetireCount > 0);
                    if (matched)
                        info.isChronic = false;
                    break;
                case 1:  // [work37] 条件1: epoch差 > threshold AND residency > 1秒
                    if (depth > 0 && readerEpoch < info.currentEpoch) {
                        const uint64_t epochGap = info.currentEpoch - readerEpoch;
                        matched = (epochGap > stuckThreshold && residencyUs > kResidencyStuckUs);
                    }
                    if (matched)
                        info.isChronic = false;
                    break;
                default:
                    break;
            }
            if (!matched)
                return false;

            info.readerIndex = i;
            info.readerEpoch = readerEpoch;
            info.enterCount = ec;
            info.isStuck = true;
            info.residencyTimeUs = residencyUs;
            // [work37 9.42] + BUG-063: ownerThreadId を先に acquire し、非0時のみ ownerTag をコピー
            info.ownerThreadId = convo::consumeAtomic(slot.ownerThreadId, std::memory_order_acquire);
            if (info.ownerThreadId != 0) {
                std::strncpy(info.ownerTag, slot.ownerTag, sizeof(info.ownerTag) - 1);
                info.ownerTag[sizeof(info.ownerTag) - 1] = '\0';
            }
            return true;
        };

        for (int pass = 3; pass >= 1; --pass) {
            for (int i = 0; i < kMaxReaders; ++i) {
                if (buildReaderInfo(i, pass))
                    return info;
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

    // ★ work88 (X3 §6.3 / INV-X3-4 / INV-ISR-04): reader registration permanently closed フラグ。
    //   closeReaderRegistration() で true（shutdown state machine の CloseReaderRegistration フェーズ）。
    //   registerReaderThread / reserveReaderThread は true 後は失敗を返す（新規登録拒否）。
    std::atomic<bool> registrationClosed_{false};

    // ── ★ dash2 §2.2 (Phase A2 — G19/G20): generation 実供給源 ──
    //   ShutdownRuntimeIdentity の epochGeneration / readerRegistrationGeneration を束縛する
    //   generation 値。型フィールドの存在だけでなく、authority（EpochDomain）からの実供給・
    //   検証経路を成立させる（H.11.11.9.3 G-D1〜D4）。
    //   - epochGeneration_: publishEpoch() ごとにインクリメント（epoch 前進と同期）
    //   - readerRegistrationGeneration_: closeReaderRegistration() ごとにインクリメント
    //     （registration 状態の世代 — 再オープンは存在しないため単調増加）
    //   ［生成は EpochDomain のみ。ShutdownRuntime は getter 経由で観測し Proof/Permit に束縛］
    std::atomic<uint64_t> epochGeneration_{0};
    std::atomic<uint64_t> readerRegistrationGeneration_{0};

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

    // ★ work88 (X3 §6.3 / INV-X3-4): reader registration を永久に閉じる（CloseReaderRegistration）。
    //   新規登録のみを拒否し、登録済み slot は exit まで継続可能（既存 Reader の epoch 安全性維持）。
    void closeReaderRegistration() noexcept {
        convo::publishAtomic(registrationClosed_, true, std::memory_order_release);
        // ★ dash2 §2.2 (G20): registration 状態の世代をインクリメント。
        //   ShutdownRuntimeIdentity::readerRegistrationGeneration の実供給源。
        //   registrationClosed_ の publish 後（release）に increment するため、
        //   acquire で読む側は「閉じた後の世代」を観測する（INV-LIFE-6 / T10）。
        (void)convo::fetchAddAtomic(readerRegistrationGeneration_, static_cast<uint64_t>(1),
                                    std::memory_order_acq_rel);
    }
    [[nodiscard]] bool readerRegistrationClosed() const noexcept {
        return convo::consumeAtomic(registrationClosed_, std::memory_order_acquire);
    }

    // ★ dash2 §2.2 (G19/G20): generation 公開アクセサ（ShutdownRuntime / Proof 生成用）。
    [[nodiscard]] uint64_t epochGeneration() const noexcept {
        return convo::consumeAtomic(epochGeneration_, std::memory_order_acquire);
    }
    [[nodiscard]] uint64_t readerRegistrationGeneration() const noexcept {
        return convo::consumeAtomic(readerRegistrationGeneration_, std::memory_order_acquire);
    }
};

} // namespace convo
