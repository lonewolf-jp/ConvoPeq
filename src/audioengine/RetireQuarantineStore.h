//==============================================================================
// RetireQuarantineStore.h — BUG-015/027 (work88) retire enqueue 失敗時の退避ストア
//
// 目的:
//   DeferredDeletionQueue::enqueue が full（= RT Reader が参照中の可能性）で失敗した際、
//   directDelete（即時解放）は RT 参照中のオブジェクトを破壊する = UAF を生む。
//   本ストアは「delete できないオブジェクトを安全に保持」し、epoch 安全到達後に
//   定期 drain で EBR 安全削除する。
//
// 設計（五次レビュー §5 — Authority Singularization）:
//   - ISRRetireRouter の Policy lane 配下に単一配置。SnapshotCoordinator /
//     DSPLifetimeManager はストアを直接保持せず、Router API（quarantineRetire()）経由で移送。
//   - QuarantinedEntry は DeferredDeletionQueue::DeletionEntry と同等フィールド
//     （ptr/deleter/epoch/type/publicationSequenceId/generation）を保持 — ownership transfer の完全性。
//   - allocation-free: std::array + index 配置（noexcept 保証下で push_back は禁止）。
//   - capacity exhaustion（store full）時に deleter を実行してはならない（UAF 構造的排除）。
//     呼出し元（Router）が jassert + HealthEvent / shutdown escalation を担当。
//   - drain は epoch 比較述語（EpochDomain::isOlder 相当）を注入して wraparound 安全に判定。
//==============================================================================
#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <mutex>
#include <type_traits>

#include "../DeferredDeletionQueue.h"  // DeletionEntryType
#include "core/TimeUtils.h"            // getCurrentTimeUs

namespace convo {
namespace isr {

// QuarantinedEntry — DeferredDeletionQueue::DeletionEntry と同等フィールド
struct QuarantinedEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
    uint64_t publicationSequenceId = 0;  // ★ 因果追跡（DeferredDeletionQueue と同型）
    uint64_t generation = 0;             // ★ 世代追跡（DSPLifetimeManager::currentRetiringGeneration_ と同型）
    const char* reason = nullptr;        // 診断用
    uint64_t enqueueTimeUs = 0;          // 診断用
};
static_assert(std::is_trivially_copyable_v<QuarantinedEntry>,
    "QuarantinedEntry must be trivially copyable for lock-free compatible storage");

/**
 * RetireQuarantineStore — retire enqueue 失敗（queue full = RT 参照中の可能性）時の退避ストア。
 *
 * スレッド安全: 全操作は NonRT（Timer / CoordinatorLoop / DSPLifetimeManager）から。
 *   mutable std::mutex で保護（RT パスからは参照されない — AudioEngine の
 *   DSPQuarantineManager / DeferredFreeThread と同パターン）。
 *
 * EBR 安全削除: drain(minReaderEpoch, isOlderFn) は epoch < minReaderEpoch（isOlder）に
 *   達したエントリのみ deleter 実行。それ以外は保持継続（RT 離脱待ち）。
 *   drainAllUnsafe() は Audio Thread 停止後のみ呼ばれる（destroyForShutdown と同契約）。
 */
class RetireQuarantineStore {
public:
    // ★ 三次レビュー: 固定 capacity・allocation-free。RT 参照中オブジェクトは通常
    //   100ms オーダーで解放されるため、過剰な backlog は異常系（HealthEvent 対象）。
    static constexpr std::size_t kMaxQuarantinedEntries = 512;

    // store へ退避。戻り値 false = store full（呼出し元は deleter を実行してはならない。
    //   type/generation/publicationSequenceId を保持し、drain 時の epoch safe-check に利用。
    //   本設計では health escalation が先行し、quarantine() が false を返さないことを目指す）。
    bool quarantine(void* ptr, void (*deleter)(void*), uint64_t epoch,
                    DeletionEntryType type, const char* reason,
                    uint64_t publicationSequenceId = 0, uint64_t generation = 0) noexcept
    {
        if (ptr == nullptr || deleter == nullptr)
            return true;  // no-op は成功扱い

        std::lock_guard<std::mutex> lock(mtx_);
        if (size_ >= kMaxQuarantinedEntries)
        {
            // ★ 監査指摘 (work88): capacity exhaustion を overflowCount_ に記録（これまで
            //   未インクリメントで telemetry が 0 のままだった）。EBR 破綻の診断・HealthEvent
            //   昇格の根拠になる。deleter は絶対に実行しない（UAF 構造的排除）。
            ++overflowCount_;
            return false;  // store full — caller must NOT delete
        }

        entries_[size_] = QuarantinedEntry{
            ptr, deleter, epoch, type,
            publicationSequenceId, generation,
            reason, convo::getCurrentTimeUs()
        };
        ++size_;
        return true;
    }

    // 定期 drain（Timer/CoordinatorLoop の tryReclaim 直後）: epoch < minReaderEpoch に
    // 達したエントリのみ deleter 実行。それ以外は保持継続（RT 離脱待ち — EBR 原則）。
    // isOlderFn: EpochDomain::isOlder(a, b)（static_cast<int64_t>(a-b)<0、wraparound 対応）を委譲。
    void drain(uint64_t minReaderEpoch,
               const std::function<bool(uint64_t, uint64_t)>& isOlderFn) noexcept
    {
        // ★ work88 監査指摘: deleter を mutex 保持中に呼ばない（三次レビュー契約）。
        //   lock 内で safe エントリを抽出 → unlock 後に deleter 実行。
        //   deleter が再entrant（別の quarantine/retire を呼ぶ）でもデッドロックしない。
        void* pendingPtrs[kMaxQuarantinedEntries]{};
        void (*pendingDeleters[kMaxQuarantinedEntries])(void*) = {};
        std::size_t pendingCount = 0;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            std::size_t w = 0;
            for (std::size_t r = 0; r < size_; ++r) {
                auto& e = entries_[r];
                if (e.ptr != nullptr && e.deleter != nullptr
                    && isOlderFn(e.epoch, minReaderEpoch))
                {
                    // epoch 安全到達後のみ deleter 対象として抽出（EBR 安全削除）
                    pendingPtrs[pendingCount] = e.ptr;
                    pendingDeleters[pendingCount] = e.deleter;
                    ++pendingCount;
                    e = QuarantinedEntry{};
                } else {
                    if (w != r)
                        entries_[w] = e;
                    ++w;
                }
            }
            size_ = w;
        }
        // unlock 後に deleter 実行（reentrancy / deadlock 回避）
        for (std::size_t i = 0; i < pendingCount; ++i)
            pendingDeleters[i](pendingPtrs[i]);
    }

    // Shutdown 専用: 全強制解放（Audio Thread 停止後 — destroyForShutdown と同契約）
    void drainAllUnsafe() noexcept
    {
        void* pendingPtrs[kMaxQuarantinedEntries]{};
        void (*pendingDeleters[kMaxQuarantinedEntries])(void*) = {};
        std::size_t pendingCount = 0;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for (std::size_t i = 0; i < size_; ++i) {
                auto& e = entries_[i];
                if (e.ptr != nullptr && e.deleter != nullptr) {
                    pendingPtrs[pendingCount] = e.ptr;
                    pendingDeleters[pendingCount] = e.deleter;
                    ++pendingCount;
                }
                e = QuarantinedEntry{};
            }
            size_ = 0;
        }
        for (std::size_t i = 0; i < pendingCount; ++i)
            pendingDeleters[i](pendingPtrs[i]);
    }

    // 滞留件数（backpressure テレメトリ / high watermark 監視用）
    [[nodiscard]] std::size_t residentCount() const noexcept
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return size_;
    }

    // store full 到達（quarantine 拒否）を検出した場合の異常カウンタ
    [[nodiscard]] std::uint64_t overflowCount() const noexcept
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return overflowCount_;
    }

private:
    mutable std::mutex mtx_;
    // ★ 三次レビュー: std::vector は noexec 保証下で allocation を引き起こすため
    //   std::array + 固定 capacity（allocation ゼロ）。push_back は行わず index で配置。
    std::array<QuarantinedEntry, kMaxQuarantinedEntries> entries_{};
    std::size_t size_ = 0;
    std::uint64_t overflowCount_ = 0;  // store full で quarantine() が拒否した回数（診断用）
};

} // namespace isr
} // namespace convo
