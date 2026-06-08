#include "ISRDSPQuarantine.h"
#include "AtomicAccess.h"
#include <algorithm>
#include <chrono>

namespace convo::isr {

DSPQuarantineManager::DSPQuarantineManager(std::size_t maxSlots)
{
    // kMaxSlots 固定。引数 maxSlots は互換性のため維持
    for (auto& flag : quarantineActiveFlags_) {
        convo::publishAtomic(flag, false, std::memory_order_relaxed);
    }
    auditLog_.reserve(maxSlots * 2);
}

bool DSPQuarantineManager::quarantineHandle(uint32_t slot, uint64_t generation,
                                              QuarantineReason reason)
{
    if (slot >= kMaxSlots)
        return false;

    // ★ 二重加算防止: 既存エントリの有無を確認
    bool alreadyActive = convo::consumeAtomic(
        quarantineActiveFlags_[slot], std::memory_order_acquire);
    if (alreadyActive)
        return false;  // 既に隔離済み

    // RT側: active フラグ設定
    convo::publishAtomic(quarantineActiveFlags_[slot], true, std::memory_order_release);

    // NonRT側: 監査記録
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    auditLog_.push_back(Entry{
        .timestampUs = static_cast<uint64_t>(now),
        .generation = generation,
        .reason = reason,
        .slot = slot,
        .resolved = false
    });
    return true;
}

void DSPQuarantineManager::reclaimSlot(uint32_t slot, uint64_t generation)
{
    if (slot >= kMaxSlots)
        return;

    // ★ generation 一致確認: 異なる場合は削除しない
    bool found = false;
    for (auto& entry : auditLog_) {
        if (entry.slot == slot && !entry.resolved) {
            if (entry.generation != generation) {
                // generation が異なる → 新しい隔離情報を誤って消さない
                return;
            }
            entry.resolved = true;
            found = true;
            break;
        }
    }
    if (!found)
        return;

    // RT側: active フラグ解除
    convo::publishAtomic(quarantineActiveFlags_[slot], false, std::memory_order_release);

    compactAuditLog();
}

std::optional<QuarantineEntry> DSPQuarantineManager::getEntry(uint32_t slot) const
{
    // ★ 最新の未解決エントリを検索（追記専用 vector の末尾からスキャン）
    for (auto it = auditLog_.rbegin(); it != auditLog_.rend(); ++it) {
        if (it->slot == slot && !it->resolved) {
            return QuarantineEntry{
                .slot = slot,
                .generation = it->generation,
                .reason = it->reason,
                .quarantineEpoch = it->timestampUs,
                .quarantineTimestampUs = it->timestampUs,
                .detailCode = 0,
                .reclaimAllowed = false
            };
        }
    }
    return std::nullopt;
}

size_t DSPQuarantineManager::residentCount() const noexcept
{
    size_t count = 0;
    for (const auto& flag : quarantineActiveFlags_) {
        if (convo::consumeAtomic(flag, std::memory_order_acquire))
            ++count;
    }
    return count;
}

uint64_t DSPQuarantineManager::getMaxEntryAgeSec() const noexcept
{
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    uint64_t maxAge = 0;
    for (const auto& entry : auditLog_) {
        if (!entry.resolved && now > static_cast<int64_t>(entry.timestampUs)) {
            uint64_t ageSec = static_cast<uint64_t>(
                (now - static_cast<int64_t>(entry.timestampUs)) / 1'000'000);
            maxAge = std::max(maxAge, ageSec);
        }
    }
    return maxAge;
}

bool DSPQuarantineManager::destroyForShutdown(uint32_t slot)
{
    if (slot >= kMaxSlots)
        return false;

    bool active = convo::consumeAtomic(
        quarantineActiveFlags_[slot], std::memory_order_acquire);
    if (!active)
        return false;

    // RT側: フラグ解除
    convo::publishAtomic(quarantineActiveFlags_[slot], false, std::memory_order_release);

    // NonRT側: 未解決エントリを resolved に
    for (auto& entry : auditLog_) {
        if (entry.slot == slot && !entry.resolved) {
            entry.resolved = true;
            break;
        }
    }

    compactAuditLog();
    return true;
}

void DSPQuarantineManager::compactAuditLog() noexcept
{
    // ★ compaction: resolved エントリが一定数超えた場合のみ
    constexpr size_t kCompactThreshold = 1024;
    if (auditLog_.size() < kCompactThreshold)
        return;

    // 先頭から resolved エントリを削除
    auto it = auditLog_.begin();
    while (it != auditLog_.end() && it->resolved) {
        ++it;
    }
    if (it != auditLog_.begin())
        auditLog_.erase(auditLog_.begin(), it);
}

} // namespace convo::isr
