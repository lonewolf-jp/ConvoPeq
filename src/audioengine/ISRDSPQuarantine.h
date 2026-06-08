#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <atomic>
#include <optional>

namespace convo::isr {

// ★ A-1.1: QuarantineReason — 隔離理由の列挙
enum class QuarantineReason {
    GenerationMismatch,
    ResolveFailure,
    PublishViolation,
    CrossfadeViolation,
    ShutdownViolation,
    RetireDeferralTimeout,
    Unknown
};

// ★ A-1.1: QuarantineEntry — 隔離エントリの構造化情報
struct QuarantineEntry {
    uint32_t slot;
    uint64_t generation;
    QuarantineReason reason;
    uint64_t quarantineEpoch;
    uint64_t quarantineTimestampUs;
    uint32_t detailCode;
    bool reclaimAllowed;
};

class DSPQuarantineManager {
public:
    explicit DSPQuarantineManager(std::size_t maxSlots = 256);

    // A-1.2: 隔離 — slot + generation(uint64_t) + reason を記録
    // returns true if quarantine was actually applied (false if already quarantined)
    bool quarantineHandle(uint32_t slot, uint64_t generation,
                          QuarantineReason reason);

    // A-1.2: slot解放（隔離解除）— generation 一致確認付き
    void reclaimSlot(uint32_t slot, uint64_t generation);

    // A-1.2: 隔離エントリ情報取得
    std::optional<QuarantineEntry> getEntry(uint32_t slot) const;

    // A-1.2: 全隔離エントリ数（監査項目）
    size_t residentCount() const noexcept;

    // A-1.2: 最長 quarantine 経過時間（秒）
    uint64_t getMaxEntryAgeSec() const noexcept;

    // A-1.4: shutdown専用解放
    bool destroyForShutdown(uint32_t slot);

private:
    static constexpr size_t kMaxSlots = 256;

    // RT側: 隔離中フラグ bitset（atomic read only）
    std::array<std::atomic<bool>, kMaxSlots> quarantineActiveFlags_{};

    // NonRT側: 監査記録ベクタ（追記専用）
    struct Entry {
        uint64_t timestampUs;
        uint64_t generation;
        QuarantineReason reason;
        uint32_t slot;
        bool resolved;  // true=隔離解除済み
    };
    std::vector<Entry> auditLog_;

    // compaction helper
    void compactAuditLog() noexcept;
};

} // namespace convo::isr
