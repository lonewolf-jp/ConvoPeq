#pragma once
#include <cstdint>

namespace convo::isr {

// RuntimeDrainAudit: Shutdown 完了条件の監査構造体。
// isAllZero() は監査ログ出力専用。shutdown 完了判定の authority にはしない。
//
// ■ 完了条件に含めるもの
//   pendingPublication  — RuntimePublicationCoordinator の publication backlog
//   pendingRetire       — RetireRuntime に未処理の retire intent
//   activeCrossfadeCount — 進行中のクロスフェード（0 or 1）
//   deferredPublish     — 未投入の deferred publish
//   routerPendingRetire — ISRRetireRouter 滞留 item 数
//   maxDeferredAgeMs    — deferred publish 最長滞留時間
//
// ■ 監査のみ（完了条件にしない）
//   quarantineResident  — 隔離保留中のエントリ数
//   oldestPendingAgeMs  — 最長滞留時間
//   maxQuarantineAgeSec — 最長 quarantine 経過時間（秒）
struct RuntimeDrainAudit {
    uint64_t pendingPublication;
    uint64_t pendingRetire;
    uint64_t activeCrossfadeCount;
    uint64_t routerPendingRetire;
    uint64_t maxDeferredAgeMs;
    uint64_t deferredPublish;
    uint64_t quarantineResident;    // 監査のみ
    uint64_t oldestPendingAgeMs;    // 監査のみ
    uint64_t maxQuarantineAgeSec;   // 監査のみ
    // ★ C-1: WorldLifecycleAudit 連携（診断目的）
    uint64_t activeWorldCount{0};
    uint64_t publishedCount{0};
    uint64_t retiredCount{0};

    // shutdown 完了を阻害している主要因を特定
    enum class BlockingReason : uint8_t {
        None,
        PendingPublication,
        PendingRetire,
        ActiveCrossfade,
        DeferredPublish,
        QuarantineResident,
        RouterPendingRetire,
        Unknown
    };

    [[nodiscard]] BlockingReason getPrimaryBlockingReason() const noexcept {
        if (pendingPublication > 0)    return BlockingReason::PendingPublication;
        if (pendingRetire > 0)         return BlockingReason::PendingRetire;
        if (activeCrossfadeCount > 0)  return BlockingReason::ActiveCrossfade;
        if (deferredPublish > 0)       return BlockingReason::DeferredPublish;
        if (quarantineResident > 0)    return BlockingReason::QuarantineResident;
        if (routerPendingRetire > 0)   return BlockingReason::RouterPendingRetire;
        return BlockingReason::Unknown;
    }

    // ★ 監査ログ出力専用。shutdown 完了判定には使用しない。
    bool isAllZero() const noexcept {
        return pendingPublication == 0
            && pendingRetire == 0
            && activeCrossfadeCount == 0
            && deferredPublish == 0;
    }
};

} // namespace convo::isr
