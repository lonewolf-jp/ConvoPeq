#pragma once

#include "IReaderEpochProvider.h"
#include "IRetireProvider.h"
#include "IPublicationProvider.h"

//==============================================================================
// IEpochProvider.h — Combined EBR (Epoch-Based Reclamation) abstract interface.
//
// [work21 Phase-D] Inherits from IReaderEpochProvider (reader mgmt + epoch
// queries, 7 methods), IPublicationProvider (publishEpoch, 1 method), and
// IRetireProvider (retire operations, 2 methods).
//==============================================================================

namespace convo {

// ★ B-1: Reader Slot 詳細情報
struct ReaderSlotDetail {
    uint64_t epoch{0};
    uint32_t depth{0};
    uint64_t residencyTimeUs{0};
    bool active{false};
};

// ★ Practical-1/6/8: Reader Stuck 診断情報
// [work37 Phase 2.1] isChronic 追加（residency > 30秒）
// [work37 Phase 9.42] ownerTag/ownerThreadId 追加（RetireBlockerSnapshot）
struct StuckReaderInfo {
    int readerIndex{-1};
    uint64_t readerEpoch{0};
    uint64_t enterCount{0};
    uint64_t currentEpoch{0};
    uint64_t minReaderEpoch{0};
    uint32_t pendingRetireCount{0};
    bool isStuck{false};
    bool isChronic{false};           // ★ work37: 30秒超の慢性滞留
    uint64_t residencyTimeUs{0}; // ★ Practical-8: 実時間ベース滞留時間
    char ownerTag[32]{};         // ★ work37 9.42: Reader 所有者タグ（"AudioThread"等）
    uint64_t ownerThreadId{0};   // ★ work37 9.42: std::thread::id ハッシュ
};

class IEpochProvider : public IReaderEpochProvider,
                       public IPublicationProvider,
                       public IRetireProvider
{
public:
    ~IEpochProvider() override = default;

    // ★ B-1: Reader Slot 詳細取得（virtual hook）
    //   Default: 非アクティブ情報を返す。
    //   EpochDomain がオーバーライドして実情報を提供。
    [[nodiscard]] virtual ReaderSlotDetail getReaderSlotDetail(int /*readerIndex*/) const noexcept
    {
        return ReaderSlotDetail{};
    }

    // ★ Practical-1: Reader Stuck 検出（virtual hook）
    //   Default: 非stuck 空情報を返す。
    //   EpochDomain がオーバーライドして実診断を提供。
    [[nodiscard]] virtual StuckReaderInfo detectStuckReaders(uint64_t /*stuckThreshold*/) const noexcept
    {
        return StuckReaderInfo{};
    }

    // ★ A-2: EBR Queue Visibility 統計（virtual hook）
    //   Default: 0 を返す。EpochDomain がオーバーライドして実統計を提供。
    [[nodiscard]] virtual uint64_t reclaimAttemptCount() const noexcept { return 0; }
    [[nodiscard]] virtual uint64_t reclaimSuccessCount() const noexcept { return 0; }
};

} // namespace convo
