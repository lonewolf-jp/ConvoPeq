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
struct StuckReaderInfo {
    int readerIndex{-1};
    uint64_t readerEpoch{0};
    uint64_t enterCount{0};
    uint64_t currentEpoch{0};
    uint64_t minReaderEpoch{0};
    uint32_t pendingRetireCount{0};
    bool isStuck{false};
    uint64_t residencyTimeUs{0}; // ★ Practical-8: 実時間ベース滞留時間
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
};

} // namespace convo
