#pragma once

#include <cstdint>

#include "../DeferredDeletionQueue.h"   // ★ T1: DeletionEntryType（telemetry 識別 metadata・D86）

//==============================================================================
// IRetireProvider.h — Retire operations abstract interface.
//
// [work21 Phase-D] Separated from IEpochProvider so that retire-aware
// consumers (SnapshotCoordinator, ISRRuntimePublicationCoordinator) depend
// on a minimal retire interface, while reader-only consumers use
// IReaderEpochProvider.
//
// This interface LIVES in src/core/ and is IMPLEMENTED by ISRRetireRouter
// (src/audioengine/), avoiding a core→audioengine dependency direction.
//==============================================================================

namespace convo {

class IRetireProvider
{
public:
    virtual ~IRetireProvider() = default;

    // ── Retire operations ──

    /** Enqueue a retire request. Returns true on success. */
    virtual bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept = 0;

    // ★ T1 (D86): telemetry type tag 付き enqueue（type==World で world retirement を識別）。
    //   デフォルト実装は type を無視して既存 3 引数版を呼ぶ（既存実装クラスは変更不要）。
    //   EpochDomain がオーバーライドして DeferredDeletionQueue に type を伝搬する。
    //   DeletionEntryType::World は telemetry 識別 metadata のみ（D86 非交渉条件 1・lifetime authority にしない）。
    virtual bool enqueueRetireTyped(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                    DeletionEntryType /*type*/) noexcept
    {
        return enqueueRetire(ptr, deleter, epoch);
    }

    /** Try to reclaim retired objects. */
    virtual void tryReclaim() noexcept = 0;

    // ★ P0-A: EpochDomain 固有メソッドをインターフェースに昇格
    //   ISRRetireRouter が dynamic_cast でアクセスするより Interface で宣言する方が型安全。
    /** Return approximate count of pending retire entries. */
    virtual uint32_t pendingRetireCount() const noexcept = 0;

    /** Drain all pending retire entries (unsafe; shutdown only). */
    virtual void drainAll() noexcept = 0;

    // ★ work70: 退役キュー滞留バイト数（診断用概算）。既定値 0。
    //   pendingRetireCount() だけでは「100個=100KB か 1GB か」が不明。
    [[nodiscard]] virtual uint64_t pendingRetireBytes() const noexcept { return 0; }

    // ★ T1 (D86): type==World の terminal deleter 実行数（world 物理破棄数・release observation の一次情報源）。
    //   デフォルト実装 0（EpochDomain がオーバーライド）。
    [[nodiscard]] virtual uint64_t worldReclaimCount() const noexcept { return 0; }
};

} // namespace convo
