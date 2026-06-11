#pragma once

#include <cstdint>

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

    /** Try to reclaim retired objects. */
    virtual void tryReclaim() noexcept = 0;

    // ★ P0-A: EpochDomain 固有メソッドをインターフェースに昇格
    //   ISRRetireRouter が dynamic_cast でアクセスするより Interface で宣言する方が型安全。
    /** Return approximate count of pending retire entries. */
    virtual uint32_t pendingRetireCount() const noexcept = 0;

    /** Drain all pending retire entries (unsafe; shutdown only). */
    virtual void drainAll() noexcept = 0;
};

} // namespace convo
