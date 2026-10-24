#pragma once

#include <cstdint>

//==============================================================================
// IReaderEpochProvider.h — Reader management + Epoch query abstract interface.
//
// [work21 Phase-D] Separated from IEpochProvider to prevent RCUReader from
// accessing retire/publish operations. This is the minimal interface needed
// by reader-side consumers (RCUReader, ObservedRuntime).
//
// Consumers in src/core/ depend only on this interface, not on EpochDomain
// or ISRRetireRouter directly.
//==============================================================================

namespace convo {

class IReaderEpochProvider
{
public:
    virtual ~IReaderEpochProvider() = default;

    // ── Reader slot management ──

    /** Allocate a new reader slot. Returns slot index or -1 on failure. */
    virtual int registerReaderThread() noexcept = 0;

    /** Try to reserve a specific reader slot. Returns true if successful.
     *
     *  V-11: this path refuses once reader registration has been closed, which is
     *  correct for a *new* reader. A thread that already owns a slot must use
     *  reacquireReaderSlot() instead; otherwise it loses EBR protection and can
     *  observe a reclaimed object.
     */
    virtual bool reserveReaderThread(int readerIndex) noexcept = 0;

    /** V-11: re-acquire a reader slot that the calling thread already owns.
     *
     *  Contract:
     *   - succeeds only when readerIndex is a slot whose recorded owner is the
     *     calling thread (ownership is checked by the provider, not the caller);
     *   - succeeds after closeReaderRegistration(), i.e. it restores protection for
     *     an already registered reader without admitting a new one;
     *   - never creates a new registration (registrationClosed_ still governs
     *     registerReaderThread() and reserveReaderThread());
     *   - does not clear quarantine state; a quarantined-idle slot that is
     *     re-acquired must still be counted while its depth > 0 (see
     *     EpochDomain::getMinReaderEpoch).
     *
     *  Default implementation refuses, so providers that do not track slot
     *  ownership remain fail-closed.
     */
    virtual bool reacquireReaderSlot(int /*readerIndex*/) noexcept { return false; }

    /** Enter reader protection for the given slot. Audio-thread safe. */
    virtual void enterReader(int readerIndex) noexcept = 0;

    /** Exit reader protection for the given slot. Audio-thread safe. */
    virtual void exitReader(int readerIndex) noexcept = 0;

    // ── Epoch queries ──

    /** Get the current global epoch. */
    virtual uint64_t currentEpoch() const noexcept = 0;

    /** Count of active (non-zero depth) reader slots. */
    virtual uint32_t activeReaderCount() const noexcept = 0;

    /** Maximum number of reader slots available. */
    virtual int readerCapacity() const noexcept = 0;

    /** Minimum epoch among all active readers. Used for safe-reclaim determination. */
    virtual uint64_t getMinReaderEpoch() const noexcept = 0;
};

} // namespace convo
