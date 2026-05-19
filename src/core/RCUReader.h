#pragma once

#include "EpochDomain.h"
#include <atomic>
#include <functional>
#include <thread>

#include "audioengine/AtomicAccess.h"

namespace convo {

/**
 * Epoch-based RCU Reader guard.
 * Usage:
 *   convo::RCUReader reader(epochDomain);
 *   convo::RCUReaderGuard guard(reader);
 *   // ... safe to read atomic pointers ...
 */
class RCUReader
{
public:
    explicit RCUReader(EpochDomain& domain) noexcept
        : epochDomain(&domain)
    {
    }
    RCUReader(const RCUReader&) = delete;
    RCUReader& operator=(const RCUReader&) = delete;
    RCUReader(RCUReader&&) = delete;
    RCUReader& operator=(RCUReader&&) = delete;

    void enter() noexcept
    {
        // acq_rel: acquire → 直前の exit() の nestingDepth release を観測してネストを安全化；
        //          release → depth > 0 を公開し、ネスト中の enter が早期 return できる。
        const uint32_t previousDepth = convo::fetchAddAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
        if (previousDepth > 0)
        {
            return;
        }

        const uint64_t threadToken = currentThreadToken();
        uint64_t expectedOwner = 0;
        // CAS acq_rel/acquire: 成功時 acq_rel → ownerThreadToken を取得し新オーナーを公開；
        //                     失敗時 acquire → 競合スレッドの最新 ownerThreadToken を観測。
        if (!convo::compareExchangeAtomic(ownerThreadToken,
                                          expectedOwner,
                                          threadToken,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire)
            && expectedOwner != threadToken)
        {
            convo::fetchSubAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
            return;
        }

        const int tid = acquireThreadSlot();
        if (tid >= 0)
            domain().enterReader(tid);
        else
        {
            // acq_rel: スロット取得失敗時の nestingDepth 減算 — acq_rel で安全に公開。
            convo::fetchSubAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
            uint64_t expectedOwnerOnRelease = threadToken;
            // CAS acq_rel/acquire: ownerThreadToken を 0 に戻す；失敗時 acquire で最新値観測。
            convo::compareExchangeAtomic(ownerThreadToken,
                                         expectedOwnerOnRelease,
                                         static_cast<uint64_t>(0),
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire);
        }
    }

    void exit() noexcept
    {
        // acq_rel: acquire → 直前の enter() release を観測；release → depth展開後値を公開。
        const uint32_t previousDepth = convo::fetchSubAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
        if (previousDepth == 0)
        {
            // underflowガード: 0 を release で再公開し、展開 depth を修正。
            convo::publishAtomic(nestingDepth, 0, std::memory_order_release);
            return;
        }

        if (previousDepth > 1)
            return;

        // acquire: enter() の ownerThreadToken acq_rel と HB し、所有者を観測。
        if (convo::consumeAtomic(ownerThreadToken, std::memory_order_acquire) != currentThreadToken())
            return;

        // acq_rel: activeThreadId を原子的に取得（+クリア）；acquire で acquireThreadSlot の release と HB。
        const int tid = convo::exchangeAtomic(activeThreadId, -1, std::memory_order_acq_rel);
        if (tid >= 0)
        {
            domain().exitReader(tid);
            // release: 次回再利用のため preferredThreadId を公開；
            //          acquireThreadSlot() の acquire と HB してキャッシュを渡す。
            convo::publishAtomic(preferredThreadId, tid, std::memory_order_release);
        }
        // release: ownerThreadToken を 0 に戻し、次の enter() の CAS acquire と HB 。
        convo::publishAtomic(ownerThreadToken, static_cast<uint64_t>(0), std::memory_order_release);
    }

private:
    static uint64_t currentThreadToken() noexcept
    {
        return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }

    int acquireThreadSlot() noexcept
    {
        // acquire: exit() の activeThreadId acq_rel と HB し、先行割り当て済みスロットを観測。
        const int activeTid = convo::consumeAtomic(activeThreadId, std::memory_order_acquire);
        if (activeTid >= 0)
            return activeTid;

        auto& manager = domain();
        // acquire: exit() の preferredThreadId release と HB し、前回キャッシュ済み tid を観測。
        const int preferredTid = convo::consumeAtomic(preferredThreadId, std::memory_order_acquire);
        int reservedTid = -1;
        if (preferredTid >= 0 && manager.reserveReaderThread(preferredTid))
        {
            reservedTid = preferredTid;
        }
        else
        {
            reservedTid = manager.registerReaderThread();
        }

        // release: 新たに割り当てた activeThreadId を公開；
        //          exit() の exchangeAtomic acquire と HB してクリア時に観測される。
        convo::publishAtomic(activeThreadId, reservedTid, std::memory_order_release);
        return reservedTid;
    }

    std::atomic<int> preferredThreadId { -1 };
    std::atomic<int> activeThreadId { -1 };
    std::atomic<uint32_t> nestingDepth { 0 };
    std::atomic<uint64_t> ownerThreadToken { 0 };
    EpochDomain* epochDomain = nullptr;

    EpochDomain& domain() noexcept
    {
        jassert(epochDomain != nullptr);
        return *epochDomain;
    }
};

class RCUReaderGuard
{
public:
    explicit RCUReaderGuard(RCUReader& r) : reader(r) { reader.enter(); }
    ~RCUReaderGuard() { reader.exit(); }

    RCUReaderGuard(const RCUReaderGuard&) = delete;
    RCUReaderGuard& operator=(const RCUReaderGuard&) = delete;

private:
    RCUReader& reader;
};

} // namespace convo
