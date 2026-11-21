#pragma once

#include "IReaderEpochProvider.h"
#include "ReaderMeasureRegistry.h"
#include <atomic>
#include <functional>
#include <thread>

#include "audioengine/AtomicAccess.h"
#include "ThreadHash.h"

namespace convo {

/**
 * Epoch-based RCU Reader guard.
 * Usage:
 *   convo::RCUReader reader(epochProvider);
 *   convo::RCUReaderGuard guard(reader);
 *   // ... safe to read atomic pointers ...
 *
 * [work21 P0-14] RCUReader now depends on IReaderEpochProvider (abstract interface)
 * instead of EpochDomain directly. This eliminates EpochDomain type exposure
 * from the public API. The provider is typically an ISRRetireRouter or EpochDomain.
 */
class RCUReader
{
public:
    explicit RCUReader(IReaderEpochProvider& provider) noexcept
        : epochProvider(&provider)
    {
    }
    RCUReader(const RCUReader&) = delete;
    RCUReader& operator=(const RCUReader&) = delete;
    RCUReader(RCUReader&&) = delete;
    RCUReader& operator=(RCUReader&&) = delete;

    // ★ Gate 1 (2026-11-11): 戻り値 = 「この呼出しが保護を確立したか」。
    //   rootEnterSucceeded_ は所有者スレッドだけが書き、thread-confined を維持する。
    //   他スレッドの入場は nested と誤認せず、状態を一切変更せずに false を返す。
    bool enter() noexcept
    {
        const uint64_t my = currentThreadToken();

        // 1) 所有者が既に居る場合: 同一スレッドなら nested、他スレッドなら何も変更せず失敗
        const uint64_t owner = convo::consumeAtomic(ownerThreadToken, std::memory_order_acquire);
        if (owner != 0)
        {
            if (owner != my)
                return false;                       // 他スレッド所有（depth も owner も触らない）

            convo::fetchAddAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
            // nested は外側の成否を返す（同一スレッドが書いた値を読む）
            return convo::consumeAtomic(rootEnterSucceeded_, std::memory_order_acquire);
        }

        // 2) 所有者が空き: CAS で所有権を先に確保する（ここが線形化点）
        uint64_t expectedOwner = 0;
        if (!convo::compareExchangeAtomic(ownerThreadToken,
                                          expectedOwner,
                                          my,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire))
        {
            if (expectedOwner != my)
                return false;                       // 競合に負けた（状態は変更しない）

            convo::fetchAddAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
            return convo::consumeAtomic(rootEnterSucceeded_, std::memory_order_acquire);
        }

        // 3) 所有権を得た: depth 0→1
        convo::fetchAddAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);

        const int tid = acquireThreadSlot();
        if (tid >= 0)
        {
            epochProvider->enterReader(tid);
            convo::publishAtomic(rootEnterSucceeded_, true, std::memory_order_release);
            measureIdx_ = convo::ReaderMeasureRegistry::claim ("RCUReader",
                                                               static_cast<std::uint32_t> (tid), 1u,
                                                               this, epochProvider);
            return true;
        }

        // 4) slot 取得失敗: owner と depth をロールバック
        convo::publishAtomic(rootEnterSucceeded_, false, std::memory_order_release);
        convo::ReaderMeasureRegistry::enterFailures_.fetch_add (1u, std::memory_order_relaxed);
        convo::fetchSubAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);
        uint64_t expectedOwnerOnRelease = my;
        convo::compareExchangeAtomic(ownerThreadToken,
                                     expectedOwnerOnRelease,
                                     static_cast<uint64_t>(0),
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire);
        return false;
    }

    // === 限定公開API (P0-14: EpochDomain型を露出しない) ===

    /** 現在のepoch番号を取得 (Router移行後も維持) */
    uint64_t snapshotEpoch() noexcept
    {
        return epochProvider->currentEpoch();
    }

    /** アクティブなReader数を診断用に取得。
     *  診断の件数なので depth のみを返す。静止判定に使ってはならない。
     *
     *  ★ WORK113 R1: 静止の完全な条件は depth 0 だけではない。予約してまだ enter していない
     *  reader は depth 0 でこの値に入らないが、後から入場できる。完全な静止を主張する側は、
     *  次の順で確かめる。
     *    1. 保護された操作で新規 admission を排除する（exclusion）
     *    2. 実行中の admission が 0（readerAdmissionsActive() == 0）
     *    3. 予約済みで未 enter の slot が 0（admissionPendingSlots() == 0）
     *    4. 活動数が 0（epochProvider->activeReaderCount() == 0）
     *  1 を拜えても 2〜4 を読むまでに入場し得るため、2〜4 は 1 の後で読む。
     *  実装は AudioEngine::readerActivityZeroNonRt()。 */
    uint32_t activeReaders() noexcept
    {
        return epochProvider->activeDepthReaderCount();
    }

    /** ★ P1-A: 今回の enter() が成功し epoch 保護下有効かを返す */
    [[nodiscard]] bool rootEnterSucceeded() const noexcept
    {
        return convo::consumeAtomic(rootEnterSucceeded_, std::memory_order_acquire);
    }

    /** ★ WORK113 (2026-11-18 Owner 指示): 保護区間内にいるか（nestingDepth > 0）。
     *  shutdown が engine 所有 reader の使用中を観測するために使う（読み取りのみ）。 */
    [[nodiscard]] bool isInside() const noexcept
    {
        return convo::consumeAtomic(nestingDepth, std::memory_order_acquire) != 0;
    }

    void exit() noexcept
    {
        // acq_rel: acquire → 直前の enter() release を観測；release → depth展開後値を公開。
        const uint32_t previousDepth = convo::fetchSubAtomic(nestingDepth, static_cast<uint32_t>(1), std::memory_order_acq_rel);

        // ★ P1-A v7.3: ネスト対応リセット
        //   previousDepth==0: underflow → 安全のためリセット
        if (previousDepth == 0)
        {
            convo::publishAtomic(rootEnterSucceeded_, false, std::memory_order_release);
            convo::ReaderMeasureRegistry::underflows_.fetch_add (1u, std::memory_order_relaxed);   // ★ 計測
            // underflowガード: 0 を release で再公開し、展開 depth を修正。
            convo::publishAtomic(nestingDepth, 0, std::memory_order_release);
            return;
        }

        // ★ previousDepth > 1: ネスト → rootEnterSucceeded_ を維持
        //   outer scope がまだ active なため false にしてはいけない。
        if (previousDepth > 1)
            return;

        // ★ previousDepth == 1: 最外層 exit
        // acquire: enter() の ownerThreadToken acq_rel と HB し、所有者を観測。
        // ★ 計測 (2026-11-11): 最外層の退出で区画を閉じる（ネストでは閉じない）
        if (measureIdx_ >= 0)
        {
            convo::ReaderMeasureRegistry::close (measureIdx_);
            measureIdx_ = -1;
        }
        if (convo::consumeAtomic(ownerThreadToken, std::memory_order_acquire) != currentThreadToken())
        {
            convo::publishAtomic(rootEnterSucceeded_, false, std::memory_order_release);  // 所有者不一致でもリセット
            return;
        }

        // ★ C-01: 成否フラグは所有者解放より「先」に落とす。逆順だと、解放直後に
        //   別スレッドの enter() が true を書いた後で旧所有者の false が上書きし、
        //   新しい所有者の nested 判定が失敗する（fail-closed で音声が素通しになる）。
        convo::publishAtomic(rootEnterSucceeded_, false, std::memory_order_release);

        // acq_rel: activeThreadId を原子的に取得（+クリア）；acquire で acquireThreadSlot の release と HB。
        const int tid = convo::exchangeAtomic(activeThreadId, -1, std::memory_order_acq_rel);
        if (tid >= 0)
        {
            epochProvider->exitReader(tid);
            // release: 次回再利用のため preferredThreadId を公開；
            //          acquireThreadSlot() の acquire と HB してキャッシュを渡す。
            convo::publishAtomic(preferredThreadId, tid, std::memory_order_release);
        }
        // release: ownerThreadToken を 0 に戻し、次の enter() の CAS acquire と HB 。
        convo::publishAtomic(ownerThreadToken, static_cast<uint64_t>(0), std::memory_order_release);
    }

private:
    // ★ work92 B-4 (big 2-6): token 生成を単調 ID 採番に統一。
    //   衝突時の RCU reader 二重登録 / epoch 停止（reclaim 停止）の潜在を排除。
    //   DspNumericPolicy.h の役割タグ（isAudioThread 等）は衝突許容のため
    //   cachedThreadHash() を維持する（RECONCILIATION §5-3 設計確定）。
    static uint64_t currentThreadToken() noexcept
    {
        return convo::acquireUniqueThreadId();
    }

    int acquireThreadSlot() noexcept
    {
        // acquire: exit() の activeThreadId acq_rel と HB し、先行割り当て済みスロットを観測。
        const int activeTid = convo::consumeAtomic(activeThreadId, std::memory_order_acquire);
        if (activeTid >= 0)
            return activeTid;

        // acquire: exit() の preferredThreadId release と HB し、前回キャッシュ済み tid を観測。
        const int preferredTid = convo::consumeAtomic(preferredThreadId, std::memory_order_acquire);
        int reservedTid = -1;
        if (preferredTid >= 0 && epochProvider->reserveReaderThread(preferredTid))
        {
            reservedTid = preferredTid;
        }
        else if (preferredTid >= 0 && epochProvider->reacquireReaderSlot(preferredTid))
        {
            // ★ V-11: registration is closed but this thread owns the slot. Re-acquire it
            //   so epoch protection is restored, instead of returning -1 and letting the
            //   caller run unprotected (or, with the 3c fail-closed guard, go permanently
            //   silent). The provider verifies ownership; a foreign thread gets false.
            reservedTid = preferredTid;
        }
        else
        {
            reservedTid = epochProvider->registerReaderThread();
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
    IReaderEpochProvider* epochProvider = nullptr;
    int measureIdx_ = -1;   // ★ 計測: ReaderMeasureRegistry のレコード番号
    // ★ P1-A / C-01: per-enter 成否追跡。所有者スレッドだけが true/false を書く設計だが、
    //   「所有者解放 → 次スレッドの enter → 旧所有者の false 書き込み」の順序が成立し得るため、
    //   フラグ自体を atomic にしてデータ競合を無くす。意味論は従来どおり thread-confined。
    std::atomic<bool> rootEnterSucceeded_ { false };
};

class RCUReaderGuard
{
public:
    // ★ Gate 1 (2026-11-11): 成否を「この呼出し」で保持する（共有フラグを読まない）。
    explicit RCUReaderGuard(RCUReader& r) noexcept : reader(&r), ok_(r.enter()) {}
    // ★ Gate 1: 入場に失敗した guard は exit を呼ばない（depth を壊さない）
    ~RCUReaderGuard() noexcept { if (reader && ok_) reader->exit(); }

    RCUReaderGuard(const RCUReaderGuard&) = delete;
    RCUReaderGuard& operator=(const RCUReaderGuard&) = delete;

    RCUReaderGuard(RCUReaderGuard&& other) noexcept
        : reader(other.reader), ok_(other.ok_) { other.reader = nullptr; other.ok_ = false; }
    RCUReaderGuard& operator=(RCUReaderGuard&& other) noexcept
    {
        if (this != &other)
        {
            if (reader && ok_) reader->exit();
            reader = other.reader;
            ok_ = other.ok_;
            other.reader = nullptr;
            other.ok_ = false;
        }
        return *this;
    }

    // ★ C-7 + Gate 1: この guard の入場が保護を確立したか
    [[nodiscard]] bool rootEnterSucceeded() const noexcept
    {
        return reader != nullptr && ok_;
    }

private:
    RCUReader* reader = nullptr;
    bool ok_ = false;
};

} // namespace convo
