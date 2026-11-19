#pragma once

#include <array>
#include <cassert>
#include <atomic>
#include <cstdio>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

#include "../DeferredDeletionQueue.h"
#include "IEpochProvider.h"
#include "RetireDomain.h"          // ★ domain tag (v2.9): 退役ドメインの刻印元
#include "DrainCapability.h"         // ★ WORK113 Queue/Gate/Drain: shutdown 権限
#include "audioengine/AtomicAccess.h"
#include "ThreadHash.h"

#if ! defined (NDEBUG) && defined (_WIN32)
    #include <intrin.h>   // __debugbreak（MSVC intrinsic・x86/x64/ARM/ARM64）

    // windows.h を持ち込むと max/min マクロが std::numeric_limits<T>::max() を壊すため、
    // 必要な API だけを宣言する。Windows SDK の宣言は `WINBASEAPI BOOL WINAPI IsDebuggerPresent(VOID)`
    // であり、BOOL=int・WINAPI=__stdcall なので、型と呼出規約をそれに合わせる
    // （kernel32 は常にリンクされる）。windows.h が先に取り込まれている場合は宣言済みなので省く。
    #if ! defined (_WINDOWS_) && ! defined (_INC_WINDOWS)
        extern "C" __declspec (dllimport) int __stdcall IsDebuggerPresent (void);
    #endif
#endif

// ★ WORK113 (2026-11-18 Owner 指示): shutdown の publisher timeout 検証用テストアクセス。
//   本体は src/tests/AudioEngineHarness/V11EngineRetireProtectionTests.cpp に定義する。
//   Production ビルドでは前方宣言と friend 宣言のみで、追加メンバも評価分岐も存在しない。
class ShutdownHoldTestAccess;

namespace convo {

class EpochDomain : public IEpochProvider
{
public:
    // ★ WORK113 (2026-11-18 Owner 指示): shutdown の publisher timeout を検証するための
    //   テスト専用アクセス（publish gate の admission を取得して gate 内に停止する）。
    //   本体は src/tests/AudioEngineHarness/V11EngineRetireProtectionTests.cpp に定義する。
    //   Production ビルドでは friend 宣言のみで、追加メンバも評価分岐も存在しない。
    friend class ::ShutdownHoldTestAccess;

    static constexpr int kMaxReaders = 64;
    static constexpr uint64_t kInactiveEpoch = std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t kReservedEpoch = std::numeric_limits<uint64_t>::max() - 1;

    //==========================================================================
    // ★ WORK113 Queue/Gate/Drain 再設計（v4.13）
    //==========================================================================
    using RtThreadProbe = bool (*)() noexcept;

    enum class DrainResult
    {
        Completed,
        AlreadyConsumed,
        CapabilityInvalid,
        ReadersNotQuiescent,
        Incomplete,
        Busy
    };

    enum class QuarantineResult
    {
        Registered,
        AlreadyRegistered
    };

    class EpochPublishGate
    {
    public:
        bool tryEnter() noexcept
        {
            if (convo::consumeAtomic(closed_, std::memory_order_acquire))
                return false;
            convo::fetchAddAtomic(active_, 1, std::memory_order_acq_rel);
            if (convo::consumeAtomic(closed_, std::memory_order_acquire))
            {
                convo::fetchSubAtomic(active_, 1, std::memory_order_acq_rel);
                return false;
            }
            return true;
        }
        void exit() noexcept { convo::fetchSubAtomic(active_, 1, std::memory_order_acq_rel); }
        void close() noexcept { convo::publishAtomic(closed_, true, std::memory_order_release); }

        // 期限内の idle だけを成功として扱う（期限到達後は idle でも false）。
        bool waitForIdle(std::chrono::milliseconds timeout) noexcept
        {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            for (;;)
            {
                const bool idle = convo::consumeAtomic(active_, std::memory_order_acquire) == 0;
                const auto now = std::chrono::steady_clock::now();
                if (now >= deadline)
                    return false;
                if (idle)
                    return true;
                std::this_thread::yield();
            }
        }

        [[nodiscard]] uint32_t active() const noexcept { return convo::consumeAtomic(active_, std::memory_order_acquire); }
        [[nodiscard]] bool closed() const noexcept { return convo::consumeAtomic(closed_, std::memory_order_acquire); }

    private:
        std::atomic<bool> closed_ { false };
        std::atomic<uint32_t> active_ { 0 };
    };

    class PublishScope
    {
    public:
        explicit PublishScope(EpochPublishGate& gate) noexcept
            : gate_(gate), entered_(gate.tryEnter()) {}
        ~PublishScope() { if (entered_) gate_.exit(); }
        PublishScope(const PublishScope&) = delete;
        PublishScope& operator=(const PublishScope&) = delete;
        [[nodiscard]] bool entered() const noexcept { return entered_; }

    private:
        EpochPublishGate& gate_;
        bool entered_;
    };

    // ── queue の寿命（L1）──────────────────────────────────────────────
    [[nodiscard]] std::shared_ptr<DeferredDeletionQueue> borrow() const noexcept
    {
        std::lock_guard<std::mutex> lock(queueRefMutex_);
        return queueRef_;
    }

    [[nodiscard]] std::shared_ptr<DeferredDeletionQueue> takeForShutdown() noexcept
    {
        std::lock_guard<std::mutex> lock(queueRefMutex_);
        convo::publishAtomic(queueRt_, nullptr, std::memory_order_release);
        return std::move(queueRef_);
    }

    [[nodiscard]] bool queueAvailable() const noexcept
    {
        return convo::consumeAtomic(queueAvailable_, std::memory_order_acquire);
    }

    // ── RT 誤用の検出（Debug のみ）──────────────────────────────────
    void setRtThreadProbe(RtThreadProbe probe) noexcept { rtProbe_ = probe; }
    [[nodiscard]] std::uint64_t rtMisuseCount() const noexcept { return convo::consumeAtomic(rtMisuseCount_, std::memory_order_acquire); }

    // ── Reader fence（shutdown 専用）────────────────────────────────
    void engageShutdownReaderFence() noexcept
    {
        convo::publishAtomic(shutdownReaderFence_, true, std::memory_order_release);
        // ★ WORK113: 以降の admission を拒否し、入場処理中の reader を待てるようにする。
        readerAdmissionGate_.close();
    }

    // ★ WORK113: reader admission（予約〜enter の完了）。fence 後は新規を拒否する。
    [[nodiscard]] bool waitForReaderAdmissions(std::chrono::milliseconds timeout) noexcept
    {
        return readerAdmissionGate_.waitForIdle(timeout);
    }
    [[nodiscard]] std::uint32_t readerAdmissionsActive() const noexcept { return readerAdmissionGate_.active(); }

    // 予約済みで未 enter の slot（admission 中の slot）を数える。admission gate と二重に確認する。
    [[nodiscard]] std::uint32_t admissionPendingSlots() const noexcept
    {
        std::uint32_t n = 0;
        for (int i = 0; i < kMaxReaders; ++i)
        {
            if (convo::consumeAtomic(readers[static_cast<size_t>(i)].admissionPending,
                                     std::memory_order_acquire) != 0)
                ++n;
        }
        return n;
    }

#if defined(CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY)
    // ★ WORK113 テスト専用シーム（CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY を定義する単一 TU だけ）。
    //   reader admission の取得直後（fence 判定より前）に 1 回呼ぶ。interleaving の
    //   決定論的再現に使う。本番・harness のビルドには存在しない。
    using ReaderAdmissionSeam = void (*) (void*);
    static void setReaderAdmissionSeam (ReaderAdmissionSeam fn, void* ctx) noexcept
    {
        convo::publishAtomic(readerAdmissionSeamCtx_, ctx, std::memory_order_release);
        convo::publishAtomic(readerAdmissionSeamFn_, fn, std::memory_order_release);
    }

    // ★ WORK113 (2026-11-19 Owner 指示): exitReader() の depth デクリメント直後で
    //   呼び出しを止める seam。「depth == 0 だが exit 処理は未完了」の窓を
    //   決定論的に作り、shutdown の quiescence 判定との競合を検査する。
    using ReaderExitSeam = void (*) (void*);
    static void setReaderExitSeam (ReaderExitSeam fn, void* ctx) noexcept
    {
        convo::publishAtomic(readerExitSeamCtx_, ctx, std::memory_order_release);
        convo::publishAtomic(readerExitSeamFn_, fn, std::memory_order_release);
    }
#endif
    [[nodiscard]] bool shutdownReaderFenceEngaged() const noexcept
    {
        return convo::consumeAtomic(shutdownReaderFence_, std::memory_order_acquire);
    }

    [[nodiscard]] std::uint32_t reacquireAfterFence() const noexcept { return convo::consumeAtomic(reacquireAfterFence_, std::memory_order_acquire); }
    [[nodiscard]] std::uint32_t enterAfterFence() const noexcept { return convo::consumeAtomic(enterAfterFence_, std::memory_order_acquire); }

    // ── 停止性: close → waitForIdle → epochAtFence ───────────────────
    bool closePublishersAndWaitIdle(std::chrono::milliseconds timeout) noexcept
    {
        publishGate_.close();
        if (! publishGate_.waitForIdle(timeout))
        {
            epochIdleTimeouts_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        convo::publishAtomic(epochAtFence_, currentEpoch(), std::memory_order_release);
        return true;
    }
    [[nodiscard]] std::uint32_t epochIdleTimeouts() const noexcept { return convo::consumeAtomic(epochIdleTimeouts_, std::memory_order_acquire); }
    [[nodiscard]] std::uint32_t latePublisherAttempts() const noexcept { return convo::consumeAtomic(latePublisherAttempts_, std::memory_order_acquire); }
    [[nodiscard]] std::uint64_t epochAtFence() const noexcept { return convo::consumeAtomic(epochAtFence_, std::memory_order_acquire); }
    [[nodiscard]] std::uint64_t drainReaderFailures() const noexcept { return convo::consumeAtomic(drainReaderFailures_, std::memory_order_acquire); }
    [[nodiscard]] std::uint64_t drainRemainingAfter() const noexcept { return convo::consumeAtomic(drainRemainingAfter_, std::memory_order_acquire); }

    // ── 隔離（自己参照・固定容量上限なし・プロセス寿命依存の意図的リーク）──
    QuarantineResult quarantineQueue(std::shared_ptr<DeferredDeletionQueue>& local) noexcept
    {
        if (! local)
            return QuarantineResult::AlreadyRegistered;

        const std::uint32_t pending = local->sizeApprox();

        // 1. 一意化（勝った 1 スレッドだけが登録する）
        if (convo::exchangeAtomic(local->quarantineEnqueued_, true, std::memory_order_acq_rel))
            return QuarantineResult::AlreadyRegistered;

        // 2. 保持の確立（登録より先。3 が失敗しても保持は成立する）
        local->quarantineSelfRef_ = local;

        // 3. 診断用の侵入型リスト（append-only・公開後は next を変更しない）
        DeferredDeletionQueue* expected = convo::consumeAtomic(quarantineHead_, std::memory_order_acquire);
        do
        {
            local->quarantineNext_.store(expected, std::memory_order_relaxed);
        }
        while (! convo::compareExchangeWeakAtomic(quarantineHead_, expected, local.get(), std::memory_order_acq_rel, std::memory_order_acquire));

        // 4. 計数（move より前・atomic のみ）
        quarantineCount_.fetch_add(1, std::memory_order_relaxed);
        quarantinePendingEntries_.fetch_add(pending, std::memory_order_relaxed);
        quarantineRetainedBytes_.fetch_add(sizeof(DeferredDeletionQueue), std::memory_order_relaxed);
        return QuarantineResult::Registered;
    }

    [[nodiscard]] static std::uint64_t quarantineCount() noexcept { return convo::consumeAtomic(quarantineCount_, std::memory_order_acquire); }
    [[nodiscard]] static std::uint64_t quarantinePendingEntries() noexcept { return convo::consumeAtomic(quarantinePendingEntries_, std::memory_order_acquire); }
    [[nodiscard]] static std::uint64_t quarantineRetainedBytes() noexcept { return convo::consumeAtomic(quarantineRetainedBytes_, std::memory_order_acquire); }

    // ── 最終 drain（shutdown 専用・capability 必須）─────────────────
    DrainResult drainQueue(DeferredDeletionQueue& queue, convo::DrainCapability&& cap) noexcept
    {
        if (! cap.consume())
            return DrainResult::AlreadyConsumed;

        if (! cap.producersJoined() || ! cap.postStopEnqueueZero()
            || ! cap.readersQuiescent() || ! cap.epochSettled())
        {
            return DrainResult::CapabilityInvalid;
        }

        const bool regClosed = convo::consumeAtomic(registrationClosed_, std::memory_order_acquire);
        const bool fence = shutdownReaderFenceEngaged();
        const std::uint32_t active = activeReaderCount();
        const std::uint32_t admissions = readerAdmissionsActive();
        const std::uint32_t pendingAdmissions = admissionPendingSlots();
        const std::uint64_t minReader = getMinReaderEpoch();
        const std::uint64_t fenceEpoch = convo::consumeAtomic(epochAtFence_, std::memory_order_acquire);
        const bool epochEqual = ! isOlder(minReader, fenceEpoch) && ! isOlder(fenceEpoch, minReader);
        const bool capEpochEqual = ! isOlder(cap.epochAtIssue(), fenceEpoch)
                                && ! isOlder(fenceEpoch, cap.epochAtIssue());

        if (! regClosed || ! fence || active != 0 || admissions != 0 || pendingAdmissions != 0
            || ! epochEqual || ! capEpochEqual)
        {
            drainReaderFailures_.fetch_add(1, std::memory_order_relaxed);
            return DrainResult::ReadersNotQuiescent;
        }

        queue.drainAllUnsafe();
        const std::uint32_t remaining = queue.sizeApprox();
        drainRemainingAfter_.store(remaining, std::memory_order_relaxed);
        return remaining == 0 ? DrainResult::Completed : DrainResult::Incomplete;
    }


    // ★ domain tag (v2.9 §3.2): 生成側に必ず意図を書かせる（デフォルト引数なし）。
    //   契約は B-1:
    //     ・許可値は Unknown / EngineISR / EQ の 3 値のみ
    //     ・Unstamped / Count / 5 以上は全ビルドで「採用しない」（Unknown に落として計数する）
    //     ・Debug では加えて assertion（ログ + デバッガ接続時ブレーク）で検出する
    explicit EpochDomain(RetireDomainId domain) noexcept
        : domainId_(sanitizeDomain(domain)), globalEpoch(1)
    {
        if (! isAssignableDomain(domain))
        {
            invalidDomainAssignments_.fetch_add (1u, std::memory_order_relaxed);
            domainAssertionFailure();
        }

        for (auto& slot : readers)
        {
            // release: コンストラクタ内で単一スレッドから初期化するが、
            //          完了後に他スレッドがオブジェクトを取得する際に acquire で可視性を保証するため release。
            convo::publishAtomic(slot.epoch, kInactiveEpoch, std::memory_order_release);
            convo::publishAtomic(slot.depth, static_cast<uint32_t>(0), std::memory_order_release);
            // ★ Phase 3: quarantineFlags ゼロ初期化
            convo::publishAtomic(slot.quarantineFlags, static_cast<uint8_t>(0), std::memory_order_release);
            convo::publishAtomic(slot.admissionPending, static_cast<uint8_t>(0), std::memory_order_release);
        }

        // ★ WORK113 Queue Lifetime: queue は shared_ptr で所有する（確保失敗は fail-closed）。
        try
        {
            auto queue = std::make_shared<DeferredDeletionQueue>();
            convo::publishAtomic(queueRt_, queue.get(), std::memory_order_release);
            queueRef_ = std::move(queue);
            convo::publishAtomic(queueAvailable_, true, std::memory_order_release);
        }
        catch (...)
        {
            queueRef_.reset();
            convo::publishAtomic(queueRt_, nullptr, std::memory_order_release);
            convo::publishAtomic(queueAvailable_, false, std::memory_order_release);
            std::fprintf(stderr, "[FAULT] epoch_queue_allocation_failed\n");
        }
    }

    // ★ domain tag: 契約の判定子（constexpr・全ビルド共通）
    static constexpr bool isAssignableDomain(RetireDomainId domain) noexcept
    {
        return domain == RetireDomainId::Unknown
            || domain == RetireDomainId::EngineISR
            || domain == RetireDomainId::EQ;
    }

    // ★ domain tag: 不正値は決して採用しない（全ビルド）。採用値は Unknown
    //   （集計上は providerUnknown に計上される）。
    static constexpr RetireDomainId sanitizeDomain(RetireDomainId domain) noexcept
    {
        return isAssignableDomain(domain) ? domain : RetireDomainId::Unknown;
    }

    [[nodiscard]] RetireDomainId domainId() const noexcept { return domainId_; }

    // ★ domain tag (v2.9 §3.1): IRetireProvider への公開
    [[nodiscard]] RetireDomainId retireDomainId() const noexcept override { return domainId_; }

    // ★ domain tag: 不正値の生成回数（診断用。JSON には出さない）
    static std::uint32_t invalidDomainAssignments() noexcept
    {
        return invalidDomainAssignments_.load (std::memory_order_relaxed);
    }

    // ★ domain tag: 設計の jassertfalse に相当する Debug 検出。
    //   EpochDomain.h は JUCE 非依存（<cassert> のみ）で、inline ctor の定義は全 TU で
    //   同一でなければならない。そのため JUCE の jassertfalse 展開
    //   （JUCE_LOG_CURRENT_ASSERTION → juce_isRunningUnderDebugger() のときだけブレーク）
    //   と同じ「条件・順序・ブレーク動作」をここで直接実装する。Release では何もしない。
    //   ただし **ログの配送先は異なる**（JUCE Logger ではなく stderr）。JUCE の実マクロを
    //   製品 ctor で使っているわけではない。
    static void domainAssertionFailure() noexcept
    {
#if ! defined (NDEBUG)
        std::fprintf (stderr,
                      "EpochDomain: invalid RetireDomainId (contract violation; "
                      "assignable values are Unknown/EngineISR/EQ)\n");
  #if defined (_WIN32)
        // デバッガ接続時のみブレーク（JUCE の juce_isRunningUnderDebugger() と同じ条件）。
        if (::IsDebuggerPresent() != 0)
            __debugbreak();
  #else
        // 非 Windows: デバッガ判定のポータブル手段が無いため、Debug では停止する
        // （assert と同じ挙動）。Windows API の宣言や未定義シンボルは持ち込まない。
        assert (false && "EpochDomain: invalid RetireDomainId");
  #endif
#endif
    }

    int registerReaderThread() noexcept override
    {
        return registerReaderThread("unnamed");
    }

    // ★ 計測: 登録元タグ（文字列版へ委譲するだけで挙動は変えない）
    int registerReaderThreadWithTag(const char* tag) noexcept override
    {
        return registerReaderThread(tag);
    }

    // ★ C-3: タグ名付き Reader 登録
    // ★ work88 (X3 §6.3 / INV-X3-4): reader registration permanently closed — shutdown 後は
    //   新規登録を拒否する（registrationClosed_）。登録済み slot の enter/exit は継続可能
    //   （既存 Reader の epoch 安全性は維持 — 十八次別視点14）。
    int registerReaderThread(const char* tag) noexcept
    {
        if (! readerAdmissionGate_.tryEnter())
        {
            enterAfterFence_.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        if (convo::consumeAtomic(registrationClosed_, std::memory_order_acquire))
        {
            readerAdmissionGate_.exit();
            return -1;   // ★ X3: reader registration permanently closed（INV-X3-4）
        }
        for (int i = 0; i < kMaxReaders; ++i)
        {
            uint64_t expected = kInactiveEpoch;
            // acq_rel/acquire: 成功側 release で slot 取得を他スレッドに公開し、
            //                  failure 側 acquire で競合の write を観測してループを継続。
            if (convo::compareExchangeAtomic(readers[static_cast<size_t>(i)].epoch,
                                             expected,
                                             kReservedEpoch,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
            {
                // release: depth ゼロ化を slot 取得後に他スレッドが観測できるよう publish。
                convo::publishAtomic(readers[static_cast<size_t>(i)].depth,
                                     static_cast<uint32_t>(0),
                                     std::memory_order_release);
                // ★ C-3: 所有者タグ設定（CAS 成功後は単一スレッドのみがアクセス可能）
                if (tag != nullptr) {
                    std::strncpy(readers[static_cast<size_t>(i)].ownerTag, tag,
                                 sizeof(readers[static_cast<size_t>(i)].ownerTag) - 1);
                    readers[static_cast<size_t>(i)].ownerTag[
                        sizeof(readers[static_cast<size_t>(i)].ownerTag) - 1] = '\0';
                }
                // ★ work92 B-4: ownerThreadId を単調 ID 採番に統一（BUG-063/BIG 2-6）。
                //   ownerThreadId 初期値 0 = 無効値のため、採番は 1 起点静止
                //   （acquireUniqueThreadId 契約）と整合。
                convo::publishAtomic(readers[static_cast<size_t>(i)].ownerThreadId,
                                     convo::acquireUniqueThreadId(),
                                     std::memory_order_release);
                // ★ WORK113: 予約を admission として保持し、enterReader で終了する。
                convo::publishAtomic(readers[static_cast<size_t>(i)].admissionPending,
                                     static_cast<uint8_t>(1),
                                     std::memory_order_release);
                return i;
            }
        }

        readerAdmissionGate_.exit();
        return -1;
    }

    bool reserveReaderThread(int readerIndex) noexcept override
    {
        if (! readerAdmissionGate_.tryEnter())
        {
            enterAfterFence_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
        {
            readerAdmissionGate_.exit();
            return false;
        }
        // ★ work88 (X3 §6.3 / INV-X3-4): reserve 経由の登録も封じる（registrationClosed_ ガード）
        if (convo::consumeAtomic(registrationClosed_, std::memory_order_acquire))
        {
            readerAdmissionGate_.exit();
            return false;
        }

        uint64_t expected = kInactiveEpoch;
        // acq_rel/acquire: registerReaderThread と同じ HB 保証が必要。
        //   成功側 release で予約を公開し、failure 側 acquire で競合書き込みを観測。
        const bool reserved = convo::compareExchangeAtomic(
            readers[static_cast<size_t>(readerIndex)].epoch,
            expected,
            kReservedEpoch,
            std::memory_order_acq_rel,
            std::memory_order_acquire);

        if (reserved)
        {
            // release: depth ゼロ化を他スレッドが観測できるよう予約成功後に publish。
            convo::publishAtomic(readers[static_cast<size_t>(readerIndex)].depth,
                                 static_cast<uint32_t>(0),
                                 std::memory_order_release);
            convo::publishAtomic(readers[static_cast<size_t>(readerIndex)].admissionPending,
                                 static_cast<uint8_t>(1),
                                 std::memory_order_release);
        }
        else
        {
            readerAdmissionGate_.exit();
        }

        return reserved;
    }

    // ── V-11 (P0): ownership-checked re-acquisition after registration close ──
    //   registerReaderThread()/reserveReaderThread() stop working once
    //   registrationClosed_ is set (INV-X3-4). That is correct for admitting a *new*
    //   reader, but the comment on registerReaderThread() above states the intent for
    //   an already registered reader: "登録済み slot の enter/exit は継続可能
    //   （常駐 Reader の epoch 完全性は維持）". reserveReaderThread() breaks that
    //   intent after close: it refuses even the owning thread, so RCUReader::enter()
    //   fails, no slot contributes to getMinReaderEpoch(), and an object retired at the
    //   previous epoch becomes reclaimable while the callback reads it (V-11).
    //
    //   Ownership is checked against ownerThreadId, which registerReaderThread() sets
    //   from acquireUniqueThreadId(): a process-wide monotonic counter that is never
    //   reused. A recycled OS tid (or a colliding cachedThreadHash) cannot pass this
    //   check, so it is a stronger owner test than comparing cached TIDs.
    //
    //   This method does NOT admit a new reader: it requires an existing owner and an
    //   inactive slot, and it leaves registrationClosed_ untouched.
    bool reacquireReaderSlot(int readerIndex) noexcept override
    {
        // ★ WORK113 Reader admission: fence 閉鎖後は拒否し、閉鎖前の入場は enter まで数える。
        if (! readerAdmissionGate_.tryEnter())
        {
            reacquireAfterFence_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

#if defined(CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY)
        // テスト用シーム（テスト専用 TU のみ）。admission 取得後・fence 判定前に停止点を作る。
        if (auto fn = convo::consumeAtomic(readerAdmissionSeamFn_, std::memory_order_acquire))
            fn (convo::consumeAtomic(readerAdmissionSeamCtx_, std::memory_order_acquire));
#endif

        // ★ WORK113: admission 取得後に fence を読み直す。閉鎖済みなら入場しない
        //   （admission はここで終了させるため、shutdown の静止確認は待たされない）。
        if (shutdownReaderFenceEngaged())
        {
            reacquireAfterFence_.fetch_add(1, std::memory_order_relaxed);
            readerAdmissionGate_.exit();
            return false;
        }

        if (readerIndex < 0 || readerIndex >= kMaxReaders)
        {
            readerAdmissionGate_.exit();
            return false;
        }

        auto& slot = readers[static_cast<size_t>(readerIndex)];

        // Ownership: caller must be the thread that registered this slot.
        // 0 is the "never registered" sentinel and unique ids start at 1.
        const uint64_t callerId = convo::acquireUniqueThreadId();
        if (callerId == 0
            || convo::consumeAtomic(slot.ownerThreadId, std::memory_order_acquire) != callerId)
        {
            readerAdmissionGate_.exit();
            return false;
        }

        // Only an inactive slot is re-acquirable. kReservedEpoch means another path is
        // mid-reservation; a live epoch means the slot is still in use.
        uint64_t expected = kInactiveEpoch;
        const bool reserved = convo::compareExchangeAtomic(slot.epoch,
                                                           expected,
                                                           kReservedEpoch,
                                                           std::memory_order_acq_rel,
                                                           std::memory_order_acquire);
        if (!reserved)
        {
            readerAdmissionGate_.exit();
            return false;
        }

        // Same publish as reserveReaderThread: depth==0 visible before enterReader().
        convo::publishAtomic(slot.depth, static_cast<uint32_t>(0), std::memory_order_release);
        // ★ WORK113: 予約を admission として保持し、enterReader の完了で終了する。
        convo::publishAtomic(slot.admissionPending, static_cast<uint8_t>(1),
                             std::memory_order_release);

        // quarantineFlags intentionally untouched: a quarantined-idle slot stays
        // quarantined, and getMinReaderEpoch() still counts it while depth > 0
        // (see the STG-10-D1 note at getMinReaderEpoch). Clearing the flag here would
        // resurrect the slot into the normal accounting and could inflate the reclaim
        // barrier.
        return true;
    }

    [[deprecated("Use RCUReader::enter() instead. See refactoring_plan.md P1-18.")]]
    void enterReader(int readerIndex) noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return;

        auto& slot = readers[static_cast<size_t>(readerIndex)];
        // ★ BUG-050: epoch を depth++ より前に store して HB ギャップを除去する。
        //   getMinReaderEpoch が depth>0 を観測した時点（depth acquire）で、
        //   [B'] epoch store（depth++ より前）が release→acquire チェーンで必ず可視化される。
        const uint64_t epoch = currentEpoch();
        convo::publishAtomic(slot.epoch, epoch, std::memory_order_release);

        // release: 後続の epoch load が depth > 0 可視後に行われることを保証。
        const uint32_t previousDepth = convo::fetchAddAtomic(slot.depth,
                                                              static_cast<uint32_t>(1),
                                                              std::memory_order_acq_rel);

        // ★ WORK113: 予約〜enter の admission を epoch / depth の公開後に終了する。
        //   これにより shutdown の waitForReaderAdmissions が戻った時点で、
        //   入場済み reader の depth>0 と epoch は必ず可視になっている。
        if (convo::exchangeAtomic(slot.admissionPending, static_cast<uint8_t>(0),
                                  std::memory_order_acq_rel) != 0)
        {
            readerAdmissionGate_.exit();
        }

        if (previousDepth > 0)
            return;  // ネスト: epoch は active Reader を反映済み（>= 外側 epoch）で安全。再設定不要。

        // ★ Practical-8: 初回 enter 時に滞留開始時刻を記録
        const uint64_t nowUs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        convo::publishAtomic(slot.residencyStartTimestampUs, nowUs, std::memory_order_release);
    }

    [[deprecated("Use RCUReader::exit() instead. See refactoring_plan.md P1-18.")]]
    void exitReader(int readerIndex) noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return;   // slot に触らないため計数しない

        // ★ WORK113 R1: depth のデクリメントより前に増やす。
        //   relaxed でよい。直後の depth の acq_rel RMW がこの増加を公開する。
        (void)convo::fetchAddAtomic(readerExitInProgress_, static_cast<std::uint32_t>(1),
                                    std::memory_order_relaxed);

        // 減算は最後の provider アクセスの後（release）。早期 return（previousDepth == 0 /
        // > 1）でもこのスコープのデストラクタで必ず 1 回だけ戻る。
        struct ExitInProgressScope
        {
            std::atomic<std::uint32_t>& counter;
            ~ExitInProgressScope() noexcept
            {
                (void)convo::fetchSubAtomic(counter, static_cast<std::uint32_t>(1),
                                            std::memory_order_release);
            }
        } exitInProgressScope { readerExitInProgress_ };

        auto& slot = readers[static_cast<size_t>(readerIndex)];
        // acq_rel: 取得側 acquire で enterReader 以降の読み取りが完了していることを観測し、
        //          放出側 release でその読み取りが slot.epoch の inactive 化より先に完了することを保証。
        const uint32_t previousDepth = convo::fetchSubAtomic(slot.depth,
                                                              static_cast<uint32_t>(1),
                                                              std::memory_order_acq_rel);
#if defined(CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY)
        // ★ seam: depth を 0 にした直後、残りの provider アクセスの前で止める。
        if (previousDepth == 1)
        {
            const auto fn = convo::consumeAtomic(readerExitSeamFn_, std::memory_order_acquire);
            if (fn != nullptr)
                fn (convo::consumeAtomic(readerExitSeamCtx_, std::memory_order_acquire));
        }
#endif
        // ★ 計測: 区画の保持時間（1ms 以上のときだけ最長値を更新）
        if (previousDepth == 1 && readerIndex >= 0 && readerIndex < kMaxReaders)
        {
            const uint64_t startUs = convo::consumeAtomic(slot.residencyStartTimestampUs, std::memory_order_acquire);
            const uint64_t nowUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            if (startUs != 0 && nowUs > startUs && (nowUs - startUs) >= 1000ull)
            {
                auto& w = holdWitness_[static_cast<size_t>(readerIndex)];
                w.count.fetch_add(1u, std::memory_order_relaxed);
                w.slot.store(static_cast<uint32_t>(readerIndex), std::memory_order_relaxed);
                w.domainKey.store(reinterpret_cast<uint64_t>(this), std::memory_order_relaxed);
                std::strncpy(w.tag, slot.ownerTag, sizeof(w.tag) - 1);
                uint64_t expected = w.maxHoldUs.load(std::memory_order_relaxed);
                while (nowUs - startUs > expected
                       && ! convo::compareExchangeWeakAtomic (w.maxHoldUs, expected, nowUs - startUs,
                                                              std::memory_order_relaxed,
                                                              std::memory_order_relaxed))
                {
                }
            }
        }
        if (previousDepth == 0)
        {
            convo::publishAtomic(slot.depth, static_cast<uint32_t>(0), std::memory_order_release);
            return;
        }

        if (previousDepth > 1)
            return;

        // ★ Practical-8: 最終 exit 時に滞留時刻をクリア
        convo::publishAtomic(slot.residencyStartTimestampUs, uint64_t{0}, std::memory_order_release);

        // release: epoch を kInactiveEpoch に戻し、reclaimers がこのスロットを safe-below 判定から除外可能にする。
        convo::publishAtomic(slot.epoch, kInactiveEpoch, std::memory_order_release);

        // ★ Phase 3: pendingQuarantine が設定されていた場合、今 quarantine を確定する。
        //   この Reader は exit 後も quarantined フラグにより getMinReaderEpoch から除外される。
        // ★ BUG-049: CAS で 0x02(pending)→0x01(quarantined) へ原子的に昇格。
        //   plain store だと別 Coordinator が pending を再設定した際に競合するため。
        const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
        if ((flags & ReaderSlot::kPendingQuarantineFlag) != 0)
        {
            uint8_t expected = static_cast<uint8_t>(ReaderSlot::kPendingQuarantineFlag);
            convo::compareExchangeAtomic(slot.quarantineFlags,
                                         expected,
                                         static_cast<uint8_t>(ReaderSlot::kQuarantinedFlag),
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire);
        }
    }

    uint64_t currentEpoch() const noexcept override
    {
        // acquire: advanceEpoch の acq_rel release-side と HB し、最新 epoch を観測する。
        return convo::consumeAtomic(globalEpoch, std::memory_order_acquire);
    }

    // [work21] IEpochProvider::publishEpoch — inline advance to avoid deprecated call
    uint64_t publishEpoch() noexcept override
    {
        // ★ WORK113 Gate: fence 後に進めない（RAII で退出を保証）。
        PublishScope scope(publishGate_);
        if (! scope.entered())
        {
            latePublisherAttempts_.fetch_add(1, std::memory_order_relaxed);
            return currentEpoch();
        }

        // ★ dash2 §2.2 (G19): epoch 前進と同期して epochGeneration_ をインクリメント。
        //   ShutdownRuntimeIdentity::epochGeneration の実供給源（T10 / Permit ABA 防止）。
        (void)convo::fetchAddAtomic(epochGeneration_, static_cast<uint64_t>(1),
                                    std::memory_order_acq_rel);
        return convo::fetchAddAtomic(globalEpoch,
                                     static_cast<uint64_t>(1),
                                     std::memory_order_acq_rel);
    }

    uint64_t current() const noexcept
    {
        return currentEpoch();
    }

    uint64_t publish() noexcept
    {
        return publishEpoch();
    }

    uint64_t getMinReaderEpoch() const noexcept override
    {
        uint64_t minEpoch = currentEpoch();

        for (const auto& slot : readers)
        {
            // acquire: enterReader release の depth 書き込みと HB し、depth 読み取り後に epoch を読む。
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);

            // ★ STG-10-D1 (RC-1): quarantined Reader は safe-epoch 計算から除外するが、
            //   その根拠は「非参加（depth==0）」でなければならない。
            //   quarantine は reader の占有に対する一時隔離であり、slot の封印ではない。
            //   registerReaderThread / reserveReaderThread は quarantineFlags を読まず
            //   epoch==kInactiveEpoch だけで再割当するため、隔離済みの slot には
            //   次の audio block で新規 reader が入場する（公開設計どおり）。
            //   したがって「kQuarantinedFlag && depth>0」は正当な active reader であり、
            //   フラグ単独で除外すると minReaderEpoch が真値より大きくなり、
            //   その reader が参照中の entry が reclaim され得る（Release では無言）。
            //   非参加の隔離 reader のみを除外する（従来と同一の挙動）。
            const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
            if ((flags & ReaderSlot::kQuarantinedFlag) != 0 && depth == 0)
                continue;

            if (depth == 0)
                continue;

            // acquire: enterReader の epoch publish release と HB し、安全に epoch 値を取得。
            const uint64_t epoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
            if (epoch == kInactiveEpoch || epoch == kReservedEpoch)
                continue;

            if (isOlder(epoch, minEpoch))
                minEpoch = epoch;
        }

        return minEpoch;
    }

    // ★ WORK113 R1 (2026-11-19 Owner 承認): 戻り値は reader の実人数ではなく、
    //   「この domain をまだ触っている reader の活動数」である。
    //   - depth > 0 のスロット数
    //   - exitReader() 実行中の件数（depth が 0 でも provider の後処理が残る）
    //   exit 中の reader は両方に現れることがある（一時的な二重計上）。
    //   静止判定は「0 か否か」しか使わないため、この保守性は安全側に倒れる。
    //   読み取り順序は depth（acquire）→ exit カウンタ（acquire）の順で固定する。
    //   depth の acq_rel RMW を観測してから exit カウンタを読むことで、
    //   depth を 0 にした exit の先行する増加を見落とさない。
    uint32_t activeReaderCount() const noexcept override
    {
        const uint32_t depthReaders = activeDepthReaderCount();

        // ★ 順序固定: depth を先に読む。逆にすると「exit 開始前に 0 を読み、
        //   depth 0 を観測する」並びで増加を見落とす。
        const uint32_t exiting =
            convo::consumeAtomic(readerExitInProgress_, std::memory_order_acquire);

        return depthReaders + exiting;
    }

    // depth > 0 のスロット数（reader の実人数）。診断・証跡用。
    // ゲート判定には activeReaderCount() を使うこと。
    uint32_t activeDepthReaderCount() const noexcept
    {
        uint32_t count = 0;

        for (const auto& slot : readers)
        {
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
            if (depth != 0)
                ++count;
        }

        return count;
    }

    int readerCapacity() const noexcept override
    {
        return kMaxReaders;
    }

    // ── ★ Phase 3: Reader Quarantine API ──

    // ★ stuck Reader を quarantined にマーク（killしない）
    //   depth==0: 即座quarantine → true
    //   depth>0: pendingQuarantine設定 → exitReader時にquarantine → false (deferred)
    // ★ BUG-049: 即座隔離と遅延隔離の並行 store 競合を CAS で排除。
    //   従来は plain store で 0x02(pending) と 0x01(quarantined) が競合し、
    //   depth==0 なのに pending が生存 → verifyReaderInvariants assert 発火。
    //   CAS を使うことで flags は 0x00→0x02(pending のみ) / 0x00→0x01 / 0x02→0x01(/昇格) の
    //   原子的遷移のみ許し、0x03(quarantined|pending) を生成しない。
    [[nodiscard]] bool quarantineReader(int readerIndex) noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return false;

        auto& slot = readers[static_cast<size_t>(readerIndex)];
        // acquire: enterReader/exitReader の深度変更を観測
        const uint32_t d = convo::consumeAtomic(slot.depth, std::memory_order_acquire);

        if (d == 0)
        {
            // 即座 quarantine: 深度0 → 他スレッドが参照していない
            // ★ 競合を許容しないため CAS を使用。
            //   ① pending(0x02) が別 Coordinator によって設定されていた → quarantined(0x01) へ昇格
            uint8_t expected = static_cast<uint8_t>(ReaderSlot::kPendingQuarantineFlag);
            if (convo::compareExchangeAtomic(slot.quarantineFlags,
                                             expected,
                                             static_cast<uint8_t>(ReaderSlot::kQuarantinedFlag),
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
                return true;
            //   ② 未設定(0x00) からの即座 quarantine
            expected = static_cast<uint8_t>(0);
            if (convo::compareExchangeAtomic(slot.quarantineFlags,
                                             expected,
                                             static_cast<uint8_t>(ReaderSlot::kQuarantinedFlag),
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
                return true;
            //   ③ 既に quarantined/その他 → 隔離は達成済み
            return false;
        }

        // depth > 0: 遅延隔離 — exitReader で depth==0 になった時点で quarantine 確定
        // ★ 既に quarantined なら遅延設定は不要。
        const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
        if ((flags & ReaderSlot::kQuarantinedFlag) != 0)
            return false;
        // 未設定(0x00) からのみ pending(0x02) を CAS で設定（0x03 生成を防止）。
        uint8_t expectPending = static_cast<uint8_t>(0);
        convo::compareExchangeAtomic(slot.quarantineFlags,
                                     expectPending,
                                     static_cast<uint8_t>(ReaderSlot::kPendingQuarantineFlag),
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire);
        return false;
    }

    // ★ Shutdown専用: 全quarantined Readerを解放（destroyForShutdown と同じパターン）
    void unquarantineAllReaders() noexcept override
    {
        for (auto& slot : readers)
        {
            convo::publishAtomic(slot.quarantineFlags,
                                 static_cast<uint8_t>(0),
                                 std::memory_order_release);
        }
    }

    // ★ quarantined Reader数の取得（kQuarantinedFlagでカウント）
    [[nodiscard]] int quarantinedReaderCount() const noexcept override
    {
        int count = 0;
        for (const auto& slot : readers)
        {
            const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
            if ((flags & ReaderSlot::kQuarantinedFlag) != 0)
                ++count;
        }
        return count;
    }

    // ★ Phase 3: Debug 検証 — quarantined slot の不変条件をチェック
    //   Release ビルドでは除去される（assert は NDEBUG で消滅）
    void verifyReaderInvariants() const noexcept
    {
        for (int i = 0; i < kMaxReaders; ++i)
        {
            const auto& slot = readers[static_cast<size_t>(i)];
            const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, std::memory_order_acquire);
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
            const uint64_t epoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
            (void)epoch;  // suppress icx -Wunused-variable (used only in assert)

            const bool isQuarantined = (flags & ReaderSlot::kQuarantinedFlag) != 0;
            const bool isPending = (flags & ReaderSlot::kPendingQuarantineFlag) != 0;

            // ★ quarantined Reader は epoch==kInactiveEpoch を期待（exitReader が epoch を inactive にした後で quarantine に遷移する）
            //   ※ 厳密には quarantined フラグ設定前に epoch が inactive になる保証はない（exitReader の acq_rel 順序で暗黙保証）。
            //     ここでは depth==0 かつ quarantined の場合は epoch も inactive であることを確認。
            if (isQuarantined && depth == 0)
            {
                assert(epoch == kInactiveEpoch || epoch == kReservedEpoch);
            }

            // ★ pendingQuarantine 設定中は depth > 0 を期待（depth==0 なら即座に quarantined に遷移するため pending は残らない）
            if (isPending)
            {
                assert(depth > 0 && "pendingQuarantine set but depth==0: should have been promoted to quarantined");
            }

            // ★ quarantined + pending は同時に成立しない
            assert(!(isQuarantined && isPending));
        }
    }

    // [work21] IEpochProvider::tryReclaim — inline reclaim to avoid deprecated call
    void tryReclaim() noexcept override
    {
        // ★ ★ A-2: 統計カウンタ (Local Aggregation によりキャッシュ競合低減)
        constexpr uint32_t kCounterAggregationInterval = 1024;
        const uint32_t localCount = reclaimLocalCounter_.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((localCount % kCounterAggregationInterval) == 0) {
            reclaimAttemptCount_.fetch_add(kCounterAggregationInterval, std::memory_order_relaxed);
        }
        auto q = borrow();
        const auto n = (q != nullptr) ? q->reclaim(getMinReaderEpoch())
                                      : static_cast<decltype(q->reclaim(getMinReaderEpoch()))>(0);
        reclaimSuccessCount_.fetch_add(n, std::memory_order_relaxed);
    }

    // ★ AC_DIAG (2026-11-13, Owner 承認 A/C): 診断専用 — 退役キュー先頭 entry と epoch 安全条件。
    //   reclaim() が削除可否に使う述語（isOlder(entry.epoch, minReaderEpoch)）と同じ式で判定する。
    //   キューは変更しない（peekHead は POD コピーのみ）。値が取れない場合は valid=false とし、
    //   推定値は入れない（呼出し側は unavailable と表示する）。
    struct RetireHeadSnapshot {
        bool     valid{false};                    // false = 先頭なし / sequence 確認失敗
        std::uint32_t headSlotIndex{0};           // ★ C3-D: head identity の 1 要素目
        std::uint32_t headSequence{0};            // ★ C3-D: head identity の 2 要素目
        uint64_t headEpoch{0};                    // ★ C3-D: head identity の 3 要素目
        uint64_t minReaderEpoch{0};               // reclaim() に渡すのと同じ値
        uint64_t currentEpoch{0};
        DeletionEntryType type{DeletionEntryType::Generic};
        convo::RetireDomainId domainId{convo::RetireDomainId::Unknown};
        uint64_t firstScanAgeMs{0};               // 0 = 未走査（firstScanTsNs==0）
        bool     scanned{false};                  // reclaim 走査で一度でも評価されたか
        bool     epochOlderThanMinReader{false};  // isOlder(headEpoch, minReaderEpoch)
        // ★ WORK113 変更1: サイドテーブルの帰属メタ（DIAG ON のみ意味を持つ）。
        std::uint64_t ownerId{0};                 // DSPCore の runtimeUuid（0 = 未取得）
        std::uint64_t measuredBytes{0};           // 計測済み帰属分（0 = 未計測）
        bool     metaAvailable{false};            // メタが供給され採用されたか
    };

    [[nodiscard]] RetireHeadSnapshot inspectRetireHead() const noexcept
    {
        RetireHeadSnapshot s;
        s.minReaderEpoch = getMinReaderEpoch();
        s.currentEpoch = currentEpoch();

        DeletionEntry e;
        std::uint32_t slot = 0;
        std::uint32_t seq = 0;
        auto q = borrow();
        if (q == nullptr || ! q->peekHead(e, slot, seq))
            return s;                             // valid=false のまま（推定しない）

        s.valid = true;
        s.headSlotIndex = slot;                   // identity は 3 要素で保持する
        s.headSequence = seq;
        s.headEpoch = e.epoch;
        s.type = e.type;
        s.domainId = e.domainId;
        s.epochOlderThanMinReader = isOlder(e.epoch, s.minReaderEpoch);

        if (e.firstScanTsNs != 0)
        {
            s.scanned = true;
            // DeferredDeletionQueue::steadyNowNs() と同じ steady_clock 基準。
            const uint64_t nowNs = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            if (nowNs > e.firstScanTsNs)
                s.firstScanAgeMs = (nowNs - e.firstScanTsNs) / 1'000'000ull;
        }

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        // ★ WORK113 変更1: 先頭エントリの帰属メタを照合して載せる（seq 不一致なら載せない）。
        DiagRetireMetaView view {};
        if (q->readMeta(slot, seq, view) && view.available)
        {
            s.ownerId = view.ownerId;
            s.measuredBytes = view.measuredBytes;
            s.metaAvailable = true;
        }
#endif
        return s;
    }

    // ★ P0-A/P2-A: IRetireProvider インターフェース実装（public 必須）
    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept override
    {
#ifndef NDEBUG
        // ★ WORK113: RT から NonRT API を呼ぶ誤用を Debug で止める（Release では判定しない）。
        if (rtProbe_ != nullptr && rtProbe_())
        {
            rtMisuseCount_.fetch_add(1, std::memory_order_relaxed);
            assert(false && "retire enqueue is not allowed on the audio thread");
        }
#endif
        auto q = borrow();
        if (! q)
            return false;
        return q->enqueue(ptr, deleter, epoch, domainId_);
    }

    // ★ WORK113: RT 専用（生ポインタのみ・mutex も shared_ptr も触らない）。
    bool enqueueRetireRt(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept override
    {
        auto* q = convo::consumeAtomic(queueRt_, std::memory_order_acquire);
        if (q == nullptr)
            return false;
        return q->enqueue(ptr, deleter, epoch, domainId_);
    }

    // ★ T1 (D86): telemetry type tag 付き enqueue（type を DeferredDeletionQueue に伝搬）。
    //   type==World で world retirement を識別（telemetry metadata のみ・lifetime authority にしない）。
    bool enqueueRetireTyped(void* ptr, void (*deleter)(void*), uint64_t epoch,
                            DeletionEntryType type) noexcept override
    {
        auto q = borrow();
        if (! q)
            return false;
        return q->enqueue(ptr, deleter, epoch, type, domainId_);
    }

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ WORK113 変更1: 帰属メタ付き enqueue（NonRT のみ）。サイドテーブルへ記録する。
    bool enqueueRetireSized(void* ptr, void (*deleter)(void*), uint64_t epoch,
                            DeletionEntryType type, const RetireMeta& meta) noexcept override
    {
        auto q = borrow();
        if (! q)
            return false;
        return q->enqueueSized(ptr, deleter, epoch, type, 0, 0, domainId_, meta);
    }

    [[nodiscard]] bool supportsRetireMeta() const noexcept override { return true; }
#endif

    // ★ T1 (D86): type==World の terminal deleter 実行数（world 物理破棄数・primary 経路）。
    [[nodiscard]] uint64_t worldReclaimCount() const noexcept override
    {
        auto q = borrow();
        return (q != nullptr) ? q->worldReclaimCount() : 0u;
    }

    // ★ T1 (D98): reference observer を DeferredDeletionQueue に伝搬（non-owning・一方向依存）。
    void setReferenceObserver(void* observer) noexcept override
    {
        auto q = borrow();
        if (q != nullptr)
            q->setReferenceObserver(
                static_cast<convo::isr::WorldRetirementReferenceObserver*>(observer));
    }

    void drainAll() noexcept override
    {
        auto q = borrow();
        if (q != nullptr)
            q->drainAllUnsafe();
    }

    [[nodiscard]] uint32_t pendingRetireCount() const noexcept override
    {
        auto q = borrow();
        return (q != nullptr) ? q->sizeApprox() : 0u;
    }

    static bool isOlder(uint64_t a, uint64_t b) noexcept
    {
        return static_cast<int64_t>(a - b) < 0;
    }

    // ★ B-1: Reader Slot 詳細取得（アクティブ Reader の epoch/depth/residency を返す）
    [[nodiscard]] ReaderSlotDetail getReaderSlotDetail(int readerIndex) const noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return ReaderSlotDetail{};

        const auto& slot = readers[static_cast<size_t>(readerIndex)];
        const uint64_t epoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
        const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
        const uint64_t startUs = convo::consumeAtomic(slot.residencyStartTimestampUs, std::memory_order_acquire);
        const auto nowUs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        const uint64_t residencyUs = (startUs != 0 && depth > 0) ? (nowUs - startUs) : 0;

        return ReaderSlotDetail{epoch, depth, residencyUs, (depth > 0)};
    }

    // [work37 Phase 2.1] 複合判定: epoch差 AND residency 時間条件
    //   条件1: epoch差 > threshold AND residency > 1秒 → Stuck
    //   条件2: residency > 30秒 (epoch差不問) → Chronic Stuck
    //   条件3: depth > 0 AND residency > 10秒 AND pendingRetire > 0 → Warning Stuck
    [[nodiscard]] StuckReaderInfo detectStuckReaders(uint64_t stuckThreshold) const noexcept override {
        StuckReaderInfo info;
        info.currentEpoch = convo::consumeAtomic(globalEpoch, std::memory_order_acquire);
        info.minReaderEpoch = getMinReaderEpoch();
        {
            auto q = borrow();
            info.pendingRetireCount = (q != nullptr) ? q->sizeApprox() : 0u;
        }

        constexpr uint64_t kResidencyStuckUs = 1'000'000;      // 1秒 — epoch差とのAND条件
        constexpr uint64_t kChronicResidencyUs = 30'000'000;   // 30秒 — epoch差不問
        constexpr uint64_t kWarningResidencyUs = 10'000'000;   // 10秒 — Warning用

        const auto nowUs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());

        // ★ BUG-048: 単一ループの break-on-first を、深刻度の高い順の 3 パス評価に変更。
        //   従来は readerIndex 昇順で「最初に見つかった」Reader を報告し、
        //   より深刻な Chronic が index の大きい Reader に居ると見逃され得た。
        //   パス順序: Pass3 Chronic → Pass2 Warning → Pass1 EpochGap。
        auto buildReaderInfo = [&](int i, int severity) -> bool {
            if (i < 0 || i >= kMaxReaders)
                return false;
            const auto& slot = readers[static_cast<size_t>(i)];
            const uint64_t readerEpoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
            if (readerEpoch == kInactiveEpoch)
                return false;

            const uint64_t ec = slot.enterCount.load(std::memory_order_relaxed);
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);

            // ★ P4.5: residencyTime を実時間ベースで計算（epoch差ではなくsteady_clock）
            const uint64_t startUs = convo::consumeAtomic(slot.residencyStartTimestampUs, std::memory_order_acquire);
            const uint64_t residencyUs = (startUs != 0 && depth > 0) ? (nowUs - startUs) : 0;

            bool matched = false;
            switch (severity) {
                case 3:  // [work37] 条件2: residency > 30秒 (epoch差不問) → Chronic Stuck
                    matched = (depth > 0 && residencyUs > kChronicResidencyUs && info.pendingRetireCount > 0);
                    if (matched)
                        info.isChronic = true;
                    break;
                case 2:  // [work37] 条件3: residency > 10秒 AND pendingRetire > 0 → Warning Stuck
                    matched = (depth > 0 && residencyUs > kWarningResidencyUs && info.pendingRetireCount > 0);
                    if (matched)
                        info.isChronic = false;
                    break;
                case 1:  // [work37] 条件1: epoch差 > threshold AND residency > 1秒
                    if (depth > 0 && readerEpoch < info.currentEpoch) {
                        const uint64_t epochGap = info.currentEpoch - readerEpoch;
                        matched = (epochGap > stuckThreshold && residencyUs > kResidencyStuckUs);
                    }
                    if (matched)
                        info.isChronic = false;
                    break;
                default:
                    break;
            }
            if (!matched)
                return false;

            info.readerIndex = i;
            info.readerEpoch = readerEpoch;
            info.enterCount = ec;
            info.isStuck = true;
            info.residencyTimeUs = residencyUs;
            // [work37 9.42] + BUG-063: ownerThreadId を先に acquire し、非0時のみ ownerTag をコピー
            info.ownerThreadId = convo::consumeAtomic(slot.ownerThreadId, std::memory_order_acquire);
            if (info.ownerThreadId != 0) {
                std::strncpy(info.ownerTag, slot.ownerTag, sizeof(info.ownerTag) - 1);
                info.ownerTag[sizeof(info.ownerTag) - 1] = '\0';
            }
            return true;
        };

        for (int pass = 3; pass >= 1; --pass) {
            for (int i = 0; i < kMaxReaders; ++i) {
                if (buildReaderInfo(i, pass))
                    return info;
            }
        }
        return info;
    }

    // ★ P2-A: 以下の deprecated API は移行完了により private 化。
    //   外部からの新規使用を禁止し、publishEpoch() / tryReclaim() を推奨。
private:
    [[deprecated("Use publishEpoch() instead.")]]
    uint64_t advanceEpoch() noexcept
    {
        PublishScope scope(publishGate_);
        if (! scope.entered())
        {
            latePublisherAttempts_.fetch_add(1, std::memory_order_relaxed);
            return currentEpoch();
        }
        return convo::fetchAddAtomic(globalEpoch,
                                     static_cast<uint64_t>(1),
                                     std::memory_order_acq_rel);
    }

    [[deprecated("Use tryReclaim() instead.")]]
    void reclaimRetired() noexcept
    {
        auto q = borrow();
        if (q != nullptr)
            (void) q->reclaim(getMinReaderEpoch());
    }

    [[deprecated("Use coordinator.enqueueRetire() instead.")]]
    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type) noexcept
    {
        auto q = borrow();
        if (! q)
            return false;
        return q->enqueue(ptr, deleter, epoch, type, domainId_);
    }

    struct ReaderSlot
    {
        std::atomic<uint64_t> epoch { kInactiveEpoch };
        std::atomic<uint32_t> depth { 0 };
        std::atomic<uint64_t> enterCount { 0 };  // ★ P3-1: enter 回数のみカウント（軽量）
        std::atomic<uint64_t> residencyStartTimestampUs { 0 }; // ★ P4.5: steady_clock ベースの滞留開始時刻
        // ★ C-3: Reader 所有者情報
        std::atomic<uint64_t> ownerThreadId { 0 };       // std::thread::id のハッシュ値
        char ownerTag[32] {};  // "AudioThread", "TimerThread" 等（CAS 排他下で設定、stale read 許容）

        // ★ Phase 3: Reader Quarantine フラグ
        //   bit 0 (kQuarantinedFlag):   隔離済 — getMinReaderEpoch から除外
        //   bit 1 (kPendingQuarantineFlag):  exitReader で depth==0 になったら隔離
        static constexpr uint8_t kQuarantinedFlag       = 0x01;
        static constexpr uint8_t kPendingQuarantineFlag  = 0x02;
        std::atomic<uint8_t> quarantineFlags{0};

        // ★ WORK113: 予約〜enter の間を admission として数える（shutdown は 0 を待つ）。
        std::atomic<uint8_t> admissionPending{0};
    };

    std::atomic<uint64_t> globalEpoch;
    std::array<ReaderSlot, kMaxReaders> readers;

    // ★ WORK113 Queue Lifetime: NonRT は borrow()、shutdown は takeForShutdown()。
    mutable std::mutex queueRefMutex_;
    std::shared_ptr<DeferredDeletionQueue> queueRef_;
    std::atomic<DeferredDeletionQueue*> queueRt_ { nullptr };
    std::atomic<bool> queueAvailable_ { false };

    // ★ WORK113 Gate / Probe / Fence / Drain 状態
    EpochPublishGate publishGate_;
    // ★ WORK113: reader admission gate（register / reserve / reacquire 〜 enterReader）。
    EpochPublishGate readerAdmissionGate_;
    RtThreadProbe rtProbe_ { nullptr };
    std::atomic<std::uint64_t> rtMisuseCount_ { 0 };
    std::atomic<bool> shutdownReaderFence_ { false };
#if defined(CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY)
    inline static std::atomic<ReaderAdmissionSeam> readerAdmissionSeamFn_ { nullptr };
    inline static std::atomic<void*> readerAdmissionSeamCtx_ { nullptr };
    inline static std::atomic<ReaderExitSeam> readerExitSeamFn_ { nullptr };
    inline static std::atomic<void*> readerExitSeamCtx_ { nullptr };
#endif
    std::atomic<std::uint32_t> reacquireAfterFence_ { 0 };
    std::atomic<std::uint32_t> enterAfterFence_ { 0 };
    std::atomic<std::uint32_t> epochIdleTimeouts_ { 0 };
    std::atomic<std::uint32_t> latePublisherAttempts_ { 0 };
    std::atomic<std::uint64_t> epochAtFence_ { 0 };
    std::atomic<std::uint64_t> drainReaderFailures_ { 0 };
    std::atomic<std::uint64_t> drainRemainingAfter_ { 0 };

    // ★ WORK113 R1 (2026-11-19 Owner 承認): exitReader() を実行中の件数。
    //   depth が 0 でも provider の後処理（residency / holdWitness / slot.epoch /
    //   quarantineFlags）が残っている reader を数える。増加は depth のデクリメント前、
    //   減算は最後の provider アクセス後（release）。詳細は exitReader() を参照。
    std::atomic<std::uint32_t> readerExitInProgress_ { 0 };

    // 隔離（append-only・自己参照で寿命を保つ）
    inline static std::atomic<DeferredDeletionQueue*> quarantineHead_ { nullptr };
    inline static std::atomic<std::uint64_t> quarantineCount_ { 0 };
    inline static std::atomic<std::uint64_t> quarantinePendingEntries_ { 0 };
    inline static std::atomic<std::uint64_t> quarantineRetainedBytes_ { 0 };

    // ★ domain tag (v2.9 §3.2): 自分のドメイン（enqueue 時に entry へ刻印する）。
    //   宣言順は domainId_ → globalEpoch（初期化子の順序警告を出さない）。
    RetireDomainId domainId_ { RetireDomainId::Unknown };

    // ★ domain tag: 不正値の生成回数（診断用・全ビルドで観測可能）
    inline static std::atomic<std::uint32_t> invalidDomainAssignments_ { 0 };

public:
    // ★ 計測 (2026-11-10): reader 区画の最長保持時間の証跡。
    //   スロット単位の固定容量で、ポインタを辿らない（破棄後も安全に読める）。
    struct HoldWitness
    {
        std::atomic<uint64_t> maxHoldUs;
        std::atomic<uint32_t> count;
        std::atomic<uint32_t> slot;
        char tag[32];
        std::atomic<uint64_t> domainKey;

        // ★ icx (Clang) 対応: 既定メンバ初期化子は外側クラス定義内では使えない。
        //   static inline 配列 holdWitness_ の {} 初期化のため constexpr ctor にする。
        constexpr HoldWitness() noexcept
            : maxHoldUs(0), count(0), slot(0xFFFFFFFFu), tag{}, domainKey(0) {}
    };
    static inline HoldWitness holdWitness_[kMaxReaders]{};

    // ★ 計測: 保持時間の証跡を JSON として書き出す（NonRT から呼ぶこと）。
    static void dumpHoldWitness(std::FILE* f) noexcept
    {
        if (f == nullptr)
            return;
        std::fprintf(f, "{\n  \"schema\": \"retire_reader_holds_v1\",\n");
        std::fprintf(f, "  \"note\": \"1ms 以上の reader 区画の最長保持時間（スロット単位・固定容量）\",\n");
        std::fprintf(f, "  \"slots\": [");
        bool first = true;
        for (int i = 0; i < kMaxReaders; ++i)
        {
            const auto& w = holdWitness_[static_cast<size_t>(i)];
            const uint64_t h = w.maxHoldUs.load(std::memory_order_relaxed);
            if (h == 0)
                continue;
            std::fprintf(f, "%s\n    {\"slot\": %d, \"tag\": \"%s\", \"maxHoldUs\": %llu, \"count\": %u}",
                         first ? "" : ",", i, w.tag,
                         (unsigned long long) h,
                         w.count.load(std::memory_order_relaxed));
            first = false;
        }
        std::fprintf(f, "%s]\n}\n", first ? "" : "\n  ");
    }

    // ★ A-2: EBR Queue Visibility 統計カウンタ
    std::atomic<uint64_t> reclaimAttemptCount_{0};
    std::atomic<uint64_t> reclaimSuccessCount_{0};
    // ★ A-2: Local Aggregation 用カウンタ (per-core cache line)
#pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
    alignas(64) std::atomic<uint32_t> reclaimLocalCounter_{0};
#pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

    // ★ work88 (X3 §6.3 / INV-X3-4 / INV-ISR-04): reader registration permanently closed フラグ。
    //   closeReaderRegistration() で true（shutdown state machine の CloseReaderRegistration フェーズ）。
    //   registerReaderThread / reserveReaderThread は true 後は失敗を返す（新規登録拒否）。
    std::atomic<bool> registrationClosed_{false};

    // ── ★ dash2 §2.2 (Phase A2 — G19/G20): generation 実供給源 ──
    //   ShutdownRuntimeIdentity の epochGeneration / readerRegistrationGeneration を束縛する
    //   generation 値。型フィールドの存在だけでなく、authority（EpochDomain）からの実供給・
    //   検証経路を成立させる（H.11.11.9.3 G-D1〜D4）。
    //   - epochGeneration_: publishEpoch() ごとにインクリメント（epoch 前進と同期）
    //   - readerRegistrationGeneration_: closeReaderRegistration() ごとにインクリメント
    //     （registration 状態の世代 — 再オープンは存在しないため単調増加）
    //   ［生成は EpochDomain のみ。ShutdownRuntime は getter 経由で観測し Proof/Permit に束縛］
    std::atomic<uint64_t> epochGeneration_{0};
    std::atomic<uint64_t> readerRegistrationGeneration_{0};

public:
    // ★ A-2: 公開アクセサ
    [[nodiscard]] uint64_t reclaimAttemptCount() const noexcept override {
        // ★ 未集計分を加算 (relaxed で十分: 診断目的のため正確性は要求されない)
        const auto local = reclaimLocalCounter_.load(std::memory_order_relaxed);
        const auto committed = convo::consumeAtomic(reclaimAttemptCount_, std::memory_order_acquire);
        return committed + (local % 1024);
    }
    [[nodiscard]] uint64_t reclaimSuccessCount() const noexcept override {
        return convo::consumeAtomic(reclaimSuccessCount_, std::memory_order_acquire);
    }

    // ★ work88 (X3 §6.3 / INV-X3-4): reader registration を永久に閉じる（CloseReaderRegistration）。
    //   新規登録のみを拒否し、登録済み slot は exit まで継続可能（既存 Reader の epoch 安全性維持）。
    void closeReaderRegistration() noexcept {
        convo::publishAtomic(registrationClosed_, true, std::memory_order_release);
        // ★ dash2 §2.2 (G20): registration 状態の世代をインクリメント。
        //   ShutdownRuntimeIdentity::readerRegistrationGeneration の実供給源。
        //   registrationClosed_ の publish 後（release）に increment するため、
        //   acquire で読む側は「閉じた後の世代」を観測する（INV-LIFE-6 / T10）。
        (void)convo::fetchAddAtomic(readerRegistrationGeneration_, static_cast<uint64_t>(1),
                                    std::memory_order_acq_rel);
    }
    [[nodiscard]] bool readerRegistrationClosed() const noexcept {
        return convo::consumeAtomic(registrationClosed_, std::memory_order_acquire);
    }

    // ★ dash2 §2.2 (G19/G20): generation 公開アクセサ（ShutdownRuntime / Proof 生成用）。
    [[nodiscard]] uint64_t epochGeneration() const noexcept {
        return convo::consumeAtomic(epochGeneration_, std::memory_order_acquire);
    }
    [[nodiscard]] uint64_t readerRegistrationGeneration() const noexcept {
        return convo::consumeAtomic(readerRegistrationGeneration_, std::memory_order_acquire);
    }
};

} // namespace convo
