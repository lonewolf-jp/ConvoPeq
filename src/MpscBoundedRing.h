//==============================================================================
// MpscBoundedRing.h — Bounded Multi-Producer Single-Consumer リング (work88 FUTURE-10)
//
// 目的:
//   FUTURE-10 前提 0 — intentQueue_ は既に複数 Producer（Builder/Rebuild スレッド・
//   Timer スレッド・CoordinatorLoop deferred resubmit）から push される **MPSC 実態**だが、
//   従来の LockFreeRingBuffer は SPSC 専用（writeIndex の非 CAS 更新が複数 Producer の
//   同時 push で競合 → エントリ破損・要素消失のデータ競合）。
//   本プリミティブは Vyukov bounded MPMC アルゴリズム（DeferredDeletionQueue.h と同型）の
//   **単一 Consumer 版**として実装する。
//
// アルゴリズム（Vyukov bounded queue）:
//   - producer: enqueuePos の CAS で slot 予約（reservation order = seqId order）。
//     予約後に payload 書き込み → seq release（publication order）。
//   - consumer: dequeuePos の slot seq が (pos+1) に達した場合のみ読み取り（単一 Consumer）。
//     未書き込み slot（producer hole）に到達した場合は false を返し、次の poll で再試行
//     （Consumer は自身の slot が書き込み完了するまで次の slot を読まない）。
//   - INV-7 (MPSC ordering): sequenceId assignment → reservation → publication →
//     consumption → completion の 4 順序を分離。pop は reservation order で行われ、
//     publication order（payload visibility）は seq 番号で検証する。
//   - 有界: Capacity 超過時は push が false を返す（呼出し元が per-type admission policy を適用）。
//     **drop はしない**（Publish/Quarantine/Recovery は runtime state transition の運搬手段）。
//
// スレッド安全:
//   - push: 複数 Producer から呼び出し可能（CAS）。
//   - pop: 単一 Consumer（CoordinatorLoop）からのみ呼び出すこと。
//   - RT パス（Audio Thread）からは push/pop されない（Producer は全て NonRT）。
//==============================================================================
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <type_traits>

#include "audioengine/AtomicAccess.h"

#ifdef CONVO_TESTING
#  define CONVO_MPSC_TEST_HOOKS
#endif

#ifdef _MSC_VER
#  pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#  pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#endif

template<typename T, size_t Capacity>
class MpscBoundedRing {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    static constexpr size_t kMask = Capacity - 1;

public:
    MpscBoundedRing() noexcept
    {
        // release: 初期化後の最初の観測（enqueue の seq acquire と HB）を保証
        for (size_t i = 0; i < Capacity; ++i)
            convo::publishAtomic(sequences_[i], static_cast<uint32_t>(i), std::memory_order_release);
    }

    MpscBoundedRing(const MpscBoundedRing&) = delete;
    MpscBoundedRing& operator=(const MpscBoundedRing&) = delete;

    // Multi-producer safe。戻り値 false = full（呼出し元が per-type admission policy を適用）。
    // Producer hole: CAS 予約（reservation）と payload 書込み（publication）の間に
    // 別 Producer が先に書いても、seq 番号で検証するため consumer は torn 読まない。
    bool push(const T& item) noexcept
    {
        uint32_t pos = convo::consumeAtomic(enqueuePos_, std::memory_order_acquire);
        while (true)
        {
            auto& seq_atom = sequences_[pos & kMask];
            const uint32_t seq = convo::consumeAtomic(seq_atom, std::memory_order_acquire);
            // Capacity ≪ INT32_MAX により int32_t 減算で安全（DeferredDeletionQueue と同型）
            const int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));

            if (diff == 0)
            {
                // slot 予約（reservation order）
                if (convo::compareExchangeAtomic(enqueuePos_, pos, static_cast<uint32_t>(pos + 1),
                                                 std::memory_order_acq_rel,  // 成功時 acq_rel
                                                 std::memory_order_acquire)) // 失敗時 acquire: 最新 enqueuePos を再観測
                {
#ifdef CONVO_TESTING
                    // ★ work88 (P2-3): producer-hole 再現フック — payload 書込み前に一時停止。
                    //   testHoleBlockPos_ と一致する reservation 位置のみ、testHoleGate_ が
                    //   true になるまで待機（デフォルト UINT32_MAX = pass-through、既存テストは
                    //   非活性）。これにより「reservation 済み・未公開（producer hole）」状態を
                    //   deterministic に作り、consumer が未書き込み slot を跨がないことを検証する。
                    if (convo::consumeAtomic(testHoleBlockPos_, std::memory_order_acquire) == pos)
                    {
                        convo::publishAtomic(testHoleReady_, true, std::memory_order_release);
                        while (!convo::consumeAtomic(testHoleGate_, std::memory_order_acquire))
                            std::this_thread::yield();  // testReleaseHole() 待ち
                    }
#endif
                    entries_[pos & kMask] = item;  // payload 書込み（publication）
                    convo::publishAtomic(seq_atom, static_cast<uint32_t>(pos + 1), std::memory_order_release);
                    return true;
                }
            }
            else if (diff < 0)
            {
                return false;  // Full
            }
            else
            {
                // CAS 失敗・競合 — 最新 enqueuePos を再観測して再試行
                pos = convo::consumeAtomic(enqueuePos_, std::memory_order_acquire);
            }
        }
    }

    // Single Consumer 専用。戻り値 false = empty（または producer hole — 次回 poll で再試行）。
    bool pop(T& item) noexcept
    {
        const uint32_t pos = convo::consumeAtomic(dequeuePos_, std::memory_order_acquire);
        auto& seq_atom = sequences_[pos & kMask];
        const uint32_t seq = convo::consumeAtomic(seq_atom, std::memory_order_acquire);
        const int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - static_cast<uint32_t>(pos + 1)));

        if (diff != 0)
            return false;  // Empty / producer hole（seq mismatch → skip）

        item = entries_[pos & kMask];
        convo::publishAtomic(seq_atom, static_cast<uint32_t>(pos + Capacity), std::memory_order_release);
        convo::publishAtomic(dequeuePos_, static_cast<uint32_t>(pos + 1), std::memory_order_release);
        return true;
    }

    // Best-effort 占有数（acquire 観測）。正確な値は単一 Producer/Consumer 時のみ。
    [[nodiscard]] size_t sizeApprox() const noexcept
    {
        const uint32_t w = convo::consumeAtomic(enqueuePos_, std::memory_order_acquire);
        const uint32_t d = convo::consumeAtomic(dequeuePos_, std::memory_order_acquire);
        return static_cast<size_t>(w - d);
    }

    // シャットダウン時: 全 Producer/Consumer 停止後にのみ呼び出すこと（LockFreeRingBuffer と同契約）。
    void clear() noexcept
    {
        convo::publishAtomic(enqueuePos_, uint32_t{0}, std::memory_order_release);
        convo::publishAtomic(dequeuePos_, uint32_t{0}, std::memory_order_release);
        for (size_t i = 0; i < Capacity; ++i)
            convo::publishAtomic(sequences_[i], static_cast<uint32_t>(i), std::memory_order_release);
    }

#ifdef CONVO_MPSC_TEST_HOOKS
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // ★ work88 (P2-3): producer-hole 再現テスト API（CONVO_TESTING 定義時のみ）。
    //   位置ベース: testHoleBlockPos_ と一致する reservation 位置の push が、payload
    //   書込み前に testHoleGate_ が true になるまで待機する。デフォルト
    //   (UINT32_MAX = kNoBlock) では pass-through のため、本番/既存テストには無影響。
    //   使用例: ① testResetHole() + testSetHoleBlock(0) を producer 起動前に設定
    //           ② testHoleBlocked() で hook 到達（= reservation 済み・未公開）を待機
    //           ③ pop が false（producer hole）であることを検証
    //           ④ testReleaseHole() で payload 公開を許可 → pop が true に
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    static constexpr uint32_t kNoBlock = (std::numeric_limits<uint32_t>::max)();

    /** reservation 位置 pos を producer-hole としてブロックする（producer 起動前に設定すること）。 */
    void testSetHoleBlock(uint32_t blockPos) noexcept
    {
        convo::publishAtomic(testHoleBlockPos_, blockPos, std::memory_order_release);
    }

    /** ブロック位置を解除（以降の push は pass-through）。 */
    void testClearHoleBlock() noexcept
    {
        convo::publishAtomic(testHoleBlockPos_, kNoBlock, std::memory_order_release);
    }

    /** ブロック中の producer の payload 書込みを許可する（publication order を解放）。 */
    void testReleaseHole() noexcept
    {
        convo::publishAtomic(testHoleGate_, true, std::memory_order_release);
    }

    /** フック状態を初期化（新規テストごとに呼ぶこと）。 */
    void testResetHole() noexcept
    {
        convo::publishAtomic(testHoleReady_, false, std::memory_order_release);
        convo::publishAtomic(testHoleGate_, false, std::memory_order_release);
    }

    /** producer が hole でブロック中（reservation 済み・payload 未公開）か。 */
    [[nodiscard]] bool testHoleBlocked() const noexcept
    {
        return convo::consumeAtomic(testHoleReady_, std::memory_order_acquire);
    }
#endif // CONVO_MPSC_TEST_HOOKS

private:
    alignas(64) std::atomic<uint32_t> sequences_[Capacity];
    alignas(64) T entries_[Capacity];
    alignas(64) std::atomic<uint32_t> enqueuePos_{0};
    alignas(64) std::atomic<uint32_t> dequeuePos_{0};
#ifdef CONVO_MPSC_TEST_HOOKS
    // ★ work88 (P2-3): producer-hole テストフック用メンバ（CONVO_TESTING 定義時のみ存在）。
    alignas(64) std::atomic<bool> testHoleReady_{false};      // producer が hook に到達した通知
    alignas(64) std::atomic<bool> testHoleGate_{false};       // payload 書込み許可ゲート
    alignas(64) std::atomic<uint32_t> testHoleBlockPos_{kNoBlock};  // ブロック対象 reservation 位置
#endif
};

#ifdef _MSC_VER
#  pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#endif
