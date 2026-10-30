#pragma once

//==============================================================================
// R-06 / U1: RT 実時間計測のための固定長リング（**純追加**）
//
// 役割:
//   RT スレッドから「起きたこと」を記録するための、固定長・静的確保・ロックなしのリング。
//   RT 側で許されるのは push() の書き込みのみ。確保・ロック・書式化・I/O を行わない。
//
// U1 の範囲（Owner 指示 2026-10-30）:
//   - 本ファイルの追加のみ。**呼び出し点は一切追加しない**。
//   - RT 経路・既存音声処理・同期構造は変更しない。
//   - したがって本ヘッダは U1 の時点では製品コードから参照されない（U2 以降で接続する）。
//
// 有効化:
//   既定 OFF。環境変数 CONVOPEQ_RT_MEASURE=1 のときだけ有効。判定は最初の呼び出しで 1 回だけ行い、
//   以後は static に固定する（RT 中に getenv を呼ばない）。
//
// 満杯時:
//   上書きしない。あふれた分は捨てて dropped に計上する（測定値は「下限」であることを明示するため）。
//
// スレッド安全:
//   単一書き手（RT）／単一読み手（非 RT）を前提とする。
//   push() は write_ の release store、drain() は read_ の release store で公開する。
//==============================================================================

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "AtomicAccess.h"   // list.md [3.1]: \u88fd\u54c1\u30b3\u30fc\u30c9\u3067\u306f atomic \u306e\u30c9\u30c3\u30c8\u8a95\u751f\u3092\u76f4\u63a5\u547c\u3093\u3067\u306f\u306a\u3089\u306a\u3044\uff08convo:: \u30d8\u30eb\u30d1\u7d4c\u7531\uff09

namespace convo {
namespace rtmeasure {

// 観測イベントの種類。値は診断出力の互換性のため明示する。
enum EventKind : std::uint32_t
{
    kBlockBegin = 1,   // arg = ブロックサイズ
    kBlockEnd   = 2,   // arg = 未使用（時刻差は非 RT 側で計算）
    kAlloc      = 3,   // arg = 確保サイズの下位 32bit
    kFree       = 4,   // arg = 未使用
    kLockWait   = 5,   // arg = 待機 µs（下限）
    kDropped    = 6    // 予約（リング自体は dropped カウンタで計上する）
};

struct Event
{
    std::uint64_t tsc  = 0;
    std::uint32_t kind = 0;
    std::uint32_t arg  = 0;
};

// 計測が有効か。最初の呼び出しで環境変数を 1 回だけ読み、以後は固定する。
[[nodiscard]] inline bool enabled() noexcept
{
    static const bool on = []() {
        const char* v = std::getenv("CONVOPEQ_RT_MEASURE");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

// 固定長リング。静的領域に確保し、RT から安全に書き込める。
class Ring
{
public:
    static constexpr std::size_t kCapacity = 8192;

    // RT 側から呼ぶ唯一の関数。ロック・確保・I/O・書式化を行わない。
    void push(std::uint32_t kind, std::uint32_t arg, std::uint64_t tsc) noexcept
    {
        const std::uint64_t w = convo::consumeAtomic(write_, std::memory_order_relaxed);
        const std::uint64_t r = convo::consumeAtomic(read_, std::memory_order_acquire);
        if (w - r >= kCapacity)
        {
            // 上書きしない（古いイベントを残す）。捨てたことはカウンタで示す。
            convo::fetchAddAtomic(dropped_, std::uint64_t{1}, std::memory_order_relaxed);
            return;
        }
        storage_[static_cast<std::size_t>(w % kCapacity)] = Event{ tsc, kind, arg };
        convo::publishAtomic(write_, w + 1, std::memory_order_release);
    }

    // 非 RT 側で回収する。out に n 件までコピーし、コピーした件数を返す。
    std::size_t drain(Event* out, std::size_t n) noexcept
    {
        if (out == nullptr || n == 0)
            return 0;
        const std::uint64_t w = convo::consumeAtomic(write_, std::memory_order_acquire);
        std::uint64_t r = convo::consumeAtomic(read_, std::memory_order_relaxed);
        std::size_t copied = 0;
        while (r < w && copied < n)
        {
            out[copied] = storage_[static_cast<std::size_t>(r % kCapacity)];
            ++copied;
            ++r;
        }
        convo::publishAtomic(read_, r, std::memory_order_release);
        return copied;
    }

    [[nodiscard]] std::uint64_t written() const noexcept
    {
        return convo::consumeAtomic(write_, std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t consumed() const noexcept
    {
        return convo::consumeAtomic(read_, std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t dropped() const noexcept
    {
        return convo::consumeAtomic(dropped_, std::memory_order_relaxed);
    }
    [[nodiscard]] std::size_t pending() const noexcept
    {
        const std::uint64_t w = convo::consumeAtomic(write_, std::memory_order_acquire);
        const std::uint64_t r = convo::consumeAtomic(read_, std::memory_order_acquire);
        return static_cast<std::size_t>(w - r);
    }
    void reset() noexcept
    {
        convo::publishAtomic(write_, std::uint64_t{0}, std::memory_order_release);
        convo::publishAtomic(read_, std::uint64_t{0}, std::memory_order_release);
        convo::publishAtomic(dropped_, std::uint64_t{0}, std::memory_order_release);
    }

    // プロセスに 1 つのリング（静的領域）。
    static Ring& instance() noexcept
    {
        static Ring ring;
        return ring;
    }

private:
    Event storage_[kCapacity]{};
    std::atomic<std::uint64_t> write_{0};
    std::atomic<std::uint64_t> read_{0};
    std::atomic<std::uint64_t> dropped_{0};
};

} // namespace rtmeasure
} // namespace convo
