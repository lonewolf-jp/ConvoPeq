#pragma once

// ★ WORK113 DQ-V2 v2 (2026-11-19 Owner 承認): テスト専用の破棄観測。
//
//   CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY を定義するテストターゲットだけが include する。
//   製品ビルドでは include 自体が #if の内側にあるため、このヘッダは読まれない。
//
//   観測対象は「DSPCore の破棄が再試行されたか」である。ポインタの値は照合にだけ使い、
//   解放後の値を識別子として保存しない。識別子は setTarget() が発行する単調増加 ID で、
//   テストはその ID と計数だけを読む。
//
//   観測点は destroyDSPCoreNodeRetryable() の
//     - 入口（attempts）
//     - Deferred を返す直前（deferredAttempts）
//     - 破棄完了から aligned_free の直前（destroyedAttempts / released）
//   である。どの呼び出しも有効なポインタで行われる（解放後に触らない）。

#include <atomic>
#include <cstdint>

#include "audioengine/AtomicAccess.h"

namespace convo::testobs
{

struct RetryableDestroyLog
{
    // 照合に使うポインタ。clearTarget() で必ず nullptr に戻す。
    std::atomic<const void*> target { nullptr };
    // setTarget() が発行する識別子（0 = 未設定）。ポインタ値の再利用に依存しない。
    std::atomic<std::uint64_t> targetId { 0 };
    std::atomic<std::uint32_t> attempts { 0 };          // 入口の呼び出し回数
    std::atomic<std::uint32_t> deferredAttempts { 0 };  // Deferred で戻った回数
    std::atomic<std::uint32_t> destroyedAttempts { 0 }; // Destroyed で戻った回数
    std::atomic<bool> released { false };               // 解放経路に到達した（解放直前）
};

inline RetryableDestroyLog& log() noexcept
{
    static RetryableDestroyLog l;
    return l;
}

// 識別子の発行元（単調増加・プロセス寿命）。
inline std::atomic<std::uint64_t>& nextTargetId() noexcept
{
    static std::atomic<std::uint64_t> n { 1 };
    return n;
}

// 観測対象を設定し、計数を 0 に戻す。戻り値はこの対象の識別子。
inline std::uint64_t setTarget (const void* p) noexcept
{
    auto& l = log();
    const std::uint64_t id =
        convo::fetchAddAtomic (nextTargetId(), std::uint64_t{1}, std::memory_order_acq_rel) + 1;
    convo::publishAtomic (l.attempts, std::uint32_t{0}, std::memory_order_relaxed);
    convo::publishAtomic (l.deferredAttempts, std::uint32_t{0}, std::memory_order_relaxed);
    convo::publishAtomic (l.destroyedAttempts, std::uint32_t{0}, std::memory_order_relaxed);
    convo::publishAtomic (l.released, false, std::memory_order_relaxed);
    convo::publishAtomic (l.targetId, id, std::memory_order_release);
    convo::publishAtomic (l.target, p, std::memory_order_release);
    return id;
}

// 観測対象を解除する。ポインタを保持し続けない（解放・アドレス再利用の前に必ず呼ぶ）。
inline void clearTarget() noexcept
{
    auto& l = log();
    convo::publishAtomic (l.target, static_cast<const void*>(nullptr), std::memory_order_release);
    convo::publishAtomic (l.targetId, std::uint64_t{0}, std::memory_order_release);
}

inline bool hasTarget() noexcept
{
    return convo::consumeAtomic (log().target, std::memory_order_acquire) != nullptr;
}

inline std::uint64_t currentTargetId() noexcept
{
    return convo::consumeAtomic (log().targetId, std::memory_order_acquire);
}

inline bool matches (const void* p) noexcept
{
    const void* t = convo::consumeAtomic (log().target, std::memory_order_acquire);
    return t != nullptr && t == p;
}

inline void noteEntry (const void* p) noexcept
{
    if (matches (p))
        (void) convo::fetchAddAtomic (log().attempts, std::uint32_t{1}, std::memory_order_acq_rel);
}

inline void noteDeferred (const void* p) noexcept
{
    if (matches (p))
        (void) convo::fetchAddAtomic (log().deferredAttempts, std::uint32_t{1},
                                      std::memory_order_acq_rel);
}

inline void noteDestroyed (const void* p) noexcept
{
    if (matches (p))
        (void) convo::fetchAddAtomic (log().destroyedAttempts, std::uint32_t{1},
                                      std::memory_order_acq_rel);
}

// 解放経路に到達したことの記録。aligned_free の直前に呼ぶ（ポインタはまだ有効）。
inline void noteReleaseImminent (const void* p) noexcept
{
    if (matches (p))
        convo::publishAtomic (log().released, true, std::memory_order_release);
}

} // namespace convo::testobs
