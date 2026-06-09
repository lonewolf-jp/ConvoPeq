#pragma once

#include <thread>
#include <type_traits>
#include <cstdint>

#include "RCUReader.h"
#include "GlobalSnapshot.h"

namespace convo {

// ObserveToken (P0-1 formalization)
// ---------------------------------
// 許可責務:
// - IEpochProvider reader guard を保持し、observe enter/exit をスコープ化する
// - 現在スレッドに束縛された snapshot pointer を参照として提供する
//
// 禁止責務:
// - publish / retire の実行
// - RuntimeGraph mutation / repair
// - cache ownership / lifetime 管理
//
// 注記:
// - 本型は bridge runtime の統制トークンであり、挙動変更を伴わない。
// - P0-1 では API 互換を維持し、ObservedRuntime を実体として残す。
struct ObservedRuntime
{
    // [work21 Phase-E] RCUReader 経由で reader enter/exit
    explicit ObservedRuntime(RCUReader& reader) noexcept
        : guard(reader)
#ifndef NDEBUG
        , ownerThreadId(std::this_thread::get_id())
#endif
    {
    }

    ObservedRuntime(const ObservedRuntime&) = delete;
    ObservedRuntime& operator=(const ObservedRuntime&) = delete;
    ObservedRuntime(ObservedRuntime&&) noexcept = default;
    ObservedRuntime& operator=(ObservedRuntime&&) noexcept = delete;

    const GlobalSnapshot* get() const noexcept
    {
#ifndef NDEBUG
        if (ownerThreadId != std::this_thread::get_id())
            return nullptr;
#endif
        return ptr;
    }

    // ★ P2-3: generation 簡易 getter
    [[nodiscard]] uint64_t generation() const noexcept
    {
        return ptr ? ptr->generation : 0;
    }

    explicit operator bool() const noexcept
    {
#ifndef NDEBUG
        return ownerThreadId == std::this_thread::get_id() && ptr != nullptr;
#else
        return ptr != nullptr;
#endif
    }

    RCUReaderGuard guard;
    const GlobalSnapshot* ptr = nullptr;
#ifndef NDEBUG
    std::thread::id ownerThreadId;
#endif
};

static_assert(!std::is_copy_constructible_v<ObservedRuntime>);
static_assert(std::is_move_constructible_v<ObservedRuntime>);

// 概念名エイリアス（non-breaking）
using ObserveToken = ObservedRuntime;

static_assert(std::is_same_v<ObserveToken, ObservedRuntime>);
static_assert(!std::is_copy_constructible_v<ObserveToken>);
static_assert(std::is_move_constructible_v<ObserveToken>);

} // namespace convo
