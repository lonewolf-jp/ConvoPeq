#pragma once

#include <thread>
#include <type_traits>

#include "EpochDomain.h"
#include "GlobalSnapshot.h"

namespace convo {

// ObserveToken (P0-1 formalization)
// ---------------------------------
// 許可責務:
// - EpochDomain reader guard を保持し、observe enter/exit をスコープ化する
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
    explicit ObservedRuntime(EpochDomain& domain, int readerIndex) noexcept
        : guard(domain, readerIndex), ownerThreadId(std::this_thread::get_id())
    {
    }

    ObservedRuntime(const ObservedRuntime&) = delete;
    ObservedRuntime& operator=(const ObservedRuntime&) = delete;
    ObservedRuntime(ObservedRuntime&&) noexcept = default;
    ObservedRuntime& operator=(ObservedRuntime&&) noexcept = default;

    const GlobalSnapshot* get() const noexcept
    {
        if (ownerThreadId != std::this_thread::get_id())
            return nullptr;
        return ptr;
    }

    explicit operator bool() const noexcept
    {
        return ownerThreadId == std::this_thread::get_id() && ptr != nullptr;
    }

    EpochDomainReaderGuard guard;
    const GlobalSnapshot* ptr = nullptr;
    std::thread::id ownerThreadId;
};

static_assert(!std::is_copy_constructible_v<ObservedRuntime>);
static_assert(std::is_move_constructible_v<ObservedRuntime>);

// 概念名エイリアス（non-breaking）
using ObserveToken = ObservedRuntime;

static_assert(std::is_same_v<ObserveToken, ObservedRuntime>);
static_assert(!std::is_copy_constructible_v<ObserveToken>);
static_assert(std::is_move_constructible_v<ObserveToken>);

} // namespace convo
