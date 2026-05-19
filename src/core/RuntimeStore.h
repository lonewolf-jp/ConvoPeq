#pragma once

#include <atomic>
#include <cassert>
#include <type_traits>
#include <utility>

#include "audioengine/AtomicAccess.h"

namespace convo {

template <typename T, typename Owner>
class RuntimeStore final
{
public:
    static_assert(std::is_class_v<Owner>, "RuntimeStore Owner must be a class type");

    class WriteAccess final
    {
    public:
        WriteAccess(const WriteAccess&) = delete;
        WriteAccess& operator=(const WriteAccess&) = delete;
        WriteAccess(WriteAccess&& other) noexcept
            : store_(std::exchange(other.store_, nullptr))
        {
        }

        WriteAccess& operator=(WriteAccess&& other) noexcept
        {
            if (this != &other)
                store_ = std::exchange(other.store_, nullptr);
            return *this;
        }

        [[nodiscard]] T* publishAndSwap(T* next) noexcept
        {
            if (store_ == nullptr)
            {
                // moved-from write handle の誤用は Debug で検知し、Release では no-op へフォールバックする。
                assert(false && "RuntimeStore::WriteAccess used after move");
                return nullptr;
            }
            // publish side: release を含む acq_rel exchange で world を公開する。
            return exchangeAtomic(store_->current, next, std::memory_order_acq_rel);
        }

    private:
        friend class RuntimeStore;

        explicit WriteAccess(RuntimeStore& store) noexcept
            : store_(&store)
        {
        }

        RuntimeStore* store_ = nullptr;
    };

    static_assert(!std::is_copy_constructible_v<WriteAccess>, "RuntimeStore::WriteAccess must remain move-only");
    static_assert(!std::is_copy_assignable_v<WriteAccess>, "RuntimeStore::WriteAccess must remain move-only");
    static_assert(!std::is_default_constructible_v<WriteAccess>, "RuntimeStore::WriteAccess must not be default-constructible");
    static_assert(std::is_move_constructible_v<WriteAccess>, "RuntimeStore::WriteAccess must be move-constructible");
    static_assert(std::is_move_assignable_v<WriteAccess>, "RuntimeStore::WriteAccess must be move-assignable");
    static_assert(std::is_nothrow_move_constructible_v<WriteAccess>, "RuntimeStore::WriteAccess move ctor must stay noexcept");
    static_assert(std::is_nothrow_move_assignable_v<WriteAccess>, "RuntimeStore::WriteAccess move assign must stay noexcept");

    RuntimeStore() = default;
    RuntimeStore(const RuntimeStore&) = delete;
    RuntimeStore& operator=(const RuntimeStore&) = delete;
    RuntimeStore(RuntimeStore&&) = delete;
    RuntimeStore& operator=(RuntimeStore&&) = delete;

    [[nodiscard]] const T* observe() const noexcept
    {
        // 返却ポインタは borrow（非所有）参照。寿命管理は publish/retire 側の契約に従う。
        // observe side: acquire load で publish 済み world の可視性を保証する。
        return consumeAtomic(current, std::memory_order_acquire);
    }

private:
    // write authority は owner のみが取得可能（rule4: publish authority 集約）。
    friend Owner;

    [[nodiscard]] WriteAccess acquireWriteAccess() noexcept
    {
        return WriteAccess(*this);
    }

    std::atomic<T*> current { nullptr };
};

} // namespace convo
