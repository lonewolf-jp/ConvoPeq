#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace convo
{
template <typename T>
class NonOwningPtr
{
public:
    constexpr NonOwningPtr() noexcept = default;
    constexpr NonOwningPtr(std::nullptr_t) noexcept : bits(0) {}
    explicit constexpr NonOwningPtr(T* ptr) noexcept
        : bits(static_cast<std::uintptr_t>(reinterpret_cast<std::uintptr_t>(ptr))) {}

    constexpr NonOwningPtr& operator=(std::nullptr_t) noexcept
    {
        bits = 0;
        return *this;
    }

    constexpr NonOwningPtr& operator=(T* ptr) noexcept
    {
        bits = static_cast<std::uintptr_t>(reinterpret_cast<std::uintptr_t>(ptr));
        return *this;
    }

    constexpr T* get() const noexcept
    {
        return reinterpret_cast<T*>(bits);
    }

    constexpr explicit operator bool() const noexcept
    {
        return bits != 0;
    }

    constexpr operator T*() const noexcept
    {
        return get();
    }

private:
    std::uintptr_t bits = 0;
};

template <typename T, typename U,
          typename = std::enable_if_t<std::is_convertible_v<U, T>>>
inline void publishAtomic(std::atomic<T>& dst,
                          U&& value,
                          std::memory_order order = std::memory_order_seq_cst) noexcept
{
    std::atomic_store_explicit(&dst, static_cast<T>(std::forward<U>(value)), order);
}

template <typename T>
inline T consumeAtomic(const std::atomic<T>& src,
                       std::memory_order order = std::memory_order_seq_cst) noexcept
{
    return std::atomic_load_explicit(&src, order);
}

template <typename T, typename U,
          typename = std::enable_if_t<std::is_convertible_v<U, T>>>
inline T exchangeAtomic(std::atomic<T>& dst,
                        U&& value,
                        std::memory_order order = std::memory_order_seq_cst) noexcept
{
    return std::atomic_exchange_explicit(&dst, static_cast<T>(std::forward<U>(value)), order);
}

template <typename T>
inline bool compareExchangeAtomic(std::atomic<T>& dst,
                                  T& expected,
                                  T desired,
                                  std::memory_order success = std::memory_order_seq_cst,
                                  std::memory_order failure = std::memory_order_seq_cst) noexcept
{
    return std::atomic_compare_exchange_strong_explicit(&dst,
                                                        &expected,
                                                        desired,
                                                        success,
                                                        failure);
}

template <typename T, typename U,
          typename = std::enable_if_t<std::is_integral_v<T> && std::is_convertible_v<U, T>>>
inline T fetchAddAtomic(std::atomic<T>& dst,
                        U value,
                        std::memory_order order = std::memory_order_seq_cst) noexcept
{
    return std::atomic_fetch_add_explicit(&dst, static_cast<T>(value), order);
}

template <typename T, typename U,
          typename = std::enable_if_t<std::is_integral_v<T> && std::is_convertible_v<U, T>>>
inline T fetchSubAtomic(std::atomic<T>& dst,
                        U value,
                        std::memory_order order = std::memory_order_seq_cst) noexcept
{
    return std::atomic_fetch_sub_explicit(&dst, static_cast<T>(value), order);
}

template <typename T, typename U,
          typename = std::enable_if_t<std::is_integral_v<T> && std::is_convertible_v<U, T>>>
inline T fetchOrAtomic(std::atomic<T>& dst,
                       U value,
                       std::memory_order order = std::memory_order_seq_cst) noexcept
{
    return std::atomic_fetch_or_explicit(&dst, static_cast<T>(value), order);
}

template <typename T, typename U,
          typename = std::enable_if_t<std::is_integral_v<T> && std::is_convertible_v<U, T>>>
inline T fetchAndAtomic(std::atomic<T>& dst,
                        U value,
                        std::memory_order order = std::memory_order_seq_cst) noexcept
{
    return std::atomic_fetch_and_explicit(&dst, static_cast<T>(value), order);
}

// Pointer-specialized aliases for readability in publication boundaries.
template <typename T>
inline void publishAtomicPtr(std::atomic<T*>& dst,
                             T* value,
                             std::memory_order order = std::memory_order_seq_cst) noexcept
{
    publishAtomic(dst, value, order);
}

template <typename T>
inline T* consumeAtomicPtr(const std::atomic<T*>& src,
                           std::memory_order order = std::memory_order_seq_cst) noexcept
{
    return consumeAtomic(src, order);
}

template <typename T>
inline T* exchangeAtomicPtr(std::atomic<T*>& dst,
                            T* value,
                            std::memory_order order = std::memory_order_seq_cst) noexcept
{
    return exchangeAtomic(dst, value, order);
}
} // namespace convo
