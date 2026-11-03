#pragma once

//==============================================================================
// U3 (R-107 Stage 3): RT 経路の新規確保を検出するプローブ
//
//   **テストバイナリ（AudioEngineHarness）にのみ**リンクする。製品ターゲットには入れない。
//   既定 OFF（CONVOPEQ_RT_ALLOC_PROBE=1 で有効化）。
//
//   置換する operator new / delete は RtAllocProbe.cpp の 1 箇所だけに置く
//   （V11EngineRetireProtectionTests.cpp の a2probe はこの名前空間の別名にする）。
//
//   計数は `Scope` が立っている区間のみ。音声コールバック（harness の audioLoop）は
//   `RtScope`（= Scope の別名）で囲むため、RT 区間の確保だけが数えられる。
//
//   RT 安全側の設計:
//     - 計数はロックフリー（atomic の fetch_add のみ）。確保・I/O・書式化をしない
//     - 有効フラグの atomic load 1 回で素通りする（OFF 時は従来と同一コスト）
//==============================================================================

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rtalloc
{
inline std::atomic<bool>& enabledRef() noexcept
{
    static std::atomic<bool> on{ false };
    return on;
}

inline std::atomic<std::uint64_t>& countRef() noexcept
{
    static std::atomic<std::uint64_t> c{ 0 };
    return c;
}

inline std::atomic<std::uint64_t>& byteRef() noexcept
{
    static std::atomic<std::uint64_t> b{ 0 };
    return b;
}

inline std::atomic<std::uint64_t>& maxByteRef() noexcept
{
    static std::atomic<std::uint64_t> m{ 0 };
    return m;
}

inline thread_local int t_depth = 0;   // RT 区間の深さ（POD・定数初期化・デストラクタなし）

[[nodiscard]] inline bool enabled() noexcept
{
    return enabledRef().load(std::memory_order_acquire);
}

inline void setEnabledForTest(bool on) noexcept
{
    enabledRef().store(on, std::memory_order_release);
}

[[nodiscard]] inline bool active() noexcept { return t_depth > 0; }

inline void reset() noexcept
{
    countRef().store(0, std::memory_order_relaxed);
    byteRef().store(0, std::memory_order_relaxed);
    maxByteRef().store(0, std::memory_order_relaxed);
}

inline void resetForTest() noexcept { reset(); }

[[nodiscard]] inline std::uint64_t count() noexcept
{
    return countRef().load(std::memory_order_relaxed);
}

[[nodiscard]] inline std::uint64_t bytes() noexcept
{
    return byteRef().load(std::memory_order_relaxed);
}

[[nodiscard]] inline std::uint64_t maxBytes() noexcept
{
    return maxByteRef().load(std::memory_order_relaxed);
}

// 確保 1 回を記録する（operator new から呼ばれる）。
//   RT 区間（Scope 内）でのみ計数する。既定でも計数は行うが、出力は測定モードのみ
//   （既定 OFF = [U3] を出さない・挙動を変えない）。
inline void noteAlloc(std::size_t n) noexcept
{
    if (t_depth <= 0)
        return;

    countRef().fetch_add(1, std::memory_order_relaxed);
    byteRef().fetch_add(static_cast<std::uint64_t>(n), std::memory_order_relaxed);

    std::uint64_t cur = maxByteRef().load(std::memory_order_relaxed);
    const auto value = static_cast<std::uint64_t>(n);
    while (value > cur
           && !maxByteRef().compare_exchange_weak(cur, value,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed))
    {
    }
}

struct Scope
{
    Scope() noexcept { ++t_depth; }
    ~Scope() noexcept { if (t_depth > 0) --t_depth; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

using RtScope = Scope;

// 環境変数 CONVOPEQ_RT_ALLOC_PROBE=1 で有効化（非 RT・1 回）。
void enableFromEnvironment() noexcept;

// [U3] 行を 1 行出力する（非 RT）。
void printReport() noexcept;
} // namespace rtalloc
