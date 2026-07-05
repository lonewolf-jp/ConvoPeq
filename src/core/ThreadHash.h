#pragma once
#include <thread>
#include <cstdint>

namespace convo {

/// Audioスレッドに最適化された thread::id -> uint64_t キャッシュ。
/// 一度計算したハッシュ値を thread_local に保持する。
inline uint64_t cachedThreadHash() noexcept
{
    // RT-SAFE: POD, const, no destructor, once/thread, avoids std::hash per callback (ISR perf)
    static thread_local const uint64_t s_cachedHash = // NOLINT(thread-local) RT-SAFE:
        static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return s_cachedHash;
}

} // namespace convo
