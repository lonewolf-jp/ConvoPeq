#pragma once
#include <thread>
#include <cstdint>
#include <atomic>

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

/// ★ work92 B-4 (big 2-6): プロセス単調 ID 採番 — RCU reader token / EpochDomain
///   ownerThreadId 専用の thread 識別子。
///
///   cachedThreadHash()（std::hash<std::thread::id>）は衝突が理論上排除されず、
///   衝突時は「別スレッドが同一 token を名乗る」→ RCU reader 二重登録 /
///   ownerThreadId 誤一致 → epoch が進まず reclaim 停止の潜在。
///   本関数はカウンタを 1 起点で単調採番する（0 は無効値予約 — EpochDomain.h:571
///   の ownerThreadId 初期値 0 と整合・RCUReader exit() の 0 リセットとも整合）。
///
///   thread_local 契約（G3 確定・AC-B4-3）:
///   - steady-state: thread_local 読み取り 1 回・追加コストゼロ
///   - first-init: そのスレッドの初回 enter() で 1 回のみ発生
///     （audio thread = 最初の callback block）。既存 cachedThreadHash() と
///     同一タイミング・同一の thread_local 機構に依存し、新規 blocking/allocation なし。
///   - カウンタ fetch_add はスレッド寿命あたり 1 回のみ（first-init 内）。
inline uint64_t acquireUniqueThreadId() noexcept
{
    static thread_local const uint64_t s_uniqueId = // NOLINT(thread-local) RT-SAFE: 同上
        []() noexcept {
            // fetch_add(relaxed): 採番の一意性のみが必要（順序・可視性の契約なし）。
            // 1 起点静止（0 は無効値予約）: 初回 fetch_add が 0 を返すため +1。
            static std::atomic<uint64_t> s_counter { 0 };
            return s_counter.fetch_add(1, std::memory_order_relaxed) + 1;
        }();
    return s_uniqueId;
}

} // namespace convo
