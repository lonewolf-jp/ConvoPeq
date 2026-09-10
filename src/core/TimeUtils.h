#pragma once
#include <chrono>
#include <cstdint>

namespace convo {

/**
 * 現在時刻をマイクロ秒で取得（std::chrono::steady_clock ベース）
 *
 * 配置理由: core/ は audioengine/ より低レイヤであり、
 * EpochDomain（core/）と RuntimeHealthMonitor（audioengine/）の
 * 両方から利用可能。
 */
inline uint64_t getCurrentTimeUs() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}

/**
 * ★ work92 B-8 (big 2-9): saturating subtraction — a >= b を保証する減算。
 *
 * クロック逆転（ steady_clock は正順保証だが複数取得点の並び替え・
 * 診断経路での古い時刻再利用等）で a < b になった場合、通常の減算は
 * uint64 巨大値に wrap し、診断表示（callbackUs / intervalUs）が破壊される。
 * RT-safe（分岐のみ・例外なし）。
 */
inline uint64_t saturatingSubUs(uint64_t a, uint64_t b) noexcept {
    return (a >= b) ? (a - b) : uint64_t{0};
}

} // namespace convo
