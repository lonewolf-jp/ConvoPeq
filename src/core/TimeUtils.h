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

} // namespace convo
