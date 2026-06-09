#pragma once

#include <cstdint>
#include <chrono>

namespace convo::isr {

// ShutdownDrainToken: shutdown drain 許可を表す move-only トークン。
// ★ move-only (copy不可)
// ★ consume() は一回限り利用
// ★ consume() 内で now > expiration 検査必須
struct ShutdownDrainToken {
    uint64_t engineInstanceId{0};
    uint64_t shutdownGeneration{0};
    uint64_t generation{0};
    uint64_t shutdownEpoch{0};
    uint64_t expiration{0};

    ShutdownDrainToken() noexcept = default;

    ShutdownDrainToken(const ShutdownDrainToken&) = delete;
    ShutdownDrainToken& operator=(const ShutdownDrainToken&) = delete;
    ShutdownDrainToken(ShutdownDrainToken&&) noexcept = default;
    ShutdownDrainToken& operator=(ShutdownDrainToken&&) noexcept = default;

    // consume: トークンを消費して有効性を確認。
    // 一回限り利用 (consume 後は無効化)
    [[nodiscard]] bool consume() noexcept {
        if (!valid_) return false;
        valid_ = false;
        if (expiration == 0) return true;  // 期限なし
        const auto now = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        if (now > expiration) return false;  // 期限切れ
        return true;
    }

    [[nodiscard]] bool isValid() const noexcept { return valid_; }

private:
    bool valid_ = true;
};

// ShutdownScope: shutdown 中のみ submitShutdownDrain() を許可するスコープ。
// ★ shutdownGeneration 拘束 (thread非依存)
// ★ Token move-only + consume() 一回限り + 期限切れ検査
class ShutdownScope {
public:
    ShutdownScope(uint64_t engineInstanceId,
                  uint64_t shutdownGeneration,
                  uint64_t expiration = 0) noexcept
        : engineInstanceId_(engineInstanceId)
        , shutdownGeneration_(shutdownGeneration)
        , expiration_(expiration)
        , active_(true)
    {
    }

    ~ShutdownScope() noexcept {
        active_ = false;
    }

    ShutdownScope(const ShutdownScope&) = delete;
    ShutdownScope& operator=(const ShutdownScope&) = delete;
    ShutdownScope(ShutdownScope&&) noexcept = default;
    ShutdownScope& operator=(ShutdownScope&&) noexcept = default;

    // createToken: 有効な ShutdownDrainToken を生成。
    // ★ Scope active 中のみ生成可能
    [[nodiscard]] ShutdownDrainToken createToken(uint64_t generation,
                                                  uint64_t shutdownEpoch) noexcept {
        if (!active_) return ShutdownDrainToken{};
        return ShutdownDrainToken{
            .engineInstanceId = engineInstanceId_,
            .shutdownGeneration = shutdownGeneration_,
            .generation = generation,
            .shutdownEpoch = shutdownEpoch,
            .expiration = expiration_
        };
    }

    [[nodiscard]] bool isActive() const noexcept { return active_; }
    [[nodiscard]] uint64_t shutdownGeneration() const noexcept { return shutdownGeneration_; }
    [[nodiscard]] uint64_t engineInstanceId() const noexcept { return engineInstanceId_; }

private:
    uint64_t engineInstanceId_;
    uint64_t shutdownGeneration_;
    uint64_t expiration_;
    bool active_;
};

} // namespace convo::isr
