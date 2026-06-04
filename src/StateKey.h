#pragma once
#include <cstdint>
#include <tuple>
#include <functional>

struct StateKey {
    int sampleRateHz = 0;
    int bitDepth = 0;
    int learningMode = 0; // enum値をintで

    bool operator==(const StateKey& other) const noexcept {
        return std::tie(sampleRateHz, bitDepth, learningMode) ==
               std::tie(other.sampleRateHz, other.bitDepth, other.learningMode);
    }
};

namespace std {
    template<>
    struct hash<StateKey> {
        // FNV-1a hash for better distribution
        std::size_t operator()(const StateKey& k) const noexcept {
            constexpr std::size_t kFnvOffsetBasis = 14695981039346656037ULL;
            constexpr std::size_t kFnvPrime = 1099511628211ULL;

            auto fnv1a = [](std::size_t h, int v) noexcept -> std::size_t {
                const auto* bytes = reinterpret_cast<const unsigned char*>(&v);
                for (std::size_t i = 0; i < sizeof(int); ++i)
                {
                    h ^= static_cast<std::size_t>(bytes[i]);
                    h *= kFnvPrime;
                }
                return h;
            };

            std::size_t h = kFnvOffsetBasis;
            h = fnv1a(h, k.sampleRateHz);
            h = fnv1a(h, k.bitDepth);
            h = fnv1a(h, k.learningMode);
            return h;
        }
    };
}
