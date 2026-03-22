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
        std::size_t operator()(const StateKey& k) const noexcept {
            std::size_t h1 = std::hash<int>{}(k.sampleRateHz);
            std::size_t h2 = std::hash<int>{}(k.bitDepth);
            std::size_t h3 = std::hash<int>{}(k.learningMode);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}
