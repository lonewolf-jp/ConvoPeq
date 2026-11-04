//============================================================================
// CorruptSampleGuard.h
//
// RES2-02 (O-20): one definition of the "corrupt magnitude" limit, shared by
// the scalar and the AVX2 guards. Previously the scalar guard compared against
// 2^53 while the AVX2 guard compared against 1e20, so the same sample could be
// treated as corruption on one path and as normal data on the other.
//
// The limit is 2^53: the largest magnitude at which every integer is exactly
// representable in a double. Anything above it, or NaN/Inf, is corruption.
//============================================================================
#pragma once

#include <bit>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace convo::numeric_policy
{
    inline constexpr double kCorruptMagnitudeLimit = 9007199254740992.0; // 2^53
    inline constexpr std::uint64_t kCorruptMagnitudeBits = 0x4340000000000000ULL; // bit pattern of 2^53

    static_assert(std::bit_cast<std::uint64_t>(kCorruptMagnitudeLimit) == kCorruptMagnitudeBits,
                  "Corrupt magnitude limit must be the bit pattern of 2^53");

    /// Scalar guard: NaN, Inf, or |x| > 2^53.
    [[nodiscard]] inline bool isCorruptSample(double x) noexcept
    {
        const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
        const std::uint64_t exp = bits & 0x7FF0000000000000ULL;
        if (exp == 0x7FF0000000000000ULL)
            return true; // NaN or Inf
        return (bits & 0x7FFFFFFFFFFFFFFFULL) > kCorruptMagnitudeBits;
    }

#if defined(__AVX2__)
    /// AVX2 guard: per-lane corruption mask (NaN, Inf, or |x| > 2^53).
    [[nodiscard]] inline __m256d corruptSampleMask4(__m256d v) noexcept
    {
        const __m256d vNan = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
        const __m256d vAbs = _mm256_andnot_pd(_mm256_set1_pd(-0.0), v);
        const __m256d vBig = _mm256_cmp_pd(vAbs, _mm256_set1_pd(kCorruptMagnitudeLimit), _CMP_GT_OQ);
        return _mm256_or_pd(vNan, vBig);
    }
#endif
} // namespace convo::numeric_policy
