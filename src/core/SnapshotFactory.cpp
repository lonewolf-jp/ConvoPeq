//==============================================================================
// SnapshotFactory.cpp
//==============================================================================
#include "SnapshotFactory.h"
#include "GlobalSnapshot.h"
#include "SnapshotParams.h"
#include <atomic>
#include <memory>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>

#include "audioengine/AtomicAccess.h"

namespace convo {

namespace {
#ifdef _DEBUG
    std::atomic<int> g_liveSnapshotCount{0};
#endif

    // ハッシュ結合ユーティリティ
    static inline uint64_t hashCombine(uint64_t seed, uint64_t value) noexcept
    {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
    }

    static inline uint64_t hashCombineFloat(uint64_t seed, float value) noexcept
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(float));
        // -0.0f と 0.0f を同一視
        bits &= 0x7FFFFFFF;
        return hashCombine(seed, static_cast<uint64_t>(bits));
    }
}

bool SnapshotFactory::areSnapshotsEquivalent(const SnapshotParams& params,
                                             const GlobalSnapshot& snapshot) noexcept
{
    // UAF 回避: 観測用ポインタではなく不変IDで比較する
    if (params.convStateId != snapshot.convStateId)
        return false;

    if (params.eqCoeffHash != snapshot.eqCoeffHash)
        return false;

    if (std::abs(params.sampleRate - snapshot.sampleRate) > 1.0e-9)
        return false;
    if (params.maxBlockSize != snapshot.maxBlockSize)
        return false;

    if (std::abs(params.inputHeadroomGain - snapshot.inputHeadroomGain) > 1.0e-12)
        return false;
    if (std::abs(params.outputMakeupGain - snapshot.outputMakeupGain) > 1.0e-12)
        return false;
    if (std::abs(params.convInputTrimGain - snapshot.convInputTrimGain) > 1.0e-12)
        return false;

    if (params.convBypass != snapshot.convBypass)
        return false;
    if (params.eqBypass != snapshot.eqBypass)
        return false;
    if (params.processingOrder != snapshot.processingOrder)
        return false;

    if (params.softClipEnabled != snapshot.softClipEnabled)
        return false;
    if (std::abs(params.saturationAmount - snapshot.saturationAmount) > 1.0e-6f)
        return false;

    if (params.oversamplingType != snapshot.oversamplingType)
        return false;
    if (params.oversamplingFactor != snapshot.oversamplingFactor)
        return false;

    if (params.ditherBitDepth != snapshot.ditherBitDepth)
        return false;
    if (params.noiseShaperType != snapshot.noiseShaperType)
        return false;

    for (std::size_t i = 0; i < params.nsCoeffs.size(); ++i)
    {
        if (std::abs(params.nsCoeffs[i] - snapshot.nsCoeffs[i]) > 1.0e-12)
            return false;
    }

    return true;
}

uint64_t SnapshotFactory::computeContentHash(const SnapshotParams& params) noexcept
{
    uint64_t h = 14695981039346656037ULL; // FNV offset basis

    h = hashCombine(h, params.convStateId);

    h = hashCombine(h, params.eqCoeffHash);
    h = hashCombine(h, std::bit_cast<uint64_t>(params.sampleRate));
    h = hashCombine(h, static_cast<uint64_t>(params.maxBlockSize));

    h = hashCombine(h, std::bit_cast<uint64_t>(params.inputHeadroomGain));
    h = hashCombine(h, std::bit_cast<uint64_t>(params.outputMakeupGain));
    h = hashCombine(h, std::bit_cast<uint64_t>(params.convInputTrimGain));

    h = hashCombine(h, params.convBypass ? 1ULL : 0ULL);
    h = hashCombine(h, params.eqBypass ? 1ULL : 0ULL);
    h = hashCombine(h, static_cast<uint64_t>(params.processingOrder));

    h = hashCombine(h, params.softClipEnabled ? 1ULL : 0ULL);
    h = hashCombineFloat(h, params.saturationAmount);

    h = hashCombine(h, static_cast<uint64_t>(params.oversamplingType));
    h = hashCombine(h, static_cast<uint64_t>(params.oversamplingFactor));

    h = hashCombine(h, static_cast<uint64_t>(params.ditherBitDepth));
    h = hashCombine(h, static_cast<uint64_t>(params.noiseShaperType));

    for (const auto& coeff : params.nsCoeffs)
        h = hashCombine(h, std::bit_cast<uint64_t>(coeff));

    return h;
}

GlobalSnapshot* SnapshotFactory::createImpl(const SnapshotParams& pending,
                                            const GlobalSnapshot* current,
                                            uint64_t generation,
                                            double sampleRate) noexcept
{
    SnapshotParams params = pending;
    if (sampleRate > 0.0)
        params.sampleRate = sampleRate;
    params.generation = generation;
    params.contentHash = computeContentHash(params);

    // ハッシュ一致時のみ重い等価判定を実施（衝突対策）
    if (current != nullptr && current->contentHash == params.contentHash)
    {
        if (areSnapshotsEquivalent(params, *current))
            return nullptr;
    }

    return create(params);
}

GlobalSnapshot* SnapshotFactory::create(const SnapshotParams& params)
{
    GlobalSnapshot* snap = new GlobalSnapshot(params);

    // contentHash は GlobalSnapshot コンストラクタで設定済み

#ifdef _DEBUG
    convo::fetchAddAtomic(g_liveSnapshotCount, 1, std::memory_order_acq_rel);
#endif

    return snap;
}

void SnapshotFactory::destroy(GlobalSnapshot* snap) noexcept
{
    if (!snap) return;

#ifdef _DEBUG
    convo::fetchSubAtomic(g_liveSnapshotCount, 1, std::memory_order_acq_rel);
#endif

    std::unique_ptr<GlobalSnapshot> owned{snap}; // RAII delete
}

} // namespace convo
