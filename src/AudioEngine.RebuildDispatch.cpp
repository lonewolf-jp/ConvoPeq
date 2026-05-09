#include <JuceHeader.h>
#include "AudioEngine.h"

extern std::atomic<bool> gShuttingDown;

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_REBUILD_DISPATCH)

// =============================================================
// Rebuild request coalescing (Stage 3)
// =============================================================
void AudioEngine::requestRebuild(convo::RebuildKind kind) noexcept
{
    if (shutdownInProgress.load(std::memory_order_acquire) ||
        gShuttingDown.load(std::memory_order_acquire))
        return;

    if (kind == convo::RebuildKind::None)
        return;

    if (kind == convo::RebuildKind::IRContent)
    {
        if (!uiConvolverProcessor.isIRFinalized())
            return;

        const int64_t nowTicks = juce::Time::getHighResolutionTicks();
        const int64_t lastTicks = lastIRContentRebuildTicks_.load(std::memory_order_relaxed);
        const int64_t minDelta = juce::Time::getHighResolutionTicksPerSecond() / 5; // 200ms

        if (lastTicks > 0 && (nowTicks - lastTicks) < minDelta)
            return;

        lastIRContentRebuildTicks_.store(nowTicks, std::memory_order_relaxed);
    }

    const uint32_t mask = convo::toMask(kind);
    const uint32_t prev = pendingRebuildMask_.fetch_or(mask, std::memory_order_acq_rel);

    if (prev == 0)
        triggerAsyncUpdate();
}

void AudioEngine::handleAsyncUpdate()
{
    if (shutdownInProgress.load(std::memory_order_acquire) ||
        gShuttingDown.load(std::memory_order_acquire))
        return;

    executeCommit();
    processRebuildRequestsInternal();
}

void AudioEngine::processRebuildRequestsInternal()
{
    // 1. mask 取得（完全 drain）
    const uint32_t mask = pendingRebuildMask_.exchange(0, std::memory_order_acq_rel);
    if (mask == 0)
        return;

    // 2. 現在の DSP パラメータ取得
    const double sr = currentSampleRate.load(std::memory_order_acquire);
    const int bs = maxSamplesPerBlock.load(std::memory_order_acquire);

    // 3. 無効状態 → 再投入（重要）
    if (sr <= 0.0 || bs <= 0)
    {
        pendingRebuildMask_.fetch_or(mask, std::memory_order_release);
        return;
    }

    // 4. 優先度制御
    // =============================

    // --- HIGH: Structural ---
    if (mask & static_cast<uint32_t>(convo::RebuildKind::Structural))
    {
        requestRebuild(sr, bs);
        return; // 他は defer
    }

    // --- MID: IRContent ---
    if (mask & static_cast<uint32_t>(convo::RebuildKind::IRContent))
    {
        requestRebuild(sr, bs);
        return;
    }

    // --- LOW: Runtime / UIOnly ---
    // 現状は何もしない（将来拡張ポイント）
}

#endif
