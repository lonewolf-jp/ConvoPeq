#include "PublicationExecutor.h"
#include "AudioEngine.h"
#include "RuntimeBuilder.h"
#include "core/TimeUtils.h"

namespace convo::isr {

PublishResult PublicationExecutor::publish(
    AudioEngine& engine,
    convo::aligned_unique_ptr<convo::FrozenRuntimeWorld> frozen) noexcept
{
    if (!frozen)
        return PublishResult::PublishFailed;

    const uint64_t publishStartUs = convo::getCurrentTimeUs();

    // ★ publish発行時のcallbackIndex（Worker→audio側の最新値）
    const uint64_t publishCallbackIdx = convo::consumeAtomic(
        engine.rtLocalState_.audioCallbackEpochCounter, std::memory_order_acquire);

    // ★ Phase4: FrozenRuntimeWorld から RuntimeState* を抽出して Coordinator に渡す
    auto* rawState = frozen->releaseState();
    if (rawState == nullptr)
        return PublishResult::PublishFailed;

    // publishWorld前にworld情報を保存（publishWorld後にrawStateは無効になる可能性あり）
    const uint64_t worldGen = rawState->generation;
    const uint64_t worldId = rawState->worldId;

    auto stateOwner = convo::aligned_unique_ptr<RuntimeState>(rawState);

    auto coordinator = engine.makeRuntimePublicationCoordinator();
    const auto outcome = coordinator.publishWorld(std::move(stateOwner));

    const uint64_t publishEndUs = convo::getCurrentTimeUs();

    switch (outcome) {
        case PublishStageResult::Success: {
            const auto seq = engine.getLastCommittedPublicationSequence();
            // ★ PublishTimingHistory に書き込み（NonRT writer → RT reader lookup）
            const uint64_t wc = convo::fetchAddAtomic(
                engine.rtLocalState_.publishTimingWriteCount,
                uint64_t{1}, std::memory_order_acq_rel);
            const uint64_t slot = wc % engine.rtLocalState_.kPublishTimingSlots;
            engine.rtLocalState_.publishTimingHistory[slot] = {
                .sequence = static_cast<uint64_t>(seq),
                .publishStartUs = publishStartUs,
                .publishEndUs = publishEndUs,
                .gen = worldGen,
                .worldId = worldId,
                .publishCallbackIndex = publishCallbackIdx
            };
            // Release-store済み（fetchAddのacq_rel）→ readerのacquire loadで可視

            // [PUBLISH] ログ
            const juce::String publishLog = juce::String("[PUBLISH] seq=")
                + juce::String(static_cast<juce::int64>(seq))
                + " gen=" + juce::String(static_cast<juce::int64>(worldGen))
                + " worldId=" + juce::String(static_cast<juce::int64>(worldId))
                + " publishDurationUs=" + juce::String(static_cast<juce::int64>(publishEndUs - publishStartUs))
                + " publishCallbackIdx=" + juce::String(static_cast<juce::int64>(publishCallbackIdx));
            DBG(publishLog);
            juce::Logger::writeToLog(publishLog);
            return PublishResult::Success;
        }
        case PublishStageResult::Rejected:
            return PublishResult::ValidationFailed;
        case PublishStageResult::Failed:
            return PublishResult::PublishFailed;
    }

    return PublishResult::PublishFailed;
}

} // namespace convo::isr
