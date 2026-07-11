#include "PublicationExecutor.h"
#include "AudioEngine.h"
#include "RuntimeBuilder.h"
#include "core/TimeUtils.h"

namespace convo::isr {

PublishResult PublicationExecutor::publish(
    AudioEngine& engine,
    convo::aligned_unique_ptr<convo::FrozenRuntimeWorld> frozen,
    convo::isr::DSPHandle existingHandle) noexcept
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

    // ★ work70 P1-a: publishWorld 直接呼び出し → commitRuntimePublication トランザクション
    auto coordinator = engine.makeRuntimePublicationCoordinator();
    const auto result = engine.commitRuntimePublication(
        coordinator, std::move(stateOwner),
        AudioEngine::RegistrationContext::alreadyRegistered(existingHandle));

    const uint64_t publishEndUs = convo::getCurrentTimeUs();

    if (!AudioEngine::PublishStageResultTraits::isCommitted(result.stage)) {
        juce::Logger::writeToLog("[PUBLISH] commitRuntimePublication FAILED gen="
            + juce::String(static_cast<juce::int64>(worldGen))
            + " ownership=" + juce::String(static_cast<int>(result.ownership)));
        // ★ work70 Phase2: OwnershipDisposition::CallerDestroy の場合、
        //   呼び出し元が DSPLifetimeManager::destroyRolledBackDSP() を呼んで
        //   物理解放する必要がある。DSPHandle は既に rollback 済み（Reclaimed）。
        //   ※ この関数では破棄を行わず、戻り値で所有権状態を通知する。
        return PublishResult::PublishFailed;
    }

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

} // namespace convo::isr
