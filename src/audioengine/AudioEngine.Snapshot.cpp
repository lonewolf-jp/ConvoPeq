#include <JuceHeader.h>
#include "AudioEngine.h"
#include "core/SnapshotAssembler.h"

namespace {
static void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_SNAPSHOT_CREATE)

void AudioEngine::createSnapshotFromCurrentState(uint64_t generation)
{
    debugAssertNotAudioThread();

    if (shutdownInProgress.load(std::memory_order_acquire))
        return;


    struct UIRcuGuard {
        const ConvolverProcessor& cp;
        explicit UIRcuGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterStateReader(1); }
        ~UIRcuGuard() { cp.exitStateReader(1); }
    } rcuGuard{uiConvolverProcessor};

    const convo::ConvolverState* convState = uiConvolverProcessor.getConvolverState();

    convo::EQParameters eqParams;
    if (const auto* eqState = uiEqEditor.getEQStateSnapshot())
        eqParams = eqState->toEQParameters();

    std::array<double, 9> nsCoeffs{};
    getCurrentAdaptiveCoefficients(nsCoeffs.data(), kAdaptiveNoiseShaperOrder);

    if (uiEqEditor.getAndClearPendingAGCChange())
        m_pendingAGCChange.store(true, std::memory_order_release);

    const double inputHeadroomGainValue = juce::Decibels::decibelsToGain(static_cast<double>(m_currentInputHeadroomDb.load(std::memory_order_relaxed)));
    const double outputMakeupGainValue = juce::Decibels::decibelsToGain(static_cast<double>(m_currentOutputMakeupDb.load(std::memory_order_relaxed)));
    const double convInputTrimGainValue = juce::Decibels::decibelsToGain(static_cast<double>(m_currentConvInputTrimDb.load(std::memory_order_relaxed)));
    const bool convBypass = m_currentConvBypass.load(std::memory_order_relaxed);
    const bool eqBypass = m_currentEqBypass.load(std::memory_order_relaxed);
    const bool softClip = m_currentSoftClipEnabled.load(std::memory_order_relaxed);
    const float satAmount = m_currentSaturationAmount.load(std::memory_order_relaxed);
    const convo::ProcessingOrder order = m_currentProcessingOrder.load(std::memory_order_relaxed);
    const convo::OversamplingType osType = m_currentOversamplingType.load(std::memory_order_relaxed);
    const int osFactor = m_currentOversamplingFactor.load(std::memory_order_relaxed);
    const int bitDepth = m_currentDitherBitDepth.load(std::memory_order_relaxed);
    const convo::NoiseShaperType nsType = m_currentNoiseShaperType.load(std::memory_order_relaxed);
    const double sampleRate = currentSampleRate.load(std::memory_order_acquire);
    const int maxBlockSize = maxSamplesPerBlock.load(std::memory_order_acquire);

    uint64_t eqCoeffHash = 0;
    if (EQCoeffCache* cache = eqCacheManager.getOrCreate(eqParams, sampleRate, maxBlockSize, generation))
        eqCoeffHash = cache->paramsHash;

    {
        const int currentIndex = latestEqFallbackReadIndex.load(std::memory_order_relaxed);
        const int publishIndex = 1 - currentIndex;
        latestEqParamsForFallback[(size_t) publishIndex] = eqParams;
        latestEqHashForFallback[(size_t) publishIndex] = eqCoeffHash;
        latestEqFallbackReadIndex.store(publishIndex, std::memory_order_release);
    }

    convo::SnapshotParams params = convo::SnapshotAssembler::assemble(
        convState,
        convState ? convState->stateId : 0,
        eqParams,
        nsCoeffs,
        inputHeadroomGainValue,
        outputMakeupGainValue,
        convInputTrimGainValue,
        convBypass,
        eqBypass,
        softClip,
        satAmount,
        order,
        osType,
        osFactor,
        bitDepth,
        nsType,
        generation,
        sampleRate,
        maxBlockSize,
        eqCoeffHash);


    int fadeSamples = m_eqFadeSamples.load(std::memory_order_relaxed);
    const bool promoteToStructural = m_pendingIRChange.exchange(false, std::memory_order_acq_rel);
    if (promoteToStructural)
    {
        // 例外昇格判定は制御スレッド側で確定する。
        // IR/構造変更はスナップショットfadeに流さず、構造クロスフェード経路へ昇格する。
        DBG("Phase6: IR change promoted to structural path on control thread");
        return;
    }

    // pending フラグは no-op 判定前に消費して、次回へ持ち越さない。
    if (m_pendingNSChange.exchange(false, std::memory_order_acq_rel))
    {
        fadeSamples = m_nsFadeSamples.load(std::memory_order_relaxed);
        DBG("Phase6: NS fade triggered");
    }
    else if (m_pendingAGCChange.exchange(false, std::memory_order_acq_rel))
    {
        fadeSamples = m_agcFadeSamples.load(std::memory_order_relaxed);
        DBG("Phase6: AGC fade triggered");
    }
    else
    {
        DBG("Phase6: EQ fade triggered");
    }

    const convo::GlobalSnapshot* newSnap = convo::SnapshotFactory::createImpl(
        params,
        m_coordinator.getCurrent(),
        generation,
        sampleRate);

    if (newSnap == nullptr)
    {
        diagLog("[VERIFY] snapshot no-op suppressed by createImpl: hash=0x"
            + juce::String::toHexString(static_cast<juce::int64>(eqCoeffHash))
            + " gen=" + juce::String(static_cast<juce::int64>(generation)));
        return;
    }

    debugLastCreatedEqHash.store(eqCoeffHash, std::memory_order_release);
    debugLastCreateAudioBlockCounter.store(m_audioBlockCounter.load(std::memory_order_acquire), std::memory_order_release);
    diagLog("[VERIFY] EQ createdHash=0x"
        + juce::String::toHexString(static_cast<juce::int64>(eqCoeffHash))
        + " gen=" + juce::String(static_cast<juce::int64>(generation)));

    // 反映は次のオーディオブロック境界を検知してから行う。
    const uint64_t boundaryBefore = m_audioBlockCounter.load(std::memory_order_acquire);
    if (!waitForAudioBlockBoundary(boundaryBefore, 20))
    {
        DBG("Phase6: boundary wait timeout, applying snapshot immediately on control thread");
    }

    if (fadeSamples > 0)
    {
        m_coordinator.startFade(newSnap, fadeSamples);
    }
    else
    {
        m_coordinator.switchImmediate(newSnap);
    }

    // フェイルセーフ: 何らかの競合で fade が開始されず current も空のままなら、
    // 即時適用にフォールバックして反映欠落を防ぐ。
    if (!m_coordinator.isFading() && m_coordinator.getCurrent() == nullptr)
    {
        DBG("[VERIFY] snapshot apply fallback: force switchImmediate");
        m_coordinator.switchImmediate(newSnap);
    }
}

#endif // CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_SNAPSHOT_CREATE
