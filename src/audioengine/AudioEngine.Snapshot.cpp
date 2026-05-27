#include <JuceHeader.h>
#include "AudioEngine.h"
#include "core/SnapshotAssembler.h"

namespace {
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}

void AudioEngine::createSnapshotFromCurrentState(uint64_t generation)
{
    debugAssertNotAudioThread();

    if (isShutdownInProgress())
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
        convo::publishAtomic(m_pendingAGCChange, true, std::memory_order_release);

    const double inputHeadroomGainValue = juce::Decibels::decibelsToGain(static_cast<double>(convo::consumeAtomic(m_currentInputHeadroomDb, std::memory_order_acquire)));
    const double outputMakeupGainValue = juce::Decibels::decibelsToGain(static_cast<double>(convo::consumeAtomic(m_currentOutputMakeupDb, std::memory_order_acquire)));
    const double convInputTrimGainValue = juce::Decibels::decibelsToGain(static_cast<double>(convo::consumeAtomic(m_currentConvInputTrimDb, std::memory_order_acquire)));
    const bool convBypass = convo::consumeAtomic(m_currentConvBypass, std::memory_order_acquire);
    const bool eqBypass = convo::consumeAtomic(m_currentEqBypass, std::memory_order_acquire);
    const bool softClip = convo::consumeAtomic(m_currentSoftClipEnabled, std::memory_order_acquire);
    const float satAmount = convo::consumeAtomic(m_currentSaturationAmount, std::memory_order_acquire);
    const convo::ProcessingOrder order = convo::consumeAtomic(m_currentProcessingOrder, std::memory_order_acquire);
    const convo::OversamplingType osType = convo::consumeAtomic(m_currentOversamplingType, std::memory_order_acquire);
    const int osFactor = convo::consumeAtomic(m_currentOversamplingFactor, std::memory_order_acquire);
    const int bitDepth = convo::consumeAtomic(m_currentDitherBitDepth, std::memory_order_acquire);
    const convo::NoiseShaperType nsType = convo::consumeAtomic(m_currentNoiseShaperType, std::memory_order_acquire);
    const double sampleRate = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);
    const int maxBlockSize = convo::consumeAtomic(maxSamplesPerBlock, std::memory_order_acquire);

    uint64_t eqCoeffHash = 0;
    if (EQCoeffCache* cache = eqCacheManager.getOrCreate(eqParams, sampleRate, maxBlockSize, generation))
    {
        eqCoeffHash = cache->paramsHash;
    }
    else
    {
        // ISR厳密化: EQ cache を生成できないスナップショットは適用しない。
        // これにより Audio Thread 側で「snapshot指定外の回復経路」へ
        // 落ちることを防ぎ、公開済み snapshot の一貫性を維持する。
        convo::fetchAddAtomic(rtAuxMutable_.eqCacheSnapshotCreateMissCountNonRt,
                              static_cast<std::uint64_t>(1),
                              std::memory_order_acq_rel);
        diagLog("[VERIFY] snapshot suppressed: eq cache unavailable");
        return;
    }

    convo::SnapshotParams params = convo::SnapshotAssembler::assemble(
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


    int fadeSamples = convo::consumeAtomic(m_eqFadeSamples, std::memory_order_acquire);
    const bool promoteToStructural = convo::exchangeAtomic(m_pendingIRChange, false, std::memory_order_acq_rel);
    if (promoteToStructural)
    {
        // 例外昇格判定は制御スレッド側で確定する。
        // IR/構造変更はスナップショットfadeに流さず、構造クロスフェード経路へ昇格する。
        DBG("Phase6: IR change promoted to structural path on control thread");
        return;
    }

    // pending フラグは no-op 判定前に消費して、次回へ持ち越さない。
    if (convo::exchangeAtomic(m_pendingNSChange, false, std::memory_order_acq_rel))
    {
        fadeSamples = convo::consumeAtomic(m_nsFadeSamples, std::memory_order_acquire);
        DBG("Phase6: NS fade triggered");
    }
    else if (convo::exchangeAtomic(m_pendingAGCChange, false, std::memory_order_acq_rel))
    {
        fadeSamples = convo::consumeAtomic(m_agcFadeSamples, std::memory_order_acquire);
        DBG("Phase6: AGC fade triggered");
    }
    else
    {
        DBG("Phase6: EQ fade triggered");
    }

    const auto runtimeReadView = readControlRuntimeView();
    const auto* observedSnapshot = getRuntimeSnapshot(runtimeReadView);

    convo::GlobalSnapshot* newSnap = convo::SnapshotFactory::createImpl(
        params,
        observedSnapshot,
        generation,
        sampleRate);

    if (newSnap == nullptr)
    {
        diagLog("[VERIFY] snapshot no-op suppressed by createImpl: hash=0x"
            + juce::String::toHexString(static_cast<juce::int64>(eqCoeffHash))
            + " gen=" + juce::String(static_cast<juce::int64>(generation)));
        return;
    }

    convo::publishAtomic(rtAuxMutable_.debugLastCreatedEqHash, eqCoeffHash, std::memory_order_release);
    diagLog("[VERIFY] EQ createdHash=0x"
        + juce::String::toHexString(static_cast<juce::int64>(eqCoeffHash))
        + " gen=" + juce::String(static_cast<juce::int64>(generation)));

    if (fadeSamples > 0)
    {
        m_coordinator.startFade(newSnap, fadeSamples);
    }
    else
    {
        m_coordinator.switchImmediate(newSnap);
    }

    // 回復経路: 何らかの競合で fade が開始されず current も空のままなら、
    // 即時適用へ切り替えて反映欠落を防ぐ。
    const auto* observedAfterApply = getRuntimeSnapshot(readControlRuntimeView());
    if (!m_coordinator.isFading() && observedAfterApply == nullptr)
    {
        DBG("[VERIFY] snapshot apply recovery: force switchImmediate");
        m_coordinator.switchImmediate(newSnap);
    }
}
