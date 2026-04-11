//============================================================================
// EQEditProcessor.cpp — v2.3 Phase 6
//============================================================================

#include "EQEditProcessor.h"
#include "AudioEngine.h"

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
EQEditProcessor::EQEditProcessor(AudioEngine& engine)
    : audioEngine(engine)
{
    // EQProcessor の基底は EQProcessor() でデフォルト初期化済み。
    // audioEngine への参照を保持して timerCallback でアクセスできるようにする。
}

//--------------------------------------------------------------
// デバウンス補助
//--------------------------------------------------------------

void EQEditProcessor::scheduleDebounce()
{
    pendingSnapshot.store(true, std::memory_order_release);
    if (!isTimerRunning())
        startTimer(kDebounceMs);
}

void EQEditProcessor::timerCallback()
{
    stopTimer();
    if (pendingSnapshot.exchange(false, std::memory_order_acq_rel))
    {
        if (!audioEngine.enqueueSnapshotCommand())
        {
            DBG("[EQEditProcessor] CommandBuffer full, snapshot command dropped");
        }
    }
}

//--------------------------------------------------------------
// バンドパラメータ setter（シャドウ）
//--------------------------------------------------------------

void EQEditProcessor::setBandFrequency(int band, float freq)
{
    EQProcessor::setBandFrequency(band, freq);
    scheduleDebounce();
}

void EQEditProcessor::setBandGain(int band, float gainDb)
{
    EQProcessor::setBandGain(band, gainDb);
    scheduleDebounce();
}

void EQEditProcessor::setBandQ(int band, float q)
{
    EQProcessor::setBandQ(band, q);
    scheduleDebounce();
}

void EQEditProcessor::setBandEnabled(int band, bool enabled)
{
    EQProcessor::setBandEnabled(band, enabled);
    scheduleDebounce();
}

void EQEditProcessor::setBandType(int band, EQBandType type)
{
    EQProcessor::setBandType(band, type);
    scheduleDebounce();
}

void EQEditProcessor::setBandChannelMode(int band, EQChannelMode mode)
{
    EQProcessor::setBandChannelMode(band, mode);
    scheduleDebounce();
}

//--------------------------------------------------------------
// グローバルパラメータ setter（シャドウ）
//--------------------------------------------------------------

void EQEditProcessor::setTotalGain(float gainDb)
{
    EQProcessor::setTotalGain(gainDb);
    scheduleDebounce();
}

void EQEditProcessor::setAGCEnabled(bool enabled)
{
    EQProcessor::setAGCEnabled(enabled);
    scheduleDebounce();
}

void EQEditProcessor::setNonlinearSaturation(float value) noexcept
{
    EQProcessor::setNonlinearSaturation(value);
    scheduleDebounce();
}

void EQEditProcessor::setFilterStructure(FilterStructure mode) noexcept
{
    EQProcessor::setFilterStructure(mode);
    scheduleDebounce();
}

//--------------------------------------------------------------
// バッチ操作（シャドウ）
//--------------------------------------------------------------

void EQEditProcessor::resetToDefaults()
{
    EQProcessor::resetToDefaults();
    scheduleDebounce();
}

bool EQEditProcessor::loadFromTextFile(const juce::File& file)
{
    if (EQProcessor::loadFromTextFile(file))
    {
        scheduleDebounce();
        return true;
    }
    return false;
}

void EQEditProcessor::setState(const juce::ValueTree& state)
{
    EQProcessor::setState(state);
    scheduleDebounce();
}

//--------------------------------------------------------------
// バイパス（シャドウ）
//--------------------------------------------------------------

void EQEditProcessor::setBypass(bool shouldBypass)
{
    EQProcessor::setBypass(shouldBypass);
    scheduleDebounce();
}
