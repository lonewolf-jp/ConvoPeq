// AudioEngineHarness.cpp
#include "AudioEngineHarness.h"

#include "audioengine/AtomicAccess.h"
#include <JuceHeader.h>
#include <xmmintrin.h>

#include "MKLRealTimeSetup.h"

AudioEngineHarness::AudioEngineHarness()
    : engine_(std::make_unique<AudioEngine>())
{
}

AudioEngineHarness::~AudioEngineHarness()
{
    stop();
}

bool AudioEngineHarness::start(double sampleRate, int blockSize)
{
    // MainApplication::initialise 相当の MKL / denormal 設定（audio thread に触れる前）
    MKLRealTime::setup();
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    // initialize(): rebuild thread 起動 + Bootstrap world 同期 publish (B4-a3)
    //               + Structural rebuild intent 投入 + CoordinatorLoop 起動
    engine_->initialize();

    // prepareToPlay(): バッファ確保 + ランタイム DSP 有無による idle publish (#2)
    engine_->prepareToPlay(blockSize, static_cast<int>(sampleRate));

    convo::publishAtomic(running_, true, std::memory_order_release);
    audioThread_ = std::thread([this, blockSize]() { audioLoop(blockSize); });
    return true;
}
void AudioEngineHarness::stop()
{
    if (convo::exchangeAtomic(running_, false, std::memory_order_acq_rel))
    {
        if (audioThread_.joinable())
            audioThread_.join();
    }
    // releaseResources(): idle publish (#4) → receipt → shutdownCoordinatorLoop join
    // (teardown publish が CoordinatorLoop 停止前に同期完了することを同時に検証する)
    if (engine_->isEnginePrepared())
        engine_->releaseResources();
}

void AudioEngineHarness::audioLoop(int blockSize)
{
    juce::AudioBuffer<float> buffer(2, blockSize);
    buffer.clear();
    juce::MidiBuffer midi;

    while (convo::consumeAtomic(running_, std::memory_order_acquire))
    {
        juce::AudioSourceChannelInfo info(&buffer, 0, blockSize);
        engine_->getNextAudioBlock(info);
        buffer.clear();
        blocksProcessed_.fetch_add(1, std::memory_order_relaxed);
    }
}
