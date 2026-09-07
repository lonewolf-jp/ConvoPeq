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
    stopAudioOnly();
    if (engine_ == nullptr)
        return; // ★ D162-2-I2: abandonEngine() 後は engine は存在しない（放棄済み）
    // ★ D167-2: harness stop() は terminal teardown（MainWindow 相当の終端）。
    //   releaseResources の reconfigure/terminal 境界（D167-2）により、terminal intent を
    //   明示しない限り releaseResources は reconfigure pass になる。harness stop() は
    //   MainWindow ~dtor + MainApplication::shutdown 相当の terminal teardown であるため
    //   requestTerminalRelease() を先に発行する（既存 pipeline を無変更で実行）。
    engine_->requestTerminalRelease();
    // releaseResources(): idle publish (#4) → receipt → shutdownCoordinatorLoop join
    // (teardown publish が CoordinatorLoop 停止前に同期完了することを同時に検証する)
    if (engine_->isEnginePrepared())
        engine_->releaseResources();
}

// ★ D162-2-I2: audio thread のみ停止（engine releaseResources を伴わない seam）。
//   CallerDestroy 系回帰テストが prepare → release → prepare の reconfigure 形を
//   harness 契約どおりの順序（releaseResources は audio thread 停止後に呼ぶ）で
//   実行できるようにする。
void AudioEngineHarness::stopAudioOnly()
{
    if (convo::exchangeAtomic(running_, false, std::memory_order_acq_rel))
    {
        if (audioThread_.joinable())
            audioThread_.join();
    }
}

// ★ D169-2-6: audio thread のみ再開（engine prepare/release を伴わない seam）。
//   device restart stress テストが「restart cycle 毎に audio run → stop → restart」
//   を engine 再構築なしで反復するために使用する。running_ false 化済み（join 済み）
//   であることが前提（stopAudioOnly と対になる起動側）。
bool AudioEngineHarness::startAudioOnly(int blockSize)
{
    if (convo::exchangeAtomic(running_, true, std::memory_order_acq_rel))
        return true; // already running
    audioThread_ = std::thread([this, blockSize]() { audioLoop(blockSize); });
    return true;
}

// ★ D162-2-I2: engine を release せず放棄（意図的 leak・OS 回収）。
//   2 回目 releaseResources（prepare → release → prepare → release の reconfigure
//   二重サイクル）は Debug で pre-existing segfault を踏む（I2 の BISECT で
//   修復無起因と確認済み・I3 課題として記録）。本 seam はテストがその経路を
//   迂回して CallerDestroy 判定のみを完遂するために存在する。
void AudioEngineHarness::abandonEngine()
{
    stopAudioOnly();
    engine_.release();
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
