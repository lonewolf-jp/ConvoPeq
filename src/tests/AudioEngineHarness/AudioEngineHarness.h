// AudioEngineHarness.h
// B4 IntegrationTest Harness: real AudioEngine + simulated audio thread (no GUI).
//
// GUI を一切起動せずに AudioEngine を実体化し、B4 publish パイプライン
//   commitRuntimePublication facade → OwnerChannel → IntentQueue → CoordinatorLoop
//   → executePublish → RuntimeStore swap → onPublishCommitted → receipt
// を実スレッドで通すための RAII ラッパー。
//
// ライフサイクル (MainWindow 相当):
//   start():  engine.initialize()  (rebuild thread + Bootstrap publish + CoordinatorLoop 起動)
//             engine.prepareToPlay() → 専用スレッドで getNextAudioBlock を回す
//   stop():   audio thread join → engine.releaseResources()  (idle publish #4 + CoordinatorLoop join)

#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "audioengine/AudioEngine.h"

class AudioEngineHarness final
{
public:
    AudioEngineHarness();
    ~AudioEngineHarness();

    AudioEngineHarness(const AudioEngineHarness&) = delete;
    AudioEngineHarness& operator=(const AudioEngineHarness&) = delete;

    bool start(double sampleRate = 48000.0, int blockSize = 512);
    void stop();

    AudioEngine& engine() noexcept { return *engine_; }
    long long blocksProcessed() const noexcept { return blocksProcessed_.load(std::memory_order_relaxed); }

private:
    // AudioEngine は ~19.4MB (内部配列保持) のためスタック配置不可 → ヒープ保持
    std::unique_ptr<AudioEngine> engine_;
    std::thread audioThread_;
    std::atomic<bool> running_ { false };
    std::atomic<long long> blocksProcessed_ { 0 };

    void audioLoop(int blockSize);
};
