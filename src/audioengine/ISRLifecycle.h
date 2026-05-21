#pragma once

#include <atomic>
#include <mutex>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace convo {
namespace isr {

/**
 * ISR 10層 Architecture Layer 0: Lifecycle Isolation Runtime
 * JUCE callback の非決定性を runtime invariant に変換する。
 */

enum class LifecyclePhase
{
    Uninitialized,  // 初期状態
    Preparing,      // prepareToPlay 実行中
    Prepared,       // audioCallback 受付可能
    AudioRunning,   // audioCallback 実行中（re-entrant 禁止）
    Releasing,      // releaseResources 実行中
    Released,       // リソース解放完了
    Shutdown        // 終端（再 prepare 禁止）
};

/**
 * callback 入口での epoch トークン
 */
struct LifecycleToken
{
    uint64_t epochId;
    LifecyclePhase expectedPhase;
};

/**
 * RT callback 内での stack-local epoch 識別子
 */
struct CallbackExecutionEpoch
{
    uint64_t lifecycleEpoch;
    uint64_t sampleCursor;
};

/**
 * LifecyclePhase ステートマシン runtime
 * JUCE callback の違反（overlap, late callback, etc.）を検出・abort する。
 *
 * 受入条件（LIF-1～LIF-6）:
 *  - LIF-1: prepareToPlay serialized
 *  - LIF-2: releaseResources は AudioRunning 中に呼べない
 *  - LIF-3: Releasing phase 中の publish 禁止
 *  - LIF-4: crossfade start は Prepared 以降のみ
 *  - LIF-5: callback 中 runtimeVersion 変化なし
 *  - LIF-6: callback 中 DSP generation 変化なし
 */
class LifecycleIsolationRuntime
{
public:
    LifecycleIsolationRuntime();
    ~LifecycleIsolationRuntime();

    // NonRT: prepareToPlay 入口
    LifecycleToken enterPrepare(int sampleRate, int blockSize);
    void leavePrepare(LifecycleToken token);

    // RT: audioCallback 入口
    LifecycleToken enterAudioCallback();
    void leaveAudioCallback(LifecycleToken token);

    // NonRT: releaseResources 入口
    LifecycleToken enterRelease();
    void leaveRelease(LifecycleToken token);

    // 終端
    void shutdown();

    // 現在 phase（RT safe: atomic read）
    LifecyclePhase current() const noexcept;

    // phase が AudioRunning であることを assert（RT callable）
    void assertAudioRunning() const noexcept;

    // artifact emit（shutdown time または CI trigger）
    void emitPhaseTrace(const std::filesystem::path& outputPath);

private:
    struct PhaseTransition
    {
        LifecyclePhase from;
        LifecyclePhase to;
        uint64_t epochId;
        uint64_t timestamp_ns;
    };

    void validateTransition(LifecyclePhase from, LifecyclePhase to);
    LifecyclePhase transitionTo(LifecyclePhase next);

    std::atomic<LifecyclePhase> phase_{ LifecyclePhase::Uninitialized };
    std::atomic<uint64_t> epochCounter_{ 0 };
    std::atomic<uint32_t> hostChaosViolations_{ 0 };
    std::atomic<uint32_t> duplicatePrepareCollapsed_{ 0 };
    std::atomic<int> lastPreparedSampleRate_{ 0 };
    std::atomic<int> lastPreparedBlockSize_{ 0 };
    std::mutex nonRtGuard_;

    // artifact 用 trace buffer
    std::mutex traceGuard_;
    std::vector<PhaseTransition> transitions_;
};

/**
 * LifecyclePhase transition に HB edge を付与する
 */
class LifecycleBarrierRuntime
{
public:
    LifecycleBarrierRuntime(LifecycleIsolationRuntime& lifecycleRuntime);

    // prepareToPlay 完了後、Prepared HB edge を emit
    void publishPreparedBarrier();

    // releaseResources 開始前、AudioStopped HB edge を emit
    void publishReleasingBarrier();

    // shutdown 完了後、Shutdown HB edge を emit
    void publishShutdownBarrier();

private:
    LifecycleIsolationRuntime& lifecycleRuntime_;
};

} // namespace isr
} // namespace convo
