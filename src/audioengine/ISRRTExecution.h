#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

namespace convo {
namespace isr {

/**
 * ISR 10層 Architecture Layer 1: RT Execution Frame Separation
 * RT callback内の全状態をスタックローカルなRTExecutionFrameに封じ込める。
 */

/**
 * crossfade 状態を管理する軽量アキュムレータ
 */
struct FadeAccumulator
{
    double gainFrom;   // crossfade 元の現在ゲイン [0.0, 1.0]
    double gainTo;     // crossfade 先の現在ゲイン [0.0, 1.0]
    bool   active;     // crossfade 実行中フラグ
};

/**
 * RT callback ごとにスタックローカルで生成・破棄される実行コンテキスト
 * RT-1: heap allocation 禁止
 */
struct RTExecutionFrame
{
    // DSP ハンドル（read-only view）
    // dspHandleRuntime から解決済み
    uint64_t activeDSPHandle;
    uint64_t fadingDSPHandle;   // crossfade 中のみ有効

    // crossfade 状態
    FadeAccumulator fade;

    // scratch メモリ（preallocated pool から取得済み pointer）
    void* scratchPtr;
    size_t scratchSize;

    // 現在 callback の sample cursor
    uint64_t sampleCursor;

    // callback 単位の一貫 view 識別子
    uint64_t callbackEpoch;

    // lifecycle epoch（LifecycleIsolationRuntime から取得）
    uint64_t lifecycleEpoch;

    // callback 開始時点の runtime graph revision
    uint64_t runtimeGraphRevision;

    // RT trace relay buffer へのポインタ（nullptr は tracing 無効）
    class RTTraceRelay* traceRelay;
};

/**
 * RT trace event の軽量構造体
 */
struct RTTraceEvent
{
    uint64_t sampleCursor;
    uint32_t eventCode;
    uint32_t eventData;
};

/**
 * RT callback から非RT側への lock-free relay buffer
 * 固定サイズリングバッファで trace event を転送
 */
class RTTraceRelay
{
public:
    static constexpr size_t RELAY_BUFFER_SIZE = 4096;

    RTTraceRelay();
    ~RTTraceRelay();

    // RT: trace event を enqueue（lock-free、固定サイズリングバッファ）
    void enqueue(const RTTraceEvent& event) noexcept;

    // NonRT: relay buffer を drain（copy out）
    void drain();

    // get current drain count
    size_t getCurrentDrainCount() const noexcept;

private:
    std::atomic<uint64_t> writeIndex_{0};
    std::atomic<uint64_t> readIndex_{0};

    std::atomic<RTTraceEvent*> buffer_{nullptr};
};

/**
 * RT capability firewall token
 * RT callback の入口で発行、出口で検証
 */
struct FirewallToken
{
    std::thread::id threadId;
    uint64_t epochId;
    bool isValid{false};
};

/**
 * RT callback の入口で authority leakage を検出・防止する firewall
 */
class RTCapabilityFirewall
{
public:
    RTCapabilityFirewall();
    ~RTCapabilityFirewall();

    // RT callback 入口: 現在の capability をチェック
    // MessageManager へのアクセス可能な状態などの違反を検出したら abort
    FirewallToken enter() noexcept;

    // RT callback 出口: 副作用の漏洩を検出
    void leave(const FirewallToken& token) noexcept;

    // audit: RT callback 内から publishAtomic が呼ばれていないか検査
    // （Debug/CI build のみ有効）
    void auditPublishAttempt(const char* callSite) noexcept;

private:
    // thread-local フラグ（RT context 検出用）
    static thread_local bool isRTContextFlag_;
};

/**
 * RT callback 内での heap allocation を検出・abort する firewall
 */
class RTAllocatorFirewall
{
public:
    // Debug/CI build: operator new / malloc override で呼ばれる
    static void onAllocAttempt(size_t size, const char* callSite) noexcept;

    // RT callback 中であることを thread-local flag で示す
    static void markRTContext(bool entering) noexcept;

    // RT context であるか確認
    static bool isRTContext() noexcept;

private:
    static thread_local bool isRTContextFlag_;
};

/**
 * RT execution frame の生成ヘルパー関数
 * RT callback 入口で呼ばれ、フレームを初期化
 */
inline RTExecutionFrame makeRTExecutionFrame(
    uint64_t activeDSP,
    uint64_t fadingDSP,
    const FadeAccumulator& fade,
    void* scratchPtr,
    size_t scratchSize,
    uint64_t sampleCursor,
    uint64_t callbackEpoch,
    uint64_t lifecycleEpoch,
    uint64_t runtimeGraphRevision,
    RTTraceRelay* traceRelay) noexcept
{
    return RTExecutionFrame{
        .activeDSPHandle   = activeDSP,
        .fadingDSPHandle   = fadingDSP,
        .fade              = fade,
        .scratchPtr        = scratchPtr,
        .scratchSize       = scratchSize,
        .sampleCursor      = sampleCursor,
        .callbackEpoch     = callbackEpoch,
        .lifecycleEpoch    = lifecycleEpoch,
        .runtimeGraphRevision = runtimeGraphRevision,
        .traceRelay        = traceRelay
    };
}

/**
 * RT-1 invariant: RTExecutionFrame は stack-local のみ
 * RT-2 invariant: RTExecutionFrame 内の DSP handle は read-only
 * RT-3 invariant: RTExecutionFrame を callback 外で保持禁止
 */

}  // namespace isr
}  // namespace convo
