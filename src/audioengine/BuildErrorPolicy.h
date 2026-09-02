#pragma once
// BuildErrorPolicy.h — D101-13 Phase D-3
// Extracted policy contract (BuildError / FailureClassification / RetryDisposition / BuildOutcome)
// to allow standalone contract tests without pulling AudioEngine.h / JUCE.

#include <cstddef>
#include <cstdint>

namespace convo {

enum class BuildError {
    None,
    InvalidInput,
    ResourceUnavailable,
    MKLFailure,          // ★ C-2: MKL 初期化・FFT 計画失敗
    ConvolverFailure,    // ★ C-2: Convolver Build 失敗
    PrepareFailure,      // ★ C-2: DSPCore::prepare() 失敗
    WarmupFailed,
    InternalError
};

// ── ★ dash2 §1.8 (Phase D — H.11.2 / §1.8.5.2): retryability の分類分離 ──
enum class FailureClassification : uint8_t {
    Permanent,       // retry 無意味（InvalidInput）
    Transient,       // retry 有効（ResourceUnavailable / WarmupFailed）
    Infrastructure,  // retry 有効・環境依存（ConvolverFailure / PrepareFailure）
    Fatal            // retry 無意味・異常終了（InternalError / MKLFailure）
};

enum class RetryDisposition : uint8_t {
    NoRetry,         // retry 禁止（Permanent / Fatal）
    RetryBackoff,    // exponential backoff 付き retry（Transient / Infrastructure）
    RetryImmediate   // immediate retry（WarmupFailed 等 latency-sensitive）
};

struct BuildOutcome {
    BuildError error = BuildError::None;
    FailureClassification classification = FailureClassification::Fatal;
    RetryDisposition retry = RetryDisposition::NoRetry;
};

// ★ §1.8.10.3: constexpr descriptor table
constexpr BuildOutcome kBuildErrorDefaultTable[] = {
    /* None */              { BuildError::None,              FailureClassification::Permanent,      RetryDisposition::NoRetry },
    /* InvalidInput */      { BuildError::InvalidInput,      FailureClassification::Permanent,      RetryDisposition::NoRetry },
    /* ResourceUnavailable */{ BuildError::ResourceUnavailable, FailureClassification::Transient,   RetryDisposition::RetryBackoff },
    /* MKLFailure */        { BuildError::MKLFailure,        FailureClassification::Fatal,          RetryDisposition::NoRetry },
    /* ConvolverFailure */  { BuildError::ConvolverFailure,  FailureClassification::Infrastructure, RetryDisposition::RetryBackoff },
    /* PrepareFailure */    { BuildError::PrepareFailure,    FailureClassification::Infrastructure, RetryDisposition::RetryBackoff },
    /* WarmupFailed */      { BuildError::WarmupFailed,      FailureClassification::Transient,      RetryDisposition::RetryImmediate },
    /* InternalError */     { BuildError::InternalError,     FailureClassification::Fatal,          RetryDisposition::NoRetry },
};
static_assert(sizeof(kBuildErrorDefaultTable) / sizeof(BuildOutcome)
                  == static_cast<size_t>(BuildError::InternalError) + 1,
              "kBuildErrorDefaultTable must cover all BuildError values");

constexpr const char* kBuildErrorNames[] = {
    "None", "InvalidInput", "ResourceUnavailable", "MKLFailure",
    "ConvolverFailure", "PrepareFailure", "WarmupFailed", "InternalError"
};
static_assert(sizeof(kBuildErrorNames) / sizeof(const char*)
                  == static_cast<size_t>(BuildError::InternalError) + 1,
              "kBuildErrorNames must cover all BuildError values");

// ★ §1.8.5.2 / H.11.2: BuildError → デフォルト分類の解決。
[[nodiscard]] inline BuildOutcome classifyBuildError(BuildError error) noexcept
{
    const auto idx = static_cast<size_t>(error);
    if (idx >= sizeof(kBuildErrorDefaultTable) / sizeof(BuildOutcome))
        return { BuildError::InternalError, FailureClassification::Fatal, RetryDisposition::NoRetry };
    return kBuildErrorDefaultTable[idx];
}

[[nodiscard]] inline const char* classifyBuildErrorToString(BuildError error) noexcept
{
    const auto idx = static_cast<size_t>(error);
    if (idx >= sizeof(kBuildErrorNames) / sizeof(const char*))
        return "Unknown";
    return kBuildErrorNames[idx];
}

// ── ★ CR-α (ND-07 contract — Site 3 warmup retry bound + backoff): ──
//   Site 3（main rebuild warmup）専用の bounded retry policy。純粋な constexpr/noexcept
//   policy logic（JUCE / RetryScheduler / Logger 非依存）。RebuildThread が呼び出し側。
//
//   ★ ドメイン分離（ND-07 §3 — 絶対遵守）:
//     kMaxWarmupConsecutiveRetries = 3 は Site 3 専用。
//     Recovery 系（Site 1/2）は obligation-level K=4
//     （kMaxObligationConsecutiveFailures、ISRRuntimePublicationCoordinator.h:401）と
//     Builder-local spin guard 4（AudioEngine.RebuildDispatch.cpp）で管理され、**別ドメイン**。
//     意図的に別値（3 vs 4）にすることで grep/telemetry 読解時の混同を検出可能にする。
inline constexpr std::uint32_t kMaxWarmupConsecutiveRetries = 3;

// ★ dash2 H.11.27.6（第十八者 #7）: backoff 値は tuning parameter（invariant にしない）。
//   invariant は NonRT 実行 / non-blocking / bounded（maxDelayMs 上限存在）/ caller-side counter。
//   normative default = 設計記録の 10→20→40→80ms（ND-07 §4 確定）。
struct RetryBackoffPolicy
{
    std::uint32_t initialDelayMs = 10;   // attempt 1 の delay
    std::uint32_t maxDelayMs     = 80;   // saturation 上限
    std::uint32_t multiplier     = 2;    // exponential 倍率
};

inline constexpr RetryBackoffPolicy kDefaultWarmupRetryBackoff { 10, 80, 2 };

// attempt (1-based) の backoff delay。saturation: 掛算が maxDelayMs に到達したら固定
// （uint64 中間計算 + 上限早期 return により overflow を構造的に排除）。
// attempt 0 → 0 / 1 → initialDelayMs / 2 → initial*mult / ... / maxDelayMs 到達後は固定。
[[nodiscard]] inline std::uint32_t retryBackoffDelayMs(
    const RetryBackoffPolicy& policy, std::uint32_t attempt) noexcept
{
    if (attempt == 0)
        return 0;
    std::uint64_t delay = policy.initialDelayMs;
    for (std::uint32_t i = 1; i < attempt; ++i)
    {
        delay *= policy.multiplier;
        if (delay >= policy.maxDelayMs)
            return policy.maxDelayMs;
    }
    return static_cast<std::uint32_t>(delay > policy.maxDelayMs ? policy.maxDelayMs : delay);
}

// ── ★ Site 3 warmup retry decision（純関数 — unit test 可能化）: ──
//   decision 条件（ND-07 §2/§5 — 全て AND・1 つでも欠ければ retry しない）:
//     contextRetryable AND !obsolete AND attempt <= maxRetries AND disposition != NoRetry
//   ★ RetryImmediate は「無条件 retry」ではない — delay 0 で bounded な retry である。
//   counter 意味論（ND-07 §2 / CR-α-1 Step 4）: counter は warmup failure 回数（schedule 前に ++）。
//     counter 1..max → retry #1..#max / counter max+1 回目の failure → Exhausted。
struct WarmupRetryAction
{
    // Schedule = schedule(req, delayMs) を実行 / Exhausted = retry 打ち切り（terminal telemetry 1 回）
    // / NoRetry = retry しない（telemetry なし）
    enum class Value : std::uint8_t { Schedule, Exhausted, NoRetry };
};

struct WarmupRetryDecision
{
    WarmupRetryAction::Value action = WarmupRetryAction::Value::NoRetry;
    std::uint32_t delayMs = 0;
};

[[nodiscard]] inline WarmupRetryDecision warmupRetryDecision(
    std::uint32_t attempt,                     // warmup failure count（この failure で ++ 済みの値）
    std::uint32_t maxRetries,                  // schedule 可能な retry 回数（kMaxWarmupConsecutiveRetries）
    bool contextRetryable,                     // shouldRetryWarmupFailure() = isLoadingIR()
    bool obsolete,                             // isObsolete()（schedule 前に再確認）
    RetryDisposition disposition,              // classifyBuildError(...).retry（policy source）
    const RetryBackoffPolicy& policy) noexcept
{
    // NoRetry / context 非該当 / obsolete は retry 打ち切り（telemetry なし）。
    if (disposition == RetryDisposition::NoRetry || !contextRetryable || obsolete)
        return { WarmupRetryAction::Value::NoRetry, 0 };
    // attempt は failure 回数: 1..maxRetries = retry #1..#maxRetries / maxRetries+1 = exhausted。
    if (attempt > maxRetries)
        return { WarmupRetryAction::Value::Exhausted, 0 };
    const std::uint32_t delay = (disposition == RetryDisposition::RetryBackoff)
                                    ? retryBackoffDelayMs(policy, attempt)
                                    : 0;   // RetryImmediate
    return { WarmupRetryAction::Value::Schedule, delay };
}

} // namespace convo
