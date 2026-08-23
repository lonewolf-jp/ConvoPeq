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

} // namespace convo
