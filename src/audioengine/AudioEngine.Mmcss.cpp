// AudioEngine.Mmcss.cpp — Unified MMCSS Layer for WASAPI/ASIO/DirectSound
//
// Authority Singularization:
//   WASAPI:       JUCE manages (AvSetMmThreadCharacteristicsW) → our call will fail(5/183) → success
//   ASIO(driver): Driver manages → our call may succeed or fail(5/183)
//   DirectSound:  Host manages → our call succeeds with Playback/HIGH
//
// RT-safety:
//   - thread_local ensures no lock contention across driver-owned threads (ASIO)
//   - Registration attempted ONCE (t_mmcssTried) → minimal RT impact
//   - Logging is guarded by #if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS (compiled out in Release)
//
// Shutdown:
//   - Message Thread sets mmcssShutdownRequested flag ONLY
//   - Audio Thread performs actual AvRevert in next callback (MSDN: same-thread requirement)

#include "AudioEngine.h"
#include "DiagnosticsConfig.h"
#include <windows.h>
#include <avrt.h>
#pragma comment(lib, "avrt.lib")

namespace {

// thread_local: safe for driver-owned threads (ASIO), no locking needed,
//               auto-cleanup on thread destruction, device switch creates new thread.
thread_local HANDLE t_mmcssHandle = nullptr; // NOLINT(thread-local) RT-SAFE: reviewed POD TLS for MMCSS, no destructor, single init per thread
thread_local DWORD  t_mmcssTaskIndex = 0;     // NOLINT(thread-local) RT-SAFE: MMCSS task index, scalar POD
thread_local bool   t_mmcssTried = false;       // NOLINT(thread-local) RT-SAFE: guard flag, written once per thread

// Local diagLog — each engine .cpp defines its own for file-scoped asyncSink independence.
void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

// Wrapper: Unicode (W) version only. A version has ANSI→UTF-16 ambiguity + RegEnumKeyEx enumeration
// issues that can produce ERROR_NO_MORE_ITEMS(1552) spuriously.
HANDLE tryTask(LPCWSTR taskName, DWORD& idx) noexcept
{
    idx = 0;
    return ::AvSetMmThreadCharacteristicsW(taskName, &idx);
}

} // anonymous namespace

// ── Public ──

// Determine MMCSS policy based on current audio backend device type.
// Uses the cached currentDeviceTypeName_ (set via setAudioDeviceTypeName from Message Thread).
// Called from both Message Thread (prepareToPlay) and Audio Thread (callback).
// Device type is immutable during a session → safe to call from either thread.
[[nodiscard]] AudioEngine::MmcssPolicy AudioEngine::getCurrentMmcssPolicy() const noexcept
{
    const auto& type = currentDeviceTypeName_;
    if (type.containsIgnoreCase("WASAPI") || type.containsIgnoreCase("Windows Audio"))
        return MmcssPolicy::JuceManaged;
    if (type.containsIgnoreCase("ASIO"))
        return MmcssPolicy::SelfManagedProAudio;
    if (type.containsIgnoreCase("DirectSound"))
        return MmcssPolicy::SelfManagedPlayback;
    return MmcssPolicy::None;
}

// Try to register the calling (audio) thread with MMCSS once.
// - WASAPI (JuceManaged / None): skip, return true (JUCE manages or unknown backend).
// - ASIO  (SelfManagedProAudio): AvSetMmThreadCharacteristicsW(L"Pro Audio") + AVRT_PRIORITY_CRITICAL.
// - DS    (SelfManagedPlayback):  AvSetMmThreadCharacteristicsW(L"Playback") + AVRT_PRIORITY_HIGH.
//
// Logs success/failure via CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS guard (zero cost in Release).
//
// Return value:
//   true  = MMCSS is active (by JUCE, driver, or our registration).
//   false = MMCSS registration failed entirely (fallback to NativeRT or no special priority).
//
// RT impact: first call only (~50-200μs for LPC call to MMCSS service). Subsequent calls: O(1) TLS read.
[[nodiscard]] bool AudioEngine::tryApplyMmcssForSelfManagedThread() noexcept
{
    if (t_mmcssTried)
        return (t_mmcssHandle != nullptr);
    t_mmcssTried = true;

    // ★ CPU affinity + NativeRT: always set on first callback, regardless of policy.
    //    applyMmcssPriority() handles SetThreadAffinityMask (all backends)
    //    and SetPriorityClass/SetThreadPriority (NativeRT mode only).
    applyMmcssPriority();

    if (!convo::consumeAtomic(useMmcssPriority, std::memory_order_acquire))
        return false; // NativeRT mode selected by user

    const auto policy = getCurrentMmcssPolicy();

    // WASAPI or unknown → JUCE manages or nothing to do
    if (policy == MmcssPolicy::JuceManaged || policy == MmcssPolicy::None)
        return true;

    // ── Determine task name and priority ──
    LPCWSTR primaryTask = nullptr;
    LPCWSTR fallback1   = nullptr;
    LPCWSTR fallback2   = nullptr;
    int avrtPriority    = AVRT_PRIORITY_CRITICAL;
    const char* policyTag = nullptr;

    if (policy == MmcssPolicy::SelfManagedProAudio) {
        primaryTask  = L"Pro Audio";
        fallback1    = L"Audio";
        fallback2    = nullptr;
        avrtPriority = AVRT_PRIORITY_CRITICAL;
        policyTag    = "ASIO";
    } else { // SelfManagedPlayback
        primaryTask  = L"Playback";
        fallback1    = L"Audio";
        fallback2    = L"Pro Audio";
        avrtPriority = AVRT_PRIORITY_HIGH;
        policyTag    = "DS";
    }

    // ── Attempt primary task registration ──
    DWORD idx = 0;
    HANDLE h = tryTask(primaryTask, idx);

    if (h != nullptr) {
        ::AvSetMmThreadPriority(h, static_cast<AVRT_PRIORITY>(avrtPriority));
        t_mmcssHandle = h;
        t_mmcssTaskIndex = idx;
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        // ★ [diagnostic] MMCSS registration SUCCESS — logged once
        juce::String prioStr = (avrtPriority == AVRT_PRIORITY_CRITICAL) ? "CRITICAL"
                             : (avrtPriority == AVRT_PRIORITY_HIGH)    ? "HIGH"
                             : (avrtPriority == AVRT_PRIORITY_NORMAL)  ? "NORMAL"
                                                                       : "LOW";
        diagLog("[MMCSS-" + juce::String(policyTag) + "] registered: task="
                + juce::String(primaryTask) + " priority=" + prioStr
                + " taskIndex=" + juce::String(static_cast<int>(idx)));
#endif
        return true;
    }

    // ── Failure analysis ──
    const DWORD err = ::GetLastError();

    // Already registered by JUCE (WASAPI) or professional driver (ASIO) → success
    //   5(ERROR_ACCESS_DENIED), 183(ERROR_ALREADY_EXISTS): 公式コード。
    //   1552(ERROR_NO_MORE_ITEMS): MSDN未定義だが、タスク名がレジストリに存在することが
    //     確認済み（Pro Audio, Audio 共に存在）で W 版 API を使用している場合、
    //     1552 は「スレッドが既に別のMMCSSタスクに所属している」ことを示す。
    //     このケースは ASIO ドライバが自前で MMCSS 登録済みの環境で発生する。
    if (err == ERROR_ACCESS_DENIED || err == ERROR_ALREADY_EXISTS
        || err == ERROR_NO_MORE_ITEMS) {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        // ★ [diagnostic] MMCSS already managed by JUCE/driver — expected, not an error
        diagLog("[MMCSS-" + juce::String(policyTag) + "] already registered by JUCE/driver (err="
                + juce::String(static_cast<int>(err)) + ") task="
                + juce::String(primaryTask));
#endif
        return true;
    }

    // Task name not found → fallback chain
    //   1531(ERROR_INVALID_TASK_NAME): タスク名がレジストリに存在しない。
    //   1552 は上で処理済みのため、この分岐に入るのは 1531 のみ。
    if (err == ERROR_INVALID_TASK_NAME) {
        auto attemptFallback = [&](LPCWSTR fallbackTask) -> bool {
            if (fallbackTask == nullptr) return false;
            DWORD idx2 = 0;
            HANDLE h2 = tryTask(fallbackTask, idx2);
            if (h2 != nullptr) {
                ::AvSetMmThreadPriority(h2, static_cast<AVRT_PRIORITY>(avrtPriority));
                t_mmcssHandle = h2;
                t_mmcssTaskIndex = idx2;
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
                juce::String prioStr = (avrtPriority == AVRT_PRIORITY_CRITICAL) ? "CRITICAL"
                                     : (avrtPriority == AVRT_PRIORITY_HIGH)    ? "HIGH"
                                     : (avrtPriority == AVRT_PRIORITY_NORMAL)  ? "NORMAL"
                                                                               : "LOW";
                diagLog("[MMCSS-" + juce::String(policyTag) + "] registered (fallback): task="
                        + juce::String(fallbackTask) + " priority=" + prioStr
                        + " taskIndex=" + juce::String(static_cast<int>(idx2)));
#endif
                return true;
            }
            return false;
        };
        if (attemptFallback(fallback1)) return true;
        if (attemptFallback(fallback2)) return true;
    }

    // All attempts failed
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ [diagnostic] MMCSS registration FAILURE — all paths exhausted
    diagLog("[MMCSS-" + juce::String(policyTag) + "] FAILED: primary err="
            + juce::String(static_cast<int>(err))
            + " task=" + juce::String(primaryTask));
#endif
    return false;
}

// Revert MMCSS on the audio thread (MUST be called from same thread that registered).
// Called when mmcssShutdownRequested flag is detected in the callback.
void AudioEngine::revertMmcssOnAudioThread() noexcept
{
    if (t_mmcssHandle != nullptr) {
        ::AvRevertMmThreadCharacteristics(t_mmcssHandle);
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        diagLog("[MMCSS] reverted on Audio Thread");
#endif
        t_mmcssHandle = nullptr;
        t_mmcssTaskIndex = 0;
    }
    t_mmcssTried = false; // Allow retry on next device open / thread creation
}
