#pragma once

//==============================================================================
// ConvolverStopProbe.h — 診断専用の停止レイテンシ計測（既定 OFF）
//
// 目的:
//   LoaderThread / ProgressiveUpgradeThread の「停止要求 → 実際の終了」までの時間と、
//   その間に実行される重い処理（resample / 位相変換 / NUC 構築 / cache 入出力）の
//   所要時間を実測する。shutdown protocol の時間予算（bound）を決めるための材料。
//
// 有効化:
//   CONVOPEQ_STOP_PROBE=1          … 計測を有効化（未設定/0 なら完全に no-op）
//   CONVOPEQ_STOP_PROBE_FILE=path  … 出力先（既定: カレントディレクトリの
//                                    convopeq-stop-probe.jsonl）
//
// 制約:
//   - 呼び出しは非 RT スレッドのみ（ローダー / アップグレード / Message）。
//     音声スレッドには一切置かない。
//   - 有効時のみ std::mutex と std::fprintf を使う。既定 OFF では enabled() の
//     静的 bool 参照だけで、production の挙動を変えない。
//   - 1 行 1 レコードの JSON Lines。ts は steady_clock のマイクロ秒（絶対値）。
//     flush は毎行（プロセス終了時の未 flush を避ける）。
//==============================================================================

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace convo
{
namespace probe
{

inline bool enabled() noexcept
{
    static const bool on = []
    {
        const char* v = std::getenv ("CONVOPEQ_STOP_PROBE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

inline std::uint64_t nowUs() noexcept
{
    using namespace std::chrono;
    return static_cast<std::uint64_t> (
        duration_cast<microseconds> (steady_clock::now().time_since_epoch()).count());
}

namespace detail
{
struct WorkerState
{
    const char* kind = "unknown";
    std::uint64_t requestUs = 0;      // 0 = 要求前
    std::uint64_t firstCheckUs = 0;   // 最初に停止要求を観測した時刻
    bool exited = false;
};

struct Sink
{
    static Sink& instance()
    {
        static Sink s;
        return s;
    }

    void noteStopRequest (const void* w, const char* kind) noexcept
    {
        const std::uint64_t now = nowUs();
        std::lock_guard<std::mutex> lock (mtx);
        auto& st = workers[(std::uintptr_t) w];
        st.kind = kind;
        const bool already = st.exited;
        st.exited = false;
        st.requestUs = now;
        st.firstCheckUs = 0;
        write ("{\"ts\":%llu,\"ev\":\"request\",\"w\":\"%p\",\"kind\":\"%s\",\"already_exited\":%d}",
               now, w, kind != nullptr ? kind : "unknown", already ? 1 : 0);
    }

    void noteFirstExitCheck (const void* w) noexcept
    {
        const std::uint64_t now = nowUs();
        std::lock_guard<std::mutex> lock (mtx);
        auto it = workers.find ((std::uintptr_t) w);
        if (it == workers.end() || it->second.requestUs == 0 || it->second.firstCheckUs != 0)
            return;
        it->second.firstCheckUs = now;
        write ("{\"ts\":%llu,\"ev\":\"first_check\",\"w\":\"%p\",\"kind\":\"%s\",\"req_to_first_check_us\":%llu}",
               now, w, it->second.kind,
               (unsigned long long) (now - it->second.requestUs));
    }

    void noteRunStart (const void* w, const char* kind) noexcept
    {
        const std::uint64_t now = nowUs();
        std::lock_guard<std::mutex> lock (mtx);
        auto& st = workers[(std::uintptr_t) w];
        st.kind = kind;
        st.requestUs = 0;
        st.firstCheckUs = 0;
        st.exited = false;
        write ("{\"ts\":%llu,\"ev\":\"run_start\",\"w\":\"%p\",\"kind\":\"%s\"}",
               now, w, kind != nullptr ? kind : "unknown");
    }

    void noteRunExit (const void* w, const char* kind) noexcept
    {
        const std::uint64_t now = nowUs();
        std::lock_guard<std::mutex> lock (mtx);
        auto& st = workers[(std::uintptr_t) w];
        st.kind = kind;
        const std::uint64_t req = st.requestUs;
        const std::uint64_t first = st.firstCheckUs;
        st.exited = true;
        st.requestUs = 0;
        st.firstCheckUs = 0;
        if (req == 0)
            write ("{\"ts\":%llu,\"ev\":\"run_exit\",\"w\":\"%p\",\"kind\":\"%s\",\"req_to_exit_us\":null}",
                   now, w, kind != nullptr ? kind : "unknown");
        else
            write ("{\"ts\":%llu,\"ev\":\"run_exit\",\"w\":\"%p\",\"kind\":\"%s\",\"req_to_exit_us\":%llu,\"req_to_first_check_us\":%llu}",
                   now, w, kind != nullptr ? kind : "unknown",
                   (unsigned long long) (now - req),
                   (unsigned long long) (first != 0 ? first - req : 0));
    }

    void noteMark (const char* op, int a, int b) noexcept
    {
        const std::uint64_t now = nowUs();
        std::lock_guard<std::mutex> lock (mtx);
        if (! openLocked()) return;
        std::fprintf (file, "{\"ts\":%llu,\"ev\":\"mark\",\"op\":\"%s\",\"a\":%d,\"b\":%d}",
                      (unsigned long long) now, op, a, b);
        std::fputc ('\n', file);
        std::fflush (file);
    }

    void noteSpan (const char* op, std::uint64_t us, int samples, int fftSize, int phaseMode,
                   bool cancelled) noexcept
    {
        const std::uint64_t now = nowUs();
        std::lock_guard<std::mutex> lock (mtx);
        write ("{\"ts\":%llu,\"ev\":\"span\",\"op\":\"%s\",\"us\":%llu,\"samples\":%d,\"fft\":%d,\"phase\":%d,\"cancelled\":%d}",
               now, op, (unsigned long long) us, samples, fftSize, phaseMode, cancelled ? 1 : 0);
    }

private:
    void write (const char* fmt, unsigned long long ts, const void* w, const char* kind,
                int already) noexcept
    {
        if (! openLocked()) return;
        std::fprintf (file, fmt, (unsigned long long) ts, w, kind, already);
        std::fputc ('\n', file);
        std::fflush (file);
    }

    void write (const char* fmt, unsigned long long ts, const void* w, const char* kind,
                unsigned long long v1, unsigned long long v2) noexcept
    {
        if (! openLocked()) return;
        std::fprintf (file, fmt, (unsigned long long) ts, w, kind, v1, v2);
        std::fputc ('\n', file);
        std::fflush (file);
    }

    void write (const char* fmt, unsigned long long ts, const void* w, const char* kind) noexcept
    {
        if (! openLocked()) return;
        std::fprintf (file, fmt, (unsigned long long) ts, w, kind);
        std::fputc ('\n', file);
        std::fflush (file);
    }

    void write (const char* fmt, unsigned long long ts, const void* w, const char* kind,
                unsigned long long v1) noexcept
    {
        if (! openLocked()) return;
        std::fprintf (file, fmt, (unsigned long long) ts, w, kind, v1);
        std::fputc ('\n', file);
        std::fflush (file);
    }

    void write (const char* fmt, unsigned long long ts, const char* op, unsigned long long us,
                int samples, int fftSize, int phaseMode, int cancelled) noexcept
    {
        if (! openLocked()) return;
        std::fprintf (file, fmt, (unsigned long long) ts, op, us, samples, fftSize, phaseMode, cancelled);
        std::fputc ('\n', file);
        std::fflush (file);
    }

    bool openLocked() noexcept
    {
        if (opened) return file != nullptr;
        opened = true;
        const char* path = std::getenv ("CONVOPEQ_STOP_PROBE_FILE");
        const char* target = (path != nullptr && path[0] != '\0')
                               ? path
                               : "convopeq-stop-probe.jsonl";
        file = std::fopen (target, "a");
        return file != nullptr;
    }

    std::mutex mtx;
    std::unordered_map<std::uintptr_t, WorkerState> workers;
    std::FILE* file = nullptr;
    bool opened = false;
};
} // namespace detail

inline void noteStopRequest (const void* w, const char* kind) noexcept
{
    if (enabled()) detail::Sink::instance().noteStopRequest (w, kind);
}

inline void noteFirstExitCheck (const void* w) noexcept
{
    if (enabled()) detail::Sink::instance().noteFirstExitCheck (w);
}

inline void noteRunStart (const void* w, const char* kind) noexcept
{
    if (enabled()) detail::Sink::instance().noteRunStart (w, kind);
}

inline void noteRunExit (const void* w, const char* kind) noexcept
{
    if (enabled()) detail::Sink::instance().noteRunExit (w, kind);
}

inline void noteMark (const char* op, int a = 0, int b = 0) noexcept
{
    if (enabled()) detail::Sink::instance().noteMark (op, a, b);
}

// 1 区間の計測。既定 OFF では t0_ == 0 のまま何もしない。
class Span
{
public:
    Span (const char* op, int samples, int fftSize, int phaseMode) noexcept
        : op_ (op), samples_ (samples), fft_ (fftSize), phase_ (phaseMode)
    {
        if (enabled()) t0_ = nowUs();
    }

    ~Span() noexcept
    {
        if (t0_ == 0) return;
        detail::Sink::instance().noteSpan (op_, nowUs() - t0_, samples_, fft_, phase_, cancelled_);
    }

    Span (const Span&) = delete;
    Span& operator= (const Span&) = delete;

    void setCancelled (bool c) noexcept { cancelled_ = c; }

private:
    const char* op_;
    int samples_;
    int fft_;
    int phase_;
    std::uint64_t t0_ = 0;
    bool cancelled_ = false;
};

} // namespace probe
} // namespace convo
