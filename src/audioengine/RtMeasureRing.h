#pragma once

//==============================================================================
// R-06 / U2 + U5 準備（A1'）: RT 実時間計測
//
// モード（**非 RT で 1 回だけ**決定し、RT 側は atomic load のみ。RT で getenv を呼ばない）:
//   CONVOPEQ_RT_MEASURE=0（既定）: 計測 OFF。**TSC 取得を行わない**。
//   CONVOPEQ_RT_MEASURE=2        : 時間計測＋共通サンプル領域への保存。**イベントリングへ push しない**。
//   CONVOPEQ_RT_MEASURE=1        : 時間計測＋共通サンプル領域への保存＋**イベントリングへ push**。
//
// 測定区間（計画書 §8.1 を維持）:
//   `getNextAudioBlock()` の入口（TSC 読取直前）から出口（return 直前）まで。
//   ホスト側バッファコピーは含めない。1 ブロック = 1 サンプル。
//   **ON（=1）のリング push は測定区間の内側**（push の後に終了 TSC を取得する）。
//
// サンプル保存:
//   両アーム共通の**事前確保済み固定領域**（`SampleBuffer`）。RT 側は書き込みと release store のみで、
//   動的確保・ロック・I/O・書式化・ループ・RMW を行わない。読み手（非 RT）が回収して統計を計算する。
//   容量超過（サンプル欠落）は **INCONCLUSIVE**。
//   イベントリングの dropped/writes と、サンプル領域の保存状況は**別概念**として扱う。
//
// ウォームアップ:
//   両アームとも最初の 2,000 ブロックを除外する（`Aggregator::setWarmupSamples`）。
//   各実行は最低 32,000 ブロック → 除外後の有効サンプル 30,000 以上。
//
// 判定:
//   p99 = 昇順ソート後 `ceil(0.99*N) - 1`（整数演算）。最大値・N も算出。
//   `dropped / (writes + dropped) > 10%` または **サンプル欠落** → INCONCLUSIVE。
//   9〜11% は RERUN（合格に丸めない）。測定値は「下限」である（`lower_bound=1`）。
//
// 前提作業（U2 で実装済み・維持）:
//   P1 `initializeFromEnvironment()`（非 RT）／P2 `dropped` は RMW なし／P3 `static_assert` ＋実行時 fail-closed／
//   P4 `Ring::instance()` は名前空間スコープ変数（関数ローカル static ガードなし）。
//==============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "AtomicAccess.h"   // list.md [3.1]: 製品コードでは atomic のドット誕生を直接呼んではならない（convo:: ヘルパ経由）

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    #define CONVOPEQ_RTM_HAS_TSC 1
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <x86intrin.h>
    #endif
#else
    #define CONVOPEQ_RTM_HAS_TSC 0
#endif

namespace convo {
namespace rtmeasure {

// P3: 64 bit atomic が lock-free であることをコンパイル時に要求する。
static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "R-06 RT measurement requires lock-free 64-bit atomics on this target");

// 観測イベントの種類。値は診断出力の互換性のため明示する。
enum EventKind : std::uint32_t
{
    kBlockBegin = 1,   // 予約（U2 以降は END 単独イベント。payload は 0）
    kBlockEnd   = 2,   // ブロック終端マーカー（経過時間は共通サンプル領域が持つ）
    kAlloc      = 3,   // 予約（U3）
    kFree       = 4,   // 予約（U3）
    kLockWait   = 5,   // 予約（U4）
    kDropped    = 6    // 予約（リング自体は dropped カウンタで計上する）
};

struct Event
{
    std::uint64_t tsc  = 0;
    std::uint32_t kind = 0;
    std::uint32_t arg  = 0;
};

// 計測モード（数値は環境変数と対応する）
enum class Mode : int
{
    Off      = 0,   // 計測しない（TSC 取得なし）
    Full     = 1,   // 時間計測 + 共通サンプル保存 + イベントリング push
    TimeOnly = 2    // 時間計測 + 共通サンプル保存（リング push なし）
};

namespace detail {

// 状態はすべて**名前空間スコープの inline 変数**（関数ローカル static の初期化ガードを RT 経路で取らない）。
inline std::atomic<int> g_mode{ static_cast<int>(Mode::Off) };
inline std::atomic<std::uint64_t> g_ticksPerSecond{ 0 };
inline std::atomic<bool> g_prepared{ false };

// テスト専用プローブ（既定 OFF。ON にしたときだけ計数する）
inline std::atomic<bool> g_tscProbe{ false };
inline std::atomic<std::uint64_t> g_tscReads{ 0 };
inline std::atomic<bool> g_orderProbe{ false };
inline std::atomic<int> g_order{ 0 };   // 1=リング push 開始 / 2=push 完了 / 3=終了 TSC 取得後

// テスト専用: 校正の成否を固定する（本番は常に false = 実校正）。サニタイザ下では校正の 1% 一致検査が
// 不安定になるため、モード解釈のテストを決定的にする目的でのみ使う。
inline std::atomic<bool> g_forceTimingOkForTest{ false };
// 直近の初期化で「要求された」モード（校正失敗でも記録する。解釈の検証用）
inline std::atomic<int> g_requestedMode{ 0 };

// TSC 読み取り（順序保証）
//   入口: lfence → rdtsc（先行 load を追い越させない）
//   出口: rdtscp（先行命令の完了を待つ）
#if CONVOPEQ_RTM_HAS_TSC
inline std::uint64_t readTscRaw() noexcept
{
    return static_cast<std::uint64_t>(__rdtsc());
}
inline void countTscReadIfProbed() noexcept
{
    if (convo::consumeAtomic(g_tscProbe, std::memory_order_relaxed))
        convo::fetchAddAtomic(g_tscReads, std::uint64_t{ 1 }, std::memory_order_relaxed);
}
inline std::uint64_t readOrderedBegin() noexcept
{
    countTscReadIfProbed();
    _mm_lfence();
    return static_cast<std::uint64_t>(__rdtsc());
}
inline std::uint64_t readOrderedEnd() noexcept
{
    countTscReadIfProbed();
    unsigned int aux = 0;
    return static_cast<std::uint64_t>(__rdtscp(&aux));
}
inline constexpr std::uint64_t kFallbackTicksPerSecond = 1000000000ull;
#else
inline std::uint64_t steadyNs() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline std::uint64_t readTscRaw() noexcept { return steadyNs(); }
inline void countTscReadIfProbed() noexcept
{
    if (convo::consumeAtomic(g_tscProbe, std::memory_order_relaxed))
        convo::fetchAddAtomic(g_tscReads, std::uint64_t{ 1 }, std::memory_order_relaxed);
}
inline std::uint64_t readOrderedBegin() noexcept { countTscReadIfProbed(); return steadyNs(); }
inline std::uint64_t readOrderedEnd() noexcept { countTscReadIfProbed(); return steadyNs(); }
inline constexpr std::uint64_t kFallbackTicksPerSecond = 1000000000ull;
#endif

// ticks/秒 の校正（非 RT のみ）。10 ms を 2 回測り、相対差 1% 超またはレンジ外なら失敗。
inline bool calibrateTiming() noexcept
{
#if CONVOPEQ_RTM_HAS_TSC
    using clock = std::chrono::steady_clock;
    constexpr int kRounds = 2;
    double rates[kRounds] = { 0.0, 0.0 };

    for (int round = 0; round < kRounds; ++round)
    {
        const auto t0 = clock::now();
        const std::uint64_t c0 = readTscRaw();

        auto t1 = t0;
        while (std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() < 10)
            t1 = clock::now();

        const std::uint64_t c1 = readTscRaw();
        const auto ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        const auto ticks = static_cast<double>(c1 - c0);

        if (ns <= 0.0 || ticks <= 0.0)
            return false;

        rates[round] = ticks * 1.0e9 / ns;
        if (!(rates[round] > 1.0e6 && rates[round] < 1.0e12))
            return false;
    }

    const double a = rates[0];
    const double b = rates[1];
    const double ref = (a < b) ? a : b;
    if (ref <= 0.0)
        return false;
    const double relDiff = (a > b) ? (a - b) / ref : (b - a) / ref;
    if (relDiff > 0.01)
        return false;

    convo::publishAtomic(g_ticksPerSecond,
                         static_cast<std::uint64_t>(((a + b) * 0.5) + 0.5),
                         std::memory_order_release);
    return true;
#else
    convo::publishAtomic(g_ticksPerSecond, kFallbackTicksPerSecond, std::memory_order_release);
    return true;
#endif
}

} // namespace detail

//==============================================================================
// イベントリング（モード Full のときだけ使う）
//==============================================================================
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4324)   // alignas(64) によるパディングは意図的（false sharing 回避）
#endif
class Ring
{
public:
    static constexpr std::size_t kCapacity = 8192;

    // RT 側から呼ぶ唯一の関数。ロック・確保・I/O・書式化・ループ・RMW を行わない。
    void push(std::uint32_t kind, std::uint32_t arg, std::uint64_t tsc) noexcept
    {
        const std::uint64_t w = writeLocal_;                                   // 書き手専用（非 atomic）
        const std::uint64_t r = convo::consumeAtomic(read_, std::memory_order_acquire);

        if (w - r >= kCapacity)
        {
            ++droppedLocal_;                                                   // 書き手専用（非 atomic）
            convo::publishAtomic(dropped_, droppedLocal_, std::memory_order_release);
            return;
        }

        storage_[static_cast<std::size_t>(w % kCapacity)] = Event{ tsc, kind, arg };
        ++writeLocal_;
        convo::publishAtomic(write_, writeLocal_, std::memory_order_release);
    }

    // 非 RT 側で回収する。
    std::size_t drain(Event* out, std::size_t n) noexcept
    {
        if (out == nullptr || n == 0)
            return 0;

        const std::uint64_t w = convo::consumeAtomic(write_, std::memory_order_acquire);
        std::uint64_t r = convo::consumeAtomic(read_, std::memory_order_acquire);

        std::size_t copied = 0;
        while (r < w && copied < n)
        {
            out[copied] = storage_[static_cast<std::size_t>(r % kCapacity)];
            ++copied;
            ++r;
        }

        convo::publishAtomic(read_, r, std::memory_order_release);
        return copied;
    }

    [[nodiscard]] std::uint64_t written() const noexcept
    {
        return convo::consumeAtomic(write_, std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t consumed() const noexcept
    {
        return convo::consumeAtomic(read_, std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t dropped() const noexcept
    {
        return convo::consumeAtomic(dropped_, std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t pending() const noexcept
    {
        const std::uint64_t w = convo::consumeAtomic(write_, std::memory_order_acquire);
        const std::uint64_t r = convo::consumeAtomic(read_, std::memory_order_acquire);
        return static_cast<std::size_t>(w - r);
    }

    // **RT 停止中のみ**呼べる。
    void reset() noexcept
    {
        for (std::size_t i = 0; i < kCapacity; ++i)
            storage_[i] = Event{};
        convo::publishAtomic(write_, std::uint64_t{ 0 }, std::memory_order_release);
        convo::publishAtomic(read_, std::uint64_t{ 0 }, std::memory_order_release);
        convo::publishAtomic(dropped_, std::uint64_t{ 0 }, std::memory_order_release);
        writeLocal_ = 0;
        droppedLocal_ = 0;
    }

private:
    Event storage_[kCapacity]{};
    alignas(64) std::atomic<std::uint64_t> write_{ 0 };
    alignas(64) std::atomic<std::uint64_t> read_{ 0 };
    alignas(64) std::atomic<std::uint64_t> dropped_{ 0 };
    std::uint64_t writeLocal_{ 0 };     // 書き手専用
    std::uint64_t droppedLocal_{ 0 };   // 書き手専用
};
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

inline Ring g_ring;

inline Ring& instance() noexcept
{
    return g_ring;
}

//==============================================================================
// 共通サンプル領域（モード Full / TimeOnly の両方で使う）
//
//   RT: 添字を進めて 1 要素書き、release store で公開（RMW なし・ループなし）。
//   非 RT: count() までを回収。
//   容量超過は overflow に計上し、集計側で INCONCLUSIVE とする。
//==============================================================================
class SampleBuffer
{
public:
    static constexpr std::size_t kCapacity = 1u << 16;   // 65,536 サンプル（512 KB）

    void push(std::uint64_t ticks) noexcept
    {
        const std::size_t c = countLocal_;                  // 書き手専用（非 atomic）
        if (c < kCapacity)
        {
            samples_[c] = ticks;
            ++countLocal_;
            convo::publishAtomic(count_, countLocal_, std::memory_order_release);
        }
        else
        {
            ++overflowLocal_;                               // 書き手専用（非 atomic）
            convo::publishAtomic(overflow_, overflowLocal_, std::memory_order_release);
        }
    }

    [[nodiscard]] std::size_t count() const noexcept
    {
        return convo::consumeAtomic(count_, std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t overflowed() const noexcept
    {
        return convo::consumeAtomic(overflow_, std::memory_order_acquire);
    }
    [[nodiscard]] const std::uint64_t* data() const noexcept { return samples_; }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return kCapacity; }

    // **RT 停止中のみ**呼べる。
    void reset() noexcept
    {
        for (std::size_t i = 0; i < kCapacity; ++i)
            samples_[i] = 0;
        convo::publishAtomic(count_, std::size_t{ 0 }, std::memory_order_release);
        convo::publishAtomic(overflow_, std::uint64_t{ 0 }, std::memory_order_release);
        countLocal_ = 0;
        overflowLocal_ = 0;
    }

private:
    alignas(64) std::atomic<std::size_t> count_{ 0 };
    alignas(64) std::atomic<std::uint64_t> overflow_{ 0 };
    std::size_t countLocal_{ 0 };          // 書き手専用
    std::uint64_t overflowLocal_{ 0 };     // 書き手専用
    std::uint64_t samples_[kCapacity]{};
};

inline SampleBuffer g_samples;

inline SampleBuffer& sampleBuffer() noexcept
{
    return g_samples;
}

// P4: 事前構築・事前タッチを**プロセスで 1 回だけ**行う（RT 開始前に非 RT から呼ぶ）。
inline void prepareOnce() noexcept
{
    bool expected = false;
    if (!convo::compareExchangeAtomic(detail::g_prepared, expected, true,
                                      std::memory_order_acq_rel, std::memory_order_acquire))
        return;

    instance().reset();
    sampleBuffer().reset();
}

// P1: 環境変数の解釈とモード設定は**非 RT で 1 回**。何度呼んでもよい（毎回読み直す）。
//     RT 稼働中に呼ぶ場合、`prepareOnce()` は初回しか領域に触らないため安全側である。
inline void initializeFromEnvironment() noexcept
{
    const bool lockFree = std::atomic<std::uint64_t>{}.is_lock_free();
    const bool forcedOk = convo::consumeAtomic(detail::g_forceTimingOkForTest, std::memory_order_relaxed);
    const bool timingOk = lockFree && (forcedOk || detail::calibrateTiming());

    prepareOnce();

    const char* v = std::getenv("CONVOPEQ_RT_MEASURE");

    Mode requested = Mode::Off;
    if (v != nullptr)
    {
        if (v[0] == '1')
            requested = Mode::Full;
        else if (v[0] == '2')
            requested = Mode::TimeOnly;
    }

    convo::publishAtomic(detail::g_requestedMode, static_cast<int>(requested), std::memory_order_release);
    convo::publishAtomic(detail::g_mode,
                         static_cast<int>(timingOk ? requested : Mode::Off),
                         std::memory_order_release);
}

// 直近の初期化で要求されたモード（校正失敗でも記録される。テスト用）
[[nodiscard]] inline Mode requestedMode() noexcept
{
    return static_cast<Mode>(convo::consumeAtomic(detail::g_requestedMode, std::memory_order_acquire));
}

// 現在のモード（RT から呼ばれる。atomic load のみ）
[[nodiscard]] inline Mode mode() noexcept
{
    return static_cast<Mode>(convo::consumeAtomic(detail::g_mode, std::memory_order_acquire));
}

// 計測が有効か（Off 以外）
[[nodiscard]] inline bool enabled() noexcept
{
    return mode() != Mode::Off;
}

// ticks → µs（**非 RT 側**で使う）
[[nodiscard]] inline std::uint64_t ticksToMicros(std::uint64_t ticks) noexcept
{
    const std::uint64_t perSec = convo::consumeAtomic(detail::g_ticksPerSecond, std::memory_order_acquire);
    if (perSec == 0)
        return 0;

    if (ticks > (std::numeric_limits<std::uint64_t>::max() / 1000000ull))
    {
        const std::uint64_t perMicro = perSec / 1000000ull;
        return ticks / (perMicro == 0 ? 1 : perMicro);
    }

    return (ticks * 1000000ull) / perSec;
}

//==============================================================================
// RAII ブロックスコープタイマ（測定区間 = `getNextAudioBlock()` 入口〜出口）
//
//   Off      : 何もしない（TSC も読まない）
//   TimeOnly : 入口/出口の TSC を測り、共通サンプル領域へ 1 件保存
//   Full     : 上記に加えて、**出口 TSC を読む前に**リングへ push する
//              （＝ push のコストが測定区間に含まれる）
//==============================================================================
class BlockScopeTimer
{
public:
    BlockScopeTimer() noexcept
        : mode_(convo::rtmeasure::mode()),   // ★ メンバではなく名前空間のモード取得関数を呼ぶ
          t0_(mode_ != Mode::Off ? detail::readOrderedBegin() : 0)
    {
    }

    ~BlockScopeTimer() noexcept
    {
        if (mode_ == Mode::Off)
            return;

        const bool probe = convo::consumeAtomic(detail::g_orderProbe, std::memory_order_relaxed);

        if (mode_ == Mode::Full)
        {
            if (probe)
                convo::publishAtomic(detail::g_order, 1, std::memory_order_relaxed);
            instance().push(kBlockEnd, 0, 0);
            if (probe)
                convo::publishAtomic(detail::g_order, 2, std::memory_order_relaxed);
        }

        const std::uint64_t t1 = detail::readOrderedEnd();   // ★ push の後（区間の終端）
        if (probe)
            convo::publishAtomic(detail::g_order, 3, std::memory_order_relaxed);

        sampleBuffer().push(t1 - t0_);
    }

    BlockScopeTimer(const BlockScopeTimer&) = delete;
    BlockScopeTimer& operator=(const BlockScopeTimer&) = delete;

    [[nodiscard]] Mode measurementMode() const noexcept { return mode_; }
    [[nodiscard]] bool active() const noexcept { return mode_ != Mode::Off; }
    [[nodiscard]] bool recordsRing() const noexcept { return mode_ == Mode::Full; }

private:
    Mode mode_;
    std::uint64_t t0_;
};

//==============================================================================
// 非 RT 側の集計（計画書 §6）。**RT からは呼ばない。**
//==============================================================================
struct Stats
{
    std::uint64_t samples      = 0;      // N（ウォームアップ除外後の有効サンプル数）
    std::uint64_t p99Ticks     = 0;
    std::uint64_t maxTicks     = 0;
    std::uint64_t p99Micros    = 0;
    std::uint64_t maxMicros    = 0;
    std::uint64_t warmupSkipped = 0;
    std::uint64_t ringDropped  = 0;      // イベントリングの破棄
    std::uint64_t ringWrites   = 0;      // イベントリングへ入った件数
    std::uint64_t sampleOverflow = 0;    // 共通サンプル領域の欠落
    std::uint64_t aggregatorOverflow = 0;
    double        dropRatio    = 0.0;    // ringDropped / (ringWrites + ringDropped)
    bool          inconclusive = false;  // リング破棄 >10% または サンプル欠落
    bool          boundary     = false;  // 9〜11%（RERUN）
    bool          valid        = false;
};

class Aggregator
{
public:
    static constexpr std::size_t kMaxSamples = 1u << 20;

    void setWarmupSamples(std::size_t n) noexcept
    {
        warmupTotal_ = n;
        warmupRemaining_ = n;
    }
    [[nodiscard]] std::size_t warmupSamples() const noexcept { return warmupTotal_; }

    void reset() noexcept
    {
        samples_.clear();
        sorted_.clear();
        readIdx_ = 0;
        warmupRemaining_ = warmupTotal_;
        warmupSkipped_ = 0;
        aggregatorOverflow_ = 0;
        stats_ = Stats{};
        collectCalls_ = 0;
        lastCollected_ = 0;
    }

    // 共通サンプル領域から新しい分を回収（非 RT）。ウォームアップ分は統計に加えない。
    std::size_t collect() noexcept
    {
        const std::size_t available = sampleBuffer().count();
        if (available <= readIdx_)
        {
            ++collectCalls_;
            lastCollected_ = 0;
            return 0;
        }

        const std::uint64_t* data = sampleBuffer().data();
        std::size_t added = 0;

        for (std::size_t i = readIdx_; i < available; ++i)
        {
            const std::uint64_t v = data[i];
            if (warmupRemaining_ > 0)
            {
                --warmupRemaining_;
                ++warmupSkipped_;
                continue;
            }
            if (samples_.size() < kMaxSamples)
            {
                samples_.push_back(v);
                ++added;
            }
            else
            {
                ++aggregatorOverflow_;
            }
        }

        readIdx_ = available;
        ++collectCalls_;
        lastCollected_ = added;
        return added;
    }

    [[nodiscard]] const Stats& compute() noexcept
    {
        Stats s;
        s.samples       = static_cast<std::uint64_t>(samples_.size());
        s.warmupSkipped = static_cast<std::uint64_t>(warmupSkipped_);
        s.ringWrites    = instance().written();
        s.ringDropped   = instance().dropped();
        s.sampleOverflow = sampleBuffer().overflowed();
        s.aggregatorOverflow = static_cast<std::uint64_t>(aggregatorOverflow_);

        const std::uint64_t attempts = s.ringWrites + s.ringDropped;
        s.dropRatio = (attempts == 0)
                          ? 0.0
                          : static_cast<double>(s.ringDropped) / static_cast<double>(attempts);

        const bool ringInconclusive = (s.dropRatio > 0.10);
        const bool sampleInconclusive = (s.sampleOverflow > 0) || (s.aggregatorOverflow > 0);

        s.inconclusive = ringInconclusive || sampleInconclusive;
        s.boundary     = (s.dropRatio >= 0.09 && s.dropRatio <= 0.11);

        if (!samples_.empty())
        {
            sorted_.assign(samples_.begin(), samples_.end());
            std::sort(sorted_.begin(), sorted_.end());

            const std::size_t n = sorted_.size();
            const std::size_t idx = ((99u * n + 99u) / 100u) - 1u;   // ceil(0.99*N)-1
            s.p99Ticks  = sorted_[idx < n ? idx : (n - 1)];
            s.maxTicks  = sorted_.back();
            s.p99Micros = ticksToMicros(s.p99Ticks);
            s.maxMicros = ticksToMicros(s.maxTicks);
            s.valid     = true;
        }

        stats_ = s;
        return stats_;
    }

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    [[nodiscard]] std::size_t collectCalls() const noexcept { return collectCalls_; }
    [[nodiscard]] std::size_t lastCollected() const noexcept { return lastCollected_; }

private:
    std::vector<std::uint64_t> samples_;
    std::vector<std::uint64_t> sorted_;
    Stats stats_{};
    std::size_t readIdx_ = 0;
    std::size_t warmupTotal_ = 2000;
    std::size_t warmupRemaining_ = 2000;
    std::size_t warmupSkipped_ = 0;
    std::size_t aggregatorOverflow_ = 0;
    std::size_t collectCalls_ = 0;
    std::size_t lastCollected_ = 0;
};

inline Aggregator g_aggregator;

inline Aggregator& aggregator() noexcept
{
    return g_aggregator;
}

// 1 秒ゲート用の最終実行時刻（ミリ秒）。非 RT からのみ触る。
inline std::atomic<std::int64_t> g_lastPollMs{ std::numeric_limits<std::int64_t>::min() };

// **非 RT からのみ**呼ぶ。1 秒に 1 回だけ回収＋統計＋出力を行う。
inline bool pollIfDue() noexcept
{
    const Mode m = mode();
    if (m == Mode::Off)
        return false;

    const auto nowMs = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    const std::int64_t last = convo::consumeAtomic(g_lastPollMs, std::memory_order_relaxed);
    if (last != std::numeric_limits<std::int64_t>::min() && (nowMs - last) < 1000)
        return false;

    convo::publishAtomic(g_lastPollMs, nowMs, std::memory_order_relaxed);

    (void) aggregator().collect();
    const Stats& s = aggregator().compute();

    const char* verdict = s.inconclusive ? "INCONCLUSIVE" : (s.boundary ? "RERUN" : "OK");
    std::printf("[RTM] mode=%d n=%llu p99=%llu us max=%llu us dropped=%llu writes=%llu"
                " ratio=%.3f overflow=%llu warmup_skipped=%llu verdict=%s lower_bound=1\n",
                static_cast<int>(m),
                static_cast<unsigned long long>(s.samples),
                static_cast<unsigned long long>(s.p99Micros),
                static_cast<unsigned long long>(s.maxMicros),
                static_cast<unsigned long long>(s.ringDropped),
                static_cast<unsigned long long>(s.ringWrites),
                s.dropRatio,
                static_cast<unsigned long long>(s.sampleOverflow),
                static_cast<unsigned long long>(s.warmupSkipped),
                verdict);
    return true;
}

// テスト用: 1 秒ゲートをリセットする（非 RT）。
inline void resetPollGateForTest() noexcept
{
    convo::publishAtomic(g_lastPollMs, std::numeric_limits<std::int64_t>::min(), std::memory_order_relaxed);
}

} // namespace rtmeasure
} // namespace convo
