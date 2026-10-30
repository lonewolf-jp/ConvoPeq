#pragma once

//==============================================================================
// R-06 / U2: RT 実時間計測（**U2 = 呼び出し点の接続 + P1/P2/P3/P4 の前提作業**）
//
// U1（`de3ef2aa`）からの差分（Owner 判断 2026-10-31 の条件を反映）:
//
//   P1: `enabled()` は環境変数を読まない。`initializeFromEnvironment()` を **RT 開始前の非 RT** から
//       1 回呼び、その結果を `std::atomic<bool>` に publish する。RT 側は acquire load のみ。
//   P2: `dropped` は RMW（fetch_add）を使わない。書き手専用の非 atomic カウンタを更新し、
//       **atomic 変数への単純 release store** で公開する。読み手は atomic load のみ。
//       非 atomic 変数（`writeLocal_` / `droppedLocal_`）は**書き手スレッドだけ**が触る。
//   P3: `std::atomic<std::uint64_t>::is_always_lock_free` を static_assert で強制し、
//       実行時 `is_lock_free()` が false の環境では**計測を自動無効化**する（fail-closed）。
//   P4: `Ring::instance()` は**関数ローカル static を使わない**（名前空間スコープの変数）。
//       関数ローカル static の初期化ガードを RT スレッドが取得する事故を構造的に排除する。
//
// 測定区間（計画書 §8.1・Owner 承認済み）:
//   `getNextAudioBlock()` の入口（TSC 読取直前）から出口（return 直前）まで。
//   ホスト側のバッファコピーは含めない。1 ブロック = 1 サンプル。
//   実装は RAII `BlockScopeTimer`（`getNextAudioBlock` は return が 12 箇所あるため、
//   デストラクタで END を記録して全経路を被覆する）。イベントは **END 単独**で、
//   ペイロードに「入口→出口の経過 ticks」を載せる（リング容量を二重消費しない）。
//
// 満杯時（計画書 §8.4・Owner 承認済み）:
//   上書きしない。あふれた分は捨てて `dropped` に計上する。測定値は「下限」である。
//   1 実行で `dropped / (written + dropped) > 10%` なら INCONCLUSIVE（合格扱いにしない）。
//
// スレッド安全:
//   単一書き手（RT）／単一読み手（非 RT）を前提とする。
//   `write_` / `read_` / `dropped_` はすべて `std::atomic`（release store / acquire load）。
//   `storage_[]` は上書きしないため、同一要素への同時アクセスは起こらない。
//
// 注意（RT 安全性の範囲）:
//   本実装は「ループ・再試行・CAS・ロック・確保・I/O・syscall を含まない」有界命令列である。
//   ただし共有キャッシュラインへの store はハードウェア依存の転送待ちを伴うため、
//   **厳密な wait-free を主張しない**（lock-free は実測済み。詳細は
//   `R06-U2-REVIEW-PACKAGE-20261031.md` §6）。
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
    kBlockBegin = 1,   // 予約（U2 では END 単独イベントを使うため未使用）
    kBlockEnd   = 2,   // arg = 未使用。tsc に「入口→出口の経過 ticks」を載せる
    kAlloc      = 3,   // arg = 確保サイズの下位 32bit
    kFree       = 4,   // arg = 未使用
    kLockWait   = 5,   // arg = 待機 µs（下限）
    kDropped    = 6    // 予約（リング自体は dropped カウンタで計上する）
};

struct Event
{
    std::uint64_t tsc  = 0;
    std::uint32_t kind = 0;
    std::uint32_t arg  = 0;
};

namespace detail {

// ---------------------------------------------------------------------------
// P4: 状態はすべて**名前空間スコープの inline 変数**に置く。
//     関数ローカル static は初期化ガード（実装依存で mutex）を持ち、
//     RT スレッドが最初に触るとその取得が RT 上で起きる。ここではそれを排除する。
// ---------------------------------------------------------------------------
inline std::atomic<bool> g_enabled{ false };
inline std::atomic<std::uint64_t> g_ticksPerSecond{ 0 };
inline std::atomic<bool> g_prepared{ false };

// テスト専用プローブ: 有効にすると時刻読み取り回数を数える（既定 OFF）。
//   本番では常に false なので、RT 経路の追加コストは「relaxed load + 分岐」のみ
//   （計測 ON のときだけ readOrderedBegin/End が呼ばれる）。
inline std::atomic<bool> g_tscProbe{ false };
inline std::atomic<std::uint64_t> g_tscReads{ 0 };

// ---------------------------------------------------------------------------
// 時刻読み取り（TSC 命令の順序保証を明文化する）
//
//   入口: `_mm_lfence()` → `__rdtsc()`
//         lfence は先行する load を追い越させない（測定区間の開始点を確定させる）。
//   出口: `__rdtscp()`
//         rdtscp は先行命令の完了を待ってから TSC を読む（区間終端を確定させる）。
//
//   この組み合わせにより「入口で読んだ値より前に区間内の処理が完了しない」
//   「出口で読んだ値は区間内の処理の完了後である」という順序が保証される。
//   MSVC / clang の双方で同一命令列になる（`intrin.h` / `x86intrin.h` のマクロ展開）。
//   TSC の時間換算は `initializeFromEnvironment()` で 1 回だけ校正し、
//   `g_ticksPerSecond`（ticks/秒）に publish する。RT 側では換算しない。
//   TSC が不変（invariant）でない環境や校正に失敗した場合は計測を無効化する（fail-closed）。
// ---------------------------------------------------------------------------
#if CONVOPEQ_RTM_HAS_TSC
inline std::uint64_t readTscRaw() noexcept
{
    return static_cast<std::uint64_t>(__rdtsc());
}
inline void countTscReadIfProbed() noexcept
{
    if (convo::consumeAtomic(g_tscProbe, std::memory_order_relaxed))
        convo::fetchAddAtomic(g_tscReads, std::uint64_t{1}, std::memory_order_relaxed);
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
inline constexpr std::uint64_t kFallbackTicksPerSecond = 1000000000ull;  // 校正失敗時は使わない（無効化する）
#else
// x86 以外（CI の Linux ビルド等）は steady_clock の ns を ticks として扱う。
inline std::uint64_t steadyNs() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}
inline void countTscReadIfProbed() noexcept
{
    if (convo::consumeAtomic(g_tscProbe, std::memory_order_relaxed))
        convo::fetchAddAtomic(g_tscReads, std::uint64_t{1}, std::memory_order_relaxed);
}
inline std::uint64_t readTscRaw() noexcept { return steadyNs(); }
inline std::uint64_t readOrderedBegin() noexcept { countTscReadIfProbed(); return steadyNs(); }
inline std::uint64_t readOrderedEnd() noexcept { countTscReadIfProbed(); return steadyNs(); }
inline constexpr std::uint64_t kFallbackTicksPerSecond = 1000000000ull;
#endif

// ticks/秒 を校正する（非 RT からのみ呼ぶ）。成功したら true。
//   10 ms の校正を **2 回**行い、両者の相対差が 1% を超えたら失敗とみなす
//   （TSC が不変でない環境・仮想化でのスケーリング等を検出するための安全側の検査）。
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
        if (!(rates[round] > 1.0e6 && rates[round] < 1.0e12))   // 1 MHz 〜 1 THz の範囲外は異常
            return false;
    }

    const double a = rates[0];
    const double b = rates[1];
    const double ref = (a < b) ? a : b;
    if (ref <= 0.0)
        return false;
    const double relDiff = (a > b) ? (a - b) / ref : (b - a) / ref;
    if (relDiff > 0.01)   // 校正の再現性が 1% を超える環境では測らない（fail-closed）
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

// 固定長リング。静的領域に確保し、RT から安全に書き込める。
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4324)   // alignas(64) によるパディングは意図的（false sharing 回避）
#endif
class Ring
{
public:
    static constexpr std::size_t kCapacity = 8192;

    // RT 側から呼ぶ唯一の関数。ロック・確保・I/O・書式化・ループを行わない。
    // P2: 破棄は「書き手専用カウンタ + 単純 release store」。RMW を使わない。
    void push(std::uint32_t kind, std::uint32_t arg, std::uint64_t tsc) noexcept
    {
        const std::uint64_t w = writeLocal_;                                   // 書き手専用（非 atomic）
        const std::uint64_t r = convo::consumeAtomic(read_, std::memory_order_acquire);

        if (w - r >= kCapacity)
        {
            // 上書きしない（古いイベントを残す）。捨てたことはカウンタで示す。
            ++droppedLocal_;                                                   // 書き手専用（非 atomic）
            convo::publishAtomic(dropped_, droppedLocal_, std::memory_order_release);  // 単純 store（RMW なし）
            return;
        }

        storage_[static_cast<std::size_t>(w % kCapacity)] = Event{ tsc, kind, arg };
        ++writeLocal_;
        convo::publishAtomic(write_, writeLocal_, std::memory_order_release);
    }

    // 非 RT 側で回収する。out に n 件までコピーし、コピーした件数を返す。
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

    // **RT 停止中のみ**呼べる。初期化時に 1 回呼ぶ（P4: 事前構築・事前タッチも兼ねる）。
    void reset() noexcept
    {
        for (std::size_t i = 0; i < kCapacity; ++i)
            storage_[i] = Event{};                                             // 初回タッチ（ページフォルト回避）
        convo::publishAtomic(write_, std::uint64_t{0}, std::memory_order_release);
        convo::publishAtomic(read_, std::uint64_t{0}, std::memory_order_release);
        convo::publishAtomic(dropped_, std::uint64_t{0}, std::memory_order_release);
        writeLocal_ = 0;
        droppedLocal_ = 0;
    }

private:
    Event storage_[kCapacity]{};
    alignas(64) std::atomic<std::uint64_t> write_{ 0 };
    alignas(64) std::atomic<std::uint64_t> read_{ 0 };
    alignas(64) std::atomic<std::uint64_t> dropped_{ 0 };
    std::uint64_t writeLocal_{ 0 };     // 書き手専用（他スレッドは触らない）
    std::uint64_t droppedLocal_{ 0 };   // 書き手専用（他スレッドは触らない）
};
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

// P4: プロセスに 1 つのリング。**名前空間スコープの変数**（関数ローカル static ではない）。
//     静的初期化は main より前（単一スレッド）に行われるため、RT 初回呼び出しでガードを取らない。
inline Ring g_ring;

inline Ring& instance() noexcept
{
    return g_ring;
}

// P4: 事前構築・事前タッチを**プロセスで 1 回だけ**行う。
//     初期化が複数回呼ばれてもリングの内容を破壊しない（計測中に initialize が再度走るケースの防御）。
inline void prepareOnce() noexcept
{
    bool expected = false;
    if (!convo::compareExchangeAtomic(detail::g_prepared, expected, true,
                                      std::memory_order_acq_rel, std::memory_order_acquire))
        return;   // 既に準備済み

    instance().reset();
    // 準備完了は compareExchange（acq_rel）で既に公開済み。RT 側は準備済みインスタンスのみを参照する。
}

// P1: **RT 開始前の非 RT から 1 回**呼ぶ。何度呼んでもよい（毎回環境変数を読み直す）。
//     RT 稼働中に呼ぶ場合、`prepareOnce()` は初回しかリングに触らないため安全側である。
//     P3 の実行時 lock-free 確認とタイミング校正もここで行う。失敗時は fail-closed（OFF）。
inline void initializeFromEnvironment() noexcept
{
    // P3: 実行時確認。lock-free でない環境では計測しない。
    const bool lockFree = std::atomic<std::uint64_t>{}.is_lock_free();

    // タイミング校正（TSC → 秒）。失敗したら計測しない。
    const bool timingOk = lockFree && detail::calibrateTiming();

    // P4: リングの事前構築・事前タッチ（初回のみ。RT 初回呼び出しでの初期化コストを除去）。
    prepareOnce();

    const char* v = std::getenv("CONVOPEQ_RT_MEASURE");
    const bool envOn = (v != nullptr && v[0] == '1');

    convo::publishAtomic(detail::g_enabled, (envOn && timingOk), std::memory_order_release);
}

// 計測が有効か。RT から呼ばれる。`getenv` も初期化ガードも伴わない（atomic load のみ）。
[[nodiscard]] inline bool enabled() noexcept
{
    return convo::consumeAtomic(detail::g_enabled, std::memory_order_acquire);
}

// ticks → µs（**非 RT 側**で使う）。RT では換算しない。
[[nodiscard]] inline std::uint64_t ticksToMicros(std::uint64_t ticks) noexcept
{
    const std::uint64_t perSec = convo::consumeAtomic(detail::g_ticksPerSecond, std::memory_order_acquire);
    if (perSec == 0)
        return 0;

    // 桁あふれ回避: ticks が巨大な場合は先に割る。
    if (ticks > (std::numeric_limits<std::uint64_t>::max() / 1000000ull))
        return ticks / (perSec / 1000000ull == 0 ? 1 : perSec / 1000000ull);

    return (ticks * 1000000ull) / perSec;
}

// RAII ブロックスコープタイマ（測定区間 = `getNextAudioBlock()` 入口〜出口）。
// 入口で構築し、**全 return 経路**でデストラクタが END イベントを 1 件だけ積む。
// 計測 OFF のときは load + 分岐のみで、時刻読み取りすら行わない。
class BlockScopeTimer
{
public:
    BlockScopeTimer() noexcept
        : on_(enabled()),
          t0_(on_ ? detail::readOrderedBegin() : 0)
    {
    }

    ~BlockScopeTimer() noexcept
    {
        if (!on_)
            return;
        const std::uint64_t t1 = detail::readOrderedEnd();
        instance().push(kBlockEnd, 0, t1 - t0_);
    }

    BlockScopeTimer(const BlockScopeTimer&) = delete;
    BlockScopeTimer& operator=(const BlockScopeTimer&) = delete;

    // 計測が有効か（＝このスコープで時刻を読んだか）。テスト用の読み取りアクセサ。
    [[nodiscard]] bool active() const noexcept { return on_; }

private:
    bool on_;
    std::uint64_t t0_;
};

//==============================================================================
// 非 RT 側の集計（計画書 §6 / U2 v2 §6）。**RT からは呼ばない。**
//
//   - 1 秒ごとに drain し、END イベントの payload（経過 ticks）をサンプルとして蓄積する。
//   - 統計: N、p99（昇順ソート後の ceil(0.99*N)-1 番目）、最大値。
//   - 破棄: dropped / (writesPublished + dropped)。10% 超は INCONCLUSIVE。
//     9〜11% は境界（RERUN。合格に丸めない）。
//   - 出力: `[RTM] ... lower_bound=1`（測定値が下限であることを明示）。
//==============================================================================
struct Stats
{
    std::uint64_t samples    = 0;      // N（集計に使ったサンプル数）
    std::uint64_t p99Ticks   = 0;
    std::uint64_t maxTicks   = 0;
    std::uint64_t p99Micros  = 0;
    std::uint64_t maxMicros  = 0;
    std::uint64_t dropped    = 0;      // リングの破棄件数
    std::uint64_t writes     = 0;      // リングに入った件数
    double        dropRatio  = 0.0;
    bool          inconclusive = false;   // 10% 超
    bool          boundary     = false;   // 9〜11%（再測定対象）
    bool          aggregatorFull = false; // 集計側の上限に達した
    bool          valid        = false;   // サンプルが 1 件以上
};

class Aggregator
{
public:
    // 集計側の上限（約 8 MB）。超えた分は数えず aggregatorFull を立てる。
    static constexpr std::size_t kMaxSamples = 1u << 20;

    void reset() noexcept
    {
        samples_.clear();
        sorted_.clear();
        stats_ = Stats{};
        lastDrainCount_ = 0;
        drainCalls_ = 0;
    }

    // リングから回収する（非 RT）。戻り値は今回回収した件数。
    std::size_t drainOnce() noexcept
    {
        constexpr std::size_t kBatch = 256;
        Event buf[kBatch];
        std::size_t total = 0;

        for (;;)
        {
            const std::size_t n = instance().drain(buf, kBatch);
            if (n == 0)
                break;

            for (std::size_t i = 0; i < n; ++i)
            {
                if (samples_.size() < kMaxSamples)
                    samples_.push_back(buf[i].tsc);
                else
                    stats_.aggregatorFull = true;
            }
            total += n;
            if (n < kBatch)
                break;
        }

        ++drainCalls_;
        lastDrainCount_ = total;
        return total;
    }

    // 統計を計算して保持する（非 RT）。累積サンプル全体を対象にする。
    [[nodiscard]] const Stats& compute() noexcept
    {
        Stats s;
        s.samples = static_cast<std::uint64_t>(samples_.size());
        s.writes  = instance().written();
        s.dropped = instance().dropped();

        const std::uint64_t attempts = s.writes + s.dropped;
        s.dropRatio = (attempts == 0)
                          ? 0.0
                          : static_cast<double>(s.dropped) / static_cast<double>(attempts);
        s.inconclusive = (s.dropRatio > 0.10);
        s.boundary     = (s.dropRatio >= 0.09 && s.dropRatio <= 0.11);

        if (!samples_.empty())
        {
            sorted_.assign(samples_.begin(), samples_.end());
            std::sort(sorted_.begin(), sorted_.end());

            const std::size_t n = sorted_.size();
            // ceil(0.99*N)-1 を整数演算で（浮動小数の丸め差を避ける）
            const std::size_t idx = ((99u * n + 99u) / 100u) - 1u;
            s.p99Ticks  = sorted_[idx < n ? idx : (n - 1)];
            s.maxTicks  = sorted_.back();
            s.p99Micros = ticksToMicros(s.p99Ticks);
            s.maxMicros = ticksToMicros(s.maxTicks);
            s.valid     = true;
        }

        s.aggregatorFull = stats_.aggregatorFull;
        stats_ = s;
        return stats_;
    }

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    [[nodiscard]] std::size_t drainCalls() const noexcept { return drainCalls_; }
    [[nodiscard]] std::size_t lastDrainCount() const noexcept { return lastDrainCount_; }

private:
    std::vector<std::uint64_t> samples_;   // 経過 ticks
    std::vector<std::uint64_t> sorted_;
    Stats stats_{};
    std::size_t lastDrainCount_ = 0;
    std::size_t drainCalls_ = 0;
};

// プロセスに 1 つの集計器（名前空間スコープ。関数ローカル static を使わない）
inline Aggregator g_aggregator;

inline Aggregator& aggregator() noexcept
{
    return g_aggregator;
}

// 1 秒ゲート用の最終実行時刻（ミリ秒）。非 RT からのみ触る。
inline std::atomic<std::int64_t> g_lastPollMs{ std::numeric_limits<std::int64_t>::min() };

// **非 RT からのみ**呼ぶ。1 秒に 1 回だけ drain＋統計＋出力を行う。
//   `runCoordinatorPhase()`（1 ms tick の非 RT スレッド）から呼ばれる。
//   戻り値: 今回 drain と出力を行ったら true。
inline bool pollIfDue() noexcept
{
    if (!enabled())
        return false;

    const auto nowMs = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    const std::int64_t last = convo::consumeAtomic(g_lastPollMs, std::memory_order_relaxed);
    if (last != std::numeric_limits<std::int64_t>::min() && (nowMs - last) < 1000)
        return false;

    convo::publishAtomic(g_lastPollMs, nowMs, std::memory_order_relaxed);

    (void) aggregator().drainOnce();
    const Stats& s = aggregator().compute();

    const char* verdict = s.inconclusive ? "INCONCLUSIVE" : (s.boundary ? "RERUN" : "OK");
    std::printf("[RTM] n=%llu p99=%llu us max=%llu us dropped=%llu writes=%llu ratio=%.3f"
                " verdict=%s lower_bound=1\n",
                static_cast<unsigned long long>(s.samples),
                static_cast<unsigned long long>(s.p99Micros),
                static_cast<unsigned long long>(s.maxMicros),
                static_cast<unsigned long long>(s.dropped),
                static_cast<unsigned long long>(s.writes),
                s.dropRatio,
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
