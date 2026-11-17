#pragma once
//==============================================================================
// DiagObjectRegistry.h — WORK113 変更 1（識別基盤・DIAG 専用）
//
// 目的:
//   DSPCore と ConvolverProcessor の生成・遅延破棄・物理破棄を 1 箇所で記録し、
//   キュー経由でも直接破棄でも同じ帰属で読めるようにする。
//
// 契約:
//   - 全 API は NonRT 専用（ctor / dtor / 計測 / メッセージスレッドからの照会）。
//   - RT 経路からは呼ばない。内部は std::mutex で直列化する。
//   - DIAG OFF では宣言ごと消える（#if で囲む）。
//   - ID は DSPCore が既存の runtimeUuid を使う。ConvolverProcessor は
//     このレジストリが採番する（kind で名前空間を分ける）。
//   - ポインタは記録用に保持するが、破棄時の特定には使わない（特定は id）。
//   - 未登録（スロット枯渇）個体も created / destroyed の総数には数える。
//   - 0 は「未設定の ID」の予約値であり、計測対象として加算しない。
//
// ログ:
//   [CORE_LIFE] / [CONV_LIFE] は logSink_ 経由で出す。sink 未設定なら何も出さない。
//   呼び出し側（AudioEngine の ctor）が juce::Logger への sink を 1 回だけ設定する。
//==============================================================================

#include <array>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS

namespace convo::diag
{

enum class ObjKind : std::uint8_t
{
    Core = 0,
    Convolver = 1
};

enum class ObjState : std::uint8_t
{
    Empty = 0,
    Live = 1,
    Deferred = 2,
    Destroyed = 3
};

struct ObjectRecord
{
    std::uint64_t id { 0 };
    const void* ptr { nullptr };
    ObjKind kind { ObjKind::Core };
    ObjState state { ObjState::Empty };
    std::uint64_t generation { 0 };
    std::uint64_t createdUs { 0 };
    std::uint64_t destroyedUs { 0 };
    std::size_t measuredBytes { 0 };
    bool idValid { false };
    bool registered { false };
};

struct Totals
{
    // DSPCore
    std::uint32_t liveCores { 0 };              // live + deferred（旧 liveCount と同義）
    std::uint32_t liveOnlyCores { 0 };
    std::uint32_t deferredCores { 0 };
    std::uint32_t destroyedCores { 0 };
    std::uint64_t measuredCoreBytes { 0 };       // live + deferred のみ（destroyed は含めない）

    // ConvolverProcessor
    std::uint32_t liveConvolvers { 0 };
    std::uint64_t measuredConvolverBytes { 0 };

    // 整合・診断
    std::uint32_t createdTotal { 0 };
    std::uint32_t destroyedTotal { 0 };
    std::uint32_t unregisteredCreated { 0 };
    std::uint32_t unregisteredDestroyed { 0 };
    std::uint32_t overflowCount { 0 };
    std::uint32_t recycleCount { 0 };
    std::uint32_t historyOverwriteCount { 0 };
    std::uint32_t uuidZeroCount { 0 };
    std::uint32_t backwardTransitions { 0 };
    std::uint32_t duplicateIds { 0 };

    [[nodiscard]] bool consistent() const noexcept
    {
        return createdTotal == (liveCores + destroyedTotal);
    }
};

class ObjectRegistry
{
public:
    static constexpr std::size_t kCoreSlots = 32;
    static constexpr std::size_t kConvolverSlots = 8;
    static constexpr std::size_t kHistorySlots = 128;

    using LogSink = void (*)(const char*);

    static ObjectRegistry& instance() noexcept
    {
        static ObjectRegistry registry;      // NonRT 専用・プロセス寿命
        return registry;
    }

    ObjectRegistry (const ObjectRegistry&) = delete;
    ObjectRegistry& operator= (const ObjectRegistry&) = delete;

    void setLogSink (LogSink sink) noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        logSink_.store (sink, std::memory_order_release);
        if (sink == nullptr)
            return;

        // ★ 構築初期（logger 設定前）のイベントを、sink 設定時にまとめて出す。
        for (std::size_t i = 0; i < pendingLogCount_; ++i)
            sink (pendingLog_[i].c_str());
        pendingLogCount_ = 0;
    }

    // 他の DIAG 出力（例: [RETIRE_RECLAIM]）が同じログ先へ書くための入口。
    void emitLine (const char* line) noexcept
    {
        if (line == nullptr)
            return;
        const LogSink sink = logSink_.load (std::memory_order_acquire);
        if (sink != nullptr)
            sink (line);
    }

    // ★ 起動初期（CLI ログファイルが開く前）のイベントを、後からまとめて出す。
    //   Timer の最初の [MEM_ATTR] で 1 回だけ呼ぶ（Message Thread）。
    void flushRecentLog() noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        const LogSink sink = logSink_.load (std::memory_order_acquire);
        if (sink == nullptr || recentCount_ == 0)
            return;

        const std::size_t start = (recentHead_ + kRecentLogSlots - recentCount_) % kRecentLogSlots;
        for (std::size_t i = 0; i < recentCount_; ++i)
            sink (recentLog_[(start + i) % kRecentLogSlots].c_str());

        recentCount_ = 0;   // 二重出力を避ける
    }

    // ── ID 採番（ConvolverProcessor 用。DSPCore は runtimeUuid を使う）──
    [[nodiscard]] std::uint64_t nextConvolverId() noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        if (nextConvolverId_ == 0)
            nextConvolverId_ = 1;
        return nextConvolverId_++;
    }

    // ── DSPCore ──
    void coreCreated (std::uint64_t id) noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        ++createdTotal_;
        ++liveCores_;

        const bool idValid = (id != 0);
        if (! idValid)
            ++uuidZeroCount_;

        auto* slot = acquireSlot (cores_, id, ObjKind::Core, idValid);
        if (slot == nullptr)
            return;

        slot->id = id;
        slot->state = ObjState::Live;
        slot->createdUs = nowUs();
        slot->destroyedUs = 0;
        slot->measuredBytes = 0;
        ++liveOnlyCores_;

        logf ("[CORE_LIFE] id=%llu event=create live=%u",
              static_cast<unsigned long long> (id),
              static_cast<unsigned> (liveCores_));
    }

    void coreMeasured (std::uint64_t id, std::size_t bytes, std::uint64_t generation) noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        auto* slot = findRegistered (cores_, id);
        if (slot == nullptr)
            return;

        slot->measuredBytes = bytes;
        slot->generation = generation;

        logf ("[CORE_LIFE] id=%llu event=measure gen=%llu bytes=%zu",
              static_cast<unsigned long long> (id),
              static_cast<unsigned long long> (generation),
              bytes);
    }

    void coreDeferred (std::uint64_t id) noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        auto* slot = findRegistered (cores_, id);
        if (slot == nullptr)
            return;

        if (slot->state != ObjState::Live)
        {
            ++backwardTransitions_;
            return;
        }

        slot->state = ObjState::Deferred;
        if (liveOnlyCores_ > 0)
            --liveOnlyCores_;
        ++deferredCores_;

        logf ("[CORE_LIFE] id=%llu event=defer deferred=%u",
              static_cast<unsigned long long> (id),
              static_cast<unsigned> (deferredCores_));
    }

    void coreDestroyed (std::uint64_t id) noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        ++destroyedTotal_;
        if (liveCores_ > 0)
            --liveCores_;

        auto* slot = findRegistered (cores_, id);
        if (slot == nullptr)
        {
            ++unregisteredDestroyed_;
            logf ("[CORE_LIFE] id=%llu event=destroy measuredBytes=na stateUs=na unregistered=1",
                  static_cast<unsigned long long> (id));
            return;
        }

        const auto stateUs = nowUs() - slot->createdUs;
        const auto bytes = slot->measuredBytes;
        const auto generation = slot->generation;

        if (slot->state == ObjState::Deferred)
        {
            if (deferredCores_ > 0)
                --deferredCores_;
        }
        else if (slot->state == ObjState::Live)
        {
            if (liveOnlyCores_ > 0)
                --liveOnlyCores_;
        }
        else
        {
            ++backwardTransitions_;
        }

        slot->state = ObjState::Destroyed;
        slot->destroyedUs = nowUs();
        ++destroyedCores_;
        pushHistory (*slot);

        logf ("[CORE_LIFE] id=%llu event=destroy gen=%llu measuredBytes=%zu stateUs=%llu",
              static_cast<unsigned long long> (id),
              static_cast<unsigned long long> (generation),
              bytes,
              static_cast<unsigned long long> (stateUs));
    }

    // ── ConvolverProcessor ──
    void convolverCreated (std::uint64_t id) noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        auto* slot = acquireSlot (convs_, id, ObjKind::Convolver, id != 0);
        if (slot == nullptr)
            return;

        slot->id = id;
        slot->state = ObjState::Live;
        slot->createdUs = nowUs();
        slot->destroyedUs = 0;
        slot->measuredBytes = 0;
        ++liveConvolvers_;

        logf ("[CONV_LIFE] id=%llu event=create live=%u",
              static_cast<unsigned long long> (id),
              static_cast<unsigned> (liveConvolvers_));
    }

    void convolverMeasured (std::uint64_t id, std::size_t bytes) noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        auto* slot = findRegistered (convs_, id);
        if (slot == nullptr)
            return;

        slot->measuredBytes = bytes;
        logf ("[CONV_LIFE] id=%llu event=measure bytes=%zu",
              static_cast<unsigned long long> (id),
              bytes);
    }

    void convolverDestroyed (std::uint64_t id) noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        auto* slot = findRegistered (convs_, id);
        const auto bytes = (slot != nullptr) ? slot->measuredBytes : std::size_t { 0 };
        const auto stateUs = (slot != nullptr) ? (nowUs() - slot->createdUs) : std::uint64_t { 0 };

        if (slot != nullptr)
        {
            slot->state = ObjState::Destroyed;
            slot->destroyedUs = nowUs();
            pushHistory (*slot);
        }
        if (liveConvolvers_ > 0)
            --liveConvolvers_;

        if (slot == nullptr)
            logf ("[CONV_LIFE] id=%llu event=destroy measuredBytes=na stateUs=na",
                  static_cast<unsigned long long> (id));
        else
            logf ("[CONV_LIFE] id=%llu event=destroy measuredBytes=%zu stateUs=%llu",
                  static_cast<unsigned long long> (id),
                  bytes,
                  static_cast<unsigned long long> (stateUs));
    }

    // ── 照会 ──
    [[nodiscard]] bool lookupCoreBytes (std::uint64_t id, std::uint64_t& measuredBytes) const noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        for (const auto& r : cores_)
        {
            if (r.registered && r.id == id && r.state != ObjState::Destroyed)
            {
                measuredBytes = static_cast<std::uint64_t> (r.measuredBytes);
                return r.idValid;
            }
        }
        return false;
    }

    [[nodiscard]] Totals totals() const noexcept
    {
        std::lock_guard<std::mutex> lock (mutex_);
        Totals t;
        t.liveCores = liveCores_;
        t.liveOnlyCores = liveOnlyCores_;
        t.deferredCores = deferredCores_;
        t.destroyedCores = destroyedCores_;
        t.liveConvolvers = liveConvolvers_;
        t.createdTotal = createdTotal_;
        t.destroyedTotal = destroyedTotal_;
        t.unregisteredCreated = unregisteredCreated_;
        t.unregisteredDestroyed = unregisteredDestroyed_;
        t.overflowCount = overflowCount_;
        t.recycleCount = recycleCount_;
        t.historyOverwriteCount = historyOverwriteCount_;
        t.uuidZeroCount = uuidZeroCount_;
        t.backwardTransitions = backwardTransitions_;
        t.duplicateIds = duplicateIds_;

        for (const auto& r : cores_)
        {
            if (r.registered && (r.state == ObjState::Live || r.state == ObjState::Deferred))
                t.measuredCoreBytes += static_cast<std::uint64_t> (r.measuredBytes);
        }
        for (const auto& r : convs_)
        {
            if (r.registered && r.state == ObjState::Live)
                t.measuredConvolverBytes += static_cast<std::uint64_t> (r.measuredBytes);
        }
        return t;
    }

    // memory_attribution.json の "cores":[...],"convolvers":[...] 断片（外側の波括弧は含めない）。
    [[nodiscard]] std::string recordsJson() const
    {
        std::lock_guard<std::mutex> lock (mutex_);
        std::string out;
        out.reserve (4096);
        out += "\"cores\":[";
        bool first = true;
        for (const auto& r : cores_)
        {
            if (! r.registered || r.state == ObjState::Empty)
                continue;
            if (! first) out += ",";
            first = false;
            char buf[320];
            std::snprintf (buf, sizeof (buf),
                           "{\"id\":%llu,\"state\":\"%s\",\"generation\":%llu,"
                           "\"measuredBytes\":%zu,\"createdUs\":%llu,\"destroyedUs\":%llu}",
                           static_cast<unsigned long long> (r.id),
                           stateName (r.state),
                           static_cast<unsigned long long> (r.generation),
                           r.measuredBytes,
                           static_cast<unsigned long long> (r.createdUs),
                           static_cast<unsigned long long> (r.destroyedUs));
            out += buf;
        }
        out += "],\"convolvers\":[";
        first = true;
        for (const auto& r : convs_)
        {
            if (! r.registered || r.state == ObjState::Empty)
                continue;
            if (! first) out += ",";
            first = false;
            char buf[320];
            std::snprintf (buf, sizeof (buf),
                           "{\"id\":%llu,\"state\":\"%s\",\"measuredBytes\":%zu,"
                           "\"createdUs\":%llu,\"destroyedUs\":%llu}",
                           static_cast<unsigned long long> (r.id),
                           stateName (r.state),
                           r.measuredBytes,
                           static_cast<unsigned long long> (r.createdUs),
                           static_cast<unsigned long long> (r.destroyedUs));
            out += buf;
        }
        out += "]";
        return out;
    }

private:
    ObjectRegistry() noexcept = default;

    static std::uint64_t nowUs() noexcept
    {
        using namespace std::chrono;
        return static_cast<std::uint64_t> (
            duration_cast<microseconds> (steady_clock::now().time_since_epoch()).count());
    }

    static const char* stateName (ObjState s) noexcept
    {
        switch (s)
        {
            case ObjState::Live:      return "live";
            case ObjState::Deferred:  return "deferred";
            case ObjState::Destroyed: return "destroyed";
            case ObjState::Empty:
            default:                  return "empty";
        }
    }

    // 空きスロットを確保する。破棄済みの再利用は recycleCount、満杯は overflowCount。
    template <typename Array>
    ObjectRecord* acquireSlot (Array& slots, std::uint64_t id, ObjKind kind, bool idValid) noexcept
    {
        for (auto& r : slots)
        {
            if (! r.registered)
            {
                r.registered = true;
                r.kind = kind;
                r.idValid = idValid;
                r.ptr = nullptr;
                return &r;
            }
        }

        for (auto& r : slots)
        {
            if (r.state == ObjState::Destroyed)
            {
                ++recycleCount_;
                r.id = 0;
                r.generation = 0;
                r.createdUs = 0;
                r.destroyedUs = 0;
                r.measuredBytes = 0;
                r.idValid = idValid;
                r.kind = kind;
                return &r;
            }
        }

        ++overflowCount_;
        ++unregisteredCreated_;
        return nullptr;
    }

    template <typename Array>
    ObjectRecord* findRegistered (Array& slots, std::uint64_t id) noexcept
    {
        if (id == 0)
            return nullptr;

        ObjectRecord* found = nullptr;
        for (auto& r : slots)
        {
            if (r.registered && r.id == id && r.state != ObjState::Destroyed)
            {
                if (found != nullptr)
                    ++duplicateIds_;
                else
                    found = &r;
            }
        }
        return found;
    }

    void pushHistory (const ObjectRecord& record) noexcept
    {
        auto& slot = history_[historyHead_ % kHistorySlots];
        if (slot.registered)
            ++historyOverwriteCount_;
        slot = record;
        ++historyHead_;
    }

    void logf (const char* fmt, ...) noexcept
    {
        char buf[512];
        va_list args;
        va_start (args, fmt);
        std::vsnprintf (buf, sizeof (buf), fmt, args);
        va_end (args);

        // 直近のイベントは常にリングへ残す（起動初期の取りこぼし対策）。
        recentLog_[recentHead_ % kRecentLogSlots] = buf;
        ++recentHead_;
        if (recentCount_ < kRecentLogSlots)
            ++recentCount_;

        const LogSink sink = logSink_.load (std::memory_order_acquire);
        if (sink == nullptr)
        {
            // sink 設定前は固定長バッファに退避し、設定時にまとめて出す（取りこぼさない）。
            if (pendingLogCount_ < pendingLog_.size())
                pendingLog_[pendingLogCount_++] = buf;
            return;
        }

        sink (buf);
    }


    mutable std::mutex mutex_;
    std::atomic<LogSink> logSink_ { nullptr };

    // sink 設定前のイベント退避（固定長・溢れは捨てる）。
    static constexpr std::size_t kPendingLogSlots = 64;
    std::array<std::string, kPendingLogSlots> pendingLog_ {};
    std::size_t pendingLogCount_ { 0 };

    // 直近イベントのリング（起動初期の flush 用）。
    static constexpr std::size_t kRecentLogSlots = 128;
    std::array<std::string, kRecentLogSlots> recentLog_ {};
    std::size_t recentHead_ { 0 };
    std::size_t recentCount_ { 0 };

    std::array<ObjectRecord, kCoreSlots> cores_ {};
    std::array<ObjectRecord, kConvolverSlots> convs_ {};
    std::array<ObjectRecord, kHistorySlots> history_ {};
    std::size_t historyHead_ { 0 };

    std::uint64_t nextConvolverId_ { 1 };

    std::uint32_t liveCores_ { 0 };
    std::uint32_t liveOnlyCores_ { 0 };
    std::uint32_t deferredCores_ { 0 };
    std::uint32_t destroyedCores_ { 0 };
    std::uint32_t liveConvolvers_ { 0 };
    std::uint32_t createdTotal_ { 0 };
    std::uint32_t destroyedTotal_ { 0 };
    std::uint32_t unregisteredCreated_ { 0 };
    std::uint32_t unregisteredDestroyed_ { 0 };
    std::uint32_t overflowCount_ { 0 };
    std::uint32_t recycleCount_ { 0 };
    std::uint32_t historyOverwriteCount_ { 0 };
    std::uint32_t uuidZeroCount_ { 0 };
    std::uint32_t backwardTransitions_ { 0 };
    std::uint32_t duplicateIds_ { 0 };
};

// 便利関数（呼び出し側の可読性のため）
inline ObjectRegistry& registry() noexcept { return ObjectRegistry::instance(); }

} // namespace convo::diag

#endif // CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
