//==============================================================================
// ReaderMeasureRegistry.h  (2026-11-11 Owner 限定承認)
//
// RCUReader と音声 reader の「最外層の入場〜退出」をインスタンス単位で記録する。
//   - 退出済みレコードはタグ別集計へ加算してから再利用する（飽和しない）。
//   - active 中のレコードは上書きしない。
//   - 記録失敗・集計取りこぼしをカウンタで明示する。
//   - すべて atomic。動的確保・mutex・ファイル出力は行わない（出力は NonRT 側）。
// EBR の入場・退出、epoch、所有権、reclaim 判定は変更しない。
//==============================================================================
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <functional>

#include "audioengine/AtomicAccess.h"   // ★ atomic 直接 dot-call を廃止

namespace convo
{

class ReaderMeasureRegistry
{
public:
    static constexpr int kRecords = 128;
    static constexpr int kTags = 32;

    struct Record
    {
        std::atomic<uint8_t>  state{0};          // 0=free 1=active
        std::atomic<uint64_t> enterUs{0};
        std::atomic<uint64_t> instKey{0};
        std::atomic<uint64_t> provKey{0};
        std::atomic<uint64_t> threadTag{0};
        std::atomic<uint32_t> slot{0xFFFFFFFFu};
        std::atomic<uint32_t> depth{0};
        std::atomic<uint32_t> seq{0};
        char tag[32]{};
    };

    struct Agg
    {
        std::atomic<uint32_t> used{0};
        std::atomic<uint64_t> count{0};
        std::atomic<uint64_t> maxHoldUs{0};
        std::atomic<uint64_t> sumHoldUs{0};
        char tag[32]{};
    };

    static inline Record recs_[kRecords]{};
    static inline Agg aggs_[kTags]{};
    static inline std::atomic<uint32_t> seq_{0};
    // 診断カウンタ
    static inline std::atomic<uint64_t> recordFailures_{0};   // 空きレコードが無い
    static inline std::atomic<uint64_t> closed_{0};           // 正常に退出した区画
    static inline std::atomic<uint64_t> aggMisses_{0};        // タグ表が満杯
    static inline std::atomic<uint64_t> enterFailures_{0};    // slot 取得失敗
    static inline std::atomic<uint64_t> underflows_{0};       // 退出の underflow

    static std::uint64_t nowUs() noexcept
    {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    static std::uint64_t threadTag() noexcept
    {
        return static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }

    static int claim (const char* tag, std::uint32_t slot, std::uint32_t depth,
                      const void* instance, const void* provider) noexcept
    {
        for (int pass = 0; pass < 2; ++pass)
        {
            for (int i = 0; i < kRecords; ++i)
            {
                std::uint8_t expected = 0;
                if (convo::compareExchangeAtomic (recs_[i].state, expected, 1,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_relaxed))
                {
                    recs_[i].enterUs.store (nowUs(), std::memory_order_relaxed);
                    recs_[i].slot.store (slot, std::memory_order_relaxed);
                    recs_[i].depth.store (depth, std::memory_order_relaxed);
                    convo::publishAtomic (recs_[i].instKey, static_cast<std::uint64_t> (
                        reinterpret_cast<std::uintptr_t> (instance)), std::memory_order_relaxed);
                    convo::publishAtomic (recs_[i].provKey, static_cast<std::uint64_t> (
                        reinterpret_cast<std::uintptr_t> (provider)), std::memory_order_relaxed);
                    recs_[i].threadTag.store (threadTag(), std::memory_order_relaxed);
                    recs_[i].seq.store (seq_.fetch_add (1u, std::memory_order_relaxed) + 1u,
                                        std::memory_order_relaxed);
                    std::memset (recs_[i].tag, 0, sizeof (recs_[i].tag));
                    if (tag != nullptr)
                        std::strncpy (recs_[i].tag, tag, sizeof (recs_[i].tag) - 1);
                    else
                        std::strncpy (recs_[i].tag, "untagged", sizeof (recs_[i].tag) - 1);
                    return i;
                }
            }
        }
        recordFailures_.fetch_add (1u, std::memory_order_relaxed);
        return -1;
    }

    static void close (int idx) noexcept
    {
        if (idx < 0 || idx >= kRecords)
            return;

        const std::uint64_t enter = recs_[idx].enterUs.load (std::memory_order_relaxed);
        const std::uint64_t hold = nowUs() - enter;
        const char* tag = recs_[idx].tag;

        // タグ別集計（レコード解放前に加算する）
        int aggIdx = -1;
        for (int i = 0; i < kTags; ++i)
        {
            if (convo::consumeAtomic (aggs_[i].used, std::memory_order_acquire) == 1
                && std::strncmp (aggs_[i].tag, tag, sizeof (aggs_[i].tag)) == 0)
            {
                aggIdx = i;
                break;
            }
        }
        if (aggIdx < 0)
        {
            for (int i = 0; i < kTags; ++i)
            {
                std::uint32_t expected = 0;
                if (convo::compareExchangeAtomic (aggs_[i].used, expected, 1,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_relaxed))
                {
                    std::memset (aggs_[i].tag, 0, sizeof (aggs_[i].tag));
                    std::strncpy (aggs_[i].tag, tag, sizeof (aggs_[i].tag) - 1);
                    aggIdx = i;
                    break;
                }
            }
        }
        if (aggIdx >= 0)
        {
            aggs_[aggIdx].count.fetch_add (1u, std::memory_order_relaxed);
            aggs_[aggIdx].sumHoldUs.fetch_add (hold, std::memory_order_relaxed);
            std::uint64_t expectedMax = aggs_[aggIdx].maxHoldUs.load (std::memory_order_relaxed);
            while (hold > expectedMax
                   && ! convo::compareExchangeWeakAtomic (aggs_[aggIdx].maxHoldUs, expectedMax, hold,
                                                          std::memory_order_relaxed,
                                                          std::memory_order_relaxed))
            {
            }
        }
        else
        {
            aggMisses_.fetch_add (1u, std::memory_order_relaxed);
        }

        convo::publishAtomic (recs_[idx].state, 0, std::memory_order_release);   // 再利用
        closed_.fetch_add (1u, std::memory_order_relaxed);
    }

    // NonRT から呼ぶこと。
    static void dump (std::FILE* f, const char* stage) noexcept
    {
        if (f == nullptr)
            return;

        const std::uint64_t now = nowUs();
        std::fprintf (f, "{\n  \"schema\": \"reader_registry_v1\",\n");
        std::fprintf (f, "  \"stage\": \"%s\",\n", stage != nullptr ? stage : "");
        std::fprintf (f, "  \"capacity\": %d,\n", kRecords);
        std::fprintf (f, "  \"counters\": {\"closed\": %llu, \"recordFailures\": %llu, \"aggMisses\": %llu, \"enterFailures\": %llu, \"underflows\": %llu},\n",
                      (unsigned long long) closed_.load (std::memory_order_relaxed),
                      (unsigned long long) recordFailures_.load (std::memory_order_relaxed),
                      (unsigned long long) aggMisses_.load (std::memory_order_relaxed),
                      (unsigned long long) enterFailures_.load (std::memory_order_relaxed),
                      (unsigned long long) underflows_.load (std::memory_order_relaxed));

        std::fprintf (f, "  \"exitedByTag\": [");
        bool first = true;
        for (int i = 0; i < kTags; ++i)
        {
            if (convo::consumeAtomic (aggs_[i].used, std::memory_order_acquire) != 1)
                continue;
            std::fprintf (f, "%s\n    {\"tag\": \"%s\", \"count\": %llu, \"maxHoldUs\": %llu, \"sumHoldUs\": %llu}",
                          first ? "" : ",", aggs_[i].tag,
                          (unsigned long long) aggs_[i].count.load (std::memory_order_relaxed),
                          (unsigned long long) aggs_[i].maxHoldUs.load (std::memory_order_relaxed),
                          (unsigned long long) aggs_[i].sumHoldUs.load (std::memory_order_relaxed));
            first = false;
        }
        std::fprintf (f, "%s],\n", first ? "" : "\n  ");

        std::fprintf (f, "  \"activeAtStage\": [");
        bool firstA = true;
        for (int i = 0; i < kRecords; ++i)
        {
            if (convo::consumeAtomic (recs_[i].state, std::memory_order_acquire) != 1)
                continue;
            const std::uint64_t enter = recs_[i].enterUs.load (std::memory_order_relaxed);
            std::fprintf (f,
                "%s\n    {\"tag\": \"%s\", \"slot\": %u, \"depth\": %u, \"thread\": %llu, \"instance\": \"0x%llx\", \"provider\": \"0x%llx\", \"enterUs\": %llu, \"holdUs\": %llu, \"seq\": %u}",
                firstA ? "" : ",", recs_[i].tag,
                recs_[i].slot.load (std::memory_order_relaxed),
                recs_[i].depth.load (std::memory_order_relaxed),
                (unsigned long long) recs_[i].threadTag.load (std::memory_order_relaxed),
                (unsigned long long) recs_[i].instKey.load (std::memory_order_relaxed),
                (unsigned long long) recs_[i].provKey.load (std::memory_order_relaxed),
                (unsigned long long) enter,
                (unsigned long long) (now - enter),
                recs_[i].seq.load (std::memory_order_relaxed));
            firstA = false;
        }
        std::fprintf (f, "%s]\n}\n", firstA ? "" : "\n  ");
    }
};

} // namespace convo
