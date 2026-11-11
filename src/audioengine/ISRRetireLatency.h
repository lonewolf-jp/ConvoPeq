#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../core/RetireDomain.h"   // ★ domain tag (v2.9): ドメイン別集計


namespace convo {
namespace isr {

/**
 * Retire latency recorder (D1: retire request -> actual free)
 *
 * 設計上の約束:
 *  - 固定容量のスロット配列とアトミック操作のみを使う。動的確保もロックもしない。
 *  - requestId は単調増加で、0 を払い出さない。ラップしたら払い出しを止める（再利用しない）。
 *  - 対応付けはポインタ（オブジェクト同一性）で行い、epoch は記録用に持つだけ。
 *  - 完了は CAS で 1 回だけ取り出す。2 回目の完了は unmatched として数え、上書きしない。
 *  - スロット満杯は overflow として数え、その要求は観測に含めない。
 *  - スロットは state で 3 段階（free / filling / ready）にし、書込み途中の公開を防ぐ。
 */
class RetireLatencyRecorder
{
public:
    enum class Path : std::uint8_t
    {
        Sync = 0,        // terminalReclaim の同期解放
        Drain,           // TerminalReclaimAuthority::drain
        DrainAll,        // TerminalReclaimAuthority::drainAll（終了時の全排水）
        Queue,           // DeferredDeletionQueue::reclaim
        QueueOverflow,   // DeletionQueue::enqueue の容量超過による即時解放
        Quarantine,      // RetireQuarantineStore::drain
        Count
    };

    static constexpr std::size_t kPathCount = static_cast<std::size_t> (Path::Count);
    static constexpr std::size_t kSlots = 512;
    static constexpr std::size_t kRanges = 8;                    // RT スレッドごとの区画数
    static constexpr std::size_t kRangeSize = kSlots / kRanges;  // 1 区画 64 スロット
    static constexpr std::size_t kObservationCapacity = 256;
    static constexpr std::uint64_t kThresholdNs = 50'000'000ull;   // 50 ms（変更しない）

    static const char* pathName (Path p) noexcept
    {
        switch (p)
        {
            case Path::Sync:          return "sync";
            case Path::Drain:         return "drain";
            case Path::DrainAll:      return "drainAll";
            case Path::Queue:         return "queue";
            case Path::QueueOverflow: return "queueOverflow";
            case Path::Quarantine:    return "quarantine";
            default:                  return "unknown";
        }
    }

    static RetireLatencyRecorder& instance() noexcept
    {
        static RetireLatencyRecorder recorder;
        return recorder;
    }

    // 要求を記録する。戻り値 0 は「記録できなかった」（容量不足・ID 枯渇）。
    std::uint64_t recordRequest (const void* object, std::uint64_t epoch,
                                 std::uint64_t minReaderEpoch = 0,
                                 std::uint32_t minReaderSlot = 0xFFFFFFFFu,
                                 bool* created = nullptr) noexcept
    {
        if (created != nullptr)
            *created = false;
        if (object == nullptr)
            return 0;

        // 終了処理（drainAll 以降）の要求は測定対象外として別集計する
        if (shutdownPhase_.load (std::memory_order_acquire))
        {
            addCount (postShutdown_);
            addCount (recordMisses_);   // ★ C5: 計測上の欠落を検出可能にする
            return 0;
        }

        const std::uint64_t id = nextId_.fetch_add (1u, std::memory_order_relaxed) + 1u;

        if (id == 0u || id == std::numeric_limits<std::uint64_t>::max())
        {
            addCount (idExhausted_);
            addCount (recordMisses_);   // ★ C5: 計測上の欠落を検出可能にする
            return 0;
        }

        // 同じポインタの再 enqueue（リトライ段階）では既存の要求を再利用する。
        for (std::size_t i = 0; i < kSlots; ++i)
        {
            const std::uint8_t state = slots_[i].state.load (std::memory_order_acquire);

            if (state != kStateFree
                && slots_[i].object.load (std::memory_order_acquire) == object)
            {
                // 既存記録。作成フラグは false のまま。未確定の最小 reader 情報だけ補完する
                //   （既知の情報は上書きしない）。
                if (minReaderSlot != 0xFFFFFFFFu
                    && slots_[i].minReaderSlot.load (std::memory_order_relaxed) == 0xFFFFFFFFu)
                {
                    slots_[i].minReaderEpoch.store (minReaderEpoch, std::memory_order_relaxed);
                    slots_[i].minReaderSlot.store (minReaderSlot, std::memory_order_relaxed);
                }

                return slots_[i].id.load (std::memory_order_relaxed);
            }
        }

        for (std::size_t i = 0; i < kSlots; ++i)
        {
            std::uint8_t expected = kStateFree;

            if (! slots_[i].state.compare_exchange_strong (expected, kStateFilling,
                                                           std::memory_order_acq_rel,
                                                           std::memory_order_relaxed))
            {
                continue;
            }

            slots_[i].object.store (object, std::memory_order_relaxed);
            slots_[i].id.store (id, std::memory_order_relaxed);
            slots_[i].tsNs.store (nowNs(), std::memory_order_relaxed);
            slots_[i].safeTsNs.store (0, std::memory_order_relaxed);
            slots_[i].epoch.store (epoch, std::memory_order_relaxed);
            slots_[i].minReaderEpoch.store (minReaderEpoch, std::memory_order_relaxed);
            slots_[i].minReaderSlot.store (minReaderSlot, std::memory_order_relaxed);
            slots_[i].state.store (kStateReady, std::memory_order_release);

            addCount (requests_);

            if (created != nullptr)
                *created = true;

            return id;
        }

        addCount (overflow_);
        addCount (recordMisses_);   // ★ C5: 計測上の欠落を検出可能にする
        return 0;
    }

    // RT 経路用: TLS で確保した区画の中だけを走査する（O(kRangeSize)、確保・ロックなし）。
    std::uint64_t recordRequestRT (const void* object, std::uint64_t epoch,
                                   std::uint64_t minReaderEpoch = 0,
                                   bool* created = nullptr) noexcept
    {
        if (created != nullptr)
            *created = false;
        if (object == nullptr)
            return 0;

        if (shutdownPhase_.load (std::memory_order_acquire))
        {
            addCount (postShutdown_);
            addCount (recordMisses_);   // ★ C5: 計測上の欠落を検出可能にする
            return 0;
        }

        const std::uint64_t id = nextId_.fetch_add (1u, std::memory_order_relaxed) + 1u;

        if (id == 0u || id == std::numeric_limits<std::uint64_t>::max())
        {
            addCount (idExhausted_);
            addCount (recordMisses_);   // ★ C5: 計測上の欠落を検出可能にする
            return 0;
        }

        const std::uint32_t range = rangeForCurrentThread();
        const std::size_t begin = static_cast<std::size_t> (range) * kRangeSize;
        const std::size_t end = begin + kRangeSize;

        for (std::size_t i = begin; i < end; ++i)
        {
            std::uint8_t expected = kStateFree;

            if (! slots_[i].state.compare_exchange_strong (expected, kStateFilling,
                                                           std::memory_order_acq_rel,
                                                           std::memory_order_relaxed))
            {
                continue;
            }

            slots_[i].object.store (object, std::memory_order_relaxed);
            slots_[i].id.store (id, std::memory_order_relaxed);
            slots_[i].tsNs.store (nowNs(), std::memory_order_relaxed);
            slots_[i].safeTsNs.store (0, std::memory_order_relaxed);
            slots_[i].epoch.store (epoch, std::memory_order_relaxed);
            slots_[i].minReaderEpoch.store (minReaderEpoch, std::memory_order_relaxed);
            slots_[i].minReaderSlot.store (0xFFFFFFFFu, std::memory_order_relaxed);
            slots_[i].state.store (kStateReady, std::memory_order_release);

            addCount (requests_);
            rtRequests_.fetch_add (1u, std::memory_order_acq_rel);

            if (created != nullptr)
                *created = true;

            return id;
        }

        addCount (overflow_);
        addCount (recordMisses_);   // ★ C5: 計測上の欠落を検出可能にする
        return 0;
    }

    // 指定 ID の記録だけを取り消す（object・id・ready の 3 条件が一致したときのみ）。
    void cancelRequestById (const void* object, std::uint64_t requestId) noexcept
    {
        if (object == nullptr || requestId == 0u)
            return;

        for (std::size_t i = 0; i < kSlots; ++i)
        {
            if (slots_[i].state.load (std::memory_order_acquire) != kStateReady)
                continue;

            if (slots_[i].object.load (std::memory_order_acquire) != object)
                continue;

            if (slots_[i].id.load (std::memory_order_relaxed) != requestId)
                continue;

            std::uint8_t expected = kStateReady;

            if (slots_[i].state.compare_exchange_strong (expected, kStateFree,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_acquire))
            {
                addCount (cancelled_);
            }

            return;
        }
    }

    // RT 版: TLS 区画内だけを走査する（O(kRangeSize)、ロック・確保なし）。
    void cancelRequestRTById (const void* object, std::uint64_t requestId) noexcept
    {
        if (object == nullptr || requestId == 0u)
            return;

        const std::uint32_t range = rangeForCurrentThread();
        const std::size_t begin = static_cast<std::size_t> (range) * kRangeSize;
        const std::size_t end = begin + kRangeSize;

        for (std::size_t i = begin; i < end; ++i)
        {
            if (slots_[i].state.load (std::memory_order_acquire) != kStateReady)
                continue;

            if (slots_[i].object.load (std::memory_order_acquire) != object)
                continue;

            if (slots_[i].id.load (std::memory_order_relaxed) != requestId)
                continue;

            std::uint8_t expected = kStateReady;

            if (slots_[i].state.compare_exchange_strong (expected, kStateFree,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_acquire))
            {
                addCount (cancelled_);
            }

            return;
        }
    }

    // enqueue に失敗した要求を取り消す（観測に含めない）。
    void cancelRequest (const void* object) noexcept
    {
        std::size_t index = 0;

        if (claim (object, index))
        {
            slots_[index].state.store (kStateFree, std::memory_order_release);
            addCount (cancelled_);
        }
    }

    // EBR 安全条件を満たした時刻を記録する（最初の 1 回だけ）。
    void recordSafe (const void* object) noexcept
    {
        if (object == nullptr)
            return;

        for (std::size_t i = 0; i < kSlots; ++i)
        {
            if (slots_[i].state.load (std::memory_order_acquire) != kStateReady)
                continue;

            if (slots_[i].object.load (std::memory_order_acquire) != object)
                continue;

            const std::uint64_t now = nowNs();
            std::uint64_t expected = 0;

            slots_[i].safeTsNs.compare_exchange_strong (expected, now,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_relaxed);
            return;
        }
    }

    // 解放完了を記録する。対応する要求が無ければ unmatched として数える。
    // ★ domain tag (v2.9 §4.2/§5): domain は enqueue 時の刻印（Unstamped=刻印漏れ・Unknown=意図的未特定）。
    //   providerAbsent は Router の provider_ == nullptr（防御的・現行 API では到達不能）のときだけ true。
    void recordComplete (const void* object, Path path, RetireDomainId domain,
                         std::uint64_t firstScanTsNs = 0, bool providerAbsent = false) noexcept
    {
        // ★ W-2: shutdown 相以降の完了を要求拒否数とは別に数える
        if (shutdownPhase_.load (std::memory_order_acquire))
            addCount (postShutdownCompletions_);

        std::size_t index = 0;

        if (! claim (object, index))
        {
            addCount (unmatchedCompletions_);
            return;
        }

        const std::uint64_t id = slots_[index].id.load (std::memory_order_relaxed);
        const std::uint64_t ts = slots_[index].tsNs.load (std::memory_order_relaxed);
        const std::uint64_t safeTs = slots_[index].safeTsNs.load (std::memory_order_relaxed);
        const std::uint64_t epoch = slots_[index].epoch.load (std::memory_order_relaxed);
        const std::uint64_t minReaderEpoch = slots_[index].minReaderEpoch.load (std::memory_order_relaxed);
        const std::uint32_t minReaderSlot = slots_[index].minReaderSlot.load (std::memory_order_relaxed);
        slots_[index].state.store (kStateFree, std::memory_order_release);

        const std::uint64_t now = nowNs();
        const std::uint64_t latency = (now >= ts) ? (now - ts) : 0u;

        addCount (completions_);
        accumulatePath (path, latency);

        // EBR 待ちと実行待ちの分離（safeTs が観測できた場合のみ）
        if (safeTs != 0 && safeTs >= ts)
        {
            const std::uint64_t ebrNs = safeTs - ts;
            const std::uint64_t execNs = (now >= safeTs) ? (now - safeTs) : 0u;
            ebrCount_.fetch_add (1u, std::memory_order_acq_rel);
            ebrSumNs_.fetch_add (ebrNs, std::memory_order_acq_rel);
            execCount_.fetch_add (1u, std::memory_order_acq_rel);
            execSumNs_.fetch_add (execNs, std::memory_order_acq_rel);
            updateMax (ebrMaxNs_, ebrNs);
            updateMax (execMaxNs_, execNs);
        }
        else
        {
            unobservedSafe_.fetch_add (1u, std::memory_order_acq_rel);
        }

        // ★ 段階③ (2026-11-11, Owner 承認): 3 分類の計算と集計
        //   既存の latency / ebr / exec の計算は上で終っている（判定は完全に不変）。
        //   safeTs は生の値を渡す（safeTs < ts の順序不整合を
        //   excluded に落とさず consistencyErrors に拾うため）。
        const Classification cls = classifyTimes (ts, firstScanTsNs, safeTs, now);
        const std::size_t pathIndex = static_cast<std::size_t> (path);

        // ★ domain tag (v2.9 §5): ドメイン別の集計（既存の paths / totals は不変・追加のみ）
        accumulateDomain (domainIndexFor (domain), path, cls, latency);
        if (domain == RetireDomainId::Unstamped)
            addCount (domainStampMissing_);
        else if (domain == RetireDomainId::Unknown)
            addCount (providerAbsent ? domainProviderAbsent_ : domainProviderUnknown_);
        if (cls.excluded)
        {
            if (pathIndex < kPathCount)
                classExcluded_[pathIndex].fetch_add (1u, std::memory_order_acq_rel);
            classExcludedTotal_.fetch_add (1u, std::memory_order_acq_rel);
        }
        else if (cls.inconsistent)
        {
            classConsistencyErrors_.fetch_add (1u, std::memory_order_acq_rel);
            accumulateUnclassified (pathIndex, cls.unclassifiedNs);
        }
        else if (cls.classified)
        {
            classClassifiedTotal_.fetch_add (1u, std::memory_order_acq_rel);
            accumulateComponent (driverCount_[pathIndex], driverSumNs_[pathIndex],
                                 driverMaxNs_[pathIndex], driverOver_[pathIndex], cls.driverWaitNs);
            accumulateComponent (ebrCountC_[pathIndex], ebrSumNsC_[pathIndex],
                                 ebrMaxNsC_[pathIndex], ebrOverC_[pathIndex], cls.ebrWaitNs);
            accumulateComponent (execCountC_[pathIndex], execSumNsC_[pathIndex],
                                 execMaxNsC_[pathIndex], execOverC_[pathIndex], cls.execWaitNs);
            // ★ 残差が実際にあるときだけ unclassified に数える（
            //   分類できた要求を unclassified に加算しない）。
            if (cls.unclassifiedNs > 0u)
                accumulateUnclassified (pathIndex, cls.unclassifiedNs);
        }
        else
        {
            accumulateUnclassified (pathIndex, cls.unclassifiedNs);
        }

        appendObservation (id, epoch, latency, path, ts,
                           (safeTs != 0 && safeTs >= ts) ? safeTs : 0,
                           minReaderEpoch, minReaderSlot,
                           firstScanTsNs, cls);
    }

    struct PathStats
    {
        std::uint64_t count = 0;
        std::uint64_t overThreshold = 0;
        std::uint64_t maxNs = 0;
    };

    // ★ 段階③: 3 分類のパス別集計（診断用。形式は設計書 §2 の JSON に対応）
    struct ClassComponent
    {
        std::uint64_t count = 0;
        std::uint64_t sumNs = 0;
        std::uint64_t maxNs = 0;
        std::uint64_t overThreshold = 0;
    };
    struct ClassifiedPath
    {
        ClassComponent driverWait;
        ClassComponent ebrWait;
        ClassComponent execWait;
        std::uint64_t unclassifiedCount = 0;
        std::uint64_t unclassifiedSumNs = 0;
        std::uint64_t excludedCount = 0;   // safeTs 未観測（分類対象外）
    };
    // ★ domain tag (v2.9 §5.1): ドメイン別の 1 行分（4 行を常に出力する）
    struct DomainStats
    {
        std::uint64_t count = 0;
        std::uint64_t classified = 0;
        std::uint64_t unclassified = 0;
        std::uint64_t excluded = 0;
        std::uint64_t consistencyErrors = 0;
        std::uint64_t sumNs = 0;
        std::uint64_t maxNs = 0;
        std::uint64_t overThreshold = 0;
        ClassifiedPath paths[kPathCount];
    };

    struct ClassifiedTotals
    {
        std::uint64_t requests = 0;
        std::uint64_t completions = 0;
        std::uint64_t classified = 0;
        std::uint64_t unclassified = 0;
        std::uint64_t excluded = 0;
        std::uint64_t consistencyErrors = 0;
        std::uint64_t pending = 0;
    };

    // ★ 段階③: 時刻から 3 分類を計算する純関数（テスト可能にするため public）
    struct Classification
    {
        bool excluded = false;      // safeTs 未観測（分類しない）
        bool inconsistent = false;  // 順序不整合（unclassified に全額）
        bool classified = false;
        std::uint64_t driverWaitNs = 0;
        std::uint64_t ebrWaitNs = 0;
        std::uint64_t execWaitNs = 0;
        std::uint64_t unclassifiedNs = 0;
    };
    static Classification classifyTimes (std::uint64_t ts,
                                         std::uint64_t firstScanTs,
                                         std::uint64_t safeTs,
                                         std::uint64_t now) noexcept
    {
        Classification c;
        const std::uint64_t latency = (now >= ts) ? (now - ts) : 0u;
        if (safeTs == 0)
        {
            c.excluded = true;            // 既存の unobservedSafe に計上される層
            return c;
        }
        if (firstScanTs == 0)
        {
            c.unclassifiedNs = latency;   // 一度も走査されない（drainAll など）
            return c;
        }
        if (firstScanTs < ts || safeTs < firstScanTs)
        {
            c.inconsistent = true;
            c.unclassifiedNs = latency;
            return c;
        }
        c.driverWaitNs = firstScanTs - ts;
        c.ebrWaitNs = safeTs - firstScanTs;
        c.execWaitNs = (now >= safeTs) ? (now - safeTs) : 0u;
        const std::uint64_t sum = c.driverWaitNs + c.ebrWaitNs + c.execWaitNs;
        if (sum > latency)
        {
            c.inconsistent = true;        // 時刻の年代があわない（0 埋めしない）
            c.driverWaitNs = 0;
            c.ebrWaitNs = 0;
            c.execWaitNs = 0;
            c.unclassifiedNs = latency;
            return c;
        }
        c.classified = true;
        c.unclassifiedNs = latency - sum;  // 実際の残差（0 埋めしない）
        return c;
    }

    struct Observation
    {
        std::uint64_t requestId = 0;
        std::uint64_t epoch = 0;
        std::uint64_t latencyNs = 0;
        std::uint8_t path = 0;
        std::uint64_t enqueueTsNs = 0;     // 要求時刻
        std::uint64_t safeTsNs = 0;        // EBR 安全到達時刻（0 = 未観測）
        std::uint64_t minReaderEpoch = 0;  // 要求時の最小 reader epoch
        std::uint32_t minReaderSlot = 0xFFFFFFFFu; // その保持スロット
        // ★ 段階③ (2026-11-11, Owner 承認): 3 分類診断（追加・既存の判定は不変）
        std::uint64_t firstScanTsNs = 0;   // 最初にreclaim走査で評価された時刻（0 = 未走査）
        std::uint64_t driverWaitNs = 0;    // firstScanTs - ts
        std::uint64_t ebrWaitNs = 0;       // safeTs - firstScanTs
        std::uint64_t execWaitNs = 0;      // now - safeTs
        std::uint64_t unclassifiedNs = 0;  // latency - (driver+ebr+exec)（負なら latency 全額）
        std::uint8_t classified = 0;       // 1 = 3 分類成立 / 0 = 未分類
    };

    struct Snapshot
    {
        std::uint64_t requests = 0;
        std::uint64_t completions = 0;
        std::uint64_t unmatchedCompletions = 0;
        std::uint64_t overflow = 0;
        std::uint64_t idExhausted = 0;
        std::uint64_t cancelled = 0;
        std::uint64_t postShutdownRequests = 0;   // drainAll 以降の要求
        std::uint64_t postShutdownCompletions = 0; // ★ W-2: shutdown 相以降の完了
        std::uint64_t recordMisses = 0;           // ★ C5: 記録できなかった要求
        std::uint64_t ebrCount = 0;
        std::uint64_t ebrSumNs = 0;
        std::uint64_t ebrMaxNs = 0;
        std::uint64_t execCount = 0;
        std::uint64_t execSumNs = 0;
        std::uint64_t execMaxNs = 0;
        std::uint64_t unobservedSafe = 0;
        std::uint64_t rtRequests = 0;
        PathStats paths[kPathCount];
        // ★ 段階③: 3 分類（診断用の追加情報）
        ClassifiedPath classifiedPaths[kPathCount];
        ClassifiedTotals classTotals;
        // ★ domain tag (v2.9 §5): ドメイン別の集計と原因カウンタ
        DomainStats domains[static_cast<std::size_t> (kRetireDomainCount)];
        std::uint64_t stampMissing = 0;
        std::uint64_t providerUnknown = 0;
        std::uint64_t providerAbsent = 0;
        std::vector<Observation> observations;
        std::vector<Observation> pending;   // 完了が観測できていない要求（先頭 8 件）
    };

    // 2 段階記録（Owner 条件）: 最終 drain 直前の状態を保持する
    struct StageStats
    {
        bool valid = false;
        std::uint64_t requests = 0;
        std::uint64_t completions = 0;
        std::uint64_t unmatchedCompletions = 0;
        std::uint64_t overflow = 0;
        std::uint64_t idExhausted = 0;
        std::uint64_t postShutdownRequests = 0;
        std::uint64_t postShutdownCompletions = 0; // ★ W-2
        std::uint64_t recordMisses = 0;   // ★ C5: 記録できなかった要求
        std::uint64_t pending = 0;
        std::uint64_t ebrCount = 0;
        std::uint64_t ebrSumNs = 0;
        std::uint64_t ebrMaxNs = 0;
        std::uint64_t execCount = 0;
        std::uint64_t execSumNs = 0;
        std::uint64_t execMaxNs = 0;
        std::uint64_t unobservedSafe = 0;
        std::uint64_t rtRequests = 0;
        PathStats paths[kPathCount];
    };

    void captureRuntimeStage() noexcept
    {
        StageStats s;
        s.valid = true;
        s.requests = requests_.load (std::memory_order_acquire);
        s.completions = completions_.load (std::memory_order_acquire);
        s.unmatchedCompletions = unmatchedCompletions_.load (std::memory_order_acquire);
        s.overflow = overflow_.load (std::memory_order_acquire);
        s.idExhausted = idExhausted_.load (std::memory_order_acquire);
        s.postShutdownRequests = postShutdown_.load (std::memory_order_acquire);
        s.postShutdownCompletions = postShutdownCompletions_.load (std::memory_order_acquire);
        s.recordMisses = recordMisses_.load (std::memory_order_acquire);
        s.ebrCount = ebrCount_.load (std::memory_order_acquire);
        s.ebrSumNs = ebrSumNs_.load (std::memory_order_acquire);
        s.ebrMaxNs = ebrMaxNs_.load (std::memory_order_acquire);
        s.execCount = execCount_.load (std::memory_order_acquire);
        s.execSumNs = execSumNs_.load (std::memory_order_acquire);
        s.execMaxNs = execMaxNs_.load (std::memory_order_acquire);
        s.unobservedSafe = unobservedSafe_.load (std::memory_order_acquire);
        s.rtRequests = rtRequests_.load (std::memory_order_acquire);
        s.pending = (s.requests > s.completions) ? (s.requests - s.completions) : 0;

        for (std::size_t p = 0; p < kPathCount; ++p)
        {
            s.paths[p].count = pathCount_[p].load (std::memory_order_acquire);
            s.paths[p].overThreshold = pathOver_[p].load (std::memory_order_acquire);
            s.paths[p].maxNs = pathMaxNs_[p].load (std::memory_order_acquire);
        }

        std::size_t slot = 0;
        const std::size_t current = runtimeStageSlot_.load (std::memory_order_acquire);
        slot = (current == 0u) ? 1u : 0u;
        runtimeStage_[slot] = s;
        runtimeStageSlot_.store (static_cast<std::size_t> (slot + 1u), std::memory_order_release);
    }

    // ★ W-2 (2026-11-10): 違反レポート生成時点（final 段）の累積値を保持する。
    //   postTeardown との差分を取るために使う。判定には使わない。
    void captureFinalStage() noexcept
    {
        const Snapshot s = snapshot();
        StageStats st;
        st.valid = true;
        st.requests = s.requests;
        st.completions = s.completions;
        st.unmatchedCompletions = s.unmatchedCompletions;
        st.overflow = s.overflow;
        st.idExhausted = s.idExhausted;
        st.postShutdownRequests = s.postShutdownRequests;
        st.postShutdownCompletions = s.postShutdownCompletions;
        st.recordMisses = s.recordMisses;
        st.pending = (s.requests > s.completions) ? (s.requests - s.completions) : 0;
        st.ebrCount = s.ebrCount;
        st.ebrSumNs = s.ebrSumNs;
        st.ebrMaxNs = s.ebrMaxNs;
        st.execCount = s.execCount;
        st.execSumNs = s.execSumNs;
        st.execMaxNs = s.execMaxNs;
        st.unobservedSafe = s.unobservedSafe;
        st.rtRequests = s.rtRequests;
        for (std::size_t p = 0; p < kPathCount; ++p)
            st.paths[p] = s.paths[p];

        finalStage_ = st;
        finalStageValid_.store (true, std::memory_order_release);
    }

    StageStats finalStage() const noexcept
    {
        return finalStage_;
    }

    StageStats runtimeStage() const noexcept
    {        const std::size_t marker = runtimeStageSlot_.load (std::memory_order_acquire);

        if (marker == 0u)
            return StageStats{};

        return runtimeStage_[(marker - 1u) == 0u ? 0u : 1u];
    }

    Snapshot snapshot() const
    {
        Snapshot s;
        s.requests = requests_.load (std::memory_order_acquire);
        s.completions = completions_.load (std::memory_order_acquire);
        s.unmatchedCompletions = unmatchedCompletions_.load (std::memory_order_acquire);
        s.overflow = overflow_.load (std::memory_order_acquire);
        s.idExhausted = idExhausted_.load (std::memory_order_acquire);
        s.cancelled = cancelled_.load (std::memory_order_acquire);
        s.postShutdownRequests = postShutdown_.load (std::memory_order_acquire);
        s.postShutdownCompletions = postShutdownCompletions_.load (std::memory_order_acquire);
        s.recordMisses = recordMisses_.load (std::memory_order_acquire);
        s.ebrCount = ebrCount_.load (std::memory_order_acquire);
        s.ebrSumNs = ebrSumNs_.load (std::memory_order_acquire);
        s.ebrMaxNs = ebrMaxNs_.load (std::memory_order_acquire);
        s.execCount = execCount_.load (std::memory_order_acquire);
        s.execSumNs = execSumNs_.load (std::memory_order_acquire);
        s.execMaxNs = execMaxNs_.load (std::memory_order_acquire);
        s.unobservedSafe = unobservedSafe_.load (std::memory_order_acquire);
        s.rtRequests = rtRequests_.load (std::memory_order_acquire);

        for (std::size_t p = 0; p < kPathCount; ++p)
        {
            s.paths[p].count = pathCount_[p].load (std::memory_order_acquire);
            s.paths[p].overThreshold = pathOver_[p].load (std::memory_order_acquire);
            s.paths[p].maxNs = pathMaxNs_[p].load (std::memory_order_acquire);
        }

        // 未完了の要求（診断用に先頭 8 件だけ）
        for (std::size_t i = 0; i < kSlots && s.pending.size() < 8; ++i)
        {
            if (slots_[i].state.load (std::memory_order_acquire) != kStateReady)
                continue;

            Observation o;
            o.requestId = slots_[i].id.load (std::memory_order_relaxed);
            o.epoch = slots_[i].epoch.load (std::memory_order_relaxed);
            o.latencyNs = 0;
            o.path = 0xFFu;
            s.pending.push_back (o);
        }

        const std::uint64_t written = observationCount_.load (std::memory_order_acquire);
        const std::uint64_t limit = written < kObservationCapacity ? written : kObservationCapacity;

        for (std::uint64_t i = 0; i < limit; ++i)
        {
            const auto& o = observations_[i];

            if (o.path == 0xFFu)
                continue;

            s.observations.push_back (o);
        }

        // ★ 段階③: 3 分類のパス別集計（診断用の追加情報）
        for (std::size_t p = 0; p < kPathCount; ++p)
        {
            auto& cp = s.classifiedPaths[p];
            cp.driverWait.count = driverCount_[p].load (std::memory_order_acquire);
            cp.driverWait.sumNs = driverSumNs_[p].load (std::memory_order_acquire);
            cp.driverWait.maxNs = driverMaxNs_[p].load (std::memory_order_acquire);
            cp.driverWait.overThreshold = driverOver_[p].load (std::memory_order_acquire);
            cp.ebrWait.count = ebrCountC_[p].load (std::memory_order_acquire);
            cp.ebrWait.sumNs = ebrSumNsC_[p].load (std::memory_order_acquire);
            cp.ebrWait.maxNs = ebrMaxNsC_[p].load (std::memory_order_acquire);
            cp.ebrWait.overThreshold = ebrOverC_[p].load (std::memory_order_acquire);
            cp.execWait.count = execCountC_[p].load (std::memory_order_acquire);
            cp.execWait.sumNs = execSumNsC_[p].load (std::memory_order_acquire);
            cp.execWait.maxNs = execMaxNsC_[p].load (std::memory_order_acquire);
            cp.execWait.overThreshold = execOverC_[p].load (std::memory_order_acquire);
            cp.unclassifiedCount = classUnclassifiedCount_[p].load (std::memory_order_acquire);
            cp.unclassifiedSumNs = classUnclassifiedSumNs_[p].load (std::memory_order_acquire);
            cp.excludedCount = classExcluded_[p].load (std::memory_order_acquire);
        }

        s.classTotals.requests = s.requests;
        s.classTotals.completions = s.completions;
        s.classTotals.classified = classClassifiedTotal_.load (std::memory_order_acquire);
        s.classTotals.unclassified = classUnclassifiedTotal_.load (std::memory_order_acquire);
        s.classTotals.excluded = classExcludedTotal_.load (std::memory_order_acquire);
        s.classTotals.consistencyErrors = classConsistencyErrors_.load (std::memory_order_acquire);
        s.classTotals.pending = (s.requests > s.completions) ? (s.requests - s.completions) : 0;

        // ★ domain tag (v2.9 §5.1): 4 ドメイン分を常に埋める（0 件でも省略しない）
        for (std::size_t d = 0; d < static_cast<std::size_t> (kRetireDomainCount); ++d)
        {
            auto& ds = s.domains[d];
            const auto& t = domainTotals_[d];
            ds.count = t.count.load (std::memory_order_acquire);
            ds.classified = t.classified.load (std::memory_order_acquire);
            ds.unclassified = t.unclassified.load (std::memory_order_acquire);
            ds.excluded = t.excluded.load (std::memory_order_acquire);
            ds.consistencyErrors = t.consistencyErrors.load (std::memory_order_acquire);
            ds.sumNs = t.sumNs.load (std::memory_order_acquire);
            ds.maxNs = t.maxNs.load (std::memory_order_acquire);
            ds.overThreshold = t.overThreshold.load (std::memory_order_acquire);

            for (std::size_t p = 0; p < kPathCount; ++p)
            {
                const auto& dp = domainPaths_[d][p];
                auto& cp = ds.paths[p];
                cp.driverWait.count = dp.driverCount.load (std::memory_order_acquire);
                cp.driverWait.sumNs = dp.driverSum.load (std::memory_order_acquire);
                cp.driverWait.maxNs = dp.driverMax.load (std::memory_order_acquire);
                cp.driverWait.overThreshold = dp.driverOver.load (std::memory_order_acquire);
                cp.ebrWait.count = dp.ebrCount.load (std::memory_order_acquire);
                cp.ebrWait.sumNs = dp.ebrSum.load (std::memory_order_acquire);
                cp.ebrWait.maxNs = dp.ebrMax.load (std::memory_order_acquire);
                cp.ebrWait.overThreshold = dp.ebrOver.load (std::memory_order_acquire);
                cp.execWait.count = dp.execCount.load (std::memory_order_acquire);
                cp.execWait.sumNs = dp.execSum.load (std::memory_order_acquire);
                cp.execWait.maxNs = dp.execMax.load (std::memory_order_acquire);
                cp.execWait.overThreshold = dp.execOver.load (std::memory_order_acquire);
                cp.unclassifiedCount = dp.unclassifiedCount.load (std::memory_order_acquire);
                cp.unclassifiedSumNs = dp.unclassifiedSum.load (std::memory_order_acquire);
                cp.excludedCount = dp.excluded.load (std::memory_order_acquire);
            }
        }

        s.stampMissing = domainStampMissing_.load (std::memory_order_acquire);
        s.providerUnknown = domainProviderUnknown_.load (std::memory_order_acquire);
        s.providerAbsent = domainProviderAbsent_.load (std::memory_order_acquire);

        return s;
    }

    // 終了処理に入ったことを記録する（以後の要求は測定対象外）
    void markShutdownPhase() noexcept
    {
        captureRuntimeStage();   // ★ 最終 drain 直前の状態を保存する
        shutdownPhase_.store (true, std::memory_order_release);
    }

    void reset() noexcept
    {
        shutdownPhase_.store (false, std::memory_order_release);
        runtimeStage_[0] = StageStats{};
        runtimeStage_[1] = StageStats{};
        runtimeStageSlot_.store (0, std::memory_order_release);
        postShutdown_.store (0, std::memory_order_release);
        postShutdownCompletions_.store (0, std::memory_order_release);
        recordMisses_.store (0, std::memory_order_release);
        for (std::size_t i = 0; i < kSlots; ++i)
        {
            slots_[i].state.store (kStateFree, std::memory_order_release);
            slots_[i].object.store (nullptr, std::memory_order_relaxed);
            slots_[i].id.store (0, std::memory_order_relaxed);
            slots_[i].tsNs.store (0, std::memory_order_relaxed);
            slots_[i].epoch.store (0, std::memory_order_relaxed);
        }

        for (std::size_t i = 0; i < kObservationCapacity; ++i)
            observations_[i] = Observation{0, 0, 0, 0xFFu};

        requests_.store (0, std::memory_order_release);
        completions_.store (0, std::memory_order_release);
        unmatchedCompletions_.store (0, std::memory_order_release);
        overflow_.store (0, std::memory_order_release);
        idExhausted_.store (0, std::memory_order_release);
        cancelled_.store (0, std::memory_order_release);
        observationCount_.store (0, std::memory_order_release);
        nextId_.store (0, std::memory_order_release);
        nextRange_.store (0, std::memory_order_release);
        ebrCount_.store (0, std::memory_order_release);
        ebrSumNs_.store (0, std::memory_order_release);
        ebrMaxNs_.store (0, std::memory_order_release);
        execCount_.store (0, std::memory_order_release);
        execSumNs_.store (0, std::memory_order_release);
        execMaxNs_.store (0, std::memory_order_release);
        unobservedSafe_.store (0, std::memory_order_release);
        rtRequests_.store (0, std::memory_order_release);

        for (std::size_t p = 0; p < kPathCount; ++p)
        {
            pathCount_[p].store (0, std::memory_order_release);
            pathOver_[p].store (0, std::memory_order_release);
            pathMaxNs_[p].store (0, std::memory_order_release);
            driverCount_[p].store (0, std::memory_order_release);
            driverSumNs_[p].store (0, std::memory_order_release);
            driverMaxNs_[p].store (0, std::memory_order_release);
            driverOver_[p].store (0, std::memory_order_release);
            ebrCountC_[p].store (0, std::memory_order_release);
            ebrSumNsC_[p].store (0, std::memory_order_release);
            ebrMaxNsC_[p].store (0, std::memory_order_release);
            ebrOverC_[p].store (0, std::memory_order_release);
            execCountC_[p].store (0, std::memory_order_release);
            execSumNsC_[p].store (0, std::memory_order_release);
            execMaxNsC_[p].store (0, std::memory_order_release);
            execOverC_[p].store (0, std::memory_order_release);
            classUnclassifiedCount_[p].store (0, std::memory_order_release);
            classUnclassifiedSumNs_[p].store (0, std::memory_order_release);
            classExcluded_[p].store (0, std::memory_order_release);
        }
        classClassifiedTotal_.store (0, std::memory_order_release);
        classUnclassifiedTotal_.store (0, std::memory_order_release);
        classUnclassifiedSumTotal_.store (0, std::memory_order_release);
        classExcludedTotal_.store (0, std::memory_order_release);
        classConsistencyErrors_.store (0, std::memory_order_release);

        // ★ domain tag: ドメイン別の集計も初期化する
        for (std::size_t d = 0; d < static_cast<std::size_t> (kRetireDomainCount); ++d)
        {
            auto& t = domainTotals_[d];
            t.count.store (0); t.classified.store (0); t.unclassified.store (0);
            t.excluded.store (0); t.consistencyErrors.store (0);
            t.sumNs.store (0); t.maxNs.store (0); t.overThreshold.store (0);
            for (std::size_t p = 0; p < kPathCount; ++p)
            {
                auto& dp = domainPaths_[d][p];
                dp.driverCount.store (0); dp.driverSum.store (0);
                dp.driverMax.store (0); dp.driverOver.store (0);
                dp.ebrCount.store (0); dp.ebrSum.store (0);
                dp.ebrMax.store (0); dp.ebrOver.store (0);
                dp.execCount.store (0); dp.execSum.store (0);
                dp.execMax.store (0); dp.execOver.store (0);
                dp.unclassifiedCount.store (0); dp.unclassifiedSum.store (0);
                dp.excluded.store (0);
            }
        }
        domainStampMissing_.store (0, std::memory_order_release);
        domainProviderUnknown_.store (0, std::memory_order_release);
        domainProviderAbsent_.store (0, std::memory_order_release);
    }

private:
    // ★ domain tag (v2.9 §5): ドメイン別の集計領域（診断専用・既存の集計は不変）
    struct DomainTotals
    {
        std::atomic<std::uint64_t> count { 0 };
        std::atomic<std::uint64_t> classified { 0 };
        std::atomic<std::uint64_t> unclassified { 0 };
        std::atomic<std::uint64_t> excluded { 0 };
        std::atomic<std::uint64_t> consistencyErrors { 0 };
        std::atomic<std::uint64_t> sumNs { 0 };
        std::atomic<std::uint64_t> maxNs { 0 };
        std::atomic<std::uint64_t> overThreshold { 0 };
    };

    struct DomainPathCounters
    {
        std::atomic<std::uint64_t> driverCount { 0 };
        std::atomic<std::uint64_t> driverSum { 0 };
        std::atomic<std::uint64_t> driverMax { 0 };
        std::atomic<std::uint64_t> driverOver { 0 };
        std::atomic<std::uint64_t> ebrCount { 0 };
        std::atomic<std::uint64_t> ebrSum { 0 };
        std::atomic<std::uint64_t> ebrMax { 0 };
        std::atomic<std::uint64_t> ebrOver { 0 };
        std::atomic<std::uint64_t> execCount { 0 };
        std::atomic<std::uint64_t> execSum { 0 };
        std::atomic<std::uint64_t> execMax { 0 };
        std::atomic<std::uint64_t> execOver { 0 };
        std::atomic<std::uint64_t> unclassifiedCount { 0 };
        std::atomic<std::uint64_t> unclassifiedSum { 0 };
        std::atomic<std::uint64_t> excluded { 0 };
    };

    static std::size_t domainIndexFor (RetireDomainId domain) noexcept
    {
        const auto raw = static_cast<std::size_t> (domain);
        return raw < static_cast<std::size_t> (kRetireDomainCount)
                   ? raw
                   : static_cast<std::size_t> (RetireDomainId::Unknown);
    }

    void accumulateDomain (std::size_t domainIdx, Path path, const Classification& cls,
                           std::uint64_t latency) noexcept
    {
        if (domainIdx >= static_cast<std::size_t> (kRetireDomainCount))
            return;

        auto& t = domainTotals_[domainIdx];
        addCount (t.count);
        t.sumNs.fetch_add (latency, std::memory_order_acq_rel);
        updateMax (t.maxNs, latency);
        if (latency > kThresholdNs)
            addCount (t.overThreshold);

        const std::size_t pathIndex = static_cast<std::size_t> (path);
        if (pathIndex >= kPathCount)
            return;

        auto& dp = domainPaths_[domainIdx][pathIndex];
        if (cls.excluded)
        {
            addCount (t.excluded);
            addCount (dp.excluded);
        }
        else if (cls.inconsistent)
        {
            addCount (t.consistencyErrors);
            addCount (t.unclassified);
            dp.unclassifiedCount.fetch_add (1u, std::memory_order_acq_rel);
            dp.unclassifiedSum.fetch_add (cls.unclassifiedNs, std::memory_order_acq_rel);
        }
        else if (cls.classified)
        {
            addCount (t.classified);
            accumulateComponent (dp.driverCount, dp.driverSum, dp.driverMax, dp.driverOver,
                                 cls.driverWaitNs);
            accumulateComponent (dp.ebrCount, dp.ebrSum, dp.ebrMax, dp.ebrOver, cls.ebrWaitNs);
            accumulateComponent (dp.execCount, dp.execSum, dp.execMax, dp.execOver,
                                 cls.execWaitNs);
            if (cls.unclassifiedNs > 0u)
            {
                addCount (t.unclassified);
                dp.unclassifiedCount.fetch_add (1u, std::memory_order_acq_rel);
                dp.unclassifiedSum.fetch_add (cls.unclassifiedNs, std::memory_order_acq_rel);
            }
        }
        else
        {
            addCount (t.unclassified);
            dp.unclassifiedCount.fetch_add (1u, std::memory_order_acq_rel);
            dp.unclassifiedSum.fetch_add (cls.unclassifiedNs, std::memory_order_acq_rel);
        }
    }

    DomainTotals domainTotals_[static_cast<std::size_t> (kRetireDomainCount)];
    DomainPathCounters domainPaths_[static_cast<std::size_t> (kRetireDomainCount)][kPathCount];
    std::atomic<std::uint64_t> domainStampMissing_ { 0 };
    std::atomic<std::uint64_t> domainProviderUnknown_ { 0 };
    std::atomic<std::uint64_t> domainProviderAbsent_ { 0 };

    static constexpr std::uint8_t kStateFree = 0;
    static constexpr std::uint8_t kStateFilling = 1;
    static constexpr std::uint8_t kStateReady = 2;

    struct Slot
    {
        std::atomic<std::uint8_t> state{kStateFree};
        std::atomic<const void*> object{nullptr};
        std::atomic<std::uint64_t> id{0};
        std::atomic<std::uint64_t> tsNs{0};        // enqueueTs
        std::atomic<std::uint64_t> safeTsNs{0};    // 0 = 未観測（強制解放など）
        std::atomic<std::uint64_t> epoch{0};       // enqueueEpoch
        std::atomic<std::uint64_t> minReaderEpoch{0};
        std::atomic<std::uint32_t> minReaderSlot{0xFFFFFFFFu};
    };

    RetireLatencyRecorder() { reset(); }

    static std::uint64_t nowNs() noexcept
    {
        using namespace std::chrono;
        return static_cast<std::uint64_t> (
            duration_cast<nanoseconds> (steady_clock::now().time_since_epoch()).count());
    }

    std::uint32_t rangeForCurrentThread() noexcept
    {
        thread_local std::uint32_t assigned = 0xFFFFFFFFu;

        if (assigned == 0xFFFFFFFFu)
            assigned = nextRange_.fetch_add (1u, std::memory_order_relaxed)
                       % static_cast<std::uint32_t> (kRanges);

        return assigned;
    }

    static void addCount (std::atomic<std::uint64_t>& counter) noexcept
    {
        counter.fetch_add (1u, std::memory_order_acq_rel);
    }

    // object に対応する ready スロットを 1 つだけ確保する（CAS で二重取得を防ぐ）。
    bool claim (const void* object, std::size_t& index) noexcept
    {
        if (object == nullptr)
            return false;

        for (std::size_t i = 0; i < kSlots; ++i)
        {
            if (slots_[i].state.load (std::memory_order_acquire) != kStateReady)
                continue;

            if (slots_[i].object.load (std::memory_order_acquire) != object)
                continue;

            std::uint8_t expected = kStateReady;

            if (slots_[i].state.compare_exchange_strong (expected, kStateFilling,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_acquire))
            {
                index = i;
                return true;
            }
        }

        return false;
    }

    // ★ 段階③: 1 成分（driver/ebr/exec）の集計
    static void accumulateComponent (std::atomic<std::uint64_t>& count,
                                     std::atomic<std::uint64_t>& sumNs,
                                     std::atomic<std::uint64_t>& maxNs,
                                     std::atomic<std::uint64_t>& overThreshold,
                                     std::uint64_t value) noexcept
    {
        count.fetch_add (1u, std::memory_order_acq_rel);
        sumNs.fetch_add (value, std::memory_order_acq_rel);
        if (value > kThresholdNs)
            overThreshold.fetch_add (1u, std::memory_order_acq_rel);
        updateMax (maxNs, value);
    }
    void accumulateUnclassified (std::size_t pathIndex, std::uint64_t value) noexcept
    {
        classUnclassifiedTotal_.fetch_add (1u, std::memory_order_acq_rel);
        classUnclassifiedSumTotal_.fetch_add (value, std::memory_order_acq_rel);
        if (pathIndex < kPathCount)
        {
            classUnclassifiedCount_[pathIndex].fetch_add (1u, std::memory_order_acq_rel);
            classUnclassifiedSumNs_[pathIndex].fetch_add (value, std::memory_order_acq_rel);
        }
    }
    static void updateMax (std::atomic<std::uint64_t>& target, std::uint64_t value) noexcept
    {
        std::uint64_t current = target.load (std::memory_order_acquire);

        while (value > current
               && ! target.compare_exchange_weak (current, value,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire))
        {
        }
    }

    void accumulatePath (Path path, std::uint64_t latency) noexcept
    {
        const std::size_t p = static_cast<std::size_t> (path);

        if (p >= kPathCount)
            return;

        pathCount_[p].fetch_add (1u, std::memory_order_acq_rel);

        if (latency > kThresholdNs)
            pathOver_[p].fetch_add (1u, std::memory_order_acq_rel);

        std::uint64_t current = pathMaxNs_[p].load (std::memory_order_acquire);

        while (latency > current
               && ! pathMaxNs_[p].compare_exchange_weak (current, latency,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_acquire))
        {
        }
    }

    void appendObservation (std::uint64_t id, std::uint64_t epoch,
                            std::uint64_t latency, Path path,
                            std::uint64_t enqueueTs, std::uint64_t safeTs,
                            std::uint64_t minReaderEpoch, std::uint32_t minReaderSlot,
                            std::uint64_t firstScanTs = 0,
                            Classification cls = Classification{}) noexcept
    {
        const std::uint64_t index = observationCount_.fetch_add (1u, std::memory_order_acq_rel);
        Observation o;
        o.requestId = id;
        o.epoch = epoch;
        o.latencyNs = latency;
        o.path = static_cast<std::uint8_t> (path);
        o.enqueueTsNs = enqueueTs;
        o.safeTsNs = safeTs;
        o.minReaderEpoch = minReaderEpoch;
        o.minReaderSlot = minReaderSlot;
        // ★ 段階③: 3 分類の値（診断用。既存フィールドは不変）
        o.firstScanTsNs = firstScanTs;
        o.driverWaitNs = cls.driverWaitNs;
        o.ebrWaitNs = cls.ebrWaitNs;
        o.execWaitNs = cls.execWaitNs;
        o.unclassifiedNs = cls.unclassifiedNs;
        o.classified = (cls.classified ? 1u : 0u);
        observations_[index % kObservationCapacity] = o;
    }

    Slot slots_[kSlots];
    Observation observations_[kObservationCapacity];

    std::atomic<std::uint64_t> nextId_{0};
    std::atomic<std::uint32_t> nextRange_{0};
    std::atomic<std::uint64_t> requests_{0};
    std::atomic<std::uint64_t> completions_{0};
    std::atomic<std::uint64_t> unmatchedCompletions_{0};
    std::atomic<std::uint64_t> overflow_{0};
    std::atomic<std::uint64_t> idExhausted_{0};
    std::atomic<std::uint64_t> cancelled_{0};
    std::atomic<std::uint64_t> postShutdown_{0};
    std::atomic<std::uint64_t> postShutdownCompletions_{0};   // ★ W-2
    // ★ C5 (2026-11-10): 記録できなかった要求の総数（overflow・ID 枯渇・shutdown 相）
    std::atomic<std::uint64_t> recordMisses_{0};
    std::atomic<bool> shutdownPhase_{false};
    StageStats runtimeStage_[2]{};
    std::atomic<std::size_t> runtimeStageSlot_{0};
    // ★ W-2: final 段（違反レポート生成時点）
    StageStats finalStage_{};
    std::atomic<bool> finalStageValid_{false};
    std::atomic<std::uint64_t> observationCount_{0};
    std::atomic<std::uint64_t> ebrCount_{0};
    std::atomic<std::uint64_t> ebrSumNs_{0};
    std::atomic<std::uint64_t> ebrMaxNs_{0};
    std::atomic<std::uint64_t> execCount_{0};
    std::atomic<std::uint64_t> execSumNs_{0};
    std::atomic<std::uint64_t> execMaxNs_{0};
    std::atomic<std::uint64_t> unobservedSafe_{0};
    std::atomic<std::uint64_t> rtRequests_{0};
    // ★ 段階③: 3 分類のパス別集計（診断用・判定には使わない）
    std::atomic<std::uint64_t> driverCount_[kPathCount];
    std::atomic<std::uint64_t> driverSumNs_[kPathCount];
    std::atomic<std::uint64_t> driverMaxNs_[kPathCount];
    std::atomic<std::uint64_t> driverOver_[kPathCount];
    std::atomic<std::uint64_t> ebrCountC_[kPathCount];
    std::atomic<std::uint64_t> ebrSumNsC_[kPathCount];
    std::atomic<std::uint64_t> ebrMaxNsC_[kPathCount];
    std::atomic<std::uint64_t> ebrOverC_[kPathCount];
    std::atomic<std::uint64_t> execCountC_[kPathCount];
    std::atomic<std::uint64_t> execSumNsC_[kPathCount];
    std::atomic<std::uint64_t> execMaxNsC_[kPathCount];
    std::atomic<std::uint64_t> execOverC_[kPathCount];
    std::atomic<std::uint64_t> classUnclassifiedCount_[kPathCount];
    std::atomic<std::uint64_t> classUnclassifiedSumNs_[kPathCount];
    std::atomic<std::uint64_t> classExcluded_[kPathCount];
    std::atomic<std::uint64_t> classClassifiedTotal_{0};
    std::atomic<std::uint64_t> classUnclassifiedTotal_{0};
    std::atomic<std::uint64_t> classUnclassifiedSumTotal_{0};
    std::atomic<std::uint64_t> classExcludedTotal_{0};
    std::atomic<std::uint64_t> classConsistencyErrors_{0};
    std::atomic<std::uint64_t> pathCount_[kPathCount];
    std::atomic<std::uint64_t> pathOver_[kPathCount];
    std::atomic<std::uint64_t> pathMaxNs_[kPathCount];
};

} // namespace isr
} // namespace convo
