//============================================================================
// DeferredDeletionQueue.h
//
// ロックフリー MPMC 削除キュー
//
// B23: 外部依存関係の明示と RT 保証の静的チェック
// - 本クラスは将来的に moodycamel::ConcurrentQueue への移行を想定しています。
// - 現状はカスタム MPMC 実装を使用していますが、エントリのトリビアルコピー可能性を維持します。
//============================================================================

#pragma once

#include "audioengine/ISRRetireLatency.h"   // ★ A(D1): 退役要求→解放完了の測定
#include "core/RetireDomain.h"              // ★ domain tag (v2.9): entry への刻印

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <type_traits>

#include "audioengine/AtomicAccess.h"
#include "audioengine/DiagObjectRegistry.h"   // ★ WORK113 変更1: [RETIRE_RECLAIM] の共通ログ先
#include "audioengine/ISRWorldRetirementReference.h"   // ★ T1 (D98): reference observer（measurement only・D97）

enum class DeletionEntryType : uint8_t {
    Generic = 0,
    World   = 1   // ★ T1 (D86): world retirement の識別 metadata（telemetry 専用・lifetime authority ではない）
};

// ★ F1'-b (2026-10-25): 破棄の結果。deferred = 「解放せず、所有権を呼び出し側に残した」。
//   pin が残っている場合に destroyDSPCoreNode がこれを返す。void(void*) では表現できないため、
//   再試行可能な破棄（TerminalReclaimAuthority とエンジンの再試行リスト）で使う。
enum class DestroyOutcome : uint8_t {
    Destroyed = 0,   // 物理解放が完了した（以後 ptr は無効）
    Deferred  = 1    // pin 残留により解放しなかった（所有権は呼び出し側に残る）
};

// ★ F1'-b: 再試行可能な deleter。void(void*) の上位互換。
using RetryableDeleter = DestroyOutcome (*)(void*) noexcept;

// ★ WORK113 変更1: sized enqueue で運ぶメタデータ（DIAG 専用の帰属情報）。
//   DeletionEntry のフィールドにもサイズにも影響しない（別配列に置く）。
struct RetireMeta
{
    std::uint64_t ownerId { 0 };        // DSPCore の runtimeUuid（0 = 未取得）
    std::uint64_t measuredBytes { 0 };  // 計測済み帰属分（0 = 未計測）
    bool available { false };           // 供給されたか（false を 0 で埋めない）
};

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
// ★ WORK113 変更1: スロット単位 seqlock（tag 偶数=安定 / 奇数=書き込み中、64 bit・飽和）。
//   全フィールド atomic・全アクセス seq_cst。読み取りは最大 3 回で打ち切る。
struct DiagRetireMetaSlot
{
    std::atomic<std::uint64_t> tag { 0 };
    std::atomic<std::uint32_t> sequence { 0 };
    std::atomic<std::uint64_t> ownerId { 0 };
    std::atomic<std::uint64_t> measuredBytes { 0 };
    std::atomic<std::uint32_t> available { 0 };
};

struct DiagRetireMetaView
{
    std::uint64_t ownerId { 0 };
    std::uint64_t measuredBytes { 0 };
    bool available { false };
};
#endif

// DeletionEntry: 削除対象のエントリ
// B23: static_assert を通すため、std::atomic を含まずトリビアルコピー可能である必要があります。
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
    // ★ domain tag (v2.9 §2.1): enqueue 時に provider が刻印し、以降は変更しない。
    //   既定値 Unstamped は「刻印漏れ」を意味する（生成側は絶対に付けない）。
    convo::RetireDomainId domainId { convo::RetireDomainId::Unstamped };
    uint64_t publicationSequenceId{0};  // ★ P2-2: 出版-退役の因果追跡用
    uint64_t generation{0};             // ★ P2-2: generation 追跡
    // ★ 段階③ (2026-11-11, Owner 承認): 最初にreclaim走査で評価された時刻（ns・0 = 未走査）
    //   producer（enqueue）が 0 で初期化し、consumer（reclaim）が最初に 1 回だけ書く。
    //   seq の publish 後にのみ書かれるため、データ競合は無い。
    uint64_t firstScanTsNs{0};
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    size_t objectBytes{0};  // ★ work70: Retire 対象のオブジェクトバイト数（sizeof、診断のみ）
#endif
};

// B23: リアルタイム安全性のための静的チェック
static_assert(std::is_trivially_copyable_v<DeletionEntry>,
    "DeletionEntry must be trivially copyable for lock-free queue operations");
static_assert(std::is_trivially_destructible_v<DeletionEntry>,
    "DeletionEntry must be trivially destructible");
// ★ domain tag (v2.9 §2.1): 刻印の型と、サイズ・アラインメントを構成別に固定する。
//   変更前の実測（2026-11-11）: 診断 OFF = 56 / 8、診断 ON = 64 / 8。
//   domainId（1 バイト）は type の後ろのパディングに吸収され、サイズは変わらない。
static_assert(sizeof(convo::RetireDomainId) == 1, "convo::RetireDomainId must be 1 byte");
static_assert(alignof(DeletionEntry) == 8, "DeletionEntry alignment changed");
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
static_assert(sizeof(DeletionEntry) == 64, "DeletionEntry size changed (diagnostics ON)");
#else
static_assert(sizeof(DeletionEntry) == 56, "DeletionEntry size changed (diagnostics OFF)");
#endif
static_assert(std::atomic<size_t>::is_always_lock_free,
    "std::atomic<size_t> must be lock-free for real-time safety");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
    "std::atomic<uint64_t> must be lock-free for real-time safety");

#pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
/**
 * DeferredDeletionQueue: ロックフリー MPMC 削除キュー (B9 修正版)
 *
 * Dmitry Vyukov の bounded MPMC queue アルゴリズムを採用。
 * シーケンス番号をエントリ外に配置することで、エントリ自体のトリビアルコピー可能性を確保しています。
 */
class DeferredDeletionQueue {
public:
    DeferredDeletionQueue() noexcept {
        for (uint32_t i = 0; i < kQueueSize; ++i)
            convo::publishAtomic(sequences[i], i, std::memory_order_release); // release: enqueue/dequeue の seq acquire と HB (初期化後の最初の観測を保証)
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        // ★ WORK113 変更1: メタはヒープに置く（EpochDomain のスタック占有を増やさない）。
        //   確保失敗時は nullptr のまま＝メタ無効（0 や推定値で埋めない）。
        diagMeta_.reset(new (std::nothrow) DiagRetireMetaSlot[kQueueSize]());
#endif
    }

    // Audio Thread から呼ばれる。ロックフリー。
    // ★ domain tag (v2.9 §3.2): domain は必須引数（既定引数を設けない）。
    //   呼び出し側が必ず意図を書くため、刻印漏れはコンパイルで検出される。
    bool enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, convo::RetireDomainId domain) noexcept {
        return enqueue(ptr, deleter, epoch, DeletionEntryType::Generic, 0, 0, domain);
    }

    // Audio Thread から呼ばれる。ロックフリー。
    bool enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type,
                 convo::RetireDomainId domain) noexcept {
        return enqueue(ptr, deleter, epoch, type, 0, 0, domain);
    }

    // ★ P2-2: publicationSequenceId + generation 付き enqueue
    bool enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type,
                 uint64_t publicationSequenceId, uint64_t generation,
                 convo::RetireDomainId domain) noexcept {
        return enqueueImpl(ptr, deleter, epoch, type, publicationSequenceId, generation, domain, nullptr);
    }

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ WORK113 変更1: メタ付き enqueue（NonRT 専用・DIAG ON のみ）。
    bool enqueueSized(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type,
                      uint64_t publicationSequenceId, uint64_t generation,
                      convo::RetireDomainId domain, const RetireMeta& meta) noexcept {
        return enqueueImpl(ptr, deleter, epoch, type, publicationSequenceId, generation, domain, &meta);
    }

    // ★ WORK113 変更1: スロットのメタを読む（reclaim / head 共用・seq 照合つき）。
    [[nodiscard]] bool readMeta(std::uint32_t slot,
                                std::uint32_t expectedSequence,
                                DiagRetireMetaView& out) const noexcept {
        if (! diagMeta_)
        {
            out = DiagRetireMetaView {};
            return false;
        }

        auto& m = diagMeta_[slot & kMask];
        constexpr std::uint64_t kTagMax = std::numeric_limits<std::uint64_t>::max() - 1;

        for (int attempt = 0; attempt < 3; ++attempt) {
            const std::uint64_t s1 = convo::consumeAtomic(m.tag, std::memory_order_acquire);
            if ((s1 & 1u) != 0u || s1 >= kTagMax)      // 書き込み中、または飽和
                continue;

            const std::uint32_t seq = convo::consumeAtomic(m.sequence, std::memory_order_acquire);
            const std::uint64_t owner = convo::consumeAtomic(m.ownerId, std::memory_order_acquire);
            const std::uint64_t bytes = convo::consumeAtomic(m.measuredBytes, std::memory_order_acquire);
            const std::uint32_t avail = convo::consumeAtomic(m.available, std::memory_order_acquire);
            const std::uint64_t s2 = convo::consumeAtomic(m.tag, std::memory_order_acquire);

            if (s1 == s2 && seq == expectedSequence) {
                out.ownerId = owner;
                out.measuredBytes = bytes;
                out.available = (avail != 0u);
                return true;
            }
        }

        metaReadRetry_.fetch_add(1, std::memory_order_relaxed);
        out = DiagRetireMetaView{};
        return false;
    }

    [[nodiscard]] std::uint64_t metaReadRetryCount() const noexcept {
        return convo::consumeAtomic(metaReadRetry_, std::memory_order_acquire);
    }
#endif

private:
    bool enqueueImpl(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type,
                     uint64_t publicationSequenceId, uint64_t generation,
                     convo::RetireDomainId domain, const RetireMeta* meta) noexcept {
        uint32_t pos = convo::consumeAtomic(enqueuePos, std::memory_order_acquire); // acquire: 前回 enqueue の CAS acq_rel と HB し最新の enqueuePos を観測
        while (true) {
            auto& seq_atom = sequences[pos & kMask];
            uint32_t seq = convo::consumeAtomic(seq_atom, std::memory_order_acquire); // acquire: dequeue の seq release と HB しスロット解放を観測
            // kQueueSize(=4096) ≪ INT32_MAX により int32_t 減算で安全。
            int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));

            if (diff == 0) {
                if (convo::compareExchangeAtomic(enqueuePos,
                                                 pos,
                                                 static_cast<uint32_t>(pos + 1),
                                                 std::memory_order_acq_rel,  // 成功時 acq_rel: acquire で dequeue release と HB; release で次回 enqueue acquire と HB
                                                 std::memory_order_acquire)) { // 失敗時 acquire: 最新 enqueuePos を観測して再試行
                    auto& entry = ringBuffer[pos & kMask];
                    entry.ptr = ptr;
                    entry.deleter = deleter;
                    entry.epoch = epoch;
                    entry.type = type;
                    entry.domainId = domain;   // ★ domain tag: enqueue 時に刻印（以降不変）
                    entry.publicationSequenceId = publicationSequenceId;
                    entry.generation = generation;
                    entry.firstScanTsNs = 0;   // ★ 段階③: スロット再利用時の残留を防ぐ
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
                    // ★ WORK113 変更1: メタは seq の publish より前に書く（読み手は seq の acquire で観測）。
                    if (meta != nullptr)
                        writeMeta(pos & kMask, pos + 1, *meta);
#endif
                    convo::publishAtomic(seq_atom, pos + 1, std::memory_order_release); // release: dequeue の seq acquire と HB しエントリ書き込み完了を公知
                    return true;
                }
            } else if (diff < 0) {
                return false; // Full
            } else {
                pos = convo::consumeAtomic(enqueuePos, std::memory_order_acquire); // acquire: CAS 失敗後 retry — 最新 enqueuePos を再観測
            }
        }
    }

public:

    // Message Thread / Timer から呼ばれる。
    uint32_t reclaim(uint64_t minReaderEpoch) {
        // ★ kMaxScan / scanned は現在の実装ではループの上限として機能していない
        //   （先頭が reclaim 不可の場合は即 break するため）。
        //   将来 FIFO 逆転への先読みスキャンを再実装する場合に再利用可能。
        constexpr int kMaxScan = 1024;
        uint32_t deqPos = convo::consumeAtomic(dequeuePos, std::memory_order_acquire); // acquire: 前回 dequeue の CAS release と HB し最新の dequeuePos を観測
        uint32_t scanPos = deqPos;
        int scanned = 0;
        uint32_t reclaimed = 0;  // ★ A-1: 解放件数カウンタ

        while (scanned < kMaxScan) {
            auto& seq_atom = sequences[scanPos & kMask];
            const uint32_t seq = convo::consumeAtomic(seq_atom, std::memory_order_acquire); // acquire: enqueue の seq release と HB しエントリ書き込み完了を観測
            const int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - (scanPos + 1)));

            if (diff != 0) {
                break; // Empty
            }

            auto& entry = ringBuffer[scanPos & kMask];
            // ★ 段階③: この entry が最初にreclaim走査で評価された時刻を一度だけ記録する。
            //   consumer（この関数）だけが書くため CAS は不要（seq の acquire 後）。
            if (entry.firstScanTsNs == 0)
                entry.firstScanTsNs = steadyNowNs();
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            // ★ WORK113 変更1: メタは CAS 前に複写する（CAS 成功後はスロットが再利用され得る）。
            DiagRetireMetaView meta {};
            const bool metaOk = readMeta(scanPos & kMask, seq, meta);
            const std::uint64_t scanTsNs = entry.firstScanTsNs;
#endif
            bool canDelete = false;

            if (isOlder(entry.epoch, minReaderEpoch)) {
                canDelete = true;
            }

            // FIFO を維持するため、現在の dequeue 先頭と一致した時だけ削除する。
            if (canDelete && scanPos == deqPos) {
                if (convo::compareExchangeAtomic(dequeuePos,
                                                 deqPos,
                                                 static_cast<uint32_t>(deqPos + 1),
                                                 std::memory_order_release,  // 成功時 release: 次回 dequeue/drainAllUnsafe の acquire と HB しスロット解放を公知
                                                 std::memory_order_acquire)) { // 失敗時 acquire: 最新 dequeuePos を観測して再試行
                    if (entry.deleter && entry.ptr) {
                        const auto entryType = entry.type;   // deleter 実行後に entry がクリアされるため事前保持
                        convo::isr::RetireLatencyRecorder::instance().recordSafe (entry.ptr);   // ★ A(D1): epoch 安全到達
                        convo::isr::RetireLatencyRecorder::instance().recordComplete (
                            entry.ptr, convo::isr::RetireLatencyRecorder::Path::Queue,
                            entry.domainId,   // ★ domain tag: entry の刻印をそのまま記録
                            entry.firstScanTsNs);   // ★ A(D1) + 段階③: 初回走査時刻
                        entry.deleter(entry.ptr);
                        // ★ T1 (D86): type==World の terminal deleter 実行後 → world 破棄観測（release observation・案 B）。
                        //   telemetry 識別 metadata のみ・lifetime authority にしない（D86 非交渉条件 2）。
                        if (entryType == DeletionEntryType::World)
                        {
                            convo::fetchAddAtomic(worldReclaimCount_, std::uint64_t{1}, std::memory_order_acq_rel);
                            // ★ T1 (D98): reference observer に release event を通知（event-driven・D95 固定点 4・
                            //   terminalization 成功後・例外/所有権変更/reclaim 再試行なし・D97）。
                            if (referenceObserver_ != nullptr)
                                referenceObserver_->onRelease();
                        }
                    }
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
                    {
                        // ★ WORK113 変更1: 解放の帰属を 1 行で残す（メタが無ければ na と明示）。
                        char line[360];
                        const double ageMs = (scanTsNs != 0)
                            ? static_cast<double>(steadyNowNs() - scanTsNs) / 1.0e6
                            : 0.0;
                        if (metaOk && meta.available)
                            std::snprintf(line, sizeof(line),
                                "[RETIRE_RECLAIM] slot=%u seq=%u epoch=%llu domain=%u type=%u gen=%llu "
                                "owner=%llu bytes=%llu ageMs=%.3f result=destroyed",
                                static_cast<unsigned>(scanPos & kMask),
                                static_cast<unsigned>(seq),
                                static_cast<unsigned long long>(entry.epoch),
                                static_cast<unsigned>(entry.domainId),
                                static_cast<unsigned>(entry.type),
                                static_cast<unsigned long long>(entry.generation),
                                static_cast<unsigned long long>(meta.ownerId),
                                static_cast<unsigned long long>(meta.measuredBytes),
                                ageMs);
                        else
                            std::snprintf(line, sizeof(line),
                                "[RETIRE_RECLAIM] slot=%u seq=%u epoch=%llu domain=%u type=%u gen=%llu "
                                "owner=na bytes=na ageMs=%.3f result=destroyed",
                                static_cast<unsigned>(scanPos & kMask),
                                static_cast<unsigned>(seq),
                                static_cast<unsigned long long>(entry.epoch),
                                static_cast<unsigned>(entry.domainId),
                                static_cast<unsigned>(entry.type),
                                static_cast<unsigned long long>(entry.generation),
                                ageMs);
                        convo::diag::registry().emitLine(line);
                    }
#endif
                    ++reclaimed;  // ★ A-1: 実際に解放した件数をカウント
                    entry.ptr = nullptr;
                    entry.deleter = nullptr;
                    entry.type = DeletionEntryType::Generic;
                    convo::publishAtomic(seq_atom, scanPos + kQueueSize, std::memory_order_release); // release: enqueue の seq acquire と HB しスロット再利用可能を公知

                    ++deqPos;        // [BUG-03] dequeuePos の新値 (deqPos+1) に追従
                    scanPos = deqPos;
                    scanned = 0;
                } else {
                    deqPos = convo::consumeAtomic(dequeuePos, std::memory_order_acquire); // acquire: CAS 失敗後の再観測 — 最新 dequeuePos を取得
                    scanPos = deqPos;
                    scanned = 0;
                }
            } else {
                // ★ 先頭エントリが削除不可 → FIFO順序のため即座に脱出
                //   後続エントリが reclaimable でも FIFO を破って削除することはできない。
                break;
            }
        }
        return reclaimed;  // ★ A-1: 解放件数を返す
    }

    // Shutdown 専用: epoch 判定を無視して全エントリを回収する。
    void drainAllUnsafe() {
        uint32_t pos = convo::consumeAtomic(dequeuePos, std::memory_order_acquire); // acquire: 前回 dequeue の CAS release と HB し最新 dequeuePos を観測
        while (true) {
            auto& seq_atom = sequences[pos & kMask];
            uint32_t seq = convo::consumeAtomic(seq_atom, std::memory_order_acquire); // acquire: enqueue の seq release と HB しエントリ書き込み完了を確認
            int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - (pos + 1)));

            if (diff == 0) {
                auto& entry = ringBuffer[pos & kMask];
                if (convo::compareExchangeAtomic(dequeuePos,
                                                 pos,
                                                 static_cast<uint32_t>(pos + 1),
                                                 std::memory_order_acq_rel,  // 成功時 acq_rel: acquire で enqueue acq_rel と HB; release で次回ドレインの acquire と HB
                                                 std::memory_order_acquire)) { // 失敗時 acquire: 最新 dequeuePos を観測して retry
                    if (entry.deleter && entry.ptr) {
                        const auto entryType = entry.type;   // deleter 実行後に entry がクリアされるため事前保持
                        convo::isr::RetireLatencyRecorder::instance().recordComplete (
                            entry.ptr, convo::isr::RetireLatencyRecorder::Path::DrainAll,
                            entry.domainId,   // ★ domain tag: entry の刻印をそのまま記録
                            entry.firstScanTsNs);   // ★ A(D1)
                        entry.deleter(entry.ptr);
                        // ★ T1 (D86): type==World の terminal deleter 実行後 → world 破棄観測（shutdown drain 含む）。
                        if (entryType == DeletionEntryType::World)
                        {
                            convo::fetchAddAtomic(worldReclaimCount_, std::uint64_t{1}, std::memory_order_acq_rel);
                            // ★ T1 (D98): reference observer に release event を通知（shutdown drain 含む・D95 固定点 3）。
                            if (referenceObserver_ != nullptr)
                                referenceObserver_->onRelease();
                        }
                    }
                    entry.ptr = nullptr;
                    entry.deleter = nullptr;
                    entry.type = DeletionEntryType::Generic;
                    convo::publishAtomic(seq_atom, pos + kQueueSize, std::memory_order_release); // release: 次の enqueue の seq acquire と HB しスロット再利用可能を公知
                    pos++;
                }
            } else {
                break; // Empty
            }
        }
    }

    [[nodiscard]] uint32_t sizeApprox() const noexcept
    {
        const uint32_t enq = convo::consumeAtomic(enqueuePos, std::memory_order_acquire);
        const uint32_t deq = convo::consumeAtomic(dequeuePos, std::memory_order_acquire);
        return static_cast<uint32_t>(enq - deq);
    }

    // ★ AC_DIAG (2026-11-13, Owner 承認 A/C): 先頭 entry の読み取り専用スナップショット。
    //   キューを変更しない（dequeue しない・CAS しない・ロックしない）。POD のコピーのみ。
    //
    //   整合性: Vyukov の bounded MPMC では「slot の seq == pos + 1」が
    //   「producer が書き込みを終えて publish 済み」を意味する（reclaim() の Empty 判定と同じ式）。
    //   コピー中に slot が解放→再利用されると seq が変わるため、コピー後に seq と dequeuePos を
    //   読み直し、変化していれば false（＝推定しない）を返す。ABA は 4096 回の周回が必要で、
    //   shutdown 時（producer join 済み・consumer は 1 本）には成立しない。
    //
    //   ★ C3-D (2026-11-13, Owner 承認): head identity の 3 要素を out 引数で返す。
    //     identity = (slotIndex, sequence, epoch)。slotIndex と sequence は「同じ物理 entry か」の
    //     判定に使う（epoch だけでは同一性を保証できない）。**失敗時は out 引数を一切書かない**
    //     （推定値を返さない契約）。epoch/type/domain で identity を代用してはならない。
    [[nodiscard]] bool peekHead(DeletionEntry& out,
                                uint32_t& slotIndex,
                                uint32_t& sequence) const noexcept
    {
        const uint32_t pos = convo::consumeAtomic(dequeuePos, std::memory_order_acquire);
        const uint32_t idx = pos & kMask;
        const uint32_t seqBefore = convo::consumeAtomic(sequences[idx], std::memory_order_acquire);
        if (static_cast<int32_t>(static_cast<uint32_t>(seqBefore - (pos + 1))) != 0)
            return false;                       // 空、または未 publish

        const DeletionEntry snapshot = ringBuffer[idx];   // POD コピー（ポインタは辿らない）

        const uint32_t seqAfter = convo::consumeAtomic(sequences[idx], std::memory_order_acquire);
        const uint32_t posAfter = convo::consumeAtomic(dequeuePos, std::memory_order_acquire);
        if (seqAfter != seqBefore || posAfter != pos)
            return false;                       // コピー中に上書き・破棄された → 値を出さない

        out = snapshot;
        slotIndex = idx;
        sequence = seqBefore;
        return true;
    }

    // ★ P1-7: 最大滞留時間 (us) 追跡
    [[nodiscard]] uint64_t getMaxRetireAgeUs() const noexcept {
        return convo::consumeAtomic(maxRetireAgeUs_, std::memory_order_acquire);
    }

    void updateMaxRetireAge(uint64_t ageUs) noexcept {
        uint64_t current = convo::consumeAtomic(maxRetireAgeUs_, std::memory_order_acquire);
        while (ageUs > current) {
            if (convo::compareExchangeAtomic(maxRetireAgeUs_, current, ageUs,
                    std::memory_order_acq_rel, std::memory_order_acquire))
                break;
        }
    }

    void clearMaxRetireAge() noexcept {
        convo::publishAtomic(maxRetireAgeUs_, static_cast<uint64_t>(0), std::memory_order_release);
    }

    // ★ T1 (D86): type==World の terminal deleter 実行数（world 物理破棄数・累積・monotonic）。
    //   release observation（案 B）の一次情報源。sampler（Non-RT）が読み取る（D83.2 責務分離）。
    [[nodiscard]] uint64_t worldReclaimCount() const noexcept {
        return convo::consumeAtomic(worldReclaimCount_, std::memory_order_acquire);
    }

    // ★ T1 (D98): reference observer 設定（non-owning・一方向依存・AudioEngine を逆参照しない）。
    void setReferenceObserver(convo::isr::WorldRetirementReferenceObserver* observer) noexcept
    {
        referenceObserver_ = observer;
    }

    // ★ WORK113 Queue Lifetime: 隔離（自己参照で寿命を保つ・移送時の追加確保なし）。
    //   quarantineQueue() だけが書き、公開後は next を変更しない（append-only）。
    std::shared_ptr<DeferredDeletionQueue> quarantineSelfRef_;
    std::atomic<bool> quarantineEnqueued_ { false };
    std::atomic<DeferredDeletionQueue*> quarantineNext_ { nullptr };

private:
    static inline bool isOlder(uint64_t a, uint64_t b) noexcept
    {
        return static_cast<int64_t>(a - b) < 0;
    }

    // ★ 段階③: RetireLatencyRecorder::nowNs() と同じ時計（steady_clock）。
    //   firstScanTs と enqueueTs / safeTs を同一基準で比較するため必須。
    static inline uint64_t steadyNowNs() noexcept
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ WORK113 変更1: メタの書き込み（単一 writer・seq_cst・64 bit tag）。
    //   tag が飽和域に入ったら奇数固定にして、以後の読み取りを unavailable にする。
    void writeMeta(std::uint32_t slot, std::uint32_t sequence, const RetireMeta& meta) noexcept
    {
        if (! diagMeta_)
            return;   // 確保失敗時はメタ無効（読む側も unavailable を返す）

        auto& m = diagMeta_[slot & kMask];
        constexpr std::uint64_t kTagMax = std::numeric_limits<std::uint64_t>::max() - 1;
        const std::uint64_t t = convo::consumeAtomic(m.tag, std::memory_order_acquire);

        if (t >= kTagMax)
        {
            convo::publishAtomic(m.available, 0u, std::memory_order_release);
            convo::publishAtomic(m.tag, std::numeric_limits<std::uint64_t>::max(), std::memory_order_release);
            return;
        }

        convo::publishAtomic(m.tag, t + 1, std::memory_order_release);          // 奇数: 書き込み中
        convo::publishAtomic(m.sequence, sequence, std::memory_order_release);
        convo::publishAtomic(m.ownerId, meta.ownerId, std::memory_order_release);
        convo::publishAtomic(m.measuredBytes, meta.measuredBytes, std::memory_order_release);
        convo::publishAtomic(m.available, meta.available ? 1u : 0u, std::memory_order_release);
        convo::publishAtomic(m.tag, t + 2, std::memory_order_release);          // 偶数: 安定
    }
#endif

    static constexpr uint32_t kQueueSize = 4096;
    static constexpr uint32_t kMask = kQueueSize - 1;

    alignas(64) std::array<DeletionEntry, kQueueSize> ringBuffer;
    alignas(64) std::array<std::atomic<uint32_t>, kQueueSize> sequences;
    alignas(64) std::atomic<uint32_t> enqueuePos{0};
    alignas(64) std::atomic<uint32_t> dequeuePos{0};
    alignas(64) std::atomic<uint64_t> maxRetireAgeUs_{0};
    alignas(64) std::atomic<uint64_t> worldReclaimCount_{0};  // ★ T1: world 破棄観測カウンタ（telemetry のみ）
    convo::isr::WorldRetirementReferenceObserver* referenceObserver_ = nullptr;  // ★ T1 (D98): non-owning（measurement only）
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ WORK113 変更1: サイドテーブル（4096 × 40 B ≒ 160 KB・DIAG ON のみ）。
    //   EpochDomain のスタック占有を増やさないためヒープに置く（構築は NonRT）。
    //   DeletionEntry のサイズ契約（64 / 56）には影響しない。
    std::unique_ptr<DiagRetireMetaSlot[]> diagMeta_;
    // 試行回数は診断専用であり、const の readMeta() から更新するため static にする
    // （mutable は lint AE-011 で禁止。インスタンス横連の合計値でも診断には十分）。
    static inline std::atomic<std::uint64_t> metaReadRetry_ { 0 };
#endif
};
#pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
