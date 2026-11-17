#pragma once

#include <cstdint>

#include "../DeferredDeletionQueue.h"   // ★ T1: DeletionEntryType（telemetry 識別 metadata・D86）
#include "RetireDomain.h"               // ★ domain tag (v2.9): retireDomainId() の戻り型

//==============================================================================
// IRetireProvider.h — Retire operations abstract interface.
//
// [work21 Phase-D] Separated from IEpochProvider so that retire-aware
// consumers (SnapshotCoordinator, ISRRuntimePublicationCoordinator) depend
// on a minimal retire interface, while reader-only consumers use
// IReaderEpochProvider.
//
// This interface LIVES in src/core/ and is IMPLEMENTED by ISRRetireRouter
// (src/audioengine/), avoiding a core→audioengine dependency direction.
//==============================================================================

namespace convo {

class IRetireProvider
{
public:
    virtual ~IRetireProvider() = default;

    // ── Retire operations ──

    /** Enqueue a retire request. Returns true on success. */
    virtual bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept = 0;

    // ★ WORK113: RT 専用 enqueue（生ポインタ経路・mutex も shared_ptr も触らない）。
    //   既定は false: 未対応の provider は RT 経路で明示的に失敗する（NonRT への誤降格を防ぐ）。
    virtual bool enqueueRetireRt(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept
    {
        (void) ptr; (void) deleter; (void) epoch;
        return false;
    }

    // ★ T1 (D86): telemetry type tag 付き enqueue（type==World で world retirement を識別）。
    //   デフォルト実装は type を無視して既存 3 引数版を呼ぶ（既存実装クラスは変更不要）。
    //   EpochDomain がオーバーライドして DeferredDeletionQueue に type を伝搬する。
    //   DeletionEntryType::World は telemetry 識別 metadata のみ（D86 非交渉条件 1・lifetime authority にしない）。
    virtual bool enqueueRetireTyped(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                    DeletionEntryType /*type*/) noexcept
    {
        return enqueueRetire(ptr, deleter, epoch);
    }

    // ★ WORK113 変更1: 帰属メタ付き enqueue（NonRT 専用）。
    //   既定実装は typed へ委譲し、メタを落としたことを計数する（黙って消さない）。
    //   メタに対応する provider は supportsRetireMeta() を true にし、この関数を override する。
    virtual bool enqueueRetireSized(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                   DeletionEntryType type, const RetireMeta& /*meta*/) noexcept
    {
        metaDroppedCount_.fetch_add(1, std::memory_order_relaxed);
        return enqueueRetireTyped(ptr, deleter, epoch, type);
    }

    [[nodiscard]] virtual bool supportsRetireMeta() const noexcept { return false; }

    [[nodiscard]] uint64_t metaDroppedCount() const noexcept
    {
        return metaDroppedCount_.load(std::memory_order_acquire);
    }

private:
    // ★ WORK113 変更1: メタ非対応 provider が落とした件数（既定実装だけが増やす）。
    std::atomic<std::uint64_t> metaDroppedCount_ { 0 };

public:

    /** Try to reclaim retired objects. */
    virtual void tryReclaim() noexcept = 0;

    // ★ domain tag (v2.9 §3.1): 退役ドメインの問い合わせ。
    //   EpochDomain は自分のドメインを返し、ISRRetireRouter は provider_ へ委譲する。
    //   既定は Unknown（provider が未対応の場合に「特定できない」と明示する）。
    //   Unstamped は刻印漏れ専用なので、ここで返してはいけない。
    [[nodiscard]] virtual RetireDomainId retireDomainId() const noexcept
    {
        return RetireDomainId::Unknown;
    }

    // ★ P0-A: EpochDomain 固有メソッドをインターフェースに昇格
    //   ISRRetireRouter が dynamic_cast でアクセスするより Interface で宣言する方が型安全。
    /** Return approximate count of pending retire entries. */
    virtual uint32_t pendingRetireCount() const noexcept = 0;

    /** Drain all pending retire entries (unsafe; shutdown only). */
    virtual void drainAll() noexcept = 0;

    // ★ work70: 退役キュー滞留バイト数（診断用概算）。既定値 0。
    //   pendingRetireCount() だけでは「100個=100KB か 1GB か」が不明。
    [[nodiscard]] virtual uint64_t pendingRetireBytes() const noexcept { return 0; }

    // ★ T1 (D86): type==World の terminal deleter 実行数（world 物理破棄数・release observation の一次情報源）。
    //   デフォルト実装 0（EpochDomain がオーバーライド）。
    [[nodiscard]] virtual uint64_t worldReclaimCount() const noexcept { return 0; }
};

} // namespace convo
