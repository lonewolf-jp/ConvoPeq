#pragma once

#include <cstdint>
#include <atomic>
#include <cassert>
#include <functional>
#include "TelemetryRecorder.h"   // FixedRingBuffer
#include "RuntimePublicationState.h"   // CorrelationId
#include "core/TimeUtils.h"  // getCurrentTimeUs
#include "AtomicAccess.h"  // ★ work37: fetchAddAtomic/consumeAtomic

namespace convo::isr {

// ★ work37: HealthEvent callback for PolicyEngine integration
using WorldHealthEventCallback = std::function<void(uint32_t eventCode, uint64_t value)>;

// ★ P3-B: World 発行/退役の診断用レコード
struct WorldLifecycleRecord {
    uint64_t worldId;
    uint64_t publishEpoch;
    uint64_t retireEpoch;        // 0 = 未退役
    uint64_t publishTimestampUs;
    uint64_t retireTimestampUs;
    CorrelationId correlationId;
};

// ★ P3-B: World ライフサイクル監査（Diagnostic 限定）
//   Shutdown 完了判定の Authority にはしない。
//   Shutdown 判定は RuntimeDrainAudit + ShutdownRuntime FSM が担当。
class WorldLifecycleAudit {
public:
    void onWorldPublished(uint64_t worldId, uint64_t epoch, CorrelationId cid) noexcept
    {
        ringBuffer_.tryPush(WorldLifecycleRecord{
            .worldId = worldId,
            .publishEpoch = epoch,
            .retireEpoch = 0,
            .publishTimestampUs = getCurrentTimeUs(),
            .retireTimestampUs = 0,
            .correlationId = cid
        });
        convo::fetchAddAtomic(publishedCount_, uint64_t{1}, std::memory_order_release);
        convo::fetchAddAtomic(activeWorldCount_, uint64_t{1}, std::memory_order_release);
    }

    void onWorldRetired(uint64_t worldId, uint64_t epoch) noexcept
    {
        // ★ v7.2: fetchSub→if(prev==0)→publishAtomic(0)
        //   load→if→fetchSub 方式は TOCTOU 競合があるため不採用。
        //   Diagnostic 限定カウンタだが監査価値を維持するため fetchSub の戻り値で判定。
        uint64_t prev = convo::fetchSubAtomic(activeWorldCount_, 1u,
            std::memory_order_acq_rel);
        if (prev == 0) {
            assert(false);  // 二重 retire 検出
            // ★ A-5: Release ビルドでも telemetry カウンタをインクリメント
            convo::fetchAddAtomic(doubleRetireCount_, 1u, std::memory_order_release);
            // アンダーフロー補正（既存）
            convo::publishAtomic(activeWorldCount_, uint64_t{0},
                std::memory_order_release);
        }

        convo::fetchAddAtomic(retiredCount_, 1u, std::memory_order_release);

        // ★ 直近の retire 情報を別途追跡（リングバッファは追記専用のため更新不可）
        convo::publishAtomic(lastRetiredWorldId_, worldId, std::memory_order_release);
        convo::publishAtomic(lastRetireEpoch_, epoch, std::memory_order_release);
        convo::publishAtomic(lastRetireTimestampUs_, getCurrentTimeUs(), std::memory_order_release);
    }

    [[nodiscard]] uint64_t activeWorldCount() const noexcept {
        return convo::consumeAtomic(activeWorldCount_, std::memory_order_acquire);
    }

    [[nodiscard]] uint64_t publishedCount() const noexcept {
        return convo::consumeAtomic(publishedCount_, std::memory_order_acquire);
    }

    [[nodiscard]] uint64_t retiredCount() const noexcept {
        return convo::consumeAtomic(retiredCount_, std::memory_order_acquire);
    }

    // ★ A-5: 二重 retire 検出カウンタ（telemetry 用）
    [[nodiscard]] uint64_t doubleRetireCount() const noexcept {
        return convo::consumeAtomic(doubleRetireCount_, std::memory_order_acquire);
    }

    // ★ 診断用ダンプ（RingBuffer から最新レコードを取得）
    void emitSnapshot() const noexcept;

    // ★ 診断用: 定期ダンプを AudioEngine の emitEvidenceTickNonRt から呼び出すためのヘルパー
    //   出力先: evidence/world_lifecycle_audit.json
    void tryDumpPeriodic() noexcept;

    // [work37 Phase 4.2] HealthEvent callback — PolicyEngine 連携用
    void setHealthEventCallback(WorldHealthEventCallback cb) noexcept {
        m_healthCallback_ = std::move(cb);
    }

    // [work37 Phase 4.2] FallbackQueue overflow 検出 → HealthEvent 発火
    void onFallbackOverflow() noexcept;

    // [work37 Phase 4.2] World 整合性異常検出 → HealthEvent 発火
    void onWorldLeakDetected(uint64_t retiredCount, uint64_t publishedCount) noexcept;

    [[nodiscard]] uint64_t fallbackOverflowCount() const noexcept {
        return convo::consumeAtomic(fallbackOverflowCount_, std::memory_order_acquire);
    }

    [[nodiscard]] uint64_t worldLeakCount() const noexcept {
        return convo::consumeAtomic(worldLeakCount_, std::memory_order_acquire);
    }

private:
    FixedRingBuffer<WorldLifecycleRecord, 4096> ringBuffer_;
    std::atomic<uint64_t> activeWorldCount_{0};
    std::atomic<uint64_t> publishedCount_{0};
    std::atomic<uint64_t> retiredCount_{0};
    std::atomic<uint64_t> lastDumpTimeUs_{0};
    static constexpr uint64_t kDumpIntervalUs = 60'000'000; // 60秒ごとにダンプ
    // ★ A-5: 二重 retire 検出カウンタ
    std::atomic<uint64_t> doubleRetireCount_{0};
    // ★ 直近 retire 追跡（リングバッファは追記専用のため retire 更新不可）
    std::atomic<uint64_t> lastRetiredWorldId_{0};
    std::atomic<uint64_t> lastRetireEpoch_{0};
    std::atomic<uint64_t> lastRetireTimestampUs_{0};
    // [work37 Phase 4.2] HealthEvent callback + Fallback/WorldLeak 検出
    WorldHealthEventCallback m_healthCallback_;
    std::atomic<uint64_t> fallbackOverflowCount_{0};
    std::atomic<uint64_t> worldLeakCount_{0};
};

} // namespace convo::isr
