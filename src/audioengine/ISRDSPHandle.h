#pragma once

#include <atomic>
#include <cstdint>
#include <array>
#include <vector>
#include <filesystem>

namespace convo {
namespace isr {

/**
 * ISR 10層 Architecture Layer 2: DSPHandle Runtime
 * DSP ownership の source-of-truth を管理し、lifetime ambiguity を排除する。
 */

/**
 * DSP スロット + 世代による handle（ABA 防止）
 */
struct DSPHandle
{
    uint32_t slot;        // レジストリスロット番号
    uint64_t generation;  // ★ B-1: 64bit化（世代番号）

    bool isNull() const noexcept
    {
        return slot == 0 && generation == 0;
    }

    static DSPHandle null() noexcept
    {
        return { 0, 0 };
    }

    bool operator==(const DSPHandle& other) const noexcept
    {
        return slot == other.slot && generation == other.generation;
    }

    bool operator!=(const DSPHandle& other) const noexcept
    {
        return !(*this == other);
    }
};

/**
 * DSP 生命周期状態
 */
enum class DSPState
{
    Constructing,    // create 呼び出し後、Active 前
    Active,          // 通常使用中
    CrossfadingIn,   // crossfade 中（新 DSP 側）
    CrossfadingOut,  // crossfade 中（旧 DSP 側）
    Retired,         // retire 完了、grace period 中
    Quarantined,     // 問題検出によりアクセス禁止
    DestroyPending,  // ★ A-1.4: shutdown時の解放予約状態（TOCTOU防止）
    Reclaimed        // メモリ解放済み
};

/**
 * DSP resolve 結果
 */
struct ResolvedDSP
{
    void* instance;  // DSP インスタンスポインタ（nullptr if invalid）
    bool valid;      // handle 検証結果
    bool isStale;    // generation mismatch の場合 true
};

/**
 * crossfade ID（複数 crossfade の同時追跡用）
 */
using CrossfadeId = uint32_t;

/**
 * crossfade 記録
 */
struct CrossfadeRecord
{
    CrossfadeId id;
    DSPHandle   fromHandle;
    DSPHandle   toHandle;
    uint64_t    startEpoch;
    bool        active;
};

/**
 * レジストリスロット内部構造
 */
struct DSPRegistrySlot
{
    std::atomic<uint64_t> generation;  // ★ B-1: 64bit化（ABA 防止世代番号）
    static_assert(std::atomic<uint64_t>::is_always_lock_free,
        "atomic<uint64_t> must be lock-free on x64 for ISR Runtime");
    void*                 instance;    // DSP インスタンスポインタ
    std::atomic<DSPState> state;       // 現在状態（atomic access）
};

/**
 * DSP ハンドル runtime
 * 全 DSP reference の source-of-truth を管理
 */
class DSPHandleRuntime
{
public:
    static constexpr size_t MAX_DSP_SLOTS = 256;

    DSPHandleRuntime();
    ~DSPHandleRuntime();

    // NonRT: DSP インスタンスを登録し DSPHandle を返す
    DSPHandle create(void* dspInstance);

    // RT/NonRT: handle を検証し、有効な参照を返す
    // stale handle（generation mismatch）は build別ポリシーで処理
    ResolvedDSP resolve(DSPHandle handle) const noexcept;

    // NonRT: crossfade 開始（from と to の state を更新）
    CrossfadeId beginCrossfade(DSPHandle from, DSPHandle to);

    // NonRT: crossfade を使わず handle を Active に昇格
    void activate(DSPHandle handle);

    // NonRT: crossfade 終了（from を Retired に遷移）
    void endCrossfade(CrossfadeId id);

    // NonRT: DSP を Retired に遷移（grace period 開始）
    void retire(DSPHandle handle);

    // NonRT: grace period 完了後のメモリ解放
    void reclaim(DSPHandle handle);

    // NonRT: 問題検出時に DSP を Quarantined に遷移
    void quarantine(DSPHandle handle);

    // ★ A-1.3: Slot 直接 quarantine — generation 一致を要求しない
    void quarantineSlot(uint32_t slot) noexcept;

    // ★ A-1.5: slot が crossfade に関与しているか確認
    bool isSlotInCrossfade(uint32_t slot) const noexcept;

    // ★ A-1.4: shutdown専用解放（2段階: DestroyPending → Reclaimed）
    void destroyQuarantineSlot(uint32_t slot, uint64_t expectedGeneration) noexcept;

    // NonRT: 現在の active runtime DSP handle を取得
    DSPHandle getActiveRuntimeDSPHandle() const noexcept;

    // NonRT: 現在の fading runtime DSP handle を取得（crossfade 中のみ有効）
    DSPHandle getFadingRuntimeDSPHandle() const noexcept;

    // スロット状態ダンプ（デバッグ・CI用）
    void emitOwnershipTrace(const std::filesystem::path& outputPath) const;

private:
    std::array<DSPRegistrySlot, MAX_DSP_SLOTS> registry_{};
    std::atomic<DSPHandle> activeRuntimeDSPHandle_{ DSPHandle::null() };
    std::atomic<DSPHandle> fadingRuntimeDSPHandle_{ DSPHandle::null() };

    std::vector<CrossfadeRecord> crossfadeRecords_;
    std::atomic<CrossfadeId> nextCrossfadeId_{1};

    DSPState getSlotState(uint32_t slot) const noexcept;
    void setSlotState(uint32_t slot, DSPState newState) noexcept;
};

/**
 * crossfade 期間中の authority 管理
 */
class CrossfadeAuthorityRuntime
{
public:
    CrossfadeAuthorityRuntime();
    ~CrossfadeAuthorityRuntime();

    // crossfade 登録
    CrossfadeId registerCrossfade(DSPHandle from, DSPHandle to);

    // crossfade 終了
    void unregisterCrossfade(CrossfadeId id);

    // 現在アクティブな crossfade 一覧を取得
    std::vector<CrossfadeRecord> getActiveCrossfades() const noexcept;

    // 特定の handle に関連する crossfade があるか確認
    bool hasCrossfadeInvolving(DSPHandle handle) const noexcept;

private:
    std::vector<CrossfadeRecord> records_;
    std::atomic<CrossfadeId> nextId_{1};
};

}  // namespace isr
}  // namespace convo
