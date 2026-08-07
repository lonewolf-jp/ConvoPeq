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
 *
 * alignas(16): 16バイト構造体のため atomic<DSPHandle> が CMPXCHG16B を
 * 使用するには 16バイトアライメントが必要。MSVC では 8アラインのまま
 * だと is_lock_free() が false になり Debug ビルドの assert が失敗する。
 * （ISRDSPHandle.cpp:12-20 / ISRDSPHandle.h:174-177 参照）
 */
struct alignas(16) DSPHandle
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
#pragma warning(push)
#pragma warning(disable : 4324) // C4324を抑制 - struct padding due to alignment
struct CrossfadeRecord
{
    CrossfadeId id;
    DSPHandle   fromHandle;
    DSPHandle   toHandle;
    uint64_t    startEpoch;
    bool        active;
};
#pragma warning(pop)

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
    // id は CrossfadeAuthorityRuntime::registerCrossfade() から注入
    void beginCrossfade(DSPHandle from, DSPHandle to, CrossfadeId id);

    // NonRT: crossfade を使わず handle を Active に昇格
    void activate(DSPHandle handle);

    // NonRT: crossfade 終了（from を Retired に遷移）
    void endCrossfade(CrossfadeId id);

    // NonRT: DSP を Retired に遷移（grace period 開始）
    void retire(DSPHandle handle);

    // NonRT: 問題検出時に DSP を Quarantined に遷移
    void quarantine(DSPHandle handle);

    // ★ work70: 登録のロールバック（Constructing → Reclaimed）
    //   Only Constructing may be rolled back. Future intermediate states require redesign.
    [[nodiscard]] bool rollbackRegistration(DSPHandle handle) noexcept;

    // ★ A-1.3: Slot 直接 quarantine — generation 一致を要求しない
    void quarantineSlot(uint32_t slot) noexcept;

    // ★ A-1.5: slot が crossfade に関与しているか確認
    bool isSlotInCrossfade(uint32_t slot) const noexcept;

    // ★ A-1.4: shutdown専用解放（2段階: DestroyPending → Reclaimed）
    void destroyQuarantineSlot(uint32_t slot, uint64_t expectedGeneration) noexcept;

    // ★ P0-4B DELETE-7: shutdown 時のみ Coordinator をバイパスした強制 reclaim
    //   通常パスは Coordinator::requestReclaim() 経由で実行される。
    void shutdownReclaim(DSPHandle handle) { reclaim(handle); }

    // NonRT: 現在の active runtime DSP handle を取得
    DSPHandle getActiveRuntimeDSPHandle() const noexcept;

    // NonRT: 現在の fading runtime DSP handle を取得（crossfade 中のみ有効）
    DSPHandle getFadingRuntimeDSPHandle() const noexcept;

    // スロット状態ダンプ（デバッグ・CI用）
    void emitOwnershipTrace(const std::filesystem::path& outputPath) const;

private:
    friend class RuntimePublicationCoordinator;

    // NonRT: grace period 完了後のメモリ解放（Coordinator 専用）
    // DELETE-1: reclaim() は Coordinator のみ呼び出し可能。
    //   外部からは Coordinator::requestReclaim() 経由で実行する。
    void reclaim(DSPHandle handle);
    std::array<DSPRegistrySlot, MAX_DSP_SLOTS> registry_{};
    // ★ ADR-005 / (A6): compile-time invariants for the ISR DSPHandle runtime.
    //   DSPHandle is a 16-byte POD; std::atomic<DSPHandle> must be lock-free,
    //   which on x64 requires 16-byte alignment (CMPXCHG16B, Haswell+ / AVX2).
    //   x64 ABI is assumed throughout ISR (no 32-bit build target).
    static_assert(std::is_trivially_copyable_v<DSPHandle>,
        "DSPHandle must be trivially copyable for ISR Runtime");
    static_assert(std::is_standard_layout_v<DSPHandle>,
        "DSPHandle must be standard layout for ISR Runtime");
    static_assert(alignof(DSPHandle) >= 16,
        "DSPHandle must be alignas(16) so atomic<DSPHandle> uses CMPXCHG16B on x64");
#if !defined(_MSC_VER)
    // Clang/GCC x64: alignas(16) guarantees lock-free atomics — enforce at compile time.
    static_assert(std::atomic<DSPHandle>::is_always_lock_free,
        "atomic<DSPHandle> must be lock-free on x64 for ISR Runtime");
#else
    // MSVC x64 (Debug/Release): the STL does not guarantee is_always_lock_free at
    //   compile time, so it is verified at runtime in DSPHandleRuntime's ctor
    //   (ISRDSPHandle.cpp:12-27) on BOTH configurations.
#endif
    std::atomic<DSPHandle> activeRuntimeDSPHandle_{ DSPHandle::null() };
    std::atomic<DSPHandle> fadingRuntimeDSPHandle_{ DSPHandle::null() };

    std::vector<CrossfadeRecord> crossfadeRecords_;

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
