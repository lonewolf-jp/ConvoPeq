#pragma once

#include <atomic>
#include <cstdint>
#include <array>
#include <vector>
#include <filesystem>
#include <mutex>  // ★ FUTURE-5 (work88): フリーリスト保護用
#include "AtomicAccess.h"  // ★ work88 (Phase 3): isRetired の atomic 状態参照用（consumeAtomic）

namespace convo {
namespace isr {

/**
 * ISR 10層 Architecture Layer 2: DSPHandle Runtime
 * DSP ownership の source-of-truth を管理し、lifetime ambiguity を排除する。
 */

/**
 * DSP スロット + 世代による handle（ABA 防止）
 *
 * alignas(16): 16バイト構造体の atomic に必要な原子的 CAS semantics を、将来の
 * lock-free 切替（atomic_ref 参照形 / wrapper 案）でも保証するための静的保証。
 * MSVC の atomic<16B by-value> は lock-pool で動作し is_lock_free()==false が正確
 * （D152-R2）。（ISRDSPHandle.cpp:12-20 / ISRDSPHandle.h:204-218 参照）
 */
#pragma warning(push)
#pragma warning(disable : 4324) // C4324: alignas(16) による意図的なパディングを許容
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
#pragma warning(pop)

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
 * ★ F1' (2026-10-25): DSPCore の寿命 lease。
 *
 *   `tryPin(handle)` が成功したときだけ生成され、生存している間は
 *   対象インスタンスが解放されないことを保証する。生ポインタは `get()` 経由でのみ
 *   公開され、`operator bool` が false の lease の `get()` は nullptr を返す。
 *
 *   規約（設計資料 F1-PIN-DESIGN-REVIEW の R1-R6）:
 *     R2: lease は「そのポインタを使い終わるスコープ全体」に置く（get() を持ち出さない）。
 *     R3: コピー不可（move のみ）。
 *     R5: tryPin / 解除は RT 可（ロック・確保・待機なし）。
 */
class DSPHandlePin
{
public:
    DSPHandlePin() noexcept = default;

    DSPHandlePin(const DSPHandlePin&) = delete;
    DSPHandlePin& operator=(const DSPHandlePin&) = delete;

    DSPHandlePin(DSPHandlePin&& other) noexcept
        : owner_(other.owner_), slot_(other.slot_), instance_(other.instance_)
    {
        other.owner_ = nullptr;
        other.instance_ = nullptr;
        other.slot_ = 0;
    }

    DSPHandlePin& operator=(DSPHandlePin&& other) noexcept
    {
        if (this != &other)
        {
            release();
            owner_ = other.owner_;
            slot_ = other.slot_;
            instance_ = other.instance_;
            other.owner_ = nullptr;
            other.instance_ = nullptr;
            other.slot_ = 0;
        }
        return *this;
    }

    ~DSPHandlePin() noexcept { release(); }

    [[nodiscard]] explicit operator bool() const noexcept { return instance_ != nullptr; }
    [[nodiscard]] void* get() const noexcept { return instance_; }
    [[nodiscard]] uint32_t slot() const noexcept { return slot_; }

private:
    friend class DSPHandleRuntime;

    DSPHandlePin(DSPHandleRuntime* owner, uint32_t slot, void* instance) noexcept
        : owner_(owner), slot_(slot), instance_(instance) {}

    void release() noexcept;

    DSPHandleRuntime* owner_ = nullptr;
    uint32_t slot_ = 0;
    void* instance_ = nullptr;
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
    // ★ F2' (2026-10-25): instance を atomic 化。create()/reclaim()/invalidateForDestroy() が
    //   書き込み、resolve() が別スレッドから読むため、plain pointer は正式なデータ競合だった
    //   （クラッシュダンプの stale slot は解放経路と並行に読まれていた）。
    //   void* と同サイズ・同アラインメントなのでスロットのレイアウトは不変（24 バイト）。
    std::atomic<void*>    instance;    // DSP インスタンスポインタ
    std::atomic<DSPState> state;       // 現在状態（atomic access）
    // ★ F1' (2026-10-25): pin 受付と pin 数を **単一ワード** に持つ。
    //   bit 63      = kPinClosed（破棄が受付を閉じた）
    //   bits 0..62  = pin 数
    //   2 ワード（受付フラグ + カウンタ）に分けると「受付が開いているか確認 → +1」の間に
    //   破棄が割り込む窓が残る（レビュー指摘 A: 受付閉鎖と寿命保護は別の不変条件）。
    //   1 ワードの CAS なら「開いていることを確認して +1」が不可分になり、
    //   「閉じた後に新しい pin が成立する」順序が存在しなくなる。
    std::atomic<uint64_t> pinState;
};

/**
 * DSP ハンドル runtime
 * 全 DSP reference の source-of-truth を管理
 */
class DSPHandleRuntime
{
public:
    // ★ F1' (2026-10-25): DSPHandlePin は release() で releasePinSlot を呼ぶ。
    friend class DSPHandlePin;

    static constexpr size_t MAX_DSP_SLOTS = 256;

    DSPHandleRuntime();
    ~DSPHandleRuntime();

    // NonRT: DSP インスタンスを登録し DSPHandle を返す
    // ★ F3' (2026-10-25): 同一インスタンスに複数の有効ハンドルを発行しない。
    //   - 既存スロットが live（Constructing/Active/CrossfadingIn/CrossfadingOut）なら
    //     そのハンドルを返す（重複排除）。
    //   - 既存スロットが Retired/DestroyPending/Quarantined なら **拒否**（null を返す）。
    //     破棄が確定したオブジェクトの再登録は「復活」であり、retire が片方のスロットしか
    //     無効化しないまま他方が Active で残る経路になる（ダンプの slot 251/254）。
    //   - Reclaimed は instance が null なので一致せず、アドレス再利用の新オブジェクトは
    //     通常どおり登録できる。
    DSPHandle create(void* dspInstance);

    // ★ F2' (2026-10-25): DSPCore のメモリ解放より前に、そのインスタンスを指す全スロットを
    //   解決不能にする（DestroyPending → Reclaimed + instance=nullptr + free list 返却）。
    //   AudioEngine::destroyDSPCoreNode（DSPCore 解放の唯一の漏斗）から呼ばれる。
    //   閉じる欠陥: 解放済みメモリを指したまま Active/Retired で残るスロット（ダンプの 251）。
    //   閉じない欠陥: resolve() が state と instance を読んだ直後〜解放の間の TOCTOU。
    //   これは pin 方式（F1'・別課題）でしか閉じない。
    //   戻り値: true = 解放してよい / false = pin が残っており **解放してはならない**（延期）。
    //   F1' で「pin の消滅を待つ」責務が入ったため、bool を返す契約へ変更した。
    [[nodiscard]] bool invalidateForDestroy(void* dspInstance,
                                            int timeoutMs = 2000) noexcept;

    // ★ F1' (2026-10-25): 寿命 lease の取得。成功した lease の生存中は
    //   invalidateForDestroy / reclaim がそのインスタンスを解放できない。
    //   RT 可（ロック・動的確保・待機なし。単一ワード CAS のみ）。
    //   失敗（null handle / generation 不一致 / 受付閉鎖 / instance なし）は空の lease。
    [[nodiscard]] DSPHandlePin tryPin(DSPHandle handle) noexcept;

    // ★ F1' (2026-10-25): 破棄側の受付閉鎖。以後の tryPin は必ず失敗する。
    //   戻り値: 閉鎖を確定させたか（既に閉鎖済みなら false）。
    //   RT 不可（破棄は NonRT）。閉鎖後に pinCount を観測して 0 を待つ。
    [[nodiscard]] bool closePinsForDestroy(uint32_t slot, uint64_t expectedGeneration) noexcept;

    // ★ F1' (2026-10-25): 指定スロットの現在の pin 数（破棄側の待機用）。
    [[nodiscard]] uint32_t pinnedCount(uint32_t slot) const noexcept;

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

    // ★ work88 (Phase 3): handle が Retired 状態か確認（保留 reclaim の再試行ガード用）。
    //   quarantineSlot が後から Quarantined に遷移した handle を requestReclaim が
    //   Retired に上書きしないようにする。Retired のままの場合のみ reclaim 対象とする。
    [[nodiscard]] bool isRetired(DSPHandle handle) const noexcept
    {
        if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS)
            return false;
        return convo::consumeAtomic(registry_[handle.slot].state, std::memory_order_acquire)
            == DSPState::Retired;
    }

    // ★ A-1.5: slot が crossfade に関与しているか確認
    bool isSlotInCrossfade(uint32_t slot) const noexcept;

    // ★ A-1.4: shutdown専用解放（2段階: DestroyPending → Reclaimed）
    void destroyQuarantineSlot(uint32_t slot, uint64_t expectedGeneration) noexcept;

    // ★ P0-4B DELETE-7: shutdown 時のみ Coordinator をバイパスした強制 reclaim は
    //   R4 Phase 7（2026-08-10）にて完全削除済み。通常パスは Coordinator の Reclaim
    //   Authority（RuntimeIntentCoordinator::reclaim(ReclaimMode, ...)）に一本化。
    //   旧 shutdownReclaim() は reclaim(handle) の単なるエイリアスであり、epoch 安全確認
    //   （RuntimeEBR）/ quiescence proof（ShutdownQuiescent）をバイパスしていたため、
    //   AC-R4-1（call sites == 0）・AC-R4-2（symbol absent）に従い除去した。

    // NonRT: 現在の active runtime DSP handle を取得
    DSPHandle getActiveRuntimeDSPHandle() const noexcept;

    // NonRT: 現在の fading runtime DSP handle を取得（crossfade 中のみ有効）
    DSPHandle getFadingRuntimeDSPHandle() const noexcept;

    // スロット状態ダンプ（デバッグ・CI用）
    void emitOwnershipTrace(const std::filesystem::path& outputPath) const;

private:
    friend class RuntimeIntentCoordinator;

    // NonRT: grace period 完了後のメモリ解放（Coordinator 専用）
    // DELETE-1: reclaim() は Coordinator のみ呼び出し可能。
    //   外部からは Coordinator::requestReclaim() 経由で実行する。
    void reclaim(DSPHandle handle);
    std::array<DSPRegistrySlot, MAX_DSP_SLOTS> registry_{};

    // ★ F1' (2026-10-25): pinState のビット定義。
    //   bit 63 = kPinClosed / bits 0..62 = pin 数。単一ワード CAS で
    //   「受付が開いていることを確認して +1」を不可分に行う。
    static constexpr uint64_t kPinClosedBit = 1ull << 63;
    static constexpr uint64_t kPinCountMask = ~kPinClosedBit;

    // ★ F1' (2026-10-25): DSPHandlePin::release() から呼ばれる（DSPHandlePin は friend）。
    void releasePinSlot(uint32_t slot) noexcept;

    // ★ F1' (2026-10-25): 破棄側の待機。targets は {slot, generation} の配列
    //   （型消去して void* で受ける。ISRDSPHandle.h の public 面を増やさないため）。
    struct PinDrainTarget { uint32_t slot; uint64_t generation; };
    [[nodiscard]] bool waitForPinsToDrain(const void* targets, size_t targetCount,
                                          int timeoutMs) noexcept;
    // ★ FUTURE-5 (work88): O(1) 確保用フリーリスト（slot 1..255 をスタック管理。slot 0 は
    //   null handle 表現のため使用しない — 既存 create() の線形スキャン開始位置と同一契約）。
    //   create() はフリーリスト pop（O(1)）、reclaim/rollbackRegistration は push（O(1)）。
    //   全操作は NonRT（registerDSPHandleForRuntime / Coordinator reclaim）のため mutex 保護で十分。
    mutable std::mutex freeListMutex_;
    std::array<uint32_t, MAX_DSP_SLOTS> freeSlots_{};
    uint32_t freeSize_ = 0;
    // ★ ADR-005 / (A6): compile-time invariants for the ISR DSPHandle runtime.
    //   DSPHandle is a 16-byte POD; its atomic requires atomic 16-byte CAS semantics
    //   (MSVC: lock-pool backend, is_lock_free()==false is accurate; all consumers NonRT).
    //   alignas(16) prepares a future lock-free switch. x64 ABI assumed (no 32-bit target).
    static_assert(std::is_trivially_copyable_v<DSPHandle>,
        "DSPHandle must be trivially copyable for ISR Runtime");
    static_assert(std::is_standard_layout_v<DSPHandle>,
        "DSPHandle must be standard layout for ISR Runtime");
    static_assert(alignof(DSPHandle) >= 16,
        "DSPHandle must be alignas(16) (lock-free switch readiness; MSVC atomic<16B> uses lock-pool)");
#if !defined(_MSC_VER)
    // Clang/GCC x64: alignas(16) guarantees lock-free atomics — enforce at compile time.
    static_assert(std::atomic<DSPHandle>::is_always_lock_free,
        "atomic<DSPHandle> must be lock-free on x64 for ISR Runtime");
#else
    // MSVC x64 (Debug/Release): atomic<16B by-value> uses the STL lock-pool backend
    //   (is_lock_free()==false is an ACCURATE report — D152-R2). The runtime value is
    //   recorded in DSPHandleRuntime's ctor (ISRDSPHandle.cpp:12-27); false is expected.
#endif
    std::atomic<DSPHandle> activeRuntimeDSPHandle_{ DSPHandle::null() };
    std::atomic<DSPHandle> fadingRuntimeDSPHandle_{ DSPHandle::null() };

    // ★ 監査指摘 (work88): beginCrossfade（CoordinatorLoop の push_back）と endCrossfade /
    //   isSlotInCrossfade（Timer 等の走査）が別スレッドから crossfadeRecords_ を操作するため
    //   mutex で保護する（全アクセス NonRT のため RT 影響なし）。
    mutable std::mutex crossfadeRecordsMutex_;
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
    // ★ 監査指摘 (work88): registerCrossfade（CoordinatorLoop）と unregisterCrossfade /
    //   getActiveCrossfades（Timer）が別スレッドから records_ を操作するため、mutex で保護する。
    //   全アクセスは NonRT（CoordinatorLoop / Timer）のため RT 影響なし。
    mutable std::mutex recordsMutex_;
    std::vector<CrossfadeRecord> records_;
    std::atomic<CrossfadeId> nextId_{1};
};

}  // namespace isr
}  // namespace convo
