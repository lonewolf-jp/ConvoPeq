#include "ISRDSPHandle.h"
#include "AtomicAccess.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>

// ★ R-51 一時診断（CONVO_R51_DIAG=1 のときだけ stderr へ出す。既定は無効）。
static bool r51DiagEnabled() noexcept
{
    static const bool enabled = []() {
        const char* v = std::getenv("CONVO_R51_DIAG");
        return v != nullptr && v[0] == '1';
    }();
    return enabled;
}

namespace convo {
namespace isr {

DSPHandleRuntime::DSPHandleRuntime()
{
    // Runtime初期化時に atomic<DSPHandle> の runtime is_lock_free() 値を一度だけ記録検証
    static const bool isLockFree = []{
        std::atomic<DSPHandle> test{ DSPHandle::null() };
        const bool ok = test.is_lock_free();
        // MSVC では 16バイト by-value atomic の is_lock_free() は lock-pool（spinlock）
        // 実装を正確に反映して false を返す（保身ではない — D152-R2 確定事実）。
        // CAS 意味論（原子的相互排他 + フルバリア）は lock-pool でも保持されるため、
        // MSVC では runtime 値の記録のみ（異常扱いしない）。Clang/GCC x64 では
        // alignas(16) により is_lock_free()==true。see ISRDSPHandle.h:204-218 (ADR-005)。
#if defined(_MSC_VER)
        (void)ok;
#else
        assert(ok && "atomic<DSPHandle> must be lock-free on x64 for ISR Runtime");
#endif
        return ok;
    }();
    (void)isLockFree; // unused in Release

    for (size_t i = 0; i < MAX_DSP_SLOTS; ++i) {
        convo::publishAtomic(registry_[i].generation, 0u, std::memory_order_relaxed);
        registry_[i].instance.store(nullptr, std::memory_order_relaxed);
        convo::publishAtomic(registry_[i].state, DSPState::Reclaimed, std::memory_order_relaxed);
        convo::publishAtomic(registry_[i].pinState, uint64_t{0}, std::memory_order_relaxed);
    }

    // ★ FUTURE-5 (work88): フリーリスト初期化（slot 1..255。slot 0 は null handle 表現のため除外）
    freeSize_ = 0;
    for (uint32_t slot = 1; slot < MAX_DSP_SLOTS; ++slot)
        freeSlots_[freeSize_++] = slot;
}

DSPHandleRuntime::~DSPHandleRuntime() = default;

DSPHandle DSPHandleRuntime::create(void* dspInstance)
{
    if (dspInstance == nullptr)
        return DSPHandle::null();

    // ★ FUTURE-5 (work88): 線形スキャン → フリーリスト pop（O(1) 確保）
    std::lock_guard<std::mutex> lock(freeListMutex_);

    // ★ F3' (2026-10-25): 1 インスタンス = 1 ハンドル。
    //   重複を許すと retire() が片方しか無効化せず、残ったスロットが解放済みメモリを
    //   Active のまま指す（2026-10-25 のダンプ: slot 251 Active / slot 254 Retired が同一
    //   インスタンス）。スロット数は 256 固定なので走査は NonRT で許容できる。
    for (size_t i = 0; i < MAX_DSP_SLOTS; ++i) {
        const auto& reg = registry_[i];
        if (convo::consumeAtomic(reg.instance, std::memory_order_acquire) != dspInstance)
            continue;
        const auto state = convo::consumeAtomic(reg.state, std::memory_order_acquire);
        if (state == DSPState::Constructing || state == DSPState::Active
            || state == DSPState::CrossfadingIn || state == DSPState::CrossfadingOut) {
            return DSPHandle{ static_cast<uint32_t>(i),
                              convo::consumeAtomic(reg.generation, std::memory_order_acquire) };
        }
        // 破棄が確定した / アクセス禁止のインスタンスは復活させない。
        if (state == DSPState::Retired || state == DSPState::DestroyPending
            || state == DSPState::Quarantined) {
            return DSPHandle::null();
        }
    }

    if (freeSize_ == 0) {
        assert(false && "DSP registry exhausted");
        return DSPHandle::null();
    }
    const uint32_t slot = freeSlots_[--freeSize_];
    auto& reg = registry_[slot];
    const auto gen = convo::consumeAtomic(reg.generation, std::memory_order_acquire) + 1u;
    convo::publishAtomic(reg.instance, dspInstance, std::memory_order_release);
    convo::publishAtomic(reg.generation, gen, std::memory_order_release);
    convo::publishAtomic(reg.state, DSPState::Constructing, std::memory_order_release);
    // ★ F1' (2026-10-25): 受付を開いた状態（closed=0, count=0）で公開する。
    //   Constructing の時点で破棄が来ても closePinsForDestroy が閉じられる。
    convo::publishAtomic(reg.pinState, uint64_t{0}, std::memory_order_release);
    return DSPHandle{ slot, gen };
}

DSPHandlePin DSPHandleRuntime::tryPin(DSPHandle handle) noexcept
{
    // ★ F1' (2026-10-25): RT 可。ロック・動的確保・待機を一切行わない。
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS)
        return {};

    auto& reg = registry_[static_cast<size_t>(handle.slot)];

    if (convo::consumeAtomic(reg.generation, std::memory_order_acquire) != handle.generation)
        return {};   // stale handle

    // 「受付が開いていることを確認して pin 数 +1」を 1 回の CAS で行う。
    //   2 ワードに分けると、確認と +1 の間に破棄が入る窓が残る（レビュー指摘 A）。
    uint64_t observed = convo::consumeAtomic(reg.pinState, std::memory_order_acquire);
    for (;;)
    {
        if ((observed & kPinClosedBit) != 0)
            return {};                                  // 破棄が受付を閉じた → 失敗（F1-1/F1-4）
        const uint64_t next = observed + 1;             // 下位 63bit のカウンタ
        if (convo::compareExchangeAtomic(reg.pinState, observed, next,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
            break;
        // observed は失敗時に最新値へ更新される
    }

    void* instance = convo::consumeAtomic(reg.instance, std::memory_order_acquire);
    if (instance == nullptr)
    {
        // F2' の無効化後（instance が既に消えている）。pin を返して失敗側へ倒す。
        releasePinSlot(handle.slot);
        return {};
    }

    return DSPHandlePin(this, handle.slot, instance);
}

void DSPHandlePin::release() noexcept
{
    if (owner_ == nullptr)
        return;
    owner_->releasePinSlot(slot_);
    owner_ = nullptr;
    instance_ = nullptr;
    slot_ = 0;
}

bool DSPHandleRuntime::closePinsForDestroy(uint32_t slot, uint64_t expectedGeneration) noexcept
{
    if (slot >= MAX_DSP_SLOTS)
        return false;
    auto& reg = registry_[slot];
    if (convo::consumeAtomic(reg.generation, std::memory_order_acquire) != expectedGeneration)
        return false;

    // 受付を閉じる（カウンタは保持したまま bit63 を立てる）。
    //   既に閉じていれば false（二重閉鎖は無害だが、呼び出し側に事実を返す）。
    uint64_t observed = convo::consumeAtomic(reg.pinState, std::memory_order_acquire);
    for (;;)
    {
        if ((observed & kPinClosedBit) != 0)
            return false;
        const uint64_t next = observed | kPinClosedBit;
        if (convo::compareExchangeAtomic(reg.pinState, observed, next,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
            return true;
    }
}

uint32_t DSPHandleRuntime::pinnedCount(uint32_t slot) const noexcept
{
    if (slot >= MAX_DSP_SLOTS)
        return 0;
    const uint64_t v = convo::consumeAtomic(registry_[slot].pinState, std::memory_order_acquire);
    return static_cast<uint32_t>(v & kPinCountMask);
}

void DSPHandleRuntime::releasePinSlot(uint32_t slot) noexcept
{
    if (slot >= MAX_DSP_SLOTS)
        return;
    auto& reg = registry_[slot];
    // 受付ビットは保持したままカウンタだけ -1（待機しない）。
    uint64_t observed = convo::consumeAtomic(reg.pinState, std::memory_order_acquire);
    for (;;)
    {
        const uint64_t count = observed & kPinCountMask;
        if (count == 0)
            return;                                     // 不整合（二重解除）は無視して安全側
        const uint64_t next = (observed & kPinClosedBit) | (count - 1);
        if (convo::compareExchangeAtomic(reg.pinState, observed, next,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
            return;
    }
}

ResolvedDSP DSPHandleRuntime::resolve(DSPHandle handle) const noexcept
{
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS) {
        return { nullptr, false, false };
    }

    const auto& reg = registry_[handle.slot];
    const auto currentGen = convo::consumeAtomic(reg.generation, std::memory_order_acquire);
    if (currentGen != handle.generation) {
        return { nullptr, false, true };
    }

    const auto state = convo::consumeAtomic(reg.state, std::memory_order_acquire);
    // V-11 candidate fix: a Retired (or DestroyPending) DSP must not resolve to a live pointer.
    //   Previously these states fell through and returned reg.instance, so callers such as
    //   AudioEngine::makeRuntimeGraphState could dereference an EQProcessor whose
    //   agc*CoeffTable storage had already been released (ScopedAlignedPtr<double>::get() AV).
    //   Treating them as non-resolvable makes the defect fail closed (nullptr); every
    //   caller already null-checks (makeRuntimeGraphState guards on `current != nullptr`,
    //   and the audio thread guards on the table pointers).
    if (state == DSPState::Reclaimed || state == DSPState::Quarantined
        || state == DSPState::Retired || state == DSPState::DestroyPending) {
        return { nullptr, false, true };   // isStale=true so callers can distinguish
    }

    // ★ F1'-prep (2026-10-25): state を読んでから instance を読む。両方とも acquire だが、
    //   「state を読んだ直後に destroy が走る」窓は依然として残る（HL-5）。
    //   ここで instance が null なら（F2' の無効化後）必ず失敗側へ倒す。
    auto* instance = convo::consumeAtomic(reg.instance, std::memory_order_acquire);
    if (instance == nullptr) {
        return { nullptr, false, true };
    }

    return { instance, true, false };
}

void DSPHandleRuntime::beginCrossfade(DSPHandle from, DSPHandle to, CrossfadeId id)
{
    assert(!from.isNull() && !to.isNull());
    convo::publishAtomic(registry_[from.slot].state, DSPState::CrossfadingOut, std::memory_order_release);
    convo::publishAtomic(registry_[to.slot].state, DSPState::CrossfadingIn, std::memory_order_release);

    // ★ 監査指摘 (work88): push_back と Timer 側の endCrossfade/isSlotInCrossfade 走査の競合を防ぐ
    std::lock_guard<std::mutex> lock(crossfadeRecordsMutex_);
    crossfadeRecords_.push_back(CrossfadeRecord{ id, from, to, 0u, true });
    convo::publishAtomic(fadingRuntimeDSPHandle_, from, std::memory_order_release);
}

void DSPHandleRuntime::activate(DSPHandle handle)
{
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS) {
        return;
    }

    convo::publishAtomic(registry_[handle.slot].state, DSPState::Active, std::memory_order_release);
    convo::publishAtomic(activeRuntimeDSPHandle_, handle, std::memory_order_release);
    // ★ STG-11-D18 (F-A): in-flight fade の fading handle を消さない。
    //   従来の無条件 null は、重ね合わせ publish 時に先行 fade の retire 経路
    //   （B-1/:1310 の observe）を消滅させ、fading DSPCore を orphan 化した。
    //   通常 path では既に null のため無変更と等価。fading handle の終端は
    //   endCrossfade（ガード付き）が担う。
}

void DSPHandleRuntime::endCrossfade(CrossfadeId id)
{
    // ★ 監査指摘 (work88): beginCrossfade（CoordinatorLoop）との並行アクセスを防ぐため lock。
    std::lock_guard<std::mutex> lock(crossfadeRecordsMutex_);
    for (auto& record : crossfadeRecords_) {
        if (record.id != id || !record.active) {
            continue;
        }

        record.active = false;
        // ★ STG-11-D18 (F-C): stale 書き込みのガード。
        //   fromHandle → Retired は無条件のまま（到達状態は CrossfadingOut か
        //   Retired のいずれかであり、両方とも Retired が正解のため）。
        convo::publishAtomic(registry_[record.fromHandle.slot].state, DSPState::Retired, std::memory_order_release);
        //   toHandle → Active は現在 CrossfadingIn の場合のみ。既に Retired
        //   （重ね合わせで即時 retire 済み）の DSP を Active に復活させない。
        if (convo::consumeAtomic(registry_[record.toHandle.slot].state, std::memory_order_acquire)
            == DSPState::CrossfadingIn) {
            convo::publishAtomic(registry_[record.toHandle.slot].state, DSPState::Active, std::memory_order_release);
        }
        //   activeHandle は from/to のいずれかに一致する場合のみ toHandle へ。
        //   より新しい publish が既に進めている場合は上書きしない（stale 防止）。
        {
            const auto currentActive = convo::consumeAtomic(activeRuntimeDSPHandle_, std::memory_order_acquire);
            if (currentActive == record.fromHandle || currentActive == record.toHandle) {
                convo::publishAtomic(activeRuntimeDSPHandle_, record.toHandle, std::memory_order_release);
            }
        }
        //   fading-handle null は現在 ==fromHandle か既に null の場合のみ。
        //   より新しい fade の handle を消さない。
        {
            const auto currentFading = convo::consumeAtomic(fadingRuntimeDSPHandle_, std::memory_order_acquire);
            if (currentFading == record.fromHandle || currentFading.isNull()) {
                convo::publishAtomic(fadingRuntimeDSPHandle_, DSPHandle::null(), std::memory_order_release);
            }
        }
        break;
    }
}

void DSPHandleRuntime::retire(DSPHandle handle)
{
    if (!handle.isNull() && handle.slot < MAX_DSP_SLOTS) {
        convo::publishAtomic(registry_[handle.slot].state, DSPState::Retired, std::memory_order_release);
    }
}

void DSPHandleRuntime::reclaim(DSPHandle handle)
{
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS)
        return;
    std::lock_guard<std::mutex> lock(freeListMutex_);
    auto& reg = registry_[handle.slot];
    // ★ 監査指摘 (work88): stale handle 検出（generation 不一致なら slot は別世代で再利用中 —
    //   誤って新規 DSP の slot を Reclaimed 化しない）。FUTURE-5 generation タグの完全化。
    if (convo::consumeAtomic(reg.generation, std::memory_order_acquire) != handle.generation)
        return;
    // ★ 監査指摘 (work88): 二重 reclaim 防止（既に Reclaimed → free list へ重複 push せず、
    //   同一 slot の二重割当（double-allocation）を構造的に排除）。
    if (convo::consumeAtomic(reg.state, std::memory_order_acquire) == DSPState::Reclaimed)
        return;

    // ★ F1'-c (2026-10-25) audit finding F: reclaim() must HONOUR outstanding pins.
    //   Before this fix, reclaim() cleared `instance` and recycled the slot even while a
    //   DSPHandlePin was held, so the destroy path could then find no matching slot and
    //   free the allocation immediately -> the pin holder was left dangling. The
    //   production retire path (DSPLifetimeManager::retire -> requestReclaimHandle ->
    //   reclaim) hit exactly that, which is why the production-path test observed
    //   "did not defer while pinned".
    //
    //   Fix: close the pin gate; if any lease is still alive, leave the slot intact
    //   (instance kept) and let invalidateForDestroy() do the pin-drain + clear later.
    (void)closePinsForDestroy(handle.slot, handle.generation);
    if (pinnedCount(handle.slot) != 0)
        return;   // 所有権はスロットに残す（destroy 経路が pin 消滅後に処理する）

    // ★ F2' (2026-10-25): 状態を先に無効化し、その後に instance を消す。
    //   逆順（instance=nullptr → state=Reclaimed）だと、resolve() が state を読んだ直後に
    //   古い state（Active/Retired）を見たまま instance を読む窓が広がる。state を先に
    //   DestroyPending にすることで、resolve() は少なくとも「解決不能」へ倒れる。
    convo::publishAtomic(reg.state, DSPState::DestroyPending, std::memory_order_release);
    convo::publishAtomic(reg.instance, nullptr, std::memory_order_release);
    convo::publishAtomic(reg.state, DSPState::Reclaimed, std::memory_order_release);
    // ★ FUTURE-5 (work88): Reclaimed 済み slot をフリーリストへ戻す（O(1) 再利用）
    if (handle.slot != 0 && freeSize_ < MAX_DSP_SLOTS)
        freeSlots_[freeSize_++] = handle.slot;
}

// ★ F2' (2026-10-25): DSPCore のメモリ解放より前に呼ばれる。
//   「解放済みメモリを指したまま Active/Retired で残るスロット」を構造的に除去する。
//   ダンプ実測: slot 251 が Active のまま instance=解放済みの 4.84MB ブロックを指し、
//   publisher が DSPCore::runtimeUuid (+0x49E408) を読んで 0xC0000005。
//
// ★ F1' (2026-10-25): pin の受付を閉じ、**既存 pin が消えるまで instance を消さない**。
//   `waitForPinsToDrain` は NonRT（破棄は NonRT）で待つ。タイムアウトしても
//   解放を強行しない — その場合は false を返し、呼び出し側が延期する
//   （設計資料 §2.2。ここで強行すると HL-5 そのものになる）。
bool DSPHandleRuntime::invalidateForDestroy(void* dspInstance, int timeoutMs) noexcept
{
    if (dspInstance == nullptr)
        return true;   // 何も無い = 解放してよい

    // 1) 受付を閉じる（この時点で新しい pin は必ず失敗する）。
    //    スロットごとに generation を控えてから閉じる。
    using Target = PinDrainTarget;
    std::array<Target, MAX_DSP_SLOTS> targets{};
    size_t targetCount = 0;
    {
        std::lock_guard<std::mutex> lock(freeListMutex_);
        for (size_t i = 0; i < MAX_DSP_SLOTS; ++i)
        {
            auto& reg = registry_[i];
            if (convo::consumeAtomic(reg.instance, std::memory_order_acquire) != dspInstance)
                continue;
            if (convo::consumeAtomic(reg.state, std::memory_order_acquire) == DSPState::Reclaimed)
                continue;
            targets[targetCount++] = Target{ static_cast<uint32_t>(i),
                                             convo::consumeAtomic(reg.generation,
                                                                  std::memory_order_acquire) };
        }
    }
    for (size_t k = 0; k < targetCount; ++k)
        (void)closePinsForDestroy(targets[k].slot, targets[k].generation);

    // 2) 既存 pin の消滅を待つ（受付は閉じているので増えない）。
    //    タイムアウトしたら **解放しない**。
    if (!waitForPinsToDrain(targets.data(), targetCount, timeoutMs))
        return false;

    // 3) instance を消す（ここに来た時点で pin 保持者はいない）。
    {
        std::lock_guard<std::mutex> lock(freeListMutex_);
        for (size_t k = 0; k < targetCount; ++k)
        {
            auto& reg = registry_[targets[k].slot];
            if (convo::consumeAtomic(reg.instance, std::memory_order_acquire) != dspInstance)
                continue;
            if (convo::consumeAtomic(reg.state, std::memory_order_acquire) == DSPState::Reclaimed)
                continue;

            convo::publishAtomic(reg.state, DSPState::DestroyPending, std::memory_order_release);
            convo::publishAtomic(reg.instance, nullptr, std::memory_order_release);
            convo::publishAtomic(reg.state, DSPState::Reclaimed, std::memory_order_release);

            if (targets[k].slot != 0 && freeSize_ < MAX_DSP_SLOTS)
                freeSlots_[freeSize_++] = targets[k].slot;
        }
    }
    return true;
}

// ★ R-51 (2026-10-30): slot を直接指定する破棄無効化。
//   instance を経由しないため、台帳（AudioEngine::runtimeDSPHandleMap_）や
//   registry_[i].instance の可視性に依存しない。pin が残っていれば false（= defer）。
bool DSPHandleRuntime::invalidateForDestroyBySlot(uint32_t slot, uint64_t generation,
                                                 int timeoutMs) noexcept
{
    if (slot >= MAX_DSP_SLOTS)
        return true;   // 未登録相当（解放してよい）

    // 1) 受付を閉じる（generation 不一致なら no-op）。
    (void)closePinsForDestroy(slot, generation);

    // 2) 既存 pin の消滅を待つ。タイムアウトは「破棄しない」。
    PinDrainTarget target{ slot, generation };
    const unsigned pinnedAtEnter = (unsigned)pinnedCount(slot);
    const bool drained = waitForPinsToDrain(&target, 1, timeoutMs);
    if (r51DiagEnabled())
        std::fprintf(stderr,
                     "[R51S] invalidateBySlot slot=%u gen=%llu pinnedAtEnter=%u drained=%d "
                     "pinnedAfter=%u regGen=%llu regState=%u\n",
                     (unsigned)slot, (unsigned long long)generation, pinnedAtEnter,
                     drained ? 1 : 0, (unsigned)pinnedCount(slot),
                     (unsigned long long)convo::consumeAtomic(registry_[slot].generation,
                                                              std::memory_order_acquire),
                     (unsigned)convo::consumeAtomic(registry_[slot].state, std::memory_order_acquire));
    if (!drained)
        return false;

    // 3) 破棄を確定（別世代に再利用されていたら触らない）。
    std::lock_guard<std::mutex> lock(freeListMutex_);
    auto& reg = registry_[slot];
    if (convo::consumeAtomic(reg.generation, std::memory_order_acquire) != generation)
        return true;
    if (convo::consumeAtomic(reg.state, std::memory_order_acquire) == DSPState::Reclaimed)
        return true;

    convo::publishAtomic(reg.state, DSPState::DestroyPending, std::memory_order_release);
    convo::publishAtomic(reg.instance, nullptr, std::memory_order_release);
    convo::publishAtomic(reg.state, DSPState::Reclaimed, std::memory_order_release);

    if (slot != 0 && freeSize_ < MAX_DSP_SLOTS)
        freeSlots_[freeSize_++] = slot;
    return true;
}

bool DSPHandleRuntime::waitForPinsToDrain(const void* targets, size_t targetCount,
                                          int timeoutMs) noexcept
{
    if (targetCount == 0)
        return true;

    const auto* t = static_cast<const PinDrainTarget*>(targets);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    for (;;)
    {
        bool allZero = true;
        for (size_t k = 0; k < targetCount; ++k)
        {
            if (pinnedCount(t[k].slot) != 0)
            {
                allZero = false;
                break;
            }
        }
        if (allZero)
            return true;
        if (std::chrono::steady_clock::now() >= deadline)
            return false;   // ★ 解放を強行しない（呼び出し側が延期する）
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void DSPHandleRuntime::quarantine(DSPHandle handle)
{
    if (!handle.isNull() && handle.slot < MAX_DSP_SLOTS) {
        convo::publishAtomic(registry_[handle.slot].state, DSPState::Quarantined, std::memory_order_release);
    }
}

bool DSPHandleRuntime::rollbackRegistration(DSPHandle handle) noexcept
{
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS) return false;
    auto& reg = registry_[handle.slot];
    DSPState expected = DSPState::Constructing;
    // state のみ CAS（instance は不変）。create() が上書きするため不要。
    const bool ok = convo::compareExchangeAtomic(reg.state, expected, DSPState::Reclaimed,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire);
    // ★ FUTURE-5 (work88): ロールバック成功（Reclaimed 化）時は slot をフリーリストへ戻す
    if (ok) {
        std::lock_guard<std::mutex> lock(freeListMutex_);
        if (handle.slot != 0 && freeSize_ < MAX_DSP_SLOTS)
            freeSlots_[freeSize_++] = handle.slot;
    }
    return ok;
}

// ★ A-1.3: Slot 直接 quarantine — generation 一致を要求しない
void DSPHandleRuntime::quarantineSlot(uint32_t slot) noexcept
{
    if (slot >= MAX_DSP_SLOTS)
        return;
    convo::publishAtomic(registry_[slot].state, DSPState::Quarantined,
                         std::memory_order_release);
}

// ★ A-1.5: slot が crossfade に関与しているか確認
bool DSPHandleRuntime::isSlotInCrossfade(uint32_t slot) const noexcept
{
    // ★ 監査指摘 (work88): beginCrossfade との並行アクセスを防ぐため lock。
    std::lock_guard<std::mutex> lock(crossfadeRecordsMutex_);
    for (const auto& record : crossfadeRecords_) {
        if (record.active &&
            (record.fromHandle.slot == slot || record.toHandle.slot == slot))
            return true;
    }
    return false;
}

// ★ A-1.4: shutdown専用解放（2段階: DestroyPending → Reclaimed）
bool DSPHandleRuntime::destroyQuarantineSlot(
    uint32_t slot, uint64_t expectedGeneration) noexcept
{
    if (slot >= MAX_DSP_SLOTS)
        return false;

    // generation check
    if (expectedGeneration != 0) {
        const auto currentGen = convo::consumeAtomic(
            registry_[slot].generation, std::memory_order_acquire);
        if (currentGen != expectedGeneration)
            return false;
    }

    // state == Quarantined is required
    const auto prevState = convo::consumeAtomic(
        registry_[slot].state, std::memory_order_acquire);
    assert(prevState == DSPState::Quarantined);
    if (prevState != DSPState::Quarantined)
        return false;

    // Phase 1: state check - not active / fading / crossfading
    const bool activeHandleMatch =
        (convo::consumeAtomic(activeRuntimeDSPHandle_, std::memory_order_acquire).slot == slot);
    const bool fadingHandleMatch =
        (convo::consumeAtomic(fadingRuntimeDSPHandle_, std::memory_order_acquire).slot == slot);
    const bool inCrossfade = isSlotInCrossfade(slot);

    if (activeHandleMatch || fadingHandleMatch || inCrossfade)
        return false;

    // F-2 (2026-10-25): pin gate.
    //   A DSPHandlePin keeps `instance_` and releases it by SLOT INDEX only
    //   (DSPHandlePin::release -> releasePinSlot(slot), which has no generation check).
    //   Recycling this slot while a pin is alive would let the old pin decrement the pin
    //   counter of the NEXT generation allocated in the same slot, silently removing that
    //   DSP's protection and freeing it while pinned. So: close the gate, and while any pin
    //   is still alive change NOTHING (state stays Quarantined, instance and slot retained).
    //   The engine retries from its reclaim tick / shutdown drain once the pins drain.
    const auto pinnedGeneration = convo::consumeAtomic(
        registry_[slot].generation, std::memory_order_acquire);
    (void)closePinsForDestroy(slot, pinnedGeneration);
    if (pinnedCount(slot) != 0)
    {
        convo::fetchAddAtomic(deferredQuarantineDestroy_, std::uint32_t{1},
                              std::memory_order_acq_rel);
        return false;   // deferred: nothing was modified
    }

    // Phase 1: mark DestroyPending (CAS, safe transition)
    auto expected = convo::consumeAtomic(
        registry_[slot].state, std::memory_order_acquire);
    while (expected == DSPState::Quarantined) {
        if (convo::compareExchangeAtomic(registry_[slot].state,
                                         expected, DSPState::DestroyPending,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
            break;
    }
    if (expected != DSPState::Quarantined)
        return false;

    // Phase 2: clear instance
    convo::publishAtomic(registry_[slot].instance, nullptr, std::memory_order_release);
    convo::publishAtomic(registry_[slot].state, DSPState::Reclaimed,
                         std::memory_order_release);
    // work88: return the reclaimed quarantine slot to the free list (prevents slot leaks).
    //   Double push cannot happen: state==Quarantined was verified above and reclaim()
    //   carries the same state guard.
    std::lock_guard<std::mutex> lock(freeListMutex_);
    if (slot != 0 && freeSize_ < MAX_DSP_SLOTS)
        freeSlots_[freeSize_++] = slot;
    return true;
}

// F-2 (2026-10-25): retry quarantine-slot destruction after the pins drain.
//   Owner: the slot (kept in state Quarantined) / Retry subject: the engine's reclaim tick
//   (AudioEngine::tryReclaimResources) and the shutdown cleanup.
std::size_t DSPHandleRuntime::retryDeferredQuarantineDestroy() noexcept
{
    if (convo::consumeAtomic(deferredQuarantineDestroy_, std::memory_order_acquire) == 0)
        return 0;

    std::size_t destroyed = 0;
    std::uint32_t stillDeferred = 0;

    for (uint32_t slot = 1; slot < MAX_DSP_SLOTS; ++slot)
    {
        if (convo::consumeAtomic(registry_[slot].state, std::memory_order_acquire)
            != DSPState::Quarantined)
            continue;
        if (convo::consumeAtomic(registry_[slot].instance, std::memory_order_acquire) == nullptr)
            continue;

        if (destroyQuarantineSlot(slot, 0))
            ++destroyed;
        else if (pinnedCount(slot) != 0)
            ++stillDeferred;
    }

    convo::publishAtomic(deferredQuarantineDestroy_, stillDeferred, std::memory_order_release);
    return destroyed;
}

std::uint32_t DSPHandleRuntime::deferredQuarantineDestroyCount() const noexcept
{
    return convo::consumeAtomic(deferredQuarantineDestroy_, std::memory_order_acquire);
}

DSPHandle DSPHandleRuntime::getActiveRuntimeDSPHandle() const noexcept
{
    return convo::consumeAtomic(activeRuntimeDSPHandle_, std::memory_order_acquire);
}

DSPHandle DSPHandleRuntime::getFadingRuntimeDSPHandle() const noexcept
{
    return convo::consumeAtomic(fadingRuntimeDSPHandle_, std::memory_order_acquire);
}

void DSPHandleRuntime::emitOwnershipTrace(const std::filesystem::path& outputPath) const
{
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        return;
    }

    file << "{\n  \"slots\": [\n";
    for (size_t i = 0; i < registry_.size(); ++i) {
        const auto state = convo::consumeAtomic(registry_[i].state, std::memory_order_acquire);
        file << "    { \"slot\": " << i << ", \"state\": " << static_cast<int>(state) << " }";
        if (i + 1u < registry_.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "  ]\n}\n";
}

CrossfadeAuthorityRuntime::CrossfadeAuthorityRuntime() = default;
CrossfadeAuthorityRuntime::~CrossfadeAuthorityRuntime() = default;

CrossfadeId CrossfadeAuthorityRuntime::registerCrossfade(DSPHandle from, DSPHandle to)
{
    const auto id = convo::fetchAddAtomic(nextId_, 1u, std::memory_order_acq_rel);
    // ★ 監査指摘 (work88): push_back による再確保と Timer 側の走査の競合を防ぐため lock。
    std::lock_guard<std::mutex> lock(recordsMutex_);
    records_.push_back(CrossfadeRecord{ id, from, to, 0u, true });
    return id;
}

void CrossfadeAuthorityRuntime::unregisterCrossfade(CrossfadeId id)
{
    // ★ 監査指摘 (work88): register（CoordinatorLoop）との並行アクセスを防ぐため lock。
    std::lock_guard<std::mutex> lock(recordsMutex_);
    for (auto& record : records_) {
        if (record.id == id) {
            record.active = false;
            break;
        }
    }
}

std::vector<CrossfadeRecord> CrossfadeAuthorityRuntime::getActiveCrossfades() const noexcept
{
    // ★ 監査指摘 (work88): register との並行アクセスを防ぐため lock（コピーは lock 内）。
    std::lock_guard<std::mutex> lock(recordsMutex_);
    std::vector<CrossfadeRecord> result;
    for (const auto& record : records_) {
        if (record.active) {
            result.push_back(record);
        }
    }
    return result;
}

bool CrossfadeAuthorityRuntime::hasCrossfadeInvolving(DSPHandle handle) const noexcept
{
    // ★ 監査指摘 (work88): 同様に lock（現時点で呼出し元ゼロだが契約として保護）。
    std::lock_guard<std::mutex> lock(recordsMutex_);
    for (const auto& record : records_) {
        if (record.active && (record.fromHandle == handle || record.toHandle == handle)) {
            return true;
        }
    }
    return false;
}

} // namespace isr
} // namespace convo
